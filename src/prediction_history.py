"""
src/prediction_history.py

Historico persistente de previsoes por evento (paper trading honesto).

A regra central e o CONGELAMENTO: a previsao de cada luta e gravada no
momento em que o relatorio do card e gerado (pre-registro, antes do
evento) e NUNCA e recalculada depois. Re-treinar o modelo na semana
seguinte nao pode reescrever o que foi previsto — sem isso, o "acertou ou
errou" do historico nao teria valor nenhum.

Fluxo por evento (zero passo extra alem do que ja existia):
  1. publicar o relatorio do card -> as previsoes entram no historico com
     actual_winner vazio ("aguardando resultados");
  2. depois do evento, preencher data/odds_template.csv como sempre
     (fluxo do evaluate) -> na proxima geracao de relatorio o historico
     puxa os vencedores dali sozinho (sync_results_from_template).

Upsert por (event_name, fighter_a, fighter_b): regerar o relatorio do
MESMO card antes do evento atualiza odds/probabilidades (odds se movem na
semana); mas linha com actual_winner preenchido esta fechada e nao muda
mais.
"""
from __future__ import annotations

import html as html_mod
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.utils import decimal_odds_to_implied_prob, remove_vig_two_way

logger = logging.getLogger(__name__)

HISTORY_COLUMNS = [
    "event_name", "event_date", "fighter_a", "fighter_b",
    "odds_a_decimal", "odds_b_decimal", "model_name",
    "model_prob_a", "model_side", "actual_winner",
    # sinal SHARP congelado no pre-registro (ver fetch_sharp_probs):
    # sharp_prob    = prob. devigada da Pinnacle para o model_side;
    # sharp_best_odd= MELHOR odd entre as casas para esse lado;
    # ev_sharp      = sharp_prob x sharp_best_odd. > 1 significa que alguma
    #                 casa paga acima do preco justo da sharp -- o unico
    #                 sinal do projeto que nao depende do nosso modelo.
    #                 (Usar a odd registrada aqui daria erro sistematico:
    #                 ela embute vig e o produto ficaria < 1 quase sempre.)
    # Vazios nos eventos anteriores a 01/ago/2026 -- a API so serve eventos
    # futuros, entao nao existe backfill. A analise comeca do evento 5.
    "sharp_prob", "sharp_best_odd", "ev_sharp",
    # distribuicao de METODO congelada junto (ago/2026), pelo mesmo motivo do
    # vencedor: sem isso, regerar um card encerrado recalculava o metodo com a
    # base ja contendo o resultado da propria luta. As tres somam 1 e sao
    # SIMETRICAS (nao dependem de quem e "A"), entao nao invertem com a ordem
    # dos lados. Vazias nos eventos anteriores -- sem backfill possivel.
    "method_ko_tko", "method_submission", "method_decision",
    # LINHA DE FECHAMENTO (ago/2026), capturada horas antes do card:
    # close_prob     = prob. devigada da Pinnacle para o model_side no fecho;
    # close_best_odd = melhor odd entre as casas no fecho (referencia);
    # clv            = close_prob - sharp_prob, em PONTOS de probabilidade.
    #                  Positivo = o mercado sharp andou NA DIRECAO do lado que
    #                  o modelo apontou, ou seja, batemos a linha de fecho.
    # Por que isso e melhor que P&L para medir edge: P&L e binario e precisa de
    # amostra enorme (uma perna a 2.95 decide a serie inteira); CLV mede
    # continuo, entao converge com muito menos evento.
    # Comparacao proposital entre DUAS medidas da mesma casa (Pinnacle
    # devigada) -- comparar a mediana do registro com a melhor odd do fecho
    # misturaria escalas e inflaria o resultado.
    "close_prob", "close_best_odd", "clv",
]

# nomes das classes do modelo de metodo -> coluna do historico
_METHOD_COLUMNS = {"KO_TKO": "method_ko_tko", "SUBMISSION": "method_submission",
                   "DECISION": "method_decision"}

# colunas adicionadas depois da criacao do arquivo: um historico antigo
# simplesmente nao as tem, e isso nao e corrupcao
_LATE_COLUMNS = ("sharp_prob", "sharp_best_odd", "ev_sharp",
                 *_METHOD_COLUMNS.values(),
                 "close_prob", "close_best_odd", "clv")

# colunas que sao texto (o resto e numerico); usadas na normalizacao de tipos
_TEXT_COLUMNS = ("event_name", "fighter_a", "fighter_b", "model_name",
                 "model_side", "actual_winner")


def _load_raw(history_csv: Path | None = None) -> pd.DataFrame:
    path = history_csv or config.PREDICTION_HISTORY_CSV
    if not Path(path).exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.read_csv(path)
    # sinal sharp (01/ago/2026) e distribuicao de metodo (09/ago/2026) foram
    # adicionados depois: um historico antigo simplesmente nao os tem — cria
    # vazios em vez de tratar como corrompido.
    for col in _LATE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    missing = [c for c in HISTORY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} sem as colunas {missing} — arquivo de historico corrompido?")
    # colunas de texto podem vir como float64 quando estao 100% vazias
    # (ex.: actual_winner antes do primeiro resultado) — normaliza para
    # object para aceitar strings no sync sem LossySetitemError (pandas 3)
    for col in _TEXT_COLUMNS:
        df[col] = df[col].astype("object")
    return df[HISTORY_COLUMNS]


def _same_fight(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    """Mascara: mesma luta em qualquer ordem de lados."""
    return ((df["fighter_a"] == a) & (df["fighter_b"] == b)) | \
           ((df["fighter_a"] == b) & (df["fighter_b"] == a))


def record_card_predictions(analysis: dict, card_name: str, event_date: str,
                            history_csv: Path | None = None,
                            sharp_probs: dict | None = None) -> int:
    """
    Grava/atualiza no historico as previsoes de um card (saida de
    card_report.analyze_card). Lutas sem previsao entram com os campos de
    modelo vazios — nunca somem em silencio. Linhas ja fechadas
    (actual_winner preenchido) sao intocaveis. Retorna quantas linhas
    foram gravadas/atualizadas.

    `sharp_probs`: {(fighter_a, fighter_b): {"sharp_prob", "best_odd"}}
    — ver line_shopping.fetch_sharp_probs. Congelado junto com a previsao
    para permitir, mais adiante, testar se as pernas com respaldo sharp se
    saem melhor que as sem.
    """
    path = Path(history_csv or config.PREDICTION_HISTORY_CSV)
    df = _load_raw(path)
    sharp_probs = sharp_probs or {}

    new_rows = []
    for fight in analysis["favorites"] + analysis["underdogs"]:
        sharp = sharp_probs.get((fight["fighter_a"], fight["fighter_b"])) or {}
        prob, best_odd = sharp.get("sharp_prob"), sharp.get("best_odd")
        has_sharp = prob is not None and best_odd is not None
        # metodo falha de forma independente do vencedor: sem ele as tres
        # colunas ficam vazias e a luta segue registrada normalmente
        method = fight.get("method_probs") or {}
        new_rows.append({
            "event_name": card_name, "event_date": event_date,
            "fighter_a": fight["fighter_a"], "fighter_b": fight["fighter_b"],
            "odds_a_decimal": fight["odds_a"], "odds_b_decimal": fight["odds_b"],
            "model_name": analysis["model_name"],
            "model_prob_a": round(float(fight["model_prob_a"]), 4),
            "model_side": fight["model_side"],
            "actual_winner": np.nan,
            "sharp_prob": round(float(prob), 4) if has_sharp else np.nan,
            "sharp_best_odd": round(float(best_odd), 3) if has_sharp else np.nan,
            "ev_sharp": round(float(prob) * float(best_odd), 4) if has_sharp else np.nan,
            **{col: (round(float(method[cls]), 4) if cls in method else np.nan)
               for cls, col in _METHOD_COLUMNS.items()},
            # fechamento entra depois, pelo capture_closing (ver abaixo)
            "close_prob": np.nan, "close_best_odd": np.nan, "clv": np.nan,
        })
    for fight in analysis["no_prediction"]:
        new_rows.append({
            "event_name": card_name, "event_date": event_date,
            "fighter_a": fight["fighter_a"], "fighter_b": fight["fighter_b"],
            "odds_a_decimal": fight["odds_a"], "odds_b_decimal": fight["odds_b"],
            "model_name": analysis["model_name"],
            "model_prob_a": np.nan, "model_side": np.nan, "actual_winner": np.nan,
            "sharp_prob": np.nan, "sharp_best_odd": np.nan, "ev_sharp": np.nan,
            **{col: np.nan for col in _METHOD_COLUMNS.values()},
            "close_prob": np.nan, "close_best_odd": np.nan, "clv": np.nan,
        })

    n_written = 0
    for row in new_rows:
        mask = (df["event_name"] == row["event_name"]) & _same_fight(df, row["fighter_a"], row["fighter_b"])
        existing = df[mask]
        if not existing.empty:
            if existing["actual_winner"].notna().any():
                continue  # linha fechada: previsao congelada, nao reescreve
            prev = existing.iloc[0]
            # linha de fechamento NUNCA e reescrita aqui: quem a grava e o
            # capture_closing, e ela e por definicao o ultimo estado antes do
            # card. Regerar o relatorio depois da captura nao pode apaga-la.
            for col in ("close_prob", "close_best_odd", "clv"):
                row[col] = prev[col]
            # sinal sharp ja capturado nao se perde ao regerar o relatorio
            # sem consultar a API (as tres colunas andam juntas)
            if pd.isna(row["sharp_prob"]):
                if pd.notna(prev["sharp_prob"]):
                    row["sharp_prob"] = prev["sharp_prob"]
                    row["sharp_best_odd"] = prev["sharp_best_odd"]
                    row["ev_sharp"] = prev["ev_sharp"]
            df = df[~mask]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        n_written += 1

    df.to_csv(path, index=False)
    if n_written:
        logger.info("Historico: %d previsao(oes) gravadas para %s (congeladas apos o evento).",
                    n_written, card_name)
    return n_written


def sync_results_from_template(history_csv: Path | None = None,
                               template_csv: Path | None = None) -> int:
    """
    Preenche actual_winner das linhas abertas do historico a partir do
    data/odds_template.csv (que o fluxo do evaluate ja preenche apos cada
    evento). Casa por dupla de lutadores em qualquer ordem + mesma
    event_date. Retorna quantas linhas foram fechadas.
    """
    path = Path(history_csv or config.PREDICTION_HISTORY_CSV)
    df = _load_raw(path)
    if df.empty:
        return 0
    template_path = Path(template_csv or config.ODDS_TEMPLATE_CSV)
    if not template_path.exists():
        return 0
    template = pd.read_csv(template_path).dropna(subset=["actual_winner"])
    if template.empty:
        return 0

    n_closed = 0
    for idx, row in df[df["actual_winner"].isna()].iterrows():
        match = template[_same_fight(template, row["fighter_a"], row["fighter_b"])
                         & (template["event_date"] == row["event_date"])]
        if match.empty:
            continue
        winner = str(match.iloc[0]["actual_winner"]).strip()
        if winner not in (str(row["fighter_a"]), str(row["fighter_b"])):
            logger.warning("Historico: vencedor '%s' do template nao bate com %s vs %s — ignorando.",
                           winner, row["fighter_a"], row["fighter_b"])
            continue
        df.loc[idx, "actual_winner"] = winner
        n_closed += 1

    if n_closed:
        df.to_csv(path, index=False)
        logger.info("Historico: %d resultado(s) sincronizado(s) do odds_template.csv.", n_closed)
    return n_closed


def open_fights_for_event(event_date: str, history_csv: Path | None = None) -> list[dict]:
    """
    Lutas do evento ainda SEM resultado e com lado apontado pelo modelo, no
    formato que line_shopping.fetch_sharp_probs consome
    ({fighter_a, fighter_b, model_side}). Usado pela captura de fechamento.
    """
    df = _load_raw(Path(history_csv or config.PREDICTION_HISTORY_CSV))
    if df.empty:
        return []
    aberto = df[(df["event_date"] == event_date)
                & df["actual_winner"].isna()
                & df["model_side"].notna()]
    return [{"fighter_a": str(r["fighter_a"]), "fighter_b": str(r["fighter_b"]),
             "model_side": str(r["model_side"])} for _, r in aberto.iterrows()]


def record_closing_lines(event_date: str, sharp_probs: dict,
                         history_csv: Path | None = None,
                         allow_update: bool = False) -> int:
    """
    Congela a linha de FECHAMENTO das lutas abertas do evento e calcula o CLV.
    Rodar poucas horas antes do card (ver scripts/capture_closing.py).

    `sharp_probs`: saida de line_shopping.fetch_sharp_probs, ja no fechamento.

        clv = close_prob - sharp_prob   (pontos de probabilidade)

    Positivo = a Pinnacle devigada andou NA DIRECAO do lado que o modelo
    apontou entre o pre-registro e o fecho, ou seja, batemos a linha de
    fechamento. Sem sharp_prob no registro nao ha CLV (fica NaN): so da para
    medir movimento havendo os dois pontos.

    `allow_update=False` (padrao, uso manual): uma vez gravado, o fechamento
    NAO e sobrescrito. Protege contra uma rodada tardia acidental — depois do
    card — sobrescrever um fechamento bom.

    `allow_update=True` (uso do captura automatica): reescreve enquanto a
    linha estiver aberta. E OBRIGATORIO no modo automatico: rodando de hora em
    hora, "a primeira vence" congelaria o preco MAIS ANTIGO capturado, que e
    exatamente o oposto de uma linha de fechamento. Quem chama so pode passar
    True tendo confirmado que o evento AINDA NAO COMECOU — em
    scripts/auto_capture.py essa garantia vem da propria API, que deixa de
    listar o evento assim que ele comeca.

    Retorna quantas linhas foram gravadas.
    """
    path = Path(history_csv or config.PREDICTION_HISTORY_CSV)
    df = _load_raw(path)
    if df.empty or not sharp_probs:
        return 0

    n = 0
    for (a, b), data in sharp_probs.items():
        if not data or data.get("sharp_prob") is None:
            continue
        mask = (df["event_date"] == event_date) & _same_fight(df, a, b)
        aberto = mask & df["actual_winner"].isna()
        idx = df[aberto if allow_update else aberto & df["close_prob"].isna()].index
        if len(idx) == 0:
            continue
        i = idx[0]
        close_p = float(data["sharp_prob"])
        df.loc[i, "close_prob"] = round(close_p, 4)
        if data.get("best_odd") is not None:
            df.loc[i, "close_best_odd"] = round(float(data["best_odd"]), 3)
        reg = df.loc[i, "sharp_prob"]
        if pd.notna(reg):
            df.loc[i, "clv"] = round(close_p - float(reg), 4)
        n += 1

    if n:
        df.to_csv(path, index=False)
        logger.info("Fechamento: %d linha(s) congelada(s) para %s.", n, event_date)
    return n


def compute_clv_summary(history_df: pd.DataFrame) -> dict | None:
    """
    Resumo do CLV da serie: media em pontos de probabilidade, quantas pernas
    bateram o fecho e o total medido.

    Por que isto vale mais que o P&L para julgar o modelo: o P&L e binario e
    dominado por variancia (uma perna a 2.95 vira a serie inteira), enquanto o
    CLV mede quanto o mercado andou, em cada perna, numa escala continua.
    Converge com muito menos amostra. None enquanto nada foi capturado.
    """
    if history_df.empty or "clv" not in history_df.columns:
        return None
    com = history_df[history_df["clv"].notna()]
    if com.empty:
        return None
    return {"n": int(len(com)),
            "media": float(com["clv"].mean()),
            "positivos": int((com["clv"] > 0).sum())}


# A captura automatica do fechamento (scripts/auto_capture.py + tarefa horaria
# do Windows) entrou no ar em 26/ago/2026, valendo a partir do card do dia 29.
# Antes disso a captura era manual e simplesmente nunca aconteceu: os 7
# primeiros eventos da serie estao sem close_prob e assim vao ficar, porque a
# API so serve eventos FUTUROS e nao existe backfill. Alarmar sobre eles seria
# ruido permanente que ninguem pode acionar, entao o alarme comeca daqui.
PRIMEIRO_EVENTO_COM_CAPTURA = "2026-08-29"

# Mesma janela do auto_capture (DIAS_DE_ANTECEDENCIA): fora dela o fechamento
# nao esta "faltando", so ainda nao e hora de captura-lo.
_JANELA_CAPTURA_DIAS = 1


def events_missing_closing(history_df: pd.DataFrame,
                           hoje: date | None = None) -> list[dict]:
    """
    Eventos com previsao registrada cujo fechamento NAO foi capturado.

    Existe porque essa falha e MUDA. Sem captura, `_clv_stat_html` apenas nao
    desenha o bloco de CLV, e o relatorio fica visualmente identico a um
    relatorio saudavel — foi assim que os 7 primeiros eventos da serie
    perderam a medicao sem ninguem notar.

    Dois estados, com urgencias opostas:

      "iminente"  o card e hoje ou amanha e o fechamento continua vazio. E o
                  UNICO momento acionavel: da para rodar
                  `python -m scripts.capture_closing --event-date ...` na mao.
                  Se a captura automatica estiver quebrada (chave de API
                  vencida, tarefa desativada), e aqui que isso aparece.

      "perdido"   o card ja passou sem captura. Irrecuperavel: aquelas pernas
                  nunca terao CLV. Fica listado para o buraco na serie ser
                  explicito, em vez de virar uma media silenciosamente
                  calculada sobre menos pernas do que se imagina.

    Evento marcado com fechamento em QUALQUER perna conta como capturado: a
    Pinnacle nao cobre card inteiro, e cobertura parcial e o normal.
    """
    if history_df.empty or "close_prob" not in history_df.columns:
        return []

    hoje = hoje or date.today()
    limite = (hoje + timedelta(days=_JANELA_CAPTURA_DIAS)).isoformat()
    hoje_iso = hoje.isoformat()

    com_previsao = history_df[history_df["model_side"].notna()]
    if com_previsao.empty:
        return []

    faltando = []
    chaves = (com_previsao[["event_name", "event_date"]].drop_duplicates()
              .sort_values("event_date"))
    for _, (event_name, event_date) in chaves.iterrows():
        data = str(event_date)
        if data < PRIMEIRO_EVENTO_COM_CAPTURA or data > limite:
            continue  # antes da automacao, ou ainda cedo demais para cobrar
        ev = com_previsao[(com_previsao["event_name"] == event_name)
                          & (com_previsao["event_date"] == event_date)]
        if ev["close_prob"].notna().any():
            continue  # capturado (cobertura parcial ja basta)
        faltando.append({
            "event_date": data,
            "event_name": str(event_name),
            "n": int(len(ev)),
            "estado": "iminente" if data >= hoje_iso else "perdido",
        })
    return faltando


def frozen_predictions_for_event(event_date: str, history_csv: Path | None = None) -> dict:
    """
    Previsoes publicadas das lutas do evento que JA TEM RESULTADO, para o
    relatorio exibir em vez de recalcular:

        {(fighter_a, fighter_b): {"model_prob_a": float | None,
                                  "method_probs": dict | None}}

    Ha uma entrada para toda luta fechada, mesmo que os dois valores sejam
    None (luta que entrou no historico sem previsao). E o que permite o
    relatorio distinguir "fechada e sem metodo congelado" de "aberta" e, na
    primeira, suprimir o metodo em vez de mostrar um recalculo contaminado.

    Motivo: card_report.analyze_card recalcula a previsao a cada geracao a
    partir de export_latest_fighter_levels(), que reflete a base ATUAL. Depois
    que o evento entra na base (fill-gap), esses niveis ja incluem o proprio
    resultado — regerar o relatorio produz previsoes que "sabem" quem ganhou.
    Medido no card de 08/ago/2026: as 10 lutas se moveram na direcao do
    vencedor real e as 2 que o modelo errou inverteram para o lado certo, o
    que faria a pagina mostrar 10/10 nas abas de Favoritos/Zebras enquanto a
    aba Historico (essa sim congelada) dizia 8/10.

    So lutas FECHADAS entram. Enquanto o resultado nao chega, regerar para
    atualizar as odds DEVE refrescar a previsao — e a mesma regra que
    record_card_predictions aplica ao reescrever linhas abertas.
    """
    df = _load_raw(Path(history_csv or config.PREDICTION_HISTORY_CSV))
    if df.empty:
        return {}
    closed = df[(df["event_date"] == event_date) & df["actual_winner"].notna()]

    frozen = {}
    for _, r in closed.iterrows():
        method = {cls: float(r[col]) for cls, col in _METHOD_COLUMNS.items()
                  if pd.notna(r[col])}
        frozen[(str(r["fighter_a"]), str(r["fighter_b"]))] = {
            "model_prob_a": float(r["model_prob_a"]) if pd.notna(r["model_prob_a"]) else None,
            # parcial nao serve: as tres classes tem de somar 1 para a aba
            # exibir odds justas coerentes
            "method_probs": method if len(method) == len(_METHOD_COLUMNS) else None,
        }
    return frozen


def load_history(history_csv: Path | None = None) -> pd.DataFrame:
    """
    Historico com colunas derivadas para exibicao:
      market_side (favorito pelo devig; NaN em pick'em de odds iguais),
      model_correct / market_correct (NaN enquanto nao ha resultado ou,
      no caso do modelo, quando nao houve previsao).
    """
    df = _load_raw(history_csv)
    if df.empty:
        return df

    market_sides, model_ok, market_ok = [], [], []
    for _, row in df.iterrows():
        prob_a, _ = remove_vig_two_way(
            decimal_odds_to_implied_prob(float(row["odds_a_decimal"])),
            decimal_odds_to_implied_prob(float(row["odds_b_decimal"])))
        if prob_a > 0.5:
            market_side = row["fighter_a"]
        elif prob_a < 0.5:
            market_side = row["fighter_b"]
        else:
            market_side = np.nan  # pick'em exato: mercado nao tem lado
        market_sides.append(market_side)

        winner = row["actual_winner"]
        has_result = pd.notna(winner)
        model_ok.append((row["model_side"] == winner) if has_result and pd.notna(row["model_side"]) else np.nan)
        market_ok.append((market_side == winner) if has_result and pd.notna(market_side) else np.nan)

    df = df.copy()
    df["market_side"] = market_sides
    df["model_correct"] = model_ok
    df["market_correct"] = market_ok
    return df


def model_side_leg(row: pd.Series) -> dict | None:
    """
    Perna do lado do modelo numa luta do historico: probabilidade e odd do
    model_side + EV (p x odd). None se a luta nao teve previsao. E a regra
    de pre-registro do paper trading: EV > 1 = perna simulada de 1 unidade.
    """
    if pd.isna(row["model_side"]) or pd.isna(row["model_prob_a"]):
        return None
    if row["model_side"] == row["fighter_a"]:
        prob, odd = float(row["model_prob_a"]), float(row["odds_a_decimal"])
    else:
        prob, odd = 1 - float(row["model_prob_a"]), float(row["odds_b_decimal"])
    return {"side": row["model_side"], "prob": prob, "odd": odd, "ev": prob * odd}


def compute_series_summary(history_df: pd.DataFrame) -> dict | None:
    """
    Placar ACUMULADO da serie de paper trading, so com eventos fechados
    (resultado preenchido): acertos de modelo e mercado, e P&L das pernas
    EV>1 (1 unidade cada, regra uniforme: toda p x odd > 1 conta).
    Tudo derivado das previsoes congeladas — nada e recalculado.
    """
    if history_df.empty:
        return None
    closed = history_df[history_df["actual_winner"].notna()]
    if closed.empty:
        return None

    legs_n, legs_won, pnl = 0, 0, 0.0
    for _, row in closed.iterrows():
        leg = model_side_leg(row)
        if leg is None or leg["ev"] <= 1:
            continue
        legs_n += 1
        if row["actual_winner"] == leg["side"]:
            legs_won += 1
            pnl += leg["odd"] - 1
        else:
            pnl -= 1

    return {
        "n_events": int(closed[["event_name", "event_date"]].drop_duplicates().shape[0]),
        "model_hits": int(closed["model_correct"].fillna(False).astype(bool).sum()),
        "model_n": int(closed["model_correct"].notna().sum()),
        "market_hits": int(closed["market_correct"].fillna(False).astype(bool).sum()),
        "market_n": int(closed["market_correct"].notna().sum()),
        "legs_n": legs_n, "legs_won": legs_won, "legs_pnl": pnl,
    }


def compute_sharp_split(history_df: pd.DataFrame) -> dict | None:
    """
    A pergunta que a serie existe para responder: as pernas EV>1 COM
    respaldo sharp (ev_sharp > 1, ou seja, a odd paga acima do preco justo
    da Pinnacle) se saem melhor que as SEM?

    Divide as pernas fechadas em dois grupos e devolve acertos e P&L de
    cada um. So considera lutas com sharp_prob gravado — eventos anteriores
    a 01/ago/2026 nao tem o dado e ficam de fora (nao ha backfill possivel).
    None se ainda nao ha nenhuma perna com o dado.
    """
    if history_df.empty:
        return None
    closed = history_df[history_df["actual_winner"].notna() & history_df["ev_sharp"].notna()]
    if closed.empty:
        return None

    groups = {"com_sharp": {"n": 0, "won": 0, "pnl": 0.0},
              "sem_sharp": {"n": 0, "won": 0, "pnl": 0.0}}
    for _, row in closed.iterrows():
        leg = model_side_leg(row)
        if leg is None or leg["ev"] <= 1:
            continue  # so pernas do criterio da serie (EV do modelo > 1)
        g = groups["com_sharp" if float(row["ev_sharp"]) > 1 else "sem_sharp"]
        g["n"] += 1
        if row["actual_winner"] == leg["side"]:
            g["won"] += 1
            g["pnl"] += leg["odd"] - 1
        else:
            g["pnl"] -= 1
    if groups["com_sharp"]["n"] == 0 and groups["sem_sharp"]["n"] == 0:
        return None
    return groups


# ---------------------------------------------------------------------------
# HTML da aba "Historico" (mesmo estilo self-contained do card_report)
# ---------------------------------------------------------------------------

def _e(text) -> str:
    return html_mod.escape(str(text))


def _initials(name: str) -> str:
    """Iniciais para o avatar de monograma (primeiro + ultimo nome)."""
    parts = [p for p in str(name).replace("'", " ").split() if p and p[0].isalnum()]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _name_hue(name: str) -> int:
    """Matiz deterministico por lutador (mesma cor em todo o relatorio)."""
    return sum(ord(c) * (i + 7) for i, c in enumerate(str(name))) % 360


# Mapa nome -> URL de foto, valido so durante uma geracao de relatorio.
# Vazio (default) = so monogramas: e o modo da pagina PUBLICADA no Pages
# (self-contained, offline, sem material com direitos autorais). O modo
# --photos do card_report preenche o mapa para o relatorio LOCAL de uso
# pessoal (imagens hotlinkadas do UFC.com, nada copiado).
_PHOTO_MAP: dict = {}


def set_photo_map(photo_map: dict | None) -> None:
    _PHOTO_MAP.clear()
    _PHOTO_MAP.update({k: v for k, v in (photo_map or {}).items() if v})


def avatar_html(name: str, small: bool = False, big: bool = False) -> str:
    """
    Retrato do lutador: monograma (iniciais, cor estavel por nome) e, se
    houver foto no mapa da geracao atual, a foto por cima — com fallback
    automatico para o monograma se a imagem nao carregar (onerror remove o
    <img> e as iniciais reaparecem).

    `big`: retrato do hero/destaque (recorte grande). `small`: inline, ao
    lado do nome em listas.
    """
    cls = "avatar" + (" sm" if small else "") + (" lg" if big else "")
    hue = _name_hue(name)
    photo = _PHOTO_MAP.get(str(name))
    # onload esconde as iniciais (fotos do UFC sao PNG transparente — sem
    # isso o texto vaza por tras do lutador); onerror remove o <img> e as
    # iniciais voltam a aparecer como fallback.
    img = (f'<img src="{_e(photo)}" alt="" loading="lazy" '
           f'onload="this.parentNode.classList.add(\'has-photo\')" '
           f'onerror="this.remove()">' if photo else "")
    # saturacao baixa de proposito: o monograma e fallback, tem de conviver
    # com a paleta neutra sem virar mancha colorida no meio da pagina
    return (f'<span class="{cls}" style="--mono:hsl({hue},14%,23%)">'
            f'<span class="avatar-txt">{_e(_initials(name))}</span>{img}</span>')


def _result_badge(correct) -> str:
    if pd.isna(correct):
        return '<span class="hist-badge none">—</span>'
    if correct:
        return '<span class="hist-badge hit">✓ acertou</span>'
    return '<span class="hist-badge miss">✗ errou</span>'


def _history_fight_row(row: pd.Series) -> str:
    has_result = pd.notna(row["actual_winner"])
    winner_html = (f'<strong>{_e(row["actual_winner"])}</strong>' if has_result
                   else '<span class="hist-pending">aguardando</span>')
    if pd.isna(row["model_side"]):
        model_html = '<span class="hist-pending">sem previsão</span>'
        model_badge = '<span class="hist-badge none">—</span>'
    else:
        prob_side = (row["model_prob_a"] if row["model_side"] == row["fighter_a"]
                     else 1 - row["model_prob_a"])
        model_html = f'{_e(row["model_side"])} <span class="hist-prob">{prob_side * 100:.0f}%</span>'
        model_badge = _result_badge(row["model_correct"]) if has_result else ""
    market_html = (_e(row["market_side"]) if pd.notna(row["market_side"])
                   else '<span class="hist-pending">pick\'em</span>')
    market_badge = (_result_badge(row["market_correct"]) if has_result else "")
    return f"""
      <tr>
        <td class="hist-fight">{avatar_html(row['fighter_a'], small=True)} {_e(row['fighter_a'])}
          <span class="vs">vs</span>
          {avatar_html(row['fighter_b'], small=True)} {_e(row['fighter_b'])}</td>
        <td>{model_html} {model_badge}</td>
        <td>{market_html} {market_badge}</td>
        <td>{winner_html}</td>
      </tr>"""


def _series_summary_html(history_df: pd.DataFrame) -> str:
    s = compute_series_summary(history_df)
    if s is None:
        return ""
    pct_model = s["model_hits"] / s["model_n"] * 100 if s["model_n"] else 0
    pct_market = s["market_hits"] / s["market_n"] * 100 if s["market_n"] else 0
    pnl_cls = "pos" if s["legs_pnl"] > 0 else ("neg" if s["legs_pnl"] < 0 else "")
    return f"""
    <div class="serie-box">
      <div class="serie-title">Série acumulada <span class="hist-date">{s['n_events']} evento(s)
        fechado(s) de ~25 até a amostra ter valor estatístico</span></div>
      <div class="serie-grid">
        <div class="serie-stat"><span class="serie-label">modelo</span>
          <span class="serie-val">{s['model_hits']}/{s['model_n']}</span>
          <span class="serie-sub">{pct_model:.0f}% de acerto</span></div>
        <div class="serie-stat"><span class="serie-label">mercado</span>
          <span class="serie-val">{s['market_hits']}/{s['market_n']}</span>
          <span class="serie-sub">{pct_market:.0f}% de acerto</span></div>
        <div class="serie-stat"><span class="serie-label">pernas EV&gt;1 (1u cada)</span>
          <span class="serie-val {pnl_cls}">{s['legs_pnl']:+.2f}u</span>
          <span class="serie-sub">{s['legs_won']}/{s['legs_n']} pernas ganhas</span></div>
        {_clv_stat_html(history_df)}
      </div>
    </div>"""


def _clv_stat_html(history_df: pd.DataFrame) -> str:
    """
    CLV médio como quarta estatística da série. Fica junto do P&L de propósito:
    é a mesma pergunta ("o modelo tem edge?") medida de um jeito que precisa de
    muito menos amostra, então ver os dois lado a lado é o ponto.
    """
    c = compute_clv_summary(history_df)
    if c is None:
        return ""
    cls = "pos" if c["media"] > 0 else ("neg" if c["media"] < 0 else "")
    return f"""<div class="serie-stat">
          <span class="serie-label">CLV médio</span>
          <span class="serie-val {cls}">{c['media'] * 100:+.1f}<small>pp</small></span>
          <span class="serie-sub">{c['positivos']}/{c['n']} bateram o fecho</span></div>"""


def _closing_alert_html(history_df: pd.DataFrame) -> str:
    """
    Aviso de fechamento nao capturado, no topo do painel.

    Fica ACIMA do placar da serie de proposito: quando o CLV nao foi
    capturado, o numero que aparece logo abaixo esta medido sobre menos
    pernas do que o card teve, e quem le precisa saber disso antes de olhar
    a media, nao depois.
    """
    faltando = events_missing_closing(history_df)
    if not faltando:
        return ""

    itens = []
    for f in faltando:
        if f["estado"] == "iminente":
            # comando em linha propria, sem quebra: partido no meio ("scripts.ca
            # / pture_closing") ele deixa de ser copiavel, que e o unico motivo
            # de estar aqui
            recado = ("o card é logo e o fechamento ainda não foi capturado — "
                      "rode isto antes de ele começar:"
                      f'<code class="clv-cmd">python -m scripts.capture_closing '
                      f"--event-date {_e(f['event_date'])}</code>")
        else:
            recado = (f"o card passou sem captura: as {f['n']} pernas ficam sem CLV, "
                      f"e não há backfill (a API só serve eventos futuros)")
        itens.append(f'<li><strong>{_e(f["event_name"])}</strong> '
                     f'<span class="clv-alert-date">{_e(f["event_date"])}</span> — {recado}</li>')

    urgente = any(f["estado"] == "iminente" for f in faltando)
    cls = "urgente" if urgente else ""
    titulo = "Fechamento pendente" if urgente else "Fechamento não capturado"
    return (f'<div class="clv-alert {cls}"><div class="clv-alert-title">{titulo}</div>'
            f'<ul>{"".join(itens)}</ul></div>')


def _sharp_split_html(history_df: pd.DataFrame) -> str:
    """Bloco 'com sinal sharp vs sem' — so aparece quando ja ha dado."""
    split = compute_sharp_split(history_df)
    if split is None:
        return ""

    def cell(label: str, g: dict, hint: str) -> str:
        if g["n"] == 0:
            return (f'<div class="serie-stat"><span class="serie-label">{label}</span>'
                    f'<span class="serie-val">—</span>'
                    f'<span class="serie-sub">sem pernas ainda</span></div>')
        cls = "pos" if g["pnl"] > 0 else ("neg" if g["pnl"] < 0 else "")
        return (f'<div class="serie-stat"><span class="serie-label">{label}</span>'
                f'<span class="serie-val {cls}">{g["pnl"]:+.2f}u</span>'
                f'<span class="serie-sub">{g["won"]}/{g["n"]} pernas · {hint}</span></div>')

    total = split["com_sharp"]["n"] + split["sem_sharp"]["n"]
    return f"""
    <div class="serie-box">
      <div class="serie-title">Com respaldo sharp vs sem
        <span class="hist-date">{total} perna(s) medida(s) — dado gravado desde 01/08/2026;
        eventos anteriores não têm e ficam de fora</span></div>
      <div class="serie-grid">
        {cell("pernas COM sinal sharp", split["com_sharp"], "odd acima do justo da Pinnacle")}
        {cell("pernas SEM sinal sharp", split["sem_sharp"], "só o modelo aponta valor")}
      </div>
      <p class="tab-explain" style="margin-top:10px">A hipótese a testar: se o grupo COM sinal
      sharp for consistentemente melhor ao longo de ~10 eventos, o critério da série passa a ser
      "EV do modelo <strong>e</strong> respaldo sharp", não só EV do modelo. Amostra pequena
      ainda não decide nada.</p>
    </div>"""


def _event_legs_pnl(ev: pd.DataFrame) -> str:
    """Chip de P&L das pernas EV>1 de UM evento fechado ('' se nao ha pernas)."""
    pnl, n = 0.0, 0
    for _, row in ev[ev["actual_winner"].notna()].iterrows():
        leg = model_side_leg(row)
        if leg is None or leg["ev"] <= 1:
            continue
        n += 1
        pnl += (leg["odd"] - 1) if row["actual_winner"] == leg["side"] else -1
    if n == 0:
        return ""
    cls = "pos" if pnl > 0 else ("neg" if pnl < 0 else "")
    return f'<span class="hist-score {cls}">pernas {pnl:+.2f}u</span>'


def render_history_panel(history_df: pd.DataFrame) -> str:
    """Conteudo do painel da aba Historico: placar acumulado da serie no
    topo + um bloco por evento (mais recente primeiro), com placar
    agregado modelo vs mercado no cabecalho."""
    if history_df.empty:
        return '<p class="note">Nenhum evento registrado ainda — o histórico começa no próximo card publicado.</p>'

    blocks = [_closing_alert_html(history_df), _series_summary_html(history_df),
              _sharp_split_html(history_df)]
    keys = (history_df[["event_name", "event_date"]].drop_duplicates()
            .sort_values("event_date", ascending=False))
    for _, (event_name, event_date) in keys.iterrows():
        ev = history_df[(history_df["event_name"] == event_name)
                        & (history_df["event_date"] == event_date)]
        closed = ev[ev["actual_winner"].notna()]
        if closed.empty:
            score_html = '<span class="hist-badge none">aguardando resultados</span>'
        else:
            model_hits = int(closed["model_correct"].fillna(False).astype(bool).sum())
            model_n = int(closed["model_correct"].notna().sum())
            market_hits = int(closed["market_correct"].fillna(False).astype(bool).sum())
            market_n = int(closed["market_correct"].notna().sum())
            score_html = (f'<span class="hist-score model-score">modelo {model_hits}/{model_n}</span>'
                          f'<span class="hist-score">mercado {market_hits}/{market_n}</span>'
                          f'{_event_legs_pnl(ev)}')
        rows = "".join(_history_fight_row(r) for _, r in ev.iterrows())
        blocks.append(f"""
    <div class="hist-event">
      <div class="hist-head">
        <div class="hist-title">{_e(event_name)} <span class="hist-date">{_e(event_date)}</span></div>
        <div>{score_html}</div>
      </div>
      <div class="hist-scroll"><table class="hist-table">
        <thead><tr><th>luta</th><th>lado do modelo</th><th>lado do mercado</th><th>vencedor</th></tr></thead>
        <tbody>{rows}</tbody>
      </table></div>
    </div>""")
    return "\n".join(blocks)


HISTORY_CSS = """
  /* alarme de fechamento nao capturado. Acima do placar porque muda como o
     placar deve ser lido. Barra lateral em vez de caixa colorida: precisa
     interromper a leitura sem virar banner de erro de sistema. */
  .clv-alert { border-left: 3px solid var(--gold); background: var(--surface);
    padding: 14px 18px; margin-bottom: 22px; }
  .clv-alert.urgente { border-left-color: var(--red); }
  .clv-alert-title { font-family: var(--font-display); font-weight: 700;
    text-transform: uppercase; letter-spacing: .1em; font-size: .68rem;
    color: var(--gold); margin-bottom: 8px; }
  .clv-alert.urgente .clv-alert-title { color: var(--red); }
  .clv-alert ul { margin: 0; padding-left: 18px; }
  .clv-alert li { font-size: .8rem; line-height: 1.5; color: var(--muted); }
  .clv-alert li + li { margin-top: 6px; }
  .clv-alert strong { color: var(--text); font-weight: 600; }
  .clv-alert-date { font-family: var(--font-num); font-size: .72rem; }
  .clv-alert code { font-family: var(--font-num); font-size: .72rem;
    background: var(--bg); padding: 1px 5px; border-radius: 2px; }
  /* o comando rola na horizontal em vez de quebrar: partido no meio do token
     ele deixa de ser copiavel, e ser copiavel e a razao de ele existir */
  .clv-alert .clv-cmd { display: block; margin-top: 7px; padding: 6px 9px;
    white-space: nowrap; overflow-x: auto; color: var(--dim); }

  /* placar da serie: numeros grandes numa regua, sem caixa */
  .serie-box { border-top: 2px solid var(--green); border-bottom: 1px solid var(--line);
    background: var(--surface); padding: 18px 22px 20px; margin-bottom: 26px; }
  .serie-title { font-family: var(--font-display); font-weight: 700; text-transform: uppercase;
    letter-spacing: .1em; font-size: .74rem; color: var(--muted); margin-bottom: 16px; }
  .serie-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 22px; }
  .serie-stat { min-width: 0; }
  /* o "vs" entre os nomes na tabela: presente, sem competir com eles */
  .vs { color: var(--muted); font-style: italic; font-size: .74rem; margin: 0 2px; }
  .serie-label { display: block; color: var(--muted); font-size: .62rem;
    text-transform: uppercase; letter-spacing: .12em; }
  .serie-val { display: block; font-family: var(--font-num); font-weight: 700;
    font-size: 1.75rem; line-height: 1.15; margin: 6px 0 3px;
    font-variant-numeric: tabular-nums; }
  .serie-val.pos { color: var(--green); }
  .serie-val.neg { color: var(--red); }
  .serie-val small { font-size: .5em; color: var(--muted); margin-left: 3px;
    letter-spacing: .08em; }
  .serie-sub { display: block; color: var(--muted); font-size: .68rem; }

  /* eventos passados */
  .hist-event { border-top: 1px solid var(--line); padding: 18px 0 6px; }
  .hist-head { display: flex; justify-content: space-between; align-items: baseline;
    gap: 10px; flex-wrap: wrap; margin-bottom: 12px; }
  .hist-title { font-family: var(--font-display); text-transform: uppercase;
    letter-spacing: .02em; font-size: 1rem; font-weight: 700; }
  .hist-date { color: var(--muted); font-size: .7rem; margin-left: 10px;
    font-family: var(--font-num); }
  .hist-score { font-family: var(--font-num); font-size: .7rem; color: var(--muted);
    margin-left: 12px; }
  .hist-score.model-score { color: var(--gold); }
  .hist-score.pos { color: var(--green); }
  .hist-score.neg { color: var(--red); }
  .hist-scroll { overflow-x: auto; }
  .hist-table { width: 100%; border-collapse: collapse; font-size: .8rem; }
  .hist-table th { text-align: left; color: var(--muted); font-size: .62rem;
    text-transform: uppercase; letter-spacing: .12em; font-weight: 600;
    padding: 0 14px 8px 0; border-bottom: 1px solid var(--line); white-space: nowrap; }
  .hist-table td { padding: 9px 14px 9px 0; border-bottom: 1px solid var(--line-soft);
    vertical-align: middle; }
  .hist-table tr:last-child td { border-bottom: none; }
  .hist-fight { font-weight: 600; color: var(--dim); }
  .hist-prob { color: var(--muted); font-family: var(--font-num); font-size: .72rem; }
  .hist-pending { color: var(--muted); font-style: italic; font-size: .76rem; }
  /* marcador de acerto/erro: quadrado solido, legivel de relance na coluna */
  .hist-badge { font-size: .62rem; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; padding: 2px 7px; white-space: nowrap; }
  .hist-badge.hit { background: var(--green); color: var(--bg); }
  .hist-badge.miss { background: var(--red); color: #fff; }
  .hist-badge.none { color: var(--muted); border: 1px solid var(--line); }
"""
