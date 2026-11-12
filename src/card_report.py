"""
src/card_report.py

Relatorio visual (HTML self-contained) para um card inteiro de UFC,
cruzando as probabilidades do modelo com odds reais de mercado fornecidas
MANUALMENTE pelo usuario (busca automatica de odds ao vivo e
deliberadamente fora de escopo -- ver README).

Abas:
  - "Favoritos mais seguros": ranking decrescente pela probabilidade de
    mercado (devig) do favorito, com marcacao de concordancia do modelo.
  - "Melhores zebras da noite": ranking decrescente pela DIVERGENCIA
    positiva = P(modelo, azarao) - P(mercado devig, azarao). Zebra "boa"
    nao e a de odd mais alta, e a que o modelo acha mais competitiva do
    que o mercado precifica.
  - "Sem previsao": lutas com lutador fora da base (estreantes) -- nunca
    descartadas silenciosamente.

Entrada: CSV com colunas fighter_a, fighter_b, odds_a_decimal,
odds_b_decimal (SEM actual_winner -- lutas futuras; o odds_template.csv
existente e outra coisa: backtest de lutas passadas com resultado).

Uso:
    python -m src.card_report data/raw/upcoming_card_odds.csv --output card_report.html
"""
from __future__ import annotations

import argparse
import hashlib
import html as html_mod
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

import config
from src.prediction_history import (frozen_predictions_for_event, load_history,
                                    record_card_predictions, render_history_panel,
                                    set_photo_map, sync_results_from_template)
from src.utils import decimal_odds_to_implied_prob, remove_vig_two_way

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Fontes que produzem o HTML. Nao inclui share_image (gera o PNG, nao a
# pagina) nem os modulos de modelo (mudam o CONTEUDO, e o conteudo ja e
# datado pelo "gerado em").
_GENERATOR_SOURCES = ("src/card_report.py", "src/design.py",
                      "src/prediction_history.py", "src/painel.py")

# "a base nao foi consultada", que e diferente de "a consulta falhou"
FRESCOR_NAO_VERIFICADO = "nao_verificado"


@lru_cache(maxsize=1)
def generator_fingerprint() -> str:
    """
    Hash curto das fontes que desenham a pagina.

    Existe porque `docs/index.html` e um ARTEFATO: ele nao e atualizado por
    commit, so por regeracao. Em 28/ago/2026 isso custou uma rodada inteira --
    a pagina no ar era de quatro dias antes, com o desenho antigo, e a unica
    pista disponivel era um "gerado em <data>" que ninguem tem como comparar
    com a versao do codigo. O carimbo da a comparacao de graca: hash diferente
    do atual = pagina velha. Quem cobra isso e tests/test_report_freshness.py.

    FIM DE LINHA NORMALIZADO. O hash era dos bytes crus, entao o mesmo
    codigo com CRLF (Windows, onde `core.autocrlf` entrega fim de linha de
    Windows no checkout) e com LF (Linux, CI, qualquer clone com .gitattributes
    diferente) dava carimbos diferentes -- e o teste de frescor acusava
    "pagina velha" por causa do sistema operacional de quem regerou, nao por
    causa do desenho. Alarme que soa sem motivo e alarme que se aprende a
    ignorar.
    """
    h = hashlib.sha256()
    for rel in _GENERATOR_SOURCES:
        cru = (config.PROJECT_ROOT / rel).read_bytes()
        h.update(cru.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
    return h.hexdigest()[:12]


def load_card_odds(csv_path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["fighter_a", "fighter_b", "odds_a_decimal", "odds_b_decimal"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV de card sem as colunas {missing} (esperadas: {required}). "
                         "Nota: este formato NAO tem actual_winner -- sao lutas futuras.")
    df = df.dropna(subset=required)
    bad = df[(df["odds_a_decimal"] <= 1.0) | (df["odds_b_decimal"] <= 1.0)]
    if not bad.empty:
        raise ValueError(f"Odds decimais devem ser > 1.0. Linhas invalidas:\n{bad[required]}")
    # `scheduled_rounds`, se vier no CSV, e ignorada: era feature do modelo de
    # faixa de round, removido com a previsao de duracao (ago/2026). Colunas
    # extras nao atrapalham, entao CSVs antigos seguem valendo.
    return df


def _default_predict_fns(model_name: str) -> tuple[Callable, Callable]:
    """Prepara os preditores (vencedor e metodo) com a base de niveis
    carregada UMA vez para o card todo. allow_debutant=True: estreante vira
    linha sintetica (stats NaN, Elo base) em vez de derrubar a luta do
    relatorio — a previsao sai marcada com aviso proprio no card."""
    from src.features import export_latest_fighter_levels
    from src.predict import predict_fight, predict_method
    levels = export_latest_fighter_levels()
    winner_fn = lambda a, b: predict_fight(a, b, model_name=model_name, levels=levels,  # noqa: E731
                                           allow_debutant=True)
    method_fn = lambda a, b: predict_method(a, b, levels=levels, allow_debutant=True)  # noqa: E731
    return winner_fn, method_fn


def _frozen_for_fight(frozen: Optional[dict], a: str, b: str) -> Optional[dict]:
    """
    Entrada congelada da luta, ou None se ela nao esta fechada. Devolve
    {"model_prob_a", "method_probs"} ja na orientacao do CSV: model_prob_a
    inverte quando o historico gravou os lados na ordem oposta; method_probs
    nao, porque as labels de metodo sao simetricas (KO/finalizacao/decisao
    nao dependem de quem e "A").
    """
    if not frozen:
        return None
    if (a, b) in frozen:
        return frozen[(a, b)]
    if (b, a) in frozen:
        entry = frozen[(b, a)]
        prob = entry["model_prob_a"]
        return {**entry,
                "model_prob_a": None if prob is None else 1.0 - prob,
                "method_probs": entry["method_probs"]}
    return None


# Teto de plausibilidade para odd decimal. A maior zebra que o projeto ja viu
# pagava 6,70 (Demopoulos, 26/set); 100 e folgado o suficiente para nunca
# recusar preco legitimo e apertado o suficiente para barrar lixo vindo de uma
# API de terceiro. Sem teto, `odd > 1.0` aceita ate infinito -- e um EV
# infinito poria a perna no topo da aba EV>1 sem nenhum aviso.
_ODD_MAX = 100.0


def _texto_da_linha(row, coluna: str) -> str:
    """Valor de texto do CSV do card, vazio quando a coluna nao existe ou
    esta em branco. CSV antigo nao tem as colunas de prova, e ficar sem prova
    e o estado correto dele -- nao um defeito a contornar."""
    if coluna not in row.index or pd.isna(row.get(coluna)):
        return ""
    return str(row[coluna]).strip()


def _odd_plausivel(x: float) -> bool:
    return x == x and float("-inf") < x < float("inf") and 1.0 < x <= _ODD_MAX


def _odds_da_casa(row) -> tuple[float | None, float | None]:
    """(odd_a, odd_b) da Thunderpick desta linha do CSV, ou (None, None).

    Os dois lados andam juntos: meia luta com odd da casa e meia com mediana
    nao e preco de lugar nenhum.
    """
    try:
        a, b = float(row["odds_a_tp"]), float(row["odds_b_tp"])
    except (KeyError, TypeError, ValueError):
        return None, None
    if not (_odd_plausivel(a) and _odd_plausivel(b)):
        return None, None
    return a, b


def analyze_card(odds_df: pd.DataFrame, model_name: str = "logreg",
                 predict_fn: Optional[Callable[[str, str], dict]] = None,
                 method_fn: Optional[Callable[[str, str], dict]] = None,
                 frozen: Optional[dict] = None,
                 somente_snapshot: bool = False) -> dict:
    """
    Categorizacao MUTUAMENTE EXCLUSIVA por luta (nunca os dois lados do
    mesmo confronto em abas diferentes):

      - model_side  = lado mais provavel segundo o MODELO (argmax);
      - market_side = favorito do MERCADO (maior prob. implicita devig);
      - model_side == market_side -> "Favoritos" (concordancia);
      - model_side != market_side -> "Zebras" (o modelo aponta o azarao do
        mercado como o lado mais provavel de VENCER -- divergencia real).

    Cada luta com previsao valida cai em exatamente UMA das duas listas;
    lutas sem previsao de vencedor vao para o grupo "sem previsao".
    Ordenacao dentro de cada lista: decrescente pela probabilidade do
    modelo para o model_side (a probabilidade de mercado fica visivel no
    card como contexto, mas nao e criterio de ordenacao).

    predict_fn / method_fn sao injetaveis para teste. As duas previsoes
    (vencedor / metodo) falham de forma INDEPENDENTE: falha so no metodo
    mantem a luta na categoria com a secao de tendencia marcada como
    indisponivel.

    `frozen`: previsoes publicadas das lutas JA FECHADAS (ver
    prediction_history.frozen_predictions_for_event). Para essas, exibe o
    vencedor E o metodo COMO FORAM PUBLICADOS em vez do recalculo — que, com
    o evento ja na base, enxergaria o proprio resultado. Luta fechada sem
    metodo congelado (eventos anteriores a ago/2026, quando as colunas
    passaram a existir) fica sem metodo: preferimos a lacuna ao numero
    contaminado.
    """
    if predict_fn is None and not somente_snapshot:
        predict_fn, default_method_fn = _default_predict_fns(model_name)
        if method_fn is None:
            method_fn = default_method_fn

    predicted, no_prediction = [], []
    for card_order, (_, row) in enumerate(odds_df.iterrows()):
        a, b = str(row["fighter_a"]).strip(), str(row["fighter_b"]).strip()
        odds_a, odds_b = float(row["odds_a_decimal"]), float(row["odds_b_decimal"])

        # DUAS odds por lado, com trabalhos diferentes (ver src/thunderpick.py):
        #   odds_a/b       mediana das casas -> define o favorito do MERCADO,
        #                  que e o benchmark contra o qual o modelo e medido;
        #   odds_a/b_casa  Thunderpick -> e a odd que da para pegar, entao e
        #                  ela que aparece no card, entra no EV e decide se a
        #                  perna e EV>1.
        # Sem odd da casa a mediana serve de substituta, mas a luta fica
        # marcada: o EV dela nao e o preco que se pega.
        tp_a, tp_b = _odds_da_casa(row)
        tem_casa = tp_a is not None and tp_b is not None
        odds_a_casa, odds_b_casa = (tp_a, tp_b) if tem_casa else (odds_a, odds_b)

        market_a, market_b = remove_vig_two_way(
            decimal_odds_to_implied_prob(odds_a), decimal_odds_to_implied_prob(odds_b))

        # card_order: posicao no CSV = posicao no card (main event primeiro).
        # As abas reordenam por probabilidade, entao sem isso o destaque do
        # topo nao teria como saber qual e a luta principal.
        base = {"fighter_a": a, "fighter_b": b, "odds_a": odds_a, "odds_b": odds_b,
                "odds_a_casa": odds_a_casa, "odds_b_casa": odds_b_casa,
                "odds_da_casa": tem_casa,
                "market_prob_a": market_a, "market_prob_b": market_b,
                "card_order": card_order,
                # PROVA DE ORGANIZACAO (ver src/casamento.organizacao_confirmada).
                # Vem do CSV do card, onde `build_card` grava de qual fonte a
                # luta veio. CSV antigo nao tem a coluna e a luta fica sem
                # prova -- que e o estado correto, nao um defeito.
                "fonte_card": (str(row["fonte"]).strip()
                               if "fonte" in row.index and pd.notna(row.get("fonte"))
                               else ""),
                # PROVA DA ODD DA CASA: e ela que determina EV e retorno, e a
                # resposta da The Odds API nao a comprova. Uma por LADO: a
                # odd do lado B e provada pela linha do lado B, nao pela do
                # adversario.
                "ref_tp": _texto_da_linha(row, "ref_tp"),
                "ref_tp_b": _texto_da_linha(row, "ref_tp_b")}

        # SNAPSHOT MANDA quando existe. O preditor so e chamado para luta que
        # ainda nao tem previsao congelada -- e, se ele falhar, uma luta JA
        # CONGELADA continua desenhavel. Ate nov/2026 nao era assim: o
        # predict_fn vinha antes do snapshot, entao um relatorio historico
        # inteiro dependia do modelo atual estar carregavel, e qualquer falha
        # nele mandava para "sem previsao" linhas que tinham snapshot.
        entry_cong = _frozen_for_fight(frozen, a, b)
        congelada_utilizavel = (entry_cong is not None
                                and entry_cong.get("model_prob_a") is not None)
        origem_previsao = "modelo"
        if somente_snapshot:
            # Nao chama preditor NENHUM: nem para tentar. Em clone limpo nao
            # ha modelo nem base, e "tentar e cair no except" dependeria de o
            # erro ser de um tipo previsto -- importar o modelo pode falhar de
            # muitas formas. Snapshot suficiente desenha; o resto fica
            # INDISPONIVEL, nunca recalculado em silencio.
            if not congelada_utilizavel:
                # Ponteiro que nao resolve vira AUSENCIA com motivo
                # explicito. Escolher outra versao em silencio seria desenhar
                # um numero que ninguem pediu.
                porque = (entry_cong or {}).get("motivo") or (
                    "sem versao de previsao gravada para esta luta; em "
                    "somente_render nada e recalculado")
                no_prediction.append({**base, "reason": porque})
                continue
            origem_previsao = "snapshot"
            p_a = float(entry_cong["model_prob_a"])
            pred = {"fighter_a": a, "fighter_b": b,
                    "prob_a_wins": p_a, "prob_b_wins": 1 - p_a,
                    "model_used": "snapshot",
                    # A versao gravada nao guarda as marcas de experiencia.
                    # False aqui NAO quer dizer "lutador experiente": quer
                    # dizer "nao registrado", e a pagina diz isso (ver
                    # `experiencia_nao_registrada`). Reconstruir da base atual
                    # olharia lutas posteriores a previsao.
                    "experiencia_registrada": False,
                    "fighter_a_low_experience": False,
                    "fighter_b_low_experience": False,
                    "fighter_a_debutant": False, "fighter_b_debutant": False}
        else:
            try:
                pred = predict_fn(a, b)
            except (ValueError, FileNotFoundError, OSError) as exc:
                if not congelada_utilizavel:
                    # sem snapshot e sem modelo: INDISPONIVEL, nunca recalculo
                    # silencioso
                    no_prediction.append({**base, "reason": str(exc)})
                    continue
                origem_previsao = "snapshot"
                p_a = float(entry_cong["model_prob_a"])
                pred = {"fighter_a": a, "fighter_b": b,
                        "prob_a_wins": p_a, "prob_b_wins": 1 - p_a,
                        "model_used": "snapshot",
                        "experiencia_registrada": False,
                        "fighter_a_low_experience": False,
                        "fighter_b_low_experience": False,
                        "fighter_a_debutant": False, "fighter_b_debutant": False}
                logger.info("%s vs %s: preditor indisponivel (%s) -- desenhando a "
                            "partir do snapshot congelado.", a, b, exc)

        # OS DOIS estreantes: a previsao seria o perfil sintetico de estreia
        # contra ele mesmo -- todas as diferencas zeram e a saida e ~50% por
        # construcao, sem nenhuma informacao sobre esta luta. Um estreante SO
        # segue valendo (apoia-se nos dados do adversario, com aviso proprio),
        # mas dois nao: exibir isso como leitura, e deixar entrar no criterio
        # EV>1, seria apresentar ignorancia como previsao. Visto no card de
        # 29/ago/2026 (Bilal Hasan x Nilson Rojas, os dois sem historico).
        if pred.get("fighter_a_debutant") and pred.get("fighter_b_debutant"):
            no_prediction.append({**base, "reason":
                                  f"{a} e {b} estreando no UFC — sem histórico dos dois "
                                  f"lados, o modelo não tem base para estimar esta luta."})
            continue

        # luta ja fechada: a previsao publicada manda. Recalcular aqui usaria
        # niveis que ja incluem o resultado desta luta (ver `frozen` acima).
        entry = entry_cong
        # `fechada` so vem das entradas montadas a partir dos snapshots, que
        # cobrem tambem luta ABERTA. Marcar como fechada o que nao esta
        # mudaria o que a pagina afirma, entao o default e True (o caminho
        # antigo so trazia luta encerrada).
        is_closed = entry is not None and entry.get("fechada", True)

        # metodo: falha independente (nao derruba a previsao de vencedor)
        method_probs = None
        if is_closed:
            # Card ja encerrado. Se o metodo foi congelado no pre-registro,
            # exibe o congelado; se nao foi (eventos anteriores a ago/2026),
            # a luta fica SEM metodo -- um recalculo aqui olharia o proprio
            # resultado, e mostrar isso seria pior que nao mostrar nada.
            method_probs = entry["method_probs"]
        elif somente_snapshot and entry is not None:
            # desenhando so do gravado: o metodo tambem vem da versao, nunca
            # de um recalculo que aqui nem seria possivel
            method_probs = entry["method_probs"]
        elif method_fn is not None:
            try:
                method_probs = method_fn(a, b)["method_probs"]
            except (ValueError, FileNotFoundError) as exc:
                logger.info("Sem tendencia de metodo para %s vs %s: %s", a, b, exc)

        frozen_a = entry["model_prob_a"] if is_closed else None
        model_a = pred["prob_a_wins"] if frozen_a is None else frozen_a
        fav_is_a = market_a >= market_b
        model_side_is_a = model_a >= 0.5
        fight = {
            **base,
            "method_probs": method_probs,
            # nomes como casados na base (fuzzy pode ter corrigido grafia)
            "matched_a": pred["fighter_a"], "matched_b": pred["fighter_b"],
            "origem_previsao": origem_previsao,
            "model_prob_a": model_a, "model_prob_b": 1 - model_a,
            "frozen": frozen_a is not None,
            "low_experience": pred["fighter_a_low_experience"] or pred["fighter_b_low_experience"],
            "experiencia_nao_registrada": pred.get("experiencia_registrada", True) is False,
            "debutants": [n for n, d in ((a, pred.get("fighter_a_debutant")),
                                         (b, pred.get("fighter_b_debutant"))) if d],
            "favorite": a if fav_is_a else b,
            "underdog": b if fav_is_a else a,
            "market_prob_fav": market_a if fav_is_a else market_b,
            "market_prob_dog": market_b if fav_is_a else market_a,
            "model_prob_fav": model_a if fav_is_a else 1 - model_a,
            "model_prob_dog": (1 - model_a) if fav_is_a else model_a,
            "model_side": a if model_side_is_a else b,
            "model_side_prob": model_a if model_side_is_a else 1 - model_a,
        }
        fight["category"] = "favorite" if fight["model_side"] == fight["favorite"] else "underdog"
        # perna do paper trading: odd e EV do lado apontado pelo modelo.
        # EV = p_modelo x odd; > 1 = "perna EV>1" (regra de pre-registro da
        # serie). EV auto-referente: assume que o modelo esta certo — e o
        # backtest mostra que o mercado esta na frente. Exibido com aviso.
        fight["model_side_odds"] = (odds_a_casa if fight["model_side"] == a
                                    else odds_b_casa)
        fight["ev"] = fight["model_side_prob"] * fight["model_side_odds"]
        predicted.append(fight)

    favorites = sorted((f for f in predicted if f["category"] == "favorite"),
                       key=lambda f: f["model_side_prob"], reverse=True)
    underdogs = sorted((f for f in predicted if f["category"] == "underdog"),
                       key=lambda f: f["model_side_prob"], reverse=True)
    # aba de metodo: ordenacao "fria" pela probabilidade da categoria mais
    # provavel de cada luta; lutas sem dado ficam num grupo a parte da aba
    method_ranking = sorted((f for f in predicted if f["method_probs"]),
                            key=lambda f: max(f["method_probs"].values()), reverse=True)
    no_method = [f for f in predicted if not f["method_probs"]]
    ev_legs = sorted((f for f in predicted if f["ev"] > 1),
                     key=lambda f: f["ev"], reverse=True)
    # luta principal = primeira do CSV, para o destaque do topo
    main_event = min(predicted, key=lambda f: f["card_order"], default=None)
    return {"favorites": favorites, "underdogs": underdogs, "no_prediction": no_prediction,
            "method_ranking": method_ranking, "no_method": no_method,
            "ev_legs": ev_legs, "main_event": main_event, "model_name": model_name}


# ---------------------------------------------------------------------------
# HTML (self-contained: CSS + JS puros inline, funciona offline)
# ---------------------------------------------------------------------------

def _e(text) -> str:
    return html_mod.escape(str(text))


def _gap_bar(model_a: float, market_a: float, delta_side: float,
             a_is_pick: bool) -> str:
    """
    A luta inteira num eixo so: onde o modelo poe a divisao, onde o mercado
    poe, e o VAO entre os dois — que e o produto do relatorio.

    Antes eram tres linhas (modelo, mercado, delta) repetindo o MESMO eixo de
    0 a 100. Repetir o eixo tres vezes obriga o leitor a reconstruir na cabeca
    a comparacao que o desenho ja podia mostrar, e gastava tres vezes a
    altura. Aqui a barra enche ate a estimativa do modelo, um tique marca a do
    mercado, e o espaco vermelho entre os dois E o delta: nao ha numero a
    conferir, o tamanho do vao ja diz.

    O eixo e a LUTA, nao um lutador: esquerda e o fighter_a do CSV, direita o
    fighter_b, na mesma ordem dos nomes logo acima.

    `delta_side` e model − mercado PARA O LADO QUE O MODELO APONTA, em pontos
    de probabilidade. Negativo significa que o modelo e menos confiante que o
    mercado no proprio palpite — o padrao do projeto em favorito pesado.

    Cor do vao: vermelho quando ha divergencia mensuravel, nada quando nao ha.
    Deliberadamente NAO uso verde/vermelho por sinal: isso atribuiria
    "bom/ruim" a uma divergencia, e divergencia nao e aposta boa — o backtest
    mostra o contrario. E nao uso ouro no estado de concordancia porque ouro ja
    significa "modelo e mercado apontam o mesmo LADO" nas abas; duas regras
    para a mesma cor ensinariam coisas diferentes.
    """
    state, thick = _delta_band(delta_side)
    lo, hi = sorted((model_a, market_a))
    span = (f'<span class="gap-span" style="left:{lo * 100:.1f}%;'
            f'width:{(hi - lo) * 100:.1f}%;height:{thick}px"></span>' if thick else "")
    lcls = "seg l" + (" side" if a_is_pick else "")
    rcls = "seg r" + ("" if a_is_pick else " side")
    return f"""
      <div class="gap" data-state="{state}">
        <span class="gap-num l{' side' if a_is_pick else ''}">{model_a * 100:.1f}</span>
        <div class="gap-track">
          <div class="{lcls}" style="width:{model_a * 100:.1f}%"></div>
          <div class="{rcls}" style="width:{(1 - model_a) * 100:.1f}%"></div>
          {span}
          <span class="gap-mark" style="left:{market_a * 100:.1f}%"></span>
        </div>
        <span class="gap-num r{'' if a_is_pick else ' side'}">{(1 - model_a) * 100:.1f}</span>
      </div>
      <div class="gap-legend">
        <span>modelo</span>
        <span class="gap-market">mercado {market_a * 100:.1f}</span>
        <span class="gap-delta">delta {delta_side * 100:+.1f}<small>pp</small></span>
      </div>"""


# Faixas do Delta Marker. NÃO afirmam significância estatística — são
# hierarquia visual. A espessura codifica a CLASSE; o comprimento do vão
# codifica o valor real. Assim a magnitude aparece duas vezes sem gastar
# uma segunda cor.
_DELTA_BANDS = [(3.0, "agreement", 0), (10.0, "small", 2),
                (20.0, "material", 3), (float("inf"), "large", 5)]


# Corpo da barra. O lado apontado pelo modelo recebe o acento DILUIDO no
# grafite, nao o acento cheio: preenchida, a barra dominava a pagina por area
# e o relatorio lia ouro-sobre-preto, quando a marca e preto e vermelho. A
# leitura de "quem esta na frente" fica com a capa solida no ponto de encontro
# (.seg.side::after), que custa 2px em vez de meia barra.
def _delta_band(delta_side: float) -> tuple[str, int]:
    """Classe e espessura do vao. Extraido porque o BLOCO da luta tambem usa a
    classe: a aresta vermelha do card e a mesma leitura do marcador."""
    mag = abs(delta_side) * 100
    return next((s, t) for lim, s, t in _DELTA_BANDS if mag < lim)


def _matched_note(fight: dict) -> str:
    notes = []
    if fight.get("matched_a") and fight["matched_a"] != fight["fighter_a"]:
        notes.append(f"“{_e(fight['fighter_a'])}” casado como “{_e(fight['matched_a'])}”")
    if fight.get("matched_b") and fight["matched_b"] != fight["fighter_b"]:
        notes.append(f"“{_e(fight['fighter_b'])}” casado como “{_e(fight['matched_b'])}”")
    if fight.get("debutants"):
        quem = " e ".join(_e(n) for n in fight["debutants"])
        notes.append(f"{quem} estreando no UFC (sem histórico na base) — previsão apoiada só nos "
                     f"dados do adversário e no perfil típico de estreia; confiança reduzida")
    elif fight.get("low_experience"):
        notes.append("pelo menos um lutador com poucas lutas na base — estimativa menos confiável")
    elif fight.get("experiencia_nao_registrada"):
        notes.append("avisos de experiência (estreante, poucas lutas) não registrados "
                     "nesta versão da previsão")
    if fight.get("entrada_bloqueada"):
        notes.append(f"sinal de entrada bloqueado: {_e(fight['entrada_bloqueada'])} "
                     f"— previsao e odds seguem exibidas; o que nao se "
                     f"propoe e a entrada")
    if not notes:
        return ""
    return f'<div class="note">⚠ {"; ".join(notes)}</div>'


_METHOD_LABELS = [("KO_TKO", "KO/TKO"), ("SUBMISSION", "Finalização"), ("DECISION", "Decisão")]

# Icones SVG inline (sem dependencia externa), neutros -- usados so na secao
# de metodo, em cinza, para nao competir com os acentos dourado/vermelho.
def anotar_sinal_de_entrada(analysis: dict, event_date: str) -> int:
    """
    Marca em cada luta se A NOSSA CASA paga acima do preco justo da sharp,
    nos DOIS lados. Devolve quantos lados ficaram marcados.

        entrada_X = sharp_prob_X x odd_X da Thunderpick

    Acima de 1 e o unico sinal do projeto que nao depende do nosso modelo E
    esta no preco que da para pegar. Nos dois lados de proposito: a entrada
    nao se importa com qual lutador o modelo prefere -- se a casa paga acima
    do justo no outro canto, e la que esta o valor.

    Roda DEPOIS de record_card_predictions, lendo o sharp congelado do
    historico: assim funciona igual numa regeracao com --no-sharp, e mostra o
    que foi registrado em vez de um recalculo de hoje.
    """
    from src.prediction_history import bloqueios_do_evento, frozen_sharp_for_event

    sharp_a = frozen_sharp_for_event(event_date) if event_date else {}
    if not sharp_a:
        return 0
    # Uma selecao exige prova; uma previsao registrada, nao. O bloqueio age
    # aqui e so aqui: a linha do historico fica como esta.
    bloqueios = bloqueios_do_evento(event_date)
    n = 0
    for f in analysis["favorites"] + analysis["underdogs"]:
        p_a = sharp_a.get((f["fighter_a"], f["fighter_b"]))
        if p_a is None or not f.get("odds_da_casa"):
            continue
        f["sharp_prob_a"] = p_a
        motivo = bloqueios.get((f["fighter_a"], f["fighter_b"]))
        if motivo:
            # sem entrada_a/entrada_b: nada a marcar, e o card diz por que
            f["entrada_bloqueada"] = motivo
            continue
        for lado, p, odd in (("a", p_a, f["odds_a_casa"]),
                             ("b", 1.0 - p_a, f["odds_b_casa"])):
            f[f"entrada_{lado}"] = p * odd
            n += int(p * odd > 1)
    return n


def _duas_odds_nota(analysis: dict) -> str:
    """
    Explica as DUAS odds da pagina. Sem isto o relatorio mente por omissao:
    a odd grande de cada lado vem da Thunderpick e a barra "mercado" ao lado
    dela vem da MEDIANA de 20+ casas, entao quem fizer 1/odd para conferir a
    barra acha um numero que nao bate e conclui que a pagina esta errada.

    So aparece quando ha odd de casa no card — descrever uma fonte que nao
    foi usada seria pior que nao descrever nada.
    """
    lutas = (analysis.get("favorites", []) + analysis.get("underdogs", [])
             + analysis.get("no_prediction", []))
    com = [f for f in lutas if f.get("odds_da_casa")]
    if not com:
        return ""
    sem = [f for f in lutas if not f.get("odds_da_casa")]
    faltando = ""
    if sem:
        nomes = ", ".join(f'{_e(f["fighter_a"])} vs {_e(f["fighter_b"])}' for f in sem[:3])
        faltando = (f' <strong>Sem preço na casa ({len(sem)} de {len(lutas)}):</strong> '
                    f'{nomes}{"…" if len(sem) > 3 else ""} — essas caem na mediana, '
                    f'e o EV delas não é o preço que você pega.')
    return (f'<p class="tab-explain"><strong>Duas odds nesta página.</strong> A odd de cada '
            f'lutador é a da <strong>Thunderpick</strong>, onde a entrada é feita — é ela '
            f'que entra no EV e decide as pernas EV&gt;1. Já a barra <em>mercado</em> e a '
            f'classificação favorito/zebra vêm da <strong>mediana de 20+ casas</strong>, que '
            f'é o benchmark contra o qual o modelo é medido. Por isso 1/odd não bate com a '
            f'barra: são fontes diferentes, de propósito.{faltando}</p>')


_MESES = ["jan", "fev", "mar", "abr", "mai", "jun",
          "jul", "ago", "set", "out", "nov", "dez"]
_DIAS = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]


def _format_event_date(event_date: str) -> str:
    """'2026-08-15' -> 'sáb · 15 ago 2026'. String vazia se nao parsear."""
    try:
        d = datetime.strptime(str(event_date), "%Y-%m-%d")
    except (ValueError, TypeError):
        return ""
    return f"{_DIAS[d.weekday()]} · {d.day} {_MESES[d.month - 1]} {d.year}"


_PAINEL_TPL = """
  <details class="prov">
    <summary>Proveniência deste snapshot &middot; {n_ok} de {n} seleções
      elegíveis</summary>
    <div class="prov-corpo">
      <p><b>De quando são os números.</b> Probabilidades: pré-registro
      congelado de {data}, modelo <code>{modelo}</code>, manifesto de treino
      <code>{manifesto}</code>. Preços: última captura registrada em
      <code>data/proveniencia/cotacoes.csv</code>, card
      <code>{card}</code>.</p>

      <p><b>Três leituras diferentes.</b> <i>Probabilidade de vitória</i> é o
      que o modelo estima. <i>Valor esperado estimado</i> é essa probabilidade
      vezes o preço &mdash; calculado com o nosso próprio modelo, portanto
      auto-referente, e não evidência de valor. <i>Qualidade do dado</i> é de
      quando o preço é, de que fonte, e se a prova dele existe.</p>

      <h3>Seleções elegíveis</h3>
      {elegiveis}

      <h3>Bloqueios</h3>
      <ul class="prov-lista">{bloqueios}</ul>

      <p class="note">Limites declarados: o instante registrado é o da
      <b>leitura</b> da cotação, não o instante em que a casa formou o preço
      (a fonte não informa). Validade operacional de {validade}h, escolhida
      pela cadência horária da captura e não por resultado histórico. Uma
      seleção elegível significa que as provas estão no lugar &mdash; não que
      a entrada é boa.</p>
    </div>
  </details>
"""


def _painel_de_proveniencia(analysis: dict, event_date: str,
                            csv_path=None) -> str:
    """
    Qual snapshot esta pagina exibe, e o que esta bloqueado.

    Existe porque a pagina mostrava numeros sem dizer de quando eram nem o
    que tinha sido excluido. Tres leituras ficam SEPARADAS de proposito,
    porque sao coisas diferentes e eram lidas como uma so:

      probabilidade de vitoria   o que o modelo estima;
      valor esperado estimado    aquela probabilidade vezes o preco. Usa o
                                 nosso proprio modelo, entao e
                                 auto-referente -- nao e evidencia de valor;
      qualidade do dado          de quando o preco e, de que fonte, e se a
                                 prova dele existe.

    Ausencia de oportunidade elegivel e RESULTADO VALIDO e aparece dita, nunca
    como painel vazio.
    """
    import collections

    from src.manifesto import sha256_arquivo
    from src.proveniencia import VALIDADE_HORAS

    if not event_date:
        return ""
    try:
        from src.decisoes import avaliar_com_precos_novos
        decisoes = avaliar_com_precos_novos(event_date, card_csv=csv_path)
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Painel de proveniencia indisponivel (%s).", exc)
        return ""
    if not decisoes:
        return ""

    elegiveis = [d for d in decisoes if d["elegivel"]]
    bloqueios = collections.Counter(d["bloqueio"] for d in decisoes
                                    if not d["elegivel"])
    card_hash = (sha256_arquivo(Path(csv_path)) or "")[:12] if csv_path else ""
    manifesto = (sha256_arquivo(config.PROJECT_ROOT / "models"
                                / "training_metadata.json") or "")[:12]

    if elegiveis:
        itens = "\n".join(
            "<li><b>{}</b> &middot; odd {} &middot; valor esperado estimado "
            "{}</li>".format(_e(d["lado"]), d["odd_usada"], d["ev"])
            for d in sorted(elegiveis, key=lambda d: -(d["ev"] or 0)))
        corpo_elegiveis = '<ul class="prov-lista">' + itens + "</ul>"
    else:
        corpo_elegiveis = ('<p class="note">Nenhuma seleção elegível neste '
                           'snapshot. Isso é um resultado, não uma falha: '
                           'significa que nenhum preço com prova completa '
                           'passou o critério.</p>')

    bloq_itens = "".join(
        "<li>{} &middot; <b>{}</b></li>".format(_e(motivo), n)
        for motivo, n in bloqueios.most_common()) or "<li>nenhum</li>"

    return _PAINEL_TPL.format(
        n_ok=len(elegiveis), n=len(decisoes), data=_e(event_date),
        modelo=_e(analysis["model_name"]),
        manifesto=manifesto or "ausente", card=card_hash or "sem hash",
        elegiveis=corpo_elegiveis, bloqueios=bloq_itens,
        validade=VALIDADE_HORAS)


def render_html(analysis: dict, freshness_gap_days: Optional[int], card_name: str = "",
                history_panel_html: str = "", event_date: str = "",
                output_hint: Optional[Path | str] = None,
                csv_path: Optional[Path | str] = None) -> str:
    """
    A pagina publicada. O desenho mora em `src/painel.py` (ficha de confronto
    na ordem do card); aqui so se junta o que a pagina recebe pronto: o
    painel de proveniencia deste snapshot e o carimbo do gerador.
    """
    from src.painel import render_pagina
    painel_prov = _painel_de_proveniencia(
        analysis, event_date, csv_path if csv_path is not None else config.UPCOMING_CARD_CSV)
    return render_pagina(
        analysis, freshness_gap_days, card_name=card_name,
        history_panel_html=history_panel_html, event_date=event_date,
        gerado_em=datetime.now().strftime("%d/%m/%Y %H:%M"),
        fingerprint=generator_fingerprint(),
        model_name=analysis.get("model_name", ""), prov_html=painel_prov)


def generate_card_report(csv_path: Path | str, output_path: Path | str,
                         model_name: str = "logreg", card_name: str = "",
                         event_date: str = "", photos: bool = False,
                         sharp: bool = True,
                         somente_render: bool = False) -> Path:
    """
    `event_date` (YYYY-MM-DD): data do evento, usada para registrar as
    previsoes no historico de paper trading (congeladas ate o resultado) e
    para casar os resultados vindos do odds_template.csv. Sem ela o
    relatorio e gerado normalmente, mas o card nao entra no historico.

    `photos`: busca fotos dos lutadores no UFC.com (hotlink + cache local;
    ver src/fighter_photos.py). Usado tambem na pagina publicada desde
    ago/2026. O HTML deixa de ser offline/self-contained; foto que nao
    carrega cai no monograma, entao o layout nunca quebra.

    `somente_render`: DESENHA e nao escreve. Pula o sync de resultados, o
    registro no historico e qualquer consulta de odds; usa apenas os
    snapshots ja gravados. Existe porque ate nov/2026 nao havia como
    regenerar a pagina sem tambem mexer nos registros -- renderizacao e
    alteracao de registro moravam no mesmo caminho, e qualquer conserto de
    layout arrastava o historico junto.

    `sharp`: consulta a The Odds API para congelar, junto de cada previsao,
    a probabilidade devigada da casa sharp (Pinnacle). Custa creditos da
    API; desligue com sharp=False em testes/regeracoes.
    """
    from src.data_collection import check_data_freshness

    odds_df = load_card_odds(csv_path)
    logger.info("Card carregado: %d lutas de %s", len(odds_df), csv_path)

    # em somente_render nem o frescor e consultado: ele le a base bruta, que
    # um clone limpo nao tem -- e desenhar o que ja esta gravado nao depende
    # de a base estar em dia.
    gap_days = (FRESCOR_NAO_VERIFICADO if somente_render
                else check_data_freshness())

    # ANTES de analisar: fecha o que ja tem resultado no odds_template. A ordem
    # importa duas vezes. (1) So depois do sync as lutas encerradas aparecem em
    # frozen_predictions_for_event, e sem isso o relatorio exibiria o recalculo
    # pos-evento. (2) record_card_predictions abaixo so respeita linha fechada;
    # com o sync depois dele, regerar um card ja resolvido reescrevia o
    # pre-registro com a previsao contaminada e SO ENTAO congelava — a previsao
    # publicada se perdia sem aviso.
    if somente_render:
        logger.info("somente_render: nenhuma escrita no historico, nenhuma "
                    "consulta de odds -- so desenha a partir do que ja esta "
                    "gravado.")
    else:
        sync_results_from_template()
    frozen = frozen_predictions_for_event(event_date) if event_date else None

    if somente_render and event_date:
        # As versoes gravadas cobrem tambem a luta ABERTA, e e isso que
        # permite desenhar sem modelo e sem base. A luta ja ENCERRADA
        # continua vindo do historico: so ele sabe que ela fechou.
        from src.snapshots import para_render
        frozen = {**para_render(event_date), **(frozen or {})}

    analysis = analyze_card(odds_df, model_name=model_name, frozen=frozen,
                            somente_snapshot=somente_render)

    n_frozen = sum(1 for f in analysis["favorites"] + analysis["underdogs"] if f["frozen"])
    logger.info("Previstas: %d | sem previsao: %d | congeladas (ja encerradas): %d",
                len(analysis["favorites"]), len(analysis["no_prediction"]), n_frozen)

    # historico: registra este card (se datado) e monta a aba com os eventos
    # passados (os resultados ja foram sincronizados acima)
    if card_name and event_date and not somente_render:
        # sinal sharp congelado junto (opcional: falha vira previsao sem o
        # extra, nunca derruba o registro)
        sharp_probs = {}
        if sharp:
            from src.line_shopping import fetch_sharp_probs
            sharp_probs = fetch_sharp_probs(
                analysis["favorites"] + analysis["underdogs"],
                event_date=event_date)
            n_ok = sum(1 for v in sharp_probs.values() if v is not None)
            logger.info("Sinal sharp obtido para %d de %d lutas previstas.",
                        n_ok, len(sharp_probs))
        record_card_predictions(analysis, card_name, event_date, sharp_probs=sharp_probs)
    elif not somente_render:
        logger.info("Sem --event-date: card nao registrado no historico de previsoes.")
    history_df = load_history()
    n_entradas = anotar_sinal_de_entrada(analysis, event_date) if event_date else 0
    if n_entradas:
        logger.info("Sinal de entrada (a casa paga acima do justo da sharp): "
                    "%d lado(s) marcado(s).", n_entradas)

    if photos:
        names = list(odds_df["fighter_a"]) + list(odds_df["fighter_b"])
        if not history_df.empty:
            names += list(history_df["fighter_a"]) + list(history_df["fighter_b"])
        if somente_render:
            # Redesenhar nao busca nada nem grava o cache: so o que ja esta
            # em data/raw/fighter_photos.json. Nome sem foto fica no monograma.
            from src.fighter_photos import _load_cache
            cache = _load_cache()
            set_photo_map({n: cache.get(str(n)) for n in names})
        else:
            from src.fighter_photos import get_photo_urls
            set_photo_map(get_photo_urls(names))
    else:
        set_photo_map({})
    history_panel = render_history_panel(history_df)

    html = render_html(analysis, gap_days, card_name=card_name,
                       history_panel_html=history_panel, event_date=event_date,
                       output_hint=output_path, csv_path=csv_path)
    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Relatorio salvo em %s", output_path.resolve())
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera relatorio HTML (favoritos/zebras) para um card de UFC com odds fornecidas manualmente.")
    parser.add_argument("csv", type=str, help="CSV com fighter_a,fighter_b,odds_a_decimal,odds_b_decimal")
    parser.add_argument("--output", type=str, default="card_report.html", help="Arquivo HTML de saida")
    parser.add_argument("--model", choices=["logreg", "gbm"], default="logreg",
                        help="Modelo calibrado a usar (default: logreg, melhor log loss)")
    parser.add_argument("--card-name", type=str, default="", help="Titulo do card (ex.: 'UFC 329')")
    parser.add_argument("--event-date", type=str, default="",
                        help="Data do evento (YYYY-MM-DD) para o historico de previsoes")
    parser.add_argument("--photos", action="store_true",
                        help="Fotos dos lutadores (UFC.com, hotlink) no relatorio LOCAL de uso "
                             "pessoal. Nao usar no relatorio publicado.")
    parser.add_argument("--somente-render", action="store_true",
                        help="So desenha a pagina a partir dos snapshots ja "
                             "gravados: nao sincroniza resultado, nao registra "
                             "previsao e nao consulta odds")
    parser.add_argument("--no-sharp", action="store_true",
                        help="Nao consultar a The Odds API para congelar o sinal sharp "
                             "(economiza creditos ao regerar um card ja registrado)")
    args = parser.parse_args()

    generate_card_report(args.csv, args.output, model_name=args.model, card_name=args.card_name,
                         event_date=args.event_date, photos=args.photos,
                         sharp=not args.no_sharp, somente_render=args.somente_render)
