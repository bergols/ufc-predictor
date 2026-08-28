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
import html as html_mod
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

import config
from src.prediction_history import (HISTORY_CSS, avatar_html,
                                    frozen_predictions_for_event, load_history,
                                    record_card_predictions, render_history_panel,
                                    set_photo_map, sync_results_from_template)
from src.utils import decimal_odds_to_implied_prob, probability_to_fair_odds, remove_vig_two_way

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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
        return {"model_prob_a": None if prob is None else 1.0 - prob,
                "method_probs": entry["method_probs"]}
    return None


def analyze_card(odds_df: pd.DataFrame, model_name: str = "logreg",
                 predict_fn: Optional[Callable[[str, str], dict]] = None,
                 method_fn: Optional[Callable[[str, str], dict]] = None,
                 frozen: Optional[dict] = None) -> dict:
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
    if predict_fn is None:
        predict_fn, default_method_fn = _default_predict_fns(model_name)
        if method_fn is None:
            method_fn = default_method_fn

    predicted, no_prediction = [], []
    for card_order, (_, row) in enumerate(odds_df.iterrows()):
        a, b = str(row["fighter_a"]).strip(), str(row["fighter_b"]).strip()
        odds_a, odds_b = float(row["odds_a_decimal"]), float(row["odds_b_decimal"])

        market_a, market_b = remove_vig_two_way(
            decimal_odds_to_implied_prob(odds_a), decimal_odds_to_implied_prob(odds_b))

        # posicao no CSV = posicao no card (main event primeiro). As abas
        # reordenam por probabilidade, entao sem isso o destaque do topo nao
        # teria como saber qual e a luta principal.
        base = {"fighter_a": a, "fighter_b": b, "odds_a": odds_a, "odds_b": odds_b,
                "market_prob_a": market_a, "market_prob_b": market_b,
                "card_order": card_order}

        try:
            pred = predict_fn(a, b)
        except ValueError as exc:
            no_prediction.append({**base, "reason": str(exc)})
            continue

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
        entry = _frozen_for_fight(frozen, a, b)
        is_closed = entry is not None

        # metodo: falha independente (nao derruba a previsao de vencedor)
        method_probs = None
        if is_closed:
            # Card ja encerrado. Se o metodo foi congelado no pre-registro,
            # exibe o congelado; se nao foi (eventos anteriores a ago/2026),
            # a luta fica SEM metodo -- um recalculo aqui olharia o proprio
            # resultado, e mostrar isso seria pior que nao mostrar nada.
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
            "model_prob_a": model_a, "model_prob_b": 1 - model_a,
            "frozen": frozen_a is not None,
            "low_experience": pred["fighter_a_low_experience"] or pred["fighter_b_low_experience"],
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
        fight["model_side_odds"] = odds_a if fight["model_side"] == a else odds_b
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
_SEG_TRACK = "#2B2B34"


def _delta_band(delta_side: float) -> tuple[str, int]:
    """Classe e espessura do vao. Extraido porque o BLOCO da luta tambem usa a
    classe: a aresta vermelha do card e a mesma leitura do marcador."""
    mag = abs(delta_side) * 100
    return next((s, t) for lim, s, t in _DELTA_BANDS if mag < lim)


def _corner(name: str, tag: str, model_side: bool, big: bool = False) -> str:
    """
    Uma linha do confronto: retrato, nome e etiqueta de mercado.

    Empilhada, nao frente a frente. O formato "tale of the tape" centralizado
    gastava ~200px por luta so em identificacao — dois blocos verticais mais
    uma linha "vs" — e empurrava a MEDICAO, que e o produto, para fora da
    primeira tela. Empilhados, os dois nomes dividem a margem esquerda com as
    barras de baixo, entao o olho le a luta inteira numa coluna so.

    O lado apontado pelo modelo vem marcado por um filete no acento, nao por
    etiqueta preenchida: a mesma informacao numa fracao da area de cor.
    """
    cls = "corner" + (" big" if big else "") + (" is-pick" if model_side else "")
    # sem `big=` no avatar de proposito: ele acrescentaria a classe `lg`, cuja
    # regra (.avatar.lg) tem especificidade maior que `.corner .avatar` e
    # venceria o --pic, devolvendo um retrato de 116px por cima do nome. O
    # tamanho aqui sai SO da variavel.
    return f"""
      <div class="{cls}">
        {avatar_html(name)}
        <span class="corner-name">{_e(name)}</span>
        <span class="corner-tag">{tag}</span>
      </div>"""


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
    if not notes:
        return ""
    return f'<div class="note">⚠ {"; ".join(notes)}</div>'


_METHOD_LABELS = [("KO_TKO", "KO/TKO"), ("SUBMISSION", "Finalização"), ("DECISION", "Decisão")]

# Icones SVG inline (sem dependencia externa), neutros -- usados so na secao
# de metodo, em cinza, para nao competir com os acentos dourado/vermelho.
_ICONS = {
    "KO_TKO": ('<svg class="mini-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" '
               'stroke-width="1.4"><path d="M8 1.5 9.6 5l3.7-1.7L11.5 7l3.4 1.9-3.8.9.9 3.8-3-2.4-2.4 3-.3-3.9-3.9.4 2.7-2.8L2 5.6l3.8-.3z"/></svg>'),
    "SUBMISSION": ('<svg class="mini-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" '
                   'stroke-width="1.4"><circle cx="5.5" cy="8" r="3.4"/><circle cx="10.5" cy="8" r="3.4"/></svg>'),
    "DECISION": ('<svg class="mini-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" '
                 'stroke-width="1.4"><rect x="3" y="2" width="10" height="12" rx="1.4"/>'
                 '<path d="M5.5 5.5h5M5.5 8h5M5.5 10.5h3"/></svg>'),
}


def _mini_dist(pairs: list[tuple[str, str]], probs: dict, icons: bool = False) -> str:
    rows = []
    for key, label in pairs:
        p = probs.get(key, 0.0) * 100
        icon = _ICONS.get(key, "") if icons else ""
        rows.append(f"""<div class="mini-row"><span class="mini-label">{icon}{label}</span>
          <div class="bar"><div class="fill neutral" style="width:{p:.1f}%"></div></div>
          <span class="prob-val">{p:.0f}%</span></div>""")
    return "".join(rows)


def _fair_odds_chip(p: float) -> str:
    """Odd justa: decimal como principal, americana como secundaria."""
    decimal, american = probability_to_fair_odds(p)
    return f'<span class="odds-chip">{decimal:.2f} <small>({american:+.0f})</small></span>'


def _odds_row(label: str, p: float, icon_key: str = "", strong: bool = False) -> str:
    icon = _ICONS.get(icon_key, "")
    cls = "mini-row strong" if strong else "mini-row"
    return f"""<div class="{cls}"><span class="mini-label">{icon}{label}</span>
      <div class="bar"><div class="fill neutral" style="width:{p * 100:.1f}%"></div></div>
      <span class="prob-val">{p * 100:.0f}%</span>{_fair_odds_chip(p)}</div>"""


def _method_card(fight: dict, rank: int) -> str:
    """Linha da aba 'Metodo de vitoria': odds justas das 3 categorias."""
    mp = fight["method_probs"]
    top = max(mp, key=mp.get)
    rows = "".join(_odds_row(label, mp[key], icon_key=key, strong=(key == top))
                   for key, label in _METHOD_LABELS)
    return f"""
    <article class="bout method">
      <div class="bout-head">
        <span class="rank">{rank:02d}</span>
        <span class="bout-names">{avatar_html(fight['fighter_a'], small=True)}{_e(fight['fighter_a'])}
          <i>vs</i>
          {avatar_html(fight['fighter_b'], small=True)}{_e(fight['fighter_b'])}</span>
      </div>
      <div class="method-grid">{rows}</div>
      {_matched_note(fight)}
    </article>"""


def _ev_card(fight: dict, rank: int) -> str:
    """Linha da aba 'Pernas EV>1': o lado do modelo com odd, EV e contexto."""
    side = fight["model_side"]
    opponent = fight["fighter_b"] if side == fight["fighter_a"] else fight["fighter_a"]
    market_side_prob = (fight["market_prob_fav"] if side == fight["favorite"]
                        else fight["market_prob_dog"])
    tipo = ("favorito do mercado" if side == fight["favorite"]
            else "azarão do mercado")
    edge = (fight["model_side_prob"] - market_side_prob) * 100
    return f"""
    <article class="bout ev">
      <div class="bout-head">
        <span class="rank">{rank:02d}</span>
        <span class="ev-value">{fight['ev']:.2f}<small>EV</small></span>
        <span class="bout-meta">{tipo} · odd {fight['model_side_odds']:.2f}</span>
      </div>
      <div class="ev-body">
        {avatar_html(side)}
        <div class="ev-id">
          <span class="corner-name">{_e(side)}</span>
          <span class="corner-tag">vs {_e(opponent)}</span>
        </div>
        <div class="ev-figs">
          <div class="fig"><span class="fig-n">{fight['model_side_prob'] * 100:.1f}<small>%</small></span>
            <span class="fig-k">modelo</span></div>
          <div class="fig"><span class="fig-n muted">{market_side_prob * 100:.1f}<small>%</small></span>
            <span class="fig-k">mercado</span></div>
          <div class="fig"><span class="fig-n {'pos' if edge > 0 else 'neg'}">{edge:+.1f}<small>pp</small></span>
            <span class="fig-k">diferença</span></div>
        </div>
      </div>
      {_matched_note(fight)}
    </article>"""


def _no_data_list(fights: list[dict], what: str) -> str:
    """Lutas com vencedor previsto mas sem dado para esta aba (falha independente)."""
    if not fights:
        return ""
    items = "\n".join(f"<li>{_e(f['fighter_a'])} vs {_e(f['fighter_b'])}</li>" for f in fights)
    return f"""
    <div class="no-pred">
      <h2>Sem previsão de {what} ({len(fights)})</h2>
      <p>Sem dados suficientes para {what} nestas lutas — a previsão de vencedor
      delas (abas Favoritos/Zebras) não é afetada.</p>
      <ul>{items}</ul>
    </div>"""


def _bout(fight: dict, rank: int, tab: str, hero: bool = False) -> str:
    """
    Confronto no formato "tale of the tape": os dois cantos frente a frente
    e, entre eles, as barras divergentes de modelo e mercado num eixo unico.
    O lado apontado pelo modelo (model_side) fica marcado. A categorizacao e
    mutuamente exclusiva: cada luta aparece em exatamente uma aba. Copy
    factual, sem linguagem de recomendacao.

    O canto ESQUERDO e sempre o fighter_a do CSV — a ordem da luta nao muda
    entre abas, so a marcacao de qual lado o modelo aponta.
    """
    a, b = fight["fighter_a"], fight["fighter_b"]
    a_is_pick = fight["model_side"] == a
    tag_a = "favorito" if fight["favorite"] == a else "azarão"
    tag_b = "favorito" if fight["favorite"] == b else "azarão"

    # So no hero. Dentro da aba o veredito e tautologico -- a aba Favoritos so
    # tem lutas em que o modelo concorda, e o criterio ja esta escrito no topo
    # dela --, entao repeti-lo em cada card e ruido que empurra a medicao para
    # baixo. A luta principal fica FORA das abas: la ele informa.
    if not hero:
        flag = ""
    elif tab == "favs":
        flag = '<span class="verdict agree">modelo concorda com o mercado</span>'
    else:
        flag = '<span class="verdict clash">modelo aponta o azarão</span>'

    model_a = fight["model_prob_a"]
    market_a = fight["market_prob_a"]
    market_side = (fight["market_prob_fav"] if fight["model_side"] == fight["favorite"]
                   else fight["market_prob_dog"])
    rank_html = "" if hero else f'<span class="rank">{rank:02d}</span>'
    delta_side = fight["model_side_prob"] - market_side
    # a aresta do card repete a classe do vao: percorrendo a pagina, o vermelho
    # marca onde o modelo discorda do mercado — que e a pergunta que o
    # relatorio existe para responder. Informacao, nao enfeite.
    state, _ = _delta_band(delta_side)
    return f"""
    <article class="bout{' hero' if hero else ''}" data-delta="{state}">
      <div class="bout-head">{rank_html}{flag}
        <span class="bout-meta">odds {fight['odds_a']:.2f} / {fight['odds_b']:.2f}</span></div>
      <div class="tape">
        {_corner(a, tag_a, a_is_pick, big=hero)}
        {_corner(b, tag_b, not a_is_pick, big=hero)}
      </div>
      <div class="splits">
        {_gap_bar(model_a, market_a, delta_side, a_is_pick)}
      </div>
      {_matched_note(fight)}
    </article>"""


_FAIR_ODDS_WARNING = """
    <p class="tab-explain warn-strong"><strong>Atenção — odds justas, não odds reais:</strong>
    estas são odds JUSTAS calculadas a partir da probabilidade do nosso modelo (sem margem de
    casa) — não são odds reais de mercado, pois não temos odds de casas de aposta para
    método/duração para comparar. Diferente do preditor de vencedor, este modelo nunca foi
    validado contra o mercado real — só contra um baseline ingênuo, com resultado modesto
    (ver README).</p>"""


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


def render_html(analysis: dict, freshness_gap_days: Optional[int], card_name: str = "",
                history_panel_html: str = "", event_date: str = "",
                output_hint: Optional[Path | str] = None) -> str:
    fav_cards = "\n".join(_bout(f, i + 1, "favs")
                          for i, f in enumerate(analysis["favorites"]))
    dog_cards = "\n".join(_bout(f, i + 1, "dogs")
                          for i, f in enumerate(analysis["underdogs"]))
    ev_cards = "\n".join(_ev_card(f, i + 1)
                         for i, f in enumerate(analysis.get("ev_legs", [])))
    method_cards = "\n".join(_method_card(f, i + 1)
                             for i, f in enumerate(analysis.get("method_ranking", [])))

    main = analysis.get("main_event")
    hero_html = ""
    if main:
        tab = "favs" if main["category"] == "favorite" else "dogs"
        hero_html = f"""
      <section class="hero-wrap" data-cat="{tab}">
        <div class="hero-label">Luta principal</div>
        {_bout(main, 0, tab, hero=True)}
      </section>"""

    no_pred_html = ""
    if analysis["no_prediction"]:
        items = "\n".join(
            f'<li><strong>{_e(f["fighter_a"])}</strong> vs <strong>{_e(f["fighter_b"])}</strong>'
            f' <span class="odds-line">(odds {f["odds_a"]:.2f} / {f["odds_b"]:.2f})</span>'
            f'<br><span class="note">{_e(f["reason"])}</span></li>'
            for f in analysis["no_prediction"])
        no_pred_html = f"""
        <section class="no-pred">
          <h2>Sem previsão ({len(analysis['no_prediction'])})</h2>
          <p>Lutas com lutador fora da base histórica (provável estreante no UFC) —
          o modelo não tem como estimar. Excluídas dos rankings acima.</p>
          <ul>{items}</ul>
        </section>"""

    if freshness_gap_days is None:
        fresh_html = ('<div class="status bad"><span class="dot"></span>'
                      'Não foi possível verificar o frescor da base de dados.</div>')
    elif freshness_gap_days > config.DATA_FRESHNESS_MAX_GAP_DAYS:
        fresh_html = (f'<div class="status bad"><span class="dot"></span>'
                      f'DADOS DESATUALIZADOS: o evento mais recente na base tem '
                      f'{freshness_gap_days} dias (limite: {config.DATA_FRESHNESS_MAX_GAP_DAYS}). '
                      f'As probabilidades do modelo não refletem lutas recentes.</div>')
    else:
        fresh_html = (f'<div class="status ok"><span class="dot"></span>'
                      f'Base de dados em dia — evento mais recente há '
                      f'{freshness_gap_days} dia(s).</div>')

    from src.design import (FONT_NOTICE, TAGLINE, WORDMARK, favicon_data_uri,
                            fonts_css, mark_svg, root_css)
    fonts_block = fonts_css()
    root_block = root_css()
    mark_icon = mark_svg()                  # cabeçalho, 24px
    mark_full = mark_svg("crest", full=True)  # brasão do evento, grande
    favicon = favicon_data_uri()

    title = _e(card_name) if card_name else "Card UFC"
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    pretty_date = _format_event_date(event_date)
    og_desc = _e(f"Probabilidade do modelo contra o mercado, luta a luta. "
                 f"{len(analysis['favorites']) + len(analysis['underdogs'])} confrontos "
                 f"analisados. Estimativa estatística — não é recomendação de aposta.")
    # a imagem só entra na tag se existir de fato: apontar para um 404 faz o
    # scraper mostrar preview quebrado, pior que preview sem imagem
    share = Path(str(output_hint)).parent / "share.png" if output_hint else None
    og_image = (f'\n<meta property="og:image" content="{_e(config.SITE_URL)}share.png">'
                f'\n<meta property="og:image:width" content="1200">'
                f'\n<meta property="og:image:height" content="630">'
                if share and share.exists() else "")
    event_kicker = f' <span class="kicker-date">{_e(pretty_date)}</span>' if pretty_date else ""

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — modelo vs. mercado</title>
<link rel="icon" href="{favicon}">
<meta name="theme-color" content="#000000">
<meta name="description" content="{og_desc}">
<meta property="og:type" content="website">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{og_desc}">
<meta property="og:url" content="{_e(config.SITE_URL)}">{og_image}
<meta name="twitter:card" content="summary_large_image">
<style>
  /* ================= sistema visual "broadcast" =================
     Regras que mantem isso parecendo grafico de transmissao e nao
     dashboard generico: cor CHAPADA (zero gradiente decorativo), canto
     reto (2px no maximo), regua de 1px em vez de caixa, tipografia
     condensada em caixa alta com tracking apertado, numero sempre
     tabular, e UM acento por contexto. Nada de sombra colorida, nada de
     hover que levanta elemento. */
{fonts_block}
  {root_block}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text); font-family: var(--font-body);
    font-size: 13px; line-height: 18px; padding: 0 0 72px; min-height: 100vh;
    -webkit-font-smoothing: antialiased;
    /* numero SEMPRE tabular: coluna de porcentagem tem de alinhar sozinha */
    font-variant-numeric: tabular-nums lining-nums;
    font-feature-settings: "tnum" 1, "lnum" 1;
  }}
  .wrap {{ max-width: 960px; margin: 0 auto; padding: 0 20px; }}
  h1, h2, h3 {{ font-family: var(--font-display); font-weight: 700;
    text-transform: uppercase; letter-spacing: .02em; }}
  /* chip com corte diagonal — reaproveitado em etiqueta, veredito e rank */
  .slash {{ display: inline-block; transform: skewX(var(--slash)); }}
  .slash > * {{ display: inline-block; transform: skewX(calc(var(--slash) * -1)); }}

  /* ---------- masthead: identidade propria do projeto ---------- */
  .masthead {{ background: var(--surface); border-bottom: 2px solid var(--brand); }}
  .masthead .wrap {{ display: flex; align-items: center; justify-content: space-between;
    gap: 16px; height: 58px; }}
  .brand {{ display: flex; align-items: center; gap: 11px; color: var(--brand); }}
  .mark {{ width: 24px; height: 24px; flex: none; }}
  .brand b {{ font-family: var(--font-display); font-weight: 700; font-style: italic;
    text-transform: uppercase; letter-spacing: .1em; font-size: .95rem; color: var(--text); }}
  .brand span {{ font-size: .66rem; color: var(--muted); letter-spacing: .12em;
    text-transform: uppercase; border-left: 1px solid var(--line); padding-left: 11px; }}
  .masthead .meta {{ font-family: var(--font-num); font-size: .66rem; color: var(--muted);
    letter-spacing: .04em; text-align: right; }}

  /* ---------- faixa do evento: cartaz ---------- */
  .event {{ padding: 34px 0 24px; border-bottom: 1px solid var(--line);
    display: flex; align-items: center; gap: 22px; }}
  .event-txt {{ min-width: 0; flex: 1; }}
  /* brasao completo: e aqui que ele tem espaco para o detalhe existir */
  .crest {{ width: 92px; height: 92px; flex: none; color: var(--text); }}
  .event .kicker {{ font-family: var(--font-display); text-transform: uppercase;
    letter-spacing: .24em; font-size: .66rem; color: var(--brand); margin-bottom: 12px;
    font-weight: 700; }}
  .kicker-date {{ color: var(--muted); border-left: 1px solid var(--line);
    margin-left: 8px; padding-left: 12px; }}
  /* título do evento: elemento único da página, então aguenta mais peso que a
     escala de lista. Piso de 30px no celular, teto de 46px no desktop. */
  .event h1 {{ font-size: clamp(30px, 5.2vw, 46px); line-height: .96;
    letter-spacing: -.025em; font-style: italic; font-weight: 800; }}
  .event .sub {{ color: var(--muted); margin-top: 14px; font-size: .76rem;
    letter-spacing: .08em; text-transform: uppercase; }}

  /* ---------- avisos: regua lateral, sem caixa arredondada ---------- */
  .notice {{ font-size: .78rem; line-height: 1.55; color: var(--muted);
    border-left: 2px solid var(--line); padding: 2px 0 2px 14px; margin: 18px 0; }}
  .notice strong {{ color: var(--dim); font-weight: 600; }}
  .notice.alert {{ border-left-color: var(--red); }}
  .status {{ display: flex; align-items: center; gap: 10px; font-size: .74rem;
    font-family: var(--font-num); letter-spacing: .02em; padding: 10px 0;
    border-top: 1px solid var(--line-soft); border-bottom: 1px solid var(--line-soft);
    margin-bottom: 26px; }}
  .status .dot {{ width: 6px; height: 6px; flex: none; }}
  .status.ok {{ color: var(--muted); }}
  .status.ok .dot {{ background: var(--green); }}
  .status.bad {{ color: #E8909A; }}
  .status.bad .dot {{ background: var(--red); }}

  /* ---------- abas ---------- */
  /* uma linha so, com rolagem quando nao cabe (padrao de aba em tela
     estreita). Quebrar em varias linhas deixava a regua de baixo ragged.
     A mascara na direita sinaliza que ha mais conteudo. */
  .tabs {{ display: flex; gap: 20px; border-bottom: 1px solid var(--line);
    margin-bottom: 22px; overflow-x: auto; scrollbar-width: none;
    -webkit-mask-image: linear-gradient(90deg, #000 calc(100% - 28px), transparent);
    mask-image: linear-gradient(90deg, #000 calc(100% - 28px), transparent); }}
  .tabs::-webkit-scrollbar {{ display: none; }}
  .tab-btn {{
    background: none; color: var(--muted); border: none; border-bottom: 2px solid transparent;
    padding: 0 0 12px; margin-bottom: -1px; cursor: pointer; white-space: nowrap;
    font-family: var(--font-display); font-weight: 700; text-transform: uppercase;
    letter-spacing: .07em; font-size: .78rem; transition: color .15s;
  }}
  .tab-btn {{ font-style: italic; }}
  .tab-btn:hover {{ color: var(--dim); }}
  .tab-btn.active {{ color: var(--text); border-bottom-color: var(--accent); }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}
  /* um acento por aba: tudo que e "cor" na aba herda daqui */
  #favs {{ --accent: var(--gold); }}
  #dogs {{ --accent: var(--red); }}
  #ev {{ --accent: var(--green); }}
  #method {{ --accent: var(--steel); }}
  #history {{ --accent: var(--dim); }}

  .tab-explain {{ color: var(--muted); font-size: .78rem; line-height: 1.6;
    border-left: 2px solid var(--line); padding: 2px 0 2px 14px; margin-bottom: 22px; }}
  .tab-explain strong {{ color: var(--dim); }}
  .tab-explain.warn-strong {{ border-left-color: var(--red); }}

  /* ---------- confronto ---------- */
  /* aresta esquerda transparente em TODA luta e so colorida nas divergentes:
     colorir sem reservar o espaco faria o conteudo saltar de card para card. */
  .bout {{ border-top: 1px solid var(--line); padding: 18px 0 20px 14px;
    border-left: 3px solid transparent; }}
  .bout[data-delta="material"] {{ border-left-color: rgba(210,10,17,.55); }}
  .bout[data-delta="large"] {{ border-left-color: var(--red); }}
  .bout:last-of-type {{ border-bottom: 1px solid var(--line); }}
  .bout-head {{ display: flex; align-items: baseline; gap: 12px; margin-bottom: 16px;
    flex-wrap: wrap; }}
  /* agora que o veredito saiu do cabecalho, o numero e o unico ancoradouro da
     ordem: a #3A3A44 ele sumia no preto e a lista perdia a contagem */
  .rank {{ font-family: var(--font-display); font-style: italic; font-weight: 800;
    font-size: 19px; color: var(--muted); letter-spacing: -.02em; }}
  /* o acento e caro: so a etiqueta do lado apontado e a barra ficam com ele.
     O veredito vem em cinza para nao competir. */
  .verdict {{ font-family: var(--font-display); text-transform: uppercase;
    letter-spacing: .1em; font-size: .67rem; font-weight: 700; color: var(--muted);
    font-style: italic; }}
  .verdict.clash::before {{ content: ""; display: inline-block; width: 4px; height: 11px;
    background: var(--red); margin-right: 8px; vertical-align: -1px;
    transform: skewX(var(--slash)); }}
  .bout-meta {{ margin-left: auto; font-family: var(--font-num); font-size: .7rem;
    color: var(--muted); }}

  /* confronto: duas linhas empilhadas que dividem a margem esquerda com as
     barras, para a luta inteira ser lida numa coluna so. */
  .tape {{ display: flex; flex-direction: column; gap: 1px; margin-bottom: 13px; }}
  /* a etiqueta de mercado segue o NOME, nao a borda direita: empurrada para a
     extrema direita ela deixava ~700px de vazio no meio em tela larga, e a
     linha parecia quebrada em vez de composta. */
  /* --pic e a UNICA fonte do tamanho do retrato: a coluna e a imagem saem
     dela. Definidos separadamente, os dois divergiram na media query e o
     retrato do hero transbordou por cima do nome. */
  .corner {{ --pic: 34px;
    display: grid; grid-template-columns: var(--pic) minmax(0, auto) 1fr;
    align-items: baseline; gap: 11px; padding: 4px 0 4px 9px;
    border-left: 2px solid transparent; }}
  .corner.big {{ --pic: 46px; }}
  .corner .avatar {{ width: var(--pic); height: var(--pic); align-self: center; }}
  .corner-tag {{ justify-self: start; }}
  /* filete no lugar da etiqueta preenchida: o lado apontado pelo modelo
     continua obvio, com uma fracao da area de cor que a etiqueta gastava. */
  .corner.is-pick {{ border-left-color: var(--accent); }}
  .corner-name {{ font-family: var(--font-display); font-weight: 800;
    font-style: italic; text-transform: uppercase; letter-spacing: -.01em;
    font-size: 17px; line-height: 21px; color: var(--dim);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .corner.is-pick .corner-name {{ color: var(--text); }}
  .corner-tag {{ font-size: 10px; line-height: 12px; color: var(--muted);
    text-transform: uppercase; letter-spacing: .09em; font-weight: 600; }}
  .corner.big .corner-name {{ font-size: 23px; line-height: 27px; }}

  /* retrato: monograma por baixo, foto por cima quando existe */
  .avatar {{ width: 58px; height: 58px; flex: none; position: relative; overflow: hidden;
    border-radius: 2px; background: var(--mono, var(--surface2));
    display: inline-flex; align-items: center; justify-content: center;
    font-family: var(--font-display); font-weight: 700; font-size: 1.05rem;
    letter-spacing: .04em; color: rgba(255,255,255,.82); text-transform: uppercase; }}
  /* og:image do UFC e 520x325 com o lutador de corpo inteiro ao centro:
     sem zoom, o rosto fica minusculo no recorte quadrado. 1.35x ancorado
     no topo enquadra cabeca e tronco. */
  .avatar img {{ position: absolute; width: 135%; height: 135%; left: -17.5%; top: -2%;
    object-fit: cover; object-position: center top; }}
  .avatar.has-photo .avatar-txt {{ visibility: hidden; }}
  .avatar.sm {{ width: 22px; height: 22px; font-size: .55rem; vertical-align: -5px;
    margin-right: 7px; }}
  .avatar.lg {{ width: 116px; height: 116px; }}
  /* na lista de confrontos o retrato e identificacao, nao imagem: redondo e
     pequeno, encostado no nome. O quadrado de 58px pedia peso de foto sem
     entregar — recorte e qualidade variam demais entre lutadores. */
  .corner .avatar {{ border-radius: 50%; }}

  /* EIXO UNICO da luta: a barra enche ate a estimativa do modelo, o tique
     marca a do mercado, e o vao vermelho entre os dois E o delta. Antes eram
     tres linhas repetindo o mesmo eixo de 0 a 100 -- tres vezes a altura para
     uma comparacao que o leitor tinha de montar na cabeca. */
  .splits {{ display: flex; flex-direction: column; gap: 7px; }}
  .gap {{ display: grid; grid-template-columns: 44px 1fr 44px; align-items: center;
    gap: 12px; }}
  .gap-track {{ display: flex; height: 14px; background: var(--surface2);
    position: relative; overflow: visible; }}
  .seg {{ height: 100%; }}
  .seg.l, .seg.r {{ background: {_SEG_TRACK}; }}
  /* acento CHEIO: agora ha uma barra por luta em vez de duas, entao a area de
     ouro ja caiu pela metade sem precisar diluir a cor. Diluir apagava a
     distincao entre as leituras, que e o que a barra existe para mostrar. */
  .seg.side {{ background: var(--accent); }}
  /* tique do mercado: contorno claro sobre a barra, nao uma segunda barra.
     Fino e alto para cruzar o eixo inteiro e ser lido como REFERENCIA. */
  .gap-mark {{ position: absolute; top: -4px; bottom: -4px; width: 2px;
    background: var(--text); transform: translateX(-1px); }}
  .gap-mark::after {{ content: ""; position: absolute; left: -2px; right: -2px;
    top: -1px; height: 3px; background: var(--text); }}
  /* o vao vai ABAIXO do eixo, como colchete de medida, nao em cima da barra.
     Centralizado ele cai dentro do ouro sempre que o modelo esta a frente do
     mercado, e vermelho sobre ouro le como sujeira em vez de medida. */
  .gap-span {{ position: absolute; top: calc(100% + 3px); background: var(--red); }}
  .gap-num {{ font-family: var(--font-num); font-size: 13px; line-height: 16px;
    font-weight: 600; color: var(--muted); font-variant-numeric: tabular-nums; }}
  .gap-num.l {{ text-align: right; }}
  .gap-num.side {{ color: var(--text); }}

  /* legenda: o que a geometria acima nao diz sozinha */
  .gap-legend {{ display: flex; align-items: baseline; gap: 16px;
    padding-left: 56px; font-family: var(--font-num); font-size: 11px;
    color: var(--muted); }}
  .gap-legend > span:first-child {{ text-transform: uppercase;
    letter-spacing: .09em; font-size: 9px; }}
  .gap-legend > span:first-child::before {{ content: ""; display: inline-block;
    width: 9px; height: 9px; background: var(--accent); margin-right: 6px;
    vertical-align: -1px; }}
  .gap-market::before {{ content: ""; display: inline-block; width: 2px;
    height: 10px; background: var(--text); margin-right: 6px;
    vertical-align: -1px; }}
  /* o vao e a conclusao da luta: e o unico numero que ganha peso */
  .gap-delta {{ margin-left: auto; font-size: 14px; font-weight: 700;
    color: var(--red); }}
  .gap-delta small {{ font-size: 9px; color: var(--muted); margin-left: 2px;
    letter-spacing: .06em; }}
  .gap[data-state="agreement"] + .gap-legend .gap-delta {{ color: var(--muted); }}

  /* destaque da luta principal */
  .hero-wrap {{ margin-bottom: 30px; }}
  .hero-label {{ font-family: var(--font-display); font-style: italic; font-weight: 700;
    text-transform: uppercase; letter-spacing: .22em; font-size: .64rem;
    color: #fff; background: var(--brand); display: inline-block;
    padding: 4px 14px 4px 11px; transform: skewX(var(--slash)); margin-bottom: 8px; }}
  .bout.hero {{ border-top: 3px solid var(--brand); border-bottom: 1px solid var(--line);
    background: var(--surface); padding: 22px 22px 24px; }}
  /* só o destaque passa de 16px, e para 22 — acima disso a área de dados
     começa a parecer cartaz e a lista perde ritmo */
  .bout.hero .corner-name {{ font-size: 22px; line-height: 22px; }}
  .bout.hero .tape {{ margin-bottom: 22px; }}

  .note {{ color: #B99B45; font-size: .72rem; margin-top: 12px; line-height: 1.5; }}

  /* ---------- aba de metodo ---------- */
  .bout.method .bout-head, .bout.ev .bout-head {{ margin-bottom: 12px; }}
  .bout-names {{ font-family: var(--font-display); font-weight: 700; text-transform: uppercase;
    letter-spacing: .01em; font-size: .98rem; display: flex; align-items: center;
    flex-wrap: wrap; }}
  .bout-names i {{ font-style: italic; color: var(--muted); font-size: .7rem;
    margin: 0 9px; font-weight: 400; }}
  .method-grid {{ display: flex; flex-direction: column; gap: 6px; }}
  .mini-row {{ display: grid; grid-template-columns: 118px 1fr 42px 96px; align-items: center;
    gap: 12px; }}
  .mini-label {{ color: var(--muted); font-size: .72rem; text-transform: uppercase;
    letter-spacing: .06em; display: inline-flex; align-items: center; gap: 7px; }}
  .mini-icon {{ width: 12px; height: 12px; color: var(--muted); flex: none; }}
  .bar {{ height: 8px; background: var(--surface2); }}
  .fill {{ height: 100%; background: #35353F; }}
  .mini-row.strong .fill {{ background: var(--accent); }}
  .mini-row.strong .mini-label {{ color: var(--dim); }}
  .prob-val {{ font-family: var(--font-num); font-size: .72rem; text-align: right;
    color: var(--muted); font-variant-numeric: tabular-nums; }}
  .mini-row.strong .prob-val {{ color: var(--text); }}
  .odds-chip {{ font-family: var(--font-num); font-size: .74rem; color: var(--dim);
    border: 1px solid var(--line); padding: 2px 0; text-align: center;
    white-space: nowrap; font-variant-numeric: tabular-nums; }}
  .odds-chip small {{ color: var(--muted); }}
  .mini-row.strong .odds-chip {{ border-color: var(--accent); color: var(--text); }}

  /* ---------- aba EV ---------- */
  /* número principal: 26px. É o protagonista da aba, mas não é manchete. */
  .ev-value {{ font-family: var(--font-num); font-size: 26px; line-height: 26px;
    font-weight: 600; color: var(--accent); }}
  .ev-value small {{ font-size: 10px; letter-spacing: .1em; color: var(--muted);
    margin-left: 6px; text-transform: uppercase; font-weight: 600; }}
  .ev-body {{ display: flex; align-items: center; gap: 14px; }}
  .ev-id {{ min-width: 0; flex: 1; }}
  .ev-figs {{ display: flex; gap: 26px; }}
  .fig {{ text-align: right; }}
  /* número secundário: 16px */
  .fig-n {{ display: block; font-family: var(--font-num); font-size: 16px;
    line-height: 18px; font-weight: 600; color: var(--text); }}
  .fig-n small {{ font-size: 10px; color: var(--muted); margin-left: 2px; }}
  .fig-n.muted {{ color: var(--muted); }}
  .fig-n.pos {{ color: var(--green); }}
  .fig-n.neg {{ color: var(--red); }}
  .fig-k {{ display: block; font-size: 10px; line-height: 12px; color: var(--muted);
    text-transform: uppercase; letter-spacing: .09em; font-weight: 600; margin-top: 4px; }}

  /* ---------- aba historico (eventos passados) ---------- */
{HISTORY_CSS}

  /* ---------- sem previsao: discreto de proposito ---------- */
  .no-pred {{ margin-top: 34px; border-top: 1px solid var(--line); padding-top: 18px; }}
  .no-pred h2 {{ font-size: .8rem; letter-spacing: .1em; color: var(--muted);
    margin-bottom: 8px; }}
  .no-pred p {{ color: var(--muted); font-size: .76rem; margin-bottom: 12px; max-width: 60ch; }}
  .no-pred ul {{ list-style: none; }}
  .no-pred li {{ padding: 8px 0; border-bottom: 1px solid var(--line-soft);
    font-size: .8rem; color: var(--muted); }}
  .no-pred li strong {{ color: var(--dim); font-weight: 600; }}

  footer {{ color: var(--muted); font-size: 11px; line-height: 17px; margin-top: 40px;
    border-top: 1px solid var(--line); padding-top: 16px; max-width: 76ch; }}
  footer strong {{ color: var(--dim); font-weight: 600; }}
  /* crédito das fontes (exigência da OFL) — presente, discreto */
  .credits {{ display: block; margin-top: 12px; color: #4E4E58; font-size: 10px;
    line-height: 15px; }}

  /* ---------- responsivo ---------- */
  @media (max-width: 720px) {{
    .wrap {{ padding: 0 14px; }}
    .event {{ padding: 24px 0 18px; gap: 14px; }}
    .crest {{ width: 62px; height: 62px; }}
    .ev-body {{ flex-wrap: wrap; }}
    .ev-figs {{ width: 100%; justify-content: space-between; gap: 12px; }}
    .fig {{ text-align: left; }}
    .mini-row {{ grid-template-columns: 92px 1fr 38px; }}
    .mini-row .odds-chip {{ grid-column: 2 / -1; margin-top: -2px; }}
  }}
  @media (max-width: 560px) {{
    /* os cantos ja empilham por padrao; aqui so o retrato encolhe e o nome
       cede tamanho para caber sem reticencias em nome longo */
    .corner {{ --pic: 28px; gap: 9px; }}
    .corner.big {{ --pic: 36px; }}
    .corner-name {{ font-size: 15px; line-height: 19px; }}
    .corner.big .corner-name {{ font-size: 19px; line-height: 23px; }}
    .gap {{ grid-template-columns: 38px 1fr 38px; gap: 9px; }}
    .gap-legend {{ padding-left: 47px; gap: 11px; }}
    .masthead .meta {{ display: none; }}
  }}
</style>
</head>
<body>
<div class="masthead">
  <div class="wrap">
    <div class="brand">{mark_icon}<b>{_e(WORDMARK)}</b><span>{_e(TAGLINE)}</span></div>
    <div class="meta">gerado {generated}<br>modelo {_e(analysis['model_name'])} · calibrado</div>
  </div>
</div>

<div class="wrap">
  <section class="event">
    {mark_full}
    <div class="event-txt">
      <div class="kicker">Card analisado{event_kicker}</div>
      <h1>{title}</h1>
      <div class="sub">Probabilidade do modelo contra o mercado · odds com vig removido</div>
    </div>
  </section>

  <div class="notice alert">
    <strong>Aviso:</strong> estimativa estatística baseada em dados históricos,
    <strong>não é recomendação de aposta</strong>. MMA tem alta variância — zebras acontecem e
    favoritos caem. O próprio modelo perde para o mercado em backtest (ver README).
    Confira a defasagem dos dados (<code>check_data_freshness</code>) antes de usar.
  </div>
  {fresh_html}

  {hero_html}

  <div class="tabs">
    <button class="tab-btn active" data-tab="favs">Favoritos mais seguros</button>
    <button class="tab-btn" data-tab="dogs">Melhores zebras</button>
    <button class="tab-btn" data-tab="ev">Pernas EV&gt;1</button>
    <button class="tab-btn" data-tab="method">Método de vitória</button>
    <button class="tab-btn" data-tab="history">Histórico</button>
  </div>

  <div id="favs" class="tab-panel active">
    <p class="tab-explain"><strong>Critério:</strong> lutas em que o lado mais provável segundo
    o modelo coincide com o favorito do mercado. Ordenação: probabilidade do modelo para esse
    lado, decrescente. Cada luta do card aparece em exatamente uma das duas abas.</p>
    {fav_cards if fav_cards else '<p class="note">Nenhuma luta nesta categoria (modelo e mercado não coincidem em nenhum confronto).</p>'}
  </div>
  <div id="dogs" class="tab-panel">
    <p class="tab-explain"><strong>Critério:</strong> lutas em que o modelo aponta o azarão do
    mercado como o lado mais provável de vencer — divergência direta de leitura, não apenas
    "azarão competitivo". Ordenação: probabilidade do modelo para esse lado, decrescente.
    Contexto necessário: em backtest o modelo <strong>não</strong> supera o mercado; trate a
    divergência como hipótese estatística, não como erro do mercado.</p>
    {dog_cards if dog_cards else '<p class="note">Nenhuma luta nesta categoria (o modelo concorda com o favorito do mercado em todos os confrontos).</p>'}
  </div>
  <div id="ev" class="tab-panel">
    <p class="tab-explain warn-strong"><strong>Leia antes de usar:</strong> EV = probabilidade do
    modelo × odd do lado que ele aponta. É <strong>auto-referente</strong> — assume que o modelo
    está certo, e o backtest mostra o contrário: o mercado está na frente em todas as métricas,
    justamente nas divergências (onde o "valor" aparece). Este é o critério de
    <strong>pré-registro do paper trading</strong> (simulação de 1 unidade por perna, placar na
    aba Histórico), não uma recomendação de aposta.</p>
    <p class="tab-explain"><strong>Critério:</strong> lutas em que p_modelo × odd &gt; 1 para o
    lado apontado pelo modelo, ordenadas por EV decrescente. Barras: probabilidade do modelo vs
    probabilidade de mercado (devig) para o mesmo lado.</p>
    {ev_cards if ev_cards else '<p class="note">Nenhuma perna com EV &gt; 1 neste card.</p>'}
  </div>
  <div id="method" class="tab-panel">
    {_FAIR_ODDS_WARNING}
    <p class="tab-explain"><strong>Critério:</strong> distribuição prevista de método de vitória
    (KO/TKO, finalização, decisão) com a odd justa de cada categoria. Ordenação: probabilidade
    da categoria mais provável de cada luta, decrescente.</p>
    {method_cards if method_cards else '<p class="note">Nenhuma luta com previsão de método.</p>'}
    {_no_data_list(analysis.get('no_method', []), 'método')}
  </div>
  <div id="history" class="tab-panel">
    <p class="tab-explain"><strong>Como ler:</strong> cada evento passado mostra o lado que o
    modelo apontou (com a probabilidade <em>congelada no momento da publicação</em>, antes do
    evento — re-treinos posteriores não reescrevem previsões), o favorito do mercado (devig) e o
    vencedor real, com ✓/✗ para cada um. O placar do cabeçalho conta só lutas com previsão e
    resultado. Amostras pequenas são ruído: o placar de um evento isolado não prova nada.</p>
    {history_panel_html if history_panel_html else '<p class="note">Nenhum evento registrado ainda.</p>'}
  </div>

  {no_pred_html}

  <footer>
    Relatório estático, <strong>não atualizado em tempo real</strong> — gerado em {generated},
    publicado manualmente por evento, sem backend. ·
    modelo: {_e(analysis['model_name'])} (calibrado) ·
    odds fornecidas manualmente pelo usuário ·
    categorias mutuamente exclusivas: "zebra" = o modelo aponta o azarão do mercado
    como lado mais provável de vencer.
    <span class="credits">{_e(FONT_NOTICE)}</span>
  </footer>
</div>
<script>
  document.querySelectorAll('.tab-btn').forEach(function (btn) {{
    btn.addEventListener('click', function () {{
      document.querySelectorAll('.tab-btn').forEach(function (b) {{ b.classList.remove('active'); }});
      document.querySelectorAll('.tab-panel').forEach(function (p) {{ p.classList.remove('active'); }});
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
    }});
  }});
</script>
</body>
</html>"""


def generate_card_report(csv_path: Path | str, output_path: Path | str,
                         model_name: str = "logreg", card_name: str = "",
                         event_date: str = "", photos: bool = False,
                         sharp: bool = True) -> Path:
    """
    `event_date` (YYYY-MM-DD): data do evento, usada para registrar as
    previsoes no historico de paper trading (congeladas ate o resultado) e
    para casar os resultados vindos do odds_template.csv. Sem ela o
    relatorio e gerado normalmente, mas o card nao entra no historico.

    `photos`: busca fotos dos lutadores no UFC.com (hotlink + cache local;
    ver src/fighter_photos.py). Usado tambem na pagina publicada desde
    ago/2026. O HTML deixa de ser offline/self-contained; foto que nao
    carrega cai no monograma, entao o layout nunca quebra.

    `sharp`: consulta a The Odds API para congelar, junto de cada previsao,
    a probabilidade devigada da casa sharp (Pinnacle). Custa creditos da
    API; desligue com sharp=False em testes/regeracoes.
    """
    from src.data_collection import check_data_freshness

    odds_df = load_card_odds(csv_path)
    logger.info("Card carregado: %d lutas de %s", len(odds_df), csv_path)

    gap_days = check_data_freshness()

    # ANTES de analisar: fecha o que ja tem resultado no odds_template. A ordem
    # importa duas vezes. (1) So depois do sync as lutas encerradas aparecem em
    # frozen_predictions_for_event, e sem isso o relatorio exibiria o recalculo
    # pos-evento. (2) record_card_predictions abaixo so respeita linha fechada;
    # com o sync depois dele, regerar um card ja resolvido reescrevia o
    # pre-registro com a previsao contaminada e SO ENTAO congelava — a previsao
    # publicada se perdia sem aviso.
    sync_results_from_template()
    frozen = frozen_predictions_for_event(event_date) if event_date else None

    analysis = analyze_card(odds_df, model_name=model_name, frozen=frozen)

    n_frozen = sum(1 for f in analysis["favorites"] + analysis["underdogs"] if f["frozen"])
    logger.info("Previstas: %d | sem previsao: %d | congeladas (ja encerradas): %d",
                len(analysis["favorites"]), len(analysis["no_prediction"]), n_frozen)

    # historico: registra este card (se datado) e monta a aba com os eventos
    # passados (os resultados ja foram sincronizados acima)
    if card_name and event_date:
        # sinal sharp congelado junto (opcional: falha vira previsao sem o
        # extra, nunca derruba o registro)
        sharp_probs = {}
        if sharp:
            from src.line_shopping import fetch_sharp_probs
            sharp_probs = fetch_sharp_probs(analysis["favorites"] + analysis["underdogs"])
            n_ok = sum(1 for v in sharp_probs.values() if v is not None)
            logger.info("Sinal sharp obtido para %d de %d lutas previstas.",
                        n_ok, len(sharp_probs))
        record_card_predictions(analysis, card_name, event_date, sharp_probs=sharp_probs)
    else:
        logger.info("Sem --event-date: card nao registrado no historico de previsoes.")
    history_df = load_history()

    if photos:
        from src.fighter_photos import get_photo_urls
        names = list(odds_df["fighter_a"]) + list(odds_df["fighter_b"])
        if not history_df.empty:
            names += list(history_df["fighter_a"]) + list(history_df["fighter_b"])
        set_photo_map(get_photo_urls(names))
    else:
        set_photo_map({})
    history_panel = render_history_panel(history_df)

    html = render_html(analysis, gap_days, card_name=card_name,
                       history_panel_html=history_panel, event_date=event_date,
                       output_hint=output_path)
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
    parser.add_argument("--no-sharp", action="store_true",
                        help="Nao consultar a The Odds API para congelar o sinal sharp "
                             "(economiza creditos ao regerar um card ja registrado)")
    args = parser.parse_args()

    generate_card_report(args.csv, args.output, model_name=args.model, card_name=args.card_name,
                         event_date=args.event_date, photos=args.photos, sharp=not args.no_sharp)
