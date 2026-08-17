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


def _split_bar(key: str, left_p: float, right_p: float, left_is_model_side: bool) -> str:
    """
    Barra divergente de 100%: os dois lados da MESMA luta dividindo um eixo
    unico, encontrando-se onde a probabilidade manda. Le-se de relance quem
    esta na frente e por quanto — melhor que duas barras separadas, em que o
    leitor tem de comparar comprimentos que nao compartilham eixo.
    """
    lcls = "seg l" + (" side" if left_is_model_side else "")
    rcls = "seg r" + ("" if left_is_model_side else " side")
    lnum = "split-num l" + (" side" if left_is_model_side else "")
    rnum = "split-num r" + ("" if left_is_model_side else " side")
    return f"""
      <div class="split-row">
        <span class="split-key">{key}</span>
        <span class="{lnum}">{left_p * 100:.1f}</span>
        <div class="split-bar">
          <div class="{lcls}" style="width:{left_p * 100:.1f}%"></div>
          <div class="{rcls}" style="width:{right_p * 100:.1f}%"></div>
          <span class="split-mid"></span>
        </div>
        <span class="{rnum}">{right_p * 100:.1f}</span>
      </div>"""


def _corner(name: str, tag: str, model_side: bool, align: str, big: bool = False) -> str:
    """Um canto do confronto: foto/monograma + nome + etiqueta."""
    tag_html = f'<span class="corner-tag">{tag}</span>'
    pick = '<span class="pick-flag">lado do modelo</span>' if model_side else ""
    cls = "corner " + align + (" big" if big else "") + (" is-pick" if model_side else "")
    return f"""
      <div class="{cls}">
        {avatar_html(name, big=big)}
        <div class="corner-id">
          <span class="corner-name">{_e(name)}</span>
          {tag_html}{pick}
        </div>
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

    p = fight["model_side_prob"] * 100
    if tab == "favs":
        flag = f'<span class="verdict agree">modelo concorda · {p:.1f}%</span>'
    else:
        flag = f'<span class="verdict clash">divergência · {p:.1f}% ao azarão</span>'

    model_a = fight["model_prob_a"]
    market_a = fight["market_prob_a"]
    rank_html = "" if hero else f'<span class="rank">{rank:02d}</span>'
    return f"""
    <article class="bout{' hero' if hero else ''}">
      <div class="bout-head">{rank_html}{flag}
        <span class="bout-meta">odds {fight['odds_a']:.2f} / {fight['odds_b']:.2f}</span></div>
      <div class="tape">
        {_corner(a, tag_a, a_is_pick, "l", big=hero)}
        <span class="tape-vs">vs</span>
        {_corner(b, tag_b, not a_is_pick, "r", big=hero)}
      </div>
      <div class="splits">
        {_split_bar("modelo", model_a, 1 - model_a, a_is_pick)}
        {_split_bar("mercado", market_a, 1 - market_a, a_is_pick)}
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


_MARK = ('<svg class="mark" viewBox="0 0 34 34" aria-hidden="true">'
         '<rect x="1" y="1" width="32" height="32" rx="2" fill="none" '
         'stroke="currentColor" stroke-width="2"/>'
         '<path d="M9 9 L15.5 17 L9 25" fill="none" stroke="currentColor" '
         'stroke-width="3" stroke-linecap="square"/>'
         '<path d="M25 9 L18.5 17 L25 25" fill="none" stroke="currentColor" '
         'stroke-width="3" stroke-linecap="square"/></svg>')


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
                history_panel_html: str = "", event_date: str = "") -> str:
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

    title = _e(card_name) if card_name else "Card UFC"
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    pretty_date = _format_event_date(event_date)
    event_kicker = f' <span class="kicker-date">{_e(pretty_date)}</span>' if pretty_date else ""

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — modelo vs. mercado</title>
<style>
  /* ================= sistema visual "broadcast" =================
     Regras que mantem isso parecendo grafico de transmissao e nao
     dashboard generico: cor CHAPADA (zero gradiente decorativo), canto
     reto (2px no maximo), regua de 1px em vez de caixa, tipografia
     condensada em caixa alta com tracking apertado, numero sempre
     tabular, e UM acento por contexto. Nada de sombra colorida, nada de
     hover que levanta elemento. */
  :root {{
    /* preto puro + vermelho de sangue: a paleta de card de luta. O vermelho e
       a cor da CASA (cabecalho, regua do hero, base das abas); as abas
       mantem cada uma seu acento semantico por cima disso. */
    --bg: #000000; --surface: #0B0B0D; --surface2: #141417; --line: #232329;
    --line-soft: #17171A;
    --text: #FFFFFF; --dim: #B8B8C0; --muted: #79798A;
    --gold: #D6AF37; --red: #D20A11; --green: #2FA36B; --steel: #7E9BD4;
    --brand: var(--red);
    --font-display: "Archivo Narrow", "Roboto Condensed", "Arial Narrow",
                    "Helvetica Neue", Arial, sans-serif;
    --font-body: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    --font-num: "SF Mono", "Consolas", "Roboto Mono", ui-monospace, monospace;
    /* corte diagonal: a forma que da a energia agressiva do pacote grafico */
    --slash: -9deg;
    --accent: var(--gold);
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text); font-family: var(--font-body);
    line-height: 1.5; padding: 0 0 72px; min-height: 100vh;
    -webkit-font-smoothing: antialiased;
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
  .event {{ padding: 34px 0 24px; border-bottom: 1px solid var(--line); position: relative; }}
  /* barra vermelha diagonal atras do titulo: a assinatura do cartaz de luta */
  .event::before {{ content: ""; position: absolute; left: -40px; top: 30px;
    width: 88px; height: 5px; background: var(--brand);
    transform: skewX(var(--slash)); }}
  .event .kicker {{ font-family: var(--font-display); text-transform: uppercase;
    letter-spacing: .24em; font-size: .66rem; color: var(--brand); margin-bottom: 12px;
    font-weight: 700; }}
  .kicker-date {{ color: var(--muted); border-left: 1px solid var(--line);
    margin-left: 8px; padding-left: 12px; }}
  .event h1 {{ font-size: clamp(2rem, 6vw, 3.9rem); line-height: .92;
    letter-spacing: -.015em; font-style: italic; }}
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
  .bout {{ border-top: 1px solid var(--line); padding: 20px 0 22px; }}
  .bout:last-of-type {{ border-bottom: 1px solid var(--line); }}
  .bout-head {{ display: flex; align-items: baseline; gap: 12px; margin-bottom: 16px;
    flex-wrap: wrap; }}
  .rank {{ font-family: var(--font-display); font-style: italic; font-weight: 700;
    font-size: 1.05rem; color: var(--line); letter-spacing: -.02em; }}
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

  /* tale of the tape: os dois cantos frente a frente */
  .tape {{ display: grid; grid-template-columns: 1fr auto 1fr; align-items: center;
    gap: 14px; margin-bottom: 18px; }}
  .tape-vs {{ font-family: var(--font-display); font-style: italic; font-weight: 700;
    text-transform: uppercase; font-size: .7rem; color: var(--muted); letter-spacing: .06em; }}
  .corner {{ display: flex; align-items: center; gap: 12px; min-width: 0; }}
  .corner.r {{ flex-direction: row-reverse; text-align: right; }}
  .corner-id {{ min-width: 0; }}
  .corner-name {{ display: block; font-family: var(--font-display); font-weight: 700;
    font-style: italic; text-transform: uppercase; letter-spacing: -.005em;
    font-size: 1.14rem; line-height: 1.06; color: var(--dim); }}
  .corner.is-pick .corner-name {{ color: var(--text); }}
  .corner-tag {{ display: block; font-size: .66rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: .1em; margin-top: 4px; }}
  .pick-flag {{ display: inline-block; margin-top: 7px; font-size: .59rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .1em; color: #fff;
    background: var(--accent); padding: 3px 9px; transform: skewX(var(--slash)); }}
  #favs .pick-flag {{ color: #000; }}

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

  /* barra divergente: um eixo, os dois lados se encontrando */
  .splits {{ display: flex; flex-direction: column; gap: 9px; }}
  .split-row {{ display: grid; grid-template-columns: 62px 46px 1fr 46px;
    align-items: center; gap: 10px; }}
  .split-key {{ font-size: .64rem; color: var(--muted); text-transform: uppercase;
    letter-spacing: .1em; }}
  .split-bar {{ display: flex; height: 9px; background: var(--surface2); position: relative; }}
  .seg {{ height: 100%; }}
  .seg.l, .seg.r {{ background: #33333D; }}
  .seg.side {{ background: var(--accent); }}
  .split-mid {{ position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px;
    background: var(--bg); }}
  .split-num {{ font-family: var(--font-num); font-size: .72rem; color: var(--muted);
    font-variant-numeric: tabular-nums; }}
  .split-num.l {{ text-align: right; }}
  .split-num.side {{ color: var(--text); }}

  /* destaque da luta principal */
  .hero-wrap {{ margin-bottom: 30px; }}
  .hero-label {{ font-family: var(--font-display); font-style: italic; font-weight: 700;
    text-transform: uppercase; letter-spacing: .22em; font-size: .64rem;
    color: #fff; background: var(--brand); display: inline-block;
    padding: 4px 14px 4px 11px; transform: skewX(var(--slash)); margin-bottom: 8px; }}
  .bout.hero {{ border-top: 3px solid var(--brand); border-bottom: 1px solid var(--line);
    background: var(--surface); padding: 22px 22px 24px; }}
  .bout.hero .corner-name {{ font-size: clamp(1.15rem, 2.6vw, 1.7rem); }}
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
  .ev-value {{ font-family: var(--font-num); font-size: 1.5rem; font-weight: 700;
    color: var(--accent); font-variant-numeric: tabular-nums; line-height: 1; }}
  .ev-value small {{ font-size: .56rem; letter-spacing: .12em; color: var(--muted);
    margin-left: 5px; text-transform: uppercase; }}
  .ev-body {{ display: flex; align-items: center; gap: 14px; }}
  .ev-id {{ min-width: 0; flex: 1; }}
  .ev-figs {{ display: flex; gap: 26px; }}
  .fig {{ text-align: right; }}
  .fig-n {{ display: block; font-family: var(--font-num); font-size: 1.02rem;
    color: var(--text); font-variant-numeric: tabular-nums; }}
  .fig-n small {{ font-size: .6rem; color: var(--muted); margin-left: 2px; }}
  .fig-n.muted {{ color: var(--muted); }}
  .fig-n.pos {{ color: var(--green); }}
  .fig-n.neg {{ color: var(--red); }}
  .fig-k {{ display: block; font-size: .6rem; color: var(--muted); text-transform: uppercase;
    letter-spacing: .1em; margin-top: 3px; }}

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

  footer {{ color: var(--muted); font-size: .7rem; line-height: 1.7; margin-top: 40px;
    border-top: 1px solid var(--line); padding-top: 16px; max-width: 72ch; }}
  footer strong {{ color: var(--dim); font-weight: 600; }}

  /* ---------- responsivo ---------- */
  @media (max-width: 720px) {{
    .wrap {{ padding: 0 14px; }}
    .event {{ padding: 24px 0 18px; }}
    .ev-body {{ flex-wrap: wrap; }}
    .ev-figs {{ width: 100%; justify-content: space-between; gap: 12px; }}
    .fig {{ text-align: left; }}
    .mini-row {{ grid-template-columns: 92px 1fr 38px; }}
    .mini-row .odds-chip {{ grid-column: 2 / -1; margin-top: -2px; }}
  }}
  @media (max-width: 560px) {{
    /* os dois cantos empilham; o VS vira regua entre eles */
    .tape {{ grid-template-columns: 1fr; gap: 10px; }}
    .corner.r {{ flex-direction: row; text-align: left; }}
    .tape-vs {{ border-top: 1px solid var(--line); padding-top: 8px; }}
    .avatar.lg {{ width: 76px; height: 76px; }}
    .split-row {{ grid-template-columns: 48px 38px 1fr 38px; gap: 7px; }}
    .split-key {{ font-size: .58rem; }}
    .masthead .meta {{ display: none; }}
  }}
</style>
</head>
<body>
<div class="masthead">
  <div class="wrap">
    <div class="brand">{_MARK}<b>Fight Model</b><span>modelo vs. mercado</span></div>
    <div class="meta">gerado {generated}<br>modelo {_e(analysis['model_name'])} · calibrado</div>
  </div>
</div>

<div class="wrap">
  <section class="event">
    <div class="kicker">Card analisado{event_kicker}</div>
    <h1>{title}</h1>
    <div class="sub">Probabilidade do modelo contra o mercado · odds com vig removido</div>
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
                       history_panel_html=history_panel, event_date=event_date)
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
