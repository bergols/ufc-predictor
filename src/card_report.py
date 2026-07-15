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
from src.predict import compute_total_rounds_market
from src.prediction_history import (HISTORY_CSS, avatar_html, load_history,
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
    # coluna opcional: rounds agendados (3 na maioria; 5 em titulo/main event).
    # Default 3 -- e conhecido antes da luta, e usado para restringir a faixa
    # de round prevista (luta de 3 rounds nao tem round 4-5).
    if "scheduled_rounds" not in df.columns:
        logger.info("CSV sem coluna scheduled_rounds -- assumindo 3 rounds para todas "
                    "(adicione a coluna com 5 para main event/titulo).")
        df["scheduled_rounds"] = 3
    df["scheduled_rounds"] = pd.to_numeric(df["scheduled_rounds"], errors="coerce").fillna(3).astype(int)
    return df


def _default_predict_fns(model_name: str) -> tuple[Callable, Callable]:
    """Prepara os preditores (vencedor e metodo/duracao) com a base de niveis
    carregada UMA vez para o card todo. allow_debutant=True: estreante vira
    linha sintetica (stats NaN, Elo base) em vez de derrubar a luta do
    relatorio — a previsao sai marcada com aviso proprio no card."""
    from src.features import export_latest_fighter_levels
    from src.predict import predict_fight, predict_method_and_duration
    levels = export_latest_fighter_levels()
    winner_fn = lambda a, b: predict_fight(a, b, model_name=model_name, levels=levels,  # noqa: E731
                                           allow_debutant=True)
    method_fn = lambda a, b, sr: predict_method_and_duration(  # noqa: E731
        a, b, levels=levels, scheduled_rounds=sr, allow_debutant=True)
    return winner_fn, method_fn


def analyze_card(odds_df: pd.DataFrame, model_name: str = "logreg",
                 predict_fn: Optional[Callable[[str, str], dict]] = None,
                 method_fn: Optional[Callable[[str, str], dict]] = None) -> dict:
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

    predict_fn / method_fn sao injetaveis para teste. As tres previsoes
    (vencedor / metodo / duracao) falham de forma INDEPENDENTE: falha so
    no metodo/duracao mantem a luta na categoria com a secao de tendencia
    marcada como indisponivel.
    """
    if predict_fn is None:
        predict_fn, default_method_fn = _default_predict_fns(model_name)
        if method_fn is None:
            method_fn = default_method_fn

    predicted, no_prediction = [], []
    for _, row in odds_df.iterrows():
        a, b = str(row["fighter_a"]).strip(), str(row["fighter_b"]).strip()
        odds_a, odds_b = float(row["odds_a_decimal"]), float(row["odds_b_decimal"])
        scheduled = int(row["scheduled_rounds"]) if "scheduled_rounds" in row.index else 3

        market_a, market_b = remove_vig_two_way(
            decimal_odds_to_implied_prob(odds_a), decimal_odds_to_implied_prob(odds_b))

        base = {"fighter_a": a, "fighter_b": b, "odds_a": odds_a, "odds_b": odds_b,
                "market_prob_a": market_a, "market_prob_b": market_b,
                "scheduled_rounds": scheduled}

        try:
            pred = predict_fn(a, b)
        except ValueError as exc:
            no_prediction.append({**base, "reason": str(exc)})
            continue

        # metodo/duracao: falha independente (nao derruba a previsao de vencedor)
        method_probs, round_band_probs = None, None
        if method_fn is not None:
            try:
                mp = method_fn(a, b, scheduled)
                method_probs = mp["method_probs"]
                round_band_probs = mp["round_band_probs"]
            except (ValueError, FileNotFoundError) as exc:
                logger.info("Sem tendencia de metodo/duracao para %s vs %s: %s", a, b, exc)

        model_a = pred["prob_a_wins"]
        fav_is_a = market_a >= market_b
        model_side_is_a = model_a >= 0.5
        fight = {
            **base,
            "method_probs": method_probs,
            "round_band_probs": round_band_probs,
            # nomes como casados na base (fuzzy pode ter corrigido grafia)
            "matched_a": pred["fighter_a"], "matched_b": pred["fighter_b"],
            "model_prob_a": model_a, "model_prob_b": 1 - model_a,
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
        # mercado de duracao (over/under 1,5 rounds) derivado das distribuicoes
        # de metodo/faixa -- so quando ambas existem (falhas sao independentes)
        fight["totals_market"] = (
            compute_total_rounds_market(method_probs, round_band_probs)
            if method_probs and round_band_probs else None)
        predicted.append(fight)

    favorites = sorted((f for f in predicted if f["category"] == "favorite"),
                       key=lambda f: f["model_side_prob"], reverse=True)
    underdogs = sorted((f for f in predicted if f["category"] == "underdog"),
                       key=lambda f: f["model_side_prob"], reverse=True)
    # abas de metodo/duracao: ordenacao "fria" pela probabilidade da categoria
    # mais provavel de cada luta; lutas sem dado ficam num grupo a parte da aba
    method_ranking = sorted((f for f in predicted if f["method_probs"]),
                            key=lambda f: max(f["method_probs"].values()), reverse=True)
    # ordenacao da aba de duracao: probabilidade do lado favorecido na LINHA
    # DE 1,5 (criterio principal; o card mostra as duas linhas)
    duration_ranking = sorted(
        (f for f in predicted if f["totals_market"]),
        key=lambda f: max(f["totals_market"]["over_1_5"], f["totals_market"]["under_1_5"]),
        reverse=True)
    no_method = [f for f in predicted if not f["method_probs"]]
    no_duration = [f for f in predicted if not f["totals_market"]]
    ev_legs = sorted((f for f in predicted if f["ev"] > 1),
                     key=lambda f: f["ev"], reverse=True)
    return {"favorites": favorites, "underdogs": underdogs, "no_prediction": no_prediction,
            "method_ranking": method_ranking, "duration_ranking": duration_ranking,
            "no_method": no_method, "no_duration": no_duration, "ev_legs": ev_legs,
            "model_name": model_name}


# ---------------------------------------------------------------------------
# HTML (self-contained: CSS + JS puros inline, funciona offline)
# ---------------------------------------------------------------------------

def _e(text) -> str:
    return html_mod.escape(str(text))


def _prob_bar(model_p: float, market_p: float) -> str:
    return f"""
      <div class="probs">
        <div class="prob-row"><span class="prob-label">modelo</span>
          <div class="bar"><div class="fill model" style="width:{model_p * 100:.1f}%"></div></div>
          <span class="prob-val">{model_p * 100:.1f}%</span></div>
        <div class="prob-row"><span class="prob-label">mercado</span>
          <div class="bar"><div class="fill market" style="width:{market_p * 100:.1f}%"></div></div>
          <span class="prob-val">{market_p * 100:.1f}%</span></div>
      </div>"""


def _side_box(name: str, tag: str, model_p: float, market_p: float, highlighted: bool) -> str:
    """Um lado da luta: avatar + nome + probabilidades. O lado apontado
    pelo modelo vem destacado com anel e a etiqueta 'lado do modelo'."""
    cls = "side highlight" if highlighted else "side"
    star = '<span class="side-star">lado do modelo</span>' if highlighted else ""
    return f"""
      <div class="{cls}">
        <div class="side-head">{avatar_html(name)}
          <div class="side-id"><strong>{_e(name)}</strong>
            <span class="side-tag">{tag}</span></div>
          {star}</div>
        {_prob_bar(model_p, market_p)}
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
    "clock": ('<svg class="mini-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" '
              'stroke-width="1.4"><circle cx="8" cy="8" r="6"/><path d="M8 4.5V8l2.4 1.6"/></svg>'),
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
    """Card da aba 'Metodo de vitoria': odds justas das 3 categorias."""
    mp = fight["method_probs"]
    top = max(mp, key=mp.get)
    rows = "".join(_odds_row(label, mp[key], icon_key=key, strong=(key == top))
                   for key, label in _METHOD_LABELS)
    return f"""
    <div class="fight-card">
      <div class="rank">#{rank}</div>
      <div class="fight-body">
        <div class="names">{avatar_html(fight['fighter_a'], small=True)} {_e(fight['fighter_a'])}
          <span class="vs">vs</span>
          {avatar_html(fight['fighter_b'], small=True)} {_e(fight['fighter_b'])}</div>
        <div class="method-box">{rows}</div>
        {_matched_note(fight)}
      </div>
    </div>"""


def _duration_card(fight: dict, rank: int) -> str:
    """Card da aba 'Duracao da luta': linhas over/under 1,5 E 2,5 rounds com odds justas."""
    tm = fight["totals_market"]

    def line_block(line_label: str, over: float, under: float) -> str:
        return (f'<div class="mini-caption">{_ICONS["clock"]} linha {line_label} rounds:</div>'
                + _odds_row(f"Over {line_label}", over, strong=over >= under)
                + _odds_row(f"Under {line_label}", under, strong=under > over))

    return f"""
    <div class="fight-card">
      <div class="rank">#{rank}</div>
      <div class="fight-body">
        <div class="names">{avatar_html(fight['fighter_a'], small=True)} {_e(fight['fighter_a'])}
          <span class="vs">vs</span>
          {avatar_html(fight['fighter_b'], small=True)} {_e(fight['fighter_b'])}
          <span class="side-tag">luta de {fight.get('scheduled_rounds', 3)} rounds</span></div>
        <div class="method-box">
          <div class="method-cols">
            <div class="method-col">{line_block("1,5", tm['over_1_5'], tm['under_1_5'])}</div>
            <div class="method-col">{line_block("2,5", tm['over_2_5'], tm['under_2_5'])}</div>
          </div>
        </div>
        {_matched_note(fight)}
      </div>
    </div>"""


def _ev_card(fight: dict, rank: int) -> str:
    """Card da aba 'Pernas EV>1': o lado do modelo com odd, EV e contexto."""
    side = fight["model_side"]
    opponent = fight["fighter_b"] if side == fight["fighter_a"] else fight["fighter_a"]
    market_side_prob = (fight["market_prob_fav"] if side == fight["favorite"]
                        else fight["market_prob_dog"])
    tipo = ("favorito do mercado" if side == fight["favorite"]
            else "azarão do mercado (zebra)")
    return f"""
    <div class="fight-card">
      <div class="fight-body">
        <div class="fight-top"><span class="rank">#{rank}</span>
          <span class="ev-chip">EV {fight['ev']:.2f}</span>
          <span class="side-tag">{tipo}</span></div>
        <div class="side-head">{avatar_html(side)}
          <div class="side-id"><strong>{_e(side)}</strong>
            <span class="side-tag">vs {_e(opponent)} · odd {fight['model_side_odds']:.2f}</span></div>
        </div>
        {_prob_bar(fight['model_side_prob'], market_side_prob)}
        {_matched_note(fight)}
      </div>
    </div>"""


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


def _fight_card(fight: dict, rank: int, tab: str) -> str:
    """
    Card de uma luta mostrando os dois lados, com o lado apontado pelo
    modelo (model_side) em destaque. A categorizacao e mutuamente
    exclusiva: cada luta aparece em exatamente uma aba. Copy factual, sem
    linguagem de recomendacao.
    """
    p = fight["model_side_prob"] * 100
    if tab == "favs":
        badge = f'<span class="badge ok">modelo concorda · {p:.1f}%</span>'
        highlight_fav = True
    else:
        badge = f'<span class="badge value">zebra · modelo dá {p:.1f}% ao azarão</span>'
        highlight_fav = False

    fav_box = _side_box(fight["favorite"], "favorito do mercado",
                        fight["model_prob_fav"], fight["market_prob_fav"], highlighted=highlight_fav)
    dog_box = _side_box(fight["underdog"], "azarão do mercado",
                        fight["model_prob_dog"], fight["market_prob_dog"], highlighted=not highlight_fav)
    vs_chip = '<div class="vs-chip"><span>VS</span></div>'
    # o lado destacado (model_side) vem primeiro
    boxes = (fav_box + vs_chip + dog_box) if highlight_fav else (dog_box + vs_chip + fav_box)

    return f"""
    <div class="fight-card">
      <div class="fight-body">
        <div class="fight-top"><span class="rank">#{rank}</span> {badge}</div>
        <div class="sides">{boxes}</div>
        <div class="odds-line">odds {fight['odds_a']:.2f} / {fight['odds_b']:.2f} ·
          mercado (devig): {_e(fight['favorite'])} {fight['market_prob_fav'] * 100:.1f}% ·
          {_e(fight['underdog'])} {fight['market_prob_dog'] * 100:.1f}%</div>
        {_matched_note(fight)}
      </div>
    </div>"""


_FAIR_ODDS_WARNING = """
    <p class="tab-explain warn-strong"><strong>Atenção — odds justas, não odds reais:</strong>
    estas são odds JUSTAS calculadas a partir da probabilidade do nosso modelo (sem margem de
    casa) — não são odds reais de mercado, pois não temos odds de casas de aposta para
    método/duração para comparar. Diferente do preditor de vencedor, este modelo nunca foi
    validado contra o mercado real — só contra um baseline ingênuo, com resultado modesto
    (ver README).</p>"""


def render_html(analysis: dict, freshness_gap_days: Optional[int], card_name: str = "",
                history_panel_html: str = "") -> str:
    fav_cards = "\n".join(_fight_card(f, i + 1, "favs")
                          for i, f in enumerate(analysis["favorites"]))
    dog_cards = "\n".join(_fight_card(f, i + 1, "dogs")
                          for i, f in enumerate(analysis["underdogs"]))
    ev_cards = "\n".join(_ev_card(f, i + 1)
                         for i, f in enumerate(analysis.get("ev_legs", [])))
    method_cards = "\n".join(_method_card(f, i + 1)
                             for i, f in enumerate(analysis.get("method_ranking", [])))
    duration_cards = "\n".join(_duration_card(f, i + 1)
                               for i, f in enumerate(analysis.get("duration_ranking", [])))

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
        fresh_html = '<div class="freshness bad">⚠ Não foi possível verificar o frescor da base de dados.</div>'
    elif freshness_gap_days > config.DATA_FRESHNESS_MAX_GAP_DAYS:
        fresh_html = (f'<div class="freshness bad">⚠ DADOS DESATUALIZADOS: o evento mais recente na base '
                      f'tem {freshness_gap_days} dias (limite: {config.DATA_FRESHNESS_MAX_GAP_DAYS}). '
                      f'As probabilidades do modelo não refletem lutas recentes.</div>')
    else:
        fresh_html = (f'<div class="freshness ok">✓ Base de dados em dia: evento mais recente há '
                      f'{freshness_gap_days} dia(s).</div>')

    title = _e(card_name) if card_name else "Card UFC"
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — modelo vs. mercado</title>
<style>
  /* ============ tema "Fight Night" — 100% offline, sem fonte externa ============ */
  :root {{
    --bg: #0B0B10; --panel: #16161D; --panel2: #1c1c25; --line: #26262f;
    --text: #F5F5F5; --muted: #9A9AA5; --gold: #F4B740; --red: #E63946;
    --green: #3fa66a; --neutral: #6e6e7e; --steel: #8FA8DC;
    --font-display: "Arial Narrow", "Helvetica Neue Condensed", "Roboto Condensed",
                    ui-sans-serif, system-ui, sans-serif;
    --font-body: ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text); font-family: var(--font-body);
    line-height: 1.5; padding: 28px 16px 64px; min-height: 100vh;
    /* vinheta radial: profundidade sem imagem externa */
    background-image: radial-gradient(ellipse 120% 90% at 50% 0%,
      #14141c 0%, var(--bg) 45%, #060609 100%);
    background-attachment: fixed;
  }}
  .wrap {{ max-width: 880px; margin: 0 auto; }}

  /* ---------- cabecalho estilo cartaz ---------- */
  header {{ text-align: center; margin-bottom: 18px; }}
  header h1 {{
    font-family: var(--font-display); font-weight: 700; text-transform: uppercase;
    font-size: clamp(1.5rem, 4.5vw, 2.4rem); letter-spacing: .06em; line-height: 1.15;
    background: linear-gradient(180deg, #ffffff 30%, #c9c9d4 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }}
  header .sub {{ color: var(--muted); margin-top: 6px; font-size: .88rem;
    letter-spacing: .04em; text-transform: uppercase; }}
  header .rule {{ width: 120px; height: 2px; margin: 14px auto 0;
    background: linear-gradient(90deg, transparent, var(--gold), transparent); }}

  .disclaimer {{
    color: var(--muted); font-size: .8rem; border-left: 3px solid var(--red);
    background: rgba(230, 57, 70, .06); border-radius: 0 8px 8px 0;
    padding: 8px 14px; margin: 16px 0 10px;
  }}
  .disclaimer strong {{ color: #d8d8e0; }}
  .freshness {{ border-radius: 8px; padding: 8px 14px; font-size: .82rem; margin-bottom: 22px; }}
  .freshness.ok {{ background: rgba(63, 166, 106, .08); border: 1px solid rgba(63, 166, 106, .45);
    color: #9fd4b5; }}
  .freshness.bad {{ background: rgba(230, 57, 70, .1); border: 1px solid var(--red); color: #f0a5ab; }}

  /* ---------- abas com indicador e transicao ---------- */
  .tabs {{ display: flex; gap: 4px; margin-bottom: 16px; border-bottom: 1px solid var(--line); }}
  .tab-btn {{
    flex: 1; background: transparent; color: var(--muted); border: none;
    border-bottom: 3px solid transparent; padding: 12px 8px 10px; cursor: pointer;
    font-family: var(--font-display); font-weight: 700; text-transform: uppercase;
    letter-spacing: .08em; font-size: 1rem; transition: color .2s, border-color .2s, background .2s;
  }}
  .tab-btn:hover {{ color: var(--text); background: rgba(255,255,255,.02); }}
  .tab-btn[data-tab="favs"].active {{ color: var(--gold); border-bottom-color: var(--gold); }}
  .tab-btn[data-tab="dogs"].active {{ color: var(--red); border-bottom-color: var(--red); }}
  .tab-btn[data-tab="method"].active, .tab-btn[data-tab="duration"].active {{
    color: var(--steel); border-bottom-color: var(--steel); }}
  .tab-btn[data-tab="history"].active {{ color: var(--green); border-bottom-color: var(--green); }}
  .tab-btn[data-tab="ev"].active {{ color: var(--green); border-bottom-color: var(--green); }}
  #ev .rank {{ color: var(--green); }}
  #ev .fill.model {{ background: linear-gradient(90deg, #1f5c3a, var(--green)); }}
  .ev-chip {{ font-family: var(--font-display); font-weight: 700; font-size: .82rem;
    color: #9fd4b5; background: rgba(63,166,106,.1); border: 1px solid rgba(63,166,106,.5);
    border-radius: 999px; padding: 2px 12px; font-variant-numeric: tabular-nums; }}
  #method .rank, #duration .rank {{ color: var(--steel); }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; animation: panelIn .28s ease-out; }}
  @keyframes panelIn {{ from {{ opacity: 0; transform: translateY(6px); }}
                        to {{ opacity: 1; transform: none; }} }}

  .tab-explain {{ color: var(--muted); font-size: .82rem; background: var(--panel);
    border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; margin-bottom: 16px; }}
  .tab-explain strong {{ color: #cfcfda; }}

  /* ---------- card de luta ---------- */
  .fight-card {{
    display: flex; gap: 14px; background: var(--panel); border: 1px solid var(--line);
    border-radius: 14px; padding: 16px 18px; margin-bottom: 14px;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
  }}
  .fight-card:hover {{ transform: translateY(-2px); border-color: #3a3a48;
    box-shadow: 0 8px 24px rgba(0, 0, 0, .45), 0 0 0 1px rgba(255,255,255,.03) inset; }}
  #favs .fight-card:hover {{ box-shadow: 0 8px 24px rgba(0,0,0,.45), 0 0 14px rgba(244,183,64,.08); }}
  #dogs .fight-card:hover {{ box-shadow: 0 8px 24px rgba(0,0,0,.45), 0 0 14px rgba(230,57,70,.10); }}
  .rank {{ font-family: var(--font-display); font-size: 1.5rem; font-weight: 700;
    min-width: 1.9em; padding-top: 2px; opacity: .9; }}
  #favs .rank {{ color: var(--gold); }}
  #dogs .rank {{ color: var(--red); }}
  .fight-body {{ flex: 1; min-width: 0; }}
  .fight-top {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
  .fight-top .rank {{ font-size: 1.25rem; min-width: 0; padding-top: 0; }}
  .names {{ font-family: var(--font-display); text-transform: uppercase; letter-spacing: .05em;
    font-size: 1.1rem; font-weight: 700; margin-bottom: 10px;
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
  .vs {{ color: var(--muted); font-size: .8rem; margin: 0 6px; }}

  /* ---------- avatares de monograma (sem foto externa: offline + direitos) ---------- */
  .avatar {{ width: 42px; height: 42px; border-radius: 50%; flex: none;
    display: inline-flex; align-items: center; justify-content: center;
    font-family: var(--font-display); font-weight: 700; font-size: .95rem;
    letter-spacing: .03em; color: #eceff4; text-transform: uppercase;
    border: 1px solid rgba(255,255,255,.12);
    box-shadow: 0 2px 8px rgba(0,0,0,.35) inset, 0 1px 3px rgba(0,0,0,.4); }}
  .avatar.sm {{ width: 24px; height: 24px; font-size: .58rem; vertical-align: -6px; }}

  /* pills */
  .badge {{ font-family: var(--font-body); text-transform: none; letter-spacing: 0;
    font-size: .7rem; font-weight: 700; padding: 3px 10px; border-radius: 999px;
    margin-left: 8px; vertical-align: 2px; white-space: nowrap; }}
  .badge.ok {{ background: rgba(244, 183, 64, .14); color: var(--gold);
    border: 1px solid rgba(244, 183, 64, .5); }}
  .badge.warn {{ background: transparent; color: #f0a5ab; border: 1px solid var(--red); }}
  .badge.value {{ background: var(--red); color: #fff; border: 1px solid var(--red);
    box-shadow: 0 2px 10px rgba(230, 57, 70, .3); }}
  .badge.novalue {{ background: var(--panel2); color: var(--muted); border: 1px solid var(--line);
    font-weight: 400; }}

  /* ---------- os dois lados + separador VS ---------- */
  .sides {{ display: flex; align-items: stretch; gap: 0; margin: 8px 0; }}
  .side {{ flex: 1; min-width: 0; background: #111117; border: 1px solid var(--line);
    border-radius: 12px; padding: 10px 14px; opacity: .68; transition: opacity .2s; }}
  .fight-card:hover .side {{ opacity: .8; }}
  .side.highlight, .fight-card:hover .side.highlight {{ opacity: 1; border-color: transparent; }}
  #favs .side.highlight {{ box-shadow: 0 0 0 1.5px var(--gold); background: #17150e; }}
  #dogs .side.highlight {{ box-shadow: 0 0 0 1.5px var(--red); background: #171012; }}
  .vs-chip {{ display: flex; align-items: center; justify-content: center; padding: 0 6px; }}
  .vs-chip span {{
    font-family: var(--font-display); font-weight: 700; font-style: italic; font-size: .78rem;
    color: var(--muted); background: var(--panel2); border: 1px solid var(--line);
    border-radius: 999px; width: 34px; height: 34px; display: flex;
    align-items: center; justify-content: center; letter-spacing: .04em;
  }}
  .side-head {{ font-size: .95rem; margin-bottom: 8px;
    display: flex; align-items: center; gap: 10px; }}
  .side-head strong {{ font-family: var(--font-display); text-transform: uppercase;
    letter-spacing: .04em; font-size: 1.02rem; line-height: 1.15; display: block; }}
  .side-id {{ min-width: 0; flex: 1; }}
  .side-tag {{ color: var(--muted); font-size: .68rem; white-space: nowrap; display: block; }}
  .side-star {{ font-size: .62rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .06em; white-space: nowrap; align-self: flex-start;
    padding: 2px 8px; border-radius: 999px; }}
  #favs .side-star {{ color: var(--gold); border: 1px solid rgba(244,183,64,.5);
    background: rgba(244,183,64,.1); }}
  #dogs .side-star {{ color: #fff; background: var(--red); border: 1px solid var(--red); }}

  /* barras: modelo ganha o acento da aba; mercado fica neutro */
  .probs {{ margin: 6px 0 2px; }}
  .prob-row {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
  .prob-label {{ color: var(--muted); font-size: .72rem; width: 54px; text-align: right; }}
  .bar {{ flex: 1; height: 9px; background: #08080c; border-radius: 999px; overflow: hidden; }}
  .fill {{ height: 100%; border-radius: 999px; }}
  #favs .fill.model {{ background: linear-gradient(90deg, #8a6620, var(--gold)); }}
  #dogs .fill.model {{ background: linear-gradient(90deg, #7e1f28, var(--red)); }}
  .fill.market {{ background: linear-gradient(90deg, #3c3c4a, #6a6a7c); }}
  .fill.neutral {{ background: linear-gradient(90deg, #4a4a58, #82828f); }}
  .prob-val {{ font-size: .8rem; width: 50px; font-variant-numeric: tabular-nums; }}

  .odds-line {{ color: var(--muted); font-size: .76rem; margin-top: 6px; }}
  .note {{ color: #d8b46a; font-size: .76rem; margin-top: 6px; }}

  /* ---------- "como a luta tende a terminar" (neutro de proposito) ---------- */
  .method-box {{ background: rgba(255,255,255,.015); border: 1px dashed var(--line);
    border-radius: 10px; padding: 10px 14px; margin: 10px 0 4px; }}
  .method-title {{ font-family: var(--font-display); text-transform: uppercase;
    letter-spacing: .06em; font-size: .8rem; font-weight: 700; color: #cfcfda; margin-bottom: 8px; }}
  .method-sub {{ display: block; font-family: var(--font-body); text-transform: none;
    letter-spacing: 0; color: var(--muted); font-size: .7rem; font-weight: 400; margin-top: 3px; }}
  .method-cols {{ display: flex; gap: 20px; flex-wrap: wrap; }}
  .method-col {{ flex: 1; min-width: 240px; }}
  .mini-caption {{ color: var(--muted); font-size: .72rem; margin-bottom: 4px; }}
  .mini-row {{ display: flex; align-items: center; gap: 8px; margin: 3px 0; }}
  .mini-label {{ color: var(--muted); font-size: .74rem; width: 104px; text-align: right;
    display: inline-flex; justify-content: flex-end; align-items: center; gap: 5px; }}
  .mini-icon {{ width: 13px; height: 13px; color: var(--neutral); flex: none; }}
  #method .mini-label, #duration .mini-label {{ width: 132px; }}
  .odds-chip {{ font-variant-numeric: tabular-nums; font-size: .82rem; font-weight: 700;
    color: var(--steel); background: rgba(143, 168, 220, .08);
    border: 1px solid rgba(143, 168, 220, .3); border-radius: 6px;
    padding: 1px 8px; min-width: 92px; text-align: center; white-space: nowrap; }}
  .odds-chip small {{ color: var(--muted); font-weight: 400; }}
  .mini-row.strong .mini-label {{ color: var(--text); font-weight: 600; }}
  .mini-row.strong .prob-val {{ font-weight: 700; }}
  .mini-row.strong .odds-chip {{ border-color: var(--steel); }}
  .tab-explain.warn-strong {{ border: 1px solid var(--red); border-left-width: 4px;
    background: rgba(230, 57, 70, .05); color: #c9c9d4; }}

  /* ---------- aba historico (eventos passados) ---------- */
{HISTORY_CSS}

  /* ---------- sem previsao: discreto ---------- */
  .no-pred {{ margin-top: 30px; border-top: 1px dashed var(--line); padding-top: 16px; opacity: .8; }}
  .no-pred h2 {{ font-family: var(--font-display); text-transform: uppercase;
    letter-spacing: .06em; font-size: .95rem; color: var(--muted); margin-bottom: 6px; }}
  .no-pred p {{ color: #77777f; font-size: .8rem; margin-bottom: 10px; }}
  .no-pred ul {{ list-style: none; }}
  .no-pred li {{ background: transparent; border: 1px solid var(--line); border-radius: 10px;
    padding: 9px 14px; margin-bottom: 8px; font-size: .86rem; color: var(--muted); }}
  .no-pred li strong {{ color: #c4c4cd; font-weight: 600; }}

  footer {{ color: #6e6e7c; font-size: .74rem; margin-top: 30px; border-top: 1px solid var(--line);
    padding-top: 12px; text-align: center; }}

  /* ---------- responsivo: colapsa para 1 coluna no celular ---------- */
  @media (max-width: 640px) {{
    body {{ padding: 18px 10px 48px; }}
    .fight-card {{ flex-direction: column; gap: 6px; padding: 14px; }}
    .rank {{ min-width: 0; }}
    .sides {{ flex-direction: column; gap: 8px; }}
    .vs-chip {{ padding: 0; margin: -4px 0; }}
    .vs-chip span {{ width: 28px; height: 28px; font-size: .68rem; }}
    .method-cols {{ flex-direction: column; gap: 10px; }}
    .mini-label {{ width: 92px; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>{title}</h1>
    <div class="sub">Modelo ({_e(analysis['model_name'])}) vs. mercado · odds com vig removido</div>
    <div class="rule"></div>
  </header>

  <div class="disclaimer">
    <strong>Aviso:</strong> estimativa estatística baseada em dados históricos,
    <strong>não é recomendação de aposta</strong>. MMA tem alta variância — zebras acontecem e
    favoritos caem. O próprio modelo perde para o mercado em backtest (ver README).
    Confira a defasagem dos dados (<code>check_data_freshness</code>) antes de usar.
  </div>
  {fresh_html}

  <div class="tabs">
    <button class="tab-btn active" data-tab="favs">Favoritos mais seguros</button>
    <button class="tab-btn" data-tab="dogs">Melhores zebras</button>
    <button class="tab-btn" data-tab="ev">Pernas EV&gt;1</button>
    <button class="tab-btn" data-tab="method">Método de vitória</button>
    <button class="tab-btn" data-tab="duration">Duração da luta</button>
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
  <div id="duration" class="tab-panel">
    {_FAIR_ODDS_WARNING}
    <p class="tab-explain"><strong>Critério:</strong> mercado de total de rounds com duas linhas —
    <strong>1,5</strong> (Under = termina no round 1) e <strong>2,5</strong> (Under = termina até
    o round 2). Decisão sempre passa das duas linhas (vai ao round 3 ou 5). Sem linhas de
    3,5/4,5: a faixa "round 3+" do modelo não separa finais tardios, e não fingimos precisão que
    não existe. Ordenação: probabilidade do lado favorecido na linha de 1,5, decrescente.</p>
    {duration_cards if duration_cards else '<p class="note">Nenhuma luta com previsão de duração.</p>'}
    {_no_data_list(analysis.get('no_duration', []), 'duração')}
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
                         event_date: str = "", photos: bool = False) -> Path:
    """
    `event_date` (YYYY-MM-DD): data do evento, usada para registrar as
    previsoes no historico de paper trading (congeladas ate o resultado) e
    para casar os resultados vindos do odds_template.csv. Sem ela o
    relatorio e gerado normalmente, mas o card nao entra no historico.

    `photos`: busca fotos dos lutadores no UFC.com (hotlink + cache local)
    para o relatorio de USO PESSOAL — nunca usado na pagina publicada
    (ver src/fighter_photos.py). O HTML deixa de ser offline/self-contained.
    """
    from src.data_collection import check_data_freshness

    odds_df = load_card_odds(csv_path)
    logger.info("Card carregado: %d lutas de %s", len(odds_df), csv_path)

    gap_days = check_data_freshness()
    analysis = analyze_card(odds_df, model_name=model_name)

    logger.info("Previstas: %d | sem previsao: %d",
                len(analysis["favorites"]), len(analysis["no_prediction"]))

    # historico: registra este card (se datado), puxa resultados ja
    # preenchidos no odds_template e monta a aba com os eventos passados
    if card_name and event_date:
        record_card_predictions(analysis, card_name, event_date)
    else:
        logger.info("Sem --event-date: card nao registrado no historico de previsoes.")
    sync_results_from_template()
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
                       history_panel_html=history_panel)
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
    args = parser.parse_args()

    generate_card_report(args.csv, args.output, model_name=args.model, card_name=args.card_name,
                         event_date=args.event_date, photos=args.photos)
