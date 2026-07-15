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
]


def _load_raw(history_csv: Path | None = None) -> pd.DataFrame:
    path = history_csv or config.PREDICTION_HISTORY_CSV
    if not Path(path).exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.read_csv(path)
    missing = [c for c in HISTORY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} sem as colunas {missing} — arquivo de historico corrompido?")
    # colunas de texto podem vir como float64 quando estao 100% vazias
    # (ex.: actual_winner antes do primeiro resultado) — normaliza para
    # object para aceitar strings no sync sem LossySetitemError (pandas 3)
    for col in ("event_name", "fighter_a", "fighter_b", "model_name", "model_side", "actual_winner"):
        df[col] = df[col].astype("object")
    return df


def _same_fight(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    """Mascara: mesma luta em qualquer ordem de lados."""
    return ((df["fighter_a"] == a) & (df["fighter_b"] == b)) | \
           ((df["fighter_a"] == b) & (df["fighter_b"] == a))


def record_card_predictions(analysis: dict, card_name: str, event_date: str,
                            history_csv: Path | None = None) -> int:
    """
    Grava/atualiza no historico as previsoes de um card (saida de
    card_report.analyze_card). Lutas sem previsao entram com os campos de
    modelo vazios — nunca somem em silencio. Linhas ja fechadas
    (actual_winner preenchido) sao intocaveis. Retorna quantas linhas
    foram gravadas/atualizadas.
    """
    path = Path(history_csv or config.PREDICTION_HISTORY_CSV)
    df = _load_raw(path)

    new_rows = []
    for fight in analysis["favorites"] + analysis["underdogs"]:
        new_rows.append({
            "event_name": card_name, "event_date": event_date,
            "fighter_a": fight["fighter_a"], "fighter_b": fight["fighter_b"],
            "odds_a_decimal": fight["odds_a"], "odds_b_decimal": fight["odds_b"],
            "model_name": analysis["model_name"],
            "model_prob_a": round(float(fight["model_prob_a"]), 4),
            "model_side": fight["model_side"],
            "actual_winner": np.nan,
        })
    for fight in analysis["no_prediction"]:
        new_rows.append({
            "event_name": card_name, "event_date": event_date,
            "fighter_a": fight["fighter_a"], "fighter_b": fight["fighter_b"],
            "odds_a_decimal": fight["odds_a"], "odds_b_decimal": fight["odds_b"],
            "model_name": analysis["model_name"],
            "model_prob_a": np.nan, "model_side": np.nan, "actual_winner": np.nan,
        })

    n_written = 0
    for row in new_rows:
        mask = (df["event_name"] == row["event_name"]) & _same_fight(df, row["fighter_a"], row["fighter_b"])
        existing = df[mask]
        if not existing.empty:
            if existing["actual_winner"].notna().any():
                continue  # linha fechada: previsao congelada, nao reescreve
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


def avatar_html(name: str, small: bool = False) -> str:
    """
    Avatar do lutador: monograma (iniciais em circulo, cor estavel por
    nome) e, se houver foto no mapa da geracao atual, a foto por cima —
    com fallback automatico para o monograma se a imagem nao carregar
    (onerror remove o <img> e as iniciais reaparecem).
    """
    cls = "avatar sm" if small else "avatar"
    hue = _name_hue(name)
    photo = _PHOTO_MAP.get(str(name))
    img = (f'<img src="{_e(photo)}" alt="" loading="lazy" onerror="this.remove()">'
           if photo else "")
    return (f'<span class="{cls}" style="background:linear-gradient(135deg,'
            f'hsl({hue},30%,32%),hsl({hue},38%,17%))">{_e(_initials(name))}{img}</span>')


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


def render_history_panel(history_df: pd.DataFrame) -> str:
    """Conteudo do painel da aba Historico: um bloco por evento (mais
    recente primeiro), com placar agregado modelo vs mercado no cabecalho."""
    if history_df.empty:
        return '<p class="note">Nenhum evento registrado ainda — o histórico começa no próximo card publicado.</p>'

    blocks = []
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
                          f'<span class="hist-score">mercado {market_hits}/{market_n}</span>')
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
  .avatar { position: relative; overflow: hidden; }
  .avatar img { position: absolute; inset: 0; width: 100%; height: 100%;
    object-fit: cover; object-position: center top; }
  .hist-event { background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
    padding: 14px 18px; margin-bottom: 14px; }
  .hist-head { display: flex; justify-content: space-between; align-items: baseline;
    gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }
  .hist-title { font-family: var(--font-display); text-transform: uppercase;
    letter-spacing: .05em; font-size: 1.05rem; font-weight: 700; }
  .hist-date { color: var(--muted); font-size: .78rem; margin-left: 8px; letter-spacing: 0; }
  .hist-score { font-size: .74rem; font-weight: 700; padding: 3px 10px; border-radius: 999px;
    background: var(--panel2); border: 1px solid var(--line); color: var(--muted); margin-left: 6px; }
  .hist-score.model-score { color: var(--gold); border-color: rgba(244,183,64,.5); }
  .hist-scroll { overflow-x: auto; }
  .hist-table { width: 100%; border-collapse: collapse; font-size: .84rem; }
  .hist-table th { text-align: left; color: var(--muted); font-size: .7rem;
    text-transform: uppercase; letter-spacing: .06em; padding: 6px 10px 6px 0;
    border-bottom: 1px solid var(--line); }
  .hist-table td { padding: 8px 10px 8px 0; border-bottom: 1px solid rgba(255,255,255,.04);
    vertical-align: middle; }
  .hist-table tr:last-child td { border-bottom: none; }
  .hist-fight { font-weight: 600; }
  .hist-prob { color: var(--muted); font-size: .76rem; }
  .hist-pending { color: var(--muted); font-style: italic; font-size: .8rem; }
  .hist-badge { font-size: .7rem; font-weight: 700; padding: 2px 8px; border-radius: 999px;
    white-space: nowrap; }
  .hist-badge.hit { background: rgba(63,166,106,.12); color: #9fd4b5;
    border: 1px solid rgba(63,166,106,.5); }
  .hist-badge.miss { background: rgba(230,57,70,.1); color: #f0a5ab; border: 1px solid var(--red); }
  .hist-badge.none { background: var(--panel2); color: var(--muted); border: 1px solid var(--line); }
"""
