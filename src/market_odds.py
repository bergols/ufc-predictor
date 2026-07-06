"""
src/market_odds.py

Comparacao modelo vs. mercado usando odds historicas REAIS, cumprindo o
item do escopo original ("comparar com odds implicitas de casas de
apostas, sem forcar conclusao otimista").

Fonte: data/complete_ufc_data.csv do repositorio jansen88/ufc-data
(GitHub), que cruza resultados oficiais com odds decimais historicas do
agregador betmma.tips.

LIMITACAO CENTRAL DE COBERTURA (verificada em jul/2026): as odds desse
dataset cobrem 2014-11-07 a 2023-09-16 e o repositorio esta parado desde
dez/2023. O conjunto de teste de PRODUCAO deste projeto comeca em nov/2023
-- ou seja, sobreposicao ZERO. Por isso este modulo roda um BACKTEST
dedicado: trunca as features no fim da janela de odds e re-treina o mesmo
pipeline (split temporal 70/15/15, calibracao e tudo), gerando um conjunto
de teste out-of-sample (~2021-2023) que cai dentro da janela com odds. O
modelo comparado nunca viu essas lutas no treino. Nada de producao e
sobrescrito (save_artifacts=False).

O fluxo manual (data/odds_template.csv, via src/evaluate.py) continua
existindo como complemento para eventos recentes sem odds nesta fonte.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.evaluate import compute_metrics
from src.utils import decimal_odds_to_implied_prob, remove_vig_two_way

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MARKET_ODDS_URL = "https://raw.githubusercontent.com/jansen88/ufc-data/master/data/complete_ufc_data.csv"

# Janela de tolerancia ao casar datas de evento entre as duas fontes
# (betmma.tips e UFCStats podem divergir por fuso horario).
_DATE_TOLERANCE_DAYS = 3
# Similaridade minima (difflib ratio) para aceitar um casamento fuzzy de nome.
_FUZZY_CUTOFF = 0.85


def _normalize_name(name: str) -> str:
    """minusculas, sem acentos, espacos colapsados -- para casar nomes entre fontes."""
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", name).strip().lower()


def download_market_odds(force: bool = False) -> pd.DataFrame:
    """Baixa (se necessario) e carrega o CSV de odds historicas."""
    if force or not config.MARKET_ODDS_CSV.exists():
        logger.info("Baixando odds historicas de %s ...", MARKET_ODDS_URL)
        df = pd.read_csv(MARKET_ODDS_URL)
        df.to_csv(config.MARKET_ODDS_CSV, index=False)
    return pd.read_csv(config.MARKET_ODDS_CSV)


def load_odds_fights() -> pd.DataFrame:
    """
    Carrega e filtra as lutas com odds completas, ja reportando a frescura
    da fonte (licao aprendida: fontes gratuitas estagnam sem aviso).
    """
    raw = download_market_odds()
    raw["event_date"] = pd.to_datetime(raw["event_date"], errors="coerce")

    odds = raw.dropna(subset=["favourite", "underdog", "favourite_odds", "underdog_odds",
                               "event_date"]).copy()
    # Filtra odds invalidas: o CSV tem lutas com odds = inf (dado quebrado na
    # origem), que passariam por um filtro "> 1.0" e virariam probabilidade
    # implicita 0 -- uma unica dessas explode o log loss do mercado (~13.8 de
    # penalidade), distorcendo a comparacao inteira. Odds decimais reais de
    # MMA ficam bem dentro de (1.0, 100).
    valid = (np.isfinite(odds["favourite_odds"]) & np.isfinite(odds["underdog_odds"])
             & (odds["favourite_odds"] > 1.0) & (odds["underdog_odds"] > 1.0)
             & (odds["favourite_odds"] < 100) & (odds["underdog_odds"] < 100))
    n_invalid = (~valid).sum()
    if n_invalid:
        logger.warning("%d luta(s) com odds invalidas (inf/<=1.0/>=100) descartadas.", n_invalid)
    odds = odds[valid]

    latest = odds["event_date"].max()
    gap_days = (pd.Timestamp.now().normalize() - latest).days
    logger.info("Odds historicas: %d lutas com odds, de %s a %s.",
                len(odds), odds["event_date"].min().date(), latest.date())
    if gap_days > config.DATA_FRESHNESS_MAX_GAP_DAYS:
        logger.warning(
            "Fonte de odds ESTAGNADA: ultima odd em %s (%d dias atras). A comparacao "
            "modelo-vs-mercado sera um backtest historico, nao um retrato do presente.",
            latest.date(), gap_days,
        )

    odds["_f1_norm"] = odds["fighter1"].map(_normalize_name)
    odds["_f2_norm"] = odds["fighter2"].map(_normalize_name)
    odds["_fav_norm"] = odds["favourite"].map(_normalize_name)
    return odds


def _match_fight(row_a: str, row_b: str, candidates: pd.DataFrame) -> Optional[pd.Series]:
    """
    Casa uma luta (nomes ja normalizados de fighter_a/fighter_b) contra as
    lutas candidatas da fonte de odds (mesma janela de datas). Tenta
    correspondencia exata do PAR de nomes; se falhar, fuzzy exigindo que os
    DOIS nomes casem com a MESMA luta candidata (cutoff em _FUZZY_CUTOFF).
    """
    if candidates.empty:
        return None

    # 1) par exato (em qualquer ordem)
    exact = candidates[
        ((candidates["_f1_norm"] == row_a) & (candidates["_f2_norm"] == row_b))
        | ((candidates["_f1_norm"] == row_b) & (candidates["_f2_norm"] == row_a))
    ]
    if len(exact) >= 1:
        return exact.iloc[0]

    # 2) fuzzy: melhor candidata em que ambos os nomes casam
    best_score, best_row = 0.0, None
    for _, cand in candidates.iterrows():
        for f1, f2 in ((cand["_f1_norm"], cand["_f2_norm"]),
                       (cand["_f2_norm"], cand["_f1_norm"])):
            s1 = SequenceMatcher(None, row_a, f1).ratio()
            s2 = SequenceMatcher(None, row_b, f2).ratio()
            if s1 >= _FUZZY_CUTOFF and s2 >= _FUZZY_CUTOFF and (s1 + s2) / 2 > best_score:
                best_score, best_row = (s1 + s2) / 2, cand
    return best_row


def match_odds_to_predictions(preds_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Casa cada luta do conjunto de teste (uma linha por fight_id -- as duas
    linhas espelhadas contam como UMA luta) com a fonte de odds, por data
    (+-_DATE_TOLERANCE_DAYS) e pelos nomes dos dois lutadores.

    Devolve um DataFrame com, por luta casada:
      - prob. do modelo para o lado FAVORITO do mercado
      - prob. implicita de mercado do favorito, sem vig (remove_vig_two_way)
      - se o favorito de fato venceu
    """
    preds = preds_df.drop_duplicates("fight_id").copy()
    preds["event_date"] = pd.to_datetime(preds["event_date"])
    preds["_a_norm"] = preds["fighter_a"].map(_normalize_name)
    preds["_b_norm"] = preds["fighter_b"].map(_normalize_name)

    rows = []
    n_sem_odds = 0
    for _, pred in preds.iterrows():
        window = odds_df[
            (odds_df["event_date"] - pred["event_date"]).abs()
            <= pd.Timedelta(days=_DATE_TOLERANCE_DAYS)
        ]
        odds_row = _match_fight(pred["_a_norm"], pred["_b_norm"], window)
        if odds_row is None:
            n_sem_odds += 1
            continue

        # De que lado (a ou b) esta o favorito do mercado?
        sim_a = SequenceMatcher(None, pred["_a_norm"], odds_row["_fav_norm"]).ratio()
        sim_b = SequenceMatcher(None, pred["_b_norm"], odds_row["_fav_norm"]).ratio()
        fav_is_a = sim_a >= sim_b

        model_prob_fav = pred["pred_gbm"] if fav_is_a else 1.0 - pred["pred_gbm"]
        model_prob_fav_logreg = pred["pred_logreg"] if fav_is_a else 1.0 - pred["pred_logreg"]
        fav_won = int(pred["label"] == 1) if fav_is_a else int(pred["label"] == 0)

        implied_fav = decimal_odds_to_implied_prob(float(odds_row["favourite_odds"]))
        implied_dog = decimal_odds_to_implied_prob(float(odds_row["underdog_odds"]))
        market_prob_fav, _ = remove_vig_two_way(implied_fav, implied_dog)

        rows.append({
            "fight_id": pred["fight_id"],
            "event_date": pred["event_date"],
            "fighter_a": pred["fighter_a"],
            "fighter_b": pred["fighter_b"],
            "favourite": odds_row["favourite"],
            "favourite_odds": odds_row["favourite_odds"],
            "underdog_odds": odds_row["underdog_odds"],
            "overround_pct": round((implied_fav + implied_dog - 1) * 100, 2),
            "model_prob_fav": model_prob_fav,
            "model_prob_fav_logreg": model_prob_fav_logreg,
            "market_prob_fav": market_prob_fav,
            "fav_won": fav_won,
        })

    matched = pd.DataFrame(rows)
    logger.info("Casamento: %d de %d lutas de teste com odds encontradas (%d sem odds).",
                len(matched), len(preds), n_sem_odds)
    return matched


def report_comparison(matched: pd.DataFrame) -> dict:
    """Log loss / Brier / acuracia do modelo e do mercado, lado a lado, com as ressalvas devidas."""
    if matched.empty:
        logger.warning("Nenhuma luta casada -- nada a comparar.")
        return {}

    results = {
        "n_fights": int(len(matched)),
        "date_range": [str(matched["event_date"].min().date()), str(matched["event_date"].max().date())],
        "gbm": compute_metrics(matched["fav_won"], matched["model_prob_fav"]),
        "logreg": compute_metrics(matched["fav_won"], matched["model_prob_fav_logreg"]),
        "market": compute_metrics(matched["fav_won"], matched["market_prob_fav"]),
        "market_overround_mean_pct": float(matched["overround_pct"].mean()),
        "favourite_win_rate": float(matched["fav_won"].mean()),
    }

    logger.info("--- Modelo vs. Mercado: %d lutas, %s a %s ---",
                results["n_fights"], results["date_range"][0], results["date_range"][1])
    logger.info("(taxa de vitoria do favorito na amostra: %.1f%%; overround medio: %.1f%%)",
                results["favourite_win_rate"] * 100, results["market_overround_mean_pct"])
    for name in ("gbm", "logreg", "market"):
        m = results[name]
        logger.info("%-8s log_loss=%.4f  brier=%.4f  accuracy=%.3f",
                    name, m["log_loss"], m["brier_score"], m["accuracy"])

    melhor_ll = min(("gbm", "logreg", "market"), key=lambda k: results[k]["log_loss"])
    if melhor_ll == "market":
        logger.info("O MERCADO teve log loss menor que ambos os modelos nesta amostra "
                    "(resultado esperado: casas de apostas incorporam mais informacao).")
    else:
        logger.info("O modelo '%s' teve log loss menor que o mercado NESTA AMOSTRA. "
                    "Isso NAO e prova de edge: verifique tamanho da amostra, periodo e "
                    "lembre que odds de fechamento reais sao dificeis de bater de forma sustentada.",
                    melhor_ll)
    return results


def run_market_comparison() -> dict:
    """
    Ponto de entrada: compara modelo vs. mercado no maior conjunto
    out-of-sample possivel dado o alcance das odds.

    Se as odds cobrirem parte do conjunto de teste de producao
    (data/processed/test_predictions.csv), usa essas lutas diretamente.
    Caso contrario (situacao atual: odds terminam ANTES do teste de
    producao comecar), roda um backtest dedicado truncando as features no
    fim da janela de odds e re-treinando o pipeline em memoria.
    """
    odds = load_odds_fights()
    odds_max_date = odds["event_date"].max()

    preds_path = config.PROCESSED_DIR / "test_predictions.csv"
    preds_producao = None
    if preds_path.exists():
        preds_producao = pd.read_csv(preds_path, parse_dates=["event_date"])
        overlap = (preds_producao["event_date"] <= odds_max_date).sum()
        if overlap > 0:
            logger.info("Usando o conjunto de teste de producao (%d linhas dentro da janela de odds).",
                        overlap)
            matched = match_odds_to_predictions(preds_producao, odds)
            results = report_comparison(matched)
            matched.to_csv(config.MARKET_COMPARISON_CSV, index=False)
            return results
        logger.warning(
            "Sobreposicao ZERO: o teste de producao comeca em %s, mas as odds terminam em %s. "
            "Rodando backtest dedicado com dados truncados em %s (mesmo pipeline, "
            "sem sobrescrever artefatos de producao).",
            preds_producao["event_date"].min().date(), odds_max_date.date(), odds_max_date.date(),
        )

    # --- Backtest dedicado: trunca no fim da janela de odds e re-treina ---
    from src.train import train_and_calibrate
    feature_df = pd.read_csv(config.FEATURES_CSV, parse_dates=["event_date"])
    truncated = feature_df[feature_df["event_date"] <= odds_max_date].copy()
    logger.info("Backtest: %d lutas ate %s (de %d no total).",
                truncated["fight_id"].nunique(), odds_max_date.date(),
                feature_df["fight_id"].nunique())
    _, backtest_preds = train_and_calibrate(truncated, save_artifacts=False)

    matched = match_odds_to_predictions(backtest_preds, odds)
    results = report_comparison(matched)
    if not matched.empty:
        matched.to_csv(config.MARKET_COMPARISON_CSV, index=False)
        logger.info("Detalhe por luta salvo em %s", config.MARKET_COMPARISON_CSV)
    return results


if __name__ == "__main__":
    run_market_comparison()
