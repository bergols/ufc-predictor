"""
src/tuning.py

Experimentos de hiperparametros com selecao HONESTA (jul/2026):

REGRA METODOLOGICA: nenhuma escolha aqui olha o conjunto de teste de
producao (2023-12+) nem as 821 lutas do backtest de mercado. Cada
candidato e avaliado em cal_select -- a metade mais recente da fatia de
calibracao -- e o vencedor so depois e re-treinado e avaliado uma unica
vez nas avaliacoes finais.

DETALHE IMPORTANTE: a fatia de calibracao de PRODUCAO (~2021-2023) se
sobrepoe a janela de teste do backtest de mercado (ago/2021-set/2023).
Escolher K/margem em cima dela contaminaria o numero final do backtest.
Por isso os experimentos rodam sobre o dataset TRUNCADO no fim da janela
de odds (2023-09-16), cujo cal_select termina ANTES da janela das 821
lutas -- limpo para as duas avaliacoes finais.

Uso:
    python -m src.tuning            # roda a grade de K e os esquemas de margem
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as sk_log_loss

import config
from src.features import FEATURE_COLUMNS
from src.ratings import compute_elo_ratings
from src.train import (
    _calibrate,
    _get_gbm_model,
    build_logreg_pipeline,
    split_calibration_slice,
    temporal_group_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Fim da janela de odds do backtest (jansen88/ufc-data) -- os experimentos
# truncam aqui para que cal_select nunca encoste na janela do backtest.
BACKTEST_TRUNCATE_DATE = pd.Timestamp("2023-09-16")

ELO_K_GRID = [16, 24, 32, 40, 64]

# Esquemas de margem por metodo (multiplicadores de K):
#   leve:      finalizacao pesa um pouco mais, decisao apertada um pouco menos
#   agressivo: diferencas maiores
#   so-bonus:  so bonifica finalizacao, sem punir decisao apertada
MARGIN_SCHEMES = {
    "sem-margem": None,
    "leve": {"FINISH": 1.25, "DECISION_CLOSE": 0.75},
    "agressivo": {"FINISH": 1.5, "DECISION_CLOSE": 0.5},
    "so-bonus": {"FINISH": 1.5},
}


def replace_elo_diff(feature_df: pd.DataFrame, fights_df: pd.DataFrame,
                     k: float, method_multipliers: dict | None = None) -> pd.DataFrame:
    """
    Recalcula a coluna elo_diff de um dataset de features ja construido,
    refazendo a passada cronologica inteira do Elo (o historico de rating
    depende do K/margem, entao nao da para "escalar" o Elo antigo).
    As demais features nao dependem do Elo e ficam intactas.
    """
    elo_pre, _ = compute_elo_ratings(fights_df, k=k, method_multipliers=method_multipliers)
    merged = feature_df.merge(fights_df[["fight_id", "fighter_1"]], on="fight_id", how="left")
    merged = merged.merge(elo_pre, on="fight_id", how="left")
    out = feature_df.copy()
    out["elo_diff"] = np.where(merged["fighter_a"] == merged["fighter_1"],
                               merged["elo_1_pre"] - merged["elo_2_pre"],
                               merged["elo_2_pre"] - merged["elo_1_pre"])
    return out


def score_on_cal_select(feature_df: pd.DataFrame) -> dict:
    """
    Treina logreg + GBM na fatia de treino, calibra (sigmoid fixo, para a
    comparacao entre candidatos ser justa e nao interagir com o experimento
    de metodo de calibracao) em cal_fit e mede log loss em cal_select.
    O conjunto de teste do split NUNCA e tocado aqui.
    """
    train_df, cal_df, _test_df_nunca_usado = temporal_group_split(feature_df)
    cal_fit, cal_select = split_calibration_slice(cal_df)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    Xf, yf = cal_fit[FEATURE_COLUMNS], cal_fit["label"]
    Xs, ys = cal_select[FEATURE_COLUMNS], cal_select["label"]

    scores = {}
    logreg = build_logreg_pipeline()
    logreg.fit(X_train, y_train)
    probs = np.clip(_calibrate(logreg, Xf, yf, "sigmoid").predict_proba(Xs)[:, 1], 1e-6, 1 - 1e-6)
    scores["logreg"] = sk_log_loss(ys, probs, labels=[0, 1])

    gbm, _ = _get_gbm_model()
    gbm.fit(X_train, y_train)
    probs = np.clip(_calibrate(gbm, Xf, yf, "sigmoid").predict_proba(Xs)[:, 1], 1e-6, 1 - 1e-6)
    scores["gbm"] = sk_log_loss(ys, probs, labels=[0, 1])

    scores["media"] = (scores["logreg"] + scores["gbm"]) / 2
    return scores


def choose_winner(scores_by_name: dict[str, float], prefer_order: list[str]) -> str:
    """
    Menor score vence; empate (ate 4 casas) vai para quem vem primeiro em
    prefer_order -- a opcao mais simples. Nao adicionar complexidade que
    nao se paga.
    """
    best = min(scores_by_name.values())
    for name in prefer_order:
        if name in scores_by_name and round(scores_by_name[name] - best, 4) <= 0:
            return name
    return min(scores_by_name, key=scores_by_name.get)


def _load_truncated() -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_df = pd.read_csv(config.FEATURES_CSV, parse_dates=["event_date"])
    fights_df = pd.read_csv(config.RAW_FIGHTS_CSV, parse_dates=["event_date"])
    # mesmo fight_id que normalize_scrape_data cria (URL da luta)
    fights_df = fights_df.reset_index(drop=True)
    fights_df["fight_id"] = fights_df["fight_url"].fillna(
        fights_df.index.to_series().astype(str) + "_" + fights_df["event_name"].astype(str)
    )
    feature_df = feature_df[feature_df["event_date"] <= BACKTEST_TRUNCATE_DATE]
    fights_df = fights_df[fights_df["event_date"] <= BACKTEST_TRUNCATE_DATE]
    return feature_df, fights_df


def tune_elo_k(ks: list[float] | None = None) -> tuple[float, dict]:
    """Experimento 2: grade de K-factor do Elo, avaliada em cal_select."""
    ks = ks or ELO_K_GRID
    feature_df, fights_df = _load_truncated()
    results = {}
    for k in ks:
        candidate = replace_elo_diff(feature_df, fights_df, k=k, method_multipliers=None)
        scores = score_on_cal_select(candidate)
        results[k] = scores
        logger.info("K=%-3d -> cal_select log loss: logreg=%.4f gbm=%.4f media=%.4f",
                    k, scores["logreg"], scores["gbm"], scores["media"])
    medias = {k: v["media"] for k, v in results.items()}
    # empate favorece o K atual de producao (32), depois a ordem da grade
    prefer = [32] + [k for k in ks if k != 32]
    winner = choose_winner(medias, prefer)
    logger.info("K vencedor em cal_select: %s", winner)
    return winner, results


def tune_elo_margin(k: float, schemes: dict | None = None) -> tuple[str, dict]:
    """Experimento 3: esquemas de margem por metodo (com o K vencedor), em cal_select."""
    schemes = schemes or MARGIN_SCHEMES
    feature_df, fights_df = _load_truncated()
    results = {}
    for name, multipliers in schemes.items():
        candidate = replace_elo_diff(feature_df, fights_df, k=k, method_multipliers=multipliers)
        scores = score_on_cal_select(candidate)
        results[name] = scores
        logger.info("margem '%s' -> cal_select log loss: logreg=%.4f gbm=%.4f media=%.4f",
                    name, scores["logreg"], scores["gbm"], scores["media"])
    medias = {name: v["media"] for name, v in results.items()}
    # empate favorece "sem-margem" (mais simples)
    winner = choose_winner(medias, ["sem-margem"] + [n for n in schemes if n != "sem-margem"])
    logger.info("Esquema de margem vencedor em cal_select: '%s'", winner)
    return winner, results


if __name__ == "__main__":
    logger.info("=== Experimento 2: grade de K do Elo (selecao em cal_select, dados truncados em %s) ===",
                BACKTEST_TRUNCATE_DATE.date())
    k_winner, _ = tune_elo_k()
    logger.info("=== Experimento 3: margem por metodo (K=%s) ===", k_winner)
    margin_winner, _ = tune_elo_margin(k_winner)
    logger.info("=== RESULTADO: K=%s, margem='%s'. Atualize config.py de acordo e re-treine. ===",
                k_winner, margin_winner)
