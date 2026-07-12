"""
src/evaluate.py

Calcula as metricas de avaliacao pedidas no escopo -- log loss, Brier
score e acuracia -- sobre o conjunto de TESTE (a fatia mais recente,
nunca vista durante treino ou calibracao). Tambem compara as
probabilidades do modelo com odds de mercado fornecidas manualmente
(ver data/odds_template.csv).

Este modulo NAO tenta forcar uma conclusao otimista. MMA tem alta
variancia; superar (ou nao) o mercado em uma amostra pequena de eventos
nao e prova estatistica de "edge" real. O objetivo e reportar os numeros
com honestidade.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

import config
from src.utils import decimal_odds_to_implied_prob, remove_vig_two_way

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_prob) -> dict:
    """Log loss, Brier score e acuracia (limiar 0.5) para um vetor de probabilidades previstas."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def evaluate_test_set(predictions_path: Path | None = None) -> dict:
    """
    Le as predicoes de teste geradas por src/train.py e calcula as
    metricas para cada modelo (regressao logistica e gradient boosting).
    """
    predictions_path = predictions_path or (config.PROCESSED_DIR / "test_predictions.csv")
    if not predictions_path.exists():
        raise FileNotFoundError(f"{predictions_path} nao existe. Rode primeiro: python -m src.train")

    df = pd.read_csv(predictions_path)
    results = {
        "logreg": compute_metrics(df["label"], df["pred_logreg"]),
        "gbm": compute_metrics(df["label"], df["pred_gbm"]),
    }
    for model_name, metrics in results.items():
        logger.info("[%s] n=%d  accuracy=%.3f  log_loss=%.3f  brier=%.3f",
                    model_name, metrics["n"], metrics["accuracy"], metrics["log_loss"], metrics["brier_score"])

    baseline_rate = df["label"].mean()
    logger.info(
        "Referencia: prever sempre o favorito por experiencia/forma nao e trivial de calcular aqui, "
        "mas a taxa base de vitoria de 'fighter_a' no teste e %.1f%% (esperado ~50%% dado o espelhamento).",
        baseline_rate * 100,
    )
    return results


def compare_to_market(odds_csv_path: Path | None = None, predictions_path: Path | None = None,
                      model_name: str = "gbm") -> pd.DataFrame:
    """
    Compara as probabilidades do modelo com as odds implicitas do mercado
    (apos remover o overround/vig) para lutas em que voce preencheu
    manualmente o template data/odds_template.csv.

    Duas vias de previsao, escolhidas por luta:

    1. Luta presente no conjunto de TESTE (test_predictions.csv): usa a
       probabilidade point-in-time gravada pelo treino (preferida -- foi
       calculada so com dados anteriores a luta).
    2. Luta POSTERIOR ao fim do teste (caso tipico do paper trading: evento
       que acabou de acontecer, base ainda nao re-treinada): preve AO VIVO
       com o modelo calibrado atual, via src.predict. Isso e valido porque
       o "nivel atual" dos lutadores nao inclui a luta em questao. Lutas
       fora do teste mas ANTERIORES ao fim dele sao puladas -- prever o
       passado com stats de hoje seria vazamento, e nao fingimos que nao.

    IMPORTANTE: isto e apenas uma comparacao descritiva. Casas de apostas
    profissionais incorporam muito mais informacao (lesoes, fluxo de
    apostas, noticias de ultima hora) do que este modelo. Bater o mercado
    em log loss/Brier em 2-3 eventos NAO e evidencia estatisticamente
    robusta de "edge" real -- seria necessaria uma amostra bem maior e,
    idealmente, apostas simuladas (paper trading) ao longo do tempo antes
    de qualquer conclusao mais forte.
    """
    if model_name not in ("logreg", "gbm"):
        raise ValueError(f"model_name deve ser 'logreg' ou 'gbm', recebi {model_name!r}")
    odds_csv_path = odds_csv_path or config.ODDS_TEMPLATE_CSV
    predictions_path = predictions_path or (config.PROCESSED_DIR / "test_predictions.csv")

    odds_df = pd.read_csv(odds_csv_path)
    odds_df = odds_df.dropna(subset=["actual_winner", "odds_a_decimal", "odds_b_decimal"])
    if odds_df.empty:
        logger.warning(
            "Nenhuma linha completa em %s (preencha fighter_a, fighter_b, odds_a_decimal, "
            "odds_b_decimal e actual_winner para 2-3 eventos passados). Pulando comparacao com mercado.",
            odds_csv_path,
        )
        return pd.DataFrame()

    if not predictions_path.exists():
        raise FileNotFoundError(f"{predictions_path} nao existe. Rode primeiro: python -m src.train")
    preds_df = pd.read_csv(predictions_path)
    test_end_date = pd.to_datetime(preds_df["event_date"]).max()

    # Carregados sob demanda na primeira luta que precisar da via ao vivo
    # (evita pagar o custo da base quando todas as lutas estao no teste).
    live_levels = None

    rows = []
    for _, row in odds_df.iterrows():
        match = preds_df[
            ((preds_df["fighter_a"] == row["fighter_a"]) & (preds_df["fighter_b"] == row["fighter_b"]))
            | ((preds_df["fighter_a"] == row["fighter_b"]) & (preds_df["fighter_b"] == row["fighter_a"]))
        ]
        pred_col = f"pred_{model_name}"
        if not match.empty:
            pred_row = match.iloc[0]
            model_prob_a = pred_row[pred_col] if pred_row["fighter_a"] == row["fighter_a"] else 1 - pred_row[pred_col]
            prediction_source = "test_set"
        else:
            event_date = pd.to_datetime(row.get("event_date", None), errors="coerce")
            if pd.isna(event_date) or event_date <= test_end_date:
                logger.warning(
                    "Luta %s vs %s nao esta nas predicoes de teste e nao e posterior ao fim do "
                    "teste (%s) -- pulando (prever o passado com stats atuais seria vazamento). "
                    "Preencha event_date se a luta for recente.",
                    row["fighter_a"], row["fighter_b"], test_end_date.date(),
                )
                continue
            from src.predict import predict_fight
            if live_levels is None:
                from src.features import export_latest_fighter_levels
                live_levels = export_latest_fighter_levels()
            try:
                live = predict_fight(row["fighter_a"], row["fighter_b"],
                                     model_name=model_name, levels=live_levels)
            except ValueError as exc:
                logger.warning("Luta %s vs %s sem previsao ao vivo (%s) -- pulando.",
                               row["fighter_a"], row["fighter_b"], exc)
                continue
            model_prob_a = live["prob_a_wins"]
            prediction_source = "live_model"

        implied_a = decimal_odds_to_implied_prob(float(row["odds_a_decimal"]))
        implied_b = decimal_odds_to_implied_prob(float(row["odds_b_decimal"]))
        market_prob_a, _ = remove_vig_two_way(implied_a, implied_b)

        actual_a_won = int(row["actual_winner"] == row["fighter_a"])

        rows.append({
            "fighter_a": row["fighter_a"],
            "fighter_b": row["fighter_b"],
            "model_prob_a": model_prob_a,
            "market_prob_a_devigged": market_prob_a,
            "market_overround_pct": round((implied_a + implied_b - 1) * 100, 2),
            "actual_a_won": actual_a_won,
            "prediction_source": prediction_source,
        })

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        return comparison_df

    model_metrics = compute_metrics(comparison_df["actual_a_won"], comparison_df["model_prob_a"])
    market_metrics = compute_metrics(comparison_df["actual_a_won"], comparison_df["market_prob_a_devigged"])

    n_live = int((comparison_df["prediction_source"] == "live_model").sum())
    logger.info("--- Modelo (%s) vs. Mercado (%d lutas; %d do teste, %d previstas ao vivo) ---",
                model_name, len(comparison_df), len(comparison_df) - n_live, n_live)
    logger.info("Modelo:  log_loss=%.3f  brier=%.3f  accuracy=%.3f",
                model_metrics["log_loss"], model_metrics["brier_score"], model_metrics["accuracy"])
    logger.info("Mercado: log_loss=%.3f  brier=%.3f  accuracy=%.3f",
                market_metrics["log_loss"], market_metrics["brier_score"], market_metrics["accuracy"])

    if len(comparison_df) < 20:
        logger.info(
            "Amostra pequena (%d lutas) -- qualquer diferenca entre modelo e mercado aqui e "
            "essencialmente ruido estatistico. Nao tire conclusoes de 'edge' com tao pouco dado.",
            len(comparison_df),
        )

    return comparison_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Metricas do teste + comparacao com odds manuais.")
    parser.add_argument("--model", choices=["logreg", "gbm"], default="gbm",
                        help="Modelo usado na comparacao com o mercado (default: gbm)")
    args = parser.parse_args()

    evaluate_test_set()
    compare_to_market(model_name=args.model)
