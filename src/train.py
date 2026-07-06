"""
src/train.py

Treina os dois modelos de classificacao binaria (vencedor da luta) pedidos
no escopo:
  - Regressao logistica (baseline simples e interpretavel)
  - Gradient boosting (LightGBM, com fallback para XGBoost e depois para
    HistGradientBoostingClassifier do sklearn, caso nenhuma das duas
    bibliotecas esteja instalada)

Split TEMPORAL (nunca aleatorio): os dados sao ordenados por data e
divididos em treino (mais antigo) / calibracao / teste (mais recente).
Um split aleatorio misturaria lutas futuras no treino e lutas passadas no
teste, inflando artificialmente as metricas (vazamento temporal).

Calibracao: cada modelo e treinado no conjunto de treino e depois
CALIBRADO (Platt scaling / sigmoid, ou isotonic regression se houver dados
suficientes) usando uma fatia separada de calibracao, para que uma
probabilidade prevista de 70% realmente corresponda a ~70% de frequencia
empirica de vitoria.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.features import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _get_gbm_model():
    """
    Tenta usar LightGBM; se nao estiver instalado, tenta XGBoost; se
    nenhum dos dois estiver disponivel, cai para o
    HistGradientBoostingClassifier do proprio scikit-learn (que tambem
    lida nativamente com valores faltantes / NaN, entao continua
    funcionando sem imputacao mesmo nesse caminho alternativo).
    """
    try:
        from lightgbm import LGBMClassifier
        logger.info("Usando LightGBM como modelo de gradient boosting.")
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=4,
            num_leaves=15, subsample=0.8, colsample_bytree=0.8,
            random_state=config.RANDOM_SEED, verbosity=-1,
        ), "lightgbm"
    except ImportError:
        pass
    try:
        from xgboost import XGBClassifier
        logger.info("LightGBM nao encontrado; usando XGBoost.")
        return XGBClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=config.RANDOM_SEED,
        ), "xgboost"
    except ImportError:
        pass
    from sklearn.ensemble import HistGradientBoostingClassifier
    logger.warning("Nem LightGBM nem XGBoost instalados; usando HistGradientBoostingClassifier (sklearn).")
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.03, max_depth=4, random_state=config.RANDOM_SEED,
    ), "sklearn_hgb"


def temporal_group_split(df: pd.DataFrame):
    """
    Faz o split temporal por LUTA (fight_id), nunca por linha isolada --
    essencial porque cada luta gera DUAS linhas espelhadas (A-B e B-A) que
    precisam ficar sempre no mesmo lado do split. Sem isso, o modelo
    poderia efetivamente "ver" o resultado de uma luta no treino (numa das
    linhas espelhadas) e ser calibrado/testado com a linha espelhada da
    MESMA luta -- um vazamento de dados sutil, mas real.
    """
    df = df.sort_values("event_date")
    unique_fights = df.drop_duplicates("fight_id")[["fight_id", "event_date"]].sort_values("event_date")
    n = len(unique_fights)
    n_train = int(n * config.TRAIN_FRACTION)
    n_cal = int(n * config.CALIBRATION_FRACTION)

    train_ids = set(unique_fights["fight_id"].iloc[:n_train])
    cal_ids = set(unique_fights["fight_id"].iloc[n_train:n_train + n_cal])
    test_ids = set(unique_fights["fight_id"].iloc[n_train + n_cal:])

    train_df = df[df["fight_id"].isin(train_ids)].copy()
    cal_df = df[df["fight_id"].isin(cal_ids)].copy()
    test_df = df[df["fight_id"].isin(test_ids)].copy()

    logger.info(
        "Split temporal -> treino: %d lutas | calibracao: %d lutas | teste: %d lutas (periodo total: %s a %s)",
        len(train_ids), len(cal_ids), len(test_ids),
        unique_fights["event_date"].min().date(), unique_fights["event_date"].max().date(),
    )
    return train_df, cal_df, test_df


def build_logreg_pipeline() -> Pipeline:
    """Regressao logistica com imputacao (medias faltantes) e padronizacao das features."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=config.RANDOM_SEED)),
    ])


def split_calibration_slice(cal_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide a fatia de calibracao em duas sub-fatias TEMPORAIS (cal_fit /
    cal_select), agrupadas por fight_id (linhas espelhadas nunca se separam),
    para selecao de hiperparametros SEM tocar no conjunto de teste.

    REGRA METODOLOGICA (documentada tambem no README): nenhuma escolha de
    hiperparametro pode ser feita olhando o teste de producao nem o backtest
    de mercado -- isso seria overfitar o numero que reportamos como avaliacao
    final. Toda selecao (metodo de calibracao, K do Elo, multiplicadores de
    margem) compara alternativas em cal_select e SO ENTAO a vencedora e
    re-treinada com a fatia de calibracao inteira e avaliada uma unica vez.
    """
    unique_fights = cal_df.drop_duplicates("fight_id")[["fight_id", "event_date"]].sort_values("event_date")
    n_fit = len(unique_fights) // 2
    fit_ids = set(unique_fights["fight_id"].iloc[:n_fit])
    cal_fit = cal_df[cal_df["fight_id"].isin(fit_ids)]
    cal_select = cal_df[~cal_df["fight_id"].isin(fit_ids)]
    return cal_fit, cal_select


def select_calibration_method(fitted_estimator, X_fit, y_fit, X_select, y_select) -> str:
    """
    Escolhe sigmoid (Platt) vs isotonic PARA UM MODELO ESPECIFICO, por log
    loss em cal_select (calibrando so em cal_fit). Racional: isotonic tem
    mais capacidade e pode "decorar" as caudas da curva com pouca massa de
    dado -- exatamente o sintoma que ja observamos no GBM (log loss pior do
    que a acuracia sugeria). Deixar os dados decidirem por modelo, em vez de
    uma regra unica por tamanho da fatia.
    """
    from sklearn.metrics import log_loss as sk_log_loss

    if len(X_select) < 50:  # sub-fatia pequena demais para uma escolha confiavel
        return "sigmoid"
    scores = {}
    for method in ("sigmoid", "isotonic"):
        calibrated = _calibrate(fitted_estimator, X_fit, y_fit, method)
        probs = np.clip(calibrated.predict_proba(X_select)[:, 1], 1e-6, 1 - 1e-6)
        scores[method] = sk_log_loss(y_select, probs, labels=[0, 1])
    # empate exato favorece sigmoid (mais simples/estavel), por ordem do dict
    return min(scores, key=scores.get)


def _calibrate(fitted_estimator, X_cal, y_cal, method: str):
    """
    Calibra um estimador JA TREINADO usando uma fatia de dados separada.
    Compativel tanto com versoes novas do scikit-learn (>=1.6, que usa
    FrozenEstimator) quanto com versoes mais antigas (cv="prefit").
    """
    try:
        from sklearn.frozen import FrozenEstimator  # scikit-learn >= 1.6
        calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_estimator), method=method)
    except ImportError:
        calibrated = CalibratedClassifierCV(fitted_estimator, method=method, cv="prefit")
    calibrated.fit(X_cal, y_cal)
    return calibrated


def train_and_calibrate(feature_df: pd.DataFrame | None = None,
                        save_artifacts: bool = True) -> tuple[dict, pd.DataFrame]:
    """
    Pipeline de treino completo:
      1. Carrega as features (ou recebe um DataFrame ja pronto)
      2. Split temporal treino / calibracao / teste
      3. Treina regressao logistica (baseline) e gradient boosting
      4. Calibra as probabilidades de ambos usando a fatia de calibracao
      5. Salva os modelos calibrados + metadados em models/
      6. Salva as predicoes no conjunto de teste (data/processed/test_predictions.csv)
         para avaliacao posterior em src/evaluate.py

    save_artifacts=False roda tudo em memoria sem gravar modelos/predicoes --
    usado por backtests (ex.: comparacao com odds historicas em
    src/market_odds.py) para nao sobrescrever os artefatos de producao.

    Retorna (metadata, test_predictions).
    """
    if feature_df is None:
        feature_df = pd.read_csv(config.FEATURES_CSV, parse_dates=["event_date"])

    train_df, cal_df, test_df = temporal_group_split(feature_df)

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Split temporal resultou em treino ou teste vazio. Voce tem dados suficientes? "
            "Verifique data/processed/fight_features.csv."
        )

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    X_cal, y_cal = cal_df[FEATURE_COLUMNS], cal_df["label"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["label"]

    # Selecao do metodo de calibracao POR MODELO via cal_fit/cal_select
    # (ver docstring de split_calibration_slice). A escolha nunca ve o teste.
    cal_fit, cal_select = split_calibration_slice(cal_df)
    Xf, yf = cal_fit[FEATURE_COLUMNS], cal_fit["label"]
    Xs, ys = cal_select[FEATURE_COLUMNS], cal_select["label"]

    # --- Regressao logistica (baseline) ---
    logreg = build_logreg_pipeline()
    logreg.fit(X_train, y_train)
    logreg_method = select_calibration_method(logreg, Xf, yf, Xs, ys)
    logger.info("Calibracao da logreg: %s (escolhido em cal_select, %d linhas)", logreg_method, len(cal_select))
    logreg_calibrated = _calibrate(logreg, X_cal, y_cal, logreg_method)

    # --- Gradient boosting ---
    gbm, gbm_name = _get_gbm_model()
    gbm.fit(X_train, y_train)
    if hasattr(gbm, "feature_importances_"):
        importances = sorted(zip(FEATURE_COLUMNS, gbm.feature_importances_),
                             key=lambda x: x[1], reverse=True)
        logger.info("Importancia das features (%s): %s", gbm_name,
                    ", ".join(f"{nome}={imp:.0f}" for nome, imp in importances))
    gbm_method = select_calibration_method(gbm, Xf, yf, Xs, ys)
    logger.info("Calibracao do GBM: %s (escolhido em cal_select, %d linhas)", gbm_method, len(cal_select))
    gbm_calibrated = _calibrate(gbm, X_cal, y_cal, gbm_method)

    # --- Predicoes no conjunto de teste (nunca usado em treino/calibracao) ---
    test_predictions = test_df[["fight_id", "event_date", "fighter_a", "fighter_b", "label"]].copy()
    test_predictions["pred_logreg"] = logreg_calibrated.predict_proba(X_test)[:, 1]
    test_predictions["pred_gbm"] = gbm_calibrated.predict_proba(X_test)[:, 1]

    if save_artifacts:
        joblib.dump(logreg_calibrated, config.LOGREG_MODEL_PATH)
        joblib.dump(gbm_calibrated, config.GBM_MODEL_PATH)
        predictions_path = config.PROCESSED_DIR / "test_predictions.csv"
        test_predictions.to_csv(predictions_path, index=False)
        with open(config.FEATURE_LIST_PATH, "w", encoding="utf-8") as f:
            json.dump(FEATURE_COLUMNS, f, indent=2)

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "gbm_model_type": gbm_name,
        "calibration_method": {"logreg": logreg_method, "gbm": gbm_method},
        "n_train_fights": int(train_df["fight_id"].nunique()),
        "n_calibration_fights": int(cal_df["fight_id"].nunique()),
        "n_test_fights": int(test_df["fight_id"].nunique()),
        "train_date_range": [str(train_df["event_date"].min().date()), str(train_df["event_date"].max().date())],
        "test_date_range": [str(test_df["event_date"].min().date()), str(test_df["event_date"].max().date())],
        "feature_columns": FEATURE_COLUMNS,
    }
    if save_artifacts:
        with open(config.TRAINING_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Modelos salvos em %s e %s", config.LOGREG_MODEL_PATH, config.GBM_MODEL_PATH)
        logger.info("Predicoes de teste salvas em %s -- rode 'python -m src.evaluate' para as metricas.",
                    config.PROCESSED_DIR / "test_predictions.csv")
    return metadata, test_predictions


if __name__ == "__main__":
    train_and_calibrate()
