"""
src/train_method.py

Fase 2 do projeto: previsao de METODO de vitoria (KO/TKO, finalizacao,
decisao). Mesmo rigor do preditor de vencedor: split temporal, calibracao
em fatia propria, avaliacao honesta contra baseline ingenuo.

A previsao de DURACAO (faixa de round de finalizacao + mercado over/under)
existiu aqui ate ago/2026 e foi REMOVIDA. Dois motivos: a margem sobre o
baseline ingenuo era minima, e as probabilidades nao sao congeladas no
historico de pre-registro -- entao regerar um card ja encerrado mexia
nelas com a base ja contendo o resultado da propria luta (ver
prediction_history.frozen_predictions_for_event). Sem odds de casas para
duracao, nao havia como validar contra mercado real tampouco. O metodo
fica, com as mesmas ressalvas de sempre. Historico completo no git.

Decisoes de modelagem:

  - Labels de metodo vem de features.categorize_method sobre o texto livre
    de fights.csv (3 classes; DQ/overturned/sem categoria ficam FORA).
  - As labels sao SIMETRICAS (nao dependem de quem e "A" ou "B"). O
    dataset espelhado duplica cada luta com o mesmo label: no TREINO isso
    e inocuo (duplicar sinal), mas CALIBRACAO e TESTE sao DEDUPLICADOS
    para uma linha por luta real -- sem isso as metricas contariam cada
    luta duas vezes.
  - Features: exatamente FEATURE_COLUMNS do preditor de vencedor (mesmas
    diferenciais point-in-time ja validadas). Features novas so se a
    validacao mostrar necessidade -- sem overengineering de cara.
  - Cobertura de fontes (verificada em jul/2026): o formato canonico
    "scrape" (github-mirror/scrape) tem method 100% preenchido (98.7%
    categorizaveis). O fallback public-dataset (Kaggle) NAO tem metodo por
    luta -- esta fase exige a fonte principal.

Uso:
    python -m src.train_method
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss

import config
from src.features import FEATURE_COLUMNS, SYMMETRIC_SUM_COLUMNS, categorize_method
from src.train import _calibrate, build_logreg_pipeline, temporal_group_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

METHOD_CLASSES = ["KO_TKO", "SUBMISSION", "DECISION"]


def _get_multiclass_gbm():
    """Mesma cadeia de fallback do preditor de vencedor, em modo multiclasse."""
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=4,
                              num_leaves=15, subsample=0.8, colsample_bytree=0.8,
                              random_state=config.RANDOM_SEED, verbosity=-1), "lightgbm"
    except ImportError:
        pass
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                             random_state=config.RANDOM_SEED), "xgboost"
    except ImportError:
        pass
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_depth=4,
                                          random_state=config.RANDOM_SEED), "sklearn_hgb"


def build_method_dataset() -> pd.DataFrame:
    """
    Junta o dataset de features (espelhado, 2 linhas por luta) com a label
    simetrica de metodo vinda de fights.csv. Lutas sem metodo categorizavel
    (DQ, overturned, NC) ficam fora.
    """
    feature_df = pd.read_csv(config.FEATURES_CSV, parse_dates=["event_date"])
    fights = pd.read_csv(config.RAW_FIGHTS_CSV, parse_dates=["event_date"])
    fights = fights.reset_index(drop=True)
    fights["fight_id"] = fights["fight_url"].fillna(
        fights.index.to_series().astype(str) + "_" + fights["event_name"].astype(str))

    fights["method_class"] = fights["method"].map(categorize_method)

    labels = fights[["fight_id", "method_class"]]
    df = feature_df.merge(labels, on="fight_id", how="left")
    n_before = df["fight_id"].nunique()
    df = df.dropna(subset=["method_class"])
    logger.info("Dataset de metodo: %d lutas (%d descartadas sem metodo categorizavel: DQ/overturned/etc.)",
                df["fight_id"].nunique(), n_before - df["fight_id"].nunique())
    return df


def naive_baseline_probs(y_train: pd.Series, classes: list[str]) -> np.ndarray:
    """Baseline ingenuo: sempre prever a distribuicao marginal do TREINO (constante)."""
    freqs = y_train.value_counts(normalize=True)
    return np.array([freqs.get(c, 0.0) for c in classes])


def _evaluate_multiclass(name: str, y_true: pd.Series, probs: np.ndarray,
                         classes: list[str], baseline_probs: np.ndarray) -> dict:
    """Log loss multiclasse, acuracia, matriz de confusao e comparacao com o baseline."""
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_pred = [classes[i] for i in probs.argmax(axis=1)]

    base_matrix = np.tile(baseline_probs, (len(y_true), 1))
    majority_class = classes[int(np.argmax(baseline_probs))]

    # CUIDADO: sklearn.log_loss assume as colunas de y_pred na ordem
    # ALFABETICA de `labels` (LabelBinarizer ordena) -- reordenamos as
    # colunas explicitamente, senao o log loss sai com classes trocadas.
    sorted_classes = sorted(classes)
    sort_idx = [classes.index(c) for c in sorted_classes]

    metrics = {
        "n": int(len(y_true)),
        "log_loss": float(log_loss(y_true, probs[:, sort_idx], labels=sorted_classes)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "baseline_log_loss": float(log_loss(y_true, base_matrix[:, sort_idx], labels=sorted_classes)),
        "baseline_accuracy": float(accuracy_score(y_true, [majority_class] * len(y_true))),
        "baseline_majority_class": majority_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "classes": classes,
        "test_support": {c: int((y_true == c).sum()) for c in classes},
    }
    metrics["beats_baseline_log_loss"] = metrics["log_loss"] < metrics["baseline_log_loss"]
    metrics["beats_baseline_accuracy"] = metrics["accuracy"] > metrics["baseline_accuracy"]

    logger.info("[%s] n=%d  log_loss=%.4f (baseline %.4f)  acc=%.3f (baseline %.3f = sempre '%s')  "
                "bate baseline? log_loss:%s acc:%s",
                name, metrics["n"], metrics["log_loss"], metrics["baseline_log_loss"],
                metrics["accuracy"], metrics["baseline_accuracy"], majority_class,
                "SIM" if metrics["beats_baseline_log_loss"] else "NAO",
                "SIM" if metrics["beats_baseline_accuracy"] else "NAO")
    cm = pd.DataFrame(metrics["confusion_matrix"], index=[f"real_{c}" for c in classes],
                      columns=[f"prev_{c}" for c in classes])
    logger.info("[%s] matriz de confusao:\n%s", name, cm.to_string())
    return metrics


def _train_pair(train_df, cal_df, test_df, label_col: str, classes: list[str],
                tag: str, feature_cols: list[str]) -> tuple[dict, dict]:
    """
    Treina logreg multinomial + GBM multiclasse, calibra (sigmoid, OvR) na
    fatia de calibracao DEDUPLICADA e avalia no teste DEDUPLICADO contra o
    baseline ingenuo. Devolve (modelos, metricas).
    """
    X_train, y_train = train_df[feature_cols], train_df[label_col]
    X_cal, y_cal = cal_df[feature_cols], cal_df[label_col]
    X_test, y_test = test_df[feature_cols], test_df[label_col]

    baseline = naive_baseline_probs(y_train, classes)
    logger.info("[%s] treino=%d linhas | cal=%d lutas | teste=%d lutas | dist treino: %s",
                tag, len(train_df), len(cal_df), len(test_df),
                {c: round(float(p), 3) for c, p in zip(classes, baseline)})

    models, metrics = {}, {}
    logreg = build_logreg_pipeline()
    logreg.fit(X_train, y_train)
    # sigmoid fixo: multiclasse via OvR; isotonic com classes raras (rounds
    # 4-5) decoraria a curva -- mesma logica conservadora ja usada antes
    logreg_cal = _calibrate(logreg, X_cal, y_cal, "sigmoid")
    probs = logreg_cal.predict_proba(X_test)
    order = [list(logreg_cal.classes_).index(c) for c in classes]
    metrics["logreg"] = _evaluate_multiclass(f"{tag}/logreg", y_test, probs[:, order], classes, baseline)
    models["logreg"] = logreg_cal

    gbm, gbm_name = _get_multiclass_gbm()
    gbm.fit(X_train, y_train)
    gbm_cal = _calibrate(gbm, X_cal, y_cal, "sigmoid")
    probs = gbm_cal.predict_proba(X_test)
    order = [list(gbm_cal.classes_).index(c) for c in classes]
    metrics["gbm"] = _evaluate_multiclass(f"{tag}/{gbm_name}", y_test, probs[:, order], classes, baseline)
    metrics["gbm_model_type"] = gbm_name
    models["gbm"] = gbm_cal
    return models, metrics


def train_method(df: pd.DataFrame | None = None, save_artifacts: bool = True) -> dict:
    """
    Pipeline da fase 2: metodo (3 classes) em todas as lutas com metodo
    categorizavel. Treino usa as linhas espelhadas (2x por luta, label
    igual); calibracao e teste sao deduplicados por fight_id.
    """
    if df is None:
        df = build_method_dataset()

    train_df, cal_df, test_df = temporal_group_split(df)
    cal_df = cal_df.drop_duplicates("fight_id")
    test_df = test_df.drop_duplicates("fight_id")

    # Features: as diferenciais do preditor de vencedor + as somas simetricas
    # (validacao mostrou que so as diffs nao carregam sinal de metodo -- ver
    # SYMMETRIC_SUM_COLUMNS em features.py). Se o CSV de features for antigo
    # e nao tiver as somas, roda so com as diffs e avisa.
    sum_cols = [c for c in SYMMETRIC_SUM_COLUMNS if c in df.columns]
    if len(sum_cols) < len(SYMMETRIC_SUM_COLUMNS):
        logger.warning("Features de soma ausentes no CSV (%s) -- re-rode 'python -m src.features'.",
                       set(SYMMETRIC_SUM_COLUMNS) - set(sum_cols))
    feature_cols = FEATURE_COLUMNS + sum_cols

    method_models, method_metrics = _train_pair(train_df, cal_df, test_df,
                                                "method_class", METHOD_CLASSES, "metodo",
                                                feature_cols)

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "feature_columns": feature_cols,
        "method": method_metrics,
        "n_train_fights": int(train_df["fight_id"].nunique()),
        "n_test_fights": int(len(test_df)),
        "train_date_range": [str(train_df["event_date"].min().date()),
                             str(train_df["event_date"].max().date())],
        "test_date_range": [str(test_df["event_date"].min().date()),
                            str(test_df["event_date"].max().date())],
    }

    if save_artifacts:
        joblib.dump(method_models["logreg"], config.METHOD_LOGREG_MODEL_PATH)
        joblib.dump(method_models["gbm"], config.METHOD_GBM_MODEL_PATH)
        with open(config.METHOD_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Modelos de metodo salvos em %s", config.MODELS_DIR)
    return metadata


if __name__ == "__main__":
    train_method()
