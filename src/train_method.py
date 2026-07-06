"""
src/train_method.py

Fase 2 do projeto: previsao de METODO de vitoria (KO/TKO, finalizacao,
decisao) e de ROUND de finalizacao. Mesmo rigor do preditor de vencedor:
split temporal, calibracao em fatia propria, avaliacao honesta contra
baseline ingenuo.

Decisoes de modelagem:

  - Labels de metodo vem de features.categorize_method sobre o texto livre
    de fights.csv (3 classes; DQ/overturned/sem categoria ficam FORA).
  - Duracao e um problema CONDICIONAL: para DECISAO o round final e sempre
    o scheduled_rounds (trivial, sem modelo). O modelo de round e treinado
    SO nas finalizacoes (KO/TKO + submissao), multiclasse round 1..5 --
    rounds 4-5 sao raros; o tamanho de amostra por classe e reportado e a
    limitacao e explicita.
  - As labels sao SIMETRICAS (nao dependem de quem e "A" ou "B"). O
    dataset espelhado duplica cada luta com o mesmo label: no TREINO isso
    e inocuo (duplicar sinal), mas CALIBRACAO e TESTE sao DEDUPLICADOS
    para uma linha por luta real -- sem isso as metricas contariam cada
    luta duas vezes.
  - Features: exatamente FEATURE_COLUMNS do preditor de vencedor (mesmas
    diferenciais point-in-time ja validadas). Features novas so se a
    validacao mostrar necessidade -- sem overengineering de cara.
  - Cobertura de fontes (verificada em jul/2026): o formato canonico
    "scrape" (github-mirror/scrape) tem method e round 100% preenchidos
    (98.7% categorizaveis) e scheduled_rounds em 97.2% (NaN: ~220 lutas
    de formatos antigos com overtime + as vindas do --fill-gap, cuja
    pagina de evento nao expoe o time format). O fallback public-dataset
    (Kaggle) NAO tem metodo por luta -- esta fase exige a fonte principal.

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
FINISH_CLASSES = ["KO_TKO", "SUBMISSION"]

# Faixas de round de finalizacao: {1, 2, 3+}. Racional (2a iteracao):
#   - round individual (1..5) nao se sustenta (rounds 4-5 tinham 9 e 4 lutas
#     de suporte no teste -- ruido);
#   - o agrupamento anterior {1, 2-3, 4-5} nao separava round 2 de round 3,
#     o que impedia calcular a linha over/under 2,5 do mercado de duracao;
#   - {1, 2, 3+} da as duas coisas: suporte razoavel em todas as classes
#     (~315/196/115 no teste) E as linhas 1,5/2,5 sem forcar numero. Bonus:
#     as 3 faixas valem para QUALQUER formato de luta (round 3+ existe tanto
#     em luta de 3 quanto de 5 rounds), entao a restricao logica antiga de
#     zerar "4-5" em luta de 3 rounds deixou de ser necessaria.
ROUND_BANDS = ["1", "2", "3+"]


def round_to_band(finish_round) -> str | None:
    """Agrupa o round de finalizacao (1..5) nas 3 faixas de ROUND_BANDS."""
    try:
        r = int(finish_round)
    except (TypeError, ValueError):
        return None
    if r <= 0:
        return None
    if r == 1:
        return "1"
    if r == 2:
        return "2"
    return "3+"


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
    Junta o dataset de features (espelhado, 2 linhas por luta) com as
    labels simetricas de metodo/round/scheduled_rounds vindas de fights.csv.
    Lutas sem metodo categorizavel (DQ, overturned, NC) ficam fora.
    """
    feature_df = pd.read_csv(config.FEATURES_CSV, parse_dates=["event_date"])
    fights = pd.read_csv(config.RAW_FIGHTS_CSV, parse_dates=["event_date"])
    fights = fights.reset_index(drop=True)
    fights["fight_id"] = fights["fight_url"].fillna(
        fights.index.to_series().astype(str) + "_" + fights["event_name"].astype(str))

    fights["method_class"] = fights["method"].map(categorize_method)
    fights["finish_round"] = pd.to_numeric(fights["round"], errors="coerce")
    if "scheduled_rounds" not in fights.columns:
        fights["scheduled_rounds"] = np.nan
    # inferencia segura: decisao termina no round agendado (validado: 99.98%
    # das decisoes com scheduled_rounds conhecido batem) -- preenche so os NaN
    is_dec = fights["method_class"] == "DECISION"
    fights.loc[is_dec & fights["scheduled_rounds"].isna(), "scheduled_rounds"] = \
        fights.loc[is_dec & fights["scheduled_rounds"].isna(), "finish_round"]

    labels = fights[["fight_id", "method_class", "finish_round", "scheduled_rounds"]]
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


def train_method_and_round(df: pd.DataFrame | None = None, save_artifacts: bool = True) -> dict:
    """
    Pipeline completo da fase 2:
      1. metodo (3 classes) em todas as lutas com metodo categorizavel;
      2. round de finalizacao (1..5) SO nas finalizacoes.
    Treino usa as linhas espelhadas (2x por luta, label igual); calibracao
    e teste sao deduplicados por fight_id.
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

    # --- 1) metodo ---
    method_models, method_metrics = _train_pair(train_df, cal_df, test_df,
                                                "method_class", METHOD_CLASSES, "metodo",
                                                feature_cols)

    # --- 2) round de finalizacao (condicional: so finalizacoes), em 3 FAIXAS ---
    fin_train = train_df[train_df["method_class"].isin(FINISH_CLASSES)].copy()
    fin_cal = cal_df[cal_df["method_class"].isin(FINISH_CLASSES)].copy()
    fin_test = test_df[test_df["method_class"].isin(FINISH_CLASSES)].copy()
    for part in (fin_train, fin_cal, fin_test):
        part["round_band"] = part["finish_round"].map(round_to_band)
    round_classes = ROUND_BANDS
    logger.info("Suporte por faixa de round no teste: %s",
                fin_test["round_band"].value_counts().to_dict())
    # O modelo de faixa recebe scheduled_rounds como FEATURE (3 vs 5 rounds
    # muda o espaco de resultados possiveis; 97.2% de cobertura, NaN imputado).
    # Alem disso, a predicao aplica uma restricao logica na saida: luta de 3
    # rounds zera a faixa "4-5" (ver predict.constrain_round_bands).
    round_feature_cols = feature_cols + ["scheduled_rounds"]
    round_models, round_metrics = _train_pair(fin_train, fin_cal, fin_test,
                                              "round_band", round_classes, "round",
                                              round_feature_cols)

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "feature_columns": feature_cols,
        "round_feature_columns": round_feature_cols,
        "method": method_metrics,
        "finish_round": round_metrics,
        "round_classes": round_classes,
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
        joblib.dump(round_models["logreg"], config.ROUND_LOGREG_MODEL_PATH)
        joblib.dump(round_models["gbm"], config.ROUND_GBM_MODEL_PATH)
        with open(config.METHOD_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Modelos de metodo/round salvos em %s", config.MODELS_DIR)
    return metadata


if __name__ == "__main__":
    train_method_and_round()
