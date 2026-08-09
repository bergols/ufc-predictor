"""
Testes da fase 2 (metodo de vitoria): baseline ingenuo, dedupe das fatias
de avaliacao, somas simetricas nas linhas espelhadas e a regressao do bug
de ordem de colunas no log loss multiclasse do sklearn.

A previsao de duracao (faixa de round) saiu em ago/2026 -- ver o cabecalho
de src/train_method.py. parse_scheduled_rounds segue testado porque a
coluna continua sendo coletada como dado bruto.
"""
import numpy as np
import pandas as pd
import pytest

import config
from src.data_collection import parse_scheduled_rounds
from src.features import SYMMETRIC_SUM_COLUMNS
from src.train import temporal_group_split
from src.train_method import (
    METHOD_CLASSES,
    _evaluate_multiclass,
    build_method_dataset,
    naive_baseline_probs,
)


class TestParseScheduledRounds:
    @pytest.mark.parametrize("raw,expected", [
        ("3 Rnd (5-5-5)", 3),
        ("5 Rnd (5-5-5-5-5)", 5),
        ("2 Rnd (5-5)", 2),
        ("1 Rnd (20)", 1),
    ])
    def test_formatos_modernos(self, raw, expected):
        assert parse_scheduled_rounds(raw) == expected

    @pytest.mark.parametrize("raw", [
        "1 Rnd + OT (12-3)", "3 Rnd + OT (5-5-5-5)", "No Time Limit",
        "1 Rnd + 2OT (15-3-3)", None, np.nan, "",
    ])
    def test_formatos_antigos_e_ausentes_viram_none(self, raw):
        assert parse_scheduled_rounds(raw) is None


class TestNaiveBaseline:
    def test_distribuicao_marginal_do_treino(self):
        y = pd.Series(["DECISION"] * 5 + ["KO_TKO"] * 3 + ["SUBMISSION"] * 2)
        probs = naive_baseline_probs(y, METHOD_CLASSES)
        assert probs == pytest.approx([0.3, 0.2, 0.5])  # ordem de METHOD_CLASSES

    def test_classe_ausente_no_treino_vira_zero(self):
        y = pd.Series(["DECISION"] * 4)
        probs = naive_baseline_probs(y, METHOD_CLASSES)
        assert probs == pytest.approx([0.0, 0.0, 1.0])


class TestLogLossColumnOrder:
    def test_regressao_ordem_alfabetica_do_sklearn(self):
        """Regressao do bug encontrado na primeira avaliacao: sklearn.log_loss
        assume colunas em ordem ALFABETICA dos labels. Com classes em ordem
        nao-alfabetica (KO_TKO, SUBMISSION, DECISION), a avaliacao precisa
        reordenar -- este teste compara com o valor calculado a mao."""
        y_true = pd.Series(["KO_TKO", "DECISION"])
        # probs nas colunas na ordem de METHOD_CLASSES = [KO, SUB, DEC]
        probs = np.array([[0.7, 0.1, 0.2],
                          [0.2, 0.1, 0.7]])
        baseline = np.array([1 / 3, 1 / 3, 1 / 3])
        m = _evaluate_multiclass("t", y_true, probs, METHOD_CLASSES, baseline)
        esperado = -(np.log(0.7) + np.log(0.7)) / 2   # acertou 0.7 nas duas
        assert m["log_loss"] == pytest.approx(esperado, abs=1e-4)
        esperado_base = -(np.log(1 / 3) * 2) / 2
        assert m["baseline_log_loss"] == pytest.approx(esperado_base, abs=1e-4)

    def test_acuracia_e_matriz(self):
        y_true = pd.Series(["KO_TKO", "DECISION", "SUBMISSION"])
        probs = np.array([[0.8, 0.1, 0.1],    # preve KO (certo)
                          [0.6, 0.2, 0.2],    # preve KO (errado, era DEC)
                          [0.1, 0.8, 0.1]])   # preve SUB (certo)
        baseline = np.array([0.3, 0.2, 0.5])
        m = _evaluate_multiclass("t", y_true, probs, METHOD_CLASSES, baseline)
        assert m["accuracy"] == pytest.approx(2 / 3)
        assert m["baseline_majority_class"] == "DECISION"
        cm = np.array(m["confusion_matrix"])  # linhas=real, colunas=previsto
        assert cm[0][0] == 1   # KO real previsto KO
        assert cm[2][0] == 1   # DEC real previsto KO


class TestBuildMethodDataset:
    @pytest.fixture
    def dataset(self, tmp_path, monkeypatch):
        """Features espelhadas (2 linhas/luta) + fights.csv sinteticos."""
        n = 12  # lutas, uma por semana
        feat_rows, fight_rows = [], []
        methods = ["KO/TKO", "Submission", "Decision - Unanimous", "Overturned"] * 3
        for i in range(n):
            date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=7 * i)
            url = f"http://u/f/{i}"
            for a, b, label in ((f"A{i}", f"B{i}", 1), (f"B{i}", f"A{i}", 0)):
                feat_rows.append({"fight_id": url, "event_date": date, "fighter_a": a,
                                  "fighter_b": b, "label": label,
                                  **{c: 0.1 for c in __import__('src.features', fromlist=['FEATURE_COLUMNS']).FEATURE_COLUMNS},
                                  **{c: 0.2 for c in SYMMETRIC_SUM_COLUMNS}})
            fight_rows.append({"event_name": "E", "event_date": date, "event_url": "http://u/e",
                               "fight_url": url, "fighter_1": f"A{i}", "fighter_2": f"B{i}",
                               "winner": f"A{i}", "weight_class": "LW", "method": methods[i],
                               "round": 3 if "Decision" in methods[i] else 1, "time": "5:00",
                               "scheduled_rounds": np.nan if i % 2 else 3})
        feat_csv = tmp_path / "features.csv"
        fights_csv = tmp_path / "fights.csv"
        pd.DataFrame(feat_rows).to_csv(feat_csv, index=False)
        pd.DataFrame(fight_rows).to_csv(fights_csv, index=False)
        monkeypatch.setattr(config, "FEATURES_CSV", feat_csv)
        monkeypatch.setattr(config, "RAW_FIGHTS_CSV", fights_csv)
        return build_method_dataset()

    def test_lutas_sem_metodo_categorizavel_ficam_fora(self, dataset):
        # 12 lutas, 3 sao "Overturned" -> 9 restam
        assert dataset["fight_id"].nunique() == 9
        assert set(dataset["method_class"].unique()) == set(METHOD_CLASSES)

    def test_treino_espelhado_e_avaliacao_deduplicada(self, dataset):
        train_df, cal_df, test_df = temporal_group_split(dataset)
        # treino mantem as 2 linhas espelhadas por luta (mesmo label simetrico)
        assert (train_df.groupby("fight_id").size() == 2).all()
        assert (train_df.groupby("fight_id")["method_class"].nunique() == 1).all()
        # avaliacao dedupe = 1 linha por luta (como feito em train_method)
        test_dedup = test_df.drop_duplicates("fight_id")
        assert (test_dedup.groupby("fight_id").size() == 1).all()
        assert test_dedup["fight_id"].nunique() == test_df["fight_id"].nunique()

    def test_dataset_nao_carrega_mais_labels_de_duracao(self, dataset):
        # finish_round/scheduled_rounds so existiam para o modelo de faixa
        for col in ("finish_round", "scheduled_rounds"):
            assert col not in dataset.columns, col


class TestSymmetricSums:
    def test_somas_identicas_nas_linhas_espelhadas(self):
        """As features *_sum sao simetricas: a+b == b+a, entao as duas linhas
        espelhadas da mesma luta tem exatamente o mesmo valor."""
        df = pd.read_csv(config.FEATURES_CSV)
        present = [c for c in SYMMETRIC_SUM_COLUMNS if c in df.columns]
        assert present == SYMMETRIC_SUM_COLUMNS, "somas ausentes do CSV de features"
        g = df.groupby("fight_id")
        for col in present:
            nun = g[col].nunique(dropna=False)
            assert (nun <= 1).all(), f"{col} difere entre linhas espelhadas"
