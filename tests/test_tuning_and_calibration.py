"""
Testes do polimento final (jul/2026): split cal_fit/cal_select, selecao de
metodo de calibracao por modelo, recalculo de Elo com K variavel,
multiplicador de K por metodo e a regra de "reverter para o mais simples
em empate".
"""
import numpy as np
import pandas as pd
import pytest

from src.ratings import compute_elo_ratings, method_k_multiplier
from src.train import build_logreg_pipeline, select_calibration_method, split_calibration_slice
from src.tuning import choose_winner, replace_elo_diff


# ---------------------------------------------------------------------------
# split_calibration_slice
# ---------------------------------------------------------------------------

def _cal_df(n_fights=10) -> pd.DataFrame:
    rows = []
    for i in range(n_fights):
        date = pd.Timestamp("2021-01-01") + pd.Timedelta(days=7 * i)
        for _ in range(2):  # duas linhas espelhadas por luta
            rows.append({"fight_id": f"f{i}", "event_date": date, "label": i % 2})
    return pd.DataFrame(rows)


class TestSplitCalibrationSlice:
    def test_linhas_espelhadas_ficam_juntas(self):
        fit, select = split_calibration_slice(_cal_df())
        assert set(fit["fight_id"]) & set(select["fight_id"]) == set()
        assert (fit.groupby("fight_id").size() == 2).all()
        assert (select.groupby("fight_id").size() == 2).all()

    def test_split_e_temporal_sem_embaralhar(self):
        fit, select = split_calibration_slice(_cal_df())
        assert fit["event_date"].max() <= select["event_date"].min()

    def test_metades_aproximadas(self):
        fit, select = split_calibration_slice(_cal_df(n_fights=11))
        assert fit["fight_id"].nunique() == 5
        assert select["fight_id"].nunique() == 6


# ---------------------------------------------------------------------------
# select_calibration_method (por modelo)
# ---------------------------------------------------------------------------

class TestSelectCalibrationMethod:
    def _fitted_model_and_data(self, seed=0):
        rng = np.random.default_rng(seed)
        n = 600
        X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
        y = pd.Series(((X["x1"] + rng.normal(scale=1.5, size=n)) > 0).astype(int))
        model = build_logreg_pipeline()
        model.fit(X.iloc[:300], y.iloc[:300])
        return model, X, y

    def test_fatia_pequena_cai_para_sigmoid(self):
        model, X, y = self._fitted_model_and_data()
        método = select_calibration_method(model, X.iloc[300:400], y.iloc[300:400],
                                            X.iloc[400:430], y.iloc[400:430])
        assert método == "sigmoid"  # <50 linhas de select: escolha padrao estavel

    def test_escolha_bate_com_comparacao_manual(self):
        from sklearn.metrics import log_loss
        from src.train import _calibrate
        model, X, y = self._fitted_model_and_data()
        Xf, yf = X.iloc[300:450], y.iloc[300:450]
        Xs, ys = X.iloc[450:], y.iloc[450:]
        escolhido = select_calibration_method(model, Xf, yf, Xs, ys)
        manual = {}
        for m in ("sigmoid", "isotonic"):
            probs = np.clip(_calibrate(model, Xf, yf, m).predict_proba(Xs)[:, 1], 1e-6, 1 - 1e-6)
            manual[m] = log_loss(ys, probs, labels=[0, 1])
        assert escolhido == min(manual, key=manual.get)


# ---------------------------------------------------------------------------
# method_k_multiplier + Elo com margem
# ---------------------------------------------------------------------------

class TestMethodKMultiplier:
    SCHEME = {"FINISH": 1.5, "DECISION_CLOSE": 0.5}

    @pytest.mark.parametrize("method,expected", [
        ("KO/TKO Punches", 1.5),
        ("SUB Rear Naked Choke", 1.5),
        ("Submission", 1.5),
        ("S-DEC", 0.5),
        ("Decision - Split", 0.5),
        ("Decision - Majority", 0.5),
        ("U-DEC", 1.0),                      # unanime: peso base (ausente no dict)
        ("Decision - Unanimous", 1.0),
        ("DQ", 1.0),                          # nao classificavel: peso base
    ])
    def test_buckets(self, method, expected):
        assert method_k_multiplier(method, self.SCHEME) == expected

    def test_none_desliga_a_margem(self):
        assert method_k_multiplier("KO/TKO", None) == 1.0
        assert method_k_multiplier("KO/TKO", {}) == 1.0

    def test_elo_com_margem_escala_o_update(self):
        fights = pd.DataFrame({
            "fight_id": ["f1"], "event_date": pd.to_datetime(["2020-01-01"]),
            "fighter_1": ["A"], "fighter_2": ["B"], "winner": ["A"],
            "method": ["KO/TKO"],
        })
        _, sem = compute_elo_ratings(fights, k=32, base_rating=1500, method_multipliers={})
        _, com = compute_elo_ratings(fights, k=32, base_rating=1500,
                                     method_multipliers={"FINISH": 1.5})
        assert sem["A"] == pytest.approx(1516.0)      # 32 * 0.5
        assert com["A"] == pytest.approx(1524.0)      # 32 * 1.5 * 0.5

    def test_sem_coluna_method_equivale_a_elo_simples(self):
        fights = pd.DataFrame({
            "fight_id": ["f1"], "event_date": pd.to_datetime(["2020-01-01"]),
            "fighter_1": ["A"], "fighter_2": ["B"], "winner": ["A"],
        })
        _, r = compute_elo_ratings(fights, k=32, base_rating=1500,
                                   method_multipliers={"FINISH": 2.0})
        assert r["A"] == pytest.approx(1516.0)


# ---------------------------------------------------------------------------
# replace_elo_diff (recalculo com K variavel) e choose_winner (reverter p/ simples)
# ---------------------------------------------------------------------------

class TestReplaceEloDiff:
    def test_recalculo_com_k_diferente(self):
        fights = pd.DataFrame({
            "fight_id": ["f1", "f2"],
            "event_date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "fighter_1": ["A", "A"], "fighter_2": ["B", "C"],
            "winner": ["A", "A"],
        })
        feature_df = pd.DataFrame({
            "fight_id": ["f2", "f2"],
            "event_date": pd.to_datetime(["2020-02-01"] * 2),
            "fighter_a": ["A", "C"], "fighter_b": ["C", "A"],
            "elo_diff": [999.0, -999.0],  # valor antigo qualquer, deve ser substituido
        })
        k16 = replace_elo_diff(feature_df, fights, k=16)
        assert k16["elo_diff"].tolist() == pytest.approx([8.0, -8.0])   # A=1508 vs C=1500
        k64 = replace_elo_diff(feature_df, fights, k=64)
        assert k64["elo_diff"].tolist() == pytest.approx([32.0, -32.0])
        # antissimetria preservada nas linhas espelhadas
        assert k64["elo_diff"].sum() == pytest.approx(0.0)


class TestChooseWinner:
    def test_vencedor_claro(self):
        scores = {"sem-margem": 0.6750, "leve": 0.6700}
        assert choose_winner(scores, ["sem-margem", "leve"]) == "leve"

    def test_empate_reverte_para_o_mais_simples(self):
        """Nao adicionar complexidade que nao se paga: em empate (ate 4
        casas), vence a opcao mais simples da lista de preferencia."""
        scores = {"sem-margem": 0.67501, "leve": 0.67498}
        assert choose_winner(scores, ["sem-margem", "leve"]) == "sem-margem"

    def test_empate_exato(self):
        scores = {"sem-margem": 0.68, "agressivo": 0.68}
        assert choose_winner(scores, ["sem-margem", "agressivo"]) == "sem-margem"
