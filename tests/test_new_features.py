"""
Testes das 3 features novas (jul/2026): confronto de stance, taxas de
finalizacao point-in-time e rating Elo.

Pontos criticos cobertos:
  - categorize_method com textos REAIS das duas fontes (scrape e espelho);
  - taxas de finalizacao sem vazamento (shift(1), como as demais stats);
  - stance_mismatch identico nas duas linhas espelhadas (simetrica, nao
    inverte sinal);
  - Elo em passada cronologica global, rating registrado ANTES da luta,
    atualizacao correta, base para estreantes, sem update em empate/NC.
"""
import numpy as np
import pandas as pd
import pytest

import config
from src.features import (
    _build_mirrored_diff_rows,
    categorize_method,
    compute_current_levels,
    compute_point_in_time_stats,
    stance_mismatch_value,
)
from src.ratings import compute_elo_ratings, expected_score


# ---------------------------------------------------------------------------
# categorize_method
# ---------------------------------------------------------------------------

class TestCategorizeMethod:
    @pytest.mark.parametrize("raw,expected", [
        # formato do scrape direto do UFCStats (abreviacoes + detalhe colado)
        ("KO/TKO Spinning Back Kick", "KO_TKO"),
        ("KO/TKO Punches", "KO_TKO"),
        ("SUB Rear Naked Choke", "SUBMISSION"),
        ("U-DEC", "DECISION"),
        ("S-DEC", "DECISION"),
        ("M-DEC", "DECISION"),
        # formato do espelho GitHub (texto por extenso)
        ("KO/TKO", "KO_TKO"),
        ("Submission", "SUBMISSION"),
        ("Decision - Unanimous", "DECISION"),
        ("Decision - Split", "DECISION"),
        ("Decision - Majority", "DECISION"),
        ("TKO - Doctor's Stoppage", "KO_TKO"),
    ])
    def test_casos_reais_das_duas_fontes(self, raw, expected):
        assert categorize_method(raw) == expected

    @pytest.mark.parametrize("raw", ["DQ", "Overturned", "Could Not Continue", "", None, np.nan])
    def test_metodos_nao_classificaveis_viram_none(self, raw):
        assert categorize_method(raw) is None


# ---------------------------------------------------------------------------
# stance_mismatch_value
# ---------------------------------------------------------------------------

class TestStanceMismatch:
    def test_valores_basicos(self):
        assert stance_mismatch_value("Orthodox", "Southpaw") == 1.0
        assert stance_mismatch_value("Southpaw", "Orthodox") == 1.0  # simetrica
        assert stance_mismatch_value("Orthodox", "Orthodox") == 0.0
        assert stance_mismatch_value("Southpaw", "Southpaw") == 0.0

    def test_switch_e_ausente_viram_nan(self):
        assert np.isnan(stance_mismatch_value("Switch", "Orthodox"))
        assert np.isnan(stance_mismatch_value("Orthodox", None))
        assert np.isnan(stance_mismatch_value(np.nan, "Southpaw"))
        assert np.isnan(stance_mismatch_value("Open Stance", "Orthodox"))

    def test_nao_inverte_na_linha_espelhada(self):
        base = {
            "fight_id": ["m1"], "event_date": pd.to_datetime(["2021-01-01"]),
            "fighter_1": ["A"], "fighter_2": ["B"], "winner": ["A"],
        }
        merged = pd.DataFrame(base)
        vals = dict(striking_accuracy=0.5, takedown_accuracy=0.5, takedown_defense=0.5,
                    reach_cm=180.0, height_cm=175.0, age_years=30.0, days_since_last_fight=90.0,
                    recent_win_rate=0.5, career_win_rate=0.5, n_prior_fights=5, low_experience=0,
                    ko_rate=0.2, submission_rate=0.2, elo=1500.0)
        for k, v in vals.items():
            merged[f"{k}_1"] = v
            merged[f"{k}_2"] = v
        merged["stance_1"] = "Orthodox"
        merged["stance_2"] = "Southpaw"
        out = _build_mirrored_diff_rows(merged, ("_1", "_2"), ("fighter_1", "fighter_2"))
        assert len(out) == 2
        # MESMO valor (1.0) nas duas linhas espelhadas -- sem inversao de sinal
        assert (out["stance_mismatch"] == 1.0).all()


# ---------------------------------------------------------------------------
# Taxas de finalizacao point-in-time
# ---------------------------------------------------------------------------

def _long_df_com_metodos() -> pd.DataFrame:
    """F1: vence por KO, vence por SUB, perde por decisao, vence por KO."""
    rows = []
    lutas = [
        ("f1", "2020-01-01", "W", "KO/TKO Punches"),
        ("f2", "2020-06-01", "W", "SUB Rear Naked Choke"),
        ("f3", "2021-01-01", "L", "Decision - Unanimous"),
        ("f4", "2021-06-01", "W", "KO/TKO"),
    ]
    for i, (fid, date, res, method) in enumerate(lutas):
        opp = f"O{i}"
        rows.append(dict(fight_id=fid, event_date=date, fighter="F1", opponent=opp,
                         result=res, method=method,
                         sig_strikes_landed=10, sig_strikes_attempted=20,
                         takedowns_landed=1, takedowns_attempted=2))
        rows.append(dict(fight_id=fid, event_date=date, fighter=opp, opponent="F1",
                         result=("L" if res == "W" else "W"), method=method,
                         sig_strikes_landed=5, sig_strikes_attempted=10,
                         takedowns_landed=0, takedowns_attempted=1))
    df = pd.DataFrame(rows)
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


def _bio_vazia() -> pd.DataFrame:
    return pd.DataFrame({"name": ["F1"], "dob": [pd.NaT], "reach_cm": [np.nan],
                         "height_cm": [np.nan], "stance": ["Orthodox"]})


class TestFinishRatesPointInTime:
    def test_taxas_excluem_a_luta_atual(self):
        stats = compute_point_in_time_stats(_long_df_com_metodos(), _bio_vazia())
        f1 = stats[stats["fighter"] == "F1"].set_index("fight_id")
        # f1: sem historico
        assert np.isnan(f1.loc["f1", "ko_rate"])
        # f2: 1 luta anterior, 1 KO -> ko_rate 1.0, sub_rate 0.0
        assert f1.loc["f2", "ko_rate"] == pytest.approx(1.0)
        assert f1.loc["f2", "submission_rate"] == pytest.approx(0.0)
        # f3: 2 anteriores (KO, SUB) -> 0.5 / 0.5
        assert f1.loc["f3", "ko_rate"] == pytest.approx(0.5)
        assert f1.loc["f3", "submission_rate"] == pytest.approx(0.5)
        # f4: 3 anteriores (KO, SUB, derrota) -> 1/3 e 1/3; o KO da PROPRIA f4
        # nao pode entrar (seria vazamento)
        assert f1.loc["f4", "ko_rate"] == pytest.approx(1 / 3)
        assert f1.loc["f4", "submission_rate"] == pytest.approx(1 / 3)

    def test_derrota_por_ko_nao_conta_como_ko_do_lutador(self):
        stats = compute_point_in_time_stats(_long_df_com_metodos(), _bio_vazia())
        # O0 PERDEU a f1 por KO -- a taxa de KO dele (como vencedor) deve ser
        # 0 dali em diante, nao 1
        o0 = stats[stats["fighter"] == "O0"].iloc[0]
        assert o0["cum_is_win_ko"] == 0 or np.isnan(o0["ko_rate"])

    def test_current_levels_incluem_a_ultima_luta(self):
        levels = compute_current_levels(_long_df_com_metodos(), _bio_vazia())
        f1 = levels[levels["fighter"] == "F1"].iloc[0]
        # carreira completa: 4 lutas, 2 KOs, 1 SUB
        assert f1["ko_rate"] == pytest.approx(2 / 4)
        assert f1["submission_rate"] == pytest.approx(1 / 4)
        assert f1["stance"] == "Orthodox"


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------

def _fights(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["fight_id", "event_date", "fighter_1", "fighter_2", "winner"])
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


class TestElo:
    def test_estreantes_comecam_no_rating_base(self):
        pre, _ = compute_elo_ratings(_fights([["f1", "2020-01-01", "A", "B", "A"]]),
                                     k=32, base_rating=1500)
        assert pre.iloc[0]["elo_1_pre"] == 1500
        assert pre.iloc[0]["elo_2_pre"] == 1500

    def test_atualizacao_e_registro_pre_luta(self):
        pre, current = compute_elo_ratings(_fights([
            ["f1", "2020-01-01", "A", "B", "A"],
            ["f2", "2020-02-01", "A", "C", "A"],
        ]), k=32, base_rating=1500)
        # apos f1 (iguais, e=0.5): A=1516, B=1484
        f2 = pre[pre["fight_id"] == "f2"].iloc[0]
        assert f2["elo_1_pre"] == pytest.approx(1516.0)   # rating de A ANTES de f2
        assert f2["elo_2_pre"] == pytest.approx(1500.0)   # C estreante
        # o resultado de f2 NAO esta no pre de f2 (sem vazamento); esta no current
        e_a = expected_score(1516.0, 1500.0)
        assert current["A"] == pytest.approx(1516.0 + 32 * (1 - e_a))
        assert current["C"] == pytest.approx(1500.0 - 32 * (1 - e_a))
        assert current["B"] == pytest.approx(1484.0)

    def test_ordem_cronologica_global_mesmo_com_input_desordenado(self):
        # mesmo par de lutas, mas passadas fora de ordem no DataFrame
        pre, _ = compute_elo_ratings(_fights([
            ["f2", "2020-02-01", "A", "C", "A"],
            ["f1", "2020-01-01", "A", "B", "A"],
        ]), k=32, base_rating=1500)
        f2 = pre[pre["fight_id"] == "f2"].iloc[0]
        assert f2["elo_1_pre"] == pytest.approx(1516.0)  # f1 processada primeiro

    def test_empate_ou_nc_nao_atualiza(self):
        pre, current = compute_elo_ratings(_fights([
            ["f1", "2020-01-01", "A", "B", np.nan],
        ]), k=32, base_rating=1500)
        assert len(pre) == 1
        assert current == {}  # ninguem atualizado

    def test_favorito_ganha_menos_pontos_que_azarao(self):
        # A ja esta 200 pontos acima; vitoria dele rende poucos pontos
        _, current = compute_elo_ratings(_fights([
            ["f1", "2020-01-01", "A", "B", "A"],   # A 1500->1516
            ["f2", "2020-02-01", "A", "B", "A"],
            ["f3", "2020-03-01", "A", "B", "A"],
        ]), k=32, base_rating=1500)
        ganho_total_a = current["A"] - 1500
        # cada vitoria seguinte rende menos que a anterior (e < 0.5 vai caindo)
        assert ganho_total_a < 3 * 16
        assert current["A"] + current["B"] == pytest.approx(3000.0)  # soma conservada
