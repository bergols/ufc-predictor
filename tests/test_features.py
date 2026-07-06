"""
Testes das funcoes criticas de src/features.py.

Foco principal: garantir que NAO ha vazamento de dados temporal --
compute_point_in_time_stats deve usar somente lutas ANTERIORES a cada
linha -- e que a geracao das linhas espelhadas (A-B / B-A) e consistente
(features antissimetricas, labels complementares, flags trocadas).

Rodar com:  python -m pytest tests/ -v   (a partir da raiz do projeto)
"""
import numpy as np
import pandas as pd
import pytest

import config
from src.features import (
    FEATURE_COLUMNS,
    MIRROR_NON_NEGATED_COLUMNS,
    _build_mirrored_diff_rows,
    _find_col,
    build_features_from_public_dataset,
    compute_current_levels,
    compute_point_in_time_stats,
)


# ---------------------------------------------------------------------------
# Dados sinteticos: F1 luta 3 vezes (W, L, W) contra oponentes diferentes
# ---------------------------------------------------------------------------

def _make_long_df() -> pd.DataFrame:
    """Historico sintetico com valores escolhidos a mao para conferencia."""
    rows = [
        # fight 1 (2020-01-01): F1 vence O1
        dict(fight_id="f1", event_date="2020-01-01", fighter="F1", opponent="O1", result="W",
             sig_strikes_landed=10, sig_strikes_attempted=20, takedowns_landed=1, takedowns_attempted=2),
        dict(fight_id="f1", event_date="2020-01-01", fighter="O1", opponent="F1", result="L",
             sig_strikes_landed=5, sig_strikes_attempted=10, takedowns_landed=0, takedowns_attempted=4),
        # fight 2 (2020-06-01): F1 perde para O2
        dict(fight_id="f2", event_date="2020-06-01", fighter="F1", opponent="O2", result="L",
             sig_strikes_landed=30, sig_strikes_attempted=40, takedowns_landed=2, takedowns_attempted=2),
        dict(fight_id="f2", event_date="2020-06-01", fighter="O2", opponent="F1", result="W",
             sig_strikes_landed=15, sig_strikes_attempted=15, takedowns_landed=3, takedowns_attempted=6),
        # fight 3 (2021-01-01): F1 vence O3 -- stats propositalmente extremas
        # para detectar vazamento (nao podem aparecer nas features da fight 3)
        dict(fight_id="f3", event_date="2021-01-01", fighter="F1", opponent="O3", result="W",
             sig_strikes_landed=100, sig_strikes_attempted=100, takedowns_landed=10, takedowns_attempted=10),
        dict(fight_id="f3", event_date="2021-01-01", fighter="O3", opponent="F1", result="L",
             sig_strikes_landed=1, sig_strikes_attempted=2, takedowns_landed=0, takedowns_attempted=1),
    ]
    df = pd.DataFrame(rows)
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


def _make_bio() -> pd.DataFrame:
    return pd.DataFrame({
        "name": ["F1", "O1", "O2", "O3"],
        "dob": pd.to_datetime(["1990-01-01", "1992-05-05", "1988-03-03", "1995-07-07"]),
        "reach_cm": [180.0, 175.0, 185.0, 178.0],
        "height_cm": [175.0, 172.0, 180.0, 176.0],
    })


@pytest.fixture
def pit_stats() -> pd.DataFrame:
    return compute_point_in_time_stats(_make_long_df(), _make_bio())


def _f1_row(pit_stats: pd.DataFrame, fight_id: str) -> pd.Series:
    rows = pit_stats[(pit_stats["fighter"] == "F1") & (pit_stats["fight_id"] == fight_id)]
    assert len(rows) == 1
    return rows.iloc[0]


# ---------------------------------------------------------------------------
# compute_point_in_time_stats
# ---------------------------------------------------------------------------

class TestPointInTimeStats:
    def test_primeira_luta_nao_tem_historico(self, pit_stats):
        row = _f1_row(pit_stats, "f1")
        assert row["n_prior_fights"] == 0
        assert np.isnan(row["striking_accuracy"])
        assert np.isnan(row["takedown_accuracy"])
        assert np.isnan(row["career_win_rate"])
        assert np.isnan(row["recent_win_rate"])
        assert np.isnan(row["days_since_last_fight"])

    def test_segunda_luta_usa_somente_a_primeira(self, pit_stats):
        row = _f1_row(pit_stats, "f2")
        assert row["n_prior_fights"] == 1
        assert row["striking_accuracy"] == pytest.approx(10 / 20)
        assert row["takedown_accuracy"] == pytest.approx(1 / 2)
        # O1 tentou 4 quedas em F1 e acertou 0 -> defesa 100%
        assert row["takedown_defense"] == pytest.approx(1.0)
        assert row["career_win_rate"] == pytest.approx(1.0)
        assert row["recent_win_rate"] == pytest.approx(1.0)
        assert row["days_since_last_fight"] == 152  # 2020-01-01 -> 2020-06-01

    def test_terceira_luta_acumula_duas_anteriores(self, pit_stats):
        row = _f1_row(pit_stats, "f3")
        assert row["n_prior_fights"] == 2
        assert row["striking_accuracy"] == pytest.approx((10 + 30) / (20 + 40))
        assert row["takedown_accuracy"] == pytest.approx((1 + 2) / (2 + 2))
        # oponentes tentaram 4+6 quedas em F1, acertaram 0+3 -> defesa 70%
        assert row["takedown_defense"] == pytest.approx(1 - 3 / 10)
        assert row["career_win_rate"] == pytest.approx(1 / 2)   # W, L
        assert row["recent_win_rate"] == pytest.approx(0.5)

    def test_sem_vazamento_da_luta_atual(self, pit_stats):
        """As stats extremas da fight 3 (100/100) NAO podem aparecer nas
        features da propria fight 3 nem de nenhuma anterior."""
        for fid in ("f1", "f2", "f3"):
            row = _f1_row(pit_stats, fid)
            if not np.isnan(row["striking_accuracy"]):
                assert row["striking_accuracy"] < 0.99, (
                    f"striking_accuracy da {fid} parece incluir a luta atual/futura (vazamento!)"
                )

    def test_idade_na_data_da_luta(self, pit_stats):
        row = _f1_row(pit_stats, "f2")
        expected_age = (pd.Timestamp("2020-06-01") - pd.Timestamp("1990-01-01")).days / 365.25
        assert row["age_years"] == pytest.approx(expected_age)

    def test_flag_low_experience(self, pit_stats):
        # MIN_FIGHTS_FOR_RELIABLE_STATS = 3 por default: todas as linhas de F1
        # tem 0-2 lutas anteriores -> flag ligada em todas
        assert config.MIN_FIGHTS_FOR_RELIABLE_STATS == 3
        f1_rows = pit_stats[pit_stats["fighter"] == "F1"]
        assert (f1_rows["low_experience"] == 1).all()

    def test_nc_nao_conta_como_vitoria_nem_derrota(self):
        long_df = _make_long_df()
        # transforma a fight 2 em no-contest para os dois lados
        long_df.loc[long_df["fight_id"] == "f2", "result"] = "NC"
        stats = compute_point_in_time_stats(long_df, _make_bio())
        row = _f1_row(stats, "f3")
        # historico de resultados vira [W, NC]: 1 vitoria, 0 derrotas em 2 lutas
        assert row["cum_is_win"] == 1
        assert row["cum_is_loss"] == 0

    def test_alinhamento_do_indice_apos_groupby_apply(self, pit_stats):
        """Regressao para o bug de desalinhamento de indice: os valores de
        recent_win_rate de um lutador nao podem 'vazar' para as linhas de
        outro lutador."""
        # O2 so lutou uma vez -> recent_win_rate NaN obrigatorio
        o2 = pit_stats[pit_stats["fighter"] == "O2"].iloc[0]
        assert np.isnan(o2["recent_win_rate"])


# ---------------------------------------------------------------------------
# compute_current_levels (nivel ATUAL, inclui a ultima luta)
# ---------------------------------------------------------------------------

class TestCurrentLevels:
    def test_inclui_a_ultima_luta(self):
        levels = compute_current_levels(_make_long_df(), _make_bio())
        f1 = levels[levels["fighter"] == "F1"].iloc[0]
        assert f1["n_prior_fights"] == 3
        assert f1["striking_accuracy"] == pytest.approx((10 + 30 + 100) / (20 + 40 + 100))
        assert f1["career_win_rate"] == pytest.approx(2 / 3)

    def test_uma_linha_por_lutador(self):
        levels = compute_current_levels(_make_long_df(), _make_bio())
        assert levels["fighter"].is_unique
        assert set(levels["fighter"]) == {"F1", "O1", "O2", "O3"}


# ---------------------------------------------------------------------------
# _build_mirrored_diff_rows (linhas espelhadas A-B / B-A)
# ---------------------------------------------------------------------------

def _make_merged() -> pd.DataFrame:
    """Tabela 'larga' minima como a produzida em build_features_from_scrape."""
    base = {
        "fight_id": ["m1", "m2"],
        "event_date": pd.to_datetime(["2021-01-01", "2021-02-01"]),
        "fighter_1": ["A", "C"],
        "fighter_2": ["B", "D"],
        # m1 tem vencedor; m2 e empate/no-contest (winner NaN)
        "winner": ["A", np.nan],
    }
    values_1 = dict(striking_accuracy=0.6, takedown_accuracy=0.5, takedown_defense=0.8,
                    reach_cm=185.0, height_cm=180.0, age_years=30.0, days_since_last_fight=100.0,
                    recent_win_rate=0.8, career_win_rate=0.7, n_prior_fights=10, low_experience=0,
                    ko_rate=0.4, submission_rate=0.1, elo=1580.0, stance="Orthodox")
    values_2 = dict(striking_accuracy=0.4, takedown_accuracy=0.3, takedown_defense=0.6,
                    reach_cm=180.0, height_cm=178.0, age_years=33.0, days_since_last_fight=200.0,
                    recent_win_rate=0.4, career_win_rate=0.5, n_prior_fights=2, low_experience=1,
                    ko_rate=0.2, submission_rate=0.3, elo=1490.0, stance="Southpaw")
    merged = pd.DataFrame(base)
    for k, v in values_1.items():
        merged[f"{k}_1"] = v
    for k, v in values_2.items():
        merged[f"{k}_2"] = v
    return merged


class TestMirroredRows:
    def test_luta_sem_vencedor_e_descartada(self):
        out = _build_mirrored_diff_rows(_make_merged(), ("_1", "_2"), ("fighter_1", "fighter_2"))
        assert set(out["fight_id"]) == {"m1"}          # m2 (empate) descartada
        assert len(out) == 2                            # duas linhas espelhadas

    def test_labels_complementares(self):
        out = _build_mirrored_diff_rows(_make_merged(), ("_1", "_2"), ("fighter_1", "fighter_2"))
        row_ab = out[out["fighter_a"] == "A"].iloc[0]
        row_ba = out[out["fighter_a"] == "B"].iloc[0]
        assert row_ab["label"] == 1
        assert row_ba["label"] == 0

    def test_features_antissimetricas(self):
        out = _build_mirrored_diff_rows(_make_merged(), ("_1", "_2"), ("fighter_1", "fighter_2"))
        row_ab = out[out["fighter_a"] == "A"].iloc[0]
        row_ba = out[out["fighter_a"] == "B"].iloc[0]
        for col in FEATURE_COLUMNS:
            if col in MIRROR_NON_NEGATED_COLUMNS:
                continue
            assert row_ab[col] == pytest.approx(-row_ba[col]), col
        # valores conferidos a mao
        assert row_ab["striking_accuracy_diff"] == pytest.approx(0.2)   # 0.6 - 0.4
        assert row_ab["experience_diff"] == 8
        assert row_ab["elo_diff"] == pytest.approx(90.0)                # 1580 - 1490
        assert row_ab["ko_rate_diff"] == pytest.approx(0.2)             # 0.4 - 0.2
        assert row_ab["submission_rate_diff"] == pytest.approx(-0.2)    # 0.1 - 0.3

    def test_flags_trocadas_no_espelho(self):
        out = _build_mirrored_diff_rows(_make_merged(), ("_1", "_2"), ("fighter_1", "fighter_2"))
        row_ab = out[out["fighter_a"] == "A"].iloc[0]
        row_ba = out[out["fighter_a"] == "B"].iloc[0]
        assert row_ab["fighter_a_low_experience"] == 0
        assert row_ab["fighter_b_low_experience"] == 1
        assert row_ba["fighter_a_low_experience"] == 1
        assert row_ba["fighter_b_low_experience"] == 0


# ---------------------------------------------------------------------------
# Adaptador do dataset publico (schema R_/B_ do rajeevw/ufcdata)
# ---------------------------------------------------------------------------

def _write_public_csv(path) -> None:
    df = pd.DataFrame({
        "R_fighter": ["Alice", "Carol", "Eve"],
        "B_fighter": ["Bianca", "Dana", "Fay"],
        "date": ["2019-01-01", "2019-02-01", "2019-03-01"],
        "Winner": ["Red", "Blue", "Draw"],
        "R_avg_SIG_STR_pct": [0.6, 0.5, 0.5],
        "B_avg_SIG_STR_pct": [0.4, 0.5, 0.5],
        "R_avg_TD_pct": [0.5, 0.4, 0.4],
        "B_avg_TD_pct": [0.3, 0.4, 0.4],
        "R_avg_opp_TD_att": [4.0, 2.0, 2.0],
        "R_avg_opp_TD_landed": [1.0, 1.0, 1.0],
        "B_avg_opp_TD_att": [5.0, 2.0, 2.0],
        "B_avg_opp_TD_landed": [4.0, 1.0, 1.0],
        "R_Reach_cms": [185.0, 170.0, 170.0],
        "B_Reach_cms": [180.0, 170.0, 170.0],
        "R_Height_cms": [180.0, 165.0, 165.0],
        "B_Height_cms": [178.0, 165.0, 165.0],
        "R_age": [30, 25, 25],
        "B_age": [33, 25, 25],
        "R_current_win_streak": [3, 0, 0],
        "B_current_win_streak": [1, 0, 0],
        "R_wins": [8, 2, 2],
        "R_losses": [2, 2, 2],
        "B_wins": [5, 1, 1],
        "B_losses": [5, 3, 3],
    })
    df.to_csv(path, index=False)


class TestPublicDatasetAdapter:
    @pytest.fixture
    def public_features(self, tmp_path, monkeypatch) -> pd.DataFrame:
        csv_path = tmp_path / "public_dataset.csv"
        _write_public_csv(csv_path)
        monkeypatch.setattr(config, "PUBLIC_DATASET_CSV", csv_path)
        return build_features_from_public_dataset()

    def test_empate_descartado_e_espelhamento(self, public_features):
        # 3 lutas no CSV, 1 e empate -> 2 lutas x 2 linhas espelhadas = 4
        assert len(public_features) == 4
        assert public_features["fight_id"].nunique() == 2
        assert "Eve" not in set(public_features["fighter_a"])

    def test_labels(self, public_features):
        alice = public_features[public_features["fighter_a"] == "Alice"].iloc[0]
        bianca = public_features[public_features["fighter_a"] == "Bianca"].iloc[0]
        dana = public_features[public_features["fighter_a"] == "Dana"].iloc[0]
        assert alice["label"] == 1     # Red (Alice) venceu
        assert bianca["label"] == 0
        assert dana["label"] == 1      # Blue (Dana) venceu

    def test_diffs_e_takedown_defense_derivada(self, public_features):
        alice = public_features[public_features["fighter_a"] == "Alice"].iloc[0]
        assert alice["striking_accuracy_diff"] == pytest.approx(0.2)
        assert alice["reach_diff_cm"] == pytest.approx(5.0)
        assert alice["age_diff_years"] == pytest.approx(-3.0)
        assert alice["career_win_rate_diff"] == pytest.approx(0.8 - 0.5)
        assert alice["experience_diff"] == pytest.approx(0.0)  # 10 vs 10 lutas
        # td_def derivada de avg_opp_TD: R = 1 - 1/4 = 0.75; B = 1 - 4/5 = 0.2
        assert alice["takedown_defense_diff"] == pytest.approx(0.75 - 0.2)

    def test_antissimetria_no_espelho(self, public_features):
        g = public_features.groupby("fight_id")
        assert (g.size() == 2).all()
        assert (g["label"].sum() == 1).all()
        for col in FEATURE_COLUMNS:
            if col in MIRROR_NON_NEGATED_COLUMNS:
                continue
            sums = g[col].sum().dropna()
            assert (sums.abs() < 1e-9).all(), col


# ---------------------------------------------------------------------------
# _find_col (correspondencia flexivel de nomes de coluna)
# ---------------------------------------------------------------------------

class TestFindCol:
    def test_match_case_insensitive_com_prefixo(self):
        cols = ["R_avg_SIG_STR_pct", "B_avg_SIG_STR_pct", "R_Reach_cms"]
        assert _find_col(cols, "R_", ["avg_sig_str_pct"]) == "R_avg_SIG_STR_pct"
        assert _find_col(cols, "B_", ["avg_sig_str_pct"]) == "B_avg_SIG_STR_pct"
        assert _find_col(cols, "R_", ["reach_cms", "reach"]) == "R_Reach_cms"

    def test_sem_match_retorna_none(self):
        assert _find_col(["R_wins", "B_wins"], "R_", ["td_def"]) is None
