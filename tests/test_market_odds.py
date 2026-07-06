"""
Testes de src/market_odds.py: casamento de nomes entre fontes, mapeamento
do lado favorito, remocao de vig e metricas modelo-vs-mercado -- tudo com
CSVs/DataFrames sinteticos pequenos.
"""
import numpy as np
import pandas as pd
import pytest

import config
from src import market_odds as mo


# ---------------------------------------------------------------------------
# _normalize_name
# ---------------------------------------------------------------------------

class TestNormalizeName:
    def test_acentos_caixa_e_espacos(self):
        assert mo._normalize_name("  José   ALDO ") == "jose aldo"
        assert mo._normalize_name("Weili Zhang") == "weili zhang"

    def test_nomes_diferentes_continuam_diferentes(self):
        assert mo._normalize_name("Jon Jones") != mo._normalize_name("Jones Jon")


# ---------------------------------------------------------------------------
# load_odds_fights (filtros de qualidade)
# ---------------------------------------------------------------------------

def _write_odds_csv(path, rows):
    cols = ["event_date", "fighter1", "fighter2", "favourite", "underdog",
            "favourite_odds", "underdog_odds"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


class TestLoadOddsFights:
    def test_odds_inf_e_invalidas_sao_descartadas(self, tmp_path, monkeypatch):
        """Regressao: o CSV real tem lutas com odds = inf, que passavam por um
        filtro '> 1.0' e viravam probabilidade 0 -- uma unica dessas explode o
        log loss do mercado (~13.8) e invalida a comparacao inteira."""
        csv = tmp_path / "market_odds.csv"
        _write_odds_csv(csv, [
            ["2022-01-01", "Ana Silva", "Bia Costa", "Ana Silva", "Bia Costa", 1.50, 2.60],
            ["2022-01-01", "Cris Rocha", "Dani Lima", "Cris Rocha", "Dani Lima", np.inf, np.inf],
            ["2022-01-01", "Eva Nunes", "Fabi Melo", "Eva Nunes", "Fabi Melo", 1.0, 5.0],
            ["2022-01-01", "Gabi Reis", "Hana Cruz", "Gabi Reis", "Hana Cruz", 250.0, 1.01],
        ])
        monkeypatch.setattr(config, "MARKET_ODDS_CSV", csv)
        odds = mo.load_odds_fights()
        assert len(odds) == 1
        assert odds.iloc[0]["fighter1"] == "Ana Silva"


# ---------------------------------------------------------------------------
# match_odds_to_predictions
# ---------------------------------------------------------------------------

def _make_preds(rows):
    cols = ["fight_id", "event_date", "fighter_a", "fighter_b", "label",
            "pred_logreg", "pred_gbm"]
    df = pd.DataFrame(rows, columns=cols)
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


@pytest.fixture
def odds_df(tmp_path, monkeypatch):
    csv = tmp_path / "market_odds.csv"
    _write_odds_csv(csv, [
        # favorito = fighter1; odds 1.4/3.0 -> devig: 0.7143/(0.7143+0.3333)=0.6818
        ["2022-03-05", "Ana Silva", "Bia Costa", "Ana Silva", "Bia Costa", 1.40, 3.00],
        # favorito = fighter2 (Dani); grafia com acento p/ testar normalizacao
        ["2022-03-05", "Cris Rocha", "Dani Lima", "Dani Lima", "Cris Rocha", 2.50, 1.55],
        # typo leve p/ testar fuzzy ("Aleksander" vs "Alexander")
        ["2022-04-09", "Aleksander Volkov", "Greta Souza", "Aleksander Volkov", "Greta Souza", 1.80, 2.05],
        # mesma dupla da 1a luta, mas 60 dias depois (nao pode casar com a de marco)
        ["2022-05-04", "Ana Silva", "Bia Costa", "Bia Costa", "Ana Silva", 1.90, 1.95],
    ])
    monkeypatch.setattr(config, "MARKET_ODDS_CSV", csv)
    return mo.load_odds_fights()


class TestMatching:
    def test_match_exato_fighter_a_e_favorito(self, odds_df):
        preds = _make_preds([
            ["f1", "2022-03-05", "Ana Silva", "Bia Costa", 1, 0.60, 0.65],
        ])
        m = mo.match_odds_to_predictions(preds, odds_df)
        assert len(m) == 1
        row = m.iloc[0]
        assert row["favourite"] == "Ana Silva"
        assert row["model_prob_fav"] == pytest.approx(0.65)          # lado A = favorito
        assert row["model_prob_fav_logreg"] == pytest.approx(0.60)
        assert row["fav_won"] == 1
        assert row["market_prob_fav"] == pytest.approx(0.6818, abs=1e-3)  # devig de 1.40/3.00

    def test_match_com_ordem_invertida_e_favorito_do_lado_b(self, odds_df):
        # nas predicoes, fighter_a = Cris (azarao); na fonte, favorito = Dani
        preds = _make_preds([
            ["f2", "2022-03-05", "Cris Rocha", "Dani Lima", 1, 0.55, 0.58],
        ])
        m = mo.match_odds_to_predictions(preds, odds_df)
        row = m.iloc[0]
        assert row["favourite"] == "Dani Lima"
        assert row["model_prob_fav"] == pytest.approx(1 - 0.58)  # complemento (favorito e o lado B)
        assert row["fav_won"] == 0                                # label=1 -> venceu o lado A (azarao)

    def test_match_fuzzy_typo_leve(self, odds_df):
        preds = _make_preds([
            ["f3", "2022-04-09", "Alexander Volkov", "Greta Souza", 0, 0.50, 0.52],
        ])
        m = mo.match_odds_to_predictions(preds, odds_df)
        assert len(m) == 1
        assert m.iloc[0]["favourite"] == "Aleksander Volkov"

    def test_data_fora_da_janela_nao_casa(self, odds_df):
        # mesma dupla existe na fonte em 03-05 e 05-04, mas a predicao e de 04-04:
        # nenhuma das duas esta dentro de +-3 dias -> nao casa
        preds = _make_preds([
            ["f4", "2022-04-04", "Ana Silva", "Bia Costa", 1, 0.6, 0.6],
        ])
        m = mo.match_odds_to_predictions(preds, odds_df)
        assert m.empty

    def test_linhas_espelhadas_contam_como_uma_luta(self, odds_df):
        preds = _make_preds([
            ["f1", "2022-03-05", "Ana Silva", "Bia Costa", 1, 0.60, 0.65],
            ["f1", "2022-03-05", "Bia Costa", "Ana Silva", 0, 0.40, 0.35],
        ])
        m = mo.match_odds_to_predictions(preds, odds_df)
        assert len(m) == 1

    def test_nome_sem_correspondencia_nao_casa(self, odds_df):
        preds = _make_preds([
            ["f5", "2022-03-05", "Pessoa Inexistente", "Outra Pessoa", 1, 0.6, 0.6],
        ])
        m = mo.match_odds_to_predictions(preds, odds_df)
        assert m.empty


# ---------------------------------------------------------------------------
# report_comparison (metricas modelo vs. mercado)
# ---------------------------------------------------------------------------

class TestReportComparison:
    def test_metricas_calculadas_a_mao(self):
        matched = pd.DataFrame({
            "fight_id": ["a", "b"],
            "event_date": pd.to_datetime(["2022-01-01", "2022-02-01"]),
            "fighter_a": ["X", "Y"], "fighter_b": ["W", "Z"],
            "favourite": ["X", "Y"],
            "favourite_odds": [1.5, 1.8], "underdog_odds": [2.6, 2.1],
            "overround_pct": [5.0, 3.2],
            "model_prob_fav": [0.60, 0.40],
            "model_prob_fav_logreg": [0.55, 0.45],
            "market_prob_fav": [0.80, 0.70],
            "fav_won": [1, 0],
        })
        r = mo.report_comparison(matched)
        assert r["n_fights"] == 2
        assert r["favourite_win_rate"] == pytest.approx(0.5)
        # mercado: -(ln 0.8 + ln 0.3)/2
        esperado_mercado = -(np.log(0.8) + np.log(0.3)) / 2
        assert r["market"]["log_loss"] == pytest.approx(esperado_mercado, abs=1e-4)
        # modelo gbm: -(ln 0.6 + ln 0.6)/2  (acertou 0.6 na 1a; 1-0.4=0.6 na 2a)
        esperado_gbm = -(np.log(0.6) + np.log(0.6)) / 2
        assert r["gbm"]["log_loss"] == pytest.approx(esperado_gbm, abs=1e-4)
        # acuracia: mercado acerta so a 1a (0.8>=0.5 certo; 0.7>=0.5 errado) = 0.5
        assert r["market"]["accuracy"] == pytest.approx(0.5)
        # brier do mercado: ((0.8-1)^2 + (0.7-0)^2)/2 = (0.04+0.49)/2
        assert r["market"]["brier_score"] == pytest.approx((0.04 + 0.49) / 2, abs=1e-6)

    def test_amostra_vazia_nao_quebra(self):
        assert mo.report_comparison(pd.DataFrame()) == {}
