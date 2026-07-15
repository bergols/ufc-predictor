"""
Testes da previsao com ESTREANTE (lutador sem historico na base).

Racional: o treino ja contem a primeira luta de cada lutador da historia
— nessas linhas as stats point-in-time sao NaN, n_prior_fights=0 e o Elo
e o rating base. Logo, prever um estreante com uma linha sintetica nesse
mesmo formato e in-distribution, nao invencao. O que protegemos aqui:

- allow_debutant=False (default) preserva o comportamento antigo
  (ValueError) — CLI e chamadas existentes nao mudam sozinhas;
- a linha sintetica tem EXATAMENTE o formato de estreia do treino;
- o card_report marca a luta com a nota de estreante (aviso visivel),
  em vez de esconder a previsao degradada como se fosse igual as outras.
"""
import numpy as np
import pandas as pd
import pytest

import config
from src import card_report
from src.predict import _resolve_fighters, debutant_level_row


@pytest.fixture
def levels():
    return pd.DataFrame([{
        "fighter": "Veterana Alfa", "striking_accuracy": 0.5, "takedown_accuracy": 0.4,
        "takedown_defense": 0.7, "reach_cm": 180.0, "height_cm": 175.0, "age_years": 30.0,
        "days_since_last_fight": 120.0, "recent_win_rate": 0.8, "career_win_rate": 0.75,
        "n_prior_fights": 12, "ko_rate": 0.3, "submission_rate": 0.1,
        "elo": 1620.0, "stance": "Orthodox", "low_experience": 0,
    }])


class TestDebutantRow:
    def test_formato_igual_ao_de_estreia_do_treino(self):
        row = debutant_level_row("Nova Lutadora")
        assert row["n_prior_fights"] == 0
        assert row["elo"] == config.ELO_BASE_RATING
        assert row["low_experience"] == 1
        for col in ("striking_accuracy", "career_win_rate", "recent_win_rate",
                    "reach_cm", "age_years", "days_since_last_fight"):
            assert pd.isna(row[col]), f"{col} deveria ser NaN (nao inventamos dado)"


class TestResolve:
    def test_default_continua_estrito(self, levels):
        with pytest.raises(ValueError, match="nao encontrado"):
            _resolve_fighters("Veterana Alfa", "Desconhecida Total", levels)

    def test_allow_debutant_gera_linha_sintetica_com_flag(self, levels):
        a, b, row_a, row_b, debut_a, debut_b = _resolve_fighters(
            "Veterana Alfa", "Desconhecida Total", levels, allow_debutant=True)
        assert (debut_a, debut_b) == (False, True)
        assert b == "Desconhecida Total"
        assert row_b["n_prior_fights"] == 0 and pd.isna(row_b["career_win_rate"])
        assert row_a["n_prior_fights"] == 12  # lado conhecido intacto

    def test_dois_estreantes_tambem_funciona(self, levels):
        _, _, row_a, row_b, debut_a, debut_b = _resolve_fighters(
            "Zumbi X", "Zumbi Y", levels, allow_debutant=True)
        assert debut_a and debut_b
        assert row_a["elo"] == row_b["elo"] == config.ELO_BASE_RATING


class TestCardReportComEstreante:
    def _predict_fn(self, a, b):
        return {"fighter_a": a, "fighter_b": b, "prob_a_wins": 0.70, "prob_b_wins": 0.30,
                "model_used": "logreg",
                "fighter_a_low_experience": False, "fighter_b_low_experience": True,
                "fighter_a_debutant": False, "fighter_b_debutant": True}

    def test_luta_com_estreante_entra_no_ranking_com_nota(self):
        odds = pd.DataFrame([{"fighter_a": "Veterana Alfa", "fighter_b": "Nova Lutadora",
                              "odds_a_decimal": 1.30, "odds_b_decimal": 3.80,
                              "scheduled_rounds": 3}])
        analysis = card_report.analyze_card(odds, predict_fn=self._predict_fn, method_fn=None)
        assert not analysis["no_prediction"]  # nao some mais
        fight = analysis["favorites"][0]
        assert fight["debutants"] == ["Nova Lutadora"]
        note = card_report._matched_note(fight)
        assert "estreando no UFC" in note
        assert "confiança reduzida" in note
