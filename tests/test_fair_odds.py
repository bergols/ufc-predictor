"""
Testes de probability_to_fair_odds (src/utils.py) e
compute_total_rounds_market (src/predict.py) -- as pecas das abas novas de
metodo/duracao do card_report.
"""
import pytest

from src.predict import compute_total_rounds_market
from src.utils import probability_to_fair_odds


class TestProbabilityToFairOdds:
    def test_meio_a_meio_cai_no_lado_negativo(self):
        """Convencao documentada: p == 0.5 exato -> decimal 2.00 / -100
        (p >= 0.5 tratado como favorito)."""
        decimal, american = probability_to_fair_odds(0.5)
        assert decimal == pytest.approx(2.0)
        assert american == -100

    @pytest.mark.parametrize("p,dec,amer", [
        (0.80, 1.25, -400),    # favorito forte
        (0.25, 4.00, +300),    # azarao
        (0.75, 1.333, -300),
        (0.10, 10.0, +900),
        (0.60, 1.667, -150),
    ])
    def test_casos_conhecidos(self, p, dec, amer):
        decimal, american = probability_to_fair_odds(p)
        assert decimal == pytest.approx(dec, abs=1e-3)
        assert american == amer

    def test_sem_vig_dois_lados_complementares(self):
        """Odds justas: as probabilidades implicitas dos dois lados somam
        exatamente 1 (nenhum overround)."""
        d1, _ = probability_to_fair_odds(0.62)
        d2, _ = probability_to_fair_odds(0.38)
        assert (1 / d1) + (1 / d2) == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5])
    def test_probabilidade_degenerada_levanta_erro(self, p):
        """p=0 (odd infinita) e p=1 (sem retorno) nao tem odd justa util --
        falha clara em vez de cap arbitrario."""
        with pytest.raises(ValueError, match="Probabilidade"):
            probability_to_fair_odds(p)

    def test_perto_das_bordas_nao_quebra(self):
        d_lo, a_lo = probability_to_fair_odds(0.001)
        assert d_lo == pytest.approx(1000.0)
        assert a_lo == pytest.approx(99900)
        d_hi, a_hi = probability_to_fair_odds(0.999)
        assert d_hi == pytest.approx(1.001)
        assert a_hi == pytest.approx(-99900)


class TestComputeTotalRoundsMarket:
    def test_duas_linhas_calculadas_a_mao(self):
        method = {"KO_TKO": 0.40, "SUBMISSION": 0.20, "DECISION": 0.40}
        bands = {"1": 0.50, "2": 0.30, "3+": 0.20}
        tm = compute_total_rounds_market(method, bands)
        # Under 1,5 = P(fin) * P(R1|fin) = 0.6 * 0.5 = 0.30
        assert tm["under_1_5"] == pytest.approx(0.30)
        assert tm["over_1_5"] == pytest.approx(0.70)
        # Under 2,5 = P(fin) * [P(R1|fin) + P(R2|fin)] = 0.6 * 0.8 = 0.48
        assert tm["under_2_5"] == pytest.approx(0.48)
        # Over 2,5 = P(decisao) + P(fin)*P(R3+|fin) = 0.4 + 0.6*0.2 = 0.52
        assert tm["over_2_5"] == pytest.approx(0.52)
        # coerencia: cada linha soma 1, e under_2_5 >= under_1_5 sempre
        assert tm["under_1_5"] + tm["over_1_5"] == pytest.approx(1.0)
        assert tm["under_2_5"] + tm["over_2_5"] == pytest.approx(1.0)
        assert tm["under_2_5"] >= tm["under_1_5"]

    def test_so_decisao_significa_over_garantido_nas_duas_linhas(self):
        method = {"KO_TKO": 0.0, "SUBMISSION": 0.0, "DECISION": 1.0}
        bands = {"1": 0.9, "2": 0.1, "3+": 0.0}
        tm = compute_total_rounds_market(method, bands)
        # sem finalizacao possivel, a decisao vai ao round 3 ou 5: passa das duas linhas
        assert tm["under_1_5"] == pytest.approx(0.0)
        assert tm["under_2_5"] == pytest.approx(0.0)
        assert tm["over_2_5"] == pytest.approx(1.0)

    def test_finalizadores_de_primeiro_round(self):
        method = {"KO_TKO": 0.70, "SUBMISSION": 0.10, "DECISION": 0.20}
        bands = {"1": 0.75, "2": 0.20, "3+": 0.05}
        tm = compute_total_rounds_market(method, bands)
        assert tm["under_1_5"] == pytest.approx(0.8 * 0.75)          # 0.60
        assert tm["under_2_5"] == pytest.approx(0.8 * 0.95)          # 0.76
