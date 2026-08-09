"""
Testes de probability_to_fair_odds (src/utils.py) -- as odds justas da aba
de metodo do card_report.

Os testes de compute_total_rounds_market sairam junto com a previsao de
duracao (ago/2026); ver o cabecalho de src/train_method.py.
"""
import pytest

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
