"""
Regularizacao de taxas com denominador pequeno (src/shrinkage.py).

Candidato, NAO promovido: em cal_select o contraste pareado por evento nao
exclui zero em nenhum grupo. O que se protege aqui e a mecanica -- se um dia
houver poder amostral para decidir, a conta precisa estar certa.
"""
import numpy as np
import pandas as pd
import pytest

from src.shrinkage import Prior, aplicar_shrinkage, estimar_prior, estimar_priors


class TestPrior:
    def test_denominador_grande_manda_a_observacao_vencer(self):
        p = Prior(media=0.5, k=5.0)
        assert p.aplicar(900, 1000) == pytest.approx((900 + 5*0.5)/1005, abs=1e-9)

    def test_denominador_zero_devolve_o_PRIOR(self):
        """Lutador sem nenhuma luta anterior recebe exatamente o prior --
        que e a leitura certa de 'nao sei nada sobre este aqui'."""
        assert Prior(0.43, 10.0).aplicar(0, 0) == pytest.approx(0.43)

    def test_denominador_minusculo_fica_perto_do_prior(self):
        """O caso que motivou tudo: 1-0 nao vale 100%."""
        encolhido = Prior(media=0.65, k=8.6).aplicar(1, 1)
        assert 0.65 < encolhido < 0.80, encolhido

    def test_e_monotonico_no_denominador(self):
        p = Prior(0.5, 10.0)
        vals = [p.aplicar(n, n) for n in (1, 5, 20, 100, 1000)]
        assert all(a < b for a, b in zip(vals, vals[1:]))
        assert vals[-1] == pytest.approx(1.0, abs=0.02)


class TestEstimativa:
    def test_mais_dispersao_entre_lutadores_da_k_MENOR(self):
        """k e o tamanho de amostra em que a evidencia individual passa a
        pesar mais que o prior. Populacao homogenea -> prior forte."""
        rng = np.random.default_rng(3)
        n = np.full(400, 50)
        disperso = estimar_prior(rng.binomial(50, rng.uniform(.2, .8, 400)), n)
        homogeneo = estimar_prior(rng.binomial(50, rng.uniform(.49, .51, 400)), n)
        assert disperso.k < homogeneo.k

    def test_amostra_pequena_demais_cai_no_neutro(self):
        p = estimar_prior([1, 2], [10, 10])
        assert p.media == 0.5 and p.k == 500.0

    def test_k_fica_dentro_dos_limites(self):
        rng = np.random.default_rng(5)
        p = estimar_prior(rng.binomial(50, 0.5, 300), np.full(300, 50))
        assert 1.0 <= p.k <= 500.0

    def test_ignora_denominador_minusculo_ao_ESTIMAR(self):
        """A observacao ruidosa e justamente o que se quer controlar; deixa-la
        entrar na estimativa inflaria a variancia e afrouxaria o prior."""
        rng = np.random.default_rng(9)
        bons = pd.DataFrame({"s": rng.binomial(50, 0.45, 300), "n": 50})
        lixo = pd.DataFrame({"s": [0, 1] * 150, "n": 1})
        so_bons = estimar_prior(bons["s"], bons["n"])
        com_lixo = estimar_prior(pd.concat([bons, lixo])["s"], pd.concat([bons, lixo])["n"])
        assert so_bons.k == pytest.approx(com_lixo.k, rel=0.2)


class TestAplicacao:
    def _stats(self):
        return pd.DataFrame({
            "cum_sig_strikes_landed": [1.0, 300.0, 0.0],
            "cum_sig_strikes_attempted": [2.0, 700.0, 0.0],
            "cum_takedowns_landed": [0.0, 10.0, 0.0],
            "cum_takedowns_attempted": [1.0, 30.0, 0.0],
            "cum_takedowns_landed_opp": [1.0, 5.0, 0.0],
            "cum_takedowns_attempted_opp": [2.0, 40.0, 0.0],
            "cum_is_win": [1.0, 12.0, 0.0],
            "cum_is_win_ko": [1.0, 5.0, 0.0],
            "cum_is_win_sub": [0.0, 2.0, 0.0],
            "n_prior_fights": [1, 18, 0],
            "striking_accuracy": [0.5, 0.4286, np.nan],
            "takedown_accuracy": [0.0, 0.3333, np.nan],
            "takedown_defense": [0.5, 0.875, np.nan],
            "career_win_rate": [1.0, 0.6667, np.nan],
            "ko_rate": [1.0, 0.2778, np.nan],
            "submission_rate": [0.0, 0.1111, np.nan],
        })

    def test_o_novato_se_move_MUITO_mais_que_o_veterano(self):
        d = self._stats()
        pri = {c: Prior(0.5, 8.0) for c in
               ("striking_accuracy", "career_win_rate", "ko_rate",
                "takedown_accuracy", "submission_rate", "takedown_defense")}
        out = aplicar_shrinkage(d, pri)
        novato = abs(out.loc[0, "career_win_rate"] - d.loc[0, "career_win_rate"])
        veterano = abs(out.loc[1, "career_win_rate"] - d.loc[1, "career_win_rate"])
        assert novato > 3 * veterano

    def test_defesa_de_queda_inverte_no_lado_certo(self):
        """takedown_defense = 1 - sofridas/tentadas. O prior tem de agir
        sobre a taxa SOFRIDA; encolher a defesa direto moveria para o lado
        errado."""
        d = self._stats()
        out = aplicar_shrinkage(d, {"takedown_defense": Prior(0.35, 4.6)})
        # 1 sofrida em 2 tentativas -> prior 0.35 puxa a taxa sofrida para
        # baixo, logo a DEFESA sobe em relacao ao cru (0.5)
        assert out.loc[0, "takedown_defense"] > d.loc[0, "takedown_defense"]
        assert 0.0 <= out.loc[0, "takedown_defense"] <= 1.0

    def test_nao_inventa_dado_para_quem_nao_tem_nenhum(self):
        """Estreante continua sendo estreante: denominador 0 vira o prior,
        nao uma estatistica individual."""
        d = self._stats()
        pri = {"career_win_rate": Prior(0.6511, 8.6)}
        out = aplicar_shrinkage(d, pri)
        assert out.loc[2, "career_win_rate"] == pytest.approx(0.6511)

    def test_nao_toca_coluna_sem_prior(self):
        d = self._stats()
        out = aplicar_shrinkage(d, {"ko_rate": Prior(0.25, 4.5)})
        pd.testing.assert_series_equal(out["striking_accuracy"], d["striking_accuracy"])

    def test_estimar_priors_cobre_todas_as_taxas(self):
        pri = estimar_priors(self._stats())
        assert set(pri) == {"striking_accuracy", "takedown_accuracy", "career_win_rate",
                            "ko_rate", "submission_rate", "takedown_defense"}
