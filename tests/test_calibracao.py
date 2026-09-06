"""
Curva de calibração / diagrama de confiabilidade (src.evaluate).

Era a última lacuna diagnóstica listada na skill `ufc-analise`: havia Brier e a
escolha sigmoid/isotonic, mas nada que mostrasse calibração POR FAIXA. Brier é
agregado — um modelo confiante demais em cima e tímido embaixo tem os dois
erros se cancelando na média, e o número final não denuncia.

Não toca no modelo: é medição.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluate import (calibration_table, expected_calibration_error,
                          log_calibration)


class TestTabela:
    def test_modelo_perfeitamente_calibrado_tem_gap_zero(self):
        # 1000 casos por faixa, com a frequência real igual à prevista
        rng = np.random.default_rng(42)
        probs, reais = [], []
        for p in (0.15, 0.35, 0.55, 0.75, 0.95):
            probs += [p] * 2000
            reais += list(rng.binomial(1, p, 2000))
        t = calibration_table(reais, probs, n_bins=10).dropna(subset=["gap"])
        assert (t["gap"].abs() < 0.03).all(), t[["faixa", "gap"]].to_string()

    def test_confianca_demais_aparece_como_gap_positivo(self):
        # promete 90%, entrega 50%
        t = calibration_table([1, 0] * 50, [0.9] * 100, n_bins=10)
        linha = t[t["n"] > 0].iloc[0]
        assert linha["faixa"] == "90%-100%"
        assert linha["gap"] == pytest.approx(0.40, abs=0.01)

    def test_timidez_aparece_como_gap_negativo(self):
        # promete 30%, entrega 100%
        t = calibration_table([1] * 100, [0.3] * 100, n_bins=10)
        linha = t[t["n"] > 0].iloc[0]
        assert linha["gap"] == pytest.approx(-0.70, abs=0.01)

    def test_faixa_vazia_nao_some_nem_quebra(self):
        """Faixa vazia é RESULTADO, não defeito: neste projeto as pontas ficam
        vazias porque o modelo comprime tudo perto de 50%."""
        t = calibration_table([1, 0], [0.5, 0.5], n_bins=10)
        assert len(t) == 10
        vazias = t[t["n"] == 0]
        assert len(vazias) == 9
        assert vazias["gap"].isna().all()

    def test_as_bordas_caem_na_faixa_certa(self):
        t = calibration_table([1, 1, 1], [0.0, 0.5, 1.0], n_bins=10)
        assert t.iloc[0]["n"] == 1        # 0.0 na primeira
        assert t.iloc[-1]["n"] == 1       # 1.0 na ultima
        assert t["n"].sum() == 3          # ninguem se perde

    def test_todo_mundo_entra_em_alguma_faixa(self):
        rng = np.random.default_rng(7)
        p = rng.random(500)
        t = calibration_table(rng.binomial(1, p), p, n_bins=10)
        assert t["n"].sum() == 500


class TestECE:
    def test_calibracao_perfeita_da_perto_de_zero(self):
        rng = np.random.default_rng(1)
        p = rng.random(20000)
        assert expected_calibration_error(rng.binomial(1, p), p) < 0.02

    def test_pondera_pelo_TAMANHO_da_faixa(self):
        """
        Sem ponderar, uma faixa com 2 observações pesaria igual a uma com 2000
        e o ECE viraria refém do ruído da ponta.
        """
        # 1000 casos calibrados + 2 casos absurdos numa faixa isolada
        y = [1] * 500 + [0] * 500 + [0, 0]
        p = [0.5] * 1000 + [0.95, 0.95]
        ece = expected_calibration_error(y, p, n_bins=10)
        assert ece < 0.01, f"a faixa de 2 casos dominou o ECE: {ece:.4f}"

    def test_sem_dado_devolve_nan(self):
        assert np.isnan(expected_calibration_error([], [], n_bins=10))


class TestAviso:
    def test_avisa_quando_a_faixa_e_pequena_demais(self, caplog):
        """
        A tabela CONVIDA a ler linha a linha, e faixa com 10 observações mente
        com confiança: ±15 pp aparecem sozinhos. Foi o que aconteceu na
        comparação com o mercado (99 lutas em 10 faixas).
        """
        rng = np.random.default_rng(3)
        p = rng.random(60)
        with caplog.at_level("WARNING"):
            log_calibration(rng.binomial(1, p), p, "amostra pequena", n_bins=10)
        assert "ruido" in caplog.text

    def test_nao_avisa_com_amostra_confortavel(self, caplog):
        rng = np.random.default_rng(4)
        p = rng.random(5000)
        with caplog.at_level("WARNING"):
            log_calibration(rng.binomial(1, p), p, "amostra grande", n_bins=10)
        assert "ruido" not in caplog.text
