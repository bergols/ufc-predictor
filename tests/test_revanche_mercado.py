"""
Revanche em src/evaluate.py::compare_to_market.

O caminho casava odds_template x test_predictions SO POR NOME e pegava
`match.iloc[0]` -- a primeira ocorrencia, que numa revanche e a luta ANTIGA.

Medido em 05/nov/2026, nos dados reais:

    Joshua Van x Alexandre Pantoja   template 19/set  ->  pegava 06/dez/2025
                                     0,5629 no lugar de 0,7083
    Petr Yan x Merab Dvalishvili     template 24/out  ->  pegava 06/dez/2025
                                     0,3604 no lugar de 0,5069

Duas lutas de 165, erradas por ~14,5 pp, contaminando a comparacao com o
mercado. Cinco pares do conjunto de teste tem revanche.
"""
import pandas as pd
import pytest

import config
from src.evaluate import compare_to_market

ANTIGA, NOVA = "2025-12-06", "2026-09-19"


@pytest.fixture
def cenario(tmp_path):
    """Um par com DUAS lutas no teste e previsoes bem diferentes."""
    preds = tmp_path / "test_predictions.csv"
    pd.DataFrame([
        {"fight_id": "f1", "event_date": ANTIGA, "fighter_a": "Joshua Van",
         "fighter_b": "Alexandre Pantoja", "label": 1, "pred_logreg": 0.5629, "pred_gbm": 0.56},
        {"fight_id": "f2", "event_date": NOVA, "fighter_a": "Joshua Van",
         "fighter_b": "Alexandre Pantoja", "label": 1, "pred_logreg": 0.7083, "pred_gbm": 0.70},
    ]).to_csv(preds, index=False)

    def template(data, a="Joshua Van", b="Alexandre Pantoja"):
        p = tmp_path / f"odds_{data}_{a[:4]}.csv"
        pd.DataFrame([{"fight_id": "", "event_name": "UFC X", "event_date": data,
                       "fighter_a": a, "fighter_b": b, "odds_a_decimal": 1.80,
                       "odds_b_decimal": 2.10, "actual_winner": a}]).to_csv(p, index=False)
        return p
    return preds, template


class TestRevancha:
    def test_pega_a_luta_da_DATA_da_linha(self, cenario):
        preds, template = cenario
        df = compare_to_market(template(NOVA), preds, model_name="logreg")
        assert len(df) == 1
        assert df.iloc[0]["model_prob_a"] == pytest.approx(0.7083)

    def test_a_luta_ANTIGA_tambem_casa_certo(self, cenario):
        """A correcao nao pode simplesmente preferir a mais recente."""
        preds, template = cenario
        df = compare_to_market(template(ANTIGA), preds, model_name="logreg")
        assert df.iloc[0]["model_prob_a"] == pytest.approx(0.5629)

    def test_lados_invertidos_na_linha_invertem_a_probabilidade(self, cenario):
        preds, template = cenario
        df = compare_to_market(template(NOVA, "Alexandre Pantoja", "Joshua Van"),
                               preds, model_name="logreg")
        assert df.iloc[0]["model_prob_a"] == pytest.approx(1 - 0.7083)

    def test_revanche_sem_data_na_linha_e_PULADA(self, cenario, tmp_path, caplog):
        """Sem data nao da para saber qual luta e. Escolher errado contamina
        a comparacao; pular so reduz a amostra."""
        preds, _ = cenario
        p = tmp_path / "sem_data.csv"
        pd.DataFrame([{"fight_id": "", "event_name": "UFC X", "event_date": None,
                       "fighter_a": "Joshua Van", "fighter_b": "Alexandre Pantoja",
                       "odds_a_decimal": 1.8, "odds_b_decimal": 2.1,
                       "actual_winner": "Joshua Van"}]).to_csv(p, index=False)
        with caplog.at_level("WARNING"):
            df = compare_to_market(p, preds, model_name="logreg")
        assert df.empty
        assert "nao tem event_date" in caplog.text

    def test_data_que_nao_existe_para_aquele_par_e_PULADA(self, cenario, tmp_path, caplog):
        preds, _ = cenario
        p = tmp_path / "data_errada.csv"
        pd.DataFrame([{"fight_id": "", "event_name": "UFC X", "event_date": "2030-01-01",
                       "fighter_a": "Joshua Van", "fighter_b": "Alexandre Pantoja",
                       "odds_a_decimal": 1.8, "odds_b_decimal": 2.1,
                       "actual_winner": "Joshua Van"}]).to_csv(p, index=False)
        with caplog.at_level("WARNING"):
            compare_to_market(p, preds, model_name="logreg")
        assert "nenhuma em 2030-01-01" in caplog.text

    def test_par_SEM_revanche_continua_casando_por_nome(self, tmp_path):
        """A correcao nao pode exigir data onde nunca houve ambiguidade."""
        preds = tmp_path / "p.csv"
        pd.DataFrame([{"fight_id": "f1", "event_date": ANTIGA, "fighter_a": "Alice",
                       "fighter_b": "Bia", "label": 1, "pred_logreg": 0.61,
                       "pred_gbm": 0.6}]).to_csv(preds, index=False)
        t = tmp_path / "t.csv"
        pd.DataFrame([{"fight_id": "", "event_name": "UFC Y", "event_date": None,
                       "fighter_a": "Alice", "fighter_b": "Bia", "odds_a_decimal": 1.7,
                       "odds_b_decimal": 2.2, "actual_winner": "Alice"}]).to_csv(t, index=False)
        df = compare_to_market(t, preds, model_name="logreg")
        assert len(df) == 1 and df.iloc[0]["model_prob_a"] == pytest.approx(0.61)


@pytest.mark.skipif(not (config.ODDS_TEMPLATE_CSV.exists()
                         and (config.PROCESSED_DIR/"test_predictions.csv").exists()),
                    reason="artefatos ausentes")
def test_nos_dados_reais_as_duas_revanches_pegam_a_luta_certa():
    # O esperado sai do test_predictions.csv ATUAL, pela data da revanche --
    # nao de um numero fixo. Ate 11/nov/2026 este teste fixava 0,7083 e
    # 0,4931, que eram a saida do modelo de 06/nov: qualquer retreino (o do
    # ciclo semanal inclusive) mudava a terceira casa e reprovava a suite sem
    # que o casamento tivesse mudado. O que se protege aqui e a escolha da
    # luta, nao o valor do modelo.
    df = compare_to_market(model_name="logreg")
    preds = pd.read_csv(config.PROCESSED_DIR / "test_predictions.csv")
    revanches = {("Joshua Van", "Alexandre Pantoja"): (NOVA, ANTIGA),
                 ("Petr Yan", "Merab Dvalishvili"): ("2026-10-24", ANTIGA)}
    achou = 0
    for (a, b), (data_nova, data_antiga) in revanches.items():
        linha = df[(df["fighter_a"] == a) & (df["fighter_b"] == b)]
        if linha.empty:
            continue
        do_par = preds[(preds["fighter_a"] == a) & (preds["fighter_b"] == b)]
        nova = do_par[do_par["event_date"].astype(str) == data_nova]["pred_logreg"]
        antiga = do_par[do_par["event_date"].astype(str) == data_antiga]["pred_logreg"]
        assert len(nova) == 1 and len(antiga) == 1, f"{a} x {b}: lutas fora do teste"
        # a luta antiga precisa ser distinguivel, senao o teste nao prova nada
        assert abs(nova.iloc[0] - antiga.iloc[0]) > 0.05, f"{a} x {b}"
        achou += 1
        assert linha.iloc[0]["model_prob_a"] == pytest.approx(nova.iloc[0], abs=1e-9), f"{a} x {b}"
    if achou == 0:
        pytest.skip("as revanches sairam do template")
