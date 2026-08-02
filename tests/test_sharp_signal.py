"""
Testes do sinal SHARP congelado no historico (sharp_prob / ev_sharp) e da
analise "pernas com respaldo sharp vs sem".

Motivacao (01/ago/2026, apos 4 eventos): as pernas parecem ir bem quando
tem respaldo sharp e mal quando nao tem, mas isso NAO era testavel — o
sinal so existia ao vivo, no line shopping, e a API de odds nao serve
eventos passados. Gravando no pre-registro, a hipotese fica mensuravel.

O que protegemos aqui:
- o sinal e gravado junto da previsao e sobrevive a uma regeracao do
  relatorio sem API (nunca se perde silenciosamente);
- historico antigo (sem as colunas) continua carregando;
- a divisao com/sem sharp so conta pernas do criterio da serie (EV>1) e
  ignora lutas sem o dado.
"""
import numpy as np
import pandas as pd
import pytest

import config
from src import prediction_history as ph


def _fight(a, b, odds_a, odds_b, prob_a, category="favorite"):
    return {"fighter_a": a, "fighter_b": b, "odds_a": odds_a, "odds_b": odds_b,
            "model_prob_a": prob_a, "model_side": a if prob_a >= 0.5 else b,
            "category": category}


def _analysis(fights, model_name="logreg"):
    return {"favorites": [f for f in fights if f["category"] == "favorite"],
            "underdogs": [f for f in fights if f["category"] == "underdog"],
            "no_prediction": [], "model_name": model_name}


@pytest.fixture
def hist(tmp_path, monkeypatch):
    path = tmp_path / "prediction_history.csv"
    monkeypatch.setattr(config, "PREDICTION_HISTORY_CSV", path)
    return path


class TestGravacao:
    def test_grava_sharp_prob_e_ev_sharp(self, hist):
        fights = [_fight("Alice", "Bruna", 2.00, 1.90, 0.60)]
        # sharp da 55% para Alice; ev_sharp = 0.55 * 2.00 = 1.10
        ph.record_card_predictions(_analysis(fights), "UFC X", "2026-08-08", hist,
                                   sharp_probs={("Alice", "Bruna"): 0.55})
        row = pd.read_csv(hist).iloc[0]
        assert row["sharp_prob"] == pytest.approx(0.55)
        assert row["ev_sharp"] == pytest.approx(1.10)

    def test_ev_sharp_usa_a_odd_do_lado_do_modelo(self, hist):
        # modelo aponta o lado B -> ev_sharp deve usar odds_b
        fights = [_fight("Alice", "Bruna", 1.50, 3.00, 0.40, category="underdog")]
        ph.record_card_predictions(_analysis(fights), "UFC X", "2026-08-08", hist,
                                   sharp_probs={("Alice", "Bruna"): 0.40})
        row = pd.read_csv(hist).iloc[0]
        assert row["model_side"] == "Bruna"
        assert row["ev_sharp"] == pytest.approx(0.40 * 3.00)

    def test_sem_sharp_grava_vazio_sem_quebrar(self, hist):
        fights = [_fight("Alice", "Bruna", 2.00, 1.90, 0.60)]
        ph.record_card_predictions(_analysis(fights), "UFC X", "2026-08-08", hist)
        row = pd.read_csv(hist).iloc[0]
        assert pd.isna(row["sharp_prob"]) and pd.isna(row["ev_sharp"])

    def test_regerar_sem_api_nao_apaga_sinal_congelado(self, hist):
        """Regressao critica: o relatorio local roda com sharp=False logo
        apos a publicacao — nao pode zerar o que acabou de ser gravado."""
        fights = [_fight("Alice", "Bruna", 2.00, 1.90, 0.60)]
        ph.record_card_predictions(_analysis(fights), "UFC X", "2026-08-08", hist,
                                   sharp_probs={("Alice", "Bruna"): 0.55})
        ph.record_card_predictions(_analysis(fights), "UFC X", "2026-08-08", hist)  # sem sharp
        row = pd.read_csv(hist).iloc[0]
        assert row["sharp_prob"] == pytest.approx(0.55)

    def test_odd_nova_recalcula_ev_sharp_do_sinal_preservado(self, hist):
        ph.record_card_predictions(_analysis([_fight("Alice", "Bruna", 2.00, 1.90, 0.60)]),
                                   "UFC X", "2026-08-08", hist,
                                   sharp_probs={("Alice", "Bruna"): 0.55})
        # odds se moveram na semana; regeracao sem API
        ph.record_card_predictions(_analysis([_fight("Alice", "Bruna", 2.20, 1.80, 0.60)]),
                                   "UFC X", "2026-08-08", hist)
        row = pd.read_csv(hist).iloc[0]
        assert row["sharp_prob"] == pytest.approx(0.55)
        assert row["ev_sharp"] == pytest.approx(0.55 * 2.20)


class TestCompatibilidade:
    def test_historico_antigo_sem_as_colunas_carrega(self, hist):
        antigo = pd.DataFrame([{
            "event_name": "UFC Velho", "event_date": "2026-07-11", "fighter_a": "Alice",
            "fighter_b": "Bruna", "odds_a_decimal": 2.0, "odds_b_decimal": 1.9,
            "model_name": "logreg", "model_prob_a": 0.6, "model_side": "Alice",
            "actual_winner": "Alice"}])
        antigo.to_csv(hist, index=False)
        df = ph.load_history(hist)
        assert len(df) == 1
        assert pd.isna(df.iloc[0]["sharp_prob"])
        assert ph.compute_series_summary(df)["legs_n"] == 1  # segue funcionando


class TestSplitSharp:
    def _hist_com_pernas(self, path):
        rows = [
            # COM sharp (ev_sharp>1), GANHOU: odd 2.0 -> +1.0
            dict(event_name="UFC X", event_date="2026-08-08", fighter_a="A1", fighter_b="B1",
                 odds_a_decimal=2.0, odds_b_decimal=1.9, model_name="logreg",
                 model_prob_a=0.60, model_side="A1", actual_winner="A1",
                 sharp_prob=0.55, ev_sharp=1.10),
            # SEM sharp (ev_sharp<1), PERDEU: -1.0
            dict(event_name="UFC X", event_date="2026-08-08", fighter_a="A2", fighter_b="B2",
                 odds_a_decimal=2.0, odds_b_decimal=1.9, model_name="logreg",
                 model_prob_a=0.60, model_side="A2", actual_winner="B2",
                 sharp_prob=0.45, ev_sharp=0.90),
            # sem o dado gravado -> fica de fora dos dois grupos
            dict(event_name="UFC Velho", event_date="2026-07-11", fighter_a="A3", fighter_b="B3",
                 odds_a_decimal=2.0, odds_b_decimal=1.9, model_name="logreg",
                 model_prob_a=0.60, model_side="A3", actual_winner="A3",
                 sharp_prob=np.nan, ev_sharp=np.nan),
        ]
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_divide_e_contabiliza_os_dois_grupos(self, hist):
        self._hist_com_pernas(hist)
        s = ph.compute_sharp_split(ph.load_history(hist))
        assert s["com_sharp"] == {"n": 1, "won": 1, "pnl": pytest.approx(1.0)}
        assert s["sem_sharp"] == {"n": 1, "won": 0, "pnl": pytest.approx(-1.0)}

    def test_none_quando_nao_ha_dado(self, hist):
        pd.DataFrame([dict(
            event_name="UFC Velho", event_date="2026-07-11", fighter_a="A", fighter_b="B",
            odds_a_decimal=2.0, odds_b_decimal=1.9, model_name="logreg", model_prob_a=0.6,
            model_side="A", actual_winner="A")]).to_csv(hist, index=False)
        assert ph.compute_sharp_split(ph.load_history(hist)) is None

    def test_painel_mostra_o_bloco_quando_ha_dado(self, hist):
        self._hist_com_pernas(hist)
        html = ph.render_history_panel(ph.load_history(hist))
        assert "Com respaldo sharp vs sem" in html
        assert "pernas COM sinal sharp" in html

    def test_painel_omite_o_bloco_sem_dado(self, hist):
        ph.record_card_predictions(_analysis([_fight("Alice", "Bruna", 2.0, 1.9, 0.6)]),
                                   "UFC X", "2026-08-08", hist)
        html = ph.render_history_panel(ph.load_history(hist))
        assert "Com respaldo sharp vs sem" not in html
