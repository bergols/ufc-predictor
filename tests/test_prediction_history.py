"""
Testes do historico de previsoes (src/prediction_history.py).

A propriedade central protegida aqui e o CONGELAMENTO: previsao gravada no
pre-registro nunca e reescrita depois que a luta tem resultado — re-treinos
posteriores nao podem "melhorar" o passado, senao o historico de acertos
e erros nao teria valor.
"""
import numpy as np
import pandas as pd
import pytest

import config
from src import prediction_history as ph


def _analysis(fights_predicted=None, fights_no_pred=None, model_name="logreg"):
    """Monta o dict minimo que analyze_card retorna, so com o que o
    historico consome."""
    predicted = fights_predicted or []
    return {
        "favorites": [f for f in predicted if f.get("category") == "favorite"],
        "underdogs": [f for f in predicted if f.get("category") == "underdog"],
        "no_prediction": fights_no_pred or [],
        "model_name": model_name,
    }


def _fight(a, b, odds_a, odds_b, prob_a, category="favorite"):
    side = a if prob_a >= 0.5 else b
    return {"fighter_a": a, "fighter_b": b, "odds_a": odds_a, "odds_b": odds_b,
            "model_prob_a": prob_a, "model_side": side, "category": category}


@pytest.fixture
def history_path(tmp_path, monkeypatch):
    path = tmp_path / "prediction_history.csv"
    monkeypatch.setattr(config, "PREDICTION_HISTORY_CSV", path)
    return path


@pytest.fixture
def template_path(tmp_path, monkeypatch):
    path = tmp_path / "odds_template.csv"
    monkeypatch.setattr(config, "ODDS_TEMPLATE_CSV", path)
    return path


class TestRecord:
    def test_grava_previstas_e_sem_previsao(self, history_path):
        analysis = _analysis(
            [_fight("Alice", "Bruna", 1.50, 2.60, 0.65)],
            [{"fighter_a": "Nova", "fighter_b": "Estreante", "odds_a": 1.20, "odds_b": 4.50}])
        n = ph.record_card_predictions(analysis, "UFC Teste", "2026-07-18", history_path)
        assert n == 2
        df = pd.read_csv(history_path)
        assert len(df) == 2
        assert df.iloc[0]["model_side"] == "Alice"
        assert pd.isna(df.iloc[1]["model_side"])  # sem previsao nao some

    def test_regerar_antes_do_evento_atualiza_odds(self, history_path):
        """Odds se movem na semana: regerar o card ANTES do resultado deve
        substituir a linha aberta (ultimo pre-registro vale)."""
        ph.record_card_predictions(_analysis([_fight("Alice", "Bruna", 1.50, 2.60, 0.65)]),
                                   "UFC Teste", "2026-07-18", history_path)
        ph.record_card_predictions(_analysis([_fight("Alice", "Bruna", 1.40, 2.90, 0.61)]),
                                   "UFC Teste", "2026-07-18", history_path)
        df = pd.read_csv(history_path)
        assert len(df) == 1
        assert df.iloc[0]["odds_a_decimal"] == pytest.approx(1.40)
        assert df.iloc[0]["model_prob_a"] == pytest.approx(0.61)

    def test_linha_fechada_e_congelada(self, history_path):
        """Depois que a luta tem resultado, nada a reescreve."""
        ph.record_card_predictions(_analysis([_fight("Alice", "Bruna", 1.50, 2.60, 0.65)]),
                                   "UFC Teste", "2026-07-18", history_path)
        df = pd.read_csv(history_path)
        df["actual_winner"] = df["actual_winner"].astype("object")
        df.loc[0, "actual_winner"] = "Alice"
        df.to_csv(history_path, index=False)

        n = ph.record_card_predictions(_analysis([_fight("Alice", "Bruna", 1.10, 8.00, 0.99)]),
                                       "UFC Teste", "2026-07-18", history_path)
        assert n == 0
        df = pd.read_csv(history_path)
        assert df.iloc[0]["model_prob_a"] == pytest.approx(0.65)  # intocada
        assert df.iloc[0]["odds_a_decimal"] == pytest.approx(1.50)

    def test_lados_invertidos_sao_a_mesma_luta(self, history_path):
        ph.record_card_predictions(_analysis([_fight("Alice", "Bruna", 1.50, 2.60, 0.65)]),
                                   "UFC Teste", "2026-07-18", history_path)
        ph.record_card_predictions(_analysis([_fight("Bruna", "Alice", 2.60, 1.50, 0.35)]),
                                   "UFC Teste", "2026-07-18", history_path)
        assert len(pd.read_csv(history_path)) == 1


class TestSync:
    def test_puxa_vencedor_do_template(self, history_path, template_path):
        ph.record_card_predictions(_analysis([_fight("Alice", "Bruna", 1.50, 2.60, 0.65)]),
                                   "UFC Teste", "2026-07-18", history_path)
        pd.DataFrame([{"event_name": "UFC Teste", "event_date": "2026-07-18",
                       "fighter_a": "Bruna", "fighter_b": "Alice",  # ordem invertida de proposito
                       "odds_a_decimal": 2.60, "odds_b_decimal": 1.50,
                       "actual_winner": "Alice"}]).to_csv(template_path, index=False)
        assert ph.sync_results_from_template(history_path, template_path) == 1
        df = pd.read_csv(history_path)
        assert df.iloc[0]["actual_winner"] == "Alice"

    def test_vencedor_que_nao_e_nenhum_dos_lados_e_ignorado(self, history_path, template_path, caplog):
        ph.record_card_predictions(_analysis([_fight("Alice", "Bruna", 1.50, 2.60, 0.65)]),
                                   "UFC Teste", "2026-07-18", history_path)
        pd.DataFrame([{"event_name": "UFC Teste", "event_date": "2026-07-18",
                       "fighter_a": "Alice", "fighter_b": "Bruna",
                       "odds_a_decimal": 1.50, "odds_b_decimal": 2.60,
                       "actual_winner": "Nome Errado"}]).to_csv(template_path, index=False)
        assert ph.sync_results_from_template(history_path, template_path) == 0
        assert pd.isna(pd.read_csv(history_path).iloc[0]["actual_winner"])

    def test_sem_template_e_noop(self, history_path, tmp_path):
        ph.record_card_predictions(_analysis([_fight("Alice", "Bruna", 1.50, 2.60, 0.65)]),
                                   "UFC Teste", "2026-07-18", history_path)
        assert ph.sync_results_from_template(history_path, tmp_path / "nao_existe.csv") == 0


class TestLoadAndRender:
    def _closed_history(self, history_path, template_path):
        ph.record_card_predictions(
            _analysis([_fight("Alice", "Bruna", 1.50, 2.60, 0.65),
                       _fight("Carla", "Dora", 2.40, 1.60, 0.55, category="underdog")]),
            "UFC Teste", "2026-07-18", history_path)
        pd.DataFrame([
            {"event_name": "UFC Teste", "event_date": "2026-07-18", "fighter_a": "Alice",
             "fighter_b": "Bruna", "odds_a_decimal": 1.50, "odds_b_decimal": 2.60,
             "actual_winner": "Alice"},
            {"event_name": "UFC Teste", "event_date": "2026-07-18", "fighter_a": "Carla",
             "fighter_b": "Dora", "odds_a_decimal": 2.40, "odds_b_decimal": 1.60,
             "actual_winner": "Dora"},
        ]).to_csv(template_path, index=False)
        ph.sync_results_from_template(history_path, template_path)

    def test_acertos_e_erros_de_modelo_e_mercado(self, history_path, template_path):
        self._closed_history(history_path, template_path)
        df = ph.load_history(history_path)
        alice = df[df["fighter_a"] == "Alice"].iloc[0]
        assert alice["market_side"] == "Alice"
        assert alice["model_correct"] == True   # noqa: E712 (valor pandas)
        assert alice["market_correct"] == True  # noqa: E712
        carla = df[df["fighter_a"] == "Carla"].iloc[0]
        assert carla["market_side"] == "Dora"
        assert carla["model_correct"] == False  # noqa: E712 (modelo apontou Carla, zebra)
        assert carla["market_correct"] == True  # noqa: E712

    def test_pickem_exato_nao_tem_lado_de_mercado(self, history_path):
        ph.record_card_predictions(_analysis([_fight("Eva", "Fabi", 1.909, 1.909, 0.60)]),
                                   "UFC Teste", "2026-07-18", history_path)
        df = ph.load_history(history_path)
        assert pd.isna(df.iloc[0]["market_side"])

    def test_painel_html_mostra_placar_e_badges(self, history_path, template_path):
        self._closed_history(history_path, template_path)
        html = ph.render_history_panel(ph.load_history(history_path))
        assert "UFC Teste" in html
        assert "modelo 1/2" in html
        assert "mercado 2/2" in html
        assert "✓ acertou" in html and "✗ errou" in html

    def test_painel_evento_aberto_mostra_aguardando(self, history_path):
        ph.record_card_predictions(_analysis([_fight("Alice", "Bruna", 1.50, 2.60, 0.65)]),
                                   "UFC Futuro", "2027-01-01", history_path)
        html = ph.render_history_panel(ph.load_history(history_path))
        assert "aguardando resultados" in html

    def test_historico_vazio_nao_quebra(self, history_path):
        html = ph.render_history_panel(ph.load_history(history_path))
        assert "Nenhum evento registrado" in html
