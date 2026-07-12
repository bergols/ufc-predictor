"""
Testes do fallback de previsao AO VIVO em evaluate.compare_to_market.

Contexto: o template manual de odds (data/odds_template.csv) e usado para
paper trading de eventos que acabaram de acontecer -- lutas que NAO estao
em test_predictions.csv porque sao posteriores ao fim da base de treino.
Antes do fallback, essas lutas eram todas puladas em silencio (WARNING) e
o fluxo documentado no README nao funcionava para o caso de uso principal.

Regras cobertas aqui:
- luta no teste historico -> usa a probabilidade gravada (point-in-time);
- luta posterior ao fim do teste -> preve ao vivo com o modelo atual;
- luta fora do teste mas ANTERIOR ao fim dele -> pulada (anti-vazamento:
  prever o passado com stats de hoje usaria informacao do futuro);
- lutador fora da base (estreante) -> pulada com aviso, sem quebrar;
- event_date ausente -> pulada (sem data nao da para provar que nao e
  vazamento).
"""
import logging

import pandas as pd
import pytest

from src import evaluate


@pytest.fixture
def preds_csv(tmp_path):
    """test_predictions.csv minimo: uma luta conhecida, teste termina em 2026-06-27."""
    path = tmp_path / "test_predictions.csv"
    pd.DataFrame({
        "fight_id": ["f1"],
        "event_date": ["2026-06-27"],
        "fighter_a": ["Alice Alpha"],
        "fighter_b": ["Bruna Beta"],
        "label": [1],
        "pred_logreg": [0.70],
        "pred_gbm": [0.65],
    }).to_csv(path, index=False)
    return path


def _write_odds(tmp_path, rows):
    path = tmp_path / "odds_template.csv"
    pd.DataFrame(rows, columns=["event_name", "event_date", "fighter_a", "fighter_b",
                                "odds_a_decimal", "odds_b_decimal", "actual_winner"]
                 ).to_csv(path, index=False)
    return path


def _fake_predict_factory(prob_a=0.60, unknown=()):
    def fake_predict_fight(fighter_a, fighter_b, model_name="gbm", levels=None):
        if fighter_a in unknown or fighter_b in unknown:
            raise ValueError(f"Lutador desconhecido em {fighter_a} vs {fighter_b}")
        return {"fighter_a": fighter_a, "fighter_b": fighter_b,
                "prob_a_wins": prob_a, "prob_b_wins": round(1 - prob_a, 4),
                "model_used": model_name,
                "fighter_a_low_experience": False, "fighter_b_low_experience": False}
    return fake_predict_fight


@pytest.fixture
def live_stubs(monkeypatch):
    """Evita carregar a base real: stub do predict_fight e dos levels."""
    import src.features
    import src.predict
    monkeypatch.setattr(src.features, "export_latest_fighter_levels",
                        lambda: pd.DataFrame({"fighter": ["stub"]}))
    monkeypatch.setattr(src.predict, "predict_fight", _fake_predict_factory())
    return monkeypatch


class TestCompareToMarketFallback:
    def test_luta_do_teste_usa_probabilidade_gravada(self, tmp_path, preds_csv):
        odds = _write_odds(tmp_path, [
            ["Evento X", "2026-06-27", "Alice Alpha", "Bruna Beta", 1.50, 2.60, "Alice Alpha"],
        ])
        df = evaluate.compare_to_market(odds, preds_csv, model_name="gbm")
        assert len(df) == 1
        assert df.iloc[0]["prediction_source"] == "test_set"
        assert df.iloc[0]["model_prob_a"] == pytest.approx(0.65)

    def test_respeita_model_name_na_via_do_teste(self, tmp_path, preds_csv):
        odds = _write_odds(tmp_path, [
            ["Evento X", "2026-06-27", "Alice Alpha", "Bruna Beta", 1.50, 2.60, "Alice Alpha"],
        ])
        df = evaluate.compare_to_market(odds, preds_csv, model_name="logreg")
        assert df.iloc[0]["model_prob_a"] == pytest.approx(0.70)

    def test_luta_espelhada_inverte_probabilidade(self, tmp_path, preds_csv):
        """Mesma luta com A/B trocados no template deve usar 1 - p."""
        odds = _write_odds(tmp_path, [
            ["Evento X", "2026-06-27", "Bruna Beta", "Alice Alpha", 2.60, 1.50, "Alice Alpha"],
        ])
        df = evaluate.compare_to_market(odds, preds_csv, model_name="gbm")
        assert df.iloc[0]["model_prob_a"] == pytest.approx(0.35)

    def test_luta_pos_teste_e_prevista_ao_vivo(self, tmp_path, preds_csv, live_stubs):
        odds = _write_odds(tmp_path, [
            ["UFC Novo", "2026-07-11", "Carla Gama", "Dora Delta", 1.80, 2.10, "Carla Gama"],
        ])
        df = evaluate.compare_to_market(odds, preds_csv, model_name="gbm")
        assert len(df) == 1
        assert df.iloc[0]["prediction_source"] == "live_model"
        assert df.iloc[0]["model_prob_a"] == pytest.approx(0.60)

    def test_luta_antiga_fora_do_teste_e_pulada(self, tmp_path, preds_csv, live_stubs, caplog):
        """Anti-vazamento: luta de antes do fim do teste que nao esta nas
        predicoes NAO pode ser prevista com os stats de hoje."""
        odds = _write_odds(tmp_path, [
            ["UFC Antigo", "2020-01-01", "Carla Gama", "Dora Delta", 1.80, 2.10, "Carla Gama"],
        ])
        with caplog.at_level(logging.WARNING, logger="src.evaluate"):
            df = evaluate.compare_to_market(odds, preds_csv, model_name="gbm")
        assert df.empty
        assert "vazamento" in caplog.text

    def test_sem_event_date_e_pulada(self, tmp_path, preds_csv, live_stubs):
        odds = _write_odds(tmp_path, [
            ["UFC Sem Data", None, "Carla Gama", "Dora Delta", 1.80, 2.10, "Carla Gama"],
        ])
        df = evaluate.compare_to_market(odds, preds_csv, model_name="gbm")
        assert df.empty

    def test_estreante_e_pulado_sem_quebrar(self, tmp_path, preds_csv, monkeypatch, caplog):
        import src.features
        import src.predict
        monkeypatch.setattr(src.features, "export_latest_fighter_levels",
                            lambda: pd.DataFrame({"fighter": ["stub"]}))
        monkeypatch.setattr(src.predict, "predict_fight",
                            _fake_predict_factory(unknown={"Estreante Epsilon"}))
        odds = _write_odds(tmp_path, [
            ["UFC Novo", "2026-07-11", "Estreante Epsilon", "Dora Delta", 1.20, 4.50, "Dora Delta"],
            ["UFC Novo", "2026-07-11", "Carla Gama", "Dora Delta", 1.80, 2.10, "Carla Gama"],
        ])
        with caplog.at_level(logging.WARNING, logger="src.evaluate"):
            df = evaluate.compare_to_market(odds, preds_csv, model_name="gbm")
        assert len(df) == 1  # so a luta prevista entra; a do estreante nao derruba o resto
        assert "sem previsao ao vivo" in caplog.text

    def test_model_name_invalido_da_erro_claro(self, tmp_path, preds_csv):
        odds = _write_odds(tmp_path, [
            ["Evento X", "2026-06-27", "Alice Alpha", "Bruna Beta", 1.50, 2.60, "Alice Alpha"],
        ])
        with pytest.raises(ValueError, match="model_name"):
            evaluate.compare_to_market(odds, preds_csv, model_name="xgboost")
