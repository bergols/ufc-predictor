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


# o record so aceita reescrever linha de evento FUTURO (ver
# TestLinhaAbertaDeEventoPassado); data fixa envelhece e inverte o teste
AMANHA = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


@pytest.fixture
def hist(tmp_path, monkeypatch):
    path = tmp_path / "prediction_history.csv"
    monkeypatch.setattr(config, "PREDICTION_HISTORY_CSV", path)
    return path


class TestGravacao:
    def test_grava_as_tres_colunas_do_sinal(self, hist):
        fights = [_fight("Alice", "Bruna", 2.00, 1.90, 0.60)]
        ph.record_card_predictions(_analysis(fights), "UFC X", AMANHA, hist,
                                   sharp_probs={("Alice", "Bruna"): {"sharp_prob": 0.55, "best_odd": 2.10}})
        row = pd.read_csv(hist).iloc[0]
        assert row["sharp_prob"] == pytest.approx(0.55)
        assert row["sharp_best_odd"] == pytest.approx(2.10)
        assert row["ev_sharp"] == pytest.approx(0.55 * 2.10)

    def test_ev_sharp_usa_a_MELHOR_odd_nao_a_registrada(self, hist):
        """
        Regressao da 1a versao: usar a odd registrada (mediana das casas,
        com vig embutido) fazia ev_sharp ficar < 1 SEMPRE — a metrica nunca
        detectaria sinal nenhum. O sinal e "alguma casa paga acima do justo
        da sharp", entao a odd relevante e a melhor do mercado.
        """
        fights = [_fight("Alice", "Bruna", 1.90, 1.90, 0.60)]  # registrada 1.90
        ph.record_card_predictions(_analysis(fights), "UFC X", AMANHA, hist,
                                   sharp_probs={("Alice", "Bruna"): {"sharp_prob": 0.55, "best_odd": 2.10}})
        row = pd.read_csv(hist).iloc[0]
        assert row["ev_sharp"] == pytest.approx(0.55 * 2.10)   # 1.155 -> sinal!
        assert row["ev_sharp"] != pytest.approx(0.55 * 1.90)   # 1.045 seria a errada

    def test_sem_sharp_grava_vazio_sem_quebrar(self, hist):
        fights = [_fight("Alice", "Bruna", 2.00, 1.90, 0.60)]
        ph.record_card_predictions(_analysis(fights), "UFC X", AMANHA, hist)
        row = pd.read_csv(hist).iloc[0]
        assert pd.isna(row["sharp_prob"]) and pd.isna(row["ev_sharp"])

    def test_regerar_sem_api_nao_apaga_sinal_congelado(self, hist):
        """Regressao critica: o relatorio local roda com sharp=False logo
        apos a publicacao — nao pode zerar o que acabou de ser gravado."""
        fights = [_fight("Alice", "Bruna", 2.00, 1.90, 0.60)]
        ph.record_card_predictions(_analysis(fights), "UFC X", AMANHA, hist,
                                   sharp_probs={("Alice", "Bruna"): {"sharp_prob": 0.55, "best_odd": 2.10}})
        ph.record_card_predictions(_analysis(fights), "UFC X", AMANHA, hist)  # sem sharp
        row = pd.read_csv(hist).iloc[0]
        assert row["sharp_prob"] == pytest.approx(0.55)

    def test_sinal_preservado_mantem_as_tres_colunas_coerentes(self, hist):
        """O sinal e uma medicao do momento (prob sharp x melhor odd de
        entao); ao regerar sem API ele e mantido inteiro, nao recalculado
        com odds novas — senao viraria uma mistura de dois instantes."""
        ph.record_card_predictions(_analysis([_fight("Alice", "Bruna", 2.00, 1.90, 0.60)]),
                                   "UFC X", AMANHA, hist,
                                   sharp_probs={("Alice", "Bruna"): {"sharp_prob": 0.55, "best_odd": 2.10}})
        # odds se moveram na semana; regeracao sem API
        ph.record_card_predictions(_analysis([_fight("Alice", "Bruna", 2.20, 1.80, 0.60)]),
                                   "UFC X", AMANHA, hist)
        row = pd.read_csv(hist).iloc[0]
        assert row["sharp_prob"] == pytest.approx(0.55)
        assert row["sharp_best_odd"] == pytest.approx(2.10)
        assert row["ev_sharp"] == pytest.approx(0.55 * 2.10)
        assert row["odds_a_decimal"] == pytest.approx(2.20)  # odd registrada atualiza


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
                                   "UFC X", AMANHA, hist)
        html = ph.render_history_panel(ph.load_history(hist))
        assert "Com respaldo sharp vs sem" not in html


class TestSinalNaNossaCasa:
    """
    `ev_sharp` usa a MELHOR odd entre as casas da The Odds API -- um livro
    onde nao apostamos. Marca valor que nao da para pegar.

    `ev_sharp_tp` responde a pergunta que decide uma entrada: a Thunderpick
    paga acima do preco justo da Pinnacle devigada? Os dois convivem; o
    antigo tem 60+ pernas de historico e nao muda de definicao no meio.
    """
    def _fight(self, prob_a=0.62, tp=(1.90, 2.05), lado_a=True):
        f = {"fighter_a": "Alice", "fighter_b": "Bruna", "odds_a": 1.80,
             "odds_b": 2.10, "model_prob_a": prob_a,
             "model_side": "Alice" if lado_a else "Bruna", "category": "favorite",
             "method_probs": None, "odds_a_casa": 1.80, "odds_b_casa": 2.10,
             "odds_da_casa": False}
        if tp:
            f.update(odds_a_casa=tp[0], odds_b_casa=tp[1], odds_da_casa=True)
        return f

    def _grava(self, hist, fight, sharp=0.55):
        kw = {}
        if sharp is not None:
            kw["sharp_probs"] = {("Alice", "Bruna"):
                                 {"sharp_prob": sharp, "best_odd": 2.00}}
        ph.record_card_predictions(_analysis([fight]), "UFC X", AMANHA, hist, **kw)
        return ph._load_raw(hist).iloc[0]

    def test_usa_a_odd_da_casa_do_lado_do_modelo(self, hist):
        r = self._grava(hist, self._fight(tp=(1.90, 2.05)), sharp=0.55)
        assert r["ev_sharp_tp"] == pytest.approx(0.55 * 1.90, abs=1e-4)
        # o antigo segue na melhor odd da API, intocado
        assert r["ev_sharp"] == pytest.approx(0.55 * 2.00, abs=1e-4)

    def test_lado_B_usa_a_odd_B(self, hist):
        r = self._grava(hist, self._fight(prob_a=0.40, lado_a=False), sharp=0.55)
        assert r["ev_sharp_tp"] == pytest.approx(0.55 * 2.05, abs=1e-4)

    def test_sem_odd_da_casa_fica_vazio(self, hist):
        """Nunca aproximado pela mediana: ela embute vig e empurraria o
        produto para baixo de 1 por construcao."""
        r = self._grava(hist, self._fight(tp=None), sharp=0.55)
        assert pd.isna(r["ev_sharp_tp"])

    def test_sem_sharp_fica_vazio(self, hist):
        r = self._grava(hist, self._fight(), sharp=None)
        assert pd.isna(r["ev_sharp_tp"])

    def test_regerar_RECALCULA_com_o_preco_novo(self, hist):
        """A linha se move na semana. Copiar o valor antigo deixaria o sinal
        descrevendo um preco que nao existe mais."""
        self._grava(hist, self._fight(tp=(1.90, 2.05)), sharp=0.55)
        r = self._grava(hist, self._fight(tp=(2.15, 1.85)), sharp=None)  # sem API
        assert r["sharp_prob"] == pytest.approx(0.55)      # recuperado
        assert r["ev_sharp_tp"] == pytest.approx(0.55 * 2.15, abs=1e-4)


class TestSharpCongeladoParaOCard:
    def test_normaliza_para_o_lado_A(self, hist):
        ph.record_card_predictions(
            _analysis([_fight("Alice", "Bruna", 1.5, 2.6, 0.65)]), "UFC X", AMANHA,
            hist, sharp_probs={("Alice", "Bruna"):
                               {"sharp_prob": 0.58, "best_odd": 1.9}})
        assert ph.frozen_sharp_for_event(AMANHA, hist)[("Alice", "Bruna")] == \
            pytest.approx(0.58)

    def test_inverte_quando_o_modelo_aponta_o_lado_B(self, hist):
        """O historico guarda o sharp do lado do MODELO. Quem desenha o card
        precisa dos dois, e a devigagem de duas vias soma 1 -- inverter e
        exato, nao estimativa."""
        ph.record_card_predictions(
            _analysis([_fight("Alice", "Bruna", 1.5, 2.6, 0.35)]), "UFC X", AMANHA,
            hist, sharp_probs={("Alice", "Bruna"):
                               {"sharp_prob": 0.58, "best_odd": 1.9}})
        assert ph.frozen_sharp_for_event(AMANHA, hist)[("Alice", "Bruna")] == \
            pytest.approx(0.42)

    def test_sem_sharp_nao_entra(self, hist):
        ph.record_card_predictions(
            _analysis([_fight("Alice", "Bruna", 1.5, 2.6, 0.65)]), "UFC X", AMANHA, hist)
        assert ph.frozen_sharp_for_event(AMANHA, hist) == {}
