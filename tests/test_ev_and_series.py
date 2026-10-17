"""
Testes da aba "Pernas EV>1" (card_report) e do placar acumulado da serie
(prediction_history.compute_series_summary).

Regra uniforme protegida aqui: perna = lado do modelo com
p_modelo x odd > 1 — TODA perna conta (inclusive borderline), 1 unidade
cada. O placar acumulado e 100% derivado das previsoes congeladas do
historico; nada e recalculado com modelos atuais.
"""
import pandas as pd
import pytest

import config
from src import card_report
from src import prediction_history as ph


def _mk_predict(probs: dict):
    def predict_fn(a, b):
        return {"fighter_a": a, "fighter_b": b,
                "prob_a_wins": probs[(a, b)], "prob_b_wins": 1 - probs[(a, b)],
                "model_used": "logreg",
                "fighter_a_low_experience": False, "fighter_b_low_experience": False,
                "fighter_a_debutant": False, "fighter_b_debutant": False}
    return predict_fn


class TestEvLegs:
    def test_seleciona_e_ordena_por_ev(self):
        card = pd.DataFrame([
            # favorito com EV>1: 0.70 x 1.60 = 1.12
            {"fighter_a": "Alice", "fighter_b": "Bruna", "odds_a_decimal": 1.60,
             "odds_b_decimal": 2.50, "scheduled_rounds": 3},
            # zebra com EV alto: lado B 0.55 x 2.60 = 1.43
            {"fighter_a": "Carla", "fighter_b": "Dora", "odds_a_decimal": 1.55,
             "odds_b_decimal": 2.60, "scheduled_rounds": 3},
            # sem valor: 0.60 x 1.40 = 0.84
            {"fighter_a": "Eva", "fighter_b": "Fabi", "odds_a_decimal": 1.40,
             "odds_b_decimal": 3.10, "scheduled_rounds": 3},
        ])
        probs = {("Alice", "Bruna"): 0.70, ("Carla", "Dora"): 0.45, ("Eva", "Fabi"): 0.60}
        analysis = card_report.analyze_card(card, predict_fn=_mk_predict(probs), method_fn=None)
        legs = analysis["ev_legs"]
        assert [f["model_side"] for f in legs] == ["Dora", "Alice"]  # EV desc
        assert legs[0]["ev"] == pytest.approx(0.55 * 2.60)
        assert legs[0]["model_side_odds"] == pytest.approx(2.60)

    def test_render_tem_aba_com_aviso_e_chips(self):
        card = pd.DataFrame([{"fighter_a": "Alice", "fighter_b": "Bruna",
                              "odds_a_decimal": 1.60, "odds_b_decimal": 2.50,
                              "scheduled_rounds": 3}])
        analysis = card_report.analyze_card(
            card, predict_fn=_mk_predict({("Alice", "Bruna"): 0.70}), method_fn=None)
        html = card_report.render_html(analysis, freshness_gap_days=5)
        assert 'data-tab="ev"' in html and "Pernas EV" in html
        # o valor do EV e o numero em destaque do card (rotulo depois, como
        # em placar de transmissao): "1.12<small>EV</small>"
        assert "1.12<small>EV</small>" in html
        assert "auto-referente" in html  # aviso honesto obrigatorio
        assert "pré-registro do paper trading" in html


class TestSeriesSummary:
    def _history(self):
        rows = [
            # evento fechado: 2 pernas EV>1 (uma ganha odd 2.0 -> +1.0;
            # uma perde -> -1.0), modelo 1/2, mercado 2/2
            dict(event_name="UFC A", event_date="2026-07-11", fighter_a="Alice",
                 fighter_b="Bruna", odds_a_decimal=2.0, odds_b_decimal=1.9,
                 model_name="logreg", model_prob_a=0.60, model_side="Alice",
                 actual_winner="Alice"),   # perna EV 1.2, GANHA (+1.0); mercado favorito=Bruna? 1.9<2.0 -> Bruna... ver abaixo
            dict(event_name="UFC A", event_date="2026-07-11", fighter_a="Carla",
                 fighter_b="Dora", odds_a_decimal=1.4, odds_b_decimal=3.2,
                 model_name="logreg", model_prob_a=0.35, model_side="Dora",
                 actual_winner="Carla"),   # perna Dora EV 0.65x3.2=2.08, PERDE (-1)
            # evento aberto: nao entra no acumulado
            dict(event_name="UFC B", event_date="2026-07-18", fighter_a="Eva",
                 fighter_b="Fabi", odds_a_decimal=1.6, odds_b_decimal=2.5,
                 model_name="logreg", model_prob_a=0.70, model_side="Eva",
                 actual_winner=None),
        ]
        return pd.DataFrame(rows)

    def test_acumulado_so_com_eventos_fechados(self, tmp_path, monkeypatch):
        path = tmp_path / "hist.csv"
        self._history().to_csv(path, index=False)
        df = ph.load_history(path)
        s = ph.compute_series_summary(df)
        assert s["n_events"] == 1
        assert s["model_n"] == 2 and s["model_hits"] == 1
        # pernas: Alice 0.6x2.0=1.2 ganhou (+1.0); Dora 0.65x3.2=2.08 perdeu (-1)
        assert s["legs_n"] == 2 and s["legs_won"] == 1
        assert s["legs_pnl"] == pytest.approx(0.0)

    def test_historico_vazio_ou_todo_aberto_retorna_none(self, tmp_path):
        assert ph.compute_series_summary(pd.DataFrame()) is None
        path = tmp_path / "hist.csv"
        h = self._history()
        h["actual_winner"] = None
        h.to_csv(path, index=False)
        assert ph.compute_series_summary(ph.load_history(path)) is None

    def test_painel_mostra_serie_acumulada(self, tmp_path):
        path = tmp_path / "hist.csv"
        self._history().to_csv(path, index=False)
        html = ph.render_history_panel(ph.load_history(path))
        assert "Série acumulada" in html
        assert "+0.00u" in html
        assert "1/2 pernas ganhas" in html
        assert "aguardando resultados" in html  # evento aberto continua listado

    def test_perna_do_lado_b_usa_odd_b(self):
        row = pd.Series(dict(fighter_a="Carla", fighter_b="Dora", odds_a_decimal=1.4,
                             odds_b_decimal=3.2, model_prob_a=0.35, model_side="Dora"))
        leg = ph.model_side_leg(row)
        assert leg["prob"] == pytest.approx(0.65)
        assert leg["odd"] == pytest.approx(3.2)
        assert leg["ev"] == pytest.approx(0.65 * 3.2)


class TestOddDaCasa:
    """
    A odd que decide a perna e a da Thunderpick, nao a mediana das casas: e
    la que a entrada e feita. A mediana fica como benchmark de mercado.

    Medido no card de 17/out/2026: Kyle Nelson pagava 3,40 na mediana e 3,91
    na casa; Steven Koslow, 1,925 contra 1,56. Perna muda de lista dependendo
    de qual odd se usa, entao isto nao e detalhe de exibicao.
    """
    def _row(self, **kw):
        base = dict(fighter_a="Carla", fighter_b="Dora", odds_a_decimal=1.40,
                    odds_b_decimal=3.20, model_prob_a=0.35, model_side="Dora",
                    odds_a_tp=None, odds_b_tp=None)
        return pd.Series({**base, **kw})

    def test_perna_usa_a_odd_da_casa(self):
        leg = ph.model_side_leg(self._row(odds_a_tp=1.33, odds_b_tp=3.70))
        assert leg["odd"] == pytest.approx(3.70)
        assert leg["ev"] == pytest.approx(0.65 * 3.70)
        assert leg["casa"] is True

    def test_sem_odd_da_casa_cai_na_mediana_e_diz_que_caiu(self):
        leg = ph.model_side_leg(self._row())
        assert leg["odd"] == pytest.approx(3.20)
        assert leg["casa"] is False

    def test_meia_odd_nao_vale(self):
        """Um lado da casa e outro da mediana nao e preco de lugar nenhum."""
        assert ph.model_side_leg(self._row(odds_a_tp=1.33))["casa"] is False

    def test_a_casa_pode_TIRAR_a_perna_do_criterio(self):
        """Odd pior na casa derruba o EV abaixo de 1: a perna some da lista,
        e e o comportamento certo -- aquele preco nao existe para nos."""
        assert self._row(odds_a_tp=1.33, odds_b_tp=3.70).pipe(ph.model_side_leg)["ev"] > 1
        assert ph.model_side_leg(self._row(odds_a_tp=1.80, odds_b_tp=1.50))["ev"] < 1

    def test_pnl_da_serie_paga_a_odd_da_casa(self, tmp_path):
        path = tmp_path / "hist.csv"
        pd.DataFrame([dict(
            event_name="UFC A", event_date="2026-10-24", fighter_a="Carla",
            fighter_b="Dora", odds_a_decimal=1.40, odds_b_decimal=3.20,
            odds_a_tp=1.33, odds_b_tp=3.70, model_name="logreg",
            model_prob_a=0.35, model_side="Dora", actual_winner="Dora"),
        ]).to_csv(path, index=False)
        s = ph.compute_series_summary(ph.load_history(path))
        assert s["legs_n"] == 1 and s["legs_won"] == 1
        assert s["legs_pnl"] == pytest.approx(2.70)   # 3.70 - 1, nao 3.20 - 1

    def test_historico_antigo_sem_as_colunas_continua_carregando(self, tmp_path):
        """Os 14 primeiros eventos foram registrados e medidos na mediana.
        Nada ali e reescrito: coluna que nao existe nasce vazia."""
        path = tmp_path / "hist.csv"
        pd.DataFrame([dict(
            event_name="UFC Velho", event_date="2026-07-11", fighter_a="Carla",
            fighter_b="Dora", odds_a_decimal=1.40, odds_b_decimal=3.20,
            model_name="logreg", model_prob_a=0.35, model_side="Dora",
            actual_winner="Dora"),
        ]).to_csv(path, index=False)
        df = ph.load_history(path)
        assert df["odds_a_tp"].isna().all()
        assert ph.compute_series_summary(df)["legs_pnl"] == pytest.approx(2.20)


class TestOddDaCasaNaoSePerde:
    """
    Mesmo modo de falha do sinal sharp, descoberto na varredura de 17/out:
    `record` reescreve a linha aberta inteira, entao regerar o relatorio a
    partir de um CSV sem as colunas da casa apagava a odd em silencio -- e a
    perna voltava a ser paga na MEDIANA, um preco que nao existe para nos.
    """
    AMANHA = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    def _fight(self, a, b, prob_a, tp=None):
        f = {"fighter_a": a, "fighter_b": b, "odds_a": 1.80, "odds_b": 2.10,
             "model_prob_a": prob_a, "model_side": a if prob_a >= 0.5 else b,
             "category": "favorite", "method_probs": None,
             "odds_a_casa": 1.80, "odds_b_casa": 2.10, "odds_da_casa": False}
        if tp:
            f.update(odds_a_casa=tp[0], odds_b_casa=tp[1], odds_da_casa=True)
        return f

    def _grava(self, path, fight):
        ph.record_card_predictions(
            {"favorites": [fight], "underdogs": [], "no_prediction": [],
             "model_name": "logreg"}, "UFC X", self.AMANHA, path)
        return ph._load_raw(path).iloc[0]

    def test_regerar_sem_as_colunas_preserva_a_odd(self, tmp_path):
        p = tmp_path / "h.csv"
        self._grava(p, self._fight("Alice", "Bruna", 0.55, (1.95, 2.30)))
        r = self._grava(p, self._fight("Alice", "Bruna", 0.55))
        assert (r["odds_a_tp"], r["odds_b_tp"]) == (1.95, 2.30)
        assert ph.model_side_leg(r)["odd"] == pytest.approx(1.95)

    def test_odd_nova_SUBSTITUI_a_guardada(self, tmp_path):
        """Preservar nao pode virar congelar: a linha se move na semana, e
        regerar antes do evento tem de trazer o preco novo."""
        p = tmp_path / "h.csv"
        self._grava(p, self._fight("Alice", "Bruna", 0.55, (1.95, 2.30)))
        r = self._grava(p, self._fight("Alice", "Bruna", 0.55, (1.80, 2.55)))
        assert (r["odds_a_tp"], r["odds_b_tp"]) == (1.80, 2.55)

    def test_lados_trocados_no_CSV_reorientam_a_odd(self, tmp_path):
        """_same_fight casa a luta nas DUAS ordens, e estas colunas seguem
        fighter_a/fighter_b (nao o model_side). Copiar direto inverteria o
        preco de cada lado -- o erro classico do projeto."""
        p = tmp_path / "h.csv"
        self._grava(p, self._fight("Alice", "Bruna", 0.55, (1.95, 2.30)))
        r = self._grava(p, self._fight("Bruna", "Alice", 0.45))
        assert r["fighter_a"] == "Bruna"
        assert (r["odds_a_tp"], r["odds_b_tp"]) == (2.30, 1.95)

    def test_o_painel_diz_quando_o_PeL_soma_duas_escalas(self, tmp_path):
        p = tmp_path / "h.csv"
        pd.DataFrame([
            dict(event_name="UFC Velho", event_date="2026-07-11", fighter_a="A",
                 fighter_b="B", odds_a_decimal=3.20, odds_b_decimal=1.40,
                 model_name="logreg", model_prob_a=0.65, model_side="A",
                 actual_winner="A"),
            dict(event_name="UFC Novo", event_date="2026-10-24", fighter_a="C",
                 fighter_b="D", odds_a_decimal=3.20, odds_b_decimal=1.40,
                 odds_a_tp=3.70, odds_b_tp=1.33, model_name="logreg",
                 model_prob_a=0.65, model_side="C", actual_winner="C"),
        ]).to_csv(p, index=False)
        df = ph.load_history(p)
        assert ph.compute_series_summary(df)["legs_casa"] == 1
        assert "1 na odd da casa, 1 na mediana" in ph.render_history_panel(df)
