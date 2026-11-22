"""
CHAOS Fase B (D3, D4, D7), pelos caminhos que gravam e desenham de verdade.

  D3  falha do modelo nao apaga previsao aberta existente;
  D4  o TITULO do evento nao e identidade da luta (data + par + ids estaveis);
  D7  somente_render usa preco e probabilidade da MESMA versao.

Nada toca producao: caminhos de `config` redirecionados para tmp_path.
"""
import pandas as pd
import pytest

import config
from src import card_report, prediction_history as ph, proveniencia, snapshots
from src.card_report import analyze_card, generate_card_report

LUTAS = [("Alice Silva", "Bia Costa"), ("Carla Rocha", "Dora Lima")]
P1 = {LUTAS[0]: 0.64, LUTAS[1]: 0.48}
P2 = {LUTAS[0]: 0.71, LUTAS[1]: 0.40}


def _dia(delta):
    return (pd.Timestamp.today() + pd.Timedelta(days=delta)).strftime("%Y-%m-%d")


DATA = _dia(3)


def _card(tp=((1.90, 2.00), (1.98, 1.90)), mediana=((1.60, 2.40), (1.90, 1.95)),
          lutas=LUTAS):
    linhas = []
    for i, (a, b) in enumerate(lutas):
        l = {"fighter_a": a, "fighter_b": b, "odds_a_decimal": mediana[i][0],
             "odds_b_decimal": mediana[i][1], "fonte": "ambas"}
        if tp:
            l.update({"odds_a_tp": tp[i][0], "odds_b_tp": tp[i][1],
                      "ref_tp": f"RA{i}", "ref_tp_b": f"RB{i}"})
        linhas.append(l)
    return pd.DataFrame(linhas)


def _preditor(tabela):
    def predict(a, b):
        if (a, b) in tabela:
            p = tabela[(a, b)]
        elif (b, a) in tabela:
            p = 1 - tabela[(b, a)]
        else:
            raise ValueError("sem fixture")
        return {"fighter_a": a, "fighter_b": b, "prob_a_wins": p,
                "prob_b_wins": 1 - p, "model_used": "fake",
                "fighter_a_low_experience": False, "fighter_b_low_experience": False,
                "fighter_a_debutant": False, "fighter_b_debutant": False}

    def method(a, b):
        return {"method_probs": {"KO_TKO": 0.4, "SUBMISSION": 0.2, "DECISION": 0.4}}
    return predict, method


def _quebrado():
    def boom(a, b):
        raise FileNotFoundError("Modelo nao encontrado em models/x.joblib")
    return boom, boom


class Mundo:
    def __init__(self, tmp, monkeypatch):
        self.tmp, self.mp = tmp, monkeypatch
        self.hist = tmp / "prediction_history.csv"
        self.card = tmp / "card.csv"
        self.template = tmp / "odds_template.csv"
        self.saida = tmp / "index.html"
        self.analises = []
        orig = card_report.render_html

        def espiao(analysis, *a, **k):
            self.analises.append(analysis)
            return orig(analysis, *a, **k)
        monkeypatch.setattr(card_report, "render_html", espiao)

    @property
    def snaps(self):
        return snapshots.arquivo_de(self.hist)

    def n_versoes(self):
        return len(snapshots.carregar(self.snaps)) if self.snaps.exists() else 0

    def snaps_bytes(self):
        return self.snaps.read_bytes() if self.snaps.exists() else b""

    def publicar(self, card_df, preditor, titulo="UFC X", data=DATA,
                 sharp_id=None, somente_render=False):
        card_df.to_csv(self.card, index=False)
        self.mp.setattr(card_report, "_default_predict_fns", lambda n: preditor)
        if sharp_id is not None:
            from src import line_shopping

            def fake(fights, **kw):
                return {(f["fighter_a"], f["fighter_b"]):
                        {"sharp_prob": 0.5, "best_odd": 2.0, "casa": "pinnacle",
                         "ref": None, "evento_id": f"{sharp_id}-{f['fighter_a']}",
                         "evidencia": "exato+exato",
                         "participantes": f"{f['fighter_a']}|{f['fighter_b']}"}
                        for f in fights}
            self.mp.setattr(line_shopping, "fetch_sharp_probs", fake)
        generate_card_report(self.card, self.saida, model_name="fake",
                             card_name=titulo, event_date=data, photos=False,
                             sharp=sharp_id is not None,
                             somente_render=somente_render)
        return self.analises[-1]

    def linhas(self, data=DATA):
        h = pd.read_csv(self.hist)
        return h[h["event_date"] == data]

    def liquidar(self, vencedores, data=DATA):
        pd.DataFrame([{"fighter_a": a, "fighter_b": b, "event_date": data,
                       "odds_a_decimal": 1.5, "odds_b_decimal": 2.5,
                       "actual_winner": w}
                      for (a, b), w in vencedores.items()]
                     ).to_csv(self.template, index=False)
        return ph.sync_results_from_template()


@pytest.fixture
def mundo(tmp_path, monkeypatch):
    from src import data_collection, event_watch
    m = Mundo(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "PREDICTION_HISTORY_CSV", m.hist)
    monkeypatch.setattr(config, "ODDS_TEMPLATE_CSV", m.template)
    monkeypatch.setattr(config, "UPCOMING_CARD_CSV", m.card)
    monkeypatch.setattr(config, "RAW_FIGHTS_CSV", tmp_path / "fights.csv")
    monkeypatch.setattr(event_watch, "CACHE_PENDENTES", tmp_path / "pendentes.json")
    pasta = tmp_path / "proveniencia"
    monkeypatch.setattr(proveniencia, "PASTA", pasta)
    monkeypatch.setattr(proveniencia, "INDICE", pasta / "chamadas.csv")
    monkeypatch.setattr(proveniencia, "INDICE_COTACOES", pasta / "cotacoes.csv")
    monkeypatch.setattr(card_report, "set_photo_map", lambda mp: None)
    monkeypatch.setattr(data_collection, "check_data_freshness", lambda: 0)
    return m


# ---------------------------------------------------------------- D3

class TestD3ModeloQuebradoNaoApagaPrevisao:
    def test_republicar_sem_modelo_preserva_linha_ponteiro_e_versoes(self, mundo):
        mundo.publicar(_card(), _preditor(P1), sharp_id="E1")
        ph.record_closing_lines(DATA, {
            (a, b): {"sharp_prob": 0.55, "best_odd": 1.9} for a, b in LUTAS},
            allow_update=True)
        antes = mundo.linhas()
        versoes_antes, bytes_antes = mundo.n_versoes(), mundo.snaps_bytes()
        assert antes["close_prob"].notna().all() and antes["sharp_prob"].notna().all()

        a = mundo.publicar(_card(), _quebrado(), titulo="UFC X")

        depois = mundo.linhas()
        pd.testing.assert_frame_equal(depois.reset_index(drop=True),
                                      antes.reset_index(drop=True))
        assert mundo.n_versoes() == versoes_antes
        assert mundo.snaps_bytes() == bytes_antes
        # a pagina e desenhada da versao em vigor, marcada como snapshot
        previstas = {f["fighter_a"]: f for f in a["favorites"] + a["underdogs"]}
        assert a["no_prediction"] == []
        assert set(previstas) == {x for x, _ in LUTAS}
        for f in previstas.values():
            assert f["origem_previsao"] == "snapshot"
        assert previstas["Alice Silva"]["model_prob_a"] == pytest.approx(0.64)
        # com o PRECO da mesma versao
        assert previstas["Alice Silva"]["odds_a_casa"] == 1.90

    def _publicar_e_fechar_linha(self, mundo):
        mundo.publicar(_card(), _preditor(P1), sharp_id="E1")
        ph.record_closing_lines(DATA, {
            (a, b): {"sharp_prob": 0.55, "best_odd": 1.9} for a, b in LUTAS},
            allow_update=True)
        return mundo.linhas().reset_index(drop=True)

    def _assert_intacto(self, mundo, antes, bytes_antes):
        depois = mundo.linhas().reset_index(drop=True)
        for col in ("model_prob_a", "model_side", "sharp_prob", "close_prob",
                    "prediction_id"):
            pd.testing.assert_series_equal(depois[col], antes[col])
        assert mundo.snaps_bytes() == bytes_antes

    def test_guarda_do_historico_ponteiro_sem_versao_e_modelo_quebrado(self, mundo):
        """Exercita a guarda de record_card_predictions (a luta chega SEM
        origem snapshot, como no_prediction): sem ela a linha com previsao
        viraria linha vazia com versao vazia nova."""
        antes = self._publicar_e_fechar_linha(mundo)
        h = pd.read_csv(mundo.hist)
        h["prediction_id"] = ""  # ponteiro que nao resolve: sem versao em vigor
        h.to_csv(mundo.hist, index=False)
        antes = mundo.linhas().reset_index(drop=True)
        bytes_antes = mundo.snaps_bytes()
        a = mundo.publicar(_card(), _quebrado())
        assert len(a["no_prediction"]) == 2
        self._assert_intacto(mundo, antes, bytes_antes)
        assert mundo.linhas()["model_prob_a"].notna().all()

    def test_guarda_do_historico_dois_estreantes_nao_apaga_previsao(self, mundo):
        antes = self._publicar_e_fechar_linha(mundo)
        bytes_antes = mundo.snaps_bytes()
        base, metodo = _preditor(P2)

        def estreantes(a, b):
            return {**base(a, b), "fighter_a_debutant": True,
                    "fighter_b_debutant": True}
        a = mundo.publicar(_card(), (estreantes, metodo))
        assert len(a["no_prediction"]) == 2
        assert all("estreando" in f["reason"] for f in a["no_prediction"])
        self._assert_intacto(mundo, antes, bytes_antes)

    def test_sem_modelo_e_sem_versao_fica_sem_previsao_com_motivo(self, mundo):
        a = mundo.publicar(_card(), _quebrado())
        assert a["favorites"] == [] and a["underdogs"] == []
        assert len(a["no_prediction"]) == 2
        assert all("Modelo nao encontrado" in f["reason"] for f in a["no_prediction"])
        # linha NOVA sem previsao entra como antes
        h = mundo.linhas()
        assert len(h) == 2 and h["model_prob_a"].isna().all()

    def test_luta_nova_sem_previsao_entra_e_a_existente_e_mantida(self, mundo):
        mundo.publicar(_card(lutas=[LUTAS[0]], tp=((1.9, 2.0),),
                             mediana=((1.6, 2.4),)), _preditor(P1))
        nova = ("Eva Lopes", "Fay Cruz")
        card = _card(lutas=[LUTAS[0], nova], tp=((1.9, 2.0), (1.8, 2.1)),
                     mediana=((1.6, 2.4), (1.7, 2.2)))
        mundo.publicar(card, _quebrado())
        h = mundo.linhas().set_index("fighter_a")
        assert h.loc["Alice Silva", "model_prob_a"] == pytest.approx(0.64)
        assert pd.isna(h.loc["Eva Lopes", "model_prob_a"])

    def test_caso_normal_com_modelo_e_odds_novas_gera_versao_nova(self, mundo):
        mundo.publicar(_card(), _preditor(P1))
        id_antigo = mundo.linhas().set_index("fighter_a").loc["Alice Silva", "prediction_id"]
        v0 = mundo.n_versoes()
        a = mundo.publicar(_card(tp=((2.30, 1.70), (1.98, 1.90))), _preditor(P2))
        linha = mundo.linhas().set_index("fighter_a").loc["Alice Silva"]
        assert mundo.n_versoes() == v0 + 2
        assert linha["prediction_id"] != id_antigo
        assert linha["model_prob_a"] == pytest.approx(0.71)
        assert linha["odds_a_tp"] == 2.30
        assert snapshots.versao(linha["prediction_id"], mundo.snaps)["substitui"] == id_antigo
        f = {f["fighter_a"]: f for f in a["favorites"] + a["underdogs"]}["Alice Silva"]
        assert f["origem_previsao"] == "modelo"
        assert f["odds_a_casa"] == 2.30  # fora de somente_render o CSV manda


# ---------------------------------------------------------------- D4

class TestD4TituloNaoEIdentidade:
    def test_renomear_o_card_mantem_uma_linha_por_luta_e_encadeia(self, mundo):
        mundo.publicar(_card(), _preditor(P1), titulo="UFC X: Alfa vs Beta")
        antes = mundo.linhas().set_index("fighter_a")
        mundo.publicar(_card(), _preditor(P2), titulo="UFC X: Alfa vs Epsilon")
        h = mundo.linhas()
        assert len(h) == 2
        assert set(h["event_name"]) == {"UFC X: Alfa vs Epsilon"}
        h = h.set_index("fighter_a")
        for nome in ("Alice Silva", "Carla Rocha"):
            v = snapshots.versao(h.loc[nome, "prediction_id"], mundo.snaps)
            assert v["substitui"] == antes.loc[nome, "prediction_id"]
        assert h.loc["Alice Silva", "model_prob_a"] == pytest.approx(0.71)

    def test_placar_conta_cada_luta_uma_vez_depois_de_renomear(self, mundo):
        mundo.publicar(_card(), _preditor(P1), titulo="UFC X: Alfa vs Beta")
        mundo.publicar(_card(), _preditor(P1), titulo="UFC X: Alfa vs Epsilon")
        assert mundo.liquidar({LUTAS[0]: "Alice Silva", LUTAS[1]: "Dora Lima"}) == 2
        resumo = ph.compute_series_summary(ph.load_history())
        assert resumo["model_n"] == 2 and resumo["n_events"] == 1

    def test_id_estavel_divergente_nao_casa(self, mundo):
        mundo.publicar(_card(), _preditor(P1), titulo="UFC X", sharp_id="A")
        mundo.publicar(_card(), _preditor(P1), titulo="UFC Y", sharp_id="B")
        h = mundo.linhas()
        assert len(h) == 4  # outra luta (outro id): nao casou, linhas proprias
        assert sorted(h["event_name"]) == ["UFC X", "UFC X", "UFC Y", "UFC Y"]

    def test_mesmo_id_estavel_casa_mesmo_com_titulo_novo(self, mundo):
        mundo.publicar(_card(), _preditor(P1), titulo="UFC X", sharp_id="A")
        mundo.publicar(_card(), _preditor(P2), titulo="UFC Y", sharp_id="A")
        h = mundo.linhas()
        assert len(h) == 2 and set(h["event_name"]) == {"UFC Y"}

    def test_dois_candidatos_compativeis_levantam_erro_sem_gravar(self, mundo):
        mundo.publicar(_card(), _preditor(P1), titulo="UFC X", sharp_id="A")
        mundo.publicar(_card(), _preditor(P1), titulo="UFC Y", sharp_id="B")
        assert len(mundo.linhas()) == 4
        hist_antes, snaps_antes = mundo.hist.read_bytes(), mundo.snaps_bytes()
        # sem sharp: nenhum id na linha nova, os dois candidatos servem
        with pytest.raises(ph.LutaAmbigua, match="poderiam ser esta luta"):
            mundo.publicar(_card(), _preditor(P2), titulo="UFC Z")
        assert mundo.hist.read_bytes() == hist_antes
        assert mundo.snaps_bytes() == snaps_antes

    def test_ambiguidade_na_segunda_luta_nao_grava_versao_da_primeira(self, mundo):
        # duplica so a SEGUNDA luta, escrevendo a linha duplicada a mao
        mundo.publicar(_card(), _preditor(P1), titulo="UFC X")
        h = pd.read_csv(mundo.hist)
        dup = h[h["fighter_a"] == "Carla Rocha"].copy()
        dup["event_name"] = "UFC W"
        pd.concat([h, dup]).to_csv(mundo.hist, index=False)
        hist_antes, snaps_antes = mundo.hist.read_bytes(), mundo.snaps_bytes()
        with pytest.raises(ph.LutaAmbigua):
            mundo.publicar(_card(), _preditor(P2), titulo="UFC Z")
        assert mundo.hist.read_bytes() == hist_antes
        assert mundo.snaps_bytes() == snaps_antes

    def test_revanche_em_outra_data_e_linha_propria(self, mundo):
        mundo.publicar(_card(), _preditor(P1), titulo="UFC X", data=DATA)
        mundo.publicar(_card(), _preditor(P2), titulo="UFC X", data=_dia(30))
        h = pd.read_csv(mundo.hist)
        assert len(h) == 4
        assert h.groupby("event_date").size().to_dict() == {DATA: 2, _dia(30): 2}

    def test_linha_fechada_continua_intocavel_apos_renomear(self, mundo):
        mundo.publicar(_card(), _preditor(P1), titulo="UFC X")
        mundo.liquidar({LUTAS[0]: "Alice Silva", LUTAS[1]: "Dora Lima"})
        fechada = mundo.linhas().reset_index(drop=True)
        mundo.publicar(_card(), _preditor(P2), titulo="UFC Y")
        pd.testing.assert_frame_equal(mundo.linhas().reset_index(drop=True), fechada)


# ---------------------------------------------------------------- D7

class TestD7SomenteRenderUsaPrecoDaVersao:
    def test_odd_da_casa_e_ev_da_versao_e_nao_do_csv(self, mundo):
        mundo.publicar(_card(), _preditor(P1))
        novo = _card(tp=((2.60, 1.55), (1.98, 1.90)), mediana=((2.40, 1.60), (1.90, 1.95)))
        a = mundo.publicar(novo, _quebrado(), somente_render=True)
        f = {f["fighter_a"]: f for f in a["favorites"] + a["underdogs"]}["Alice Silva"]
        assert f["origem_previsao"] == "snapshot"
        assert (f["odds_a_casa"], f["odds_b_casa"]) == (1.90, 2.00)
        assert (f["odds_a"], f["odds_b"]) == (1.60, 2.40)
        assert f["ev"] == pytest.approx(0.64 * 1.90)
        assert f["ref_tp"] == "RA0" and f["ref_tp_b"] == "RB0"
        ev_legs = {x["fighter_a"] for x in a["ev_legs"]}
        assert "Alice Silva" in ev_legs  # 0.64 x 1.90 = 1.216 > 1
        html = mundo.saida.read_text(encoding="utf-8")
        assert "1.90" in html and "2.60" not in html

    def test_perna_ev_nao_nasce_do_csv(self, mundo):
        mundo.publicar(_card(tp=((1.40, 3.00), (1.98, 1.90))), _preditor(P1))
        # Alice: 0.64 x 1.40 = 0.896 (nao e EV>1); o CSV passa a 2.60
        a = mundo.publicar(_card(tp=((2.60, 3.00), (1.98, 1.90))), _quebrado(),
                           somente_render=True)
        assert "Alice Silva" not in {x["fighter_a"] for x in a["ev_legs"]}

    def test_lados_invertidos_no_csv_reorientam_preco(self, mundo):
        mundo.publicar(_card(), _preditor(P1))
        invertido = _card(lutas=[(LUTAS[0][1], LUTAS[0][0]), LUTAS[1]],
                          tp=((9.0, 9.5), (1.98, 1.90)),
                          mediana=((9.0, 9.5), (1.90, 1.95)))
        a = mundo.publicar(invertido, _quebrado(), somente_render=True)
        f = {f["fighter_a"]: f for f in a["favorites"] + a["underdogs"]}["Bia Costa"]
        assert f["model_prob_a"] == pytest.approx(0.36)
        assert (f["odds_a_casa"], f["odds_b_casa"]) == (2.00, 1.90)
        assert (f["odds_a"], f["odds_b"]) == (2.40, 1.60)
        assert (f["ref_tp"], f["ref_tp_b"]) == ("RB0", "RA0")

    def test_versao_sem_odd_da_casa_usa_mediana_da_versao_marcada(self, mundo):
        mundo.publicar(_card(tp=None), _preditor(P1))
        novo = _card(tp=((2.60, 1.55), (1.98, 1.90)), mediana=((2.40, 1.60), (1.90, 1.95)))
        a = mundo.publicar(novo, _quebrado(), somente_render=True)
        f = {f["fighter_a"]: f for f in a["favorites"] + a["underdogs"]}["Alice Silva"]
        assert f["odds_da_casa"] is False
        assert (f["odds_a_casa"], f["odds_b_casa"]) == (1.60, 2.40)
        assert f["ev"] == pytest.approx(0.64 * 1.60)

    def test_versao_sem_mediana_fica_indisponivel_e_nunca_usa_o_csv(self):
        card = _card()
        frozen = {LUTAS[0]: {"model_prob_a": 0.64, "method_probs": None,
                             "fechada": False, "motivo": None,
                             "odds_a_decimal": None, "odds_b_decimal": None,
                             "odds_a_tp": None, "odds_b_tp": None,
                             "ref_tp": "", "ref_tp_b": ""}}
        a = analyze_card(card.iloc[[0]], frozen=frozen, somente_snapshot=True)
        assert a["favorites"] == [] and a["underdogs"] == []
        (f,) = a["no_prediction"]
        assert "nao tem odds de mercado" in f["reason"]
        assert f["odds_a"] is None and f["odds_da_casa"] is False

    def test_luta_encerrada_usa_o_preco_registrado_na_linha(self, mundo):
        mundo.publicar(_card(), _preditor(P1))
        mundo.liquidar({LUTAS[0]: "Alice Silva", LUTAS[1]: "Dora Lima"})
        novo = _card(tp=((2.60, 1.55), (1.98, 1.90)), mediana=((2.40, 1.60), (1.90, 1.95)))
        a = mundo.publicar(novo, _quebrado(), somente_render=True)
        f = {f["fighter_a"]: f for f in a["favorites"] + a["underdogs"]}["Alice Silva"]
        assert f["frozen"] is True
        assert (f["odds_a_casa"], f["odds_a"]) == (1.90, 1.60)

    def test_o_csv_continua_definindo_quais_lutas_e_a_ordem(self, mundo):
        mundo.publicar(_card(), _preditor(P1))
        invertido = _card(lutas=[LUTAS[1], LUTAS[0]],
                          tp=((1.98, 1.90), (1.90, 2.00)),
                          mediana=((1.90, 1.95), (1.60, 2.40)))
        a = mundo.publicar(invertido, _quebrado(), somente_render=True)
        ordem = sorted(a["favorites"] + a["underdogs"], key=lambda f: f["card_order"])
        assert [f["fighter_a"] for f in ordem] == ["Carla Rocha", "Alice Silva"]
