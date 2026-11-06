"""
Versões imutáveis da previsão (`src/snapshots.py`).

REGRESSÃO. `prediction_history.csv` guardava UMA linha por luta e a reescrevia
a cada regeração enquanto o evento não acontecia. A regra estava documentada e
tinha motivo, mas o efeito é que a versão anterior **desaparecia**: não havia
como dizer o que a página mostrava na véspera, nem contra qual versão uma
decisão foi tomada. Medido em 06/11/2026, regerar o card de 07/11 moveu seis
odds da mediana, seis da casa e uma probabilidade — nada errado, e nada
recuperável.

O que estes testes fixam:

  * regerar cria uma versão NOVA e a anterior continua endereçável;
  * cada versão carrega instante, manifesto, modelos e provas de preço;
  * a decisão aponta para a versão EXATA que usou, não para "a mais nova";
  * previsão criada depois do início **daquela luta** não sustenta seleção;
  * a semeadura das linhas antigas documenta que a temporalidade delas não é
    recuperável, em vez de inventar uma data.
"""
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import config
from src import decisoes, identidade, prediction_history as ph, proveniencia
from src import snapshots


def _agora():
    return datetime.now(timezone.utc)


def _iso(delta_horas: float) -> str:
    return (_agora() + timedelta(hours=delta_horas)).isoformat(timespec="seconds")


@pytest.fixture
def mundo(tmp_path, monkeypatch):
    pasta = tmp_path / "proveniencia"
    monkeypatch.setattr(proveniencia, "PASTA", pasta)
    monkeypatch.setattr(proveniencia, "INDICE", pasta / "chamadas.csv")
    monkeypatch.setattr(proveniencia, "INDICE_COTACOES", pasta / "cotacoes.csv")
    hist = tmp_path / "prediction_history.csv"
    monkeypatch.setattr(config, "PREDICTION_HISTORY_CSV", hist)
    monkeypatch.setattr(snapshots, "ARQUIVO", tmp_path / "snapshots.csv")
    monkeypatch.setattr(decisoes, "ARQUIVO", tmp_path / "decisoes.csv")
    snapshots._CACHE.clear()
    return {"hist": hist, "snap": tmp_path / "snapshots.csv",
            "log": tmp_path / "decisoes.csv", "card": tmp_path / "card.csv"}


def _provas(*, inicio, a="Alice Silva", b="Bia Costa", odd_a=1.80, odd_b=2.10):
    corpo = json.dumps([{"home": a, "away": b, "odd_home": odd_a,
                         "odd_away": odd_b, "inicio": inicio, "status": 1}],
                       ensure_ascii=False, sort_keys=True)
    chamada = proveniencia.registrar_chamada(
        corpo, fonte="thunderpick", endpoint="https://thunderpick.io/x",
        markets="h2h")
    comum = dict(chamada_ref=chamada, fonte="thunderpick", casa="Thunderpick",
                 mercado="h2h", evento_id=inicio, participante_a=a,
                 participante_b=b, status="aberto", inicio_programado=inicio)
    return (proveniencia.registrar_cotacao(lado=a, odd=odd_a, **comum),
            proveniencia.registrar_cotacao(lado=b, odd=odd_b, **comum))


def _analise(prob=0.64, *, a="Alice Silva", b="Bia Costa", ref_a="", ref_b="",
             odd_a=1.80, odd_b=2.10):
    return {
        "model_name": "logreg", "no_prediction": [], "underdogs": [],
        "favorites": [{
            "fighter_a": a, "fighter_b": b,
            "odds_a": 1.60, "odds_b": 2.40,
            "odds_da_casa": True, "odds_a_casa": odd_a, "odds_b_casa": odd_b,
            "model_prob_a": prob, "model_side": a if prob >= 0.5 else b,
            "method_probs": {"KO_TKO": 0.4, "SUBMISSION": 0.2,
                             "DECISION": 0.4},
            "fonte_card": "ambas", "ref_tp": ref_a, "ref_tp_b": ref_b,
        }],
    }


def _gravar(mundo, analise, data):
    return ph.record_card_predictions(analise, "UFC Teste", data,
                                      history_csv=mundo["hist"])


class TestRegerarCriaVersaoEmVezDeApagar:
    def test_duas_regeracoes_sao_duas_versoes(self, mundo):
        data = _agora().strftime("%Y-%m-%d")
        ra, rb = _provas(inicio=_iso(5))
        _gravar(mundo, _analise(0.6400, ref_a=ra, ref_b=rb), data)
        _gravar(mundo, _analise(0.6800, ref_a=ra, ref_b=rb), data)

        versoes = snapshots.versoes_da_luta(data, "Alice Silva", "Bia Costa",
                                            mundo["snap"])
        assert len(versoes) == 2
        assert [float(p) for p in versoes["model_prob_a"]] == [0.64, 0.68]
        # a segunda declara qual versao ela sucede; a primeira nao sucede nada
        assert versoes.iloc[0]["substitui"] == ""
        assert versoes.iloc[1]["substitui"] == versoes.iloc[0]["prediction_id"]

    def test_a_versao_anterior_continua_enderecavel(self, mundo):
        """O ponto inteiro: depois de regerar, o que a pagina mostrava antes
        ainda pode ser lido."""
        data = _agora().strftime("%Y-%m-%d")
        ra, rb = _provas(inicio=_iso(5))
        _gravar(mundo, _analise(0.6400, ref_a=ra, ref_b=rb), data)
        primeira = snapshots.carregar(mundo["snap"]).iloc[0]["prediction_id"]
        _gravar(mundo, _analise(0.6800, ref_a=ra, ref_b=rb), data)

        v = snapshots.versao(primeira, mundo["snap"])
        assert v is not None
        assert float(v["model_prob_a"]) == pytest.approx(0.64)

    def test_o_historico_aponta_para_a_versao_em_vigor(self, mundo):
        data = _agora().strftime("%Y-%m-%d")
        ra, rb = _provas(inicio=_iso(5))
        _gravar(mundo, _analise(0.6400, ref_a=ra, ref_b=rb), data)
        _gravar(mundo, _analise(0.6800, ref_a=ra, ref_b=rb), data)
        hist = pd.read_csv(mundo["hist"])
        atual = snapshots.carregar(mundo["snap"]).iloc[-1]
        assert hist.iloc[0]["prediction_id"] == atual["prediction_id"]
        assert float(atual["model_prob_a"]) == pytest.approx(0.68)

    def test_cada_versao_leva_instante_e_artefatos(self, mundo):
        data = _agora().strftime("%Y-%m-%d")
        ra, rb = _provas(inicio=_iso(5))
        _gravar(mundo, _analise(ref_a=ra, ref_b=rb), data)
        v = snapshots.carregar(mundo["snap"]).iloc[0]
        assert v["criado_em"].endswith("+00:00")
        assert v["inicio_programado"]
        assert v["ref_tp"] == ra and v["ref_tp_b"] == rb
        # manifesto/modelos ficam VAZIOS num ambiente sem treino -- faltante,
        # nunca inventado
        assert "manifesto_sha256" in v.index

    def test_ids_distintos_por_versao(self, mundo):
        data = _agora().strftime("%Y-%m-%d")
        ra, rb = _provas(inicio=_iso(5))
        _gravar(mundo, _analise(0.64, ref_a=ra, ref_b=rb), data)
        _gravar(mundo, _analise(0.64, ref_a=ra, ref_b=rb), data)
        ids = list(snapshots.carregar(mundo["snap"])["prediction_id"])
        assert len(set(ids)) == 2, "versoes iguais em instantes diferentes SAO duas"


class TestPrevisaoDepoisDoInicio:
    def test_versao_criada_apos_o_inicio_e_marcada(self, mundo):
        data = _agora().strftime("%Y-%m-%d")
        ra, rb = _provas(inicio=_iso(-2))   # a luta comecou ha duas horas
        _gravar(mundo, _analise(ref_a=ra, ref_b=rb), data)
        v = snapshots.carregar(mundo["snap"]).iloc[0]
        assert v["pos_inicio"] == "True"

    def test_o_gate_bloqueia_selecao_apoiada_nela(self, mundo):
        data = _agora().strftime("%Y-%m-%d")
        ra, rb = _provas(inicio=_iso(-2))
        _gravar(mundo, _analise(ref_a=ra, ref_b=rb), data)
        linha = pd.read_csv(mundo["hist"]).iloc[0]
        assert (identidade.motivo_de_bloqueio(linha)
                == identidade.BLOQUEIO_POS_INICIO)

    def test_a_conta_e_POR_LUTA_e_nao_por_card(self, mundo):
        """Num card as lutas comecam ao longo de cinco horas. Usar o horario
        do evento liberaria uma previsao feita depois do primeiro combate."""
        data = _agora().strftime("%Y-%m-%d")
        ja, ja_b = _provas(inicio=_iso(-2), a="Alice Silva", b="Bia Costa")
        nao, nao_b = _provas(inicio=_iso(3), a="Carla Rocha", b="Dora Lima")
        _gravar(mundo, _analise(ref_a=ja, ref_b=ja_b), data)
        _gravar(mundo, _analise(a="Carla Rocha", b="Dora Lima",
                                ref_a=nao, ref_b=nao_b), data)
        por_luta = {v["fighter_a"]: v["pos_inicio"]
                    for _, v in snapshots.carregar(mundo["snap"]).iterrows()}
        assert por_luta == {"Alice Silva": "True", "Carla Rocha": "False"}

    def test_inicio_desconhecido_nao_vira_ainda_nao_comecou(self, mundo):
        """Terceira resposta: sem horario, `pos_inicio` fica vazio em vez de
        False. Quem exige atualidade ja bloqueia cotacao sem inicio."""
        data = _agora().strftime("%Y-%m-%d")
        _gravar(mundo, _analise(), data)
        assert snapshots.carregar(mundo["snap"]).iloc[0]["pos_inicio"] == ""
        assert snapshots.ja_comecou("") is None
        assert snapshots.ja_comecou("sei la") is None


class TestDecisaoApontaParaAVersaoExata:
    def test_a_decisao_carrega_o_prediction_id(self, mundo):
        data = _agora().strftime("%Y-%m-%d")
        ra, rb = _provas(inicio=_iso(5))
        _gravar(mundo, _analise(ref_a=ra, ref_b=rb), data)
        pd.DataFrame([{
            "fighter_a": "Alice Silva", "fighter_b": "Bia Costa",
            "odds_a_decimal": 1.60, "odds_b_decimal": 2.40, "fonte": "ambas",
            "odds_a_tp": 1.80, "odds_b_tp": 2.10,
            "ref_tp": ra, "ref_tp_a": ra, "ref_tp_b": rb,
        }]).to_csv(mundo["card"], index=False)
        d = decisoes.avaliar_com_precos_novos(data, mundo["card"],
                                              mundo["hist"])[0]
        atual = snapshots.carregar(mundo["snap"]).iloc[-1]
        assert d["prediction_id"] == atual["prediction_id"]
        assert d["elegivel"], d["bloqueio"]

    def test_decisao_sobre_previsao_pos_inicio_e_bloqueada(self, mundo):
        data = _agora().strftime("%Y-%m-%d")
        ra, rb = _provas(inicio=_iso(-2))
        _gravar(mundo, _analise(ref_a=ra, ref_b=rb), data)
        pd.DataFrame([{
            "fighter_a": "Alice Silva", "fighter_b": "Bia Costa",
            "odds_a_decimal": 1.60, "odds_b_decimal": 2.40, "fonte": "ambas",
            "odds_a_tp": 1.80, "odds_b_tp": 2.10,
            "ref_tp": ra, "ref_tp_a": ra, "ref_tp_b": rb,
        }]).to_csv(mundo["card"], index=False)
        d = decisoes.avaliar_com_precos_novos(data, mundo["card"],
                                              mundo["hist"])[0]
        assert not d["elegivel"]
        assert d["bloqueio"] == identidade.BLOQUEIO_POS_INICIO


class TestSemeaduraDoHistorico:
    def _historico_antigo(self, path):
        pd.DataFrame([{
            "event_name": "UFC Velho", "event_date": "2026-01-10",
            "fighter_a": "Alice Silva", "fighter_b": "Bia Costa",
            "odds_a_decimal": 1.6, "odds_b_decimal": 2.4,
            "model_name": "logreg", "model_prob_a": 0.61,
            "model_side": "Alice Silva", "actual_winner": "Alice Silva",
            "estado_resultado": "vencedor",
        }]).to_csv(path, index=False)

    def test_semeia_e_documenta_a_temporalidade_perdida(self, mundo):
        self._historico_antigo(mundo["hist"])
        assert snapshots.semear_do_historico(mundo["hist"], mundo["snap"]) == 1
        v = snapshots.carregar(mundo["snap"]).iloc[0]
        assert v["criado_em"] == "", "instante inventado e pior que lacuna"
        assert "NAO RECUPERAVEL" in v["temporalidade"]
        assert float(v["model_prob_a"]) == pytest.approx(0.61)

    def test_e_idempotente(self, mundo):
        self._historico_antigo(mundo["hist"])
        snapshots.semear_do_historico(mundo["hist"], mundo["snap"])
        assert snapshots.semear_do_historico(mundo["hist"], mundo["snap"]) == 0
        assert len(snapshots.carregar(mundo["snap"])) == 1

    def test_o_historico_passa_a_apontar_para_a_semeadura(self, mundo):
        self._historico_antigo(mundo["hist"])
        snapshots.semear_do_historico(mundo["hist"], mundo["snap"])
        hist = pd.read_csv(mundo["hist"])
        v = snapshots.carregar(mundo["snap"]).iloc[0]
        assert hist.iloc[0]["prediction_id"] == v["prediction_id"]

    def test_semeadura_nao_afirma_pos_inicio(self, mundo):
        """O instante de criacao e justamente o que nao se recuperou; dizer
        que a versao nasceu antes ou depois do inicio seria invencao."""
        self._historico_antigo(mundo["hist"])
        snapshots.semear_do_historico(mundo["hist"], mundo["snap"])
        assert snapshots.carregar(mundo["snap"]).iloc[0]["pos_inicio"] == ""


class TestArquivoSoRecebeLinhas:
    def test_registrar_nunca_reescreve(self, mundo):
        snapshots.registrar([{"event_date": "2026-01-10", "fighter_a": "A",
                              "fighter_b": "B", "model_prob_a": 0.5}],
                            mundo["snap"])
        antes = mundo["snap"].read_text(encoding="utf-8")
        snapshots.registrar([{"event_date": "2026-01-10", "fighter_a": "A",
                              "fighter_b": "B", "model_prob_a": 0.9}],
                            mundo["snap"])
        depois = mundo["snap"].read_text(encoding="utf-8")
        assert depois.startswith(antes), "linha ja escrita foi tocada"
        assert len(snapshots.carregar(mundo["snap"])) == 2

    def test_para_render_devolve_a_versao_em_vigor(self, mundo):
        data = _agora().strftime("%Y-%m-%d")
        ra, rb = _provas(inicio=_iso(5))
        _gravar(mundo, _analise(0.6400, ref_a=ra, ref_b=rb), data)
        _gravar(mundo, _analise(0.6800, ref_a=ra, ref_b=rb), data)
        mapa = snapshots.para_render(data, mundo["snap"])
        entrada = mapa[("Alice Silva", "Bia Costa")]
        assert entrada["model_prob_a"] == pytest.approx(0.68)
        assert entrada["fechada"] is False
        assert entrada["method_probs"] == {"KO_TKO": 0.4, "SUBMISSION": 0.2,
                                           "DECISION": 0.4}


class TestRegerarNaoDestroiProva:
    """
    REGRESSAO. A prova do casamento (`evento_id_api`, `casamento`,
    `participantes_fonte`) nasce junto do sinal sharp. Uma regeracao com
    `--no-sharp` chegava aqui com as tres colunas vazias e a linha passava a
    ser bloqueada por "casamento nao comprovado" -- um bloqueio sobre a nossa
    escrituracao, nao sobre a luta. Visto em 06/11/2026: regerar o card de
    07/11 apagou a prova de nove linhas.
    """

    def _com_sharp(self, mundo, data, ra, rb, *, a="Alice Silva",
                   b="Bia Costa"):
        return ph.record_card_predictions(
            _analise(ref_a=ra, ref_b=rb, a=a, b=b), "UFC Teste", data,
            history_csv=mundo["hist"],
            sharp_probs={(a, b): {
                "sharp_prob": 0.60, "best_odd": 1.70, "casa": "Pinnacle",
                "ref": "ref-sharp", "evento_id": "ev-1",
                "evidencia": "exato+alias_confirmado",
                "participantes": f"{a}|{b}"}})

    def test_regerar_sem_sharp_preserva_a_prova_do_casamento(self, mundo):
        data = _agora().strftime("%Y-%m-%d")
        ra, rb = _provas(inicio=_iso(5))
        self._com_sharp(mundo, data, ra, rb)
        _gravar(mundo, _analise(ref_a=ra, ref_b=rb), data)   # sem sharp
        linha = pd.read_csv(mundo["hist"]).iloc[0]
        assert linha["evento_id_api"] == "ev-1"
        assert linha["casamento"] == "exato+alias_confirmado"
        assert linha["participantes_fonte"] == "Alice Silva|Bia Costa"
        assert linha["sharp_ref"] == "ref-sharp"

    def test_lados_trocados_invertem_as_metades_do_casamento(self, mundo):
        """`casamento` e por lado. Copiar direto com o CSV invertido colaria
        a evidencia de um lutador no outro."""
        data = _agora().strftime("%Y-%m-%d")
        ra, rb = _provas(inicio=_iso(5))
        self._com_sharp(mundo, data, ra, rb)
        invertida = _analise(prob=0.36, a="Bia Costa", b="Alice Silva",
                             ref_a=rb, ref_b=ra)
        ph.record_card_predictions(invertida, "UFC Teste", data,
                                   history_csv=mundo["hist"])
        linha = pd.read_csv(mundo["hist"]).iloc[0]
        assert linha["fighter_a"] == "Bia Costa"
        assert linha["casamento"] == "alias_confirmado+exato"

    def test_prova_ausente_continua_ausente(self, mundo):
        """Nao ha preenchimento retroativo: linha que nunca teve prova de
        casamento segue sem ela, e bloqueada."""
        data = _agora().strftime("%Y-%m-%d")
        ra, rb = _provas(inicio=_iso(5))
        _gravar(mundo, _analise(ref_a=ra, ref_b=rb), data)
        _gravar(mundo, _analise(ref_a=ra, ref_b=rb), data)
        linha = pd.read_csv(mundo["hist"], dtype=str).fillna("").iloc[0]
        assert linha["casamento"] == ""


class TestRenderEscolheAVersaoVigente:
    """
    REGRESSAO. `para_render` varria o arquivo de versoes e chaveava por
    `(fighter_a, fighter_b)` LITERAL. Duas versoes da mesma luta com os lados
    invertidos viravam duas entradas CONCORRENTES, e quem desenhava pegava a
    que casasse com a ordem do card -- que podia ser a antiga. E "a ultima
    linha do arquivo" nao e a versao em vigor: a versao em vigor e a que o
    HISTORICO aponta.

    O caso do enunciado: versao antiga Alpha x Beta com p(Alpha)=0,70, versao
    vigente Beta x Alpha com p(Beta)=0,20. Desenhar em qualquer ordem tem de
    mostrar Alpha=0,80 e Beta=0,20, com o prediction_id da vigente.
    """
    ANTIGA = {"event_date": "2026-11-07", "event_name": "UFC Teste",
              "fighter_a": "Alpha", "fighter_b": "Beta",
              "model_name": "logreg", "model_prob_a": 0.70,
              "model_side": "Alpha"}
    VIGENTE = {"event_date": "2026-11-07", "event_name": "UFC Teste",
               "fighter_a": "Beta", "fighter_b": "Alpha",
               "model_name": "logreg", "model_prob_a": 0.20,
               "model_side": "Alpha"}

    def _montar(self, mundo, ordem_do_historico=("Alpha", "Beta")):
        antiga = snapshots.registrar([self.ANTIGA], mundo["snap"])[0]
        vigente = snapshots.registrar([self.VIGENTE], mundo["snap"])[0]
        a, b = ordem_do_historico
        pd.DataFrame([{
            "event_name": "UFC Teste", "event_date": "2026-11-07",
            "fighter_a": a, "fighter_b": b,
            "odds_a_decimal": 1.6, "odds_b_decimal": 2.4,
            "model_name": "logreg", "model_prob_a": 0.70,
            "model_side": "Alpha", "actual_winner": None,
            "estado_resultado": "aguardando",
            # o historico aponta para a VIGENTE
            "prediction_id": vigente,
        }]).to_csv(mundo["hist"], index=False)
        return antiga, vigente

    @pytest.mark.parametrize("ordem_do_historico",
                             [("Alpha", "Beta"), ("Beta", "Alpha")])
    def test_uma_entrada_so_e_a_da_versao_vigente(self, mundo,
                                                  ordem_do_historico):
        _antiga, vigente = self._montar(mundo, ordem_do_historico)
        mapa = snapshots.para_render("2026-11-07", mundo["snap"],
                                     mundo["hist"])
        assert len(mapa) == 1, "versoes concorrentes da mesma luta"
        (chave, entrada), = mapa.items()
        assert entrada["prediction_id"] == vigente
        # chaveada na ordem DA VERSAO vigente
        assert chave == ("Beta", "Alpha")
        assert entrada["model_prob_a"] == pytest.approx(0.20)

    @pytest.mark.parametrize("ordem_do_card",
                             [("Alpha", "Beta"), ("Beta", "Alpha")])
    @pytest.mark.parametrize("ordem_do_historico",
                             [("Alpha", "Beta"), ("Beta", "Alpha")])
    def test_desenhar_em_qualquer_ordem_da_Alpha_080(self, mundo,
                                                     ordem_do_historico,
                                                     ordem_do_card):
        from src.card_report import analyze_card
        _antiga, vigente = self._montar(mundo, ordem_do_historico)
        ca, cb = ordem_do_card
        card = pd.DataFrame([{"fighter_a": ca, "fighter_b": cb,
                              "odds_a_decimal": 1.6, "odds_b_decimal": 2.4}])
        frozen = snapshots.para_render("2026-11-07", mundo["snap"],
                                       mundo["hist"])
        a = analyze_card(card, frozen=frozen, somente_snapshot=True)
        lutas = a["favorites"] + a["underdogs"]
        assert len(lutas) == 1, a["no_prediction"]
        luta = lutas[0]
        prob = {luta["fighter_a"]: luta["model_prob_a"],
                luta["fighter_b"]: luta["model_prob_b"]}
        assert prob["Alpha"] == pytest.approx(0.80)
        assert prob["Beta"] == pytest.approx(0.20)
        assert luta["model_side"] == "Alpha"

    def test_ponteiro_invalido_vira_ausencia_com_motivo(self, mundo):
        """Sem versao resolvida, a luta fica INDISPONIVEL e o motivo aparece
        -- nunca outra versao escolhida em silencio."""
        from src.card_report import analyze_card
        self._montar(mundo)
        hist = pd.read_csv(mundo["hist"], dtype=str).fillna("")
        hist.loc[0, "prediction_id"] = "0123456789abcdef"
        hist.to_csv(mundo["hist"], index=False)
        frozen = snapshots.para_render("2026-11-07", mundo["snap"],
                                       mundo["hist"])
        card = pd.DataFrame([{"fighter_a": "Alpha", "fighter_b": "Beta",
                              "odds_a_decimal": 1.6, "odds_b_decimal": 2.4}])
        a = analyze_card(card, frozen=frozen, somente_snapshot=True)
        assert a["favorites"] + a["underdogs"] == []
        assert len(a["no_prediction"]) == 1
        assert "nao existe no arquivo de versoes" in a["no_prediction"][0]["reason"]

    def test_ponteiro_vazio_vira_ausencia_com_motivo(self, mundo):
        from src.card_report import analyze_card
        self._montar(mundo)
        hist = pd.read_csv(mundo["hist"], dtype=str).fillna("")
        hist.loc[0, "prediction_id"] = ""
        hist.to_csv(mundo["hist"], index=False)
        frozen = snapshots.para_render("2026-11-07", mundo["snap"],
                                       mundo["hist"])
        card = pd.DataFrame([{"fighter_a": "Alpha", "fighter_b": "Beta",
                              "odds_a_decimal": 1.6, "odds_b_decimal": 2.4}])
        a = analyze_card(card, frozen=frozen, somente_snapshot=True)
        assert "nao aponta para nenhuma versao" in a["no_prediction"][0]["reason"]

    def test_render_aceita_temporalidade_desconhecida_declarando_a(self, mundo):
        """DESENHAR uma previsao historica sem instante de criacao e legitimo
        desde que a pagina diga que nao recuperou o instante. Quem propoe
        ENTRADA e que nao pode -- e o gate bloqueia."""
        snapshots.registrar([{**self.ANTIGA,
                              "temporalidade": snapshots.TEMPORALIDADE_PERDIDA}],
                            mundo["snap"], criado_em="")
        pid = snapshots.carregar(mundo["snap"]).iloc[-1]["prediction_id"]
        pd.DataFrame([{
            "event_name": "UFC Teste", "event_date": "2026-11-07",
            "fighter_a": "Alpha", "fighter_b": "Beta",
            "odds_a_decimal": 1.6, "odds_b_decimal": 2.4,
            "model_name": "logreg", "model_prob_a": 0.70,
            "model_side": "Alpha", "actual_winner": None,
            "estado_resultado": "aguardando", "prediction_id": pid,
        }]).to_csv(mundo["hist"], index=False)
        mapa = snapshots.para_render("2026-11-07", mundo["snap"], mundo["hist"])
        entrada = mapa[("Alpha", "Beta")]
        assert entrada["model_prob_a"] == pytest.approx(0.70)
        assert "NAO RECUPERAVEL" in entrada["temporalidade"]
        # e o gate, que propoe entrada, recusa
        linha = pd.read_csv(mundo["hist"]).iloc[0]
        assert identidade.motivo_de_bloqueio(
            linha, snapshots_csv=mundo["snap"]) == identidade.BLOQUEIO_VERSAO
