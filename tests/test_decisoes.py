"""
Decisões posteriores (`src/decisoes.py`) e o painel de proveniência.

A propriedade central: **o pré-registro não é tocado**. Recapturar o preço de
um card já pré-registrado e escrever o preço novo na linha antiga produziria
uma linha que não descreve instante nenhum — probabilidade de um dia, preço de
outro —, e o CLV calculado dela mediria a confusão, não o movimento de linha.

Então o preço novo vira uma decisão com instante próprio, que APONTA para o
pré-registro. Os testes abaixo medem exatamente isso: o histórico sai do teste
byte a byte igual, e a decisão carrega a sua própria prova.

Duas classes aqui são regressões de defeitos encontrados em revisão
independente do commit 7d70038, e as duas davam número plausível em vez de
erro:

  `TestProvaDoLadoCerto`    cinco decisões elegíveis citavam a cotação do
                            ADVERSÁRIO. O card guardava só a prova do lado A,
                            e a conferência não comparava o lado da prova com
                            o lado escolhido.
  `TestOrientacaoDaProbabilidade`
                            `model_prob_a` pertence à orientação do SNAPSHOT,
                            e era lida pela orientação do CARD. Card invertido
                            devolvia 1-p, a probabilidade do adversário.
"""
import hashlib
import json

import pandas as pd
import pytest

import config
from src import decisoes, proveniencia, snapshots

_INICIO = "2026-11-07T23:00:00+00:00"


def _historico(path, *, prob=0.64, estado="aguardando", lado="Alice Silva",
               a="Alice Silva", b="Bia Costa", snapshot=True,
               prob_no_snapshot=None, snap=None):
    """
    Pre-registro de uma luta MAIS a versao imutavel que ele aponta.

    A decisao resolve `prediction_id` antes de qualquer calculo, entao um
    historico sem versao nao decide nada -- e e isso que
    `snapshot=False` testa. `prob_no_snapshot` deixa a versao DISCORDAR da
    linha, que e o caso em que a decisao tem de bloquear em vez de escolher.
    """
    pid = ""
    if snapshot:
        pid = snapshots.registrar([{
            "event_date": "2026-11-07", "event_name": "UFC Teste",
            "fighter_a": a, "fighter_b": b, "model_name": "logreg",
            "model_prob_a": prob if prob_no_snapshot is None else prob_no_snapshot,
            "model_side": lado, "odds_a_decimal": 1.60,
            "odds_b_decimal": 2.40, "manifesto_sha256": "man-1",
        }], snap)
        # falha de escrita devolve lista vazia, e a linha fica SEM ponteiro --
        # que e exatamente o que o teste da falha de persistencia precisa
        pid = pid[0] if pid else ""
    pd.DataFrame([{
        "event_name": "UFC Teste", "event_date": "2026-11-07",
        "fighter_a": a, "fighter_b": b,
        "odds_a_decimal": 1.60, "odds_b_decimal": 2.40,
        "model_name": "logreg", "model_prob_a": prob,
        "model_side": lado, "actual_winner": None,
        # o preco ANTIGO: ficou no pre-registro, e nao tem prova
        "odds_a_tp": 1.70, "odds_b_tp": 2.25,
        "estado_resultado": estado, "prediction_id": pid,
    }]).to_csv(path, index=False)
    return pid


def _card(path, *, odd_a=1.80, odd_b=2.10, fonte="ambas", ref_a="", ref_b="",
          a="Alice Silva", b="Bia Costa"):
    pd.DataFrame([{
        "fighter_a": a, "fighter_b": b,
        "odds_a_decimal": 1.60, "odds_b_decimal": 2.40,
        "fonte": fonte, "odds_a_tp": odd_a, "odds_b_tp": odd_b,
        "ref_tp": ref_a, "ref_tp_a": ref_a, "ref_tp_b": ref_b,
    }]).to_csv(path, index=False)


@pytest.fixture
def mundo(tmp_path, monkeypatch):
    pasta = tmp_path / "proveniencia"
    monkeypatch.setattr(proveniencia, "PASTA", pasta)
    monkeypatch.setattr(proveniencia, "INDICE", pasta / "chamadas.csv")
    monkeypatch.setattr(proveniencia, "INDICE_COTACOES", pasta / "cotacoes.csv")
    hist = tmp_path / "prediction_history.csv"
    card = tmp_path / "card.csv"
    monkeypatch.setattr(config, "PREDICTION_HISTORY_CSV", hist)
    monkeypatch.setattr(config, "UPCOMING_CARD_CSV", card)
    monkeypatch.setattr(decisoes, "ARQUIVO", tmp_path / "decisoes.csv")
    # as versoes moram na pasta do historico que aponta para elas
    monkeypatch.setattr(snapshots, "ARQUIVO",
                        tmp_path / "previsoes_snapshots.csv")
    snapshots._CACHE.clear()
    return {"hist": hist, "card": card, "log": tmp_path / "decisoes.csv",
            "snap": tmp_path / "previsoes_snapshots.csv"}


def _provas(*, odd_a=1.80, odd_b=2.10, inicio=_INICIO, a="Alice Silva",
            b="Bia Costa", status=1) -> tuple[str, str]:
    """
    Registra a chamada e as DUAS cotações, como a captura real faz, e devolve
    (ref do lado A, ref do lado B).

    O corpo tem a forma que a casa devolve de verdade — `home`/`away` com
    `odd_home`/`odd_away` —, porque a conferência exige que a odd declarada
    apareça no conteúdo persistido. Corpo de faz-de-conta aprovaria qualquer
    número.
    """
    corpo = json.dumps([{"home": a, "away": b, "odd_home": odd_a,
                         "odd_away": odd_b, "inicio": inicio,
                         "status": status}], ensure_ascii=False,
                       sort_keys=True)
    chamada = proveniencia.registrar_chamada(
        corpo, fonte="thunderpick", endpoint="https://thunderpick.io/x",
        markets="h2h")
    comum = dict(chamada_ref=chamada, fonte="thunderpick", casa="Thunderpick",
                 mercado="h2h", evento_id=inicio, participante_a=a,
                 participante_b=b, status="aberto", inicio_programado=inicio)
    return (proveniencia.registrar_cotacao(lado=a, odd=odd_a, **comum),
            proveniencia.registrar_cotacao(lado=b, odd=odd_b, **comum))


def _uma_decisao(mundo, **kw):
    return decisoes.avaliar_com_precos_novos("2026-11-07", mundo["card"],
                                             mundo["hist"], **kw)[0]


class TestNaoTocaOPreRegistro:
    def test_avaliar_nao_escreve_no_historico(self, mundo):
        _historico(mundo["hist"])
        ra, rb = _provas()
        _card(mundo["card"], ref_a=ra, ref_b=rb)
        antes = hashlib.sha256(mundo["hist"].read_bytes()).hexdigest()
        decisoes.avaliar_com_precos_novos("2026-11-07", mundo["card"],
                                          mundo["hist"])
        assert hashlib.sha256(mundo["hist"].read_bytes()).hexdigest() == antes

    def test_registrar_nao_escreve_no_historico(self, mundo):
        _historico(mundo["hist"])
        ra, rb = _provas()
        _card(mundo["card"], ref_a=ra, ref_b=rb)
        antes = hashlib.sha256(mundo["hist"].read_bytes()).hexdigest()
        ds = decisoes.avaliar_com_precos_novos("2026-11-07", mundo["card"],
                                               mundo["hist"])
        assert decisoes.registrar(ds, mundo["log"]) == 1
        assert hashlib.sha256(mundo["hist"].read_bytes()).hexdigest() == antes

    def test_a_probabilidade_vem_do_PRE_REGISTRO(self, mundo):
        _historico(mundo["hist"], prob=0.64)
        ra, rb = _provas()
        _card(mundo["card"], ref_a=ra, ref_b=rb)
        d = _uma_decisao(mundo)
        assert d["model_prob_a"] == pytest.approx(0.64)
        assert d["previsao_de"].startswith("2026-11-07/")

    def test_o_preco_vem_do_CARD_e_nao_do_historico(self, mundo):
        """O historico tem 1.70 congelado; o card tem 1.80 de hoje. A decisao
        usa o de hoje, e o de ontem continua no pre-registro."""
        _historico(mundo["hist"])
        ra, rb = _provas(odd_a=1.80)
        _card(mundo["card"], odd_a=1.80, ref_a=ra, ref_b=rb)
        d = _uma_decisao(mundo)
        assert d["odd_usada"] == pytest.approx(1.80)
        assert d["ev"] == pytest.approx(0.64 * 1.80, abs=1e-4)
        hist = pd.read_csv(mundo["hist"])
        assert hist.iloc[0]["odds_a_tp"] == pytest.approx(1.70)


class TestDuasAvaliacoesSaoDuasDecisoes:
    def test_anexa_em_vez_de_sobrepor(self, mundo):
        """Sobrepor apagaria justamente a informacao de que o preco andou."""
        _historico(mundo["hist"])
        ra, rb = _provas(odd_a=1.80)
        _card(mundo["card"], odd_a=1.80, ref_a=ra, ref_b=rb)
        decisoes.registrar(decisoes.avaliar_com_precos_novos(
            "2026-11-07", mundo["card"], mundo["hist"]), mundo["log"])
        ra, rb = _provas(odd_a=1.95)
        _card(mundo["card"], odd_a=1.95, ref_a=ra, ref_b=rb)
        decisoes.registrar(decisoes.avaliar_com_precos_novos(
            "2026-11-07", mundo["card"], mundo["hist"]), mundo["log"])
        log = pd.read_csv(mundo["log"])
        assert len(log) == 2
        assert sorted(log["odd_usada"]) == pytest.approx([1.80, 1.95])

    def test_a_decisao_aponta_para_o_manifesto(self, mundo):
        _historico(mundo["hist"])
        ra, rb = _provas()
        _card(mundo["card"], ref_a=ra, ref_b=rb)
        decisoes.registrar(decisoes.avaliar_com_precos_novos(
            "2026-11-07", mundo["card"], mundo["hist"]), mundo["log"])
        log = pd.read_csv(mundo["log"])
        # ligacao com os DADOS que sustentam a previsao; vazia so se nao ha
        # manifesto no disco, e nunca inventada
        assert "manifesto_sha256" in log.columns

    def test_retificacao_e_anexada_sem_apagar_a_decisao_errada(self, mundo):
        """A decisao errada FICA. Apaga-la esconderia que ela existiu, e e o
        registro dela que torna a correcao auditavel."""
        _historico(mundo["hist"])
        ra, rb = _provas()
        _card(mundo["card"], ref_a=ra, ref_b=rb)
        ds = decisoes.avaliar_com_precos_novos("2026-11-07", mundo["card"],
                                               mundo["hist"])
        decisoes.registrar(ds, mundo["log"])
        decisoes.registrar(ds, mundo["log"], retifica="prova do lado certo")
        log = pd.read_csv(mundo["log"])
        assert len(log) == 2
        assert log["retifica"].isna().sum() == 1
        assert "prova do lado certo" in set(log["retifica"].dropna())

    def test_cabecalho_antigo_e_migrado_sem_perder_linha(self, mundo):
        """Coluna nova num arquivo escrito com o cabecalho antigo
        desalinharia as linhas em silencio."""
        antigas = ("decidido_em", "event_date", "fighter_a", "fighter_b",
                   "lado", "odd_usada", "ev", "elegivel")
        mundo["log"].write_text(
            ",".join(antigas) + "\n"
            "2026-11-05T00:00:00+00:00,2026-11-07,Alice Silva,Bia Costa,"
            "Alice Silva,1.7,1.09,True\n", encoding="utf-8")
        _historico(mundo["hist"])
        ra, rb = _provas()
        _card(mundo["card"], ref_a=ra, ref_b=rb)
        decisoes.registrar(decisoes.avaliar_com_precos_novos(
            "2026-11-07", mundo["card"], mundo["hist"]), mundo["log"])
        log = pd.read_csv(mundo["log"])
        assert list(log.columns) == list(decisoes.COLUNAS)
        assert len(log) == 2
        # a linha antiga continua la, com o que ela de fato afirmava
        velha = log.iloc[0]
        assert velha["odd_usada"] == pytest.approx(1.7)
        assert pd.isna(velha["ref_preco"])


class TestProvaDoLadoCerto:
    """
    REGRESSAO. O card guardava so `ref_tp`, que e a prova do LADO A, e a
    conferencia nunca comparava o lado da prova com o lado escolhido. Uma
    decisao no lado B citava a cotacao do adversario: mesma chamada, mesmo
    corpo, mesmo hash, so o lado trocado -- e passava inteira.
    """

    def test_decisao_no_lado_B_usa_a_prova_do_lado_B(self, mundo):
        _historico(mundo["hist"], lado="Bia Costa")
        ra, rb = _provas(odd_a=1.80, odd_b=2.10)
        _card(mundo["card"], ref_a=ra, ref_b=rb)
        d = _uma_decisao(mundo)
        assert d["lado"] == "Bia Costa"
        assert d["odd_usada"] == pytest.approx(2.10)
        assert d["ref_preco"] == rb
        assert d["elegivel"], d["bloqueio"]

    def test_prova_do_adversario_NAO_aprova(self, mundo):
        """O defeito exato: ref do lado A numa decisao do lado B."""
        _historico(mundo["hist"], lado="Bia Costa")
        ra, _ = _provas()
        _card(mundo["card"], ref_a=ra, ref_b=ra)
        d = _uma_decisao(mundo)
        assert not d["elegivel"]
        assert "adversario" in d["bloqueio"]

    def test_sem_prova_do_lado_B_bloqueia_em_vez_de_cair_no_lado_A(self, mundo):
        _historico(mundo["hist"], lado="Bia Costa")
        ra, _ = _provas()
        _card(mundo["card"], ref_a=ra, ref_b="")
        d = _uma_decisao(mundo)
        assert not d["elegivel"]
        assert "sem referencia" in d["bloqueio"]

    def test_prova_de_outro_confronto_bloqueia(self, mundo):
        _historico(mundo["hist"])
        outra, _ = _provas(a="Carla Rocha", b="Dora Lima")
        _card(mundo["card"], ref_a=outra, ref_b=outra)
        d = _uma_decisao(mundo)
        assert not d["elegivel"]
        assert "confronto" in d["bloqueio"]

    def test_odd_que_nao_esta_no_corpo_bloqueia(self, mundo):
        """A linha de cotacao afirma um preco; o corpo persistido e quem o
        comprova. Afirmacao que o conteudo nao sustenta nao e prova."""
        _historico(mundo["hist"])
        ra, rb = _provas(odd_a=1.80)
        # a decisao usa 1.99, que nao esta em lugar nenhum do corpo
        _card(mundo["card"], odd_a=1.99, ref_a=ra, ref_b=rb)
        d = _uma_decisao(mundo)
        assert not d["elegivel"]
        assert "prova registra" in d["bloqueio"]

    def test_mercado_suspenso_bloqueia(self, mundo):
        """status != 1 e mercado suspenso: preco congelado pela casa nao e
        preco de entrada."""
        _historico(mundo["hist"])
        ra, rb = _provas(status=3)
        _card(mundo["card"], ref_a=ra, ref_b=rb)
        d = _uma_decisao(mundo)
        assert not d["elegivel"]
        assert "suspenso" in d["bloqueio"]


class TestOrientacaoDaProbabilidade:
    """
    REGRESSAO. `model_prob_a` e a probabilidade de `fighter_a` DO SNAPSHOT. O
    card e capturado depois e pode listar o mesmo confronto na ordem inversa;
    lendo a probabilidade pela ordem do card, a decisao usava 1-p -- a
    probabilidade do adversario -- e o retorno esperado saia errado sem
    nenhum sinal de erro.

    Alpha com p=0,70 e odd 1,80: retorno bruto esperado 1,26, qualquer que
    seja a ordem em que as duas pontas listam o confronto.
    """
    ESPERADO = 1.26

    def _montar(self, mundo, *, hist_ordem, card_ordem):
        a, b = hist_ordem
        prob = 0.70 if a == "Alpha" else 0.30
        _historico(mundo["hist"], prob=prob, lado="Alpha", a=a, b=b)
        ca, cb = card_ordem
        odd_a, odd_b = (1.80, 2.40) if ca == "Alpha" else (2.40, 1.80)
        ra, rb = _provas(odd_a=odd_a, odd_b=odd_b, a=ca, b=cb)
        _card(mundo["card"], a=ca, b=cb, odd_a=odd_a, odd_b=odd_b,
              ref_a=ra, ref_b=rb)

    @pytest.mark.parametrize("hist_ordem,card_ordem", [
        (("Alpha", "Beta"), ("Alpha", "Beta")),   # as duas na mesma ordem
        (("Alpha", "Beta"), ("Beta", "Alpha")),   # CARD invertido
        (("Beta", "Alpha"), ("Alpha", "Beta")),   # SNAPSHOT invertido
        (("Beta", "Alpha"), ("Beta", "Alpha")),   # as duas invertidas
    ])
    def test_inverter_A_e_B_preserva_o_retorno_esperado(self, mundo,
                                                        hist_ordem,
                                                        card_ordem):
        self._montar(mundo, hist_ordem=hist_ordem, card_ordem=card_ordem)
        d = _uma_decisao(mundo)
        assert d["lado"] == "Alpha"
        assert d["prob_lado"] == pytest.approx(0.70)
        assert d["odd_usada"] == pytest.approx(1.80)
        assert d["ev"] == pytest.approx(self.ESPERADO, abs=1e-4)
        assert d["elegivel"], d["bloqueio"]

    def test_o_card_invertido_nao_entrega_a_probabilidade_do_adversario(
            self, mundo):
        """O numero que o defeito produzia era 0,30 x 1,80 = 0,54: plausivel,
        e errado."""
        self._montar(mundo, hist_ordem=("Alpha", "Beta"),
                     card_ordem=("Beta", "Alpha"))
        d = _uma_decisao(mundo)
        assert d["ev"] != pytest.approx(0.54, abs=1e-4)

    def test_lado_que_nao_e_do_confronto_bloqueia(self, mundo):
        """Inventar orientacao para um nome que nao esta na versao seria pior
        que nao decidir. A decisao entra BLOQUEADA e visivel na contagem, em
        vez de desaparecer."""
        _historico(mundo["hist"], lado="Fulana de Fora")
        ra, rb = _provas()
        _card(mundo["card"], ref_a=ra, ref_b=rb)
        d = _uma_decisao(mundo)
        assert not d["elegivel"]
        assert "nao e nenhum dos dois participantes da versao" in d["bloqueio"]
        assert d["prob_lado"] == ""


class TestElegibilidadeDaDecisao:
    def test_prova_completa_fica_elegivel(self, mundo):
        _historico(mundo["hist"])
        ra, rb = _provas()
        _card(mundo["card"], ref_a=ra, ref_b=rb)
        d = _uma_decisao(mundo)
        assert d["elegivel"] and d["bloqueio"] == ""

    def test_preco_sem_prova_bloqueia(self, mundo):
        _historico(mundo["hist"])
        _card(mundo["card"])
        d = _uma_decisao(mundo)
        assert not d["elegivel"]
        assert "preco de entrada" in d["bloqueio"]

    def test_organizacao_nao_confirmada_bloqueia(self, mundo):
        _historico(mundo["hist"])
        ra, rb = _provas()
        _card(mundo["card"], fonte="so_api", ref_a=ra, ref_b=rb)
        d = _uma_decisao(mundo)
        assert not d["elegivel"]
        assert "organizacao" in d["bloqueio"]

    def test_luta_retirada_do_card_bloqueia(self, mundo):
        _historico(mundo["hist"], estado="retirado_do_card")
        ra, rb = _provas()
        _card(mundo["card"], ref_a=ra, ref_b=rb)
        d = _uma_decisao(mundo)
        assert not d["elegivel"]

    def test_preco_lido_depois_do_inicio_nao_e_pre_luta(self, mundo):
        """Carater pre-luta e metade da exigencia de atualidade."""
        _historico(mundo["hist"])
        ra, rb = _provas(inicio="2020-01-01T00:00:00+00:00")
        _card(mundo["card"], ref_a=ra, ref_b=rb)
        d = _uma_decisao(mundo)
        assert not d["elegivel"]

    def test_luta_do_card_sem_pre_registro_nao_virou_decisao(self, mundo, caplog):
        """Inventar previsao agora seria pre-registro disfarcado de decisao."""
        _historico(mundo["hist"])
        ra, rb = _provas(a="Carla Rocha", b="Dora Lima")
        pd.DataFrame([{
            "fighter_a": "Carla Rocha", "fighter_b": "Dora Lima",
            "odds_a_decimal": 1.9, "odds_b_decimal": 1.95, "fonte": "ambas",
            "odds_a_tp": 1.80, "odds_b_tp": 2.10,
            "ref_tp": ra, "ref_tp_a": ra, "ref_tp_b": rb,
        }]).to_csv(mundo["card"], index=False)
        with caplog.at_level("WARNING"):
            ds = decisoes.avaliar_com_precos_novos("2026-11-07", mundo["card"],
                                                   mundo["hist"])
        assert ds == []
        assert "NAO tem pre-registro" in caplog.text


class TestContextoDoCorpo:
    """
    REGRESSAO. A leitura do corpo persistido colhia pares (nome, odd) SOLTOS,
    varrendo a estrutura inteira. O corpo de um card tem dezenas de lutas e
    mais de cem precos, e "Alpha paga 1,80 em algum lugar deste JSON" nao e
    prova de que Alpha x Beta paga 1,80. Agora cada cotacao e lida com o
    contexto dela -- evento, inicio, os dois participantes, casa, mercado,
    lado, preco e status -- e todos sao conferidos contra a mesma linha do
    indice.
    """

    def _corpo_da_casa(self, lutas) -> str:
        return json.dumps(
            [{"home": h, "away": v, "odd_home": oh, "odd_away": ov,
              "inicio": ini, "status": st}
             for h, v, oh, ov, ini, st in lutas],
            ensure_ascii=False, sort_keys=True)

    def _registrar(self, corpo, *, fonte="thunderpick", casa="Thunderpick",
                   mercado="h2h", evento_id=_INICIO, a="Alpha", b="Beta",
                   lado="Alpha", odd=1.80, inicio=_INICIO,
                   atualizada="") -> str:
        chamada = proveniencia.registrar_chamada(
            corpo, fonte=fonte, endpoint="https://exemplo/x", markets="h2h")
        return proveniencia.registrar_cotacao(
            chamada_ref=chamada, fonte=fonte, casa=casa, mercado=mercado,
            evento_id=evento_id, participante_a=a, participante_b=b,
            lado=lado, odd=odd, status="aberto", inicio_programado=inicio,
            fonte_atualizada_em=atualizada)

    def test_contexto_completo_aprova(self, mundo):
        ref = self._registrar(
            self._corpo_da_casa([("Alpha", "Beta", 1.80, 2.10, _INICIO, 1)]))
        ok, motivo = proveniencia.validar_cotacao(
            ref, casa="Thunderpick", mercado="h2h",
            participantes=("Alpha", "Beta"), lado="Alpha", odd=1.80)
        assert ok, motivo

    def test_nome_e_odd_em_OUTRO_confronto_nao_aprovam(self, mundo):
        """O caso do enunciado: Alpha x Outro no corpo, Alpha x Beta no
        indice. A odd do lado Alpha existe no corpo -- em outra luta."""
        ref = self._registrar(
            self._corpo_da_casa([("Alpha", "Outro", 1.80, 2.05, _INICIO, 1)]),
            a="Alpha", b="Beta")
        ok, motivo = proveniencia.validar_cotacao(
            ref, participantes=("Alpha", "Beta"), lado="Alpha", odd=1.80)
        assert not ok
        assert "OUTRO confronto" in motivo
        assert "Alpha x Outro" in motivo

    def test_confronto_certo_no_mesmo_corpo_ainda_aprova(self, mundo):
        """O corpo tem as duas lutas; so uma e a nossa. A presenca da outra
        nao pode nem aprovar por engano nem reprovar por excesso."""
        ref = self._registrar(self._corpo_da_casa([
            ("Alpha", "Outro", 1.80, 2.05, _INICIO, 1),
            ("Alpha", "Beta", 1.80, 2.10, _INICIO, 1),
        ]))
        ok, motivo = proveniencia.validar_cotacao(
            ref, participantes=("Alpha", "Beta"), lado="Alpha", odd=1.80)
        assert ok, motivo

    def test_inicio_divergente_bloqueia(self, mundo):
        """Mesmos nomes, mesma odd, outra data: e a revanche, nao esta luta."""
        ref = self._registrar(
            self._corpo_da_casa([("Alpha", "Beta", 1.80, 2.10,
                                  "2027-03-01T23:00:00+00:00", 1)]),
            inicio=_INICIO, evento_id=_INICIO)
        ok, motivo = proveniencia.validar_cotacao(
            ref, participantes=("Alpha", "Beta"), lado="Alpha", odd=1.80)
        assert not ok
        assert "contexto diferente" in motivo

    def test_status_suspenso_no_corpo_bloqueia(self, mundo):
        ref = self._registrar(
            self._corpo_da_casa([("Alpha", "Beta", 1.80, 2.10, _INICIO, 3)]))
        ok, motivo = proveniencia.validar_cotacao(
            ref, participantes=("Alpha", "Beta"), lado="Alpha", odd=1.80)
        assert not ok
        assert "suspenso" in motivo

    def test_fonte_da_cotacao_tem_de_ser_a_do_corpo(self, mundo):
        """Uma resposta da The Odds API nao comprova a Thunderpick. A linha
        dizia uma fonte e apontava para o corpo de outra."""
        ref = self._registrar(
            self._corpo_da_casa([("Alpha", "Beta", 1.80, 2.10, _INICIO, 1)]),
            fonte="the_odds_api")
        # a linha afirma thunderpick; o corpo e do feed
        linha = proveniencia._linha_de_cotacao(ref)
        chamada = linha["chamada_ref"]
        outra = proveniencia.registrar_cotacao(
            chamada_ref=chamada, fonte="thunderpick", casa="Thunderpick",
            mercado="h2h", evento_id=_INICIO, participante_a="Alpha",
            participante_b="Beta", lado="Alpha", odd=1.80, status="aberto",
            inicio_programado=_INICIO)
        ok, motivo = proveniencia.validar_cotacao(outra, lado="Alpha", odd=1.80)
        assert not ok
        assert "corpo guardado e de" in motivo

    def test_casa_do_feed_tem_de_bater(self, mundo):
        """No feed a casa esta no corpo. A prova da Betfair nao serve para
        uma decisao que diz usar a Pinnacle."""
        corpo = json.dumps([{
            "id": "ev1", "commence_time": _INICIO,
            "home_team": "Alpha", "away_team": "Beta",
            "bookmakers": [{"key": "betfair", "title": "Betfair",
                            "last_update": "2026-11-07T10:00:00Z",
                            "markets": [{"key": "h2h",
                                         "outcomes": [{"name": "Alpha", "price": 1.80},
                                                      {"name": "Beta", "price": 2.10}]}]}],
        }], ensure_ascii=False, sort_keys=True)
        ref = self._registrar(corpo, fonte="the_odds_api", casa="Pinnacle",
                              evento_id="ev1",
                              atualizada="2026-11-07T10:00:00Z")
        ok, motivo = proveniencia.validar_cotacao(ref, lado="Alpha", odd=1.80)
        assert not ok
        certa = self._registrar(corpo, fonte="the_odds_api", casa="Betfair",
                                evento_id="ev1",
                                atualizada="2026-11-07T10:00:00Z")
        ok, motivo = proveniencia.validar_cotacao(certa, lado="Alpha", odd=1.80)
        assert ok, motivo

    def test_mercado_divergente_bloqueia(self, mundo):
        corpo = json.dumps([{
            "id": "ev1", "commence_time": _INICIO,
            "home_team": "Alpha", "away_team": "Beta",
            "bookmakers": [{"key": "betfair", "title": "Betfair",
                            "markets": [{"key": "totals",
                                         "outcomes": [{"name": "Alpha", "price": 1.80}]}]}],
        }], ensure_ascii=False, sort_keys=True)
        ref = self._registrar(corpo, fonte="the_odds_api", casa="Betfair",
                              mercado="h2h", evento_id="ev1")
        ok, motivo = proveniencia.validar_cotacao(ref, lado="Alpha", odd=1.80)
        assert not ok

    def test_corpo_em_forma_desconhecida_bloqueia(self, mundo):
        """Formato que nao sabemos ler nao e aprovacao por omissao."""
        ref = self._registrar('{"alguma": "coisa", "preco": 1.8}')
        ok, motivo = proveniencia.validar_cotacao(ref, lado="Alpha", odd=1.80)
        assert not ok
        assert "forma reconhecida" in motivo


class TestDecisaoResolveAVersao:
    """
    REGRESSAO. `prediction_id` era carregado como ETIQUETA: a decisao o
    copiava para o log e lia probabilidade, lado e participantes das colunas
    MUTAVEIS do historico -- as mesmas que a regeracao reescreve. Ponteiro que
    nao e resolvido nao e ponteiro, e uma divergencia entre a versao e a linha
    passava sem nenhum sinal.

    Agora a versao e resolvida ANTES de qualquer calculo, e tudo o que a
    decisao reporta vem dela.
    """

    def _card_pronto(self, mundo):
        ra, rb = _provas()
        _card(mundo["card"], ref_a=ra, ref_b=rb)

    def test_a_prova_vem_da_versao_e_nao_do_historico(self, mundo):
        """O log cita o modelo e o manifesto DA VERSAO."""
        _historico(mundo["hist"], snap=mundo["snap"])
        self._card_pronto(mundo)
        d = _uma_decisao(mundo)
        assert d["manifesto_sha256"] == "man-1"
        assert d["model_name"] == "logreg"
        assert d["elegivel"], d["bloqueio"]

    def test_a_prediction_id_vazio_bloqueia(self, mundo):
        _historico(mundo["hist"], snapshot=False)
        self._card_pronto(mundo)
        d = _uma_decisao(mundo)
        assert not d["elegivel"]
        assert "nao aponta para nenhuma versao" in d["bloqueio"]
        assert d["model_prob_a"] == "", "nada pode ser lido do historico"

    def test_b_id_inexistente_bloqueia(self, mundo):
        _historico(mundo["hist"], snap=mundo["snap"])
        hist = pd.read_csv(mundo["hist"], dtype=str).fillna("")
        hist.loc[0, "prediction_id"] = "0123456789abcdef"
        hist.to_csv(mundo["hist"], index=False)
        self._card_pronto(mundo)
        d = _uma_decisao(mundo)
        assert not d["elegivel"]
        assert "nao existe no arquivo de versoes" in d["bloqueio"]

    def test_c_snapshot_064_com_historico_095_nunca_usa_095(self, mundo):
        """O caso do enunciado. A versao diz 0,64 e a linha diz 0,95: uma das
        duas foi mexida e nao da para saber qual. A decisao bloqueia, e o
        unico numero que ela reporta e o da versao."""
        _historico(mundo["hist"], prob=0.95, prob_no_snapshot=0.64,
                   snap=mundo["snap"])
        self._card_pronto(mundo)
        d = _uma_decisao(mundo)
        assert not d["elegivel"]
        assert "discordam em probabilidade" in d["bloqueio"]
        assert d["model_prob_a"] == pytest.approx(0.64)
        assert d["prob_lado"] == pytest.approx(0.64)
        assert d["model_prob_a"] != pytest.approx(0.95)
        assert "0.95" not in str(d["ev"])

    def test_d_id_de_outro_confronto_bloqueia(self, mundo):
        outro = snapshots.registrar([{
            "event_date": "2026-11-07", "event_name": "UFC Teste",
            "fighter_a": "Carla Rocha", "fighter_b": "Dora Lima",
            "model_name": "logreg", "model_prob_a": 0.70,
            "model_side": "Carla Rocha",
        }], mundo["snap"])[0]
        _historico(mundo["hist"], snap=mundo["snap"])
        hist = pd.read_csv(mundo["hist"], dtype=str).fillna("")
        hist.loc[0, "prediction_id"] = outro
        hist.to_csv(mundo["hist"], index=False)
        self._card_pronto(mundo)
        d = _uma_decisao(mundo)
        assert not d["elegivel"]
        assert "de outro confronto" in d["bloqueio"]
        assert "Carla Rocha x Dora Lima" in d["bloqueio"]

    def test_id_de_outro_evento_bloqueia(self, mundo):
        outro = snapshots.registrar([{
            "event_date": "2026-03-01", "event_name": "UFC Antigo",
            "fighter_a": "Alice Silva", "fighter_b": "Bia Costa",
            "model_name": "logreg", "model_prob_a": 0.70,
            "model_side": "Alice Silva",
        }], mundo["snap"])[0]
        _historico(mundo["hist"], snap=mundo["snap"])
        hist = pd.read_csv(mundo["hist"], dtype=str).fillna("")
        hist.loc[0, "prediction_id"] = outro
        hist.to_csv(mundo["hist"], index=False)
        self._card_pronto(mundo)
        d = _uma_decisao(mundo)
        assert not d["elegivel"]
        assert "de outro evento" in d["bloqueio"]

    def test_id_duplicado_bloqueia(self, mundo):
        """Duas linhas com o mesmo id: nao da para dizer qual e a versao, e
        escolher uma seria inventar a resposta."""
        pid = _historico(mundo["hist"], snap=mundo["snap"])
        linhas = mundo["snap"].read_text(encoding="utf-8").splitlines()
        mundo["snap"].write_text("\n".join(linhas + [linhas[-1]]) + "\n",
                                 encoding="utf-8")
        snapshots._CACHE.clear()
        self._card_pronto(mundo)
        d = _uma_decisao(mundo)
        assert not d["elegivel"]
        assert "mais de uma vez" in d["bloqueio"]
        assert pid in d["bloqueio"]

    def test_temporalidade_desconhecida_bloqueia_e_fica_declarada(self, mundo):
        """Semeadura nao tem instante de criacao. Sem ele nao da para
        certificar que a previsao e pre-luta, e inventar a certificacao seria
        pior que recusar a entrada."""
        snapshots.registrar([{
            "event_date": "2026-11-07", "event_name": "UFC Teste",
            "fighter_a": "Alice Silva", "fighter_b": "Bia Costa",
            "model_name": "logreg", "model_prob_a": 0.64,
            "model_side": "Alice Silva",
            "temporalidade": snapshots.TEMPORALIDADE_PERDIDA,
        }], mundo["snap"], criado_em="")
        semeada = snapshots.carregar(mundo["snap"]).iloc[-1]["prediction_id"]
        _historico(mundo["hist"], snapshot=False)
        hist = pd.read_csv(mundo["hist"], dtype=str).fillna("")
        hist.loc[0, "prediction_id"] = semeada
        hist.to_csv(mundo["hist"], index=False)
        self._card_pronto(mundo)
        d = _uma_decisao(mundo)
        assert not d["elegivel"]
        assert "temporalidade" in d["bloqueio"]
        assert "pre-luta" in d["bloqueio"]

    def test_falha_ao_persistir_a_versao_nao_libera_selecao(self, mundo,
                                                            monkeypatch):
        """Disco cheio, permissao negada: `registrar` devolve lista vazia e a
        linha fica sem ponteiro. Isso nao pode virar selecao liberada."""
        monkeypatch.setattr(snapshots, "registrar", lambda *a, **k: [])
        _historico(mundo["hist"], snapshot=True, snap=mundo["snap"])
        self._card_pronto(mundo)
        d = _uma_decisao(mundo)
        assert not d["elegivel"]
        assert "nao aponta para nenhuma versao" in d["bloqueio"]
