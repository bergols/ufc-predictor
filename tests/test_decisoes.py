"""
Decisões posteriores (`src/decisoes.py`) e o painel de proveniência.

A propriedade central: **o pré-registro não é tocado**. Recapturar o preço de
um card já pré-registrado e escrever o preço novo na linha antiga produziria
uma linha que não descreve instante nenhum — probabilidade de um dia, preço de
outro —, e o CLV calculado dela mediria a confusão, não o movimento de linha.

Então o preço novo vira uma decisão com instante próprio, que APONTA para o
pré-registro. Os testes abaixo medem exatamente isso: o histórico sai do teste
byte a byte igual, e a decisão carrega a sua própria prova.
"""
import hashlib

import pandas as pd
import pytest

import config
from src import decisoes, proveniencia


def _historico(path, *, prob=0.64, estado="aguardando"):
    pd.DataFrame([{
        "event_name": "UFC Teste", "event_date": "2026-11-07",
        "fighter_a": "Alice Silva", "fighter_b": "Bia Costa",
        "odds_a_decimal": 1.60, "odds_b_decimal": 2.40,
        "model_name": "logreg", "model_prob_a": prob,
        "model_side": "Alice Silva", "actual_winner": None,
        # o preco ANTIGO: ficou no pre-registro, e nao tem prova
        "odds_a_tp": 1.70, "odds_b_tp": 2.25,
        "estado_resultado": estado,
    }]).to_csv(path, index=False)


def _card(path, *, odd_a=1.80, fonte="ambas", ref=""):
    pd.DataFrame([{
        "fighter_a": "Alice Silva", "fighter_b": "Bia Costa",
        "odds_a_decimal": 1.60, "odds_b_decimal": 2.40,
        "fonte": fonte, "odds_a_tp": odd_a, "odds_b_tp": 2.10,
        "ref_tp": ref,
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
    return {"hist": hist, "card": card, "log": tmp_path / "decisoes.csv"}


def _prova(mundo, *, inicio="2026-11-07T23:00:00+00:00"):
    chamada = proveniencia.registrar_chamada('[{"x":1}]', fonte="thunderpick",
                                             endpoint="https://thunderpick.io/x")
    return proveniencia.registrar_cotacao(
        chamada_ref=chamada, fonte="thunderpick", casa="Thunderpick",
        mercado="h2h", evento_id="tp1", participante_a="Alice Silva",
        participante_b="Bia Costa", lado="Alice Silva", odd=1.80,
        status="aberto", inicio_programado=inicio)


class TestNaoTocaOPreRegistro:
    def test_avaliar_nao_escreve_no_historico(self, mundo):
        _historico(mundo["hist"])
        _card(mundo["card"], ref=_prova(mundo))
        antes = hashlib.sha256(mundo["hist"].read_bytes()).hexdigest()
        decisoes.avaliar_com_precos_novos("2026-11-07", mundo["card"],
                                          mundo["hist"])
        assert hashlib.sha256(mundo["hist"].read_bytes()).hexdigest() == antes

    def test_registrar_nao_escreve_no_historico(self, mundo):
        _historico(mundo["hist"])
        _card(mundo["card"], ref=_prova(mundo))
        antes = hashlib.sha256(mundo["hist"].read_bytes()).hexdigest()
        ds = decisoes.avaliar_com_precos_novos("2026-11-07", mundo["card"],
                                               mundo["hist"])
        assert decisoes.registrar(ds, mundo["log"]) == 1
        assert hashlib.sha256(mundo["hist"].read_bytes()).hexdigest() == antes

    def test_a_probabilidade_vem_do_PRE_REGISTRO(self, mundo):
        _historico(mundo["hist"], prob=0.64)
        _card(mundo["card"], ref=_prova(mundo))
        d = decisoes.avaliar_com_precos_novos("2026-11-07", mundo["card"],
                                              mundo["hist"])[0]
        assert d["model_prob_a"] == pytest.approx(0.64)
        assert d["previsao_de"].startswith("2026-11-07/")

    def test_o_preco_vem_do_CARD_e_nao_do_historico(self, mundo):
        """O historico tem 1.70 congelado; o card tem 1.80 de hoje. A decisao
        usa o de hoje, e o de ontem continua no pre-registro."""
        _historico(mundo["hist"])
        _card(mundo["card"], odd_a=1.80, ref=_prova(mundo))
        d = decisoes.avaliar_com_precos_novos("2026-11-07", mundo["card"],
                                              mundo["hist"])[0]
        assert d["odd_usada"] == pytest.approx(1.80)
        assert d["ev"] == pytest.approx(0.64 * 1.80, abs=1e-4)
        hist = pd.read_csv(mundo["hist"])
        assert hist.iloc[0]["odds_a_tp"] == pytest.approx(1.70)


class TestDuasAvaliacoesSaoDuasDecisoes:
    def test_anexa_em_vez_de_sobrepor(self, mundo):
        """Sobrepor apagaria justamente a informacao de que o preco andou."""
        _historico(mundo["hist"])
        _card(mundo["card"], odd_a=1.80, ref=_prova(mundo))
        decisoes.registrar(decisoes.avaliar_com_precos_novos(
            "2026-11-07", mundo["card"], mundo["hist"]), mundo["log"])
        _card(mundo["card"], odd_a=1.95, ref=_prova(mundo))
        decisoes.registrar(decisoes.avaliar_com_precos_novos(
            "2026-11-07", mundo["card"], mundo["hist"]), mundo["log"])
        log = pd.read_csv(mundo["log"])
        assert len(log) == 2
        assert sorted(log["odd_usada"]) == pytest.approx([1.80, 1.95])

    def test_a_decisao_aponta_para_o_manifesto(self, mundo):
        _historico(mundo["hist"])
        _card(mundo["card"], ref=_prova(mundo))
        decisoes.registrar(decisoes.avaliar_com_precos_novos(
            "2026-11-07", mundo["card"], mundo["hist"]), mundo["log"])
        log = pd.read_csv(mundo["log"])
        # ligacao com os DADOS que sustentam a previsao; vazia so se nao ha
        # manifesto no disco, e nunca inventada
        assert "manifesto_sha256" in log.columns


class TestElegibilidadeDaDecisao:
    def test_prova_completa_fica_elegivel(self, mundo):
        _historico(mundo["hist"])
        _card(mundo["card"], ref=_prova(mundo))
        d = decisoes.avaliar_com_precos_novos("2026-11-07", mundo["card"],
                                              mundo["hist"])[0]
        assert d["elegivel"] and d["bloqueio"] == ""

    def test_preco_sem_prova_bloqueia(self, mundo):
        _historico(mundo["hist"])
        _card(mundo["card"], ref="")
        d = decisoes.avaliar_com_precos_novos("2026-11-07", mundo["card"],
                                              mundo["hist"])[0]
        assert not d["elegivel"]
        assert "preco de entrada" in d["bloqueio"]

    def test_organizacao_nao_confirmada_bloqueia(self, mundo):
        _historico(mundo["hist"])
        _card(mundo["card"], fonte="so_api", ref=_prova(mundo))
        d = decisoes.avaliar_com_precos_novos("2026-11-07", mundo["card"],
                                              mundo["hist"])[0]
        assert not d["elegivel"]
        assert "organizacao" in d["bloqueio"]

    def test_luta_retirada_do_card_bloqueia(self, mundo):
        _historico(mundo["hist"], estado="retirado_do_card")
        _card(mundo["card"], ref=_prova(mundo))
        d = decisoes.avaliar_com_precos_novos("2026-11-07", mundo["card"],
                                              mundo["hist"])[0]
        assert not d["elegivel"]

    def test_preco_lido_depois_do_inicio_nao_e_pre_luta(self, mundo):
        """Carater pre-luta e metade da exigencia de atualidade."""
        _historico(mundo["hist"])
        _card(mundo["card"], ref=_prova(mundo, inicio="2020-01-01T00:00:00+00:00"))
        d = decisoes.avaliar_com_precos_novos("2026-11-07", mundo["card"],
                                              mundo["hist"])[0]
        assert not d["elegivel"]

    def test_luta_do_card_sem_pre_registro_nao_virou_decisao(self, mundo, caplog):
        """Inventar previsao agora seria pre-registro disfarcado de decisao."""
        _historico(mundo["hist"])
        pd.DataFrame([{
            "fighter_a": "Carla Rocha", "fighter_b": "Dora Lima",
            "odds_a_decimal": 1.9, "odds_b_decimal": 1.95, "fonte": "ambas",
            "odds_a_tp": 1.98, "odds_b_tp": 1.90, "ref_tp": _prova(mundo),
        }]).to_csv(mundo["card"], index=False)
        with caplog.at_level("WARNING"):
            ds = decisoes.avaliar_com_precos_novos("2026-11-07", mundo["card"],
                                                   mundo["hist"])
        assert ds == []
        assert "NAO tem pre-registro" in caplog.text
