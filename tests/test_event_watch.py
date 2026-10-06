"""
Alarme de evento perdido (src/event_watch.py).

O alarme de fechamento varre o historico. Este nao pode: quando o card passa
sem pre-registro NAO HA linha no historico para varrer. Foi assim que o
UFC 332 (03/out/2026) sumiu da serie sem nenhum aviso -- a semana simplesmente
nao aconteceu, e so apareceu numa conferencia manual.

Dai as duas fontes: a base de resultados para o que ja passou, a API para o
que vem. Sao testadas separadas porque falham separadas.
"""
import json
from datetime import date, timedelta

import pandas as pd
import pytest

from src import event_watch as ew

HOJE = date(2026, 10, 6)


def _hist(datas):
    return pd.DataFrame([{"event_date": d, "event_name": f"UFC {i}",
                          "fighter_a": "A", "fighter_b": "B"}
                         for i, d in enumerate(datas)])


def _base(pares):
    return pd.DataFrame([{"event_date": d, "event_name": n} for d, n in pares])


def _api(datas_iso):
    return [{"commence_time": t} for t in datas_iso]


class TestEventosPerdidos:
    def test_card_que_aconteceu_sem_pre_registro_e_acusado(self):
        p = ew.eventos_perdidos(
            _hist(["2026-09-26"]),
            _base([("2026-09-26", "UFC FN"), ("2026-10-03", "UFC 332")]),
            hoje=HOJE)
        assert [e["event_date"] for e in p] == ["2026-10-03"]
        assert p[0]["event_name"] == "UFC 332"
        assert p[0]["estado"] == "perdido"

    def test_nada_antes_do_primeiro_evento_da_serie(self):
        """A base tem 790 eventos desde 1994. Listar todos como perdidos
        transformaria o alarme em papel de parede."""
        p = ew.eventos_perdidos(
            _hist(["2026-09-26"]),
            _base([("1994-03-11", "UFC 2"), ("2019-07-06", "UFC 239"),
                   ("2026-09-26", "UFC FN")]),
            hoje=HOJE)
        assert p == []

    def test_card_de_hoje_ainda_nao_esta_perdido(self):
        """Hoje ainda da para registrar. Cobrar agora seria cobrar cedo."""
        p = ew.eventos_perdidos(
            _hist(["2026-09-26"]),
            _base([("2026-09-26", "UFC FN"), (HOJE.isoformat(), "UFC Hoje")]),
            hoje=HOJE)
        assert p == []

    def test_um_dia_de_folga_no_casamento_de_data(self):
        """Card de sabado a noite vira domingo em UTC, e card adiado em cima
        da hora muda de data sem mudar de identidade."""
        p = ew.eventos_perdidos(_hist(["2026-09-26"]),
                                _base([("2026-09-27", "UFC FN")]), hoje=HOJE)
        assert p == []

    def test_sem_historico_nao_inventa_alarme(self):
        assert ew.eventos_perdidos(pd.DataFrame(), _base([("2026-10-03", "X")]),
                                   hoje=HOJE) == []


class TestEventosARegistrar:
    def test_card_listado_pela_api_sem_pre_registro(self):
        r = ew.eventos_a_registrar(
            _api(["2026-10-10T23:00:00Z"] * 9), _hist(["2026-09-26"]), hoje=HOJE)
        assert len(r) == 1
        assert r[0] == {"event_date": "2026-10-10", "n_lutas": 9,
                        "dias": 4, "estado": "a registrar"}

    def test_card_ja_registrado_nao_alarma(self):
        r = ew.eventos_a_registrar(_api(["2026-10-10T23:00:00Z"] * 9),
                                   _hist(["2026-10-10"]), hoje=HOJE)
        assert r == []

    def test_card_principal_de_madrugada_UTC_conta_no_dia_americano(self):
        """Prelims 22:00Z do sabado e card principal 02:00Z do domingo sao o
        MESMO card. Pela data crua em UTC virariam dois alarmes, um deles num
        dia em que nao ha evento nenhum."""
        r = ew.eventos_a_registrar(
            _api(["2026-10-10T22:00:00Z"] * 6 + ["2026-10-11T02:00:00Z"] * 6),
            _hist(["2026-09-26"]), hoje=HOJE)
        assert len(r) == 1
        assert r[0]["event_date"] == "2026-10-10"
        assert r[0]["n_lutas"] == 12

    def test_luta_avulsa_nao_vira_card(self):
        """O feed e de MMA, nao de UFC: vem luta de outra organizacao e
        confronto especulativo de 2027. Em 06/out/2026 havia uma luta solta
        numa terca-feira, que alarmaria toda semana."""
        r = ew.eventos_a_registrar(_api(["2026-10-07T23:00:00Z"]),
                                   _hist(["2026-09-26"]), hoje=HOJE)
        assert r == []

    def test_card_longe_demais_ainda_nao_e_cobrado(self):
        r = ew.eventos_a_registrar(_api(["2026-12-05T23:00:00Z"] * 9),
                                   _hist(["2026-09-26"]), hoje=HOJE)
        assert r == []

    def test_commence_time_ilegivel_nao_quebra(self):
        r = ew.eventos_a_registrar([{"commence_time": "ontem"}, {}],
                                   _hist(["2026-09-26"]), hoje=HOJE)
        assert r == []


class TestCache:
    """O painel nao faz rede: a metade futura do alarme chega por aqui."""

    @pytest.fixture
    def cache(self, tmp_path):
        return tmp_path / "eventos_a_registrar.json"

    def test_ida_e_volta(self, cache):
        pend = [{"event_date": "2026-10-10", "n_lutas": 9, "dias": 4,
                 "estado": "a registrar"}]
        ew.salvar_pendentes(pend, hoje=HOJE, caminho=cache)
        assert ew.carregar_pendentes(hoje=HOJE, caminho=cache) == pend

    def test_cache_velho_nao_vale(self, cache):
        """Alarme de 'card amanha' escrito ha uma semana nao e informacao."""
        ew.salvar_pendentes([{"event_date": "2026-10-10"}],
                            hoje=HOJE - timedelta(days=7), caminho=cache)
        assert ew.carregar_pendentes(hoje=HOJE, caminho=cache) == []

    def test_evento_que_ja_passou_sai_do_cache(self, cache):
        """Quem cobra card passado e eventos_perdidos, com a base na mao --
        aqui ele viraria alarme eterno."""
        ew.salvar_pendentes([{"event_date": "2026-10-05"}], hoje=HOJE, caminho=cache)
        assert ew.carregar_pendentes(hoje=HOJE, caminho=cache) == []

    def test_cache_ausente_ou_corrompido_nao_inventa(self, cache):
        assert ew.carregar_pendentes(hoje=HOJE, caminho=cache) == []
        cache.write_text("{nao e json", encoding="utf-8")
        assert ew.carregar_pendentes(hoje=HOJE, caminho=cache) == []

    def test_lista_vazia_e_gravada_com_carimbo(self, cache):
        """'Verifiquei e nao ha pendencia' e diferente de 'nao verifiquei'."""
        ew.salvar_pendentes([], hoje=HOJE, caminho=cache)
        assert json.loads(cache.read_text(encoding="utf-8"))["verificado_em"] == HOJE.isoformat()
        assert not ew.precisa_verificar(hoje=HOJE, caminho=cache)


class TestComportaDeCusto:
    """O auto_capture roda de hora em hora. Uma chamada por rodada seriam
    ~720 creditos/mes numa chave gratuita de 500."""

    @pytest.fixture
    def cache(self, tmp_path):
        return tmp_path / "eventos_a_registrar.json"

    def test_sem_cache_verifica(self, cache):
        assert ew.precisa_verificar(hoje=HOJE, caminho=cache)

    def test_ja_verificado_hoje_nao_repete(self, cache):
        ew.salvar_pendentes([], hoje=HOJE, caminho=cache)
        assert not ew.precisa_verificar(hoje=HOJE, caminho=cache)

    def test_verificacao_de_ontem_ja_venceu(self, cache):
        ew.salvar_pendentes([], hoje=HOJE - timedelta(days=1), caminho=cache)
        assert ew.precisa_verificar(hoje=HOJE, caminho=cache)

    def test_cache_corrompido_verifica_de_novo(self, cache):
        cache.write_text("{", encoding="utf-8")
        assert ew.precisa_verificar(hoje=HOJE, caminho=cache)
