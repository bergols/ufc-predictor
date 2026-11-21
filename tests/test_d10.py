"""
CHAOS D10: janela do casamento entre a data nominal (local) do card e o
`commence_time` UTC das lutas. Derivacao em docs/chaos/D10.md: a diferenca
(data UTC da luta - event_date) fica em [-1, +1] para fusos UTC-12..UTC+14.

A janela so LOCALIZA candidatos; identidade continua vindo dos participantes
e da unicidade. Os testes de captura passam pelo caminho real
(scripts.auto_capture -> casar -> fetch_sharp_probs -> record_closing_lines)
com API e relogio simulados; nada toca producao.
"""
import logging
from datetime import datetime as _dt, timezone

import pandas as pd
import pytest

from src import casamento as cs
from src import card_report
# fixtures/auxiliares do harness da Fase C (testes/ nao e pacote: import por modulo)
from test_fase_c import mundo, fuso, _evento  # noqa: F401


def _ev(id_, quando, a, b):
    return {"id": id_, "commence_time": quando, "home_team": a, "away_team": b,
            "bookmakers": [{"title": "Pinnacle", "markets": [
                {"key": "h2h", "outcomes": [{"name": a, "price": 1.8},
                                            {"name": b, "price": 2.1}]}]}]}


def _casar(events, event_date, a="Alice Silva", b="Bia Costa"):
    return cs.casar(a, b, events, event_date=event_date)


# ------------------------------------------------------------- janela

class TestJanela:
    @pytest.mark.parametrize("quando", ["2026-11-28T23:00:00Z", "2026-11-29T02:00:00Z",
                                        "2026-11-29T06:00:00Z"])
    def test_asia_australia_domingo_as_tres_casam(self, quando):
        c = _casar([_ev("e1", quando, "Alice Silva", "Bia Costa")], "2026-11-29")
        assert c.ok, c.motivo

    @pytest.mark.parametrize("quando", ["2026-11-28T22:00:00Z", "2026-11-29T05:00:00Z"])
    def test_eua_sabado_prelim_e_principal_casam(self, quando):
        assert _casar([_ev("e1", quando, "Alice Silva", "Bia Costa")], "2026-11-28").ok

    @pytest.mark.parametrize("data,quando", [
        ("2026-12-01", "2026-11-30T23:00:00Z"),     # virada de mes
        ("2027-01-01", "2026-12-31T22:00:00Z"),     # virada de ano (local a frente)
        ("2026-12-31", "2027-01-01T04:00:00Z"),     # virada de ano (local atras)
        ("2026-03-01", "2026-02-28T20:00:00Z"),     # virada de fevereiro
    ])
    def test_viradas_de_dia_mes_ano(self, data, quando):
        assert _casar([_ev("e1", quando, "Alice Silva", "Bia Costa")], data).ok

    @pytest.mark.parametrize("quando,data", [
        ("2026-11-29T08:00:00+09:00", "2026-11-29"),   # = 28/11 23:00Z
        ("2026-11-28T21:00:00-05:00", "2026-11-28"),   # = 29/11 02:00Z
        ("2026-11-30T05:00:00+14:00", "2026-11-30"),   # = 29/11 15:00Z
    ])
    def test_offset_explicito_e_normalizado_para_utc(self, quando, data):
        assert _casar([_ev("e1", quando, "Alice Silva", "Bia Costa")], data).ok

    def test_offset_e_convertido_antes_de_comparar_datas(self):
        # 2026-12-01T05:00+14:00 = 30/11 15:00Z: 2 dias de 28/11 -> fora (a data
        # local da fonte, 01/12, nao e a que se compara)
        c = _casar([_ev("e1", "2026-12-01T05:00:00+14:00", "Alice Silva", "Bia Costa")],
                   "2026-11-28")
        assert not c.ok and c.motivo == cs.FORA_DA_JANELA

    @pytest.mark.parametrize("quando", ["2026-11-26T23:59:00Z", "2026-12-01T00:00:00Z"])
    def test_dois_ou_mais_dias_fora(self, quando):
        c = _casar([_ev("e1", quando, "Alice Silva", "Bia Costa")], "2026-11-28")
        assert not c.ok and c.motivo == cs.FORA_DA_JANELA

    @pytest.mark.parametrize("quando", ["2026-11-29T05:00:00", "2026-11-29", "???", "", None])
    def test_sem_fuso_ou_ilegivel_nao_casa_com_motivo(self, quando):
        c = _casar([_ev("e1", quando, "Alice Silva", "Bia Costa")], "2026-11-29")
        assert not c.ok and c.motivo == cs.INICIO_ILEGIVEL

    def test_sem_event_date_continua_bloqueando(self):
        c = _casar([_ev("e1", "2026-11-29T02:00:00Z", "Alice Silva", "Bia Costa")], None)
        assert not c.ok and c.motivo == cs.FORA_DA_JANELA


# ------------------------------------------------------------- identidade

class TestIdentidadePreservada:
    def test_revanche_semanas_depois_ou_antes_so_a_da_janela_casa(self):
        evs = [_ev("antes", "2026-10-31T23:00:00Z", "Alice Silva", "Bia Costa"),
               _ev("alvo", "2026-11-28T23:00:00Z", "Alice Silva", "Bia Costa"),
               _ev("depois", "2026-12-19T23:00:00Z", "Alice Silva", "Bia Costa")]
        c = _casar(evs, "2026-11-29")
        assert c.ok and c.evento_id == "alvo"

    def test_mesmo_par_em_dois_eventos_dentro_da_janela_e_multiplos(self):
        evs = [_ev("e1", "2026-11-28T23:00:00Z", "Alice Silva", "Bia Costa"),
               _ev("e2", "2026-11-29T05:00:00Z", "Bia Costa", "Alice Silva")]
        c = _casar(evs, "2026-11-29")
        assert not c.ok and c.motivo == cs.MULTIPLOS and c.candidatos == 2

    def test_par_valido_mais_gemeo_sem_fuso_nao_escolhe(self):
        evs = [_ev("e1", "2026-11-28T23:00:00Z", "Alice Silva", "Bia Costa"),
               _ev("e2", "2026-11-29T05:00:00", "Alice Silva", "Bia Costa")]
        c = _casar(evs, "2026-11-29")
        assert not c.ok and c.motivo == cs.MULTIPLOS

    def test_adversario_diferente_nao_casa(self):
        c = _casar([_ev("e1", "2026-11-28T23:00:00Z", "Alice Silva", "Carla Dias")],
                   "2026-11-29")
        assert not c.ok and c.motivo == cs.AUSENTE

    def test_homonimo_continua_bloqueado_na_janela_nova(self):
        c = cs.casar("Jean Silva", "Bia Costa",
                     [_ev("e1", "2026-11-28T23:00:00Z", "Jean Silva", "Bia Costa")],
                     event_date="2026-11-29")
        assert not c.ok and c.motivo == cs.HOMONIMO

    def test_so_aproximado_continua_bloqueado(self):
        c = cs.casar("Alice Silva", "Bia Costa",
                     [_ev("e1", "2026-11-28T23:00:00Z", "Alise Silva", "Bia Costa")],
                     event_date="2026-11-29")
        assert not c.ok and c.motivo == cs.SO_APROXIMADO


# ------------------------------------------------------------- caminho real

ASIA = [("Ana Alfa", "Bia Beta"), ("Caio Gama", "Davi Delta"), ("Eli Echo", "Fabi Foxtrot")]
INICIOS = [_dt(2026, 11, 28, 23, 0, tzinfo=timezone.utc),
           _dt(2026, 11, 29, 2, 0, tzinfo=timezone.utc),
           _dt(2026, 11, 29, 6, 0, tzinfo=timezone.utc)]


def _preditor_asia(a, b):
    return ({"fighter_a": a, "fighter_b": b, "prob_a_wins": 0.60, "prob_b_wins": 0.40,
             "model_used": "fake", "fighter_a_low_experience": False,
             "fighter_b_low_experience": False, "fighter_a_debutant": False,
             "fighter_b_debutant": False})


def _preparar_asia(mundo, monkeypatch):
    monkeypatch.setattr(card_report, "_default_predict_fns",
                        lambda n: (_preditor_asia, lambda a, b: {"method_probs": {
                            "KO_TKO": 0.4, "SUBMISSION": 0.2, "DECISION": 0.4}}))
    mundo.card_csv(["fighter_a,fighter_b,odds_a_decimal,odds_b_decimal"]
                   + [f"{a},{b},1.80,2.10" for a, b in ASIA])
    mundo.publicar(sharp=False, data="2026-11-29")


def _api(mundo, passo):
    # o preco da Pinnacle muda a cada rodada: prova QUAL rodada gravou
    mundo.api([_evento(a, b, 1.70 + 0.1 * passo, 2.25, ini)
               for (a, b), ini in zip(ASIA, INICIOS)])


class TestCapturaAsia:
    def test_grava_as_tres_ate_o_inicio_de_cada_uma_e_nada_depois(
            self, mundo, fuso, monkeypatch, caplog):
        _preparar_asia(mundo, monkeypatch)
        caplog.set_level(logging.INFO)

        _api(mundo, 1)                       # sab 22:00Z: as tres abertas
        mundo.capturar(_dt(2026, 11, 28, 22, 0, tzinfo=timezone.utc))
        r1 = mundo.linhas()["close_prob"].copy()
        assert r1.notna().all(), caplog.text

        _api(mundo, 2)                       # dom 01:00Z: a de sabado ja comecou
        mundo.capturar(_dt(2026, 11, 29, 1, 0, tzinfo=timezone.utc))
        r2 = mundo.linhas()["close_prob"].copy()
        assert r2["Ana Alfa"] == r1["Ana Alfa"]           # congelada no inicio
        assert r2["Caio Gama"] != r1["Caio Gama"] and r2["Eli Echo"] != r1["Eli Echo"]

        _api(mundo, 3)                       # dom 05:00Z: so a das 06:00Z
        mundo.capturar(_dt(2026, 11, 29, 5, 0, tzinfo=timezone.utc))
        r3 = mundo.linhas()["close_prob"].copy()
        assert r3["Ana Alfa"] == r1["Ana Alfa"]
        assert r3["Caio Gama"] == r2["Caio Gama"]
        assert r3["Eli Echo"] != r2["Eli Echo"]

        _api(mundo, 4)                       # dom 06:30Z: nada mais abre
        mundo.capturar(_dt(2026, 11, 29, 6, 30, tzinfo=timezone.utc))
        assert (mundo.linhas()["close_prob"] == r3).all()
        assert "SEM FECHAMENTO" not in caplog.text
        assert "confronto ausente" not in caplog.text

    def test_inicio_sem_fuso_da_luta_de_sabado_nao_grava(self, mundo, fuso, monkeypatch):
        _preparar_asia(mundo, monkeypatch)
        _api(mundo, 1)
        mundo.resposta[0]["commence_time"] = "2026-11-28T23:00:00"
        mundo.capturar(_dt(2026, 11, 28, 22, 0, tzinfo=timezone.utc))
        h = mundo.linhas()
        assert pd.isna(h.loc["Ana Alfa", "close_prob"])
        assert pd.notna(h.loc["Caio Gama", "close_prob"])


class TestCapturaEUA:
    def test_prelims_sabado_principal_domingo_event_date_sabado(self, mundo, fuso):
        mundo.publicar(sharp=False)                     # EVENTO = 2026-11-28
        from test_fase_c import PRELIM, MAIN
        mundo.api([_evento("Ana Alfa", "Bia Beta", 1.70, 2.25, MAIN),
                   _evento("Caio Gama", "Davi Delta", 1.45, 2.95, PRELIM)])
        mundo.capturar(_dt(2026, 11, 28, 20, 0, tzinfo=timezone.utc))
        assert mundo.linhas()["close_prob"].notna().all()
