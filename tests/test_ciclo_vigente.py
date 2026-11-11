"""
O ciclo nao troca de card antes de o atual acontecer.

Em 29/out/2026 o ciclo publicou o card de 07/nov com o de 31/out ainda por
vir. Quem foi pegar as entradas da noite encontrou o card da semana seguinte.

Registrar cedo tambem custa QUALIDADE: aquele 07/nov entrou com sharp em 0 de
10 e odd da casa em 2 de 10, porque o mercado mal tinha aberto. "Perto da
abertura do mercado" nao e "antes do card anterior acontecer".
"""
from datetime import date

import pandas as pd
import pytest

import config
from scripts import ciclo_semanal as cs

HOJE = date(2026, 10, 31)


@pytest.fixture
def hist(tmp_path, monkeypatch):
    p = tmp_path / "prediction_history.csv"
    monkeypatch.setattr(config, "PREDICTION_HISTORY_CSV", p)
    return p


def _grava(p, linhas):
    pd.DataFrame([{"event_name": n, "event_date": d, "fighter_a": a, "fighter_b": b,
                   "odds_a_decimal": 1.5, "odds_b_decimal": 2.6, "model_name": "logreg",
                   "model_prob_a": 0.6, "model_side": a, "actual_winner": v}
                  for n, d, a, b, v in linhas]).to_csv(p, index=False)


class TestCardVigente:
    def test_evento_de_HOJE_ainda_e_o_vigente(self, hist):
        """O card e a noite. Ele so deixa de ser vigente amanha."""
        _grava(hist,
               [("UFC Hoje", "2026-10-31", "A", "B", None)])
        assert cs._card_vigente(HOJE) == ("2026-10-31", "UFC Hoje")

    def test_o_mais_proximo_vence_quando_ha_dois_futuros(self, hist):
        _grava(hist,
               [("UFC Semana Que Vem", "2026-11-07", "C", "D", None),
                ("UFC Hoje", "2026-10-31", "A", "B", None)])
        assert cs._card_vigente(HOJE)[0] == "2026-10-31"

    def test_tudo_passado_nao_tem_vigente(self, hist):
        _grava(hist,
               [("UFC Velho", "2026-10-24", "A", "B", "A")])
        assert cs._card_vigente(HOJE) is None

    def test_evento_passado_e_vigente_mesmo_ABERTO(self, hist):
        """Luta cancelada deixa linha aberta para sempre; isso nao pode fazer
        um card de julho virar 'vigente' e travar o ciclo."""
        _grava(hist,
               [("UFC Velho", "2026-07-25", "A", "B", None)])
        assert cs._card_vigente(HOJE) is None

    def test_historico_vazio(self, tmp_path, monkeypatch):
        p = tmp_path / "vazio.csv"
        p.write_text("event_name,event_date,fighter_a,fighter_b,odds_a_decimal,"
                     "odds_b_decimal,model_name,model_prob_a,model_side,actual_winner\n",
                     encoding="utf-8")
        monkeypatch.setattr(config, "PREDICTION_HISTORY_CSV", p)
        assert cs._card_vigente(HOJE) is None


class TestNaoRegistraAntesDaHora:
    def test_com_card_por_acontecer_NAO_registra_outro(self, hist, caplog):
        """A regra inteira numa linha."""
        _grava(hist, [("UFC Hoje", "2026-10-31", "A", "B", None)])
        with caplog.at_level("INFO"):
            assert cs._proximo_a_registrar(HOJE) is None
        assert "nao registro outro antes dele acontecer" in caplog.text

    def test_sem_card_por_acontecer_ele_PODE_registrar(self, hist, tmp_path, monkeypatch):
        """Sem evento futuro, volta a procurar -- a trava e so para o vigente,
        nao um freio permanente."""
        _grava(hist, [("UFC Velho", "2026-10-24", "A", "B", "A")])
        from src import event_watch
        monkeypatch.setattr(event_watch, "CACHE_PENDENTES", tmp_path / "cache.json")
        event_watch.salvar_pendentes(
            [{"event_date": "2026-11-07", "n_lutas": 11, "dias": 7}],
            hoje=HOJE, caminho=tmp_path / "cache.json")
        achado = cs._proximo_a_registrar(HOJE)
        assert achado and achado["event_date"] == "2026-11-07"


@pytest.mark.parametrize("chromium", [False, True])
def test_ciclo_encaminha_a_opcao_de_navegador_ao_build(monkeypatch, chromium):
    from types import SimpleNamespace

    comandos = []

    def executar(comando, **kwargs):
        comandos.append(comando)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cs.subprocess, "run", executar)
    assert cs._montar_card("2026-11-14", chromium=chromium) == 0
    assert ("--chromium-ufcstats" in comandos[0]) == chromium


def test_nome_do_card_usa_o_mesmo_navegador(monkeypatch):
    from src import card_sources

    recebidos = []

    def buscar(data, *, chromium=False):
        recebidos.append((data, chromium))
        return {"nome": "UFC: Alpha vs. Beta"}

    monkeypatch.setattr(card_sources, "card_do_ufcstats", buscar)
    assert cs._nome_do_card("2026-11-14", chromium=True) == "UFC: Alpha vs Beta"
    assert recebidos == [("2026-11-14", True)]
