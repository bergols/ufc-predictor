"""
Ciclo operacional: Chromium só no UFCStats e --dry-run sem API de odds.

Em nov/2026 o Edge desta máquina passou a devolver ERR_BLOCKED_BY_CLIENT para
ufcstats.com. A opção `--chromium-ufcstats` troca o navegador SÓ das leituras
do UFCStats (card e nome do evento); a Thunderpick continua no Edge com
janela, porque o Cloudflare barra o modo headless.

E o `--dry-run` chegava a `fetch_live_odds` quando não havia card vigente nem
evento pendente: uma inspeção gastava crédito de API. Aqui ele para antes.

Nenhum teste abre navegador nem rede: o Playwright é trocado por um falso que
só anota os argumentos de `launch` e interrompe.
"""
import hashlib
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import config
from scripts import ciclo_semanal


class _Lancou(BaseException):
    """BaseException de propósito: atravessa os `except Exception` que
    tentam o Chromium quando o Edge falha, e para no primeiro launch."""


def _playwright_falso(monkeypatch, chamadas):
    class Chromium:
        def launch(self, **kw):
            chamadas.append(kw)
            raise _Lancou

    @contextmanager
    def sync_playwright():
        yield SimpleNamespace(chromium=Chromium())

    import playwright.sync_api
    monkeypatch.setattr(playwright.sync_api, "sync_playwright", sync_playwright)


# ---------------------------------------------------------------------------
# Navegador das leituras do UFCStats

@pytest.mark.parametrize(("kwargs", "esperado"), [
    ({}, {"channel": "msedge", "headless": True}),
    ({"chromium": False}, {"channel": "msedge", "headless": True}),
    ({"chromium": True}, {"headless": True}),
])
def test_navegador_real_escolhe_o_navegador(monkeypatch, kwargs, esperado):
    from src import data_collection
    chamadas = []
    _playwright_falso(monkeypatch, chamadas)
    with pytest.raises(_Lancou):
        with data_collection.navegador_real(**kwargs):
            pass
    assert chamadas == [esperado]


def test_chromium_ausente_da_diagnostico_claro(monkeypatch):
    from src import data_collection

    class Chromium:
        def launch(self, **kw):
            raise FileNotFoundError("Executable doesn't exist at chrome.exe")

    @contextmanager
    def sync_playwright():
        yield SimpleNamespace(chromium=Chromium())

    import playwright.sync_api
    monkeypatch.setattr(playwright.sync_api, "sync_playwright", sync_playwright)
    with pytest.raises(RuntimeError, match="Chromium do Playwright indisponivel"):
        with data_collection.navegador_real(chromium=True):
            pass


@pytest.mark.parametrize("chromium", [False, True])
def test_card_do_ufcstats_repassa_chromium(monkeypatch, chromium):
    from src import card_sources, data_collection
    vistos = []

    @contextmanager
    def navegador(**kw):
        vistos.append(kw)
        yield

    monkeypatch.setattr(data_collection, "navegador_real", navegador)
    monkeypatch.setattr(data_collection, "_get_soup", lambda url: None)
    with pytest.raises(RuntimeError):
        card_sources.card_do_ufcstats("2026-11-14", chromium=chromium)
    assert vistos == [{"chromium": chromium}]


def test_thunderpick_continua_no_edge_com_janela(monkeypatch):
    from src import thunderpick
    chamadas = []
    _playwright_falso(monkeypatch, chamadas)
    with pytest.raises(_Lancou):
        thunderpick.fetch_ufc_odds()
    assert chamadas == [{"channel": "msedge", "headless": False}]


# ---------------------------------------------------------------------------
# Propagação da opção: build_card e as duas leituras do ciclo

class _Parar(Exception):
    pass


@pytest.mark.parametrize(("argv", "esperado"), [
    ([], False), (["--chromium-ufcstats"], True),
])
def test_build_card_repassa_a_opcao(monkeypatch, argv, esperado):
    from scripts import build_card
    from src import card_sources, line_shopping
    vistos = []

    def card(event_date, **kw):
        vistos.append((event_date, kw))
        raise _Parar

    def odds(*a, **kw):
        raise AssertionError("build_card consultou odds antes de ler o card")

    monkeypatch.setattr(card_sources, "card_do_ufcstats", card)
    monkeypatch.setattr(line_shopping, "fetch_live_odds", odds)
    monkeypatch.setattr(sys, "argv", ["build_card", "--event-date", "2026-11-14", *argv])
    with pytest.raises(_Parar):
        build_card.main()
    assert vistos == [("2026-11-14", {"chromium": esperado})]


@pytest.mark.parametrize(("chromium", "cauda"), [
    (False, []), (True, ["--chromium-ufcstats"]),
])
def test_montar_card_repassa_a_opcao(monkeypatch, chromium, cauda):
    comandos = []
    monkeypatch.setattr(ciclo_semanal.subprocess, "run",
                        lambda cmd, **kw: comandos.append(cmd) or SimpleNamespace(returncode=0))
    assert ciclo_semanal._montar_card("2026-11-14", chromium=chromium) == 0
    assert comandos == [[sys.executable, "-m", "scripts.build_card",
                         "--event-date", "2026-11-14", *cauda]]


@pytest.mark.parametrize("chromium", [False, True])
def test_nome_do_card_repassa_a_opcao(monkeypatch, chromium):
    from src import card_sources
    vistos = []

    def card(event_date, **kw):
        vistos.append(kw)
        return {"nome": "UFC 334: Gane vs. Hokit"}

    monkeypatch.setattr(card_sources, "card_do_ufcstats", card)
    assert ciclo_semanal._nome_do_card("2026-11-14", chromium=chromium) == "UFC 334: Gane vs Hokit"
    assert vistos == [{"chromium": chromium}]


def _sem_base(monkeypatch):
    from src import data_collection
    for nome in ("download_github_mirror_dataset", "fill_recent_gap_with_browser",
                 "merge_manual_recent_fights", "check_data_freshness"):
        monkeypatch.setattr(data_collection, nome, lambda *a, **kw: None)
    monkeypatch.setattr(ciclo_semanal, "_eventos_para_fechar", lambda hoje: [])


@pytest.mark.parametrize(("argv", "esperado"), [
    ([], False), (["--chromium-ufcstats"], True),
])
def test_ciclo_leva_a_opcao_as_duas_leituras_do_ufcstats(monkeypatch, argv, esperado):
    """Registro do próximo card: build_card (subprocesso) e nome do evento."""
    _sem_base(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["ciclo_semanal", "--no-push", *argv])
    monkeypatch.setattr(ciclo_semanal, "_git",
                        lambda *a: SimpleNamespace(stdout="main", returncode=0))
    monkeypatch.setattr(ciclo_semanal, "_card_vigente", lambda hoje: None)
    monkeypatch.setattr(ciclo_semanal, "_proximo_a_registrar",
                        lambda hoje, **kw: {"event_date": "2026-11-14"})
    montou, nomeou = [], []
    monkeypatch.setattr(ciclo_semanal, "_montar_card",
                        lambda d, **kw: montou.append((d, kw)) or 0)

    def nome(d, **kw):
        nomeou.append((d, kw))
        raise _Parar  # para antes de anexar ao template ou publicar

    monkeypatch.setattr(ciclo_semanal, "_nome_do_card", nome)
    with pytest.raises(_Parar):
        ciclo_semanal.main()
    assert montou == [("2026-11-14", {"chromium": esperado})]
    assert nomeou == [("2026-11-14", {"chromium": esperado})]


@pytest.mark.parametrize(("argv", "esperado"), [
    ([], False), (["--chromium-ufcstats"], True),
])
def test_ciclo_leva_a_opcao_ao_trazer_o_card_vigente(monkeypatch, argv, esperado):
    _sem_base(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["ciclo_semanal", "--no-push", *argv])
    monkeypatch.setattr(ciclo_semanal, "_git",
                        lambda *a: SimpleNamespace(stdout="main", returncode=0))
    monkeypatch.setattr(ciclo_semanal, "_card_vigente",
                        lambda hoje: ("2026-11-14", "UFC 334"))
    monkeypatch.setattr(ciclo_semanal, "_csv_e_do_card", lambda d: False)

    def montar(d, **kw):
        assert kw == {"chromium": esperado}
        raise _Parar

    monkeypatch.setattr(ciclo_semanal, "_montar_card", montar)
    with pytest.raises(_Parar):
        ciclo_semanal.main()


def test_tarefa_agendada_roda_com_chromium():
    bat = (Path(config.PROJECT_ROOT) / "scripts" / "ciclo_semanal.bat").read_text(encoding="utf-8")
    linhas = [l for l in bat.splitlines() if "-m scripts.ciclo_semanal" in l]
    assert len(linhas) == 1
    assert "--chromium-ufcstats" in linhas[0].split()
    assert "--dry-run" not in linhas[0]


# ---------------------------------------------------------------------------
# --dry-run nunca consulta API de odds

def _odds_proibidas(monkeypatch):
    from src import line_shopping

    def proibido(*a, **kw):
        raise AssertionError("dry-run consultou API de odds")

    monkeypatch.setattr(line_shopping, "fetch_live_odds", proibido)
    monkeypatch.setattr(line_shopping, "get_api_key", proibido)


def _sem_vigente_nem_pendente(monkeypatch):
    from src import event_watch, prediction_history
    monkeypatch.setattr(ciclo_semanal, "_card_vigente", lambda hoje: None)
    monkeypatch.setattr(event_watch, "carregar_pendentes", lambda **kw: [])
    monkeypatch.setattr(prediction_history, "load_history", lambda *a, **kw: pd.DataFrame())


def test_proximo_a_registrar_em_dry_run_para_antes_da_api(monkeypatch, caplog):
    _odds_proibidas(monkeypatch)
    _sem_vigente_nem_pendente(monkeypatch)
    from datetime import date
    with caplog.at_level("INFO"):
        assert ciclo_semanal._proximo_a_registrar(date(2026, 11, 11), dry_run=True) is None
    assert "dependeria de captura nova" in caplog.text


def test_proximo_a_registrar_fora_do_dry_run_ainda_consulta(monkeypatch):
    """A correção não pode desligar o registro operacional."""
    from datetime import date

    from src import event_watch, line_shopping
    _sem_vigente_nem_pendente(monkeypatch)
    consultas = []
    monkeypatch.setattr(line_shopping, "get_api_key", lambda: "k")
    monkeypatch.setattr(line_shopping, "fetch_live_odds",
                        lambda k: consultas.append(k) or [])
    monkeypatch.setattr(event_watch, "eventos_a_registrar", lambda *a, **kw: [])
    assert ciclo_semanal._proximo_a_registrar(date(2026, 11, 11)) is None
    assert consultas == ["k"]


def _impressoes():
    caminhos = [
        config.PREDICTION_HISTORY_CSV, config.ODDS_TEMPLATE_CSV, config.UPCOMING_CARD_CSV,
        config.DATA_DIR / "previsoes_snapshots.csv", config.DATA_DIR / "decisoes.csv",
        config.DATA_DIR / "proveniencia" / "chamadas.csv",
        config.DATA_DIR / "proveniencia" / "cotacoes.csv",
        config.DATA_DIR / "ciclo_estado.json",
        Path(config.PROJECT_ROOT) / "docs" / "index.html",
    ]
    return {str(p): (hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None)
            for p in caminhos}


def test_dry_run_sem_card_e_sem_pendente_nao_toca_em_nada(monkeypatch, caplog):
    """O caminho que antes chegava a fetch_live_odds."""
    _sem_base(monkeypatch)
    _odds_proibidas(monkeypatch)
    _sem_vigente_nem_pendente(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["ciclo_semanal", "--dry-run", "--chromium-ufcstats"])

    def proibido(*a, **kw):
        raise AssertionError("dry-run tentou git, subprocesso ou treino")

    monkeypatch.setattr(ciclo_semanal, "_git", proibido)
    monkeypatch.setattr(ciclo_semanal.subprocess, "run", proibido)
    monkeypatch.setattr(ciclo_semanal, "_salvar_estado", proibido)
    monkeypatch.setattr(ciclo_semanal, "_anexar_ao_template", proibido)
    monkeypatch.setattr(ciclo_semanal, "_montar_card", proibido)
    from src import card_sources, train
    monkeypatch.setattr(card_sources, "card_do_ufcstats", proibido)
    monkeypatch.setattr(train, "train_and_calibrate", proibido)
    import scripts.publish_report
    monkeypatch.setattr(scripts.publish_report, "publish", proibido)

    antes = _impressoes()
    with caplog.at_level("INFO"):
        assert ciclo_semanal.main() == 0
    assert _impressoes() == antes
    assert "dependeria de captura nova" in caplog.text
    assert "Nenhum arquivo tocado" in caplog.text
