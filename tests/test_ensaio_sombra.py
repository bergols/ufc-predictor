from types import SimpleNamespace

import pytest

import config
from scripts import ensaio_sombra


def test_copia_com_git_recusa_antes_da_api(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".git").mkdir()
    with pytest.raises(SystemExit):
        ensaio_sombra.main(["--event-date", "2026-11-14", "--card-name", "Teste"])


@pytest.mark.parametrize("falha", [False, True])
def test_ensaio_roda_as_etapas_em_ordem_e_para_ao_falhar(tmp_path, monkeypatch, falha):
    import joblib
    from src import features, line_shopping

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    for nome in ("RAW_FIGHTS_CSV", "RAW_FIGHT_STATS_CSV", "RAW_FIGHTERS_CSV"):
        caminho = tmp_path / nome
        caminho.touch()
        monkeypatch.setattr(config, nome, caminho)
    config.MODELS_DIR.mkdir()
    (config.MODELS_DIR / f"{config.MODELO_PRODUCAO}_calibrated.joblib").touch()
    monkeypatch.setattr(joblib, "load", lambda _: None)
    monkeypatch.setattr(features, "export_latest_fighter_levels", lambda: None)
    monkeypatch.setattr(line_shopping, "get_api_key", lambda: "ficticia")
    comandos = []

    def rodar(cmd, **kwargs):
        comandos.append(cmd)
        return SimpleNamespace(returncode=1 if falha else 0)

    monkeypatch.setattr(ensaio_sombra.subprocess, "run", rodar)
    retorno = ensaio_sombra.main(["--event-date", "2026-11-14", "--card-name", "Teste"])
    assert retorno == int(falha)
    assert [cmd[2] for cmd in comandos] == (
        ["scripts.build_card"] if falha else
        ["scripts.build_card", "src.card_report", "scripts.sombra"])
    if not falha:
        assert "--no-sharp" in comandos[1]
        assert "--dry-run" in comandos[2]
        card = comandos[0][comandos[0].index("--out") + 1]
        assert comandos[1][3] == card
        assert comandos[2][comandos[2].index("--card") + 1] == card


def test_chromium_explicito_nao_abre_edge(monkeypatch):
    import sys
    from contextlib import nullcontext

    from src import data_collection

    lancamentos = []
    browser = SimpleNamespace(new_page=lambda: object(), close=lambda: None)

    def abrir(**kw):
        lancamentos.append(kw)
        return browser

    playwright = SimpleNamespace(chromium=SimpleNamespace(launch=abrir))
    modulo = SimpleNamespace(sync_playwright=lambda: nullcontext(playwright))
    monkeypatch.setitem(sys.modules, "playwright.sync_api", modulo)
    with data_collection.navegador_real(chromium=True):
        pass
    assert lancamentos == [{"headless": True}]
