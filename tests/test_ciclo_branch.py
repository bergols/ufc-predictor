import sys
from types import SimpleNamespace

import pytest

from scripts import ciclo_semanal
from src import data_collection


@pytest.mark.parametrize(("branch", "returncode"), [
    ("pesquisa-vantagem", 0), ("fecha-evento-2026-11-07", 0), ("", 0), ("", 1),
])
def test_ciclo_recusa_branch_incorreta_antes_de_qualquer_acesso(
        monkeypatch, branch, returncode):
    monkeypatch.setattr(sys, "argv", ["ciclo_semanal", "--no-push"])
    monkeypatch.setattr(ciclo_semanal, "_git", lambda *args:
                        SimpleNamespace(stdout=branch, returncode=returncode))

    def nao_acessar():
        raise AssertionError("Ciclo acessou dados antes da trava de branch")

    monkeypatch.setattr(data_collection, "download_github_mirror_dataset", nao_acessar)
    monkeypatch.setattr(data_collection, "check_data_freshness", nao_acessar)
    assert ciclo_semanal.main() == 1


@pytest.mark.parametrize(("args", "branch"), [([], "main"), (["--dry-run"], "pesquisa-vantagem")])
def test_main_e_inspecao_passam_pela_trava(monkeypatch, args, branch):
    monkeypatch.setattr(sys, "argv", ["ciclo_semanal", *args])
    monkeypatch.setattr(ciclo_semanal, "_git", lambda *a:
                        SimpleNamespace(stdout=branch, returncode=0))

    class AntesDeAcessarDados(Exception):
        pass

    def parar():
        raise AntesDeAcessarDados

    monkeypatch.setattr(data_collection, "download_github_mirror_dataset", parar)
    monkeypatch.setattr(data_collection, "check_data_freshness", parar)
    with pytest.raises(AntesDeAcessarDados):
        ciclo_semanal.main()
