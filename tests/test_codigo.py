"""
Ligacao entre o candidato e o codigo que decide (src/codigo.py) e emenda
obrigatoria (EMENDA_V2_MANIFESTO.md, regras 4 e 6).

Os casos de codigo rodam num repositorio git TEMPORARIO, nunca no do projeto.
"""
import subprocess

import pytest

import config
from src import candidatos, codigo, sombra


def _git(raiz, *args):
    subprocess.run(["git", "-C", str(raiz), "-c", "user.name=t", "-c",
                    "user.email=t@t", "-c", "core.autocrlf=false", *args],
                   check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "proj"
    for rel, txt in {"src/__init__.py": "", "src/decide.py": "W = 0.0\n",
                     "scripts/roda.py": "print('x')\n", "config.py": "A = 1\n",
                     "data/h.csv": "a,b\n", "tests/test_x.py": "", "README.md": "x\n"}.items():
        p = r / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8", newline="\n")
    _git(r, "init", "-q")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "inicio")
    return r


def _cand(repo):
    commit, impressao = codigo.registrar(repo)
    return {"codigo_commit": commit, "codigo_sha256": impressao}


def _commit(repo, msg="mudanca"):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)


class TestCodigo:
    def test_caminho_valido(self, repo):
        assert codigo.conferir(_cand(repo), repo) is None

    def test_commit_posterior_so_de_dados_nao_invalida(self, repo):
        c = _cand(repo)
        (repo / "data" / "h.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        (repo / "data" / "candidatos.json").write_text("[]", encoding="utf-8")
        (repo / "README.md").write_text("y\n", encoding="utf-8")
        (repo / "tests" / "test_x.py").write_text("# fora do alcance\n", encoding="utf-8")
        _commit(repo, "so dados")
        assert codigo.conferir(c, repo) is None

    def test_codigo_relevante_alterado_em_commit_posterior_bloqueia(self, repo):
        c = _cand(repo)
        (repo / "src" / "decide.py").write_text("W = 0.5\n", encoding="utf-8")
        _commit(repo)
        assert "difere do congelado" in codigo.conferir(c, repo)

    @pytest.mark.parametrize("rel", ["src/decide.py", "scripts/roda.py", "config.py"])
    def test_alteracao_local_relevante_bloqueia(self, repo, rel):
        c = _cand(repo)
        (repo / rel).write_text("# alterado localmente\n", encoding="utf-8")
        assert "difere do congelado" in codigo.conferir(c, repo)
        with pytest.raises(codigo.NaoVerificavel, match="alterado fora do commit"):
            codigo.registrar(repo)

    def test_arquivo_novo_nao_rastreado_bloqueia(self, repo):
        c = _cand(repo)
        (repo / "src" / "novo.py").write_text("X = 1\n", encoding="utf-8")
        assert "difere do congelado" in codigo.conferir(c, repo)
        with pytest.raises(codigo.NaoVerificavel):
            codigo.registrar(repo)

    def test_arquivo_removido_localmente_bloqueia(self, repo):
        c = _cand(repo)
        (repo / "scripts" / "roda.py").unlink()
        assert "difere do congelado" in codigo.conferir(c, repo)

    def test_so_fim_de_linha_nao_muda_o_codigo(self, repo):
        c = _cand(repo)
        (repo / "src" / "decide.py").write_bytes(b"W = 0.0\r\n")
        assert codigo.conferir(c, repo) is None

    def test_commit_inexistente_bloqueia_por_nao_verificavel(self, repo):
        c = {**_cand(repo), "codigo_commit": "0" * 40}
        assert "nao foi possivel verificar" in codigo.conferir(c, repo)

    def test_git_indisponivel_bloqueia(self, repo, monkeypatch):
        c = _cand(repo)

        def sem_git(*a, **k):
            raise OSError("git nao encontrado")

        monkeypatch.setattr(codigo.subprocess, "run", sem_git)
        assert "git indisponivel" in codigo.conferir(c, repo)

    def test_fora_de_um_repositorio_bloqueia(self, repo, tmp_path):
        c = _cand(repo)
        fora = tmp_path / "copia"
        for p in repo.rglob("*"):
            if ".git" not in p.parts and p.is_file():
                d = fora / p.relative_to(repo)
                d.parent.mkdir(parents=True, exist_ok=True)
                d.write_bytes(p.read_bytes())
        assert "nao foi possivel verificar" in codigo.conferir(c, fora)

    def test_impressao_registrada_que_nao_e_a_do_commit_bloqueia(self, repo):
        c = {**_cand(repo), "codigo_sha256": "f" * 64}
        assert "nao corresponde a impressao registrada" in codigo.conferir(c, repo)

    @pytest.mark.parametrize("faltando", ["codigo_commit", "codigo_sha256"])
    def test_candidato_sem_registro_de_codigo_bloqueia(self, repo, faltando):
        c = _cand(repo)
        c.pop(faltando)
        assert "nao registra o codigo congelado" in codigo.conferir(c, repo)

    def test_a_qualificacao_aplica_a_conferencia(self, monkeypatch):
        """sombra.qualificar chama codigo.conferir para candidato com manifesto."""
        vistos = []
        monkeypatch.setattr(codigo, "conferir",
                            lambda c, raiz=None: vistos.append(c) or "CODIGO X")
        cand = {"candidato_id": "x", "manifesto_modelo_sha256": "m",
                "emendas": [], "congelado_em": "2026-11-11T00:00:00+00:00"}
        _, motivos = sombra.qualificar(
            candidato=cand, decidido_em="2026-11-11T01:00:00+00:00",
            inicio="2026-11-11T05:00:00+00:00", ev_previsao=_Ev(), ev_mediana=_Ev(),
            ev_preco_lado=_Ev(), event_date="2099-01-01", queimados=set(),
            manifesto_previsao="m", fontes_do_confronto=("ambas", "ambas"))
        assert vistos and "CODIGO X" in motivos


class _Ev:
    classe = "comprovada"
    instante = "2026-11-11T00:30:00+00:00"
    detalhe = ""


class TestEmendaObrigatoria:
    def _ok(self):
        return {"arquivo": "EMENDA_V2_MANIFESTO.md",
                "sha256": candidatos.sha256_arquivo(
                    config.PROJECT_ROOT / "EMENDA_V2_MANIFESTO.md")}

    @pytest.mark.parametrize("emendas", [None, [], "nao-lista"])
    def test_lista_ausente_ou_vazia(self, emendas):
        c = {} if emendas is None else {"emendas": emendas}
        m = sombra._motivos_da_emenda(c)
        assert m and "lista de emendas ausente ou vazia" in m[0]

    def test_lista_sem_a_emenda_obrigatoria(self):
        outra = {"arquivo": "PROTOCOLO_VANTAGEM_V2.md",
                 "sha256": candidatos.sha256_arquivo(
                     config.PROJECT_ROOT / "PROTOCOLO_VANTAGEM_V2.md")}
        m = sombra._motivos_da_emenda({"emendas": [outra]})
        assert m == ["o candidato nao cita a emenda obrigatoria EMENDA_V2_MANIFESTO.md"]

    def test_hash_divergente(self):
        m = sombra._motivos_da_emenda({"emendas": [{**self._ok(), "sha256": "0" * 64}]})
        assert m == ["a emenda EMENDA_V2_MANIFESTO.md disponivel nao confere com "
                     "o hash do candidato"]

    def test_arquivo_inexistente(self):
        m = sombra._motivos_da_emenda(
            {"emendas": [self._ok(), {"arquivo": "EMENDA_FANTASMA.md", "sha256": "a" * 64}]})
        assert m == ["a emenda EMENDA_FANTASMA.md nao existe no projeto"]

    def test_referencia_incompleta(self):
        m = sombra._motivos_da_emenda({"emendas": [self._ok(), {"arquivo": "X.md"}]})
        assert m and m[0].startswith("referencia de emenda incompleta")

    def test_valida(self):
        assert sombra._motivos_da_emenda({"emendas": [self._ok()]}) == []

    @pytest.mark.parametrize("emendas", [None, [],
                                         [config.PROJECT_ROOT / "PROTOCOLO_VANTAGEM_V2.md"]])
    def test_congelar_com_manifesto_exige_a_emenda(self, tmp_path, emendas):
        with pytest.raises(ValueError, match="exige a emenda EMENDA_V2_MANIFESTO.md"):
            candidatos.congelar(versao_protocolo="vantagem-v2", peso_w=0.0,
                                arquivo=tmp_path / "c.json", manifesto_modelo_sha256="m",
                                emendas=emendas, codigo_commit="c", codigo_sha256="s")
        assert not (tmp_path / "c.json").exists()

    def test_congelar_com_emenda_inexistente(self, tmp_path):
        with pytest.raises(ValueError, match="Emenda indisponivel"):
            candidatos.congelar(
                versao_protocolo="vantagem-v2", peso_w=0.0, arquivo=tmp_path / "c.json",
                manifesto_modelo_sha256="m", codigo_commit="c", codigo_sha256="s",
                emendas=[tmp_path / "EMENDA_V2_MANIFESTO.md"])

    @pytest.mark.parametrize("faltando", ["codigo_commit", "codigo_sha256"])
    def test_congelar_com_manifesto_exige_o_codigo(self, tmp_path, faltando):
        kw = {"codigo_commit": "c", "codigo_sha256": "s"}
        kw[faltando] = ""
        with pytest.raises(ValueError, match="exige codigo_commit e codigo_sha256"):
            candidatos.congelar(
                versao_protocolo="vantagem-v2", peso_w=0.0, arquivo=tmp_path / "c.json",
                manifesto_modelo_sha256="m",
                emendas=[config.PROJECT_ROOT / "EMENDA_V2_MANIFESTO.md"], **kw)

    def test_cli_com_emenda_vazia_nao_congela(self, tmp_path, monkeypatch, caplog):
        from scripts import candidato as cli
        from src import decisoes
        monkeypatch.setattr(candidatos, "ARQUIVO", tmp_path / "c.json")
        monkeypatch.setattr(decisoes, "manifesto_sha256", lambda *a: "m" * 64)
        monkeypatch.setattr(codigo, "registrar", lambda *a: ("c" * 40, "s" * 64))
        assert cli.main(["--peso", "0.0", "--emenda"]) == 1
        assert "exige a emenda EMENDA_V2_MANIFESTO.md" in caplog.text
        assert not (tmp_path / "c.json").exists()

    def test_cli_com_codigo_alterado_nao_congela(self, tmp_path, monkeypatch, caplog):
        from scripts import candidato as cli
        from src import decisoes
        monkeypatch.setattr(candidatos, "ARQUIVO", tmp_path / "c.json")
        monkeypatch.setattr(decisoes, "manifesto_sha256", lambda *a: "m" * 64)

        def sujo(*a):
            raise codigo.NaoVerificavel("codigo alterado fora do commit: src/x.py")

        monkeypatch.setattr(codigo, "registrar", sujo)
        assert cli.main(["--peso", "0.0", "--mostrar"]) == 1
        assert "Codigo sem versao verificavel" in caplog.text
        assert not (tmp_path / "c.json").exists()

    def test_os_quatro_candidatos_antigos_seguem_sem_os_campos(self):
        antigos = candidatos.carregar()[:4]
        assert [c["candidato_id"] for c in antigos] == [
            "253046a9cee9958c", "1f4a6f4961a658bd", "ea79fb4b4887c34d",
            "0ae27ee8b716ab72"]
        for c in antigos:
            assert not ({"emendas", "codigo_commit", "codigo_sha256",
                         "manifesto_modelo_sha256"} & set(c))
