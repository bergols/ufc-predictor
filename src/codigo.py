"""
src/codigo.py

Liga um candidato ao CODIGO que decide (EMENDA_V2_MANIFESTO.md, regra 4).

Alcance da conferencia -- exatamente isto, e nada mais:

    todo arquivo `*.py` sob `src/` e `scripts/`, e `config.py`, na raiz do
    projeto.

Fica de fora: dados (`data/`), modelos (`models/`), documentos, testes e
pacotes instalados. A versao do modelo ja e conferida pelo manifesto, e a do
protocolo e das emendas pelo hash de cada arquivo.

Impressao digital: sha256 sobre a lista ORDENADA de `caminho\\0sha256(conteudo)`,
com CRLF normalizado para LF (o checkout no Windows troca o fim de linha, e
isso nao muda o codigo).

Na hora de congelar, grava-se o commit e a impressao. Na hora de decidir, a
impressao do disco e a do commit gravado precisam ser IGUAIS a guardada:

- commit posterior so de dados nao muda a impressao, entao nao invalida;
- arquivo de codigo alterado, criado ou removido localmente muda a do disco;
- codigo alterado em commit posterior tambem muda a do disco;
- git indisponivel, commit inexistente ou arquivo ilegivel: nao da para
  verificar, e isso BLOQUEIA.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import config

PASTAS = ("src", "scripts")
ARQUIVOS = ("config.py",)


class NaoVerificavel(RuntimeError):
    pass


def _relevante(caminho: str) -> bool:
    c = caminho.replace("\\", "/")
    if c in ARQUIVOS:
        return True
    return c.endswith(".py") and c.split("/", 1)[0] in PASTAS and "__pycache__" not in c


def _impressao(itens: dict[str, bytes]) -> str:
    h = hashlib.sha256()
    for caminho in sorted(itens):
        corpo = itens[caminho].replace(b"\r\n", b"\n")
        h.update(caminho.encode("utf-8") + b"\0"
                 + hashlib.sha256(corpo).hexdigest().encode("ascii") + b"\n")
    return h.hexdigest()


def impressao_disco(raiz: Path | None = None) -> str:
    raiz = Path(raiz or config.PROJECT_ROOT)
    itens = {}
    try:
        for pasta in PASTAS:
            for p in (raiz / pasta).rglob("*.py"):
                rel = p.relative_to(raiz).as_posix()
                if _relevante(rel):
                    itens[rel] = p.read_bytes()
        for nome in ARQUIVOS:
            itens[nome] = (raiz / nome).read_bytes()
    except OSError as exc:
        raise NaoVerificavel(f"arquivo de codigo ilegivel ({exc})") from exc
    return _impressao(itens)


def _git(raiz: Path, *args) -> bytes:
    try:
        r = subprocess.run(["git", "-C", str(raiz), *args], capture_output=True)
    except OSError as exc:
        raise NaoVerificavel(f"git indisponivel ({exc})") from exc
    if r.returncode != 0:
        raise NaoVerificavel(
            f"git {' '.join(args[:2])} falhou: "
            f"{r.stderr.decode('utf-8', 'replace').strip()[:120]}")
    return r.stdout


def impressao_commit(commit: str, raiz: Path | None = None) -> str:
    raiz = Path(raiz or config.PROJECT_ROOT)
    if not commit:
        raise NaoVerificavel("sem commit registrado")
    nomes = _git(raiz, "ls-tree", "-r", "--name-only", commit, "--",
                 *PASTAS, *ARQUIVOS).decode("utf-8").splitlines()
    itens = {n: _git(raiz, "show", f"{commit}:{n}") for n in nomes if _relevante(n)}
    if not all(n in itens for n in ARQUIVOS):
        raise NaoVerificavel(f"o commit {commit[:12]} nao tem {', '.join(ARQUIVOS)}")
    return _impressao(itens)


def alteracoes_locais(raiz: Path | None = None) -> list[str]:
    """Caminhos de codigo alterados, criados ou removidos fora do HEAD."""
    raiz = Path(raiz or config.PROJECT_ROOT)
    saida = _git(raiz, "status", "--porcelain", "--untracked-files=all", "--",
                 *PASTAS, *ARQUIVOS).decode("utf-8")
    sujos = []
    for linha in saida.splitlines():
        for parte in linha[3:].split(" -> "):
            parte = parte.strip().strip('"')
            if parte and _relevante(parte):
                sujos.append(parte)
    return sujos


def registrar(raiz: Path | None = None) -> tuple[str, str]:
    """(commit, impressao) para um congelamento. Levanta se nao for limpo."""
    raiz = Path(raiz or config.PROJECT_ROOT)
    sujos = alteracoes_locais(raiz)
    if sujos:
        raise NaoVerificavel("codigo alterado fora do commit: " + ", ".join(sujos[:5]))
    commit = _git(raiz, "rev-parse", "HEAD").decode("ascii").strip()
    disco, no_commit = impressao_disco(raiz), impressao_commit(commit, raiz)
    if disco != no_commit:
        raise NaoVerificavel("o codigo em disco difere do commit")
    return commit, disco


def conferir(candidato: dict, raiz: Path | None = None) -> str | None:
    """Motivo do bloqueio, ou None quando o codigo da decisao e o congelado."""
    commit = str(candidato.get("codigo_commit") or "")
    guardada = str(candidato.get("codigo_sha256") or "")
    if not commit or not guardada:
        return ("o candidato nao registra o codigo congelado "
                "(codigo_commit e codigo_sha256)")
    try:
        do_commit = impressao_commit(commit, raiz)
        do_disco = impressao_disco(raiz)
    except NaoVerificavel as exc:
        return f"nao foi possivel verificar o codigo da decisao: {exc}"
    if do_commit != guardada:
        return (f"o commit {commit[:12]} nao corresponde a impressao registrada "
                f"do codigo")
    if do_disco != guardada:
        return ("o codigo que decide difere do congelado no candidato "
                f"(commit {commit[:12]}; alcance: src/, scripts/, config.py)")
    return None
