"""
Persistencia dos registros: trava entre processos, gravacao atomica e leitura
integra.

POR QUE EXISTE (CHAOS, rodada 1, achados D1, D2 e D9). Os registros do projeto
sao CSVs que varios processos leem, alteram e regravam (captura horaria,
publicacao do card, fechamento, reconciliacao). Tres defeitos reais:

  D1  `df.to_csv(path)` direto: uma queda no meio da escrita deixava o arquivo
      pela metade, a proxima execucao lia SEM ERRO e gravava por cima (81
      linhas perdidas);
  D2  ler -> alterar -> gravar sem trava: duas execucoes simultaneas
      perdiam a atualizacao uma da outra (ponteiro `prediction_id` revertido,
      fechamento apagado);
  D9  anexar (`open(..., "a")`) sem conferir que o arquivo termina em "\\n": a
      linha nova colava na parcial.

O QUE ESTE MODULO FAZ
  * `trava()`: UMA trava, reentrante no processo e exclusiva entre processos,
    feita com a trava do sistema operacional (`fcntl.flock` em POSIX,
    `msvcrt.locking` no Windows). O SO a solta se o processo morrer: nao existe
    trava orfa para limpar a mao. Uma trava so elimina deadlock por ordem.
  * `gravar_atomico()`: temporario no mesmo diretorio + fsync + `os.replace`.
    O arquivo ou esta como estava ou esta completo; nunca pela metade por
    culpa do nosso escritor.
  * `validar_csv()` / `ler_csv_integro()`: recusam arquivo que nao termina em
    "\\n" ou com linha cujo numero de campos difere do cabecalho, com a
    instrucao de recuperacao. Nada e reparado automaticamente.
  * `anexar_csv()`: sob a trava, valida o arquivo atual e o regrava inteiro
    (conteudo integro + linhas novas) de forma atomica.

ALCANCE REAL (ver docs/chaos/FASE_A.md): protege quem usa estas funcoes.
Editor externo, planilha aberta, `git checkout` ou copia manual nao passam por
aqui. NFS/SMB: `flock` pode nao excluir entre maquinas. Windows: o ramo
`msvcrt` esta escrito e testado com dublê, mas a validacao real e pendente.
Corte que cai exatamente numa fronteira de linha (arquivo termina em "\\n" e
cada linha tem os campos certos) e indistinguivel de um arquivo menor e NAO e
detectado.
"""
from __future__ import annotations

import csv
import io
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import config

NOME_TRAVA = ".registros.lock"

# Quanto esperar pela trava antes de desistir com erro claro. Uma gravacao
# dura milissegundos; 60 s so se esgota com processo travado de verdade.
TIMEOUT_S = float(os.environ.get("UFC_TRAVA_TIMEOUT_S", "60"))
_INTERVALO_S = 0.02


class RegistroIncompleto(RuntimeError):
    """Arquivo de registro truncado ou inconsistente. Nada foi gravado."""


class TravaOcupada(RuntimeError):
    """A trava dos registros nao ficou livre dentro do prazo."""


# ---------------------------------------------------------------------------
# Trava
# ---------------------------------------------------------------------------

def _raiz_da_trava(alvo) -> Path:
    """
    Diretorio cujo `.registros.lock` protege `alvo`.

    TODOS os registros reais (data/, data/proveniencia/, data/raw/) dividem a
    trava de `config.DATA_DIR`: assim nenhuma ordem de aquisicao entre eles
    pode dar deadlock. Fora de data/ (testes com tmp_path) a trava fica ao lado
    do arquivo, e nunca cria nada no data/ real.
    """
    base = Path(config.DATA_DIR).resolve()
    if alvo is None:
        return base
    p = Path(alvo).resolve()
    try:
        p.relative_to(base)
        return base
    except ValueError:
        return p.parent


class _Trava:
    def __init__(self, caminho: Path):
        self.caminho = caminho
        self.rlock = threading.RLock()
        self.nivel = 0
        self.fd: int | None = None


_TRAVAS: dict[str, _Trava] = {}
_TRAVAS_MUTEX = threading.Lock()


def _tentar_travar(fd: int) -> bool:
    """Uma tentativa sem bloquear. True se conseguiu."""
    if sys.platform == "win32":
        import msvcrt
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _destravar(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return
    import fcntl
    fcntl.flock(fd, fcntl.LOCK_UN)


def _dono(caminho: Path) -> str:
    try:
        t = caminho.read_text(encoding="utf-8").strip()
        return f"pid {t}" if t else "dono desconhecido"
    except OSError:
        return "dono desconhecido"


@contextmanager
def trava(alvo=None, *, timeout: float | None = None):
    """
    Trava exclusiva dos registros. Reentrante: a mesma thread pode entrar de
    novo (uma funcao sob trava chamando outra) sem bloquear a si mesma.

    `alvo` so escolhe ONDE fica o arquivo de trava (ver `_raiz_da_trava`):
    todo registro de data/ usa a mesma. Levanta `TravaOcupada`, dizendo qual
    arquivo e quem a segura, se nao ficar livre em `timeout` segundos.
    """
    prazo = TIMEOUT_S if timeout is None else timeout
    caminho = _raiz_da_trava(alvo) / NOME_TRAVA
    chave = str(caminho)
    with _TRAVAS_MUTEX:
        t = _TRAVAS.get(chave)
        if t is None:
            t = _TRAVAS[chave] = _Trava(caminho)
    if not t.rlock.acquire(timeout=max(prazo, 0)):
        raise TravaOcupada(
            f"outra thread deste processo segura a trava {caminho} ha mais de "
            f"{prazo:g} s")
    try:
        if t.nivel == 0:
            caminho.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(caminho), os.O_RDWR | os.O_CREAT, 0o644)
            limite = time.monotonic() + max(prazo, 0)
            while not _tentar_travar(fd):
                if time.monotonic() >= limite:
                    dono = _dono(caminho)
                    os.close(fd)
                    raise TravaOcupada(
                        f"a trava dos registros {caminho} esta com outro "
                        f"processo ({dono}) ha mais de {prazo:g} s. Se ele "
                        f"morreu o SO ja a liberou; se esta preso, encerre-o "
                        f"e rode de novo. Nada foi gravado.")
                time.sleep(_INTERVALO_S)
            t.fd = fd
            if sys.platform != "win32":      # no Windows o byte 0 esta travado
                try:
                    os.ftruncate(fd, 0)
                    os.write(fd, str(os.getpid()).encode())
                except OSError:
                    pass
        t.nivel += 1
    except BaseException:
        t.rlock.release()
        raise
    try:
        yield
    finally:
        t.nivel -= 1
        try:
            if t.nivel == 0 and t.fd is not None:
                fd, t.fd = t.fd, None
                try:
                    _destravar(fd)
                finally:
                    os.close(fd)
        finally:
            t.rlock.release()


# ---------------------------------------------------------------------------
# Gravacao atomica
# ---------------------------------------------------------------------------

def _fsync_dir(pasta: Path) -> None:
    if sys.platform == "win32":
        return
    try:
        fd = os.open(str(pasta), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def gravar_atomico(caminho, conteudo) -> None:
    """
    Grava `conteudo` em `caminho` por substituicao. `conteudo` e str (UTF-8),
    bytes, ou um callable que recebe o Path do TEMPORARIO e o escreve (ex.:
    `lambda p: df.to_csv(p, index=False)`; o nome do temporario termina com o
    nome do destino). Passo a passo:
    temporario no MESMO diretorio, flush + fsync, `os.replace`. Quem le ve o
    arquivo antigo ou o novo, nunca um pedaco. Se algo falhar, o temporario e
    removido e o destino fica como estava.

    No Windows, `os.replace` pode ser negado enquanto outro programa mantem o
    arquivo aberto; tenta por ~2 s antes de desistir (PermissionError).
    NAO substitui a trava: duas gravacoes atomicas concorrentes ainda se
    atropelam num ler->alterar->gravar. Use `trava()`.
    """
    destino = Path(caminho)
    destino.parent.mkdir(parents=True, exist_ok=True)
    # o nome do temporario TERMINA com o do destino (mantem a extensao .csv)
    tmp = destino.parent / f".{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp.{destino.name}"
    try:
        if callable(conteudo):
            conteudo(tmp)
            with open(tmp, "r+b") as f:
                os.fsync(f.fileno())
        else:
            dados = (conteudo.encode("utf-8") if isinstance(conteudo, str)
                     else bytes(conteudo))
            with open(tmp, "wb") as f:
                f.write(dados)
                f.flush()
                os.fsync(f.fileno())
        for tentativa in range(40):
            try:
                os.replace(tmp, destino)
                break
            except PermissionError:
                if sys.platform != "win32" or tentativa == 39:
                    raise
                time.sleep(0.05)
        _fsync_dir(destino.parent)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Leitura integra
# ---------------------------------------------------------------------------

def _recuperacao(caminho: Path, linha: int | None = None) -> str:
    try:
        rel = caminho.resolve().relative_to(Path(config.PROJECT_ROOT).resolve())
    except (ValueError, OSError):
        rel = caminho
    onde = f" ou corrija a linha {linha}" if linha else ""
    return (f"Nada foi gravado. Restaure com: git checkout -- {rel}{onde}. "
            f"Nao rode a publicacao/captura de novo antes disso.")


def validar_csv(caminho) -> str:
    """
    Devolve o texto do CSV se ele esta integro; `RegistroIncompleto` se nao.
    Integro = termina em "\\n" (ou e vazio) e toda linha tem tantos campos
    quanto o cabecalho, respeitando aspas. Arquivo inexistente devolve "".
    """
    p = Path(caminho)
    try:
        bruto = p.read_bytes()
    except FileNotFoundError:
        return ""
    try:
        texto = bruto.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RegistroIncompleto(
            f"{p}: bytes que nao sao UTF-8 ({exc}). {_recuperacao(p)}") from exc
    if not texto:
        return texto
    if not texto.endswith("\n"):
        ultima = texto.count("\n") + 1
        raise RegistroIncompleto(
            f"{p}: a ultima linha (linha {ultima}) esta incompleta -- o arquivo "
            f"nao termina em quebra de linha (escrita interrompida?). "
            f"{_recuperacao(p, ultima)}")
    leitor = csv.reader(io.StringIO(texto, newline=""), strict=True)
    n = None
    try:
        for campos in leitor:
            if not campos:
                continue                      # linha em branco: o pandas tambem pula
            if n is None:
                n = len(campos)
            elif len(campos) != n:
                raise RegistroIncompleto(
                    f"{p}: a linha {leitor.line_num} tem {len(campos)} campos e "
                    f"o cabecalho tem {n} (linha cortada ou colada?). "
                    f"{_recuperacao(p, leitor.line_num)}")
    except csv.Error as exc:
        raise RegistroIncompleto(
            f"{p}: CSV malformado perto da linha {leitor.line_num} ({exc}). "
            f"{_recuperacao(p, leitor.line_num)}") from exc
    return texto


def ler_csv_integro(caminho, **kw):
    """`pd.read_csv` depois de `validar_csv`. Mesmos kwargs do pandas."""
    import pandas as pd
    texto = validar_csv(caminho)
    if not texto:
        raise pd.errors.EmptyDataError(f"{caminho}: arquivo vazio")
    return pd.read_csv(io.StringIO(texto), **kw)


# ---------------------------------------------------------------------------
# Anexar
# ---------------------------------------------------------------------------

def anexar_csv(caminho, colunas, linhas) -> int:
    """
    Anexa `linhas` (dicts) a `caminho`. Sob a trava: valida o arquivo atual
    (`RegistroIncompleto` se estiver truncado, SEM gravar) e regrava
    conteudo integro + linhas novas de forma atomica. Cria o arquivo com
    cabecalho se for novo. Devolve quantas linhas anexou.
    """
    p = Path(caminho)
    linhas = list(linhas)
    with trava(p):
        atual = validar_csv(p)
        fim = "\r\n"
        if atual:
            primeira = atual.split("\n", 1)[0]
            fim = "\r\n" if primeira.endswith("\r") else "\n"
        buf = io.StringIO(newline="")
        w = csv.DictWriter(buf, fieldnames=list(colunas), lineterminator=fim)
        if not atual:
            w.writeheader()
        for d in linhas:
            w.writerow(d)
        gravar_atomico(p, atual + buf.getvalue())
    return len(linhas)
