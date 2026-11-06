
"""
src/manifesto.py

Manifesto de reprodução de um treino — item 13 da auditoria de integridade.

O que faltava
-------------
`models/training_metadata.json` guardava `trained_at`, método de calibração,
contagens e intervalos de data. Com isso dá para descrever um treino, e não
para reproduzi-lo nem para provar que dois resultados vieram do mesmo dado.

Faltavam três coisas, e são as três que este módulo acrescenta:

  - **hash dos dados de entrada**: sha256 do CSV de features e de cada CSV
    bruto. Sem isso, "treinado em 05/nov" não distingue duas bases
    diferentes;
  - **versões das bibliotecas**: o mesmo código com outra versão de sklearn
    ou lightgbm produz outro modelo. Python inclusive;
  - **composição exata das fatias**: a lista de `fight_id` de treino,
    calibração e teste. Intervalo de datas não basta — o split agrupa por
    luta e qualquer mudança na base muda quem cai onde.

As listas de `fight_id` são grandes (milhares), então vão num arquivo
separado e o manifesto guarda o hash delas. Quem quiser conferir compara o
hash; quem quiser auditar abre o arquivo.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from pathlib import Path

import config

logger = logging.getLogger(__name__)

FATIAS_PATH = config.PROJECT_ROOT / "models" / "training_slices.json"

# Bibliotecas cuja versao muda o modelo produzido. Faltar uma e pior que
# listar a mais: o manifesto passa a afirmar algo que nao verificou.
_BIBLIOTECAS = ("numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "joblib")


def sha256_arquivo(caminho: Path, _bloco: int = 1 << 20) -> str | None:
    """sha256 de um arquivo, ou None se ele nao existe."""
    caminho = Path(caminho)
    if not caminho.exists():
        return None
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for pedaco in iter(lambda: f.read(_bloco), b""):
            h.update(pedaco)
    return h.hexdigest()


def sha256_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def versoes_bibliotecas() -> dict:
    """Versao instalada de cada biblioteca que afeta o modelo."""
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": sys.version.split()[0], "plataforma": platform.platform()}
    for nome in _BIBLIOTECAS:
        try:
            out[nome] = version(nome)
        except PackageNotFoundError:
            out[nome] = None       # ausente e informacao, nao erro
    return out


def hashes_das_entradas() -> dict:
    """sha256 de cada arquivo que entra no treino."""
    alvos = {
        "features": config.FEATURES_CSV,
        "fights_raw": config.RAW_FIGHTS_CSV,
        "fight_stats_raw": config.RAW_FIGHT_STATS_CSV,
        "fighters_raw": config.RAW_FIGHTERS_CSV,
    }
    return {nome: sha256_arquivo(p) for nome, p in alvos.items()}


def montar(train_ids, cal_ids, test_ids, salvar: bool = True) -> dict:
    """
    Bloco de reproducao do manifesto. Escreve as listas de `fight_id` em
    `models/training_slices.json` e devolve os hashes delas.

    As listas vao ORDENADAS: a ordem do split nao e informacao do dado, e
    deixar que ela mexa no hash tornaria dois treinos identicos
    indistinguiveis de dois diferentes.
    """
    fatias = {nome: sorted(map(str, ids)) for nome, ids in
              (("treino", train_ids), ("calibracao", cal_ids), ("teste", test_ids))}
    hashes = {nome: sha256_texto("\n".join(v)) for nome, v in fatias.items()}
    sobreposicao = {
        "treino_x_calibracao": len(set(fatias["treino"]) & set(fatias["calibracao"])),
        "treino_x_teste": len(set(fatias["treino"]) & set(fatias["teste"])),
        "calibracao_x_teste": len(set(fatias["calibracao"]) & set(fatias["teste"])),
    }
    if salvar:
        FATIAS_PATH.parent.mkdir(exist_ok=True)
        FATIAS_PATH.write_text(json.dumps(fatias, indent=1), encoding="utf-8")
        logger.info("Composicao das fatias em %s", FATIAS_PATH)

    return {
        "bibliotecas": versoes_bibliotecas(),
        "entradas_sha256": hashes_das_entradas(),
        "fatias_sha256": hashes,
        "fatias_arquivo": str(FATIAS_PATH.relative_to(config.PROJECT_ROOT)),
        # redundante com os arquivos, mas e o que se confere primeiro: fatia
        # que se sobrepoe e vazamento, e o manifesto tem de denunciar sozinho
        "fatias_sobrepostas": sobreposicao,
    }


def conferir(manifesto: dict) -> list[str]:
    """
    Divergencias entre o manifesto e o estado ATUAL do disco. Lista vazia =
    o treino descrito pode ser reproduzido com o que esta aqui.
    """
    problemas = []
    for nome, esperado in (manifesto.get("entradas_sha256") or {}).items():
        atual = hashes_das_entradas().get(nome)
        if esperado != atual:
            problemas.append(f"entrada {nome}: manifesto {esperado} != disco {atual}")
    for nome, esperado in (manifesto.get("bibliotecas") or {}).items():
        atual = versoes_bibliotecas().get(nome)
        if esperado != atual:
            problemas.append(f"biblioteca {nome}: manifesto {esperado} != instalado {atual}")
    for par, n in (manifesto.get("fatias_sobrepostas") or {}).items():
        if n:
            problemas.append(f"VAZAMENTO: {n} luta(s) em {par}")
    return problemas
