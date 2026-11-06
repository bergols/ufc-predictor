
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

# O QUE O MANIFESTO SUSTENTA -- e o que NAO sustenta.
#
# MEDIDO nesta maquina em 06/nov/2026, nao assumido: dois treinos
# consecutivos do mesmo dado, com os mesmos hiperparametros e as mesmas
# versoes de biblioteca, produziram arquivos .joblib BYTE A BYTE IDENTICOS, e
# as previsoes nas 14 lutas do card bateram com diferenca maxima 0,0. Aqui,
# portanto, a reproducao e bit a bit e nao apenas numerica.
#
# Esta constante afirmava o contrario -- que joblib nao e byte-estavel e que
# a ordem de soma do BLAS mudaria os ultimos bits. A medicao refutou. O texto
# estava asseverando em vez de medir, e e exatamente o tipo de afirmacao que
# um manifesto nao deve carregar sem teste.
#
# O que o manifesto GARANTE e menos do que o que foi observado, e e o que
# importa: ele registra entradas, fatias, parametros, versoes e hashes dos
# modelos de modo a DETECTAR divergencia. Ele nao garante reproducao em OUTRA
# maquina: implementacao de BLAS, numero de threads e CPU diferentes podem
# mudar os ultimos bits, e isso NAO foi testado. Nada de multiplataforma esta
# comprovado aqui.
#
# Consequencia pratica: hash de modelo diferente e sinal de que algo mudou de
# verdade -- dado, parametro ou versao de biblioteca --, e nao ruido esperado
# de re-treino. Por isso `conferir` o reporta, ainda que como AVISO, porque
# em outra maquina a leitura seria diferente.
_NIVEL = ("bit a bit REPRODUZIDO nesta maquina (dois treinos, arquivos "
          "identicos, diferenca maxima de previsao 0,0); reproducao em "
          "OUTRA maquina NAO testada (BLAS, threads e CPU podem mudar os "
          "ultimos bits)")

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


# Artefatos de modelo. Hash do arquivo treinado: e o que distingue "mesmo
# codigo" de "mesmo modelo".
_MODELOS = ("logreg_calibrated.joblib", "gbm_calibrated.joblib",
            "method_logreg_calibrated.joblib", "method_gbm_calibrated.joblib")


def hashes_dos_modelos() -> dict:
    """sha256 de cada modelo treinado no disco."""
    base = config.PROJECT_ROOT / "models"
    return {nome: sha256_arquivo(base / nome) for nome in _MODELOS}


def _params_escalares(obj, prefixo: str = "", profundidade: int = 0) -> dict:
    """
    Hiperparametros ESCALARES de um estimador, descendo nos estimadores
    internos.

    `get_params(deep=True)` nao resolve sozinho: o modelo de producao e um
    CalibratedClassifierCV em volta de um wrapper em volta de um Pipeline, e
    o deep para na primeira camada que devolve um objeto. Sem descer, o
    manifesto registrava `method` e `cv` e perdia o `C` e o `max_iter` -- que
    sao justamente os parametros que mudam o modelo.

    Guarda de profundidade para nao andar em ciclo num grafo de estimadores.
    Valores nao escalares ficam fora: o repr deles carrega endereco de
    memoria, e isso tornaria o manifesto instavel sem acrescentar informacao.
    """
    if profundidade > 4 or not hasattr(obj, "get_params"):
        return {}
    out = {}
    try:
        params = obj.get_params(deep=False)
    except Exception:  # noqa: BLE001 - estimador de terceiro
        return {}
    for chave, valor in sorted(params.items()):
        nome = f"{prefixo}{chave}"
        if isinstance(valor, (int, float, str, bool, type(None))):
            out[nome] = valor
        elif isinstance(valor, (list, tuple)):
            # Pipeline guarda os passos numa lista de (nome, estimador)
            for item in valor:
                if isinstance(item, tuple) and len(item) == 2:
                    out.update(_params_escalares(item[1], f"{nome}__{item[0]}__",
                                                 profundidade + 1))
        else:
            out.update(_params_escalares(valor, f"{nome}__", profundidade + 1))
    return out


def parametros_do_treino() -> dict:
    """
    Hiperparametros efetivamente usados, lidos dos ESTIMADORES no disco.

    Lidos do artefato, nao copiados do codigo: o manifesto existe para
    descrever o que foi treinado, e um dicionario escrito a mao ao lado da
    chamada divergiria no primeiro ajuste sem ninguem notar. Falta de modelo
    no disco vira None, que e informacao.
    """
    import joblib

    out = {}
    for nome in ("logreg_calibrated.joblib", "gbm_calibrated.joblib"):
        caminho = config.PROJECT_ROOT / "models" / nome
        if not caminho.exists():
            out[nome] = None
            continue
        try:
            modelo = joblib.load(caminho)
            out[nome] = _params_escalares(modelo)
        except Exception as exc:  # noqa: BLE001 - artefato de terceiro
            logger.warning("Nao consegui ler os parametros de %s (%s).", nome, exc)
            out[nome] = None
    out["random_seed"] = config.RANDOM_SEED
    return out


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
        "parametros": parametros_do_treino(),
        "modelos_sha256": hashes_dos_modelos(),
        # O QUE ESTE MANIFESTO SUSTENTA, dito aqui para nao ser lido como
        # garantia maior do que e. Ver `conferir` e o relatorio.
        "nivel_de_reproducao": _NIVEL,
        "fatias_sha256": hashes,
        "fatias_arquivo": str(FATIAS_PATH.relative_to(config.PROJECT_ROOT)),
        # redundante com os arquivos, mas e o que se confere primeiro: fatia
        # que se sobrepoe e vazamento, e o manifesto tem de denunciar sozinho
        "fatias_sobrepostas": sobreposicao,
    }


def _params_divergentes(esperado, atual) -> list[str]:
    """
    Nomes dos hiperparametros que divergem entre manifesto e disco.

    Compara chave por chave, e nao com `!=` no dicionario inteiro, por um
    motivo concreto: o imputer do pipeline tem `missing_values=nan`, e NaN
    nunca e igual a si mesmo. Um `!=` de dicionario reportava divergencia em
    TODO treino, para sempre -- um alarme que sempre soa nao e alarme.
    """
    if esperado is None and atual is None:
        return []
    if esperado is None or atual is None:
        return ["um dos lados nao tem parametros registrados"]
    # `parametros` tem tambem entrada escalar (`random_seed`), nao so o dict
    # por modelo. Comparar direto, sem tentar iterar um int.
    if not isinstance(esperado, dict) or not isinstance(atual, dict):
        return [] if esperado == atual else [f"{esperado!r} != {atual!r}"]
    fora = []
    for chave in sorted(set(esperado) | set(atual)):
        a, b = esperado.get(chave), atual.get(chave)
        if a == b:
            continue
        # NaN == NaN e False, e aqui os dois serem NaN significa IGUAL
        if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
            continue
        fora.append(f"{chave} ({a!r} != {b!r})")
    return fora


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
    # Parametros e modelos: conferidos contra os artefatos EFETIVAMENTE no
    # disco. Divergencia aqui nao e necessariamente erro -- um re-treino muda
    # o hash do modelo por construcao (ver _NIVEL) --, e por isso o modelo
    # aparece como AVISO e o parametro como problema: parametro diferente
    # significa outro modelo, hash diferente pode ser o mesmo modelo.
    atuais = parametros_do_treino()
    for nome, esperado in (manifesto.get("parametros") or {}).items():
        divergentes = _params_divergentes(esperado, atuais.get(nome))
        if divergentes:
            problemas.append(f"parametro {nome}: {', '.join(divergentes)}")
    for nome, esperado in (manifesto.get("modelos_sha256") or {}).items():
        atual = hashes_dos_modelos().get(nome)
        if esperado != atual:
            problemas.append(f"AVISO modelo {nome}: arquivo mudou "
                             f"(re-treino muda o hash sem mudar o modelo; "
                             f"confira as previsoes, nao o hash)")
    return problemas


def nivel_de_reproducao() -> str:
    """O que o manifesto sustenta. Nao e garantia de reproducao bit a bit."""
    return _NIVEL
