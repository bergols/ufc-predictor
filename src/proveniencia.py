"""
src/proveniencia.py

Prova de onde cada cotação veio — item 8 da auditoria de integridade.

O que faltava
-------------
O histórico guardava os preços (`sharp_prob`, `sharp_best_odd`,
`close_prob`, `close_best_odd`, `odds_*_tp`) e **nada sobre a origem deles**:
nem o instante da cotação, nem qual casa deu cada preço, nem o mercado
consultado, nem qualquer referência verificável à resposta bruta. Depois do
fato não havia como auditar se um preço estava disponível naquele momento,
nem reconstruir uma captura.

O que este módulo registra
--------------------------
Uma linha por CHAMADA à API, em `data/proveniencia/chamadas.csv`, e uma
cópia comprimida da resposta bruta em `data/proveniencia/<ref>.json.gz`:

    ref            primeiros 16 hex do sha256 do corpo da resposta
    captured_at    instante REAL da resposta, em UTC, do relógio local
    fonte          the_odds_api
    endpoint       caminho consultado, SEM a chave
    regions        regiões pedidas
    markets        mercados pedidos
    n_eventos      quantos eventos vieram
    sha256         hash do corpo inteiro
    corpo_ref      caminho do arquivo com a resposta

A `ref` é carimbada nas linhas do histórico que nasceram daquela chamada
(`sharp_ref`, `close_ref`), então cada preço aponta para a resposta que o
produziu e o hash permite conferir que o arquivo não mudou.

Limite que fica declarado
-------------------------
`captured_at` é o instante em que a RESPOSTA chegou aqui — não o instante em
que a casa formou aquele preço, que a API não informa. É proveniência boa o
bastante para auditar a captura, e **não** é prova de que o preço esteve
disponível em qualquer outro momento.

A URL nunca é guardada com a chave: ela é sanitizada antes de virar registro.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import config

logger = logging.getLogger(__name__)

PASTA = config.DATA_DIR / "proveniencia"
INDICE = PASTA / "chamadas.csv"

_COLUNAS = ("ref", "captured_at", "fonte", "endpoint", "regions", "markets",
            "n_eventos", "sha256", "corpo_ref", "limite")

_LIMITE = ("captured_at e o instante em que a RESPOSTA chegou, nao o instante "
           "em que a casa formou o preco (a API nao informa). Nao prova "
           "disponibilidade em outro momento.")


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def registrar_chamada(corpo: str, *, fonte: str, endpoint: str,
                      regions: str = "", markets: str = "",
                      n_eventos: int | None = None,
                      guardar_corpo: bool = True) -> str:
    """
    Grava a proveniência de uma chamada e devolve a `ref`.

    `endpoint` entra sanitizado: chave de API nunca vira registro. Falha de
    escrita não derruba a captura — proveniência ausente é um problema menor
    que previsão perdida, e o log avisa.
    """
    from src.line_shopping import sanitizar

    digest = hashlib.sha256(corpo.encode("utf-8")).hexdigest()
    ref = digest[:16]
    corpo_ref = ""
    try:
        PASTA.mkdir(parents=True, exist_ok=True)
        if guardar_corpo:
            destino = PASTA / f"{ref}.json.gz"
            if not destino.exists():
                with gzip.open(destino, "wt", encoding="utf-8") as f:
                    f.write(corpo)
            # caminho relativo quando esta dentro do projeto (sobrevive a
            # mover o repositorio); absoluto quando nao esta, em vez de
            # estourar ValueError e perder a proveniencia inteira
            try:
                corpo_ref = str(destino.relative_to(config.PROJECT_ROOT))
            except ValueError:
                corpo_ref = str(destino)
        novo = not INDICE.exists()
        with INDICE.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_COLUNAS)
            if novo:
                w.writeheader()
            w.writerow({"ref": ref, "captured_at": _agora(), "fonte": fonte,
                        "endpoint": sanitizar(endpoint), "regions": regions,
                        "markets": markets,
                        "n_eventos": "" if n_eventos is None else n_eventos,
                        "sha256": digest, "corpo_ref": corpo_ref,
                        "limite": _LIMITE})
    except OSError as exc:
        logger.warning("Nao consegui gravar a proveniencia (%s) -- a captura "
                       "segue, mas esta cotacao fica sem prova.", exc)
    return ref


def conferir(ref: str) -> tuple[bool, str]:
    """(ok, motivo): o corpo guardado ainda bate com o hash do indice?"""
    if not INDICE.exists():
        return False, "indice de proveniencia ausente"
    linhas = [r for r in csv.DictReader(INDICE.open(encoding="utf-8"))
              if r["ref"] == ref]
    if not linhas:
        return False, f"ref {ref} nao esta no indice"
    reg = linhas[-1]
    if not reg["corpo_ref"]:
        return False, "corpo da resposta nao foi guardado"
    caminho = config.PROJECT_ROOT / reg["corpo_ref"]
    if not caminho.exists():
        return False, f"corpo ausente em {reg['corpo_ref']}"
    with gzip.open(caminho, "rt", encoding="utf-8") as f:
        atual = hashlib.sha256(f.read().encode("utf-8")).hexdigest()
    if atual != reg["sha256"]:
        return False, "o corpo guardado NAO bate com o hash registrado"
    return True, "ok"


def cotacao_verificavel(ref) -> bool:
    """A cotação desta linha tem prova utilizável?"""
    import pandas as pd
    if ref is None or (isinstance(ref, float) and pd.isna(ref)) or not str(ref).strip():
        return False
    return conferir(str(ref))[0]
