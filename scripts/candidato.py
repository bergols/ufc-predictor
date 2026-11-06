"""
scripts/candidato.py

Congela um candidato, ou lista os que já estão congelados.

    # listar
    python -m scripts.candidato --listar

    # congelar (so quando a regra da secao 7 do vantagem-v2 permitir)
    python -m scripts.candidato --peso 0.0 --versao vantagem-v2 \
        --descricao "candidato C do v1, preservado sob o v2" \
        --dev 2026-07-11 2026-07-18 --consultados 2026-09-05

Append-only: candidato congelado não é editado. Mudar peso, protocolo ou
população é um candidato **novo**, com instante próprio, e o antigo continua
endereçável com o resultado que teve.
"""
from __future__ import annotations

import argparse
import logging
import sys

from src import candidatos

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--listar", action="store_true")
    ap.add_argument("--peso", type=float, default=None)
    ap.add_argument("--versao", default="vantagem-v2")
    ap.add_argument("--descricao", default="")
    ap.add_argument("--dev", nargs="*", default=[],
                    help="eventos usados para ESTIMAR o parametro")
    ap.add_argument("--consultados", nargs="*", default=[],
                    help="eventos que ja gastaram analise inferencial")
    args = ap.parse_args(argv)

    if args.listar or args.peso is None:
        todos = candidatos.carregar()
        if not todos:
            logger.info("Nenhum candidato congelado.")
            return 0
        for c in todos:
            logger.info("%s | congelado em %s | %s | w = %s | "
                        "%d evento(s) queimado(s)",
                        c["candidato_id"], c["congelado_em"],
                        c["versao_protocolo"], c["peso_w"],
                        len(candidatos.eventos_queimados(c)))
            if c.get("descricao"):
                logger.info("    %s", c["descricao"])
        em_vigor = candidatos.atual()
        logger.info("Em vigor: %s", em_vigor["candidato_id"] if em_vigor else "-")
        return 0

    reg = candidatos.congelar(
        versao_protocolo=args.versao, peso_w=args.peso,
        eventos_desenvolvimento=args.dev,
        eventos_consultados=args.consultados,
        descricao=args.descricao)
    logger.info("candidato_id: %s | protocolo_sha256: %s",
                reg["candidato_id"], reg["protocolo_sha256"][:16] or "(ausente)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
