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
    ap.add_argument("--emenda", nargs="*", default=["EMENDA_V2_MANIFESTO.md"],
                    help="emendas citadas pelo hash (EMENDA_V2_MANIFESTO.md)")
    ap.add_argument("--mostrar", action="store_true",
                    help="mostra o registro que seria congelado e NAO grava")
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

    import json

    import config
    from src import codigo
    from src.decisoes import manifesto_sha256
    manifesto = manifesto_sha256()
    if not manifesto:
        logger.error("Sem models/training_metadata.json: o candidato ficaria sem "
                     "modelo amarrado e nenhuma decisao dele qualificaria.")
        return 1
    # O codigo precisa ser uma versao verificavel (EMENDA_V2_MANIFESTO.md,
    # regra 4; alcance em src/codigo.py): src/, scripts/ e config.py limpos
    # e iguais ao commit.
    try:
        commit, impressao = codigo.registrar()
    except codigo.NaoVerificavel as exc:
        logger.error("Codigo sem versao verificavel: %s", exc)
        return 1
    try:
        reg = candidatos.congelar(
            versao_protocolo=args.versao, peso_w=args.peso,
            eventos_desenvolvimento=args.dev,
            eventos_consultados=args.consultados,
            descricao=args.descricao,
            manifesto_modelo_sha256=manifesto,
            emendas=[config.PROJECT_ROOT / e for e in args.emenda],
            codigo_commit=commit, codigo_sha256=impressao,
            gravar=not args.mostrar)
    except ValueError as exc:
        logger.error("Candidato NAO congelado: %s", exc)
        return 1
    if args.mostrar:
        print(json.dumps(reg, indent=2, ensure_ascii=False))
        logger.info("--mostrar: NADA gravado.")
        return 0
    logger.info("candidato_id: %s | protocolo_sha256: %s | manifesto do modelo: %s",
                reg["candidato_id"], reg["protocolo_sha256"][:16] or "(ausente)",
                manifesto[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
