"""
scripts/sombra.py

Registra um evento em MODO SOMBRA, ou liquida o que já aconteceu.

    # antes das lutas: registra os tres candidatos, os precos e as provas
    python -m scripts.sombra --event-date 2026-11-07

    # depois: anexa o resultado, sem reescrever o que foi previsto
    python -m scripts.sombra --event-date 2026-11-07 --liquidar

    # so mostrar, sem gravar
    python -m scripts.sombra --event-date 2026-11-07 --dry-run

Manual de propósito: nada aqui é agendado, nenhuma API é consultada e nada é
publicado. Escreve só em `data/experimentos/sombra/`, e confere o sha256 dos
registros de produção antes e depois.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import config
from src import sombra

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROTEGIDOS = (
    config.PREDICTION_HISTORY_CSV,
    config.DATA_DIR / "previsoes_snapshots.csv",
    config.DATA_DIR / "decisoes.csv",
    config.DATA_DIR / "proveniencia" / "chamadas.csv",
    config.DATA_DIR / "proveniencia" / "cotacoes.csv",
    config.UPCOMING_CARD_CSV,
)


def _marcar() -> dict:
    def h(p: Path):
        try:
            return hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            return None
    return {str(p): h(p) for p in PROTEGIDOS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--card", default=None, help="CSV do card (default: o de producao)")
    ap.add_argument("--liquidar", action="store_true",
                    help="anexa resultados em vez de registrar previsoes")
    ap.add_argument("--dry-run", action="store_true", help="mostra e nao grava")
    args = ap.parse_args(argv)

    antes = _marcar()

    if args.liquidar:
        n = 0 if args.dry_run else sombra.liquidar(args.event_date)
        if args.dry_run:
            logger.info("--dry-run: nada gravado.")
    else:
        linhas = sombra.montar(args.event_date, card_csv=args.card)
        if not linhas:
            logger.error("Nenhuma luta no card de %s.", args.event_date)
            return 1
        sel = sum(1 for d in linhas if d.get("selecao"))
        from collections import Counter
        classes = Counter(d.get("classe") for d in linhas)
        logger.info("Evento %s | candidato %s congelado em %s | peso w = %s",
                    args.event_date, linhas[0].get("candidato_id") or "(nenhum)",
                    linhas[0].get("candidato_congelado_em") or "(sem instante)",
                    linhas[0].get("peso_w"))
        logger.info("Classificacao: %s", dict(classes))
        logger.info("%d luta(s): %d selecao(oes), %d recusa(s).",
                    len(linhas), sel, len(linhas) - sel)
        for d in linhas:
            if d.get("selecao"):
                logger.info("  SELECAO  %s | odd %s | retorno bruto esperado %s "
                            "| lucro liquido esperado %s | classe %s",
                            d.get("lado_selecionado"),
                            d.get("odd_lado_selecionado"),
                            d.get("retorno_bruto_esperado"),
                            d.get("lucro_liquido_esperado"), d.get("classe"))
            else:
                logger.info("  recusa   %s x %s -- %s",
                            d.get("fighter_a"), d.get("fighter_b"),
                            d.get("motivo"))
            if d.get("motivos_da_classe"):
                logger.info("           classe %s: %s", d.get("classe"),
                            d.get("motivos_da_classe"))
        n = 0 if args.dry_run else len(sombra.registrar(linhas))
        if args.dry_run:
            logger.info("--dry-run: nada gravado.")

    depois = _marcar()
    mexidos = [k for k in antes if antes[k] != depois[k]]
    if mexidos:
        logger.error("ABORTANDO: a sombra ALTEROU registro de producao: %s",
                     ", ".join(mexidos))
        return 2
    logger.info("Registros de producao intactos (%d arquivos).", len(antes))
    return 0 if (args.dry_run or n >= 0) else 1


if __name__ == "__main__":
    sys.exit(main())
