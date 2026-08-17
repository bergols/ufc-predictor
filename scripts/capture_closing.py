"""
scripts/capture_closing.py

Congela a LINHA DE FECHAMENTO das lutas ainda abertas de um evento e calcula
o CLV de cada perna. Rode poucas horas antes do card:

    python -m scripts.capture_closing --event-date 2026-08-22

Por que existe
--------------
O P&L da serie e uma medida ruim de edge no curto prazo: e binario e dominado
por variancia — uma perna a 2.95 decide o placar inteiro. O CLV (closing line
value) mede, em cada perna, quantos pontos de probabilidade o mercado sharp
andou entre o pre-registro e o fecho. Sendo continuo, converge com muito menos
amostra: da leitura com dezenas de pernas em vez de centenas.

    clv = close_prob - sharp_prob   (pontos de probabilidade)

Positivo = a Pinnacle devigada andou NA DIRECAO do lado apontado pelo modelo,
isto e, o preco que pegamos no registro era melhor que o do fecho.

A comparacao e entre DUAS medidas da mesma casa (Pinnacle devigada) de
proposito. Comparar a odd mediana do registro com a melhor odd do fecho
misturaria escalas — a melhor e sistematicamente maior que a mediana — e
inflaria o CLV de graca.

Sem backfill: a API so serve eventos futuros. O que nao for capturado antes do
card esta perdido, exatamente como aconteceu com o sinal sharp nos 4 primeiros
eventos da serie.
"""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Congela a linha de fechamento e o CLV das pernas de um evento.")
    parser.add_argument("--event-date", type=str, required=True,
                        help="Data do evento (YYYY-MM-DD), como gravada no historico")
    args = parser.parse_args()

    from src.line_shopping import fetch_sharp_probs
    from src.prediction_history import (open_fights_for_event, record_closing_lines,
                                        compute_clv_summary, load_history)

    fights = open_fights_for_event(args.event_date)
    if not fights:
        logger.warning("Nenhuma luta aberta com previsao para %s — nada a capturar. "
                       "(Evento ja fechado? Data errada?)", args.event_date)
        return 1

    logger.info("Buscando fechamento de %d luta(s) de %s...", len(fights), args.event_date)
    sharp = fetch_sharp_probs(fights)
    n_ok = sum(1 for v in sharp.values() if v)
    logger.info("Pinnacle cobriu %d de %d.", n_ok, len(fights))

    n = record_closing_lines(args.event_date, sharp)
    if n == 0:
        logger.warning("Nada gravado — fechamento ja capturado antes, ou a API nao "
                       "devolveu nenhuma luta. Linha ja gravada nunca e sobrescrita.")
        return 0

    resumo = compute_clv_summary(load_history())
    if resumo:
        logger.info("CLV da serie: %+.4f em media (%d/%d pernas bateram o fecho).",
                    resumo["media"], resumo["positivos"], resumo["n"])
    logger.info("Pronto. Commite data/prediction_history.csv para registrar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
