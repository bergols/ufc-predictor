"""
scripts/new_event.py

Fluxo COMPLETO de um evento novo, em um comando so:

  1. atualiza a base com os eventos recentes faltantes (--fill-gap com
     navegador real headless; pule com --skip-data-update se ja rodou);
  2. re-treina tudo (features -> preditor de vencedor -> metodo/round);
  3. publica o relatorio no GitHub Pages (registra as previsoes no
     historico de paper trading, congeladas ate o resultado);
  4. gera o relatorio LOCAL com fotos dos lutadores (card_report.html,
     uso pessoal — nunca publicado).

ANTES de rodar, edite os dois CSVs de odds do evento:
  - data/raw/upcoming_card_odds.csv  (card futuro -> relatorio)
  - data/odds_template.csv           (mesmas lutas, actual_winner vazio;
                                      preencher os vencedores APOS o evento)

Uso:
    python -m scripts.new_event data/raw/upcoming_card_odds.csv \
        --card-name "UFC 331: Fulano vs Beltrano" --event-date 2026-08-01
"""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline completo de um evento novo (dados -> treino -> publicacao -> relatorio local).")
    parser.add_argument("csv", type=str, help="CSV de odds do card (data/raw/upcoming_card_odds.csv)")
    parser.add_argument("--card-name", type=str, required=True, help="Nome do evento")
    parser.add_argument("--event-date", type=str, required=True,
                        help="Data do evento (YYYY-MM-DD) — obrigatoria: sem ela o historico nao registra")
    parser.add_argument("--model", choices=["logreg", "gbm"], default="logreg")
    parser.add_argument("--skip-data-update", action="store_true",
                        help="Pula fill-gap + re-treino (use se a base ja esta em dia)")
    parser.add_argument("--no-push", action="store_true",
                        help="Publica localmente sem push (para conferir antes)")
    parser.add_argument("--no-photos", action="store_true",
                        help="Nao gera o relatorio local com fotos")
    parser.add_argument("--local-output", type=str, default="card_report.html",
                        help="Arquivo do relatorio local com fotos (default: card_report.html)")
    args = parser.parse_args()

    if not args.skip_data_update:
        logger.info("=== [1/4] Atualizando a base (fill-gap com navegador) ===")
        from src.data_collection import (check_data_freshness, fill_recent_gap_with_browser,
                                         merge_manual_recent_fights)
        fill_recent_gap_with_browser()
        merge_manual_recent_fights()
        check_data_freshness()

        logger.info("=== [2/4] Re-treinando (features -> vencedor -> metodo/round) ===")
        from src import features, train, train_method
        features.build_feature_dataset()
        train.train_and_calibrate()
        train_method.train_method_and_round()
    else:
        logger.info("=== [1-2/4] Base e treino: pulados (--skip-data-update) ===")

    logger.info("=== [3/4] Publicando no GitHub Pages ===")
    from scripts.publish_report import publish
    rc = publish(args.csv, args.card_name, model=args.model,
                 event_date=args.event_date, no_push=args.no_push)
    if rc != 0:
        return rc

    if not args.no_photos:
        logger.info("=== [4/4] Relatorio local com fotos (%s) ===", args.local_output)
        from src.card_report import generate_card_report
        generate_card_report(args.csv, args.local_output, model_name=args.model,
                             card_name=args.card_name, event_date=args.event_date, photos=True)
    else:
        logger.info("=== [4/4] Relatorio local com fotos: pulado (--no-photos) ===")

    logger.info("Evento pronto. Apos o card: preencha os vencedores em data/odds_template.csv "
                "e rode 'python -m src.evaluate --model %s' (o historico fecha sozinho na "
                "proxima publicacao).", args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
