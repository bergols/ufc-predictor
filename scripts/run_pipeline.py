"""
scripts/run_pipeline.py

Orquestra o pipeline inteiro do zero: coleta de dados -> features ->
treino -> avaliacao. Util para rodar tudo de uma vez; cada etapa tambem
pode ser rodada isoladamente (veja o README.md).

Uso:
    python -m scripts.run_pipeline --source public-dataset
    python -m scripts.run_pipeline --source scrape --limit-events 50
"""
from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Roda o pipeline completo de treino do UFC Predictor.")
    parser.add_argument("--source", choices=["github-mirror", "scrape", "public-dataset"],
                         default="github-mirror",
                         help="Fonte de dados brutos (default: github-mirror, atualizado diariamente; "
                              "public-dataset e o fallback do Kaggle, congelado em jun/2019).")
    parser.add_argument("--limit-events", type=int, default=None,
                         help="Limita o numero de eventos no scraping (so vale com --source scrape).")
    parser.add_argument("--skip-collection", action="store_true",
                         help="Pula a coleta de dados (use se ja rodou antes e so quer re-treinar).")
    parser.add_argument("--fill-gap", action="store_true",
                         help="Apos a coleta, completa eventos recentes faltantes raspando so eles "
                              "do UFCStats.com com um navegador real headless (requer Playwright).")
    args = parser.parse_args()

    from src import data_collection, evaluate, features, train

    if not args.skip_collection:
        logger.info("=== Etapa 1/4: coleta de dados ===")
        if args.source == "scrape":
            data_collection.run_full_scrape(limit_events=args.limit_events)
        elif args.source == "github-mirror":
            data_collection.download_github_mirror_dataset()
        else:
            data_collection.download_public_dataset_fallback()
        if args.fill_gap:
            data_collection.fill_recent_gap_with_browser()
        data_collection.merge_manual_recent_fights()
    else:
        logger.info("=== Etapa 1/4: coleta de dados (pulada) ===")

    # Aviso bem visivel se a fonte estiver estagnada (ja aconteceu sem ninguem notar)
    data_collection.check_data_freshness()

    logger.info("=== Etapa 2/4: engenharia de features ===")
    features.build_feature_dataset()

    logger.info("=== Etapa 3/4: treino + calibracao ===")
    train.train_and_calibrate()

    logger.info("=== Etapa 4/4: avaliacao ===")
    evaluate.evaluate_test_set()
    evaluate.compare_to_market()

    logger.info("Pipeline concluido. Use 'python -m src.predict \"Fighter A\" \"Fighter B\"' para prever uma luta.")


if __name__ == "__main__":
    main()
