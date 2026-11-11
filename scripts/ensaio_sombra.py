"""Executa captura, previsão e dry-run em sequência numa cópia exportada.

    python -m scripts.ensaio_sombra --event-date AAAA-MM-DD --card-name "Evento"

Exige uma cópia sem .git, com base, modelos e credencial já disponíveis.
A montagem executa uma chamada à The Odds API. Falhas encerram a sequência.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-date", required=True)
    parser.add_argument("--card-name", required=True)
    parser.add_argument("--chromium-ufcstats", action="store_true")
    args = parser.parse_args(argv)

    if (config.PROJECT_ROOT / ".git").exists():
        parser.error("execute em uma copia exportada, sem .git, com base e modelos")
    arquivos = [
        config.RAW_FIGHTS_CSV, config.RAW_FIGHT_STATS_CSV, config.RAW_FIGHTERS_CSV,
        config.MODELS_DIR / f"{config.MODELO_PRODUCAO}_calibrated.joblib",
    ]
    ausentes = [str(p) for p in arquivos if not p.is_file()]
    if ausentes:
        parser.error("arquivos necessarios ausentes: " + ", ".join(ausentes))

    # Carregar o modelo antes da chamada paga identifica incompatibilidade do ambiente.
    import joblib
    from src.features import export_latest_fighter_levels
    from src.line_shopping import get_api_key

    joblib.load(arquivos[-1])
    export_latest_fighter_levels()
    get_api_key()
    pasta = config.DATA_DIR / "experimentos" / "ensaio" / args.event_date
    pasta.mkdir(parents=True, exist_ok=True)
    card = pasta / "card.csv"
    relatorio = pasta / "relatorio.html"
    comandos = [
        ["scripts.build_card", "--event-date", args.event_date, "--out", str(card)],
        ["src.card_report", str(card), "--event-date", args.event_date,
         "--card-name", args.card_name, "--output", str(relatorio), "--no-sharp"],
        ["scripts.sombra", "--event-date", args.event_date, "--card", str(card), "--dry-run"],
    ]
    if args.chromium_ufcstats:
        comandos[0].append("--chromium-ufcstats")
    for comando in comandos:
        resultado = subprocess.run([sys.executable, "-m", *comando],
                                   cwd=config.PROJECT_ROOT)
        if resultado.returncode:
            logger.error("Etapa %s falhou. Captura e arquivos preservados em %s.",
                         comando[0], pasta)
            return resultado.returncode
    logger.info("Ensaio concluido em %s.", pasta)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    sys.exit(main())
