"""
scripts/publish_report.py

Fluxo completo de publicacao do relatorio de card no GitHub Pages, em um
comando so:

    python -m scripts.publish_report data/raw/upcoming_card_odds.csv --card-name "UFC 329"

Faz: gera docs/index.html (o arquivo que o GitHub Pages serve na raiz do
link) -> git add (index + o CSV de odds versionado) -> commit -> push.
O link publicado e fixo; cada evento novo e so editar o CSV de odds e
rodar este script de novo.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DOCS_INDEX = config.PROJECT_ROOT / "docs" / "index.html"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(config.PROJECT_ROOT), *args],
                          capture_output=True, text=True)


def publish(csv: str, card_name: str, model: str = "logreg", event_date: str = "",
            no_push: bool = False) -> int:
    """Gera docs/index.html, commita (HTML + historico + CSV de odds) e faz
    push. Reutilizado por scripts/new_event.py. Retorna exit code."""
    from src.card_report import generate_card_report

    DOCS_INDEX.parent.mkdir(exist_ok=True)
    generate_card_report(csv, DOCS_INDEX, model_name=model, card_name=card_name,
                         event_date=event_date)

    to_add = ["docs/index.html"]
    if config.PREDICTION_HISTORY_CSV.exists():
        to_add.append("data/prediction_history.csv")
    csv_path = Path(csv).resolve()
    try:
        to_add.append(str(csv_path.relative_to(config.PROJECT_ROOT)))
    except ValueError:
        pass  # CSV fora do repo: publica so o HTML

    _git("add", *to_add)
    commit = _git("commit", "-m", f"Atualiza relatório: {card_name}")
    if commit.returncode != 0:
        if "nothing to commit" in commit.stdout + commit.stderr:
            logger.info("Nada mudou desde a ultima publicacao -- nenhum commit criado.")
            return 0
        logger.error("git commit falhou:\n%s%s", commit.stdout, commit.stderr)
        return 1
    logger.info("Commit criado: Atualiza relatório: %s", card_name)

    if no_push:
        logger.info("--no-push: publicacao local pronta; rode 'git push' quando quiser publicar.")
        return 0

    push = _git("push")
    if push.returncode != 0:
        logger.error("git push falhou (remote configurado? credenciais?):\n%s%s",
                     push.stdout, push.stderr)
        return 1
    logger.info("Publicado! O GitHub Pages atualiza o link fixo em ~1 minuto.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera e publica o relatorio de card no GitHub Pages.")
    parser.add_argument("csv", type=str, help="CSV de odds do card (fighter_a,fighter_b,odds_*,scheduled_rounds)")
    parser.add_argument("--card-name", type=str, required=True,
                        help="Nome do evento (vai para o titulo da pagina e a mensagem do commit)")
    parser.add_argument("--model", choices=["logreg", "gbm"], default="logreg")
    parser.add_argument("--event-date", type=str, default="",
                        help="Data do evento (YYYY-MM-DD) -- registra as previsoes no "
                             "historico de paper trading (aba Historico do relatorio)")
    parser.add_argument("--no-push", action="store_true",
                        help="Gera e commita, mas nao faz push (para conferir antes)")
    args = parser.parse_args()
    return publish(args.csv, args.card_name, model=args.model,
                   event_date=args.event_date, no_push=args.no_push)


if __name__ == "__main__":
    sys.exit(main())
