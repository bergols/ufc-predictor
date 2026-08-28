"""
scripts/auto_capture.py

Captura a linha de fechamento SOZINHO, sem ninguem precisar lembrar.

Feito para rodar de hora em hora num agendador (Agendador de Tarefas do
Windows). E seguro rodar a qualquer hora, quantas vezes for: na esmagadora
maioria das rodadas ele sai em silencio sem tocar em nada e sem gastar um
credito de API.

Por que existe
--------------
O evento 7 (22/ago/2026) perdeu o CLV porque a captura manual nao aconteceu.
Nao ha backfill -- a API so serve eventos futuros --, entao as 11 medicoes
daquele card sumiram para sempre. Automatizar e a unica correcao real: o
passo manual dependia de alguem lembrar num sabado a tarde.

Como decide
-----------
1. Olha o historico LOCAL: existe evento com previsao registrada, sem
   resultado e com data de hoje ou amanha? Se nao, encerra. Esta e a
   comporta de custo -- nenhuma chamada de API em dia sem card.
2. So entao consulta a API (1 chamada) e procura o evento pelo confronto.
3. Se a API AINDA LISTA o evento, ele nao comecou: captura, sobrescrevendo o
   fechamento anterior. Rodando de hora em hora, a ultima gravacao antes do
   card e a que fica -- que e a definicao de linha de fechamento.
4. Se a API nao lista mais, o card ja comecou ou passou: nao grava nada. A
   propria API e a comporta que impede gravar fechamento depois do evento,
   sem o script precisar confiar em relogio ou fuso.

Uso manual (nao faz mal, mesma logica):
    python -m scripts.auto_capture
    python -m scripts.auto_capture --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# So consulta a API se o evento for hoje ou amanha. Segura o custo: sem isso,
# rodando de hora em hora, seriam ~72 creditos por DIA em vez de ~40 por
# EVENTO (a chave gratuita da 500/mes e o projeto roda ~4 eventos/mes).
DIAS_DE_ANTECEDENCIA = 1


def _eventos_abertos(history) -> list[tuple[str, str]]:
    """[(event_date, event_name)] com previsao registrada e sem resultado."""
    if history.empty:
        return []
    aberto = history[history["actual_winner"].isna() & history["model_side"].notna()]
    if aberto.empty:
        return []
    return sorted({(str(r["event_date"]), str(r["event_name"]))
                   for _, r in aberto.iterrows()})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Captura a linha de fechamento automaticamente, quando for a hora.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Decide e explica, mas nao grava nada")
    args = parser.parse_args()

    from src.line_shopping import fetch_live_odds, get_api_key, match_event
    from src.prediction_history import (load_history, open_fights_for_event,
                                        record_closing_lines, compute_clv_summary,
                                        events_missing_closing)
    from src.line_shopping import fetch_sharp_probs

    hoje = date.today()
    limite = hoje + timedelta(days=DIAS_DE_ANTECEDENCIA)
    history = load_history()

    # Alarme antes de qualquer decisao: se um card ja passou sem fechamento,
    # este job falhou em silencio e ninguem soube. Roda de graca (so olha o
    # CSV local) e sai em toda rodada horaria, entao o log guarda a hora exata
    # em que o buraco apareceu.
    for f in events_missing_closing(history, hoje):
        if f["estado"] == "perdido":
            logger.error("SEM FECHAMENTO: %s (%s) já passou e as %d pernas ficaram "
                         "sem CLV. Não há backfill — a API só serve eventos futuros. "
                         "Verifique se a tarefa horária está ativa e se a chave de "
                         "API é válida.", f["event_name"], f["event_date"], f["n"])
        else:
            logger.warning("Fechamento ainda não capturado para %s (%s), que é hoje "
                           "ou amanhã. Se esta rodada não gravar, rode na mão antes "
                           "do card: python -m scripts.capture_closing --event-date %s",
                           f["event_name"], f["event_date"], f["event_date"])

    candidatos = [(d, nome) for d, nome in _eventos_abertos(history)
                  if d and hoje.isoformat() <= d <= limite.isoformat()]
    if not candidatos:
        logger.info("Nenhum evento aberto para hoje ou amanhã — nada a fazer "
                    "(nenhum crédito de API gasto).")
        return 0

    try:
        events = fetch_live_odds(get_api_key())
    except Exception as exc:  # noqa: BLE001 - job agendado nunca deve estourar
        logger.warning("API indisponível (%s) — tentará de novo na próxima rodada.", exc)
        return 0

    total = 0
    for event_date, event_name in candidatos:
        fights = open_fights_for_event(event_date)
        if not fights:
            continue

        # O commence_time vem da propria API: se o evento ainda esta listado,
        # nao comecou. Nao dependemos de relogio local nem de fuso.
        primeiro = fights[0]
        api_event, _, _ = match_event(primeiro["fighter_a"], primeiro["fighter_b"], events)
        if api_event is None:
            logger.warning("%s (%s): a API não lista mais este evento — já começou ou "
                           "passou. NÃO gravando fechamento (seria fechamento falso).",
                           event_name, event_date)
            continue

        inicio = api_event.get("commence_time", "?")
        try:
            faltam = (datetime.fromisoformat(inicio.replace("Z", "+00:00"))
                      - datetime.now(timezone.utc)).total_seconds() / 3600
            quando = f"começa em {faltam:.1f}h"
        except (ValueError, AttributeError):
            quando = f"início {inicio}"

        if args.dry_run:
            logger.info("[dry-run] %s (%s): %s — capturaria %d luta(s).",
                        event_name, event_date, quando, len(fights))
            continue

        sharp = fetch_sharp_probs(fights, events=events)
        n = record_closing_lines(event_date, sharp, allow_update=True)
        total += n
        logger.info("%s (%s): %s — fechamento gravado em %d de %d luta(s).",
                    event_name, event_date, quando, n, len(fights))

    if total:
        resumo = compute_clv_summary(load_history())
        if resumo:
            logger.info("CLV da série: %+.4f em média (%d/%d pernas bateram o fecho).",
                        resumo["media"], resumo["positivos"], resumo["n"])
        logger.info("Lembre de commitar data/prediction_history.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
