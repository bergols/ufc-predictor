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
3. Decide LUTA A LUTA: captura so a que a API ainda lista E cujo
   commence_time esta no futuro. Rodando de hora em hora, a ultima gravacao
   antes do card e a que fica -- que e a definicao de linha de fechamento.
4. Luta sem listagem, com inicio ilegivel ou ja comecada nao grava nada, e o
   motivo vai para o log. Sem prova de que nao comecou, nao e fechamento.

   Ate out/2026 o card inteiro era julgado pela PRIMEIRA luta. Isso capturava
   odd AO VIVO de prelim ja em andamento, e perdia o resto do card quando a
   primeira luta saia da listagem antes das outras (card atravessando a
   meia-noite UTC).

O segundo trabalho (out/2026): CARD A REGISTRAR
-----------------------------------------------
O UFC 332 (03/out) passou sem pre-registro nenhum e ninguem soube -- sem
linha no historico, nao ha o que um alarme de historico encontre. Agora a
mesma rodada horaria tambem pergunta "a API ja lista algum card que nao esta
na serie?", uma vez por dia, e deixa a resposta em data/eventos_a_registrar.json
para o painel mostrar sem fazer rede. Ver src/event_watch.py.

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

    from src.casamento import casar
    from src.line_shopping import (fetch_live_odds, get_api_key, match_event,
                                   sanitizar)
    from src.prediction_history import (load_history, open_fights_for_event,
                                        record_closing_lines, compute_clv_summary,
                                        events_missing_closing)
    from src.line_shopping import fetch_sharp_probs
    from src.event_watch import (eventos_a_registrar, eventos_perdidos,
                                 precisa_verificar, salvar_pendentes)

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

    # Card que passou sem pre-registro nenhum nao deixa linha no historico,
    # entao o alarme de fechamento nao tem o que encontrar. Este olha a base
    # de resultados. Tambem de graca, tambem em toda rodada.
    for e in eventos_perdidos(history, hoje=hoje):
        logger.error("SEM PRE-REGISTRO: %s (%s) aconteceu e nao esta na serie. "
                     "Irrecuperavel -- previsao retroativa nao e previsao.",
                     e["event_name"], e["event_date"])

    candidatos = [(d, nome) for d, nome in _eventos_abertos(history)
                  if d and hoje.isoformat() <= d <= limite.isoformat()]
    # A verificacao de cards a registrar precisa da API justamente nos dias em
    # que NAO ha nada registrado -- o oposto da comporta de captura. Uma vez
    # por dia; nas outras 23 rodadas o cache responde.
    checar_proximos = precisa_verificar(hoje)
    if not candidatos and not checar_proximos:
        logger.info("Nenhum evento aberto para hoje ou amanhã e a verificação de "
                    "cards a registrar já rodou hoje — nada a fazer (nenhum "
                    "crédito de API gasto).")
        return 0

    try:
        events = fetch_live_odds(get_api_key())
    except Exception as exc:  # noqa: BLE001 - job agendado nunca deve estourar
        logger.warning("API indisponível (%s) — tentará de novo na próxima rodada.",
                       sanitizar(exc))
        return 0

    if checar_proximos:
        pendentes = eventos_a_registrar(events, history, hoje=hoje)
        # grava inclusive a lista vazia: "verifiquei e nao ha pendencia" e
        # diferente de "nao verifiquei", e so o carimbo distingue as duas
        if not args.dry_run:
            salvar_pendentes(pendentes, hoje=hoje)
        for e in pendentes:
            logger.warning("A REGISTRAR: card de %s (em %d dia(s)), %d luta(s) já "
                           "listadas pela API e nenhum pré-registro. O mercado "
                           "abriu: monte o card agora, não na véspera.",
                           e["event_date"], e["dias"], e["n_lutas"])
        if not pendentes:
            logger.info("Nenhum card a registrar nos próximos dias.")

    if not candidatos:
        return 0

    total = 0
    for event_date, event_name in candidatos:
        fights = open_fights_for_event(event_date)
        if not fights:
            continue

        # RELOGIO POR LUTA (out/2026). Antes o card inteiro era julgado pela
        # PRIMEIRA luta: se a API ainda a listava, capturava todas. Dois
        # modos de falha nisso:
        #   - prelim que ja comecou enquanto o resto do card nao: a odd dela
        #     ja e ao vivo, e entrava como "fechamento";
        #   - card que atravessa a meia-noite UTC: a primeira luta some da
        #     listagem e o resto, que ainda nao comecou, perde a captura.
        # Agora cada luta responde por si: precisa estar listada E ter
        # commence_time no FUTURO. Inicio ausente ou ilegivel nao captura --
        # sem prova de que nao comecou, nao grava.
        agora = datetime.now(timezone.utc)
        abertas, bloqueadas = [], []
        for luta in fights:
            # O casamento tem de ser inequivoco ANTES de olhar o relogio:
            # commence_time de outro evento provaria que OUTRA luta nao
            # comecou, e a captura entraria como fechamento desta.
            c = casar(luta["fighter_a"], luta["fighter_b"], events,
                      event_date=event_date)
            if not c.ok:
                bloqueadas.append((luta, f"casamento nao inequivoco: {c.motivo}"))
                continue
            inicio = c.commence_time
            try:
                quando = datetime.fromisoformat(str(inicio).replace("Z", "+00:00"))
            except (ValueError, AttributeError, TypeError):
                bloqueadas.append((luta, f"início ilegível ({inicio!r})"))
                continue
            if quando <= agora:
                bloqueadas.append((luta, f"já começou ({quando.isoformat()})"))
                continue
            abertas.append(luta)

        for luta, motivo in bloqueadas:
            logger.warning("%s (%s): %s vs %s NÃO capturada — %s. Gravar aqui seria "
                           "fechamento falso (odd ao vivo ou pós-início).",
                           event_name, event_date, luta["fighter_a"], luta["fighter_b"], motivo)
        if not abertas:
            continue

        if args.dry_run:
            logger.info("[dry-run] %s (%s): capturaria %d de %d luta(s).",
                        event_name, event_date, len(abertas), len(fights))
            continue

        sharp = fetch_sharp_probs(abertas, events=events,
                                  event_date=event_date)
        n = record_closing_lines(event_date, sharp, allow_update=True)
        total += n
        logger.info("%s (%s): fechamento gravado em %d de %d luta(s) abertas "
                    "(%d bloqueadas).", event_name, event_date, n, len(abertas),
                    len(bloqueadas))

    if total:
        resumo = compute_clv_summary(load_history())
        if resumo:
            logger.info("CLV da série: %+.4f em média (%d/%d pernas bateram o fecho).",
                        resumo["media"], resumo["positivos"], resumo["n"])
        logger.info("Lembre de commitar data/prediction_history.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
