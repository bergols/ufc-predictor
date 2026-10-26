"""
scripts/build_card.py

Monta `data/raw/upcoming_card_odds.csv` a partir das DUAS fontes, sem passo
manual e sem omissao silenciosa.

    python -m scripts.build_card --event-date 2026-10-31
    python -m scripts.build_card --event-date 2026-10-31 --so-ufcstats
    python -m scripts.build_card --event-date 2026-10-31 --dry-run

O que faz, nesta ordem:
  1. le o card do UFCStats (ordem das lutas, primeira linha = luta principal);
  2. le as lutas da data na The Odds API;
  3. UNE as duas, marcando de onde cada luta veio (src/card_sources.py);
  4. troca os nomes pela grafia CANONICA da base -- nunca a da fonte;
  5. pega a mediana das casas (benchmark de mercado);
  6. pega a odd da Thunderpick (preco de entrada);
  7. escreve o CSV e RELATA o que ficou de fora e por que.

Por que existe
--------------
A lista de lutas era escrita a mao a partir do UFCStats, e a regra "UFCStats
e a autoridade" errou nas duas direcoes: salvou duas lutas fantasma no evento
12 e custou tres lutas reais no 15. Nenhuma fonte sozinha serve, e o erro
nascia justamente no passo manual, onde a discordancia entre elas era
invisivel.

Luta que so a API conhece entra POR PADRAO. Os dois erros nao custam igual:
incluir uma luta fantasma sai barato (ela nunca acontece, fica aberta, sai dos
denominadores sozinha e a trava de evento passado cuida do resto), enquanto
excluir uma luta real deixa a serie medindo um card que nao existiu. Use
--so-ufcstats para o comportamento antigo.
"""
from __future__ import annotations

import argparse
import logging
import statistics as st
import sys
from pathlib import Path

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _canonico(nome: str, conhecidos: list[str]) -> tuple[str, str]:
    from src.utils import best_name_match
    if nome in conhecidos:
        return nome, "exato"
    achado = best_name_match(nome, conhecidos)
    return (achado, "fuzzy") if achado else (nome, "estreante")


def _medianas(event: dict, api_a: str, api_b: str):
    pa, pb = [], []
    for bm in event.get("bookmakers", []):
        for mk in bm.get("markets", []):
            if mk.get("key") != "h2h":
                continue
            d = {o["name"]: o["price"] for o in mk.get("outcomes", [])}
            if api_a in d and api_b in d:
                pa.append(d[api_a])
                pb.append(d[api_b])
    if not pa:
        return None, None, 0
    return round(st.median(pa), 3), round(st.median(pb), 3), len(pa)


def main() -> int:
    parser = argparse.ArgumentParser(description="Monta o CSV do card unindo UFCStats + The Odds API.")
    parser.add_argument("--event-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out", default=str(config.UPCOMING_CARD_CSV))
    parser.add_argument("--so-ufcstats", action="store_true",
                        help="Nao incluir luta que so a API conhece (comportamento antigo)")
    parser.add_argument("--sem-thunderpick", action="store_true",
                        help="Nao buscar a odd da casa (abre uma janela de navegador)")
    parser.add_argument("--dry-run", action="store_true", help="Relata e nao escreve")
    args = parser.parse_args()

    from src.card_sources import card_do_ufcstats, resumir, unir
    from src.features import export_latest_fighter_levels
    from src.line_shopping import fetch_live_odds, get_api_key

    ufc = card_do_ufcstats(args.event_date)
    api_events = fetch_live_odds(get_api_key())
    card = unir(ufc["lutas"], api_events, args.event_date)
    logger.info("Uniao: %s", resumir(card))

    conhecidos = list(export_latest_fighter_levels()["fighter"])
    linhas, fora = [], []
    for luta in card:
        if luta.fonte == "so_api" and args.so_ufcstats:
            fora.append((luta, "so a API conhece e --so-ufcstats foi pedido"))
            continue
        if luta.fonte == "so_ufcstats":
            fora.append((luta, "nenhuma casa da API precificou -- sem mediana nao "
                               "ha benchmark de mercado"))
            continue

        ca, how_a = _canonico(luta.fighter_a, conhecidos)
        cb, how_b = _canonico(luta.fighter_b, conhecidos)
        for nome, canon, how in ((luta.fighter_a, ca, how_a), (luta.fighter_b, cb, how_b)):
            if how == "fuzzy":
                logger.warning("Nome ajustado para a grafia da base: %r -> %r", nome, canon)
            elif how == "estreante":
                logger.info("%r sem historico na base -- estreante.", canon)

        from src.line_shopping import match_event
        ev, api_a, api_b = match_event(ca, cb, api_events)
        oa, ob, n_casas = _medianas(ev, api_a, api_b) if ev else (None, None, 0)
        if oa is None:
            fora.append((luta, "casou na API mas sem preco h2h utilizavel"))
            continue
        if n_casas <= 3:
            logger.warning("%s vs %s: mediana de apenas %d casa(s) -- preco fragil.",
                           ca, cb, n_casas)
        linhas.append({"fighter_a": ca, "fighter_b": cb,
                       "odds_a_decimal": f"{oa:.3f}", "odds_b_decimal": f"{ob:.3f}",
                       "fonte": luta.fonte, "n_casas": n_casas})

    print(f"\n== CARD: {ufc['nome']} ({args.event_date}) ==")
    for r in linhas:
        marca = "  << SO NA API, confira" if r["fonte"] == "so_api" else ""
        print(f"  {r['fighter_a']:24} x {r['fighter_b']:24} "
              f"{r['odds_a_decimal']:>7} {r['odds_b_decimal']:>7} "
              f"({r['n_casas']:2} casas){marca}")
    if fora:
        print(f"\n== FORA DO CSV ({len(fora)}) -- decisao sua, nao omissao ==")
        for luta, motivo in fora:
            print(f"  {luta.fighter_a} x {luta.fighter_b}  [{luta.fonte}] -- {motivo}")
    print()

    if args.dry_run:
        logger.info("--dry-run: nada escrito.")
        return 0
    if not linhas:
        logger.error("Nenhuma luta com odds -- CSV nao escrito.")
        return 1

    saida = Path(args.out)
    saida.write_text(
        "fighter_a,fighter_b,odds_a_decimal,odds_b_decimal\n"
        + "".join(f"{r['fighter_a']},{r['fighter_b']},"
                 f"{r['odds_a_decimal']},{r['odds_b_decimal']}\n" for r in linhas),
        encoding="utf-8")
    logger.info("%d luta(s) escritas em %s", len(linhas), saida)

    if not args.sem_thunderpick:
        from src.thunderpick import DesafioDoNavegador, anexar_ao_card
        try:
            anexar_ao_card(saida, event_date=args.event_date)
        except DesafioDoNavegador as exc:
            logger.warning("%s", exc)
            logger.warning("Seguindo so com a mediana: o EV deste card NAO sera o "
                           "preco que voce pega.")

    logger.info("Agora: confira o CSV, anexe as mesmas lutas ao odds_template.csv "
                "e publique com scripts.publish_report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
