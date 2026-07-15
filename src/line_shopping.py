"""
src/line_shopping.py

Line shopping das pernas do card: consulta odds ao vivo de varias casas
(The Odds API) e mostra, para cada perna EV>1 do modelo, onde o preco
esta melhor — mais a referencia "sharp" da Pinnacle (probabilidade
devigada) para sinalizar quando alguma casa paga acima do preco justo.

Duas leituras DIFERENTES na tabela, com pesos diferentes:

  - EV do MODELO (p_modelo x melhor odd): auto-referente — assume que o
    modelo esta certo, e o backtest mostra o mercado na frente. Serve
    para o paper trading, nao como evidencia de valor real.
  - Sinal vs SHARP (p_pinnacle_devig x melhor odd): metodo com base real
    — se uma casa recreativa paga acima do justo da Pinnacle, ha valor
    documentavel independente do nosso modelo. Raro e pequeno quando
    aparece; e o que profissionais de fato usam.

Limitacoes: a The Odds API NAO cobre as casas licenciadas no Brasil
(.bet.br) — os precos aqui sao das versoes internacionais das marcas
(referencia proxima, nao identica). Confira o preco na sua casa antes de
qualquer aposta. Chave gratuita: ~500 creditos/mes (cada regiao
consultada custa 1 credito por chamada); a chave fica FORA do git em
data/raw/odds_api_key.txt ou na env ODDS_API_KEY.

Uso:
    python -m src.line_shopping data/raw/upcoming_card_odds.csv
    python -m src.line_shopping data/raw/upcoming_card_odds.csv --all
"""
from __future__ import annotations

import argparse
import logging
import os

import requests

import config
from src.utils import best_name_match, decimal_odds_to_implied_prob, remove_vig_two_way

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY_MMA = "mma_mixed_martial_arts"
DEFAULT_REGIONS = "eu,uk,us"
KEY_FILE = config.RAW_DIR / "odds_api_key.txt"
SHARP_BOOK = "Pinnacle"


def get_api_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if key:
        return key
    if KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            return key
    raise RuntimeError(
        "Chave da The Odds API nao encontrada. Defina a variavel ODDS_API_KEY ou "
        f"salve a chave em {KEY_FILE} (arquivo fora do git). Chave gratuita em the-odds-api.com.")


def fetch_live_odds(api_key: str, sport_key: str = SPORT_KEY_MMA,
                    regions: str = DEFAULT_REGIONS) -> list[dict]:
    """Uma chamada por execucao: todos os eventos futuros do esporte, com
    todas as casas das regioes pedidas (h2h/moneyline, decimal)."""
    resp = requests.get(
        f"{API_BASE}/sports/{sport_key}/odds/",
        params={"apiKey": api_key, "regions": regions, "markets": "h2h",
                "oddsFormat": "decimal"},
        timeout=20)
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining")
    if remaining is not None:
        logger.info("Creditos The Odds API restantes neste mes: %s", remaining)
    return resp.json()


def match_event(fighter_a: str, fighter_b: str, events: list[dict]):
    """Casa uma luta nossa com um evento da API (fuzzy nos dois nomes, em
    qualquer ordem). Retorna (evento, nome_a_na_api, nome_b_na_api)."""
    for ev in events:
        names = [ev.get("home_team") or "", ev.get("away_team") or ""]
        match_a = fighter_a if fighter_a in names else best_name_match(fighter_a, names)
        match_b = fighter_b if fighter_b in names else best_name_match(fighter_b, names)
        if match_a and match_b and match_a != match_b:
            return ev, match_a, match_b
    return None, None, None


def collect_prices(event: dict, fighter_api_name: str) -> dict[str, float]:
    """Odd decimal do lutador em cada casa: {nome_da_casa: odd}."""
    prices: dict[str, float] = {}
    for bm in event.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                if outcome.get("name") == fighter_api_name:
                    prices[bm.get("title") or bm.get("key")] = float(outcome["price"])
    return prices


def sharp_fair_prob(event: dict, side_name: str, other_name: str) -> float | None:
    """Probabilidade devigada do lado `side_name` segundo a casa sharp
    (Pinnacle). None se a Pinnacle nao cobre o evento."""
    side_prices = collect_prices(event, side_name)
    other_prices = collect_prices(event, other_name)
    if SHARP_BOOK not in side_prices or SHARP_BOOK not in other_prices:
        return None
    prob_side, _ = remove_vig_two_way(
        decimal_odds_to_implied_prob(side_prices[SHARP_BOOK]),
        decimal_odds_to_implied_prob(other_prices[SHARP_BOOK]))
    return prob_side


def build_rows(fights: list[dict], events: list[dict]) -> list[dict]:
    """Uma linha de line shopping por luta: melhor preco do lado do modelo
    entre as casas + EV do modelo + sinal vs sharp."""
    rows = []
    for fight in fights:
        side = fight["model_side"]
        other = fight["fighter_b"] if side == fight["fighter_a"] else fight["fighter_a"]
        event, *api_names = match_event(fight["fighter_a"], fight["fighter_b"], events)
        if event is None:
            rows.append({"side": side, "other": other, "prob": fight["model_side_prob"],
                         "ref_odd": fight["model_side_odds"], "found": False})
            continue
        api_a, api_b = api_names
        api_side = api_a if side == fight["fighter_a"] else api_b
        api_other = api_b if side == fight["fighter_a"] else api_a
        prices = collect_prices(event, api_side)
        if not prices:
            rows.append({"side": side, "other": other, "prob": fight["model_side_prob"],
                         "ref_odd": fight["model_side_odds"], "found": False})
            continue
        best_book, best_odd = max(prices.items(), key=lambda kv: kv[1])
        fair = sharp_fair_prob(event, api_side, api_other)
        rows.append({
            "side": side, "other": other, "found": True,
            "prob": fight["model_side_prob"],
            "ref_odd": fight["model_side_odds"],
            "best_odd": best_odd, "best_book": best_book,
            "n_books": len(prices),
            "worst_odd": min(prices.values()),
            "ev_model": fight["model_side_prob"] * best_odd,
            "sharp_prob": fair,
            "ev_sharp": (fair * best_odd) if fair is not None else None,
        })
    return rows


def print_table(rows: list[dict]) -> None:
    print(f"\n{'perna':<24} {'p_mod':>6} {'faixa odds':>12} {'melhor':>7}  "
          f"{'casa':<18} {'EV mod':>7} {'EV sharp':>9}")
    print("-" * 92)
    for r in rows:
        if not r["found"]:
            print(f"{r['side']:<24} {r['prob']*100:>5.1f}%  (sem odds na API — conferir manualmente; "
                  f"ref {r['ref_odd']:.2f})")
            continue
        sharp = f"{r['ev_sharp']:.3f}" if r["ev_sharp"] is not None else "—"
        flag = "  << +EV vs sharp!" if (r["ev_sharp"] or 0) > 1 else ""
        print(f"{r['side']:<24} {r['prob']*100:>5.1f}% {r['worst_odd']:>5.2f}-{r['best_odd']:<5.2f} "
              f"{r['best_odd']:>7.2f}  {r['best_book']:<18} {r['ev_model']:>7.3f} {sharp:>9}{flag}")
    print("\nLembretes: EV mod e auto-referente (backtest: mercado na frente). O sinal que "
          "importa e o 'EV sharp' > 1 (raro). Casas .bet.br nao estao na API — confira o "
          "preco na sua casa antes de entrar; use esta tabela como referencia do preco justo.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Line shopping das pernas do card em casas internacionais (The Odds API).")
    parser.add_argument("csv", type=str, help="CSV do card (data/raw/upcoming_card_odds.csv)")
    parser.add_argument("--model", choices=["logreg", "gbm"], default="logreg")
    parser.add_argument("--all", action="store_true",
                        help="Todas as lutas com previsao (default: so pernas EV>1)")
    parser.add_argument("--regions", type=str, default=DEFAULT_REGIONS,
                        help=f"Regioes da API (default: {DEFAULT_REGIONS}; cada regiao = 1 credito/chamada)")
    args = parser.parse_args()

    from src.card_report import analyze_card, load_card_odds
    analysis = analyze_card(load_card_odds(args.csv), model_name=args.model)
    fights = ((analysis["favorites"] + analysis["underdogs"]) if args.all
              else analysis["ev_legs"])
    if not fights:
        print("Nenhuma perna EV>1 neste card (use --all para ver todas as lutas).")
        return 0

    events = fetch_live_odds(get_api_key(), regions=args.regions)
    rows = build_rows(fights, events)
    rows.sort(key=lambda r: r.get("ev_model", 0), reverse=True)
    print_table(rows)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
