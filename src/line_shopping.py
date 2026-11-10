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

import re

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


_SEGREDO = re.compile(r"(apiKey|api_key|token)=[^&\s'\"]+", re.IGNORECASE)


def sanitizar(texto) -> str:
    """Tira a chave de qualquer texto antes de ele virar log.

    O `requests` poe a URL INTEIRA na mensagem do HTTPError, e a chave vai
    como query string. Registrar `exc` direto escrevia a chave em claro em
    data/auto_capture.log -- achado de 31/out/2026, 10 ocorrencias. Os logs
    sao gitignored, entao a chave nunca saiu da maquina, mas estava em texto
    puro no disco.
    """
    return _SEGREDO.sub(r"\1=***", str(texto))


# Ref da ULTIMA chamada bem sucedida. Quem grava preco carimba a linha com
# ela, ligando a cotacao a resposta que a produziu.
ULTIMA_REF: str | None = None
ULTIMA_CAPTURA_EM: str | None = None


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
    eventos = resp.json()
    # PROVA DA COTACAO (item 8): instante real da resposta, hash e copia do
    # corpo. A ref fica no modulo para quem gravar preco carimbar a linha.
    from src.proveniencia import _agora, registrar_chamada
    global ULTIMA_REF, ULTIMA_CAPTURA_EM
    ULTIMA_CAPTURA_EM = _agora()
    ULTIMA_REF = registrar_chamada(
        resp.text, fonte="the_odds_api",
        endpoint=f"{API_BASE}/sports/{sport_key}/odds/",
        regions=regions, markets="h2h", n_eventos=len(eventos),
        captured_at=ULTIMA_CAPTURA_EM)
    return eventos


# Nome na BASE (UFCStats) -> nome na The Odds API, para os casos em que
# nenhum score de string pode casar porque nao e grafia diferente, e NOME
# diferente. Sem isto a luta perde o sinal sharp e o fechamento em silencio: o
# relatorio sai igual a um relatorio saudavel, so sem aquela perna.
#
# Ja custou uma perna de verdade -- John Castaneda vs Alatengheili (26/set) foi
# a unica luta do card sem sharp e sem fechamento. E a terceira vez que o
# padrao aparece (Rongzhu/Zhu Rong em 12/set foi salvo pelo fallback de ordem
# invertida), entao vira tabela em vez de conserto de ocasiao.
#
# So entra par CONFIRMADO olhando os dois lados da luta. Apelido parecido nao
# basta: o risco aqui e casar duas pessoas diferentes, que e pior que nao
# casar ninguem.
APELIDOS_API = {
    "Alatengheili": "Heili Alateng",      # nome unico partido e invertido
    "Loopy Godinez": "Lupita Godinez",    # "Loopy" e apelido; Lupita e o nome
    "Timmy Cuamba": "Timothy Cuamba",     # diminutivo; 'timmy' x 'timothy' = 0.50
    "Joe Kropschot": "Joseph Kropschot",  # idem; 'joe' x 'joseph' = 0.67
}


def match_event(fighter_a: str, fighter_b: str, events: list[dict]):
    """
    SUGESTAO de evento para uma luta nossa, por correspondencia aproximada.
    Retorna (evento, nome_a_na_api, nome_b_na_api).

    NAO USE PARA APROVAR UMA SELECAO. Esta funcao pega o PRIMEIRO evento cujos
    dois nomes casem, sem conferir data, sem contar quantos candidatos havia e
    sem exigir evidencia forte de identidade. Serve para DESCOBRIR o card
    (`scripts/build_card.py`, `src/card_sources.py`), onde um palpite errado
    aparece numa revisao humana antes de virar registro.

    Para cotacao -- sharp, fechamento, EV, CLV -- use `src.casamento.casar`,
    que devolve veredito e evidencia e bloqueia com motivo explicito.
    """
    fighter_a = APELIDOS_API.get(fighter_a, fighter_a)
    fighter_b = APELIDOS_API.get(fighter_b, fighter_b)
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


def fetch_sharp_probs(fights: list[dict], regions: str = DEFAULT_REGIONS,
                      events: list[dict] | None = None, *,
                      event_date: str | None = None) -> dict:
    """
    Sinal sharp por luta, para o lado apontado pelo modelo:
    {(fighter_a, fighter_b): {"sharp_prob": p, "best_odd": o}} (ou None).

      sharp_prob = probabilidade DEVIGADA da casa sharp (Pinnacle) -- a
                   melhor estimativa publica de probabilidade real;
      best_odd   = MELHOR odd oferecida entre todas as casas.

    O sinal e `sharp_prob * best_odd > 1`: alguma casa paga acima do preco
    justo da sharp. Usar a odd mediana/registrada aqui seria erro
    sistematico -- ela embute vig, entao o produto ficaria abaixo de 1
    quase sempre e o sinal nunca apareceria.

    Congelado no PRE-REGISTRO (src/prediction_history.py): sem isso nao da
    para testar depois se as pernas com respaldo sharp se saem melhor que
    as sem (a API so serve eventos futuros; nao ha backfill).

    Nunca levanta: qualquer falha (sem chave, rede fora, evento ausente)
    vira dict vazio/None -- gravar a previsao importa mais que o extra.
    """
    # `events` ja buscado: quem chama duas vezes na mesma rodada (ex.: o
    # auto_capture, que precisa do commence_time antes de decidir capturar)
    # passa a lista e evita gastar creditos da API em dobro.
    if events is None:
        try:
            events = fetch_live_odds(get_api_key(), regions=regions)
        except (RuntimeError, requests.RequestException) as exc:
            logger.warning("Sinal sharp indisponivel (%s) -- previsoes seguem sem ele.",
                           sanitizar(exc))
            return {}

    from src.casamento import casar
    from src.proveniencia import registrar_cotacao

    out: dict = {}
    for fight in fights:
        a, b = fight["fighter_a"], fight["fighter_b"]
        side = fight["model_side"]
        c = casar(a, b, events, event_date=event_date,
                  desambiguado=bool(fight.get("desambiguado")))
        if not c.ok:
            # Bloqueio VIAJA: quem grava precisa poder dizer o motivo, e
            # `None` nao distingue "a fonte nao lista" de "dois candidatos".
            logger.info("Sem cotacao para %s vs %s: %s (candidatos: %d).",
                        a, b, c.motivo, c.candidatos)
            out[(a, b)] = {"sharp_prob": None, "bloqueio": c.motivo,
                           "candidatos": c.candidatos,
                           "evento_id": c.evento_id}
            continue
        api_side = c.nome_a if side == a else c.nome_b
        api_other = c.nome_b if side == a else c.nome_a
        prob = sharp_fair_prob(c.evento, api_side, api_other)
        prices = collect_prices(c.evento, api_side)
        if prob is None or not prices:
            out[(a, b)] = {"sharp_prob": None,
                           "bloqueio": "casa sharp nao cobre o confronto",
                           "candidatos": 1, "evento_id": c.evento_id}
            continue
        casa, melhor = max(prices.items(), key=lambda kv: kv[1])
        # Prova do preco EFETIVAMENTE usado: a melhor odd, daquela casa,
        # naquele mercado, para aquele lado, ligada ao corpo guardado.
        ref = registrar_cotacao(
            chamada_ref=ULTIMA_REF, fonte="the_odds_api", casa=casa,
            mercado="h2h", evento_id=c.evento_id,
            participante_a=c.nome_a, participante_b=c.nome_b,
            lado=api_side, odd=melhor,
            fonte_atualizada_em=_atualizada_em(c.evento, casa),
            status="listado", inicio_programado=c.commence_time or "")
        out[(a, b)] = {"sharp_prob": prob, "best_odd": melhor, "casa": casa,
                       "ref": ref, "chamada_ref": ULTIMA_REF,
                       "evento_id": c.evento_id, "mercado": "h2h",
                       "evidencia": c.evidencia,
                       "participantes": f"{c.nome_a}|{c.nome_b}",
                       "inicio_programado": c.commence_time or ""}
    return out


def _atualizada_em(event: dict, casa: str) -> str:
    """`last_update` que a fonte declara para aquela casa, ou vazio.

    A The Odds API informa isso por casa; quando nao informa, o campo fica
    vazio em vez de receber o nosso relogio no lugar -- carimbar a nossa hora
    como se fosse a da fonte seria fabricar a prova que falta."""
    for bm in event.get("bookmakers", []):
        if (bm.get("title") or bm.get("key")) == casa:
            return str(bm.get("last_update") or "")
    return ""


def build_rows(fights: list[dict], events: list[dict],
               *, event_date: str | None = None) -> list[dict]:
    """
    Uma linha de line shopping por luta: melhor preco do lado do modelo
    entre as casas + EV do modelo + sinal vs sharp.

    Isto E selecao -- a tabela existe para decidir onde entrar --, entao usa
    `casamento.casar` e nao a sugestao aproximada. Luta cujo casamento nao e
    inequivoco sai com `found: False` e o motivo em `bloqueio`, em vez de
    uma linha com preco de procedencia duvidosa.
    """
    from src.casamento import casar

    rows = []
    for fight in fights:
        side = fight["model_side"]
        other = fight["fighter_b"] if side == fight["fighter_a"] else fight["fighter_a"]
        c = casar(fight["fighter_a"], fight["fighter_b"], events,
                  event_date=event_date,
                  desambiguado=bool(fight.get("desambiguado")))
        event = c.evento if c.ok else None
        if event is None:
            rows.append({"side": side, "other": other, "prob": fight["model_side_prob"],
                         "ref_odd": fight["model_side_odds"], "found": False,
                         "bloqueio": c.motivo})
            continue
        api_side = c.nome_a if side == fight["fighter_a"] else c.nome_b
        api_other = c.nome_b if side == fight["fighter_a"] else c.nome_a
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
          f"{'livro':<18} {'EV mod':>7} {'EV sharp':>9}")
    print("-" * 92)
    for r in rows:
        if not r["found"]:
            print(f"{r['side']:<24} {r['prob']*100:>5.1f}%  (sem odds na API; "
                  f"na nossa casa: {r['ref_odd']:.2f})")
            continue
        sharp = f"{r['ev_sharp']:.3f}" if r["ev_sharp"] is not None else "—"
        flag = "  << +EV vs sharp!" if (r["ev_sharp"] or 0) > 1 else ""
        print(f"{r['side']:<24} {r['prob']*100:>5.1f}% {r['worst_odd']:>5.2f}-{r['best_odd']:<5.2f} "
              f"{r['best_odd']:>7.2f}  {r['best_book']:<18} {r['ev_model']:>7.3f} {sharp:>9}{flag}")
    print("\nLembretes: EV mod e auto-referente (backtest: mercado na frente). O sinal que "
          "importa e o 'EV sharp' > 1 (raro).\nATENCAO as escalas: 'melhor' e a melhor odd "
          "entre as casas da API, que NAO inclui a Thunderpick. A odd em que a entrada sai "
          "esta no CSV do card (odds_a_tp/odds_b_tp) e e a que o relatorio mostra -- esta "
          "tabela serve para achar preco justo, nao para decidir o tamanho da perna.")


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
