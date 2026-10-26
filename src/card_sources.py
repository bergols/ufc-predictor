"""
src/card_sources.py

Quais lutas estao no card, segundo as DUAS fontes — e o que fazer quando elas
discordam.

Por que existe
--------------
Ate out/2026 a regra era "UFCStats e a autoridade da ordem do card", e a lista
de lutas era escrita a mao a partir da pagina dele. A regra errou nas duas
direcoes, em eventos diferentes:

  - evento 12 (26/set): a API listava Darren Till vs Yoel Romero e Mickey Gall
    vs Sedriques Dumas, que NAO estavam no card. A regra salvou duas lutas
    fantasma.
  - evento 15 (24/out): o UFCStats estava incompleto. Bolaji Oki vs Daniel
    Santos e Kody Steele vs Noah Gugnon aconteceram e ficaram de fora do
    pre-registro, mais uma terceira que nenhuma das fontes mostrava.

Ou seja: nenhuma das duas fontes e confiavel sozinha, e a que estava mandando
nao e a mais completa. A uniao com desempate EXPLICITO e o conserto — cada
luta carrega de onde veio, e luta que so uma fonte conhece aparece marcada em
vez de sumir.

Desempate
---------
  ordem      do UFCStats, que e quem publica a sequencia do card (a primeira
             linha vira o destaque do relatorio via card_order);
  completude da API, que costuma listar antes e as vezes lista mais;
  nomes      SEMPRE da nossa base (best_name_match), nunca da fonte — e a
             grafia da base que o modelo conhece.

Luta que so uma fonte conhece nao e descartada nem aceita em silencio: sai
marcada em `fonte`, para a decisao ser de quem monta o card.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import config

logger = logging.getLogger(__name__)

URL_UPCOMING = "http://ufcstats.com/statistics/events/upcoming"

# Mesma folga do event_watch: card de sabado a noite nos EUA comeca no domingo
# em UTC, e card adiado em cima da hora muda de data sem mudar de identidade.
_TOLERANCIA_DATA_DIAS = 1


@dataclass
class LutaDoCard:
    fighter_a: str
    fighter_b: str
    fonte: str                       # "ambas" | "so_ufcstats" | "so_api"
    ordem: int | None = None         # posicao no card (UFCStats); None se so API
    nomes_api: tuple[str, str] | None = None
    avisos: list[str] = field(default_factory=list)

    @property
    def completa(self) -> bool:
        return self.fonte == "ambas"


# ---------------------------------------------------------------------------
# UFCStats: a ordem do card
# ---------------------------------------------------------------------------

def _eventos_futuros(soup) -> list[dict]:
    eventos = []
    for linha in soup.select("tr.b-statistics__table-row"):
        link = linha.select_one("a.b-link")
        if not link or not link.get("href"):
            continue
        data_tag = linha.select_one("span.b-statistics__date")
        texto = data_tag.get_text(strip=True) if data_tag else ""
        try:
            quando = datetime.strptime(texto, "%B %d, %Y").date()
        except ValueError:
            quando = None
        eventos.append({"nome": link.get_text(strip=True),
                        "url": link["href"], "data": quando})
    return eventos


def _lutas_do_evento(soup) -> list[tuple[str, str]]:
    """Confrontos na ORDEM da pagina. Card futuro nao tem resultado, entao
    nao da para reaproveitar scrape_event (que espera as colunas de metodo,
    round e tempo preenchidas)."""
    lutas = []
    for linha in soup.select("tr.b-fight-details__table-row"):
        nomes = [a.get_text(strip=True)
                 for a in linha.select("td.b-fight-details__table-col a.b-link")
                 if a.get("href", "").find("fighter-details") >= 0]
        if len(nomes) == 2:
            lutas.append((nomes[0], nomes[1]))
    return lutas


def card_do_ufcstats(event_date: str | None = None) -> dict:
    """
    {"nome", "data", "url", "lutas": [(a, b), ...]} do proximo card, ou do
    card da data pedida. Usa navegador real (o site exige JavaScript).
    """
    from src.data_collection import _get_soup, navegador_real

    with navegador_real():
        soup = _get_soup(URL_UPCOMING)
        if soup is None:
            raise RuntimeError("nao consegui abrir a lista de eventos futuros do UFCStats")
        eventos = _eventos_futuros(soup)
        if not eventos:
            raise RuntimeError("a lista de eventos futuros veio vazia")

        alvo = eventos[0]
        if event_date:
            pedida = date.fromisoformat(event_date)
            candidatos = [e for e in eventos if e["data"]
                          and abs((e["data"] - pedida).days) <= _TOLERANCIA_DATA_DIAS]
            if not candidatos:
                raise RuntimeError(
                    f"nenhum evento futuro do UFCStats em {event_date} "
                    f"(proximos: {[str(e['data']) for e in eventos[:3]]})")
            alvo = candidatos[0]

        lutas = _lutas_do_evento(_get_soup(alvo["url"]))

    logger.info("UFCStats: %s (%s), %d luta(s).", alvo["nome"], alvo["data"], len(lutas))
    return {"nome": alvo["nome"], "data": alvo["data"], "url": alvo["url"], "lutas": lutas}


# ---------------------------------------------------------------------------
# A uniao
# ---------------------------------------------------------------------------

def _data_do_evento_api(commence_time) -> date | None:
    from src.event_watch import _data_do_card
    return _data_do_card(commence_time)


def unir(lutas_ufcstats: list[tuple[str, str]], api_events: list[dict],
         event_date: str) -> list[LutaDoCard]:
    """
    Card unificado, na ordem do UFCStats, com as lutas que so a API conhece
    anexadas no fim (nao se sabe a posicao delas no card).

    O casamento entre as fontes usa o mesmo match_event do line_shopping —
    inclusive a tabela de apelidos, que ja resolveu cinco casos em que a API
    usa outro NOME e nao outra grafia.
    """
    from src.line_shopping import match_event

    pedida = date.fromisoformat(event_date)
    do_dia = [e for e in api_events
              if (d := _data_do_evento_api(e.get("commence_time"))) is not None
              and abs((d - pedida).days) <= _TOLERANCIA_DATA_DIAS]

    usados, card = set(), []
    for i, (a, b) in enumerate(lutas_ufcstats):
        # cada entrada da API serve a UMA luta: sem isto, dois confrontos com
        # nomes parecidos poderiam casar no mesmo evento e herdar a mesma odd
        disponiveis = [e for e in do_dia if id(e) not in usados]
        ev, api_a, api_b = match_event(a, b, disponiveis)
        if ev is None:
            card.append(LutaDoCard(a, b, "so_ufcstats", ordem=i,
                                   avisos=["nenhuma casa da API precificou"]))
            continue
        usados.add(id(ev))
        card.append(LutaDoCard(a, b, "ambas", ordem=i, nomes_api=(api_a, api_b)))

    for e in do_dia:
        if id(e) in usados:
            continue
        a, b = str(e.get("home_team") or ""), str(e.get("away_team") or "")
        card.append(LutaDoCard(a, b, "so_api", ordem=None, nomes_api=(a, b),
                               avisos=["nao esta no card do UFCStats; "
                                       "confira antes de registrar"]))
    return card


def resumir(card: list[LutaDoCard]) -> str:
    por_fonte: dict[str, int] = {}
    for luta in card:
        por_fonte[luta.fonte] = por_fonte.get(luta.fonte, 0) + 1
    partes = [f"{n} {k}" for k, n in sorted(por_fonte.items())]
    return f"{len(card)} luta(s): " + ", ".join(partes)
