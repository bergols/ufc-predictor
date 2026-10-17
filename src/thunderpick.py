"""
src/thunderpick.py

Odds da Thunderpick — a casa onde as entradas sao feitas de verdade.

Por que existe
--------------
Ate out/2026 o CSV do card trazia a MEDIANA de 20-25 casas da The Odds API.
Isso responde "qual e o preco justo do mercado", que e a pergunta certa para
medir o modelo contra o mercado — e a pergunta errada para decidir uma
entrada. A odd que importa na hora de apostar e a da casa onde a aposta e
feita, e a Thunderpick nao esta em nenhuma regiao da The Odds API (41 casas
entre us, us2, uk, eu e au; nenhuma e ela).

As duas convivem de proposito, e cada uma tem um trabalho:

  odds_a/b_decimal    mediana das casas. Define o FAVORITO DO MERCADO, que e
                      o benchmark contra o qual o modelo e medido. Consenso
                      de 20+ casas e uma regua melhor que uma casa so.
  odds_a/b_tp         Thunderpick. E a odd mostrada no relatorio, a que entra
                      no EV, no criterio EV>1 e no P&L das pernas — porque e
                      a odd que da para pegar.

Medido no card de 17/out/2026, a diferenca nao e cosmetica: Kyle Nelson 3,40
na mediana e 3,91 na Thunderpick; Anheliger 1,86 contra 2,36; mas Koslow
1,925 contra 1,56. Pernas EV>1 mudam de lista dependendo de qual odd se usa.

Como obtem
----------
Navegador real, pelo mesmo motivo do UFCStats: o dominio inteiro responde 403
a requisicao sem navegador, API incluida. Nao resolvemos desafio nenhum —
abrimos a pagina publica de odds num navegador de verdade e lemos o que ela
carrega.

E navegador COM JANELA (headless=False). Medido em 17/out/2026: o Edge
headless recebe o desafio do Cloudflare e para na pagina "Um momento..."; o
mesmo Edge com janela entra direto, 200 e as 27 lutas. A diferenca nao e
contornada por nos e nao tentamos contorna-la — ou o navegador passa sendo o
que e, ou o CSV e preenchido a mao. Por isso tambem este passo NAO entra no
job horario: ele roda uma vez por evento, na montagem do card.

O caminho e: pagina da competicao UFC -> ids das lutas nos links -> chamadas
a /api/matches?matchesIds=... de DENTRO da pagina (mesma origem). Uma carga
cobre a competicao inteira, o que hoje sao 27 lutas, cinco eventos a frente.

A lista e VIRTUALIZADA e isso quase passou batido: o DOM guarda so as linhas
visiveis, entao ler a pagina uma vez devolve um PEDACO do catalogo. Medido em
17/out, o topo dava as 13 lutas daquela noite e o fim da pagina dava 4 de
dezembro -- com o card de 24/out, que era o proximo a montar, invisivel nas
duas pontas. Por isso a coleta rola a pagina acumulando ids, em vez de ler o
DOM uma vez. Falha aqui nao e barulhenta: devolveria um card com menos lutas
do que ele tem, e o pre-registro sairia incompleto sem ninguem notar.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import config
from src.utils import best_name_match

logger = logging.getLogger(__name__)

URL_UFC = "https://thunderpick.io/sports/martial-arts/ufc/396"
_ORIGEM = "https://thunderpick.io"

# Nome na BASE (UFCStats) -> nome na Thunderpick. Mesma ideia da APELIDOS_API
# do line_shopping, tabela separada porque as duas fontes erram diferente: a
# Thunderpick usa o sobrenome ANTIGO de quem casou e separa nomes coreanos.
# So entra par conferido olhando os dois lados da luta.
APELIDOS_TP = {
    "Katlyn Cerminara": "Katlyn Chookagian",   # nome de solteira na casa
    "SuYoung You": "Soo Young Yoo",            # romanizacao partida
    "Timmy Cuamba": "Timothy Cuamba",          # diminutivo
    "Loopy Godinez": "Lupita Godinez",         # apelido no lugar do nome
}


@dataclass(frozen=True)
class LutaTP:
    home: str
    away: str
    odd_home: float
    odd_away: float
    inicio: str

    @property
    def nomes(self) -> list[str]:
        return [self.home, self.away]


# JS rodado DENTRO da pagina: pega os ids dos links e pede as odds a propria
# API da casa. De fora da pagina a chamada e barrada; de dentro e a mesma
# requisicao que o site faz para se desenhar.
_SELETOR_LUTA = 'a[href*="/ufc/396/"]'

_JS_COLETA = """
async () => {
  const espera = ms => new Promise(r => setTimeout(r, ms));
  const SEL = 'a[href*="/ufc/396/"]';

  // Consentimento: RECUSA o nao essencial. Nunca aceita -- e a escolha mais
  // preservadora, e o aviso bloqueia a renderizacao da lista enquanto esta
  // aberto (visto em 17/out: sidebar desenhada, zero lutas).
  const recusar = [...document.querySelectorAll('button')]
      .find(b => /decline all|recusar/i.test(b.textContent || ''));
  if (recusar) { recusar.click(); await espera(1500); }

  // A lista e VIRTUALIZADA: o DOM guarda so as linhas visiveis. Uma leitura
  // unica devolve um pedaco do catalogo -- medido em 17/out, o topo dava as
  // 13 lutas da noite e o fim da pagina dava 4 de dezembro, com o card de
  // 24/out invisivel nas duas pontas. Rola de a pouco acumulando.
  const vistos = new Set();
  const colher = () => [...document.querySelectorAll(SEL)]
      .map(a => (a.getAttribute('href') || '').split('/').pop())
      .filter(x => /^[0-9]+$/.test(x))
      .forEach(x => vistos.add(x));

  window.scrollTo(0, 0);
  await espera(1200);
  colher();
  let y = 0;
  const passo = Math.max(300, window.innerHeight * 0.6);
  for (let i = 0; i < 80 && y < document.body.scrollHeight; i++) {
    y += passo;
    window.scrollTo(0, y);
    await espera(700);
    colher();
  }

  // a API aceita os ids em lote, mas nao o catalogo inteiro de uma vez
  const ids = [...vistos];
  const out = [];
  for (let i = 0; i < ids.length; i += 20) {
    const url = '/api/matches?' + ids.slice(i, i + 20).map(x => 'matchesIds=' + x).join('&');
    const j = await (await fetch(url, {headers: {accept: 'application/json'}})).json();
    for (const m of (j.data.matches || [])) {
      if (!m.market || !m.market.home || !m.market.away) continue;
      out.push({home: m.teams.home.name, away: m.teams.away.name,
                odd_home: m.market.home.odds, odd_away: m.market.away.odds,
                inicio: m.startTime, status: m.market.status});
    }
  }
  return out;
}
"""


class DesafioDoNavegador(RuntimeError):
    """A casa respondeu com o desafio do Cloudflare em vez da pagina."""


def fetch_ufc_odds(timeout_ms: int = 60000, headless: bool = False) -> list[LutaTP]:
    """Todas as lutas de UFC listadas hoje na Thunderpick, com a odd de cada
    lado. Lista vazia quando a casa nao lista nada; levanta
    DesafioDoNavegador quando a pagina nem chega a abrir — aqui nunca
    inventamos preco nem devolvemos silencio no lugar de falha."""
    from playwright.sync_api import TimeoutError as PWTimeout
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="msedge", headless=headless)
        except Exception:  # noqa: BLE001 - sem Edge, cai no chromium do Playwright
            browser = p.chromium.launch(headless=headless)
        try:
            page = browser.new_page()
            page.goto(URL_UFC, timeout=timeout_ms)
            # a lista e desenhada no cliente, entao a pagina pronta NAO basta.
            # E networkidle nao serve: a casa mantem conexao aberta para mexer
            # a odd ao vivo, entao a rede nunca fica ociosa e a espera estoura
            # sempre. O sinal certo e o proprio link da luta existir.
            try:
                page.wait_for_selector(_SELETOR_LUTA, timeout=timeout_ms)
            except PWTimeout:
                titulo = page.title()
                raise DesafioDoNavegador(
                    f"A pagina de odds nao abriu (titulo: {titulo!r}). Com "
                    f"headless=True isso e o desafio do Cloudflare, que nao "
                    f"contornamos: rode com janela, ou preencha odds_a_tp/"
                    f"odds_b_tp a mao no CSV do card.") from None
            cru = page.evaluate(_JS_COLETA)
        finally:
            browser.close()

    lutas = []
    for m in cru or []:
        # status != 1 e mercado suspenso (luta comecando, odd congelada pela
        # casa). Preco suspenso nao e preco de entrada.
        if m.get("status") != 1:
            continue
        try:
            lutas.append(LutaTP(str(m["home"]), str(m["away"]),
                                float(m["odd_home"]), float(m["odd_away"]),
                                str(m.get("inicio", ""))))
        except (KeyError, TypeError, ValueError):
            logger.warning("Entrada ilegivel da Thunderpick ignorada: %r", m)
    logger.info("Thunderpick: %d luta(s) de UFC com mercado aberto.", len(lutas))
    return lutas


def _casa(nome: str, candidatos: list[str]) -> str | None:
    alvo = APELIDOS_TP.get(nome, nome)
    if alvo in candidatos:
        return alvo
    return best_name_match(alvo, candidatos)


def match_fight(fighter_a: str, fighter_b: str,
                lutas: list[LutaTP]) -> tuple[float, float] | None:
    """
    (odd_a, odd_b) na Thunderpick para esta luta, ja na ORDEM DO NOSSO CSV.

    Devolver na nossa ordem e o ponto: a casa lista home/away por criterio
    dela, e uma odd invertida seria pior que odd nenhuma — entraria no EV
    como numero plausivel.
    """
    for luta in lutas:
        ma = _casa(fighter_a, luta.nomes)
        mb = _casa(fighter_b, luta.nomes)
        if not ma or not mb or ma == mb:
            continue
        if ma == luta.home:
            return luta.odd_home, luta.odd_away
        return luta.odd_away, luta.odd_home
    return None


def anexar_ao_card(csv_path, lutas: list[LutaTP] | None = None,
                   headless: bool = False) -> int:
    """
    Escreve `odds_a_tp` / `odds_b_tp` no CSV do card. Retorna quantas lutas
    casaram.

    As odds entram no CSV e sao CONGELADAS com o pre-registro, nunca buscadas
    na hora de gerar o relatorio: buscar ao vivo faria o EV publicado andar
    sozinho depois do registro, que e exatamente o que a serie existe para
    nao fazer.
    """
    import pandas as pd

    lutas = fetch_ufc_odds(headless=headless) if lutas is None else lutas
    df = pd.read_csv(csv_path)
    a_tp, b_tp, faltando = [], [], []
    for _, r in df.iterrows():
        par = match_fight(str(r["fighter_a"]), str(r["fighter_b"]), lutas)
        if par is None:
            faltando.append(f'{r["fighter_a"]} vs {r["fighter_b"]}')
            a_tp.append(None)
            b_tp.append(None)
        else:
            a_tp.append(par[0])
            b_tp.append(par[1])
    df["odds_a_tp"], df["odds_b_tp"] = a_tp, b_tp
    df.to_csv(csv_path, index=False)

    casadas = len(df) - len(faltando)
    logger.info("Thunderpick: %d de %d luta(s) do card com odd.", casadas, len(df))
    for f in faltando:
        logger.warning("Sem odd na Thunderpick: %s -- a perna fica com a mediana "
                       "e o EV dela nao e o preco que voce pega.", f)
    return casadas


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Anexa as odds da Thunderpick ao CSV do card.")
    parser.add_argument("csv", nargs="?", default=str(config.UPCOMING_CARD_CSV))
    parser.add_argument("--listar", action="store_true",
                        help="So mostra o que a casa lista, sem escrever nada")
    parser.add_argument("--headless", action="store_true",
                        help="Sem janela. Hoje cai no desafio do Cloudflare; "
                             "existe para o dia em que isso mudar.")
    args = parser.parse_args()

    if args.listar:
        for luta in fetch_ufc_odds(headless=args.headless):
            print(f"{luta.inicio[:10]}  {luta.home:24} {luta.odd_home:>6.2f}  x "
                  f"{luta.odd_away:>6.2f}  {luta.away}")
        sys.exit(0)
    sys.exit(0 if anexar_ao_card(args.csv, headless=args.headless) else 1)
