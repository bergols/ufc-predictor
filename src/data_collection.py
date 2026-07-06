"""
src/data_collection.py

Responsavel por obter os dados brutos de lutas e lutadores do UFC.

Duas estrategias, nesta ordem de preferencia:

1. Scraping direto do UFCStats.com (dados publicos, atualizados,
   granulares por luta). Implementado com requests + BeautifulSoup.
2. Fallback: carregar um dataset publico ja compilado (ex.: o dataset
   "UFC-Fight historical data" de rajeevw, disponivel no Kaggle e
   replicado em varios repositorios GitHub), caso o scraping falhe
   (site fora do ar, mudanca de layout, bloqueio, etc.) ou voce
   prefira comecar mais rapido sem fazer scraping.

Os dados brutos sao salvos em CSV (data/raw/) e, opcionalmente, tambem
numa base SQLite unica (mais facil de consultar depois com SQL).

IMPORTANTE sobre uso responsavel do scraper:
- Ha uma pausa (config.SCRAPER_DELAY_SECONDS) entre requisicoes.
- Rode o scraping completo raramente (ex.: uma vez, depois so
  incrementalmente para eventos novos), nao a cada execucao do pipeline.
- Respeite os termos de uso do UFCStats.com.

NOTA DE MANUTENCAO: o scraper depende da estrutura HTML atual do
UFCStats.com (nomes de classes CSS). Sites mudam de layout de vez em
quando; se o scraping comecar a retornar DataFrames vazios, o primeiro
passo de debug e abrir uma pagina de evento/luta/lutador no navegador,
inspecionar o HTML e ajustar os seletores CSS abaixo.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import config
from src.utils import parse_date, parse_height_to_cm, parse_pct, parse_reach_to_cm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "http://ufcstats.com"
EVENTS_URL = f"{BASE_URL}/statistics/events/completed?page=all"
FIGHTERS_INDEX_URL = f"{BASE_URL}/statistics/fighters?char={{letter}}&page=all"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": config.SCRAPER_USER_AGENT})


# Quando setado, substitui o fetch via requests em _get_soup. Usado pelo
# gap-filler com navegador real (fill_recent_gap_with_browser), ja que o
# UFCStats.com exige execucao de JavaScript e o requests nao passa do gate.
_SOUP_FETCHER = None


def _get_soup(url: str) -> Optional[BeautifulSoup]:
    """Baixa uma URL e devolve o HTML parseado. Retorna None em erro (nao derruba o scraping inteiro)."""
    if _SOUP_FETCHER is not None:
        return _SOUP_FETCHER(url)
    try:
        resp = SESSION.get(url, timeout=15)
        resp.raise_for_status()
        time.sleep(config.SCRAPER_DELAY_SECONDS)
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as exc:
        logger.warning("Falha ao baixar %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# 1) Lista de eventos e lutas
# ---------------------------------------------------------------------------

def list_event_urls() -> list[str]:
    """Retorna a URL de todos os eventos ja realizados, do mais antigo ao mais recente."""
    soup = _get_soup(EVENTS_URL)
    if soup is None:
        return []
    urls = []
    for row in soup.select("tr.b-statistics__table-row"):
        link = row.select_one("a.b-link")
        if link and link.get("href"):
            urls.append(link["href"])
    return list(reversed(urls))  # mais antigo primeiro, ajuda a retomar/debugar


def scrape_event(event_url: str) -> list[dict]:
    """
    Extrai as lutas de um evento: nomes dos lutadores, resultado, metodo,
    round, tempo, categoria de peso, e a URL de detalhes da luta (usada
    depois para pegar os totais estatisticos por lutador).
    """
    soup = _get_soup(event_url)
    if soup is None:
        return []

    event_name_tag = soup.select_one("h2.b-content__title")
    event_name = event_name_tag.get_text(strip=True) if event_name_tag else None

    event_date = None
    for item in soup.select("li.b-list__box-list-item"):
        text = item.get_text(" ", strip=True)
        if text.startswith("Date:"):
            event_date = parse_date(text.replace("Date:", "").strip())
            break

    fights = []
    for row in soup.select("tr.b-fight-details__table-row"):
        fight_link_tag = row.select_one("a.b-flag") or row.select_one("a[href*='fight-details']")
        fight_url = fight_link_tag["href"] if fight_link_tag and fight_link_tag.get("href") else None

        cols = row.select("td.b-fight-details__table-col")
        if len(cols) < 10:
            continue

        fighter_links = cols[1].select("a")
        fighter_names = [a.get_text(strip=True) for a in fighter_links]
        if len(fighter_names) != 2:
            continue
        fighter_urls = [a.get("href") for a in fighter_links]

        weight_class = cols[6].get_text(strip=True)
        method = cols[7].get_text(" ", strip=True)
        round_num = cols[8].get_text(strip=True)
        time_str = cols[9].get_text(strip=True)

        # A primeira coluna tem UM flag por luta: "win" (verde) quando houve
        # vencedor -- e nesse caso o vencedor e sempre o primeiro nome listado
        # em cols[1]. Em empate/no-contest os flags dizem "draw"/"nc" e o
        # winner fica None (a luta e descartada depois, em features.py).
        flag_texts = [a.get_text(strip=True).lower() for a in cols[0].select("a.b-flag")]
        winner_name = fighter_names[0] if any(t == "win" for t in flag_texts) else None

        fights.append({
            "event_name": event_name,
            "event_date": event_date,
            "event_url": event_url,
            "fight_url": fight_url,
            "fighter_1": fighter_names[0],
            "fighter_2": fighter_names[1],
            "fighter_1_url": fighter_urls[0],
            "fighter_2_url": fighter_urls[1],
            "winner": winner_name,
            "weight_class": weight_class,
            "method": method,
            "round": round_num,
            "time": time_str,
        })
    return fights


def scrape_fight_totals(fight_url: str) -> Optional[dict]:
    """
    Extrai as estatisticas TOTAIS (agregado da luta toda, nao por round) de
    uma luta: golpes significativos acertados/tentados, quedas (takedowns)
    acertadas/tentadas, tempo de controle, etc, para cada um dos dois
    lutadores. Essas estatisticas alimentam o calculo de medias
    "ponto-no-tempo" (point-in-time) feito em features.py.
    """
    soup = _get_soup(fight_url)
    if soup is None:
        return None

    # CUIDADO com o seletor: a pagina tem 4 tabelas -- "Totals" (luta inteira,
    # SEM classe CSS, 1 linha no tbody), "Totals per round" (classe
    # b-fight-details__table js-fight-table, 1 linha POR ROUND), e as duas
    # equivalentes de significant strikes. Selecionar
    # "table.b-fight-details__table" pegava a tabela PER-ROUND e lia so o
    # round 1 como se fosse o total da luta. A tabela de totais e a primeira
    # do documento.
    table = soup.find("table")
    if table is None:
        return None

    rows = table.select("tbody tr")
    if not rows:
        return None
    cells = rows[0].select("td")
    if len(cells) < 10:
        return None

    def col_values(idx: int) -> list[str]:
        return [p.get_text(strip=True) for p in cells[idx].select("p")]

    fighters = col_values(0)
    kd = col_values(1)
    sig_str = col_values(2)        # ex.: "45 of 90"
    sig_str_pct = col_values(3)
    td = col_values(5)             # ex.: "2 of 5"
    td_pct = col_values(6)
    sub_att = col_values(7)
    rev = col_values(8)
    ctrl = col_values(9)           # ex.: "3:24"

    def split_of(value: str):
        if not value or " of " not in value:
            return None, None
        landed, attempted = value.split(" of ")
        try:
            return int(landed), int(attempted)
        except ValueError:
            return None, None

    def ctrl_to_seconds(value: str) -> Optional[int]:
        if not value or ":" not in value:
            return None
        mins, secs = value.split(":")
        try:
            return int(mins) * 60 + int(secs)
        except ValueError:
            return None

    result = []
    for i in range(min(2, len(fighters))):
        sig_landed, sig_attempted = split_of(sig_str[i]) if i < len(sig_str) else (None, None)
        td_landed, td_attempted = split_of(td[i]) if i < len(td) else (None, None)
        result.append({
            "fighter": fighters[i],
            "fight_url": fight_url,
            "knockdowns": int(kd[i]) if i < len(kd) and kd[i].isdigit() else None,
            "sig_strikes_landed": sig_landed,
            "sig_strikes_attempted": sig_attempted,
            "sig_strike_pct": parse_pct(sig_str_pct[i]) if i < len(sig_str_pct) else None,
            "takedowns_landed": td_landed,
            "takedowns_attempted": td_attempted,
            "takedown_pct": parse_pct(td_pct[i]) if i < len(td_pct) else None,
            "sub_attempts": int(sub_att[i]) if i < len(sub_att) and sub_att[i].isdigit() else None,
            "reversals": int(rev[i]) if i < len(rev) and rev[i].isdigit() else None,
            "control_seconds": ctrl_to_seconds(ctrl[i]) if i < len(ctrl) else None,
        })
    return {"fight_url": fight_url, "stats": result}


# ---------------------------------------------------------------------------
# 2) Perfis de lutadores (bio: altura, alcance, idade, stance)
# ---------------------------------------------------------------------------

def list_fighter_urls() -> list[str]:
    """Percorre o indice de lutadores (A-Z) e retorna as URLs de todos os perfis."""
    urls: list[str] = []
    for letter in "abcdefghijklmnopqrstuvwxyz":
        soup = _get_soup(FIGHTERS_INDEX_URL.format(letter=letter))
        if soup is None:
            continue
        for link in soup.select("a.b-link.b-link_style_black"):
            href = link.get("href")
            if href and href not in urls:
                urls.append(href)
    return urls


def scrape_fighter_profile(fighter_url: str) -> Optional[dict]:
    """
    Extrai bio (altura, alcance, data de nascimento, stance) e as medias de
    carreira "como estao HOJE" (career-to-date no momento do scrape).

    ATENCAO: as medias de carreira aqui sao acumuladas ate a data do
    scrape, portanto NAO devem ser usadas como feature de uma luta
    passada especifica (isso vazaria informacao do futuro para o
    modelo). Elas servem so como contexto/fallback. As features de
    treino de verdade usam as medias "ponto-no-tempo" calculadas em
    features.py a partir do historico luta-a-luta (fight_stats.csv).
    """
    soup = _get_soup(fighter_url)
    if soup is None:
        return None

    name_tag = soup.select_one("span.b-content__title-highlight")
    name = name_tag.get_text(strip=True) if name_tag else None

    bio_items = {}
    for item in soup.select("ul.b-list__box-list li.b-list__box-list-item"):
        text = item.get_text(" ", strip=True)
        if ":" in text:
            key, _, val = text.partition(":")
            bio_items[key.strip()] = val.strip()

    def to_float(val: Optional[str]) -> Optional[float]:
        if val in (None, "", "--", "---"):
            return None
        try:
            return float(val)
        except ValueError:
            return None

    return {
        "fighter_url": fighter_url,
        "name": name,
        "height_cm": parse_height_to_cm(bio_items.get("Height")),
        "reach_cm": parse_reach_to_cm(bio_items.get("Reach")),
        "stance": bio_items.get("STANCE"),
        "dob": parse_date(bio_items.get("DOB")),
        "career_slpm": to_float(bio_items.get("SLpM")),
        "career_str_acc": parse_pct(bio_items.get("Str. Acc.")),
        "career_sapm": to_float(bio_items.get("SApM")),
        "career_str_def": parse_pct(bio_items.get("Str. Def")),
        "career_td_avg": to_float(bio_items.get("TD Avg.")),
        "career_td_acc": parse_pct(bio_items.get("TD Acc.")),
        "career_td_def": parse_pct(bio_items.get("TD Def.")),
        "career_sub_avg": to_float(bio_items.get("Sub. Avg.")),
    }


# ---------------------------------------------------------------------------
# 3) Orquestracao do scraping completo
# ---------------------------------------------------------------------------

def run_full_scrape(limit_events: Optional[int] = None) -> None:
    """
    Executa o scraping completo: eventos -> lutas -> stats por luta -> lutadores.
    Salva tudo em CSV ao final de cada etapa.

    limit_events: util para testar rapido numa amostra pequena antes de
    rodar o scraping completo (que pode levar horas, dado o rate limiting
    deliberado de 1 req/segundo).
    """
    logger.info("Buscando lista de eventos...")
    event_urls = list_event_urls()
    if limit_events:
        # Pega os MAIS RECENTES: mais representativos do layout atual do site
        # e dos lutadores ativos (util para smoke test).
        event_urls = event_urls[-limit_events:]
    logger.info("%d eventos encontrados.", len(event_urls))

    all_fights = []
    all_fight_stats = []
    for i, event_url in enumerate(event_urls, 1):
        logger.info("[%d/%d] Evento: %s", i, len(event_urls), event_url)
        fights = scrape_event(event_url)
        all_fights.extend(fights)
        for fight in fights:
            if not fight["fight_url"]:
                continue
            stats = scrape_fight_totals(fight["fight_url"])
            if stats:
                all_fight_stats.extend(stats["stats"])

    fights_df = pd.DataFrame(all_fights)
    fight_stats_df = pd.DataFrame(all_fight_stats)
    fights_df.to_csv(config.RAW_FIGHTS_CSV, index=False)
    fight_stats_df.to_csv(config.RAW_FIGHT_STATS_CSV, index=False)
    logger.info("Salvo: %s (%d linhas), %s (%d linhas)",
                config.RAW_FIGHTS_CSV, len(fights_df), config.RAW_FIGHT_STATS_CSV, len(fight_stats_df))

    logger.info("Buscando lista de lutadores...")
    # Raspa apenas os perfis dos lutadores que aparecem nas lutas coletadas
    # (as paginas de evento ja trazem o link de cada perfil). Isso evita
    # percorrer o indice A-Z inteiro (milhares de requisicoes) -- especialmente
    # importante quando limit_events esta ativo para um teste rapido.
    fighter_urls = sorted({
        url
        for fight in all_fights
        for url in (fight.get("fighter_1_url"), fight.get("fighter_2_url"))
        if url
    })
    if not fighter_urls:
        logger.warning("Nenhuma URL de lutador nas lutas coletadas; caindo para o indice A-Z completo.")
        fighter_urls = list_fighter_urls()
    logger.info("%d lutadores encontrados.", len(fighter_urls))
    fighters = []
    for i, url in enumerate(fighter_urls, 1):
        profile = scrape_fighter_profile(url)
        if profile:
            fighters.append(profile)
        if i % 50 == 0:
            logger.info("  ... %d/%d lutadores processados", i, len(fighter_urls))

    fighters_df = pd.DataFrame(fighters)
    fighters_df.to_csv(config.RAW_FIGHTERS_CSV, index=False)
    logger.info("Salvo: %s (%d linhas)", config.RAW_FIGHTERS_CSV, len(fighters_df))

    save_raw_to_sqlite(fights_df, fight_stats_df, fighters_df)


def save_raw_to_sqlite(fights_df: pd.DataFrame, fight_stats_df: pd.DataFrame, fighters_df: pd.DataFrame) -> None:
    """Salva as tres tabelas brutas tambem numa base SQLite unica, para consultas ad-hoc via SQL."""
    with sqlite3.connect(config.SQLITE_DB_PATH) as conn:
        fights_df.to_sql("fights", conn, if_exists="replace", index=False)
        fight_stats_df.to_sql("fight_stats", conn, if_exists="replace", index=False)
        fighters_df.to_sql("fighters", conn, if_exists="replace", index=False)
    logger.info("Base SQLite atualizada em %s", config.SQLITE_DB_PATH)


# ---------------------------------------------------------------------------
# Verificacao de frescor dos dados coletados
# ---------------------------------------------------------------------------

def check_data_freshness(max_gap_days: Optional[int] = None) -> Optional[int]:
    """
    Compara a data do evento mais recente nos dados brutos disponiveis com a
    data de hoje e emite um WARNING bem visivel se o gap passar de
    config.DATA_FRESHNESS_MAX_GAP_DAYS.

    Motivacao: nenhuma fonte gratuita testada garante estar sempre em dia --
    o job do espelho GitHub, por exemplo, parou silenciosamente em mai/2026 e
    so foi percebido semanas depois, manualmente. Este check faz o pipeline
    reclamar sozinho.

    Retorna o gap em dias (ou None se nao ha dado nenhum).
    """
    max_gap_days = max_gap_days if max_gap_days is not None else config.DATA_FRESHNESS_MAX_GAP_DAYS

    latest = None
    if config.RAW_FIGHTS_CSV.exists():
        dates = pd.read_csv(config.RAW_FIGHTS_CSV, usecols=["event_date"],
                            parse_dates=["event_date"])["event_date"]
        latest = dates.max()
    elif config.PUBLIC_DATASET_CSV.exists():
        df = pd.read_csv(config.PUBLIC_DATASET_CSV)
        date_col = next((c for c in df.columns if c.lower() == "date"), None)
        if date_col:
            latest = pd.to_datetime(df[date_col], errors="coerce").max()

    if latest is None or pd.isna(latest):
        logger.warning("Verificacao de frescor: nenhum dado bruto encontrado.")
        return None

    gap_days = (pd.Timestamp.now().normalize() - latest.normalize()).days
    if gap_days > max_gap_days:
        logger.warning(
            "=== DADOS DESATUALIZADOS === Evento mais recente nos dados: %s (%d dias atras; "
            "limite configurado: %d). A fonte automatica provavelmente estagnou. Opcoes: "
            "tentar outra fonte (--source), preencher data/raw/manual_recent_fights.csv "
            "(ver README, secao 'Frescor dos dados'), ou seguir ciente de que o modelo "
            "nao viu as lutas mais recentes.",
            latest.date(), gap_days, max_gap_days,
        )
    else:
        logger.info("Frescor dos dados OK: evento mais recente em %s (%d dias atras).",
                    latest.date(), gap_days)
    return gap_days


# ---------------------------------------------------------------------------
# 4) Espelho GitHub atualizado diariamente (Greco1899/scrape_ufc_stats)
# ---------------------------------------------------------------------------
#
# O repositorio GPL-3.0 Greco1899/scrape_ufc_stats roda um scraper proprio do
# UFCStats.com diariamente (GCP Cloud Run + Cloud Scheduler) e commita os CSVs
# atualizados no proprio repo. Como o UFCStats.com ao vivo esta atras de um
# gate anti-bot (ver nota no topo deste arquivo), esse espelho e hoje a melhor
# fonte de dados RECENTES para este projeto.
#
# A licenca GPL-3.0 cobre o CODIGO daquele repositorio; os CSVs sao dados
# factuais compilados do UFCStats.com. Aqui apenas baixamos os CSVs (nao
# usamos o codigo deles). Atribuicao no README.
#
# MANUTENCAO: esta fonte depende do job automatizado de um terceiro continuar
# rodando. Se os arquivos pararem de atualizar, caia para o dataset publico do
# Kaggle (--source public-dataset) ou reavalie o scraping direto.

GITHUB_MIRROR_BASE_URL = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/"
GITHUB_MIRROR_FILES = [
    "ufc_event_details.csv",
    "ufc_fight_details.csv",
    "ufc_fight_results.csv",
    "ufc_fight_stats.csv",
    "ufc_fighter_details.csv",
    "ufc_fighter_tott.csv",
]


def _download_github_mirror_files() -> None:
    config.GITHUB_MIRROR_DIR.mkdir(parents=True, exist_ok=True)
    for filename in GITHUB_MIRROR_FILES:
        url = GITHUB_MIRROR_BASE_URL + filename
        logger.info("Baixando %s ...", url)
        resp = SESSION.get(url, timeout=120)
        resp.raise_for_status()
        (config.GITHUB_MIRROR_DIR / filename).write_bytes(resp.content)


def _split_of_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Vetorizado: converte strings "X of Y" em duas Series numericas (landed, attempted)."""
    extracted = series.astype(str).str.extract(r"(\d+)\s+of\s+(\d+)")
    return pd.to_numeric(extracted[0], errors="coerce"), pd.to_numeric(extracted[1], errors="coerce")


def _ctrl_to_seconds_series(series: pd.Series) -> pd.Series:
    """Vetorizado: converte tempo de controle "m:ss" em segundos ("--"/vazio vira NaN)."""
    extracted = series.astype(str).str.extract(r"^(\d+):(\d{2})$")
    return pd.to_numeric(extracted[0], errors="coerce") * 60 + pd.to_numeric(extracted[1], errors="coerce")


def parse_scheduled_rounds(time_format) -> Optional[int]:
    """
    Converte o "TIME FORMAT" do UFCStats em numero de rounds AGENDADOS
    (3 na maioria; 5 em main events/titulo). Formatos modernos: "3 Rnd
    (5-5-5)", "5 Rnd (5-5-5-5-5)". Formatos antigos com overtime ("1 Rnd +
    OT (12-3)") ou sem limite ("No Time Limit") viram None -- nao ha
    "round agendado" comparavel na era moderna (~220 lutas dos anos 90).
    """
    if time_format is None or (isinstance(time_format, float) and np.isnan(time_format)):
        return None
    match = re.match(r"^\s*(\d+)\s+Rnd\s+\([\d\-]+\)\s*$", str(time_format))
    return int(match.group(1)) if match else None


def convert_github_mirror_to_canonical(src_dir=None) -> None:
    """
    Converte os 6 CSVs do espelho GitHub para o formato canonico "scrape"
    (fights.csv / fight_stats.csv / fighters.csv) que o resto do pipeline ja
    entende -- assim features.py::build_features_from_scrape reaproveita o
    calculo point-in-time correto, em vez do adaptador "best effort" do
    dataset largo.

    Particularidades do schema de origem (verificadas em execucao real):
      - ufc_fight_stats.csv e POR ROUND -> agregamos somando os rounds.
      - ufc_fight_stats.csv nao tem URL da luta -> mapeamos via EVENT+BOUT
        usando ufc_fight_details.csv.
      - As chaves EVENT/BOUT tem espacos em branco inconsistentes entre os
        arquivos (ex.: EVENT com espaco a direita em fight_results) -> strip
        em todas antes de qualquer join.
      - OUTCOME e "W/L" (primeiro do BOUT venceu), "L/W", "D/D" ou "NC/NC".
        Empate/no-contest vira winner NaN (nunca string vazia!) e e
        descartado depois em features.py.
    """
    src_dir = src_dir or config.GITHUB_MIRROR_DIR

    results = pd.read_csv(src_dir / "ufc_fight_results.csv")
    events = pd.read_csv(src_dir / "ufc_event_details.csv")
    details = pd.read_csv(src_dir / "ufc_fight_details.csv")
    stats = pd.read_csv(src_dir / "ufc_fight_stats.csv")
    tott = pd.read_csv(src_dir / "ufc_fighter_tott.csv")

    for df, cols in ((results, ["EVENT", "BOUT"]), (events, ["EVENT"]),
                     (details, ["EVENT", "BOUT"]), (stats, ["EVENT", "BOUT", "FIGHTER"]),
                     (tott, ["FIGHTER"])):
        for col in cols:
            df[col] = df[col].astype(str).str.strip()

    # --- fights.csv -------------------------------------------------------
    events = events.rename(columns={"URL": "event_url"})
    events["event_date"] = pd.to_datetime(events["DATE"].str.strip(), format="%B %d, %Y", errors="coerce")

    fights = results.merge(events[["EVENT", "event_date", "event_url"]], on="EVENT", how="left")

    name_parts = fights["BOUT"].str.split(" vs. ", n=1, regex=False)
    fights["fighter_1"] = name_parts.str[0].str.strip()
    fights["fighter_2"] = name_parts.str[1].str.strip()

    outcome = fights["OUTCOME"].astype(str).str.strip().str.upper()
    fights["winner"] = np.where(outcome == "W/L", fights["fighter_1"],
                        np.where(outcome == "L/W", fights["fighter_2"], None))
    n_no_winner = fights["winner"].isna().sum()

    n_no_date = fights["event_date"].isna().sum()
    if n_no_date:
        # Sem data nao da para ordenar temporalmente (essencial contra
        # vazamento) -- melhor descartar do que adivinhar.
        logger.warning("%d luta(s) sem data de evento -- descartadas.", n_no_date)
        fights = fights.dropna(subset=["event_date"])

    fights_out = pd.DataFrame({
        "event_name": fights["EVENT"],
        "event_date": fights["event_date"],
        "event_url": fights["event_url"],
        "fight_url": fights["URL"],
        "fighter_1": fights["fighter_1"],
        "fighter_2": fights["fighter_2"],
        "winner": fights["winner"],
        "weight_class": fights["WEIGHTCLASS"],
        "method": fights["METHOD"],
        "round": fights["ROUND"],
        "time": fights["TIME"],
        # rounds agendados (3/5), usado pela previsao de duracao; NaN em
        # formatos antigos com overtime e nas lutas vindas do --fill-gap
        # (a pagina de evento nao expoe o time format)
        "scheduled_rounds": fights["TIME FORMAT"].map(parse_scheduled_rounds),
    })

    # --- fight_stats.csv (agrega os rounds em totais por luta) -------------
    stats = stats.dropna(subset=["ROUND"])  # descarta linhas-lixo sem round

    sig_landed, sig_attempted = _split_of_series(stats["SIG.STR."])
    td_landed, td_attempted = _split_of_series(stats["TD"])
    per_round = pd.DataFrame({
        "EVENT": stats["EVENT"],
        "BOUT": stats["BOUT"],
        "fighter": stats["FIGHTER"],
        "sig_strikes_landed": sig_landed,
        "sig_strikes_attempted": sig_attempted,
        "takedowns_landed": td_landed,
        "takedowns_attempted": td_attempted,
        "knockdowns": pd.to_numeric(stats["KD"], errors="coerce"),
        "sub_attempts": pd.to_numeric(stats["SUB.ATT"], errors="coerce"),
        "reversals": pd.to_numeric(stats["REV."], errors="coerce"),
        "control_seconds": _ctrl_to_seconds_series(stats["CTRL"]),
    })
    totals = (per_round
              .groupby(["EVENT", "BOUT", "fighter"], as_index=False)
              .sum(min_count=1))  # min_count=1: luta toda sem dado vira NaN, nao 0

    # Mapeia EVENT+BOUT -> URL da luta. EVENT+BOUT duplicado (caso unico
    # conhecido: Sakuraba vs. Silveira 2x na mesma noite, UFC Japan 1997) e
    # ambiguo para as stats -- descartamos essas lutas do join, com aviso.
    dup_mask = details.duplicated(["EVENT", "BOUT"], keep=False)
    if dup_mask.any():
        logger.warning("%d luta(s) com EVENT+BOUT ambiguo -- stats dessas lutas descartadas.",
                       dup_mask.sum())
    url_map = details[~dup_mask][["EVENT", "BOUT", "URL"]].rename(columns={"URL": "fight_url"})
    totals = totals.merge(url_map, on=["EVENT", "BOUT"], how="inner")

    stats_out = totals[["fighter", "fight_url", "knockdowns",
                        "sig_strikes_landed", "sig_strikes_attempted",
                        "takedowns_landed", "takedowns_attempted",
                        "sub_attempts", "reversals", "control_seconds"]]

    # --- fighters.csv -------------------------------------------------------
    fighters_out = pd.DataFrame({
        "fighter_url": tott["URL"],
        "name": tott["FIGHTER"],
        "height_cm": tott["HEIGHT"].map(parse_height_to_cm),
        "reach_cm": tott["REACH"].map(parse_reach_to_cm),
        "stance": tott["STANCE"],
        "dob": tott["DOB"].map(parse_date),
    })

    fights_out.to_csv(config.RAW_FIGHTS_CSV, index=False)
    stats_out.to_csv(config.RAW_FIGHT_STATS_CSV, index=False)
    fighters_out.to_csv(config.RAW_FIGHTERS_CSV, index=False)
    save_raw_to_sqlite(fights_out, stats_out, fighters_out)

    logger.info(
        "Convertido: %d lutas (%d sem vencedor definido) de %d eventos (%s a %s), "
        "%d linhas de stats, %d lutadores.",
        len(fights_out), n_no_winner, fights_out["event_url"].nunique(),
        fights_out["event_date"].min().date(), fights_out["event_date"].max().date(),
        len(stats_out), len(fighters_out),
    )


def download_github_mirror_dataset() -> None:
    """Baixa os 6 CSVs do espelho GitHub e converte para o formato canonico 'scrape'."""
    _download_github_mirror_files()
    convert_github_mirror_to_canonical()


# ---------------------------------------------------------------------------
# 5) Gap-filler: completa eventos recentes faltantes com um navegador real
# ---------------------------------------------------------------------------

def fill_recent_gap_with_browser(max_events: int = 30) -> int:
    """
    Completa os eventos que aconteceram DEPOIS do dado mais recente ja em
    data/raw/fights.csv, raspando somente esses eventos do UFCStats.com com
    um navegador real headless (Playwright + Edge/Chromium do sistema).

    Por que navegador real: o UFCStats.com exige execucao de JavaScript
    ("This site requires JavaScript") antes de servir as paginas; um
    navegador de verdade atende a esse requisito naturalmente. Este projeto
    NAO resolve o desafio fora de um navegador (isso seria contornar a
    protecao). Mantemos o mesmo rate limiting educado de sempre
    (config.SCRAPER_DELAY_SECONDS) e buscamos apenas os poucos eventos
    faltantes, nao o site inteiro.

    Retorna o numero de lutas adicionadas.
    """
    global _SOUP_FETCHER
    from playwright.sync_api import sync_playwright

    if not config.RAW_FIGHTS_CSV.exists():
        raise FileNotFoundError(
            "data/raw/fights.csv nao existe -- rode antes a coleta principal "
            "(ex.: python -m src.data_collection --source github-mirror)."
        )

    fights_existing = pd.read_csv(config.RAW_FIGHTS_CSV, parse_dates=["event_date"])
    stats_existing = (pd.read_csv(config.RAW_FIGHT_STATS_CSV)
                      if config.RAW_FIGHT_STATS_CSV.exists() else pd.DataFrame())
    fighters_existing = (pd.read_csv(config.RAW_FIGHTERS_CSV, parse_dates=["dob"])
                         if config.RAW_FIGHTERS_CSV.exists() else pd.DataFrame())

    known_event_urls = set(fights_existing.get("event_url", pd.Series(dtype=str)).dropna())
    known_fighter_urls = set(fighters_existing.get("fighter_url", pd.Series(dtype=str)).dropna())
    latest_date = fights_existing["event_date"].max()
    logger.info("Evento mais recente na base: %s. Buscando eventos posteriores...", latest_date.date())

    new_fights: list[dict] = []
    new_stats: list[dict] = []
    new_fighters: list[dict] = []

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="msedge", headless=True)
        except Exception:  # noqa: BLE001 - sem Edge, tenta o chromium do Playwright
            browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        def browser_fetcher(url: str) -> Optional[BeautifulSoup]:
            try:
                page.goto(url, timeout=60000)
                page.wait_for_load_state("networkidle", timeout=60000)
                html = page.content()
                # o gate JS faz um reload apos validar; se ainda estivermos na
                # pagina de desafio, espera o reload terminar
                for _ in range(5):
                    if "Checking your browser" not in html:
                        break
                    page.wait_for_timeout(2000)
                    html = page.content()
                time.sleep(config.SCRAPER_DELAY_SECONDS)
                return BeautifulSoup(html, "html.parser")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Falha ao carregar %s no navegador: %s", url, exc)
                return None

        _SOUP_FETCHER = browser_fetcher
        try:
            event_urls = list_event_urls()  # mais antigo -> mais recente
            missing = [u for u in reversed(event_urls) if u not in known_event_urls][:max_events]
            logger.info("%d evento(s) ainda nao presentes na base.", len(missing))

            for i, event_url in enumerate(missing, 1):
                fights = scrape_event(event_url)
                if not fights:
                    logger.warning("Evento %s sem lutas parseadas -- pulando.", event_url)
                    continue
                event_date = fights[0]["event_date"]
                logger.info("[%d/%d] %s (%s): %d lutas",
                            i, len(missing), fights[0]["event_name"],
                            event_date.date() if event_date is not None else "?", len(fights))
                new_fights.extend(fights)
                for fight in fights:
                    if fight["fight_url"]:
                        totals = scrape_fight_totals(fight["fight_url"])
                        if totals:
                            new_stats.extend(totals["stats"])
                    for url_key in ("fighter_1_url", "fighter_2_url"):
                        f_url = fight.get(url_key)
                        if f_url and f_url not in known_fighter_urls:
                            profile = scrape_fighter_profile(f_url)
                            if profile:
                                new_fighters.append(profile)
                            known_fighter_urls.add(f_url)
        finally:
            _SOUP_FETCHER = None
            browser.close()

    if not new_fights:
        logger.info("Nenhum evento novo encontrado -- base ja esta em dia com o UFCStats.")
        return 0

    fights_new_df = pd.DataFrame(new_fights)
    fights_new_df = fights_new_df[[c for c in fights_existing.columns if c in fights_new_df.columns]]
    fights_all = pd.concat([fights_existing, fights_new_df], ignore_index=True)
    fights_all = fights_all.drop_duplicates(subset=["fight_url"], keep="first")

    stats_all = pd.concat([stats_existing, pd.DataFrame(new_stats)], ignore_index=True)
    stats_all = stats_all.drop_duplicates(subset=["fight_url", "fighter"], keep="first")

    fighters_new_df = pd.DataFrame(new_fighters)
    if not fighters_new_df.empty:
        fighters_new_df = fighters_new_df[[c for c in fighters_existing.columns
                                           if c in fighters_new_df.columns]]
    fighters_all = pd.concat([fighters_existing, fighters_new_df], ignore_index=True)
    if "fighter_url" in fighters_all.columns:
        fighters_all = fighters_all.drop_duplicates(subset=["fighter_url"], keep="first")

    fights_all.to_csv(config.RAW_FIGHTS_CSV, index=False)
    stats_all.to_csv(config.RAW_FIGHT_STATS_CSV, index=False)
    fighters_all.to_csv(config.RAW_FIGHTERS_CSV, index=False)
    save_raw_to_sqlite(fights_all, stats_all, fighters_all)

    logger.info("Gap preenchido: +%d lutas, +%d linhas de stats, +%d lutadores. "
                "Base agora vai ate %s.",
                len(fights_new_df), len(new_stats), len(fighters_new_df),
                pd.to_datetime(fights_all["event_date"]).max().date())
    return len(fights_new_df)


# ---------------------------------------------------------------------------
# 6) Entrada manual: lutas recentes digitadas a mao (rede de seguranca final)
# ---------------------------------------------------------------------------

MANUAL_FIGHTS_CSV = config.RAW_DIR / "manual_recent_fights.csv"
MANUAL_FIGHTS_COLUMNS = ["event_name", "event_date", "fighter_1", "fighter_2",
                          "winner", "weight_class", "method", "round", "time"]


def merge_manual_recent_fights() -> int:
    """
    Mescla data/raw/manual_recent_fights.csv (se existir e tiver linhas) em
    data/raw/fights.csv. Formato: mesmas colunas do fights.csv canonico,
    menos as URLs (geramos um fight_url sintetico "manual::..." estavel).
    Deixe winner VAZIO para empate/no-contest.

    Serve como rede de seguranca quando toda fonte automatica esta
    defasada: ~10 eventos digitados a mao ja mantem labels, win-rate e
    recencia em dia (as stats de golpes/quedas dessas lutas ficam NaN, o
    que o pipeline ja trata).

    Retorna o numero de lutas adicionadas.
    """
    if not MANUAL_FIGHTS_CSV.exists() or not config.RAW_FIGHTS_CSV.exists():
        return 0
    manual = pd.read_csv(MANUAL_FIGHTS_CSV, parse_dates=["event_date"])
    manual = manual.dropna(subset=["event_date", "fighter_1", "fighter_2"])
    if manual.empty:
        return 0

    missing_cols = [c for c in MANUAL_FIGHTS_COLUMNS if c not in manual.columns]
    if missing_cols:
        raise ValueError(f"{MANUAL_FIGHTS_CSV} sem as colunas: {missing_cols} "
                         f"(esperadas: {MANUAL_FIGHTS_COLUMNS})")

    # winner vazio -> NaN (empate/NC); winner preenchido deve ser um dos dois nomes
    manual["winner"] = manual["winner"].where(
        manual["winner"].astype(str).str.strip().ne(""), other=np.nan)
    valid = manual["winner"].isna() | (manual["winner"] == manual["fighter_1"]) | (manual["winner"] == manual["fighter_2"])
    if not valid.all():
        bad = manual.loc[~valid, ["fighter_1", "fighter_2", "winner"]]
        raise ValueError(f"winner deve ser igual a fighter_1, fighter_2 ou vazio. Linhas invalidas:\n{bad}")

    manual["fight_url"] = ("manual::" + manual["event_date"].dt.strftime("%Y-%m-%d")
                           + "::" + manual["fighter_1"] + "_vs_" + manual["fighter_2"])

    fights_existing = pd.read_csv(config.RAW_FIGHTS_CSV, parse_dates=["event_date"])
    existing_keys = set(zip(fights_existing["event_date"], fights_existing["fighter_1"],
                            fights_existing["fighter_2"]))
    is_new = [
        (d, f1, f2) not in existing_keys and (d, f2, f1) not in existing_keys
        for d, f1, f2 in zip(manual["event_date"], manual["fighter_1"], manual["fighter_2"])
    ]
    manual_new = manual[is_new]
    if manual_new.empty:
        logger.info("Entrada manual: todas as lutas ja estao na base.")
        return 0

    fights_all = pd.concat([fights_existing, manual_new], ignore_index=True)
    fights_all.to_csv(config.RAW_FIGHTS_CSV, index=False)
    logger.info("Entrada manual: +%d luta(s) mescladas de %s (base agora ate %s).",
                len(manual_new), MANUAL_FIGHTS_CSV,
                pd.to_datetime(fights_all['event_date']).max().date())
    return len(manual_new)


# ---------------------------------------------------------------------------
# 7) Fallback: dataset publico ja compilado
# ---------------------------------------------------------------------------

PUBLIC_DATASET_CANDIDATES = [
    # (descricao, URL raw do GitHub)
    # O repositorio original (WarrierRajeev/UFC-Predictions) nao versiona o CSV
    # (data/ esta no .gitignore), entao usamos um espelho publico do mesmo
    # data.csv do Kaggle (kaggle.com/datasets/rajeevw/ufcdata), schema R_/B_.
    ("espelho de rajeevw/ufcdata (kaggle.com/datasets/rajeevw/ufcdata)",
     "https://raw.githubusercontent.com/josh649/DataVisulisationUFCProject/master/dataVisualisation_Ufc_project/ufc.csv"),
]


def download_public_dataset_fallback() -> Optional[pd.DataFrame]:
    """
    Tenta baixar um dataset publico ja compilado, para os casos em que o
    scraping direto nao e viavel (site fora do ar, bloqueio de rede,
    mudanca de layout) ou voce quer comecar mais rapido.

    Se o download automatico falhar, baixe manualmente em
    https://www.kaggle.com/datasets/rajeevw/ufcdata e salve o arquivo como
    data/raw/public_dataset.csv -- o resto do pipeline funciona igual a
    partir dai (veja src/features.py:load_raw_data, que sabe ler tanto o
    formato "scrape" quanto esse formato "dataset publico").
    """
    for description, url in PUBLIC_DATASET_CANDIDATES:
        logger.info("Tentando baixar dataset publico: %s", description)
        try:
            df = pd.read_csv(url)
            df.to_csv(config.PUBLIC_DATASET_CSV, index=False)
            logger.info("Dataset salvo em %s (%d linhas)", config.PUBLIC_DATASET_CSV, len(df))
            return df
        except Exception as exc:  # noqa: BLE001 - degradar com elegancia, tentar proximo candidato
            logger.warning("Falha ao baixar de %s: %s", url, exc)
    if config.PUBLIC_DATASET_CSV.exists():
        logger.warning(
            "Download automatico falhou, mas ja existe um arquivo em %s -- usando esse.",
            config.PUBLIC_DATASET_CSV,
        )
        return pd.read_csv(config.PUBLIC_DATASET_CSV)
    # Sem dataset nenhum: interrompe aqui com uma mensagem acionavel, em vez de
    # deixar o pipeline seguir e quebrar mais adiante com um erro confuso.
    raise RuntimeError(
        "Nao foi possivel baixar nenhum dataset publico automaticamente. "
        "Baixe manualmente do Kaggle (kaggle.com/datasets/rajeevw/ufcdata) "
        f"e salve como {config.PUBLIC_DATASET_CSV}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Coleta de dados historicos de UFC.")
    parser.add_argument(
        "--source", choices=["github-mirror", "scrape", "public-dataset"], default="github-mirror",
        help="'github-mirror' baixa CSVs atualizados diariamente (default, recomendado); "
             "'scrape' faz scraping do UFCStats.com (hoje bloqueado por anti-bot); "
             "'public-dataset' baixa o dataset compilado do Kaggle (para em jun/2019).",
    )
    parser.add_argument(
        "--limit-events", type=int, default=None,
        help="Limita o numero de eventos no scraping (util para testar rapido, ex.: --limit-events 5).",
    )
    parser.add_argument(
        "--fill-gap", action="store_true",
        help="Apos a coleta, completa eventos recentes faltantes raspando so eles "
             "do UFCStats.com com um navegador real headless (requer Playwright).",
    )
    args = parser.parse_args()

    if args.source == "scrape":
        run_full_scrape(limit_events=args.limit_events)
    elif args.source == "github-mirror":
        download_github_mirror_dataset()
    else:
        download_public_dataset_fallback()

    if args.fill_gap:
        fill_recent_gap_with_browser()
    merge_manual_recent_fights()
    check_data_freshness()
