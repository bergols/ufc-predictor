"""
src/fighter_photos.py

Busca de fotos de lutadores nas paginas de atleta do UFC.com — para o
relatorio LOCAL de uso pessoal (flag --photos do card_report).

Deliberadamente NAO usada na pagina publicada no GitHub Pages: fotos
promocionais sao material com direitos autorais e a pagina do Pages e
publica; o uso aqui e visualizacao pessoal, com as imagens carregadas
direto do site do UFC (hotlink, nada e copiado nem redistribuido).

Funcionamento: nome -> slug (ufc.com/athlete/<slug>) -> meta og:image.
Resultados (inclusive misses) ficam em cache local
(data/raw/fighter_photos.json) para nao rebuscar a cada geracao; apague
o arquivo para forcar re-busca. Fetch educado: 1 req/s, e qualquer falha
vira apenas o avatar de monograma de sempre.
"""
from __future__ import annotations

import json
import logging
import re
import time
import unicodedata

import requests

import config

logger = logging.getLogger(__name__)

PHOTO_CACHE_PATH = config.RAW_DIR / "fighter_photos.json"
_ATHLETE_URL = "https://www.ufc.com/athlete/{slug}"


def name_to_slug(name: str) -> str:
    """'Benoit St. Denis' -> 'benoit-st-denis' (padrao das URLs do UFC.com)."""
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    return slug


def _load_cache() -> dict:
    if PHOTO_CACHE_PATH.exists():
        try:
            return json.loads(PHOTO_CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Cache de fotos corrompido (%s) — recomecando vazio.", PHOTO_CACHE_PATH)
    return {}


def _slug_variants(name: str) -> list[str]:
    """
    Slugs candidatos, do mais provavel ao menos: o nome como veio; 'st'
    expandido para 'saint' (Benoit St. Denis -> benoit-saint-denis); e a
    ordem das palavras invertida (nomes asiaticos aparecem ora como
    'Cong Wang', ora como 'Wang Cong' -> wang-cong).
    """
    primary = name_to_slug(name)
    variants = [primary]

    def add(slug: str) -> None:
        if slug and slug not in variants:
            variants.append(slug)

    # apostrofo colado em vez de hifen: Lone'er -> loneer
    add(name_to_slug(str(name).replace("'", "")))
    # 'st' expandido: Benoit St. Denis -> benoit-saint-denis
    add(re.sub(r"(^|-)st-", r"\1saint-", primary))
    parts = str(name).split()
    # ordem invertida: Cong Wang -> wang-cong
    if len(parts) > 1:
        add(name_to_slug(" ".join(reversed(parts))))
    # nomes de 3+ palavras com as duas primeiras coladas: Seok Hyun Ko -> seokhyun-ko
    if len(parts) >= 3:
        add(name_to_slug(parts[0] + parts[1] + " " + " ".join(parts[2:])))
    return variants


def _fetch_photo_url(name: str) -> str | None:
    """Tenta os slugs candidatos em ordem; None se nenhum servir (sem raise)."""
    from bs4 import BeautifulSoup
    for slug in _slug_variants(name):
        url = _ATHLETE_URL.format(slug=slug)
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": config.SCRAPER_USER_AGENT})
        except requests.RequestException as exc:
            logger.info("Falha de rede buscando foto de '%s': %s", name, exc)
            return None
        if resp.status_code == 200:
            og = BeautifulSoup(resp.text, "html.parser").find("meta", property="og:image")
            if og and og.get("content"):
                return og["content"]
        time.sleep(config.SCRAPER_DELAY_SECONDS)
    logger.info("Sem pagina de atleta para '%s' (tentados: %s).", name, _slug_variants(name))
    return None


def get_photo_urls(names: list[str]) -> dict[str, str | None]:
    """
    Mapa nome -> URL de foto (ou None). Consulta o cache primeiro; so vai
    a rede para nomes ineditos (misses tambem sao cacheados — apague
    data/raw/fighter_photos.json para re-tentar).
    """
    cache = _load_cache()
    to_fetch = [n for n in dict.fromkeys(str(n) for n in names) if n not in cache]
    for i, name in enumerate(to_fetch):
        if i > 0:
            time.sleep(config.SCRAPER_DELAY_SECONDS)
        cache[name] = _fetch_photo_url(name)
    if to_fetch:
        PHOTO_CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
        found = sum(1 for n in to_fetch if cache[n])
        logger.info("Fotos: %d/%d encontradas para nomes novos (cache: %s).",
                    found, len(to_fetch), PHOTO_CACHE_PATH)
    return {n: cache.get(str(n)) for n in names}
