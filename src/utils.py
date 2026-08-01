"""
Funcoes utilitarias compartilhadas entre os modulos do pipeline.
"""
from __future__ import annotations

import difflib
import re
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd


def parse_height_to_cm(height_str: Optional[str]) -> Optional[float]:
    """Converte string tipo "6' 2\"" (pes e polegadas, formato do UFCStats) para cm."""
    if not height_str or not isinstance(height_str, str):
        return None
    match = re.match(r"(\d+)'\s*(\d+)", height_str.strip())
    if not match:
        return None
    feet, inches = int(match.group(1)), int(match.group(2))
    total_inches = feet * 12 + inches
    return round(total_inches * 2.54, 1)


def parse_reach_to_cm(reach_str: Optional[str]) -> Optional[float]:
    """Converte string tipo "74\"" (polegadas, formato do UFCStats) para cm."""
    if not reach_str or not isinstance(reach_str, str):
        return None
    match = re.match(r"(\d+(\.\d+)?)", reach_str.strip())
    if not match:
        return None
    inches = float(match.group(1))
    return round(inches * 2.54, 1)


def parse_pct(pct_str: Optional[str]) -> Optional[float]:
    """Converte string tipo "45%" para float 0.45. Retorna None se vazio/"---"."""
    if pct_str is None:
        return None
    pct_str = str(pct_str).strip()
    if pct_str in ("", "---", "--", "None", "nan"):
        return None
    pct_str = pct_str.replace("%", "")
    try:
        return round(float(pct_str) / 100.0, 4)
    except ValueError:
        return None


def parse_date(date_str: Optional[str]):
    """Converte datas do UFCStats (ex.: 'March 09, 2024') em Timestamp do pandas."""
    if not date_str:
        return None
    date_str = str(date_str).strip()
    if date_str in ("", "--", "nan", "None"):
        return None
    for fmt in ("%B %d, %Y", "%Y-%m-%d", "%b %d, %Y", "%m/%d/%Y"):
        try:
            return pd.Timestamp(datetime.strptime(date_str, fmt))
        except ValueError:
            continue
    try:
        return pd.Timestamp(date_str)
    except (ValueError, TypeError):
        return None


def moneyline_to_decimal(moneyline: float) -> float:
    """Converte odds americanas (moneyline, ex.: -150, +130) para odds decimais."""
    if moneyline > 0:
        return 1 + moneyline / 100.0
    return 1 + 100.0 / abs(moneyline)


def decimal_odds_to_implied_prob(decimal_odds: float) -> float:
    """Probabilidade implicita bruta (com vig/overround embutido) de uma odd decimal."""
    return 1.0 / decimal_odds


def probability_to_fair_odds(p: float) -> tuple[float, float]:
    """
    Converte uma probabilidade na odd JUSTA equivalente (sem vig/margem de
    casa): decimal = 1/p, e a moneyline americana correspondente.

    Convencao adotada: p == 0.5 exato cai no lado NEGATIVO (-100), tratando
    ">= 0.5" como favorito -- decimal 2.00 / -100. (Nas americanas, +100 e
    -100 representam a mesma odd justa; e so uma escolha de apresentacao.)

    p fora do intervalo aberto (0, 1) levanta ValueError: odd justa de
    probabilidade 0 (infinita) ou 1 (sem retorno) nao e representavel nem
    util num relatorio -- melhor falhar claramente do que inventar um cap.

    Retorna (decimal, american).
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"Probabilidade deve estar em (0, 1) para ter odd justa; recebi {p!r}.")
    decimal = 1.0 / p
    if p >= 0.5:
        american = -100.0 * p / (1.0 - p)
    else:
        american = 100.0 * (1.0 - p) / p
    return round(decimal, 3), round(american)


def remove_vig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    """
    Remove o overround (vig) de um mercado de 2 resultados, normalizando as
    probabilidades implicitas para somarem 1. E o metodo proporcional simples;
    nao e o unico metodo existente (ha tambem "power" e "Shin"), mas e o mais
    transparente e suficiente para uma comparacao honesta modelo-vs-mercado.
    """
    total = prob_a + prob_b
    if total <= 0:
        return prob_a, prob_b
    return prob_a / total, prob_b / total


# Sufixos geracionais: NAO sao sobrenome ("Levi Rodrigues Jr." -> rodrigues,
# nao "jr"; "Kai Kamaka III" -> kamaka). Sem isso, a mesma pessoa listada com
# e sem sufixo deixa de casar.
_NAME_SUFFIXES = {"jr", "jnr", "sr", "snr", "ii", "iii", "iv"}


def _name_tokens(name: str) -> list[str]:
    """
    Tokens minusculos do nome, sem sufixos geracionais.

    Apostrofos sao REMOVIDOS, nao usados como separador: "L'udovit" e um
    token ("ludovit"), nao ["l", "udovit"] -- um token de 1 letra quebra as
    guardas de primeiro nome/sobrenome. Mesma logica para "Lone'er".
    """
    cleaned = str(name).lower().replace("'", "").replace("’", "")
    tokens = [t for t in re.split(r"[\s.\-]+", cleaned) if t]
    while len(tokens) > 1 and tokens[-1] in _NAME_SUFFIXES:
        tokens.pop()
    return tokens


def _first(name: str) -> str:
    toks = _name_tokens(name)
    return toks[0] if toks else ""


def _surname(name: str) -> str:
    toks = _name_tokens(name)
    return toks[-1] if toks else ""


def best_name_match(name: str, candidates: Iterable[str], cutoff: float = 0.75,
                    surname_cutoff: float = 0.6, firstname_cutoff: float = 0.7) -> Optional[str]:
    """
    Faz fuzzy matching de um nome de lutador contra uma lista de nomes conhecidos.
    Util porque o usuario pode digitar o nome com grafia/acentos levemente
    diferentes do que esta salvo na base de dados.

    Guarda de PONTAS (anti-falso-positivo): o difflib sozinho casa nomes que
    compartilham SO uma das pontas do nome -- perigoso num esporte cheio de
    nomes/sobrenomes repetidos. Dois modos de falha ja pegos em producao:
      - so o primeiro nome igual: "Muhammad Said" ~0.79 com "Muhammad Naimov";
      - so o sobrenome igual: "Michael Oliveira" com "Charles Oliveira" (!).
    Alem do score global do difflib, exigimos que o PRIMEIRO nome E o
    SOBRENOME batam (similaridade >= os cutoffs). Variantes legitimas passam
    porque as duas pontas batem ("St. Denis"->"Saint Denis"; "Seok Hyun Ko"
    ->"Seokhyeon Ko"; typo "Magomad"->"Magomed"); pessoas diferentes que so
    dividem uma ponta sao rejeitadas (viram estreante/erro claro).
    """
    candidates = list(candidates)
    hit = _match_with_guards(name, candidates, cutoff, surname_cutoff, firstname_cutoff)
    if hit is not None:
        return hit
    # Fallback de ORDEM INVERTIDA: o UFC lista nomes do leste asiatico nos dois
    # sentidos ("Cong Wang" nas odds, "Wang Cong" na base) -- pessoas iguais que
    # nenhum score de string casa direto. So aceita se as duas pontas baterem
    # apos a inversao, entao nao afrouxa as guardas.
    tokens = _name_tokens(name)
    if len(tokens) == 2:
        return _match_with_guards(" ".join(reversed(tokens)), candidates,
                                  cutoff, surname_cutoff, firstname_cutoff)
    return None


def _match_with_guards(name: str, candidates: list[str], cutoff: float,
                       surname_cutoff: float, firstname_cutoff: float) -> Optional[str]:
    # varios candidatos, para poder pular um 1o lugar reprovado nas guardas
    matches = difflib.get_close_matches(name, candidates, n=5, cutoff=cutoff)
    q_first, q_last = _first(name), _surname(name)
    for cand in matches:
        c_first = _first(cand)
        last_sim = difflib.SequenceMatcher(None, q_last, _surname(cand)).ratio()
        first_sim = difflib.SequenceMatcher(None, q_first, c_first).ratio()
        # primeiro nome "bate" se um e prefixo do outro (variante truncada,
        # ex.: Seok/Seokhyeon) OU se a similaridade e alta (typo, ex.:
        # Magomad/Magomed). Assim "Michael" nao casa com "Charles" (0.57,
        # sem prefixo), mas variantes legitimas passam.
        prefix_ok = (len(q_first) >= 3 and len(c_first) >= 3
                     and (q_first.startswith(c_first) or c_first.startswith(q_first)))
        first_ok = prefix_ok or first_sim >= firstname_cutoff
        if first_ok and last_sim >= surname_cutoff:
            return cand
    return None
