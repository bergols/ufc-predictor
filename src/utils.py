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


def _surname(name: str) -> str:
    """Ultimo token alfanumerico do nome, minusculo (o 'sobrenome')."""
    tokens = [t for t in re.split(r"[\s.'-]+", str(name).lower()) if t]
    return tokens[-1] if tokens else ""


def best_name_match(name: str, candidates: Iterable[str], cutoff: float = 0.75,
                    surname_cutoff: float = 0.6) -> Optional[str]:
    """
    Faz fuzzy matching de um nome de lutador contra uma lista de nomes conhecidos.
    Util porque o usuario pode digitar o nome com grafia/acentos levemente
    diferentes do que esta salvo na base de dados.

    Guarda de SOBRENOME (anti-falso-positivo): o difflib sozinho casa nomes
    que so compartilham o primeiro nome -- perigoso num esporte cheio de
    "Muhammad"/"Magomed" (ex.: "Muhammad Said" ~0.79 com "Muhammad Naimov",
    pessoas diferentes). Alem do score global, exigimos que o sobrenome
    (ultimo token) tenha similaridade >= surname_cutoff. Variantes legitimas
    passam porque o sobrenome bate ("St. Denis"->"Saint Denis": denis=denis;
    "Seok Hyun Ko"->"Seokhyeon Ko": ko=ko); pessoas diferentes que so dividem
    o primeiro nome sao rejeitadas (Said vs Naimov: ~0.2).
    """
    candidates = list(candidates)
    # varios candidatos, para poder pular um 1o lugar reprovado no sobrenome
    matches = difflib.get_close_matches(name, candidates, n=5, cutoff=cutoff)
    target_surname = _surname(name)
    for cand in matches:
        surname_sim = difflib.SequenceMatcher(None, target_surname, _surname(cand)).ratio()
        if surname_sim >= surname_cutoff:
            return cand
    return None
