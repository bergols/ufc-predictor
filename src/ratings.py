"""
src/ratings.py

Rating Elo por lutador, calculado numa UNICA passada cronologica GLOBAL
por todas as lutas -- diferente das demais estatisticas (que sao "por
lutador" e podem ser agrupadas), o Elo de cada lutador depende do rating
do OPONENTE no momento da luta, entao a ordem global importa.

Point-in-time como sempre: para cada luta registramos o rating de cada
lutador ANTES dela (nunca incluindo o resultado dela mesma); so depois
atualizamos os dois ratings com o resultado.

Decisoes:
  - Estreantes comecam em config.ELO_BASE_RATING (default 1500).
  - K-factor em config.ELO_K_FACTOR (default 32), facil de variar.
  - Lutas sem vencedor (winner NaN = empate OU no-contest) NAO atualizam
    rating: nao da para distinguir empate de luta anulada só pelo winner,
    e ambos juntos sao <2% das lutas -- o custo de ignorar o meio ponto
    do empate raro e menor que o de tratar no-contest como empate.
  - Lutas na MESMA data sao processadas na ordem em que aparecem no
    DataFrame (nao ha hora do dia nos dados); o efeito e desprezivel
    porque um mesmo lutador quase nunca luta duas vezes no mesmo dia.
"""
from __future__ import annotations

import logging

import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probabilidade esperada de A vencer B segundo a formula padrao do Elo."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def method_k_multiplier(method, multipliers: dict | None) -> float:
    """
    Multiplicador do K-factor pela "decisividade" da vitoria (extensao
    padrao de Elo esportivo: margem de vitoria). Buckets:
      - "FINISH" (KO/TKO ou finalizacao): vitoria decisiva;
      - "DECISION_CLOSE" (decisao dividida/majoritaria): vitoria apertada;
      - "DECISION" (decisao unanime e default): peso base.
    `multipliers` e um dict como {"FINISH": 1.25, "DECISION_CLOSE": 0.75};
    chaves ausentes valem 1.0. None desliga a margem (Elo simples).
    """
    if not multipliers:
        return 1.0
    from src.features import categorize_method
    cat = categorize_method(method)
    if cat in ("KO_TKO", "SUBMISSION"):
        return multipliers.get("FINISH", 1.0)
    if cat == "DECISION":
        m = str(method).upper()
        # cobre as duas fontes: "S-DEC"/"M-DEC" (scrape) e
        # "Decision - Split"/"- Majority" (espelho GitHub)
        if "S-DEC" in m or "SPLIT" in m or "M-DEC" in m or "MAJORITY" in m:
            return multipliers.get("DECISION_CLOSE", 1.0)
        return multipliers.get("DECISION", 1.0)
    return 1.0  # DQ/overturned/desconhecido: peso base


def compute_elo_ratings(fights_df: pd.DataFrame,
                        k: float | None = None,
                        base_rating: float | None = None,
                        method_multipliers: dict | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Recebe a tabela de lutas (colunas minimas: fight_id, event_date,
    fighter_1, fighter_2, winner; opcional: method, usada pela margem) e
    devolve:

      - pre_ratings: DataFrame com uma linha por luta
        (fight_id, elo_1_pre, elo_2_pre) contendo o rating de cada lutador
        ANTES daquela luta -- e o que vira feature, sem vazamento;
      - current: dict {lutador: rating atual} apos processar tudo
        (usado pelo CLI de predicao para lutas futuras).

    method_multipliers (default: config.ELO_METHOD_MULTIPLIERS) escala o K
    pela decisividade da vitoria (ver method_k_multiplier). None = Elo
    simples. Validado em cal_select (jul/2026): a margem NAO bateu o Elo
    simples, entao o default de producao e None -- o parametro fica para
    experimentos.
    """
    k = k if k is not None else config.ELO_K_FACTOR
    base_rating = base_rating if base_rating is not None else config.ELO_BASE_RATING
    if method_multipliers is None:
        method_multipliers = config.ELO_METHOD_MULTIPLIERS

    ordered = fights_df.sort_values("event_date", kind="stable")
    has_method = "method" in ordered.columns

    ratings: dict[str, float] = {}
    rows = []
    for row in ordered.itertuples(index=False):
        f1, f2 = row.fighter_1, row.fighter_2
        r1 = ratings.get(f1, base_rating)
        r2 = ratings.get(f2, base_rating)
        rows.append({"fight_id": row.fight_id, "elo_1_pre": r1, "elo_2_pre": r2})

        winner = row.winner
        if pd.isna(winner):
            continue  # empate/no-contest: nao atualiza (ver docstring)
        k_eff = k * method_k_multiplier(row.method if has_method else None, method_multipliers)
        s1 = 1.0 if winner == f1 else 0.0
        e1 = expected_score(r1, r2)
        ratings[f1] = r1 + k_eff * (s1 - e1)
        ratings[f2] = r2 + k_eff * ((1.0 - s1) - (1.0 - e1))

    return pd.DataFrame(rows), ratings
