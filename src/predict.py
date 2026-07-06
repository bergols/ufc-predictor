"""
src/predict.py

CLI simples: informe os nomes de dois lutadores e receba a probabilidade
estimada de vitoria de cada um.

Uso:
    python -m src.predict "Fighter A" "Fighter B"
    python -m src.predict "Fighter A" "Fighter B" --model logreg

Como funciona: para cada lutador, buscamos o "nivel atual" dele (stats
point-in-time apos a ultima luta registrada na base -- ver
src/features.py::export_latest_fighter_levels), calculamos as mesmas
features diferenciais (A - B) usadas no treino, e passamos pelo modelo
calibrado salvo em models/.

LIMITACAO IMPORTANTE: so funciona para lutadores que ja tem pelo menos uma
luta na base de dados usada no treino. Para uma estreia no UFC (lutador
sem historico), o modelo nao tem como estimar -- isso e uma limitacao
conhecida e esperada desta v1.
"""
from __future__ import annotations

import argparse
import json
import logging

import joblib
import numpy as np
import pandas as pd

import config
from src.features import (FEATURE_COLUMNS, SYMMETRIC_SUM_COLUMNS,
                          export_latest_fighter_levels, stance_mismatch_value)
from src.utils import best_name_match

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _resolve_fighters(fighter_a_name: str, fighter_b_name: str, levels: pd.DataFrame):
    """Casa os dois nomes contra a base (exato, depois fuzzy). ValueError se nao achar."""
    known_names = levels["fighter"].dropna().unique().tolist()
    match_a = fighter_a_name if fighter_a_name in known_names else best_name_match(fighter_a_name, known_names)
    match_b = fighter_b_name if fighter_b_name in known_names else best_name_match(fighter_b_name, known_names)
    if match_a is None:
        raise ValueError(f"Lutador '{fighter_a_name}' nao encontrado na base de dados.")
    if match_b is None:
        raise ValueError(f"Lutador '{fighter_b_name}' nao encontrado na base de dados.")
    if match_a != fighter_a_name:
        print(f"(nao encontrei '{fighter_a_name}' exatamente -- usando '{match_a}')")
    if match_b != fighter_b_name:
        print(f"(nao encontrei '{fighter_b_name}' exatamente -- usando '{match_b}')")
    row_a = levels[levels["fighter"] == match_a].iloc[0]
    row_b = levels[levels["fighter"] == match_b].iloc[0]
    return match_a, match_b, row_a, row_b


def _diff_feature_row(row_a: pd.Series, row_b: pd.Series) -> dict:
    """Features diferenciais (as mesmas do treino do preditor de vencedor)."""
    return {
        "striking_accuracy_diff": row_a["striking_accuracy"] - row_b["striking_accuracy"],
        "takedown_accuracy_diff": row_a["takedown_accuracy"] - row_b["takedown_accuracy"],
        "takedown_defense_diff": row_a["takedown_defense"] - row_b["takedown_defense"],
        "reach_diff_cm": row_a["reach_cm"] - row_b["reach_cm"],
        "height_diff_cm": row_a["height_cm"] - row_b["height_cm"],
        "age_diff_years": row_a["age_years"] - row_b["age_years"],
        "days_since_last_fight_diff": row_a["days_since_last_fight"] - row_b["days_since_last_fight"],
        "recent_win_rate_diff": row_a["recent_win_rate"] - row_b["recent_win_rate"],
        "career_win_rate_diff": row_a["career_win_rate"] - row_b["career_win_rate"],
        "experience_diff": row_a["n_prior_fights"] - row_b["n_prior_fights"],
        "ko_rate_diff": row_a["ko_rate"] - row_b["ko_rate"],
        "submission_rate_diff": row_a["submission_rate"] - row_b["submission_rate"],
        "elo_diff": row_a["elo"] - row_b["elo"],
        "stance_mismatch": stance_mismatch_value(row_a["stance"], row_b["stance"]),
        "fighter_a_low_experience": row_a["low_experience"],
        "fighter_b_low_experience": row_b["low_experience"],
    }


def _sum_feature_row(row_a: pd.Series, row_b: pd.Series) -> dict:
    """Somas simetricas usadas pelos modelos de metodo/round (fase 2)."""
    return {
        "ko_rate_sum": row_a["ko_rate"] + row_b["ko_rate"],
        "submission_rate_sum": row_a["submission_rate"] + row_b["submission_rate"],
        "striking_accuracy_sum": row_a["striking_accuracy"] + row_b["striking_accuracy"],
        "takedown_accuracy_sum": row_a["takedown_accuracy"] + row_b["takedown_accuracy"],
        "career_win_rate_sum": row_a["career_win_rate"] + row_b["career_win_rate"],
        "experience_total": row_a["n_prior_fights"] + row_b["n_prior_fights"],
    }


def predict_fight(fighter_a_name: str, fighter_b_name: str, model_name: str = "gbm",
                  levels: pd.DataFrame | None = None) -> dict:
    """
    `levels` opcional: DataFrame de export_latest_fighter_levels() ja
    computado. Quem prever varias lutas em sequencia (ex.: src/card_report)
    passa o mesmo levels para nao recarregar a base a cada luta; nesse caso
    o aviso de frescor tambem fica a cargo do chamador.
    """
    if levels is None:
        # Loga um WARNING se a base estiver defasada (o "nivel atual" dos
        # lutadores pode estar velho sem que o usuario perceba).
        from src.data_collection import check_data_freshness
        check_data_freshness()
        levels = export_latest_fighter_levels()

    match_a, match_b, row_a, row_b = _resolve_fighters(fighter_a_name, fighter_b_name, levels)
    X = pd.DataFrame([_diff_feature_row(row_a, row_b)])[FEATURE_COLUMNS]

    model_path = config.GBM_MODEL_PATH if model_name == "gbm" else config.LOGREG_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado em {model_path}. Rode primeiro: python -m src.train")
    model = joblib.load(model_path)

    prob_a = float(model.predict_proba(X)[:, 1][0])
    return {
        "fighter_a": match_a,
        "fighter_b": match_b,
        "prob_a_wins": round(prob_a, 4),
        "prob_b_wins": round(1 - prob_a, 4),
        "model_used": model_name,
        "fighter_a_low_experience": bool(row_a["low_experience"]),
        "fighter_b_low_experience": bool(row_b["low_experience"]),
    }


def compute_total_rounds_market(method_probs: dict, round_band_probs: dict) -> dict:
    """
    Mercado de duracao no formato over/under de rounds, com DUAS linhas
    (1,5 e 2,5) -- o que as faixas {1, 2, 3+} do modelo sustentam sem
    forcar numero (linhas 3,5/4,5 exigiriam separar o "3+", que nao temos).

    As probabilidades sao INCONDICIONAIS (da luta, nao "dado que houve
    finalizacao"): decisao sempre termina no round agendado (3 ou 5), ou
    seja, sempre passa das duas linhas. Logo:

      Under 1,5 = P(finalizacao) * P(faixa "1" | finalizacao)
      Under 2,5 = P(finalizacao) * [P("1" | fin) + P("2" | fin)]
      Over  X   = 1 - Under X   (inclui decisao e finalizacoes tardias)

    Reaproveita a saida de predict_method_and_duration; nada e retreinado.
    """
    p_finish = method_probs.get("KO_TKO", 0.0) + method_probs.get("SUBMISSION", 0.0)
    under_1_5 = p_finish * round_band_probs.get("1", 0.0)
    under_2_5 = p_finish * (round_band_probs.get("1", 0.0) + round_band_probs.get("2", 0.0))
    return {"under_1_5": round(under_1_5, 4), "over_1_5": round(1.0 - under_1_5, 4),
            "under_2_5": round(under_2_5, 4), "over_2_5": round(1.0 - under_2_5, 4)}


def predict_method_and_duration(fighter_a_name: str, fighter_b_name: str,
                                levels: pd.DataFrame | None = None,
                                scheduled_rounds: int | None = None) -> dict:
    """
    Distribuicao prevista de METODO (KO/TKO, finalizacao, decisao) e, se
    houver finalizacao, da FAIXA de round (1 / 2-3 / 4-5). Usa os modelos
    logreg calibrados da fase 2 (os que bateram o baseline ingenuo).

    Falha de forma INDEPENDENTE da previsao de vencedor: quem chama (ex.:
    card_report) pode manter o vencedor e marcar so metodo/duracao como
    "sem previsao". IMPORTANTE ao interpretar: e uma TENDENCIA estatistica
    (margem pequena sobre o baseline "sempre decisao"/"sempre round 1"),
    nao uma previsao pontual confiavel.
    """
    if levels is None:
        levels = export_latest_fighter_levels()

    for path in (config.METHOD_LOGREG_MODEL_PATH, config.ROUND_LOGREG_MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Modelo nao encontrado em {path}. Rode primeiro: python -m src.train_method")

    match_a, match_b, row_a, row_b = _resolve_fighters(fighter_a_name, fighter_b_name, levels)
    feature_row = {**_diff_feature_row(row_a, row_b), **_sum_feature_row(row_a, row_b)}
    X_method = pd.DataFrame([feature_row])[FEATURE_COLUMNS + SYMMETRIC_SUM_COLUMNS]
    # o modelo de faixa recebe scheduled_rounds tambem como feature (NaN se
    # desconhecido; o imputer do pipeline trata)
    feature_row_round = {**feature_row, "scheduled_rounds":
                         scheduled_rounds if scheduled_rounds is not None else np.nan}
    X_round = pd.DataFrame([feature_row_round])[FEATURE_COLUMNS + SYMMETRIC_SUM_COLUMNS
                                                + ["scheduled_rounds"]]

    method_model = joblib.load(config.METHOD_LOGREG_MODEL_PATH)
    round_model = joblib.load(config.ROUND_LOGREG_MODEL_PATH)

    method_probs = dict(zip(method_model.classes_, method_model.predict_proba(X_method)[0]))
    round_probs = dict(zip(round_model.classes_, round_model.predict_proba(X_round)[0]))

    return {
        "fighter_a": match_a,
        "fighter_b": match_b,
        "method_probs": {c: round(float(method_probs.get(c, 0.0)), 4)
                         for c in ("KO_TKO", "SUBMISSION", "DECISION")},
        # condicional: distribuicao da faixa de round DADO que houve finalizacao.
        # As faixas {1, 2, 3+} valem para qualquer formato de luta (round 3+
        # existe tanto em luta de 3 quanto de 5 rounds), entao a restricao
        # logica antiga (zerar "4-5" em luta de 3 rounds) deixou de existir;
        # scheduled_rounds segue como FEATURE do modelo.
        "round_band_probs": {c: round(float(round_probs.get(c, 0.0)), 4)
                             for c in ("1", "2", "3+")},
        "scheduled_rounds": scheduled_rounds,
        "model_used": "logreg",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preve o vencedor de uma luta de UFC (classificacao binaria).")
    parser.add_argument("fighter_a", type=str, help="Nome do primeiro lutador")
    parser.add_argument("fighter_b", type=str, help="Nome do segundo lutador")
    parser.add_argument("--model", choices=["gbm", "logreg"], default="gbm",
                         help="Qual modelo calibrado usar (default: gbm)")
    parser.add_argument("--json", action="store_true", help="Imprime a saida em JSON (util para scripts)")
    args = parser.parse_args()

    result = predict_fight(args.fighter_a, args.fighter_b, model_name=args.model)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"\n{result['fighter_a']}: {result['prob_a_wins']:.1%}")
        print(f"{result['fighter_b']}: {result['prob_b_wins']:.1%}")
        if result["fighter_a_low_experience"] or result["fighter_b_low_experience"]:
            print("\nAviso: pelo menos um dos lutadores tem poucas lutas registradas na base -- "
                  "estimativa menos confiavel que o normal.")
        print("\nLembrete: MMA tem alta variancia. Esta e uma estimativa estatistica, nao uma certeza.")
