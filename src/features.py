"""
src/features.py

Engenharia de features: transforma os dados brutos (de qualquer uma das
duas fontes suportadas -- scraping do UFCStats.com ou dataset publico
compilado) num dataset de features diferenciais pronto para treino.

CONCEITO CENTRAL -- "point-in-time":
Para cada luta, as estatisticas de cada lutador usadas como feature devem
refletir SOMENTE o que se sabia sobre ele ANTES daquela luta acontecer
(media de lutas anteriores). Usar a media de carreira atual (de hoje) para
prever uma luta de 2015 seria vazamento de dados do futuro (data leakage
temporal) -- o modelo pareceria muito melhor do que realmente e.

Por isso, quando a fonte e o scraping (fight-a-fight granular), este
modulo recalcula medias "expanding" (cumulativas, excluindo a luta atual)
a partir do historico real. Quando a fonte e o dataset publico
pre-compilado, ele ja vem com medias acumuladas ate a data de cada luta,
entao usamos essas colunas diretamente (ver build_features_from_public_dataset).

FEATURES SAO SEMPRE DIFERENCIAIS (fighter_a - fighter_b), nunca valores
absolutos isolados -- isso e o que foi pedido no escopo do projeto, e
tambem torna o modelo mais generalizavel (o que importa e a VANTAGEM
relativa entre os dois lutadores, nao o valor bruto de cada um).

Cada luta gera DUAS linhas no dataset final (uma com A=fighter_1,
B=fighter_2 e outra espelhada com A=fighter_2, B=fighter_1). Isso remove
qualquer vies de "ordem" (o modelo nao pode aprender que "o primeiro nome
listado tende a ganhar"). As duas linhas espelhadas compartilham o mesmo
fight_id -- o split temporal (em train.py) agrupa por fight_id para que as
duas nunca fiquem em conjuntos (treino/calibracao/teste) diferentes.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Colunas finais do dataset de features (mantidas numa constante para que
# train.py, evaluate.py e predict.py concordem exatamente sobre quais sao
# as colunas de entrada do modelo).
FEATURE_COLUMNS = [
    "striking_accuracy_diff",
    "takedown_accuracy_diff",
    "takedown_defense_diff",
    "reach_diff_cm",
    "height_diff_cm",
    "age_diff_years",
    "days_since_last_fight_diff",
    "recent_win_rate_diff",
    "career_win_rate_diff",
    "experience_diff",
    "ko_rate_diff",
    "submission_rate_diff",
    "elo_diff",
    "stance_mismatch",
    "fighter_a_low_experience",
    "fighter_b_low_experience",
]

# Colunas que NAO invertem de sinal na linha espelhada (B-A): flags de cada
# lado (que apenas trocam entre si) e features simetricas por construcao
# (stance_mismatch nao depende de quem e "A" ou "B").
MIRROR_NON_NEGATED_COLUMNS = {"stance_mismatch", "fighter_a_low_experience", "fighter_b_low_experience"}

# Features SIMETRICAS de "soma" (a+b), usadas pelos modelos de METODO/ROUND
# (fase 2) e NAO pelo preditor de vencedor. Racional: diffs (a-b) cancelam o
# sinal de que o metodo precisa -- se a luta termina em nocaute depende do
# nivel COMBINADO de finalizacao dos dois, nao da diferenca entre eles.
# Simetricas por construcao (a+b == b+a), entao identicas nas linhas
# espelhadas, como stance_mismatch.
SYMMETRIC_SUM_COLUMNS = [
    "ko_rate_sum",
    "submission_rate_sum",
    "striking_accuracy_sum",
    "takedown_accuracy_sum",
    "career_win_rate_sum",
    "experience_total",
]

META_COLUMNS = ["fight_id", "event_date", "fighter_a", "fighter_b", "label"]


# ---------------------------------------------------------------------------
# 0) Helpers compartilhados: categoria de metodo de vitoria e confronto de stance
# ---------------------------------------------------------------------------

def categorize_method(method) -> Optional[str]:
    """
    Normaliza o texto livre de metodo de vitoria em 3 buckets: "KO_TKO",
    "SUBMISSION" ou "DECISION". Cobre as variacoes das duas fontes:
      - scrape do UFCStats: "KO/TKO Spinning Back Kick", "SUB Rear Naked
        Choke", "U-DEC" / "S-DEC" / "M-DEC", "TKO Doctor's Stoppage"
      - espelho GitHub: "KO/TKO", "Submission", "Decision - Unanimous" /
        "- Split" / "- Majority", "TKO - Doctor's Stoppage"
    Qualquer outra coisa (DQ, Overturned, Could Not Continue, vazio) vira
    None -- e usada apenas como FEATURE de entrada (taxa historica de
    finalizacao), nao como alvo de previsao.
    """
    if method is None or (isinstance(method, float) and np.isnan(method)):
        return None
    m = str(method).upper()
    if not m or m in ("NAN", "NONE"):
        return None
    if "SUB" in m:
        return "SUBMISSION"
    if "KO" in m:  # cobre "KO/TKO" e "TKO ..." (e nao casa com DQ/Overturned)
        return "KO_TKO"
    if "DEC" in m:  # cobre "DECISION - ..." e as abreviacoes "U-DEC"/"S-DEC"/"M-DEC"
        return "DECISION"
    return None


def stance_mismatch_value(stance_a, stance_b) -> float:
    """
    1.0 se as stances divergem (ex.: Orthodox vs Southpaw), 0.0 se sao
    iguais. "Switch" (ambidestro: o confronto muda a cada momento) e
    valores ausentes viram NaN -- incerto, nao forcamos um lado.
    E uma feature SIMETRICA: nao depende da ordem (A, B).
    """
    known = {"ORTHODOX", "SOUTHPAW"}
    a = str(stance_a).strip().upper() if isinstance(stance_a, str) and stance_a.strip() else None
    b = str(stance_b).strip().upper() if isinstance(stance_b, str) and stance_b.strip() else None
    if a not in known or b not in known:
        return np.nan
    return 1.0 if a != b else 0.0


# ---------------------------------------------------------------------------
# 1) Pipeline a partir dos dados de scraping (granular, fight-a-fight)
# ---------------------------------------------------------------------------

def _attach_opponent_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada linha (fighter, fight_id), anexa as estatisticas OFENSIVAS do
    OPONENTE naquela mesma luta. Isso e necessario para calcular defesa de
    quedas (takedown defense) e defesa de striking do ponto de vista do
    fighter: "quantas vezes o oponente tentou/acertou golpes/quedas EM MIM".
    """
    self_join = long_df.merge(long_df, on="fight_id", suffixes=("", "_opp"))
    self_join = self_join[self_join["fighter"] != self_join["fighter_opp"]].copy()
    keep_cols = [
        "fight_id", "fighter",
        "sig_strikes_landed_opp", "sig_strikes_attempted_opp",
        "takedowns_landed_opp", "takedowns_attempted_opp",
    ]
    return long_df.merge(self_join[keep_cols], on=["fight_id", "fighter"], how="left")


def compute_point_in_time_stats(long_df: pd.DataFrame, fighters_bio: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe uma tabela "longa" (uma linha por lutador por luta) com as
    colunas minimas:
        fight_id, event_date, fighter, opponent, result ('W'/'L'/'NC'),
        sig_strikes_landed, sig_strikes_attempted,
        takedowns_landed, takedowns_attempted
    e devolve a mesma tabela com colunas adicionais de estatisticas
    "ponto-no-tempo" (calculadas usando SOMENTE lutas anteriores aquela
    linha, nunca a luta atual ou futuras).
    """
    df = long_df.sort_values(["fighter", "event_date"]).copy()
    df = _attach_opponent_stats(df)

    df["is_win"] = (df["result"] == "W").astype(int)
    df["is_loss"] = (df["result"] == "L").astype(int)

    # Vitorias por metodo (KO/TKO e finalizacao), para as taxas historicas
    # de finalizacao. Mesmo tratamento point-in-time das demais stats.
    method_cat = df["method"].map(categorize_method) if "method" in df.columns else pd.Series(None, index=df.index)
    df["is_win_ko"] = ((df["is_win"] == 1) & (method_cat == "KO_TKO")).astype(int)
    df["is_win_sub"] = ((df["is_win"] == 1) & (method_cat == "SUBMISSION")).astype(int)

    group = df.groupby("fighter", group_keys=False)

    # Cumulativos EXCLUINDO a luta atual (shift(1) antes de acumular).
    def expanding_shifted_sum(series: pd.Series) -> pd.Series:
        return series.shift(1).expanding().sum()

    for col in ["sig_strikes_landed", "sig_strikes_attempted",
                "takedowns_landed", "takedowns_attempted",
                "sig_strikes_landed_opp", "sig_strikes_attempted_opp",
                "takedowns_landed_opp", "takedowns_attempted_opp",
                "is_win", "is_loss", "is_win_ko", "is_win_sub"]:
        df[f"cum_{col}"] = group[col].apply(expanding_shifted_sum)

    df["n_prior_fights"] = group.cumcount()  # numero de lutas ANTES desta (0, 1, 2, ...)

    with np.errstate(divide="ignore", invalid="ignore"):
        df["striking_accuracy"] = df["cum_sig_strikes_landed"] / df["cum_sig_strikes_attempted"]
        df["takedown_accuracy"] = df["cum_takedowns_landed"] / df["cum_takedowns_attempted"]
        # Defesa de queda = 1 - (quedas que o OPONENTE acertou EM MIM / quedas que o oponente tentou EM MIM)
        df["takedown_defense"] = 1 - (df["cum_takedowns_landed_opp"] / df["cum_takedowns_attempted_opp"])

    df["career_win_rate"] = df["cum_is_win"] / df["n_prior_fights"].replace(0, np.nan)
    # Taxas de finalizacao como proporcao do TOTAL de lutas anteriores
    # (nao so das vitorias): mede "quantas vezes esse lutador terminou uma
    # luta por KO/finalizacao", que embute tanto estilo quanto eficacia.
    df["ko_rate"] = df["cum_is_win_ko"] / df["n_prior_fights"].replace(0, np.nan)
    df["submission_rate"] = df["cum_is_win_sub"] / df["n_prior_fights"].replace(0, np.nan)

    # Forma recente: taxa de vitoria nas ultimas N lutas (excluindo a atual).
    def recent_win_rate(sub: pd.DataFrame) -> pd.Series:
        return sub["is_win"].shift(1).rolling(window=config.N_RECENT_FIGHTS, min_periods=1).mean()

    # NOTA: group_keys=False + apply(func que devolve Serie com o MESMO indice
    # do grupo) ja preserva o indice original de `df` na concatenacao -- nao
    # chamar reset_index aqui, ou a atribuicao abaixo desalinharia os valores
    # (o indice de `df` neste ponto NAO e um RangeIndex limpo, ja que veio de
    # um sort_values que so reordena, sem resetar os rotulos do indice).
    df["recent_win_rate"] = group.apply(recent_win_rate)

    # Dias desde a ultima luta.
    df["prev_fight_date"] = group["event_date"].shift(1)
    df["days_since_last_fight"] = (df["event_date"] - df["prev_fight_date"]).dt.days

    # Idade na data da luta (a partir da data de nascimento do bio).
    bio = fighters_bio.set_index("name")["dob"] if "name" in fighters_bio.columns else pd.Series(dtype="object")
    df["dob"] = df["fighter"].map(bio)
    df["age_years"] = (df["event_date"] - df["dob"]).dt.days / 365.25

    # Flag de dados insuficientes (poucas lutas anteriores para as medias serem confiaveis).
    df["low_experience"] = (df["n_prior_fights"] < config.MIN_FIGHTS_FOR_RELIABLE_STATS).astype(int)

    return df


def normalize_scrape_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Le fights.csv + fight_stats.csv + fighters.csv (formato produzido por
    src/data_collection.py::run_full_scrape) e devolve:
      - fights_df: uma linha por luta (fight_id, event_date, fighter_1, fighter_2, winner)
      - long_df: uma linha por (lutador, luta) com as stats daquela luta
      - fighters_bio: bio estatica por lutador (nome, altura, alcance, dob)
    """
    fights_raw = pd.read_csv(config.RAW_FIGHTS_CSV, parse_dates=["event_date"])
    fight_stats_raw = pd.read_csv(config.RAW_FIGHT_STATS_CSV)
    fighters_raw = pd.read_csv(config.RAW_FIGHTERS_CSV, parse_dates=["dob"])

    fights_raw = fights_raw.reset_index(drop=True)
    fights_raw["fight_id"] = fights_raw["fight_url"].fillna(
        fights_raw.index.to_series().astype(str) + "_" + fights_raw["event_name"].astype(str)
    )

    # Tabela longa: uma linha por lutador por luta, com o resultado dele.
    long_rows = []
    for _, row in fights_raw.iterrows():
        for fighter_col, opponent_col in (("fighter_1", "fighter_2"), ("fighter_2", "fighter_1")):
            fighter = row[fighter_col]
            opponent = row[opponent_col]
            if row["winner"] is None or (isinstance(row["winner"], float) and np.isnan(row["winner"])):
                result = "NC"
            else:
                result = "W" if row["winner"] == fighter else "L"
            long_rows.append({
                "fight_id": row["fight_id"],
                "event_date": row["event_date"],
                "fighter": fighter,
                "opponent": opponent,
                "result": result,
                "method": row.get("method"),
            })
    long_df = pd.DataFrame(long_rows)
    long_df = long_df.merge(
        fight_stats_raw[["fight_url", "fighter", "sig_strikes_landed", "sig_strikes_attempted",
                          "takedowns_landed", "takedowns_attempted"]],
        left_on=["fight_id", "fighter"], right_on=["fight_url", "fighter"], how="left",
    )

    # Nomes duplicados existem no UFC (ex.: dois "Bruno Silva"); como a chave
    # de juncao aqui e o NOME (as tabelas de luta nao carregam a URL do perfil
    # nas duas pontas), mantemos o primeiro e avisamos. Sem isso, Series.map
    # com indice duplicado quebra em compute_point_in_time_stats.
    n_dup = fighters_raw["name"].duplicated().sum()
    if n_dup:
        logger.warning("%d nome(s) de lutador duplicado(s) em fighters.csv -- mantendo a primeira ocorrencia.", n_dup)
    fighters_bio = fighters_raw.drop_duplicates(subset="name", keep="first")
    return fights_raw, long_df, fighters_bio


def build_features_from_scrape() -> pd.DataFrame:
    """Constroi o dataset final de features a partir dos dados de scraping do UFCStats.com."""
    fights_raw, long_df, fighters_bio = normalize_scrape_data()
    stats_df = compute_point_in_time_stats(long_df, fighters_bio)

    # Junta as stats point-in-time de cada lutador de volta na tabela de lutas (1 vs 2).
    pit_cols = ["fight_id", "fighter", "striking_accuracy", "takedown_accuracy",
                "takedown_defense", "career_win_rate", "recent_win_rate",
                "ko_rate", "submission_rate",
                "days_since_last_fight", "age_years", "n_prior_fights", "low_experience"]
    stats_slim = stats_df[pit_cols]

    bio_cols = ["name", "reach_cm", "height_cm"] + (["stance"] if "stance" in fighters_bio.columns else [])
    bio_slim = fighters_bio[bio_cols].rename(columns={"name": "fighter"})
    if "stance" not in bio_slim.columns:
        bio_slim["stance"] = np.nan

    merged = fights_raw.merge(stats_slim, left_on=["fight_id", "fighter_1"], right_on=["fight_id", "fighter"],
                               how="left").drop(columns=["fighter"])
    merged = merged.rename(columns={c: f"{c}_1" for c in pit_cols if c not in ("fight_id", "fighter")})
    merged = merged.merge(stats_slim, left_on=["fight_id", "fighter_2"], right_on=["fight_id", "fighter"],
                           how="left").drop(columns=["fighter"])
    merged = merged.rename(columns={c: f"{c}_2" for c in pit_cols if c not in ("fight_id", "fighter")})

    merged = merged.merge(bio_slim, left_on="fighter_1", right_on="fighter", how="left").drop(columns=["fighter"])
    merged = merged.rename(columns={"reach_cm": "reach_cm_1", "height_cm": "height_cm_1", "stance": "stance_1"})
    merged = merged.merge(bio_slim, left_on="fighter_2", right_on="fighter", how="left").drop(columns=["fighter"])
    merged = merged.rename(columns={"reach_cm": "reach_cm_2", "height_cm": "height_cm_2", "stance": "stance_2"})

    # Rating Elo pre-luta (passada cronologica global; ver src/ratings.py).
    from src.ratings import compute_elo_ratings
    elo_pre, _ = compute_elo_ratings(fights_raw)
    merged = merged.merge(elo_pre.rename(columns={"elo_1_pre": "elo_1", "elo_2_pre": "elo_2"}),
                          on="fight_id", how="left")

    return _build_mirrored_diff_rows(merged, side_suffixes=("_1", "_2"),
                                      fighter_cols=("fighter_1", "fighter_2"))


# ---------------------------------------------------------------------------
# 2) Pipeline a partir do dataset publico ja compilado (formato "largo")
# ---------------------------------------------------------------------------

# Palavras-chave usadas para localizar, de forma flexivel, as colunas
# equivalentes no dataset publico (o nome exato das colunas pode variar
# ligeiramente entre versoes/forks do dataset). Ajuste esta tabela se o
# CSV que voce baixou tiver nomes diferentes -- rode
# `python -c "import pandas as pd; print(pd.read_csv('data/raw/public_dataset.csv').columns.tolist())"`
# para ver as colunas reais do seu arquivo.
_PUBLIC_DATASET_COLUMN_HINTS = {
    "striking_accuracy": ["avg_sig_str_pct", "sig_str_acc", "str_acc"],
    "takedown_accuracy": ["avg_td_pct", "td_acc"],
    "takedown_defense": ["td_def"],
    "reach_cm": ["reach_cms", "reach"],
    "height_cm": ["height_cms", "height"],
    "age_years": ["age"],
    "recent_win_rate": ["current_win_streak", "win_streak"],
    "stance": ["stance"],
    # career_win_rate, n_prior_fights e as taxas de finalizacao sao tratados
    # a parte (via wins/losses/win_by_*), nao por este dicionario.
}


def _find_col(columns: list[str], prefix: str, keywords: list[str]) -> Optional[str]:
    """Procura, entre as colunas de um DataFrame, uma que comece com `prefix` e contenha algum keyword."""
    cols_lower = {c.lower(): c for c in columns}
    for kw in keywords:
        target = f"{prefix}{kw}".lower()
        for lower_name, original in cols_lower.items():
            if target in lower_name:
                return original
    return None


def build_features_from_public_dataset() -> pd.DataFrame:
    """
    Constroi o dataset de features a partir do dataset publico compilado
    (ex.: rajeevw/ufcdata). Esse dataset ja vem em formato "largo" com
    prefixos R_ (vermelho) / B_ (azul) e ja traz medias acumuladas ATE a
    data de cada luta -- ou seja, ja e "point-in-time" por construcao, e
    por isso aqui so precisamos calcular as diferencas R-B diretamente,
    sem recalcular medias cumulativas do zero.

    Isso e um adaptador "best effort": os nomes de coluna exatos podem
    mudar entre versoes do dataset. Onde uma coluna esperada nao e
    encontrada, o valor fica NaN (tratado depois pelo imputer no treino)
    e um aviso e logado -- confira o log apos rodar para saber se alguma
    feature ficou sem dado por causa de nome de coluna diferente.
    """
    df = pd.read_csv(config.PUBLIC_DATASET_CSV)
    cols = df.columns.tolist()

    date_col = _find_col(cols, "", ["date"]) or "date"
    winner_col = _find_col(cols, "", ["winner"]) or "Winner"
    r_name_col = _find_col(cols, "", ["r_fighter"]) or "R_fighter"
    b_name_col = _find_col(cols, "", ["b_fighter"]) or "B_fighter"

    out = pd.DataFrame()
    out["event_date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["fighter_1"] = df[r_name_col]
    out["fighter_2"] = df[b_name_col]
    out["fight_id"] = df.index.astype(str) + "_" + out["fighter_1"].astype(str) + "_vs_" + out["fighter_2"].astype(str)

    winner_raw = df[winner_col].astype(str).str.lower()
    out["winner"] = np.where(winner_raw.str.startswith("red"), out["fighter_1"],
                     np.where(winner_raw.str.startswith("blue"), out["fighter_2"], None))

    def diff_metric(name: str) -> pd.Series:
        r_col = _find_col(cols, "R_", _PUBLIC_DATASET_COLUMN_HINTS[name])
        b_col = _find_col(cols, "B_", _PUBLIC_DATASET_COLUMN_HINTS[name])
        if r_col is None or b_col is None:
            logger.warning("Nao encontrei colunas para '%s' no dataset publico (procurei prefixo R_/B_ + %s). "
                            "Feature ficara vazia (NaN).", name, _PUBLIC_DATASET_COLUMN_HINTS[name])
            return pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)
        return df[r_col], df[b_col]

    def takedown_defense(prefix: str) -> pd.Series:
        # O data.csv do rajeevw/ufcdata nao traz "TD Def" pronto, mas traz as
        # medias do que os OPONENTES fizeram contra o lutador
        # ({R,B}_avg_opp_TD_att / _landed) -- defesa = 1 - landed/att.
        direct = _find_col(cols, prefix, _PUBLIC_DATASET_COLUMN_HINTS["takedown_defense"])
        if direct is not None:
            return df[direct]
        opp_att = _find_col(cols, prefix, ["avg_opp_td_att"])
        opp_landed = _find_col(cols, prefix, ["avg_opp_td_landed"])
        if opp_att and opp_landed:
            att = df[opp_att]
            return 1 - (df[opp_landed] / att.where(att > 0))
        logger.warning("Nao encontrei colunas para takedown_defense (prefixo %s). Feature ficara vazia (NaN).", prefix)
        return pd.Series(np.nan, index=df.index)

    r_str_acc, b_str_acc = diff_metric("striking_accuracy")
    r_td_acc, b_td_acc = diff_metric("takedown_accuracy")
    r_td_def, b_td_def = takedown_defense("R_"), takedown_defense("B_")
    r_reach, b_reach = diff_metric("reach_cm")
    r_height, b_height = diff_metric("height_cm")
    r_age, b_age = diff_metric("age_years")
    r_streak, b_streak = diff_metric("recent_win_rate")

    r_wins_col = _find_col(cols, "R_", ["wins"])
    r_losses_col = _find_col(cols, "R_", ["losses"])
    b_wins_col = _find_col(cols, "B_", ["wins"])
    b_losses_col = _find_col(cols, "B_", ["losses"])
    r_wins = df[r_wins_col] if r_wins_col else pd.Series(np.nan, index=df.index)
    r_losses = df[r_losses_col] if r_losses_col else pd.Series(np.nan, index=df.index)
    b_wins = df[b_wins_col] if b_wins_col else pd.Series(np.nan, index=df.index)
    b_losses = df[b_losses_col] if b_losses_col else pd.Series(np.nan, index=df.index)

    r_total_fights = r_wins.fillna(0) + r_losses.fillna(0)
    b_total_fights = b_wins.fillna(0) + b_losses.fillna(0)

    out["striking_accuracy_diff"] = r_str_acc - b_str_acc
    out["takedown_accuracy_diff"] = r_td_acc - b_td_acc
    out["takedown_defense_diff"] = r_td_def - b_td_def
    out["reach_diff_cm"] = r_reach - b_reach
    out["height_diff_cm"] = r_height - b_height
    out["age_diff_years"] = r_age - b_age
    out["days_since_last_fight_diff"] = np.nan  # nao disponivel de forma confiavel neste dataset
    out["recent_win_rate_diff"] = r_streak - b_streak
    out["career_win_rate_diff"] = (r_wins / r_total_fights.replace(0, np.nan)) - (b_wins / b_total_fights.replace(0, np.nan))
    out["experience_diff"] = r_total_fights - b_total_fights

    # Taxas de finalizacao a partir das colunas win_by_* do dataset largo.
    def finish_rate(prefix: str, keywords: list[str], total: pd.Series) -> pd.Series:
        col = _find_col(cols, prefix, keywords)
        if col is None:
            return pd.Series(np.nan, index=df.index)
        return df[col].fillna(0) / total.replace(0, np.nan)

    out["ko_rate_diff"] = (finish_rate("R_", ["win_by_ko"], r_total_fights)
                           - finish_rate("B_", ["win_by_ko"], b_total_fights))
    out["submission_rate_diff"] = (finish_rate("R_", ["win_by_submission"], r_total_fights)
                                   - finish_rate("B_", ["win_by_submission"], b_total_fights))
    # Elo exigiria uma passada cronologica propria neste formato largo; fica
    # NaN no fallback (o caminho principal, formato scrape, calcula de verdade).
    out["elo_diff"] = np.nan

    r_stance_col = _find_col(cols, "R_", _PUBLIC_DATASET_COLUMN_HINTS["stance"])
    b_stance_col = _find_col(cols, "B_", _PUBLIC_DATASET_COLUMN_HINTS["stance"])
    if r_stance_col and b_stance_col:
        out["stance_mismatch"] = [stance_mismatch_value(a, b)
                                  for a, b in zip(df[r_stance_col], df[b_stance_col])]
    else:
        out["stance_mismatch"] = np.nan

    out["fighter_a_low_experience"] = (r_total_fights < config.MIN_FIGHTS_FOR_RELIABLE_STATS).astype(int)
    out["fighter_b_low_experience"] = (b_total_fights < config.MIN_FIGHTS_FOR_RELIABLE_STATS).astype(int)

    out["fighter_a"] = out["fighter_1"]
    out["fighter_b"] = out["fighter_2"]
    out["label"] = (out["winner"] == out["fighter_a"]).astype(int)
    out.loc[out["winner"].isna(), "label"] = np.nan

    row_ab = out[["fight_id", "event_date", "fighter_a", "fighter_b", "label"] + FEATURE_COLUMNS].copy()

    row_ba = row_ab.copy()
    row_ba["fighter_a"], row_ba["fighter_b"] = row_ab["fighter_b"], row_ab["fighter_a"]
    row_ba["label"] = 1 - row_ab["label"]
    for col in FEATURE_COLUMNS:
        if col in MIRROR_NON_NEGATED_COLUMNS:
            continue
        row_ba[col] = -row_ab[col]
    row_ba["fighter_a_low_experience"] = row_ab["fighter_b_low_experience"]
    row_ba["fighter_b_low_experience"] = row_ab["fighter_a_low_experience"]

    result = pd.concat([row_ab, row_ba], ignore_index=True)
    result = result.dropna(subset=["label"])  # descarta empates/no-contest (sem vencedor definido)
    return result.sort_values("event_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3) Helper compartilhado: gerar as linhas espelhadas (A-B) e (B-A)
# ---------------------------------------------------------------------------

def _build_mirrored_diff_rows(merged: pd.DataFrame, side_suffixes: tuple[str, str],
                               fighter_cols: tuple[str, str]) -> pd.DataFrame:
    """Usado pelo pipeline de scraping para transformar a tabela 'lutador 1 vs lutador 2' em linhas diferenciais espelhadas."""
    s1, s2 = side_suffixes
    f1, f2 = fighter_cols

    def diffs(a_suffix: str, b_suffix: str, fa: str, fb: str, sign: int) -> pd.DataFrame:
        out = pd.DataFrame()
        out["fight_id"] = merged["fight_id"]
        out["event_date"] = merged["event_date"]
        out["fighter_a"] = merged[fa]
        out["fighter_b"] = merged[fb]
        # Importante: se nao houve vencedor definido (empate/no-contest), o label
        # deve ficar NaN (para ser descartado a seguir), nao False/0 -- uma
        # comparacao direta "merged['winner'] == merged[fa]" trataria NaN vs.
        # string como False, rotulando erradamente esses casos como derrota.
        out["label"] = np.where(merged["winner"].isna(), np.nan,
                                 (merged["winner"] == merged[fa]).astype(float))
        out["striking_accuracy_diff"] = sign * (merged[f"striking_accuracy{a_suffix}"] - merged[f"striking_accuracy{b_suffix}"])
        out["takedown_accuracy_diff"] = sign * (merged[f"takedown_accuracy{a_suffix}"] - merged[f"takedown_accuracy{b_suffix}"])
        out["takedown_defense_diff"] = sign * (merged[f"takedown_defense{a_suffix}"] - merged[f"takedown_defense{b_suffix}"])
        out["reach_diff_cm"] = sign * (merged[f"reach_cm{a_suffix}"] - merged[f"reach_cm{b_suffix}"])
        out["height_diff_cm"] = sign * (merged[f"height_cm{a_suffix}"] - merged[f"height_cm{b_suffix}"])
        out["age_diff_years"] = sign * (merged[f"age_years{a_suffix}"] - merged[f"age_years{b_suffix}"])
        out["days_since_last_fight_diff"] = sign * (merged[f"days_since_last_fight{a_suffix}"] - merged[f"days_since_last_fight{b_suffix}"])
        out["recent_win_rate_diff"] = sign * (merged[f"recent_win_rate{a_suffix}"] - merged[f"recent_win_rate{b_suffix}"])
        out["career_win_rate_diff"] = sign * (merged[f"career_win_rate{a_suffix}"] - merged[f"career_win_rate{b_suffix}"])
        out["experience_diff"] = sign * (merged[f"n_prior_fights{a_suffix}"] - merged[f"n_prior_fights{b_suffix}"])
        out["ko_rate_diff"] = sign * (merged[f"ko_rate{a_suffix}"] - merged[f"ko_rate{b_suffix}"])
        out["submission_rate_diff"] = sign * (merged[f"submission_rate{a_suffix}"] - merged[f"submission_rate{b_suffix}"])
        out["elo_diff"] = sign * (merged[f"elo{a_suffix}"] - merged[f"elo{b_suffix}"])
        # Simetrica: mesmo valor nas duas linhas espelhadas (nao inverte sinal).
        out["stance_mismatch"] = [
            stance_mismatch_value(sa, sb)
            for sa, sb in zip(merged[f"stance{a_suffix}"], merged[f"stance{b_suffix}"])
        ]
        # Somas simetricas (fase 2: metodo/round). a+b == b+a, entao o valor e
        # naturalmente identico nas duas linhas espelhadas.
        out["ko_rate_sum"] = merged[f"ko_rate{a_suffix}"] + merged[f"ko_rate{b_suffix}"]
        out["submission_rate_sum"] = merged[f"submission_rate{a_suffix}"] + merged[f"submission_rate{b_suffix}"]
        out["striking_accuracy_sum"] = merged[f"striking_accuracy{a_suffix}"] + merged[f"striking_accuracy{b_suffix}"]
        out["takedown_accuracy_sum"] = merged[f"takedown_accuracy{a_suffix}"] + merged[f"takedown_accuracy{b_suffix}"]
        out["career_win_rate_sum"] = merged[f"career_win_rate{a_suffix}"] + merged[f"career_win_rate{b_suffix}"]
        out["experience_total"] = merged[f"n_prior_fights{a_suffix}"] + merged[f"n_prior_fights{b_suffix}"]
        out["fighter_a_low_experience"] = merged[f"low_experience{a_suffix}"]
        out["fighter_b_low_experience"] = merged[f"low_experience{b_suffix}"]
        return out

    row_ab = diffs(s1, s2, f1, f2, sign=1)
    row_ba = diffs(s2, s1, f2, f1, sign=1)  # ja invertido pois trocamos os suffixes/fighters

    result = pd.concat([row_ab, row_ba], ignore_index=True)
    result = result.dropna(subset=["label"])
    result["label"] = result["label"].astype(int)
    return result.sort_values("event_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4) Nivel ATUAL de cada lutador (para prever uma luta futura ainda nao ocorrida)
# ---------------------------------------------------------------------------

_CURRENT_LEVEL_COLUMNS = [
    "fighter", "striking_accuracy", "takedown_accuracy", "takedown_defense",
    "reach_cm", "height_cm", "age_years", "days_since_last_fight",
    "recent_win_rate", "career_win_rate", "ko_rate", "submission_rate",
    "elo", "stance", "n_prior_fights", "low_experience",
]


def compute_current_levels(long_df: pd.DataFrame, fighters_bio: pd.DataFrame) -> pd.DataFrame:
    """
    Como compute_point_in_time_stats, mas SEM excluir a luta mais recente
    (usa o historico completo, incluindo a ultima luta) -- porque aqui
    queremos o nivel ATUAL de cada lutador (depois da ultima luta
    registrada), para montar features de uma luta FUTURA hipotetica.
    Tambem recalcula idade e dias-desde-ultima-luta em relacao a HOJE,
    e nao em relacao a uma proxima luta (que nesse caso ainda nao existe).
    """
    df = long_df.sort_values(["fighter", "event_date"]).copy()
    df = _attach_opponent_stats(df)
    df["is_win"] = (df["result"] == "W").astype(int)

    method_cat = df["method"].map(categorize_method) if "method" in df.columns else pd.Series(None, index=df.index)
    df["is_win_ko"] = ((df["is_win"] == 1) & (method_cat == "KO_TKO")).astype(int)
    df["is_win_sub"] = ((df["is_win"] == 1) & (method_cat == "SUBMISSION")).astype(int)

    group = df.groupby("fighter", group_keys=False)
    for col in ["sig_strikes_landed", "sig_strikes_attempted",
                "takedowns_landed", "takedowns_attempted",
                "sig_strikes_landed_opp", "sig_strikes_attempted_opp",
                "takedowns_landed_opp", "takedowns_attempted_opp",
                "is_win", "is_win_ko", "is_win_sub"]:
        df[f"cum_{col}"] = group[col].apply(lambda s: s.expanding().sum())

    df["n_fights"] = group.cumcount() + 1

    with np.errstate(divide="ignore", invalid="ignore"):
        df["striking_accuracy"] = df["cum_sig_strikes_landed"] / df["cum_sig_strikes_attempted"]
        df["takedown_accuracy"] = df["cum_takedowns_landed"] / df["cum_takedowns_attempted"]
        df["takedown_defense"] = 1 - (df["cum_takedowns_landed_opp"] / df["cum_takedowns_attempted_opp"])
    df["career_win_rate"] = df["cum_is_win"] / df["n_fights"]
    df["ko_rate"] = df["cum_is_win_ko"] / df["n_fights"]
    df["submission_rate"] = df["cum_is_win_sub"] / df["n_fights"]

    def recent_win_rate(sub: pd.DataFrame) -> pd.Series:
        return sub["is_win"].rolling(window=config.N_RECENT_FIGHTS, min_periods=1).mean()

    df["recent_win_rate"] = group.apply(recent_win_rate)  # ver nota em compute_point_in_time_stats

    latest = df.sort_values("event_date").groupby("fighter").tail(1).copy()

    bio = fighters_bio.set_index("name") if "name" in fighters_bio.columns else pd.DataFrame()
    today = pd.Timestamp.now().normalize()
    latest["days_since_last_fight"] = (today - latest["event_date"]).dt.days
    latest["dob"] = latest["fighter"].map(bio["dob"]) if "dob" in bio.columns else pd.NaT
    latest["age_years"] = (today - latest["dob"]).dt.days / 365.25
    latest["reach_cm"] = latest["fighter"].map(bio["reach_cm"]) if "reach_cm" in bio.columns else np.nan
    latest["height_cm"] = latest["fighter"].map(bio["height_cm"]) if "height_cm" in bio.columns else np.nan
    latest["stance"] = latest["fighter"].map(bio["stance"]) if "stance" in bio.columns else np.nan
    latest["low_experience"] = (latest["n_fights"] < config.MIN_FIGHTS_FOR_RELIABLE_STATS).astype(int)
    latest = latest.rename(columns={"n_fights": "n_prior_fights"})

    # Elo atual e preenchido por export_latest_fighter_levels (precisa da
    # tabela de LUTAS, nao da tabela longa por lutador que temos aqui).
    if "elo" not in latest.columns:
        latest["elo"] = np.nan

    return latest[_CURRENT_LEVEL_COLUMNS]


def _export_latest_levels_from_public_dataset() -> pd.DataFrame:
    """
    Aproximacao equivalente para quem esta usando o dataset publico: usa a
    ultima linha (como R_ ou como B_) em que cada lutador aparece, com as
    medias acumuladas que o dataset ja fornece "antes" daquela luta. E uma
    aproximacao (fica levemente desatualizada em relacao a ultima luta em
    si) -- suficiente para uma v1, mas o caminho via scraping
    (compute_current_levels) e mais preciso porque e recalculado do zero.
    """
    df = pd.read_csv(config.PUBLIC_DATASET_CSV)
    cols = df.columns.tolist()
    date_col = _find_col(cols, "", ["date"]) or "date"
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce")

    r_name_col = _find_col(cols, "", ["r_fighter"]) or "R_fighter"
    b_name_col = _find_col(cols, "", ["b_fighter"]) or "B_fighter"

    records = []
    for prefix, name_col in (("R_", r_name_col), ("B_", b_name_col)):
        sub = pd.DataFrame({"fighter": df[name_col], "_date": df["_date"]})
        for canonical, hints in _PUBLIC_DATASET_COLUMN_HINTS.items():
            col = _find_col(cols, prefix, hints)
            sub[canonical] = df[col] if col else np.nan
        if sub["takedown_defense"].isna().all():
            # mesmo fallback do build_features_from_public_dataset: derivar a
            # defesa de queda das medias do que os oponentes tentaram/acertaram
            opp_att_col = _find_col(cols, prefix, ["avg_opp_td_att"])
            opp_landed_col = _find_col(cols, prefix, ["avg_opp_td_landed"])
            if opp_att_col and opp_landed_col:
                att = df[opp_att_col]
                sub["takedown_defense"] = 1 - (df[opp_landed_col] / att.where(att > 0))
        wins_col = _find_col(cols, prefix, ["wins"])
        losses_col = _find_col(cols, prefix, ["losses"])
        wins = df[wins_col].fillna(0) if wins_col else pd.Series(0, index=df.index)
        losses = df[losses_col].fillna(0) if losses_col else pd.Series(0, index=df.index)
        sub["n_prior_fights"] = wins + losses
        total = (wins + losses).replace(0, np.nan)
        sub["career_win_rate"] = wins / total
        ko_col = _find_col(cols, prefix, ["win_by_ko"])
        sub_col = _find_col(cols, prefix, ["win_by_submission"])
        sub["ko_rate"] = (df[ko_col].fillna(0) / total) if ko_col else np.nan
        sub["submission_rate"] = (df[sub_col].fillna(0) / total) if sub_col else np.nan
        records.append(sub)

    combined = pd.concat(records, ignore_index=True)
    combined = combined.sort_values("_date").groupby("fighter").tail(1).copy()
    combined["days_since_last_fight"] = (pd.Timestamp.now().normalize() - combined["_date"]).dt.days
    combined["low_experience"] = (combined["n_prior_fights"] < config.MIN_FIGHTS_FOR_RELIABLE_STATS).astype(int)
    for col in _CURRENT_LEVEL_COLUMNS:
        if col not in combined.columns:
            combined[col] = np.nan
    return combined[_CURRENT_LEVEL_COLUMNS]


def export_latest_fighter_levels() -> pd.DataFrame:
    """
    Ponto de entrada usado por src/predict.py: devolve, para cada lutador
    conhecido na base, seu "nivel atual" (apos a ultima luta registrada),
    pronto para montar as features diferenciais de uma luta futura
    hipotetica entre dois lutadores quaisquer.
    """
    if config.RAW_FIGHTS_CSV.exists() and config.RAW_FIGHT_STATS_CSV.exists() and config.RAW_FIGHTERS_CSV.exists():
        fights_raw, long_df, fighters_bio = normalize_scrape_data()
        levels = compute_current_levels(long_df, fighters_bio)
        # Elo ATUAL (apos a ultima luta registrada) de cada lutador.
        from src.ratings import compute_elo_ratings
        _, current_elo = compute_elo_ratings(fights_raw)
        levels["elo"] = levels["fighter"].map(current_elo)
        return levels
    elif config.PUBLIC_DATASET_CSV.exists():
        return _export_latest_levels_from_public_dataset()
    raise FileNotFoundError("Nenhum dado bruto encontrado. Rode primeiro src/data_collection.py.")


# ---------------------------------------------------------------------------
# 5) Ponto de entrada unico
# ---------------------------------------------------------------------------

def build_feature_dataset(save: bool = True) -> pd.DataFrame:
    """
    Detecta automaticamente qual fonte de dados brutos esta disponivel
    (prioriza o scraping granular, cai para o dataset publico) e constroi
    o dataset final de features, salvando em data/processed/fight_features.csv.
    """
    if config.RAW_FIGHTS_CSV.exists() and config.RAW_FIGHT_STATS_CSV.exists() and config.RAW_FIGHTERS_CSV.exists():
        logger.info("Usando dados de scraping (%s).", config.RAW_DIR)
        feature_df = build_features_from_scrape()
    elif config.PUBLIC_DATASET_CSV.exists():
        logger.info("Usando dataset publico compilado (%s).", config.PUBLIC_DATASET_CSV)
        feature_df = build_features_from_public_dataset()
    else:
        raise FileNotFoundError(
            "Nenhum dado bruto encontrado. Rode primeiro: "
            "python -m src.data_collection --source public-dataset  (ou --source scrape)"
        )

    if save:
        feature_df.to_csv(config.FEATURES_CSV, index=False)
        logger.info("Features salvas em %s (%d linhas, %d lutas unicas)",
                    config.FEATURES_CSV, len(feature_df), feature_df["fight_id"].nunique())
    return feature_df


if __name__ == "__main__":
    build_feature_dataset()
