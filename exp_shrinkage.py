"""Experimento §6B: shrinkage das taxas, avaliado em cal_select, por grupo."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import config
from src.features import (FEATURE_COLUMNS, compute_point_in_time_stats,
                          normalize_scrape_data)
from src.shrinkage import aplicar_shrinkage, estimar_priors
from src.train import (_calibrate, _get_gbm_model, build_logreg_pipeline,
                       split_calibration_slice, temporal_group_split)
from src.tuning import BACKTEST_TRUNCATE_DATE

fights_raw, long_df, bio = normalize_scrape_data()
st = compute_point_in_time_stats(long_df, bio)
st = st[st["event_date"] <= BACKTEST_TRUNCATE_DATE]

# experiencia do lado menos experiente, para o corte por grupo
k_exp = st.set_index(["fight_id", "fighter"])["n_prior_fights"]


PIT = ["fight_id", "fighter", "striking_accuracy", "takedown_accuracy",
       "takedown_defense", "career_win_rate", "recent_win_rate", "ko_rate",
       "submission_rate", "days_since_last_fight", "age_years",
       "n_prior_fights", "low_experience"]


def montar(stats):
    """stats (long) -> features espelhadas. Mesmos passos de
    build_features_from_scrape, que nao aceita stats injetadas."""
    from src.features import _build_mirrored_diff_rows
    from src.ratings import compute_elo_ratings

    slim = stats[PIT]
    bio_cols = ["name", "reach_cm", "height_cm"] + (["stance"] if "stance" in bio.columns else [])
    bio_slim = bio[bio_cols].rename(columns={"name": "fighter"})
    if "stance" not in bio_slim.columns:
        bio_slim["stance"] = np.nan

    ren = {c: None for c in PIT if c not in ("fight_id", "fighter")}
    m = fights_raw.merge(slim, left_on=["fight_id", "fighter_1"],
                         right_on=["fight_id", "fighter"], how="left").drop(columns=["fighter"])
    m = m.rename(columns={c: f"{c}_1" for c in ren})
    m = m.merge(slim, left_on=["fight_id", "fighter_2"],
                right_on=["fight_id", "fighter"], how="left").drop(columns=["fighter"])
    m = m.rename(columns={c: f"{c}_2" for c in ren})
    m = m.merge(bio_slim, left_on="fighter_1", right_on="fighter", how="left").drop(columns=["fighter"])
    m = m.rename(columns={"reach_cm": "reach_cm_1", "height_cm": "height_cm_1", "stance": "stance_1"})
    m = m.merge(bio_slim, left_on="fighter_2", right_on="fighter", how="left").drop(columns=["fighter"])
    m = m.rename(columns={"reach_cm": "reach_cm_2", "height_cm": "height_cm_2", "stance": "stance_2"})
    elo_pre, _ = compute_elo_ratings(fights_raw)
    m = m.merge(elo_pre.rename(columns={"elo_1_pre": "elo_1", "elo_2_pre": "elo_2"}),
                on="fight_id", how="left")
    f = _build_mirrored_diff_rows(m, side_suffixes=("_1", "_2"),
                                  fighter_cols=("fighter_1", "fighter_2"))
    return f[f["event_date"] <= BACKTEST_TRUNCATE_DATE]


def avaliar(f, rotulo):
    train_df, cal_df, _teste_nunca_tocado = temporal_group_split(f)
    cal_fit, cal_select = split_calibration_slice(cal_df)
    Xtr, ytr = train_df[FEATURE_COLUMNS], train_df["label"]
    Xf, yf = cal_fit[FEATURE_COLUMNS], cal_fit["label"]
    est = build_logreg_pipeline()
    est.fit(Xtr, ytr)
    cal = _calibrate(est, Xf, yf, "sigmoid")
    out = cal_select[["fight_id", "event_date", "fighter_a", "fighter_b", "label"]].copy()
    out["p"] = np.clip(cal.predict_proba(cal_select[FEATURE_COLUMNS])[:, 1], 1e-6, 1 - 1e-6)
    out["rotulo"] = rotulo
    return out, train_df


# ---- baseline -------------------------------------------------------------
f_base = montar(st)
base, train_df = avaliar(f_base, "baseline")

# ---- candidato: priors estimados SO no treino desta janela ----------------
ids_treino = set(train_df["fight_id"])
treino_long = st[st["fight_id"].isin(ids_treino)]
print(f"priors estimados em {len(treino_long)} linhas de TREINO "
      f"({treino_long['event_date'].min().date()} a {treino_long['event_date'].max().date()})")
priors = estimar_priors(treino_long)
f_shr = montar(aplicar_shrinkage(st, priors))
cand, _ = avaliar(f_shr, "shrinkage")

# ---- grupos ---------------------------------------------------------------
for d in (base, cand):
    na = k_exp.reindex(pd.MultiIndex.from_arrays([d["fight_id"], d["fighter_a"]])).to_numpy()
    nb = k_exp.reindex(pd.MultiIndex.from_arrays([d["fight_id"], d["fighter_b"]])).to_numpy()
    d["min_n"] = np.nanmin(np.vstack([na, nb]), axis=0)
    d["grupo"] = pd.cut(d["min_n"], [-1, 0, 2, 5, 999], labels=["0", "1-2", "3-5", "6+"])

G = ["0", "1-2", "3-5", "6+"]
print(f"\ncal_select: {len(base)} linhas, {base['fight_id'].nunique()} lutas\n")
print(f"{'grupo':6} {'lutas':>6} | {'ll base':>8} {'ll shr':>8} {'d ll':>8} | "
      f"{'br base':>8} {'br shr':>8} | {'AUC base':>9} {'AUC shr':>8}")
print("-" * 86)
for g in G + ["TODOS"]:
    b = base if g == "TODOS" else base[base["grupo"] == g]
    c = cand if g == "TODOS" else cand[cand["grupo"] == g]
    if len(b) < 40:
        print(f"{g:6} {b['fight_id'].nunique():6} | amostra pequena demais")
        continue
    llb, llc = log_loss(b["label"], b["p"]), log_loss(c["label"], c["p"])
    print(f"{g:6} {b['fight_id'].nunique():6} | {llb:8.4f} {llc:8.4f} {llc-llb:+8.4f} | "
          f"{brier_score_loss(b['label'],b['p']):8.4f} {brier_score_loss(c['label'],c['p']):8.4f} | "
          f"{roc_auc_score(b['label'],b['p']):9.4f} {roc_auc_score(c['label'],c['p']):8.4f}")

# ---- contraste pareado, bootstrap por evento ------------------------------
print("\ncontraste shrinkage - baseline (mesmos eventos reamostrados, 2000x):")
rng = np.random.default_rng(13)
eventos = base["event_date"].unique()
bi = {e: d for e, d in base.groupby("event_date")}
ci = {e: d for e, d in cand.groupby("event_date")}
for g in G + ["TODOS"]:
    difs = []
    for _ in range(2000):
        pick = rng.choice(eventos, len(eventos), replace=True)
        b = pd.concat([bi[e] for e in pick])
        c = pd.concat([ci[e] for e in pick])
        if g != "TODOS":
            b, c = b[b["grupo"] == g], c[c["grupo"] == g]
        if len(b) < 20 or b["label"].nunique() < 2:
            continue
        difs.append(log_loss(c["label"], c["p"]) - log_loss(b["label"], b["p"]))
    if len(difs) < 100:
        print(f"  {g:6} amostra insuficiente")
        continue
    lo, hi = np.percentile(difs, [2.5, 97.5])
    print(f"  {g:6} d log loss {np.mean(difs):+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]  "
          f"{'MELHORA' if hi < 0 else ('PIORA' if lo > 0 else 'inconclusivo')}")
