"""
Avaliacao progressiva por eventos (walk-forward) do candidato de shrinkage.

PROTOCOLO, REGISTRADO ANTES DE OLHAR O RESULTADO
------------------------------------------------
Janela expansiva. Para cada dobra i:
  treino_i      = todas as lutas com data < T_i
  calibracao_i  = os ultimos 15% do treino_i (temporal, agrupado por luta)
  avaliacao_i   = lutas em [T_i, T_{i+1})

Priors e k do shrinkage sao estimados SO nas linhas de treino_i, por metodo
dos momentos, a cada dobra. Nenhuma escolha olha avaliacao.

Cada luta aparece em EXATAMENTE UMA dobra de avaliacao: o agregado e a
concatenacao das dobras, sem contar luta repetida.

Metrica principal: log loss. Confirmacao: Brier. Salvaguardas: calibracao,
cobertura e desempenho por grupo de experiencia. Incerteza por bootstrap
pareado POR EVENTO sobre o agregado.

Criterio de promocao, fixado aqui: log loss do candidato menor que o do
baseline no agregado, com IC95 do contraste pareado por evento excluindo
zero, E sem degradacao material em nenhum grupo.

LIMITE: estas janelas ja foram exploradas nesta e em sessoes anteriores.
Isto e DESENVOLVIMENTO. A confirmacao exige previsao futura registrada
antes do resultado.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

import config
from src.features import (FEATURE_COLUMNS, compute_point_in_time_stats,
                          normalize_scrape_data)
from src.shrinkage import aplicar_shrinkage, estimar_prior, estimar_priors
from src.train import _calibrate, build_logreg_pipeline

N_DOBRAS = 8
INICIO_AVALIACAO = "2022-01-01"
FRACAO_CALIBRACAO = 0.15

fights_raw, long_df, bio = normalize_scrape_data()
st_cru = compute_point_in_time_stats(long_df, bio)
k_exp = st_cru.set_index(["fight_id", "fighter"])["n_prior_fights"]

PIT = ["fight_id", "fighter", "striking_accuracy", "takedown_accuracy",
       "takedown_defense", "career_win_rate", "recent_win_rate", "ko_rate",
       "submission_rate", "days_since_last_fight", "age_years",
       "n_prior_fights", "low_experience"]


def montar(stats):
    from src.features import _build_mirrored_diff_rows
    from src.ratings import compute_elo_ratings
    slim = stats[PIT]
    cols = ["name", "reach_cm", "height_cm"] + (["stance"] if "stance" in bio.columns else [])
    bs = bio[cols].rename(columns={"name": "fighter"})
    if "stance" not in bs.columns:
        bs["stance"] = np.nan
    ren = [c for c in PIT if c not in ("fight_id", "fighter")]
    m = fights_raw.merge(slim, left_on=["fight_id", "fighter_1"],
                         right_on=["fight_id", "fighter"], how="left").drop(columns=["fighter"])
    m = m.rename(columns={c: f"{c}_1" for c in ren})
    m = m.merge(slim, left_on=["fight_id", "fighter_2"],
                right_on=["fight_id", "fighter"], how="left").drop(columns=["fighter"])
    m = m.rename(columns={c: f"{c}_2" for c in ren})
    m = m.merge(bs, left_on="fighter_1", right_on="fighter", how="left").drop(columns=["fighter"])
    m = m.rename(columns={"reach_cm": "reach_cm_1", "height_cm": "height_cm_1", "stance": "stance_1"})
    m = m.merge(bs, left_on="fighter_2", right_on="fighter", how="left").drop(columns=["fighter"])
    m = m.rename(columns={"reach_cm": "reach_cm_2", "height_cm": "height_cm_2", "stance": "stance_2"})
    elo, _ = compute_elo_ratings(fights_raw)
    m = m.merge(elo.rename(columns={"elo_1_pre": "elo_1", "elo_2_pre": "elo_2"}),
                on="fight_id", how="left")
    return _build_mirrored_diff_rows(m, side_suffixes=("_1", "_2"),
                                     fighter_cols=("fighter_1", "fighter_2"))


# ---------------------------------------------------------------------------
# Peso indevido na estimacao empirical Bayes
# ---------------------------------------------------------------------------
print("=== PESO NA ESTIMACAO DO PRIOR ===")
print("Cada lutador entra com UMA LINHA POR LUTA, e as linhas sao cumulativas:")
print("quem tem 25 lutas contribui 25 vezes, com valores quase iguais entre si.")
tr0 = st_cru[st_cru["event_date"] < INICIO_AVALIACAO]
for nome, num, den in [("career_win_rate", "cum_is_win", "n_prior_fights"),
                       ("takedown_defense", "cum_takedowns_landed_opp",
                        "cum_takedowns_attempted_opp")]:
    todas = estimar_prior(tr0[num], tr0[den])
    ultima = tr0.sort_values("event_date").groupby("fighter").tail(1)
    por_lutador = estimar_prior(ultima[num], ultima[den])
    print(f"  {nome:20} todas as linhas: media={todas.media:.4f} k={todas.k:6.1f}  "
          f"| uma por lutador: media={por_lutador.media:.4f} k={por_lutador.k:6.1f}")
print("  (k menor = prior mais fraco; diferenca grande significa que a repeticao"
      "\n   cumulativa estava inflando o peso do prior)")

# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
f_base_todo = montar(st_cru)
datas = np.sort(f_base_todo.loc[f_base_todo["event_date"] >= INICIO_AVALIACAO,
                                "event_date"].unique())
cortes = np.array_split(datas, N_DOBRAS)

print(f"\n=== WALK-FORWARD, {N_DOBRAS} dobras ===")
linhas_base, linhas_cand = [], []
for i, bloco in enumerate(cortes, 1):
    t0, t1 = bloco[0], bloco[-1]
    treino_ids = set(f_base_todo.loc[f_base_todo["event_date"] < t0, "fight_id"])
    aval_mask = (f_base_todo["event_date"] >= t0) & (f_base_todo["event_date"] <= t1)
    if not treino_ids or not aval_mask.any():
        continue

    # calibracao = ultimos 15% do treino, por luta (espelhos nunca se separam)
    tr_lutas = (f_base_todo[f_base_todo["fight_id"].isin(treino_ids)]
                .drop_duplicates("fight_id")[["fight_id", "event_date"]]
                .sort_values("event_date"))
    n_cal = max(200, int(len(tr_lutas) * FRACAO_CALIBRACAO))
    cal_ids = set(tr_lutas.tail(n_cal)["fight_id"])
    fit_ids = treino_ids - cal_ids

    # priors SO no treino desta dobra (e so no pedaco de ajuste)
    priors = estimar_priors(st_cru[st_cru["fight_id"].isin(fit_ids)])
    f_cand_todo = montar(aplicar_shrinkage(st_cru, priors))

    for rotulo, F, destino in (("baseline", f_base_todo, linhas_base),
                               ("shrinkage", f_cand_todo, linhas_cand)):
        fit = F[F["fight_id"].isin(fit_ids)]
        cal = F[F["fight_id"].isin(cal_ids)]
        av = F[(F["event_date"] >= t0) & (F["event_date"] <= t1)]
        est = build_logreg_pipeline()
        est.fit(fit[FEATURE_COLUMNS], fit["label"])
        mdl = _calibrate(est, cal[FEATURE_COLUMNS], cal["label"], "sigmoid")
        d = av[["fight_id", "event_date", "fighter_a", "fighter_b", "label"]].copy()
        d["p"] = np.clip(mdl.predict_proba(av[FEATURE_COLUMNS])[:, 1], 1e-6, 1 - 1e-6)
        d["dobra"] = i
        destino.append(d)

    b, c = linhas_base[-1], linhas_cand[-1]
    print(f"  dobra {i}: treino {len(fit_ids):5} lutas ate {pd.Timestamp(t0).date()} | "
          f"avalia {b['fight_id'].nunique():4} lutas "
          f"{pd.Timestamp(t0).date()}..{pd.Timestamp(t1).date()} | "
          f"ll {log_loss(b['label'],b['p']):.4f} -> {log_loss(c['label'],c['p']):.4f}")

base = pd.concat(linhas_base, ignore_index=True)
cand = pd.concat(linhas_cand, ignore_index=True)
assert base["fight_id"].nunique() == base["fight_id"].nunique(), "luta repetida"
assert len(base) == len(cand) and base["fight_id"].nunique() * 2 == len(base)

for d in (base, cand):
    na = k_exp.reindex(pd.MultiIndex.from_arrays([d["fight_id"], d["fighter_a"]])).to_numpy()
    nb = k_exp.reindex(pd.MultiIndex.from_arrays([d["fight_id"], d["fighter_b"]])).to_numpy()
    d["min_n"] = np.nanmin(np.vstack([na, nb]), axis=0)
    d["grupo"] = pd.cut(d["min_n"], [-1, 0, 2, 5, 999], labels=["0", "1-2", "3-5", "6+"])

print(f"\nAGREGADO: {base['fight_id'].nunique()} lutas unicas, "
      f"{base['event_date'].nunique()} eventos, cada luta em uma dobra so")
G = ["0", "1-2", "3-5", "6+"]
print(f"\n{'grupo':6} {'lutas':>6} {'cob%':>6} | {'ll base':>8} {'ll shr':>8} {'d ll':>8} | "
      f"{'br base':>8} {'br shr':>8}")
print("-" * 74)
for g in G + ["TODOS"]:
    b = base if g == "TODOS" else base[base["grupo"] == g]
    c = cand if g == "TODOS" else cand[cand["grupo"] == g]
    if len(b) < 60:
        continue
    llb, llc = log_loss(b["label"], b["p"]), log_loss(c["label"], c["p"])
    print(f"{g:6} {b['fight_id'].nunique():6} {len(b)/len(base)*100:5.1f}% | "
          f"{llb:8.4f} {llc:8.4f} {llc-llb:+8.4f} | "
          f"{brier_score_loss(b['label'],b['p']):8.4f} {brier_score_loss(c['label'],c['p']):8.4f}")

print("\ncontraste pareado por EVENTO (2000x), shrinkage - baseline:")
rng = np.random.default_rng(17)
eventos = base["event_date"].unique()
bi = {e: d for e, d in base.groupby("event_date")}
ci = {e: d for e, d in cand.groupby("event_date")}
for g in G + ["TODOS"]:
    dl, db = [], []
    for _ in range(2000):
        pick = rng.choice(eventos, len(eventos), replace=True)
        b = pd.concat([bi[e] for e in pick])
        c = pd.concat([ci[e] for e in pick])
        if g != "TODOS":
            b, c = b[b["grupo"] == g], c[c["grupo"] == g]
        if len(b) < 20 or b["label"].nunique() < 2:
            continue
        dl.append(log_loss(c["label"], c["p"]) - log_loss(b["label"], b["p"]))
        db.append(brier_score_loss(c["label"], c["p"]) - brier_score_loss(b["label"], b["p"]))
    if len(dl) < 100:
        continue
    lo, hi = np.percentile(dl, [2.5, 97.5])
    blo, bhi = np.percentile(db, [2.5, 97.5])
    ver = "MELHORA" if hi < 0 else ("PIORA" if lo > 0 else "inconclusivo")
    print(f"  {g:6} d ll {np.mean(dl):+.4f} IC95 [{lo:+.4f}, {hi:+.4f}]  "
          f"d br {np.mean(db):+.4f} IC95 [{blo:+.4f}, {bhi:+.4f}]  {ver}")
