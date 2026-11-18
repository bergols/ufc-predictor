"""Auditoria do CLV (README, regra 1, 0b00315). Somente leitura.

CRITERIOS DE VALIDADE -- fixados antes de calcular qualquer media:
  P  populacao da regra: linha com model_side, sharp_prob, close_prob e clv
     preenchidos (perna no lado do modelo, de qualquer luta capturada).
  V1 aritmetica: clv == close_prob - sharp_prob (tolerancia 1e-4).
  V2 unicidade: uma linha por (event_date, par de lutadores); prediction_id
     unico quando existe.
  V3 prova da entrada: sharp_ref resolve em cotacoes.csv, corpo confere pelo
     hash, e sharp_fair_prob(Pinnacle, model_side) recalculado do corpo ==
     sharp_prob (5e-5).
  V4 prova do fechamento: idem com close_ref e close_prob.
  V5 tempo: captura da entrada < captura do fechamento < inicio programado
     da luta (do proprio corpo da chamada de fechamento).
  V6 mesma casa nas duas pontas: as duas probabilidades da Pinnacle (V3+V4).
Populacao VERIFICAVEL = P com V1..V6. Nada e comprovado retroativamente.
"""
import gzip
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src.line_shopping import sharp_fair_prob  # noqa: E402
from src.proveniencia import _mesmo_nome, conferir  # noqa: E402

ler = dict(dtype=str, keep_default_na=False)
h = pd.read_csv("data/prediction_history.csv", **ler)
cot = pd.read_csv("data/proveniencia/cotacoes.csv", **ler)
num = lambda s: pd.to_numeric(s, errors="coerce")  # noqa: E731
P = h[(h.model_side != "") & (h.sharp_prob != "") & (h.close_prob != "") & (h.clv != "")].copy()
print(f"P: {len(P)} pernas, {P.event_date.nunique()} eventos "
      f"({P.event_date.min()} a {P.event_date.max()})")
print("linhas com clv mas sem sharp/close:", int(((h.clv != "") & ((h.sharp_prob == "") | (h.close_prob == ""))).sum()))

_corpo = {}


def corpo(ref):
    if ref not in _corpo:
        _corpo[ref] = json.load(gzip.open(f"data/proveniencia/{ref}.json.gz", "rt", encoding="utf-8"))
    return _corpo[ref]


def prova(ref, r, prob_col):
    """(ok, motivo, captured_at, inicio)"""
    if not ref:
        return False, "sem ref (legado)", None, None
    c = cot[cot.ref == ref]
    if len(c) != 1:
        return False, f"ref em {len(c)} cotacoes", None, None
    c = c.iloc[0]
    ok, mot = conferir(c.chamada_ref)
    if not ok:
        return False, f"corpo: {mot}", None, None
    evs = [e for e in corpo(c.chamada_ref) if str(e.get("id")) == c.evento_id]
    if len(evs) != 1:
        return False, "evento ausente no corpo", None, None
    ev = evs[0]
    nomes = [ev.get("home_team", ""), ev.get("away_team", "")]
    lado = next((n for n in nomes if _mesmo_nome(r.model_side, n)), None)
    outro = next((n for n in nomes if n != lado), None)
    if lado is None:
        return False, "model_side nao esta no evento", None, None
    p = sharp_fair_prob(ev, lado, outro)
    if p is None or abs(p - float(getattr(r, prob_col))) > 5e-5:
        return False, f"{prob_col} recalculado {p} != {getattr(r, prob_col)}", None, None
    return True, "", datetime.fromisoformat(c.captured_at), datetime.fromisoformat(
        str(ev.get("commence_time")).replace("Z", "+00:00"))


motivos = []
for r in P.itertuples():
    m = []
    if abs(float(r.close_prob) - float(r.sharp_prob) - float(r.clv)) > 1e-4:
        m.append("V1 clv != close-sharp")
    ok3, m3, t_ent, _ = prova(r.sharp_ref, r, "sharp_prob")
    ok4, m4, t_fec, ini = prova(r.close_ref, r, "close_prob")
    if not ok3:
        m.append("V3 entrada: " + m3)
    if not ok4:
        m.append("V4 fechamento: " + m4)
    if ok3 and ok4 and not (t_ent < t_fec < ini):
        m.append(f"V5 ordem de tempo ({t_ent} / {t_fec} / {ini})")
    motivos.append("; ".join(m))
P["motivo"] = motivos
par = P.apply(lambda r: (r.event_date, tuple(sorted((r.fighter_a, r.fighter_b)))), axis=1)
dup = par.duplicated(keep=False)
P.loc[dup, "motivo"] = P.loc[dup, "motivo"] + "; V2 duplicada"
pid = P.prediction_id[P.prediction_id != ""]
print("V2: pares duplicados:", int(dup.sum()), "| prediction_id repetidos:", int(pid.duplicated().sum()))

V = P[P.motivo == ""]
print("\nEXCLUSOES (por evento e motivo):")
ex = P[P.motivo != ""].copy()
ex["m"] = ex.motivo.str.replace(r"recalculado .*", "recalculado diverge", regex=True)
print(ex.groupby(["event_date", "m"]).size().to_string())


def resumo(nome, d):
    x = num(d.clv)
    pe = d.assign(x=x).groupby("event_date").x.agg(["count", "mean", "sum"])
    rng = np.random.default_rng(20261106)
    evs = pe.index.to_numpy()
    boot = []
    for _ in range(10000):
        amo = rng.choice(evs, size=len(evs), replace=True)
        s = sum(pe.loc[e, "sum"] for e in amo); n = sum(pe.loc[e, "count"] for e in amo)
        boot.append(s / n)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    top = pe["sum"].abs().max() / pe["sum"].abs().sum() if len(pe) else float("nan")
    print(f"\n[{nome}] n={len(d)} pernas | {len(pe)} eventos | media={x.mean()*100:+.2f} pp | "
          f"mediana={x.median()*100:+.2f} pp | positivas={int((x > 0).sum())} zero={int((x == 0).sum())} "
          f"neg={int((x < 0).sum())}")
    print(f"  IC95 bootstrap por EVENTO (descritivo): [{lo*100:+.2f}, {hi*100:+.2f}] pp | "
          f"maior evento = {top:.1%} de soma(|soma do CLV por evento|)")
    sem = d.assign(x=x).drop(columns=[]).groupby("event_date").x.sum()
    sem_top = (x.sum() - sem.loc[sem.abs().idxmax()]) / (len(d) - pe.loc[sem.abs().idxmax(), "count"])
    print(f"  media sem o evento de maior |soma| ({sem.abs().idxmax()}): {sem_top*100:+.2f} pp")
    print("  por evento (n, media pp):", "; ".join(
        f"{e[5:]}:{int(c)},{m*100:+.2f}" for e, (c, m, _) in pe.iterrows()))


resumo("SERIE COMPLETA (P)", P)
if len(V):
    resumo("VERIFICAVEL (V1..V6)", V)
else:
    print("\n[VERIFICAVEL] n=0")
