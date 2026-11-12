"""Entradas de CLV de 14/11: o que a ENTRADA comprova. Somente leitura.

Criterios (os da regra de CLV, aplicados a entrada):
 E1 sharp_ref resolve em UMA cotacao e o corpo confere pelo hash;
 E2 o evento da cotacao existe no corpo e os dois nomes do evento casam
    com os dois lutadores da linha (exato ou alias versionado);
 E3 a Pinnacle tem preco dos DOIS lados naquele evento;
 E4 sharp_fair_prob(Pinnacle, model_side) recalculado == sharp_prob (5e-5),
    devig proporcional de duas vias (soma 1);
 E5 captura da entrada < commence_time daquele evento no corpo;
 E6 a linha nao tem bloqueio_cotacao e o lado da prova e o model_side.
"""
import gzip
import json
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.getcwd())
from src.line_shopping import (SHARP_BOOK, collect_prices, decimal_odds_to_implied_prob,  # noqa: E402
                               remove_vig_two_way, sharp_fair_prob)
from src.proveniencia import _mesmo_nome, conferir  # noqa: E402

ler = dict(dtype=str, keep_default_na=False)
h = pd.read_csv("data/prediction_history.csv", **ler)
h = h[h.event_date == "2026-11-14"]
cot = pd.read_csv("data/proveniencia/cotacoes.csv", **ler)
cham = pd.read_csv("data/proveniencia/chamadas.csv", **ler)
a, b = remove_vig_two_way(0.6, 0.5)
print(f"devig: remove_vig_two_way(0.6, 0.5) = ({a:.4f}, {b:.4f}) soma {a + b:.4f} (proporcional)")
ok_n = 0
for r in h.itertuples():
    f = []
    det = ""
    if r.bloqueio_cotacao:
        f.append(f"E6 bloqueio: {r.bloqueio_cotacao}")
    c = cot[cot.ref == r.sharp_ref] if r.sharp_ref else cot.iloc[0:0]
    if len(c) != 1:
        f.append(f"E1 sharp_ref em {len(c)} cotacoes")
    else:
        c = c.iloc[0]
        okc, mot = conferir(c.chamada_ref)
        if not okc:
            f.append(f"E1 corpo: {mot}")
        corpo = json.load(gzip.open(f"data/proveniencia/{c.chamada_ref}.json.gz", "rt", encoding="utf-8"))
        evs = [e for e in corpo if str(e.get("id")) == c.evento_id]
        if len(evs) != 1:
            f.append("E2 evento ausente")
        else:
            ev = evs[0]
            nomes = [ev["home_team"], ev["away_team"]]
            casa_a = [n for n in nomes if _mesmo_nome(r.fighter_a, n)]
            casa_b = [n for n in nomes if _mesmo_nome(r.fighter_b, n)]
            if len(casa_a) != 1 or len(casa_b) != 1 or casa_a == casa_b:
                f.append(f"E2 nomes {nomes} x ({r.fighter_a}, {r.fighter_b})")
            else:
                lado = casa_a[0] if r.model_side == r.fighter_a else casa_b[0]
                outro = casa_b[0] if lado == casa_a[0] else casa_a[0]
                if not _mesmo_nome(c.lado, lado):
                    f.append(f"E6 lado da prova {c.lado} != {lado}")
                pa, pb = collect_prices(ev, lado), collect_prices(ev, outro)
                if SHARP_BOOK not in pa or SHARP_BOOK not in pb:
                    f.append("E3 Pinnacle sem os dois lados")
                else:
                    p = sharp_fair_prob(ev, lado, outro)
                    det = (f"Pin {pa[SHARP_BOOK]}/{pb[SHARP_BOOK]} -> {p:.4f}; "
                           f"prova casa={c.casa} odd={c.odd}")
                    if abs(p - float(r.sharp_prob)) > 5e-5:
                        f.append(f"E4 recalculado {p:.4f} != {r.sharp_prob}")
                ini = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
                cap = datetime.fromisoformat(c.captured_at)
                if not cap < ini:
                    f.append("E5 captura apos o inicio")
                det += f"; cap {c.captured_at[11:19]}Z < inicio {ev['commence_time'][5:16]}"
    ok = not f
    ok_n += ok
    print(f"{'OK ' if ok else 'NAO'} {r.fighter_a} x {r.fighter_b} | lado {r.model_side} | "
          f"sharp {r.sharp_prob or '-'} | {r.casamento or '-'} | {r.fonte_card}")
    print("     " + (det if ok else "; ".join(f)))
print(f"\nentradas elegiveis: {ok_n} de {len(h)}")
