"""
C5 — Reexecucao do fechamento/liquidacao.
  Falha: o fluxo de fechamento real (fechamento.preencher_template com base
  de resultados fixture -> sync na publicacao -> placar/P&L) roda 3 vezes, e
  a captura manual roda de novo depois do fechamento.
  Esperado: resultados, fechamento e P&L identicos apos a 1a rodada.
  Propriedade: liquidar e idempotente.

C6 — Lados invertidos, revanche, adversario substituido.
  (a) o CSV do card volta com os lados invertidos;
  (b) adversario trocado (Ana x Bia -> Ana x Eva), com o nome do card mudando
      junto (o nome do card e o confronto principal);
  (c) revanche: o mesmo par, em data posterior, com o confronto antigo ja
      liquidado no historico.
  Esperado: nenhuma previsao/preco de um confronto usado em outro; uma linha
  por luta e data; o placar nao conta a mesma luta duas vezes.
"""
import sys

import pandas as pd

from chaos_common import *
from src.prediction_history import compute_series_summary, load_history

ok_total = True
TEMPLATE = APP / "data/odds_template.csv"
EVS = [evento_api("Ana Alfa", "Bia Beta", 1.85, 2.05, (2.15, 2.2)),
       evento_api("Caio Gama", "Davi Delta", 1.55, 2.6, (1.6, 2.8))]


def preparar(card=CARD, lutas=LUTAS):
    reset(); escrever_card(lutas=lutas, tp=[(2.15, 1.75), (1.6, 2.5)]); preditor_fixo()
    sem_frescor(); api_simulada(EVS); publicar_card(card=card)
    with TEMPLATE.open("a", encoding="utf-8") as f:
        for a, b in lutas:
            f.write(f",{card},{EVENTO},{a},{b},1.8,2.1,\n")


def base_de_resultados():
    return pd.DataFrame({"event_date": [EVENTO, EVENTO],
                         "fighter_1": ["Ana Alfa", "Davi Delta"],
                         "fighter_2": ["Bia Beta", "Caio Gama"],
                         "winner": ["Ana Alfa", "Davi Delta"], "method": ["KO", "DEC"]})


def estado():
    h = load_history()
    ev = h[h["event_date"] == EVENTO]
    return (ev[["fighter_a", "actual_winner", "close_prob", "estado_resultado"]]
            .astype(str).sort_values("fighter_a").values.tolist(),
            compute_series_summary(h))


# ---------- C5 ----------
preparar()
import scripts.auto_capture as ac
sys.argv = ["x"]; ac.main()                         # fechamento capturado
from src.fechamento import preencher_template
snaps = []
for rodada in range(3):
    preencher_template(EVENTO, fights_df=base_de_resultados())
    publicar_card(sharp=False)                      # sync dentro da publicacao
    snaps.append(estado())
import scripts.capture_closing as cc
sys.argv = ["x", "--event-date", EVENTO]
try:
    cc.main()
except SystemExit:
    pass
snaps.append(estado())
linhas_tpl = sum(1 for l in TEMPLATE.read_text().splitlines() if EVENTO in l)
ok_total &= resultado(
    "C5 fechamento + liquidacao reexecutados 3x, mais captura manual tardia",
    all(s == snaps[0] for s in snaps) and linhas_tpl == 2,
    "resultado, close e placar identicos em todas as rodadas; 2 linhas no template",
    f"rodadas identicas={all(s == snaps[0] for s in snaps)}; linhas no template={linhas_tpl}; "
    f"placar={snaps[-1][1]['legs_n']} pernas, pnl={snaps[-1][1]['legs_pnl']:.3f}")

# ---------- C6a lados invertidos ----------
preparar()
p_antes = do_evento().set_index("fighter_a").loc["Ana Alfa", "model_prob_a"]
escrever_card(lutas=[("Bia Beta", "Ana Alfa"), ("Caio Gama", "Davi Delta")],
              odds=((2.10, 1.80), (1.50, 2.70)), tp=[(1.75, 2.15), (1.6, 2.5)])
publicar_card(sharp=False)
ev = do_evento()
linha = ev[(ev["fighter_a"] == "Bia Beta") | (ev["fighter_b"] == "Bia Beta")]
r = linha.iloc[0]
prob_ana = r["model_prob_a"] if r["fighter_a"] == "Ana Alfa" else 1 - r["model_prob_a"]
odd_ana = r["odds_a_tp"] if r["fighter_a"] == "Ana Alfa" else r["odds_b_tp"]
ok_total &= resultado(
    "C6a card republicado com lados invertidos",
    len(linha) == 1 and abs(prob_ana - p_antes) < 1e-9 and abs(odd_ana - 2.15) < 1e-9
    and r["model_side"] == "Ana Alfa",
    "1 linha; P(Ana)=0.60, odd da casa da Ana=2.15, lado apontado=Ana",
    f"linhas={len(linha)}; P(Ana)={prob_ana:.4f}; odd Ana={odd_ana}; lado={r['model_side']}")

# ---------- C6b adversario trocado + nome do card muda ----------
preparar(card="UFC Fight Night: Alfa vs Beta")
preditor_fixo({("Ana Alfa", "Eva Epsilon"): 0.75, ("Caio Gama", "Davi Delta"): 0.70})
escrever_card(lutas=[("Ana Alfa", "Eva Epsilon"), ("Caio Gama", "Davi Delta")],
              odds=((1.40, 3.00), (1.50, 2.70)), tp=[(1.45, 2.9), (1.6, 2.5)])
api_simulada([evento_api("Ana Alfa", "Eva Epsilon", 1.45, 2.9), EVS[1]])
publicar_card(card="UFC Fight Night: Alfa vs Epsilon", sharp=True)
ev = do_evento()
caio = ev[ev["fighter_a"] == "Caio Gama"]
ana_bia = ev[(ev["fighter_b"] == "Bia Beta")]
ana_eva = ev[(ev["fighter_b"] == "Eva Epsilon")]
ok_total &= resultado(
    "C6b adversario trocado com renomeacao do card",
    len(caio) == 1 and len(ana_eva) == 1
    and ana_eva.iloc[0]["model_prob_a"] == 0.75,
    "Caio x Davi continua com UMA linha; Ana x Eva com previsao propria (0.75); "
    "Ana x Bia fica aberta, sem preco da Eva",
    f"linhas Caio x Davi={len(caio)} (event_name: {caio['event_name'].tolist()}); "
    f"Ana x Eva={ana_eva['model_prob_a'].tolist()}; Ana x Bia={ana_bia['model_prob_a'].tolist()}")
# liquida e mede o placar
with TEMPLATE.open("a", encoding="utf-8") as f:
    f.write(f",UFC Fight Night: Alfa vs Epsilon,{EVENTO},Caio Gama,Davi Delta,1.5,2.7,\n")
preencher_template(EVENTO, fights_df=pd.DataFrame({
    "event_date": [EVENTO], "fighter_1": ["Caio Gama"], "fighter_2": ["Davi Delta"],
    "winner": ["Caio Gama"], "method": ["KO"]}))
publicar_card(card="UFC Fight Night: Alfa vs Epsilon", sharp=False)
h = load_history()
fechadas = h[(h["event_date"] == EVENTO) & h["actual_winner"].notna()]
ok_total &= resultado(
    "C6b' placar depois de liquidar o evento renomeado",
    len(fechadas) == 1,
    "Caio x Davi conta UMA vez no placar",
    f"linhas liquidadas de Caio x Davi = {len(fechadas)} -> o placar soma a mesma luta {len(fechadas)}x")

# ---------- C6c revanche ----------
reset()
import pandas as pd2
hp = APP / "data/prediction_history.csv"
h = pd.read_csv(hp)
antiga = h[h["actual_winner"].notna()].iloc[-1]
a, b = antiga["fighter_a"], antiga["fighter_b"]
preditor_fixo({(a, b): 0.33, ("Caio Gama", "Davi Delta"): 0.70})
escrever_card(lutas=[(a, b), ("Caio Gama", "Davi Delta")], odds=((2.5, 1.55), (1.5, 2.7)))
api_simulada([evento_api(a, b, 2.4, 1.6), EVS[1]])
sem_frescor()
publicar_card(card=f"UFC Fight Night: {a.split()[-1]} vs {b.split()[-1]} 2", sharp=True)
h2 = pd.read_csv(hp)
velha = h2[(h2["event_date"] == antiga["event_date"]) & (h2["fighter_a"] == a) & (h2["fighter_b"] == b)].iloc[0]
nova = h2[(h2["event_date"] == EVENTO) & (h2["fighter_a"] == a)]
ok_total &= resultado(
    "C6c revanche do mesmo par em data nova",
    len(nova) == 1 and nova.iloc[0]["model_prob_a"] == 0.33
    and velha["model_prob_a"] == antiga["model_prob_a"]
    and str(velha["prediction_id"]) == str(antiga["prediction_id"]),
    "linha nova propria (0.33); linha antiga e liquidada intacta",
    f"nova={nova['model_prob_a'].tolist()} antiga={velha['model_prob_a']} (era {antiga['model_prob_a']})")

print("\nC5/C6:", "sem defeito" if ok_total else "DEFEITO(S) acima")
