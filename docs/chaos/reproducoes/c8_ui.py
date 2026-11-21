"""
C8 — UI com modelo/base ausentes ou snapshot parcial.

Falhas injetadas:
 (a) modelo ausente (predict_fight levanta FileNotFoundError, como em
     src/predict.py:151) numa REPUBLICACAO normal de card ja pre-registrado;
 (b) somente_render depois que o CSV do card mudou de preco;
 (c) somente_render com ponteiro de snapshot que nao resolve.
Esperado: numeros registrados preservados; indisponibilidade comunicada;
nada recalculado em silencio.
Propriedade: o historico nunca perde a previsao em vigor por falha do
modelo, e a pagina desenhada a partir do registro mostra o que foi
registrado.
"""
import re

from chaos_common import *


def preparar():
    reset(); escrever_card(tp=[(1.90, 2.00), (1.55, 2.60)]); preditor_fixo(); sem_frescor()
    api_simulada([evento_api("Ana Alfa", "Bia Beta", 1.85, 2.05, (1.95, 2.2)),
                  evento_api("Caio Gama", "Davi Delta", 1.55, 2.6, (1.6, 2.8))])
    publicar_card()


ok_total = True

# (a) modelo ausente numa republicacao normal
preparar()
antes = do_evento().set_index("fighter_a")[["model_prob_a", "model_side", "prediction_id"]]
import src.card_report as cr


def sem_modelo(a, b):
    raise FileNotFoundError("Modelo nao encontrado em models/logreg_calibrated.joblib")


cr._default_predict_fns = lambda m: (sem_modelo, sem_modelo)
publicar_card(sharp=False)
depois = do_evento().set_index("fighter_a")[["model_prob_a", "model_side", "prediction_id"]]
perdeu = depois["model_prob_a"].isna().sum()
ok_total &= resultado(
    "C8a modelo ausente na republicacao normal", perdeu == 0,
    "pre-registro aberto preservado (ou publicacao recusada) e pagina diz 'indisponivel'",
    f"antes:\n{antes.to_string()}\n    depois:\n{depois.to_string()}")

# (b) somente_render apos o CSV mudar de preco
preparar()
reg = do_evento().set_index("fighter_a")
pid = reg.loc["Ana Alfa", "prediction_id"]
from src import snapshots
v = snapshots.versao(pid)
escrever_card(odds=((2.40, 1.60), (1.50, 2.70)), tp=[(2.60, 1.55), (1.55, 2.60)])
from src.card_report import generate_card_report
import src.card_report as _cr
saida = APP / "docs/render.html"
_capt = {}
_orig_render = _cr.render_html
def _espiao_b(analysis, *a, **k):
    _capt["an"] = analysis
    return _orig_render(analysis, *a, **k)
_cr.render_html = _espiao_b
generate_card_report(APP / "data/raw/upcoming_card_odds.csv", saida, card_name=CARD,
                     event_date=EVENTO, photos=False, sharp=False, somente_render=True)
_cr.render_html = _orig_render
html = saida.read_text(encoding="utf-8")
registrado_ev = float(v["model_prob_a"]) * float(v["odds_a_tp"])
desenhado_ev = 0.60 * 2.60
# Confere a luta da Ana no objeto desenhado. (A versao anterior deste
# cenario procurava "2,60" no HTML inteiro e colidia com a odd da casa do
# lado B de Caio x Davi, que tambem e 2,60 no registro.)
an = _capt["an"]
ana = [f for f in an["favorites"] + an["underdogs"] if "Ana Alfa" in (f["fighter_a"], f["fighter_b"])]
odd_ana = ev_ana = None
if ana:
    f = ana[0]
    odd_ana = f["odds_a_casa"] if f["fighter_a"] == "Ana Alfa" else f["odds_b_casa"]
    ev_ana = f["ev"] if f["model_side"] == "Ana Alfa" else None
ok_b = (odd_ana is not None and abs(odd_ana - float(v["odds_a_tp"])) < 1e-9
        and ev_ana is not None and abs(ev_ana - registrado_ev) < 1e-6
        and ("1,90" in html or "1.90" in html))
ok_total &= resultado(
    "C8b somente_render apos mudanca de preco no CSV", ok_b,
    f"luta da Ana com odd da casa registrada ({v['odds_a_tp']}) e EV registrado ({registrado_ev:.3f})",
    f"odd da casa da Ana desenhada={odd_ana} | EV da Ana={ev_ana} | "
    f"EV que sairia do CSV novo = 0.60 x 2.60 = {desenhado_ev:.3f}")

# (c) ponteiro que nao resolve
preparar()
import pandas as pd
hp = APP / "data/prediction_history.csv"
h = pd.read_csv(hp)
m = (h["event_date"] == EVENTO) & (h["fighter_a"] == "Ana Alfa")
h.loc[m, "prediction_id"] = "ffffffffffffffff"
h.to_csv(hp, index=False)
an = None
import src.card_report as cr2
orig = cr2.render_html
def espiao(analysis, *a, **k):
    global an
    an = analysis
    return orig(analysis, *a, **k)
cr2.render_html = espiao
generate_card_report(APP / "data/raw/upcoming_card_odds.csv", saida, card_name=CARD,
                     event_date=EVENTO, photos=False, sharp=False, somente_render=True)
sem = [f for f in an["no_prediction"] if f["fighter_a"] == "Ana Alfa"]
ok_total &= resultado(
    "C8c ponteiro de snapshot quebrado em somente_render", bool(sem and sem[0].get("reason")),
    "luta vai para 'sem previsao' com motivo; nenhuma outra versao escolhida",
    f"no_prediction={[(f['fighter_a'], f['reason'][:70]) for f in sem]}")

print("\nC8:", "sem defeito" if ok_total else "DEFEITO(S) acima")
