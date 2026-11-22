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
saida = APP / "docs/render.html"
generate_card_report(APP / "data/raw/upcoming_card_odds.csv", saida, card_name=CARD,
                     event_date=EVENTO, photos=False, sharp=False, somente_render=True)
html = saida.read_text(encoding="utf-8")
registrado_ev = float(v["model_prob_a"]) * float(v["odds_a_tp"])
desenhado_ev = 0.60 * 2.60
tem_registrado = "1,90" in html or "1.90" in html
tem_novo = "2,60" in html or "2.60" in html
ok_total &= resultado(
    "C8b somente_render apos mudanca de preco no CSV", tem_registrado and not tem_novo,
    f"odd da casa registrada no snapshot ({v['odds_a_tp']}) e EV registrado ({registrado_ev:.3f})",
    f"html mostra 1,90? {tem_registrado} | mostra 2,60 (CSV novo)? {tem_novo} | "
    f"EV que a pagina calcula = 0.60 x 2.60 = {desenhado_ev:.3f}")

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
