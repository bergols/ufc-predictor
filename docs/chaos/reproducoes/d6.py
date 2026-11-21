import sys, math
from datetime import datetime, timedelta, timezone
import chaos_common as C
from chaos_common import *
import scripts.auto_capture as ac
VALS = [("None", None), ("abc", "abc"), ("1.0", 1.0), ("NaN", float("nan")), ("negativo", -1.5)]
inicio = datetime.now(timezone.utc) + timedelta(hours=20)

def ev_ana(pin_a, extra_betx=None, tercera=None, pin_b=2.2222):
    e = evento_api("Ana Alfa", "Bia Beta", pin_a, pin_b, inicio=inicio)
    if extra_betx is not None:
        e["bookmakers"].append({"key":"betx","title":"BetX","markets":[{"key":"h2h","outcomes":[
            {"name":"Ana Alfa","price":extra_betx},{"name":"Bia Beta","price":2.40}]}]})
    if tercera is not None:
        e["bookmakers"].append({"key":"casaz","title":"CasaZ","markets":[{"key":"h2h","outcomes":[
            {"name":"Ana Alfa","price":tercera},{"name":"Bia Beta","price":2.30}]}]})
    return e
def ev_caio(): return evento_api("Caio Gama","Davi Delta",1.50,2.75,(1.58,2.9),inicio=inicio)

def prep():
    reset(); escrever_card(); preditor_fixo(); sem_frescor()
    api_simulada([evento_api("Ana Alfa","Bia Beta",1.85,2.05,(1.95,2.2),inicio=inicio),
                  evento_api("Caio Gama","Davi Delta",1.55,2.6,(1.6,2.8),inicio=inicio)])
    publicar_card()

def cot():
    p = APP/"data/cotacoes.csv"
    return p.read_text(encoding="utf-8") if p.exists() else ""

def rodar():
    sys.argv=["x"]
    try: return ac.main(), None
    except Exception as e: return None, e

ok_total = True
# (a)
for nome, v in VALS:
    prep(); antes = cot().splitlines()
    api_simulada([ev_ana(v), ev_caio()])
    rc, exc = rodar()
    novas = [l for l in cot().splitlines() if l not in antes]
    ev = do_evento().set_index("fighter_a")
    ana, caio = ev.loc["Ana Alfa","close_prob"], ev.loc["Caio Gama","close_prob"]
    anaodd = [l for l in novas if "Ana Alfa" in l or "2.2222" in l]
    from src.line_shopping import fetch_sharp_probs
    api_simulada([ev_ana(v), ev_caio()])
    lutas = [{"fighter_a":"Ana Alfa","fighter_b":"Bia Beta","model_side":"Ana Alfa"},{"fighter_a":"Caio Gama","fighter_b":"Davi Delta","model_side":"Caio Gama"}]
    import src.line_shopping as ls
    sp = fetch_sharp_probs(lutas, events=ls.fetch_live_odds("k"), event_date=EVENTO)
    mot = (sp.get(("Ana Alfa","Bia Beta")) or {}).get("bloqueio")
    cond = exc is None and str(ana)=="nan" and str(caio)!="nan" and not anaodd and mot and "preco" in mot
    ok_total &= resultado(f"D6a Pinnacle preco {nome} num lado", cond,
      "Ana sem fechamento com motivo; Caio capturada; nenhuma linha em cotacoes.csv da Ana/odd 2.2222",
      f"exc={exc!r} close Ana={ana} Caio={caio} motivo={mot!r} novas_linhas_Ana={len(anaodd)}")

# (b) invalido so em casa nao-sharp
for nome, v in VALS + [("0.5", 0.5)]:
    prep()
    api_simulada([ev_ana(1.80, extra_betx=v, tercera=2.05), ev_caio()])
    rc, exc = rodar()
    ev = do_evento().set_index("fighter_a")
    cbo = ev.loc["Ana Alfa","close_best_odd"]; cp = ev.loc["Ana Alfa","close_prob"]
    novas = [l for l in cot().splitlines() if "Ana Alfa" in l and "betx" in l.lower()]
    cond = exc is None and str(cp)!="nan" and abs(float(cbo)-2.05)<1e-9
    ok_total &= resultado(f"D6b casa nao-sharp (BetX) com preco {nome}", cond,
       "capturada; close_best_odd=2.05 (CasaZ, a melhor valida)", f"exc={exc!r} close_prob={cp} close_best_odd={cbo} linhas_cotacao_betx_Ana={len(novas)}")

# (c) pre-registro
for nome, v in VALS:
    reset(); escrever_card(); preditor_fixo(); sem_frescor()
    api_simulada([ev_ana(v, tercera=2.0), ev_caio()])
    try:
        publicar_card(); exc = None
    except Exception as e:
        exc = e
    h = do_evento() if exc is None else None
    if h is not None and len(h):
        ev = h.set_index("fighter_a")
        blq = ev.loc["Ana Alfa","bloqueio_cotacao"]; sp_ana = ev.loc["Ana Alfa","sharp_prob"]; sp_caio = ev.loc["Caio Gama","sharp_prob"]
    else: blq=sp_ana=sp_caio="?"
    cond = exc is None and isinstance(blq,str) and "preco" in blq and str(sp_ana)=="nan" and str(sp_caio)!="nan"
    ok_total &= resultado(f"D6c pre-registro sharp=True, Pinnacle preco {nome}", cond,
       "nao estoura; bloqueio_cotacao com motivo; Caio com sharp", f"exc={exc!r} bloqueio={blq!r} sharp Ana={sp_ana} Caio={sp_caio}")
print("D6:", "ok" if ok_total else "DEFEITO(S)")
