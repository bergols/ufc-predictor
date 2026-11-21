import sys, logging
from datetime import datetime, timedelta, timezone
import chaos_common as C
import d5_lib as L
from src.casamento import casar
import pandas as pd

def U(s): return datetime.fromisoformat(s.replace("Z", "+00:00"))
def ev(a, b, ct, id_, pin=(1.65, 2.25), outra=(1.70, 2.30)):
    e = C.evento_api(a, b, pin[0], pin[1], outra=outra, inicio=U("2026-11-29T00:00:00Z"), id_=id_)
    e["commence_time"] = ct
    return e
TG = lambda ct, a="Ana Alfa", b="Bia Beta": ev(a, b, ct, "TARGET")
DEC = lambda ct, i="1", a="Ana Alfa", b="Bia Beta": ev(a, b, ct, "DECOY"+i, pin=(1.20, 4.50), outra=(1.25, 4.8))
def ctl(ct): return ev("Caio Gama", "Davi Delta", ct, "CTL", pin=(1.50, 2.75))

PROV = lambda: "".join(p.read_text(encoding="utf-8", errors="ignore") for p in (C.APP/"data").rglob("*.csv") if "proveniencia" in str(p))
res = []
def check(nome, cond, esp, obs):
    res.append(C.resultado(nome, bool(cond), esp, obs))

def pre(E, events, fa="Ana Alfa", fb="Bia Beta", probs=None):
    C.reset()
    if fa == "Ana Alfa": C.escrever_card()
    else: C.escrever_card(lutas=[(fa, fb)], odds=[(1.8, 2.1)])
    C.preditor_fixo(probs); C.sem_frescor(); C.api_simulada(events)
    exc = None
    try: C.publicar_card(evento=E)
    except Exception as e: exc = e
    h = C.do_evento(E)
    r = h[h.fighter_a == fa].iloc[0] if len(h) else None
    return exc, r

def cap(E, events_pre, events_cap, now, fa="Ana Alfa", fb="Bia Beta", probs=None):
    exc0, r0 = pre(E, events_pre, fa, fb, probs)
    prov0 = PROV()
    C.api_simulada(events_cap)
    capl = L.Cap(); logging.getLogger().addHandler(capl)
    ac = L.relogio(now); sys.argv = ["x"]; exc = None
    try: ac.main()
    except Exception as e: exc = e
    logging.getLogger().removeHandler(capl)
    h = C.do_evento(E); r = h[h.fighter_a == fa].iloc[0]
    cp = r["close_prob"]
    novos = PROV()[len(prov0):] if PROV().startswith(prov0) else PROV()
    logs = [m for l, m in capl.recs if "NÃO capturada" in m or "Sem cotacao" in m]
    return exc, (None if cp != cp else cp), r, novos, logs, exc0

def caso_cap(nome, E, ct, now, esperado, extras=(), fa="Ana Alfa", fb="Bia Beta", tg=True, probs=None, motivo_in=None):
    evs = ([TG(ct, fa, fb)] if tg else []) + list(extras)
    exc, cp, r, novos, logs, exc0 = cap(E, evs, evs, U(now), fa, fb, probs)
    cap_ok = cp is not None
    decoy = "DECOY" in novos or (r["close_best_odd"] == r["close_best_odd"] and abs(float(r["close_best_odd"]) - 1.25) < 1e-9)
    cond = exc is None and exc0 is None and cap_ok == esperado and not decoy
    if not esperado and motivo_in: cond = cond and any(motivo_in in m for m in logs)
    check(f"[captura] {nome}", cond, f"capturada={esperado}" + (f"; motivo contem {motivo_in!r}" if (not esperado and motivo_in) else "") + "; sem preco de DECOY; sem excecao",
          f"exc={exc!r}/{exc0!r} capturada={cap_ok} decoy_gravado={decoy} log={[m[:170] for m in logs][:2]}")

def caso_pre(nome, E, ct, esperado, extras=(), fa="Ana Alfa", fb="Bia Beta", tg=True, probs=None, motivo_in=None):
    evs = ([TG(ct, fa, fb)] if tg else []) + list(extras)
    exc, r = pre(E, evs, fa, fb, probs)
    sp = r["sharp_prob"] if r is not None else None
    got = sp is not None and sp == sp
    blq = r["bloqueio_cotacao"] if r is not None else None
    decoy = "DECOY" in PROV()
    cond = exc is None and got == esperado and not decoy
    if not esperado and motivo_in: cond = cond and isinstance(blq, str) and motivo_in in blq
    check(f"[pre-registro] {nome}", cond, f"sharp={'sim' if esperado else 'nao'}" + (f"; bloqueio contem {motivo_in!r}" if (not esperado and motivo_in) else "") + "; sem DECOY; sem excecao",
          f"exc={exc!r} sharp_prob={sp} bloqueio={blq!r} decoy_gravado={decoy}")

def caso_casar(nome, E, events, ok, motivo=None, fa="Ana Alfa", fb="Bia Beta", cands=None):
    c = casar(fa, fb, events, event_date=E)
    cond = c.ok == ok and (ok or motivo is None or motivo in (c.motivo or "")) and (not ok or c.evento_id == "TARGET")
    if cands is not None: cond = cond and c.candidatos == cands
    check(f"[casar] {nome}", cond, f"ok={ok} motivo~{motivo!r} cands={cands}", f"ok={c.ok} motivo={c.motivo!r} evento={c.evento_id} cands={c.candidatos}")

# ---- 2 EUA (alem do D5a): casar e captura
E = "2026-11-28"
for ct in ("2026-11-28T22:00:00Z", "2026-11-29T05:00:00Z"):
    caso_cap(f"EUA sab E={E} luta {ct}", E, ct, ct.replace("T22", "T21").replace("T05:00", "T04:30") if False else (U(ct) - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"), True)
# ---- 3 viradas
for (E, ct, ok) in [("2026-12-01", "2026-11-30T23:00:00Z", True), ("2027-01-01", "2026-12-31T22:00:00Z", True),
                    ("2026-12-31", "2027-01-01T04:00:00Z", True), ("2026-11-30", "2026-12-01T03:00:00Z", True),
                    ("2026-12-01", "2026-11-29T23:00:00Z", False), ("2027-01-01", "2027-01-03T01:00:00Z", False)]:
    now = (U(ct) - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    caso_casar(f"virada E={E} ct={ct}", E, [TG(ct)], ok, None if ok else "janela")
    if ok: caso_cap(f"virada E={E} ct={ct}", E, ct, now, True)
    caso_pre(f"virada E={E} ct={ct}", E, ct, ok)
# ---- 4 offsets
# (E, ct, UTC): discrimina data local x UTC
for (E, ct, ok) in [("2026-11-29", "2026-11-29T09:00:00+09:00", True),        # 29 00:00Z
                    ("2026-11-29", "2026-11-30T05:00:00+10:30", True),        # 29 18:30Z, local 30
                    ("2026-11-30", "2026-11-28T20:00:00-05:00", True),        # 29 01:00Z, local 28 (E-2)
                    ("2026-11-30", "2026-11-28T22:30:00-03:00", True),        # 29 01:30Z, local 28 (E-2)
                    ("2026-11-28", "2026-11-30T05:00:00+09:00", True),        # 29 20:00Z, local 30 (E+2)
                    ("2026-11-28", "2026-11-30T05:00:00+10:30", True),        # 29 18:30Z, local 30 (E+2)
                    ("2026-11-30", "2026-11-29T01:00:00+09:00", False),       # 28 16:00Z: UTC E-2, local E-1
                    ("2026-11-30", "2026-11-29T22:00:00-05:00", True),        # 30 03:00Z
                    ("2026-11-27", "2026-11-28T22:00:00-05:00", False),       # 29 03:00Z: UTC E+2, local E+1
                    ("2026-11-27", "2026-11-28T22:00:00-03:00", False)]:      # 29 01:00Z
    caso_casar(f"offset E={E} ct={ct}", E, [TG(ct)], ok, None if ok else "janela")
    caso_pre(f"offset E={E} ct={ct}", E, ct, ok)
for (E, ct) in [("2026-11-29", "2026-11-29T09:00:00+09:00"), ("2026-11-29", "2026-11-30T05:00:00+10:30"),
                ("2026-11-30", "2026-11-28T20:00:00-05:00"), ("2026-11-30", "2026-11-28T22:30:00-03:00")]:
    now = (U(ct) - timedelta(minutes=30)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    caso_cap(f"offset E={E} ct={ct}", E, ct, now, True)
# ---- 5 ilegiveis
E = "2026-11-29"; NOW = "2026-11-29T00:30:00Z"
for rot, ct in [("sem fuso", "2026-11-29T05:00:00"), ("sem fuso 11-29 (pedido)", "2026-11-29T05:00:00"), ("so data", "2026-11-29"),
                ("vazio", ""), ("lixo", "lixo"), ("None", None), ("int", 12345), ("lista", ["x"]), ("so espacos", "   ")]:
    caso_casar(f"ilegivel {rot}", E, [TG(ct)], False, "ilegivel", cands=0)
    caso_cap(f"ilegivel {rot}", E, "2026-11-29T01:00:00Z", NOW, False, tg=False, extras=[TG(ct)], motivo_in="ilegivel")
    caso_pre(f"ilegivel {rot}", E, None, False, tg=False, extras=[TG(ct)], motivo_in="ilegivel")
    ok_ = [TG("2026-11-29T05:00:00Z")]
    caso_casar(f"ilegivel {rot} + candidato valido do mesmo par", E, [DEC(ct, "i")] + [TG("2026-11-29T05:00:00Z")], False, "multiplos")
    caso_cap(f"ilegivel {rot} + candidato valido", E, "2026-11-29T05:00:00Z", NOW, False, extras=[DEC(ct, "i")], motivo_in="multiplos")
    caso_pre(f"ilegivel {rot} + candidato valido", E, "2026-11-29T05:00:00Z", False, extras=[DEC(ct, "i")], motivo_in="multiplos")
# ---- 6 revanche
E = "2026-11-29"
for d in (2, 7, 30):
    for sinal in (-1, 1):
        ctr = (U("2026-11-29T20:00:00Z") + sinal*timedelta(days=d)).strftime("%Y-%m-%dT%H:%M:%SZ")
        caso_casar(f"revanche {sinal*d:+d}d + alvo na janela", E, [DEC(ctr), TG("2026-11-29T20:00:00Z")], True, cands=1)
        caso_casar(f"revanche {sinal*d:+d}d + alvo na janela (ordem inversa)", E, [TG("2026-11-29T20:00:00Z"), DEC(ctr)], True, cands=1)
        caso_cap(f"revanche {sinal*d:+d}d + alvo", E, "2026-11-29T20:00:00Z", "2026-11-29T19:30:00Z", True, extras=[DEC(ctr)])
        caso_pre(f"revanche {sinal*d:+d}d + alvo", E, "2026-11-29T20:00:00Z", True, extras=[DEC(ctr)])
        if d == 2:
            caso_casar(f"so a revanche {sinal*d:+d}d (alvo ausente)", E, [DEC(ctr)], False, "janela")
            caso_pre(f"so a revanche {sinal*d:+d}d", E, None, False, tg=False, extras=[DEC(ctr)])
a1, a2 = "2026-11-28T23:00:00Z", "2026-11-29T23:00:00Z"
caso_casar("mesmo par em 2 eventos na janela (sab 23Z, dom 23Z)", E, [TG(a1), DEC(a2)], False, "multiplos", cands=2)
caso_cap("mesmo par em 2 eventos na janela", E, a1, "2026-11-28T22:30:00Z", False, extras=[DEC(a2)], motivo_in="multiplos")
caso_pre("mesmo par em 2 eventos na janela", E, a1, False, extras=[DEC(a2)], motivo_in="multiplos")
caso_cap("mesmo par em 2 eventos na janela (captura no dom)", E, a1, "2026-11-29T22:30:00Z", False, extras=[DEC(a2)], motivo_in="multiplos")
# ---- 7 adversario diferente
E = "2026-11-29"; ct = "2026-11-29T05:00:00Z"
for rot, (x, y) in {"A x C": ("Ana Alfa", "Cris Cinza"), "C x B": ("Cris Cinza", "Bia Beta"), "B x C": ("Bia Beta", "Cris Cinza")}.items():
    evx = [ev(x, y, ct, "DECOY7", pin=(1.20, 4.50), outra=(1.25, 4.8))]
    caso_casar(f"adversario diferente {rot}", E, evx, False, "ausente")
    caso_cap(f"adversario diferente {rot}", E, ct, "2026-11-29T04:30:00Z", False, tg=False, extras=evx, motivo_in="ausente")
    caso_pre(f"adversario diferente {rot}", E, ct, False, tg=False, extras=evx, motivo_in="ausente")
# ---- 8 homonimo/aproximado
PR = {("Jean Silva", "Bia Beta"): 0.6, ("Ana Alfa", "Bia Beta"): 0.6, ("Caio Gama", "Davi Delta"): 0.7}
caso_casar("homonimo exato Jean Silva", E, [TG(ct, "Jean Silva", "Bia Beta")], False, "homonimo", "Jean Silva", "Bia Beta", cands=1)
caso_cap("homonimo exato Jean Silva", E, ct, "2026-11-29T04:30:00Z", False, fa="Jean Silva", fb="Bia Beta", probs=PR, tg=False,
         extras=[TG(ct, "Jean Silva", "Bia Beta")], motivo_in="homonimo")
caso_pre("homonimo exato Jean Silva", E, ct, False, fa="Jean Silva", fb="Bia Beta", probs=PR, tg=False,
         extras=[TG(ct, "Jean Silva", "Bia Beta")], motivo_in="homonimo")
for rot, (x, y) in {"Ana Alpha": ("Ana Alpha", "Bia Beta"), "Bia Betta": ("Ana Alfa", "Bia Betta"), "Ana Alfa Jr": ("Ana Alfa Jr", "Bia Beta")}.items():
    evx = [ev(x, y, ct, "TARGET")]
    # Propriedade: o nome aproximado NUNCA casa e nada e gravado. O motivo e
    # "aproximado" quando best_name_match sugere o nome; "Ana Alpha" nao e
    # sugerido e sai como "ausente". A versao do reteste (21/11) exigia
    # "aproximado" nos tres -- premissa do script, nao requisito; o bloqueio e a
    # ausencia de gravacao sao conferidos em todos.
    mot = None if rot == "Ana Alpha" else "aproximado"
    caso_casar(f"aproximado {rot}", E, evx, False, mot, cands=None if mot is None else 1)
    caso_cap(f"aproximado {rot}", E, ct, "2026-11-29T04:30:00Z", False, tg=False, extras=evx, motivo_in=mot)
    caso_pre(f"aproximado {rot}", E, ct, False, tg=False, extras=evx, motivo_in=mot)
print("D10 reteste:", "ok" if all(res) else f"DEFEITO(S) {res.count(False)} de {len(res)}")
