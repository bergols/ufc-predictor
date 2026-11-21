import sys, time, logging, os
from datetime import date as _date, datetime as _dt, timedelta, timezone

def setup_tz(name):
    os.environ["TZ"] = name; time.tzset()

def mk(name):
    setup_tz(name)

def boot():
    global C
    import chaos_common as C
    return C

class Cap(logging.Handler):
    def __init__(self): super().__init__(); self.recs = []
    def emit(self, r): self.recs.append((r.levelname, r.getMessage()))

def relogio(agora):
    import scripts.auto_capture as ac
    class D(_date):
        @classmethod
        def today(cls): return agora.astimezone().date()
    class T(_dt):
        @classmethod
        def now(cls, tz=None): return agora if tz else agora.astimezone().replace(tzinfo=None)
    ac.date, ac.datetime = D, T
    return ac

FIGHTS = [("Ana Alfa","Bia Beta"),("Caio Gama","Davi Delta"),("Eli Echo","Fabio Foxtrot")]
PROBS = {FIGHTS[0]:0.60, FIGHTS[1]:0.70, FIGHTS[2]:0.55}

def utc(y,m,d,h,mi=0): return _dt(y,m,d,h,mi,tzinfo=timezone.utc)

calls = {"n": 0}

def api_set(listar):
    """listar: callable() -> lista de eventos. Conta chamadas."""
    import requests, src.line_shopping as ls
    def g(url, params=None, timeout=None, **kw):
        assert "the-odds-api" in url
        calls["n"] += 1
        r = listar()
        if isinstance(r, BaseException): raise r
        return C.Resp(r)
    ls.requests.get = g; requests.get = g

def evs(starts, i, now=None, so_futuras=True):
    out = []
    for (a,b), st in zip(FIGHTS, starts):
        if so_futuras and now is not None and st <= now: continue
        out.append(C.evento_api(a,b,1.60+0.01*i,2.30-0.01*i,outra=(1.7+0.01*i,2.4),inicio=st))
    return out

def preparar(event_date, starts, order=(0,1,2)):
    C.reset()
    lutas = [FIGHTS[k] for k in order]
    C.escrever_card(lutas=lutas, odds=[(1.8,2.1)]*3)
    C.preditor_fixo(PROBS); C.sem_frescor()
    C.api_simulada(evs(starts, 0, so_futuras=False))
    C.publicar_card(evento=event_date)

def close_state(event_date):
    h = C.do_evento(event_date)
    return {(r.fighter_a): (r.close_prob if r.close_prob==r.close_prob else None) for r in h.itertuples()}

def rounds(event_date, starts, t0, t1, minuto=5, so_futuras=True, listar_fn=None):
    """roda ac.main() a cada hora; devolve lista de dicts."""
    cap = Cap(); logging.getLogger().addHandler(cap)
    res = []; i = 0
    t = t0.replace(minute=minuto)
    prev = close_state(event_date)
    call0 = calls["n"]
    while t <= t1:
        i += 1
        api_set((lambda tt=t, ii=i: listar_fn(tt, ii)) if listar_fn else (lambda tt=t, ii=i: evs(starts, ii, tt, so_futuras)))
        cap.recs.clear(); c0 = calls["n"]
        ac = relogio(t); sys.argv = ["x"]; ac.main()
        cur = close_state(event_date)
        wrote = [a for a in cur if cur[a] is not None and cur[a] != prev.get(a)]
        res.append(dict(t=t, wrote=wrote, calls=calls["n"]-c0, logs=list(cap.recs)))
        prev = cur
        t += timedelta(hours=1)
    logging.getLogger().removeHandler(cap)
    return res
