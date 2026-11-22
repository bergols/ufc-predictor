from d5_lib import *
import d5_lib as L
C = L.boot()
FA = FIGHTS
def ok(nome, cond, esp, obs):
    return C.resultado(nome, bool(cond), esp, obs)

def checar(nome, E, starts, res, so_futuras):
    bl=[m for r in res[:1] for l,m in r['logs'] if 'NÃO capturada' in m]
    print('    [info 1a rodada bloqueios]', [b[:230] for b in bl])
    # starts por indice de FIGHTS; nomes fighter_a
    nomes = [f[0] for f in FIGHTS]
    viol_pos = []; ultimo = {}; buracos = []
    for r in res:
        for a in r["wrote"]:
            st = starts[nomes.index(a)]
            if st <= r["t"]: viol_pos.append((a, r["t"].isoformat()))
            ultimo[a] = r["t"]
    # sem buraco: luta escrita em toda rodada com t<start dentro da janela
    for k, a in enumerate(nomes):
        for r in res:
            if r["t"] < starts[k] and a not in r["wrote"] and r["calls"] > 0:
                buracos.append((a, r["t"].isoformat()))
    folga = {a: str(starts[nomes.index(a)] - t) for a, t in ultimo.items()}
    cond_folga = all(timedelta(0) < starts[nomes.index(a)] - t < timedelta(hours=1) for a, t in ultimo.items()) and len(ultimo) == 3
    ok(f"{nome} sem gravacao pos-inicio", not viol_pos, "0 gravacoes com inicio<=agora", f"{viol_pos}")
    ok(f"{nome} ultima gravacao < 1h antes do inicio (todas 3 lutas)", cond_folga, "folga em (0,1h)", f"{folga}")
    ok(f"{nome} sem buraco (grava em toda rodada ate o inicio)", not buracos, "nenhuma rodada com luta futura sem gravar", f"{buracos[:5]} total={len(buracos)}")
    sem = [(r["t"].isoformat()) for r in res for l,m in r["logs"] if "SEM FECHAMENTO" in m and f"({E})" in m]
    ok(f"{nome} alarme: nenhum SEM FECHAMENTO (evento com captura, ate o fim das rodadas)", not sem, "nenhum", f"{sem[:3]}")
    sem_perd = [(r["t"].isoformat()) for r in res for l,m in r["logs"] if "perdido" in m.lower() and f"({E})" in m]
    ok(f"{nome} alarme: 'perdido' nao aparece (evento capturado)", not sem_perd, "nenhum", f"{sem_perd[:3]}")
    return res

# (a) EUA
E = "2026-11-28"
S_US = [utc(2026,11,29,5), utc(2026,11,28,22), utc(2026,11,28,23,30)]
for so_fut in (True, False):
    rotulo = f"D5a EUA (listagem {'so futuras' if so_fut else 'inclui iniciadas'})"
    preparar(E, S_US)
    res = rounds(E, S_US, utc(2026,11,28,20), utc(2026,11,30,12), so_futuras=so_fut)
    checar(rotulo, E, S_US, res, so_fut)

# (b) Australia: event_date domingo, lutas sab 23:00 UTC ... dom 06:00 UTC
E2 = "2026-11-29"
S_AU = [utc(2026,11,29,6), utc(2026,11,28,23), utc(2026,11,29,2)]
for so_fut in (True, False):
    rotulo = f"D5b Asia/Australia (listagem {'so futuras' if so_fut else 'inclui iniciadas'})"
    preparar(E2, S_AU)
    res = rounds(E2, S_AU, utc(2026,11,28,20), utc(2026,12,1,12), so_futuras=so_fut)
    checar(rotulo, E2, S_AU, res, so_fut)

# (c) contagem de chamadas, evento nunca preenchido
preparar(E, S_US)
L.calls["n"] = 0
res = rounds(E, S_US, utc(2026,11,25,0), utc(2026,12,4,23), so_futuras=True)
tot = sum(r["calls"] for r in res)
por_dia = {}
for r in res: por_dia[r["t"].date().isoformat()] = por_dia.get(r["t"].date().isoformat(), 0) + r["calls"]
print(f"D5c INFO chamadas de API 25/11..04/12 (rodadas :05 horarias, evento nunca preenchido): total={tot} por_dia={por_dia}")

# (d) evento sem captura: API nao lista nada
preparar(E, S_US)
res = rounds(E, S_US, utc(2026,11,26,20), utc(2026,12,1,12), listar_fn=lambda t,i: [])
antes = [r["t"] for r in res if any("SEM FECHAMENTO" in m and f"({E})" in m for l,m in r["logs"])]
prim = antes[0].isoformat() if antes else None
fim_janela = utc(2026,11,29,10)
cedo = [t.isoformat() for t in antes if t < fim_janela]
ok("D5d evento SEM captura: 'SEM FECHAMENTO' so depois da janela (dom 10:00 UTC), e aparece",
   antes and not cedo, "primeiro alarme >= 2026-11-29T10:00Z; nenhum antes (ultima luta 05:00Z)",
   f"primeiro={prim} antes_da_janela={cedo[:3]} n_rodadas_com_alarme={len(antes)}")
ult_antes = [r for r in res if utc(2026,11,29,5) > r["t"] > utc(2026,11,28,20)]
fut = [r["t"].isoformat() for r in res if r["t"] < utc(2026,11,29,5) and any("SEM FECHAMENTO" in m and f"({E})" in m for l,m in r["logs"])]
ok("D5d nenhum SEM FECHAMENTO enquanto ha luta futura (< dom 05:00Z)", not fut, "nenhum", f"{fut[:3]}")
msgs = [m for r in res for l,m in r["logs"] if "SEM FECHAMENTO" in m and f"({E})" in m][:1]
print("    amostra:", msgs)
