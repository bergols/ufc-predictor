"""
C4 — Duas escritas simultaneas no historico (captura horaria x publicacao).

Falha injetada: os dois processos reais (scripts.auto_capture e a
publicacao do card) rodam ao mesmo tempo; latencia de disco de 3 s e
injetada entre o read e o write de UM deles para tornar a intercalacao
deterministica. (c) roda sem injecao nenhuma, N vezes, e conta perdas.
Esperado: nenhuma linha perdida, nenhum ponteiro revertido, nenhum
fechamento apagado; ou a segunda escrita recusa por trava.
Propriedade: o estado final contem as duas escritas.
"""
import os, subprocess, sys, time
import pandas as pd
from chaos_common import *
RUN = os.environ["CHAOS_PY"]          # interpretador isolado criado por rodar.sh
CH = str(Path(__file__).parent / "c4_proc.py")

def filho(*args, bg=False):
    cmd = [RUN, CH, *args]
    if bg:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return subprocess.run(cmd, capture_output=True, text=True)

def preparar():
    reset(); escrever_card(); preditor_fixo(); sem_frescor()
    api_simulada([evento_api("Ana Alfa","Bia Beta",1.85,2.05,(1.95,2.2)), evento_api("Caio Gama","Davi Delta",1.55,2.6,(1.6,2.8))])
    publicar_card()

def pids(out):
    for l in out.splitlines():
        if l.startswith("PIDS_PUBLICADOS"):
            return l.split()[1]
ok_total = True

# (a) captura lenta x publicacao: a publicacao termina no meio da captura
preparar()
a = filho("captura", "3", bg=True); time.sleep(1.2)
b = filho("publica"); a.wait()
publicados = pids(b.stdout)
final = ",".join(sorted(do_evento()["prediction_id"].astype(str)))
ok_total &= resultado("C4a captura (lenta) x publicacao", final == publicados,
    "historico aponta para as versoes que a pagina publicada mostra",
    f"pagina/snapshots publicados={publicados} | historico final={final} | "
    f"close gravado={do_evento()['close_prob'].notna().sum()}/2")

# (b) publicacao lenta x captura: a captura termina no meio da publicacao
preparar()
b = filho("publica", "3", bg=True); time.sleep(1.5)
a = filho("captura"); out_b = b.communicate()[0]
cap = "fechamento gravado em 2" in (a.stdout + a.stderr)
close_final = do_evento()["close_prob"].notna().sum()
ok_total &= resultado("C4b publicacao (lenta) x captura", (not cap) or close_final == 2,
    "fechamento capturado continua no historico",
    f"captura relatou gravar 2 lutas? {cap} | close no historico final={close_final}/2")

# (c) sem injecao: N rodadas simultaneas
N = 12; perdas = 0
for i in range(N):
    preparar()
    a = filho("captura", bg=True); b = filho("publica", bg=True)
    oa = a.communicate()[0]; ob = b.communicate()[0]
    ev = do_evento()
    ok_ptr = ",".join(sorted(ev["prediction_id"].astype(str))) == pids(ob)
    ok_close = ("fechamento gravado em 2" not in oa) or ev["close_prob"].notna().sum() == 2
    perdas += (not ok_ptr) or (not ok_close)
ok_total &= resultado(f"C4c {N} rodadas simultaneas sem latencia injetada", perdas == 0,
    "0 rodadas com escrita perdida", f"{perdas}/{N} rodadas com escrita perdida")
print("\nC4:", "sem defeito" if ok_total else "DEFEITO(S) acima")
