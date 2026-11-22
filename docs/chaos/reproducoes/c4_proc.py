"""Processo filho do C4. uso: c4_proc.py captura|publica [lento_s]"""
import sys, time
import pandas as pd
from chaos_common import *
papel = sys.argv[1]; lento = float(sys.argv[2]) if len(sys.argv) > 2 else 0
if lento:
    orig = pd.DataFrame.to_csv
    def devagar(self, path=None, *a, **k):
        if path is not None and str(path).endswith("prediction_history.csv"):
            time.sleep(lento)        # latencia de disco injetada ENTRE ler e gravar
        return orig(self, path, *a, **k)
    pd.DataFrame.to_csv = devagar
preditor_fixo(); sem_frescor()
if papel == "captura":
    api_simulada([evento_api("Ana Alfa","Bia Beta",1.70,2.25,(1.75,2.3)), evento_api("Caio Gama","Davi Delta",1.45,2.95,(1.5,3.0))])
    import scripts.auto_capture as ac; sys.argv=["x"]; ac.main()
else:
    escrever_card(odds=((1.70,2.25),(1.45,2.95)))
    api_simulada([evento_api("Ana Alfa","Bia Beta",1.70,2.25,(1.75,2.3)), evento_api("Caio Gama","Davi Delta",1.45,2.95,(1.5,3.0))])
    publicar_card()
    print("PIDS_PUBLICADOS", ",".join(sorted(do_evento()["prediction_id"].astype(str))))
