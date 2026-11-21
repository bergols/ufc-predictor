"""
C1 — API indisponivel, timeout ou resposta parcial.

Falha injetada: requests.get da The Odds API (a) levanta Timeout, (b) devolve
200 com JSON truncado, (c) devolve evento em que a Pinnacle omite o preco de
um lado, ao lado de outra luta integra.
Esperado: nenhuma odd inventada; nada antigo regravado como atual; a falha
de uma luta nao impede a captura da outra; o job agendado nao estoura.
Propriedade: close_prob/close_best_odd/close_ref so mudam com preco provado
da rodada corrente.
"""
import hashlib
import sys

import requests

from chaos_common import *


def h(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def preparar():
    reset(); escrever_card(); preditor_fixo(); sem_frescor()
    api_simulada([evento_api("Ana Alfa", "Bia Beta", 1.85, 2.05, (1.95, 2.2)),
                  evento_api("Caio Gama", "Davi Delta", 1.55, 2.6, (1.6, 2.8))])
    publicar_card()


def rodar_auto_capture():
    import scripts.auto_capture as ac
    sys.argv = ["auto_capture"]
    try:
        return ac.main(), None
    except Exception as exc:  # noqa: BLE001
        return None, exc


ok_total = True
HIST = APP / "data/prediction_history.csv"

# (a) timeout
preparar()
antes = h(HIST)
api_simulada(requests.Timeout("timeout simulado"))
rc, exc = rodar_auto_capture()
ok_total &= resultado("C1a timeout no auto_capture", exc is None and h(HIST) == antes,
                      "sai com 0, historico intacto", f"rc={rc} exc={exc!r} historico_intacto={h(HIST)==antes}")

# (b) JSON truncado com HTTP 200
preparar()
antes = h(HIST)
api_simulada(Resp('[{"id": "ev-ana-bia", "commence_time": "2026-11-2'))
rc, exc = rodar_auto_capture()
ok_total &= resultado("C1b JSON truncado no auto_capture", exc is None and h(HIST) == antes,
                      "sai com 0, historico intacto", f"rc={rc} exc={exc!r} historico_intacto={h(HIST)==antes}")

# (c) resposta parcial: Pinnacle sem 'price' num lado da 1a luta; 2a luta integra
preparar()
api_simulada([evento_api("Ana Alfa", "Bia Beta", 1.85, 2.05, (1.95, 2.2), sem_preco=True),
              evento_api("Caio Gama", "Davi Delta", 1.50, 2.75, (1.58, 2.9))])
rc, exc = rodar_auto_capture()
ev = do_evento().set_index("fighter_a")
caio_close = ev.loc["Caio Gama", "close_prob"]
ana_close = ev.loc["Ana Alfa", "close_prob"]
ok = exc is None and str(caio_close) != "nan" and str(ana_close) == "nan"
ok_total &= resultado(
    "C1c Pinnacle sem preco num lado (resposta parcial)", ok,
    "luta incompleta fica sem fechamento; a outra luta e capturada; job nao estoura",
    f"exc={exc!r}; close Ana={ana_close} close Caio={caio_close}")

print("\nC1:", "sem defeito" if ok_total else "DEFEITO(S) acima")
