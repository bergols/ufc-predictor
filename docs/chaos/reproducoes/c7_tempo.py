"""
C7 — Mudanca de dia/fuso, cotacao vencida, luta ja iniciada.

Falhas injetadas (relogio simulado no modulo scripts.auto_capture, fuso
America/Sao_Paulo como na maquina agendada):
 (a) card de sabado que atravessa a meia-noite local: as 00:30 de domingo
     (BRT) a prelim ja comecou e a luta principal comeca as 02:00;
 (b) no mesmo card, as 22:00 de sabado: uma luta ja iniciada e outra nao;
 (c) cotacao de entrada com 7 h de idade, e cotacao lida depois do inicio.
Esperado: cada luta e capturada ate o SEU inicio e bloqueada depois dele,
independentemente da data local; cotacao vencida/pos-inicio reprovada.
Propriedade: o bloqueio e por luta, nunca pelo calendario do card.
"""
import os
import sys
import time
from datetime import date as _date, datetime as _dt, timedelta, timezone

os.environ["TZ"] = "America/Sao_Paulo"
time.tzset()

from chaos_common import *

ok_total = True
E = "2026-11-28"                                  # sabado
PRELIM = _dt(2026, 11, 29, 1, 0, tzinfo=timezone.utc)     # 22:00 BRT sab
MAIN = _dt(2026, 11, 29, 5, 0, tzinfo=timezone.utc)       # 02:00 BRT dom


def relogio(agora_utc):
    import scripts.auto_capture as ac

    class D(_date):
        @classmethod
        def today(cls):
            return agora_utc.astimezone().date()

    class T(_dt):
        @classmethod
        def now(cls, tz=None):
            return agora_utc if tz else agora_utc.astimezone().replace(tzinfo=None)
    ac.date, ac.datetime = D, T
    return ac


def preparar():
    reset(); escrever_card(); preditor_fixo(); sem_frescor()
    evs = [evento_api("Ana Alfa", "Bia Beta", 1.85, 2.05, inicio=MAIN),
           evento_api("Caio Gama", "Davi Delta", 1.55, 2.6, inicio=PRELIM)]
    api_simulada(evs)
    publicar_card(evento=E)
    api_simulada([evento_api("Ana Alfa", "Bia Beta", 1.70, 2.25, inicio=MAIN),
                  evento_api("Caio Gama", "Davi Delta", 1.45, 2.95, inicio=PRELIM)])


def capturar(agora):
    ac = relogio(agora)
    sys.argv = ["x"]
    ac.main()
    ev = do_evento(E).set_index("fighter_a")
    return ev.loc["Ana Alfa", "close_prob"], ev.loc["Caio Gama", "close_prob"]


# (a) 00:30 BRT de domingo: principal ainda nao comecou
preparar()
agora = _dt(2026, 11, 29, 3, 30, tzinfo=timezone.utc)
main_c, prelim_c = capturar(agora)
ok_total &= resultado(
    "C7a card atravessa a meia-noite local (00:30 BRT, principal as 02:00)",
    str(main_c) != "nan" and str(prelim_c) == "nan",
    "principal capturada (faltam 1h30); prelim ja iniciada bloqueada",
    f"hoje local={agora.astimezone().date()} (evento {E}); close principal={main_c}; close prelim={prelim_c}")

# (b) 22:30 BRT de sabado: prelim ja comecou, principal nao
preparar()
main_c, prelim_c = capturar(_dt(2026, 11, 29, 1, 30, tzinfo=timezone.utc))
ok_total &= resultado(
    "C7b luta ja iniciada x luta futura no mesmo card",
    str(main_c) != "nan" and str(prelim_c) == "nan",
    "so a luta futura recebe fechamento",
    f"close principal={main_c}; close prelim (iniciada)={prelim_c}")

# (c) cotacao vencida e cotacao lida apos o inicio
from src.proveniencia import conferir_cotacao
ref = do_evento(E).set_index("fighter_a").loc["Ana Alfa", "sharp_ref"]
import src.proveniencia as pv
linha = pv._linha_de_cotacao(ref)
lida = _dt.fromisoformat(linha["captured_at"])
r7h = conferir_cotacao(ref, exigir_atual=True, agora=lida + timedelta(hours=7))
r5h = conferir_cotacao(ref, exigir_atual=True, agora=lida + timedelta(hours=5))
# isola a regra do inicio: a mesma cotacao lida 1 h antes do inicio,
# conferida 1 min depois do inicio (idade 1 h, dentro da validade)
import pandas as pd
cot = pv.INDICE_COTACOES
dfc = pd.read_csv(cot, dtype=str, keep_default_na=False)
dfc.loc[dfc["ref"] == ref, "captured_at"] = (MAIN - timedelta(hours=1)).isoformat()
dfc.to_csv(cot, index=False)
pos = conferir_cotacao(ref, exigir_atual=True, agora=MAIN + timedelta(minutes=1))
ok_total &= resultado(
    "C7c cotacao vencida (7 h) / valida (5 h) / luta iniciada",
    (not r7h[0]) and pos[0] is False and (r5h[0] or "inicio" in r5h[1]),
    "7 h reprova por validade; apos o inicio reprova; 5 h e antes do inicio aprova",
    f"7h={r7h}; 5h={r5h}; apos inicio={pos}")

print("\nC7:", "sem defeito" if ok_total else "DEFEITO(S) acima")
