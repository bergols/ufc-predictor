"""
C3 — CSV truncado ou JSON invalido.

Falhas injetadas:
 (a) prediction_history.csv com a ULTIMA linha cortada no meio (sem \n);
 (b) previsoes_snapshots.csv sem \n final (ultima linha parcial) antes de
     uma publicacao que anexa versoes;
 (c) card CSV com a ultima luta truncada (sem odds_b);
 (d) data/eventos_a_registrar.json invalido, lido pelo painel/auto_capture.
Esperado: falha com motivo; nenhum registro valido sobrescrito; nenhuma
luta omitida em silencio.
"""
import pandas as pd

from chaos_common import *

ok_total = True
HIST = APP / "data/prediction_history.csv"
SNAP = APP / "data/previsoes_snapshots.csv"
EVS = [evento_api("Ana Alfa", "Bia Beta", 1.85, 2.05, (1.95, 2.2)),
       evento_api("Caio Gama", "Davi Delta", 1.55, 2.6, (1.6, 2.8))]


def preparar():
    reset(); escrever_card(); preditor_fixo(); sem_frescor(); api_simulada(EVS)


# (a) ultima linha do historico cortada
preparar()
txt = HIST.read_text(encoding="utf-8").rstrip("\n")
ultima = txt.splitlines()[-1]
HIST.write_text(txt[: len(txt) - len(ultima) // 2], encoding="utf-8")
alvo = ultima.split(",")
erro = None
try:
    publicar_card(sharp=False)
except Exception as exc:  # noqa: BLE001
    erro = exc
h = pd.read_csv(HIST)
linha = h[(h["fighter_a"] == alvo[2]) & (h["fighter_b"] == alvo[3])]
ok_total &= resultado(
    "C3a historico com ultima linha truncada", erro is not None,
    "publicacao recusa o arquivo com motivo",
    f"erro={erro!r}; linha truncada regravada assim: "
    f"{linha[['fighter_a','fighter_b','model_prob_a','prediction_id','estado_resultado']].to_dict('records')}")

# (b) snapshots sem \n final
preparar()
s = SNAP.read_text(encoding="utf-8").rstrip("\n")
SNAP.write_text(s[:-20], encoding="utf-8")     # ultima versao perde 20 chars e o \n
erro_b = None
try:
    publicar_card(sharp=False)
except Exception as exc:  # noqa: BLE001
    erro_b = exc
from src import snapshots
pids = do_evento()["prediction_id"].tolist()
def legivel(p):
    try:
        return snapshots.versao(p) is not None
    except Exception as exc:  # noqa: BLE001
        return f"erro: {type(exc).__name__}"
resolvidos = {p: legivel(p) for p in pids}
linha_colada = [l[:60] + "..." for l in SNAP.read_text().splitlines() if any(p in l for p in pids)][:1]
ok_total &= resultado(
    "C3b arquivo de versoes com ultima linha parcial",
    erro_b is not None and not pids,
    "recusa ANTES de gravar o historico; nenhuma versao nova colada na linha parcial",
    f"erro na publicacao={type(erro_b).__name__ if erro_b else None} (depois de gravar o historico); "
    f"historico ja aponta para {pids}; versao legivel? {resolvidos}; linha colada: {linha_colada}")

# (c) card truncado
preparar()
card = APP / "data/raw/upcoming_card_odds.csv"
escrever_card()
t = card.read_text(encoding="utf-8").rstrip("\n")
linhas = t.splitlines()
linhas[-1] = "Caio Gama,Davi Delta,1.5"          # corte no meio da linha: sem odds_b
card.write_text("\n".join(linhas), encoding="utf-8")
print("    card truncado:\n      " + card.read_text().replace("\n", "\n      "))
erro = None
try:
    publicar_card(sharp=False)
except Exception as exc:  # noqa: BLE001
    erro = exc
n = len(do_evento()) if (HIST.exists()) else 0
ok_total &= resultado(
    "C3c card CSV com a ultima luta truncada", erro is not None or n == 2,
    "recusa com motivo, ou a luta aparece marcada; nunca some em silencio",
    f"erro={erro!r}; lutas do evento registradas={n} de 2")

# (d) JSON invalido no cache de cards a registrar
preparar()
import src.event_watch as ew
cache = getattr(ew, "ARQUIVO", None) or getattr(ew, "PENDENTES", None)
alvo_json = Path(cache) if cache else APP / "data/eventos_a_registrar.json"
alvo_json.write_text('{"verificado_em": "2026-11-2', encoding="utf-8")
erro = None
try:
    from src.painel import gerar  # noqa: F401
except Exception:
    pass
try:
    leitor = [getattr(ew, n) for n in dir(ew) if n.startswith(("carregar", "ler", "pendentes"))]
    out = [f() for f in leitor if callable(f)]
except Exception as exc:  # noqa: BLE001
    erro, out = exc, None
ok_total &= resultado(
    "C3d JSON invalido em eventos_a_registrar", erro is None,
    "leitura trata como 'nao verificado' sem estourar",
    f"leitores={[f.__name__ for f in leitor]} erro={erro!r} saida={out}")

print("\nC3:", "sem defeito" if ok_total else "DEFEITO(S) acima")
