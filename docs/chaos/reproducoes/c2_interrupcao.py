"""
C2 — Interrupcao entre gravar corpo, indice e referencia.

Falhas injetadas (no caminho real de publicacao/captura):
 (a) processo morre DURANTE a escrita do historico (to_csv grava metade e cai);
 (b) processo morre DURANTE a escrita do corpo .json.gz da chamada; a mesma
     resposta volta numa rodada seguinte;
 (c) falha de disco ao gravar o indice de cotacoes (OSError engolido);
 (d) processo morre DEPOIS de anexar os snapshots e ANTES do historico.
Esperado: prova incompleta e DETECTADA (verificacao reprova com motivo) e o
proximo uso nao trata o estado parcial como valido.
Propriedade: toda ref gravada no historico ou resolve para prova integra ou
e reprovada pela verificacao.
"""
import gzip

import pandas as pd

from chaos_common import *

ok_total = True
EVS = [evento_api("Ana Alfa", "Bia Beta", 1.85, 2.05, (1.95, 2.2)),
       evento_api("Caio Gama", "Davi Delta", 1.55, 2.6, (1.6, 2.8))]
HIST = APP / "data/prediction_history.csv"


def preparar(publicar=True):
    reset(); escrever_card(); preditor_fixo(); sem_frescor(); api_simulada(EVS)
    if publicar:
        publicar_card()


# (a) crash no meio do to_csv do historico; proxima rodada normal
preparar()
linhas_antes = len(pd.read_csv(HIST))
orig_to_csv = pd.DataFrame.to_csv


def to_csv_que_cai(self, path=None, *a, **k):
    if path is not None and str(path).endswith("prediction_history.csv"):
        texto = orig_to_csv(self, None, *a, **k)
        Path(path).write_text(texto[: int(len(texto) * 0.6)], encoding="utf-8")
        raise KeyboardInterrupt("queda simulada no meio da escrita")
    return orig_to_csv(self, path, *a, **k)


pd.DataFrame.to_csv = to_csv_que_cai
try:
    publicar_card(sharp=False)
except KeyboardInterrupt:
    pass
pd.DataFrame.to_csv = orig_to_csv
erro = None
try:
    publicar_card(sharp=False)          # proxima publicacao, sem nenhum aviso
    linhas_depois = len(pd.read_csv(HIST))
except Exception as exc:  # noqa: BLE001
    erro, linhas_depois = exc, None
ok_total &= resultado(
    "C2a queda no meio da escrita do historico", erro is not None,
    "proxima execucao recusa o historico truncado com motivo",
    f"linhas antes={linhas_antes}; proxima publicacao erro={erro!r}; linhas depois={linhas_depois} "
    f"(perdidas em silencio: {None if linhas_depois is None else linhas_antes - linhas_depois})")

# (b) corpo .gz truncado e mesma resposta de novo
preparar(publicar=False)
import src.line_shopping as ls
import src.proveniencia as pv
corpo = json.dumps(EVS)
ref = hashlib_ref = __import__("hashlib").sha256(corpo.encode()).hexdigest()[:16]
destino = pv.PASTA / f"{ref}.json.gz"
with gzip.open(destino, "wt", encoding="utf-8") as f:   # queda no meio do corpo
    f.write(corpo[: len(corpo) // 2])
raw = destino.read_bytes()
destino.write_bytes(raw[: len(raw) // 2])
ls.fetch_live_odds("dummy")                               # mesma resposta volta
ok_ref, motivo = pv.conferir(ref)
ok_total &= resultado(
    "C2b corpo truncado + mesma resposta na rodada seguinte", (not ok_ref),
    "conferir() reprova a chamada com motivo (prova incompleta detectada)",
    f"conferir={ok_ref} motivo={motivo!r}; corpo reparado? {pv.conferir(ref)[0]}")
print("    nota: o corpo truncado nunca e reescrito (proveniencia.py: `if not destino.exists()`),"
      " entao toda resposta identica futura herda a prova quebrada.")

# (c) OSError ao gravar o indice de cotacoes durante o pre-registro
preparar(publicar=False)
orig_open = Path.open


def open_falho(self, mode="r", *a, **k):
    if self.name == "cotacoes.csv" and "a" in mode:
        raise OSError(28, "No space left on device (simulado)")
    return orig_open(self, mode, *a, **k)


Path.open = open_falho
publicar_card()
Path.open = orig_open
ev = do_evento()
refs = ev["sharp_ref"].dropna().tolist()
verif = {r: pv.preco_verificavel(r) for r in refs}
ok_total &= resultado(
    "C2c indice de cotacoes nao gravado (disco cheio)",
    all(v is False for v in verif.values()),
    "sharp_ref gravado no historico nao passa na verificacao",
    f"sharp_ref -> verificavel: {verif}")

# (d) queda depois dos snapshots e antes do historico
preparar(publicar=False)
snap = APP / "data/previsoes_snapshots.csv"
n_snap_antes = len(pd.read_csv(snap))


def to_csv_que_cai2(self, path=None, *a, **k):
    if path is not None and str(path).endswith("prediction_history.csv"):
        raise KeyboardInterrupt("queda antes de gravar o historico")
    return orig_to_csv(self, path, *a, **k)


pd.DataFrame.to_csv = to_csv_que_cai2
try:
    publicar_card(sharp=False)
except KeyboardInterrupt:
    pass
pd.DataFrame.to_csv = orig_to_csv
n_snap = len(pd.read_csv(snap))
evento_no_hist = (pd.read_csv(HIST)["event_date"] == EVENTO).sum()
ok_total &= resultado(
    "C2d queda entre snapshot e historico", evento_no_hist == 0,
    "historico sem ponteiro para as versoes orfas; nenhuma delas vira 'versao em vigor'",
    f"versoes orfas anexadas={n_snap - n_snap_antes}; linhas do evento no historico={evento_no_hist}")

print("\nC2:", "sem defeito" if ok_total else "DEFEITO(S) acima")
