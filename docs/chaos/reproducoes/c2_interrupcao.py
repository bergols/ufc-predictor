"""
C2 — Interrupcao entre gravar corpo, indice e referencia.

Falhas injetadas (no caminho real de publicacao/captura):
 (a) processo morre DURANTE a gravacao do historico, em DOIS pontos: no
     to_csv (grava 60% e cai) e no os.replace do temporario (cai antes de
     trocar). A injecao precisa atingir a gravacao (contador de acertos > 0);
 (b) processo morre DURANTE a escrita do corpo .json.gz da chamada; a mesma
     resposta volta numa rodada seguinte;
 (c) falha de disco (OSError) ao gravar cotacoes.csv, no os.replace/temporario
     (caminho atual) e na abertura para anexar (caminho de 2f73748);
 (d) processo morre DEPOIS de anexar os snapshots e ANTES do historico.

Propriedades verificadas (Fase A em diante):
 (a) destino byte a byte igual ao anterior E a execucao seguinte integra sem
     linhas perdidas; OU recusa explicita com motivo se o destino ficou
     incompleto. Perda silenciosa e o defeito.
 (b) sem nova resposta a chamada continua reprovada; com a mesma resposta o
     corpo e regravado e a prova volta a conferir.
 (c) sharp_ref gravado no historico nao passa na verificacao e nada fica
     anexado pela metade em cotacoes.csv.
 (d) sem ponteiro no historico para versoes orfas.

Esperado original (rodada 1, 2f73748) -- registro historico, NAO e mais o teste:
 "prova incompleta e DETECTADA (verificacao reprova com motivo) e o proximo
 uso nao trata o estado parcial como valido."
  C2a: "proxima execucao recusa o historico truncado com motivo"
  C2b: "conferir() reprova a chamada com motivo (prova incompleta detectada)"
  C2c: "sharp_ref gravado no historico nao passa na verificacao"
 Por que mudou: a Fase A tornou a recusa desnecessaria no caso (a) (gravacao
 atomica: o destino nunca fica truncado) e reparavel no (b) (corpo ruim e
 regravado). Exigir recusa/reprovacao permanente descrevia o remendo da
 rodada 1, nao a propriedade (nenhuma perda silenciosa, prova recuperavel).
 O (c) manteve a exigencia, mas a injecao agora atinge o caminho atual e
 tambem exige que nada fique anexado pela metade.
"""
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


# (a) queda durante a gravacao do historico
import os
orig_to_csv = pd.DataFrame.to_csv
orig_replace = os.replace


def _roda_a(modo):
    """modo 'to_csv': to_csv grava 60% e cai; 'replace': cai antes do os.replace."""
    preparar()
    antes = HIST.read_bytes()
    linhas_antes = len(pd.read_csv(HIST))
    acertos = [0]

    def to_csv_que_cai(self, path=None, *a, **k):
        if modo == "to_csv" and path is not None and str(path).endswith("prediction_history.csv"):
            acertos[0] += 1
            texto = orig_to_csv(self, None, *a, **k)
            Path(path).write_text(texto[: int(len(texto) * 0.6)], encoding="utf-8")
            raise KeyboardInterrupt("queda simulada no meio da escrita")
        return orig_to_csv(self, path, *a, **k)

    def replace_que_cai(src, dst, *a, **k):
        if modo == "replace" and str(dst).endswith("prediction_history.csv"):
            acertos[0] += 1
            raise KeyboardInterrupt("queda simulada antes do os.replace")
        return orig_replace(src, dst, *a, **k)

    pd.DataFrame.to_csv = to_csv_que_cai
    os.replace = replace_que_cai
    try:
        publicar_card(sharp=False, evento=EVENTO2)
    except KeyboardInterrupt:
        pass
    finally:
        pd.DataFrame.to_csv = orig_to_csv
        os.replace = orig_replace
    igual = HIST.read_bytes() == antes
    erro = None
    try:
        publicar_card(sharp=False, evento=EVENTO2)     # proxima execucao, sem aviso
        depois = pd.read_csv(HIST)
        linhas_depois = len(depois)
    except Exception as exc:  # noqa: BLE001
        erro, linhas_depois = exc, None
    integra = linhas_depois is not None and linhas_depois >= linhas_antes
    recusa = erro is not None and str(erro).strip() != ""
    ok = acertos[0] > 0 and ((igual and integra) or (not igual and recusa))
    return ok, (f"injecao[{modo}] acertos={acertos[0]}; destino identico={igual}; "
                f"linhas antes={linhas_antes} depois={linhas_depois}; "
                f"proxima execucao erro={erro!r}")


# o card de EVENTO ja esta publicado em preparar(); a queda e testada numa
# publicacao que ESCREVE (evento novo), senao nao ha gravacao para interromper.
EVENTO2 = (HOJE + timedelta(days=2)).isoformat()
res_a = [_roda_a(m) for m in ("to_csv", "replace")]
ok_total &= resultado(
    "C2a queda no meio da gravacao do historico", all(o for o, _ in res_a),
    "destino identico ao anterior e proxima execucao integra sem linhas perdidas "
    "(ou recusa explicita com motivo); a injecao atinge a gravacao",
    " || ".join(t for _, t in res_a))

# (b) corpo .gz truncado e mesma resposta de novo
preparar(publicar=False)
import src.line_shopping as ls
import src.proveniencia as pv
corpo = json.dumps(EVS)
ref = __import__("hashlib").sha256(corpo.encode()).hexdigest()[:16]
destino = pv.PASTA / f"{ref}.json.gz"
ls.fetch_live_odds("dummy")                               # chamada real: corpo + indice
ok_antes, _ = pv.conferir(ref)
raw = destino.read_bytes()
destino.write_bytes(raw[: len(raw) // 2])                 # queda no meio do corpo
ok_trunc, motivo = pv.conferir(ref)                       # sem nova resposta
ls.fetch_live_odds("dummy")                               # mesma resposta volta
ok_depois, motivo2 = pv.conferir(ref)
ok_total &= resultado(
    "C2b corpo truncado + mesma resposta na rodada seguinte",
    ok_antes and (not ok_trunc) and ok_depois,
    "antes ok; truncado sem nova resposta continua reprovado; com a mesma resposta a prova volta a conferir",
    f"antes={ok_antes}; truncado={ok_trunc} motivo={motivo!r}; apos mesma resposta={ok_depois} ({motivo2!r})")

# (c) falha de disco ao gravar cotacoes.csv durante o pre-registro
preparar(publicar=False)
COT = pv.INDICE_COTACOES
orig_open = Path.open
acertos_c = [0]


def open_falho(self, mode="r", *a, **k):          # 2f73748: anexa com open("a")
    if self.name == "cotacoes.csv" and "a" in mode:
        acertos_c[0] += 1
        raise OSError(28, "No space left on device (simulado)")
    return orig_open(self, mode, *a, **k)


def replace_falho(src, dst, *a, **k):             # atual: temporario + os.replace
    if str(dst).endswith("cotacoes.csv"):
        acertos_c[0] += 1
        raise OSError(28, "No space left on device (simulado)")
    return orig_replace(src, dst, *a, **k)


Path.open = open_falho
os.replace = replace_falho
try:
    publicar_card()
finally:
    Path.open = orig_open
    os.replace = orig_replace
ev = do_evento()
refs = ev["sharp_ref"].dropna().tolist()
verif = {r: pv.preco_verificavel(r) for r in refs}
import csv as _csv
meio = False
if COT.exists():
    bruto = COT.read_text(encoding="utf-8")
    linhas_csv = list(_csv.reader(bruto.splitlines()))
    meio = bool(bruto) and (not bruto.endswith("\n") or
                            any(len(l) != len(linhas_csv[0]) for l in linhas_csv))
vazio = [r for r in verif]                              # refs gravadas
ok_total &= resultado(
    "C2c cotacoes.csv nao gravado (disco cheio)",
    acertos_c[0] > 0 and all(v is False for v in verif.values()) and not meio,
    "injecao atinge a gravacao; sharp_ref gravado nao passa na verificacao; nada anexado pela metade",
    f"acertos da injecao={acertos_c[0]}; sharp_ref -> verificavel: {verif}; "
    f"cotacoes.csv existe={COT.exists()} anexado pela metade={meio}")

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
