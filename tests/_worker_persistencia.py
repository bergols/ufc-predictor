"""
Processo filho dos testes de concorrencia (tests/test_persistencia.py).
NAO e um teste (nome comeca com _). Usa as funcoes REAIS de gravacao.

uso: python tests/_worker_persistencia.py <papel> <pasta> <rodadas> [tag]
  publica : record_card_predictions, `rodadas` vezes
  fecha   : record_closing_lines (allow_update), `rodadas` vezes
  snap    : snapshots.registrar, `rodadas` vezes
  cot     : proveniencia.registrar_cotacao, `rodadas` vezes
  segura  : segura a trava ate o arquivo <pasta>/solta existir
Variavel UFC_TESTE_LATENCIA (s): atraso injetado ANTES de cada gravacao
atomica, ou seja, DENTRO da trava, entre o ler e o gravar.
"""
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import config  # noqa: E402
from src import persistencia, prediction_history as ph, proveniencia, snapshots  # noqa: E402

EVENTO = (date.today() + timedelta(days=3)).isoformat()
CARD = "UFC Teste Concorrencia"
LUTAS = [("Ana Alfa", "Bia Beta"), ("Caio Gama", "Davi Delta")]

lat = float(os.environ.get("UFC_TESTE_LATENCIA", "0") or 0)
if lat:
    _orig = persistencia.gravar_atomico

    def _lento(caminho, conteudo):
        time.sleep(lat)
        return _orig(caminho, conteudo)
    persistencia.gravar_atomico = _lento
    ph.gravar_atomico = _lento
    snapshots.gravar_atomico = _lento


def analise(i):
    fav = [{"fighter_a": a, "fighter_b": b, "odds_a": 1.8, "odds_b": 2.1,
            "model_prob_a": round(0.60 + 0.001 * i, 4), "model_side": a,
            "category": "favorite", "method_probs": None} for a, b in LUTAS]
    return {"favorites": fav, "underdogs": [], "no_prediction": [],
            "model_name": "logreg"}


def sharp(i):
    return {(a, b): {"sharp_prob": round(0.50 + 0.001 * i, 4), "best_odd": 1.9}
            for a, b in LUTAS}


def main():
    papel, pasta, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
    tag = sys.argv[4] if len(sys.argv) > 4 else papel
    hist = pasta / "prediction_history.csv"
    if papel == "publica":
        for i in range(n):
            ph.record_card_predictions(analise(i), CARD, EVENTO, hist)
    elif papel == "fecha":
        for i in range(n):
            ph.record_closing_lines(EVENTO, sharp(i), hist, allow_update=True)
    elif papel == "snap":
        arq = pasta / "previsoes_snapshots.csv"
        for i in range(n):
            snapshots.registrar([{"event_date": EVENTO, "event_name": CARD,
                                  "fighter_a": f"{tag}{i}", "fighter_b": "X",
                                  "model_prob_a": 0.5}], arq)
    elif papel == "cot":
        proveniencia.INDICE_COTACOES = pasta / "prov" / "cotacoes.csv"
        for i in range(n):
            proveniencia.registrar_cotacao(
                chamada_ref="c", fonte="f", casa="h", mercado="m",
                evento_id="e", participante_a=f"{tag}{i}", participante_b="Y",
                lado="a", odd=1.9)
    elif papel == "segura":
        with persistencia.trava(hist):
            (pasta / "travado").write_text("ok")
            fim = time.monotonic() + 30
            while not (pasta / "solta").exists() and time.monotonic() < fim:
                time.sleep(0.02)


if __name__ == "__main__":
    main()
