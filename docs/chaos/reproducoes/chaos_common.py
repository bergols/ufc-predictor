"""
Infra comum dos cenarios CHAOS do UFC Predictor.

Roda SEMPRE dentro da copia descartavel criada por rodar.sh, sem credenciais
e com a rede bloqueada por bloqueio_rede.py (instalado como sitecustomize). Cada cenario comeca com `reset()`,
que devolve data/ e docs/ da copia ao estado do commit inicial.

Fixtures:
  - API The Odds simulada (`resposta_api`), injetada em requests.get;
  - preditor fixo no lugar do modelo (nao ha modelo nem base no clone e
    treinar esta fora do escopo): substitui card_report._default_predict_fns.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# A copia descartavel e criada por rodar.sh e chega por CHAOS_COPIA.
# Recusa rodar no repositorio que contem estes scripts (o real) e em
# qualquer checkout que ainda tenha remoto configurado.
APP = Path(os.environ.get("CHAOS_COPIA", "")).resolve()
_REPO = Path(__file__).resolve().parents[3]
assert os.environ.get("CHAOS_COPIA") and APP != _REPO, \
    "CHAOS: rode via docs/chaos/reproducoes/rodar.sh (copia descartavel)"
assert not subprocess.run(["git", "remote"], cwd=APP, capture_output=True,
                          text=True).stdout.strip(), "CHAOS: copia com remoto"
os.chdir(APP)
sys.path.insert(0, str(APP))

# chave ficticia: o proxy nao existe aqui e a rede esta bloqueada
os.environ["ODDS_API_KEY"] = "dummy-chaos"

HOJE = date.today()
EVENTO = (HOJE + timedelta(days=1)).isoformat()      # card de amanha
CARD = "UFC Fight Night: Chaos vs Teste"
LUTAS = [("Ana Alfa", "Bia Beta"), ("Caio Gama", "Davi Delta")]


LIMPO = APP.parent / "limpo"      # data/ e docs/ intocados, salvos por rodar.sh


def reset():
    """Devolve data/ e docs/ DA COPIA ao estado em que ela foi criada."""
    import shutil
    for pasta in ("data", "docs"):
        shutil.rmtree(APP / pasta)
        shutil.copytree(LIMPO / pasta, APP / pasta)


def escrever_card(lutas=LUTAS, odds=((1.80, 2.10), (1.50, 2.70)), tp=None,
                  caminho=None) -> Path:
    caminho = Path(caminho or APP / "data/raw/upcoming_card_odds.csv")
    linhas = ["fighter_a,fighter_b,odds_a_decimal,odds_b_decimal,fonte,odds_a_tp,odds_b_tp"]
    for i, ((a, b), (oa, ob)) in enumerate(zip(lutas, odds)):
        ta, tb = (tp[i] if tp else ("", ""))
        linhas.append(f"{a},{b},{oa},{ob},ambas,{ta},{tb}")
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return caminho


def evento_api(a, b, pin_a, pin_b, outra=(None, None), inicio=None, id_=None,
               sem_preco=False):
    inicio = inicio or (datetime.now(timezone.utc) + timedelta(hours=20))
    bms = [{"key": "pinnacle", "title": "Pinnacle",
            "last_update": datetime.now(timezone.utc).isoformat(),
            "markets": [{"key": "h2h", "outcomes": [
                {"name": a, **({} if sem_preco else {"price": pin_a})},
                {"name": b, "price": pin_b}]}]}]
    if outra[0]:
        bms.append({"key": "betx", "title": "BetX", "markets": [{"key": "h2h", "outcomes": [
            {"name": a, "price": outra[0]}, {"name": b, "price": outra[1]}]}]})
    return {"id": id_ or f"ev-{a[:3]}-{b[:3]}".lower(), "sport_key": "mma_mixed_martial_arts",
            "commence_time": inicio.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "home_team": a, "away_team": b, "bookmakers": bms}


class Resp:
    def __init__(self, corpo, status=200):
        self.text = corpo if isinstance(corpo, str) else json.dumps(corpo)
        self.status_code = status
        self.headers = {"x-requests-remaining": "999"}

    def raise_for_status(self):
        import requests
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} simulado")

    def json(self):
        import requests
        try:
            return json.loads(self.text)
        except ValueError as exc:
            raise requests.exceptions.JSONDecodeError(str(exc), self.text, 0) from exc


def api_simulada(resposta):
    """Instala `resposta` (lista de eventos, Resp, ou excecao) no lugar da rede."""
    import requests
    import src.line_shopping as ls

    def falso_get(url, params=None, timeout=None, **kw):
        assert "the-odds-api" in url, url
        if isinstance(resposta, BaseException):
            raise resposta
        if isinstance(resposta, Resp):
            return resposta
        return Resp(resposta)
    ls.requests.get = falso_get
    requests.get = falso_get


def preditor_fixo(probs=None):
    """Troca o modelo por probabilidades fixas (A vence)."""
    probs = probs or {("Ana Alfa", "Bia Beta"): 0.60, ("Caio Gama", "Davi Delta"): 0.70}
    import src.card_report as cr

    def winner(a, b):
        if (a, b) in probs:
            p = probs[(a, b)]
        elif (b, a) in probs:
            p = 1 - probs[(b, a)]
        else:
            raise ValueError(f"sem fixture para {a} vs {b}")
        return {"fighter_a": a, "fighter_b": b, "prob_a_wins": p, "prob_b_wins": 1 - p,
                "model_used": "fixture", "fighter_a_low_experience": False,
                "fighter_b_low_experience": False, "fighter_a_debutant": False,
                "fighter_b_debutant": False}

    def method(a, b):
        return {"method_probs": {"ko_tko": 0.3, "submission": 0.2, "decision": 0.5}}
    cr._default_predict_fns = lambda model_name: (winner, method)


def sem_frescor():
    import src.data_collection as dc
    dc.check_data_freshness = lambda *a, **k: 0


def publicar_card(card=CARD, evento=EVENTO, csv=None, sharp=True, saida=None):
    """Caminho real de publicacao (sem git): generate_card_report."""
    from src.card_report import generate_card_report
    csv = csv or APP / "data/raw/upcoming_card_odds.csv"
    saida = saida or APP / "docs/index.html"
    return generate_card_report(csv, saida, card_name=card, event_date=evento,
                                photos=False, sharp=sharp)


def historico():
    import pandas as pd
    return pd.read_csv(APP / "data/prediction_history.csv")


def do_evento(evento=EVENTO):
    h = historico()
    return h[h["event_date"] == evento]


def resultado(nome, ok, esperado, observado):
    tag = "PASSOU" if ok else "DEFEITO"
    print(f"[{tag}] {nome}\n    esperado:  {esperado}\n    observado: {observado}")
    return ok
