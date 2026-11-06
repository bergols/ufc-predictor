"""
scripts/experimento_vantagem.py

Executa o experimento de `PROTOCOLO_VANTAGEM.md` com UM comando:

    python -m scripts.experimento_vantagem

Lê registros de produção e escreve só em `data/experimentos/`. Confere o
sha256 dos registros reais antes e depois, e **aborta** se algum mudou: um
experimento que mexe no pré-registro contaminou justamente o que ele mede.

Nada aqui altera modelo, histórico, snapshots, decisões, proveniência ou o
site. Nenhuma API é consultada.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import vantagem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Registros de producao que este experimento NAO pode tocar.
PROTEGIDOS = (
    config.PREDICTION_HISTORY_CSV,
    config.DATA_DIR / "previsoes_snapshots.csv",
    config.DATA_DIR / "decisoes.csv",
    config.DATA_DIR / "proveniencia" / "chamadas.csv",
    config.DATA_DIR / "proveniencia" / "cotacoes.csv",
    config.ODDS_TEMPLATE_CSV,
    config.UPCOMING_CARD_CSV,
)


def _sha256(p: Path) -> str | None:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def _marcar() -> dict:
    return {str(p): _sha256(p) for p in PROTEGIDOS}


def _commit() -> str:
    try:
        r = subprocess.run(["git", "-C", str(config.PROJECT_ROOT),
                            "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _bibliotecas() -> dict:
    import platform
    import sklearn
    import scipy
    return {"python": platform.python_version(), "plataforma": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "scipy": scipy.__version__}


def _por_grupo(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    linhas = []
    for grupo, bloco in df.groupby("grupo_experiencia"):
        linha = {"grupo": grupo, "n": len(bloco)}
        for nome, col in cols.items():
            m = vantagem.metricas(bloco[col], bloco["y"])
            linha[f"log_loss_{nome}"] = m["log_loss"]
            linha[f"auc_{nome}"] = m["auc"]
        linhas.append(linha)
    return pd.DataFrame(linhas).sort_values("grupo")


def _por_evento(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    linhas = []
    for (data, nome_ev), bloco in df.groupby(["event_date", "event_name"]):
        linha = {"event_date": data, "event_name": nome_ev, "n": len(bloco)}
        for nome, col in cols.items():
            linha[f"log_loss_{nome}"] = round(
                vantagem.log_loss(bloco[col], bloco["y"]), 4)
        linha["dif_C_menos_A"] = round(
            linha["log_loss_combinacao"] - linha["log_loss_mercado"], 4)
        linhas.append(linha)
    return pd.DataFrame(linhas).sort_values("event_date")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--saida", type=str, default=None,
                    help="pasta dos artefatos (default: data/experimentos)")
    ap.add_argument("--history", type=str, default=None,
                    help="historico alternativo (para teste)")
    ap.add_argument("--bootstrap", type=int, default=vantagem.N_BOOTSTRAP,
                    help=f"reamostragens (default {vantagem.N_BOOTSTRAP})")
    args = ap.parse_args(argv)

    pasta = Path(args.saida or vantagem.PASTA)
    pasta.mkdir(parents=True, exist_ok=True)
    antes = _marcar()

    # ---- populacao -------------------------------------------------------
    pop, excl = vantagem.carregar_populacao(args.history)
    if pop.empty:
        logger.error("Nenhuma luta elegivel. Nada a medir.")
        return 1
    pop = vantagem.fatias_por_evento(pop)
    pop = vantagem.experiencia_previa(pop)
    excl.to_csv(pasta / f"exclusoes_{vantagem.VERSAO}.csv", index=False)
    logger.info("Populacao: %d lutas elegiveis em %d eventos; %d exclusoes.",
                len(pop), pop["event_date"].nunique(), len(excl))

    dev = pop[pop["fatia"] == "desenvolvimento"].reset_index(drop=True)
    aval = pop[pop["fatia"] == "avaliacao"].reset_index(drop=True)
    logger.info("Desenvolvimento: %d lutas / %d eventos | Avaliacao: %d / %d",
                len(dev), dev["event_date"].nunique(),
                len(aval), aval["event_date"].nunique())
    if dev.empty or aval.empty:
        logger.error("Uma das fatias ficou vazia -- sem avaliacao possivel.")
        return 1

    # ---- peso, SO no desenvolvimento ------------------------------------
    w, perdas = vantagem.peso_otimo(dev["p_mercado"], dev["p_modelo"], dev["y"])
    logger.info("Peso do modelo estimado no DESENVOLVIMENTO: w = %.2f", w)
    if w == 0.0:
        logger.warning("w = 0: no desenvolvimento o modelo NAO acrescentou "
                       "informacao ao mercado.")

    for bloco in (pop, dev, aval):
        bloco["p_combinacao"] = vantagem.combinar(bloco["p_mercado"],
                                                  bloco["p_modelo"], w)

    cols = {"mercado": "p_mercado", "modelo": "p_modelo",
            "combinacao": "p_combinacao"}

    # ---- metricas --------------------------------------------------------
    resumo = {"versao_protocolo": vantagem.VERSAO, "commit": _commit(),
              "semente": vantagem.SEMENTE, "eps": vantagem.EPS,
              "peso_modelo_w": w, "bibliotecas": _bibliotecas(),
              "temporalidade": vantagem.TEMPORALIDADE,
              "n_exclusoes": int(len(excl)),
              "fatias": {}, "confirmatorio": {}, "exploratorio": {}}

    for nome_fatia, bloco in (("desenvolvimento", dev), ("avaliacao", aval)):
        resumo["fatias"][nome_fatia] = {
            "n_lutas": int(len(bloco)),
            "n_eventos": int(bloco["event_date"].nunique()),
            "eventos": sorted(bloco["event_date"].unique().tolist()),
            "cobertura": {k: int(bloco[c].notna().sum()) for k, c in cols.items()},
            **{k: vantagem.metricas(bloco[c], bloco["y"]) for k, c in cols.items()},
        }

    # ---- CONTRASTE CONFIRMATORIO: C - A na avaliacao ---------------------
    conf = vantagem.bootstrap_por_evento(aval, "p_combinacao", "p_mercado",
                                         n=args.bootstrap)
    conf["detectavel_metade_ic"] = vantagem.detectavel(
        aval, "p_combinacao", "p_mercado")
    inconclusivo = conf["ic95_baixo"] <= 0 <= conf["ic95_alto"]
    conf["veredito"] = (
        "negativo" if (w == 0.0 or conf["diferenca"] >= 0)
        else "inconclusivo" if inconclusivo
        else "favoravel a combinacao")
    resumo["confirmatorio"] = {
        "contraste": "log loss(combinacao) - log loss(mercado), fatia de avaliacao",
        **conf,
    }
    logger.info("CONFIRMATORIO  dif = %+.4f  IC95 [%+.4f, %+.4f]  -> %s",
                conf["diferenca"], conf["ic95_baixo"], conf["ic95_alto"],
                conf["veredito"])

    # ---- critérios de promoção (§6.3) -----------------------------------
    eventos_aval = sorted(aval["event_date"].unique())
    meio = len(eventos_aval) // 2
    metades = {}
    for rotulo, evs in (("primeira", eventos_aval[:meio]),
                        ("segunda", eventos_aval[meio:])):
        b = aval[aval["event_date"].isin(evs)]
        metades[rotulo] = round(vantagem.log_loss(b["p_combinacao"], b["y"])
                                - vantagem.log_loss(b["p_mercado"], b["y"]), 4)
    sem_um = {}
    for e in eventos_aval:
        b = aval[aval["event_date"] != e]
        sem_um[e] = round(vantagem.log_loss(b["p_combinacao"], b["y"])
                          - vantagem.log_loss(b["p_mercado"], b["y"]), 4)
    resumo["promocao"] = {
        "1_diferenca_negativa": bool(conf["diferenca"] < 0),
        "2_ic_todo_abaixo_de_zero": bool(conf["ic95_alto"] < 0),
        "3_w_maior_que_zero": bool(w > 0),
        "4_sinal_nas_duas_metades": bool(all(v < 0 for v in metades.values())),
        "5_sinal_sem_o_evento_de_maior_efeito": bool(all(v < 0 for v in sem_um.values())),
        "metades": metades, "deixando_um_evento_de_fora": sem_um,
        "promove": False,
    }
    resumo["promocao"]["promove"] = all(
        resumo["promocao"][k] for k in (
            "1_diferenca_negativa", "2_ic_todo_abaixo_de_zero",
            "3_w_maior_que_zero", "4_sinal_nas_duas_metades",
            "5_sinal_sem_o_evento_de_maior_efeito"))

    # ---- EXPLORATORIO ----------------------------------------------------
    resumo["exploratorio"] = {
        "B_menos_A_avaliacao": vantagem.bootstrap_por_evento(
            aval, "p_modelo", "p_mercado", n=args.bootstrap),
        "C_menos_A_desenvolvimento": vantagem.bootstrap_por_evento(
            dev, "p_combinacao", "p_mercado", n=args.bootstrap),
        "perdas_por_peso_no_desenvolvimento": perdas,
        "cobertura_sharp_avaliacao": int(aval["sharp_prob"].notna().sum()),
        "cobertura_preco_executavel_avaliacao": int(
            (aval["odd_a_tp"].notna() & aval["odd_b_tp"].notna()).sum()),
    }

    # ---- artefatos -------------------------------------------------------
    pop_out = pop[["event_date", "event_name", "fight_uid", "fighter_a",
                   "fighter_b", "y", "fatia", "grupo_experiencia",
                   "lutas_previas_min", "odd_a", "odd_b", "p_mercado",
                   "p_modelo", "p_combinacao", "sharp_prob", "sharp_casa",
                   "sharp_ref", "odd_a_tp", "odd_b_tp", "ref_tp",
                   "prediction_id", "temporalidade"]]
    pop_out.to_csv(pasta / f"previsoes_{vantagem.VERSAO}.csv", index=False)

    for nome_fatia, bloco in (("desenvolvimento", dev), ("avaliacao", aval)):
        for cand, col in cols.items():
            vantagem.tabela_de_calibracao(bloco[col], bloco["y"]).to_csv(
                pasta / f"calibracao_{nome_fatia}_{cand}.csv", index=False)
        _por_grupo(bloco, cols).to_csv(
            pasta / f"grupos_{nome_fatia}.csv", index=False)
    _por_evento(pop, cols).to_csv(pasta / f"por_evento_{vantagem.VERSAO}.csv",
                                  index=False)

    manifesto = {
        "versao_protocolo": vantagem.VERSAO, "commit": _commit(),
        "semente": vantagem.SEMENTE, "grade_do_peso": vantagem.GRADE_W.tolist(),
        "eps": vantagem.EPS, "n_reamostragens": int(args.bootstrap),
        "bibliotecas": _bibliotecas(),
        "hashes_das_entradas": {str(p): _sha256(p) for p in (
            config.PREDICTION_HISTORY_CSV, config.RAW_FIGHTS_CSV)},
        "fatias": {f: sorted(pop[pop["fatia"] == f]["fight_uid"]
                             .dropna().astype(str).tolist())
                   for f in ("desenvolvimento", "avaliacao")},
        "eventos_por_fatia": {f: sorted(pop[pop["fatia"] == f]["event_date"]
                                        .unique().tolist())
                              for f in ("desenvolvimento", "avaliacao")},
        "n_exclusoes": int(len(excl)),
        "limitacao_temporal": vantagem.TEMPORALIDADE,
    }
    (pasta / f"manifesto_{vantagem.VERSAO}.json").write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    (pasta / f"resumo_{vantagem.VERSAO}.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- os registros reais nao podem ter mudado -------------------------
    depois = _marcar()
    mexidos = [k for k in antes if antes[k] != depois[k]]
    if mexidos:
        logger.error("ABORTANDO: o experimento ALTEROU registro de producao: %s",
                     ", ".join(mexidos))
        return 2
    logger.info("Registros de producao intactos (%d arquivos conferidos).",
                len(antes))
    logger.info("Artefatos em %s", pasta.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
