"""
scripts/demo_mediana.py

Demonstra o caminho COMPLETO da referência de mercado, da captura à
classificação da decisão, com um card **sintético**.

    python -m scripts.demo_mediana

Nenhuma chamada de API, nenhum arquivo de produção tocado: tudo é escrito num
diretório temporário que some no fim. Serve para ver o fluxo inteiro sem
esperar um card real e sem gastar crédito.

O que ele percorre:

    1. uma resposta da The Odds API, sintética, persistida como o pipeline
       persiste (`registrar_chamada`);
    2. o casamento ESTRITO do confronto e o registro da mediana com a captura,
       as casas e as duas odds (`mediana.casar_e_registrar`);
    3. o preço executável da casa, por lado, com prova;
    4. a versão imutável da previsão, com `criado_em`;
    5. a decisão em modo sombra, reabrindo a captura, revalidando o hash e
       RECALCULANDO as duas medianas;
    6. a classificação: `avaliacao_prospectiva` com a prova, e
       `prospectiva_parcial` sem ela.
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import config
from src import candidatos, mediana, proveniencia, snapshots, sombra

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def _iso(h: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=h)).isoformat(
        timespec="seconds")


def _evento(inicio: str) -> dict:
    casas = ((3.40, 1.34), (3.50, 1.33), (3.60, 1.32))
    return {
        "id": "ev-sintetico", "sport_key": "mma_mixed_martial_arts",
        "commence_time": inicio, "home_team": "Alpha", "away_team": "Beta",
        "bookmakers": [
            {"key": f"casa{i}", "title": f"Casa {i}",
             "last_update": _iso(-1),
             "markets": [{"key": "h2h", "last_update": _iso(-1),
                          "outcomes": [{"name": "Alpha", "price": oa},
                                       {"name": "Beta", "price": ob}]}]}
            for i, (oa, ob) in enumerate(casas)],
    }


def _montar(base: Path, *, com_mediana: bool) -> list[dict]:
    inicio = _iso(5)
    data = inicio[:10]

    ev = _evento(inicio)
    chamada = proveniencia.registrar_chamada(
        json.dumps([ev], ensure_ascii=False), fonte="the_odds_api",
        endpoint="https://api.the-odds-api.com/v4/sports/x/odds/",
        regions="eu", markets="h2h", n_eventos=1)
    ref_med, dados, motivo = mediana.casar_e_registrar(
        "Alpha", "Beta", [ev], event_date=data, chamada_ref=chamada)
    if not ref_med:
        print(f"  casamento bloqueou: {motivo}")
        return []
    print(f"  captura        chamada_ref={chamada}")
    print(f"  mediana        ref={ref_med}  "
          f"{dados['odd_a']} / {dados['odd_b']}  "
          f"({dados['n_casas']} casas: {', '.join(dados['casas'])})")

    corpo = json.dumps([{"home": "Alpha", "away": "Beta", "odd_home": 4.50,
                         "odd_away": 1.30, "inicio": inicio, "status": 1}],
                       ensure_ascii=False, sort_keys=True)
    ch_tp = proveniencia.registrar_chamada(
        corpo, fonte="thunderpick", endpoint="https://thunderpick.io/x",
        markets="h2h")
    comum = dict(chamada_ref=ch_tp, fonte="thunderpick", casa="Thunderpick",
                 mercado="h2h", evento_id=inicio, participante_a="Alpha",
                 participante_b="Beta", status="aberto",
                 inicio_programado=inicio)
    ref_a = proveniencia.registrar_cotacao(lado="Alpha", odd=4.50, **comum)
    ref_b = proveniencia.registrar_cotacao(lado="Beta", odd=1.30, **comum)
    print(f"  preco casa     Alpha 4.50 ref={ref_a} | Beta 1.30 ref={ref_b}")

    pid = snapshots.registrar([{
        "event_date": data, "event_name": "UFC Sintetico",
        "fighter_a": "Alpha", "fighter_b": "Beta", "model_name": "logreg",
        "model_prob_a": 0.30, "model_side": "Beta",
        "manifesto_sha256": "man-sintetico", "modelos_sha256": "mod-sintetico",
    }], base / "snaps.csv", criado_em=_iso(-2))[0]
    print(f"  previsao       prediction_id={pid} criada 2h atras")

    pd.DataFrame([{
        "event_name": "UFC Sintetico", "event_date": data,
        "fighter_a": "Alpha", "fighter_b": "Beta",
        "odds_a_decimal": dados["odd_a"], "odds_b_decimal": dados["odd_b"],
        "model_name": "logreg", "model_prob_a": 0.30, "model_side": "Beta",
        "actual_winner": None, "estado_resultado": "aguardando",
        "prediction_id": pid,
    }]).to_csv(base / "h.csv", index=False)
    pd.DataFrame([{
        "fighter_a": "Alpha", "fighter_b": "Beta",
        "odds_a_decimal": dados["odd_a"], "odds_b_decimal": dados["odd_b"],
        "fonte": "ambas", "n_casas": dados["n_casas"],
        "ref_mediana": ref_med if com_mediana else "",
        "odds_a_tp": 4.50, "odds_b_tp": 1.30,
        "ref_tp": ref_a, "ref_tp_a": ref_a, "ref_tp_b": ref_b,
    }]).to_csv(base / "card.csv", index=False)

    candidatos.congelar(versao_protocolo="vantagem-v2", peso_w=0.0,
                        congelado_em=_iso(-3), arquivo=base / "cand.json")
    return sombra.montar(data, card_csv=base / "card.csv",
                         history_csv=base / "h.csv")


def main(argv=None) -> int:
    for rotulo, com in (("COM a prova da mediana", True),
                        ("SEM a prova da mediana", False)):
        with tempfile.TemporaryDirectory(prefix="demo_mediana_") as tmp:
            base = Path(tmp)
            pasta = base / "prov"
            # tudo redirecionado: nenhum arquivo de producao e tocado
            proveniencia.PASTA = pasta
            proveniencia.INDICE = pasta / "chamadas.csv"
            proveniencia.INDICE_COTACOES = pasta / "cotacoes.csv"
            mediana.INDICE = pasta / "medianas.csv"
            snapshots.ARQUIVO = base / "snaps.csv"
            candidatos.ARQUIVO = base / "cand.json"
            snapshots._CACHE.clear()

            print(f"\n=== {rotulo} " + "=" * (46 - len(rotulo)))
            linhas = _montar(base, com_mediana=com)
            for d in linhas:
                print(f"  mediana        {d['temporalidade_mediana']}"
                      f"  -- {d['evidencia_mediana'][:88]}")
                print(f"  CLASSE         {d['classe']}")
                if d.get("motivos_da_classe"):
                    print(f"                 {d['motivos_da_classe'][:88]}")
                if d.get("selecao"):
                    print(f"  SELECAO        {d['lado_selecionado']} @ "
                          f"{d['odd_lado_selecionado']} (executavel, nao a "
                          f"mediana) | retorno bruto esperado "
                          f"{d['retorno_bruto_esperado']}")
                else:
                    print(f"  recusa         {d['motivo'][:88]}")
    print("\nNenhum arquivo de producao foi tocado; os temporarios sumiram.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
