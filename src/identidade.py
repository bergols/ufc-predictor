"""
src/identidade.py

Identidade estavel e estado do resultado para as linhas do historico.

Por que existe
--------------
Ate nov/2026 o `prediction_history.csv` chaveava por NOME + DATA, e todos os
desfechos sem vencedor compartilhavam `actual_winner` vazio. Duas
consequencias, as duas confirmadas na auditoria de integridade:

  - sem id, duas lutas do mesmo par (revanche) so se distinguem pela data,
    e qualquer juncao que esqueca a data pega a errada -- foi exatamente o
    que acontecia em `compare_to_market`;
  - sem estado, luta cancelada, empate, no contest e coleta ainda incompleta
    sao a mesma coisa depois do fato. O painel conta todas como "sem
    resultado" e ninguem consegue auditar qual foi qual.

Regra que este modulo segue
---------------------------
**Deriva do que e verificavel, marca o resto como desconhecido.** Nada e
arbitrado. Luta que nao esta na base de resultados nao vira "cancelada" por
conveniencia: ela so vira `nao_aconteceu` quando o EVENTO daquela data existe
na base e aquele par nao esta nele -- aí a ausencia e evidencia. Enquanto o
evento inteiro nao entrou, o estado e `aguardando`.

Nenhuma previsao e tocada: este modulo so preenche as duas colunas novas.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config
from src.utils import best_name_match

logger = logging.getLogger(__name__)

VENCEDOR = "vencedor"
EMPATE_NC = "empate_ou_nc"
NAO_ACONTECEU = "nao_aconteceu"
AGUARDANDO = "aguardando"
DESCONHECIDO = "desconhecido"

ESTADOS = (VENCEDOR, EMPATE_NC, NAO_ACONTECEU, AGUARDANDO, DESCONHECIDO)


def _lado(nome: str, candidatos: list[str]) -> str | None:
    if nome in candidatos:
        return nome
    return best_name_match(nome, candidatos)


def resolver(fighter_a: str, fighter_b: str, event_date: str,
             tem_vencedor: bool, base: pd.DataFrame) -> tuple[str | None, str]:
    """
    (fight_uid, estado) desta linha.

    `base` e a tabela de resultados com fight_id, event_date, fighter_1,
    fighter_2, winner.
    """
    do_dia = base[base["_data"] == str(event_date)[:10]]
    if do_dia.empty:
        # o evento inteiro ainda nao entrou na base: a ausencia nao prova
        # nada sobre esta luta
        return None, AGUARDANDO if not tem_vencedor else DESCONHECIDO

    achadas = []
    for r in do_dia.itertuples():
        lados = [str(r.fighter_1), str(r.fighter_2)]
        ma, mb = _lado(fighter_a, lados), _lado(fighter_b, lados)
        if ma and mb and ma != mb:
            achadas.append(r)

    if len(achadas) > 1:
        return None, DESCONHECIDO          # ambiguo: nao arbitra
    if not achadas:
        # o evento existe na base e este par nao esta nele -- a ausencia
        # passa a ser evidencia (adversario trocado, luta cancelada)
        return None, NAO_ACONTECEU
    luta = achadas[0]
    uid = str(luta.fight_id) if pd.notna(luta.fight_id) else None
    if pd.isna(luta.winner):
        return uid, EMPATE_NC
    return uid, VENCEDOR


def preencher(history_df: pd.DataFrame,
              fights_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Copia do historico com `fight_uid` e `estado_resultado` preenchidos.

    Nao sobrescreve valor ja presente: uma vez resolvida, a identidade da
    linha nao muda. Nenhuma outra coluna e tocada.
    """
    if fights_df is None:
        fights_df = pd.read_csv(config.RAW_FIGHTS_CSV,
                                usecols=["fight_url", "event_date", "fighter_1",
                                         "fighter_2", "winner"])
    base = fights_df.copy()
    # base vazia e estado real -- clone novo, ou coleta que ainda nao rodou.
    # Sem as colunas, tudo vira "aguardando", que e a leitura certa.
    for col in ("fight_id", "event_date", "fighter_1", "fighter_2", "winner"):
        if col not in base.columns:
            base[col] = base["fight_url"] if (col == "fight_id"
                                              and "fight_url" in base.columns) else np.nan
    base["_data"] = base["event_date"].astype(str).str[:10]

    out = history_df.copy()
    for col in ("fight_uid", "estado_resultado"):
        # object desde o inicio: coluna criada como float64 (so NaN) recusa
        # string no pandas 3 com "Invalid value 'u1' for dtype 'float64'"
        if col not in out.columns:
            out[col] = pd.Series([np.nan] * len(out), index=out.index, dtype="object")
        else:
            out[col] = out[col].astype("object")

    n_uid = n_estado = 0
    for i, r in out.iterrows():
        se_tem = pd.notna(r.get("actual_winner"))
        precisa_uid = pd.isna(r.get("fight_uid"))
        precisa_estado = pd.isna(r.get("estado_resultado"))
        if not (precisa_uid or precisa_estado):
            continue
        uid, estado = resolver(str(r["fighter_a"]), str(r["fighter_b"]),
                               r["event_date"], se_tem, base)
        # coerencia: linha COM vencedor preenchido a mao vale como vencedor
        # mesmo que a base ainda nao tenha a luta
        if se_tem and estado in (AGUARDANDO, NAO_ACONTECEU, DESCONHECIDO):
            estado = VENCEDOR
        if precisa_uid and uid:
            out.at[i, "fight_uid"] = uid
            n_uid += 1
        if precisa_estado:
            out.at[i, "estado_resultado"] = estado
            n_estado += 1

    logger.info("identidade: %d fight_uid e %d estado_resultado preenchidos.",
                n_uid, n_estado)
    return out
