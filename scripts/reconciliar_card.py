"""
scripts/reconciliar_card.py

Reconcilia o card REGISTRADO com o card ATUAL do UFCStats, e corrige o
historico sem transferir previsao entre confrontos.

    python -m scripts.reconciliar_card --event-date 2026-11-07
    python -m scripts.reconciliar_card --event-date 2026-11-07 --aplicar

Por que existe
--------------
O card de 07/nov/2026 mudou de identidade depois de registrado: o evento
passou de "UFC Fight Night: Bonfim vs Brady" para "UFC Fight Night: Cortes
Acosta vs. Walker", o main event original saiu, e o ciclo registrou as lutas
novas sem perceber que uma delas ja existia com OUTRA GRAFIA -- Gabriel
Lourenco e Gabriel Lorenco, a mesma luta em duas linhas.

A fonte de verdade aqui e a pagina do EVENTO no UFCStats, que identifica
explicitamente organizacao e evento. Numero de casas de aposta NAO prova
pertencimento ao card: serve so como indicador de cobertura de mercado.

Regras que este script segue
----------------------------
  - previsao original nunca e sobrescrita;
  - confronto retirado do card MANTEM a linha e recebe estado proprio
    (`retirado_do_card`), ficando inelegivel para os denominadores;
  - confronto novo entra como registro novo -- nao herda previsao nem odd de
    ninguem;
  - duplicata por grafia e resolvida mantendo a linha ORIGINAL e adotando a
    grafia do UFCStats; os dados de MERCADO capturados na linha duplicada
    (sharp, odd da casa) sao preservados, porque descrevem a mesma luta e
    foram capturados antes do evento;
  - divergencia que nao da para resolver nao e escrita: sai no relatorio
    como `divergencia_nao_resolvida`.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field

import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIRMADO = "no_card"
RETIRADO = "retirado_do_card"
NOVO = "novo_no_card"
DUPLICATA = "duplicata_de_grafia"
DIVERGENCIA = "divergencia_nao_resolvida"


@dataclass
class Item:
    fighter_a: str
    fighter_b: str
    situacao: str
    origem: str                       # registrado | ufcstats | ambos
    ordem: int | None = None
    nota: str = ""
    casado_com: tuple[str, str] | None = None
    notas: list[str] = field(default_factory=list)


def _chave(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted([str(a).strip(), str(b).strip()]))


def _mesmo_confronto(par1, par2) -> bool:
    """Mesma luta apesar de grafia. Exige que os DOIS lados casem, cada um
    com um lado distinto -- a mesma guarda do resto do projeto."""
    from src.utils import best_name_match
    a1, b1 = par1
    cand = list(par2)
    ma = a1 if a1 in cand else best_name_match(a1, cand)
    mb = b1 if b1 in cand else best_name_match(b1, cand)
    return bool(ma and mb and ma != mb)


def reconciliar(registrado: pd.DataFrame, card_ufcstats: list[tuple[str, str]]) -> list[Item]:
    """Uma entrada por confronto, dos dois lados da comparacao."""
    itens: list[Item] = []
    usados_card = set()

    for _, r in registrado.iterrows():
        par = _chave(r["fighter_a"], r["fighter_b"])
        exato = [i for i, (a, b) in enumerate(card_ufcstats) if _chave(a, b) == par]
        if exato:
            usados_card.add(exato[0])
            itens.append(Item(r["fighter_a"], r["fighter_b"], CONFIRMADO, "ambos",
                              ordem=exato[0]))
            continue
        aprox = [i for i, (a, b) in enumerate(card_ufcstats)
                 if i not in usados_card and _mesmo_confronto(par, _chave(a, b))]
        if len(aprox) == 1:
            i = aprox[0]
            usados_card.add(i)
            itens.append(Item(r["fighter_a"], r["fighter_b"], DUPLICATA, "ambos",
                              ordem=i, casado_com=card_ufcstats[i],
                              nota="grafia diferente do UFCStats"))
            continue
        if len(aprox) > 1:
            itens.append(Item(r["fighter_a"], r["fighter_b"], DIVERGENCIA, "registrado",
                              nota=f"casa com {len(aprox)} lutas do card"))
            continue
        itens.append(Item(r["fighter_a"], r["fighter_b"], RETIRADO, "registrado",
                          nota="nao esta no card atual do UFCStats"))

    for i, (a, b) in enumerate(card_ufcstats):
        if i not in usados_card:
            itens.append(Item(a, b, NOVO, "ufcstats", ordem=i,
                              nota="no card e sem registro"))
    return itens


def _relatorio(itens: list[Item], evento: str, data: str) -> None:
    print(f"\n=== RECONCILIACAO: {evento} ({data}) ===")
    print(f"{'situacao':26} {'ord':>4}  confronto")
    print("-" * 80)
    for s in (CONFIRMADO, DUPLICATA, NOVO, RETIRADO, DIVERGENCIA):
        for it in [x for x in itens if x.situacao == s]:
            ordem = f"{it.ordem:>4}" if it.ordem is not None else "   -"
            extra = f"  ({it.nota})" if it.nota else ""
            alvo = f"  -> {it.casado_com[0]} x {it.casado_com[1]}" if it.casado_com else ""
            print(f"{s:26} {ordem}  {it.fighter_a} x {it.fighter_b}{alvo}{extra}")
    print()
    for s in (CONFIRMADO, DUPLICATA, NOVO, RETIRADO, DIVERGENCIA):
        n = sum(1 for x in itens if x.situacao == s)
        if n:
            print(f"  {s:26} {n}")


def main() -> int:
    p = argparse.ArgumentParser(description="Reconcilia card registrado x UFCStats.")
    p.add_argument("--event-date", required=True)
    p.add_argument("--aplicar", action="store_true",
                   help="Escreve as correcoes no historico (padrao: so relata)")
    args = p.parse_args()

    from src.card_sources import card_do_ufcstats
    from src.prediction_history import _load_raw

    ufc = card_do_ufcstats(args.event_date)
    h = _load_raw(config.PREDICTION_HISTORY_CSV)
    reg = h[h["event_date"] == args.event_date]
    if reg.empty:
        logger.error("Nenhuma linha registrada para %s.", args.event_date)
        return 1

    itens = reconciliar(reg, ufc["lutas"])
    _relatorio(itens, ufc["nome"], args.event_date)

    div = [i for i in itens if i.situacao == DIVERGENCIA]
    if div:
        logger.error("%d divergencia(s) nao resolvida(s) -- NADA escrito. "
                     "Resolva a mao antes de aplicar.", len(div))
        return 1
    if not args.aplicar:
        logger.info("Relatorio apenas. Use --aplicar para escrever.")
        return 0

    out = h.copy()
    idx = {(_chave(r["fighter_a"], r["fighter_b"])): i
           for i, r in out[out["event_date"] == args.event_date].iterrows()}

    # 1. duplicata de grafia: mantem a ORIGINAL (a mais antiga no arquivo),
    #    adota a grafia do UFCStats, herda so os dados de MERCADO da outra
    for it in [x for x in itens if x.situacao == DUPLICATA]:
        i_orig = idx[_chave(it.fighter_a, it.fighter_b)]
        i_dup = idx.get(_chave(*it.casado_com))
        if i_dup is None or i_dup == i_orig:
            continue
        manter, descartar = min(i_orig, i_dup), max(i_orig, i_dup)
        for col in ("sharp_prob", "sharp_best_odd", "ev_sharp", "ev_sharp_tp",
                    "odds_a_tp", "odds_b_tp", "close_prob", "close_best_odd", "clv"):
            if pd.isna(out.at[manter, col]) and pd.notna(out.at[descartar, col]):
                out.at[manter, col] = out.at[descartar, col]
        # adotar a grafia do UFCStats nao pode TROCAR os lados: model_prob_a
        # e do lado A, e inverter aqui faria a previsao descrever o
        # adversario. So renomeia quando a orientacao bate.
        from src.utils import best_name_match
        novo_a, novo_b = it.casado_com
        bate = (best_name_match(out.at[manter, "fighter_a"], [novo_a, novo_b]) == novo_a
                and best_name_match(out.at[manter, "fighter_b"], [novo_a, novo_b]) == novo_b)
        if bate:
            out.at[manter, "fighter_a"] = novo_a
            out.at[manter, "fighter_b"] = novo_b
        else:
            logger.warning("%s x %s: grafia do UFCStats inverteria os lados -- "
                           "nome mantido como estava.",
                           out.at[manter, "fighter_a"], out.at[manter, "fighter_b"])
        logger.info("duplicata resolvida: mantida a linha %d (previsao original "
                    "%.4f), grafia adotada do UFCStats, linha %d removida.",
                    manter, out.at[manter, "model_prob_a"], descartar)
        out = out.drop(index=descartar)

    # 2. retirado do card: mantem a linha, marca inelegivel
    for it in [x for x in itens if x.situacao == RETIRADO]:
        i = idx[_chave(it.fighter_a, it.fighter_b)]
        if i in out.index:
            out.at[i, "estado_resultado"] = RETIRADO
            logger.info("retirado do card: %s x %s -- linha mantida, inelegivel.",
                        it.fighter_a, it.fighter_b)

    # 3. estado faltante nas linhas novas
    from src.identidade import AGUARDANDO
    massa = (out["event_date"] == args.event_date) & out["estado_resultado"].isna()
    out.loc[massa, "estado_resultado"] = AGUARDANDO

    out.to_csv(config.PREDICTION_HISTORY_CSV, index=False)
    logger.info("Historico: %d -> %d linhas.", len(h), len(out))

    # 4. o template segue o historico. Ele tambem guarda a grafia antiga da
    #    duplicata, e linha orfa no template nunca fecha nada -- so suja a
    #    conferencia entre as quatro fontes.
    t = pd.read_csv(config.ODDS_TEMPLATE_CSV)
    do_dia = t["event_date"].astype(str) == args.event_date
    validos = {_chave(r["fighter_a"], r["fighter_b"])
               for _, r in out[out["event_date"] == args.event_date].iterrows()}
    orfas = do_dia & ~t.apply(lambda r: _chave(r["fighter_a"], r["fighter_b"]) in validos,
                              axis=1)
    if orfas.any():
        for _, r in t[orfas].iterrows():
            logger.info("template: removendo orfa %s x %s (nao esta no historico "
                        "deste evento)", r["fighter_a"], r["fighter_b"])
        t[~orfas].to_csv(config.ODDS_TEMPLATE_CSV, index=False)
        logger.info("Template: %d -> %d linhas.", len(t), int((~orfas).sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
