"""
src/fechamento.py

Preenche `actual_winner` no `odds_template.csv` a partir da base de
resultados — e PARA quando não tem certeza.

Por que existe
--------------
Todo fechamento até out/2026 foi feito à mão: eu listava os resultados da
base e escrevia o vencedor de cada linha. É mecânico, mas é também o ponto
onde um erro é irreversível — vencedor errado entra num denominador que nunca
mais é revisto.

A automação só vale se ela for mais conservadora que a mão. Por isso aqui o
critério não é "achou alguma coisa", é **não restou dúvida**:

  ok        os dois lutadores casaram, cada um com um lado distinto da luta
            na base, e o vencedor é um dos dois. É o único caso que escreve.
  empate    a luta existe e não tem vencedor (empate, no contest). Fica em
            branco de propósito — sai dos denominadores sozinha. Aconteceu em
            10/out (Kareckaite x Gatto, decisão majoritária).
  ausente   nenhuma luta da data casa com esse par. Normalmente adversário
            trocado ou luta cancelada — em 26/set a Melissa Amaya lutou
            contra OUTRA pessoa, e a linha como registrada nunca aconteceu.
  ambigua   mais de uma luta da data casa, ou o vencedor da base não bate com
            nenhum dos dois lados. Nunca escreve.

A grafia escrita é SEMPRE a da nossa linha, nunca a da base: o sync rejeita
vencedor que não seja exatamente um dos dois lados, e a base escreve nomes
diferentes dos nossos com frequência (Mehemmedeli/Mahammadali Osmanli).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import config
from src.utils import best_name_match

logger = logging.getLogger(__name__)


@dataclass
class Relato:
    fighter_a: str
    fighter_b: str
    estado: str                 # ok | empate | ausente | ambigua
    vencedor: str | None = None
    detalhe: str = ""


def _lado(nome: str, candidatos: list[str]) -> str | None:
    if nome in candidatos:
        return nome
    return best_name_match(nome, candidatos)


def casar_resultado(fighter_a: str, fighter_b: str,
                    lutas_do_dia: pd.DataFrame) -> Relato:
    """Resultado desta luta na base, já traduzido para a NOSSA grafia."""
    achadas = []
    for _, r in lutas_do_dia.iterrows():
        lados = [str(r["fighter_1"]), str(r["fighter_2"])]
        ma, mb = _lado(fighter_a, lados), _lado(fighter_b, lados)
        if ma and mb and ma != mb:
            achadas.append((r, ma, mb))

    if not achadas:
        return Relato(fighter_a, fighter_b, "ausente",
                      detalhe="nenhuma luta da data casa com este par "
                              "(adversário trocado ou luta cancelada)")
    if len(achadas) > 1:
        return Relato(fighter_a, fighter_b, "ambigua",
                      detalhe=f"{len(achadas)} lutas da data casam com este par")

    r, ma, mb = achadas[0]
    if pd.isna(r.get("winner")):
        return Relato(fighter_a, fighter_b, "empate",
                      detalhe=f"sem vencedor na base ({r.get('method', '')}".strip() + ")")

    vencedor_base = str(r["winner"])
    if vencedor_base == ma:
        return Relato(fighter_a, fighter_b, "ok", vencedor=fighter_a)
    if vencedor_base == mb:
        return Relato(fighter_a, fighter_b, "ok", vencedor=fighter_b)
    return Relato(fighter_a, fighter_b, "ambigua",
                  detalhe=f"vencedor da base ({vencedor_base!r}) não é nenhum dos "
                          f"dois lados casados ({ma!r}, {mb!r})")


def preencher_template(event_date: str, template_csv: Path | None = None,
                       fights_df: pd.DataFrame | None = None,
                       dry_run: bool = False) -> tuple[int, list[Relato]]:
    """
    Escreve `actual_winner` nas linhas abertas do evento. Devolve
    (quantas escreveu, relatos de TODAS as linhas).

    Edita o arquivo linha a linha, preservando a formatação: reescrever com
    pandas trocaria `1.540` por `1.54` no arquivo inteiro e afogaria o diff
    real num diff de formatação.
    """
    caminho = Path(template_csv or config.ODDS_TEMPLATE_CSV)
    if fights_df is None:
        fights_df = pd.read_csv(config.RAW_FIGHTS_CSV)
    do_dia = fights_df[fights_df["event_date"].astype(str).str[:10] == event_date]
    if do_dia.empty:
        logger.warning("Nenhuma luta de %s na base -- o evento ja entrou?", event_date)
        return 0, []

    linhas = caminho.read_text(encoding="utf-8").splitlines(keepends=True)
    relatos, n = [], 0
    for i, linha in enumerate(linhas):
        if event_date not in linha:
            continue
        campos = linha.rstrip("\r\n").split(",")
        if len(campos) < 8 or campos[2] != event_date:
            continue
        if campos[-1].strip():
            continue                      # ja fechada: nao reescreve
        rel = casar_resultado(campos[3], campos[4], do_dia)
        relatos.append(rel)
        if rel.estado != "ok":
            continue
        campos[-1] = rel.vencedor
        linhas[i] = ",".join(campos) + linha[len(linha.rstrip("\r\n")):]
        n += 1

    if n and not dry_run:
        caminho.write_text("".join(linhas), encoding="utf-8")
    for rel in relatos:
        if rel.estado == "ok":
            logger.info("%s x %s -> %s", rel.fighter_a, rel.fighter_b, rel.vencedor)
        else:
            logger.warning("%s x %s: %s EM BRANCO -- %s",
                           rel.fighter_a, rel.fighter_b, rel.estado.upper(), rel.detalhe)
    return n, relatos
