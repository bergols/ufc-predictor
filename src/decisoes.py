"""
src/decisoes.py

Registro de DECISÕES POSTERIORES, separado do pré-registro.

O problema
----------
O pré-registro congelado (`data/prediction_history.csv`) é uma afirmação com
data: *em tal instante, com tal modelo e tal base, a probabilidade era p, e o
preço era o*. O valor dele vem exatamente de não ser reescrito.

Mas o preço anda. Recapturar a odd de um card que já foi pré-registrado e
escrever o preço novo na linha antiga produziria uma linha que não descreve
momento nenhum: probabilidade de terça, preço de sexta, apresentada como se
fosse um registro único. O CLV calculado a partir dela não mede movimento de
linha, mede a confusão.

A solução
---------
Combinar previsão antiga com preço novo é uma **decisão nova**, e ela vai num
arquivo próprio, `data/decisoes.csv`, uma linha por avaliação:

    decidido_em          instante desta decisão (UTC)
    event_date           evento
    fighter_a/b          confronto, na ordem do pré-registro
    previsao_de          `event_date` + instante do pré-registro de onde a
                         probabilidade veio — a decisão APONTA para o
                         pré-registro, não o copia por cima
    model_prob_a         a probabilidade congelada, reproduzida aqui para
                         leitura, nunca para substituir a original
    model_name           modelo que a produziu
    manifesto_sha256     hash do manifesto de treino daquele modelo: é o que
                         liga a decisão aos DADOS que a sustentam
    odd_usada / lado     o preço novo e de que lado
    ref_preco            proveniência daquele preço
    ev                   valor esperado estimado com o preço novo
    elegivel             a decisão pode virar entrada?
    bloqueio             se não, por quê

Nada aqui altera o histórico. Uma decisão é um documento adicional, e o
pré-registro original continua sendo a única coisa que se usa para medir
acerto do modelo.

O que este arquivo NÃO é
------------------------
Não é registro de aposta, não é recomendação e não observa resultado: `ev` é
**valor esperado estimado**, calculado com a probabilidade do nosso modelo,
que é auto-referente. Perna elegível quer dizer *as provas estão no lugar*, e
não *esta entrada é boa*.
"""
from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

import config

logger = logging.getLogger(__name__)

ARQUIVO = config.DATA_DIR / "decisoes.csv"

COLUNAS = ("decidido_em", "event_date", "fighter_a", "fighter_b",
           "previsao_de", "model_prob_a", "model_name", "manifesto_sha256",
           "lado", "odd_usada", "ref_preco", "ev", "elegivel", "bloqueio",
           "limite")

_LIMITE = ("ev e valor esperado ESTIMADO com a probabilidade do nosso modelo, "
           "que e auto-referente. elegivel significa que as provas estao no "
           "lugar, nao que a entrada e boa. Nao e registro de aposta.")


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def manifesto_sha256(metadata_path: Path | None = None) -> str:
    """
    sha256 do manifesto de treino em vigor — a ligação entre esta decisão e
    os dados que a sustentam. Vazio quando não há manifesto: a decisão fica
    registrada sem essa ligação, em vez de receber um hash inventado.
    """
    from src.manifesto import sha256_arquivo
    caminho = Path(metadata_path or (config.PROJECT_ROOT / "models" /
                                     "training_metadata.json"))
    return sha256_arquivo(caminho) or ""


def registrar(decisoes: list[dict], arquivo: Path | None = None) -> int:
    """
    Anexa decisões ao log. Devolve quantas foram escritas.

    Anexa, nunca reescreve: duas avaliações do mesmo confronto em instantes
    diferentes são duas decisões, e sobrepor uma à outra apagaria justamente
    a informação de que o preço se moveu.
    """
    if not decisoes:
        return 0
    caminho = Path(arquivo or ARQUIVO)
    agora = _agora()
    manifesto = manifesto_sha256()
    try:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        novo = not caminho.exists()
        with caminho.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUNAS)
            if novo:
                w.writeheader()
            for d in decisoes:
                linha = {c: "" for c in COLUNAS}
                linha.update({k: v for k, v in d.items() if k in COLUNAS})
                linha["decidido_em"] = agora
                linha.setdefault("manifesto_sha256", manifesto)
                if not linha["manifesto_sha256"]:
                    linha["manifesto_sha256"] = manifesto
                linha["limite"] = _LIMITE
                w.writerow(linha)
    except OSError as exc:
        logger.warning("Nao consegui gravar as decisoes (%s).", exc)
        return 0
    logger.info("%d decisao(oes) registrada(s) em %s", len(decisoes), caminho)
    return len(decisoes)


def avaliar_evento(event_date: str, history_csv: Path | None = None) -> list[dict]:
    """
    Monta uma decisão por perna do evento, a partir do que está GRAVADO.

    Lê o pré-registro (probabilidade congelada e preço da casa), aplica o
    gate de elegibilidade e devolve as linhas prontas para `registrar`. Não
    consulta rede e não escreve no histórico: é leitura mais aritmética.

    O `ev` usa a odd da casa quando ela existe, porque é essa que se pega —
    a mesma regra de `prediction_history.model_side_leg`.
    """
    import pandas as pd

    from src.identidade import motivo_de_bloqueio
    from src.prediction_history import _evento_ja_passou, _load_raw, model_side_leg

    df = _load_raw(Path(history_csv or config.PREDICTION_HISTORY_CSV))
    if df.empty:
        return []
    ev_df = df[df["event_date"] == event_date]
    futuro = not _evento_ja_passou(event_date)

    out = []
    for _, r in ev_df.iterrows():
        perna = model_side_leg(r)
        motivo = motivo_de_bloqueio(r, exigir_atual=futuro)
        prob = r.get("model_prob_a")
        out.append({
            "event_date": event_date,
            "fighter_a": r["fighter_a"], "fighter_b": r["fighter_b"],
            # APONTA para o pre-registro em vez de copia-lo por cima
            "previsao_de": f'{event_date}/{r.get("event_name") or ""}',
            "model_prob_a": "" if pd.isna(prob) else round(float(prob), 4),
            "model_name": r.get("model_name") or "",
            "lado": r.get("model_side") or "",
            "odd_usada": "" if perna is None else perna.get("odd"),
            "ref_preco": r.get("ref_tp") or r.get("sharp_ref") or "",
            "ev": "" if perna is None else round(
                float(perna["prob"]) * float(perna["odd"]), 4),
            "elegivel": motivo is None,
            "bloqueio": motivo or "",
        })
    return out


def avaliar_com_precos_novos(event_date: str, card_csv: Path | None = None,
                             history_csv: Path | None = None,
                             agora=None) -> list[dict]:
    """
    Decisão posterior: previsão CONGELADA cruzada com o preço de HOJE.

    É esta função que resolve o problema do item 3. Recapturar a odd de um
    card já pré-registrado e escrever o preço novo na linha antiga produziria
    uma linha que não descreve instante nenhum — probabilidade de terça,
    preço de sexta. Então o preço novo não entra no histórico: ele entra
    aqui, numa decisão com instante próprio, que APONTA para o pré-registro
    de onde a probabilidade veio.

    A probabilidade vem do HISTÓRICO (congelada). O preço, a prova do preço e
    a prova de organização vêm do CSV DO CARD, recém-capturado. A
    elegibilidade é avaliada com as provas do CARD, não com as da linha
    antiga: a linha antiga tem um preço sem prova, e é correto que ela
    continue bloqueada.

    O que se exige para `elegivel`:

      estado        a luta não pode ter saído do card;
      organizacao   `fonte` do card em {ambas, so_ufcstats};
      preco         `ref_tp` verificável, dentro da validade e lida ANTES do
                    início programado.

    O sinal sharp é coisa separada e não entra aqui: `ev` usa a probabilidade
    do NOSSO modelo, é auto-referente, e vale como valor esperado estimado —
    não como evidência de valor.
    """
    import pandas as pd

    from src.casamento import ORG_NAO_CONFIRMADA, organizacao_confirmada
    from src.identidade import (BLOQUEIO_ESTADO, BLOQUEIO_PRECO_ENTRADA,
                                DESCONHECIDO, NAO_ACONTECEU, RETIRADO_DO_CARD)
    from src.prediction_history import _load_raw
    from src.proveniencia import preco_verificavel

    card = pd.read_csv(Path(card_csv or config.UPCOMING_CARD_CSV))
    hist = _load_raw(Path(history_csv or config.PREDICTION_HISTORY_CSV))
    if hist.empty:
        return []
    congelado = {}
    for _, r in hist[hist["event_date"] == event_date].iterrows():
        congelado[frozenset((str(r["fighter_a"]).strip(),
                             str(r["fighter_b"]).strip()))] = r

    out = []
    for _, c in card.iterrows():
        a, b = str(c["fighter_a"]).strip(), str(c["fighter_b"]).strip()
        pre = congelado.get(frozenset((a, b)))
        if pre is None:
            # Luta no card e sem pre-registro: nao ha previsao congelada para
            # cruzar, e inventar uma agora seria pre-registro disfarcado de
            # decisao. Fica fora, visivel na contagem.
            logger.warning("%s vs %s esta no card e NAO tem pre-registro -- "
                           "nenhuma decisao montada.", a, b)
            continue
        lado = pre.get("model_side")
        prob_a = pre.get("model_prob_a")
        if pd.isna(lado) or pd.isna(prob_a):
            continue
        lado_e_a = str(lado).strip() == a
        prob = float(prob_a) if lado_e_a else 1 - float(prob_a)
        odd = c.get("odds_a_tp") if lado_e_a else c.get("odds_b_tp")
        ref = c.get("ref_tp")

        motivo = ""
        if pre.get("estado_resultado") in (RETIRADO_DO_CARD, NAO_ACONTECEU,
                                           DESCONHECIDO):
            motivo = BLOQUEIO_ESTADO
        elif not organizacao_confirmada(c.get("fonte")):
            motivo = ORG_NAO_CONFIRMADA
        elif pd.isna(odd) or not preco_verificavel(ref, exigir_atual=True):
            motivo = BLOQUEIO_PRECO_ENTRADA

        out.append({
            "event_date": event_date, "fighter_a": a, "fighter_b": b,
            "previsao_de": f'{event_date}/{pre.get("event_name") or ""}',
            "model_prob_a": round(float(prob_a), 4),
            "model_name": pre.get("model_name") or "",
            "lado": str(lado).strip(),
            "odd_usada": "" if pd.isna(odd) else float(odd),
            "ref_preco": "" if pd.isna(ref) else str(ref),
            "ev": "" if pd.isna(odd) else round(prob * float(odd), 4),
            "elegivel": motivo == "", "bloqueio": motivo,
        })
    return out
