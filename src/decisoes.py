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
    previsao_de          `event_date` + nome do evento do pré-registro de
                         onde a probabilidade veio
    prediction_id        a VERSÃO EXATA da previsão usada, em
                         `data/previsoes_snapshots.csv`. Regerar o card cria
                         uma versão nova; a decisão fica apontando para a que
                         ela de fato usou, e não para "a última"
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
           "previsao_de", "prediction_id",
           "model_prob_a", "model_name", "manifesto_sha256",
           "lado", "prob_lado", "odd_usada", "ref_preco", "ev", "elegivel",
           "bloqueio", "temporalidade", "retifica", "limite")

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


def _alinhar_cabecalho(caminho: Path) -> None:
    """
    Põe o arquivo no cabeçalho atual sem perder nenhuma linha antiga.

    Quando uma coluna nova aparece, anexar com o cabeçalho novo num arquivo
    escrito com o antigo desalinharia as linhas em silêncio. Então o arquivo
    é reescrito com o cabeçalho de hoje e **todas** as linhas que já estavam
    lá, com as colunas novas vazias — porque elas de fato não foram
    registradas naquele momento, e preenchê-las agora seria inventar o que a
    decisão antiga afirmava.
    """
    if not caminho.exists():
        return
    with caminho.open(newline="", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        if tuple(leitor.fieldnames or ()) == COLUNAS:
            return
        antigas = list(leitor)
        sobrando = [c for c in (leitor.fieldnames or ()) if c not in COLUNAS]
    with caminho.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS)
        w.writeheader()
        for linha in antigas:
            w.writerow({c: linha.get(c, "") for c in COLUNAS})
    logger.info("decisoes.csv migrado para o cabecalho atual: %d linha(s) "
                "preservada(s).", len(antigas))
    if sobrando:
        logger.warning("Colunas que existiam e nao existem mais foram "
                       "descartadas na migracao: %s", ", ".join(sobrando))


def registrar(decisoes: list[dict], arquivo: Path | None = None, *,
              retifica: str = "") -> int:
    """
    Anexa decisões ao log. Devolve quantas foram escritas.

    Anexa, nunca reescreve: duas avaliações do mesmo confronto em instantes
    diferentes são duas decisões, e sobrepor uma à outra apagaria justamente
    a informação de que o preço se moveu.

    `retifica` carimba o motivo da retificação nas linhas desta rodada,
    quando elas corrigem decisões anteriores. A decisão errada **fica** no
    arquivo: apagá-la esconderia que ela existiu, e é justamente o registro
    dela que torna a correção auditável.
    """
    if not decisoes:
        return 0
    caminho = Path(arquivo or ARQUIVO)
    agora = _agora()
    manifesto = manifesto_sha256()
    try:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        _alinhar_cabecalho(caminho)
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
                if retifica and not linha["retifica"]:
                    linha["retifica"] = retifica
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
        # a prova citada e a DO LADO da perna: a do lado A nao fala do preco
        # que paga uma entrada no lado B
        lado_a = (perna is not None
                  and str(perna["side"]).strip() == str(r["fighter_a"]).strip())
        ref = (r.get("ref_tp") if lado_a else r.get("ref_tp_b")) if perna else None
        out.append({
            "event_date": event_date,
            "fighter_a": r["fighter_a"], "fighter_b": r["fighter_b"],
            # APONTA para o pre-registro em vez de copia-lo por cima
            "previsao_de": f'{event_date}/{r.get("event_name") or ""}',
            "model_prob_a": "" if pd.isna(prob) else round(float(prob), 4),
            "model_name": r.get("model_name") or "",
            "prediction_id": "" if pd.isna(r.get("prediction_id"))
                             else str(r.get("prediction_id") or ""),
            "lado": r.get("model_side") or "",
            "prob_lado": "" if perna is None else round(perna["prob"], 4),
            "odd_usada": "" if perna is None else perna.get("odd"),
            "ref_preco": "" if ref is None or pd.isna(ref) else str(ref),
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
      preco         a cotação DO LADO ESCOLHIDO verificável, conferida
                    contra o que a decisão afirma (confronto, orientação,
                    casa, mercado, odd, status), dentro da validade e lida
                    ANTES do início programado.

    O sinal sharp é coisa separada e não entra aqui: `ev` usa a probabilidade
    do NOSSO modelo, é auto-referente, e vale como valor esperado estimado —
    não como evidência de valor.
    """
    import pandas as pd

    from src.casamento import ORG_NAO_CONFIRMADA, organizacao_confirmada
    from src.identidade import (BLOQUEIO_ESTADO, BLOQUEIO_POS_INICIO,
                                BLOQUEIO_PRECO_ENTRADA, BLOQUEIO_VERSAO,
                                DESCONHECIDO, NAO_ACONTECEU,
                                RETIRADO_DO_CARD)
    from src.snapshots import MOTIVO_POS_INICIO, arquivo_de, prob_do_lado, resolver
    from src.prediction_history import _load_raw
    from src.proveniencia import validar_cotacao

    card = pd.read_csv(Path(card_csv or config.UPCOMING_CARD_CSV))
    caminho_h = Path(history_csv or config.PREDICTION_HISTORY_CSV)
    hist = _load_raw(caminho_h)
    if hist.empty:
        return []
    snaps = arquivo_de(caminho_h)
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
            continue        # A VERSAO IMUTAVEL MANDA. O historico guarda um PONTEIRO
        # (`prediction_id`), e ponteiro tem de ser resolvido ANTES de
        # qualquer calculo. Ate a revisao de 06/11/2026 a decisao carregava o
        # id como etiqueta e lia probabilidade, lado e participantes das
        # colunas MUTAVEIS do historico -- as mesmas que a regeracao
        # reescreve. Uma divergencia entre a versao e a linha passava sem
        # nenhum sinal.
        pid = pre.get("prediction_id")
        pid = "" if pd.isna(pid) else str(pid or "").strip()
        versao, porque_versao = resolver(pid, event_date=event_date,
                                         fighter_a=a, fighter_b=b,
                                         arquivo=snaps)

        if versao is None:
            # Sem versao utilizavel nao ha o que decidir: a decisao entra
            # BLOQUEADA e visivel na contagem, e nada e lido do historico.
            # Falha ao persistir a versao nao pode liberar selecao.
            out.append({
                "event_date": event_date, "fighter_a": a, "fighter_b": b,
                "previsao_de": f'{event_date}/{pre.get("event_name") or ""}',
                "prediction_id": pid,
                "model_prob_a": "", "model_name": "", "manifesto_sha256": "",
                "lado": "", "prob_lado": "", "odd_usada": "", "ref_preco": "",
                "ev": "", "elegivel": False,
                # pos-inicio tem motivo proprio: nao e "versao ilegivel",
                # e previsao feita depois de a luta comecar
                "bloqueio": (BLOQUEIO_POS_INICIO
                             if porque_versao == MOTIVO_POS_INICIO
                             else f"{BLOQUEIO_VERSAO}: {porque_versao}"),
                "temporalidade": "",
            })
            continue

        lado_txt = str(versao.get("model_side") or "").strip()
        prob = prob_do_lado(versao, lado_txt)
        if prob is None:
            out.append({
                "event_date": event_date, "fighter_a": a, "fighter_b": b,
                "previsao_de": f'{event_date}/{pre.get("event_name") or ""}',
                "prediction_id": pid, "model_prob_a": "", "model_name": "",
                "manifesto_sha256": "", "lado": lado_txt, "prob_lado": "",
                "odd_usada": "", "ref_preco": "", "ev": "", "elegivel": False,
                "bloqueio": (f"{BLOQUEIO_VERSAO}: o lado {lado_txt!r} nao e "
                             f"nenhum dos dois participantes da versao "
                             f"({versao.get('fighter_a')} x "
                             f"{versao.get('fighter_b')})"),
                "temporalidade": str(versao.get("temporalidade") or ""),
            })
            continue

        prob_a_versao = float(versao["model_prob_a"])

        # DIVERGENCIA ENTRE A VERSAO E A LINHA. Se as duas discordam, uma das
        # duas foi mexida, e nao da para saber qual: a decisao bloqueia em vez
        # de escolher. O que ela NUNCA faz e usar o numero do historico --
        # mesmo bloqueada, o unico valor que ela reporta vem da versao.
        divergencias = []
        hist_prob = pre.get("model_prob_a")
        if not pd.isna(hist_prob):
            esperado = (prob_a_versao
                        if str(pre["fighter_a"]).strip().casefold()
                        == str(versao["fighter_a"]).strip().casefold()
                        else 1.0 - prob_a_versao)
            if abs(float(hist_prob) - esperado) > 5e-5:
                divergencias.append(
                    f"probabilidade (versao {esperado:.4f}, "
                    f"historico {float(hist_prob):.4f})")
        hist_lado = pre.get("model_side")
        if not pd.isna(hist_lado) and str(hist_lado).strip().casefold() != lado_txt.casefold():
            divergencias.append(
                f"lado (versao {lado_txt!r}, historico {str(hist_lado).strip()!r})")

        # O PRECO vem do card, do lado correspondente NA ORDEM DO CARD, com a
        # prova DAQUELE lado. Nao ha recurso a prova do lado A quando falta a
        # do lado B: a cotacao do adversario vem da mesma chamada e do mesmo
        # corpo, so com o lado trocado, e passaria por toda a conferencia.
        if lado_txt.casefold() == a.casefold():
            odd, ref = c.get("odds_a_tp"), c.get("ref_tp_a", c.get("ref_tp"))
        elif lado_txt.casefold() == b.casefold():
            odd, ref = c.get("odds_b_tp"), c.get("ref_tp_b")
        else:
            logger.warning("%s nao e nenhum dos dois lados do card (%s x %s) "
                           "-- nenhuma decisao montada.", lado_txt, a, b)
            continue

        motivo = ""
        if divergencias:
            motivo = (f"{BLOQUEIO_VERSAO}: a versao e o historico discordam em "
                      + "; ".join(divergencias))
        elif pre.get("estado_resultado") in (RETIRADO_DO_CARD, NAO_ACONTECEU,
                                             DESCONHECIDO):
            motivo = BLOQUEIO_ESTADO
        elif not organizacao_confirmada(c.get("fonte")):
            motivo = ORG_NAO_CONFIRMADA
        elif pd.isna(odd):
            motivo = BLOQUEIO_PRECO_ENTRADA
        else:
            # A DECISAO E CONFERIDA CONTRA O QUE AFIRMA: confronto, lado,
            # casa, mercado, odd, status do mercado e horario, e a odd tem de
            # aparecer no corpo persistido. `preco_verificavel` sozinho so
            # dizia que ALGUMA cotacao daquela chamada conferia.
            ok, porque = validar_cotacao(
                ref, casa="Thunderpick", mercado="h2h", lado=lado_txt,
                participantes=(a, b), odd=float(odd), exigir_atual=True,
                agora=agora)
            if not ok:
                motivo = f"{BLOQUEIO_PRECO_ENTRADA}: {porque}"

        out.append({
            "event_date": event_date, "fighter_a": a, "fighter_b": b,
            "previsao_de": f'{event_date}/{pre.get("event_name") or ""}',
            # aponta para a VERSAO EXATA, nao para "a mais nova"
            "prediction_id": pid,
            # TUDO lido da versao: probabilidade na orientacao DELA, lado,
            # modelo e hash do manifesto que a sustenta
            "model_prob_a": round(prob_a_versao, 4),
            "model_name": str(versao.get("model_name") or ""),
            "manifesto_sha256": str(versao.get("manifesto_sha256") or ""),
            "lado": lado_txt, "prob_lado": round(prob, 4),
            "odd_usada": "" if pd.isna(odd) else float(odd),
            "ref_preco": "" if ref is None or pd.isna(ref) else str(ref),
            "ev": "" if pd.isna(odd) else round(prob * float(odd), 4),
            "elegivel": motivo == "", "bloqueio": motivo,
            # temporalidade desconhecida continua DECLARADA desconhecida
            "temporalidade": str(versao.get("temporalidade") or ""),
        })
    return out
