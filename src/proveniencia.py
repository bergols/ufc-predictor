"""
src/proveniencia.py

Prova de onde cada cotação veio — item 8 da auditoria de integridade.

O que faltava
-------------
O histórico guardava os preços (`sharp_prob`, `sharp_best_odd`,
`close_prob`, `close_best_odd`, `odds_*_tp`) e **nada sobre a origem deles**:
nem o instante da cotação, nem qual casa deu cada preço, nem o mercado
consultado, nem qualquer referência verificável à resposta bruta. Depois do
fato não havia como auditar se um preço estava disponível naquele momento,
nem reconstruir uma captura.

O que este módulo registra
--------------------------
Uma linha por CHAMADA à API, em `data/proveniencia/chamadas.csv`, e uma
cópia comprimida da resposta bruta em `data/proveniencia/<ref>.json.gz`:

    ref            primeiros 16 hex do sha256 do corpo da resposta
    captured_at    instante REAL da resposta, em UTC, do relógio local
    fonte          the_odds_api
    endpoint       caminho consultado, SEM a chave
    regions        regiões pedidas
    markets        mercados pedidos
    n_eventos      quantos eventos vieram
    sha256         hash do corpo inteiro
    corpo_ref      caminho do arquivo com a resposta

A `ref` é carimbada nas linhas do histórico que nasceram daquela chamada
(`sharp_ref`, `close_ref`), então cada preço aponta para a resposta que o
produziu e o hash permite conferir que o arquivo não mudou.

Limite que fica declarado
-------------------------
`captured_at` é o instante em que a RESPOSTA chegou aqui — não o instante em
que a casa formou aquele preço, que a API não informa. É proveniência boa o
bastante para auditar a captura, e **não** é prova de que o preço esteve
disponível em qualquer outro momento.

A URL nunca é guardada com a chave: ela é sanitizada antes de virar registro.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import config

logger = logging.getLogger(__name__)

PASTA = config.DATA_DIR / "proveniencia"
INDICE = PASTA / "chamadas.csv"

_COLUNAS = ("ref", "captured_at", "fonte", "endpoint", "regions", "markets",
            "n_eventos", "sha256", "corpo_ref", "limite")

_LIMITE = ("captured_at e o instante em que a RESPOSTA chegou, nao o instante "
           "em que a casa formou o preco (a API nao informa). Nao prova "
           "disponibilidade em outro momento.")


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def registrar_chamada(corpo: str, *, fonte: str, endpoint: str,
                      regions: str = "", markets: str = "",
                      n_eventos: int | None = None,
                      guardar_corpo: bool = True) -> str:
    """
    Grava a proveniência de uma chamada e devolve a `ref`.

    `endpoint` entra sanitizado: chave de API nunca vira registro. Falha de
    escrita não derruba a captura — proveniência ausente é um problema menor
    que previsão perdida, e o log avisa.
    """
    from src.line_shopping import sanitizar

    digest = hashlib.sha256(corpo.encode("utf-8")).hexdigest()
    ref = digest[:16]
    corpo_ref = ""
    try:
        PASTA.mkdir(parents=True, exist_ok=True)
        if guardar_corpo:
            destino = PASTA / f"{ref}.json.gz"
            if not destino.exists():
                with gzip.open(destino, "wt", encoding="utf-8") as f:
                    f.write(corpo)
            # caminho relativo quando esta dentro do projeto (sobrevive a
            # mover o repositorio); absoluto quando nao esta, em vez de
            # estourar ValueError e perder a proveniencia inteira
            try:
                corpo_ref = str(destino.relative_to(config.PROJECT_ROOT))
            except ValueError:
                corpo_ref = str(destino)
        novo = not INDICE.exists()
        with INDICE.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_COLUNAS)
            if novo:
                w.writeheader()
            w.writerow({"ref": ref, "captured_at": _agora(), "fonte": fonte,
                        "endpoint": sanitizar(endpoint), "regions": regions,
                        "markets": markets,
                        "n_eventos": "" if n_eventos is None else n_eventos,
                        "sha256": digest, "corpo_ref": corpo_ref,
                        "limite": _LIMITE})
    except OSError as exc:
        logger.warning("Nao consegui gravar a proveniencia (%s) -- a captura "
                       "segue, mas esta cotacao fica sem prova.", exc)
    return ref


def conferir(ref: str) -> tuple[bool, str]:
    """(ok, motivo): o corpo guardado ainda bate com o hash do indice?"""
    if not INDICE.exists():
        return False, "indice de proveniencia ausente"
    linhas = [r for r in csv.DictReader(INDICE.open(encoding="utf-8"))
              if r["ref"] == ref]
    if not linhas:
        return False, f"ref {ref} nao esta no indice"
    reg = linhas[-1]
    if not reg["corpo_ref"]:
        return False, "corpo da resposta nao foi guardado"
    caminho = config.PROJECT_ROOT / reg["corpo_ref"]
    if not caminho.exists():
        return False, f"corpo ausente em {reg['corpo_ref']}"
    with gzip.open(caminho, "rt", encoding="utf-8") as f:
        atual = hashlib.sha256(f.read().encode("utf-8")).hexdigest()
    if atual != reg["sha256"]:
        return False, "o corpo guardado NAO bate com o hash registrado"
    return True, "ok"


def cotacao_verificavel(ref) -> bool:
    """A cotação desta linha tem prova utilizável?"""
    import pandas as pd
    if ref is None or (isinstance(ref, float) and pd.isna(ref)) or not str(ref).strip():
        return False
    return conferir(str(ref))[0]


# ---------------------------------------------------------------------------
# Proveniencia por COTACAO (item 2)
# ---------------------------------------------------------------------------
#
# A chamada prova o CORPO da resposta. Nao prova qual preco, de qual casa, em
# qual mercado, para qual participante foi efetivamente usado no calculo -- e
# nao prova nada sobre a Thunderpick, que nao esta naquele corpo: ela e
# raspada de outra fonte, por outro caminho. Uma resposta da The Odds API que
# comprova a Pinnacle NAO comprova a cotacao da casa.
#
# Entao cada preco usado ganha a sua propria linha, com a referencia a chamada
# que o contem. Preco sem linha aqui nao sustenta selecao.

INDICE_COTACOES = PASTA / "cotacoes.csv"

_COLUNAS_COT = ("ref", "chamada_ref", "captured_at", "fonte", "casa",
                "mercado", "evento_id", "participante_a", "participante_b",
                "lado", "odd", "fonte_atualizada_em", "status",
                "inicio_programado", "limite")

# POLITICA DE VALIDADE. Escolhida pela CADENCIA DE CAPTURA, nao por lucro
# historico: `scripts/auto_capture.py` roda de hora em hora, entao uma cotacao
# com mais de 6 horas significa seis tentativas seguidas falhadas -- esta
# velha por defeito, nao por desenho. O numero nao foi varrido contra
# resultado nenhum, e mudar o criterio depois de ver P&L seria escolher
# parametro para melhorar o passado.
#
# Vale para SELECAO (propor uma entrada agora). Nao se aplica a cotacao de
# fechamento ja congelada, que e por definicao historica e cujo trabalho e
# justamente descrever o passado.
VALIDADE_HORAS = 6

_LIMITE_COT = ("captured_at e quando a cotacao foi LIDA por nos. "
               "fonte_atualizada_em e o que a fonte declara, quando declara. "
               "Nenhum dos dois prova que o preco esteve disponivel em outro "
               "momento, nem que seria aceito num bilhete.")


def registrar_cotacao(*, chamada_ref: str | None, fonte: str, casa: str,
                      mercado: str, evento_id: str, participante_a: str,
                      participante_b: str, lado: str, odd: float,
                      fonte_atualizada_em: str = "", status: str = "",
                      inicio_programado: str = "") -> str:
    """
    Registra UM preco efetivamente usado e devolve a `ref` dele.

    `chamada_ref` liga o preco ao corpo bruto guardado. Sem ela a cotacao
    fica registrada mas NAO verificavel -- e `cotacao_verificavel` reprova,
    porque uma linha que afirma um preco sem apontar para o conteudo que o
    contem e testemunho de si mesma.
    """
    import hashlib
    semente = (f"{fonte}|{casa}|{mercado}|{evento_id}|{participante_a}|"
               f"{participante_b}|{lado}|{odd}|{chamada_ref or ''}")
    ref = hashlib.sha256(semente.encode("utf-8")).hexdigest()[:16]
    try:
        PASTA.mkdir(parents=True, exist_ok=True)
        novo = not INDICE_COTACOES.exists()
        with INDICE_COTACOES.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_COLUNAS_COT)
            if novo:
                w.writeheader()
            w.writerow({"ref": ref, "chamada_ref": chamada_ref or "",
                        "captured_at": _agora(), "fonte": fonte, "casa": casa,
                        "mercado": mercado, "evento_id": evento_id,
                        "participante_a": participante_a,
                        "participante_b": participante_b, "lado": lado,
                        "odd": odd,
                        "fonte_atualizada_em": fonte_atualizada_em,
                        "status": status,
                        "inicio_programado": inicio_programado,
                        "limite": _LIMITE_COT})
    except OSError as exc:
        logger.warning("Nao consegui gravar a proveniencia da cotacao (%s) -- "
                       "este preco fica sem prova.", exc)
    return ref


def _linha_de_cotacao(ref: str) -> dict | None:
    if not INDICE_COTACOES.exists():
        return None
    achadas = [r for r in csv.DictReader(INDICE_COTACOES.open(encoding="utf-8"))
               if r["ref"] == ref]
    return achadas[-1] if achadas else None


def conferir_cotacao(ref: str, *, exigir_atual: bool = False,
                     agora: datetime | None = None) -> tuple[bool, str]:
    """
    (ok, motivo) para UM preco usado.

    Confere, em ordem:
      1. a linha existe;
      2. ela aponta para uma chamada, e o corpo daquela chamada ainda bate
         com o hash registrado;
      3. com `exigir_atual`, que a leitura esteja dentro de VALIDADE_HORAS e
         ANTES do inicio programado -- as duas metades de "pre-luta e atual".
    """
    reg = _linha_de_cotacao(ref)
    if reg is None:
        return False, f"cotacao {ref} nao esta no indice"
    if not reg.get("chamada_ref"):
        return False, "cotacao sem referencia ao conteudo persistido"
    ok, motivo = conferir(reg["chamada_ref"])
    if not ok:
        return False, f"conteudo da chamada nao confere: {motivo}"
    if not exigir_atual:
        return True, "ok"

    agora = agora or datetime.now(timezone.utc)
    try:
        lida = datetime.fromisoformat(reg["captured_at"])
    except ValueError:
        return False, "captured_at ilegivel"
    idade = (agora - lida).total_seconds() / 3600.0
    if idade > VALIDADE_HORAS:
        return False, (f"cotacao de {idade:.1f}h, acima da validade de "
                       f"{VALIDADE_HORAS}h")
    inicio = (reg.get("inicio_programado") or "").strip()
    if not inicio:
        return False, "sem inicio programado: carater pre-luta nao verificavel"
    try:
        comeca = datetime.fromisoformat(inicio.replace("Z", "+00:00"))
    except ValueError:
        return False, f"inicio programado ilegivel ({inicio!r})"
    if lida >= comeca:
        return False, "cotacao lida DEPOIS do inicio programado: nao e pre-luta"
    if comeca <= agora:
        return False, "a luta ja comecou: cotacao nao serve para entrada"
    return True, "ok"


def preco_verificavel(ref, *, exigir_atual: bool = False) -> bool:
    """O preco desta linha tem prova utilizavel?"""
    import pandas as pd
    if ref is None or (isinstance(ref, float) and pd.isna(ref)) or not str(ref).strip():
        return False
    r = str(ref).strip()
    # aceita tanto ref de COTACAO (o caminho novo) quanto ref de CHAMADA (as
    # linhas gravadas antes de a cotacao ter indice proprio). A de chamada
    # nunca satisfaz `exigir_atual`: ela nao carrega inicio programado.
    if _linha_de_cotacao(r) is not None:
        return conferir_cotacao(r, exigir_atual=exigir_atual)[0]
    if exigir_atual:
        return False
    return conferir(r)[0]

# ---------------------------------------------------------------------------
# VALIDACAO DA DECISAO CONTRA A PROVA
# ---------------------------------------------------------------------------
#
# `conferir_cotacao` responde "esta linha existe e o corpo dela confere". Isso
# e menos do que parece: nao diz que a linha descreve ESTE confronto, ESTE
# lado, ESTA casa, ESTE mercado nem ESTA odd. Uma referencia do adversario
# passa por ali inteira -- o corpo e o mesmo, a chamada e a mesma, so o lado
# esta trocado -- e foi esse o primeiro defeito encontrado na revisao: cinco
# decisoes elegiveis citavam a cotacao do outro lado do confronto.
#
# O segundo defeito estava um nivel abaixo. A leitura do corpo persistido
# colhia pares (nome, odd) SOLTOS, varrendo a estrutura inteira. Um nome e uma
# odd que aparecessem em OUTRO confronto do mesmo corpo aprovavam a prova: o
# corpo de um card tem 30 lutas e 60 precos, e "Alpha paga 1,80 em algum lugar
# deste JSON" nao e prova de que Alpha x Beta paga 1,80.
#
# Entao a leitura passa a preservar o CONTEXTO de cada cotacao -- evento,
# inicio, os DOIS participantes, casa, mercado, lado, preco e status -- e
# todos sao conferidos contra a mesma linha do indice. Prova que nao bate com
# a afirmacao nao e prova daquela afirmacao.

_SUSPENSO = ("suspens", "suspend", "fechad", "closed", "congelad", "frozen",
             "cancelad", "cancel", "liquidad", "settled", "encerrad")


def _norm(nome) -> str:
    return str("" if nome is None else nome).strip().casefold()


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and float("-inf") < v < float("inf") else None


def _instante(x):
    """datetime de um horario da fonte, ou None quando ausente/ilegivel."""
    s = str("" if x is None else x).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _mesmo_instante(a, b) -> bool:
    """Mesmo momento, tolerando Z contra +00:00. Horario ilegivel de um lado
    cai para comparacao textual, em vez de passar por omissao."""
    da, db = _instante(a), _instante(b)
    if da is not None and db is not None:
        return da == db
    return _norm(a) == _norm(b)


def _suspenso(status) -> bool:
    """
    O mercado estava fechado? Numero e o codigo da casa (1 = aberto na
    Thunderpick); texto e conferido contra as palavras de fechamento.

    Status vazio NAO e tratado como suspenso: ausencia de informacao nao e
    informacao de fechamento. E nao e por essa frouxidao que a selecao passa,
    porque a cotacao ainda precisa aparecer no corpo persistido com o contexto
    inteiro batendo.
    """
    s = _norm(status)
    if not s or s in ("none", "nan"):
        return False
    n = _num(s)
    if n is not None:
        return n != 1.0
    return any(p in s for p in _SUSPENSO)


def _apelidos(fonte: str) -> dict:
    """
    Tabela de apelidos CONFERIDOS da fonte daquela cotacao. Cada fonte erra o
    nome de um jeito, e so entra par conferido a mao. Aqui nao ha
    correspondencia aproximada: ela sugere candidato, nao aprova prova.
    """
    f = _norm(fonte)
    try:
        if "thunderpick" in f:
            from src.thunderpick import APELIDOS_TP
            return APELIDOS_TP
        from src.line_shopping import APELIDOS_API
        return APELIDOS_API
    except ImportError:  # pragma: no cover - defensivo
        return {}


def _mesmo_nome(nosso, na_prova, fonte: str = "") -> bool:
    """O nome que a prova registra e o nosso lutador? Literal ou apelido."""
    p = _norm(na_prova)
    if not p:
        return False
    alvos = {_norm(nosso)}
    apelido = _apelidos(fonte).get(str("" if nosso is None else nosso).strip())
    if apelido:
        alvos.add(_norm(apelido))
    return p in alvos


def _mesmo_par(nossos, na_prova, fonte: str = "") -> bool:
    """O par da prova e o par deste confronto, em qualquer das duas ordens."""
    a, b = nossos[0], nossos[1]
    x, y = na_prova[0], na_prova[1]
    direta = _mesmo_nome(a, x, fonte) and _mesmo_nome(b, y, fonte)
    invertida = _mesmo_nome(a, y, fonte) and _mesmo_nome(b, x, fonte)
    return direta or invertida


def _linha_de_chamada(ref: str) -> dict | None:
    if not INDICE.exists():
        return None
    achadas = [r for r in csv.DictReader(INDICE.open(encoding="utf-8"))
               if r["ref"] == ref]
    return achadas[-1] if achadas else None


def cotacoes_do_corpo(chamada_ref) -> list[dict] | None:
    """
    Cotacoes lidas do corpo persistido, cada uma COM O SEU CONTEXTO.

    Devolve dicionarios com `evento_id`, `inicio`, `participante_a`,
    `participante_b`, `casa`, `mercado`, `lado`, `odd`, `status` e
    `atualizada_em` -- tudo tirado da MESMA sub-estrutura do corpo, para que
    nada seja conferido contra o contexto de outro confronto.

    `None` quando o corpo nao existe, nao confere com o hash ou nao e JSON
    legivel. Lista vazia quando o corpo e legivel e nao esta em nenhuma das
    formas reconhecidas -- e isso tambem bloqueia, porque uma cotacao que nao
    se acha no conteudo guardado nao esta comprovada por ele.

    Reconhece as duas formas que este projeto persiste:

      casa       `{home, away, odd_home, odd_away, inicio, status}` -- o
                 participante e a odd dele vem do MESMO objeto, que e a prova
                 da orientacao;
      feed       evento com `home_team`/`away_team` e `bookmakers[] ->
                 markets[] -> outcomes[]`, de onde saem casa, mercado e
                 `last_update` junto com o preco.

    Casa e mercado que o corpo nao declara (a pagina da Thunderpick nao
    nomeia a propria casa) herdam o que a LINHA DA CHAMADA registra: `fonte` e
    `markets`. E a mesma prova bruta, nao um preenchimento nosso.
    """
    ref = str("" if chamada_ref is None else chamada_ref).strip()
    if not ref:
        return None
    if not conferir(ref)[0]:
        return None
    reg = _linha_de_chamada(ref)
    if reg is None or not reg.get("corpo_ref"):
        return None
    caminho = Path(reg["corpo_ref"])
    if not caminho.is_absolute():
        caminho = config.PROJECT_ROOT / caminho
    try:
        with gzip.open(caminho, "rt", encoding="utf-8") as f:
            corpo = json.load(f)
    except (OSError, ValueError):
        return None
    out: list[dict] = []
    _colher(corpo, out, casa_padrao=reg.get("fonte") or "",
            mercado_padrao=(reg.get("markets") or "").split(",")[0].strip())
    return out


def _registro(**kw) -> dict:
    base = {"evento_id": "", "inicio": "", "participante_a": "",
            "participante_b": "", "casa": "", "mercado": "", "lado": "",
            "odd": None, "status": "", "atualizada_em": ""}
    base.update(kw)
    return base


def _colher(no, out: list, *, casa_padrao: str, mercado_padrao: str) -> None:
    if isinstance(no, list):
        for item in no:
            _colher(item, out, casa_padrao=casa_padrao,
                    mercado_padrao=mercado_padrao)
        return
    if not isinstance(no, dict):
        return

    # forma da casa: os dois nomes e as duas odds no MESMO objeto
    if isinstance(no.get("home"), str) and isinstance(no.get("away"), str):
        inicio = str(no.get("inicio") or no.get("startTime") or "")
        for lado_k, odd_k in (("home", "odd_home"), ("away", "odd_away")):
            v = _num(no.get(odd_k))
            if v is None:
                continue
            out.append(_registro(
                # a casa nao da id de evento proprio neste corpo; o inicio e
                # o que `registrar_cotacao` guarda como evento_id, e e o que
                # identifica a luta no catalogo dela
                evento_id=inicio, inicio=inicio,
                participante_a=no["home"], participante_b=no["away"],
                casa=casa_padrao, mercado=mercado_padrao,
                lado=no[lado_k], odd=v, status=no.get("status", "")))
        return

    # forma do feed de odds: evento -> casas -> mercados -> resultados
    if (isinstance(no.get("home_team"), str)
            and isinstance(no.get("away_team"), str)
            and isinstance(no.get("bookmakers"), list)):
        for bm in no["bookmakers"]:
            if not isinstance(bm, dict):
                continue
            casa = str(bm.get("title") or bm.get("key") or "")
            for mk in bm.get("markets") or []:
                if not isinstance(mk, dict):
                    continue
                for res in mk.get("outcomes") or []:
                    if not isinstance(res, dict):
                        continue
                    v = _num(res.get("price", res.get("odds")))
                    if v is None or not isinstance(res.get("name"), str):
                        continue
                    out.append(_registro(
                        evento_id=str(no.get("id") or ""),
                        inicio=str(no.get("commence_time") or ""),
                        participante_a=no["home_team"],
                        participante_b=no["away_team"],
                        casa=casa, mercado=str(mk.get("key") or ""),
                        lado=res["name"], odd=v,
                        status=res.get("status", mk.get("status", "")),
                        atualizada_em=str(bm.get("last_update")
                                          or mk.get("last_update") or "")))
        return

    for v in no.values():
        _colher(v, out, casa_padrao=casa_padrao, mercado_padrao=mercado_padrao)


def _confere_contexto(reg: dict, r: dict) -> str | None:
    """
    None quando o registro do corpo descreve a MESMA cotacao da linha do
    indice. Caso contrario, o campo que divergiu.

    Campo que o corpo nao declara nao e conferido contra ele -- mas tambem
    nao vira aprovacao: o par de participantes, o lado e a odd sao exigidos
    sempre, e e o conjunto deles que identifica a cotacao.
    """
    if not _mesmo_par((reg.get("participante_a"), reg.get("participante_b")),
                      (r["participante_a"], r["participante_b"])):
        return "confronto"
    if _norm(reg.get("lado")) != _norm(r["lado"]):
        return "lado"
    declarada = _num(reg.get("odd"))
    if declarada is None or r["odd"] is None:
        return "odd"
    if abs(declarada - r["odd"]) > 1e-6:
        return "odd"
    if r["evento_id"] and reg.get("evento_id") and \
            not _mesmo_instante(reg["evento_id"], r["evento_id"]) and \
            _norm(reg["evento_id"]) != _norm(r["evento_id"]):
        return "evento"
    if r["inicio"] and reg.get("inicio_programado") and \
            not _mesmo_instante(reg["inicio_programado"], r["inicio"]):
        return "inicio programado"
    if r["casa"] and reg.get("casa") and _norm(reg["casa"]) != _norm(r["casa"]):
        return "casa"
    if r["mercado"] and reg.get("mercado") and \
            _norm(reg["mercado"]) != _norm(r["mercado"]):
        return "mercado"
    if r["atualizada_em"] and reg.get("fonte_atualizada_em") and \
            not _mesmo_instante(reg["fonte_atualizada_em"], r["atualizada_em"]):
        return "atualizacao declarada pela fonte"
    return None


def validar_cotacao(ref, *, casa: str | None = None,
                    mercado: str | None = None, participantes=None,
                    lado=None, odd=None, evento_id: str | None = None,
                    exigir_atual: bool = False,
                    agora: datetime | None = None) -> tuple[bool, str]:
    """
    (ok, motivo) para a decisao que AFIRMA usar esta cotacao.

    Confere, em ordem:

      1. a linha de cotacao existe e o corpo da chamada dela ainda bate com o
         hash registrado (`conferir_cotacao`), mais atualidade e carater
         pre-luta quando `exigir_atual`;
      2. a linha e a chamada declaram a MESMA fonte;
      3. casa, mercado, evento, par de participantes, ORIENTACAO (o lado da
         prova e o lado escolhido, nao o do adversario) e odd batem com o que
         a decisao afirma;
      4. o mercado nao esta suspenso;
      5. existe no corpo persistido uma cotacao cujo CONTEXTO INTEIRO bate
         com a linha do indice -- evento, inicio, os dois participantes,
         casa, mercado, lado, preco, status e atualizacao declarada.

    O passo 5 e o que impede nome e odd achados em OUTRO confronto do mesmo
    corpo de aprovarem esta prova.

    Campo passado como None nao e conferido; campo passado tem de bater.
    Divergencia e bloqueio com motivo, nunca aviso.
    """
    r = str("" if ref is None else ref).strip()
    if not r or r.lower() in ("nan", "none"):
        return False, "sem referencia de cotacao"
    reg = _linha_de_cotacao(r)
    if reg is None:
        return False, f"cotacao {r} nao esta no indice de cotacoes"
    ok, motivo = conferir_cotacao(r, exigir_atual=exigir_atual, agora=agora)
    if not ok:
        return False, motivo
    fonte = reg.get("fonte", "")

    chamada = _linha_de_chamada(str(reg.get("chamada_ref") or "").strip())
    if chamada is None:
        return False, "a chamada citada nao esta no indice de chamadas"
    if _norm(chamada.get("fonte")) != _norm(fonte):
        return False, (f"a cotacao diz vir de {fonte!r} e o corpo guardado e "
                       f"de {chamada.get('fonte')!r}")

    if casa and _norm(reg.get("casa")) != _norm(casa):
        return False, f"a prova e da casa {reg.get('casa')!r}, nao de {casa!r}"
    if mercado and _norm(reg.get("mercado")) != _norm(mercado):
        return False, (f"a prova e do mercado {reg.get('mercado')!r}, nao de "
                       f"{mercado!r}")
    if evento_id and _norm(reg.get("evento_id")) != _norm(evento_id):
        return False, (f"a prova e do evento {reg.get('evento_id')!r}, nao de "
                       f"{evento_id!r}")

    if participantes:
        na_prova = (reg.get("participante_a"), reg.get("participante_b"))
        if not _mesmo_par(list(participantes), na_prova, fonte):
            return False, (f"a prova e do confronto {na_prova[0]} x "
                           f"{na_prova[1]}, que nao e este")

    if lado is not None and not _mesmo_nome(lado, reg.get("lado"), fonte):
        return False, (f"a prova e do lado {reg.get('lado')!r} e a decisao e "
                       f"de {lado!r}: referencia do adversario")

    declarada = _num(reg.get("odd"))
    if declarada is None:
        return False, "odd da prova ilegivel"
    if odd is not None:
        pedida = _num(odd)
        if pedida is None:
            return False, "odd da decisao ilegivel"
        if abs(declarada - pedida) > 1e-6:
            return False, (f"a decisao usa {pedida}, a prova registra "
                           f"{declarada}")

    if _suspenso(reg.get("status")):
        return False, (f"mercado {reg.get('status')!r} na prova: preco "
                       "suspenso nao e preco de entrada")

    registros = cotacoes_do_corpo(reg.get("chamada_ref"))
    if registros is None:
        return False, "corpo persistido ausente, alterado ou ilegivel"
    if not registros:
        return False, ("corpo persistido nao esta em forma reconhecida: a "
                       "cotacao nao esta comprovada por ele")

    divergencias = []
    for candidato in registros:
        falta = _confere_contexto(reg, candidato)
        if falta is None:
            if _suspenso(candidato["status"]):
                return False, ("mercado suspenso no corpo persistido: preco "
                               "suspenso nao e preco de entrada")
            return True, "ok"
        divergencias.append((falta, candidato))

    # diagnostico: o caso perigoso e a odd daquele lado existir no corpo, mas
    # em outro confronto. Dizer isso e melhor que dizer "nao achei".
    for falta, candidato in divergencias:
        if falta == "confronto" and _norm(candidato["lado"]) == _norm(reg.get("lado")):
            return False, (f"a odd {declarada} para {reg.get('lado')!r} aparece "
                           f"no corpo, mas em OUTRO confronto "
                           f"({candidato['participante_a']} x "
                           f"{candidato['participante_b']})")
    perto = [f for f, _ in divergencias
             if f not in ("confronto", "lado", "odd")]
    if perto:
        return False, (f"a cotacao esta no corpo com contexto diferente: "
                       f"{perto[0]} divergente")
    return False, (f"a odd {declarada} para {reg.get('lado')!r} em "
                   f"{reg.get('participante_a')} x {reg.get('participante_b')} "
                   "NAO aparece no corpo persistido daquela chamada")
