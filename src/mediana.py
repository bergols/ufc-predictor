"""
src/mediana.py

Proveniência da MEDIANA de casas — a referência de mercado.

Por que existe
--------------
`PROTOCOLO_VANTAGEM_V2.md` §4.2: a classe máxima alcançável no modo sombra era
`prospectiva_parcial` porque a mediana era calculada e gravada **sem prova**.
A odd da Pinnacle e a da Thunderpick já tinham proveniência por preço; a
mediana, que é o candidato A inteiro, não tinha nenhuma — não havia casa,
mercado, instante de leitura nem corpo persistido por linha.

Isto conserta exatamente essa lacuna, **sem chamada nova**: a resposta bruta da
The Odds API já é persistida por `line_shopping.fetch_live_odds`
(`registrar_chamada`). O que faltava era amarrar o número calculado àquele
corpo.

O que a prova amarra
--------------------
    chamada_ref      a captura exata, com hash do corpo
    captured_at      o instante REAL em que lemos a resposta
    evento_id        o id do evento NA FONTE
    participante_a/b os dois nomes, na orientação usada no cálculo
    mercado          h2h
    casas            as casas efetivamente incluídas, e quantas
    odd_a / odd_b    as duas odds que entraram na decisão

**Duas capturas com corpo idêntico** colapsam na mesma `chamada_ref` — o hash é
do conteúdo. Por isso a `ref` da mediana inclui o `captured_at`: corpos iguais
lidos em horários diferentes são **provas diferentes**, e a de ontem não
comprova disponibilidade hoje.

Uma regra só
------------
`calcular` é a mesma função que `scripts/build_card.py` usa para montar o card.
Duas implementações da "mesma" regra divergiriam em silêncio, e a conferência
compararia o número contra outra coisa.

O que isto NÃO é
----------------
A mediana continua **não sendo cotação executável**: ninguém aceita aposta na
mediana de 25 casas. Ela é referência de mercado (candidato A). O retorno
simulado continua usando a cotação comprovada do lado escolhido.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import statistics as st
from pathlib import Path

import config

logger = logging.getLogger(__name__)

INDICE = config.DATA_DIR / "proveniencia" / "medianas.csv"

COLUNAS = ("ref", "chamada_ref", "captured_at", "fonte", "mercado",
           "evento_id", "participante_a", "participante_b", "odd_a", "odd_b",
           "n_casas", "casas", "casas_sha256", "fonte_atualizada_em",
           "inicio_programado", "regra", "precisao", "limite")

MERCADO = "h2h"
PRECISAO = 3
REGRA = ("mediana por lado sobre as casas que precificam OS DOIS lados no "
         "mesmo mercado h2h; arredondada em 3 casas. Mesma funcao usada por "
         "scripts/build_card.py.")

_LIMITE = ("captured_at e quando a resposta chegou ate nos, nao quando a casa "
           "formou o preco. A MEDIANA NAO E COTACAO EXECUTAVEL: e referencia "
           "de mercado, e nenhum retorno e calculado sobre ela.")


def _texto(v) -> str:
    s = "" if v is None else str(v).strip()
    return "" if s.lower() in ("nan", "none") else s


def calcular(evento: dict, nome_a: str, nome_b: str
             ) -> tuple[float | None, float | None, list[str]]:
    """
    (mediana do lado A, do lado B, casas incluídas).

    **A regra do projeto, em um lugar só.** Só entram casas que precificam os
    DOIS lados no mesmo mercado `h2h`: uma casa que lista só um lado
    desequilibraria o par e a devigagem sairia de duas amostras diferentes.
    """
    pa, pb, casas = [], [], []
    for bm in (evento or {}).get("bookmakers", []) or []:
        if not isinstance(bm, dict):
            continue
        for mk in bm.get("markets", []) or []:
            if not isinstance(mk, dict) or mk.get("key") != MERCADO:
                continue
            d = {o["name"]: o["price"] for o in (mk.get("outcomes") or [])
                 if isinstance(o, dict) and "name" in o and "price" in o}
            if nome_a in d and nome_b in d:
                pa.append(d[nome_a])
                pb.append(d[nome_b])
                casas.append(_texto(bm.get("key") or bm.get("title")))
    if not pa:
        return None, None, []
    return (round(st.median(pa), PRECISAO), round(st.median(pb), PRECISAO),
            sorted(casas))


def _hash_das_casas(casas) -> str:
    return hashlib.sha256("|".join(sorted(casas)).encode("utf-8")).hexdigest()[:16]


def registrar(*, chamada_ref: str, captured_at: str, evento_id: str,
              participante_a: str, participante_b: str, odd_a: float,
              odd_b: float, casas, inicio_programado: str = "",
              fonte_atualizada_em: str = "", fonte: str = "the_odds_api",
              arquivo: Path | None = None) -> str:
    """
    Registra UMA referência de mercado e devolve a `ref`. Só anexa.

    `captured_at` entra na semente da `ref` de propósito: duas capturas com
    corpo idêntico têm a mesma `chamada_ref`, e sem o horário na semente elas
    virariam a mesma prova.
    """
    caminho = Path(arquivo or INDICE)
    casas = sorted(_texto(c) for c in (casas or []) if _texto(c))
    semente = "|".join([chamada_ref or "", captured_at or "", evento_id or "",
                        participante_a, participante_b, f"{odd_a}", f"{odd_b}",
                        _hash_das_casas(casas)])
    ref = hashlib.sha256(semente.encode("utf-8")).hexdigest()[:16]
    try:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        novo = not caminho.exists()
        with caminho.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUNAS)
            if novo:
                w.writeheader()
            w.writerow({
                "ref": ref, "chamada_ref": chamada_ref or "",
                "captured_at": captured_at or "", "fonte": fonte,
                "mercado": MERCADO, "evento_id": evento_id or "",
                "participante_a": participante_a,
                "participante_b": participante_b,
                "odd_a": odd_a, "odd_b": odd_b, "n_casas": len(casas),
                "casas": ",".join(casas), "casas_sha256": _hash_das_casas(casas),
                "fonte_atualizada_em": fonte_atualizada_em,
                "inicio_programado": inicio_programado,
                "regra": REGRA, "precisao": PRECISAO, "limite": _LIMITE})
    except OSError as exc:
        logger.warning("Nao consegui gravar a proveniencia da mediana (%s) -- "
                       "esta referencia de mercado fica sem prova.", exc)
        return ""
    return ref


def linha(ref, arquivo: Path | None = None) -> dict | None:
    caminho = Path(arquivo or INDICE)
    r = _texto(ref)
    if not r or not caminho.exists():
        return None
    achadas = [x for x in csv.DictReader(caminho.open(encoding="utf-8"))
               if x["ref"] == r]
    return achadas[-1] if achadas else None


def _corpo(chamada_ref: str):
    """O corpo persistido daquela chamada, já com o hash conferido."""
    from src.proveniencia import INDICE as INDICE_CHAMADAS
    from src.proveniencia import conferir
    ok, motivo = conferir(chamada_ref)
    if not ok:
        return None, motivo
    reg = None
    for x in csv.DictReader(INDICE_CHAMADAS.open(encoding="utf-8")):
        if x["ref"] == chamada_ref:
            reg = x
    if reg is None or not reg.get("corpo_ref"):
        return None, "corpo nao guardado"
    caminho = Path(reg["corpo_ref"])
    if not caminho.is_absolute():
        caminho = config.PROJECT_ROOT / caminho
    try:
        with gzip.open(caminho, "rt", encoding="utf-8") as f:
            return json.load(f), ""
    except (OSError, ValueError) as exc:
        return None, f"corpo ilegivel ({exc})"


def recomputar(ref, arquivo: Path | None = None) -> tuple[dict | None, str]:
    """
    Reabre a captura referenciada, confere o hash e **recalcula** as duas
    medianas com a mesma regra e precisão.

    Devolve (`{odd_a, odd_b, casas, n_casas}`, "") ou (None, motivo).
    """
    reg = linha(ref, arquivo)
    if reg is None:
        return None, f"referencia de mercado {ref} nao esta no indice"
    corpo, motivo = _corpo(_texto(reg.get("chamada_ref")))
    if corpo is None:
        return None, f"captura referenciada indisponivel: {motivo}"
    alvo = _texto(reg.get("evento_id"))
    evento = next((e for e in corpo if isinstance(e, dict)
                   and _texto(e.get("id")) == alvo), None)
    if evento is None:
        return None, (f"o evento {alvo} nao esta na captura referenciada: a "
                      "prova e de outro confronto")
    oa, ob, casas = calcular(evento, _texto(reg.get("participante_a")),
                             _texto(reg.get("participante_b")))
    if oa is None:
        return None, ("a captura nao tem casa precificando os dois lados "
                      "deste confronto")
    return {"odd_a": oa, "odd_b": ob, "casas": casas,
            "n_casas": len(casas)}, ""


def evidencia(ref, *, participantes=None, odd_a=None, odd_b=None,
              inicio: str = "", decidido_em=None, arquivo: Path | None = None):
    """
    `temporalidade.Evidencia` da referência de mercado.

    Confere, em ordem: a linha existe; o confronto é este; a captura abre e o
    hash bate; as medianas **recalculadas** batem com as usadas na decisão; as
    casas são as mesmas; e a ordem temporal — captura anterior à decisão,
    dentro da validade, e anterior ao início programado daquela luta.
    """
    from src import temporalidade as t
    from src.proveniencia import VALIDADE_HORAS

    r = _texto(ref)
    if not r:
        return t.Evidencia(t.NAO_CERTIFICADA, detalhe=(
            "a referencia de mercado nao esta gravada: " + t.MOTIVO_MEDIANA))
    reg = linha(r, arquivo)
    if reg is None:
        return t.Evidencia(t.DESCONHECIDA, referencia=r, detalhe=(
            f"referencia de mercado {r} nao esta no indice"))
    lido = _texto(reg.get("captured_at"))
    inicio_prova = _texto(reg.get("inicio_programado"))
    inicio = _texto(inicio) or inicio_prova

    if participantes:
        par_prova = {_texto(reg.get("participante_a")).casefold(),
                     _texto(reg.get("participante_b")).casefold()}
        from src.proveniencia import _mesmo_par
        if not _mesmo_par(list(participantes),
                          (reg.get("participante_a"), reg.get("participante_b")),
                          _texto(reg.get("fonte"))) and par_prova != {
                _texto(p).casefold() for p in participantes}:
            return t.Evidencia(t.INCOMPATIVEL, instante=lido, referencia=r,
                               inicio=inicio, detalhe=(
                                   f"a prova e do confronto "
                                   f"{reg.get('participante_a')} x "
                                   f"{reg.get('participante_b')}, que nao e este"))

    recalc, motivo = recomputar(r, arquivo)
    if recalc is None:
        return t.Evidencia(t.DESCONHECIDA, instante=lido, referencia=r,
                           inicio=inicio, detalhe=motivo)
    for nome, gravada, refeita in (("odd_a", reg.get("odd_a"), recalc["odd_a"]),
                                   ("odd_b", reg.get("odd_b"), recalc["odd_b"])):
        try:
            if abs(float(gravada) - float(refeita)) > 1e-9:
                return t.Evidencia(t.INCOMPATIVEL, instante=lido, referencia=r,
                                   inicio=inicio, detalhe=(
                                       f"{nome} recalculada da captura "
                                       f"({refeita}) diverge da registrada "
                                       f"({gravada})"))
        except (TypeError, ValueError):
            return t.Evidencia(t.DESCONHECIDA, instante=lido, referencia=r,
                               inicio=inicio, detalhe=f"{nome} ilegivel")
    if _hash_das_casas(recalc["casas"]) != _texto(reg.get("casas_sha256")):
        return t.Evidencia(t.INCOMPATIVEL, instante=lido, referencia=r,
                           inicio=inicio, detalhe=(
                               "as casas da captura nao sao as registradas"))

    # as odds EFETIVAMENTE usadas na decisao
    for nome, usada, provada in (("odd_a", odd_a, recalc["odd_a"]),
                                 ("odd_b", odd_b, recalc["odd_b"])):
        if usada is None:
            continue
        if abs(float(usada) - float(provada)) > 1e-9:
            return t.Evidencia(t.INCOMPATIVEL, instante=lido, referencia=r,
                               inicio=inicio, detalhe=(
                                   f"a decisao usou {nome}={usada} e a captura "
                                   f"comprova {provada}"))

    dl = t._instante(lido)
    di = t._instante(inicio)
    if dl is None:
        return t.Evidencia(t.DESCONHECIDA, instante=lido, referencia=r,
                           inicio=inicio, detalhe=(
                               "captured_at ausente, ilegivel ou sem fuso"))
    if di is None:
        return t.Evidencia(t.DESCONHECIDA, instante=lido, referencia=r,
                           detalhe=("sem inicio programado verificavel para "
                                    "comparar"))
    if inicio_prova and t._instante(inicio_prova) != di:
        return t.Evidencia(t.INCOMPATIVEL, instante=lido, referencia=r,
                           inicio=inicio, detalhe=(
                               "inicio programado divergente entre as provas"))
    if dl >= di:
        return t.Evidencia(t.INCOMPATIVEL, instante=lido, referencia=r,
                           inicio=inicio, detalhe=(
                               "a captura foi LIDA depois do inicio programado"))
    if decidido_em is not None:
        dd = t._instante(decidido_em)
        if dd is None:
            return t.Evidencia(t.DESCONHECIDA, instante=lido, referencia=r,
                               inicio=inicio,
                               detalhe="horario da decisao ilegivel ou sem fuso")
        if dl > dd:
            return t.Evidencia(t.INCOMPATIVEL, instante=lido, referencia=r,
                               inicio=inicio, detalhe=(
                                   "a captura e POSTERIOR a decisao: nao "
                                   "existia no instante em que se decidiu"))
        if dd >= di:
            return t.Evidencia(t.INCOMPATIVEL, instante=lido, referencia=r,
                               inicio=inicio,
                               detalhe="decisao apos o inicio programado")
        if (dd - dl).total_seconds() / 3600 >= VALIDADE_HORAS:
            return t.Evidencia(t.INCOMPATIVEL, instante=lido, referencia=r,
                               inicio=inicio, detalhe=(
                                   "captura vencida no instante da decisao"))
    return t.Evidencia(t.COMPROVADA, instante=lido, referencia=r, inicio=inicio,
                       detalhe=(f"mediana de {recalc['n_casas']} casa(s) "
                                "recalculada da captura referenciada, hash "
                                "conferido, anterior a decisao e ao inicio"))


def casar_e_registrar(fighter_a: str, fighter_b: str, eventos: list,
                      *, event_date: str, chamada_ref: str,
                      captured_at: str = "", arquivo: Path | None = None
                      ) -> tuple[str, dict | None, str]:
    """
    (`ref`, `{odd_a, odd_b, n_casas, casas, evento_id, inicio}`, motivo).

    Usa `casamento.casar`, o casamento ESTRITO: candidato único na janela, os
    dois participantes como lados de um `h2h` do mesmo evento, evidência forte
    nos dois lados e nenhum homônimo sem desambiguação. Revanche, confronto
    errado e correspondência ambígua bloqueiam aqui — a prova não pode ser de
    outra luta.
    """
    from src.casamento import casar
    from src.line_shopping import _atualizada_em

    c = casar(fighter_a, fighter_b, eventos, event_date=event_date)
    if not c.ok:
        return "", None, f"casamento bloqueado: {c.motivo}"
    oa, ob, casas = calcular(c.evento, c.nome_a, c.nome_b)
    if oa is None:
        return "", None, ("nenhuma casa precifica os dois lados no h2h deste "
                          "confronto")
    atualizada = ""
    for bm in (c.evento or {}).get("bookmakers", []) or []:
        if _texto(bm.get("key")) in casas or _texto(bm.get("title")) in casas:
            atualizada = _texto(bm.get("last_update"))
            break
    if not _texto(captured_at):
        from src.proveniencia import _linha_de_chamada
        reg = _linha_de_chamada(_texto(chamada_ref))
        captured_at = _texto((reg or {}).get("captured_at"))
    ref = registrar(chamada_ref=chamada_ref, captured_at=captured_at,
                    evento_id=c.evento_id or "", participante_a=c.nome_a,
                    participante_b=c.nome_b, odd_a=oa, odd_b=ob, casas=casas,
                    inicio_programado=c.commence_time or "",
                    fonte_atualizada_em=atualizada, arquivo=arquivo)
    return ref, {"odd_a": oa, "odd_b": ob, "n_casas": len(casas),
                 "casas": casas, "evento_id": c.evento_id,
                 "inicio": c.commence_time or "",
                 "nome_a": c.nome_a, "nome_b": c.nome_b}, ""
