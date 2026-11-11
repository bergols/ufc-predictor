"""
src/sombra.py

Modo sombra: registra o que o candidato congelado decidiria, **sem apostar e
sem tocar a produção**.

O que mudou no v2 (ver ERRATA_VANTAGEM_V1.md)
---------------------------------------------
**1. "Prospectivo" passou a exigir prova positiva.** O v1 classificava como
avaliação prospectiva todo evento fora da lista de desenvolvimento. Isso é
ausência de contra-evidência: um evento pode estar fora da lista porque ninguém
o olhou, ou porque o candidato nem existia quando a decisão foi tomada. Agora a
qualificação exige, cada item verificado:

    candidato congelado ANTES da decisão        (candidatos.congelar)
    decisão antes do início PROGRAMADO da luta  (não da data do evento)
    previsão com temporalidade COMPROVADA       (temporalidade.da_previsao)
    preço executável disponível no instante     (temporalidade.do_preco)
    referência de mercado certificada           (hoje: impossível)
    evento nunca usado em estimação nem em inferência

Evento histórico já examinado — incluindo os oito da avaliação do v1 — **nunca**
pode ser prospectivo, qualquer que seja o resto.

Como a mediana de casas não tem proveniência por preço, a classe máxima
alcançável hoje é `prospectiva_parcial`: tudo comprovado **menos** a referência
de mercado. Isso não é "prospectiva plenamente auditável", e o rótulo diz isso.

**2. A seleção deixou de depender do modelo quando `w = 0`.** O v1 apostava em
`model_side`, o lado que o MODELO aponta, mesmo com o peso do modelo em zero —
então a probabilidade era a do mercado e a escolha do lado continuava do
modelo. Agora a política avalia **os dois lados** com `p_C` e a odd executável
de cada um, e escolhe o de maior lucro esperado líquido positivo. Com `w = 0`,
mudar `model_side` não muda nada.

Append-only, e por quê
----------------------
`sombra_previsoes.csv` só recebe linhas. A liquidação entra em
`sombra_resultados.csv`, aponta para a previsão pelo `registro_id` e **não a
reescreve**. Reescrever a previsão depois do resultado produziria uma linha que
não descreve instante nenhum.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import config
from src.temporalidade import _instante

logger = logging.getLogger(__name__)

PASTA = config.DATA_DIR / "experimentos" / "sombra"
REGISTROS = PASTA / "sombra_previsoes.csv"
RESULTADOS = PASTA / "sombra_resultados.csv"

COLUNAS = (
    "registro_id", "decidido_em", "versao_protocolo", "candidato_id",
    "candidato_congelado_em", "protocolo_sha256", "peso_w",
    "classe", "motivos_da_classe",
    "event_date", "event_name", "fighter_a", "fighter_b",
    "prediction_id", "model_name", "manifesto_sha256", "modelos_sha256",
    "inicio_programado", "origem_do_inicio",
    "temporalidade_previsao", "evidencia_previsao", "instante_previsao",
    "ref_mediana", "temporalidade_mediana", "evidencia_mediana",
    "mediana_lida_em",
    "p_mercado_a", "p_modelo_a", "p_combinacao_a", "model_side",
    # os DOIS lados, cada um com a prova dele
    "odd_mediana_a", "odd_mediana_b",
    "odd_executavel_a", "ref_preco_a", "temporalidade_preco_a",
    "evidencia_preco_a", "lucro_liquido_a",
    "odd_executavel_b", "ref_preco_b", "temporalidade_preco_b",
    "evidencia_preco_b", "lucro_liquido_b",
    "lado_selecionado", "prob_lado_selecionado", "odd_lado_selecionado",
    "retorno_bruto_esperado", "lucro_liquido_esperado",
    "casa", "mercado", "preco_lido_em",
    "selecao", "motivo", "limite",
)

COLUNAS_RESULTADO = ("liquidado_em", "registro_id", "candidato_id", "classe",
                     "event_date", "fighter_a", "fighter_b", "lado_selecionado",
                     "vencedor", "acertou", "unidades", "retorno_unidades",
                     "fonte_do_resultado", "limite")

_LIMITE = (
    "REGISTRO DE PESQUISA, nao de aposta, e nenhum dinheiro envolvido. "
    "retorno_bruto_esperado e p x odd (1,26 = receber 1,26 por 1 apostado); "
    "lucro_liquido_esperado e p x odd - 1 (0,26 = ganhar 0,26 por 1 apostado). "
    "A probabilidade e do candidato C; com w=0 ela E a do mercado devigada, "
    "entao o valor esperado e auto-referente e nao e evidencia de valor. "
    "selecao=True significa que a politica pre-registrada entraria, nao que a "
    "entrada e boa."
)

# POLITICA, fixada no vantagem-v2 ANTES dos novos resultados (secao 5 de
# PROTOCOLO_VANTAGEM_V2.md). Unidade fixa, um lado por luta no maximo, sem
# Kelly, martingale, recuperacao de perda ou otimizacao de stake. Nao e
# configuravel por chamada: politica escolhida na hora e politica escolhida
# olhando o resultado.
UNIDADES_POR_PERNA = 1.0
LIMIAR_LUCRO_LIQUIDO = 0.0   # entra so com lucro esperado liquido POSITIVO

CLASSE_PROSPECTIVA = "avaliacao_prospectiva"
CLASSE_PARCIAL = "prospectiva_parcial"
CLASSE_DESENVOLVIMENTO = "desenvolvimento"
CLASSE_NAO_QUALIFICADA = "nao_qualificada"

MOTIVO_SEM_PRE_REGISTRO = "luta sem previsao congelada: nada a sombrear"
MOTIVO_SEM_MEDIANA = "sem os dois lados da mediana: margem nao retiravel"
MOTIVO_NENHUM_LADO_COM_PROVA = ("nenhum lado tem preco executavel com prova "
                                "valida: a mediana nao e cotacao executavel")
MOTIVO_SEM_LUCRO = ("nenhum lado tem lucro esperado liquido positivo com o "
                    "preco executavel comprovado")
MOTIVO_EMPATE = ("empate exato de lucro esperado liquido entre os dois lados: "
                 "recusa, porque escolher um seria desempatar por critério "
                 "nao registrado")


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _vazio(v) -> bool:
    import pandas as pd
    return (v is None or (isinstance(v, float) and pd.isna(v))
            or not str(v).strip() or str(v).strip().lower() in ("nan", "none"))


def peso_registrado(manifesto: Path | None = None) -> tuple[float, list[str]]:
    """
    (peso `w`, eventos de desenvolvimento) do experimento já executado.

    O peso NÃO é reestimado aqui. Sem o resumo no disco a sombra não roda:
    inventar um peso na hora de decidir é escolher parâmetro olhando a decisão.
    """
    from src import vantagem
    caminho = Path(manifesto or (vantagem.PASTA / f"resumo_{vantagem.VERSAO}.json"))
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho} nao existe. Rode primeiro: "
            f"python -m scripts.experimento_vantagem")
    r = json.loads(caminho.read_text(encoding="utf-8"))
    man = Path(str(caminho).replace("resumo_", "manifesto_"))
    eventos = {"desenvolvimento": [], "avaliacao": []}
    if man.exists():
        eventos.update((json.loads(man.read_text(encoding="utf-8"))
                        .get("eventos_por_fatia", {})))
    # TODOS os eventos que o experimento olhou ficam queimados: os 8 de
    # desenvolvimento estimaram o peso e os 8 de avaliacao gastaram a
    # inferencia. Nenhum dos 16 pode voltar como prospectivo.
    queimados = sorted({str(e) for e in eventos.get("desenvolvimento", [])}
                       | {str(e) for e in eventos.get("avaliacao", [])})
    return float(r["peso_modelo_w"]), queimados


# ---------------------------------------------------------------------------
# qualificacao: prova POSITIVA, nunca ausencia de contra-evidencia
# ---------------------------------------------------------------------------

def qualificar(*, candidato: dict, decidido_em: str, inicio: str,
               ev_previsao, ev_mediana, ev_preco_lado,
               event_date: str, queimados: set[str],
               manifesto_previsao: str | None = None) -> tuple[str, list[str]]:
    """
    (classe, motivos). Cada exigência é verificada e o que falhou aparece.

    Nenhuma das exigências é satisfeita por omissão: horário desconhecido é
    falha, candidato sem instante de congelamento é falha, e evento já
    examinado é falha definitiva.
    """
    from src.candidatos import sha256_arquivo

    motivos = []

    if str(event_date) in queimados:
        return CLASSE_DESENVOLVIMENTO, [
            "o evento ja foi usado em estimacao ou em analise inferencial: "
            "nunca pode valer como prospectivo"]

    congelado = _instante((candidato or {}).get("congelado_em"))
    decidido = _instante(decidido_em)
    comeca = _instante(inicio)

    if candidato is None or not candidato.get("candidato_id"):
        motivos.append("nao ha candidato congelado: o parametro usado na "
                       "decisao nao estava fixado antes dela")
    elif congelado is None:
        motivos.append("o candidato nao registra instante de congelamento")
    elif decidido is None:
        motivos.append("instante da decisao ilegivel")
    elif not candidato.get("protocolo_sha256"):
        motivos.append("o candidato nao registra o hash do protocolo")
    elif candidato["protocolo_sha256"] != sha256_arquivo(
        config.PROJECT_ROOT / Path(candidato.get("protocolo_arquivo", "")).name
    ):
        motivos.append("o protocolo disponivel nao confere com o hash do candidato")
    elif congelado >= decidido:
        motivos.append("o candidato foi congelado DEPOIS da decisao: o "
                       "parametro pode ter sido escolhido olhando o caso")

    # O peso combina previsoes de UM modelo. Previsao de outro modelo (outro
    # manifesto de treino) nao pode ser atribuida ao candidato.
    if candidato and candidato.get("candidato_id"):
        do_candidato = str(candidato.get("manifesto_modelo_sha256") or "")
        if not do_candidato:
            motivos.append("o candidato nao registra o manifesto do modelo: "
                           "nao da para dizer que a previsao e do modelo com "
                           "que o peso foi fixado")
        elif str(manifesto_previsao or "") != do_candidato:
            motivos.append(
                f"previsao de outro modelo (manifesto "
                f"{str(manifesto_previsao or '?')[:12]}) que o do candidato "
                f"({do_candidato[:12]})")

    if comeca is None:
        motivos.append("sem inicio programado verificavel daquela luta: nao da "
                       "para dizer que a decisao foi anterior a ela")
    elif decidido is not None and decidido >= comeca:
        motivos.append("a decisao foi registrada DEPOIS do inicio programado "
                       "daquela luta")

    if ev_previsao.classe != "comprovada":
        motivos.append(f"temporalidade da previsao: {ev_previsao.classe}")
    else:
        criado = _instante(ev_previsao.instante)
        if criado is None or decidido is None or criado > decidido:
            motivos.append("a previsao nao estava disponivel no instante da decisao")
    if ev_preco_lado is None or ev_preco_lado.classe != "comprovada":
        classe = "ausente" if ev_preco_lado is None else ev_preco_lado.classe
        motivos.append(f"temporalidade do preco executavel: {classe}")

    if ev_mediana.classe not in ("nao_certificada", "comprovada"):
        motivos.append(ev_mediana.detalhe)

    if motivos:
        return CLASSE_NAO_QUALIFICADA, motivos
    if ev_mediana.classe == "nao_certificada":
        return CLASSE_PARCIAL, [
            ev_mediana.detalhe + " A comparacao com o mercado NAO e plenamente auditavel"]
    return CLASSE_PROSPECTIVA, []


# ---------------------------------------------------------------------------
# a politica de selecao
# ---------------------------------------------------------------------------

def _avaliar_lado(*, nome, prob, odd, ref, inicio, participantes, decidido_em):
    """
    (lucro líquido esperado, evidência, disponível) de UM lado.

    Lado sem preço comprovado fica **indisponível** — nunca recebe o preço do
    adversário, que vem da mesma chamada e do mesmo corpo com o lado trocado.
    """
    from src import temporalidade
    if _vazio(odd) or _vazio(ref):
        falta = temporalidade.Evidencia(
            temporalidade.AUSENTE,
            detalhe=("sem odd executavel deste lado" if _vazio(odd)
                     else "sem referencia de cotacao deste lado"))
        return None, falta, False
    ev = temporalidade.do_preco(ref, inicio=inicio, lado=nome, odd=float(odd),
                                participantes=participantes,
                                casa="Thunderpick", mercado="h2h", decidido_em=decidido_em)
    if ev.classe != temporalidade.COMPROVADA:
        return None, ev, False
    return float(prob) * float(odd) - 1.0, ev, True


def politica(*, p_c_a: float, a: str, b: str, odd_a, odd_b, ref_a, ref_b,
             inicio: str, decidido_em: str | None = None) -> dict:
    """
    A política do `vantagem-v2`, fixada antes dos novos resultados.

    1. lucro esperado líquido dos **dois** lados: `p_C(lado) × odd(lado) − 1`;
    2. cada lado precisa da **sua própria** prova válida;
    3. no máximo **um** lado por luta: o de maior lucro líquido **positivo**;
    4. empate exato → recusa com motivo;
    5. lado sem preço comprovado fica indisponível;
    6. unidade fixa, nenhuma aposta real.

    `model_side` **não entra**. Era por ali que a seleção continuava dependendo
    do modelo mesmo com `w = 0`.
    """
    participantes = (a, b)
    lucro_a, ev_a, ok_a = _avaliar_lado(
        nome=a, prob=p_c_a, odd=odd_a, ref=ref_a, inicio=inicio,
        participantes=participantes, decidido_em=decidido_em)
    lucro_b, ev_b, ok_b = _avaliar_lado(
        nome=b, prob=1.0 - p_c_a, odd=odd_b, ref=ref_b, inicio=inicio,
        participantes=participantes, decidido_em=decidido_em)

    out = {"lucro_liquido_a": lucro_a, "lucro_liquido_b": lucro_b,
           "evidencia_a": ev_a, "evidencia_b": ev_b,
           "lado": "", "prob": None, "odd": None,
           "selecao": False, "motivo": ""}

    if not ok_a and not ok_b:
        out["motivo"] = MOTIVO_NENHUM_LADO_COM_PROVA
        return out

    positivos = [(l, nome, prob, odd) for l, nome, prob, odd, ok in (
        (lucro_a, a, p_c_a, odd_a, ok_a),
        (lucro_b, b, 1.0 - p_c_a, odd_b, ok_b))
        if ok and l is not None and l > LIMIAR_LUCRO_LIQUIDO]
    if not positivos:
        out["motivo"] = MOTIVO_SEM_LUCRO
        return out
    if len(positivos) == 2 and positivos[0][0] == positivos[1][0]:
        out["motivo"] = MOTIVO_EMPATE
        return out

    lucro, nome, prob, odd = max(positivos, key=lambda t: t[0])
    out.update({"lado": nome, "prob": float(prob), "odd": float(odd),
                "selecao": True, "motivo": ""})
    return out


# ---------------------------------------------------------------------------
# registro
# ---------------------------------------------------------------------------

def registrar(linhas: list[dict], arquivo: Path | None = None) -> list[str]:
    """Anexa linhas de sombra e devolve os `registro_id`. Só anexa."""
    if not linhas:
        return []
    caminho = Path(arquivo or REGISTROS)
    agora = _agora()
    ids = []
    try:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        novo = not caminho.exists()
        with caminho.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUNAS)
            if novo:
                w.writeheader()
            for d in linhas:
                linha = {c: "" for c in COLUNAS}
                linha.update({k: ("" if _vazio(v) else v)
                              for k, v in d.items() if k in COLUNAS})
                linha["decidido_em"] = d.get("decidido_em", agora)
                semente = "|".join(str(linha[c]) for c in (
                    "event_date", "fighter_a", "fighter_b", "candidato_id",
                    "lado_selecionado", "ref_preco_a", "ref_preco_b",
                    "decidido_em"))
                linha["registro_id"] = hashlib.sha256(
                    semente.encode("utf-8")).hexdigest()[:16]
                linha["limite"] = _LIMITE
                w.writerow(linha)
                ids.append(linha["registro_id"])
    except OSError as exc:
        logger.warning("Nao consegui gravar a sombra (%s).", exc)
        return []
    logger.info("%d linha(s) de sombra em %s", len(ids), caminho)
    return ids


def montar(event_date: str, card_csv: Path | None = None,
           history_csv: Path | None = None, *, candidato: dict | None = None,
           queimados: set[str] | None = None, agora=None) -> list[dict]:
    """
    Monta as linhas de sombra de um evento, sem escrever nada.

    Quatro coisas separadas, e cada uma com o seu horário e a sua prova:

        (a) previsão do modelo        versão imutável, `criado_em`
        (b) referência do mercado     mediana devigada, SEM proveniência
        (c) preço executável          casa, por lado, com `captured_at`
        (d) instante da decisão       agora

    A previsão pode ser anterior ao preço — é o caso normal: o card é montado
    antes e o preço é relido depois. Isso fica **documentado** por linha, e a
    política exige que o preço esteja comprovadamente anterior ao início da
    luta, não anterior à previsão.
    """
    import pandas as pd

    from src import candidatos as mod_cand
    from src import mediana as mod_mediana
    from src import temporalidade, vantagem
    from src.prediction_history import _load_raw
    from src.proveniencia import _linha_de_cotacao
    from src.snapshots import arquivo_de, prob_do_lado, resolver

    if candidato is None:
        candidato = mod_cand.atual()
    if queimados is None:
        queimados = set(mod_cand.eventos_queimados(candidato or {}))
        for anterior in mod_cand.carregar():
            queimados.update(mod_cand.eventos_queimados(anterior))
        try:
            queimados.update(peso_registrado()[1])
        except FileNotFoundError:
            pass
    peso = float((candidato or {}).get("peso_w", 0.0))
    decidido_em = agora.isoformat(timespec="seconds") if hasattr(agora, "isoformat") \
        else (agora or _agora())

    card = pd.read_csv(Path(card_csv or config.UPCOMING_CARD_CSV))
    hist = _load_raw(Path(history_csv or config.PREDICTION_HISTORY_CSV))
    caminho_snapshots = arquivo_de(Path(history_csv or config.PREDICTION_HISTORY_CSV))
    congelado = {}
    if not hist.empty:
        for _, r in hist[hist["event_date"] == event_date].iterrows():
            congelado[frozenset((str(r["fighter_a"]).strip(),
                                 str(r["fighter_b"]).strip()))] = r

    out = []
    for _, c in card.iterrows():
        a, b = str(c["fighter_a"]).strip(), str(c["fighter_b"]).strip()
        base = {
            "decidido_em": decidido_em,
            "versao_protocolo": (candidato or {}).get("versao_protocolo",
                                                      vantagem.VERSAO),
            "candidato_id": (candidato or {}).get("candidato_id", ""),
            "candidato_congelado_em": (candidato or {}).get("congelado_em", ""),
            "protocolo_sha256": (candidato or {}).get("protocolo_sha256", ""),
            "peso_w": peso, "event_date": event_date,
            "fighter_a": a, "fighter_b": b,
            # forma ESTAVEL: toda linha tem as mesmas chaves, com ou sem
            # selecao. Chave que aparece so as vezes faz quem le a saida
            # tratar ausencia de selecao como ausencia de dado.
            "lado_selecionado": "", "selecao": False, "motivo": "",
            "classe": CLASSE_NAO_QUALIFICADA, "motivos_da_classe": "",
        }

        pre = congelado.get(frozenset((a, b)))
        if pre is None:
            out.append({**base, "classe": CLASSE_NAO_QUALIFICADA,
                        "motivos_da_classe": MOTIVO_SEM_PRE_REGISTRO,
                        "selecao": False, "motivo": MOTIVO_SEM_PRE_REGISTRO})
            continue
        base["event_name"] = pre.get("event_name")

        pid = "" if pd.isna(pre.get("prediction_id")) else str(
            pre.get("prediction_id")).strip()
        ref_a = c.get("ref_tp_a", c.get("ref_tp"))
        ref_b = c.get("ref_tp_b")
        ref_med = c.get("ref_mediana")
        inicio, origem = temporalidade.inicio_programado(
            ref_a, ref_b, pre.get("sharp_ref"), pre.get("close_ref"))
        if not inicio and not _vazio(ref_med):
            # a propria prova da mediana carrega o inicio programado
            linha_med = mod_mediana.linha(ref_med)
            if linha_med and str(linha_med.get("inicio_programado") or "").strip():
                inicio = str(linha_med["inicio_programado"]).strip()
                origem = f"medianas.csv:{str(ref_med).strip()}"
        ev_prev = temporalidade.da_previsao(pid, inicio=inicio, arquivo=caminho_snapshots)
        base.update({
            "prediction_id": pid, "inicio_programado": inicio,
            "origem_do_inicio": origem,
            "temporalidade_previsao": ev_prev.classe,
            "evidencia_previsao": ev_prev.detalhe,
            "instante_previsao": ev_prev.instante,
        })

        versao, porque = resolver(pid, event_date=event_date,
                                  fighter_a=a, fighter_b=b, arquivo=caminho_snapshots)
        if versao is None:
            motivo = f"versao da previsao nao verificavel: {porque}"
            out.append({**base, "classe": CLASSE_NAO_QUALIFICADA,
                        "motivos_da_classe": motivo, "selecao": False,
                        "motivo": motivo})
            continue
        base.update({"model_name": versao.get("model_name"),
                     "manifesto_sha256": versao.get("manifesto_sha256"),
                     "modelos_sha256": versao.get("modelos_sha256"),
                     "model_side": versao.get("model_side")})

        if pd.isna(c.get("odds_a_decimal")) or pd.isna(c.get("odds_b_decimal")):
            out.append({**base, "classe": CLASSE_NAO_QUALIFICADA,
                        "motivos_da_classe": MOTIVO_SEM_MEDIANA,
                        "selecao": False, "motivo": MOTIVO_SEM_MEDIANA})
            continue

        # A REFERENCIA DE MERCADO, conferida contra a captura original.
        # Reabre o corpo, valida o hash, RECALCULA as duas medianas com a
        # mesma regra e precisao da montagem do card, e compara com as odds
        # que de fato entraram nesta decisao. Sem prova, continua
        # `nao_certificada` e a classe maxima fica parcial.
        ev_med = mod_mediana.evidencia(
            ref_med, participantes=(a, b), inicio=inicio,
            odd_a=float(c["odds_a_decimal"]), odd_b=float(c["odds_b_decimal"]),
            decidido_em=decidido_em, event_date=event_date)
        base.update({"ref_mediana": "" if _vazio(ref_med) else str(ref_med).strip(),
                     "temporalidade_mediana": ev_med.classe,
                     "evidencia_mediana": ev_med.detalhe,
                     "mediana_lida_em": ev_med.instante})

        p_merc_a = vantagem.devig(c["odds_a_decimal"], c["odds_b_decimal"])
        p_mod_a = prob_do_lado(versao, a)
        if p_mod_a is None:
            motivo = "a versao nao cobre os participantes do card"
            out.append({**base, "classe": CLASSE_NAO_QUALIFICADA,
                        "motivos_da_classe": motivo, "selecao": False,
                        "motivo": motivo})
            continue
        p_c_a = float(vantagem.combinar([p_merc_a], [p_mod_a], peso)[0])
        base.update({
            "p_mercado_a": round(p_merc_a, 4), "p_modelo_a": round(p_mod_a, 4),
            "p_combinacao_a": round(p_c_a, 4),
            "odd_mediana_a": float(c["odds_a_decimal"]),
            "odd_mediana_b": float(c["odds_b_decimal"]),
        })

        dec = politica(p_c_a=p_c_a, a=a, b=b, odd_a=c.get("odds_a_tp"),
                       odd_b=c.get("odds_b_tp"), ref_a=ref_a, ref_b=ref_b,
                       inicio=inicio, decidido_em=decidido_em)
        base.update({
            "odd_executavel_a": c.get("odds_a_tp"),
            "ref_preco_a": "" if _vazio(ref_a) else str(ref_a),
            "temporalidade_preco_a": dec["evidencia_a"].classe,
            "evidencia_preco_a": dec["evidencia_a"].detalhe,
            "lucro_liquido_a": None if dec["lucro_liquido_a"] is None
                               else round(dec["lucro_liquido_a"], 4),
            "odd_executavel_b": c.get("odds_b_tp"),
            "ref_preco_b": "" if _vazio(ref_b) else str(ref_b),
            "temporalidade_preco_b": dec["evidencia_b"].classe,
            "evidencia_preco_b": dec["evidencia_b"].detalhe,
            "lucro_liquido_b": None if dec["lucro_liquido_b"] is None
                               else round(dec["lucro_liquido_b"], 4),
        })

        # Qual prova de preco a qualificacao confere: a do lado SELECIONADO
        # quando houve selecao. Sem selecao a observacao ainda existe -- a
        # recusa e um dado -- e o que se confere e a melhor prova disponivel,
        # porque dizer "preco ausente" numa luta cujo preco esta provado
        # descreveria errado o que foi encontrado.
        if dec["lado"] == a:
            ev_preco_escolhido = dec["evidencia_a"]
        elif dec["lado"] == b:
            ev_preco_escolhido = dec["evidencia_b"]
        else:
            ev_preco_escolhido = next(
                (e for e in (dec["evidencia_a"], dec["evidencia_b"])
                 if e.classe == temporalidade.COMPROVADA), dec["evidencia_a"])
        classe, motivos = qualificar(
            candidato=candidato, decidido_em=decidido_em, inicio=inicio,
            ev_previsao=ev_prev, ev_mediana=ev_med,
            ev_preco_lado=ev_preco_escolhido, event_date=event_date,
            queimados=queimados,
            manifesto_previsao=versao.get("manifesto_sha256"))
        base.update({"classe": classe,
                     "motivos_da_classe": " | ".join(motivos)})

        if dec["selecao"] and classe in (CLASSE_PARCIAL, CLASSE_PROSPECTIVA):
            cot = _linha_de_cotacao(str(
                dec["evidencia_a"].referencia if dec["lado"] == a
                else dec["evidencia_b"].referencia))
            bruto = dec["prob"] * dec["odd"]
            base.update({
                "lado_selecionado": dec["lado"],
                "prob_lado_selecionado": round(dec["prob"], 4),
                "odd_lado_selecionado": round(dec["odd"], 4),
                "retorno_bruto_esperado": round(bruto, 4),
                "lucro_liquido_esperado": round(bruto - 1.0, 4),
                "casa": (cot or {}).get("casa", ""),
                "mercado": (cot or {}).get("mercado", ""),
                "preco_lido_em": (cot or {}).get("captured_at", ""),
                "selecao": True, "motivo": "",
            })
        else:
            base.update({"selecao": False, "motivo": dec["motivo"] or
                         base["motivos_da_classe"]})
        out.append(base)
    return out


def liquidar(event_date: str, history_csv: Path | None = None,
             arquivo: Path | None = None,
             registros: Path | None = None) -> int:
    """
    Anexa o resultado das linhas de sombra daquele evento, **sem reescrever**
    o que foi previsto.

    Captura repetida da mesma luta: cada `registro_id` liquida **uma vez** e
    cada luta conta **uma aposta** — a mais antiga entre as linhas com seleção.
    As outras liquidam com `unidades = 0` e o motivo no campo. Sem isso, três
    capturas do mesmo card contariam três apostas na mesma luta.
    """
    import pandas as pd

    from src.prediction_history import _load_raw

    reg_path = Path(registros or REGISTROS)
    if not reg_path.exists():
        logger.info("Nenhum registro de sombra para liquidar.")
        return 0
    prev = pd.read_csv(reg_path, dtype=str).fillna("")
    do_evento = prev[prev["event_date"] == str(event_date)]
    if do_evento.empty:
        logger.info("Nenhuma linha de sombra de %s.", event_date)
        return 0

    saida = Path(arquivo or RESULTADOS)
    ja = set()
    if saida.exists():
        ja = set(pd.read_csv(saida, dtype=str).fillna("")["registro_id"])

    # UMA aposta por luta e por candidato: a primeira decisao com selecao.
    primeira = {}
    com_selecao = do_evento[
        do_evento["selecao"].isin(("True", "true", "1"))
        & do_evento["classe"].isin((CLASSE_PARCIAL, CLASSE_PROSPECTIVA))]
    for _, r in com_selecao.sort_values("decidido_em").iterrows():
        chave = (frozenset((r["fighter_a"], r["fighter_b"])), r["candidato_id"])
        primeira.setdefault(chave, r["registro_id"])

    hist = _load_raw(Path(history_csv or config.PREDICTION_HISTORY_CSV))
    vencedores = {}
    if not hist.empty:
        for _, r in hist[hist["event_date"] == str(event_date)].iterrows():
            if not pd.isna(r.get("actual_winner")):
                vencedores[frozenset((str(r["fighter_a"]).strip(),
                                      str(r["fighter_b"]).strip()))] = \
                    str(r["actual_winner"]).strip()

    agora = _agora()
    novas = []
    for _, r in do_evento.iterrows():
        if r["registro_id"] in ja:
            continue
        venc = vencedores.get(frozenset((r["fighter_a"], r["fighter_b"])))
        if venc is None:
            continue
        lado = r["lado_selecionado"]
        acertou = bool(lado) and venc == lado
        chave = (frozenset((r["fighter_a"], r["fighter_b"])), r["candidato_id"])
        e_a_que_conta = primeira.get(chave) == r["registro_id"]
        if e_a_que_conta and r["odd_lado_selecionado"]:
            unidades = UNIDADES_POR_PERNA
            retorno = (unidades * (float(r["odd_lado_selecionado"]) - 1.0)
                       if acertou else -unidades)
            limite = ("retorno de pesquisa, unidade fixa, sem dinheiro "
                      "envolvido. Nao e registro de aposta.")
        else:
            unidades, retorno = 0.0, 0.0
            limite = ("sem unidade: recusa, ou captura repetida da mesma luta "
                      "(so a primeira decisao com selecao conta como aposta).")
        ja.add(r["registro_id"])
        novas.append({
            "liquidado_em": agora, "registro_id": r["registro_id"],
            "candidato_id": r["candidato_id"], "classe": r["classe"],
            "event_date": r["event_date"], "fighter_a": r["fighter_a"],
            "fighter_b": r["fighter_b"], "lado_selecionado": lado,
            "vencedor": venc, "acertou": acertou, "unidades": unidades,
            "retorno_unidades": round(retorno, 4),
            "fonte_do_resultado": "data/prediction_history.csv:actual_winner",
            "limite": limite,
        })
    if not novas:
        logger.info("Nada novo a liquidar em %s.", event_date)
        return 0
    try:
        saida.parent.mkdir(parents=True, exist_ok=True)
        novo = not saida.exists()
        with saida.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUNAS_RESULTADO)
            if novo:
                w.writeheader()
            for linha in novas:
                w.writerow(linha)
    except OSError as exc:
        logger.warning("Nao consegui gravar a liquidacao (%s).", exc)
        return 0
    logger.info("%d liquidacao(oes) anexada(s) em %s", len(novas), saida)
    return len(novas)
