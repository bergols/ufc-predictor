"""
src/sombra.py

Modo sombra: registra o que os três candidatos teriam decidido, **sem apostar
e sem tocar a produção**.

Por que existe
--------------
O experimento de `PROTOCOLO_VANTAGEM.md` mediu 16 eventos já ocorridos e deu
negativo: no desenvolvimento o peso do modelo saiu **zero**. Um negativo em 8
eventos de avaliação não encerra a pergunta — ele diz que esta amostra não
sustenta vantagem. Para a pergunta andar é preciso acumular eventos
**prospectivos**, e para isso o registro tem de ser feito ANTES da luta, numa
trilha que ninguém reescreve depois.

Separado da produção de propósito: arquivos em
`data/experimentos/sombra/`, nunca em `data/`. O fluxo de produção
(`prediction_history`, `previsoes_snapshots`, `decisoes`) não é lido para
escrita e não é alterado.

Append-only, e por quê
----------------------
`sombra_previsoes.csv` só recebe linhas. A liquidação entra em
`sombra_resultados.csv`, que também só recebe linhas e **aponta** para a
linha prevista pelo `registro_id`. Reescrever a previsão depois do resultado é
o erro que o projeto inteiro existe para não cometer: produziria uma linha que
não descreve instante nenhum.

Desenvolvimento x avaliação prospectiva
---------------------------------------
Evento que já entrou no ajuste do peso é **desenvolvimento**, não avaliação
prospectiva — e o campo `fatia` diz qual é, por linha. Classificar como
prospectivo um evento que ajudou a escolher o parâmetro seria contar a mesma
informação duas vezes.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import config

logger = logging.getLogger(__name__)

PASTA = config.DATA_DIR / "experimentos" / "sombra"
REGISTROS = PASTA / "sombra_previsoes.csv"
RESULTADOS = PASTA / "sombra_resultados.csv"

COLUNAS = (
    "registro_id", "decidido_em", "versao_protocolo", "peso_w", "fatia",
    "event_date", "event_name", "fighter_a", "fighter_b",
    "prediction_id", "model_name", "manifesto_sha256", "modelos_sha256",
    "p_mercado_a", "p_modelo_a", "p_combinacao_a",
    "lado", "prob_lado_mercado", "prob_lado_modelo", "prob_lado_combinacao",
    "odd_mediana_lado", "odd_executavel_lado", "ref_preco", "casa", "mercado",
    "preco_lido_em", "inicio_programado",
    "retorno_bruto_esperado", "lucro_liquido_esperado",
    "selecao", "motivo", "temporalidade", "limite",
)

COLUNAS_RESULTADO = ("liquidado_em", "registro_id", "event_date", "fighter_a",
                     "fighter_b", "lado", "vencedor", "acertou", "unidades",
                     "retorno_unidades", "fonte_do_resultado", "limite")

_LIMITE = (
    "REGISTRO DE PESQUISA, nao de aposta. retorno_bruto_esperado e p x odd "
    "(1,26 = receber 1,26 por 1 apostado); lucro_liquido_esperado e p x odd - 1. "
    "A probabilidade e do NOSSO modelo ou do mercado, entao o valor esperado e "
    "auto-referente e nao e evidencia de valor. selecao=True significa que a "
    "politica pre-registrada entraria, nao que a entrada e boa."
)

# Politica de simulacao, da secao 8 do protocolo: UNIDADE FIXA, uma unidade por
# perna, entrada quando o retorno bruto esperado passa de 1 com preco
# EXECUTAVEL. Sem Kelly, martingale, recuperacao de perda ou otimizacao de
# stake. Nao e configuravel por chamada de proposito: parametro de politica que
# se escolhe na hora e parametro escolhido olhando o resultado.
UNIDADES_POR_PERNA = 1.0
LIMIAR_RETORNO_BRUTO = 1.0

MOTIVO_SEM_PRE_REGISTRO = "luta sem previsao congelada: nada a sombrear"
MOTIVO_SEM_MEDIANA = "sem os dois lados da mediana: margem nao retiravel"
MOTIVO_SEM_EXECUTAVEL = ("sem preco executavel com prova: a mediana de casas "
                         "nao e cotacao executavel e nao gera retorno")
MOTIVO_EV_BAIXO = "retorno bruto esperado nao passa de 1 com o preco executavel"


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _vazio(v) -> bool:
    import pandas as pd
    return (v is None or (isinstance(v, float) and pd.isna(v))
            or not str(v).strip() or str(v).strip().lower() in ("nan", "none"))


def peso_registrado(manifesto: Path | None = None) -> tuple[float, list[str]]:
    """
    (peso `w`, eventos de desenvolvimento) do experimento já executado.

    O peso NÃO é reestimado aqui: ele vem do resumo gravado, que é o que o
    protocolo mandou estimar só no desenvolvimento. Sem o resumo no disco a
    sombra não roda — inventar um peso agora seria escolher parâmetro na hora
    de decidir.
    """
    from src import vantagem
    caminho = Path(manifesto or (vantagem.PASTA / f"resumo_{vantagem.VERSAO}.json"))
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho} nao existe. Rode primeiro: "
            f"python -m scripts.experimento_vantagem")
    r = json.loads(caminho.read_text(encoding="utf-8"))
    man = Path(str(caminho).replace("resumo_", "manifesto_"))
    eventos_dev = []
    if man.exists():
        eventos_dev = (json.loads(man.read_text(encoding="utf-8"))
                       .get("eventos_por_fatia", {}).get("desenvolvimento", []))
    return float(r["peso_modelo_w"]), [str(e) for e in eventos_dev]


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
                    "event_date", "fighter_a", "fighter_b", "lado",
                    "prediction_id", "ref_preco", "decidido_em"))
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
           history_csv: Path | None = None, *, peso: float | None = None,
           eventos_dev: list[str] | None = None, agora=None) -> list[dict]:
    """
    Monta as linhas de sombra de um evento, sem escrever nada.

    A probabilidade do modelo vem da **versão imutável** apontada pelo
    histórico (`prediction_id`), igual ao caminho de produção: resolver antes
    de calcular. A do mercado vem da mediana devigada do card. O preço
    executável e a prova dele vêm do card e do índice de cotações.

    Luta sem previsão congelada, sem os dois lados da mediana, ou sem preço
    executável com prova entra como **recusa com motivo** — nunca desaparece
    da contagem.
    """
    import pandas as pd

    from src import vantagem
    from src.prediction_history import _load_raw
    from src.proveniencia import _linha_de_cotacao, validar_cotacao
    from src.snapshots import prob_do_lado, resolver

    if peso is None:
        peso, eventos_do_manifesto = peso_registrado()
        eventos_dev = eventos_dev if eventos_dev is not None else eventos_do_manifesto
    eventos_dev = eventos_dev or []

    card = pd.read_csv(Path(card_csv or config.UPCOMING_CARD_CSV))
    hist = _load_raw(Path(history_csv or config.PREDICTION_HISTORY_CSV))
    congelado = {}
    if not hist.empty:
        for _, r in hist[hist["event_date"] == event_date].iterrows():
            congelado[frozenset((str(r["fighter_a"]).strip(),
                                 str(r["fighter_b"]).strip()))] = r

    fatia = ("desenvolvimento" if str(event_date) in eventos_dev
             else "avaliacao_prospectiva")
    out = []
    for _, c in card.iterrows():
        a, b = str(c["fighter_a"]).strip(), str(c["fighter_b"]).strip()
        base = {"versao_protocolo": vantagem.VERSAO, "peso_w": peso,
                "fatia": fatia, "event_date": event_date,
                "fighter_a": a, "fighter_b": b,
                "temporalidade": vantagem.TEMPORALIDADE}

        pre = congelado.get(frozenset((a, b)))
        if pre is None:
            out.append({**base, "selecao": False,
                        "motivo": MOTIVO_SEM_PRE_REGISTRO})
            continue
        base["event_name"] = pre.get("event_name")

        pid = pre.get("prediction_id")
        pid = "" if pd.isna(pid) else str(pid).strip()
        versao, porque = resolver(pid, event_date=event_date,
                                  fighter_a=a, fighter_b=b)
        if versao is None:
            out.append({**base, "prediction_id": pid, "selecao": False,
                        "motivo": f"versao da previsao nao verificavel: {porque}"})
            continue
        base.update({
            "prediction_id": pid,
            "model_name": versao.get("model_name"),
            "manifesto_sha256": versao.get("manifesto_sha256"),
            "modelos_sha256": versao.get("modelos_sha256"),
        })

        if pd.isna(c.get("odds_a_decimal")) or pd.isna(c.get("odds_b_decimal")):
            out.append({**base, "selecao": False, "motivo": MOTIVO_SEM_MEDIANA})
            continue

        p_merc_a = vantagem.devig(c["odds_a_decimal"], c["odds_b_decimal"])
        # orientacao: a versao tem a ordem DELA; trazer para a ordem do card
        p_mod_a = prob_do_lado(versao, a)
        if p_mod_a is None:
            out.append({**base, "selecao": False,
                        "motivo": "a versao nao cobre os participantes do card"})
            continue
        p_comb_a = float(vantagem.combinar([p_merc_a], [p_mod_a], peso)[0])

        lado = str(versao.get("model_side") or "").strip()
        if lado not in (a, b):
            out.append({**base, "selecao": False,
                        "motivo": "o lado da versao nao esta no card"})
            continue
        e_lado_a = lado == a
        prob = {k: (v if e_lado_a else 1.0 - v) for k, v in
                (("mercado", p_merc_a), ("modelo", p_mod_a),
                 ("combinacao", p_comb_a))}
        base.update({
            "p_mercado_a": round(p_merc_a, 4), "p_modelo_a": round(p_mod_a, 4),
            "p_combinacao_a": round(p_comb_a, 4), "lado": lado,
            "prob_lado_mercado": round(prob["mercado"], 4),
            "prob_lado_modelo": round(prob["modelo"], 4),
            "prob_lado_combinacao": round(prob["combinacao"], 4),
            "odd_mediana_lado": float(c["odds_a_decimal"] if e_lado_a
                                      else c["odds_b_decimal"]),
        })

        odd_exec = c.get("odds_a_tp") if e_lado_a else c.get("odds_b_tp")
        ref = (c.get("ref_tp_a", c.get("ref_tp")) if e_lado_a
               else c.get("ref_tp_b"))
        if pd.isna(odd_exec):
            out.append({**base, "selecao": False, "motivo": MOTIVO_SEM_EXECUTAVEL})
            continue
        ok, porque_preco = validar_cotacao(
            ref, casa="Thunderpick", mercado="h2h", lado=lado,
            participantes=(a, b), odd=float(odd_exec), exigir_atual=True,
            agora=agora)
        cot = _linha_de_cotacao(str(ref).strip()) if not _vazio(ref) else None
        base.update({
            "odd_executavel_lado": float(odd_exec),
            "ref_preco": "" if _vazio(ref) else str(ref),
            "casa": (cot or {}).get("casa", ""),
            "mercado": (cot or {}).get("mercado", ""),
            "preco_lido_em": (cot or {}).get("captured_at", ""),
            "inicio_programado": (cot or {}).get("inicio_programado", ""),
        })
        if not ok:
            out.append({**base, "selecao": False,
                        "motivo": f"{MOTIVO_SEM_EXECUTAVEL}: {porque_preco}"})
            continue

        # A POLITICA usa a probabilidade do candidato C, que com w = 0 e
        # identica a do mercado. Isso e deliberado: o experimento concluiu que
        # o modelo nao acrescentou informacao, e a sombra registra a decisao
        # que o procedimento de fato produz, nao a que daria mais entradas.
        bruto = prob["combinacao"] * float(odd_exec)
        base.update({"retorno_bruto_esperado": round(bruto, 4),
                     "lucro_liquido_esperado": round(bruto - 1.0, 4)})
        if bruto <= LIMIAR_RETORNO_BRUTO:
            out.append({**base, "selecao": False, "motivo": MOTIVO_EV_BAIXO})
            continue
        out.append({**base, "selecao": True, "motivo": ""})
    return out


def liquidar(event_date: str, history_csv: Path | None = None,
             arquivo: Path | None = None,
             registros: Path | None = None) -> int:
    """
    Anexa o resultado das linhas de sombra daquele evento, **sem reescrever**
    o que foi previsto.

    O resultado vem do histórico de produção (`actual_winner`), que é a fonte
    única de verdade sobre quem ganhou. Linha de sombra sem resultado ainda no
    histórico é deixada em paz e volta na próxima liquidação.
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
        acertou = bool(r["lado"]) and venc == r["lado"]
        if r["selecao"] in ("True", "true", "1") and r["odd_executavel_lado"]:
            unidades = UNIDADES_POR_PERNA
            retorno = (unidades * (float(r["odd_executavel_lado"]) - 1.0)
                       if acertou else -unidades)
        else:
            unidades, retorno = 0.0, 0.0
        novas.append({
            "liquidado_em": agora, "registro_id": r["registro_id"],
            "event_date": r["event_date"], "fighter_a": r["fighter_a"],
            "fighter_b": r["fighter_b"], "lado": r["lado"], "vencedor": venc,
            "acertou": acertou, "unidades": unidades,
            "retorno_unidades": round(retorno, 4),
            "fonte_do_resultado": "data/prediction_history.csv:actual_winner",
            "limite": ("retorno de pesquisa, unidade fixa, sem dinheiro "
                       "envolvido. Nao e registro de aposta."),
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
