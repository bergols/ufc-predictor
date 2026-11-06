"""
src/vantagem.py

Experimento: o modelo acrescenta informação ao mercado?

Implementa o que `PROTOCOLO_VANTAGEM.md` (versão `vantagem-v1`) pré-registrou.
Este módulo **lê** registros de produção e **escreve** só em
`data/experimentos/`. Não altera modelo, histórico, snapshots, decisões,
proveniência nem o site.

Três candidatos, na mesma população e nos mesmos horários:

    A  mercado      implícita devigada da mediana de casas
    B  modelo       `model_prob_a` COMO FOI CONGELADA no pré-registro
    C  combinação   pooling logístico com um peso, w ∈ [0, 1]

O ponto da formulação C é que **`w = 0` é permitido**: o procedimento pode
concluir que o modelo não acrescenta informação. Um procedimento que não pode
chegar a essa conclusão não está medindo nada.

Por que a probabilidade do modelo não é recalculada
---------------------------------------------------
Recalcular hoje a previsão de uma luta de agosto usaria uma base que já contém
o resultado daquela luta — `export_latest_fighter_levels` reflete a base ATUAL.
Medido em 08/ago/2026: as dez lutas se moveram na direção do vencedor real. A
previsão que vale é a que foi publicada antes, e ela está congelada no
histórico.

Limitação temporal declarada, não contornada
--------------------------------------------
O pré-registro guarda a DATA DO EVENTO, não o instante da gravação. As versões
com `criado_em` existem desde 06/11/2026; as 207 linhas anteriores foram
semeadas com temporalidade **não recuperável**. Toda saída deste módulo carrega
esse texto por linha. Nenhum horário é inventado.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import temporalidade

logger = logging.getLogger(__name__)

VERSAO = "vantagem-v1"
PASTA = config.DATA_DIR / "experimentos"
SEMENTE = 20261106
N_BOOTSTRAP = 10_000

# Aperto antes do logit, fixado no protocolo: sem ele uma probabilidade 0 ou 1
# estoura o log loss e uma unica linha decidiria a comparacao inteira.
EPS = 1e-3

# Grade do peso, fixada no protocolo. Inclui 0,00 de proposito.
GRADE_W = np.round(np.arange(0.0, 1.0001, 0.01), 2)

TEMPORALIDADE = (
    "instante da previsao conhecido na granularidade do EVENTO, nao do "
    "registro: o pre-registro guardava a data do evento e as versoes com "
    "criado_em existem so desde 2026-11-06. Preco: capturado na montagem do "
    "card, antes do evento; proveniencia por preco existe desde out/2026 e "
    "falta nas linhas anteriores."
)

MOTIVOS_EXCLUSAO = {
    "sem_resultado": "actual_winner vazio: sem resultado nao ha o que avaliar",
    "sem_previsao": "model_prob_a vazio: luta sem previsao (estreantes nos dois lados)",
    "sem_odds": "odds_a_decimal ou odds_b_decimal vazio: a retirada da margem "
                "exige os DOIS lados",
    "estado": "estado_resultado diferente de 'vencedor': luta cancelada, "
              "adversario trocado ou estado nao verificado",
    "vencedor_fora": "actual_winner nao e nenhum dos dois participantes da linha",
}


# ---------------------------------------------------------------------------
# probabilidade
# ---------------------------------------------------------------------------

def devig(odd_a: float, odd_b: float) -> float:
    """
    Probabilidade do lado A, com a margem da casa retirada.

    A odd decimal implica `1/odd`. Num mercado de dois resultados as duas
    implícitas somam mais de 1 — o overround. Normalização proporcional, que é
    o método mais transparente e **não** o único (há "power" e "Shin", que
    distribuem a margem de forma desigual entre favorito e azarão). A escolha
    está no protocolo, feita antes dos resultados.
    """
    from src.utils import remove_vig_two_way
    ia, ib = 1.0 / float(odd_a), 1.0 / float(odd_b)
    return remove_vig_two_way(ia, ib)[0]


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def combinar(p_mercado, p_modelo, w: float):
    """
    Pooling logístico: `logit(p_C) = (1-w)·logit(p_A) + w·logit(p_B)`.

    Simétrico por construção: inverter A/B nega os dois logits, logo nega o
    combinado, logo `p_C -> 1 - p_C`. A decisão econômica para o mesmo atleta
    não muda. Travado em `tests/test_vantagem.py`.
    """
    return _expit((1.0 - w) * _logit(p_mercado) + w * _logit(p_modelo))


# ---------------------------------------------------------------------------
# metricas
# ---------------------------------------------------------------------------

def log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def auc(p, y) -> float | None:
    """AUC, ou None quando a amostra tem uma classe só (aí ela não existe)."""
    y = np.asarray(y, dtype=int)
    if len(set(y.tolist())) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, np.asarray(p, dtype=float)))


def metricas(p, y) -> dict:
    return {"n": int(len(y)), "log_loss": round(log_loss(p, y), 4),
            "brier": round(brier(p, y), 4),
            "auc": None if auc(p, y) is None else round(auc(p, y), 4)}


def tabela_de_calibracao(p, y, faixas=None, eventos=None,
                        semente: int = SEMENTE) -> pd.DataFrame:
    """
    Curva de calibração com CONTAGEM e incerteza por faixa.

    A contagem é obrigatória na leitura: numa faixa com poucas observações o
    intervalo é largo e a coluna "obs" sozinha não diz nada.

    **Wilson assume independência, e lutas do mesmo card não são
    independentes** (E9 da errata). Quando `eventos` é passado, a tabela
    acrescenta `ic95_agrupado_*` por bootstrap de evento — a incerteza que
    respeita o desenho — e marca `wilson_otimista` na faixa que contém mais de
    uma luta do mesmo evento. Sem `eventos`, a coluna `wilson_otimista` fica
    vazia e o Wilson vale como referência de faixa, não como incerteza do
    desenho.
    """
    faixas = faixas or [0, .2, .4, .5, .6, .8, 1.0]
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    ev = None if eventos is None else np.asarray(eventos, dtype=object)
    rng = np.random.default_rng(semente)
    linhas = []
    for lo, hi in zip(faixas[:-1], faixas[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        n = int(m.sum())
        if n == 0:
            linhas.append({"faixa": f"[{lo:.0%},{hi:.0%})", "n": 0,
                           "n_eventos": 0, "prev_media": None, "obs": None,
                           "ic95_baixo": None, "ic95_alto": None,
                           "wilson_otimista": None,
                           "ic95_agrupado_baixo": None,
                           "ic95_agrupado_alto": None})
            continue
        k = float(y[m].sum())
        obs = k / n
        # Wilson 95%
        z = 1.959964
        den = 1 + z * z / n
        centro = (obs + z * z / (2 * n)) / den
        meio = z * np.sqrt(obs * (1 - obs) / n + z * z / (4 * n * n)) / den
        # incerteza AGRUPADA por evento: a que respeita o desenho
        agr_baixo = agr_alto = None
        n_eventos = 0
        otimista = None
        if ev is not None:
            dentro = ev[m]
            unicos = sorted(set(dentro.tolist()))
            n_eventos = len(unicos)
            otimista = bool(n > n_eventos)
            if n_eventos >= 2:
                por = {e: y[m][dentro == e] for e in unicos}
                amostras = np.empty(2000, dtype=float)
                for i in range(2000):
                    sorteio = rng.choice(unicos, size=n_eventos, replace=True)
                    vals = np.concatenate([por[e] for e in sorteio])
                    amostras[i] = float(vals.mean())
                lo_a, hi_a = np.percentile(amostras, [2.5, 97.5])
                agr_baixo, agr_alto = round(float(lo_a), 4), round(float(hi_a), 4)
        linhas.append({"faixa": f"[{lo:.0%},{hi:.0%})", "n": n,
                       "n_eventos": n_eventos,
                       "prev_media": round(float(p[m].mean()), 4),
                       "obs": round(obs, 4),
                       "ic95_baixo": round(max(0.0, centro - meio), 4),
                       "ic95_alto": round(min(1.0, centro + meio), 4),
                       "wilson_otimista": otimista,
                       "ic95_agrupado_baixo": agr_baixo,
                       "ic95_agrupado_alto": agr_alto})
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# populacao
# ---------------------------------------------------------------------------

def carregar_populacao(history_csv: Path | None = None
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    (elegíveis, exclusões). Lê o pré-registro e **não o altera**.

    Elegibilidade e motivos de exclusão estão na §2 do protocolo. Nada é
    imputado: ausência de dado é exclusão registrada, nunca suposição.
    """
    from src.prediction_history import _load_raw

    caminho = Path(history_csv or config.PREDICTION_HISTORY_CSV)
    df = _load_raw(caminho)
    if df.empty:
        vazio = pd.DataFrame()
        return vazio, vazio

    excl = []
    linhas = []
    for _, r in df.iterrows():
        chave = {"event_date": r.get("event_date"),
                 "event_name": r.get("event_name"),
                 "fighter_a": r.get("fighter_a"), "fighter_b": r.get("fighter_b")}
        if pd.isna(r.get("actual_winner")):
            excl.append({**chave, "motivo": MOTIVOS_EXCLUSAO["sem_resultado"]})
            continue
        if str(r.get("estado_resultado") or "").strip() != "vencedor":
            excl.append({**chave, "motivo": MOTIVOS_EXCLUSAO["estado"]})
            continue
        if pd.isna(r.get("model_prob_a")):
            excl.append({**chave, "motivo": MOTIVOS_EXCLUSAO["sem_previsao"]})
            continue
        if pd.isna(r.get("odds_a_decimal")) or pd.isna(r.get("odds_b_decimal")):
            excl.append({**chave, "motivo": MOTIVOS_EXCLUSAO["sem_odds"]})
            continue
        a = str(r["fighter_a"]).strip()
        b = str(r["fighter_b"]).strip()
        vencedor = str(r["actual_winner"]).strip()
        if vencedor not in (a, b):
            excl.append({**chave, "motivo": MOTIVOS_EXCLUSAO["vencedor_fora"]})
            continue
        # CLASSIFICACAO TEMPORAL POR LINHA. Constar do pre-registro e indicio,
        # nao prova: o v1 tratou disponibilidade do dado como comprovacao
        # temporal. Ver ERRATA_VANTAGEM_V1.md e src/temporalidade.py.
        inicio, origem_inicio = temporalidade.inicio_programado(
            r.get("ref_tp"), r.get("ref_tp_b"), r.get("sharp_ref"),
            r.get("close_ref"))
        ev_prev = temporalidade.da_previsao(r.get("prediction_id"), inicio=inicio)
        ev_med = temporalidade.da_mediana()
        ev_exec = temporalidade.do_preco(
            r.get("ref_tp"), inicio=inicio, lado=a,
            participantes=(a, b), casa="Thunderpick", mercado="h2h",
            odd=None if pd.isna(r.get("odds_a_tp")) else float(r["odds_a_tp"]))
        certificada = bool(ev_prev) and bool(ev_exec)

        linhas.append({
            "event_date": str(r["event_date"]), "event_name": r.get("event_name"),
            "fight_uid": r.get("fight_uid"), "fighter_a": a, "fighter_b": b,
            # ORIENTACAO: tudo nesta linha e do lado A. y=1 significa que A ganhou.
            "y": 1 if vencedor == a else 0,
            "odd_a": float(r["odds_a_decimal"]), "odd_b": float(r["odds_b_decimal"]),
            "p_mercado": devig(r["odds_a_decimal"], r["odds_b_decimal"]),
            "p_modelo": float(r["model_prob_a"]),
            "model_side": r.get("model_side"),
            "participantes_fonte": r.get("participantes_fonte"),
            "sharp_prob": r.get("sharp_prob"), "sharp_casa": r.get("sharp_casa"),
            "sharp_ref": r.get("sharp_ref"), "close_prob": r.get("close_prob"),
            "odd_a_tp": r.get("odds_a_tp"), "odd_b_tp": r.get("odds_b_tp"),
            "ref_tp": r.get("ref_tp"), "prediction_id": r.get("prediction_id"),
            # o que foi EFETIVAMENTE encontrado, por item
            "inicio_programado": inicio, "origem_do_inicio": origem_inicio,
            "temporalidade_previsao": ev_prev.classe,
            "evidencia_previsao": ev_prev.detalhe,
            "instante_previsao": ev_prev.instante,
            "temporalidade_mediana": ev_med.classe,
            "evidencia_mediana": ev_med.detalhe,
            "temporalidade_preco_executavel": ev_exec.classe,
            "evidencia_preco_executavel": ev_exec.detalhe,
            "instante_preco_executavel": ev_exec.instante,
            "certificada": certificada,
            "classe_analise": ("certificada" if certificada
                               else "exploratoria_historica"),
            "temporalidade": TEMPORALIDADE,
        })
    return pd.DataFrame(linhas), pd.DataFrame(excl)


def fatias_por_evento(pop: pd.DataFrame, n_dev: int | None = None) -> pd.DataFrame:
    """
    Marca `fatia` em cada linha: "desenvolvimento" ou "avaliacao".

    Divisão **cronológica por evento**: todas as lutas de um card ficam juntas.
    Dividir dentro do evento vazaria contexto do próprio card. Metade dos
    eventos em cada fatia, como o protocolo fixou; o default é metade para
    cima, para o desenvolvimento nunca ficar com menos que a avaliação.
    """
    pop = pop.copy()
    eventos = sorted(pop["event_date"].unique())
    corte = n_dev if n_dev is not None else (len(eventos) + 1) // 2
    dev = set(eventos[:corte])
    pop["fatia"] = np.where(pop["event_date"].isin(dev),
                            "desenvolvimento", "avaliacao")
    return pop


def experiencia_previa(pop: pd.DataFrame,
                       fights_csv: Path | None = None) -> pd.DataFrame:
    """
    Grupo de experiência prévia no UFC: 0, 1-2, 3-5, 6+ lutas.

    Contagem **point-in-time**: só lutas com data ANTERIOR à do evento. Usar a
    contagem de hoje para uma luta de agosto contaria lutas que ainda não
    aconteceram. O grupo é o do lutador MENOS experiente dos dois, que é o
    recorte do `DISCRIMINACAO.md`.

    Sem a base bruta (clone limpo), devolve o grupo como "desconhecido" em vez
    de inventar contagem.
    """
    pop = pop.copy()
    caminho = Path(fights_csv or config.RAW_FIGHTS_CSV)
    if not caminho.exists():
        logger.info("Base bruta ausente (%s): grupo de experiencia fica "
                    "'desconhecido'.", caminho)
        pop["lutas_previas_min"] = np.nan
        pop["grupo_experiencia"] = "desconhecido"
        return pop

    f = pd.read_csv(caminho, low_memory=False)
    f["event_date"] = pd.to_datetime(f["event_date"], errors="coerce")
    historico: dict[str, list] = {}
    for _, r in f.iterrows():
        for col in ("fighter_1", "fighter_2"):
            nome = str(r.get(col) or "").strip()
            if nome:
                historico.setdefault(nome, []).append(r["event_date"])

    def conta(nome, quando):
        datas = historico.get(str(nome).strip(), [])
        return int(sum(1 for d in datas if pd.notna(d) and d < quando))

    minimos, grupos = [], []
    for _, r in pop.iterrows():
        quando = pd.to_datetime(r["event_date"])
        n = min(conta(r["fighter_a"], quando), conta(r["fighter_b"], quando))
        minimos.append(n)
        grupos.append("0" if n == 0 else "1-2" if n <= 2
                       else "3-5" if n <= 5 else "6+")
    pop["lutas_previas_min"] = minimos
    pop["grupo_experiencia"] = grupos
    return pop


# ---------------------------------------------------------------------------
# peso e bootstrap
# ---------------------------------------------------------------------------

def peso_otimo(p_mercado, p_modelo, y, grade=None) -> tuple[float, dict]:
    """
    `w` que minimiza log loss, em grade fixada no protocolo.

    Estimado **só** na fatia de desenvolvimento. O critério é log loss, nunca
    lucro de janela nenhuma. `w = 0` está na grade e é um resultado legítimo:
    significa que o modelo não acrescentou informação ali.
    """
    grade = GRADE_W if grade is None else np.asarray(grade, dtype=float)
    perdas = {float(w): log_loss(combinar(p_mercado, p_modelo, float(w)), y)
              for w in grade}
    melhor = min(perdas, key=perdas.get)
    return float(melhor), {str(k): round(v, 6) for k, v in perdas.items()}


def bootstrap_por_evento(df: pd.DataFrame, col_a: str, col_b: str,
                         n: int = N_BOOTSTRAP, semente: int = SEMENTE) -> dict:
    """
    IC 95% da diferença PAREADA de log loss (`col_a − col_b`), reamostrando
    **eventos** com reposição.

    O evento é a unidade porque as lutas de um card compartilham condições;
    tratá-las como independentes estreitaria o intervalo artificialmente. Cada
    reamostragem usa os MESMOS eventos para os dois candidatos, então a
    diferença é pareada e a variação comum ao par sai da conta.
    """
    rng = np.random.default_rng(semente)
    eventos = df["event_date"].unique()
    por_evento = {e: df[df["event_date"] == e] for e in eventos}
    obs = log_loss(df[col_a], df["y"]) - log_loss(df[col_b], df["y"])
    amostras = np.empty(n, dtype=float)
    for i in range(n):
        escolhidos = rng.choice(eventos, size=len(eventos), replace=True)
        bloco = pd.concat([por_evento[e] for e in escolhidos], ignore_index=True)
        amostras[i] = (log_loss(bloco[col_a], bloco["y"])
                       - log_loss(bloco[col_b], bloco["y"]))
    baixo, alto = np.percentile(amostras, [2.5, 97.5])
    return {"diferenca": round(float(obs), 4),
            "ic95_baixo": round(float(baixo), 4),
            "ic95_alto": round(float(alto), 4),
            "n_eventos": int(len(eventos)), "n_lutas": int(len(df)),
            "n_reamostragens": int(n), "semente": int(semente),
            "fracao_negativa": round(float(np.mean(amostras < 0)), 4)}


def precisao_observada(df: pd.DataFrame, col_a: str, col_b: str,
                       semente: int = SEMENTE) -> float:
    """
    Metade da largura do IC95 observado: a **precisão desta amostra**.

    O `vantagem-v1` chamou este número de "menor efeito detectável". Está
    errado, e a errata corrige: meia-largura de IC descreve a precisão com que
    ESTA amostra estimou ESTE contraste. Não é efeito mínimo detectável, que
    é uma quantidade de análise de poder e depende de efeito mínimo de
    interesse, nível de significância, poder desejado e variabilidade — nada
    disso entra aqui.

    Em particular: um efeito **menor** que esta meia-largura pode perfeitamente
    existir e ser detectado por uma amostra maior; e um efeito maior que ela
    pode passar sem significância nesta amostra, porque o IC depende do que
    foi observado, não só do tamanho.
    """
    r = bootstrap_por_evento(df, col_a, col_b, n=2000, semente=semente)
    return round((r["ic95_alto"] - r["ic95_baixo"]) / 2, 4)
