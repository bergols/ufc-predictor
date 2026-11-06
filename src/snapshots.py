"""
src/snapshots.py

Versões imutáveis da previsão — `data/previsoes_snapshots.csv`.

O problema
----------
`prediction_history.csv` guardava UMA linha por luta, e a linha de um evento
que ainda não aconteceu era **reescrita** a cada regeração do relatório. A
regra estava documentada e tinha motivo (enquanto o resultado não chega,
atualizar as odds deve refrescar a previsão), mas o efeito é que a versão
anterior desaparecia: não havia como dizer o que a página mostrava na
terça-feira, nem contra qual versão uma decisão foi tomada.

Medido em 06/11/2026: regerar o card de 07/11 moveu seis odds da mediana,
seis da casa e uma probabilidade (Gabriel Lorenco, 0,4767 → 0,4772). Nada
disso estava errado — e nada disso era recuperável.

O que mudou
-----------
Toda versão da previsão de uma luta vira uma linha **imutável** aqui, com

    prediction_id      identificador da versão (sha256 curto do conteúdo
                       mais o instante mais a versão que ela sucede)
    criado_em          instante UTC em que ESTA versão foi criada
    substitui          `prediction_id` da versão anterior, ou vazio
    manifesto_sha256   manifesto de treino em vigor quando ela foi criada
    modelos_sha256     hash dos arquivos de modelo usados
    ref_tp/ref_tp_b/sharp_ref
                       as provas de preço daquela versão
    inicio_programado  início da luta, como a fonte da cotação declara
    pos_inicio         a versão nasceu DEPOIS do início daquela luta?

O arquivo **só recebe linhas**. Nenhuma função aqui reescreve ou apaga, e o
histórico passa a carregar `prediction_id` apontando para a versão em vigor.
Substituir deixou de acontecer: o que havia antes continua endereçável.

`pos_inicio` e por que ele existe
---------------------------------
Previsão criada depois de a luta começar não é previsão. Ela não é apagada —
o arquivo não esconde linha —, mas é marcada, e o gate de seleção bloqueia
qualquer entrada apoiada nela. A conta é **por luta**, não por card: num
evento as lutas começam ao longo de cinco horas, e usar o horário do card
liberaria uma previsão feita depois do primeiro combate.

Versões históricas
------------------
As linhas que já existiam no histórico quando este arquivo nasceu ganham um
snapshot de semeadura com `criado_em` **vazio** e `temporalidade` dizendo que
o instante não é recuperável: o arquivo antigo não guardava quando cada
versão foi escrita. Inventar uma data agora seria pior que admitir a lacuna.
"""
from __future__ import annotations

import csv
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import config

logger = logging.getLogger(__name__)

ARQUIVO = config.DATA_DIR / "previsoes_snapshots.csv"

COLUNAS = ("prediction_id", "criado_em", "event_date", "event_name",
           "fighter_a", "fighter_b", "model_name", "model_prob_a",
           "model_side", "odds_a_decimal", "odds_b_decimal",
           "odds_a_tp", "odds_b_tp", "ref_tp", "ref_tp_b",
           "sharp_prob", "sharp_best_odd", "sharp_ref", "fonte_card",
           "method_ko_tko", "method_submission", "method_decision",
           "manifesto_sha256", "modelos_sha256", "inicio_programado",
           "pos_inicio", "substitui", "temporalidade", "limite")

_LIMITE = ("linha imutavel: este arquivo so recebe linhas. criado_em e o "
           "instante em que ESTA versao foi criada, nao o instante do preco. "
           "pos_inicio=True significa previsao criada depois do inicio "
           "daquela luta, e nenhuma selecao se apoia nela.")

TEMPORALIDADE_PERDIDA = (
    "NAO RECUPERAVEL: semeado do historico em 2026-11-06. O arquivo antigo "
    "guardava uma linha por luta e a reescrevia, entao nem o instante desta "
    "versao nem as versoes anteriores dela existem. Esta linha e o ultimo "
    "estado conhecido, nao o primeiro.")


def arquivo_de(history_csv: Path | None = None) -> Path:
    """
    O arquivo de versões que acompanha AQUELE histórico.

    As versões pertencem ao histórico que aponta para elas, então elas moram
    na mesma pasta. Quem redireciona o histórico — todo teste — redireciona
    as versões junto, sem precisar lembrar de um segundo caminho. Em
    06/11/2026 foi exatamente esse esquecimento que fez a suíte escrever
    cotações sintéticas no índice de proveniência de verdade.
    """
    h = Path(history_csv or config.PREDICTION_HISTORY_CSV)
    padrao = Path(ARQUIVO)
    return padrao if h.parent == padrao.parent else h.parent / padrao.name


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def novo_id(linha: dict, criado_em: str) -> str:
    """
    Identificador da versão: hash do conteúdo que a define, mais o instante,
    mais a versão que ela sucede.

    O instante entra porque duas versões idênticas em momentos diferentes
    **são** duas versões — foi o preço que não andou, e isso é informação. A
    versão sucedida entra porque a mesma previsão escrita depois de outra não
    é a mesma linha da cadeia.
    """
    semente = "|".join(str(linha.get(c, "")) for c in (
        "event_date", "event_name", "fighter_a", "fighter_b", "model_name",
        "model_prob_a", "model_side", "odds_a_decimal", "odds_b_decimal",
        "odds_a_tp", "odds_b_tp", "ref_tp", "ref_tp_b", "sharp_ref",
        "manifesto_sha256", "substitui")) + "|" + criado_em
    return hashlib.sha256(semente.encode("utf-8")).hexdigest()[:16]


def _vazio(v) -> bool:
    import pandas as pd
    return (v is None or (isinstance(v, float) and pd.isna(v))
            or not str(v).strip() or str(v).strip().lower() in ("nan", "none"))


def inicio_da_luta(*refs) -> str:
    """
    Início programado da luta, lido do índice de cotações pela referência de
    preço. Vazio quando nenhuma referência o declara — e vazio **não** vira
    "ainda não começou": quem decide trata ausência como não verificável.
    """
    from src.proveniencia import _linha_de_cotacao
    for ref in refs:
        if _vazio(ref):
            continue
        linha = _linha_de_cotacao(str(ref).strip())
        if linha and (linha.get("inicio_programado") or "").strip():
            return linha["inicio_programado"].strip()
    return ""


def ja_comecou(inicio: str, agora: datetime | None = None) -> bool | None:
    """
    True/False quando o início é conhecido e legível; **None** quando não é.
    None é a terceira resposta de propósito: "não sei se já começou" não é
    "não começou".
    """
    s = str(inicio or "").strip()
    if not s:
        return None
    try:
        comeca = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (agora or datetime.now(timezone.utc)) >= comeca


def registrar(linhas: list[dict], arquivo: Path | None = None,
              *, criado_em: str | None = None) -> list[str]:
    """
    Anexa versões e devolve os `prediction_id` na ordem recebida.

    Só anexa. Uma versão já escrita não é tocada nunca — é essa a
    propriedade que torna o arquivo utilizável como prova.
    """
    if not linhas:
        return []
    caminho = Path(arquivo or ARQUIVO)
    quando = criado_em if criado_em is not None else _agora()
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
                linha["criado_em"] = d.get("criado_em", quando)
                linha["prediction_id"] = (d.get("prediction_id")
                                          or novo_id(linha, linha["criado_em"]))
                linha["limite"] = _LIMITE
                w.writerow(linha)
                ids.append(linha["prediction_id"])
    except OSError as exc:
        logger.warning("Nao consegui gravar os snapshots de previsao (%s) -- "
                       "a versao fica sem registro proprio.", exc)
        return []
    return ids


def carregar(arquivo: Path | None = None):
    """DataFrame com todas as versões, ou vazio quando o arquivo não existe."""
    import pandas as pd
    caminho = Path(arquivo or ARQUIVO)
    if not caminho.exists():
        return pd.DataFrame(columns=list(COLUNAS))
    return pd.read_csv(caminho, dtype=str).fillna("")


def versoes_da_luta(event_date: str, fighter_a: str, fighter_b: str,
                    arquivo: Path | None = None):
    """
    Todas as versões daquela luta, da mais antiga para a mais nova.

    Casa o confronto nas DUAS ordens: o CSV do card pode trocar os lados
    entre capturas, e isso não faz dela outra luta.
    """
    df = carregar(arquivo)
    if df.empty:
        return df
    par = {str(fighter_a).strip().casefold(), str(fighter_b).strip().casefold()}
    sel = df.apply(
        lambda r: (str(r["event_date"]).strip() == str(event_date).strip()
                   and {str(r["fighter_a"]).strip().casefold(),
                        str(r["fighter_b"]).strip().casefold()} == par),
        axis=1)
    return df[sel] if len(df) else df


def versao(prediction_id: str, arquivo: Path | None = None) -> dict | None:
    """A versão exata, por id. É para cá que cada decisão aponta."""
    pid = str(prediction_id or "").strip()
    if not pid:
        return None
    df = carregar(arquivo)
    if df.empty:
        return None
    achadas = df[df["prediction_id"] == pid]
    return None if achadas.empty else achadas.iloc[-1].to_dict()


_CACHE: dict = {}


def _mapa_pos_inicio(arquivo: Path | None = None) -> dict:
    """
    `{prediction_id: nasceu depois do inicio}`, relido só quando o arquivo
    muda. O gate consulta isto uma vez por linha do histórico, e reler 200
    vezes o mesmo CSV seria o caminho mais curto para alguém tirar a
    verificação do caminho.
    """
    caminho = Path(arquivo or ARQUIVO)
    try:
        st = caminho.stat()
        chave = (str(caminho), st.st_mtime_ns, st.st_size)
    except OSError:
        return {}
    if _CACHE.get("chave") != chave:
        df = carregar(caminho)
        _CACHE["chave"] = chave
        _CACHE["mapa"] = {
            str(r["prediction_id"]):
                str(r.get("pos_inicio", "")).strip().lower() == "true"
            for _, r in df.iterrows()} if len(df) else {}
    return _CACHE["mapa"]


def criada_apos_o_inicio(prediction_id: str, arquivo: Path | None = None) -> bool:
    """
    A versão citada nasceu depois de a luta começar?

    `False` também quando a versão não é encontrada ou o início é
    desconhecido — e nisso ela **não** fica liberada: a exigência de
    atualidade já bloqueia cotação sem início programado, e é lá que a
    ausência de horário é cobrada.
    """
    pid = str(prediction_id or "").strip()
    if not pid or pid.lower() in ("nan", "none"):
        return False
    return _mapa_pos_inicio(arquivo).get(pid, False)


def semear_do_historico(history_csv: Path | None = None,
                        arquivo: Path | None = None) -> int:
    """
    Dá a cada linha do histórico que ainda não tem versão um snapshot de
    semeadura, com `criado_em` **vazio** e a temporalidade documentada.

    Idempotente: linha que já tem `prediction_id` é deixada em paz. Não
    inventa instante e não reconstrói versões que o arquivo antigo não
    guardou — ele guardava uma linha por luta e a reescrevia, então as
    versões anteriores não existem em lugar nenhum.
    """
    import pandas as pd

    from src.prediction_history import _load_raw

    caminho_h = Path(history_csv or config.PREDICTION_HISTORY_CSV)
    hist = _load_raw(caminho_h)
    if hist.empty:
        return 0
    if "prediction_id" not in hist.columns:
        hist["prediction_id"] = pd.NA
    hist["prediction_id"] = hist["prediction_id"].astype("object")

    pendentes, indices = [], []
    for i, r in hist.iterrows():
        if not _vazio(r.get("prediction_id")):
            continue
        inicio = inicio_da_luta(r.get("ref_tp"), r.get("ref_tp_b"),
                                r.get("sharp_ref"))
        pendentes.append({
            "event_date": r.get("event_date"), "event_name": r.get("event_name"),
            "fighter_a": r.get("fighter_a"), "fighter_b": r.get("fighter_b"),
            "model_name": r.get("model_name"),
            "model_prob_a": r.get("model_prob_a"),
            "model_side": r.get("model_side"),
            "odds_a_decimal": r.get("odds_a_decimal"),
            "odds_b_decimal": r.get("odds_b_decimal"),
            "odds_a_tp": r.get("odds_a_tp"), "odds_b_tp": r.get("odds_b_tp"),
            "ref_tp": r.get("ref_tp"), "ref_tp_b": r.get("ref_tp_b"),
            "sharp_prob": r.get("sharp_prob"),
            "sharp_best_odd": r.get("sharp_best_odd"),
            "sharp_ref": r.get("sharp_ref"), "fonte_card": r.get("fonte_card"),
            "method_ko_tko": r.get("method_ko_tko"),
            "method_submission": r.get("method_submission"),
            "method_decision": r.get("method_decision"),
            "inicio_programado": inicio,
            # semeadura nao afirma pos_inicio: o instante de criacao e
            # justamente o que nao se recuperou
            "pos_inicio": "", "substitui": "",
            "criado_em": "", "temporalidade": TEMPORALIDADE_PERDIDA,
        })
        indices.append(i)
    if not pendentes:
        return 0
    ids = registrar(pendentes, arquivo)
    if not ids:
        return 0
    for i, pid in zip(indices, ids):
        hist.loc[i, "prediction_id"] = pid
    hist.to_csv(caminho_h, index=False)
    logger.info("%d linha(s) do historico ganharam snapshot de semeadura "
                "(temporalidade nao recuperavel).", len(ids))
    return len(ids)


# ---------------------------------------------------------------------------
# RESOLUCAO DO PONTEIRO
# ---------------------------------------------------------------------------
#
# O historico guarda `prediction_id`. Isso e um PONTEIRO, e ponteiro tem de
# ser resolvido antes de qualquer calculo: ate a revisao de 06/11/2026 a
# decisao carregava o id como etiqueta e lia probabilidade, lado e
# participantes das colunas MUTAVEIS do historico. Uma divergencia entre a
# versao e a linha passava sem sinal -- e a linha e justamente o que se
# reescreve.

MOTIVO_SEM_ID = "o historico nao aponta para nenhuma versao da previsao"
MOTIVO_AUSENTE = "a versao citada nao existe no arquivo de versoes"
MOTIVO_DUPLICADA = "o id da versao aparece mais de uma vez no arquivo"
MOTIVO_OUTRO_CONFRONTO = "a versao citada e de outro confronto"
MOTIVO_OUTRO_EVENTO = "a versao citada e de outro evento"
MOTIVO_POS_INICIO = "a versao foi criada depois do inicio da luta"
MOTIVO_TEMPORALIDADE = ("a temporalidade da versao e desconhecida: nao da "
                        "para certificar que ela e pre-luta")
MOTIVO_SEM_PROBABILIDADE = "a versao nao tem probabilidade gravada"


def _mesmo_confronto(ax, bx, ay, by) -> bool:
    return ({str(ax).strip().casefold(), str(bx).strip().casefold()}
            == {str(ay).strip().casefold(), str(by).strip().casefold()})


def _indice(arquivo: Path | None = None) -> dict:
    """`{prediction_id: (quantas vezes aparece, ultima linha)}`, relido so
    quando o arquivo muda."""
    caminho = Path(arquivo or ARQUIVO)
    try:
        st = caminho.stat()
        chave = (str(caminho), st.st_mtime_ns, st.st_size)
    except OSError:
        return {}
    if _CACHE.get("chave_idx") != chave:
        df = carregar(caminho)
        idx: dict = {}
        if len(df):
            for _, r in df.iterrows():
                pid = str(r["prediction_id"]).strip()
                n, _antiga = idx.get(pid, (0, None))
                idx[pid] = (n + 1, r.to_dict())
        _CACHE["chave_idx"] = chave
        _CACHE["idx"] = idx
    return _CACHE["idx"]


def resolver(prediction_id, *, event_date=None, fighter_a=None,
             fighter_b=None, exigir_temporalidade: bool = True,
             arquivo: Path | None = None) -> tuple[dict | None, str | None]:
    """
    (versao, motivo). A versao vem com `model_prob_a`, `model_side`,
    `fighter_a`/`fighter_b` DA VERSAO, modelo e hashes; `motivo` e None
    quando ela pode ser usada.

    Confere, em ordem: o id existe e nao esta vazio; aparece UMA vez; e do
    confronto e do evento pedidos; nao nasceu depois do inicio da luta; tem
    temporalidade conhecida; e tem probabilidade.

    `exigir_temporalidade=False` serve para QUEM SO DESENHA: a pagina pode
    mostrar uma previsao historica cujo instante de criacao nao se recuperou,
    desde que diga que nao recuperou. Quem propoe ENTRADA nao pode.
    """
    pid = str("" if prediction_id is None else prediction_id).strip()
    if not pid or pid.lower() in ("nan", "none"):
        return None, MOTIVO_SEM_ID
    achada = _indice(arquivo).get(pid)
    if achada is None:
        return None, f"{MOTIVO_AUSENTE} ({pid})"
    quantas, v = achada
    if quantas > 1:
        return None, f"{MOTIVO_DUPLICADA} ({pid}, {quantas}x)"
    if fighter_a is not None and fighter_b is not None and not _mesmo_confronto(
            v.get("fighter_a"), v.get("fighter_b"), fighter_a, fighter_b):
        return None, (f"{MOTIVO_OUTRO_CONFRONTO}: a versao {pid} e de "
                      f"{v.get('fighter_a')} x {v.get('fighter_b')}")
    if event_date is not None and str(v.get("event_date", "")).strip() != str(event_date).strip():
        return None, (f"{MOTIVO_OUTRO_EVENTO}: a versao {pid} e de "
                      f"{v.get('event_date')}")
    if str(v.get("pos_inicio", "")).strip().lower() == "true":
        return None, MOTIVO_POS_INICIO
    if exigir_temporalidade and _vazio(v.get("criado_em")):
        return None, MOTIVO_TEMPORALIDADE
    if _vazio(v.get("model_prob_a")):
        return None, MOTIVO_SEM_PROBABILIDADE
    return v, None


def prob_do_lado(versao: dict, lado) -> float | None:
    """
    Probabilidade do lado escolhido, na ORIENTACAO DA VERSAO.

    `model_prob_a` e a probabilidade de `versao["fighter_a"]`. Ler esse
    numero pela ordem de qualquer outra tabela devolve 1-p, a probabilidade
    do adversario, com cara de numero plausivel.
    """
    try:
        p = float(versao["model_prob_a"])
    except (KeyError, TypeError, ValueError):
        return None
    alvo = str("" if lado is None else lado).strip().casefold()
    if alvo == str(versao.get("fighter_a", "")).strip().casefold():
        return p
    if alvo == str(versao.get("fighter_b", "")).strip().casefold():
        return 1.0 - p
    return None

def _metodo_da_versao(v: dict) -> dict | None:
    metodo = {}
    for cls, col in (("KO_TKO", "method_ko_tko"),
                     ("SUBMISSION", "method_submission"),
                     ("DECISION", "method_decision")):
        if not _vazio(v.get(col)):
            try:
                metodo[cls] = float(v[col])
            except (TypeError, ValueError):
                return None
    return metodo if len(metodo) == 3 else None


def para_render(event_date: str, arquivo: Path | None = None,
                history_csv: Path | None = None) -> dict:
    """
    `{(fighter_a, fighter_b): {...}}` da versão **em vigor** de cada luta do
    evento, para desenhar a página sem carregar modelo nem a base atual.

    Quem manda é o PONTEIRO do histórico (`prediction_id`), não uma varredura
    do arquivo de versões. A varredura tinha dois defeitos:

      * ela chaveava por `(fighter_a, fighter_b)` literal, então duas versões
        da mesma luta com os lados invertidos viravam **duas entradas
        concorrentes** — e quem desenhava pegava a que casasse com a ordem do
        card, que podia ser a antiga;
      * "a última linha do arquivo" não é a versão em vigor; a versão em vigor
        é a que o histórico aponta.

    A chave de cada entrada é a ordem **da versão**, e a inversão para a ordem
    do card é feita por quem lê (`card_report._frozen_for_fight`), que já sabe
    inverter a probabilidade.

    Ponteiro que não resolve devolve entrada com `model_prob_a=None` e
    `motivo` — ausência de previsão com motivo explícito, nunca outra versão
    escolhida em silêncio.

    `fechada` fica False: quem sabe se a luta encerrou é o histórico, e marcar
    como fechada o que não está mudaria o que a página afirma.
    """
    import pandas as pd

    from src.prediction_history import _load_raw

    hist = _load_raw(Path(history_csv or config.PREDICTION_HISTORY_CSV))
    if hist.empty:
        return {}
    do_evento = hist[hist["event_date"].astype(str).str.strip()
                     == str(event_date).strip()]
    out: dict = {}
    for _, r in do_evento.iterrows():
        a = str(r["fighter_a"]).strip()
        b = str(r["fighter_b"]).strip()
        pid = r.get("prediction_id")
        pid = "" if pd.isna(pid) else str(pid or "").strip()
        # DESENHAR nao exige temporalidade conhecida: a pagina pode mostrar
        # uma previsao historica cujo instante de criacao nao se recuperou,
        # desde que diga isso. Quem propoe ENTRADA e que nao pode.
        v, motivo = resolver(pid, event_date=event_date, fighter_a=a,
                             fighter_b=b, exigir_temporalidade=False,
                             arquivo=arquivo)
        if v is None:
            out[(a, b)] = {"model_prob_a": None, "method_probs": None,
                           "fechada": False, "prediction_id": pid,
                           "motivo": motivo, "temporalidade": ""}
            continue
        try:
            prob = float(v["model_prob_a"])
        except (TypeError, ValueError):
            prob = None
        # chaveada na ordem DA VERSAO; quem le inverte se precisar
        out[(str(v["fighter_a"]).strip(), str(v["fighter_b"]).strip())] = {
            "model_prob_a": prob,
            "method_probs": _metodo_da_versao(v),
            "fechada": False,
            "prediction_id": str(v["prediction_id"]).strip(),
            "motivo": None,
            "temporalidade": str(v.get("temporalidade") or ""),
        }
    return out
