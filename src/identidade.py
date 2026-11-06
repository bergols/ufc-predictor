"""
src/identidade.py

Identidade estavel e estado do resultado para as linhas do historico.

Por que existe
--------------
Ate nov/2026 o `prediction_history.csv` chaveava por NOME + DATA, e todos os
desfechos sem vencedor compartilhavam `actual_winner` vazio. Duas
consequencias, as duas confirmadas na auditoria de integridade:

  - sem id, duas lutas do mesmo par (revanche) so se distinguem pela data,
    e qualquer juncao que esqueca a data pega a errada -- foi exatamente o
    que acontecia em `compare_to_market`;
  - sem estado, luta cancelada, empate, no contest e coleta ainda incompleta
    sao a mesma coisa depois do fato. O painel conta todas como "sem
    resultado" e ninguem consegue auditar qual foi qual.

Regra que este modulo segue
---------------------------
**Deriva do que e verificavel, marca o resto como desconhecido.** Nada e
arbitrado. Luta que nao esta na base de resultados nao vira "cancelada" por
conveniencia: ela so vira `nao_aconteceu` quando o EVENTO daquela data existe
na base e aquele par nao esta nele -- aí a ausencia e evidencia. Enquanto o
evento inteiro nao entrou, o estado e `aguardando`.

Nenhuma previsao e tocada: este modulo so preenche as duas colunas novas.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config
from src.utils import best_name_match

logger = logging.getLogger(__name__)

# Origem do fight_uid, com o grau de certeza embutido no nome. O id NUNCA e
# derivado de nome ou data: ele e sempre a URL da luta no UFCStats. O que
# varia e como a LINHA foi casada com aquela luta.
UID_EXATO = "ufcstats_exato"      # os dois nomes bateram literalmente
UID_FUZZY = "ufcstats_fuzzy"      # um dos lados casou por similaridade

VENCEDOR = "vencedor"
EMPATE_NC = "empate_ou_nc"
NAO_ACONTECEU = "nao_aconteceu"
AGUARDANDO = "aguardando"
DESCONHECIDO = "desconhecido"

# Retirado do card depois de registrado: a luta existiu como pre-registro e
# saiu antes de acontecer. Mantem a linha e o historico, e fica inelegivel.
# Nasceu em 07/nov/2026, quando Gabriel Bonfim x Sean Brady -- o main event
# que dava nome ao evento -- saiu do card.
RETIRADO_DO_CARD = "retirado_do_card"

ESTADOS = (VENCEDOR, EMPATE_NC, NAO_ACONTECEU, AGUARDANDO, DESCONHECIDO,
           RETIRADO_DO_CARD)


def _lado(nome: str, candidatos: list[str]) -> str | None:
    if nome in candidatos:
        return nome
    return best_name_match(nome, candidatos)


def resolver(fighter_a: str, fighter_b: str, event_date: str,
             tem_vencedor: bool, base: pd.DataFrame
             ) -> tuple[str | None, str, str | None]:
    """
    (fight_uid, estado, origem_do_uid) desta linha.

    `base` e a tabela de resultados com fight_id, event_date, fighter_1,
    fighter_2, winner.
    """
    do_dia = base[base["_data"] == str(event_date)[:10]]
    if do_dia.empty:
        # o evento inteiro ainda nao entrou na base: a ausencia nao prova
        # nada sobre esta luta
        return None, (AGUARDANDO if not tem_vencedor else DESCONHECIDO), None

    achadas, exato = [], False
    for r in do_dia.itertuples():
        lados = [str(r.fighter_1), str(r.fighter_2)]
        ma, mb = _lado(fighter_a, lados), _lado(fighter_b, lados)
        if ma and mb and ma != mb:
            achadas.append(r)
            exato = fighter_a in lados and fighter_b in lados

    if len(achadas) > 1:
        return None, DESCONHECIDO, None    # ambiguo: nao arbitra
    if not achadas:
        # o evento existe na base e este par nao esta nele -- a ausencia
        # passa a ser evidencia (adversario trocado, luta cancelada)
        return None, NAO_ACONTECEU, None
    luta = achadas[0]
    uid = str(luta.fight_id) if pd.notna(luta.fight_id) else None
    estado = EMPATE_NC if pd.isna(luta.winner) else VENCEDOR
    return uid, estado, (UID_EXATO if exato else UID_FUZZY)


def preencher(history_df: pd.DataFrame,
              fights_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Copia do historico com `fight_uid` e `estado_resultado` preenchidos.

    Nao sobrescreve valor ja presente: uma vez resolvida, a identidade da
    linha nao muda. Nenhuma outra coluna e tocada.
    """
    if fights_df is None:
        fights_df = pd.read_csv(config.RAW_FIGHTS_CSV,
                                usecols=["fight_url", "event_url", "event_date",
                                         "fighter_1", "fighter_2", "winner"])
    base = fights_df.copy()
    # base vazia e estado real -- clone novo, ou coleta que ainda nao rodou.
    # Sem as colunas, tudo vira "aguardando", que e a leitura certa.
    for col in ("fight_id", "event_url", "event_date", "fighter_1",
                "fighter_2", "winner"):
        if col not in base.columns:
            base[col] = base["fight_url"] if (col == "fight_id"
                                              and "fight_url" in base.columns) else np.nan
    base["_data"] = base["event_date"].astype(str).str[:10]

    out = history_df.copy()
    for col in ("event_uid", "fight_uid", "estado_resultado", "uid_origem"):
        # object desde o inicio: coluna criada como float64 (so NaN) recusa
        # string no pandas 3 com "Invalid value 'u1' for dtype 'float64'"
        if col not in out.columns:
            out[col] = pd.Series([np.nan] * len(out), index=out.index, dtype="object")
        else:
            out[col] = out[col].astype("object")

    n_uid = n_estado = 0
    for i, r in out.iterrows():
        se_tem = pd.notna(r.get("actual_winner"))
        precisa_uid = pd.isna(r.get("fight_uid"))
        precisa_estado = pd.isna(r.get("estado_resultado"))
        # linha que ja tem id mas nao tem ORIGEM precisa ser reconferida:
        # sem a origem nao da para saber se aquele id veio de casamento
        # literal ou por similaridade
        precisa_origem = pd.notna(r.get("fight_uid")) and pd.isna(r.get("uid_origem"))
        if not (precisa_uid or precisa_estado or precisa_origem):
            continue
        uid, estado, origem = resolver(str(r["fighter_a"]), str(r["fighter_b"]),
                                       r["event_date"], se_tem, base)
        # coerencia: linha COM vencedor preenchido a mao vale como vencedor
        # mesmo que a base ainda nao tenha a luta
        if se_tem and estado in (AGUARDANDO, NAO_ACONTECEU, DESCONHECIDO):
            estado = VENCEDOR
        if precisa_uid and uid:
            out.at[i, "fight_uid"] = uid
            out.at[i, "uid_origem"] = origem
            n_uid += 1
        elif precisa_origem and origem:
            out.at[i, "uid_origem"] = origem
        if precisa_estado:
            out.at[i, "estado_resultado"] = estado
            n_estado += 1

    logger.info("identidade: %d fight_uid e %d estado_resultado preenchidos.",
                n_uid, n_estado)
    return out


class IdentidadeInconsistente(RuntimeError):
    """CSV, template, historico e estado local discordam sobre que card e."""


def validar_identidade_do_card(csv_card=None, template=None, historico=None,
                               estado=None) -> dict:
    """
    Confere que as QUATRO fontes concordam sobre qual evento o card atual e,
    e levanta `IdentidadeInconsistente` quando nao concordam.

    Em 01/nov/2026 o ciclo publicou com o nome e a data de um evento
    enquanto o CSV era de outro, e gravou 12 lutas sob a data errada. Derivar
    do CSV ajuda, mas par de nomes sozinho tambem e ambiguo -- revanche tem
    o mesmo par em datas diferentes. Por isso a conferencia cruza as quatro
    fontes em vez de eleger uma.

    Devolve {"event_name", "event_date", "n_lutas"} quando tudo bate.
    """
    import json

    csv_card = pd.read_csv(csv_card or config.UPCOMING_CARD_CSV)
    h = pd.read_csv(historico or config.PREDICTION_HISTORY_CSV)
    t = pd.read_csv(template or config.ODDS_TEMPLATE_CSV)

    pares = {tuple(sorted([a, b])) for a, b in zip(csv_card["fighter_a"],
                                                   csv_card["fighter_b"])}
    if not pares:
        raise IdentidadeInconsistente("o CSV do card esta vazio")

    def _pares(df):
        return {tuple(sorted([a, b])) for a, b in zip(df["fighter_a"], df["fighter_b"])}

    no_hist = h[h.apply(lambda r: tuple(sorted([r["fighter_a"], r["fighter_b"]]))
                        in pares, axis=1)]
    if no_hist.empty:
        raise IdentidadeInconsistente(
            "nenhuma luta do CSV aparece no historico -- card nao registrado")

    chaves = no_hist[["event_name", "event_date"]].drop_duplicates()
    if len(chaves) > 1:
        raise IdentidadeInconsistente(
            f"as lutas do CSV aparecem em {len(chaves)} eventos: "
            f"{sorted(map(tuple, chaves.to_numpy()))}")
    nome, data = str(chaves.iloc[0]["event_name"]), str(chaves.iloc[0]["event_date"])

    faltando_hist = pares - _pares(no_hist)
    if faltando_hist:
        raise IdentidadeInconsistente(
            f"{len(faltando_hist)} luta(s) do CSV fora do historico: "
            f"{sorted(faltando_hist)[:3]}")

    do_dia_t = t[t["event_date"].astype(str) == data]
    faltando_t = pares - _pares(do_dia_t)
    if faltando_t:
        raise IdentidadeInconsistente(
            f"{len(faltando_t)} luta(s) do CSV fora do template em {data}: "
            f"{sorted(faltando_t)[:3]}")

    caminho_estado = estado or (config.DATA_DIR / "ciclo_estado.json")
    if caminho_estado.exists():
        try:
            e = json.loads(caminho_estado.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            e = None
        if e and (str(e.get("data")) != data or str(e.get("nome")) != nome):
            raise IdentidadeInconsistente(
                f"estado local diz {e.get('nome')!r} ({e.get('data')}) e o CSV "
                f"e de {nome!r} ({data})")

    return {"event_name": nome, "event_date": data, "n_lutas": len(pares)}


def preencher_event_uid(history_df: pd.DataFrame,
                        fights_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Preenche `event_uid` com a URL do evento no UFCStats.

    A identidade do evento e o ID; a DATA e atributo. Duas coisas que a data
    nao resolve: dois eventos no mesmo dia, e titulo que muda (em 07/nov o
    evento foi renomeado quando o main event saiu do card).

    So preenche quando TODAS as lutas da data apontam para o mesmo
    event_url na base -- data com mais de um evento fica PENDENTE, nao
    arbitrada. Evento que ainda nao entrou na base tambem fica pendente.
    """
    if fights_df is None:
        fights_df = pd.read_csv(config.RAW_FIGHTS_CSV,
                                usecols=["event_url", "event_date"])
    b = fights_df.copy()
    b["_data"] = b["event_date"].astype(str).str[:10]
    por_data = b.groupby("_data")["event_url"].agg(lambda x: set(x.dropna()))

    out = history_df.copy()
    if "event_uid" not in out.columns:
        out["event_uid"] = pd.Series([np.nan] * len(out), index=out.index, dtype="object")
    out["event_uid"] = out["event_uid"].astype("object")

    n, pendentes = 0, []
    for data, grupo in out.groupby(out["event_date"].astype(str).str[:10]):
        if grupo["event_uid"].notna().all():
            continue
        urls = por_data.get(data, set())
        if len(urls) == 1:
            out.loc[grupo.index, "event_uid"] = next(iter(urls))
            n += len(grupo)
        else:
            pendentes.append((data, len(urls)))
    for data, q in pendentes:
        logger.info("event_uid pendente em %s: %d evento(s) na base nesta data.",
                    data, q)
    logger.info("event_uid: %d linha(s) preenchidas, %d data(s) pendentes.",
                n, len(pendentes))
    return out


def identificacao_inequivoca(row) -> bool:
    """
    Esta linha pode entrar em operacao que exija identidade INEQUIVOCA?

    Exige `fight_uid` E casamento LITERAL. Os nove `ufcstats_fuzzy` ficam de
    fora: o uid deles saiu da propria correspondencia aproximada, entao a
    existencia da URL nao prova o casamento -- ela e consequencia dele.
    """
    return bool(pd.notna(row.get("fight_uid"))
                and row.get("uid_origem") == UID_EXATO)


# Motivos de bloqueio de uma perna. Texto curto porque vai para log e para a
# pagina; o detalhe fica aqui.
BLOQUEIO_IDENTIDADE = "identidade nao verificavel"
BLOQUEIO_COTACAO = "cotacao sem prova"
BLOQUEIO_ESTADO = "estado nao verificado"
BLOQUEIO_CASAMENTO = "casamento confronto/cotacao nao comprovado"
BLOQUEIO_ORGANIZACAO = "organizacao nao confirmada"
BLOQUEIO_ATUALIDADE = "cotacao nao atual ou nao pre-luta"
BLOQUEIO_PRECO_ENTRADA = "preco de entrada sem prova"
BLOQUEIO_POS_INICIO = "previsao criada apos o inicio da luta"


def motivo_de_bloqueio(row, *, exigir_atual: bool = False) -> str | None:
    """
    Por que esta linha NAO pode entrar numa selecao — ou None se pode.

    Selecao aqui e qualquer uso que proponha uma ENTRADA: marcar perna EV>1
    como oportunidade, destacar sinal sharp no card, montar lista de apostas.
    Nao e o mesmo que entrar numa metrica historica: metrica descreve o que
    foi registrado, selecao propoe uma acao agora, e so a segunda exige
    prova. Nenhuma previsao historica e reescrita por causa deste gate.

    Seis exigencias, todas verificaveis a partir do que esta gravado:

      estado        nao pode ser `retirado_do_card`, `nao_aconteceu` nem
                    `desconhecido` -- luta que saiu do card nao e entrada;
      identidade    `fight_uid` presente E casamento literal. Os nove
                    `ufcstats_fuzzy` ficam de fora: o uid deles saiu da
                    propria correspondencia aproximada;
      organizacao   o confronto tem de estar no card do UFCStats daquela data
                    (`fonte_card` em {ambas, so_ufcstats}). O feed de odds e
                    de MMA, nao de UFC, e numero de casas nao prova nada;
      casamento     a cotacao tem de declarar a evidencia que ligou o preco
                    aos DOIS participantes, ao evento e a orientacao dos
                    lados (`casamento`, `evento_id_api`,
                    `participantes_fonte`);
      cotacao       `sharp_ref` presente e conferivel contra o corpo guardado;
      atualidade    com `exigir_atual`, o preco tem de estar dentro da
                    validade operacional e ter sido lido ANTES do inicio
                    programado (ver proveniencia.VALIDADE_HORAS).

    `exigir_atual` e False por padrao de proposito: a maior parte dos
    chamadores descreve o passado, onde "atual" nao quer dizer nada. Quem
    propoe uma entrada AGORA passa True.

    Linha de evento FUTURO nao tem `fight_uid` -- a luta ainda nao esta na
    base de resultados --, entao a exigencia de identidade interna so vale
    onde pode ser satisfeita. Para `aguardando` o que identifica o confronto
    e a prova EXTERNA: card do UFCStats mais evidencia de casamento. Isso
    atende identificacao antes da luta sem inventar URL de resultado que
    ainda nao existe.
    """
    import pandas as pd

    from src.casamento import FORTES, organizacao_confirmada
    from src.proveniencia import preco_verificavel, validar_cotacao
    from src.snapshots import criada_apos_o_inicio

    def vazio(v) -> bool:
        return (v is None or (isinstance(v, float) and pd.isna(v))
                or not str(v).strip())

    estado = row.get("estado_resultado")
    if estado in (RETIRADO_DO_CARD, NAO_ACONTECEU, DESCONHECIDO):
        return BLOQUEIO_ESTADO
    # A dispensa de identidade interna vale para luta que AINDA NAO
    # aconteceu -- ela nao tem fight_uid porque nao esta na base de
    # resultados. Uma linha com vencedor preenchido nao se encaixa nisso,
    # mesmo que o estado tenha ficado desatualizado em `aguardando`: confiar
    # so na string deixaria uma linha resolvida escapar da exigencia.
    resolvida = not vazio(row.get("actual_winner"))
    if (estado != AGUARDANDO or resolvida) and not identificacao_inequivoca(row):
        return BLOQUEIO_IDENTIDADE
    # Previsao criada DEPOIS de a luta comecar nao e previsao. A versao fica
    # registrada -- o arquivo nao esconde linha --, mas nenhuma entrada se
    # apoia nela. A conta e POR LUTA: num card as lutas comecam ao longo de
    # cinco horas, e usar o horario do evento liberaria uma previsao feita
    # depois do primeiro combate.
    if criada_apos_o_inicio(row.get("prediction_id")):
        return BLOQUEIO_POS_INICIO
    if not organizacao_confirmada(row.get("fonte_card")):
        return BLOQUEIO_ORGANIZACAO
    # O casamento tem de estar DECLARADO e FORTE nos dois lados. Linha antiga,
    # gravada antes destas colunas, nao ganha aprovacao retroativa.
    if any(vazio(row.get(c)) for c in
           ("evento_id_api", "casamento", "participantes_fonte")):
        return BLOQUEIO_CASAMENTO
    lados = str(row.get("casamento")).split("+")
    if len(lados) != 2 or any(l not in FORTES for l in lados):
        return BLOQUEIO_CASAMENTO
    if not preco_verificavel(row.get("sharp_ref")):
        return BLOQUEIO_COTACAO
    # O PRECO QUE DETERMINA EV E RETORNO e a odd da casa, nao a da fonte
    # sharp. A prova de uma nao cobre a outra: vem de fonte diferente, por
    # caminho diferente. Perna sem odd da casa cai na mediana, que nao tem
    # prova por preco nenhuma -- entao tambem nao sustenta entrada.
    #
    # E a prova e a DO LADO ESCOLHIDO. A conferencia antiga olhava sempre o
    # lado A: uma perna no lado B passava apresentando a cotacao do
    # adversario, que vem da mesma chamada e do mesmo corpo, so com o lado
    # trocado. Agora a decisao e conferida contra o que afirma -- confronto,
    # orientacao, casa, mercado, odd e status -- e a odd tem de aparecer no
    # corpo persistido.
    lado = row.get("model_side")
    if vazio(lado):
        lado = row.get("fighter_a")
    lado_a = str(lado).strip() == str(row.get("fighter_a")).strip()
    odd = row.get("odds_a_tp") if lado_a else row.get("odds_b_tp")
    ref = row.get("ref_tp") if lado_a else row.get("ref_tp_b")
    if vazio(odd):
        return BLOQUEIO_PRECO_ENTRADA
    prova = dict(casa="Thunderpick", mercado="h2h", lado=lado, odd=odd,
                 participantes=(row.get("fighter_a"), row.get("fighter_b")))
    if not validar_cotacao(ref, **prova)[0]:
        return BLOQUEIO_PRECO_ENTRADA
    if exigir_atual:
        if not preco_verificavel(row.get("sharp_ref"), exigir_atual=True):
            return BLOQUEIO_ATUALIDADE
        if not validar_cotacao(ref, exigir_atual=True, **prova)[0]:
            return BLOQUEIO_ATUALIDADE
    return None


def elegivel_para_selecao(row, *, exigir_atual: bool = False) -> bool:
    return motivo_de_bloqueio(row, exigir_atual=exigir_atual) is None
