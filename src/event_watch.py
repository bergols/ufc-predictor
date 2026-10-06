"""
src/event_watch.py

Alarme de EVENTO PERDIDO: card que acontece sem pre-registro nenhum.

Por que existe
--------------
Havia alarme para fechamento nao capturado (`events_missing_closing`), mas
nenhum para o buraco maior. Se o card inteiro passa sem pre-registro, nao ha
linha no historico — e um alarme que varre o historico nao tem o que varrer.
A serie simplesmente pula a semana e nada reclama.

Foi o que aconteceu com o UFC 332 (03/out/2026): o card passou, ninguem
registrou, e o buraco so apareceu porque fomos conferir os dois fins de semana
na mao. Diferente do fechamento, aqui nem a API ajudaria depois: nao existe
previsao retroativa. Passou, perdeu.

Dois estados, com urgencias opostas — o mesmo molde do alarme de fechamento:

  "a registrar"  a API ja lista o card e nao ha pre-registro. E o UNICO
                 momento acionavel, e e o momento CERTO: a API listar o evento
                 significa que o mercado abriu, que e exatamente quando o
                 README manda registrar (quanto mais cedo, mais movimento de
                 linha o CLV captura).

  "perdido"      o card ja esta na base de resultados e nunca teve
                 pre-registro. Irrecuperavel. Fica listado para o buraco na
                 serie ser explicito, em vez de a semana sumir em silencio.

As duas metades tem fontes diferentes de proposito: "perdido" sai da nossa
propria base de resultados (offline, sem custo, sempre disponivel) e
"a registrar" sai da The Odds API, a unica fonte que enxerga o futuro. O
painel nao faz rede, entao a metade futura chega ate ele pelo cache que o
auto_capture escreve.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)

# A partir de quantos dias antes do card o alarme cobra. Dez dias e mais que a
# antecedencia tipica com que a API lista um card de UFC, entao na pratica ele
# cobra desde que o mercado abre.
JANELA_ALERTA_DIAS = 10

# commence_time vem em UTC, e a data do evento neste projeto e a data LOCAL
# americana do card. Prelims comecam ~22:00 UTC e o card principal ~02:00 UTC
# do dia seguinte: tirar 8h joga os dois para a data americana correta.
_FUSO_CARD_HORAS = 8

# Casamento por data entre a API e o historico aceita um dia de folga: alem do
# fuso, card adiado em cima da hora muda de data sem mudar de identidade.
_TOLERANCIA_DATA_DIAS = 1

# O feed da API e "mma_mixed_martial_arts", nao "ufc": junto com os cards vem
# luta avulsa de outra organizacao e confronto especulativo marcado para 2027
# ou 2028. Medido em 06/out/2026: o card real de 10/out aparecia com 12 lutas;
# o resto da janela era 1 luta solta numa terca. Card de UFC a menos de dez
# dias sempre tem mais que isto listado, entao o corte separa card de ruido
# sem precisar saber o nome do evento -- que a API nao da.
_MIN_LUTAS_CARD = 5

CACHE_PENDENTES = config.DATA_DIR / "eventos_a_registrar.json"

# Cache mais velho que isto nao vale: um alarme de "card amanha" escrito ha
# tres dias nao e informacao, e ruido fossilizado.
_VALIDADE_CACHE_DIAS = 2


def _datas_registradas(history_df: pd.DataFrame) -> set[date]:
    if history_df.empty or "event_date" not in history_df.columns:
        return set()
    d = pd.to_datetime(history_df["event_date"], errors="coerce").dropna()
    return {x.date() for x in d}


def _ja_registrado(dia: date, registradas: set[date]) -> bool:
    return any(abs((dia - r).days) <= _TOLERANCIA_DATA_DIAS for r in registradas)


def eventos_perdidos(history_df: pd.DataFrame,
                     fights_df: pd.DataFrame | None = None,
                     hoje: date | None = None) -> list[dict]:
    """
    Cards que ja aconteceram e nunca tiveram pre-registro.

    So conta evento a partir do PRIMEIRO da serie: antes disso o projeto nao
    registrava nada, e listar 790 eventos historicos como "perdidos" seria
    transformar o alarme em papel de parede.
    """
    registradas = _datas_registradas(history_df)
    if not registradas:
        return []   # a serie nao comecou: nao ha de onde faltar

    if fights_df is None:
        if not config.RAW_FIGHTS_CSV.exists():
            return []
        fights_df = pd.read_csv(config.RAW_FIGHTS_CSV,
                                usecols=["event_date", "event_name"])
    if fights_df.empty:
        return []

    hoje = hoje or date.today()
    inicio = min(registradas)
    base = fights_df.copy()
    base["event_date"] = pd.to_datetime(base["event_date"], errors="coerce")
    base = (base.dropna(subset=["event_date"])
                .drop_duplicates(subset=["event_date", "event_name"])
                .sort_values("event_date"))

    perdidos = []
    for _, r in base.iterrows():
        dia = r["event_date"].date()
        if dia < inicio or dia >= hoje:
            continue
        if _ja_registrado(dia, registradas):
            continue
        perdidos.append({"event_date": dia.isoformat(),
                         "event_name": str(r["event_name"]),
                         "estado": "perdido"})
    return perdidos


def _data_do_card(commence_time) -> date | None:
    try:
        t = datetime.fromisoformat(str(commence_time).replace("Z", "+00:00"))
    except (ValueError, AttributeError, TypeError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return (t - timedelta(hours=_FUSO_CARD_HORAS)).date()


def eventos_a_registrar(api_events: list[dict], history_df: pd.DataFrame,
                        hoje: date | None = None,
                        janela_dias: int = JANELA_ALERTA_DIAS) -> list[dict]:
    """
    Cards que a API ja lista, dentro da janela, e que nao tem pre-registro.

    A API de MMA devolve uma entrada por LUTA, nao por evento: as lutas do
    mesmo card sao agrupadas pela data (ver _data_do_card). O numero de lutas
    listadas entra no resultado porque card com 2 lutas listadas e mercado
    abrindo e card com 12 e mercado formado — muda a urgencia de registrar.
    """
    hoje = hoje or date.today()
    limite = hoje + timedelta(days=janela_dias)
    registradas = _datas_registradas(history_df)

    por_dia: dict[date, int] = {}
    for ev in api_events or []:
        dia = _data_do_card(ev.get("commence_time"))
        if dia is None or dia < hoje or dia > limite:
            continue
        por_dia[dia] = por_dia.get(dia, 0) + 1

    return [{"event_date": dia.isoformat(), "n_lutas": n,
             "dias": (dia - hoje).days, "estado": "a registrar"}
            for dia, n in sorted(por_dia.items())
            if n >= _MIN_LUTAS_CARD and not _ja_registrado(dia, registradas)]


def salvar_pendentes(pendentes: list[dict], hoje: date | None = None,
                     caminho: Path | None = None) -> None:
    """Grava o que a API disse, para o painel ler sem fazer rede.

    Lista vazia TAMBEM e gravada: "verifiquei hoje e nao ha pendencia" e uma
    informacao diferente de "nao verifiquei", e so o carimbo de data distingue
    as duas.
    """
    caminho = Path(caminho or CACHE_PENDENTES)
    caminho.write_text(
        json.dumps({"verificado_em": (hoje or date.today()).isoformat(),
                    "eventos": pendentes}, ensure_ascii=False, indent=2),
        encoding="utf-8")


def precisa_verificar(hoje: date | None = None,
                      caminho: Path | None = None) -> bool:
    """Se a verificacao de hoje ainda nao aconteceu.

    E a comporta de custo: o auto_capture roda de hora em hora, e uma chamada
    por rodada seriam ~720 creditos/mes numa chave gratuita de 500. Uma por
    dia custa ~30 e enxerga o mesmo -- card nao aparece e desaparece da API
    dentro do mesmo dia.
    """
    caminho = Path(caminho or CACHE_PENDENTES)
    if not caminho.exists():
        return True
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        return date.fromisoformat(dados["verificado_em"]) < (hoje or date.today())
    except (json.JSONDecodeError, KeyError, ValueError, OSError, TypeError):
        return True


def carregar_pendentes(hoje: date | None = None,
                       caminho: Path | None = None) -> list[dict]:
    """O que a ultima verificacao achou, se ela ainda vale. Cache ausente,
    ilegivel ou velho devolve lista vazia — o painel nunca inventa alarme."""
    caminho = Path(caminho or CACHE_PENDENTES)
    if not caminho.exists():
        return []
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        verificado = date.fromisoformat(dados["verificado_em"])
    except (json.JSONDecodeError, KeyError, ValueError, OSError, TypeError):
        logger.warning("%s ilegivel -- ignorando o cache de eventos pendentes.", caminho)
        return []

    hoje = hoje or date.today()
    if (hoje - verificado).days > _VALIDADE_CACHE_DIAS:
        return []
    # evento que ja passou desde a verificacao sai: quem cobra card passado e
    # eventos_perdidos, com a base de resultados na mao
    return [e for e in dados.get("eventos", [])
            if str(e.get("event_date", "")) >= hoje.isoformat()]
