"""
src/casamento.py

Casamento inequívoco entre um confronto nosso e uma cotação de fonte externa.

O que estava errado
-------------------
`line_shopping.match_event` percorria os eventos da The Odds API e devolvia
**o primeiro** cujos dois nomes casassem por correspondência aproximada. Cinco
modos de falha, todos silenciosos — a linha saía com preço e aparência de
saudável:

  1. **primeiro candidato ganha.** Dois eventos plausíveis na lista e ele
     escolhia um sem contar que havia outro;
  2. **nenhuma conferência de data.** Revanche é o caso óbvio: os mesmos dois
     nomes em eventos diferentes casam igualmente bem, e o preço de um
     evento entrava na linha do outro;
  3. **nenhuma conferência de organização.** O feed é `mma_mixed_martial_arts`,
     não UFC: em 05/nov/2026 trouxe onze lutas de outras organizações para a
     data do card;
  4. **fuzzy aprovava sozinho.** `best_name_match` existe para *sugerir*
     grafia; usá-lo como prova de identidade já casou "Cameron Nelson" com
     "Cameron Else" e inverteu o lado apontado numa perna EV>1;
  5. **homônimo.** Nome exato não é identidade: a base tem nove nomes
     repetidos, entre eles `Jean Silva` e `Bruno Silva`, ambos ativos.

O que este módulo faz
---------------------
Uma função, `casar()`, que devolve um objeto com **veredito e evidência** em
vez de um evento solto. Ela aprova um casamento somente quando, ao mesmo
tempo:

  - existe **exatamente um** evento candidato na janela de data;
  - os dois participantes aparecem como os dois lados de um mercado `h2h`
    daquele evento, o que prova de uma vez o mercado e a **orientação dos
    lados**;
  - a identidade de cada lado tem evidência **forte**: igualdade literal ou
    apelido confirmado à mão. Correspondência aproximada entra como
    `fuzzy` e **sugere sem aprovar**;
  - nenhum dos dois nomes é homônimo conhecido sem desambiguação externa.

Qualquer outro caso é bloqueio **com motivo explícito**, nunca silêncio.

O que este módulo NÃO prova
---------------------------
**Pertencimento ao UFC.** A The Odds API não informa organização, e número de
casas não é prova: é correlação, e o próprio filtro de `build_card` usa isso
apenas para decidir *inclusão no CSV*. A prova de organização vem de o
confronto estar no card do UFCStats daquela data — `fonte` em
`{ambas, so_ufcstats}` — e é conferida por `organizacao_confirmada()`,
separadamente, porque é uma pergunta sobre o CARD e não sobre a cotação.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import config
from src.utils import best_name_match

logger = logging.getLogger(__name__)

# Um card atravessa a meia-noite UTC: o main event de um sabado a noite nos
# EUA tem commence_time no domingo. Entao a janela e o dia do evento mais o
# seguinte -- larga o bastante para o card inteiro, estreita o bastante para
# nao alcancar o evento da semana seguinte.
JANELA_DIAS = 1

# Evidencia de identidade, da mais forte para a mais fraca.
EXATO = "exato"                      # igualdade literal do nome
ALIAS = "alias_confirmado"           # tabela conferida a mao, nos dois lados
FUZZY = "fuzzy"                      # difflib: sugere, nao aprova
FORTES = (EXATO, ALIAS)

# Motivos de bloqueio. Texto curto porque vai para o log e para a pagina.
AUSENTE = "confronto ausente na fonte"
MULTIPLOS = "multiplos candidatos para o confronto"
SO_APROXIMADO = "casamento apenas aproximado"
HOMONIMO = "homonimo sem desambiguacao"
SEM_MERCADO = "sem mercado h2h com os dois participantes"
FORA_DA_JANELA = "nenhum candidato na janela de data"

# Nomes que a base de lutadores tem REPETIDOS. Para estes, igualdade literal
# nao e identidade: ha duas pessoas com o nome, e casar pelo texto escolhe
# uma delas por sorte. Exige-se desambiguacao externa (a URL do UFCStats do
# lutador, que existe antes da luta).
#
# A lista e derivada da base, nao escrita a mao -- ver `homonimos_da_base()`.
# Fica fixa aqui para que o bloqueio nao dependa de ler um CSV de 10 MB a
# cada chamada, e `test_homonimos_conferem_com_a_base` falha se a base
# ganhar um nome repetido novo.
HOMONIMOS = frozenset({
    "Mike Davis", "Anthony Figueroa", "Joey Gomez", "Tony Johnson",
    "Michael McDonald", "Anthony Romero", "Jean Silva", "Bruno Silva",
    "Victor Valenzuela",
})


def homonimos_da_base(fighters_csv=None) -> frozenset:
    """Nomes que aparecem em mais de uma ficha de lutador na base."""
    import pandas as pd
    f = pd.read_csv(fighters_csv or config.RAW_FIGHTERS_CSV, usecols=["name"])
    n = f["name"].value_counts()
    return frozenset(n[n > 1].index)


@dataclass
class Casamento:
    """
    Veredito de um casamento, com a evidência que o sustenta.

    `ok` é True somente quando a cotação pode ser usada numa seleção. Quando
    é False, `motivo` diz por quê e `candidatos` diz quantos eventos
    plausíveis havia — um bloqueio por `MULTIPLOS` com dois candidatos é
    informação diferente de um bloqueio por `AUSENTE` com zero.

    `evento` carrega o dict bruto da API para quem precisa dos preços; ele
    NÃO é persistido. O que se persiste é `evento_id` (o id externo),
    `nome_a`/`nome_b` (como a fonte chama cada lado, que é a prova de
    orientação) e as duas evidências.
    """
    ok: bool
    motivo: str | None = None
    evento_id: str | None = None
    commence_time: str | None = None
    nome_a: str | None = None
    nome_b: str | None = None
    evidencia_a: str | None = None
    evidencia_b: str | None = None
    candidatos: int = 0
    evento: dict | None = field(default=None, repr=False)

    @property
    def evidencia(self) -> str:
        """Resumo persistível: `exato+alias_confirmado`, por exemplo."""
        return f"{self.evidencia_a or '-'}+{self.evidencia_b or '-'}"

    def __bool__(self) -> bool:
        return self.ok


def _data_do_evento(ev: dict) -> date | None:
    bruto = ev.get("commence_time") or ""
    try:
        return datetime.fromisoformat(bruto.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _na_janela(ev: dict, event_date: str | None) -> bool:
    """O evento da fonte cai na janela do nosso evento?

    Sem `event_date` NÃO se conclui que está na janela: conferir data é o que
    separa revanche de revanche, e um chamador que não informa a data não
    ganha a conferência de graça. Ele ganha o bloqueio.
    """
    if not event_date:
        return False
    quando = _data_do_evento(ev)
    if quando is None:
        return False
    nossa = date.fromisoformat(str(event_date)[:10])
    return 0 <= (quando - nossa).days <= JANELA_DIAS


def _lados_h2h(ev: dict) -> list[str]:
    """
    Nomes que aparecem como lados de um mercado `h2h` do evento.

    Lido dos OUTCOMES, não de `home_team`/`away_team`: é o outcome que carrega
    o preço, então casar com ele é o que prova que a cotação pertence àquele
    participante. Um evento listado sem mercado h2h não serve para cotar.
    """
    nomes: list[str] = []
    for bm in ev.get("bookmakers", []):
        for mercado in bm.get("markets", []):
            if mercado.get("key") != "h2h":
                continue
            for o in mercado.get("outcomes", []):
                nome = o.get("name")
                if nome and nome not in nomes:
                    nomes.append(nome)
    return nomes


def _evidencia(nosso: str, lados: list[str], apelidos: dict) -> tuple[str | None, str | None]:
    """
    (nome_na_fonte, evidência) para um lado nosso, ou (None, None).

    A ordem é a força da prova: literal, depois apelido confirmado, e só
    então aproximado — que volta marcado como `FUZZY` para o chamador
    bloquear.
    """
    if nosso in lados:
        return nosso, EXATO
    apelido = apelidos.get(nosso)
    if apelido and apelido in lados:
        return apelido, ALIAS
    aproximado = best_name_match(nosso, lados)
    if aproximado:
        return aproximado, FUZZY
    return None, None


def casar(fighter_a: str, fighter_b: str, events: list[dict], *,
          event_date: str | None, apelidos: dict | None = None,
          desambiguado: bool = False) -> Casamento:
    """
    Casa um confronto nosso com UM evento da fonte, ou bloqueia com motivo.

    `event_date` é obrigatório como palavra-chave e não tem default útil: sem
    data não há como separar uma revanche da luta original, e o resultado é
    bloqueio por `FORA_DA_JANELA`.

    `desambiguado=True` declara que a identidade dos dois lados já foi fixada
    por um id externo (a URL do UFCStats do lutador, por exemplo), o que
    libera a trava de homônimo. Quem passa isso tem de ter o id de verdade —
    o parâmetro é uma afirmação do chamador, e o teste
    `test_homonimo_sem_id_externo_bloqueia` guarda o caminho em que ele não
    é passado.
    """
    from src.line_shopping import APELIDOS_API
    apelidos = APELIDOS_API if apelidos is None else apelidos

    na_janela = [ev for ev in events if _na_janela(ev, event_date)]
    if not na_janela:
        return Casamento(False, FORA_DA_JANELA, candidatos=0)

    candidatos: list[Casamento] = []
    sem_mercado = 0
    for ev in na_janela:
        lados = _lados_h2h(ev)
        nome_a, ev_a = _evidencia(fighter_a, lados, apelidos)
        nome_b, ev_b = _evidencia(fighter_b, lados, apelidos)
        if not (nome_a and nome_b) or nome_a == nome_b:
            continue
        if not lados:
            sem_mercado += 1
            continue
        candidatos.append(Casamento(
            True, None, evento_id=str(ev.get("id") or ""),
            commence_time=ev.get("commence_time"),
            nome_a=nome_a, nome_b=nome_b,
            evidencia_a=ev_a, evidencia_b=ev_b, candidatos=1, evento=ev))

    if not candidatos:
        return Casamento(False, SEM_MERCADO if sem_mercado else AUSENTE,
                         candidatos=0)
    if len(candidatos) > 1:
        # Dois eventos plausiveis na mesma janela: nao se escolhe, se bloqueia.
        melhor = candidatos[0]
        return Casamento(False, MULTIPLOS, candidatos=len(candidatos),
                         evento_id=melhor.evento_id,
                         commence_time=melhor.commence_time)

    c = candidatos[0]
    # Fuzzy sugere e nao aprova: o candidato volta IDENTIFICADO, para quem
    # quiser confirmar a mao, e marcado como bloqueado.
    if c.evidencia_a not in FORTES or c.evidencia_b not in FORTES:
        return Casamento(False, SO_APROXIMADO, candidatos=1,
                         evento_id=c.evento_id, commence_time=c.commence_time,
                         nome_a=c.nome_a, nome_b=c.nome_b,
                         evidencia_a=c.evidencia_a, evidencia_b=c.evidencia_b,
                         evento=c.evento)
    # Nome exato nao e identidade quando ha duas pessoas com o nome.
    if not desambiguado and (fighter_a in HOMONIMOS or fighter_b in HOMONIMOS):
        return Casamento(False, HOMONIMO, candidatos=1,
                         evento_id=c.evento_id, commence_time=c.commence_time,
                         nome_a=c.nome_a, nome_b=c.nome_b,
                         evidencia_a=c.evidencia_a, evidencia_b=c.evidencia_b,
                         evento=c.evento)
    return c


# ---------------------------------------------------------------------------
# Organizacao: pergunta sobre o CARD, nao sobre a cotacao
# ---------------------------------------------------------------------------

ORG_NAO_CONFIRMADA = "organizacao nao confirmada (so a API conhece a luta)"


def organizacao_confirmada(fonte) -> bool:
    """
    O confronto está no card do UFCStats daquela data?

    `fonte` vem de `build_card`: `ambas` e `so_ufcstats` significam que o
    UFCStats lista a luta naquele evento, o que é a prova de que é UFC.
    `so_api` significa que só o feed de MMA a conhece — pode ser qualquer
    organização, e **número de casas não resolve isso**: é correlação, usada
    em `build_card` apenas para decidir se a linha entra no CSV, nunca como
    prova de pertencimento.

    Ausência da coluna conta como NÃO confirmada. Um CSV antigo, escrito
    antes de `fonte` ser persistida, não ganha prova retroativa.
    """
    import pandas as pd
    if fonte is None or (isinstance(fonte, float) and pd.isna(fonte)):
        return False
    return str(fonte).strip() in ("ambas", "so_ufcstats")
