"""
src/temporalidade.py

Classificação temporal **por observação**: o que foi comprovado, o que é
desconhecido e o que é incompatível.

Por que existe
--------------
`vantagem-v1` chamou 179 lutas de "população temporalmente adequada" porque
elas constavam do pré-registro. Isso confunde **disponibilidade do dado** com
**comprovação temporal**. Constar de um registro escrito antes do evento é um
bom indício de que a linha é pré-luta; não é prova de que *aquela* previsão foi
criada antes daquela luta, nem de que *aquele* preço estava disponível naquele
instante.

A errata está em `ERRATA_VANTAGEM_V1.md`. Este módulo é o conserto.

O que conta como prova, e o que não conta
-----------------------------------------
Conta: um instante registrado numa fonte verificável, comparável ao início
programado **daquela luta**.

  - previsão: `criado_em` da versão em `data/previsoes_snapshots.csv`;
  - preço: `captured_at` da linha em `data/proveniencia/cotacoes.csv`, com o
    corpo da chamada conferindo pelo hash.

**Não** conta, e cada um destes já foi usado como se contasse:

  - a data do evento. Ela diz quando a luta aconteceu, não quando escrevemos;
  - a coluna de odds estar preenchida. Preço presente não é preço comprovado, e
    muito menos preço executável comprovado;
  - a linha existir num arquivo cujo propósito é ser pré-registro. O propósito
    do arquivo não é evidência sobre a linha;
  - a ausência de contra-evidência.

As quatro classes
-----------------
    comprovada      há instante verificável e ele é anterior ao início daquela
                    luta;
    desconhecida    não há instante verificável, ou não há início programado
                    para comparar. **Não é o mesmo que incompatível**;
    incompativel    há instante verificável e ele é POSTERIOR ao início;
    ausente         o item não existe (não há preço, não há referência).

`nao_certificada` é um rótulo de fonte, não de linha: vale para a mediana de
casas, que não tem proveniência por preço nenhuma — não há o que conferir por
linha, e o problema não é falta de horário, é falta de origem.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

COMPROVADA = "comprovada"
DESCONHECIDA = "desconhecida"
INCOMPATIVEL = "incompativel"
AUSENTE = "ausente"
NAO_CERTIFICADA = "nao_certificada"

MOTIVO_MEDIANA = (
    "a mediana de 20-28 casas NAO tem proveniencia por preco: nao ha casa, "
    "mercado, instante de leitura nem corpo persistido por linha. A coluna "
    "estar preenchida nao e prova de captura pre-luta, e a mediana nao e "
    "cotacao executavel. Fonte nao certificavel, nao linha sem horario."
)


@dataclass(frozen=True)
class Evidencia:
    """O que foi efetivamente encontrado, não o que se esperava encontrar."""
    classe: str
    instante: str = ""      # o instante verificável, quando existe
    referencia: str = ""    # onde ele foi encontrado
    inicio: str = ""        # o início programado contra o qual foi comparado
    detalhe: str = ""

    def como_dict(self) -> dict:
        return asdict(self)

    def __bool__(self) -> bool:
        return self.classe == COMPROVADA


def _texto(v) -> str:
    try:
        import pandas as pd
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
    except ImportError:  # pragma: no cover
        if v is None:
            return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none") else s


def _instante(x) -> datetime | None:
    s = _texto(x)
    if not s:
        return None
    try:
        valor = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return valor if valor.utcoffset() is not None else None
    except ValueError:
        return None


def inicio_programado(*refs, arquivo: Path | None = None) -> tuple[str, str]:
    """
    (início programado, de onde veio) a partir das referências de cotação.

    É o único horário de luta que o projeto registra numa fonte verificável. A
    data do evento **não** serve: ela não diz a que hora o combate começou, e
    num card as lutas começam ao longo de cinco horas.
    """
    from src.proveniencia import INDICE_COTACOES, _linha_de_cotacao
    for ref in refs:
        r = _texto(ref)
        if not r:
            continue
        linha = _linha_de_cotacao(r)
        if linha and _texto(linha.get("inicio_programado")):
            return _texto(linha["inicio_programado"]), f"cotacoes.csv:{r}"
    return "", ""


def da_previsao(prediction_id, *, inicio: str = "",
                arquivo: Path | None = None) -> Evidencia:
    """
    A previsão foi criada antes do início **daquela luta**?

    Só o `criado_em` da versão responde. Versão semeada não tem esse campo — e
    isso é `desconhecida`, nunca `comprovada` por estar no pré-registro.
    """
    from src.snapshots import versao as buscar_versao
    pid = _texto(prediction_id)
    if not pid:
        return Evidencia(DESCONHECIDA, detalhe=(
            "o registro nao aponta para nenhuma versao da previsao: nao ha "
            "instante de criacao para conferir"))
    v = buscar_versao(pid, arquivo)
    if v is None:
        return Evidencia(DESCONHECIDA, referencia=pid, detalhe=(
            f"a versao {pid} nao esta no arquivo de versoes"))
    criado = _texto(v.get("criado_em"))
    if not criado:
        return Evidencia(DESCONHECIDA, referencia=pid, inicio=inicio,
                         detalhe=(_texto(v.get("temporalidade")) or
                                  "a versao nao registra criado_em"))
    if not _texto(inicio):
        return Evidencia(DESCONHECIDA, instante=criado, referencia=pid,
                         detalhe=("a versao tem criado_em, mas nao ha inicio "
                                  "programado verificavel para comparar"))
    dc, di = _instante(criado), _instante(inicio)
    if dc is None or di is None:
        return Evidencia(DESCONHECIDA, instante=criado, referencia=pid,
                         inicio=inicio, detalhe="horario ilegivel")
    if dc >= di:
        return Evidencia(INCOMPATIVEL, instante=criado, referencia=pid,
                         inicio=inicio, detalhe=(
                             "a versao foi criada DEPOIS do inicio programado "
                             "daquela luta"))
    return Evidencia(COMPROVADA, instante=criado, referencia=pid, inicio=inicio,
                     detalhe="criado_em da versao anterior ao inicio programado")


def do_preco(ref, *, inicio: str = "", odd=None, lado=None,
             participantes=None, casa: str | None = None,
             mercado: str | None = None, decidido_em=None) -> Evidencia:
    """
    O preço estava disponível, comprovadamente, antes do início daquela luta?

    Exige a linha de cotação **e** que o corpo da chamada confira pelo hash: um
    instante registrado numa linha que não aponta para conteúdo verificável é
    testemunho de si mesma. Quando `odd`/`lado`/`participantes` são passados, a
    linha também tem de descrever aquela afirmação — prova do adversário não é
    prova deste lado.
    """
    from src.proveniencia import _linha_de_cotacao, validar_cotacao
    r = _texto(ref)
    if not r:
        return Evidencia(AUSENTE, detalhe=(
            "nao ha referencia de cotacao: nenhum preco comprovado para este lado"))
    linha = _linha_de_cotacao(r)
    if linha is None:
        return Evidencia(DESCONHECIDA, referencia=r, detalhe=(
            f"a cotacao {r} nao esta no indice de cotacoes"))
    lido = _texto(linha.get("captured_at"))
    inicio = _texto(inicio) or _texto(linha.get("inicio_programado"))

    ok, porque = validar_cotacao(
        r, casa=casa, mercado=mercado, lado=lado, odd=odd,
        participantes=participantes)
    if not ok:
        return Evidencia(DESCONHECIDA, instante=lido, referencia=r,
                         inicio=inicio, detalhe=(
                             f"a prova nao sustenta esta afirmacao: {porque}"))
    if not lido:
        return Evidencia(DESCONHECIDA, referencia=r, inicio=inicio,
                         detalhe="a linha nao registra captured_at")
    if not inicio:
        return Evidencia(DESCONHECIDA, instante=lido, referencia=r, detalhe=(
            "o preco tem instante de leitura, mas nao ha inicio programado "
            "verificavel para comparar"))
    dl, di = _instante(lido), _instante(inicio)
    if dl is None or di is None:
        return Evidencia(DESCONHECIDA, instante=lido, referencia=r,
                         inicio=inicio, detalhe="horario ilegivel")
    if dl >= di:
        return Evidencia(INCOMPATIVEL, instante=lido, referencia=r,
                         inicio=inicio, detalhe=(
                             "o preco foi LIDO depois do inicio programado: "
                             "nao e preco pre-luta"))
    if decidido_em is not None:
        from src.proveniencia import VALIDADE_HORAS

        decisao = _instante(decidido_em)
        inicio_fonte = _instante(linha.get("inicio_programado"))
        if decisao is None or inicio_fonte is None:
            return Evidencia(DESCONHECIDA, lido, r, inicio,
                             "horario da decisao ou da cotacao ilegivel")
        if inicio_fonte != di:
            return Evidencia(INCOMPATIVEL, lido, r, inicio,
                             "inicio programado divergente entre as provas")
        if dl > decisao:
            return Evidencia(INCOMPATIVEL, lido, r, inicio,
                             "cotacao lida depois da decisao")
        if decisao >= di:
            return Evidencia(INCOMPATIVEL, lido, r, inicio,
                             "decisao apos o inicio programado")
        idade = (decisao - dl).total_seconds() / 3600
        if idade >= VALIDADE_HORAS:
            return Evidencia(INCOMPATIVEL, lido, r, inicio,
                             "cotacao vencida no instante da decisao")
    return Evidencia(COMPROVADA, instante=lido, referencia=r, inicio=inicio,
                     detalhe=("captured_at anterior ao inicio programado, com "
                              "corpo persistido conferindo"))


def da_mediana() -> Evidencia:
    """A mediana de casas não é certificável por linha. Sempre o mesmo veredito."""
    return Evidencia(NAO_CERTIFICADA, detalhe=MOTIVO_MEDIANA)


def resumo(evidencias: dict) -> str:
    """`previsao=desconhecida; mediana=nao_certificada; ...` para a saída."""
    return "; ".join(f"{k}={v.classe}" for k, v in evidencias.items())
