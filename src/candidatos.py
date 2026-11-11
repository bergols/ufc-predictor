"""
src/candidatos.py

Registro **append-only** dos candidatos congelados.

Por que existe
--------------
`vantagem-v1` deixou duas coisas implícitas que esta rodada torna explícitas:

1. **"prospectivo" exige prova positiva.** O v1 classificava como avaliação
   prospectiva todo evento que não estivesse na lista de desenvolvimento. Isso
   é ausência de contra-evidência, não prova: um evento pode estar fora da
   lista porque ninguém o olhou, ou porque o candidato ainda não existia quando
   a decisão foi tomada. Para uma observação valer como prospectiva é preciso
   que o candidato estivesse **congelado antes** da decisão — e isso só se sabe
   se houver um instante de congelamento registrado.

2. **quando uma versão nova pode ser estimada.** Manter `w = 0` congelado deixa
   C idêntico a A, e acumular eventos não muda esse contraste: ele é zero por
   construção. A regra de reestimação tem de ser escrita antes, senão a decisão
   de reestimar vai ser tomada olhando o resultado.

Cada candidato guarda: id, instante de congelamento, versão e sha256 do
protocolo, peso `w`, os eventos usados para estimar, os eventos já consultados
em análise inferencial, e a regra de reestimação em vigor.

O arquivo **só recebe linhas**. Candidato congelado não é editado: uma mudança
de peso, de protocolo ou de população é um candidato **novo**, com instante
próprio, e o antigo continua endereçável com o resultado que teve.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import config

logger = logging.getLogger(__name__)

ARQUIVO = config.DATA_DIR / "experimentos" / "candidatos.json"

# Regra de reestimacao, fixada no vantagem-v2 ANTES dos proximos resultados.
REGRA_DE_REESTIMACAO = (
    "O peso w so pode ser reestimado quando houver pelo menos 10 eventos "
    "registrados em modo sombra que NUNCA tenham entrado em estimacao nem em "
    "analise inferencial. A reestimacao transforma esses eventos em "
    "DESENVOLVIMENTO: eles passam para `eventos_desenvolvimento` do candidato "
    "novo e nao voltam a servir de avaliacao. O candidato novo so e avaliado em "
    "eventos POSTERIORES ao seu congelamento. Nao se faz teste de "
    "significancia a cada evento: a analise inferencial roda nas datas "
    "previamente definidas, e o acompanhamento entre elas e DESCRITIVO."
)

# Quando a analise inferencial pode rodar. Entre as datas o relatorio e
# descritivo: olhar o p-valor a cada evento ate ele ficar favoravel e o mesmo
# erro de afrouxar limiar depois de ver o resultado.
MARCOS_INFERENCIAIS = (
    "a cada 20 eventos de avaliacao acumulados no candidato em vigor, contados "
    "pelos eventos que ele qualificou como avaliacao. Fora desses marcos, so "
    "relatorio descritivo: contagem, cobertura e classificacao temporal, sem "
    "teste e sem intervalo de confianca apresentado como decisao."
)


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_arquivo(caminho: Path) -> str:
    try:
        corpo = Path(caminho).read_bytes().replace(b"\r\n", b"\n")
        return hashlib.sha256(corpo).hexdigest()
    except OSError:
        return ""


def carregar(arquivo: Path | None = None) -> list[dict]:
    caminho = Path(arquivo or ARQUIVO)
    if not caminho.exists():
        return []
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
    except ValueError:
        logger.warning("%s ilegivel: tratando como vazio.", caminho)
        return []
    return dados if isinstance(dados, list) else []


def congelar(*, versao_protocolo: str, peso_w: float,
             protocolo_path: Path | None = None,
             eventos_desenvolvimento: list[str] | None = None,
             eventos_consultados: list[str] | None = None,
             descricao: str = "", arquivo: Path | None = None,
             congelado_em: str | None = None,
             manifesto_modelo_sha256: str = "") -> dict:
    """
    Congela um candidato e devolve o registro. Só anexa.

    `eventos_consultados` são os eventos que já entraram em análise
    inferencial — eles não podem voltar como avaliação, e estão separados de
    `eventos_desenvolvimento` porque são coisas diferentes: um treinou o
    parâmetro, o outro gastou a inferência.

    `manifesto_modelo_sha256` amarra o candidato ao modelo cujas previsões
    ele combina (sha256 de models/training_metadata.json, o mesmo que vai em
    cada versão de previsoes_snapshots.csv). Sem ele a sombra não qualifica
    a decisão: depois do retreino de 11/nov/2026 o candidato em vigor teria
    passado a combinar previsões de um modelo que não existia quando foi
    congelado.
    """
    caminho = Path(arquivo or ARQUIVO)
    quando = congelado_em or _agora()
    protocolo = Path(protocolo_path or (config.PROJECT_ROOT /
                                        "PROTOCOLO_VANTAGEM_V2.md"))
    protocolo_sha = sha256_arquivo(protocolo)
    if not protocolo_sha:
        raise ValueError(f"Protocolo indisponivel: {protocolo}")
    reg = {
        "candidato_id": "",
        "congelado_em": quando,
        "versao_protocolo": versao_protocolo,
        "protocolo_arquivo": protocolo.name,
        "protocolo_sha256": protocolo_sha,
        "peso_w": float(peso_w),
        "eventos_desenvolvimento": sorted(eventos_desenvolvimento or []),
        "eventos_consultados": sorted(eventos_consultados or []),
        "regra_de_reestimacao": REGRA_DE_REESTIMACAO,
        "marcos_inferenciais": MARCOS_INFERENCIAIS,
        "descricao": descricao,
    }
    chaves = ["congelado_em", "versao_protocolo", "protocolo_sha256", "peso_w",
              "eventos_desenvolvimento", "eventos_consultados"]
    if manifesto_modelo_sha256:
        reg["manifesto_modelo_sha256"] = manifesto_modelo_sha256
        chaves.append("manifesto_modelo_sha256")
    semente = json.dumps({k: reg[k] for k in chaves},
                         sort_keys=True, ensure_ascii=False)
    reg["candidato_id"] = hashlib.sha256(semente.encode("utf-8")).hexdigest()[:16]

    atuais = carregar(caminho)
    if any(c.get("candidato_id") == reg["candidato_id"] for c in atuais):
        logger.info("Candidato %s ja estava congelado.", reg["candidato_id"])
        return reg
    try:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(
            json.dumps(atuais + [reg], indent=2, ensure_ascii=False),
            encoding="utf-8")
    except OSError as exc:
        logger.warning("Nao consegui gravar o candidato (%s).", exc)
        return reg
    logger.info("Candidato %s congelado em %s (w = %s).",
                reg["candidato_id"], quando, reg["peso_w"])
    return reg


def atual(arquivo: Path | None = None) -> dict | None:
    """O último candidato congelado, ou None quando não há nenhum."""
    todos = carregar(arquivo)
    return todos[-1] if todos else None


def por_id(candidato_id: str, arquivo: Path | None = None) -> dict | None:
    pid = str(candidato_id or "").strip()
    for c in carregar(arquivo):
        if c.get("candidato_id") == pid:
            return c
    return None


def eventos_queimados(candidato: dict) -> set[str]:
    """
    Eventos que **não podem** valer como avaliação para este candidato:
    os que estimaram o parâmetro e os que já gastaram inferência.
    """
    return ({str(e) for e in candidato.get("eventos_desenvolvimento", [])}
            | {str(e) for e in candidato.get("eventos_consultados", [])})
