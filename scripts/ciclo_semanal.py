"""
scripts/ciclo_semanal.py

O ciclo de evento inteiro, sem ninguém pedir: fecha o card que passou e
registra o próximo.

    python -m scripts.ciclo_semanal                 # faz tudo e dá push
    python -m scripts.ciclo_semanal --dry-run       # relata e não escreve nada
    python -m scripts.ciclo_semanal --no-push       # trabalha local
    python -m scripts.ciclo_semanal --chromium-ufcstats   # UFCStats pelo Chromium

Feito para rodar numa tarefa agendada (ver scripts/ciclo_semanal.bat). É
seguro rodar a qualquer hora, quantas vezes for: em dia sem nada a fazer ele
sai em silêncio, sem tocar em arquivo nem gastar crédito de API.

O que faz, nesta ordem
----------------------
  1. atualiza a base de resultados;
  2. FECHA todo evento passado que ainda está aberto — preenche os vencedores
     a partir da base (src/fechamento.py) e regenera o relatório, que é o que
     sincroniza o histórico;
  3. re-treina, se fechou alguma coisa (previsão com modelo defasado
     desperdiça o evento seguinte);
  4. garante que a PÁGINA mostra o card vigente (o registrado mais próximo
     que ainda não aconteceu);
  5. REGISTRA o próximo card -- mas só quando NÃO há card registrado por
     acontecer. Monta com as duas fontes (scripts/build_card.py) e publica,
     o que congela previsão + sinal sharp;
  6. roda a suíte e só então dá push.

Nunca troca de card antes de o atual acontecer
----------------------------------------------
Em 29/out o ciclo publicou o card de 07/nov com o de 31/out ainda por vir, e
quem foi pegar as entradas do dia encontrou o card errado. Registrar cedo
também sai caro em qualidade: aquele 07/nov entrou com sharp em 0 de 10 e odd
da casa em 2 de 10, porque o mercado mal tinha aberto. "Perto da abertura do
mercado" não é "antes do card anterior acontecer".

Onde ele PARA de propósito
--------------------------
Automatizar aqui só vale se a automação for mais conservadora que a mão, e o
ponto perigoso é um só: escrever vencedor errado numa linha que nunca mais
será revista. Por isso `src/fechamento.py` escreve apenas quando os dois
lutadores casam com lados distintos da luta na base e o vencedor é um deles.
Empate, adversário trocado, luta cancelada e qualquer ambiguidade ficam em
BRANCO, com aviso — exatamente como eu faria.

Validado contra os quatro fechamentos feitos à mão (26/set a 24/out):
43 de 44 linhas idênticas. A única diferença é uma grafia que a base escreve
de outro jeito (Mehemmedeli/Mahammadali Osmanli), onde a automação se recusa
a adivinhar. Essa classe some sozinha: desde o build_card o template já nasce
com a grafia canônica da base.

A suíte roda ANTES do push. Tarefa agendada que publica página quebrada é
pior que tarefa agendada que não publica.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import date

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Nome do evento atual guardado pelo próprio ciclo, para saber com que
# --card-name regenerar o relatório sem adivinhar.
_ESTADO = config.DATA_DIR / "ciclo_estado.json"


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(config.PROJECT_ROOT), *args],
                          capture_output=True, text=True)


# Linha de luta cancelada nunca recebe resultado e fica aberta para sempre.
# Sem uma janela, toda rodada tentaria fechar julho de novo e o log viraria
# papel de parede de avisos que nao mudam. Alem disso, card de 45 dias atras
# ja esta resolvido -- se nao fechou, e buraco permanente, e quem mostra isso
# e o painel.
_JANELA_FECHAMENTO_DIAS = 45


def _eventos_para_fechar(hoje: date) -> list[tuple[str, str]]:
    """[(event_date, event_name)] com previsão registrada, sem resultado, já
    passados e dentro da janela. Do mais antigo para o mais novo."""
    from datetime import timedelta

    from src.prediction_history import load_history
    h = load_history()
    if h.empty:
        return []
    limite = (hoje - timedelta(days=_JANELA_FECHAMENTO_DIAS)).isoformat()
    abertos = h[h["actual_winner"].isna() & h["model_side"].notna()]
    return sorted({(str(r["event_date"]), str(r["event_name"]))
                   for _, r in abertos.iterrows()
                   if limite <= str(r["event_date"]) < hoje.isoformat()})


def _card_vigente(hoje: date) -> tuple[str, str] | None:
    """
    (event_date, event_name) do card que DEVE estar na página: o evento
    registrado mais próximo cuja data ainda não passou. None quando todos os
    registrados já aconteceram.
    """
    from src.prediction_history import load_history
    h = load_history()
    if h.empty:
        return None
    futuros = h[h["event_date"].astype(str) >= hoje.isoformat()]
    if futuros.empty:
        return None
    data = str(futuros["event_date"].min())
    nome = str(futuros[futuros["event_date"].astype(str) == data]["event_name"].iloc[0])
    return data, nome


def _csv_e_do_card(data: str) -> bool:
    """O CSV do card atual descreve o evento desta data?"""
    import pandas as pd
    try:
        card = pd.read_csv(config.UPCOMING_CARD_CSV)
        h = pd.read_csv(config.PREDICTION_HISTORY_CSV)
    except (OSError, pd.errors.EmptyDataError):
        return False
    pares = {tuple(sorted([a, b])) for a, b in zip(card["fighter_a"], card["fighter_b"])}
    do_dia = h[h["event_date"].astype(str) == data]
    return bool(pares) and pares <= {tuple(sorted([a, b]))
                                     for a, b in zip(do_dia["fighter_a"], do_dia["fighter_b"])}


def _proximo_a_registrar(hoje: date, *, dry_run: bool = False):
    """
    Card novo a registrar -- ou None.

    REGRA, de 31/out/2026: nunca registrar enquanto houver card registrado
    que ainda NAO aconteceu. Sem isso o ciclo trocava a pagina para a semana
    seguinte com o card da noite ainda por vir, e quem foi pegar as entradas
    do dia encontrou o card errado. Aconteceu em 29/out: o de 31/out estava
    aberto e ele publicou o de 07/nov.

    Registrar cedo tambem sai caro em qualidade: aquele 07/nov entrou com
    sharp em 0 de 10 e odd da casa em 2 de 10, porque o mercado mal tinha
    aberto. "Perto da abertura" nao e "antes do card anterior acontecer".

    Em `dry_run` nunca consulta API de odds: sem card vigente e sem
    pendente, descobrir o proximo evento exigiria uma captura nova. Relata
    isso e devolve None.
    """
    from src.event_watch import carregar_pendentes, eventos_a_registrar
    from src.line_shopping import fetch_live_odds, get_api_key, sanitizar
    from src.prediction_history import load_history

    vigente = _card_vigente(hoje)
    if vigente is not None:
        logger.info("Card vigente: %s (%s) -- nao registro outro antes dele acontecer.",
                    vigente[1], vigente[0])
        return None

    pendentes = carregar_pendentes(hoje=hoje, history_df=load_history())
    if pendentes:
        return pendentes[0]
    if dry_run:
        logger.info("[dry-run] sem card vigente e sem pendente: inspecionar o "
                    "proximo evento dependeria de captura nova de odds -- parado aqui, "
                    "sem consultar API.")
        return None
    try:
        eventos = fetch_live_odds(get_api_key())
    except Exception as exc:  # noqa: BLE001 - job agendado nunca estoura
        logger.warning("API indisponível (%s) -- sem registrar card novo nesta rodada.",
                       sanitizar(exc))
        return None
    achados = eventos_a_registrar(eventos, load_history(), hoje=hoje)
    return achados[0] if achados else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Fecha o card que passou e registra o próximo.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--sem-testes", action="store_true",
                        help="Não roda a suíte antes do push (não recomendado)")
    parser.add_argument("--chromium-ufcstats", action="store_true",
                        help="Usar o Chromium do Playwright nas leituras do UFCStats")
    args = parser.parse_args()
    if not args.dry_run:
        branch = _git("branch", "--show-current")
        if branch.returncode != 0 or branch.stdout.strip() != "main":
            logger.error("Ciclo operacional exige checkout de main. "
                         "Branch atual: %s. Use --dry-run para inspecionar.",
                         branch.stdout.strip() or "(indisponivel)")
            return 1
    hoje = date.today()
    mexeu = False

    # ---- 1. base ---------------------------------------------------------
    from src.data_collection import (check_data_freshness, download_github_mirror_dataset,
                                     fill_recent_gap_with_browser, merge_manual_recent_fights)
    if not args.dry_run:
        download_github_mirror_dataset()
        try:
            fill_recent_gap_with_browser()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gap-filler falhou (%s) -- seguindo com o espelho.", exc)
        merge_manual_recent_fights()
    check_data_freshness()

    # ---- 2. fechar o que passou -----------------------------------------
    from src.fechamento import preencher_template
    fechou = []
    for event_date, event_name in _eventos_para_fechar(hoje):
        logger.info("=== Fechando %s (%s) ===", event_name, event_date)
        n, relatos = preencher_template(event_date, dry_run=args.dry_run)
        if not relatos:
            logger.info("%s ainda não está na base -- fica para a próxima rodada.", event_date)
            continue
        pendentes = [r for r in relatos if r.estado == "ambigua"]
        for r in pendentes:
            logger.error("DECISÃO SUA: %s x %s -- %s", r.fighter_a, r.fighter_b, r.detalhe)
        if n:
            fechou.append((event_date, event_name))
            mexeu = True

    # ---- 3. regenerar (é o que sincroniza o histórico) -------------------
    if fechou and not args.dry_run:
        from scripts.publish_report import publish
        from src.identidade import (IdentidadeInconsistente,
                                    validar_identidade_do_card)
        try:
            # PARADA DURA antes de escrever: CSV, template, historico e
            # estado local tem de concordar sobre que card e este. Foi a
            # falta disto que deixou o ciclo gravar 12 lutas sob a data
            # errada em 01/nov.
            validar_identidade_do_card()
        except IdentidadeInconsistente as exc:
            logger.error("IDENTIDADE DO CARD INCONSISTENTE -- nada regenerado: %s", exc)
            logger.error("Resolva a mao (scripts/reconciliar_card.py ajuda) "
                         "antes de deixar o ciclo escrever.")
            return 1
        atual = _card_atual()
        logger.info("=== Regenerando o relatório para sincronizar o histórico ===")
        publish(str(config.UPCOMING_CARD_CSV), atual["nome"], event_date=atual["data"],
                no_push=True, sharp=False)
        _git("add", "data/odds_template.csv")
        _git("commit", "-m", "Fecha o(s) evento(s): "
             + ", ".join(f"{n} ({d})" for d, n in fechou))

    # ---- 4. re-treinar ---------------------------------------------------
    if fechou and not args.dry_run:
        logger.info("=== Re-treinando com o evento novo na base ===")
        from src import features, train, train_method
        features.build_feature_dataset()
        train.train_and_calibrate()
        train_method.train_method()

    # ---- 5. a página mostra o card vigente ------------------------------
    # Um card pode estar registrado no histórico e NÃO estar na página: foi o
    # que sobrou do conserto de 31/out. Quando o vigente muda (o anterior
    # acabou), é aqui que o CSV e a página o acompanham -- com sharp, porque
    # as linhas estão abertas e podem ter sido registradas antes de o mercado
    # abrir direito.
    vigente = _card_vigente(hoje)
    if vigente and not _csv_e_do_card(vigente[0]) and not args.dry_run:
        data, nome = vigente
        logger.info("=== Página está em outro card; trazendo %s (%s) ===", nome, data)
        rc = _montar_card(data, chromium=args.chromium_ufcstats)
        if rc != 0:
            logger.error("build_card falhou para %s -- página NÃO trocada.", data)
        else:
            from scripts.publish_report import publish
            publish(str(config.UPCOMING_CARD_CSV), nome, event_date=data,
                    no_push=True, sharp=True)
            _salvar_estado(nome, data)
            mexeu = True
    elif vigente and not _csv_e_do_card(vigente[0]):
        logger.info("[dry-run] traria a página para %s (%s)", vigente[1], vigente[0])

    # ---- 6. registrar o próximo -----------------------------------------
    proximo = _proximo_a_registrar(hoje, dry_run=args.dry_run)
    if proximo and not args.dry_run:
        data = proximo["event_date"]
        logger.info("=== Montando o card de %s ===", data)
        rc = _montar_card(data, chromium=args.chromium_ufcstats)
        if rc != 0:
            logger.error("build_card falhou para %s -- card NÃO registrado.", data)
        else:
            nome = _nome_do_card(data, chromium=args.chromium_ufcstats)
            _anexar_ao_template(nome, data)
            from scripts.publish_report import publish
            publish(str(config.UPCOMING_CARD_CSV), nome, event_date=data,
                    no_push=True, sharp=True)
            _salvar_estado(nome, data)
            mexeu = True
    elif proximo:
        logger.info("[dry-run] registraria o card de %s", proximo["event_date"])
    else:
        logger.info("Nenhum card novo a registrar nesta rodada.")

    if not mexeu:
        logger.info("Nada a fazer. Nenhum arquivo tocado.")
        return 0
    if args.dry_run:
        return 0

    # ---- 7. testes, e só então push --------------------------------------
    if not args.sem_testes:
        logger.info("=== Suíte ===")
        if subprocess.run([sys.executable, "-m", "pytest", "-q"],
                          cwd=config.PROJECT_ROOT).returncode != 0:
            logger.error("SUÍTE VERMELHA -- trabalho commitado localmente, SEM push. "
                         "Rode 'git log' e resolva antes de publicar.")
            return 1
    if args.no_push:
        logger.info("--no-push: pronto local. Rode 'git push' quando quiser.")
        return 0
    push = _git("push")
    if push.returncode != 0:
        logger.error("git push falhou:\n%s%s", push.stdout, push.stderr)
        return 1
    logger.info("Publicado.")
    return 0


# ---------------------------------------------------------------------------

def _card_atual() -> dict:
    """
    De que card e o CSV que esta em data/raw/upcoming_card_odds.csv.

    SEMPRE derivado do conteudo do CSV, nunca de estado guardado. O arquivo
    de estado pode estar adiantado em relacao ao CSV, e em 01/nov/2026 estava:
    o ciclo fechou o card de 31/out, foi regenerar o relatorio e publicou com
    o nome e a data de Bonfim vs Brady (07/nov) enquanto o CSV ainda era o de
    Moicano vs Nolan. Resultado: as 12 lutas de 31/out gravadas DE NOVO sob
    event_date 07/nov.

    O estrago ficou local -- TestCardTemplateHistorico falhou e a trava de
    suite-antes-do-push impediu a publicacao --, mas a origem era confiar num
    estado que ninguem garantia estar em dia.
    """
    import pandas as pd
    card = pd.read_csv(config.UPCOMING_CARD_CSV)
    h = pd.read_csv(config.PREDICTION_HISTORY_CSV)
    pares = {tuple(sorted([a, b])) for a, b in zip(card["fighter_a"], card["fighter_b"])}
    achadas = h[h.apply(lambda r: tuple(sorted([r["fighter_a"], r["fighter_b"]])) in pares,
                        axis=1)]
    if achadas.empty:
        raise RuntimeError(
            "o CSV do card nao casa com nenhuma luta do historico -- "
            "publique a mao em vez de adivinhar o evento")
    chaves = achadas[["event_name", "event_date"]].drop_duplicates()
    if len(chaves) > 1:
        raise RuntimeError(
            f"as lutas do CSV aparecem em {len(chaves)} eventos diferentes "
            f"({list(chaves.itertuples(index=False, name=None))}) -- nao publico "
            f"sem saber qual e")
    return {"nome": str(chaves.iloc[0]["event_name"]),
            "data": str(chaves.iloc[0]["event_date"])}


def _salvar_estado(nome: str, data: str) -> None:
    import json
    _ESTADO.write_text(json.dumps({"nome": nome, "data": data}, ensure_ascii=False),
                       encoding="utf-8")


def _montar_card(event_date: str, *, chromium: bool = False) -> int:
    comando = [sys.executable, "-m", "scripts.build_card", "--event-date", event_date]
    if chromium:
        comando.append("--chromium-ufcstats")
    return subprocess.run(comando, cwd=config.PROJECT_ROOT).returncode


def _nome_do_card(event_date: str, *, chromium: bool = False) -> str:
    from src.card_sources import card_do_ufcstats
    try:
        # "UFC Fight Night: Moicano vs. Nolan" -> sem o ponto, como a série usa
        return card_do_ufcstats(event_date, chromium=chromium)["nome"].replace(" vs. ", " vs ")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Não consegui o nome do card (%s) -- usando a data.", exc)
        return f"UFC {event_date}"


def _anexar_ao_template(nome: str, data: str) -> None:
    from pathlib import Path
    card = Path(config.UPCOMING_CARD_CSV).read_text(encoding="utf-8").splitlines()[1:]
    p = Path(config.ODDS_TEMPLATE_CSV)
    s = p.read_text(encoding="utf-8")
    # Completa o que FALTA, em vez de desistir porque a data ja aparece. O
    # card cresce durante a semana: em 05/nov o historico tinha SEIS lutas
    # novas de 07/nov que nunca chegaram ao template, e luta sem linha no
    # template nunca fecha.
    import pandas as pd
    ja = set()
    try:
        t = pd.read_csv(p)
        ja = {tuple(sorted([a, b])) for a, b, d
              in zip(t["fighter_a"], t["fighter_b"], t["event_date"].astype(str))
              if d == data}
    except (OSError, KeyError, pd.errors.EmptyDataError):
        pass
    if not s.endswith("\n"):
        s += "\n"
    novas = []
    for linha in card:
        c = linha.split(",")
        if tuple(sorted([c[0], c[1]])) in ja:
            continue
        novas.append(f",{nome},{data},{c[0]},{c[1]},"
                     f"{float(c[2]):.3f},{float(c[3]):.3f},\n")
    if not novas:
        logger.info("Template ja tem as %d luta(s) do card de %s.", len(card), data)
        return
    p.write_text(s + "".join(novas), encoding="utf-8")
    logger.info("%d luta(s) nova(s) anexadas ao template de %s.", len(novas), data)


if __name__ == "__main__":
    sys.exit(main())
