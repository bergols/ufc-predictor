"""
scripts/ciclo_semanal.py

O ciclo de evento inteiro, sem ninguém pedir: fecha o card que passou e
registra o próximo.

    python -m scripts.ciclo_semanal                 # faz tudo e dá push
    python -m scripts.ciclo_semanal --dry-run       # relata e não escreve nada
    python -m scripts.ciclo_semanal --no-push       # trabalha local

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
  4. REGISTRA o próximo card, se a API já o lista e ele ainda não está na
     série: monta com as duas fontes (scripts/build_card.py) e publica, o que
     congela previsão + sinal sharp;
  5. roda a suíte e só então dá push.

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


def _proximo_a_registrar(hoje: date):
    from src.event_watch import carregar_pendentes, eventos_a_registrar
    from src.line_shopping import fetch_live_odds, get_api_key
    from src.prediction_history import load_history

    pendentes = carregar_pendentes(hoje=hoje, history_df=load_history())
    if pendentes:
        return pendentes[0]
    try:
        eventos = fetch_live_odds(get_api_key())
    except Exception as exc:  # noqa: BLE001 - job agendado nunca estoura
        logger.warning("API indisponível (%s) -- sem registrar card novo nesta rodada.", exc)
        return None
    achados = eventos_a_registrar(eventos, load_history(), hoje=hoje)
    return achados[0] if achados else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Fecha o card que passou e registra o próximo.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--sem-testes", action="store_true",
                        help="Não roda a suíte antes do push (não recomendado)")
    args = parser.parse_args()
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

    # ---- 5. registrar o próximo -----------------------------------------
    proximo = _proximo_a_registrar(hoje)
    if proximo and not args.dry_run:
        data = proximo["event_date"]
        logger.info("=== Montando o card de %s ===", data)
        rc = subprocess.run([sys.executable, "-m", "scripts.build_card",
                             "--event-date", data], cwd=config.PROJECT_ROOT).returncode
        if rc != 0:
            logger.error("build_card falhou para %s -- card NÃO registrado.", data)
        else:
            nome = _nome_do_card(data)
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

    # ---- 6. testes, e só então push --------------------------------------
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
    import json
    if _ESTADO.exists():
        try:
            return json.loads(_ESTADO.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    # sem estado: deduz do histórico a partir do CSV do card
    import pandas as pd
    card = pd.read_csv(config.UPCOMING_CARD_CSV)
    h = pd.read_csv(config.PREDICTION_HISTORY_CSV)
    pares = {tuple(sorted([a, b])) for a, b in zip(card["fighter_a"], card["fighter_b"])}
    achadas = h[h.apply(lambda r: tuple(sorted([r["fighter_a"], r["fighter_b"]])) in pares,
                        axis=1)]
    if achadas.empty:
        raise RuntimeError("não sei de que card é o CSV atual -- rode o publish à mão")
    return {"nome": str(achadas.iloc[0]["event_name"]),
            "data": str(achadas.iloc[0]["event_date"])}


def _salvar_estado(nome: str, data: str) -> None:
    import json
    _ESTADO.write_text(json.dumps({"nome": nome, "data": data}, ensure_ascii=False),
                       encoding="utf-8")


def _nome_do_card(event_date: str) -> str:
    from src.card_sources import card_do_ufcstats
    try:
        # "UFC Fight Night: Moicano vs. Nolan" -> sem o ponto, como a série usa
        return card_do_ufcstats(event_date)["nome"].replace(" vs. ", " vs ")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Não consegui o nome do card (%s) -- usando a data.", exc)
        return f"UFC {event_date}"


def _anexar_ao_template(nome: str, data: str) -> None:
    from pathlib import Path
    card = Path(config.UPCOMING_CARD_CSV).read_text(encoding="utf-8").splitlines()[1:]
    p = Path(config.ODDS_TEMPLATE_CSV)
    s = p.read_text(encoding="utf-8")
    if data in s:
        logger.info("Template já tem o card de %s.", data)
        return
    if not s.endswith("\n"):
        s += "\n"
    novas = []
    for linha in card:
        c = linha.split(",")
        novas.append(f",{nome},{data},{c[0]},{c[1]},"
                     f"{float(c[2]):.3f},{float(c[3]):.3f},\n")
    p.write_text(s + "".join(novas), encoding="utf-8")
    logger.info("%d linha(s) anexadas ao template.", len(novas))


if __name__ == "__main__":
    sys.exit(main())
