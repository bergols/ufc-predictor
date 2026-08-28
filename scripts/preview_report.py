"""
scripts/preview_report.py

Bancada de PREVIEW do relatorio: gera a pagina a partir de lutas inventadas,
para conferir DESENHO sem tocar em dado nenhum.

    python -m scripts.preview_report
    python -m scripts.preview_report --shots     # + capturas de tela

Por que existe
--------------
Mudanca de layout nao quebra teste: teste ve markup, nao ve tela. Os defeitos
de design que apareceram no projeto foram todos desse tipo -- retrato
transbordando por cima do nome em 390px, uma regra de especificidade maior
devolvendo um avatar de 116px onde a variavel pedia 34, um segmento de barra
na mesma cor do trilho fazendo a barra parecer vazia, um SVG sem dimensao
sumindo da legenda. Nenhum acusaria em pytest; todos sao obvios na captura.

Por que nao regerar o relatorio de verdade
------------------------------------------
`generate_card_report` PREVE e GRAVA: chama o modelo, consulta a API do sinal
sharp e escreve em data/prediction_history.csv. Regerar so para olhar layout
gastaria credito de API e, pior, sobrescreveria o sinal sharp do pre-registro
-- que e a regua do CLV. Aqui a rota e outra: monta o dict `analysis` na mao e
chama `render_html` direto, que so desenha.

As lutas cobrem de proposito as quatro faixas de divergencia (as arestas
vermelhas do card), perna EV, distribuicao de metodo, nome longo, lutador sem
foto (cai no monograma) e o aviso de pouca experiencia.

O painel do Historico vem do CSV REAL, porque e o unico jeito de ver o
alarme de fechamento e o placar da serie no estado em que eles estao hoje --
leitura, nunca escrita.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SAIDA = config.PROJECT_ROOT / "docs" / "_preview.html"


def _luta(a: str, b: str, model_a: float, market_a: float,
          odds_a: float, odds_b: float, pick_a: bool = True, **extra) -> dict:
    """O minimo que `_bout`/`_ev_card`/`_method_card` consomem de uma luta."""
    favorito = a if market_a >= 0.5 else b
    lado = a if pick_a else b
    d = {
        "fighter_a": a, "fighter_b": b,
        "model_side": lado, "favorite": favorito,
        "model_prob_a": model_a, "market_prob_a": market_a,
        "model_side_prob": model_a if pick_a else 1 - model_a,
        "market_prob_fav": max(market_a, 1 - market_a),
        "market_prob_dog": min(market_a, 1 - market_a),
        "odds_a": odds_a, "odds_b": odds_b,
        "category": "favorite" if lado == favorito else "underdog",
        "method_probs": None,
    }
    d.update(extra)
    return d


def montar_analysis() -> dict:
    # uma luta por faixa de divergencia: agreement / small / material / large
    favoritas = [
        _luta("Julia Polastri", "Xiong Jingnan", 0.737, 0.672, 1.42, 2.91,
              low_experience=True),                                   # small
        _luta("Andre Lima", "Namsrai Batbayar", 0.648, 0.640, 1.38, 3.00),  # agreement
        _luta("Charles Oliveira", "Jose Aldo", 0.612, 0.510, 1.55, 2.45),   # material
        # nome longo + sem foto na base: testa reticencia e monograma
        _luta("Christian Leroy Duncan", "Marc-Andre Barriault", 0.583, 0.545, 1.72, 2.18),
    ]
    principal = _luta("Umar Nurmagomedov", "Song Yadong", 0.569, 0.800, 1.19, 4.75)  # large
    favoritas.append(principal)
    favoritas.sort(key=lambda f: f["model_side_prob"], reverse=True)

    zebras = [_luta("Song Yadong", "Petr Yan", 0.560, 0.245, 3.10, 1.35)]

    for f, ev, odd in [(zebras[0], 1.74, 3.10), (favoritas[0], 1.05, 1.42)]:
        f["ev"], f["model_side_odds"] = ev, odd
    for f, mp in [(favoritas[0], {"KO_TKO": 0.41, "SUBMISSION": 0.17, "DECISION": 0.42}),
                  (favoritas[1], {"KO_TKO": 0.22, "SUBMISSION": 0.11, "DECISION": 0.67}),
                  (zebras[0], {"KO_TKO": 0.58, "SUBMISSION": 0.21, "DECISION": 0.21})]:
        f["method_probs"] = mp

    return {
        "favorites": favoritas, "underdogs": zebras,
        # a luta principal E uma das lutas das listas (mesma referencia), como
        # no analyze_card real -- separa-la aqui esconderia bug de marcacao
        "main_event": principal,
        "ev_legs": [zebras[0], favoritas[0]],
        "method_ranking": [zebras[0], favoritas[0], favoritas[1]],
        "no_method": [],
        "no_prediction": [{"fighter_a": "Bilal Hasan", "fighter_b": "Nilson Rojas",
                           "odds_a": 1.80, "odds_b": 2.05,
                           "reason": "os dois estreando no UFC — sem histórico na base"}],
        "model_name": "logreg",
    }


def gerar() -> Path:
    from src.card_report import render_html
    from src.prediction_history import load_history, render_history_panel, set_photo_map

    fotos = config.RAW_DIR / "fighter_photos.json"
    if fotos.exists():
        set_photo_map(json.loads(fotos.read_text(encoding="utf-8")))

    SAIDA.parent.mkdir(exist_ok=True)
    html = render_html(
        montar_analysis(), freshness_gap_days=2,
        card_name="Preview — lutas fictícias",
        event_date="2026-08-29",
        history_panel_html=render_history_panel(load_history()))
    SAIDA.write_text(html, encoding="utf-8")
    return SAIDA


def capturar(destino: Path) -> int:
    """Uma captura por aba, em desktop e em celular."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error("playwright nao instalado. `pip install playwright` e "
                     "`playwright install chromium`.")
        return 1

    abas = [("favoritos", None), ("zebras", "Melhores zebras"),
            ("ev", "Pernas EV>1"), ("metodo", "Método de vitória"),
            ("historico", "Histórico")]
    telas = [("desktop", 1180, 1500), ("mobile", 390, 1100)]
    destino.mkdir(parents=True, exist_ok=True)
    url = SAIDA.as_uri()

    with sync_playwright() as p:
        navegador = p.chromium.launch()
        for tela, w, h in telas:
            for aba, botao in abas:
                pagina = navegador.new_page(viewport={"width": w, "height": h})
                # networkidle, nao "load": as fotos sao hotlink do UFC.com e o
                # <img> tem onerror que cai no monograma. Sem esperar a rede,
                # a captura sai com TODOS os retratos em iniciais e some
                # justamente o enquadramento que se quer conferir.
                pagina.goto(url, wait_until="networkidle")
                if botao:
                    pagina.get_by_role("button", name=botao).click()
                    pagina.wait_for_timeout(250)
                pagina.wait_for_timeout(350)
                arquivo = destino / f"{tela}_{aba}.png"
                pagina.screenshot(path=str(arquivo))
                pagina.close()
        navegador.close()
    logger.info("%d capturas em %s", len(abas) * len(telas), destino)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera um preview do relatorio com lutas ficticias (nao toca em dado).")
    parser.add_argument("--shots", action="store_true",
                        help="Captura uma tela por aba, em desktop e celular (usa Playwright)")
    parser.add_argument("--shots-dir", type=str, default="",
                        help="Onde salvar as capturas (padrao: docs/_preview_shots/)")
    args = parser.parse_args()

    caminho = gerar()
    logger.info("Preview em %s", caminho)
    logger.info("Abra com: start %s", caminho)

    if args.shots:
        destino = Path(args.shots_dir) if args.shots_dir else SAIDA.parent / "_preview_shots"
        return capturar(destino)
    return 0


if __name__ == "__main__":
    sys.exit(main())
