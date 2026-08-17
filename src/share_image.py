"""
src/share_image.py

Gera a imagem de previa (Open Graph) do relatorio: o brasao completo, o nome
do card e o placar da serie, em 1200x630 -- o formato que WhatsApp, Twitter e
afins mostram quando alguem manda o link.

Por que Playwright em vez de uma biblioteca de imagem: o projeto ja depende
dele para o --fill-gap, e assim a imagem sai do MESMO CSS e dos MESMOS tokens
que a pagina (src/design.py). Desenhar o card de novo numa API de canvas seria
uma segunda fonte da verdade para a identidade, que e exatamente o que
design.py existe para evitar.

Falha de forma SUAVE: sem navegador, sem rede, qualquer erro -> devolve None e
a publicacao segue sem a previa. Preview de link nunca pode derrubar o
relatorio.
"""
from __future__ import annotations

import logging
from pathlib import Path

import config
from src.design import TAGLINE, WORDMARK, fonts_css, mark_svg, root_css

logger = logging.getLogger(__name__)

SHARE_W, SHARE_H = 1200, 630


def _html(card_name: str, event_date: str, footer: str) -> str:
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
{fonts_css()}
{root_css()}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ width: {SHARE_W}px; height: {SHARE_H}px; background: var(--bg);
  color: var(--text); font-family: var(--font-body); overflow: hidden;
  display: flex; flex-direction: column; justify-content: space-between;
  padding: 62px 68px; position: relative; }}
/* barra diagonal da marca, sangrando pela borda */
body::before {{ content: ""; position: absolute; left: -60px; top: 150px;
  width: 260px; height: 10px; background: var(--brand);
  transform: skewX(var(--slash)); }}
.top {{ display: flex; align-items: center; gap: 22px; }}
.crest {{ width: 104px; height: 104px; flex: none; color: var(--text); }}
.brand b {{ font-family: var(--font-display); font-weight: 800; font-style: italic;
  text-transform: uppercase; letter-spacing: .09em; font-size: 40px; display: block; }}
.brand span {{ font-size: 17px; color: var(--muted); letter-spacing: .16em;
  text-transform: uppercase; }}
h1 {{ font-family: var(--font-display); font-weight: 800; font-style: italic;
  text-transform: uppercase; font-size: 78px; line-height: .95;
  letter-spacing: -.025em; }}
.date {{ font-family: var(--font-num); font-weight: 600; font-size: 19px;
  color: var(--brand); letter-spacing: .14em; text-transform: uppercase;
  margin-bottom: 18px; }}
.foot {{ display: flex; justify-content: space-between; align-items: flex-end;
  border-top: 1px solid var(--line); padding-top: 20px; font-size: 16px;
  color: var(--muted); letter-spacing: .04em; }}
</style></head><body>
  <div class="top">{mark_svg("crest", full=True)}
    <div class="brand"><b>{WORDMARK}</b><span>{TAGLINE}</span></div>
  </div>
  <div>
    <div class="date">{event_date}</div>
    <h1>{card_name}</h1>
  </div>
  <div class="foot"><span>{footer}</span><span>não é recomendação de aposta</span></div>
</body></html>"""


def build_share_image(card_name: str, event_date: str = "", footer: str = "",
                      output: Path | None = None) -> Path | None:
    """Devolve o caminho do PNG, ou None se nao deu para gerar."""
    output = Path(output or config.PROJECT_ROOT / "docs" / "share.png")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.info("Playwright ausente — publicando sem imagem de prévia.")
        return None

    html = _html(card_name, event_date, footer)
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(channel="msedge", headless=True)
            except Exception:  # noqa: BLE001 - sem Edge, tenta o chromium proprio
                browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": SHARE_W, "height": SHARE_H},
                                    device_scale_factor=1)
            page.set_content(html, wait_until="load")
            page.wait_for_timeout(350)  # deixa a fonte embutida assentar
            output.parent.mkdir(exist_ok=True)
            page.screenshot(path=str(output))
            browser.close()
    except Exception as exc:  # noqa: BLE001 - previa nunca derruba a publicacao
        logger.warning("Não foi possível gerar a imagem de prévia (%s) — seguindo sem ela.", exc)
        return None

    logger.info("Imagem de prévia salva em %s", output)
    return output
