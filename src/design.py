"""
src/design.py

FONTE DA VERDADE da identidade visual do Fight Model.

Tudo que define a cara do projeto — paleta, tipografia, ângulo do corte
diagonal e a marca — mora aqui. O relatório (`src/card_report.py`) e o guia
de identidade (`scripts/brand_page.py`) consomem os MESMOS tokens, então o
guia não tem como divergir da página: se um valor muda aqui, muda nos dois.

Antes disto os tokens viviam soltos dentro da f-string do CSS do relatório,
que é justamente o jeito de a identidade virar decoração e apodrecer.

Princípios do sistema (o "porquê" de cada regra está no guia):
  1. cor CHAPADA — zero gradiente decorativo;
  2. canto reto e régua de 1px no lugar de caixa;
  3. UM acento por contexto, gasto com parcimônia;
  4. número sempre tabular;
  5. corte diagonal como motivo recorrente;
  6. a marca é a barra divergente — o que o programa faz.
"""
from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import config

# --------------------------------------------------------------------------
# Paleta. `role` é o que decide o uso — cor sem papel definido vira enfeite.
# --------------------------------------------------------------------------
PALETTE = [
    ("bg",        "#000000", "Fundo. Preto puro, contraste de cartaz."),
    ("surface",   "#0B0B0D", "Superfície elevada: cabeçalho, destaque da luta principal."),
    ("surface2",  "#141417", "Trilho de barra e chip — o 'vazio' de um gráfico."),
    ("line",      "#232329", "Régua de 1px. Separa confronto, seção e tabela."),
    ("line-soft", "#17171A", "Régua interna, mais discreta (linha de tabela)."),
    ("text",      "#FFFFFF", "Texto principal e o lado apontado pelo modelo."),
    ("dim",       "#B8B8C0", "Texto secundário e o lado NÃO apontado."),
    ("muted",     "#79798A", "Rótulo, metadado, ressalva."),
    ("red",       "#D20A11", "COR DA CASA. Cabeçalho, régua do hero, divergência."),
    ("gold",      "#D6AF37", "Acento da aba Favoritos — concordância com o mercado."),
    ("green",     "#2FA36B", "Acento de EV e de número positivo (P&L, CLV)."),
    ("steel",     "#7E9BD4", "Acento da aba Método — frio de propósito, é tendência fraca."),
]

FONTS = {
    # Display: Barlow Condensed ExtraBold Italic. É a camada editorial inteira —
    # título do evento, nome de lutador, aba, veredito.
    "display": ('"FM Display", "Archivo Narrow", "Roboto Condensed", '
                '"Arial Narrow", "Helvetica Neue", Arial, sans-serif'),
    # Corpo e NÚMERO: IBM Plex Sans Condensed. Uma família só para os dois, com
    # tabular-nums fazendo o alinhamento — não precisa de uma monoespaçada
    # separada, e o número fica com cara de estatística de transmissão em vez
    # de terminal.
    "body": ('"FM Sans", -apple-system, "Segoe UI", Roboto, '
             '"Helvetica Neue", Arial, sans-serif'),
    "num": ('"FM Sans", "SF Mono", "Consolas", ui-monospace, monospace'),
}

# --------------------------------------------------------------------------
# Fontes embutidas
# --------------------------------------------------------------------------
# Ficam em base64 no proprio HTML: a pagina nao pode depender de CDN (regra do
# projeto, com teste que quebra se aparecer url(http...) no arquivo). Sao os
# subconjuntos LATIN, que cobrem todos os acentos do portugues -- ~63KB de
# woff2, ~85KB depois do base64.
#
# Renomeadas para "FM Display"/"FM Sans" de proposito: a OFL trata subconjunto
# como versao modificada, e "Plex" e Reserved Font Name da IBM. Redistribuir um
# subset com o nome original seria justamente o que a clausula proibe. O nome
# so vale dentro deste CSS; o credito correto vai no rodape (ver FONT_NOTICE).
FONT_FILES = [
    ("FM Display", "barlow-800i.woff2", 800, "italic"),
    ("FM Sans", "plex-400.woff2", 400, "normal"),
    ("FM Sans", "plex-600.woff2", 600, "normal"),
]

FONT_NOTICE = (
    "Tipografia: Barlow (Copyright 2017 The Barlow Project Authors) e "
    "IBM Plex Sans Condensed (Copyright © 2017 IBM Corp. com Reserved Font Name "
    "“Plex”), ambas sob SIL Open Font License 1.1 — scripts.sil.org/OFL. "
    "Embutidos aqui os subconjuntos latinos, renomeados “FM Display” e “FM Sans” "
    "por serem versões reduzidas."
)


@lru_cache(maxsize=1)
def fonts_css() -> str:
    """Blocos @font-face com o woff2 embutido em base64.

    Fonte ausente não derruba a geração: o stack de fallback assume e a página
    sai com a tipografia do sistema — pior visual, nunca layout quebrado.
    """
    blocks = []
    for family, filename, weight, style in FONT_FILES:
        path = config.PROJECT_ROOT / "assets" / "fonts" / filename
        if not path.exists():
            continue
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        blocks.append(
            f"@font-face {{ font-family: '{family}'; font-style: {style}; "
            f"font-weight: {weight}; font-display: swap; "
            f"src: url(data:font/woff2;base64,{b64}) format('woff2'); }}")
    return "\n".join(blocks)

# Ângulo do corte diagonal. Um só, usado em toda parte: etiqueta, veredito,
# faixa da luta principal e a barra do cabeçalho do evento.
SLASH = "-9deg"


def root_css() -> str:
    """Bloco `:root` com todos os tokens. Sem chaves duplicadas — quem embute
    numa f-string deve inserir via `{root_css()}`, não colar o texto."""
    cores = "\n".join(f"    --{name}: {hexv};" for name, hexv, _ in PALETTE)
    return f""":root {{
{cores}
    --brand: var(--red);
    --font-display: {FONTS['display']};
    --font-body: {FONTS['body']};
    --font-num: {FONTS['num']};
    --slash: {SLASH};
    --accent: var(--gold);
  }}"""


# --------------------------------------------------------------------------
# A marca — brasão "instrumento de medição"
# --------------------------------------------------------------------------
# Leitura do desenho: o escudo é o instrumento; as DUAS BARRAS de comprimentos
# diferentes são as duas leituras (modelo e mercado); a régua vertical com
# marcações é a medição; e o corte vermelho na diagonal é a diferença entre as
# duas — o delta, que é o que o projeto de fato mede.
#
# Duas versões porque uma só não resolve: a completa tem detalhe que morre
# abaixo de ~64px (a régua tem traços de 0.9 de espessura, que a 24px viram
# meio pixel), e um ícone que aguenta 24px não tem presença quando grande.
# Branco = currentColor (inverte sozinho em fundo claro); vermelho é fixo.
_CREST_FULL = (
    '<path d="M5 5.5L12.8 6.4L15.8 5.4L24.2 5.4L27.2 6.4L35 5.5V15.8L32.7 18.5V28L20 36L7.3 28V18.5L5 15.8V5.5Z"'
    ' stroke="currentColor" stroke-width="2.2" stroke-linejoin="miter"/>'
    '<path d="M8.2 8.7L13.4 9.4L16.2 8.5H23.8L26.6 9.4L31.8 8.7V15L29.7 17.3V26.2L20 32.7L10.3 26.2V17.3L8.2 15V8.7Z"'
    ' stroke="currentColor" stroke-width="1.5" stroke-linejoin="miter"/>'
    '<path d="M20 8.8V31.2" stroke="currentColor" stroke-width="1.2"/>'
    '<path d="M18.1 11.3H21.9M18.8 14H21.2M18.1 16.7H21.9M18.8 19.4H21.2M18.1 22.1H21.9M18.8 24.8H21.2M18.1 27.5H21.9"'
    ' stroke="currentColor" stroke-width="0.9"/>'
    '<path d="M20 7L18.5 9.2H21.5L20 7Z" fill="currentColor"/>'
    '<path d="M20 33L18.5 30.8H21.5L20 33Z" fill="currentColor"/>'
    '<path d="M10.5 13.8H28.7L27.2 16.9H10.5V13.8Z" fill="currentColor"/>'
    '<path d="M11.3 22.1H25.4L23.8 25.2H11.3V22.1Z" fill="#D20A11"/>'
    '<path d="M12.2 29.1L26.7 10.4L24.2 18.6L28.2 14.1L13.2 31.4L15.7 24.7L12.2 29.1Z" fill="#D20A11"/>'
    '<path d="M6.1 18.3L8.2 20.6V26.8L10.4 28.3L8.1 29.6L5.9 28.1V19.7L6.1 18.3Z" fill="#D20A11"/>'
    '<path d="M33.9 18.3L31.8 20.6V26.8L29.6 28.3L31.9 29.6L34.1 28.1V19.7L33.9 18.3Z" fill="#D20A11"/>'
)

# Ícone: silhueta + as duas leituras + o delta. Quatro formas, nada com menos
# de ~1.5px a 24px.
_CREST_ICON = (
    '<path d="M8 8L14 9.1L17 8H23L26 9.1L32 8V17L29.7 19.2V28L20 34L10.3 28V19.2L8 17V8Z"'
    ' stroke="currentColor" stroke-width="2.6" stroke-linejoin="miter"/>'
    '<path d="M11.2 14H28.2L26.7 17.2H11.2V14Z" fill="currentColor"/>'
    '<path d="M11.8 22.1H25.1L23.5 25.3H11.8V22.1Z" fill="#D20A11"/>'
    '<path d="M13.2 29.4L27.1 10.6L24.6 18.7L28.1 14.3L14.1 31.2L16.5 24.5L13.2 29.4Z" fill="#D20A11"/>'
)


def mark_svg(css_class: str = "mark", full: bool = False) -> str:
    """`full=True` devolve o brasão completo (só para exibição grande)."""
    body = _CREST_FULL if full else _CREST_ICON
    return (f'<svg class="{css_class}" viewBox="0 0 40 40" fill="none" '
            f'aria-hidden="true">{body}</svg>')


def favicon_data_uri() -> str:
    """Ícone da aba: o mesmo SVG reduzido, com o branco fixado (não há
    `currentColor` para herdar dentro de um favicon)."""
    body = _CREST_ICON.replace("currentColor", "%23fff").replace("#D20A11", "%23D20A11")
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40" '
           f'fill="none">{body}</svg>')
    return "data:image/svg+xml," + svg.replace('"', "'").replace("<", "%3C").replace(">", "%3E").replace("#", "%23")


WORDMARK = "Fight Model"
TAGLINE = "modelo vs. mercado"
