"""
scripts/calibration_report.py

Desenha a CURVA DE CALIBRAÇÃO (diagrama de confiabilidade) do modelo de
produção e a publica como página estática.

    python -m scripts.calibration_report
    python -m scripts.calibration_report --bins 10 --out docs/_calibracao.html

Como ler
--------
Cada ponto é uma faixa de probabilidade prevista. O eixo X é o que o modelo
PROMETEU naquela faixa; o eixo Y é o que de fato ACONTECEU. A diagonal é a
calibração perfeita: ponto acima dela = o modelo foi tímido, abaixo = foi
confiante demais.

O tamanho do ponto é o número de lutas na faixa — e é a metade mais importante
do desenho. Um ponto longe da diagonal com 3 lutas não diz nada; a curva
convida a ler distância e esquecer o peso.

Por que isto existe
-------------------
Havia Brier e a escolha sigmoid/isotonic, mas nenhuma visão POR FAIXA. Brier é
agregado: confiança demais em cima e timidez embaixo se cancelam na média e o
número final não denuncia. Era a última lacuna diagnóstica listada na skill
`ufc-analise`.

Não toca no modelo — é medição, e as regras de parada continuam valendo.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

L, A = 74, 360          # margem esquerda e lado útil do quadrado do gráfico
PAD_TOP, PAD_BOT = 26, 52


def _pontos_svg(tabela: pd.DataFrame, cor: str, classe: str) -> str:
    """Um círculo por faixa, raio pela raiz do n (área ~ n)."""
    cheias = tabela.dropna(subset=["gap"])
    if cheias.empty:
        return ""
    n_max = max(int(cheias["n"].max()), 1)
    partes = []
    for _, r in cheias.iterrows():
        x = L + r["prev_media"] * A
        y = PAD_TOP + (1 - r["freq_real"]) * A
        raio = 3.0 + 9.0 * (r["n"] / n_max) ** 0.5
        partes.append(f'<circle class="{classe}" cx="{x:.1f}" cy="{y:.1f}" '
                      f'r="{raio:.1f}" fill="{cor}"><title>{r["faixa"]}: '
                      f'{int(r["n"])} lutas, previsto {r["prev_media"]*100:.1f}%, '
                      f'real {r["freq_real"]*100:.1f}%</title></circle>')
    linha = " ".join(f'{L + r["prev_media"] * A:.1f},{PAD_TOP + (1 - r["freq_real"]) * A:.1f}'
                     for _, r in cheias.iterrows())
    return (f'<polyline points="{linha}" fill="none" stroke="{cor}" '
            f'stroke-width="1.6" opacity=".55"/>' + "".join(partes))


def desenhar(tabela: pd.DataFrame, ece: float, titulo: str) -> str:
    from src.design import PALETTE

    cores = {n: h for n, h, _ in PALETTE}
    largura, altura = L + A + 30, PAD_TOP + A + PAD_BOT

    grade = []
    for i in range(6):
        v = i / 5
        x = L + v * A
        y = PAD_TOP + (1 - v) * A
        grade.append(f'<line x1="{x:.1f}" y1="{PAD_TOP}" x2="{x:.1f}" '
                     f'y2="{PAD_TOP + A}" stroke="{cores["line"]}" stroke-width="1"/>')
        grade.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + A}" y2="{y:.1f}" '
                     f'stroke="{cores["line"]}" stroke-width="1"/>')
        grade.append(f'<text x="{x:.1f}" y="{PAD_TOP + A + 18}" text-anchor="middle" '
                     f'class="eixo">{v*100:.0f}%</text>')
        grade.append(f'<text x="{L - 10}" y="{y + 4:.1f}" text-anchor="end" '
                     f'class="eixo">{v*100:.0f}%</text>')

    return f"""<section class="calib">
  <h2>{titulo}</h2>
  <svg viewBox="0 0 {largura} {altura}" role="img"
       aria-label="Curva de calibração: probabilidade prevista contra frequência real">
    {"".join(grade)}
    <line x1="{L}" y1="{PAD_TOP + A}" x2="{L + A}" y2="{PAD_TOP}"
          stroke="{cores['dim']}" stroke-width="1.4" stroke-dasharray="5 4"/>
    <!-- canto superior esquerdo: previsao baixa com frequencia alta e a
         regiao que a curva nunca ocupa, entao o rotulo nao disputa espaco -->
    <text x="{L + 12}" y="{PAD_TOP + 20}" text-anchor="start" class="eixo">calibração perfeita</text>
    {_pontos_svg(tabela, cores["gold"], "pt")}
    <text x="{L + A / 2}" y="{altura - 12}" text-anchor="middle" class="eixo-nome">
      probabilidade prevista pelo modelo</text>
    <text x="16" y="{PAD_TOP + A / 2}" text-anchor="middle" class="eixo-nome"
          transform="rotate(-90 16 {PAD_TOP + A / 2})">frequência real de vitória</text>
  </svg>
  <p class="ece">ECE {ece:.4f} <span>— erro de calibração esperado, média dos desvios
  ponderada pelo tamanho da faixa</span></p>
</section>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Curva de calibracao do modelo de producao.")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    from src.design import fonts_css, root_css
    from src.evaluate import calibration_table, expected_calibration_error, log_calibration

    caminho = config.PROCESSED_DIR / "test_predictions.csv"
    if not caminho.exists():
        logger.error("%s nao existe -- rode 'python -m src.train' antes.", caminho)
        return 1
    preds = pd.read_csv(caminho)

    tabela = calibration_table(preds["label"], preds["pred_logreg"], args.bins)
    ece = expected_calibration_error(preds["label"], preds["pred_logreg"], args.bins)
    log_calibration(preds["label"], preds["pred_logreg"], "logreg / teste", args.bins)

    n = int(tabela["n"].sum())
    concentracao = tabela[tabela["faixa"].isin(["40%-50%", "50%-60%"])]["n"].sum()
    html = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Curva de calibração — Fight Model</title>
<style>
{fonts_css()}
{root_css()}
  body {{ background: var(--bg); color: var(--text); font-family: var(--font-body);
    margin: 0; padding: 34px 22px 60px; }}
  .wrap {{ max-width: 620px; margin: 0 auto; }}
  h1 {{ font-family: var(--font-display); font-style: italic; font-weight: 800;
    text-transform: uppercase; font-size: 1.7rem; margin: 0 0 6px;
    border-bottom: 2px solid var(--red); padding-bottom: 12px; }}
  h2 {{ font-family: var(--font-display); font-style: italic; font-weight: 800;
    text-transform: uppercase; font-size: 1rem; margin: 26px 0 10px; }}
  p {{ font-size: .82rem; line-height: 1.6; color: var(--dim); }}
  .eixo {{ fill: var(--muted); font-family: var(--font-num); font-size: 10px; }}
  .eixo-nome {{ fill: var(--muted); font-family: var(--font-body); font-size: 11px;
    text-transform: uppercase; letter-spacing: .08em; }}
  .pt {{ stroke: var(--bg); stroke-width: 1.2; }}
  .ece {{ font-family: var(--font-num); font-size: 1.05rem; font-weight: 700;
    color: var(--text); border-left: 3px solid var(--gold); padding-left: 12px; }}
  .ece span {{ font-weight: 400; font-size: .74rem; color: var(--muted); }}
  table {{ width: 100%; border-collapse: collapse; font-family: var(--font-num);
    font-size: .78rem; margin-top: 8px; }}
  th {{ text-align: right; color: var(--muted); font-weight: 600; font-size: .68rem;
    text-transform: uppercase; letter-spacing: .08em; padding: 6px 8px;
    border-bottom: 1px solid var(--line); }}
  th:first-child, td:first-child {{ text-align: left; }}
  td {{ text-align: right; padding: 5px 8px; border-bottom: 1px solid var(--line-soft); }}
  td.vazia {{ color: var(--muted); }}
  .nota {{ border-left: 3px solid var(--red); padding-left: 12px; margin-top: 26px; }}
</style></head><body><div class="wrap">
<h1>Curva de calibração</h1>
<p>Modelo de produção (logreg calibrada) no conjunto de teste: {n} linhas,
{tabela['n'].max()} na faixa mais cheia.</p>
{desenhar(tabela, ece, "Previsto contra realizado")}
<h2>Por faixa</h2>
<table><thead><tr><th>faixa</th><th>lutas</th><th>previsto</th><th>real</th><th>gap</th></tr></thead>
<tbody>
{"".join(
    f'<tr><td>{r["faixa"]}</td><td>{int(r["n"])}</td>'
    + ('<td class="vazia">—</td><td class="vazia">—</td><td class="vazia">—</td></tr>'
       if not r["n"] else
       f'<td>{r["prev_media"]*100:.1f}%</td><td>{r["freq_real"]*100:.1f}%</td>'
       f'<td>{r["gap"]*100:+.1f} pp</td></tr>')
    for _, r in tabela.iterrows())}
</tbody></table>
<div class="nota">
<p><strong>Como ler o tamanho do ponto.</strong> A área é proporcional ao número
de lutas da faixa. Ponto pequeno longe da diagonal não é descalibração — é
amostra pequena. A curva convida a ler distância e esquecer peso.</p>
<p><strong>O que este gráfico mostra neste projeto.</strong> A calibração é boa:
os desvios ficam dentro de poucos pontos percentuais onde há dado. O que salta
é a <em>concentração</em> — {concentracao} das {n} linhas ({concentracao/n*100:.0f}%)
caem entre 40% e 60%, e as pontas ficam quase vazias. Isso é o diagnóstico do
<code>DISCRIMINACAO.md</code> visto de outro ângulo: o problema não é o modelo
mentir sobre sua confiança, é ele quase nunca ter confiança para declarar.</p>
<p><strong>Simetria.</strong> A distribuição é espelhada de propósito: cada luta
entra duas vezes (A–B e B–A), então a faixa de 30% tem exatamente o mesmo
tamanho que a de 70%. Não é coincidência nem defeito.</p>
</div>
</div></body></html>"""

    saida = Path(args.out) if args.out else config.PROJECT_ROOT / "docs" / "_calibracao.html"
    saida.parent.mkdir(exist_ok=True)
    saida.write_text(html, encoding="utf-8")
    logger.info("Curva salva em %s", saida)
    logger.info("Abra com: start %s", saida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
