"""
src/shrinkage.py

Regularizacao das taxas com denominador pequeno (candidato, out/2026).

O problema, medido antes de escrever uma linha disto: no conjunto de teste, o
grupo cujo lado menos experiente tem 1 ou 2 lutas no UFC tem AUC 0,601 contra
0,708 dos com 3+. O contraste pareado por evento exclui zero em AUC, log loss
e Brier. Nao e a estreia -- o estreante (AUC 0,683) nao se separa de nenhum
dos dois.

A causa plausivel esta no DENOMINADOR. Mediana de tentativas acumuladas do
lado menos experiente, no teste:

    grupo 1-2    88 golpes significativos,  2 quedas tentadas
    grupo 3-5   284 golpes,                 7 quedas
    grupo 6+    775 golpes,                18 quedas

`takedown_defense = 1 - landed_opp/attempted_opp` com DUAS tentativas contra
nao e uma taxa, e uma moeda. Pior: ela entra no modelo com a mesma cara de
uma taxa medida em 18 tentativas. O estreante nao cai nessa armadilha porque
o valor dele e NaN e o modelo trata como ausente.

A correcao classica e encolher a taxa observada na direcao de um prior,
com peso proporcional ao tamanho da amostra:

    taxa_encolhida = (sucessos + k * prior) / (denominador + k)

Com denominador grande a taxa observada manda; com denominador pequeno o
prior manda. `prior` e `k` sao estimados SO NO TREINO de cada janela
(empirical Bayes por metodo dos momentos), nunca escolhidos olhando
avaliacao -- e por isso nao ha grade de busca aqui.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Taxas regularizadas: (coluna de saida, numerador acumulado, denominador
# acumulado). Todas sao "sucessos sobre tentativas" com o mesmo tratamento.
TAXAS = {
    "striking_accuracy": ("cum_sig_strikes_landed", "cum_sig_strikes_attempted"),
    "takedown_accuracy": ("cum_takedowns_landed", "cum_takedowns_attempted"),
    "career_win_rate": ("cum_is_win", "n_prior_fights"),
    "ko_rate": ("cum_is_win_ko", "n_prior_fights"),
    "submission_rate": ("cum_is_win_sub", "n_prior_fights"),
}
# defesa de queda e 1 - (acertadas CONTRA / tentadas CONTRA): encolhe a taxa
# sofrida e inverte depois, para o prior agir no lado certo da conta
TAXA_INVERTIDA = {
    "takedown_defense": ("cum_takedowns_landed_opp", "cum_takedowns_attempted_opp"),
}

# Limites de sanidade para o k estimado. Sem eles, uma variancia quase zero
# na amostra de treino manda k para milhares e a feature vira constante.
_K_MIN, _K_MAX = 1.0, 500.0
# Observacao com denominador minusculo nao ajuda a ESTIMAR o prior (ela e
# justamente o ruido que queremos controlar), entao fica fora da estimativa.
_DEN_MIN_PARA_ESTIMAR = 5


@dataclass(frozen=True)
class Prior:
    media: float
    k: float

    def aplicar(self, sucessos, denominador):
        s = pd.to_numeric(sucessos, errors="coerce")
        n = pd.to_numeric(denominador, errors="coerce")
        return (s + self.k * self.media) / (n + self.k)


def estimar_prior(sucessos, denominador) -> Prior:
    """
    Empirical Bayes por metodo dos momentos, numa amostra de TREINO.

        media = taxa agregada (soma dos sucessos / soma dos denominadores)
        k     = media*(1-media)/var(taxas observadas) - 1

    `k` sai dos dados, nao de escolha: e o tamanho de amostra em que a
    evidencia individual passa a pesar mais que o prior. Variancia nula ou
    nao finita devolve k no teto -- sem dispersao entre lutadores, o prior
    descreve todo mundo.
    """
    s = pd.to_numeric(pd.Series(sucessos), errors="coerce")
    n = pd.to_numeric(pd.Series(denominador), errors="coerce")
    ok = n.notna() & s.notna() & (n >= _DEN_MIN_PARA_ESTIMAR)
    if ok.sum() < 30 or n[ok].sum() <= 0:
        return Prior(0.5, _K_MAX)

    media = float(s[ok].sum() / n[ok].sum())
    taxas = (s[ok] / n[ok]).to_numpy(dtype="float64")
    var = float(np.var(taxas, ddof=1)) if len(taxas) > 1 else 0.0
    if not np.isfinite(var) or var <= 0:
        k = _K_MAX
    else:
        k = media * (1.0 - media) / var - 1.0
    return Prior(float(np.clip(media, 1e-6, 1 - 1e-6)),
                 float(np.clip(k, _K_MIN, _K_MAX)))


def estimar_priors(treino_df: pd.DataFrame) -> dict[str, Prior]:
    """Um prior por taxa, estimado so nas linhas de treino."""
    priors = {}
    for saida, (num, den) in {**TAXAS, **TAXA_INVERTIDA}.items():
        if num not in treino_df.columns or den not in treino_df.columns:
            continue
        priors[saida] = estimar_prior(treino_df[num], treino_df[den])
    for nome, p in sorted(priors.items()):
        logger.info("prior %-20s media=%.4f k=%.1f", nome, p.media, p.k)
    return priors


def aplicar_shrinkage(stats_df: pd.DataFrame, priors: dict[str, Prior]) -> pd.DataFrame:
    """
    Devolve uma copia com as taxas substituidas pelas encolhidas.

    NAO muda a politica de ausencia: lutador sem nenhuma luta anterior
    continua com denominador 0 e, por construcao, recebe exatamente o prior
    -- que e a leitura certa de "nao sei nada sobre este aqui". O que muda e
    o lutador com 1 ou 2 lutas, que deixa de ter uma taxa de aparencia
    firme calculada sobre duas tentativas.
    """
    out = stats_df.copy()
    for saida, (num, den) in TAXAS.items():
        if saida in priors and num in out.columns:
            out[saida] = priors[saida].aplicar(out[num], out[den])
    for saida, (num, den) in TAXA_INVERTIDA.items():
        if saida in priors and num in out.columns:
            out[saida] = 1.0 - priors[saida].aplicar(out[num], out[den])
    return out
