"""
Por que uma feature SIMETRICA nao informa a logreg (out/2026).

Cada luta vira DUAS linhas espelhadas, A-B e B-A, com labels opostos. Uma
feature que vale o MESMO nas duas e, por construcao, incapaz de separa-las:
o modelo linear ve o mesmo valor com y=1 e com y=0, e o coeficiente vai a
zero. Ela so desloca o intercepto.

Isto nao e teoria. Na logreg de producao `stance_mismatch` tem coeficiente
EXATAMENTE 0,000000, e foi o que explicou o nulo exato do experimento de
categoria de peso: peso tambem e propriedade da luta, nao de um lado.

A consequencia pratica, que este teste existe para nao deixar esquecer: para
uma propriedade da LUTA entrar no preditor de vencedor, ela precisa aparecer
como INTERACAO com um diferencial (diff x simetrica e antissimetrica, logo
informativa) -- nunca como efeito principal. As interacoes com peso foram
testadas e pioraram, mas a regra estrutural vale para a proxima candidata.
"""
import numpy as np
import pandas as pd
import pytest

from src.train import build_logreg_pipeline


def _espelhado(n=400, seed=7):
    """n lutas -> 2n linhas espelhadas, com um diff informativo e uma
    coluna simetrica correlacionada com o resultado POR LUTA."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(size=n)
    simetrica = rng.normal(size=n)
    vence_a = rng.random(n) < 1 / (1 + np.exp(-skill))
    linhas = []
    for i in range(n):
        linhas.append({"diff": skill[i], "sim": simetrica[i], "label": int(vence_a[i])})
        linhas.append({"diff": -skill[i], "sim": simetrica[i], "label": int(not vence_a[i])})
    return pd.DataFrame(linhas)


def _coefs(df, cols):
    pipe = build_logreg_pipeline()
    pipe.fit(df[cols], df["label"])
    logreg = pipe.named_steps[list(pipe.named_steps)[-1]]
    return dict(zip(cols, logreg.coef_[0]))


class TestFeatureSimetrica:
    def test_coeficiente_de_feature_simetrica_e_zero(self):
        c = _coefs(_espelhado(), ["diff", "sim"])
        assert abs(c["sim"]) < 1e-6, f"coeficiente {c['sim']:.8f} deveria ser zero"
        assert abs(c["diff"]) > 0.1, "o diferencial precisa continuar informando"

    def test_vale_ate_quando_a_simetrica_PREDIZ_o_resultado(self):
        """O caso que engana: mesmo construindo a coluna simetrica a partir do
        proprio vencedor, ela continua inerte -- porque descreve a LUTA, e as
        duas linhas da luta tem labels opostos."""
        df = _espelhado()
        por_luta = df.groupby(df.index // 2)["label"].first()
        df["sim"] = np.repeat(por_luta.to_numpy(dtype=float), 2) * 3.0
        c = _coefs(df, ["diff", "sim"])
        assert abs(c["sim"]) < 1e-6, f"coeficiente {c['sim']:.8f} deveria ser zero"

    def test_a_INTERACAO_com_um_diferencial_informa(self):
        """O caminho que sobra para uma propriedade da luta: diff x simetrica
        inverte de sinal no espelho, entao carrega informacao."""
        df = _espelhado()
        df["inter"] = df["diff"] * df["sim"]
        c = _coefs(df, ["diff", "sim", "inter"])
        assert abs(c["sim"]) < 1e-6
        assert abs(c["inter"]) > 1e-6, "a interacao deveria ser aproveitavel"

    def test_stance_mismatch_e_simetrica_no_dataset_real(self):
        """Ancoragem no dado de verdade: se um dia stance_mismatch deixar de
        ser identica nas duas linhas, esta conclusao muda."""
        import config
        if not config.FEATURES_CSV.exists():
            pytest.skip("features nao geradas")
        f = pd.read_csv(config.FEATURES_CSV, usecols=["fight_id", "stance_mismatch"])
        assert f.groupby("fight_id")["stance_mismatch"].nunique(dropna=False).le(1).all()
