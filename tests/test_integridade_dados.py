"""
Invariantes sobre os DADOS REAIS (não sobre fixtures).

Os testes de unidade cobrem funções; estes cobrem os artefatos. A diferença
importa porque os três defeitos encontrados em 06/09/2026 passavam por todos os
testes de unidade e só apareciam no arquivo gerado:

  - stats duplicadas viraram 128 linhas por luta em 22 lutas (14% do treino);
  - homônimo trocava a bio do lutador (idade 48,9 no main event);
  - sharp_prob seguia o lado ANTIGO quando a previsão virava, invertendo o
    sinal da divergência.

Todos são de junção e de chave — o arquivo fica plausível, os números ficam
errados. É esse tipo de coisa que se cobra aqui.

Pulam quando o artefato não existe: um clone novo não tem dado processado.
"""
import numpy as np
import pandas as pd
import pytest

import config
from src.utils import decimal_odds_to_implied_prob, remove_vig_two_way

FEATURES = config.FEATURES_CSV
HISTORICO = config.PREDICTION_HISTORY_CSV
BRUTOS = config.RAW_FIGHTS_CSV


@pytest.mark.skipif(not BRUTOS.exists(), reason="dados brutos ausentes")
class TestBrutos:
    def test_uma_luta_por_fight_url(self):
        f = pd.read_csv(BRUTOS)
        assert f["fight_url"].duplicated().sum() == 0

    def test_stats_no_maximo_duas_linhas_por_luta(self):
        """O defeito das 128 linhas comecava aqui, com 4 em vez de 2."""
        if not config.RAW_FIGHT_STATS_CSV.exists():
            pytest.skip("stats ausentes")
        s = pd.read_csv(config.RAW_FIGHT_STATS_CSV)
        demais = s.groupby("fight_url").size()
        assert (demais <= 2).all(), f"{(demais > 2).sum()} luta(s) com stats repetidas"

    def test_vencedor_e_um_dos_dois_lados(self):
        f = pd.read_csv(BRUTOS).dropna(subset=["winner"])
        fora = ~f.apply(lambda r: r["winner"] in (r["fighter_1"], r["fighter_2"]), axis=1)
        assert not fora.any(), f"{fora.sum()} luta(s) com vencedor de fora"


@pytest.mark.skipif(not FEATURES.exists(), reason="features nao geradas")
class TestFeatures:
    def test_duas_linhas_espelhadas_por_luta(self):
        n = pd.read_csv(FEATURES, usecols=["fight_id"]).groupby("fight_id").size()
        assert (n == 2).all(), f"{(n != 2).sum()} luta(s) fora do par"

    def test_labels_das_linhas_espelhadas_sao_complementares(self):
        df = pd.read_csv(FEATURES, usecols=["fight_id", "label"])
        assert set(df.groupby("fight_id")["label"].sum().unique()) <= {1}

    def test_colunas_de_diferenca_invertem_entre_espelhos(self):
        """a-b numa linha tem de ser -(b-a) na outra. Se uma juncao duplicar,
        as duas linhas deixam de ser espelho e isto acusa."""
        df = pd.read_csv(FEATURES)
        cols = [c for c in df.columns if "_diff" in c]
        assert cols, "nenhuma coluna _diff encontrada"
        for c in cols:
            somas = df.groupby("fight_id")[c].apply(
                lambda x: abs(x.sum()) if x.notna().all() else 0.0)
            assert (somas < 1e-6).all(), f"{c} nao inverte em {(somas >= 1e-6).sum()} luta(s)"


@pytest.mark.skipif(not HISTORICO.exists(), reason="historico ausente")
class TestHistorico:
    def _h(self):
        return pd.read_csv(HISTORICO)

    def test_uma_linha_por_luta_e_data(self):
        h = self._h()
        assert h.duplicated(subset=["event_date", "fighter_a", "fighter_b"]).sum() == 0

    def test_model_side_e_actual_winner_sao_lados_validos(self):
        h = self._h()
        for col in ("model_side", "actual_winner"):
            sub = h.dropna(subset=[col])
            fora = ~sub.apply(lambda r: r[col] in (r["fighter_a"], r["fighter_b"]), axis=1)
            assert not fora.any(), f"{fora.sum()} linha(s) com {col} de fora da luta"

    def test_clv_e_a_diferenca_que_diz_ser(self):
        h = self._h().dropna(subset=["clv", "close_prob", "sharp_prob"])
        if h.empty:
            pytest.skip("sem CLV medido ainda")
        erro = (h["clv"] - (h["close_prob"] - h["sharp_prob"])).abs()
        assert (erro < 1e-6).all(), f"maior erro {erro.max():.6f}"

    @pytest.mark.parametrize("coluna", ["sharp_prob", "close_prob"])
    def test_probabilidade_de_mercado_descreve_o_LADO_DO_MODELO(self, coluna):
        """
        O bug de 06/09: sharp_prob e close_prob sao do lado que o modelo
        aponta. Quando a previsao vira entre dois pre-registros, o valor
        guardado passa a descrever o ADVERSARIO -- e o CLV troca de sinal sem
        nenhum aviso.

        Aqui a odd registrada serve de testemunha independente: devigada, ela
        diz aproximadamente onde o mercado punha aquele lado. Tolerancia larga
        (0,15) porque a odd e mediana entre casas e a sharp e so a Pinnacle; o
        que se quer pegar e a INVERSAO, que erra por muito mais que isso.
        """
        h = self._h().dropna(subset=[coluna, "model_side",
                                     "odds_a_decimal", "odds_b_decimal"])
        if h.empty:
            pytest.skip(f"sem {coluna} ainda")
        invertidas = []
        for _, r in h.iterrows():
            da, db = remove_vig_two_way(
                decimal_odds_to_implied_prob(r["odds_a_decimal"]),
                decimal_odds_to_implied_prob(r["odds_b_decimal"]))
            esperado = da if r["model_side"] == r["fighter_a"] else db
            if abs(r[coluna] - esperado) > abs(r[coluna] - (1 - esperado)) + 0.15:
                invertidas.append(f"{r['event_date']} {r['fighter_a']} vs {r['fighter_b']}")
        assert not invertidas, f"{coluna} parece do lado errado em: {invertidas[:3]}"


@pytest.mark.skipif(not (FEATURES.exists() and BRUTOS.exists()),
                    reason="dados ausentes")
class TestPointInTime:
    def test_experiencia_nao_conta_a_luta_corrente_nem_o_futuro(self):
        """
        A propriedade que sustenta o projeto inteiro, conferida contra uma
        recontagem INDEPENDENTE do pipeline.

        Noite de torneio fica de fora: nos anos 90 o mesmo lutador lutava 2-4
        vezes na mesma data, e ai "lutas anteriores" depende da ordem dentro da
        noite, que a recontagem por data nao conhece. Nao e vazamento -- a
        vitoria da luta 1 e informacao real antes da luta 2 -- e essas lutas
        estao fora da era de treino de qualquer jeito.
        """
        ft = pd.read_csv(FEATURES, parse_dates=["event_date"],
                         usecols=["fight_id", "event_date", "fighter_a",
                                  "fighter_b", "experience_diff"])
        base = pd.read_csv(BRUTOS, parse_dates=["event_date"])
        longo = pd.concat([
            base[["fighter_1", "event_date"]].rename(columns={"fighter_1": "fighter"}),
            base[["fighter_2", "event_date"]].rename(columns={"fighter_2": "fighter"}),
        ]).dropna()
        por_dia = longo.groupby(["fighter", "event_date"]).size()
        noite_de_torneio = set(por_dia[por_dia > 1].index)

        longo = longo.sort_values("event_date")
        longo["n_antes"] = longo.groupby("fighter").cumcount()
        antes = longo.groupby(["fighter", "event_date"])["n_antes"].min().sort_index()

        amostra = ft.sample(min(2500, len(ft)), random_state=config.RANDOM_SEED)
        divergentes = []
        for _, r in amostra.iterrows():
            ka, kb = (r["fighter_a"], r["event_date"]), (r["fighter_b"], r["event_date"])
            if ka not in antes.index or kb not in antes.index:
                continue
            if ka in noite_de_torneio or kb in noite_de_torneio:
                continue
            esperado = int(antes.loc[ka]) - int(antes.loc[kb])
            if abs(r["experience_diff"] - esperado) > 1e-6:
                divergentes.append(
                    f"{r['event_date'].date()} {r['fighter_a']} vs {r['fighter_b']}: "
                    f"{r['experience_diff']:.0f} != {esperado}")
        assert not divergentes, f"{len(divergentes)} divergencia(s): {divergentes[:3]}"
