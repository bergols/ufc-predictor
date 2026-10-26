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


@pytest.mark.skipif(not FEATURES.exists(), reason="features nao geradas")
class TestSimetriaDosEspelhos:
    def test_colunas_de_soma_sao_IGUAIS_entre_os_espelhos(self):
        """
        As `_diff` invertem de sinal entre A-B e B-A; as `_sum` e o
        `experience_total` descrevem a LUTA, nao um lado, entao tem de ser
        identicas nas duas linhas. Se uma juncao duplicar ou trocar de lado,
        as `_diff` podem continuar simetricas por acaso e estas nao.

        Era o lado do espelho que nenhum teste olhava.
        """
        df = pd.read_csv(FEATURES)
        cols = [c for c in df.columns if c.endswith("_sum") or c == "experience_total"]
        assert cols, "nenhuma coluna simetrica encontrada"
        for c in cols:
            distintos = df.groupby("fight_id")[c].nunique(dropna=False)
            assert (distintos <= 1).all(), \
                f"{c} difere entre os espelhos em {int((distintos > 1).sum())} luta(s)"

    def test_nenhuma_feature_tem_infinito(self):
        """Divisao por zero numa taxa (lutador com 0 tentativas) vira inf e
        passa por todo o resto do pipeline sem reclamar ate o modelo."""
        df = pd.read_csv(FEATURES).select_dtypes(include=[np.number])
        ruins = [c for c in df.columns
                 if np.isinf(np.nan_to_num(df[c].to_numpy(dtype="float64"))).any()]
        assert not ruins, f"coluna(s) com infinito: {ruins}"


@pytest.mark.skipif(not (config.UPCOMING_CARD_CSV.exists() and HISTORICO.exists()
                         and config.ODDS_TEMPLATE_CSV.exists()),
                    reason="artefatos de card ausentes")
class TestCardTemplateHistorico:
    """
    Os tres arquivos do ciclo de evento descrevem o MESMO card e sao escritos
    em momentos diferentes: o CSV na montagem, o template logo depois, o
    historico na publicacao. Sair de sincronia e silencioso -- o relatorio sai
    bonito com uma luta a menos, ou o template fica com luta que nunca foi
    registrada e o sync depois nao acha onde encaixar o vencedor.
    """
    def _par(self, df):
        return {tuple(sorted([a, b])) for a, b in zip(df["fighter_a"], df["fighter_b"])}

    def test_o_card_atual_bate_com_template_e_historico(self):
        """
        O CSV do card manda: as lutas que estao nele tem de estar no historico
        e no template, todas sob UMA data de evento.

        Nao da para perguntar "qual e o evento aberto mais recente" -- linha
        aberta sobrevive ao evento quando a luta e cancelada ou empata
        (Kareckaite x Gatto, 10/out), e ai a pergunta devolve um card antigo.
        Foi o que este teste respondeu errado no primeiro dia de vida.
        """
        card = pd.read_csv(config.UPCOMING_CARD_CSV)
        hist = pd.read_csv(HISTORICO)
        pares = self._par(card)
        achadas = hist[hist.apply(
            lambda r: tuple(sorted([r["fighter_a"], r["fighter_b"]])) in pares, axis=1)]
        faltando = pares - self._par(achadas)
        assert not faltando, f"luta(s) do card fora do historico: {sorted(faltando)[:3]}"
        datas = set(achadas["event_date"])
        assert len(datas) == 1, f"o card do CSV aparece em {len(datas)} datas: {sorted(datas)}"

        data = datas.pop()
        t = pd.read_csv(config.ODDS_TEMPLATE_CSV)
        t = t[t["event_date"] == data]
        assert pares == self._par(t), "card e template divergem"
        assert len(hist[hist["event_date"] == data]) == len(card),             "o historico tem lutas desta data que nao estao no CSV do card"

    def test_template_sem_luta_repetida(self):
        t = pd.read_csv(config.ODDS_TEMPLATE_CSV)
        dup = t.duplicated(subset=["event_date", "fighter_a", "fighter_b"])
        assert not dup.any(), f"{int(dup.sum())} linha(s) repetida(s) no template"
