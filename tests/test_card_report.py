"""
Testes de src/card_report.py: ranking de favoritos/zebras, tratamento de
lutador desconhecido (grupo "sem previsao", nunca descartado em silencio)
e validacao do CSV de entrada -- com predict_fn injetado (sem depender de
modelo treinado nem da base real).
"""
import pandas as pd
import pytest

from src.card_report import analyze_card, load_card_odds, render_html

# Card sintetico com casos de CONCORDANCIA (modelo aponta o favorito do
# mercado), DIVERGENCIA (modelo aponta o azarao) e um desconhecido (Zed).
CARD = pd.DataFrame({
    "fighter_a": ["Alice Silva", "Carla Rocha", "Elisa Prado", "Gina Reis", "Zed Desconhecido"],
    "fighter_b": ["Bia Costa", "Dave Lima", "Fern Gil", "Hilda Luz", "Eva Nunes"],
    "odds_a_decimal": [1.20, 1.40, 1.60, 1.50, 1.50],
    "odds_b_decimal": [5.00, 2.90, 2.40, 2.60, 2.60],
})

# probabilidades do "modelo" fake para o lado A de cada luta
FAKE_MODEL_PROB_A = {
    ("Alice Silva", "Bia Costa"): 0.80,      # concorda com o favorito (forte)
    ("Carla Rocha", "Dave Lima"): 0.45,      # diverge: modelo aponta Dave (0.55)
    ("Elisa Prado", "Fern Gil"): 0.62,       # concorda com o favorito (moderado)
    ("Gina Reis", "Hilda Luz"): 0.35,        # diverge: modelo aponta Hilda (0.65)
}


def fake_predict(a: str, b: str) -> dict:
    if "Desconhecido" in a or "Desconhecido" in b:
        raise ValueError(f"Lutador '{a}' nao encontrado na base de dados.")
    prob_a = FAKE_MODEL_PROB_A[(a, b)]
    return {
        "fighter_a": a, "fighter_b": b,
        "prob_a_wins": prob_a, "prob_b_wins": 1 - prob_a,
        "model_used": "fake",
        "fighter_a_low_experience": False, "fighter_b_low_experience": False,
    }


FAKE_METHOD = {
    "method_probs": {"KO_TKO": 0.5, "SUBMISSION": 0.2, "DECISION": 0.3},
    "round_band_probs": {"1": 0.4, "2": 0.35, "3+": 0.25},
}


def fake_method_fn(a: str, b: str, scheduled_rounds: int) -> dict:
    # falha SO para a luta da Carla: metodo indisponivel, vencedor segue normal
    if a == "Carla Rocha":
        raise ValueError("sem dados de metodo para esta luta")
    return {"fighter_a": a, "fighter_b": b, **FAKE_METHOD, "model_used": "fake"}


@pytest.fixture
def analysis() -> dict:
    return analyze_card(CARD, model_name="fake", predict_fn=fake_predict,
                        method_fn=fake_method_fn)


class TestAnalyzeCard:
    def test_desconhecido_vai_para_sem_previsao_e_nao_quebra(self, analysis):
        assert len(analysis["no_prediction"]) == 1
        sem = analysis["no_prediction"][0]
        assert sem["fighter_a"] == "Zed Desconhecido"
        assert "nao encontrado" in sem["reason"]
        # e nao aparece nos rankings
        nomes = [f["fighter_a"] for f in analysis["favorites"] + analysis["underdogs"]]
        assert "Zed Desconhecido" not in nomes

    def test_categorias_mutuamente_exclusivas(self, analysis):
        """REGRA CENTRAL: cada luta com previsao valida aparece em exatamente
        UMA das duas listas -- nunca as duas, nunca nenhuma."""
        key = lambda f: (f["fighter_a"], f["fighter_b"])  # noqa: E731
        favs = {key(f) for f in analysis["favorites"]}
        dogs = {key(f) for f in analysis["underdogs"]}
        assert favs & dogs == set(), "luta presente nas duas abas!"
        # 4 lutas previstas: 2 concordancias + 2 divergencias, sem sobra
        assert len(favs) + len(dogs) == 4
        assert len(favs) == 2 and len(dogs) == 2

    def test_favoritos_ordenados_pela_prob_do_modelo(self, analysis):
        favs = analysis["favorites"]
        # Alice (modelo 0.80) vem antes de Elisa (modelo 0.62) -- criterio e a
        # probabilidade do MODELO, nao a de mercado
        assert [f["model_side"] for f in favs] == ["Alice Silva", "Elisa Prado"]
        assert favs[0]["model_side_prob"] == pytest.approx(0.80)
        assert favs[1]["model_side_prob"] == pytest.approx(0.62)
        # em todo favorito, o lado do modelo coincide com o favorito do mercado
        assert all(f["model_side"] == f["favorite"] for f in favs)

    def test_devig_conferido_a_mao(self, analysis):
        alice = next(f for f in analysis["favorites"] if f["favorite"] == "Alice Silva")
        # implicitas: 1/1.20=0.8333, 1/5.00=0.20; soma=1.0333
        assert alice["market_prob_fav"] == pytest.approx(0.8333 / 1.0333, abs=1e-3)
        assert alice["market_prob_dog"] == pytest.approx(0.20 / 1.0333, abs=1e-3)
        # os dois lados devigados somam 1
        assert alice["market_prob_fav"] + alice["market_prob_dog"] == pytest.approx(1.0)

    def test_zebras_sao_divergencias_reais_ordenadas_pelo_modelo(self, analysis):
        """Zebra = o modelo aponta o AZARAO do mercado como lado mais provavel
        de vencer (model_side != market_side), ordenado pela prob. do modelo."""
        dogs = analysis["underdogs"]
        # Hilda (modelo 0.65) vem antes de Dave (modelo 0.55)
        assert [f["model_side"] for f in dogs] == ["Hilda Luz", "Dave Lima"]
        assert dogs[0]["model_side_prob"] == pytest.approx(0.65)
        assert dogs[1]["model_side_prob"] == pytest.approx(0.55)
        # em toda zebra, o lado do modelo e o azarao do mercado
        assert all(f["model_side"] == f["underdog"] for f in dogs)
        # conferencia a mao do mercado devig de Dave (contexto exibido no card)
        market_dave = (1 / 2.90) / (1 / 1.40 + 1 / 2.90)
        dave = dogs[1]
        assert dave["market_prob_dog"] == pytest.approx(market_dave, abs=1e-4)

    def test_favorito_pode_estar_no_lado_b(self):
        card = pd.DataFrame({
            "fighter_a": ["Azarao Aqui"], "fighter_b": ["Favorita La"],
            "odds_a_decimal": [3.00], "odds_b_decimal": [1.40],
        })
        def predict(a, b):
            return {"fighter_a": a, "fighter_b": b, "prob_a_wins": 0.30, "prob_b_wins": 0.70,
                    "model_used": "fake", "fighter_a_low_experience": False,
                    "fighter_b_low_experience": False}
        res = analyze_card(card, predict_fn=predict)
        # modelo (B, 0.70) coincide com o favorito do mercado (B) -> favoritos
        assert len(res["underdogs"]) == 0
        f = res["favorites"][0]
        assert f["favorite"] == "Favorita La"
        assert f["model_side"] == "Favorita La"
        assert f["model_side_prob"] == pytest.approx(0.70)
        assert f["model_prob_dog"] == pytest.approx(0.30)


class TestMethodDurationIntegration:
    def test_falha_de_metodo_nao_derruba_previsao_de_vencedor(self, analysis):
        """As tres previsoes falham independentemente: a luta da Carla tem
        vencedor previsto (segue categorizada -- como zebra) mas metodo
        indisponivel."""
        carla = next(f for f in analysis["underdogs"] if f["fighter_a"] == "Carla Rocha")
        assert carla["method_probs"] is None
        assert carla["round_band_probs"] is None
        assert "model_side_prob" in carla  # vencedor intacto

    def test_luta_com_metodo_carrega_distribuicoes(self, analysis):
        alice = next(f for f in analysis["favorites"] if f["fighter_a"] == "Alice Silva")
        assert alice["method_probs"] == FAKE_METHOD["method_probs"]
        assert alice["round_band_probs"] == FAKE_METHOD["round_band_probs"]

    def test_sem_method_fn_nao_quebra(self):
        res = analyze_card(CARD, predict_fn=fake_predict)  # method_fn ausente
        assert all(f["method_probs"] is None
                   for f in res["favorites"] + res["underdogs"])

    def test_totals_market_calculado_e_falha_independente(self, analysis):
        alice = next(f for f in analysis["method_ranking"] if f["fighter_a"] == "Alice Silva")
        # FAKE_METHOD: P(fin)=0.7, bands {1: .4, 2: .35, 3+: .25}
        # under 1,5 = .7*.4 = .28; under 2,5 = .7*(.4+.35) = .525
        assert alice["totals_market"]["under_1_5"] == pytest.approx(0.28, abs=1e-4)
        assert alice["totals_market"]["over_1_5"] == pytest.approx(0.72, abs=1e-4)
        assert alice["totals_market"]["under_2_5"] == pytest.approx(0.525, abs=1e-4)
        assert alice["totals_market"]["over_2_5"] == pytest.approx(0.475, abs=1e-4)
        # Carla (metodo falhou) fica fora dos rankings novos, mas listada
        assert all(f["fighter_a"] != "Carla Rocha" for f in analysis["method_ranking"])
        assert all(f["fighter_a"] != "Carla Rocha" for f in analysis["duration_ranking"])
        assert any(f["fighter_a"] == "Carla Rocha" for f in analysis["no_method"])
        assert any(f["fighter_a"] == "Carla Rocha" for f in analysis["no_duration"])

    def test_render_abas_novas_e_sem_secao_antiga(self, analysis):
        html = render_html(analysis, freshness_gap_days=5)
        # abas novas presentes
        assert 'data-tab="method"' in html and "Método de vitória" in html
        assert 'data-tab="duration"' in html and "Duração da luta" in html
        # as DUAS linhas de over/under aparecem
        assert "Over 1,5" in html and "Under 1,5" in html
        assert "Over 2,5" in html and "Under 2,5" in html
        assert "KO/TKO" in html and "Finalização" in html and "Decisão" in html
        # a sub-secao antiga NAO existe mais dentro dos cards de favoritos/zebras
        assert "Como a luta tende a terminar" not in html
        # aviso especifico de odds justas, presente nas DUAS abas novas
        assert html.count("odds JUSTAS calculadas a partir da probabilidade do nosso modelo") == 2
        assert "validado contra o mercado real" in html
        # odds justas conferidas a mao: under 1,5 = 0.28 -> 3.57 (+257);
        # under 2,5 = 0.525 -> 1/0.525 = 1.905 -> exibe 1.91 (-111)
        assert "3.57" in html and "+257" in html
        assert "1.91" in html and "-111" in html
        # a luta da Carla aparece nas listas de indisponibilidade das abas novas
        assert "Sem previsão de método (1)" in html
        assert "Sem previsão de duração (1)" in html


class TestScheduledRounds:
    # NOTA: a restricao logica antiga (constrain_round_bands, zerar "4-5" em
    # luta de 3 rounds) foi removida junto com seus testes -- as faixas novas
    # {1, 2, 3+} sao validas para qualquer formato de luta. scheduled_rounds
    # continua relevante como FEATURE do modelo de faixa (testes abaixo).

    def test_coluna_do_csv_chega_ao_preditor_de_metodo(self):
        recebidos = {}
        def spy_method_fn(a, b, sr):
            recebidos[(a, b)] = sr
            return {"fighter_a": a, "fighter_b": b, **FAKE_METHOD, "model_used": "fake"}
        card = CARD.copy()
        card["scheduled_rounds"] = [5, 3, 3, 3, 3]
        analyze_card(card, predict_fn=fake_predict, method_fn=spy_method_fn)
        assert recebidos[("Alice Silva", "Bia Costa")] == 5
        assert recebidos[("Carla Rocha", "Dave Lima")] == 3

    def test_csv_sem_coluna_assume_3(self, tmp_path):
        p = tmp_path / "card.csv"
        pd.DataFrame({"fighter_a": ["X"], "fighter_b": ["Y"],
                      "odds_a_decimal": [1.5], "odds_b_decimal": [2.6]}).to_csv(p, index=False)
        df = load_card_odds(p)
        assert (df["scheduled_rounds"] == 3).all()

    def test_render_mostra_formato_da_luta_na_aba_duracao(self):
        res = analyze_card(CARD, predict_fn=fake_predict, method_fn=fake_method_fn)
        html = render_html(res, freshness_gap_days=5)
        # o card de duracao informa o formato (CARD default = 3 rounds)
        assert "luta de 3 rounds" in html


class TestLoadCardOdds:
    def test_coluna_faltando_da_erro_claro(self, tmp_path):
        p = tmp_path / "card.csv"
        pd.DataFrame({"fighter_a": ["X"], "fighter_b": ["Y"],
                      "odds_a_decimal": [1.5]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="odds_b_decimal"):
            load_card_odds(p)

    def test_odds_invalidas_dao_erro(self, tmp_path):
        p = tmp_path / "card.csv"
        pd.DataFrame({"fighter_a": ["X"], "fighter_b": ["Y"],
                      "odds_a_decimal": [0.95], "odds_b_decimal": [2.0]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="> 1.0"):
            load_card_odds(p)


class TestRenderHtml:
    def test_conteudo_essencial(self, analysis):
        html = render_html(analysis, freshness_gap_days=5, card_name="Card Teste")
        for token in ["Favoritos mais seguros", "Melhores zebras",
                      "Método de vitória", "Duração da luta",
                      "não é recomendação de aposta", "Sem previsão (1)",
                      "Zed Desconhecido", "Base de dados em dia"]:
            assert token in html, token

    def test_dados_desatualizados_aparecem_no_relatorio(self, analysis):
        html = render_html(analysis, freshness_gap_days=45)
        assert "DADOS DESATUALIZADOS" in html
        assert "45 dias" in html

    def test_sem_dependencias_externas(self, analysis):
        import re
        html = render_html(analysis, freshness_gap_days=5)
        assert not re.search(r'src="http|href="http|@import|url\(http', html)
