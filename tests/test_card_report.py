"""
Testes de src/card_report.py: ranking de favoritos/zebras, tratamento de
lutador desconhecido (grupo "sem previsao", nunca descartado em silencio)
e validacao do CSV de entrada -- com predict_fn injetado (sem depender de
modelo treinado nem da base real).
"""
import pandas as pd
import pytest

from src.card_report import analyze_card, load_card_odds, render_html
from src.prediction_history import frozen_predictions_for_event

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
}


def fake_method_fn(a: str, b: str) -> dict:
    # falha SO para a luta da Carla: metodo indisponivel, vencedor segue normal
    if a == "Carla Rocha":
        raise ValueError("sem dados de metodo para esta luta")
    return {"fighter_a": a, "fighter_b": b, **FAKE_METHOD, "model_used": "fake"}


def _find(res: dict, fighter_a: str) -> dict:
    """A luta cujo fighter_a e o dado, esteja ela em favoritos ou zebras."""
    for f in res["favorites"] + res["underdogs"]:
        if f["fighter_a"] == fighter_a:
            return f
    raise AssertionError(f"luta de {fighter_a} nao encontrada em nenhuma aba")


def _frozen(prob_a: float | None, method: dict | None = None) -> dict:
    """Entrada congelada da luta da Alice, no formato de
    prediction_history.frozen_predictions_for_event."""
    return {("Alice Silva", "Bia Costa"): {"model_prob_a": prob_a,
                                           "method_probs": method}}


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


class TestMethodIntegration:
    def test_falha_de_metodo_nao_derruba_previsao_de_vencedor(self, analysis):
        """As duas previsoes falham independentemente: a luta da Carla tem
        vencedor previsto (segue categorizada -- como zebra) mas metodo
        indisponivel."""
        carla = next(f for f in analysis["underdogs"] if f["fighter_a"] == "Carla Rocha")
        assert carla["method_probs"] is None
        assert "model_side_prob" in carla  # vencedor intacto

    def test_luta_com_metodo_carrega_distribuicoes(self, analysis):
        alice = next(f for f in analysis["favorites"] if f["fighter_a"] == "Alice Silva")
        assert alice["method_probs"] == FAKE_METHOD["method_probs"]

    def test_sem_method_fn_nao_quebra(self):
        res = analyze_card(CARD, predict_fn=fake_predict)  # method_fn ausente
        assert all(f["method_probs"] is None
                   for f in res["favorites"] + res["underdogs"])

    def test_luta_sem_metodo_fica_fora_do_ranking_mas_listada(self, analysis):
        assert all(f["fighter_a"] != "Carla Rocha" for f in analysis["method_ranking"])
        assert any(f["fighter_a"] == "Carla Rocha" for f in analysis["no_method"])

    def test_render_aba_de_metodo(self, analysis):
        html = render_html(analysis, freshness_gap_days=5)
        assert 'data-tab="method"' in html and "Método de vitória" in html
        assert "KO/TKO" in html and "Finalização" in html and "Decisão" in html
        # a sub-secao antiga NAO existe mais dentro dos cards de favoritos/zebras
        assert "Como a luta tende a terminar" not in html
        # aviso de odds justas: so a aba de metodo sobrou, entao aparece 1x
        assert html.count("odds JUSTAS calculadas a partir da probabilidade do nosso modelo") == 1
        assert "validado contra o mercado real" in html
        assert "Sem previsão de método (1)" in html


class TestDuracaoRemovida:
    """
    A previsao de duracao saiu em ago/2026 (ver cabecalho de
    src/train_method.py). Estes testes impedem que a aba volte por acidente
    junto com algum merge.
    """
    def test_nenhum_vestigio_da_aba_no_html(self, analysis):
        html = render_html(analysis, freshness_gap_days=5)
        for token in ['data-tab="duration"', 'id="duration"', "Duração da luta",
                      "Over 1,5", "Under 1,5", "Over 2,5", "Under 2,5",
                      "Sem previsão de duração"]:
            assert token not in html, token

    def test_analise_nao_expoe_mais_as_chaves_de_duracao(self, analysis):
        for chave in ("duration_ranking", "no_duration"):
            assert chave not in analysis, chave
        for f in analysis["favorites"] + analysis["underdogs"]:
            assert "totals_market" not in f
            assert "round_band_probs" not in f

    def test_scheduled_rounds_no_csv_e_ignorada_sem_quebrar(self, tmp_path):
        # CSVs antigos trazem a coluna; ela nao pode mais causar erro
        p = tmp_path / "card.csv"
        pd.DataFrame({"fighter_a": ["X"], "fighter_b": ["Y"],
                      "odds_a_decimal": [1.5], "odds_b_decimal": [2.6],
                      "scheduled_rounds": [5]}).to_csv(p, index=False)
        df = load_card_odds(p)
        assert len(df) == 1

    def test_method_fn_recebe_so_os_dois_lutadores(self):
        recebidos = []
        def spy_method_fn(a, b):
            recebidos.append((a, b))
            return {"fighter_a": a, "fighter_b": b, **FAKE_METHOD, "model_used": "fake"}
        analyze_card(CARD, predict_fn=fake_predict, method_fn=spy_method_fn)
        assert ("Alice Silva", "Bia Costa") in recebidos


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
                      "Método de vitória",
                      "não é recomendação de aposta", "Sem previsão (1)",
                      "Zed Desconhecido", "Base de dados em dia"]:
            assert token in html, token

    def test_dados_desatualizados_aparecem_no_relatorio(self, analysis):
        html = render_html(analysis, freshness_gap_days=45)
        assert "DADOS DESATUALIZADOS" in html
        assert "45 dias" in html

    def test_sem_dependencias_externas_alem_das_fotos(self, analysis):
        """
        CSS, JS, icones e fontes tem de ser embutidos: a pagina nao pode
        depender de CDN nem de fonte remota, senao o layout quebra offline.
        As FOTOS dos lutadores sao a unica excecao (hotlink do UFC.com, ver
        fighter_photos.py) e nao entram aqui porque a analise de teste roda
        sem mapa de fotos — que e exatamente o caminho de fallback.
        """
        import re
        html = render_html(analysis, freshness_gap_days=5)
        assert not re.search(r'src="http|href="http|@import|url\(http', html)


class TestPrevisaoCongelada:
    """
    analyze_card recalcula a previsao a cada geracao usando os niveis ATUAIS
    dos lutadores. Depois que o evento entra na base (fill-gap), esses niveis
    ja incluem o proprio resultado -- regerar um card encerrado produzia
    previsoes que enxergavam quem ganhou.

    Medido no card de 08/ago/2026: as 10 lutas se moveram na direcao do
    vencedor real (Manoel Sousa 0.546 -> 0.673, Gamrot 0.350 -> 0.258) e as 2
    que o modelo errou inverteram para o lado certo. A pagina mostraria o
    modelo com 10/10 nas abas de Favoritos/Zebras enquanto a aba Historico,
    essa sim congelada, dizia 8/10.
    """
    def test_luta_fechada_usa_a_previsao_publicada(self):
        # o modelo "agora" diz 0.80 para Alice; publicado foi 0.30
        res = analyze_card(CARD, predict_fn=fake_predict, frozen=_frozen(0.30))
        alice = _find(res, "Alice Silva")
        assert alice["model_prob_a"] == 0.30
        assert alice["frozen"] is True

    def test_luta_aberta_segue_recalculando(self):
        # sem resultado ainda, regerar para atualizar odds deve refrescar a
        # previsao -- mesma regra de record_card_predictions p/ linha aberta
        res = analyze_card(CARD, predict_fn=fake_predict, frozen={})
        alice = _find(res, "Alice Silva")
        assert alice["model_prob_a"] == 0.80
        assert alice["frozen"] is False

    def test_ordem_invertida_no_historico_e_espelhada(self):
        # model_prob_a e relativo ao fighter_a de QUEM GRAVOU
        frozen = {("Bia Costa", "Alice Silva"): {"model_prob_a": 0.70,
                                                 "method_probs": None}}
        res = analyze_card(CARD, predict_fn=fake_predict, frozen=frozen)
        alice = _find(res, "Alice Silva")
        assert alice["model_prob_a"] == pytest.approx(0.30)
        assert alice["frozen"] is True

    def test_congelamento_manda_na_categoria_e_no_ev(self):
        # o recalculo poria Alice em "favoritos" (0.80, concorda com o
        # mercado); a previsao publicada aponta Bia -> zebra, e o EV tem de
        # sair da odd de Bia, nao da de Alice
        res = analyze_card(CARD, predict_fn=fake_predict, frozen=_frozen(0.30))
        alice = _find(res, "Alice Silva")
        assert alice["category"] == "underdog"
        assert alice["model_side"] == "Bia Costa"
        assert alice["ev"] == pytest.approx(0.70 * 5.00)

    def test_sem_frozen_nao_muda_nada(self):
        base = analyze_card(CARD, predict_fn=fake_predict)
        com = analyze_card(CARD, predict_fn=fake_predict, frozen=None)
        assert [f["model_prob_a"] for f in base["favorites"]] == \
               [f["model_prob_a"] for f in com["favorites"]]


class TestMetodoCongelado:
    """
    O metodo tem o mesmo problema do vencedor: analyze_card o recalcula a
    cada geracao, e depois do fill-gap a base ja contem o resultado da luta.
    Congelado no pre-registro desde ago/2026 (3 colunas no historico).
    """
    PUBLICADO = {"KO_TKO": 0.10, "SUBMISSION": 0.15, "DECISION": 0.75}

    def test_luta_fechada_exibe_o_metodo_publicado(self):
        frozen = _frozen(0.30, self.PUBLICADO)
        res = analyze_card(CARD, predict_fn=fake_predict, method_fn=fake_method_fn,
                           frozen=frozen)
        alice = _find(res, "Alice Silva")
        # o recalculo diria FAKE_METHOD (KO 0.5); o publicado manda
        assert alice["method_probs"] == self.PUBLICADO

    def test_luta_aberta_segue_recalculando_o_metodo(self):
        res = analyze_card(CARD, predict_fn=fake_predict, method_fn=fake_method_fn,
                           frozen={})
        alice = _find(res, "Alice Silva")
        assert alice["method_probs"] == FAKE_METHOD["method_probs"]

    def test_fechada_sem_metodo_congelado_fica_sem_metodo(self):
        """Eventos anteriores as colunas de metodo: preferimos a lacuna ao
        numero contaminado -- a luta cai na lista 'sem previsao de método'."""
        res = analyze_card(CARD, predict_fn=fake_predict, method_fn=fake_method_fn,
                           frozen=_frozen(0.30, None))
        alice = _find(res, "Alice Silva")
        assert alice["method_probs"] is None
        assert all(f["fighter_a"] != "Alice Silva" for f in res["method_ranking"])
        assert any(f["fighter_a"] == "Alice Silva" for f in res["no_method"])

    def test_metodo_nao_inverte_com_a_ordem_dos_lados(self):
        """As labels de metodo sao simetricas: KO/finalizacao/decisao nao
        dependem de quem e 'A'."""
        frozen = {("Bia Costa", "Alice Silva"): {"model_prob_a": 0.70,
                                                 "method_probs": self.PUBLICADO}}
        res = analyze_card(CARD, predict_fn=fake_predict, method_fn=fake_method_fn,
                           frozen=frozen)
        assert _find(res, "Alice Silva")["method_probs"] == self.PUBLICADO

    def test_method_fn_nem_e_chamado_para_luta_fechada(self):
        chamadas = []
        def spy(a, b):
            chamadas.append((a, b))
            return {"fighter_a": a, "fighter_b": b, **FAKE_METHOD, "model_used": "fake"}
        analyze_card(CARD, predict_fn=fake_predict, method_fn=spy,
                     frozen=_frozen(0.30, self.PUBLICADO))
        assert ("Alice Silva", "Bia Costa") not in chamadas
        assert ("Carla Rocha", "Dave Lima") in chamadas   # aberta: recalcula


class TestFrozenPredictionsForEvent:
    HEADER = ("event_name,event_date,fighter_a,fighter_b,odds_a_decimal,odds_b_decimal,"
              "model_name,model_prob_a,model_side,actual_winner,sharp_prob,sharp_best_odd,"
              "ev_sharp,method_ko_tko,method_submission,method_decision\n")

    def _hist(self, tmp_path, linhas):
        p = tmp_path / "history.csv"
        p.write_text(self.HEADER + "".join(linhas), encoding="utf-8")
        return p

    def test_so_lutas_fechadas_entram(self, tmp_path):
        p = self._hist(tmp_path, [
            "Card,2026-08-08,Alice Silva,Bia Costa,1.2,5.0,logreg,0.30,Bia Costa,Bia Costa,,,,0.1,0.15,0.75\n",
            "Card,2026-08-08,Carla Rocha,Dave Lima,1.4,2.9,logreg,0.45,Dave Lima,,,,,0.2,0.2,0.6\n",
        ])
        frozen = frozen_predictions_for_event("2026-08-08", history_csv=p)
        assert frozen == {("Alice Silva", "Bia Costa"): {
            "model_prob_a": 0.30,
            "method_probs": {"KO_TKO": 0.10, "SUBMISSION": 0.15, "DECISION": 0.75}}}

    def test_outro_evento_nao_vaza(self, tmp_path):
        p = self._hist(tmp_path, [
            "Card,2026-08-01,Alice Silva,Bia Costa,1.2,5.0,logreg,0.30,Bia Costa,Bia Costa,,,,,,\n",
        ])
        assert frozen_predictions_for_event("2026-08-08", history_csv=p) == {}

    def test_luta_sem_previsao_entra_com_valores_none(self, tmp_path):
        """Precisa entrar: e assim que o relatorio sabe que a luta esta
        FECHADA e nao deve recalcular nada."""
        p = self._hist(tmp_path, [
            "Card,2026-08-08,Zed Desconhecido,Eva Nunes,1.5,2.6,logreg,,,Eva Nunes,,,,,,\n",
        ])
        assert frozen_predictions_for_event("2026-08-08", history_csv=p) == {
            ("Zed Desconhecido", "Eva Nunes"): {"model_prob_a": None, "method_probs": None}}

    def test_metodo_parcial_e_descartado(self, tmp_path):
        # as tres classes precisam somar 1; meia distribuicao nao serve
        p = self._hist(tmp_path, [
            "Card,2026-08-08,Alice Silva,Bia Costa,1.2,5.0,logreg,0.30,Bia Costa,Bia Costa,,,,0.1,,\n",
        ])
        frozen = frozen_predictions_for_event("2026-08-08", history_csv=p)
        assert frozen[("Alice Silva", "Bia Costa")]["method_probs"] is None

    def test_historico_antigo_sem_as_colunas_de_metodo(self, tmp_path):
        """Historico gravado antes de ago/2026 nao tem as 3 colunas -- nao e
        corrupcao, e luta fechada sem metodo congelado."""
        p = tmp_path / "history.csv"
        p.write_text("event_name,event_date,fighter_a,fighter_b,odds_a_decimal,odds_b_decimal,"
                     "model_name,model_prob_a,model_side,actual_winner\n"
                     "Card,2026-08-08,Alice Silva,Bia Costa,1.2,5.0,logreg,0.30,Bia Costa,Bia Costa\n",
                     encoding="utf-8")
        frozen = frozen_predictions_for_event("2026-08-08", history_csv=p)
        assert frozen == {("Alice Silva", "Bia Costa"): {"model_prob_a": 0.30,
                                                         "method_probs": None}}

    def test_historico_inexistente_nao_quebra(self, tmp_path):
        assert frozen_predictions_for_event("2026-08-08",
                                            history_csv=tmp_path / "nao_existe.csv") == {}
