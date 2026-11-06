"""
Ciclo completo de um evento, ponta a ponta, em dados sintéticos.

Os testes existentes cobrem cada peça isolada. O que faltava era a
**sequência**: montar o card, pré-registrar, congelar o sinal sharp com
proveniência, publicar, capturar o fechamento, preencher o vencedor e regerar
o relatório — na ordem em que isso acontece de verdade, por
`generate_card_report`.

A propriedade que o ciclo inteiro existe para proteger é uma só:

    a previsão publicada antes da luta é a previsão que fica.

Então o teste **troca o preditor** entre a publicação e o fechamento. Se
alguma etapa recalcular, a probabilidade gravada muda e o teste falha. É o
mesmo erro que a ordem `sync` → `record` em `generate_card_report` previne, e
aqui ele é verificado de fora, pelo resultado, e não pela leitura do código.

Nada aqui toca artefato de produção: todos os caminhos de `config` são
redirecionados para `tmp_path`, e a trava de `conftest.py` confere isso.
"""
import pandas as pd
import pytest

import config
from src import card_report, prediction_history as ph, proveniencia


def _dia(delta: int) -> str:
    return (pd.Timestamp.today() + pd.Timedelta(days=delta)).strftime("%Y-%m-%d")


CARD = pd.DataFrame([
    # a primeira linha e a luta principal, como no CSV de verdade
    {"fighter_a": "Alice Silva", "fighter_b": "Bia Costa",
     "odds_a_decimal": 1.60, "odds_b_decimal": 2.40,
     "odds_a_tp": 1.72, "odds_b_tp": 2.30, "fonte": "ambas",
     "ref_tp": "REF_TP"},
    {"fighter_a": "Carla Rocha", "fighter_b": "Dora Lima",
     "odds_a_decimal": 1.90, "odds_b_decimal": 1.95,
     "odds_a_tp": 1.98, "odds_b_tp": 1.90, "fonte": "ambas",
     "ref_tp": "REF_TP"},
])

# As probabilidades do pre-registro e as do "re-treino" posterior. Diferentes
# de proposito: e a diferenca entre as duas que denuncia um recalculo.
ANTES = {("Alice Silva", "Bia Costa"): 0.6400,
         ("Carla Rocha", "Dora Lima"): 0.4800}
DEPOIS = {("Alice Silva", "Bia Costa"): 0.9900,
          ("Carla Rocha", "Dora Lima"): 0.0100}


def _preditor(tabela):
    """Par (predict_fn, method_fn) no formato que `analyze_card` consome."""
    def predict(a, b):
        return {"fighter_a": a, "fighter_b": b,
                "prob_a_wins": tabela[(a, b)],
                "prob_b_wins": 1 - tabela[(a, b)],
                "model_used": "fake",
                "fighter_a_low_experience": False,
                "fighter_b_low_experience": False,
                "fighter_a_debutant": False, "fighter_b_debutant": False}

    def method(a, b):
        return {"method_probs": {"KO_TKO": 0.4, "SUBMISSION": 0.2,
                                 "DECISION": 0.4}}
    return predict, method


@pytest.fixture
def mundo(tmp_path, monkeypatch):
    """Um projeto inteiro em miniatura, dentro de tmp_path."""
    from src import event_watch

    card_csv = tmp_path / "upcoming_card_odds.csv"
    CARD.to_csv(card_csv, index=False)

    template = tmp_path / "odds_template.csv"
    tpl = CARD[["fighter_a", "fighter_b", "odds_a_decimal", "odds_b_decimal"]].copy()
    tpl["actual_winner"] = ""
    tpl.to_csv(template, index=False)

    historico = tmp_path / "prediction_history.csv"
    monkeypatch.setattr(config, "PREDICTION_HISTORY_CSV", historico)
    monkeypatch.setattr(config, "ODDS_TEMPLATE_CSV", template)
    monkeypatch.setattr(config, "UPCOMING_CARD_CSV", card_csv)
    monkeypatch.setattr(config, "RAW_FIGHTS_CSV", tmp_path / "fights.csv")
    monkeypatch.setattr(event_watch, "CACHE_PENDENTES", tmp_path / "pendentes.json")

    # proveniencia isolada: o indice de verdade nao recebe linha de teste
    pasta = tmp_path / "proveniencia"
    monkeypatch.setattr(proveniencia, "PASTA", pasta)
    monkeypatch.setattr(proveniencia, "INDICE", pasta / "chamadas.csv")
    # PROJECT_ROOT NAO e redirecionado: `generator_fingerprint` le o codigo
    # do gerador a partir dele, e a proveniencia ja lida com pasta fora da
    # raiz guardando o caminho absoluto.

    # sem foto e sem freshness: dependeriam de rede e da base real
    monkeypatch.setattr(card_report, "set_photo_map", lambda m: None)
    from src import data_collection
    monkeypatch.setattr(data_collection, "check_data_freshness", lambda: 0)

    return {"card_csv": card_csv, "template": template,
            "historico": historico, "saida": tmp_path / "index.html",
            "tmp": tmp_path}


def _publica(mundo, monkeypatch, tabela, data, sharp_com_prova=None, **kw):
    """
    Publica o card. `sharp_com_prova`:

        None   nao consulta sharp nenhum (sharp=False);
        True   congela sharp COM proveniencia registrada;
        False  congela sharp SEM proveniencia -- o estado de hoje, em que o
               preco existe e a prova dele nao.
    """
    monkeypatch.setattr(card_report, "_default_predict_fns",
                        lambda nome: _preditor(tabela))
    if sharp_com_prova is not None:
        ref = None
        if sharp_com_prova:
            # Prova COMPLETA: corpo guardado (a chamada) MAIS a linha do preco
            # efetivamente usado, com inicio programado -- sem ele o carater
            # pre-luta nao e verificavel e o gate bloqueia, corretamente.
            chamada = proveniencia.registrar_chamada(
                '[{"id":"ev1","home_team":"Alice Silva"}]',
                fonte="the_odds_api", endpoint="/v4/odds", regions="eu",
                markets="h2h", n_eventos=1)
            ref = proveniencia.registrar_cotacao(
                chamada_ref=chamada, fonte="the_odds_api", casa="Pinnacle",
                mercado="h2h", evento_id="ev1",
                participante_a="Alice Silva", participante_b="Bia Costa",
                lado="Alice Silva", odd=2.00, status="listado",
                inicio_programado=f"{data}T23:00:00+00:00")
            # A odd da CASA tem a sua propria prova: a da Pinnacle nao a
            # cobre. Sem esta linha o gate bloqueia por preco de entrada.
            corpo_tp = proveniencia.registrar_chamada(
                '[{"home":"Alice Silva","odd_home":1.72}]',
                fonte="thunderpick", endpoint="https://thunderpick.io/x")
            ref_tp = proveniencia.registrar_cotacao(
                chamada_ref=corpo_tp, fonte="thunderpick", casa="Thunderpick",
                mercado="h2h", evento_id="tp1",
                participante_a="Alice Silva", participante_b="Bia Costa",
                lado="Alice Silva", odd=1.72, status="aberto",
                inicio_programado=f"{data}T23:00:00+00:00")
            card = pd.read_csv(mundo["card_csv"])
            card["ref_tp"] = ref_tp
            card.to_csv(mundo["card_csv"], index=False)
        from src import line_shopping

        def fake_sharp(fights, **kw):
            return {(f["fighter_a"], f["fighter_b"]):
                    {"sharp_prob": 0.50, "best_odd": 2.00,
                     "casa": "pinnacle", "ref": ref,
                     "evento_id": "ev-1", "evidencia": "exato+exato",
                     "participantes": f'{f["fighter_a"]}|{f["fighter_b"]}',
                     "mercado": "h2h"}
                    for f in fights}
        monkeypatch.setattr(line_shopping, "fetch_sharp_probs", fake_sharp)
    return card_report.generate_card_report(
        mundo["card_csv"], mundo["saida"], model_name="fake",
        card_name="UFC Teste", event_date=data, photos=False,
        sharp=sharp_com_prova is not None, **kw)


class TestCicloCompleto:
    def test_o_ciclo_inteiro_preserva_a_previsao_publicada(self, mundo,
                                                           monkeypatch):
        futuro = _dia(3)

        # ---- 1. pre-registro, antes da luta -------------------------------
        _publica(mundo, monkeypatch, ANTES, futuro)
        h = pd.read_csv(mundo["historico"])
        assert len(h) == 2, h
        assert set(h["estado_resultado"]) == {"aguardando"}
        assert h["actual_winner"].isna().all()
        publicadas = dict(zip(zip(h["fighter_a"], h["fighter_b"]),
                              h["model_prob_a"]))
        assert publicadas[("Alice Silva", "Bia Costa")] == pytest.approx(0.64)
        # a odd da casa foi congelada no registro, nao buscada na hora
        assert h.set_index("fighter_a").loc["Alice Silva", "odds_a_tp"] == 1.72

        # ---- 2. sinal sharp com proveniencia ------------------------------
        corpo = '[{"id":"ev1","home_team":"Alice Silva"}]'
        ref = proveniencia.registrar_chamada(
            corpo, fonte="the_odds_api", endpoint="/v4/odds", regions="eu",
            markets="h2h", n_eventos=1)
        assert proveniencia.cotacao_verificavel(ref)
        ph.record_closing_lines(
            futuro,
            {("Alice Silva", "Bia Costa"): {
                "sharp_prob": 0.62, "best_odd": 1.68, "ref": ref,
                # a prova do casamento viaja com a cotacao de fechamento
                "evento_id": "ev-1", "evidencia": "exato+exato",
                "participantes": "Alice Silva|Bia Costa"}},
            history_csv=mundo["historico"])
        h = pd.read_csv(mundo["historico"])
        alice = h[h["fighter_a"] == "Alice Silva"].iloc[0]
        assert alice["close_ref"] == ref

        # ---- 3. o evento acontece: vencedor no template -------------------
        # dtype explicito: coluna vazia volta do CSV como float64, e gravar
        # um nome nela estoura TypeError no pandas 3
        tpl = pd.read_csv(mundo["template"], dtype={"actual_winner": "object"})
        tpl.loc[tpl["fighter_a"] == "Alice Silva", "actual_winner"] = "Alice Silva"
        # a data passa para o passado: e o relogio que decide, nao o texto
        passado = _dia(-1)
        # o sync casa luta E data -- template sem `event_date` nao fecha nada
        tpl["event_date"] = passado
        tpl.to_csv(mundo["template"], index=False)
        h = pd.read_csv(mundo["historico"])
        h["event_date"] = passado
        h.to_csv(mundo["historico"], index=False)

        # ---- 4. regeracao com o modelo JA RE-TREINADO ---------------------
        # Aqui esta o teste de verdade: o preditor agora devolve 0.99 / 0.01.
        # Se qualquer etapa recalcular, a previsao publicada se perde.
        _publica(mundo, monkeypatch, DEPOIS, passado)
        h = pd.read_csv(mundo["historico"])
        assert len(h) == 2, "regeracao duplicou linha"
        agora = dict(zip(zip(h["fighter_a"], h["fighter_b"]), h["model_prob_a"]))
        assert agora == pytest.approx(publicadas), (
            "a previsao publicada foi reescrita pelo modelo posterior")

        # ---- 5. o resultado fechou e a aferição usou a previsão congelada --
        alice = h[h["fighter_a"] == "Alice Silva"].iloc[0]
        assert alice["actual_winner"] == "Alice Silva"
        # acerto nao e coluna: e derivado de model_side vs actual_winner,
        # justamente para nao haver um campo gravado que possa discordar
        assert alice["model_side"] == alice["actual_winner"]
        dora = h[h["fighter_a"] == "Carla Rocha"].iloc[0]
        assert pd.isna(dora["actual_winner"]), (
            "luta sem vencedor no template nao pode ser fechada por adivinhacao")
        assert mundo["saida"].exists()

        # ---- 6. identidade e estado resolvidos -----------------------------
        assert alice["estado_resultado"] != "aguardando", (
            "luta fechada ficou com estado de luta futura")
        # a prova do casamento viajou com o preco
        assert str(alice["casamento"]) == "exato+exato"
        assert str(alice["evento_id_api"]) == "ev-1"
        assert "Alice Silva" in str(alice["participantes_fonte"])
        assert str(alice["fonte_card"]) == "ambas"

        # ---- 7. liquidacao e metricas --------------------------------------
        hist = ph.load_history(mundo["historico"])
        resumo = ph.compute_series_summary(hist)
        assert resumo is not None, "serie nao liquidou nada"
        assert resumo["model_n"] == 1, resumo
        assert resumo["model_hits"] == 1, resumo
        assert resumo["n_events"] == 1, resumo
        # a perna liquidou pelo preco da CASA, nao pela mediana
        perna = ph.model_side_leg(alice)
        assert perna["casa"] is True
        assert perna["odd"] == pytest.approx(1.72)
        # e o acerto foi medido contra a previsao CONGELADA
        assert perna["prob"] == pytest.approx(publicadas[("Alice Silva",
                                                          "Bia Costa")])

    def test_luta_sem_vencedor_sai_dos_denominadores(self, mundo, monkeypatch):
        """Luta cancelada fica em branco e some sozinha das contas — nunca
        entra como erro do modelo."""
        _publica(mundo, monkeypatch, ANTES, _dia(-1))
        h = ph.load_history(mundo["historico"])
        resumo = ph.compute_series_summary(h)
        assert resumo is None or resumo.get("model_n", 0) == 0

    def test_render_puro_nao_escreve_nada(self, mundo, monkeypatch):
        import hashlib
        _publica(mundo, monkeypatch, ANTES, _dia(3))
        antes = hashlib.sha256(mundo["historico"].read_bytes()).hexdigest()
        _publica(mundo, monkeypatch, DEPOIS, _dia(3), somente_render=True)
        assert hashlib.sha256(mundo["historico"].read_bytes()).hexdigest() == antes


class TestGateNoCiclo:
    def test_entrada_sem_prova_de_cotacao_nao_e_marcada(self, mundo,
                                                        monkeypatch):
        """O card de verdade hoje: sharp registrado, cotação SEM
        proveniência. O sinal de entrada não pode ser proposto."""
        futuro = _dia(3)
        _publica(mundo, monkeypatch, ANTES, futuro, sharp_com_prova=False)
        bloqueios = ph.bloqueios_do_evento(futuro, mundo["historico"])
        assert bloqueios[("Alice Silva", "Bia Costa")] == "cotacao sem prova"

        analise = card_report.analyze_card(CARD, predict_fn=_preditor(ANTES)[0],
                                           method_fn=_preditor(ANTES)[1])
        n = card_report.anotar_sinal_de_entrada(analise, futuro)
        assert n == 0, "entrada proposta sobre cotacao sem prova"
        marcadas = [f for f in analise["favorites"] + analise["underdogs"]
                    if f.get("entrada_bloqueada")]
        assert marcadas, "o bloqueio tem de ficar visivel, nao silencioso"
        assert all("entrada_a" not in f for f in marcadas)

    def test_com_prova_a_entrada_volta_a_ser_avaliada(self, mundo, monkeypatch):
        futuro = _dia(3)
        _publica(mundo, monkeypatch, ANTES, futuro, sharp_com_prova=True)
        assert ph.bloqueios_do_evento(futuro, mundo["historico"]) == {}
        analise = card_report.analyze_card(CARD, predict_fn=_preditor(ANTES)[0],
                                           method_fn=_preditor(ANTES)[1])
        card_report.anotar_sinal_de_entrada(analise, futuro)
        previstas = analise["favorites"] + analise["underdogs"]
        assert not any(f.get("entrada_bloqueada") for f in previstas)
        # sharp 0.50 contra odd da casa 1.72/2.30: o lado B paga acima do justo
        assert any(f.get("entrada_b", 0) > 1 for f in previstas)
