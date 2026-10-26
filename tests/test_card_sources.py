"""
Uniao das duas fontes do card (src/card_sources.py).

A regra antiga -- "UFCStats e a autoridade" -- errou nas DUAS direcoes:
salvou duas lutas fantasma no evento 12 (a API listava Darren Till vs Yoel
Romero, que nao aconteceu) e custou tres lutas reais no evento 15 (o UFCStats
estava incompleto). Nenhuma fonte serve sozinha, e o erro nascia no passo
manual, onde a discordancia era invisivel.

O que se protege aqui: a ordem vem do UFCStats, a completude vem da API, e
luta que so uma fonte conhece sai MARCADA em vez de sumir.
"""
import pytest

from src.card_sources import LutaDoCard, resumir, unir

DATA = "2026-10-31"


def _api(home, away, quando=f"{DATA}T23:00:00Z"):
    return {"home_team": home, "away_team": away, "commence_time": quando,
            "bookmakers": []}


class TestUniao:
    def test_luta_nas_duas_fontes(self):
        card = unir([("Alice Silva", "Bia Costa")],
                    [_api("Alice Silva", "Bia Costa")], DATA)
        assert [(c.fonte, c.ordem) for c in card] == [("ambas", 0)]
        assert card[0].completa

    def test_so_no_ufcstats_nao_some(self):
        """O caso Bekoev vs Petroski: nenhuma casa precificou. Sem mediana nao
        ha benchmark, mas a luta tem de APARECER na lista."""
        card = unir([("Alice Silva", "Bia Costa")], [], DATA)
        assert card[0].fonte == "so_ufcstats"
        assert card[0].avisos

    def test_so_na_API_entra_marcada_e_no_fim(self):
        """O caso do evento 15: o UFCStats estava incompleto e tres lutas
        reais ficaram de fora. Elas entram -- mas sem posicao no card, porque
        ninguem sabe onde estavam."""
        card = unir([("Alice Silva", "Bia Costa")],
                    [_api("Alice Silva", "Bia Costa"),
                     _api("Carla Rocha", "Dora Lima")], DATA)
        assert [c.fonte for c in card] == ["ambas", "so_api"]
        assert card[1].ordem is None
        assert "UFCStats" in card[1].avisos[0]

    def test_a_ORDEM_do_card_e_a_do_ufcstats(self):
        """A primeira linha vira o destaque do relatorio (card_order). A API
        devolve em ordem propria, que nao e a do card."""
        ufc = [("Main Event", "Co Main"), ("Alice Silva", "Bia Costa")]
        api = [_api("Alice Silva", "Bia Costa"), _api("Main Event", "Co Main")]
        card = unir(ufc, api, DATA)
        assert [(c.fighter_a, c.ordem) for c in card] == [("Main Event", 0),
                                                          ("Alice Silva", 1)]

    def test_usa_a_tabela_de_apelidos(self):
        """'Joe Kropschot' no UFCStats e 'Joseph Kropschot' na API. Sem o
        apelido a luta viraria duas: uma so_ufcstats e uma so_api."""
        card = unir([("Nick Klein", "Joe Kropschot")],
                    [_api("Nick Klein", "Joseph Kropschot")], DATA)
        assert [c.fonte for c in card] == ["ambas"]

    def test_evento_de_outra_data_e_ignorado(self):
        card = unir([("Alice Silva", "Bia Costa")],
                    [_api("Alice Silva", "Bia Costa", "2026-12-12T23:00:00Z")], DATA)
        assert card[0].fonte == "so_ufcstats"

    def test_um_dia_de_folga_para_o_fuso(self):
        """Card de sabado a noite nos EUA comeca no domingo em UTC."""
        card = unir([("Alice Silva", "Bia Costa")],
                    [_api("Alice Silva", "Bia Costa", "2026-11-01T02:00:00Z")], DATA)
        assert card[0].fonte == "ambas"

    def test_uma_entrada_da_API_serve_a_UMA_luta(self):
        """Dois confrontos nao podem herdar a mesma odd por casamento duplo."""
        card = unir([("Alice Silva", "Bia Costa"), ("Alice Silva", "Bia Costa")],
                    [_api("Alice Silva", "Bia Costa")], DATA)
        assert [c.fonte for c in card] == ["ambas", "so_ufcstats"]

    def test_card_vazio_nao_quebra(self):
        assert unir([], [], DATA) == []


class TestResumo:
    def test_conta_por_fonte(self):
        card = [LutaDoCard("A", "B", "ambas"), LutaDoCard("C", "D", "ambas"),
                LutaDoCard("E", "F", "so_api")]
        assert resumir(card) == "3 luta(s): 2 ambas, 1 so_api"
