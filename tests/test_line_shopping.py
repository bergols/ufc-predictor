"""
Testes do line shopping (src/line_shopping.py) — tudo com eventos
falsos da API, nenhum teste toca a rede.

O que protegemos:
- casamento de luta com evento da API (fuzzy, lados em qualquer ordem);
- escolha do melhor preco entre casas para o lado do MODELO;
- probabilidade sharp = devig da Pinnacle (e None sem Pinnacle);
- EV sharp > 1 so quando alguma casa paga acima do justo da sharp;
- luta ausente da API nao quebra a tabela (linha 'sem odds').
"""
import pytest

from src import line_shopping as ls


def _event(home, away, books):
    """books: {casa: (odd_home, odd_away)}"""
    return {
        "home_team": home, "away_team": away,
        "bookmakers": [
            {"key": name.lower(), "title": name,
             "markets": [{"key": "h2h", "outcomes": [
                 {"name": home, "price": odds[0]},
                 {"name": away, "price": odds[1]},
             ]}]}
            for name, odds in books.items()
        ],
    }


@pytest.fixture
def events():
    return [
        _event("Jose Delgado", "Austin Bashi", {
            "Pinnacle": (1.95, 1.95),        # devig 50/50 -> justo 2.00
            "CasaBoa": (2.10, 1.80),         # paga ACIMA do justo p/ Delgado
            "CasaRuim": (1.85, 1.95),
        }),
        _event("Chase Hooper", "Mitch Ramirez", {
            "CasaBoa": (1.30, 3.60),         # sem Pinnacle neste evento
        }),
    ]


def _fight(a, b, side, prob, odd):
    return {"fighter_a": a, "fighter_b": b, "model_side": side,
            "model_side_prob": prob, "model_side_odds": odd}


class TestMatch:
    def test_casa_lados_em_qualquer_ordem(self, events):
        ev, ma, mb = ls.match_event("Austin Bashi", "Jose Delgado", events)
        assert ev is events[0]
        assert (ma, mb) == ("Austin Bashi", "Jose Delgado")

    def test_luta_ausente_retorna_none(self, events):
        ev, _, _ = ls.match_event("Alguem Novo", "Outro Cara", events)
        assert ev is None


class TestPrices:
    def test_melhor_preco_e_da_casa_certa(self, events):
        prices = ls.collect_prices(events[0], "Jose Delgado")
        assert prices == {"Pinnacle": 1.95, "CasaBoa": 2.10, "CasaRuim": 1.85}

    def test_sharp_devig_50_50(self, events):
        prob = ls.sharp_fair_prob(events[0], "Jose Delgado", "Austin Bashi")
        assert prob == pytest.approx(0.5)

    def test_sem_pinnacle_retorna_none(self, events):
        assert ls.sharp_fair_prob(events[1], "Chase Hooper", "Mitch Ramirez") is None


class TestRows:
    def test_linha_completa_com_sinal_sharp(self, events):
        fights = [_fight("Austin Bashi", "Jose Delgado", "Jose Delgado", 0.6366, 2.00)]
        rows = ls.build_rows(fights, events)
        r = rows[0]
        assert r["found"] and r["best_book"] == "CasaBoa"
        assert r["best_odd"] == pytest.approx(2.10)
        assert r["ev_model"] == pytest.approx(0.6366 * 2.10)
        # sharp: justo 2.00, melhor odd 2.10 -> EV sharp 1.05 (> 1, sinal!)
        assert r["ev_sharp"] == pytest.approx(0.5 * 2.10)

    def test_sem_pinnacle_ev_sharp_none(self, events):
        fights = [_fight("Chase Hooper", "Mitch Ramirez", "Chase Hooper", 0.7947, 1.315)]
        r = ls.build_rows(fights, events)[0]
        assert r["found"] and r["ev_sharp"] is None

    def test_luta_fora_da_api_vira_linha_nao_encontrada(self, events):
        fights = [_fight("Alguem Novo", "Outro Cara", "Alguem Novo", 0.55, 1.90)]
        r = ls.build_rows(fights, events)[0]
        assert r["found"] is False
        assert r["ref_odd"] == pytest.approx(1.90)


class TestKey:
    def test_sem_chave_da_erro_claro(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ODDS_API_KEY", raising=False)
        monkeypatch.setattr(ls, "KEY_FILE", tmp_path / "nao_existe.txt")
        with pytest.raises(RuntimeError, match="ODDS_API_KEY"):
            ls.get_api_key()

    def test_env_tem_prioridade(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ODDS_API_KEY", "chave-env")
        keyfile = tmp_path / "key.txt"
        keyfile.write_text("chave-arquivo", encoding="utf-8")
        monkeypatch.setattr(ls, "KEY_FILE", keyfile)
        assert ls.get_api_key() == "chave-env"
