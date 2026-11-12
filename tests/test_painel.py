"""
Pagina em formato "ficha de confronto" (src/painel.py).

Garantias que o desenho novo nao pode perder:
- o card aparece na ORDEM do card, uma ficha por luta, a principal primeiro;
- cada numero exibido e o da analise, sem recalculo;
- dado ausente aparece como indisponivel, nunca como zero ou em branco;
- a casa so e nomeada quando alguma luta tem odd dela;
- redesenhar (somente_render) nao vai a rede nem grava o cache de fotos.
"""
import re

import pandas as pd
import pytest

import config
from src import card_report
from src.card_report import analyze_card, render_html

CARD = pd.DataFrame([
    {"fighter_a": "Alice Silva", "fighter_b": "Bia Costa",
     "odds_a_decimal": 1.60, "odds_b_decimal": 2.40,
     "odds_a_tp": 1.72, "odds_b_tp": 2.30},
    {"fighter_a": "Carla Rocha", "fighter_b": "Dora Lima",
     "odds_a_decimal": 1.90, "odds_b_decimal": 1.95},
    {"fighter_a": "Eva Melo", "fighter_b": "Fabi Neri",
     "odds_a_decimal": 2.80, "odds_b_decimal": 1.45,
     "odds_a_tp": 2.90, "odds_b_tp": 1.44},
])
PROBS = {("Alice Silva", "Bia Costa"): 0.64, ("Carla Rocha", "Dora Lima"): 0.48,
         ("Eva Melo", "Fabi Neri"): 0.61}


def _predict(a, b):
    p = PROBS[(a, b)]
    return {"fighter_a": a, "fighter_b": b, "prob_a_wins": p, "prob_b_wins": 1 - p,
            "model_used": "fake", "fighter_a_low_experience": False,
            "fighter_b_low_experience": False}


@pytest.fixture
def analise():
    return analyze_card(CARD, predict_fn=_predict, method_fn=None)


@pytest.fixture
def pagina(analise):
    return render_html(analise, freshness_gap_days=5, card_name="UFC Teste")


def _fichas(html):
    return re.findall(r'<article class="ficha[^"]*".*?</article>', html, re.S)


def test_uma_ficha_por_luta_na_ordem_do_card(pagina):
    fichas = _fichas(pagina)
    assert len(fichas) == 3
    ordem = [re.findall(r"<h3>(.*?)</h3>", f)[0] for f in fichas]
    assert ordem == ["Alice Silva", "Carla Rocha", "Eva Melo"]
    assert "Luta principal" in fichas[0] and 'class="ficha principal"' in fichas[0]
    assert "Luta 2" in fichas[1] and "Luta 3" in fichas[2]


def test_numeros_sao_os_da_analise(analise, pagina):
    lutas = {f["fighter_a"]: f for f in analise["favorites"] + analise["underdogs"]}
    for f, ficha in zip(sorted(lutas.values(), key=lambda x: x["card_order"]), _fichas(pagina)):
        assert f"{f['model_prob_a'] * 100:.1f}%" in ficha
        assert f"{f['model_prob_b'] * 100:.1f}%" in ficha
        assert f"{f['market_prob_a'] * 100:.1f}%" in ficha


def test_odd_da_casa_mostrada_e_ausente_dita(pagina):
    alice, carla, _ = _fichas(pagina)
    assert "1.72" in alice and "2.30" in alice
    assert "indisp." in carla and ">indisponível<" in carla   # sem odd da casa
    assert "Sem odd da casa" in carla


def test_inicio_ausente_e_indisponivel(pagina):
    assert all("início indisponível" in f for f in _fichas(pagina))


def test_filtros_marcam_divergencia_casa_e_bloqueio(analise, pagina):
    fichas = _fichas(pagina)
    lutas = sorted(analise["favorites"] + analise["underdogs"], key=lambda x: x["card_order"])
    for f, ficha in zip(lutas, fichas):
        marcas = re.search(r'data-f="([^"]*)"', ficha).group(1).split()
        assert ("diverge" in marcas) == (f["model_side"] != f["favorite"])
        assert ("casa" in marcas) == bool(f["odds_da_casa"])
        assert ("bloq" in marcas) == (not f["odds_da_casa"])


def test_favorito_criterio_e_vantagem_sao_ditos_separados(pagina):
    assert "favorito do mercado:" in pagina
    assert "EV&gt;1 pelo critério de registro" in pagina
    assert "Nenhuma vantagem sobre o mercado está demonstrada" in pagina
    assert "lucro garantido" not in pagina


def test_casa_so_e_nomeada_quando_usada():
    so_mediana = CARD.drop(columns=["odds_a_tp", "odds_b_tp"])
    a = analyze_card(so_mediana, predict_fn=_predict, method_fn=None)
    assert "Thunderpick" not in render_html(a, freshness_gap_days=5)


def test_sem_rolagem_lateral_garantida_pelo_css(pagina):
    """Codigo e referencia longos quebram; a tabela de detalhes e fixa."""
    assert "word-break:break-all" in pagina
    assert "table-layout:fixed" in pagina
    assert 'name="viewport" content="width=device-width, initial-scale=1"' in pagina


def test_somente_render_usa_so_o_cache_de_fotos(tmp_path, monkeypatch):
    from src import fighter_photos

    def rede(*a, **k):
        raise AssertionError("somente_render foi buscar foto na rede")

    monkeypatch.setattr(fighter_photos, "get_photo_urls", rede)
    monkeypatch.setattr(fighter_photos, "_fetch_photo_url", rede)
    cache = fighter_photos.PHOTO_CACHE_PATH
    antes = cache.read_bytes() if cache.exists() else None
    csv = tmp_path / "card.csv"
    CARD.to_csv(csv, index=False)
    card_report.generate_card_report(csv, tmp_path / "s.html", card_name="UFC Teste",
                                     event_date="2026-11-14", photos=True,
                                     somente_render=True)
    assert (tmp_path / "s.html").exists()
    assert (cache.read_bytes() if cache.exists() else None) == antes
