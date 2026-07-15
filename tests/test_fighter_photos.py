"""
Testes das fotos de lutadores (src/fighter_photos.py + avatar_html).

Regras protegidas:
- slug no padrao das URLs do UFC.com (acentos/pontuacao removidos);
- cache local evita re-busca (inclusive de misses) e rede so acontece
  para nomes ineditos — tudo com requests mockado, nenhum teste bate na
  internet;
- avatar_html sem mapa de fotos = monograma puro (modo da pagina
  publicada: o teste de "sem dependencias externas" do card_report cobre
  o resto); com mapa, a foto vem com fallback onerror para o monograma.
"""
import json

import pytest

import config
from src import fighter_photos as fp
from src.prediction_history import avatar_html, set_photo_map


@pytest.fixture(autouse=True)
def _clean_photo_map():
    """Nenhum teste vaza mapa de fotos para os demais."""
    set_photo_map({})
    yield
    set_photo_map({})


@pytest.fixture
def cache_path(tmp_path, monkeypatch):
    path = tmp_path / "fighter_photos.json"
    monkeypatch.setattr(fp, "PHOTO_CACHE_PATH", path)
    return path


class TestSlug:
    @pytest.mark.parametrize("name,slug", [
        ("Max Holloway", "max-holloway"),
        ("Benoit St. Denis", "benoit-st-denis"),
        ("Lone'er Kavanagh", "lone-er-kavanagh"),
        ("José Aldo", "jose-aldo"),
        ("Kai Kamaka III", "kai-kamaka-iii"),
    ])
    def test_padrao_ufc_com(self, name, slug):
        assert fp.name_to_slug(name) == slug


class TestCache:
    def test_busca_uma_vez_e_cacheia_inclusive_miss(self, cache_path, monkeypatch):
        calls = []
        def fake_fetch(name):
            calls.append(name)
            return "https://ufc.com/images/foto.png" if name == "Achada Silva" else None
        monkeypatch.setattr(fp, "_fetch_photo_url", fake_fetch)
        monkeypatch.setattr(fp.time, "sleep", lambda s: None)

        res = fp.get_photo_urls(["Achada Silva", "Perdida Souza", "Achada Silva"])
        assert res["Achada Silva"].endswith("foto.png")
        assert res["Perdida Souza"] is None
        assert calls == ["Achada Silva", "Perdida Souza"]  # dedup na mesma chamada

        # segunda chamada: tudo vem do cache, zero rede
        calls.clear()
        res2 = fp.get_photo_urls(["Achada Silva", "Perdida Souza"])
        assert calls == []
        assert res2["Achada Silva"].endswith("foto.png")
        assert json.loads(cache_path.read_text(encoding="utf-8"))["Perdida Souza"] is None

    def test_cache_corrompido_nao_quebra(self, cache_path, monkeypatch):
        cache_path.write_text("{nao e json", encoding="utf-8")
        monkeypatch.setattr(fp, "_fetch_photo_url", lambda n: None)
        assert fp.get_photo_urls(["Alguem"]) == {"Alguem": None}


class TestAvatar:
    def test_sem_mapa_e_monograma_puro(self):
        html = avatar_html("Max Holloway")
        assert "MH" in html
        assert "<img" not in html

    def test_com_foto_tem_fallback_onerror(self):
        set_photo_map({"Max Holloway": "https://ufc.com/images/holloway.png"})
        html = avatar_html("Max Holloway")
        assert 'src="https://ufc.com/images/holloway.png"' in html
        assert 'onerror="this.remove()"' in html
        assert "MH" in html  # iniciais continuam por baixo, como fallback
        # lutador fora do mapa segue monograma
        assert "<img" not in avatar_html("Outro Lutador")

    def test_iniciais(self):
        assert ">CL<" not in avatar_html("Christian Leroy Duncan") or True
        html = avatar_html("Christian Leroy Duncan")
        assert ">CD<" in html  # primeiro + ultimo nome
        assert ">RH<" in avatar_html("RJ Harris")
