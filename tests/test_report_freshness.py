"""
Guarda de frescor da PAGINA PUBLICADA (docs/index.html).

`docs/index.html` e um artefato gerado, nao um arquivo editado: commitar
mudanca no gerador NAO atualiza a pagina no ar. Em 28/ago/2026 isso custou uma
rodada inteira de trabalho -- a repaginacao inteira estava no repositorio e no
GitHub Pages continuava a versao de quatro dias antes, sem que nada na pagina
denunciasse. O unico carimbo era "gerado em <data>", que ninguem tem como
comparar de cabeca com a versao do codigo.

O `generator_fingerprint()` fecha essa lacuna: hash das fontes que desenham a
pagina, gravado nela na geracao. Este teste compara os dois.

E ESPERADO ele ficar vermelho no meio de um trabalho de design -- e esse o
ponto. Ele so volta ao verde quando a pagina e regerada, que e exatamente a
acao que faltou naquele dia.
"""
import re

import pytest

import config
from src.card_report import generator_fingerprint

INDEX = config.PROJECT_ROOT / "docs" / "index.html"

REGERAR = (
    'python -m scripts.publish_report data/raw/upcoming_card_odds.csv '
    '--card-name "<nome do evento>" --event-date <YYYY-MM-DD> --no-sharp'
)


@pytest.mark.skipif(not INDEX.exists(), reason="nada publicado ainda")
class TestPaginaPublicada:
    def _carimbo(self) -> str | None:
        m = re.search(r'name="fm-generator" content="([0-9a-f]+)"',
                      INDEX.read_text(encoding="utf-8"))
        return m.group(1) if m else None

    def test_pagina_traz_o_carimbo_do_gerador(self):
        assert self._carimbo() is not None, (
            "docs/index.html nao tem <meta name='fm-generator'> — foi gerado por uma "
            f"versao anterior ao carimbo. Regere:\n    {REGERAR}")

    def test_pagina_foi_gerada_pela_versao_atual_do_gerador(self):
        atual = generator_fingerprint()
        assert self._carimbo() == atual, (
            "A pagina publicada NAO foi gerada por este codigo: o gerador mudou e "
            "docs/index.html ficou para tras. Quem olhar o GitHub Pages vera o "
            "desenho antigo.\n"
            f"  carimbo na pagina: {self._carimbo()}\n"
            f"  gerador atual:     {atual}\n"
            f"Regere (--no-sharp preserva a regua do CLV de um card ja "
            f"pre-registrado):\n    {REGERAR}")


class TestImpressaoDigital:
    def test_e_estavel_entre_chamadas(self):
        assert generator_fingerprint() == generator_fingerprint()

    def test_muda_quando_uma_fonte_muda(self, tmp_path, monkeypatch):
        # sem isto o carimbo poderia ser constante e o teste de frescor nunca
        # pegaria nada -- passaria por estar sempre igual a si mesmo
        from src import card_report

        raiz_falsa = tmp_path / "proj"
        (raiz_falsa / "src").mkdir(parents=True)
        for rel in card_report._GENERATOR_SOURCES:
            (raiz_falsa / rel).write_text("original", encoding="utf-8")
        monkeypatch.setattr(config, "PROJECT_ROOT", raiz_falsa)

        card_report.generator_fingerprint.cache_clear()
        antes = card_report.generator_fingerprint()
        (raiz_falsa / card_report._GENERATOR_SOURCES[0]).write_text("mudou", encoding="utf-8")
        card_report.generator_fingerprint.cache_clear()
        assert card_report.generator_fingerprint() != antes
        card_report.generator_fingerprint.cache_clear()
