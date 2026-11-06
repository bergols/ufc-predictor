"""
Chave de API nunca vai para log.

Achado de 31/out/2026: o `requests` poe a URL INTEIRA na mensagem do
HTTPError, com a chave na query string, e os jobs registravam `exc` direto.
Resultado: a chave em claro em data/auto_capture.log. Os logs sao gitignored
-- ela nunca saiu da maquina --, mas estava em texto puro no disco, e
qualquer colagem de log a exporia.

A chave usada aqui e FICTICIA de proposito: teste que carrega segredo real
vira o proximo vazamento.
"""
import pytest

from src.line_shopping import sanitizar

FALSA = "CHAVE_FICTICIA_PARA_TESTE_0000"
URL = ("401 Client Error: Unauthorized for url: https://api.the-odds-api.com/"
       f"v4/sports/mma_mixed_martial_arts/odds/?apiKey={FALSA}&regions=eu%2Cuk")


class TestSanitizar:
    def test_tira_a_chave_da_url_do_requests(self):
        saida = sanitizar(URL)
        assert FALSA not in saida
        assert "apiKey=***" in saida

    @pytest.mark.parametrize("texto", [
        f"...?api_key={FALSA}&x=1",
        f"...?APIKEY={FALSA}",
        f"...?token={FALSA}",
        f'{{"url": "x?apiKey={FALSA}"}}',
        f"x?apiKey={FALSA}&regions=eu",
    ])
    def test_pega_as_variantes_e_para_no_delimitador(self, texto):
        assert FALSA not in sanitizar(texto)

    def test_preserva_o_resto_da_mensagem(self):
        saida = sanitizar(URL)
        assert "401 Client Error" in saida
        assert "regions=eu%2Cuk" in saida      # so a chave sai, o resto fica

    def test_aceita_excecao_e_nao_so_string(self):
        assert FALSA not in sanitizar(RuntimeError(URL))

    def test_texto_sem_chave_passa_intacto(self):
        assert sanitizar("timeout apos 20s") == "timeout apos 20s"


class TestOsJobsUsamOSanitizador:
    """Teste estrutural: o ponto de log precisa chamar sanitizar(exc). Sem
    isto o conserto se perde no proximo except que alguem escrever."""

    @pytest.mark.parametrize("arquivo", [
        "scripts/auto_capture.py",
        "scripts/ciclo_semanal.py",
        "src/line_shopping.py",
    ])
    def test_nenhum_except_de_rede_loga_exc_cru(self, arquivo):
        import config
        texto = (config.PROJECT_ROOT / arquivo).read_text(encoding="utf-8")
        for linha in texto.splitlines():
            if "logger." in linha and "indispon" in linha.lower():
                assert "exc)" not in linha, (
                    f"{arquivo}: '{linha.strip()}' loga a excecao crua -- "
                    f"use sanitizar(exc)")
