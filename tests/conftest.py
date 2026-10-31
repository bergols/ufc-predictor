"""
Trava: nenhum teste escreve em artefato de produção.

Em 31/out/2026 um teste meu chamou `_grava(config.PREDICTION_HISTORY_CSV, ...)`
sem usar a fixture de isolamento, e sobrescreveu o histórico REAL com duas
linhas sintéticas. Deu para restaurar do git, mas o arquivo é o pré-registro
congelado do projeto inteiro -- o único artefato que não se regenera.

Ler produção é legítimo e vários testes fazem isso de propósito (os de
integridade conferem os dados de verdade). Escrever, nunca. Esta fixture
compara tamanho e mtime antes e depois de CADA teste e falha nomeando o
arquivo e o teste.
"""
import pytest

import config

_PROTEGIDOS = (
    config.PREDICTION_HISTORY_CSV,   # o pre-registro congelado: insubstituivel
    config.ODDS_TEMPLATE_CSV,
    config.UPCOMING_CARD_CSV,
    config.RAW_FIGHTS_CSV,
    config.FEATURES_CSV,
)


def _estado():
    marcas = {}
    for p in _PROTEGIDOS:
        try:
            st = p.stat()
            marcas[p] = (st.st_size, st.st_mtime_ns)
        except OSError:
            marcas[p] = None
    return marcas


@pytest.fixture(autouse=True)
def producao_intocada(request):
    antes = _estado()
    yield
    for caminho, marca in _estado().items():
        if marca != antes[caminho]:
            pytest.fail(
                f"{request.node.nodeid} ESCREVEU em um artefato de producao: "
                f"{caminho}. Use tmp_path + monkeypatch.setattr(config, ...). "
                f"Restaure com: git checkout -- {caminho}")
