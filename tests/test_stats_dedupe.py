"""
Uma linha de stats por (luta, lutador) — src.data_collection._dedupe_stats.

Descoberto em 06/set/2026, e o estrago era desproporcional ao defeito. Dois
eventos "Noche UFC" (2023 e 2025) vieram do espelho GitHub com as stats
repetidas: 50 linhas em dobro, num arquivo de 17 mil. No CSV era invisivel.

Mas as features juntam stats por lutador VARIAS vezes (perfil de A, perfil de
B, historico expandido de cada um), e duplicata em junção multiplica: 22 lutas
sairam com **128 linhas** no lugar de 2. Isso e 2816 de 20206 linhas -- 14% do
conjunto de treino eram um punhado de lutas repetidas 64 vezes cada.

Quem denunciou foi o teste de simetria do modelo de metodo (`ko_rate_sum`
diferente entre linhas espelhadas), nao um erro. Sem ele o modelo teria sido
treinado assim em silencio.
"""
import pandas as pd
import pytest

import config
from src.data_collection import _dedupe_stats


def _stats(linhas: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame([{"fight_url": u, "fighter": f, "knockdowns": k}
                         for u, f, k in linhas])


class TestDedupe:
    def test_remove_duplicata_de_luta_e_lutador(self):
        df = _stats([("f1", "Alice", 1), ("f1", "Bruna", 0),
                     ("f1", "Alice", 1), ("f1", "Bruna", 0)])
        out = _dedupe_stats(df, "teste")
        assert len(out) == 2
        assert set(out["fighter"]) == {"Alice", "Bruna"}

    def test_nao_mexe_no_que_esta_correto(self):
        df = _stats([("f1", "Alice", 1), ("f1", "Bruna", 0),
                     ("f2", "Alice", 2), ("f2", "Carla", 3)])
        assert len(_dedupe_stats(df, "teste")) == 4

    def test_mesmo_lutador_em_lutas_diferentes_nao_e_duplicata(self):
        # a chave e o PAR: Alice aparece em duas lutas legitimamente
        df = _stats([("f1", "Alice", 1), ("f2", "Alice", 0)])
        assert len(_dedupe_stats(df, "teste")) == 2

    def test_avisa_quando_descarta(self, caplog):
        df = _stats([("f1", "Alice", 1), ("f1", "Alice", 1)])
        with caplog.at_level("WARNING"):
            _dedupe_stats(df, "espelho GitHub")
        assert "duplicada" in caplog.text and "espelho GitHub" in caplog.text

    def test_vazio_e_sem_colunas_nao_quebram(self):
        assert _dedupe_stats(pd.DataFrame(), "teste").empty
        outro = pd.DataFrame({"qualquer": [1, 2]})
        assert len(_dedupe_stats(outro, "teste")) == 2


@pytest.mark.skipif(not config.FEATURES_CSV.exists(), reason="features nao geradas")
class TestFeaturesReais:
    def test_toda_luta_tem_exatamente_duas_linhas_espelhadas(self):
        """
        O sintoma que o defeito produzia, cobrado direto no artefato: uma luta
        com stats em dobro nao vira 4 linhas, vira 128. Se este teste falhar,
        NAO retreine -- rode a coleta de novo para o dedupe agir na origem.
        """
        df = pd.read_csv(config.FEATURES_CSV, usecols=["fight_id"])
        n = df.groupby("fight_id").size()
        fora = n[n != 2]
        assert fora.empty, (
            f"{len(fora)} luta(s) sem as 2 linhas espelhadas "
            f"(contagens vistas: {sorted(set(fora))[:5]}). "
            "Rode: python -m src.data_collection --source github-mirror --fill-gap")
