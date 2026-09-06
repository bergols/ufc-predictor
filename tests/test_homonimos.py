"""
Homonimos do UFC (src.features._resolve_homonimos).

Ha oito nomes repetidos em fighters.csv -- dois "Bruno Silva", dois "Jean
Silva", e por ai. A chave de juncao das features e o NOME: as tabelas de luta
nao carregam a URL do perfil, entao nao existe identificador para separar os
dois.

Antes ficavamos com a PRIMEIRA ocorrencia, o que e um sorteio. E o sorteio
errava: o Jean Silva do card de set/2026 -- peso-pena, nascido em 1996 --
recebia a bio do homonimo nascido em 1977. Idade 48,9 e alcance vazio, sendo
que `reach_diff_cm` e a feature mais importante do GBM e `age_diff_years` a
terceira. O erro ia direto para a previsao do main event.

Criterio: idade na ESTREIA. Um veterano pode lutar aos 47, mas ninguem COMECA
no UFC nessa idade -- e por isso a estreia discrimina onde a idade atual nao
discriminava (com faixa sobre a idade atual, 48 anos ainda passava por
plausivel e o caso ficava sem solucao).

Sem resolucao possivel, a bio vira NULA. Faltante e tratado (imputacao na
logreg, nativo no LightGBM); errado nao e.
"""
import numpy as np
import pandas as pd

from src.features import _resolve_homonimos


def _bios(registros: list[tuple[str, str | None, float | None]]) -> pd.DataFrame:
    return pd.DataFrame({
        "name": [n for n, _, _ in registros],
        "dob": pd.to_datetime([d for _, d, _ in registros]),
        "reach_cm": [r for _, _, r in registros],
    })


def _lutas(por_lutador: dict[str, list[str]]) -> pd.DataFrame:
    linhas = [{"fighter": f, "event_date": pd.Timestamp(d)}
              for f, datas in por_lutador.items() for d in datas]
    return pd.DataFrame(linhas)


class TestResolucao:
    def test_o_caso_do_jean_silva(self):
        # estreia em 2024: nascido em 1977 teria 46 anos (nao acontece),
        # nascido em 1996 tem 27
        bios = _bios([("Jean Silva", "1977-10-08", None),
                      ("Jean Silva", "1996-12-27", 175.3)])
        out = _resolve_homonimos(bios, _lutas({"Jean Silva": ["2024-06-29", "2026-01-24"]}))
        assert len(out) == 1
        assert out.iloc[0]["dob"].year == 1996
        assert out.iloc[0]["reach_cm"] == 175.3

    def test_dois_plausiveis_viram_bio_nula(self):
        # dois nascidos com um ano de diferenca: nao ha como escolher, e
        # escolher errado poe altura/alcance de outra pessoa na feature
        bios = _bios([("Bruno Silva", "1990-03-16", 180.0),
                      ("Bruno Silva", "1989-07-13", 190.0)])
        out = _resolve_homonimos(bios, _lutas({"Bruno Silva": ["2024-01-01"]}))
        assert len(out) == 1
        assert pd.isna(out.iloc[0]["reach_cm"])
        assert pd.isna(out.iloc[0]["dob"])

    def test_nome_unico_passa_intacto(self):
        bios = _bios([("Alice", "1995-01-01", 170.0), ("Bruna", "1993-05-05", 180.0)])
        out = _resolve_homonimos(bios, _lutas({"Alice": ["2024-01-01"]}))
        assert len(out) == 2
        assert set(out["reach_cm"]) == {170.0, 180.0}

    def test_sem_luta_conhecida_nao_da_para_decidir(self):
        bios = _bios([("Fulano", "1980-01-01", 180.0), ("Fulano", "1995-01-01", 190.0)])
        out = _resolve_homonimos(bios, _lutas({"Outro": ["2024-01-01"]}))
        assert len(out) == 1
        assert pd.isna(out.iloc[0]["reach_cm"])

    def test_dtype_de_data_sobrevive_ao_nulo(self):
        """
        pd.NA numa coluna datetime64 a transforma em object, e a subtracao de
        datas quebra depois -- foi o que aconteceu na primeira versao.
        """
        bios = _bios([("X", "1990-01-01", 180.0), ("X", "1991-01-01", 190.0)])
        out = _resolve_homonimos(bios, _lutas({"X": ["2024-01-01"]}))
        assert pd.api.types.is_datetime64_any_dtype(out["dob"])
        # a conta que quebrava
        assert pd.isna((pd.Timestamp("2026-01-01") - out["dob"]).dt.days.iloc[0])

    def test_nao_perde_ninguem(self):
        bios = _bios([("A", "1990-01-01", 1.0), ("A", "1991-01-01", 2.0),
                      ("B", "1992-01-01", 3.0), ("C", "1993-01-01", 4.0)])
        out = _resolve_homonimos(bios, _lutas({"A": ["2020-01-01"]}))
        assert set(out["name"]) == {"A", "B", "C"}
        assert out["name"].duplicated().sum() == 0
