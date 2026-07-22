"""
Testes do fuzzy matching de nomes com guarda de sobrenome
(utils.best_name_match).

Bug que motivou a guarda (descoberto no card de 25/jul/2026): "Muhammad
Said" (estreante) casava com "Muhammad Naimov" (~0.79 no difflib) so pelo
primeiro nome, produzindo previsao com as stats do lutador ERRADO. Num
esporte cheio de "Muhammad"/"Magomed"/"Islam", casar so pelo primeiro
nome e um erro perigoso. A guarda exige que o sobrenome tambem bata.
"""
import pytest

from src.utils import best_name_match


DB = ["Muhammad Naimov", "Benoit Saint Denis", "Seokhyeon Ko",
      "Islam Makhachev", "Magomed Ankalaev", "Jose Aldo"]


class TestSurnameGuard:
    def test_primeiro_nome_igual_sobrenome_diferente_e_rejeitado(self):
        # o bug original: nao pode virar Naimov
        assert best_name_match("Muhammad Said", DB) is None
        assert best_name_match("Islam Dulatov", DB) is None       # nao e Makhachev
        assert best_name_match("Magomed Zaynukov", DB) is None    # nao e Ankalaev

    def test_variantes_legitimas_passam_porque_sobrenome_bate(self):
        assert best_name_match("Benoit St. Denis", DB) == "Benoit Saint Denis"
        assert best_name_match("Seok Hyun Ko", DB) == "Seokhyeon Ko"

    def test_typo_no_primeiro_nome_com_sobrenome_certo_passa(self):
        assert best_name_match("Magomad Ankalaev", DB) == "Magomed Ankalaev"

    def test_match_exato(self):
        assert best_name_match("Islam Makhachev", DB) == "Islam Makhachev"

    def test_desconhecido_total_e_none(self):
        assert best_name_match("Fulano Beltrano", DB) is None

    def test_lista_vazia(self):
        assert best_name_match("Qualquer Nome", []) is None
