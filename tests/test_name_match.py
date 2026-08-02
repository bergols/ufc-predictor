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
      "Islam Makhachev", "Magomed Ankalaev", "Jose Aldo", "Charles Oliveira"]


class TestSurnameGuard:
    def test_primeiro_nome_igual_sobrenome_diferente_e_rejeitado(self):
        # o bug original: nao pode virar Naimov
        assert best_name_match("Muhammad Said", DB) is None
        assert best_name_match("Islam Dulatov", DB) is None       # nao e Makhachev
        assert best_name_match("Magomed Zaynukov", DB) is None    # nao e Ankalaev

    def test_sobrenome_igual_primeiro_nome_diferente_e_rejeitado(self):
        # o 2o bug (real, card de Belgrado): "Michael Oliveira" (outra pessoa)
        # nao pode virar "Charles Oliveira" (ex-campeao) so pelo sobrenome
        assert best_name_match("Michael Oliveira", DB) is None
        assert best_name_match("Ricardo Oliveira", DB) is None

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


class TestTokenizacao:
    """
    Regressoes das guardas de ponta: apostrofo e sufixo geracional nao podem
    virar "primeiro nome"/"sobrenome". Os dois quebraram em producao quando
    as guardas foram adicionadas (31/jul):
      - a API de odds escreve "L'udovit Klein"; com apostrofo como separador
        o primeiro nome virava "l" (1 letra) e a perna sumia do line shopping;
      - a base tem "Levi Rodrigues Jr."/"Kai Kamaka III"; com o sufixo como
        ultimo token, o "sobrenome" virava "jr"/"iii" e o lutador conhecido
        era tratado como estreante.
    """
    @pytest.mark.parametrize("query,db_name", [
        ("Ludovit Klein", "L'udovit Klein"),      # apostrofo so na base
        ("L'udovit Klein", "Ludovit Klein"),      # apostrofo so na consulta
        ("Levi Rodrigues", "Levi Rodrigues Jr."),  # sufixo so na base
        ("Levi Rodrigues Jr.", "Levi Rodrigues"),  # sufixo so na consulta
        ("Kai Kamaka III", "Kai Kamaka"),
        ("Mark O. Madsen", "Mark Madsen"),         # inicial do meio
    ])
    def test_mesma_pessoa_casa_apesar_da_grafia(self, query, db_name):
        assert best_name_match(query, [db_name]) == db_name

    def test_sufixo_nao_confunde_pessoas_diferentes(self):
        # sufixo removido nao pode fazer sobrenomes diferentes casarem
        assert best_name_match("Levi Ferreira Jr.", ["Levi Rodrigues Jr."]) is None


class TestNomesDoMeio:
    """
    As fontes discordam de quantos nomes de batismo o lutador tem: as odds
    trazem "Carlos Diego Ferreira", a base tem "Diego Ferreira" (mesmo
    veterano). Exigir que o PRIMEIRO token batesse transformava-o em
    "estreante" e a previsao saia com perfil vazio — aconteceu no card de
    08/ago/2026, e ainda por cima numa perna EV. Basta um nome de batismo
    em comum.
    """
    @pytest.mark.parametrize("query,db_name", [
        ("Carlos Diego Ferreira", "Diego Ferreira"),   # nome extra na consulta
        ("Diego Ferreira", "Carlos Diego Ferreira"),   # nome extra na base
        ("Billy Goff", "Billy Ray Goff"),              # nome do meio so na base
        ("Mark O. Madsen", "Mark Madsen"),             # inicial do meio
        ("Yadier DelValle", "Yadier del Valle"),       # sobrenome composto
    ])
    def test_nome_do_meio_nao_impede_o_casamento(self, query, db_name):
        assert best_name_match(query, [db_name]) == db_name

    def test_mas_sobrenome_igual_com_batismo_diferente_segue_rejeitado(self):
        # a protecao original nao pode ser perdida ao afrouxar o primeiro nome
        assert best_name_match("Michael Oliveira", ["Charles Oliveira"]) is None
        assert best_name_match("Carlos Silva Ferreira", ["Diego Ferreira"]) is None


class TestOrdemInvertida:
    """
    O UFC lista nomes do leste asiatico nas duas ordens: as odds trazem
    "Cong Wang", a base tem "Wang Cong". Sem o fallback de inversao ela era
    tratada como estreante (aconteceu de verdade no UFC 329).
    """
    @pytest.mark.parametrize("query,db_name", [
        ("Cong Wang", "Wang Cong"),
        ("Wang Cong", "Cong Wang"),
        ("Yadong Song", "Song Yadong"),
    ])
    def test_inversao_casa_mesma_pessoa(self, query, db_name):
        assert best_name_match(query, [db_name]) == db_name

    @pytest.mark.parametrize("query,db_name", [
        ("Cong Wang", "Wang Sai"),        # sobrenome igual, outra pessoa
        ("Cong Wang", "Anying Wang"),
        ("Michael Oliveira", "Charles Oliveira"),
    ])
    def test_inversao_nao_cria_falso_positivo(self, query, db_name):
        assert best_name_match(query, [db_name]) is None
