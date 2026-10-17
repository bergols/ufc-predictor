"""
Odds da Thunderpick (src/thunderpick.py) — a casa onde as entradas sao
feitas. Nada aqui toca a rede: a coleta e o navegador, o que se testa e o
casamento de luta e a escrita no CSV do card.

A propriedade que mais importa e a ORDEM: a casa lista home/away por criterio
dela, e uma odd invertida seria pior que odd nenhuma — entraria no EV como
numero plausivel e so apareceria no P&L, meses depois.
"""
import pandas as pd
import pytest

from src import thunderpick as tp


def _luta(home, away, oh, oa, inicio="2026-10-24T23:00:00+00:00"):
    return tp.LutaTP(home, away, oh, oa, inicio)


class TestCasamento:
    def test_mesma_ordem_do_nosso_csv(self):
        lutas = [_luta("Alice Silva", "Bia Costa", 1.50, 2.60)]
        assert tp.match_fight("Alice Silva", "Bia Costa", lutas) == (1.50, 2.60)

    def test_ordem_INVERTIDA_na_casa_devolve_invertido(self):
        """A casa listou Bia como home; o nosso CSV tem Alice em A. A odd de
        cada lado tem de seguir o LUTADOR, nunca a posicao."""
        lutas = [_luta("Bia Costa", "Alice Silva", 2.60, 1.50)]
        assert tp.match_fight("Alice Silva", "Bia Costa", lutas) == (1.50, 2.60)

    def test_luta_que_a_casa_nao_lista(self):
        lutas = [_luta("Alice Silva", "Bia Costa", 1.50, 2.60)]
        assert tp.match_fight("Carla Rocha", "Dave Lima", lutas) is None

    def test_apelido_da_casa(self):
        """Katlyn Cerminara luta com o sobrenome novo no UFCStats e aparece
        como Chookagian na casa. Sobrenome diferente: nenhum score de string
        casa, e a guarda de pontas do best_name_match rejeita de proposito."""
        lutas = [_luta("Katlyn Chookagian", "Mackenzie Stiller", 1.35, 3.21)]
        assert tp.match_fight("Katlyn Cerminara", "Mackenzie Stiller",
                              lutas) == (1.35, 3.21)

    def test_grafia_proxima_ainda_casa_pela_fuzzy(self):
        lutas = [_luta("Jhonata Diniz", "Tanner Boser", 1.41, 2.83)]
        assert tp.match_fight("Tanner Boser", "Jhonatha Diniz", lutas) == (2.83, 1.41)

    def test_nao_casa_os_dois_lados_no_mesmo_nome(self):
        """Dois lutadores nossos caindo no mesmo nome da casa e casamento
        falso, nao luta encontrada."""
        lutas = [_luta("Alice Silva", "Alice Souza", 1.50, 2.60)]
        assert tp.match_fight("Alice Silva", "Alice Silva", lutas) is None


class TestAnexarAoCard:
    def _csv(self, tmp_path, linhas):
        p = tmp_path / "card.csv"
        pd.DataFrame(linhas).to_csv(p, index=False)
        return p

    def test_escreve_as_duas_colunas(self, tmp_path):
        p = self._csv(tmp_path, [
            {"fighter_a": "Alice Silva", "fighter_b": "Bia Costa",
             "odds_a_decimal": 1.55, "odds_b_decimal": 2.50}])
        n = tp.anexar_ao_card(p, [_luta("Alice Silva", "Bia Costa", 1.50, 2.80)])
        assert n == 1
        d = pd.read_csv(p).iloc[0]
        assert (d["odds_a_tp"], d["odds_b_tp"]) == (1.50, 2.80)
        # a mediana nao e tocada: ela e o benchmark de mercado
        assert (d["odds_a_decimal"], d["odds_b_decimal"]) == (1.55, 2.50)

    def test_luta_sem_odd_na_casa_fica_vazia_e_avisa(self, tmp_path, caplog):
        p = self._csv(tmp_path, [
            {"fighter_a": "Alice Silva", "fighter_b": "Bia Costa",
             "odds_a_decimal": 1.55, "odds_b_decimal": 2.50},
            {"fighter_a": "Carla Rocha", "fighter_b": "Dave Lima",
             "odds_a_decimal": 2.00, "odds_b_decimal": 1.80}])
        with caplog.at_level("WARNING"):
            n = tp.anexar_ao_card(p, [_luta("Alice Silva", "Bia Costa", 1.50, 2.80)])
        assert n == 1
        d = pd.read_csv(p)
        assert pd.isna(d.iloc[1]["odds_a_tp"])
        assert "Carla Rocha" in caplog.text

    def test_rodar_duas_vezes_atualiza_em_vez_de_duplicar(self, tmp_path):
        """Linha se move durante a semana; regerar o card antes do evento tem
        de trazer o preco novo, nao uma terceira coluna."""
        p = self._csv(tmp_path, [
            {"fighter_a": "Alice Silva", "fighter_b": "Bia Costa",
             "odds_a_decimal": 1.55, "odds_b_decimal": 2.50}])
        tp.anexar_ao_card(p, [_luta("Alice Silva", "Bia Costa", 1.50, 2.80)])
        tp.anexar_ao_card(p, [_luta("Alice Silva", "Bia Costa", 1.62, 2.44)])
        d = pd.read_csv(p)
        assert list(d.columns).count("odds_a_tp") == 1
        assert (d.iloc[0]["odds_a_tp"], d.iloc[0]["odds_b_tp"]) == (1.62, 2.44)


class TestDataDoCard:
    """
    A casa lista a competicao INTEIRA -- 27 lutas de cinco eventos, ate
    dezembro. Sem filtrar por data, um par de nomes repetido em outro card
    (revanche, ou dois homonimos) traria a odd do evento errado, e a odd
    errada e pior que odd nenhuma: entra no EV como numero plausivel.
    """
    def test_so_casa_a_luta_do_card_certo(self):
        lutas = [_luta("Alice Silva", "Bia Costa", 9.99, 1.01,
                       inicio="2026-12-12T23:00:00+00:00"),
                 _luta("Alice Silva", "Bia Costa", 1.50, 2.60,
                       inicio="2026-10-24T23:00:00+00:00")]
        assert tp.match_fight("Alice Silva", "Bia Costa", lutas,
                              event_date="2026-10-24") == (1.50, 2.60)

    def test_sem_data_continua_pegando_a_primeira(self):
        lutas = [_luta("Alice Silva", "Bia Costa", 1.50, 2.60)]
        assert tp.match_fight("Alice Silva", "Bia Costa", lutas) == (1.50, 2.60)

    def test_um_dia_de_folga_para_o_fuso(self):
        """Card de sabado a noite nos EUA comeca no domingo em UTC."""
        lutas = [_luta("Alice Silva", "Bia Costa", 1.50, 2.60,
                       inicio="2026-10-25T02:00:00+00:00")]
        assert tp.match_fight("Alice Silva", "Bia Costa", lutas,
                              event_date="2026-10-24") == (1.50, 2.60)

    def test_card_de_outra_semana_nao_casa(self):
        lutas = [_luta("Alice Silva", "Bia Costa", 1.50, 2.60,
                       inicio="2026-11-14T23:00:00+00:00")]
        assert tp.match_fight("Alice Silva", "Bia Costa", lutas,
                              event_date="2026-10-24") is None

    def test_data_ilegivel_nao_descarta_a_luta(self):
        lutas = [_luta("Alice Silva", "Bia Costa", 1.50, 2.60, inicio="")]
        assert tp.match_fight("Alice Silva", "Bia Costa", lutas,
                              event_date="2026-10-24") == (1.50, 2.60)
