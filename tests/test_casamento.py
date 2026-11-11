"""
Casamento entre confronto e cotação (`src/casamento.py`).

Cada teste aqui corresponde a um modo de falha que o `match_event` antigo
tinha e que saía **silencioso**: a linha ficava com preço, e um preço errado
é pior que preço nenhum, porque entra nas contas com aparência de válido.

O que se mede é sempre o VEREDITO e o MOTIVO, não se a função achou algo.
Bloqueio sem motivo explícito não serve: o painel precisa dizer por quê.
"""
import pytest

from src import casamento as cs


def _ev(id_, data, a, b, *, casas=("Pinnacle",), com_mercado=True):
    """Um evento da The Odds API, só com o que o casamento lê."""
    mercados = []
    if com_mercado:
        mercados = [{"key": "h2h", "outcomes": [{"name": a, "price": 1.8},
                                                {"name": b, "price": 2.1}]}]
    return {"id": id_, "commence_time": f"{data}T23:00:00Z",
            "home_team": a, "away_team": b,
            "bookmakers": [{"title": c, "markets": mercados} for c in casas]}


DATA = "2026-11-07"


class TestCaminhoFeliz:
    def test_nome_literal_nos_dois_lados_aprova(self):
        c = cs.casar("Alice Silva", "Bia Costa",
                     [_ev("e1", DATA, "Alice Silva", "Bia Costa")],
                     event_date=DATA)
        assert c.ok and c.motivo is None
        assert c.evento_id == "e1"
        assert c.evidencia == "exato+exato"

    def test_a_orientacao_dos_lados_volta_declarada(self):
        """O que prova orientação é o nome do OUTCOME que carrega o preço."""
        c = cs.casar("Alice Silva", "Bia Costa",
                     [_ev("e1", DATA, "Bia Costa", "Alice Silva")],
                     event_date=DATA)
        assert c.ok
        assert c.nome_a == "Alice Silva" and c.nome_b == "Bia Costa"

    def test_inversao_A_B_na_fonte_nao_inverte_o_nosso_lado(self):
        direto = cs.casar("Alice Silva", "Bia Costa",
                          [_ev("e1", DATA, "Alice Silva", "Bia Costa")],
                          event_date=DATA)
        invertido = cs.casar("Bia Costa", "Alice Silva",
                             [_ev("e1", DATA, "Alice Silva", "Bia Costa")],
                             event_date=DATA)
        assert direto.ok and invertido.ok
        assert invertido.nome_a == "Bia Costa"
        assert invertido.nome_b == "Alice Silva"

    def test_card_que_atravessa_a_meia_noite_UTC_ainda_casa(self):
        """Sabado a noite nos EUA e domingo em UTC. Janela de um dia."""
        c = cs.casar("Alice Silva", "Bia Costa",
                     [_ev("e1", "2026-11-08", "Alice Silva", "Bia Costa")],
                     event_date=DATA)
        assert c.ok


class TestAliasConfirmado:
    def test_alias_da_tabela_e_evidencia_forte(self):
        c = cs.casar("Alatengheili", "Bia Costa",
                     [_ev("e1", DATA, "Heili Alateng", "Bia Costa")],
                     event_date=DATA)
        assert c.ok, c.motivo
        assert c.evidencia_a == cs.ALIAS
        assert c.nome_a == "Heili Alateng"

    def test_alias_fora_da_tabela_nao_inventa_prova(self):
        c = cs.casar("Alice Silva", "Bia Costa",
                     [_ev("e1", DATA, "Alicia Silveira", "Bia Costa")],
                     event_date=DATA)
        assert not c.ok
        assert c.motivo in (cs.SO_APROXIMADO, cs.AUSENTE)


class TestFuzzySugereSemAprovar:
    def test_grafia_aproximada_bloqueia_mas_identifica(self):
        """O valor do fuzzy e apontar o candidato para conferencia humana --
        nao aprovar a selecao."""
        c = cs.casar("Alice Silva", "Bia Costa",
                     [_ev("e1", DATA, "Alice Sylva", "Bia Costa")],
                     event_date=DATA)
        assert not c.ok
        assert c.motivo == cs.SO_APROXIMADO
        assert c.evento_id == "e1"          # identificado
        assert c.evidencia_a == cs.FUZZY    # e declarado como fraco
        assert c.nome_a == "Alice Sylva"

    def test_um_lado_forte_e_outro_fuzzy_ainda_bloqueia(self):
        c = cs.casar("Alice Silva", "Bia Costa",
                     [_ev("e1", DATA, "Alice Silva", "Bia Kosta")],
                     event_date=DATA)
        assert not c.ok and c.motivo == cs.SO_APROXIMADO
        assert c.evidencia_a == cs.EXATO and c.evidencia_b == cs.FUZZY


class TestRevanche:
    def test_mesmos_nomes_em_duas_datas_pega_a_data_certa(self):
        eventos = [_ev("velho", "2026-03-01", "Alice Silva", "Bia Costa"),
                   _ev("novo", DATA, "Alice Silva", "Bia Costa")]
        c = cs.casar("Alice Silva", "Bia Costa", eventos, event_date=DATA)
        assert c.ok
        assert c.evento_id == "novo", "pegou a cotacao da luta anterior"

    def test_revanche_na_MESMA_janela_e_ambigua_e_bloqueia(self):
        """Dois eventos plausiveis no mesmo fim de semana: nao se escolhe."""
        eventos = [_ev("e1", DATA, "Alice Silva", "Bia Costa"),
                   _ev("e2", "2026-11-08", "Alice Silva", "Bia Costa")]
        c = cs.casar("Alice Silva", "Bia Costa", eventos, event_date=DATA)
        assert not c.ok
        assert c.motivo == cs.MULTIPLOS
        assert c.candidatos == 2

    def test_sem_data_nao_ha_casamento(self):
        """Nao informar a data nao e permissao para adivinhar."""
        c = cs.casar("Alice Silva", "Bia Costa",
                     [_ev("e1", DATA, "Alice Silva", "Bia Costa")],
                     event_date=None)
        assert not c.ok and c.motivo == cs.FORA_DA_JANELA


class TestEventoErrado:
    def test_evento_de_outra_semana_nao_casa(self):
        c = cs.casar("Alice Silva", "Bia Costa",
                     [_ev("e1", "2026-11-14", "Alice Silva", "Bia Costa")],
                     event_date=DATA)
        assert not c.ok and c.motivo == cs.FORA_DA_JANELA

    def test_adversario_substituido_nao_casa_pelo_lado_que_sobrou(self):
        """Metade do confronto batendo nao e o confronto. Era um caminho real
        de erro: o outro lado entrava por fuzzy e o preco vinha de uma luta
        que nao e a nossa."""
        c = cs.casar("Alice Silva", "Bia Costa",
                     [_ev("e1", DATA, "Alice Silva", "Zelda Ferreira")],
                     event_date=DATA)
        assert not c.ok
        assert c.motivo in (cs.AUSENTE, cs.SO_APROXIMADO)

    def test_confronto_ausente_da_fonte(self):
        c = cs.casar("Alice Silva", "Bia Costa",
                     [_ev("e1", DATA, "Carla Rocha", "Dora Lima")],
                     event_date=DATA)
        assert not c.ok and c.motivo == cs.AUSENTE


class TestMercado:
    def test_evento_sem_mercado_h2h_nao_serve_para_cotar(self):
        c = cs.casar("Alice Silva", "Bia Costa",
                     [_ev("e1", DATA, "Alice Silva", "Bia Costa",
                          com_mercado=False)],
                     event_date=DATA)
        assert not c.ok
        assert c.motivo in (cs.SEM_MERCADO, cs.AUSENTE)

    def test_mercado_de_outra_chave_nao_conta(self):
        ev = _ev("e1", DATA, "Alice Silva", "Bia Costa")
        ev["bookmakers"][0]["markets"][0]["key"] = "totals"
        c = cs.casar("Alice Silva", "Bia Costa", [ev], event_date=DATA)
        assert not c.ok


class TestHomonimo:
    def test_homonimo_sem_id_externo_bloqueia(self):
        """`Jean Silva` existe duas vezes na base. Nome exato escolhe uma das
        duas pessoas por sorte."""
        c = cs.casar("Jean Silva", "Bia Costa",
                     [_ev("e1", DATA, "Jean Silva", "Bia Costa")],
                     event_date=DATA)
        assert not c.ok and c.motivo == cs.HOMONIMO
        assert c.evidencia_a == cs.EXATO, "a evidencia de NOME era forte"

    def test_homonimo_com_id_externo_passa(self):
        c = cs.casar("Jean Silva", "Bia Costa",
                     [_ev("e1", DATA, "Jean Silva", "Bia Costa")],
                     event_date=DATA, desambiguado=True)
        assert c.ok

    def test_a_lista_de_homonimos_confere_com_a_base(self):
        """Se a base ganhar um nome repetido novo, este teste falha e a
        constante tem de ser atualizada -- em vez de o bloqueio silenciosamente
        deixar de cobrir alguem."""
        assert cs.homonimos_da_base() == cs.HOMONIMOS


class TestOrganizacao:
    def test_so_a_api_nao_prova_UFC(self):
        assert not cs.organizacao_confirmada("so_api")

    def test_card_do_ufcstats_prova(self):
        assert cs.organizacao_confirmada("ambas")
        assert cs.organizacao_confirmada("so_ufcstats")

    def test_fonte_ausente_nao_ganha_prova_retroativa(self):
        import numpy as np
        for v in (None, "", float("nan"), np.nan, "qualquer"):
            assert not cs.organizacao_confirmada(v)


class TestNomesDoEnsaioReal:
    def test_mckinney_diferenca_de_caixa_e_exata(self):
        c = cs.casar("Jim Miller", "Terrance McKinney",
                     [_ev("e1", DATA, "Jim Miller", "Terrance Mckinney")],
                     event_date=DATA)
        assert c.ok and c.evidencia_b == cs.EXATO

    @pytest.mark.parametrize(("nosso", "fonte"), [
        ("Bia Mesquita", "Beatriz Mesquita"),
        ("Beatriz Mesquita", "Bia Mesquita"),
        ("Charles Radtke", "Charlie Radtke"),
    ])
    def test_alias_confirmado_preserva_o_outro_lado(self, nosso, fonte):
        c = cs.casar(nosso, "Oponente", [_ev("e1", DATA, fonte, "Oponente")],
                     event_date=DATA)
        assert c.ok and c.nome_a == fonte
        errado = cs.casar(nosso, "Outro", [_ev("e1", DATA, fonte, "Oponente")],
                         event_date=DATA)
        assert not errado.ok

    def test_homonimo_em_minusculas_continua_bloqueado(self):
        c = cs.casar("jean silva", "Oponente",
                     [_ev("e1", DATA, "Jean Silva", "Oponente")], event_date=DATA)
        assert not c.ok and c.motivo == cs.HOMONIMO

    def test_duas_variantes_na_mesma_fonte_continuam_ambiguas(self):
        nome, _ = cs._evidencia("Terrance McKinney",
                               ["Terrance McKinney", "Terrance Mckinney"], {})
        assert nome is None
