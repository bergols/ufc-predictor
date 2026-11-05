"""
Identidade estavel e estado do resultado (src/identidade.py).

Dois itens da auditoria de integridade, que estavam so como proposta:

  6  sem id, duas lutas do mesmo par (revanche) so se distinguem pela data,
     e qualquer juncao que esqueca a data pega a errada -- foi exatamente o
     que acontecia em compare_to_market;
  11 sem estado, cancelada / empate / no contest / coleta incompleta viram a
     mesma coisa depois do fato.

A regra que se protege aqui: DERIVA do que e verificavel, marca o resto como
desconhecido. Nada e arbitrado, e nenhuma previsao e tocada.
"""
import numpy as np
import pandas as pd
import pytest

import config
from src.identidade import (AGUARDANDO, DESCONHECIDO, EMPATE_NC, ESTADOS,
                            NAO_ACONTECEU, VENCEDOR, preencher, resolver)

DATA = "2026-11-07"


def _base(linhas):
    """(fight_id, fighter_1, fighter_2, winner) -> tabela de resultados."""
    b = pd.DataFrame([{"fight_id": i, "event_date": DATA, "fighter_1": a,
                       "fighter_2": v2, "winner": w}
                      for i, a, v2, w in linhas],
                     columns=["fight_id", "event_date", "fighter_1", "fighter_2", "winner"])
    b["_data"] = DATA
    return b


def _hist(linhas):
    return pd.DataFrame([{"event_name": "UFC X", "event_date": d, "fighter_a": a,
                          "fighter_b": b, "actual_winner": w}
                         for d, a, b, w in linhas])


class TestResolver:
    def test_luta_com_vencedor_na_base(self):
        uid, est = resolver("Alice", "Bia", DATA, True,
                            _base([("u1", "Alice", "Bia", "Alice")]))
        assert (uid, est) == ("u1", VENCEDOR)

    def test_lados_invertidos_na_base_casam_igual(self):
        uid, est = resolver("Alice", "Bia", DATA, True,
                            _base([("u1", "Bia", "Alice", "Alice")]))
        assert (uid, est) == ("u1", VENCEDOR)

    def test_empate_tem_id_mas_nao_tem_vencedor(self):
        """Kareckaite x Gatto, 10/out: decisao majoritaria. A luta ACONTECEU,
        entao tem id; o que nao existe e o vencedor."""
        uid, est = resolver("Alice", "Bia", DATA, False,
                            _base([("u1", "Alice", "Bia", None)]))
        assert (uid, est) == ("u1", EMPATE_NC)

    def test_evento_na_base_e_par_ausente_vira_nao_aconteceu(self):
        """Melissa Amaya, 26/set: lutou contra OUTRA pessoa. Aqui a ausencia
        do par E evidencia, porque o evento inteiro ja entrou na base."""
        uid, est = resolver("Alice", "Bia", DATA, False,
                            _base([("u1", "Carla", "Dora", "Carla")]))
        assert (uid, est) == (None, NAO_ACONTECEU)

    def test_evento_fora_da_base_e_AGUARDANDO_e_nao_cancelada(self):
        """A distincao que o item 11 existe para fazer: enquanto o evento
        inteiro nao entrou, a ausencia nao prova nada sobre esta luta."""
        uid, est = resolver("Alice", "Bia", DATA, False, _base([]))
        assert (uid, est) == (None, AGUARDANDO)

    def test_par_duplicado_na_base_nao_e_arbitrado(self):
        uid, est = resolver("Alice", "Bia", DATA, False,
                            _base([("u1", "Alice", "Bia", "Alice"),
                                   ("u2", "Alice", "Bia", "Bia")]))
        assert (uid, est) == (None, DESCONHECIDO)

    def test_grafia_proxima_casa(self):
        uid, _ = resolver("Tanner Boser", "Jhonatha Diniz", DATA, True,
                          _base([("u1", "Tanner Boser", "Jhonata Diniz", "Tanner Boser")]))
        assert uid == "u1"

    def test_todo_estado_devolvido_e_valido(self):
        for base, tem in ((_base([]), False),
                          (_base([("u1", "Alice", "Bia", "Alice")]), True),
                          (_base([("u1", "Carla", "Dora", "Carla")]), False)):
            assert resolver("Alice", "Bia", DATA, tem, base)[1] in ESTADOS


class TestPreencher:
    def test_so_mexe_nas_duas_colunas_novas(self):
        h = _hist([(DATA, "Alice", "Bia", "Alice")])
        out = preencher(h, _base([("u1", "Alice", "Bia", "Alice")]))
        for c in h.columns:
            assert out[c].equals(h[c]), c
        assert out.iloc[0]["fight_uid"] == "u1"
        assert out.iloc[0]["estado_resultado"] == VENCEDOR

    def test_nao_sobrescreve_valor_ja_resolvido(self):
        """Uma vez resolvida, a identidade da linha nao muda."""
        h = _hist([(DATA, "Alice", "Bia", "Alice")])
        h["fight_uid"] = "ja_tinha"
        h["estado_resultado"] = VENCEDOR
        out = preencher(h, _base([("u1", "Alice", "Bia", "Alice")]))
        assert out.iloc[0]["fight_uid"] == "ja_tinha"

    def test_vencedor_preenchido_a_mao_vale_mesmo_sem_a_base(self):
        """Linha fechada a mao antes de o evento entrar na base nao pode
        regredir para 'aguardando'."""
        out = preencher(_hist([(DATA, "Alice", "Bia", "Alice")]), _base([]))
        assert out.iloc[0]["estado_resultado"] == VENCEDOR

    def test_distingue_os_tres_desfechos_sem_vencedor(self):
        h = _hist([(DATA, "Alice", "Bia", None),      # empate
                   (DATA, "Carla", "Dora", None),     # nao aconteceu
                   ("2030-01-01", "Eva", "Fabi", None)])  # evento fora da base
        out = preencher(h, _base([("u1", "Alice", "Bia", None),
                                  ("u2", "Gina", "Hilda", "Gina")]))
        assert list(out["estado_resultado"]) == [EMPATE_NC, NAO_ACONTECEU, AGUARDANDO]


@pytest.mark.skipif(not config.PREDICTION_HISTORY_CSV.exists(), reason="historico ausente")
class TestNoHistoricoReal:
    def test_todas_as_linhas_tem_estado_valido(self):
        h = pd.read_csv(config.PREDICTION_HISTORY_CSV)
        assert "estado_resultado" in h.columns
        assert set(h["estado_resultado"].dropna()) <= set(ESTADOS)

    def test_linha_com_vencedor_tem_estado_vencedor(self):
        h = pd.read_csv(config.PREDICTION_HISTORY_CSV)
        com = h[h["actual_winner"].notna()]
        assert (com["estado_resultado"] == VENCEDOR).all()

    def test_fight_uid_nao_repete_entre_lutas_diferentes(self):
        """O id so serve para desambiguar revanche se for unico por luta."""
        h = pd.read_csv(config.PREDICTION_HISTORY_CSV).dropna(subset=["fight_uid"])
        por_uid = h.groupby("fight_uid")[["fighter_a", "fighter_b"]].nunique()
        assert (por_uid <= 1).all().all()
