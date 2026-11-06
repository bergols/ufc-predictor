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
        uid, est, _ = resolver("Alice", "Bia", DATA, True,
                               _base([("u1", "Alice", "Bia", "Alice")]))
        assert (uid, est) == ("u1", VENCEDOR)

    def test_lados_invertidos_na_base_casam_igual(self):
        uid, est, _ = resolver("Alice", "Bia", DATA, True,
                               _base([("u1", "Bia", "Alice", "Alice")]))
        assert (uid, est) == ("u1", VENCEDOR)

    def test_empate_tem_id_mas_nao_tem_vencedor(self):
        """Kareckaite x Gatto, 10/out: decisao majoritaria. A luta ACONTECEU,
        entao tem id; o que nao existe e o vencedor."""
        uid, est, _ = resolver("Alice", "Bia", DATA, False,
                               _base([("u1", "Alice", "Bia", None)]))
        assert (uid, est) == ("u1", EMPATE_NC)

    def test_evento_na_base_e_par_ausente_vira_nao_aconteceu(self):
        """Melissa Amaya, 26/set: lutou contra OUTRA pessoa. Aqui a ausencia
        do par E evidencia, porque o evento inteiro ja entrou na base."""
        uid, est, _ = resolver("Alice", "Bia", DATA, False,
                               _base([("u1", "Carla", "Dora", "Carla")]))
        assert (uid, est) == (None, NAO_ACONTECEU)

    def test_evento_fora_da_base_e_AGUARDANDO_e_nao_cancelada(self):
        """A distincao que o item 11 existe para fazer: enquanto o evento
        inteiro nao entrou, a ausencia nao prova nada sobre esta luta."""
        uid, est, _ = resolver("Alice", "Bia", DATA, False, _base([]))
        assert (uid, est) == (None, AGUARDANDO)

    def test_par_duplicado_na_base_nao_e_arbitrado(self):
        uid, est, _ = resolver("Alice", "Bia", DATA, False,
                               _base([("u1", "Alice", "Bia", "Alice"),
                                      ("u2", "Alice", "Bia", "Bia")]))
        assert (uid, est) == (None, DESCONHECIDO)

    def test_grafia_proxima_casa(self):
        uid, _, _ = resolver("Tanner Boser", "Jhonatha Diniz", DATA, True,
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


class TestOrigemDoId:
    """O fight_uid NUNCA e derivado de nome ou data -- e sempre a URL da luta
    no UFCStats. O que varia e COMO a linha foi casada com aquela luta, e e
    isso que `uid_origem` guarda."""

    def test_nomes_literais_marcam_exato(self):
        from src.identidade import UID_EXATO
        _, _, origem = resolver("Alice", "Bia", DATA, True,
                                _base([("u1", "Alice", "Bia", "Alice")]))
        assert origem == UID_EXATO

    def test_casamento_por_similaridade_marca_fuzzy(self):
        from src.identidade import UID_FUZZY
        _, _, origem = resolver("Tanner Boser", "Jhonatha Diniz", DATA, True,
                                _base([("u1", "Tanner Boser", "Jhonata Diniz",
                                        "Tanner Boser")]))
        assert origem == UID_FUZZY

    def test_sem_id_nao_ha_origem(self):
        for base, tem in ((_base([]), False),
                          (_base([("u1", "Carla", "Dora", "Carla")]), False)):
            uid, _, origem = resolver("Alice", "Bia", DATA, tem, base)
            assert uid is None and origem is None


@pytest.mark.skipif(not config.PREDICTION_HISTORY_CSV.exists(), reason="historico ausente")
class TestIdsNoHistoricoReal:
    def test_todo_id_tem_origem_declarada(self):
        from src.identidade import UID_EXATO, UID_FUZZY
        h = pd.read_csv(config.PREDICTION_HISTORY_CSV)
        com = h[h["fight_uid"].notna()]
        assert com["uid_origem"].notna().all(), "id sem origem declarada"
        assert set(com["uid_origem"]) <= {UID_EXATO, UID_FUZZY}

    def test_linha_sem_id_nao_declara_origem(self):
        h = pd.read_csv(config.PREDICTION_HISTORY_CSV)
        assert h[h["fight_uid"].isna()]["uid_origem"].isna().all()


class TestValidacaoDoCard:
    """Parada dura antes de qualquer escrita. Derivar do CSV ajuda, mas par
    de nomes sozinho e ambiguo -- revanche repete o par em datas diferentes."""

    def _arquivos(self, tmp_path, pares_csv, pares_hist, data_hist="2026-11-07",
                  nome="UFC X", pares_tpl=None):
        csv = tmp_path / "card.csv"
        pd.DataFrame([{"fighter_a": a, "fighter_b": b, "odds_a_decimal": 1.8,
                       "odds_b_decimal": 2.1} for a, b in pares_csv]).to_csv(csv, index=False)
        hist = tmp_path / "hist.csv"
        pd.DataFrame([{"event_name": nome, "event_date": data_hist, "fighter_a": a,
                       "fighter_b": b, "actual_winner": None}
                      for a, b in pares_hist]).to_csv(hist, index=False)
        tpl = tmp_path / "tpl.csv"
        pd.DataFrame([{"event_name": nome, "event_date": data_hist, "fighter_a": a,
                       "fighter_b": b, "odds_a_decimal": 1.8, "odds_b_decimal": 2.1,
                       "actual_winner": None}
                      for a, b in (pares_tpl if pares_tpl is not None else pares_csv)]
                     ).to_csv(tpl, index=False)
        return csv, hist, tpl

    def test_tudo_batendo_passa(self, tmp_path):
        from src.identidade import validar_identidade_do_card
        pares = [("Alice", "Bia"), ("Carla", "Dora")]
        csv, hist, tpl = self._arquivos(tmp_path, pares, pares)
        out = validar_identidade_do_card(csv, tpl, hist, tmp_path / "nao_existe.json")
        assert out["n_lutas"] == 2 and out["event_date"] == "2026-11-07"

    def test_luta_do_csv_fora_do_TEMPLATE_para_tudo(self, tmp_path):
        """O caso real de 05/nov: 6 lutas no historico e nao no template.
        Sem linha no template elas nunca fechariam."""
        from src.identidade import IdentidadeInconsistente, validar_identidade_do_card
        pares = [("Alice", "Bia"), ("Carla", "Dora")]
        csv, hist, tpl = self._arquivos(tmp_path, pares, pares,
                                        pares_tpl=[("Alice", "Bia")])
        with pytest.raises(IdentidadeInconsistente, match="fora do template"):
            validar_identidade_do_card(csv, tpl, hist, tmp_path / "x.json")

    def test_luta_do_csv_fora_do_HISTORICO_para_tudo(self, tmp_path):
        from src.identidade import IdentidadeInconsistente, validar_identidade_do_card
        csv, hist, tpl = self._arquivos(tmp_path, [("Alice", "Bia"), ("Carla", "Dora")],
                                        [("Alice", "Bia")])
        with pytest.raises(IdentidadeInconsistente, match="fora do historico"):
            validar_identidade_do_card(csv, tpl, hist, tmp_path / "x.json")

    def test_csv_espalhado_por_dois_eventos_para_tudo(self, tmp_path):
        """Exatamente a corrupcao de 01/nov vista pelo outro lado."""
        from src.identidade import IdentidadeInconsistente, validar_identidade_do_card
        csv = tmp_path / "card.csv"
        pd.DataFrame([{"fighter_a": "Alice", "fighter_b": "Bia", "odds_a_decimal": 1.8,
                       "odds_b_decimal": 2.1},
                      {"fighter_a": "Carla", "fighter_b": "Dora", "odds_a_decimal": 1.8,
                       "odds_b_decimal": 2.1}]).to_csv(csv, index=False)
        hist = tmp_path / "hist.csv"
        pd.DataFrame([{"event_name": "UFC A", "event_date": "2026-10-31",
                       "fighter_a": "Alice", "fighter_b": "Bia", "actual_winner": None},
                      {"event_name": "UFC B", "event_date": "2026-11-07",
                       "fighter_a": "Carla", "fighter_b": "Dora", "actual_winner": None}]
                     ).to_csv(hist, index=False)
        tpl = tmp_path / "tpl.csv"
        pd.DataFrame([{"event_name": "UFC B", "event_date": "2026-11-07",
                       "fighter_a": "Carla", "fighter_b": "Dora", "odds_a_decimal": 1.8,
                       "odds_b_decimal": 2.1, "actual_winner": None}]).to_csv(tpl, index=False)
        with pytest.raises(IdentidadeInconsistente, match="2 eventos"):
            validar_identidade_do_card(csv, tpl, hist, tmp_path / "x.json")

    def test_estado_local_discordando_para_tudo(self, tmp_path):
        import json
        from src.identidade import IdentidadeInconsistente, validar_identidade_do_card
        pares = [("Alice", "Bia")]
        csv, hist, tpl = self._arquivos(tmp_path, pares, pares, nome="UFC X")
        est = tmp_path / "estado.json"
        est.write_text(json.dumps({"nome": "UFC OUTRO", "data": "2026-12-12"}),
                       encoding="utf-8")
        with pytest.raises(IdentidadeInconsistente, match="estado local"):
            validar_identidade_do_card(csv, tpl, hist, est)
