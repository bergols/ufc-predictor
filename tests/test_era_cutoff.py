"""
Corte de epoca do conjunto de treino (src.train.apply_era_cutoff).

Antes do UFC 28 (nov/2000) o esporte era outro: sem categoria de peso e sem
limite de rounds, um peso-pesado podia enfrentar um meio-medio. Treinar nessas
lutas ensina relacoes que nao existem no esporte que o modelo tem de prever.

O corte e de LINHA DE TREINO, nao de historico: as features point-in-time de
uma luta de 2005 continuam acumulando o que o atleta fez antes de 2001.
"""
import pandas as pd
import pytest

import config
from src.train import apply_era_cutoff


def _features(datas: list[str]) -> pd.DataFrame:
    """Uma luta por data, com as DUAS linhas espelhadas (A-B e B-A)."""
    linhas = []
    for i, d in enumerate(datas):
        for a, b in (("A", "B"), ("B", "A")):
            linhas.append({"fight_id": f"f{i}", "event_date": pd.Timestamp(d),
                           "fighter_a": a, "fighter_b": b, "label": 1})
    return pd.DataFrame(linhas)


class TestCorteDeEpoca:
    def test_descarta_o_que_e_anterior_ao_corte(self):
        df = _features(["1995-06-01", "1999-01-01", "2004-05-01", "2020-03-01"])
        out = apply_era_cutoff(df)
        assert out["event_date"].min() == pd.Timestamp("2004-05-01")
        assert out["fight_id"].nunique() == 2

    def test_a_data_do_corte_fica_dentro(self):
        # o limiar e inclusivo: TRAINING_ERA_START e o primeiro dia da era
        df = _features([config.TRAINING_ERA_START])
        assert len(apply_era_cutoff(df)) == 2

    def test_luta_nunca_e_cortada_pela_metade(self):
        # as duas linhas espelhadas dividem fight_id e data; se uma sobrasse
        # sem a outra, a simetria vermelho/azul do treino quebraria em silencio
        df = _features(["1996-01-01", "2010-01-01", "2019-07-07"])
        out = apply_era_cutoff(df)
        assert (out.groupby("fight_id").size() == 2).all()

    def test_desligado_e_no_op(self, monkeypatch):
        monkeypatch.setattr(config, "TRAINING_ERA_START", "")
        df = _features(["1995-06-01", "2020-03-01"])
        assert len(apply_era_cutoff(df)) == len(df)

    def test_cortar_ANTES_do_split_desloca_as_fronteiras(self):
        """
        Guarda da ordem: em producao o corte vem DEPOIS do split, sobre a
        fatia de treino. Parecia indiferente -- as lutas descartadas sao as
        mais antigas e cairiam no treino de qualquer jeito -- e nao e.

        O split e PROPORCIONAL (70/15/15 das lutas unicas). Tirar lutas da
        frente encolhe o total, as duas fronteiras andam, e o conjunto de
        TESTE muda junto. Com os dados reais foi de 1305 para 1266 lutas, o
        que faria a comparacao antes/depois do corte medir conjuntos
        diferentes e passar por melhora do modelo.

        Este teste FIXA o comportamento errado para explicar por que a ordem
        em train_and_calibrate e a que e. Se alguem mover a chamada para
        antes do split, e aqui que a razao esta escrita.
        """
        from src.train import temporal_group_split

        datas = ([f"1996-01-{d:02d}" for d in range(1, 21)]          # pre-corte
                 + [f"20{a:02d}-06-01" for a in range(5, 85)])       # pos-corte
        df = _features(datas)

        _, _, teste_inteiro = temporal_group_split(df)
        _, _, teste_cortando_antes = temporal_group_split(apply_era_cutoff(df))
        assert set(teste_inteiro["fight_id"]) != set(teste_cortando_antes["fight_id"])

        # ja na ordem de producao -- cortar a fatia de TREINO -- o teste fica
        # intacto, que e o que torna a comparacao honesta
        treino, _, teste_ordem_certa = temporal_group_split(df)
        assert set(teste_ordem_certa["fight_id"]) == set(teste_inteiro["fight_id"])
        assert apply_era_cutoff(treino)["event_date"].min() >= pd.Timestamp(
            config.TRAINING_ERA_START)


class TestConstanteDeConfiguracao:
    def test_a_data_e_a_das_regras_unificadas(self):
        # se alguem mexer nisto, que seja deliberado: a data tem motivo
        # estrutural (UFC 28, nov/2000), nao veio de varredura de resultado
        assert config.TRAINING_ERA_START == "2001-01-01"
