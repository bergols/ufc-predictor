"""
Testes dos itens 8 e 13 da auditoria, e do bloqueio de seleção.

Três propriedades, nesta ordem de importância:

  1. o manifesto DENUNCIA divergência — um manifesto que sempre aprova não
     prova reprodução nenhuma;
  2. a proveniência detecta corpo alterado — hash que não se confere é
     decoração;
  3. o bloqueio age na SELEÇÃO e não no registro — previsão congelada
     continua congelada, só a entrada proposta cai.
"""
import gzip
import json

import pandas as pd
import pytest

import config
from src import identidade, manifesto, proveniencia


@pytest.fixture
def prov_isolada(tmp_path, monkeypatch):
    """Proveniência num diretório próprio. Sem isso, rodar a suíte grava no
    índice de verdade."""
    pasta = tmp_path / "proveniencia"
    monkeypatch.setattr(proveniencia, "PASTA", pasta)
    monkeypatch.setattr(proveniencia, "INDICE", pasta / "chamadas.csv")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    return pasta


class TestProveniencia:
    def test_ref_e_deterministica_e_o_corpo_e_recuperavel(self, prov_isolada):
        corpo = '[{"id": "abc", "home_team": "Alice"}]'
        ref = proveniencia.registrar_chamada(
            corpo, fonte="the_odds_api", endpoint="/v4/sports/x/odds",
            regions="eu", markets="h2h", n_eventos=1)
        assert ref == proveniencia.registrar_chamada(
            corpo, fonte="the_odds_api", endpoint="/v4/sports/x/odds")
        ok, motivo = proveniencia.conferir(ref)
        assert ok, motivo
        guardado = gzip.open(prov_isolada / f"{ref}.json.gz", "rt",
                             encoding="utf-8").read()
        assert json.loads(guardado) == json.loads(corpo)

    def test_corpo_alterado_reprova(self, prov_isolada):
        ref = proveniencia.registrar_chamada('{"preco": 1.9}',
                                             fonte="x", endpoint="/y")
        assert proveniencia.conferir(ref)[0]
        with gzip.open(prov_isolada / f"{ref}.json.gz", "wt",
                       encoding="utf-8") as f:
            f.write('{"preco": 2.9}')   # alguem "corrigiu" o preco depois
        ok, motivo = proveniencia.conferir(ref)
        assert not ok
        assert "NAO bate" in motivo
        assert not proveniencia.cotacao_verificavel(ref)

    def test_chave_de_api_nao_entra_no_indice(self, prov_isolada):
        proveniencia.registrar_chamada(
            "[]", fonte="the_odds_api",
            endpoint="/v4/odds?apiKey=0123456789abcdef0123456789abcdef&regions=eu")
        texto = proveniencia.INDICE.read_text(encoding="utf-8")
        assert "0123456789abcdef" not in texto

    def test_ref_ausente_ou_vazia_nao_e_verificavel(self, prov_isolada):
        for ref in (None, "", "   ", float("nan"), "naoexiste0000000"):
            assert not proveniencia.cotacao_verificavel(ref)

    def test_falha_de_escrita_nao_derruba_a_captura(self, prov_isolada,
                                                   monkeypatch):
        """Proveniência ausente é pior que captura perdida? Não. A captura
        segue e a cotação fica sem prova — que o bloqueio depois pega."""
        def explode(*a, **k):
            raise OSError("disco cheio")
        monkeypatch.setattr(proveniencia.Path, "mkdir", explode)
        ref = proveniencia.registrar_chamada("[]", fonte="x", endpoint="/y")
        assert ref                       # a captura continua
        assert not proveniencia.cotacao_verificavel(ref)


class TestManifesto:
    def test_denuncia_entrada_diferente(self, tmp_path, monkeypatch):
        alvo = tmp_path / "features.csv"
        alvo.write_text("a,b\n1,2\n", encoding="utf-8")
        monkeypatch.setattr(config, "FEATURES_CSV", alvo)
        m = manifesto.montar(["1"], ["2"], ["3"], salvar=False)
        assert manifesto.conferir(m) == []
        alvo.write_text("a,b\n9,9\n", encoding="utf-8")
        problemas = manifesto.conferir(m)
        assert any("entrada features" in p for p in problemas)

    def test_denuncia_biblioteca_diferente(self, tmp_path, monkeypatch):
        m = manifesto.montar(["1"], ["2"], ["3"], salvar=False)
        m["bibliotecas"]["numpy"] = "0.0.0-inventada"
        assert any("biblioteca numpy" in p for p in manifesto.conferir(m))

    def test_denuncia_vazamento_entre_fatias(self):
        m = manifesto.montar(["1", "2"], ["2", "3"], ["4"], salvar=False)
        assert m["fatias_sobrepostas"]["treino_x_calibracao"] == 1
        assert any("VAZAMENTO" in p for p in manifesto.conferir(m))

    def test_ordem_do_split_nao_muda_o_hash(self):
        a = manifesto.montar(["3", "1", "2"], ["9"], ["8"], salvar=False)
        b = manifesto.montar(["1", "2", "3"], ["9"], ["8"], salvar=False)
        assert a["fatias_sha256"]["treino"] == b["fatias_sha256"]["treino"]

    def test_fatia_diferente_muda_o_hash(self):
        a = manifesto.montar(["1", "2"], ["9"], ["8"], salvar=False)
        b = manifesto.montar(["1", "2", "3"], ["9"], ["8"], salvar=False)
        assert a["fatias_sha256"]["treino"] != b["fatias_sha256"]["treino"]

    def test_salva_a_composicao_das_fatias(self, tmp_path, monkeypatch):
        destino = tmp_path / "slices.json"
        monkeypatch.setattr(manifesto, "FATIAS_PATH", destino)
        monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
        manifesto.montar(["b", "a"], ["c"], ["d"])
        salvo = json.loads(destino.read_text(encoding="utf-8"))
        assert salvo == {"treino": ["a", "b"], "calibracao": ["c"],
                         "teste": ["d"]}


def _linha(**kw):
    base = {"fighter_a": "Alice", "fighter_b": "Bruna",
            "fight_uid": "uid-1", "uid_origem": "ufcstats_exato",
            "estado_resultado": "confirmado", "sharp_ref": "ref-boa"}
    base.update(kw)
    return pd.Series(base)


class TestBloqueioDeSelecao:
    @pytest.fixture(autouse=True)
    def cotacao_sempre_verificavel(self, monkeypatch):
        """Isola o gate da proveniência real: cada teste diz se a cotação
        tem prova, e o que se mede aqui é a decisão, não o hash."""
        monkeypatch.setattr(proveniencia, "cotacao_verificavel",
                            lambda ref: str(ref) == "ref-boa")

    def test_libera_linha_com_identidade_e_cotacao(self):
        assert identidade.elegivel_para_selecao(_linha())

    def test_bloqueia_fuzzy(self):
        assert (identidade.motivo_de_bloqueio(_linha(uid_origem="ufcstats_fuzzy"))
                == identidade.BLOQUEIO_IDENTIDADE)

    def test_bloqueia_sem_fight_uid(self):
        assert (identidade.motivo_de_bloqueio(_linha(fight_uid=None))
                == identidade.BLOQUEIO_IDENTIDADE)

    def test_bloqueia_cotacao_sem_prova(self):
        for ref in (None, "", "ref-inventada"):
            assert (identidade.motivo_de_bloqueio(_linha(sharp_ref=ref))
                    == identidade.BLOQUEIO_COTACAO)

    def test_bloqueia_luta_retirada_do_card(self):
        assert (identidade.motivo_de_bloqueio(
            _linha(estado_resultado=identidade.RETIRADO_DO_CARD))
            == identidade.BLOQUEIO_ESTADO)

    def test_evento_futuro_nao_e_cobrado_por_fight_uid(self):
        """Luta que ainda não aconteceu não tem linha na base de resultados.
        Exigir uid dela bloquearia TODO card futuro — o que esvaziaria o
        gate em vez de aplicá-lo."""
        futura = _linha(fight_uid=None, uid_origem=None,
                        estado_resultado=identidade.AGUARDANDO)
        assert identidade.elegivel_para_selecao(futura)
        sem_prova = _linha(fight_uid=None, uid_origem=None,
                           estado_resultado=identidade.AGUARDANDO,
                           sharp_ref=None)
        assert (identidade.motivo_de_bloqueio(sem_prova)
                == identidade.BLOQUEIO_COTACAO)
