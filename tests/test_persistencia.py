"""
Persistencia dos registros (src/persistencia.py) -- CHAOS Fase A: D1, D2, D9.

O que estes testes fixam, no caminho REAL de gravacao:

  * arquivo truncado e RECUSADO (RegistroIncompleto) e fica byte a byte igual;
    nenhum escritor grava por cima nem anexa sobre linha parcial;
  * o historico so recebe ponteiro para versao gravada e relida;
  * escritores em PROCESSOS separados nao perdem atualizacao, nao colam linhas
    e nao repetem ids;
  * a trava e reentrante, e liberada apos excecao, e o timeout diz o que
    esta acontecendo;
  * o caso normal (registro -> fechamento -> sync) continua dando os mesmos
    valores.
"""
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

import config
from src import persistencia, snapshots
from src import prediction_history as ph
from src.persistencia import (RegistroIncompleto, TravaOcupada, anexar_csv,
                              gravar_atomico, ler_csv_integro, trava,
                              validar_csv)

import tests._worker_persistencia as W

WORKER = Path(W.__file__)
EVENTO, CARD, LUTAS = W.EVENTO, W.CARD, W.LUTAS


def _rodar(papel, pasta, n, tag=None, latencia=0.0, bg=False):
    import os
    env = {**os.environ, "UFC_TESTE_LATENCIA": str(latencia)}
    cmd = [sys.executable, str(WORKER), papel, str(pasta), str(n)]
    if tag:
        cmd.append(tag)
    if bg:
        return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def _concluir(p):
    out = p.communicate(timeout=120)[0]
    assert p.returncode == 0, out
    return out


@pytest.fixture
def mundo(tmp_path):
    hist = tmp_path / "prediction_history.csv"
    return {"pasta": tmp_path, "hist": hist,
            "snap": tmp_path / "previsoes_snapshots.csv"}


def _publicar(mundo, i=0):
    return ph.record_card_predictions(W.analise(i), CARD, EVENTO, mundo["hist"])


# ---------------------------------------------------------------- trava
class TestTrava:
    def test_reentrante(self, tmp_path):
        alvo = tmp_path / "x.csv"
        with trava(alvo):
            with trava(alvo):
                with trava():          # None: raiz de data/, outra trava, tambem ok
                    pass

    def test_um_arquivo_de_trava_ao_lado_do_alvo_fora_de_data(self, tmp_path):
        with trava(tmp_path / "a.csv"):
            pass
        assert (tmp_path / persistencia.NOME_TRAVA).exists()
        assert not (Path(config.DATA_DIR) / "proveniencia" / persistencia.NOME_TRAVA).exists()

    def test_registros_de_data_dividem_a_mesma_trava(self):
        raiz = persistencia._raiz_da_trava
        d = Path(config.DATA_DIR)
        assert (raiz(d / "prediction_history.csv") == raiz(d / "proveniencia" / "x.csv")
                == raiz(d / "raw" / "y.csv") == raiz(None))

    def test_liberada_apos_excecao(self, mundo):
        with pytest.raises(ValueError):
            with trava(mundo["hist"]):
                raise ValueError("boom")
        # nenhum processo/thread segura mais: outro processo entra de primeira
        with trava(mundo["hist"], timeout=0.5):
            pass
        r = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, %r)\n"
             "from src.persistencia import trava\n"
             "with trava(%r, timeout=2): print('entrou')" % (str(WORKER.parents[1]), str(mundo["hist"]))],
            capture_output=True, text=True)
        assert "entrou" in r.stdout, r.stderr

    def test_exclusiva_entre_processos_e_timeout_claro(self, mundo):
        p = _rodar("segura", mundo["pasta"], 0, bg=True)
        try:
            for _ in range(200):
                if (mundo["pasta"] / "travado").exists():
                    break
                time.sleep(0.02)
            assert (mundo["pasta"] / "travado").exists()
            inicio = time.monotonic()
            with pytest.raises(TravaOcupada) as e:
                with trava(mundo["hist"], timeout=0.3):
                    pytest.fail("entrou com a trava ocupada")
            assert time.monotonic() - inicio < 5
            msg = str(e.value)
            assert persistencia.NOME_TRAVA in msg and "outro processo" in msg
            assert "Nada foi gravado" in msg
            # e um escritor real recusa em vez de gravar sem trava
            with pytest.raises(TravaOcupada):
                with trava(mundo["hist"], timeout=0.2):
                    pass
        finally:
            (mundo["pasta"] / "solta").write_text("x")
            p.communicate(timeout=30)
        with trava(mundo["hist"], timeout=5):     # liberada com o fim do processo
            pass

    def test_morte_do_processo_libera_a_trava(self, mundo):
        p = _rodar("segura", mundo["pasta"], 0, bg=True)
        for _ in range(200):
            if (mundo["pasta"] / "travado").exists():
                break
            time.sleep(0.02)
        p.kill()
        p.communicate()
        with trava(mundo["hist"], timeout=5):     # SO soltou: nada para limpar a mao
            pass

    def test_threads_do_mesmo_processo_se_excluem(self, mundo):
        ordem = []

        def outra():
            with trava(mundo["hist"]):
                ordem.append("outra")
        with trava(mundo["hist"]):
            t = threading.Thread(target=outra)
            t.start()
            time.sleep(0.2)
            ordem.append("principal")
        t.join(5)
        assert ordem == ["principal", "outra"]

    @pytest.mark.skipif(sys.platform != "win32", reason="ramo msvcrt: validar no Windows")
    def test_windows_msvcrt_exclui_entre_processos(self, mundo):
        p = _rodar("segura", mundo["pasta"], 0, bg=True)
        try:
            for _ in range(200):
                if (mundo["pasta"] / "travado").exists():
                    break
                time.sleep(0.02)
            with pytest.raises(TravaOcupada):
                with trava(mundo["hist"], timeout=0.3):
                    pass
        finally:
            (mundo["pasta"] / "solta").write_text("x")
            p.communicate(timeout=30)

    def test_ramo_windows_com_duble_do_msvcrt(self, mundo, monkeypatch):
        """Nao substitui a validacao no Windows: confere so a SEQUENCIA de
        chamadas do ramo msvcrt (trava nao bloqueante no byte 0, destrava)."""
        chamadas = []

        class Falso:
            LK_NBLCK, LK_UNLCK = 2, 0

            @staticmethod
            def locking(fd, modo, n):
                chamadas.append((modo, n))
        monkeypatch.setitem(sys.modules, "msvcrt", Falso)
        monkeypatch.setattr(sys, "platform", "win32")
        with trava(mundo["hist"]):
            with trava(mundo["hist"]):
                pass
        assert chamadas == [(2, 1), (0, 1)]       # uma trava, uma destrava


# ---------------------------------------------------------------- atomico
class TestGravarAtomico:
    def test_grava_e_nao_deixa_temporario(self, tmp_path):
        d = tmp_path / "a.csv"
        gravar_atomico(d, "x,y\n1,2\n")
        assert d.read_bytes() == b"x,y\n1,2\n"
        assert [p.name for p in tmp_path.iterdir()] == ["a.csv"]

    def test_falha_antes_do_replace_preserva_destino(self, tmp_path, monkeypatch):
        d = tmp_path / "a.csv"
        d.write_bytes(b"x\n1\n")
        monkeypatch.setattr(persistencia.os, "replace",
                            lambda *a: (_ for _ in ()).throw(OSError("disco")))
        with pytest.raises(OSError):
            gravar_atomico(d, "x\n9\n")
        assert d.read_bytes() == b"x\n1\n"
        assert [p.name for p in tmp_path.iterdir()] == ["a.csv"]

    def test_queda_no_meio_da_escrita_nao_toca_o_destino(self, tmp_path):
        d = tmp_path / "a.csv"
        d.write_bytes(b"x\n1\n")

        def cai(tmp):
            tmp.write_text("x\n9")            # metade
            raise KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            gravar_atomico(d, cai)
        assert d.read_bytes() == b"x\n1\n"
        assert [p.name for p in tmp_path.iterdir()] == ["a.csv"]


# ---------------------------------------------------------------- leitura
class TestLeituraIntegra:
    def test_sem_quebra_final(self, tmp_path):
        f = tmp_path / "a.csv"
        f.write_bytes(b"a,b\n1,2\n3,")
        with pytest.raises(RegistroIncompleto) as e:
            validar_csv(f)
        assert "linha 3" in str(e.value) and "git checkout --" in str(e.value)
        assert "Nada foi gravado" in str(e.value)

    def test_linha_com_campos_a_menos_e_a_mais(self, tmp_path):
        f = tmp_path / "a.csv"
        f.write_bytes(b"a,b\n1,2\n3\n")
        with pytest.raises(RegistroIncompleto, match="linha 3"):
            ler_csv_integro(f)
        f.write_bytes(b"a,b\n1,2,3\n")
        with pytest.raises(RegistroIncompleto, match="linha 2"):
            ler_csv_integro(f)

    def test_aspas_com_virgula_e_quebra_sao_campos_validos(self, tmp_path):
        f = tmp_path / "a.csv"
        f.write_bytes(b'a,b\n"x, y","l1\nl2"\n')
        assert ler_csv_integro(f).iloc[0]["a"] == "x, y"

    def test_aspas_abertas_no_fim(self, tmp_path):
        f = tmp_path / "a.csv"
        f.write_bytes(b'a,b\n1,"abc\n')
        with pytest.raises(RegistroIncompleto):
            validar_csv(f)

    def test_ausente_e_vazio(self, tmp_path):
        assert validar_csv(tmp_path / "nao.csv") == ""


# ---------------------------------------------------------------- anexar
class TestAnexar:
    def test_cria_com_cabecalho_e_anexa_preservando_bytes(self, tmp_path):
        f = tmp_path / "a.csv"
        anexar_csv(f, ("a", "b"), [{"a": 1, "b": "x,y"}])
        antes = f.read_bytes()
        anexar_csv(f, ("a", "b"), [{"a": 2, "b": "z"}])
        depois = f.read_bytes()
        assert depois.startswith(antes)
        assert list(pd.read_csv(f)["b"]) == ["x,y", "z"]

    def test_recusa_linha_parcial_sem_gravar(self, tmp_path):
        f = tmp_path / "a.csv"
        f.write_bytes(b"a,b\n1,2\n3,")
        with pytest.raises(RegistroIncompleto):
            anexar_csv(f, ("a", "b"), [{"a": 9, "b": 9}])
        assert f.read_bytes() == b"a,b\n1,2\n3,"

    def test_preserva_crlf_do_arquivo(self, tmp_path):
        f = tmp_path / "a.csv"
        f.write_bytes(b"a,b\r\n1,2\r\n")
        anexar_csv(f, ("a", "b"), [{"a": 3, "b": 4}])
        assert f.read_bytes() == b"a,b\r\n1,2\r\n3,4\r\n"


# ---------------------------------------------------------------- D1/D9
def _truncar(f: Path, modo: str):
    b = f.read_bytes()
    if modo == "60%":
        f.write_bytes(b[: int(len(b) * 0.6)])
    else:                       # so a ultima linha, no meio dela
        corpo = b.rstrip(b"\n")
        ini = corpo.rfind(b"\n") + 1
        f.write_bytes(b[: ini + (len(corpo) - ini) // 2])


@pytest.mark.parametrize("modo", ["60%", "ultima_linha"])
class TestHistoricoTruncado:
    def _truncado(self, mundo, modo):
        _publicar(mundo)
        _truncar(mundo["hist"], modo)
        return mundo["hist"].read_bytes(), mundo["snap"].read_bytes()

    def test_publicacao_recusa(self, mundo, modo):
        h, s = self._truncado(mundo, modo)
        with pytest.raises(RegistroIncompleto):
            _publicar(mundo, 1)
        assert mundo["hist"].read_bytes() == h
        assert mundo["snap"].read_bytes() == s       # nem versao nova orfa

    def test_fechamento_recusa(self, mundo, modo):
        h, _ = self._truncado(mundo, modo)
        with pytest.raises(RegistroIncompleto):
            ph.record_closing_lines(EVENTO, W.sharp(1), mundo["hist"], allow_update=True)
        assert mundo["hist"].read_bytes() == h

    def test_sync_recusa(self, mundo, modo, tmp_path):
        h, _ = self._truncado(mundo, modo)
        tpl = tmp_path / "odds_template.csv"
        tpl.write_text("x,event_name,event_date,fighter_a,fighter_b,a,b,actual_winner\n"
                       f",{CARD},{EVENTO},Ana Alfa,Bia Beta,1.8,2.1,Ana Alfa\n")
        with pytest.raises(RegistroIncompleto):
            ph.sync_results_from_template(mundo["hist"], tpl)
        assert mundo["hist"].read_bytes() == h


class TestSnapshotsSemQuebraFinal:
    def test_publicacao_falha_antes_de_gravar_o_historico(self, mundo):
        _publicar(mundo)
        mundo["snap"].write_bytes(mundo["snap"].read_bytes().rstrip(b"\n"))
        h, s = mundo["hist"].read_bytes(), mundo["snap"].read_bytes()
        with pytest.raises(RegistroIncompleto):
            _publicar(mundo, 1)
        assert mundo["hist"].read_bytes() == h
        assert mundo["snap"].read_bytes() == s

    def test_leitura_de_versao_recusa(self, mundo):
        _publicar(mundo)
        mundo["snap"].write_bytes(mundo["snap"].read_bytes().rstrip(b"\n"))
        snapshots._CACHE.clear()
        with pytest.raises(RegistroIncompleto):
            snapshots.carregar(mundo["snap"])
        with pytest.raises(RegistroIncompleto):
            snapshots._indice(mundo["snap"])


class TestErroDeGravacaoPropaga:
    def test_registrar_nao_devolve_lista_vazia(self, mundo, monkeypatch):
        monkeypatch.setattr(persistencia.os, "replace",
                            lambda *a: (_ for _ in ()).throw(OSError(28, "sem espaco")))
        with pytest.raises(OSError):
            snapshots.registrar([{"event_date": EVENTO, "fighter_a": "A",
                                  "fighter_b": "B"}], mundo["snap"])

    def test_historico_nao_ganha_ponteiro_sem_versao(self, mundo, monkeypatch):
        _publicar(mundo)
        h = mundo["hist"].read_bytes()
        monkeypatch.setattr(persistencia.os, "replace",
                            lambda *a: (_ for _ in ()).throw(OSError(28, "sem espaco")))
        with pytest.raises(OSError):
            _publicar(mundo, 1)
        assert mundo["hist"].read_bytes() == h

    def test_ponteiro_sem_versao_relida_e_recusado(self, mundo, monkeypatch):
        monkeypatch.setattr(snapshots, "versao", lambda *a, **k: None)
        with pytest.raises(RegistroIncompleto, match="NAO foi alterado"):
            _publicar(mundo)
        assert not mundo["hist"].exists()

    def test_proveniencia_nao_derruba_a_captura_mas_nao_anexa_sobre_parcial(
            self, tmp_path, monkeypatch):
        from src import proveniencia as pv
        monkeypatch.setattr(pv, "PASTA", tmp_path / "prov")
        monkeypatch.setattr(pv, "INDICE", tmp_path / "prov" / "chamadas.csv")
        monkeypatch.setattr(pv, "INDICE_COTACOES", tmp_path / "prov" / "cotacoes.csv")
        pv.registrar_chamada('{"a":1}', fonte="f", endpoint="http://x")
        pv.registrar_cotacao(chamada_ref="c", fonte="f", casa="h", mercado="m",
                             evento_id="e", participante_a="A", participante_b="B",
                             lado="a", odd=1.9)
        for arq in (pv.INDICE, pv.INDICE_COTACOES):
            arq.write_bytes(arq.read_bytes().rstrip(b"\n"))
        antes = (pv.INDICE.read_bytes(), pv.INDICE_COTACOES.read_bytes())
        ref = pv.registrar_chamada('{"a":2}', fonte="f", endpoint="http://x")   # nao levanta
        assert ref
        pv.registrar_cotacao(chamada_ref="c", fonte="f", casa="h", mercado="m",
                             evento_id="e", participante_a="A", participante_b="B",
                             lado="b", odd=2.0)
        assert (pv.INDICE.read_bytes(), pv.INDICE_COTACOES.read_bytes()) == antes

    def test_corpo_truncado_e_regravado(self, tmp_path, monkeypatch):
        import gzip
        from src import proveniencia as pv
        monkeypatch.setattr(pv, "PASTA", tmp_path / "prov")
        monkeypatch.setattr(pv, "INDICE", tmp_path / "prov" / "chamadas.csv")
        corpo = '{"a": [1, 2, 3]}'
        ref = pv.registrar_chamada(corpo, fonte="f", endpoint="http://x")
        gz = pv.PASTA / f"{ref}.json.gz"
        gz.write_bytes(gz.read_bytes()[: len(gz.read_bytes()) // 2])
        pv.registrar_chamada(corpo, fonte="f", endpoint="http://x")
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            assert f.read() == corpo


# ---------------------------------------------------------------- D2
class TestConcorrenciaEntreProcessos:
    def _conferir(self, mundo, rodadas_fecha, rodadas_publica):
        validar_csv(mundo["hist"])
        validar_csv(mundo["snap"])
        snapshots._CACHE.clear()
        hist = pd.read_csv(mundo["hist"])
        snap = pd.read_csv(mundo["snap"], dtype=str)
        assert snap["prediction_id"].is_unique
        ev = hist[hist["event_date"] == EVENTO]
        assert len(ev) == 2
        for _, r in ev.iterrows():
            # ponteiro nao revertido: aponta para a ULTIMA versao da luta
            versoes = snap[(snap["fighter_a"] == r["fighter_a"])
                           & (snap["event_date"] == EVENTO)]
            assert r["prediction_id"] == versoes.iloc[-1]["prediction_id"]
            assert len(versoes) == rodadas_publica + 1          # + a semente
            # fechamento da ultima rodada nao foi apagado
            assert r["close_prob"] == pytest.approx(0.50 + 0.001 * (rodadas_fecha - 1))
            assert r["model_prob_a"] == pytest.approx(0.60 + 0.001 * (rodadas_publica - 1))

    @pytest.mark.parametrize("latencia", [0.0, 0.03])
    def test_fechamento_x_publicacao(self, mundo, latencia):
        _publicar(mundo, 0)
        # semente tem 1 versao por luta; as rodadas somam
        N = 12 if not latencia else 6
        a = _rodar("publica", mundo["pasta"], N, latencia=latencia, bg=True)
        b = _rodar("fecha", mundo["pasta"], N, latencia=latencia, bg=True)
        _concluir(a)
        _concluir(b)
        self._conferir(mundo, N, N)

    def test_varias_rodadas_curtas_sem_latencia(self, mundo):
        for k in range(6):
            for f in (mundo["hist"], mundo["snap"]):
                f.unlink(missing_ok=True)
            _publicar(mundo, 0)
            a = _rodar("publica", mundo["pasta"], 4, bg=True)
            b = _rodar("fecha", mundo["pasta"], 4, bg=True)
            _concluir(a)
            _concluir(b)
            self._conferir(mundo, 4, 4)

    def test_anexacoes_concorrentes_de_snapshots_e_cotacoes(self, mundo):
        ps = [_rodar("snap", mundo["pasta"], 15, tag=t, bg=True) for t in "abc"]
        ps += [_rodar("cot", mundo["pasta"], 15, tag=t, bg=True) for t in "xyz"]
        for p in ps:
            _concluir(p)
        snap = pd.read_csv(validar_and(mundo["snap"]), dtype=str)
        assert len(snap) == 45 and snap["prediction_id"].is_unique
        assert set(snap["fighter_a"]) == {f"{t}{i}" for t in "abc" for i in range(15)}
        cot = pd.read_csv(validar_and(mundo["pasta"] / "prov" / "cotacoes.csv"), dtype=str)
        assert len(cot) == 45
        assert set(cot["participante_a"]) == {f"{t}{i}" for t in "xyz" for i in range(15)}


def validar_and(caminho):
    validar_csv(caminho)
    return caminho


# ---------------------------------------------------------------- caso normal
class TestCasoNormal:
    def test_registro_fechamento_sync_mesmos_valores(self, mundo, tmp_path):
        _publicar(mundo, 0)
        df = pd.read_csv(mundo["hist"])
        assert len(df) == 2
        a = df[df["fighter_a"] == "Ana Alfa"].iloc[0]
        assert a["model_prob_a"] == pytest.approx(0.60)
        assert a["model_side"] == "Ana Alfa" and a["odds_a_decimal"] == pytest.approx(1.8)
        assert snapshots.versao(a["prediction_id"], mundo["snap"])["model_prob_a"] == "0.6"

        n = ph.record_closing_lines(EVENTO, {("Ana Alfa", "Bia Beta"):
                                             {"sharp_prob": 0.57, "best_odd": 1.95}},
                                    mundo["hist"])
        assert n == 1
        df = pd.read_csv(mundo["hist"])
        a = df[df["fighter_a"] == "Ana Alfa"].iloc[0]
        assert a["close_prob"] == pytest.approx(0.57)
        assert a["close_best_odd"] == pytest.approx(1.95)
        assert pd.isna(a["clv"])            # sem sharp no registro: sem CLV
        # fechado uma vez, nao sobrescreve sem allow_update
        assert ph.record_closing_lines(EVENTO, {("Ana Alfa", "Bia Beta"):
                                                {"sharp_prob": 0.9}}, mundo["hist"]) == 0

        # regerar mantem o fechamento e cria NOVA versao (a antiga continua)
        antigo = a["prediction_id"]
        _publicar(mundo, 5)
        df = pd.read_csv(mundo["hist"])
        a = df[df["fighter_a"] == "Ana Alfa"].iloc[0]
        assert a["close_prob"] == pytest.approx(0.57)
        assert a["prediction_id"] != antigo
        assert snapshots.versao(antigo, mundo["snap"]) is not None
        assert a["model_prob_a"] == pytest.approx(0.605)

        tpl = tmp_path / "odds_template.csv"
        tpl.write_text("x,event_name,event_date,fighter_a,fighter_b,a,b,actual_winner\n"
                       f",{CARD},{EVENTO},Bia Beta,Ana Alfa,2.1,1.8,Ana Alfa\n")
        assert ph.sync_results_from_template(mundo["hist"], tpl) == 1
        df = pd.read_csv(mundo["hist"])
        a = df[df["fighter_a"] == "Ana Alfa"].iloc[0]
        assert a["actual_winner"] == "Ana Alfa"
        assert df["actual_winner"].notna().sum() == 1
