"""
Experimento de vantagem (`src/vantagem.py`) e modo sombra (`src/sombra.py`).

O que estes testes protegem, na ordem em que o protocolo exige:

  temporalidade   o peso sai só do desenvolvimento; a experiência prévia é
                  point-in-time; nenhuma fatia parte um evento no meio;
  orientação      inverter A/B inverte as três probabilidades e **não** muda a
                  decisão econômica do mesmo atleta;
  isolamento      o experimento e a sombra escrevem só em
                  `data/experimentos/` e abortam se um registro de produção
                  mudar;
  preservação     a sombra só anexa, e a liquidação não reescreve o que foi
                  previsto.

O procedimento tem de poder concluir que o modelo **não** acrescenta
informação: `w = 0` é um resultado legítimo, e há teste para os dois lados.
"""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import config
from src import sombra, vantagem


# ---------------------------------------------------------------------------
# orientacao e simetria
# ---------------------------------------------------------------------------

class TestOrientacao:
    def test_devig_soma_um(self):
        pa = vantagem.devig(1.60, 2.40)
        pb = vantagem.devig(2.40, 1.60)
        assert pa + pb == pytest.approx(1.0)
        assert pa > pb, "o lado de odd menor tem de ser o mais provavel"

    def test_devig_tira_a_margem(self):
        """1/1.60 + 1/2.40 = 1,0417: ha 4,17% de overround para retirar."""
        bruta = 1 / 1.60
        assert vantagem.devig(1.60, 2.40) < bruta

    @pytest.mark.parametrize("w", [0.0, 0.17, 0.5, 0.83, 1.0])
    def test_inverter_A_e_B_inverte_a_combinacao(self, w):
        pm, pmod = 0.62, 0.41
        direto = vantagem.combinar([pm], [pmod], w)[0]
        invertido = vantagem.combinar([1 - pm], [1 - pmod], w)[0]
        assert invertido == pytest.approx(1 - direto, abs=1e-12)

    def test_a_decisao_economica_do_mesmo_atleta_nao_muda(self):
        """Inverter a ordem das colunas nao pode mudar se a perna entra: a
        probabilidade do ATLETA e a odd do ATLETA andam juntas."""
        w, odd_alpha = 0.3, 2.10
        p_alpha = vantagem.combinar([0.62], [0.41], w)[0]
        p_alpha_invertido = 1 - vantagem.combinar([0.38], [0.59], w)[0]
        assert p_alpha_invertido == pytest.approx(p_alpha, abs=1e-12)
        assert (p_alpha * odd_alpha) == pytest.approx(p_alpha_invertido * odd_alpha)

    def test_w_zero_e_w_um_sao_os_extremos_exatos(self):
        pm, pmod = 0.62, 0.41
        assert vantagem.combinar([pm], [pmod], 0.0)[0] == pytest.approx(pm, abs=1e-9)
        assert vantagem.combinar([pm], [pmod], 1.0)[0] == pytest.approx(pmod, abs=1e-9)


# ---------------------------------------------------------------------------
# o peso, e a possibilidade de concluir que o modelo nao acrescenta nada
# ---------------------------------------------------------------------------

class TestPeso:
    def test_mercado_perfeito_da_peso_zero(self):
        """Se o mercado acerta e o modelo e ruido, o procedimento TEM de
        concluir w = 0. Um procedimento que nao pode chegar a essa conclusao
        nao esta medindo nada."""
        rng = np.random.default_rng(7)
        y = rng.integers(0, 2, 400)
        p_merc = np.where(y == 1, 0.95, 0.05)
        p_mod = rng.uniform(0.3, 0.7, 400)
        w, _ = vantagem.peso_otimo(p_merc, p_mod, y)
        assert w == 0.0

    def test_modelo_informativo_da_peso_positivo(self):
        """O outro lado: com informacao de verdade no modelo, o peso sobe. Sem
        isto, w=0 poderia ser artefato da grade."""
        rng = np.random.default_rng(11)
        y = rng.integers(0, 2, 600)
        p_merc = np.clip(np.where(y == 1, 0.6, 0.4)
                         + rng.normal(0, 0.08, 600), 0.05, 0.95)
        p_mod = np.clip(np.where(y == 1, 0.75, 0.25)
                        + rng.normal(0, 0.08, 600), 0.05, 0.95)
        w, _ = vantagem.peso_otimo(p_merc, p_mod, y)
        assert w > 0.0

    def test_a_grade_inclui_zero_e_um(self):
        assert 0.0 in vantagem.GRADE_W and 1.0 in vantagem.GRADE_W


# ---------------------------------------------------------------------------
# temporalidade
# ---------------------------------------------------------------------------

def _historico(path, eventos=4, por_evento=3):
    """Eventos semanais de verdade: a cadencia real da serie e sabado a
    sabado, e somar dias no numero do mes produzia 2026-07-32."""
    from datetime import date, timedelta
    linhas = []
    primeiro = date(2026, 7, 11)
    for e in range(eventos):
        data = (primeiro + timedelta(days=7 * e)).isoformat()
        for i in range(por_evento):
            a, b = f"A{e}{i}", f"B{e}{i}"
            linhas.append({
                "event_name": f"UFC {e}", "event_date": data,
                "fighter_a": a, "fighter_b": b,
                "odds_a_decimal": 1.8, "odds_b_decimal": 2.1,
                "model_name": "logreg", "model_prob_a": 0.55,
                "model_side": a, "actual_winner": a if i % 2 == 0 else b,
                "estado_resultado": "vencedor",
                "fight_uid": f"uid-{e}-{i}",
            })
    pd.DataFrame(linhas).to_csv(path, index=False)


class TestTemporalidade:
    def test_nenhum_evento_e_partido_entre_fatias(self, tmp_path, monkeypatch):
        h = tmp_path / "h.csv"
        _historico(h, eventos=6, por_evento=4)
        pop, _ = vantagem.carregar_populacao(h)
        pop = vantagem.fatias_por_evento(pop)
        por_evento = pop.groupby("event_date")["fatia"].nunique()
        assert (por_evento == 1).all(), "um evento ficou nas duas fatias"

    def test_o_desenvolvimento_vem_ANTES_da_avaliacao(self, tmp_path):
        h = tmp_path / "h.csv"
        _historico(h, eventos=6)
        pop = vantagem.fatias_por_evento(vantagem.carregar_populacao(h)[0])
        dev = pop[pop["fatia"] == "desenvolvimento"]["event_date"].max()
        aval = pop[pop["fatia"] == "avaliacao"]["event_date"].min()
        assert dev < aval

    def test_experiencia_previa_e_point_in_time(self, tmp_path, monkeypatch):
        """Contar a luta que ainda nao aconteceu seria vazamento do futuro."""
        base = tmp_path / "fights.csv"
        pd.DataFrame([
            {"event_date": "2026-01-10", "fighter_1": "Alpha", "fighter_2": "Zeta"},
            {"event_date": "2026-02-10", "fighter_1": "Alpha", "fighter_2": "Teta"},
            # DEPOIS do evento avaliado: nao pode contar
            {"event_date": "2026-12-10", "fighter_1": "Alpha", "fighter_2": "Iota"},
        ]).to_csv(base, index=False)
        pop = pd.DataFrame([{"event_date": "2026-07-11", "fighter_a": "Alpha",
                             "fighter_b": "Novato"}])
        out = vantagem.experiencia_previa(pop, base)
        assert int(out.iloc[0]["lutas_previas_min"]) == 0, "o grupo e do MENOS experiente"
        pop2 = pd.DataFrame([{"event_date": "2026-07-11", "fighter_a": "Alpha",
                              "fighter_b": "Zeta"}])
        out2 = vantagem.experiencia_previa(pop2, base)
        # Alpha tem 2 lutas antes de julho (nao 3: a de dezembro nao conta);
        # Zeta tem 1. O minimo e 1.
        assert int(out2.iloc[0]["lutas_previas_min"]) == 1
        assert out2.iloc[0]["grupo_experiencia"] == "1-2"

    def test_sem_base_bruta_o_grupo_fica_desconhecido(self, tmp_path):
        """Clone limpo nao tem a base. Grupo inventado seria pior que lacuna."""
        pop = pd.DataFrame([{"event_date": "2026-07-11", "fighter_a": "A",
                             "fighter_b": "B"}])
        out = vantagem.experiencia_previa(pop, tmp_path / "nao_existe.csv")
        assert out.iloc[0]["grupo_experiencia"] == "desconhecido"
        assert pd.isna(out.iloc[0]["lutas_previas_min"])

    def test_toda_linha_declara_a_limitacao_temporal(self, tmp_path):
        h = tmp_path / "h.csv"
        _historico(h)
        pop, _ = vantagem.carregar_populacao(h)
        assert (pop["temporalidade"] == vantagem.TEMPORALIDADE).all()
        assert "nao do registro" in vantagem.TEMPORALIDADE


# ---------------------------------------------------------------------------
# populacao e exclusoes
# ---------------------------------------------------------------------------

class TestPopulacao:
    def _uma(self, path, **kw):
        base = {"event_name": "UFC X", "event_date": "2026-07-11",
                "fighter_a": "Alpha", "fighter_b": "Beta",
                "odds_a_decimal": 1.8, "odds_b_decimal": 2.1,
                "model_name": "logreg", "model_prob_a": 0.55,
                "model_side": "Alpha", "actual_winner": "Alpha",
                "estado_resultado": "vencedor"}
        base.update(kw)
        pd.DataFrame([base]).to_csv(path, index=False)

    @pytest.mark.parametrize("kw,pedaco", [
        ({"actual_winner": None}, "sem resultado"),
        ({"model_prob_a": None}, "sem previsao"),
        ({"odds_b_decimal": None}, "os DOIS lados"),
        ({"estado_resultado": "retirado_do_card"}, "cancelada"),
        ({"actual_winner": "Gama"}, "nao e nenhum dos dois"),
    ])
    def test_cada_exclusao_tem_motivo_registrado(self, tmp_path, kw, pedaco):
        h = tmp_path / "h.csv"
        self._uma(h, **kw)
        pop, excl = vantagem.carregar_populacao(h)
        assert pop.empty
        assert len(excl) == 1
        assert pedaco in excl.iloc[0]["motivo"]

    def test_nada_e_imputado(self, tmp_path):
        """Ausencia de dado e exclusao registrada, nunca suposicao."""
        h = tmp_path / "h.csv"
        self._uma(h, model_prob_a=None)
        pop, excl = vantagem.carregar_populacao(h)
        assert len(pop) == 0 and len(excl) == 1

    def test_y_segue_a_orientacao_da_linha(self, tmp_path):
        h = tmp_path / "h.csv"
        self._uma(h, actual_winner="Beta")
        pop, _ = vantagem.carregar_populacao(h)
        assert int(pop.iloc[0]["y"]) == 0, "y=1 significa que o lado A ganhou"


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------

class TestBootstrap:
    def _df(self):
        rng = np.random.default_rng(3)
        linhas = []
        for e in range(8):
            for _ in range(10):
                y = int(rng.integers(0, 2))
                linhas.append({"event_date": f"2026-0{e + 1}-01", "y": y,
                               "p1": 0.6 if y else 0.4, "p2": 0.55 if y else 0.45})
        return pd.DataFrame(linhas)

    def test_reamostra_EVENTOS_e_nao_lutas(self):
        df = self._df()
        r = vantagem.bootstrap_por_evento(df, "p1", "p2", n=200)
        assert r["n_eventos"] == 8 and r["n_lutas"] == 80

    def test_mesma_semente_da_o_mesmo_intervalo(self):
        df = self._df()
        a = vantagem.bootstrap_por_evento(df, "p1", "p2", n=300, semente=42)
        b = vantagem.bootstrap_por_evento(df, "p1", "p2", n=300, semente=42)
        assert a == b

    def test_candidato_identico_da_diferenca_zero(self):
        """Pareado: com os mesmos numeros nos dois lados a diferenca e zero em
        TODA reamostragem, porque os eventos sorteados sao os mesmos."""
        df = self._df()
        r = vantagem.bootstrap_por_evento(df, "p1", "p1", n=200)
        assert r["diferenca"] == 0.0
        assert r["ic95_baixo"] == 0.0 and r["ic95_alto"] == 0.0


# ---------------------------------------------------------------------------
# isolamento: o experimento nao toca producao
# ---------------------------------------------------------------------------

class TestIsolamento:
    def test_o_experimento_nao_altera_registro_de_producao(self, tmp_path,
                                                           monkeypatch):
        from scripts import experimento_vantagem as exp

        def marcas():
            out = {}
            for p in exp.PROTEGIDOS:
                try:
                    out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
                except OSError:
                    out[str(p)] = None
            return out

        h = tmp_path / "h.csv"
        _historico(h, eventos=6, por_evento=5)
        antes = marcas()
        rc = exp.main(["--saida", str(tmp_path / "out"), "--history", str(h),
                       "--bootstrap", "50"])
        assert rc == 0
        assert marcas() == antes

    def test_os_artefatos_vao_para_a_pasta_pedida(self, tmp_path):
        from scripts import experimento_vantagem as exp
        h = tmp_path / "h.csv"
        _historico(h, eventos=6, por_evento=5)
        saida = tmp_path / "out"
        assert exp.main(["--saida", str(saida), "--history", str(h),
                         "--bootstrap", "50"]) == 0
        nomes = {p.name for p in saida.iterdir()}
        assert f"resumo_{vantagem.VERSAO}.json" in nomes
        assert f"manifesto_{vantagem.VERSAO}.json" in nomes
        assert f"exclusoes_{vantagem.VERSAO}.csv" in nomes

    def test_o_manifesto_registra_o_que_permite_reproduzir(self, tmp_path):
        from scripts import experimento_vantagem as exp
        h = tmp_path / "h.csv"
        _historico(h, eventos=6, por_evento=5)
        saida = tmp_path / "out"
        exp.main(["--saida", str(saida), "--history", str(h), "--bootstrap", "50"])
        m = json.loads((saida / f"manifesto_{vantagem.VERSAO}.json")
                       .read_text(encoding="utf-8"))
        for chave in ("commit", "semente", "bibliotecas", "hashes_das_entradas",
                      "fatias", "eventos_por_fatia", "limitacao_temporal"):
            assert chave in m, chave
        assert m["fatias"]["desenvolvimento"] and m["fatias"]["avaliacao"]


# ---------------------------------------------------------------------------
# modo sombra
# ---------------------------------------------------------------------------

@pytest.fixture
def mundo_sombra(tmp_path, monkeypatch):
    from src import proveniencia, snapshots
    pasta = tmp_path / "prov"
    monkeypatch.setattr(proveniencia, "PASTA", pasta)
    monkeypatch.setattr(proveniencia, "INDICE", pasta / "chamadas.csv")
    monkeypatch.setattr(proveniencia, "INDICE_COTACOES", pasta / "cotacoes.csv")
    monkeypatch.setattr(config, "PREDICTION_HISTORY_CSV", tmp_path / "h.csv")
    monkeypatch.setattr(config, "UPCOMING_CARD_CSV", tmp_path / "card.csv")
    monkeypatch.setattr(snapshots, "ARQUIVO", tmp_path / "snaps.csv")
    monkeypatch.setattr(sombra, "REGISTROS", tmp_path / "sombra.csv")
    monkeypatch.setattr(sombra, "RESULTADOS", tmp_path / "sombra_res.csv")
    snapshots._CACHE.clear()
    return {"hist": tmp_path / "h.csv", "card": tmp_path / "card.csv",
            "snap": tmp_path / "snaps.csv", "reg": tmp_path / "sombra.csv",
            "res": tmp_path / "sombra_res.csv"}


def _mundo_completo(m, *, odd_exec=3.75, prob=0.55, vencedor=None):
    """Card + versao + prova de preco, como a producao monta."""
    import json as _json
    from datetime import datetime, timedelta, timezone

    from src import proveniencia, snapshots
    inicio = (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat(
        timespec="seconds")
    corpo = _json.dumps([{"home": "Alpha", "away": "Beta",
                          "odd_home": odd_exec, "odd_away": 1.30,
                          "inicio": inicio, "status": 1}],
                        ensure_ascii=False, sort_keys=True)
    chamada = proveniencia.registrar_chamada(
        corpo, fonte="thunderpick", endpoint="https://thunderpick.io/x",
        markets="h2h")
    comum = dict(chamada_ref=chamada, fonte="thunderpick", casa="Thunderpick",
                 mercado="h2h", evento_id=inicio, participante_a="Alpha",
                 participante_b="Beta", status="aberto",
                 inicio_programado=inicio)
    ref_a = proveniencia.registrar_cotacao(lado="Alpha", odd=odd_exec, **comum)
    ref_b = proveniencia.registrar_cotacao(lado="Beta", odd=1.30, **comum)
    pid = snapshots.registrar([{
        "event_date": "2026-11-07", "event_name": "UFC Sombra",
        "fighter_a": "Alpha", "fighter_b": "Beta", "model_name": "logreg",
        "model_prob_a": prob, "model_side": "Alpha",
        "manifesto_sha256": "man-1", "modelos_sha256": "mod-1",
    }], m["snap"])[0]
    pd.DataFrame([{
        "event_name": "UFC Sombra", "event_date": "2026-11-07",
        "fighter_a": "Alpha", "fighter_b": "Beta",
        "odds_a_decimal": 3.60, "odds_b_decimal": 1.32,
        "model_name": "logreg", "model_prob_a": prob, "model_side": "Alpha",
        "actual_winner": vencedor, "estado_resultado": "aguardando",
        "prediction_id": pid,
    }]).to_csv(m["hist"], index=False)
    pd.DataFrame([{
        "fighter_a": "Alpha", "fighter_b": "Beta",
        "odds_a_decimal": 3.60, "odds_b_decimal": 1.32, "fonte": "ambas",
        "odds_a_tp": odd_exec, "odds_b_tp": 1.30,
        "ref_tp": ref_a, "ref_tp_a": ref_a, "ref_tp_b": ref_b,
    }]).to_csv(m["card"], index=False)
    return pid


class TestSombra:
    def test_registra_os_tres_candidatos_e_as_provas(self, mundo_sombra):
        _mundo_completo(mundo_sombra)
        linhas = sombra.montar("2026-11-07", peso=0.3, eventos_dev=[])
        d = linhas[0]
        for c in ("p_mercado_a", "p_modelo_a", "p_combinacao_a",
                  "manifesto_sha256", "modelos_sha256", "prediction_id",
                  "ref_preco", "casa", "mercado", "preco_lido_em",
                  "inicio_programado", "versao_protocolo", "peso_w"):
            assert not sombra._vazio(d.get(c)), c
        assert d["casa"] == "Thunderpick" and d["mercado"] == "h2h"

    def test_com_peso_zero_a_combinacao_E_o_mercado(self, mundo_sombra):
        """O experimento concluiu w=0. A sombra registra a decisao que o
        procedimento de fato produz, nao a que daria mais entradas."""
        _mundo_completo(mundo_sombra)
        d = sombra.montar("2026-11-07", peso=0.0, eventos_dev=[])[0]
        assert d["p_combinacao_a"] == pytest.approx(d["p_mercado_a"], abs=1e-4)

    def test_separa_retorno_bruto_de_lucro_liquido(self, mundo_sombra):
        _mundo_completo(mundo_sombra)
        d = sombra.montar("2026-11-07", peso=0.0, eventos_dev=[])[0]
        assert d["lucro_liquido_esperado"] == pytest.approx(
            d["retorno_bruto_esperado"] - 1.0, abs=1e-4)

    def test_sem_preco_executavel_recusa_com_motivo(self, mundo_sombra):
        _mundo_completo(mundo_sombra)
        card = pd.read_csv(mundo_sombra["card"])
        card["odds_a_tp"] = None
        card.to_csv(mundo_sombra["card"], index=False)
        d = sombra.montar("2026-11-07", peso=0.0, eventos_dev=[])[0]
        assert d["selecao"] is False
        assert "mediana de casas nao e cotacao executavel" in d["motivo"]

    def test_prova_do_adversario_recusa(self, mundo_sombra):
        _mundo_completo(mundo_sombra)
        card = pd.read_csv(mundo_sombra["card"], dtype=str)
        card.loc[0, "ref_tp_a"] = card.loc[0, "ref_tp_b"]
        card.loc[0, "ref_tp"] = card.loc[0, "ref_tp_b"]
        card.to_csv(mundo_sombra["card"], index=False)
        d = sombra.montar("2026-11-07", peso=0.0, eventos_dev=[])[0]
        assert d["selecao"] is False
        assert "adversario" in d["motivo"]

    def test_evento_usado_no_ajuste_e_desenvolvimento(self, mundo_sombra):
        _mundo_completo(mundo_sombra)
        d = sombra.montar("2026-11-07", peso=0.0,
                          eventos_dev=["2026-11-07"])[0]
        assert d["fatia"] == "desenvolvimento"
        d2 = sombra.montar("2026-11-07", peso=0.0, eventos_dev=["2026-01-01"])[0]
        assert d2["fatia"] == "avaliacao_prospectiva"

    def test_so_anexa(self, mundo_sombra):
        _mundo_completo(mundo_sombra)
        linhas = sombra.montar("2026-11-07", peso=0.0, eventos_dev=[])
        sombra.registrar(linhas, mundo_sombra["reg"])
        antes = mundo_sombra["reg"].read_text(encoding="utf-8")
        sombra.registrar(linhas, mundo_sombra["reg"])
        depois = mundo_sombra["reg"].read_text(encoding="utf-8")
        assert depois.startswith(antes), "linha ja escrita foi tocada"
        assert len(pd.read_csv(mundo_sombra["reg"])) == 2

    def test_a_liquidacao_nao_reescreve_a_previsao(self, mundo_sombra):
        _mundo_completo(mundo_sombra, vencedor="Alpha")
        linhas = sombra.montar("2026-11-07", peso=0.0, eventos_dev=[])
        sombra.registrar(linhas, mundo_sombra["reg"])
        antes = hashlib.sha256(
            mundo_sombra["reg"].read_bytes()).hexdigest()
        n = sombra.liquidar("2026-11-07", arquivo=mundo_sombra["res"],
                            registros=mundo_sombra["reg"])
        assert n == 1
        assert hashlib.sha256(
            mundo_sombra["reg"].read_bytes()).hexdigest() == antes
        res = pd.read_csv(mundo_sombra["res"])
        assert res.iloc[0]["registro_id"] in set(
            pd.read_csv(mundo_sombra["reg"], dtype=str)["registro_id"])

    def test_liquidar_duas_vezes_nao_duplica(self, mundo_sombra):
        _mundo_completo(mundo_sombra, vencedor="Alpha")
        sombra.registrar(sombra.montar("2026-11-07", peso=0.0, eventos_dev=[]),
                         mundo_sombra["reg"])
        sombra.liquidar("2026-11-07", arquivo=mundo_sombra["res"],
                        registros=mundo_sombra["reg"])
        assert sombra.liquidar("2026-11-07", arquivo=mundo_sombra["res"],
                               registros=mundo_sombra["reg"]) == 0

    def test_perna_recusada_nao_movimenta_unidade(self, mundo_sombra):
        """Recusa nao e aposta perdida: e aposta que nao houve."""
        _mundo_completo(mundo_sombra, odd_exec=1.05, vencedor="Beta")
        linhas = sombra.montar("2026-11-07", peso=0.0, eventos_dev=[])
        assert linhas[0]["selecao"] is False
        sombra.registrar(linhas, mundo_sombra["reg"])
        sombra.liquidar("2026-11-07", arquivo=mundo_sombra["res"],
                        registros=mundo_sombra["reg"])
        res = pd.read_csv(mundo_sombra["res"])
        assert float(res.iloc[0]["unidades"]) == 0.0
        assert float(res.iloc[0]["retorno_unidades"]) == 0.0

    def test_a_politica_e_de_unidade_fixa(self):
        assert sombra.UNIDADES_POR_PERNA == 1.0
        assert sombra.LIMIAR_RETORNO_BRUTO == 1.0
