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
def mundo(tmp_path, monkeypatch):
    """Mundo isolado: proveniência, versões, candidatos e sombra em tmp."""
    from src import candidatos, proveniencia, snapshots
    prov = tmp_path / "prov"
    monkeypatch.setattr(proveniencia, "PASTA", prov)
    monkeypatch.setattr(proveniencia, "INDICE", prov / "chamadas.csv")
    monkeypatch.setattr(proveniencia, "INDICE_COTACOES", prov / "cotacoes.csv")
    monkeypatch.setattr(config, "PREDICTION_HISTORY_CSV", tmp_path / "h.csv")
    monkeypatch.setattr(config, "UPCOMING_CARD_CSV", tmp_path / "card.csv")
    monkeypatch.setattr(snapshots, "ARQUIVO", tmp_path / "snaps.csv")
    monkeypatch.setattr(candidatos, "ARQUIVO", tmp_path / "candidatos.json")
    monkeypatch.setattr(sombra, "REGISTROS", tmp_path / "sombra.csv")
    monkeypatch.setattr(sombra, "RESULTADOS", tmp_path / "sombra_res.csv")
    snapshots._CACHE.clear()
    return {"tmp": tmp_path, "hist": tmp_path / "h.csv",
            "card": tmp_path / "card.csv", "snap": tmp_path / "snaps.csv",
            "cand": tmp_path / "candidatos.json",
            "reg": tmp_path / "sombra.csv", "res": tmp_path / "sombra_res.csv"}


def _iso(h):
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) + timedelta(hours=h)).isoformat(
        timespec="seconds")


def _montar_mundo(m, *, odd_a=3.60, odd_b=1.30, prob_modelo=0.30,
                  inicio_h=5, criado_h=-2, lado=None, vencedor=None,
                  sem_ref_b=False, odds_mediana=None):
    """
    Card + versão com `criado_em` + prova de preço dos DOIS lados.

    `criado_h` negativo = previsão criada antes de agora; o início fica
    `inicio_h` horas à frente, então a previsão é comprovadamente pré-luta.
    """
    import json as _json

    from src import candidatos, proveniencia, snapshots
    inicio = _iso(inicio_h)
    corpo = _json.dumps([{"home": "Alpha", "away": "Beta",
                          "odd_home": odd_a, "odd_away": odd_b,
                          "inicio": inicio, "status": 1}],
                        ensure_ascii=False, sort_keys=True)
    chamada = proveniencia.registrar_chamada(
        corpo, fonte="thunderpick", endpoint="https://thunderpick.io/x",
        markets="h2h")
    comum = dict(chamada_ref=chamada, fonte="thunderpick", casa="Thunderpick",
                 mercado="h2h", evento_id=inicio, participante_a="Alpha",
                 participante_b="Beta", status="aberto",
                 inicio_programado=inicio)
    ref_a = proveniencia.registrar_cotacao(lado="Alpha", odd=odd_a, **comum)
    ref_b = proveniencia.registrar_cotacao(lado="Beta", odd=odd_b, **comum)
    pid = snapshots.registrar([{
        "event_date": "2026-11-07", "event_name": "UFC Sombra",
        "fighter_a": "Alpha", "fighter_b": "Beta", "model_name": "logreg",
        "model_prob_a": prob_modelo, "model_side": lado or "Beta",
        "manifesto_sha256": "man-1", "modelos_sha256": "mod-1",
    }], m["snap"], criado_em=_iso(criado_h))[0]
    med_a, med_b = odds_mediana or (3.50, 1.33)
    pd.DataFrame([{
        "event_name": "UFC Sombra", "event_date": "2026-11-07",
        "fighter_a": "Alpha", "fighter_b": "Beta",
        "odds_a_decimal": med_a, "odds_b_decimal": med_b,
        "model_name": "logreg", "model_prob_a": prob_modelo,
        "model_side": lado or "Beta", "actual_winner": vencedor,
        "estado_resultado": "aguardando", "prediction_id": pid,
    }]).to_csv(m["hist"], index=False)
    pd.DataFrame([{
        "fighter_a": "Alpha", "fighter_b": "Beta",
        "odds_a_decimal": med_a, "odds_b_decimal": med_b, "fonte": "ambas",
        "odds_a_tp": odd_a, "odds_b_tp": odd_b,
        "ref_tp": ref_a, "ref_tp_a": ref_a,
        "ref_tp_b": ("" if sem_ref_b else ref_b),
    }]).to_csv(m["card"], index=False)
    cand = candidatos.congelar(
        versao_protocolo="vantagem-v2", peso_w=0.0,
        eventos_desenvolvimento=[], eventos_consultados=[],
        congelado_em=_iso(-3), arquivo=m["cand"])
    return {"pid": pid, "ref_a": ref_a, "ref_b": ref_b, "inicio": inicio,
            "cand": cand}


# ---------------------------------------------------------------------------
# classificacao temporal (E1 e E2 da errata)
# ---------------------------------------------------------------------------

class TestClassificacaoTemporal:
    def test_previsao_comprovada(self, mundo):
        from src import temporalidade
        d = _montar_mundo(mundo)
        ev = temporalidade.da_previsao(d["pid"], inicio=d["inicio"])
        assert ev.classe == temporalidade.COMPROVADA
        assert ev.instante and ev.inicio == d["inicio"]

    def test_versao_semeada_e_DESCONHECIDA_nao_comprovada(self, mundo):
        """Constar do pre-registro e indicio, nao prova: semeadura nao tem
        criado_em."""
        from src import snapshots, temporalidade
        pid = snapshots.registrar([{
            "event_date": "2026-11-07", "fighter_a": "Alpha",
            "fighter_b": "Beta", "model_prob_a": 0.5,
            "temporalidade": snapshots.TEMPORALIDADE_PERDIDA,
        }], mundo["snap"], criado_em="")[0]
        ev = temporalidade.da_previsao(pid, inicio=_iso(5))
        assert ev.classe == temporalidade.DESCONHECIDA
        assert "NAO RECUPERAVEL" in ev.detalhe

    def test_previsao_criada_depois_do_inicio_e_INCOMPATIVEL(self, mundo):
        from src import temporalidade
        d = _montar_mundo(mundo, inicio_h=1, criado_h=3)
        ev = temporalidade.da_previsao(d["pid"], inicio=d["inicio"])
        assert ev.classe == temporalidade.INCOMPATIVEL

    def test_sem_inicio_nao_da_comprovada(self, mundo):
        """Horario desconhecido e desconhecido, nao aprovado por omissao."""
        from src import temporalidade
        d = _montar_mundo(mundo)
        ev = temporalidade.da_previsao(d["pid"], inicio="")
        assert ev.classe == temporalidade.DESCONHECIDA

    def test_desconhecida_nao_e_incompativel(self):
        from src import temporalidade
        assert temporalidade.DESCONHECIDA != temporalidade.INCOMPATIVEL

    def test_preco_comprovado_exige_prova_do_lado(self, mundo):
        from src import temporalidade
        d = _montar_mundo(mundo)
        ok = temporalidade.do_preco(d["ref_a"], inicio=d["inicio"], lado="Alpha",
                                    odd=3.60, participantes=("Alpha", "Beta"),
                                    casa="Thunderpick", mercado="h2h")
        assert ok.classe == temporalidade.COMPROVADA
        nao = temporalidade.do_preco(d["ref_a"], inicio=d["inicio"], lado="Beta",
                                     odd=1.30, participantes=("Alpha", "Beta"),
                                     casa="Thunderpick", mercado="h2h")
        assert nao.classe == temporalidade.DESCONHECIDA
        assert "adversario" in nao.detalhe

    def test_sem_referencia_o_preco_e_AUSENTE(self):
        from src import temporalidade
        assert temporalidade.do_preco("").classe == temporalidade.AUSENTE

    def test_a_mediana_e_sempre_nao_certificada(self):
        from src import temporalidade
        ev = temporalidade.da_mediana()
        assert ev.classe == temporalidade.NAO_CERTIFICADA
        assert "nao e cotacao executavel" in ev.detalhe
        assert "nao e prova de captura pre-luta" in ev.detalhe

    def test_a_data_do_evento_nao_serve_de_inicio(self, mundo):
        """`inicio_programado` sai da cotacao, nunca da data do evento: num
        card as lutas comecam ao longo de cinco horas."""
        from src import temporalidade
        d = _montar_mundo(mundo)
        inicio, origem = temporalidade.inicio_programado(d["ref_a"])
        assert inicio == d["inicio"] and origem.startswith("cotacoes.csv:")
        assert temporalidade.inicio_programado("") == ("", "")


class TestPopulacaoClassificada:
    def test_sem_evidencia_a_populacao_certificada_e_ZERO(self, tmp_path):
        from src import vantagem
        h = tmp_path / "h.csv"
        _historico(h, eventos=4, por_evento=3)
        pop, _ = vantagem.carregar_populacao(h)
        assert len(pop) == 12
        assert int(pop["certificada"].sum()) == 0
        assert (pop["classe_analise"] == "exploratoria_historica").all()
        assert (pop["temporalidade_previsao"] == "desconhecida").all()
        assert (pop["temporalidade_mediana"] == "nao_certificada").all()

    def test_cada_linha_registra_a_evidencia_encontrada(self, tmp_path):
        from src import vantagem
        h = tmp_path / "h.csv"
        _historico(h, eventos=2, por_evento=2)
        pop, _ = vantagem.carregar_populacao(h)
        for col in ("evidencia_previsao", "evidencia_mediana",
                    "evidencia_preco_executavel"):
            assert pop[col].astype(str).str.len().gt(10).all(), col


# ---------------------------------------------------------------------------
# qualificacao prospectiva por prova POSITIVA (E3)
# ---------------------------------------------------------------------------

class TestQualificacaoProspectiva:
    def test_sem_prova_da_mediana_o_maximo_e_PARCIAL(self, mundo):
        _montar_mundo(mundo)
        d = sombra.montar("2026-11-07")[0]
        assert d["classe"] == sombra.CLASSE_PARCIAL
        assert "NAO e plenamente auditavel" in d["motivos_da_classe"]

    def test_decisao_depois_do_inicio_nao_qualifica(self, mundo):
        from datetime import datetime, timedelta, timezone
        _montar_mundo(mundo, inicio_h=1)
        depois = datetime.now(timezone.utc) + timedelta(hours=3)
        d = sombra.montar("2026-11-07", agora=depois)[0]
        assert d["classe"] == sombra.CLASSE_NAO_QUALIFICADA
        assert "DEPOIS do inicio programado" in d["motivos_da_classe"]

    def test_horario_desconhecido_nao_qualifica(self, mundo):
        _montar_mundo(mundo, sem_ref_b=True)
        card = pd.read_csv(mundo["card"], dtype=str)
        card.loc[0, ["ref_tp", "ref_tp_a"]] = ""
        card.to_csv(mundo["card"], index=False)
        d = sombra.montar("2026-11-07")[0]
        assert d["classe"] == sombra.CLASSE_NAO_QUALIFICADA
        assert "sem inicio programado verificavel" in d["motivos_da_classe"]

    def test_candidato_congelado_DEPOIS_da_decisao_nao_qualifica(self, mundo):
        from src import candidatos
        _montar_mundo(mundo)
        tarde = candidatos.congelar(
            versao_protocolo="vantagem-v2", peso_w=0.0,
            congelado_em=_iso(+1), arquivo=mundo["cand"],
            descricao="congelado depois da decisao")
        d = sombra.montar("2026-11-07", candidato=tarde)[0]
        assert d["classe"] == sombra.CLASSE_NAO_QUALIFICADA
        assert "congelado DEPOIS da decisao" in d["motivos_da_classe"]

    def test_sem_candidato_nao_qualifica(self, mundo):
        _montar_mundo(mundo)
        d = sombra.montar("2026-11-07", candidato={}, queimados=set())[0]
        assert d["classe"] == sombra.CLASSE_NAO_QUALIFICADA
        assert "nao ha candidato congelado" in d["motivos_da_classe"]

    def test_evento_queimado_e_DESENVOLVIMENTO_para_sempre(self, mundo):
        """Os 16 eventos do v1 -- os 8 de estimacao E os 8 de avaliacao --
        nunca podem voltar como prospectivos."""
        _montar_mundo(mundo)
        d = sombra.montar("2026-11-07", queimados={"2026-11-07"})[0]
        assert d["classe"] == sombra.CLASSE_DESENVOLVIMENTO
        assert "ja foi usado" in d["motivos_da_classe"]

    def test_previsao_sem_temporalidade_nao_qualifica(self, mundo):
        from src import snapshots
        _montar_mundo(mundo)
        snaps = pd.read_csv(mundo["snap"], dtype=str).fillna("")
        snaps.loc[0, "criado_em"] = ""
        snaps.to_csv(mundo["snap"], index=False)
        snapshots._CACHE.clear()
        d = sombra.montar("2026-11-07")[0]
        assert d["classe"] == sombra.CLASSE_NAO_QUALIFICADA
        assert "temporalidade" in d["motivos_da_classe"]

    def test_os_quatro_itens_ficam_separados_na_linha(self, mundo):
        """(a) previsao, (b) mediana, (c) preco executavel, (d) decisao."""
        _montar_mundo(mundo)
        d = sombra.montar("2026-11-07")[0]
        assert d["instante_previsao"]
        assert d["temporalidade_mediana"]
        assert d["temporalidade_preco_a"]
        assert d["decidido_em"]
        assert d["instante_previsao"] < d["decidido_em"]


# ---------------------------------------------------------------------------
# politica de selecao (E4)
# ---------------------------------------------------------------------------

class TestPolitica:
    def test_com_w_zero_mudar_model_side_NAO_altera_a_selecao(self, mundo):
        """O defeito do v1: a probabilidade vinha do mercado e o LADO vinha do
        modelo."""
        _montar_mundo(mundo, odd_a=4.50, lado="Alpha")
        com_alpha = sombra.montar("2026-11-07")[0]
        _montar_mundo(mundo, odd_a=4.50, lado="Beta")
        com_beta = sombra.montar("2026-11-07")[0]
        assert com_alpha["selecao"] is True, "o caso precisa ter selecao"
        assert com_alpha["model_side"] != com_beta["model_side"]
        assert com_alpha["lado_selecionado"] == com_beta["lado_selecionado"]
        assert com_alpha["selecao"] == com_beta["selecao"]

    def test_inverter_A_e_B_preserva_a_decisao_do_mesmo_atleta(self, mundo):
        d = _montar_mundo(mundo)
        direto = sombra.politica(
            p_c_a=0.30, a="Alpha", b="Beta", odd_a=3.60, odd_b=1.30,
            ref_a=d["ref_a"], ref_b=d["ref_b"], inicio=d["inicio"])
        invertido = sombra.politica(
            p_c_a=0.70, a="Beta", b="Alpha", odd_a=1.30, odd_b=3.60,
            ref_a=d["ref_b"], ref_b=d["ref_a"], inicio=d["inicio"])
        assert direto["lado"] == invertido["lado"]
        assert direto["odd"] == invertido["odd"]
        assert direto["prob"] == pytest.approx(invertido["prob"])

    def test_um_lado_sem_prova_fica_indisponivel(self, mundo):
        """E nunca recebe o preco do adversario."""
        d = _montar_mundo(mundo)
        r = sombra.politica(p_c_a=0.30, a="Alpha", b="Beta", odd_a=3.60,
                            odd_b=1.30, ref_a=d["ref_a"], ref_b="",
                            inicio=d["inicio"])
        assert r["lucro_liquido_b"] is None
        assert r["evidencia_b"].classe == "ausente"

    def test_nenhum_lado_com_expectativa_positiva_recusa(self, mundo):
        d = _montar_mundo(mundo, odd_a=1.05, odd_b=1.05)
        r = sombra.politica(p_c_a=0.50, a="Alpha", b="Beta", odd_a=1.05,
                            odd_b=1.05, ref_a=d["ref_a"], ref_b=d["ref_b"],
                            inicio=d["inicio"])
        assert r["selecao"] is False
        assert "lucro esperado liquido positivo" in r["motivo"]

    def test_ambos_positivos_escolhe_o_maior_e_SO_um(self, mundo):
        d = _montar_mundo(mundo, odd_a=4.00, odd_b=3.00)
        r = sombra.politica(p_c_a=0.50, a="Alpha", b="Beta", odd_a=4.00,
                            odd_b=3.00, ref_a=d["ref_a"], ref_b=d["ref_b"],
                            inicio=d["inicio"])
        assert r["selecao"] is True
        assert r["lado"] == "Alpha"
        assert r["lucro_liquido_a"] > r["lucro_liquido_b"] > 0

    def test_empate_exato_recusa_com_motivo(self, mundo):
        d = _montar_mundo(mundo, odd_a=3.00, odd_b=3.00)
        r = sombra.politica(p_c_a=0.50, a="Alpha", b="Beta", odd_a=3.00,
                            odd_b=3.00, ref_a=d["ref_a"], ref_b=d["ref_b"],
                            inicio=d["inicio"])
        assert r["selecao"] is False
        assert "empate exato" in r["motivo"]

    def test_modelo_e_mercado_discordantes_nao_mudam_a_politica(self, mundo):
        """Com w=0 a politica usa p do MERCADO, discorde o modelo quanto
        quiser."""
        _montar_mundo(mundo, odd_a=4.50, prob_modelo=0.95, lado="Alpha")
        alto = sombra.montar("2026-11-07")[0]
        _montar_mundo(mundo, odd_a=4.50, prob_modelo=0.05, lado="Beta")
        baixo = sombra.montar("2026-11-07")[0]
        assert alto["p_modelo_a"] != baixo["p_modelo_a"]
        assert alto["p_combinacao_a"] == baixo["p_combinacao_a"]
        assert alto["lado_selecionado"] == baixo["lado_selecionado"]

    def test_a_politica_nao_e_configuravel_por_chamada(self):
        assert sombra.UNIDADES_POR_PERNA == 1.0
        assert sombra.LIMIAR_LUCRO_LIQUIDO == 0.0


# ---------------------------------------------------------------------------
# preservacao (itens 1 e 6)
# ---------------------------------------------------------------------------

class TestPreservacao:
    def test_a_sombra_so_anexa(self, mundo):
        _montar_mundo(mundo)
        linhas = sombra.montar("2026-11-07")
        sombra.registrar(linhas, mundo["reg"])
        antes = mundo["reg"].read_text(encoding="utf-8")
        sombra.registrar(linhas, mundo["reg"])
        assert mundo["reg"].read_text(encoding="utf-8").startswith(antes)

    def test_a_liquidacao_nao_reescreve_a_previsao(self, mundo):
        _montar_mundo(mundo, vencedor="Alpha")
        sombra.registrar(sombra.montar("2026-11-07"), mundo["reg"])
        antes = hashlib.sha256(mundo["reg"].read_bytes()).hexdigest()
        assert sombra.liquidar("2026-11-07", arquivo=mundo["res"],
                               registros=mundo["reg"]) == 1
        assert hashlib.sha256(mundo["reg"].read_bytes()).hexdigest() == antes

    def test_captura_repetida_conta_UMA_aposta(self, mundo):
        """Tres capturas do mesmo card nao sao tres apostas na mesma luta."""
        from datetime import datetime, timedelta, timezone
        _montar_mundo(mundo, odd_a=4.50, vencedor="Alpha")
        agora = datetime.now(timezone.utc)
        for k in range(3):
            sombra.registrar(
                sombra.montar("2026-11-07", agora=agora + timedelta(minutes=k)),
                mundo["reg"])
        sombra.liquidar("2026-11-07", arquivo=mundo["res"],
                        registros=mundo["reg"])
        res = pd.read_csv(mundo["res"])
        assert len(res) == 3
        assert abs(float(res["unidades"].sum()) - 1.0) < 1e-9, (
            "a mesma luta contou mais de uma vez")

    def test_candidato_congelado_nao_e_editado(self, mundo):
        from src import candidatos
        _montar_mundo(mundo)
        candidatos.congelar(versao_protocolo="vantagem-v2", peso_w=0.5,
                            congelado_em=_iso(-1), arquivo=mundo["cand"])
        depois = candidatos.carregar(mundo["cand"])
        assert len(depois) == 2
        assert depois[0]["peso_w"] == 0.0 and depois[1]["peso_w"] == 0.5

    def test_eventos_queimados_juntam_estimacao_e_inferencia(self):
        from src import candidatos
        c = {"eventos_desenvolvimento": ["a", "b"], "eventos_consultados": ["c"]}
        assert candidatos.eventos_queimados(c) == {"a", "b", "c"}

    def test_os_16_eventos_do_v1_ficam_queimados(self):
        """O v1 so bloqueava os 8 de desenvolvimento; os 8 de avaliacao
        gastaram a inferencia e tambem nao podem voltar."""
        _peso, queimados = sombra.peso_registrado()
        assert len(queimados) == 16
        assert "2026-07-11" in queimados and "2026-10-31" in queimados


# ---------------------------------------------------------------------------
# precisao e incerteza (E5, E6, E9)
# ---------------------------------------------------------------------------

class TestPrecisaoEIncerteza:
    def test_a_funcao_se_chama_precisao_observada(self):
        from src import vantagem as v
        assert hasattr(v, "precisao_observada")
        assert not hasattr(v, "detectavel"), (
            "meia-largura de IC nao e efeito minimo detectavel")
        doc = v.precisao_observada.__doc__.lower()
        assert "efeito mínimo detectável" in doc
        assert "precisão desta amostra" in doc

    def test_calibracao_marca_wilson_otimista(self):
        from src import vantagem as v
        p = np.array([0.3] * 6)
        y = np.array([1, 0, 1, 0, 1, 0])
        ev = ["e1", "e1", "e1", "e2", "e2", "e2"]
        t = v.tabela_de_calibracao(p, y, eventos=ev)
        faixa = t[t["n"] > 0].iloc[0]
        assert faixa["n"] == 6 and faixa["n_eventos"] == 2
        assert bool(faixa["wilson_otimista"]) is True
        assert faixa["ic95_agrupado_baixo"] is not None

    def test_sem_eventos_nao_finge_incerteza_agrupada(self):
        from src import vantagem as v
        t = v.tabela_de_calibracao(np.array([0.3, 0.3]), np.array([1, 0]))
        faixa = t[t["n"] > 0].iloc[0]
        assert faixa["wilson_otimista"] is None
        assert faixa["ic95_agrupado_baixo"] is None
