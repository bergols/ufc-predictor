"""
Proveniência da MEDIANA de casas — a referência de mercado (`src/mediana.py`).

Até `vantagem-v2` a classe máxima no modo sombra era `prospectiva_parcial`
porque a mediana era calculada e gravada **sem prova**: a odd da Pinnacle e a
da Thunderpick tinham proveniência por preço, e o candidato A inteiro não tinha
nenhuma.

O que estes testes fixam:

  * a mediana é amarrada à captura exata, com hash do corpo e horário real;
  * **corpos idênticos em horários diferentes são provas diferentes**;
  * o casamento é o ESTRITO: revanche, confronto errado e ambiguidade
    bloqueiam, e a prova nunca é de outra luta;
  * a conferência **recalcula** as medianas da captura e compara com as que
    entraram na decisão;
  * a ordem temporal é verificada contra o instante da decisão;
  * sem prova, a classificação continua **parcial** — nunca é promovida por
    omissão.

Nenhuma chamada de API é consumida: tudo sai de respostas sintéticas.
"""
import gzip
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import config
from src import mediana, proveniencia, sombra, temporalidade


def _iso(h):
    return (datetime.now(timezone.utc) + timedelta(hours=h)).isoformat(
        timespec="seconds")


def _evento(*, id_="ev1", inicio=None, a="Alpha", b="Beta",
            casas=((1.80, 2.20), (1.90, 2.10), (2.00, 2.00)),
            so_um_lado=False, atualizada="2026-11-09T18:00:00Z"):
    """Um evento na forma que a The Odds API devolve."""
    bms = []
    for i, (oa, ob) in enumerate(casas):
        saidas = [{"name": a, "price": oa}]
        if not so_um_lado:
            saidas.append({"name": b, "price": ob})
        bms.append({"key": f"casa{i}", "title": f"Casa {i}",
                    "last_update": atualizada,
                    "markets": [{"key": "h2h", "last_update": atualizada,
                                 "outcomes": saidas}]})
    return {"id": id_, "sport_key": "mma_mixed_martial_arts",
            "commence_time": inicio or _iso(5), "home_team": a, "away_team": b,
            "bookmakers": bms}


@pytest.fixture
def prov(tmp_path, monkeypatch):
    pasta = tmp_path / "prov"
    monkeypatch.setattr(proveniencia, "PASTA", pasta)
    monkeypatch.setattr(proveniencia, "INDICE", pasta / "chamadas.csv")
    monkeypatch.setattr(proveniencia, "INDICE_COTACOES", pasta / "cotacoes.csv")
    monkeypatch.setattr(mediana, "INDICE", pasta / "medianas.csv")
    return {"pasta": pasta, "med": pasta / "medianas.csv",
            "chamadas": pasta / "chamadas.csv", "tmp": tmp_path}


def _capturar(eventos, *, captured_at=None):
    """Registra a resposta bruta como o pipeline faz, e devolve a chamada_ref."""
    ref = proveniencia.registrar_chamada(
        json.dumps(eventos, ensure_ascii=False), fonte="the_odds_api",
        endpoint="https://api.the-odds-api.com/v4/sports/x/odds/",
        regions="eu", markets="h2h", n_eventos=len(eventos))
    if captured_at:
        linhas = pd.read_csv(_INDICE(), dtype=str).fillna("")
        linhas.loc[linhas.index[-1], "captured_at"] = captured_at
        linhas.to_csv(_INDICE(), index=False)
    return ref


def _INDICE():
    return proveniencia.INDICE


# ---------------------------------------------------------------------------
# a regra, em um lugar so
# ---------------------------------------------------------------------------

class TestRegra:
    def test_mediana_por_lado_com_tres_casas(self):
        oa, ob, casas = mediana.calcular(_evento(), "Alpha", "Beta")
        assert oa == 1.90 and ob == 2.10
        assert casas == ["casa0", "casa1", "casa2"]

    def test_so_entra_casa_que_precifica_OS_DOIS_lados(self):
        """Uma casa que lista so um lado desequilibraria o par e a devigagem
        sairia de duas amostras diferentes."""
        oa, ob, casas = mediana.calcular(_evento(so_um_lado=True),
                                         "Alpha", "Beta")
        assert oa is None and ob is None and casas == []

    def test_build_card_usa_a_MESMA_funcao(self):
        """Duas implementacoes da mesma regra divergiriam em silencio."""
        import inspect

        from scripts import build_card
        assert "src.mediana" in inspect.getsource(build_card._medianas)
        ev = _evento()
        assert build_card._medianas(ev, "Alpha", "Beta")[:2] == (1.90, 2.10)

    def test_precisao_de_tres_casas(self):
        ev = _evento(casas=((1.8333, 2.0), (1.8334, 2.0), (1.8335, 2.0)))
        oa, _ob, _c = mediana.calcular(ev, "Alpha", "Beta")
        assert oa == 1.833


# ---------------------------------------------------------------------------
# a referencia: captura exata, hash e horario
# ---------------------------------------------------------------------------

class TestReferencia:
    def test_amarra_captura_evento_participantes_casas_e_odds(self, prov):
        ev = _evento()
        ch = _capturar([ev])
        ref, dados, motivo = mediana.casar_e_registrar(
            "Alpha", "Beta", [ev], event_date=ev["commence_time"][:10],
            chamada_ref=ch)
        assert ref and not motivo
        linha = mediana.linha(ref)
        assert linha["chamada_ref"] == ch
        assert linha["evento_id"] == "ev1"
        assert linha["participante_a"] == "Alpha"
        assert linha["participante_b"] == "Beta"
        assert linha["mercado"] == "h2h"
        assert linha["n_casas"] == "3"
        assert linha["casas"] == "casa0,casa1,casa2"
        assert float(linha["odd_a"]) == 1.90 and float(linha["odd_b"]) == 2.10
        assert linha["captured_at"] and linha["inicio_programado"]
        assert dados["n_casas"] == 3

    def test_corpo_IDENTICO_em_horarios_diferentes_da_refs_diferentes(self, prov):
        """O hash e do conteudo, entao duas capturas iguais colapsam na mesma
        chamada_ref. A prova de ontem nao comprova disponibilidade hoje."""
        ev = _evento(inicio=_iso(10))
        ch1 = _capturar([ev])
        r1 = mediana.registrar(chamada_ref=ch1, captured_at=_iso(-5),
                               evento_id="ev1", participante_a="Alpha",
                               participante_b="Beta", odd_a=1.90, odd_b=2.10,
                               casas=["casa0", "casa1", "casa2"],
                               inicio_programado=ev["commence_time"])
        ch2 = _capturar([ev])
        assert ch1 == ch2, "corpos identicos tem a mesma chamada_ref"
        r2 = mediana.registrar(chamada_ref=ch2, captured_at=_iso(-1),
                               evento_id="ev1", participante_a="Alpha",
                               participante_b="Beta", odd_a=1.90, odd_b=2.10,
                               casas=["casa0", "casa1", "casa2"],
                               inicio_programado=ev["commence_time"])
        assert r1 != r2, "as duas capturas viraram a mesma prova"
        assert mediana.linha(r1)["captured_at"] != mediana.linha(r2)["captured_at"]

    def test_orientacao_A_B_fica_na_prova(self, prov):
        ev = _evento()
        ch = _capturar([ev])
        ref, _d, _m = mediana.casar_e_registrar(
            "Beta", "Alpha", [ev], event_date=ev["commence_time"][:10],
            chamada_ref=ch)
        linha = mediana.linha(ref)
        assert (linha["participante_a"], linha["participante_b"]) == ("Beta", "Alpha")
        assert float(linha["odd_a"]) == 2.10 and float(linha["odd_b"]) == 1.90


class TestCasamentoEstrito:
    def test_revanche_bloqueia(self, prov):
        """Os mesmos nomes em datas diferentes: dois candidatos na janela."""
        hoje = _evento(id_="ev1", inicio=_iso(5))
        amanha = _evento(id_="ev2", inicio=_iso(20))
        ch = _capturar([hoje, amanha])
        ref, _d, motivo = mediana.casar_e_registrar(
            "Alpha", "Beta", [hoje, amanha],
            event_date=hoje["commence_time"][:10], chamada_ref=ch)
        assert not ref and "casamento bloqueado" in motivo

    def test_confronto_errado_bloqueia(self, prov):
        ev = _evento(a="Gama", b="Delta")
        ch = _capturar([ev])
        ref, _d, motivo = mediana.casar_e_registrar(
            "Alpha", "Beta", [ev], event_date=ev["commence_time"][:10],
            chamada_ref=ch)
        assert not ref and "casamento bloqueado" in motivo

    def test_correspondencia_apenas_aproximada_bloqueia(self, prov):
        """Fuzzy sugere candidato; nao aprova prova."""
        ev = _evento(a="Alphah", b="Beta")
        ch = _capturar([ev])
        ref, _d, motivo = mediana.casar_e_registrar(
            "Alpha", "Beta", [ev], event_date=ev["commence_time"][:10],
            chamada_ref=ch)
        assert not ref and "casamento bloqueado" in motivo


# ---------------------------------------------------------------------------
# conferencia: reabrir, revalidar, RECALCULAR
# ---------------------------------------------------------------------------

class TestConferencia:
    def _pronto(self, prov, **kw):
        ev = _evento(**kw)
        ch = _capturar([ev])
        ref, dados, _m = mediana.casar_e_registrar(
            "Alpha", "Beta", [ev], event_date=ev["commence_time"][:10],
            chamada_ref=ch)
        return ev, ch, ref, dados

    def test_captura_integra_e_decisao_pre_luta_sao_COMPROVADAS(self, prov):
        ev, _ch, ref, _d = self._pronto(prov)
        e = mediana.evidencia(ref, participantes=("Alpha", "Beta"),
                              odd_a=1.90, odd_b=2.10, decidido_em=_iso(0))
        assert e.classe == temporalidade.COMPROVADA, e.detalhe
        assert "recalculada da captura" in e.detalhe

    def test_recalcula_e_pega_mediana_divergente(self, prov):
        _ev, _ch, ref, _d = self._pronto(prov)
        linhas = pd.read_csv(prov["med"], dtype=str).fillna("")
        linhas.loc[0, "odd_a"] = "9.99"
        linhas.to_csv(prov["med"], index=False)
        e = mediana.evidencia(ref, participantes=("Alpha", "Beta"))
        assert e.classe == temporalidade.INCOMPATIVEL
        assert "diverge da registrada" in e.detalhe

    def test_odd_usada_na_decisao_diferente_da_provada_bloqueia(self, prov):
        _ev, _ch, ref, _d = self._pronto(prov)
        e = mediana.evidencia(ref, participantes=("Alpha", "Beta"),
                              odd_a=1.55, odd_b=2.10)
        assert e.classe == temporalidade.INCOMPATIVEL
        assert "a decisao usou odd_a=1.55" in e.detalhe

    def test_hash_divergente_bloqueia(self, prov):
        _ev, ch, ref, _d = self._pronto(prov)
        linhas = pd.read_csv(prov["chamadas"], dtype=str).fillna("")
        linhas.loc[linhas["ref"] == ch, "sha256"] = "0" * 64
        linhas.to_csv(prov["chamadas"], index=False)
        e = mediana.evidencia(ref, participantes=("Alpha", "Beta"))
        assert e.classe == temporalidade.DESCONHECIDA
        assert "NAO bate com o hash" in e.detalhe

    def test_corpo_ausente_bloqueia(self, prov):
        _ev, ch, ref, _d = self._pronto(prov)
        (prov["pasta"] / f"{ch}.json.gz").unlink()
        e = mediana.evidencia(ref, participantes=("Alpha", "Beta"))
        assert e.classe == temporalidade.DESCONHECIDA
        assert "corpo ausente" in e.detalhe

    def test_prova_de_outro_confronto_bloqueia(self, prov):
        _ev, _ch, ref, _d = self._pronto(prov)
        e = mediana.evidencia(ref, participantes=("Gama", "Delta"))
        assert e.classe == temporalidade.INCOMPATIVEL
        assert "nao e este" in e.detalhe

    def test_evento_fora_da_captura_bloqueia(self, prov):
        _ev, _ch, ref, _d = self._pronto(prov)
        linhas = pd.read_csv(prov["med"], dtype=str).fillna("")
        linhas.loc[0, "evento_id"] = "ev-que-nao-existe"
        linhas.to_csv(prov["med"], index=False)
        e = mediana.evidencia(ref, participantes=("Alpha", "Beta"))
        assert e.classe == temporalidade.DESCONHECIDA
        assert "nao esta na captura referenciada" in e.detalhe

    def test_casas_trocadas_bloqueiam(self, prov):
        _ev, _ch, ref, _d = self._pronto(prov)
        linhas = pd.read_csv(prov["med"], dtype=str).fillna("")
        linhas.loc[0, "casas_sha256"] = "deadbeefdeadbeef"
        linhas.to_csv(prov["med"], index=False)
        e = mediana.evidencia(ref, participantes=("Alpha", "Beta"))
        assert e.classe == temporalidade.INCOMPATIVEL
        assert "casas da captura" in e.detalhe


class TestOrdemTemporal:
    def _com(self, prov, *, captured_at, inicio_h=5):
        ev = _evento(inicio=_iso(inicio_h))
        ch = _capturar([ev])
        ref = mediana.registrar(
            chamada_ref=ch, captured_at=captured_at, evento_id="ev1",
            participante_a="Alpha", participante_b="Beta", odd_a=1.90,
            odd_b=2.10, casas=["casa0", "casa1", "casa2"],
            inicio_programado=ev["commence_time"])
        return ref

    def test_captura_POSTERIOR_a_decisao_bloqueia(self, prov):
        ref = self._com(prov, captured_at=_iso(1))
        e = mediana.evidencia(ref, decidido_em=_iso(0))
        assert e.classe == temporalidade.INCOMPATIVEL
        assert "POSTERIOR a decisao" in e.detalhe

    def test_captura_vencida_bloqueia(self, prov):
        ref = self._com(prov, captured_at=_iso(-9), inicio_h=5)
        e = mediana.evidencia(ref, decidido_em=_iso(0))
        assert e.classe == temporalidade.INCOMPATIVEL
        assert "vencida" in e.detalhe

    def test_captura_depois_do_inicio_bloqueia(self, prov):
        ref = self._com(prov, captured_at=_iso(2), inicio_h=1)
        e = mediana.evidencia(ref, decidido_em=_iso(3))
        assert e.classe == temporalidade.INCOMPATIVEL
        assert "depois do inicio programado" in e.detalhe

    def test_decisao_pos_inicio_bloqueia(self, prov):
        ref = self._com(prov, captured_at=_iso(-1), inicio_h=1)
        e = mediana.evidencia(ref, decidido_em=_iso(3))
        assert e.classe == temporalidade.INCOMPATIVEL
        assert "decisao apos o inicio" in e.detalhe

    def test_horario_sem_fuso_nao_e_aceito(self, prov):
        ref = self._com(prov, captured_at="2026-11-09 18:00:00")
        e = mediana.evidencia(ref, decidido_em=_iso(0))
        assert e.classe == temporalidade.DESCONHECIDA
        assert "sem fuso" in e.detalhe

    def test_inicio_divergente_entre_as_provas_bloqueia(self, prov):
        ref = self._com(prov, captured_at=_iso(-1), inicio_h=5)
        e = mediana.evidencia(ref, inicio=_iso(9), decidido_em=_iso(0))
        assert e.classe == temporalidade.INCOMPATIVEL
        assert "divergente entre as provas" in e.detalhe

    def test_last_update_da_casa_fica_separado_do_nosso_relogio(self, prov):
        ev = _evento(atualizada="2026-11-09T12:34:56Z")
        ch = _capturar([ev])
        ref, _d, _m = mediana.casar_e_registrar(
            "Alpha", "Beta", [ev], event_date=ev["commence_time"][:10],
            chamada_ref=ch)
        linha = mediana.linha(ref)
        assert linha["fonte_atualizada_em"] == "2026-11-09T12:34:56Z"
        assert linha["captured_at"] != linha["fonte_atualizada_em"]


class TestSemProva:
    def test_sem_referencia_continua_nao_certificada(self, prov):
        e = mediana.evidencia("")
        assert e.classe == temporalidade.NAO_CERTIFICADA
        assert "nao esta gravada" in e.detalhe

    def test_referencia_inexistente_e_desconhecida(self, prov):
        e = mediana.evidencia("0123456789abcdef")
        assert e.classe == temporalidade.DESCONHECIDA
        assert "nao esta no indice" in e.detalhe


# ---------------------------------------------------------------------------
# o caminho COMPLETO no modo sombra
# ---------------------------------------------------------------------------

@pytest.fixture
def mundo(tmp_path, monkeypatch):
    from src import candidatos, snapshots
    pasta = tmp_path / "prov"
    monkeypatch.setattr(proveniencia, "PASTA", pasta)
    monkeypatch.setattr(proveniencia, "INDICE", pasta / "chamadas.csv")
    monkeypatch.setattr(proveniencia, "INDICE_COTACOES", pasta / "cotacoes.csv")
    monkeypatch.setattr(mediana, "INDICE", pasta / "medianas.csv")
    monkeypatch.setattr(config, "PREDICTION_HISTORY_CSV", tmp_path / "h.csv")
    monkeypatch.setattr(config, "UPCOMING_CARD_CSV", tmp_path / "card.csv")
    monkeypatch.setattr(snapshots, "ARQUIVO", tmp_path / "snaps.csv")
    monkeypatch.setattr(candidatos, "ARQUIVO", tmp_path / "candidatos.json")
    monkeypatch.setattr(sombra, "REGISTROS", tmp_path / "sombra.csv")
    monkeypatch.setattr(sombra, "RESULTADOS", tmp_path / "sombra_res.csv")
    snapshots._CACHE.clear()
    return {"tmp": tmp_path, "pasta": pasta, "med": pasta / "medianas.csv",
            "hist": tmp_path / "h.csv", "card": tmp_path / "card.csv",
            "snap": tmp_path / "snaps.csv", "cand": tmp_path / "candidatos.json",
            "reg": tmp_path / "sombra.csv", "res": tmp_path / "sombra_res.csv"}


def _card_completo(m, *, com_mediana=True, inicio_h=5, odd_exec_a=3.60,
                   inverter=False, vencedor=None, data=None):
    """
    Card + previsao com `criado_em` + prova do preco executavel dos dois lados
    + prova da MEDIANA. O caminho inteiro, com resposta sintetica.
    """
    from src import candidatos, snapshots
    inicio = _iso(inicio_h)
    # a data do evento sai do PROPRIO inicio: a janela do casamento e de um
    # dia, e fixar uma data literal faria o teste passar ou falhar conforme o
    # dia em que ele roda
    data = data or inicio[:10]
    a, b = ("Beta", "Alpha") if inverter else ("Alpha", "Beta")

    # 1. a captura da The Odds API, como o pipeline persiste
    ev = _evento(inicio=inicio, a="Alpha", b="Beta",
                 casas=((3.40, 1.34), (3.50, 1.33), (3.60, 1.32)))
    chamada = _capturar([ev])
    ref_med, dados, motivo = mediana.casar_e_registrar(
        a, b, [ev], event_date=data, chamada_ref=chamada)
    assert ref_med, motivo
    med_a, med_b = dados["odd_a"], dados["odd_b"]

    # 2. o preco EXECUTAVEL da casa, por lado.
    # As odds e a probabilidade pertencem ao ATLETA, nao a coluna: inverter
    # A/B tem de trocar as colunas E o que elas carregam, senao o teste
    # compara dois mundos diferentes em vez de duas orientacoes do mesmo.
    odd_alpha, odd_beta = odd_exec_a, 1.30
    odd_exec_a = odd_beta if inverter else odd_alpha
    odd_exec_b = odd_alpha if inverter else odd_beta
    p_alpha = 0.30
    prob_col_a = (1 - p_alpha) if inverter else p_alpha
    corpo = json.dumps([{"home": a, "away": b, "odd_home": odd_exec_a,
                         "odd_away": odd_exec_b, "inicio": inicio,
                         "status": 1}], ensure_ascii=False, sort_keys=True)
    ch_tp = proveniencia.registrar_chamada(
        corpo, fonte="thunderpick", endpoint="https://thunderpick.io/x",
        markets="h2h")
    comum = dict(chamada_ref=ch_tp, fonte="thunderpick", casa="Thunderpick",
                 mercado="h2h", evento_id=inicio, participante_a=a,
                 participante_b=b, status="aberto", inicio_programado=inicio)
    ref_a = proveniencia.registrar_cotacao(lado=a, odd=odd_exec_a, **comum)
    ref_b = proveniencia.registrar_cotacao(lado=b, odd=odd_exec_b, **comum)

    # 3. a versao imutavel da previsao, criada ANTES
    pid = snapshots.registrar([{
        "event_date": data, "event_name": "UFC Sintetico",
        "fighter_a": a, "fighter_b": b, "model_name": "logreg",
        "model_prob_a": prob_col_a, "model_side": "Beta",
        "manifesto_sha256": "man-1", "modelos_sha256": "mod-1",
    }], m["snap"], criado_em=_iso(-2))[0]
    pd.DataFrame([{
        "event_name": "UFC Sintetico", "event_date": data,
        "fighter_a": a, "fighter_b": b,
        "odds_a_decimal": med_a, "odds_b_decimal": med_b,
        "model_name": "logreg", "model_prob_a": prob_col_a,
        "model_side": "Beta", "actual_winner": vencedor, "estado_resultado": "aguardando",
        "prediction_id": pid,
    }]).to_csv(m["hist"], index=False)

    # 4. o card, com a referencia de mercado
    pd.DataFrame([{
        "fighter_a": a, "fighter_b": b,
        "odds_a_decimal": med_a, "odds_b_decimal": med_b, "fonte": "ambas",
        "n_casas": dados["n_casas"],
        "ref_mediana": ref_med if com_mediana else "",
        "odds_a_tp": odd_exec_a, "odds_b_tp": odd_exec_b,
        "ref_tp": ref_a, "ref_tp_a": ref_a, "ref_tp_b": ref_b,
    }]).to_csv(m["card"], index=False)

    candidatos.congelar(versao_protocolo="vantagem-v2", peso_w=0.0,
                        congelado_em=_iso(-3), arquivo=m["cand"])
    return {"ref_med": ref_med, "med_a": med_a, "med_b": med_b,
            "ref_a": ref_a, "ref_b": ref_b, "inicio": inicio, "pid": pid,
            "data": data}


class TestFluxoCompleto:
    def test_com_a_mediana_provada_vira_AVALIACAO_PROSPECTIVA(self, mundo):
        d = _card_completo(mundo)
        linha = sombra.montar(d["data"])[0]
        assert linha["temporalidade_mediana"] == temporalidade.COMPROVADA, \
            linha["evidencia_mediana"]
        assert linha["classe"] == sombra.CLASSE_PROSPECTIVA, \
            linha["motivos_da_classe"]
        assert linha["ref_mediana"] == d["ref_med"]
        assert linha["mediana_lida_em"]

    def test_sem_a_prova_da_mediana_continua_PARCIAL(self, mundo):
        d = _card_completo(mundo, com_mediana=False)
        linha = sombra.montar(d["data"])[0]
        assert linha["temporalidade_mediana"] == temporalidade.NAO_CERTIFICADA
        assert linha["classe"] == sombra.CLASSE_PARCIAL
        assert "NAO e plenamente auditavel" in linha["motivos_da_classe"]

    def test_mediana_adulterada_bloqueia_a_qualificacao(self, mundo):
        d = _card_completo(mundo)
        linhas = pd.read_csv(mundo["med"], dtype=str).fillna("")
        linhas.loc[0, "odd_a"] = "9.99"
        linhas.to_csv(mundo["med"], index=False)
        linha = sombra.montar(d["data"])[0]
        assert linha["temporalidade_mediana"] == temporalidade.INCOMPATIVEL
        assert linha["classe"] != sombra.CLASSE_PROSPECTIVA

    def test_decisao_pos_inicio_bloqueia_o_fluxo_inteiro(self, mundo):
        d = _card_completo(mundo, inicio_h=1)
        tarde = datetime.now(timezone.utc) + timedelta(hours=3)
        linha = sombra.montar(d["data"], agora=tarde)[0]
        assert linha["classe"] == sombra.CLASSE_NAO_QUALIFICADA
        assert "inicio programado" in linha["motivos_da_classe"]

    def test_inverter_A_e_B_preserva_atleta_e_referencia_economica(self, mundo):
        """O mesmo atleta, a mesma odd, a mesma decisao -- com as colunas
        trocadas e a prova da mediana invertida junto."""
        d = _card_completo(mundo, odd_exec_a=4.50)
        direto = sombra.montar(d["data"])[0]
        d = _card_completo(mundo, odd_exec_a=4.50, inverter=True)
        invertido = sombra.montar(d["data"])[0]
        assert direto["classe"] == invertido["classe"] == sombra.CLASSE_PROSPECTIVA
        assert direto["lado_selecionado"] == invertido["lado_selecionado"]
        assert direto["selecao"] is True and invertido["selecao"] is True
        assert (direto["odd_lado_selecionado"]
                == invertido["odd_lado_selecionado"])
        assert (direto["retorno_bruto_esperado"]
                == pytest.approx(invertido["retorno_bruto_esperado"], abs=1e-4))

    def test_a_mediana_nao_vira_preco_executavel(self, mundo):
        """O retorno simulado continua usando a cotacao comprovada do lado
        escolhido, nunca a mediana."""
        d = _card_completo(mundo, odd_exec_a=4.50)
        linha = sombra.montar(d["data"])[0]
        assert linha["selecao"] is True
        assert linha["odd_lado_selecionado"] == 4.50
        assert linha["odd_lado_selecionado"] != d["med_a"]

    def test_captura_repetida_nao_duplica_unidades(self, mundo):
        d = _card_completo(mundo, odd_exec_a=4.50, vencedor="Alpha")
        agora = datetime.now(timezone.utc)
        for k in range(3):
            sombra.registrar(
                sombra.montar(d["data"], agora=agora + timedelta(minutes=k)),
                mundo["reg"])
        sombra.liquidar(d["data"], arquivo=mundo["res"],
                        registros=mundo["reg"])
        res = pd.read_csv(mundo["res"])
        assert len(res) == 3
        assert abs(float(res["unidades"].sum()) - 1.0) < 1e-9

    def test_card_antigo_sem_a_coluna_continua_legivel(self, mundo):
        """Registro antigo nao ganha vinculo retroativo: ele e lido, fica sem
        certificacao, e ninguem sai procurando uma captura posterior para
        preencher o buraco."""
        d = _card_completo(mundo)
        card = pd.read_csv(mundo["card"])
        card = card.drop(columns=["ref_mediana"])
        card.to_csv(mundo["card"], index=False)
        linha = sombra.montar(d["data"])[0]
        assert linha["temporalidade_mediana"] == temporalidade.NAO_CERTIFICADA
        assert linha["classe"] == sombra.CLASSE_PARCIAL
