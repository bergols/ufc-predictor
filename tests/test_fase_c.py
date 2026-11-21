"""
CHAOS Fase C (D5, D6, D8), pelos caminhos que gravam de verdade.

  D5  janela de captura em UTC: o card de sabado ainda e candidato depois da
      meia-noite local; o corte por luta continua sendo o commence_time; o
      alarme "perdido" so aparece depois da janela;
  D6  outcome sem preco valido e ignorado e a luta fica com motivo explicito;
  D8  linha incompleta do card e denunciada antes de qualquer registro.

Nada toca producao: caminhos de `config` redirecionados para tmp_path e
relogio/API simulados (sem sleep, sem rede).
"""
import hashlib
import logging
import time
from datetime import date as _date, datetime as _dt, timedelta, timezone

import pandas as pd
import pytest

import config
from src import card_report, event_watch, line_shopping, prediction_history as ph
from src import proveniencia
from src.card_report import generate_card_report, load_card_odds

LUTAS = [("Ana Alfa", "Bia Beta"), ("Caio Gama", "Davi Delta")]
EVENTO = "2026-11-28"                                          # sabado
PRELIM = _dt(2026, 11, 29, 1, 0, tzinfo=timezone.utc)          # 22:00 BRT sab
MAIN = _dt(2026, 11, 29, 5, 0, tzinfo=timezone.utc)            # 02:00 BRT dom


def _evento(a, b, pin_a, pin_b, inicio, sem_preco=False, preco_a="padrao"):
    pa = pin_a if preco_a == "padrao" else preco_a
    out_a = {"name": a} if sem_preco else {"name": a, "price": pa}
    return {"id": f"ev-{a[:3]}-{b[:3]}".lower(), "sport_key": "mma_mixed_martial_arts",
            "commence_time": inicio.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "home_team": a, "away_team": b,
            "bookmakers": [
                {"key": "pinnacle", "title": "Pinnacle",
                 "markets": [{"key": "h2h", "outcomes": [
                     out_a, {"name": b, "price": pin_b}]}]},
                {"key": "betx", "title": "BetX",
                 "markets": [{"key": "h2h", "outcomes": [
                     {"name": a, "price": 1.95}, {"name": b, "price": 2.2}]}]}]}


def _preditor(a, b):
    p = {LUTAS[0]: 0.60, LUTAS[1]: 0.70}[(a, b)]
    return ({"fighter_a": a, "fighter_b": b, "prob_a_wins": p, "prob_b_wins": 1 - p,
             "model_used": "fake", "fighter_a_low_experience": False,
             "fighter_b_low_experience": False, "fighter_a_debutant": False,
             "fighter_b_debutant": False})


class Mundo:
    def __init__(self, tmp, mp):
        self.tmp, self.mp = tmp, mp
        self.hist = tmp / "prediction_history.csv"
        self.card = tmp / "card.csv"
        self.cotacoes = tmp / "proveniencia" / "cotacoes.csv"
        self.chamadas_api = 0
        self.resposta = []

    def card_csv(self, linhas=None):
        linhas = linhas or [
            "fighter_a,fighter_b,odds_a_decimal,odds_b_decimal",
            "Ana Alfa,Bia Beta,1.80,2.10", "Caio Gama,Davi Delta,1.50,2.70"]
        self.card.write_text("\n".join(linhas) + "\n", encoding="utf-8")

    def publicar(self, sharp=False, data=EVENTO):
        self.card_csv() if not self.card.exists() else None
        generate_card_report(self.card, self.tmp / "index.html", model_name="fake",
                             card_name="UFC Teste", event_date=data, photos=False,
                             sharp=sharp)

    def api(self, eventos):
        self.resposta = eventos

    def relogio(self, agora_utc):
        import scripts.auto_capture as ac

        class D(_date):
            @classmethod
            def today(cls):
                return agora_utc.astimezone().date()

        class T(_dt):
            @classmethod
            def now(cls, tz=None):
                return agora_utc if tz else agora_utc.astimezone().replace(tzinfo=None)
        self.mp.setattr(ac, "date", D)
        self.mp.setattr(ac, "datetime", T)
        return ac

    def capturar(self, agora_utc):
        ac = self.relogio(agora_utc)
        self.mp.setattr("sys.argv", ["auto_capture"])
        return ac.main()

    def linhas(self):
        return pd.read_csv(self.hist).set_index("fighter_a")

    def bytes_historico(self):
        return hashlib.sha256(self.hist.read_bytes()).hexdigest() if self.hist.exists() else ""


@pytest.fixture
def mundo(tmp_path, monkeypatch):
    from src import data_collection
    m = Mundo(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "PREDICTION_HISTORY_CSV", m.hist)
    monkeypatch.setattr(config, "ODDS_TEMPLATE_CSV", tmp_path / "odds_template.csv")
    monkeypatch.setattr(config, "UPCOMING_CARD_CSV", m.card)
    monkeypatch.setattr(config, "RAW_FIGHTS_CSV", tmp_path / "fights.csv")
    monkeypatch.setattr(event_watch, "CACHE_PENDENTES", tmp_path / "pendentes.json")
    monkeypatch.setattr(event_watch, "precisa_verificar", lambda hoje=None: False)
    pasta = tmp_path / "proveniencia"
    monkeypatch.setattr(proveniencia, "PASTA", pasta)
    monkeypatch.setattr(proveniencia, "INDICE", pasta / "chamadas.csv")
    monkeypatch.setattr(proveniencia, "INDICE_COTACOES", m.cotacoes)
    monkeypatch.setattr(card_report, "set_photo_map", lambda mp: None)
    monkeypatch.setattr(data_collection, "check_data_freshness", lambda: 0)
    monkeypatch.setattr(card_report, "_default_predict_fns",
                        lambda n: (_preditor, lambda a, b: {"method_probs": {
                            "KO_TKO": 0.4, "SUBMISSION": 0.2, "DECISION": 0.4}}))

    def fake_live(chave, regions=None, **kw):
        m.chamadas_api += 1
        return m.resposta
    monkeypatch.setattr(line_shopping, "fetch_live_odds", fake_live)
    monkeypatch.setattr(line_shopping, "get_api_key", lambda: "chave-falsa")
    return m


# Instantes que os testes usam (e em que o fuso local precisa valer -03:00
# quando a maquina nao tem time.tzset, ou seja, no Windows).
_INSTANTES_DO_FUSO = (_dt(2026, 11, 26, 0, 0, tzinfo=timezone.utc),
                      _dt(2026, 11, 29, 3, 30, tzinfo=timezone.utc),
                      _dt(2026, 12, 1, 0, 0, tzinfo=timezone.utc))


@pytest.fixture(params=["America/Sao_Paulo", "UTC"])
def fuso(request, monkeypatch):
    if hasattr(time, "tzset"):
        monkeypatch.setenv("TZ", request.param)
        time.tzset()
        yield request.param
        monkeypatch.undo()
        time.tzset()
        return
    # Windows: sem tzset nao ha como trocar o fuso do processo. "UTC" fica
    # para o Linux; "America/Sao_Paulo" roda com o fuso NATIVO da maquina,
    # que tem de ser -03:00 nos instantes do teste (nao se pula: um fuso
    # diferente invalidaria o que o teste afirma).
    if request.param == "UTC":
        pytest.skip("Windows sem tzset; UTC coberto no Linux")
    for inst in _INSTANTES_DO_FUSO:
        desloc = inst.astimezone().utcoffset()
        if desloc != timedelta(hours=-3):
            pytest.fail(
                f"fuso nativo da maquina e {desloc} em {inst.isoformat()}; "
                "este teste exige -03:00 (America/Sao_Paulo). Ajuste o fuso do "
                "Windows para (UTC-03:00) Brasilia e rode de novo.")
    yield request.param


# ---------------------------------------------------------------- D5

class TestD5JanelaDeCaptura:
    def _preparar(self, mundo):
        mundo.publicar(sharp=False)
        mundo.api([_evento("Ana Alfa", "Bia Beta", 1.70, 2.25, MAIN),
                   _evento("Caio Gama", "Davi Delta", 1.45, 2.95, PRELIM)])

    def test_apos_meia_noite_local_captura_a_principal_e_bloqueia_a_prelim(
            self, mundo, fuso, caplog):
        self._preparar(mundo)
        caplog.set_level(logging.INFO)
        mundo.capturar(_dt(2026, 11, 29, 3, 30, tzinfo=timezone.utc))  # 00:30 BRT
        h = mundo.linhas()
        assert pd.notna(h.loc["Ana Alfa", "close_prob"])
        assert pd.isna(h.loc["Caio Gama", "close_prob"])
        assert "SEM FECHAMENTO" not in caplog.text

    def test_principal_iniciada_nada_captura_e_nao_ha_alarme_perdido_ainda(
            self, mundo, fuso, caplog):
        self._preparar(mundo)
        caplog.set_level(logging.INFO)
        mundo.capturar(_dt(2026, 11, 29, 5, 30, tzinfo=timezone.utc))  # 02:30 BRT
        h = mundo.linhas()
        assert h["close_prob"].isna().all()
        # ainda dentro da janela: nao e "perdido" (a decisao e da regra por luta)
        assert "SEM FECHAMENTO" not in caplog.text

    def test_perdido_aparece_depois_da_janela_e_sem_chamada_de_api(
            self, mundo, fuso, caplog):
        self._preparar(mundo)
        caplog.set_level(logging.INFO)
        fim = _dt(2026, 11, 29, 10, 0, tzinfo=timezone.utc)
        mundo.capturar(fim + timedelta(minutes=1))
        assert "SEM FECHAMENTO" in caplog.text
        assert mundo.chamadas_api == 0           # custo: nenhuma chamada extra
        assert mundo.linhas()["close_prob"].isna().all()

    def test_custo_antes_da_janela_nenhuma_chamada(self, mundo, fuso):
        self._preparar(mundo)
        mundo.capturar(_dt(2026, 11, 26, 23, 59, tzinfo=timezone.utc))
        assert mundo.chamadas_api == 0

    def test_ultima_hora_da_janela_ainda_consulta(self, mundo, fuso):
        self._preparar(mundo)
        mundo.capturar(_dt(2026, 11, 29, 9, 59, tzinfo=timezone.utc))
        assert mundo.chamadas_api == 1

    def test_inicio_ilegivel_nao_captura_na_janela_nova(self, mundo, fuso):
        self._preparar(mundo)
        ev = _evento("Ana Alfa", "Bia Beta", 1.70, 2.25, MAIN)
        ev["commence_time"] = "???"
        mundo.api([ev])
        mundo.capturar(_dt(2026, 11, 29, 3, 30, tzinfo=timezone.utc))
        assert mundo.linhas()["close_prob"].isna().all()

    def test_historico_alarme_iminente_x_perdido(self, mundo):
        mundo.publicar(sharp=False)
        h = ph.load_history()
        dentro = ph.events_missing_closing(
            h, agora=_dt(2026, 11, 29, 9, 59, tzinfo=timezone.utc))
        fora = ph.events_missing_closing(
            h, agora=_dt(2026, 11, 29, 10, 0, tzinfo=timezone.utc))
        assert [f["estado"] for f in dentro] == ["iminente"]
        assert [f["estado"] for f in fora] == ["perdido"]
        assert ph.events_missing_closing(
            h, agora=_dt(2026, 11, 26, 23, 59, tzinfo=timezone.utc)) == []


# ---------------------------------------------------------------- D6

@pytest.mark.parametrize("ruim", [None, "abc", 1.0, 0.5, float("nan"), True])
def test_collect_prices_ignora_preco_invalido(ruim):
    ev = _evento("Ana Alfa", "Bia Beta", 1.70, 2.25, MAIN, preco_a=ruim)
    precos = line_shopping.collect_prices(ev, "Ana Alfa")
    assert "Pinnacle" not in precos and precos == {"BetX": 1.95}


class TestD6PrecoAusente:
    def _eventos(self):
        return [_evento("Ana Alfa", "Bia Beta", 1.70, 2.25, MAIN, sem_preco=True),
                _evento("Caio Gama", "Davi Delta", 1.45, 2.95, MAIN)]

    def test_auto_capture_nao_estoura_captura_a_integra_e_nao_grava_cotacao(self, mundo):
        mundo.publicar(sharp=False)
        mundo.api(self._eventos())
        mundo.capturar(_dt(2026, 11, 29, 3, 30, tzinfo=timezone.utc))
        h = mundo.linhas()
        assert pd.notna(h.loc["Caio Gama", "close_prob"])
        assert pd.isna(h.loc["Ana Alfa", "close_prob"])
        assert pd.isna(h.loc["Ana Alfa", "close_best_odd"])
        cot = pd.read_csv(mundo.cotacoes, dtype=str)
        assert not ((cot["participante_a"] == "Ana Alfa")).any()
        assert (cot["participante_a"] == "Caio Gama").any()

    def test_motivo_explicito_no_retorno(self, mundo):
        fights = [{"fighter_a": a, "fighter_b": b, "model_side": a} for a, b in LUTAS]
        out = line_shopping.fetch_sharp_probs(fights, events=self._eventos(),
                                              event_date=EVENTO)
        ana = out[LUTAS[0]]
        assert ana["sharp_prob"] is None
        assert "preco ausente/invalido" in ana["bloqueio"]
        assert out[LUTAS[1]]["sharp_prob"] is not None

    def test_pre_registro_com_sharp_nao_estoura(self, mundo):
        mundo.api(self._eventos())
        mundo.publicar(sharp=True)
        h = mundo.linhas()
        assert pd.isna(h.loc["Ana Alfa", "sharp_prob"])
        assert "preco ausente/invalido" in h.loc["Ana Alfa", "bloqueio_cotacao"]
        assert pd.notna(h.loc["Caio Gama", "sharp_prob"])


# ---------------------------------------------------------------- D8

class TestD8LinhaIncompleta:
    def test_card_truncado_levanta_com_a_linha_e_nada_e_gravado(self, mundo):
        mundo.publicar(sharp=False)            # registro previo, para comparar bytes
        antes = mundo.bytes_historico()
        snaps = mundo.hist.with_name("previsoes_snapshots.csv")
        snaps_antes = snaps.read_bytes() if snaps.exists() else b""
        mundo.card_csv(["fighter_a,fighter_b,odds_a_decimal,odds_b_decimal",
                        "Ana Alfa,Bia Beta,1.80,2.10", "Caio Gama,Davi Delta,1.5"])
        with pytest.raises(ValueError) as exc:
            generate_card_report(mundo.card, mundo.tmp / "i.html", model_name="fake",
                                 card_name="UFC Teste", event_date=EVENTO,
                                 photos=False, sharp=False)
        assert "linha 3" in str(exc.value) and "Caio Gama" in str(exc.value)
        assert mundo.bytes_historico() == antes
        assert (snaps.read_bytes() if snaps.exists() else b"") == snaps_antes

    def test_odd_em_texto_e_denunciada(self, mundo):
        mundo.card_csv(["fighter_a,fighter_b,odds_a_decimal,odds_b_decimal",
                        "Ana Alfa,Bia Beta,1.80,2.10", "Caio Gama,Davi Delta,abc,2.7"])
        with pytest.raises(ValueError, match="linha 3"):
            load_card_odds(mundo.card)

    def test_lutador_ausente_e_denunciado(self, mundo):
        mundo.card_csv(["fighter_a,fighter_b,odds_a_decimal,odds_b_decimal",
                        ",Bia Beta,1.80,2.10"])
        with pytest.raises(ValueError, match="linha 2"):
            load_card_odds(mundo.card)

    def test_card_normal_e_linhas_vazias_seguem_iguais(self, mundo):
        mundo.card_csv(["fighter_a,fighter_b,odds_a_decimal,odds_b_decimal,odds_a_tp",
                        "Ana Alfa,Bia Beta,1.80,2.10,",
                        "", ",,,,",
                        "Caio Gama,Davi Delta,1.50,2.70,1.6"])
        df = load_card_odds(mundo.card)
        assert list(df["fighter_a"]) == ["Ana Alfa", "Caio Gama"]
        assert list(df["odds_a_decimal"]) == [1.8, 1.5]
        assert "_linha_arquivo" not in df.columns


# ---------------------------------------------------------------- ajustes

class TestAjustes:
    @pytest.mark.parametrize("inicio", ["2026-11-29T05:00:00", "2026-11-29"])
    def test_inicio_sem_fuso_bloqueia_so_essa_luta_sem_estourar(self, mundo, inicio, caplog):
        mundo.publicar(sharp=False)
        ana = _evento("Ana Alfa", "Bia Beta", 1.70, 2.25, MAIN)
        ana["commence_time"] = inicio
        mundo.api([ana, _evento("Caio Gama", "Davi Delta", 1.45, 2.95, MAIN)])
        caplog.set_level(logging.INFO)
        mundo.capturar(_dt(2026, 11, 29, 3, 30, tzinfo=timezone.utc))
        h = mundo.linhas()
        assert pd.isna(h.loc["Ana Alfa", "close_prob"])
        assert pd.notna(h.loc["Caio Gama", "close_prob"])
        assert "sem fuso" in caplog.text

    @pytest.mark.parametrize("ruim", [None, "abc", "1.9", 1.0, float("inf")])
    def test_mediana_ignora_preco_invalido(self, ruim):
        from src import mediana
        ev = {"bookmakers": [
            {"key": "x", "markets": [{"key": "h2h", "outcomes": [
                {"name": "A", "price": ruim}, {"name": "B", "price": 2.0}]}]},
            {"key": "y", "markets": [{"key": "h2h", "outcomes": [
                {"name": "A", "price": 1.8}, {"name": "B", "price": 2.2}]}]},
            {"key": "z", "markets": [{"key": "h2h", "outcomes": [
                {"name": "A", "price": 1.6}, {"name": "B", "price": 2.4}]}]}]}
        pa, pb, casas = mediana.calcular(ev, "A", "B")
        assert casas == ["y", "z"] and pa == 1.7 and pb == 2.3

    @pytest.mark.parametrize("odd", ["inf", "nan", "-inf"])
    def test_card_com_odd_infinita_ou_nan_e_denunciado(self, mundo, odd):
        mundo.card_csv(["fighter_a,fighter_b,odds_a_decimal,odds_b_decimal",
                        "Ana Alfa,Bia Beta,1.80,2.10", f"Caio Gama,Davi Delta,{odd},2.7"])
        with pytest.raises(ValueError, match="linha 3"):
            load_card_odds(mundo.card)

    def test_motivo_so_e_preco_invalido_quando_e_da_pinnacle(self, mundo):
        fights = [{"fighter_a": "Ana Alfa", "fighter_b": "Bia Beta",
                   "model_side": "Ana Alfa"}]
        # preco invalido so na BetX; Pinnacle simplesmente nao cobre
        ev = _evento("Ana Alfa", "Bia Beta", 1.70, 2.25, MAIN)
        ev["bookmakers"] = [b for b in ev["bookmakers"] if b["key"] != "pinnacle"]
        ev["bookmakers"][0]["markets"][0]["outcomes"][0]["price"] = None
        ev["bookmakers"][0]["markets"][0]["outcomes"][1]["price"] = 2.2
        out = line_shopping.fetch_sharp_probs(fights, events=[ev], event_date=EVENTO)
        assert out[LUTAS[0]]["bloqueio"] == "casa sharp nao cobre o confronto"


def test_odd_com_virgula_decimal_sem_aspas_e_denunciada_e_nada_grava(mundo):
    mundo.publicar(sharp=False)
    antes = mundo.bytes_historico()
    snaps = mundo.hist.with_name("previsoes_snapshots.csv")
    snaps_antes = snaps.read_bytes() if snaps.exists() else b""
    mundo.card_csv(["fighter_a,fighter_b,odds_a_decimal,odds_b_decimal",
                    "Ana Alfa,Bia Beta,1.80,2.10", "Eva X,Fia Y,1.5,2.5",
                    "Caio Gama,Davi Delta,1,85,2.70"])
    with pytest.raises(ValueError, match="linha 4 ") as exc:
        generate_card_report(mundo.card, mundo.tmp / "i.html", model_name="fake",
                             card_name="UFC Teste", event_date=EVENTO,
                             photos=False, sharp=False)
    assert "esperados 4" in str(exc.value) and "encontrados 5" in str(exc.value)
    assert exc.value.__cause__ is not None
    assert mundo.bytes_historico() == antes
    assert (snaps.read_bytes() if snaps.exists() else b"") == snaps_antes
