"""
Preenchimento automatico de vencedores (src/fechamento.py).

O ponto perigoso do ciclo inteiro: vencedor errado entra num denominador que
nunca mais e revisto. Por isso o criterio nao e "achou alguma coisa", e "nao
restou duvida" -- e o que se protege aqui e a RECUSA, nao o acerto.
"""
import pandas as pd
import pytest

import config
from src.fechamento import casar_resultado, preencher_template

DATA = "2026-11-07"


def _base(linhas):
    return pd.DataFrame([{"fighter_1": a, "fighter_2": b, "winner": v,
                          "method": m, "event_date": DATA}
                         for a, b, v, m in linhas])


class TestCasamento:
    def test_vencedor_na_NOSSA_grafia(self):
        """O sync rejeita vencedor que nao seja exatamente um dos dois lados
        da nossa linha, e a base escreve nomes diferentes dos nossos."""
        b = _base([("Alice Silva", "Bia Costa", "Alice Silva", "KO/TKO")])
        r = casar_resultado("Alice Silva", "Bia Costa", b)
        assert (r.estado, r.vencedor) == ("ok", "Alice Silva")

    def test_lados_invertidos_na_base(self):
        b = _base([("Bia Costa", "Alice Silva", "Alice Silva", "KO/TKO")])
        r = casar_resultado("Alice Silva", "Bia Costa", b)
        assert (r.estado, r.vencedor) == ("ok", "Alice Silva")

    def test_grafia_proxima_casa_e_traduz(self):
        """Variante de grafia casa e o vencedor sai na NOSSA grafia. A guarda
        de pontas do best_name_match continua valendo: 'Bea' x 'Bia' seria
        recusado, e e esse rigor que mantem o automatico seguro em nome
        curto (foi 'Cameron Nelson' x 'Cameron Else' que custou uma perna)."""
        b = _base([("Tanner Boser", "Jhonata Diniz", "Jhonata Diniz", "KO/TKO")])
        r = casar_resultado("Tanner Boser", "Jhonatha Diniz", b)
        assert (r.estado, r.vencedor) == ("ok", "Jhonatha Diniz")   # a NOSSA grafia

    def test_nome_curto_parecido_NAO_casa(self):
        b = _base([("Alice Silva", "Bia Costa", "Bia Costa", "Submission")])
        assert casar_resultado("Alice Silva", "Bea Costa", b).estado == "ausente"

    def test_EMPATE_fica_em_branco(self):
        """Kareckaite x Gatto, 10/out: decisao majoritaria sem vencedor. Sai
        dos denominadores sozinha -- inventar vencedor seria o pior erro."""
        b = _base([("Alice Silva", "Bia Costa", None, "Decision - Majority")])
        r = casar_resultado("Alice Silva", "Bia Costa", b)
        assert r.estado == "empate" and r.vencedor is None

    def test_adversario_trocado_fica_em_branco(self):
        """Melissa Amaya, 26/set: lutou contra OUTRA pessoa, e a linha como
        registrada nunca aconteceu."""
        b = _base([("Alice Silva", "Outra Pessoa", "Alice Silva", "KO/TKO")])
        r = casar_resultado("Alice Silva", "Bia Costa", b)
        assert r.estado == "ausente" and r.vencedor is None

    def test_luta_cancelada_fica_em_branco(self):
        assert casar_resultado("Alice Silva", "Bia Costa", _base([])).estado == "ausente"

    def test_par_repetido_na_base_e_ambiguo(self):
        b = _base([("Alice Silva", "Bia Costa", "Alice Silva", "KO/TKO"),
                   ("Alice Silva", "Bia Costa", "Bia Costa", "Submission")])
        assert casar_resultado("Alice Silva", "Bia Costa", b).estado == "ambigua"

    def test_vencedor_que_nao_e_nenhum_dos_lados_e_ambiguo(self):
        b = _base([("Alice Silva", "Bia Costa", "Terceira Pessoa", "KO/TKO")])
        assert casar_resultado("Alice Silva", "Bia Costa", b).estado == "ambigua"

    def test_nome_longe_demais_NAO_adivinha(self):
        """Mehemmedeli x Mahammadali Osmanli: eu preenchi a mao depois de
        conferir o adversario. A automacao se recusa, e e o lado certo de
        errar."""
        b = _base([("Mahammadali Osmanli", "Ilimbek Akylbek",
                    "Ilimbek Akylbek", "DQ")])
        assert casar_resultado("Mehemmedeli Osmanli", "Ilimbek Akylbek", b).estado \
            == "ausente"


class TestEscrita:
    def _template(self, tmp_path, winner=""):
        p = tmp_path / "odds_template.csv"
        p.write_text(
            "fight_id,event_name,event_date,fighter_a,fighter_b,"
            "odds_a_decimal,odds_b_decimal,actual_winner\n"
            f",UFC X,{DATA},Alice Silva,Bia Costa,1.540,2.500,{winner}\n"
            f",UFC X,{DATA},Carla Rocha,Dora Lima,1.800,2.100,\n",
            encoding="utf-8")
        return p

    def test_escreve_so_o_que_tem_certeza(self, tmp_path):
        b = _base([("Alice Silva", "Bia Costa", "Alice Silva", "KO/TKO")])
        n, relatos = preencher_template(DATA, self._template(tmp_path), b)
        assert n == 1
        assert [r.estado for r in relatos] == ["ok", "ausente"]

    def test_nao_reescreve_linha_ja_fechada(self, tmp_path):
        """Resultado ja preenchido a mao manda. A automacao nunca corrige
        quem ja decidiu."""
        p = self._template(tmp_path, winner="Bia Costa")
        b = _base([("Alice Silva", "Bia Costa", "Alice Silva", "KO/TKO")])
        n, relatos = preencher_template(DATA, p, b)
        assert n == 0
        assert "Bia Costa" in p.read_text(encoding="utf-8")

    def test_preserva_a_formatacao_do_arquivo(self, tmp_path):
        """Reescrever com pandas trocaria 1.540 por 1.54 no arquivo inteiro e
        afogaria o diff real num diff de formatacao."""
        p = self._template(tmp_path)
        b = _base([("Alice Silva", "Bia Costa", "Alice Silva", "KO/TKO")])
        preencher_template(DATA, p, b)
        assert "1.540,2.500" in p.read_text(encoding="utf-8")

    def test_dry_run_nao_escreve(self, tmp_path):
        p = self._template(tmp_path)
        b = _base([("Alice Silva", "Bia Costa", "Alice Silva", "KO/TKO")])
        n, _ = preencher_template(DATA, p, b, dry_run=True)
        assert n == 1 and p.read_text(encoding="utf-8").rstrip().endswith(",")


@pytest.mark.skipif(not (config.ODDS_TEMPLATE_CSV.exists() and config.RAW_FIGHTS_CSV.exists()),
                    reason="artefatos ausentes")
class TestContraOsFechamentosReais:
    """A validacao que importa: a automacao reproduz o que foi feito a mao?"""

    def test_reproduz_os_fechamentos_manuais(self):
        t = pd.read_csv(config.ODDS_TEMPLATE_CSV)
        b = pd.read_csv(config.RAW_FIGHTS_CSV)
        b["d"] = b["event_date"].astype(str).str[:10]
        divergentes, total = [], 0
        for data in ("2026-09-26", "2026-10-10", "2026-10-17", "2026-10-24"):
            do_dia = b[b["d"] == data]
            for _, r in t[t["event_date"] == data].iterrows():
                total += 1
                rel = casar_resultado(r["fighter_a"], r["fighter_b"], do_dia)
                mao = r["actual_winner"] if pd.notna(r["actual_winner"]) else None
                if mao != rel.vencedor:
                    divergentes.append(f"{data} {r['fighter_a']}: {mao!r} != {rel.vencedor!r}")
        if total == 0:
            pytest.skip("eventos de referencia ausentes do template")
        # a unica divergencia conhecida e a grafia Mehemmedeli/Mahammadali,
        # onde a automacao se RECUSA a adivinhar -- nunca o contrario
        assert len(divergentes) <= 1, divergentes
        for d in divergentes:
            assert "None" in d, f"automacao inventou vencedor: {d}"
