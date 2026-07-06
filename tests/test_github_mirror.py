"""
Testes do adaptador do espelho GitHub (Greco1899/scrape_ufc_stats) em
src/data_collection.py::convert_github_mirror_to_canonical.

Cobrem as particularidades reais do schema de origem observadas em execucao:
  - stats POR ROUND que precisam ser agregadas em totais por luta;
  - chaves EVENT/BOUT com espacos em branco inconsistentes entre arquivos;
  - OUTCOME "W/L"/"L/W"/"D/D"/"NC/NC" -- empate/no-contest deve virar
    winner NaN (nunca string vazia nem label errado; regressao do bug de
    rotulo de empates);
  - EVENT+BOUT ambiguo (duas lutas iguais no mesmo evento) descartado das stats;
  - luta sem nenhuma stat vira NaN, nao 0 (min_count=1 no groupby.sum).
"""
import numpy as np
import pandas as pd
import pytest

import config
from src.data_collection import convert_github_mirror_to_canonical
from src.features import build_features_from_scrape


def _write_mirror_csvs(src_dir) -> None:
    # NOTE os espacos deliberados: "Event One " (results) vs "Event One"
    # (events/details/stats) -- reproduz o dado real, onde o join quebrava
    # sem strip.
    pd.DataFrame({
        "EVENT": ["Event One", "Event Two"],
        "URL": ["http://u/event/1", "http://u/event/2"],
        "DATE": ["January 01, 2020", "June 01, 2020"],
        "LOCATION": ["Las Vegas, Nevada, USA", "Newark, New Jersey, USA"],
    }).to_csv(src_dir / "ufc_event_details.csv", index=False)

    pd.DataFrame({
        "EVENT": ["Event One ", "Event One ", "Event Two ", "Event Two ", "Event Two "],
        "BOUT": ["Ana Silva vs. Bia Costa", "Cris Rocha vs. Dani Lima",
                 "Ana Silva vs. Eva Nunes", "Cris Rocha vs. Bia Costa",
                 "Fabi Melo vs. Gabi Reis"],
        "OUTCOME": ["W/L", "L/W", "D/D", "NC/NC", "W/L"],
        "WEIGHTCLASS": ["Strawweight Bout"] * 5,
        "METHOD": ["Decision - Unanimous", "KO/TKO", "Decision - Majority", "Overturned", "Submission"],
        "ROUND": [3, 2, 3, 1, 1],
        "TIME": ["5:00", "4:29", "5:00", "2:10", "3:33"],
        "TIME FORMAT": ["3 Rnd (5-5-5)"] * 5,
        "REFEREE": ["Ref A"] * 5,
        "DETAILS": [""] * 5,
        "URL": [f"http://u/fight/{i}" for i in range(1, 6)],
    }).to_csv(src_dir / "ufc_fight_results.csv", index=False)

    # details: mapeia EVENT+BOUT -> URL; inclui um par duplicado (fight 6 e 7
    # com mesmo EVENT+BOUT, como Sakuraba vs. Silveira em 1997) que deve ser
    # excluido do join de stats por ambiguidade.
    pd.DataFrame({
        "EVENT": ["Event One", "Event One", "Event Two", "Event Two", "Event Two",
                  "Event Two", "Event Two"],
        "BOUT": ["Ana Silva vs. Bia Costa", "Cris Rocha vs. Dani Lima",
                 "Ana Silva vs. Eva Nunes", "Cris Rocha vs. Bia Costa",
                 "Fabi Melo vs. Gabi Reis",
                 "Hana Cruz vs. Iva Dias", "Hana Cruz vs. Iva Dias"],
        "URL": [f"http://u/fight/{i}" for i in range(1, 8)],
    }).to_csv(src_dir / "ufc_fight_details.csv", index=False)

    def round_row(bout, rnd, fighter, sig, td, ctrl, kd=0.0):
        return {"EVENT": "Event One", "BOUT": bout, "ROUND": f"Round {rnd}",
                "FIGHTER": fighter, "KD": kd, "SIG.STR.": sig, "SIG.STR. %": "50%",
                "TOTAL STR.": sig, "TD": td, "TD %": "---", "SUB.ATT": 0.0,
                "REV.": 0.0, "CTRL": ctrl, "HEAD": sig, "BODY": "0 of 0",
                "LEG": "0 of 0", "DISTANCE": sig, "CLINCH": "0 of 0", "GROUND": "0 of 0"}

    pd.DataFrame([
        # luta 1: 2 rounds -> deve somar 10+20 / 20+30; ctrl 1:30 + '--' -> 90s
        round_row("Ana Silva vs. Bia Costa", 1, "Ana Silva", "10 of 20", "1 of 2", "1:30", kd=1.0),
        round_row("Ana Silva vs. Bia Costa", 2, "Ana Silva", "20 of 30", "0 of 1", "--"),
        round_row("Ana Silva vs. Bia Costa", 1, "Bia Costa", "5 of 15", "0 of 0", "0:00"),
        round_row("Ana Silva vs. Bia Costa", 2, "Bia Costa", "7 of 10", "2 of 3", "2:05"),
        # luta 2: stats totalmente ausentes ("--" sem "of") -> NaN, nao 0
        round_row("Cris Rocha vs. Dani Lima", 1, "Cris Rocha", "--", "--", "--"),
        round_row("Cris Rocha vs. Dani Lima", 1, "Dani Lima", "--", "--", "--"),
    ]).to_csv(src_dir / "ufc_fight_stats.csv", index=False)

    pd.DataFrame({
        "FIGHTER": ["Ana Silva", "Bia Costa", "Cris Rocha", "Dani Lima",
                    "Eva Nunes", "Fabi Melo", "Gabi Reis"],
        "HEIGHT": ['5\' 10"', '5\' 6"', "--", '5\' 7"', '5\' 5"', '5\' 4"', '5\' 8"'],
        "WEIGHT": ["115 lbs."] * 7,
        "REACH": ['72"', "--", '68"', '66"', '65"', '64"', '67"'],
        "STANCE": ["Orthodox", "Southpaw", None, "Orthodox", "Switch", "Orthodox", "Orthodox"],
        "DOB": ["Jul 13, 1990", "--", "Mar 01, 1995", "Jan 20, 1988",
                "May 05, 1992", "Feb 02, 1991", "Nov 11, 1993"],
        "URL": [f"http://u/fighter/{i}" for i in range(1, 8)],
    }).to_csv(src_dir / "ufc_fighter_tott.csv", index=False)


@pytest.fixture
def converted(tmp_path, monkeypatch):
    """Roda a conversao com CSVs sinteticos, escrevendo os canonicos em tmp."""
    src = tmp_path / "mirror"
    src.mkdir()
    _write_mirror_csvs(src)
    monkeypatch.setattr(config, "RAW_FIGHTS_CSV", tmp_path / "fights.csv")
    monkeypatch.setattr(config, "RAW_FIGHT_STATS_CSV", tmp_path / "fight_stats.csv")
    monkeypatch.setattr(config, "RAW_FIGHTERS_CSV", tmp_path / "fighters.csv")
    monkeypatch.setattr(config, "SQLITE_DB_PATH", tmp_path / "ufc.sqlite")
    convert_github_mirror_to_canonical(src_dir=src)
    return {
        "fights": pd.read_csv(tmp_path / "fights.csv", parse_dates=["event_date"]),
        "stats": pd.read_csv(tmp_path / "fight_stats.csv"),
        "fighters": pd.read_csv(tmp_path / "fighters.csv", parse_dates=["dob"]),
    }


class TestWinner:
    def test_wl_vence_o_primeiro_do_bout(self, converted):
        fights = converted["fights"]
        f1 = fights[fights["fight_url"] == "http://u/fight/1"].iloc[0]
        assert f1["winner"] == "Ana Silva"

    def test_lw_vence_o_segundo_do_bout(self, converted):
        fights = converted["fights"]
        f2 = fights[fights["fight_url"] == "http://u/fight/2"].iloc[0]
        assert f2["winner"] == "Dani Lima"

    def test_empate_e_nc_viram_nan_nunca_string_vazia(self, converted):
        """Regressao do bug de rotulo: empate/no-contest deve virar NaN no CSV
        (para ser descartado em features.py), jamais string vazia ou um dos
        nomes."""
        fights = converted["fights"]
        draw = fights[fights["fight_url"] == "http://u/fight/3"].iloc[0]   # D/D
        nc = fights[fights["fight_url"] == "http://u/fight/4"].iloc[0]     # NC/NC
        assert pd.isna(draw["winner"])
        assert pd.isna(nc["winner"])
        # nenhuma luta com winner string vazia no arquivo inteiro
        assert (fights["winner"].fillna("x").astype(str).str.strip() != "").all()

    def test_winner_sempre_e_um_dos_lutadores(self, converted):
        fights = converted["fights"].dropna(subset=["winner"])
        assert ((fights["winner"] == fights["fighter_1"])
                | (fights["winner"] == fights["fighter_2"])).all()


class TestJoinsEDatas:
    def test_espacos_nas_chaves_nao_quebram_o_join_de_eventos(self, converted):
        """EVENT em fight_results tem espaco a direita no dado real; sem strip
        o join com event_details falhava para 100% das linhas."""
        fights = converted["fights"]
        assert fights["event_date"].notna().all()
        assert fights.loc[fights["event_name"] == "Event One", "event_date"].iloc[0] == pd.Timestamp("2020-01-01")

    def test_nomes_separados_do_bout(self, converted):
        fights = converted["fights"]
        assert set(fights["fighter_1"]) == {"Ana Silva", "Cris Rocha", "Fabi Melo"}
        assert set(fights["fighter_2"]) == {"Bia Costa", "Dani Lima", "Eva Nunes", "Gabi Reis"}


class TestAgregacaoDeStats:
    def test_soma_dos_rounds(self, converted):
        stats = converted["stats"]
        ana = stats[(stats["fight_url"] == "http://u/fight/1") & (stats["fighter"] == "Ana Silva")].iloc[0]
        assert ana["sig_strikes_landed"] == 30       # 10 + 20
        assert ana["sig_strikes_attempted"] == 50    # 20 + 30
        assert ana["takedowns_landed"] == 1
        assert ana["takedowns_attempted"] == 3
        assert ana["knockdowns"] == 1
        assert ana["control_seconds"] == 90          # 1:30 + '--' (ignorado)

    def test_luta_sem_dado_vira_nan_nao_zero(self, converted):
        """min_count=1: '--' em todos os rounds deve dar NaN (dado ausente),
        nao 0 (que significaria 'tentou zero quedas')."""
        stats = converted["stats"]
        cris = stats[(stats["fight_url"] == "http://u/fight/2") & (stats["fighter"] == "Cris Rocha")].iloc[0]
        assert np.isnan(cris["sig_strikes_landed"])
        assert np.isnan(cris["takedowns_attempted"])
        assert np.isnan(cris["control_seconds"])

    def test_event_bout_ambiguo_fica_fora_das_stats(self, converted):
        stats = converted["stats"]
        assert "http://u/fight/6" not in set(stats["fight_url"])
        assert "http://u/fight/7" not in set(stats["fight_url"])


class TestFighters:
    def test_conversoes_de_bio(self, converted):
        fighters = converted["fighters"].set_index("name")
        assert fighters.loc["Ana Silva", "height_cm"] == pytest.approx(177.8)   # 5'10"
        assert fighters.loc["Ana Silva", "reach_cm"] == pytest.approx(182.9)    # 72"
        assert fighters.loc["Ana Silva", "dob"] == pd.Timestamp("1990-07-13")
        assert np.isnan(fighters.loc["Cris Rocha", "height_cm"])                # '--'
        assert pd.isna(fighters.loc["Bia Costa", "dob"])                        # '--'


class TestFimAFim:
    def test_features_a_partir_dos_arquivos_convertidos(self, converted, tmp_path, monkeypatch):
        """O formato convertido deve ser aceito sem ajuste por
        build_features_from_scrape (e empates/NC descartados la)."""
        feature_df = build_features_from_scrape()
        # 5 lutas no total, 2 sem vencedor (D/D e NC/NC) -> 3 lutas x 2 linhas
        assert feature_df["fight_id"].nunique() == 3
        assert len(feature_df) == 6
        g = feature_df.groupby("fight_id")
        assert (g["label"].sum() == 1).all()
