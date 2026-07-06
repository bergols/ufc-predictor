"""
Testes da verificacao de frescor (check_data_freshness) e da mesclagem de
lutas recentes digitadas a mao (merge_manual_recent_fights).

Contexto: as fontes automaticas gratuitas podem estagnar silenciosamente
(aconteceu com o espelho GitHub em mai/2026). O pipeline precisa (a)
reclamar sozinho quando os dados estao velhos e (b) aceitar preenchimento
manual dos eventos faltantes sem quebrar o formato canonico.
"""
import logging

import numpy as np
import pandas as pd
import pytest

import config
from src import data_collection as dc


def _write_fights_csv(path, dates, winners=None):
    n = len(dates)
    winners = winners if winners is not None else [f"F{i}a" for i in range(n)]
    pd.DataFrame({
        "event_name": [f"Evento {i}" for i in range(n)],
        "event_date": pd.to_datetime(dates),
        "event_url": [f"http://u/e/{i}" for i in range(n)],
        "fight_url": [f"http://u/f/{i}" for i in range(n)],
        "fighter_1": [f"F{i}a" for i in range(n)],
        "fighter_2": [f"F{i}b" for i in range(n)],
        "winner": winners,
        "weight_class": ["Lightweight Bout"] * n,
        "method": ["Decision - Unanimous"] * n,
        "round": [3] * n,
        "time": ["5:00"] * n,
    }).to_csv(path, index=False)


@pytest.fixture
def raw_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_FIGHTS_CSV", tmp_path / "fights.csv")
    monkeypatch.setattr(config, "PUBLIC_DATASET_CSV", tmp_path / "public_dataset.csv")
    monkeypatch.setattr(dc, "MANUAL_FIGHTS_CSV", tmp_path / "manual_recent_fights.csv")
    return tmp_path


class TestFreshnessCheck:
    def test_dados_frescos_nao_avisam(self, raw_paths, caplog):
        _write_fights_csv(config.RAW_FIGHTS_CSV,
                          [pd.Timestamp.now().normalize() - pd.Timedelta(days=3)])
        with caplog.at_level(logging.INFO, logger="src.data_collection"):
            gap = dc.check_data_freshness()
        assert gap == 3
        assert "DADOS DESATUALIZADOS" not in caplog.text

    def test_dados_velhos_disparam_warning(self, raw_paths, caplog):
        _write_fights_csv(config.RAW_FIGHTS_CSV,
                          [pd.Timestamp.now().normalize() - pd.Timedelta(days=50)])
        with caplog.at_level(logging.WARNING, logger="src.data_collection"):
            gap = dc.check_data_freshness()
        assert gap == 50
        assert "DADOS DESATUALIZADOS" in caplog.text

    def test_limite_customizado(self, raw_paths, caplog):
        _write_fights_csv(config.RAW_FIGHTS_CSV,
                          [pd.Timestamp.now().normalize() - pd.Timedelta(days=5)])
        with caplog.at_level(logging.WARNING, logger="src.data_collection"):
            dc.check_data_freshness(max_gap_days=2)
        assert "DADOS DESATUALIZADOS" in caplog.text

    def test_sem_dado_nenhum_retorna_none(self, raw_paths):
        assert dc.check_data_freshness() is None


class TestMergeManual:
    def _write_manual(self, rows):
        pd.DataFrame(rows, columns=["event_name", "event_date", "fighter_1", "fighter_2",
                                     "winner", "weight_class", "method", "round", "time"]
                     ).to_csv(dc.MANUAL_FIGHTS_CSV, index=False)

    def test_adiciona_lutas_novas(self, raw_paths):
        _write_fights_csv(config.RAW_FIGHTS_CSV, ["2026-05-16"])
        self._write_manual([
            ["UFC Manual 1", "2026-06-14", "Ana Silva", "Bia Costa", "Ana Silva",
             "Flyweight Bout", "KO/TKO", 1, "2:10"],
        ])
        assert dc.merge_manual_recent_fights() == 1
        fights = pd.read_csv(config.RAW_FIGHTS_CSV, parse_dates=["event_date"])
        assert len(fights) == 2
        nova = fights.iloc[-1]
        assert nova["winner"] == "Ana Silva"
        assert str(nova["fight_url"]).startswith("manual::")

    def test_nao_duplica_luta_ja_existente(self, raw_paths):
        _write_fights_csv(config.RAW_FIGHTS_CSV, ["2026-06-14"])
        fights = pd.read_csv(config.RAW_FIGHTS_CSV)
        # mesma luta (mesma data e mesmos nomes, em qualquer ordem)
        self._write_manual([
            ["Evento 0", "2026-06-14", fights["fighter_2"].iloc[0], fights["fighter_1"].iloc[0],
             "", "Lightweight Bout", "Decision - Unanimous", 3, "5:00"],
        ])
        assert dc.merge_manual_recent_fights() == 0

    def test_winner_vazio_vira_nan(self, raw_paths):
        """Regressao da familia de bugs de empate: winner em branco no CSV
        manual deve virar NaN (luta descartada nas features), nunca string
        vazia nem um label falso."""
        _write_fights_csv(config.RAW_FIGHTS_CSV, ["2026-05-16"])
        self._write_manual([
            ["UFC Manual 2", "2026-06-20", "Cris Rocha", "Dani Lima", "",
             "Bantamweight Bout", "Overturned", 2, "3:00"],
        ])
        assert dc.merge_manual_recent_fights() == 1
        fights = pd.read_csv(config.RAW_FIGHTS_CSV)
        assert pd.isna(fights.iloc[-1]["winner"])

    def test_winner_invalido_da_erro_claro(self, raw_paths):
        _write_fights_csv(config.RAW_FIGHTS_CSV, ["2026-05-16"])
        self._write_manual([
            ["UFC Manual 3", "2026-06-20", "Cris Rocha", "Dani Lima", "Nome Errado",
             "Bantamweight Bout", "KO/TKO", 2, "3:00"],
        ])
        with pytest.raises(ValueError, match="winner"):
            dc.merge_manual_recent_fights()

    def test_arquivo_ausente_e_noop(self, raw_paths):
        _write_fights_csv(config.RAW_FIGHTS_CSV, ["2026-05-16"])
        assert dc.merge_manual_recent_fights() == 0
