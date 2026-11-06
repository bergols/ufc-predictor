"""
`somente_render`: desenhar a partir dos snapshots, sem escrever e sem modelo.

Ate nov/2026 duas coisas estavam amarradas no mesmo caminho:

  - renderizar e ALTERAR registro (sync de resultados + gravacao do
    pre-registro), entao nao dava para regerar a pagina sem mexer no
    historico;
  - a previsao de CADA luta passava pelo preditor ANTES de olhar o snapshot,
    entao um relatorio historico inteiro dependia do modelo atual carregar.

O teste faz os preditores LANCAREM ERRO de proposito. Luta com snapshot
suficiente tem de continuar desenhada, a partir do snapshot. Luta sem
snapshot tem de aparecer como INDISPONIVEL -- nunca recalculada em silencio.
"""
import hashlib
from pathlib import Path

import pandas as pd
import pytest

import config
from src.card_report import analyze_card, generate_card_report


def _explode(*a, **k):
    raise ValueError("preditor indisponivel de proposito (teste)")


CARD = pd.DataFrame([
    {"fighter_a": "Alice Silva", "fighter_b": "Bia Costa",
     "odds_a_decimal": 1.60, "odds_b_decimal": 2.40},
    {"fighter_a": "Carla Rocha", "fighter_b": "Dora Lima",
     "odds_a_decimal": 1.90, "odds_b_decimal": 1.95},
])

FROZEN_COMPLETO = {
    ("Alice Silva", "Bia Costa"): {
        "model_prob_a": 0.6421,
        "method_probs": {"KO_TKO": 0.4, "SUBMISSION": 0.2, "DECISION": 0.4}},
    ("Carla Rocha", "Dora Lima"): {
        "model_prob_a": 0.4812,
        "method_probs": {"KO_TKO": 0.3, "SUBMISSION": 0.3, "DECISION": 0.4}},
}


class TestPreditorQuebradoComSnapshot:
    def test_luta_com_snapshot_continua_desenhada(self):
        a = analyze_card(CARD, predict_fn=_explode, method_fn=_explode,
                         frozen=FROZEN_COMPLETO)
        previstas = a["favorites"] + a["underdogs"]
        assert len(previstas) == 2, a["no_prediction"]
        assert a["no_prediction"] == []

    def test_a_probabilidade_vem_do_SNAPSHOT_e_nao_do_modelo(self):
        a = analyze_card(CARD, predict_fn=_explode, method_fn=_explode,
                         frozen=FROZEN_COMPLETO)
        por_nome = {f["fighter_a"]: f for f in a["favorites"] + a["underdogs"]}
        assert por_nome["Alice Silva"]["model_prob_a"] == pytest.approx(0.6421)
        assert por_nome["Carla Rocha"]["model_prob_a"] == pytest.approx(0.4812)

    def test_a_origem_da_previsao_fica_declarada(self):
        a = analyze_card(CARD, predict_fn=_explode, method_fn=_explode,
                         frozen=FROZEN_COMPLETO)
        assert all(f["origem_previsao"] == "snapshot"
                   for f in a["favorites"] + a["underdogs"])

    def test_com_o_modelo_vivo_a_origem_e_o_modelo(self):
        def ok(x, y):
            return {"fighter_a": x, "fighter_b": y, "prob_a_wins": 0.55,
                    "prob_b_wins": 0.45, "model_used": "logreg",
                    "fighter_a_low_experience": False,
                    "fighter_b_low_experience": False,
                    "fighter_a_debutant": False, "fighter_b_debutant": False}
        metodo = {"method_probs": {"KO_TKO": 0.4, "SUBMISSION": 0.2,
                                  "DECISION": 0.4}}
        a = analyze_card(CARD, predict_fn=ok, method_fn=lambda x, y: metodo)
        assert all(f["origem_previsao"] == "modelo"
                   for f in a["favorites"] + a["underdogs"])


class TestSemSnapshotMostraIndisponibilidade:
    def test_sem_snapshot_vai_para_sem_previsao(self):
        """Indisponibilidade explicita, nunca recalculo silencioso."""
        a = analyze_card(CARD, predict_fn=_explode, method_fn=_explode, frozen=None)
        assert a["favorites"] == [] and a["underdogs"] == []
        assert len(a["no_prediction"]) == 2
        assert all("indisponivel" in f["reason"] for f in a["no_prediction"])

    def test_snapshot_PARCIAL_nao_cobre_a_outra_luta(self):
        so_uma = {("Alice Silva", "Bia Costa"): FROZEN_COMPLETO[("Alice Silva", "Bia Costa")]}
        a = analyze_card(CARD, predict_fn=_explode, method_fn=_explode, frozen=so_uma)
        assert [f["fighter_a"] for f in a["favorites"] + a["underdogs"]] == ["Alice Silva"]
        assert [f["fighter_a"] for f in a["no_prediction"]] == ["Carla Rocha"]

    def test_snapshot_sem_probabilidade_nao_serve(self):
        """Luta fechada que entrou no historico SEM previsao continua sem."""
        vazio = {("Alice Silva", "Bia Costa"): {"model_prob_a": None,
                                                "method_probs": None}}
        a = analyze_card(CARD, predict_fn=_explode, method_fn=_explode, frozen=vazio)
        assert [f["fighter_a"] for f in a["no_prediction"]] == ["Alice Silva", "Carla Rocha"]


@pytest.mark.skipif(not (config.UPCOMING_CARD_CSV.exists()
                         and config.PREDICTION_HISTORY_CSV.exists()),
                    reason="artefatos ausentes")
class TestNaoEscreveNada:
    def _hashes(self):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (config.PREDICTION_HISTORY_CSV, config.ODDS_TEMPLATE_CSV,
                          config.UPCOMING_CARD_CSV)}

    def test_render_nao_toca_nenhum_registro(self, tmp_path):
        antes = self._hashes()
        generate_card_report(config.UPCOMING_CARD_CSV, tmp_path / "saida.html",
                             model_name=config.MODELO_PRODUCAO,
                             card_name="UFC Teste", event_date="2026-11-07",
                             photos=False, somente_render=True)
        assert self._hashes() == antes
        assert (tmp_path / "saida.html").exists()
