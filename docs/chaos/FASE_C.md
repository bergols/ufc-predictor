# CHAOS Fase C (D5, D6, D8)

- D5: janela de captura em UTC, `prediction_history.janela_captura`: de
  `event_date 00:00 UTC - 1 dia` ate `event_date 00:00 UTC + 1 dia + 10 h`.
  Dentro dela o card e candidato; o corte final segue a regra por luta
  (commence_time futuro). `events_missing_closing(agora=...)` usa a mesma
  janela: "iminente" dentro, "perdido" depois. Custo extra: no maximo 10
  chamadas de API (rodadas horarias) por evento sem resultado.
- D6: `collect_prices` ignora outcome sem preco numerico > 1 (log); a luta sai
  com bloqueio "preco ausente/invalido na resposta" e sem cotacao registrada.
- D8: `load_card_odds` levanta ValueError com linha do arquivo e lutadores.
- Testes: `tests/test_fase_c.py`.
