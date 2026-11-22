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

## Ajustes da rodada 2
- `auto_capture`: `commence_time` sem fuso (datetime ingenuo ou so data) bloqueia
  a luta ("inicio sem fuso, nao verificavel"); nao assume UTC; as demais seguem.
- `mediana.calcular`: usa `line_shopping._preco_valido`; preco invalido e
  ignorado com log e a casa sem as duas pontas validas nao entra.
- `load_card_odds`: odd `inf`/`-inf`/`nan` (texto) e ilegivel e denunciada.
- `_tem_preco_invalido`: o motivo "preco ausente/invalido na resposta" so vale
  para preco invalido da Pinnacle nos lados da luta; senao "casa sharp nao cobre".
- Decisao: `price` precisa ser numero (int/float). String numerica ("1.9") e
  rejeitada, pois a The Odds API envia numero e nao gravamos formato nao declarado.
- Nao cobre: `src/casamento.py` (fora de escopo); preco invalido de casa nao-sharp
  apenas e ignorado (a luta segue capturada pela Pinnacle).
