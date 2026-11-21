# CHAOS Fase B (D3, D4, D7)

Base 9ee6765. Testes: `tests/test_fase_b.py` (19; 15 falham na base).

- **D3.** `record_card_predictions` nunca troca linha com previsao por linha sem previsao (mantem linha e ponteiro, avisa) e ignora luta com `origem_previsao != "modelo"`. A pagina desenha a luta aberta com preditor falho a partir da versao em vigor (`para_render`), origem `snapshot`. Com o modelo funcionando nada muda.
- **D4.** Identidade = `event_date` + par nao ordenado. Ids estaveis (`fight_uid`, `evento_id_api`, `event_uid`) presentes nos dois lados e diferentes = outra luta. Um candidato compativel: atualiza a linha e o `event_name`. Mais de um: `LutaAmbigua` antes de qualquer escrita. Sem migracao nem deduplicacao.
- **D7.** Em `somente_render`, odds de mercado, odd da casa e provas vem da mesma versao da probabilidade (reorientadas se os lados estao invertidos); luta encerrada usa a linha do historico; sem mediana na versao = indisponivel com motivo.
- **Nao cobre.** O painel de proveniencia (`avaliar_com_precos_novos`) ainda le o CSV atual; `fonte_card` do desenho vem do CSV; `n_events` agrupa por (titulo, data), entao evento parcialmente liquidado e renomeado conta como dois.
