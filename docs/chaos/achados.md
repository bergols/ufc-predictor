# Achados CHAOS — controle central

Fonte: rodada 1 (`RELATORIO_RODADA1.md`), auditada em `2f73748`.
Só o coordenador edita este arquivo. Estados:
`reproduzido → em correção → em revisão → pronto para integrar → integrado`
("integrado" só depois do merge em `main`).

## Gravidade
| nível | critério |
|---|---|
| alta | perda/corrupção de registro, ou indisponibilidade, possível com ação plausível |
| média | dado gravado/exibido errado em condição plausível, detectável ou corrigível |
| baixa | mensagem errada ou exibição imprecisa sem perda de dado |

## Painel
| ID | achado | grav. | fase | origem | reprodução | depende de | responsável | branch / commit | estado |
|---|---|---|---|---|---|---|---|---|---|
| D1 | histórico truncado lido sem erro e regravado | alta | A | novo | `c2_interrupcao` C2a, `c3_truncado` C3a | — | ufc-implementacao | `claude/chaos-fase-a` @ `9ee6765` (PR #11, rascunho) | **validada no Linux; pendente no Windows** (PR #11 rascunho) |
| D2 | escritores concorrentes perdem atualização | alta | A | novo | `c4_concorrencia` C4a–c | D1 | ufc-implementacao | `claude/chaos-fase-a` @ `9ee6765` (PR #11, rascunho) | **validada no Linux; pendente no Windows** (PR #11 rascunho) |
| D9 | anexação sobre linha incompleta; ponteiro para versão ilegível | média | A | novo | `c3_truncado` C3b | D1 | ufc-implementacao | `claude/chaos-fase-a` @ `9ee6765` (PR #11, rascunho) | **validada no Linux; pendente no Windows** (PR #11 rascunho) |
| D3 | modelo ausente troca previsão aberta por linha vazia | alta | B | novo | `c8_ui` C8a | Fase A | — | — | reproduzido |
| D4 | título do card diferente duplica a luta e o placar | alta | B | novo | `c5_c6_fechamento_confronto` C6b | Fase A | — | — | reproduzido |
| D7 | somente-render mistura probabilidade registrada com preço atual | média | B | novo | `c8_ui` C8b | Fase A | — | — | reproduzido |
| D5 | captura para após meia-noite local; alarme "perdido" falso | alta | C | **conhecido** (limitação da captura após meia-noite), reconfirmado com dados reais de 07 e 14/11 | `c7_tempo` C7a | Fase A | — | — | reproduzido |
| D6 | preço ausente na resposta derruba captura e pré-registro | média | C | novo | `c1_api` C1c | Fase A | — | — | reproduzido |
| D8 | linha incompleta do card some do pré-registro | média | C | novo | `c3_truncado` C3c | Fase A | — | — | reproduzido |

Conhecido e fora de escopo nesta rodada: 90 linhas sintéticas "Alice Silva"
em `data/proveniencia/cotacoes.csv` (incidente de 06/11, ver
`tests/conftest.py`). Não remover sem autorização.

## Critério de aceite comum
1. a reprodução indicada FALHA no código anterior e PASSA depois;
2. teste do caminho real (função que grava / fluxo), com o caso normal;
3. falha produz motivo claro; nada é descartado em silêncio;
4. registros reais com hashes iguais antes/depois;
5. reexecução não duplica resultado nem retorno;
6. suíte completa: só as 3 falhas esperadas (base `data/processed` ausente).

## Fase A — persistência (D1, D2, D9)
**Aceite específico**
- todo ciclo ler→alterar→gravar de `prediction_history.csv` sob trava entre
  processos, em todos os escritores (`prediction_history.py`,
  `snapshots.semear_do_historico`, `scripts/reconciliar_card.py`);
- anexações (`snapshots`, `proveniencia` chamadas/cotações, `mediana`,
  `decisoes`) e reescritas (`odds_template` em `fechamento.py`,
  `ciclo_semanal`, `reconciliar_card`) sob a mesma trava;
- substituição por temporário + `os.replace`; corpo `.json.gz` idem;
- leitura rejeita arquivo sem `\n` final ou com linha de campos divergentes,
  com motivo e instrução de recuperação, ANTES de qualquer gravação;
- histórico só recebe `prediction_id` de versão gravada e relida;
- C2a, C3a, C3b, C4a, C4b, C4c (≥ 30 rodadas) passam;
- teste com processos separados; Windows: teste preparado, validação pendente
  até rodar na máquina operacional.

| item | registro |
|---|---|
| commits | `286daa1` (implementação), `8ffa120` (ajustes pós-revisão), `9ee6765` (doc) |
| testes | `tests/test_persistencia.py`: 37 passam, 1 pulado fora do Windows; 44 passam no final; suíte 970 passam / 4 falhas (3 esperadas por `data/processed` + `test_report_freshness`, ver nota) |
| reprodução antes/depois | antes: C2a, C3a, C3b, C4a–c falham. Depois (reteste `ufc-chaos`): C3a, C3b, C4a–c passam; C4c 0/30. C2a/C2b/C2c falham só pelo esperado antigo dos scripts; as propriedades passam (destino intacto; corpo reparado com nova resposta e reprovado sem ela; ref órfã reprovada) |
| reteste extra | SIGKILL no `os.replace` e no dono da trava ok; reentrância ok; exclusão entre threads ok; timeout com motivo ok; caso normal idêntico à base |
| limitações confirmadas | corte exatamente em fronteira de linha não detectável; temporário órfão fica no diretório (ignorado); editores externos não usam a trava |
| alcance real | protege contra **falha durante a escrita protegida** (queda, kill, disco cheio no meio da gravação: o destino fica intacto) e **recusa** arquivo com linha parcial, campos divergentes, aspas abertas ou cabeçalho divergente. **NÃO** detecta arquivo previamente cortado exatamente em fronteira de linha (fica menor e válido) nem protege contra escritor que não use a trava (editor, planilha, git, sincronização) |
| Linux | validado (container) |
| Windows | **pendente**. Nenhum Windows disponível à sessão em 21/11 (só um macOS). Roteiro pronto: `reproducoes/validar_windows.ps1` (clone temporário em `%TEMP%`, sem credenciais; exige que `test_windows_msvcrt_exclui_entre_processos` PASSE, skip conta como falha) |
| revisão `ufc-integridade` | 2 médios e 3 baixos; 4 corrigidos em `8ffa120` e reconferidos (testes focados 226 ok; C3a/C3b/C4a–c ok, C4c 0/30): `sombra.py` fora da trava; `anexar_csv` sem conferência de cabeçalho; `TravaOcupada` na proveniência; `fechamento` sem template. Resíduo fora da fase: `thunderpick.py:425` (`upcoming_card_odds.csv`, entrada do card) |
| registros reais | `data/` idêntico a `2f73748` (114 arquivos); os 30 CSV reais passam em `validar_csv` (só leitura) |
| PR | [#11](https://github.com/bergols/ufc-predictor/pull/11) rascunho, base #10 |

Falhas da suíte por dados ausentes (não são verdes; não foram enfraquecidas):
| teste | arquivo ausente no clone (fora do git) |
|---|---|
| `test_casamento::TestHomonimo::test_a_lista_de_homonimos_confere_com_a_base` | `data/raw/fighters.csv` |
| `test_identidade::TestEventUid::test_linha_sem_event_uid_fica_pendente_e_nao_presumida` | `data/raw/fights.csv` |
| `test_train_method::TestSymmetricSums::test_somas_identicas_nas_linhas_espelhadas` | `data/processed/fight_features.csv` |
Validação: `validar_windows.ps1 -DadosDe <árvore operacional>` copia esses 3 arquivos (só leitura na origem) para o clone temporário e roda os 3 testes. **Pendente.**

Nota `test_report_freshness`: o carimbo do gerador inclui `prediction_history.py`, então `docs/index.html` fica "desatualizada". Resolver exige regerar a página, o que é publicação e não está autorizado. **Decisão do proprietário**; o teste não será afrouxado.

Nota `chamadas.csv`: linhas antigas com `corpo_ref` em barra invertida (Windows) dão "corpo ausente" ao conferir no Linux. É anterior à Fase A e está fora de escopo.
