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
| D1 | histórico truncado lido sem erro e regravado | alta | A | novo | `c2_interrupcao` C2a, `c3_truncado` C3a | — | ufc-implementacao | `claude/chaos-fase-a` @ `9ee6765` (PR #11, rascunho) | **integrado** em `main` (`4f1c4db`, 22/11) — Linux (sessão) + Windows (informado pelo operador) |
| D2 | escritores concorrentes perdem atualização | alta | A | novo | `c4_concorrencia` C4a–c | D1 | ufc-implementacao | `claude/chaos-fase-a` @ `9ee6765` (PR #11, rascunho) | **integrado** em `main` (`4f1c4db`, 22/11) — Linux (sessão) + Windows (informado pelo operador) |
| D9 | anexação sobre linha incompleta; ponteiro para versão ilegível | média | A | novo | `c3_truncado` C3b | D1 | ufc-implementacao | `claude/chaos-fase-a` @ `9ee6765` (PR #11, rascunho) | **integrado** em `main` (`4f1c4db`, 22/11) — Linux (sessão) + Windows (informado pelo operador) |
| D3 | modelo ausente troca previsão aberta por linha vazia | alta | B | novo | `c8_ui` C8a | Fase A | ufc-implementacao | `claude/chaos-fase-b` @ `b75cd25` (PR #12, rascunho) | **integrado** em `main` (`4f1c4db`, 22/11) — Linux (sessão) + Windows (informado pelo operador) |
| D4 | título do card diferente duplica a luta e o placar | alta | B | novo | `c5_c6_fechamento_confronto` C6b | Fase A | ufc-implementacao | `claude/chaos-fase-b` @ `b75cd25` (PR #12, rascunho) | **integrado** em `main` (`4f1c4db`, 22/11) — Linux (sessão) + Windows (informado pelo operador) |
| D7 | somente-render mistura probabilidade registrada com preço atual | média | B | novo | `c8_ui` C8b | Fase A | ufc-implementacao | `claude/chaos-fase-b` @ `b75cd25` (PR #12, rascunho) | **integrado** em `main` (`4f1c4db`, 22/11) — Linux (sessão) + Windows (informado pelo operador) |
| D5 | captura para após meia-noite local; alarme "perdido" falso | alta | C | **conhecido** (limitação da captura após meia-noite), reconfirmado com dados reais de 07 e 14/11 | `c7_tempo` C7a | Fase A | ufc-implementacao | `claude/chaos-fase-c` @ `d0af598` (PR #13, rascunho) | **integrado** em `main` (`4f1c4db`, 22/11) — Linux (sessão) + Windows (informado pelo operador) |
| D6 | preço ausente na resposta derruba captura e pré-registro | média | C | novo | `c1_api` C1c | Fase A | ufc-implementacao | `claude/chaos-fase-c` @ `d0af598` (PR #13, rascunho) | **integrado** em `main` (`4f1c4db`, 22/11) — Linux (sessão) + Windows (informado pelo operador) |
| D8 | linha incompleta do card some do pré-registro | média | C | novo | `c3_truncado` C3c | Fase A | ufc-implementacao | `claude/chaos-fase-c` @ `d0af598` (PR #13, rascunho) | **integrado** em `main` (`4f1c4db`, 22/11) — Linux (sessão) + Windows (informado pelo operador) |
| D10 | card da Ásia/Austrália: luta com início no dia UTC anterior à `event_date` não casava com a API (`casamento._na_janela` só aceitava 0..+1 dia) | média | D10 | **novo** (reteste da Fase C), pré-existente em `2f73748` | `d5_utc`, `d5_sp`, `d10_reteste` | Fase C | ufc-implementacao | `claude/chaos-d10` @ `4c74a4d` (PR #14, rascunho) | **integrado** em `main` (`4f1c4db`, 22/11) — Linux (sessão) + Windows (informado pelo operador) |

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

## Fase B — significado das previsões (D3, D4, D7)
| item | registro |
|---|---|
| commits | `cbb9445` (código e testes), `fce91aa` (FASE_B.md), `1deef22` (script C8b sem colisão; mesmo esperado), `bb82a86` (testes da guarda D3), `b75cd25` (`docs/index.html` por somente-render) |
| identidade (D4) | mesma `event_date` + par não ordenado; `fight_uid`/`evento_id_api`/`event_uid` divergentes não casam; 1 candidato → atualiza e troca o título; >1 → `LutaAmbigua` antes de gravar |
| diagnóstico real | somente leitura: 0 pares repetidos por data, 0 datas com dois títulos; nada migrado ou deduplicado |
| testes | `tests/test_fase_b.py` 21 (15 falham em `9ee6765`; os 2 da guarda D3 falham com a guarda neutralizada); testes antigos: só fixtures |
| reprodução | C8a, C8b, C6b, C6b' falham em `9ee6765` e passam; sem regressão da Fase A |
| reteste `ufc-chaos` | sem defeito novo (D3 parcial, D4 renomeado com inversão e após fechamento, liquidação 2×, `LutaAmbigua` sem gravar, D7 com inversão / encerrada / sem casa) |
| revisão `ufc-testes-qualidade` | 1 lacuna (guarda D3 sem teste), corrigida em `bb82a86` |
| `docs/index.html` | regerado só na branch por somente-render; 11 lutas = versões apontadas (0 divergências); `data/` da worktree com sha256 igual antes e depois |
| suíte | 992 passaram, **3 falharam (dados ausentes)**: não está verde |
| Windows | pendente (roteiro no #10) |
| fora de escopo | painel de proveniência lê o CSV; `fonte_card` do desenho vem do CSV; `n_events` conta por título e data |
| PR | [#12](https://github.com/bergols/ufc-predictor/pull/12) rascunho, base #11 |

## Fase C — captura e entradas incompletas (D5, D6, D8)
| item | registro |
|---|---|
| commits | `e763381` (D5, D6, D8), `c5edc2e` (ajustes da revisão), `ae9308a` (D8b': card malformado → `ValueError` com a linha), `038e6ec` e `d0af598` (página por somente-render, 0 divergências), `45107d1`/`2d7db02` (reproduções: asserções de propriedade e runner) |
| janela D5 | `event_date` 00:00 UTC − 1 dia até `event_date` 00:00 UTC + 1 dia + 10 h; corte por luta (`commence_time` futuro; sem fuso/ilegível bloqueia a luta); "perdido" só após a janela; custo medido +9 chamadas/evento (≤ 10) |
| testes | `tests/test_fase_c.py` 37 (19/26 falham em `b75cd25`; 10/11 dos ajustes falham em `e763381`) |
| reprodução | C7a, C1c, C3c falham em `b75cd25` e passam; sem regressão de A/B (C2a/b/c: esperado antigo) |
| reteste `ufc-chaos` | EUA hora a hora em BRT e UTC: sem gravação pós-início, sem buraco, < 1 h antes de cada luta, alarmes coerentes; D6 e D8 extras ok. Achou D10 (pré-existente) |
| revisão `ufc-integridade` | 5 apontamentos: 4 corrigidos em `c5edc2e`; resíduo: `ciclo_semanal` grava o template antes de o D8 recusar a publicação (idempotente, sem perda) |
| suíte | final `d0af598`: 1030 passaram, **3 falharam (dados ausentes)**; `test_report_freshness` passa após a página regerada |
| Windows | pendente (janela testada só com TZ simulado) |
| PR | [#13](https://github.com/bergols/ufc-predictor/pull/13) rascunho, base #12 |

## D10 — casamento com data UTC diferente da nominal
| item | registro |
|---|---|
| commits | `a0aabe2` (correção + `D10.md` + testes), `ac06d4a` (fixture `fuso` sem `tzset`, para Windows), `e14bc76` (merge da Fase C), `3487623` (`D10.md` sem afirmações não verificáveis), `4c74a4d` (reteste preservado) |
| significado das datas | `docs/chaos/D10.md`: `event_date` = data nominal sem fuso (comporta-se como local; local/fuso do evento não estão no clone); `commence_time` = UTC; diferença física em [−1, +1] dia |
| regra | janela simétrica [−1, +1] dia em UTC, `commence_time` normalizado com offset; sem fuso/ilegível não assume UTC (`INICIO_ILEGIVEL`); participantes dos dois lados, candidato único (`MULTIPLOS`), homônimo/aproximado e revanche preservados |
| testes | `tests/test_d10.py` 33 (17 falham antes) |
| reteste `ufc-chaos` | 149 verificações pelo caminho real (casar, captura, pré-registro): 149 ok em `4c74a4d`; 23 DEFEITO + 1 ERRO em `038e6ec` |
| revisão `ufc-integridade` | sem defeito alto/médio; imprecisão do `D10.md` corrigida; resíduos baixos: motivo impreciso em casos raros, superbloqueio seguro (`MULTIPLOS`) por evento fuzzy/ilegível, `event_watch` pode dividir card da Ásia no alerta "a registrar" (não grava) |
| runner (topo) | 241 PASSOU, 0 DEFEITO, 0 ERRO (c1–c8, d5_*, d6, d8, d10_reteste) |
| suíte (topo) | 1063 passaram, **3 falharam (dados ausentes)**: não está verde |
| PR | [#14](https://github.com/bergols/ufc-predictor/pull/14) rascunho, base #13 — **topo da pilha** |

## Ordem de integração e validação pendente
| ordem | PR | branch | SHA |
|---|---|---|---|
| 1 | #10 | `claude/affectionate-bardeen-2i6676` | (esta branch) |
| 2 | #11 | `claude/chaos-fase-a` | `9ee6765` |
| 3 | #12 | `claude/chaos-fase-b` | `b75cd25` |
| 4 | #13 | `claude/chaos-fase-c` | `d0af598` |
| 5 | #14 | `claude/chaos-d10` | `4c74a4d3a3ed09fcd07a045686ebc80ad562a6f0` (topo) |

**Windows: validado — informado pelo operador (22/11), não conferido pela sessão.**
Execução real no Windows 11, Python 3.13.14, sobre `4c74a4d3a3ed09fcd07a045686ebc80ad562a6f0`: suíte completa **1069 passed, 12 skipped (justificados), código 0**; `msvcrt` real, concorrência entre processos, captura em BRT e os 3 testes com dados **passaram**. Logs na máquina do operador, na pasta temporária do usuário (`%TEMP%`), em `ufc-validacao-20261121-224218` — nome como informado pelo operador. A sessão não alcança essa máquina (só um macOS está acessível), então os logs **não foram lidos nem conferidos** aqui, e o horário da execução não foi verificado. Conferido pela sessão: o SHA validado é o topo atual da pilha (nenhuma mudança de código desde então) e o total 1069 + 12 = 1081 coincide com o total do Linux (1063 + 15 + 3), com os 3 testes de dados passando e 3 testes que só rodam no Windows executados. Qual versão do roteiro rodou não é verificável daqui.

Correção documental (22/11): a versão anterior deste registro trazia o caminho completo da pasta, com o nome de usuário da máquina operacional (removido por ser dado pessoal), e convertia o horário do nome da pasta para BRT/UTC — isso era inferência da sessão, não dado conferido, e foi retirado. A aceitação desta validação, informada pelo operador, foi registrada pelo proprietário na autorização de integração.

Registro anterior, mantido: *Windows: pendente — validação final NÃO executada (22/11) pela sessão.* A sessão só alcança um macOS (pastas "claude projetos" e "sitio-silva"); nenhum Windows acessível. Nada foi aprovado, nenhum dublê substituiu a execução. Roteiro pronto (`9b8288c`): confere o código de saída de toda execução do pytest, lê o status de cada teste na saída `-v` (skip nunca aprova; só o `[UTC]` da janela do D5 é dispensado no Windows), exige os 3 testes com dados como PASSED, reprova suíte completa com código ≠ 0 e confere sha256 de `data/` da árvore operacional antes/depois. Comando na máquina operacional:
`powershell -ExecutionPolicy Bypass -File docs\chaos\reproducoes\validar_windows.ps1 -Python <.venv\Scripts\python.exe> -DadosDe <árvore operacional> -SuiteCompleta`.
`reproducoes/validar_windows.ps1` está fixado no SHA do topo e exige PASSED (pulado = falha) em: `msvcrt` real, trava entre processos/morte/threads, escrita atômica, concorrência entre processos e `TestD5JanelaDeCaptura[America/Sao_Paulo]` com a máquina em −03:00. Sem Windows acessível à sessão, nada foi aprovado.

**Dados ausentes: pendente.** O mesmo roteiro com `-DadosDe <árvore operacional>` copia `data/raw/fighters.csv`, `data/raw/fights.csv` e `data/processed/fight_features.csv` (só leitura na origem) para o clone e roda os 3 testes.

**Alcance da Fase A (lembrete):** protege falha durante a escrita protegida e recusa arquivo com linha parcial/campos ou cabeçalho divergentes; **não** detecta arquivo cortado exatamente em fronteira de linha nem protege contra escritor que não usa a trava.

## Integração (22/11)
Autorizada pelo proprietário: merge commit, ordem #10 → #11 → #12 → #13 → #14, base de cada PR ajustada para `main` antes do merge, e publicação pelo GitHub Pages.

| PR | head integrado | merge em `main` |
|---|---|---|
| #10 | `88868fb` | `320367a` |
| #11 | `9ee6765` | `d6d2915` |
| #12 | `b75cd25` | `e920d17` |
| #13 | `d0af598` | `2886b9a` |
| #14 | `4c74a4d3a3ed09fcd07a045686ebc80ad562a6f0` (validado) | `4f1c4db` |

- `src/`, `scripts/`, `config.py`, `tests/` e `requirements.txt` da `main` (`4f1c4db`) **idênticos** a `4c74a4d`; diferenças só em `docs/chaos/` (PLANO, README, achados, roteiro Windows) vindas do #10. `data/` idêntico a `2f73748`.
- Antes da publicação, `docs/chaos/`, `docs/index.html` e `.claude/agents/` foram varridos (credenciais, e-mail, nomes, caminhos de máquina, ids de sessão): uma ocorrência — o caminho dos logs Windows com o nome de usuário da máquina operacional — removida em `88868fb`. **Limite:** o texto antigo continua no histórico git (commit `9e9dca4`, agora ancestral da `main`); reescrever histórico exigiria force push, não autorizado. O Pages serve só a árvore atual.
- Pages: build do `4f1c4db` concluído com sucesso; os builds intermediários (#96–#99) foram cancelados pelo próprio Pages ao serem substituídos.
- Fora desta autorização (passo seguinte do proprietário): instalação na máquina operacional, tarefas agendadas, modelos, candidatos e registros reais.
