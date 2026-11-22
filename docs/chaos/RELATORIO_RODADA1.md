# CHAOS — UFC Predictor — rodada 1 (diagnóstico, sem correção)

Base: `bergols/ufc-predictor` em `2f7374831883c4610bbd197b015153bff5312112`, 21/11/2026.
Nenhuma correção aplicada. Escala de gravidade: ver `docs/chaos/achados.md`.

## Skill
Não há skill de Chaos Engineering instalada. CHAOS é uma **frente de trabalho** (modelo do ERP Sítio Silva); daquele projeto só foi emprestada a escala de gravidade.

## Isolamento
- Cópia descartável do repositório, sem remoto (hoje: `docs/chaos/reproducoes/rodar.sh`).
- Cada processo roda com `env -i`, sem `GH_TOKEN`, AWS, chave de API nem proxy. Um `sitecustomize.py` bloqueia `socket.connect`/`getaddrinfo`.
- API The Odds e modelo são simulados por fixtures (`iso/chaos/chaos_common.py`). Não rodei treino, tarefa agendada, publicação, git push, merge nem subagente.
- Produção ao final: SHA igual, `git status` limpo e os 117 arquivos de `data/` e `docs/` com sha256 idêntico ao inicial.

## Pré-voo (caminhos atuais)
| etapa | caminho |
|---|---|
| captura | `scripts/auto_capture.py` → `line_shopping.fetch_live_odds/fetch_sharp_probs` → `prediction_history.record_closing_lines` |
| previsão | `card_report.generate_card_report` → `analyze_card` → `prediction_history.record_card_predictions` |
| snapshots | `src/snapshots.py` (só anexa; o histórico aponta por `prediction_id`) |
| prova | `src/proveniencia.py` (corpo `.json.gz` → `chamadas.csv` → `cotacoes.csv`) |
| renderização | `generate_card_report(..., somente_render)`, `src/painel.py` |
| liquidação | `fechamento.preencher_template` → `sync_results_from_template` → `compute_series_summary` |

Suíte de base na cópia: 927 passam, 14 pulados, 3 falham (dependem de `data/processed`, que não está versionado).

---

## Defeitos (prioridade; D5 é limitação já conhecida, os demais são novos)

### D1 — ALTA — histórico truncado é lido sem erro e regravado (perda silenciosa)
- **Onde:** `src/prediction_history.py:167` (`pd.read_csv` sem checagem) e as escritas não atômicas em `:486`, `:546` e `:635` (`df.to_csv(path)` direto sobre o arquivo).
- **Reprodução:** `c2_interrupcao.py` (C2a) e `c3_truncado.py` (C3a). Uma queda no meio do `to_csv` deixa o arquivo pela metade. A publicação seguinte roda normalmente.
- **Esperado × observado:** o esperado é recusar o arquivo com motivo. O observado foram 233 → 152 linhas: 81 linhas do pré-registro perdidas sem nenhum aviso. Com só a última linha cortada, ela volta gravada com `estado_resultado='aguar'` e `prediction_id` vazio.
- **Impacto:** o artefato insubstituível do projeto é perdido e o estado truncado é gravado por cima. Dá para recuperar pelo git só se ninguém commitar antes.
- **Correção:** escrever em arquivo temporário no mesmo diretório e fazer `os.replace`. Na leitura, rejeitar arquivo cuja última linha não termine em `\n` ou tenha número de campos diferente do cabeçalho.
- **Teste:** truncar o histórico em 60% e na última linha, e verificar que `publicar` levanta erro e que o arquivo fica byte a byte igual.

### D2 — ALTA — captura horária e publicação concorrentes perdem escrita
- **Onde:** as mesmas escritas de D1 (ler, modificar e gravar sem trava): `record_closing_lines` (`:567–635`) contra `record_card_predictions` (`:245–486`).
- **Reprodução:** `c4_concorrencia.py`, com dois processos reais (`scripts.auto_capture` e a publicação do card).
  - (a) Captura lenta: o histórico volta a apontar para as versões antigas (`6b87…`, `cadd…`) enquanto a página publicada mostra `834e…`, `a29c…`.
  - (b) Publicação lenta: a captura registra no log "fechamento gravado em 2", mas o histórico termina com 0/2.
  - (c) **Sem nenhuma latência injetada**, 8/12 e 3/12 rodadas simultâneas perderam escrita, em duas execuções.
- **Impacto:** no dia do evento o `auto_capture` roda de hora em hora e a publicação manual se repete (o log de 21/11 tem 6 commits "Atualiza relatório"). O ponteiro do pré-registro pode divergir da página publicada, ou o fechamento some. Se isso acontecer na última hora antes da luta, não há recaptura.
- **Correção:** trava exclusiva por arquivo (lockfile com `O_EXCL` e espera limitada) que envolva ler, modificar e gravar, mais a escrita atômica de D1.
- **Teste:** repetir C4a, C4b e C4c (50 rodadas) e exigir 0 perdas, ou a segunda escrita falhando com mensagem de trava.

### D3 — ALTA — republicar sem modelo apaga a previsão aberta do histórico
- **Onde:** `src/card_report.py:301`. Sem modelo, a luta aberta vai para `no_prediction` porque `frozen` só contém lutas *fechadas*. Depois, `src/prediction_history.py:307` e `:439` substituem a linha aberta por uma linha vazia, com nova versão.
- **Reprodução:** `c8_ui.py` (C8a). `predict_fight` levanta `FileNotFoundError` (como em `src/predict.py:151`) numa publicação normal.
- **Esperado × observado:** o esperado é preservar o pré-registro, ou recusar a publicação, e comunicar a indisponibilidade. O observado foi `model_prob_a` 0.60/0.70 → `NaN`, `model_side` → `NaN` e ponteiro trocado para versões vazias. A página sai "sem previsão".
- **Impacto:** acontece com uma ação plausível (publicar num clone ou máquina sem `models/`, ou depois de um treino que falhou). A versão anterior continua no arquivo de snapshots, mas o histórico deixa de apontar para ela, e a luta sai dos denominadores se ninguém perceber antes do card.
- **Correção:** em `record_card_predictions`, linha *sem previsão* nunca substitui linha aberta *com previsão*. Mantém a anterior e registra o motivo. Alternativa: `generate_card_report` aborta quando nenhum preditor carrega e existe pré-registro aberto.
- **Teste:** publicar com o preditor normal, republicar com `FileNotFoundError` e verificar que o `prediction_id` e o `model_prob_a` não mudam.

### D4 — ALTA — renomear o card duplica a luta e conta duas vezes no placar
- **Onde:** `src/prediction_history.py:335`. A chave da linha é `event_name` mais o par de lutadores, não `event_date` mais o par.
- **Reprodução:** `c5_c6_fechamento_confronto.py` (C6b). A luta principal troca de adversário, então o nome do card muda ("Alfa vs Beta" → "Alfa vs Epsilon"), e o card é republicado.
- **Esperado × observado:** o esperado é uma linha por luta e data. O observado é que Caio x Davi ganha **2 linhas** abertas. Depois da liquidação, as **duas** ficam fechadas e `compute_series_summary` conta a mesma luta duas vezes (acertos e P&L).
- **Impacto:** placar e retorno da série ficam corrompidos em silêncio. O nome do card muda justamente quando há substituição na luta principal.
- **Correção:** casar por `event_date` mais `_same_fight`, como já fazem `sync_results_from_template` e `record_closing_lines`. Atualizar `event_name` na linha existente.
- **Teste:** publicar, renomear e republicar, depois liquidar. Exigir uma linha por par e data e contagem idêntica à do card sem renomeação.

### D5 — ALTA — (limitação já conhecida, reconfirmada) depois da meia-noite local o card inteiro sai da janela de captura
- **Onde:** `scripts/auto_capture.py:119–120`, que exige `hoje <= event_date` com a data **local**. A regra por luta (`:174–188`) nem chega a rodar.
- **Reprodução:** `c7_tempo.py` (C7a), com fuso America/Sao_Paulo e relógio às 00:30 de domingo para um card de sábado cuja luta principal começa às 02:00.
- **Esperado × observado:** o esperado é capturar a luta principal e bloquear só a prelim já iniciada. O observado é que nenhuma luta é capturada, e o job ainda registra `SEM FECHAMENTO … já passou`, um alarme falso.
- **Evidência em produção (só leitura):** em 07/11 e 14/11 a última captura foi às 02:12 UTC (23:12 em Brasília). As lutas posteriores ficaram com "fechamento" de 0,1 a 2,3 h antes do início, e a principal de 14/11 (Gane) ficou com 2,3 h.
- **Impacto:** o fechamento e o CLV do card principal de todo evento nos EUA são medidos com preço que não é o de fechamento.
- **Correção:** a comporta de custo usa uma janela em UTC (por exemplo, `event_date` até `event_date + 1 dia` em UTC) e deixa o corte para a regra por luta (`commence_time`). `events_missing_closing` deve usar a mesma regra.
- **Teste:** relógio simulado às 00:30 e às 02:30 BRT. A luta principal é capturada na primeira e bloqueada na segunda, e não há alarme "perdido" enquanto existe luta futura.

### D6 — MÉDIA — resposta parcial da API derruba a captura e o pré-registro
- **Onde:** `src/line_shopping.py:180`, `float(outcome["price"])` sem checagem.
- **Reprodução:** `c1_api.py` (C1c) e `c1c_trace.py`. A Pinnacle vem sem `price` num dos lados de uma luta.
- **Esperado × observado:** o esperado é que a luta incompleta fique sem fechamento e as demais sejam capturadas. O observado é `KeyError('price')` no `auto_capture` (que diz "nunca deve estourar") e também na publicação (`fetch_sharp_probs` diz "Nunca levanta"). A luta íntegra também perde o fechamento.
- **Impacto:** nenhum preço inventado, mas o card inteiro fica sem captura e sem pré-registro enquanto a resposta vier assim.
- **Correção:** ignorar `outcome` sem preço numérico maior que 1 e registrar o motivo da luta como bloqueio.
- **Teste:** C1c passa, com a luta íntegra capturada e a parcial vazia com motivo.

### D7 — MÉDIA — `somente_render` desenha odds e EV do CSV atual, não os registrados
- **Onde:** `src/card_report.py:226` e o bloco `base`. Com `somente_snapshot`, a probabilidade vem da versão, mas as odds vêm da linha do CSV.
- **Reprodução:** `c8_ui.py` (C8b). Depois do registro (odd da casa 1,90), o CSV passa a ter 2,60 e o card é renderizado com `--somente-render`.
- **Esperado × observado:** o esperado é a odd 1,90 e o EV 1,140, como registrados. O observado é a odd 2,60 e o EV 0,60 × 2,60 = 1,560, uma combinação que nunca foi registrada e que pode criar uma perna EV>1 inexistente.
- **Impacto:** a página "desenhada do registro" mostra números recalculados. Os registros em si ficam intactos.
- **Correção:** em `somente_snapshot`, tirar odds, odd da casa e refs da versão (`odds_*_decimal`, `odds_*_tp`, `ref_tp*`). Se faltarem, marcar como indisponível.
- **Teste:** C8b com o HTML contendo 1,90 e não 2,60, e a lista EV>1 igual à da versão.

### D8 — MÉDIA — linha truncada no CSV do card some do pré-registro em silêncio
- **Onde:** `src/card_report.py:93`, `dropna(subset=required)` sem aviso.
- **Reprodução:** `c3_truncado.py` (C3c), com a última linha `Caio Gama,Davi Delta,1.5`.
- **Esperado × observado:** o esperado é recusar com motivo. O observado é que só 1 de 2 lutas é registrada e o único sinal é "Card carregado: 1 lutas".
- **Impacto:** a luta fica sem pré-registro, e não há backfill. Isso contraria a regra do projeto "nunca omitida em silêncio".
- **Correção:** levantar `ValueError` listando as linhas incompletas.
- **Teste:** C3c com erro nomeando a linha e o histórico inalterado.

### D9 — MÉDIA — append sobre linha parcial cola a versão nova e torna o arquivo ilegível
- **Onde:** `src/snapshots.py:179`. O mesmo padrão aparece em `proveniencia.py:104` e `:209` e em `decisoes.py:148`: abre em modo `a` sem garantir `\n` no fim.
- **Reprodução:** `c3_truncado.py` (C3b). `previsoes_snapshots.csv` fica sem os últimos 20 caracteres (sobra de queda anterior), e o card é publicado.
- **Esperado × observado:** o esperado é recusar *antes* de gravar. O observado é que a versão nova é colada na linha parcial, o histórico já grava o ponteiro para ela, e só depois `painel` estoura com `ParserError` (29 × 31 campos). A partir daí qualquer leitura do arquivo de versões falha.
- **Impacto:** a falha é detectada, mas tarde e piorando o estado. O arquivo "só anexa" precisa de reparo manual.
- **Correção:** antes de anexar, conferir que o arquivo termina em `\n` e que a última linha tem o número de campos do cabeçalho. Se não tiver, abortar antes de qualquer outra escrita.
- **Teste:** C3b com erro levantado antes de `record_card_predictions` gravar o histórico, e os dois arquivos inalterados.

## Limitação já conhecida
- **Cotações sintéticas no índice real:** `data/proveniencia/cotacoes.csv` contém 90 linhas "Alice Silva/Bia Costa" (chamada `bb0939371760e289`, ausente de `chamadas.csv`). O incidente está documentado em `tests/conftest.py`, que cita "dez cotações", e a trava contra novas escritas existe, mas as linhas continuam no arquivo. Elas não passam na verificação, porque a chamada não está no índice.

## Cenários que passaram
- **C1a/C1b (timeout, JSON truncado):** o job sai sem gravar e o histórico fica intacto.
- **C2b (corpo `.gz` truncado):** `conferir()` reprova com "corpo ilegível". Resíduo: uma resposta idêntica depois disso não repara o corpo, porque `proveniencia.py:93` só grava se o arquivo não existir.
- **C2c (índice de cotações sem gravar):** a `sharp_ref` órfã reprova na verificação.
- **C2d (queda entre snapshot e histórico):** as versões órfãs não viram versão em vigor.
- **C3d (JSON inválido do cache de cards a registrar):** tratado como vazio.
- **C5 (liquidação 3× mais captura manual tardia):** é idempotente.
- **C6a (lados invertidos):** a probabilidade e a odd da casa são reorientadas corretamente.
- **C6c (revanche):** a linha nova é própria e a antiga fica intacta.
- **C7b (luta iniciada × futura):** bloqueio por luta correto.
- **C7c (cotação de 7 h; leitura antes × depois do início):** reprovada com motivo.
- **C8c (ponteiro de snapshot quebrado):** a luta vai para "sem previsão" com motivo.

## Não exercitados
- Raspagem da Thunderpick (navegador real) e `build_card` com UFCStats: exigem rede ou navegador.
- `ciclo_semanal` ponta a ponta: inclui treino e push.
- Concorrência no Windows (a máquina agendada): os testes rodaram em Linux, onde `O_APPEND` é atômico para linhas pequenas.
- `painel` com base de resultados real: a base não está no clone.

## Scripts e resultados
- `docs/chaos/reproducoes/`: um script por cenário (cabeçalho com falha injetada, esperado e propriedade) e `rodar.sh`.
- `docs/chaos/resultado_rodada1.txt`: saída da execução em `2f73748`.
- Instruções: `docs/chaos/README.md`.
