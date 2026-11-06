# Protocolo: o modelo acrescenta informação ao mercado?

**Versão do protocolo: `vantagem-v1`.**
**Escrito em 2026-11-06, ANTES de executar qualquer comparação.** Nenhum número
desta rodada foi observado antes deste arquivo existir. Branch
`pesquisa-vantagem`, a partir de `b29b1b4`.

Este arquivo é pré-registro. Mudar critério depois de ver resultado é a regra
zero do `README.md`, e "faltou pouco" não é motivo.

---

## 0. O que esta rodada NÃO faz

- não altera o modelo de produção, hiperparâmetro, feature ou algoritmo;
- não inclui shrinkage novo, dados pré-UFC ou features novas;
- não toca `data/prediction_history.csv`, `data/previsoes_snapshots.csv`,
  `data/decisoes.csv`, `data/proveniencia/` nem `docs/`;
- não promete rentabilidade, e resultado negativo ou inconclusivo mantém o
  baseline.

Artefatos experimentais vão para `data/experimentos/` — diretório separado,
fora dos registros de produção.

---

## 1. Hipótese

**H1.** As probabilidades do nosso modelo acrescentam informação às
probabilidades de referência do mercado: uma combinação das duas prevê melhor
o vencedor do que o mercado sozinho.

**H0.** Não acrescentam: o mercado sozinho é tão bom ou melhor, e o peso ótimo
do modelo é zero.

A formulação é deliberada. A pergunta **não** é "o modelo ganha do mercado" —
isso já foi medido e a resposta é não (log loss 0,619 contra 0,502 nas lutas
casadas de 2021-23, e AUC 0,540 contra 0,635; ver `DISCRIMINACAO.md`). A
pergunta é se há informação **residual**, que é compatível com o modelo ser pior
no total.

---

## 2. População elegível e exclusões

**Fonte única: `data/prediction_history.csv`** — o pré-registro congelado da
série de paper trading. É o único conjunto do projeto com previsão feita
**antes** da luta e preço capturado **antes** da luta. A justificativa está
inteira na §3.

Elegível a observação que satisfaz **todas**:

| exigência | motivo |
|---|---|
| `actual_winner` preenchido | sem resultado não há o que avaliar |
| `model_prob_a` preenchido | luta sem previsão (estreante nos dois lados) não entra |
| `odds_a_decimal` e `odds_b_decimal` preenchidos | precisa dos **dois** lados para retirar a margem |
| `estado_resultado == "vencedor"` | luta cancelada ou com adversário trocado sai |

**Exclusões são registradas uma a uma**, com motivo, em
`data/experimentos/exclusoes_vantagem-v1.csv`. Nada é imputado: ausência de dado
não é substituída por suposição.

Contagem no momento da escrita deste protocolo (conferida antes de qualquer
comparação): **179 lutas em 16 eventos**, de 2026-07-11 a 2026-10-31. É pouco, e
a §9 diz o que isso custa em precisão.

---

## 3. Auditoria das odds: o que existe e o que não existe

Esta seção é a condição do experimento. Foi executada **antes** do protocolo
fechar, e mudou o desenho dele.

### 3.1 `data/raw/market_odds.csv` — NÃO serve para decisão

7340 linhas, 1994-03-11 a 2023-09-16. Problemas, medidos:

- **um único `odds_extract_ts` em todo o arquivo**: `2023-12-28 11:40:25`;
- toda linha com horário foi extraída **103 a 3338 dias DEPOIS** do evento
  (mediana 1734 dias, ~4,8 anos); **zero** antes;
- **nenhuma coluna de casa** e **nenhuma coluna de mercado**;
- um único par `favourite_odds` / `underdog_odds` por luta, sem orientação por
  lado além dos nomes de favorito e azarão.

Consequência: **não há prova de que esses preços estiveram disponíveis antes da
luta**, nem de qual casa vieram. Usá-los para simular decisão antiga seria usar
odds de hoje para decidir ontem. Este arquivo serve como **referência de nível
de mercado** em análise exploratória, nunca como preço executável e nunca como
base de P&L apresentado como válido.

### 3.2 `data/processed/market_comparison.csv` — desenvolvimento, não avaliação

821 lutas, 2021-08-21 a 2023-09-16. **Zero** dentro da janela de teste do modelo
(que começa em 2024-05-04), e a janela se sobrepõe ao treino (que termina em
2021-09-25) e à calibração de produção — o próprio `src/tuning.py` trunca o
dataset em 2023-09-16 por causa disso.

Então as probabilidades de modelo deste arquivo são **parcialmente dentro da
amostra**. Ele é **dado de desenvolvimento**. Não é teste intocado e não é
validação prospectiva.

### 3.3 `data/prediction_history.csv` — o único conjunto temporalmente adequado

Cada linha nasceu de um card **antes** do evento, com a previsão congelada e o
preço capturado naquele momento. É o que a §2 usa. Para cada observação o
experimento registra:

- evento e confronto; participantes e orientação (`fighter_a`/`fighter_b` como
  gravados, e `participantes_fonte` quando existe);
- instante da previsão: **o evento é a granularidade real** — o pré-registro
  guarda a data do evento, não o timestamp da gravação. A partir de 06/11/2026
  `data/previsoes_snapshots.csv` passou a gravar `criado_em` por versão; as 207
  linhas anteriores foram semeadas com temporalidade declarada **não
  recuperável**. Esta é a limitação temporal mais importante do conjunto e está
  declarada em cada linha de saída;
- instante da captura do preço: `data/proveniencia/cotacoes.csv` quando a linha
  tem `sharp_ref`/`ref_tp`; **ausente** para a maioria das linhas de 2026-07 a
  2026-09, porque a proveniência por preço só existe desde out/2026;
- casa e mercado: `sharp_casa` (Pinnacle, h2h) em 122 das 187 resolvidas;
  Thunderpick (h2h) em **19**;
- odds dos dois lados: `odds_a_decimal`/`odds_b_decimal` (mediana) em todas;
  `odds_a_tp`/`odds_b_tp` em 19;
- referência verificável à fonte: `sharp_ref`, `close_ref`, `ref_tp` onde
  existem;
- qualidade e limitação da evidência temporal: campo `temporalidade` por linha
  na saída, com o texto exato da limitação.

### 3.4 Retirada da margem da casa

A odd decimal de um lado implica `1/odd`. Num mercado de dois resultados a soma
das duas implícitas passa de 1 — é o overround (vig). Normalização
proporcional (`src.utils.remove_vig_two_way`):

```
p_A = (1/odd_A) / ((1/odd_A) + (1/odd_B))
```

É o método mais transparente, e não é o único: existem "power" e "Shin", que
distribuem a margem de forma desigual entre favorito e azarão. A escolha do
proporcional está registrada **aqui, antes dos resultados**, e não será trocada
depois de ver qual favorece o modelo.

### 3.5 Duas coisas diferentes, nunca misturadas

| | o que é | de onde vem | para que serve |
|---|---|---|---|
| **(a) referência do mercado** | mediana de 20–28 casas, devigada | `odds_a/b_decimal` | probabilidade de comparação. É o candidato A |
| **(b) preço executável** | odd da casa onde a aposta seria feita | `odds_a/b_tp` (Thunderpick) | único preço que pode gerar retorno |

**A mediana de casas não é cotação executável.** Ninguém aceita uma aposta na
mediana de 25 casas. Qualquer retorno calculado sobre ela é ficção, e o
experimento não calcula nenhum.

---

## 4. Janelas temporais

Divisão **cronológica por evento**: todas as lutas de um evento ficam na mesma
fatia. Dividir dentro do evento vazaria contexto do próprio card.

| fatia | eventos | período | uso |
|---|---|---|---|
| **desenvolvimento** | os 8 primeiros | 2026-07-11 a 2026-08-29 | estimar o peso da combinação; qualquer escolha de parâmetro |
| **avaliação** | os 8 últimos | 2026-09-05 a 2026-10-31 | medir, uma vez |

Nenhuma escolha olha a fatia de avaliação. O treino e a calibração do modelo em
si são anteriores às duas (treino termina em 2021-09-25), então as
probabilidades do candidato B são **fora da amostra** nas duas fatias.

O conjunto de teste de 2024-05-04 a 2026-10-31 do `src/evaluate.py` já foi
consultado muitas vezes neste projeto. **Trato-o como desenvolvimento**, não
como teste intocado, e ele não é usado para medir nada aqui.

---

## 5. Candidatos

Na mesma população e nos mesmos horários.

**A. Mercado.** Implícita dos dois lados da mediana, devigada pela §3.4.

**B. Modelo atual.** `model_prob_a` como foi **congelada no pré-registro**.
Não é recalculada: recalcular hoje usaria uma base que já contém o resultado
daquela luta. O baseline não é alterado.

**C. Combinação — uma única formulação, escolhida agora.**

Pooling logístico com um peso:

```
logit(p_C) = (1 − w) · logit(p_A) + w · logit(p_B),    w ∈ [0, 1]
```

Propriedades que motivam a escolha, todas verificáveis antes de rodar:

- **`w = 0` é permitido e é o ponto de H0**: o procedimento pode concluir que o
  modelo não acrescenta informação;
- **simétrico por construção**: inverter A/B nega os dois logits, logo nega o
  logit combinado, logo `p_C → 1 − p_C`. A decisão econômica para o mesmo
  atleta não muda. Travado em teste;
- um único parâmetro, estimado por minimização de log loss **só na fatia de
  desenvolvimento**, em grade de `0,00` a `1,00` em passos de `0,01`;
- probabilidades são apertadas em `[0,001, 0,999]` antes do logit, para o
  log loss não estourar em 0 ou 1. O limiar está fixado aqui.

**O peso não é escolhido pelo lucro de nenhuma janela.** Nem pelo lucro da
janela de desenvolvimento: o critério de estimação é log loss.

---

## 6. Métricas e contrastes

**Principal: log loss.** Mede calibração e discriminação juntas e é a métrica
própria para probabilidade.

Complementares: Brier score; AUC como **diagnóstico** de ordenação; curvas de
calibração com contagem e incerteza por faixa; cobertura e número de
observações; estabilidade entre as janelas; e os grupos de experiência prévia no
UFC (0, 1–2, 3–5, 6+ lutas), derivados de `data/raw/fights.csv` contando lutas
anteriores à data do evento.

### 6.1 Contraste CONFIRMATÓRIO — um só

**Δ log loss (C − A) na fatia de avaliação**, com o peso `w` fixado no
desenvolvimento.

É a tradução direta de H1: a combinação bate o mercado sozinho? Qualquer outro
número desta rodada é **exploratório** e está rotulado como tal, inclusive:

- B contra A (já se sabe que B perde; serve de sanidade);
- Δ log loss em desenvolvimento;
- Brier, AUC, calibração, grupos de experiência, estabilidade;
- qualquer análise sobre `sharp_prob`, `close_prob` ou CLV;
- qualquer coisa calculada sobre `market_comparison.csv`.

### 6.2 Incerteza

**Bootstrap por evento (cluster bootstrap).** O evento é a unidade de
reamostragem: as lutas de um card compartilham condições e tratá-las como
independentes estreitaria o intervalo artificialmente.

- 10 000 reamostragens de eventos **com reposição**, `random_state = 20261106`;
- **pareado**: cada reamostragem usa os **mesmos eventos e as mesmas
  observações** para todos os candidatos, e a estatística é a **diferença
  pareada** de log loss;
- intervalo percentil de 95%.

Com 8 eventos na avaliação, a reamostragem tem no máximo 8 clusters distintos. O
intervalo vai ser largo, e isso é a resposta honesta sobre o que esta amostra
sustenta — não um defeito a contornar com outra unidade de reamostragem.

### 6.3 Critério de promoção

Para C substituir A como referência de decisão, **todas**:

1. Δ log loss (C − A) na avaliação **negativo** (C melhor);
2. limite superior do IC 95% da diferença pareada **abaixo de zero**;
3. `w > 0` estimado no desenvolvimento;
4. o sinal se mantém nas duas metades da avaliação (4 + 4 eventos);
5. a melhora não vem de um único evento: remover o evento de maior efeito
   mantém o sinal.

Nada aqui promove C a produção automaticamente. Promoção é decisão humana, em
rodada própria, com o baseline atual mantido até então.

### 6.4 Negativo e inconclusivo

- **Negativo**: Δ log loss (C − A) ≥ 0 na avaliação, **ou** `w = 0` no
  desenvolvimento. Conclusão: nesta amostra o modelo não acrescenta informação
  ao mercado. Baseline mantido.
- **Inconclusivo**: o IC 95% da diferença pareada **contém zero**. Conclusão: a
  amostra não decide. Baseline mantido.

**Ausência de significância não é prova de equivalência.** O relatório diz o
tamanho do efeito que esta amostra conseguiria detectar, em vez de afirmar que
não há efeito.

### 6.5 O que não vale como objetivo nem como explicação

- **não** é objetivo aumentar a fração de previsões acima de 75%. Essa fração é
  critério de aceite de **feature nova** (ver a skill `ufc-analise`), não de
  combinação com o mercado, e usá-la aqui seria importar um critério fora do
  contexto;
- **não** atribuo diferença a mecanismo nenhum sem ablação que o teste;
- **CLV** é diagnóstico complementar e nunca prova isolada de vantagem;
- **P&L não é variável de decisão** (regra de parada do `README.md`).

---

## 7. Dados ausentes

Sem imputação. Linha que não satisfaz a §2 é **excluída e registrada** com
motivo. A cobertura (quantas das lutas elegíveis cada candidato cobre) é
reportada junto de cada métrica, porque comparar candidatos em populações
diferentes seria comparar outra coisa.

Observação já conhecida: `sharp_prob` existe em 122 das 187 resolvidas e
`odds_a/b_tp` em 19. Nenhuma das duas entra no contraste confirmatório,
justamente para não trocar a população.

---

## 8. Retorno: o que só pode ser feito com preço executável

O retorno só é calculado onde há preço **executável** com temporalidade
verificável — hoje **19 lutas**, o que não sustenta conclusão nenhuma. Quando
sustentar, a política de simulação é esta, registrada agora:

- **unidade fixa**, uma unidade por perna;
- critério de entrada: valor esperado estimado maior que 1 com o preço
  executável;
- **sem** Kelly, martingale, recuperação de perdas ou otimização de stake;
- métricas: retorno líquido por unidade; número de apostas e de eventos;
  exposição por evento; drawdown; concentração do lucro nas maiores vencedoras;
  sensibilidade à retirada de uma aposta e de um evento; incerteza por bootstrap
  de evento.

**Duas quantidades distintas, nunca trocadas:**

| expressão | nome | leitura |
|---|---|---|
| `p × odd` | retorno bruto esperado por unidade | 1,26 significa receber 1,26 por 1 apostado |
| `p × odd − 1` | lucro esperado líquido por unidade | 0,26 significa ganhar 0,26 por 1 apostado |

O painel do site mostra a primeira. O critério `EV > 1` é sobre a primeira.

---

## 9. Quantos eventos faltam

A avaliação não tem prazo. O número de eventos necessário sai da precisão
desejada e da variabilidade observada, calculado **depois** de medir o desvio
por evento na fatia de avaliação — e o relatório reporta esse cálculo em vez de
declarar vantagem a partir de um fim de semana.

**Um único fim de semana não demonstra nada**, qualquer que seja o resultado.

---

## 10. Modo sombra

Fluxo separado da produção (`scripts/sombra.py`, registros em
`data/experimentos/sombra/`), **append-only**. Antes de cada luta registra:
versão do modelo e manifesto; probabilidades de A, B e C; preços disponíveis e
as provas; instante de decisão; seleção ou recusa com motivo; e a versão deste
protocolo (`vantagem-v1`).

Resultado e liquidação entram depois, em arquivo próprio, **sem reescrever o que
foi previsto**.

Evento já usado para ajustar o peso é classificado como **desenvolvimento**, não
como avaliação prospectiva. O campo `fatia` de cada linha diz qual é.

Nada é agendado, nenhuma API paga é consumida e nada é publicado sem
autorização específica. O comando é manual e está no `README` do experimento.

---

## 11. Reprodução

O manifesto do experimento (`data/experimentos/manifesto_<versao>.json`)
registra: commit do código; sha256 dos dados de entrada; parâmetros e semente;
versões das bibliotecas; os `fight_uid` de cada fatia; as previsões fora da
amostra dos três candidatos; e as exclusões com motivo.

Hashes dos registros reais de produção são conferidos antes e depois de cada
execução. Segredo nenhum entra em log, relatório ou commit.
