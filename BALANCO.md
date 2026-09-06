# Balanço do projeto — 06/09/2026

Dez eventos pré-registrados, nove fechados, 350 testes. Este documento é a
resposta honesta a "onde isto está?". Complementa o `DISCRIMINACAO.md`, que
explica *por que* o modelo não separa; aqui o assunto é o projeto inteiro.

---

## 1. O que o projeto mede, e o que não se propõe a fazer

O objetivo declarado é **um modelo honesto e bem calibrado de previsão de
luta**. O paper trading existe como régua de realidade, não como finalidade —
e essa distinção decide quase tudo o que vem abaixo.

Consequência prática: quando o P&L e o CLV discordam, quem manda é o CLV.
Quando um card de 10/12 tenta convencer de que o modelo melhorou, a resposta é
que 11 lutas não medem nada.

## 2. O modelo, hoje

Conjunto de teste: **1309 lutas, 09/mar/2024 a 05/set/2026** — cronológico,
agrupado por luta (as duas linhas espelhadas nunca se separam).

| | AUC | log loss | Brier |
|---|---|---|---|
| logreg (produção) | **0,6707** | 0,6477 | 0,2281 |
| GBM | 0,6553 | 0,6547 | 0,2314 |

Treino: 5840 lutas (após o corte das Regras Unificadas). Calibração: 1307,
com `sigmoid` para a logreg e `isotonic` para o GBM, escolhidas em `cal_select`
— nunca no teste.

**A logreg continua ganhando do GBM.** É o resultado que sustenta o diagnóstico
do `DISCRIMINACAO.md`: o teto está nas features, não na capacidade do modelo.
Mais flexibilidade piora.

## 3. A régua: o mercado

Nas 99 lutas do teste que têm odds:

| | acurácia | log loss | Brier |
|---|---|---|---|
| modelo | 60,6% | 0,659 | 0,233 |
| **mercado** | **78,8%** | **0,535** | **0,176** |

Não é um detalhe: o mercado erra menos em toda métrica, e por margem larga.
O modelo não é competitivo com a linha de aposta, e o projeto sabe disso desde
a análise de discriminação — a diferença está em **resolução** (poder de
separar), não em calibração.

## 4. A série: nove eventos fechados

| data | evento | modelo | mercado | CLV |
|---|---|---|---|---|
| 11/07 | UFC 329: McGregor vs Holloway 2 | 6/11 | 11/14 | — |
| 18/07 | Du Plessis vs Usman | **10/11** | 7/11 | — |
| 25/07 | Ankalaev vs Guskov | 7/11 | 10/11 | — |
| 01/08 | Medic vs Rodriguez | 9/14 | 12/14 | — |
| 08/08 | Gamrot vs Salkilld | 8/10 | 9/10 | — |
| 15/08 | UFC 330: Makhachev vs Machado Garry | 6/11 | 8/11 | — |
| 22/08 | Hernandez vs Rodrigues | **8/11** | 7/11 | — |
| 29/08 | Nurmagomedov vs Song | **10/12** | 8/13 | +0,23 pp |
| 05/09 | Hooker vs Parnasse | 6/11 | 10/13 | +1,58 pp |

**Acumulado: modelo 70/102 (68,6%), mercado 82/108 (75,9%).**

O modelo bateu o mercado em **3 dos 9** eventos. Perdeu nos outros seis. A
diferença de ~7 pontos percentuais no acumulado é consistente com o que o
teste histórico já dizia — não é surpresa, é confirmação.

**Pernas EV>1: 20 de 40, P&L −0,77u.** Praticamente empate, depois de nove
eventos. Vale lembrar que essa série passou de −2,32u para +2,73u num único
card (29/08, cinco pernas, todas ganhas) e voltou para o vermelho no seguinte.
É a variância que a regra de parada descreve, acontecendo na prática.

## 5. O CLV: a régua que só agora existe

Os sete primeiros eventos **não têm CLV** e nunca terão — a API só serve
eventos futuros e não há backfill. A captura automática entrou no ar em
26/ago; o evento 8 foi a primeira medição da série.

**Acumulado: +0,87 pp em média, 15 de 23 pernas bateram o fechamento.**

O detalhe mais interessante do projeto inteiro está aqui: no evento 9 o card
foi ruim (6/11, −3,50u nas pernas) e o **CLV foi +1,58 pp, com 8 de 11 pernas
batendo o fecho**. As duas réguas apontaram para lados opostos no mesmo card.

Isso é exatamente o que se espera quando o processo pega preço bom e a
amostra é pequena — e é o motivo de o CLV ter sido escolhido como medida
principal. Mas 23 pernas não decidem nada. A hipótese registrada no painel
pede ~10 eventos; estamos em dois.

## 6. Saúde dos dados

Três defeitos encontrados em 06/09, todos da mesma família — **junção com
chave não única, ou valor que depende de um lado**. Nenhum quebrava nada:
todos deixavam o arquivo plausível e o número errado.

| defeito | efeito | alcance |
|---|---|---|
| stats duplicadas | 22 lutas com **128 linhas** em vez de 2 — 14% do treino | entrou no refresh de 06/09; não afetou a série |
| homônimo na bio | idade 48,9 e alcance vazio no main event de 12/09 | antigo; tocou **2** das 124 lutas da série |
| `sharp_prob` do lado antigo | divergência com **sinal trocado** (+34 no lugar de −22,5) | 1 linha, criada hoje |

Os três foram encontrados por acidente, no meio de outra tarefa. Foi o que
motivou a varredura de invariantes e os 12 testes de integridade novos.

**O que a varredura confirmou:**

- **Vazamento temporal: zero.** `experience_diff` conferido contra recontagem
  independente do pipeline nas 17434 linhas. As 174 divergências eram todas de
  noite de torneio (anos 90, mesmo lutador 2-4 vezes na mesma data), que é
  ordem dentro da noite e não vazamento — e está fora da era de treino.
- **O congelamento se sustenta.** Nos 25 commits que já tocaram o
  `prediction_history.csv`, **nenhuma linha fechada foi alterada depois de
  fechada**. É a promessa central do projeto, e ela é verdadeira.
- `sharp_prob` e `close_prob` coerentes com o lado do modelo em todas as 62 e
  23 linhas restantes.

## 7. O que está medido e o que ainda é chute

**Medido:**
- a calibração é boa: **ECE 0,0127** no teste, desvios de poucos pontos
  percentuais onde há dado — o problema nunca foi o modelo mentir sobre a
  própria confiança;
- o modelo é pior que o mercado, e por quanto;
- o gargalo é resolução, não calibração;
- mais capacidade de modelo piora;
- `opp_quality_diff` não passou;
- o corte por época muda pouco (AUC +0,002 em teste fixo) — é validade de
  dado, não ganho.

**Não medido (e registrado como tal):**
- se as pernas com lutador de pouca experiência são piores — **hipótese
  pré-registrada**, gatilho em 60 pernas por grupo, ~25 eventos;
- se o CLV positivo persiste — 23 pernas de ~60 necessárias;
- se o grupo com respaldo sharp se sai melhor — 5 contra 13 pernas, nada;
- categoria de peso como feature: candidata reconhecida, nunca avaliada.

## 8. O que vale e o que não vale fazer

**Não vale:** mexer no modelo ou nos hiperparâmetros. As três regras de parada
continuam válidas, e nada do que aconteceu desde então as contradiz.

**Não vale:** ajustar regra a partir de dois casos chamativos. A tentativa de
excluir estreantes é o exemplo — a direção batia com a intuição e o p era
0,341.

**Vale:** deixar a série rodar. Esse é o item de maior valor e o que exige
menos trabalho. O projeto não tem, hoje, amostra para responder nenhuma das
perguntas em aberto, e todas dependem do mesmo recurso: eventos.

**Vale:** manter a instrumentação barulhenta. Os três bugs de hoje custaram
uma tarde porque foram encontrados por acaso; os 12 testes novos são o que
transforma isso em falha imediata da próxima vez.

**Feito em 06/09:** a curva de calibração (`scripts/calibration_report.py`).
Confirmou o diagnóstico por outro ângulo — ECE 0,0127, mas 52% das linhas
entre 40% e 60% e as pontas vazias. O modelo não mente sobre a própria
confiança; ele quase nunca tem confiança para declarar.

---

## Uma leitura do conjunto

O projeto ficou **muito melhor em medir do que em prever**, e isso é
deliberado. O modelo é fraco, sabe-se por quê, e a explicação é estrutural: as
features não carregam o que decide uma luta. Nenhuma rodada de tuning vai
resolver isso, e o `DISCRIMINACAO.md` já demonstrou por que.

O que foi construído em volta — pré-registro congelado e verificado, CLV
automatizado, invariantes cobrados, decisões registradas antes dos dados —
vale mais que o modelo, e é o que permite dizer "o modelo é pior que o
mercado" com números em vez de impressão.

A pergunta honesta não é "como melhorar o modelo", é **"o que a série vai
dizer em seis meses"**. E, pela primeira vez, existe régua para responder.
