# Balanço do projeto — 26/10/2026

Dezesseis eventos pré-registrados, quinze fechados, 461 testes. Este documento
é a resposta honesta a "onde isto está?". Complementa o `DISCRIMINACAO.md`, que
explica *por que* o modelo não separa; aqui o assunto é o projeto inteiro.

Substitui o balanço de 06/09/2026, que tinha nove eventos fechados.

---

## 1. O que o projeto mede, e o que não se propõe a fazer

O objetivo declarado é **um modelo honesto e bem calibrado de previsão de
luta**. O paper trading existe como régua de realidade, não como finalidade —
e essa distinção decide quase tudo o que vem abaixo.

Consequência prática: quando o P&L e o CLV discordam, quem manda é o CLV.
Quando um card de 10/12 tenta convencer de que o modelo melhorou, a resposta é
que 11 lutas não medem nada.

**Novidade desde setembro:** o projeto passou a distinguir *medir* de *apostar*.
A odd da mediana das casas mede o mercado; a odd da Thunderpick é a que decide
uma entrada. As duas convivem, com trabalhos separados (seção 7).

## 2. O modelo, hoje

Conjunto de teste: **1322 lutas, 27/abr/2024 a 24/out/2026** — cronológico,
agrupado por luta (as duas linhas espelhadas nunca se separam).

| | AUC | log loss | Brier | acurácia |
|---|---|---|---|---|
| logreg (produção) | **0,6760** | 0,6460 | 0,2273 | 62,6% |
| GBM | 0,6584 | 0,6548 | 0,2313 | 61,8% |

Treino: 5903 lutas (fev/2001 a set/2021, após o corte das Regras Unificadas).
Calibração: 1320, com `sigmoid` para a logreg e `isotonic` para o GBM,
escolhidas em `cal_select` — nunca no teste.

**A logreg continua ganhando do GBM**, sete semanas depois e com mais dado. É o
resultado que sustenta o diagnóstico do `DISCRIMINACAO.md`: o teto está nas
features, não na capacidade do modelo. Mais flexibilidade piora.

## 3. A régua: o mercado

Nas **165** lutas do teste que têm odds (eram 99 em setembro):

| | acurácia | log loss | Brier |
|---|---|---|---|
| modelo | 61,8% | 0,660 | 0,233 |
| **mercado** | **69,7%** | **0,607** | **0,209** |

O mercado erra menos em toda métrica. A diferença encolheu um pouco em relação
a setembro (era 60,6% contra 78,8%), mas a amostra também mudou — não é
melhora do modelo, é outra janela.

## 4. A série: quinze eventos fechados

| data | evento | modelo | mercado | pernas EV>1 | CLV |
|---|---|---|---|---|---|
| 11/07 | UFC 329: McGregor vs Holloway 2 | 6/11 | 11/14 | 8 · −1,10u | — |
| 18/07 | Du Plessis vs Usman | **10/11** | 7/11 | 6 · +5,54u | — |
| 25/07 | Ankalaev vs Guskov | 7/11 | 10/11 | 5 · −3,44u | — |
| 01/08 | Medic vs Rodriguez | 9/14 | 12/14 | 3 · −3,00u | — |
| 08/08 | Gamrot vs Salkilld | 8/10 | 9/10 | 2 · −0,32u | — |
| 15/08 | UFC 330: Makhachev vs Machado Garry | 6/11 | 8/11 | 3 · −1,11u | — |
| 22/08 | Hernandez vs Rodrigues | **8/11** | 7/11 | 2 · +1,10u | — |
| 29/08 | Nurmagomedov vs Song | **10/12** | 8/13 | 5 · +5,05u | +0,23 pp |
| 05/09 | Hooker vs Parnasse | 6/11 | 10/13 | 6 · −3,50u | +1,58 pp |
| 12/09 | Noche UFC: Silva vs Delgado | 5/12 | 9/13 | 6 · −3,90u | +0,80 pp |
| 19/09 | UFC 331: Van vs Pantoja 2 | 7/12 | 7/12 | 3 · +1,88u | −0,53 pp |
| 26/09 | Rosas Jr. vs Barcelos | **6/9** | 5/10 | 3 · +2,65u | −0,50 pp |
| 10/10 | Allen vs Duncan | 5/11 | 5/11 | 6 · −0,24u | −0,44 pp |
| 17/10 | Buckley vs Malott | **5/11** | 4/11 | 5 · +1,75u | **+2,58 pp** |
| 24/10 | UFC 333: Volkanovski vs Evloev | 4/10 | 6/9 | 7 · −0,39u | +0,04 pp |

**Acumulado: modelo 102/167 (61,1%), mercado 118/174 (67,8%).**

O modelo bateu o mercado em **5 dos 15** eventos. A diferença de ~7 pontos
percentuais no acumulado é consistente com o que o teste histórico diz — não é
surpresa, é confirmação.

**Pernas EV>1: 35 de 70, +0,98u.** A série ficou positiva pela primeira vez no
evento 14, depois de ter estado em −4,67u no evento 11. Seis dessas 70 pernas
foram pagas na odd da casa e 64 na mediana — **o acumulado soma duas escalas
de preço**, e o painel diz isso.

Um buraco permanente: o **UFC 332 (03/out) passou sem pré-registro** e não
entra na série. É irrecuperável e fica listado no painel para a semana não
sumir em silêncio.

## 5. O CLV

Os sete primeiros eventos **não têm CLV** e nunca terão — a API só serve
eventos futuros e não há backfill.

**Acumulado: +0,48 pp em média, 51 de 84 pernas (61%) bateram o fechamento.**

Oito eventos medidos, e as duas réguas já discordaram nos dois sentidos: card
ruim com CLV ótimo (05/09) e card bom com CLV negativo (19/09). O melhor CLV da
série (+2,58 pp) veio no card em que o **mercado** teve o pior resultado dele.

A hipótese registrada no painel pede ~10 eventos; estamos em oito. Começa a ter
massa, mas ainda não decide nada.

## 6. O que foi medido no modelo desde setembro

**Categoria de peso: testada e REJEITADA** (out/2026). Era a última candidata
de feature documentada. Nenhuma forma passou o critério de aceite — efeito
principal dá log loss idêntico a quatro casas, e as interações com `elo_diff`,
`age_diff_years` e `reach_diff_cm` *pioram* (+0,0003 a +0,0022).

**E o nulo exato tinha explicação estrutural:** feature simétrica não informa o
preditor de vencedor. Cada luta vira duas linhas espelhadas com labels opostos,
e uma coluna que vale o mesmo nas duas não consegue separá-las. Conferido na
logreg de produção: **`stance_mismatch` tem coeficiente 0,000000** — é peso
morto desde sempre. Travado em `tests/test_feature_simetrica.py`.

**Onde o modelo é fraco** (diagnóstico no teste, nenhuma seleção):

| grupo | lutas | AUC | acurácia |
|---|---|---|---|
| alguém com < 3 lutas prévias | 577 | **0,631** | 57,9% |
| ambos com 3+ | 745 | **0,708** | 66,2% |

A discriminação está quase toda no segundo grupo, e o primeiro é **43,6% do
teste**. É a primeira vez que essa intuição tem amostra — a série mede a mesma
coisa com 22 contra 18 pernas. **Não aciona a hipótese pré-registrada**, que é
sobre CLV de pernas e tem gatilho próprio (seção 9).

**Calibração: ECE 0,0240** no teste, contra 0,0127 em setembro. Piorou, mas
segue na casa dos dois pontos percentuais; nas lutas com odds o ECE é 0,1340,
porque ali a amostra por faixa é pequena.

## 7. As duas odds e o sinal de entrada

Mudança de out/2026, a pedido: a odd que decide uma entrada passou a ser a da
**Thunderpick**, onde a aposta sai de verdade.

| coluna | fonte | trabalho |
|---|---|---|
| `odds_a/b_decimal` | mediana de 20-25 casas | define o favorito do **mercado** — o benchmark |
| `odds_a/b_tp` | Thunderpick | é a odd no card, no **EV**, no critério EV>1 e no **P&L** |

E um sinal novo, `ev_sharp_tp = sharp_prob × odd da casa`: *a nossa casa paga
acima do preço justo da Pinnacle?* O `ev_sharp` antigo responde outra coisa —
usa a melhor odd entre as casas da API, que não inclui a Thunderpick, então
marca valor num livro onde não se aposta. No card de 31/out: 4 pernas EV>1,
3 lutas com `ev_sharp>1`, **1 com `ev_sharp_tp>1`**.

A marca aparece nos **dois lados** do card, porque a entrada não se importa com
qual lutador o modelo prefere — e o primeiro card já mostrou por quê: em
Moicano vs Nolan o modelo aponta Nolan, mas quem tem valor na casa é o Moicano.

Os dois cards medidos até agora empatam sobre se a casa ajuda: no de 17/out ela
teria pagado menos (+0,10u contra +1,75u na mediana), no de 24/out pagou mais
(−0,39u contra −1,31u).

## 8. Saúde dos dados e instrumentação

A varredura de 17/out passou limpa nos artefatos: 180 linhas do histórico, 37
commits auditados, **nenhuma linha fechada alterada depois de fechada**. Os
oito defeitos encontrados desde setembro estavam todos no código, não nos
dados.

O mais grave era familiar: **a odd da casa era apagada ao regerar o relatório**
— mesmo modo de falha do sinal sharp, terceira vez que a mesma classe aparece
(coluna que o `record` não sabe preservar).

Alarmes que não existiam em setembro:

- **card sem pré-registro** (`src/event_watch.py`), nos dois estados: "a
  registrar" quando a API lista o card e "perdido" quando já passou. Nasceu por
  causa do UFC 332 e já pegou todos os cards seguintes;
- **teto de odd** (100,0): `odd > 1.0` sozinho aceitava infinito, e um EV
  infinito poria a perna no topo da aba EV>1 sem aviso;
- **simetria do espelho** nas colunas `_sum`, e consistência entre card,
  template e histórico.

Cinco tabelas de apelido já foram necessárias (Alatengheili, Loopy Godinez,
Timmy Cuamba, Joe Kropschot, Katlyn Cerminara). O padrão é sempre o mesmo: a
fonte usa outro NOME, não outra grafia, e nenhum score de string casa.

## 9. O que está medido e o que ainda é chute

**Medido:**
- o modelo é pior que o mercado, e por quanto;
- o gargalo é resolução, não calibração;
- mais capacidade de modelo piora;
- `opp_quality_diff` não passou; **categoria de peso não passou**;
- **feature simétrica não pode passar** — é estrutural, não empírico;
- **o modelo é muito pior em luta com lutador de pouco histórico** (AUC 0,631
  contra 0,708), e isso é quase metade do teste.

**Não medido (e registrado como tal):**
- se as pernas com lutador de pouca experiência são piores — **hipótese
  pré-registrada**, gatilho em 60 pernas com CLV por grupo; hoje **15 e 16**,
  faltam ~20 eventos;
- se o CLV positivo persiste — 84 pernas, oito eventos de ~10;
- se o grupo com respaldo sharp se sai melhor — 11 contra 33 pernas, e a
  direção está *invertida* em relação à intuição (com sharp +0,30u em 11, sem
  sharp +6,66u em 33);
- se `ev_sharp_tp` separa melhor que `ev_sharp` — começa a medir no evento 16.

## 10. O que vale e o que não vale fazer

**Não vale:** mexer no modelo. As três regras de parada continuam válidas, e
agora há um motivo a mais: a lista de candidatas de feature documentadas está
**vazia**, e uma classe inteira de candidatas futuras é provadamente inerte.

**Não vale:** usar o diagnóstico da seção 6 para aplicar o corte de experiência
antes do gatilho. São perguntas diferentes, e afrouxar limiar com o resultado
na tela é a regra zero.

**Vale:** deixar a série rodar. Continua sendo o item de maior valor e o que
exige menos trabalho. Todas as perguntas em aberto dependem do mesmo recurso:
eventos.

**Vale:** resolver a fonte do card. No UFC 333 o UFCStats estava incompleto e
três lutas reais ficaram de fora; no evento 12 ele salvou duas lutas fantasma
que a API listava. A regra "UFCStats é a autoridade" não é de graça, e a união
das duas fontes com desempate explícito é a única melhoria operacional concreta
que restou.

---

## Uma leitura do conjunto

O projeto ficou **muito melhor em medir do que em prever**, e isso é
deliberado. A diferença em relação a setembro é que agora a parte "prever" tem
uma resposta fechada: as features documentadas acabaram, e a última delas
falhou com explicação estrutural, não com "faltou pouco".

O que mudou de verdade nestas sete semanas foi a **distância entre medir e
apostar**. O projeto passou a saber qual odd mede o mercado e qual odd decide
uma entrada, e a mostrar na página o único sinal que não depende do próprio
modelo. Isso não torna o modelo melhor — e o balanço continua dizendo que ele
é pior que o mercado — mas torna a decisão de entrar menos cega.

A pergunta honesta não é "como melhorar o modelo", é **"o que a série vai dizer
em seis meses"**. A diferença para setembro é que agora dá para nomear as três
coisas que ela vai responder, com gatilho escrito para cada uma.
