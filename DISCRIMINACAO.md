# Por que o modelo não separa luta desequilibrada

Análise fechada em 24/08/2026, combinada desde 16/08. Diagnóstico do
problema central do preditor de vencedor: ele é **bem calibrado e quase não
discrimina**.

Todos os números abaixo saem de `cal_select` ou da amostra de mercado, nunca
do teste final — a mesma regra anti-overfit do resto do projeto.

## O sintoma

Só **3,1%** das previsões do modelo passam de 75%, contra um mercado que
chega a 92%. Em favorito pesado a distância é gritante: Marcio Barbosa saiu
54,0% no modelo contra 86,4% no mercado.

Antes de tudo, uma coisa que **não** é o problema: a calibração está ótima.
No teste completo o erro por faixa fica entre −2,3pp e +2,3pp — quando o
modelo diz 75%, sai 75,3%. Não há bug de calibração para consertar.

## A medida certa: AUC

AUC mede separação **independente de calibração** — é a probabilidade de o
modelo dar nota maior ao vencedor que ao perdedor. Nas 821 lutas com odds
reais (2021-08 a 2023-09):

| | AUC | Brier | log loss |
|---|---|---|---|
| modelo (logreg) | **0,5400** | 0,2358 | 0,6638 |
| modelo (GBM) | 0,5368 | 0,2376 | 0,6678 |
| **mercado** | **0,6349** | 0,2076 | 0,6026 |

0,54 é quase moeda. Na tarefa "este favorito vence?", o modelo praticamente
não ordena.

### A decomposição que fecha o diagnóstico

O Brier se decompõe em `incerteza − resolução + confiabilidade`, onde
resolução é o poder de separar e confiabilidade é o erro de calibração:

| | resolução | confiabilidade |
|---|---|---|
| modelo | 0,0021 | 0,0201 |
| mercado | **0,0118** | **0,0009** |

O mercado tem **5,6× mais resolução**. É esse o buraco — não a calibração.

## O teto são as FEATURES, não o modelo

Teste direto: se um modelo muito mais flexível não passar da logreg, o
limite está no que as features carregam.

| modelo | AUC | log loss | previsões > 75% |
|---|---|---|---|
| logreg (produção) | **0,6571** | 0,6543 | 2,46% |
| GBM padrão | 0,6373 | 0,6626 | 1,92% |
| GBM alta capacidade | 0,5764 | 0,7358 | 24,50% |
| GBM capacidade extrema | 0,5666 | 1,0219 | 61,83% |

Mais capacidade piora. E repare nas duas últimas linhas: **o GBM extremo
produz 61,8% de previsões acima de 75% com AUC de 0,567** — ele fica
confiante e errado. Isso prova que "previsão mais confiante" não é
"discriminação melhor", e é exatamente por isso que o critério de aceite
exige as duas coisas juntas.

## De onde vem o pouco que existe

AUC treinando só com subconjuntos de features:

| features | AUC |
|---|---|
| só `age_diff_years` | **0,6207** |
| só `elo_diff` | 0,5844 |
| só `reach_diff_cm` | 0,5063 |
| `elo` + idade | 0,6378 |
| tudo **menos** `elo_diff` | 0,6484 |
| **as 16** | **0,6571** |

**Idade sozinha carrega quase tudo.** As 16 features juntas entregam 0,657
contra 0,621 só da idade — as outras quinze somam ~0,036 de AUC. Envergadura
é ruído puro. O Elo, que parecia a feature nobre, entrega menos que a idade.

## Candidato testado e REJEITADO

Hipótese: as features não codificam **qualidade do adversário**. Um cartel
5-0 contra fracos é idêntico a 5-0 contra contenders.

Construí `opp_quality_diff` — média do Elo dos oponentes já enfrentados,
point-in-time, mesma disciplina anti-vazamento do resto. Cobertura de 74,7%,
correlação de 0,604 com o próprio Elo (informação distinta, não redundante).

Resultado em `cal_select`:

| | log loss | previsões > 75% |
|---|---|---|
| atual (16 features) | 0,6543 | 2,46% |
| + `opp_quality_diff` | 0,6532 | 2,46% |

Pelo critério de aceite do README: melhora o log loss, **não melhora a
discriminação** → **rejeitada**. É precisamente o caso que a regra foi
escrita para pegar — um modelo mais bem calibrado na mesma ignorância — e
ela pegou no primeiro candidato.

Artefato em `data/processed/opp_quality.csv`, caso alguém queira retomar.

## Conclusão

O modelo faz bem o que dá para fazer com o que tem, e o que tem é pouco:
**idade, mais um resto que quase não soma**. A discordância dele com o
mercado em favorito pesado é ignorância bem calibrada, não insight — e
apostar nela custou −14,3% por perna em 230 lutas do backtest.

Três consequências práticas:

1. **Parar de mexer no modelo e nos hiperparâmetros.** O teto não está lá.
   O GBM de capacidade extrema é a prova.
2. **Mais feature derivada das mesmas colunas provavelmente não resolve.**
   `opp_quality_diff` era o melhor candidato dessa família e não passou.
3. **O que faltaria é informação de outra natureza** — estilo, contexto de
   camp, lesão, corte de peso —, que não existe em fonte gratuita e estável.
   Sem isso, a expectativa honesta é que o modelo continue abaixo do mercado.

Isso não invalida o projeto: valida a instrumentação. As regras de parada
existem para transformar isso em decisão em vez de mais uma rodada de
tuning.

---

# Hipótese pré-registrada: pernas EV>1 com lutador de pouca experiência

**Registrada em 06/09/2026, com 9 eventos fechados e 40 pernas.** Escrita
*antes* de a amostra existir justamente para que a decisão não seja tomada
com o resultado na tela — mesma disciplina do resto do projeto.

## De onde veio

Dois casos chamativos em eventos seguidos: Dan Hooker (o modelo dava 70,1%
contra 19,0% do mercado, porque Salahdine Parnasse não tinha uma linha na
base) e, no evento seguinte, o que *parecia* o mesmo problema mas era bug de
homônimo. A suspeita: previsão apoiada em perfil sintético de estreia vira
perna EV>1 e envenena o paper trading.

## O que os dados dizem HOJE (baseline, não conclusão)

Experiência reconstruída no momento da luta — lutas anteriores de cada lado
na base, com data anterior ao evento. Limiar: `MIN_FIGHTS_FOR_RELIABLE_STATS`
= 3.

| grupo | acerto | P&L | CLV médio |
|---|---|---|---|
| alguém com < 3 lutas prévias | 9/22 = 41% | −4,36u | (2 medidas) |
| ambos com ≥ 3 | 11/18 = 61% | +3,59u | +0,46 pp (5 medidas) |
| — subgrupo estreante puro (0 lutas) | 3/8 | −1,12u | +0,74 pp (4 medidas) |

Seleção: 46% de **todas** as lutas previstas têm alguém abaixo de 3 lutas;
entre as pernas EV>1, 55%.

**Nada disso é significativo.** Fisher exato: p = 0,341 no desempenho,
p = 0,160 na seleção. Com 22 e 18 pernas, os dois resultados são compatíveis
com moeda. A direção bate com a intuição, e é exatamente por isso que o
critério precisa estar escrito antes.

## O gatilho

Quando houver **pelo menos 60 pernas EV>1 fechadas com CLV medido em cada
grupo**:

- **Teste primário — CLV.** Mann-Whitney bicaudal entre o CLV das pernas com
  pouca experiência e o das demais. É o CLV e não o P&L porque a regra do
  projeto é explícita: P&L não é variável de decisão, é binário e dominado
  por variância.
- **Teste secundário — acerto.** Fisher exato bicaudal sobre ganhas/perdidas,
  só como sanidade.

**Se o primário der p < 0,05 E a direção for "pouca experiência pior":** o
critério de pré-registro EV>1 passa a exigir os dois lados com 3+ lutas
prévias. Pernas fora disso continuam aparecendo no relatório, com o aviso que
já existe, mas não entram no paper trading.

**Em qualquer outro caso — inclusive p entre 0,05 e 0,10 — nada muda**, e o
assunto só é reaberto ao dobrar a amostra. "Faltou pouco" não é motivo, como
diz a regra zero.

## Por que 60, e o que isso custa

Para a diferença observada (41% vs 61%) atingir significância seriam
necessárias ~100 pernas por grupo; 60 detecta um efeito maior que esse com
folga, e é o que cabe em prazo humano. No ritmo atual (~4,4 pernas por
evento, ~55% delas do grupo fino), são cerca de **25 eventos**, algo como seis
meses.

Vale saber o que a mudança custaria se for acionada: pouca experiência não é
caso de borda neste esporte — é quase metade do card. Aplicar o corte hoje
removeria mais da metade das pernas, e a série passaria a levar o dobro do
tempo para dizer qualquer coisa sobre qualquer outra pergunta.

## O argumento que não depende disso

Existe uma linha separada, que não precisa de amostra: previsão apoiada em
perfil **sintético** não é estimativa de probabilidade, é um marcador de "não
sei" com cara de número. Esse argumento é estrutural e justificaria excluir
apenas o **estreante puro** (zero lutas) — 8 das 40 pernas, não 22. Se um dia
for adotado, que seja por esse motivo e declarado como tal, não pelo
resultado das 8.
