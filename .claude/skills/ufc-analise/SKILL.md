---
name: ufc-analise
description: >-
  Modeling and measurement conventions for the ufc_predictor UFC win-probability
  model (UFCStats data, logreg + LightGBM): how features are built point-in-time,
  how the mirrored rows and the chronological split work, how the model is
  evaluated against the betting-market baseline, and the two criteria a new
  feature must meet before it can be accepted. Use this skill whenever the user
  touches the model itself — feature nova, treino, re-treinar, backtest,
  calibração, discriminação, AUC, log loss, Brier, Elo, hiperparâmetro, tuning,
  vazamento temporal — or asks whether the model is good, why it disagrees with
  the market, or whether some change is worth making. Trigger it even for a
  casual "será que melhora se eu adicionar X?", because the acceptance criteria
  and the anti-overfit split discipline are exactly what such a question needs.
---

# Modelo: convenções e critérios

**O objetivo é um modelo honesto e bem calibrado de previsão de luta.** O paper
trading existe como régua de realidade, não como finalidade — o projeto segue
valendo mesmo se nunca houver edge sobre o mercado. Isso muda o que conta como
sucesso: calibração e discriminação honestas valem mais que P&L positivo.

## Como as features são construídas

`src/features.py`. Regras não óbvias:

- **Point-in-time.** Toda estatística usa `shift(1).expanding()` — só lutas
  anteriores àquela data. Usar média de carreira atual para prever uma luta de
  2015 seria vazamento do futuro.
- **Sempre diferenciais** (`fighter_a − fighter_b`), nunca valor absoluto.
- **Cada luta gera DUAS linhas espelhadas** (A−B e B−A) com o mesmo `fight_id`,
  para o modelo não aprender viés de ordem. O projeto não usa "red/blue corner":
  são `fighter_1/2` do UFCStats espelhados em A/B.
- **`stance_mismatch` é a exceção deliberada**: simétrica (1 se divergem), não
  inverte sinal na linha espelhada.
- **Taxas de finalização** são sobre o total de lutas anteriores, não sobre as
  vitórias — embutem estilo, não só eficácia.
- **Empate/NC saem do dataset** (`dropna(subset=["label"])`); método não
  categorizável (DQ, overturned) vira `None` e serve só como feature.
- **Poucas lutas não é descarte**: vira flag `low_experience` e o valor faltante
  é imputado (logreg) ou tratado nativamente (GBM).

## Split e disciplina anti-overfit

`src/train.py`. 70/15/15 **cronológico**, agrupado por `fight_id` para as duas
linhas espelhadas nunca caírem em lados diferentes — seria vazamento sutil mas
real.

A fatia de calibração é subdividida em `cal_fit` / `cal_select`. **Toda escolha
— método de calibração, K do Elo, feature nova — é decidida em `cal_select`.**
O teste final roda uma vez só, depois. Otimizar olhando o teste seria otimizar
exatamente o número reportado como avaliação.

Detalhe: a fatia de calibração de produção se sobrepõe à janela do backtest de
mercado, então os experimentos de `src/tuning.py` rodam sobre o dataset truncado
em 2023-09-16.

## Avaliação e baselines

```bash
python -m src.evaluate      # log loss, Brier, acurácia no teste
python -m src.market_odds   # 821 lutas com odds reais
```

- **Baseline do acaso**: 50% / 0.693 de log loss.
- **Baseline do favorito das odds**: existe **só** em `market_odds.py`, que loga
  a taxa de acerto do favorito na amostra (68,0%). O `evaluate.py` não tem
  baseline de favorito no teste principal.
- **O mercado é o teto de referência** e ganha: 0.502 contra 0.619 de log loss
  nas lutas casadas. Isso é o resultado esperado, não um bug.

## Critério de aceite de uma feature nova

Medido em `cal_select`, nunca no teste. **Precisa melhorar as duas coisas:**

1. log loss do preditor de vencedor;
2. a fração de previsões acima de 75% — hoje ~3%, contra um mercado que chega a 92%.

Só o log loss significaria um modelo mais bem calibrado na mesma ignorância. A
prova de que os dois critérios são necessários: um GBM de capacidade extrema
produz 61,8% de previsões acima de 75% com AUC de 0,567 — confiante e errado.

## O diagnóstico atual (ver `DISCRIMINACAO.md`)

O modelo é **bem calibrado e quase não discrimina**. AUC 0,540 contra 0,635 do
mercado; resolução 5,6× menor. **O teto são as features, não o modelo** — mais
capacidade piora. Idade sozinha entrega AUC 0,621 das 0,657 do conjunto inteiro.

Não sugira mexer em hiperparâmetro ou classe de modelo para resolver isso: já
foi medido e não está lá.

## Já testado e rejeitado — não re-sugerir

- **`opp_quality_diff`** (qualidade do adversário, média do Elo dos oponentes já
  enfrentados): melhorou log loss, não mexeu na discriminação → rejeitado.
- **Margem por método no Elo** (`ELO_METHOD_MULTIPLIERS`): nenhum esquema bateu o
  Elo simples em `cal_select`.
- **Previsão de duração / faixa de round**: removida em ago/2026 — margem mínima
  sobre o baseline e probabilidades não congeláveis no pré-registro.
- **Categoria de peso** (out/2026): testada em `cal_select` nas três formas —
  `weight_lb` puro, com `is_womens`, e como interação com `elo_diff`,
  `age_diff_years` e `reach_diff_cm`. **Nenhuma melhora nada**: o efeito
  principal dá log loss idêntico a 4 casas e as interações *pioram*
  (+0,0003 a +0,0022). A fração acima de 75% não se move em caso nenhum.
  Era a última lacuna de feature da lista.

## Feature SIMÉTRICA não informa o preditor de vencedor

Descoberto ao explicar o nulo *exato* do peso. Cada luta vira duas linhas
espelhadas com labels opostos; uma coluna que vale o mesmo nas duas é
incapaz de separá-las, e o coeficiente vai a zero — ela só desloca o
intercepto. Na logreg de produção **`stance_mismatch` tem coeficiente
0,000000**: é peso morto, e foi assim o tempo todo.

(As flags `low_experience` não caem nisso: elas *trocam entre si* no espelho
— idênticas em só 68% das lutas — e os coeficientes saem exatamente opostos,
±0,106. O par funciona como um diferencial disfarçado.)

Consequência para qualquer candidata futura que seja propriedade da LUTA e
não de um lado: ela só entra como **interação com um diferencial**
(diff × simétrica é antissimétrica, logo informativa), nunca como efeito
principal. Travado em `tests/test_feature_simetrica.py`.

## Calibração

`src.evaluate.calibration_table` / `expected_calibration_error` dão a curva por
faixa; `python -m scripts.calibration_report` desenha e publica.

**Medido em 06/09/2026: ECE 0,0127 no teste.** A calibração é boa — os desvios
ficam em poucos pontos percentuais onde há dado. O que salta é a concentração:
52% das linhas caem entre 40% e 60% e as pontas ficam quase vazias. É o
diagnóstico do `DISCRIMINACAO.md` por outro ângulo — o modelo não mente sobre
a própria confiança, ele quase nunca tem confiança para declarar.

Faixa pequena mente: abaixo de 20 observações o desvio é ruído, e o
`log_calibration` avisa. Leia o ECE, não as linhas.

## Lacunas em aberto

Não são decisões tomadas — são coisas que ninguém tratou ainda:

Nenhuma lacuna de feature conhecida em aberto. As duas que estavam listadas
foram fechadas: o **corte por época** virou `TRAINING_ERA_START` (Regras
Unificadas, 2001) e a **categoria de peso** foi testada e rejeitada (acima).

O que resta não é falta de ideia, é o diagnóstico: o teto são as features que
existem, e nenhuma das candidatas documentadas mexeu nele.

## Onde o modelo é fraco (diagnóstico, out/2026)

Medido no conjunto de teste (1322 lutas), por grupo de experiência — leitura,
**não seleção**: nada foi escolhido olhando estes números.

| grupo | lutas | AUC | log loss | acerto |
|---|---|---|---|---|
| alguém com < 3 lutas prévias | 577 | **0,631** | 0,666 | 57,9% |
| ambos com 3+ | 745 | **0,708** | 0,631 | 66,2% |

A discriminação do modelo está quase toda no segundo grupo, e o primeiro é
**43,6% do teste**. O AUC global de 0,676 é a mistura dos dois.

**Isto NÃO aciona a hipótese pré-registrada** e não pode: aquela mede o CLV
de pernas EV>1, com gatilho em 60 pernas por grupo, e está em 15 e 16. São
perguntas diferentes — habilidade do modelo aqui, lucratividade de perna lá.
Ler este resultado como permissão para aplicar o corte antes do gatilho é
exatamente a trapaça que a regra zero descreve.

Limitação conhecida: não dá para saber se o mercado também é pior nesse
grupo (só 99 lutas do teste têm odds), então o gap pode ser do esporte e não
das nossas features.

## Hipótese pré-registrada (não reabrir sem o gatilho)

**Pernas EV>1 com lutador de pouca experiência** — a suspeita de que previsão
apoiada em perfil sintético envenena o paper trading. Medida em 06/09/2026:
41% de acerto contra 61%, mas **p = 0,341** — a direção bate com a intuição e a
amostra não sustenta nada.

O critério de decisão já está escrito em `DISCRIMINACAO.md`, com gatilho
(60 pernas com CLV por grupo), teste (Mann-Whitney sobre CLV) e a ação exata.
**Se este assunto voltar, leia de lá em vez de reanalisar** — reanalisar com o
resultado na tela é como o limiar acaba afrouxado. Faltam ~25 eventos.

## Regras de parada

Estão no `README.md` e foram escritas **antes** dos dados existirem. A regra zero:
afrouxar um limiar depois de ver o resultado é trapaça; mudança exige motivo no
commit, e "faltou pouco" não é motivo. Consulte-as antes de concluir qualquer
coisa a partir da série — em especial, **P&L não é variável de decisão** e
acurácia contra o mercado é sanidade, não decisão.
