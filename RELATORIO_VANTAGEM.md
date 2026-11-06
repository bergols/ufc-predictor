# Relatório: o modelo acrescenta informação ao mercado?

**Resultado: NEGATIVO nesta amostra.** O peso ótimo do modelo saiu **zero** na
fatia de desenvolvimento, e com isso a combinação é idêntica ao mercado. O
baseline continua em produção, sem alteração.

Protocolo `vantagem-v1`, pré-registrado em `PROTOCOLO_VANTAGEM.md` **antes** de
qualquer comparação. Branch `pesquisa-vantagem`, a partir de `b29b1b4`.
Reproduzível com um comando:

```bash
python -m scripts.experimento_vantagem
```

---

## 1. Diagnóstico: o que existe, o que é dado e o que foi reproduzido

Três coisas diferentes, separadas de propósito.

| | situação |
|---|---|
| **código disponível** | pipeline completo: `src/features.py`, `src/train.py`, `src/evaluate.py`, `src/market_odds.py`, `src/tuning.py` |
| **dados disponíveis** | base do UFCStats (8 974 lutas), features (`fight_features.csv`), odds históricas (`market_odds.csv`, 7 340 linhas), pré-registro ao vivo (207 linhas) |
| **resultados efetivamente reproduzidos nesta rodada** | só os deste relatório. Os números de `DISCRIMINACAO.md` e do `src/evaluate.py` foram **lidos**, não re-executados |

**Modelo em produção:** `logreg` (`config.MODELO_PRODUCAO`), regressão logística
calibrada por sigmoid, em `models/logreg_calibrated.joblib`. O LightGBM existe
treinado e não é o de produção. **Nada foi alterado.**

**Features e fontes:** 100% derivadas do UFCStats, sempre diferenciais
(`fighter_a − fighter_b`) e sempre *point-in-time* (`shift(1).expanding()`).
Cada luta gera duas linhas espelhadas com o mesmo `fight_id`.

**Separações temporais existentes** (de `models/training_metadata.json`): 70/15/15
cronológico agrupado por `fight_id` — treino 2001-02-23 a 2021-09-25 (5 911
lutas), calibração 1 322, teste 2024-05-04 a 2026-10-31 (1 324).

> O conjunto de teste **já foi consultado muitas vezes** neste projeto. Trato-o
> como **desenvolvimento**. Não é teste intocado e não é validação prospectiva,
> e não foi usado para medir nada aqui.

**Probabilidades fora da amostra:** `data/processed/test_predictions.csv` (2 648
linhas espelhadas do teste) e, o que importa para esta pergunta, as
probabilidades **congeladas antes de cada luta** em
`data/prediction_history.csv`.

---

## 2. Auditoria das odds: a condição que mudou o desenho

### 2.1 `data/raw/market_odds.csv` não serve para decisão

Medido:

| achado | valor |
|---|---|
| `odds_extract_ts` distintos em todo o arquivo | **1** (`2023-12-28 11:40:25`) |
| atraso entre a extração da odd e o evento | mínimo **+103 dias**, mediana **+1 734**, máximo **+3 338** |
| linhas extraídas **antes** do evento | **0** |
| coluna de casa | **nenhuma** |
| coluna de mercado | **nenhuma** |
| odds por lado | um par `favourite_odds` / `underdog_odds` |

Não há prova de que esses preços estiveram disponíveis antes da luta, nem de
qual casa vieram. **Nenhum backtest de rentabilidade foi produzido sobre eles**,
e nenhum seria válido. Servem como referência de nível de mercado em análise
exploratória rotulada.

### 2.2 `market_comparison.csv` é desenvolvimento

821 lutas, 2021-08-21 a 2023-09-16. **Zero** dentro da janela de teste do
modelo, e a janela se sobrepõe ao treino e à calibração de produção — o próprio
`src/tuning.py` trunca o dataset em 2023-09-16 por causa disso. As
probabilidades de modelo ali são **parcialmente dentro da amostra**.

### 2.3 O único conjunto temporalmente adequado

`data/prediction_history.csv`: previsão congelada antes da luta, preço capturado
antes da luta. **179 lutas elegíveis em 16 eventos**, 2026-07-11 a 2026-10-31,
com **28 exclusões** registradas uma a uma em
`data/experimentos/exclusoes_vantagem-v1.csv`.

Cobertura, medida:

| | lutas |
|---|---|
| resolvidas | 187 |
| elegíveis (resultado + previsão + os dois lados da mediana + estado) | **179** |
| com Pinnacle devigada (`sharp_prob`) | 122 |
| com fechamento | 96 |
| **com preço executável nos dois lados** (Thunderpick) | **19** |

### 2.4 Retirada da margem

`p_A = (1/odd_A) / ((1/odd_A) + (1/odd_B))` — normalização proporcional,
`src.utils.remove_vig_two_way`. É o método mais transparente e **não** é o
único: "power" e "Shin" distribuem a margem de forma desigual entre favorito e
azarão. A escolha está no protocolo, feita antes dos resultados.

### 2.5 Duas coisas que nunca se misturam

- **(a) referência do mercado**: mediana de 20–28 casas, devigada. É o candidato A.
- **(b) preço executável**: a odd da casa onde a aposta seria feita.

**A mediana de casas não é cotação executável.** Ninguém aceita uma aposta na
mediana de 25 casas. Nenhum retorno foi calculado sobre ela.

### 2.6 Limitação temporal declarada, não contornada

O pré-registro guarda a **data do evento**, não o instante da gravação.
`criado_em` por versão existe desde 2026-11-06; as 207 linhas anteriores foram
semeadas com temporalidade **não recuperável**. Esse texto viaja em **cada
linha** de `data/experimentos/previsoes_vantagem-v1.csv`. Nenhum horário foi
inventado.

---

## 3. O experimento

Três candidatos, mesma população, mesmos horários:

- **A. Mercado** — implícita devigada da mediana;
- **B. Modelo** — `model_prob_a` **como foi congelada**, sem recálculo. Recalcular
  hoje usaria uma base que já contém o resultado daquela luta;
- **C. Combinação** — pooling logístico, uma única formulação escolhida antes:
  `logit(p_C) = (1−w)·logit(p_A) + w·logit(p_B)`, `w ∈ [0,1]`, estimado por log
  loss **só no desenvolvimento**, grade de 0,01.

Divisão cronológica por evento: **8 eventos de desenvolvimento** (2026-07-11 a
2026-08-29, 91 lutas) e **8 de avaliação** (2026-09-05 a 2026-10-31, 88 lutas).
Nenhum evento parte entre fatias.

---

## 4. Resultado

### 4.1 O peso saiu zero

| `w` | log loss no desenvolvimento |
|---|---|
| **0,00** | **0,558859** |
| 0,05 | 0,560162 |
| 0,10 | 0,561672 |
| 0,20 | 0,565331 |
| 0,30 | 0,569870 |
| 0,50 | 0,581722 |
| 0,70 | 0,597467 |
| 1,00 | 0,628789 |

A curva é **estritamente crescente a partir de zero**. Não é um platô onde o
mínimo caiu na borda por acidente: qualquer peso no modelo piora a previsão, e
o gradiente é claro desde o primeiro passo da grade.

Com `w = 0`, **C é identicamente A**.

### 4.2 Contraste confirmatório

> **Δ log loss (C − A) na avaliação = +0,0000**, IC 95% [−0,0000, +0,0000].
> **Veredito: negativo.**

A diferença é exatamente zero por construção, porque o procedimento escolheu
não usar o modelo. Essa é a resposta, não um defeito do contraste: a formulação
foi desenhada para que `w = 0` fosse alcançável, e foi alcançado.

### 4.3 Métricas por fatia

| fatia | candidato | log loss | Brier | AUC |
|---|---|---|---|---|
| desenvolvimento (91) | mercado | **0,5589** | 0,1870 | 0,7832 |
| | modelo | 0,6288 | 0,2175 | 0,7079 |
| | combinação | 0,5589 | 0,1870 | 0,7832 |
| avaliação (88) | mercado | **0,6660** | 0,2360 | 0,6234 |
| | modelo | 0,7057 | 0,2558 | 0,5358 |
| | combinação | 0,6660 | 0,2360 | 0,6234 |

### 4.4 Exploratório: modelo contra mercado

**Δ log loss (B − A) na avaliação = +0,0398**, IC 95% **[−0,0472, +0,1264]**.

O modelo é pior na ponta, e **o intervalo contém zero**. Então esta amostra
*também* não estabelece que o modelo é pior — ela não estabelece quase nada. O
que está estabelecido é o resultado do procedimento pré-registrado: no
desenvolvimento o peso ótimo foi zero.

**Ausência de significância não é prova de equivalência.** A meia-largura do
intervalo é **0,0868**: esta amostra só separaria de zero um efeito maior que
~0,087 de log loss. É um efeito enorme — maior que a diferença inteira entre
modelo e mercado nas 821 lutas de 2021-23 (0,619 contra 0,502, ou 0,117).

### 4.5 Estabilidade entre janelas: ruim

O mercado piora de 0,5589 para 0,6660 de log loss e de 0,7832 para 0,6234 de
AUC entre as duas fatias. A janela de avaliação teve mais zebras. Por evento, o
log loss do mercado varia de **0,4053** (25/jul) a **0,7964** (19/set) — quase o
dobro.

**O desvio por evento da diferença (modelo − mercado) na avaliação é 0,1344.**
Esse número é o que decide quantos eventos faltam, na §6.

### 4.6 Grupos de experiência prévia (exploratório, n pequeno)

Avaliação, grupo do lutador **menos** experiente, contagem *point-in-time*:

| lutas prévias | n | log loss mercado | AUC mercado | log loss modelo | AUC modelo |
|---|---|---|---|---|---|
| 0 | 8 | 0,4729 | 0,8333 | 0,7220 | 0,5000 |
| 1–2 | 25 | 0,7417 | **0,4779** | 0,7979 | **0,2721** |
| 3–5 | 28 | 0,6336 | 0,6889 | 0,6423 | 0,6667 |
| 6+ | 27 | 0,6866 | 0,5667 | 0,6813 | 0,6000 |

No grupo 1–2 **os dois** ficam em torno ou abaixo do acaso — o mercado em 0,478
e o modelo em 0,272. Com 25 lutas, um AUC de 0,27 é perfeitamente compatível com
ruído, e é exatamente o tipo de número que não se interpreta. **Não atribuo isso
a mecanismo nenhum** sem ablação que o teste, e não acionei nada a partir dele.

Observação do `DISCRIMINACAO.md` que continua valendo: não se sabe se o mercado
também é pior nesse grupo em amostra grande.

### 4.7 Calibração

Mercado, avaliação (com contagem e IC de Wilson):

| faixa | n | prev. média | observado | IC 95% |
|---|---|---|---|---|
| [0%,20%) | 3 | 0,164 | 0,000 | [0,000, 0,562] |
| [20%,40%) | 21 | 0,299 | 0,476 | [0,283, 0,676] |
| [40%,50%) | 10 | 0,457 | 0,600 | [0,313, 0,832] |
| [50%,60%) | 17 | 0,544 | 0,765 | [0,527, 0,904] |
| [60%,80%) | 33 | 0,676 | 0,606 | [0,437, 0,753] |
| [80%,100%) | 4 | 0,842 | 0,750 | [0,301, 0,954] |

Toda faixa tem menos de 35 observações e os intervalos são largos demais para
sustentar leitura. **Abaixo de ~20 observações o desvio é ruído** — a tabela
está aqui com as contagens justamente para que ninguém leia a coluna "observado"
sem elas.

### 4.8 Retorno: não calculado, e por quê

Preço executável com temporalidade verificável existe em **19** das 179 lutas
(nenhuma antes de out/2026, porque a casa só entrou no pipeline então), e **19
na avaliação**. Não sustenta retorno, drawdown, concentração nem sensibilidade.
**Nenhum P&L foi produzido.** A política de simulação está registrada no
protocolo §8 e implementada no modo sombra, para quando houver dados.

Duas quantidades, nunca trocadas: `p × odd` é **retorno bruto esperado por
unidade** (1,26 = receber 1,26 por 1 apostado); `p × odd − 1` é **lucro esperado
líquido por unidade** (0,26 = ganhar 0,26 por 1 apostado).

CLV não foi usado como prova de nada — é diagnóstico complementar.

---

## 5. Modo sombra

Fluxo separado da produção, append-only, em `data/experimentos/sombra/`.

```bash
# ANTES das lutas: registra os tres candidatos, precos e provas
python -m scripts.sombra --event-date 2026-11-07

# so olhar, sem gravar
python -m scripts.sombra --event-date 2026-11-07 --dry-run

# DEPOIS: anexa o resultado, sem reescrever o que foi previsto
python -m scripts.sombra --event-date 2026-11-07 --liquidar
```

Cada linha grava: versão do protocolo, peso `w`, fatia
(`desenvolvimento` / `avaliacao_prospectiva`), confronto e orientação,
`prediction_id` da versão imutável, modelo, `manifesto_sha256`,
`modelos_sha256`, as três probabilidades, o lado, a mediana e o preço
executável daquele lado, a referência da prova, casa, mercado, instante de
leitura do preço, início programado, retorno bruto esperado, lucro líquido
esperado, seleção ou recusa **com motivo**.

A liquidação entra em `sombra_resultados.csv`, que aponta para a linha prevista
pelo `registro_id` e **não toca** a previsão. Perna recusada liquida com
`unidades = 0`: recusa não é aposta perdida, é aposta que não houve.

Ensaio no card de 2026-11-07 (`--dry-run`, nada gravado): 14 lutas, **1 seleção
e 13 recusas** — 12 por retorno bruto esperado não passar de 1 e 1 por não haver
preço executável com prova. Com `w = 0` a política usa a probabilidade do
mercado, então a seleção única (Billy Elekana, odd 3,75, retorno bruto esperado
**1,0156**) é a casa pagando um fio acima da referência do mercado.

> Isso é **diferente** das "13 seleções elegíveis" do painel do site. Lá a
> probabilidade é a do nosso modelo e o critério é elegibilidade de prova; aqui
> a probabilidade é a do candidato escolhido pelo procedimento. Perguntas
> diferentes, números diferentes, e misturá-los seria o erro.

Nada agendado, nenhuma API consumida, nada publicado.

---

## 6. Quantos eventos faltam

Do desvio por evento medido na avaliação (**0,1344** na diferença modelo −
mercado), para separar um efeito de zero com 95%:

| efeito em log loss | eventos necessários | tempo aproximado |
|---|---|---|
| 0,05 | **~28** | ~7 meses |
| 0,03 | ~78 | ~1,5 ano |
| 0,02 | ~174 | ~3,3 anos |
| 0,01 | ~695 | ~13 anos |

A série tem **16** eventos. Com 8 na avaliação, a amostra atual detecta ~0,087 —
mais que a diferença total entre modelo e mercado já medida em amostra grande.

**Um fim de semana não demonstra nada**, qualquer que seja o resultado. E
nenhuma quantidade de eventos transforma um peso zero em vantagem: o que os
eventos compram é a capacidade de detectar um efeito pequeno, não a existência
dele.

---

## 7. O que continua inconclusivo

1. **Se existe informação residual pequena.** `w = 0` foi medido em 8 eventos.
   Um peso verdadeiro de 0,05 produziria uma melhora da ordem de 0,001 de log
   loss, muito abaixo do que esta amostra separa de zero.
2. **Se o modelo é pior que o mercado nesta janela.** O IC de B − A contém zero.
   Que ele seja pior **em amostra grande** é outro resultado, de
   `DISCRIMINACAO.md`, e não foi re-medido aqui.
3. **Qualquer coisa sobre retorno.** 19 lutas com preço executável.
4. **Os grupos de experiência.** n entre 8 e 28 por grupo.
5. **A temporalidade das 207 linhas semeadas** não é recuperável.
6. **Se o mercado também erra mais no grupo de pouca experiência.**

## 8. Limitações que não são incerteza, são ausência

- odds históricas de 1994–2023 **sem** casa, **sem** mercado e extraídas
  **depois** das lutas;
- `market_comparison.csv` sobreposto a treino e calibração;
- preço executável só a partir de out/2026;
- instante de criação da previsão só a partir de 06/11/2026;
- a mediana não tem proveniência por preço.

---

## 9. Conclusão

**O modelo não acrescentou informação ao mercado nesta amostra.** O procedimento
pré-registrado pôde chegar a essa conclusão e chegou: peso zero no
desenvolvimento, com a curva de perda subindo monotonicamente a partir dele.

**O baseline continua em produção, inalterado.** Nenhum experimento novo foi
aberto, nenhum hiperparâmetro foi tocado, nenhuma feature foi adicionada.

Isto é coerente com o diagnóstico que o projeto já tinha: o modelo é bem
calibrado e quase não discrimina (AUC 0,540 contra 0,635 do mercado, resolução
5,6× menor), e **o teto são as features**. Um modelo que discrimina menos que o
mercado e não vê nada que o mercado não veja não tem informação residual para
oferecer — e é isso que o peso zero diz.

**Nenhuma vantagem estatística foi demonstrada, e nada aqui indica prontidão
para arriscar dinheiro.** O resultado é negativo, está registrado como negativo,
e o critério que o produziu foi escrito antes.
