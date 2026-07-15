# UFC Predictor — previsão de vencedor

Pipeline em Python que estima a probabilidade de vitória de cada lutador
em uma luta de UFC/MMA, usando dados históricos públicos. Alvo principal:
o **vencedor** (classificação binária, bem validado). Desde jul/2026 há
também uma fase 2 de **método de vitória e faixa de round** — tratada
explicitamente como *tendência estatística* de confiabilidade modesta
(ver "Método de vitória e duração").

## Status de execução

O pipeline **está executado e validado de ponta a ponta** (última rodada:
jul/2026; ~8,7 mil lutas, 1994 até jun/2026): coleta → features → treino →
avaliação → CLI de predição, com uma suíte de 177 testes
(`python -m pytest tests/`) cobrindo features point-in-time, espelhamento,
adaptadores de dados, Elo, calibração e a comparação com odds de mercado.
Métricas atuais na seção "Avaliação"; evolução na seção "Histórico de
versões".

**Atenção — scraping do UFCStats.com:** o site passou a servir um desafio
anti-bot em JavaScript (proof-of-work) antes de qualquer página, então o
scraper com `requests` + BeautifulSoup **não consegue acessar o site ao
vivo**. Os seletores CSS e toda a lógica de parsing foram validados contra
snapshots arquivados (Wayback Machine) e estão corretos. Este projeto
deliberadamente **não** resolve o desafio fora de um navegador; a via
sancionada para dados recentes é o `--fill-gap` (navegador real headless,
ver "Frescor dos dados").

## Estrutura do projeto

```
ufc_predictor/
├── config.py              # caminhos e parâmetros centrais (split, seeds, Elo, frescor)
├── requirements.txt
├── data/
│   ├── raw/               # dados brutos (espelho GitHub convertido, odds, entrada manual)
│   ├── processed/         # features + predições de teste + comparação com mercado
│   └── odds_template.csv  # template manual de odds (complemento)
├── models/                # modelos treinados e calibrados (.joblib) + metadados
├── src/
│   ├── data_collection.py # fontes de dados (github-mirror/scrape/kaggle), gap-fill, frescor
│   ├── features.py        # engenharia de features (diferenciais, point-in-time)
│   ├── ratings.py         # rating Elo (passada cronológica global)
│   ├── train.py           # split temporal, treino, calibração por modelo
│   ├── train_method.py    # fase 2: método de vitória + faixa de round
│   ├── evaluate.py        # log loss, Brier, acurácia + odds manuais
│   ├── market_odds.py     # comparação com odds reais (backtest dedicado)
│   ├── tuning.py          # experimentos de hiperparâmetros (seleção em cal_select)
│   ├── card_report.py     # relatório HTML de card futuro (favoritos/zebras)
│   ├── predict.py         # CLI de predição
│   └── utils.py           # parsing, conversão de odds, fuzzy name matching
├── scripts/
│   └── run_pipeline.py    # roda tudo de ponta a ponta
└── tests/                 # suíte pytest (177 testes)
```

## Primeiros passos

```bash
cd ufc_predictor
pip install -r requirements.txt

# Recomendado: espelho GitHub + preenchimento dos eventos recentes faltantes
# com navegador real headless (o espelho está estagnado desde mai/2026)
python -m scripts.run_pipeline --source github-mirror --fill-gap

# Sem o --fill-gap, os dados vão só até onde o espelho parou
python -m scripts.run_pipeline --source github-mirror

# Fallback: dataset compilado do Kaggle (congelado em jun/2019)
python -m scripts.run_pipeline --source public-dataset

# Scraping completo direto do UFCStats.com com requests — HOJE INDISPONÍVEL
# (gate anti-bot no site; ver "Frescor dos dados"). Mantido caso a situação mude.
python -m scripts.run_pipeline --source scrape
```

Isso roda as 4 etapas em sequência: coleta → features → treino → avaliação.
Cada etapa também pode ser rodada isoladamente:

```bash
python -m src.data_collection --source github-mirror --fill-gap
python -m src.features
python -m src.train
python -m src.evaluate
python -m src.market_odds   # comparação com odds reais (backtest)
```

Depois de treinado, para prever uma luta futura:

```bash
python -m src.predict "Islam Makhachev" "Arman Tsarukyan"
```

## Relatório de card (favoritos e zebras)

Para analisar um card inteiro de uma vez contra odds reais, gere o
relatório HTML (self-contained, abre offline em qualquer navegador):

```bash
python -m src.card_report data/raw/upcoming_card_odds.csv \
    --output card_report.html --card-name "UFC 329" --model logreg
```

As duas abas são **mutuamente exclusivas por construção** — cada luta com
previsão válida aparece em exatamente uma delas (comparando `model_side`,
o lado mais provável segundo o modelo, com `market_side`, o favorito do
mercado após devig):

- **Aba "Favoritos mais seguros"**: lutas em que o modelo aponta o MESMO
  lado que o mercado. Ordenação: probabilidade do modelo para esse lado,
  decrescente.
- **Aba "Melhores zebras da noite"**: lutas em que o modelo aponta o
  **azarão do mercado** como lado mais provável de vencer — divergência
  direta de leitura, não apenas "azarão competitivo". Mesma ordenação
  (probabilidade do modelo). A probabilidade de mercado fica visível nos
  cards como contexto, mas não é critério de ordenação.
- **Estreantes no UFC**: lutador sem histórico na base entra com um
  perfil sintético de estreia — o MESMO formato que o treino viu na
  primeira luta de todo lutador da história (stats NaN, 0 lutas
  anteriores, Elo no rating base). A previsão é in-distribution, mas
  apoia-se só nos dados do adversário: o card carrega um aviso explícito
  de confiança reduzida. (O grupo "Sem previsão" segue existindo para
  falhas genuínas — nada some em silêncio.)
- Cada card de Favoritos/Zebras mostra **os dois lados** do confronto
  (lado apontado pelo modelo em destaque).
- **Aba "Método de vitória"**: odds **justas** por categoria (KO/TKO,
  finalização, decisão), decimal como formato principal e moneyline
  americana entre parênteses. Ordenação: probabilidade da categoria mais
  provável, decrescente.
- **Aba "Duração da luta"**: mercado over/under com **duas linhas — 1,5 e
  2,5 rounds** (Under 1,5 = termina no round 1; Under 2,5 = termina até o
  round 2; decisão sempre passa das duas linhas, pois vai ao round 3 ou 5).
  Cálculo em `predict.compute_total_rounds_market`: Under X = P(finalização)
  × P(faixas até X | finalização) — probabilidades *incondicionais* da
  luta. Sem linhas de 3,5/4,5: a faixa "Round 3+" do modelo não separa
  finais tardios, e não fingimos precisão que não existe.
- **Convenção das odds justas** (`utils.probability_to_fair_odds`): sem
  vig — decimal = 1/p; os dois lados de um mercado somam probabilidade
  implícita 1. p = 0.5 exato cai no lado negativo da americana
  (2.00 / -100). p fora de (0, 1) levanta erro em vez de inventar cap.
  **Estas odds NÃO são validadas contra mercado real**: não temos odds de
  casas para método/duração; esse modelo só foi comparado a um baseline
  ingênuo (resultado modesto, ver "Método de vitória e duração") — as duas
  abas carregam um aviso específico sobre isso.
- Falhas independentes: luta sem dados de método/duração aparece numa
  lista "sem previsão" dentro dessas abas, sem afetar a previsão de
  vencedor nas abas de Favoritos/Zebras.
- **Aba "Histórico"** (paper trading): eventos passados com o lado que o
  modelo apontou, o favorito do mercado (devig), o vencedor real e ✓/✗
  para cada um, além do placar agregado por evento. As previsões são
  **congeladas no momento da publicação** (`data/prediction_history.csv`,
  ver `src/prediction_history.py`): re-treinos posteriores não reescrevem
  previsões já registradas, e linha com resultado preenchido nunca muda.
  O registro exige `--event-date YYYY-MM-DD` na geração/publicação; os
  vencedores são sincronizados automaticamente do `data/odds_template.csv`
  (o mesmo que o `src.evaluate` já usa) na geração seguinte. Os
  denominadores dos placares diferem de propósito: o mercado tem lado em
  toda luta, o modelo só nas que conseguiu prever.
- **Avatares e fotos**: todo lutador tem um avatar de monograma (iniciais,
  cor estável por nome — zero dependência externa; é o modo da página
  publicada). Para o relatório **local de uso pessoal**, a flag `--photos`
  busca as fotos reais nas páginas de atleta do UFC.com (hotlink, com
  cache em `data/raw/fighter_photos.json`; apague o arquivo para
  re-buscar). Com fotos o HTML deixa de ser offline/self-contained, e
  essa flag **não deve ser usada na publicação** — fotos promocionais são
  material com direitos autorais; uso estritamente pessoal. Foto que não
  carrega cai de volta no monograma automaticamente.
- O relatório embute o resultado do `check_data_freshness()` e um aviso
  fixo de que isso é estimativa estatística, não recomendação de aposta.

**Formato do CSV de entrada** (`fighter_a,fighter_b,odds_a_decimal,odds_b_decimal`,
odds decimais > 1.0): **as odds são fornecidas por você a cada evento** —
intencionalmente não há busca automática de odds ao vivo (fontes gratuitas
e estáveis para isso não existem, como o histórico deste projeto mostrou;
ver "Frescor dos dados"). Use `src/utils.py::moneyline_to_decimal` para
converter moneyline americana.

### Publicação no GitHub Pages (link fixo)

O relatório é publicado em **https://bergols.github.io/ufc-predictor/** —
o link não muda; cada evento novo sobrescreve o `docs/index.html` (padrão
do GitHub Pages). Fluxo por evento, em um comando:

```bash
# 1. edite data/raw/upcoming_card_odds.csv com o card e as odds do evento
# 2. gere + commite + publique:
python -m scripts.publish_report data/raw/upcoming_card_odds.csv --card-name "UFC 330: Fulano vs Beltrano" --event-date 2026-08-01
```

(`--no-push` gera e commita sem publicar, para conferir localmente antes;
`git push` depois completa.) A página publicada carrega no rodapé o aviso
de que é um **relatório estático, gerado manualmente por evento, sem
backend** — não se atualiza sozinha. Configuração do Pages (feita uma
única vez): Settings → Pages → "Deploy from a branch" → branch `main`,
pasta `/docs`.

Não confunda os dois CSVs de odds:

| arquivo | para quê | tem `actual_winner`? |
|---|---|---|
| `data/raw/upcoming_card_odds.csv` | card FUTURO → relatório favoritos/zebras | não (luta ainda não aconteceu) |
| `data/odds_template.csv` | backtest manual de lutas PASSADAS (`src/evaluate.py`) | sim (resultado conhecido) |

## Fontes de dados

1. **Espelho GitHub (fonte principal; histórico até mai/2026 + `--fill-gap`
   para o resto)** — `src/data_collection.py::download_github_mirror_dataset()`.
   O repositório
   [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats)
   rodava um scraper próprio do UFCStats.com diariamente (job automatizado em
   GCP Cloud Run + Cloud Scheduler) commitando 6 CSVs no repo.
   Nosso adaptador baixa esses CSVs e os converte para o formato canônico
   "scrape" (`fights.csv` / `fight_stats.csv` / `fighters.csv`), agregando
   as estatísticas por round em totais por luta — assim o cálculo
   *point-in-time* correto de `features.py::build_features_from_scrape` é
   reaproveitado integralmente. A agregação foi validada contra os totais
   oficiais das páginas do UFCStats (via snapshots arquivados).

   **Nota de manutenção:** esta fonte depende do job automatizado de um
   terceiro continuar rodando — **e ele já parou uma vez**: os commits
   diários de dados cessaram em 2026-05-21 (issue #34 do repo relata o
   scraper deles rodando "sem raspar nada", sintoma do gate anti-bot do
   UFCStats). Por isso existe o `--fill-gap` e a verificação automática de
   frescor (ver seção "Frescor dos dados").

   **Atribuição/licença:** o repositório Greco1899/scrape_ufc_stats é
   licenciado GPL-3.0 — isso cobre o *código* dele (que não usamos; apenas
   baixamos os CSVs). Os dados em si são estatísticas factuais compiladas
   do UFCStats.com. Isto não é aconselhamento jurídico; mantida a
   atribuição aqui por transparência.

2. **Dataset público compilado (fallback secundário, congelado em jun/2019)**
   — o `data.csv` do `rajeevw/ufcdata` (Kaggle), baixado automaticamente de
   um espelho no GitHub (o repositório original não versiona o CSV). Schema
   com prefixos `R_`/`B_` verificado em execução real: todas as features
   casam, e a defesa de queda (que não existe pronta nesse dataset) é
   derivada de `{R,B}_avg_opp_TD_att/_landed`. O adaptador em
   `src/features.py::build_features_from_public_dataset()` usa
   correspondência flexível de nomes de coluna (`_PUBLIC_DATASET_COLUMN_HINTS`)
   e loga um aviso para qualquer feature que não encontrar. Se o download
   automático falhar, baixe manualmente em
   https://www.kaggle.com/datasets/rajeevw/ufcdata e salve como
   `data/raw/public_dataset.csv`.

   **Se alguma feature aparecer sempre vazia (NaN) depois de rodar** com
   esse fallback, rode
   `python -c "import pandas as pd; print(pd.read_csv('data/raw/public_dataset.csv').columns.tolist())"`
   e ajuste `_PUBLIC_DATASET_COLUMN_HINTS` em `src/features.py` com os
   nomes reais das colunas do seu CSV. (Nesse fallback, `elo_diff` fica NaN
   por design — o formato largo não permite a passada cronológica do Elo.)

3. **UFCStats.com (scraping direto — hoje bloqueado)** —
   `src/data_collection.py::run_full_scrape()`. Extrai eventos, lutas,
   estatísticas por luta e perfis de lutadores. Os seletores CSS e o parsing
   estão validados (via Wayback Machine), mas o site ao vivo está atrás de
   um gate anti-bot em JavaScript que este projeto deliberadamente não
   contorna. Use `--limit-events N` para testar rápido se o acesso voltar.

## Frescor dos dados (leia — isso faz parte do fluxo normal)

**Nenhuma fonte gratuita testada até agora garante estar sempre 100% em
dia.** O espelho GitHub parou silenciosamente em mai/2026; o dataset do
Kaggle está congelado em 2019; e datasets alternativos encontrados (ex.:
juan-villa/UFC-finish-rate-analysis, com resultados até jun/2026) também
atualizam com atraso e sem garantia — além de não trazerem golpes
*tentados*, o que inviabiliza as features de accuracy. Tratar a defasagem
como rotina, não como exceção:

1. **Verificação automática**: após toda coleta (e no CLI de predição), o
   pipeline compara a data do evento mais recente com hoje e emite um
   WARNING "DADOS DESATUALIZADOS" se o gap passar de
   `config.DATA_FRESHNESS_MAX_GAP_DAYS` (default 14 dias). Se você vir esse
   aviso, use uma das opções abaixo.

2. **`--fill-gap` (preferido)**: raspa somente os eventos faltantes do
   UFCStats.com usando um navegador real headless (Playwright + Edge do
   sistema). Contexto: o UFCStats exige execução de JavaScript antes de
   servir páginas, o que bloqueia `requests`; um navegador de verdade
   atende a esse requisito naturalmente. Este projeto **não** resolve o
   desafio fora de um navegador (isso seria contornar a proteção do site),
   e o gap-filler mantém o rate limiting educado (1 req/s) e busca só os
   poucos eventos que faltam.

3. **Entrada manual (rede de segurança final)**: crie
   `data/raw/manual_recent_fights.csv` com as colunas
   `event_name,event_date,fighter_1,fighter_2,winner,weight_class,method,round,time`
   (uma linha por luta; `winner` **vazio** para empate/no-contest, nunca
   "Draw" ou similar). O pipeline mescla o arquivo automaticamente em toda
   coleta, sem duplicar lutas já presentes. As stats de golpes/quedas
   dessas lutas ficam NaN (tratado normalmente), mas vitórias/derrotas,
   forma recente e recência — que são o que mais importa para manter o
   modelo em dia — entram corretas. ~10 eventos são ~120 linhas, uma
   digitação chata porém viável.

## Decisões técnicas importantes

- **Features são sempre diferenciais** (`fighter_a - fighter_b`), nunca
  valores absolutos isolados — striking accuracy, takedown accuracy,
  takedown defense, reach, altura, idade, tempo de inatividade, forma
  recente (últimas 5 lutas) e experiência (nº de lutas anteriores).
  Exceção deliberada: `stance_mismatch` é *simétrica* (1 se as stances
  divergem, 0 se iguais) e por isso não inverte sinal na linha espelhada.

- **Features adicionadas em jul/2026** (medidas contra o backtest de
  mercado, ver seção seguinte):
  1. `stance_mismatch` — confronto de stance (Orthodox vs Southpaw = 1;
     iguais = 0; "Switch"/ausente = NaN, incerto não vira palpite).
  2. `ko_rate_diff` / `submission_rate_diff` — taxa acumulada point-in-time
     (mesmo `shift(1)` anti-vazamento de sempre) de vitórias por KO/TKO e
     por finalização, como proporção do total de lutas anteriores. O texto
     livre de método é normalizado por `features.categorize_method` (cobre
     as variações das duas fontes: "U-DEC"/"Decision - Unanimous", etc.).
  3. `elo_diff` — rating Elo (`src/ratings.py`, base 1500; K=64 escolhido
     por validação, ver seção de metodologia), calculado numa única passada
     cronológica global; o rating registrado para cada luta é o de ANTES
     dela (point-in-time), e empate/no-contest não atualiza rating.
     Adicional às win rates existentes — o GBM decide a importância
     relativa (o Elo entrou em 3º-4º lugar, atrás de idade e reach).
     Margem por método de vitória (K maior para finalizações) foi
     implementada e testada, mas **não bateu o Elo simples** em validação —
     produção usa `ELO_METHOD_MULTIPLIERS = None`.

- **Point-in-time, não career-to-date**: ao usar a via de scraping, as
  estatísticas de cada lutador para uma luta específica são calculadas
  usando *apenas* as lutas anteriores àquela data (`src/features.py::
  compute_point_in_time_stats`, com `shift(1)` antes de acumular). Usar a
  média de carreira atual do UFCStats para prever uma luta de 2015 seria
  vazamento de dados do futuro.

- **Dataset espelhado**: cada luta gera duas linhas (A−B e B−A), para que
  o modelo não aprenda nenhum viés de "quem é listado primeiro". As duas
  linhas de uma mesma luta compartilham `fight_id` e o split temporal
  agrupa por `fight_id`, garantindo que nunca fiquem em lados diferentes
  do split (o que seria vazamento).

- **Dados faltantes**: lutadores com poucas lutas (`config.
  MIN_FIGHTS_FOR_RELIABLE_STATS`, default 3) não são descartados — ganham
  uma flag `fighter_a_low_experience` / `fighter_b_low_experience`, e os
  valores faltantes são imputados pela média (regressão logística) ou
  tratados nativamente como NaN (gradient boosting, que lida com valores
  faltantes sem imputação).

- **Split temporal, não aleatório**: `config.TRAIN_FRACTION` (70%) /
  `CALIBRATION_FRACTION` (15%) / `TEST_FRACTION` (15%), ordenado por data.
  Treina no passado, calibra e testa no futuro relativo ao treino.

- **Calibração escolhida POR MODELO, por dados**: sigmoid (Platt) vs
  isotonic é decidido separadamente para a logreg e para o GBM, comparando
  log loss em `cal_select` (ver metodologia abaixo). Motivo: isotonic tem
  mais capacidade e estava "decorando" as caudas da curva do GBM (log loss
  pior do que a acurácia sugeria); com a seleção por dados, ambos os
  modelos escolheram sigmoid na rodada atual — e o log loss do GBM em
  produção caiu de 0.693 para 0.657. Feita com `CalibratedClassifierCV`
  sobre o modelo já treinado, usando somente a fatia de calibração — nunca
  treino nem teste.

- **Metodologia de seleção de hiperparâmetros (anti-overfit da
  avaliação)**: nenhuma escolha (método de calibração, K do Elo, margem por
  método) é feita olhando o teste de produção nem o backtest de mercado —
  isso seria otimizar exatamente o número reportado como avaliação final.
  A fatia de calibração é dividida temporalmente em `cal_fit` / `cal_select`
  (`train.split_calibration_slice`); alternativas são comparadas em
  `cal_select`, a vencedora é re-treinada com a fatia inteira, e só então a
  avaliação final roda uma única vez. Detalhe extra: como a fatia de
  calibração de produção sobrepõe a janela do backtest de mercado, os
  experimentos de Elo (`python -m src.tuning`) rodam sobre o dataset
  truncado no fim da janela de odds, cujo `cal_select` termina antes das
  821 lutas do backtest.

- **Dois modelos**: regressão logística (baseline) e gradient boosting
  (LightGBM → XGBoost → HistGradientBoostingClassifier do sklearn, nessa
  ordem de preferência conforme o que estiver instalado).

## Avaliação e comparação com o mercado

Resultado atual (v3, jul/2026) com a fonte principal + `--fill-gap`
(8.603 lutas com vencedor definido, 1994 até jun/2026; teste = 1.291 lutas
mais recentes, nunca vistas no treino nem usadas em nenhuma escolha de
hiperparâmetro):

| modelo                  | acurácia | log loss | Brier |
|-------------------------|----------|----------|-------|
| regressão logística     | 0.621    | 0.652    | 0.230 |
| LightGBM                | 0.612    | 0.657    | 0.232 |

(Referências: chute aleatório = 50% / 0.693 / 0.250. A evolução por versão
está na seção "Histórico de versões". Números na faixa esperada; nada
artificialmente perfeito, o que seria sintoma de vazamento.)

`python -m src.evaluate` calcula log loss, Brier score e acurácia no
conjunto de teste.

## Comparação com o mercado (odds reais)

`python -m src.market_odds` compara as probabilidades do modelo com odds
decimais históricas reais do agregador betmma.tips, via o dataset
compilado [jansen88/ufc-data](https://github.com/jansen88/ufc-data)
(`data/complete_ufc_data.csv`, baixado automaticamente).

**Cobertura real da fonte (verificada em jul/2026):** 3.496 lutas com odds
válidas, de **nov/2014 a 16/09/2023** — o repositório está parado desde
dez/2023. Como o conjunto de teste de produção começa depois disso
(sobreposição zero), o módulo roda um **backtest dedicado**: trunca os
dados no fim da janela de odds e re-treina o mesmo pipeline (split
temporal 70/15/15 + calibração, em memória, sem sobrescrever os artefatos
de produção), gerando um teste out-of-sample de ago/2021 a set/2023.
Nota: os overrounds dessa fonte são baixos (às vezes negativos) porque o
betmma.tips agrega as *melhores* odds entre casas; e 6 lutas com odds
`inf` (dado quebrado na origem) são filtradas.

Resultado nas mesmas 821 lutas casadas (de 1.082 do teste do backtest;
261 sem odds na fonte), por versão do modelo (log loss; Brier/acurácia da
versão atual entre parênteses):

| lado                | v1     | v2     | v3 (atual)                    |
|---------------------|--------|--------|-------------------------------|
| **mercado (devig)** | 0.603  | 0.603  | **0.603** (0.208 / 0.677)     |
| regressão logística | 0.676  | 0.668  | **0.664** (0.236 / 0.587)     |
| LightGBM            | 0.721  | 0.670  | **0.666** (0.237 / 0.596)     |

**Leitura honesta: cada rodada encurtou o gap de log loss (v1: 0.073 →
v3: 0.061 para o melhor modelo), mas o mercado continua na frente com
folga nas três métricas** — inclusive ~8 p.p. de acurácia. Era o esperado:
casas de apostas incorporam muito mais informação (lesões, camp, notícia
de última hora, fluxo de apostas) do que estas features estatísticas. Um
detalhe honesto da v3: a calibração sigmoid melhorou o log loss da logreg
mas custou ~2 p.p. de acurácia no backtest (suaviza probabilidades perto
de 0.5) — trade-off aceito porque a métrica primária para comparar com
odds é log loss, e a escolha foi feita em `cal_select`, não aqui. O valor
do exercício é a régua: essa distância é o quanto o modelo precisaria
melhorar antes de qualquer conversa sobre "edge". Detalhe por luta em
`data/processed/market_comparison.csv`.

O fluxo manual continua disponível como **complemento** para eventos
recentes sem odds nessa fonte (mesma lógica de camadas dos dados de luta):
preencha `data/odds_template.csv` com odds decimais e o vencedor real e
rode `python -m src.evaluate` (`--model logreg` para o modelo de produção).
Lutas do template que não estão no conjunto de teste são **previstas ao
vivo** com o modelo calibrado atual — desde que o `event_date` seja
posterior ao fim do teste (evento recém-acontecido, caso típico do paper
trading); lutas antigas fora do teste são puladas por anti-vazamento, e
estreantes sem histórico são pulados com aviso. A coluna
`prediction_source` no resultado distingue as duas vias. Dica de fluxo:
preencha o template com as lutas e odds ANTES do evento (deixando
`actual_winner` vazio — linhas incompletas são ignoradas) e complete só o
vencedor depois. Mas lembre que 2-3 eventos são ruído estatístico, não
evidência. Trate qualquer resultado favorável com
ceticismo até acumular amostra bem maior e, idealmente, um período de
"paper trading" (apostas simuladas) antes de considerar dinheiro real.

## Método de vitória e duração (fase 2)

`python -m src.train_method` treina dois classificadores multiclasse
(logreg multinomial + GBM, mesma dupla de sempre), com o mesmo rigor do
preditor de vencedor: split temporal por `fight_id`, calibração em fatia
própria e baseline ingênuo obrigatório. Detalhes de modelagem:

- **Labels simétricas** (não dependem de quem é "A"/"B"): o treino usa as
  linhas espelhadas (duplicar sinal é inócuo), mas **calibração e teste
  são deduplicados** para uma linha por luta real — sem isso as métricas
  contariam cada luta duas vezes.
- **Features**: as diferenciais do preditor de vencedor **+ 6 somas
  simétricas** (`ko_rate_sum`, etc., ver `SYMMETRIC_SUM_COLUMNS`). A
  validação mostrou que só as diffs não carregam sinal de método — se a
  luta termina em nocaute depende do nível *combinado* de finalização dos
  dois, e a diferença (a−b) cancela exatamente essa informação.
- **Duração é condicional**: para DECISÃO, o round final é trivialmente o
  `scheduled_rounds` (validado: 99,98% batem). O modelo de round roda só
  sobre finalizações, em **3 faixas — Round 1 / Round 2 / Round 3+**
  (suporte no teste: 315/196/115). Histórico do agrupamento: round
  individual (1..5) não se sustenta (rounds 4-5 tinham 9 e 4 lutas — ruído);
  a 1ª versão usava {1, 2–3, 4–5}, mas não separava round 2 de round 3, o
  que impedia a linha over/under 2,5 do mercado de duração; {1, 2, 3+}
  resolve as duas coisas sem forçar classe rara.
- **`scheduled_rounds` (3 ou 5)** entra como *feature* do modelo de faixa
  (informe a coluna opcional no CSV do card; default 3, use 5 para main
  event/título). A restrição lógica antiga na saída (zerar "Rounds 4–5" em
  luta de 3 rounds) **deixou de ser necessária** com o reagrupamento: a
  faixa "Round 3+" existe em qualquer formato de luta, então nenhuma faixa
  é logicamente impossível.
- **Cobertura de fontes**: exige o formato canônico "scrape"
  (github-mirror/gap-fill) — `method`/`round` 100% preenchidos, 98,7%
  categorizáveis; `scheduled_rounds` em 97,2% (NaN em formatos antigos com
  overtime e nas lutas do `--fill-gap`). O fallback Kaggle **não tem**
  método por luta.

### Avaliação (teste temporal deduplicado, honesta como sempre)

Método (1.287 lutas; baseline = "sempre decisão"):

| modelo | log loss | acurácia | bate o baseline? |
|---|---|---|---|
| baseline ingênuo | 1.015 | 51,4% | — |
| logreg (usada em produção) | **0.994** | **52,8%** | sim, nos dois |
| LightGBM | 1.001 | 51,3% | só em log loss |

Faixa de round entre finalizações (626 lutas; classes {Round 1, Round 2,
Round 3+} com suporte 315/196/115; baseline = "sempre round 1";
`scheduled_rounds` como feature):

| modelo | log loss | acurácia | bate o baseline? |
|---|---|---|---|
| baseline ingênuo | 1.024 | 50,3% | — |
| logreg (usada em produção) | **1.005** | **51,8%** | sim, nos dois |
| LightGBM | 1.015 | 50,3% | só em log loss |

(O log loss absoluto subiu em relação ao agrupamento anterior de 3 faixas
porque as classes mudaram — os números não são comparáveis entre
agrupamentos; contra o próprio baseline, esta versão é a primeira que
bate em log loss E acurácia.)

(Na granularidade de 5 rounds, nenhum modelo batia o baseline em acurácia
— por isso as 3 faixas.) **Leitura honesta: o sinal existe mas é modesto**
(~2 p.p. de log loss sobre o baseline). Por isso a interface trata isso
como "como a luta tende a terminar" — contexto, não previsão. Suporte por
classe no teste: KO 406 / SUB 220 / DEC 661; faixas R1 315 / R2–3 298 /
R4–5 13.

Uso programático: `src/predict.py::predict_method_and_duration("A", "B")`
retorna as duas distribuições; falha de forma independente da previsão de
vencedor (lutas sem dados ficam "sem tendência" no relatório, mantendo o
vencedor).

## Histórico de versões

| versão | o quê | resultado-chave |
|--------|-------|-----------------|
| v0 (base) | Pipeline completo escrito e depois validado em execução real: dataset Kaggle (rajeevw, até 2019), features diferenciais point-in-time, split temporal, calibração, CLI. Bugs corrigidos na primeira execução (URL 404, vencedor do scraper, coluna de defesa de queda). | logreg 57,8% acc / 0.681 log loss (teste 2017–2019) |
| v1 (dados atuais) | Fonte trocada para o espelho Greco1899 (dados até o presente); depois, quando o espelho estagnou (mai/2026): verificação automática de frescor, `--fill-gap` com navegador real e entrada manual. | ~59% acc (teste 2023–2026); base sempre ≤14 dias atrás do presente |
| — (mercado) | Comparação com odds reais (jansen88/ufc-data, betmma.tips): backtest dedicado truncado em set/2023, 821 lutas casadas. | mercado 0.603 de log loss vs 0.676 do melhor modelo — sem edge |
| v2 (features) | `stance_mismatch`, `ko_rate_diff`/`submission_rate_diff` (point-in-time), `elo_diff` (K=32). | backtest: logreg 0.668, GBM 0.670 |
| v3 (polimento) | Calibração sigmoid/isotonic escolhida POR MODELO em `cal_select`; K do Elo validado em grade (K=64 venceu, ganho marginal); margem por método testada e **rejeitada** (não bateu o Elo simples). Metodologia anti-overfit documentada. | backtest: logreg 0.664, GBM 0.666; produção: logreg 62,1% acc / 0.652 log loss. Mercado segue à frente (0.603). |
| v4 (fase 2, atual) | Relatório de card (favoritos/zebras, dois lados por card) + previsão de método de vitória e faixa de round (features de soma simétricas; avaliação deduplicada; round agrupado em 3 faixas por suporte de amostra). | método: logreg 0.994 vs baseline 1.015 de log loss; round: 0.755 vs 0.783 — sinal modesto, tratado como tendência, não previsão. |

Tuning do preditor de vencedor encerrado por ora — próximos ganhos
provavelmente exigem informação nova (odds de abertura como feature,
dados de camp/lesão), não mais engenharia sobre as mesmas colunas.

## Limitações conhecidas (leia antes de usar)

- MMA tem alta variância; nenhum modelo deste tipo terá acurácia muito
  acima de ~60-65% de forma consistente — e isso já seria um resultado
  respeitável, não uma "fórmula mágica".
- `src/predict.py` no CLI exige lutadores com histórico na base. Nas vias
  programáticas (`allow_debutant=True`, usada pelo relatório de card e
  pelo evaluate), estreantes são previstos com o perfil sintético de
  estreia — previsão válida porém mais fraca (só um lado tem dados), e
  sempre sinalizada como tal.
- O modelo perde para o mercado de apostas em todas as métricas testadas
  (ver "Comparação com o mercado") — use como estudo, não como fonte de
  apostas.
- O scraper depende da estrutura HTML atual do UFCStats.com. Se o site
  mudar de layout, os seletores CSS em `src/data_collection.py` vão
  precisar de ajuste.
- O adaptador do dataset público (fallback Kaggle) faz correspondência de
  nomes de coluna "best effort" — confira o log ao rodar pela primeira vez.
