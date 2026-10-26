---
name: ufc-evento
description: >-
  Step-by-step conventions for running one event cycle of the ufc_predictor
  project (UFC win-probability model fed by UFCStats data): assembling the
  upcoming card CSV, pre-registering frozen predictions, publishing the HTML
  report, capturing the closing line, and closing the event with the real
  winners. Use this skill whenever the user is working on a card or evento —
  montar/fechar um card, pré-registro, publicar o relatório, odds de um card
  futuro, capture_closing, CLV, preencher vencedores, luta cancelada — or
  touches data/raw/upcoming_card_odds.csv, data/odds_template.csv or
  data/prediction_history.csv. Trigger it even when the user only says
  something like "o card acabou", "monta o próximo evento" or "pode fechar",
  because the ordering rules below are what keep the frozen pre-registration
  from being silently destroyed.
---

# Ciclo de um evento

O valor do projeto está no **pré-registro congelado**: a previsão é gravada
antes da luta e nunca reescrita depois que há resultado. Quase toda armadilha
abaixo é uma forma de apagar isso sem perceber. Na dúvida, prefira não gravar.

## Estrutura relevante

- `data/raw/upcoming_card_odds.csv` — card futuro (`fighter_a,fighter_b,odds_a_decimal,odds_b_decimal,odds_a_tp,odds_b_tp`). A **primeira linha é a luta principal** (`card_order` alimenta o destaque do relatório).
- `data/odds_template.csv` — mesmas lutas + `actual_winner`, preenchido depois do evento.
- `data/prediction_history.csv` — o pré-registro. Ver `src/prediction_history.py`.
- `scripts/` — `new_event.py` (fluxo completo), `publish_report.py`, `auto_capture.py`, `capture_closing.py`.
- Regras completas no `README.md`; critérios de decisão em "Regras de parada".

## Montar um evento novo

1. **Atualize a base e re-treine** antes de qualquer previsão — registrar com modelo defasado desperdiça o evento:
   ```bash
   python -m src.data_collection --source github-mirror --fill-gap
   python -m src.features && python -m src.train && python -m src.train_method
   ```
   `--source github-mirror --fill-gap` é o único caminho vivo. O `scrape` direto está bloqueado por anti-bot e o `public-dataset` está congelado em 2019.

2. **Monte o card com as DUAS fontes**, num comando:

   ```bash
   python -m scripts.build_card --event-date AAAA-MM-DD
   ```

   Ele lê a ordem do card no UFCStats, as lutas da data na The Odds API, une
   as duas, troca os nomes pela grafia canônica da base, pega mediana +
   Thunderpick e escreve o CSV. Rode com `--dry-run` antes para revisar.

   **Nenhuma fonte serve sozinha**, e a regra antiga ("UFCStats é a
   autoridade") errou nas duas direções: salvou duas lutas fantasma no evento
   12 e custou três lutas reais no 15. Agora cada luta carrega de onde veio, e
   a que só uma fonte conhece sai **marcada**, nunca omitida em silêncio:

   | marca | o que fazer |
   |---|---|
   | `ambas` | nada, é o caso normal |
   | `so_api` | entra por padrão — confira se existe. Excluir luta real custa mais que incluir fantasma, que some sozinha pelos denominadores |
   | `so_ufcstats` | fica **fora do CSV** (sem mediana não há benchmark) e é listada no fim. Decisão sua |

   Olhe também o aviso de **mediana de poucas casas**: preço de 3 livros não é
   consenso de mercado, e foi exatamente o perfil da perna de maior EV que
   perdeu no evento 15 (Malkoun, uma casa só).

3. **As odds são duas**, com trabalhos diferentes — nunca troque uma pela outra:

   | coluna | fonte | para que serve |
   |---|---|---|
   | `odds_a/b_decimal` | mediana de 20-25 casas (The Odds API) | define o **favorito do mercado**, que é o benchmark contra o qual o modelo é medido |
   | `odds_a/b_tp` | **Thunderpick** | é a odd que dá para pegar: entra no EV, no critério EV>1 e no P&L |

   A mediana é escrita à mão a partir de `src.line_shopping`. A da casa sai
   sozinha com `python -m src.thunderpick` (o `new_event` já roda isso no
   passo 3). **Janela aberta**, não headless: a Thunderpick responde 403 a
   requisição sem navegador e serve desafio do Cloudflare ao headless — que
   não contornamos. Sem odd da casa a luta cai na mediana e fica marcada; o
   EV dela não é o preço que você pega.

4. **Use a grafia canônica da base**, não a da API. Confira com
   `best_name_match(nome, levels['fighter'])` antes de escrever. Isso já evitou
   dois desastres: `Ian Garry` virando estreante no main event, e
   `Cameron Nelson` casando com `Cameron Else` — que **inverteu o lado
   apontado** numa perna EV>1.

5. **Anexe as mesmas lutas ao `odds_template.csv`** com `actual_winner` vazio,
   commite como "Pre-registro evento N", e publique:
   ```bash
   python -m scripts.publish_report data/raw/upcoming_card_odds.csv \
       --card-name "UFC ..." --event-date AAAA-MM-DD
   ```
   `--event-date` é obrigatório: sem ele nada entra no histórico.

**Registre perto da abertura do mercado.** Quanto mais cedo, mais movimento de
linha o CLV captura; registrar na véspera mede quase nada.

## Antes do card

`scripts/auto_capture.py` roda de hora em hora (tarefa agendada) e captura a
linha de fechamento sozinho. Ele só toca a API se houver evento aberto hoje ou
amanhã, e só grava enquanto a API **ainda lista** o evento — é assim que ele
sabe que a luta não começou, sem depender de relógio ou fuso.

Não há backfill: o que não for capturado antes do card está perdido (aconteceu
no evento 7). Para forçar manualmente: `python -m scripts.capture_closing --event-date AAAA-MM-DD`.

## Fechar o evento

1. `fill_recent_gap_with_browser()` para trazer os resultados do UFCStats.
2. Preencha `actual_winner` no `odds_template.csv` — **com a grafia daquela
   linha**, não a do UFCStats. O sync rejeita vencedor que não seja
   exatamente um dos dois lados.
3. **Luta cancelada fica em branco.** Linha sem resultado sai dos
   denominadores sozinha. Nunca invente vencedor. Já ocorreu duas vezes
   (Dulatov–Turman, Johnson–Ochoa).
4. O histórico fecha sozinho na próxima geração de relatório
   (`sync_results_from_template`).

## Armadilhas que já custaram caro

- **`sync` roda antes de `record`** em `generate_card_report`. Invertido, regerar
  um card cujos vencedores acabaram de ser preenchidos reescreve o pré-registro
  com o recálculo pós-evento e só então congela — apagando a previsão publicada
  sem aviso.
- **Luta encerrada exibe a previsão congelada, não o recálculo.** Depois do
  `--fill-gap` a base já contém o resultado da própria luta; recalcular produz
  previsões que "sabem" quem ganhou.
- **Fechamento gravado nunca é sobrescrito** no modo manual; no automático
  reescreve enquanto a luta está aberta (senão congelaria o preço mais antigo).
- **Não misture as duas odds.** A mediana mede o mercado; a da Thunderpick
  decide a entrada. Comparar a mediana do registro com a *melhor* odd de outro
  momento mistura escalas e infla qualquer conclusão — foi por isso que o CLV
  passou a comparar Pinnacle devigada com Pinnacle devigada.
- **Odd da casa é congelada no CSV**, nunca buscada na hora de gerar o
  relatório: ao vivo, o EV publicado andaria sozinho depois do pré-registro.
- **Os eventos 1 a 14 foram medidos na mediana** e continuam assim. As colunas
  da casa nascem vazias ali, e `model_side_leg` cai na mediana nesse caso —
  reescrever seria trocar o preço de pernas já resolvidas.
- Após publicar, o `docs/index.html` é sobrescrito — o link é fixo por design.

## Sobre EV>1

É o critério de pré-registro do paper trading, **não recomendação de aposta**. O
EV é auto-referente e o backtest mostra o mercado à frente justamente nas
divergências (−14,3% por perna em 230 lutas). Card com muitas pernas EV>1
costuma ser card cheio de estreante, onde o modelo regride para 55/45 — é
alarme, não oportunidade.
