# Correção e candidato para atletas com pouca experiência

Branch `correcao-estreantes`, a partir de `61b3503`. Sem push, sem publicação,
sem alteração retroativa de previsão.

Python 3.13.14 · sklearn 1.9.0 · pandas 3.0.5 · numpy 2.5.1 · lightgbm 4.7.0
Modelo de referência treinado em 25/10/2026 (treino 2001-02-23 a 2021-09-25,
teste 2024-04-27 a 2026-10-24). Testes no início: **493**. No fim: **505**.

`Revisao_Critica_UFC_Predictor.md` não existe no repositório nem foi fornecido;
trabalhei sem ele e verifiquei cada achado contra o código atual.

---

## 1. Diagnóstico: o alvo é 1–2 lutas, não a estreia

Conjunto de teste, 1322 lutas, grupo pelo **lado menos experiente**:

| lutas UFC prévias | lutas | AUC | log loss | Brier | ECE | acerto |
|---|---|---|---|---|---|---|
| 0 (estreante) | 240 | 0,683 | 0,6459 | 0,2272 | 0,0208 | 62,1% |
| **1–2** | **337** | **0,601** | **0,6800** | **0,2434** | 0,0300 | **54,9%** |
| 3–5 | 315 | 0,717 | 0,6276 | 0,2184 | 0,0394 | 67,3% |
| 6+ | 430 | 0,701 | 0,6330 | 0,2213 | 0,0317 | 65,3% |

**Contrastes com bootstrap conjunto por evento** (mesmos eventos reamostrados,
2000×), não sobreposição de intervalos isolados:

| contraste | Δ AUC | IC95 | exclui zero |
|---|---|---|---|
| 1–2 − 3–5 | −0,116 | [−0,204, −0,022] | **sim** |
| 1–2 − 6+ | −0,100 | [−0,169, −0,027] | **sim** |
| 1–2 − 0 | −0,082 | [−0,165, +0,005] | não |
| 0 − 3–5 | −0,034 | [−0,123, +0,060] | não |

Log loss e Brier repetem o padrão: 1–2 se separa de 3–5 e 6+, e não se separa
do estreante.

**O que isto autoriza dizer:** o grupo 1–2 tem discriminação pior que os
experientes, com significância. **O que não autoriza:** afirmar que o
estreante vai bem — ele fica no meio, indistinguível dos dois lados. Uma
versão anterior deste relatório afirmou isso; estava errada.

### Por que 1–2 é plausivelmente pior (hipótese, não conclusão)

**Denominadores.** Mediana acumulada do lado menos experiente, no teste:

| grupo | golpes significativos tentados | quedas tentadas |
|---|---|---|
| 1–2 | 88 | **2** |
| 3–5 | 284 | 7 |
| 6+ | 775 | 18 |

`takedown_defense = 1 − acertadas_contra / tentadas_contra` com **duas**
tentativas contra não é uma taxa; e entra no modelo com a mesma aparência de
uma medida sobre 18 tentativas. O estreante escapa disso porque o valor é
`NaN` e o modelo o trata como ausente.

**Calibração por faixa** (5 faixas; as faixas são simétricas por construção,
porque cada luta entra espelhada). Gap = previsto − real; positivo =
confiante demais:

| grupo | faixa 60–80% | n |
|---|---|---|
| 0 | **−5 pp** (tímido) | 85 |
| **1–2** | **+6 pp** (confiante demais) | 144 |
| 3–5 | −7 pp (tímido) | 155 |
| 6+ | −6 pp (tímido) | 197 |

1–2 é **o único grupo em que o sinal se inverte**. Note que o ECE não separa
(3–5 tem o maior, 0,0394) — quem separa é o sinal, não a magnitude.

**Quem é o pouco experiente e onde falta dado:**

| grupo | só um lado fino | os dois | NaN striking_accuracy | NaN takedown_defense |
|---|---|---|---|---|
| 0 | 52,1% | 47,9% | 100,0% | 100,0% |
| 1–2 | **79,8%** | 20,2% | 0,0% | **32,3%** |
| 3–5 | 0% | 0% | 0,0% | 4,4% |

No grupo 1–2 o caso típico é **um lado fino contra um veterano** — o
diferencial mistura uma estimativa ruidosa com uma sólida.

### Contra o mercado, nas mesmas lutas

165 lutas do teste com odds:

| grupo | lutas | log loss modelo | log loss mercado | acerto modelo | acerto mercado |
|---|---|---|---|---|---|
| 0 | 32 | 0,616 | 0,501 | 68,8% | 75,0% |
| **1–2** | 40 | 0,713 | 0,645 | **45,0%** | 65,0% |
| 3–5 | 43 | 0,624 | 0,607 | **74,4%** | 69,8% |
| 6+ | 50 | 0,678 | 0,644 | 60,0% | 70,0% |
| total | 165 | 0,660 | 0,607 | 61,8% | 69,7% |

Amostras de 32 a 50 lutas: direcional, não decisivo.

---

## 2. §3 — integridade: confirmado, resolvido, não reproduzido

| # | item | estado |
|---|---|---|
| 1 | relógio = primeira luta do card | **confirmado → corrigido** |
| 2 | registro retroativo | resolvido antes (`_evento_ja_passou`) |
| 3 | avaliação com nível atual | resolvido antes (`frozen_predictions_for_event`) |
| 4 | revanche casada só por nomes | **não reproduzido** — `record` filtra por `event_name`, `sync` por `event_date`; revanche em datas distintas é distinguida |
| 5 | relatório recalcula luta passada | resolvido antes |
| 6 | homônimos / IDs estáveis | **confirmado, não corrigido** — features usam `fight_url` como `fight_id`, mas o histórico não tem ID: chaveia por nome + data |
| 7 | lado/CLV incoerente | resolvido antes (guarda de inversão + teste) |
| 10 | preço sintético somado a executável | resolvido antes (`odds_tp` separado, `legs_casa` no painel) |
| 11 | estados de cancelamento/empate/NC | **confirmado, não corrigido** — tudo compartilha `actual_winner` vazio, sem estado nomeado |
| 12 | odds inválidas | resolvido antes (teto 100, NaN, negativo) |
| 13 | manifesto com hashes e versões | **parcial** — há `training_metadata.json` e `generator_fingerprint`; faltam hash de dataset e versões de bibliotecas |
| 14 | chave em log de exceção | **confirmado → corrigido** |
| 15 | modelo padrão divergente | **confirmado → corrigido** |

### Corrigidos nesta branch

**1 — relógio por luta.** `auto_capture` julgava o card inteiro pela primeira
luta (`fights[0]`). Dois modos de falha: prelim já em andamento entrava com
odd **ao vivo** gravada como fechamento; e card atravessando a meia-noite UTC
perdia a captura do resto quando a primeira luta saía da listagem. Agora cada
luta precisa estar listada **e** ter `commence_time` no futuro. Início ausente
ou ilegível não captura, e o motivo vai para o log.

**15 — modelo padrão.** `src.evaluate` e `src.predict` usavam `gbm`; relatório,
`line_shopping`, `new_event` e `publish_report` usavam `logreg`. Rodar
`python -m src.evaluate` sem `--model` reportava métricas de um modelo que não
é o publicado. Agora é `config.MODELO_PRODUCAO`, num lugar só.

**14 — chave da API em log.** O `requests` põe a URL inteira na mensagem do
`HTTPError`, com `apiKey` na query string, e os jobs registravam `exc` direto:
dez ocorrências em claro em `data/auto_capture.log`. Os dois logs são
gitignored e `git log -S` confirma que a chave **nunca** entrou no
repositório. Corrigido com `sanitizar()` nos três pontos que logam exceção de
rede; logs existentes limpos no lugar e valor real conferido como ausente.
**Recomendo rotacionar a chave** — ela existiu em claro no disco.

### Confirmados e não corrigidos, com motivo

**6 — IDs no histórico.** Migrar o `prediction_history.csv` para IDs estáveis
exige reescrever 202 linhas congeladas. É possível preservando o original,
mas é mudança de alto risco no artefato central e não cabia junto com um
experimento de modelo. Proposta: coluna `fight_uid` nova, preenchida só para
eventos futuros, sem tocar no passado.

**11 — estados persistentes.** Hoje cancelada, empate, NC e coleta incompleta
compartilham `actual_winner` vazio. `src/fechamento.py` já **distingue** os
casos em tempo de execução (`empate`, `ausente`, `ambigua`), mas não os
persiste. Proposta: coluna `estado_resultado`, preenchida daqui para frente.

---

## 3. §6B — shrinkage: implementado, medido, **não promovido**

`src/shrinkage.py`. Empirical Bayes por método dos momentos:

    taxa_encolhida = (sucessos + k · prior) / (denominador + k)

`prior` e `k` estimados **só nas linhas de treino** da janela — não há grade
de busca, então não há como escolher pseudocontagem olhando avaliação.

Priors estimados em 10088 linhas de treino (1994-03-11 a 2019-05-11):

| taxa | prior | k |
|---|---|---|
| striking_accuracy | 0,4373 | 18,7 |
| career_win_rate | 0,6511 | 8,6 |
| takedown_accuracy | 0,4292 | 5,8 |
| takedown_defense | 0,3451 | 4,6 |
| ko_rate | 0,2453 | 4,5 |
| submission_rate | 0,1378 | 3,9 |

### O tratamento aconteceu, e é graduado pela experiência

Mediana de |Δ| na feature:

| taxa | grupo 1–2 | 3–5 | 6+ |
|---|---|---|---|
| career_win_rate | **0,3126** | 0,1031 | 0,0430 |
| ko_rate | 0,2004 | 0,1293 | 0,0388 |
| takedown_defense | **0,1222** | 0,0486 | 0,0147 |
| striking_accuracy | 0,0155 | 0,0038 | 0,0010 |

`takedown_defense` do grupo 1–2 também deixa de ser `NaN` em 24,4% → 0,3% dos
casos. Veterano quase não se move, que é o comportamento desejado.

### Resultado em `cal_select` (540 lutas, dataset truncado em 2023-09-16)

| grupo | lutas | log loss base | log loss shr | Δ | AUC base | AUC shr |
|---|---|---|---|---|---|---|
| 0 | 105 | 0,6808 | 0,6814 | +0,0006 | 0,6080 | 0,6084 |
| 1–2 | 179 | 0,6721 | 0,6686 | **−0,0035** | 0,6182 | **0,6357** |
| 3–5 | 124 | 0,6871 | 0,6840 | −0,0031 | 0,5870 | 0,5950 |
| 6+ | 132 | 0,6458 | 0,6447 | −0,0011 | 0,6849 | 0,6845 |
| TODOS | 540 | 0,6708 | 0,6688 | −0,0020 | 0,6247 | 0,6313 |

Contraste pareado, bootstrap conjunto por evento:

| grupo | Δ log loss | IC95 | veredito |
|---|---|---|---|
| 0 | +0,0006 | [−0,0038, +0,0052] | inconclusivo |
| 1–2 | −0,0036 | [−0,0105, +0,0033] | inconclusivo |
| 3–5 | −0,0031 | [−0,0093, +0,0034] | inconclusivo |
| 6+ | −0,0010 | [−0,0050, +0,0030] | inconclusivo |
| TODOS | −0,0020 | [−0,0046, +0,0007] | inconclusivo |

**Recomendação: não promover.** Nenhum intervalo exclui zero.

Isto **não** é "sem efeito". O sinal é consistente em direção nos quatro
grupos e maior exatamente onde a hipótese previa, e o grupo 0 — onde a
feature é toda `NaN` e o shrinkage não tem o que encolher — é o único com Δ
positivo, o que é o controle interno esperado. O que falta é **poder**: 540
lutas no total e 105–179 por grupo.

A decisão de não promover segue a regra combinada, não uma leitura de que a
ideia falhou.

---

## 4. Critério de aceite: redação atual e substituição proposta

Redação exata em `.claude/skills/ufc-analise/SKILL.md`:

> **Precisa melhorar as duas coisas:**
> 1. log loss do preditor de vencedor;
> 2. a fração de previsões acima de 75% — hoje ~3%, contra um mercado que
>    chega a 92%.
>
> Só o log loss significaria um modelo mais bem calibrado na mesma ignorância.
> A prova de que os dois critérios são necessários: um GBM de capacidade
> extrema produz 61,8% de previsões acima de 75% com AUC de 0,567 —
> confiante e errado.

O problema: o critério 2 pede **aumento** da fração acima de 75%. O próprio
contraexemplo citado mostra que essa fração sobe quando o modelo fica
confiante e errado. A conjunção com log loss mitiga, mas a meta continua
apontando para o lado errado.

**Proposta** (desempenho probabilístico, calibração, cobertura, estabilidade):

1. **Probabilístico, obrigatório.** Δ log loss **e** Δ Brier negativos em
   `cal_select`, com IC95 por bootstrap pareado por evento excluindo zero.
2. **Discriminação, obrigatório.** AUC não piora além do IC — a resolução é o
   gargalo declarado.
3. **Calibração, obrigatório.** ECE não piora materialmente **e** o gap da
   faixa de maior confiança não passa de negativo (tímido) para positivo
   (confiante demais).
4. **Cobertura, obrigatório.** A fração de lutas com previsão não cai.
5. **Estabilidade, obrigatório.** Sem degradação material em nenhum grupo de
   experiência, e sinal de mesmo sentido em janelas temporais distintas.
6. **A fração acima de 75% vira diagnóstico reportado, nunca meta.**

Pelo critério proposto, o shrinkage também não passa — reprova no item 1.
Não proponho a troca para aprovar este candidato.

---

## 5. §5 — pré-UFC: levantamento, sem integração

Resultado local, por construção: **a cobertura pré-UFC no repositório é zero**.
`data/raw/fights.csv` tem 794 eventos, todos UFC. Para um atleta com 1 ou 2
lutas no UFC, o histórico anterior simplesmente não existe aqui — é essa
ausência que produz os denominadores de 2 tentativas documentados acima.

Não integrei nada e não consultei fonte paga. O que precisa ser verificado
antes de qualquer integração, e **não** verifiquei:

- termos de uso e permissão de coleta de cada fonte;
- cobertura histórica real por período e organização;
- identidade: resolver atleta entre fontes sem nome como chave;
- **disponibilidade temporal** — saber se o dado já era público na data da
  previsão, que é o requisito que distingue dado anterior ao cutoff de dado
  conhecido na época;
- separação profissional / amador / exibição;
- deduplicação da mesma luta entre fontes.

Vale registrar contra quem isso vai ajudar: o grupo 1–2, não o estreante. Ele
é o que tem estatística ruidosa, e um histórico regional daria denominador. O
estreante continuaria com `NaN`, que o modelo já trata bem.

---

## 6. Limitações

- **O conjunto de teste é desenvolvimento.** Foi consultado muitas vezes nesta
  e em sessões anteriores. Nada aqui é backtest prospectivo estrito.
- **Não há avaliação progressiva por janelas.** Usei o protocolo do projeto
  (`cal_select` temporal, teste nunca tocado). O walk-forward do §9 não foi
  implementado, e é ele que resolveria a falta de poder do §6B.
- **ROI e drawdown não reportados** — exigiriam preço executável por grupo, e a
  série real tem 70 pernas no total.
- Os grupos usam o **lado menos experiente** da luta. Outro agrupamento
  (por atleta, não por luta) daria outros números.

## 7. Recomendação

**Manter o baseline.** Promover o shrinkage exigiria IC excluindo zero, e não
exige.

Próximo passo de maior retorno, na ordem: implementar a avaliação progressiva
por janelas do §9 e reavaliar o shrinkage nela — é o único caminho que
converte um sinal consistente e fraco em decisão. Depois, o levantamento
externo do §5, que ataca a causa em vez do sintoma.

## 8. Como reproduzir

```bash
git checkout correcao-estreantes
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe exp_shrinkage.py
```

`src/shrinkage.py` está no repositório e não é usado por nenhum caminho de
produção — a importação não altera o modelo publicado. Reverter a branch
inteira desfaz tudo; nenhuma previsão histórica foi tocada.

---

# Segunda rodada — verificações e walk-forward (05/11/2026)

## A. Matriz completa dos 15 itens

| # | item | status | arquivo / função | evidência | teste |
|---|---|---|---|---|---|
| 1 | relógio = primeira luta | **corrigido** | `scripts/auto_capture.py::main` | `fights[0]` era o relógio do card inteiro | dry-run; sem teste automatizado (depende da API) |
| 2 | registro retroativo | resolvido antes | `prediction_history::_evento_ja_passou` | — | `TestLinhaAbertaDeEventoPassado` |
| 3 | nível atual em luta passada | resolvido antes | `prediction_history::frozen_predictions_for_event` | — | `TestOrdemSyncAntesDeRecord` |
| 4 | revanche por nomes | **corrigido** | `evaluate::compare_to_market` | 5 pares com revanche no teste; 2 pegavam a luta errada por ~14,5 pp | `test_revanche_mercado.py` |
| 5 | recalcula luta passada | resolvido antes | `card_report::analyze_card` (`frozen`) | — | `TestOrdemSyncAntesDeRecord` |
| 6 | IDs estáveis | **corrigido** | `src/identidade.py` → `fight_uid` | 187 de 202 linhas com id do UFCStats | `test_identidade.py` |
| 7 | lado/CLV incoerente | resolvido antes | guarda de inversão em `record_card_predictions` | — | `test_integridade_dados.py` |
| 8 | timestamps, fonte, prova de cotação | **pendente** | — | odds não guardam resposta bruta nem horário por casa | — |
| 9 | split atravessando evento | resolvido antes | `train::temporal_group_split` | agrupado por `fight_id` | `test_tuning_and_calibration.py` |
| 10 | preço sintético + executável | resolvido antes | `odds_*_tp`, `legs_casa` | — | `TestOddDaCasa` |
| 11 | estados persistentes | **corrigido** | `src/identidade.py` → `estado_resultado` | 187 vencedor, 10 aguardando, 4 nao_aconteceu, 1 empate_ou_nc | `test_identidade.py` |
| 12 | odds inválidas | resolvido antes | `card_report::_odd_plausivel` | teto 100, NaN, negativo | `test_card_report.py` |
| 13 | manifesto com hashes e versões | **pendente** | `models/training_metadata.json` | tem parâmetros e ranges; faltam hash de dataset e versões de bibliotecas | — |
| 14 | chave em log | **corrigido** | `line_shopping::sanitizar` | 10 ocorrências em claro em `data/auto_capture.log` | `test_segredo_em_log.py` |
| 15 | modelo padrão divergente | **corrigido** | `config.MODELO_PRODUCAO` | `evaluate` e `predict` usavam gbm | — |

Itens **8** e **13** seguem abertos: a integridade **não** está concluída.

## B. As quatro verificações

**#2 — revanche em `compare_to_market`.** Reproduzido nos dados reais. O
código casava por nome e tomava `match.iloc[0]`. Joshua Van × Alexandre
Pantoja (linha de 19/set) recebia a previsão de 06/dez/2025 — 0,5629 no lugar
de 0,7083. Petr Yan × Merab Dvalishvili (24/out) recebia 0,3604 no lugar de
0,5069. Cinco pares do teste têm revanche. Corrigido com filtro por
`event_date`; par com revanche e linha sem data é **pulado**, não arbitrado.

**#3 — equivalência da montagem.** Com shrinkage desligado: as **mesmas 14.412
linhas** (mesmos `fight_id` e lados), **labels idênticos**, **padrão de NaN
idêntico** nas 16 features, maior diferença absoluta **5,684e-14**. A única
diferença entre braços é o tratamento.

**#4 — 1.322 → 540.** São **períodos disjuntos**, não filtro:

| conjunto | lutas | período |
|---|---|---|
| teste de produção | 1.324 | 2024-05-04 a 2026-10-31 |
| dataset truncado em 2023-09-16 | 7.206 | 1994-03-11 a 2023-09-16 |
| └ treino (70%) | 5.044 | até 2019-05-11 |
| └ calibração (15%) | 1.080 | 2019-05-11 a 2021-08-21 |
| &nbsp;&nbsp;└ `cal_select` | **540** | 2020-07-25 a 2021-08-21 |

Baseline e candidato usaram o mesmo split do mesmo dataset — mesma população.
Mas o experimento avalia 2020-2021 e o diagnóstico descreve 2024-2026.

## C. Peso na estimação empirical Bayes

Cada lutador entra com uma linha por luta, cumulativas: quem tem 25 lutas
contribui 25 vezes com valores quase iguais.

| taxa | todas as linhas | uma linha por lutador |
|---|---|---|
| career_win_rate | média 0,6449 · k 8,8 | média **0,5895** · k 8,7 |
| takedown_defense | média 0,3427 · k 4,9 | média 0,3631 · k 4,8 |

**O k praticamente não muda** — a repetição não inflou a força do prior. Mas a
média de `career_win_rate` muda **5,5 pontos**, porque veterano de cartel alto
contribui com muitas linhas. O centro do prior está enviesado para cima e isso
desloca todo valor encolhido. Não corrigi: trocar estimador com o resultado na
tela é mudar critério depois do fato.

## D. Walk-forward — o resultado que manda

Protocolo registrado antes de olhar (cabeçalho de `exp_walkforward.py`):
janela expansiva, 8 dobras, priors e k estimados só no treino de cada dobra,
cada luta em exatamente uma dobra. **2.511 lutas únicas, 207 eventos** — 4,6×
o `cal_select`.

| dobra | avalia | período | ll base → shr |
|---|---|---|---|
| 1 | 316 | 2022-01-15 … 2022-07-30 | 0,6610 → 0,6600 |
| 2 | 310 | 2022-08-06 … 2023-03-25 | 0,6715 → 0,6701 |
| 3 | 304 | 2023-04-08 … 2023-10-14 | 0,6439 → 0,6429 |
| 4 | 316 | 2023-10-21 … 2024-06-08 | 0,6578 → 0,6558 |
| 5 | 315 | 2024-06-15 … 2025-02-01 | 0,6588 → 0,6585 |
| 6 | 309 | 2025-02-08 … 2025-08-23 | 0,6369 → 0,6348 |
| 7 | 329 | 2025-09-06 … 2026-04-25 | 0,6364 → **0,6380** |
| 8 | 312 | 2026-05-02 … 2026-10-31 | 0,6540 → **0,6551** |

As duas dobras mais recentes **pioram** — instabilidade temporal, e é a
salvaguarda falando.

| grupo | lutas | cobertura | Δ log loss | IC95 | Δ Brier | IC95 | veredito |
|---|---|---|---|---|---|---|---|
| 0 | 480 | 19,1% | **−0,0025** | [−0,0049, −0,0001] | −0,0011 | [−0,0022, +0,0001] | ll melhora, Brier **não confirma** |
| 1–2 | 631 | 25,1% | **+0,0015** | [−0,0028, +0,0058] | +0,0005 | [−0,0015, +0,0025] | inconclusivo |
| 3–5 | 597 | 23,8% | −0,0001 | [−0,0035, +0,0032] | +0,0001 | [−0,0014, +0,0016] | inconclusivo |
| 6+ | 803 | 32,0% | −0,0014 | [−0,0038, +0,0009] | −0,0006 | [−0,0017, +0,0004] | inconclusivo |
| **TODOS** | **2.511** | 100% | **−0,0006** | [−0,0021, +0,0009] | −0,0003 | [−0,0010, +0,0004] | **inconclusivo** |

### O que muda em relação à primeira rodada

**O walk-forward contradiz o `cal_select`.** Lá o grupo 1–2 tinha o maior
ganho (−0,0036); aqui ele é o **único com ponto estimado positivo** (+0,0015).
Amostra 4,6× maior e protocolo mais exigente: o walk-forward manda.

**O grupo que se move é o 0, não o 1–2.** O aviso de que o grupo zero não é
controle intocado estava certo — re-treino e recalibração mudam as previsões
dele. Há mecanismo plausível: com denominador zero a taxa encolhida vira
exatamente o prior, substituindo o `NaN` que o pipeline imputava. Mas o Brier
não confirma.

**Decisão: não promover.** O critério foi fixado antes — log loss menor no
agregado com IC excluindo zero. Deu −0,0006, IC [−0,0021, +0,0009]. Reprova.

### Interpretação, ajustada

- "Inconclusivo" **não** comprova ausência de efeito nem falta de poder. Com
  2.511 lutas o intervalo ainda cobre zero; o que se sabe é que, se há efeito,
  ele é menor do que este desenho distingue.
- A afirmação da primeira rodada de que o grupo 0 era "controle interno"
  estava **errada**.
- Histórico pré-UFC pode ajudar **estreantes e 1–2**. Para o estreante ele
  substituiria o prior por informação individual.
- O walk-forward esclareceu estabilidade e **não** produziu evidência
  suficiente para promover.

## E. Pendências

1. Itens **8** e **13** abertos — integridade não concluída.
2. Viés na média do prior por repetição cumulativa (seção C), não corrigido.
3. `test_report_freshness` **vermelho** na branch: `docs/index.html` é anterior
   às mudanças em `prediction_history.py`. Não regenerei porque o CSV do card
   aponta para evento já encerrado e a tarefa pedia não publicar.
4. **Corrupção encontrada e corrigida:** em 01/nov o ciclo gravou as 12 lutas
   de 31/out sob `event_date` 07/nov, porque `_card_atual()` confiava no
   `ciclo_estado.json` em vez do conteúdo do CSV. `TestCardTemplateHistorico`
   falhou e a trava de suíte-antes-do-push impediu a publicação. As 12 linhas
   foram removidas (as originais de 31/out seguem fechadas e intactas) e
   `_card_atual()` passa a derivar sempre do CSV.
5. O card de 07/nov mudou desde o registro de 29/out — três lutas saíram, três
   entraram, incluindo o main event. Precisa de passada própria.
6. Confirmação do candidato exige previsão futura registrada antes do
   resultado. Todas as janelas aqui são desenvolvimento.
