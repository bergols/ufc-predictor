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

---

# Terceira rodada — reconciliação do card de 07/11 e alcance da corrupção (06/11/2026)

Rodada de modelagem **encerrada como inconclusiva**; shrinkage **não
promovido**; nenhuma nova busca de parâmetros.

## A. Reconciliação do card de 07/11

**Fonte de verdade:** a página do evento no UFCStats, que identifica
explicitamente organização e evento —
`http://ufcstats.com/event-details/110404b505fb562c`. Número de casas de
aposta **não** foi usado como prova de pertencimento; ele permanece apenas
como indicador de cobertura de mercado.

**O evento mudou de identidade depois de registrado:**

| | em 29/out (registro) | em 06/nov (UFCStats) |
|---|---|---|
| nome | UFC Fight Night: Bonfim vs Brady | **UFC Fight Night: Cortes Acosta vs Walker** |
| main event | Gabriel Bonfim × Sean Brady | **Waldo Cortes Acosta × Johnny Walker** |
| lutas | 10 | 14 |

### Tabela confronto a confronto

`ord` = posição na página do evento (0 = main event). Todas as 14 lutas vêm
da mesma fonte e do mesmo evento.

| ord | confronto (grafia UFCStats) | situação | registrado em | `fight_uid` |
|---|---|---|---|---|
| 0 | Waldo Cortes Acosta × Johnny Walker | novo no card | 05/nov | — (evento futuro) |
| 1 | Tatiana Suarez × Virna Jandiroba | no card | 29/out | — |
| 2 | Davey Grant × Elijah Smith | novo no card | 05/nov | — |
| 3 | Billy Elekana × Lucas Fernando | no card | 29/out | — |
| 4 | Priscila Cachoeira × Nina Milosevic | no card | 29/out | — |
| 5 | Mitch Raposo × Marcos Degli | no card | 29/out | — |
| 6 | Keiichiro Nakamura × Ollie Schmid | no card | 29/out | — |
| 7 | Gabriel Lorenco × Thomas Petersen | **duplicata de grafia resolvida** | 29/out | — |
| 8 | Karine Silva × Gabriella Fernandes | no card | 29/out | — |
| 9 | Jose Delano × Murtazali Magomedov | no card | 29/out | — |
| 10 | Austin Bashi × Carlos Calderon | novo no card | 05/nov | — |
| 11 | Mantas Kondratavicius × Andre Petroski | novo no card | 05/nov | — |
| 12 | Seokhyeon Ko × Wellington Turman | no card | 29/out | — |
| 13 | Jonny Parsons × Jose Souza | novo no card | 05/nov | — |
| — | **Gabriel Bonfim × Sean Brady** | **retirada confirmada** | 29/out | — |

`fight_uid` vazio em todas: o evento ainda não aconteceu, então não existe
luta na base de resultados para dar id. Isso é `aguardando`, não ausência de
identidade.

**Nenhuma divergência não resolvida.** O script interrompe a escrita se
houver (`scripts/reconciliar_card.py` sai com código 1 e não grava nada).

### O que foi feito com cada caso

**Duplicata de grafia** — `Gabriel Lourenco` (29/out) e `Gabriel Lorenco`
(05/nov) eram a mesma luta em duas linhas. Nenhuma das duas grafias está na
base: ele é estreante, então `best_name_match` devolveu o input intacto nas
duas vezes e o `_same_fight` não casou.

Resolução: **mantida a linha original** (previsão 0,4767, de 29/out), adotada
a grafia do UFCStats, e herdados apenas os dados de **mercado** da linha
duplicada (sharp 0,6089, odd da casa 2,35) — que descrevem a mesma luta e
foram capturados antes do cutoff. A previsão de 05/nov (0,4772) foi
descartada, não mesclada. Há guarda contra inverter lados ao renomear: se a
grafia nova trocasse A e B, o nome fica como estava e sai aviso.

**Retirada confirmada** — Gabriel Bonfim × Sean Brady mantém a linha, a
previsão (0,5195) e a odd da casa (1,79), e recebe
`estado_resultado = retirado_do_card`. Fica **inelegível**: sai dos
denominadores e dos testes de consistência card↔template↔histórico. Nenhuma
previsão ou odd foi transferida para outro adversário.

**Novos confrontos** — cinco entraram como registros novos, com previsão e
sinal sharp próprios, capturados antes do evento. Não herdaram nada.

### Main event e rounds

Main event confirmado por **dois sinais independentes da fonte
autoritativa**: a ordem da tabela do evento (linha 0) e o **nome do próprio
evento** ("Cortes Acosta vs. Walker"). Não foi derivado da primeira linha da
API — a API não ordena o card.

**Rounds programados: não disponíveis.** A página de evento futuro do
UFCStats não traz `scheduled_rounds`; ele só aparece na página da luta depois
do evento. O projeto não usa essa informação: a previsão de duração/faixa de
round foi removida em ago/2026. Registrado como lacuna conhecida, não
preenchido por inferência.

## B. Alcance da corrupção de datas

**Origem:** `scripts/ciclo_semanal.py`, execução de 01/nov/2026 12:00. O passo
3 chamava `publish(CSV, atual["nome"], event_date=atual["data"])` onde
`_card_atual()` lia `data/ciclo_estado.json`, que estava adiantado em relação
ao CSV. As 12 lutas de 31/out foram gravadas sob `event_date` 07/nov.

| alcance | afetado? | evidência |
|---|---|---|
| `data/prediction_history.csv` | **sim** | 22 linhas em 07/nov no commit `b7e30d4` |
| `data/odds_template.csv` | não | conferido: 10 linhas em 07/nov, nenhuma de 31/out |
| `data/raw/fights.csv` (treino) | **não** | base vem do UFCStats, não do histórico |
| `data/processed/fight_features.csv` | **não** | `event_date` 2026-11-07 ausente das features |
| modelos treinados | **não** | features não afetadas |
| métricas da série | **não** | as 12 linhas estavam sem `actual_winner`; acurácia, P&L e CLV só contam linha fechada |
| `docs/index.html` | gerado com elas | commit `b7e30d4`, **nunca publicado** |
| `main` / remoto | **não** | `main` está em `61b3503`; os 6 commits afetados vivem só na branch, e a suíte barrou o push |

**Execuções:** uma só (01/nov 12:00). As rodadas de 02, 03, 04 e 05/nov
pararam em "Card vigente ... nao registro outro antes dele acontecer".

**Commits com o estado corrompido:** `b7e30d4` (22 linhas). Corrigido em
`4c30bc7` (12 linhas removidas). Nada foi recalculado porque nenhuma métrica
derivada havia sido afetada — as linhas nunca entraram em denominador.

**O que detectou:** `TestCardTemplateHistorico`, criado dois dias antes, e a
trava de suíte-antes-do-push, que registrou `SUÍTE VERMELHA -- trabalho
commitado localmente, SEM push`.

## C. Identidade do card validada entre as quatro fontes

`src/identidade.py::validar_identidade_do_card()` cruza **CSV, template,
histórico e estado local** e levanta `IdentidadeInconsistente` em qualquer
divergência. Ligada ao `ciclo_semanal` **antes** de qualquer escrita: em
conflito o ciclo sai com código 1 e não regenera nada.

Ela já achou um defeito no primeiro uso: `_anexar_ao_template` desistia
quando a data já aparecia no arquivo, então as **6 lutas novas de 07/nov
nunca chegaram ao template** — e luta sem linha no template nunca fecha.
Corrigido para completar o que falta em vez de pular o bloco inteiro.

## D. Os 20 registros sem `fight_uid`

O `fight_uid` **nunca** é derivado de nome ou data: é sempre a URL da luta no
UFCStats. O que varia é como a linha foi casada com aquela luta, e isso agora
é declarado em `uid_origem`:

| origem | linhas | significado |
|---|---|---|
| `ufcstats_exato` | 178 | os dois nomes bateram literalmente |
| `ufcstats_fuzzy` | 9 | um dos lados casou por similaridade — **certeza menor** |
| (vazio) | 20 | sem id |

Os 20 sem id, por motivo:

| motivo | linhas | podem ser identificadas? |
|---|---|---|
| `aguardando` (07/nov, evento futuro) | 14 | sim, quando o evento entrar na base |
| `nao_aconteceu` (cancelada / adversário trocado) | 4 | **não** — a luta não existe |
| `retirado_do_card` | 1 | **não** |
| **`vencedor` sem id** | **1** | **não hoje** |

A última é a única preocupante: **Mehemmedeli Osmanli × Ilimbek Akylbek
(26/set)**, fechada à mão, cuja grafia na base é `Mahammadali Osmanli` —
distante demais para o matcher, que se recusa a adivinhar. Tem resultado e
não tem identidade inequívoca.

**Regra aplicada:** linha sem `fight_uid` fica fora de qualquer operação que
exija identificação inequívoca. Hoje isso significa: não é usada para casar
revanche, e `compare_to_market` já pula par ambíguo sem data. Os testes
cobram que todo id tenha origem declarada e que linha sem id não declare
origem.

## E. O experimento, registrado como pedido

**Denominador zero muda de comportamento.** Com o shrinkage, lutador sem
nenhuma luta anterior deixa de ter `NaN` imputado pelo pipeline e passa a
receber exatamente o prior. Isso **não** é neutro, e explica por que o grupo
0 foi o único a se mover no walk-forward (Δ log loss −0,0025, IC95 [−0,0049,
−0,0001]; Brier **não** confirma).

**O centro do prior é sensível à ponderação.** Uma linha por luta, cumulativa:
a média de `career_win_rate` cai de 0,6449 para 0,5895 ao usar uma linha por
lutador — **5,5 pontos**. Todo valor encolhido se desloca junto.

**A estabilidade de k não valida a estimação.** O k quase não muda (8,8 vs
8,7), mas isso diz apenas que a *força* do prior é robusta à ponderação; não
diz nada sobre o *centro*, que não é, nem sobre a forma da distribuição
assumida. Tratar os dois como a mesma coisa seria o erro.

## F. Itens 8 e 13, descritos com precisão

**Item 8 — timestamps, fonte e prova de cotação.** O histórico guarda
`sharp_prob`, `sharp_best_odd`, `close_prob`, `close_best_odd`, `odds_*_tp` e
as medianas, mas **não guarda**: o instante exato de cada cotação; qual casa
deu cada preço (só a Pinnacle é implícita no sharp); o mercado consultado;
o status de disponibilidade; e nenhuma referência verificável à resposta
bruta da API. Consequência: não dá para auditar depois se um preço estava
realmente disponível naquele instante, nem reconstruir uma captura. Fechar
exigiria persistir a resposta da API (ou um hash dela com a URL sanitizada) e
uma coluna de `captured_at` por campo de preço.

**Item 13 — manifesto de reprodução.** Existe `models/training_metadata.json`
com `trained_at`, método de calibração, contagens e intervalos de datas de
treino/calibração/teste, e `generator_fingerprint()` carimba o HTML
publicado. **Faltam**: hash do dataset de features e dos CSVs brutos usados
no treino; versões das bibliotecas (sklearn, pandas, numpy, lightgbm) no
momento do treino; e a lista de `fight_id` de cada fatia. Sem isso, um treino
não é reprodutível bit a bit, e não há como provar que dois resultados vieram
do mesmo dado.

Nenhum dos dois foi implementado nesta rodada. **A integridade não está
concluída.**

## G. Estado dos testes

**549 passando, 1 vermelho** — `test_report_freshness`, porque
`docs/index.html` é anterior às mudanças em `prediction_history.py`. Não
regenerei: a tarefa pede sem publicação.

Testes novos nesta rodada: origem e grau de certeza do id, validação de
identidade entre as quatro fontes (incluindo o cenário exato da corrupção de
01/nov), e o estado `retirado_do_card` como inelegível.

---

# Fechamento operacional (06/11/2026)

Baseline mantido. Shrinkage **não promovido**. Nenhum experimento novo de
modelo. Sem push, sem publicação, sem captura de odds.

## 1. Renderização separada da escrita

`generate_card_report` fazia sync de resultados, registro no histórico e
desenho no mesmo caminho: não havia como regenerar a página sem mexer nos
registros, e qualquer conserto de layout arrastava o histórico junto. Essa
era a causa, não o teste vermelho.

Novo parâmetro `somente_render=True` (e `--somente-render` no CLI): pula
`sync_results_from_template`, pula `record_card_predictions` e pula toda
consulta de odds. Desenha a partir dos snapshots já gravados.

Conferência antes/depois, por hash sha256:

| arquivo | antes | depois |
|---|---|---|
| `prediction_history.csv` | `c629581ed238f377` | `c629581ed238f377` |
| `odds_template.csv` | `39181fed84a0ac4b` | `39181fed84a0ac4b` |
| `upcoming_card_odds.csv` | `803eb259f0acf312` | `803eb259f0acf312` |

**Nenhum registro tocado.** Só `docs/index.html` mudou.

## 2. Identidade estável do evento × título de apresentação

A identidade estável é a **data**; `event_name` é título e muda. Em 07/nov o
main event saiu e o UFCStats renomeou o evento.

Confirmado que o rename não quebrou o agrupamento:

| verificação | histórico | template |
|---|---|---|
| datas com mais de um título | 0 | 0 |
| títulos usados em mais de uma data | 0 | 0 |
| título divergente entre os dois arquivos | 0 | — |

17 datas, 207 linhas, 16 eventos fechados na série — os mesmos de antes do
rename. **Não duplicou o evento, não criou outro agrupamento, não dividiu a
série.** Três testes passam a cobrar isso (`TestIdentidadeDoEventoVsTitulo`),
porque um rename parcial produziria exatamente a divisão silenciosa.

## 3. Consolidação Gabriel Lourenco/Lorenco, revista

A consolidação anterior estava errada no ponto que importa: ela juntava a
**probabilidade de 29/out** com **preços capturados em 05/nov** na mesma
linha, e o resultado se parecia com um pré-registro original que nunca
existiu nessa forma.

Desfeito. Agora:

- a linha do histórico volta ao pré-registro original — `model_prob_a`
  0,4767 e **as odds de mercado que ela tinha na origem**, isto é, nenhuma
  (`sharp_prob`, `sharp_best_odd`, `ev_sharp`, `odds_a_tp`, `odds_b_tp` e
  `ev_sharp_tp` vazios);
- o mercado posterior foi para `data/snapshots_mercado.csv`, em linha
  própria, com `captured_at`, `origem` e uma coluna `precisao_do_horario`
  dizendo que o carimbo é o **início da execução do job**, lido de
  `data/ciclo_semanal.log`, e **não** o instante da cotação.

```
2026-11-07,Gabriel Lorenco,Thomas Petersen,2026-11-05T12:00:01-03:00,
  "ciclo_semanal; linha duplicada por grafia, removida na reconciliacao de 05/11",
  "inicio da execucao do job ...; nao e o instante da cotacao",
  0.6089,1.61,0.9803,2.35,1.55,0.9438
```

Se esses preços forem usados para simular uma decisão posterior, isso é uma
**série própria com versão própria** — não entra no pré-registro. Nada foi
feito nesse sentido nesta etapa.

A grafia adotada continua sendo a do UFCStats (`Gabriel Lorenco`), com a
guarda contra inversão de lados.

## 4. Correção sobre o grupo zero

A afirmação anterior — "o grupo 0 foi o único a se mover" — está **errada**.
Todos os grupos tiveram diferença nas métricas no walk-forward:

| grupo | Δ log loss | Δ Brier |
|---|---|---|
| 0 | −0,0025 | −0,0011 |
| 1–2 | +0,0015 | +0,0005 |
| 3–5 | −0,0001 | +0,0001 |
| 6+ | −0,0014 | −0,0006 |

O que distingue o grupo 0 é só que o **intervalo do log loss não cobriu
zero** — e o Brier dele cobre. Isso não é "o único que mudou".

E a explicação que dei — de que o ganho vem do denominador zero passar a
receber o prior no lugar do `NaN` imputado — é **hipótese**, não resultado.
É verdade que as entradas do grupo mudam por esse mecanismo, mas atribuir a
diferença de métrica especificamente a ele exigiria **ablação**: rodar o
shrinkage aplicando-o só onde o denominador é maior que zero e comparar.
Não foi feito, e esta etapa não abre experimento novo. Fica registrado como
hipótese a testar.

## 5. Os nove casamentos por similaridade

`uid_origem = ufcstats_fuzzy`. **A existência da URL não prova o casamento**:
o `fight_uid` foi obtido *a partir* da correspondência aproximada, então ele
não pode validá-la. São correspondências com grau de certeza menor, listadas
para auditoria:

| data | nossa grafia | grafia na base | leitura |
|---|---|---|---|
| 11/07 | Benoit St. Denis | Benoit Saint Denis | abreviação; alta |
| 11/07 | Cong Wang | Wang Cong | ordem invertida; alta |
| 18/07 | Levi Rodrigues | Levi Rodrigues Jr. | sufixo; alta |
| 18/07 | Seok Hyun Ko | Seokhyeon Ko | romanização; alta |
| 25/07 | Muhammad **Said** | Muhammad **Saidov** | **sobrenome truncado — a mais frágil das nove** |
| 08/08 | Yadier DelValle | Yadier del Valle | espaçamento; alta |
| 08/08 | Billy Goff | Billy Ray Goff | nome do meio; alta |
| 08/08 | Carlos Diego Ferreira | Diego Ferreira | primeiro nome omitido; média |
| 29/08 | Cameron Nelson | Cam Nelson | diminutivo; média |

Duas merecem atenção. **Muhammad Said / Muhammad Saidov** é exatamente a
família de falso positivo que a docstring do `best_name_match` descreve
(`"Muhammad Said" ~0,79 com "Muhammad Naimov"`). E **Cameron Nelson** é o
nome do incidente de 29/ago que custou uma perna EV>1 (`"Cameron Nelson"`
casou com `"Cameron Else"`), ainda que aqui o par seja plausível.

Nenhuma das nove foi confirmada contra uma segunda fonte. Elas continuam
utilizáveis, mas marcadas — e qualquer operação que exija identificação
inequívoca deve tratá-las como as 20 sem id.

## 6. Entrega

**Suíte: 550 passando, 0 falhando.**

Commits desta rodada, na branch `correcao-estreantes`:

| hash | assunto |
|---|---|
| `1631f2f` | §3 integridade: relógio por luta, modelo padrão único, chave fora do log |
| `4c30bc7` | verificações, identidade/estado, walk-forward |
| `4df881e` | reconciliação do card de 07/11, alcance da corrupção, identidade validada |
| (este) | fechamento operacional |

`main` segue em `61b3503`, intocada. Nada publicado.

**Resumo do diff desta etapa:** `src/card_report.py` (modo render-only +
flag de CLI), `src/prediction_history.py` (linha nova nasce `aguardando`),
`tests/test_integridade_dados.py` (3 testes de identidade × título),
`data/prediction_history.csv` (mercado posterior removido de uma linha),
`data/snapshots_mercado.csv` (novo), `docs/index.html` (regerado sem tocar
registro).

## 7. Pendências que ficam abertas

**Item 8 — timestamps, fonte e prova de cotação.** O histórico guarda os
preços mas não o instante de cada cotação, qual casa deu cada um, o mercado
consultado, o status de disponibilidade, nem referência verificável à
resposta bruta. Não dá para auditar depois se um preço estava disponível
naquele instante. O `snapshots_mercado.csv` criado no item 3 é o primeiro
registro com `captured_at` e `origem` explícitos — e com a ressalva de que o
horário é o do job, não o da cotação. Fechar o item exige persistir a
resposta da API (ou hash dela com a URL sanitizada) e um `captured_at` por
campo de preço, no fluxo normal.

**Item 13 — manifesto de reprodução.** Há `models/training_metadata.json`
(`trained_at`, método de calibração, contagens, intervalos de data) e o
`generator_fingerprint()` no HTML. Faltam: hash do dataset de features e dos
CSVs brutos usados no treino, versões de sklearn/pandas/numpy/lightgbm no
momento do treino, e a lista de `fight_id` de cada fatia. Sem isso, um treino
não é reprodutível bit a bit.

**A integridade segue parcialmente pendente.** Rodada encerrada.

---

# Encerramento da rodada (06/11/2026)

Baseline mantido. **Shrinkage registrado como NÃO PROMOVIDO.** Nenhum
experimento novo. Sem push, sem publicação.

## 1. Identidade do evento: ID, não data

A definição anterior estava errada — eu havia escrito que "a identidade
estável é a data". **Data é atributo.** Dois eventos podem cair no mesmo
dia, e o título muda (07/nov virou "Cortes Acosta vs Walker" quando o main
event saiu).

Coluna nova `event_uid`: a URL do evento no UFCStats. Preenchida só quando
**todas** as lutas daquela data na base apontam para o mesmo `event_url`;
data com mais de um evento fica **pendente**, não arbitrada.

| | linhas |
|---|---|
| com `event_uid` | 192 |
| **pendentes** | **15** — evento de 07/nov, ainda não aconteceu |

As 15 pendentes são explicitamente pendentes, não identidade presumida. Um
teste cobra que nenhuma linha sem `event_uid` pertença a evento que já está
na base.

## 2. Os nove fuzzy: não confirmados e bloqueados

`identificacao_inequivoca(row)` exige `fight_uid` **e** `uid_origem ==
ufcstats_exato`. Os nove `ufcstats_fuzzy` ficam de fora: o uid deles saiu da
própria correspondência aproximada, então a URL não prova o casamento — ela
é consequência dele.

| | linhas |
|---|---|
| identificação inequívoca | **178** |
| fuzzy, não confirmados | **9** |
| sem id | 20 |

Nenhuma previsão antiga foi apagada ou reescrita. As nove linhas continuam
no histórico, com seus resultados e suas previsões; o que muda é que elas
não são aceitas onde identidade inequívoca é requisito.

### Métricas que podem ser afetadas por um casamento fuzzy errado

O risco não é o `fight_uid` em si — ele não entra em nenhuma métrica hoje.
O risco é que **o mesmo matcher aproximado** é usado em dois outros lugares:

| caminho | o que um casamento errado faria | métricas expostas |
|---|---|---|
| `fechamento.casar_resultado` | preencher `actual_winner` da luta errada | **acerto do modelo, acerto do mercado, P&L das pernas EV>1, divisão com/sem sharp** |
| `identidade.resolver` | atribuir `estado_resultado` errado | elegibilidade (denominadores, consistência card↔template↔histórico) |
| `compare_to_market` | — | nenhuma: casa por nome exato **e** data |
| séries de CLV | — | nenhuma: `clv` é `close_prob − sharp_prob`, não depende de id |

**Exposição real hoje: nenhuma.** Os nove são de eventos de 11/jul a
29/ago, todos fechados **à mão**, antes de `src/fechamento.py` existir
(26/out). Nenhum `actual_winner` da série foi preenchido por casamento
aproximado. O mecanismo, porém, está vivo para os eventos futuros — por isso
o bloqueio, e não só a anotação.

As duas mais frágeis continuam sendo **Muhammad Said / Muhammad Saidov** (a
família de falso positivo descrita na docstring do `best_name_match`) e
**Cameron Nelson / Cam Nelson** (nome do incidente de 29/ago que custou uma
perna EV>1).

## 3. `somente_render` com os preditores quebrados

O teste faz `predict_fn` e `method_fn` **lançarem erro**. Ele revelou um
defeito real: `analyze_card` chamava o preditor **antes** de olhar o
snapshot, então um relatório histórico inteiro dependia de o modelo atual
carregar, e qualquer falha mandava para "sem previsão" linhas que tinham
snapshot congelado.

Corrigido: o snapshot manda quando existe. Agora, com os preditores
quebrados:

- luta **com** snapshot suficiente continua desenhada, com a probabilidade
  do snapshot (`origem_previsao = "snapshot"`);
- luta **sem** snapshot vai para "sem previsão" com o motivo — indisponível,
  nunca recalculada em silêncio;
- snapshot parcial cobre só as lutas que tem;
- snapshot gravado **sem** probabilidade (luta que entrou no histórico sem
  previsão) continua sem — não vira oportunidade de recálculo.

Oito testes em `tests/test_somente_render.py`, incluindo um que confere por
sha256 que `generate_card_report(..., somente_render=True)` não altera
histórico, template nem CSV do card.

## 4. Horário do job como proveniência limitada

`data/snapshots_mercado.csv` ganhou `certifica_pre_cutoff = nao` e o campo
`limite_da_proveniencia`:

> `captured_at` é o **início da execução** do job, lido de
> `data/ciclo_semanal.log`. Não é o instante da cotação e **não certifica**
> que o preço foi obtido antes do cutoff da luta — o job roda por minutos e
> a casa move a linha dentro disso. Use como proveniência aproximada, nunca
> como prova temporal.

## 5. Entrega

**Suíte: 567 passando, 0 falhando.**

Commits da branch `correcao-estreantes` (`main` intocada em `61b3503`):

| hash | assunto |
|---|---|
| `1631f2f` | §3 integridade: relógio por luta, modelo padrão único, chave fora do log |
| `4c30bc7` | verificações, identidade/estado, walk-forward |
| `4df881e` | reconciliação do card de 07/11, alcance da corrupção, identidade validada |
| `1a1b374` | fechamento operacional: render separado da escrita |
| (este) | encerramento: identidade por ID, fuzzy bloqueado, render testado |

**Diff consolidado contra `main`: 28 arquivos, +3915 / −894.**

Novos: `RELATORIO_ESTREANTES.md`, `src/identidade.py`, `src/shrinkage.py`,
`scripts/reconciliar_card.py`, `data/snapshots_mercado.csv`,
`exp_shrinkage.py`, `exp_walkforward.py`, e cinco arquivos de teste
(`test_identidade`, `test_revanche_mercado`, `test_segredo_em_log`,
`test_shrinkage`, `test_somente_render`).

Alterados: `config.py`, `src/card_report.py`, `src/prediction_history.py`,
`src/evaluate.py`, `src/predict.py`, `src/line_shopping.py`,
`scripts/auto_capture.py`, `scripts/build_card.py`,
`scripts/ciclo_semanal.py`, `tests/test_integridade_dados.py`, mais os
artefatos de dado e a página.

## 6. Pendências explícitas

**Item 8 — timestamps, fonte e prova de cotação. ABERTO.** O histórico
guarda `sharp_prob`, `sharp_best_odd`, `close_prob`, `close_best_odd`,
`odds_*_tp` e as medianas, e **não guarda**: o instante de cada cotação; qual
casa deu cada preço; o mercado consultado; o status de disponibilidade; nem
referência verificável à resposta bruta da API. Consequência: não há como
auditar depois se um preço estava disponível naquele instante, nem
reconstruir uma captura. O `snapshots_mercado.csv` é o primeiro registro com
`captured_at`/`origem` — e já nasce declarando que **não certifica
pré-cutoff**. Fechar exige `captured_at` por campo de preço no fluxo normal,
mais a resposta da API persistida (ou hash dela com a URL sanitizada).

**Item 13 — manifesto de reprodução. ABERTO.** Existem
`models/training_metadata.json` (`trained_at`, método de calibração,
contagens, intervalos de data) e o `generator_fingerprint()` carimbado no
HTML. **Faltam**: hash do dataset de features e dos CSVs brutos usados no
treino; versões de sklearn, pandas, numpy e lightgbm no momento do treino; e
a lista de `fight_id` de cada fatia. Sem isso um treino não é reprodutível
bit a bit, e não dá para provar que dois resultados vieram do mesmo dado.

**Além desses dois:** as 15 linhas sem `event_uid` (evento de 07/nov) e as
20 sem `fight_uid` — das quais **uma tem vencedor e não tem identidade
verificável** (Mehemmedeli × Mahammadali Osmanli, 26/set).

---

**Rodada encerrada.** Correções operacionais documentadas, shrinkage não
promovido, integridade parcialmente pendente. A branch é candidata à
integração, não um estado final.

---

# Errata (06/11/2026)

Duas afirmações da seção "Os nove fuzzy: não confirmados e bloqueados"
estavam curtas demais. Ficam corrigidas aqui, sem reescrever o texto
original.

## 1. "Exposição real hoje: nenhuma" — errado

O que o fechamento manual estabelece é mais estreito: **aquele caminho
automático** (`fechamento.casar_resultado`) não preencheu nenhum
`actual_winner` das nove linhas, porque elas foram fechadas antes de o
módulo existir.

Isso **não comprova que a identidade está correta**. O `fight_uid` das nove
veio de correspondência aproximada e segue não confirmado; e o preenchimento
manual tem a sua própria margem de erro, do mesmo tipo — eu também casei
nome por leitura. Dizer "exposição nenhuma" trocou "um caminho de erro está
afastado" por "não há erro", que é outra coisa.

Leitura correta: **as nove permanecem não verificadas.** O que se sabe é que
o erro, se existir, não entrou por `casar_resultado`.

## 2. O CLV depende de casamento correto, mesmo sem `fight_uid`

A tabela de métricas expostas dizia "CLV: nenhuma — `clv` é
`close_prob − sharp_prob`, não depende de id". A aritmética não depende, mas
**os dois termos dependem de casar a nossa luta com o evento certo da API**.

`fetch_sharp_probs` e `record_closing_lines` chegam ao evento por
`match_event`, que usa correspondência aproximada mais a tabela
`APELIDOS_API`. Um casamento errado ali anexa a `sharp_prob` e a
`close_prob` de **outra luta** à nossa linha, e o CLV resultante é lixo com
aparência de número — sem que nenhum `fight_uid` tenha sido consultado.

Métricas afetadas por esse caminho: **CLV por perna, CLV médio da série,
contagem de pernas que bateram o fechamento, `ev_sharp`, `ev_sharp_tp` e a
divisão com/sem respaldo sharp.**

Isto é um caminho de exposição **separado** dos nove fuzzy e **ativo hoje**,
em todo evento registrado. A guarda que existe é indireta: o teste
`test_probabilidade_de_mercado_descreve_o_LADO_DO_MODELO` usa a odd
registrada como testemunha independente e pegaria uma inversão grosseira,
com tolerância de 0,15. Casamento com uma luta de probabilidades parecidas
passaria.

**Fica registrado como pendência**, junto dos itens 8 e 13: não há
verificação que confirme, por luta, que a cotação veio do confronto certo.

---

# Fechamento operacional (06/11/2026)

Esta seção encerra as correções operacionais. O escopo é fechado: **nenhum
experimento de modelo novo**, baseline mantido, shrinkage segue fora da
produção.

## O que a errata deixou em aberto, e o que foi feito

A errata registrava uma pendência: *"não há verificação que confirme, por
luta, que a cotação veio do confronto certo"*. Era o caminho de exposição
mais largo do projeto, e estava ativo em todo evento registrado. Está
fechado.

### 1. Casamento entre confronto e cotação

`match_event` percorria os eventos da API e devolvia **o primeiro** cujos
dois nomes casassem por correspondência aproximada. Cinco modos de falha,
todos silenciosos — a linha saía com preço e com aparência de saudável:

| defeito | consequência |
|---|---|
| primeiro candidato ganha | dois eventos plausíveis, e ele escolhia um sem contar que havia outro |
| nenhuma conferência de data | **revanche**: os mesmos dois nomes em datas diferentes casam igualmente bem, e o preço de um evento entrava na linha do outro |
| nenhuma conferência de organização | o feed é `mma_mixed_martial_arts`; em 05/nov trouxe onze lutas de outras organizações para a data do card |
| fuzzy aprovava sozinho | é a forma de erro que casou "Cameron Nelson" com "Cameron Else" e **inverteu o lado** numa perna EV>1 |
| homônimo | a base tem **nove** nomes repetidos, entre eles `Jean Silva` e `Bruno Silva`, ambos ativos. Nome exato não é identidade |

`src/casamento.py::casar` aprova somente quando, ao mesmo tempo: há
**exatamente um** candidato na janela de data; os dois participantes são os
dois lados de um mercado `h2h` daquele evento — o que prova mercado e
**orientação dos lados** de uma vez; a evidência de cada lado é **forte**
(igualdade literal ou apelido conferido à mão); e nenhum dos nomes é
homônimo sem desambiguação externa. Todo o resto é bloqueio **com motivo
explícito** e contagem de candidatos.

Correspondência aproximada continua existindo, com o papel certo: ela
**sugere**. O candidato volta identificado, marcado como `fuzzy`, para
conferência humana — e não aprova seleção. `match_event` permanece restrito
a *descobrir* o card (`build_card`, `card_sources`), com esse limite escrito
na própria docstring.

**Organização** é pergunta sobre o card, não sobre a cotação: prova-se por o
confronto estar no card do UFCStats daquela data. `build_card` calculava a
coluna `fonte` e **a jogava fora na escrita**; agora persiste. Número de
casas não é prova de pertencimento ao UFC — segue usado só para decidir
inclusão no CSV, que é outra decisão.

Para evento **futuro** a identificação não depende de a luta já ter
acontecido: usa-se o card do UFCStats mais a evidência de casamento, ambos
verificáveis antes da luta. O id interno (`fight_uid`) e os ids externos
(`evento_id_api`, `participantes_fonte`) ficam em colunas separadas. Nenhuma
URL de resultado é inventada.

### 2. Proveniência do preço realmente utilizado

A prova de uma chamada cobre **o corpo daquela resposta**. Ela não diz qual
preço foi usado no cálculo, e **não diz nada sobre a Thunderpick** — que vem
de outra fonte, por outro caminho. Uma resposta que comprova a Pinnacle não
comprova a cotação da casa.

Agora há duas camadas:

- `data/proveniencia/chamadas.csv` + corpo comprimido com sha256 — uma linha
  por chamada, inclusive da Thunderpick, gravada **antes de filtrar**, para
  que o descarte também seja auditável;
- `data/proveniencia/cotacoes.csv` — uma linha por **preço efetivamente
  usado**: fonte, casa, mercado, evento, os dois participantes, lado, odd,
  `last_update` que a fonte declara, status, início programado e a
  referência ao corpo persistido.

**Política de validade:** `VALIDADE_HORAS = 6`, escolhida pela **cadência de
captura** — `auto_capture` roda de hora em hora, então uma cotação com mais
de seis horas significa seis tentativas seguidas falhadas. O número não foi
varrido contra resultado nenhum. Escolher esse parâmetro depois de olhar P&L
seria ajustar o critério para melhorar o passado.

**Limite declarado e preservado:** o instante registrado é o da **leitura**
da cotação, não o instante em que a casa formou o preço — a fonte não
informa. Não prova disponibilidade em outro momento, nem que o preço seria
aceito num bilhete.

### 3. O gate de seleção

Seis exigências, todas verificáveis a partir do que está gravado: estado,
identidade, organização, casamento declarado e forte nos dois lados, preço
verificável e — quando a seleção é para agora — atualidade e caráter
pré-luta das **duas** referências (sharp e casa).

O gate age **apenas onde se propõe uma entrada**. Métrica descreve o que foi
registrado; seleção propõe uma ação. Só a segunda exige prova. **Nenhuma
previsão histórica foi reescrita.**

### 4. Decisão posterior, em vez de reescrita

Recapturar a odd de um card já pré-registrado e escrever o preço novo na
linha antiga produziria uma linha que não descreve instante nenhum —
probabilidade de um dia, preço de outro —, e o CLV calculado dela mediria a
confusão, não o movimento de linha.

`data/decisoes.csv` guarda uma decisão por avaliação, com instante próprio,
**apontando** para o pré-registro de onde a probabilidade veio e para o
manifesto de treino que a sustenta. O log **anexa, nunca sobrepõe**: duas
avaliações do mesmo confronto em instantes diferentes são duas decisões, e
sobrepor apagaria justamente a informação de que o preço andou.

A elegibilidade da decisão usa as provas do **card recém-capturado**. A
linha antiga tem um preço sem prova e é correto que continue bloqueada: não
há prova retroativa.

### 5. Reprodução — e uma afirmação minha que a medição refutou

O manifesto ganhou as duas famílias que faltavam: **parâmetros** e
**modelo**. Os parâmetros são lidos dos estimadores no disco, não copiados
do código. `get_params(deep=True)` não bastava — o modelo de produção é um
`CalibratedClassifierCV` em volta de um wrapper em volta de um `Pipeline`, e
o deep para na primeira camada que devolve objeto. Sem descer, o manifesto
registrava `method` e `cv` e **perdia o `C` e o `max_iter`**, que são
justamente os parâmetros que mudam o modelo.

Sobre o nível de reprodução, **eu estava errado**. Eu havia escrito na
constante que a reprodução era apenas numérica dentro de tolerância e **não**
bit a bit, justificando com joblib não ser byte-estável e com a ordem de
soma do BLAS depender de threads. Era asseveração, não medição.

Medido: dois treinos consecutivos do mesmo dado, mesmos hiperparâmetros,
mesmas versões, produziram arquivos `.joblib` **byte a byte idênticos** e
previsões com diferença máxima **0,0** nas 14 lutas do card.

O que o manifesto **garante** continua sendo menos do que isso, e é o que
importa: ele detecta divergência. **Reprodução em outra máquina não foi
testada** — BLAS, threads e CPU podem mudar os últimos bits. Nada de
multiplataforma está comprovado.

Um alarme que sempre soaria foi corrigido: a comparação de parâmetros usava
`!=` no dicionário inteiro, e o imputer tem `missing_values=nan`. NaN nunca é
igual a si mesmo, então **todo** treino reportaria divergência, para sempre.

### 6. Um buraco que o teste integrado encontrou

`sync_results_from_template` preenchia `actual_winner` e deixava
`estado_resultado` em `aguardando`. Não era cosmético: o gate dispensa a
exigência de identidade para linha `aguardando` — evento futuro não tem
`fight_uid` porque a luta ainda não está na base de resultados —, então uma
linha **já resolvida** escapava dessa exigência.

Consertado em duas camadas: o sync avança o estado, e o gate deixou de
confiar só na string. O histórico real já estava consistente (187 linhas com
vencedor, todas com estado `vencedor`), então o buraco era latente e
apareceria no próximo fechamento.

## Estado do card de 07/11/2026

Reconciliado com as duas fontes: **14 confrontos confirmados**, todos com
UFCStats e API concordando (`fonte=ambas`), 13 com odd da casa e
proveniência por preço. Onze lutas de outras organizações ficaram fora,
listadas. A luta retirada (Bonfim × Brady, o main event que dava nome ao
evento) já estava marcada `retirado_do_card`.

| | |
|---|---|
| confrontos confirmados | **14** |
| seleções elegíveis | **13** |
| seleções bloqueadas | **1** — preço de entrada sem prova (a casa não lista Cortes Acosta × Walker) |

`data/prediction_history.csv` saiu desta etapa **byte a byte igual**.

Avaliado contra o pré-registro antigo — isto é, usando as provas da linha
antiga em vez das do card novo — **207 de 207 linhas ficam bloqueadas**: 192
por organização não confirmada (o CSV antigo não tinha a coluna `fonte`, e
ela não ganha prova retroativa), 10 por identidade e 5 por estado. É o
resultado correto, e é desconfortável de propósito.

## Testes

**635 aprovados, 0 ignorados, 0 falhas.**

Arquivos novos desta etapa:

- `tests/test_casamento.py` — revanche, homônimo, alias confirmado, inversão
  A/B, adversário substituído, evento errado, confronto ambíguo, mercado
  ausente, janela de meia-noite UTC, e um teste que confere a lista de
  homônimos **contra a base**, de modo a falhar se a base ganhar um nome
  repetido novo;
- `tests/test_decisoes.py` — o pré-registro sai byte a byte igual, o preço
  vem do card e a probabilidade do histórico, duas avaliações viram duas
  decisões, e cada forma de bloqueio;
- `tests/test_ciclo_completo.py` — a cadeia inteira, **trocando o preditor
  entre a publicação e o fechamento**: se alguma etapa recalcular, a
  probabilidade gravada muda e o teste falha;
- `tests/test_proveniencia_e_bloqueio.py` — proveniência, manifesto e as
  seis exigências do gate.

Dois testes afirmavam exatamente o comportamento que havia de ser corrigido,
e foram reescritos com o motivo no corpo:
`test_grafia_proxima_ainda_casa_pela_fuzzy` → `test_grafia_proxima_NAO_aprova_mais`
e `test_sem_data_continua_pegando_a_primeira` → `test_sem_data_NAO_casa_mais`.

## Segredos

A chave da API não está em nenhum arquivo versionado; `git log -S` confirma
que nunca entrou num commit; os endpoints gravados na proveniência estão
sanitizados; nenhum corpo guardado contém segredo. As quatro ocorrências de
`apiKey` no diff são a documentação do conserto, a própria regex do
sanitizador e testes com chave fictícia.

## Interface

Conferida em 1280×900 e em 375×812. No celular `scrollWidth == innerWidth`
(375), sem estouro horizontal. O painel de proveniência declara horários,
fonte, versão do modelo, manifesto e limitações, lista as seleções elegíveis
e os bloqueios por motivo, e separa as três leituras que eram lidas como
uma:

- **probabilidade de vitória** — o que o modelo estima;
- **valor esperado estimado** — essa probabilidade vezes o preço, calculado
  com o nosso próprio modelo, portanto **auto-referente**, e não evidência
  de valor;
- **qualidade do dado** — de quando o preço é, de que fonte, e se a prova
  dele existe.

Ausência de seleção elegível aparece **dita**, como resultado válido, nunca
como painel vazio.

## Pendências

1. **Validade de 6 h não foi validada contra comportamento de mercado.** Ela
   descreve a nossa cadência de captura, não a velocidade com que a linha
   anda. Validar exigiria medir movimento de preço, que é trabalho de outra
   natureza.
2. **Reprodução multiplataforma não testada.** O que se mediu vale para esta
   máquina.
3. **Os nove `ufcstats_fuzzy` continuam não confirmados.** Estão bloqueados
   para seleção e permanecem no histórico. Confirmá-los exige conferência
   manual contra o UFCStats, uma a uma.
4. **O pré-registro histórico não tem prova de preço** e não vai ter: a
   proveniência vale da sua criação em diante. As 207 linhas seguem válidas
   como registro de previsão e **não** sustentam seleção.
5. **`odds_a/b_decimal` (a mediana) não tem proveniência por preço.** Ela é
   escrita a partir da saída de `line_shopping` e serve de benchmark de
   mercado, não de preço de entrada — mas uma perna sem odd da casa cai nela,
   e por isso essa perna é bloqueada.

## Duas declarações, separadas de propósito

**Operação verificada.** O software faz o que esta seção descreve: o
casamento entre confronto e cotação é inequívoco ou bloqueia com motivo; a
proveniência cobre cada preço usado, inclusive o da casa; o pré-registro não
é reescrito; os bloqueios funcionam de ponta a ponta; e 685 testes passam.

**Vantagem estatística demonstrada: não.** Nada aqui produz evidência de
vantagem para apostas. O backtest mostra o mercado à frente nas divergências
(−14,3% por perna em 230 lutas), o EV exibido é auto-referente, e o conjunto
de teste foi consultado muitas vezes. Software correto não é vantagem: é
apenas a condição para que uma medição futura queira dizer alguma coisa.
Uma seleção **elegível** significa que as provas estão no lugar — não que a
entrada é boa.

---

# Revisão independente: dois defeitos no commit 7d70038 (06/11/2026)

A revisão do commit que fechava a versão operacional encontrou dois defeitos
que a minha própria verificação não pegou. Os dois têm a mesma assinatura: em
vez de falhar, entregavam **número plausível**. É a pior forma de erro num
sistema cujo trabalho é dizer quando não sabe.

## 1. Cinco decisões elegíveis citavam a cotação do adversário

`anexar_ao_card` já registrava **uma linha de proveniência por lado** — a odd
do lado A e a do lado B são preços diferentes, de participantes diferentes. Mas
guardava no CSV do card só a referência do lado A, e descartava a do lado B. E
a conferência (`preco_verificavel`) respondia apenas *"esta linha existe e o
corpo dela confere com o hash"*.

Isso é menos do que parece. Não dizia que a linha descrevia **este** confronto,
**este** lado, **esta** casa, **este** mercado nem **esta** odd. Uma perna no
lado B passava inteira apresentando a cotação do outro lutador: mesma chamada,
mesmo corpo, mesmo hash, só o lado trocado. Cinco das treze seleções elegíveis
de 07/11 estavam nesse estado.

O conserto tem três camadas:

| camada | o que mudou |
|---|---|
| persistência | o card leva `ref_tp_a` e `ref_tp_b`; o histórico leva `ref_tp_b` |
| escolha | a decisão pega a prova **do lado escolhido**, sem recurso à do lado A quando falta a do lado B — era por aí que o defeito entrava |
| conferência | `proveniencia.validar_cotacao` confere a decisão **contra o que ela afirma** |

`validar_cotacao` exige, em ordem: a linha existe e o corpo dela confere; casa,
mercado e evento batem; o **par de participantes** é o deste confronto; a
**orientação** é a do lado escolhido, e não a do adversário; a odd declarada é a
da prova; o mercado não está suspenso; e, por fim, que aquela odd, para aquele
lado, **apareça no corpo persistido**. Campo que divergir é bloqueio com
motivo, nunca aviso.

Referência de outro confronto bloqueia. Mercado suspenso bloqueia (`status != 1`
na casa, ou palavra de fechamento num feed de texto). Odd que o conteúdo
guardado não sustenta bloqueia — porque a linha de cotação *afirma* um preço e o
corpo é quem o *comprova*; afirmação que o conteúdo não sustenta não é prova.

O mesmo passou a valer no gate de seleção: ele conferia sempre `ref_tp`, que é o
lado A, e uma perna no lado B — como as nove pernas em que o modelo escolhe o
lado B — era aprovada pela prova do adversário.

## 2. A probabilidade era lida pela orientação errada

`model_prob_a` é a probabilidade de `fighter_a` **do snapshot** — do lutador que
estava na coluna A quando a previsão foi congelada. O card é capturado depois e
pode listar o mesmo confronto na ordem inversa.

`avaliar_com_precos_novos` lia a probabilidade pela ordem **do card**. Com as
ordens invertidas, a decisão usava `1-p`: a probabilidade do adversário.

Agora a probabilidade vem dos **participantes do pré-registro**, e só depois a
odd correspondente é escolhida no card atual. A regressão fixa o caso do
enunciado e as quatro combinações de ordem:

| ordem do snapshot | ordem do card | retorno bruto esperado |
|---|---|---|
| Alpha, Beta | Alpha, Beta | 1,26 |
| Alpha, Beta | **Beta, Alpha** | 1,26 |
| **Beta, Alpha** | Alpha, Beta | 1,26 |
| **Beta, Alpha** | **Beta, Alpha** | 1,26 |

Alpha com p = 0,70 e odd 1,80. O defeito produzia 0,54 na segunda linha —
plausível, e errado.

**Este defeito não chegou a morder no card de 07/11**, porque o card e o
snapshot estavam na mesma ordem: as cinco decisões tinham o número certo e a
prova errada. Latente não é inofensivo: bastava uma reconciliação de card
trocando os lados.

## Retificação, sem apagar nada

`data/decisoes.csv` ganhou `prob_lado` (a probabilidade do lado escolhido, para
que o EV seja auditável sem refazer a conta de cabeça) e `retifica`.

As 14 decisões antigas **ficam** no arquivo. Apagar a decisão errada esconderia
que ela existiu, e é justamente o registro dela que torna a correção auditável.
As 14 novas apontam para a rodada anterior pelo campo `retifica`. A migração de
cabeçalho reescreve o arquivo com as colunas de hoje e **todas** as linhas que
já estavam lá, com as colunas novas vazias — porque elas de fato não foram
registradas naquele momento, e preenchê-las agora seria inventar o que a decisão
antiga afirmava.

As referências por lado do card de 07/11 foram **lidas do índice**, não
registradas agora: as duas linhas de cada luta já estavam gravadas na captura de
16:27 UTC, e só a do lado B era descartada na escrita. As odds do card conferem
com as do índice nas 13 lutas. Nenhuma prova foi fabricada para trás.

## Contagem depois da correção

| | |
|---|---|
| confrontos confirmados (07/11) | **14**, todos `fonte=ambas` |
| seleções elegíveis | **13** |
| bloqueadas | **1** — preço de entrada sem prova (a casa não lista Cortes Acosta × Walker) |

O total não mudou: as cinco decisões afetadas tinham o número certo e passam a
ter a prova certa. Sobre as linhas do **pré-registro** (que é coisa diferente de
decisão), as 15 de 07/11 seguem bloqueadas — 14 por organização não confirmada,
1 por estado — e, no histórico inteiro, 207 de 207.

**Suíte: 652 testes, 0 falhas** (eram 635). As duas classes de regressão estão
em `tests/test_decisoes.py`; `tests/test_ciclo_completo.py` cobre o caso da
prova da outra luta atravessando o ciclo inteiro.

## O que esta revisão diz sobre a verificação anterior

A verificação anterior conferiu que **existia prova** para cada preço usado.
Não conferiu que a prova **falava daquele preço**. São perguntas diferentes, e a
segunda é a que importa: um hash válido de um corpo válido é compatível com a
cotação errada do confronto certo.

---

# Segunda revisão independente: o corpo e as versões (06/11/2026)

A revisão do `54c64c3` confirmou as correções anteriores e achou mais dois
bloqueadores. O primeiro estava **um nível abaixo** do defeito que eu tinha
acabado de consertar, o que é informação sobre a verificação e não só sobre o
código.

## 1. A prova lia o corpo sem contexto

Eu havia feito `validar_cotacao` exigir que a odd declarada **aparecesse no
corpo persistido**. A leitura do corpo, porém, colhia pares `(nome, odd)`
**soltos**, varrendo a estrutura inteira.

O corpo de uma captura da casa tem 31 lutas e 62 preços. O do feed de odds tem
68 eventos e 1289 preços. Nesse volume, "Alpha paga 1,80 em algum lugar deste
JSON" não é prova de que **Alpha × Beta** paga 1,80 — e era isso que a
conferência estava aceitando. Uma prova do confronto errado, com o nome certo e
o preço certo, passava.

`cotacoes_do_corpo` passa a devolver cada cotação **com o contexto dela**,
tirado da mesma sub-estrutura:

| campo | de onde vem |
|---|---|
| evento, início | do objeto do evento |
| os **dois** participantes | do mesmo objeto que traz o preço |
| casa, mercado | do nó da casa e do mercado (feed), ou da linha da chamada (casa) |
| lado, preço | do par declarado, nunca de aproximar um nome de um número próximo |
| status, `last_update` | do mesmo nó |

Duas formas reconhecidas: `{home, away, odd_home, odd_away, inicio, status}`
(a casa) e `evento → bookmakers → markets → outcomes` (o feed). Casa e mercado
que o corpo não nomeia — a página da Thunderpick não diz o nome da própria casa
— herdam o que a **linha da chamada** registra (`fonte`, `markets`): é a mesma
prova bruta, não um preenchimento nosso. Forma desconhecida devolve lista
vazia, que bloqueia; não é aprovação por omissão.

E `validar_cotacao` ganhou uma checagem que faltava: a linha de cotação e a
chamada têm de declarar a **mesma fonte**. Uma linha podia dizer `thunderpick`
apontando para o corpo do feed.

O diagnóstico nomeia o caso perigoso em vez de dizer "não achei":

> a odd 1.8 para `'Alpha'` aparece no corpo, mas em OUTRO confronto (Alpha × Outro)

## 2. O pré-registro estava sendo substituído

`prediction_history.csv` guardava **uma linha por luta** e a reescrevia a cada
regeração enquanto o evento não acontecia. A regra estava documentada e tinha
motivo — enquanto o resultado não chega, atualizar as odds deve refrescar a
previsão. O efeito é que a versão anterior **desaparecia**.

Medido nesta sessão: regerar o card de 07/11 moveu seis odds da mediana, seis
da casa e uma probabilidade (Gabriel Lorenco, 0,4767 → 0,4772). Nada disso
estava errado. Nada disso era recuperável, e nenhuma decisão podia dizer contra
**qual** versão foi tomada.

`src/snapshots.py` → `data/previsoes_snapshots.csv`, que **só recebe linhas**.
Toda versão leva:

```
prediction_id      identificador da versão
criado_em          instante UTC em que ESTA versão foi criada
substitui          prediction_id da versão que ela sucede
manifesto_sha256   manifesto de treino em vigor
modelos_sha256     hash dos arquivos de modelo
ref_tp/ref_tp_b/sharp_ref    as provas de preço daquela versão
inicio_programado  início da luta, como a fonte da cotação declara
pos_inicio         a versão nasceu depois do início daquela luta?
```

O histórico passa a ser **ponteiro** para a versão em vigor (`prediction_id`).
Nenhuma versão é apagada ou reescrita: substituir deixou de acontecer, e o que
havia antes continua endereçável. Cada decisão em `decisoes.csv` carrega o
`prediction_id` da versão **exata** que usou, não "a mais nova".

### `pos_inicio` é por luta

Previsão criada depois de a luta começar não é previsão. A versão fica
registrada — o arquivo não esconde linha — e o gate bloqueia qualquer entrada
apoiada nela (`BLOQUEIO_POS_INICIO`).

A conta é **por luta**, não por card: num evento as lutas começam ao longo de
cinco horas, e usar o horário do card liberaria uma previsão feita depois do
primeiro combate. Início desconhecido fica **vazio**, não `False`: a terceira
resposta é visível, e quem exige atualidade já bloqueia cotação sem início
programado.

### Versões históricas: a lacuna fica declarada

As 207 linhas que já existiam ganharam um snapshot de semeadura com `criado_em`
**vazio** e este texto em `temporalidade`:

> NAO RECUPERAVEL: semeado do historico em 2026-11-06. O arquivo antigo
> guardava uma linha por luta e a reescrevia, entao nem o instante desta versao
> nem as versoes anteriores dela existem. Esta linha e o ultimo estado
> conhecido, nao o primeiro.

Inventar uma data agora seria pior que admitir a lacuna. Hoje o arquivo tem 235
versões: 207 de semeadura e 28 criadas pelas duas regenerações desta sessão,
cada uma declarando a versão que sucede.

## 3. `somente_render` em clone limpo

Ele ainda carregava o presente por dois caminhos:

1. `check_data_freshness()` lê a base bruta;
2. a previsão de luta **aberta** passava pelo preditor, porque `frozen` só
   cobria luta encerrada.

Agora `analyze_card(..., somente_snapshot=True)` **não chama preditor nenhum**
— nem para tentar. "Tentar e cair no `except`" dependeria de o erro ser de um
tipo previsto, e importar um modelo falha de muitas formas. As versões gravadas
cobrem a luta aberta; luta sem versão fica **INDISPONÍVEL**, nunca recalculada
em silêncio.

**Confirmado em clone limpo** (`git clone --single-branch`, sem `models/`, sem
`data/processed/`, sem `data/raw/*` além dos dois arquivos versionados):

```
python -m src.card_report data/raw/upcoming_card_odds.csv \
  --output saida.html --card-name "UFC Fight Night: Cortes Acosta vs Walker" \
  --event-date 2026-11-07 --somente-render
```

Desenhou as 14 lutas e **os 614 percentuais da página saíram idênticos** aos da
página publicada. As diferenças são as que devem existir: sem fotos, sem
`og:image`, manifesto marcado como `ausente` e o frescor da base declarado como
**não verificável** — porque ele não foi verificado. O clone não ganhou nenhum
arquivo: o modo não escreve.

## 4. Achado de passagem: regerar destruía a prova do casamento

`evento_id_api`, `casamento` e `participantes_fonte` nascem junto do sinal
sharp. Uma regeração com `--no-sharp` chegava com as três vazias, e a linha
passava a ser bloqueada por "casamento não comprovado" — um bloqueio sobre a
**nossa escrituração**, não sobre a luta. Foi o que as regenerações desta sessão
fizeram com nove linhas de 07/11.

Agora são preservadas, e as duas metades de `casamento` invertem quando o CSV
troca os lados — a evidência é por lado, e copiar direto colaria a de um
lutador no outro. As nove linhas tiveram os valores restaurados do estado
anterior às regenerações de hoje; foram as regenerações que os apagaram.

## Contagem depois das correções

| | |
|---|---|
| confrontos confirmados (07/11) | **14** |
| seleções elegíveis | **13** |
| bloqueadas | **1** — preço de entrada sem prova (a casa não lista Cortes Acosta × Walker) |

As linhas do **pré-registro** de 07/11 seguem bloqueadas, 15 de 15: **9** por
cotação sem prova — o sinal sharp daquelas linhas nunca teve referência por
preço, e isso já era assim antes desta sessão —, **5** por casamento não
comprovado e **1** por estado. Prova que não existe não ganha prova retroativa.

`decisoes.csv` tem 42 linhas: três rodadas de 14, cada uma com o seu `retifica`.
O log é trilha, não estado.

**Suíte: 685 testes, 0 falhas** (eram 652 na revisão anterior). Hashes dos
arquivos de produção idênticos antes e depois de uma rodada completa.

## O que estas duas revisões dizem sobre a verificação

A primeira conferiu que **existia prova** para cada preço. Não conferiu que a
prova **falava daquele preço**.

A segunda conferiu que a prova falava daquele preço. Não conferiu que ela
falava **daquele confronto** — o nome e o número certos, achados em outra luta
do mesmo arquivo, passavam.

A cada rodada o erro fica mais fino e mais específico, o que é o
comportamento esperado. Também é a razão de nenhuma dessas rodadas autorizar a
frase "está verificado" sem dizer *verificado contra o quê*.
