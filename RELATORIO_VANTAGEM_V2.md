# Relatório `vantagem-v2`

**2026-11-06.** Corrige a validade metodológica do experimento e prepara a
coleta prospectiva. Leia primeiro `ERRATA_VANTAGEM_V1.md`.

**O baseline continua em produção, inalterado.** Nenhum experimento de feature
ou de algoritmo foi aberto.

**Classe destes resultados: desenvolvimento / exploração.** O contraste
confirmatório rodou uma vez, no `c21f7b9`. Nada recalculado depois da errata
ganha retrospectivamente esse rótulo, e o próprio código escreve isso em
`classe_dos_resultados`.

---

## 1. Temporalidade certificada: zero

A correção que mais muda a leitura do projeto. O v1 chamou 179 lutas de
"população temporalmente adequada" porque constavam do pré-registro — isso
confunde **disponibilidade do dado** com **comprovação temporal**.

Medido por linha (`src/temporalidade.py`), nas 179 elegíveis:

| item | `comprovada` | `desconhecida` | `incompativel` | `ausente` | `nao_certificada` |
|---|---|---|---|---|---|
| previsão | **0** | **179** | 0 | 0 | — |
| referência de mercado (mediana) | — | — | — | — | **179** |
| preço executável | **0** | 0 | 0 | **179** | — |

> **Observações com temporalidade certificada: 0 de 179.**

Por quê, item por item:

- **previsão**: as 179 apontam para versões de **semeadura**, cujo `criado_em`
  é vazio e cuja temporalidade está declarada **não recuperável** desde
  06/11/2026. Não há instante para comparar com o início da luta;
- **mediana**: não tem proveniência por preço — sem casa, mercado, instante de
  leitura ou corpo persistido por linha. Não é falta de horário, é falta de
  **origem**, e a fonte não é certificável por linha;
- **preço executável**: nenhuma das 179 tem `ref_tp`. A proveniência por preço
  só passou a existir em out/2026, e a coluna só foi preenchida no card de
  07/11, que ainda não tem resultado. **19 lutas têm a odd da casa preenchida e
  nenhuma tem a prova dela** — preço presente não é preço comprovado.

As 179 permanecem como **análise histórica exploratória com temporalidade não
certificada**. Nenhum horário, proveniência ou previsão antiga foi fabricado.

## 2. A auditoria das odds, com o motivo certo

O v1 listou "toda linha extraída 103 a 3338 dias **depois** do evento" como se
fosse a prova de que `data/raw/market_odds.csv` não serve. **A data de extração
é quando nós obtivemos o dado, não quando a casa formou o preço** — uma odd de
fechamento legítima, raspada quatro anos depois de um arquivo, pode ser
perfeitamente pré-luta.

O defeito real, independente do atraso:

- **nenhuma coluna de casa** — não se sabe de quem é o preço;
- **nenhuma coluna de mercado**;
- **um único par** favorito/azarão, sem os dois lados por casa;
- **um único `odds_extract_ts`** para o arquivo inteiro, o que o torna inútil
  como instante de qualquer linha específica;
- **nenhum conteúdo bruto persistido** para conferir.

Conclusão igual (não serve para simular decisão antiga), **motivo corrigido**:
não é "a odd veio depois da luta", é "não há como saber de onde a odd veio nem
quando esteve disponível".

## 3. Resultado do candidato, recalculado

Com a classificação temporal no lugar, os números do contraste não mudam —
porque `w = 0` continua sendo o mínimo e `C ≡ A`:

| fatia | mercado | modelo | combinação |
|---|---|---|---|
| desenvolvimento (91 lutas, 8 eventos) | **0,5589** | 0,6288 | 0,5589 |
| avaliação (88 lutas, 8 eventos) | **0,6660** | 0,7057 | 0,6660 |

Δ log loss (C − A) na avaliação = **+0,0000**, IC95 [−0,0000, +0,0000].
Zero **por construção**, porque o procedimento escolheu não usar o modelo.

Exploratório, B − A na avaliação: **+0,0398**, IC95 **[−0,0472, +0,1264]** — o
intervalo contém zero.

Artefatos em `data/experimentos/v2/`. **Os do v1 continuam em
`data/experimentos/`, intocados** (hashes conferidos).

## 4. Precisão, poder e dimensionamento — cinco coisas separadas

| | o que é | valor aqui |
|---|---|---|
| **IC observado** | precisão com que esta amostra estimou este contraste | meia-largura **0,0868** para B − A |
| **precisão desejada** | largura de IC que queremos alcançar | **não escolhida** |
| **efeito mínimo de interesse** | menor diferença que mudaria a decisão | **não escolhido** |
| **significância** | erro tipo I aceito | 5% (do IC95) |
| **poder** | chance de detectar um efeito que exista | **não calculado** |

A função foi renomeada de `detectavel` para **`precisao_observada`**.
Meia-largura de IC **não** é efeito mínimo detectável: um efeito menor que ela
pode existir e ser detectado por amostra maior, e um efeito maior que ela pode
passar sem significância nesta amostra.

**Correção aritmética:** o v1 disse que 0,0868 era "maior que 0,117".
**0,0868 é menor que 0,117.**

**O desvio de B − A não dimensiona C − A.** Com `w = 0`, C − A é zero por
construção e tem variabilidade **zero**. Reaproveitar o desvio de B − A
pressupõe que um candidato futuro, ainda não formulado, terá a variabilidade do
candidato B.

### 4.1 Sensibilidade, com os pressupostos à vista

*Se* um candidato futuro tiver desvio por evento `s` do contraste de interesse,
e *se* o efeito mínimo de interesse for `δ`, o número de **eventos
independentes** para um IC95 de meia-largura `δ` é da ordem de `(1,96·s/δ)²`:

| `s` | `δ = 0,05` | `δ = 0,03` | `δ = 0,02` |
|---|---|---|---|
| 0,08 | ~10 | ~28 | ~62 |
| 0,13 | ~26 | ~73 | ~163 |
| 0,20 | ~62 | ~171 | ~385 |

**Isto é sensibilidade, não plano.** `s` é desconhecido para um candidato que
não existe, `δ` não foi escolhido, e a fórmula ignora que os eventos têm números
diferentes de lutas. **Falta informação para dimensionar a amostra**, e a tabela
de "meses" do v1 sai.

### 4.2 Incerteza da calibração

Wilson por faixa assume independência, e lutas do mesmo card não são
independentes. As tabelas agora traem o próprio limite: colunas `n_eventos`,
`wilson_otimista` e `ic95_agrupado_*` por bootstrap de evento. Medido na
avaliação, mercado:

| faixa | n | n_eventos | obs | Wilson | agrupado por evento |
|---|---|---|---|---|---|
| [20%,40%) | 21 | 8 | 0,476 | [0,283, 0,676] | **[0,250, 0,708]** |
| [40%,50%) | 10 | 6 | 0,600 | [0,313, 0,832] | **[0,250, 0,833]** |
| [50%,60%) | 17 | 8 | 0,765 | [0,527, 0,904] | **[0,462, 0,958]** |
| [60%,80%) | 33 | 8 | 0,606 | [0,437, 0,753] | **[0,455, 0,758]** |
| [80%,100%) | 4 | 3 | 0,750 | [0,301, 0,954] | **[0,000, 1,000]** |

O agrupado é mais largo em toda faixa — Wilson era otimista, como a errata (E9)
diz.

### 4.3 Amostra pequena não transforma resultado em ruído

O v1 disse que o AUC de 0,272 do modelo no grupo de 1–2 lutas prévias (n = 25)
é "perfeitamente compatível com ruído". **Compatível com ruído não é ser
ruído.** O número é 0,272, o `n` é 25, o intervalo é largo e inclui tanto 0,5
quanto valores bem abaixo. **A amostra não decide** — nem a favor nem contra.

## 5. Quatro conclusões reduzidas ao que foi medido

1. **"o modelo não acrescenta informação ao mercado"** → *nesta* combinação
   (pooling logístico de um peso), *nesta* população (179 lutas **não
   certificadas**), *nesta* janela, o peso ótimo foi **zero**. **Não** demonstra
   ausência universal de informação residual: outra formulação — por segmento,
   com interação, não linear — não foi testada;
2. **"o teto são as features"** → é conclusão de `DISCRIMINACAO.md`, por outro
   caminho. `w = 0` é **compatível** com ela e **não a demonstra**;
3. **"um peso de 0,05 daria ~0,001 de log loss"** → número inventado por
   analogia com a curva de perda. **Retirado**;
4. **o resultado é negativo** → isto permanece, com o escopo de (1).

## 6. Política de seleção, corrigida e fixada antes dos novos resultados

O v1 apostava em `model_side` mesmo com `w = 0`: a probabilidade vinha do
mercado e o **lado** continuava vindo do modelo — e apostar no lado que o modelo
aponta é, na prática, apostar no favorito.

A política do v2 (§5 de `PROTOCOLO_VANTAGEM_V2.md`):

1. lucro esperado líquido dos **dois** lados: `p_C(lado) × odd(lado) − 1`;
2. cada lado com a **sua própria** prova válida;
3. no máximo **um** lado por luta: o de maior lucro líquido **positivo**;
4. **empate exato** → recusa com motivo;
5. lado sem preço comprovado fica **indisponível**, nunca recebe o preço do
   adversário;
6. unidade fixa, nenhuma aposta real.

`model_side` não entra. Medido no card de 07/11: a política nova seleciona
**Davey Grant** (odd 3,62) — o lado **oposto** ao que o modelo aponta naquela
luta — e **Billy Elekana** (odd 3,75). O v1 selecionava uma perna só, a do
modelo.

**Capturas repetidas**: cada luta conta **uma** aposta por candidato, a primeira
decisão com seleção; as demais liquidam com `unidades = 0` e o motivo no campo.
Travado em teste.

## 7. Qualificação prospectiva: prova positiva

O v1 chamava de prospectivo todo evento fora da lista de desenvolvimento — isto
é, **ausência de contra-evidência**. Agora cada item é verificado:

| exigência | situação hoje |
|---|---|
| candidato congelado antes da decisão | ✅ `253046a9cee9958c`, congelado em 2026-11-06T21:19:26Z |
| decisão antes do início programado daquela luta | ✅ verificável pela cotação |
| previsão com temporalidade comprovada | ✅ **a partir de 07/11**, com `criado_em` |
| preço executável comprovado no instante | ✅ onde há `ref_tp` por lado |
| referência de mercado certificada | ❌ **a mediana não tem proveniência por preço** |
| evento nunca usado em estimação nem inferência | ✅ os 16 do v1 ficam **queimados para sempre** |

**A classe máxima alcançável hoje é `prospectiva_parcial`** — tudo comprovado
**menos** a referência de mercado. **Isso não é "prospectiva plenamente
auditável"**, e o rótulo diz isso em cada linha.

Ensaio no card de 2026-11-07 (`--dry-run`, nada gravado): das 14 lutas,
**13 `prospectiva_parcial`** e **1 `nao_qualificada`** (Cortes Acosta × Walker,
sem cotação alguma, logo sem início programado verificável). Duas seleções.

O v1 bloqueava apenas os 8 eventos de desenvolvimento; os 8 de avaliação
**gastaram a inferência** e agora também estão bloqueados. São 16.

## 8. Congelamento e reestimação

Manter `w = 0` deixa C idêntico a A. **Acumular eventos não muda esse
contraste**, e o v2 não finge que muda.

O candidato está preservado em `data/experimentos/candidatos.json`,
append-only. Regra fixada agora:

> `w` só pode ser reestimado com **pelo menos 10 eventos** de sombra que nunca
> entraram em estimação nem em inferência. A reestimação **transforma esses
> eventos em desenvolvimento**, e o candidato novo só é avaliado em eventos
> **posteriores** ao seu congelamento.

**Marcos inferenciais:** a cada **20 eventos de avaliação** do candidato em
vigor. Fora deles, relatório **descritivo** — contagem, cobertura, classificação
temporal — sem teste e sem intervalo apresentado como decisão.

## 9. Comandos

```bash
# experimento (recalculo -> exploratorio depois da errata)
python -m scripts.experimento_vantagem --saida data/experimentos/v2

# inspecionar candidatos congelados
python -m scripts.candidato --listar

# congelar um candidato (so quando a regra da secao 8 permitir)
python -m scripts.candidato --peso 0.0 --versao vantagem-v2 \
    --descricao "..." --dev 2026-07-11 --consultados 2026-09-05

# INSPECIONAR o proximo evento, sem gravar
python -m scripts.sombra --event-date 2026-11-07 --dry-run

# REGISTRAR antes das lutas
python -m scripts.sombra --event-date 2026-11-07

# LIQUIDAR depois, sem reescrever a previsao
python -m scripts.sombra --event-date 2026-11-07 --liquidar
```

## 10. O que ainda depende de dados ausentes

1. **Avaliação prospectiva plena** exige proveniência por preço da **referência
   de mercado**: casa, mercado, dois lados, instante de leitura, corpo bruto e
   hash. O pipeline já faz isso para a Pinnacle e para a Thunderpick; a mediana
   é calculada e gravada **sem** prova. **Não foi consertado nesta rodada** e não
   exige API nova — usa a resposta que já é persistida;
2. **Qualquer P&L.** Preço executável comprovado existe em **0** das 179 lutas
   resolvidas. As 19 com odd da casa não têm a prova dela;
3. **A temporalidade das 179** não é recuperável e não vai ser;
4. **Dimensionamento de amostra**: `s` do contraste futuro e `δ` de interesse
   não existem;
5. **Os grupos de experiência**: `n` de 8 a 28 por célula;
6. **Reprodução em outra máquina** não testada.

## 11. Duas declarações

**Infraestrutura verificada.** A classificação temporal é por observação e
declara zero quando é zero; prospectivo exige prova positiva de seis itens; a
seleção não depende mais do modelo quando o peso dele é zero; capturas repetidas
contam uma aposta; registros do v1 preservados e os 16 eventos queimados;
**775 testes, 0 falhas**; hashes dos sete registros de produção idênticos antes
e depois.

**Vantagem estatística demonstrada: não**, e esta rodada reduziu o que o v1
afirmava em vez de ampliar. A população certificada caiu de "179" para **zero** e
a classe prospectiva máxima passou a ser **parcial**. Nada aqui indica
rentabilidade nem prontidão para arriscar dinheiro.

## Verificação de 09/11/2026

A seleção em modo sombra agora exige que a previsão já exista no instante
registrado e que cada cotação usada tenha sido lida até esse instante, dentro
das seis horas de validade e antes do início programado. Registros de
qualificação insuficiente ficam fora das unidades simuladas. Repetir um mesmo
`registro_id` resulta em uma única liquidação.

O candidato `253046a9cee9958c` tinha o hash do protocolo vazio. Ele foi
preservado. O novo congelamento `1f4a6f4961a658bd` associa o documento pelo
hash, com o mesmo peso e os mesmos eventos de desenvolvimento e avaliação.
O horário novo vale para decisões futuras. No ensaio de leitura do card de
07/11, as 14 lutas foram recusadas como não qualificadas.

A coleta ainda chega, no máximo, a `prospectiva_parcial`. Falta vincular a
mediana à resposta bruta, aos dois lados e ao horário da captura. As séries
existentes conservam a classificação temporal registrada anteriormente.

---

# Rodada 3 (2026-11-09): a mediana passou a ser verificável

Fecha a lacuna que mantinha a classe máxima em `prospectiva_parcial`. Sobre o
commit `48595a9`. **Nenhuma chamada de API foi consumida**: tudo saiu de
respostas sintéticas e de arquivos já capturados.

## O que foi ligado

`src/mediana.py`, índice em `data/proveniencia/medianas.csv`. Cada referência
de mercado amarra: a captura exata (`chamada_ref` + hash do corpo), o instante
real de leitura, o id do evento na fonte, os dois participantes na orientação
usada, o mercado `h2h`, as casas incluídas e as duas odds.

Quatro decisões de projeto, cada uma com motivo:

- **`captured_at` entra na semente da `ref`.** O hash é do conteúdo, então duas
  capturas com corpo idêntico colapsam na mesma `chamada_ref`. Sem o horário na
  semente elas virariam a mesma prova, e a de ontem comprovaria disponibilidade
  hoje;
- **uma regra só.** `scripts/build_card.py::_medianas` passou a ser um embrulho
  de `mediana.calcular`. Duas implementações da mesma regra divergiriam em
  silêncio e a conferência compararia o número contra outra coisa;
- **casamento estrito**, `casamento.casar` e não `match_event`. Revanche,
  confronto errado e correspondência ambígua bloqueiam, e a luta entra no card
  **sem** prova em vez de com prova errada. Se a prova estrita achar odds
  diferentes das que o card usou, a prova é descartada e registrada como
  ausente;
- **`captured_at` e `last_update` da casa ficam em campos separados.** O
  primeiro é o nosso relógio; o segundo é o que a fonte declara, quando
  declara.

A conferência reabre a captura, valida o hash, **recalcula** as duas medianas
com a mesma regra e precisão, e compara com as odds que entraram na decisão e
com as casas registradas. Depois checa a ordem temporal contra o instante da
decisão: captura anterior à decisão, dentro da validade, anterior ao início
programado daquela luta, e os dois horários com fuso.

## Resultado no modo sombra

| | classe |
|---|---|
| com a mediana provada e tudo mais verificado | **`avaliacao_prospectiva`** |
| sem a prova da mediana, resto verificado | `prospectiva_parcial` |
| mediana adulterada, hash divergente, confronto errado, captura vencida ou posterior à decisão | bloqueio **com motivo específico** |

A mediana continua **não sendo preço executável**: o retorno simulado usa a
cotação comprovada do lado escolhido. No ensaio sintético a seleção sai em
Alpha @ 4,50 (executável), com a mediana em 3,50.

## Comprovado com dados sintéticos

Tudo nesta rodada. `scripts/demo_mediana.py` percorre o caminho inteiro —
captura, casamento, registro, preço executável, versão da previsão, decisão e
classificação — em diretório temporário:

```bash
python -m scripts.demo_mediana
```

Saída: `avaliacao_prospectiva` com a prova, `prospectiva_parcial` sem ela.

**35 testes** em `tests/test_mediana.py`: regra e precisão; casa que precifica
só um lado; captura íntegra virando prospectiva; inversão A/B preservando
atleta, odd e retorno; corpo idêntico em horários diferentes; revanche;
confronto errado; correspondência ambígua; hash divergente; corpo ausente;
mediana recalculada divergente da registrada; odd usada diferente da provada;
casas trocadas; captura vencida, posterior à decisão e pós-início; horário sem
fuso; início divergente entre provas; ausência de prova mantendo parcial; e
captura repetida sem duplicar unidades.

## Comprovado com captura real: nada ainda

`data/proveniencia/medianas.csv` **não existe** em produção. Nenhum card real
passou pelo caminho novo, porque montar um exige `scripts.build_card`, que
consome chamada da The Odds API — fora do escopo desta rodada.

Dry-run no card real de 2026-11-07, hoje (09/11):

```bash
python -m scripts.sombra --event-date 2026-11-07 --dry-run
```

14 lutas, **14 `nao_qualificada`**, 0 seleções. Correto e esperado: o evento já
aconteceu, a decisão seria posterior ao início, as cotações estão muito além da
validade, e o card foi montado antes da emenda — não tem `ref_mediana`. **Linha
antiga não ganha vínculo retroativo**, e não se procura uma captura posterior
para preencher o buraco.

## Registros preservados

O índice de medianas é **novo**; nada existente foi convertido. O CSV do card
ganhou `n_casas` e `ref_mediana`; card antigo sem essas colunas continua
legível e fica **sem certificação** (teste próprio).

Três candidatos em `data/experimentos/candidatos.json`, append-only:
`253046a9`, `1f4a6f49` e **`ea79fb4b` (em vigor)**. O terceiro existe porque a
emenda §4.2 mudou o `sha256` do protocolo, e `qualificar` confere esse hash:
**mudar o protocolo exige candidato novo**. Os anteriores continuam no arquivo
e continuam legíveis — o `protocolo_sha256` identifica qual texto cada um
congelou, e os bytes daquele texto estão no histórico do git. Nenhum candidato
foi editado. `w = 0` e os 16 eventos queimados seguem iguais.

## Verificação

**815 testes, 1 falha.** A falha é
`tests/test_identidade.py::TestEventUid::test_linha_sem_event_uid_fica_pendente_e_nao_presumida`
e **não vem destas mudanças** — reproduz com elas removidas. É estado de dados
de produção: o evento de 2026-11-07 já está na base bruta, e as 15 linhas dele
no histórico continuam `aguardando`, sem resultado e sem `event_uid`. O ciclo
do evento não foi fechado. É tarefa do fluxo de produção, na `main`, e esta
rodada não altera produção.

Hashes conferidos antes e depois: `prediction_history.csv`,
`previsoes_snapshots.csv`, `decisoes.csv`, `proveniencia/chamadas.csv`,
`proveniencia/cotacoes.csv`, `raw/upcoming_card_odds.csv`, `odds_template.csv`
e `docs/index.html` — **oito, todos idênticos**.

## O que ainda falta de evidência real

1. **uma captura real pelo caminho novo.** Exige `scripts.build_card`, que
   consome crédito da The Odds API. Nada foi gasto nesta rodada;
2. **um card futuro** para registrar em sombra: o de 07/11 já passou;
3. **fechar o ciclo de 07/11** na produção — é o que a falha acima aponta;
4. a mediana certificada **não** muda `w = 0`: C continua idêntico a A, e
   acumular eventos não muda esse contraste.

## Correções após a revisão independente de `33c19b0`

- A montagem normal grava `ref_mediana` ao casar o mesmo evento e reproduzir
  as mesmas medianas e contagem de casas do card. O teste exercita
  `build_card.main()` com fontes simuladas e verifica a prova gravada.
- Cada início é conferido na própria fonte. A decisão é bloqueada ao chegar
  ao primeiro horário entre o feed e a casa, mesmo que a outra fonte ainda
  indique início futuro.
- A verificação exige uma ocorrência com o mesmo corpo e horário no índice
  de chamadas, recompõe a referência e confere os metadados. A classe parcial
  fica reservada à ausência de prova. Uma prova quebrada bloqueia com motivo.
- A emenda recebe um novo congelamento (`0ae27ee8b716ab72`), com o mesmo peso
  e os mesmos eventos excluídos. Os congelamentos anteriores são preservados.

A liquidação segue por candidato. Totais entre candidatos não representam
apostas únicas. O fechamento de 07/11 e a triagem dos dados sintéticos do
índice histórico permanecem separados desta revisão. A captura real segue
pendente.

## Ajustes após o primeiro ensaio real

O ensaio relatado para 14/11 abriu uma captura real e verificou dez medianas.
A etapa de previsão ocorreu cerca de 36 horas depois, quando os preços já
estavam vencidos. Ele permanece como diagnóstico de integração.

A revisão tratou as diferenças de caixa em McKinney, os aliases explícitos
Bia/Beatriz Mesquita e Charles/Charlie Radtke e a união por dois nomes com
evidência forte e candidato único. A grafia do UFCStats continua no card.
Adversários diferentes, como no caso Juan Diaz, permanecem em linhas distintas
e com a origem declarada.

Fontes dos aliases:
- https://www.ufc.com/athlete/bia-mesquita
- https://www.ufc.com/news/official-judges-scorecards-ufc-293-adesanya-vs-strickland

O painel do relatório passou a receber o CSV da execução. A sequência abaixo
exige uma cópia exportada sem .git, com base, modelos e credencial disponíveis.
Ela verifica esses requisitos antes da captura e encerra ao falhar uma etapa.

```bash
python -m scripts.ensaio_sombra --event-date 2026-11-14 --card-name "UFC 334: Gane vs. Hokit" --chromium-ufcstats
```

A opção de Chromium é aplicada apenas à leitura do UFCStats. O card, o HTML
local e as provas ficam na cópia isolada. A geração usa --no-sharp e a etapa
final usa --dry-run, em sequência.

Verificação local desta revisão: 824 testes passaram, 14 foram pulados e quatro
falharam. Três falhas precisam dos CSVs locais ausentes e uma exige regenerar
docs/index.html após a alteração do gerador. O HTML versionado foi preservado.
Os nove caminhos de registros e HTML foram conferidos por hash. O ensaio com
uma captura nova ainda precisa ser executado no ambiente com base e modelos.

## Segundo ensaio real relatado em 11/11

O ensaio isolado de `572df6e` relatou 13 lutas, das quais 12 receberam
`avaliacao_prospectiva` e uma ficou sem preço executável. A captura do feed
(`be3ca5abde49f8cc`) ocorreu às 15:02:50 UTC, o preço da casa às 15:03:14,
a previsão às 15:03:19 e o dry-run às 15:03:21. Os arquivos desse ensaio estão
na cópia temporária local do operador. Ele permanece fora da amostra oficial.

A revisão seguinte adiciona Charles/Charlie Radtke à tabela da Thunderpick e
encaminha `--chromium-ufcstats` no ciclo, tanto na montagem quanto na consulta
do nome. A opção permite reproduzir a escolha de navegador do ensaio.

Testes desta revisão: 39 focados passaram. Na suíte completa, 830 passaram,
14 foram pulados e quatro falharam pelas mesmas limitações locais já
registradas acima (três CSVs ausentes e HTML anterior ao gerador). Os nove
caminhos de registros e HTML conservaram os hashes.

O segundo ensaio já qualificou o fluxo completo com captura real. A mudança
do alias foi testada nas duas orientações e contra adversário diferente.
A próxima captura pode ocorrer no próximo registro planejado, com o código
desta revisão, sem repetir uma chamada apenas para testar o alias.
