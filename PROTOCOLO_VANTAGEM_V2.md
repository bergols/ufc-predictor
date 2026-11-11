# Protocolo `vantagem-v2`

**Escrito em 2026-11-06, sobre o commit `c21f7b9`**, em resposta a
`ERRATA_VANTAGEM_V1.md`. Branch `pesquisa-vantagem`.

`PROTOCOLO_VANTAGEM.md` (v1) **continua válido como está** e não foi reescrito.
Este arquivo não finge que as correções estavam previstas no v1: ele existe
porque não estavam.

**O que é confirmatório e o que não é.** O contraste confirmatório do v1 rodou
uma vez, no `c21f7b9`. Tudo recalculado depois da errata é **desenvolvimento /
exploração**. O v2 **não** reabre contraste confirmatório: ele conserta a
infraestrutura e define quando um contraste futuro poderá ser confirmatório.

---

## 1. O que o v2 não faz

- não altera modelo de produção, hiperparâmetro, feature, algoritmo ou
  estratégia;
- não abre experimento de feature nem de algoritmo;
- não toca `data/prediction_history.csv`, `previsoes_snapshots.csv`,
  `decisoes.csv`, `proveniencia/`, `docs/`;
- não consome API paga, não agenda nada, não publica nada;
- não apaga nem substitui registro histórico ou prospectivo existente.

## 2. Hipótese (inalterada)

**H1.** As probabilidades do modelo acrescentam informação às do mercado.
**H0.** Não acrescentam, e o peso ótimo do modelo é zero.

O v1 mediu, *nesta combinação, nesta população e nesta janela*, `w = 0`. Isso
**não** demonstra ausência universal de informação residual, e o v2 não trata a
pergunta como encerrada.

## 3. Classificação temporal por observação

Substitui a afirmação do v1 de que o pré-registro era "população temporalmente
adequada". **Disponibilidade do dado e comprovação temporal são coisas
diferentes.**

### 3.1 O que conta como prova

Um instante registrado em fonte verificável, comparável ao **início programado
daquela luta**:

- **previsão**: `criado_em` da versão em `data/previsoes_snapshots.csv`;
- **preço**: `captured_at` da linha em `data/proveniencia/cotacoes.csv`, com o
  corpo da chamada conferindo pelo hash **e** a linha descrevendo aquele
  confronto, lado, casa, mercado e odd.

### 3.2 O que não conta

- a **data do evento** — num card as lutas começam ao longo de cinco horas;
- a coluna de odds estar **preenchida** — preço presente não é preço comprovado,
  e muito menos executável comprovado;
- a linha existir num arquivo cujo **propósito** é ser pré-registro;
- a **ausência de contra-evidência**.

### 3.3 As classes, por item

| classe | significado |
|---|---|
| `comprovada` | há instante verificável e é anterior ao início daquela luta |
| `desconhecida` | não há instante verificável, ou não há início para comparar |
| `incompativel` | há instante verificável e é **posterior** ao início |
| `ausente` | o item não existe |
| `nao_certificada` | a **fonte** não é certificável por linha (a mediana) |

`desconhecida` e `incompativel` não são a mesma coisa e não são agregadas.

Uma observação é **certificada** quando previsão **e** preço usado são
`comprovada`. Cada linha de saída carrega a classe e a **evidência efetivamente
encontrada**, não a esperada.

### 3.4 Situação medida hoje

| item | `comprovada` | `desconhecida` | `incompativel` | `ausente` | `nao_certificada` |
|---|---|---|---|---|---|
| previsão | 0 | **179** | 0 | 0 | — |
| mediana | — | — | — | — | **179** |
| preço executável | 0 | 0 | 0 | **179** | — |

**População certificada: zero.** Declarada como zero. As 179 lutas são
**análise histórica exploratória com temporalidade não certificada** e assim
permanecem, salvo evidência independente verificável por linha. Nada foi
fabricado para preencher a lacuna.

## 4. Qualificação prospectiva: prova positiva

Uma observação só vale como **avaliação prospectiva** se **todas** forem
verificadas:

1. **candidato congelado antes da decisão** — `congelado_em` do candidato em
   `data/experimentos/candidatos.json` anterior ao `decidido_em`;
2. **decisão antes do início programado daquela luta** — não da data do evento;
3. **previsão com temporalidade `comprovada`**;
4. **preço executável `comprovado`** no instante da decisão, do lado usado;
5. **referência de mercado certificada**;
6. **evento nunca usado** em estimação nem em análise inferencial.

Quatro coisas separadas, cada uma com horário e prova próprios:

| | o que é |
|---|---|
| (a) previsão do modelo | versão imutável, `criado_em` |
| (b) referência do mercado | mediana devigada, **sem** proveniência |
| (c) preço executável | odd da casa, por lado, com `captured_at` |
| (d) instante da decisão | quando a linha de sombra foi escrita |

**(a) pode ser anterior a (c)** — é o caso normal: o card é montado antes e o
preço é relido depois. Isso fica **documentado por linha**, e a exigência é que
(c) seja comprovadamente anterior ao início da luta, não anterior a (a).

### 4.1 Classes de qualificação

| classe | quando |
|---|---|
| `avaliacao_prospectiva` | 1 a 6, **todas** |
| `prospectiva_parcial` | 1 a 4 e 6, mas (5) falha |
| `desenvolvimento` | o evento já foi usado em estimação ou inferência |
| `nao_qualificada` | qualquer outra falha, com os motivos listados |

**Hoje a classe máxima alcançável é `prospectiva_parcial`**, porque a mediana não
tem proveniência por preço. Isso **não** é "prospectiva plenamente auditável", e
o relatório não a apresenta como tal.

**Os 16 eventos do v1 ficam queimados para sempre** — os 8 de desenvolvimento
estimaram o peso, os 8 de avaliação gastaram a inferência. O v1 só bloqueava os
primeiros.

### 4.2 Contrato de dados da referência de mercado — IMPLEMENTADO

> **Emenda de 2026-11-09.** Quando este protocolo foi escrito, a mediana era
> calculada e gravada **sem proveniência por preço**, e por isso a classe
> máxima alcançável era `prospectiva_parcial`. O contrato abaixo passou a ser
> implementado em `src/mediana.py`. O texto original dizia "não foi feito nesta
> rodada"; ficou feito, e a emenda registra isso em vez de reescrever a
> história.
>
> **Consequência de versionamento:** esta emenda muda o `sha256` deste arquivo.
> Candidatos congelados contra o texto anterior continuam no
> `data/experimentos/candidatos.json` e continuam legíveis — o `protocolo_sha256`
> de cada um identifica exatamente qual texto ele congelou, e os bytes daquele
> texto estão no histórico do git. Um candidato **não** é editado para apontar
> para o texto novo: mudar o protocolo exige **candidato novo**, e é assim que a
> cadeia fica auditável.

Para alcançar `avaliacao_prospectiva`, a referência de mercado precisa de:
captura identificada com hash do corpo e instante real de leitura, id do evento
na fonte, os dois participantes com a orientação A/B, o mercado `h2h`, as casas
incluídas e as odds de cada lado usadas no cálculo.

Implementação (`src/mediana.py`, índice em `data/proveniencia/medianas.csv`):

- **sem chamada nova**: reaproveita a resposta bruta que
  `line_shopping.fetch_live_odds` já persiste com `registrar_chamada`;
- **corpos idênticos em horários diferentes são provas diferentes**: o hash é do
  conteúdo, então duas capturas iguais têm a mesma `chamada_ref`; por isso a
  `ref` da mediana inclui o `captured_at`;
- **uma regra só**: `mediana.calcular` é a função que `scripts/build_card.py`
  usa para montar o card. Duas implementações da "mesma" regra divergiriam em
  silêncio e a conferência compararia o número contra outra coisa;
- **casamento estrito**: `casamento.casar`, não `match_event`. Revanche,
  confronto errado e correspondência ambígua bloqueiam, e a luta entra no card
  **sem** prova de mercado em vez de com prova errada;
- **conferência**: reabre a captura, valida o hash, **recalcula** as duas
  medianas com a mesma regra e precisão, e compara com as odds que entraram na
  decisão e com as casas registradas;
- **ordem temporal**: a captura tem de ser anterior à decisão, dentro da
  validade e anterior ao início programado daquela luta. `captured_at` (nosso
  relógio) e `last_update` da casa ficam em campos separados;
- **continua não sendo preço executável**: o retorno simulado usa a cotação
  comprovada do lado escolhido, nunca a mediana.

A revisão de 09/11 explicita o corte temporal quando os horários divergem.
Cada captura é conferida contra o início programado da própria fonte. A decisão
precisa ser anterior ao menor dos dois horários. A ocorrência da captura de
mercado precisa constar em `chamadas.csv` com o mesmo hash e horário de leitura.
Prova ausente permite a classe parcial. Prova apresentada com inconsistência
ou sem possibilidade de verificação bloqueia a qualificação com o motivo.

Linha antiga, de card montado antes desta emenda, **não ganha vínculo
retroativo**: ela fica sem certificação, e não se procura uma captura posterior
para preencher o buraco.

## 5. Política de seleção, fixada antes dos novos resultados

O v1 apostava em `model_side` mesmo com `w = 0` — então a probabilidade era a do
mercado e a escolha do lado continuava do modelo (E4 da errata).

1. calcular o **lucro esperado líquido** dos **dois** lados: `p_C(lado) ×
   odd_executavel(lado) − 1`;
2. cada lado precisa da **sua própria** prova válida (§3.1), conferida contra o
   que a linha afirma;
3. selecionar **no máximo um** lado por luta: o de **maior lucro esperado
   líquido positivo**;
4. **empate exato** → recusa com motivo, porque desempatar exigiria critério não
   registrado;
5. lado sem preço comprovado fica **indisponível** e **nunca** recebe o preço do
   adversário;
6. **unidade fixa**, uma unidade por perna, **nenhuma aposta real**. Sem Kelly,
   martingale, recuperação de perda ou otimização de stake.

`model_side` **não entra**. O favorito **não** é substituto de aposta de valor.
A política foi escrita antes de comparar qual variante geraria mais lucro.

**Capturas repetidas da mesma luta:** cada `registro_id` liquida uma vez, e cada
luta conta **uma** aposta por candidato — a primeira decisão com seleção. As
demais liquidam com `unidades = 0` e o motivo no campo. Sem isso, três capturas
do mesmo card contariam três apostas na mesma luta.

## 6. Precisão, poder e dimensionamento

Cinco coisas, separadas:

| | o que é |
|---|---|
| **IC observado** | precisão com que esta amostra estimou este contraste |
| **precisão desejada** | largura de IC que se quer alcançar — escolha nossa |
| **efeito mínimo de interesse** | menor diferença que mudaria a decisão |
| **significância** | probabilidade de erro tipo I aceita |
| **poder** | probabilidade de detectar um efeito que exista |

Meia-largura de IC **não** é efeito mínimo detectável. A função foi renomeada de
`detectavel` para `precisao_observada`.

**O desvio de B − A não dimensiona C − A.** Com `w = 0`, C − A é zero por
construção e tem variabilidade zero; o desvio de B − A é de outro contraste e
pressupor que um candidato futuro terá aquela variabilidade é pressupor o que não
se sabe.

### 6.1 Sensibilidade, com os pressupostos à vista

*Se* um candidato futuro tiver desvio por evento `s` do contraste C − A, e *se*
o efeito mínimo de interesse for `δ`, então o número de **eventos
independentes** é da ordem de `(1,96·s/δ)²` para um IC95 de meia-largura `δ`.

Isso é sensibilidade, não plano: `s` é desconhecido para um candidato que ainda
não existe, `δ` não foi escolhido, e a fórmula ignora que os eventos têm números
diferentes de lutas. **Falta informação para dimensionar a amostra**, e o v2
declara isso em vez de entregar um número de eventos ou de meses.

Para referência, não para planejamento, os valores de `s` observados nos 16
eventos do v1 estão em `data/experimentos/por_evento_vantagem-v1.csv`.

### 6.2 Incerteza da calibração

IC de Wilson por faixa assume independência, e lutas do mesmo card não são
independentes. As tabelas declaram essa limitação e marcam a faixa como
**otimista** quando ela contém mais de uma luta do mesmo evento. A alternativa
agrupada (bootstrap de evento por faixa) está implementada e é a que vale quando
houver n suficiente.

### 6.3 Amostra pequena não transforma resultado em ruído

"Compatível com ruído" não é "é ruído". O relatório dá o número, o `n` e a
largura, e diz que a amostra **não decide** — nem a favor nem contra.

## 7. Congelamento e reestimação

O candidato do v1 (`w = 0`) está **preservado e congelado** em
`data/experimentos/candidatos.json`, com os 16 eventos marcados como queimados.
Manter `w = 0` deixa C idêntico a A: **acumular eventos não muda esse
contraste**, e o v2 não finge que muda.

**Regra de reestimação, fixada agora:**

> O peso `w` só pode ser reestimado quando houver **pelo menos 10 eventos**
> registrados em modo sombra que nunca tenham entrado em estimação nem em
> análise inferencial. A reestimação **transforma esses eventos em
> desenvolvimento**: eles passam para `eventos_desenvolvimento` do candidato
> novo e não voltam a servir de avaliação. O candidato novo só é avaliado em
> eventos **posteriores** ao seu congelamento.

**Marcos inferenciais:** a análise inferencial roda a cada **20 eventos de
avaliação** acumulados no candidato em vigor. Fora dos marcos o relatório é
**descritivo** — contagem, cobertura, classificação temporal — sem teste e sem
intervalo apresentado como decisão. Olhar o p-valor a cada evento até ele ficar
favorável é o mesmo erro de afrouxar limiar depois de ver o resultado.

Cada decisão de sombra grava `candidato_id`, `candidato_congelado_em`,
`protocolo_sha256`, `peso_w` e `versao_protocolo`.

## 8. Comandos

```bash
# experimento (recalculo -> EXPLORATORIO depois da errata)
python -m scripts.experimento_vantagem

# congelar um candidato (so quando a regra da secao 7 permitir)
python -m scripts.candidato --peso 0.0 --versao vantagem-v2 \
    --descricao "..." --dev 2026-07-11 ... --consultados 2026-09-05 ...

# inspecionar o candidato em vigor e o que ja esta registrado
python -m scripts.sombra --event-date AAAA-MM-DD --dry-run

# registrar ANTES das lutas
python -m scripts.sombra --event-date AAAA-MM-DD

# liquidar DEPOIS, sem reescrever a previsao
python -m scripts.sombra --event-date AAAA-MM-DD --liquidar
```

## 9. Proteção da produção

Artefatos em `data/experimentos/`. Testes em arquivos temporários. O sha256 de
sete registros de produção é conferido antes e depois de cada execução, e o
comando **aborta** se algum mudar. Nenhum segredo em log, relatório ou commit.
