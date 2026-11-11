# Errata do `vantagem-v1`

**Emitida em 2026-11-06**, sobre a execução original do protocolo `vantagem-v1`
nos commits `09f7905` (protocolo), `f63666f` (infraestrutura) e `c21f7b9`
(resultados).

`PROTOCOLO_VANTAGEM.md` e os artefatos de `data/experimentos/` **continuam como
estavam**. Nada foi reescrito, nada foi apagado, e o v1 não é apresentado como
se estas correções estivessem previstas nele. O protocolo corrigido é
`PROTOCOLO_VANTAGEM_V2.md`, com data e commit próprios.

**Consequência de rótulo, e ela vale para tudo o que vem depois:** o contraste
confirmatório do v1 foi executado uma vez, no `c21f7b9`. Resultados
**recalculados** depois desta errata são **desenvolvimento / exploração** e
**não** ganham retrospectivamente o rótulo confirmatório. O código escreve isso
no próprio resumo (`classe_dos_resultados`).

---

## E1. "População temporalmente adequada" era uma afirmação sem prova

**O que o v1 disse.** Que `data/prediction_history.csv` era "o único conjunto
temporalmente adequado", com 179 lutas, porque "cada linha nasceu de um card
antes do evento".

**O erro.** Isso confunde **disponibilidade do dado** com **comprovação
temporal**. Constar de um arquivo cujo propósito é ser pré-registro é indício
sobre o arquivo, não evidência sobre a linha. O v1 não conferiu, por linha, se
havia instante verificável de criação da previsão nem de leitura do preço — e
não havia.

**Medido agora, por linha** (`src/temporalidade.py`):

| item | classe | lutas |
|---|---|---|
| temporalidade da previsão | `desconhecida` | **179 de 179** |
| referência de mercado (mediana) | `nao_certificada` | **179 de 179** |
| preço executável | `ausente` | **179 de 179** |
| **certificadas** (previsão **e** preço comprovados) | | **0** |

Zero. As 179 têm `prediction_id`, mas todas apontam para versões de
**semeadura**, cujo `criado_em` é vazio e cuja temporalidade está declarada
**não recuperável** desde 06/11/2026. Nenhuma tem `sharp_ref` nem `ref_tp`,
porque a proveniência por preço só passou a existir em out/2026 e a coluna só
foi preenchida no card de 07/11, que ainda não tem resultado.

**Correção.** As 179 lutas são **análise histórica exploratória com
temporalidade não certificada**. Não são população prospectiva e não são
avaliação. A população certificada é **zero**, declarada como zero. Nenhum
horário, proveniência ou previsão antiga foi fabricado para preencher a lacuna.

## E2. A auditoria das odds acusou a coisa errada

**O que o v1 disse.** Que em `data/raw/market_odds.csv` "toda linha com horário
foi extraída 103 a 3338 dias **depois** do evento", listado como se fosse prova
de que as odds não servem.

**O erro.** A data de extração é quando **nós** obtivemos o dado, não quando a
casa formou o preço. Uma odd de fechamento legítima, raspada quatro anos depois
de um site que a arquivou, pode ser perfeitamente pré-luta. O atraso da extração
**não** prova que a odd foi formada depois da luta.

**Correção.** O defeito real é **falta de comprovação de origem e de
disponibilidade temporal**, e esse defeito existe de forma independente:

- **nenhuma coluna de casa** — não se sabe de quem é o preço;
- **nenhuma coluna de mercado** — não se sabe de qual mercado;
- **um único par** favorito/azarão por luta, sem os dois lados por casa;
- **um único `odds_extract_ts`** no arquivo inteiro, igual para todas as linhas,
  o que o torna inútil como instante de qualquer coisa específica;
- **nenhum conteúdo bruto persistido** para conferir.

Conclusão inalterada (não há como usar esses preços para simular decisão
antiga), **motivo corrigido**: não é "a odd veio depois da luta", é "não há como
saber de onde a odd veio nem quando ela esteve disponível".

## E3. "Prospectivo" era definido por ausência de contra-evidência

**O que o v1 fez.** `src/sombra.py` classificava como `avaliacao_prospectiva`
todo evento que não estivesse na lista de desenvolvimento.

**O erro.** Estar fora de uma lista não é prova de nada. Um evento pode estar
fora porque ninguém o olhou, porque o candidato ainda não existia quando a
decisão foi tomada, ou porque a decisão foi registrada depois do início da luta.

**Correção** (§4 do v2). Qualificação por **prova positiva**, cada item
verificado e o que falhou registrado: candidato congelado **antes** da decisão;
decisão antes do **início programado daquela luta**; previsão com temporalidade
**comprovada**; preço executável **comprovado** no instante da decisão;
referência de mercado certificada; e o evento nunca usado em estimação nem em
inferência.

Como a mediana não tem proveniência por preço, a classe máxima alcançável hoje é
**`prospectiva_parcial`** — tudo comprovado **menos** a referência de mercado.
Isso **não** é "prospectiva plenamente auditável", e o rótulo diz isso.

**Os 16 eventos do v1 — os 8 de desenvolvimento e os 8 de avaliação — ficam
queimados para sempre.** O v1 só bloqueava os 8 de desenvolvimento; os 8 de
avaliação gastaram a inferência e também não podem voltar como prospectivos.

## E4. A seleção continuava dependendo do modelo com `w = 0`

**O que o v1 fez.** A sombra apostava em `model_side` — o lado que o **modelo**
aponta — e calculava o valor esperado com `p_C`. Com `w = 0`, `p_C` é a
probabilidade do mercado, mas a escolha do lado continuava vindo do modelo.

**O erro.** O procedimento concluiu que o modelo não acrescenta informação e
seguiu usando o modelo para escolher onde apostar. Além disso, apostar no lado
que o modelo aponta é, na prática, apostar no favorito — o que é
indistinguível de não ter critério de valor.

**Correção** (§5 do v2). A política avalia **os dois lados** com `p_C` e a odd
executável de cada um, exige prova própria por lado, escolhe no máximo um (o de
maior lucro esperado líquido positivo) e recusa em empate exato. `model_side`
não entra. Medido: no card de 07/11 a política nova seleciona **Davey Grant**
(odd 3,62), o lado **oposto** ao que o modelo aponta naquela luta.

## E5. Meia-largura de IC não é efeito mínimo detectável

**O que o v1 disse.** "A meia-largura do intervalo é 0,0868: esta amostra só
separaria de zero um efeito maior que ~0,087 de log loss."

**O erro.** Meia-largura de IC observado descreve a **precisão com que esta
amostra estimou este contraste**. Não é efeito mínimo detectável, que é
quantidade de análise de poder e depende de efeito mínimo de interesse, nível de
significância, poder desejado e variabilidade — nenhum deles entrou naquele
cálculo. A função foi renomeada de `detectavel` para `precisao_observada`.

Cinco coisas que o v1 misturava e o v2 separa: **IC observado**, **precisão
desejada**, **efeito mínimo de interesse**, **significância**, **poder**.

## E6. Erro aritmético

**O que o v1 disse.** Que a meia-largura 0,0868 era "maior que a diferença
modelo-mercado inteira medida em amostra grande (0,117)".

**0,0868 é menor que 0,117.** O argumento que o número ia sustentar — "a
precisão desta amostra é grosseira em relação à diferença conhecida" — segue
válido pela ordem de grandeza, mas a frase está errada e fica corrigida.

## E7. O desvio de B−A não dimensiona C−A

**O que o v1 fez.** Usou o desvio por evento da diferença **modelo − mercado**
(0,1344) para montar uma tabela de "quantos eventos faltam", apresentada com
meses.

**O erro.** O contraste de interesse é **C − A**, que com `w = 0` é **zero por
construção** e tem variabilidade **zero**. O desvio de B − A é de outro
contraste. Reaproveitá-lo pressupõe que um candidato futuro, ainda não
formulado, terá a variabilidade do candidato B — o que não se sabe.

**Correção** (§6 do v2). A tabela de meses sai. Fica uma análise de
sensibilidade explícita: *se* um candidato futuro tiver variabilidade por evento
`s` e *se* o efeito mínimo de interesse for `δ`, então o número de eventos
independentes é da ordem de `(1,96·s/δ)²`, com os pressupostos nomeados e a
incerteza de `s` declarada. **Falta informação para dimensionar a amostra**, e o
v2 diz isso em vez de entregar um número definitivo.

## E8. Três conclusões além do que foi medido

1. **"o modelo não acrescenta informação ao mercado"** — medido: *nesta
   combinação* (pooling logístico de um peso), *nesta população* (179 lutas não
   certificadas), *nesta janela*, o peso ótimo foi zero. Isso **não** demonstra
   ausência universal de informação residual: outra formulação (por segmento,
   com interação, não linear) não foi testada.
2. **"o teto são as features"** — é conclusão de `DISCRIMINACAO.md`, obtida por
   outro caminho. `w = 0` é **compatível** com ela e **não a demonstra**.
3. **"um peso verdadeiro de 0,05 produziria ~0,001 de log loss"** — número
   inventado por analogia com a curva de perda do desenvolvimento. A relação
   entre peso verdadeiro e ganho não é essa, e o cálculo não foi feito. Sai.

## E9. Incerteza da calibração ignorava dependência por evento

**O que o v1 fez.** IC de Wilson por faixa de calibração.

**O erro.** Wilson assume observações independentes. As lutas de um mesmo card
compartilham condições, e o próprio protocolo reconhece isso ao reamostrar
**eventos** no bootstrap. Usar Wilson por faixa e bootstrap por evento no mesmo
relatório é tratar a mesma dependência de duas formas.

**Correção.** As tabelas de calibração passam a declarar a limitação no próprio
arquivo, e o v2 acrescenta a opção de incerteza agrupada por evento. O intervalo
de Wilson continua ali como referência **de faixa**, rotulado como
**otimista** quando há mais de uma luta do mesmo evento na faixa.

## E10. "Amostra pequena" não autoriza chamar tudo de ruído

**O que o v1 disse.** Que o AUC de 0,272 do modelo no grupo de 1–2 lutas
prévias (n = 25) é "perfeitamente compatível com ruído".

**O erro.** Compatível com ruído não é o mesmo que **ser** ruído. Com n = 25 o
intervalo é largo e inclui tanto 0,5 quanto valores bem abaixo; declarar a
segunda hipótese descartada é a mesma falácia de ler ausência de significância
como equivalência, de cabeça para baixo.

**Correção.** O v2 reporta o número, o n e a largura do intervalo, e diz que a
amostra **não decide** — nem a favor nem contra.

---

## O que NÃO mudou

- o resultado `w = 0` no desenvolvimento, e a curva de perda crescendo a partir
  dele;
- o contraste do v1 valendo zero por construção;
- a conclusão de que **não há vantagem demonstrada** e de que o baseline fica em
  produção;
- os artefatos de `data/experimentos/` da execução original;
- o fato de que **nenhum P&L foi produzido**, e de que nenhum poderia ter sido.

As correções desta errata tornam o experimento **mais** restritivo, não menos: a
população certificada caiu de "179" para **zero**, e a classe prospectiva
máxima alcançável hoje passou a ser **parcial**.
