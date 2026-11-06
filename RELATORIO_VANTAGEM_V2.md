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
