# Identidade: Chiasson × Mesquita (UFC 334, 2026-11-14)

Pré-registro **operacional** de 14/11 (`9dc7402`, `771248d`). Nada aqui vale
como avaliação prospectiva do protocolo vantagem-v2.

## O que aconteceu

| UTC | evento |
|---|---|
| 16:12:23 | `build_card` lê a The Odds API (chamada `40f4e68b69ca0d81`). O evento `aab7065b18217457721e43f992ea4905` traz **"Macy Chiasson × Beatriz Mesquita"**, com 19 casas e início em 2026-11-14T23:15Z. |
| 16:12:4x | O `unir` não casa "Beatriz" com o "**Bia Mesquita**" do card do UFCStats (evento `402ea3b5ee233852`). A luta sai duas vezes: `so_ufcstats` "Bia Mesquita", fora do CSV por falta de mediana, e `so_api` "Beatriz Mesquita", dentro do CSV. |
| ~16:13 | Uma única célula do CSV é trocada à mão: `Beatriz Mesquita` → `Bia Mesquita`. Ficam iguais a orientação (Chiasson = A), a mediana 3,10 / 1,36 e a marca `so_api`. Foi decisão do dono do projeto, tomada antes de qualquer previsão. |
| 16:15:33 | `publish_report` grava a previsão (`prediction_id` `500d4beee5a5ae4c`, `criado_em` 16:15:33Z, manifesto `dba2c21a0851`). Ela já usa o histórico de Bia Mesquita, que tem lutas na base. Com o nome da API, o modelo a trataria como estreante. |

A **correção veio antes da previsão**. Nenhuma versão foi gravada com
"Beatriz", e nenhuma previsão foi substituída.

## Fonte da equivalência

Há quatro coisas distintas aqui. Uma versão anterior desta nota as confundia
e dizia que não havia fonte nem alias; as duas afirmações estavam erradas.

**1. Alias na branch de pesquisa.** `src/line_shopping.py::APELIDOS_API` tem
`"Bia Mesquita": "Beatriz Mesquita"`. Ele entrou no commit `572df6e`
(2026-11-11 14:59Z, branch `pesquisa-vantagem`) e cita
`https://www.ufc.com/athlete/bia-mesquita`. A tabela da casa,
`thunderpick.APELIDOS_TP`, não tem entrada para Mesquita.

**2. O que a fonte mostra.** Lida em 2026-11-11, a URL redireciona (301) para
`https://www.ufc.com.br/athlete/bia-mesquita`, com título "Bia Mesquita | UFC".
Na seção de perguntas e respostas a própria atleta diz: *"Most people call me
'Bia,' short for Beatriz"*. A página **não** traz "Beatriz Mesquita" como nome
completo e não tem campo de nome civil. A prova, portanto, é esta: a atleta
"Bia Mesquita" do UFC se chama Beatriz. Ela sustenta o alias. A página não
foi arquivada no repositório, só a citação acima.

**3. A main não tinha o alias.** O registro operacional rodou na `main`
(`771248d`). Ali `APELIDOS_API` não contém Mesquita. O `unir` da `main` casa
as fontes por `match_event` (tabela de apelidos e correspondência aproximada),
e ele não juntou "Bia" com "Beatriz". A troca manual de ~16:13Z foi
decisão do dono do projeto com base em três fatos:
   1. o card do UFCStats `402ea3b5ee233852` lista "Macy Chiasson vs Bia
      Mesquita" (ficha `fighter-details/eff9074bc066f0f2`);
   2. o corpo `40f4e68b69ca0d81` lista "Macy Chiasson × Beatriz Mesquita" na
      mesma data, com a adversária idêntica;
   3. o sobrenome coincide.

   A fonte do item 2 não foi consultada naquele momento.

**4. O registro operacional continua `so_api`.** A linha de 14/11 e a sua
versão (`500d4beee5a5ae4c`) guardam `fonte_card = so_api` e não são editadas.
Sem casamento entre as fontes no registro, a confirmação por duas fontes **não**
ocorreu ali, e a fonte encontrada depois não libera a luta retroativamente
para decisão oficial.

## Como reproduzir o nome canônico sem editar o corpo bruto

O corpo fica como foi recebido. A troca é um mapeamento aplicado **depois** de
ler o corpo, só a este evento da API. Na branch de pesquisa, o mesmo mapeamento
sai de `APELIDOS_API`.

```python
import gzip, json
from src.proveniencia import conferir
from scripts.build_card import _medianas

assert conferir("40f4e68b69ca0d81") == (True, "ok")          # hash do corpo
corpo = json.load(gzip.open("data/proveniencia/40f4e68b69ca0d81.json.gz",
                            "rt", encoding="utf-8"))
ev = next(e for e in corpo if e["id"] == "aab7065b18217457721e43f992ea4905")
assert {ev["home_team"], ev["away_team"]} == {"Macy Chiasson", "Beatriz Mesquita"}
print(_medianas(ev, "Macy Chiasson", "Beatriz Mesquita"))    # (3.1, 1.36, 19)

CANONICO = {("aab7065b18217457721e43f992ea4905", "Beatriz Mesquita"): "Bia Mesquita"}
```

O CSV que o `build_card` gerou antes da troca não foi versionado. Ele é
reproduzível com `unir(...)` sobre esse corpo e a lista do UFCStats.

## O que ficou comprovado e o que não

| item | situação |
|---|---|
| corpo da chamada 1 (hash, instante, evento, 19 casas) | **comprovado** |
| mediana 3,10 / 1,36 a partir do corpo | **reproduzível**, mas sem referência por luta na main (`ref_mediana` não existia no registro) |
| previsão: versão imutável, `criado_em`, manifesto | **comprovado** |
| identidade Beatriz ↔ Bia, no momento da correção | **julgamento**: a `main` não tinha alias, e a fonte não foi consultada |
| identidade Beatriz ↔ Bia, depois | **sustentada pela fonte** ("short for Beatriz", ufc.com, lida em 2026-11-11); não vale retroativamente para o registro |
| confirmação por duas fontes no registro | **não**: a linha continua `so_api` e assim fica |
| preço executável (Thunderpick) | **ausente**: "confronto ausente na casa" |
| sinal sharp e CLV | **ausentes**: `bloqueio_cotacao` = "confronto ausente na fonte", sem `sharp_ref` |
| início programado na versão | **ausente**: nenhuma prova de preço cobriu a luta, então `pos_inicio` está vazio |
| fechamento | deve casar direto: a linha está como a base escreve ("Bia Mesquita") |
