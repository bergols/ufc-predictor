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

**Nenhuma fonte afirma "Beatriz Mesquita = Bia Mesquita".** A equivalência é
um julgamento humano apoiado em três fatos verificáveis:

1. O card do UFCStats `402ea3b5ee233852` lista "Macy Chiasson vs Bia Mesquita"
   em 2026-11-14. A ficha é `fighter-details/eff9074bc066f0f2`, que está em
   `data/raw/fighters.csv`.
2. O corpo `40f4e68b69ca0d81` lista "Macy Chiasson × Beatriz Mesquita" na
   mesma data, com a adversária grafada exatamente igual.
3. O sobrenome coincide. Nenhum dos dois corpos da API contém "Bia Mesquita".

Não existe alias versionado para esse par; esta rodada proibia criar alias.

## Como reproduzir o nome canônico sem editar o corpo bruto

O corpo fica como foi recebido. A troca é um mapeamento externo, aplicado
**depois** de ler o corpo, só a este evento da API:

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
| identidade Beatriz ↔ Bia | **não comprovada por fonte**: é julgamento documentado |
| confirmação por duas fontes | **não**: a linha continua `so_api`. O UFCStats tem o confronto, mas o casamento entre as fontes não ocorreu |
| preço executável (Thunderpick) | **ausente**: "confronto ausente na casa" |
| sinal sharp e CLV | **ausentes**: `bloqueio_cotacao` = "confronto ausente na fonte", sem `sharp_ref` |
| início programado na versão | **ausente**: nenhuma prova de preço cobriu a luta, então `pos_inicio` está vazio |
| fechamento | deve casar direto: a linha está como a base escreve ("Bia Mesquita") |
