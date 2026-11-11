# Emenda ao vantagem-v2: candidato amarrado ao modelo

Esta emenda **não altera** o `PROTOCOLO_VANTAGEM_V2.md`. Hash normalizado
`789d62820ef08d2e…`, que é o do candidato `0ae27ee8b716ab72`, ainda em vigor.
Ela vale para candidatos congelados **depois** dela, que a citam pelo hash.

## Motivo

O v2 amarra o candidato ao protocolo e ao peso `w`, mas não ao modelo cujas
previsões ele combina. Em 2026-11-11 o modelo de produção foi retreinado pelo
procedimento existente: o manifesto passou de `4c292becc442…` (modelo de 06/11)
para `dba2c21a0851…`. Sem esta emenda, uma decisão de sombra combinaria
previsões do modelo novo sob um candidato congelado com o antigo.

## Regra

1. Um candidato registra `manifesto_modelo_sha256`. É o sha256 de
   `models/training_metadata.json`, o mesmo valor que cada versão de
   `data/previsoes_snapshots.csv` guarda em `manifesto_sha256`.
2. Uma decisão só qualifica, em qualquer classe acima de `nao_qualificada`, se
   o `manifesto_sha256` da versão de previsão usada for **igual** ao do
   candidato.
3. Um candidato sem `manifesto_modelo_sha256`, como os quatro congelados até
   2026-11-10, não qualifica nenhuma decisão a partir desta emenda. Esses
   candidatos **não são editados**.
4. O candidato registra também o hash desta emenda (`emendas`) e o commit do
   código com que foi congelado (`codigo_commit`). Para congelar, a árvore de
   `src/` e `scripts/` precisa estar limpa nesse commit. Se o hash da emenda não
   conferir, a decisão é `nao_qualificada`.
5. Retreinar o modelo produz um manifesto novo e exige um candidato novo antes
   da próxima decisão. Previsão de um modelo nunca é atribuída a candidato de
   outro.

## Candidato previsto para o modelo `dba2c21a0851…`

- `w = 0`, **sem reestimação**. Com `w = 0` a probabilidade combinada é a do
  mercado, e a seleção não depende do lado do modelo (travado em
  `tests/test_vantagem.py::TestPolitica`). É uma **referência baseada somente
  no mercado**, **não** um peso validado para o modelo retreinado: nenhum `w`
  foi estimado com previsões desse modelo.
- Os mesmos 16 eventos queimados do `0ae27ee8` (8 de desenvolvimento e 8
  consultados). A regra de reestimação do v2 não muda: pelo menos 10 eventos
  de sombra nunca usados.

## Identidade

Esta emenda não muda as regras de identidade. Uma luta cujo par só uma fonte
conhece (`fonte_card = so_api`), ou cuja equivalência de nome é julgamento
sem fonte, como Chiasson × Mesquita em 2026-11-14 (ver
`IDENTIDADE_2026-11-14.md`), **não** entra em decisão oficial.
