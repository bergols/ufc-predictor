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
4. **Código.** O candidato registra `codigo_commit` e `codigo_sha256`, a
   impressão digital do código que decide (implementação em `src/codigo.py`):
   - **Alcance:** todo arquivo `*.py` sob `src/` e `scripts/`, e `config.py`.
     Ficam fora dados, modelos, documentos, testes e pacotes instalados. O
     modelo é conferido pelo manifesto (regra 2), e protocolo e emendas pelo
     hash (regra 6).
   - **Impressão:** sha256 sobre a lista ordenada de `caminho` + sha256 do
     conteúdo, com CRLF normalizado para LF.
   - **No congelamento:** nenhum arquivo do alcance pode estar alterado,
     criado ou removido fora do commit. A impressão do disco tem de ser igual
     à do commit.
   - **Na decisão:** a impressão do disco e a do `codigo_commit` têm de ser
     iguais à registrada. Isso **não** é igualdade de HEAD. Commit posterior
     só de dados, de documentos ou do próprio candidato mantém a impressão e
     não invalida. Código do alcance alterado, localmente ou em commit
     posterior, invalida.
   - **Sem como verificar** (git indisponível, commit inexistente, arquivo
     ilegível, cópia sem repositório): a decisão é `nao_qualificada`, com o
     motivo.
5. Retreinar o modelo produz um manifesto novo e exige um candidato novo antes
   da próxima decisão. Previsão de um modelo nunca é atribuída a candidato de
   outro.
6. **Emenda obrigatória.** Um candidato com `manifesto_modelo_sha256` precisa
   citar esta emenda em `emendas`, com arquivo e sha256.
   - O congelamento é recusado com lista ausente ou vazia, com a emenda
     faltando ou com arquivo inexistente.
   - Na decisão, também é `nao_qualificada` quando a emenda falta na lista,
     uma referência está incompleta, o arquivo não existe ou o hash diverge.
     Cada caso tem motivo próprio.

Os quatro candidatos congelados até 2026-11-10 não são editados e continuam sem
esses campos (regra 3).

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

Uma decisão oficial exige o confronto confirmado pelas duas fontes, por nome
exato ou alias confirmado, **no pré-registro e na captura da decisão**
(`fonte` do card e `fonte_card` da linha = `ambas`).

Chiasson × Mesquita em 2026-11-14 não cumpre isso e **não** entra em decisão
oficial:
- o pré-registro foi feito na `main`, que não tinha o alias Bia ↔ Beatriz, e
  ficou `so_api`;
- o alias existe na branch de pesquisa desde `572df6e`, e a fonte que ele cita
  (página do UFC) sustenta a equivalência ("short for Beatriz");
- isso não muda o registro já feito nem o libera retroativamente.

Detalhes em `IDENTIDADE_2026-11-14.md`.
