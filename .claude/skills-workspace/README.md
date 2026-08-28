# skills-workspace

Ferramental para **medir** as skills do projeto (`.claude/skills/ufc-evento`,
`.claude/skills/ufc-analise`). Nada aqui é carregado em tempo de execução: é
material de manutenção, versionado para que a medição seja reproduzível.

## Conjuntos de avaliação

`trigger-evento.json` e `trigger-analise.json` — 16 consultas cada (8 que devem
disparar, 8 que não devem), escritas em português informal, do jeito que as
perguntas realmente chegam.

Os negativos de uma são deliberadamente os casos de uso da outra. As duas skills
competem pelo mesmo projeto, e o erro caro não é deixar de disparar: é a
`ufc-analise` abrir num "fecha o card" e passar régua de modelagem num trabalho
de operação (ou o contrário). É isso que esses conjuntos medem.

## Resultado da última medição (27/08/2026, claude-opus-5)

| skill | treino | holdout |
|---|---|---|
| `ufc-evento` | 10/10 | 5/6 |
| `ufc-analise` | 10/10 | 6/6 |

As duas pararam em `all_passed (iteration 1)` — o otimizador não reescreveu
nenhuma descrição porque não havia o que melhorar.

A única falha, em `ufc-evento`: *"o upcoming_card_odds.csv ta com o nome errado
do lutador, o fuzzy casou errado"*, 0/3. Provavelmente o gabarito é que está
errado — o `best_name_match` é código do modelo (`src/utils.py`), não etapa do
ciclo semanal, e não disparar ali é defensável. Deixado como está para não
ajustar o teste até ele passar.

## Como rodar

Precisa do `claude` CLI (`npm i -g @anthropic-ai/claude-code`) e de estar logado.
Rode **de dentro do repositório**:

```bash
python C:/Users/bergj/.claude/skills/skill-creator/otimizar.py \
    --eval-set .claude/skills-workspace/trigger-evento.json \
    --skill-path .claude/skills/ufc-evento \
    --model claude-opus-5 --timeout 60 --verbose
```

O `otimizar.py` (cópia em `skill-creator-patches/`) existe porque este projeto
também tem um pacote `scripts/`: rodando `python -m scripts.run_loop` de dentro
do repo, o `scripts/` do projeto vence no `sys.path` e o import quebra; rodando
de fora, o `find_project_root()` acha a pasta do usuário em vez do repositório.
O lançador resolve os dois e imprime o projeto detectado antes de começar.

**Rode uma skill de cada vez.** A medição move `.claude/skills/<skill>` para
`.claude/.skills-ocultas-eval/` e restaura no fim; duas execuções simultâneas
embaralham esse estado.

## Patches em `skill-creator-patches/`

O `skill-creator` oficial mora fora do repositório
(`~/.claude/skills/skill-creator/`) e tinha três defeitos que tornavam a medição
**silenciosamente inútil** — o placar saía `0/3` em toda consulta, positiva ou
negativa, o que parece um empate de 50% em vez de um erro:

1. **Choque de nomes.** O harness mede a descrição candidata criando uma cópia
   temporária com nome único (`ufc-evento-skill-a1b2c3d4`) e vendo se o Claude a
   invoca. Como a skill **real** está instalada no projeto, o Claude invocava a
   real — `Skill{"skill": "ufc-evento"}` — e a detecção, procurando o hash, lia
   isso como "não disparou". Pior: se aceitasse, estaria medindo a descrição
   atual, não a candidata. Agora a skill real sai do caminho durante o eval.

2. **Corrida entre workers.** Os 10 workers criam cada um o seu
   `<skill>-skill-<hash>` no mesmo `.claude/commands/`, todos visíveis ao mesmo
   tempo — o Claude do worker A podia invocar a cópia do worker B, e A contava
   falso negativo. As cópias são idênticas, então a detecção passou a aceitar
   qualquer uma.

3. **Erro invisível.** O `claude` escreve várias falhas (ex.: "Not logged in") no
   **stdout**, e o script só reportava stderr — o relatório saía vazio e o erro
   virava adivinhação.

Mais `read_text()` sem `encoding` em cinco scripts: em cp1252 os acentos do
`SKILL.md` viram lixo silencioso dentro da descrição avaliada.

### Reaplicar

Uma atualização do skill-creator apaga tudo isso sem aviso, e o sintoma é só o
placar voltar a `0/3`. Para reaplicar:

```bash
cd ~/.claude/skills/skill-creator
cp -r scripts scripts.bak-antes-do-patch
patch -p1 --binary < /caminho/para/.claude/skills-workspace/skill-creator-patches/skill-creator.diff
cp /caminho/para/.claude/skills-workspace/skill-creator-patches/otimizar.py .
```

O diff foi verificado: aplicado sobre a versão original, reproduz os seis
arquivos byte a byte. Se ele falhar, o upstream mudou — as três explicações
acima têm detalhe suficiente para refazer à mão. Os trechos alterados estão
marcados com `# [patch local]` no código.

Sinal de que os patches sumiram: o eval devolve `0/3` **em todas** as consultas,
com `precision=100% recall=0%`.
