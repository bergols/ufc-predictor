# Instalação da pilha CHAOS na máquina operacional

Alvo: `main` em `99d1be50a9c3a36b446210a68e2d8ce58f9d4788` (integração #10–#15).
O código (`src/`, `scripts/`, `config.py`, `tests/`, `requirements.txt`) é
idêntico ao SHA validado no Windows, `4c74a4d3a3ed09fcd07a045686ebc80ad562a6f0`
(validação informada pelo operador: 1069 aprovados, 12 pulados justificados,
código 0).

## O que muda e o que não muda

Entre `2f73748` (último estado de produção antes da frente CHAOS) e o alvo:

| muda | não muda |
|---|---|
| código e testes (D1–D10), `docs/chaos/`, `docs/index.html`, `.claude/agents/`, `.gitignore` | `data/`, `requirements.txt`, `scripts/*.bat` (tarefas agendadas), modelos, candidatos |

Então a instalação é **só avançar o git** (`merge --ff-only`). Sem `pip`, sem
alterar tarefas, sem migrar registro.

## Risco que o pré-voo cobre

Desde `2f73748` a captura horária pode ter gravado registros mais novos que o
git (card de 21/11). O código novo **recusa** registro com linha incompleta ou
cabeçalho divergente. Se um registro local já estivesse assim, a captura
horária passaria a parar depois da instalação. Por isso o roteiro roda
`preflight_registros.py`, com o código novo num clone temporário, sobre os
registros REAIS, só lendo, e para antes de qualquer mudança se algo reprovar.

## Passos

Na máquina operacional, num PowerShell, fora do horário em que as tarefas
rodam:

1. Baixe da `main` uma cópia de `docs/chaos/reproducoes/instalar_operacional.ps1`
   para uma pasta **fora** da árvore operacional (ex.: `%USERPROFILE%\Downloads`).
   Dentro da árvore ela viraria arquivo não rastreado que o alvo sobrescreve; o
   roteiro recusa rodar de dentro da árvore.
2. **Verificar** (padrão; a árvore não é alterada, só `git fetch`):
   ```powershell
   powershell -ExecutionPolicy Bypass -File "%USERPROFILE%\Downloads\instalar_operacional.ps1" -Arvore "<árvore operacional>" -Python "<árvore operacional>\.venv\Scripts\python.exe"
   ```
3. Ler o pré-voo: branch `main`; HEAD ancestral do alvo; alterações locais
   sem colisão; nenhum processo do projeto rodando; próxima execução das
   tarefas; registros aprovados.
4. **Aplicar**: o mesmo comando com `-Aplicar`. Faz backup de `data\` e
   `models\` em `%USERPROFILE%\ufc-backup-instalacao-<data>`, **conferido por
   sha256**; imediatamente antes do merge confere de novo que `data\` não
   mudou, que nenhum processo do projeto roda e que nenhuma tarefa está a menos
   de 10 min; então `git merge --ff-only`, e confere HEAD, `data\` e `models\`,
   preflight de novo e o teste de frescor da página.

## Paradas previstas

Paradas **antes do merge** não alteram a árvore (ficam só a pasta temporária
`%TEMP%\ufc-instalacao-<data>` e, no `-Aplicar`, o backup). Paradas **depois
do merge** (HEAD inesperado, preflight ou teste de frescor reprovado) informam
que o código já foi instalado e dão o comando de reversão. Se `data\` mudar na
janela do merge, é gravação legítima de algum job: o roteiro avisa e deixa o
preflight decidir, sem mandar reverter.

- árvore fora da branch `main` (o roteiro não troca branch);
- commit local não enviado (HEAD não é ancestral do alvo) — decidir antes;
- alteração local num arquivo que o alvo muda;
- `data/`, `requirements.txt` ou `.bat` mudando entre HEAD e alvo (não deveria);
- alteração local ou arquivo não rastreado (conferido arquivo a arquivo) num caminho que o alvo muda;
- processo do projeto rodando ou tarefa a menos de 10 min de rodar (no pré-voo e de novo logo antes do merge);
- backup que não confere com o manifesto sha256;
- registro real reprovado no preflight;
- `data\` mudando durante o pré-voo ou a instalação.

## Reversão

```powershell
git -C "<árvore operacional>" reset --keep <HEAD anterior impresso pelo roteiro>
```
`--keep` volta o código e preserva as alterações locais de `data\`. O backup
de `data\`/`models\` fica para comparação; não é restaurado automaticamente.

## Depois da instalação

- A primeira gravação cria `data\.registros.lock` (ignorado pelo git).
- Candidatos congelados antes destas correções passam a ser bloqueados na
  conferência de código (`src/codigo.py`) — comportamento desenhado; não
  reescrever candidato antigo.
- Limite da Fase A: protege falha durante a escrita protegida e recusa arquivo
  com linha incompleta; não detecta arquivo cortado exatamente em fronteira de
  linha nem protege contra escritor que não use a trava (editor, planilha,
  sincronização de pasta).

## Estado

Preparado, **não executado**: esta sessão não alcança a máquina operacional, e
o roteiro não foi executado num PowerShell real. Revisado por `ufc-integridade`
(somente leitura): 8 defeitos baixos/médios apontados e corrigidos (arquivo não
rastreado não detectado, renomeações e nomes com acento no diff, janela entre
backup e merge, backup não conferido, cabeçalho comparado mais frouxo que o
código, arquivo vazio, falhas sem mensagem clara, afirmações do guia). O
`preflight_registros.py` foi testado em Linux: aprova os registros versionados
sem alterá-los; reprova arquivo truncado e cabeçalho com BOM; aceita arquivo
vazio.
