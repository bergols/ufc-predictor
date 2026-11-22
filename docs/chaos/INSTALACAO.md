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

1. **Verificar** (padrão, não altera nada):
   ```powershell
   powershell -ExecutionPolicy Bypass -File docs\chaos\reproducoes\instalar_operacional.ps1 -Arvore "<árvore operacional>" -Python "<árvore operacional>\.venv\Scripts\python.exe"
   ```
   Rode o roteiro a partir de uma cópia dele (por exemplo, baixada da `main`):
   na `2f73748` da árvore operacional o arquivo ainda não existe.
2. Ler o pré-voo: branch `main`; HEAD ancestral do alvo; alterações locais
   sem colisão; nenhum processo do projeto rodando; próxima execução das
   tarefas; registros aprovados.
3. **Aplicar**: o mesmo comando com `-Aplicar`. Faz backup de `data\` e
   `models\` em `%USERPROFILE%\ufc-backup-instalacao-<data>`, `git merge
   --ff-only`, confere HEAD, sha256 de `data\` igual, preflight de novo e o
   teste de frescor da página.

## Paradas previstas (nada é alterado quando o roteiro para)

- árvore fora da branch `main` (o roteiro não troca branch);
- commit local não enviado (HEAD não é ancestral do alvo) — decidir antes;
- alteração local num arquivo que o alvo muda;
- `data/`, `requirements.txt` ou `.bat` mudando entre HEAD e alvo (não deveria);
- processo do projeto rodando ou tarefa a menos de 10 min de rodar;
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
o roteiro não foi executado num PowerShell real (só revisado). O
`preflight_registros.py` foi testado em Linux contra os registros versionados
(aprovados, sem alterar nada) e contra uma cópia corrompida (reprovada com
motivo).
