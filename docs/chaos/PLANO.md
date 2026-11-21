# Plano da frente CHAOS — coordenador

## Papéis
| papel | quem | faz | não faz |
|---|---|---|---|
| coordenador | sessão principal | plano, dependências, decisões; distribui tarefas com arquivos e aceite; reúne alterações; prepara integração (branches, PRs rascunho); edita `docs/chaos/` | merge; publicação; mexer em registro real |
| implementação | `ufc-implementacao` | corrige um grupo de achados numa worktree própria | modelo/estatística; registros reais; push/PR |
| reprodução | `ufc-chaos` | reproduz e retesta em cópia descartável | alterar implementação |
| revisão de integridade | `ufc-integridade` (só leitura) | defeitos concretos de gravação, concorrência, identidade, snapshots, proveniência, orientação, tempo | melhorias opcionais |
| revisão de testes | `ufc-testes-qualidade` (só leitura) | testes no caminho real, contratos e valores da UI | executar |

Fluxo por fase: coordenador → ufc-implementacao → ufc-chaos (reteste) →
UMA revisão focada (integridade na Fase A) → coordenador prepara PR rascunho.

## Regras de execução
- No máximo 2 subagentes ativos; padrão 1 implementador, depois 1 revisor.
- Um arquivo, um dono por vez: nenhuma fase toca arquivo de outra fase em
  andamento. Fases B e C só começam sobre a Fase A (mesmos arquivos de
  gravação).
- Testes focados durante a implementação; suíte completa uma vez por fase
  estabilizada; repetição só por mudança/falha concreta.
- Fable: reservado para uma revisão final de alto risco, só por decisão
  explícita.
- Agentes em `.claude/agents/` são carregados pelo Claude Code ao INICIAR uma
  sessão. Na sessão em que foram criados (21/11/2026) eles não estavam
  registrados; o coordenador acionou subagentes genéricos (modelo Sonnet)
  com a definição do agente no prompt. Isso está registrado em cada fase.

## Branches e ordem de integração
| ordem | branch | base | conteúdo |
|---|---|---|---|
| 1 | `claude/affectionate-bardeen-2i6676` | `main` (`2f73748`) | artefatos CHAOS, agentes, plano, controle |
| 2 | `claude/chaos-fase-a` | 1 | D1, D2, D9 |
| 3 | `claude/chaos-fase-b` | 2 | D3, D4, D7 |
| 4 | `claude/chaos-fase-c` | 3 | D5, D6, D8 |
PRs em rascunho, empilhados; integrar na ordem acima (merge é do proprietário).

## Decisões
- **D-1 (Fase A): uma trava única, reentrante, entre processos** para todos
  os registros, via trava do sistema operacional num arquivo-irmão
  (`fcntl.flock` / `msvcrt.locking`): é liberada pelo SO se o processo morrer,
  então não há trava órfã para limpar à mão. Uma trava só elimina deadlock
  por ordem de aquisição.
- **D-2:** substituição por temporário no mesmo diretório + `os.replace`
  (também para anexar: copia o conteúdo íntegro + linhas novas). O arquivo
  nunca fica pela metade por causa do nosso escritor.
- **D-3:** arquivo já incompleto é rejeitado com motivo e instrução de
  recuperação (restaurar do git); nenhum escritor grava sobre ele. Não há
  reparo automático de registro real.
- **D-4 (consequência conhecida):** qualquer alteração em `src/`/`scripts/`
  muda a impressão de código (`src/codigo.py`). Candidato congelado antes
  destas correções passa a ser BLOQUEADO na conferência de código, que é o
  comportamento desenhado. Não reescrever candidato antigo.
- **D-5:** D5 é limitação conhecida (captura após meia-noite), não descoberta
  inédita; a rodada 1 só a reconfirmou com dados reais.

## Ponto de retomada
Atualizado ao fim de cada etapa. Última atualização: ver git log deste arquivo.
- [x] artefatos da rodada 1 preservados em `docs/chaos/`
- [x] agentes criados em `.claude/agents/`
- [x] PR #10 (rascunho) da base de coordenação: `claude/affectionate-bardeen-2i6676` → `main`
- [x] Fase A: implementação → reteste CHAOS → revisão integridade → PR #11 (Windows pendente)
  - [x] implementação `286daa1` (subagente genérico Sonnet com a definição de
    `ufc-implementacao` no prompt; os agentes registrados só apareceram depois)
  - [x] reteste `ufc-chaos` (agente registrado): sem defeito da Fase A
  - [x] revisão `ufc-integridade` (agente registrado): 4 ajustes aceitos
  - [x] ajustes pós-revisão `8ffa120`, reconferidos; PR #11 (rascunho, base #10)
  - [ ] **validação Windows** na máquina operacional (bloqueia declarar proteção lá): roteiro `reproducoes/validar_windows.ps1` pronto; nenhum Windows acessível à sessão
  - [~] `docs/index.html`: autorizado regerar SÓ na branch da Fase B, por somente-render, depois do D7
  - [ ] Linux: resultado; Windows: validação pendente até rodar na máquina operacional
  - [ ] hashes de `data/` comparados com os do início da Fase A
- [x] Fase B — D3, D4, D7 (validada no Linux; Windows pendente) (branch `claude/chaos-fase-b`, worktree `wt-fase-b`, base `9ee6765`)
  - [x] diagnóstico somente leitura do histórico real: 0 pares repetidos na mesma data, 0 datas com dois títulos (sem dano do D4 a migrar)
  - [x] implementação (`ufc-implementacao`): `cbb9445`, `fce91aa`, `bb82a86`
  - [x] reteste (`ufc-chaos`): sem defeito novo
  - [x] revisão focada (`ufc-testes-qualidade`): 1 lacuna, corrigida
  - [x] `docs/index.html` por somente-render na branch (`b75cd25`), 0 divergências contra os snapshots
  - [x] PR #12 (rascunho, base #11)
- [ ] Fase C — D5, D6, D8 (próxima; branch a partir de `b75cd25`) — não iniciada
