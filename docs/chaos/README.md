# CHAOS — frente de testes adversariais do UFC Predictor

CHAOS é uma **frente de trabalho**, não uma skill instalada. Fluxo:
coordenador → implementação → reprodução CHAOS → revisão → integração
(papéis em `docs/chaos/PLANO.md`, agentes em `.claude/agents/`).

| arquivo | conteúdo |
|---|---|
| `achados.md` | controle central: ID, gravidade, responsável, branch, aceite, estado |
| `PLANO.md` | plano, dependências, decisões e ponto de retomada do coordenador |
| `RELATORIO_RODADA1.md` | relatório da rodada 1 (SHA auditado `2f73748`) |
| `resultado_rodada1.txt` | saída das reproduções em `2f73748` |
| `reproducoes/` | um script por cenário + `rodar.sh` |

## Rodar as reproduções

```bash
docs/chaos/reproducoes/rodar.sh                       # todos
docs/chaos/reproducoes/rodar.sh c4_concorrencia       # um cenário
CHAOS_FONTE=/caminho/da/worktree docs/chaos/reproducoes/rodar.sh   # retestar correção
```

`rodar.sh` copia a fonte **sem `.git`** para um diretório temporário, roda
cada cenário com `env -i` (sem credenciais) e com a rede bloqueada
(`bloqueio_rede.py` instalado como `sitecustomize`). API de odds e modelo são
fixtures (`chaos_common.py`). Os scripts recusam rodar fora dessa cópia.
Cada linha `[DEFEITO]` / `[PASSOU]` diz o esperado e o observado.

Requisitos: Linux/macOS com bash, Python 3.11+ e `requirements.txt`.

## Limitações do ambiente da rodada 1
- Linux, container efêmero. **Nada foi validado no Windows** (máquina das
  tarefas agendadas); concorrência e substituição de arquivo no Windows
  ficam pendentes onde o achado disser.
- Sem modelo treinado nem `data/processed` no clone: o modelo é fixture e 3
  testes da suíte que dependem da base são falhas esperadas.
- Raspagem da Thunderpick, `build_card` e `ciclo_semanal` não exercitados
  (rede, navegador, treino, push).
