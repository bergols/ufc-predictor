r"""
Lancador do otimizador de descricao, para projetos que TAMBEM tem um pacote
`scripts/` -- caso do ufc_predictor.

O problema, em duas metades que se contradizem:

  1. run_eval.find_project_root() sobe a partir do diretorio ATUAL procurando
     um `.claude/`. Rodando de dentro do skill-creator ele acha a pasta do
     usuario como "projeto", e testa a descricao num contexto onde as skills do
     repositorio nao existem -- devolvendo 0/3 em tudo.
  2. Rodar de dentro do repositorio corrige isso, mas ai
     `python -m scripts.run_loop` importa o `scripts/` DO PROJETO (o diretorio
     atual vence no sys.path), que nao tem run_loop.

A saida: este arquivo mora no skill-creator, entao sys.path[0] aponta para ca e
o import certo ganha; e ele e chamado com o repositorio como diretorio atual,
entao find_project_root acha o repositorio. Os dois requisitos, sem conflito.

Uso -- de dentro do repositorio:

    cd E:\projetos\ufc_predictor
    $env:PYTHONUTF8=1
    python C:\Users\bergj\.claude\skills\skill-creator\otimizar.py ^
        --eval-set .claude\skills-workspace\trigger-evento.json ^
        --skill-path .claude\skills\ufc-evento ^
        --model claude-opus-5 --verbose

Ele imprime o projeto detectado e as skills visiveis antes de comecar: se
aparecer a pasta do usuario em vez do repositorio, pare -- o resultado seria
0/3 e so gastaria cota.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.run_eval import find_project_root  # noqa: E402
from scripts.run_loop import main  # noqa: E402

if __name__ == "__main__":
    raiz = find_project_root()
    skills = raiz / ".claude" / "skills"
    print(f"[otimizar] projeto detectado: {raiz}", file=sys.stderr)
    if skills.is_dir():
        print(f"[otimizar] skills visiveis: {[p.name for p in skills.iterdir()]}",
              file=sys.stderr)
    else:
        print("[otimizar] AVISO: nao ha .claude/skills aqui. Voce esta rodando "
              "do diretorio do repositorio?", file=sys.stderr)
    sys.exit(main())
