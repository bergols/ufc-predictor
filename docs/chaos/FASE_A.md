# Fase A: persistência (D1, D2, D9)

Módulo novo: `src/persistencia.py`. Decisões D-1 a D-4 do `PLANO.md`.

## O que faz
- `trava(alvo)`: trava única, reentrante, entre processos (`fcntl.flock`; no
  Windows `msvcrt.locking` no byte 0). Arquivo `.registros.lock`: todos os
  registros de `data/` (inclusive `proveniencia/` e `raw/`) usam o de
  `data/.registros.lock`; fora de `data/` (testes) fica ao lado do alvo. O SO a
  solta se o processo morrer. Espera até 60 s (`UFC_TRAVA_TIMEOUT_S`) e levanta
  `TravaOcupada` dizendo o arquivo de trava e o pid do dono.
- `gravar_atomico`: temporário no mesmo diretório (`.<pid>.<id>.tmp.<nome>`),
  fsync, `os.replace`; no Windows repete por ~2 s se o arquivo estiver aberto.
- `validar_csv` / `ler_csv_integro`: recusam arquivo sem `\n` final ou com linha
  de campos diferente do cabeçalho (aspas respeitadas), com `RegistroIncompleto`
  (caminho, linha, "Nada foi gravado. Restaure com: git checkout -- <arquivo>").
- `anexar_csv`: sob a trava, valida e regrava conteúdo íntegro + linhas novas.

## Onde foi aplicado (ciclo ler→alterar→gravar inteiro sob a trava)
`prediction_history` (record_card_predictions, sync_results_from_template,
record_closing_lines, leitura), `snapshots` (registrar, semear_do_historico,
carregar/_indice), `proveniencia` (chamada, cotação, corpo `.json.gz` atômico e
regravado se não bater com o hash), `mediana`, `decisoes`, `fechamento`
(odds_template), `scripts/ciclo_semanal`, `scripts/reconciliar_card`.
`snapshots.registrar` propaga erro (antes devolvia `[]`), e o histórico só
recebe `prediction_id` que foi gravado e relido.
Proveniência/mediana mantêm "falha não derruba a captura" (log de erro, ref não
verificável), mas nunca anexam sobre linha incompleta.

## Limites (o que NÃO cobre)
- Quem não usa o módulo: editor de texto, planilha, `git checkout`, cópia manual.
- NFS/SMB/pasta sincronizada (OneDrive, Dropbox): `flock` pode não excluir entre
  máquinas; `os.replace` pode não ser atômico.
- Windows: ramo `msvcrt` escrito e testado só com dublê; validação real
  PENDENTE (rodar `tests/test_persistencia.py` na máquina operacional; o teste
  `test_windows_msvcrt_exclui_entre_processos` só roda lá).
- Corte que cai exatamente numa fronteira de linha e deixa campos corretos é
  indistinguível de arquivo menor: não detectado.
- Queda (kill -9) no meio da escrita deixa um temporário órfão
  `.<pid>.<id>.tmp.<nome>` ao lado; inofensivo, pode apagar. O destino fica
  como estava.
- Registro já truncado NÃO é reparado: é recusado. Recuperação:
  `git checkout -- <arquivo>` (ou corrigir a linha apontada) e só então rodar de
  novo. Snapshots órfãos (queda entre snapshot e histórico) seguem possíveis e
  inofensivos (C2d).
- Também sob a trava (ajustes pós-revisão, `8ffa120`): `src/sombra.py`
  (`registrar` e `liquidar`, em `data/experimentos/sombra/`). `anexar_csv`
  recusa cabeçalho divergente das colunas do código antes de gravar.
  Proveniência e mediana tratam `TravaOcupada` como falha de proveniência
  (log de erro, ref não verificável), sem derrubar a captura.
- Validado em Linux (container): reproduções C3a, C3b e C4a–c (C4c com 30
  rodadas, 0 perdas); SIGKILL durante o `os.replace` e no dono da trava;
  reentrância; exclusão entre threads; timeout.
- Fora de escopo: `upcoming_card_odds.csv` (inclusive a gravação direta em
  `src/thunderpick.py`, que é entrada do card, não registro), HTML, caches de
  foto/manifesto.
- Mudou a impressão de código (`src/codigo.py`) e o carimbo do gerador
  (`prediction_history.py` é fonte do gerador): `docs/index.html` precisa ser
  regerada numa publicação (ver `test_report_freshness`).
