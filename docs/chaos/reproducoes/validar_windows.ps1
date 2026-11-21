<#
Validacao Windows da pilha CHAOS (Fases A, B, C e D10) -- roda num CLONE
TEMPORARIO, nunca na arvore operacional.

  powershell -ExecutionPolicy Bypass -File validar_windows.ps1 `
      -Python "C:\caminho\do\ufc-predictor\.venv\Scripts\python.exe" `
      [-DadosDe "C:\caminho\do\ufc-predictor"] [-SuiteCompleta]

Padrao: branch do TOPO da pilha (claude/chaos-d10) no SHA fixado abaixo, que
e o SHA que vai ser integrado. Aprovacao vale so para esse SHA.

O que faz:
 1. clona a branch do GitHub em %TEMP% (credencial do proprio git, nenhuma no
    script), faz checkout do SHA COMPLETO fixado e aborta se o HEAD diferir;
    remove o remoto do clone;
 2. limpa variaveis de credencial do processo;
 3. confere o fuso da maquina (precisa ser Brasilia, -03:00);
 4. roda tests/test_persistencia.py, tests/test_fase_c.py e tests/test_d10.py
    e EXIGE como PASSED (pulado conta como falha):
      - msvcrt real: TestTrava::test_windows_msvcrt_exclui_entre_processos
      - trava entre processos, morte do processo, threads
      - escrita atomica: TestGravarAtomico::*
      - concorrencia entre processos: TestConcorrenciaEntreProcessos::*
      - janela de captura em BRT: TestD5JanelaDeCaptura::*[America/Sao_Paulo]
    e zero FAILED/ERROR nesses arquivos;
 5. com -DadosDe: COPIA (so leitura na origem) os 3 arquivos de base fora do
    git para o clone e roda os 3 testes que dependem deles;
 6. com -SuiteCompleta: roda a suite inteira e mostra o resumo.
O interpretador em -Python so executa; PYTHONDONTWRITEBYTECODE=1. Nao chama
API, nao treina, nao publica, nao toca data/ da arvore operacional.
#>
param(
    [Parameter(Mandatory = $true)][string]$Python,
    [string]$DadosDe = "",
    [switch]$SuiteCompleta,
    [string]$Branch = "claude/chaos-d10",
    # SHA COMPLETO do topo da pilha a integrar (atualizado pelo coordenador).
    [string]$Sha = "__SHA_FINAL__"
)
$ErrorActionPreference = "Stop"
if ($Sha -notmatch '^[0-9a-f]{40}$') { throw "-Sha precisa ser o SHA completo (40 hex); recebido '$Sha'" }

$destino = Join-Path $env:TEMP ("ufc-validacao-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
git clone --quiet --single-branch --branch $Branch https://github.com/bergols/ufc-predictor.git $destino
git -C $destino checkout --quiet --detach $Sha
if ($LASTEXITCODE -ne 0) { throw "commit $Sha nao existe na branch $Branch" }
# PowerShell nao diferencia maiusculas: $shaClone precisa ter nome DIFERENTE de $Sha
$shaClone = (git -C $destino rev-parse HEAD).Trim()
if ($shaClone -ne $Sha) { throw "HEAD do clone ($shaClone) difere do SHA fixado ($Sha): abortando" }
git -C $destino remote remove origin
Write-Host "clone temporario: $destino  (SHA testado: $shaClone)"

foreach ($v in "ODDS_API_KEY", "GH_TOKEN", "GITHUB_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY") {
    Remove-Item "Env:$v" -ErrorAction SilentlyContinue
}
Remove-Item Env:TZ -ErrorAction SilentlyContinue
$env:PYTHONDONTWRITEBYTECODE = "1"
Set-Location $destino
if (Test-Path "data\raw\odds_api_key.txt") { throw "clone contem chave de API: abortando" }

$fuso = [TimeZoneInfo]::Local
Write-Host "fuso da maquina: $($fuso.Id) (offset $($fuso.BaseUtcOffset))"
if ($fuso.BaseUtcOffset -ne [TimeSpan]::FromHours(-3)) {
    throw "a janela em BRT exige a maquina em Brasilia (-03:00); fuso atual $($fuso.Id)"
}

$log = Join-Path $destino "validacao_windows.log"
& $Python -m pytest tests/test_persistencia.py tests/test_fase_c.py tests/test_d10.py `
    -rA -p no:cacheprovider 2>&1 | Tee-Object -FilePath $log
$texto = Get-Content $log

$exigidos = @(
    "tests/test_persistencia.py::TestTrava::test_windows_msvcrt_exclui_entre_processos",
    "tests/test_persistencia.py::TestTrava::test_exclusiva_entre_processos_e_timeout_claro",
    "tests/test_persistencia.py::TestTrava::test_morte_do_processo_libera_a_trava",
    "tests/test_persistencia.py::TestTrava::test_threads_do_mesmo_processo_se_excluem",
    "tests/test_persistencia.py::TestGravarAtomico::",
    "tests/test_persistencia.py::TestConcorrenciaEntreProcessos::",
    "tests/test_fase_c.py::TestD5JanelaDeCaptura::"
)
$falhou = $false
foreach ($e in $exigidos) {
    $passou = $texto | Where-Object { $_ -like "PASSED $e*" }
    if ($e -like "*TestD5JanelaDeCaptura::") {
        $passou = $passou | Where-Object { $_ -like "*America/Sao_Paulo*" }
    }
    if (-not $passou) { Write-Host "FALTOU PASSAR: $e" -ForegroundColor Red; $falhou = $true }
}
$ruins = $texto | Where-Object { $_ -match '^(FAILED|ERROR) ' }
if ($ruins) { $ruins | ForEach-Object { Write-Host $_ -ForegroundColor Red }; $falhou = $true }
$pulados = $texto | Where-Object { $_ -match '^SKIPPED' -and $_ -match 'msvcrt|Sao_Paulo|Atomico|Concorrencia' }
if ($pulados) { $pulados | ForEach-Object { Write-Host "PULADO (conta como falha): $_" -ForegroundColor Red }; $falhou = $true }
if ($falhou) { Write-Host "REPROVADO no Windows ($shaClone). Ver $log" -ForegroundColor Red; exit 1 }
Write-Host "OK: msvcrt real, trava entre processos, escrita atomica, concorrencia e janela em BRT PASSARAM no Windows ($shaClone)." -ForegroundColor Green

if ($DadosDe) {
    $arquivos = "data\raw\fights.csv", "data\raw\fighters.csv", "data\processed\fight_features.csv"
    foreach ($a in $arquivos) {
        $o = Join-Path $DadosDe $a
        if (-not (Test-Path $o)) { throw "dado ausente na origem: $o" }
        New-Item -ItemType Directory -Force -Path (Split-Path (Join-Path $destino $a)) | Out-Null
        Copy-Item $o (Join-Path $destino $a)
    }
    & $Python -m pytest -rA -p no:cacheprovider `
        "tests/test_casamento.py::TestHomonimo::test_a_lista_de_homonimos_confere_com_a_base" `
        "tests/test_identidade.py::TestEventUid::test_linha_sem_event_uid_fica_pendente_e_nao_presumida" `
        "tests/test_train_method.py::TestSymmetricSums::test_somas_identicas_nas_linhas_espelhadas" 2>&1 |
        Tee-Object -FilePath (Join-Path $destino "validacao_dados.log")
    if ($LASTEXITCODE -ne 0) { Write-Host "FALHA nos testes com dados. Ver validacao_dados.log" -ForegroundColor Red; exit 1 }
    Write-Host "OK: os 3 testes que dependem da base passaram com os dados da maquina operacional." -ForegroundColor Green
}
if ($SuiteCompleta) {
    & $Python -m pytest -q -p no:cacheprovider 2>&1 | Tee-Object -FilePath (Join-Path $destino "validacao_suite.log")
}
Write-Host "Envie ao coordenador: SHA testado $shaClone e os arquivos .log de $destino"
Write-Host "Aprovacao so vale para este SHA. Mudou o codigo? Rode de novo com -Sha <novo>."
