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
 6. com -SuiteCompleta: roda a suite inteira; codigo != 0 REPROVA o roteiro.
 7. confere que data/ da arvore operacional ficou com sha256 igual.
Toda execucao do pytest tem o codigo de saida conferido; skip nunca aprova
um teste obrigatorio.
O interpretador em -Python so executa; PYTHONDONTWRITEBYTECODE=1. Nao chama
API, nao treina, nao publica, nao toca data/ da arvore operacional.
#>
param(
    [Parameter(Mandatory = $true)][string]$Python,
    [string]$DadosDe = "",
    [switch]$SuiteCompleta,
    [string]$Branch = "claude/chaos-d10",
    # SHA COMPLETO do topo da pilha a integrar (atualizado pelo coordenador).
    [string]$Sha = "4c74a4d3a3ed09fcd07a045686ebc80ad562a6f0"
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

# Roda o pytest sem que texto no stderr derrube o roteiro (no Windows
# PowerShell 5.1, 2>&1 com ErrorActionPreference=Stop transforma cada linha de
# stderr em erro terminal) e DEVOLVE o codigo de saida do pytest.
function Invoke-Pytest([string]$LogPath, [string[]]$Argumentos) {
    $anterior = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python -m pytest @Argumentos 2>&1 | ForEach-Object { "$_" } |
            Tee-Object -FilePath $LogPath | Out-Host
        $rc = $LASTEXITCODE
    } finally { $ErrorActionPreference = $anterior }
    return $rc
}
# Le o status de CADA teste na saida -v ("<id> PASSED|SKIPPED|FAILED|ERROR").
# Para cada id/prefixo obrigatorio: precisa existir >= 1 teste e TODOS PASSED.
# Skip nunca aprova. Excecao unica: com -SoBrt, os parametros [UTC] da janela
# do D5 sao ignorados (no Windows nao ha tzset; UTC e coberto no Linux) e os
# [America/Sao_Paulo] sao obrigatorios.
function Test-Exigidos([string]$LogPath, [string[]]$Ids, [switch]$SoBrt) {
    $status = @{}
    foreach ($l in Get-Content $LogPath) {
        if ($l -match '^(tests/\S+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)') { $status[$Matches[1]] = $Matches[2] }
    }
    $ok = $true
    foreach ($e in $Ids) {
        $alvo = @($status.Keys | Where-Object { $_.StartsWith($e) })
        if ($SoBrt -and $e -like "*TestD5JanelaDeCaptura::") {
            $alvo = @($alvo | Where-Object { $_ -notmatch '\[UTC\]$' })
            if (-not ($alvo | Where-Object { $_ -like "*America/Sao_Paulo*" })) { Write-Host "SEM VARIANTE BRT EXECUTADA: $e" -ForegroundColor Red; $ok = $false }
        }
        if ($alvo.Count -eq 0) { Write-Host "NAO EXECUTADO: $e" -ForegroundColor Red; $ok = $false; continue }
        foreach ($t in $alvo) {
            if ($status[$t] -ne "PASSED") { Write-Host "$($status[$t]) (reprova): $t" -ForegroundColor Red; $ok = $false }
        }
    }
    foreach ($t in $status.Keys) {
        if ($status[$t] -in @("FAILED", "ERROR")) { Write-Host "$($status[$t]): $t" -ForegroundColor Red; $ok = $false }
    }
    return $ok
}

# --- origem dos dados: so leitura, hash antes/depois ----------------------
$arquivosBase = "data\raw\fights.csv", "data\raw\fighters.csv", "data\processed\fight_features.csv"
$hashOrigemAntes = @{}
if ($DadosDe) {
    Write-Host "arvore operacional: branch $(git -C $DadosDe rev-parse --abbrev-ref HEAD) @ $(git -C $DadosDe rev-parse HEAD) (so leitura)"
    foreach ($a in $arquivosBase) {
        $o = Join-Path $DadosDe $a
        if (-not (Test-Path $o)) { throw "dado ausente na origem: $o" }
        $hashOrigemAntes[$a] = (Get-FileHash $o -Algorithm SHA256).Hash
    }
    Get-ChildItem (Join-Path $DadosDe "data") -Recurse -File | Get-FileHash -Algorithm SHA256 |
        ForEach-Object { "$($_.Hash)  $($_.Path)" } | Set-Content (Join-Path $destino "origem_data_antes.sha256")
}

# --- 1. testes obrigatorios: msvcrt real, trava, atomico, concorrencia, BRT --
$log = Join-Path $destino "validacao_windows.log"
$rc = Invoke-Pytest $log @("tests/test_persistencia.py", "tests/test_fase_c.py", "tests/test_d10.py", "-v", "-rA", "-p", "no:cacheprovider")
$exigidos = @(
    "tests/test_persistencia.py::TestTrava::test_windows_msvcrt_exclui_entre_processos",
    "tests/test_persistencia.py::TestTrava::test_exclusiva_entre_processos_e_timeout_claro",
    "tests/test_persistencia.py::TestTrava::test_morte_do_processo_libera_a_trava",
    "tests/test_persistencia.py::TestTrava::test_threads_do_mesmo_processo_se_excluem",
    "tests/test_persistencia.py::TestGravarAtomico::",
    "tests/test_persistencia.py::TestConcorrenciaEntreProcessos::",
    "tests/test_fase_c.py::TestD5JanelaDeCaptura::"
)
$ok = Test-Exigidos $log $exigidos -SoBrt
if ($rc -ne 0) { Write-Host "pytest dos testes obrigatorios saiu com codigo $rc" -ForegroundColor Red; $ok = $false }
if (-not $ok) { Write-Host "REPROVADO no Windows ($shaClone). Ver $log" -ForegroundColor Red; exit 1 }
Write-Host "OK: msvcrt real, trava entre processos, escrita atomica, concorrencia e janela em BRT PASSARAM no Windows ($shaClone)." -ForegroundColor Green

# --- 2. os 3 testes que dependem da base (dados COPIADOS para o clone) -------
if ($DadosDe) {
    foreach ($a in $arquivosBase) {
        New-Item -ItemType Directory -Force -Path (Split-Path (Join-Path $destino $a)) | Out-Null
        Copy-Item (Join-Path $DadosDe $a) (Join-Path $destino $a)
    }
    $idsDados = @(
        "tests/test_casamento.py::TestHomonimo::test_a_lista_de_homonimos_confere_com_a_base",
        "tests/test_identidade.py::TestEventUid::test_linha_sem_event_uid_fica_pendente_e_nao_presumida",
        "tests/test_train_method.py::TestSymmetricSums::test_somas_identicas_nas_linhas_espelhadas"
    )
    $logDados = Join-Path $destino "validacao_dados.log"
    $rc = Invoke-Pytest $logDados (@("-v", "-rA", "-p", "no:cacheprovider") + $idsDados)
    $ok = Test-Exigidos $logDados $idsDados
    if ($rc -ne 0) { Write-Host "pytest dos testes com dados saiu com codigo $rc" -ForegroundColor Red; $ok = $false }
    if (-not $ok) { Write-Host "REPROVADO: testes com dados. Ver $logDados" -ForegroundColor Red; exit 1 }
    Write-Host "OK: os 3 testes que dependem da base PASSARAM com os dados da maquina operacional." -ForegroundColor Green
}

# --- 3. suite completa ------------------------------------------------------
if ($SuiteCompleta) {
    $logSuite = Join-Path $destino "validacao_suite.log"
    $rc = Invoke-Pytest $logSuite @("-v", "-rA", "-p", "no:cacheprovider")
    Get-Content $logSuite | Select-Object -Last 3 | ForEach-Object { Write-Host $_ }
    if ($rc -ne 0) { Write-Host "REPROVADO: suite completa saiu com codigo $rc. Ver $logSuite" -ForegroundColor Red; exit 1 }
    Write-Host "OK: suite completa aprovada (codigo 0)." -ForegroundColor Green
}

# --- preservacao da origem ----------------------------------------------------
if ($DadosDe) {
    foreach ($a in $arquivosBase) {
        $depois = (Get-FileHash (Join-Path $DadosDe $a) -Algorithm SHA256).Hash
        if ($depois -ne $hashOrigemAntes[$a]) { Write-Host "ORIGEM ALTERADA: $a" -ForegroundColor Red; exit 1 }
    }
    Get-ChildItem (Join-Path $DadosDe "data") -Recurse -File | Get-FileHash -Algorithm SHA256 |
        ForEach-Object { "$($_.Hash)  $($_.Path)" } | Set-Content (Join-Path $destino "origem_data_depois.sha256")
    $dif = Compare-Object (Get-Content (Join-Path $destino "origem_data_antes.sha256")) (Get-Content (Join-Path $destino "origem_data_depois.sha256"))
    if ($dif) { Write-Host "ORIGEM ALTERADA durante a validacao:" -ForegroundColor Red; $dif | Out-Host; exit 1 }
    Write-Host "OK: data/ da arvore operacional com sha256 igual antes/depois; branch $(git -C $DadosDe rev-parse --abbrev-ref HEAD) @ $(git -C $DadosDe rev-parse HEAD)." -ForegroundColor Green
}
Write-Host "Windows: $((Get-CimInstance Win32_OperatingSystem).Caption) $((Get-CimInstance Win32_OperatingSystem).Version) | Python: $(& $Python --version 2>&1)"
Write-Host "Envie ao coordenador: SHA testado $shaClone e os arquivos .log de $destino"
Write-Host "Aprovacao so vale para este SHA. Mudou o codigo? Rode de novo com -Sha <novo>."
