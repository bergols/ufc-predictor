<#
Validacao Windows da Fase A (CHAOS) -- roda num CLONE TEMPORARIO, nunca na
arvore operacional.

  powershell -ExecutionPolicy Bypass -File validar_windows.ps1 `
      -Python "C:\caminho\do\ufc-predictor\.venv\Scripts\python.exe" `
      [-DadosDe "C:\caminho\do\ufc-predictor"]

O que faz:
 1. clona a branch claude/chaos-fase-a do GitHub em %TEMP% (credencial do
    proprio git, nenhuma no script) e remove o remoto do clone;
 2. limpa variaveis de credencial do processo (ODDS_API_KEY etc.);
 3. roda tests/test_persistencia.py e EXIGE que
    test_windows_msvcrt_exclui_entre_processos tenha PASSADO (skip = falha);
 4. com -DadosDe: COPIA (so leitura na origem) data/raw/fights.csv,
    data/raw/fighters.csv e data/processed/fight_features.csv para o clone e
    roda os 3 testes que dependem desses dados.
O interpretador indicado em -Python so e usado para executar; com
PYTHONDONTWRITEBYTECODE=1 nada e gravado no venv. Nao chama API, nao treina,
nao publica, nao toca data/ da arvore operacional.
#>
param(
    [Parameter(Mandatory = $true)][string]$Python,
    [string]$DadosDe = "",
    [string]$Branch = "claude/chaos-fase-a"
)
$ErrorActionPreference = "Stop"
$destino = Join-Path $env:TEMP ("ufc-validacao-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
git clone --quiet --single-branch --branch $Branch https://github.com/bergols/ufc-predictor.git $destino
git -C $destino remote remove origin
$sha = git -C $destino rev-parse --short HEAD
Write-Host "clone temporario: $destino  (branch $Branch @ $sha)"

foreach ($v in "ODDS_API_KEY", "GH_TOKEN", "GITHUB_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY") {
    Remove-Item "Env:$v" -ErrorAction SilentlyContinue
}
$env:PYTHONDONTWRITEBYTECODE = "1"
Set-Location $destino
if (Test-Path "data\raw\odds_api_key.txt") { throw "clone contem chave de API: abortando" }

$log = Join-Path $destino "validacao_windows.log"
& $Python -m pytest tests/test_persistencia.py -rA -p no:cacheprovider 2>&1 | Tee-Object -FilePath $log
$alvo = "test_persistencia.py::TestTrava::test_windows_msvcrt_exclui_entre_processos"
$ok = Select-String -Path $log -SimpleMatch "PASSED tests/$alvo" -Quiet
if (-not $ok) { Write-Host "FALHA: o teste msvcrt NAO passou (ou foi pulado). Ver $log" -ForegroundColor Red; exit 1 }
Write-Host "OK: $alvo executou e PASSOU no Windows ($sha)." -ForegroundColor Green

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
Write-Host "Envie ao coordenador: o SHA acima e os arquivos .log de $destino"
