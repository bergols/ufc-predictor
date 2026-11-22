<#
Instalacao da pilha CHAOS na MAQUINA OPERACIONAL (Windows).

  # 1) so verificar (padrao): nada e alterado
  powershell -ExecutionPolicy Bypass -File instalar_operacional.ps1 `
      -Arvore "C:\caminho\do\ufc-predictor" `
      -Python "C:\caminho\do\ufc-predictor\.venv\Scripts\python.exe"

  # 2) aplicar, depois de ler o pre-voo aprovado
  powershell -ExecutionPolicy Bypass -File instalar_operacional.ps1 -Arvore ... -Python ... -Aplicar

O que muda entre 2f73748 e o alvo: so codigo, testes e documentacao. NAO
mudam: data/, requirements.txt, os .bat das tarefas agendadas. Por isso a
instalacao e apenas avancar o git (merge --ff-only), sem pip e sem mexer nas
tarefas.

Pre-voo (sempre; so leitura, exceto `git fetch`, que atualiza referencias):
  - arvore e repositorio git na branch main (o roteiro NAO troca branch);
  - HEAD atual e ancestral do alvo (senao ha commit local nao enviado: PARA);
  - alteracoes locais nao colidem com arquivos que o alvo muda (senao PARA);
  - nenhum processo do projeto rodando (auto_capture, ciclo, publicacao);
  - tarefas agendadas do projeto: estado e proxima execucao (nao altera;
    PARA se faltar menos de 10 min para alguma rodar);
  - clone temporario do alvo (git clone local, so leitura na origem) e
    preflight_registros.py sobre os registros REAIS: se algum registro nao
    passar nas regras do codigo novo, PARA (senao a captura horaria pararia).
Aplicar (-Aplicar):
  - backup de data\ e models\ fora do repositorio, com manifesto sha256;
  - git merge --ff-only <alvo> (sem merge commit, sem reset, sem stash);
  - confere HEAD = alvo, data\ com sha256 igual ao de antes e as mesmas
    alteracoes locais; roda o preflight de novo e o teste de frescor da pagina;
  - imprime o comando de reversao.
Nao chama API, nao treina, nao publica, nao altera tarefas, modelos,
candidatos nem registros.
#>
param(
    [Parameter(Mandatory = $true)][string]$Arvore,
    [Parameter(Mandatory = $true)][string]$Python,
    # main apos a integracao #10-#15 (codigo = 4c74a4d, validado no Windows)
    [string]$Alvo = "99d1be50a9c3a36b446210a68e2d8ce58f9d4788",
    [switch]$Aplicar
)
$ErrorActionPreference = "Stop"
$carimbo = Get-Date -Format "yyyyMMdd-HHmmss"
$trabalho = Join-Path $env:TEMP "ufc-instalacao-$carimbo"
New-Item -ItemType Directory -Force -Path $trabalho | Out-Null
$logGeral = Join-Path $trabalho "instalacao.log"
Start-Transcript -Path $logGeral -Append | Out-Null

function Parar([string]$msg) { Write-Host "PARADO: $msg" -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1 }
# git sem que stderr (progresso/avisos) vire erro terminal no PS 5.1
# Nome nao pode ser 'Git': o PowerShell ignora caixa e a funcao sombrearia o
# executavel git (recursao infinita).
function Invoke-Git([string[]]$a) {
    $ant = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $out = & git -C $Arvore @a 2>&1 | ForEach-Object { "$_" }; $rc = $LASTEXITCODE }
    finally { $ErrorActionPreference = $ant }
    return [pscustomobject]@{ rc = $rc; out = $out }
}
function Py([string]$cwd, [string[]]$a) {
    $ant = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    Push-Location $cwd
    try { & $Python @a 2>&1 | ForEach-Object { "$_" } | Out-Host; $rc = $LASTEXITCODE }
    finally { Pop-Location; $ErrorActionPreference = $ant }
    return $rc
}
function Manifesto([string]$pasta, [string]$saida) {
    Get-ChildItem $pasta -Recurse -File | Where-Object { $_.Name -ne ".registros.lock" } |
        Sort-Object FullName | Get-FileHash -Algorithm SHA256 |
        ForEach-Object { "$($_.Hash)  $($_.Path.Substring($pasta.Length))" } | Set-Content $saida
}

if ($Alvo -notmatch '^[0-9a-f]{40}$') { Parar "-Alvo precisa ser SHA completo" }
$Arvore = (Resolve-Path $Arvore).Path
if (-not (Test-Path (Join-Path $Arvore ".git"))) { Parar "$Arvore nao e repositorio git" }
foreach ($v in "ODDS_API_KEY", "GH_TOKEN", "GITHUB_TOKEN") { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
$env:PYTHONDONTWRITEBYTECODE = "1"
Write-Host "=== PRE-VOO ($carimbo) -- arvore $Arvore; alvo $Alvo"
Write-Host "Windows: $((Get-CimInstance Win32_OperatingSystem).Caption) | Python: $(& $Python --version 2>&1)"

# 1. branch e HEAD
$branch = (Invoke-Git @("rev-parse", "--abbrev-ref", "HEAD")).out | Select-Object -Last 1
if ($branch -ne "main") { Parar "a arvore esta na branch '$branch'; o roteiro nao troca branch" }
$antigo = ((Invoke-Git @("rev-parse", "HEAD")).out | Select-Object -Last 1).Trim()
Write-Host "HEAD atual: $antigo (main)"
if ($antigo -eq $Alvo) { Write-Host "Ja esta no alvo. Nada a instalar." -ForegroundColor Green; Stop-Transcript | Out-Null; exit 0 }

# 2. alvo disponivel e HEAD ancestral dele
$f = Invoke-Git @("fetch", "--quiet", "origin", "main")
if ($f.rc -ne 0) { Parar "git fetch falhou: $($f.out -join ' ')" }
if ((Invoke-Git @("cat-file", "-e", "$Alvo^{commit}")).rc -ne 0) { Parar "alvo $Alvo nao existe apos o fetch" }
if ((Invoke-Git @("merge-base", "--is-ancestor", $antigo, $Alvo)).rc -ne 0) {
    $loc = (Invoke-Git @("log", "--oneline", "$Alvo..$antigo")).out
    Parar "HEAD tem commit(s) que nao estao no alvo (nao enviados?):`n$($loc -join "`n")`nResolva (envie ou decida) antes de instalar."
}

# 3. alteracoes locais x arquivos que o alvo muda
$mudam = (Invoke-Git @("diff", "--name-only", $antigo, $Alvo)).out
$locais = (Invoke-Git @("status", "--porcelain")).out
Write-Host "Alteracoes locais ($(@($locais).Count)):"; $locais | ForEach-Object { Write-Host "  $_" }
$colide = @()
foreach ($l in $locais) {
    $arq = $l.Substring(3).Trim('"') -replace '\\', '/'
    if ($mudam -contains $arq) { $colide += $arq }
}
if ($colide) { Parar "alteracoes locais em arquivos que o alvo muda (o ff falharia ou perderia trabalho): $($colide -join ', ')" }
if ($mudam | Where-Object { $_ -like "data/*" }) { Parar "o alvo altera data/ -- inesperado: $(($mudam | Where-Object { $_ -like 'data/*' }) -join ', ')" }
foreach ($sens in "requirements.txt") { if ($mudam -contains $sens) { Parar "$sens muda entre $antigo e o alvo: exige instalar dependencias, fora deste roteiro" } }
$bats = $mudam | Where-Object { $_ -like "*.bat" }
if ($bats) { Parar ".bat das tarefas mudam ($($bats -join ', ')): fora deste roteiro" }
Write-Host "Arquivos que o alvo muda: $(@($mudam).Count) (nenhum em data/, requirements.txt ou .bat)"

# 4. processos e tarefas
$procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'scripts[\\.]auto_capture|scripts[\\.]ciclo_semanal|scripts[\\.]publish_report|src[\\.]card_report|scripts[\\.]new_event' }
if ($procs) { $procs | ForEach-Object { Write-Host "  pid $($_.ProcessId): $($_.CommandLine)" }; Parar "processo do projeto em execucao; tente de novo quando terminar" }
$tarefas = Get-ScheduledTask | Where-Object { ($_.Actions | ForEach-Object { "$($_.Execute) $($_.Arguments) $($_.WorkingDirectory)" }) -match 'auto_capture|ciclo_semanal' }
foreach ($t in $tarefas) {
    $i = Get-ScheduledTaskInfo -TaskName $t.TaskName -TaskPath $t.TaskPath
    Write-Host "Tarefa '$($t.TaskName)': estado $($t.State); proxima $($i.NextRunTime); ultima $($i.LastRunTime) (resultado $($i.LastTaskResult)) -- NAO alterada"
    if ($i.NextRunTime -and ($i.NextRunTime - (Get-Date)).TotalMinutes -lt 10 -and ($i.NextRunTime -gt (Get-Date))) {
        Parar "a tarefa '$($t.TaskName)' roda em menos de 10 min; rode o roteiro depois dela"
    }
}
if (-not $tarefas) { Write-Host "Nenhuma tarefa agendada apontando para auto_capture/ciclo_semanal encontrada (confira manualmente)." -ForegroundColor Yellow }

# 5. clone temporario do alvo + preflight dos registros reais
$clone = Join-Path $trabalho "clone"
$ant = $ErrorActionPreference; $ErrorActionPreference = "Continue"
& git clone --quiet --no-hardlinks $Arvore $clone 2>&1 | Out-Null
& git -C $clone checkout --quiet --detach $Alvo 2>&1 | Out-Null
$ErrorActionPreference = $ant
$noClone = (& git -C $clone rev-parse HEAD).Trim()
if ($noClone -ne $Alvo) { Parar "clone temporario em $noClone, esperado $Alvo" }
& git -C $clone remote remove origin 2>$null
Manifesto (Join-Path $Arvore "data") (Join-Path $trabalho "data_antes.sha256")
$rc = Py $clone @("docs/chaos/reproducoes/preflight_registros.py", $Arvore)
if ($rc -ne 0) { Parar "registros reais nao passam nas regras do codigo novo (ver acima). Nada foi alterado." }
Manifesto (Join-Path $Arvore "data") (Join-Path $trabalho "data_preflight.sha256")
if (Compare-Object (Get-Content (Join-Path $trabalho "data_antes.sha256")) (Get-Content (Join-Path $trabalho "data_preflight.sha256"))) {
    Parar "data\ mudou durante o pre-voo (algum job gravou?). Rode de novo."
}
Write-Host "PRE-VOO APROVADO." -ForegroundColor Green

if (-not $Aplicar) {
    Write-Host "Modo verificacao: nada foi alterado. Para instalar, rode de novo com -Aplicar. Log: $logGeral"
    Stop-Transcript | Out-Null; exit 0
}

# === APLICAR ================================================================
Write-Host "=== APLICAR"
$backup = Join-Path $env:USERPROFILE "ufc-backup-instalacao-$carimbo"
New-Item -ItemType Directory -Force -Path $backup | Out-Null
foreach ($p in "data", "models") {
    $o = Join-Path $Arvore $p
    if (Test-Path $o) { Copy-Item $o (Join-Path $backup $p) -Recurse }
}
Copy-Item (Join-Path $trabalho "data_antes.sha256") $backup
"HEAD antes: $antigo`nalvo: $Alvo`n$($locais -join "`n")" | Set-Content (Join-Path $backup "estado_git.txt")
Write-Host "Backup de data\ e models\ em $backup"

$procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'scripts[\\.]auto_capture|scripts[\\.]ciclo_semanal|scripts[\\.]publish_report|src[\\.]card_report|scripts[\\.]new_event' }
if ($procs) { Parar "processo do projeto comecou durante o backup; nada foi alterado no git" }

$m = Invoke-Git @("merge", "--ff-only", $Alvo)
$m.out | ForEach-Object { Write-Host "  $_" }
if ($m.rc -ne 0) { Parar "git merge --ff-only falhou (codigo $($m.rc)); a arvore continua em $antigo" }
$novo = ((Invoke-Git @("rev-parse", "HEAD")).out | Select-Object -Last 1).Trim()
if ($novo -ne $Alvo) { Parar "HEAD $novo apos o merge, esperado $Alvo" }

Manifesto (Join-Path $Arvore "data") (Join-Path $trabalho "data_depois.sha256")
$dif = Compare-Object (Get-Content (Join-Path $trabalho "data_antes.sha256")) (Get-Content (Join-Path $trabalho "data_depois.sha256"))
if ($dif) { $dif | Out-Host; Parar "data\ mudou durante a instalacao. Reverter: git -C `"$Arvore`" reset --keep $antigo ; comparar com o backup $backup" }
$locaisDepois = (Invoke-Git @("status", "--porcelain")).out
if ((@($locais) -join "`n") -ne (@($locaisDepois) -join "`n")) { Write-Host "Atencao: alteracoes locais mudaram:" -ForegroundColor Yellow; $locaisDepois | Out-Host }

$rc = Py $clone @("docs/chaos/reproducoes/preflight_registros.py", $Arvore)
if ($rc -ne 0) { Parar "preflight pos-instalacao reprovou. Reverter: git -C `"$Arvore`" reset --keep $antigo" }
$rc = Py $Arvore @("-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_report_freshness.py")
if ($rc -ne 0) { Parar "teste de frescor da pagina reprovou na arvore. Reverter: git -C `"$Arvore`" reset --keep $antigo" }

Write-Host "INSTALADO: $antigo -> $Alvo. data\ com sha256 igual; tarefas, modelos, candidatos e registros nao alterados." -ForegroundColor Green
Write-Host "Reversao (mantem alteracoes locais de data\): git -C `"$Arvore`" reset --keep $antigo"
Write-Host "Backup: $backup | Log: $logGeral"
Write-Host "Primeira rodada horaria criara data\.registros.lock (ignorado pelo git). Candidatos congelados antes de 4c74a4d serao bloqueados na conferencia de codigo (comportamento desenhado)."
Stop-Transcript | Out-Null
exit 0
