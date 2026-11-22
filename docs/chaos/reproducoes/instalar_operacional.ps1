<#
Instalacao da pilha CHAOS na MAQUINA OPERACIONAL (Windows).

Rode a partir de uma copia deste arquivo FORA da arvore operacional (por
exemplo em %USERPROFILE%\Downloads): na arvore, ainda em 2f73748, o arquivo
nao existe, e uma copia dentro dela viraria arquivo nao rastreado que o alvo
sobrescreveria.

  # 1) so verificar (padrao): a arvore nao e alterada
  powershell -ExecutionPolicy Bypass -File <copia>\instalar_operacional.ps1 `
      -Arvore "C:\caminho\do\ufc-predictor" `
      -Python "C:\caminho\do\ufc-predictor\.venv\Scripts\python.exe"

  # 2) aplicar, depois de ler o pre-voo aprovado
  ... -Aplicar

Entre 2f73748 e o alvo NAO mudam data/, requirements.txt nem os .bat das
tarefas; a instalacao e apenas `git merge --ff-only`, sem pip e sem tocar nas
tarefas.

Pre-voo (sempre; na arvore so leitura, exceto `git fetch`, que atualiza
referencias remotas):
  - branch main (o roteiro NAO troca branch); HEAD ancestral do alvo;
  - alteracoes locais (inclusive nao rastreados, arquivo a arquivo) sem
    colisao com arquivos que o alvo muda;
  - nenhum processo do projeto rodando; tarefas do projeto a mais de 10 min;
  - clone temporario do alvo (copia local, so leitura na origem) e
    preflight_registros.py sobre os registros REAIS.
Aplicar (-Aplicar):
  - backup de data\ e models\ fora do repositorio, conferido por sha256;
  - IMEDIATAMENTE antes do merge: data\ e models\ iguais ao manifesto do
    pre-voo, nenhum processo, tarefa a mais de 10 min -- senao PARA sem mexer
    no git;
  - git merge --ff-only <alvo>; confere HEAD, data\ e models\, preflight e
    frescor da pagina.
Quando o roteiro para ANTES do merge, a arvore nao muda (ficam so a pasta
temporaria e, no Aplicar, o backup). Paradas DEPOIS do merge dizem o estado
exato e o comando de reversao.
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
$PADRAO_PROC = 'scripts[\\./]auto_capture|scripts[\\./]ciclo_semanal|scripts[\\./]publish_report|src[\\./]card_report|scripts[\\./]new_event'

function Parar([string]$msg) {
    Write-Host "PARADO: $msg" -ForegroundColor Red
    Write-Host "Log: $logGeral"
    Stop-Transcript | Out-Null; exit 1
}
# Nao pode se chamar 'Git': o PowerShell ignora caixa e a funcao sombrearia o
# executavel (recursao infinita). stderr do git nao vira erro terminal (PS 5.1).
function Invoke-Git([string]$repo, [string[]]$a) {
    $ant = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $out = & git --no-optional-locks -c core.quotepath=false -C $repo @a 2>&1 | ForEach-Object { "$_" }; $rc = $LASTEXITCODE }
    finally { $ErrorActionPreference = $ant }
    return [pscustomobject]@{ rc = $rc; out = @($out) }
}
function Py([string]$cwd, [string[]]$a) {
    $ant = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    Push-Location $cwd
    try { & $Python @a 2>&1 | ForEach-Object { "$_" } | Out-Host; $rc = $LASTEXITCODE }
    finally { Pop-Location; $ErrorActionPreference = $ant }
    return $rc
}
# Manifesto sha256 de uma pasta (texto). Falha de leitura vira parada clara.
function Manifesto([string]$pasta) {
    if (-not (Test-Path $pasta)) { return "" }
    try {
        $l = Get-ChildItem $pasta -Recurse -File | Where-Object { $_.Name -ne ".registros.lock" } |
            Sort-Object FullName | ForEach-Object {
                $h = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
                "$h  $($_.FullName.Substring($pasta.Length))"
            }
        return (@($l) -join "`n")
    } catch { Parar "nao consegui ler $pasta para o manifesto (arquivo aberto por outro programa?): $($_.Exception.Message)" }
}
function ProcessosDoProjeto { @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match $PADRAO_PROC }) }
function TarefasProximas {
    $perto = @()
    $tarefas = @(Get-ScheduledTask | Where-Object { ($_.Actions | ForEach-Object { "$($_.Execute) $($_.Arguments) $($_.WorkingDirectory)" }) -match 'auto_capture|ciclo_semanal' })
    foreach ($t in $tarefas) {
        $i = Get-ScheduledTaskInfo -TaskName $t.TaskName -TaskPath $t.TaskPath
        if ($i.NextRunTime -and $i.NextRunTime -gt (Get-Date) -and ($i.NextRunTime - (Get-Date)).TotalMinutes -lt 10) { $perto += "$($t.TaskName) as $($i.NextRunTime)" }
    }
    return [pscustomobject]@{ tarefas = $tarefas; perto = $perto }
}

if ($Alvo -notmatch '^[0-9a-f]{40}$') { Parar "-Alvo precisa ser SHA completo" }
$Arvore = (Resolve-Path $Arvore).Path
if (-not (Test-Path (Join-Path $Arvore ".git"))) { Parar "$Arvore nao e repositorio git" }
if ($PSCommandPath -and $PSCommandPath.StartsWith($Arvore, [StringComparison]::OrdinalIgnoreCase)) {
    Parar "rode uma copia do roteiro FORA da arvore operacional (ver cabecalho)"
}
foreach ($v in "ODDS_API_KEY", "GH_TOKEN", "GITHUB_TOKEN") { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
$env:PYTHONDONTWRITEBYTECODE = "1"
Write-Host "=== PRE-VOO ($carimbo) -- arvore $Arvore; alvo $Alvo"
Write-Host "Windows: $((Get-CimInstance Win32_OperatingSystem).Caption) $((Get-CimInstance Win32_OperatingSystem).Version) | Python: $(& $Python --version 2>&1)"

# 1. branch e HEAD
$branch = ((Invoke-Git $Arvore @("rev-parse", "--abbrev-ref", "HEAD")).out | Select-Object -Last 1)
if ($branch -ne "main") { Parar "a arvore esta na branch '$branch'; o roteiro nao troca branch" }
$antigo = ((Invoke-Git $Arvore @("rev-parse", "HEAD")).out | Select-Object -Last 1).Trim()
Write-Host "HEAD atual: $antigo (main)"
if ($antigo -eq $Alvo) { Write-Host "Ja esta no alvo. Nada a instalar." -ForegroundColor Green; Stop-Transcript | Out-Null; exit 0 }

# 2. alvo disponivel e HEAD ancestral dele
$f = Invoke-Git $Arvore @("fetch", "--quiet", "origin", "main")
if ($f.rc -ne 0) { Parar "git fetch falhou: $($f.out -join ' ')" }
if ((Invoke-Git $Arvore @("cat-file", "-e", "$Alvo^{commit}")).rc -ne 0) { Parar "alvo $Alvo nao existe apos o fetch" }
if ((Invoke-Git $Arvore @("merge-base", "--is-ancestor", $antigo, $Alvo)).rc -ne 0) {
    $loc = (Invoke-Git $Arvore @("log", "--oneline", "$Alvo..$antigo")).out
    Parar "HEAD tem commit(s) que nao estao no alvo (nao enviados?):`n$($loc -join "`n")`nResolva antes de instalar."
}

# 3. alteracoes locais (arquivo a arquivo, sem deteccao de renomeacao) x arquivos que o alvo muda
$mudam = (Invoke-Git $Arvore @("diff", "--no-renames", "--name-only", $antigo, $Alvo)).out
$st = Invoke-Git $Arvore @("status", "--porcelain", "--untracked-files=all", "--no-renames")
if ($st.rc -ne 0) { Parar "git status falhou: $($st.out -join ' ')" }
$locais = @($st.out | Where-Object { $_ })
Write-Host "Alteracoes locais ($($locais.Count)):"; $locais | ForEach-Object { Write-Host "  $_" }
$colide = @()
foreach ($l in $locais) {
    $arq = $l.Substring(3).Trim().Trim('"') -replace '\\', '/'
    if ($mudam -contains $arq) { $colide += $arq }
}
if ($colide) { Parar "alteracoes locais (ou nao rastreados) em arquivos que o alvo muda: $($colide -join ', ')" }
$emData = @($mudam | Where-Object { $_ -like "data/*" })
if ($emData) { Parar "o alvo altera data/ -- inesperado: $($emData -join ', ')" }
if ($mudam -contains "requirements.txt") { Parar "requirements.txt muda entre $antigo e o alvo: exige instalar dependencias, fora deste roteiro" }
$bats = @($mudam | Where-Object { $_ -like "*.bat" })
if ($bats) { Parar ".bat das tarefas mudam ($($bats -join ', ')): fora deste roteiro" }
Write-Host "Arquivos que o alvo muda: $($mudam.Count) (nenhum em data/, requirements.txt ou .bat)"

# 4. processos e tarefas (estado so reportado; nada e alterado)
$p = ProcessosDoProjeto
if ($p) { $p | ForEach-Object { Write-Host "  pid $($_.ProcessId): $($_.CommandLine)" }; Parar "processo do projeto em execucao; tente quando terminar" }
$t = TarefasProximas
foreach ($x in $t.tarefas) {
    $i = Get-ScheduledTaskInfo -TaskName $x.TaskName -TaskPath $x.TaskPath
    Write-Host "Tarefa '$($x.TaskName)': estado $($x.State); proxima $($i.NextRunTime); ultima $($i.LastRunTime) (resultado $($i.LastTaskResult)) -- NAO alterada"
}
if (-not $t.tarefas) { Write-Host "Nenhuma tarefa agendada apontando para auto_capture/ciclo_semanal encontrada (confira manualmente)." -ForegroundColor Yellow }
if ($t.perto) { Parar "tarefa a menos de 10 min: $($t.perto -join '; '). Rode depois dela." }

# 5. clone temporario do alvo + preflight dos registros reais
$clone = Join-Path $trabalho "clone"
$ant = $ErrorActionPreference; $ErrorActionPreference = "Continue"
& git clone --quiet --no-hardlinks $Arvore $clone 2>&1 | Out-Null; $rcClone = $LASTEXITCODE
$ErrorActionPreference = $ant
if ($rcClone -ne 0 -or -not (Test-Path (Join-Path $clone ".git"))) { Parar "git clone local falhou (codigo $rcClone)" }
if ((Invoke-Git $clone @("checkout", "--quiet", "--detach", $Alvo)).rc -ne 0) { Parar "checkout do alvo no clone falhou" }
$noClone = ((Invoke-Git $clone @("rev-parse", "HEAD")).out | Select-Object -Last 1)
if (-not $noClone -or $noClone.Trim() -ne $Alvo) { Parar "clone temporario em '$noClone', esperado $Alvo" }
Invoke-Git $clone @("remote", "remove", "origin") | Out-Null
$dataAntes = Manifesto (Join-Path $Arvore "data")
$modelsAntes = Manifesto (Join-Path $Arvore "models")
Set-Content (Join-Path $trabalho "data_antes.sha256") $dataAntes
Set-Content (Join-Path $trabalho "models_antes.sha256") $modelsAntes
$rc = Py $clone @("docs/chaos/reproducoes/preflight_registros.py", $Arvore)
if ($rc -ne 0) { Parar "registros reais nao passam nas regras do codigo novo (ver acima). A arvore nao foi alterada." }
if ((Manifesto (Join-Path $Arvore "data")) -ne $dataAntes) { Parar "data\ mudou durante o pre-voo (algum job gravou?). A arvore nao foi alterada pelo roteiro; rode de novo." }
Write-Host "PRE-VOO APROVADO." -ForegroundColor Green

if (-not $Aplicar) {
    Write-Host "Modo verificacao: a arvore nao foi alterada (so 'git fetch'). Para instalar, rode de novo com -Aplicar."
    Write-Host "Log: $logGeral"
    Stop-Transcript | Out-Null; exit 0
}

# === APLICAR ================================================================
Write-Host "=== APLICAR"
$backup = Join-Path $env:USERPROFILE "ufc-backup-instalacao-$carimbo"
New-Item -ItemType Directory -Force -Path $backup | Out-Null
try {
    foreach ($pp in "data", "models") {
        $o = Join-Path $Arvore $pp
        if (Test-Path $o) { Copy-Item $o (Join-Path $backup $pp) -Recurse }
    }
} catch { Parar "falha no backup ($($_.Exception.Message)). A arvore nao foi alterada; backup parcial em $backup" }
if ((Manifesto (Join-Path $backup "data")) -ne $dataAntes) { Parar "backup de data\ nao confere com o manifesto do pre-voo (data\ mudou durante a copia?). A arvore nao foi alterada." }
if ((Manifesto (Join-Path $backup "models")) -ne $modelsAntes) { Parar "backup de models\ nao confere com o manifesto. A arvore nao foi alterada." }
Copy-Item (Join-Path $trabalho "data_antes.sha256"), (Join-Path $trabalho "models_antes.sha256") $backup
"HEAD antes: $antigo`nalvo: $Alvo`n$($locais -join "`n")" | Set-Content (Join-Path $backup "estado_git.txt")
Write-Host "Backup conferido por sha256: $backup"

# Ultima checagem IMEDIATAMENTE antes do merge: nada mudou e ninguem vai rodar.
if (ProcessosDoProjeto) { Parar "processo do projeto comecou; a arvore nao foi alterada" }
$t = TarefasProximas
if ($t.perto) { Parar "tarefa a menos de 10 min: $($t.perto -join '; '). A arvore nao foi alterada." }
if ((Manifesto (Join-Path $Arvore "data")) -ne $dataAntes) { Parar "data\ mudou desde o pre-voo (job gravou?). A arvore nao foi alterada; rode de novo." }

$m = Invoke-Git $Arvore @("merge", "--ff-only", $Alvo)
$m.out | ForEach-Object { Write-Host "  $_" }
if ($m.rc -ne 0) { Parar "git merge --ff-only falhou (codigo $($m.rc)); a arvore continua em $antigo" }
$novo = ((Invoke-Git $Arvore @("rev-parse", "HEAD")).out | Select-Object -Last 1).Trim()
if ($novo -ne $Alvo) { Parar "HEAD $novo apos o merge, esperado $Alvo. Reverter: git -C `"$Arvore`" reset --keep $antigo" }
Write-Host "Codigo instalado: $antigo -> $Alvo"

# Conferencias pos-merge. O merge nao toca data\ (garantido no pre-voo); se
# data\ mudou, foi gravacao legitima de algum job na janela -- nao e motivo de
# reverter: o preflight decide se os registros continuam validos.
$dataDepois = Manifesto (Join-Path $Arvore "data")
if ($dataDepois -ne $dataAntes) {
    Write-Host "Atencao: data\ mudou entre a ultima checagem e agora (gravacao de job na janela do merge). Conferindo registros..." -ForegroundColor Yellow
}
if ((Manifesto (Join-Path $Arvore "models")) -ne $modelsAntes) { Write-Host "Atencao: models\ mudou (nao pelo roteiro). Compare com $backup." -ForegroundColor Yellow }
$locaisDepois = @((Invoke-Git $Arvore @("status", "--porcelain", "--untracked-files=all", "--no-renames")).out | Where-Object { $_ })
if (($locais -join "`n") -ne ($locaisDepois -join "`n")) { Write-Host "Alteracoes locais depois do merge:" -ForegroundColor Yellow; $locaisDepois | ForEach-Object { Write-Host "  $_" } }

$rc = Py $clone @("docs/chaos/reproducoes/preflight_registros.py", $Arvore)
if ($rc -ne 0) { Parar "codigo JA instalado ($Alvo), mas o preflight pos-instalacao reprovou um registro. Reverter o codigo: git -C `"$Arvore`" reset --keep $antigo ; backup em $backup" }
$rc = Py $Arvore @("-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_report_freshness.py")
if ($rc -ne 0) { Parar "codigo JA instalado ($Alvo), mas o teste de frescor da pagina reprovou (pytest ausente no .venv?). Reverter: git -C `"$Arvore`" reset --keep $antigo" }

Write-Host "INSTALADO: $antigo -> $Alvo. Registros aprovados no codigo novo; data\ e models\ nao alterados pelo roteiro; tarefas nao alteradas." -ForegroundColor Green
Write-Host "Reversao (mantem alteracoes locais de data\): git -C `"$Arvore`" reset --keep $antigo"
Write-Host "Backup: $backup | Log: $logGeral | Pasta temporaria (pode apagar): $trabalho"
Write-Host "A primeira gravacao criara data\.registros.lock (ignorado pelo git). Candidatos congelados antes de 4c74a4d serao bloqueados na conferencia de codigo (comportamento desenhado)."
Stop-Transcript | Out-Null
exit 0
