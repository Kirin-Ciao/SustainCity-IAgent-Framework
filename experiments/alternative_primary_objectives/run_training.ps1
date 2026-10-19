[CmdletBinding()]
param(
    [string[]]$Archetype,
    [string[]]$ACP,
    [switch]$DryRun,
    [switch]$StartupTest,
    [switch]$Resume,
    [int]$PauseAfterTimesteps,
    [string]$OutputRoot,
    [string]$Python
)
$ErrorActionPreference='Stop'
$repositoryPath=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
if (-not $Python) { $Python=Join-Path $repositoryPath '.venv\Scripts\python.exe' }
$launchLogDirectory=Join-Path $PSScriptRoot 'logs\launchers'
New-Item -ItemType Directory -Path $launchLogDirectory -Force | Out-Null
$launchStamp=(Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ')
$launchTranscript=Join-Path $launchLogDirectory ($launchStamp+'_'+[guid]::NewGuid().ToString('N')+'.txt')
$trainingExitCode=2
$transcriptStarted=$false
try {
    Start-Transcript -LiteralPath $launchTranscript | Out-Null
    $transcriptStarted=$true
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Project Python executable not found: $Python" }
    $Python=(Resolve-Path -LiteralPath $Python).Path
    $configDirectory=Join-Path $PSScriptRoot 'configs'
    foreach ($requiredFile in @((Join-Path $configDirectory 'batch_manifest.json'),
        (Join-Path $repositoryPath 'sustaincity_iagent\alternative_training.py'),
        (Join-Path $repositoryPath 'sustaincity_iagent\alternative_launch.py'))) {
        if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) { throw "Required launcher file not found: $requiredFile" }
    }
    Write-Output "Project Python: $Python"
    Write-Output "Working directory: $repositoryPath"
    Write-Output "Launcher transcript: $launchTranscript"
    if (-not $OutputRoot) { $OutputRoot=Join-Path $PSScriptRoot 'runs' }
    $trainingArgs=@('-B','-u','-X','utf8','-m','sustaincity_iagent.alternative_training','batch',
        '--config-dir',$configDirectory,'--output-root',$OutputRoot)
    foreach ($item in $Archetype) { $trainingArgs+=@('--archetype',$item) }
    foreach ($item in $ACP) { $trainingArgs+=@('--acp',$item) }
    if ($DryRun) { $trainingArgs+='--dry-run' }
    if ($StartupTest) { $trainingArgs+='--startup-test' }
    if ($Resume) { $trainingArgs+='--resume' }
    if ($PSBoundParameters.ContainsKey('PauseAfterTimesteps')) {
        if ($PauseAfterTimesteps -le 0) { throw 'PauseAfterTimesteps must be positive when supplied.' }
        $trainingArgs+=@('--pause-after',"$PauseAfterTimesteps")
    }
    Push-Location -LiteralPath $repositoryPath
    try { & $Python @trainingArgs; $trainingExitCode=$LASTEXITCODE }
    finally { Pop-Location }
} catch {
    Write-Output ($_ | Format-List * -Force | Out-String)
    Write-Output $_.ScriptStackTrace
    $trainingExitCode=2
} finally {
    if ($transcriptStarted) { Stop-Transcript | Out-Null }
}
exit $trainingExitCode
