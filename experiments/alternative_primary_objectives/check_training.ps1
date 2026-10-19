[CmdletBinding()]
param([string]$Python,[switch]$Watch,[double]$Interval=2,[string]$RunsRoot,[int]$Iterations)
$ErrorActionPreference='Stop'
$repositoryPath=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
if (-not $Python) { $Python=Join-Path $repositoryPath '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Project Python executable not found: $Python" }
$Python=(Resolve-Path -LiteralPath $Python).Path
if (-not $RunsRoot) { $RunsRoot=Join-Path $PSScriptRoot 'runs' }
if (-not (Test-Path -LiteralPath $RunsRoot -PathType Container)) { throw "Training output directory not found: $RunsRoot" }
$RunsRoot=(Resolve-Path -LiteralPath $RunsRoot).Path
$monitorArgs=@('-B','-u','-X','utf8','-m','sustaincity_iagent.alternative_monitor','status',
    '--output-root',$RunsRoot,'--interval',"$Interval")
if ($Watch) { $monitorArgs+='--watch' }
if ($Iterations -gt 0) { $monitorArgs+=@('--iterations',"$Iterations") }
Push-Location -LiteralPath $repositoryPath
try { & $Python @monitorArgs; $monitorExitCode=$LASTEXITCODE }
finally { Pop-Location }
exit $monitorExitCode
