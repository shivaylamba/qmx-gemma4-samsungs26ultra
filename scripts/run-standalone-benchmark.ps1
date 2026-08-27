param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteModel,
    [string]$ModelLabel = "model",
    [string]$AndroidSdk = "$env:LOCALAPPDATA\Android\Sdk",
    [string]$DeviceDirectory = "/data/local/tmp/qmx-standalone",
    [string]$OutputDirectory = "",
    [string]$Serial = "",
    [int]$Repetitions = 4,
    [int[]]$ThreadCounts = @(1, 4),
    [ValidateSet("qmx", "neon")]
    [string[]]$Modes = @("qmx", "neon"),
    [string]$CpuMask = "",
    [int]$PrefillTokens = 128,
    [int]$DecodeTokens = 128,
    [int]$ContextSize = 2048,
    [int]$BatchSize = 2048,
    [int]$MicroBatchSize = 512
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$adb = Join-Path $AndroidSdk "platform-tools\adb.exe"
if (-not (Test-Path -LiteralPath $adb)) {
    throw "adb was not found at $adb"
}

if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $projectRoot ("results\standalone-" + (Get-Date -Format "yyyy-MM-dd-HHmmss"))
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$adbArgs = @()
if ($Serial) {
    $adbArgs += @("-s", $Serial)
}

function Invoke-Adb {
    & $adb @adbArgs @args
    if ($LASTEXITCODE -ne 0) {
        throw "adb failed: $args"
    }
}

function Get-TemperatureC {
    $line = Invoke-Adb shell dumpsys battery | Select-String "^  temperature:"
    return [double]($line.ToString().Split(":")[-1].Trim()) / 10
}

Invoke-Adb get-state | Out-Null
Invoke-Adb shell test -r $RemoteModel
Invoke-Adb shell cmd power set-fixed-performance-mode-enabled true
Invoke-Adb shell svc power stayon true

$rows = @()
try {
    foreach ($threads in $ThreadCounts) {
        foreach ($trial in 1..$Repetitions) {
            foreach ($mode in $Modes) {
                $remoteDirectory = "$DeviceDirectory/$mode"
                $environment = if ($mode -eq "qmx") { "GGML_KLEIDIAI_SME=1 " } else { "" }
                $launcher = if ($CpuMask) { "taskset $CpuMask " } else { "" }
                $temperatureStart = Get-TemperatureC
                $started = (Get-Date).ToString("o")
                $command = "cd $remoteDirectory && LD_LIBRARY_PATH=. ${environment}${launcher}./llama-batched-bench --output-format jsonl -m $RemoteModel -c $ContextSize -b $BatchSize -ub $MicroBatchSize -npp $PrefillTokens -ntg $DecodeTokens -npl 1 -t $threads -fa on"
                $savedErrorActionPreference = $ErrorActionPreference
                $ErrorActionPreference = "Continue"
                $raw = @(& $adb @adbArgs shell $command 2>&1 | ForEach-Object { $_.ToString() })
                $benchmarkExitCode = $LASTEXITCODE
                $ErrorActionPreference = $savedErrorActionPreference
                if ($benchmarkExitCode -ne 0) {
                    throw "Benchmark failed for $mode, $threads threads, trial $trial."
                }
                $ended = (Get-Date).ToString("o")
                $temperatureEnd = Get-TemperatureC
                $logName = "{0}-{1}-{2}t-run{3}.log" -f $ModelLabel, $mode, $threads, $trial
                $raw | Set-Content -LiteralPath (Join-Path $OutputDirectory $logName) -Encoding utf8
                $jsonLine = $raw | Where-Object { $_ -match "^\{" } | Select-Object -Last 1
                if (-not $jsonLine) {
                    throw "No JSON result was emitted for $mode, $threads threads, trial $trial."
                }
                $data = $jsonLine | ConvertFrom-Json
                $rows += [PSCustomObject]@{
                    model = $ModelLabel
                    mode = $mode
                    threads = $threads
                    trial = $trial
                    warmup = ($trial -eq 1)
                    temp_start_c = $temperatureStart
                    temp_end_c = $temperatureEnd
                    started = $started
                    ended = $ended
                    pp_tokens = $data.pp
                    tg_tokens = $data.tg
                    pp_seconds = $data.t_pp
                    pp_tok_s = $data.speed_pp
                    tg_seconds = $data.t_tg
                    tg_tok_s = $data.speed_tg
                }
                Write-Host ("{0} {1}t run {2}: prefill {3:F3} tok/s, decode {4:F3} tok/s, {5:F1}->{6:F1} C" -f $mode, $threads, $trial, $data.speed_pp, $data.speed_tg, $temperatureStart, $temperatureEnd)
            }
        }
    }
} finally {
    & $adb @adbArgs shell cmd power set-fixed-performance-mode-enabled false | Out-Null
    & $adb @adbArgs shell svc power stayon false | Out-Null
}

$csvPath = Join-Path $OutputDirectory "$ModelLabel-trials.csv"
$rows | Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding utf8
Write-Host "Raw logs and trial data written to $OutputDirectory"
