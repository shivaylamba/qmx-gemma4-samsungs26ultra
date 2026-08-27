param(
    [string]$AndroidSdk = "$env:LOCALAPPDATA\Android\Sdk",
    [string]$NdkVersion = "28.1.13356709",
    [string]$DeviceDirectory = "/data/local/tmp/qmx-standalone",
    [string]$Serial = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$adb = Join-Path $AndroidSdk "platform-tools\adb.exe"
if (-not (Test-Path -LiteralPath $adb)) {
    throw "adb was not found at $adb"
}
$libomp = Join-Path $AndroidSdk "ndk\$NdkVersion\toolchains\llvm\prebuilt\windows-x86_64\lib\clang\19\lib\linux\aarch64\libomp.so"
if (-not (Test-Path -LiteralPath $libomp)) {
    throw "The arm64 OpenMP runtime was not found at $libomp"
}

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

foreach ($backend in @("qmx", "neon")) {
    $source = Join-Path $projectRoot "build\standalone\$backend\bin"
    $destination = "$DeviceDirectory/$backend"
    if (-not (Test-Path -LiteralPath (Join-Path $source "llama-batched-bench"))) {
        throw "Build $backend first. Missing $source\llama-batched-bench"
    }

    Invoke-Adb shell mkdir -p $destination
    foreach ($name in @(
        "llama-batched-bench",
        "libllama-batched-bench-impl.so",
        "libllama-common.so",
        "libllama.so",
        "libggml.so",
        "libggml-base.so",
        "libggml-cpu.so"
    )) {
        $path = Join-Path $source $name
        if (Test-Path -LiteralPath $path) {
            Invoke-Adb push $path "$destination/"
        }
    }
    Invoke-Adb push $libomp "$destination/"
    Invoke-Adb shell chmod 755 "$destination/llama-batched-bench"
}

Write-Host "Standalone runners deployed under $DeviceDirectory"
