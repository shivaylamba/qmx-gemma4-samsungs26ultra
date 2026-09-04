$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$llamaDirectory = Join-Path $projectRoot "third_party\llama.cpp-current"
$patches = @(
    @{
        Path = Join-Path $projectRoot "patches\llama.cpp-gemma4-thinking.patch"
        Label = "Gemma 4 chat-template"
    },
    @{
        Path = Join-Path $projectRoot "patches\llama.cpp-qualcomm-qmx.patch"
        Label = "Qualcomm QMX SME1"
    },
    @{
        Path = Join-Path $projectRoot "patches\llama.cpp-qwen3tts-streaming.patch"
        Label = "Qwen3-TTS streaming PCM"
    }
)

git -C $projectRoot submodule update --init --recursive
if ($LASTEXITCODE -ne 0) {
    throw "Could not initialize the pinned submodules."
}

foreach ($patch in $patches) {
    $ErrorActionPreference = "Continue"
    git -C $llamaDirectory apply --unidiff-zero --check $patch.Path *> $null
    $patchCanApply = $LASTEXITCODE -eq 0
    $ErrorActionPreference = "Stop"

    if ($patchCanApply) {
        git -C $llamaDirectory apply --unidiff-zero $patch.Path
        if ($LASTEXITCODE -ne 0) {
            throw "Could not apply the $($patch.Label) patch."
        }
        Write-Host "Applied the $($patch.Label) patch."
    } else {
        $ErrorActionPreference = "Continue"
        git -C $llamaDirectory apply --unidiff-zero --reverse --check $patch.Path *> $null
        $patchIsApplied = $LASTEXITCODE -eq 0
        $ErrorActionPreference = "Stop"
        if (-not $patchIsApplied) {
            throw "The llama.cpp checkout does not match the $($patch.Label) patch. Reset the submodule and run setup again."
        }
        Write-Host "$($patch.Label) patch is already applied."
    }
}

Write-Host "Setup complete. Build with .\gradlew.bat :app:assembleDebug"
