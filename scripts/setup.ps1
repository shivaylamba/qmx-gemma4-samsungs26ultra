$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$llamaDirectory = Join-Path $projectRoot "third_party\llama.cpp-current"
$patchFile = Join-Path $projectRoot "patches\llama.cpp-gemma4-thinking.patch"

git -C $projectRoot submodule update --init --recursive
if ($LASTEXITCODE -ne 0) {
    throw "Could not initialize the llama.cpp submodule."
}

$ErrorActionPreference = "Continue"
git -C $llamaDirectory apply --check $patchFile *> $null
$patchCanApply = $LASTEXITCODE -eq 0
$ErrorActionPreference = "Stop"

if ($patchCanApply) {
    git -C $llamaDirectory apply $patchFile
    if ($LASTEXITCODE -ne 0) {
        throw "Could not apply the Gemma 4 chat-template patch."
    }
    Write-Host "Applied the Gemma 4 chat-template patch."
} else {
    $ErrorActionPreference = "Continue"
    git -C $llamaDirectory apply --reverse --check $patchFile *> $null
    $patchIsApplied = $LASTEXITCODE -eq 0
    $ErrorActionPreference = "Stop"
    if (-not $patchIsApplied) {
        throw "The llama.cpp checkout does not match the pinned patch. Reset the submodule and run setup again."
    }
    Write-Host "Gemma 4 chat-template patch is already applied."
}

Write-Host "Setup complete. Build with .\gradlew.bat :app:assembleDebug"
