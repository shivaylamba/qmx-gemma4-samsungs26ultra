param(
    [string]$AndroidSdk = "$env:LOCALAPPDATA\Android\Sdk",
    [string]$NdkVersion = "28.1.13356709",
    [string]$CMakeVersion = "3.31.6",
    [ValidateSet("all", "qmx", "neon")]
    [string]$Backend = "all"
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sourceDir = Join-Path $projectRoot "third_party\llama.cpp-current"
$ndkDir = Join-Path $AndroidSdk "ndk\$NdkVersion"
$cmakeExe = Join-Path $AndroidSdk "cmake\$CMakeVersion\bin\cmake.exe"
$ninjaExe = Join-Path $AndroidSdk "cmake\$CMakeVersion\bin\ninja.exe"
$toolchainFile = Join-Path $ndkDir "build\cmake\android.toolchain.cmake"

foreach ($path in @($sourceDir, $cmakeExe, $ninjaExe, $toolchainFile)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path does not exist: $path"
    }
}

$backends = if ($Backend -eq "all") { @("qmx", "neon") } else { @($Backend) }

foreach ($name in $backends) {
    $buildDir = Join-Path $projectRoot "build\standalone\$name"
    New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

    $commonArgs = @(
        "-S", $sourceDir,
        "-B", $buildDir,
        "-G", "Ninja",
        "-DCMAKE_MAKE_PROGRAM=$ninjaExe",
        "-DCMAKE_TOOLCHAIN_FILE=$toolchainFile",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DANDROID_ABI=arm64-v8a",
        "-DANDROID_PLATFORM=android-29",
        "-DLLAMA_CURL=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DGGML_LLAMAFILE=OFF",
        "-DGGML_SYSTEM_ARCH=ARM",
        "-DGGML_CPU_AARCH64=ON",
        "-DGGML_CPU_ALL_VARIANTS=OFF",
        "-DBUILD_SHARED_LIBS=ON"
    )

    if ($name -eq "qmx") {
        $arch = "armv9.2-a+sve2+sme+dotprod+i8mm"
        $backendArgs = @(
            "-DGGML_CPU_KLEIDIAI=ON",
            "-DGGML_CPU_ARM_ARCH=$arch",
            "-DCMAKE_C_FLAGS=-march=$arch",
            "-DCMAKE_CXX_FLAGS=-march=$arch"
        )
    } else {
        $arch = "armv8-a"
        $backendArgs = @(
            "-DGGML_CPU_KLEIDIAI=OFF",
            "-DGGML_CPU_ARM_ARCH=$arch",
            "-DCMAKE_C_FLAGS=-march=$arch",
            "-DCMAKE_CXX_FLAGS=-march=$arch"
        )
    }

    & $cmakeExe @commonArgs @backendArgs
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed for $name."
    }

    & $cmakeExe --build $buildDir --target llama-batched-bench --parallel
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed for $name."
    }

    Write-Host "Built $name standalone artifacts in $buildDir\bin"
}
