# Build libhey_siri_kws.so for Android ARM.
# Output: dist/<abi>/{lib,include,runtime}/
param(
    [string]$Abi = "arm64-v8a",
    [int]$ApiLevel = 24,
    [string]$NdkRoot = $env:ANDROID_NDK_HOME
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root "build\$Abi"
$DistDir = Join-Path $Root "dist\$Abi"

function Get-NdkTriple($AbiName) {
    switch ($AbiName) {
        "arm64-v8a" { return "aarch64-linux-android" }
        "armeabi-v7a" { return "arm-linux-androideabi" }
        "x86_64" { return "x86_64-linux-android" }
        "x86" { return "i686-linux-android" }
        default { throw "Unsupported ABI: $AbiName" }
    }
}

if (-not $NdkRoot -or -not (Test-Path $NdkRoot)) {
    $SdkNdk = Join-Path $env:LOCALAPPDATA "Android\Sdk\ndk"
    if (Test-Path $SdkNdk) {
        $NdkRoot = (Get-ChildItem $SdkNdk -Directory | Sort-Object Name -Descending | Select-Object -First 1).FullName
    }
}
if (-not $NdkRoot -or -not (Test-Path $NdkRoot)) {
    throw "Set ANDROID_NDK_HOME to Android NDK path."
}

$Toolchain = Join-Path $NdkRoot "build\cmake\android.toolchain.cmake"
if (-not (Test-Path $Toolchain)) {
    throw "NDK cmake toolchain not found: $Toolchain"
}

# Embed MFCC-input streaming model (no Flex ops)
$ModelSrc = Join-Path $Root "model_mfcc\stream_state_internal.tflite"
$ModelDst = Join-Path $Root "src\model_data.cc"
if (-not (Test-Path $ModelSrc)) {
    throw "Missing model: $ModelSrc"
}
python (Join-Path $Root "scripts\embed_tflite.py") $ModelSrc $ModelDst

# Download TFLite deps if needed
$TfliteLibDir = Join-Path $Root "third_party\tflite\lib\$Abi"
$TfliteLib = Join-Path $TfliteLibDir "libtensorflowlite_jni.so"
if (-not (Test-Path $TfliteLib)) {
    $TfliteLib = Join-Path $TfliteLibDir "libtensorflowlite.so"
}
if (-not (Test-Path $TfliteLib)) {
    & (Join-Path $Root "scripts\prepare_deps.ps1")
}

New-Item -ItemType Directory -Force -Path $BuildDir, "$DistDir\lib", "$DistDir\include" | Out-Null

cmake -S $Root -B $BuildDir `
    -G "Ninja" `
    -DCMAKE_TOOLCHAIN_FILE="$Toolchain" `
    -DANDROID_ABI="$Abi" `
    -DANDROID_PLATFORM="android-$ApiLevel" `
    -DANDROID_STL=c++_shared `
    -DTFLITE_ROOT="$(Join-Path $Root 'third_party\tflite')" `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_INSTALL_PREFIX="$DistDir"

cmake --build $BuildDir
if ($LASTEXITCODE -ne 0) { throw "cmake --build failed" }

cmake --install $BuildDir
if ($LASTEXITCODE -ne 0) { throw "cmake --install failed" }

# Refresh runtime deps under dist (no stale Flex libs)
$RuntimeLibDir = Join-Path $DistDir "runtime"
if (Test-Path $RuntimeLibDir) {
    Remove-Item $RuntimeLibDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $RuntimeLibDir | Out-Null

$RuntimeLibs = @()
if (Test-Path (Join-Path $TfliteLibDir "libtensorflowlite_jni.so")) {
    Copy-Item (Join-Path $TfliteLibDir "libtensorflowlite_jni.so") $RuntimeLibDir -Force
    $RuntimeLibs += "libtensorflowlite_jni.so"
} elseif (Test-Path (Join-Path $TfliteLibDir "libtensorflowlite.so")) {
    Copy-Item (Join-Path $TfliteLibDir "libtensorflowlite.so") $RuntimeLibDir -Force
    $RuntimeLibs += "libtensorflowlite.so"
}

$NdkTriple = Get-NdkTriple $Abi
$CppShared = Join-Path $NdkRoot "toolchains\llvm\prebuilt\windows-x86_64\sysroot\usr\lib\$NdkTriple\libc++_shared.so"
if (-not (Test-Path $CppShared)) {
    throw "Missing libc++_shared.so for $Abi at $CppShared"
}
Copy-Item $CppShared $RuntimeLibDir -Force
$RuntimeLibs += "libc++_shared.so"

# Optional metadata for integrators
$LabelsSrc = Join-Path $Root "model_mfcc\labels.txt"
if (Test-Path $LabelsSrc) {
    Copy-Item $LabelsSrc (Join-Path $DistDir "labels.txt") -Force
}

Write-Host ""
Write-Host "=== Build OK ==="
Write-Host "  $($DistDir)\lib\libhey_siri_kws.so"
Write-Host "  $($DistDir)\include\hey_siri_kws.h"
Write-Host ""
Write-Host "Runtime deps (ship with your APK jniLibs/$Abi/):"
Write-Host "  $($RuntimeLibDir)\"
foreach ($name in $RuntimeLibs) {
    Write-Host "    $name"
}
