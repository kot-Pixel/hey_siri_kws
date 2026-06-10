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

# Embed MFCC-input streaming model (prefer INT8 if present)
$ModelInt8 = Join-Path $Root "model_mfcc\stream_state_internal_int8.tflite"
$ModelFloat = Join-Path $Root "model_mfcc\stream_state_internal.tflite"
if (Test-Path $ModelInt8) {
    $ModelSrc = $ModelInt8
} else {
    $ModelSrc = $ModelFloat
}
$ModelDst = Join-Path $Root "src\model_data.cc"
if (-not (Test-Path $ModelSrc)) {
    throw "Missing model: $ModelSrc"
}
python (Join-Path $Root "scripts\embed_tflite.py") $ModelSrc $ModelDst

# Download TFLite deps if needed (dynamic fallback only)
$TfliteStaticDir = Join-Path $Root "third_party\tflite-static\$Abi\lib"
$TfliteStaticMarker = Join-Path $TfliteStaticDir "libtensorflow-lite.a"
$UseTfliteStatic = Test-Path $TfliteStaticMarker

$TfliteLibDir = Join-Path $Root "third_party\tflite\lib\$Abi"
$TfliteLib = Join-Path $TfliteLibDir "libtensorflowlite_jni.so"
if (-not $UseTfliteStatic) {
    if (-not (Test-Path $TfliteLib)) {
        $TfliteLib = Join-Path $TfliteLibDir "libtensorflowlite.so"
    }
    if (-not (Test-Path $TfliteLib)) {
        & (Join-Path $Root "scripts\prepare_deps.ps1")
    }
}

New-Item -ItemType Directory -Force -Path $BuildDir, "$DistDir\lib", "$DistDir\include" | Out-Null

$AndroidStl = if ($UseTfliteStatic) { "c++_static" } else { "c++_shared" }
$CmakeArgs = @(
    "-S", $Root,
    "-B", $BuildDir,
    "-G", "Ninja",
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain",
    "-DANDROID_ABI=$Abi",
    "-DANDROID_PLATFORM=android-$ApiLevel",
    "-DANDROID_STL=$AndroidStl",
    "-DTFLITE_ROOT=$(Join-Path $Root 'third_party\tflite')",
    "-DTFLITE_STATIC_ROOT=$(Join-Path $Root 'third_party\tflite-static')",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=$DistDir"
)
if ($UseTfliteStatic) {
    $CmakeArgs += "-DUSE_TFLITE_STATIC=ON"
} else {
    $CmakeArgs += "-DUSE_TFLITE_STATIC=OFF"
}

cmake @CmakeArgs

cmake --build $BuildDir
if ($LASTEXITCODE -ne 0) { throw "cmake --build failed" }

cmake --install $BuildDir
if ($LASTEXITCODE -ne 0) { throw "cmake --install failed" }

# Strip debug symbols from release .so
$OutSo = Join-Path $DistDir "lib\libhey_siri_kws.so"
$Strip = Join-Path $NdkRoot "toolchains\llvm\prebuilt\windows-x86_64\bin\llvm-strip.exe"
if ((Test-Path $OutSo) -and (Test-Path $Strip)) {
    & $Strip --strip-unneeded $OutSo
}

# Runtime deps: only when using dynamic TFLite
$RuntimeLibDir = Join-Path $DistDir "runtime"
if (Test-Path $RuntimeLibDir) {
    Remove-Item $RuntimeLibDir -Recurse -Force
}

$RuntimeLibs = @()
if (-not $UseTfliteStatic) {
    New-Item -ItemType Directory -Force -Path $RuntimeLibDir | Out-Null
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
}

# Optional metadata for integrators
$LabelsSrc = Join-Path $Root "model_mfcc\labels.txt"
if (Test-Path $LabelsSrc) {
    Copy-Item $LabelsSrc (Join-Path $DistDir "labels.txt") -Force
}

Write-Host ""
Write-Host "=== Build OK ==="
Write-Host "  $($DistDir)\lib\libhey_siri_kws.so"
Write-Host "  $($DistDir)\include\hey_siri_kws.h"
if ($UseTfliteStatic) {
    Write-Host ""
    $Mb = [math]::Round((Get-Item (Join-Path $DistDir "lib\libhey_siri_kws.so")).Length / 1MB, 2)
    Write-Host "Static TFLite linked (no extra runtime .so). Size: ${Mb} MB"
} else {
    Write-Host ""
    Write-Host "Runtime deps (ship with your APK jniLibs/$Abi/):"
    Write-Host "  $($RuntimeLibDir)\"
    foreach ($name in $RuntimeLibs) {
        Write-Host "    $name"
    }
}
