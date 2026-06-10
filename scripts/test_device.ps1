# Build and run kws_smoke_test on a connected Android device (adb).
param(
    [string]$Abi = "arm64-v8a",
    [int]$ApiLevel = 24,
    [string]$NdkRoot = $env:ANDROID_NDK_HOME
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root "build\smoke_test_$Abi"
$DistLib = Join-Path $Root "dist\$Abi\lib\libhey_siri_kws.so"

if (-not (Test-Path $DistLib)) {
    throw "Missing $DistLib. Run scripts/build.ps1 -Abi $Abi first."
}

if (-not $NdkRoot -or -not (Test-Path $NdkRoot)) {
    $SdkNdk = Join-Path $env:LOCALAPPDATA "Android\Sdk\ndk"
    if (Test-Path $SdkNdk) {
        $NdkRoot = (Get-ChildItem $SdkNdk -Directory | Sort-Object Name -Descending | Select-Object -First 1).FullName
    }
}
if (-not $NdkRoot) { throw "Set ANDROID_NDK_HOME." }

$Toolchain = Join-Path $NdkRoot "build\cmake\android.toolchain.cmake"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

cmake -S (Join-Path $Root "tools") -B $BuildDir `
    -G Ninja `
    -DCMAKE_TOOLCHAIN_FILE="$Toolchain" `
    -DANDROID_ABI="$Abi" `
    -DANDROID_PLATFORM="android-$ApiLevel" `
    -DANDROID_STL=c++_static `
    -DCMAKE_BUILD_TYPE=Release

cmake --build $BuildDir
if ($LASTEXITCODE -ne 0) { throw "smoke test build failed" }

$Bin = Join-Path $BuildDir "kws_smoke_test"
if (-not (Test-Path $Bin)) { throw "Missing $Bin" }

$RemoteDir = "/data/local/tmp/hey_siri_kws_test"
adb shell "mkdir -p $RemoteDir"
adb push $DistLib "$RemoteDir/libhey_siri_kws.so"
adb push $Bin "$RemoteDir/kws_smoke_test"
adb shell "chmod 755 $RemoteDir/kws_smoke_test"

Write-Host ""
Write-Host "Running smoke test on device ..."
adb shell "cd $RemoteDir && LD_LIBRARY_PATH=$RemoteDir ./kws_smoke_test"
$code = $LASTEXITCODE
if ($code -ne 0) { throw "Device smoke test failed (exit $code)" }

Write-Host ""
Write-Host "=== Device smoke test PASSED ==="
