# Build and run kws_bench_test on a connected Android device (adb).
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
    -DCMAKE_BUILD_TYPE=Release | Out-Null

cmake --build $BuildDir --target kws_bench_test
if ($LASTEXITCODE -ne 0) { throw "bench test build failed" }

$Bin = Join-Path $BuildDir "kws_bench_test"
$RemoteDir = "/data/local/tmp/hey_siri_kws_test"
adb shell "mkdir -p $RemoteDir"
adb push $DistLib "$RemoteDir/libhey_siri_kws.so" | Out-Null
adb push $Bin "$RemoteDir/kws_bench_test" | Out-Null
adb shell "chmod 755 $RemoteDir/kws_bench_test"

Write-Host ""
Write-Host "Running benchmark on device ..."
adb shell "cd $RemoteDir && LD_LIBRARY_PATH=$RemoteDir ./kws_bench_test"
