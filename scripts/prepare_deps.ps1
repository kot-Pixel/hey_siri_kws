# Download TensorFlow Lite native libs + minimal C API headers for NDK build.
param(
    [string]$TfliteVersion = "2.14.0"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$ThirdParty = Join-Path $Root "third_party\tflite"
$DownloadDir = Join-Path $Root "third_party\downloads"

New-Item -ItemType Directory -Force -Path $DownloadDir, "$ThirdParty\include", "$ThirdParty\lib\arm64-v8a", "$ThirdParty\lib\armeabi-v7a" | Out-Null

function Download-File($Url, $Out) {
    if (Test-Path $Out) {
        if ((Get-Item $Out).PSIsContainer) {
            Remove-Item $Out -Recurse -Force
        } else {
            return
        }
    }
    Write-Host "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $Out
}

function Get-AarRoot($AarPath) {
    if (Test-Path $AarPath) {
        $item = Get-Item $AarPath
        if ($item.PSIsContainer) {
            return $AarPath
        }
    }

    $ZipPath = "$AarPath.zip"
    if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
    Copy-Item $AarPath $ZipPath -Force

    $Temp = Join-Path $DownloadDir ("extract_" + [IO.Path]::GetFileNameWithoutExtension($AarPath))
    if (Test-Path $Temp) { Remove-Item -Recurse -Force $Temp }
    Expand-Archive -Path $ZipPath -DestinationPath $Temp -Force
    Remove-Item $ZipPath -Force
    return $Temp
}

function Copy-NativeLibs($AarRoot, $Abi) {
    $NativeDir = Join-Path $AarRoot "jni\$Abi"
    if (-not (Test-Path $NativeDir)) {
        throw "No native libs for $Abi under $AarRoot"
    }
    Copy-Item "$NativeDir\*.so" (Join-Path $ThirdParty "lib\$Abi") -Force
}

function Copy-LiteTree($SourceRoot, $RelativePath, $DestRoot) {
    $Source = Join-Path $SourceRoot $RelativePath
    if (-not (Test-Path $Source)) {
        throw "Missing TensorFlow Lite path: $RelativePath"
    }
    $Dest = Join-Path $DestRoot $RelativePath
    $DestParent = Split-Path $Dest -Parent
    New-Item -ItemType Directory -Force -Path $DestParent | Out-Null
    Copy-Item $Source $Dest -Recurse -Force
}

function Install-CApiHeaders($LiteSrcRoot) {
    $DestRoot = Join-Path $ThirdParty "include\tensorflow\lite"
    if (Test-Path $DestRoot) {
        Remove-Item $DestRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $DestRoot | Out-Null

    foreach ($rel in @("c", "core\c", "core\async\c", "builtin_ops.h")) {
        Copy-LiteTree $LiteSrcRoot $rel $DestRoot
    }
}

$LiteAar = Join-Path $DownloadDir "tensorflow-lite-$TfliteVersion.aar"
$FlexAar = Join-Path $DownloadDir "tensorflow-lite-select-tf-ops-$TfliteVersion.aar"
$HeadersZip = Join-Path $DownloadDir "tensorflow-$TfliteVersion.zip"

Download-File "https://repo1.maven.org/maven2/org/tensorflow/tensorflow-lite/$TfliteVersion/tensorflow-lite-$TfliteVersion.aar" $LiteAar
Download-File "https://repo1.maven.org/maven2/org/tensorflow/tensorflow-lite-select-tf-ops/$TfliteVersion/tensorflow-lite-select-tf-ops-$TfliteVersion.aar" $FlexAar

$LiteRoot = Get-AarRoot $LiteAar
$FlexRoot = Get-AarRoot $FlexAar
foreach ($abi in @("arm64-v8a", "armeabi-v7a")) {
    Copy-NativeLibs $LiteRoot $abi
    Copy-NativeLibs $FlexRoot $abi
}

Download-File "https://github.com/tensorflow/tensorflow/archive/refs/tags/v$TfliteVersion.zip" $HeadersZip
$ExtractRoot = Join-Path $DownloadDir "tensorflow-$TfliteVersion"
if (Test-Path $ExtractRoot) {
    Remove-Item $ExtractRoot -Recurse -Force
}
Expand-Archive -Path $HeadersZip -DestinationPath $DownloadDir -Force

$LiteSrc = Join-Path $ExtractRoot "tensorflow\lite"
Install-CApiHeaders $LiteSrc
Remove-Item $ExtractRoot -Recurse -Force

Write-Host "Dependencies ready under $ThirdParty"
Write-Host "  headers: C API only (no java/)"
Write-Host "  lib/arm64-v8a: $((Get-ChildItem (Join-Path $ThirdParty 'lib\arm64-v8a')).Name -join ', ')"
Write-Host "  lib/armeabi-v7a: $((Get-ChildItem (Join-Path $ThirdParty 'lib\armeabi-v7a')).Name -join ', ')"
