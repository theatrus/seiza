# Copy the Visual C++ runtime ONNX Runtime needs beside it, for the zip and the
# MSI. seiza.exe itself stays on a static CRT, but Microsoft's onnxruntime.dll
# links the DLLs dynamically, and a clean Windows has neither.
#
# The list comes from what the libraries import, and has to match what
# Product.wxs and the zip ship, so an ONNX Runtime that needs another DLL
# stops the build here rather than failing on a user's machine.
param(
    [string]$Directory = (Join-Path $PSScriptRoot "..\..\seiza-cli\onnxruntime")
)

$ErrorActionPreference = "Stop"
$expected = @("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll")
$directoryPath = (Resolve-Path -LiteralPath $Directory).Path

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$installationPath = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath | Select-Object -First 1
if ([string]::IsNullOrWhiteSpace($installationPath)) {
    throw "A Visual Studio installation with the x64 C++ tools was not found"
}

$dumpbin = Get-ChildItem -File `
    -Path (Join-Path $installationPath "VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe") |
    Sort-Object FullName -Descending | Select-Object -First 1
if ($null -eq $dumpbin) { throw "dumpbin.exe was not found under $installationPath" }

# Newest redistributable that holds the C++ library, past the v143-style folder
# of merge modules that sits beside the versioned ones.
$redist = Get-ChildItem -Directory `
    -Path (Join-Path $installationPath "VC\Redist\MSVC\*\x64\Microsoft.VC*.CRT") |
    Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "msvcp140.dll") } |
    Sort-Object FullName -Descending | Select-Object -First 1
if ($null -eq $redist) { throw "The Visual C++ redistributable was not found under $installationPath" }
Write-Output "redistributable: $($redist.FullName)"

function Get-VcImports {
    param([string]$Path)
    $report = @(& $dumpbin.FullName /DEPENDENTS $Path)
    if ($LASTEXITCODE -ne 0) { throw "dumpbin.exe failed on $Path" }
    return @($report | ForEach-Object { $_.Trim().ToLowerInvariant() } |
        Where-Object { $_ -match "^(msvcp|vcruntime|concrt|vcomp|vccorlib)[0-9a-z_]*\.dll$" })
}

$queue = [System.Collections.Generic.Queue[string]]::new()
foreach ($library in @("onnxruntime.dll", "onnxruntime_providers_shared.dll")) {
    $queue.Enqueue((Join-Path $directoryPath $library))
}
$needed = [System.Collections.Generic.SortedSet[string]]::new()
while ($queue.Count -gt 0) {
    foreach ($import in (Get-VcImports -Path $queue.Dequeue())) {
        if ($needed.Add($import)) {
            $source = Join-Path $redist.FullName $import
            if (-not (Test-Path -LiteralPath $source)) { throw "$import is not in $($redist.FullName)" }
            Copy-Item -LiteralPath $source -Destination $directoryPath -Force
            $queue.Enqueue((Join-Path $directoryPath $import))
        }
    }
}

$staged = @($needed)
Write-Output "staged: $($staged -join ', ')"
$difference = Compare-Object -ReferenceObject $expected -DifferenceObject $staged
if ($null -ne $difference) {
    throw "ONNX Runtime needs $($staged -join ', '), and the packages ship $($expected -join ', '): change Product.wxs and this list together"
}
