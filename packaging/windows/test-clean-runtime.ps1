# Run a tiny model from a staged install in a Windows container, which unlike a
# GitHub runner has no Visual Studio and so no Visual C++ runtime of its own.
#
# It runs twice. Without the runtime's DLLs the library must fail to load: if it
# loads, the container is not clean and the second run would prove nothing.
param(
    [Parameter(Mandatory = $true)]
    [string]$Stage,

    [string]$Image = "mcr.microsoft.com/windows/servercore:ltsc2025"
)

$ErrorActionPreference = "Stop"
$stagePath = (Resolve-Path -LiteralPath $Stage).Path
$fixtures = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\seiza-model\tests\fixtures")).Path
$vcRuntime = @("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll")

function Invoke-InContainer {
    param([string]$Directory, [string]$Command)
    & docker run --rm -v "${Directory}:C:\seiza" -v "${fixtures}:C:\fixtures" $Image cmd /c $Command 2>&1 |
        Out-Host
    return $LASTEXITCODE
}

& docker pull $Image
if ($LASTEXITCODE -ne 0) { throw "docker pull failed" }

$bare = Join-Path ([System.IO.Path]::GetTempPath()) "seiza-without-vc-runtime"
Remove-Item -Recurse -Force $bare -ErrorAction SilentlyContinue
New-Item -ItemType Directory $bare | Out-Null
Copy-Item -Path (Join-Path $stagePath "*") -Destination $bare
foreach ($file in $vcRuntime) { Remove-Item -LiteralPath (Join-Path $bare $file) }

Write-Output "--- without the Visual C++ runtime, which must fail"
$withoutCode = Invoke-InContainer -Directory $bare -Command "C:\seiza\seiza.exe model backends"
if ($withoutCode -eq 0) {
    throw "the library loaded without the Visual C++ runtime, so this container proves nothing"
}

Write-Output "--- with it"
$withCode = Invoke-InContainer -Directory $stagePath -Command "C:\seiza\seiza.exe model backends"
if ($withCode -ne 0) { throw "seiza model backends failed with exit code $withCode" }
$run = "C:\seiza\seiza.exe model denoise C:\fixtures\tiny-input.fits -o C:\Windows\Temp\clean.fits --model C:\fixtures\tiny --backend cpu"
$runCode = Invoke-InContainer -Directory $stagePath -Command $run
if ($runCode -ne 0) { throw "seiza model denoise failed with exit code $runCode" }
Write-Output "OK: a tiny model ran with only what the package ships"
