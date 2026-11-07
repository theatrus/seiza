# Run a tiny model from a staged install in a Windows container, which unlike a
# GitHub runner has no Visual Studio and so no Visual C++ runtime of its own.
# The full server image, since Server Core has no opengl32.dll and seiza.exe
# does not start there at all.
#
# Each step guards the next. seiza.exe must start, or nothing here says anything
# about the runtime. Without the Visual C++ DLLs ONNX Runtime must then fail to
# load, and say so: if it loads, or the failure is something else, the container
# is not clean and the run with them would prove nothing.
param(
    [Parameter(Mandatory = $true)]
    [string]$Stage,

    [string]$Image = "mcr.microsoft.com/windows/server:ltsc2025"
)

$ErrorActionPreference = "Stop"
$stagePath = (Resolve-Path -LiteralPath $Stage).Path
$fixtures = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\seiza-model\tests\fixtures")).Path
$vcRuntime = @("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll")

function Invoke-InContainer {
    param([string]$Directory, [string]$Command)
    $output = & docker run --rm -v "${Directory}:C:\seiza" -v "${fixtures}:C:\fixtures" $Image cmd /c $Command 2>&1 |
        Out-String
    $code = $LASTEXITCODE
    Write-Host $output
    Write-Host "exit code $code"
    return [pscustomobject]@{ Code = $code; Output = $output }
}

& docker pull $Image
if ($LASTEXITCODE -ne 0) { throw "docker pull failed" }

$bare = Join-Path ([System.IO.Path]::GetTempPath()) "seiza-without-vc-runtime"
Remove-Item -Recurse -Force $bare -ErrorAction SilentlyContinue
New-Item -ItemType Directory $bare | Out-Null
Copy-Item -Path (Join-Path $stagePath "*") -Destination $bare
foreach ($file in $vcRuntime) { Remove-Item -LiteralPath (Join-Path $bare $file) }

Write-Host "--- seiza.exe starts here, without the Visual C++ runtime"
$start = Invoke-InContainer -Directory $bare -Command "C:\seiza\seiza.exe --version"
if ($start.Code -ne 0) {
    throw "seiza.exe does not start in this container (exit code $($start.Code)), so it can show nothing about the runtime"
}

Write-Host "--- ONNX Runtime does not load without it"
$without = Invoke-InContainer -Directory $bare -Command "C:\seiza\seiza.exe model backends"
if ($without.Code -eq 0) {
    throw "the library loaded without the Visual C++ runtime, so this container proves nothing"
}
if ($without.Output -notmatch "ONNX Runtime is not available") {
    throw "seiza failed without the Visual C++ runtime, but not by failing to load the library"
}

Write-Host "--- it loads with it, and runs a model"
$with = Invoke-InContainer -Directory $stagePath -Command "C:\seiza\seiza.exe model backends"
if ($with.Code -ne 0) { throw "seiza model backends failed with exit code $($with.Code)" }
if ($with.Output -notmatch [regex]::Escape("C:\seiza\onnxruntime.dll")) {
    throw "seiza did not load the library beside it"
}
$command = "C:\seiza\seiza.exe model denoise C:\fixtures\tiny-input.fits -o C:\Windows\Temp\clean.fits --model C:\fixtures\tiny --backend cpu"
$run = Invoke-InContainer -Directory $stagePath -Command $command
if ($run.Code -ne 0) { throw "seiza model denoise failed with exit code $($run.Code)" }
Write-Host "OK: a tiny model ran with only what the package ships"
