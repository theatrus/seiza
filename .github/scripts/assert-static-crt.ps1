param(
    [Parameter(Mandatory = $true)]
    [string]$Binary
)

$ErrorActionPreference = 'Stop'
$binaryPath = (Resolve-Path -LiteralPath $Binary).Path
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'

if (-not (Test-Path -LiteralPath $vswhere)) {
    throw "vswhere.exe was not found at $vswhere"
}

$installationPath = & $vswhere `
    -latest `
    -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath |
    Select-Object -First 1

if ([string]::IsNullOrWhiteSpace($installationPath)) {
    throw 'A Visual Studio installation with the x64 C++ tools was not found'
}

$dumpbin = Get-ChildItem `
    -Path (Join-Path $installationPath 'VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe') `
    -File |
    Sort-Object FullName -Descending |
    Select-Object -First 1

if ($null -eq $dumpbin) {
    throw "dumpbin.exe was not found under $installationPath"
}

$imports = @(& $dumpbin.FullName /DEPENDENTS $binaryPath)
if ($LASTEXITCODE -ne 0) {
    throw "dumpbin.exe failed with exit code $LASTEXITCODE"
}

$imports | Write-Output
$runtimePattern = '(?i)\b(?:VCRUNTIME[\d_]*|MSVCP[\d_]*|MSVCR[\d_]*|CONCRT[\d_]*|UCRTBASE|api-ms-win-crt-[A-Za-z0-9-]+)\.dll\b'
$runtimeImports = @($imports | Select-String -Pattern $runtimePattern)

if ($runtimeImports.Count -ne 0) {
    $names = ($runtimeImports.Matches.Value | Sort-Object -Unique) -join ', '
    throw "Windows runtime DLL imports found: $names"
}

Write-Output 'No dynamically linked Visual C++ or Universal CRT imports found.'
