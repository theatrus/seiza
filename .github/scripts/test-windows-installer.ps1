param(
    [Parameter(Mandatory = $true)]
    [string]$Msi
)

$ErrorActionPreference = "Stop"

$Msi = (Resolve-Path -LiteralPath $Msi).Path
$tempDir = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { $env:TEMP }
$log = Join-Path $tempDir "seiza-msi-install.log"
$installedBinary = Join-Path $env:LOCALAPPDATA "Programs\Seiza\seiza.exe"
$installed = $false

try {
    $install = Start-Process msiexec.exe -ArgumentList "/i", "`"$Msi`"", "/qn", "/norestart", "/l*v", "`"$log`"" -Wait -PassThru
    if ($install.ExitCode -ne 0) {
        if (Test-Path -LiteralPath $log) {
            Get-Content -LiteralPath $log
        }
        throw "MSI install failed with exit code $($install.ExitCode)"
    }
    $installed = $true

    if (-not (Test-Path -LiteralPath $installedBinary)) {
        throw "Installed binary not found at $installedBinary"
    }

    & $installedBinary --version
    if ($LASTEXITCODE -ne 0) {
        throw "Installed seiza --version failed with exit code $LASTEXITCODE"
    }

    & $installedBinary setup --help | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Installed seiza setup --help failed with exit code $LASTEXITCODE"
    }
}
finally {
    if ($installed) {
        $uninstall = Start-Process msiexec.exe -ArgumentList "/x", "`"$Msi`"", "/qn", "/norestart" -Wait -PassThru
        if ($uninstall.ExitCode -ne 0) {
            throw "MSI uninstall failed with exit code $($uninstall.ExitCode)"
        }
        if (Test-Path -LiteralPath $installedBinary) {
            throw "MSI uninstall left $installedBinary behind"
        }
    }
}

Write-Output "MSI smoke test passed"
