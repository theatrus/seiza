param(
    [Parameter(Mandatory = $true)]
    [string]$Msi,

    [ValidateSet("perUser", "perMachine")]
    [string]$Scope = "perUser",

    [switch]$WithoutPath
)

$ErrorActionPreference = "Stop"

$Msi = (Resolve-Path -LiteralPath $Msi).Path
$tempDir = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { $env:TEMP }
$log = Join-Path $tempDir "seiza-msi-$Scope-install.log"
$installDirectory = if ($Scope -eq "perMachine") {
    Join-Path $env:ProgramFiles "Seiza"
}
else {
    Join-Path $env:LOCALAPPDATA "Apps\Seiza"
}
$installedBinary = Join-Path $installDirectory "seiza.exe"
$pathRegistry = if ($Scope -eq "perMachine") {
    "Registry::HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
}
else {
    "Registry::HKEY_CURRENT_USER\Environment"
}
$scopeProperties = if ($Scope -eq "perMachine") {
    @("ALLUSERS=1")
}
else {
    @("ALLUSERS=2", "MSIINSTALLPERUSER=1")
}
$featureProperties = if ($WithoutPath) {
    @("ADDLOCAL=MainFeature", "REMOVE=PathFeature")
}
else {
    @("ADDLOCAL=MainFeature,PathFeature")
}
$installArguments = @(
    "/i",
    "`"$Msi`"",
    "/qn",
    "/norestart",
    "APPLICATIONFOLDER=`"$installDirectory`"",
    "/l*v",
    "`"$log`""
) + $scopeProperties + $featureProperties
$installed = $false

function Get-InstallerPathEntries {
    $pathValue = Get-ItemPropertyValue -LiteralPath $pathRegistry -Name Path -ErrorAction SilentlyContinue
    if (-not $pathValue) {
        return @()
    }

    return @($pathValue -split ";" | ForEach-Object { $_.TrimEnd("\") })
}

try {
    $install = Start-Process msiexec.exe -ArgumentList $installArguments -Wait -PassThru
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

    $pathEntries = Get-InstallerPathEntries
    $installedPathEntry = $installDirectory.TrimEnd("\")
    if ($WithoutPath -and $pathEntries -contains $installedPathEntry) {
        throw "PATH unexpectedly contains $installDirectory"
    }
    if (-not $WithoutPath -and $pathEntries -notcontains $installedPathEntry) {
        throw "PATH does not contain $installDirectory"
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
        $pathEntries = Get-InstallerPathEntries
        if ($pathEntries -contains $installDirectory.TrimEnd("\")) {
            throw "MSI uninstall left $installDirectory in PATH"
        }
    }
}

Write-Output "MSI smoke test passed ($Scope, PATH installed: $(-not $WithoutPath))"
