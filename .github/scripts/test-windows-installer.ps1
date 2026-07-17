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
$machineCatalogDirectory = Join-Path $env:ProgramData "Seiza\catalogs"
$programMenuDirectory = if ($Scope -eq "perMachine") {
    Join-Path $env:ProgramData "Microsoft\Windows\Start Menu\Programs\Seiza"
}
else {
    Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Seiza"
}
$catalogSetupShortcut = Join-Path $programMenuDirectory "Seiza Catalog Setup.lnk"
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

function Get-OptionalRegistryValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LiteralPath,

        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $registryItem = Get-ItemProperty -LiteralPath $LiteralPath -ErrorAction Stop
    $property = $registryItem.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }

    return $property.Value
}

function Get-InstallerPathEntries {
    $pathValue = Get-OptionalRegistryValue -LiteralPath $pathRegistry -Name Path
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
    if (-not (Test-Path -LiteralPath $catalogSetupShortcut -PathType Leaf)) {
        throw "Catalog setup shortcut not found at $catalogSetupShortcut"
    }

    $shortcutArguments = (New-Object -ComObject WScript.Shell).CreateShortcut($catalogSetupShortcut).Arguments
    if ($shortcutArguments -notmatch "(?:^|\s)--from-installer(?:\s|$)") {
        throw "Catalog setup shortcut does not enable durable installer-mode errors"
    }
    if ($Scope -eq "perMachine") {
        if ($shortcutArguments -notmatch "(?:^|\s)--elevate(?:\s|$)") {
            throw "All-users catalog setup shortcut does not request elevation"
        }
        if ($shortcutArguments -notlike "*$($machineCatalogDirectory.TrimEnd('\'))*") {
            throw "All-users catalog setup shortcut does not target $machineCatalogDirectory"
        }
    }
    elseif ($shortcutArguments -match "(?:^|\s)--elevate(?:\s|$)") {
        throw "Current-user catalog setup shortcut unexpectedly requests elevation"
    }

    $pathEntries = Get-InstallerPathEntries
    $installedPathEntry = $installDirectory.TrimEnd("\")
    if ($WithoutPath -and $pathEntries -contains $installedPathEntry) {
        throw "PATH unexpectedly contains $installDirectory"
    }
    if (-not $WithoutPath -and $pathEntries -notcontains $installedPathEntry) {
        throw "PATH does not contain $installDirectory"
    }

    if ($Scope -eq "perMachine") {
        if (-not (Test-Path -LiteralPath $machineCatalogDirectory -PathType Container)) {
            throw "Shared catalog directory not found at $machineCatalogDirectory"
        }

        $catalogDirectoryValue = Get-OptionalRegistryValue `
            -LiteralPath $pathRegistry `
            -Name "SEIZA_CATALOG_DIR"
        if (-not $catalogDirectoryValue -or $catalogDirectoryValue.TrimEnd("\") -ne $machineCatalogDirectory.TrimEnd("\")) {
            throw "SEIZA_CATALOG_DIR is not set to $machineCatalogDirectory"
        }

        $usersSid = "S-1-5-32-545"
        $requiredRights = [System.Security.AccessControl.FileSystemRights]::Modify
        $usersCanModify = (Get-Acl -LiteralPath $machineCatalogDirectory).Access | Where-Object {
            try {
                $sid = $_.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value
                $sid -eq $usersSid -and ($_.FileSystemRights -band $requiredRights) -eq $requiredRights
            }
            catch {
                $false
            }
        }
        if (-not $usersCanModify) {
            throw "Built-in Users group does not have modify access to $machineCatalogDirectory"
        }
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
        if (Test-Path -LiteralPath $catalogSetupShortcut) {
            throw "MSI uninstall left $catalogSetupShortcut behind"
        }
        $pathEntries = Get-InstallerPathEntries
        if ($pathEntries -contains $installDirectory.TrimEnd("\")) {
            throw "MSI uninstall left $installDirectory in PATH"
        }
        if ($Scope -eq "perMachine") {
            $catalogDirectoryValue = Get-OptionalRegistryValue `
                -LiteralPath $pathRegistry `
                -Name "SEIZA_CATALOG_DIR"
            if ($catalogDirectoryValue) {
                throw "MSI uninstall left SEIZA_CATALOG_DIR configured"
            }
        }
    }
}

Write-Output "MSI smoke test passed ($Scope, PATH installed: $(-not $WithoutPath))"
