# Windows installer

The WiX package presents the Apache 2.0 license and lets the user choose between
an all-users install (the default) and a current-user install. The all-users
choice installs under 64-bit Program Files and causes Windows Installer to
request administrator approval through UAC.

The feature-selection page includes **Add Seiza to PATH**, selected by default.
For a current-user install it updates the user's `PATH`; for an all-users install
it updates the system `PATH`. The final page can launch `seiza setup` to guide the
user through catalog selection and downloading. That work remains entirely in
the CLI; the MSI contains no catalog URLs or download custom actions.

For an all-users install, the MSI creates the shared
`%ProgramData%\Seiza\catalogs` directory, grants local users write access, and
sets the system `SEIZA_CATALOG_DIR` environment variable. The final-page setup
wizard downloads directly to that shared directory. A current-user install
continues to use `%LOCALAPPDATA%\Seiza\seiza\data\catalogs`. Explicit
`SEIZA_STAR_DATA` and `SEIZA_BLIND_INDEX` environment variables remain
higher-priority file overrides.

The all-users catalog wizard explicitly relaunches itself with the Windows
`runas` verb, so Windows requests administrator approval before downloading to
the shared directory. Installer-launched setup also keeps its console open on
failure, prints the complete error chain, and waits for Enter so download or
filesystem errors cannot disappear with the window. The Start menu shortcut
uses the same behavior and can be used to retry setup later.

Every setup-wizard choice includes the object catalog, Solar System objects,
active transients, and at least one usable plate-solving catalog. The menu
describes choices by use case: lightweight hinted solving, denser Gaia solving,
deep blind solving, or the complete bundle.

The welcome, completion, and banner artwork in `assets/` uses Seiza-specific
constellation and astrometry imagery instead of the stock WiX graphics.

The project pins WiX 4, which provides the required MSI and WixUI features
without requiring CI to accept the maintenance-fee EULA introduced by newer
WiX releases.

Build the release binary and MSI from PowerShell:

```powershell
cargo build --release -p seiza-cli
dotnet build packaging/windows/seiza.wixproj -c Release -p:SeizaVersion=0.6.1
```

The MSI is written to `dist/`. A silent current-user install with the default
PATH feature can be performed with:

```powershell
msiexec /i dist/seiza-cli-0.6.1-windows-x86_64.msi ALLUSERS=2 MSIINSTALLPERUSER=1 ADDLOCAL=MainFeature,PathFeature /qn /norestart
```

An elevated all-users install uses `ALLUSERS=1`:

```powershell
msiexec /i dist/seiza-cli-0.6.1-windows-x86_64.msi ALLUSERS=1 ADDLOCAL=MainFeature,PathFeature /qn /norestart
```

To omit the PATH change, install only the required feature:

```powershell
msiexec /i dist/seiza-cli-0.6.1-windows-x86_64.msi ALLUSERS=2 MSIINSTALLPERUSER=1 ADDLOCAL=MainFeature REMOVE=PathFeature /qn /norestart
```

Silent installs never launch catalog setup.
