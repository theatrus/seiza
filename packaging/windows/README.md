# Windows installer

The WiX package installs as **Seiza CLI** — the name shown in Apps & Features,
the installer UI, and the Start menu. The name is distinct from the **Seiza for
Windows** desktop app (the seiza-win repository) so both products can be
installed side by side and told apart. The binary itself stays `seiza.exe`.

Alongside it the package installs `astap.exe`, a duplicate of the same file.
N.I.N.A. asks for a plate-solver path naming a file called `astap.exe`, and
seiza takes ASTAP-compatible mode when run under that name, so pointing
N.I.N.A. at the install directory needs no copying or renaming. WiX makes the
copy at install time (`CopyFile`), so the MSI carries one binary, the copy
keeps the signature and version resource of the original, and uninstall
removes both.

The package presents the Apache 2.0 license and lets the user choose between
an all-users install (the default) and a current-user install. The all-users
choice installs under 64-bit Program Files and causes Windows Installer to
request administrator approval through UAC.

The feature-selection page includes **Add Seiza CLI to PATH**, selected by default.
For a current-user install it updates the user's `PATH`; for an all-users install
it updates the system `PATH`. The final page can launch `seiza setup` to guide the
user through catalog selection and downloading. That work remains entirely in
the CLI; the MSI contains no catalog URLs or download custom actions.

For an all-users install, the MSI creates the shared
`%ProgramData%\Seiza\catalogs` directory, grants local users write access, and
sets the system `SEIZA_CATALOG_DIR` environment variable. This directory stays
under the plain `Seiza` name on purpose: the Seiza for Windows app resolves the
same variable, so both products share one catalog store. The final-page setup
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
deep blind solving, the optional G≤20 deep catalog (about 9 GB), or every
published catalog.

The welcome, completion, and banner artwork in `assets/` uses Seiza-specific
constellation and astrometry imagery instead of the stock WiX graphics.

The project pins WiX 4, which provides the required MSI and WixUI features
without requiring CI to accept the maintenance-fee EULA introduced by newer
WiX releases.

Build the release binary and MSI from PowerShell:

```powershell
cargo build --release -p seiza-cli
dotnet build packaging/windows/seiza.wixproj -c Release -p:SeizaVersion=0.8.0
```

The MSI is written to `dist/`. A silent current-user install with the default
PATH feature can be performed with:

```powershell
msiexec /i dist/seiza-cli-0.8.0-windows-x86_64.msi ALLUSERS=2 MSIINSTALLPERUSER=1 ADDLOCAL=MainFeature,PathFeature /qn /norestart
```

An elevated all-users install uses `ALLUSERS=1`:

```powershell
msiexec /i dist/seiza-cli-0.8.0-windows-x86_64.msi ALLUSERS=1 ADDLOCAL=MainFeature,PathFeature /qn /norestart
```

To omit the PATH change, install only the required feature:

```powershell
msiexec /i dist/seiza-cli-0.8.0-windows-x86_64.msi ALLUSERS=2 MSIINSTALLPERUSER=1 ADDLOCAL=MainFeature REMOVE=PathFeature /qn /norestart
```

Silent installs never launch catalog setup.
