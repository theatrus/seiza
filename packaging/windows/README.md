# Windows installer

The WiX package presents the Apache 2.0 license and lets the user choose between
a current-user install (the default) and an all-users install. The all-users
choice installs under 64-bit Program Files and causes Windows Installer to
request administrator approval through UAC.

The feature-selection page includes **Add Seiza to PATH**, selected by default.
For a current-user install it updates the user's `PATH`; for an all-users install
it updates the system `PATH`. The final page can launch `seiza setup` to guide the
user through catalog selection and downloading. That work remains entirely in
the CLI; the MSI contains no catalog URLs or download custom actions.

The project pins WiX 4, which provides the required MSI and WixUI features
without requiring CI to accept the maintenance-fee EULA introduced by newer
WiX releases.

Build the release binary and MSI from PowerShell:

```powershell
cargo build --release -p seiza-cli
dotnet build packaging/windows/seiza.wixproj -c Release -p:SeizaVersion=0.5.0
```

The MSI is written to `dist/`. A silent current-user install with the default
PATH feature can be performed with:

```powershell
msiexec /i dist/seiza-cli-0.5.0-windows-x86_64.msi ALLUSERS=2 MSIINSTALLPERUSER=1 ADDLOCAL=MainFeature,PathFeature /qn /norestart
```

An elevated all-users install uses `ALLUSERS=1`:

```powershell
msiexec /i dist/seiza-cli-0.5.0-windows-x86_64.msi ALLUSERS=1 ADDLOCAL=MainFeature,PathFeature /qn /norestart
```

To omit the PATH change, install only the required feature:

```powershell
msiexec /i dist/seiza-cli-0.5.0-windows-x86_64.msi ALLUSERS=2 MSIINSTALLPERUSER=1 ADDLOCAL=MainFeature REMOVE=PathFeature /qn /norestart
```

Silent installs never launch catalog setup.
