# Windows installer

The WiX package installs `seiza.exe` per-user, adds it to the user's `PATH`,
and offers to launch `seiza setup` from the final installer page. Catalog
selection and downloading remain entirely in the CLI; the MSI contains no
catalog URLs or download custom actions.

The project pins WiX 4, which provides the required MSI and WixUI features
without requiring CI to accept the maintenance-fee EULA introduced by newer
WiX releases.

Build the release binary and MSI from PowerShell:

```powershell
cargo build --release -p seiza-cli
dotnet build packaging/windows/seiza.wixproj -c Release -p:SeizaVersion=0.5.0
```

The MSI is written to `dist/`. A silent install never launches catalog setup:

```powershell
msiexec /i dist/seiza-cli-0.5.0-windows-x86_64.msi /qn /norestart
```
