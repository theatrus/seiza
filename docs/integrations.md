# Integrations

[Documentation](README.md) · [Project overview](../README.md)

## N.I.N.A. (ASTAP-compatible mode)

Configure N.I.N.A. to run Seiza through its ASTAP integration:

1. Install the Windows MSI from [Seiza releases](https://github.com/theatrus/seiza/releases),
   or extract the portable ZIP. Both include `seiza.exe` and `astap.exe`.
2. Open **Seiza Catalog Setup** from the Start menu, or run `seiza setup`.
   Choose a blind-solving preset if you also want to use Seiza as the blind solver.
3. In N.I.N.A., open **Options → Plate Solving**, select **Plate Solver: ASTAP**,
   and set the ASTAP path to `astap.exe`. For an all-users MSI installation,
   the path is `C:\Program Files\Seiza CLI\astap.exe`.
4. To use Seiza for blind solves, select the same executable in N.I.N.A.'s
   blind-solver settings.

For a portable catalog directory, download the data and set its location:

```bat
seiza download-data prebuilt --output C:\seiza-data
setx SEIZA_CATALOG_DIR C:\seiza-data
```

Restart N.I.N.A. after setting the environment variable so it uses the new value.

Seiza recognizes ASTAP arguments such as `-f`, `-fov`, `-ra`, and `-spd`.
It writes the `.ini` result N.I.N.A. reads, including the CD matrix for scale,
rotation, and image flip. An executable named `astap.exe` also starts in this
mode without those flags. Catalog discovery uses `SEIZA_CATALOG_DIR`;
`SEIZA_STAR_DATA` and `SEIZA_BLIND_INDEX` can override individual files.
See the [ASTAP compatibility specification](design/astap-mode.md).

## Siril (solve-field compatible mode)

Configure Siril to run Seiza through its astrometry.net integration:

1. Run `seiza install-solve-field --dir <dir>`. Seiza creates the `solve-field`,
   `bin/bash`, and `tmp/` layout Siril requires. On Windows, Seiza also supports
   Siril's `bin/bash` wrapper, so Cygwin is unnecessary.
2. Run `seiza setup` and choose a blind-solving preset. For a portable catalog
   directory, run `seiza download-data prebuilt --output <directory>` and set
   `SEIZA_CATALOG_DIR` to that directory before starting Siril.
3. In Siril, open **Preferences → Astrometry → astrometry.net install dir**
   and select the directory created in step 1.
4. Choose the astrometry.net solver when plate solving.

Siril supplies its detected star table and reads the solution from the `.wcs`
file. Seiza fits the SIP order Siril requests. When the source image is beside
the star table, Seiza remeasures stellar flux from the pixels for brightness
ranking and uses the table's coordinate system. See the
[solve-field compatibility specification](design/solve-field-mode.md).

## Application APIs

- [Python API](../seiza-py/README.md): install with `pip install seiza`; use
  NumPy arrays for solving, calibration, processing, stacking, and parallax.
- [C ABI](../seiza-cabi/README.md): call the core algorithms from Swift, .NET,
  C, or C++ through the generated header.
- [Rust crates](README.md#rust-crates): choose the crate for the operation you
  need. Use `seiza` for solving, detection, and catalog access.
- [Worker JSON protocol](design/worker-protocol.md): send solving requests over
  stdin/stdout, locally or through a remote server. See
  [worker examples](solving.md#persistent-and-remote-workers).
- [seiza-server](https://github.com/theatrus/seiza-server): submit queued solves
  through a web interface or astrometry.net-compatible HTTP API. The hosted
  instance is [seiza.fyi](https://seiza.fyi).
