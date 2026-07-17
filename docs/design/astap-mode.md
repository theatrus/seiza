# ASTAP-compatible mode: seiza as a N.I.N.A. plate solver

Status: implemented (`seiza` auto-detects ASTAP-style invocations; a copy
renamed `astap.exe` behaves identically)

## Goal

Let N.I.N.A. (and anything else that shells out to ASTAP) use seiza as
its plate solver with zero changes on their side: the user points the
"ASTAP path" at the seiza binary and everything — hinted solves, blind
failover, centering loops — works.

## Why impersonation instead of a plugin

Verified against the N.I.N.A. source (isbeorn/nina@develop, MPL-2.0):

- There is **no pluggable plate-solver interface**.
  `PlateSolverFactory.cs` is a hardcoded switch over a fixed enum of
  solvers, with ASTAP as the default.
- The plugin SDK's extension point for swapping core behavior,
  `IPluggableBehavior`, covers only `IStarDetection`, `IStarAnnotator`,
  and `IAutoFocusVMFactory`. A plugin can *consume* the built-in
  solvers but cannot register a new one.
- External solvers are driven through `CLISolver`, which runs an
  executable, ignores its exit code, and judges success **solely by a
  result file**. ASTAP's contract is the simplest of the bunch: a flat
  `key=value` `.ini` file.

Impersonating ASTAP therefore gets full integration (imaging, plate
solve tool, slew-and-center, meridian flip recovery) with no plugin
submission and no licensing entanglement.

## The contract

### Invocation

N.I.N.A. always saves the frame as a temporary **FITS** file and runs:

```
seiza[.exe] -f "<image.fits>" -fov <deg> -z <downsample> -s <maxstars> [-r <deg> -ra <hours> -spd <deg>]
```

| Flag | Meaning |
|------|---------|
| `-f` | input FITS path |
| `-fov` | field of view of the image **height**, degrees |
| `-z` | downsample factor (0 = auto) |
| `-s` | max stars to use |
| `-r` | search radius, degrees (`180` in blind mode) |
| `-ra` | hint RA in **hours** (absent in blind mode) |
| `-spd` | hint as **south polar distance**: Dec + 90, degrees |

Mapping to seiza:

- `-ra`/`-spd` present → hinted solve with
  `ra_deg = ra × 15`, `dec = spd − 90`, radius `-r`,
  scale derived from `-fov`: `scale ≈ fov × 3600 / image_height` px.
- Absent (or `-r 180`) → blind solve with a scale window around the
  `-fov`-derived value.
- Unknown flags must be ignored (forward compatibility with whatever
  N.I.N.A. adds).
- Timeout budget: N.I.N.A. kills the process after 10 minutes; stay
  far under it (we do).

Seiza also accepts its `--detection-backend auto|u8|f32`,
`--detection-fallback none|f32`, and
`--detection-fallback-hypotheses N` extensions in this mode. Auto uses compact
u8 detection first and defaults to a lazy f32 retry after a solve miss. Blind
u8 verification is capped at 64 hypotheses when that retry is available; the
f32 retry receives the full configured budget. Set the cap to zero to give u8
the full budget. For FITS, an f32 retry reopens the source and uses linear
high-precision samples rather than applying the u8 MTF display stretch.

### Result file

Write `<image-basename>.ini` **next to the input FITS** (same stem).
N.I.N.A. parses it as flat `key=value` lines:

```
PLTSOLVD=T
CRVAL1=<center RA, degrees, J2000>
CRVAL2=<center Dec, degrees>
CRPIX1=<reference pixel x (image center)>
CRPIX2=<reference pixel y>
CD1_1=<deg/px>  CD1_2=  CD2_1=  CD2_2=
```

- `PLTSOLVD` must be exactly `T`; anything else is failure. On
  failure write `PLTSOLVD=F` plus optional `ERROR=` / `WARNING=`
  (surfaced verbatim in the N.I.N.A. UI).
- N.I.N.A. derives pixel scale from `sqrt(CD1_2² + CD2_2²)` and the
  position angle *and flip* from the full CD matrix, so parity must be
  encoded ASTAP's way:

```
CDELT1 = −s   (mirror the sign for a flipped field)
CDELT2 = +s
CD1_1 =  CDELT1·cos θ    CD1_2 = −CDELT2·sin θ
CD2_1 =  CDELT1·sin θ    CD2_2 =  CDELT2·cos θ
```

seiza's TAN WCS already carries the full CD matrix including parity;
this is a straight re-expression, but sign errors here are the most
likely bug — validate against real ASTAP output on the same frame.
- The `.wcs` sidecar ASTAP writes is optional; N.I.N.A. reads only the
  `.ini`.

### Windows binary version gotcha

`EnsureSolverValid` reads the executable's Windows *FileVersion*
resource. If missing, N.I.N.A. assumes "ASTAP older than 0.9.1.0" and
rejects `-z 0` (auto-downsample). The Windows build must stamp a PE
FileVersion ≥ `0.9.1.0` (the `winres`/`winresource` crate in
`build.rs`, Windows targets only).

## Star catalog resolution

N.I.N.A. passes no catalog argument, so ASTAP mode resolves its star
data on its own, in order:

1. `SEIZA_STAR_DATA` environment variable;
2. a `seiza.toml` next to the executable (`star_data = ...`);
3. well-known per-OS data directories
   (`%LOCALAPPDATA%/seiza`, `~/.local/share/seiza`);
4. if nothing is found: fail fast with
   `ERROR=no star catalog; run: seiza download-data prebuilt --output <dir>`
   in the `.ini` — pointing at the hosted datasets
   (downloads.seiza.fyi) rather than attempting a surprise 400 MB
   download inside an imaging loop.

The Gaia G≤15 set (367 MB) is the recommended general-purpose default;
the Gaia G≤17 set plus the maintained `blind-gaia16.idx` is recommended
for blind solving small, fine-scale fields. Tycho-2 lite (25 MB) suffices
for wide fields.

The optional blind index is resolved independently in the same locations,
using `SEIZA_BLIND_INDEX`, `blind_index = ...` in `seiza.toml`, or the
well-known filename `blind-gaia16.idx`. It is memory-mapped and reused as-is.
If it is absent, the runtime-build fallback covers only the bright
(G≤12.7) tiers regardless of catalog depth — building the deep tiers over
a 154M-star catalog takes minutes and gigabytes, which inside an imaging
loop reads as a hang. Small fine-scale fields need the prebuilt index.

## Mode detection

Two entry styles, same code path:

- `seiza astap -f ... -fov ...` — explicit subcommand;
- argv auto-detection: if `argv[1]` starts with `-f`/`-ra`/`-fov`,
  treat the whole command line as ASTAP-style. This lets a copy (or
  symlink/hardlink) named `astap.exe` behave correctly when N.I.N.A.
  or other tools invoke it blind.

## N.I.N.A. behaviors we inherit for free

- **Hints**: RA/Dec come from the mount/profile; FOV from profile
  focal length + pixel size. If focal length is unset N.I.N.A. throws
  before invoking any solver — not ours to fix.
- **Blind failover**: `ImageSolver.Solve` re-runs the configured blind
  solver with null coordinates when the hinted solve fails. ASTAP is
  selectable in both slots, so one seiza binary serves both.
- **Retries, slews, centering**: handled above the solver
  (`CaptureSolver` / `CenteringSolver`); failed frames get copied to
  `%LOCALAPPDATA%\NINA\PlateSolver\Failed\` for diagnosis.

## Validation plan

1. Golden-file tests for the `.ini` writer (success, failure, parity
   flip, rotation quadrants).
2. Side-by-side with real ASTAP on the hosted integration-suite
   frames: center within arcseconds, scale within 0.5 %, position
   angle within 0.5°, flip identical.
3. A live N.I.N.A. run (Windows, task #40): Options → Plate Solving →
   solver ASTAP → path seiza.exe; verify imaging-tab solve, blind
   failover, and slew-and-center.

## Out of scope (for now)

- The N.I.N.A. plugin marketplace (nothing to submit — no plugin).
- SIP distortion terms in the result (ASTAP's `.ini` has no field for
  them; N.I.N.A. doesn't consume them).
- PlateSolve2/3 and astrometry.net CLI contracts — possible later
  shims behind the same core if other suites matter.
