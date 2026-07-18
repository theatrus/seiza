# solve-field compatible mode: seiza as Siril's plate solver

Status: implemented (a copy of the binary named `solve-field` behaves as a
local astrometry.net; Siril's astrometry.net integration drives it unchanged)

## Goal

Let Siril (and anything else that shells out to a local astrometry.net
`solve-field`) use seiza with zero changes on their side: the user points
Siril's astrometry.net path preference at a directory containing seiza named
`solve-field`. This mirrors the ASTAP-compatible mode used by N.I.N.A.

## The contract

Siril's `local_asnet_platesolve` (src/algos/astrometry_solver.c):

1. **Version handshake**: runs `solve-field --version` and requires a
   *single line* of output; multi-line output is treated as a pre-0.88
   astrometry.net. Seiza prints `0.94-seiza-<version>`. The string is only
   echoed into logs and the `PLTSOLVD` FITS comment.
2. **Input**: Siril detects stars itself and writes them to
   `<image>.xyls`, a FITS binary table with float32 `X`/`Y`/`FLUX`/
   `BACKGROUND` columns (FITS 1-based pixel convention) and
   `IMAGEW`/`IMAGEH` keywords. No pixels are exchanged.
3. **Arguments**: `-C <stopfile>` (abort when the file appears),
   `--temp-axy -p -O -N none -R none -M none -B none -U none -S none
   --crpix-center -s FLUX`, optionally `-u arcsecperpix -L <low> -H <high>`
   (scale window), `-l <seconds>`, `-t <order>` (SIP) or `-T` (linear), and
   optionally `--ra <deg> --dec <deg> --radius <deg>`, then the table path.
4. **Output**: stdout line prefixed `Field center: (RA,Dec)` signals
   success; a line prefixed `Did not solve` signals failure. On success the
   solution is read from `<image>.wcs`, a header-only FITS file whose
   keywords wcslib parses — including SIP `A_*`/`B_*`/`AP_*`/`BP_*` terms.
   Exit codes are ignored.

## Seiza's behavior

- A position hint plus a scale window runs the hinted solver (radius
  clamped to a practical window) with blind fallback; otherwise the blind
  solver runs over the requested or default scale range, using a prebuilt
  index when one is resolvable.
- Star catalogs resolve exactly as in ASTAP mode: `SEIZA_STAR_DATA` /
  `SEIZA_BLIND_INDEX`, `seiza.toml` or a `stars-*.bin` next to the binary,
  then the shared catalog directories (`SEIZA_CATALOG_DIR`, `seiza setup`
  locations).
- `-t <order>` maps to seiza's SIP fit. When the polynomial fails its
  acceptance guards the `.wcs` is linear, which Siril reports as
  `SOLVE_LINONLY` and handles gracefully — the same semantics as a real
  solve-field that could not tweak.
- The `.wcs` writer and the Python bindings share one FITS keyword
  generator (`Wcs::fits_header_cards`), which is cross-validated against
  wcslib in CI (`seiza-py/tests/test_astropy_crossval.py`).

## Platform notes

Linux and macOS work as a plain drop-in. On Windows Siril launches
astrometry.net through a cygwin `bin/bash` wrapper, so a bare renamed
binary is not sufficient; Windows support would require emulating that
shell layout and is out of scope for now (Windows Siril users can use the
worker-protocol integration path instead when it lands upstream).

## Validation

Unit tests cover the xyls parser, Siril's exact argument shape, and the
`.wcs` card structure. An end-to-end rehearsal (real image, Siril-format
xyls written via astropy, Siril's literal argv, `.wcs` parsed with wcslib)
solved a 6165x4026 M31 frame blind-in-position with SIP order 3 at 1.45"
RMS over 130 matched stars.
