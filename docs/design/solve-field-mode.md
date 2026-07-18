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
- `-l <seconds>` is accepted but not enforced: typical seiza solves are
  sub-second and Siril retains its own cancel path via the `-C` stop file,
  which seiza checks between solve phases (not mid-search).
- `-t <order>` maps to seiza's SIP fit. When the polynomial fails its
  acceptance guards the `.wcs` is linear, which Siril reports as
  `SOLVE_LINONLY` and handles gracefully — the same semantics as a real
  solve-field that could not tweak.
- The `.wcs` writer and the Python bindings share one FITS keyword
  generator (`Wcs::fits_header_cards`), which is cross-validated against
  wcslib in CI (`seiza-py/tests/test_astropy_crossval.py`).

## Platform notes

Linux and macOS work as a plain drop-in: a copy of seiza named
`solve-field`.

On Windows Siril launches astrometry.net through
`<asnet_dir>/bin/bash -l -c ...` — historically a cygwin shell. Seiza does
not need the shell: a copy installed as `<asnet_dir>/bin/bash.exe`
recognizes the two commands Siril issues. The version probe
(`-c "solve-field --version"`) is answered directly, and the solve
invocation (`-c /tmp/asnet.sh`) resolves the cygwin-style path against the
layout root (`<asnet_dir>/tmp/asnet.sh`), parses the deterministic script
Siril writes there (`p="<table>"`, `c="<stopfile>"`, one solve-field
command line), substitutes the variables, and dispatches into the same
solve path. Windows paths in the script are treated literally — no shell
escaping semantics apply.

`seiza install-solve-field --dir <dir>` creates the complete layout on any
platform: `solve-field`, `bin/bash`, and the `tmp/` directory Siril
requires to exist before it writes its script.

## Known limitation: brightness-faithful FLUX ranking required

Seiza's matcher seeds triangles from the brightest image stars and assumes
the list's flux ordering roughly tracks catalog brightness. Siril's `.xyls`
FLUX column is the fitted PSF **amplitude**, which on stretched or
nonlinear images correlates poorly with photometric brightness (measured on
a real M31 export: only 2 of the 24 highest-amplitude stars were among the
100 photometrically brightest; the catalog-bright stars sat at amplitude
ranks 48-1875). Star positions in such a table are excellent (sub-0.2 px
against seiza's own detection), but no brightness-ranked subset overlaps
the catalog's bright stars, so hinted and blind solving both fail where
seiza's own detector on the same pixels succeeds.

The mode works around this automatically: when the source image sits
next to the `.xyls` (Siril derives the table path from the image path)
and its dimensions match `IMAGEW`/`IMAGEH`, seiza re-detects stars with
its own photometric flux and cross-matches the table's positions to
calibrate orientation and sub-pixel frame offset, then solves in the
exact table frame. Verified end-to-end against Siril 1.4.4: the
previously failing stretched M31 export now plate-solves inside Siril
with SIP order 3. The fallback (no image found, dimensions mismatch from
a cropped/downsampled selection, or a weak cross-match) uses the table
stars unchanged and still requires brightness-faithful ranking.

Removing the root — so pure star-list inputs solve regardless of flux
quality — is the catalog-seeded quad design recorded in
[rank-robust-matching.md](rank-robust-matching.md).

## Validation

Unit tests cover the xyls parser, Siril's exact argument shape, and the
`.wcs` card structure. An end-to-end rehearsal (real image, Siril-format
xyls written via astropy, Siril's literal argv, `.wcs` parsed with wcslib)
solved a 6165x4026 M31 frame blind-in-position with SIP order 3 at 1.45"
RMS over 130 matched stars. The Windows launch path was simulated
end-to-end: the version probe and an `asnet.sh` in Siril's exact format,
invoked through an installed `bin/bash`, produced the same solution.
