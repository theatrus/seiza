# Siril integration

Status: reference for Siril's solver landscape and the seiza integration
paths — one shipped, one available today, one proposed upstream

This records what was learned from Siril's source (1.4/1.5 series,
`src/algos/astrometry_solver.*`) so it does not have to be re-derived,
and how each seiza integration path maps onto it.

## Siril's plate-solving options

Siril's solver backend is a two-value enum with no plugin system:

1. **Built-in Siril solver** (`SOLVER_SIRIL`, the default): Siril's own
   triangle matcher. It requires an approximate target position (FITS
   headers or a user-entered object) plus focal length and pixel size;
   when the direct attempt fails it runs a "near solve" outward within a
   configurable radius. It is **not blind** — it cannot solve a frame
   with no positional idea.
2. **Local astrometry.net** (`SOLVER_LOCALASNET`): shells out to
   `solve-field` as a subprocess. This is the only external-solver slot
   and the **only path to true blind solving** — the "ignore target
   position" and "ignore sampling" options exist exclusively for it.

Two common misconceptions, verified absent from the source:

- **No online astrometry.net** (nova.astrometry.net): no API client
  exists. The confusion arises because the *built-in* solver can fetch
  its reference **catalog stars** online (VizieR/Gaia TAP cone searches
  when local KStars/Gaia catalog files are absent) — online catalog
  data feeding a local solve, not online solving.
- **No ASTAP support**: seiza's ASTAP-compatible mode does nothing for
  Siril; the solve-field contract is the one that matters.

## Contract details worth knowing

Learned by reading `local_asnet_platesolve` and validated against a real
Siril 1.4.4:

- The `.xyls` star table's **Y axis is bottom-up** (Siril writes
  `ry - ypos + 0.5`), and the `.wcs` must describe that same frame.
- The `FLUX` column is the fitted **PSF amplitude** (`psf_star->A`), not
  integrated flux — on stretched images it ranks stars far from
  photometric order (see
  [rank-robust-matching.md](rank-robust-matching.md)).
- `solve-field --version` must print a single line; multi-line output is
  classified as a pre-0.88 astrometry.net.
- A requested SIP order with no distortion terms in the `.wcs` is
  reported as `SOLVE_LINONLY` and handled gracefully — a solver may
  decline to fit distortion without failing the solve.
- On Windows, Siril launches the solver through
  `<asnet_dir>/bin/bash -l -c`, writing a deterministic launch script —
  emulatable without cygwin.

## Seiza integration paths

| Path | Status | Siril changes | Notes |
| --- | --- | --- | --- |
| solve-field drop-in | **Shipped** (0.7.1) | none | `seiza install-solve-field`; fills the `SOLVER_LOCALASNET` slot on all platforms |
| sirilpy script | Available | none | `pip install seiza`; script-driven, not in the solver dropdown |
| `SOLVER_SEIZA` upstream | Proposed | enum + branch + prefs + GUI | First-class dropdown entry driving `seiza worker` |

**solve-field drop-in** — the shipped path; contract, mitigations, and
platform mechanics in [solve-field-mode.md](solve-field-mode.md). Its
main value over the built-in solver: blind solving without an
astrometry.net installation (historically a multi-GB cygwin bundle on
Windows), fast wide searches, and SIP from seiza's solver.

**sirilpy script** — Siril's Python scripting runs out-of-process over a
socket with shared memory. A script calls `get_image_stars()` (the same
amplitude-flux PSF stars as the xyls; rank-robust matching applies),
solves with `seiza.solve(..., sip_order=N)`, and injects the solution
with `set_image_header(solution.fits_header_text())`, which Siril
reparses through wcslib. Works today for scripted workflows; it is not a
solver-dropdown option.

**`SOLVER_SEIZA` upstream** — the eventual first-class integration: a
new enum value dispatching in `plate_solver()`
(src/algos/astrometry_solver.c), a preference for the seiza path, and a
GUI dropdown entry, driving `seiza worker` over the versioned JSON-RPC
protocol the way `local_asnet_platesolve` drives solve-field. This
mirrors the asnet integration Siril already maintains and needs no Rust
in Siril's build. Worth proposing once the drop-in has real-world miles.

## Upstream contribution opportunities

- **Write integrated flux to the xyls `FLUX` column**
  (`save_list_as_FITS_table`, src/algos/star_finder.c currently writes
  `psf_star->A`). Real astrometry.net sorts by the same column
  (`-s FLUX`) and degrades on stretched images too; this helps both
  solvers regardless of any seiza work.
- **`SOLVER_SEIZA`** as above, once justified by adoption.
