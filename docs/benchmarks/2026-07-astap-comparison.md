# ASTAP vs seiza on real narrow- and wide-field FITS frames

Date: 2026-07-14

This report compares ASTAP and seiza on two real, consecutive narrow-field
H-alpha exposures plus an independent 61-megapixel wide-field OIII exposure.
It also investigates an initially surprising result:
seiza's hinted solve was much slower than both ASTAP and seiza's own blind
solve when the default 2-degree hinted search radius was used.

## Test system

- Windows 11 x64, build 26200
- Intel Core i7-12700H (14 cores / 20 logical processors)
- seiza 0.4.1 at `57ddfd378b8f2c329ff128729d02bd21330e24b9`, built with
  `cargo build --release --locked -p seiza-cli`
- seiza deep Gaia G<=17 catalog (`stars-deep-gaia17.bin`) and maintained
  G<=16 blind index (`blind-gaia16.idx`)
- ASTAP CLI 2026.06.29 with the D50 catalog

The catalogs are not identical. Each solver used the catalog recommended for
small, fine-scale fields and for its own blind-solving implementation.

## Input frames

Both images are 6248 x 4176 16-bit FITS frames from a ZWO ASI2600MM Pro on a
2350 mm C9.25, exposed for 300 seconds through an H-alpha filter. Their
nominal scale from the 3.76 um pixels and focal length is 0.3298 arcsec/pixel;
the solved scale is about 0.3265 arcsec/pixel. The image height is 0.382568
degrees.

The narrow frames point at the Cocoon Nebula near RA 328.43 degrees, Dec 47.27
degrees. Both solvers converged to the same center and scale. A representative
solution was RA 328.4308 degrees, Dec 47.2731 degrees at 0.3265
arcsec/pixel.

The independent frame is a 9576 x 6388 16-bit FITS exposure from a ZWO
ASI6200MM Pro on a 518 mm refractor, exposed for 300 seconds through an OIII
filter. Its nominal scale is 1.4972 arcsec/pixel and its image height is
2.6567 degrees. It points at the California Nebula near RA 60.24 degrees, Dec
36.52 degrees. ASTAP and seiza solved it at about 1.4947 arcsec/pixel and
agreed on the center to roughly an arcsecond. The source frames and catalogs
are intentionally not committed.

## Method

Wall time includes process startup, FITS loading and stretching, star
detection, catalog access, solving, and result output. The primary comparison
used three repetitions per image, for six measurements per solver/mode. Runs
used the normal OS file cache; this is representative of repeated local plate
solves during an imaging session, not a forced cold-storage benchmark.

Three workloads were measured:

1. **Hinted, 2-degree search radius**: FITS/header sky position and nominal
   scale/FOV supplied to both solvers.
2. **Position blind, scale known**: no useful sky position; seiza received a
   plausible 0.25-0.45 arcsec/pixel range, while ASTAP received the known FOV.
   ASTAP started from a deterministic wrong position (RA 0h, Dec 0 degrees)
   with a 180-degree search radius.
3. **Full blind**: no useful position or scale. seiza searched 0.1-20
   arcsec/pixel using its prebuilt index; ASTAP used automatic FOV detection
   from the same deterministic wrong position. This mode was run once per
   image because ASTAP's exhaustive sweep is expensive.

## Baseline results

All reported solves succeeded.

| Workload | ASTAP median wall time | seiza median wall time | Relative result |
|---|---:|---:|---:|
| Hinted, 2-degree radius (6 each) | 0.271 s | 4.655 s | ASTAP 17.2x faster |
| Position blind, scale known (6 each) | 31.281 s | 1.609 s | seiza 19.4x faster |
| Full blind (2 each) | 149.629 s | 2.314 s | seiza 64.6x faster |

The full-blind ASTAP number is specific to this image scale and deterministic
starting position; a spiral-search solver's time varies with the unknown
field's distance from that start. It is included to make the workload fully
reproducible, not as a universal all-sky average.

## Why the hinted path was slow

The elapsed time printed by seiza's hinted CLI starts immediately before
`seiza::solve::solve`, after FITS loading, stretching, detection, and catalog
opening. On the first frame, the CLI reported 4.39 seconds inside `solve()` at
a 2-degree radius, versus 0.07-0.08 seconds at radii up to 0.25 degrees. Total
wall time changed by the same amount, so image I/O and detection were not the
bottleneck.

The hinted solver divides the search circle into FOV-sized windows. For each
window it takes up to 100 catalog stars and forms all triangles before it
votes on candidates. That is up to `C(100, 3) = 161,700` catalog triangles per
window. The current implementation builds candidates for every window before
checking whether the center window already solves.

For these images, the tolerance-expanded diagonal field radius makes the
window step about 0.413 degrees:

| Hint radius | Windows | Upper bound on catalog triangles | Median wall time (3 runs) | Matches |
|---:|---:|---:|---:|---:|
| 0.02 deg | 1 | 161,700 | 0.221 s | 81 |
| 0.05 deg | 1 | 161,700 | 0.227 s | 82 |
| 0.10 deg | 1 | 161,700 | 0.220 s | 83 |
| 0.25 deg | 1 | 161,700 | 0.224 s | 84 |
| 0.50 deg | 5 | 808,500 | 0.477 s | 82 |
| 1.00 deg | 21 | 3,395,700 | 1.550 s | 81 |
| 2.00 deg | 69 | 11,157,300 | 4.667 s | 44 |

The stepwise runtime increase tracks the number of windows closely. The lower
match count at 2 degrees is also consistent with unrelated windows filling
the global candidate budget before refinement; the solution remained valid,
but searched more work and retained fewer matched stars.

## Tight-hint comparison

Repeating the normal hinted benchmark with a 0.25-degree radius produced:

| Solver | Attempts | Successes | Median wall time | Range |
|---|---:|---:|---:|---:|
| ASTAP | 6 | 6 | 0.277 s | 0.265-0.306 s |
| seiza | 6 | 6 | 0.245 s | 0.233-0.257 s |

With unnecessary search windows removed, seiza was about 11.5% faster than
ASTAP on this hinted workload. This confirms that release-mode compilation,
FITS decoding, and star detection were not responsible for the original gap.

## Independent wide-field frame

The 61-megapixel California Nebula frame tests a different camera, optical
system, filter, target, pixel scale, and FOV. Its tolerance-expanded diagonal
field radius is about 2.87 degrees, so the normal 2-degree hint produces only
the center window even before the optimization proposed below.

| Workload | ASTAP median wall time | seiza median wall time | Relative result |
|---|---:|---:|---:|
| Hinted, 2-degree radius (3 each) | 0.644 s | 0.431 s | seiza 1.50x faster |
| Position blind, scale known (3 each) | 1.022 s | 1.747 s | ASTAP 1.71x faster |
| Full blind (1 each) | 3.874 s | 2.500 s | seiza 1.55x faster |

The hinted seiza solve reported 107 matched stars at 1.768 arcsecond RMS; its
full-blind solve reported 120 matches at 1.725 arcsecond RMS. This independent
case reinforces the diagnosis: hinted performance is already good when the
configured radius does not multiply the number of FOV-sized windows.

## Implemented optimization

An accurate hint should be cheap even when the configured fallback radius is
large. The solver can try the center FOV-sized window first and return a
well-supported solution immediately. If the center attempt fails, it must
preserve the existing multi-window search so stale mount coordinates still
solve anywhere inside the requested radius.

The implementation uses the fast path only when the requested radius contains
additional FOV-sized windows. It accepts the center-window result only with at
least 12 matched stars and RMS below 2 pixels, the same support thresholds
used when the blind solver verifies a whole-sky hypothesis. Weaker center
candidates fall through to the original full-radius algorithm. A radius
smaller than one window step already contains only the center, so it runs once;
this avoids duplicating the tightly localized searches used by blind
verification.

Regression coverage demonstrates both behaviors:

- a strong solution at the hinted center does not enumerate the surrounding
  2-degree search grid;
- an offset field still falls back to the full search and solves.
- a one-window failure is not retried.

After rebuilding `seiza-cli` in release mode, the unchanged 2-degree hinted
benchmark produced:

| Frame | ASTAP median (3 runs) | seiza median (3 runs) | seiza result |
|---|---:|---:|---:|
| Cocoon 0020, 0.33 arcsec/pixel | 0.274 s | 0.248 s | 1.11x faster |
| Cocoon 0021, 0.33 arcsec/pixel | 0.288 s | 0.265 s | 1.09x faster |
| California 0226, 1.50 arcsec/pixel | 0.626 s | 0.422 s | 1.48x faster |

Across the six narrow-frame seiza runs, the median fell from 4.655 seconds to
about 0.261 seconds, a 17.8x improvement. The first invocation immediately
after relinking was a 0.731-second process-startup/OS outlier, but its
solver-reported time was still 0.09 seconds and it returned 81 matches; the
per-frame median is insensitive to that outlier.

A real fallback check shifted the Cocoon hint by 2 degrees in RA (about 1.36
degrees on the sky) while retaining the 2-degree radius. The optimized release
binary rejected the center fast path, found the actual field through the
original grid search in 4.76 seconds, and returned 43 matches at 0.586
arcsecond RMS.
