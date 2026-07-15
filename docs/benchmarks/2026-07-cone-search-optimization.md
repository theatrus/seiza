# File-backed cone-search optimization

Date: 2026-07-14

This follow-up investigates `TileCatalog::cone_search` after an ETW CPU
profile identified it as the largest remaining blind-solve hotspot once
speculative verification was limited. All measurements used `seiza-cli` built
with `cargo build --release --locked -p seiza-cli` and the deep Gaia G<=17
`SEIZAST2` catalog.

## Cause

Stars in each catalog tile are stored brightest-first. Cone search keeps a
bounded heap of the brightest requested stars and asks the tile iterator to
stop once the remaining records are fainter than the heap threshold.

The legacy interleaved `SEIZAST1` iterator honored that early-stop result, but
the current columnar `SEIZAST2` iterator discarded it. Every selected tile was
therefore scanned to the end. The inner loop also called the general Vincenty
angular-separation function for every candidate, repeating the fixed cone
center's trigonometry and evaluating a square root and `atan2` when only an
inside/outside comparison was required.

The optimization:

- honors the existing early-stop callback for `SEIZAST2` tiles;
- precomputes the fixed cone terms and compares the spherical unit-vector dot
  product with `cos(radius)`;
- returns an empty result immediately when the requested limit is zero.

The file format, tile covering, magnitude ordering, heap policy, and returned
coordinates are unchanged.

## Expanded image corpus

The end-to-end corpus contains 13 real FITS frames across four groups:

- four 6248 x 4176, 300-second SII frames at approximately 1.035 arcsec/pixel;
- six 6248 x 4176, 30-second green-filter frames at the same image scale;
- two 6248 x 4176, 300-second H-alpha frames at approximately 0.33
  arcsec/pixel;
- one 9576 x 6388, 300-second OIII frame at approximately 1.50 arcsec/pixel.

The added data covers two additional sky regions and includes both short
broadband and long narrowband exposures. All 13 frames solved before and after
the change with identical matched-star counts.

## Results

With the normal all-core verification policy, two runs per image produced a
corpus-wide median wall time of 1.551 seconds before the change and 1.467
seconds after it, a 5.4% reduction. This includes process startup, FITS
loading, detection, index access, solving, output, and memory-map teardown.

An interleaved A/B used two otherwise identical release executables and an
experimental four-hypothesis verification batch. Across 39 paired runs, the
optimized executable saved a median 64.7 ms of wall time. Its overall median
was 1.003 seconds versus 1.069 seconds for the baseline, a 6.1% reduction.
Every solve returned the same match count.

A second interleaved A/B separated the two performance mechanisms over four
representative fields and 20 pairs. After the early-stop fix, the precomputed
dot-product membership test saved a further median 22.8 ms. Both changes are
therefore retained. The four-wide verification batch was used only to expose
the cone-search cost more clearly and is not part of this patch.

## Validation

The regression test directly verifies that a columnar tile stops after the
callback returns false. Existing randomized comparisons against the in-memory
brute-force catalog cover RA wrapping and both poles. The complete workspace
suite passes 95 local tests; the two network-hosted integration tests remain
intentionally ignored. Strict Clippy, formatting, and the release build pass.
