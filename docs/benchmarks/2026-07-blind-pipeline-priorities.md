# Blind-pipeline and grayscale-detection priorities

Date: 2026-07-14

This follow-up evaluates the remaining optimization priorities after the
hinted-search and file-backed cone-search changes landed. Privileged ETW/WPR
profiling was intentionally deferred. Every experiment used release builds,
process wall and CPU time, targeted memory polling, unit tests, and real-image
solution quality.

## Test corpus

The corpus contains 13 real FITS frames in four groups:

- four 6248 x 4176, 300-second SII frames at approximately 1.035 arcsec/pixel;
- six 6248 x 4176, 30-second green-filter frames at the same scale;
- two 6248 x 4176, 300-second H-alpha frames at approximately 0.33
  arcsec/pixel;
- one 9576 x 6388, 300-second OIII frame at approximately 1.50 arcsec/pixel.

The baseline is merged `main` at `3874387`, containing PRs #14 and #15. All
executables were built with `cargo build --release --locked -p seiza-cli`.
Paired tests alternated executable order to reduce thermal and cache bias.

## Accepted: adaptive hypothesis verification

Blind verification previously launched a full triangle solve for one ranked
hypothesis per logical CPU. The correct hypothesis usually ranks first, so a
20-thread machine performed up to 19 unnecessary solves before Rayon could
observe the successful result.

The new schedule tries 4 hypotheses, then 8, then doubles to the full thread
count after misses. It never drops a hypothesis and recovers full parallelism
for difficult or unsolved fields.

Across 26 paired solves, adaptive verification saved a median 458 ms of wall
time and 9.35 CPU-seconds. Median CPU fell from 11.27 seconds to 1.82 seconds,
with identical match counts in every pair. The existing synthetic noise test
also passes, exercising the widening failure path.

## Accepted: compact blind-funnel fast path

The global image-quad generator already uses the subset ladder 8, 10, 12, 16,
20, 26, and 32, but it accumulated every rung before ranking any hypothesis.
An experimental run limited the global set to the 8-star rung while retaining
all locally-windowed quads. All 13 real frames solved, with a 535 ms median
wall time and 1.13 CPU-seconds.

The implementation now tries that compact funnel first. Only a normal solve
miss falls through to the unchanged complete ladder; catalog and index errors
propagate immediately. This preserves the original search coverage for fields
whose signal is not represented in the early hypotheses.

## Accepted: native 8-bit grayscale detection

FITS input is already autostretched to 8-bit grayscale. The detector then
converted it to a full f32 luma image and allocated a second image-sized f32
threshold-excess buffer. The specialized path instead:

- computes per-tile median and MAD with 256-bin histograms over the u8 input;
- stores threshold excess as u8;
- normalizes component weights back to the detector's existing 0-1 public
  flux and peak scale;
- retains the general f32 path for other image representations.

A regression test compares the native path with the f32 reference and gets
identical detections, areas, centroids, fluxes, and peaks. Across 26 paired
blind solves it saved a median 48 ms wall and 172 ms CPU, with identical
solutions.

Polling private committed memory separately from mapped-file working set gave:

| Frame | f32 detector | u8 detector | Reduction |
|---|---:|---:|---:|
| 26 MP H-alpha | 253.9 MB | 153.5 MB | 100.4 MB |
| 61 MP OIII | 590.3 MB | 179.1 MB | 411.2 MB |

## Rejected experiments

Two profile-driven ideas no longer help after the earlier changes:

- A flat direct-index detection grid regressed median wall time by 3 ms and
  mean wall time by 16 ms. Zeroing roughly 94,000 cell heads cost more than
  hashing only 200 detections after the compact funnel reduced coarse queries.
- Fixed triangle descriptor buckets regressed median wall time by 35.5 ms and
  CPU by 156 ms. Fragmented traversal lost more cache locality than sorting a
  contiguous triangle array. Switching only to an unstable in-place sort was
  effectively neutral (-1 ms median wall savings with a worse mean), so the
  stable sort remains unchanged.

## Combined result

The final release was compared with merged `main` over 26 paired blind solves:

| Metric | Merged main | Final | Reduction |
|---|---:|---:|---:|
| Median wall time | 1.592 s | 0.528 s | 1.065 s (66.8%) |
| Median solver time | 1.245 s | 0.340 s | 0.905 s (72.7%) |
| Median CPU time | 11.148 s | 1.078 s | 10.102 s (90.6%) |

Every pair returned identical match counts and RMS. A separate hinted A/B on
fine-, medium-, and wide-scale frames produced 15 successful pairs: median
wall time fell from 247 ms to 205 ms, with identical match counts and RMS.

The exact final source passes formatting, strict Clippy, the release build,
and all 106 local workspace tests. The two hosted/network integration tests
remain intentionally ignored.
