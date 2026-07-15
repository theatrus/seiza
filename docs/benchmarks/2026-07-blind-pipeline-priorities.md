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

At that checkpoint the source passed formatting, strict Clippy, the release
build, and all 112 local workspace tests. The two hosted/network integration
tests remained intentionally ignored.

## JPEG validation corpus (2026-07-15)

An external AstroBin-derived corpus was added after the FITS measurements. It
contains 25 RGB JPEGs with independent TOML WCS sidecars, spanning 3311 x 3312
through 11622 x 8056 pixels and 0.163 through 2.822 arcsec/pixel. The sidecars
provide `crval`, `crpix`, and a CD matrix; the expected pixel scale was computed
as `3600 * sqrt(abs(det(CD)))`. Sidecar dimensions matched the decoded JPEG
dimensions in every case.

The Seiza release executable at `e6f24aa` and ASTAP CLI 2026.06.29 with its
D50 database were run once per image and mode. Hinted solves used the sidecar
center, a 2-degree radius, and the sidecar scale/FOV. Position-blind solves
omitted the center and used the same scale knowledge. For Seiza that was a 20%
scale envelope; for ASTAP it was the sidecar-derived image height and a
180-degree search. Seiza fully blind solves used the CLI defaults of 0.1
through 20 arcsec/pixel and 400 verified hypotheses. Seiza blind modes used
the prebuilt G<=16 index with `--index-mag-limit 16`.

| Solver and mode | Result | Median wall (successes) | Median CPU (successes) | Median center error | Maximum center error |
|---|---:|---:|---:|---:|---:|
| Seiza hinted | 25/25 | 398 ms | 453 ms | 4.63 arcsec | 7.86 arcsec |
| ASTAP hinted | 13/25 | 1.17 s | 1.09 s | 2.80 arcsec | 8.60 arcsec |
| Seiza position blind, scale +/-20% | 25/25 | 798 ms | 1.50 s | 4.57 arcsec | 7.92 arcsec |
| ASTAP position blind, known FOV | 12/25 | 4.71 s | 3.69 s | 2.76 arcsec | 8.60 arcsec |
| Seiza fully blind, 0.1-20 arcsec/pixel | 23/25 | 969 ms | 2.44 s | 4.85 arcsec | 7.92 arcsec |

For both complete Seiza 25-image matrices, median Seiza RMS was 2.18 arcsec
and median absolute scale disagreement with the sidecars was 0.016%. ASTAP's
successful hinted and position-blind solves had median absolute scale
disagreement of 0.035% and 0.037%, respectively. Every reported success from
both solvers agreed with the independent WCS; there were no false positive
solutions.

Among images both programs solved, Seiza was faster in all 13 hinted pairs
(3.48x median ASTAP/Seiza wall ratio) and 9 of 12 position-blind pairs (2.91x
median ratio). ASTAP won the three difficult Seiza position-blind outliers:
Crescent, Sh2-101, and the full-frame NGC 7331 image. Conversely, ASTAP's
position-blind fine-scale NGC 3310 pair took 87 and 117 seconds, and the
Whirlpool image took 83 seconds; Seiza solved those three in 0.3 through 1.1
seconds.

ASTAP's failed cases wrote `PLTSOLVD=F` and reported `No solution found`; none
timed out. Its failures cluster in heavily processed narrowband fields,
including all three C34 images, all four sidecar-backed Sh2-119 images, and all
three WR134 images. Seiza solved every one of those in both primary modes.

The two fully blind misses were the Crescent and full-frame NGC 7331 images.
Both cleanly exhausted the default 400-hypothesis budget in approximately 26
seconds rather than timing out. The position-blind sweep also exposed four
search-funnel outliers: Sh2-101 at 11.6 seconds, Crescent at 10.8 seconds, and
the two NGC 7331 variants at 6.3 and 2.5 seconds. Sh2-101 then solved fully
blind in 0.84 seconds, showing that scale filtering currently changes
hypothesis ordering and does not monotonically reduce search work.

Six additional JPEGs have descriptive Markdown with approximate object
coordinates but no full WCS and were excluded from quantitative agreement
counts. One AVIF plus TOML pair was also outside this JPEG validation. Because
the tested JPEGs decode as RGB, this initial matrix exercised the detector's
general f32 luma path rather than the native `ImageLuma8` path. It therefore
provided the comparison corpus for the follow-up below.

These are single-run validation timings, retained with per-image stdout and
stderr, rather than a repeated microbenchmark. They are suitable for checking
format support, solution correctness, and large search regressions.

## Accepted: selectable generic u8/f32 detection for JPEG

The detector now exposes `DetectBackend::{Auto, U8, F32}` through
`DetectConfig` and the global CLI option `--detection-backend`. `Auto` uses u8
for all decoded 8-bit image variants, including RGB JPEGs, and retains f32 for
higher-precision inputs. Forced u8 and f32 modes remain available for callers,
diagnostics, and reproducible comparisons.

The shared implementation is statically generic rather than a runtime loop
over an erased pixel type. `detect_stars_luma<T: DetectionSample>` orchestrates
background estimation, thresholding, and connected components. Trait methods
retain the optimized histogram/parallel implementation for u8 and the
selection/SIMD implementation for f32, and Rust monomorphizes component
extraction for each excess type.

CI exercises both monomorphized implementations on Ubuntu, Windows, and macOS.
A sub-u8-contrast 16-bit fixture must produce a detection through forced f32
and no detection through forced u8, so the test cannot pass if both enum values
accidentally route to one backend. Separate tests require auto to match forced
u8 for RGB8 and forced f32 for 16-bit input. CLI parsing covers the default and
all explicit backend values and rejects unknown values.

A detection-only A/B used all 25 WCS-backed JPEGs twice per backend, alternating
backend order within each image. Across 50 pairs, u8 saved a median 8.3 ms wall
and 39.1 ms CPU and won 38 wall-time pairs. On the 11622 x 8056 Whirlpool JPEG,
three alternating runs reduced median peak private memory from 1080.6 MiB to
543.1 MiB (537.5 MiB, 49.7%) and median CPU from 1.17 seconds to 0.92 seconds.

The solve validation deliberately forced each backend:

| Backend | Hinted | Position blind | Median center-error change vs f32 |
|---|---:|---:|---:|
| f32 | 25/25 | 25/25 | baseline |
| u8 | 25/25 | 24/25 | +0.011 arcsec hinted / +0.001 arcsec blind |
| auto | 25/25 | 25/25 | uses the selected successful result |

Forced u8 changed the detection ordering enough for Sh2-101 to exhaust the
default 400 blind hypotheses, while forced f32 solved it. To avoid silently
trading coverage for memory, normal `auto` solves retry detection and solving
with f32 only when converted 8-bit luma fails. The final auto matrix therefore
preserved all 50 hinted/position-blind successes; the one compatibility retry
took 35.5 seconds. Explicit `u8` never falls back, and explicit `f32` goes
directly through the preserved high-precision path.

## Accepted: linear f32 FITS detection and selectable fallback

The first FITS fallback prototype reused the MTF display stretch and merely
changed its output type to f32. That is not the right numeric representation
for this detector. Its local median/MAD threshold is invariant under positive
affine scaling, so a nonlinear visibility stretch adds no detection power. It
can instead amplify background noise, compress bright cores, change component
areas, and reorder the stars supplied to the solver.

Forced f32 FITS loading now produces linear normalized grayscale samples:
u8 and u16 values retain their full source distinctions through division by
their type maximum, while integer and floating-point FITS types receive finite
min/max affine normalization. Planar RGB collapses to linear luminance and CFA
input follows the existing debayer path. Non-finite float samples become zero.
The MTF remains in the compact u8 path and in preview generation, where
compressing the sensor range is necessary.

Solve retry behavior is centralized in one `SolveInvocation` used by hinted,
blind, and ASTAP-compatible calls. It owns the primary detections, lazily
reloads f32 detections after a normal solve miss, caches them across multiple
solve strategies, and exposes the detections that produced the successful
solution. The global `--detection-fallback none|f32` option defaults to `f32`;
forced backends remain reproducible and never retry. FITS fallback explicitly
reopens the source rather than trying to recover precision from the MTF/u8
buffer.

A release validation forced both backends with fallback disabled on fine,
medium, and wide 16-bit FITS fields:

| Mode | Backend | Solved | Matched stars (fine / medium / wide) | RMS arcsec (fine / medium / wide) |
|---|---|---:|---|---|
| Hinted | u8 MTF | 3/3 | 81 / 87 / 107 | 0.639 / 0.430 / 1.768 |
| Hinted | linear f32 | 3/3 | 81 / 87 / 107 | 0.623 / 0.252 / 1.704 |
| Blind | u8 MTF | 3/3 | 89 / 103 / 120 | 0.640 / 0.429 / 1.725 |
| Blind | linear f32 | 3/3 | 89 / 103 / 119 | 0.624 / 0.259 / 1.673 |

Every solution agreed with the earlier validation. The f32 result had lower
reported RMS in all six pairs, while match counts were identical except for
one star on the wide blind field. These are correctness checks, not a timing
claim; each case was run once.

CI fixtures now prove that adjacent 16-bit FITS values survive the f32 loader,
Auto still selects the MTF/u8 loader, a solve miss really reopens FITS for f32,
and disabling fallback produces exactly one solve attempt. Together with the
existing backend-routing tests, the exact source passes formatting, strict
Clippy, the locked release build, and all 116 local workspace tests. The two
hosted/network integration tests remain intentionally ignored.

## Full detection rerun and linear-f32 threshold check

After the fallback implementation, detection was rerun once per backend on
all 13 FITS frames and all 25 WCS-backed JPEGs. Backend order alternated by
image, every one of the 76 release-process invocations succeeded, and output
was capped at the brightest 100 stars only to avoid timing console output.
Detection still performed its normal complete component search before that
final output truncation.

On FITS, u8 saved a paired median 27.7 ms of wall time and was faster on 12 of
13 images. Median CPU was 93.8 ms for u8 and 203.1 ms for linear f32. On JPEG,
u8 saved a paired median 7.3 ms and won 18 of 25 images; median CPU was 281.2
ms for u8 and 312.5 ms for f32. The single-run JPEG result is consistent with
the earlier 50-pair timing, but the repeated result remains the stronger
performance measurement.

Position matching exposed the expected effect of using different numeric
representations. The median FITS frame had 49 of its u8 top 50 detections
within two pixels of an f32 top-100 detection, while the median top-10 set
overlap was 7 of 10. JPEGs were closer: the corresponding medians were 49 of
50 and 10 of 10. Most FITS differences were ranking changes rather than lost
objects.

One 30-second green-filter Bode's Galaxy frame (`0003`) was the important
outlier. Only 22 of the u8 top 50 appeared anywhere in the f32 candidate list
within two pixels. U8 found 428 candidates, linear f32 found 534, and 319 were
spatially common. Inspection against the solved catalog overlay confirmed that
several high-ranked u8-only components were real bright stars rather than hot
pixels or galaxy structure.

Running the already-MTF-compressed pixels through f32 arithmetic reproduced
all 428 u8 detections and the complete top-50 ordering. The discrepancy is
therefore caused by the nonlinear MTF input representation, not by the generic
u8 and f32 detector implementations disagreeing on equivalent samples.

A linear-f32 sigma sweep initially looked promising: increasing sigma from 4
to 6 improved the outlier's u8-top-50 recovery from 22 to 43 and reduced its
candidate count from 534 to 411. Across all 13 FITS frames, sigma 6 improved
the minimum top-50 recovery from 22 to 43 without materially changing the
other 12 frames. It is not safe for solving, however:

| Linear-f32 setting | Hinted FITS solves | Blind FITS solves | Median hinted/blind matches |
|---|---:|---:|---:|
| sigma 4 | 13/13 | 13/13 | 87 / 103 |
| sigma 6 | 12/13 | 12/13 | 96 / 110 among successes |

The same `0003` frame failed in both modes at sigma 6. A focused sweep showed
that sigma 4 solved it with 16 hinted and 27 blind matches, whereas every
tested value from 4.5 through 6 failed both modes; blind attempts exhausted
the 400-hypothesis budget. This is a brightness-ordering and solver-funnel
interaction, not a simple count or geometric-overlap problem. Consequently,
the detector keeps sigma 4 and no tuning change was accepted from this rerun.
