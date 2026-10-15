# Solver lessons from olive-solve

[olive-solve](https://github.com/oakamil/olive-solve) (Apache 2.0) is a
Rust rewrite of ESA's tetra3 solver, by way of cedar-solve, for small
finder cameras: a Raspberry Pi, one lens, and an IMU that tracks the camera
between frames. This note lists what seiza can learn from it, and what it
cannot.

Status: 1 and 2 are built, as described under each. The detector (3),
fuller results (4) and horizon pruning (5) come next. The documentation
problems at the end are fixed.

## Why its speed does not carry over

olive-solve solves a clean frame in under a millisecond on a Pi. Most of
that comes from the narrow job, not from tricks seiza lacks:

- Its database holds 42,000 Hipparcos stars and patterns sized for one
  field of view, 10 to 30 degrees. Seiza solves from 0.1 to 20 arcsec per
  pixel against Gaia, with an index of 1.6 GB.
- It tries 4-star patterns brightest first, in the order cedar-solve uses
  (every pattern holding star *l* before star *l+1* is added), and returns
  on the first one that passes. On a clean frame that is often the first
  pattern.
- It caps the stars it uses at 150 and thins them to spread across the
  frame.

Seiza's blind solver votes over many patterns and checks the best-ranked
hypotheses, which suits its wider job. The ideas below are the ones that
fit seiza anyway.

## 1. A false-match probability

**olive-solve.** After a candidate rotation is found, it projects the
catalog stars in the frame and counts how many of the N detections land
within the match radius of one. It accepts the match only if so many
matches are unlikely by chance: with `p` the chance that one detection
falls within the radius of some catalog star, the chance of the count is a
binomial tail, and it must fall below a threshold divided by the number of
patterns the database holds. The solution reports that chance.

**Seiza.** Acceptance is a fixed count:

- A blind solve needs at least 12 matches under 2 px RMS.
- A hinted solve needs 6 votes among the brightest stars, then keeps any
  refinement that holds 6 pairs. It has no RMS check.
- No solution reports how sure it is.

A fixed count ignores how crowded the frame is. With 600 detections, 4000
Gaia stars in a 6000 x 4000 frame and a 4 px radius, one detection has a
0.8% chance of landing near some catalog star, so about 5 chance matches
are expected, and 12 or more happen in about 1 try in 200. Checking 400
hypotheses makes a false accept at 12 matches likely. The benchmark notes
record exactly this: six
annotated JPEGs solved to unrelated fields with 12 to 21 matches, and
`docs/benchmarks/2026-07-blind-pipeline-priorities.md` concluded that they
need "an explicit confidence model".

**Proposal.** Compute, for the final transform of every solve:

- N: the detections the final match considered.
- M: the catalog stars that project inside the frame.
- r: the match radius in pixels; A: the frame area in pixels.
- `p = min(1, M·π·r²/A)`. Overlapping circles cover less than `M·π·r²`,
  so this overstates `p`, which errs towards refusing.
- k: the matched pairs, and d = 3, the pairs that fixing an affine
  transform takes and that match by construction.
- One hypothesis: `P1 = P(Binomial(N - d, p) ≥ k - d)`.
- The search: `P = min(1, T · P1)`, where T is the number of candidate
  transforms the search scored, since each was a chance to be fooled.

The tail is summed in log space with `ln Γ`, so a strong solve does not
underflow before it is compared. `Solution` gains
`false_match_probability`.

**Built.** Every `Solution` carries `false_match_probability`. The hinted
search counts the candidate transforms it voted on, the rank-robust search
the transforms it took a census of, and a blind solve multiplies by the
fields it scored. The CLI prints it (`by chance`), and the worker, the C ABI
and Python return it.

Measured on 27 local images (the M45 and Praesepe FITS series, XISF frames,
ten AstroBin JPEGs and phone photos) with the deep Gaia catalog and the
G≤16 index:

- The 23 that solve blind scored between 7e-53 (a phone photo, 36
  matches) and below f64's range (Praesepe, 151 matches); most were below
  1e-200.
- Random and stroke-shaped detections (like rendered labels) never passed
  the 12-match blind floor, and failed after 28 to 108 s each.
- With the blind floor lowered to 6 matches and 3 px, every one of 12 such
  trials solved to some field, at probabilities of 0.18 to 1.
- 51 hinted solves of real images at the right scale but 30 to 300 degrees
  from the true field, and 40 of random detections, all failed at 4 or 5
  votes.

So the limits are:

- Blind: at most one in a billion (`BlindParams::max_false_match_probability`),
  alongside the 12-match and 2 px rules, which stay.
- Hinted: at most one in a thousand. The hint is itself strong evidence, and
  a real but sparse field with few matches beyond the fit's three can score
  near 1e-5; this refuses only matches that chance explains easily.
- The center-window fast path skips the wider search only below one in a
  billion.

The annotated JPEGs from the benchmark notes were not on hand; they are the
next check.

## 2. Time limits and cancellation

**olive-solve.** Every 128 pattern loops it checks a deadline (5 s by
default) and a cancel flag, and returns `Timeout` or `Cancelled`.

**Seiza.** `solve.rs` and `blind.rs` have neither. Work is bounded only by
hypothesis counts and the rank-robust probe budget, and a clean blind
failure on the JPEG set once took 26 s. solve-field's `-l` is parsed and
ignored, and the worker cannot stop a solve. N.I.N.A. and other ASTAP or
solve-field callers expect a solver to honour a time limit.

**Built**, as proposed:

- Move `CancelSignal` from seiza-stacking into `seiza`, and re-export it
  from seiza-stacking so its path there still works. The C ABI's
  `SeizaCancelSignal` and Python's `cancel: Callable[[], bool]` already wrap
  it.
- A `SolveLimits { deadline: Option<Instant>, cancel: Option<CancelSignal> }`,
  passed to new `solve_with_limits` and `solve_blind_with_limits` functions;
  `solve` and `solve_blind` call them with no limits.
- Check it between search windows and every 1024 candidates in hinted
  solving, between quad batches and probe batches in rank-robust matching,
  and between ladder rungs, coarse-scoring chunks and verification batches
  in blind solving. Work inside Rayon closures checks the same signal and
  stops early.
- `seiza::Error` becomes `#[non_exhaustive]` and gains `TimedOut` and
  `Cancelled`, so callers can tell them from a failed solve.
- Hosts: `--time-limit` on `seiza solve` and `seiza solve-blind`;
  solve-field mode honours `-l`, and its `-C` stop file now stops the solve
  itself rather than only between phases; the worker takes `timeLimitMs`
  per solve (error -32011), and a `cancel` notification read on its own
  thread (error -32012); the C ABI gains
  `seiza_solve_image_with_limits_json`; Python gains `time_limit=`
  (`TimeoutError`) and `cancel=` (polled every 50 ms; Ctrl-C works too).

A blind solve of IC 1318 that failed after 48 s stops at 5.4 s with
`--time-limit 5`.

## 3. Detector

**olive-solve** has several things seiza's detector (`seiza/src/detect.rs`)
lacks:

- Its block-median background interpolates bilinearly between block
  medians. Seiza takes one median per 64 px tile with hard steps between
  tiles, so the threshold jumps at tile edges on a gradient.
- It grows stars only from pixels that survive a 3 x 3 erosion, which
  drops single hot pixels and cosmic-ray hits. Seiza relies on
  `min_area = 3`.
- It can sum 2 x 2 or 4 x 4 pixels before detecting. Detection is seiza's
  largest cost on 60 to 94 MP frames, and ASTAP mode parses `-z`
  (downsample) and ignores it.
- A per-row median background removes horizontal banding from uncooled
  CMOS sensors.
- It samples every fourth row for histogram medians, and can re-threshold
  at another sigma without recomputing the background, which would make
  seiza's detection retries cheaper.

**Proposal.** Interpolate the tile background; seed components from eroded
pixels; add 2x and 4x binning to `DetectConfig` and wire ASTAP's `-z` to
it; add a row-median background mode. Keep the u8 and f32 paths in step.

## 4. Fuller results

**olive-solve** returns the matched image and catalog stars, RMS, 90th
percentile and largest residuals, the false-match probability, whether the
image is mirrored, and its solve time.

**Seiza** returns the WCS, the match count and the RMS. The C ABI adds
detected and projected star positions for drawing, but not which matched
which.

**Proposal.** `Solution` gains the matched pairs (pixel position, catalog
RA, Dec and magnitude, and residual), the 90th percentile and largest
residuals, and the probability from 1. The worker, C ABI and Python return
them. PSF Guard and the apps can use the pairs for photometry and for
showing where distortion is left.

## 5. Horizon pruning

**olive-solve** rejects a candidate when any of its pattern stars, or the
final boresight, lies below the horizon for a given latitude and local
sidereal time.

**Seiza** searches the whole sky blind even when the caller knows where and
when the image was taken. N.I.N.A. and the apps know both, and FITS headers
often carry `SITELAT`, `SITELONG` and `DATE-OBS`.

**Proposal.** An optional observer (latitude, longitude, time) in
`BlindParams`. Blind solving drops hypotheses whose centre lies below a
minimum altitude before coarse scoring, which roughly halves the work and
removes false fields on the other side of the Earth. Use a margin of a few
degrees for refraction and long exposures. The CLI reads the observer from
FITS headers, with flags to override.

## Later

- **Fewer disk reads per index lookup.** Seiza binary-searches 1.6 GB of
  memory-mapped keys, which touches many pages on a cold solve. A small
  in-memory table over the top bits of each key would cut a lookup to one
  or two pages. Storing each pattern's largest edge in its own array would
  let the scale filter run without reading the 44-byte pattern records.
  olive-solve gets the same effect from a dense 16-bit tag array beside its
  hash table.
- **A regression set in the repo.** olive-solve checks 738 recorded centroid
  lists against known answers, and mirrored copies of its sample images,
  and prints solve times. Seiza's real-image solve tests need the network
  and cover five images. Saving centroid lists and true WCS from the
  benchmark sets would let CI catch ranking regressions.
- **One-term radial distortion.** For wide lenses with too few stars for
  SIP (order 2 needs 18 pairs), a single radial term `k·r³` needs far
  fewer, and maps onto SIP terms for output.

## Not taken

- The single-field database and returning on the first pattern that
  passes, which do not fit seiza's range of scales.
- The f32 build and fixed-point arithmetic for ARMv6 and ARMv7, unless a
  Raspberry Pi target becomes a goal.
- IMU fusion, which belongs in a finder application.
- The gRPC server that passes images through shared memory.
- olive-solve's own bugs:
  - Its key packing assumes fewer than 256 bins, though its comment says
    10 bits.
  - A distortion refit overwrites the undistorted centroids that later
    candidates reuse.
  - It never re-matches stars after its final fit.

## Problems found in seiza

Fixed with this note:

- `seiza/src/lib.rs` said blind solving was out of scope.
- `diverse_brightest` said it keeps `count / 4 + 1` stars per cell; the
  code keeps `count / 8 + 1`.
- The `solve.rs` header promised outlier clipping; refinement only drops
  pairs farther apart than the match radius.

Still to fix:

- Blind solving buckets its votes by `(ra · 2) as i64`, with no wrap at
  0/360 degrees and no `cos(dec)` scaling, so a field across RA 0 splits
  its votes between buckets that never meet. This goes with item 1, since
  it changes how hypotheses rank.
- Python's `Solution.center` is the sky position of pixel
  `((w - 1) / 2, (h - 1) / 2)`; the CLI, worker and C ABI use `(w / 2, h / 2)`,
  half a pixel away.
- The C ABI's solve JSON gives CRPIX counted from 0, while the worker and
  FITS headers count from 1.
