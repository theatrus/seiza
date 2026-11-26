# Binned stacking

Status: phases 0, 1 and 2 implemented (#306, #313, and the PR that adds
this line). Where they depart from this design, see
[As built (phase 1)](#as-built-phase-1) and
[As built (phase 2)](#as-built-phase-2).

## Summary

Add an integer bin factor `b` (1, 2, 3 or 4) to `StackOptions`. Each light
is decoded, calibrated, defect-corrected, cosmetically filtered and debayered
at native resolution, as today, and then area-mean binned by `b` before star
detection. Registration, resampling, normalization, rejection and
integration then run on 1/b² of the pixels. The stack's grid is the binned
reference's grid. The ledger keeps everything a later pass needs to
reintegrate the same frames at full resolution: the frames' native
transforms follow exactly from the binned ones. That pass re-detects stars
at native resolution and refines each transform from its binned seed. It
does not search for the transform again.

On a 26 MP mono frame (16-core Ryzen 5950X), measured per-frame latency
falls from 311 ms to 159 ms at b=2 and to 118 ms at b=4. On a 2×2 mosaic
canvas (94 MP) it falls from 400 ms to 183 ms and 129 ms, and the canvas's
memory falls by b². Decoding and calibrating at native resolution are fixed
costs, so the per-frame gain is about 2-3×, not b². The memory gain is the
full b², which matters most for a mosaic that would not fit at native
resolution. See [Measurements](#7-measurements).

Recommendations, in order of importance:

1. Express every pixel-valued option and constant in native pixels, and
   convert it at one place in the core. Hosts never scale thresholds
   themselves.
2. First collapse the four file-reading preparation paths into one helper.
   Then add binning there, and in the two entry points for prepared
   pixels.
3. Keep `RegisteredFrameMapping` and the context format as they are: one
   grid, the working one. Record the bin factor beside them, and derive
   native transforms by an exact formula. This departs from the brief's
   "record transforms in native coordinates". It serves the same purpose
   with no format change, and replay stays bit-identical for free (see
   §2.6).
4. Correct the naive coordinate conversion. Scaling the translation by `b`
   is wrong whenever the frame is rotated. After a meridian flip, the error
   is `b - 1` native pixels.
5. Ship mono and demosaic-then-bin OSC first. Add a superpixel demosaic as
   phase 2: 2.7× instead of 1.9× for OSC at b=2, but it needs per-channel
   offsets in the resampler.

## 1. Where binning sits

### Today's preparation paths

The same sequence (check the masters, calibrate, cosmetic filter, debayer)
is written out five times:

| Path | Location |
| --- | --- |
| Pipelined push | `seiza-stacking/src/pipeline.rs:868` `prepare_decoded` |
| Sequential push | `seiza-stacking/src/stack.rs:1369` `LiveStacker::push` |
| Reference, and expansion planning | `seiza-stacking/src/stack.rs:1776` `prepare_reference` (called from `stack.rs:602`, `stack.rs:670`, `expand.rs:841`, `expand.rs:906`) |
| Replay | `seiza-stacking/src/replay.rs:2197` `read_source` + `replay.rs:2232` `read_admitted` |
| Prepared pixels | `stack.rs:1469` `push_linear`, `stack.rs:858` `from_linear` (the host prepared them) |

Phase 0 is to route the four file paths through `prepare_reference`
(renamed `prepare_light`), which already runs exactly this sequence. Binning
then goes in once. The parity tests pin pipelined, sequential and replayed
preparation together bit for bit
(`a_pipelined_stack_is_bit_identical_to_a_sequential_one`,
`reintegration_matches_with_and_without_the_frame_cache`,
`retained_frames_replay_bit_identically_without_reading_sources`), so a
missed path fails loudly. The defect maps on `claude/learned-defect-maps`
apply in the same native block. That branch is not merged and this design
leaves it alone. It only requires that they run before `bin`.

### Order

```
decode → declared-range rescale → validate against masters (native headers)
→ bias/dark/flat (native) → learned defect map (native, PR pending)
→ cosmetic impulse filter (native) → debayer (native)
→ BIN b → rewrite headers → star detection → registration → resample
→ normalize → integrate
```

Calibration masters, defect maps and the cosmetic filter work on single
photosites and on native-sensor statistics, so they stay native. A hot pixel
binned into a 4×4 block is still 1/16 of its amplitude above the block's
mean, which is no longer an impulse a spatial filter can find. Matching the
masters reads the light's own `XBINNING`, so it happens before the header
rewrite. A host's residual-flat patch (`seiza-cabi/src/lib.rs:11979`) is
also applied at detector sampling before the frame is pushed, so binning
does not affect it.

### The kernel

Use an area mean over each b×b block, ignoring NaN, and drop the remainder
rows and columns:

- `out[x, y] = mean of the finite samples in [b·x, b·x + b) × [b·y, b·y + b)`,
  and NaN only when none are finite.
- Output size `⌊W/b⌋ × ⌊H/b⌋`. Every block is complete, so every output
  pixel's centre is exactly `b·x + (b - 1)/2`. A partial edge block would
  put its centroid up to `(b - 1)/2` native pixels off its nominal centre.
  Common sensors divide evenly (IMX455 6248×4176, IMX571 6244×4168 at b=2
  and 4). A remainder costs at most `b - 1` edge columns, and the
  full-resolution pass gets them back.
- Use the mean, not the sum, so samples keep their ADU scale. Saturation
  levels, declared ranges, `normalize_for_detection` and the
  normalization-gain gates (`stack.rs:330-331`) then mean what they mean
  today. A block that is only partly saturated averages below the ceiling.
  Detection already excludes stars with `peak >= 0.9` (normalized) where
  that matters (`reference.rs:102`).
- Do not decimate. Taking every b-th pixel aliases and throws away the
  factor-b noise gain that is half the point.

`LinearImage::binned(b)` in `image.rs` returns the image, and
`FitsFrame::binned(b)` also rewrites the headers (§5). Both run in parallel
over rows. Measured cost: 6 ms (b=2) and 4.5 ms (b=4) for a 26 MP mono
frame, and 18 ms for 3-channel b=2.

### Bayer / OSC

`image.rs:283` offers `Vng`, `Mhc` and `Bilinear`. There is no superpixel
debayer. The two candidates are:

| | Demosaic, then bin | Superpixel (2×2 RGGB cell → 1 RGB pixel) |
| --- | --- | --- |
| Bin factors | any | even only (b=4 is superpixel, then a 2×2 mean) |
| Channel sites | all three at the block centre | G at the centre; R and B ∓0.5 native px diagonally (∓0.25 binned px at b=2) |
| Cost, 26 MP | VNG 100 ms + bin 18 ms | 10 ms |
| Colour against a full-resolution stack | same demosaic, so same colour | different interpolation |
| OSC per frame at b=2 (est.) | 339 ms (1.9×) | 231 ms (2.7×) |

Recommendation: demosaic then bin in phase 1. It is exact and reuses the
demosaic the user chose, and a promoted stack matches its preview in
colour. Superpixel comes in phase 2 as `Demosaic::Superpixel`, accepted
only with an even `b`. The resampler then has to apply each channel's
sub-pixel offset. Bayer drizzle already samples per photosite
(`resample_region_photosites`), so the shift has a home, but today the
resampler takes one inverse map per image. Without that work, superpixel
stacks keep R and B offset by about 0.35 binned px diagonally, which shows
as coloured star fringes.

`CfaIntegration::BayerDrizzle` integrates single photosites, which binning
destroys. `StackOptions::validate` refuses it with `bin_factor > 1`. Drizzle
during reintegration still reads native mosaics (§5.3).

## 2. Coordinates

Pixel centres are whole numbers (AGENTS.md). Native pixel `X` lies in binned
pixel `x`:

```
x = (X + 0.5)/b - 0.5        X = b·x + c,  c = (b - 1)/2
```

Call the binned-to-native map `C_b(x) = b·x + c`, applied to each axis.
Every geometric object converts by conjugation, `T_native = C_b ∘ T ∘ C_b⁻¹`,
which is exact in real arithmetic.

### 2.1 Similarity (`registration.rs:12`, source → reference)

`T(x) = s·R·x + t`. Conjugating gives:

```
s_n = s,   θ_n = θ,
t_n = b·t + (I - s·R)·(c, c)
```

Scaling the translation alone (`t_n = b·t`) is off by `(I - sR)·c`. That
term is small near the identity, but after a meridian flip (`R ≈ -I`) it is
`2c = b - 1` native pixels: 1 px at b=2 and 3 px at b=4. The throwaway
harness (§7) uses the formula above, and its converted transforms land
within 0.01-0.03 px of truth at b=2 and b=4 on well-sampled stars.

### 2.2 Polynomial warp (`registration.rs:336`, reference → source)

The warp evaluates `Σ x_k · term_k((X - center)/scale)`. Conjugating gives:

```
center_n = b·center + c,   scale_n = b·scale,
x_n[0] = b·x[0] + c,       x_n[k≥1] = b·x[k]    (same for y)
```

The conversion is exact and needs no refit, for order 1, 2 or 3.

### 2.3 Radial lens distortion (`lens.rs:30`, inside a warp)

`distort(p) = center + (p - center)·(1 + k1·q + k2·q²)`, with
`q = |p - center|²/radius²`. Conjugating gives:

```
center_n = C_b(center),  radius_n = b·radius,  k1, k2 unchanged,
width_n = b·width,  height_n = b·height
```

`LensDistortion::validate` (`lens.rs:105`) ties `radius` to
`0.5·hypot(width, height)`, and `b·width` keeps that true. If the native
sensor has remainder columns, the record describes the `b·w × b·h` grid it
was measured on, which validation accepts. The warp applies
`distort ∘ polynomial`, and `C∘D∘P∘C⁻¹ = (C∘D∘C⁻¹)∘(C∘P∘C⁻¹)`, so each part
converts on its own.

### 2.4 Normalization map (`normalization.rs`)

Tiles are `tile_size` square, and `axis_weights` (`normalization.rs:861`)
interpolates with the pixel-centre convention `(coord + 0.5)/tile - 0.5`.
With `width_n = b·width`, `height_n = b·height` and
`tile_size_n = b·tile_size`, every native pixel reads the same tile weights
as the binned pixel that contains it. The gain and offset arrays carry over
unchanged. A global map is one tile.

### 2.5 Canvas geometry (`expand.rs`)

Panel `ReferenceRegion`s are integers and scale by `b`. The anchor's WCS is
shifted by integer canvas growth (`expand.rs:1989`
`shift_reference_pixel`), and its `CRPIX` follows §5.1. Planned
registrations convert as in 2.1-2.3. `focal_length_pixels`
(`expand.rs:2005`, `FOCALLEN·1000/XPIXSZ`) comes out right once `XPIXSZ`
is rewritten ×b (§5.2).

### 2.6 What the ledger stores

The brief asks that transforms be recorded in native coordinates. Since the
conversion is an exact, invertible conjugation, both choices hold the same
information. They differ in cost:

| | Store the working grid + `bin_factor` (recommended) | Store native |
| --- | --- | --- |
| `RegisteredFrameMapping` | unchanged: transform, warp, normalization and reference size all on one grid | mixed: transform and warp native, normalization and size binned |
| Context format | unchanged. `bin_factor` rides in the serialized options, and older readers refuse it (`deny_unknown_fields`) | the mapping's compact postcard layout (`mapping.rs:77`) changes meaning, so it needs a new section or a format bump |
| Bit-identical replay at the same `b` | free | needs a canonicalization step: derive the working transform from the native one before the live pass resamples, or the cached and uncached replays differ in the last bit |
| Full-resolution pass | `mapping.rebinned(b, 1)` | read directly |
| Public mapping consumers | unaffected | `integrate_registered_frames` and hosts, which keep the mapping JSON the C ABI returns with each frame's diagnostics (`seiza-cabi/src/lib.rs:629`), would misread a native transform paired with binned frames |

Recommendation: store the working grid. Add
`RegisteredFrameMapping::rebinned(from, to)` in `mapping.rs`, which applies
§2.1-2.4, and `Ledger::native_mapping(index)`. Reintegrating at any factor,
including b=1, is then one code path that takes a target factor. A reviewer
who wants native storage anyway takes on the right-hand column.

## 3. The full-resolution pass

### 3.1 Error budget

The simple rule is that a registration error of ε binned px is b·ε native
px. That holds for misfit such as uncorrected distortion. For centroid
noise it is pessimistic: a star's photons are the same at any binning, so
its centroid noise in binned px is about its native noise divided by b, and
in native px it is about the same. What binning adds is pixel-phase bias
once a star is undersampled (FWHM below about 1.5 binned px). Measured on
the synthetic fields (§7), as native-px error against truth:

| Case | b=1 | b=2 | b=4 |
| --- | --- | --- | --- |
| Field, FWHM 3.5 px | 0.003 rms / 0.014 max | 0.002 / 0.004 | 0.017 / 0.033 |
| 2×2 canvas, FWHM 3.5 px | 0.003 / 0.020 | 0.005 / 0.014 | 0.031 / 0.093 |
| Field, FWHM 2.0 px | 0.000 / 0.001 | 0.005 / 0.015 | 0.114 / 0.278, matches 196 → 146 |

The synthetic stars are clean, so these are lower bounds. The trend is the
finding: b=2 costs almost nothing, and b=4 on tight stars costs about 0.1-0.3
native px. Against the 0.24 px centroid-noise floor quoted at
`registration.rs:296`, that would visibly soften a final stack.

### 3.2 Refine, don't search

Every refinement step already exists:

- `Registrar::refine_similarity_stars(initial, stars)` (`registration.rs:884`)
  pairs stars within `match_tolerance_pixels` of where the seed puts them
  and refits. It does no triangle or translation search.
- `Registrar::warp_stars(transform, stars)` (`registration.rs:983`) fits the
  polynomial around the refined similarity, and `warp_stars_seeded`
  (`registration.rs:998`) pairs through a converted warp where a similarity
  misses at the edges of a wide field.
- On a canvas, expansion's `Pending::refine` (`expand.rs:1464`) uses
  `refine_similarity` (`registration.rs:1137`) to refit a placement found
  elsewhere.

For each admitted frame, the pass reads and prepares it at native
resolution, detects its stars once, and refines from `rebinned(b, 1)`. The
seed's error (≤ 0.3 px measured) is far inside the 2.5 px pairing radius.
Its floor is native star detection: 85-90 ms per 26 MP frame, against the
124 ms a full native registration costs. Refining by default costs little.
Skipping it is a reasonable option at b=2 when stars have FWHM ≥ 3 native
px.

### 3.3 Reused and redone

| | Full-resolution pass |
| --- | --- |
| Which frames are admitted, sources, calibration records, tolerances | reused from the ledger; no admission decision is revisited, as in `reintegrate` |
| Transforms and warps | seeds, converted exactly, then refined (3.2) |
| Lens distortion (canvas) | converted exactly (2.3); not refitted |
| Frame weights (inverse noise variance) | reused. White noise scales by 1/b in every frame alike, so the ratios hold. `frame_noise` is cheap if a reviewer prefers to measure again |
| Normalization | global gains and offsets reused; local maps re-expressed (2.4). Under `LocalBackground` the existing renormalizer (`replay.rs:432`) refits offsets against the integrated reference, as `reintegrate` does now |
| Rejection | redone. Fates are per output sample, and a native sample is not a binned one. This is the three-pass leave-one-out reintegration itself |
| Star detection | redone at native resolution (the refine floor) |
| Triangle and translation search | not run |
| Reference | the reference frame's own source, prepared at native resolution. Its stars feed `Registrar::new` once |

### 3.4 Mosaics

A single field refines each frame against the native reference. A canvas
has no native image to refine against. The binned canvas was built from
binned frames, and its panel windows (`expand.rs:162`) hold binned pixels.

Phase 1 promotes a canvas with converted transforms and no refine, and
reports the binned stack's median registration RMS × b as its expected
error. Phase 2 builds the native canvas the way planning does: first
integrate the panel-starting frames through their converted placements,
then refine every other frame against the windows of that native canvas.
This is open question 3.

### 3.5 API

Add `LiveStacker::reintegrate_full_resolution(&self, options: &BatchStackOptions, refine: bool, progress)`.
It returns a `BatchStackResult` on the native grid, `b·w × b·h`. It refuses
stacks whose `reintegration_unavailable()` gives a reason, and it is a no-op
equivalent of `reintegrate` at b=1. It needs a C export and a Python twin
(§6).

## 4. Thresholds in pixels

Policy: options and constants keep their native-pixel meaning. A
`Scale { b }` helper converts once, where the working grid is built:
lengths `÷ b`, areas `÷ b²`, with a floor where undersampling bites.
Measurements reported back (RMS, drift, diagnostics) are multiplied back to
native px. A host writes the same numbers at any bin factor. Fractions,
counts, sigmas and angles do not change.

| Threshold | Where | Default | Rule at bin `b` |
| --- | --- | --- | --- |
| Registration RMS gate | `stack.rs:326` (`acceptance.maximum_registration_rms_pixels`), checked `stack.rs:2009`, `expand.rs:2105` | 2.0 px | compare `rms·b` against it |
| Match tolerance | `registration.rs:671` (`match_tolerance_pixels`) | 2.5 px | `max(2.5/b, 1.0)` binned px. At b=4, 0.625 px would drop pairs to undersampling bias. The spatial index bin (`registration.rs:823`) and the translation-vote bin (`registration.rs:1837`, ×2) follow |
| Drift floor | `registration.rs:692` (`maximum_drift_pixels`) | 256 px | `÷ b` |
| Drift fraction | `registration.rs:694` | 0.15 | unchanged (relative to the binned frame) |
| Drift check | `registration.rs:951`, `1110` | | as the floor; report `drift·b` |
| Canvas drift override | `stack.rs:765` (diagonal of the canvas), `expand.rs:200` (2× panel diagonal) | | computed on the working grid; store ×b if it is kept as an option value |
| Scale and rotation gates, descriptor and scale tolerances | `stack.rs:327-328`, `registration.rs:669-670` | | unchanged (dimensionless) |
| Polynomial pairing radius | `registration.rs:658` `DISTORTION_PAIR_PIXELS` | 2.0 px | `max(2.0/b, 1.0)` |
| Polynomial clip floor | `registration.rs:1070` (`3·median + 0.05`) | 0.05 px | `÷ b` |
| Edge margin for centroids | `registration.rs:1145` `MARGIN` | 6 px | `max(⌈6/b⌉, 2)` |
| Detection background tile | `seiza/src/detect.rs:59` | 64 px | `max(64/b, 16)`. Registration builds `DetectConfig` at `registration.rs:1921` and can set it there |
| Detection minimum area | `detect.rs:61` | 3 px | `3` at b=1, `2` at b=2, `1` at b≥3. At b=4 with FWHM 2.0 native, matches fell from 196 to 146; this floor is why. Hot pixels were removed at native and averaged down by b² |
| Detection maximum area | `detect.rs:65` | 20 000 px | `÷ b²` |
| Elongation | `detect.rs` | 2.5 | unchanged; noisier on 1-2 px blobs (open question 5) |
| Local normalization tile | `normalization.rs:20,38`; ≥16 enforced at `stack.rs:250`, `normalization.rs:80,409,417,600`; Python's default 256 (`seiza-py/src/stacking.rs:77`) | host | `max(tile/b, 16)` |
| Canvas outlier floor | `expand.rs:62` `RESIDUAL_OUTLIER_FLOOR_PIXELS` | 0.3 px | `÷ b`. The factor (5× median, `expand.rs:58`) is relative. The 1.1-1.4 px and 0.04-0.20 px in its comment are measurements, not thresholds |
| Panel photometry pairing | `expand.rs:97` `PHOTOMETRY_PAIRING_PIXELS` | 1.5 px | `max(1.5/b, 0.75)` |
| Panel geometry error | `expand.rs:119` `MAXIMUM_GEOMETRY_ERROR_PIXELS` | 0.02 px | `÷ b` |
| Panel match chance | `expand.rs:262` (disc π·tol²) | | self-consistent: star density per px² and tolerance² both scale |
| Lens pairing | `lens.rs:266` `PAIRING_PIXELS` | [6, 4, 3, 2] px | `÷ b`, floor 1.0 |
| Lens improvement | `lens.rs:271` `MINIMUM_IMPROVEMENT_PIXELS` | 0.1 px | `÷ b`. At b=4, 0.025 binned px is close to the centroid noise, so the lens test may pass on noise; ratio and shift bounds (`lens.rs:270,276-277`) are relative. Open question 4 |
| Lens focal length | `expand.rs:2005` | FOCALLEN/XPIXSZ | correct once `XPIXSZ` is rewritten ×b |
| Overlap fraction | `stack.rs:2059` | | unchanged (both sides binned) |
| Crop off-centre floor | `crop.rs:91` `OFF_CENTER_PIXELS` | 32 px | `÷ b` (on stacked grids; the fraction at `crop.rs:94` is relative) |
| Drizzle scale | `drizzle.rs:29`, validated 1..=4 at `drizzle.rs:51` | 1 | relative to the working grid; scale `b` gives native size (§5.3) |
| Drizzle drop | `drizzle.rs:35` | 0.9 mono, 1.0 Bayer | relative to a source pixel. The drop shape follows the mapping's Jacobian (`drizzle.rs:795`), so native sources drop at 1/b binned px automatically |
| Drizzle tiles | `drizzle.rs:82,87,258` | 64, 16, 16 | internal tiling; unchanged |
| Cosmetic filter | `cosmetic.rs:68` | sigma-based | runs at native resolution, before `bin` |
| Defect maps | `claude/learned-defect-maps` | | native, before `bin` |
| Residual flat | `seiza-calibration/src/residual_flat.rs:39,45` (`smoothing_sigma` 2 px, `minimum_connected_pixels` 64) | | host applies it at detector sampling before pushing; unaffected |
| Reference scoring | `reference.rs:57` (own 2×2 half-resolution luma), `reference.rs:107` (64 px blocks) | | runs on raw frames before stacking; unaffected |
| Satellite trails | rejection in `batch.rs`, sigma-based | | per-sample significance holds. A 1 px trail in a b×b block adds peak/b over σ/b noise, the same σ multiple as native. `seiza-satellites` alignment works in its own pixels on outputs and needs only a correct WCS |
| Frame noise and SNR | `snr.rs` (`MAX_SAMPLED_ROWS` 512, `SIGNAL_FRACTION` 0.01) | | weights are ratios, so unaffected. Per-pixel noise falls by b, so absolute SNR is not comparable across bin factors; the depth curve's shape is. Report `bin_factor` with every `SnrSample` |
| Colour calibration apertures | `color_calibration.rs:1031` (FWHM search from 7 px), `840-844` | | measures FWHM itself on its input; adapts |

## 5. Outputs

### 5.1 WCS

When the reference is binned, its headers are rewritten once, so every
writer downstream (`write_linear_image_fits_f32`, `write_color_fits_f32`,
`drizzle_reference_headers`, expansion's anchor) sees binned-grid WCS.
`drizzle_reference_headers` (`fits.rs:615`) already maps a header to a
finer grid. Generalize it to `rescaled_reference_headers(headers, s: f64)`
and call it with `s = 1/b`:

- `CRPIX' = (CRPIX - 0.5)·s + 0.5`. FITS counts from 1, so
  `CRPIX = X + 1`, and this is exactly `x = (X + 0.5)/b - 0.5` plus 1.
- `CDELT` and `CD` scale by `1/s = b`.
- SIP: a coefficient of order `p + q` scales by `s^(1-p-q)`. `A_DMAX` and
  `B_DMAX` scale by `s`.
- Drizzling a binned stack at scale `b` composes back to the native WCS.

`seiza_compose::transform_wcs` (`seiza-compose/src/color_calibration.rs`, exported as
`seiza_wcs_transform_json`) refuses SIP under anything but a translation.
The header rewrite above handles SIP, so the stack never needs it.

### 5.2 Headers

Rewritten in `FitsFrame::binned`, after the masters are matched:

| Card | Value |
| --- | --- |
| `XBINNING`, `YBINNING` (and alias `CCDXBIN`, `fits.rs:12`) | × b |
| `XPIXSZ`, `YPIXSZ` | × b (SBFITSEXT defines them after binning; `focal_length_pixels` and solver scale priors read them) |
| `SEIZABIN` (new) | b, "software bin after calibration" |
| `NAXIS1/2` | binned size (written from the image) |

`SEIZABIN` lets a reader tell a software-binned stack from one taken with
camera binning. Neither card feeds calibration matching, which reads the
raw lights.

### 5.3 Drizzle

`reintegrate_drizzled` drops each frame's calibrated, undebayered native
pixels (`replay.rs:2243` `read_calibrated`) through its mapping. For a
binned stack the mapping it needs is native source → working reference,
`C_b⁻¹ ∘ T_native`, a similarity of scale `s/b`. `ForwardMap`
(`drizzle.rs:795`) builds each drop's shape from the mapping's Jacobian, so
native drops come out 1/b binned px across without other changes. Drizzle
scale `b` then gives a native-size image with rejection decided at binned
resolution. That is a valid "promote" when the transforms have been
refined. It rejects less than a native reintegration: a single-pixel
outlier of k σ shows as k/b σ in its binned sample, so 10σ at b=4 survives
a 3σ clip. Document it as a fast option, not the final one. Drizzle on an
expanded canvas stays unsupported (`replay.rs:1796`).

### 5.4 Live stacking

- The core bins inside `from_linear`, `push_linear`, `push`, the pipeline
  and the reference, so hosts push native frames. Prepared-pixel stacks
  (`FrameInputMode::PreparedOnly`) bin too, after the host's own
  preparation.
- Snapshots, views and exports are on the working grid. Hosts already read
  `width`/`height` from the snapshot, and display gets b² cheaper.
- `bin_factor` is part of `StackOptions`, so it enters the configuration
  fingerprint. A stack cannot change it midway; a different factor means a
  new stack or `reintegrate_full_resolution`.
- `retain_frames_for_reintegration` keeps working-grid frames, 1/b² of
  today's 313 MB per 26 MP colour frame.
- `max_canvas_megasamples` (`seiza-cabi/src/lib.rs:3481`,
  `seiza-py/src/stacking.rs:1201`) bounds memory, so it counts working
  samples. b=2 fits a mosaic of four times the sky.
- `FrameDiagnostics.registration_rms_pixels` and `registration_drift_pixels`,
  and `LiveStacker::effective_maximum_drift_pixels` (`stack.rs:849`),
  report native px.

## 6. API surface

### Integer factor, not a percentage

Use an integer. The area mean over b×b is exact, cheap, noise-optimal and
free of interpolation, and its coordinate map is the closed form in §2. A
fractional scale (70%) is a resampling, with correlated noise and a lossy
mapping. Hosts can still label b=2 and b=4 "50%" and "25%". The range is
1..=4, which matches drizzle's 1..=4, so any binned stack can be drizzled
back to native size.

### Core

```rust
pub struct StackOptions {
    // ...
    /// Area-mean bin each light b×b after calibration and before
    /// registration. 1 (the default) is not serialized, so existing options,
    /// fingerprints and contexts keep their exact bytes.
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub bin_factor: u32,
}
```

`validate` adds checks for `1..=4`, for `BayerDrizzle` only at `bin_factor
== 1`, and later for `Superpixel` only with an even `bin_factor`.

New public items: `LinearImage::binned`, `FitsFrame::binned`,
`RegisteredFrameMapping::rebinned`, `LiveStacker::bin_factor()`,
`LiveStacker::reintegrate_full_resolution`, and
`rescaled_reference_headers`, which generalizes `drizzle_reference_headers`
and keeps it as a wrapper.

### C ABI and Python (AGENTS.md: same PR)

| Core | C ABI | Python |
| --- | --- | --- |
| `StackOptions.bin_factor` | JSON key `"bin_factor"`. Stack options JSON is the core's own serde, snake_case, as `"match_tolerances"` and `"cfa_integration"` already are (`seiza-cabi/src/lib.rs:3146`) | `StackOptions(bin_factor=1)` keyword + `bin_factor` property (`seiza-py/src/stacking.rs:73`) |
| `LiveStacker::bin_factor` | `seiza_live_stacker_bin_factor` | `LiveStacker.bin_factor` |
| `LiveStacker::reintegrate_full_resolution` | `seiza_live_stacker_reintegrate_full_resolution` (cancel as an argument, progress callback as `seiza_live_stacker_reintegrate`) | `LiveStacker.reintegrate_full_resolution(*, refine=True, sigma_low, sigma_high, scratch_directory)` |
| `LinearImage::binned` | `seiza_image_bin` (host previews), or a `none` entry if hosts never call it | `seiza.bin_image` |
| `RegisteredFrameMapping::rebinned` | `none`: the ledger uses it inside reintegration | `none` |
| `rescaled_reference_headers` | `none`: the stack writers call it | `none` |

Python needs the stub in `seiza-py/seiza.pyi`, the README and
`seiza-py/tests/`. `.github/binding-parity.toml` needs the new function
twins (beside `seiza_live_stacker_reintegrate`, line 561) and `none`
entries with reasons for the two internal helpers. Run
`check_binding_parity.py`.

The C key is snake_case where AGENTS.md says camelCase. That follows
`StackOptions`' existing JSON rather than a request struct of the C ABI's
own; it is open question 6.

## 7. Measurements

A throwaway harness (a `seiza-stacking` example, not committed) renders synthetic 6248×4176 mono frames (IMX455 size) with about 6000
Gaussian stars per frame area, sky 500 ± 22 ADU, ±40 px dithers and ±0.3°
rotations, 8 frames. It bins with the kernel above and times single-frame
latency of each stage through the public API (`Registrar::register`,
`LiveStacker::push_linear`). Release build, Ryzen 9 5950X (16 cores, 32
threads), warm file cache. "Push − register" is resampling, normalization,
rejection and accumulation.

### Single field, mono, 32 threads (ms per frame)

| Stage | b=1 | b=2 | b=4 |
| --- | ---: | ---: | ---: |
| Decode (f32 FITS, native) | 52 | 52 | 52 |
| Calibrate, dark + flat (native) | 23 | 23 | 23 |
| Bin | — | 6 | 4.5 |
| Register | 124 | 44 | 28 |
| Push − register | 112 | 34 | 11 |
| **Total** | **311** | **159 (2.0×)** | **118 (2.6×)** |
| Cosmetic filter, if enabled (native) | +179 | +179 | +179 |
| Reference registrar / stacker open | 91 / 127 | 24 / 35 | 9 / 12 |
| Snapshot | 97 | 23 | 7 |

At 8 threads, closer to a laptop: 357 / 190 / 111 ms (1.9× and 3.2×), with
the cosmetic filter at 378 ms.

Registration at b=4 still costs 28 ms. Triangle and candidate scoring do
not depend on pixel count, so registration stops scaling past b=2.
Decoding and calibration stay native, so they set a floor of about 75 ms.

### 2×2 mosaic canvas, mono (11871×7934 = 94 MP canvas, 26 MP frames)

| Stage | b=1 | b=2 | b=4 |
| --- | ---: | ---: | ---: |
| Decode + calibrate (native) | 74 | 74 | 74 |
| Bin | — | 8 | 6 |
| Register against the canvas | 121 | 55 | 38 |
| Push − register (whole-canvas resample) | 205 | 45 | 11 |
| **Total** | **400** | **183 (2.2×)** | **129 (3.1×)** |
| Stacker open / snapshot | 503 / 357 | 131 / 86 | 35 / 22 |

The harness registers against the whole canvas with relaxed gates. It
stands in for `with_expanding_reference`'s per-frame cost, which also
resamples onto the whole canvas (`stack.rs:2042`). The canvas-footprint
proposal in a sibling session bounds that resample to each frame's
footprint without loss, about 112 ms here in place of 205. Measured against
that, binning's per-frame gain on a mosaic is about 2×, like a single field.
Its memory gain stays b²: the accumulator, the reference and the frame
cache shrink 4× and 16×.

### OSC (estimated)

Measured on 26 MP: VNG 100 ms, bin of the RGB result 18 ms, superpixel
10 ms. The other stages are estimated as ×3 the mono figures for the
channel-proportional ones. b=1 is about 635 ms. At b=2, demosaic then bin
is about 339 ms (1.9×) and superpixel about 231 ms (2.7×). At b=4,
demosaic then bin is about 251 ms (2.5×) and superpixel then 2×2 about
151 ms (4.2×).

### Full-resolution pass (estimated)

Per frame: native decode and calibration 75 ms, native detection 85-90 ms
(the refine floor), refine about 1 ms, then the same reintegration a b=1
stack's `reintegrate` runs. Previewing at b=4 and then promoting costs
about 118 + 160 ms of preparation per frame. A native live stack costs
311 ms. Binning trades total CPU for a fast, low-memory live stack. It does
not reduce the cost of a final stack.

### Caveats

The stars are clean point-sampled Gaussians with no distortion, uniform
seeing and flat sky. Real registration errors and matched counts will be
worse, so §3.1's figures are lower bounds. The quadratic model and the lens
pass were not exercised. Timings are single-frame latency with every
thread on one frame. The pipeline runs several frames at once, which hides
registration's serial portion.

## Implementation plan

1. **Phase 0**: one `prepare_light` for the four file-reading paths. No
   behaviour change; the existing parity tests hold.
2. **Phase 1 (one PR across core, C ABI and Python)**: `bin_factor`; the
   kernel; the header and WCS rewrite; the `Scale` threshold conversion
   (§4); `rebinned` and `reintegrate_full_resolution` with refine for
   single fields; canvas promotion without refine; mono and
   demosaic-then-bin OSC. Tests: the conjugation formulas against
   point-by-point conversion (with a half turn); bin then promote matching
   a native stack within tolerance on a synthetic set; contexts round-trip
   with `bin_factor`; a default-options context stays byte-identical; WCS
   round trip `bin → drizzle(b)`.
3. **Phase 2**: superpixel with per-channel offsets in the resampler; native
   canvas refinement for mosaics (3.4).

### Sequencing

Two branches in flight touch the same code. Start after both land:

- `claude/learned-defect-maps` changes all four preparation paths
  (`pipeline.rs:902`, `replay.rs:2221`, `stack.rs:1415`, `stack.rs:1807` on that branch):
  it replaces the cosmetic call with `StackOptions::repair`, and it adds
  fields to `StackOptions`. Phase 0 done first would force that PR to
  rebase onto a refactor of exactly its lines. Done after it, phase 0's
  `prepare_light` becomes validate → `apply` → `repair` → debayer, and
  phase 1 adds `bin` at the end.
- The canvas-footprint work (`claude/canvas-footprint-integration-225ad0`)
  bounds `prepare_frame`'s resample to each frame's footprint. Phase 1
  changes the same function's gates and geometry. Its lossless saving
  (205 → about 112 ms per frame on the measured canvas) also changes
  binning's case for mosaics. After it lands, binning's per-frame gain on
  a canvas is about 2×; the b² memory saving is unchanged.

## Decisions

Taken on the recommendations above; reopen any of them in review.

1. **Ledger representation**: keep the working grid. `RegisteredFrameMapping`
   stays on one grid, the stack records `bin_factor` in its options, and
   `RegisteredFrameMapping::rebinned` derives native transforms (§2.6). The
   brief's purpose, a full-resolution replay without searching for
   transforms again, holds without a context format change.
2. **Promotion output**: `reintegrate_full_resolution` returns a
   `BatchStackResult`, as `reintegrate` does. A native `LiveStacker` that
   keeps taking frames would need a native reference and accumulator built
   from the replay. Leave it until someone asks.
3. **Mosaic promotion in phase 1**: converted transforms, no refine. The
   result reports the expected error, the binned stack's median
   registration RMS × b, so a host can warn. The measured worst case was
   0.09 native px at b=4 on clean stars. Native canvas refinement is
   phase 2.
4. **Lens fit**: whenever `b > 1`, the lens pass measures stars on the
   native anchor and panel-start frames. There are only a handful of them
   (about 90 ms each), and its thresholds then keep their native meaning.
   The fitted distortion converts to the working grid by the inverse of
   §2.3. This keeps `MINIMUM_IMPROVEMENT_PIXELS` out of centroid noise at
   b=4.
5. **C JSON naming**: `"bin_factor"`, as `StackOptions`' own serde spells
   it beside `"match_tolerances"` and `"cfa_integration"`. No camelCase
   alias: one spelling per key.
6. **Factor 3**: allowed, range 1..=4. It costs nothing for mono or for
   demosaic-then-bin. Superpixel, when it comes, accepts even factors only.
7. **`LinearImage::binned` for hosts**: core-only in phase 1, with a `none`
   parity entry. Hosts' previews already downsample in the C ABI
   (`seiza-cabi/src/resample.rs:58,100`).
8. **Self-describing mappings**: yes. The human-readable mapping JSON
   carries an optional `bin_factor`, omitted at 1, as `warp` is omitted
   when absent. Compact binary encodings keep their layout, and the stack's
   options carry the factor there.

## As built (phase 1)

Phase 1 follows the decisions above, with these differences:

- **`rebinned` takes the target factor only.** A mapping records its own
  bin factor (decision 8), so `RegisteredFrameMapping::rebinned(to)` carries
  it from that factor to any that divides it. There is no
  `Ledger::native_mapping`; the full-resolution pass rebins the whole ledger.
- **`reintegrate_full_resolution` returns a `FullResolutionStack`**, not a
  bare `BatchStackResult`: the result, the reference headers carried to the
  finer grid (WCS rescaled, binning and pixel-size cards back to the
  sensor's, no `SEIZABIN`), how many frames were refined, and the median
  registration RMS in sensor pixels that decision 3 asks a canvas to report.
  The C call returns the snapshot and gives the last two through out
  parameters; Python returns a `(snapshot, refined, rms)` tuple. Each frame's
  live registration RMS is recorded in the ledger for this, but a context does
  not keep it, so a reopened stack that is not refined reports none.
- **Drizzling a binned stack was refused** in phase 1; phase 2 adds it.
- **Global normalization is reused, and it is noise-driven.** A global gain
  is the ratio of the reference's dispersion to a resampled frame's, which
  on a sparse sky is the noise's, and binning changes it: synthetic frames
  took gains of 1.47 at full resolution and 1.36 binned by 2. The promoted
  stack keeps the binned gains, as §3.3 says, so it differs from a native
  stack by that much in its stars. With `LocalBackground` (photometric gains)
  the promoted stack is within a fraction of one frame's sky noise of a
  native one, and with no normalization a refined promotion matches the
  native stack to the bit in at least 99% of its pixels.
- **Not converted:** `ColorCrop::OFF_CENTER_PIXELS` (colour composition does
  not know a stack's bin factor; its fractional bound dominates on large
  frames). The pipeline's in-flight memory estimate counts native frames for
  preparation and binned ones for integration. Phase 2 adds `SnrSample`'s
  `bin_factor`.
- **Detection's minimum area** uses the interim rule of §4 (3, 2 at b=2, 1
  from 3), and `Scale::detection` says so; open question 1 still stands.

## As built (phase 2)

- **Superpixel.** `Demosaic::Superpixel` takes each 2x2 cell's red, the
  mean of its greens and its blue, and needs an even bin factor: `develop`
  debayers a superpixel frame and bins it on by `b / 2`, and at an odd
  factor (the full-resolution pass, the native lens pass) uses VNG. Each
  prepared frame's channel offsets (`ChannelSites`, `±0.5 / b` working
  pixels for red and blue) follow from its Bayer layout. Mappings retain
  them as `channel_sites` in JSON and in context metadata so public crop
  extraction reproduces live registration. `resample_region_sited` reads each
  channel from its own site, one channel at a time, wherever a frame is resampled: registration, replay,
  and canvas placement. The integrated reference, an external reference and
  a canvas's anchor are not otherwise resampled, so each is resampled once
  through the identity onto its pixel centres; frames kept for
  reintegration leave a superpixel reference out, and replay prepares it
  the same way. Star detection still reads a frame's luminance at its raw
  sites, a centroid shift of about `0.21 d_R + 0.07 d_B` that frames of one
  orientation share and that a meridian flip doubles: 0.035 binned pixels
  at a factor of 2. On synthetic RGGB frames with a flip, red and blue
  centroids in the stack agree within 0.075 binned pixels; read from the
  pixel centres they were 0.29 apart.
- **Canvas refinement** does without the panel starts, which a context
  does not keep. Every frame is carried onto the native canvas through its
  converted mapping, scaled by its recorded gain and mean offset, and
  summed; each frame is then refit by `refine_similarity` against the mean
  of the other frames over its footprint (leave one out, so a panel of two
  dithers does not refine against itself). A frame whose mapping holds a
  warp, from the registration model or a lens, keeps it; so does one whose
  refit would move a corner further than the match tolerance. On a
  synthetic 2x1 mosaic binned by 2, the worst pairwise error against the
  frames' true offsets fell from 0.052 to 0.042 native pixels.
- **Drizzle of a binned stack** drops each frame's sensor pixels through
  `RegisteredFrameMapping::sensor_sourced`, `T ∘ C⁻¹` with the warp `C ∘ W`,
  so drops are a binned pixel's part across and a scale of `b` gives the
  sensor's resolution; rejection stays on the binned grid. Its stars land
  within 0.05·b pixels of where the full-resolution stack puts them.
- **`SnrSample::bin_factor`** comes from a new `LiveStacker::measure_depth`,
  which both bindings call; the free `measure_depth` of a bare `StackView`
  cannot know the factor and gives 1.

## Still open

1. **Detection on undersampled stars**: is lowering `min_area` with `b`
   (§4) enough, or should detection's elongation and area filters be
   bypassed for b ≥ 3? Settle it with real frames at FWHM ≈ 2 px before
   phase 1 fixes the rule. The synthetic b=4 case lost a quarter of its
   matches.
