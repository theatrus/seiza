# Color composition from mono stacks

Seiza composes calibrated mono stacks into RGB images. The pure Rust and Python
color functions consume already aligned `LinearImage`/array values and return
three-channel `f32`. The CLI first registers every non-reference filter stack
onto L (LRGB), R (RGB), or H-alpha (narrowband) using the same bounded
star/similarity registration and resampling as the stacker. `--no-register`
is the explicit fast path for masters already on one pixel grid. Display
stretching remains outside the stack accumulator.

## Input scaling

The default quick-look path independently maps each filter's robust 0.1% and
99.5% levels to `[0, 1]`. Percentiles are estimated from at most one million
samples per channel, bounding scratch memory for large frames. This is an
affine black/scale match for visualization, not photometric color calibration.
Use `ColorNormalization::None` or CLI `--normalization none` when the masters
are already background-subtracted and intensity-matched. Because the published
Foraxx expressions operate on normalized display values, its inputs must also
already lie in `[0, 1]` when normalization is disabled; Seiza rejects
sensor-unit Foraxx inputs instead of silently clipping them. The default
percentile normalization handles those inputs directly.

Embedding applications may instead apply an independent ordered stretch stack
to each aligned mono channel, then set `ColorOptions::input_transfer` to
`ColorTransfer::DisplayReferred` with normalization disabled. RGB, LRGB, and
static narrowband compositions preserve that transfer in their result. Foraxx
uses the prepared values directly and skips its shared internal Auto-MTF pass,
so different H-alpha, OIII, and SII parameters are neither collapsed nor
applied twice. The caller remains responsible for ensuring finite input samples
are display-scaled consistently.

Composition reads prepared values directly from caller-owned input planes and
allocates one three-plane output; it does not materialize normalized copies of
every channel. Percentile estimation temporarily holds at most `max_samples`
finite values for one channel at a time.

## Cropping to the shared area

Registering one filter stack onto another leaves `NaN` where the source frame
did not reach, so an uncropped composition keeps blank edges. `ColorOptions::crop`
(CLI `--crop`, Python `crop=`) trims the result to the area every channel
covers. A pixel counts as covered when every sample of every required channel
there is finite, so the kept region is the inner area common to all of them:

| Mode | Keeps |
| --- | --- |
| `none` (default) | the whole reference grid, blank edges included |
| `bounds` | the bounding box of the covered pixels |
| `inscribed` | the largest rectangle every channel covers in full |

Dithered or drifting channels overlap in a rectangle, and `bounds` is enough
for them. Rotated overlaps — a meridian flip, a rotator, two optical trains —
leave uncovered corners inside that box, and only `inscribed` removes them.
Use `inscribed` when a later step cannot handle `NaN`. Both modes read edges
and interior gaps alike, so an uncovered pixel in the middle of the frame
bounds `inscribed` just as an edge does.

The crop is chosen before normalization, and percentile levels are then
estimated from the kept region alone. Every channel is scaled from the same
patch of sky, and a bright edge that the crop discards cannot set a white
point. Composition maps output pixels back into the caller's full-size input
planes, so cropping copies no channel.

The written FITS moves `CRPIX` by the crop origin and records that origin in
`SEIZACRX`/`SEIZACRY`. `CRPIX` is the only WCS card a translation changes: the
rotation matrix, the reference world coordinate, and SIP distortion terms are
all expressed relative to it, so the cropped image keeps the reference
solution.

A crop also reports what each channel covered: its own covered bounding box
and pixel count, and how far its coverage center sits from the median center
of the others. A channel past the larger of 32 pixels and 2% of the frame is
flagged `off_center` — the one that pulled the crop in. With fewer than three
channels there is no majority to disagree with, so nothing is flagged. The
Rust result carries this as `ColorComposition::crop`, the CLI prints a warning
per flagged channel, Python exposes the same measurements through
`seiza.crop_report`, and native front-ends read them as JSON from
`seiza_color_crop_report_json`.

`crop_report` takes borrowed `ChannelSamples` rather than owned images, so a
host asking about its own buffers copies nothing. The C ABI hands its channel
pointers straight through.

## RGB and linear LRGB

RGB directly interleaves the prepared red, green, and blue masters. LRGB uses
linear-light luminance

```text
Yrgb = 0.2126 R + 0.7152 G + 0.0722 B
Yout = (1 - weight) Yrgb + weight L
RGBout = RGB * Yout / Yrgb
```

Scaling the triplet retains its linear RGB chromaticity while replacing its
luminance. A zero-chrominance pixel becomes neutral at `Yout`. A weight of one
uses all source L; zero is an RGB no-op.

Super-luminance mode instead adds every prepared channel into the target
luminance:

```text
Ysuper = L + R + G + B
RGBout = RGB * Ysuper / Yrgb
```

The selected normalization is applied before the sum. The RGB triplet is still
scaled together, retaining its chromaticity, but the additive linear `f32`
result can exceed one. This is a quick-visualization and stack-depth option,
not a photometric calibration. Rust callers select it with
`combine_super_lrgb`; the CLI and Python APIs use luminance mode `super`.

The same target works without a luminance stack. Synthetic super-luminance
composes RGB alone and scales the triplet to `R + G + B`, as if one luminance
exposure had collected every channel's light:

```text
Ysuper = R + G + B
RGBout = RGB * Ysuper / Yrgb
```

Rust callers select it with `combine_super_rgb`; the CLI (`color rgb`) and
Python (`combine_rgb`) use luminance mode `super`, and the FITS output is
marked `SEIZACLR='SUPER-RGB'`.

This follows PixInsight's documented model rather than reproducing every
`LRGBCombination` control. PixInsight staff describe CIE XYZ separation for
linear images, on-demand luminance/chrominance rather than a stored L channel,
and a luminance source-to-target ratio. PixInsight also has a configurable RGB
working space, midtones transfer functions, saturation adjustment, and
chrominance noise reduction. Seiza's quick-look path intentionally fixes the
linear-sRGB/Rec.709 Y coefficients above and does not claim pixel identity with
PixInsight. See the [PixInsight staff explanation of linear XYZ and luminance
ratios](https://www.pixinsight.com/forum/index.php?threads/l-l-rgb-combines.1092/).

## Narrowband palettes

The direct palettes are static linear-light assignments. A three-letter name
lists the physical filter mapped to red, green, and blue:

| Palette | Red | Green | Blue |
| --- | --- | --- | --- |
| SHO | SII | H-alpha | OIII |
| SOH | SII | OIII | H-alpha |
| HSO | H-alpha | SII | OIII |
| HOS | H-alpha | OIII | SII |
| OSH | OIII | SII | H-alpha |
| OHS | OIII | H-alpha | SII |
| HOO | H-alpha | OIII | OIII |

`NarrowbandMatrix` is the library extension point for arbitrary static mixes.
Each output channel has explicit SII, H-alpha, and OIII coefficients.

## Foraxx

Foraxx is a dynamic display palette, not a linear-light matrix. Its author
defines per-pixel factors on stretched `[0, 1]` channels. Seiza therefore
normalizes each input, applies its existing PixInsight/N.I.N.A.-family
median/MAD midtones transfer, and then evaluates the published expressions.
The default puts the median at 0.2 and clips shadows 2.8 normal-equivalent MADs
below it. These dynamic-palette controls live in `ForaxxOptions`, separately
from the normalization-only `ColorOptions`, so linear compositions do not
depend on irrelevant display settings:

```text
fO = OIII ^ (1 - OIII)
fG = (OIII * Ha) ^ (1 - OIII * Ha)

Foraxx-SHO:
  R = fO * SII + (1 - fO) * Ha
  G = fG * Ha  + (1 - fG) * OIII
  B = OIII

Foraxx-HOO:
  R = Ha
  G = fG * Ha  + (1 - fG) * OIII
  B = OIII
```

The FITS writer records `SEIZATRF='DISPLAY'` for Foraxx and for any composition
whose inputs were explicitly marked display-referred. Linear inputs remain
`SEIZATRF='LINEAR'` for RGB, LRGB, super-LRGB, super-RGB, direct palettes, and
custom matrices. It records `SEIZACLR='SUPER-LRGB'` or `SEIZACLR='SUPER-RGB'`
for super-luminance output and the composition or palette name otherwise. A display-referred preview is written
directly; a linear-light preview receives the normal display-only asinh stretch.
The original equations and the stretched-input requirement come from
Ludo/ForaxX's [Dynamic narrowband combinations with
PixelMath](https://thecoldestnights.com/2020/06/pixinsight-dynamic-narrowband-combinations-with-pixelmath/).

## Real-data validation

The CLI path was exercised in release mode against Sh2-132 data from
`Askar107PHQ/_Source/2025/Sh2 132`: twelve 300-second light frames for each of
H-alpha, OIII, and SII. Each channel was calibrated with the available
per-filter flats and a 300-second dark, then stacked independently before color
composition.

| Stage | Result |
| --- | --- |
| H-alpha stack | 12 accepted, 0 rejected |
| OIII stack | 12 accepted, 0 rejected; approximately 179.6-degree meridian-flipped frames accepted |
| SII stack | 12 accepted, 0 rejected; approximately 179.6-degree meridian-flipped frames accepted |
| OIII to H-alpha registration | 0.253 px RMS, 11.8 px drift, -0.007 degrees, 150 matched stars |
| SII to H-alpha registration | 0.212 px RMS, 6.0 px drift, -0.023 degrees, 193 matched stars |

Both direct SHO and Foraxx-SHO produced full-resolution 6248 by 4176 FITS and
PNG outputs. The FITS metadata marked direct SHO as `LINEAR` and Foraxx-SHO as
`DISPLAY`. Cropped, downscaled versions of those PNG previews are shown in the
[main README](../../README.md#color-from-mono-stacks); the large source and
intermediate FITS files are intentionally not checked in.

## Boundaries

- Color composition does not alter or feed back into a live stack.
- It does not infer filter identity from filenames or FITS headers.
- It does not perform gradient removal, spectrophotometric calibration, star
  removal, hue curves, or chrominance denoising.
- FITS output preserves WCS cards from the selected reference input, with
  `CRPIX` moved to a crop's own grid. CLI inputs are resampled to the reference
  grid by default; library callers must align arrays.
- Pixels outside any required registered channel are `NaN` in all three FITS
  planes and render black in the quick-look PNG, avoiding colored crop borders.
  `--crop` removes those edges instead of masking them.
- Cropping trims to a rectangle. It does not reproject, rescale, or pad a
  channel to make one fit.
- PNG output is for fast visual assessment; floating-point FITS remains the
  editable result for linear RGB/LRGB and direct palettes.
