# Restoration comparison on the AstroBin telescope corpus

## Purpose

This is a small feasibility sample from `/Volumes/astrobin/_ByTelescope`, not a
benchmark claim. It compares two settings of the classical prototype with the
installed BlurXTerminator AI4 model on real integrated images and establishes a
repeatable review convention for later model work.

The conservative result used damped Richardson-Lucy with four iterations, a
35% blend, a `0.001` channel-range noise fraction, and a maximum multiplicative
correction of `2`. The stronger classical comparison used eight iterations, a
65% blend, a `0.0005` noise fraction, and a maximum correction of `3`. It is a
parameter-sweep result, not a recommended default. The Gaussian PSF FWHM was
measured from round, unsaturated stars in the input.

BlurXTerminator 2.1.5 used its installed AI4 defaults: automatic PSF, `0.63`
stellar sharpening, `0.05` halo adjustment, and `0.50` nonstellar sharpening.
All panels show the same 1200 by 1200 pixel crop, resampled identically, with
the input-derived display stretch applied to all four images. **Left to right:
input, conservative Seiza, strong classical Seiza, and BlurXTerminator.**

## Results

| Telescope | Object/filter | Input FWHM | Conservative | Strong classical | BlurXTerminator | Background sigma input / conservative / strong / BXT |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| C925 | NGC 6543 / HA | 7.358 px | 7.088 px (-3.7%) | 6.400 px (-13.0%) | 2.691 px (-63.4%) | 0.00026130 / 0.00026126 / 0.00026133 / 0.00026120 |
| SpaceCat61 | Sh2 119 / HA | 2.030 px | 1.719 px (-15.3%) | 1.359 px (-33.0%) | 1.371 px (-32.5%) | 0.00012518 / 0.00012515 / 0.00012591 / 0.00012248 |
| Askar107PHQ | NGC 7000 / HA | 3.293 px | 2.976 px (-9.6%) | 2.393 px (-27.3%) | 1.536 px (-53.4%) | 0.00026709 / 0.00026697 / 0.00026660 / 0.00026528 |
| Radian61 | NGC 7000 / Hydrogen-alpha | 3.636 px | 3.290 px (-9.5%) | 2.595 px (-28.6%) | 1.502 px (-58.7%) | 0.00077916 / 0.00078029 / 0.00079114 / 0.00077909 |

Star selection is run independently on each result and the accepted count can
change substantially, especially when a model alters star shape. The FWHM
figures are robust second-moment estimates, not matched photometric PSF fits, so
they are directionally useful but should not be treated as instrument
characterization or proof of recovered ground truth. Background values are in
normalized linear sample units.

BlurXTerminator changed total image flux by `-0.004%` (C925), `-0.140%`
(SpaceCat61), `-0.307%` (Askar107PHQ), and `-0.150%` (Radian61). The latter
three outputs reached PixInsight's normalized upper bound of `1.0`, so some
bright-core clipping is part of these particular results. Seiza renormalized
each channel and retained values above `1.0` in its Float32 FITS output.

The strong classical pass numerically matches BlurXTerminator on SpaceCat61,
but it also raises output peaks to `1.887`, `1.956`, and `1.797` in the
SpaceCat61, Askar107PHQ, and Radian61 images respectively. More importantly,
the shared-stretch Radian61 crop exposes dark rings around stars. Twelve- and
sixteen-iteration trials made star selection unstable and raised background
noise by roughly 3-6% on that field. This is the practical limit of tuning the
current circular Gaussian model: a lower FWHM alone is not evidence of a better
restoration.

The explicit asinh black/white points were `0.00430/0.02000` for C925,
`0.00100/0.00771` for SpaceCat61, `0.00090/0.00337` for Askar107PHQ, and
`0.05930/0.08143` for Radian61. Every panel used strength `10`.

### C925: NGC 6543, HA

![C925 NGC 6543 input, conservative Seiza, strong classical Seiza, and BlurXTerminator](../images/deconvolution/c925-ngc6543-ha-before-after.jpg)

The relatively wide measured PSF produces the smallest aggregate improvement
under Seiza's conservative damping. The stronger pass recovers a little more,
but remains far from BlurXTerminator. This field is the clearest evidence that
parameter strength cannot compensate for a mismatched Gaussian PSF.

### SpaceCat61: Sh2 119, HA

![SpaceCat61 Sh2 119 input, conservative Seiza, strong classical Seiza, and BlurXTerminator](../images/deconvolution/spacecat61-sh2-119-ha-before-after.jpg)

The strong classical pass reaches nearly the same aggregate FWHM as
BlurXTerminator here while retaining the broad nebular structure at the shared
stretch. This is the favorable case, but future evaluation still needs to match
and measure the same stars rather than independently selected populations.

### Askar107PHQ: NGC 7000, HA

![Askar107PHQ NGC 7000 input, conservative Seiza, strong classical Seiza, and BlurXTerminator](../images/deconvolution/askar107phq-ngc7000-ha-before-after.jpg)

The stronger Seiza result closes part of the stellar-width gap without matching
BlurXTerminator's much larger change in stellar cores and nonstellar dust
detail. The learned result is a useful external reference, not evidence that
every enhanced small structure is known to exist in the underlying scene.

### Radian61: NGC 7000, Hydrogen-alpha

![Radian61 NGC 7000 input, conservative Seiza, strong classical Seiza, and BlurXTerminator](../images/deconvolution/radian61-ngc7000-ha-before-after.jpg)

The strong classical column shows the failure mode that the summary FWHM misses:
dark rings appear around many stars and background noise increases. The
conservative result is safer; BlurXTerminator shrinks stars much more strongly
without the same visible rings. This source has both `EXPOSURE=90` and
`EXPTIME=901` in its inherited headers, so the corpus record must not silently
choose one as authoritative training metadata.

## Source provenance

The evaluated sources, relative to `/Volumes/astrobin/_ByTelescope`, were:

```text
C925/_Process/2026/NGC 6543/master/
  masterLight_BIN-1_6248x4176_EXPOSURE-300.00s_FILTER-HA_mono_autocrop.xisf
SpaceCat61/Sh2 119/process_08022025/master/
  masterLight_BIN-1_6248x4176_EXPOSURE-300.00s_FILTER-HA_mono_autocrop.xisf
Askar107PHQ/_Process/Stacked/North American - 20260131/master/
  masterLight_BIN-1_6248x4176_EXPOSURE-300.00s_FILTER-HA_mono_autocrop.xisf
Radian61/NGC7000/2025-07-18/Siril/
  Target-Hydrogen-alpha-session_1.fits
```

The XISF images were decoded as mono Float32 with `xisf` 0.9.7 and written by
Astropy 6.0.1 to temporary Float32 FITS files while copying object, filter,
instrument, telescope, exposure, pixel-size, focal-length, coordinate, and
available PSF/noise cards. BlurXTerminator was run in PixInsight 1.9.4 on those
same linear FITS inputs and saved to temporary Float32 FITS outputs. Temporary
inputs and full restored outputs are not committed; only the derived review
crops are present here.

The BlurXTerminator FITS saves did not contain a process-specific keyword or
history record, so its version, AI model, and parameters are recorded externally
in this note. A Seiza learned operation should instead make those fields part of
the output provenance contract.

## Reproduction

For each converted/source FITS, measure representative unsaturated-star FWHM,
then run:

```text
seiza deconvolve input.fits --output restored.fits \
  --psf-fwhm MEASURED_PIXELS --iterations 4 --amount 0.35 \
  --noise-fraction 0.001 --max-correction 2
```

The opt-in strong classical comparison used:

```text
seiza deconvolve input.fits --output restored-strong.fits \
  --psf-fwhm MEASURED_PIXELS --iterations 8 --amount 0.65 \
  --noise-fraction 0.0005 --max-correction 3
```

It should only be used as an evaluation setting with matched stretches and
ringing checks. It is not a general-purpose preset.

The BlurXTerminator instance was configured in PixInsight as:

```javascript
var process = new BlurXTerminator;
process.ai_file = "BlurXTerminator.4.mlpackage";
process.correct_only = false;
process.correct_first = false;
process.nonstellar_then_stellar = false;
process.lum_only = false;
process.sharpen_stars = 0.63;
process.adjust_halos = 0.05;
process.nonstellar_psf_diameter = 0.00;
process.auto_nonstellar_psf = true;
process.sharpen_nonstellar = 0.50;
process.executeOn(ImageWindow.activeWindow.mainView);
```

Resolve black and white points from the *input* once, and use the same explicit
stretch for every image:

```text
seiza stretch input.fits --output before.png asinh \
  --black INPUT_BLACK --white INPUT_WHITE --strength 10
seiza stretch restored.fits --output after.png asinh \
  --black INPUT_BLACK --white INPUT_WHITE --strength 10
seiza stretch restored-strong.fits --output strong.png asinh \
  --black INPUT_BLACK --white INPUT_WHITE --strength 10
seiza stretch bxt.fits --output bxt.png asinh \
  --black INPUT_BLACK --white INPUT_WHITE --strength 10
```

The evidence supports a lightweight experimental pass and confirms that a
learned model can provide materially stronger restoration. It does not
demonstrate recovery of ground truth, validate every optical PSF, or justify
training directly on BlurXTerminator outputs. The
[model-based restoration plan](../design/ml-restoration-training.md) defines the
stronger dataset and evaluation contract.
