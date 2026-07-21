# Model-based astronomical restoration plan

## Short answer

Before-and-after examples can train a useful model when each pair is the same
linear image, precisely registered, and accompanied by enough provenance to
explain the transformation. Expert-processed pairs alone teach a model to
imitate an operator's choices; they do not establish the missing scene as
ground truth. The safer plan is to use synthetic, physically parameterized
degradation for most supervised training, then use expert pairs for
fine-tuning, ranking, and realism checks.

The first target should be restrained restoration of calibrated, stacked,
linear mono images. It should not be advertised as recovery of unknowable
detail. The output remains an astronomical measurement product and must carry
the model, parameters, and input provenance.

## Training record

One logical example should contain:

- a high-quality linear reference image;
- a synthetically degraded input or an exactly registered real input;
- telescope, camera, filter, binning, pixel size, focal length, exposure,
  object, session, and processing-history identifiers when available;
- the degradation parameters: PSF family and coefficients, field position,
  Poisson level, read noise, tracking kernel, saturation, and resampling;
- masks for saturation, bad pixels, registration boundaries, and unsupported
  regions;
- a stable group identifier for object, telescope, night, and acquisition
  session; and
- rights, consent, and retention metadata for contributed examples.

Store linear floating-point samples. Display stretches may accompany an
example for review, but must never become the training target accidentally.
For an expert pair, preserve both original FITS headers and record every
operation that produced the target. Reject pairs that cannot be registered to
sub-pixel precision or whose target includes unrelated edits such as star
removal, denoising, local contrast painting, or color remapping.

## Building trustworthy pairs

### Primary path: known forward models

Start from sharp, high-SNR references and generate inputs with a randomized
forward model:

1. Convolve with Gaussian and Moffat PSFs, then add modest ellipticity,
   diffraction structure, field variation, and short tracking kernels.
2. Apply pixel integration and resampling at the recorded image scale.
3. Add shot noise, sky background, read noise, fixed-pattern residuals, and
   quantization in physically plausible order.
4. Model saturation and blooming, but mask those pixels from restoration loss.
5. Keep the random seed and complete degradation description.

The undegraded reference is then a real target, and the known PSF enables
direct tests against classical deconvolution. Use actual dark/flat residuals
and measured PSF distributions from the corpus to tune the simulation rather
than inventing a convenient noise model.

### Secondary path: expert before/after pairs

Use contributed pairs to adapt the synthetic baseline to real optics and
processing preferences. They are most useful for pairwise ranking (which of
two conservative outputs is preferred), residual fine-tuning at a low weight,
and validation by independent reviewers. Multiple acceptable targets for one
input are better than treating one operator's rendition as unique truth.

### Split before extracting patches

Assign train, validation, and test groups by object *and* acquisition session
before cropping patches. A split made after patch extraction leaks adjacent
sky, stars, sensor defects, and processing fingerprints across sets. Keep a
separate telescope holdout and an out-of-distribution set containing unusual
PSFs, sparse fields, saturated stars, galaxies, reflection nebulae, and weak
background signal.

## Baseline model

A small residual U-Net is a reasonable first learned baseline. Condition it on
measured PSF width/shape, robust noise estimates, pixel scale, filter class,
and normalized field coordinates. Predict a bounded residual and an
uncertainty map; return `input + strength * residual` with a user-adjustable
strength defaulting below one. Tile large images with overlap and smooth
weighting, and test tiled output against whole-image inference.

Do not start with an adversarial loss. A practical first loss is a weighted
combination of:

- Charbonnier or robust L1 error in linear space;
- multi-scale gradient and frequency-domain error;
- local and global flux conservation;
- star centroid, encircled-energy, and shape error;
- background residual whiteness without suppressing faint structure; and
- uncertainty calibration and identity loss on already-sharp inputs.

An initial model can operate on mono data. RGB and narrowband combinations
need an explicit cross-channel policy, and raw Bayer restoration should remain
out of scope until the mosaic forward model is represented directly.

## Evaluation gates

No single sharpness score is sufficient. Promotion should require all of:

- synthetic holdout improvement against known truth;
- real-image PSF FWHM and encircled-energy improvement without centroid drift;
- bounded aperture-photometry and total-flux error;
- astrometric residuals no worse than the input;
- stable background MAD and power spectrum;
- no new detections in blank-sky injections or structure copied from other
  training targets;
- graceful identity behavior for already-sharp, low-SNR, and unsupported
  inputs; and
- blind human review at identical display stretches.

Compare the model with the conservative Richardson-Lucy prototype, a no-op,
and simple sharpening. The model only earns inclusion when it improves the
known-truth tests and real-image measurements without increasing artifact or
hallucination failures.

## Product and provenance contract

Every restored FITS should record the operation name, model family, immutable
model hash, model version, strength, tile size/overlap, conditioning values,
software version, and uncertainty summary. Keep the original input unchanged.
Reject or warn on unsupported sample transfer, Bayer data, missing conditioning,
or inputs outside the training distribution. The classical operation remains
available as a deterministic fallback with separately named provenance.

## Staged delivery

1. Turn the classical prototype and corpus procedure into an evaluation
   harness with fixed crops, metrics, and identical display stretches.
2. Build a manifest for linear images and measure real PSF/noise distributions.
3. Train a mono residual baseline on synthetic degradations; freeze a grouped
   holdout before tuning.
4. Collect registered expert examples and preference labels, with consent and
   processing provenance.
5. Fine-tune conservatively, run hallucination/photometry/astrometry gates, and
   publish a model card.
6. Prototype inference behind an experimental command that always emits the
   model hash and uncertainty, then decide whether the evidence supports a
   product feature.

The current real-corpus baseline and reproducible display comparisons are in
the [July 2026 corpus note](../benchmarks/2026-07-deconvolution-corpus.md).
