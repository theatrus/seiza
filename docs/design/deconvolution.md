# Conservative deconvolution prototype

## Goal

Test whether Seiza can offer a useful *light* restoration pass for calibrated
linear astrophotography data without implying the semantic reconstruction that
a learned, model-based method would perform. The first prototype is deliberately
classical and explicit: damped Richardson-Lucy deconvolution using a measured,
circular Gaussian point-spread function.

## Operation

For observed image `y`, current estimate `x`, and normalized PSF convolution
operator `H`, each Richardson-Lucy iteration applies:

```text
x_next = x * H^T(y / max(Hx, epsilon))
```

The Gaussian PSF is symmetric, so `H^T = H`, and it is applied as two separable
one-dimensional convolutions with reflected boundaries. Seiza then adds four
guardrails:

1. Clamp the multiplicative correction in each iteration.
2. Damp updates whose residual is near a channel-relative noise floor.
3. Renormalize flux independently in each channel.
4. Blend only part of the restored estimate into the original image.

Negative background-corrected values receive a reversible channel offset while
the non-negative Richardson-Lucy update runs. Mono and interleaved RGB are
supported; RGB channels use one shared PSF but independent updates and flux
normalization.

## Suggested first pass

Measure the median FWHM of several unsaturated stars near the area of interest,
then run after stacking and background correction but before stretching:

```text
seiza deconvolve stack-bg.fits --output stack-light-dc.fits \
  --psf-fwhm 3.1 --iterations 4 --amount 0.35 --noise-fraction 0.001
```

Compare at identical display stretches. Inspect unsaturated star cores, bright
star halos, nebular edges, the background noise spectrum, and image borders.
Increase the blend before increasing iterations. Values above roughly 8-10
iterations should be considered stress tests, not normal processing.

## What this can and cannot show

This can test recovery of contrast lost to a modest, approximately symmetric
blur. It is deterministic, flux-preserving, relatively lightweight, and does
not synthesize structures from a training prior.

It does not estimate the PSF, vary it across the field, distinguish atmospheric
seeing from tracking or aberrations, understand saturated stars, or prevent all
ringing. A wrong FWHM is a wrong forward model. Raw Bayer mosaics are rejected
because independent mosaic-sample deconvolution creates color and checkerboard
artifacts.

## Next experiments

- Estimate local PSFs from unsaturated detected stars and report confidence.
- Compare Gaussian and Moffat kernels on real star wings.
- Partition the field into smoothly blended PSF regions for spatial variation.
- Add star/support masks and stronger regularization for low-SNR backgrounds.
- Establish objective tests: held-out synthetic forward models, FWHM/encircled
  energy, flux error, background power, and ringing around high-contrast stars.
- Treat any learned restoration as a separate, provenance-bearing operation
  with model identity and uncertainty, rather than another Richardson-Lucy mode.

See the [real-corpus examples](../benchmarks/2026-07-deconvolution-corpus.md)
for the first measured classical/learned comparison and the
[model-based restoration plan](ml-restoration-training.md) for a training-data,
evaluation, and deployment proposal.
