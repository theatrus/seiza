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

Registered images may use `NaN` where no source pixel covers the reference
grid. `deconvolve_masked` treats those samples as a fixed mask. Each forward
and correction convolution divides by its valid PSF support, and the output
keeps the same mask. The strict `deconvolve` call still rejects non-finite
samples.

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

An eight-iteration, 65% blend sweep closed much of the FWHM gap on three corpus
fields, and matched the installed learned reference on one. It also created
obvious dark rings on a fourth field. That setting is preserved in the
[corpus note](../benchmarks/2026-07-deconvolution-corpus.md) as a failure-bearing
comparison, not a preset. Do not trade visual and photometric validity for a
smaller aggregate FWHM.

## What this can and cannot show

This can test recovery of contrast lost to a modest, approximately symmetric
blur. It is deterministic, flux-preserving, relatively lightweight, and does
not synthesize structures from a training prior.

The fixed-PSF call does not estimate the PSF or vary it across the field;
`deconvolve_field` does both, with the field below. Neither distinguishes
atmospheric seeing from tracking or aberrations, understands saturated stars,
or prevents all ringing. A wrong FWHM is a wrong forward model. Raw Bayer mosaics are rejected
because independent mosaic-sample deconvolution creates color and checkerboard
artifacts.

## Field-varying PSF

`deconvolve_field` runs the same damped update with the PSF a
`seiza_stars::PsfField` measured from the image's own stars, which varies
across the field. It deconvolves towards a round Gaussian target `T`, never
narrower than 1.5 px, instead of towards a point. The target is one of
three:

- a fixed FWHM;
- a fraction of the local FWHM;
- the field's best round PSF, its narrowest local FWHM everywhere, which
  corrects tilt, coma and elongation without sharpening the field's best
  part.

The forward model is the remaining kernel `R`, with `P = T ∗ R`, so the
estimate is the image as it would be at `T`. `RemainingKernel` samples `R`
on a grid of nodes, 256 px apart by default. At each node it deconvolves
the PSF stamp by `T` with Richardson–Lucy on the stamp itself, using FFT
convolutions and starting from the Gaussian `R`. The kernel then stays
positive and carries the PSF's measured wings, which `T` lacks.

Each source pixel is blurred by its own kernel: `H x = Σₖ Rₖ ∗ (wₖ · x)`,
with bilinear hat weights `wₖ` that sum to one. Light is conserved exactly.
The adjoint `Σₖ wₖ · (R̃ₖ ∗ y)` carries the correction back. Each node's
convolution is an FFT over its tile, with reflection at the image edges.
Where neighbouring kernels differ, a flat sky stays flat to a few tenths of
a percent in the synthetic tests, least well at the image edges. All four
guardrails apply, and so does the missing-data mask.

`reblur_consistency` checks any estimate at `T`, from this or a learned
model, against its input: blurred by `R`, it must give the input back to
within the input's noise.

On corpus deep stacks, with ten iterations, a full blend, the default noise
floor and the best round target:

| Stack | Target | FWHM before | FWHM after | Ellipticity, median | Reblur χ² (input as output) |
|---|---|---|---|---|---|
| RedCat61, Sh2 136 B (tilted) | 2.03 px | 2.04–3.61 | 1.92–2.32 | 0.24 → 0.16 | 8.7 (13.3) |
| Askar107PHQ, Heart and Soul Hα | 2.36 px | 2.36–5.18 | 2.14–2.68 | 0.06 → 0.05 | 1.08 (1.59) |
| Ultracat, Coma Cluster G | 1.89 px | 1.91–4.36 | 1.81–2.72 | 0.12 → 0.06 | 1.06 (2.21) |

The FWHM ranges are of the PSF field measured before and after, on a 6 × 4
grid. The noise map is `sqrt(variance / coverage)`. Saturated cores and
pixels with under half the median coverage are left out of χ², because a
stack's thin edges have almost no variance. In the sky, the RedCat's χ² is
1.6. Its bright stars, near 2 px FWHM, are undersampled, and the PSF field
fits them to only 5% to 7%. That leaves χ² of 115 over the 6% of pixels in
stars and nebula. The input itself, offered as the result, scores 179 there.
The runs took 6 to 12 s for a 26 Mpx stack and 17 s for 61 Mpx.

## Next experiments

- Compare Gaussian and Moffat kernels on real star wings.
- Add star masks, edge tapering, and total-variation or wavelet regularization
  for low-SNR backgrounds and ringing control.
- Run field-varying Richardson–Lucy on the starless layer of catalogue-guided
  star removal, so that stars never take part in an iteration.
- Establish objective tests: held-out synthetic forward models, FWHM/encircled
  energy, flux error, background power, and ringing around high-contrast stars.
- Treat any learned restoration as a separate, provenance-bearing operation
  with model identity and uncertainty, rather than another Richardson-Lucy mode.

See the [real-corpus examples](../benchmarks/2026-07-deconvolution-corpus.md)
for the first measured classical/learned comparison and the
[model-based restoration plan](ml-restoration-training.md) for a training-data,
evaluation, and deployment proposal.
