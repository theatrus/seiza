# Learned restoration plan

## Short answer

Seiza should train three small models of its own:

- a **noise model** that denoises linear stacks;
- a **PSF model** that sharpens linear stacks by deconvolving towards a
  target PSF;
- a **star model** that splits an image into starless and star layers.

Each is named for what it does, carries a version and an immutable hash, and
ships beside a deterministic classical operation that does the same job less
well and remains the fallback.

Seiza can make training pairs that a tool working only from finished images
cannot. It owns the stacker. It can:

- stack disjoint sets of one target's sub-frames onto one grid, with
  per-pixel variance and coverage;
- split sub-frames by measured seeing;
- measure the PSF across the field;
- put Gaia stars where the solved image says they are.

Real pairs made this way are the backbone of the noise model and fine-tune
the other two. The PSF and star models get most of their data from synthetic
composites: real backgrounds, with stars re-rendered at chosen PSFs. Each
composite's truth is exact.

Expert before-and-after pairs teach a model to imitate an operator. They do
not establish the scene as ground truth.

The first targets are calibrated, stacked, linear images: mono first, then
RGB. Raw Bayer data stays out of scope until the mosaic forward model is
represented directly. Nothing here is advertised as recovering detail the data
does not hold. Every output is a measurement product that records its model,
parameters and inputs.

The models are trained in PyTorch. Each is exported once to ONNX and once to
Core ML from the same weights. Seiza runs them through Core ML on Apple
hardware, through ONNX Runtime with CUDA or TensorRT on NVIDIA, and through
ONNX Runtime on the CPU elsewhere. Tiling, normalisation, conditioning and
blending stay in Seiza's Rust code, so every backend runs the same tensors.

## The three models

| Model | Operation | Input | Output | Real pairs | Classical baseline |
|---|---|---|---|---|---|
| Noise | `denoise` | linear image, noise map | clean estimate, its removed noise split into high- and low-frequency bands | disjoint sub-frame stacks | wavelet shrinkage driven by the variance map |
| PSF | `sharpen` | linear image, local PSF stamp, noise map, target-ratio map, halo and bloom maps | image at the target PSF | seeing-split stacks, cross-instrument pairs | field-varying Richardson–Lucy |
| Star | `separate-stars` | linear image, noise map | star layer; starless = input − stars | synthetic composites on real backgrounds | catalogue-guided PSF subtraction |

The star model feeds parallax rendering, which needs separate starless and
star images, and `star_stretch`, which stretches stars apart from the nebula.
Both currently get those layers from an [external tool](external-tools.md).

## Shared foundations

The PSF and star models both rest on two pieces of classical code. Both are
useful on their own, so they land first.

### PSF field

`seiza-stars` fits Gaussian and fixed-β Moffat PSFs to `u16` data. A Moffat
profile underestimates the wings, which hold the flux a sharpening model has
to move back into the core. `PsfField` measures an empirical PSF instead:

1. Select isolated stars by detection and catalogue distance. Cut a stamp
   around each: 33 px, or larger for wide PSFs.
2. Centre each stamp to sub-pixel accuracy and normalise it to unit flux.
   Unsaturated stars give the core. Saturated stars give the wings, matched
   to the core over an overlap radius where both are reliable.
3. Fit the stamps as a mean PSF plus a few principal components. The
   components' coefficients vary as quadratic polynomials in `x` and `y`,
   which follows the standard PSF-field method of the survey pipelines. For
   RGB, fit each channel, because refractors give blue a broader halo.
4. Report the star count, the fit residual, and summaries at any point:
   FWHM, the ellipticity components `e1` and `e2`, and Moffat β.

`PsfField::stamp_at(x, y)` returns the local PSF. Several consumers use it:

- the PSF model's conditioning;
- field-varying Richardson–Lucy, the next experiment listed in the
  [deconvolution design](deconvolution.md#next-experiments);
- star rendering for both models' training data;
- linear star reduction and halo removal in the
  [editing tools](editing-tools.md#star-tools-seiza-stacking);
- the [field diagnosis](editing-tools.md#measurements-seiza-analysis).

With fewer than about 30 usable stars, `PsfField` refuses. The caller may
then accept a single global PSF, which is reported as a warning.

#### What landed

`PsfField` is in `seiza-stars`, with its C twin (`seiza_psf_field_measure`
and the calls on its handle) and its Python twin (`seiza.PsfField`). It
settles what the plan left open:

- **Candidates.** Detection runs on the image less a 64 px mesh background,
  lightly smoothed. Adaptive moments give each star's centre, width and
  shape.
- **Isolation.** A neighbour within 5 magnitudes of a star disqualifies it
  inside 1.5 FWHM of its 2.5 FWHM flux aperture, as does one brighter than
  the star anywhere in its stamp. Every other detected neighbour is masked
  out of the stamp, however faint: a star a hundred times fainter still
  outshines the PSF's wings where it sits. Rejecting stars for such
  neighbours left one star in 270 in a dense Milky Way field. Catalogue
  neighbours within 5 magnitudes count the same way. No core star lies
  within two stamp radii of a bloomed star, one whose saturated disc is
  3 px or more in radius.
- **Saturation.** Unless the caller gives a level, it is the brightest
  stars' common clip level: three or more peaks within 2% of one another,
  at 90% of the brightest or above. A single hot pixel above them does not
  hide it, and bright stars spread over a few percent do not fake it. Stars
  from 80% of it give wings, not core.
- **Stamps.** Each is shifted to its centroid by Lanczos-3 and normalised to
  its aperture flux. Its sky comes from a ring just outside the stamp,
  leaving out the neighbours' masks, because a sky ring at the stamp's own
  edge sits in the star's wings. A pixel that most stars mask, as when
  their neighbours share an offset, takes the mean of its ring. The whole
  model is normalised to unit sum when it is sampled.
- **Fit.** Stars are weighted by inverse variance, capped at the upper
  quartile so the brightest do not outvote the field. Stars whose core
  residual is three times the median, and above 10%, are dropped, and the
  field is fitted again.
- **Degree.** A quadratic is not always enough. A tilted sensor whose best
  focus crosses the frame makes the FWHM a V across it. In a synthetic
  tilt, a quadratic overshoots the far corner's FWHM by 18%; a cubic comes
  within 5%. Each channel takes the degree, 1 to 3, that best predicts
  held-out stars over five folds.
- **Wings.** Each saturated star is scaled to the core model at its own
  position, over an annulus. The annulus starts 1.5 px outside the star's
  saturated pixels and ends where the core model's ring average falls to
  twenty times its own noise. The star's sky comes from a ring three stamp
  radii out, past its own wings. The median of the stars' excess over the
  core model is added from that annulus outwards. The core model thus
  keeps the field's variation, and the wings add the halo. In a synthetic
  field with a wide halo, core stars alone lose a quarter of the light 9 to
  15 px out, because their sky ring sits in the halo. The wings bring it
  back to within 8%.
- **Shape.** `shape_at` fits an elliptical Moffat with free β to the local
  stamp. FWHM is the mean of the two axes' widths. `e1` and `e2` are
  `χ cos 2φ` and `χ sin 2φ`, with `χ = (a² − b²)/(a² + b²)`. β stops at 25,
  which undersampled stars near 2 px FWHM often reach.
- **Fit quality.** Each channel reports the median χ² per pixel against the
  background noise, which bright stars' photon noise raises well above 1,
  and the median relative residual within 1.5 FWHM.
- **Sparse fields.** A varying field needs ten stars for each component,
  and so at least ten. With fewer, or fewer than the minimum, a field
  refuses with `TooFewStars`. With the global policy, three stars or more
  give one PSF and a warning. Asking for no components gives one PSF too,
  with no warning.

Measured on corpus deep stacks (mono, one channel each), every field took
the cubic. Times are for the measurement alone.

| Stack | Scale | Core / wing stars | FWHM, px | Ellipticity | Relative residual |
|---|---|---|---|---|---|
| Ultracat, Coma Cluster G, 9576 × 6388 | 1.5″ | 398 / 40 | 1.9 centre to 4.1 bottom right | to 0.16 | 5.0% |
| C925, Question Mark Galaxy B | 0.33″ | 68 / 0 | 8.8 to 10.1, 71 px stamps | about 0.06 | 5.4%, χ² 1.0 |
| Askar107PHQ, Heart and Soul Hα | 1.04″ | 376 / 9 | 2.4 centre to 4.9 top left | to 0.09 | 2.4% |
| RedCat61, Sh2 136 B | 2.59″ | 388 / 39 | 2.0 lower left to 3.5 top right | to 0.43 | 5.4% |
| RedCat61, Sh2 136 L | 2.59″ | 389 / 39 | 1.9 to 3.0 | to 0.40 | 6.8% |
| RedCat61, Teddy Bear OIII | 2.59″ | 394 / 40 | 2.2 to 3.3 | to 0.37 | 5.4% |
| Radian61, Sh2 119 G | 2.83″ | 397 / 29 | 1.9 to 2.7 | to 0.43 | 6.9% |

The RedCat61's tilt shows clearly. On both Sh2 136 stacks, best focus lies
low and left of centre, where the ellipticity is about 0.05. Towards the
right edge the FWHM rises steeply and the ellipticity reaches 0.4, its
direction turning across the frame. A plane through the FWHM map explains
only 56% to 63% of its variance, because the FWHM is a V, not a slope. The
Teddy Bear stack, from a different session, adds field curvature, and there
a plane explains 8%. Stars near 2 px FWHM are undersampled, and Lanczos shifting of
their stamps leaves the 5% to 7% relative residuals. The oversampled C925
field fits to its noise.

### Catalogue-guided star removal

Solve the image and list its Gaia stars above the [star policy](#star-policy)
threshold. Fit each star's flux with the local PSF stamp, all overlapping
stars jointly, then subtract. A bright or saturated star's PSF photometry is
unreliable, so its radial profile is subtracted as the halo tool measures it.
Its core is then filled from the annulus outside it, and Seiza records where
each fill was made.

The result is a deterministic star removal that needs no model. Parallax can
use it without the external tool. The starless images that pass review are
the star model's backgrounds, and the starting point for the PSF model's
references.

The fit, the removal and the rendering have landed as `seiza_stars::StarFit`
(C `seiza_star_fit_new`, Python `seiza.StarFit`). The halo subtraction and
the core fills have not.

- **Stars.** It takes the catalogue stars at their image positions, so the
  solve and the catalogue stay with the caller. In Python those come from
  `StarCatalog.cone_search` and `Wcs.world_to_pixel`.
- **Fit.** Each star's window is three FWHM in radius. Stars whose windows
  overlap are fitted jointly, with a background plane per group, by
  weighted least squares against the noise map. A chain of more than 48
  stars is split. Three passes fit each group against the data less every
  other group's stars, so wings that reach into a neighbour's window are
  accounted for.
- **Weights.** A sample's variance adds the PSF model's own error, its
  relative residual times the model there. Otherwise a bright core's
  percent-level model error drives the plane under a faint neighbour.
- **Saturation.** Samples above a given level are left out, so a saturated
  star's flux comes from its wings, which the field measured from saturated
  stars.
- **Positions.** They are held fixed by default: PSF photometry at known
  positions, as star truth needs. A solve good to 0.15 px still left
  residual dipoles, so a refined mode adds each star's shift to the linear
  fit, through the stamp's gradient, and lets it move up to 1 px.
- **Render.** `render` draws the stars through a pixel-integrated round
  Gaussian, and `sharpen` subtracts them and renders them again. That is
  the classical star sharpener of the Gaia plan.

On corpus deep stacks, solved from detections with the set's pointing as
the prior and SIP order 3, against the installed G 17 catalogue:

| Stack | Solve | Stars | Instrumental mag against G | χ² near stars before, after (refined) |
|---|---|---|---|---|
| Askar107PHQ, Heart and Soul Hα | 0.15 px | 11,936 | slope 0.98, scatter 0.17 mag | 466 → 2.4 |
| RedCat61, Sh2 136 B | 0.21 px | 38,808 | slope 1.00, scatter 0.41 mag | 446 → 31 |
| Ultracat, Coma Cluster G | 0.13 px | 4,339 | slope 1.06, scatter 0.31 mag | 1,638 → 42 |

The χ² is over 13 px boxes around the 400 brightest unsaturated stars, so
the PSF model's few-percent error in their cores dominates it. Refinement
moved stars by a median 0.1 to 0.23 px, and cut what fixed positions left:
from 4.4 to 2.4 on the Askar field, 107 to 42 on Coma and 38 to 31 on the
RedCat. The scatter against G is mostly the filters' colour terms, which a
photometric model in BP−RP would take up. Sharpened to a 2 px target, the
Askar field's central FWHM went from 2.36 to 2.12 px and its ellipticity
from 0.04 to 0.001. Fitting the 38,808 RedCat stars took 0.6 s.

## Real data

### Corpus

The telescope corpus holds about 45,500 FITS and XISF files from seven
telescopes: refractors from 61 to 131 mm, an SCT and two remote rigs. Each
telescope keeps raw lights by night under
`_Source/<year>/<target>/<night>/LIGHT`. Masters and calibration frames from
earlier processing are under `_Process`. One night of Bode's Galaxy, for
example, holds 162 five-minute Hα sub-frames of 6248 × 4176 pixels.

The first crawl (2026-10-31) read 41,245 lights in 1,232 sets of 71 targets,
3,349 masters from earlier processing, and 1,520 masters in the calibration
library. The library calibrates 744 sets; 198 of those have the 40 frames a
noise pair needs, 16,369 frames in all. The rest lack a matching flat,
mostly because the library has no masters for three telescopes, whose
calibration frames are still raw: Askar107PHQ alone has 11,826 lights
waiting on them.

The sub-frames matter more than the masters, because pairs are built from
them, and Seiza's own stacks of them look like what users will feed the
models. Masters from earlier processing add variety, such as drizzled and
differently integrated images. They are inputs for the reblur loss and
references at review, never targets themselves: their processing history is
not Seiza's.

PSF Guard's finished stacks are the third source. Each is one target,
filter and exposure over every night it was imaged, stacked by Seiza inside
PSF Guard with the settings a user chose. They are the stacks users bring
to the models, at full depth, and they include Askar107PHQ's, which PSF
Guard calibrated from frames the master library does not hold yet. They
carry no variance or coverage, so they train and are tested with the
estimated noise map. They are inputs for the reblur loss and catalogue
removal, references at review and inputs to the gates, and never targets:
no independent stack of the same frames exists.

The first import (2026-11-01) copied 527 stacks: 259 from Askar107PHQ, 106
from Ultracat, 97 from C925 and 65 from the two remote rigs, all mono but
two, 57 GB in all. Of the 56,545 frames they list, 56,044 were found in the
corpus by file name, so 514 stacks joined their sets' split group through
their own frames and the other 13 through their object's name.

### Building it

The pipeline lives in [`training/`](../../training/README.md), one pass per
step. Each pass writes into the manifest, a SQLite file with one record per
frame and one per set, and can be rerun alone. A set is a target, filter,
camera, gain, offset, exposure, binning and night on one telescope.

1. **Crawl.** Read every header through `seiza-image-io`. Classify frames by
   `IMAGETYP`, group the lights into sets, and record each frame's exposure,
   temperature, gain, offset and saturation level. Headers that disagree, such
   as Radian61's `EXPOSURE=90` beside `EXPTIME=901`, are flagged. The
   manifest never picks one silently.
2. **Plan calibration.** `plan_calibration` chooses each set's bias, dark
   and flat from the calibration-master library, under the matching policy
   the desktop apps use, and the plan must suit every light of the set. A
   set without a flat, or without both a dark and a bias, is left out, and
   the reason is recorded.
3. **Take in finished stacks.** Copy in each PSF Guard group stack at its
   current revision. Its scheduler database, found by the targets its jobs
   name, gives its telescope, and the file names the scheduler recorded for
   its frames find them in the corpus, and so their sets. The stack joins
   those sets' split group.
4. **Measure every frame.** `seiza-analysis` records FWHM, eccentricity, star
   count, background and noise. One frame per set is solved, for the pixel
   scale and the WCS. A frame that lost most of its stars, or whose stars are
   elongated well beyond the set's, is flagged. Worse seeing costs stars too,
   so transparency is judged later, at registration.
5. **Assign split groups.** Each set gets its train, validation or test group
   now, before any product is built (see [splits](#splits)).
6. **Calibrate and register once per set.** Select the reference
   automatically. Register and resample every frame onto it, with the CLI's
   default stacking options, and keep the registered frames on local NVMe
   while the set is processed. Every subset is then integrated from those
   same frames by `LiveStacker.reintegrate_frames`, which normalizes and
   weights each frame as a reintegration of every frame does, against one
   background reference. Subsets share the grid and the normalization
   exactly, and the expensive steps run once per set rather than once per
   subset. A frame the stacker had to brighten by half again over the set's
   median photometric gain lost its light to cloud, and is left out of every
   product. A camera that turned during the night leaves two orientations
   in one set; one Ultracat night has 55 frames 10.5° from the other 15,
   with the same rotator angle in every header. Registration turns away the
   frames beyond its rotation limit, so when it turns away more than it
   admits, the set is registered again on a reference from the larger
   orientation. The limit stays: a flat taken at one angle does not fit
   frames 10° from it. The registered frames are deleted when the set is
   done.
7. **Build the products**, then **review** them, as below.

The manifest decides which sets qualify for which recipe:

- **noise pairs** need at least 40 accepted sub-frames;
- **seeing splits** need a FWHM difference of at least 25% between the best
  and the worst quartile;
- **cross-instrument pairs** need the same object from two focal lengths.
  NGC 7000 is already in the corpus from Askar107PHQ and Radian61.

### Products per set

| Product | Recipe | Used by |
|---|---|---|
| Deep stack, with variance, coverage and saturation records | all accepted frames | references, catalogue removal, noise targets |
| Noise pairs | A of 4, 8, 16 and 32 frames, at most a fifth of the set, three draws each; B the rest | noise model |
| Seeing split | best and worst quartiles at matched frame counts, each with its PSF field | PSF model |
| PSF field | the deep stack | conditioning, star rendering |
| Starless candidate and star list | catalogue-guided removal on the solved deep stack | star backgrounds, once reviewed |
| Finished stack | a PSF Guard group stack, copied in, image only | inputs at user depth, reblur loss, catalogue removal, gates |
| Deep halves | the deep stack's frames dealt alternately into two stacks | noise pairs at a night's whole depth, clean targets |
| Multi-night deep stack, with variance and coverage, and its halves | every accepted frame of one object, telescope and filter, over all its nights; the halves hold whole nights | references and pairs deeper than any night |

Cross-instrument pairs are built per object, across sets, once both sets'
deep stacks exist.

A copied finished stack carries no variance or coverage, so the noise model
cannot be run on it with a noise map of its own. The deepest objects, tens
of hours over many nights, are therefore stacked again from the corpus's own
frames by the `multi-night` pass. One stacker registers every night onto the
deepest night's reference, calibrating each night with its own masters, and
keeps the frames, so the whole stack and its halves are reintegrated from
the same registration. The halves are dealt whole nights, so they share no
frame and their frame counts come out near even: pairs at a depth no single
night reaches. The whole stack is a near-clean reference for the reference
night's own products, which share its grid. The pass can run on a machine
that mounts the corpus under another path; the products record the
manifest's paths.

### Review

The pipeline writes one review page per set, showing these side by side at
one shared stretch, with the measurements beside them:

- an input sub-frame and the deep stack;
- the starless candidate.

A reviewer accepts or rejects the set and its starless background. Each
decision goes into the manifest with the reviewer and the date. Nothing that
needs review enters training unreviewed. Starless backgrounds are reviewed
before R4a, the first run that needs them.

### Size

A 26 MP mono plane in f32 is 104 MB. A mono set's products come to about 100
planes, or 10 GB, most of them noise pairs. RGB triples that. While a
160-frame set is processed, its registered frames take another 17 GB. The
total follows from the number of qualifying sets, which the manifest gives.
Products are stored as `.npy` planes, so a loader reads patches at random
without loading whole images, and can sit on a network share.

## Noise model

### Pairs: disjoint sub-frame stacks

For a qualifying target:

1. Pick one reference with `--reference-image`, so every stack shares one
   grid and WCS exactly.
2. Draw a small subset A (2–25% of the frames) and a disjoint subset B (the
   rest, or a subset at least four times larger). Interleave both in time, so
   that seeing, gradients and airmass match.
3. Stack each subset with identical normalisation, weighting and rejection.
4. The input is stack A with its variance and coverage. The Python snapshot
   already exposes both. The target is stack B.

The noise in A and in B is independent, so this is Noise2Noise: with an L2
loss, the best prediction of a noisy, independent target is the clean signal.
B's own noise adds only a constant to the loss. That constant is known from
B's variance map, so the error against the clean signal can be measured
without bias on real data: `E[(f(A) − B)²] − σ_B²`.

Every target yields many pairs, from different subset sizes and draws. The
inputs carry what users' stacks carry: noise correlated by Lanczos or
bilinear resampling, rejection's effects, residual pattern noise and, for
OSC, the chroma noise left by VNG demosaicing. Drizzled and binned variants
are made in proportion to their use. A pair is rejected when the background
or FWHM of A and B differ beyond a tolerance after local normalisation.

Synthetic noise is the minority. It extends the data to signal-to-noise ratios
the corpus lacks: Poisson–Gaussian noise per synthetic sub-frame, passed
through the same resampling kernel, added to a deep stack.

Once the PSF model exists, a further variant runs the frozen PSF model on both
A and B. Their noise stays independent, so the noise model also learns the
noise of a sharpened image.

### Noise map and strength

The per-pixel noise is `σ = sqrt(variance / effective coverage)`, from the
stack. Stacks made elsewhere carry no variance, and nor does an image after
sharpening. For those, Seiza estimates a smooth map from the image's own
second differences, which is the estimator the `snr` module uses, applied in
local windows. The noise model trains with both kinds of map, so either works.

Effective coverage is not the frame count. In the first corpus build, the
noise of a 4-frame half varied by up to 2.5 times between draws of one
night. In the noisiest draw, every frame had registered at a whole-pixel
shift, so resampling never smoothed it, and three had been dimmed by haze,
so normalization scaled them up by 1.24 to 1.41. The variance an
inverse-noise-weighted stack reports is a weight-1 frame's, so the map must
divide by the sum of the frames' weights. It must also carry each frame's
resampling, which sets how strongly the noise is correlated between
neighbouring pixels. Every pair records the measured and the predicted
noise of its difference, so the map can be checked against the data.

A user's strength control scales the map before it reaches the model. A
larger σ tells the model to remove more, as in FFDNet, so no separate model is
needed per strength.

What the model removes is then split into two bands, each with its own
amount: high frequency, the pixel-to-pixel grain, and low frequency, the
blotches and mottle a few pixels and wider that resampling, drizzle and
gradient removal leave. The removed noise `r = x − f(x)` is split by a
Gaussian of scale `s`, 5 px by default: `r_low = G_s ∗ r` and
`r_high = r − r_low`. The result is `x − a_high · r_high − a_low · r_low`.
Changing either amount, or `s`, is instant and needs no second inference.

The amounts default to less than 1. Taking out everything the model finds
gives the least error against a clean target, but the result looks smooth
and flat: in the first large runs the empty sky kept 2% of its noise, as
flat as a stack about 2,500 times as deep, and fine texture near the noise
floor went with it. Keeping a share `k` of the noise leaves an image as
noisy as a stack `1/k²` as deep, so a 4-frame stack keeping 30% looks as
grainy as 45 frames. The blend adds back only what was in the input, so it
also returns any detail the model took out, and invents nothing. Grain
reads as texture where mottle reads as a fault, so the low band is taken
out more fully than the high. The defaults are chosen on real stacks, by
eye against a deeper stack of the same night, and the training harness
measures runs at them as well as at full removal.

So both bands must be learned, not only the grain. Real pairs carry real
low-frequency noise: resampling correlates neighbouring pixels, and a
night's background residuals differ between disjoint halves. Synthetic
noise adds correlated components at scales drawn from 2 to 32 px. The loss
and the gates are measured per band: the residual's power spectrum must be
flat in each, and each pair records the noise of its difference at the
pixel scale and binned by `s`, beside the prediction, so the low band's
reduction is measured against real data, not only the grain's.

## PSF model

### What it computes

The model maps an image with PSF `P` at each point to the same sky observed
through a target PSF `T`. `T` is a round Gaussian, never a delta function,
and never narrower than about 1.5 px. Deconvolving towards a band-limited
target stops the model inventing detail finer than the data's sampling, and
limits ringing.

The target is given as a per-pixel map of the ratio `FWHM(T) / FWHM(P)`. One
map serves every control:

- **Star and extended strengths.** The map takes one ratio inside the star
  mask and another outside, feathered between them. Training draws random
  smooth maps, so the model never assumes the map is uniform.
- **Uniform correction.** The map is set so that `T` is everywhere the
  field's best round PSF. This corrects tilt, coma and elongation across the
  field without sharpening the image overall.

Two more maps, each an amount from 0 to 1 inside the star mask and 0
outside it, shape the stars beyond their width:

- **Halo reduction** scales the target PSF's wings, beyond the radius where
  the core has fallen to a few percent of its peak, down from the measured
  wings. Halos from filter reflections and scattered light shrink, and the
  core keeps its flux.
- **Bloom reduction** shrinks the swollen, flat-topped disc of a saturated
  star towards the size an unsaturated star of its flux would have at `T`.

At 0, both leave the stars as `T` renders them, which is what every real
pair asks for. Star sharpening, non-stellar sharpening, halo reduction and
bloom reduction are therefore four controls a user sets apart, in one
inference.

### Conditioning on the local PSF

Each tile's PSF stamp comes from `PsfField::stamp_at` at the tile's centre.
A small convolutional encoder, in the same exported graph, turns the stamp
into a vector. That vector scales and shifts the U-Net's channels at the
bottleneck and in each decoder level, which is FiLM. Encoding the stamp
directly, rather than a few parameters, lets the model see coma, astigmatism,
spikes and wings. Overlap blending smooths the step from one tile's PSF to the
next.

The conditioning stamp is perturbed in training: FWHM changes by ±10% and
ellipticity is jittered. The model must tolerate the measurement error it will
meet. A gate tests a deliberately wrong PSF.

### Pairs

There are four sources:

1. **Synthetic composites with re-rendered stars (known truth).** A reference
   is a deep stack's reviewed starless image, with PSF `P_ref`. Public
   space-telescope mosaics are added, resampled to 1–4″/px, with each source's
   licence recorded. Stars come from that field's Gaia list, or from a random
   sky position's list, with flux from the zero point and colour from BP−RP.
   The input is built as:
   - the background convolved with a kernel `K` such that `P_in = P_ref ∗ K`.
     `P_in` is drawn from the corpus's measured PSF stamps, including field
     variation, spikes for reflectors and short tracking kernels;
   - plus the stars, rendered with `P_in`;
   - plus noise, as for the noise model.

   The target is the background convolved to `T`, plus the stars rendered with
   `T`, with their wings scaled by the halo amount and their saturated discs
   shrunk by the bloom amount, both drawn per composite. Extended structure has a hard limit: `T` can never be narrower than
   `P_ref`, because detail lost in the reference cannot be restored, so the
   extended part of the ratio map is clamped to that limit. Stars have no such
   limit. They are rendered exactly at any PSF, so the model learns star
   tightening against true point sources, down to the 1.5 px floor.
2. **Seeing-split pairs (real).** Sort a session's sub-frames by measured
   FWHM. Stack the worst quartile as input and the best quartile as target,
   on one grid with matched depth, and measure the PSF field of each. The
   ratio map is the measured ratio between the two. The target is not truth,
   but it is a real observation of the same sky through real optics with a
   smaller PSF.
3. **Star-rendered real pairs (real, exact for stars).** On a solved stack,
   fit the stars at their Gaia positions, remove them, and render them again
   at `T`. Hybrid seeing-split pairs take their background from the best
   quartile and their stars from that render. The
   [Gaia-anchored sharpening plan](gaia-anchored-sharpening.md) describes
   both, with the Gaia gates and a classical star sharpener.
4. **Cross-instrument pairs (real, few).** Solve both stacks. Reproject the
   longer focal length's stack onto the shorter one's grid, then convolve it to
   the target PSF. Fit a linear photometric match first, because cameras and
   filters differ. These pairs validate only.

Saturated star cores are masked from every loss on real images, using the
stacker's saturation records. Composites render their saturated stars, so
bloom reduction learns against exact targets there.

### Reblur consistency

Any real image with a measured PSF gives one more signal. `T` is narrower
than `P`, so there is a remaining kernel `R` with `P = T ∗ R`. For Gaussians
`R` is a Gaussian with `σ_R² = σ_P² − σ_T²`. For empirical stamps it is a
regularised deconvolution of `P` by `T`, which Seiza computes with `rustfft`.
Blurring the output by `R` must give back the input, to within the input's
noise.

This is used twice:

- **As a training loss**, at a low weight, on real images that have no target:
  `|R ∗ f(y) − y|²` weighted by the noise map. It adapts the model to real
  optics. The supervised synthetic loss keeps it from settling on a noisy
  inverse filter, which would satisfy consistency alone.
- **As a gate on every image the model is shown**, with no truth needed. The
  reduced χ² of the reblur residual should be close to 1, and its spectrum
  should be flat. Structure in the residual means the model has added or lost
  something that the data do not support.

`seiza-deconvolution` now computes both. `RemainingKernel` holds `R` on a
grid of nodes, at most 256 px apart by default, for each channel. At each
node it deconvolves the `PsfField` stamp by `T` with Richardson–Lucy, using
`rustfft` convolutions and starting from the Gaussian `R`. A Tikhonov
division rang badly when `T` was nearly as wide as `P`, where `R` is close
to a point. `reblur_consistency` blurs an estimate by `R`, each source pixel
by its own kernel, and reports, per channel:

- the reduced χ² of the residual against the input's noise map;
- the residual's power in eight radial frequency bands, from 128 px
  Hann-windowed tiles;
- the bands' spectral flatness, their geometric over their arithmetic mean.

A stack's noise is correlated by resampling, so even a perfect estimate
leaves a residual whose power falls towards Nyquist. Its flatness must be
compared with that of the stack's own noise, from the difference of its
halves, rather than with 1. The C ABI has `seiza_remaining_kernel_new` and
`seiza_reblur_consistency_json`, and Python `seiza.RemainingKernel` and
`seiza.reblur_consistency`.

## Star model

### Star policy

What counts as a star is a policy, and both the training truth and the gates
follow it:

- A point source goes into the star layer when its peak S/N in the input is at
  least 3. Fainter stars stay in the starless layer as background texture.
- Each star's halo and diffraction spikes go with it, out to the stamp radius
  of the PSF field.
- Compact objects in the object catalogue stay in the starless layer: small
  galaxies, planetary nebulae and the cores of globular clusters. Gaia
  sources inside them are not rendered as stars.
- Under a star, the starless layer is the local background, interpolated.

Reflection halos larger than the stamp radius stay with the classical halo
tool. They are offset and few, and the model's receptive field does not reach
them.

### Composites with split noise

Synthetic composites are built on the reviewed starless backgrounds from
catalogue-guided removal. The stars are rendered with:

- the field's empirical PSF stamps, per channel;
- spikes measured from reflector images in the corpus;
- clipping and blooming at the camera's saturation level;
- colour from BP−RP through the camera's response, as fitted by colour
  calibration;
- for OSC, a pass through the Bayer pattern and Seiza's own VNG demosaicing,
  so the model sees the colour artefacts real OSC stars have.

New stars are rendered where the real ones were removed, so every fill sits
under a star. The rest are scattered at that field's Gaia density.

The noise is split by its source. The background already carries its own real
noise. The stars' photon noise is drawn separately, with variance proportional
to the rendered flux and correlated by the same resampling kernel. The targets
are then:

- **starless:** the background, with its own noise;
- **stars:** the rendered stars plus their own photon noise.

They sum to the input exactly. The model learns to move a bright star's photon
noise into the star layer, so the starless layer has no noisy patch where the
star was. The model predicts the star layer, which is sparse and easier to
learn, and returns `starless = input − stars`. That keeps the sum exact. The
star layer can go slightly negative in faint wings. `stretch_rgb_stars`
already clips negative residuals in its display layer only.

### Real images without truth

Two losses work on any real, solved image:

- **Consistency.** Add synthetic stars `S` to a real image `X`. The starless
  output must not change, and the star layer must grow by exactly `S`. As a
  loss, that is `|starless(X + S) − starless(X)|` and
  `|stars(X + S) − stars(X) − S|`.
- **Catalogue labels.** Away from every Gaia star above the policy threshold,
  the star layer must hold nothing above noise. At catalogued compact objects
  it must hold nothing at all. This applies only where Gaia is complete, to
  G 19 and outside crowded fields.

Model v1 trains on composites plus these two losses. Its starless results on
new fields, once reviewed, add to the backgrounds for v2.

### Receptive field

A star with a large halo needs a wider view than denoising does. The star
model has six levels instead of four, which takes the receptive field from
about 140 px to about 500 px. With channel widths capped at 256, the two extra
levels run on 1/256 and 1/1024 of the pixels, and add about a tenth to the
cost. Its tiles are 1024 px.

## Training record

One example holds:

- the linear input, the linear target, and both variance and coverage maps;
- the recipe: noise pair, seeing split, cross-instrument or composite, with
  its subset sizes, PSF stamps, ratio map, halo and bloom amounts, rendered
  star list and random seed;
- telescope, camera, filter, binning, pixel size, focal length, exposure,
  object, session and processing-history identifiers;
- masks for saturation, bad pixels, registration boundaries, removed-star
  fills and unsupported regions;
- a stable group identifier for object, telescope, night and session; and
- rights, consent and retention metadata for anything contributed.

Store linear floating-point samples. A display stretch may accompany an
example for review, but must never become the target by accident.

## Splits

Assign train, validation and test by object *and* session before cropping
patches. A split made after patch extraction leaks adjacent sky, stars, sensor
defects and processing fingerprints across sets. Every product inherits its
set's group: its subsets and its starless background.
An object imaged on several nights or through several telescopes keeps all
of them in one group, and so do fields that overlap on the sky. In the
corpus that leaves about 44 independent regions from 71 targets: the
telescopes image the same objects, and wide fields join their neighbours.

Hold one telescope out entirely: its sets are never trained on, and are
scored apart. Moving every group it touches to test instead would move a
third to a half of the corpus there, so its sky may reach training through
other telescopes, and it tests the instrument rather than the sky. Keep an
out-of-distribution set with:

- unusual PSFs;
- sparse fields;
- saturated stars;
- galaxies and reflection nebulae;
- background signal near the noise.

## Architecture

All three models are one family: a residual U-Net with two 3×3 convolutions
per level, base width 16–32, at 1–3 M parameters. The noise and PSF models
have four levels; the star model has six. The PSF model adds the stamp
encoder and FiLM.

The network's input is `asinh((x − b) / (k·σ̄))`, where `b` is the background
and `σ̄` the median noise. The compression is invertible and spans the five
decades between sky noise and star cores. It also keeps activations inside
f16's range, which Core ML's GPU and Neural Engine use. The output is a
residual in linear units of `σ̄`. Losses are computed in linear space, so flux
is judged where it is physical.

The operations are limited to those that Core ML, TensorRT and ONNX Runtime's
CPU provider all run natively:

- 3×3 and 1×1 convolutions;
- stride-2 downsampling and nearest-neighbour upsampling;
- concatenation, addition and broadcast multiplication;
- ReLU or SiLU.

There are no normalisation layers. Batch normalisation is folded into the
convolutions at export. Group and instance normalisation are not used,
because per-tile statistics make tiles disagree.

At base width 24, a four-level forward pass costs about 0.2 MFLOP per pixel,
or about 15 TFLOP for a 61 MP frame once tile overlap is counted. The smallest
width that passes the gates ships.

The variants are:

- the noise model, mono and RGB, with one σ map per channel;
- the PSF model, mono, run per channel on RGB with that channel's stamp;
- the star model, mono and RGB.

## Losses

- **Noise:** L2 against the independent target, with per-pixel weights from
  the variance map. Add flux conservation in background boxes and faint-signal
  boxes, and a penalty on a residual whose power spectrum is not flat, which
  is the blotchy "worm" texture.
- **PSF:** Charbonnier error against the target. Add multi-scale gradient
  error, per-star aperture flux within three times the FWHM, centroid shift,
  reblur consistency on real images, and an identity loss where the ratio map
  is 1.
- **Star:** L1 error on both layers, with the star layer weighted by inverse
  noise so faint stars count. Add the consistency and catalogue losses, and a
  penalty on star-layer energy outside star masks, which is nebula leaking
  into the stars.

None uses an adversarial loss. Augmentation is the eight flips and rotations,
plus exposure scaling. Sub-pixel shifts are not used, because they would break
registration. Patches are sampled more heavily where there are stars and
nebula, but sky is kept, so the models learn to leave it alone.

## Training runs

Training code lives in a new top-level `training/` directory, outside the
workspace like `seiza-py`. It is PyTorch, and builds every pair and composite
with the `seiza` Python package, so training data passes through the stacker,
PSF field and star rendering the product uses. Configurations and run records
are committed. Data, the dataset store and weights are not.

A run record holds:

- the commit;
- the manifest and dataset hashes;
- the configuration and seed;
- every gate's metrics;
- the hash of the final weights.

Full runs use PyTorch on CUDA, on one 32 GB NVIDIA GPU. Smoke runs and
debugging use PyTorch's Metal backend on Apple Silicon. Synthetic degradation
and star rendering run on the GPU inside the training loop, seeded. Real pairs
and reviewed backgrounds are built once and cached on local NVMe.

| Run | Model | Data | Purpose |
|---|---|---|---|
| R0 | noise, mono, width 16 | one target's noise pairs, 10k steps | End to end: pairs, training, ONNX and Core ML export, the Rust backends, parity tests and timings. Its measurements replace the estimates below. |
| R1 | noise, mono | all qualifying mono targets, plus 20% synthetic | First noise model to face the gates. |
| R2 | noise, RGB | OSC targets and RGB composites | Chroma noise from demosaicing. |
| R3a | PSF, mono | synthetic composites with re-rendered stars | Learn the mapping against exact truth. |
| R3b | PSF, mono, from R3a | seeing splits and reblur consistency on real fields | Adapt to real optics; first PSF model to face the gates. |
| R4a | star, mono and RGB | composites with split noise over reviewed backgrounds | Learn the separation against exact truth. |
| R4b | star, from R4a | consistency and catalogue losses on real fields; v1-bootstrapped backgrounds | Adapt to real fields; first star model to face the gates. |
| R5 | noise v2, PSF and star second round | sharpened noise pairs, more seeing splits and backgrounds | Second round. |

Every run works on overlapping tiles, never on whole images, so memory,
activations and the exported graph's shape stay fixed however large the
image. Training cuts each memory-mapped plane into 256 × 256 tiles (512 for
the star model) on a grid whose stride is the tile less its overlap, with a
fresh random offset each epoch. Neighbouring tiles share at least half the
receptive field, so the model meets every sky position both near a tile's
centre and near its edge. Validation and the gates run the inference path,
tile, infer and blend, so their numbers include the seams.

Each run uses batch 32, AdamW at 2e-4 with cosine decay, bf16, and
200k–400k steps. By the FLOP estimate, a
full run takes from a few hours to overnight on that GPU. Width {16, 24, 32}
and depth are swept only on R1, R3a and R4a, and the winners carry forward.

## Inference in Seiza

A new crate, `seiza-restore`, runs the models and holds the three operations.
The classical operations stay where they belong: Richardson–Lucy in
`seiza-deconvolution`, and the PSF field and catalogue-guided star removal in
`seiza-stars` and `seiza-stacking`. `seiza-restore` has its own
`OUTPUT_VERSION` and `output_version` test, under the rules in
[AGENTS.md](../../AGENTS.md). That test runs a small committed fixture model
on ONNX Runtime's CPU provider, which is deterministic.

### Backends

Each model is exported from the same PyTorch weights twice:

- `torch.onnx.export` writes ONNX at a fixed opset (17) with a static tile
  shape, which TensorRT and Core ML both prefer;
- `coremltools` writes a Core ML ML Program package from the PyTorch graph.
  It no longer converts from ONNX.

Seiza picks a backend at run time:

| Hardware | Backend | Rust binding | Precision |
|---|---|---|---|
| Apple Silicon | Core ML; GPU by default, Neural Engine once its parity passes | `objc2-core-ml`, behind a `coreml` feature (macOS only) | f16 on the Neural Engine; f16 or f32 on the GPU |
| NVIDIA | ONNX Runtime with TensorRT, or CUDA when TensorRT is absent | `ort` with `load-dynamic`, behind an `onnx` feature | f16 or f32 |
| Other GPUs and CPUs | ONNX Runtime CPU provider | `ort` | f32 |

The backend only runs a tile tensor through a graph. Everything else is
Seiza's Rust code, shared by every backend:

- the PSF field and the stamp for each tile;
- normalisation and the noise and ratio maps;
- tiling and overlap blending;
- the reblur check;
- provenance.

ONNX Runtime is loaded dynamically, so a crates.io build needs no native
library at build time. The Windows app ships `onnxruntime` with the CUDA and
TensorRT providers as an optional component. Homebrew depends on the
`onnxruntime` formula. The Python wheel finds the library in the
`onnxruntime` or `onnxruntime-gpu` package. On macOS, Core ML is part of the
system. A TensorRT engine is built for each GPU and tile shape on first use,
and cached beside the model.

Integrated Intel GPUs, which PSF Guard must support, run on the CPU provider
at first. ONNX Runtime's OpenVINO and DirectML providers drop in through the
same binding if R0's CPU timings show the need.

Without a usable backend, a learned operation returns
`RestoreError::BackendUnavailable` with the reason, and the classical
operation remains.

Initial targets for a 61 MP mono frame, to be replaced by R0's measurements:

- 10 s on an Apple M-series GPU;
- 5 s on a current NVIDIA GPU with TensorRT;
- 3 minutes on the CPU.

### Weights and distribution

A model bundle holds:

- the weights as safetensors, which are the source of truth;
- `model.onnx` and `model.mlpackage`, both exported from those weights;
- a model card in JSON.

The model card gives:

- the input contract: domain, normalisation, tile shape, conditioning
  channels and their ranges;
- the trained ranges of FWHM, pixel scale and signal-to-noise ratio;
- the version, the training run, and a hash of each artefact;
- the data licences.

A bundle is verified and installed through `seiza-download`, as catalogues
are. A host can also ship it. Bundles are not embedded in crates. Core ML
compiles the package on first load, and Seiza caches the compiled model.

### Tiling and parity

Inference cuts the image into tiles of the export's static shape, runs
each, and blends them. Within half the receptive field of a tile's inner
edge, its output saw the convolutions' padding rather than the image, so
that band takes no weight; past it, the tile's weight rises as a raised
cosine across the rest of the overlap. Neighbouring tiles therefore overlap
by at least the receptive field. At the image's own edge a tile keeps its
full weight, because whole-image inference pads there too. With that rule,
tiled output matches whole-image output to within float error, and a test
holds it there; a plain cosine over the whole overlap lets the padded band
leak into the blend.

Each tile carries its own PSF stamp and its own slices of the noise,
ratio, halo and bloom maps. Peak memory is a few tiles' worth whatever the
image's size, and the backends compile one tile shape. The model card
records the tile shape, the overlap and the zero-weight margin.

The export script writes fixture tiles with PyTorch's f32 output. Each backend
must match them:

- the ONNX Runtime CPU provider within 1e-4 relative;
- Core ML, CUDA and TensorRT within 0.01σ of the input's noise.

The Neural Engine is enabled only for a model that passes its own parity run.
Core ML parity runs on macOS CI. CUDA and TensorRT parity runs on the NVIDIA
machine before each model release. Results differ slightly between backends,
so hosts include the backend and precision in their cache keys.

### Native API

Following the [Rust-first rules](../../AGENTS.md#the-rust-api-comes-first):

```rust
let backend = Backend::detect(&BackendOptions::default())?;

let model = NoiseModel::open(&noise_bundle, &backend)?;
let noise = NoiseMap::from_stack(&snapshot)?; // or NoiseMap::estimate(&image)?
let result = model.denoise(&image, &noise, &DenoiseOptions::default())?;
// Per-band amounts of the removed noise: no second inference.
let image = result.blend(&NoiseBands { high: 0.9, low: 0.5, scale_px: 5.0 });

let psf = PsfField::measure(&image, &PsfFieldOptions::default())?;
let sharpened = PsfModel::open(&psf_bundle, &backend)?.sharpen(
    &image,
    &psf,
    &noise,
    &SharpenOptions {
        target: SharpenTarget::Ratio { stars: 0.5, extended: 0.75 },
        halo_reduction: 0.3,
        bloom_reduction: 0.0,
        ..Default::default()
    },
)?;
// sharpened.report holds the reblur χ² and the trained-range checks.

let StarLayers { starless, stars } = StarModel::open(&star_bundle, &backend)?
    .separate(&sharpened.image, &NoiseMap::estimate(&sharpened.image)?)?;
```

The types follow the same rules:

- `Backend` is an enum: Core ML with its compute units, or ONNX Runtime with
  its execution provider. It has `FromStr` and `Display` for a
  `--ml-backend coreml|tensorrt|cuda|cpu` flag.
- `SharpenTarget` is `Ratio { stars, extended }` or `Uniform`.
- `halo_reduction` and `bloom_reduction` are amounts from 0 to 1, applied
  inside the star mask.
- `NoiseBands` holds the high- and low-frequency amounts and the scale that
  divides them.

Refusals are variants of a `RestoreError`, made where they happen:

- `WrongDomain`;
- `BayerInput`;
- `ShapeMismatch`;
- `TooFewStars { found, needed }`;
- `ModelContract`, for a bundle whose contract this build cannot honour;
- `BackendUnavailable { backend, reason }`;
- `Cancelled`.

An input outside the trained ranges is a warning in the result's report. It
becomes a refusal only when `strict` is set. The C export and the Python twin
of each operation land in the same PR as the operation.

## Pipeline order

1. Background
2. Colour calibration
3. Sharpen
4. Denoise
5. Separate stars
6. Stretch
7. Display tools

Colour calibration measures stars before anything reshapes them. Sharpening
comes before denoising because it needs real star profiles, noise included,
and a denoiser changes those profiles. After sharpening, the stack's variance
no longer describes the noise, so the next step estimates a map. That is why
the noise model trains on estimated maps and on sharpened pairs. Each learned
step can be replaced by its classical baseline without changing the order.

## Evaluation gates

A model is promoted only when all of these pass:

- improvement against known truth on the synthetic holdout;
- no new detections in pure-noise and blank-sky injections, and no structure
  copied from other training targets;
- identity behaviour on inputs that are already sharp, have low signal, or
  are unsupported;
- photometry within bounds and astrometric residuals no worse than the
  input;
- blind review by people at identical display stretches, using the
  [corpus note](../benchmarks/2026-07-deconvolution-corpus.md) convention;
- a measurable advantage over its classical baseline and a no-op, with an
  external learned tool as a reference only.

Each model also has gates of its own.

**Noise model:**

- unbiased MSE against deep stacks;
- mean residual in background and faint-signal boxes within 0.05σ;
- a flat residual spectrum;
- star flux and FWHM unchanged within 1%;
- injected Gaussian blobs at 1–3σ keep at least 80% of their amplitude.

Every noise run also makes contact sheets, so runs are judged by eye
beside their measures. The store lists a few chosen images: nights from the
validation sets, from the held-out telescope and from a training set, mono
narrowband and broadband, bright structure and the faintest. For each, one
sheet takes a 4-frame half as the input and the night's deep stack as the
reference. The other takes the deep stack itself, with nothing deeper, and
shows in place of a reference the noise taken out, magnified, where any
structure is detail the run removed. Each row is a 1:1 crop. The panels
show the input, then everything the run finds taken out, then the same
fixed shares kept, all under the reference's stretch. The shares never
change, so one run's sheet can be laid beside another's. A night's deep
stack is far less noisy than anything the noise model trains on, and the
first large runs, measuring at the floor on its halves, still took out a
trace of edges and bright-star profiles there. The deep stacks' halves are
measured the same way as the noise pairs, under their own size.

**PSF model:**

- FWHM ratio achieved within 10% of the ratio requested, on seeing splits and
  on the corpus fields;
- per-star flux within 1%;
- centroid shift below 0.05 px;
- reblur reduced χ² between 0.9 and 1.1, with a flat residual spectrum, on
  every real field;
- in uniform mode, the FWHM spread across the field halved, with the median
  FWHM moved by less than 5%;
- no dark ring deeper than 1σ below the background around bright stars,
  which is the failure the strong Richardson–Lucy setting showed;
- no such ring either when the conditioning PSF is 15% too wide.

**Star model:**

- residual flux at Gaia positions in the starless layer below 2% of each
  star's flux;
- star-layer flux within 2% of PSF photometry;
- no holes or rings, and no noisy patches: at removed bright stars, both the
  starless layer's level and its noise are within 10% of the surroundings;
- catalogued compact objects keep at least 95% of their flux in the starless
  layer;
- star-layer energy outside star masks below a bound;
- parallax renders reviewed.

## Product and provenance contract

Every restored FITS records:

- the operation, the model family and version, and the model hash;
- the backend and precision;
- the options and the conditioning values;
- the tile size and overlap;
- the software version and `seiza-restore`'s `OUTPUT_VERSION`;
- a summary of the trained-range checks and, for sharpening, the reblur χ².

The input is never changed. Hosts key cached results on the crate's
`OUTPUT_VERSION`, the model hash, the backend and the options. Every learned
operation ships marked experimental until its gates pass on the corpus. It
always has a classical fallback with separately named provenance.

## Staged delivery

1. **Real data pipeline.** Build the crawl, calibration plans, frame
   measurements, split groups and the per-set noise products in `training/`,
   with review pages. Done, with seeing splits, on
   `LiveStacker.reintegrate_frames`; see the [corpus passes](../../training/README.md).
   Still to do: masters built from the raw calibration frames of the
   telescopes the master library does not cover, and a stacking option that
   writes variance and coverage beside the stack, which Python already
   returns and the CLI does not.
2. **Foundations.** Land `PsfField` with empirical stamps, then field-varying
   Richardson–Lucy, then catalogue-guided star removal, each with its C and
   Python twins. These are useful without any model, and both learned models
   need them. `PsfField` has [landed](#what-landed), and so have
   field-varying Richardson–Lucy and the [reblur check](#reblur-consistency)
   ([deconvolution design](deconvolution.md#field-varying-psf)), and the
   fit, removal and rendering of [catalogue stars](#catalogue-guided-star-removal).
   Still to do: the bright stars' halo subtraction and the core fills.
3. **Smoke run.** R0 covers the export, the Core ML and ONNX Runtime backends
   with their parity tests, and the timings.
4. **Noise model.** R1 and R2 face the gates. Land `denoise` (experimental),
   and the classical wavelet baseline.
5. **PSF model.** Render composites and run R3a, then R3b, and land
   `sharpen` (experimental).
6. **Star model.** Review backgrounds and build composites with split noise,
   then run R4a and R4b. Land `separate-stars` (experimental).
7. **Second round.** R5, model cards, and a decision for each model on
   whether the evidence supports dropping the experimental mark.

The current real-corpus baseline and reproducible display comparisons are in
the [July 2026 corpus note](../benchmarks/2026-07-deconvolution-corpus.md).
