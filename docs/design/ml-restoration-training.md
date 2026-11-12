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

**Every model works from the image alone.** A shipped model processes a
linear FITS image from any stacker, given nothing else: no variance,
coverage, rejection or other record that Seiza's stacker writes. Whatever
else a model reads, Seiza makes from the image itself by fixed operations:
- a noise level or noise map;
- a PSF stamp from the PSF field measured on it;
- star and halo maps placed by its own plate solution.

Training may use everything Seiza's stacks hold, such as disjoint halves as
targets and variance for loss weights, since a trained model never sees
it. Where Seiza's own records do help, a model may have a variant that
reads them, for stacks Seiza made. That variant is optional and may be a
separate model. The image-only model is never dropped for it.

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
| Noise | `denoise` | linear image; its noise level, or a noise map made from it | clean estimate, its removed noise split into high- and low-frequency bands | disjoint sub-frame stacks | wavelet shrinkage driven by the variance map |
| PSF | `sharpen` | linear image, local PSF stamp, noise map, target-ratio map, halo and bloom maps | image at the target PSF | seeing-split stacks, cross-instrument pairs | field-varying Richardson–Lucy |
| Star | `separate-stars` | linear image, noise map | star layer; starless = input − stars | synthetic composites on real backgrounds | catalogue-guided PSF subtraction |

The star model feeds parallax rendering, which needs separate starless and
star images, and `star_stretch`, which stretches stars apart from the nebula.

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
use it as it stands. The starless images that pass review are
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
library. The library calibrates 744 sets; 198 of those have the 40 frames
noise pairs first needed, 16,369 frames in all. The rest lack a matching flat,
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
   set without a dark or a bias is left out, and the reason is recorded. A
   set without a flat is built all the same and marked `ready:no-flat`: the
   halves of a noise pair share their calibration, so an uncorrected flat
   field is signal both halves hold, not noise. A model that learns the flat
   field's work, such as sharpening, and the multi-night stacks leave these
   sets out.
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

- **noise pairs** need at least 20 accepted sub-frames, which give pairs of
  4 frames against the other 16 or more; pairs of 8 need 40. (The rule was
  40 until 2026-11-03, when it left 105 built train sets of 20 to 39 frames
  without pairs. A target of 16 frames is noisier than one of 36, which
  slows training but does not bias it. `recipes` decides the sets' recipes
  again from their measurements, and `build --pairs-only` builds the noise
  pairs of sets already built, leaving their other products as they are.)
- **seeing splits** need a FWHM difference of at least 25% between the best
  and the worst quartile, and 32 accepted frames. (On 2026-11-05, 30 of the
  699 built sets with a flat had one, 26 of them on the C925: a night's
  quartiles differ by 5% on the median set of 32 frames or more, and by
  17% on the C925's. The frames' widths are measured as a frame needs,
  smoothed first, and smoothing widens a narrow star most: Gaussian stars
  3.5 and 5.9 pixels wide measure 5.5 and 7.0, so the 1.2–2.6″/px
  astrographs, whose stars are near 3 pixels, measure near 5.6 whatever
  the seeing, and need 50% or more of real difference to pass. None of the
  325 sets without a flat spreads 15%, so letting them in would add none;
  they stay out of sharpening in any case. The multi-night pass judges its
  pairs on their stacks, measured unsmoothed, as below.)
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
| Multi-night deep stack, with variance and coverage, its halves and its seeing quarters | every accepted frame of one object, telescope and filter, over all its nights; the halves hold whole nights; the quarters are the best and worst of all its frames by FWHM, each with its measured PSF | references, pairs deeper than any night, PSF pairs |

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
manifest's paths. Its seeing quarters are the PSF model's seeing-split
pairs over many nights; `--seeing` also stacks every object whose frames
give one, however few its hours.

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

Stacked on one reference with the same normalization, the halves still do
not agree in flux. In the first corpus, B's bright flux was 0.58 to 1.42
times A's, pair by pair, and a tenth of the pairs were more than 2.5% off.
A model cannot tell one pair's difference from the next, so it learns
their mean. Before training, then, B is matched to A. Each half's
background is a plane fitted over the same faint pixels, those where B
blurred widely is below its median, leaving out the pixels either half
puts four times its spread from its plane: a short stack keeps hot pixels
its rejection missed, and left in they lift the plane. The gain is the
ratio of the halves' flux above those planes over B's bright pixels,
grown to take in the stars' wings. (A least-squares gain will not do: on
the sky, B's noise has nothing to do with A's, and drags the slope down.)

Where haze has spread light into halos about the stars in one half, no
gain fits. In one validation pair A held 1.27 times B's flux in the stars'
wings and 0.98 in their cores, and matched, its brightest cores were 19%
apart; a model trained on such a pair learns to add or take away halos. A
pair whose matched halves differ by more than 3% in the wings (200 to
1,000σ̄ above the background) or in the cores (above 1,000σ̄) is left out,
which is one train pair in 20. Measured against matched halves, on the
pairs that agree, every run's error on 4-frame halves fell by about 40%:
much of what had been measured was the halves' own difference. Trained on
matched halves that agree, a 41M model trained 80k steps dimmed bright
structure half as much, 0.3% against 0.6%, and its error on 4-frame halves
fell 3% overall and 14% on faint signal, measured against the halves as
they are.

Matched, B takes A's background, and a short half's background is the
higher: on the validation pairs of 4 frames, A's faint sky stood 0.03σ̄
above B's, in every pair, most likely because the rejection of 4 frames
lets more of their outliers through. A model trained on halves as they
are learns to bring its input's sky down toward B's; one trained on
matched halves keeps its input's level. On the sky, the matched models
raise the median of their input by about 0.04σ̄, the others by 0.02σ̄,
and against the clean targets they sit 0.05σ̄ high, the others 0.03σ̄.
That is the input's level kept, not a fault in the matching, but 0.05σ̄
is the limit the gates below allow.

Synthetic noise is the minority. It extends the data to signal-to-noise ratios
the corpus lacks: Poisson–Gaussian noise per synthetic sub-frame, passed
through the same resampling kernel, added to a deep stack.

A short stack also keeps what its rejection missed: hot and cold pixels
and cosmic-ray tracks. Training adds synthetic ones to the input, never to
the target, each spread over its neighbours as Lanczos registration
spreads a frame's pixel. Two things decide what they cost. The first is
their floor. Starting at 5σ̄, the height of a faint star, they taught a
model to take about a tenth of the faint stars' flux with them. Starting
at 10σ̄ (cosmic rays at 20σ̄), at one per 256 × 256 pixels, a quarter as
many, they are taken out as fully, and faint stars keep their flux as
well as in a model trained without them. The second is their scale. In
units of the image's typical noise σ̄, an outlier on a bright nebula,
whose noise is many times σ̄, is a bump of ordinary noise, and the wider
and longer-trained models learned to shave such bumps: they dimmed bright
structure by up to 0.7%, most of it away from the stars. So each outlier
is scaled to the noise at its own pixel, and stands as far above a nebula
as above the sky. Trained so, on matched halves that agree, a 23M model
trained 80k steps dimmed bright structure by 0.1%, where the same model
trained on outliers in σ̄ and halves as they are dimmed it by 0.7%. On the
validation set where the dimming was worst, it took 0.15% from the
extended structure between 200 and 2,000σ̄ off the stars, against 3.75%.
It also had the lowest error of any run on 4-frame halves (against
matched halves that agree, 5% below the 41M model's), on 8-frame faint
signal and against the clean targets. It left a little more of the
coarse noise on empty sky, 4.0% against 3.0%.

Once the PSF model exists, a further variant runs the frozen PSF model on both
A and B. Their noise stays independent, so the noise model also learns the
noise of a sharpened image.

### Noise map and strength

The per-pixel noise is `σ = sqrt(variance / effective coverage)`, from the
stack. Stacks made elsewhere carry no variance, and nor does an image after
sharpening. So the shipped noise model cannot depend on it.

The network has read a map modelled from the stack's variance: the sky's
variance as medians over 32-pixel blocks, plus a term growing with the
image's own signal. It reads only the map's shape, as `σ / σ̄`. Training
scales that plane by up to 1.65 times either way per tile, so the network
learns not to trust its level. One number, the image's median noise `σ̄`,
sets the units the image is read in, and every stacker writes a different
scale.

Without the stack's variance, there are two candidates:
- **Blind:** the network reads only the image, in units of its median noise
  measured from the image's own second differences, the estimator the
  `snr` module uses. It finds where the noise changes itself.
- **Image map:** the same map made from the image. The variance of its
  second differences, as medians over the same 32-pixel blocks, gives the
  sky term. The signal term is fitted on the blocks a nebula fills.

On 24 train sets' 4-frame halves, against the noise of the halves'
difference on the sky:
- the image map read 1.15–1.31 times it;
- the map modelled from the stack's variance read 1.19–1.61 times it.

So the image alone gives the steadier level.

The best model trained with the stack's map was measured on the rebuilt
validation sets, unchanged, reading three maps in turn: the stack's
modelled map, the image map, and a constant, the image map's median
everywhere. Against B halves matched to A's flux:

| Map read | 4-frame error | 8-frame error | Faint, 4 / 8 frames | Against clean targets | Empty sky's grain and mottle left |
|---|---|---|---|---|---|
| The stack's, as trained | 1.709 | 1.184 | 5.02 / 1.88 | 0.757 | 2.2% / 3.2% |
| The image map | 1.760 | 1.181 | 5.17 / 1.76 | 0.757 | 2.0% / 3.2% |
| A constant | 1.761 | 1.192 | 5.18 / 1.86 | 0.762 | 2.0% / 3.2% |

The constant does as well as the image map, so the model barely uses
where the map changes, only its level. Read from the image instead of the
stack, that level costs 3% on 4-frame halves and nothing on 8-frame ones.
Against clean targets and on empty sky, nothing changes.

The 3% is a mean, and most pairs lose less:
- with the constant, a 4-frame pair's error changed by +1.4% at the median,
  from −3% to +22%;
- an 8-frame pair's changed by +0.3% at the median, and at most +6%.

Level does not explain the worst pairs. On nine pairs from both ends and the
middle, the stack's level was 0.85–1.31 times the image's, and the gap did
not follow the change in error: a gap of 30% cost 1.3%, and the worst pair's
gap was 5%. Why those pairs lose is not known. Ten of the twelve validation
sets are nights of the Elephant's Trunk, so other targets are barely tested.

Stacks made by another stacker test the level on images Seiza never
touched. Two models trained with the stack's map were run on 16 such light
masters from seven telescopes, three of them drizzled 2×, reading a
constant at the image's level. With no truth, the measures are taken on
empty sky: how much of the noise is left at the pixel scale and binned
4 × 4, from second differences; pixels put more than 100σ̄ outside their
input's neighbourhood; and how far the result moves over 64-pixel blocks.
- On the 13 plain masters, 2–5% of the pixel-scale noise was left, and
  2–40% at 4 pixels, the most where faint nebulosity passes for sky. At
  most 3 pixels went outside. The model trained with the gradient and the
  binned share moved the 64-pixel level by at most 0.045σ̄; the narrower
  recipe moved it by up to 0.16σ̄.
- On the drizzled masters, 18–20% was left at the pixel scale and about
  70% at 4 pixels, as fine specks. Drizzle spreads each pixel's noise over
  its neighbours, so second differences read it at about half its level,
  and the model is told the image is cleaner than it is.

So the level now accounts for how the noise is spread. An image's
correlation is twice the second-difference noise of the image binned 2 × 2,
over that of the image itself, on the sky; white noise gives 1. Seiza's
4-frame train halves read 1.12–1.45, median 1.24; the other stacker's plain
masters 1.22–1.38, and its drizzled ones 2.45–2.58. Where an image reads
above 1.24, its level is raised in proportion. On the drizzled masters that
left 4–6% of the pixel-scale noise and 12–25% at 4 pixels, and the specks
went. The plain masters were raised by 11% at most and barely changed. The
blind runs train with this level. Should it fall short, pairs upsampled or
drizzled 2× can join training, the other way from the binned share.

Seiza's own 2× drizzle reads the same: four sets stacked plain and drizzled
from the same registration read 1.25–1.29 plain and 2.51–2.90 drizzled, and
with the raised level the models left 5–7% of the pixel-scale noise on the
drizzles, as on the other stacker's. One of the four, a 12-frame Hα night of
the Elephant's Trunk with a dark but no flat, holds about 190 dark spots of
10–16 pixels, down to −2,700σ̄ plain and −3,500σ̄ drizzled, far below zero.
They are most likely hot pixels the dark took too much of (not yet
checked in the frames). With a dark, Seiza maps no defects, and rejection
keeps what every frame has.
Every model filled them and overshot, turning a hole into a spike up to
+3,300σ̄ above the sky, ringed by speckle; two leftover hot clusters were
ringed by a dark moat of about −220σ̄. Whatever made a spot, the model is
not the place to decide it is a fault, and ringing it is never right. So
training now plants dark spots (`--dark-spots`) in both halves alike, 30 to
5,000σ̄ deep and 1 to 16 pixels, half spread as registration spreads a pixel
and half sharp as drizzle leaves them, for the model to keep as they are.
The cold pixels the outlier plant puts in the input alone stop at −30σ̄, so
a shallow one is still taken out.

The run that took the sky out, with the gradient and a tenth binned, on the
restacked corpus, broke on the drizzled masters' edges. Where a drizzle's
coverage thins, over its last 100–200 rows or columns, the noise rises to
1.3–1.8 times the image's and is spread wider still. There the model moved
pixels by up to 4,100σ̄, 1.6 million of them on one 2× drizzle, where the
input stayed within 20σ̄. The input was otherwise ordinary, and with rows
cut off the image the band stayed on the same rows, not on the same tiles.
Synthetic sky reproduces it: Gaussian noise spread over boxes 2–4 pixels
wide, at 1.5 to 3 times the level the model is told, made the same model
step up to 6,200σ̄. Of 16 models probed so, one more blows up: an earlier,
narrower one reading the plain sky, by up to 13,300σ̄ at 1 to 1.5 times.
The other 14 stay within 61σ̄, most within 30σ̄. The sky, the width and
the rewinds do not explain it: the first rewound five times, its loss
once reaching 10¹³, but the second never rewound. Some trainings end with
weights that throw a texture they never saw, and the halves' measures do
not show it. So every run now records its largest step on spread noise,
20 cases of spread and level (`stability-noise` probes a run trained
before), and training can spread a share of its tiles over a box 1.5 to 3
pixels wide at a random offset, both halves and the map alike, as drizzle
spreads a frame's pixel (`--spread`). The two runs trained with 15% of
their tiles spread stay within 31σ̄; what they cost is in the runs below.

So a model needs only the image and its noise level, which the image gives.
Runs trained blind are being measured against runs that read the stack's
map. The image map is the fallback if the blind runs lose. A model that
reads the stack's map stays possible as an optional variant for Seiza's own
stacks.

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

A user's strength control is the share of what the model finds that is
taken out, below, so it needs no map and no separate model per strength.

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
   smaller PSF. An object stacked over many nights is split the same way
   over all of them, the best frames of every night against the worst: both
   halves far deeper, and the ratio wider than one night's spread of
   seeing gives. A pair teaches sharpening only where its halves differ in
   the PSF alone, so each half's PSF is measured, never assumed, and the
   model reads its shape as well as its width. Whatever moves with seeing,
   such as airmass, focus drift, wind or haze, is checked per pair:
   background, flux and PSF shape.

   The `multi-night` pass makes these pairs. An object's frames screen it:
   their quarters' mean widths must differ by 10%, since the frames were
   measured smoothed. Its stacks decide, measured without smoothing, on
   the stars found in both. A pair is kept but marked rejected, with the
   reason, unless the worst half is 25% wider and wider in each ninth of
   the field (focus or tilt that moved between nights), its stars' axis
   ratio falls by no more than 0.05 (wind, tracking; seeing is a round
   blur), its stars' flux in wide apertures and its extended light's in
   64-pixel block medians differ by one gain within 5% (haze takes light
   from the cores into halos, which an aperture loses and a block keeps),
   and its skies differ by one plane within a quarter of the noise of the
   halves' difference. The gain and the plane are recorded, not rejected,
   for the trainer to match, as the noise trainer's `--match-flux` does:
   both are normalization's. Local-background normalization measures each
   frame's photometric gain in a 5-pixel aperture, which loses more of a
   wider star, so it brightens frames as their seeing worsens: a synthetic
   night whose seeing grows from Gaussian σ 1.6 to 3.6 px is scaled from
   1.01 to 1.66, and the Elephant's Trunk's OIII worst quarter (C925, 9.1
   px) by 1.7, its noise with it. Its tiles' offsets, medians that a
   blurred frame's star wings lift, leave its sky a little low, and take
   back part of the gain where a nebula fills a tile: the Trunk's Ha worst
   quarter has its stars brightened 2.1 times and its nebula 1.5, which no
   one gain matches.

   Measured unsmoothed, the six multi-night pairs stacked first differ by
   12% to 28% on the astrographs and by 43% and 69% on the C925, where
   their frames' quarters differ by 3.5% to 12% and by 32% and 48%: an
   astrograph's excess width is multiplied by 2.3 to 3.5, the C925's by
   1.4. One is kept, the Trunk's OIII (1.43, gain 0.59); three differ by
   too little, California's OIII loses its stars' roundness (axis ratio
   0.91 to 0.72), and the Trunk's Ha its flux. Over the 127 objects with
   two nights or more with a flat (21,638 frames, 1,125 hours), 58 pass
   the screen, 38 of them on the C925, 15 on Ultracat and 5 on the
   Redcat61; from those factors about 55 reach 25% on their stacks, 42 in
   the train split, before the other checks, which turned away two of the
   first three to reach it. One night's split is still judged on its
   frames alone.
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

### Training data

Two passes make the PSF model's real data. Both only read the manifest.

- **`psf-fields`** measures each source product's `PsfField` once, with
  no catalogue: the deep stacks, a night's and many nights', and both
  halves of every seeing pair, of the train and validation sets calibrated
  with a flat (never `ready:no-flat`). A field too sparse to vary is
  measured again as one global PSF; one too sparse for that is refused. A
  field cannot be saved, so its stamps and shapes are sampled on nodes at
  most 256 px apart and read back bilinearly. Each record, under
  `STORE/psf/<product id>/`, also keeps the stars the field used, every
  star the measuring detector finds unsmoothed (the seeing check's
  stars), the clipped cores the detector turns away, and the clip level,
  from the image's own peaks as `PsfField` finds it.
- **`cut-sharpen`** cuts 320 px windows onto local disk, each split's
  bank apart, as the noise bank is cut: memory-mapped, the same for a
  seed however many processes prepare it, and resumable to the same bank.
  The known-kernel bank holds deep stacks' windows with the stack's noise
  map, a mask of clipped cores, and the PSF stamp (129 px) and shape at
  the centre; the trainer blurs them. The seeing bank holds each pair's
  worst half as input and best half as target: the input's noise map
  made from its image alone, both stacks' own maps for the loss, a mask
  of either's clipped cores, both PSFs' stamps and shapes at the centre,
  and their FWHM ratio there, as the field's Moffat fit has it and as the
  stamps' own widths do (the fit fails on stamps sharper than about
  2 px). The target is put on the input's flux and sky by the check's
  gain and planes, and each pair keeps its stars' median centroid offset. A pair made before the check, every
  night's split among them, is judged by the cut with the same check, its
  verdict kept with the bank. The point sources are each input's
  detections until the Gaia pass. The cut also takes pairs a caller
  supplies, each with a record kept beside its windows.

The windows are raw. The sharpener is to train on the noise model's
output; those windows will be cut from the denoised stacks beside these.

First run (2026-11-08), 4 processes, 52 minutes for the fields and 7 for
the banks:

- **Fields.** Of 773 products, 755 vary, 8 sparse C925 stacks, most of
  them narrowband, take one global PSF from 5–24 stars, and 4 are
  refused. The deep stacks' centre FWHM runs 1.8–10.5 px on the
  astrographs (median 2.2: Ultracat, RedCat61, Askar107PHQ) and
  4.7–15.6 px on the C925 (median 7.5). The frames' smoothed
  measurements read about 5.7 px where the astrographs' stacks read 2.2.
- **Seeing pairs.** On the C925 the fields' centre ratio, best over
  worst, agrees with the multi-night checks' star medians (0.48–0.75
  against 0.50–0.78). On the undersampled astrographs it is far smaller
  (0.34 against 0.53, 0.55 against 0.66): the detector reads 2 px stars
  wide. One Ultracat object's worst halves (its night's split, and its
  many nights') carry a focus gradient as well as seeing, 2.7 px at one
  corner to 8.9 px at the other, with ring-shaped stamps; they pass the
  check, which asks for a round blur, wider everywhere.
- **Centroids.** On 17 C925 seeing pairs the halves' star centroids sit
  0.35–0.98 px apart, nearly all along one axis, though both halves share
  one grid. The C925's PSF is lopsided, its centroid pulled from its peak
  by a tail, and a sharper PSF pulls it less. A first cut that dropped
  pairs past 0.25 px lost them all; they are kept, with their offsets,
  and only a move past 1.5 px is taken for a fault. A stamp centred on
  its centroid cannot show the offset, so the trainer must shift the
  target or the stamp by it, until stamps are centred on Gaia positions.
- **Banks.** The known-kernel banks hold 20,480 train windows from 512
  stacks and 5,600 validation windows from 140 (19 and 5 GB). The seeing
  banks hold 31 train pairs (4,960 windows, 11 GB: 28 C925, 2 Ultracat,
  1 RedCat61) and 18 validation pairs (2,880, 6 GB, all C925), with
  ratios 0.24–1.13: 1.2% of train windows above 0.8, and 0.04% above 1,
  at field corners where the best half is the wider. Of the 59 pairs, six
  fail the shape check, one is too close in seeing, one worst half has
  too few stars for a field, and two fail the frames' screen. A night's
  split is judged by the cut: 20 of 27 pass.

Open: how the trainer uses the centroid offsets, whether to train on
focus pairs, and which window ratios to keep. The trainer's first
version answers each with a default, below, that the runs have still to
confirm.

### Trainer

`train-sharpen` trains the PSF model on these banks and `evaluate-sharpen`
measures a trained one. Runs are recorded under `STORE/models/sharpen/`
as noise runs are: the configuration, the history of each log window
with the loss on a few fixed validation batches, the weights, the ONNX
graph and the measures. A run is saved whole every thousand steps, and
one stopped goes on from there to the same weights it would have
reached unstopped, since each batch is drawn from its step alone.

- **Network.** The four-level residual U-Net, 32 wide (2.8 M
  parameters), with the stamp encoder: a 3×3 convolution, then stride-2
  3×3 convolutions from 129 px down to one, giving 64 numbers that scale
  and shift the channels at the bottleneck and after each decoder level.
  Its graph holds only convolutions, nearest upsampling, concatenation,
  addition, multiplication and the sigmoid of SiLU. It reads four
  planes: the image and the noise map as the noise model reads them, the
  ratio, and a point-source map, the input's detections rendered with
  the local stamp in `asinh`, left empty a tenth of the time. The noise
  map is made from the image alone. The stamp is read as two planes, over
  its peak and through `asinh`, which shows its wings. The output is a
  residual added to the image's `asinh` plane and taken back through
  `sinh`: on the sky a residual in σ̄, as the noise model's, and at a
  star's core nearly a log. A sharpened core differs from its input by
  hundreds to thousands of σ̄, and a residual in σ̄ that starts from zero
  never reached it: two smoke runs gave cores residuals under 1σ̄ where
  their targets asked 700σ̄ at the median, and narrowed no star.
- **Known-kernel pairs**, 60% of a batch. An elliptical Gaussian `K`, of
  ellipticity up to 0.3 at a random angle, is drawn so that
  `FWHM(P_in) / FWHM(P_ref)` is 1.1 to 2, both measured as the bank
  measures stamp widths. The input is the window blurred by `K`, with
  fresh white noise up to half the stack's own, as the sharpener will
  read the noise model's output. The target is the window blurred by the
  Gaussian of covariance `Σ_T − Σ_ref`, with `Σ_T` round at `Σ_ref`'s
  largest eigenvalue: second moments add under convolution, so the
  target's PSF is round in them. A stamp's second moments are its wings'
  as much as its core's, and rounded this way the train bank's targets
  are 1% to 42% wider than `P_ref`, 9% at the median. The widening is
  therefore drawn from no less than keeps the target narrower than the
  input. The network reads `P_in`'s stamp, never `P_ref`'s. A tenth are
  identity windows: `K` a point, the ratio 1, the target the input.
  Measured on the validation stacks' own stars, the targets come to 0.98
  to 1.06 of the ratio asked (5th to 95th percentile).
- **Seeing pairs**, 25%, and pairs a caller supplies in the seeing
  bank's format, 15%; without those, seeing pairs take their share. The
  target is moved by its pair's median centroid offset, through its
  transform, so the model is not taught to move stars. The ratio is the
  stamps' widths', which every window has. Windows whose target is wider
  than their input are left out, as are both pairs of the Ultracat field
  whose focus changes across it. On the validation pairs the targets'
  own stars come to the stamps' ratio, 1.00 at the median.
- **Sampling.** No object, a telescope, target and filter over all its
  nights, takes more than 3% of its bank's draws, or an even share in a
  bank of fewer than 34 objects: an object stacked over many nights is
  in the deep bank once for each night and again as their stack, all of
  one sky.
- **Loss.** Charbonnier error in units of the target stack's own noise,
  read no higher than ten times its tile's median, with clipped cores,
  and the band a blur or a shift reads past a window's edge, masked out.
  At a star's core a stack's noise map holds its frames' scatter with the
  seeing, thousands of σ̄: read in it uncapped, a blurred star cost all
  but nothing, and the first smoke run learnt to denoise its pairs and
  never to sharpen. The loss adds the same error on the error's
  gradient, averaged over 1, 2 and 4 px blocks and each times its side,
  and the Charbonnier error of each point source's flux within three
  times the input's FWHM, in units of that flux's noise, counted once for
  each pixel of its aperture, as a pixel's error is; each at a tenth.
  Those weights come from fitting one batch of known-kernel tiles. The
  stars are a two-hundredth of a tile's pixels and the targets' own noise
  fills the rest, so terms that add noise drown them: with the gradient
  term at half weight, or the flux term a mean over the stars, which
  pulled each aperture's pixels a hundred times as hard as the pixel
  term pulls its own, the cores never moved; with both at a tenth, as
  set, they reached four fifths of their targets in 300 steps.
- **Augmentation.** The eight flips and turns, alike for the image, every
  map, the apertures and the stamp. Exposure scaling scales a
  known-kernel window's signal by up to two either way before its noise
  is added; a pair's noise is its stacks' own, which a scale would move
  off its maps, so pairs are not scaled.
- **Evaluation** runs the validation banks' windows whole through the
  tiled inference that ships, each tile conditioned on its own stamp,
  and measures inside the network's reach of a window's edge: the error
  against the target before and after, in the target's noise; the width
  ratio, output over input, of the stars' own stamps, stacked from
  isolated detections and measured as the bank measures a field's,
  against the ratio asked; the flux kept within three times the FWHM;
  the deepest ring one to three FWHM out, as the lowest azimuthal mean
  and the lowest pixel, in σ̄; and the change on identity windows. The
  C925's validation windows hold about one detection each, so their
  star measures rest on few windows; whole images will give many more.

The first smoke runs (2026-11-09, PyTorch's Metal backend, 2,000 steps
of 8 tiles 192 px, 3 minutes each) proved the trainer, not a model. The first
three narrowed no star, which led to the asinh residual, the cap on the
target's noise and the light gradient and flux terms above. With them,
the training loss fell from 0.94 to 0.83 and the validation batches'
from 0.96 to 0.80. On 200 windows of each validation bank, the seeing
pairs' error against their targets fell to 0.72 of the input's, mostly
noise taken out; the known-kernel windows' stars narrowed by 2% where
30% was asked, so it has only begun to sharpen. Flux in the apertures
was kept within 0.5% at the median, no bright star but one in 177 had a
ring deeper than 1σ̄, and identity windows changed by 0.03σ̄ at the
median. Resumed from a checkpoint, a run gives the same weights as one
never stopped.

Left for later: reblur consistency on images without a target and a
centroid-shift loss, which join the loss's terms; perturbing the
conditioning stamp and the gate on a wrong PSF; random smooth ratio maps,
and the halo and bloom controls; composites with re-rendered stars and
Gaia's point-source maps; windows cut from the denoised stacks; and
contact sheets, whole-image evaluation and the bundle, which will call
the same inference.

### Space-telescope pairs

A stack's own windows, blurred, teach a model to give the stack back:
nothing in them is sharper than the stack's seeing. Hubble and JWST
mosaics are real sky whose nebulae, dust and galaxies are sharp to the
pixel, so a pair made from one has a truth sharper than any ground-based
stack holds.

**The source.** Drizzled mosaics from the public MAST archive, each a
science image with its weight map: 71 Hubble mosaics (156 GB) and 88
JWST mosaics, kept in the store under `external/`.

**The cut** (`cut-external`, `sharpen_external.py`):

1. A pixel counts where the mosaic's weight is at least half of its deep
   parts' (the 90th percentile). Where few exposures overlap, cosmic
   rays survive the drizzle and would be taught as sky. Those left are
   found by their sharpness and dropped.
2. The mosaic is brought down by a factor (`--bins`): smoothed by a
   round Gaussian of σ half the factor, then every factor-th pixel. A
   block mean in its place gave square stars. A pixel is kept where a
   quarter of its weight fell on pixels the mosaic holds.
3. The result is measured as a stack is: its PSF field and point
   sources by the PSF pass's code, its noise map from the image alone.
   A mosaic with too few stars for a field takes a round Gaussian, 2.4
   of its own pixels wide before the factor.
4. Windows (`--window`, 320 px or 1024 px) are cut where the mosaic
   covers. A window may miss 2% of its pixels, which are filled from
   about them and kept out of the loss.

The bank has the known-kernel banks' layout, and the trainer draws from
it beside the native deep bank (`--more-kernel-bank PATH:WEIGHT`).

| Bank | Factors | Window | Windows |
| --- | --- | --- | --- |
| Hubble, windows required whole | 2, 3 | 320 px | 842 |
| Hubble | 2, 3 | 320 px | 10,694 |
| Hubble | 2, 3 | 1024 px | 364 |
| JWST | 1, 2 | 1024 px | 1,173 |

Brought down two and three times, most Hubble mosaics hold no clean
window 1024 px across; that bank is being cut again at factors 1 and 2.

**The blur** (`kernel_example`, the same for native deep windows). Each
window carries its PSF stamp `P_ref`, measured where it was cut.

- **The kernel.** A widening is drawn evenly from `--kernel-ratios`
  (1.1 to 2.0 at first, now 1.1 to 3.0), with an ellipticity up to 0.3
  at any angle. `K` is the elliptical Gaussian that makes `P_ref ∗ K`
  that many times as wide as `P_ref`: sized from widths adding in
  quadrature, then refined on the stamp itself until the ratio is within
  1%. The input is the window blurred by `K`, and the network reads
  `P_in = P_ref ∗ K`.
- **The target** is the window blurred only as far as makes `P_ref`
  round in its second moments and no narrower than 1.5 px. With two
  strengths, a ratio is drawn for stars and one for the rest, each
  between the sharpest the window allows and 1 and nearer the sharpest;
  the target is the window at those two widths, joined by a mask
  rendered from the listed stars.
- **Exposure.** The signal above the background is scaled by up to 2
  either way.
- **Noise.** Fresh noise is added to the input, up to half the window's
  own (`--kernel-noise`), shared with its neighbours by a blur of up to
  0.6 px. With `--pass-noise` the same noise is added to the target, so
  the network is asked to carry it through and not to remove it. A
  space-telescope window has next to no noise of its own, so with
  `--external-snr LOW,HIGH` it is given noise of its contrast (the range
  of its 8 px means from the 10th to the 90th percentile) over a ratio
  drawn between the two.
- **Identity.** One window in ten is left as it is, asked for a ratio
  of 1, with itself as the target.
- **A real PSF for `K`** (`--real-kernels`): a stamp drawn from the
  native bank blurs the window where it widens `P_ref` within the
  ratios. Tried at ratios up to 4 with reblur windows, it cost round
  stars and is off.

Open: ten to twenty mosaics fail in the cut with a division by zero,
not yet traced.

### What the sharpening runs showed

Runs since (2026-11-09, width 32, four levels, 20,000 steps of 8 tiles
unless said; about 8 minutes each) are judged on whole images: contact
sheets beside a reference sharpener at a mild and a strong setting, star
width, flux, rings, cross and roundness, noise gain, and the rims of
nebulae.

- **Stars.** Runs reach the width asked. Moving each pair's target by
  its light's offset keeps stars in place. Passing the input's noise
  through to the target (`--pass-noise`, with the known-kernel noise up
  to the stack's own) ended the rings and leaves the sky's noise at 0.99
  of the input's. A bank binned by block means put a cross on stars and
  was dropped (see "Narrow stars gain flux" below for its successor).
- **Two strengths.** With `--nonstellar` the network reads one ratio for
  stars and one for the rest, and a known-kernel window's target is the
  two blurs of the window, each where its part of the light is.
- **Length.** 20,000 steps is short. On held-out known-kernel windows
  such runs give 0.6 to 0.75 of the target's edge gain at asks of 0.4 to
  0.85; at 100,000 steps, 0.92 to 0.94. A wider and deeper network
  (64, five levels) at 30,000 steps gave no more than the small one.
  Asks under 0.4 are next to untrained with kernel ratios up to 2.0.
- **Space-telescope windows.** Drizzled Hubble and JWST mosaics from the
  public archive, brought down two and three times (JWST one and two),
  are a known-kernel bank whose truth holds edges sharper than any
  seeing, blurred in training by the native bank's PSFs
  (`cut-external`, `--more-kernel-bank`). Required whole, a window was
  lost to one cosmic ray's pixels, and over half the mosaics gave none;
  a window may now miss 2% of its pixels, filled from about them and
  kept out of the loss, and the Hubble bank went from 842 windows to
  10,694. The larger bank made stars rounder and edges no sharper.
  Blurring by real PSFs at ratios up to 4 with reblur windows cost round
  stars and is set aside.
- **The edge term** (`--edge-weight`) is the share of the target's
  smoothed gradient the output falls short of along the target's own
  direction, away from the listed stars; going past the target costs
  nothing in it. At 2 a 20,000-step run gives 0.77 to 0.87 of the
  target's edge gain and stars as round as before. At 5 it goes past
  the truth at mild asks, up to twice, and leaves a dotted texture in
  dark nebula on denoised inputs.
- **The measure that misled.** A whole-image edge gain (smoothed
  gradient at the strongest edges, output over input) put such a run
  level with the reference's strong setting, which the sheets did not
  bear out: its rims of dark nebula were barely crisper than the
  input's. The measure used now averages thousands of profiles taken
  across the rims, which leaves the rim and averages the noise away,
  and gives the rise from 20% to 80% in pixels. On the Elephant's Trunk
  the input's is 5.7 px, the runs' 4.1 to 4.5, the reference's 4.6
  (mild) and 3.7 (strong).
- **Why the rims stay soft.** On the same image denoised first, the
  runs do draw crisp rims. The space-telescope windows, which alone
  hold edges sharper than the seeing, were met with their own noise,
  next to none. A native stack's own noise is in its target, where no
  run can follow it, so under noise the native windows teach caution.
  `--external-snr LOW,HIGH` gives each external window noise of its
  contrast over a ratio drawn between the two. Runs with it showed no
  change at the rims by eye.
- **The trained range.** With kernel ratios up to 2.0 a run never meets
  an ask much under 0.5. Asked for 0.3, the longer and the wider runs
  smooth the image and lose faint stars, while the same runs are sharp
  at asks they were trained on. Runs now draw ratios from 1.1 to 3.0
  (`--kernel-ratios`), the sheets compare at asks inside the range, and
  a panel asked for less is marked as below it.
- **Small tiles.** At four levels and 320 px windows the network sees
  little beyond a rim's own width. The smaller and shorter runs leave
  speckle along the lit edges of dark nebula; a 100,000-step run leaves
  less.
- **More context** (2026-11-10). Banks cut again with 1024 px windows
  (`--window`), the Hubble bank brought down one and two times where it
  was two and three; six levels at width 32 (42M parameters), 768 px
  tiles in batches of 4, 30,000 steps at half the learning rate, kernel
  ratios 1.1 to 3.0, the edge term at 2; about 70 minutes. Whole images
  at stars 0.46 and the rest 0.5:
  - On the Elephant's Trunk the rims rise from 20% to 80% in 3.76 px
    (input 5.72; the small runs 3.94 to 4.18; the reference 3.74 at its
    strong setting, 4.64 at its mild), their steepest slope 2.17 times
    the input's (the reference 2.21). What the run changes follows what
    the reference changes: a correlation of 0.71 at the pixel scale
    (the small run 0.61) and 0.88 to 0.91 from 1 to 8 px.
  - Stars there come to 0.47 of their width with their flux kept
    (1.002), and the sky's noise passes at 0.99 (pixel) and 1.00 (4 px).
  - By eye the rims are crisper than the small runs' and still carry a
    grain along the lit edge that the reference's do not.
  - About the 40 brightest stars on bright nebula in the California
    there is no dark surround (the lowest ring 16 to 56 px out is at
    +0.05σ at the median, the input's +0.03), where two of the small
    runs dipped to −0.2 and −0.3σ.
  - A network of six levels has an overlap of 532 px, more than a
    512 px tile: such models run on tiles twice their overlap (1280 px).
- **Attention bought nothing.** The same run with two transformer
  blocks at its bottleneck (`--attention 2`, 59M parameters; a 768 px
  tile is 24 × 24 positions there) ended at a validation loss of 0.849
  against 0.841. With the blocks taken out of the trained network its
  output on a rim's crop moves by 0.007σ rms, where the run itself
  changes that crop by 1.5σ; over the whole image, by 0.3σ against
  26σ. Its output is the same on 768 px and 1280 px tiles to 0.003σ
  rms. So the
  two runs are two draws of one recipe, and their difference is the
  spread between runs: rims of 3.76 and 3.59 px on the Trunk, median
  star ellipticity 0.021 and 0.034 there, 0.074 and 0.065 on the
  California. Attention is not kept.
- **Narrow stars gain flux.** On every sheet whose stars are 2.3 to
  2.5 px across (five of six) the larger run adds 6 to 15% to a star's
  flux; the small run up to 7%; the reference stays within 3%. Stars 8
  and 11 px across keep theirs to 2%. On the California the gain is
  10% out to eight widths, so it is not light moved in from the wings;
  the faint third of the stars gains twice what the bright third does,
  and a denoised input half as much.
  - It grows as the ask nears the 1.5 px floor. On the California
    (2.33 px): 0.4% at an ask of 1.0, 3% at 0.8, 5% at 0.75, 8% at 0.7,
    12.5% at the floor (0.65). The stars' width follows the ask only so
    far: 0.87, 0.68, 0.66, 0.64, 0.62 of the input's. At the reference's
    width (0.65) the run is 5% high.
  - The known-kernel banks hold next to no star under 2 px (3% of the
    native windows, 7% of the Hubble bank's), so an ask that takes a
    2.3 px star to 1.5 px is one the runs hardly met.
  - `cut-sharpen --bank binned` now brings each stack down as the
    mosaics are, through a round Gaussian and then every `bin`-th pixel,
    by the factor that leaves its stars nearest 1.5 px. At 1024 px
    windows that is 2,334 windows of 322 stacks, their stars 1.45 to
    1.78 px across; the 190 stacks of wide stars, brought down five and
    six times, are smaller than a window.
  - Its stars are rounder than the block means' and may not be round.
    Of stars that fall on a pixel's middle, the four diagonal
    neighbours hold 0.87 of what a round star's would, read from the
    pixels one and two out along the axes (block means, at the same
    width: 0.78). The unbinned stacks' stars read 0.90 and the Hubble
    bank's 0.94, but at 2.3 px, which is not like for like.
  - A run with the bank at half the native bank's weight puts no cross
    on the California's stars (0.10; without it 0.13 and 0.06) and
    takes the flux gain from 12.5% to 9.1%: at the floor only, and at
    the reference's width it is 5% high as before. Narrow stars in the
    known-kernel banks were not what was missing: on the new bank's own
    windows the run without it leaves stars 7% *short* of their targets.
  - **The gain is there in training, and the edge term is a small part
    of it.** On the
    supplied pairs with targets under 1.8 px the larger run's stars hold
    9% more flux than their inputs, where their targets hold 2% and
    wider targets' stars 1.5%. A tile lists 32 stars at most for the
    flux term and keeps the edge term off those; on tiles of narrow
    stars two in three are not listed, and the edge term, which counts
    falling short and never going past, asks each of them for a steeper
    star. Along "every star a little brighter" its slope at the stars
    not listed is -2.0 and the other terms' together +0.16; at the
    listed stars the others' is +1.07, the edge term's none. The
    network cannot tell which stars a tile lists, and settles between.
    Wide-starred native windows list most of their stars and show no
    gain (1.004 against their targets, listed or not).
  - `--edge-clear` keeps the edge term two input widths clear of every
    detected point source. With it the slope at the stars not listed is
    +0.16. A run with it took the California's gain from 12.5% to 10.8%
    only, and with the brought-down bank as well to 7.1%: the slope said
    which way the term pulled, not how far the stars had been pulled by
    it. What holds the rest of the gain is not found.
- **Roundness is the run's, not the ask's.** The larger run leaves the
  California's stars at a median ellipticity of 0.074 (input 0.092) at
  every ask from 0.85 to the floor; a small run and the reference leave
  0.044 and 0.046.
  - On held-out known-kernel tiles, whose inputs are blurred up to 0.3
    elliptical and whose targets are round, the larger run's stars are
    as far from their targets' shape as its inputs are: the quadrupoles
    of a star's error about its middle, over their noise, have a median
    of 5.3 after it and 7.6 before, and weighed as the loss weighs them
    come to the same sum. The pixel and gradient terms do not ask for a
    star's shape hard enough to get it.
  - **The roundness term (`--round-weight`) is not the way to ask.** At
    weight 5 it gives the sheets round stars (California 0.009, Trunk
    0.021) and a flux within 2% of the input's, but the stars stop at
    0.70 of the input's width where they were asked for 0.65 (flux
    against width lies on the same curve as without the term: it did
    not mend the gain, it kept the stars from the widths that have it),
    the California's rims steepen ×1.29 where they steepened ×1.66, and
    the change the run shares with the reference at the finest scale
    falls from 0.87 to 0.65 of the reference's. On held-out tiles its
    edge term is up a tenth and its flux term doubled. At weight 20 all
    of it goes further: stars at 0.88 of the input's width on the
    California, the Trunk's rims rising over 4.44 px where they rose
    over 3.76, the shared change at 0.48.
  - The term reads what is not a star's shape. It takes the moments of
    all the light above an aperture's mean, and an aperture three
    widths out is mostly noise and nebula: the *targets* of held-out
    tiles read 0.15 to 0.32 by it. Pushed toward none, the run's output
    read 0.04 to 0.09: its stars did come nearer their targets' shape
    (quadrupoles of 4.4 at the median), but by the term's own measure
    taken against the targets' its apertures ended further from them
    than its inputs' were (0.18 against 0.16 on known-kernel tiles).
  - The shape term (`--shape-weight`) asks through the error instead:
    each listed star's two quadrupoles of output less target, in a
    Gaussian window of about a sharpened star's width, each over its
    noise, weighed as the flux term weighs its sums. It is linear in
    the error, so noise and nebula the output shares with its target
    cancel, a wrong level or place has no quadrupole, and a star is
    asked to be as round as its target's and no rounder.
  - At 0.1, the flux term's weight, it leaves the California's stars at
    0.048 (the reference 0.046) and the Trunk's at 0.020, and costs the
    rims nothing: the Trunk's steepen ×2.61 (without it ×2.17), the
    California's ×1.67, and the change shared with the reference is
    0.82 to 1.10 of the reference's at every scale to 8 px. At 0.3 the
    California's stars are less round (0.057) and its rims steepen
    ×1.36. It is kept at 0.1.
- **The reference's contrast is broader than a Gaussian's.** Across the
  Trunk's rims, out to 32 px either side, the reference's strong setting
  darkens the dark side 8 to 16 px out about twice as much as the larger
  run does (0.68 against 0.30 at 8 px, 0.42 against 0.17 at 12, in the
  image's own units, in which a pixel's noise is 1.4 there and 0.5 on
  the multi-night Trunk; so are the rims' levels below) and lifts the
  lit ridge a quarter more. Asking the run for
  more of the rest (0.5 down to 0.25) makes the rim far steeper than the
  reference's (×4.7 against ×2.2) and the ridge too tall, and leaves
  the dark side short still (0.49 at 8 px at an ask of 0.33). A Gaussian
  K changes nothing further than about its own width from an edge, so no
  known-kernel window asks for more; a real PSF has wings. (The change
  by band beyond 8 px, which reads the same at every ask, is the stars'
  halos and says nothing of nebula.)
  - `--kernel-halo MAX` gives each known-kernel window's input PSF a
    halo: a share drawn up to MAX of the window's light goes through a
    round Gaussian two to four times as wide as K's core. The target
    keeps of the halo what its ask leaves between the sharpest its stack
    allows (none) and 1 (all), stars and the rest each by their own. The
    stamp the network reads holds the halo, as a measured PSF's does.
    Taking a halo of share h back is, to first order, an unsharp mask of
    the halo's width and amount h / (1 - h); but a run taught this way
    takes back the halo it finds, and a real image's may be slight.
  - **A masked unsharp mask gives the dark side.** Applied to a run's
    Trunk output (stars 11.5 px): what stands above a 12 px Gaussian in
    the output smoothed by 4 px, at amount 1, on nebula and off stars,
    with what stands above added at a quarter of what lies below. The
    rims' dark side is then 0.64 under the input
    8 px out (the output as it was: 0.33) and 0.41 at 12 (0.17), the lit
    crest a tenth higher, and the knots along the rims are as they were.
    Read from the output itself at 1.5 px it roughens the rim (knots
    from 2.1% to 3.6%), and weighed alike both ways it lifts the lit
    side by two thirds. On the same field at 7.9 px the same amount
    does as much; at 2.3 px (the California) a tenth of it already
    roughens the rim.
  - `--contrast AMOUNT` asks for it in the target: a known-kernel
    window's rest is given AMOUNT times one less its ratio of that
    unsharp mask, read from the truth before any noise is added, its
    widths in input PSF widths, none of it for stars 2 px across and all
    of it from 7 px up. The target is then no longer the truth: it is an
    enhancement, and it follows the ask.
  - It is masked three ways. By place: on nebula, and clear of every
    listed point source and of any compact peak far above its
    surroundings (a field lists a few hundred stars, and a bright one
    left out was given a dark ring). By scale: the detail is read from a
    third of an input width up, which is what keeps a rim's knots as
    they were; read from 1.5 px with every other mask in place, the
    knots still rose from 2.1% to 2.4%. By what is there: detail counts
    as it stands over the noise of its own band, half at three times
    it, so a truth's faint mottle is not raised.
- **Speckle along lit rims.** Runs break a lit rim into knots a few
  pixels across where the reference draws a line. The measure: within
  8 px of the steepest edges, off stars, the share of pixels where a
  blob filter (1 px less 2.5, and 1.5 less 4) stands more than three
  times the sky's noise above the rim's own mean along itself. On the
  Trunk at 1 to 2.5 px: input 0.4%, the runs 1.7 to 3.1%, the reference
  1.0%; two draws of one recipe gave 1.8 and 3.1, so a run counts as
  better only well under 1.7. With stars 11.5 px across nothing real is
  1 to 4 px wide there. On the California the runs and the reference
  are level (2.7%).
  - It comes with the larger model. The small runs (four levels, tiles
    of 256 in eights, blurred up to twice) read 0.8 to 1.2%, the
    reference's 1.0%; every larger run (six levels 32 wide, tiles of 768
    in fours, half the steps) reads 1.7 to 3.3%.
    - Not the widest blur. The larger run blurred up to 2 in place of 3
      reads 2.0% (3.2% on the multi-night Trunk, the reference 2.3%)
      with the reference's rims: ×2.31 and a rise over 3.67 px against
      ×2.21 and 3.74. Two small runs alike but for the widest blur had
      read 0.95 and 2.0%, but both were rendered past what the first was
      taught.
    - Not the bank. The small recipe, 64 wide for 60,000 steps, read
      0.8% from the banks of 256 px windows and reads 0.9% from those of
      1024 (the multi-night Trunk 2.2% against the reference's 2.3, the
      California 2.8 against 2.7).
    - Not how much is asked. The larger run blurred up to 2 reads 1.6 to
      2.0% at every ask of the rest from 0.2 to 0.7, while its rims'
      rise goes from 3.30 to 4.24 px; at the small run's sharpness (4.24
      against 4.17 px) it reads 1.65% against 0.92%. Blurred up to 3 it
      had followed the ask: 1.0% at 0.7, 1.3% at 0.6, 1.8% at 0.5, 2.5%
      at 0.4.
    - The small run is not the answer as it stands. Its rims rise over
      4.17 px at an ask of 0.5, the least it was taught, and the dark
      side 8 px out is a third of the reference's (0.22 against 0.68);
      asked for less than 0.5 it stops sharpening the rest at all (a
      rise over 5.6 px at 0.4, the input's 5.7).
    - Blurred up to 3, the small run can be asked for less, and at an
      ask of 0.4 its rims are the reference's on the lit side: ×2.92
      against ×2.21, a rise over 3.86 px against 3.74, the ridge 5.3 and
      6.7 over the input 4 and 6 px in against 5.4 and 6.5. Its knots
      read 1.3% there (1.1% at 0.5; on the multi-night Trunk 1.4% against
      the reference's 2.3). Under 0.4 the ridge overshoots. The dark
      side stays near 0.3 at every ask.
    - Six levels 32 wide, the small recipe otherwise, are not the
      speckle on the deep Trunk (1.1%) and are on the multi-night one
      (3.6%), and by eye it leaves bright dots along a lit ridge's
      outer side; its California stars are less round (0.067 against
      0.043).
      The depth deepens the dark side to 0.47 with nothing asked.
    - Blurred up to 3 as well, six levels are the four-level run with
      more speckle. The rims are the same (×1.93 and a rise over 4.08 px
      against ×1.92 and 4.18) and the dark side is back at 0.27; knots
      read 1.3% against 1.1, on the multi-night Trunk 2.6% against 1.4
      (3.6 blurred up to 2), and the California's stars gain 12% of
      their flux against 7. Depth is left out of the small recipe.
    - Tiles of 768 in fours for 30,000 steps, the small recipe
      otherwise, are not the speckle either: 1.1%, and 1.5% on the
      multi-night Trunk (the California 3.1% against 2.8). They buy
      nothing at the rims (×1.95 and a rise over 4.26 px, the dark side
      at 0.18) and leave the California's stars less round (0.063
      against 0.053). So neither the depth nor the tile speckles the
      deep Trunk alone, the two together do, and the small recipe keeps
      its four levels and its tiles of 256.
  - Clean truth does not mend it (3.1%), nor the halo (3.3%), nor the
    edge term kept off stars (2.9%).
  - Contrast in the target is learned and makes it worse. A larger run
    taught a contrast of 2 on clean truth, with the gradient term out to
    16 px and before the mask by what is there, darkens the Trunk's dark
    side 8 px out by 0.38 (0.25 to 0.30 without, the reference 0.68) and the
    multi-night Trunk's by 0.40, the reference's own; its knots read
    2.9% and, on the multi-night Trunk, 6.5%. On the small run blurred up
    to 3, a contrast of 2 with every mask is learned on the multi-night
    Trunk (0.33 against 0.12 without) and hardly on the deep one (0.28
    against 0.26), for knots of 2.1% against 1.4 and 1.2% against 1.1.
    Finding the compact peaks took 2 to 17 s a 1024 px window (a square
    maximum), and that run 5.3 hours against 1.5; taken a row and a
    column at a time it is the same mask in under half a second.
  - A blur of 3 is there for the stars, which are asked for under half
    their width; it also asks what is not a star for a third of its
    width. `--rest-floor RATIO` keeps the rest's ask at RATIO or above
    in known-kernel windows whatever their stars are asked.
- **Detail inside a cloud.** By eye the reference's strong setting
  draws more structure inside a dark cloud than any run does. The
  measure: on a sheet's crops, further than 24 px from the steepest
  edges and 12 px from stars, the image's spread in a band (a Gaussian
  of 4 px less one of 8, and one of 8 less one of 16) over the input's.

  | | Trunk, stars 11.5 px | multi-night Trunk, 7.9 px | Trunk denoised first |
  |---|---|---|---|
  | The reference, strong | 1.13, 1.11 | 1.22, 1.11 | 1.48, 1.14 |
  | Small run, blurred up to 2 | 0.81, 0.87 | | |
  | The same on tiles of 768 | 0.96, 0.99 | 0.97, 1.00 | 1.05, 1.03 |
  | Small run, blurred up to 3 | 0.99, 0.99 | 1.06, 1.02 | 1.14, 1.07 |
  | The same with a contrast of 2 | 0.99, 0.98 | 1.07, 1.03 | 1.10, 1.06 |
  | The same with a clarity of 1 | 0.97, 0.98 | 1.19, 1.07 | 1.21, 1.10 |
  | Six levels 32 wide, blurred up to 2 | 0.92, 0.96 | 0.91, 0.98 | 1.06, 1.05 |
  | Six levels 32 wide, blurred up to 3 | 0.96, 0.98 | 1.00, 1.01 | 1.20, 1.07 |

  - It goes with the PSF's width. On the California (stars 2.3 px) the
    runs and the reference are level in every band from 1 to 32 px (1.02
    to 1.07), so what the reference adds on the Trunk lies between about
    a third of the stars' width and one and a half of them, and no
    contrast of a fixed size is it.
  - The widest blur moves it: 0.81 to 0.99 between 4 and 8 px, blurred
    up to 3 in place of 2. Tiles of 768 in place of 256 move it as far
    (0.96) with the blur left at 2.
  - `--contrast` does not: it is masked by what is there and to nebula
    standing over the noise, and inside a dark cloud neither holds.
  - **Local contrast that keeps strong edges gives it.** Applied to the
    small run's Trunk output: what stands, in the output smoothed by a
    third of the stars' width, above a guided filter of it over squares
    2.8 widths across, 0.3 of it. The base follows the image wherever a
    square's variance stands over the pixel noise squared, so a rim or
    a star fills the squares it lies in and the detail beside it is read
    from the squares that miss it. Inside the cloud the two bands then
    read 1.13 and 1.12 (the multi-night Trunk 1.21 and 1.09), and 1.05
    from 16 to 32 px against the reference's 1.04. The rims hardly move:
    the dark side 8 px out 0.32 from 0.26, the lit side 12 px in 0.39
    from 0.15 (the reference 0.37; a plain unsharp mask of a like reach,
    masked to nebula, puts it at 0.9 and the cloud's detail at 1.04),
    knots 1.14% from 1.11%. The California is unchanged.
  - `--clarity AMOUNT` asks for it in the target: a known-kernel
    window's target is given AMOUNT times one less the rest's ratio of
    it, 0.6 of the detail at an amount of 1, with `--contrast`'s limits
    on the stars' width and its clearing of listed point sources. It is
    added after the noise the target passes. Read from the truth, as
    `--contrast` is, it would ask a run to know from a noisy, blurred
    window what faint structure lay under it, which is where the runs
    hold back; read from the target with its noise, what is to be raised
    is in the input's own band, whatever it is. A window with next to no
    noise of its own (a space-telescope one) is judged by a twentieth of
    its range of light in the noise's place, and a denoised one by its
    stack's noise.
  - A run taught a clarity of 1 learns it where stars are 7.9 px across:
    1.19 and 1.07 on the multi-night Trunk, against the step's 1.21 and
    1.09. Where they are 11.5 px it does not at an ask of 0.5 (0.97 and
    0.98; denoised first 1.21 and 1.10, against the step's 1.33 and
    1.20), and does as more is asked: 1.04 and 1.01 at 0.4, 1.12 and
    1.03 at 0.33, where the run without reads 1.01, 0.99 and 1.03, 0.99.
    Its rims and knots are the run's without it: knots 1.1%, and 1.6%
    against 1.4 on the multi-night Trunk; at an ask of 0.4, ×2.98 and a
    rise over 3.95 px with knots of 1.3%, and the dark side 8 px out at
    0.41 against 0.3.
  - Wide inputs are seldom asked for much. A window is asked for no more
    than its blur took, and an input 9 to 12 px across is mostly wide
    truth blurred a little: those are 6% of the known-kernel windows
    drawn, three in ten of them asked for 0.58 or less, and their lift
    is 0.017 of a pixel's noise where windows 7 to 9 px across are given
    0.057. Blurred up to 4, the bank's median truth (2.9 px) is 11.8 px
    across.
- **Grain on images denoised first.** On noisy images the runs' grain is
  the reference's, in sky, nebula, bright nebula and along rims. On
  images denoised first it is not: along the California's rims the
  reference leaves 0.58 of the input's pixel-to-pixel spread and the
  larger run 0.85; in the Iris's bright nebula 0.41 and 0.61, along its
  rims 0.35 and 0.59. A native known-kernel window holds its stack's
  noise blurred in its input and sharp in its target, which asks a run
  to sharpen every soft blob of noise into grain.
  - `--clean-truth SHARE` takes that share of a bank's known-kernel
    windows from its denoised plane and meets them as noisy windows are
    met, under fresh noise of half to all of the stack's own, passed to
    the target: nothing in the target is noise made sharp.
- **Denoised sheets hold the reference too**: run on the same denoised
  input, and listed as a file (`"file"`), since an input denoised by a
  run is no product.

### What is next for sharpening

In order, each only if the one before leaves a gap at the rims:

1. **Detail inside a cloud on the run that is clean at the rims.** The
   small recipe blurred up to 3 (four levels 64 wide, tiles of 256 in
   eights, 60,000 steps, the banks of 1024 px windows) reads clean and
   reaches the reference's lit side at an ask of 0.4. On it, one change
   each:
   - blurred up to 4, since 3 in place of 2 took the detail between 4
     and 8 px from 0.81 to 0.99;
   - a clarity of 1 blurred up to 4, since blurred up to 3 it is learned
     for stars 7.9 px across and, at an ask of 0.5, not for 11.5.

   Six levels 32 wide, blurred up to 3 as well, were tried for the dark
   side and gave none of it.

   Then clean truth for 0.7 of the native windows, and the bank of
   brought-down stacks, which took a third of the narrow stars' flux
   gain off.
2. **Noisy space-telescope windows** (`--external-snr`) at more weight,
   blurred up to four times, on the run above.
3. **A classical deconvolution as an input plane.** A few
   Richardson–Lucy iterations with the measured PSF, read by the network
   beside the image: the network then cleans a sharp, ringing estimate
   in place of inventing the sharpening. Computed from the image and
   its PSF alone, so the model still needs nothing but the image.
4. **A feature loss and a critic, held by reblur consistency.**
   - Features from a network trained on photographs do not fit linear,
     single-channel data that is mostly sky. The feature network is the
     encoder of a trained noise model, which already reads this data in
     units of its noise; features are compared with stars masked out,
     as the edge term is, so the loss cannot pinch a star.
   - The critic is a few layers over 64 px patches, told the PSF and the
     ratios asked, scoring a patch as truth (the target, the
     space-telescope windows) or output. It is trained beside the
     sharpener, from a converged run and at a small weight.
   - Reblur consistency is the leash: the output blurred by the input's
     PSF must give the input back, weighted so that sharpness cannot be
     bought with detail the data does not hold.
   - The passed-through noise must be neither removed nor sharpened for
     either loss's sake: both read a lightly smoothed residual, or the
     denoised windows alone.
   - Invented sharpness shows on the known-truth probe as a share of
     the target's edge gain above 1.
5. **An unrolled deconvolution**: the network as several deconvolution
   steps with a learned correction between them and the PSF used at
   each. Closer to the physics and likely steadier across PSFs; a
   larger rewrite, and a follow-on to the above.

Beside these, for images whose stars are a pixel or two wide: **a
network that works finer than the image** (`--oversample 2`). Each pixel
of the planes is repeated over a 2 by 2 square going in, the U-Net runs
on that grid, and its last convolution takes each square back to one
pixel, so what goes in and comes out is the image's size and the
exported graph, the tiling and the runtime are as they were. The first
level can then place what falls between pixels, where the runs' narrow
stars come out less round than the reference's (the California: 0.07
against 0.046) and gain flux. A tile costs four times as much; its reach
and grid are halved in the image's pixels, so a six-level model tiles at
512 with a margin of 125. First run: the larger run's recipe on tiles of
384, which costs what a tile of 768 does now.

Also open: applying a mild model twice with the PSF measured again
between; a stricter best-seeing cut for the seeing pairs' targets; and
the grain along lit rims.

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

The noise model reads its noise map beside the image, as `asinh(σ / σ̄)`.
Read in `σ̄` as it is, the map reaches 50σ̄ and more on a bright star, and
one model took a star's map for a cue to add tens of thousands of σ̄ around
it. A model given the reading it was not trained on fails as badly, so the
model card names it.

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
  the variance map. Add the same weighted error averaged over 4, 8 and 16
  pixel blocks, each times its side, so that a block's noise weighs as a
  pixel's does: the per-pixel loss barely feels noise left at those
  scales, and a model leaves it as the blotchy "worm" texture. A block
  average of the target is still unbiased. Average the error unclipped and
  make each block's term robust; leave out blocks holding a star core.
  Add flux conservation in background boxes and faint-signal boxes.
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

### What the first noise runs showed

The first noise runs trained at batch 16 with a one-cycle schedule, for
40k to 80k steps, from 10M to 41M parameters. They changed the plan above
in these ways:

- **Precision.** Training in bf16 cost star cores 7–24% and mid-brightness
  detail. Runs train in fp32; TF32 convolutions are fine.
- **Clipping and rewinding.** With the synthetic outliers, models of width
  48 and 64 failed without a limit on the gradient's norm: one diverged at
  step 10,750, the loss reaching 1.8 million, and one stopped learning at
  step 3,000. Clipping the norm at 1 trained both. A run records the norm
  before the clip. A clip bounds each step, not where the weights go over
  many: the 41M recipe that reads the map through asinh, run again with
  only its seed changed, blew up at step 2,500, near the top of the
  schedule's warm-up, and never came back. Its activations grew to 10⁷
  through the levels, every channel of the full-resolution block before
  the output stayed below zero, and its output was a constant: for the
  other 77,500 steps the loss sat at the input's own, the gradient at
  0.03. At its first seed the same recipe blew up at step 4,000 and
  recovered. Without normalisation, which would make tiles disagree, a
  network that blows up can die, so runs keep their weights and the
  optimizer's state at each log window of 250 steps, and go back two
  windows when one's mean loss is more than twice the lowest so far, or
  not finite (`--rewind 2`). Training then goes on with the next steps'
  batches. The ordinary swings of these runs stay under 1.9 times.
- **Targets over size.** Three and a half times the training pairs, from
  136 sets instead of 45, gained a 10M model 2–3%. A 23M model trained on
  them kept fine detail at 0.964, against 0.925 for the same model on the
  45 sets. Pairs from 241 sets instead of 136, the added 105 of 20 to 39
  frames, gained the 23M model 5% on 4-frame halves (3% against matched
  halves that agree) and 8% on their faint signal, and its faint stars
  came out a little narrower at every depth. It lost 4–7% on their
  mid-brightness signal. On those pairs the 41M model beat it at every
  depth: by 3% on 4-frame halves, 14% on their faint signal and 5% on
  their star cores, matched, and by 2% on star cores at a night's whole
  depth, with the least mottle left on empty sky of any run. More
  targets made room for the larger model.
- **Steps.** Trained with the synthetic outliers for 40k steps, the 23M
  model left 9% of the coarse mottle on empty sky; for 80k, 3%. The mottle
  was undertraining, not a lack of capacity. At 80k, 23M matched 41M on
  4-frame halves; 41M was better at a night's whole depth. On the pairs
  of 241 sets, 160k steps instead of 80k gained the 23M model 4% on
  4-frame halves, and it kept faint peaks and stars' widths best of any
  run, but its star cores at a night's whole depth and against the clean
  targets were 6% worse. The 41M model at 80k, in less time, was better
  at every depth but 4 frames overall. The 41M model gained from 160k
  steps the same way: against matched halves that agree, 10% on 4-frame
  faint signal and 9% on their mid-brightness signal, the least error on
  4-frame halves of any run and the least mottle left on empty sky, but
  its star cores were 3–8% worse at every depth and 6% against the clean
  targets, and on one pair it blew up (see the noise map, below). At
  either size, more steps trade star cores for faint signal.
- **Noise map.** The network read the noise map in σ̄ as it is, which
  reaches 50σ̄ and more on a bright star, beside the image through asinh.
  The 41M model trained 160k steps added up to 71,000σ̄ in a square some
  190 pixels wide around one star of one 4-frame half, whose map peaked
  at 55σ̄. The same star in another half of that night, its map at 50σ̄,
  it left alone. Its activations ran from 44 to 27,000 at the fourth level
  down. With the map flattened it did nothing, with the map 10% higher it
  did more, and clipping the image changed nothing. No other run blew up on
  any pair, and none did on the tiles whose maps reach highest, even with
  their maps tripled, so the cue is a shape, not a height. Read through
  asinh, as the image is, a bright star's map comes under 5. The 41M
  model trained so for 80k steps stayed at 16–18σ̄ on those tiles' sky
  with their maps raised up to threefold, where the same recipe reading
  the map as it is rose from 17 to 34σ̄, and on the pair that blew up its
  error was 3.6, as the other's. Against matched halves that agree it
  was 5% better on 4-frame halves, the least error there of any run, 25%
  on their faint signal and 30% on their mid-brightness signal, 25% on
  8-frame faint signal, and 1–2% on star cores at 4 and 8 frames. At a
  night's whole depth it was even, its star cores 2% worse. The worst
  0.1% of 4-frame sky, what a short stack cannot reject, carried 13% of
  its sky error against 23%. It gave up two things: it left 7% of the
  coarse mottle on empty sky instead of 3%, alike on every pair, and kept
  1% less of faint stars' flux, their widths about 0.1% wider. A second
  seed, to tell whether those come with the map's reading, died in
  training (see clipping and rewinding). Given the map as
  it is, the same model put out 1,500–4,900σ̄ on sky, so the model card
  names the reading.
- **Mottle.** The whole images showed mottle from a few to some tens of
  pixels across. On real pairs, from f(A) − f(B) against A − B on
  star-free sky, where signal and gradients cancel, the 41M models took
  out 86–89% of a 4-frame half's noise at the pixel, but left 44–54% of it
  smoothed at 4 pixels, 59–72% at 8 and 80–88% at 16; the run reading the
  map through asinh left the most. Over the 4-frame halves of all twelve
  validation sets the two left 47% and 58% at 4 pixels, 65% and 79% at 8,
  and 89% and 96% at 16. In grain the output was as clean as a
  200-frame stack, in mottle as an 8-frame one, and that mismatch reads as
  blotches. On the flat sky of noise alone they left 3–7%: with nothing
  in the frame like signal, a model takes out coarse noise it leaves on a
  real sky, where faint signal looks like it. A pixel's noise lies mostly
  at the finest scale, so the per-pixel loss barely feels noise left over
  8 or 16 pixels. The next runs weigh the error averaged over blocks of
  those sides as well (see the losses), and every run records the noise
  it leaves at 4, 8 and 16 pixels on the validation pairs. The first such
  run, the error clipped at the robust loss's 3σ̄ before the average, left
  43%, 61% and 84% instead of 58%, 79% and 96%, but striped star cores and
  bright nebulae column by column, its change from one column to the next
  up to 15 times that from one row to the next, and dug rings around
  stars. Against clean targets its error on mid-brightness signal was 18
  times as large, on every pair, while its sky was as good. The clip let
  it meet a block's mean with errors of alternating sign, each past the
  clip, where the robust per-pixel loss charged little for them, and the
  sky, where errors stay within the clip, showed nothing. Runs since
  average the error unclipped, make each block's term robust instead, and
  leave out blocks holding a target brighter than 200σ̄, where the
  per-pixel loss is robust too. The first, weighing the blocks four times,
  left 43%, 62% and 86%, close to the clipped run, with no stripes: in
  bright star cores its change from one column to the next is within 12%
  of that from one row to the next, and zoomed in it looks like noise. No
  ring shows around the Iris's stars; around the Elephant's Trunk's
  brighter ones it took 1–2σ̄ off 6–12 pixels out, where the deep stack
  itself lies 15–19σ̄ below the 4-frame half. It kept more of the faintest
  stars' flux (83% of it at 1–2σ̄ on the Trunk, where the run before kept
  64%). But the weight took share from the per-pixel loss. It changed
  bright star cores by 2σ̄ where the run before changed them by 11σ̄ and
  the deep differs by 10σ̄, leaving their noise in; against clean targets
  its error was 20% larger on mid-brightness signal and 6% on cores; on
  the flat sky of noise alone it left 3.4% of the finest noise instead of
  1.4%; and of the sharp spots a 4-frame half has and the deep lacks, hot
  pixels and cosmic rays its rejection let through, it left 76 over 1σ̄ on
  the Trunk where the run before left 18, spots of 6–7σ̄ coming out as
  dots of 2–3σ̄. Both runs turn the strongest such spots into a small ring
  rather than taking them out. Weighed once, the next run left less still,
  41%, 60% and 85%, and with the per-pixel loss's share back it had the
  lowest error on the 4-frame validation halves of any run (2.01 against
  2.05); no stripes (cores within 5%); no ring around the brightest stars,
  where the run before the scale loss had one of +7σ̄ eight pixels out;
  fewer of the sharp spots left than any run on all four images (9% of
  them on the Trunk, though 25 over 1σ̄ to 19); 85% of the faintest stars'
  flux on the Trunk; and in the Iris's faint dust 4% more contrast between
  neighbouring pixels. Its costs are small: 5% more error on 8-frame
  halves, against clean targets 7% more on mid-brightness signal and 5% on
  cores, and on the flat sky 2.3% of the finest noise left instead of
  1.4%. The rewind guard brought it back once, at step 10,750. A weight of
  one is the recipe from here; the next runs train it twice as long, and
  at a second seed for the spread between seeds. Twice as long, the run
  blew up seven times in its first 60,500 steps, from a sound window to a
  mean loss of 300 to 3.5 million in the next. The guard brought it back
  six times and then stopped it, since it blew up again 250 steps after
  its last rewind. The schedule holds the rate higher for longer in a long
  run. Each rewind now takes a fifth off the learning rate for the rest of
  the run, and the guard stops a run only after six rewinds within 20,000
  steps. At a second seed the recipe left 43%, 62% and 88% of the noise at
  4, 8 and 16 pixels, against the first seed's 41%, 60% and 85%: the
  seeds differ by about 2 points, the scale loss took off 11–18. The
  other gains held as well: no stripes, 5% of the sharp spots left on the
  Trunk (14 over 1σ̄), 84% of the faintest stars' flux there, and the
  Iris's dust within 1% of the first seed's sharpness. Its error on the
  4-frame halves, though, was 2.23 against 2.01, from the sky's worst
  pixels and faint signal, its trimmed sky the same; so the first seed's
  lead over the run before the scale loss (2.05) is within the spread, as
  are differences of 5% against clean targets on faint and mid-brightness
  signal. Around the Trunk's 27 brightest stars it raised the light 2–4
  pixels out by 6–9σ̄, where the first seed raised it by 3–5σ̄. Against
  B halves matched to A's flux, on the pairs whose halves agree, the run
  before the scale loss is better than either seed away from the sky: on
  4-frame halves its error on faint signal was 11.1 against 15.9 and 21.5,
  on mid-brightness signal 29.5 against 41.0 and 50.0, on cores 2,937
  against about 3,280, with the trimmed sky the same. The scale loss takes
  the coarse noise out at a cost pixel by pixel in signal, and the cost
  differs by a third between seeds. Twice as long with the guard that
  lowers the rate, the recipe rewound ten times, the last at step 82,250,
  and ended at 11% of the schedule's rate. It came out where the two seeds
  did: noise left 42%, 60% and 84%, error 2.01 on 4-frame halves, and
  1.19 on matched agreeing ones against the seeds' 1.18 and 1.27. On the
  perturbed images (below), though, it blew up where the 80,000-step runs
  did not: on the Trunk binned and under a gradient it put 54,000 to
  560,000 pixels more than 100σ̄ from the input, up to 855,000σ̄, and on
  the Iris binned up to 11,000σ̄, nine times its worst unbinned. Bounding
  each output pixel by its 5 × 5 neighbourhood of the input, widened by
  three times its noise, the guard the runtime is to have, brought its
  worst pixel back to the unchanged image's, about 2,400σ̄ in a bright
  star's core; it moved 2.6% to 8% of the pixels there, against under a
  thousandth of a percent on the unchanged image, which makes the
  share it moves a measure of a model's failing. So the longer run gains
  nothing, and every candidate is run on the perturbed images. The next
  runs add the 101 train sets built without a flat: 1,700 pairs over 342
  sets, against 1,301 over 241, at the same width and at half as wide
  again. At the same width and the first seed, with them the recipe came
  out between the two seeds on every measure:
  - noise left 42%, 61% and 85%;
  - error 2.14 on 4-frame halves and 1.29 on 8-frame ones, against the
    seeds' 2.01–2.23 and 1.22–1.31;
  - against B halves matched to A's flux, 2.83 against 2.71 and 2.90;
  - star cores against clean targets within 0.2%.

  It never rewound. It left more of the Trunk's sharp spots: 13% of them,
  55 over 1σ̄, against the seeds' 5–9% and 14–25. Nothing blew up on the
  perturbed images. Under the gradient the guard moved 0.2% of the Trunk's
  pixels, as it did for the first seed, and 0.001% with the sky taken out
  first. So at this width the sets without a flat neither help nor hurt
  the noise model.

  With the sky taken out before the network and the pairs changed alike
  (below), the recipe on the same sets became far harder to break.
  - **The perturbed images.**
    - On the Iris binned and sharpened it changed no pixel by more than
      100σ̄, where the recipe without them changed 29,509 and 67,012. Its
      largest change was 76σ̄ and 65σ̄, against 862σ̄ and 999σ̄. On the SNR
      binned it changed 4 pixels, against 6,728.
    - It moved the coarse level by under 0.012σ̄ on every image except the
      Trunk binned (0.12σ̄). Without the changes the coarse level moved by
      up to 0.64σ̄.
    - Faint stars kept their flux under the changes: 1.02 on the Iris
      binned against 0.86.
  - **Bright stars.**
    - The ring around the Trunk's bright stars nearly went: their cores
      dipped 15σ̄ where the recipe's dipped 63σ̄.
    - Stripes in star cores halved.
  - **It removed less noise.**
    - At 16 pixels it left 93% on the Trunk against 83%, and 3–4 points
      more on two of the other three images.
    - It kept about twice the sharp spots.
    - Its error on 4-frame halves was 2.43, outside the seeds' 2.01–2.23.
  - **Rewinds.** One, at step 8,500.

  Taking the sky out alone, the recipe left 42%, 62% and 88% of the
  noise, with an error of 2.23 on 4-frame halves. So the sky taken out
  costs about 3 points of the coarse noise, and the changed pairs cost the
  fine noise.

  At half as wide again (93 million weights), the recipe without either
  change was the best yet on the unchanged images:
  - error 1.97 on 4-frame halves and 1.22 on 8-frame ones;
  - noise left 42%, 60% and 84%;
  - under 1% of the Trunk's sharp spots, 10 over 1σ̄.

  It was as easily broken as the narrower one, though:
  - the Iris binned and sharpened still had 15,800 and 46,000 pixels more
    than 100σ̄ from the input;
  - under the gradient it left 125% of California's coarse noise;
  - it raised the light around the Trunk's three brightest stars by 5–7σ̄
    out to 8 pixels.

  Against B halves matched to A's flux, the wider model's error was 2.63:
  - against 2.83 for the narrower model on the same sets, and 2.71 for the
    best seed;
  - in faint regions 9.4, against 12.2.

  So width pays when the pairs are left as they are.

  With the sky taken out, the narrower model was trained again with only
  the sky gradient and the binned share. That was nearly as hard to break
  as all four changes:
  - **The perturbed images.**
    - On the Iris binned and sharpened it changed 10 and 8 pixels by more
      than 100σ̄, with a largest change of 300σ̄ and 161σ̄. With the sky
      taken out alone it changed 15,357 and 27,926.
    - On the SNR and California sharpened it changed 20 and 18 pixels,
      against 16,830 and 4,794 without either change.
  - **What it cost.**
    - Its error on 4-frame halves was 2.32: 2.23 with the sky alone, 2.43
      with all four changes.
    - Against matched halves its error was 2.98 against 2.90 and 3.11. In
      faint regions it was 16.8 against 14.7 and 20.4.
    - It left 46% of the finest noise, as all four changes did, against
      42% with the sky alone. So the gradient and binned share, not the
      shifts and sharpening, cost the fine noise.
    - It kept 20% of the Trunk's sharp spots, 118 over 1σ̄, against 12%
      and 48.

  All four changes on the wider model bought nothing from the width:
  - its error was 2.41 on 4-frame halves against 2.43, and 3.07 against
    matched halves against 3.11;
  - in faint regions its error was 21.8 against 20.4;
  - it kept 23% of the Trunk's sharp spots.

  It left 49%, 64% and 78% of the noise, the least coarse noise of any run
  and the most fine. It stayed hard to break: 1 and 0 pixels over 100σ̄ on
  the Iris binned and sharpened. Under the gradient, though, the guard moved
  0.17% of the Trunk's pixels, against 0.015% for the narrower model with
  the gradient and binned share alone.

  The next runs are the first on the corpus restacked with the stacker's
  fixed gains and tile offsets. Both are at the wider width:
  - the recipe without the changes, as the new corpus's baseline;
  - the sky taken out with the gradient and half the binned share, a tenth
    of each batch.

  The baseline came out level with the same recipe trained on the old
  stacks, both measured on the rebuilt validation sets:
  - against matched halves, 1.650 against 1.709 on 4-frame halves and
    1.173 against 1.184 on 8-frame ones, inside the spread between seeds;
    against clean targets, 0.756 against 0.757;
  - on the rebuilt sheets, it left 37% of the Trunk's noise at 4 pixels
    against 45%, and about the same on the other three sheets and at 16
    pixels; faint stars kept the same flux;
  - it kept a little more of the sharp spots, 28% of the Iris's against
    16%; it was less alike turned (0.26σ̄ RMS on the Trunk against 0.19σ̄);
    and its worst pixel before the guard was a star core 4,400σ̄ out on the
    Trunk, against 2,800σ̄;
  - 2–4 pixels around the Trunk's stars it lifted the light by about
    1.5σ̄ for faint ones and 4–13σ̄ for the brightest, against 0.6σ̄ and
    1–4σ̄; on the Iris the two were alike.

  So the fixed stacks change little in what a model learns. What they
  change is the measure: the same model's error on matched 4-frame halves
  fell from 2.63 to 1.71 with the restack, as the halves now agree.

  The run with the sky taken out, the gradient and a tenth binned lost to
  the baseline on the halves and kept more of the image:
  - against matched halves, 1.944 against 1.650 on 4-frame halves and
    1.253 against 1.173 on 8-frame ones; against clean targets, 0.762
    against 0.756;
  - it left 46%, 65% and 89% of the noise at 4, 8 and 16 pixels on 4-frame
    halves, against 40%, 60% and 87%, and 64% at 4 pixels on 8-frame ones
    against 41%; on the Trunk, 63% at 4 pixels against 37%, and 35% of
    the sharp spots against none;
  - it threw fewer pixels: its worst before the guard was 1,300σ̄ on the
    Trunk, against 4,400σ̄, and on the Iris binned and sharpened 15 and 25
    pixels went over 100σ̄, against 17,900 and 40,200;
  - it dimmed star cores less and barely lifted the light around the
    brightest: +0.5–0.9σ̄ 2–4 pixels out on the Trunk, against +10–13σ̄
    for the baseline and +3–4σ̄ for its recipe on the old stacks.

  Like the narrower model with the gradient and binned share, it buys its
  robustness with fine noise and speckle left behind. It is also the run
  that blows up on spread noise (see the drizzled masters above): on the
  California nebula shifted half a pixel, 26,700 pixels went over 100σ̄
  before the guard, against 45 for the baseline.

  The two runs after these train blind, reading only the image's noise
  level, with dark spots planted in both halves and 15% of their tiles
  spread: the baseline's recipe, and the sky taken out with the gradient and
  a tenth binned. Both are stable on spread noise, their largest steps 31σ̄
  and 28σ̄ against 5,500σ̄ for the sky recipe trained without the spread.
  Both lose on the halves, and by about the same:
  - against matched halves, 2.135 and 2.114 on 4-frame halves against the
    baseline's 1.650, and 1.316 and 1.289 on 8-frame ones against 1.173;
    against clean targets, 0.814 and 0.787 against 0.756. The trimmed sky
    is the same for all; the loss is in what stands out of the sky and
    around faint signal;
  - both left 56% of the noise at 4 pixels on 4-frame halves, against 40%,
    and 53% on 8-frame ones, against 41%;
  - on the Trunk they kept 12% and 17% of the sharp spots, against none for
    the baseline and 35% for the sky recipe without the spread;
  - 2–4 pixels around the Trunk's bright stars they lifted the light by
    8–11σ̄, against the baseline's 4–6σ̄, and dimmed those stars' cores by
    92σ̄ and 84σ̄, against 56σ̄. Faint stars kept the same flux, and the
    faint texture matched the deep stack as well as before;
  - on the probe they threw fewer pixels than the baseline: the guard moved
    under a millionth of the Trunk, against 14 millionths. The California
    shifted half a pixel sent 213 and 110 pixels over 100σ̄, against 45 for
    the baseline and 26,700 for the sky recipe without the spread.

  On the drizzled masters nothing broke as before. On the other stacker's
  2× drizzles the Sunflower sent no pixel over 100σ̄, against 1.66 million,
  and the California 115 and 6, against 110,700. But the baseline's recipe
  trained so is not as safe as the baseline. It sent 2,500 of the Iris's
  pixels over 100σ̄, by at most 140σ̄, and 107–433 on each of Seiza's four
  2× drizzles, where the baseline sent none on the Iris and 0–4 on three of
  Seiza's (240 on the Elephant's Trunk, by up to 3,300σ̄, against the
  blind run's 665σ̄). The sky recipe trained so sent 223 on the Iris, 182 on
  the Trunk and 0–6 on the other three. On the drizzles both left less noise
  than the baseline: 4% at 1 pixel against 5–7%, and 7–10% at 4 pixels
  against 21–28%, though on the halves they leave more at 4 pixels.

  So the two lose to the baseline for what they share, not for their sky.
  They share three changes: blind training, the dark spots and the spread.
  Given only the constant map in place of the stack's, the same recipe
  trained on the old stacks lost 3%, so blind training should cost little.
  The next run is the baseline's recipe, blind and with the dark spots,
  without the spread.

  That run is no better, so the spread is not the cost. On matched halves
  it reached 2.156 on 4-frame halves and 1.410 on 8-frame ones, against
  2.135 and 1.316 with the spread, and it too left 56% of the noise at 4
  pixels. Its trimmed sky matches the baseline's; the sky's outliers carry
  the loss, 30% of the sky's error against 23%. Nor is the level it is told
  too low: on the Trunk the constant map reads 5.06, against 4.55 for the
  stack's modelled map, yet on the probe it left 87% of the Trunk's noise
  at 4 pixels, against the baseline's 37%, while on the Iris and the
  California it matched the baseline.

  Without the spread the run is also unsafe on real stacks, though its
  largest step on spread noise is 29σ̄. On Seiza's plain master of the
  Elephant's Trunk it sent 79,823 pixels over 100σ̄, the worst to 62
  million σ̄, all in one patch of ordinary sky about 150 pixels across
  beside a bright star, where the input stands 1–2σ̄ above the background;
  its 2× drizzle failed in the same place, with 86,486. It sent 11,407 on
  one of the other stacker's masters, where the run with the spread sent
  12, and 26,300 on the sharpened SNR of the probe. In training its loss
  jumped once to 108,000, at step 50,750, and was rewound. So the spread is
  what kept the blind runs with dark spots safe, and a run that passes the
  spread-noise probe can still fail on a real stack: the Trunk's master is
  now checked for every run.

  A model may read a map made from the image alone, so blind training is
  not required. To find which change costs, the next two runs each make one
  change to the baseline, without the spread: the dark spots alone, with
  the stack's modelled map, and blind training alone.

  The dark spots are most of the cost. On matched halves the run with them
  reached 2.021 on 4-frame halves, 22% behind the baseline's 1.650, and
  left 48% of the noise at 4 pixels against 40%; the blind run reached
  1.746, 6% behind, and 1.310 on 8-frame halves, 12% behind. Neither blew
  up on a real stack: the Trunk's plain master sent 33 and 61 pixels over
  100σ̄, the other stacker's master that took 11,407 sent 0 and 5. So the
  blind run with dark spots ran away from the two together. The blind run
  leaves more of a drizzle's fine noise, 43–56% at 4 pixels against 19–26%
  with the stack's map, and sends a few hundred pixels over 100σ̄ on two
  drizzles. The run with dark spots still filled the Trunk's holes and
  overshot them by about 200σ̄.

  So a model no longer learns holes: the runtime keeps them from it. A
  card's `reading.holes` asks for the guard. Pixels more than 30 of the
  image map's noise below the sky's floor, widened by the 3 pixels
  registration spreads a pixel, are filled with the median of the pixels
  round them before the network runs. After it, the input is put back
  wherever it differs from the output by more than 5 of the map's noise,
  so a hole and its lobes come back as they were, and the margin's noise
  is taken out. The floor is the lowest of each block's 20th percentile
  and its neighbours', never above the image's background, so a large
  star, a nebula's edge or a dark lane holds no hole. With the baseline on
  the Trunk's night it took the blow-ups from 74 to 30 pixels plain and
  from 240 to 10 drizzled, and the largest change inside a hole from
  6,858σ̄ to 15σ̄; the noise left on the sky did not change. What remains
  are dark moats beside leftover hot pixels, a problem of their own.
  Across 368 stacks it found no hole in 15 of the other stacker's 16
  masters, and no masked spot anywhere larger than 20 pixels. Two 4-frame
  halves hold hundreds of 1–3 pixel holes found in one half of their pair
  only, which rejection over four frames left behind; the guard keeps
  those too, since depth does not tell them from a hole every frame holds.

  The runtime also raises the image map for spread noise now, as training
  does, where a card names the correlation its model trained at.

  Training follows the guard (`--holes`): each pair's input has its holes
  filled as the runtime fills them, and its target where either half
  holds one, so the network never sees a hole and is never asked for one,
  and validation runs the guard. The next runs drop the dark spots: the
  baseline read blind but weighed by the stack's modelled map, then the
  same under the guard.

  Weighing the loss by the stack's modelled map wins back most of what
  blind training cost. The network reads only the image, as the blind run
  did, but each pixel's error, the outliers' size and the halves' match
  follow the stack's own map, so nebula pixels are no longer weighed as
  sky. It reached 1.704 on 4-frame halves, 3% behind the baseline where
  the blind run was 6% behind, and 1.181 on 8-frame halves, level with the
  baseline's 1.173 where the blind run was 12% behind; the deep halves
  kept 0.656 of their error against 0.641. Its bright stars came out the
  cleanest of any run: on the Trunk, stars of 200 to 2,000σ̄ kept +1.2σ̄ 2
  pixels from the core against the baseline's +6.4 and the blind run's
  +13.0, and the brightest stars stayed within 2.5σ̄ where the baseline
  left +13. It kept 7% of the Trunk's speckle, between the two, and its
  largest step on spread noise was 28σ̄.

  Measured under the guard, the same run reads 2.217 on 4-frame halves and
  1.320 on 8-frame halves. That is the guard's own cost on this measure,
  not the network's: the trimmed sky's error does not move on any pair,
  and the rise comes from three sets, two of them the Trunk's nights,
  whose short halves hold cold pixels deeper than 30σ̄ that the guard keeps
  and the other half lacks. A run trained under the guard is compared with
  this run measured under it.

  The runtime, given the card's guard and the trained correlation, gives
  the harness's output on the Trunk's night stacked plain by Seiza, 26 MP
  with 11,842 pixels guarded: 1.5e-6σ̄ RMS and 7.4e-4σ̄ at most on the CPU
  in f32.

  Trained under the guard, the same recipe came out level with that run
  measured under it: 2.208 against 2.217 on 4-frame halves and 1.317
  against 1.320 on 8-frame ones, and the deep halves kept 0.644 of their
  error against 0.641; against clean targets, 0.797 against 0.799, with
  those pairs guarded too. Both left 56% of the noise at 4 pixels, where the
  network trained without the guard, measured without it, left 42%: the
  cold pixels the guard keeps count as noise left too. What training
  under the guard bought shows on stacks that hold holes or spread noise:
  - on the Trunk's night, plain and drizzled, it sent 37 and 44 pixels
    over 100σ̄, against 63 and 288, of them 0 and 1 above the input's
    range, against 30 and 241; its largest steps were 354σ̄ and 707σ̄, against
    1,122σ̄ and 3,125σ̄;
  - on Seiza's other three drizzles it sent 1 pixel over 100σ̄, against
    14, and left 6–8% of the pixel-scale noise, against 8–11%;
  - on the other stacker's 16 masters the two were alike, leaving 2–8% of
    the pixel-scale noise and sending at most 4 pixels over 100σ̄ on any
    one;
  - before the guard put anything back, its worst pixel on the Trunk's
    4-frame image was 1,387σ̄, against 3,418σ̄. Its largest step on
    spread noise was 27σ̄.

  It gave back part of the weighted run's clean bright stars: on the
  Trunk, stars of 200 to 2,000σ̄ kept +3.7σ̄ 2 pixels from the core,
  against +1.2σ̄ and the baseline's +6.4σ̄, and the two brightest about
  +12σ̄ out to 4 pixels, as the baseline did, against 2.5σ̄ at most. On
  the Iris's deep stack the two were alike.

  Its bundle gives the harness's output in the runtime on the Trunk's
  night under the guard: 1.6e-6σ̄ RMS plain and 1.3e-6σ̄ on the 2×
  drizzle, which holds 19,517 guarded pixels, and 9.2e-4σ̄ at most.

  What it still sends over 100σ̄ on the Trunk's night are the dark moats:
  pixels at the sky's level between leftover hot pixels of about
  2,000σ̄, which it puts 140–710σ̄ below the sky. Both runs dig them
  alike, and the guard cannot see them, since they are not below the
  sky's floor. A model's output almost never falls far below the lowest
  input pixel within 2 pixels: on the probe's four images, this run's
  fell more than 20 times the image map's noise below it on 0–3 pixels
  an image, and more than 5 times on 0–9. A floor there would end the
  moats at almost no cost elsewhere.

  Two results are worse with the stack's map in the loss, and not yet
  explained:
  - **Under the probe's sky gradient**, California's coarse level moved by
    0.92σ̄ and 0.47σ̄ with these two runs, against 0.02σ̄ for the baseline
    and 0.08σ̄ for the blind run. With the sky flattened before the
    network, both moved it by 0.009σ̄. On the unchanged images that leaves
    up to 8% more of the noise at 16 pixels.
  - **On the Trunk's 4-frame image**, the run trained under the guard left
    82% of the noise at 4 pixels, against 46%, though the image holds no
    hole and the other three images match. Its sky, the dark lanes
    between the nebula, is a small share of the frame, and on the sheets
    the two runs look alike.
- **Deep halves.** Training on a night's deep stack dealt into two halves,
  a quarter of every batch, taught a model to keep mottle: it left 37% of
  the coarse noise on empty sky. The halves share the night's flats,
  masters and gradients, a pattern that is large next to the noise of 20
  or more frames. The halves are measured only, as are the halves of an
  object stacked over many nights, whose seeing differs between them.
- **Stars.** Against the input, every run kept stars' widths within the 1%
  gate at every depth. Against the deep reference, faint stars from 4-frame
  halves came out 0.6–1.3% wider.
- **Perturbations.** To tell a model that has learnt to denoise from one
  that has learnt its data, runs are also measured on the four full-size
  4-frame images changed in ways the corpus never was, the same change to
  both halves and the deep stack, with the noise map carried through: a
  quarter turn; a half-pixel shift by Lanczos-3, which correlates the
  noise between neighbours as a second registration would; 2 × 2 binning,
  which halves the stars' width in pixels; and a sky gradient of 40σ̄
  across the frame and 25σ̄ from the centre to the corners. Turned, every
  run gives the same measures, and its output turned back lies within
  0.07–0.27σ̄ of the unturned one's. The image from a train set changes
  under each no more than the held-out ones do. So the runs have not
  learnt their images; they have learnt their conditions. With the
  gradient, the run before the scale loss amplified the coarse noise,
  leaving 1.3 to 3.4 times what the halves differ by at 16 pixels, and one
  scale-loss seed did the same on three images (1.3–1.8 times), where the
  other stayed at 71–85%. The network reads the image less one background
  for the whole frame, through asinh, so a sky far above or below it reads
  as smooth signal with its noise compressed. Taking out a smooth sky first
  (medians over 256-pixel blocks, spread back) and putting it back after
  brought every run to 80–95% at 16 pixels and gave the faint stars their
  flux back; on the unchanged images it cost at most 1% of the coarse noise
  on three and up to 6% on the fourth, a nebula filling the frame. One seed
  did worse with it on one image, which is open. Binning made stars about
  two pixels wide, narrower than any in the corpus, and cost the
  scale-loss runs on the held-out images: on the Iris one moved the
  image's coarse level by 0.64σ̄ (0.01 unbinned), kept 86% of the faintest
  stars' flux where the run before the scale loss kept all of it, and
  rendered faint texture less like the deep stack than the binned input
  itself did. The extra resampling made two runs leave more 4-pixel noise
  on the Trunk than its halves differ by: it spreads the hot pixels into
  soft spots. The remedy to try is training on pairs changed the same
  ways, which a noise pair allows, since one linear change or added sky
  applied to both halves keeps their noise independent, together with the
  smooth sky taken out before the network reads the image. Training now
  offers both. The network can read the image less its smooth sky in
  training, evaluation and the sheets alike, the sky carried on along its
  slope past the outer blocks and through their median: the probe's held
  its edge blocks level and left 3σ of a 50σ gradient in the corners.
  Tiles can be shifted by a random sub-pixel offset or sharpened by a
  random unsharp mask, both halves alike and the noise map through the
  filter's squared weights; inputs can gain a sky plane, which the loss,
  taking the input back out of its estimate, does not see, so its weights
  stay on the target's own brightness; and a share of every batch can
  come from the pairs binned 2 × 2. The first run with them shifts a
  quarter of the tiles and sharpens a quarter, puts a plane of up to
  20σ̄ under half, and draws a fifth of each batch binned, on the 342
  train sets; a second takes the sky out alone.
- **Whole nights.** On the held-out objects' deep stacks, measured against
  their multi-night stacks, every run kept the faint stars' flux as the
  input had it, within 3%, and the faint texture within 0.01 of the
  input's correlation with the reference. On California's, the scale-loss
  runs kept 90% of the faintest stars' flux where the input kept 101% and
  the run before them 96%, though on the 4-frame half they kept more than
  it. In California's deep and multi-night stacks every run lowers the
  brightest stars' peaks, by 90 to 280σ̄ (the scale-loss runs by a third
  to a half as much as the run before them), and changes star cores more
  down the columns than across them (0.83–0.94), which all three share, so
  it is the data's. Elsewhere the deep and multi-night stacks show no
  stripes, and one seed lifts the light 2–8 pixels around the Trunk's
  brightest stars by 2–9σ̄, as it does on the 4-frame half. The Trunk's multi-night stack lies on
  another night's grid than its deep sheet, so that deep stack is checked
  against itself only.

## Inference in Seiza

The [runtime plan](ml-runtime.md) proposes changes to this section from
measurements of the current models: one ONNX Runtime binding that drives
Core ML, no separate Core ML export, and a parity budget of its own for the
Neural Engine. It also records that the output bound below has not been
implemented, and that no run has yet been measured with an estimated noise
map.

A new crate, `seiza-model`, runs the models and holds the three operations.
The classical operations stay where they belong: Richardson–Lucy in
`seiza-deconvolution`, and the PSF field and catalogue-guided star removal in
`seiza-stars` and `seiza-stacking`. `seiza-model` has its own
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
`ModelError::BackendUnavailable` with the reason, and the classical
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
  channels, how each is read, and their ranges;
- the trained ranges of FWHM, pixel scale and signal-to-noise ratio;
- the version, the training run, and a hash of each artefact;
- the data licences.

A bundle is verified and installed through `seiza-download`, as catalogues
are. A host can also ship it. Bundles are not embedded in crates. Core ML
compiles the package on first load, and Seiza caches the compiled model.

### Tiling and parity

Inference cuts the image into tiles of the export's static shape, runs
each, and blends them. Two rules make that reproduce whole-image inference.

The first is where tiles start. A network that halves the image four times
samples it on a 16-pixel grid, so what it makes of a pixel depends on where
the tile holding it starts. Moved a pixel, a tile of a trained noise model
gave output up to 0.8σ different; moved 16, it differed by float error.
Tiles therefore start only on multiples of 2^(levels − 1): the stride is cut
to such a multiple, and the image is padded by reflection at its far edges
until the last tile, flush with them, starts on one too.

The second is the margin. Within the network's reach of a tile's inner
edge, its output saw the convolutions' padding rather than the image, so
that band takes no weight; past it, the tile's weight rises as a raised
cosine across the rest of the overlap. The reach is half the receptive
field, but on the side nearest upsampling copies from, it is a pixel longer
at each scale finer than the coarsest: at five levels, 122 pixels left and
up and 107 right and down. At the image's own edge a tile keeps its full
weight, because whole-image inference pads there too.

With both rules, tiled output matches whole-image output to within float
error, and tests hold it there: a plain cosine over the whole overlap,
tiles off the grid, or a margin of half the receptive field each make them
fail. The first noise runs were measured with tiles every 264 pixels and a
margin of 108. On a 4-frame half, a second grid half a stride over gave
output 0.007σ different in root mean square, and up to 16σ beside bright
stars. With both rules, the two grids differ by 2e-6σ, and at most by
2e-7 of the residual, at a star core. The old tiling moved the measures
by less than 0.01%, so the comparisons between those runs stand.

Every noise run writes, besides its contact sheets, twelve images whole,
as FITS and PNG, each with a seam map: the difference between the outputs
of two tile grids half a stride apart, in units of the noise, in which a
seam shows as a line of the grid. The map's root mean square and largest
value are recorded with the run. The images are four objects' 4-frame
validation halves, the same nights' deep stacks, and the objects' stacks
over all their nights, so a run is seen on what a user would denoise, a
whole night or more, as well as on the short stacks it is scored on. A
deep stack is measured against the object's multi-night stack, which lies
on its grid; a multi-night stack, which has nothing deeper, only against
itself, for rings and stripes.

Each tile carries its own PSF stamp and its own slices of the noise,
ratio, halo and bloom maps. Peak memory is a few tiles' worth whatever the
image's size, and the backends compile one tile shape. The model card
records the tile shape, the overlap, the zero-weight margin and the grid
tiles start on.

The noise model's output is bounded by its input. Each pixel is held
between the lowest and highest input in the 5 × 5 box around it, widened by
three times the noise map there: smoothing and the removal of an outlier
both stay inside that range, and a network that runs away does not. Tried
on four 4-frame validation pairs with two runs, the bound moved 1 to 79
pixels of 26 million and left the error as it was, except on the one pair
where a run blew up: there it moved 34,000 and brought the error from
121,000 to 13. The share of pixels it moves is
reported, and a tile where it moves many falls back to its input.

The export script writes fixture tiles with PyTorch's f32 output, and a
whole image with its output through tiles. Each backend, and the runtime's
tiling, must match them:

- the ONNX Runtime CPU provider within 1e-4 relative;
- Core ML's GPU, CUDA and TensorRT within 0.01σ of the input's noise;
- the Neural Engine, at f16, within 0.02σ̄ RMS on the sky and 0.002σ̄ in the
  mean.

The Neural Engine is enabled only for a model that passes its own parity run.
Core ML parity runs on macOS CI. CUDA and TensorRT parity runs on the NVIDIA
machine before each model release. Results differ slightly between backends,
so hosts include the backend and precision in their cache keys.

### Native API

Following the [Rust-first rules](../../AGENTS.md#the-rust-api-comes-first):

```rust
let options = BackendOptions::default();
let backend = Backend::detect(&noise_bundle, &options)?;

let mut model = NoiseModel::open(&noise_bundle, backend, &options)?;
let result = model.denoise(&image, &DenoiseOptions::default())?;
// Per-band amounts of the removed noise: no second inference.
let image = result.blend(&NoiseBands { high: 0.9, low: 0.5, scale_px: 5.0 });

let psf = PsfField::measure(&image, &PsfFieldOptions::default())?;
let sharpened = PsfModel::open(&psf_bundle, backend, &options)?.sharpen(
    &image,
    &psf,
    &SharpenOptions {
        target: SharpenTarget::Ratio { stars: 0.5, extended: 0.75 },
        halo_reduction: 0.3,
        bloom_reduction: 0.0,
        ..Default::default()
    },
)?;
// sharpened.report holds the reblur χ² and the trained-range checks.

let StarLayers { starless, stars } = StarModel::open(&star_bundle, backend, &options)?
    .separate(&sharpened.image)?;
```

The types follow the same rules:

- `Backend` is an enum: Core ML with its compute units, or ONNX Runtime with
  its execution provider. It has `FromStr` and `Display` for the
  `--backend coreml|tensorrt|cuda|cpu` flag of `seiza model`.
- `SharpenTarget` is `Ratio { stars, extended }` or `Uniform`.
- `halo_reduction` and `bloom_reduction` are amounts from 0 to 1, applied
  inside the star mask.
- `NoiseBands` holds the high- and low-frequency amounts and the scale that
  divides them.

Refusals are variants of a `ModelError`, made where they happen:

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
- on real pairs' star-free sky, the noise left at 4, 8 and 16 px, from
  f(A) − f(B) against A − B smoothed at each, within a limit set from the
  first runs that weigh those scales in the loss, not on a flat sky of
  noise alone, where a model leaves far less;
- star flux and FWHM unchanged within 1%;
- injected Gaussian blobs at 1–3σ keep at least 80% of their amplitude.

Against B halves, the error is measured both as they are and against
halves matched to A, on the pairs whose halves agree (see the pairs
above). Runs are compared by region, sky, faint, mid and cores, not by the
error over the whole image, which a few defects can rule: in one pair a
cold blob of 37 pixels, thousands of σ̄ below the background in both
halves and by different amounts, held two thirds of the error, and a model that was better
in every region scored worse over the whole. Each pair is also checked on
its own: a run whose output on any pair's sky strays far from the noise
fails, however good its means without that pair.

Every noise run also makes contact sheets, so runs are judged by eye
beside their measures. The store lists a few chosen images: nights from the
validation sets, from the held-out telescope and from a training set, mono
narrowband and broadband, bright structure and the faintest. For each, one
sheet takes a 4-frame half as the input and the night's deep stack as the
reference. The other takes the deep stack itself, with nothing deeper, and
shows in place of a reference the noise taken out divided by each pixel's
own noise. Where a run takes out only noise that panel is even grain at
every brightness, so any shape in it is signal the run changed; magnified
without the division, a bright core's own photon noise looks like damage.
An object stacked over many nights adds a third sheet, the reference
night's 4-frame half against the whole stack. Each row is a 1:1 crop. The panels
show the input, then everything the run finds taken out, then the same
fixed shares kept, all under the reference's stretch. The shares never
change, so one run's sheet can be laid beside another's. The list can name
baselines for an image, the same input denoised some other way, which
then sit beside the run's output, and, for an image written whole, are
written whole once in the same stretch, so every run is judged against
the same thing. A night's deep
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
- the software version and `seiza-model`'s `OUTPUT_VERSION`;
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
