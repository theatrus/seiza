# Gaia-anchored sharpening

## Short answer

A solved stack and the Gaia catalogue tell Seiza where every star is, to a
small fraction of a pixel, and where none is. With the stack's own PSF
photometry at those fixed positions, they also say how bright each star is.
That makes the stellar part of any real image a known quantity at any PSF:
the same stars, at the same positions and fluxes, can be rendered through
the target PSF. Seiza can therefore give the [PSF model](ml-restoration-training.md#psf-model)
exact star targets on real data. Synthetic composites remain the only exact
targets for nebulae.

Gaia is used for six things:

1. **Star truth on real stacks.** Fit the stars at Gaia positions, remove
   them, and render them again at the target PSF. Real backgrounds, noise,
   field-varying PSFs, coma and spikes all come with them.
2. **A better PSF field.** Stamps centred on Gaia positions carry no
   centroiding error, and neighbours Gaia resolves are fitted, not guessed.
3. **The star mask.** The star and non-stellar strengths need a mask that does
   not depend on what the model, or a detector, can see. Gaia sets it.
4. **Gates that need no person.** Astrometry, photometric linearity,
   faint-star completeness, false stars and resolved doubles are each
   measured against Gaia.
5. **A classical star sharpener.** Remove the fitted stars and render them
   again at a narrower PSF. It needs no model, so it is both the baseline for
   star sharpening and its fallback.
6. **Star removal and its fills.** Gaia says which pixels to fill and,
   where it lists no star, gives holes whose true contents are known, for
   training the fill and for gating it.

The sharpening model also takes a point-source map, from Gaia when the
image is solved, because whether a source is a point decides how small it
may become.

Non-stellar deconvolution gets no such truth. It keeps the plan's
composites, seeing splits and reblur consistency, with a better PSF field
and a cleaner split between what is star and what is not.

## What Seiza has

- Gaia DR3 bundles, merged with Hipparcos for the brightest stars:
  - the solver catalogue;
  - the deep catalogues, to G 17 and to G 20, the second optional and large;
  - the photometry catalogue with BP and RP, to G 15.

  `seiza-sources` reads proper motions and the G, BP and RP magnitudes.
- A solver that fits SIP distortion and takes an epoch (`solve`,
  `solve_with_prior`), and star detection with measurements
  (`detect_measured_stars`).
- In the corpus, one solved frame per set and its WCS (`measure.py`). The
  deep stack, the halves and the seeing-split stacks share the registration
  reference's grid.

Still missing: a WCS on the stack grid itself, fitted to sub-pixel accuracy,
and the `PsfField` of the restoration plan.

## Per-set Gaia products

A corpus pass, `gaia`, runs after `build` on every set whose frame solved.

1. **Solve the deep stack.** Detect stars on the deep stack and solve them
   with the set's frame solution as the prior. Use SIP order 3 and the
   set's mean observation epoch. Record the residual of the bright,
   unsaturated, isolated stars. A set whose residual exceeds 0.1 px gets no
   Gaia products, and the reason is recorded.
2. **List the stars.** Cone-search the deepest installed catalogue over the
   footprint plus a margin of the widest PSF stamp. Propagate positions by
   proper motion to the epoch, then map them to pixels.
   - Flag the stars that are hard to fit: a duplicated source, a high RUWE
     (a likely unresolved binary), Gaia's galaxy and quasar candidates
     where the bundle carries those fields, and anything inside a compact
     object of the object catalogue.
   - Add RUWE to the deep bundles if they lack it.
3. **Fit fluxes at fixed positions.** Run PSF photometry on the deep stack,
   with the positions held fixed. Fit stars whose stamps overlap jointly,
   with a local background plane.
   - Fit a photometric model on the unsaturated, isolated stars:
     `−2.5 log f = G − zp + c₁(BP−RP) + c₂(BP−RP)²`. Record its scatter.
   - The model gives the flux a saturated star should have, which the
     bloom target needs, and a prior for the faint stars.
   - In narrowband, emission-line stars and continuum slopes widen the
     scatter. Fit a saturated star's flux there from its unsaturated wings
     instead, through the PSF field, and keep the model's value only as a
     cross-check.
4. **Measure the PSF field.** Build `PsfField` from Gaia-centred stamps of
   stars with no Gaia neighbour within the stamp radius.
   - Faint stars, too faint alone, are stacked by position in bins of
     magnitude and field position to give the wings.
   - Saturated stars give the far wings, matched over an overlap radius as
     the plan already describes.
5. **Draw the star mask.** Each star's radius is where its fitted profile
   falls to half the local noise. Halos the classical halo tool measures are
   added. Compact objects stay outside the mask, following the
   [star policy](ml-restoration-training.md#star-policy).

The same products are made for each input a pair uses: A halves,
seeing-split stacks and finished stacks. Fluxes are fitted again on each
one, because their normalisation and PSF differ.

## Star truth on real stacks

Let `y` be a real input stack whose PSF field is `P`. Let `T` be the target
PSF the ratio map asks for. Let `fᵢ` and `xᵢ` be the fitted fluxes and the
Gaia positions.

- Remove the stars: `B = y − Σ fᵢ P(x − xᵢ)`, which is catalogue-guided
  removal on `y` itself.
- Render the target: `t = B + Σ fᵢ T(x − xᵢ)`.

Inside the star mask, `t` is exact up to the fit's error, which is recorded
per star. Outside it, `t = y`. These pairs teach **star sharpening with no
non-stellar sharpening**: the ratio map is `T/P` inside the mask and 1
outside it. They also teach halo and bloom reduction on real optics, by
rendering the target's wings scaled and its saturated discs at the
photometric model's flux.

**Hybrid seeing-split pairs** join the two kinds of truth:

- The input is the worst-quartile stack.
- The target's background is the best-quartile stack with its own fitted
  stars removed.
- The target's stars are the deep stack's fluxes rendered at any `T` down
  to the 1.5 px floor.

The ratio map is then the measured `P_best/P_worst` outside the mask and
`T/P_worst` inside it. The two strengths differ in every such pair, which is
the case the controls exist for.

Losses follow the truth:

- inside the mask, against `t`, where it is exact;
- outside it, against the seeing-split background, weighted by its noise;
- everywhere, reblur consistency at the plan's low weight.

A star whose fit's χ² is poor, or that a flag above marks, is masked from
the loss instead of trusted.

## The classical star sharpener

Gaia also gives a sharpener with no model: remove the fitted stars at `P`
and render them again at `T`.

- Halo and bloom reduction are the same render with scaled wings and a
  disc sized to its flux.
- Stars that Gaia lacks are left as they are, and the run reports them.
  They come from the residual of the removal: variables, novae, asteroids,
  and stars in fields too crowded for Gaia.

It is exact where the fit is good, and visibly wrong where it is not, which
makes it both the baseline the PSF model must beat on stars and the
fallback for solved images. Non-stellar sharpening stays with field-varying
Richardson–Lucy, run on `B` with the new PSF field, so stars never take part
in an iteration and cannot ring.

## Gates against Gaia

Each gate is measured on every validation stack that has Gaia products, for
the input, the output and the deep stack, so each number has the data's own
reference beside it:

- **Astrometry.** Centroids of the output's stars, measured at the Gaia
  positions, move by less than 0.05 px. They are measured against Gaia,
  not against the input's own centroids, which share its errors.
- **Photometric linearity.** Aperture flux against the photometric model's
  flux, from the brightest unsaturated star to the faintest: a slope of 1
  and no curvature, and each star within 1%.
- **Achieved PSF.** FWHM and ellipticity of Gaia-centred stamps of the
  output, by magnitude and by field position, against the ratio map's
  request.
- **Faint-star completeness.** The share of Gaia stars detected at S/N ≥ 3,
  in magnitude bins down to the input's limit. Sharpening, and denoising,
  must not lower it. The deep stack says which faint stars are real.
- **False stars.** Detections in the output with no Gaia counterpart to
  G 20, outside compact objects and away from the removal's flagged
  residuals. Their rate must not exceed the input's.
- **Resolved doubles.** Gaia pairs separated by one to three FWHM of `P`:
  the output's separation within 0.1 px of Gaia's, and its flux ratio
  within 5% of the fitted ratio.
- **Rings.** The radial profile around each bright Gaia star, after the
  model, does not dip more than 1σ below the background. The restoration
  plan already has this gate; Gaia positions now centre it.

Completeness and false stars apply to the noise model as well. They test,
against a reference that does not depend on the model, whether denoising
loses faint stars or makes new ones.

## Star removal and fills

Star removal leaves holes, and the star model fills them by carrying the
surrounding structure in. Gaia helps on both sides:

- **Where to fill.** The holes are the star mask: Gaia positions and
  radii from the fitted fluxes. A fill never reaches a pixel no star
  covers, and a faint star a detector misses is still removed.
- **Fills with known truth.** Gaia also says where no star is. On a real
  stack, cut holes of the field's own star-mask shapes and sizes where Gaia
  lists no star to G 20 within the hole and a margin. Fill them, and compare
  the fill with the pixels that were there. That truth is exact, on real
  nebulae with real noise, and the holes can be placed over the edges and
  filaments where fills fail. It trains the fill and gates it: the fill's
  error is within the noise, and the fill's own noise matches its
  surroundings within 10%, as the star model's gate asks.

## Model inputs

Whether a source is a point is what lets the model shrink it. A star can
be made as small as the sampling allows, because a point source has no
detail of its own to lose. A nebula's knot cannot be made smaller than the
data resolved it. So the sharpening model is told where the point sources
are, and how small they may become:

- **A point-source map.** Each Gaia star is rendered at `P` with its
  predicted flux. This is an input channel of its own, beside the image,
  the PSF stamp, the noise map and the control maps. On an unsolved image,
  or one without the deep catalogue, the map is drawn from `PsfField`'s
  detections instead. Training uses that substitute a third of the time,
  and leaves the map empty a tenth of the time, so the model degrades to
  the detection-only case and no further. A gate compares the two: on the
  validation stacks, a detection-only map costs no more than run-to-run
  spread outside crowded fields.
- **A star-size floor.** The smallest PSF to which stars may shrink is a
  map of its own: the sampling floor of 1.5 px by default, or, for a
  natural look, the PSF of the set's best-seeing frames, which the frame
  measurements record. The star part of the ratio map cannot ask for
  less.

Gaia's role at training time does not make it a requirement at inference
time: an image may be unsolved, the deep catalogue may not be installed,
and a moving object is in no catalogue. The detection-based map covers
each case, and the gate above measures what it costs.

## Narrowband, crowding and other limits

- In narrowband, Gaia predicts star fluxes poorly. Positions, existence and
  the mask still hold, and fitted fluxes carry the truth.
- Gaia is incomplete for the very brightest stars, where Hipparcos fills in,
  and in the densest fields. Crowded fields use the fitted residual as a
  second star list and are flagged.
- Gaia resolves pairs down to about 0.4″. A closer pair is one star to Gaia,
  and to every amateur stack.
- High proper-motion stars move by arcseconds over a decade. The epoch
  propagation in the solver handles them.
- Stars on a nebula's bright edge fit worse. Their χ² flags them, and the
  loss drops them.

## Staged delivery

1. **Stack WCS and Gaia list.** Add the `gaia` corpus pass: solve the deep
   stack and record its residual, then propagate and list the stars. Only
   the training code changes.
2. **`PsfField` at fixed positions.** Land `PsfField` and fixed-position PSF
   photometry in `seiza-stars`, with their C and Python twins. This is the
   restoration plan's foundation, with Gaia centring added.
3. **Gaia gates in training.** Run completeness, false stars, linearity and
   astrometry on the noise model's validation stacks first. The noise model
   exists now, and the faint-star question is open for it.
4. **Classical star sharpener.** Land removal and rendering at `T`, with
   halo and bloom, as an operation of its own. It is the baseline for step 6.
5. **Pairs.** Make star-rendered real pairs and hybrid seeing-split pairs
   for the train sets, and the star model's fill holes where Gaia lists no
   star. Keep the composites for nebulae.
6. **PSF model runs**, on the CUDA machine, all with the point-source map:
   - composites alone;
   - with the star-rendered real pairs added;
   - with the hybrid pairs added.

   Each run is measured with the map from Gaia and with the map from
   detections, against the Gaia gates and the plan's reblur and ring gates.
7. **Land `sharpen` (experimental)** with both strengths, the star-size
   floor, halo and bloom, and the classical star sharpener as its fallback.

Progress, November 2026 (built on branches, not yet merged):

- **Step 2, in part.** `PsfField` measures empirical stamps across the
  field, as a PCA basis whose coefficients follow a polynomial. The
  polynomial's degree, 1 to 3, is chosen for each channel by five-fold
  cross-validation, because a tilted sensor needs a cubic. The catalogue is
  used for isolation only; stamps are not yet centred on Gaia positions.
  - On a refractor with a known sensor tilt, the FWHM runs from 2.0 to 3.5
    px in a V across the frame, with ellipticity up to 0.43 at one edge.
    A plane explains only 56–63% of that FWHM variance.
  - On a refractor whose bright stars bloom, the wings come from 29
    saturated stars.
- **Step 4, in part.** Catalogue stars are fitted at their solved
  positions, subtracted, and rendered again at `T`. Fitted flux follows G
  with slopes 0.98–1.06, with a scatter of 0.17–0.41 mag over three
  stacks. On one stack, sharpening to 2 px took the centre's FWHM from
  2.36 to 2.12 px. Halo profiles, core fills, bloom and the BP−RP model
  are not built.
- **Extended sharpening's classical baseline.** Richardson–Lucy runs with
  the field-varying PSF, through a remaining kernel `R`, where `P = R ∗ T`
  at each node, towards a round `T`. It narrows the tilted stack's FWHM
  range from 2.0–3.6 to 1.9–2.3 px and its ellipticity from 0.24 to 0.16.
  The reblur χ² is 1.06–1.08 on two stacks, but 8.7 on the tilted one.
  Without deconvolution, the input itself scores 13.3. On sky alone, the
  output scores 1.6, so the excess sits on the stars, many of them
  undersampled near 2 px, where the PSF model's residual is 5–7%.
- **Open.** A reference for the residual spectrum's flatness: a stack's
  noise is correlated, so even a perfect estimate falls towards Nyquist.
  The difference of the half stacks would give one. Doing better on near-2
  px stars needs a super-resolved PSF basis.
