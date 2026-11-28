# A learned star removal model

Status: planned, 2026-11-25. Nothing is trained yet.

Seiza's third learned model, beside the noise model and the sharpening model
(`ml-restoration-training.md`, `ml-runtime.md`): it takes an image and gives
back two, the image without its stars and its stars alone, diffraction spikes
included. Its task is `starless` (`seiza model starless`, beside `denoise`
and `sharpen`), its published models `starless-v1-x<run>`, its runs X1, X2
and on.

## What it gives

- **Two layers that add back.** Removal happens before any stretch, on the
  linear image. The network gives the stars layer; the starless image is the
  input less it, so starless + stars is the input exactly. The stars layer
  holds the stars' light and nothing else: no noise, no nebula. The noise
  stays in the starless image, as the other models leave noise where it is.
- **Layers for Seiza's own star workflow.** The two are stretched apart and
  put back together: the starless image with a stretch for the nebula, the
  stars with Seiza's star stretch (`seiza color star-stretch`,
  `seiza_stacking::stretch_rgb_stars`, which takes the linear original and
  starless images, finds the stars as their difference and applies a
  rational stretch that keeps small stars small), then screened back over it
  (`seiza color screen-star-layer`). So what the model must get right is the
  linear starless image, and with it a stars layer that is only stars: any
  noise or nebula it held would be stretched as stars.
- **Colour.** The star stretch and screen work on RGB, and colour stacks are
  common: a colour image's stars are removed from its three channels
  together, from the same positions and the same shapes, so no channel keeps
  a star another lost and no coloured rim is left.
- **Spikes are stars, with a layer of their own.** A star's diffraction
  spikes leave the starless image clean, and can be given as a spikes image
  apart from the stars layer, so a user may keep stars without spikes, or
  spikes alone to restyle them; starless + stars + spikes is still the
  input. How many there are and where they point depends on the telescope's
  obstructions: none for a refractor or most catadioptrics, four from two
  crossed vanes, six from three, the 6 + 2 of JWST's hexagonal segments and
  its strut, and they may be thin, thick, chromatic or curved. Nothing in
  the model assumes one pattern. The bank keeps each composite's spikes'
  light apart for this: a made pupil's PSF less the same optics' without
  its vanes or hexagon, so the obstruction's rings stay with the star.
- **The catalogue and a network.** Seiza is a plate solver, so the model
  need not work from the image alone. The image is solved, its stars listed
  from Seiza's deep Gaia catalogue (to G 20), fitted with its own PSF field
  and taken out (`seiza.StarFit`), as the clean skies below are made; the
  network then does what fitting cannot. An image that will not solve falls
  back to the stars the detector finds on it.
- **Healing is the work.** What a fit leaves is where the network earns its
  place: the holes and remnants of bright and clipped cores, spikes, and
  above all the gaps, healed with the nebula or background that runs under
  them, so that nothing shows where a star was.
- **After sharpening too.** A common order is to sharpen an image and then
  take its stars out, so the model must take stars out of a sharpened image
  as well as a raw one: tighter cores, and whatever a sharpening leaves
  about them. The training composites are also met sharpened, by Seiza's
  sharpening model and as the sharpening goal sharpens, with the truth made
  alike: the starless truth is the clean sky sharpened, and the stars layer
  what sharpening the composite adds to it, so the two still add back to
  the sharpened input.
- **Halos are stars.** A bright star's halo (its wide wings, and the rings
  a filter's or a corrector's reflections throw) reaches far past any PSF
  stamp, tens to hundreds of pixels, and belongs in the stars layer with
  the star. On a deep stack of 8.6 px stars, the stars brighter than G 13
  leave light after the fit that the stack's two independent halves agree
  on, 13 times the noise of the mean at their cores; past 20 px the
  brightest leave about 3% of their light, and on a refractor's deep stack
  of 2.1 px stars the clipped stars of G 3 to 8 leave 9 to 13% of theirs,
  out past 300 px. A stack holds too few bright isolated stars to read a
  halo's shape cleanly over its nebula: the halos are read over many stacks
  of each telescope (2026-11-25).
- **Halo reduction is a setting.** How much of a halo leaves with its star
  is the user's to choose, from none (the halo stays in the starless image)
  to all of it, in either design. So the halos' light is given apart too: a
  halo layer, of which a chosen share goes back into the starless image and
  the rest with the stars, the layers still adding back to the input. The
  bank keeps each composite's halos' and rings' light apart for this, as it
  does the spikes'.

## The data

### Known truth first

A star removal network trained on what another method removes learns that
method's mistakes. So the first data has its truth known exactly, both
layers:

1. **Clean skies** (`starless_sky`). Each stack is solved, its stars listed
   from the deep Gaia catalogue to G 20 with the detector's stars the
   catalogue lacks, and those a first fit at their catalogue places
   measures at a signal to noise of 3 or more fitted with the stack's own
   PSF field (places refined from a signal to noise of 5, the faintest
   kept at the catalogue's) and taken out; the field's systematic miss,
   read from bright isolated stars' leftovers, is taken out with them.
   What is left is the stack's sky and nebula with its noise. About each
   star that is clipped, fits poorly, leaves its core far over or under its
   noise, or leaves it leaning to one side (a dipole: a star fitted a
   fraction of a pixel off), a disc is masked and kept out of the loss
   rather than patched. Around 1,100
   deep and multi-night stacks of seven telescopes give these, and the
   Hubble and JWST fields give sharper skies of other telescopes. On a deep
   stack of 8.6 px stars, with G 17: a tenth of a percent of each star's
   light left after the correction, and the stars between G 13 and G 17
   leaving nothing the stack's halves agree on more than on the sky beside
   them; with G 20, 7,968 stars taken out where G 17 listed 1,796 (a fit at
   the catalogue's places measures far fainter stars than the detector
   finds: it found 3 the G 17 list lacked), the fainter ones leaving small
   remnants the halves share, for the network to heal, and 3.9% of the
   image masked (2026-11-25).

   The first bank's windows showed the clean skies keeping bright stars'
   leftovers, dipoles and cores 30 to 100 times the sky's noise over or
   under. They passed because each was judged against a photon noise that
   had soaked up the PSF's shape miss. The leftovers' variance is now read
   as two terms, the stars' photon noise, growing with their light m as
   g·m, and the shape's miss, growing as (f·m)², and leftovers are judged
   against the photon noise alone. Deep stacks read g near 0 (many frames
   make a star's own photon noise small) and f of 8 to 13%, so a bright
   star's leftover is masked wherever it shows above the sky's noise. On a
   dense field of 2.3 px stars the masked share goes from 3.7% to 11.7%,
   and from refining at a signal to noise of 50 to refining at 5 alone it
   had gone from 6.0% to 3.7%.

   The pass over all 655 stacks of the corpus took 5.2 hours on two
   workers (2026-11-26) and gave 645 clean skies. Nine stacks had too few
   isolated stars for a PSF field that varies. Measured again with one PSF
   for all, seven of them gave clean skies, so 652 in all. The other two
   had no usable star at all, and one stack would not solve. The masked
   share's median is 4.6%,
   its quartiles 2.3% and 10.1%, and a tenth of the stacks are over 20%.
   The 34 over 30% are nearly all dense fields of undersampled stars, with
   FWHM about 2 px and 100,000 to 900,000 stars. There the PSF's shape
   miss is 10 to 30%, a pixel's phase changing the shape the field gives.
   Windows are cut only where 80% of the pixels are valid, so those fields
   give few windows: the first banks under-represent the densest
   undersampled fields until their fit is better.

   The validation bank's windows show the masks covering the dipoles and
   the clipped cores. They do not cover a bright star's halo. The fit's
   stamps are 33 px, or 7 FWHM where that is wider, so a star's light past
   them is never taken out. A masked disc grows with the logarithm of the
   star's peak over the noise, and even a clipped star's disc reaches only
   13 to 15 FWHM, about 30 px on 2.3 px stars. Its glow past that stays in
   the truth unmasked, and the network is taught to keep the halos of the
   brightest real stars. (Clipped stars' fitted fluxes are not capped by the
   clip: fitted from their wings, they read one to two times their
   catalogue brightness.)

   Taking that light out does not work on a stack. On 2.3 px stars, 129 px
   stamps leave nothing past 9 px. But on 8.3 px stars, every stamp size
   took out a flat 1 to 4 millionths of a star's light a pixel too much
   (5 to 8% of it), because a stamp measured from clipped stars reads its
   background far from the star. Reading what the fit leaves as a profile
   by flux, inside and outside the stamp, and taking that out, fails too,
   because what is left grows with a star's brightness. In a crowded field
   the faint stars on a bright one's wing take some of it for their own
   light: on a dense stack such a profile held −15% of a star's light, and
   would have added light about every bright star. A truth must hold only
   what was measured, so version 3 of the clean skies takes nothing more
   out and masks instead. About each clipped or very bright star, what is
   left is read ring by ring in eight sectors. A ring glows where, in seven
   of its sectors at least, the median stands over 0.3 of the noise above
   the same sector 1.55 times further out. A star's glow is round; a nebula
   about it slopes and passes on one side only. The star's disc reaches on
   while its rings glow (the reason "glow"): 10 to 16 stars on deep stacks
   of 2.3 px stars, with the masked share unchanged to a tenth of a
   percent.
2. **Stars put back** (`starless_bank`). Stars are drawn into a clean sky at
   random places, between half and one and a half times as many as the
   clean-sky pass took out of it, their magnitudes drawn from those it took
   out and their light from the stack's own magnitude scale, so the field's
   crowding and brightnesses are its own. Each is the shape of a PSF:
   - the stack's own, its field's stamps with their wings and asymmetry,
     carried on past the stamps' edge (65 px) by a Moffat wing matched to
     their light where they end;
   - or one made from a pupil: a central obstruction of a given share, 0,
     2, 3 or 4 vanes of given width at any angle, or JWST's segments, its
     diffraction pattern summed over the filter's wavelengths and blurred by
     seeing. These give every spike pattern, and spikes long enough to cross
     a whole tile for the brightest stars;
   - with a halo for the bright ones: wide wings and reflection rings, from
     the halos measured about our own stacks' bright stars (their mean
     profiles past the stamp, by brightness and telescope) and made ones of
     other radii and widths, so the network learns to take a halo out and
     heal what it covered.

   A star's stamp reaches as far as its light stays over a twentieth of the
   noise and fades out in a circle past that, so no star, halo or ring ends
   in a square edge, however bright: the first bank cut stamps at 31 or
   255 px, and wide stars ended in squares.

   Pooled over the 652 clean skies (2026-11-26), the light that bright,
   isolated stars leave past 20 px is small. On an SCT of 7.4 px stars
   (254 stacks), G 3 to 8 leave 2.5% of their light there and G 10 to 12
   leave 1.4%, fading out by about 100 px. On fast astrographs of 2.2 px
   stars (391 stacks), G 3 to 8 leave 0.8 to 2.4%. What looked like a
   reflection ring at 16 to 24 px on those, and a shelf at 24 to 37 px on
   the SCT, is the light past the fit's stamps (33 and 53 px): with
   stamps of 129 px the ring is gone. The first banks' made halos, 3 to 15%
   of a star's light, and rings, 0.5 to 3%, are stronger than these. Bank
   version 3 draws halos from 0.5% and rings from 0.2%, evenly in the
   logarithm, so that faint halos, the common kind, are among those
   learned.

   The fit's estimate in the bank (not the truth) draws each bright star's
   stamp out to 0.6 of its half side and blends it from there into a
   Moffat wing of the field's FWHM and β, matched on that ring, carried
   past the stamp and faded round. A measured stamp's outer part holds the
   field's noise and any pedestal, which a bright star's flux raises into a
   square of noise; X1 learned that square. On a test field of β 2.5
   stars, the estimate's model follows the stars' own wings to within 20%
   from 18 to 25 px, where the fit's alone falls to under a third of them.
3. **Noise.** The stars' own photon noise, at the stack's photon term, is
   added to the input, with the clean sky's noise already there.
4. **Saturation.** Cores are clipped where the stack clips.

   The truth: the starless image is the clean sky, with its noise; the
   stars layer is the input less it, the stars as the image shows them,
   clipped and with their photon noise, so the two add back to the input
   exactly.
5. **The fit.** Each composite is also fitted as the runtime will fit an
   image: its stars listed at the places they were put, as a catalogue
   lists them, with the PSF field measured on the composite itself. Its
   starless estimate and stars model are kept beside the truth, so both
   designs train from one bank.

### Pairs of real images

Known truth teaches what to remove; real pairs teach what the starless
image should look like where a star was, on fields like the user's. The
corpus's stacks will be paired with a goal starless image and stars image
for each (as the sharpening model's corpus pairs were), first on the train
split, and drawn beside the known-truth windows by a share.

## Measuring

Every run is measured on held-out fields before any is kept, at full size and
on sheets, beside the goal:

- **Light left behind.** Each Gaia star's light still in the starless image,
  by magnitude, from a ring photometry against its surroundings: what is left
  of bright stars, and whether faint stars are removed at all.
- **Holes and rings.** Dark dips and bright rings where stars were, in units
  of the noise.
- **Spikes left behind.** The starless image's mean along each spike
  direction of the brightest stars, against the same at other angles.
- **Nebula in the stars layer.** The stars layer's correlation with the
  starless image's structure away from stars; nebula must not leave with the
  stars.
- **What fills the gaps.** On independent halves of a stack, the structure
  both starless images share where stars were, against what they share
  elsewhere: a fill invented from noise is not shared.
- **The sum.** starless + stars = input to float precision.
- **Through the star workflow.** The layers star-stretched and screened as a
  user would, beside the goal's put through the same steps: star sizes,
  coloured rims, and what the screen shows where stars were.

## The model

The two designs are trained on one bank and compared (`train-starless`):

- `fit`: a residual U-Net reads the image, the catalogue fit's starless
  estimate and stars model, the noise map and the stars' places, and
  corrects the estimate: it heals the holes, remnants, spikes and halos
  the fit leaves.
- `image`: the same network reads the image, the noise map and the stars'
  places alone, and corrects the image.

Each starts as its baseline, the fit's estimate or the image, and its loss
is logged beside the baseline's. A tile is read in units of its own noise,
its sky taken off, through asinh, so a bright core and the sky's noise both
count; the loss is a Charbonnier penalty over the valid pixels, ten times
over where stars were.

Either design may also give the halos and the spikes (`--layers`): two
planes more from the same network, each the light the bank keeps apart
for it, read about zero through the same asinh and penalised alike beside
the starless image. The input is then four layers that add back to it: the
starless image, the halos and rings, the spikes, and the stars' cores and
wings, which is what is left and carries the stars' photon noise. Of the
halos a chosen share goes back into the starless image and the rest with
the stars (`starless_model.kept`), so halo reduction is a setting from 0 to
1, and the spikes are their own image. A spike's light is signed: the
vanes take from the core the light they throw out along the spikes, so the
spikes layer holds a small dip at each core, and a star without its spikes
has the core of the same telescope with no vanes.

Later, a U-Net as the other two (width 96 to 128, five levels), with the PSF
stamp encoder the sharpening model has, so it knows the stars' shape at each
tile. Big stars' halos and
spikes reach hundreds of pixels: tiles of 1024 and a binned pass for the
widest (`scale_to`), as sharpening does. The stars layer is given through the
asinh residual the sharpening model uses, so faint and bright stars weigh
alike.

### First runs (2026-11-26)

X1 (`fit`) and X2 (`image`) were trained for 10,000 steps of 16 tiles of
256 px, at width 32 and five levels, on the first full bank: 8,028 windows
of 512 px from 511 stacks, half with the stack's own stars and half with
made ones. Each took 11 minutes on one GPU. They were measured on 128
windows of the validation bank's 560, from 140 held-out stacks. Errors
below are in noise units, as medians over windows:

| | stars, rms | sky, rms | cores, mean | stars, stretched | sky, stretched |
|---|---|---|---|---|---|
| the fit's estimate alone | 0.884 | 0.067 | 0.049 | 0.081 | 0.0044 |
| X1, fit-first | 0.446 | 0.072 | 0.034 | 0.059 | 0.0039 |
| X2, image only | 0.556 | 0.125 | 0.048 | 0.079 | 0.0065 |

X1 halves the fit's error where stars were and leaves the sky as it found
it. Its mean error over windows falls from 8.6 to 0.51, because the bright
stars the fit leaves holes and remnants about are healed. X2 starts from the
image with its stars in (validation loss 0.245, against 0.045 for the fit's
estimate) and ends at 0.035, behind X1's 0.028. Its sky error is twice X1's,
because it takes some of the sky's noise out with the stars.

The sheets show what the numbers miss. The fit's star models end in square
stamps (33 to 65 px), whose outer parts hold the field's measured noise:
inside, the star is taken out with that noise raised by its flux, and
outside, its wing and halo are left. X1 learned that edge. About bright stars it leaves a
square-edged patch, and in the stars layer the sky's noise shows in a square
about each star. X2 has no squares, and over a bright star on nebula its
fill carries the nebula through where X1 leaves a flat blob. Both fill the
widest halos with a smooth patch a little darker than the sky about it, and
put the sky's noise there into the stars layer: where a star buries the
sky's noise, a network trained to the median gives none back.

Then (2026-11-26; results under Second runs):

- Clean skies of version 3 (bright stars' glows masked) and banks of
  version 3: the fit's estimate drawn with smooth wings past a trusted ring
  (the runtime's fit to do the same), halos and rings from the faint ones
  measured up, and each composite's halos' and spikes' light kept apart.
  The wings are most of a draw's cost: the field's shape (FWHM and β) is
  read at a star in about 1 ms, and it was read for 34,000 of a deep
  stack's 77,000 stars to give 6,300 of them wings. A bound on each star's
  light on the trusted ring now passes over the rest before their shapes
  are read: the same estimate to the bit, in 40% less time. The runtime's
  fit is to do the same.
- X3 (`fit`) and X4 (`image`), as X1 and X2 on the new banks, with X1 and
  X2 measured again on the new validation bank beside them.
- Where a star buries the sky's noise, the starless image is given noise
  to match after the network, and the same taken out of the stars layer,
  so the layers still add back to the input (`starless_model.filled`;
  built). It reads only the image: its grain (variance about a 3 px mean)
  against that grain's median over 96 px blocks, read again away from
  where the first reading found it lacking, so a wide fill does not lower
  its own measure; noise of the image's level and neighbour correlation
  (0.18 on our stacks) is drawn where 30% of the grain or more is
  missing, fully from 50%. A noise map will not do: it is raised about a
  bright star by the star (23 times the sky's at one core). X1's fills keep
  60 to 70% of the grain's spread in blotches, and lack it only within 10
  to 15 px of the core, where the fill now draws it. The measures are of
  the network's own output; `noise_kept` gives the grain kept where stars
  were, beside the filled output's.
- Later: a halo layer and a spikes layer from the network, for halo
  reduction as a setting and spikes as their own image; then the wider
  network.

### Second runs (2026-11-27)

The version 3 banks: 8,024 train windows from 511 stacks (one dropped,
too few stars for a PSF field), 3,908 with the stack's own stars and 4,116
with made ones, and 560 validation windows from 140 held-out stacks. X3
(`fit`) and X4 (`image`) were trained as X1 and X2 were, 10 minutes each.
All four were measured on 128 windows of the new validation bank, the
network's output and the same with matched noise drawn (`filled`).
Medians over windows, in noise units:

| | stars, rms | sky, rms | cores, mean | stars, stretched | noise kept |
|---|---|---|---|---|---|
| the fit's estimate alone | 0.829 | 0.060 | 0.072 | 0.076 | 1.064 |
| X1, fit-first (v2 bank) | 0.443 | 0.066 | 0.055 | 0.059 | 0.976 |
| X3, fit-first | 0.455 | 0.068 | -0.003 | 0.060 | 0.971 |
| X3, noise filled | 0.459 | 0.069 | -0.003 | 0.060 | 0.973 |
| X2, image only (v2 bank) | 0.560 | 0.125 | 0.033 | 0.080 | 0.919 |
| X4, image only | 0.552 | 0.114 | 0.042 | 0.078 | 0.915 |
| X4, noise filled | 0.574 | 0.118 | 0.037 | 0.080 | 0.935 |

- No run leaves a square on the new bank: the fit's estimate has none to
  learn, and X1, which learned the old one's, has none to copy.
- X3 is level with X1 and fills cores at the right level (its mean error
  there is 0, X1's +0.055). X4 is a little ahead of X2, most on the sky.
- About the brightest stars, the image-only runs do better. The fit's wing
  is round, so a pupil's spikes past the trusted ring are left to the
  network, and both fit-first runs leave them in part, X3 more than X1.
  12 to 40 px from the six brightest stars of the windows measured, X3
  leaves +0.02 to +0.41 noise of halo light on average, X1 -0.02 to +0.14,
  X2 and X4 within ±0.1.
- The drawn noise raises the grain kept where stars were (X4 0.915 to
  0.935); being noise of its own, not the truth's, it raises the rms a
  little.
- All four runs' validation losses flatten by 7,750 steps, as the one-cycle
  learning rate falls away; whether more steps or a wider network buy more
  is still to be seen.

### Third runs (2026-11-27)

X5 (`fit`) and X6 (`image`) are X3 and X4 with the halo and spike layers,
on the same banks, width and steps (11 minutes each), so the starless
image can be set beside theirs like for like. Beside the starless
measures, each layer is measured where its truth tops the noise: how much
of the truth it holds (the scale of the truth that best meets it, 1 all, 0
none), and the error where the truth holds under a tenth of the noise,
light put in the layer that is not its own. Medians over the same 128
windows, in noise units:

| | stars, rms | sky, rms | cores, mean | halos found | spikes found |
|---|---|---|---|---|---|
| X3, fit-first | 0.455 | 0.068 | -0.003 | | |
| X5, fit-first, layers | 0.463 | 0.084 | 0.016 | 0.75 | 0.003 |
| X4, image only | 0.552 | 0.114 | 0.042 | | |
| X6, image only, layers | 0.562 | 0.124 | 0.050 | 0.68 | 0.003 |

- The halo layer works: 70 to 75% of the halos' light found, and next to
  none put where there is no halo (0.007 to 0.009 noise rms).
- The spike layer learned nothing. Its penalty stayed at its zero-output
  baseline through both runs, and a run on the train bank's 327 windows
  with spikes above the noise alone moved it by about a tenth, while the
  halos and the starless image learned.
- The starless image is a little behind X3's and X4's, most on the sky.
- Every bank so far moved its made stars into place by Fourier transform,
  which rings along the row and the column through a core narrower than
  its sampling: every bright star carries a dotted cross along the pixel
  axes that no telescope makes, counted as the star's light. On composites
  cut alike but for this, the light 20 to 60 px from the brightest stars
  is 7 to 18 times as great along the axes as off them; moved by
  Lanczos-3, 0.4 to 2.6 (more than 1 where a spike runs near an axis). The
  cross is often brighter than a pupil's own spikes. The starless image
  need not tell the two apart, as both go; the spike layer must, and the
  cross is on every bright star, spikes on some, so a layer trained to the
  median that cannot yet tell them apart stays at zero. That is likely,
  not yet shown, to be why the spikes were not learned.
- The fit draws its stars by Lanczos-3, as `seiza.StarFit` does, so its
  estimate kept the crosses and the fit-first runs left part of them: some
  of what the second runs read as spikes left past the trusted ring was
  this cross.
- Banks of version 4 move made stars by Lanczos-3 (`starless_bank
  .add_star`), as the fit draws them; they are being cut. The spike layer
  is to be tried on them before more runs.

### Fourth runs (2026-11-27)

The version 4 banks are cut from the same stacks and draws as version 3
(8,024 train windows from 511 stacks, 560 validation windows), with made
stars moved by Lanczos-3. X7 (`fit`) and X8 (`image`) are X5 and X6 trained
on them, alike in all else (11 minutes each). All six runs so far with
these settings were measured on 128 windows of the version 4 validation
bank. Medians over windows, in noise units:

| | stars, rms | sky, rms | cores, mean | stars, stretched | halos found | spikes found |
|---|---|---|---|---|---|---|
| the fit's estimate alone | 0.617 | 0.047 | 0.023 | 0.069 | | |
| X3, fit-first (v3 bank) | 0.441 | 0.061 | 0.005 | 0.058 | | |
| X5, fit-first, layers (v3 bank) | 0.447 | 0.077 | 0.009 | 0.057 | 0.69 | 0.002 |
| X7, fit-first, layers | 0.415 | 0.066 | 0.018 | 0.054 | 0.82 | 0.003 |
| X4, image only (v3 bank) | 0.545 | 0.111 | 0.045 | 0.075 | | |
| X6, image only, layers (v3 bank) | 0.554 | 0.118 | 0.060 | 0.075 | 0.70 | 0.003 |
| X8, image only, layers | 0.544 | 0.102 | 0.030 | 0.075 | 0.70 | 0.003 |

- The fit's own estimate is much nearer the truth on the new bank (stars
  0.62 against 0.83 on version 3): it draws stars as the bank now does, so
  it has no cross to leave.
- X7 is the best run so far: the least error where stars were, plain and
  stretched, and 82% of the halos' light found. About the six brightest
  stars measured, 12 to 40 px out, it leaves 0.15 to 0.33 noise rms of
  error, mean -0.08 to +0.17, where X3 leaves 0.22 to 0.53, mean +0.03 to
  +0.42. The fit-first design is now level with the image-only runs about
  bright stars (X4 and X8 0.18 to 0.28), as well as ahead everywhere else.
- X8 is ahead of X6 on the sky and the cores, and level with X4 on the
  stars. Its mean over windows is worse than its median: a few windows of
  the stacks' own stars pull its mean stars error to 0.66 (X4 0.62).
- The spike layer still learned nothing (0.3% found). The cross was not
  the reason, or not the only one. Looked into on the train bank's 396
  windows with spikes over the noise, with a small network (width 16,
  800 steps), measured on the validation bank's 19 such windows:
  - The data holds. The spikes stand out in the stars' light, the spike
    truth lies on them, and the light less the truth is round.
  - On those windows the visible spikes carry 3.7% of the loss's weight
    and a third of the spike penalty a zero output pays; the dip the
    spikes leave in the core a quarter; light under the noise the rest. A
    run with the loss as it is gains 18% on the dip, 14% on the faint
    light and under 1% on the visible spikes, finding 0.4% of their light.
  - The spike channel trained alone finds 6.5% by the same penalty, 21%
    by squared error.
  - In the whole loss, the spike channel's penalty changed: squared error
    finds 13%, four times weighted 15%; a Charbonnier penalty quadratic up
    to 0.25 (about 1.3 noise), four times weighted, 12%, with the least
    light put where there are no spikes (0.035 noise rms, against 0.056
    for squared error and 0.023 for the loss as it is).
  - So two things hold the spikes back. The penalty seeks the median, so a
    pixel near a bright star stays at zero until the network can tell
    where spikes run, as it learns to only by first leaning toward them.
    And the spike channel's share of the loss is small beside the starless
    image's. That penalty suits the starless image, whose edges it keeps,
    but not a sparse layer.
- Next for spikes: the spike channel's penalty quadratic near zero and
  weighted up, perhaps with windows with spikes drawn more often.

### Fifth runs (2026-11-27)

X9 (`fit`) and X10 (`image`) are X7 and X8 with the spike layer's penalty
quadratic up to 0.25 and weighed four times (`--spike-quadratic 0.25
--spike-weight 4`), alike in all else. Medians over the same 128 windows:

| | stars, rms | sky, rms | cores, mean | halos found | spikes found | spikes stray |
|---|---|---|---|---|---|---|
| X7, fit-first | 0.415 | 0.066 | 0.018 | 0.82 | 0.003 | 0.004 |
| X9, fit-first, spikes eased | 0.420 | 0.060 | 0.014 | 0.79 | 0.056 | 0.017 |
| X8, image only | 0.544 | 0.102 | 0.030 | 0.70 | 0.003 | 0.004 |
| X10, image only, spikes eased | 0.539 | 0.105 | 0.024 | 0.69 | 0.045 | 0.014 |

- The spike layer moved, from 0.3% of the spikes' light found to 5%, 5 to
  15% on the windows with the most spikes, but it is far from taking them.
  The starless image is as before: every run still leaves most of a
  pupil's spikes in it, and the halo layer takes some of their light.
- The windows with spikes over the noise are 5% of the train bank (396 of
  8,024), so a run of 10,000 batches of 16 sees about 8,000 of them. On
  those windows alone (a width 16 network, batches of 8), with the eased
  penalty the spike layer finds about 10% up to 2,000 steps, then 78% at
  2,500 and 69 to 86% to 4,000, with 0.04 noise rms put where there are no
  spikes; by squared error it climbs sooner (35% at 2,000) to 78% at 2,500
  but falls back to 47% by 4,000, with more stray light (0.05 to 0.07).
  The layer learns once it has seen some 20,000 windows with spikes, so
  the fifth runs stopped short of it.
- Next for spikes: the eased penalty, with windows with spikes drawn as a
  share of every batch, enough for some 40,000 of them in a run.

### Sixth runs (2026-11-27)

X11 (`fit`) and X12 (`image`) are X9 and X10 with a quarter of every batch
drawn from the train bank's 397 windows with spikes over the noise
(`--spike-share 0.25`, `starless_model.spiky_windows`), some 40,000 of them
a run. Medians over the same 128 windows:

| | stars, rms | sky, rms | cores, mean | halos found | spikes found | spikes stray |
|---|---|---|---|---|---|---|
| X9, fit-first | 0.420 | 0.060 | 0.014 | 0.79 | 0.056 | 0.017 |
| X11, fit-first, windows with spikes | 0.422 | 0.053 | 0.030 | 0.83 | 0.18 | 0.022 |
| X10, image only | 0.539 | 0.105 | 0.024 | 0.69 | 0.045 | 0.014 |
| X12, image only, windows with spikes | 0.544 | 0.107 | 0.009 | 0.53 | 0.56 | 0.020 |

The layers add back to the input as the truth's do, so each layer's error,
projected on the spike truth, shares out the spikes' light. Over the
validation bank's 19 windows with spikes:

| | spike layer | left in the starless image | halo layer | stars layer |
|---|---|---|---|---|
| X7 | 0.01 | 0.13 | 0.47 | 0.39 |
| X9 | 0.10 | 0.11 | 0.54 | 0.25 |
| X11 | 1.03 | -0.01 | 0.08 | -0.10 |
| X8 | 0.00 | 0.04 | 0.37 | 0.59 |
| X10 | 0.05 | 0.04 | 0.49 | 0.42 |
| X12 | 0.88 | 0.02 | 0.14 | -0.05 |

- The spike layer is learned. X11 puts all the spikes' light in it and none
  in the starless image; X12 nearly all. The medians over windows are
  lower (0.18 and 0.56), as most windows' spikes are faint; their means are
  0.36 and 0.47.
- Before, the halo layer took about half of the spikes' light, so its
  share found read higher than it was. X12's halo layer finds less now
  (0.53 over all windows, 0.72 on a pupil's).
- The starless image is as good as before: X11 has the best sky yet
  (0.053), its stars level with X9's, its cores a little brighter (0.030).

### Whole images (2026-11-27)

X11 and X12 were run over four whole validation stacks, real stars with
no truth: the Elephant's Trunk in Hα (C925, 4176 × 6248, its brightest
star G 4.7), Sh2 230 in Hα, and NGC 6820 in Hα and in R (Ultracat, 6388 ×
9576, 165,000 to 590,000 catalogue stars). Each was fitted at its
catalogue stars' places as a bank fits a composite (12 s for the C925
stack, 1 to 3 minutes for the Ultracat ones), then read in 512 px tiles.
A whole Ultracat stack takes about 20 s through either network.

- The stars are gone and the nebula runs on through where they were, the
  brightest included, in Hα and in the dense R field. The fit's estimate
  alone leaves dark and bright marks at every bright star; the network
  heals them. X11 leaves a faint ring at the Trunk's brightest double star,
  where X12 leaves none.
- Tiles read about their own level and noise, as in training, disagree a
  little where they meet. Kept hard-edged (the middle 384 px of each),
  they leave steps of up to 0.7 noise (the 99.9th percentile of two
  offset tilings' difference). Overlapped by half and blended by
  raised-cosine weights, 0.27 to 0.29, 0.025 rms. Read about the whole
  image's level and noise, the blend leaves 0.03, but the starless image
  changes by 0.4 to 0.7 noise rms in bright nebula (10 to 50 noise over
  the image's median), as no tile read that way in training. The examples
  use the blend about each tile's own level; a runtime wants one level for
  the image, which training with the level moved at random would allow.
- The C925's stars throw no spikes, yet the spike layer puts a small dip
  in their cores, about 1% of the stars' light, balanced by as much spread
  thin about them. It adds back with the stars. X11 and X12 do the same on
  the composites' spike-free stars (0.8 to 1.1% of their light, X7 0.1%),
  so it comes of how the spike layer was trained, not of real stars.
- Both take some nebula out. Of the light each takes out of a stack, the
  share taken where no catalogue star is (outside a disc about every
  source the clean-sky pass lists, 3 star widths, wider for bright ones):
  X12 0.6 to 1.5% on the four nebulae, X11 0.4 to 0.9%. X12 dims bright
  nebula ridges and takes compact knots; X11 took a small bright shell
  about a listed star in Sh2 230 for its halo (43% in the halo layer, 53%
  in the stars). The points plane marks a star by its light's share of the
  brightest's, so a G 20 star's mark is a few millionths: in training most
  faint stars look unmarked, and a network learns to take an unmarked
  compact source as a star. A plane of where any star is marked (`marks`)
  and the made galaxies, compact and kept, answer that.

### Seventh runs (2026-11-27)

X13 (`fit`) and X14 (`image`) are X11 and X12 with each training tile read
about a level and noise moved from its own as far as a whole image's are
from its tiles' (`--level-jitter`, `starless_model.jittered`): nine times
in twenty 1.5 to 60 noise below, two in five near, three in twenty above.
X15 and X16 add a quarter of every batch from windows with bright stars
and no spikes (`--spikeless-share 0.25`), against the spikes' false dip.
Medians over the same 128 windows, each read about its own level:

| | stars, rms | sky, rms | cores, mean | halos found | spikes found |
|---|---|---|---|---|---|
| X11, fit-first | 0.422 | 0.053 | 0.030 | 0.83 | 0.18 |
| X13, level moved | 0.432 | 0.061 | 0.028 | 0.78 | 0.19 |
| X15, and bright spike-free windows | 0.434 | 0.059 | 0.065 | 0.77 | 0.14 |
| X12, image only | 0.544 | 0.107 | 0.009 | 0.53 | 0.56 |
| X14, level moved | 0.568 | 0.108 | 0.049 | 0.56 | 0.56 |
| X16, and bright spike-free windows | 0.567 | 0.101 | 0.079 | 0.67 | 0.61 |

Read about a level moved from the window's own, as a whole image read
about one moves it, the stars' rms (48 windows):

| | own level | 3 noise below | 10 below | 30 below |
|---|---|---|---|---|
| X11 | 0.453 | 0.589 | 0.994 | 3.55 |
| X13 | 0.478 | 0.484 | 0.510 | 0.652 |
| X12 | 0.543 | 0.972 | 2.30 | 3.75 |
| X14 | 0.567 | 0.584 | 0.616 | 0.736 |

- Trained about their own level only, the runs fall apart read about
  another: two to eight times the error 10 to 30 noise off, as a bright
  nebula's tiles are from a whole image's median. Trained with the level
  moved, they hold, for about 5% more error about their own. A runtime
  that reads an image about one level, with no seams between its tiles,
  needs this.
- The spikes are learned as before.
- The bright spike-free windows leave the false dip as it was (1.1% of the
  stars' light below zero with or without, 0.9% for X16), so it is not for
  want of seeing such stars; they are not kept.

### Eighth runs (2026-11-27)

X17 to X20 are X13 and X14 with twice the channels (64; X17, X18) or four
times the steps (40,000; X19, X20), all else the same. Medians over the
same 128 windows:

| | stars, rms | sky, rms | halos found | spikes found |
|---|---|---|---|---|
| X13, fit-first | 0.432 | 0.061 | 0.78 | 0.19 |
| X17, wider | 0.468 | 0.079 | 0.35 | 0.07 |
| X19, four times as long | 0.405 | 0.044 | 0.83 | 0.68 |
| X14, image only | 0.568 | 0.108 | 0.56 | 0.56 |
| X18, wider | 0.542 | 0.106 | 0.65 | 0.70 |
| X20, four times as long | 0.504 | 0.095 | 0.71 | 0.77 |

- Training longer is the gain. X19 is the best run so far: 6% less error
  on the stars than X13 and 29% less on the sky, and the fit-first run now
  finds the spikes (0.68, from 0.19), which were still being learned at
  10,000 steps. X20 has 11% less error than X14. Both level off by 36,000
  steps (validation 0.0769 at 30,000, 0.0754 at 40,000 for X19).
- The wider fit-first run stalls: its validation loss holds at 0.089 from
  step 2,000 on, the halos' grows, and it finds few spikes. The wider
  image-only run has 5% less error than X14, less than X20 gains.
- Fit-first stays ahead: 20% less error on the stars than image-only, at
  either length.
- The long runs hold about a moved level better too. Stars' rms (48
  windows) at their own level, then 3, 10 and 30 noise below: X19 0.417,
  0.423, 0.449, 0.496 (X13 0.478 to 0.652); X20 0.499, 0.509, 0.514, 0.570
  (X14 0.567 to 0.736).
- The spikes' false dip halves with the training: 0.6% of the stars' light
  below zero for both, from 1.1%.

### Galaxies

A galaxy is not a star. Small ones look like stars, and a large one's
nucleus does. They are kept in the starless image by default; a setting
moves them to an output of their own (`galaxies`), as the halos' share is
a setting, and the layers still add back to the input.

The runs so far take them out. Two test-split stacks the runs never saw,
the Coma Cluster (Ultracat, G, 254 frames; 395 galaxies in the object
catalogue over it) and the field of M51 (C925, L, 125 frames), share of
each galaxy's light (inside its catalogue ellipse, over the median about
it) left in the starless image:

| | size, px | X11 keeps | X12 keeps |
|---|---|---|---|
| eight of Coma's smallest | 8 | 0.16 to 0.98 | about 0 |
| NGC 4889, elliptical | 52 × 34 | 0.25 | 0.03 |
| NGC 4874, elliptical | 46 × 40 | 0.40 | 0.12 |
| NGC 4921, spiral | 39 × 36 | 0.64 | 0.78 |
| IC 4278 | 39 × 37 | 0.99 | 0.16 |
| IC 4277, edge on | 62 × 7 | 0.99 | 0 |

- The image-only run takes nearly every compact galaxy out with the stars.
- The fit-first run keeps more, but where Gaia lists a galaxy's nucleus as
  a source the fit takes it out as a star, and the network leaves the hole:
  NGC 4874's and NGC 4889's cores, and the nuclei of M51 and NGC 5195.
- Seiza's deep catalogue is Gaia DR3's sources to G 20, with no class, so
  it lists compact galaxies and galaxies' nuclei. The clean skies took them
  out as stars too, so the truth lacks them, and the point plane marks them.
- The made stars fall anywhere, and the clean skies are mostly the Milky
  Way's nebulae, so the networks have seldom seen a compact source to keep.

The plan:

1. **Which sources are galaxies.** The object catalogue lists galaxies
   with their ellipses (`seiza.field_objects`), down to 8 px on the
   Ultracat, and a runtime has it. A catalogue source within a galaxy's
   ellipse is the galaxy's, unless it is a star before it: point-like in
   the first fit and away from the galaxy's middle. Gaia's own galaxy
   candidates and astrometry could make this surer later, in the deep
   catalogue's bundles.
2. **Clean skies keep galaxies.** The galaxies' sources are not taken out,
   so the truth keeps them, and each stack's galaxies are kept with it.
3. **The point plane marks stars only**, and the fit leaves the galaxies'
   sources out, in the banks and at runtime alike.
4. **Made galaxies.** Composites get galaxies as well as stars: a bulge and
   a disc, 2 to 60 px across, of any shape and angle, the smallest as
   small as the stars, at the brightnesses of the stack's own. A
   `galaxies` truth plane holds them; the starless truth keeps them.
   Within the real galaxies' ellipses, the galaxy layer is not penalised
   (their own light is not known apart from the sky's).
5. **A galaxy layer.** The network gives `galaxies` beside the halos and
   spikes. By default it stays in the starless image; a share kept (`kept`)
   moves the rest to the galaxies output.
6. **Measures.** On composites, the made galaxies' light kept in the
   starless image and found in the layer. On the Coma Cluster and M51
   above, the catalogue galaxies' light kept.

## Order of work

1. The clean skies: Gaia fits with the PSF field, and checks that what is
   subtracted is the stars (the halves agree on what is left). Built.
2. The pupil PSF maker and the star-injection bank, with tests that put back
   known stars and find them. Built, and the trainer for both designs.
3. The corpus pairs on the train split.
4. The measures and the sheets.
5. First runs: X1 and X2, above.
6. A `star-removal` task in `seiza-model` beside `denoise` and `sharpen`,
   mono and RGB: the starless image and the stars layer, written as files
   `seiza color star-stretch` and `screen-star-layer` take, through the C
   ABI, Python and the CLI.
