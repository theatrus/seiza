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

Next (under way, 2026-11-26):

- Clean skies of version 3 (bright stars' glows masked) and banks of
  version 3: the fit's estimate drawn with smooth wings past a trusted ring
  (the runtime's fit to do the same), halos and rings from the faint ones
  measured up, and each composite's halos' and spikes' light kept apart.
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
