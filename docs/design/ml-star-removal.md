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
- **Spikes are stars.** A star's diffraction spikes go into the stars layer
  and leave the starless image clean. How many there are and where they point
  depends on the telescope's obstructions: none for a refractor or most
  catadioptrics, four from two crossed vanes, six from three, the 6 + 2 of
  JWST's hexagonal segments and its strut, and they may be thin, thick,
  chromatic or curved. Nothing in the model assumes one pattern.
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

## The data

### Known truth first

A star removal network trained on what another method removes learns that
method's mistakes. So the first data has its truth known exactly, both
layers:

1. **Clean skies** (`starless_sky`). Each stack is solved, its stars listed
   from the deep Gaia catalogue to G 20 with the detector's stars the
   catalogue lacks, and those a first fit at their catalogue places
   measures at a signal to noise of 3 or more fitted with the stack's own
   PSF field (bright ones' places refined, faint ones' kept) and taken out;
   the field's systematic miss, read from bright isolated stars' leftovers,
   is taken out with them. What is left is the stack's sky and nebula with
   its noise. About each star that is clipped, fits poorly, or leaves its
   core far over or under its noise (judged with its photon noise), a disc
   is masked and kept out of the loss rather than patched. Around 1,100
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
2. **Stars put back.** Stars are drawn into a clean sky at random places, with
   the brightness counts of a real field (from Gaia's counts at the field's
   latitude), each the shape of a PSF:
   - a real one, a stamp of another field's PSF, its wings and its
     asymmetry included;
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
3. **Noise.** The stars' own photon noise is added to the input only, with
   the clean sky's noise already there. The target stars layer has none.
4. **Saturation.** Some stars are clipped at a level, as real cores are, with
   their bleed where the camera would bleed; the target keeps the star's
   unclipped light, so the network learns to take a clipped core out whole.

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

A U-Net as the other two (width 96 to 128, five levels), reading the image,
its noise map and its point sources, with the PSF stamp encoder the sharpening
model has, so it knows the stars' shape at each tile. Big stars' halos and
spikes reach hundreds of pixels: tiles of 1024 and a binned pass for the
widest (`scale_to`), as sharpening does. The stars layer is given through the
asinh residual the sharpening model uses, so faint and bright stars weigh
alike.

## Order of work

1. The clean skies: Gaia fits with the PSF field, and checks that what is
   subtracted is the stars (the halves agree on what is left).
2. The pupil PSF maker and the star-injection bank, with tests that put back
   known stars and find them.
3. The corpus pairs on the train split.
4. The measures and the sheets.
5. First runs.
6. A `star-removal` task in `seiza-model` beside `denoise` and `sharpen`,
   mono and RGB: the starless image and the stars layer, written as files
   `seiza color star-stretch` and `screen-star-layer` take, through the C
   ABI, Python and the CLI.
