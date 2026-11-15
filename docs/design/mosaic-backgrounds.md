# Registered mosaic backgrounds

`BatchStackOptions::mosaic_panels` assigns a panel ID to every registered
input. Repeated IDs group observations of the same panel. Both registered
integration functions use it; C accepts `mosaicPanels`, Python accepts
`mosaic_panels`. Leaving it out preserves existing output.

Inputs must already be calibrated, registered on the same pixel grid, and
matched in photometric gain. Missing samples are NaN, including holes within
a panel. Zero and negative finite samples are data. At most 256 distinct
panels are accepted.

## Expanded canvases

`LiveStacker::reintegrate` and the other replays of a stack on an expanded
canvas match panel backgrounds when `StackOptions::match_mosaic_backgrounds`
is enabled (C: `matchMosaicBackgrounds`, Python: `match_mosaic_backgrounds`,
CLI: `--match-mosaic-backgrounds`). It defaults to false independently of
normalization. The `mosaic_panels` option itself is still refused on replay,
since the stack knows its own panels. Each
admitted frame belongs to the panel planning made that its footprint shares
the most pixels with. The matcher reads each frame as the replay normalizes
it, after its recorded gain and level and any background refit, so it sees
the samples the rejection passes see. Banded replays read every kept frame
once more, a row of background cells at a time; whole-frame replays prepare
every frame once more before their three passes.

Nothing is matched where a stack keeps its inputs' intensities
(`NormalizationMode::None`), where its frames all lie on one panel, or where
planning made more than 256 panels. A reintegration of some frames alone fits
the planes from those frames, so two subsets' stacks can differ by a smooth
plane as well as by their frames. The live pass is unchanged: it cannot
revisit a panel once later panels have joined.

On two synthetic panels of six frames, whose nights' skies sloped opposite
ways by a third of the sky level, the reintegrated stack stepped by 4.1 ADU
where the second panel's coverage begins. With panel matching it steps by 0.1
there and 0.5 where the first panel ends, measured against the same panels
under level skies, whose noise is 0.7 ADU a pixel. Where skies agree already
the matcher fits their noise: on two panels of two frames each, noise 3 to 6
ADU a sample, it moved the stack by at most 0.6 ADU.

Matching is opt-in because uncorrected vignetting and unequal photometric
gains can be mistaken for additive backgrounds. Earlier experiments with an
automatic shared-bowl model improved some mosaics but worsened others,
including colour gradients. Those measurements do not validate this model.
Calibrate and match photometric gains before enabling it; compare the result
with matching disabled.

## Matching and blending

Each input contributes a bounded lattice of weighted samples to its panel.
Samples retain their pixel positions and independent channel support until
panels are compared. Partial edge cells contribute wherever the panels have
matching finite samples. The lattice includes adjacent pixel pairs to retain both parities of
alternating channel masks. A cell's difference between two panels is the
median of its paired differences, so shared stars and emission cancel first
and what is left of them cannot pull it. Panels whose gains differ by a few
percent leave that share of every shared star in their differences, all of
one sign; a mean clipped about the median averaged those wings in, and in a
dense field tilted the planes of skies that agreed.

Bright cells are excluded from the plane fit. When bright emission leaves
separate background groups, paired residuals match those groups by a robust
constant. They do not add gradient constraints or contradict planes already
connected through dim overlaps. All usable dim overlaps constrain the panels'
additive planes in one solve; residual clipping rejects outlying cells. A
penalty on correction energy over each footprint fixes the common-plane
ambiguity and bounds extrapolation from narrow overlaps. It weighs how far a
correction varies over its footprint thirty times more than the correction's
mean: an overlap a tenth of a panel wide measures a slope across itself over
a tenth of the panel, and a slope read there must not be carried over the
whole panel. A slope measured along an overlap as long as the panel keeps all
but a thirtieth of itself. Narrow overlaps with too few cells for a plane can
still constrain a relative constant.

Vignetting is not inferred from overlap differences. The difference between
two quadratic bowls about different centres is a plane, for any number of
panels. An unpenalized common bowl therefore absorbs real sky gradients too.
Only additive planes are fitted; disconnected overlap components do not share
parameters. Flat-fielding belongs to calibration.

At each matching sample position, corrected panel values form a common
background weighted by distance from each channel's footprint edge and the
recorded frame weights. Only differences from this common value are averaged
into residual cells and interpolated. A cell's residual is the median of its
samples' differences, so neither a sampled transient hit nor a star the
panels' gains disagree about can spread into a smooth correction. A cell
holding matching samples at fewer than half its positions, a sliver of a
panel's edge, measures none and takes its neighbours' residual: its few
samples are the frame's outermost pixels, and its centre lies beyond the
panel, where interpolation would lean on it hardest. On two panels of a dense
field such slivers held 20 to 40 ADU under skies that agreed, and bent the
panel's last eight columns by 10 ADU. A residual is also shrunk by how far it
stands above its own noise, the scatter of its samples' differences over the
root of their count: under three standard errors it is zero, and well above
it keeps nearly all of itself. A panel of a few short exposures otherwise
prints that scatter into the stack as blotches a cell wide.
Full-resolution samples still undergo
Seiza's usual weighted rejection; coverage and per-channel holes are preserved.
This is background matching, not an absolute sky estimate: a gradient shared
by all panels cannot be measured from their overlaps. Overlaps smaller than
the sampling spacing may have no usable matching samples; disconnected
panels without sampled shared coverage cannot constrain each other's sky.

The overlap solve and separation of background from detail were informed by
[Mega Merge Mosaic](https://github.com/Astrometrical/mega-merge-mosaic/blob/main/docs/DESIGN.md).
This implementation uses Seiza's image I/O, registered samples, cancellation,
CPU pool and integration. It does not add the external CLI or copy its file
readers, star ownership or pyramid engine. Registration and photometric gain
errors still require correction upstream; background blending cannot repair
doubled stars or an incorrect stellar flux scale.

## Cost

Cells are at least 16 pixels wide, increased to keep the longest grid side
at most 128 cells. Each cell reads at most 64 sample positions per frame.
The grid spacing also increases with panel and channel count to keep the
aggregate weighted sample arrays within 128 MiB. Each retained position uses
16 bytes per panel and channel. Cell means, distances and corrections need
another 16 bytes per cell per panel and channel while fitting.

The plane solve has three unknowns per panel and one dense matrix per channel.
It holds about 40 bytes per usable cell in each pairwise overlap while clipping
the fit. Level-only matching uses another matrix with one unknown per
disconnected background group. Overlap collection scales with the number of
panel pairs and retained sample positions.

The banded path gathers summaries while loading each input into scratch;
full-resolution correction runs as bands are read for integration. Scratch
files and final image buffers retain the existing integration costs. The
whole-frame fallback makes an additional input read to fit backgrounds and
checks that every later read matches those original samples.

## Regression checks

The output-version cases include a sampled Gaussian star under a steep sky,
a partial-cell overlap, independent RGB missing samples, a sampled transient
hit, four overlapping panels with different skies and shared diffuse emission,
and an expanded canvas of two panels whose nights' skies sloped opposite ways,
reintegrated. Core tests reject false vignetting inference on four planar panels
and check that adding a disconnected pair leaves existing corrections unchanged.
They also measure recovery of the known scene
up to the unobservable common plane, preserve a bright star and missing
pixels, and compare whole-frame and banded output bit for bit. C and Python
tests exercise the same panel-ID option and reject mismatched ID counts.

[Heart and Soul before/after comparisons and measured residuals](../images/mosaic/README.md).
