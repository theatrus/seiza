# Registered mosaic backgrounds

`BatchStackOptions::mosaic_panels` assigns a panel ID to every registered
input. Repeated IDs group observations of the same panel. Both registered
integration functions use it; C accepts `mosaicPanels`, Python accepts
`mosaic_panels`. Leaving it out preserves existing output.

Inputs must already be calibrated, registered on the same pixel grid, and
matched in photometric gain. Missing samples are NaN, including holes within
a panel. Zero and negative finite samples are data. At most 256 distinct
panels are accepted. A `LiveStacker` still manages its own background refit;
this option belongs to the registered-frame integration path.

## Matching and blending

Each input contributes a bounded lattice of weighted samples to its panel.
Samples retain their pixel positions and independent channel support until
panels are compared. Partial edge cells contribute wherever the panels have
matching finite samples. The lattice includes adjacent pixel pairs to retain both parities of
alternating channel masks. Clipping operates on paired differences, so shared
stars and emission cancel before averaging or rejection.

Bright cells are excluded from the plane fit. When bright emission leaves
separate background groups, paired residuals match those groups by a robust
constant. They do not add gradient constraints or contradict planes already
connected through dim overlaps. All usable dim overlaps constrain the panels'
additive planes in one solve; residual clipping rejects outlying cells. A
small penalty on correction energy over each footprint bounds extrapolation
from narrow overlaps and fixes the common-plane ambiguity. Narrow overlaps
with too few cells for a plane can still constrain a relative constant.

At each matching sample position, corrected panel values form a common
background weighted by distance from each channel's footprint edge and the
recorded frame weights. Only differences from this common value are averaged
into residual cells and interpolated. Residual differences are clipped before
averaging so a sampled transient hit cannot spread into a smooth correction.
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
hit, and four overlapping panels with different skies
and shared diffuse emission. Core tests measure recovery of the known scene
up to the unobservable common plane, preserve a bright star and missing
pixels, and compare whole-frame and banded output bit for bit. C and Python
tests exercise the same panel-ID option and reject mismatched ID counts.

[Heart and Soul before/after comparisons and measured residuals](../images/mosaic/README.md).
