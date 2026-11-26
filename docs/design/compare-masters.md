# Comparing finished masters

Status: proposal, for discussion on
[#502](https://github.com/theatrus/seiza/issues/502). No code yet.

`seiza analyze` (`seiza_analysis::analyze`, Python's `analyze_project` and
`analysis_html`) measures *lights* before they are stacked. Collaborations,
and anyone processing old data again, need the same for *finished masters*:
many stacks of one target from different rigs, integrators and processing,
put side by side on fair terms. The WR 134 collaboration that raised #502
compared 84 masters from 17 contributors with Python built on Seiza 0.56 and
RC-Astro. This document proposes the Seiza call that would make such a tool
a thin layer, and the order to build it in.

## What a fair comparison needs

Two masters of one target differ in ways that say nothing about their
quality: scale, rotation, field, crop, pixel units, sky level and stretch.
A comparison has to take those away before it measures anything, and it has
to measure what is left on one scale:

1. **Know where each master looks.** Each needs a plate solution, with SIP
   distortion on a wide field, or residual alignment errors read as blur.
2. **Put every master on one grid** around the target, at one pixel scale.
3. **Put every master on one brightness scale**, per channel: a gain and an
   offset that map it onto a reference.
4. **Measure on that grid and scale**: noise at a fixed size on the sky,
   star FWHM, residual alignment against the reference, and coverage.
5. **Build a reference worth comparing to**: an inverse-variance consensus
   of the matched masters.
6. **Show them with one stretch per channel**, resolved once and applied to
   every master, since a stretch resolved per image hides the differences
   the comparison is for.

## What Seiza already has

Most of the pieces landed alongside #502:

| Step | Building block | Where |
| --- | --- | --- |
| Read a master's pixels | `seiza_image_io::read_image` (#496, #512) | FITS and XISF, compressed XISF included |
| Read what it records of its integration | `seiza_image_io::read_master_info` (#499, #517) | total exposure, frames, filter, integrator, drizzle scale, each with its source |
| Plate solution from the header | `seiza::Wcs::from_fits_values` | TAN with SIP |
| Solve a master that has none | `seiza::image_solve::solve_image` | keeps the SIP fit of a strongly distorted field and says when one is dropped (#495, #511) |
| One grid | `seiza::Wcs::from_center_scale_rotation` | north up with rotation 0 |
| Reproject onto it | `seiza_stacking::reproject` (#497, #515) | SIP on both sides, antialiased when downsampling, per-pixel coverage |
| Match gain and offset | `seiza_stacking::match_linear` (#500, #507) | per channel, with the gain's standard error |
| Noise at a sky scale | `seiza_stacking::noise_at_scale` (#500, #507) | block means, so resampled noise reads right |
| Star FWHM | `seiza_stars::PsfField::measure` and `shape` | per channel |
| One stretch for many images | `seiza_stretch` `resolve_for` / `resolve`, then apply (#498, #510) | the plan reads back from its JSON to the same bits |
| Many RC-Astro runs | `RcAstroCli::process_images` / `split_stars_batch` (#501) | one start of the CLI per batch, for processing variants |
| HTML reports | `seiza_analysis::render_html`, `render_stack_collection_html` | embedded templates and charts |

What is missing is the workflow that runs them in order, the consensus
stack, residual alignment, and the report.

## Proposed API

Rust first, as `AGENTS.md` asks: typed values in and out, no JSON between a
Rust host and the work. A new `compare` module in `seiza-analysis`, which
already holds the report templates and depends on `seiza-stacking`,
`seiza-image-io` and `seiza-stars`; it gains `seiza` (the solver) and
`seiza-stretch`.

```rust
/// One master to compare.
pub struct MasterInput {
    pub path: PathBuf,
    /// Shown in the report; the file stem when absent.
    pub label: Option<String>,
    /// A solution to use instead of the header's, such as one a host
    /// solved itself.
    pub wcs: Option<Wcs>,
}

/// The grid every master is reprojected onto.
pub struct ComparisonGrid {
    /// RA and Dec of the grid's middle pixel, in degrees.
    pub center_deg: (f64, f64),
    pub scale_arcsec_px: f64,
    pub width: usize,
    pub height: usize,
}

impl ComparisonGrid {
    /// A north-up grid over `center_deg` covering `field_deg`, at the
    /// coarsest master's scale, so no master is upsampled.
    pub fn around(center_deg: (f64, f64), field_deg: (f64, f64),
                  masters: &[MasterSolution]) -> Self;
    pub fn wcs(&self) -> Wcs;
}

/// Where a master's solution comes from when its header has none.
pub enum MissingSolution {
    /// Leave the master out, with a warning in the comparison.
    Skip,
    /// Solve it against these catalogues.
    Solve(SolveCatalogs),
}

pub struct CompareOptions {
    pub grid: ComparisonGrid,
    pub missing_solution: MissingSolution,
    pub reproject: ReprojectOptions,
    pub matching: LinearMatchOptions,
    /// The size on the sky noise is measured at, in arcseconds
    /// (default: 4 grid pixels, and never under 3 of the coarsest
    /// master's native pixels).
    pub noise_scale_arcsec: f64,
    /// How far from the consensus, in its robust sigmas, a master's sample
    /// may lie before the consensus leaves it out (default 3).
    pub consensus_clip_sigma: f64,
}

/// Compare masters of one target, which may come from different rigs and
/// integrators. Masters are grouped into channels by the filter they
/// record (colour masters give three), and each group is matched, measured
/// and combined on its own.
pub fn compare_masters(
    inputs: &[MasterInput],
    options: &CompareOptions,
    cancel: Option<&CancelSignal>,
    progress: &mut dyn FnMut(CompareProgress),
) -> Result<MasterComparison, CompareError>;

pub struct MasterComparison {
    pub grid: ComparisonGrid,
    pub channels: Vec<ChannelComparison>,
    pub masters: Vec<MasterReport>,
}

pub struct ChannelComparison {
    /// The filter, such as "Ha", or "R", "G", "B" of colour masters.
    pub name: String,
    /// The inverse-variance consensus on the grid, in the units of the
    /// master it was matched to first.
    pub consensus: LinearImage,
    /// The stretch resolved on the consensus, for every master's preview.
    pub stretch: StretchPlan,
}

pub struct MasterReport {
    pub label: String,
    pub path: PathBuf,
    pub info: MasterInfo,
    pub solution: SolutionReport,     // from header or solved; SIP kept or dropped
    pub coverage: f64,                // share of the grid it covers
    pub channels: Vec<MasterChannel>, // one per channel it contributes to
    pub warnings: Vec<CompareWarning>,
}

pub struct MasterChannel {
    pub channel: usize,                // index into MasterComparison::channels
    pub matched: ChannelMatch,         // gain, offset, gain_error, blocks
    /// Noise at `noise_scale_arcsec`, in consensus units after matching.
    pub noise: f64,
    /// (consensus noise / this noise)², the share of the consensus's depth
    /// this master alone reaches; with `MasterInfo`'s exposure, depth per
    /// hour.
    pub relative_depth: f64,
    pub fwhm_arcsec: Option<f64>,
    /// Median and RMS offset of its stars from the consensus's, in
    /// arcseconds: what its solution left misaligned.
    pub alignment: Option<AlignmentResidual>,
    /// Consensus weight it was given.
    pub weight: f64,
}

/// The report as one self-contained HTML file.
pub fn comparison_html(
    comparison: &MasterComparison,
    options: &ComparisonHtmlOptions,
) -> Result<String, Error>;
```

A refusal a host acts on is a `CompareError` variant made where it happens:
no master with a solution, a grid larger than the configured limit, no
channel with two masters. A master that cannot be used (unreadable, no
solution under `Skip`, no overlap with the grid, a fit with no common
signal) is a `CompareWarning` on that master, not an error for the run.

### The C ABI and Python

Following `AGENTS.md`:

- C: `seiza_compare_masters_json` runs the whole workflow from one JSON
  request (the paths, labels, grid and options) and answers the report
  without the images; consensus and preview images go to files named in
  the request, read through `seiza-image-io`.
  `seiza_comparison_html_json` renders a report. Request and answer types
  live in `seiza-analysis`, behind its `json` feature, hold no logic and
  call `compare_masters`.
- Python binds the native call: `seiza.compare_masters(paths, *, grid,
  labels=None, ...)` returns a `MasterComparison` whose consensus images
  are NumPy arrays, and `seiza.comparison_html(comparison)`, like
  `analyze_project` and `analysis_html`. Both get parity entries naming
  the C exports, stub entries and tests.

## How each step works

**Solutions.** The header's TAN+SIP solution, or `MasterInput::wcs`, or a
solve when `MissingSolution::Solve`. A solve that dropped its SIP fit is
kept and says so in `SolutionReport`, since a linear solution of a wide
field misaligns its corners by arcseconds (#495) and the alignment residual
will show it.

**Grid.** North up at the coarsest master's scale by default, so every
master is downsampled or kept, never upsampled; `reproject` antialiases
the downsampling. A host can choose any other.

**Channels.** A mono master's channel is the filter it records; a colour
master contributes to R, G and B. Masters with no filter, or one that
matches no other master, form their own channel and are reported but not
matched. Comparing an L master with an R master is left to the host.

**Consensus, in two passes**, so 84 masters on a large grid never sit in
memory together:

1. A provisional reference per channel: the master with the most recorded
   exposure, or, without exposures, the lowest `noise_at_scale` in its own
   units divided by its median. Every other master is reprojected, matched
   to it with `match_linear`, and its matched noise measured; nothing else
   is kept.
2. Each master is reprojected again, matched, and accumulated into a
   weighted mean with weight `1 / noise²`, leaving out samples more than
   `consensus_clip_sigma` from the provisional reference. The consensus is
   that mean, and every master is matched to it once more for the numbers
   the report gives.

Reprojecting twice costs time, not memory; a host with the memory can keep
the first pass's reprojections (an option, not the default).

**Depth** is `noise_at_scale` of the matched master at the block size that
`noise_scale_arcsec` gives on the grid, after subtracting the consensus, so
nebulosity does not read as noise. Its ratio to the consensus's noise,
squared, is the master's depth as a share of the consensus's, and with the
exposure `MasterInfo` records, depth per hour of integration.

**FWHM** is measured on the master at its own scale, before reprojection,
with `PsfField::measure`, and given in arcseconds through its solution, so
resampling does not widen it.

**Alignment.** Stars detected on the reprojected master and on the
consensus are matched within a few pixels; the median and RMS of their
offsets is what the master's solution and the reprojection left
misaligned.

**Coverage** is the mean of `Reprojection::coverage` over the grid.

**Stretch.** One `StretchPlan` per channel, resolved on the consensus and
applied to every matched master, so a brighter or noisier master looks it.

## The report

`comparison_html` writes one self-contained file, like the analysis
report: a metrics table (sortable by depth, depth per hour, FWHM,
alignment and coverage), and per channel a blink view and a split view
between any two masters and the consensus, all with the shared stretch.
Previews are embedded at a size the options cap, so 84 masters stay
portable.

The collaboration's tool also has blind pairwise ranking (people pick the
better of two crops and Bradley–Terry turns the picks into a ranking) and
colour composites. Both are proposed for later: ranking needs people's
votes collected and stored, which a static report cannot do alone, and
composites need a choice of palette per collaboration.

## Order of work

Each phase is one PR with the core, the C ABI and Python together.

1. **Measure.** `compare_masters` with solutions from headers or
   `MasterInput::wcs` only (`MissingSolution::Skip`), the grid, two-pass
   consensus, matching, depth, coverage and FWHM. Tests on synthetic
   masters with known gains, offsets, noise and seeing, plus one on real
   masters behind `#[ignore]`.
2. **Report.** Shared stretches and `comparison_html` with the table,
   blink and split views.
3. **Solve and align.** `MissingSolution::Solve` and the alignment
   residual.
4. **Later, if wanted.** Pairwise ranking, colour composites, and
   processing variants: running each master through RC-Astro tools in
   batches (#501) to compare the same data with and without them.

`compare_masters` adds new outputs and changes none of stacking's,
stretching's or the pipeline's, so it moves no `OUTPUT_VERSION`. A host
that caches a comparison keys it on the inputs' paths and modification
times, the options, and the Seiza version.

## Open questions

- **Which reference?** The proposal builds a consensus. Some comparisons
  want one named master as the reference instead (the deepest, or a
  published one); that would be an option.
- **Channels across filters.** Should an L master be compared with the
  luminance of RGB masters? The proposal leaves it to the host.
- **Large grids.** 84 masters on a 6000 × 4000 grid take the two-pass
  consensus a long time. Comparing a crop around the target is often
  enough; should the grid default to a crop?
- **Where the code lives.** A module in `seiza-analysis`, as proposed, or a
  crate of its own if `seiza-analysis` should not depend on the solver.
