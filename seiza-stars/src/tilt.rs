//! Sensor tilt and field-curvature analysis from detected stars.
//!
//! The parallelogram view divides the frame into a 3×3 grid. Each cell
//! aggregates the stars detected inside it, and the summary compares corner
//! cells against the center. The triangle view instead groups stars around
//! three adjustment-screw axes supplied by the caller.
//!
//! The nine-cell analysis was ported from PSF Guard's tilt inspector, where
//! that math lived in the browser and was therefore unreachable from any
//! command line, binding, or report. Triangle sectoring reuses the same
//! measurements while keeping its aggregation and confidence policy here for
//! every host.

/// One star's contribution to the analysis: position plus the point-spread
/// measurements the fitter produced for it.
#[derive(Debug, Clone, Copy)]
pub struct TiltStar {
    /// Centroid X in pixels.
    pub x: f64,
    /// Centroid Y in pixels.
    pub y: f64,
    /// Half-flux radius in pixels.
    pub hfr: f64,
    /// PSF eccentricity, 0 for round.
    pub eccentricity: f64,
    /// PSF elongation direction in radians, when a PSF model was fitted.
    pub theta: Option<f64>,
}

/// Aggregate star statistics for one cell of the 3×3 grid.
#[derive(Debug, Clone, PartialEq)]
pub struct CellStats {
    /// Grid row, 0 at the top.
    pub row: usize,
    /// Grid column, 0 at the left.
    pub col: usize,
    /// Stars whose centroid landed in this cell.
    pub star_count: usize,
    /// Median half-flux radius, `None` with no stars.
    pub median_hfr: Option<f64>,
    /// Median eccentricity, `None` with no stars.
    pub median_eccentricity: Option<f64>,
    /// Mean elongation direction in radians over `[0, π)`. Orientation is
    /// axial — a star elongated at θ looks the same at θ+π — so this is the
    /// circular mean over doubled angles. `None` without fitted PSFs.
    pub mean_theta: Option<f64>,
    /// Agreement of elongation directions, 0 (random) to 1 (aligned).
    /// Aligned directions across a region point at astigmatism or tilt;
    /// random directions are seeing noise.
    pub theta_coherence: f64,
}

/// The four corners of the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Corner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

impl Corner {
    /// The corner's name in kebab-case, as the inspector UI spells it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TopLeft => "top-left",
            Self::TopRight => "top-right",
            Self::BottomLeft => "bottom-left",
            Self::BottomRight => "bottom-right",
        }
    }
}

/// One corner's median HFR, `None` when the corner held no stars.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CornerHfr {
    pub corner: Corner,
    pub hfr: Option<f64>,
}

/// The tilt-versus-curvature verdict over the whole frame.
#[derive(Debug, Clone, PartialEq)]
pub struct TiltSummary {
    /// Median HFR of the center cell.
    pub center_hfr: Option<f64>,
    /// Median HFR of each corner cell.
    pub corners: [CornerHfr; 4],
    /// Median HFR over every cell that has stars.
    pub mean_hfr: Option<f64>,
    /// `(worst corner − best corner) / mean HFR`, as a percentage. The
    /// parallelogram tilt indicator: one soft corner against a sharp opposite
    /// one.
    pub tilt_percent: Option<f64>,
    /// `mean(corners) / center − 1` as a percentage. Uniformly soft corners
    /// with a sharp center indicate field curvature, not tilt.
    pub curvature_percent: Option<f64>,
    pub worst_corner: Option<Corner>,
    pub best_corner: Option<Corner>,
}

/// Minimum number of measured stars a triangle sector needs before it can
/// contribute to a tilt verdict.
pub const TRIANGLE_MINIMUM_STARS_PER_REGION: usize = 3;

/// Radius of the triangle diagram's center disk as a fraction of the distance
/// from the image center to a corner.
pub const TRIANGLE_INNER_RADIUS_FRACTION: f64 = 0.25;

/// Radius of the triangle diagram's annulus as a fraction of the image's
/// shorter dimension. This is the radius of the largest circle that fits in
/// the frame.
pub const TRIANGLE_OUTER_RADIUS_FRACTION: f64 = 0.5;

/// Aggregate HFR statistics for the circular center of a triangle analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriangleCenterStats {
    pub star_count: usize,
    pub median_hfr: Option<f64>,
}

/// Aggregate HFR statistics for one adjustment-screw sector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriangleSectorStats {
    /// Stable, human-facing sector identifier in `1..=3`.
    pub sector: u8,
    /// Sector axis in image-coordinate degrees: zero points toward the top of
    /// the image and positive angles turn clockwise.
    pub axis_angle_degrees: f64,
    pub star_count: usize,
    pub median_hfr: Option<f64>,
}

/// Triangle tilt measurements and the confidence policy used to judge them.
#[derive(Debug, Clone, PartialEq)]
pub struct TriangleTiltSummary {
    /// Normalized first-sector axis in image-coordinate degrees over
    /// `[0, 360)`: zero points up and positive angles turn clockwise.
    pub angle_degrees: f64,
    pub inner_radius_pixels: f64,
    pub outer_radius_pixels: f64,
    pub minimum_stars_per_region: usize,
    /// True when the annulus is non-empty and all three sectors meet
    /// [`TRIANGLE_MINIMUM_STARS_PER_REGION`]. The center does not gate the
    /// triangle because it is contextual rather than part of the differential
    /// tilt verdict.
    pub ready: bool,
    pub center: TriangleCenterStats,
    /// Always ordered by [`TriangleSectorStats::sector`] as 1, 2, 3.
    pub sectors: [TriangleSectorStats; 3],
    /// Median HFR of every usable star selected in the annulus, not a median
    /// of the three sector medians.
    pub overall_median_hfr: Option<f64>,
    /// `100 * (worst sector median - best sector median) /
    /// overall_median_hfr`. Withheld together with best/worst sector until
    /// [`Self::ready`] is true.
    pub tilt_percent: Option<f64>,
    pub best_sector: Option<u8>,
    pub worst_sector: Option<u8>,
}

/// Invalid caller input to [`analyze_triangle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriangleTiltError {
    NonFiniteAngle,
}

impl std::fmt::Display for TriangleTiltError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteAngle => formatter.write_str("triangle angle must be finite"),
        }
    }
}

impl std::error::Error for TriangleTiltError {}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    Some(if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    })
}

fn normalize_degrees(angle_degrees: f64) -> f64 {
    let normalized = angle_degrees.rem_euclid(360.0);
    if normalized == 0.0 { 0.0 } else { normalized }
}

fn triangle_sector_index(star_angle_degrees: f64, first_axis_degrees: f64) -> usize {
    let relative_angle = normalize_degrees(star_angle_degrees - first_axis_degrees);
    (normalize_degrees(relative_angle + 60.0) / 120.0).floor() as usize
}

/// Group measured stars for a three-screw triangle tilt diagram.
///
/// Image coordinates have their origin at the top-left, +X points right, and
/// +Y points down. `angle_degrees` locates sector 1: zero points to the top of
/// the image and positive angles turn clockwise. Sectors 2 and 3 are 120° and
/// 240° clockwise from sector 1.
///
/// The center disk radius is 25% of the center-to-corner distance. The outer
/// radius is half the shorter image dimension, producing a true circular
/// annulus that stays inside the frame. A star exactly on the inner boundary
/// belongs to the annulus; a star exactly on the outer boundary is included.
/// The three sectors are a complete, non-overlapping partition of that
/// annulus. Relative to sector 1, their half-open angular ranges are
/// `[300°, 360°) ∪ [0°, 60°)`, `[60°, 180°)`, and `[180°, 300°)`.
///
/// Only stars with finite in-frame centroids and finite positive HFR are
/// usable. If an unusually elongated or empty frame makes the inner radius at
/// least the outer radius, the returned provenance remains valid but the
/// analysis is not ready and contains no annular measurements.
pub fn analyze_triangle(
    stars: &[TiltStar],
    width: usize,
    height: usize,
    angle_degrees: f64,
) -> Result<TriangleTiltSummary, TriangleTiltError> {
    if !angle_degrees.is_finite() {
        return Err(TriangleTiltError::NonFiniteAngle);
    }

    let angle_degrees = normalize_degrees(angle_degrees);
    let center_x = width as f64 / 2.0;
    let center_y = height as f64 / 2.0;
    let inner_radius_pixels = TRIANGLE_INNER_RADIUS_FRACTION * center_x.hypot(center_y);
    let outer_radius_pixels = TRIANGLE_OUTER_RADIUS_FRACTION * width.min(height) as f64;
    let has_annulus = inner_radius_pixels < outer_radius_pixels;

    let mut center_hfrs = Vec::new();
    let mut annular_hfrs = Vec::new();
    let mut sector_hfrs: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
    for star in stars {
        if !star.x.is_finite()
            || !star.y.is_finite()
            || !star.hfr.is_finite()
            || star.hfr <= 0.0
            || star.x < 0.0
            || star.x >= width as f64
            || star.y < 0.0
            || star.y >= height as f64
        {
            continue;
        }

        let dx = star.x - center_x;
        let dy = star.y - center_y;
        let radius = dx.hypot(dy);
        if radius < inner_radius_pixels {
            center_hfrs.push(star.hfr);
            continue;
        }
        if !has_annulus || radius > outer_radius_pixels {
            continue;
        }

        // atan2(dx, -dy) implements the documented image convention: zero
        // points up and positive angles turn clockwise.
        let star_angle_degrees = normalize_degrees(dx.atan2(-dy).to_degrees());
        let sector_index = triangle_sector_index(star_angle_degrees, angle_degrees);
        annular_hfrs.push(star.hfr);
        sector_hfrs[sector_index].push(star.hfr);
    }

    let center = TriangleCenterStats {
        star_count: center_hfrs.len(),
        median_hfr: median(center_hfrs),
    };
    let sectors = std::array::from_fn(|index| {
        let values = std::mem::take(&mut sector_hfrs[index]);
        TriangleSectorStats {
            sector: index as u8 + 1,
            axis_angle_degrees: normalize_degrees(angle_degrees + index as f64 * 120.0),
            star_count: values.len(),
            median_hfr: median(values),
        }
    });
    let overall_median_hfr = median(annular_hfrs);
    let ready = has_annulus
        && sectors
            .iter()
            .all(|sector| sector.star_count >= TRIANGLE_MINIMUM_STARS_PER_REGION)
        && overall_median_hfr.is_some_and(|hfr| hfr > 0.0);

    let (tilt_percent, best_sector, worst_sector) = if ready {
        triangle_verdict(&sectors, overall_median_hfr)
    } else {
        (None, None, None)
    };

    Ok(TriangleTiltSummary {
        angle_degrees,
        inner_radius_pixels,
        outer_radius_pixels,
        minimum_stars_per_region: TRIANGLE_MINIMUM_STARS_PER_REGION,
        ready,
        center,
        sectors,
        overall_median_hfr,
        tilt_percent,
        best_sector,
        worst_sector,
    })
}

/// The triangle's tilt percentage, best and worst sector, or `None`s unless
/// every sector and the overall median have a value.
fn triangle_verdict(
    sectors: &[TriangleSectorStats; 3],
    overall_median_hfr: Option<f64>,
) -> (Option<f64>, Option<u8>, Option<u8>) {
    let Some(overall) = overall_median_hfr.filter(|hfr| *hfr > 0.0) else {
        return (None, None, None);
    };
    let Some(hfrs) = sectors
        .iter()
        .map(|sector| sector.median_hfr)
        .collect::<Option<Vec<f64>>>()
    else {
        return (None, None, None);
    };
    let mut best_index = 0;
    let mut worst_index = 0;
    for index in 1..hfrs.len() {
        if hfrs[index] < hfrs[best_index] {
            best_index = index;
        }
        if hfrs[index] > hfrs[worst_index] {
            worst_index = index;
        }
    }
    (
        Some((hfrs[worst_index] - hfrs[best_index]) / overall * 100.0),
        Some(sectors[best_index].sector),
        Some(sectors[worst_index].sector),
    )
}

/// Aggregate stars into the 3×3 grid.
pub fn analyze_cells(stars: &[TiltStar], width: usize, height: usize) -> Vec<CellStats> {
    let mut cells = Vec::with_capacity(9);
    for row in 0..3 {
        for col in 0..3 {
            let x0 = col as f64 * width as f64 / 3.0;
            let x1 = (col + 1) as f64 * width as f64 / 3.0;
            let y0 = row as f64 * height as f64 / 3.0;
            let y1 = (row + 1) as f64 * height as f64 / 3.0;
            let cell_stars: Vec<&TiltStar> = stars
                .iter()
                .filter(|star| star.x >= x0 && star.x < x1 && star.y >= y0 && star.y < y1)
                .collect();

            // Circular mean over doubled angles: orientation has period π.
            // Weighted by eccentricity — a round star has no direction to
            // vote.
            let mut sum_cos = 0.0;
            let mut sum_sin = 0.0;
            let mut weight = 0.0;
            for star in &cell_stars {
                if let Some(theta) = star.theta
                    && star.eccentricity > 0.0
                {
                    sum_cos += (2.0 * theta).cos() * star.eccentricity;
                    sum_sin += (2.0 * theta).sin() * star.eccentricity;
                    weight += star.eccentricity;
                }
            }
            let magnitude = sum_cos.hypot(sum_sin);
            let (mean_theta, theta_coherence) = if weight > 0.0 && magnitude > 1e-9 {
                let mut angle = sum_sin.atan2(sum_cos) / 2.0;
                if angle < 0.0 {
                    angle += std::f64::consts::PI;
                }
                (Some(angle), magnitude / weight)
            } else {
                (None, 0.0)
            };

            cells.push(CellStats {
                row,
                col,
                star_count: cell_stars.len(),
                median_hfr: median(cell_stars.iter().map(|star| star.hfr).collect()),
                median_eccentricity: median(
                    cell_stars.iter().map(|star| star.eccentricity).collect(),
                ),
                mean_theta,
                theta_coherence,
            });
        }
    }
    cells
}

const CORNER_CELLS: [(Corner, usize, usize); 4] = [
    (Corner::TopLeft, 0, 0),
    (Corner::TopRight, 0, 2),
    (Corner::BottomLeft, 2, 0),
    (Corner::BottomRight, 2, 2),
];

/// Derive the tilt-versus-curvature summary from the grid.
pub fn tilt_summary(cells: &[CellStats]) -> TiltSummary {
    let cell_at =
        |row: usize, col: usize| cells.iter().find(|cell| cell.row == row && cell.col == col);
    let center_hfr = cell_at(1, 1).and_then(|cell| cell.median_hfr);
    let corners = CORNER_CELLS.map(|(corner, row, col)| CornerHfr {
        corner,
        hfr: cell_at(row, col).and_then(|cell| cell.median_hfr),
    });
    let mean_hfr = median(cells.iter().filter_map(|cell| cell.median_hfr).collect());

    let measured: Vec<(Corner, f64)> = corners
        .iter()
        .filter_map(|corner| corner.hfr.map(|hfr| (corner.corner, hfr)))
        .collect();

    // Tilt is a differential measure; with corners missing it would compare
    // a corner against nothing and report noise.
    let mut tilt_percent = None;
    let mut worst_corner = None;
    let mut best_corner = None;
    if measured.len() == 4
        && let Some(mean) = mean_hfr
        && mean > 0.0
    {
        let mut sorted = measured.clone();
        sorted.sort_by(|a, b| f64::total_cmp(&a.1, &b.1));
        best_corner = Some(sorted[0].0);
        worst_corner = Some(sorted[3].0);
        tilt_percent = Some((sorted[3].1 - sorted[0].1) / mean * 100.0);
    }

    let mut curvature_percent = None;
    if measured.len() == 4
        && let Some(center) = center_hfr
        && center > 0.0
    {
        let corner_mean = measured.iter().map(|(_, hfr)| hfr).sum::<f64>() / measured.len() as f64;
        curvature_percent = Some((corner_mean / center - 1.0) * 100.0);
    }

    TiltSummary {
        center_hfr,
        corners,
        mean_hfr,
        tilt_percent,
        curvature_percent,
        worst_corner,
        best_corner,
    }
}

/// Stars a grid cell or triangle region must hold in one frame before that
/// frame's median counts toward a multi-frame analysis. Fewer give a median
/// that is mostly one star's seeing.
pub const FRAME_MINIMUM_STARS_PER_REGION: usize = 3;

/// One frame's measured stars, for [`analyze_frames`].
#[derive(Debug, Clone, Copy)]
pub struct TiltFrame<'a> {
    /// Frame width in pixels.
    pub width: usize,
    /// Frame height in pixels.
    pub height: usize,
    /// The frame's measured stars, in its own pixel coordinates.
    pub stars: &'a [TiltStar],
}

/// Options for [`analyze_frames`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MultiFrameTiltOptions {
    /// First adjustment-screw axis for a triangle analysis, in the image
    /// degrees [`analyze_triangle`] takes. `None` skips the triangle.
    pub triangle_angle_degrees: Option<f64>,
}

/// How one value spread across the frames that measured it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameSpread {
    /// The median over frames.
    pub median: f64,
    /// The 25th percentile over frames.
    pub lower_quartile: f64,
    /// The 75th percentile over frames.
    pub upper_quartile: f64,
    /// Frames that measured the value.
    pub frame_count: usize,
}

impl FrameSpread {
    fn of(mut values: Vec<f64>) -> Option<Self> {
        values.retain(|value| value.is_finite());
        if values.is_empty() {
            return None;
        }
        values.sort_by(f64::total_cmp);
        let at = |fraction: f64| {
            let position = fraction * (values.len() - 1) as f64;
            let below = position.floor() as usize;
            let above = position.ceil() as usize;
            values[below] + (values[above] - values[below]) * (position - below as f64)
        };
        Some(Self {
            median: at(0.5),
            lower_quartile: at(0.25),
            upper_quartile: at(0.75),
            frame_count: values.len(),
        })
    }
}

/// One frame's own tilt verdicts within a multi-frame analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameTilt {
    /// Stars the frame contributed.
    pub star_count: usize,
    /// The frame's parallelogram verdict, as [`tilt_summary`] gives it.
    pub summary: TiltSummary,
    /// The frame's triangle verdict, when the options asked for one.
    pub triangle: Option<TriangleTiltSummary>,
}

/// One grid cell over many frames.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiFrameCell {
    /// Grid row, 0 at the top.
    pub row: usize,
    /// Grid column, 0 at the left.
    pub col: usize,
    /// Stars in this cell over every frame.
    pub star_count: usize,
    /// The cell's HFR over each frame's median HFR: 1 for a cell as sharp
    /// as its frame, more for a softer one. Only frames with at least
    /// [`FRAME_MINIMUM_STARS_PER_REGION`] stars in the cell count. Dividing
    /// by the frame's own HFR takes out focus and seeing, which change
    /// through a night and move every cell alike.
    pub relative_hfr: Option<FrameSpread>,
    /// [`Self::relative_hfr`]'s median times
    /// [`MultiFrameTilt::median_frame_hfr`]: the cell's HFR, in pixels, on
    /// a typical frame.
    pub median_hfr: Option<f64>,
    /// Median eccentricity of every star in the cell.
    pub median_eccentricity: Option<f64>,
    /// Mean elongation direction of every star in the cell, as
    /// [`CellStats::mean_theta`].
    pub mean_theta: Option<f64>,
    /// Agreement of every star's elongation direction, as
    /// [`CellStats::theta_coherence`].
    pub theta_coherence: f64,
}

/// The triangle analysis over many frames.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiFrameTriangleTilt {
    /// The combined verdict. Star counts sum over every frame. Each median
    /// HFR is the region's HFR over its frame's annulus median, combined
    /// by median over the frames that were [`TriangleTiltSummary::ready`],
    /// then scaled by their median annulus HFR. `ready` is true when at
    /// least one frame was.
    pub summary: TriangleTiltSummary,
    /// Frames whose own triangle was ready.
    pub ready_frame_count: usize,
    /// Each sector's HFR over its frame's annulus median, over ready frames,
    /// ordered as [`TriangleTiltSummary::sectors`].
    pub sector_relative_hfr: [Option<FrameSpread>; 3],
    /// The ready frames' own tilt percentages.
    pub frame_tilt_percent: Option<FrameSpread>,
    /// Ready frames whose worst sector is the combined worst sector.
    pub worst_sector_frame_count: usize,
}

/// Sensor tilt and field curvature measured over many frames.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiFrameTilt {
    /// Frame width in pixels, shared by every frame.
    pub width: usize,
    /// Frame height in pixels, shared by every frame.
    pub height: usize,
    /// Each frame's own verdict, in the order given.
    pub frames: Vec<FrameTilt>,
    /// The median over frames of each frame's median cell HFR
    /// ([`TiltSummary::mean_hfr`]).
    pub median_frame_hfr: Option<f64>,
    /// The nine cells, row by row.
    pub cells: Vec<MultiFrameCell>,
    /// The combined verdict: [`tilt_summary`] over the cells'
    /// [`MultiFrameCell::median_hfr`].
    pub summary: TiltSummary,
    /// The frames' own tilt percentages, over frames that had one. A wide
    /// spread against a small median says the tilt is not settled.
    pub frame_tilt_percent: Option<FrameSpread>,
    /// The frames' own curvature percentages, over frames that had one.
    pub frame_curvature_percent: Option<FrameSpread>,
    /// Frames whose own worst corner is the combined worst corner. Real
    /// tilt puts the same corner worst frame after frame; seeing and
    /// guiding move it around.
    pub worst_corner_frame_count: usize,
    /// The triangle analysis, when the options asked for one.
    pub triangle: Option<MultiFrameTriangleTilt>,
}

/// Why [`analyze_frames`] refused its frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MultiFrameTiltError {
    /// No frames were given.
    NoFrames,
    /// A frame has no pixels.
    EmptyFrame { frame: usize },
    /// A frame's size differs from the first frame's. Tilt is a property
    /// of the sensor, so every frame must come from the same one, unbinned
    /// alike and uncropped.
    MismatchedDimensions {
        frame: usize,
        width: usize,
        height: usize,
        expected_width: usize,
        expected_height: usize,
    },
    /// The triangle angle was refused.
    Triangle(TriangleTiltError),
}

impl std::fmt::Display for MultiFrameTiltError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoFrames => formatter.write_str("tilt analysis needs at least one frame"),
            Self::EmptyFrame { frame } => write!(formatter, "frame {frame} has no pixels"),
            Self::MismatchedDimensions {
                frame,
                width,
                height,
                expected_width,
                expected_height,
            } => write!(
                formatter,
                "frame {frame} is {width}x{height}, but the first frame is \
                 {expected_width}x{expected_height}"
            ),
            Self::Triangle(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for MultiFrameTiltError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Triangle(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TriangleTiltError> for MultiFrameTiltError {
    fn from(error: TriangleTiltError) -> Self {
        Self::Triangle(error)
    }
}

/// Measure sensor tilt and field curvature over many frames of one sensor.
///
/// One frame's verdict carries that frame's seeing, guiding and wind, so a
/// soft corner may be real or may be noise. This analyses each frame as
/// [`analyze_cells`], [`tilt_summary`] and [`analyze_triangle`] do, then
/// combines them. Each region's HFR is divided by its frame's own median
/// HFR before frames are combined, so a focus drift or a change in seeing
/// over the night does not swamp the pattern. Elongation directions pool
/// every star of every frame.
///
/// The frames need not be registered: tilt is fixed to the sensor, and so
/// are the pixel coordinates stars are measured in.
pub fn analyze_frames(
    frames: &[TiltFrame<'_>],
    options: &MultiFrameTiltOptions,
) -> Result<MultiFrameTilt, MultiFrameTiltError> {
    let first = frames.first().ok_or(MultiFrameTiltError::NoFrames)?;
    let (width, height) = (first.width, first.height);
    for (index, frame) in frames.iter().enumerate() {
        if frame.width == 0 || frame.height == 0 {
            return Err(MultiFrameTiltError::EmptyFrame { frame: index });
        }
        if (frame.width, frame.height) != (width, height) {
            return Err(MultiFrameTiltError::MismatchedDimensions {
                frame: index,
                width: frame.width,
                height: frame.height,
                expected_width: width,
                expected_height: height,
            });
        }
    }
    if let Some(angle) = options.triangle_angle_degrees
        && !angle.is_finite()
    {
        return Err(TriangleTiltError::NonFiniteAngle.into());
    }

    let mut frame_tilts = Vec::with_capacity(frames.len());
    let mut relative: [Vec<f64>; 9] = std::array::from_fn(|_| Vec::new());
    for frame in frames {
        let cells = analyze_cells(frame.stars, width, height);
        let summary = tilt_summary(&cells);
        if let Some(scale) = summary.mean_hfr.filter(|hfr| *hfr > 0.0) {
            for cell in &cells {
                if cell.star_count >= FRAME_MINIMUM_STARS_PER_REGION
                    && let Some(hfr) = cell.median_hfr
                {
                    relative[cell.row * 3 + cell.col].push(hfr / scale);
                }
            }
        }
        let triangle = options
            .triangle_angle_degrees
            .map(|angle| analyze_triangle(frame.stars, width, height, angle))
            .transpose()?;
        frame_tilts.push(FrameTilt {
            star_count: frame.stars.len(),
            summary,
            triangle,
        });
    }

    let median_frame_hfr = median(
        frame_tilts
            .iter()
            .filter_map(|frame| frame.summary.mean_hfr)
            .collect(),
    );
    let all_stars: Vec<TiltStar> = frames
        .iter()
        .flat_map(|frame| frame.stars.iter().copied())
        .collect();
    let pooled = analyze_cells(&all_stars, width, height);
    let cells: Vec<MultiFrameCell> = pooled
        .into_iter()
        .map(|cell| {
            let relative_hfr =
                FrameSpread::of(std::mem::take(&mut relative[cell.row * 3 + cell.col]));
            MultiFrameCell {
                row: cell.row,
                col: cell.col,
                star_count: cell.star_count,
                median_hfr: relative_hfr
                    .zip(median_frame_hfr)
                    .map(|(relative, scale)| relative.median * scale),
                relative_hfr,
                median_eccentricity: cell.median_eccentricity,
                mean_theta: cell.mean_theta,
                theta_coherence: cell.theta_coherence,
            }
        })
        .collect();
    let summary = tilt_summary(
        &cells
            .iter()
            .map(|cell| CellStats {
                row: cell.row,
                col: cell.col,
                star_count: cell.star_count,
                median_hfr: cell.median_hfr,
                median_eccentricity: cell.median_eccentricity,
                mean_theta: cell.mean_theta,
                theta_coherence: cell.theta_coherence,
            })
            .collect::<Vec<_>>(),
    );
    let worst_corner_frame_count = summary.worst_corner.map_or(0, |worst| {
        frame_tilts
            .iter()
            .filter(|frame| frame.summary.worst_corner == Some(worst))
            .count()
    });
    let triangle = options
        .triangle_angle_degrees
        .map(|angle| combine_triangles(&frame_tilts, width, height, angle))
        .transpose()?;

    Ok(MultiFrameTilt {
        width,
        height,
        median_frame_hfr,
        frame_tilt_percent: FrameSpread::of(
            frame_tilts
                .iter()
                .filter_map(|frame| frame.summary.tilt_percent)
                .collect(),
        ),
        frame_curvature_percent: FrameSpread::of(
            frame_tilts
                .iter()
                .filter_map(|frame| frame.summary.curvature_percent)
                .collect(),
        ),
        worst_corner_frame_count,
        cells,
        summary,
        frames: frame_tilts,
        triangle,
    })
}

/// Combine the frames' own triangles: each region relative to its frame's
/// annulus median, over the frames that were ready.
fn combine_triangles(
    frames: &[FrameTilt],
    width: usize,
    height: usize,
    angle_degrees: f64,
) -> Result<MultiFrameTriangleTilt, TriangleTiltError> {
    // The geometry depends only on the frame size and angle.
    let template = analyze_triangle(&[], width, height, angle_degrees)?;
    let triangles: Vec<&TriangleTiltSummary> = frames
        .iter()
        .filter_map(|frame| frame.triangle.as_ref())
        .collect();
    let ready: Vec<&TriangleTiltSummary> = triangles
        .iter()
        .copied()
        .filter(|triangle| triangle.ready)
        .collect();
    let relative_to_annulus = |triangle: &TriangleTiltSummary, hfr: Option<f64>| {
        triangle
            .overall_median_hfr
            .filter(|overall| *overall > 0.0)
            .zip(hfr)
            .map(|(overall, hfr)| hfr / overall)
    };
    let scale = median(
        ready
            .iter()
            .filter_map(|triangle| triangle.overall_median_hfr)
            .collect(),
    );
    let sector_relative_hfr: [Option<FrameSpread>; 3] = std::array::from_fn(|index| {
        FrameSpread::of(
            ready
                .iter()
                .filter_map(|triangle| {
                    relative_to_annulus(triangle, triangle.sectors[index].median_hfr)
                })
                .collect(),
        )
    });
    let center_relative = FrameSpread::of(
        ready
            .iter()
            .filter(|triangle| triangle.center.star_count >= FRAME_MINIMUM_STARS_PER_REGION)
            .filter_map(|triangle| relative_to_annulus(triangle, triangle.center.median_hfr))
            .collect(),
    );
    let scaled = |spread: Option<FrameSpread>| {
        spread
            .zip(scale)
            .map(|(spread, scale)| spread.median * scale)
    };
    let sectors = std::array::from_fn(|index| TriangleSectorStats {
        star_count: triangles
            .iter()
            .map(|triangle| triangle.sectors[index].star_count)
            .sum(),
        median_hfr: scaled(sector_relative_hfr[index]),
        ..template.sectors[index]
    });
    let ready_frame_count = ready.len();
    let (tilt_percent, best_sector, worst_sector) = if ready_frame_count > 0 {
        triangle_verdict(&sectors, scale)
    } else {
        (None, None, None)
    };
    let worst_sector_frame_count = worst_sector.map_or(0, |worst| {
        ready
            .iter()
            .filter(|triangle| triangle.worst_sector == Some(worst))
            .count()
    });
    Ok(MultiFrameTriangleTilt {
        summary: TriangleTiltSummary {
            ready: tilt_percent.is_some(),
            center: TriangleCenterStats {
                star_count: triangles
                    .iter()
                    .map(|triangle| triangle.center.star_count)
                    .sum(),
                median_hfr: scaled(center_relative),
            },
            sectors,
            overall_median_hfr: scale,
            tilt_percent,
            best_sector,
            worst_sector,
            ..template
        },
        ready_frame_count,
        sector_relative_hfr,
        frame_tilt_percent: FrameSpread::of(
            ready
                .iter()
                .filter_map(|triangle| triangle.tilt_percent)
                .collect(),
        ),
        worst_sector_frame_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn star(x: f64, y: f64, hfr: f64) -> TiltStar {
        TiltStar {
            x,
            y,
            hfr,
            eccentricity: 0.2,
            theta: None,
        }
    }

    fn triangle_star(
        width: usize,
        height: usize,
        radius: f64,
        angle_degrees: f64,
        hfr: f64,
    ) -> TiltStar {
        let angle = angle_degrees.to_radians();
        star(
            width as f64 / 2.0 + radius * angle.sin(),
            height as f64 / 2.0 - radius * angle.cos(),
            hfr,
        )
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
    }

    /// One star in the middle of every 3×3 cell of a 300×300 frame.
    fn star_per_cell(hfr_at: impl Fn(usize, usize) -> f64) -> Vec<TiltStar> {
        let mut stars = Vec::new();
        for row in 0..3 {
            for col in 0..3 {
                stars.push(star(
                    col as f64 * 100.0 + 50.0,
                    row as f64 * 100.0 + 50.0,
                    hfr_at(row, col),
                ));
            }
        }
        stars
    }

    fn cell_at(cells: &[CellStats], row: usize, col: usize) -> &CellStats {
        cells
            .iter()
            .find(|cell| cell.row == row && cell.col == col)
            .unwrap()
    }

    #[test]
    fn assigns_stars_to_their_region_and_takes_medians() {
        let stars = vec![
            star(10.0, 10.0, 2.0),
            star(20.0, 20.0, 3.0),
            star(30.0, 30.0, 4.0),
            star(290.0, 290.0, 5.0),
        ];
        let cells = analyze_cells(&stars, 300, 300);
        let top_left = cell_at(&cells, 0, 0);
        assert_eq!(top_left.star_count, 3);
        assert_eq!(top_left.median_hfr, Some(3.0));
        let bottom_right = cell_at(&cells, 2, 2);
        assert_eq!(bottom_right.star_count, 1);
        assert_eq!(bottom_right.median_hfr, Some(5.0));
        let empty = cell_at(&cells, 1, 1);
        assert_eq!(empty.star_count, 0);
        assert_eq!(empty.median_hfr, None);
    }

    #[test]
    fn averages_elongation_directions_over_the_axial_period() {
        // Orientations near π and near 0 are almost the same axis; a naive
        // mean would point them perpendicular instead.
        let mut a = star(10.0, 10.0, 2.0);
        a.theta = Some(0.05);
        a.eccentricity = 0.5;
        let mut b = star(20.0, 20.0, 2.0);
        b.theta = Some(std::f64::consts::PI - 0.05);
        b.eccentricity = 0.5;
        let cells = analyze_cells(&[a, b], 300, 300);
        let cell = cell_at(&cells, 0, 0);
        let mean = cell.mean_theta.expect("directions were fitted");
        let axis_error = mean.min(std::f64::consts::PI - mean);
        assert!(axis_error < 0.01, "{mean}");
        assert!(cell.theta_coherence > 0.9, "{}", cell.theta_coherence);
    }

    #[test]
    fn reports_low_coherence_for_random_directions() {
        let stars: Vec<TiltStar> = (0..4)
            .map(|i| {
                let mut s = star(10.0 + i as f64, 10.0 + i as f64, 2.0);
                s.theta = Some(i as f64 * std::f64::consts::PI / 4.0);
                s.eccentricity = 0.5;
                s
            })
            .collect();
        let cells = analyze_cells(&stars, 300, 300);
        assert!(cell_at(&cells, 0, 0).theta_coherence < 0.3);
    }

    #[test]
    fn derives_tilt_from_corner_spread_against_the_mean() {
        // One soft corner: classic tilt signature.
        let cells = analyze_cells(
            &star_per_cell(|row, col| if row == 0 && col == 0 { 3.0 } else { 2.0 }),
            300,
            300,
        );
        let summary = tilt_summary(&cells);
        assert_eq!(summary.worst_corner, Some(Corner::TopLeft));
        assert!(summary.best_corner.is_some());
        let tilt = summary.tilt_percent.unwrap();
        assert!((tilt - 50.0).abs() < 1.0, "{tilt}");
    }

    #[test]
    fn derives_curvature_from_corners_against_the_center() {
        // All corners equally soft, sharp center: curvature, near-zero tilt.
        let cells = analyze_cells(
            &star_per_cell(|row, col| {
                if row == 1 && col == 1 {
                    2.0
                } else if row != 1 && col != 1 {
                    3.0
                } else {
                    2.5
                }
            }),
            300,
            300,
        );
        let summary = tilt_summary(&cells);
        assert!(summary.tilt_percent.unwrap().abs() < 1e-5);
        let curvature = summary.curvature_percent.unwrap();
        assert!((curvature - 50.0).abs() < 1.0, "{curvature}");
    }

    #[test]
    fn refuses_a_tilt_number_with_an_empty_corner() {
        // Tilt is differential; with a corner missing it would compare a
        // corner against nothing and report noise.
        let stars: Vec<TiltStar> = star_per_cell(|_, _| 2.0)
            .into_iter()
            .filter(|candidate| !(candidate.x < 100.0 && candidate.y < 100.0))
            .collect();
        let summary = tilt_summary(&analyze_cells(&stars, 300, 300));
        assert_eq!(summary.tilt_percent, None);
        assert_eq!(summary.worst_corner, None);
    }

    #[test]
    fn triangle_aggregates_center_and_three_screw_sectors() {
        let (width, height) = (400, 400);
        let mut stars = Vec::new();
        for hfr in [1.0, 1.1, 1.2] {
            stars.push(triangle_star(width, height, 20.0, 0.0, hfr));
        }
        for (angle, hfr) in [(0.0, 2.0), (120.0, 3.0), (240.0, 4.0)] {
            for offset in [-5.0, 0.0, 5.0] {
                stars.push(triangle_star(width, height, 120.0, angle + offset, hfr));
            }
        }

        let summary = analyze_triangle(&stars, width, height, 0.0).unwrap();
        assert_eq!(summary.angle_degrees, 0.0);
        assert_close(summary.inner_radius_pixels, 50.0 * 2.0_f64.sqrt());
        assert_eq!(summary.outer_radius_pixels, 200.0);
        assert_eq!(summary.minimum_stars_per_region, 3);
        assert!(summary.ready);
        assert_eq!(summary.center.star_count, 3);
        assert_eq!(summary.center.median_hfr, Some(1.1));
        assert_eq!(
            summary.sectors.map(|sector| (
                sector.sector,
                sector.axis_angle_degrees,
                sector.star_count
            )),
            [(1, 0.0, 3), (2, 120.0, 3), (3, 240.0, 3)]
        );
        assert_eq!(
            summary.sectors.map(|sector| sector.median_hfr),
            [Some(2.0), Some(3.0), Some(4.0)]
        );
        assert_eq!(summary.overall_median_hfr, Some(3.0));
        assert_close(summary.tilt_percent.unwrap(), 200.0 / 3.0);
        assert_eq!(summary.best_sector, Some(1));
        assert_eq!(summary.worst_sector, Some(3));
    }

    #[test]
    fn triangle_rotation_normalizes_axes_and_rejects_non_finite_angles() {
        let (width, height) = (400, 400);
        let mut stars = Vec::new();
        for angle in [330.0, 90.0, 210.0] {
            for _ in 0..3 {
                stars.push(triangle_star(width, height, 120.0, angle, 2.0));
            }
        }

        let summary = analyze_triangle(&stars, width, height, -30.0).unwrap();
        assert_eq!(summary.angle_degrees, 330.0);
        assert_eq!(
            summary.sectors.map(|sector| sector.axis_angle_degrees),
            [330.0, 90.0, 210.0]
        );
        assert!(summary.ready);
        assert_eq!(summary.best_sector, Some(1));
        assert_eq!(summary.worst_sector, Some(1));
        assert_eq!(summary.tilt_percent, Some(0.0));
        assert_eq!(
            analyze_triangle(&stars, width, height, f64::INFINITY),
            Err(TriangleTiltError::NonFiniteAngle)
        );
    }

    #[test]
    fn triangle_sector_boundaries_are_half_open_and_complete() {
        assert_eq!(triangle_sector_index(0.0, 0.0), 0);
        assert_eq!(triangle_sector_index(59.999, 0.0), 0);
        assert_eq!(triangle_sector_index(60.0, 0.0), 1);
        assert_eq!(triangle_sector_index(179.999, 0.0), 1);
        assert_eq!(triangle_sector_index(180.0, 0.0), 2);
        assert_eq!(triangle_sector_index(299.999, 0.0), 2);
        assert_eq!(triangle_sector_index(300.0, 0.0), 0);
        assert_eq!(triangle_sector_index(20.0, 20.0), 0);
        assert_eq!(triangle_sector_index(80.0, 20.0), 1);
    }

    #[test]
    fn triangle_retains_sparse_measurements_but_withholds_the_verdict() {
        let (width, height) = (400, 400);
        let mut stars = Vec::new();
        for (sector, count, hfr) in [(0.0, 3, 2.0), (120.0, 3, 3.0), (240.0, 2, 4.0)] {
            for _ in 0..count {
                stars.push(triangle_star(width, height, 120.0, sector, hfr));
            }
        }

        let summary = analyze_triangle(&stars, width, height, 0.0).unwrap();
        assert!(!summary.ready);
        assert_eq!(summary.sectors[2].star_count, 2);
        assert_eq!(summary.sectors[2].median_hfr, Some(4.0));
        assert!(summary.overall_median_hfr.is_some());
        assert_eq!(summary.tilt_percent, None);
        assert_eq!(summary.best_sector, None);
        assert_eq!(summary.worst_sector, None);
    }

    #[test]
    fn triangle_uses_documented_radial_boundaries_and_ignores_invalid_stars() {
        // 300x400 makes the center-to-corner distance exactly 250 pixels, so
        // the 62.5-pixel inner boundary is exactly representable.
        let (width, height) = (300, 400);
        let inner = 62.5;
        let stars = [
            triangle_star(width, height, inner - 0.01, 0.0, 1.0),
            triangle_star(width, height, inner, 0.0, 2.0),
            triangle_star(width, height, 150.0, 0.0, 3.0),
            triangle_star(width, height, 160.0, 45.0, 99.0),
            star(-1.0, 150.0, 99.0),
            star(150.0, 200.0, f64::NAN),
        ];

        let summary = analyze_triangle(&stars, width, height, 0.0).unwrap();
        assert_eq!(summary.center.star_count, 1);
        assert_eq!(summary.center.median_hfr, Some(1.0));
        assert_eq!(summary.sectors[0].star_count, 2);
        assert_eq!(summary.sectors[0].median_hfr, Some(2.5));
        assert_eq!(summary.overall_median_hfr, Some(2.5));
    }

    #[test]
    fn triangle_overall_median_uses_every_annular_star() {
        let (width, height) = (400, 400);
        let mut stars = Vec::new();
        for (angle, count, hfr) in [(0.0, 3, 1.0), (120.0, 3, 2.0), (240.0, 100, 100.0)] {
            for _ in 0..count {
                stars.push(triangle_star(width, height, 120.0, angle, hfr));
            }
        }

        let summary = analyze_triangle(&stars, width, height, 0.0).unwrap();
        assert!(summary.ready);
        assert_eq!(summary.overall_median_hfr, Some(100.0));
        assert_eq!(summary.tilt_percent, Some(99.0));
    }

    #[test]
    fn triangle_reports_an_empty_annulus_for_an_extreme_aspect_ratio() {
        let (width, height) = (1000, 100);
        let stars = [star(500.0, 50.0, 2.0), star(500.0, 0.0, 3.0)];
        let summary = analyze_triangle(&stars, width, height, 0.0).unwrap();

        assert!(summary.inner_radius_pixels >= summary.outer_radius_pixels);
        assert!(!summary.ready);
        assert_eq!(summary.center.star_count, 2);
        assert_eq!(summary.sectors.map(|sector| sector.star_count), [0, 0, 0]);
        assert_eq!(summary.overall_median_hfr, None);
        assert_eq!(summary.tilt_percent, None);
    }

    /// `per_cell` stars in each 3×3 cell of a 300×300 frame, with HFR from
    /// `hfr_at(row, col, index)`.
    fn frame_stars(per_cell: usize, hfr_at: impl Fn(usize, usize, usize) -> f64) -> Vec<TiltStar> {
        let mut stars = Vec::new();
        for row in 0..3 {
            for col in 0..3 {
                for index in 0..per_cell {
                    let offset = 10.0 + 80.0 * index as f64 / per_cell as f64;
                    stars.push(star(
                        col as f64 * 100.0 + offset,
                        row as f64 * 100.0 + offset,
                        hfr_at(row, col, index),
                    ));
                }
            }
        }
        stars
    }

    fn frames_of(stars: &[Vec<TiltStar>]) -> Vec<TiltFrame<'_>> {
        stars
            .iter()
            .map(|stars| TiltFrame {
                width: 300,
                height: 300,
                stars,
            })
            .collect()
    }

    /// A small deterministic jitter in `[-0.5, 0.5)`.
    fn jitter(seed: usize) -> f64 {
        let mut state = (seed as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        state ^= state >> 31;
        state = state.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        state ^= state >> 29;
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    #[test]
    fn combines_frames_relative_to_each_frames_own_focus() {
        // The top-left corner is 30% softer on every frame. Focus drifts
        // from 2 to 4 px over the night, and each star has seeing noise
        // large enough that single frames often name another corner.
        let stars: Vec<Vec<TiltStar>> = (0..30)
            .map(|frame| {
                let focus = 2.0 + 2.0 * frame as f64 / 29.0;
                frame_stars(5, |row, col, index| {
                    let tilt = if (row, col) == (0, 0) { 1.3 } else { 1.0 };
                    let seed = frame * 1000 + row * 100 + col * 10 + index;
                    focus * tilt * (1.0 + 0.6 * jitter(seed))
                })
            })
            .collect();
        let result = analyze_frames(&frames_of(&stars), &MultiFrameTiltOptions::default()).unwrap();

        assert_eq!(result.frames.len(), 30);
        assert_eq!(result.summary.worst_corner, Some(Corner::TopLeft));
        let tilt = result.summary.tilt_percent.unwrap();
        assert!((20.0..40.0).contains(&tilt), "tilt {tilt}");
        let top_left = &result.cells[0];
        assert_eq!((top_left.row, top_left.col), (0, 0));
        assert_eq!(top_left.star_count, 150);
        let relative = top_left.relative_hfr.unwrap();
        assert_eq!(relative.frame_count, 30);
        assert!((1.2..1.4).contains(&relative.median), "{relative:?}");
        assert!(relative.lower_quartile <= relative.median);
        assert!(relative.median <= relative.upper_quartile);
        let centre = result.cells[4].relative_hfr.unwrap();
        assert!((0.9..1.1).contains(&centre.median), "{centre:?}");
        // On a typical frame the centre sits near the night's median focus.
        let typical = result.median_frame_hfr.unwrap();
        assert!((2.8..3.2).contains(&typical), "typical {typical}");
        assert!(result.worst_corner_frame_count > 15);
        assert!(result.worst_corner_frame_count < 30);
        assert_eq!(result.frame_tilt_percent.unwrap().frame_count, 30);
        assert!(result.triangle.is_none());
    }

    #[test]
    fn sparse_cells_do_not_vote() {
        // Two stars in a cell give a median that is mostly one star's
        // seeing, so the frame's value for that cell is left out.
        let sparse = frame_stars(2, |_, _, _| 2.0);
        let full = frame_stars(4, |_, _, _| 2.0);
        let result = analyze_frames(
            &frames_of(&[sparse, full]),
            &MultiFrameTiltOptions::default(),
        )
        .unwrap();
        for cell in &result.cells {
            assert_eq!(cell.star_count, 6);
            assert_eq!(cell.relative_hfr.unwrap().frame_count, 1);
            assert_close(cell.median_hfr.unwrap(), 2.0);
        }
        assert_close(result.summary.tilt_percent.unwrap(), 0.0);
    }

    #[test]
    fn refuses_frames_from_different_sensors() {
        assert_eq!(
            analyze_frames(&[], &MultiFrameTiltOptions::default()),
            Err(MultiFrameTiltError::NoFrames)
        );
        let stars = frame_stars(3, |_, _, _| 2.0);
        let frames = [
            TiltFrame {
                width: 300,
                height: 300,
                stars: &stars,
            },
            TiltFrame {
                width: 300,
                height: 200,
                stars: &stars,
            },
        ];
        assert_eq!(
            analyze_frames(&frames, &MultiFrameTiltOptions::default()),
            Err(MultiFrameTiltError::MismatchedDimensions {
                frame: 1,
                width: 300,
                height: 200,
                expected_width: 300,
                expected_height: 300,
            })
        );
        let empty = [TiltFrame {
            width: 0,
            height: 300,
            stars: &stars,
        }];
        assert_eq!(
            analyze_frames(&empty, &MultiFrameTiltOptions::default()),
            Err(MultiFrameTiltError::EmptyFrame { frame: 0 })
        );
        let options = MultiFrameTiltOptions {
            triangle_angle_degrees: Some(f64::NAN),
        };
        assert_eq!(
            analyze_frames(&frames[..1], &options),
            Err(MultiFrameTiltError::Triangle(
                TriangleTiltError::NonFiniteAngle
            ))
        );
    }

    #[test]
    fn combines_ready_triangles() {
        // Sector 2 (120° clockwise from up) is 20% softer; focus differs
        // between frames, and one frame is too sparse to be ready.
        let frame = |focus: f64, per_sector: usize| {
            let mut stars = Vec::new();
            for sector in 0..3 {
                for index in 0..per_sector {
                    let angle =
                        sector as f64 * 120.0 - 20.0 + 40.0 * index as f64 / per_sector as f64;
                    let hfr = if sector == 1 { 1.2 * focus } else { focus };
                    stars.push(triangle_star(300, 300, 120.0, angle, hfr));
                }
            }
            for index in 0..4 {
                stars.push(star(148.0 + index as f64, 150.0, focus));
            }
            stars
        };
        let stars = vec![frame(2.0, 5), frame(3.0, 5), frame(4.0, 2)];
        let options = MultiFrameTiltOptions {
            triangle_angle_degrees: Some(0.0),
        };
        let triangle = analyze_frames(&frames_of(&stars), &options)
            .unwrap()
            .triangle
            .unwrap();
        assert_eq!(triangle.ready_frame_count, 2);
        assert!(triangle.summary.ready);
        assert_eq!(triangle.summary.worst_sector, Some(2));
        assert_eq!(triangle.worst_sector_frame_count, 2);
        assert_eq!(triangle.summary.sectors[1].star_count, 12);
        assert_eq!(triangle.summary.center.star_count, 12);
        let relative = triangle.sector_relative_hfr[1].unwrap();
        assert_eq!(relative.frame_count, 2);
        assert_close(relative.median, 1.2);
        // The two ready frames' annulus medians are 2 and 3 px.
        assert_close(triangle.summary.overall_median_hfr.unwrap(), 2.5);
        assert_close(triangle.summary.tilt_percent.unwrap(), 20.0);
        assert_eq!(triangle.frame_tilt_percent.unwrap().frame_count, 2);
    }

    #[test]
    fn frame_spread_interpolates_quartiles() {
        let spread = FrameSpread::of(vec![4.0, 1.0, 3.0, 2.0, f64::NAN]).unwrap();
        assert_eq!(spread.frame_count, 4);
        assert_close(spread.median, 2.5);
        assert_close(spread.lower_quartile, 1.75);
        assert_close(spread.upper_quartile, 3.25);
        assert_eq!(FrameSpread::of(Vec::new()), None);
    }
}
