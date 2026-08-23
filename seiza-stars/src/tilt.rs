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
        let mut best_index = 0;
        let mut worst_index = 0;
        for index in 1..sectors.len() {
            let hfr = sectors[index]
                .median_hfr
                .expect("ready sector has a median");
            if hfr < sectors[best_index].median_hfr.unwrap() {
                best_index = index;
            }
            if hfr > sectors[worst_index].median_hfr.unwrap() {
                worst_index = index;
            }
        }
        let best_hfr = sectors[best_index].median_hfr.unwrap();
        let worst_hfr = sectors[worst_index].median_hfr.unwrap();
        (
            Some((worst_hfr - best_hfr) / overall_median_hfr.unwrap() * 100.0),
            Some(sectors[best_index].sector),
            Some(sectors[worst_index].sector),
        )
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
}
