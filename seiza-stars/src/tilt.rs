//! Sensor tilt and field-curvature analysis from detected stars.
//!
//! The frame divides into a 3×3 grid — the layout ASTAP's HFD inspection and
//! PixInsight's aberration mosaic both use. Each cell aggregates the stars
//! detected inside it; the summary compares corner cells against the center
//! the way ASTAP derives its tilt numbers.
//!
//! Ported from PSF Guard's tilt inspector, where this math lived in the
//! browser and was therefore unreachable from any command line, binding, or
//! report. The numbers are the same; only the address changed.

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
    /// `(worst corner − best corner) / mean HFR`, as a percentage. ASTAP's
    /// tilt indicator: one soft corner against a sharp opposite one.
    pub tilt_percent: Option<f64>,
    /// `mean(corners) / center − 1` as a percentage. Uniformly soft corners
    /// with a sharp center indicate field curvature, not tilt.
    pub curvature_percent: Option<f64>,
    pub worst_corner: Option<Corner>,
    pub best_corner: Option<Corner>,
}

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
}
