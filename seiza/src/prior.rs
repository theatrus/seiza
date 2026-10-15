//! Solving with what an image already says about itself.
//!
//! Many images arrive knowing roughly where they point: a WCS from an
//! earlier solve (a FITS header, or a PixInsight solution in an XISF file),
//! the mount's RA and Dec, or a pixel scale from the camera's pixel size and
//! the telescope's focal length. [`SolvePrior`] gathers those from header
//! cards, and [`solve_with_prior`] uses them in order of strength:
//!
//! 1. Check the header's WCS with a hinted solve at its own center and
//!    scale. A WCS can be stale — an image cropped or rotated without its
//!    size changing keeps it — so it is checked, not trusted.
//! 2. Otherwise run a hinted solve around the header's position and scale.
//! 3. Otherwise solve blind, trying a range around the header's scale
//!    before the caller's whole range.
//! 4. If nothing solves but the header holds a WCS, return that WCS as it
//!    stands, marked [`SolveSource::HeaderUnchecked`]: a processed image
//!    whose stars no longer match the catalog can still carry a good
//!    solution from before processing.

use crate::blind::{BlindIndex, BlindParams, solve_blind_with_limits};
use crate::catalog::StarCatalog;
use crate::detect::DetectedStar;
use crate::solve::{Solution, SolveHint, SolveLimits, is_stop, solve_with_limits};
use crate::wcs::{Wcs, image_center};

/// What an image's headers say about where it points.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SolvePrior {
    /// A TAN or TAN-SIP solution the headers carry, as written. It may be
    /// stale.
    pub wcs: Option<Wcs>,
    /// Where the image points, (RA, Dec) in degrees: from `RA`/`DEC`,
    /// `OBJCTRA`/`OBJCTDEC`, or `CRVAL1`/`CRVAL2` when the WCS is unusable.
    pub center: Option<(f64, f64)>,
    /// Arcseconds per pixel: from the WCS, `PIXSCALE`/`SCALE`/`SECPIX`, or
    /// `XPIXSZ` (micrometres, binning included, as capture programs write
    /// it) over `FOCALLEN` (millimetres).
    pub scale_arcsec_px: Option<f64>,
}

/// Arcseconds per radian over 1000, for micrometres over millimetres.
const ARCSEC_PER_MICRON_PER_MM: f64 = 206.264_806_247;

impl SolvePrior {
    /// Read a prior from header cards: `number` looks a numeric card up by
    /// keyword, `text` a string card. Values that cannot be right — a
    /// declination past a pole, a scale that is not positive — are left out.
    pub fn from_header(
        number: impl Fn(&str) -> Option<f64>,
        text: impl Fn(&str) -> Option<String>,
    ) -> Self {
        let wcs = Wcs::from_fits_values(&number, &text).filter(|wcs| {
            wcs.crval.0.is_finite()
                && wcs.crval.1.is_finite()
                && plausible_scale(wcs.scale_arcsec_per_px())
        });
        let numeric_center = match (number("RA"), number("DEC")) {
            (Some(ra), Some(dec)) => sky_position(ra, dec),
            _ => None,
        };
        let sexagesimal_center = || {
            let ra = parse_sexagesimal(&text("OBJCTRA").or_else(|| text("RA"))?)? * 15.0;
            let dec = parse_sexagesimal(&text("OBJCTDEC").or_else(|| text("DEC"))?)?;
            sky_position(ra, dec)
        };
        let crval_center = || match (number("CRVAL1"), number("CRVAL2")) {
            (Some(ra), Some(dec)) => sky_position(ra, dec),
            _ => None,
        };
        let center = numeric_center
            .or_else(sexagesimal_center)
            .or_else(crval_center);

        let first = |keys: &[&str]| keys.iter().find_map(|key| number(key));
        let scale = wcs
            .as_ref()
            .map(Wcs::scale_arcsec_per_px)
            .or_else(|| first(&["PIXSCALE", "SCALE", "SECPIX"]))
            .or_else(|| {
                let pixel_um = first(&["XPIXSZ", "PIXSIZE1"])?;
                let focal_mm = first(&["FOCALLEN", "FOCALLENGTH", "FOCAL"])?;
                Some(ARCSEC_PER_MICRON_PER_MM * pixel_um / focal_mm)
            })
            .filter(|scale| plausible_scale(*scale));
        Self {
            wcs,
            center,
            scale_arcsec_px: scale,
        }
    }

    /// Whether the headers said nothing useful.
    pub fn is_empty(&self) -> bool {
        self.wcs.is_none() && self.center.is_none() && self.scale_arcsec_px.is_none()
    }
}

fn plausible_scale(scale: f64) -> bool {
    scale.is_finite() && scale > 1e-3 && scale < 3600.0
}

fn sky_position(ra: f64, dec: f64) -> Option<(f64, f64)> {
    (ra.is_finite() && dec.is_finite() && dec.abs() <= 90.0).then(|| (ra.rem_euclid(360.0), dec))
}

/// `"05 35 17.3"`, `"-05:23:28"` or `"12h30m"` as a decimal number in the
/// first field's unit; `None` for anything else, including a plain decimal
/// number, which the numeric card already gave.
fn parse_sexagesimal(text: &str) -> Option<f64> {
    let text = text.trim();
    let negative = text.starts_with('-');
    let fields: Vec<f64> = text
        .trim_start_matches(['+', '-'])
        .split(|c: char| c.is_whitespace() || matches!(c, ':' | 'h' | 'm' | 's' | 'd' | '°'))
        .filter(|field| !field.is_empty())
        .map(|field| field.trim_end_matches(['\'', '"']).parse::<f64>())
        .collect::<Result<_, _>>()
        .ok()?;
    if !(2..=3).contains(&fields.len())
        || fields
            .iter()
            .any(|field| !field.is_finite() || *field < 0.0)
        || fields[1..].iter().any(|field| *field >= 60.0)
    {
        return None;
    }
    let value = fields[0] + fields[1] / 60.0 + fields.get(2).copied().unwrap_or(0.0) / 3600.0;
    Some(if negative { -value } else { value })
}

/// Where a [`solve_with_prior`] solution came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveSource {
    /// A hinted solve at the header WCS's own center and scale confirmed
    /// it; the solution is the fresh fit.
    HeaderChecked,
    /// A hinted solve around the header's position and scale.
    HeaderHint,
    /// A blind solve.
    Blind,
    /// Nothing solved, so the header's WCS is returned as written, with no
    /// matched stars and a false-match probability of 1.
    HeaderUnchecked,
}

impl SolveSource {
    /// `"header-checked"`, `"header-hint"`, `"blind"` or
    /// `"header-unchecked"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HeaderChecked => "header-checked",
            Self::HeaderHint => "header-hint",
            Self::Blind => "blind",
            Self::HeaderUnchecked => "header-unchecked",
        }
    }
}

/// A solution and where it came from.
#[derive(Clone, Debug)]
pub struct PriorSolution {
    pub solution: Solution,
    pub source: SolveSource,
}

/// How far a header's position may be from the truth: a mount's pointing,
/// or a WCS from before a crop.
const HINT_RADIUS_DEG: f64 = 2.0;
/// How far a scale from pixel size and focal length may be off: reducers,
/// rounded focal lengths and binning written the other way.
const HINT_SCALE_TOLERANCE: f64 = 0.25;
/// How far a header WCS's own scale may be off.
const WCS_SCALE_TOLERANCE: f64 = 0.05;
/// The blind range tried first around a header scale, as a factor each
/// way.
const BLIND_SCALE_FACTOR: f64 = 1.3;

/// Solve `stars`, using `prior` as the module describes. `index` enables
/// the blind stage; without it only the header stages run. `params` sets
/// the blind search and the SIP order of every stage. A stop from `limits`
/// ends the solve at once.
pub fn solve_with_prior(
    stars: &[DetectedStar],
    catalog: &(dyn StarCatalog + Sync),
    index: Option<&BlindIndex>,
    prior: &SolvePrior,
    params: &BlindParams,
    dimensions: (u32, u32),
    limits: &SolveLimits,
) -> Result<PriorSolution, crate::Error> {
    let mut reasons = Vec::new();
    let found = |solution, source| Ok(PriorSolution { solution, source });
    let mut attempt = |label: &str,
                       result: Result<Solution, crate::Error>|
     -> Result<Option<Solution>, crate::Error> {
        match result {
            Ok(solution) => Ok(Some(solution)),
            Err(error) if is_stop(&error) => Err(error),
            Err(error) => {
                reasons.push(format!("{label}: {}", reason(error)));
                Ok(None)
            }
        }
    };

    // 1. The header's own solution, at its own center and scale.
    let wcs_hint = prior.wcs.as_ref().map(|wcs| {
        let middle = image_center(dimensions.0, dimensions.1);
        (
            wcs.pixel_to_world(middle.0, middle.1),
            wcs.scale_arcsec_per_px(),
        )
    });
    if let Some((center, scale)) = wcs_hint {
        let hint = SolveHint {
            center,
            radius_deg: 0.0,
            scale_arcsec_px: scale,
            scale_tolerance: WCS_SCALE_TOLERANCE,
            sip_order: params.sip_order,
        };
        let result = solve_with_limits(stars, catalog, &hint, dimensions, limits);
        if let Some(solution) = attempt("header WCS", result)? {
            return found(solution, SolveSource::HeaderChecked);
        }
    }

    // 2. Around the header's position, at its scale.
    let position = wcs_hint.map(|(center, _)| center).or(prior.center);
    let scale = wcs_hint.map(|(_, scale)| scale).or(prior.scale_arcsec_px);
    if let (Some(center), Some(scale)) = (position, scale) {
        let hint = SolveHint {
            center,
            radius_deg: HINT_RADIUS_DEG,
            scale_arcsec_px: scale,
            scale_tolerance: HINT_SCALE_TOLERANCE,
            sip_order: params.sip_order,
        };
        let result = solve_with_limits(stars, catalog, &hint, dimensions, limits);
        if let Some(solution) = attempt("header position", result)? {
            return found(solution, SolveSource::HeaderHint);
        }
    }

    // 3. Blind, near the header's scale first.
    if let Some(index) = index {
        let mut ranges = Vec::new();
        if let Some(scale) = scale {
            ranges.push((scale / BLIND_SCALE_FACTOR, scale * BLIND_SCALE_FACTOR));
        }
        ranges.push((params.min_scale_arcsec_px, params.max_scale_arcsec_px));
        for (min, max) in ranges {
            let range = BlindParams {
                min_scale_arcsec_px: min,
                max_scale_arcsec_px: max,
                ..params.clone()
            };
            let result = solve_blind_with_limits(stars, catalog, index, &range, dimensions, limits);
            if let Some(solution) = attempt(&format!("blind {min:.3}–{max:.3}\"/px"), result)? {
                return found(solution, SolveSource::Blind);
            }
        }
    }

    // 4. The header's solution as written.
    if let Some(wcs) = &prior.wcs {
        return found(
            Solution {
                wcs: wcs.clone(),
                matched_stars: 0,
                rms_arcsec: 0.0,
                false_match_probability: 1.0,
            },
            SolveSource::HeaderUnchecked,
        );
    }
    Err(crate::Error::Solve(if reasons.is_empty() {
        "the image's headers give no position or scale, and no blind index is open".into()
    } else {
        reasons.join("; ")
    }))
}

fn reason(error: crate::Error) -> String {
    match error {
        crate::Error::Solve(reason) => reason,
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn prior(cards: &[(&str, &str)]) -> SolvePrior {
        let cards: HashMap<&str, &str> = cards.iter().copied().collect();
        SolvePrior::from_header(
            |key| cards.get(key).and_then(|value| value.trim().parse().ok()),
            |key| cards.get(key).map(|value| value.to_string()),
        )
    }

    #[test]
    fn a_header_gives_its_position_and_scale() {
        // Degrees, as N.I.N.A. writes RA and DEC.
        let mount = prior(&[("RA", "83.82"), ("DEC", "-5.39")]);
        assert_eq!(mount.center, Some((83.82, -5.39)));
        assert_eq!(mount.scale_arcsec_px, None);

        // Hours and degrees, as OBJCTRA and OBJCTDEC are written.
        let object = prior(&[("OBJCTRA", "05 35 17.3"), ("OBJCTDEC", "-05 23 28")]);
        let (ra, dec) = object.center.unwrap();
        assert!((ra - 83.822_08).abs() < 1e-4, "{ra}");
        assert!((dec + 5.391_11).abs() < 1e-4, "{dec}");
        let colons = prior(&[("OBJCTRA", "5:35:17.3"), ("OBJCTDEC", "+22:00:52")]);
        assert!((colons.center.unwrap().1 - 22.014_44).abs() < 1e-4);

        // Pixel size over focal length: 3.76 µm at 530 mm.
        let optics = prior(&[("XPIXSZ", "3.76"), ("FOCALLEN", "530")]);
        let scale = optics.scale_arcsec_px.unwrap();
        assert!((scale - 1.4633).abs() < 1e-3, "{scale}");
        assert_eq!(
            prior(&[
                ("PIXSCALE", "1.25"),
                ("XPIXSZ", "3.76"),
                ("FOCALLEN", "530")
            ])
            .scale_arcsec_px,
            Some(1.25)
        );

        // Nothing usable.
        assert!(prior(&[("DEC", "95"), ("RA", "10")]).is_empty());
        assert!(prior(&[("FOCALLEN", "0"), ("XPIXSZ", "3.76")]).is_empty());
        assert!(prior(&[("OBJCTRA", "12 75 00"), ("OBJCTDEC", "10 00")]).is_empty());
    }

    #[test]
    fn a_header_wcs_gives_position_and_scale_too() {
        let with_wcs = prior(&[
            ("CTYPE1", "RA---TAN"),
            ("CTYPE2", "DEC--TAN"),
            ("CRVAL1", "150.5"),
            ("CRVAL2", "33.2"),
            ("CRPIX1", "2000.5"),
            ("CRPIX2", "1500.5"),
            ("CD1_1", "-0.0005"),
            ("CD1_2", "0"),
            ("CD2_1", "0"),
            ("CD2_2", "0.0005"),
        ]);
        let wcs = with_wcs.wcs.as_ref().unwrap();
        assert_eq!(wcs.crval, (150.5, 33.2));
        assert!((with_wcs.scale_arcsec_px.unwrap() - 1.8).abs() < 1e-9);
        // The pointing comes from RA/DEC when given, else CRVAL.
        assert_eq!(with_wcs.center, Some((150.5, 33.2)));

        // An unusable projection still leaves CRVAL as a position.
        let sin = prior(&[
            ("CTYPE1", "RA---SIN"),
            ("CTYPE2", "DEC--SIN"),
            ("CRVAL1", "10"),
            ("CRVAL2", "20"),
        ]);
        assert!(sin.wcs.is_none());
        assert_eq!(sin.center, Some((10.0, 20.0)));
    }

    #[test]
    fn the_strongest_evidence_wins_and_a_header_wcs_is_the_last_resort() {
        use crate::catalog::{CatalogStar, MemoryCatalog};
        let dims = (4000u32, 3000u32);
        let truth =
            Wcs::from_center_scale_rotation((150.5, 33.2), (1999.5, 1499.5), 2.0, 30.0, false);
        let mut seed = 5u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut catalog = Vec::new();
        let mut stars = Vec::new();
        for _ in 0..3000 {
            let ra = 150.5 + (next() - 0.5) * 12.0;
            let dec = 33.2 + (next() - 0.5) * 12.0;
            let mag = (4.0 + next() * 8.0) as f32;
            catalog.push(CatalogStar { ra, dec, mag });
            if let Some((x, y)) = truth.world_to_pixel(ra, dec)
                && x > 0.0
                && y > 0.0
                && x < dims.0 as f64
                && y < dims.1 as f64
            {
                stars.push(DetectedStar {
                    x: x + (next() - 0.5) * 0.6,
                    y: y + (next() - 0.5) * 0.6,
                    flux: 10f64.powf(-0.4 * mag as f64) * 1e6,
                    peak: 0.5,
                    area: 20,
                });
            }
        }
        stars.sort_by(|a, b| b.flux.total_cmp(&a.flux));
        let catalog = MemoryCatalog::new(catalog);
        let params = BlindParams::default();
        let limits = SolveLimits::default();
        let solve = |prior: &SolvePrior, stars: &[DetectedStar]| {
            solve_with_prior(stars, &catalog, None, prior, &params, dims, &limits)
        };

        // A good header WCS is confirmed by a fresh fit.
        let header = SolvePrior {
            wcs: Some(truth.clone()),
            ..SolvePrior::default()
        };
        let checked = solve(&header, &stars).unwrap();
        assert_eq!(checked.source, SolveSource::HeaderChecked);
        assert!(checked.solution.matched_stars > 20);

        // A mount position and an optics scale find the field nearby.
        let mount = SolvePrior {
            center: Some((151.0, 33.6)),
            scale_arcsec_px: Some(2.2),
            ..SolvePrior::default()
        };
        assert_eq!(
            solve(&mount, &stars).unwrap().source,
            SolveSource::HeaderHint
        );

        // With stars that match nothing, the header's WCS stands as written,
        // and says it proves nothing; without one, the solve fails.
        let junk: Vec<DetectedStar> = (0..40)
            .map(|index| DetectedStar {
                x: 37.0 * index as f64 % 4000.0,
                y: 91.0 * index as f64 % 3000.0,
                flux: 1000.0 - index as f64,
                peak: 0.5,
                area: 12,
            })
            .collect();
        let unchecked = solve(&header, &junk).unwrap();
        assert_eq!(unchecked.source, SolveSource::HeaderUnchecked);
        assert_eq!(unchecked.solution.wcs, truth);
        assert_eq!(unchecked.solution.matched_stars, 0);
        assert_eq!(unchecked.solution.false_match_probability, 1.0);
        assert!(solve(&mount, &junk).is_err());
        assert!(solve(&SolvePrior::default(), &stars).is_err());

        // A stop is not a failure to fall back from.
        let stopped = SolveLimits {
            deadline: Some(std::time::Instant::now()),
            cancel: None,
        };
        assert!(matches!(
            solve_with_prior(&junk, &catalog, None, &header, &params, dims, &stopped),
            Err(crate::Error::TimedOut)
        ));
    }
}
