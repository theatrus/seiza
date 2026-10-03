//! Photometric colour calibration of a linear RGB image against Gaia.
//!
//! Each catalogue star with a measured BP − RP colour is measured in the
//! image's three channels by aperture photometry. Its instrumental colours,
//! `−2.5 log10(R / G)` and `−2.5 log10(B / G)`, follow the catalogue colour
//! nearly linearly across ordinary stars, so a robust straight-line fit of
//! each against BP − RP says what the camera records for a star of any
//! colour. Evaluated at the white reference's colour — the Sun's by default —
//! the fits give the channel gains that render such a star neutral.
//!
//! This needs no filter or sensor curves: the stars themselves measure the
//! camera's response. Background neutralization then offsets each channel so
//! the sky's median is the same grey in all three.

use crate::{Error, LinearImage, Result};
use rayon::prelude::*;
use seiza_stats::{median_in_place, robust_sigma_in_place};
use serde::Serialize;

/// The Gaia BP − RP colour of the Sun, a G2V star (Casagrande &
/// VandenBerg 2018): the default white reference.
pub const SOLAR_BP_RP: f32 = 0.82;

/// A catalogue star placed on the image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorReferenceStar {
    /// Zero-based pixel position, from the image's astrometric solution.
    pub x: f64,
    pub y: f64,
    /// Gaia G magnitude.
    pub g: f32,
    /// Gaia BP − RP colour, or `None` for a star that only counts as a
    /// neighbour: one without a colour, or whose colour is unreliable.
    pub bp_rp: Option<f32>,
}

impl ColorReferenceStar {
    /// Place a Gaia star, given at epoch J2016.0, on an image observed at
    /// `epoch` (a Julian year such as 2024.9) through its solution `wcs`.
    /// Proper motions are in mas/yr, with `pmra` already multiplied by
    /// cos(dec). `None` when the star projects behind the tangent plane.
    #[allow(clippy::too_many_arguments)]
    pub fn from_gaia(
        wcs: &seiza::Wcs,
        ra: f64,
        dec: f64,
        pmra: Option<f64>,
        pmdec: Option<f64>,
        epoch: Option<f64>,
        g: f32,
        bp_rp: Option<f32>,
    ) -> Option<Self> {
        let years = epoch.map_or(0.0, |epoch| epoch - 2016.0);
        let (ra, dec) = propagate(ra, dec, pmra.unwrap_or(0.0), pmdec.unwrap_or(0.0), years);
        let (x, y) = wcs.world_to_pixel(ra, dec)?;
        Some(Self { x, y, g, bp_rp })
    }
}

/// Move a position `years` along its proper motion (mas/yr, `pmra` times
/// cos(dec)) on the sphere: along the great circle the motion starts on,
/// which stays right near the poles where stepping RA would not.
fn propagate(ra: f64, dec: f64, pmra: f64, pmdec: f64, years: f64) -> (f64, f64) {
    let (sin_ra, cos_ra) = ra.to_radians().sin_cos();
    let (sin_dec, cos_dec) = dec.to_radians().sin_cos();
    let position = [cos_dec * cos_ra, cos_dec * sin_ra, sin_dec];
    // Unit vectors towards increasing RA (east) and Dec (north).
    let east = [-sin_ra, cos_ra, 0.0];
    let north = [-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec];
    let to_radians = years / 3.6e6 * std::f64::consts::PI / 180.0;
    let (de, dn) = (pmra * to_radians, pmdec * to_radians);
    let moved = [0, 1, 2].map(|axis| position[axis] + de * east[axis] + dn * north[axis]);
    let norm = moved.iter().map(|value| value * value).sum::<f64>().sqrt();
    let [x, y, z] = moved.map(|value| value / norm);
    (
        y.atan2(x).to_degrees().rem_euclid(360.0),
        z.asin().to_degrees(),
    )
}

impl ColorReferenceStar {
    /// Whether this star can calibrate colour rather than only crowd others.
    fn colour(&self) -> Option<f32> {
        self.bp_rp
    }
}

/// A Gaia DR3 source as colour calibration uses it: ICRS position at
/// J2016.0, proper motion in mas/yr (`pmra` times cos(dec)), G magnitude,
/// BP − RP colour, and RUWE.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaiaColorSource {
    pub ra: f64,
    pub dec: f64,
    pub pmra: Option<f64>,
    pub pmdec: Option<f64>,
    pub g: f32,
    pub bp_rp: Option<f32>,
    pub ruwe: Option<f32>,
}

/// Every source on a `width` × `height` image observed at `epoch` (a Julian
/// year; `None` keeps J2016.0), placed through its solution `wcs`. Sources
/// without a BP − RP colour, or with a RUWE of 1.4 or more (likely binaries
/// or blends, whose colours mislead), keep no colour: they cannot calibrate,
/// but still disqualify the calibrators they crowd.
pub fn place_gaia_sources(
    wcs: &seiza::Wcs,
    width: usize,
    height: usize,
    epoch: Option<f64>,
    sources: &[GaiaColorSource],
) -> Vec<ColorReferenceStar> {
    sources
        .iter()
        .filter_map(|source| {
            ColorReferenceStar::from_gaia(
                wcs,
                source.ra,
                source.dec,
                source.pmra,
                source.pmdec,
                epoch,
                source.g,
                source
                    .bp_rp
                    .filter(|colour| colour.is_finite())
                    .filter(|_| source.ruwe.is_none_or(|ruwe| ruwe < 1.4)),
            )
        })
        .filter(|star| {
            star.x >= 0.0 && star.y >= 0.0 && star.x < width as f64 && star.y < height as f64
        })
        .collect()
}

/// Settings for [`calibrate_color`].
#[derive(Clone, Debug, PartialEq)]
pub struct ColorCalibrationOptions {
    /// The BP − RP colour rendered neutral: [`SOLAR_BP_RP`] by default.
    pub white_bp_rp: f32,
    /// Stars bluer or redder than this BP − RP range are left out of the
    /// fits: hot stars are few, and cool dwarfs bend away from a straight
    /// line.
    pub fit_bp_rp_range: (f32, f32),
    /// Aperture radius in pixels; `None` makes it twice the measured FWHM,
    /// enough to hold each channel's whole star even where colour changes
    /// the PSF.
    pub aperture_radius: Option<f64>,
    /// Robust sigma beyond which a star is rejected from a fit.
    pub rejection_sigma: f64,
    /// The fewest stars either fit may rest on.
    pub minimum_stars: usize,
    /// Offset each channel so the sky's median is the same in all three.
    pub neutralize_background: bool,
    /// The faintest G magnitude a calibrator may have. `None` makes it two
    /// magnitudes brighter than the faintest star supplied, so that every
    /// neighbour bright enough to disturb a calibrator's photometry is in
    /// the catalogue and can disqualify it.
    pub calibrator_max_g: Option<f32>,
}

impl Default for ColorCalibrationOptions {
    fn default() -> Self {
        Self {
            white_bp_rp: SOLAR_BP_RP,
            fit_bp_rp_range: (0.0, 1.8),
            aperture_radius: None,
            rejection_sigma: 3.0,
            minimum_stars: 20,
            neutralize_background: true,
            calibrator_max_g: None,
        }
    }
}

/// A straight-line fit of an instrumental colour against BP − RP.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ColorFit {
    /// The instrumental colour, in magnitudes, at BP − RP = 0.
    pub intercept: f64,
    /// Magnitudes of instrumental colour per magnitude of BP − RP.
    pub slope: f64,
    /// Robust scatter of the stars about the line, in magnitudes.
    pub scatter: f64,
    /// Stars the fit kept after rejection.
    pub stars: usize,
}

impl ColorFit {
    /// The fitted instrumental colour of a star of colour `bp_rp`.
    pub fn at(&self, bp_rp: f64) -> f64 {
        self.intercept + self.slope * bp_rp
    }
}

/// One star's measurement, for diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ColorStarMeasurement {
    pub x: f64,
    pub y: f64,
    pub g: f32,
    pub bp_rp: f32,
    /// Background-subtracted fluxes in R, G, B, aperture-corrected.
    pub flux: [f64; 3],
    /// Whether each fit (red, blue) kept the star.
    pub used: [bool; 2],
}

/// The result of [`calibrate_color`]: per-channel gains and offsets, and
/// the evidence for them.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ColorCalibration {
    /// Multipliers for R, G, B; green's is 1.
    pub gains: [f32; 3],
    /// Offsets added after the gains, for background neutralization; zero
    /// without it.
    pub offsets: [f32; 3],
    /// Each channel's sky median before calibration.
    pub background: [f32; 3],
    /// `−2.5 log10(R / G)` against BP − RP.
    pub red_fit: ColorFit,
    /// `−2.5 log10(B / G)` against BP − RP.
    pub blue_fit: ColorFit,
    /// The white reference's BP − RP.
    pub white_bp_rp: f32,
    /// The aperture radius used, in pixels.
    pub aperture_radius: f64,
    /// Per-channel factors that carried aperture fluxes to total fluxes,
    /// relative to green.
    pub aperture_correction: [f64; 3],
    /// Catalogue stars offered, and those measured cleanly in all channels.
    pub stars_offered: usize,
    pub stars_measured: usize,
    /// Every cleanly measured star.
    pub measurements: Vec<ColorStarMeasurement>,
}

impl ColorCalibration {
    /// Scale and offset each channel of a linear RGB image in place.
    /// Non-finite samples are left as they are.
    pub fn apply(&self, image: &mut LinearImage) -> Result<()> {
        if image.channels != 3 {
            return Err(Error::Color(
                "colour calibration needs a three-channel image".into(),
            ));
        }
        image.data.par_chunks_mut(3 * 4096).for_each(|pixels| {
            for pixel in pixels.chunks_exact_mut(3) {
                for (channel, value) in pixel.iter_mut().enumerate() {
                    if value.is_finite() {
                        *value = value.mul_add(self.gains[channel], self.offsets[channel]);
                    }
                }
            }
        });
        Ok(())
    }
}

/// Fit channel gains (and background offsets) that render a star of the
/// white reference's colour neutral, from catalogue stars on the image.
///
/// `image` is linear RGB, background not yet removed. Stars too close to an
/// edge, saturated in any channel, blended with a catalogue neighbour, or
/// too faint to measure in every channel are left out. Fails when fewer than
/// [`ColorCalibrationOptions::minimum_stars`] stars remain for either fit.
pub fn calibrate_color(
    image: &LinearImage,
    stars: &[ColorReferenceStar],
    options: &ColorCalibrationOptions,
) -> Result<ColorCalibration> {
    if image.channels != 3 {
        return Err(Error::Color(
            "colour calibration needs a three-channel image".into(),
        ));
    }
    if !options.white_bp_rp.is_finite()
        || !(options.rejection_sigma.is_finite() && options.rejection_sigma > 0.0)
        || options.fit_bp_rp_range.0 >= options.fit_bp_rp_range.1
        || options
            .aperture_radius
            .is_some_and(|radius| !(radius.is_finite() && radius >= 1.0))
        || options.minimum_stars < 3
    {
        return Err(Error::Color("invalid colour calibration options".into()));
    }
    let ceilings = channel_ceilings(image);
    let fwhm = measure_fwhm(image, stars, &ceilings);
    let aperture_radius = match options.aperture_radius {
        Some(radius) => radius,
        None => {
            2.0 * fwhm.ok_or_else(|| {
                Error::Color("too few clean catalogue stars to measure the image's FWHM".into())
            })?
        }
    }
    .max(2.0);
    // A star's profile width, for telling a saturated flat top from a peak.
    let sigma = fwhm.unwrap_or(aperture_radius / 2.0) / 2.3548;
    let geometry = ApertureGeometry::new(aperture_radius, sigma);
    let faintest = stars.iter().map(|star| star.g).fold(f32::MIN, f32::max);
    let calibrator_max_g = options.calibrator_max_g.unwrap_or(faintest - 2.0);
    // Every supplied star, coloured or not, counts as a neighbour when its
    // light reaches into the aperture: within the aperture radius plus
    // three of its sigmas. The sky annulus takes a median, which a
    // neighbour or two there does not move.
    let isolated = isolated_stars(stars, geometry.reach());
    let measurements = isolated
        .par_iter()
        .filter(|star| star.g <= calibrator_max_g)
        .filter_map(|star| {
            let bp_rp = star.colour()?;
            let (x, y) = refine_centroid(image, star.x, star.y, aperture_radius)?;
            let flux = measure(image, x, y, &geometry, &ceilings)?;
            Some(ColorStarMeasurement {
                x,
                y,
                g: star.g,
                bp_rp,
                flux,
                used: [false; 2],
            })
        })
        .collect::<Vec<_>>();
    let mut measurements = measurements;
    // Colour changes a star's profile: optics, seeing and demosaicing all
    // spread red light further than green, so a fixed aperture misses a
    // different share of each channel. Bright, isolated stars measured in
    // an aperture twice as large say how much each channel misses.
    let aperture_correction =
        aperture_correction(image, stars, &measurements, &geometry, &ceilings);
    for measurement in &mut measurements {
        for (flux, correction) in measurement.flux.iter_mut().zip(aperture_correction) {
            *flux *= correction;
        }
    }
    let in_range =
        |bp_rp: f32| (options.fit_bp_rp_range.0..=options.fit_bp_rp_range.1).contains(&bp_rp);
    let mut fit_channel = |channel: usize, slot: usize| {
        let points = measurements
            .iter()
            .enumerate()
            .filter(|(_, star)| in_range(star.bp_rp))
            .map(|(index, star)| {
                (
                    index,
                    f64::from(star.bp_rp),
                    -2.5 * (star.flux[channel] / star.flux[1]).log10(),
                )
            })
            .collect::<Vec<_>>();
        let (fit, kept) = robust_line(&points, options.rejection_sigma).ok_or_else(|| {
            Error::Color(format!(
                "only {} catalogue stars measured cleanly; a colour fit needs more",
                points.len()
            ))
        })?;
        if fit.stars < options.minimum_stars {
            return Err(Error::Color(format!(
                "only {} stars survived the {} colour fit; at least {} are needed",
                fit.stars,
                if channel == 0 { "red" } else { "blue" },
                options.minimum_stars
            )));
        }
        for index in kept {
            measurements[index].used[slot] = true;
        }
        Ok::<_, Error>(fit)
    };
    let red_fit = fit_channel(0, 0)?;
    let blue_fit = fit_channel(2, 1)?;
    let white = f64::from(options.white_bp_rp);
    // A white-reference star records R/G = 10^(−0.4 r): the gain undoes it.
    let gains = [
        10f64.powf(0.4 * red_fit.at(white)) as f32,
        1.0,
        10f64.powf(0.4 * blue_fit.at(white)) as f32,
    ];
    let background = sky_medians(image);
    let offsets = if options.neutralize_background {
        let target = gains[1] * background[1];
        [
            target - gains[0] * background[0],
            0.0,
            target - gains[2] * background[2],
        ]
    } else {
        [0.0; 3]
    };
    Ok(ColorCalibration {
        gains,
        offsets,
        background,
        red_fit,
        blue_fit,
        white_bp_rp: options.white_bp_rp,
        aperture_radius,
        aperture_correction,
        stars_offered: stars.len(),
        stars_measured: measurements.len(),
        measurements,
    })
}

/// Per-channel factors that scale fluxes in `geometry`'s aperture to the
/// total, from up to 100 of the brightest measured stars with no catalogue
/// neighbour reaching a twice-as-large aperture. Normalized to
/// green, since only the channels' ratios matter; all ones when too few
/// stars qualify.
fn aperture_correction(
    image: &LinearImage,
    stars: &[ColorReferenceStar],
    measurements: &[ColorStarMeasurement],
    geometry: &ApertureGeometry,
    ceilings: &[f32; 3],
) -> [f64; 3] {
    let large = ApertureGeometry::new(geometry.radius * 2.0, geometry.sigma);
    let isolated = isolated_stars(stars, large.reach())
        .into_iter()
        .map(|star| ((star.x.round() as i64, star.y.round() as i64), star.g))
        .collect::<std::collections::HashMap<_, _>>();
    let mut candidates = measurements
        .iter()
        .filter(|measurement| {
            // The measured centroid sits within a pixel of the catalogue
            // position the isolation check used.
            let (x, y) = (measurement.x.round() as i64, measurement.y.round() as i64);
            (-1..=1).any(|dx| (-1..=1).any(|dy| isolated.contains_key(&(x + dx, y + dy))))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.g.total_cmp(&right.g));
    let ratios = candidates
        .par_iter()
        .take(100)
        .filter_map(|measurement| {
            let total = measure(image, measurement.x, measurement.y, &large, ceilings)?;
            let small = measure(image, measurement.x, measurement.y, geometry, ceilings)?;
            Some([
                total[0] / small[0],
                total[1] / small[1],
                total[2] / small[2],
            ])
        })
        .collect::<Vec<_>>();
    if ratios.len() < 10 {
        return [1.0; 3];
    }
    let median = |channel: usize| {
        let mut values = ratios
            .iter()
            .map(|ratio| ratio[channel])
            .collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    let green = median(1);
    [median(0) / green, 1.0, median(2) / green]
}

/// Each channel's largest finite sample: an aperture reaching 85% of it is
/// treated as saturated.
fn channel_ceilings(image: &LinearImage) -> [f32; 3] {
    let mut ceilings = [f32::MIN; 3];
    for (channel, ceiling) in ceilings.iter_mut().enumerate() {
        *ceiling = image
            .data
            .par_iter()
            .skip(channel)
            .step_by(3)
            .copied()
            .filter(|value| value.is_finite())
            .reduce(|| f32::MIN, f32::max);
    }
    ceilings
}

/// The pixels of an aperture and its sky annulus, as offsets from the
/// nearest pixel to the star's centre.
struct ApertureGeometry {
    radius: f64,
    inner: f64,
    outer: f64,
    /// The stars' Gaussian sigma in pixels.
    sigma: f64,
}

impl ApertureGeometry {
    /// How near a neighbour's centre may come before its light, out to three
    /// sigmas, falls inside the aperture.
    fn reach(&self) -> f64 {
        self.radius + 3.0 * self.sigma
    }

    fn new(radius: f64, sigma: f64) -> Self {
        let inner = radius + 3.0;
        Self {
            radius,
            inner,
            outer: inner + radius.max(5.0),
            sigma,
        }
    }
}

/// The catalogue stars with no neighbour within `radius` pixels bright
/// enough to disturb their photometry: one less than 3 magnitudes fainter.
fn isolated_stars(stars: &[ColorReferenceStar], radius: f64) -> Vec<ColorReferenceStar> {
    // Bin by cell so each star checks only nearby ones.
    let cell = radius.max(1.0);
    let key = |star: &ColorReferenceStar| {
        (
            (star.x / cell).floor() as i64,
            (star.y / cell).floor() as i64,
        )
    };
    let mut cells = std::collections::HashMap::<(i64, i64), Vec<usize>>::new();
    for (index, star) in stars.iter().enumerate() {
        cells.entry(key(star)).or_default().push(index);
    }
    stars
        .par_iter()
        .enumerate()
        .filter(|&(index, star)| {
            let (cx, cy) = key(star);
            !(cx - 1..=cx + 1).any(|x| {
                (cy - 1..=cy + 1).any(|y| {
                    cells.get(&(x, y)).is_some_and(|members| {
                        members.iter().any(|&other| {
                            let neighbour = &stars[other];
                            other != index
                                && neighbour.g < star.g + 3.0
                                && (neighbour.x - star.x).hypot(neighbour.y - star.y) < radius
                        })
                    })
                })
            })
        })
        .map(|(_, star)| *star)
        .collect()
}

/// The luminance centroid near `(x, y)`, refined twice; `None` when it
/// wanders more than a radius from the catalogue position, which means the
/// star was not found there.
fn refine_centroid(image: &LinearImage, x: f64, y: f64, radius: f64) -> Option<(f64, f64)> {
    let (mut cx, mut cy) = (x, y);
    for _ in 0..2 {
        let (x0, y0, x1, y1) = window(image, cx, cy, radius + 4.0)?;
        let mut sky = Vec::new();
        for py in y0..y1 {
            for px in x0..x1 {
                let distance = (px as f64 - cx).hypot(py as f64 - cy);
                if distance > radius {
                    sky.push(luminance(image, px, py)?);
                }
            }
        }
        let sky = median_in_place(&mut sky)?;
        let (mut sum, mut sx, mut sy) = (0.0, 0.0, 0.0);
        for py in y0..y1 {
            for px in x0..x1 {
                if (px as f64 - cx).hypot(py as f64 - cy) <= radius {
                    let value = f64::from((luminance(image, px, py)? - sky).max(0.0));
                    sum += value;
                    sx += value * px as f64;
                    sy += value * py as f64;
                }
            }
        }
        if sum <= 0.0 {
            return None;
        }
        (cx, cy) = (sx / sum, sy / sum);
    }
    ((cx - x).hypot(cy - y) <= radius).then_some((cx, cy))
}

/// The pixel bounds `[x0, x1) × [y0, y1)` of a square of half-side `reach`
/// about `(x, y)`, or `None` when it leaves the image.
fn window(image: &LinearImage, x: f64, y: f64, reach: f64) -> Option<(usize, usize, usize, usize)> {
    let x0 = (x - reach).floor();
    let y0 = (y - reach).floor();
    let x1 = (x + reach).ceil() + 1.0;
    let y1 = (y + reach).ceil() + 1.0;
    (x0 >= 0.0 && y0 >= 0.0 && x1 <= image.width as f64 && y1 <= image.height as f64).then_some((
        x0 as usize,
        y0 as usize,
        x1 as usize,
        y1 as usize,
    ))
}

fn luminance(image: &LinearImage, x: usize, y: usize) -> Option<f32> {
    let index = (y * image.width + x) * 3;
    let sum = image.data[index] + image.data[index + 1] + image.data[index + 2];
    sum.is_finite().then_some(sum / 3.0)
}

/// Background-subtracted aperture flux in each channel, or `None` when any
/// sample is missing, the star is saturated in a channel, or it is too
/// faint to measure in one.
///
/// A saturated star is recognized by its flat top: more aperture pixels
/// within 3% of its peak than a star of this width can have. A cut at a
/// fraction of the image's maximum misses one after flat-fielding, which
/// lifts the stars in a vignetted corner above clipped stars in the centre.
fn measure(
    image: &LinearImage,
    x: f64,
    y: f64,
    geometry: &ApertureGeometry,
    ceilings: &[f32; 3],
) -> Option<[f64; 3]> {
    let (x0, y0, x1, y1) = window(image, x, y, geometry.outer)?;
    let mut aperture = [Vec::new(), Vec::new(), Vec::new()];
    let mut annulus = [Vec::new(), Vec::new(), Vec::new()];
    for py in y0..y1 {
        for px in x0..x1 {
            let distance = (px as f64 - x).hypot(py as f64 - y);
            let index = (py * image.width + px) * 3;
            let pixel = &image.data[index..index + 3];
            if distance <= geometry.radius {
                for channel in 0..3 {
                    let value = pixel[channel];
                    if !value.is_finite() || value >= 0.85 * ceilings[channel] {
                        return None;
                    }
                    aperture[channel].push(value);
                }
            } else if (geometry.inner..=geometry.outer).contains(&distance) {
                for channel in 0..3 {
                    if pixel[channel].is_finite() {
                        annulus[channel].push(pixel[channel]);
                    }
                }
            }
        }
    }
    // A Gaussian peak stays within 3% of its top out to 0.247 sigma: about
    // 0.19 sigma² pixels. Three times that, and at least four, is a plateau.
    let plateau_limit = (0.57 * geometry.sigma * geometry.sigma).max(4.0);
    let mut flux = [0.0_f64; 3];
    for channel in 0..3 {
        let samples = &mut annulus[channel];
        if samples.len() < 20 {
            return None;
        }
        let sky = median_in_place(samples)?;
        let sigma = robust_sigma_in_place(samples, sky)?;
        let values = &aperture[channel];
        let peak = values.iter().copied().fold(f32::MIN, f32::max) - sky;
        let near_peak = values
            .iter()
            .filter(|&&value| value - sky >= 0.97 * peak)
            .count();
        if near_peak as f64 > plateau_limit {
            return None;
        }
        flux[channel] = values
            .iter()
            .map(|&value| f64::from(value - sky))
            .sum::<f64>();
        // Ask for a signal-to-noise ratio of at least 20 from the sky noise
        // alone, which keeps each colour's error near 0.05 magnitudes. NaN
        // noise fails this too.
        let noise = f64::from(sigma) * (values.len() as f64).sqrt();
        if flux[channel].partial_cmp(&(20.0 * noise)) != Some(std::cmp::Ordering::Greater) {
            return None;
        }
    }
    Some(flux)
}

/// The median FWHM, from luminance second moments, of up to 200 of the
/// brightest catalogue stars that look clean. A first pass inside a 7 px
/// radius is refined inside three times its result, so wide stars — an
/// oversampled or drizzled stack's — are not truncated.
fn measure_fwhm(
    image: &LinearImage,
    stars: &[ColorReferenceStar],
    ceilings: &[f32; 3],
) -> Option<f64> {
    let mut ordered = stars.to_vec();
    ordered.sort_by(|left, right| left.g.total_cmp(&right.g));
    let ceiling = ceilings.iter().copied().fold(f32::MAX, f32::min);
    let mut radius = 7.0_f64;
    let mut fwhm = None;
    for _ in 0..3 {
        let isolated = isolated_stars(&ordered, radius * 1.5);
        let mut widths = isolated
            .par_iter()
            .take(1000)
            .filter_map(|star| second_moment_fwhm(image, star.x, star.y, radius, ceiling))
            .filter(|width| width.is_finite() && *width > 0.5)
            .collect::<Vec<_>>();
        widths.truncate(200);
        if widths.len() < 10 {
            return fwhm;
        }
        widths.sort_by(f64::total_cmp);
        let width = widths[widths.len() / 2];
        fwhm = Some(width);
        let next = (3.0 * width).max(7.0);
        if (next - radius).abs() < 1.0 {
            break;
        }
        radius = next;
    }
    fwhm
}

/// One star's FWHM from luminance second moments inside `radius`, with the
/// sky from the ring out to 1.4 times it.
fn second_moment_fwhm(
    image: &LinearImage,
    x: f64,
    y: f64,
    radius: f64,
    ceiling: f32,
) -> Option<f64> {
    let (x, y) = refine_centroid(image, x, y, (radius * 0.8).max(4.0))?;
    let reach = radius * 1.4;
    let (x0, y0, x1, y1) = window(image, x, y, reach)?;
    let mut sky = Vec::new();
    for py in y0..y1 {
        for px in x0..x1 {
            let distance = (px as f64 - x).hypot(py as f64 - y);
            if distance > radius && distance <= reach {
                sky.push(luminance(image, px, py)?);
            }
        }
    }
    let sky = median_in_place(&mut sky)?;
    let (mut sum, mut second) = (0.0, 0.0);
    for py in y0..y1 {
        for px in x0..x1 {
            let distance = (px as f64 - x).hypot(py as f64 - y);
            if distance <= radius {
                let value = luminance(image, px, py)?;
                if value >= 0.85 * ceiling {
                    return None;
                }
                let value = f64::from((value - sky).max(0.0));
                sum += value;
                second += value * distance * distance;
            }
        }
    }
    // For a Gaussian, the mean squared radius is 2σ².
    (sum > 0.0).then(|| 2.3548 * (second / sum / 2.0).sqrt())
}

/// Each channel's sky level: the median of the pixels whose luminance lies
/// below the image's median plus one robust sigma, which leaves out stars
/// and most nebulosity.
fn sky_medians(image: &LinearImage) -> [f32; 3] {
    let stride = (image.width * image.height / 2_000_000).max(1);
    let pixels = (0..image.width * image.height)
        .step_by(stride)
        .filter_map(|pixel| {
            let sample = &image.data[pixel * 3..pixel * 3 + 3];
            sample
                .iter()
                .all(|value| value.is_finite())
                .then(|| [sample[0], sample[1], sample[2]])
        })
        .collect::<Vec<_>>();
    let mut luminances = pixels
        .iter()
        .map(|pixel| (pixel[0] + pixel[1] + pixel[2]) / 3.0)
        .collect::<Vec<_>>();
    let Some(median) = median_in_place(&mut luminances) else {
        return [0.0; 3];
    };
    let sigma = robust_sigma_in_place(&mut luminances, median).unwrap_or(0.0);
    let limit = median + sigma;
    let mut channels = [Vec::new(), Vec::new(), Vec::new()];
    for pixel in &pixels {
        if (pixel[0] + pixel[1] + pixel[2]) / 3.0 <= limit {
            for channel in 0..3 {
                channels[channel].push(pixel[channel]);
            }
        }
    }
    channels.map(|mut values| median_in_place(&mut values).unwrap_or(0.0))
}

/// A least-squares line through `(index, x, y)` points, refitted after
/// rejecting points more than `sigma` robust deviations off it until none
/// are. Returns the fit and the indices it kept.
fn robust_line(points: &[(usize, f64, f64)], sigma: f64) -> Option<(ColorFit, Vec<usize>)> {
    let mut kept = points
        .iter()
        .filter(|(_, x, y)| x.is_finite() && y.is_finite())
        .copied()
        .collect::<Vec<_>>();
    let mut last = None;
    for _ in 0..10 {
        if kept.len() < 3 {
            return last;
        }
        let n = kept.len() as f64;
        let (mean_x, mean_y) = kept
            .iter()
            .fold((0.0, 0.0), |(sx, sy), (_, x, y)| (sx + x / n, sy + y / n));
        let (sxx, sxy) = kept.iter().fold((0.0, 0.0), |(sxx, sxy), (_, x, y)| {
            (
                sxx + (x - mean_x).powi(2),
                sxy + (x - mean_x) * (y - mean_y),
            )
        });
        let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
        let intercept = mean_y - slope * mean_x;
        let mut residuals = kept
            .iter()
            .map(|(_, x, y)| (y - intercept - slope * x) as f32)
            .collect::<Vec<_>>();
        let center = median_in_place(&mut residuals.clone())?;
        let scatter = f64::from(robust_sigma_in_place(&mut residuals, center)?);
        let fit = ColorFit {
            intercept,
            slope,
            scatter,
            stars: kept.len(),
        };
        let before = kept.len();
        let fitted = kept.iter().map(|(index, _, _)| *index).collect::<Vec<_>>();
        kept.retain(|(_, x, y)| (y - intercept - slope * x).abs() <= sigma * scatter.max(1e-4));
        // Rejection that keeps trimming a star or two a round has still
        // converged in all that matters: keep the last fit.
        last = Some((fit, fitted));
        if kept.len() == before {
            break;
        }
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic RGB star field whose camera records red at 0.6 and blue at
    /// 1.4 times a neutral response for a solar-coloured star, with redder
    /// stars brighter in red: instrumental R/G falls by 0.5 magnitudes per
    /// magnitude of BP − RP, B/G rises by 0.8.
    fn field() -> (LinearImage, Vec<ColorReferenceStar>) {
        let (width, height) = (1200, 1000);
        let sky = [120.0_f32, 100.0, 90.0];
        let mut data = vec![0.0_f32; width * height * 3];
        for (index, value) in data.iter_mut().enumerate() {
            let noise = ((index * 7919) % 13) as f32 * 0.3;
            *value = sky[index % 3] + noise;
        }
        let mut stars = Vec::new();
        for index in 0..140 {
            let x = 30.0 + ((index * 7919) % 1140) as f64 + ((index % 7) as f64) * 0.13;
            let y = 30.0 + ((index * 6271) % 940) as f64 + ((index % 5) as f64) * 0.17;
            let bp_rp = 0.1 + ((index * 37) % 160) as f32 / 100.0;
            let g = 9.0 + ((index * 13) % 40) as f32 / 10.0;
            let flux = 2.0e5 * 10f64.powf(-0.4 * f64::from(g - 9.0));
            let delta = f64::from(bp_rp - SOLAR_BP_RP);
            // R/G = 0.6 × 10^(0.4 × 0.5 × delta); B/G = 1.4 × 10^(−0.4 × 0.8 × delta).
            let ratios = [
                0.6 * 10f64.powf(0.4 * 0.5 * delta),
                1.0,
                1.4 * 10f64.powf(-0.4 * 0.8 * delta),
            ];
            let sigma = 1.5_f64;
            for py in (y as usize).saturating_sub(10)..(y as usize + 11).min(height) {
                for px in (x as usize).saturating_sub(10)..(x as usize + 11).min(width) {
                    let r2 = (px as f64 - x).powi(2) + (py as f64 - y).powi(2);
                    let profile = (-r2 / (2.0 * sigma * sigma)).exp()
                        / (2.0 * std::f64::consts::PI * sigma * sigma);
                    for channel in 0..3 {
                        data[(py * width + px) * 3 + channel] +=
                            (flux * ratios[channel] * profile) as f32;
                    }
                }
            }
            stars.push(ColorReferenceStar {
                x,
                y,
                g,
                bp_rp: Some(bp_rp),
            });
        }
        (LinearImage::new(width, height, 3, data).unwrap(), stars)
    }

    #[test]
    fn calibration_recovers_the_camera_response_and_neutralizes_the_sky() {
        let (mut image, stars) = field();
        let calibration =
            calibrate_color(&image, &stars, &ColorCalibrationOptions::default()).unwrap();
        assert!(
            calibration.stars_measured >= 40,
            "{}",
            calibration.stars_measured
        );
        assert!(
            (calibration.red_fit.slope + 0.5).abs() < 0.03,
            "{:?}",
            calibration.red_fit
        );
        assert!(
            (calibration.blue_fit.slope - 0.8).abs() < 0.03,
            "{:?}",
            calibration.blue_fit
        );
        assert!(
            (calibration.gains[0] - 1.0 / 0.6).abs() < 0.02,
            "{:?}",
            calibration.gains
        );
        assert!(
            (calibration.gains[2] - 1.0 / 1.4).abs() < 0.02,
            "{:?}",
            calibration.gains
        );

        calibration.apply(&mut image).unwrap();
        let sky = sky_medians(&image);
        assert!(
            (sky[0] - sky[1]).abs() < 0.5 && (sky[2] - sky[1]).abs() < 0.5,
            "{sky:?}"
        );
        // After calibration a solar-coloured star measures neutral.
        let geometry = ApertureGeometry::new(calibration.aperture_radius, 1.5);
        let solar = stars
            .iter()
            .filter(|star| (star.bp_rp.unwrap() - SOLAR_BP_RP).abs() < 0.05)
            .find_map(|star| measure(&image, star.x, star.y, &geometry, &[f32::MAX; 3]))
            .unwrap();
        assert!((solar[0] / solar[1] - 1.0).abs() < 0.03, "{solar:?}");
        assert!((solar[2] / solar[1] - 1.0).abs() < 0.03, "{solar:?}");
    }

    #[test]
    fn a_flat_topped_star_is_saturated_even_below_the_image_maximum() {
        let (mut image, stars) = field();
        let geometry = ApertureGeometry::new(7.0, 1.5);
        let ceilings = [f32::MAX; 3];
        let star = stars[0];
        assert!(measure(&image, star.x, star.y, &geometry, &ceilings).is_some());
        // Clip the star flat, as a sensor's full well does, well below the
        // brightest sample elsewhere in the image.
        let (cx, cy) = (star.x.round() as usize, star.y.round() as usize);
        let (width, _) = (image.width, image.height);
        for py in cy - 10..=cy + 10 {
            for px in cx - 10..=cx + 10 {
                for channel in 0..3 {
                    let value = &mut image.data[(py * width + px) * 3 + channel];
                    *value = value.min(400.0);
                }
            }
        }
        assert!(measure(&image, star.x, star.y, &geometry, &ceilings).is_none());
    }

    #[test]
    fn a_colourless_neighbour_still_disqualifies_a_calibrator() {
        let calibrator = ColorReferenceStar {
            x: 100.0,
            y: 100.0,
            g: 12.0,
            bp_rp: Some(0.8),
        };
        let binary = ColorReferenceStar {
            x: 108.0,
            y: 100.0,
            g: 11.0,
            bp_rp: None,
        };
        assert!(isolated_stars(&[calibrator, binary], 17.0).is_empty());
        assert_eq!(isolated_stars(&[calibrator], 17.0).len(), 1);
    }

    #[test]
    fn proper_motion_stays_on_the_sphere_at_the_pole() {
        // 10"/yr north for a century from 10" short of the north pole
        // carries the star across it: 990" past, on the far meridian.
        let (ra, dec) = propagate(30.0, 90.0 - 10.0 / 3600.0, 0.0, 10_000.0, 100.0);
        assert!((dec - (90.0 - 990.0 / 3600.0)).abs() < 1e-5, "{dec}");
        assert!((ra - 210.0).abs() < 1e-6, "{ra}");
        // Far from the pole it matches the linear step.
        let (ra, dec) = propagate(56.75, 24.1, 1000.0, 0.0, 10.0);
        let expected = 56.75 + 10.0 / 3600.0 / 24.1_f64.to_radians().cos();
        assert!((ra - expected).abs() < 1e-7 && (dec - 24.1).abs() < 1e-6);
    }

    #[test]
    fn too_few_stars_or_a_mono_image_are_refused() {
        let (image, stars) = field();
        assert!(calibrate_color(&image, &stars[..5], &ColorCalibrationOptions::default()).is_err());
        let mono = LinearImage::new(10, 10, 1, vec![1.0; 100]).unwrap();
        assert!(calibrate_color(&mono, &stars, &ColorCalibrationOptions::default()).is_err());
    }

    #[test]
    fn proper_motion_moves_the_star_to_the_observation_epoch() {
        let wcs =
            seiza::Wcs::from_center_scale_rotation((56.75, 24.1), (500.0, 500.0), 1.0, 0.0, false);
        let at_2016 = ColorReferenceStar::from_gaia(
            &wcs,
            56.75,
            24.1,
            Some(0.0),
            Some(1000.0),
            None,
            9.0,
            Some(0.8),
        )
        .unwrap();
        let at_2026 = ColorReferenceStar::from_gaia(
            &wcs,
            56.75,
            24.1,
            Some(0.0),
            Some(1000.0),
            Some(2026.0),
            9.0,
            Some(0.8),
        )
        .unwrap();
        // 1"/yr north for ten years is ten 1" pixels.
        assert!(((at_2016.y - at_2026.y).abs() - 10.0).abs() < 0.01);
        assert!((at_2016.x - at_2026.x).abs() < 0.01);
    }
}
