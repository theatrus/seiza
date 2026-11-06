//! An empirical PSF that varies across the field.
//!
//! A Gaussian or a fixed-β Moffat fitted to each star underestimates the
//! wings, which hold the flux a sharpening step has to move back into the
//! core. [`PsfField`] measures the PSF from the stars themselves instead:
//!
//! 1. It finds isolated stars, by detection and, when a catalogue is given,
//!    by catalogue distance: no neighbour within a few FWHM of the star's
//!    flux aperture, and none brighter than the star inside its stamp.
//!    Fainter neighbours further out are masked out of the stamp. No core
//!    star lies within two stamp radii of a bloomed star.
//! 2. It cuts a stamp around each, 33 px or larger for wide PSFs, centred to
//!    a fraction of a pixel by Lanczos-3 interpolation and normalised to
//!    unit flux. Unsaturated stars give the core. Saturated stars give the
//!    wings, scaled to the core over an overlap annulus where both are
//!    reliable.
//! 3. It fits the core stamps as a mean PSF plus a few principal
//!    components, whose coefficients vary as polynomials in `x` and `y`:
//!    the PSF-field method of the survey pipelines. The degree, 1 to 3, is
//!    the one that best predicts stars held out of the fit. Each channel of
//!    an RGB image has its own field.
//!
//! [`PsfField::stamp_at`] returns the local PSF anywhere in the image, and
//! [`PsfField::shape_at`] its FWHM, ellipticity and Moffat β. Pixels count
//! from 0 with centres on whole numbers; a stamp's centre is its middle
//! pixel.

mod candidates;
mod gaussian;
mod model;
mod shape;
mod stamps;
mod star_fit;

use std::borrow::Cow;
use std::fmt;
use std::str::FromStr;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use candidates::{Mesh, PeakGrid, adaptive_moments, clip_level, find_peaks, median_f64, smooth};
use model::{ChannelModel, MAX_DEGREE, StarStamp};
pub use shape::{MAXIMUM_BETA, PsfShape};
use stamps::{LANCZOS_REACH, Plane, aperture_sum, radii};
pub use star_fit::{FittedStar, StarFit, StarFitError, StarFitOptions, StarPositions};

/// The stamp side [`PsfField`] uses unless the PSF is wide: 33 px.
pub const DEFAULT_STAMP_SIZE: usize = 33;

/// The fewest stars a single global PSF is fitted from, when
/// [`SparseFieldPolicy::Global`] lets [`PsfField`] fall back to one.
pub const GLOBAL_MINIMUM_STARS: usize = 3;

/// Peaks measured as candidates at most, brightest first.
const CANDIDATE_LIMIT: usize = 20_000;

/// Saturated stars measured for the wings at most, brightest first.
const WING_STAR_LIMIT: usize = 40;

/// The radius in pixels of a saturated disc from which a star counts as
/// bloomed, and keeps core stars twice the isolation radius away.
const BLOOM_RADIUS: usize = 3;

/// Saturated stars a channel needs before its wings replace the core's.
const WING_MINIMUM_STARS: usize = 3;

/// Cells across and down that core stars are drawn from in turn, so the
/// polynomials see the whole field and not only its densest part.
const SELECTION_GRID: usize = 6;

/// What [`PsfField::measure`] does when fewer stars than
/// [`PsfFieldOptions::minimum_stars`] are usable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SparseFieldPolicy {
    /// Refuse with [`PsfFieldError::TooFewStars`].
    #[default]
    Refuse,
    /// Fit one PSF for the whole image from the stars there are (at least
    /// [`GLOBAL_MINIMUM_STARS`]), and report
    /// [`PsfFieldWarning::GlobalPsf`].
    Global,
}

impl SparseFieldPolicy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Refuse => "refuse",
            Self::Global => "global",
        }
    }
}

impl fmt::Display for SparseFieldPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for SparseFieldPolicy {
    type Err = PsfFieldError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "refuse" => Ok(Self::Refuse),
            "global" => Ok(Self::Global),
            other => Err(PsfFieldError::InvalidOptions(format!(
                "unknown sparse-field policy {other:?}; expected refuse or global"
            ))),
        }
    }
}

/// A catalogue star at its position in the image, for isolating the stars a
/// [`PsfField`] is built from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CatalogueStar {
    /// Column, in pixels from 0.
    pub x: f64,
    /// Row, in pixels from 0.
    pub y: f64,
    /// Brightness as a magnitude in any one band (Gaia G, say): only
    /// differences between stars are used.
    pub magnitude: f64,
}

/// Settings for [`PsfField::measure`].
#[derive(Clone, Debug, PartialEq)]
pub struct PsfFieldOptions {
    /// Stamp side in pixels, odd. `None` uses [`DEFAULT_STAMP_SIZE`], or
    /// seven FWHM when the PSF is wider than that allows.
    pub stamp_size: Option<usize>,
    /// Principal components beyond the mean PSF. The field uses fewer when
    /// it has fewer than ten stars for each, and a varying field needs at
    /// least ten stars. Zero asks for one PSF for the whole image.
    pub components: usize,
    /// The fewest usable stars a field is fitted from, and never fewer than
    /// ten for a varying one. With fewer, [`Self::sparse`] decides.
    pub minimum_stars: usize,
    /// The most stars the core is fitted from, drawn evenly over the field.
    pub maximum_stars: usize,
    /// The lowest signal-to-noise ratio, in a 2.5 FWHM aperture, of a star
    /// that gives the core.
    pub minimum_snr: f64,
    /// Detection threshold, in noise σ of the lightly smoothed image.
    pub detection_sigma: f64,
    /// The peak level, in the image's units, from which a star counts as
    /// saturated and gives wings rather than core. `None` finds it from the
    /// brightest stars' common clip level, and counts stars from 80% of it.
    pub saturation: Option<f32>,
    /// How much fainter than a star, in magnitudes, a neighbour near its
    /// core may be without disqualifying it. A neighbour within
    /// `isolation_magnitudes` of it disqualifies it when it lies within
    /// 1.5 FWHM of its flux aperture, and one brighter than the star
    /// anywhere in its stamp does too. Every other detected neighbour, and
    /// every catalogue neighbour within `isolation_magnitudes`, is masked
    /// out of its stamp.
    pub isolation_magnitudes: f64,
    /// What to do with fewer than [`Self::minimum_stars`] usable stars.
    pub sparse: SparseFieldPolicy,
    /// The degree, 1 to 3, of the polynomials the components' coefficients
    /// follow across the field. `None` chooses it by how well each degree
    /// predicts stars held out of the fit: a plane or a quadratic for most
    /// fields, a cubic for a tilted sensor whose best focus crosses the
    /// frame.
    pub degree: Option<usize>,
}

impl Default for PsfFieldOptions {
    fn default() -> Self {
        Self {
            stamp_size: None,
            components: 3,
            minimum_stars: 30,
            maximum_stars: 400,
            minimum_snr: 30.0,
            detection_sigma: 5.0,
            saturation: None,
            isolation_magnitudes: 5.0,
            sparse: SparseFieldPolicy::Refuse,
            degree: None,
        }
    }
}

impl PsfFieldOptions {
    /// Check that every setting lies in its supported range.
    pub fn validate(&self) -> Result<(), PsfFieldError> {
        let invalid = |message: &str| Err(PsfFieldError::InvalidOptions(message.into()));
        if let Some(size) = self.stamp_size
            && (size % 2 == 0 || !(9..=255).contains(&size))
        {
            return invalid("stamp size must be odd and in 9..=255");
        }
        if self.components > 10 {
            return invalid("components must be at most 10");
        }
        if self.minimum_stars < GLOBAL_MINIMUM_STARS {
            return invalid("minimum stars must be at least 3");
        }
        if self.maximum_stars < self.minimum_stars {
            return invalid("maximum stars must be at least the minimum");
        }
        if !self.minimum_snr.is_finite() || self.minimum_snr <= 0.0 {
            return invalid("minimum SNR must be finite and positive");
        }
        if !self.detection_sigma.is_finite() || self.detection_sigma <= 0.0 {
            return invalid("detection sigma must be finite and positive");
        }
        if let Some(level) = self.saturation
            && !(level.is_finite() && level > 0.0)
        {
            return invalid("saturation must be finite and positive");
        }
        if !self.isolation_magnitudes.is_finite() || self.isolation_magnitudes < 0.0 {
            return invalid("isolation magnitudes must be finite and not negative");
        }
        if let Some(degree) = self.degree
            && !(1..=MAX_DEGREE).contains(&degree)
        {
            return invalid("degree must be 1, 2 or 3");
        }
        Ok(())
    }
}

/// Why [`PsfField::measure`] refused.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum PsfFieldError {
    /// The image's dimensions, channel count or sample count are wrong.
    InvalidImage(String),
    /// A setting is out of range.
    InvalidOptions(String),
    /// Too few usable stars for a varying field: `usable` were found and
    /// `required` are needed, [`PsfFieldOptions::minimum_stars`] and at least
    /// ten. With [`SparseFieldPolicy::Global`], `required` is
    /// [`GLOBAL_MINIMUM_STARS`].
    TooFewStars {
        /// Stars that passed selection.
        usable: usize,
        /// Stars the field needed.
        required: usize,
    },
}

impl fmt::Display for PsfFieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidImage(message) => write!(f, "invalid PSF field image: {message}"),
            Self::InvalidOptions(message) => write!(f, "invalid PSF field options: {message}"),
            Self::TooFewStars { usable, required } => write!(
                f,
                "too few isolated, unsaturated stars for a PSF field: {usable} usable, {required} needed"
            ),
        }
    }
}

impl std::error::Error for PsfFieldError {}

/// Something a caller should know about a [`PsfField`] that was fitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PsfFieldWarning {
    /// Too few stars for a varying field: one PSF serves the whole image.
    GlobalPsf {
        /// Stars the global PSF was fitted from.
        usable: usize,
        /// Stars a varying field needs.
        required: usize,
    },
}

impl fmt::Display for PsfFieldWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GlobalPsf { usable, required } => write!(
                f,
                "one global PSF from {usable} stars: a varying field needs {required}"
            ),
        }
    }
}

/// What a star gave the field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PsfStarRole {
    /// An unsaturated star: the core and its variation.
    Core,
    /// A saturated star: the wings beyond the core.
    Wing,
}

impl PsfStarRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::Core => "core",
            Self::Wing => "wing",
        }
    }
}

impl fmt::Display for PsfStarRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PsfStarRole {
    type Err = PsfFieldError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "core" => Ok(Self::Core),
            "wing" => Ok(Self::Wing),
            other => Err(PsfFieldError::InvalidOptions(format!(
                "unknown PSF star role {other:?}; expected core or wing"
            ))),
        }
    }
}

/// A star a [`PsfField`] was fitted from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PsfStar {
    /// Centre column, in pixels from 0, from adaptive moments.
    pub x: f64,
    /// Centre row, in pixels from 0.
    pub y: f64,
    /// Light above the background in a 2.5 FWHM aperture, summed over
    /// channels.
    pub flux: f64,
    /// Signal-to-noise ratio in that aperture, against the background noise.
    pub snr: f64,
    /// Gaussian-equivalent FWHM from the star's moments, in pixels.
    pub fwhm: f64,
    /// Whether it gave the core or the wings.
    pub role: PsfStarRole,
}

/// How well one channel's field fits its stars.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PsfChannelFit {
    /// The median over core stars of the χ² per stamp pixel, against each
    /// star's background noise. Bright stars' photon noise is not in the
    /// denominator, so values above 1 are expected for them.
    pub reduced_chi_squared: f64,
    /// The median over core stars of the RMS residual within 1.5 FWHM, as
    /// a fraction of the model's RMS there.
    pub relative_residual: f64,
    /// The degree of the polynomials the coefficients follow across the
    /// field: 0 for a global PSF, else 1 to 3.
    pub degree: usize,
    /// Saturated stars whose wings were added to the core model's outer
    /// part.
    pub wing_stars: usize,
    /// The radii in pixels over which the core model hands over to the
    /// wings, when saturated stars gave any.
    pub wing_overlap: Option<(f64, f64)>,
}

/// One PSF stamp: `size × size` pixels per channel, interleaved like the
/// image, each channel summing to one, centred on the middle pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct PsfStamp {
    /// Side in pixels, odd.
    pub size: usize,
    /// Channels per pixel, one or three.
    pub channels: usize,
    /// `size × size × channels` samples, row by row, channels interleaved.
    pub data: Vec<f32>,
}

impl PsfStamp {
    /// A pixel-integrated round Gaussian of `fwhm` pixels: one channel,
    /// `size × size`, centred on the middle pixel and summing to one. It is
    /// the target PSF stars are rendered at and deconvolution aims at.
    ///
    /// # Panics
    ///
    /// If `fwhm` is not finite and positive, or `size` is zero.
    pub fn gaussian(fwhm: f64, size: usize) -> Self {
        assert!(
            fwhm.is_finite() && fwhm > 0.0,
            "FWHM must be finite and positive"
        );
        assert!(size > 0, "size must not be zero");
        let half = (size / 2) as isize;
        let profile = gaussian::pixel_profile(
            0.0,
            -half,
            size as isize - 1 - half,
            fwhm / gaussian::FWHM_PER_SIGMA,
        );
        let mut data: Vec<f32> = (0..size * size)
            .map(|index| (profile[index % size] * profile[index / size]) as f32)
            .collect();
        let total: f32 = data.iter().sum();
        data.iter_mut().for_each(|value| *value /= total);
        Self {
            size,
            channels: 1,
            data,
        }
    }

    /// One channel's `size × size` samples, row by row.
    ///
    /// # Panics
    ///
    /// If `channel` is not below [`Self::channels`].
    pub fn plane(&self, channel: usize) -> Vec<f32> {
        assert!(channel < self.channels, "channel {channel} out of range");
        self.data
            .iter()
            .skip(channel)
            .step_by(self.channels)
            .copied()
            .collect()
    }

    /// The shape of one channel's PSF, or `None` when the channel is out of
    /// range or the profile cannot be fitted.
    pub fn shape(&self, channel: usize) -> Option<PsfShape> {
        if channel >= self.channels {
            return None;
        }
        let plane: Vec<f64> = self.plane(channel).into_iter().map(f64::from).collect();
        shape::fit_shape(&plane, self.size)
    }
}

/// An empirical PSF that varies across an image, measured from its stars.
#[derive(Clone, Debug)]
pub struct PsfField {
    width: usize,
    height: usize,
    stamp_size: usize,
    models: Vec<ChannelModel>,
    fits: Vec<PsfChannelFit>,
    stars: Vec<PsfStar>,
    warnings: Vec<PsfFieldWarning>,
    global: bool,
}

/// A measured candidate star.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    x: f64,
    y: f64,
    flux: f64,
    snr: f64,
    fwhm: f64,
    saturated: bool,
    /// The smoothed peak value, which neighbour tests compare.
    brightness: f32,
}

impl PsfField {
    /// Measure the PSF field of a linear image: `width × height` pixels of
    /// one or three interleaved channels. Non-finite samples, such as a
    /// registered stack's blank borders, are never part of a stamp.
    pub fn measure(
        data: &[f32],
        width: usize,
        height: usize,
        channels: usize,
        options: &PsfFieldOptions,
    ) -> Result<Self, PsfFieldError> {
        Self::measure_with_catalogue(data, width, height, channels, &[], options)
    }

    /// [`Self::measure`], with catalogue stars at their image positions.
    /// A star then gives a stamp only when a catalogue star lies within the
    /// larger of 2 px and one FWHM of it, and catalogue neighbours count as
    /// detected ones do (see [`PsfFieldOptions::isolation_magnitudes`]),
    /// with brightness from their magnitudes. Faint catalogue neighbours that
    /// detection misses are masked or disqualify a star this way.
    pub fn measure_with_catalogue(
        data: &[f32],
        width: usize,
        height: usize,
        channels: usize,
        catalogue: &[CatalogueStar],
        options: &PsfFieldOptions,
    ) -> Result<Self, PsfFieldError> {
        options.validate()?;
        if width == 0 || height == 0 || !matches!(channels, 1 | 3) {
            return Err(PsfFieldError::InvalidImage(
                "dimensions must be non-zero and channel count must be 1 or 3".into(),
            ));
        }
        let expected = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(channels))
            .ok_or_else(|| PsfFieldError::InvalidImage("image dimensions overflow".into()))?;
        if data.len() != expected {
            return Err(PsfFieldError::InvalidImage(format!(
                "pixel buffer has {} samples; expected {expected}",
                data.len()
            )));
        }

        // Detection runs on the channels' sum, which is non-finite wherever
        // any channel is.
        let luma: Cow<'_, [f32]> = if channels == 1 {
            Cow::Borrowed(data)
        } else {
            Cow::Owned(
                data.par_chunks(channels)
                    .map(|pixel| pixel.iter().sum())
                    .collect(),
            )
        };
        let luma_plane = Plane {
            data: &luma,
            width,
            height,
            channels: 1,
            channel: 0,
        };
        let planes: Vec<Plane<'_>> = (0..channels)
            .map(|channel| Plane {
                data,
                width,
                height,
                channels,
                channel,
            })
            .collect();
        let mesh = Mesh::measure(&luma, width, height);
        let smoothed = {
            let residual = mesh.residual(&luma);
            smooth(&residual, width, height)
        };
        let noise = Mesh::measure(&smoothed, width, height);
        let peaks = find_peaks(
            &smoothed,
            width,
            height,
            &noise,
            options.detection_sigma as f32,
        );
        drop(smoothed);
        // A varying field needs ten stars for each component, so at least
        // ten; one PSF everywhere, asked for with no components, needs only
        // the minimum.
        let required = if options.components > 0 {
            options.minimum_stars.max(10)
        } else {
            options.minimum_stars
        };
        let too_few = |usable: usize| PsfFieldError::TooFewStars {
            usable,
            required: match options.sparse {
                SparseFieldPolicy::Refuse => required,
                SparseFieldPolicy::Global => GLOBAL_MINIMUM_STARS,
            },
        };
        if peaks.is_empty() {
            return Err(too_few(0));
        }

        // Each channel's saturation level.
        let raw_peak = |plane: &Plane<'_>, x: usize, y: usize| -> f32 {
            let mut peak = f32::NEG_INFINITY;
            for py in y as isize - 1..=y as isize + 1 {
                for px in x as isize - 1..=x as isize + 1 {
                    let value = plane.at(px, py);
                    if value.is_finite() {
                        peak = peak.max(value as f32);
                    }
                }
            }
            peak
        };
        let saturation: Vec<f32> = planes
            .iter()
            .map(|plane| {
                options.saturation.unwrap_or_else(|| {
                    clip_level(
                        peaks
                            .iter()
                            .take(300)
                            .map(|peak| raw_peak(plane, peak.x, peak.y))
                            .collect(),
                    )
                    .unwrap_or(f32::INFINITY)
                })
            })
            .collect();
        let is_saturated = |x: usize, y: usize| {
            planes
                .iter()
                .zip(&saturation)
                .any(|(plane, &level)| raw_peak(plane, x, y) >= level)
        };

        // A first look at the PSF's width, from the brightest unsaturated
        // stars, sets the stamp and the measuring radii.
        let probe: Vec<f64> = peaks
            .iter()
            .filter(|peak| {
                peak.x >= 16 && peak.y >= 16 && peak.x + 16 < width && peak.y + 16 < height
            })
            .filter(|peak| !is_saturated(peak.x, peak.y))
            .take(400)
            .collect::<Vec<_>>()
            .par_iter()
            .filter_map(|peak| {
                let background = f64::from(mesh.background_at(peak.x as f64, peak.y as f64));
                adaptive_moments(
                    |x, y| luma_plane.at(x, y),
                    peak.x as f64,
                    peak.y as f64,
                    12.0,
                    background,
                    1.5,
                )
                .map(|moments| moments.fwhm())
                .filter(|fwhm| fwhm.is_finite() && *fwhm > 0.5)
            })
            .collect();
        if probe.is_empty() {
            return Err(too_few(0));
        }
        let fwhm0 = median_f64(&mut probe.clone());
        let stamp_size = options.stamp_size.unwrap_or_else(|| {
            let wide = (7.0 * fwhm0).ceil() as usize | 1;
            wide.clamp(DEFAULT_STAMP_SIZE, 255)
        });
        let half = stamp_size / 2;
        let reach = (half + LANCZOS_REACH + 1).max(half + 6) as isize;
        let aperture = (2.5 * fwhm0).max(3.0);
        let moment_radius = (4.0 * fwhm0).clamp(4.0, half as f64);

        // Measure the candidates.
        let candidates: Vec<Candidate> = peaks
            .par_iter()
            .take(CANDIDATE_LIMIT)
            .filter_map(|peak| {
                let (px, py) = (peak.x as isize, peak.y as isize);
                if !luma_plane.window_is_finite(px, py, reach) {
                    return None;
                }
                let (background, sigma) = luma_plane.ring_background(
                    px,
                    py,
                    half as isize - 1,
                    half as isize,
                    |_, _| true,
                )?;
                let moments = adaptive_moments(
                    |x, y| luma_plane.at(x, y),
                    peak.x as f64,
                    peak.y as f64,
                    moment_radius,
                    background,
                    (fwhm0 / 2.354_820_045).max(0.8),
                )?;
                if (moments.x - peak.x as f64).abs() > 2.0
                    || (moments.y - peak.y as f64).abs() > 2.0
                {
                    return None;
                }
                let (cx, cy) = (moments.x.round() as isize, moments.y.round() as isize);
                let reach_ap = aperture.ceil() as isize;
                let (mut flux, mut count) = (0.0, 0usize);
                for y in cy - reach_ap..=cy + reach_ap {
                    for x in cx - reach_ap..=cx + reach_ap {
                        let (dx, dy) = (x as f64 - moments.x, y as f64 - moments.y);
                        if dx * dx + dy * dy <= aperture * aperture {
                            flux += luma_plane.at(x, y) - background;
                            count += 1;
                        }
                    }
                }
                let snr = if sigma > 0.0 {
                    flux / (sigma * (count as f64).sqrt())
                } else {
                    f64::INFINITY
                };
                Some(Candidate {
                    x: moments.x,
                    y: moments.y,
                    flux,
                    snr,
                    fwhm: moments.fwhm(),
                    saturated: is_saturated(peak.x, peak.y),
                    brightness: peak.value,
                })
            })
            .collect();

        // Isolation.
        let grid = PeakGrid::new(&peaks, width, height, half.max(8));
        let catalogue_grid = CatalogueGrid::new(catalogue, width, height, half.max(8));
        // A neighbour within `isolation_magnitudes` of a star disqualifies
        // it when its mask would reach the flux aperture, as does one
        // brighter than the star anywhere in its stamp. Other neighbours are
        // masked out of the stamp, `mask_radius` around them.
        let isolation_radius = half as f64 * std::f64::consts::SQRT_2 + 1.0;
        let ratio = 10_f64.powf(-0.4 * options.isolation_magnitudes) as f32;
        let mask_radius = (1.5 * fwhm0).max(3.0);
        let hard_radius = aperture + mask_radius;
        let match_radius = fwhm0.max(2.0);
        let catalogue_isolated = |star: &Candidate, magnitudes: f64| -> bool {
            if catalogue.is_empty() {
                return true;
            }
            let Some((own, own_magnitude)) = catalogue_grid.nearest(star.x, star.y, match_radius)
            else {
                return false;
            };
            !catalogue_grid.any_near(star.x, star.y, hard_radius, |index, magnitude| {
                index != own && magnitude < own_magnitude + magnitudes
            }) && !catalogue_grid.any_near(star.x, star.y, isolation_radius, |index, magnitude| {
                index != own && magnitude < own_magnitude
            })
        };
        // Every detected neighbour is masked, however faint: a star a
        // hundred times fainter still outshines the PSF's wings where it
        // sits.
        let neighbours = |star: &Candidate| -> Vec<(f64, f64)> {
            let mut found: Vec<(f64, f64)> = grid
                .near(star.x, star.y, isolation_radius + mask_radius, 2.0)
                .into_iter()
                .map(|peak| (peak.x as f64, peak.y as f64))
                .collect();
            if let Some((own, own_magnitude)) = catalogue_grid.nearest(star.x, star.y, match_radius)
            {
                catalogue_grid.for_each_near(
                    star.x,
                    star.y,
                    isolation_radius + mask_radius,
                    |index, _, magnitude| {
                        if index != own && magnitude < own_magnitude + options.isolation_magnitudes
                        {
                            let neighbour = catalogue[index];
                            found.push((neighbour.x, neighbour.y));
                        }
                    },
                );
            }
            found
        };
        // A bloomed star, whose saturated disc is 3 px or more in radius,
        // has wings that reach well past its own stamp, so a core star keeps
        // twice the isolation radius from any.
        let bloomed = |x: usize, y: usize| {
            let reach = BLOOM_RADIUS as isize * 2;
            let area = (-reach..=reach)
                .flat_map(|dy| (-reach..=reach).map(move |dx| (dx, dy)))
                .filter(|&(dx, dy)| {
                    let (px, py) = (x as isize + dx, y as isize + dy);
                    planes
                        .iter()
                        .zip(&saturation)
                        .any(|(plane, &level)| plane.at(px, py) >= f64::from(level))
                })
                .count();
            area as f64 >= std::f64::consts::PI * (BLOOM_RADIUS * BLOOM_RADIUS) as f64
        };
        let saturated_peaks: Vec<_> = peaks
            .iter()
            .filter(|peak| is_saturated(peak.x, peak.y) && bloomed(peak.x, peak.y))
            .copied()
            .collect();
        let saturated_grid = PeakGrid::new(&saturated_peaks, width, height, half.max(8));
        let core_candidates: Vec<Candidate> = candidates
            .iter()
            .filter(|star| {
                !star.saturated
                    && saturated_grid.brightest_near(star.x, star.y, 2.0 * isolation_radius, 0.0)
                        == f32::NEG_INFINITY
                    && star.snr >= options.minimum_snr
                    && star.fwhm >= 0.5 * fwhm0
                    && star.fwhm <= 2.0 * fwhm0
                    && star.flux > 0.0
                    && grid.brightest_near(star.x, star.y, hard_radius, 2.0)
                        < ratio * star.brightness
                    && grid.brightest_near(star.x, star.y, isolation_radius, 2.0) < star.brightness
                    && catalogue_isolated(star, options.isolation_magnitudes)
            })
            .copied()
            .collect();
        let mut wing_candidates: Vec<Candidate> = candidates
            .iter()
            .filter(|star| {
                star.saturated
                    && star.fwhm >= 0.5 * fwhm0
                    && grid.brightest_near(star.x, star.y, isolation_radius, 2.0) < star.brightness
                    && catalogue_isolated(star, 2.5)
            })
            .copied()
            .collect();
        wing_candidates.sort_by(|a, b| b.flux.total_cmp(&a.flux));
        wing_candidates.truncate(WING_STAR_LIMIT);

        let core = balanced_selection(core_candidates, width, height, options.maximum_stars);
        let sparse = core.len() < required;
        if sparse
            && (options.sparse == SparseFieldPolicy::Refuse || core.len() < GLOBAL_MINIMUM_STARS)
        {
            return Err(too_few(core.len()));
        }
        let global = sparse || options.components == 0;

        // Core stamps, for every channel; a star must give one in each.
        let field_u = |x: f64| field_coordinate(x, width);
        let field_v = |y: f64| field_coordinate(y, height);
        let mask_of = |star: &Candidate, near: &[(f64, f64)]| -> Option<Vec<bool>> {
            if near.is_empty() {
                return None;
            }
            let origin = (star.x - half as f64, star.y - half as f64);
            Some(
                (0..stamp_size * stamp_size)
                    .map(|pixel| {
                        let x = origin.0 + (pixel % stamp_size) as f64;
                        let y = origin.1 + (pixel / stamp_size) as f64;
                        near.iter()
                            .all(|&(nx, ny)| (x - nx).hypot(y - ny) > mask_radius)
                    })
                    .collect(),
            )
        };
        let stamp_of = |plane: &Plane<'_>,
                        star: &Candidate,
                        valid: &Option<Vec<bool>>,
                        near: &[(f64, f64)]| {
            let (px, py) = (star.x.round() as isize, star.y.round() as isize);
            // Neighbours' light is not sky.
            let clear = |x: isize, y: isize| {
                near.iter()
                    .all(|&(nx, ny)| (x as f64 - nx).hypot(y as f64 - ny) > mask_radius)
            };
            let (background, sigma) =
                plane.ring_background(px, py, half as isize + 1, half as isize + 6, clear)?;
            let values = plane.shifted_stamp(star.x, star.y, stamp_size, background)?;
            let flux = aperture_sum(&values, stamp_size, aperture);
            if flux.is_nan() || flux <= 0.0 {
                return None;
            }
            Some(StarStamp {
                u: field_u(star.x),
                v: field_v(star.y),
                values: values.iter().map(|value| value / flux).collect(),
                noise: sigma.max(1e-12) / flux,
                valid: valid.clone(),
            })
        };
        let mut core_stars: Vec<(Candidate, Vec<StarStamp>)> = core
            .par_iter()
            .filter_map(|star| {
                let near = neighbours(star);
                let valid = mask_of(star, &near);
                let stamps: Option<Vec<StarStamp>> = planes
                    .iter()
                    .map(|plane| stamp_of(plane, star, &valid, &near))
                    .collect();
                stamps.map(|stamps| (*star, stamps))
            })
            .collect();
        if core_stars.len() < GLOBAL_MINIMUM_STARS || (!sparse && core_stars.len() < required) {
            return Err(too_few(core_stars.len()));
        }

        let radius = radii(stamp_size);
        let core_mask: Vec<bool> = radius
            .iter()
            .map(|&r| r <= (1.5 * fwhm0).max(2.0))
            .collect();
        let degree = if global { Some(0) } else { options.degree };
        let floor = if sparse {
            GLOBAL_MINIMUM_STARS
        } else {
            required
        };
        let fit_all = |stars: &[(Candidate, Vec<StarStamp>)]| -> Vec<ChannelModel> {
            let components = if global {
                0
            } else {
                options.components.min(stars.len() / 10)
            };
            (0..channels)
                .into_par_iter()
                .map(|channel| {
                    let stamps: Vec<&StarStamp> =
                        stars.iter().map(|(_, stamps)| &stamps[channel]).collect();
                    model::fit(&stamps, stamp_size, components, degree)
                })
                .collect()
        };
        // Fit, drop the stars the fit does not describe, fit again.
        let mut models = fit_all(&core_stars);
        for _ in 0..3 {
            let mut outlier = vec![false; core_stars.len()];
            for (channel, model) in models.iter().enumerate() {
                let stamps: Vec<&StarStamp> = core_stars
                    .iter()
                    .map(|(_, stamps)| &stamps[channel])
                    .collect();
                let quality = model::quality(model, &stamps, &core_mask);
                for (flag, is_outlier) in outlier.iter_mut().zip(model::outliers(&quality.relative))
                {
                    *flag |= is_outlier;
                }
            }
            let remaining = outlier.iter().filter(|flag| !**flag).count();
            if remaining == core_stars.len() || remaining < floor {
                break;
            }
            let mut flags = outlier.into_iter();
            core_stars.retain(|_| !flags.next().unwrap_or(false));
            models = fit_all(&core_stars);
        }

        // Wings from saturated stars, per channel.
        let mut fits = Vec::with_capacity(channels);
        let mut wing_used = vec![false; wing_candidates.len()];
        for (channel, model) in models.iter_mut().enumerate() {
            let stamps: Vec<&StarStamp> = core_stars
                .iter()
                .map(|(_, stamps)| &stamps[channel])
                .collect();
            let (fit, used) = finish_channel(
                model,
                &stamps,
                &core_mask,
                &WingSource {
                    plane: &planes[channel],
                    saturation: saturation[channel],
                    stars: &wing_candidates,
                    radius: &radius,
                    fwhm0,
                },
                field_u,
                field_v,
            );
            for index in used {
                wing_used[index] = true;
            }
            fits.push(fit);
        }

        let mut stars: Vec<PsfStar> = core_stars
            .iter()
            .map(|(star, _)| psf_star(star, PsfStarRole::Core))
            .collect();
        stars.extend(
            wing_candidates
                .iter()
                .zip(&wing_used)
                .filter(|(_, used)| **used)
                .map(|(star, _)| psf_star(star, PsfStarRole::Wing)),
        );
        let warnings = if sparse {
            vec![PsfFieldWarning::GlobalPsf {
                usable: core_stars.len(),
                required,
            }]
        } else {
            Vec::new()
        };
        Ok(Self {
            width,
            height,
            stamp_size,
            models,
            fits,
            stars,
            warnings,
            global,
        })
    }

    /// The image width the field was measured on.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The image height the field was measured on.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Channels: one, or three for RGB, each with its own field.
    pub fn channels(&self) -> usize {
        self.models.len()
    }

    /// The side of every stamp, in pixels; odd.
    pub fn stamp_size(&self) -> usize {
        self.stamp_size
    }

    /// Principal components beyond the mean PSF, per channel. Zero for a
    /// global PSF.
    pub fn component_count(&self) -> usize {
        self.models
            .first()
            .map_or(0, |model| model.components.len())
    }

    /// Unsaturated stars the core was fitted from.
    pub fn star_count(&self) -> usize {
        self.stars
            .iter()
            .filter(|star| star.role == PsfStarRole::Core)
            .count()
    }

    /// Every star the field was fitted from: the core's, then the wings'.
    pub fn stars(&self) -> &[PsfStar] {
        &self.stars
    }

    /// How well each channel's field fits its stars.
    pub fn channel_fits(&self) -> &[PsfChannelFit] {
        &self.fits
    }

    /// Whether one PSF serves the whole image: the options asked for no
    /// components, or there were too few stars for a varying field and
    /// [`SparseFieldPolicy::Global`] allowed one PSF, which
    /// [`Self::warnings`] then reports.
    pub fn is_global(&self) -> bool {
        self.global
    }

    /// What a caller should know about the fit.
    pub fn warnings(&self) -> &[PsfFieldWarning] {
        &self.warnings
    }

    /// The local PSF at pixel `(x, y)`: each channel's stamp, summing to
    /// one, centred on the middle pixel. Points outside the image take the
    /// PSF at the nearest edge, since the polynomials do not extrapolate
    /// safely.
    pub fn stamp_at(&self, x: f64, y: f64) -> PsfStamp {
        let (u, v) = (
            field_coordinate(x, self.width).clamp(-1.0, 1.0),
            field_coordinate(y, self.height).clamp(-1.0, 1.0),
        );
        let channels = self.models.len();
        let pixels = self.stamp_size * self.stamp_size;
        let mut data = vec![0.0; pixels * channels];
        for (channel, model) in self.models.iter().enumerate() {
            let values = model.raw_at(u, v);
            let total: f64 = values.iter().sum();
            let scale = if total > 0.0 { 1.0 / total } else { 1.0 };
            for (index, value) in values.into_iter().enumerate() {
                data[index * channels + channel] = (value * scale) as f32;
            }
        }
        PsfStamp {
            size: self.stamp_size,
            channels,
            data,
        }
    }

    /// The local PSF's shape in one channel at pixel `(x, y)`, or `None`
    /// when `channel` is out of range or the profile cannot be fitted.
    pub fn shape_at(&self, channel: usize, x: f64, y: f64) -> Option<PsfShape> {
        if channel >= self.models.len() {
            return None;
        }
        self.stamp_at(x, y).shape(channel)
    }
}

/// `x` on `[−1, 1]` across a length of `length` pixels.
fn field_coordinate(x: f64, length: usize) -> f64 {
    let half = ((length as f64 - 1.0) / 2.0).max(0.5);
    (x - (length as f64 - 1.0) / 2.0) / half
}

fn psf_star(star: &Candidate, role: PsfStarRole) -> PsfStar {
    PsfStar {
        x: star.x,
        y: star.y,
        flux: star.flux,
        snr: star.snr,
        fwhm: star.fwhm,
        role,
    }
}

/// Up to `limit` stars, highest SNR first within each of a grid of cells,
/// taken from the cells in turn.
fn balanced_selection(
    stars: Vec<Candidate>,
    width: usize,
    height: usize,
    limit: usize,
) -> Vec<Candidate> {
    let mut cells: Vec<Vec<Candidate>> = vec![Vec::new(); SELECTION_GRID * SELECTION_GRID];
    for star in stars {
        let column =
            ((star.x / width as f64 * SELECTION_GRID as f64) as usize).min(SELECTION_GRID - 1);
        let row =
            ((star.y / height as f64 * SELECTION_GRID as f64) as usize).min(SELECTION_GRID - 1);
        cells[row * SELECTION_GRID + column].push(star);
    }
    for cell in &mut cells {
        cell.sort_by(|a, b| a.snr.total_cmp(&b.snr));
    }
    let mut chosen = Vec::with_capacity(limit);
    while chosen.len() < limit {
        let before = chosen.len();
        for cell in &mut cells {
            if chosen.len() >= limit {
                break;
            }
            if let Some(star) = cell.pop() {
                chosen.push(star);
            }
        }
        if chosen.len() == before {
            break;
        }
    }
    chosen
}

/// A spatial index over catalogue stars.
struct CatalogueGrid<'a> {
    stars: &'a [CatalogueStar],
    cell: f64,
    columns: usize,
    rows: usize,
    buckets: Vec<Vec<u32>>,
}

impl<'a> CatalogueGrid<'a> {
    fn new(stars: &'a [CatalogueStar], width: usize, height: usize, cell: usize) -> Self {
        let cell = cell.max(1) as f64;
        let columns = (width as f64 / cell).ceil() as usize + 1;
        let rows = (height as f64 / cell).ceil() as usize + 1;
        let mut buckets = vec![Vec::new(); columns * rows];
        for (index, star) in stars.iter().enumerate() {
            if !(star.x.is_finite() && star.y.is_finite() && star.magnitude.is_finite()) {
                continue;
            }
            let column = (star.x / cell).floor();
            let row = (star.y / cell).floor();
            if column < 0.0 || row < 0.0 || column >= columns as f64 || row >= rows as f64 {
                continue;
            }
            buckets[row as usize * columns + column as usize].push(index as u32);
        }
        Self {
            stars,
            cell,
            columns,
            rows,
            buckets,
        }
    }

    /// Visit each catalogue star within `radius`: its index, distance and
    /// magnitude.
    fn for_each_near(&self, x: f64, y: f64, radius: f64, mut visit: impl FnMut(usize, f64, f64)) {
        let reach = (radius / self.cell).ceil() as isize;
        let (column, row) = (
            (x / self.cell).floor() as isize,
            (y / self.cell).floor() as isize,
        );
        for r in row - reach..=row + reach {
            for c in column - reach..=column + reach {
                if r < 0 || c < 0 || r >= self.rows as isize || c >= self.columns as isize {
                    continue;
                }
                for &index in &self.buckets[r as usize * self.columns + c as usize] {
                    let star = self.stars[index as usize];
                    let distance = (star.x - x).hypot(star.y - y);
                    if distance <= radius {
                        visit(index as usize, distance, star.magnitude);
                    }
                }
            }
        }
    }

    /// The nearest catalogue star within `radius`: its index and magnitude.
    fn nearest(&self, x: f64, y: f64, radius: f64) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64, f64)> = None;
        self.for_each_near(x, y, radius, |index, distance, magnitude| {
            if best.is_none_or(|(_, nearest, _)| distance < nearest) {
                best = Some((index, distance, magnitude));
            }
        });
        best.map(|(index, _, magnitude)| (index, magnitude))
    }

    /// Whether a catalogue star within `radius` passes `test(index, magnitude)`.
    fn any_near(&self, x: f64, y: f64, radius: f64, test: impl Fn(usize, f64) -> bool) -> bool {
        let mut found = false;
        self.for_each_near(x, y, radius, |index, _, magnitude| {
            found |= test(index, magnitude);
        });
        found
    }
}

/// Where a channel's wings come from: its plane, its saturation level, the
/// saturated stars, each stamp pixel's radius and the field's FWHM.
struct WingSource<'a> {
    plane: &'a Plane<'a>,
    saturation: f32,
    stars: &'a [Candidate],
    radius: &'a [f64],
    fwhm0: f64,
}

/// Add the wings to one channel's model, then measure how well its stars
/// fit the model the field will return: the fit and the wing stars used.
fn finish_channel(
    model: &mut ChannelModel,
    stamps: &[&StarStamp],
    core_mask: &[bool],
    wings: &WingSource<'_>,
    field_u: impl Fn(f64) -> f64,
    field_v: impl Fn(f64) -> f64,
) -> (PsfChannelFit, Vec<usize>) {
    let attached = attach_wings(
        model,
        stamps,
        wings.plane,
        wings.saturation,
        wings.stars,
        wings.radius,
        wings.fwhm0,
        field_u,
        field_v,
    );
    let quality = model::quality(model, stamps, core_mask);
    let fit = PsfChannelFit {
        reduced_chi_squared: median_f64(&mut quality.chi_squared.clone()),
        relative_residual: median_f64(&mut quality.relative.clone()),
        degree: model.degree,
        wing_stars: attached.used.len(),
        wing_overlap: attached.overlap,
    };
    (fit, attached.used)
}

/// The saturated stars a channel's wings came from, and where the core
/// hands over to them.
struct Wings {
    used: Vec<usize>,
    overlap: Option<(f64, f64)>,
}

/// Add to `model`'s outer part what the saturated stars hold beyond it.
/// Each star is scaled to the core model at its position over an annulus
/// where both are reliable: outside the star's saturated pixels, and inside
/// the radius where the core model still stands well above its own noise.
/// The median of their excess over the core model is blended in from that
/// annulus outwards, so the core's variation across the field is kept and
/// the wings add the halo faint stars cannot show.
#[allow(clippy::too_many_arguments)]
fn attach_wings(
    model: &mut ChannelModel,
    stamps: &[&StarStamp],
    plane: &Plane<'_>,
    saturation: f32,
    wing_stars: &[Candidate],
    radius: &[f64],
    fwhm0: f64,
    field_u: impl Fn(f64) -> f64,
    field_v: impl Fn(f64) -> f64,
) -> Wings {
    let none = Wings {
        used: Vec::new(),
        overlap: None,
    };
    let size = model.size;
    let half = (size / 2) as isize;
    // The mean's own noise per pixel, and the radius out to which its ring
    // averages stand twenty times above it.
    let precision: f64 = stamps
        .iter()
        .map(|stamp| 1.0 / stamp.noise.max(1e-12).powi(2))
        .sum();
    let mean_noise = 1.0 / precision.sqrt();
    let rings = half as usize + 1;
    let mut ring_sum = vec![0.0; rings + 1];
    let mut ring_count = vec![0usize; rings + 1];
    for (value, &r) in model.mean.iter().zip(radius) {
        let ring = r.round() as usize;
        if ring <= rings {
            ring_sum[ring] += value;
            ring_count[ring] += 1;
        }
    }
    let reliable = (1..=rings)
        .find(|&ring| {
            let count = ring_count[ring].max(1) as f64;
            ring_sum[ring] / count * count.sqrt() < 20.0 * mean_noise
        })
        .unwrap_or(rings) as f64;

    let mut scaled: Vec<(usize, Vec<f64>, f64, f64)> = Vec::new();
    for (index, star) in wing_stars.iter().enumerate() {
        let (px, py) = (star.x.round() as isize, star.y.round() as isize);
        // The sky from a ring three stamp radii out, where even a bright
        // star's own wings have faded.
        let Some((background, _)) =
            plane.ring_background(px, py, 3 * half, 3 * half + 6, |_, _| true)
        else {
            continue;
        };
        let Some(values) = plane.shifted_stamp(star.x, star.y, size, background) else {
            continue;
        };
        // The farthest saturated pixel sets where the star becomes usable.
        let mut saturated_radius: f64 = 0.0;
        for y in py - half..=py + half {
            for x in px - half..=px + half {
                if plane.at(x, y) >= f64::from(saturation) {
                    saturated_radius =
                        saturated_radius.max((x as f64 - star.x).hypot(y as f64 - star.y));
                }
            }
        }
        let inner = (saturated_radius + 1.5).max(fwhm0);
        let outer = (inner + fwhm0.max(2.0)).min(reliable);
        if outer - inner < 1.0 {
            continue;
        }
        let local = model.raw_at(field_u(star.x), field_v(star.y));
        let (mut cross, mut square) = (0.0, 0.0);
        for ((value, m), &r) in values.iter().zip(&local).zip(radius) {
            if r >= inner && r <= outer {
                cross += m * value;
                square += m * m;
            }
        }
        if !(square > 0.0 && cross > 0.0) {
            continue;
        }
        // What the star holds beyond the core model at its own position:
        // the halo the core stars are too faint to show, without the
        // field's variation, which the core model keeps.
        let scale = cross / square;
        scaled.push((
            index,
            values
                .iter()
                .zip(&local)
                .map(|(value, m)| value / scale - m)
                .collect(),
            inner,
            outer,
        ));
    }
    if scaled.len() < WING_MINIMUM_STARS {
        return none;
    }
    let mut inners: Vec<f64> = scaled.iter().map(|entry| entry.2).collect();
    let mut outers: Vec<f64> = scaled.iter().map(|entry| entry.3).collect();
    let (start, end) = (median_f64(&mut inners), median_f64(&mut outers));
    for (pixel, &r) in radius.iter().enumerate() {
        if r <= start {
            continue;
        }
        let mut samples: Vec<f64> = scaled
            .iter()
            .filter(|entry| r >= entry.2)
            .map(|entry| entry.1[pixel])
            .collect();
        if samples.is_empty() {
            continue;
        }
        let excess = median_f64(&mut samples);
        let t = if r >= end {
            1.0
        } else {
            let s = (r - start) / (end - start);
            s * s * (3.0 - 2.0 * s)
        };
        model.mean[pixel] += t * excess;
    }
    Wings {
        used: scaled.iter().map(|entry| entry.0).collect(),
        overlap: Some((start, end)),
    }
}

#[cfg(test)]
mod tests;
