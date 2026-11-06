//! Catalogue-guided star photometry, removal and rendering.
//!
//! Given the stars a catalogue places in a solved image, [`StarFit`] fits
//! each star's flux with the local [`PsfField`] stamp: overlapping stars
//! jointly, each group with a background plane of its own. The fit can then
//! remove the stars, [`StarFit::subtract`], and render them again through a
//! narrower round Gaussian, [`StarFit::render`]. Removing them and rendering
//! them at the target, [`StarFit::sharpen`], is the classical star
//! sharpener, and the rendered stars are exact star truth for training.

use std::fmt;
use std::str::FromStr;

use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::gaussian::{FWHM_PER_SIGMA, pixel_profile};
use super::stamps::lanczos_weights;
use super::{CatalogueStar, PsfField};

/// Stars a group is solved jointly with at most; a larger chain of
/// overlapping stars is split, and the parts solved in turn.
const GROUP_LIMIT: usize = 48;

/// Passes over the groups. Each pass fits every group with the other
/// groups' current stars subtracted, so light that a star's wings spread
/// into its neighbours' windows is accounted for.
const PASSES: usize = 3;

/// The most memory the fit keeps each star's placed stamp in, in bytes.
/// Beyond it, as for a very deep catalogue, stamps are placed again when
/// they are needed.
const PLACED_CACHE_BYTES: usize = 1 << 30;

/// How the stars' positions enter the fit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StarPositions {
    /// The catalogue's positions, held fixed: PSF photometry at known
    /// positions, as star truth needs.
    #[default]
    Fixed,
    /// Each star may move, by up to [`StarFitOptions::max_shift`], to where
    /// the image puts it: for a solution not accurate to a small fraction of
    /// a pixel.
    Refined,
}

impl StarPositions {
    fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Refined => "refined",
        }
    }
}

impl fmt::Display for StarPositions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for StarPositions {
    type Err = StarFitError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "fixed" => Ok(Self::Fixed),
            "refined" => Ok(Self::Refined),
            other => Err(StarFitError::InvalidOptions(format!(
                "unknown star positions {other:?}; expected fixed or refined"
            ))),
        }
    }
}

/// Settings for [`StarFit::fit`].
#[derive(Clone, Debug, PartialEq)]
pub struct StarFitOptions {
    /// Radius in pixels of the window each star is fitted over. `None` uses
    /// three times the field's central FWHM, at least 4 px. Stars whose
    /// windows overlap are fitted jointly.
    pub fit_radius: Option<f64>,
    /// The level, in the image's units, from which samples are left out of
    /// the fit as saturated. A saturated star's flux then comes from its
    /// wings. `None` leaves every sample in.
    pub saturation: Option<f32>,
    /// Whether the positions are held or refined.
    pub positions: StarPositions,
    /// The farthest a refined star may move, in pixels.
    pub max_shift: f64,
}

impl Default for StarFitOptions {
    fn default() -> Self {
        Self {
            fit_radius: None,
            saturation: None,
            positions: StarPositions::Fixed,
            max_shift: 1.0,
        }
    }
}

impl StarFitOptions {
    /// Check that every setting lies in its supported range.
    pub fn validate(&self) -> Result<(), StarFitError> {
        if let Some(radius) = self.fit_radius
            && !(radius.is_finite() && (1.0..=128.0).contains(&radius))
        {
            return Err(StarFitError::InvalidOptions(
                "fit radius must be in 1..=128 pixels".into(),
            ));
        }
        if let Some(level) = self.saturation
            && !(level.is_finite() && level > 0.0)
        {
            return Err(StarFitError::InvalidOptions(
                "saturation must be finite and positive".into(),
            ));
        }
        if !(self.max_shift.is_finite() && self.max_shift >= 0.0 && self.max_shift <= 5.0) {
            return Err(StarFitError::InvalidOptions(
                "maximum shift must be in 0..=5 pixels".into(),
            ));
        }
        Ok(())
    }
}

/// Why [`StarFit::fit`] or a call on a fit refused.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum StarFitError {
    /// The image does not match the PSF field's size and channels, or a
    /// buffer has the wrong length.
    InvalidImage(String),
    /// A setting is out of range.
    InvalidOptions(String),
}

impl fmt::Display for StarFitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidImage(message) => write!(f, "invalid star fit image: {message}"),
            Self::InvalidOptions(message) => write!(f, "invalid star fit options: {message}"),
        }
    }
}

impl std::error::Error for StarFitError {}

/// One catalogue star's fit.
#[derive(Clone, Debug, PartialEq)]
pub struct FittedStar {
    /// Column the star was fitted at, in pixels from 0: the catalogue's,
    /// or refined.
    pub x: f64,
    /// Row the star was fitted at.
    pub y: f64,
    /// The catalogue magnitude it was given.
    pub magnitude: f64,
    /// Total flux per channel, in the image's units: the PSF stamp's sum
    /// times the fitted amplitude. NaN where no sample constrained it.
    pub flux: Vec<f64>,
    /// The flux's standard error per channel, from the noise map when one
    /// is given and from the fit's own residual otherwise.
    pub flux_error: Vec<f64>,
    /// The fitted background plane at the star, per channel.
    pub background: Vec<f64>,
    /// The χ² per sample of the fit within the star's window, against the
    /// noise map, or against the group's residual scatter without one.
    pub reduced_chi_squared: f64,
    /// Whether samples at or above the saturation level were left out of
    /// its window.
    pub saturated: bool,
}

/// Catalogue stars fitted with a PSF field, ready to remove and render.
#[derive(Clone, Debug)]
pub struct StarFit {
    width: usize,
    height: usize,
    channels: usize,
    stamp_size: usize,
    stars: Vec<FittedStar>,
    /// The PSF field the stars were fitted with.
    field: PsfField,
    /// The fitted stars through the field's PSF: what `subtract` removes.
    model: Vec<f32>,
}

/// A PSF stamp shifted to a star's sub-pixel position, per channel: the
/// pixel the stamp's middle lands on, and its samples.
#[derive(Clone)]
struct Placed {
    centre_x: isize,
    centre_y: isize,
    planes: Vec<Vec<f32>>,
}

/// Each star's placed stamp, kept while the fit runs, or `None` where it is
/// placed again on demand.
type PlacedCache = Vec<Option<Placed>>;

/// `stamp` (`size × size`, centred) moved by `(dx, dy)` of a pixel, each in
/// `[−0.5, 0.5]`, with Lanczos-3; zero beyond its edge.
fn shifted(stamp: &[f32], size: usize, dx: f64, dy: f64) -> Vec<f32> {
    // Output sample u is the stamp at u − dx: from x0 = floor(u − dx) with
    // the same fraction for every u.
    let (offset_x, tx) = if dx > 0.0 { (-1, 1.0 - dx) } else { (0, -dx) };
    let (offset_y, ty) = if dy > 0.0 { (-1, 1.0 - dy) } else { (0, -dy) };
    let (wx, wy) = (lanczos_weights(tx), lanczos_weights(ty));
    let at = |x: isize, y: isize| -> f64 {
        if x < 0 || y < 0 || x >= size as isize || y >= size as isize {
            0.0
        } else {
            f64::from(stamp[y as usize * size + x as usize])
        }
    };
    let mut rows = vec![0.0; size * (size + 6)];
    // Rows y0 − 2 ..= y0 + 3 for every output row, interpolated along x.
    for (row_index, y) in (-3_isize..size as isize + 3).enumerate() {
        for u in 0..size as isize {
            let x0 = u + offset_x;
            rows[row_index * size + u as usize] = wx
                .iter()
                .enumerate()
                .map(|(k, w)| w * at(x0 + k as isize - 2, y))
                .sum();
        }
    }
    let mut out = vec![0.0; size * size];
    for v in 0..size as isize {
        let y0 = v + offset_y;
        for u in 0..size {
            out[v as usize * size + u] = wy
                .iter()
                .enumerate()
                .map(|(k, w)| {
                    let y = y0 + k as isize - 2;
                    w * rows[(y + 3) as usize * size + u]
                })
                .sum::<f64>() as f32;
        }
    }
    out
}

/// A pixel-integrated round Gaussian of `fwhm` at `(x, y)` with total
/// `flux`, added into `plane` (`width × height`).
fn add_gaussian(
    plane: &mut [f32],
    width: usize,
    height: usize,
    x: f64,
    y: f64,
    fwhm: f64,
    flux: f64,
) {
    let sigma = fwhm / FWHM_PER_SIGMA;
    let reach = (4.0 * sigma + 2.0).ceil() as isize;
    let (cx, cy) = (x.round() as isize, y.round() as isize);
    let (x0, x1) = ((cx - reach).max(0), (cx + reach).min(width as isize - 1));
    let (y0, y1) = ((cy - reach).max(0), (cy + reach).min(height as isize - 1));
    if x0 > x1 || y0 > y1 {
        return;
    }
    let (px, py) = (
        pixel_profile(x, x0, x1, sigma),
        pixel_profile(y, y0, y1, sigma),
    );
    for (row, yv) in (y0..=y1).zip(&py) {
        let row = &mut plane[row as usize * width..(row as usize + 1) * width];
        for (column, xv) in (x0..=x1).zip(&px) {
            row[column as usize] += (flux * yv * xv) as f32;
        }
    }
}

impl StarFit {
    /// Fit the flux of each of `stars`, at their image positions, in a
    /// linear image of `field`'s size and channels, interleaved. `noise`,
    /// when given, is the per-sample noise as a standard deviation in the
    /// image's layout; a sample whose noise is not finite and positive is
    /// left out. Stars outside the image, or whose window holds no usable
    /// sample, get NaN fluxes and are neither removed nor rendered.
    pub fn fit(
        data: &[f32],
        field: &PsfField,
        stars: &[CatalogueStar],
        noise: Option<&[f32]>,
        options: &StarFitOptions,
    ) -> Result<Self, StarFitError> {
        options.validate()?;
        let (width, height, channels) = (field.width(), field.height(), field.channels());
        let expected = width * height * channels;
        if data.len() != expected {
            return Err(StarFitError::InvalidImage(format!(
                "image has {} samples; the PSF field's image had {expected}",
                data.len()
            )));
        }
        if let Some(noise) = noise
            && noise.len() != expected
        {
            return Err(StarFitError::InvalidImage(format!(
                "noise map has {} samples; expected {expected}",
                noise.len()
            )));
        }
        let size = field.stamp_size();
        let half = (size / 2) as isize;
        let radius = options.fit_radius.unwrap_or_else(|| {
            let centre = field.shape_at(0, (width as f64 - 1.0) / 2.0, (height as f64 - 1.0) / 2.0);
            (3.0 * centre.map_or(3.0, |shape| shape.fwhm)).clamp(4.0, half as f64)
        });

        let mut fitted: Vec<FittedStar> = stars
            .iter()
            .map(|star| FittedStar {
                x: star.x,
                y: star.y,
                magnitude: star.magnitude,
                flux: vec![f64::NAN; channels],
                flux_error: vec![f64::NAN; channels],
                background: vec![f64::NAN; channels],
                reduced_chi_squared: f64::NAN,
                saturated: false,
            })
            .collect();
        let inside: Vec<usize> = (0..stars.len())
            .filter(|&i| {
                let star = &stars[i];
                star.x.is_finite()
                    && star.y.is_finite()
                    && star.x > -radius
                    && star.y > -radius
                    && star.x < width as f64 - 1.0 + radius
                    && star.y < height as f64 - 1.0 + radius
            })
            .collect();
        let groups = groups(&inside, stars, 2.0 * radius, width, height);
        let mut fit = Self {
            width,
            height,
            channels,
            stamp_size: size,
            stars: Vec::new(),
            field: field.clone(),
            model: Vec::new(),
        };
        // Each star's stamp is placed once, and again only when refinement
        // moves it, unless that would hold too much memory.
        let keep = inside.len() * size * size * channels * std::mem::size_of::<f32>()
            <= PLACED_CACHE_BYTES;
        let place = |fit: &Self, stars: &[FittedStar], cache: &mut PlacedCache| {
            if keep {
                let placed: Vec<(usize, Placed)> = inside
                    .par_iter()
                    .map(|&i| (i, fit.placed(stars[i].x, stars[i].y)))
                    .collect();
                for (i, stamp) in placed {
                    cache[i] = Some(stamp);
                }
            }
        };
        let mut cache: PlacedCache = vec![None; stars.len()];
        place(&fit, &fitted, &mut cache);
        let mut amplitudes = vec![vec![0.0_f64; channels]; stars.len()];
        let passes = match options.positions {
            StarPositions::Fixed => PASSES,
            StarPositions::Refined => PASSES + 2,
        };
        for pass in 0..passes {
            // The model of every star but a group's own, which each group
            // is fitted against.
            let model = fit.model_of(&fitted, &amplitudes, &cache);
            let refine = options.positions == StarPositions::Refined && pass + 1 < passes;
            let solved: Vec<Vec<(usize, Solution)>> = groups
                .par_iter()
                .map(|group| {
                    fit.solve_group(
                        group,
                        &fitted,
                        &amplitudes,
                        &cache,
                        data,
                        noise,
                        &model,
                        radius,
                        options,
                        refine,
                    )
                })
                .collect();
            for (index, solution) in solved.into_iter().flatten() {
                let star = &mut fitted[index];
                amplitudes[index] = solution.amplitude.clone();
                star.flux = solution.amplitude;
                star.flux_error = solution.error;
                star.background = solution.background;
                star.reduced_chi_squared = solution.chi_squared;
                star.saturated = solution.saturated;
                if let Some((dx, dy)) = solution.shift {
                    let (ox, oy) = (stars[index].x, stars[index].y);
                    let (nx, ny) = (star.x + dx, star.y + dy);
                    let moved = (nx - ox).hypot(ny - oy);
                    let limit = options.max_shift;
                    let scale = if moved > limit { limit / moved } else { 1.0 };
                    star.x = ox + (nx - ox) * scale;
                    star.y = oy + (ny - oy) * scale;
                }
            }
            if refine {
                place(&fit, &fitted, &mut cache);
            }
        }
        fit.model = fit.model_of(&fitted, &amplitudes, &cache);
        fit.stars = fitted;
        Ok(fit)
    }

    /// The image width the fit covers.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The image height the fit covers.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Channels, one or three.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// The fitted stars, in the order they were given.
    pub fn stars(&self) -> &[FittedStar] {
        &self.stars
    }

    /// The fitted stars through the measured PSF, on black: the star layer
    /// [`Self::subtract`] removes.
    pub fn model(&self) -> Vec<f32> {
        self.model.clone()
    }

    /// `data` with the fitted stars removed: the image less [`Self::model`].
    /// The background and everything a star does not explain stay.
    pub fn subtract(&self, data: &[f32]) -> Result<Vec<f32>, StarFitError> {
        self.check(data.len())?;
        Ok(data
            .iter()
            .zip(&self.model)
            .map(|(value, star)| value - star)
            .collect())
    }

    /// The fitted stars rendered through a round Gaussian of `target_fwhm`
    /// pixels, pixel-integrated, at their positions and fluxes, on black.
    pub fn render(&self, target_fwhm: f64) -> Result<Vec<f32>, StarFitError> {
        if !(target_fwhm.is_finite() && target_fwhm > 0.0) {
            return Err(StarFitError::InvalidOptions(
                "target FWHM must be finite and positive".into(),
            ));
        }
        let (width, height, channels) = (self.width, self.height, self.channels);
        let mut planes: Vec<Vec<f32>> = (0..channels)
            .into_par_iter()
            .map(|channel| {
                let mut plane = vec![0.0_f32; width * height];
                for star in &self.stars {
                    let flux = star.flux[channel];
                    if flux.is_finite() {
                        add_gaussian(&mut plane, width, height, star.x, star.y, target_fwhm, flux);
                    }
                }
                plane
            })
            .collect();
        Ok(interleave(&mut planes, width * height))
    }

    /// The classical star sharpener: `data` with the fitted stars removed
    /// and rendered again through a round Gaussian of `target_fwhm` pixels.
    /// Stars the catalogue lacks stay as they were.
    pub fn sharpen(&self, data: &[f32], target_fwhm: f64) -> Result<Vec<f32>, StarFitError> {
        let removed = self.subtract(data)?;
        let rendered = self.render(target_fwhm)?;
        Ok(removed.iter().zip(&rendered).map(|(a, b)| a + b).collect())
    }

    fn check(&self, length: usize) -> Result<(), StarFitError> {
        let expected = self.width * self.height * self.channels;
        if length == expected {
            Ok(())
        } else {
            Err(StarFitError::InvalidImage(format!(
                "image has {length} samples; expected {expected}"
            )))
        }
    }

    /// The PSF stamp at a star's position, shifted to it, per channel and
    /// normalised to unit sum.
    fn placed(&self, x: f64, y: f64) -> Placed {
        let stamp = self.field.stamp_at(x, y);
        let (centre_x, centre_y) = (x.round() as isize, y.round() as isize);
        let (dx, dy) = (x - centre_x as f64, y - centre_y as f64);
        let planes = (0..self.channels)
            .map(|channel| shifted(&stamp.plane(channel), self.stamp_size, dx, dy))
            .collect();
        Placed {
            centre_x,
            centre_y,
            planes,
        }
    }

    /// A star's placed stamp: kept, or placed again.
    fn placed_of<'a>(
        &self,
        cache: &'a PlacedCache,
        index: usize,
        star: &FittedStar,
    ) -> std::borrow::Cow<'a, Placed> {
        match &cache[index] {
            Some(placed) => std::borrow::Cow::Borrowed(placed),
            None => std::borrow::Cow::Owned(self.placed(star.x, star.y)),
        }
    }

    /// Σ amplitude · stamp over the stars with finite amplitudes.
    fn model_of(
        &self,
        stars: &[FittedStar],
        amplitudes: &[Vec<f64>],
        cache: &PlacedCache,
    ) -> Vec<f32> {
        let (width, height, channels, size) =
            (self.width, self.height, self.channels, self.stamp_size);
        let half = (size / 2) as isize;
        const BAND: usize = 256;
        let mut output = vec![0.0_f32; width * height * channels];
        output
            .par_chunks_mut(BAND * width * channels)
            .enumerate()
            .for_each(|(band, chunk)| {
                let (top, bottom) = (band * BAND, ((band + 1) * BAND).min(height));
                for (index, (star, amplitude)) in stars.iter().zip(amplitudes).enumerate() {
                    if !amplitude.iter().any(|a| a.is_finite() && *a != 0.0) {
                        continue;
                    }
                    let cy = star.y.round() as isize;
                    if cy + half < top as isize || cy - half >= bottom as isize {
                        continue;
                    }
                    let placed = self.placed_of(cache, index, star);
                    for v in 0..size as isize {
                        let y = placed.centre_y + v - half;
                        if y < top as isize || y >= bottom as isize {
                            continue;
                        }
                        for u in 0..size as isize {
                            let x = placed.centre_x + u - half;
                            if x < 0 || x >= width as isize {
                                continue;
                            }
                            let base = ((y as usize - top) * width + x as usize) * channels;
                            for channel in 0..channels {
                                let a = amplitude[channel];
                                if a.is_finite() {
                                    chunk[base + channel] += (a * f64::from(
                                        placed.planes[channel][(v * size as isize + u) as usize],
                                    ))
                                        as f32;
                                }
                            }
                        }
                    }
                }
            });
        output
    }

    /// Fit one group's stars jointly, per channel, against the data less
    /// every other star's model.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn solve_group(
        &self,
        group: &[usize],
        stars: &[FittedStar],
        amplitudes: &[Vec<f64>],
        cache: &PlacedCache,
        data: &[f32],
        noise: Option<&[f32]>,
        model: &[f32],
        radius: f64,
        options: &StarFitOptions,
        refine: bool,
    ) -> Vec<(usize, Solution)> {
        let (width, height, channels, size) =
            (self.width, self.height, self.channels, self.stamp_size);
        let half = (size / 2) as isize;
        let placed: Vec<std::borrow::Cow<'_, Placed>> = group
            .iter()
            .map(|&i| self.placed_of(cache, i, &stars[i]))
            .collect();
        // The pixels of the group's windows.
        let (mut x0, mut y0, mut x1, mut y1) = (isize::MAX, isize::MAX, isize::MIN, isize::MIN);
        let reach = radius.ceil() as isize;
        for &i in group {
            let (cx, cy) = (stars[i].x.round() as isize, stars[i].y.round() as isize);
            x0 = x0.min(cx - reach);
            y0 = y0.min(cy - reach);
            x1 = x1.max(cx + reach);
            y1 = y1.max(cy + reach);
        }
        let (x0, y0) = (x0.max(0), y0.max(0));
        let (x1, y1) = (x1.min(width as isize - 1), y1.min(height as isize - 1));
        let mut pixels: Vec<(isize, isize, Vec<usize>)> = Vec::new();
        if x0 <= x1 && y0 <= y1 {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let owners: Vec<usize> = group
                        .iter()
                        .enumerate()
                        .filter(|&(_, &i)| {
                            (x as f64 - stars[i].x).hypot(y as f64 - stars[i].y) <= radius
                        })
                        .map(|(k, _)| k)
                        .collect();
                    if !owners.is_empty() {
                        pixels.push((x, y, owners));
                    }
                }
            }
        }
        let count = group.len();
        // Unknowns: each star's amplitude, then, when refining, its
        // amplitude times its shift in x and y, then the background plane.
        let per_star = if refine { 3 } else { 1 };
        let unknowns = count * per_star + 3;
        let (cx, cy) = (
            group.iter().map(|&i| stars[i].x).sum::<f64>() / count as f64,
            group.iter().map(|&i| stars[i].y).sum::<f64>() / count as f64,
        );
        let stamp_value = |k: usize, channel: usize, x: isize, y: isize| -> f64 {
            let p = &placed[k];
            let (u, v) = (x - p.centre_x + half, y - p.centre_y + half);
            if u < 0 || v < 0 || u >= size as isize || v >= size as isize {
                0.0
            } else {
                f64::from(p.planes[channel][(v * size as isize + u) as usize])
            }
        };
        let mut results: Vec<Solution> = (0..count).map(|_| Solution::empty(channels)).collect();
        let mut shift_sums = vec![(0.0, 0.0, 0.0); count];
        for channel in 0..channels {
            // The PSF model is good to its relative residual, so a bright
            // core's samples are worth less than their photon noise says:
            // otherwise the model's percent-level error there drives the
            // background plane under a faint neighbour.
            let model_error = self
                .field
                .channel_fits()
                .get(channel)
                .map_or(0.01, |fit| fit.relative_residual)
                .clamp(0.005, 0.2);
            let mut normal = DMatrix::<f64>::zeros(unknowns, unknowns);
            let mut right = DVector::<f64>::zeros(unknowns);
            // Each usable sample: its pixel, target, weight and basis.
            let mut used: Vec<(usize, f64, f64, Vec<f64>)> = Vec::new();
            let mut saturated = vec![false; count];
            for (pixel, (x, y, owners)) in pixels.iter().enumerate() {
                let index = (*y as usize * width + *x as usize) * channels + channel;
                let value = data[index];
                if !value.is_finite() {
                    continue;
                }
                if options.saturation.is_some_and(|level| value >= level) {
                    for &k in owners {
                        saturated[k] = true;
                    }
                    continue;
                }
                // The data less every star outside this group.
                let mut own = 0.0;
                for (k, &i) in group.iter().enumerate() {
                    let a = amplitudes[i][channel];
                    if a.is_finite() {
                        own += a * stamp_value(k, channel, *x, *y);
                    }
                }
                let weight = match noise {
                    Some(noise) => {
                        let sigma = noise[index];
                        if !(sigma.is_finite() && sigma > 0.0) {
                            continue;
                        }
                        let stars = f64::from(model[index]).max(0.0) * model_error;
                        1.0 / (f64::from(sigma).powi(2) + stars * stars)
                    }
                    None => 1.0,
                };
                let target = f64::from(value) - (f64::from(model[index]) - own);
                let mut basis = vec![0.0; unknowns];
                for k in 0..count {
                    let p = stamp_value(k, channel, *x, *y);
                    basis[k * per_star] = p;
                    if refine {
                        // −∂P/∂x and −∂P/∂y: moving the star by +δ
                        // changes the model by −δ·∇P.
                        let gx = 0.5
                            * (stamp_value(k, channel, x - 1, *y)
                                - stamp_value(k, channel, x + 1, *y));
                        let gy = 0.5
                            * (stamp_value(k, channel, *x, y - 1)
                                - stamp_value(k, channel, *x, y + 1));
                        basis[k * per_star + 1] = gx;
                        basis[k * per_star + 2] = gy;
                    }
                }
                basis[unknowns - 3] = 1.0;
                basis[unknowns - 2] = *x as f64 - cx;
                basis[unknowns - 1] = *y as f64 - cy;
                for a in 0..unknowns {
                    if basis[a] == 0.0 {
                        continue;
                    }
                    right[a] += weight * basis[a] * target;
                    for b in 0..unknowns {
                        normal[(a, b)] += weight * basis[a] * basis[b];
                    }
                }
                used.push((pixel, target, weight, basis));
            }
            // Unknowns no sample constrains are pinned to zero.
            let mut free: Vec<usize> = (0..unknowns).filter(|&a| normal[(a, a)] > 0.0).collect();
            if free.len() < 4 || used.len() <= free.len() {
                continue;
            }
            let solution = loop {
                let reduced =
                    DMatrix::from_fn(free.len(), free.len(), |a, b| normal[(free[a], free[b])]);
                let vector = DVector::from_fn(free.len(), |a, _| right[free[a]]);
                if let Some(cholesky) = reduced.clone().cholesky() {
                    let solution = cholesky.solve(&vector);
                    let inverse = cholesky.inverse();
                    break Some((solution, inverse));
                }
                // Drop the least constrained unknown and try again.
                let weakest = (0..free.len())
                    .min_by(|&a, &b| reduced[(a, a)].total_cmp(&reduced[(b, b)]))
                    .unwrap_or(0);
                free.remove(weakest);
                if free.len() < 4 {
                    break None;
                }
            };
            let Some((solution, inverse)) = solution else {
                continue;
            };
            let mut full = vec![0.0; unknowns];
            let mut variance = vec![f64::NAN; unknowns];
            for (a, &unknown) in free.iter().enumerate() {
                full[unknown] = solution[a];
                variance[unknown] = inverse[(a, a)];
            }
            // Residuals: per star over its window, and the group's scatter.
            let mut chi = vec![(0.0, 0usize); count];
            let mut total = 0.0;
            for (pixel, target, weight, basis) in &used {
                let predicted: f64 = basis.iter().zip(&full).map(|(b, s)| b * s).sum();
                let r2 = weight * (target - predicted).powi(2);
                total += r2;
                for &k in &pixels[*pixel].2 {
                    chi[k].0 += r2;
                    chi[k].1 += 1;
                }
            }
            let dof = (used.len() - free.len()).max(1) as f64;
            let scatter = if noise.is_some() { 1.0 } else { total / dof };
            for k in 0..count {
                let amplitude_index = k * per_star;
                let constrained = free.contains(&amplitude_index);
                let result = &mut results[k];
                if constrained {
                    result.amplitude[channel] = full[amplitude_index];
                    result.error[channel] = (variance[amplitude_index] * scatter).sqrt();
                }
                result.background[channel] = full[unknowns - 3]
                    + full[unknowns - 2] * (stars[group[k]].x - cx)
                    + full[unknowns - 1] * (stars[group[k]].y - cy);
                let (sum, n) = chi[k];
                if n > 0 {
                    let value = sum / n as f64 / scatter;
                    result.chi_squared = if result.chi_squared.is_finite() {
                        result.chi_squared.max(value)
                    } else {
                        value
                    };
                }
                result.saturated |= saturated[k];
                if refine && constrained && full[amplitude_index] > 0.0 {
                    let weight = full[amplitude_index];
                    shift_sums[k].0 += full[amplitude_index + 1];
                    shift_sums[k].1 += full[amplitude_index + 2];
                    shift_sums[k].2 += weight;
                }
            }
        }
        if refine {
            for (result, (sx, sy, weight)) in results.iter_mut().zip(&shift_sums) {
                if *weight > 0.0 {
                    result.shift = Some((sx / weight, sy / weight));
                }
            }
        }
        group.iter().copied().zip(results).collect()
    }
}

/// One star's fit in one group.
struct Solution {
    amplitude: Vec<f64>,
    error: Vec<f64>,
    background: Vec<f64>,
    chi_squared: f64,
    saturated: bool,
    shift: Option<(f64, f64)>,
}

impl Solution {
    fn empty(channels: usize) -> Self {
        Self {
            amplitude: vec![f64::NAN; channels],
            error: vec![f64::NAN; channels],
            background: vec![f64::NAN; channels],
            chi_squared: f64::NAN,
            saturated: false,
            shift: None,
        }
    }
}

/// Interleave per-channel planes.
fn interleave(planes: &mut [Vec<f32>], pixels: usize) -> Vec<f32> {
    if planes.len() == 1 {
        return std::mem::take(&mut planes[0]);
    }
    let channels = planes.len();
    let mut out = vec![0.0; pixels * channels];
    for (channel, plane) in planes.iter().enumerate() {
        for (index, value) in plane.iter().enumerate() {
            out[index * channels + channel] = *value;
        }
    }
    out
}

/// The stars `indices` picks, grouped where their windows overlap
/// (closer than `link`), each group split into parts of at most
/// [`GROUP_LIMIT`] stars.
fn groups(
    indices: &[usize],
    stars: &[CatalogueStar],
    link: f64,
    width: usize,
    height: usize,
) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..indices.len()).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    // A grid of cells `link` wide finds each star's neighbours.
    let cell = link.max(1.0);
    let columns = ((width as f64 + 2.0 * link) / cell).ceil() as usize + 1;
    let rows = ((height as f64 + 2.0 * link) / cell).ceil() as usize + 1;
    let key = |x: f64, y: f64| -> (usize, usize) {
        (
            (((x + link) / cell).floor().max(0.0) as usize).min(columns - 1),
            (((y + link) / cell).floor().max(0.0) as usize).min(rows - 1),
        )
    };
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); columns * rows];
    for (k, &i) in indices.iter().enumerate() {
        let (c, r) = key(stars[i].x, stars[i].y);
        buckets[r * columns + c].push(k);
    }
    for (k, &i) in indices.iter().enumerate() {
        let (c, r) = key(stars[i].x, stars[i].y);
        for dr in -1_isize..=1 {
            for dc in -1_isize..=1 {
                let (nc, nr) = (c as isize + dc, r as isize + dr);
                if nc < 0 || nr < 0 || nc >= columns as isize || nr >= rows as isize {
                    continue;
                }
                for &other in &buckets[nr as usize * columns + nc as usize] {
                    if other <= k {
                        continue;
                    }
                    let j = indices[other];
                    if (stars[i].x - stars[j].x).hypot(stars[i].y - stars[j].y) < link {
                        let (a, b) = (root(&mut parent, k), root(&mut parent, other));
                        if a != b {
                            parent[a] = b;
                        }
                    }
                }
            }
        }
    }
    let mut by_root: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for (k, &index) in indices.iter().enumerate() {
        let r = root(&mut parent, k);
        by_root.entry(r).or_default().push(index);
    }
    let mut out = Vec::new();
    for (_, group) in by_root {
        split(group, stars, &mut out);
    }
    out.sort_by_key(|group| group[0]);
    out
}

/// Split `group` by its wider extent's median until each part holds at
/// most [`GROUP_LIMIT`] stars.
fn split(mut group: Vec<usize>, stars: &[CatalogueStar], out: &mut Vec<Vec<usize>>) {
    if group.len() <= GROUP_LIMIT {
        group.sort_unstable();
        out.push(group);
        return;
    }
    let extent = |f: &dyn Fn(&CatalogueStar) -> f64| {
        let values: Vec<f64> = group.iter().map(|&i| f(&stars[i])).collect();
        values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - values.iter().copied().fold(f64::INFINITY, f64::min)
    };
    let along_x = extent(&|s| s.x) >= extent(&|s| s.y);
    group.sort_by(|&a, &b| {
        let (sa, sb) = (&stars[a], &stars[b]);
        if along_x {
            sa.x.total_cmp(&sb.x)
        } else {
            sa.y.total_cmp(&sb.y)
        }
    });
    let right = group.split_off(group.len() / 2);
    split(group, stars, out);
    split(right, stars, out);
}

#[cfg(test)]
mod tests;
