//! N.I.N.A.'s built-in star detector, ported from `StarDetection.cs`.
//!
//! Detection follows N.I.N.A. 3.2 and the 3.3 nightlies, which share it: the
//! autostretched frame is cut to 8 bits, shrunk to at most 1552 pixels
//! across, and passed through Accord's Canny edge detector, SIS threshold
//! and 3 × 3 dilation; each blob of edge pixels is a candidate star. The raw
//! frame then decides which candidates are stars and measures them.
//!
//! [`MeasurementVersion`] chooses how a star is measured. 3.2 takes a
//! first-moment HFR inside 1.2 times the blob's radius, against the mean of
//! the surrounding box. 3.3 refines a windowed centroid, takes the
//! curve-of-growth HFR inside 1.5 times the radius against a sigma-clipped
//! median background, and adds FWHM and eccentricity.
//!
//! The port keeps N.I.N.A.'s quirks, since its star counts and HFR values
//! are what an imaging catalog recorded: they are noted where they occur.

use std::fmt;
use std::str::FromStr;

use crate::accord_imaging::{
    self, BlobCounter, CannyEdgeDetector, DetectionUtility, SimpleShapeChecker,
};

/// Star detection parameters matching N.I.N.A.
#[derive(Debug, Clone)]
pub struct StarDetectionParams {
    pub sensitivity: StarSensitivity,
    pub noise_reduction: NoiseReduction,
    /// Keep only blobs inside the outer crop and not wholly inside the inner
    /// one, as N.I.N.A. does when a crop ratio is set.
    pub use_roi: bool,
    /// Fraction of the frame's width and height the outer crop keeps; 1
    /// keeps all of it.
    pub outer_crop_ratio: f64,
    /// Fraction of the frame's width and height the inner crop covers.
    pub inner_crop_ratio: f64,
    /// Image scale in arcseconds per pixel. [`StarSensitivity::High`] picks
    /// its resize from it; without it, N.I.N.A. halves the image.
    pub image_scale_arcsec: Option<f64>,
    /// Which N.I.N.A. version's star measurement to run.
    pub measurement: MeasurementVersion,
}

impl Default for StarDetectionParams {
    fn default() -> Self {
        Self {
            sensitivity: StarSensitivity::Normal,
            noise_reduction: NoiseReduction::None,
            use_roi: false,
            outer_crop_ratio: 1.0,
            inner_crop_ratio: 1.0,
            image_scale_arcsec: None,
            measurement: MeasurementVersion::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StarSensitivity {
    Normal,
    High,
    Highest,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseReduction {
    None,
    Normal,
    High,
    Highest,
    Median,
}

/// The N.I.N.A. release whose star measurement to reproduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeasurementVersion {
    /// N.I.N.A. 3.1: first-moment HFR about the blob's centre, keeping every
    /// star that passes the brightness test.
    Nina31,
    /// N.I.N.A. 3.2: 3.1's HFR, then a centroid from the same pixels, and
    /// stars whose centroid touches the box's edge are dropped.
    #[default]
    Nina32,
    /// N.I.N.A. 3.3: curve-of-growth HFR, FWHM and eccentricity in pixels.
    Nina33,
}

impl fmt::Display for MeasurementVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Nina31 => "3.1",
            Self::Nina32 => "3.2",
            Self::Nina33 => "3.3",
        })
    }
}

/// A measurement version other than `3.1`, `3.2` or `3.3`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownMeasurementVersion(pub String);

impl fmt::Display for UnknownMeasurementVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown N.I.N.A. measurement version {:?}: expected 3.1, 3.2 or 3.3",
            self.0
        )
    }
}

impl std::error::Error for UnknownMeasurementVersion {}

impl FromStr for MeasurementVersion {
    type Err = UnknownMeasurementVersion;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "3.1" => Ok(Self::Nina31),
            "3.2" => Ok(Self::Nina32),
            "3.3" => Ok(Self::Nina33),
            other => Err(UnknownMeasurementVersion(other.to_string())),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Rectangle {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

/// Detected star result
#[derive(Debug, Clone)]
pub struct DetectedStar {
    /// Half-flux radius in pixels. Under 3.3 it is NaN when the star has no
    /// flux above the background, and the frame's average leaves it out.
    pub hfr: f64,
    pub position: (f64, f64),
    pub average_brightness: f64,
    /// Brightest raw pixel inside the blob's circle.
    pub max_brightness: f64,
    pub background: f64,
    /// Background-subtracted flux in the original image's units, measured
    /// inside the same full-resolution aperture used for HFR.
    pub flux: f64,
    /// Full width at half maximum in pixels, from 3.3's measurement.
    pub fwhm: Option<f64>,
    /// Eccentricity of the flux-weighted second moments, from 3.3's
    /// measurement: 0 for a round star.
    pub eccentricity: Option<f64>,
}

/// Star detection result
#[derive(Debug, Clone)]
pub struct StarDetectionResult {
    pub average_hfr: f64,
    pub hfr_std_dev: f64,
    pub star_list: Vec<DetectedStar>,
    /// Mean FWHM in pixels of the stars that have one, from 3.3's
    /// measurement.
    pub average_fwhm: Option<f64>,
    /// Mean eccentricity of the stars that have one, from 3.3's
    /// measurement.
    pub average_eccentricity: Option<f64>,
}

const MAX_WIDTH: usize = 1552;

/// Detection state: the resize and blob size limits.
struct DetectionState<'a> {
    original_data: &'a [u16],
    width: usize,
    height: usize,
    resize_factor: f64,
    inverse_resize_factor: f64,
    min_star_size: i32,
    max_star_size: i32,
}

/// A candidate star. Positions are single precision, as Accord's `Point`.
#[derive(Debug, Clone)]
struct Star {
    position: (f32, f32),
    radius: f64,
    rectangle: Rectangle,
    mean_brightness: f64,
    surrounding_mean: f64,
    max_pixel_value: f64,
    hfr: f64,
    fwhm: f64,
    eccentricity: f64,
    average: f64,
    flux: f64,
}

/// Find and measure stars as N.I.N.A. does.
///
/// `detection_data_16bit` is the frame N.I.N.A. renders, which is
/// autostretched whenever it detects stars; edges are found on it.
/// `original_data_16bit` is the raw frame, on which candidates are tested
/// and measured. Both are `width` × `height`, row by row.
pub fn detect_stars_with_original(
    detection_data_16bit: &[u16],
    original_data_16bit: &[u16],
    width: usize,
    height: usize,
    params: &StarDetectionParams,
) -> StarDetectionResult {
    let state = initial_state(original_data_16bit, width, height, params);

    crate::debug_detection!(
        "N.I.N.A. detection: image {}x{}, resize factor {:.3}, star size {}..{}",
        width,
        height,
        state.resize_factor,
        state.min_star_size,
        state.max_star_size
    );

    // Accord's Convert16bppTo8bpp keeps the high byte.
    let mut bitmap = {
        use rayon::prelude::*;
        detection_data_16bit
            .par_iter()
            .map(|&pixel| (pixel >> 8) as u8)
            .collect::<Vec<u8>>()
    };

    // N.I.N.A. reduces noise only on images it is about to shrink.
    if params.noise_reduction != NoiseReduction::None && width > MAX_WIDTH {
        bitmap = reduce_noise(&bitmap, width, height, params.noise_reduction);
    }

    let (mut bitmap, resized_width, resized_height) = DetectionUtility::resize_for_detection(
        &bitmap,
        width,
        height,
        MAX_WIDTH,
        state.resize_factor,
    );

    prepare_for_structure_detection(&mut bitmap, resized_width, resized_height, params);

    let mut blobs = BlobCounter::new();
    blobs.process_image(&bitmap, resized_width, resized_height);

    crate::debug_detection!(
        "N.I.N.A. detection found {} blobs",
        blobs.get_objects_information().len()
    );

    let star_list = identify_stars(params, &state, &blobs, resized_width, resized_height);

    let mut result = StarDetectionResult {
        average_hfr: 0.0,
        hfr_std_dev: 0.0,
        star_list,
        average_fwhm: None,
        average_eccentricity: None,
    };
    let hfrs = result
        .star_list
        .iter()
        .map(|s| s.hfr)
        .filter(|hfr| !hfr.is_nan())
        .collect::<Vec<_>>();
    if !hfrs.is_empty() {
        let mean = hfrs.iter().sum::<f64>() / hfrs.len() as f64;
        result.average_hfr = mean;
        if hfrs.len() > 1 {
            let squares = hfrs.iter().map(|hfr| (hfr - mean).powi(2)).sum::<f64>();
            result.hfr_std_dev = (squares / (hfrs.len() - 1) as f64).sqrt();
        }
    }
    let mean_of = |values: Vec<f64>| {
        (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
    };
    result.average_fwhm = mean_of(result.star_list.iter().filter_map(|s| s.fwhm).collect());
    result.average_eccentricity = mean_of(
        result
            .star_list
            .iter()
            .filter_map(|s| s.eccentricity)
            .collect(),
    );
    result
}

fn initial_state<'a>(
    original_data: &'a [u16],
    width: usize,
    height: usize,
    params: &StarDetectionParams,
) -> DetectionState<'a> {
    let max_fraction = MAX_WIDTH as f64 / width as f64;
    let resize_factor = if width <= MAX_WIDTH {
        1.0
    } else {
        match params.sensitivity {
            StarSensitivity::Highest => f64::max(2.0 / 3.0, max_fraction),
            StarSensitivity::High => match params.image_scale_arcsec.filter(|s| !s.is_nan()) {
                None => f64::max(0.5, max_fraction),
                // N.I.N.A.'s cascade: the last test that matches wins.
                Some(scale) => {
                    let mut factor = 1.0;
                    if scale < 0.5 {
                        factor = 0.25;
                    }
                    if (0.5..=1.5).contains(&scale) {
                        factor = 1.0 / 3.0;
                    }
                    if (1.5..=2.5).contains(&scale) {
                        factor = 0.5;
                    }
                    if scale > 1.5 {
                        factor = f64::max(2.0 / 3.0, max_fraction);
                    }
                    factor
                }
            },
            StarSensitivity::Normal => max_fraction,
        }
    };
    DetectionState {
        original_data,
        width,
        height,
        resize_factor,
        inverse_resize_factor: 1.0 / resize_factor,
        min_star_size: ((5.0 * resize_factor).floor() as i32).max(2),
        max_star_size: (150.0 * resize_factor).ceil() as i32,
    }
}

fn reduce_noise(
    image: &[u8],
    width: usize,
    height: usize,
    noise_reduction: NoiseReduction,
) -> Vec<u8> {
    match noise_reduction {
        NoiseReduction::None => image.to_vec(),
        NoiseReduction::Normal => accord_imaging::fast_gaussian_blur(image, width, height, 1),
        NoiseReduction::High => accord_imaging::fast_gaussian_blur(image, width, height, 2),
        NoiseReduction::Highest => accord_imaging::fast_gaussian_blur(image, width, height, 3),
        NoiseReduction::Median => accord_imaging::median_3x3(image, width, height),
    }
}

fn prepare_for_structure_detection(
    image: &mut [u8],
    width: usize,
    height: usize,
    params: &StarDetectionParams,
) {
    // A Gaussian noise reduction has already blurred the image; otherwise
    // Canny blurs it first.
    let blur = matches!(
        params.noise_reduction,
        NoiseReduction::None | NoiseReduction::Median
    );
    CannyEdgeDetector {
        low_threshold: 10,
        high_threshold: 80,
        blur,
    }
    .apply_in_place(image, width, height);
    let threshold = accord_imaging::sis_threshold(image, width, height);
    crate::debug_detection!("N.I.N.A. detection: SIS threshold {}", threshold);
    let dilated = accord_imaging::binary_dilation_3x3(image, width, height);
    image.copy_from_slice(&dilated);
}

fn identify_stars(
    params: &StarDetectionParams,
    state: &DetectionState,
    blobs: &BlobCounter,
    bitmap_width: usize,
    bitmap_height: usize,
) -> Vec<DetectedStar> {
    let v33 = params.measurement == MeasurementVersion::Nina33;
    let shape_checker = SimpleShapeChecker;
    let inverse = state.inverse_resize_factor;
    let (image_width, image_height) = (state.width as i32, state.height as i32);
    let minimum_bright_pixels = (state.width.max(state.height) as f64 / 1000.0).ceil() as usize;
    let mut star_list = Vec::new();
    let mut sum_radius = 0.0;
    let mut sum_squares = 0.0;
    let mut rejected = [0usize; 5];

    for (index, blob) in blobs.get_objects_information().iter().enumerate() {
        let b = blob.rectangle;
        if b.width > state.max_star_size
            || b.height > state.max_star_size
            || b.width < state.min_star_size
            || b.height < state.min_star_size
        {
            rejected[0] += 1;
            continue;
        }
        if params.use_roi
            && !DetectionUtility::in_roi(
                bitmap_width,
                bitmap_height,
                &b,
                params.outer_crop_ratio,
                params.inner_crop_ratio,
            )
        {
            continue;
        }

        let points = blobs.edge_points(index);
        let rect = Rectangle {
            x: (b.x as f64 * inverse).floor() as i32,
            y: (b.y as f64 * inverse).floor() as i32,
            width: (b.width as f64 * inverse).ceil() as i32,
            height: (b.height as f64 * inverse).ceil() as i32,
        };
        let large_x = (rect.x - rect.width).max(0);
        let large_y = (rect.y - rect.height).max(0);
        let large_rect = Rectangle {
            x: large_x,
            y: large_y,
            width: (rect.width * 3).min(image_width - large_x),
            height: (rect.height * 3).min(image_height - large_y),
        };

        let (mut cx, mut cy, mut r) = (0.0f32, 0.0f32, 0.0f32);
        let circle = shape_checker.is_circle(&points, &mut cx, &mut cy, &mut r);
        // A shape that is not a circle keeps the checker's centre, which is
        // the origin for fewer than 8 edge points: such a candidate then
        // finds no pixels in its circle and is dropped.
        let position = (cx * inverse as f32, cy * inverse as f32);
        let radius = if circle {
            r as f64 * inverse
        } else {
            if box_eccentricity(rect.width as f64, rect.height as f64) > 0.8 {
                rejected[1] += 1;
                continue;
            }
            // Integer division in N.I.N.A.
            (rect.width.max(rect.height) / 2) as f64
        };
        let mut star = Star {
            position,
            radius,
            rectangle: rect,
            mean_brightness: 0.0,
            surrounding_mean: 0.0,
            max_pixel_value: 0.0,
            hfr: 0.0,
            fwhm: f64::NAN,
            eccentricity: f64::NAN,
            average: 0.0,
            flux: 0.0,
        };

        // Pixels of the large box, column by column as N.I.N.A. reads them.
        let mut star_pixel_sum = 0.0;
        let mut star_pixel_count = 0usize;
        let mut inner_star_pixels = Vec::new();
        let mut background_sum = 0.0;
        let mut background_sum_squares = 0.0;
        let mut background_pixels = Vec::new();
        let mut pixel_data = Vec::new();
        let (px, py) = (star.position.0 as f64, star.position.1 as f64);
        for x in large_rect.x..large_rect.x + large_rect.width {
            for y in large_rect.y..large_rect.y + large_rect.height {
                let value = state.original_data[x as usize + state.width * y as usize];
                let in_rect = x >= rect.x
                    && x < rect.x + rect.width
                    && y >= rect.y
                    && y < rect.y + rect.height;
                if in_rect {
                    if inside_circle(x as f64, y as f64, px, py, star.radius) {
                        star_pixel_sum += value as f64;
                        star_pixel_count += 1;
                        inner_star_pixels.push(value);
                        star.max_pixel_value = star.max_pixel_value.max(value as f64);
                    }
                    if !v33 {
                        pixel_data.push((x, y, value));
                    }
                } else if v33 {
                    background_pixels.push(value as f64);
                } else {
                    background_sum += value as f64;
                    // ushort * ushort is an int in C#, which wraps above
                    // 46340².
                    background_sum_squares += (value as i32).wrapping_mul(value as i32) as f64;
                }
                if v33 {
                    pixel_data.push((x, y, value));
                }
            }
        }

        let (background, sigma) = if v33 {
            if star_pixel_count == 0 || background_pixels.is_empty() {
                continue;
            }
            estimate_background(background_pixels)
        } else {
            let count = (large_rect.height * large_rect.width - rect.height * rect.width) as f64;
            let mean = background_sum / count;
            let stdev = ((background_sum_squares - count * mean * mean) / count).sqrt();
            (mean, stdev)
        };
        // With no pixel in the circle this is NaN and the test below fails,
        // as in N.I.N.A. 3.2.
        star.mean_brightness = star_pixel_sum / star_pixel_count as f64;
        star.surrounding_mean = background;

        let bright = inner_star_pixels
            .iter()
            .filter(|&&value| value as f64 > background + 1.5 * sigma)
            .count();
        if !(star.mean_brightness >= background + (0.1 * background).min(sigma)
            && bright > minimum_bright_pixels)
        {
            if star_pixel_count == 0 {
                rejected[2] += 1;
            } else {
                rejected[3] += 1;
            }
            continue;
        }

        match params.measurement {
            MeasurementVersion::Nina31 => {
                // 3.1 measures about the blob's centre, keeps the star
                // whatever its centroid, and counts its radius.
                measure_v32(&mut star, &pixel_data, false);
                sum_radius += star.radius;
                sum_squares += star.radius * star.radius;
                star_list.push(star);
                continue;
            }
            MeasurementVersion::Nina32 => measure_v32(&mut star, &pixel_data, true),
            MeasurementVersion::Nina33 => measure_v33(&mut star, &pixel_data),
        }

        // Keep only stars whose centroid is off the box's edges. N.I.N.A.
        // 3.2 compares the far edges with the position itself, so those two
        // tests pass for any box at least 3 pixels across.
        let (x, y) = (star.position.0 as f64, star.position.1 as f64);
        let (w, h) = (rect.width as f64, rect.height as f64);
        let inside = if v33 {
            x > (rect.x + 1) as f64
                && y > (rect.y + 1) as f64
                && x < rect.x as f64 + w - 2.0
                && y < rect.y as f64 + h - 2.0
        } else {
            x > (rect.x + 1) as f64 && y > (rect.y + 1) as f64 && x < x + w - 2.0 && y < y + h - 2.0
        };
        if inside {
            sum_radius += star.radius;
            sum_squares += star.radius * star.radius;
            star_list.push(star);
        } else {
            rejected[4] += 1;
        }
    }
    crate::debug_detection!(
        "N.I.N.A. rejected: size {}, elongated {}, empty circle {}, faint {}, edge {}",
        rejected[0],
        rejected[1],
        rejected[2],
        rejected[3],
        rejected[4]
    );

    if star_list.is_empty() {
        return Vec::new();
    }

    let count = star_list.len() as f64;
    let avg = sum_radius / count;
    let stdev = ((sum_squares - count * avg * avg) / count).sqrt();
    crate::debug_detection!(
        "N.I.N.A. radius filter: {} stars, average {:.2}, standard deviation {:.2}",
        star_list.len(),
        avg,
        stdev
    );
    let upper = if params.sensitivity == StarSensitivity::Highest {
        // More permissive towards large stars.
        2.0
    } else {
        1.5
    };
    star_list.retain(|s| s.radius <= avg + upper * stdev && s.radius >= avg - 1.5 * stdev);
    crate::debug_detection!("N.I.N.A. radius filter kept {} stars", star_list.len());

    star_list
        .into_iter()
        .map(|s| DetectedStar {
            hfr: s.hfr,
            position: (s.position.0 as f64, s.position.1 as f64),
            average_brightness: s.average,
            max_brightness: s.max_pixel_value,
            background: s.surrounding_mean,
            flux: s.flux,
            fwhm: (!s.fwhm.is_nan()).then_some(s.fwhm),
            eccentricity: (!s.eccentricity.is_nan()).then_some(s.eccentricity),
        })
        .collect()
}

/// N.I.N.A. 3.2's `Star.Calculate`: first-moment HFR of the
/// background-subtracted pixels within 1.2 radii, and, when `centroid` is
/// set, a centroid from the same pixels. 3.1's `CalculateHfr` is the same
/// without the centroid.
fn measure_v32(star: &mut Star, pixel_data: &[(i32, i32, u16)], centroid: bool) {
    if pixel_data.is_empty() {
        star.hfr = 0.0;
        return;
    }
    let outer_radius = star.radius * 1.2;
    let (cx, cy) = (star.position.0 as f64, star.position.1 as f64);
    let (mut sum, mut sum_dist, mut all_sum) = (0.0, 0.0, 0.0);
    let (mut sum_x, mut sum_y) = (0.0, 0.0);
    for &(x, y, value) in pixel_data {
        // Math.Round rounds half to even.
        let value = (value as f64 - star.surrounding_mean)
            .round_ties_even()
            .max(0.0);
        all_sum += value;
        if inside_circle(x as f64, y as f64, cx, cy, outer_radius) {
            sum += value;
            sum_dist += value * ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
            sum_x += (x - star.rectangle.x) as f64 * value;
            sum_y += (y - star.rectangle.y) as f64 * value;
        }
    }
    star.hfr = if sum > 0.0 {
        sum_dist / sum
    } else {
        std::f64::consts::SQRT_2 * outer_radius
    };
    star.flux = sum;
    star.average = all_sum / pixel_data.len() as f64;
    if centroid && sum != 0.0 {
        star.position = (
            (sum_x / sum) as f32 + star.rectangle.x as f32,
            (sum_y / sum) as f32 + star.rectangle.y as f32,
        );
    }
}

/// Out to 1.5 times the blob's radius in N.I.N.A. 3.3.
const MEASUREMENT_RADIUS_SCALE: f64 = 1.5;
/// 3.3's radial profile bins, in pixels.
const RADIAL_PROFILE_BIN_WIDTH: f64 = 0.5;
const CENTROID_MAX_ITERATIONS: usize = 3;
const CENTROID_CONVERGENCE_PIXELS: f32 = 0.01;

/// One background-subtracted pixel within 3.3's measurement radius.
struct RadialSample {
    distance: f64,
    raw_flux: f64,
    positive_flux: f64,
    x: i32,
    y: i32,
}

/// N.I.N.A. 3.3's `Star.Calculate`.
fn measure_v33(star: &mut Star, pixel_data: &[(i32, i32, u16)]) {
    star.hfr = f64::NAN;
    star.fwhm = f64::NAN;
    star.eccentricity = f64::NAN;
    star.average = f64::NAN;
    if pixel_data.is_empty() {
        return;
    }
    let centroid_radius = measurement_radius(star, star.position, pixel_data);
    let (center, total_flux) = windowed_centroid(star, pixel_data, centroid_radius);
    if total_flux <= 0.0 {
        return;
    }
    star.position = center;

    let radius = measurement_radius(star, star.position, pixel_data);
    let (cx, cy) = (star.position.0 as f64, star.position.1 as f64);
    let samples = pixel_data
        .iter()
        .filter_map(|&(x, y, value)| {
            let (dx, dy) = (x as f64 - cx, y as f64 - cy);
            let distance = (dx * dx + dy * dy).sqrt();
            (distance <= radius).then(|| {
                let flux = value as f64 - star.surrounding_mean;
                RadialSample {
                    distance,
                    raw_flux: flux,
                    positive_flux: flux.max(0.0),
                    x,
                    y,
                }
            })
        })
        .collect::<Vec<_>>();
    if samples.is_empty() {
        return;
    }
    let total_flux = samples.iter().map(|s| s.positive_flux).sum::<f64>();
    if total_flux <= 0.0 {
        return;
    }
    let positive = samples.iter().filter(|s| s.positive_flux > 0.0).count();
    star.average = total_flux / positive as f64;
    star.flux = total_flux;
    star.hfr = half_flux_radius(&samples, total_flux);
    star.fwhm = radial_fwhm(&samples, star.max_pixel_value - star.surrounding_mean);
    star.eccentricity = moment_eccentricity(&samples, star.position);
}

/// 3.3's `GetMeasurementRadius`: 1.5 radii, at least 2, but no further than
/// the nearest edge of the measured box less half a pixel, and at least 1.
fn measurement_radius(star: &Star, center: (f32, f32), pixel_data: &[(i32, i32, u16)]) -> f64 {
    let (mut min_x, mut max_x) = (f64::MAX, f64::MIN);
    let (mut min_y, mut max_y) = (f64::MAX, f64::MIN);
    for &(x, y, _) in pixel_data {
        min_x = min_x.min(x as f64);
        max_x = max_x.max(x as f64);
        min_y = min_y.min(y as f64);
        max_y = max_y.max(y as f64);
    }
    let (cx, cy) = (center.0 as f64, center.1 as f64);
    let available = (cx - min_x)
        .min(max_x - cx)
        .min((cy - min_y).min(max_y - cy))
        - 0.5;
    let requested = (star.radius * MEASUREMENT_RADIUS_SCALE).max(2.0);
    requested.min(available).max(1.0)
}

/// 3.3's `CalculateCentroid`: a Gaussian-windowed centroid of the pixels
/// above background within `radius`, iterated up to three times. Returns the
/// centre and the unweighted flux of the last pass.
fn windowed_centroid(
    star: &Star,
    pixel_data: &[(i32, i32, u16)],
    radius: f64,
) -> ((f32, f32), f64) {
    let sigma = (radius / 2.0).max(1.0);
    let radius_squared = radius * radius;
    let neg_half_inv_sigma_squared = -0.5 / (sigma * sigma);
    let mut center = star.position;
    let mut last_total_flux = 0.0;
    for _ in 0..CENTROID_MAX_ITERATIONS {
        let (cx, cy) = (center.0 as f64, center.1 as f64);
        let (mut sum_weighted, mut sum_x, mut sum_y, mut sum_flux) = (0.0, 0.0, 0.0, 0.0);
        let mut any = false;
        for &(x, y, value) in pixel_data {
            let (dx, dy) = (x as f64 - cx, y as f64 - cy);
            let distance_squared = dx * dx + dy * dy;
            if distance_squared > radius_squared {
                continue;
            }
            let flux = value as f64 - star.surrounding_mean;
            if flux <= 0.0 {
                continue;
            }
            let weighted = flux * (distance_squared * neg_half_inv_sigma_squared).exp();
            sum_weighted += weighted;
            sum_x += x as f64 * weighted;
            sum_y += y as f64 * weighted;
            sum_flux += flux;
            any = true;
        }
        if !any || sum_weighted <= 0.0 {
            return (center, 0.0);
        }
        let refined = ((sum_x / sum_weighted) as f32, (sum_y / sum_weighted) as f32);
        last_total_flux = sum_flux;
        let (dx, dy) = (refined.0 - center.0, refined.1 - center.1);
        let moved = ((dx * dx + dy * dy) as f64).sqrt() as f32;
        if moved <= CENTROID_CONVERGENCE_PIXELS {
            return (refined, last_total_flux);
        }
        center = refined;
    }
    (center, last_total_flux)
}

/// 3.3's curve-of-growth HFR: the radius, interpolated between pixels, at
/// which the positive flux sorted by distance reaches half its total.
fn half_flux_radius(samples: &[RadialSample], total_flux: f64) -> f64 {
    let target = total_flux / 2.0;
    let (mut cumulative, mut previous_flux, mut previous_radius) = (0.0, 0.0, 0.0);
    // LINQ's OrderBy is stable, and so is this sort.
    let mut order = samples.iter().collect::<Vec<_>>();
    order.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    for sample in order {
        if sample.positive_flux <= 0.0 {
            continue;
        }
        cumulative += sample.positive_flux;
        if cumulative < target {
            previous_flux = cumulative;
            previous_radius = sample.distance;
            continue;
        }
        if cumulative == previous_flux || sample.distance <= previous_radius {
            return sample.distance;
        }
        let fraction = (target - previous_flux) / (cumulative - previous_flux);
        return previous_radius + fraction * (sample.distance - previous_radius);
    }
    samples.iter().map(|s| s.distance).fold(f64::MIN, f64::max)
}

/// 3.3's FWHM: twice the radius where the azimuthally averaged profile, in
/// half-pixel bins and never rising outward, falls to half the peak.
fn radial_fwhm(samples: &[RadialSample], peak_flux: f64) -> f64 {
    if peak_flux <= 0.0 {
        return f64::NAN;
    }
    let half_maximum = peak_flux / 2.0;
    let max_distance = samples.iter().map(|s| s.distance).fold(f64::MIN, f64::max);
    let bins = (max_distance / RADIAL_PROFILE_BIN_WIDTH).ceil() as usize + 1;
    let mut sums = vec![0.0; bins];
    let mut distance_sums = vec![0.0; bins];
    let mut counts = vec![0usize; bins];
    for sample in samples {
        let index = ((sample.distance / RADIAL_PROFILE_BIN_WIDTH).floor() as usize).min(bins - 1);
        sums[index] += sample.raw_flux;
        distance_sums[index] += sample.distance;
        counts[index] += 1;
    }
    let (mut previous_radius, mut previous_value) = (0.0, peak_flux);
    for i in 0..bins {
        if counts[i] == 0 {
            continue;
        }
        let radius = distance_sums[i] / counts[i] as f64;
        let mut value = sums[i] / counts[i] as f64;
        if i > 0 && value > previous_value {
            value = previous_value;
        }
        if value <= half_maximum {
            // C#'s double.Epsilon is the smallest subnormal.
            if (value - previous_value).abs() < f64::from_bits(1) || radius <= previous_radius {
                return 2.0 * radius;
            }
            let fraction = (half_maximum - previous_value) / (value - previous_value);
            return 2.0 * (previous_radius + fraction * (radius - previous_radius));
        }
        previous_radius = radius;
        previous_value = value;
    }
    f64::NAN
}

/// 3.3's eccentricity, `sqrt(1 − λ_minor / λ_major)` of the positive-flux
/// weighted second central moments.
fn moment_eccentricity(samples: &[RadialSample], center: (f32, f32)) -> f64 {
    let (cx, cy) = (center.0 as f64, center.1 as f64);
    let (mut total, mut xx, mut yy, mut xy) = (0.0, 0.0, 0.0, 0.0);
    for sample in samples.iter().filter(|s| s.positive_flux > 0.0) {
        let (dx, dy) = (sample.x as f64 - cx, sample.y as f64 - cy);
        let f = sample.positive_flux;
        total += f;
        xx += f * dx * dx;
        yy += f * dy * dy;
        xy += f * dx * dy;
    }
    if total <= 0.0 {
        return f64::NAN;
    }
    let (xx, yy, xy) = (xx / total, yy / total, xy / total);
    let trace = xx + yy;
    let root = ((xx - yy) * (xx - yy) + 4.0 * xy * xy).max(0.0).sqrt();
    let major = (trace + root) / 2.0;
    let minor = ((trace - root) / 2.0).max(0.0);
    if major <= 0.0 {
        return f64::NAN;
    }
    (1.0 - minor / major).max(0.0).sqrt()
}

/// 3.3's `EstimateBackground`: the median of the pixels after up to three
/// rounds of 3σ clipping, with σ from the MAD (or the spread about the
/// median when the MAD is 0). Returns the background and σ.
fn estimate_background(mut values: Vec<f64>) -> (f64, f64) {
    fn median_of_sorted(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let middle = values.len() / 2;
        if values.len().is_multiple_of(2) {
            (values[middle - 1] + values[middle]) / 2.0
        } else {
            values[middle]
        }
    }
    fn spread(values: &[f64], center: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let squares = values
            .iter()
            .map(|v| (v - center) * (v - center))
            .sum::<f64>();
        (squares / values.len() as f64).sqrt()
    }

    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mut iteration = 0;
    while iteration < 3 && values.len() > 1 {
        values.sort_by(f64::total_cmp);
        let median = median_of_sorted(&values);
        let mut deviations = values
            .iter()
            .map(|v| (v - median).abs())
            .collect::<Vec<_>>();
        deviations.sort_by(f64::total_cmp);
        let mad = median_of_sorted(&deviations);
        let sigma = if mad > 0.0 {
            1.4826 * mad
        } else {
            spread(&values, median)
        };
        if sigma <= 0.0 {
            return (median, 0.0);
        }
        let clipped = values
            .iter()
            .copied()
            .filter(|v| (v - median).abs() <= 3.0 * sigma)
            .collect::<Vec<_>>();
        if clipped.len() == values.len() || clipped.is_empty() {
            return (median, sigma);
        }
        values = clipped;
        iteration += 1;
    }
    values.sort_by(f64::total_cmp);
    let median = median_of_sorted(&values);
    (median, spread(&values, median))
}

fn inside_circle(x: f64, y: f64, center_x: f64, center_y: f64, radius: f64) -> bool {
    (x - center_x).powi(2) + (y - center_y).powi(2) <= radius.powi(2)
}

/// N.I.N.A.'s eccentricity of a bounding box, used to drop elongated blobs.
fn box_eccentricity(width: f64, height: f64) -> f64 {
    let x = width.max(height);
    let y = width.min(height);
    (x.powi(2) - y.powi(2)).sqrt() / x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inside_circle() {
        assert!(inside_circle(0.0, 0.0, 0.0, 0.0, 1.0));
        assert!(inside_circle(0.5, 0.5, 0.0, 0.0, 1.0));
        assert!(!inside_circle(1.5, 0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_eccentricity() {
        assert_eq!(box_eccentricity(10.0, 10.0), 0.0);
        assert!(box_eccentricity(10.0, 5.0) > 0.8);
    }

    #[test]
    fn measurement_version_reads_and_writes_its_number() {
        assert_eq!("3.1".parse(), Ok(MeasurementVersion::Nina31));
        assert_eq!("3.2".parse(), Ok(MeasurementVersion::Nina32));
        assert_eq!("3.3".parse(), Ok(MeasurementVersion::Nina33));
        assert!("3.4".parse::<MeasurementVersion>().is_err());
        assert_eq!(MeasurementVersion::Nina33.to_string(), "3.3");
        assert_eq!(MeasurementVersion::default(), MeasurementVersion::Nina32);
    }

    fn star(position: (f32, f32), radius: f64, rectangle: Rectangle, background: f64) -> Star {
        Star {
            position,
            radius,
            rectangle,
            mean_brightness: 0.0,
            surrounding_mean: background,
            max_pixel_value: 0.0,
            hfr: 0.0,
            fwhm: f64::NAN,
            eccentricity: f64::NAN,
            average: 0.0,
            flux: 0.0,
        }
    }

    #[test]
    fn hfr_measurement_keeps_background_subtracted_flux() {
        let rect = Rectangle {
            x: 2,
            y: 2,
            width: 3,
            height: 3,
        };
        let mut pixels = Vec::new();
        for x in 2..5 {
            for y in 2..5 {
                pixels.push((x, y, if (x, y) == (3, 3) { 200 } else { 100 }));
            }
        }
        let mut measured = star((3.0, 3.0), 1.0, rect, 100.0);
        measure_v32(&mut measured, &pixels, true);
        assert_eq!(measured.flux, 100.0);
        assert_eq!(measured.position, (3.0, 3.0));
        assert_eq!(measured.hfr, 0.0);
    }

    /// A Gaussian star of `sigma` pixels centred at (`cx`, `cy`) in a
    /// `size` × `size` box on a sky of 1000, as 3.3 measures it.
    fn gaussian_box(size: i32, cx: f64, cy: f64, sigma: f64, peak: f64) -> Vec<(i32, i32, u16)> {
        let mut pixels = Vec::new();
        for x in 0..size {
            for y in 0..size {
                let d2 = (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2);
                let value = 1000.0 + peak * (-d2 / (2.0 * sigma * sigma)).exp();
                pixels.push((x, y, value.round() as u16));
            }
        }
        pixels
    }

    #[test]
    fn nina33_measures_a_gaussian_star() {
        // HFR of a Gaussian is σ·sqrt(2 ln 2) ≈ 1.177σ and its FWHM 2.355σ.
        // 3.3 integrates only out to 1.5 radii, so with a radius of 2σ the
        // aperture holds 1 − e^−4.5 ≈ 99% of the flux.
        let sigma = 2.0;
        let pixels = gaussian_box(41, 20.3, 19.6, sigma, 30000.0);
        let rect = Rectangle {
            x: 14,
            y: 14,
            width: 13,
            height: 13,
        };
        let mut measured = star((20.0, 20.0), 2.0 * sigma, rect, 1000.0);
        measured.max_pixel_value = 31000.0;
        measure_v33(&mut measured, &pixels);
        assert!(
            (measured.position.0 - 20.3).abs() < 0.02,
            "{:?}",
            measured.position
        );
        assert!(
            (measured.position.1 - 19.6).abs() < 0.02,
            "{:?}",
            measured.position
        );
        let hfr = 1.177 * sigma;
        assert!(
            (measured.hfr - hfr).abs() / hfr < 0.04,
            "HFR {}",
            measured.hfr
        );
        let fwhm = 2.355 * sigma;
        assert!(
            (measured.fwhm - fwhm).abs() / fwhm < 0.06,
            "FWHM {}",
            measured.fwhm
        );
        assert!(measured.eccentricity < 0.15, "{}", measured.eccentricity);
    }

    #[test]
    fn nina33_eccentricity_grows_with_elongation() {
        let mut pixels = Vec::new();
        for x in 0..41 {
            for y in 0..41 {
                let (dx, dy) = ((x as f64 - 20.0) / 3.0, (y as f64 - 20.0) / 1.5);
                let value = 1000.0 + 30000.0 * (-(dx * dx + dy * dy) / 2.0).exp();
                pixels.push((x, y, value.round() as u16));
            }
        }
        let rect = Rectangle {
            x: 12,
            y: 12,
            width: 17,
            height: 17,
        };
        let mut measured = star((20.0, 20.0), 6.0, rect, 1000.0);
        measured.max_pixel_value = 31000.0;
        measure_v33(&mut measured, &pixels);
        // Axis ratio 2: e = sqrt(1 − 1/4) ≈ 0.87.
        assert!(
            (measured.eccentricity - 0.866).abs() < 0.05,
            "{}",
            measured.eccentricity
        );
    }

    #[test]
    fn background_is_a_clipped_median() {
        let mut values = vec![100.0; 50];
        values.extend([101.0; 30]);
        values.extend([99.0; 30]);
        values.extend([5000.0; 5]);
        let (background, sigma) = estimate_background(values);
        assert_eq!(background, 100.0);
        assert!(sigma < 1.5, "{sigma}");
    }

    /// A field of Gaussian stars of the given σ on a noisy sky, wider than
    /// N.I.N.A.'s detection width so it is shrunk as real frames are.
    fn star_field(width: usize, height: usize, sigma: f64) -> Vec<u16> {
        let mut seed = 11u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut image = (0..width * height)
            .map(|_| 1000.0 + 20.0 * (next() - 0.5))
            .collect::<Vec<f64>>();
        let reach = (6.0 * sigma) as i64;
        for row in 0..12 {
            for column in 0..20 {
                let cx = 60.0 + column as f64 * (width as f64 - 120.0) / 19.0 + next();
                let cy = 60.0 + row as f64 * (height as f64 - 120.0) / 11.0 + next();
                let peak = 8000.0 + 20000.0 * next();
                for y in (cy as i64 - reach).max(0)..(cy as i64 + reach).min(height as i64) {
                    for x in (cx as i64 - reach).max(0)..(cx as i64 + reach).min(width as i64) {
                        let d2 = (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2);
                        image[y as usize * width + x as usize] +=
                            peak * (-d2 / (2.0 * sigma * sigma)).exp();
                    }
                }
            }
        }
        image.into_iter().map(|v| v.min(65535.0) as u16).collect()
    }

    /// N.I.N.A.'s autostretch, for the detection image.
    fn stretched(data: &[u16]) -> Vec<u16> {
        let mut sorted = data.to_vec();
        sorted.sort_unstable();
        let median = sorted[sorted.len() / 2] as f64 / 65535.0;
        let mut deviations = sorted
            .iter()
            .map(|&v| (v as f64 / 65535.0 - median).abs())
            .collect::<Vec<_>>();
        deviations.sort_by(f64::total_cmp);
        let mad = deviations[deviations.len() / 2];
        let mtf = |m: f64, x: f64| {
            if x <= 0.0 {
                0.0
            } else if x >= 1.0 {
                1.0
            } else {
                (m - 1.0) * x / ((2.0 * m - 1.0) * x - m)
            }
        };
        let shadows = median - 2.8 * 1.4826 * mad;
        let midtone = mtf(0.2, median - shadows);
        data.iter()
            .map(|&v| (mtf(midtone, v as f64 / 65535.0 - shadows) * 65535.0) as u16)
            .collect()
    }

    #[test]
    fn detects_a_shrunk_star_field_under_both_measurements() {
        let (width, height) = (3200, 2000);
        let sigma = 3.0;
        let data = star_field(width, height, sigma);
        let detection = stretched(&data);
        let v32 = detect_stars_with_original(
            &detection,
            &data,
            width,
            height,
            &StarDetectionParams::default(),
        );
        let v33 = detect_stars_with_original(
            &detection,
            &data,
            width,
            height,
            &StarDetectionParams {
                measurement: MeasurementVersion::Nina33,
                ..StarDetectionParams::default()
            },
        );
        // Most of the 240 stars are found; the radius filter drops those
        // whose blob is far from the average size.
        assert!(v32.star_list.len() > 160, "{}", v32.star_list.len());
        assert!(v33.star_list.len() > 160, "{}", v33.star_list.len());
        assert!(v32.average_fwhm.is_none());
        // 3.3's HFR is close to a Gaussian's; 3.2's first moment over a
        // wider box reads larger.
        let hfr = 1.177 * sigma;
        assert!(
            (v33.average_hfr - hfr).abs() / hfr < 0.08,
            "{}",
            v33.average_hfr
        );
        assert!(v32.average_hfr > v33.average_hfr, "{}", v32.average_hfr);
        let fwhm = v33.average_fwhm.unwrap();
        assert!(
            (fwhm - 2.355 * sigma).abs() / (2.355 * sigma) < 0.1,
            "{fwhm}"
        );
        assert!(v33.average_eccentricity.unwrap() < 0.3);
    }
}
