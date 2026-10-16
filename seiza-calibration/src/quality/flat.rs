//! Flat frames: exposure, illumination pattern, and compact dips.
//!
//! A flat records how much light each pixel receives, so anything that blocks
//! light while it is shot ends up in every light it corrects. Dust shadows
//! belong there: they sit in the light path for the whole session and the
//! flat is meant to divide them out. A shadow that changes while the flats are
//! shot does not: no single flat matches the lights, and a master combined
//! with a median keeps whatever most of its frames showed.
//!
//! [`measure_flat`] finds the flat's dust as [`crate::dust`] defines it:
//! compact dips at full resolution — small patches that pass less light than
//! the response around them — and, on the frame binned to about 1,024
//! pixels, the large out-of-focus disks and donuts of dust further up the
//! train, which are too wide and too shallow to see pixel by pixel. Across a set,
//! [`screen_flats`] matches the dips by the area they cover and sorts them into
//! steady ones, present in nearly every frame at nearly the same depth, and
//! changing ones. Steady dips are counted and left alone. Changing dips mark
//! the frames that hold them, and those present in more than half the frames
//! mark the master, because a median keeps them. A frame covered in dips is
//! marked even when they hold still, since dust seldom comes in hundreds.
//!
//! A raw colour (CFA) frame is measured on its 2x2 sums, which cancels the
//! filter pattern and keeps features a few pixels wide.

use super::{
    FLAT_QUALITY_ALGORITHM_VERSION, Finding, FindingCode, FrameVerdict, Limits, MIN_PEERS,
    first_flat_quality_algorithm_version, median_of,
};
use crate::dust::{self, DustOptions, DustShape, Optics, Vignetting};
use crate::flat_response::metric;
use crate::{Error, LinearImageRef, Result};
use rayon::prelude::*;
use seiza_stats::{median_in_place, robust_sigma_in_place};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;

/// Side of the square bins whose medians model the smooth response, in image
/// pixels. A median over a bin this size sees through a droplet or a dust
/// speck, and interpolating between bins still follows vignetting.
const BACKGROUND_BIN: usize = 64;
/// Cells along the long side of an illumination profile.
const ILLUMINATION_CELLS: usize = 16;
/// Most samples read from one illumination cell.
const ILLUMINATION_CELL_SAMPLES: usize = 16_384;
/// Most features one measurement keeps. The deepest are kept; the count
/// still covers all of them.
const MAX_FEATURES: usize = 100_000;
/// Empty pixels allowed between two parts of one feature.
const LINK_GAP: usize = 2;
/// A dip counts in screening once it is this many times as deep as the
/// detection depth. Shallower ones are mostly noise, and their depth and
/// presence wander from frame to frame.
const CLEAR_DEPTH: f64 = 2.0;
/// Most rows read to estimate noise.
const NOISE_ROWS: usize = 512;
/// Most samples kept from those rows for each noise estimate.
const NOISE_SAMPLES: usize = 1 << 20;
/// Fewest samples a noise estimate needs.
const MIN_NOISE_SAMPLES: usize = 1024;
/// Rows of the plane each parallel task scans for dips.
const BAND_ROWS: usize = 32;
/// Samples within this fraction of full scale count as clipped.
const CLIP_TOLERANCE: f32 = 0.001;
/// Read every this many samples to find a level.
const LEVEL_STRIDE: usize = 16;
/// Smallest plane, in pixels along each side, that can be measured.
const MIN_PLANE_SIDE: usize = 32;
/// A pixel-pass dip and a large feature on one centre whose radii are within
/// this factor are one shadow.
const SAME_SHADOW_SIZE: f32 = 1.5;

/// Settings for [`measure_flat`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FlatMeasureOptions {
    /// The value a saturated sample reads. Samples within 0.1% of it count as
    /// clipped. 65535 suits cameras that write 16-bit FITS.
    pub full_scale: f32,
    /// Shallowest dip counted as a feature, as a fraction of the response
    /// around it.
    pub feature_depth: f32,
    /// A dip must also reach this many times the noise of the smoothed
    /// response, so a noisy flat does not count its own noise. A dip of
    /// four plane pixels or fewer must also clear the threshold that keeps
    /// the number of noise dips in a frame near
    /// [`DustOptions::false_dips`]: a 26-megapixel frame holds a handful of
    /// lone pixels beyond 5σ by chance.
    pub feature_sigma: f32,
    /// Look for large disks, donuts and vignetting on the binned frame too.
    pub large_features: bool,
    /// The light cone, when known: sizes each disk's distance from the
    /// sensor and says whether a donut can occur.
    pub optics: Option<Optics>,
}

impl Default for FlatMeasureOptions {
    fn default() -> Self {
        Self {
            full_scale: 65535.0,
            feature_depth: 0.02,
            feature_sigma: 5.0,
            large_features: true,
            optics: None,
        }
    }
}

impl FlatMeasureOptions {
    fn validate(&self) -> Result<()> {
        if !self.full_scale.is_finite() || self.full_scale <= 0.0 {
            return Err(refused("full_scale must be finite and positive"));
        }
        if !self.feature_depth.is_finite() || !(0.0..=0.5).contains(&self.feature_depth) {
            return Err(refused(
                "feature_depth must be finite and between 0 and 0.5",
            ));
        }
        if self.feature_depth == 0.0 {
            return Err(refused("feature_depth must be greater than zero"));
        }
        if !self.feature_sigma.is_finite() || self.feature_sigma < 0.0 {
            return Err(refused("feature_sigma must be finite and not negative"));
        }
        if self.optics.is_some_and(|optics| !optics.is_usable()) {
            return Err(refused(
                "optics needs a positive focal ratio and pixel size and an obstruction below 1",
            ));
        }
        Ok(())
    }
}

/// One dust feature of a flat: a compact or irregular dip, or a large disk,
/// donut or patch. The definition is [`crate::dust::DustFeature`]'s, so a
/// flat's spots and a comparison's agree.
pub type FlatFeature = dust::DustFeature;

/// A coarse map of how light falls across a flat: vignetting, gradients, and
/// broad shadows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IlluminationProfile {
    /// Cells across.
    pub columns: usize,
    /// Cells down.
    pub rows: usize,
    /// Row-major median of each cell, divided by the median cell so a typical
    /// cell reads 1. A cell with no finite samples reads 0.
    pub response: Vec<f32>,
}

/// What [`measure_flat`] found in one flat frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlatMeasurement {
    /// [`FLAT_QUALITY_ALGORITHM_VERSION`] when this was measured; 1 when
    /// stored without one.
    #[serde(default = "first_flat_quality_algorithm_version")]
    pub algorithm_version: u32,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// The full-scale value the frame was measured against.
    pub full_scale: f32,
    /// Median sample.
    pub level: f32,
    /// Median of each channel: one entry for mono, three for RGB, and four
    /// for a CFA frame, one per site of the 2x2 pattern in the order top
    /// left, top right, bottom left, bottom right.
    pub channel_levels: Vec<f32>,
    /// Share of finite samples within 0.1% of full scale.
    pub clipped_fraction: f64,
    /// Pixel-scale noise of the response, as a fraction of it. A CFA frame
    /// is measured on its 2x2 sums.
    pub noise: f32,
    /// How deep a dip had to be to count as a feature in this frame: the
    /// larger of the configured depth and the configured multiple of the
    /// smoothed noise.
    pub detection_depth: f32,
    /// Coarse illumination map.
    pub illumination: IlluminationProfile,
    /// Features found, including any beyond the kept list.
    pub feature_count: usize,
    /// Large features (disks, donuts, patches) first, deepest first, then
    /// compact and irregular dips, deepest first; at most 100,000.
    pub features: Vec<FlatFeature>,
    /// Shadows hanging from the border that are not the image circle's own
    /// fall-off: an off-axis guider's prism, a filter holder.
    #[serde(default)]
    pub vignetting: Vec<Vignetting>,
}

/// Measure one flat frame.
///
/// `image` holds the raw samples, before bias or dark subtraction or
/// normalization; levels and clipping are judged against
/// [`FlatMeasureOptions::full_scale`]. Set `cfa` for a raw colour frame with
/// one channel. Non-finite samples are ignored.
///
/// Memory beyond the frame is small for mono frames: a few rows at a time
/// plus the features. A CFA or RGB frame needs one extra plane a quarter or a
/// third of its size.
pub fn measure_flat(
    image: LinearImageRef<'_>,
    cfa: bool,
    options: &FlatMeasureOptions,
) -> Result<FlatMeasurement> {
    options.validate()?;
    if cfa && image.channels() != 1 {
        return Err(invalid("a CFA frame must have one channel"));
    }
    let (plane_width, plane_height) = if cfa {
        (image.width() / 2, image.height() / 2)
    } else {
        (image.width(), image.height())
    };
    if plane_width < MIN_PLANE_SIDE || plane_height < MIN_PLANE_SIDE {
        return Err(invalid(format!(
            "a flat must be at least {MIN_PLANE_SIDE} pixels on each side to measure{}",
            if cfa { " after 2x2 binning" } else { "" }
        )));
    }

    let (level, channel_levels) = levels(image, cfa);
    let clipped_fraction = clipped_fraction(image.data(), options.full_scale);
    let plane = Plane::new(image, cfa);
    let illumination = illumination_profile(&plane);
    let background = Background::fit(&plane);
    let (noise, smoothed_noise) = residual_noise(&plane, &background);
    let detection_depth = options
        .feature_depth
        .max(options.feature_sigma * smoothed_noise);
    let runs = dip_runs(&plane, &background, detection_depth);
    let mut features = features_from_runs(&runs, plane.scale);
    // Lone pixels a little past the threshold are what noise leaves over
    // many megapixels, and the border's outermost pixels are the median's
    // edge: neither is dust.
    let dust_options = DustOptions {
        optics: options.optics,
        ..DustOptions::default()
    };
    let lone_floor = dust::lone_dip_floor(
        plane.width * plane.height,
        smoothed_noise,
        dust_options.false_dips,
    );
    let scale = plane.scale as u32;
    let (edge_right, edge_bottom) = (
        (plane.width as u32 * scale).saturating_sub(1 + 2 * scale),
        (plane.height as u32 * scale).saturating_sub(1 + 2 * scale),
    );
    features.retain(|feature| {
        let plane_pixels = feature.pixels / (scale * scale).max(1);
        let on_edge = feature.left < 2 * scale
            || feature.top < 2 * scale
            || feature.right > edge_right
            || feature.bottom > edge_bottom;
        !on_edge && (plane_pixels > 4 || feature.depth >= lone_floor)
    });
    features.sort_by(|a, b| b.depth.total_cmp(&a.depth));
    let mut vignetting = Vec::new();
    if options.large_features
        && let Some(map) = large_dust(image, cfa, &dust_options)
    {
        // Compact and irregular dips are found pixel by pixel above; the
        // binned plane adds what only it can see.
        let mut large: Vec<FlatFeature> = map
            .features
            .into_iter()
            .filter(|feature| feature.shape.is_large())
            .collect();
        large.sort_by(|a, b| b.depth.total_cmp(&a.depth));
        // A round shadow deep enough for the pixel pass is found by both:
        // the dip on a large feature's centre and of about its size is that
        // feature again. Clearly smaller dips on it are dust of their own.
        features.retain(|dip| {
            !large.iter().any(|shadow| {
                (dip.x - shadow.x).hypot(dip.y - shadow.y) <= 0.5 * shadow.radius
                    && dip.radius * SAME_SHADOW_SIZE >= shadow.radius
                    && dip.radius <= SAME_SHADOW_SIZE * shadow.radius
            })
        });
        large.extend(features);
        features = large;
        vignetting = map.vignetting;
    }
    let feature_count = features.len();
    features.truncate(MAX_FEATURES);

    Ok(FlatMeasurement {
        algorithm_version: FLAT_QUALITY_ALGORITHM_VERSION,
        width: image.width(),
        height: image.height(),
        full_scale: options.full_scale,
        level,
        channel_levels,
        clipped_fraction,
        noise,
        detection_depth,
        illumination,
        feature_count,
        features,
        vignetting,
    })
}

/// The large features of one flat: the frame binned to about 1,024 pixels
/// and searched as [`crate::dust::find_dust`] does.
fn large_dust(
    image: LinearImageRef<'_>,
    cfa: bool,
    options: &DustOptions,
) -> Option<dust::DustMap> {
    let (width, height) = (image.width(), image.height());
    let bin = metric::bin_factor(width, height, cfa);
    let plane = metric::bin_frame(image.data(), width, height, image.channels(), bin);
    let mut map = dust::find_dust(&plane, bin, options)?;
    map.frames = 1;
    Some(map)
}

/// Limits for [`screen_flats`]. Each [`Limits`] says where a frame turns
/// suspect and then poor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FlatScreening {
    /// Share of clipped samples; higher is worse. Clipped pixels carry no
    /// response at all.
    pub clipped: Limits,
    /// Median of the brightest channel as a share of full scale; higher is
    /// worse. Sensors leave their linear range before they clip.
    pub bright: Limits,
    /// Median of the dimmest channel as a share of full scale; lower is
    /// worse. A dim flat is noisy, and bias errors weigh more in it.
    pub dim: Limits,
    /// Largest difference between a frame's illumination profile and the
    /// set's median profile, as a fraction; higher is worse. A light leak, a
    /// moved panel, or a rotated camera changes the pattern.
    pub illumination: Limits,
    /// Clear compact features in one frame, steady or not; higher is worse.
    /// Disks and donuts are not counted: they are the light path's.
    /// A feature is clear when it is at least twice the frame's detection
    /// depth. Dust rarely comes near these counts, while a frame covered in
    /// droplets passes them. This catches droplets that do not change while
    /// the set is shot, and judges a frame that has no peers.
    pub features: Limits,
    /// Changing features in one frame; higher is worse.
    pub changing_features: Limits,
    /// Changing features in more than half the frames, judged on the set;
    /// higher is worse.
    pub carried_features: Limits,
    /// A feature is steady when it is found in at least this share of the
    /// frames...
    pub steady_presence: f64,
    /// ...and its depth varies by no more than this fraction of its deepest
    /// reading, plus the typical detection depth to allow for noise.
    pub steady_depth_range: f64,
    /// Features in different frames whose bounding boxes come within this
    /// many pixels of each other are the same feature. Boxes rather than
    /// centres, because noise can split a ring into arcs in one frame and
    /// not the next.
    pub match_distance: u32,
}

impl Default for FlatScreening {
    fn default() -> Self {
        Self {
            clipped: Limits::new(0.0005, 0.005),
            bright: Limits::new(0.8, 0.92),
            dim: Limits::new(0.1, 0.03),
            illumination: Limits::new(0.02, 0.05),
            features: Limits::new(100.0, 500.0),
            changing_features: Limits::new(3.0, 20.0),
            carried_features: Limits::new(1.0, 10.0),
            steady_presence: 0.9,
            steady_depth_range: 0.5,
            match_distance: 2,
        }
    }
}

impl FlatScreening {
    fn validate(&self) -> Result<()> {
        if !self.steady_presence.is_finite() || !(0.5..=1.0).contains(&self.steady_presence) {
            return Err(refused(
                "steady_presence must be finite and between 0.5 and 1",
            ));
        }
        if !self.steady_depth_range.is_finite() || !(0.0..=1.0).contains(&self.steady_depth_range) {
            return Err(refused(
                "steady_depth_range must be finite and between 0 and 1",
            ));
        }
        Ok(())
    }
}

/// [`screen_flats`]' judgement of one flat set.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FlatSetReport {
    /// One verdict per frame, in input order.
    pub frames: Vec<FrameVerdict>,
    /// The verdict on a master built from every frame: whether changing
    /// features would survive into it.
    pub master: FrameVerdict,
    /// Features found in nearly every frame at nearly the same depth. Such a
    /// feature belongs to the light path and the flat corrects it; this is
    /// for information only.
    pub steady_features: usize,
    /// Features that come and go or change depth across the set.
    pub changing_features: usize,
    /// Changing features present in more than half the frames, which a
    /// median-based master keeps.
    pub carried_features: usize,
}

/// Judge a set of flats that should match: one filter, one session, one
/// rotation, one optical train.
///
/// Every frame is checked on its own for clipping, exposure, and the number
/// of clear compact features it holds. With three or more frames, each is
/// also compared with the set: its illumination profile against the median
/// profile, and its features against everyone else's.
///
/// Features are matched by the boxes they cover. One found in at least
/// [`FlatScreening::steady_presence`] of the frames with depths that agree
/// within [`FlatScreening::steady_depth_range`] is steady; anything else is
/// changing. A feature never deeper than twice the typical detection depth is
/// too faint to tell and is not counted either way.
///
/// Fails when the measurements differ in size, since they cannot then be one
/// set.
pub fn screen_flats(
    frames: &[FlatMeasurement],
    screening: &FlatScreening,
) -> Result<FlatSetReport> {
    screening.validate()?;
    if let Some(first) = frames.first()
        && let Some(other) = frames.iter().find(|frame| {
            frame.width != first.width
                || frame.height != first.height
                || frame.illumination.columns != first.illumination.columns
                || frame.illumination.rows != first.illumination.rows
                || frame.illumination.response.len() != first.illumination.response.len()
        })
    {
        return Err(refused(format!(
            "flat measurements of {}x{} and {}x{} frames cannot form one set",
            first.width, first.height, other.width, other.height
        )));
    }

    let mut findings: Vec<Vec<Finding>> = frames
        .iter()
        .map(|frame| {
            let mut findings = Vec::new();
            screening.clipped.check_above(
                &mut findings,
                FindingCode::Clipped,
                frame.clipped_fraction,
            );
            let full_scale = f64::from(frame.full_scale);
            let levels = frame
                .channel_levels
                .iter()
                .map(|level| f64::from(*level) / full_scale);
            let brightest = levels.clone().fold(f64::NAN, f64::max);
            let dimmest = levels.fold(f64::NAN, f64::min);
            screening
                .bright
                .check_above(&mut findings, FindingCode::Bright, brightest);
            screening
                .dim
                .check_below(&mut findings, FindingCode::Dim, dimmest);
            let clear = frame
                .features
                .iter()
                .filter(|feature| !feature.shape.is_large())
                .filter(|feature| {
                    f64::from(feature.depth) >= CLEAR_DEPTH * f64::from(frame.detection_depth)
                })
                .count();
            screening
                .features
                .check_above(&mut findings, FindingCode::Features, clear as f64);
            findings
        })
        .collect();

    let mut report = FlatSetReport::default();
    if frames.len() >= MIN_PEERS {
        for (findings, difference) in findings.iter_mut().zip(illumination_differences(frames)) {
            screening
                .illumination
                .check_above(findings, FindingCode::Illumination, difference);
        }
        let tally = tally_features(frames, screening);
        for (findings, changing) in findings.iter_mut().zip(&tally.changing_per_frame) {
            screening.changing_features.check_above(
                findings,
                FindingCode::ChangingFeatures,
                *changing as f64,
            );
        }
        let mut master = Vec::new();
        screening.carried_features.check_above(
            &mut master,
            FindingCode::CarriedFeatures,
            tally.carried as f64,
        );
        report.master = FrameVerdict::from_findings(master);
        report.steady_features = tally.steady;
        report.changing_features = tally.changing;
        report.carried_features = tally.carried;
    }
    report.frames = findings
        .into_iter()
        .map(FrameVerdict::from_findings)
        .collect();
    Ok(report)
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidImage(message.into())
}

fn refused(message: impl Into<String>) -> Error {
    Error::FrameQuality(message.into())
}

fn median_or_zero(mut values: Vec<f32>) -> f32 {
    median_in_place(&mut values).unwrap_or(0.0)
}

/// The median of every sixteenth sample, and of each channel or CFA site.
fn levels(image: LinearImageRef<'_>, cfa: bool) -> (f32, Vec<f32>) {
    let (width, height, channels) = (image.width(), image.height(), image.channels());
    let data = image.data();
    let level = median_or_zero(
        data.iter()
            .step_by(LEVEL_STRIDE)
            .copied()
            .filter(|value| value.is_finite())
            .collect(),
    );
    let channel_levels = if cfa {
        // Every fourth row and column of each site: one site sample in
        // sixteen, like the overall level.
        (0..4)
            .map(|site| {
                let (column, row) = (site & 1, site >> 1);
                let mut values = Vec::new();
                for y in (row..height).step_by(8) {
                    values.extend(
                        data[y * width..(y + 1) * width]
                            .iter()
                            .skip(column)
                            .step_by(8)
                            .copied()
                            .filter(|value| value.is_finite()),
                    );
                }
                median_or_zero(values)
            })
            .collect()
    } else {
        (0..channels)
            .map(|channel| {
                median_or_zero(
                    data.iter()
                        .skip(channel)
                        .step_by(channels * LEVEL_STRIDE)
                        .copied()
                        .filter(|value| value.is_finite())
                        .collect(),
                )
            })
            .collect()
    };
    (level, channel_levels)
}

fn clipped_fraction(data: &[f32], full_scale: f32) -> f64 {
    let clip_at = full_scale * (1.0 - CLIP_TOLERANCE);
    let (clipped, finite) = data
        .par_chunks(1 << 16)
        .map(|chunk| {
            chunk
                .iter()
                .filter(|value| value.is_finite())
                .fold((0_u64, 0_u64), |(clipped, finite), value| {
                    (clipped + u64::from(*value >= clip_at), finite + 1)
                })
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    if finite == 0 {
        0.0
    } else {
        clipped as f64 / finite as f64
    }
}

/// The single-channel image the spatial measures run on.
struct Plane<'a> {
    data: Cow<'a, [f32]>,
    width: usize,
    height: usize,
    /// Image pixels per plane pixel along each axis.
    scale: usize,
}

impl<'a> Plane<'a> {
    /// Mono frames are borrowed. A CFA frame becomes its 2x2 sums, dropping
    /// an odd last row or column; an RGB frame becomes its channel mean.
    fn new(image: LinearImageRef<'a>, cfa: bool) -> Self {
        let (width, height, channels) = (image.width(), image.height(), image.channels());
        let data = image.data();
        if cfa {
            let (plane_width, plane_height) = (width / 2, height / 2);
            let mut binned = vec![0.0_f32; plane_width * plane_height];
            binned
                .par_chunks_mut(plane_width)
                .enumerate()
                .for_each(|(y, row)| {
                    let top = &data[2 * y * width..(2 * y + 1) * width];
                    let bottom = &data[(2 * y + 1) * width..(2 * y + 2) * width];
                    for (x, sample) in row.iter_mut().enumerate() {
                        *sample = top[2 * x] + top[2 * x + 1] + bottom[2 * x] + bottom[2 * x + 1];
                    }
                });
            Self {
                data: Cow::Owned(binned),
                width: plane_width,
                height: plane_height,
                scale: 2,
            }
        } else if channels == 1 {
            Self {
                data: Cow::Borrowed(data),
                width,
                height,
                scale: 1,
            }
        } else {
            let mean = data
                .par_chunks(channels)
                .map(|pixel| pixel.iter().sum::<f32>() / channels as f32)
                .collect();
            Self {
                data: Cow::Owned(mean),
                width,
                height,
                scale: 1,
            }
        }
    }

    fn row(&self, y: usize) -> &[f32] {
        &self.data[y * self.width..(y + 1) * self.width]
    }
}

fn illumination_profile(plane: &Plane<'_>) -> IlluminationProfile {
    let (width, height) = (plane.width, plane.height);
    let (columns, rows) = if width >= height {
        (
            ILLUMINATION_CELLS,
            ((ILLUMINATION_CELLS * height + width / 2) / width).max(1),
        )
    } else {
        (
            ((ILLUMINATION_CELLS * width + height / 2) / height).max(1),
            ILLUMINATION_CELLS,
        )
    };
    let cells: Vec<f32> = (0..columns * rows)
        .into_par_iter()
        .map(|cell| {
            let (column, row) = (cell % columns, cell / columns);
            let (x0, x1) = (column * width / columns, (column + 1) * width / columns);
            let (y0, y1) = (row * height / rows, (row + 1) * height / rows);
            let area = (x1 - x0) * (y1 - y0);
            let stride = ((area as f64 / ILLUMINATION_CELL_SAMPLES as f64)
                .sqrt()
                .ceil() as usize)
                .max(1);
            let mut values = Vec::with_capacity(area / (stride * stride) + 1);
            for y in (y0..y1).step_by(stride) {
                values.extend(
                    plane.row(y)[x0..x1]
                        .iter()
                        .step_by(stride)
                        .copied()
                        .filter(|value| value.is_finite()),
                );
            }
            median_in_place(&mut values).unwrap_or(f32::NAN)
        })
        .collect();
    let typical = median_or_zero(
        cells
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect(),
    );
    let response = cells
        .iter()
        .map(|cell| {
            let ratio = cell / typical;
            if typical > 0.0 && ratio.is_finite() && ratio > 0.0 {
                ratio
            } else {
                0.0
            }
        })
        .collect();
    IlluminationProfile {
        columns,
        rows,
        response,
    }
}

/// The smooth response: bin medians, interpolated bilinearly between bin
/// centres. Past the outer centres it follows a parabola through the last
/// three, so the extra fall of vignetting into a corner is not read as a dip.
struct Background {
    columns: usize,
    values: Vec<f32>,
    x_weights: Vec<Weights>,
    y_weights: Vec<Weights>,
}

/// Up to three bins and the weight each takes at one position along an axis.
type Weights = [(usize, f32); 3];

impl Background {
    fn fit(plane: &Plane<'_>) -> Self {
        let bin = (BACKGROUND_BIN / plane.scale).max(8);
        let (width, height) = (plane.width, plane.height);
        let (columns, rows) = (width.div_ceil(bin), height.div_ceil(bin));
        let mut values: Vec<f32> = (0..rows)
            .into_par_iter()
            .flat_map_iter(|bin_row| {
                let (y0, y1) = (bin_row * bin, ((bin_row + 1) * bin).min(height));
                let mut samples = Vec::with_capacity(bin * bin);
                (0..columns)
                    .map(|bin_column| {
                        let (x0, x1) = (bin_column * bin, ((bin_column + 1) * bin).min(width));
                        samples.clear();
                        for y in y0..y1 {
                            samples.extend(
                                plane.row(y)[x0..x1]
                                    .iter()
                                    .copied()
                                    .filter(|value| value.is_finite()),
                            );
                        }
                        median_in_place(&mut samples).unwrap_or(f32::NAN)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let fill = median_or_zero(
            values
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .collect(),
        );
        for value in &mut values {
            if !value.is_finite() {
                *value = fill;
            }
        }
        let centres = |count: usize, length: usize| -> Vec<f32> {
            (0..count)
                .map(|index| {
                    let (start, end) = (index * bin, ((index + 1) * bin).min(length));
                    (start + end - 1) as f32 / 2.0
                })
                .collect()
        };
        Self {
            columns,
            values,
            x_weights: axis_weights(&centres(columns, width), width),
            y_weights: axis_weights(&centres(rows, height), height),
        }
    }

    /// Fill `out` with the smooth response along row `y`. `blend` holds one
    /// value per bin column between calls.
    fn row(&self, y: usize, blend: &mut Vec<f32>, out: &mut [f32]) {
        let columns = self.columns;
        blend.clear();
        blend.resize(columns, 0.0);
        for (row, weight) in self.y_weights[y] {
            if weight != 0.0 {
                let values = &self.values[row * columns..(row + 1) * columns];
                for (sum, value) in blend.iter_mut().zip(values) {
                    *sum += weight * value;
                }
            }
        }
        for (sample, weights) in out.iter_mut().zip(&self.x_weights) {
            *sample = weights
                .iter()
                .map(|(column, weight)| weight * blend[*column])
                .sum();
        }
    }
}

/// The bins and weights that give the smooth response at each position along
/// an axis: linear between the two centres around a position, quadratic
/// through the three outer centres beyond them.
fn axis_weights(centres: &[f32], length: usize) -> Vec<Weights> {
    let count = centres.len();
    (0..length)
        .map(|position| {
            let position = position as f32;
            if count == 1 {
                return [(0, 1.0), (0, 0.0), (0, 0.0)];
            }
            let (first, last) = (centres[0], centres[count - 1]);
            if count >= 3 && (position < first || position > last) {
                let start = if position < first { 0 } else { count - 3 };
                let nodes = [centres[start], centres[start + 1], centres[start + 2]];
                return std::array::from_fn(|k| {
                    let (a, b) = (nodes[(k + 1) % 3], nodes[(k + 2) % 3]);
                    let weight =
                        (position - a) * (position - b) / ((nodes[k] - a) * (nodes[k] - b));
                    (start + k, weight)
                });
            }
            let index = centres
                .partition_point(|centre| *centre <= position)
                .saturating_sub(1)
                .min(count - 2);
            let t = (position - centres[index]) / (centres[index + 1] - centres[index]);
            [(index, 1.0 - t), (index + 1, t), (0, 0.0)]
        })
        .collect()
}

/// Fill `out` with the plane's response along row `y` relative to the smooth
/// response: 0 where they agree, -0.1 where a tenth of the light is missing.
/// `NaN` where either is unusable.
fn residual_row(
    plane: &Plane<'_>,
    background: &Background,
    y: usize,
    blend: &mut Vec<f32>,
    out: &mut [f32],
) {
    background.row(y, blend, out);
    for (residual, value) in out.iter_mut().zip(plane.row(y)) {
        let smooth = *residual;
        *residual = if smooth > 0.0 && value.is_finite() {
            value / smooth - 1.0
        } else {
            f32::NAN
        };
    }
}

/// Fill `out[1..width - 1]` with the 3x3 median of three residual rows;
/// `NaN` where any of the nine is. The ends are left `NaN`.
fn median_row(above: &[f32], middle: &[f32], below: &[f32], out: &mut [f32]) {
    let width = out.len();
    out[0] = f32::NAN;
    out[width - 1] = f32::NAN;
    for x in 1..width - 1 {
        let window = [
            above[x - 1],
            above[x],
            above[x + 1],
            middle[x - 1],
            middle[x],
            middle[x + 1],
            below[x - 1],
            below[x],
            below[x + 1],
        ];
        out[x] = if window.iter().all(|value| value.is_finite()) {
            median9(window)
        } else {
            f32::NAN
        };
    }
}

/// Median of nine values with a fixed exchange network.
fn median9(mut p: [f32; 9]) -> f32 {
    fn order(p: &mut [f32; 9], a: usize, b: usize) {
        let (low, high) = (p[a].min(p[b]), p[a].max(p[b]));
        p[a] = low;
        p[b] = high;
    }
    for (a, b) in [
        (1, 2),
        (4, 5),
        (7, 8),
        (0, 1),
        (3, 4),
        (6, 7),
        (1, 2),
        (4, 5),
        (7, 8),
        (0, 3),
        (5, 8),
        (4, 7),
        (3, 6),
        (1, 4),
        (2, 5),
        (4, 7),
        (4, 2),
        (6, 4),
        (4, 2),
    ] {
        order(&mut p, a, b);
    }
    p[4]
}

/// Robust noise of the residual and of its 3x3 median, from evenly spaced
/// rows. Zero when the plane gives too few samples.
fn residual_noise(plane: &Plane<'_>, background: &Background) -> (f32, f32) {
    let (width, height) = (plane.width, plane.height);
    let interior = height - 2;
    let stride = interior.div_ceil(NOISE_ROWS).max(1);
    let rows: Vec<usize> = (1..height - 1).step_by(stride).collect();
    let column_stride = (rows.len() * (width - 2)).div_ceil(NOISE_SAMPLES).max(1);
    let (mut raw, mut smoothed) = rows
        .par_iter()
        .map(|&y| {
            let mut blend = Vec::new();
            let mut window = [vec![0.0; width], vec![0.0; width], vec![0.0; width]];
            for (offset, row) in window.iter_mut().enumerate() {
                residual_row(plane, background, y + offset - 1, &mut blend, row);
            }
            let mut median = vec![0.0; width];
            median_row(&window[0], &window[1], &window[2], &mut median);
            let finite = |values: &[f32]| -> Vec<f32> {
                values[1..width - 1]
                    .iter()
                    .step_by(column_stride)
                    .copied()
                    .filter(|value| value.is_finite())
                    .collect()
            };
            (finite(&window[1]), finite(&median))
        })
        .reduce(
            || (Vec::new(), Vec::new()),
            |mut a, mut b| {
                a.0.append(&mut b.0);
                a.1.append(&mut b.1);
                a
            },
        );
    (spread(&mut raw), spread(&mut smoothed))
}

fn spread(values: &mut [f32]) -> f32 {
    if values.len() < MIN_NOISE_SAMPLES {
        return 0.0;
    }
    let Some(centre) = median_in_place(values) else {
        return 0.0;
    };
    robust_sigma_in_place(values, centre).unwrap_or(0.0)
}

/// A run of adjacent plane pixels on one row whose smoothed residual lies at
/// or below the negated detection depth.
#[derive(Clone, Copy, Debug)]
struct Run {
    y: usize,
    /// First column.
    start: usize,
    /// One past the last column.
    end: usize,
    /// Lowest smoothed residual along the run.
    lowest: f32,
}

/// Every run of dip pixels, ordered by row and then column.
fn dip_runs(plane: &Plane<'_>, background: &Background, depth: f32) -> Vec<Run> {
    let (width, height) = (plane.width, plane.height);
    let threshold = -depth;
    let bands = (height - 2).div_ceil(BAND_ROWS);
    (0..bands)
        .into_par_iter()
        .flat_map_iter(|band| {
            let first = 1 + band * BAND_ROWS;
            let last = (first + BAND_ROWS).min(height - 1);
            let mut blend = Vec::new();
            let mut window = [vec![0.0; width], vec![0.0; width], vec![0.0; width]];
            residual_row(plane, background, first - 1, &mut blend, &mut window[0]);
            residual_row(plane, background, first, &mut blend, &mut window[1]);
            let mut median = vec![0.0; width];
            let mut runs = Vec::new();
            for y in first..last {
                residual_row(plane, background, y + 1, &mut blend, &mut window[2]);
                median_row(&window[0], &window[1], &window[2], &mut median);
                let mut x = 0;
                while x < width {
                    if median[x] <= threshold {
                        let start = x;
                        let mut lowest = median[x];
                        while x < width && median[x] <= threshold {
                            lowest = lowest.min(median[x]);
                            x += 1;
                        }
                        runs.push(Run {
                            y,
                            start,
                            end: x,
                            lowest,
                        });
                    } else {
                        x += 1;
                    }
                }
                window.rotate_left(1);
            }
            runs
        })
        .collect()
}

/// Join runs into features. Runs no more than [`LINK_GAP`] empty pixels
/// apart, across or down, belong to one feature, so a ring that noise has
/// broken into arcs still counts once.
fn features_from_runs(runs: &[Run], scale: usize) -> Vec<FlatFeature> {
    let mut parent: Vec<usize> = (0..runs.len()).collect();
    fn root(parent: &mut [usize], mut index: usize) -> usize {
        while parent[index] != index {
            parent[index] = parent[parent[index]];
            index = parent[index];
        }
        index
    }
    fn join(parent: &mut [usize], a: usize, b: usize) {
        let (a, b) = (root(parent, a), root(parent, b));
        parent[a.max(b)] = a.min(b);
    }
    // Runs are ordered by row, then column. Keep the index ranges of the
    // last few rows that held runs.
    let mut recent: Vec<(usize, std::ops::Range<usize>)> = Vec::new();
    let mut start = 0;
    while start < runs.len() {
        let y = runs[start].y;
        let end = start + runs[start..].iter().take_while(|run| run.y == y).count();
        for current in start..end {
            let a = runs[current];
            if current + 1 < end && runs[current + 1].start <= a.end + LINK_GAP {
                join(&mut parent, current, current + 1);
            }
            for (row, range) in &recent {
                if y - row > LINK_GAP + 1 {
                    continue;
                }
                // Runs in a row do not overlap, so their ends rise with their
                // starts: skip those that end too far left, stop at the first
                // that starts too far right.
                let above = &runs[range.clone()];
                let first = above.partition_point(|b| b.end + LINK_GAP < a.start);
                for (offset, b) in above[first..].iter().enumerate() {
                    if b.start > a.end + LINK_GAP {
                        break;
                    }
                    join(&mut parent, current, range.start + first + offset);
                }
            }
        }
        recent.retain(|(row, _)| y - row <= LINK_GAP);
        recent.push((y, start..end));
        start = end;
    }

    #[derive(Clone, Copy)]
    struct Sum {
        pixels: u64,
        x: f64,
        y: f64,
        left: usize,
        top: usize,
        right: usize,
        bottom: usize,
        lowest: f32,
    }
    let mut sums: HashMap<usize, Sum> = HashMap::new();
    for (index, run) in runs.iter().enumerate() {
        let pixels = (run.end - run.start) as u64;
        let sum = sums.entry(root(&mut parent, index)).or_insert(Sum {
            pixels: 0,
            x: 0.0,
            y: 0.0,
            left: run.start,
            top: run.y,
            right: run.end - 1,
            bottom: run.y,
            lowest: 0.0,
        });
        sum.pixels += pixels;
        sum.x += (run.start + run.end - 1) as f64 * pixels as f64 / 2.0;
        sum.y += run.y as f64 * pixels as f64;
        sum.left = sum.left.min(run.start);
        sum.top = sum.top.min(run.y);
        sum.right = sum.right.max(run.end - 1);
        sum.bottom = sum.bottom.max(run.y);
        sum.lowest = sum.lowest.min(run.lowest);
    }
    let scale_f = scale as f64;
    let first = |plane: usize| u32::try_from(plane * scale).unwrap_or(u32::MAX);
    let last = |plane: usize| u32::try_from(plane * scale + scale - 1).unwrap_or(u32::MAX);
    // No size floor: the 3x3 median has already removed lone bad pixels and
    // pairs, and a floor would let a small, deep dip flicker in and out of
    // the list from frame to frame as noise moves its edge.
    sums.into_values()
        .map(|sum| {
            let centre = |total: f64| ((total / sum.pixels as f64 + 0.5) * scale_f - 0.5) as f32;
            let pixels = u32::try_from(sum.pixels * (scale * scale) as u64).unwrap_or(u32::MAX);
            // Window dust is often not round: a dip filling less than half
            // its box, or three times as long as it is wide, is irregular.
            let (box_w, box_h) = (sum.right - sum.left + 1, sum.bottom - sum.top + 1);
            let fill = sum.pixels as f64 / (box_w * box_h) as f64;
            let aspect = box_w.max(box_h) as f64 / box_w.min(box_h) as f64;
            let irregular = sum.pixels >= 6 && (fill < 0.5 || aspect >= 3.0);
            FlatFeature {
                x: centre(sum.x),
                y: centre(sum.y),
                left: first(sum.left),
                top: first(sum.top),
                right: last(sum.right),
                bottom: last(sum.bottom),
                pixels,
                depth: -sum.lowest,
                shape: if irregular {
                    DustShape::Irregular
                } else {
                    DustShape::Compact
                },
                radius: (pixels as f32 / std::f32::consts::PI).sqrt(),
                hole: None,
                distance_mm: None,
                seen_in: None,
            }
        })
        .collect()
}

/// For each frame, the largest difference between its illumination profile
/// and the set's per-cell median, after matching overall scale.
fn illumination_differences(frames: &[FlatMeasurement]) -> Vec<f64> {
    let cells = frames[0].illumination.response.len();
    let typical: Vec<f64> = (0..cells)
        .map(|cell| {
            median_of(
                frames
                    .iter()
                    .map(|frame| f64::from(frame.illumination.response[cell]))
                    .filter(|value| *value > 0.0),
            )
        })
        .collect();
    frames
        .iter()
        .map(|frame| {
            let ratios: Vec<f64> = frame
                .illumination
                .response
                .iter()
                .zip(&typical)
                .filter_map(|(value, typical)| {
                    let value = f64::from(*value);
                    (value > 0.0 && *typical > 0.0).then(|| value / typical)
                })
                .collect();
            let scale = median_of(ratios.iter().copied());
            ratios
                .iter()
                .map(|ratio| (ratio / scale - 1.0).abs())
                .fold(f64::NAN, f64::max)
        })
        .collect()
}

struct FeatureTally {
    changing_per_frame: Vec<usize>,
    steady: usize,
    changing: usize,
    carried: usize,
}

/// Match features across frames by position and sort each into steady or
/// changing.
fn tally_features(frames: &[FlatMeasurement], screening: &FlatScreening) -> FeatureTally {
    let count = frames.len();
    let points: Vec<(usize, FlatFeature)> = frames
        .iter()
        .enumerate()
        .flat_map(|(frame, measurement)| {
            measurement
                .features
                .iter()
                .filter(|feature| !feature.shape.is_large())
                .map(move |feature| (frame, *feature))
        })
        .collect();

    // Join every pair of features whose boxes come within the match distance.
    // Each feature is filed under every grid cell its widened box touches, so
    // two that should join always share a cell.
    const CELL: u32 = 32;
    let margin = screening.match_distance;
    let widened = |feature: &FlatFeature| {
        (
            feature.left.saturating_sub(margin),
            feature.top.saturating_sub(margin),
            feature.right.saturating_add(margin),
            feature.bottom.saturating_add(margin),
        )
    };
    let mut parent: Vec<usize> = (0..points.len()).collect();
    fn root(parent: &mut [usize], mut index: usize) -> usize {
        while parent[index] != index {
            parent[index] = parent[parent[index]];
            index = parent[index];
        }
        index
    }
    let mut grid: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (index, (_, feature)) in points.iter().enumerate() {
        let (left, top, right, bottom) = widened(feature);
        for row in top / CELL..=bottom / CELL {
            for column in left / CELL..=right / CELL {
                let neighbours = grid.entry((column, row)).or_default();
                for &other in neighbours.iter() {
                    let candidate = &points[other].1;
                    if left <= candidate.right
                        && candidate.left <= right
                        && top <= candidate.bottom
                        && candidate.top <= bottom
                    {
                        let (a, b) = (root(&mut parent, index), root(&mut parent, other));
                        parent[a.max(b)] = a.min(b);
                    }
                }
                neighbours.push(index);
            }
        }
    }

    // Each place's deepest reading in each frame; zero where a frame has none.
    let mut places: HashMap<usize, Vec<f32>> = HashMap::new();
    for (index, (frame, feature)) in points.iter().enumerate() {
        let depths = places
            .entry(root(&mut parent, index))
            .or_insert_with(|| vec![0.0; count]);
        depths[*frame] = depths[*frame].max(feature.depth);
    }

    let typical = median_of(frames.iter().map(|frame| f64::from(frame.detection_depth)));
    let required = (screening.steady_presence * count as f64).ceil() as usize;
    let mut tally = FeatureTally {
        changing_per_frame: vec![0; count],
        steady: 0,
        changing: 0,
        carried: 0,
    };
    for depths in places.values() {
        let present = depths.iter().filter(|depth| **depth > 0.0).count();
        let deepest = depths.iter().copied().fold(0.0_f32, f32::max);
        let shallowest = depths
            .iter()
            .copied()
            .filter(|depth| *depth > 0.0)
            .fold(f32::INFINITY, f32::min);
        if f64::from(deepest) < CLEAR_DEPTH * typical {
            continue;
        }
        // Noise moves a dip's measured depth by up to about the detection
        // depth, so a shallow dip gets that much slack on top of the share.
        let range = f64::from(deepest - shallowest);
        let allowed = screening.steady_depth_range * f64::from(deepest) + typical;
        if present >= required && range <= allowed {
            tally.steady += 1;
            continue;
        }
        tally.changing += 1;
        if present * 2 > count {
            tally.carried += 1;
        }
        for (frame, depth) in depths.iter().enumerate() {
            if *depth > 0.0 {
                tally.changing_per_frame[frame] += 1;
            }
        }
    }
    tally
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality::Grade;

    const WIDTH: usize = 384;
    const HEIGHT: usize = 256;
    const LEVEL: f32 = 30_000.0;

    /// Deterministic standard normal noise.
    fn gaussian(index: usize, seed: u64) -> f32 {
        let mut state = (index as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(seed.wrapping_mul(0xD1B5_4A32_D192_ED03));
        let mut uniform = || {
            state ^= state >> 30;
            state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            state ^= state >> 27;
            state = state.wrapping_mul(0x94D0_49BB_1331_11EB);
            state ^= state >> 31;
            ((state >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
        };
        let (a, b) = (uniform(), uniform());
        ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()) as f32
    }

    /// A vignetted flat with shot-like noise of about 1%.
    fn flat(seed: u64) -> Vec<f32> {
        let (cx, cy) = (WIDTH as f32 / 2.0, HEIGHT as f32 / 2.0);
        let corner = cx * cx + cy * cy;
        (0..WIDTH * HEIGHT)
            .map(|index| {
                let (x, y) = ((index % WIDTH) as f32, (index / WIDTH) as f32);
                let r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                let response = LEVEL * (1.0 - 0.25 * r2 / corner);
                response * (1.0 + 0.01 * gaussian(index, seed))
            })
            .collect()
    }

    /// Multiply a disc of `radius` around (`x`, `y`) by `1 - depth`.
    fn dip(data: &mut [f32], x: f32, y: f32, radius: f32, depth: f32) {
        for (index, sample) in data.iter_mut().enumerate() {
            let (px, py) = ((index % WIDTH) as f32, (index / WIDTH) as f32);
            if (px - x).powi(2) + (py - y).powi(2) <= radius * radius {
                *sample *= 1.0 - depth;
            }
        }
    }

    /// Spots on a grid, away from the edges.
    fn spots(count: usize) -> Vec<(f32, f32)> {
        (0..count)
            .map(|index| {
                let (column, row) = (index % 10, index / 10);
                (30.0 + 34.0 * column as f32, 30.0 + 40.0 * row as f32)
            })
            .collect()
    }

    fn measure(data: &[f32]) -> FlatMeasurement {
        measure_flat(
            LinearImageRef::new(data, WIDTH, HEIGHT, 1).unwrap(),
            false,
            &FlatMeasureOptions::default(),
        )
        .unwrap()
    }

    fn codes(verdict: &FrameVerdict) -> Vec<(FindingCode, Grade)> {
        verdict
            .findings
            .iter()
            .map(|finding| (finding.code, finding.grade))
            .collect()
    }

    #[test]
    fn median9_matches_a_sort() {
        for seed in 0..200 {
            let values: [f32; 9] = std::array::from_fn(|index| gaussian(index, seed));
            let mut sorted = values;
            sorted.sort_by(f32::total_cmp);
            assert_eq!(median9(values), sorted[4]);
        }
    }

    #[test]
    fn a_clean_flat_has_no_features_and_reads_its_noise() {
        let measurement = measure(&flat(1));
        assert_eq!(measurement.feature_count, 0, "{:?}", measurement.features);
        assert!(
            (measurement.noise - 0.01).abs() < 0.001,
            "{}",
            measurement.noise
        );
        assert!((measurement.level / LEVEL - 0.9).abs() < 0.05);
        assert_eq!(measurement.clipped_fraction, 0.0);
        assert_eq!(measurement.illumination.columns, 16);
        assert_eq!(measurement.illumination.rows, 11);
    }

    #[test]
    fn a_droplet_is_found_where_it_is() {
        let mut data = flat(2);
        // A dark ring with a bright centre, as a drop of water makes.
        dip(&mut data, 100.0, 80.0, 3.5, 0.6);
        dip(&mut data, 100.0, 80.0, 1.5, -2.0);
        let measurement = measure(&data);
        assert_eq!(measurement.feature_count, 1, "{:?}", measurement.features);
        let feature = measurement.features[0];
        assert!((feature.x - 100.0).abs() < 1.5 && (feature.y - 80.0).abs() < 1.5);
        assert!(feature.left >= 95 && feature.right <= 105, "{feature:?}");
        assert!(feature.top >= 75 && feature.bottom <= 85, "{feature:?}");
        assert!(feature.depth > 0.3, "{}", feature.depth);
    }

    #[test]
    fn an_irregular_dip_is_reported_once() {
        // Window dust 60 x 5 pixels, 10% deep: found pixel by pixel, and on
        // the binned plane as an irregular dip too.
        let mut data = flat(4);
        for y in 120..125 {
            for x in 150..210 {
                data[y * WIDTH + x] *= 0.9;
            }
        }
        let measurement = measure(&data);
        let near: Vec<_> = measurement
            .features
            .iter()
            .filter(|f| (f.x - 180.0).abs() < 10.0 && (f.y - 122.0).abs() < 10.0)
            .collect();
        assert_eq!(near.len(), 1, "{near:?}");
        assert_eq!(measurement.feature_count, 1, "{:?}", measurement.features);
    }

    #[test]
    fn a_deep_round_shadow_is_reported_once() {
        // A round shadow deep enough for the pixel pass (8%, 24 px across)
        // is also a disk on the binned plane.
        let mut data = flat(4);
        dip(&mut data, 150.0, 120.0, 12.0, 0.08);
        let measurement = measure(&data);
        let near: Vec<_> = measurement
            .features
            .iter()
            .filter(|f| (f.x - 150.0).abs() < 12.0 && (f.y - 120.0).abs() < 12.0)
            .collect();
        assert_eq!(near.len(), 1, "{near:?}");
    }

    #[test]
    fn a_wide_shallow_disk_is_reported_before_the_compact_dips() {
        // Filter dust: 0.5% deep and 80 pixels across, far below the 2%
        // compact threshold, beside a droplet.
        let mut data = flat(4);
        dip(&mut data, 150.0, 120.0, 40.0, 0.005);
        dip(&mut data, 320.0, 60.0, 3.5, 0.5);
        let measurement = measure(&data);
        assert_eq!(measurement.feature_count, 2, "{:?}", measurement.features);
        let disk = measurement.features[0];
        assert!(disk.shape.is_large(), "{disk:?}");
        assert!(
            (disk.x - 150.0).abs() < 3.0 && (disk.radius - 40.0).abs() < 4.0,
            "{:?}",
            measurement.features
        );
        assert!(!measurement.features[1].shape.is_large());
        // Screening counts only the droplet.
        let report = screen_flats(&[measurement], &FlatScreening::default()).unwrap();
        assert_eq!(report.frames[0].grade, Grade::Ok);
        let compact_only = measure_flat(
            LinearImageRef::new(&data, WIDTH, HEIGHT, 1).unwrap(),
            false,
            &FlatMeasureOptions {
                large_features: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(compact_only.feature_count, 1);
    }

    #[test]
    fn a_cfa_frame_is_measured_on_its_2x2_sums() {
        // A Bayer flat: red and blue sites at half the green response.
        let mut data = flat(3);
        for (index, sample) in data.iter_mut().enumerate() {
            let (x, y) = (index % WIDTH, index / WIDTH);
            if (x + y) % 2 == 0 {
                *sample *= 0.5;
            }
        }
        dip(&mut data, 200.0, 120.0, 4.0, 0.5);
        let measurement = measure_flat(
            LinearImageRef::new(&data, WIDTH, HEIGHT, 1).unwrap(),
            true,
            &FlatMeasureOptions::default(),
        )
        .unwrap();
        assert_eq!(measurement.channel_levels.len(), 4);
        let ratio = measurement.channel_levels[1] / measurement.channel_levels[0];
        assert!((ratio - 2.0).abs() < 0.1, "{ratio}");
        assert_eq!(measurement.feature_count, 1, "{:?}", measurement.features);
        let feature = measurement.features[0];
        assert!((feature.x - 200.0).abs() < 2.0 && (feature.y - 120.0).abs() < 2.0);
    }

    #[test]
    fn fading_droplets_mark_the_frames_that_hold_them() {
        // Twenty frames. Forty droplets dry out over the first nine; by the
        // tenth frame they are gone. A median master would reject them.
        let droplets = spots(40);
        let frames: Vec<FlatMeasurement> = (0..20)
            .map(|frame| {
                let mut data = flat(10 + frame as u64);
                if frame < 9 {
                    let depth = 0.6 * (1.0 - frame as f32 / 9.0);
                    for &(x, y) in &droplets {
                        dip(&mut data, x, y, 3.0, depth);
                    }
                }
                measure(&data)
            })
            .collect();
        let report = screen_flats(&frames, &FlatScreening::default()).unwrap();
        for (frame, verdict) in report.frames.iter().enumerate() {
            if frame < 9 {
                assert_eq!(verdict.grade, Grade::Poor, "frame {frame}: {verdict:?}");
                assert_eq!(
                    codes(verdict),
                    vec![(FindingCode::ChangingFeatures, Grade::Poor)]
                );
            } else {
                assert_eq!(verdict.grade, Grade::Ok, "frame {frame}: {verdict:?}");
            }
        }
        assert_eq!(report.changing_features, 40);
        assert_eq!(report.carried_features, 0);
        assert_eq!(report.master.grade, Grade::Ok);
    }

    #[test]
    fn droplets_through_most_of_a_set_mark_the_master() {
        // The droplets fade across all twenty frames but never quite go, so
        // every frame holds them and a median keeps them.
        let droplets = spots(40);
        let frames: Vec<FlatMeasurement> = (0..20)
            .map(|frame| {
                let mut data = flat(40 + frame as u64);
                let depth = 0.6 - 0.025 * frame as f32;
                for &(x, y) in &droplets {
                    dip(&mut data, x, y, 3.0, depth);
                }
                measure(&data)
            })
            .collect();
        let report = screen_flats(&frames, &FlatScreening::default()).unwrap();
        assert!(
            report
                .frames
                .iter()
                .all(|verdict| verdict.grade == Grade::Poor)
        );
        assert_eq!(report.carried_features, 40);
        assert_eq!(
            codes(&report.master),
            vec![(FindingCode::CarriedFeatures, Grade::Poor)]
        );
        assert_eq!(report.steady_features, 0);
    }

    #[test]
    fn droplets_that_hold_still_are_caught_by_their_number() {
        // Droplets that neither fade nor move look steady, like dust. Their
        // number gives them away.
        let droplets = spots(40);
        let frames: Vec<FlatMeasurement> = (0..5)
            .map(|frame| {
                let mut data = flat(180 + frame as u64);
                for &(x, y) in &droplets {
                    dip(&mut data, x, y, 3.0, 0.5);
                }
                measure(&data)
            })
            .collect();
        let report = screen_flats(&frames, &FlatScreening::default()).unwrap();
        assert!(
            report
                .frames
                .iter()
                .all(|verdict| verdict.grade == Grade::Ok)
        );
        assert_eq!(report.steady_features, 40);
        assert_eq!(report.changing_features, 0);
        let strict = FlatScreening {
            features: Limits::new(20.0, 30.0),
            ..Default::default()
        };
        let report = screen_flats(&frames, &strict).unwrap();
        for verdict in &report.frames {
            assert_eq!(codes(verdict), vec![(FindingCode::Features, Grade::Poor)]);
        }
        assert_eq!(report.master.grade, Grade::Ok);
    }

    #[test]
    fn parts_two_pixels_apart_are_one_feature() {
        let run = |y, start, end| Run {
            y,
            start,
            end,
            lowest: -0.5,
        };
        let count = |runs: &[Run]| features_from_runs(runs, 1).len();
        // Two empty columns between runs on a row join them; three do not.
        assert_eq!(count(&[run(5, 0, 3), run(5, 5, 8)]), 1);
        assert_eq!(count(&[run(5, 0, 3), run(5, 6, 8)]), 2);
        // Two empty rows between runs join them; three do not.
        assert_eq!(count(&[run(5, 0, 3), run(8, 0, 3)]), 1);
        assert_eq!(count(&[run(5, 0, 3), run(9, 0, 3)]), 2);
        // Diagonal neighbours within reach join, and the box covers both.
        let joined = features_from_runs(&[run(5, 0, 3), run(7, 5, 7)], 2);
        assert_eq!(joined.len(), 1);
        let feature = joined[0];
        assert_eq!(
            (feature.left, feature.top, feature.right, feature.bottom),
            (0, 10, 13, 15)
        );
        assert_eq!(feature.pixels, 20);
    }

    #[test]
    fn steady_dust_is_counted_but_not_held_against_anyone() {
        // Dust shadows of several sizes and depths in every frame of a
        // session: small specks and wide, shallow donuts.
        let frames: Vec<FlatMeasurement> = (0..10)
            .map(|frame| {
                let mut data = flat(70 + frame as u64);
                dip(&mut data, 60.0, 60.0, 4.0, 0.15);
                dip(&mut data, 250.0, 70.0, 12.0, 0.06);
                dip(&mut data, 150.0, 180.0, 18.0, 0.08);
                dip(&mut data, 150.0, 180.0, 9.0, -0.04);
                dip(&mut data, 320.0, 200.0, 40.0, 0.03);
                measure(&data)
            })
            .collect();
        assert!(frames.iter().all(|frame| frame.feature_count >= 3));
        let report = screen_flats(&frames, &FlatScreening::default()).unwrap();
        assert!(
            report
                .frames
                .iter()
                .all(|verdict| verdict.grade == Grade::Ok),
            "{:?}",
            report.frames
        );
        assert_eq!(report.master.grade, Grade::Ok);
        // The wide round shadows are disks and donuts, reported once as large
        // dust, which the light path holds and screening leaves alone; the
        // speck is a steady dip.
        assert!(
            frames
                .iter()
                .all(|frame| frame.features.iter().filter(|f| f.shape.is_large()).count() >= 2),
            "{:?}",
            frames[0].features
        );
        assert!(report.steady_features >= 1, "{report:?}");
        assert_eq!(report.changing_features, 0);
    }

    #[test]
    fn clipped_and_dim_flats_are_poor() {
        let good = flat(100);
        let clipped: Vec<f32> = good
            .iter()
            .map(|value| (value * 2.5).min(65535.0))
            .collect();
        let dim: Vec<f32> = good.iter().map(|value| value * 0.05).collect();
        let bright: Vec<f32> = good.iter().map(|value| value * 1.9).collect();
        let frames = [
            measure(&good),
            measure(&clipped),
            measure(&dim),
            measure(&bright),
        ];
        assert!(frames[1].clipped_fraction > 0.5);
        let report = screen_flats(&frames, &FlatScreening::default()).unwrap();
        assert_eq!(report.frames[0].grade, Grade::Ok, "{:?}", report.frames[0]);
        assert!(codes(&report.frames[1]).contains(&(FindingCode::Clipped, Grade::Poor)));
        assert!(codes(&report.frames[1]).contains(&(FindingCode::Bright, Grade::Poor)));
        assert_eq!(
            codes(&report.frames[2]),
            vec![(FindingCode::Dim, Grade::Poor)]
        );
        assert_eq!(
            codes(&report.frames[3]),
            vec![(FindingCode::Bright, Grade::Suspect)]
        );
    }

    #[test]
    fn a_shifted_pattern_stands_out_from_its_peers() {
        // One frame lit from the side, as a moved panel or a light leak does.
        let frames: Vec<FlatMeasurement> = (0..5)
            .map(|frame| {
                let mut data = flat(120 + frame as u64);
                if frame == 3 {
                    for (index, sample) in data.iter_mut().enumerate() {
                        let x = (index % WIDTH) as f32 / WIDTH as f32;
                        *sample *= 0.94 + 0.12 * x;
                    }
                }
                measure(&data)
            })
            .collect();
        let report = screen_flats(&frames, &FlatScreening::default()).unwrap();
        for (frame, verdict) in report.frames.iter().enumerate() {
            if frame == 3 {
                assert_eq!(
                    codes(verdict),
                    vec![(FindingCode::Illumination, Grade::Poor)]
                );
            } else {
                assert_eq!(verdict.grade, Grade::Ok, "frame {frame}: {verdict:?}");
            }
        }
    }

    #[test]
    fn two_frames_are_judged_only_on_their_own() {
        let mut first = flat(150);
        dip(&mut first, 100.0, 100.0, 4.0, 0.5);
        let frames = [measure(&first), measure(&flat(151))];
        let report = screen_flats(&frames, &FlatScreening::default()).unwrap();
        assert!(
            report
                .frames
                .iter()
                .all(|verdict| verdict.grade == Grade::Ok)
        );
        assert_eq!(report.changing_features, 0);
    }

    #[test]
    fn measurements_round_trip_and_mismatched_sets_are_refused() {
        let measurement = measure(&flat(160));
        let json = serde_json::to_string(&measurement).unwrap();
        assert_eq!(
            serde_json::from_str::<FlatMeasurement>(&json).unwrap(),
            measurement
        );
        let mut other = measurement.clone();
        other.width += 2;
        assert!(screen_flats(&[measurement, other], &FlatScreening::default()).is_err());
    }

    #[test]
    fn bad_options_and_small_frames_are_refused() {
        let data = vec![1.0_f32; 16 * 16];
        let image = LinearImageRef::new(&data, 16, 16, 1).unwrap();
        assert!(measure_flat(image, false, &FlatMeasureOptions::default()).is_err());
        let data = flat(170);
        let image = LinearImageRef::new(&data, WIDTH, HEIGHT, 1).unwrap();
        for options in [
            FlatMeasureOptions {
                full_scale: 0.0,
                ..Default::default()
            },
            FlatMeasureOptions {
                feature_depth: 0.0,
                ..Default::default()
            },
        ] {
            assert!(measure_flat(image, false, &options).is_err());
        }
        let screening = FlatScreening {
            steady_presence: 0.2,
            ..Default::default()
        };
        assert!(screen_flats(&[], &screening).is_err());
    }
}
