//! Bias and dark frames: level, noise, banding, and hot pixels.
//!
//! A bias is a dark of zero length, so one measurement serves both. Frames are
//! measured raw, as the camera wrote them; a CFA frame needs no special
//! handling because no light reaches it.

use super::{
    FRAME_QUALITY_ALGORITHM_VERSION, Finding, FindingCode, FrameVerdict, Limits, MIN_PEERS,
    frame_quality_algorithm_version, median_of,
};
use crate::{Error, LinearImageRef, Result};
use rayon::prelude::*;
use seiza_stats::{NORMAL_MAD_SCALE, median_in_place, robust_sigma_in_place};
use serde::{Deserialize, Serialize};

/// Read every this many samples to find the level, as
/// `seiza_stacking::DarkLevel` does, so the two agree.
const LEVEL_STRIDE: usize = 16;
/// Most rows read to estimate noise.
const NOISE_ROWS: usize = 512;
/// Most second differences kept from those rows along each axis.
const NOISE_SAMPLES: usize = 1 << 20;
/// Fewest second differences a noise estimate needs.
const MIN_NOISE_SAMPLES: usize = 1024;
/// A second difference has coefficients 1, -2, 1, so independent samples
/// with noise sigma give a difference with sqrt(6) sigma.
const SECOND_DIFFERENCE_GAIN_SQUARED: f64 = 6.0;
/// Second differences further than this many robust sigmas from their median
/// are left out of the noise.
const DIFFERENCE_CLIP_SIGMA: f64 = 4.0;
/// Samples further than this many pixel spreads from the level are left out
/// of row and column means, so hot pixels and cosmic rays do not read as
/// banding.
const PROFILE_CLIP_SIGMA: f32 = 5.0;
/// Rows or columns in the running median that removes slow trends, such as
/// amplifier glow, before banding is measured.
const TREND_WINDOW: usize = 33;

/// Settings for [`measure_dark`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DarkMeasureOptions {
    /// A sample this many times the pixel noise above the level counts as a
    /// hot pixel.
    pub hot_sigma: f32,
}

impl Default for DarkMeasureOptions {
    fn default() -> Self {
        Self { hot_sigma: 10.0 }
    }
}

/// What [`measure_dark`] found in one bias or dark frame. Values are in the
/// frame's own sample units.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DarkMeasurement {
    /// [`super::FRAME_QUALITY_ALGORITHM_VERSION`] when this was measured.
    #[serde(default = "frame_quality_algorithm_version")]
    pub algorithm_version: u32,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Median of every sixteenth sample: the estimator
    /// `seiza_stacking::DarkLevel` uses.
    pub level: f32,
    /// Normal-equivalent median absolute deviation of those samples around
    /// the level. It includes glow and pattern as well as noise.
    pub sigma: f32,
    /// Pixel-scale noise from second differences, along whichever axis is
    /// quieter so banding does not inflate it. Outliers are clipped, and the
    /// rest give a root mean square, which moves smoothly even on integer
    /// samples.
    pub noise: f32,
    /// Robust spread of row means once a slow trend is removed.
    pub row_banding: f32,
    /// Robust spread of column means once a slow trend is removed.
    pub column_banding: f32,
    /// Samples above [`Self::hot_threshold`].
    pub hot_pixels: u64,
    /// The level plus [`DarkMeasureOptions::hot_sigma`] times the noise.
    pub hot_threshold: f32,
}

/// Measure one bias or dark frame. Non-finite samples are ignored.
pub fn measure_dark(
    image: LinearImageRef<'_>,
    options: &DarkMeasureOptions,
) -> Result<DarkMeasurement> {
    if !options.hot_sigma.is_finite() || options.hot_sigma <= 0.0 {
        return Err(Error::FrameQuality(
            "hot_sigma must be finite and positive".into(),
        ));
    }
    let (width, height, channels) = (image.width(), image.height(), image.channels());
    if width < 3 || height < 3 {
        return Err(Error::InvalidImage(
            "a dark must be at least 3 pixels on each side to measure".into(),
        ));
    }
    let data = image.data();
    let mut samples: Vec<f32> = data
        .iter()
        .step_by(LEVEL_STRIDE)
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let level = median_in_place(&mut samples).unwrap_or(0.0);
    let sigma = robust_sigma_in_place(&mut samples, level).unwrap_or(0.0);
    let noise = pixel_noise(image).unwrap_or(sigma);

    let spread = sigma.max(noise);
    let (low, high) = (
        level - PROFILE_CLIP_SIGMA * spread,
        level + PROFILE_CLIP_SIGMA * spread,
    );
    let hot_threshold = level + options.hot_sigma * noise;
    let row_width = width * channels;
    let profiles = data
        .par_chunks(row_width)
        // Each task keeps a column profile of its own; bound how many.
        .with_min_len(64)
        .fold(
            || Profiles::new(width),
            |mut profiles, row| {
                let (mut sum, mut count) = (0.0_f64, 0_u32);
                for (index, value) in row.iter().enumerate() {
                    if *value > hot_threshold {
                        profiles.hot_pixels += 1;
                    }
                    if *value >= low && *value <= high {
                        let column = index / channels;
                        profiles.column_sums[column] += f64::from(*value);
                        profiles.column_counts[column] += 1;
                        sum += f64::from(*value);
                        count += 1;
                    }
                }
                profiles.rows.push(if count == 0 {
                    f64::NAN
                } else {
                    sum / f64::from(count)
                });
                profiles
            },
        )
        .collect::<Vec<_>>();
    let mut rows = Vec::with_capacity(height);
    let mut column_sums = vec![0.0_f64; width];
    let mut column_counts = vec![0_u32; width];
    let mut hot_pixels = 0;
    for part in profiles {
        rows.extend(part.rows);
        for (total, sum) in column_sums.iter_mut().zip(part.column_sums) {
            *total += sum;
        }
        for (total, count) in column_counts.iter_mut().zip(part.column_counts) {
            *total += count;
        }
        hot_pixels += part.hot_pixels;
    }
    let columns: Vec<f64> = column_sums
        .iter()
        .zip(&column_counts)
        .map(|(sum, count)| {
            if *count == 0 {
                f64::NAN
            } else {
                sum / f64::from(*count)
            }
        })
        .collect();

    Ok(DarkMeasurement {
        algorithm_version: FRAME_QUALITY_ALGORITHM_VERSION,
        width,
        height,
        level,
        sigma,
        noise,
        row_banding: banding(&rows),
        column_banding: banding(&columns),
        hot_pixels,
        hot_threshold,
    })
}

/// Limits for [`screen_bias`]. Each [`Limits`] says where a frame turns
/// suspect and then poor; for each, higher is worse.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BiasScreening {
    /// Distance of a frame's level from the set median, in units of the set's
    /// median pixel noise.
    pub level: Limits,
    /// Pixel noise as a ratio to the set median.
    pub noise: Limits,
    /// Row or column banding as a ratio to the frame's own pixel noise.
    pub banding: Limits,
}

impl Default for BiasScreening {
    fn default() -> Self {
        Self {
            level: Limits::new(1.0, 3.0),
            noise: Limits::new(1.1, 1.3),
            banding: Limits::new(0.25, 0.5),
        }
    }
}

/// Judge a set of bias frames that should match: one camera, gain, offset,
/// readout mode, and binning.
///
/// Banding is judged on each frame alone. With three or more frames, each is
/// also compared with the set median for level and noise. Returns one verdict
/// per frame, in input order.
pub fn screen_bias(frames: &[DarkMeasurement], screening: &BiasScreening) -> Vec<FrameVerdict> {
    let level = median_of(frames.iter().map(|frame| f64::from(frame.level)));
    let noise = median_of(frames.iter().map(|frame| f64::from(frame.noise)));
    let peers = frames.len() >= MIN_PEERS && noise > 0.0;
    frames
        .iter()
        .map(|frame| {
            let mut findings = Vec::new();
            if peers {
                screening.level.check_above(
                    &mut findings,
                    FindingCode::Level,
                    (f64::from(frame.level) - level).abs() / noise,
                );
                screening.noise.check_above(
                    &mut findings,
                    FindingCode::Noise,
                    f64::from(frame.noise) / noise,
                );
            }
            check_banding(&mut findings, screening.banding, frame);
            FrameVerdict::from_findings(findings)
        })
        .collect()
}

/// Limits for [`screen_darks`]. Each [`Limits`] says where a frame turns
/// suspect and then poor; for each, higher is worse.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DarkScreening {
    /// How far a frame's level sits above the quietest frame's, in units of
    /// that frame's [`DarkMeasurement::sigma`]. The poor limit matches
    /// `seiza_stacking::DarkLevelScreening`'s default.
    pub level_excess: Limits,
    /// Pixel noise as a ratio to the median of the frames whose level is
    /// below the suspect level excess.
    pub noise: Limits,
    /// Hot pixel count as a ratio to the median of the frames whose level is
    /// below the suspect level excess.
    pub hot_pixels: Limits,
    /// Row or column banding as a ratio to the frame's own pixel noise.
    pub banding: Limits,
}

impl Default for DarkScreening {
    fn default() -> Self {
        Self {
            level_excess: Limits::new(0.5, 1.0),
            noise: Limits::new(1.1, 1.3),
            hot_pixels: Limits::new(1.5, 3.0),
            banding: Limits::new(0.25, 0.5),
        }
    }
}

/// Judge a set of darks that should match: one camera mode, one exposure,
/// one sensor temperature.
///
/// Banding is judged on each frame alone. With two or more frames, each
/// frame's level is compared with the quietest frame's, which is the rule
/// `seiza_stacking::screen_dark_levels` applies: light can only add to a
/// dark, so the quietest frame is the one to trust. With three or more, noise
/// and hot pixels are compared with the median of the frames whose level
/// passed. Light raises the noise and so the hot pixel threshold, and a set
/// where most frames caught light would otherwise make its one good frame
/// look hot. Returns one verdict per frame, in input order.
pub fn screen_darks(frames: &[DarkMeasurement], screening: &DarkScreening) -> Vec<FrameVerdict> {
    let quietest = frames
        .iter()
        .min_by(|a, b| a.level.total_cmp(&b.level))
        .map(|frame| (frame.level, frame.sigma));
    let excess = |frame: &DarkMeasurement| match quietest {
        Some((level, sigma)) if sigma > 0.0 => f64::from(frame.level - level) / f64::from(sigma),
        Some((level, _)) if frame.level <= level => 0.0,
        _ => f64::INFINITY,
    };
    let clean: Vec<&DarkMeasurement> = frames
        .iter()
        .filter(|frame| excess(frame) < screening.level_excess.suspect)
        .collect();
    let noise = median_of(clean.iter().map(|frame| f64::from(frame.noise)));
    let hot = median_of(clean.iter().map(|frame| frame.hot_pixels as f64));
    let peers = frames.len() >= MIN_PEERS;
    frames
        .iter()
        .map(|frame| {
            let mut findings = Vec::new();
            if frames.len() >= 2 && quietest.is_some_and(|(_, sigma)| sigma > 0.0) {
                screening.level_excess.check_above(
                    &mut findings,
                    FindingCode::LevelExcess,
                    excess(frame),
                );
            }
            if peers && noise > 0.0 {
                screening.noise.check_above(
                    &mut findings,
                    FindingCode::Noise,
                    f64::from(frame.noise) / noise,
                );
            }
            if peers && hot.is_finite() {
                // One hot pixel against none is no evidence; count from one.
                screening.hot_pixels.check_above(
                    &mut findings,
                    FindingCode::HotPixels,
                    frame.hot_pixels as f64 / hot.max(1.0),
                );
            }
            check_banding(&mut findings, screening.banding, frame);
            FrameVerdict::from_findings(findings)
        })
        .collect()
}

fn check_banding(findings: &mut Vec<Finding>, limits: Limits, frame: &DarkMeasurement) {
    if frame.noise <= 0.0 {
        return;
    }
    let noise = f64::from(frame.noise);
    limits.check_above(
        findings,
        FindingCode::RowBanding,
        f64::from(frame.row_banding) / noise,
    );
    limits.check_above(
        findings,
        FindingCode::ColumnBanding,
        f64::from(frame.column_banding) / noise,
    );
}

/// Row means, column sums, and hot pixels from one run of rows.
struct Profiles {
    rows: Vec<f64>,
    column_sums: Vec<f64>,
    column_counts: Vec<u32>,
    hot_pixels: u64,
}

impl Profiles {
    fn new(width: usize) -> Self {
        Self {
            rows: Vec::new(),
            column_sums: vec![0.0; width],
            column_counts: vec![0; width],
            hot_pixels: 0,
        }
    }
}

/// Pixel noise from second differences along each axis, averaged over
/// channels; the quieter axis of each channel is kept. `None` when the frame
/// gives too few differences.
fn pixel_noise(image: LinearImageRef<'_>) -> Option<f32> {
    let (width, height, channels) = (image.width(), image.height(), image.channels());
    let data = image.data();
    let stride = (height - 2).div_ceil(NOISE_ROWS).max(1);
    let rows: Vec<usize> = (1..height - 1).step_by(stride).collect();
    let column_stride = (rows.len() * (width - 2)).div_ceil(NOISE_SAMPLES).max(1);
    let per_channel: Vec<f64> = (0..channels)
        .map(|channel| {
            let (mut horizontal, mut vertical) = rows
                .par_iter()
                .map(|&y| {
                    let mut horizontal = Vec::with_capacity(width / column_stride + 1);
                    let mut vertical = Vec::with_capacity(width / column_stride + 1);
                    for x in (1..width - 1).step_by(column_stride) {
                        let index = (y * width + x) * channels + channel;
                        let value = data[index];
                        let (left, right) = (data[index - channels], data[index + channels]);
                        let (above, below) = (
                            data[index - width * channels],
                            data[index + width * channels],
                        );
                        if [value, left, right].iter().all(|v| v.is_finite()) {
                            horizontal.push(left - 2.0 * value + right);
                        }
                        if [value, above, below].iter().all(|v| v.is_finite()) {
                            vertical.push(above - 2.0 * value + below);
                        }
                    }
                    (horizontal, vertical)
                })
                .reduce(
                    || (Vec::new(), Vec::new()),
                    |mut a, mut b| {
                        a.0.append(&mut b.0);
                        a.1.append(&mut b.1);
                        a
                    },
                );
            match (
                second_difference_noise(&mut horizontal),
                second_difference_noise(&mut vertical),
            ) {
                (Some(h), Some(v)) => Some(h.min(v)),
                _ => None,
            }
        })
        .collect::<Option<_>>()?;
    Some((per_channel.iter().sum::<f64>() / per_channel.len() as f64) as f32)
}

/// The noise second differences imply. The median absolute deviation sets a
/// clipping range that keeps hot pixels and cosmic rays out, but on integer
/// samples it moves in whole steps, an eighth of the noise or more; the root
/// mean square of the differences inside that range moves smoothly.
fn second_difference_noise(differences: &mut [f32]) -> Option<f64> {
    if differences.len() < MIN_NOISE_SAMPLES {
        return None;
    }
    let centre = median_in_place(differences)?;
    let mut deviations: Vec<f32> = differences
        .iter()
        .map(|difference| (difference - centre).abs())
        .collect();
    let spread = f64::from(median_in_place(&mut deviations)?) * NORMAL_MAD_SCALE;
    let limit = if spread > 0.0 {
        DIFFERENCE_CLIP_SIGMA * spread
    } else {
        f64::INFINITY
    };
    let (sum, count) = deviations
        .iter()
        .map(|deviation| f64::from(*deviation))
        .filter(|deviation| *deviation <= limit)
        .fold((0.0, 0_u64), |(sum, count), deviation| {
            (sum + deviation * deviation, count + 1)
        });
    (count > 0).then(|| (sum / count as f64 / SECOND_DIFFERENCE_GAIN_SQUARED).sqrt())
}

/// Robust spread of a profile around its running median.
fn banding(profile: &[f64]) -> f32 {
    let half = TREND_WINDOW / 2;
    let mut window = Vec::with_capacity(TREND_WINDOW);
    let residuals: Vec<f64> = (0..profile.len())
        .filter(|index| profile[*index].is_finite())
        .map(|index| {
            let (start, end) = (
                index.saturating_sub(half),
                (index + half + 1).min(profile.len()),
            );
            window.clear();
            window.extend(
                profile[start..end]
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite()),
            );
            let trend = seiza_stats::median_f64(&window).unwrap_or(profile[index]);
            profile[index] - trend
        })
        .collect();
    if residuals.len() < 3 {
        return 0.0;
    }
    let centre = seiza_stats::median_f64(&residuals).unwrap_or(0.0);
    seiza_stats::robust_sigma_f64(&residuals, centre).unwrap_or(0.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality::Grade;

    const WIDTH: usize = 300;
    const HEIGHT: usize = 200;

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

    /// A frame at `level` with Gaussian read noise and `hot` hot pixels.
    fn frame(seed: u64, level: f32, noise: f32, hot: usize) -> Vec<f32> {
        let mut data: Vec<f32> = (0..WIDTH * HEIGHT)
            .map(|index| level + noise * gaussian(index, seed))
            .collect();
        let samples = data.len();
        for index in 0..hot {
            data[(index * 7919) % samples] = level + 500.0;
        }
        data
    }

    fn measure(data: &[f32]) -> DarkMeasurement {
        measure_dark(
            LinearImageRef::new(data, WIDTH, HEIGHT, 1).unwrap(),
            &DarkMeasureOptions::default(),
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
    fn a_clean_bias_reads_its_level_and_noise() {
        let measurement = measure(&frame(1, 500.0, 4.0, 0));
        assert!((measurement.level - 500.0).abs() < 0.2);
        assert!(
            (measurement.noise - 4.0).abs() < 0.2,
            "{}",
            measurement.noise
        );
        assert!((measurement.sigma - 4.0).abs() < 0.2);
        // White noise leaves row means spread by about 4 / sqrt(300).
        assert!(measurement.row_banding < 0.4, "{}", measurement.row_banding);
        assert!(
            measurement.column_banding < 0.5,
            "{}",
            measurement.column_banding
        );
        assert_eq!(measurement.hot_pixels, 0);
    }

    #[test]
    fn bias_banding_and_outliers_are_found() {
        let mut frames: Vec<Vec<f32>> = (0..6).map(|seed| frame(seed, 500.0, 4.0, 0)).collect();
        // Frame 1: rows offset by up to three counts, as interference does.
        for (index, sample) in frames[1].iter_mut().enumerate() {
            *sample += 3.0 * gaussian(index / WIDTH, 99);
        }
        // Frame 2: column stripes.
        for (index, sample) in frames[2].iter_mut().enumerate() {
            *sample += 2.5 * gaussian(index % WIDTH, 98);
        }
        // Frame 3: read at a higher offset. Frame 4: noisier.
        for sample in &mut frames[3] {
            *sample += 20.0;
        }
        frames[4] = frame(40, 500.0, 6.0, 0);
        let measurements: Vec<DarkMeasurement> = frames.iter().map(|data| measure(data)).collect();
        let verdicts = screen_bias(&measurements, &BiasScreening::default());
        assert_eq!(verdicts[0].grade, Grade::Ok, "{:?}", verdicts[0]);
        assert_eq!(verdicts[5].grade, Grade::Ok, "{:?}", verdicts[5]);
        assert_eq!(
            codes(&verdicts[1]),
            vec![(FindingCode::RowBanding, Grade::Poor)]
        );
        assert_eq!(
            codes(&verdicts[2]),
            vec![(FindingCode::ColumnBanding, Grade::Poor)]
        );
        assert_eq!(codes(&verdicts[3]), vec![(FindingCode::Level, Grade::Poor)]);
        assert_eq!(codes(&verdicts[4]), vec![(FindingCode::Noise, Grade::Poor)]);
    }

    #[test]
    fn dark_hot_pixels_and_stray_light_stand_out() {
        let mut frames: Vec<Vec<f32>> = (0..5).map(|seed| frame(seed, 600.0, 5.0, 200)).collect();
        frames[1] = frame(11, 600.0, 5.0, 900);
        // Frame 3 caught light: its level rose by two pixel spreads.
        for sample in &mut frames[3] {
            *sample += 10.0;
        }
        let measurements: Vec<DarkMeasurement> = frames.iter().map(|data| measure(data)).collect();
        assert_eq!(measurements[0].hot_pixels, 200);
        let verdicts = screen_darks(&measurements, &DarkScreening::default());
        for index in [0, 2, 4] {
            assert_eq!(verdicts[index].grade, Grade::Ok, "{:?}", verdicts[index]);
        }
        assert_eq!(
            codes(&verdicts[1]),
            vec![(FindingCode::HotPixels, Grade::Poor)]
        );
        assert_eq!(
            codes(&verdicts[3]),
            vec![(FindingCode::LevelExcess, Grade::Poor)]
        );
    }

    #[test]
    fn light_in_most_frames_does_not_make_the_good_one_look_hot() {
        // Warm pixels at three heights above the level. Light raises a
        // frame's noise and with it the hot pixel threshold, so the frames
        // that caught light count fewer of them.
        let warm = |data: &mut [f32], level: f32| {
            for (index, height) in [(0, 60.0), (1, 150.0), (2, 500.0)] {
                let count = if index == 2 { 20 } else { 100 };
                for k in 0..count {
                    let at = ((index * 100 + k) * 7919 + 13) % data.len();
                    data[at] = level + height;
                }
            }
        };
        let mut frames = [
            frame(1, 600.0, 5.0, 0),
            frame(2, 610.0, 8.0, 0),
            frame(3, 650.0, 20.0, 0),
        ];
        for (data, level) in frames.iter_mut().zip([600.0, 610.0, 650.0]) {
            warm(data.as_mut_slice(), level);
        }
        let measurements: Vec<DarkMeasurement> = frames.iter().map(|data| measure(data)).collect();
        assert_eq!(measurements[0].hot_pixels, 220);
        assert!(measurements[2].hot_pixels < 100);
        let verdicts = screen_darks(&measurements, &DarkScreening::default());
        assert_eq!(verdicts[0].grade, Grade::Ok, "{:?}", verdicts[0]);
        for verdict in &verdicts[1..] {
            assert_eq!(verdict.grade, Grade::Poor, "{verdict:?}");
            assert!(codes(verdict).contains(&(FindingCode::LevelExcess, Grade::Poor)));
            assert!(codes(verdict).contains(&(FindingCode::Noise, Grade::Poor)));
        }
    }

    #[test]
    fn noise_moves_smoothly_on_integer_samples() {
        // Rounded samples quantize a median absolute deviation into whole
        // steps; the clipped root mean square follows the true noise.
        for (seed, sigma) in [(1, 3.6_f32), (2, 3.9), (3, 4.2)] {
            let data: Vec<f32> = frame(seed, 500.0, sigma, 0)
                .into_iter()
                .map(f32::round)
                .collect();
            let noise = measure(&data).noise;
            assert!((noise - sigma).abs() < 0.1, "{sigma}: {noise}");
        }
    }

    #[test]
    fn small_sets_skip_peer_checks() {
        let measurements = [
            measure(&frame(1, 500.0, 4.0, 0)),
            measure(&frame(2, 530.0, 9.0, 0)),
        ];
        assert!(
            screen_bias(&measurements, &BiasScreening::default())
                .iter()
                .all(|verdict| verdict.grade == Grade::Ok)
        );
        // Two darks still show which caught light.
        let verdicts = screen_darks(&measurements, &DarkScreening::default());
        assert_eq!(verdicts[0].grade, Grade::Ok);
        assert_eq!(
            codes(&verdicts[1]),
            vec![(FindingCode::LevelExcess, Grade::Poor)]
        );
    }

    #[test]
    fn rgb_frames_and_bad_options() {
        let data: Vec<f32> = (0..WIDTH * HEIGHT * 3)
            .map(|index| 100.0 + 2.0 * gaussian(index, 7))
            .collect();
        let measurement = measure_dark(
            LinearImageRef::new(&data, WIDTH, HEIGHT, 3).unwrap(),
            &DarkMeasureOptions::default(),
        )
        .unwrap();
        assert!(
            (measurement.noise - 2.0).abs() < 0.1,
            "{}",
            measurement.noise
        );
        let image = LinearImageRef::new(&data, WIDTH, HEIGHT, 3).unwrap();
        assert!(measure_dark(image, &DarkMeasureOptions { hot_sigma: 0.0 }).is_err());
        let json = serde_json::to_string(&measurement).unwrap();
        assert_eq!(
            serde_json::from_str::<DarkMeasurement>(&json).unwrap(),
            measurement
        );
    }
}
