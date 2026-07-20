//! Pixel evidence for orbital satellite-track predictions.
//!
//! The orbital path remains the provenance-bearing prediction. This module
//! searches only a bounded corridor around that path and reports pixel
//! evidence separately. It does not identify arbitrary trails without an
//! orbital candidate.

use std::{error::Error, fmt};

use crate::{PixelPoint, PixelSegment};

/// Version of the alignment algorithm and its reported evidence semantics.
pub const PIXEL_TRAIL_ALIGNMENT_VERSION: u32 = 3;

// Search the previous default endpoint-tilt range at every mean translation.
// Keeping this range independent of the full corridor makes the work linear in
// orbital position uncertainty without sacrificing angular coverage.
const COARSE_TILT_SEARCH_RADIUS_WORKING_PX: f64 = 32.0;

/// Whether the predicted path was tested and supported by image pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum PixelTrailAlignmentStatus {
    Detected,
    NotDetected,
    NotEvaluated,
}

/// Why a prediction could not be evaluated for pixel evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum PixelTrailNotEvaluatedReason {
    EmptyPath,
    TooShort,
    InsufficientCoverage,
}

/// Tunables for prediction-constrained pixel alignment.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PixelTrailAlignmentConfig {
    /// Longest side of the internally downsampled image.
    pub maximum_working_dimension: usize,
    /// Normal search radius in working-image pixels.
    pub search_radius_working_px: f64,
    pub coarse_step_px: f64,
    pub refine_step_px: f64,
    /// Paths shorter than this are not evaluated.
    pub minimum_working_length_px: f64,
    pub minimum_samples: usize,
    pub maximum_samples: usize,
    /// Minimum fraction of sampled path positions that must have complete
    /// center-line and sideband coverage.
    pub minimum_coverage_fraction: f64,
    pub minimum_contrast_sigma: f64,
    pub minimum_continuity: f64,
}

impl Default for PixelTrailAlignmentConfig {
    fn default() -> Self {
        Self {
            maximum_working_dimension: 2_048,
            search_radius_working_px: 192.0,
            coarse_step_px: 2.0,
            refine_step_px: 0.5,
            minimum_working_length_px: 30.0,
            minimum_samples: 80,
            maximum_samples: 1_200,
            minimum_coverage_fraction: 0.5,
            minimum_contrast_sigma: 2.0,
            minimum_continuity: 0.65,
        }
    }
}

/// Pixel evidence associated with one predicted satellite track.
///
/// `aligned_segments` deliberately preserves the prediction's polyline
/// structure. It is populated only for a detection; the original prediction
/// remains unchanged.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PixelTrailAlignment {
    pub status: PixelTrailAlignmentStatus,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub not_evaluated_reason: Option<PixelTrailNotEvaluatedReason>,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub aligned_segments: Vec<PixelSegment>,
    pub start_normal_offset_px: f64,
    pub end_normal_offset_px: f64,
    pub mean_normal_offset_px: f64,
    pub angle_delta_deg: f64,
    pub contrast_adu: f64,
    pub contrast_sigma: f64,
    pub continuity: f64,
    /// Fraction of the sampled predicted path used by the fitted score.
    pub coverage: f64,
    pub search_radius_px: f64,
}

impl PixelTrailAlignment {
    pub fn detected(&self) -> bool {
        self.status == PixelTrailAlignmentStatus::Detected
    }

    pub fn evaluated(&self) -> bool {
        self.status != PixelTrailAlignmentStatus::NotEvaluated
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrailAlignmentError {
    InvalidDimensions,
    PixelCount { expected: usize, actual: usize },
    InvalidAduScale,
    InvalidConfiguration,
}

impl fmt::Display for TrailAlignmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions => write!(formatter, "image dimensions must be non-zero"),
            Self::PixelCount { expected, actual } => write!(
                formatter,
                "image pixel count mismatch: expected {expected}, received {actual}"
            ),
            Self::InvalidAduScale => {
                write!(formatter, "ADU per stored unit must be finite and positive")
            }
            Self::InvalidConfiguration => {
                write!(formatter, "invalid trail-alignment configuration")
            }
        }
    }
}

impl Error for TrailAlignmentError {}

/// Reusable per-frame working image. Construct once, then align every orbital
/// candidate without repeating the full-frame downsample.
pub struct PixelTrailAligner {
    working: WorkingImage,
    config: PixelTrailAlignmentConfig,
}

impl PixelTrailAligner {
    /// Build an aligner from row-major monochrome pixels.
    ///
    /// `adu_per_stored_unit` converts a pixel difference into physical ADU.
    /// For a source whose stored values were divided by a scale, pass the
    /// reciprocal of that scale.
    pub fn from_u16(
        width: usize,
        height: usize,
        pixels: &[u16],
        adu_per_stored_unit: f64,
        config: PixelTrailAlignmentConfig,
    ) -> Result<Self, TrailAlignmentError> {
        if width == 0 || height == 0 {
            return Err(TrailAlignmentError::InvalidDimensions);
        }
        let expected = width
            .checked_mul(height)
            .ok_or(TrailAlignmentError::InvalidDimensions)?;
        if pixels.len() != expected {
            return Err(TrailAlignmentError::PixelCount {
                expected,
                actual: pixels.len(),
            });
        }
        if !adu_per_stored_unit.is_finite() || adu_per_stored_unit <= 0.0 {
            return Err(TrailAlignmentError::InvalidAduScale);
        }
        if !config_is_valid(&config) {
            return Err(TrailAlignmentError::InvalidConfiguration);
        }
        Ok(Self {
            working: WorkingImage::from_u16(
                width,
                height,
                pixels,
                adu_per_stored_unit,
                config.maximum_working_dimension,
            ),
            config,
        })
    }

    /// Search a narrow corridor around a typed, sensor-space predicted path.
    pub fn align_track(&self, clipped_segments: &[PixelSegment]) -> PixelTrailAlignment {
        let working = &self.working;
        let config = &self.config;
        let search_radius_px = config.search_radius_working_px * working.scale_to_sensor;
        let Some(path) = WorkingPath::from_sensor(clipped_segments, working.scale_to_sensor) else {
            return not_evaluated(search_radius_px, PixelTrailNotEvaluatedReason::EmptyPath);
        };
        if path.total_length < config.minimum_working_length_px {
            return not_evaluated(search_radius_px, PixelTrailNotEvaluatedReason::TooShort);
        }

        let noise = working.local_noise_sigma().max(1.0e-6);
        let path_samples = path.samples(
            (path.total_length.ceil() as usize)
                .clamp(config.minimum_samples, config.maximum_samples),
        );
        let mut best: Option<(f64, f64, LineScore)> = None;
        let tilt_radius = COARSE_TILT_SEARCH_RADIUS_WORKING_PX.min(config.search_radius_working_px);
        search_mean_and_tilt_offsets(
            -config.search_radius_working_px,
            config.search_radius_working_px,
            tilt_radius,
            config.coarse_step_px,
            |start_offset, end_offset| {
                if start_offset.abs() > config.search_radius_working_px
                    || end_offset.abs() > config.search_radius_working_px
                {
                    return;
                }
                let Some(score) = working.path_score(
                    &path_samples,
                    start_offset,
                    end_offset,
                    noise,
                    config.minimum_samples,
                    config.minimum_coverage_fraction,
                ) else {
                    return;
                };
                if best
                    .as_ref()
                    .is_none_or(|(_, _, current)| score.objective > current.objective)
                {
                    best = Some((start_offset, end_offset, score));
                }
            },
        );

        let Some((coarse_start, coarse_end, _)) = best else {
            return not_evaluated(
                search_radius_px,
                PixelTrailNotEvaluatedReason::InsufficientCoverage,
            );
        };
        search_offsets_2d(
            coarse_start - config.coarse_step_px,
            coarse_start + config.coarse_step_px,
            coarse_end - config.coarse_step_px,
            coarse_end + config.coarse_step_px,
            config.refine_step_px,
            |start_offset, end_offset| {
                if start_offset.abs() > config.search_radius_working_px
                    || end_offset.abs() > config.search_radius_working_px
                {
                    return;
                }
                let Some(score) = working.path_score(
                    &path_samples,
                    start_offset,
                    end_offset,
                    noise,
                    config.minimum_samples,
                    config.minimum_coverage_fraction,
                ) else {
                    return;
                };
                if best
                    .as_ref()
                    .is_none_or(|(_, _, current)| score.objective > current.objective)
                {
                    best = Some((start_offset, end_offset, score));
                }
            },
        );

        let (start_offset, end_offset, score) = best.expect("coarse search produced a candidate");
        let aligned_segments = path.aligned_segments(
            start_offset,
            end_offset,
            working.scale_to_sensor,
            working.width,
            working.height,
        );
        let detected = score.contrast_sigma >= config.minimum_contrast_sigma
            && score.continuity >= config.minimum_continuity
            && !aligned_segments.is_empty();
        let angle_delta_deg = if detected {
            angle_delta_degrees(&path, &aligned_segments)
        } else {
            0.0
        };

        PixelTrailAlignment {
            status: if detected {
                PixelTrailAlignmentStatus::Detected
            } else {
                PixelTrailAlignmentStatus::NotDetected
            },
            not_evaluated_reason: None,
            aligned_segments: if detected {
                aligned_segments
            } else {
                Vec::new()
            },
            start_normal_offset_px: start_offset * working.scale_to_sensor,
            end_normal_offset_px: end_offset * working.scale_to_sensor,
            mean_normal_offset_px: (start_offset + end_offset) * 0.5 * working.scale_to_sensor,
            angle_delta_deg,
            contrast_adu: score.contrast * working.adu_per_stored_unit,
            contrast_sigma: score.contrast_sigma,
            continuity: score.continuity,
            coverage: score.coverage,
            search_radius_px,
        }
    }
}

fn config_is_valid(config: &PixelTrailAlignmentConfig) -> bool {
    config.maximum_working_dimension >= 2
        && config.search_radius_working_px.is_finite()
        && config.search_radius_working_px > 0.0
        && config.coarse_step_px.is_finite()
        && config.coarse_step_px > 0.0
        && config.refine_step_px.is_finite()
        && config.refine_step_px > 0.0
        && config.minimum_working_length_px.is_finite()
        && config.minimum_working_length_px > 0.0
        && config.minimum_samples >= 2
        && config.maximum_samples >= config.minimum_samples
        && config.minimum_coverage_fraction.is_finite()
        && (0.0..=1.0).contains(&config.minimum_coverage_fraction)
        && config.minimum_contrast_sigma.is_finite()
        && config.minimum_contrast_sigma >= 0.0
        && config.minimum_continuity.is_finite()
        && (0.0..=1.0).contains(&config.minimum_continuity)
}

#[derive(Clone, Copy, Debug)]
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn add_scaled(self, direction: Self, amount: f64) -> Self {
        Self {
            x: self.x + direction.x * amount,
            y: self.y + direction.y * amount,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct WorkingSegment {
    start: Point,
    end: Point,
    normal: Point,
    length: f64,
    distance_start: f64,
}

#[derive(Debug)]
struct WorkingPath {
    segments: Vec<WorkingSegment>,
    total_length: f64,
}

impl WorkingPath {
    fn from_sensor(segments: &[PixelSegment], scale_to_sensor: f64) -> Option<Self> {
        let mut working = Vec::with_capacity(segments.len());
        let mut total_length = 0.0;
        for segment in segments {
            let start = Point {
                x: segment.start.x / scale_to_sensor,
                y: segment.start.y / scale_to_sensor,
            };
            let end = Point {
                x: segment.end.x / scale_to_sensor,
                y: segment.end.y / scale_to_sensor,
            };
            let dx = end.x - start.x;
            let dy = end.y - start.y;
            let length = dx.hypot(dy);
            if !start.x.is_finite()
                || !start.y.is_finite()
                || !end.x.is_finite()
                || !end.y.is_finite()
                || !length.is_finite()
                || length <= f64::EPSILON
            {
                continue;
            }
            working.push(WorkingSegment {
                start,
                end,
                normal: Point {
                    x: -dy / length,
                    y: dx / length,
                },
                length,
                distance_start: total_length,
            });
            total_length += length;
        }
        (!working.is_empty()).then_some(Self {
            segments: working,
            total_length,
        })
    }

    fn samples(&self, count: usize) -> Vec<PathSample> {
        let mut samples = Vec::with_capacity(count);
        let mut segment_index = 0;
        for index in 0..count {
            let progress = (index as f64 + 0.5) / count as f64;
            let distance = progress * self.total_length;
            while segment_index + 1 < self.segments.len()
                && distance
                    > self.segments[segment_index].distance_start
                        + self.segments[segment_index].length
            {
                segment_index += 1;
            }
            let segment = &self.segments[segment_index];
            let local = ((distance - segment.distance_start) / segment.length).clamp(0.0, 1.0);
            samples.push(PathSample {
                point: Point {
                    x: (segment.end.x - segment.start.x).mul_add(local, segment.start.x),
                    y: (segment.end.y - segment.start.y).mul_add(local, segment.start.y),
                },
                normal: segment.normal,
                progress,
            });
        }
        samples
    }

    fn aligned_segments(
        &self,
        path_start_offset: f64,
        path_end_offset: f64,
        scale_to_sensor: f64,
        width: usize,
        height: usize,
    ) -> Vec<PixelSegment> {
        let max_x = width as f64 - 1.001;
        let max_y = height as f64 - 1.001;
        self.segments
            .iter()
            .filter_map(|segment| {
                let start_progress = segment.distance_start / self.total_length;
                let end_progress = (segment.distance_start + segment.length) / self.total_length;
                let start_offset = (path_end_offset - path_start_offset)
                    .mul_add(start_progress, path_start_offset);
                let end_offset =
                    (path_end_offset - path_start_offset).mul_add(end_progress, path_start_offset);
                let start = segment.start.add_scaled(segment.normal, start_offset);
                let end = segment.end.add_scaled(segment.normal, end_offset);
                clip_line(start, end, max_x, max_y).map(|(start, end)| PixelSegment {
                    start: PixelPoint {
                        x: start.x * scale_to_sensor,
                        y: start.y * scale_to_sensor,
                    },
                    end: PixelPoint {
                        x: end.x * scale_to_sensor,
                        y: end.y * scale_to_sensor,
                    },
                })
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
struct PathSample {
    point: Point,
    normal: Point,
    progress: f64,
}

struct WorkingImage {
    width: usize,
    height: usize,
    data: Vec<f64>,
    scale_to_sensor: f64,
    adu_per_stored_unit: f64,
}

impl WorkingImage {
    fn from_u16(
        sensor_width: usize,
        sensor_height: usize,
        pixels: &[u16],
        adu_per_stored_unit: f64,
        maximum_working_dimension: usize,
    ) -> Self {
        let longest = sensor_width.max(sensor_height);
        let factor = longest.div_ceil(maximum_working_dimension).max(1);
        let width = sensor_width.div_ceil(factor);
        let height = sensor_height.div_ceil(factor);
        let mut data = vec![0.0; width * height];
        for working_y in 0..height {
            let y0 = working_y * factor;
            let y1 = (y0 + factor).min(sensor_height);
            for working_x in 0..width {
                let x0 = working_x * factor;
                let x1 = (x0 + factor).min(sensor_width);
                let mut sum = 0_u64;
                let mut count = 0_u64;
                for y in y0..y1 {
                    let row = &pixels[y * sensor_width..(y + 1) * sensor_width];
                    for value in &row[x0..x1] {
                        sum += u64::from(*value);
                        count += 1;
                    }
                }
                data[working_y * width + working_x] = sum as f64 / count.max(1) as f64;
            }
        }
        Self {
            width,
            height,
            data,
            scale_to_sensor: factor as f64,
            adu_per_stored_unit,
        }
    }

    fn sample(&self, x: f64, y: f64) -> Option<f64> {
        if x < 0.0
            || y < 0.0
            || x >= self.width.saturating_sub(1) as f64
            || y >= self.height.saturating_sub(1) as f64
        {
            return None;
        }
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let fx = x - x0 as f64;
        let fy = y - y0 as f64;
        let i00 = self.data[y0 * self.width + x0];
        let i10 = self.data[y0 * self.width + x0 + 1];
        let i01 = self.data[(y0 + 1) * self.width + x0];
        let i11 = self.data[(y0 + 1) * self.width + x0 + 1];
        Some(
            i00 * (1.0 - fx) * (1.0 - fy)
                + i10 * fx * (1.0 - fy)
                + i01 * (1.0 - fx) * fy
                + i11 * fx * fy,
        )
    }

    fn local_noise_sigma(&self) -> f64 {
        let mut differences = Vec::with_capacity((self.width * self.height) / 32);
        for y in (1..self.height.saturating_sub(2)).step_by(6) {
            for x in (1..self.width.saturating_sub(2)).step_by(6) {
                let value = self.data[y * self.width + x];
                differences.push((value - self.data[y * self.width + x + 2]).abs());
                differences.push((value - self.data[(y + 2) * self.width + x]).abs());
            }
        }
        quantile(&mut differences, 0.5) / 0.953_872_552_4
    }

    fn path_score(
        &self,
        path_samples: &[PathSample],
        start_offset: f64,
        end_offset: f64,
        noise: f64,
        minimum_samples: usize,
        minimum_coverage_fraction: f64,
    ) -> Option<LineScore> {
        let mut contrasts = Vec::with_capacity(path_samples.len());
        for sample in path_samples {
            let offset = (end_offset - start_offset).mul_add(sample.progress, start_offset);
            let point = sample.point.add_scaled(sample.normal, offset);
            let Some(center_low) = self.sample_offset(point, sample.normal, -0.45) else {
                continue;
            };
            let Some(center_mid) = self.sample_offset(point, sample.normal, 0.0) else {
                continue;
            };
            let Some(center_high) = self.sample_offset(point, sample.normal, 0.45) else {
                continue;
            };
            let Some(side_low_outer) = self.sample_offset(point, sample.normal, -5.0) else {
                continue;
            };
            let Some(side_low_inner) = self.sample_offset(point, sample.normal, -3.0) else {
                continue;
            };
            let Some(side_high_inner) = self.sample_offset(point, sample.normal, 3.0) else {
                continue;
            };
            let Some(side_high_outer) = self.sample_offset(point, sample.normal, 5.0) else {
                continue;
            };
            let center = (center_low + center_mid + center_high) / 3.0;
            let mut sides = [
                side_low_outer,
                side_low_inner,
                side_high_inner,
                side_high_outer,
            ];
            contrasts.push(center - quantile(&mut sides, 0.5));
        }
        let coverage = contrasts.len() as f64 / path_samples.len() as f64;
        if contrasts.len() < minimum_samples / 2 || coverage < minimum_coverage_fraction {
            return None;
        }
        let continuity = contrasts
            .iter()
            .filter(|contrast| **contrast > noise * 0.75)
            .count() as f64
            / contrasts.len() as f64;
        let contrast = quantile(&mut contrasts, 0.60).max(0.0);
        let contrast_sigma = contrast / noise;
        Some(LineScore {
            contrast,
            contrast_sigma,
            continuity,
            coverage,
            objective: contrast_sigma * (0.5 + 0.5 * continuity),
        })
    }

    fn sample_offset(&self, point: Point, normal: Point, offset: f64) -> Option<f64> {
        self.sample(point.x + normal.x * offset, point.y + normal.y * offset)
    }
}

#[derive(Clone, Copy, Debug)]
struct LineScore {
    contrast: f64,
    contrast_sigma: f64,
    continuity: f64,
    coverage: f64,
    objective: f64,
}

fn not_detected(search_radius_px: f64) -> PixelTrailAlignment {
    PixelTrailAlignment {
        status: PixelTrailAlignmentStatus::NotDetected,
        not_evaluated_reason: None,
        aligned_segments: Vec::new(),
        start_normal_offset_px: 0.0,
        end_normal_offset_px: 0.0,
        mean_normal_offset_px: 0.0,
        angle_delta_deg: 0.0,
        contrast_adu: 0.0,
        contrast_sigma: 0.0,
        continuity: 0.0,
        coverage: 0.0,
        search_radius_px,
    }
}

fn not_evaluated(
    search_radius_px: f64,
    reason: PixelTrailNotEvaluatedReason,
) -> PixelTrailAlignment {
    let mut result = not_detected(search_radius_px);
    result.status = PixelTrailAlignmentStatus::NotEvaluated;
    result.not_evaluated_reason = Some(reason);
    result
}

fn quantile(values: &mut [f64], fraction: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * fraction).round() as usize;
    values[index]
}

fn search_mean_and_tilt_offsets(
    minimum_mean: f64,
    maximum_mean: f64,
    tilt_radius: f64,
    step: f64,
    mut visit: impl FnMut(f64, f64),
) {
    let mut mean = minimum_mean;
    while mean <= maximum_mean + step * 0.25 {
        let mut tilt = -tilt_radius;
        while tilt <= tilt_radius + step * 0.25 {
            visit(mean - tilt, mean + tilt);
            tilt += step;
        }
        mean += step;
    }
}

fn search_offsets_2d(
    start_minimum: f64,
    start_maximum: f64,
    end_minimum: f64,
    end_maximum: f64,
    step: f64,
    mut visit: impl FnMut(f64, f64),
) {
    let mut start = start_minimum;
    while start <= start_maximum + step * 0.25 {
        let mut end = end_minimum;
        while end <= end_maximum + step * 0.25 {
            visit(start, end);
            end += step;
        }
        start += step;
    }
}

/// Clip a line segment to a rectangle using Liang-Barsky parameters.
fn clip_line(start: Point, end: Point, max_x: f64, max_y: f64) -> Option<(Point, Point)> {
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let mut t0: f64 = 0.0;
    let mut t1: f64 = 1.0;
    for (p, q) in [
        (-dx, start.x),
        (dx, max_x - start.x),
        (-dy, start.y),
        (dy, max_y - start.y),
    ] {
        if p.abs() < f64::EPSILON {
            if q < 0.0 {
                return None;
            }
            continue;
        }
        let ratio = q / p;
        if p < 0.0 {
            t0 = t0.max(ratio);
        } else {
            t1 = t1.min(ratio);
        }
        if t0 > t1 {
            return None;
        }
    }
    Some((
        Point {
            x: start.x + t0 * dx,
            y: start.y + t0 * dy,
        },
        Point {
            x: start.x + t1 * dx,
            y: start.y + t1 * dy,
        },
    ))
}

fn angle_delta_degrees(path: &WorkingPath, aligned: &[PixelSegment]) -> f64 {
    let predicted_start = path.segments.first().expect("path is non-empty").start;
    let predicted_end = path.segments.last().expect("path is non-empty").end;
    let aligned_start = aligned.first().expect("alignment is non-empty").start;
    let aligned_end = aligned.last().expect("alignment is non-empty").end;
    let predicted = (predicted_end.y - predicted_start.y)
        .atan2(predicted_end.x - predicted_start.x)
        .to_degrees();
    let aligned = (aligned_end.y - aligned_start.y)
        .atan2(aligned_end.x - aligned_start.x)
        .to_degrees();
    let mut delta = aligned - predicted;
    while delta > 180.0 {
        delta -= 360.0;
    }
    while delta < -180.0 {
        delta += 360.0;
    }
    delta
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(start: (f64, f64), end: (f64, f64)) -> PixelSegment {
        PixelSegment {
            start: PixelPoint {
                x: start.0,
                y: start.1,
            },
            end: PixelPoint { x: end.0, y: end.1 },
        }
    }

    fn shift(segment: PixelSegment, amount: f64) -> PixelSegment {
        let dx = segment.end.x - segment.start.x;
        let dy = segment.end.y - segment.start.y;
        let length = dx.hypot(dy);
        let normal = PixelPoint {
            x: -dy / length,
            y: dx / length,
        };
        PixelSegment {
            start: PixelPoint {
                x: segment.start.x + normal.x * amount,
                y: segment.start.y + normal.y * amount,
            },
            end: PixelPoint {
                x: segment.end.x + normal.x * amount,
                y: segment.end.y + normal.y * amount,
            },
        }
    }

    fn test_image(width: usize, height: usize, trails: &[PixelSegment]) -> Vec<u16> {
        let mut data = vec![0_u16; width * height];
        for y in 0..height {
            for x in 0..width {
                let noise = ((x * 17 + y * 31 + x * y * 3) % 23) as u16;
                data[y * width + x] = 1_000 + noise;
            }
        }
        draw_trails(width, height, &mut data, trails);
        data
    }

    fn flat_test_image(width: usize, height: usize, trails: &[PixelSegment]) -> Vec<u16> {
        let mut data = vec![1_000_u16; width * height];
        draw_trails(width, height, &mut data, trails);
        data
    }

    fn draw_trails(width: usize, height: usize, data: &mut [u16], trails: &[PixelSegment]) {
        for trail in trails {
            let steps = ((trail.end.x - trail.start.x)
                .hypot(trail.end.y - trail.start.y)
                .ceil() as usize)
                * 3;
            for index in 0..=steps {
                let progress = index as f64 / steps.max(1) as f64;
                let x = (trail.end.x - trail.start.x).mul_add(progress, trail.start.x);
                let y = (trail.end.y - trail.start.y).mul_add(progress, trail.start.y);
                for offset_y in -1..=1 {
                    for offset_x in -1..=1 {
                        let pixel_x = x.round() as isize + offset_x;
                        let pixel_y = y.round() as isize + offset_y;
                        if pixel_x >= 0
                            && pixel_y >= 0
                            && (pixel_x as usize) < width
                            && (pixel_y as usize) < height
                        {
                            let index = pixel_y as usize * width + pixel_x as usize;
                            data[index] = data[index].saturating_add(55);
                        }
                    }
                }
            }
        }
    }

    fn aligner(width: usize, height: usize, pixels: &[u16]) -> PixelTrailAligner {
        PixelTrailAligner::from_u16(
            width,
            height,
            pixels,
            1.0,
            PixelTrailAlignmentConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn aligns_a_shifted_faint_trail_without_changing_the_prediction() {
        let predicted = vec![segment((20.0, 110.0), (610.0, 245.0))];
        let actual = vec![shift(predicted[0], 7.0)];
        let pixels = test_image(640, 360, &actual);

        let alignment = aligner(640, 360, &pixels).align_track(&predicted);

        assert!(alignment.detected(), "{alignment:?}");
        assert_eq!(alignment.aligned_segments.len(), 1);
        assert!(alignment.mean_normal_offset_px.abs() > 4.0);
        assert!(
            alignment.coverage >= PixelTrailAlignmentConfig::default().minimum_coverage_fraction
        );
        assert_eq!(predicted, vec![segment((20.0, 110.0), (610.0, 245.0))]);
    }

    #[test]
    fn evaluates_noise_as_not_detected() {
        let predicted = vec![segment((20.0, 110.0), (610.0, 245.0))];
        let pixels = test_image(640, 360, &[]);

        let alignment = aligner(640, 360, &pixels).align_track(&predicted);

        assert!(alignment.evaluated());
        assert_eq!(alignment.status, PixelTrailAlignmentStatus::NotDetected);
        assert!(alignment.aligned_segments.is_empty());
    }

    #[test]
    fn follows_the_full_curved_prediction_instead_of_its_endpoint_chord() {
        let predicted = vec![
            segment((20.0, 60.0), (190.0, 60.0)),
            segment((190.0, 60.0), (320.0, 170.0)),
            segment((320.0, 170.0), (450.0, 280.0)),
            segment((450.0, 280.0), (620.0, 280.0)),
        ];
        let actual = predicted
            .iter()
            .copied()
            .map(|segment| shift(segment, 6.0))
            .collect::<Vec<_>>();
        let pixels = test_image(640, 360, &actual);

        let alignment = aligner(640, 360, &pixels).align_track(&predicted);

        assert!(alignment.detected(), "{alignment:?}");
        assert_eq!(alignment.aligned_segments.len(), predicted.len());
        assert!(alignment.mean_normal_offset_px.abs() > 3.0);
    }

    #[test]
    fn finds_a_large_orbital_offset_and_refines_its_tilt() {
        let predicted = vec![segment((20.0, 80.0), (620.0, 80.0))];
        let actual = vec![segment((20.0, 180.0), (620.0, 210.0))];
        let pixels = flat_test_image(640, 360, &actual);

        let alignment = aligner(640, 360, &pixels).align_track(&predicted);

        assert!(alignment.detected(), "{alignment:?}");
        assert!(
            (alignment.start_normal_offset_px - 100.0).abs() < 5.0,
            "{alignment:?}"
        );
        assert!(
            (alignment.end_normal_offset_px - 130.0).abs() < 5.0,
            "{alignment:?}"
        );
        assert!(
            (alignment.angle_delta_deg - 2.86).abs() < 0.25,
            "{alignment:?}"
        );
    }

    #[test]
    fn preserves_the_full_previous_endpoint_tilt_range() {
        let predicted = vec![segment((20.0, 180.0), (620.0, 180.0))];
        let actual = vec![segment((20.0, 148.0), (620.0, 212.0))];
        let pixels = flat_test_image(640, 360, &actual);

        let alignment = aligner(640, 360, &pixels).align_track(&predicted);

        assert!(alignment.detected(), "{alignment:?}");
        assert!(
            (alignment.start_normal_offset_px + 32.0).abs() < 2.0,
            "{alignment:?}"
        );
        assert!(
            (alignment.end_normal_offset_px - 32.0).abs() < 2.0,
            "{alignment:?}"
        );
    }

    #[test]
    fn short_paths_are_not_misreported_as_negative_evidence() {
        let predicted = vec![segment((20.0, 20.0), (30.0, 20.0))];
        let pixels = test_image(64, 64, &[]);

        let alignment = aligner(64, 64, &pixels).align_track(&predicted);

        assert!(!alignment.evaluated());
        assert_eq!(
            alignment.not_evaluated_reason,
            Some(PixelTrailNotEvaluatedReason::TooShort)
        );
    }

    #[test]
    fn paths_without_enough_sideband_coverage_are_not_evaluated() {
        let predicted = vec![segment((5.0, 4.0), (58.0, 4.0))];
        let pixels = test_image(64, 8, &[]);

        let alignment = aligner(64, 8, &pixels).align_track(&predicted);

        assert!(!alignment.evaluated(), "{alignment:?}");
        assert_eq!(
            alignment.not_evaluated_reason,
            Some(PixelTrailNotEvaluatedReason::InsufficientCoverage)
        );
        assert_eq!(alignment.coverage, 0.0);
    }

    #[test]
    fn a_small_valid_fragment_cannot_stand_in_for_a_long_path() {
        let trail = segment((10.0, 32.0), (49.0, 32.0));
        let pixels = test_image(128, 64, &[trail]);
        let working = WorkingImage::from_u16(128, 64, &pixels, 1.0, 2_048);
        let mut samples = (0..1_160)
            .map(|index| PathSample {
                point: Point {
                    x: -100.0,
                    y: -100.0,
                },
                normal: Point { x: 0.0, y: 1.0 },
                progress: index as f64 / 1_200.0,
            })
            .collect::<Vec<_>>();
        samples.extend((0..40).map(|index| PathSample {
            point: Point {
                x: 10.0 + index as f64,
                y: 32.0,
            },
            normal: Point { x: 0.0, y: 1.0 },
            progress: (1_160 + index) as f64 / 1_200.0,
        }));

        let score = working.path_score(
            &samples,
            0.0,
            0.0,
            working.local_noise_sigma().max(1.0e-6),
            80,
            0.5,
        );

        assert!(score.is_none());
    }
}
