//! PSF Guard's master-flat stability measurement (Apache-2.0, 835961cb).
//! File decoding uses the shared image I/O path; only one source is open at a time.
use std::path::PathBuf;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlatStabilityInput {
    pub path: PathBuf,
    pub exposure_seconds: Option<f64>,
    #[serde(default)]
    pub skipped: bool,
}
const FLAT_FEATURE_DEPTH: f32 = 0.10;
/// The flat-set check's generation, recorded with each judgement. 2: dips
/// only, so stars in sky flats are no longer taken for dew; a judgement the
/// first generation made is made again.
pub const FLAT_STABILITY_VERSION: u32 = 2;
/// A frame disagrees with the master at a feature when its own value departs
/// from the master's by more than this fraction of the local response. Half
/// the feature depth, so a spot that only drifts slowly across the run still
/// shows up in the frames on either side of the median. Photon noise in a
/// flat sits near one percent.
const FLAT_AGREEMENT_DEPTH: f32 = 0.05;
/// Half-width of the box that estimates the smooth response. Wider than any
/// droplet, narrower than a dust donut, which the frames all agree on anyway.
const FLAT_SMOOTH_RADIUS: usize = 16;

/// How the frames of one flat set agree about the features their master
/// kept. A feature every frame shows the same way is a real property of the
/// optical path (a dust shadow, a dead cluster) and corrects the lights. A
/// feature the frames disagree about was changing while they were shot, and
/// the master's copy of it corrects nothing.
#[derive(Clone, Debug, serde::Serialize)]
pub struct FlatSetStability {
    pub algorithm_version: u32,
    /// Input index and diagnostic for unreadable or incompatible sources.
    pub errors: Vec<(usize, String)>,
    /// Master pixels that depart from the smooth response around them.
    pub feature_pixels: usize,
    /// Disagreeing pixels at or above which a frame counts as drifting: a
    /// share of the sensor rather than of the features, so a dusty optical
    /// path with many honest features cannot hide a droplet run.
    pub drift_threshold: usize,
    /// Per input frame, in `frames` order. Unreadable/incompatible inputs
    /// reach the drift threshold; deliberately skipped inputs are zero.
    pub disagreement: Vec<usize>,
}

/// The sample grids a flat master is smooth on: one per colour channel of an
/// RGB image, one per CFA phase of a raw colour frame (each phase has its
/// own response, and a box mean across phases would call every pixel a
/// feature), and one for a mono frame.
struct SamplePlanes {
    width: usize,
    height: usize,
    channels: usize,
    cfa: bool,
}

impl SamplePlanes {
    fn new(image: &crate::LinearImage, bayer: Option<crate::BayerLayout>) -> Self {
        let channels = image.channels.max(1);
        Self {
            width: image.width,
            height: image.height,
            channels,
            cfa: channels == 1 && bayer.is_some(),
        }
    }

    fn count(&self) -> usize {
        if self.cfa { 4 } else { self.channels }
    }

    /// Which plane a sample index belongs to.
    fn plane_of(&self, index: usize) -> usize {
        if self.cfa {
            let (x, y) = (index % self.width, index / self.width);
            (y & 1) * 2 + (x & 1)
        } else {
            index % self.channels
        }
    }

    /// The plane's own grid size.
    fn size(&self, plane: usize) -> (usize, usize) {
        if self.cfa {
            let (ox, oy) = (plane & 1, plane >> 1);
            (
                (self.width + 1 - ox.min(self.width)) / 2,
                (self.height + 1 - oy.min(self.height)) / 2,
            )
        } else {
            (self.width, self.height)
        }
    }

    /// Sample index of a plane-grid position.
    fn index(&self, plane: usize, x: usize, y: usize) -> usize {
        if self.cfa {
            let (ox, oy) = (plane & 1, plane >> 1);
            (2 * y + oy) * self.width + 2 * x + ox
        } else {
            (y * self.width + x) * self.channels + plane
        }
    }

    /// The smooth response under every sample: a box mean on each plane.
    fn smooth(&self, data: &[f32]) -> Vec<f32> {
        let mut smooth = vec![0f32; data.len()];
        let radius = if self.cfa {
            FLAT_SMOOTH_RADIUS / 2
        } else {
            FLAT_SMOOTH_RADIUS
        };
        for plane in 0..self.count() {
            let (width, height) = self.size(plane);
            let values: Vec<f32> = (0..height)
                .flat_map(|y| (0..width).map(move |x| (x, y)))
                .map(|(x, y)| data[self.index(plane, x, y)])
                .collect();
            let blurred = box_mean_plane(&values, width, height, radius);
            for y in 0..height {
                for x in 0..width {
                    smooth[self.index(plane, x, y)] = blurred[y * width + x];
                }
            }
        }
        smooth
    }
}

/// Measure PSF Guard's dip-only feature agreement on mono, RGB or CFA planes.
/// The master is already integrated/calibrated. Optional masters calibrate each
/// source using its supplied exposure before comparison. `normalized_full_scale`
/// rescales explicitly declared unit-bound files (65535 preserves PSF Guard ADU).
/// Missing/incompatible files count as drifting and include input diagnostics.
/// Sources are decoded through shared image I/O, one at a time.
pub fn measure_flat_stability(
    frames: &[FlatStabilityInput],
    master: &crate::LinearImage,
    bayer: Option<crate::BayerLayout>,
    calibration: Option<&crate::CalibrationMasters>,
    normalized_full_scale: Option<f32>,
) -> FlatSetStability {
    let planes = SamplePlanes::new(master, bayer);
    let pixels = master.width * master.height;
    let smooth = planes.smooth(&master.data);
    let features: Vec<usize> = (0..master.data.len())
        .filter(|&index| {
            let value = master.data[index];
            let surround = smooth[index];
            value.is_finite() && surround > 0.0 && surround - value > FLAT_FEATURE_DEPTH * surround
        })
        .collect();
    // Below this many feature pixels there is nothing to judge a set by: a
    // handful of dead pixels, or a clean master. A frame drifts once it
    // disagrees at a fifth of the features, capped at one pixel in two
    // hundred thousand of the sensor so a dusty optical path with many
    // honest features cannot hide a droplet run, and never below sixteen so
    // noise cannot reach it.
    let minimum = (pixels / 100_000).max(16);
    let drift_threshold = (pixels / 200_000).min(features.len() / 5).max(16);
    if features.len() < minimum {
        return FlatSetStability {
            algorithm_version: FLAT_STABILITY_VERSION,
            errors: Vec::new(),
            feature_pixels: features.len(),
            drift_threshold,
            disagreement: vec![0; frames.len()],
        };
    }
    // Each frame is put on the master's scale by the median ratio of its
    // samples to the master's, per plane, so a plane's own response does
    // not read as disagreement.
    let step = (master.data.len() / 200_000).max(1);
    // A frame the integrator left out was never in the master and counts
    // as agreeing; a frame the check cannot read back or calibrate is not
    // trusted and counts as drifting.
    let mut errors = Vec::new();
    let disagreement = frames
        .iter()
        .enumerate()
        .map(|(index, frame)| {
            if frame.skipped {
                return 0;
            }
            let mut opened = match crate::FitsFrame::open(&frame.path) {
                Ok(opened) => opened,
                Err(error) => {
                    errors.push((index, error.to_string()));
                    return drift_threshold;
                }
            };
            if let Some(scale) = normalized_full_scale {
                opened.rescale_declared_unit_bounds(scale);
            }
            if opened.image.width != master.width
                || opened.image.height != master.height
                || opened.image.channels.max(1) != planes.channels
            {
                errors.push((index, "flat input dimensions differ from the master".into()));
                return drift_threshold;
            }
            // The integrator scaled any dark by the catalog's exposure; do
            // the same rather than trusting a header the catalog may not.
            let exposure = frame.exposure_seconds.or(opened.exposure_seconds);
            if let Some(calibration) = calibration
                && let Err(error) = calibration.apply(&mut opened.image, exposure, opened.bayer)
            {
                errors.push((index, error.to_string()));
                return drift_threshold;
            }
            let data = &opened.image.data;
            let mut ratios: Vec<Vec<f32>> = vec![Vec::new(); planes.count()];
            for index in (0..data.len()).step_by(step) {
                let (value, reference) = (data[index], master.data[index]);
                if reference > 0.05 && reference.is_finite() && value.is_finite() {
                    ratios[planes.plane_of(index)].push(value / reference);
                }
            }
            let gains: Vec<f32> = ratios.into_iter().map(median_f32).collect();
            features
                .iter()
                .filter(|&&index| {
                    let gain = gains[planes.plane_of(index)];
                    gain.is_finite()
                        && gain > 0.0
                        && (data[index] / gain - master.data[index]).abs()
                            > FLAT_AGREEMENT_DEPTH * smooth[index]
                })
                .count()
        })
        .collect();
    FlatSetStability {
        algorithm_version: FLAT_STABILITY_VERSION,
        errors,
        feature_pixels: features.len(),
        drift_threshold,
        disagreement,
    }
}

fn median_f32(mut values: Vec<f32>) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    let middle = values.len() / 2;
    let (_, median, _) = values.select_nth_unstable_by(middle, |a, b| a.total_cmp(b));
    *median
}

/// Mean over a (2r+1)-square box clamped at the edges, in two separable
/// passes. Non-finite samples count as zero.
fn box_mean_plane(plane: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let mut rows = vec![0f32; plane.len()];
    let mut prefix = vec![0f64; width.max(height) + 1];
    for y in 0..height {
        let row = &plane[y * width..(y + 1) * width];
        for x in 0..width {
            let value = row[x];
            prefix[x + 1] = prefix[x]
                + if value.is_finite() {
                    f64::from(value)
                } else {
                    0.0
                };
        }
        for x in 0..width {
            let low = x.saturating_sub(radius);
            let high = (x + radius + 1).min(width);
            rows[y * width + x] = ((prefix[high] - prefix[low]) / (high - low) as f64) as f32;
        }
    }
    let mut out = vec![0f32; plane.len()];
    for x in 0..width {
        for y in 0..height {
            prefix[y + 1] = prefix[y] + f64::from(rows[y * width + x]);
        }
        for y in 0..height {
            let low = y.saturating_sub(radius);
            let high = (y + radius + 1).min(height);
            out[y * width + x] = ((prefix[high] - prefix[low]) / (high - low) as f64) as f32;
        }
    }
    out
}

/// File form of the loaded-master measurement, shared by C and Python.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlatStabilityFileRequest {
    pub master: PathBuf,
    pub inputs: Vec<FlatStabilityInput>,
    pub bias: Option<PathBuf>,
    pub dark: Option<PathBuf>,
    pub dark_exposure_seconds: Option<f64>,
    pub normalized_full_scale: Option<f32>,
}
/// Load the master and optional calibration, then run the same bounded check.
pub fn measure_flat_stability_files(
    request: FlatStabilityFileRequest,
) -> crate::Result<FlatSetStability> {
    if request
        .normalized_full_scale
        .is_some_and(|s| !s.is_finite() || s <= 0.0)
    {
        return Err(crate::Error::Calibration(
            "normalized_full_scale must be finite and positive".into(),
        ));
    }
    let open = |path: &std::path::Path| -> crate::Result<crate::FitsFrame> {
        let mut frame = crate::FitsFrame::open(path)?;
        if let Some(scale) = request.normalized_full_scale {
            frame.rescale_declared_unit_bounds(scale);
        }
        Ok(frame)
    };
    let master = open(&request.master)?;
    let calibration = crate::CalibrationMasters::from_fits_frames(
        request.bias.as_deref().map(open).transpose()?,
        request.dark.as_deref().map(open).transpose()?,
        None,
        request.dark_exposure_seconds,
    )?;
    Ok(measure_flat_stability(
        &request.inputs,
        &master.image,
        master.bayer,
        Some(&calibration),
        request.normalized_full_scale,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn coherent_spots_agree_transient_spots_and_unreadable_inputs_drift() {
        let dir = tempfile::tempdir().unwrap();
        let mut master = crate::LinearImage::new(128, 128, 1, vec![1.0; 128 * 128]).unwrap();
        for y in 56..72 {
            for x in 56..72 {
                master.data[y * 128 + x] = 0.5;
            }
        }
        let agreeing = dir.path().join("agree.fits");
        let clean = dir.path().join("different.fits");
        crate::write_linear_image_fits_f32(&agreeing, &master, &[], &[]).unwrap();
        let no_spot = crate::LinearImage::new(128, 128, 1, vec![1.0; 128 * 128]).unwrap();
        crate::write_linear_image_fits_f32(&clean, &no_spot, &[], &[]).unwrap();
        let missing = dir.path().join("missing.fits");
        let input = |path, skipped| FlatStabilityInput {
            path,
            skipped,
            exposure_seconds: None,
        };
        let inputs = [
            input(agreeing, false),
            input(clean, false),
            input(missing.clone(), false),
            input(missing, true),
        ];
        let result = measure_flat_stability(&inputs, &master, None, None, None);
        assert_eq!(result.algorithm_version, 2);
        assert!(result.feature_pixels >= 16);
        assert_eq!(result.disagreement[0], 0);
        assert!(result.disagreement[1] >= result.drift_threshold);
        assert_eq!(result.disagreement[2], result.drift_threshold);
        assert_eq!(result.disagreement[3], 0);
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].0, 2);
    }
}
