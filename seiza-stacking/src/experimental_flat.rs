//! Opt-in experimental positive spatial replacement for flat inputs.
use crate::{BayerLayout, CancelSignal, Error, LinearImage, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Experimental preprocessing of individual calibrated flats before masking
/// and normalization. May erase real photoresponse; disabled by default on
/// master builds. Negative impulses are intentionally left alone.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExperimentalFlatPreprocessingOptions {
    /// Positive local MAD sigma threshold (default 10).
    pub sigma: f32,
    /// Also require this fractional excess over the absolute local median.
    pub minimum_relative_excess: f32,
    /// Minimum local sigma in calibrated ADU; must be positive.
    pub noise_floor_adu: f32,
}

impl Default for ExperimentalFlatPreprocessingOptions {
    fn default() -> Self {
        Self {
            sigma: 10.0,
            minimum_relative_excess: 0.01,
            noise_floor_adu: 1.0,
        }
    }
}

impl ExperimentalFlatPreprocessingOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        if !self.sigma.is_finite()
            || self.sigma <= 0.0
            || !self.minimum_relative_excess.is_finite()
            || self.minimum_relative_excess < 0.0
            || !self.noise_floor_adu.is_finite()
            || self.noise_floor_adu <= 0.0
        {
            return Err(Error::Calibration("experimental flat preprocessing requires positive finite sigma/noise floor and nonnegative finite relative excess".into()));
        }
        Ok(())
    }
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_unstable_by(f32::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) * 0.5
    }
}

// Conservative positive-only 5x5 prototype, with a raw-ADU noise floor and
// minimum relative excess. Every decision reads unchanged same-plane data.
/// Experimental positive-only 5×5 replacement on calibrated, unnormalized flats.
///
/// Reads unchanged same-plane samples, including CFA phase separation. A mask,
/// when supplied, has one byte per coordinate (any changed channel marks 1).
/// Failure/cancellation leaves the image and mask unchanged. Both bindings
/// reach this through the experimental master-build option.
pub fn preprocess_flat_spatial(
    image: &mut LinearImage,
    bayer: Option<BayerLayout>,
    options: &ExperimentalFlatPreprocessingOptions,
    mask: Option<&mut [u8]>,
    cancel: Option<&CancelSignal>,
) -> Result<u64> {
    options.validate()?;
    if bayer.is_some() && image.channels != 1 {
        return Err(Error::Calibration(
            "experimental CFA preprocessing requires one channel".into(),
        ));
    }
    if mask
        .as_ref()
        .is_some_and(|m| m.len() != image.width * image.height)
    {
        return Err(Error::Calibration(
            "experimental preprocessing mask dimensions differ".into(),
        ));
    }
    if cancel.is_some_and(CancelSignal::is_cancelled) {
        return Err(Error::Cancelled);
    }
    let sigma = options.sigma;
    let relative = options.minimum_relative_excess;
    let floor = options.noise_floor_adu;
    let step = if bayer.is_some() { 2 } else { 1 };
    let source = &image.data;
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    let mut output = source.clone();
    let replaced = output
        .par_chunks_mut(width * channels)
        .enumerate()
        .map(|(y, row)| -> Result<u64> {
            if cancel.is_some_and(CancelSignal::is_cancelled) {
                return Err(Error::Cancelled);
            }
            let mut replaced = 0;
            for x in 0..width {
                for c in 0..channels {
                    let value = source[(y * width + x) * channels + c];
                    if !value.is_finite() {
                        continue;
                    }
                    let mut neighbors = [0.0; 24];
                    let mut n = 0;
                    for dy in -2..=2isize {
                        for dx in -2..=2isize {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let Some(sx) = x.checked_add_signed(dx * step).filter(|&v| v < width)
                            else {
                                continue;
                            };
                            let Some(sy) = y.checked_add_signed(dy * step).filter(|&v| v < height)
                            else {
                                continue;
                            };
                            let neighbor = source[(sy * width + sx) * channels + c];
                            if neighbor.is_finite() {
                                neighbors[n] = neighbor;
                                n += 1;
                            }
                        }
                    }
                    if n < 12 {
                        continue;
                    }
                    let center = median(&mut neighbors[..n]);
                    for v in &mut neighbors[..n] {
                        *v = (*v - center).abs();
                    }
                    let scale = (1.4826 * median(&mut neighbors[..n])).max(floor);
                    let excess = value - center;
                    if excess > sigma * scale && excess > relative * center.abs() {
                        row[x * channels + c] = center;
                        replaced += 1;
                    }
                }
            }
            Ok(replaced)
        })
        .try_reduce(|| 0, |a, b| Ok(a + b))?;
    if cancel.is_some_and(CancelSignal::is_cancelled) {
        return Err(Error::Cancelled);
    }
    if let Some(mask) = mask {
        for ((before, after), marked) in source
            .chunks(channels)
            .zip(output.chunks(channels))
            .zip(mask)
        {
            *marked = u8::from(
                before
                    .iter()
                    .zip(after)
                    .any(|(a, b)| a.to_bits() != b.to_bits()),
            );
        }
    }
    image.data = output;
    Ok(replaced)
}
