//! Choosing the reference frame of a batch.
//!
//! The reference fixes the stack's grid and, under local background
//! normalization, the background every frame is matched to. The first frame
//! of a night is often a poor choice: low in the sky, in twilight, its
//! gradient and haze end up in the stack. A quick look at every frame finds
//! a clear, sharp one instead.

use crate::{FitsFrame, LinearImage, Result};
use rayon::prelude::*;
use seiza::{DetectBackend, DetectConfig};
use std::path::{Path, PathBuf};

/// A quick measure of how good a frame would be as the reference.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceScore {
    /// Stars detected at five sigma on the half-resolution luminance.
    pub stars: usize,
    /// Median detected star area in half-resolution pixels; smaller is
    /// sharper.
    pub median_star_area: f32,
    /// Median sky level of the half-resolution luminance.
    pub background: f32,
    /// Spread of the luminance's 64-pixel block medians (scaled MAD): how
    /// far the sky departs from flat. Cloud and twilight gradients raise it.
    pub background_variation: f32,
    /// Median over the brightest unsaturated stars of sqrt(flux x peak)
    /// over the sky noise. Poor seeing, trailing, haze and twilight all
    /// lower it.
    pub score: f32,
}

/// Stars the scoring detection keeps.
const SCORING_STARS: usize = 2_000;
/// Brightest unsaturated stars whose signal sets the score.
const SCORED_STARS: usize = 200;

/// Robust sigma (scaled MAD) of every 97th finite sample.
fn sampled_sigma(values: &[f32]) -> Option<f32> {
    let mut sample = values
        .iter()
        .step_by(97)
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let center = seiza_stats::median_in_place(&mut sample.clone())?;
    for value in &mut sample {
        *value = (*value - center).abs();
    }
    Some(1.4826 * seiza_stats::median_in_place(&mut sample)?).filter(|sigma| *sigma > 0.0)
}

/// Score one frame on a half-resolution luminance (each 2x2 block summed,
/// which for a Bayer frame sums one of each color site); see
/// [`ReferenceScore::score`].
pub fn reference_score(frame: &FitsFrame) -> Option<ReferenceScore> {
    let luma = half_resolution_luminance(&frame.image)?;
    let (width, height) = (frame.image.width / 2, frame.image.height / 2);
    let mut normalized = luma.clone();
    let (minimum, maximum) = normalized
        .iter()
        .filter(|value| value.is_finite())
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), &value| {
            (low.min(value), high.max(value))
        });
    // Also refuses NaN bounds, which an all-NaN frame would give.
    if maximum.partial_cmp(&minimum) != Some(std::cmp::Ordering::Greater) {
        return None;
    }
    for value in &mut normalized {
        *value = if value.is_finite() {
            (*value - minimum) / (maximum - minimum)
        } else {
            0.0
        };
    }
    let config = DetectConfig {
        backend: DetectBackend::F32,
        sigma: 5.0,
        max_stars: SCORING_STARS,
        ..DetectConfig::default()
    };
    let stars = seiza::detect_stars_luma_f32(&normalized, width as u32, height as u32, &config);
    let mut areas = stars
        .iter()
        .map(|star| star.area as f32)
        .collect::<Vec<_>>();
    let median_star_area = seiza_stats::median_in_place(&mut areas)?;
    let mut sky = luma
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    let background = seiza_stats::median_in_place(&mut sky)?;
    let noise = sampled_sigma(&normalized)?;
    // Each bright star's sqrt(flux x peak) over the sky noise. Flux over
    // peak grows with a star's area, so this falls with poor seeing and with
    // trails (which the detector breaks into many short pieces), as well as
    // with haze, which dims stars, and twilight, which raises the noise.
    let mut signal = stars
        .iter()
        .filter(|star| star.peak < 0.9 && star.flux > 0.0 && star.peak > 0.0)
        .take(SCORED_STARS)
        .map(|star| ((star.flux * f64::from(star.peak)).sqrt() / f64::from(noise)) as f32)
        .collect::<Vec<_>>();
    let score = seiza_stats::median_in_place(&mut signal)?;
    let background_variation = block_variation(&luma, width, height, 64)?;
    Some(ReferenceScore {
        stars: stars.len(),
        median_star_area,
        background,
        background_variation,
        score,
    })
}

/// Scaled MAD of the medians of `block`-pixel blocks.
fn block_variation(luma: &[f32], width: usize, height: usize, block: usize) -> Option<f32> {
    let (columns, rows) = (width / block, height / block);
    let mut medians = (0..columns * rows)
        .into_par_iter()
        .filter_map(|cell| {
            let (bx, by) = ((cell % columns) * block, (cell / columns) * block);
            let mut values = (by..by + block)
                .flat_map(|y| luma[y * width + bx..y * width + bx + block].iter().copied())
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            seiza_stats::median_in_place(&mut values)
        })
        .collect::<Vec<_>>();
    let center = seiza_stats::median_in_place(&mut medians.clone())?;
    let mut deviations = medians
        .iter_mut()
        .map(|value| (*value - center).abs())
        .collect::<Vec<_>>();
    Some(1.4826 * seiza_stats::median_in_place(&mut deviations)?)
}

/// Among frames whose star score is at least this share of the best, the
/// one with the flattest background becomes the reference.
const CANDIDATE_SHARE: f32 = 0.7;

fn half_resolution_luminance(image: &LinearImage) -> Option<Vec<f32>> {
    let (width, height) = (image.width / 2, image.height / 2);
    if width == 0 || height == 0 {
        return None;
    }
    let channels = image.channels;
    let sample = |x: usize, y: usize| {
        let base = (y * image.width + x) * channels;
        image.data[base..base + channels].iter().sum::<f32>()
    };
    Some(
        (0..width * height)
            .into_par_iter()
            .map(|index| {
                let (x, y) = ((index % width) * 2, (index / width) * 2);
                sample(x, y) + sample(x + 1, y) + sample(x, y + 1) + sample(x + 1, y + 1)
            })
            .collect(),
    )
}

/// Open and score every path, a few at a time, and return the index of the
/// chosen reference (see [`CANDIDATE_SHARE`]) with every score in path order (`None` for a frame
/// that could not be read or scored). Errors only when no frame scores.
pub fn choose_reference(
    paths: &[PathBuf],
    concurrency: usize,
) -> Result<(usize, Vec<Option<ReferenceScore>>)> {
    let score = |path: &Path| {
        FitsFrame::open(path)
            .ok()
            .and_then(|frame| reference_score(&frame))
    };
    let mut scores = Vec::with_capacity(paths.len());
    for chunk in paths.chunks(concurrency.max(1)) {
        scores.extend(chunk.par_iter().map(|path| score(path)).collect::<Vec<_>>());
    }
    // Stars alone can pick a frame half under cloud whose clear half is
    // excellent, and local background normalization would then copy that
    // cloud into every frame: among frames with nearly the best stars, take
    // the flattest sky.
    let top = scores
        .iter()
        .flatten()
        .map(|score| score.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let best = scores
        .iter()
        .enumerate()
        .filter_map(|(index, score)| score.as_ref().map(|score| (index, score)))
        .filter(|(_, score)| score.score >= CANDIDATE_SHARE * top)
        .min_by(|left, right| {
            left.1
                .background_variation
                .total_cmp(&right.1.background_variation)
        })
        .map(|(index, _)| index)
        .ok_or_else(|| crate::Error::Stack("no frame could be scored as a reference".into()))?;
    Ok((best, scores))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `image` with uniform noise of the given half-range added.
    fn noisy(image: &LinearImage, amplitude: f32, seed: u32) -> LinearImage {
        let mut state = seed | 1;
        LinearImage::new(
            image.width,
            image.height,
            image.channels,
            image
                .data
                .iter()
                .map(|value| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    value + ((state % 2001) as f32 / 1000.0 - 1.0) * amplitude
                })
                .collect(),
        )
        .unwrap()
    }

    fn frame(image: LinearImage) -> FitsFrame {
        FitsFrame {
            image,
            headers: Vec::new(),
            exposure_seconds: None,
            bayer: None,
            source: None,
            bounds: None,
        }
    }

    /// The test star field with each sample replaced by the mean of a
    /// horizontal run of `length` samples: a trail or a smear.
    fn smeared(length: usize) -> LinearImage {
        let field = crate::registration::test_star_field(false);
        let width = field.width;
        let data = (0..field.data.len())
            .map(|index| {
                let x = index % width;
                let start = x.saturating_sub(length / 2);
                let end = (start + length).min(width);
                let row = index - x;
                field.data[row + start..row + end].iter().sum::<f32>() / (end - start) as f32
            })
            .collect();
        LinearImage::new(width, field.height, 1, data).unwrap()
    }

    #[test]
    fn sharp_clear_frames_outscore_trailed_noisy_or_hazy_ones() {
        let field = crate::registration::test_star_field(false);
        let score = |image: &LinearImage| reference_score(&frame(noisy(image, 8.0, 7))).unwrap();
        let sharp = score(&field);
        let trailed = score(&smeared(9));
        let hazy = score(
            &LinearImage::new(
                field.width,
                field.height,
                1,
                field
                    .data
                    .iter()
                    .map(|value| 100.0 + (value - 100.0) * 0.4)
                    .collect(),
            )
            .unwrap(),
        );
        let twilight = reference_score(&frame(noisy(&field, 40.0, 7))).unwrap();
        for (name, other) in [
            ("trailed", &trailed),
            ("hazy", &hazy),
            ("twilight", &twilight),
        ] {
            assert!(
                sharp.score > 1.3 * other.score,
                "{name}: {sharp:?} against {other:?}"
            );
        }
    }

    #[test]
    fn the_flattest_of_nearly_equal_frames_is_chosen() {
        let directory = tempfile::tempdir().unwrap();
        let field = noisy(&crate::registration::test_star_field(false), 8.0, 11);
        let tilted = LinearImage::new(
            field.width,
            field.height,
            1,
            field
                .data
                .iter()
                .enumerate()
                .map(|(index, value)| value + (index % field.width) as f32 * 2.0)
                .collect(),
        )
        .unwrap();
        let paths = [("tilted.fits", &tilted), ("flat.fits", &field)]
            .into_iter()
            .map(|(name, image)| {
                let path = directory.path().join(name);
                crate::write_processed_image_fits_f32(&path, image, &[], &[]).unwrap();
                path
            })
            .collect::<Vec<_>>();
        let (best, scores) = choose_reference(&paths, 2).unwrap();
        assert_eq!(best, 1, "{scores:?}");
    }
}
