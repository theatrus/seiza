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
    /// Median over the same stars of flux over peak, in half-resolution
    /// pixels: the area a star's light is spread over. Poor seeing, focus,
    /// trailing and elongation all raise it.
    pub psf_area: f32,
    /// Eccentricity of the same stars' mean shape, 0 for round: each star's
    /// adaptive second moments give its ellipticity as a vector, at twice
    /// its orientation, and the stars' mean vector is a stretch they share.
    /// Guiding, wind and trailing stretch every star alike and show here;
    /// noise and a neighbour's light point each star a different way and
    /// cancel, as a pattern that varies across the field, like tilt's,
    /// largely does. Measured on a mono frame's own samples and on the
    /// half-resolution luminance otherwise, so a Bayer pattern cannot pass
    /// for elongation. `None` when fewer than [`SHAPED_STARS_MINIMUM`] stars
    /// could be measured.
    pub eccentricity: Option<f32>,
}

/// Fewest stars whose shapes give a frame's [`ReferenceScore::eccentricity`].
pub const SHAPED_STARS_MINIMUM: usize = 10;

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
    let scored = stars
        .iter()
        .filter(|star| star.peak < 0.9 && star.flux > 0.0 && star.peak > 0.0)
        .take(SCORED_STARS)
        .collect::<Vec<_>>();
    let mut signal = scored
        .iter()
        .map(|star| ((star.flux * f64::from(star.peak)).sqrt() / f64::from(noise)) as f32)
        .collect::<Vec<_>>();
    let score = seiza_stats::median_in_place(&mut signal)?;
    let mut spread = scored
        .iter()
        .map(|star| (star.flux / f64::from(star.peak)) as f32)
        .collect::<Vec<_>>();
    let psf_area = seiza_stats::median_in_place(&mut spread)?;
    let background_variation = block_variation(&luma, width, height, 64)?;
    Some(ReferenceScore {
        stars: stars.len(),
        median_star_area,
        background,
        background_variation,
        score,
        psf_area,
        eccentricity: mean_shape_eccentricity(frame, &normalized, width, height, &scored),
    })
}

/// The eccentricity of `stars`' mean shape, found on the half-resolution
/// grid `normalized` is: measured on a mono frame's own samples, where a
/// star's shape is twice as finely sampled, and on `normalized` otherwise.
fn mean_shape_eccentricity(
    frame: &FitsFrame,
    normalized: &[f32],
    width: usize,
    height: usize,
    stars: &[&seiza::DetectedStar],
) -> Option<f32> {
    let full = frame.bayer.is_none() && frame.image.channels == 1;
    let (samples, grid_width, grid_height, scale) = if full {
        (
            frame.image.data.as_slice(),
            frame.image.width,
            frame.image.height,
            2.0,
        )
    } else {
        (normalized, width, height, 1.0)
    };
    let mut sky = samples
        .iter()
        .step_by(97)
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let sky = seiza_stats::median_in_place(&mut sky)?;
    let shapes = stars
        .par_iter()
        .filter_map(|star| {
            // A half-resolution pixel's middle is two full pixels' corner.
            let (x, y) = if full {
                (star.x * scale + 0.5, star.y * scale + 0.5)
            } else {
                (star.x, star.y)
            };
            star_shape(samples, grid_width, grid_height, x, y, sky)
        })
        .collect::<Vec<_>>();
    if shapes.len() < SHAPED_STARS_MINIMUM {
        return None;
    }
    let count = shapes.len() as f64;
    let (one, two) = shapes.iter().fold((0.0_f64, 0.0_f64), |(one, two), shape| {
        (one + shape.0, two + shape.1)
    });
    Some(eccentricity_of(one.hypot(two) / count) as f32)
}

/// The eccentricity, `sqrt(1 − b²/a²)`, of a shape whose ellipticity is
/// `chi`, `(a² − b²) / (a² + b²)`.
fn eccentricity_of(chi: f64) -> f64 {
    let chi = chi.clamp(0.0, 1.0 - 1e-12);
    (2.0 * chi / (1.0 + chi)).sqrt()
}

/// The ellipticity of the star near `(x, y)` as a vector at twice its
/// orientation, `((Ixx − Iyy), 2 Ixy) / (Ixx + Iyy)`, from adaptive
/// (matched Gaussian) second moments above `sky`, or `None` when they do
/// not settle on a finite shape inside a 15-pixel reach.
fn star_shape(
    samples: &[f32],
    width: usize,
    height: usize,
    x: f64,
    y: f64,
    sky: f32,
) -> Option<(f64, f64)> {
    let (mut cx, mut cy) = (x, y);
    let mut weight = [2.25_f64, 0.0, 2.25];
    for _ in 0..30 {
        let determinant = weight[0] * weight[2] - weight[1] * weight[1];
        if determinant.is_nan() || determinant <= 0.0 {
            return None;
        }
        let (ixx, ixy, iyy) = (
            weight[2] / determinant,
            -weight[1] / determinant,
            weight[0] / determinant,
        );
        let reach = (3.0 * weight[0].max(weight[2]).sqrt())
            .ceil()
            .clamp(3.0, 15.0) as isize;
        let (px, py) = (cx.round() as isize, cy.round() as isize);
        if px - reach < 0
            || py - reach < 0
            || px + reach >= width as isize
            || py + reach >= height as isize
        {
            return None;
        }
        let (mut sum, mut sx, mut sy, mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for row in py - reach..=py + reach {
            for column in px - reach..=px + reach {
                let value = samples[row as usize * width + column as usize];
                if !value.is_finite() {
                    return None;
                }
                let (dx, dy) = (column as f64 - cx, row as f64 - cy);
                let gaussian = (-0.5 * (ixx * dx * dx + 2.0 * ixy * dx * dy + iyy * dy * dy)).exp();
                let sample = f64::from(value - sky) * gaussian;
                sum += sample;
                sx += sample * dx;
                sy += sample * dy;
                sxx += sample * dx * dx;
                sxy += sample * dx * dy;
                syy += sample * dy * dy;
            }
        }
        if sum.is_nan() || sum <= 0.0 {
            return None;
        }
        let (mx, my) = (sx / sum, sy / sum);
        cx += mx;
        cy += my;
        if (cx - x).abs() > 3.0 || (cy - y).abs() > 3.0 {
            return None;
        }
        // A matched Gaussian weight halves the covariance it measures.
        let next = [
            2.0 * (sxx / sum - mx * mx),
            2.0 * (sxy / sum - mx * my),
            2.0 * (syy / sum - my * my),
        ];
        let change = (next[0] - weight[0]).abs() + (next[2] - weight[2]).abs();
        weight = [
            0.5 * (weight[0] + next[0]),
            0.5 * (weight[1] + next[1]),
            0.5 * (weight[2] + next[2]),
        ];
        if weight[0] <= 0.01 || weight[2] <= 0.01 || weight[0].max(weight[2]) > 100.0 {
            return None;
        }
        if change < 1e-4 * (weight[0] + weight[2]) {
            break;
        }
    }
    let trace = weight[0] + weight[2];
    let shape = ((weight[0] - weight[2]) / trace, 2.0 * weight[1] / trace);
    (trace.is_finite() && trace > 0.0 && shape.0.hypot(shape.1) < 1.0).then_some(shape)
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
/// chosen reference (see [`rank_references`]) with every score in path order (`None` for a frame
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
    let best = rank_references(&scores)
        .first()
        .copied()
        .ok_or_else(|| crate::Error::Stack("no frame could be scored as a reference".into()))?;
    Ok((best, scores))
}

/// The indices of the scored frames, best reference first: the frames
/// within [`CANDIDATE_SHARE`] of the best score, flattest sky first, then
/// the rest by falling score. Frames with no score are left out.
pub fn rank_references(scores: &[Option<ReferenceScore>]) -> Vec<usize> {
    // Stars alone can pick a frame half under cloud whose clear half is
    // excellent, and local background normalization would then copy that
    // cloud into every frame: among frames with nearly the best stars, take
    // the flattest sky.
    let top = scores
        .iter()
        .flatten()
        .map(|score| score.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let (mut near, mut rest): (Vec<_>, Vec<_>) = scores
        .iter()
        .enumerate()
        .filter_map(|(index, score)| score.as_ref().map(|score| (index, score)))
        .partition(|(_, score)| score.score >= CANDIDATE_SHARE * top);
    near.sort_by(|left, right| {
        left.1
            .background_variation
            .total_cmp(&right.1.background_variation)
    });
    rest.sort_by(|left, right| right.1.score.total_cmp(&left.1.score));
    near.into_iter()
        .chain(rest)
        .map(|(index, _)| index)
        .collect()
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
    fn trailed_stars_measure_elongated_and_spread_out() {
        let field = crate::registration::test_star_field(false);
        let score = |image: &LinearImage| reference_score(&frame(noisy(image, 2.0, 5))).unwrap();
        let round = score(&field);
        let trailed = score(&smeared(7));
        let round_eccentricity = round.eccentricity.unwrap();
        let trailed_eccentricity = trailed.eccentricity.unwrap();
        assert!(round_eccentricity < 0.35, "{round:?}");
        assert!(
            trailed_eccentricity > round_eccentricity + 0.3,
            "{round:?} against {trailed:?}"
        );
        assert!(
            trailed.psf_area > 1.3 * round.psf_area,
            "{round:?} against {trailed:?}"
        );
    }

    #[test]
    fn eccentricity_follows_a_stars_axis_ratio() {
        // One Gaussian star, sigma 2.4 by 1.2 px at 30 degrees: e = sqrt(1 - 1/4).
        let (width, height) = (64, 64);
        let (cos, sin) = (30_f64.to_radians().cos(), 30_f64.to_radians().sin());
        let samples = (0..width * height)
            .map(|index| {
                let (dx, dy) = ((index % width) as f64 - 31.3, (index / width) as f64 - 32.6);
                let u = dx * cos + dy * sin;
                let v = -dx * sin + dy * cos;
                (10.0 + 1000.0 * (-0.5 * (u * u / 5.76 + v * v / 1.44)).exp()) as f32
            })
            .collect::<Vec<_>>();
        let (one, two) = star_shape(&samples, width, height, 31.0, 33.0, 10.0).unwrap();
        let eccentricity = eccentricity_of(one.hypot(two));
        assert!(
            (eccentricity - 0.75_f64.sqrt()).abs() < 0.02,
            "{eccentricity}"
        );
        // The vector points at twice the star's angle: 60 degrees.
        assert!(
            (two.atan2(one).to_degrees() - 60.0).abs() < 1.0,
            "{one} {two}"
        );
        // A star too near the edge for its reach is not measured.
        assert_eq!(star_shape(&samples, width, height, 1.0, 1.0, 10.0), None);
    }

    #[test]
    fn references_rank_the_flattest_near_the_best_then_by_score() {
        let score = |score: f32, background_variation: f32| {
            Some(ReferenceScore {
                stars: 10,
                median_star_area: 4.0,
                background: 0.1,
                background_variation,
                score,
                psf_area: 4.0,
                eccentricity: Some(0.2),
            })
        };
        // The best score is 10, so 7.5 and 8 are near it and 6 is not.
        let scores = [
            score(6.0, 0.1),
            None,
            score(10.0, 0.5),
            score(7.5, 0.2),
            score(8.0, 0.2),
            score(3.0, 0.0),
        ];
        assert_eq!(rank_references(&scores), [3, 4, 2, 0, 5]);
        assert!(rank_references(&[None]).is_empty());
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
