//! Recovering the pedestal a camera added, from the pixels alone.
//!
//! Dividing by a flat only works on a signal that starts at zero. Cameras do
//! not oblige: they add a constant offset so read noise has somewhere to go,
//! and that offset is normally removed by subtracting a bias or dark master.
//! Without one, dividing by the flat scales the offset along with the signal
//! and leaves an inverted print of the vignette across the frame.
//!
//! When the offset cannot be measured — no bias, no dark — it can still be
//! fitted. Across a frame the sky background varies with the flat's own
//! response: where the optics pass less light the sky is dimmer, in
//! proportion. The pedestal is the part that does not vary, so fitting
//! background against flat response and reading off the intercept recovers
//! it.
//!
//! ```text
//!   background = sky × response + pedestal
//!                                 ^^^^^^^^ the intercept, what we want
//! ```
//!
//! This is a fit, not a measurement, and it declines rather than guesses. It
//! needs enough of the frame to sample, enough variation in the flat to give
//! the line a lever, and a slope that makes physical sense. A caller with a
//! header value for the offset should corroborate against it rather than
//! trust either alone.
//!
//! # It reads low, on purpose
//!
//! Sky brightness per tile is taken well below the median so stars and
//! nebulosity cannot drag it up, which also puts it below the true sky by
//! roughly 0.8 times the frame's noise. That offset lands in the intercept,
//! so the fitted pedestal comes back low by about the same amount.
//!
//! This is the safe direction. Subtracting too little pedestal leaves a trace
//! of the vignette; subtracting too much drives the background negative and
//! the flat division then amplifies the error. Differences between frames are
//! unaffected — the bias is common to both.

use crate::LinearImageRef;

/// Tiles needed before a fit is worth attempting, and after clipping.
const MIN_TILES: usize = 32;

/// The flat has to vary by at least this much across the frame. A flat that
/// is nearly uniform gives the line no lever, and the intercept it implies is
/// noise.
const MIN_RESPONSE_SPAN: f32 = 0.02;

/// Where in a tile's brightness distribution the sky sits. Well below the
/// median, so stars and nebulosity do not drag the estimate up.
const BACKGROUND_PERCENTILE: f64 = 0.2;

/// Sigma clip applied to the fit residuals, and how many times.
const CLIP_SIGMA: f64 = 2.5;
const CLIP_ROUNDS: usize = 3;

/// Fit the pedestal in `light`, in the light's own units.
///
/// Both images must be mono and the same size. Returns `None` when the frame
/// cannot support a fit — too few usable tiles, too flat a flat, or a slope
/// that says the model does not describe this field.
///
/// A colour-filter-array frame is not a candidate: interleaved photosites
/// respond differently per channel, and the single-line model does not hold
/// across them. Callers should debayer first or skip.
pub fn fit_flat_pedestal(light: LinearImageRef<'_>, flat: LinearImageRef<'_>) -> Option<f32> {
    if light.channels() != 1 || flat.channels() != 1 {
        return None;
    }
    if light.width() != flat.width() || light.height() != flat.height() {
        return None;
    }
    let (width, height) = (light.width(), light.height());
    let tile = (width.min(height) / 32).clamp(8, 128);
    let (light_data, flat_data) = (light.data(), flat.data());

    // One point per tile: what the optics passed there, and how bright the sky
    // was there.
    let mut pairs: Vec<(f32, f32)> = Vec::new();
    let mut light_tile = Vec::with_capacity(tile * tile);
    let mut flat_tile = Vec::with_capacity(tile * tile);
    for tile_y in (0..height.saturating_sub(tile - 1)).step_by(tile) {
        for tile_x in (0..width.saturating_sub(tile - 1)).step_by(tile) {
            light_tile.clear();
            flat_tile.clear();
            for y in tile_y..tile_y + tile {
                let row = y * width + tile_x;
                light_tile.extend(light_data[row..row + tile].iter().filter(|v| v.is_finite()));
                flat_tile.extend(flat_data[row..row + tile].iter().filter(|v| v.is_finite()));
            }
            if light_tile.is_empty() || flat_tile.is_empty() {
                continue;
            }
            let background_rank = ((light_tile.len() as f64 * BACKGROUND_PERCENTILE) as usize)
                .min(light_tile.len() - 1);
            let (_, background, _) =
                light_tile.select_nth_unstable_by(background_rank, f32::total_cmp);
            let background = *background;
            let median_rank = flat_tile.len() / 2;
            let (_, response, _) = flat_tile.select_nth_unstable_by(median_rank, f32::total_cmp);
            pairs.push((*response, background));
        }
    }
    if pairs.len() < MIN_TILES {
        return None;
    }
    let span = pairs.iter().map(|(v, _)| *v).fold(f32::MIN, f32::max)
        - pairs.iter().map(|(v, _)| *v).fold(f32::MAX, f32::min);
    if span < MIN_RESPONSE_SPAN {
        return None;
    }

    let mut kept = pairs;
    let mut slope = 0.0f64;
    let mut intercept = 0.0f64;
    for _ in 0..CLIP_ROUNDS {
        let count = kept.len() as f64;
        let sum_v: f64 = kept.iter().map(|(v, _)| f64::from(*v)).sum();
        let sum_b: f64 = kept.iter().map(|(_, b)| f64::from(*b)).sum();
        let sum_vv: f64 = kept
            .iter()
            .map(|(v, _)| f64::from(*v) * f64::from(*v))
            .sum();
        let sum_vb: f64 = kept
            .iter()
            .map(|(v, b)| f64::from(*v) * f64::from(*b))
            .sum();
        let denominator = count * sum_vv - sum_v * sum_v;
        if denominator.abs() < f64::EPSILON {
            return None;
        }
        slope = (count * sum_vb - sum_v * sum_b) / denominator;
        intercept = (sum_b - slope * sum_v) / count;
        let residual_sigma = (kept
            .iter()
            .map(|(v, b)| {
                let residual = f64::from(*b) - (slope * f64::from(*v) + intercept);
                residual * residual
            })
            .sum::<f64>()
            / count)
            .sqrt();
        if residual_sigma == 0.0 {
            break;
        }
        let limit = residual_sigma * CLIP_SIGMA;
        kept.retain(|(v, b)| (f64::from(*b) - (slope * f64::from(*v) + intercept)).abs() <= limit);
        if kept.len() < MIN_TILES {
            return None;
        }
    }
    // The sky term cannot be negative: an anti-correlation between background
    // and flat response means the model does not describe this field — a
    // gradient running against the vignette, say.
    if slope < 0.0 || !intercept.is_finite() {
        return None;
    }
    Some(intercept as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIDTH: usize = 512;
    const HEIGHT: usize = 512;

    /// A flat with a radial vignette: 1.0 at centre, falling towards corners.
    fn vignette(depth: f32) -> Vec<f32> {
        let mut data = vec![0.0f32; WIDTH * HEIGHT];
        let (cx, cy) = (WIDTH as f32 / 2.0, HEIGHT as f32 / 2.0);
        let max_r2 = cx * cx + cy * cy;
        for (index, sample) in data.iter_mut().enumerate() {
            let (x, y) = ((index % WIDTH) as f32, (index / WIDTH) as f32);
            let r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            *sample = 1.0 - depth * (r2 / max_r2);
        }
        data
    }

    /// Deterministic noise, so a fit can be checked against a chosen number.
    fn noise_at(index: usize, seed: u64, sigma: f32) -> f32 {
        let mut state = (index as u64).wrapping_mul(6_364_136_223_846_793_005) ^ seed;
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let first = ((state >> 11) as f64 / (1u64 << 53) as f64).mul_add(0.999_999, 5e-7);
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let second = ((state >> 11) as f64 / (1u64 << 53) as f64).mul_add(0.999_999, 5e-7);
        ((-2.0 * first.ln()).sqrt() * (std::f64::consts::TAU * second).cos()) as f32 * sigma
    }

    /// A light whose sky follows the flat, plus a constant pedestal and the
    /// shot noise that makes the percentile behave as it does on a real frame.
    fn light_over(flat: &[f32], sky: f32, pedestal: f32, sigma: f32, seed: u64) -> Vec<f32> {
        flat.iter()
            .enumerate()
            .map(|(index, response)| sky * response + pedestal + noise_at(index, seed, sigma))
            .collect()
    }

    fn image(data: &[f32]) -> LinearImageRef<'_> {
        LinearImageRef::new(data, WIDTH, HEIGHT, 1).unwrap()
    }

    #[test]
    fn recovers_a_pedestal_from_the_sky_against_the_vignette() {
        const SIGMA: f32 = 15.0;
        let flat = vignette(0.35);
        for pedestal in [200.0f32, 512.0, 1000.0] {
            let light = light_over(&flat, 900.0, pedestal, SIGMA, 7);
            let fitted = fit_flat_pedestal(image(&light), image(&flat)).expect("fittable");
            // Low by roughly 0.8 sigma, and never high: see the module docs.
            assert!(
                fitted <= pedestal + SIGMA && fitted >= pedestal - 3.0 * SIGMA,
                "fitted {fitted} for a pedestal of {pedestal}"
            );
        }
    }

    #[test]
    fn the_difference_between_two_pedestals_comes_back_exactly() {
        // The percentile bias is common to both frames, so it cancels. This
        // is the property a caller comparing frames actually depends on.
        let flat = vignette(0.35);
        let low = fit_flat_pedestal(
            image(&light_over(&flat, 900.0, 200.0, 15.0, 11)),
            image(&flat),
        )
        .expect("fittable");
        let high = fit_flat_pedestal(
            image(&light_over(&flat, 900.0, 700.0, 15.0, 11)),
            image(&flat),
        )
        .expect("fittable");
        assert!(
            ((high - low) - 500.0).abs() < 2.0,
            "recovered a difference of {} for a true 500",
            high - low
        );
    }

    #[test]
    fn a_frame_with_no_pedestal_fits_near_zero() {
        let flat = vignette(0.35);
        let light = light_over(&flat, 900.0, 0.0, 15.0, 13);
        let fitted = fit_flat_pedestal(image(&light), image(&flat)).expect("fittable");
        assert!(fitted.abs() < 45.0, "fitted {fitted} where there is none");
    }

    #[test]
    fn a_flat_with_no_vignette_gives_the_line_no_lever() {
        // Uniform response: every tile sits at the same x, so the intercept is
        // unconstrained. Declining beats returning a number.
        let flat = vignette(0.0);
        let light = light_over(&flat, 900.0, 500.0, 15.0, 17);
        assert!(fit_flat_pedestal(image(&light), image(&flat)).is_none());
    }

    #[test]
    fn a_gradient_running_against_the_vignette_is_refused() {
        // Sky brightening towards the corners, where the optics pass less.
        // The model does not describe this field, and a negative slope says so.
        let flat = vignette(0.35);
        let light: Vec<f32> = flat
            .iter()
            .map(|response| 900.0 * (2.0 - response) + 400.0)
            .collect();
        assert!(fit_flat_pedestal(image(&light), image(&flat)).is_none());
    }

    #[test]
    fn mismatched_or_colour_frames_are_refused() {
        let flat = vignette(0.35);
        let light = light_over(&flat, 900.0, 500.0, 15.0, 17);
        let small = vec![0.0f32; 16 * 16];
        assert!(
            fit_flat_pedestal(
                image(&light),
                LinearImageRef::new(&small, 16, 16, 1).unwrap()
            )
            .is_none(),
            "different sizes"
        );

        let colour = vec![0.0f32; WIDTH * HEIGHT * 3];
        let colour = LinearImageRef::new(&colour, WIDTH, HEIGHT, 3).unwrap();
        assert!(
            fit_flat_pedestal(colour, image(&flat)).is_none(),
            "a CFA or colour frame does not fit one line"
        );
    }
}
