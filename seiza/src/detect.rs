//! Star detection: tile-based background estimation, sigma thresholding,
//! connected components, and flux-weighted centroids.

use image::GenericImageView;

/// A detected star in pixel coordinates (0-indexed, sub-pixel centroid).
#[derive(Debug, Clone, PartialEq)]
pub struct DetectedStar {
    pub x: f64,
    pub y: f64,
    /// Background-subtracted integrated flux
    pub flux: f64,
    /// Peak background-subtracted pixel value
    pub peak: f32,
    /// Component area in pixels
    pub area: u32,
}

#[derive(Debug, Clone)]
pub struct DetectConfig {
    /// Background/noise estimation tile size in pixels
    pub tile_size: u32,
    /// Detection threshold in noise sigmas above background
    pub sigma: f32,
    /// Reject components smaller than this many pixels (hot pixels)
    pub min_area: u32,
    /// Reject components larger than this many pixels (nebulosity, trails)
    pub max_area: u32,
    /// Reject components more elongated than this ratio of principal axes
    /// (text, trails, and edges; stars are nearly round)
    pub max_elongation: f32,
    /// Ignore detections within this many pixels of the image edges —
    /// captions, watermarks, and frame artifacts live there
    pub ignore_border: u32,
    /// Keep at most this many stars, brightest first
    pub max_stars: usize,
}

impl Default for DetectConfig {
    fn default() -> Self {
        Self {
            tile_size: 64,
            sigma: 4.0,
            min_area: 3,
            max_area: 2500,
            max_elongation: 2.5,
            ignore_border: 0,
            max_stars: 500,
        }
    }
}

/// Detect stars in an image. The image is converted to luma internally;
/// callers with large images should downsample first for speed (centroids
/// are in the coordinates of the image as passed).
pub fn detect_stars(image: &image::DynamicImage, config: &DetectConfig) -> Vec<DetectedStar> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let luma = image.to_luma32f();
    let pixels = luma.as_raw();

    let (background, noise) = estimate_background(pixels, width, height, config.tile_size);

    // Background-subtracted, thresholded excess, computed once (the flood
    // fill below revisits pixels up to nine times)
    let excess = threshold_excess(
        pixels,
        width,
        height,
        config.tile_size,
        &background,
        &noise,
        config.sigma,
    );
    let above = |x: u32, y: u32| -> f32 { excess[(y * width + x) as usize] };

    // Connected components over the thresholded mask (8-connectivity),
    // iterative flood fill to keep the stack bounded.
    let mut visited = vec![false; (width * height) as usize];
    let mut stars = Vec::new();
    let mut stack = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            if visited[idx] || above(x, y) <= 0.0 {
                continue;
            }

            let mut flux = 0.0f64;
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            let mut sxx = 0.0f64;
            let mut syy = 0.0f64;
            let mut sxy = 0.0f64;
            let mut peak = 0.0f32;
            let mut area = 0u32;

            stack.push((x, y));
            visited[idx] = true;
            while let Some((cx, cy)) = stack.pop() {
                let value = above(cx, cy);
                let v = value as f64;
                flux += v;
                sx += cx as f64 * v;
                sy += cy as f64 * v;
                sxx += cx as f64 * cx as f64 * v;
                syy += cy as f64 * cy as f64 * v;
                sxy += cx as f64 * cy as f64 * v;
                peak = peak.max(value);
                area += 1;

                let x0 = cx.saturating_sub(1);
                let y0 = cy.saturating_sub(1);
                for ny in y0..=(cy + 1).min(height - 1) {
                    for nx in x0..=(cx + 1).min(width - 1) {
                        let nidx = (ny * width + nx) as usize;
                        if !visited[nidx] && above(nx, ny) > 0.0 {
                            visited[nidx] = true;
                            stack.push((nx, ny));
                        }
                    }
                }
            }

            if area >= config.min_area && area <= config.max_area && flux > 0.0 {
                let (cx, cy) = (sx / flux, sy / flux);
                let border = config.ignore_border as f64;
                if cx < border
                    || cy < border
                    || cx >= width as f64 - border
                    || cy >= height as f64 - border
                {
                    continue;
                }
                if elongation(
                    sxx / flux - cx * cx,
                    syy / flux - cy * cy,
                    sxy / flux - cx * cy,
                ) <= config.max_elongation as f64
                {
                    stars.push(DetectedStar {
                        x: cx,
                        y: cy,
                        flux,
                        peak,
                        area,
                    });
                }
            }
        }
    }

    stars.sort_by(|a, b| b.flux.total_cmp(&a.flux));
    stars.truncate(config.max_stars);
    stars
}

/// Per-tile background (median) and noise (MAD-based sigma) maps.
fn estimate_background(
    pixels: &[f32],
    width: u32,
    height: u32,
    tile_size: u32,
) -> (Vec<f32>, Vec<f32>) {
    use rayon::prelude::*;

    let tiles_x = width.div_ceil(tile_size);
    let tiles_y = height.div_ceil(tile_size);
    // Tiles are independent: estimate them in parallel
    (0..tiles_x * tiles_y)
        .into_par_iter()
        .map(|t| {
            let (tx, ty) = (t % tiles_x, t / tiles_x);
            let x1 = ((tx + 1) * tile_size).min(width);
            let y1 = ((ty + 1) * tile_size).min(height);
            let mut tile = Vec::with_capacity((tile_size * tile_size) as usize);
            for y in ty * tile_size..y1 {
                let row = (y * width) as usize;
                tile.extend_from_slice(&pixels[row + (tx * tile_size) as usize..row + x1 as usize]);
            }
            let median = median_in_place(&mut tile);
            for v in &mut tile {
                *v = (*v - median).abs();
            }
            let mad = median_in_place(&mut tile);
            // 1.4826 * MAD estimates sigma for a normal distribution; floor
            // avoids a zero threshold on perfectly flat synthetic tiles
            (median, (1.4826 * mad).max(1e-6))
        })
        .unzip()
}

/// `max(pixel - background, 0)` where the excess clears the sigma
/// threshold, else 0 — vectorized per tile-row segment (the background
/// and threshold are constant within one).
fn threshold_excess(
    pixels: &[f32],
    width: u32,
    height: u32,
    tile_size: u32,
    background: &[f32],
    noise: &[f32],
    sigma: f32,
) -> Vec<f32> {
    use wide::{CmpGt, f32x8};

    let tiles_x = width.div_ceil(tile_size);
    let mut excess = vec![0.0f32; pixels.len()];
    for y in 0..height {
        let ty = y / tile_size;
        let row = (y * width) as usize;
        for tx in 0..tiles_x {
            let x0 = (tx * tile_size) as usize;
            let x1 = (((tx + 1) * tile_size).min(width)) as usize;
            let tile = (ty * tiles_x + tx) as usize;
            let bg = f32x8::splat(background[tile]);
            let threshold = f32x8::splat(sigma * noise[tile]);

            let seg = &pixels[row + x0..row + x1];
            let out = &mut excess[row + x0..row + x1];
            let mut chunks = seg.chunks_exact(8);
            let mut out_chunks = out.chunks_exact_mut(8);
            for (chunk, out_chunk) in (&mut chunks).zip(&mut out_chunks) {
                let value = f32x8::from(<[f32; 8]>::try_from(chunk).unwrap()) - bg;
                let keep = value.cmp_gt(threshold);
                out_chunk.copy_from_slice(&keep.blend(value, f32x8::ZERO).to_array());
            }
            let done = seg.len() - chunks.remainder().len();
            for (value, out_value) in chunks.remainder().iter().zip(&mut out[done..]) {
                let v = value - background[tile];
                if v > sigma * noise[tile] {
                    *out_value = v;
                }
            }
        }
    }
    excess
}

/// Ratio of the principal axes of the flux distribution (≥ 1).
fn elongation(mxx: f64, myy: f64, mxy: f64) -> f64 {
    let trace = mxx + myy;
    let root = ((mxx - myy).powi(2) + 4.0 * mxy * mxy).sqrt();
    let l1 = ((trace + root) / 2.0).max(0.0);
    let l2 = ((trace - root) / 2.0).max(1e-12);
    (l1 / l2).sqrt()
}

fn median_in_place(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    let (_, median, _) = values.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    *median
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, Luma};

    /// Deterministic synthetic star field: Gaussian spots + mild noise
    fn synthetic_field(width: u32, height: u32, stars: &[(f64, f64, f32)]) -> DynamicImage {
        let mut noise_state = 0x2545F4914F6CDD1Du64;
        let mut rand = move || {
            noise_state ^= noise_state << 13;
            noise_state ^= noise_state >> 7;
            noise_state ^= noise_state << 17;
            (noise_state >> 40) as f32 / 16777216.0 // [0, 1)
        };

        let buffer = ImageBuffer::from_fn(width, height, |x, y| {
            let mut value = 0.05 + 0.01 * rand();
            for &(sx, sy, amplitude) in stars {
                let d2 = (x as f64 - sx).powi(2) + (y as f64 - sy).powi(2);
                value += amplitude * (-d2 / (2.0 * 1.6f64.powi(2))).exp() as f32;
            }
            Luma([(value.min(1.0) * 65535.0) as u16])
        });
        DynamicImage::ImageLuma16(buffer)
    }

    #[test]
    fn finds_synthetic_stars_with_subpixel_accuracy() {
        let truth: Vec<(f64, f64, f32)> = vec![
            (50.3, 60.7, 0.9),
            (200.5, 30.2, 0.6),
            (128.0, 128.0, 0.4),
            (33.7, 220.4, 0.25),
            (240.1, 240.9, 0.15),
        ];
        let image = synthetic_field(256, 256, &truth);
        let stars = detect_stars(&image, &DetectConfig::default());

        assert_eq!(stars.len(), truth.len(), "{stars:?}");
        // Brightest first
        assert!(stars[0].flux > stars[1].flux);

        for (sx, sy, _) in &truth {
            let best = stars
                .iter()
                .map(|s| ((s.x - sx).powi(2) + (s.y - sy).powi(2)).sqrt())
                .fold(f64::INFINITY, f64::min);
            assert!(best < 0.5, "star at ({sx}, {sy}) missed by {best}px");
        }
    }

    #[test]
    fn empty_and_flat_images_yield_nothing() {
        let flat = synthetic_field(128, 128, &[]);
        assert!(detect_stars(&flat, &DetectConfig::default()).is_empty());
    }

    #[test]
    fn rejects_elongated_shapes_like_text_strokes() {
        // A bright horizontal bar (a "watermark stroke") plus one round star
        let mut noise_state = 0xDEADBEEFu64;
        let mut rand = move || {
            noise_state ^= noise_state << 13;
            noise_state ^= noise_state >> 7;
            noise_state ^= noise_state << 17;
            (noise_state >> 40) as f32 / 16777216.0
        };
        let buffer = ImageBuffer::from_fn(220, 120, |x, y| {
            let mut value = 0.05 + 0.01 * rand();
            if (40..=160).contains(&x) && (90..=93).contains(&y) {
                value += 0.8; // stroke
            }
            let d2 = (x as f64 - 60.0).powi(2) + (y as f64 - 40.0).powi(2);
            value += 0.7 * (-d2 / (2.0 * 1.6f64.powi(2))).exp() as f32;
            Luma([(value.min(1.0) * 65535.0) as u16])
        });
        let image = DynamicImage::ImageLuma16(buffer);

        let stars = detect_stars(&image, &DetectConfig::default());
        assert_eq!(stars.len(), 1, "{stars:?}");
        assert!((stars[0].x - 60.0).abs() < 0.5);
        assert!((stars[0].y - 40.0).abs() < 0.5);
    }

    #[test]
    fn max_stars_caps_output_brightest_first() {
        let truth: Vec<(f64, f64, f32)> = (0..20)
            .map(|i| {
                (
                    20.0 + 30.0 * (i % 5) as f64,
                    20.0 + 30.0 * (i / 5) as f64,
                    0.2 + 0.03 * i as f32,
                )
            })
            .collect();
        let image = synthetic_field(160, 160, &truth);
        let config = DetectConfig {
            max_stars: 7,
            ..Default::default()
        };
        let stars = detect_stars(&image, &config);
        assert_eq!(stars.len(), 7);
        assert!(stars.windows(2).all(|w| w[0].flux >= w[1].flux));
    }
}
