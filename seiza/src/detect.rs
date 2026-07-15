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

/// Numeric representation used by the star detector.
///
/// `Auto` keeps 8-bit inputs compact and preserves higher-precision inputs by
/// using the f32 path. The forced modes are useful for reproducible A/B tests
/// and callers that need the previous f32 behavior for an 8-bit image.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum DetectBackend {
    #[default]
    Auto,
    U8,
    F32,
}

#[derive(Debug, Clone)]
pub struct DetectConfig {
    /// Numeric representation used for background estimation and thresholding.
    pub backend: DetectBackend,
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
            backend: DetectBackend::Auto,
            tile_size: 64,
            sigma: 4.0,
            min_area: 3,
            // Broad and saturated stars in high-resolution processed images
            // can cover several thousand pixels. Elongation still rejects
            // trails and text-like components.
            max_area: 20_000,
            max_elongation: 2.5,
            ignore_border: 0,
            max_stars: 500,
        }
    }
}

/// Detect stars in an image. In [`DetectBackend::Auto`], all 8-bit image
/// variants use compact u8 luma while higher-precision formats use f32 luma.
/// Callers with large images may downsample first for speed (centroids are in
/// the coordinates of the image as passed).
pub fn detect_stars(image: &image::DynamicImage, config: &DetectConfig) -> Vec<DetectedStar> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Vec::new();
    }

    match config.backend {
        DetectBackend::Auto if is_8_bit(image) => detect_stars_u8(image, width, height, config),
        DetectBackend::Auto | DetectBackend::F32 => detect_stars_f32(image, width, height, config),
        DetectBackend::U8 => detect_stars_u8(image, width, height, config),
    }
}

fn is_8_bit(image: &image::DynamicImage) -> bool {
    matches!(
        image,
        image::DynamicImage::ImageLuma8(_)
            | image::DynamicImage::ImageLumaA8(_)
            | image::DynamicImage::ImageRgb8(_)
            | image::DynamicImage::ImageRgba8(_)
    )
}

fn detect_stars_f32(
    image: &image::DynamicImage,
    width: u32,
    height: u32,
    config: &DetectConfig,
) -> Vec<DetectedStar> {
    let luma = image.to_luma32f();
    detect_stars_luma(luma.as_raw(), width, height, config)
}

fn detect_stars_u8(
    image: &image::DynamicImage,
    width: u32,
    height: u32,
    config: &DetectConfig,
) -> Vec<DetectedStar> {
    if let image::DynamicImage::ImageLuma8(luma) = image {
        return detect_stars_luma(luma.as_raw(), width, height, config);
    }
    let luma = image.to_luma8();
    detect_stars_luma(luma.as_raw(), width, height, config)
}

/// Shared, statically-dispatched detector pipeline. Each sample type can use
/// its own optimized background and threshold implementation while component
/// extraction remains monomorphized over the same representation.
fn detect_stars_luma<T: DetectionSample>(
    pixels: &[T],
    width: u32,
    height: u32,
    config: &DetectConfig,
) -> Vec<DetectedStar> {
    let (background, noise) = T::estimate_background(pixels, width, height, config.tile_size);
    let excess = T::threshold_excess(
        pixels,
        width,
        height,
        config.tile_size,
        &background,
        &noise,
        config.sigma,
    );
    extract_stars(&excess, width, height, config)
}

trait DetectionSample: Copy {
    type Background: Copy + Send + Sync;

    fn estimate_background(
        pixels: &[Self],
        width: u32,
        height: u32,
        tile_size: u32,
    ) -> (Vec<Self::Background>, Vec<f32>);

    fn threshold_excess(
        pixels: &[Self],
        width: u32,
        height: u32,
        tile_size: u32,
        background: &[Self::Background],
        noise: &[f32],
        sigma: f32,
    ) -> Vec<Self>;

    fn is_positive(self) -> bool;
    fn normalized(self) -> f32;
}

impl DetectionSample for f32 {
    type Background = f32;

    fn estimate_background(
        pixels: &[Self],
        width: u32,
        height: u32,
        tile_size: u32,
    ) -> (Vec<Self::Background>, Vec<f32>) {
        estimate_background(pixels, width, height, tile_size)
    }

    fn threshold_excess(
        pixels: &[Self],
        width: u32,
        height: u32,
        tile_size: u32,
        background: &[Self::Background],
        noise: &[f32],
        sigma: f32,
    ) -> Vec<Self> {
        threshold_excess(pixels, width, height, tile_size, background, noise, sigma)
    }

    fn is_positive(self) -> bool {
        self > 0.0
    }

    fn normalized(self) -> f32 {
        self
    }
}

impl DetectionSample for u8 {
    type Background = u8;

    fn estimate_background(
        pixels: &[Self],
        width: u32,
        height: u32,
        tile_size: u32,
    ) -> (Vec<Self::Background>, Vec<f32>) {
        estimate_background_u8(pixels, width, height, tile_size)
    }

    fn threshold_excess(
        pixels: &[Self],
        width: u32,
        _height: u32,
        tile_size: u32,
        background: &[Self::Background],
        noise: &[f32],
        sigma: f32,
    ) -> Vec<Self> {
        threshold_excess_u8(pixels, width, tile_size, background, noise, sigma)
    }

    fn is_positive(self) -> bool {
        self != 0
    }

    fn normalized(self) -> f32 {
        self as f32 / 255.0
    }
}

fn extract_stars<T: DetectionSample>(
    excess: &[T],
    width: u32,
    height: u32,
    config: &DetectConfig,
) -> Vec<DetectedStar> {
    // Connected components over the thresholded mask (8-connectivity),
    // iterative flood fill to keep the stack bounded.
    let mut visited = vec![false; (width * height) as usize];
    let mut stars = Vec::new();
    let mut stack = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            if visited[idx] || !excess[idx].is_positive() {
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
                let value = excess[(cy * width + cx) as usize].normalized();
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
                        if !visited[nidx] && excess[nidx].is_positive() {
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

/// Per-tile median and MAD in the native 8-bit domain. A 256-bin histogram
/// avoids copying and selecting a full tile of f32 values.
fn estimate_background_u8(
    pixels: &[u8],
    width: u32,
    height: u32,
    tile_size: u32,
) -> (Vec<u8>, Vec<f32>) {
    use rayon::prelude::*;

    let tiles_x = width.div_ceil(tile_size);
    let tiles_y = height.div_ceil(tile_size);
    (0..tiles_x * tiles_y)
        .into_par_iter()
        .map(|tile| {
            let (tx, ty) = (tile % tiles_x, tile / tiles_x);
            let x0 = tx * tile_size;
            let x1 = ((tx + 1) * tile_size).min(width);
            let y0 = ty * tile_size;
            let y1 = ((ty + 1) * tile_size).min(height);
            let mut histogram = [0u32; 256];
            for y in y0..y1 {
                let row = (y * width) as usize;
                for &value in &pixels[row + x0 as usize..row + x1 as usize] {
                    histogram[value as usize] += 1;
                }
            }
            let count = ((x1 - x0) * (y1 - y0)) as usize;
            let target = count / 2 + 1;
            let mut cumulative = 0usize;
            let median = histogram
                .iter()
                .position(|&bin| {
                    cumulative += bin as usize;
                    cumulative >= target
                })
                .unwrap_or(0);

            let mut inside = histogram[median] as usize;
            let mut mad = 0usize;
            while inside < target && mad < 255 {
                mad += 1;
                if median + mad < histogram.len() {
                    inside += histogram[median + mad] as usize;
                }
                if mad <= median {
                    inside += histogram[median - mad] as usize;
                }
            }
            // The f32 path floors noise at 1e-6 in normalized units.
            (median as u8, (1.4826 * mad as f32).max(255.0e-6))
        })
        .unzip()
}

fn threshold_excess_u8(
    pixels: &[u8],
    width: u32,
    tile_size: u32,
    background: &[u8],
    noise: &[f32],
    sigma: f32,
) -> Vec<u8> {
    use rayon::prelude::*;

    let tiles_x = width.div_ceil(tile_size);
    let mut excess = vec![0u8; pixels.len()];
    excess
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, output)| {
            let ty = y as u32 / tile_size;
            let input = &pixels[y * width as usize..(y + 1) * width as usize];
            for tx in 0..tiles_x {
                let x0 = (tx * tile_size) as usize;
                let x1 = ((tx + 1) * tile_size).min(width) as usize;
                let tile = (ty * tiles_x + tx) as usize;
                let threshold = sigma * noise[tile];
                let background = background[tile];
                for (value, excess) in input[x0..x1].iter().zip(&mut output[x0..x1]) {
                    let difference = value.saturating_sub(background);
                    if difference as f32 > threshold {
                        *excess = difference;
                    }
                }
            }
        });
    excess
}

/// `max(pixel - background, 0)` where the excess clears the sigma
/// threshold, else 0 — vectorized per tile-row segment (the background
/// and threshold are constant within one). Multiversioned so release
/// binaries built for baseline x86-64 still dispatch AVX2 at runtime.
#[multiversion::multiversion(targets("x86_64+avx2+fma", "x86_64+sse4.1", "aarch64+neon"))]
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
    use image::{DynamicImage, ImageBuffer, Luma, Rgb};

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
    fn native_luma8_path_matches_f32_reference() {
        let truth = vec![
            (40.3, 55.7, 0.9),
            (170.5, 35.2, 0.6),
            (110.0, 130.0, 0.35),
            (210.1, 165.9, 0.2),
        ];
        let luma = synthetic_field(256, 192, &truth).to_luma8();
        let image = DynamicImage::ImageLuma8(luma);
        let u8_config = DetectConfig {
            backend: DetectBackend::U8,
            ..Default::default()
        };
        let f32_config = DetectConfig {
            backend: DetectBackend::F32,
            ..Default::default()
        };
        let actual = detect_stars(&image, &u8_config);
        let expected = detect_stars(&image, &f32_config);

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.area, expected.area);
            assert!((actual.x - expected.x).abs() < 1e-6);
            assert!((actual.y - expected.y).abs() < 1e-6);
            assert!((actual.flux - expected.flux).abs() < 1e-6);
            assert!((actual.peak - expected.peak).abs() < 1e-6);
        }
    }

    /// A signal that is real in 16-bit/f32 luma but rounds into the same u8
    /// bucket as its background. This makes it impossible for a test to pass
    /// by accidentally routing both forced backends through one code path.
    fn sub_u8_contrast_field() -> DynamicImage {
        let buffer = ImageBuffer::from_fn(64, 64, |x, y| {
            let value: u16 = if (30..33).contains(&x) && (30..33).contains(&y) {
                32_868
            } else {
                32_768
            };
            Luma([value])
        });
        DynamicImage::ImageLuma16(buffer)
    }

    #[test]
    fn forced_backends_execute_distinct_numeric_paths() {
        let image = sub_u8_contrast_field();
        let f32_stars = detect_stars(
            &image,
            &DetectConfig {
                backend: DetectBackend::F32,
                ..Default::default()
            },
        );
        let u8_stars = detect_stars(
            &image,
            &DetectConfig {
                backend: DetectBackend::U8,
                ..Default::default()
            },
        );

        assert_eq!(f32_stars.len(), 1, "f32 must retain sub-u8 contrast");
        assert!(u8_stars.is_empty(), "u8 must quantize the contrast away");
    }

    #[test]
    fn auto_uses_f32_backend_for_high_precision_input() {
        let image = sub_u8_contrast_field();
        let auto = detect_stars(&image, &DetectConfig::default());
        let forced = detect_stars(
            &image,
            &DetectConfig {
                backend: DetectBackend::F32,
                ..Default::default()
            },
        );

        assert_eq!(auto, forced);
        assert_eq!(auto.len(), 1);
    }

    #[test]
    fn auto_uses_u8_backend_for_rgb8() {
        let truth = vec![(40.3, 55.7, 0.9), (170.5, 35.2, 0.6)];
        let luma = synthetic_field(224, 128, &truth).to_luma8();
        let rgb = ImageBuffer::from_fn(luma.width(), luma.height(), |x, y| {
            let value = luma.get_pixel(x, y).0[0];
            Rgb([value, value, value])
        });
        let image = DynamicImage::ImageRgb8(rgb);
        let auto = detect_stars(&image, &DetectConfig::default());
        let forced = detect_stars(
            &image,
            &DetectConfig {
                backend: DetectBackend::U8,
                ..Default::default()
            },
        );

        assert_eq!(auto, forced);
        assert_eq!(auto.len(), truth.len());
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

    #[test]
    fn retains_broad_saturated_stars_in_processed_images() {
        let mut noise_state = 0xA11CE5EEDu64;
        let mut rand = move || {
            noise_state ^= noise_state << 13;
            noise_state ^= noise_state >> 7;
            noise_state ^= noise_state << 17;
            (noise_state >> 40) as f32 / 16777216.0
        };
        let buffer = ImageBuffer::from_fn(320, 320, |x, y| {
            let d2 = (x as f64 - 160.0).powi(2) + (y as f64 - 160.0).powi(2);
            let value = 0.04 + 0.01 * rand() + 0.95 * (-d2 / (2.0 * 22.0f64.powi(2))).exp() as f32;
            Luma([(value.min(1.0) * 65535.0) as u16])
        });
        let image = DynamicImage::ImageLuma16(buffer);

        let stars = detect_stars(&image, &DetectConfig::default());
        assert_eq!(stars.len(), 1, "{stars:?}");
        assert!(
            stars[0].area > 2500,
            "broad star area was {}",
            stars[0].area
        );
        let offset = (stars[0].x - 160.0).hypot(stars[0].y - 160.0);
        assert!(offset < 2.0, "broad star centroid shifted by {offset}px");
    }
}
