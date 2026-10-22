//! Star detection: tile-based background estimation, sigma thresholding,
//! connected components, and flux-weighted centroids.

use image::GenericImageView;
use seiza_stats::{median_in_place, robust_sigma_in_place};

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

/// Measure each of `stars` again on an image's linear samples, for the
/// positions a plate solve fits.
///
/// Detection finds stars as connected regions above a per-tile background,
/// often in a stretched 8-bit copy, and places each at the flux-weighted
/// mean of its region. That centroid is biased: the stretch is not linear,
/// a halo can spread over thousands of pixels, and a vignetting gradient
/// across a flat tile background pulls stars toward the brighter side, so
/// a wide field's positions shrink toward a point. Here each star is
/// measured much as Asterion measures its stars: the background is a plane
/// fitted to an annulus around it, so a gradient's slope goes too, the width
/// comes from the pixels above half the peak, and the centroid is iterated
/// with a Gaussian weight 0.6 times that width.
///
/// `sample(x, y)` reads a `width` × `height` grid of linear samples
/// covering the image in blocks of `binning` × `binning` pixels (see
/// `seiza_fits::LinearSampler`); NaN samples are skipped. A star that
/// cannot be measured, or whose centroid wanders off its peak, keeps its
/// position. Flux, peak and area are left as detected.
pub fn refine_centroids(
    stars: &mut [DetectedStar],
    width: usize,
    height: usize,
    binning: usize,
    sample: &(dyn Fn(usize, usize) -> f32 + Sync),
) {
    use rayon::prelude::*;

    if width < 16 || height < 16 || binning == 0 {
        return;
    }
    let bin = binning as f64;
    let centre = (bin - 1.0) / 2.0;
    stars.par_iter_mut().for_each(|star| {
        let seed = ((star.x - centre) / bin, (star.y - centre) / bin);
        if let Some((x, y)) =
            measure_centroid(seed, star.area as f64 / (bin * bin), width, height, sample)
        {
            star.x = x * bin + centre;
            star.y = y * bin + centre;
        }
    });
}

/// The plane `[a, b, c]`, value `a + b dx + c dy`, that fits `samples` of
/// `(dx, dy, value)` by least squares, fitted again without the samples more
/// than three MADs off, such as other stars in the annulus.
fn background_plane(samples: &[(f64, f64, f64)]) -> Option<[f64; 3]> {
    let fit = |samples: &mut dyn Iterator<Item = &(f64, f64, f64)>| -> Option<[f64; 3]> {
        let mut normal = [[0.0f64; 3]; 3];
        let mut target = [0.0f64; 3];
        for &(dx, dy, value) in samples {
            let row = [1.0, dx, dy];
            for i in 0..3 {
                for j in 0..3 {
                    normal[i][j] += row[i] * row[j];
                }
                target[i] += row[i] * value;
            }
        }
        solve3x3(normal, target)
    };
    let first = fit(&mut samples.iter())?;
    let residual =
        |&(dx, dy, value): &(f64, f64, f64)| value - (first[0] + first[1] * dx + first[2] * dy);
    let mut deviations: Vec<f64> = samples
        .iter()
        .map(|sample| residual(sample).abs())
        .collect();
    deviations.sort_by(f64::total_cmp);
    let limit = 3.0 * 1.4826 * deviations[deviations.len() / 2];
    let kept: Vec<&(f64, f64, f64)> = samples
        .iter()
        .filter(|sample| residual(sample).abs() <= limit)
        .collect();
    if kept.len() < 12 {
        return Some(first);
    }
    fit(&mut kept.into_iter()).or(Some(first))
}

/// The solution of a 3 x 3 linear system by Cramer's rule, or `None` when it
/// is singular.
fn solve3x3(m: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(&m);
    if d.abs() < 1e-12 {
        return None;
    }
    let mut out = [0.0; 3];
    for (column, value) in out.iter_mut().enumerate() {
        let mut replaced = m;
        for row in 0..3 {
            replaced[row][column] = b[row];
        }
        *value = det(&replaced) / d;
    }
    Some(out)
}

/// [`refine_centroids`] for one star seeded at grid position `seed`, with
/// its detected `area` in grid samples.
fn measure_centroid(
    seed: (f64, f64),
    area: f64,
    width: usize,
    height: usize,
    sample: &(dyn Fn(usize, usize) -> f32 + Sync),
) -> Option<(f64, f64)> {
    let inside = |x: i64, y: i64| x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height;
    let read = |x: i64, y: i64| -> Option<f64> {
        if !inside(x, y) {
            return None;
        }
        let value = sample(x as usize, y as usize);
        value.is_finite().then_some(value as f64)
    };

    // Background: a plane fitted to an annulus clear of the star and its
    // halo. A flat level would leave a sky gradient's slope under the star,
    // and with only positive excess counted, the uphill side would pull the
    // centroid toward it.
    let radius = (area / std::f64::consts::PI).sqrt().clamp(1.0, 20.0);
    let inner = (3.0 * radius).max(5.0);
    let outer = inner + 4.0;
    let (sx, sy) = (seed.0.round() as i64, seed.1.round() as i64);
    let reach = outer.ceil() as i64;
    let mut ring = Vec::new();
    for y in sy - reach..=sy + reach {
        for x in sx - reach..=sx + reach {
            let d = ((x - sx) as f64).hypot((y - sy) as f64);
            if d >= inner
                && d <= outer
                && let Some(value) = read(x, y)
            {
                ring.push(((x - sx) as f64, (y - sy) as f64, value));
            }
        }
    }
    if ring.len() < 12 {
        return None;
    }
    let plane = background_plane(&ring)?;
    let excess = |x: i64, y: i64| {
        read(x, y).map(|value| {
            value - (plane[0] + plane[1] * (x - sx) as f64 + plane[2] * (y - sy) as f64)
        })
    };

    // The peak, within a pixel of the seed.
    let (mut mx, mut my, mut peak) = (sx, sy, excess(sx, sy)?);
    for y in sy - 1..=sy + 1 {
        for x in sx - 1..=sx + 1 {
            if let Some(value) = excess(x, y)
                && value > peak
            {
                (mx, my, peak) = (x, y, value);
            }
        }
    }
    if peak <= 0.0 || mx < 2 || my < 2 || mx as usize + 2 >= width || my as usize + 2 >= height {
        return None;
    }

    // Width: the pixels above half the peak, in a window that grows until
    // it holds the star.
    let half = peak / 2.0;
    let mut window = 4i64;
    let count = loop {
        let mut count = 0usize;
        for y in my - window..=my + window {
            for x in mx - window..=mx + window {
                if excess(x, y).is_some_and(|value| value > half) {
                    count += 1;
                }
            }
        }
        let side = (2 * window + 1) as usize;
        if window < 32 && count > side * side / 2 {
            window *= 2;
        } else {
            break count;
        }
    };
    let fwhm = 2.0 * (count as f64 / std::f64::consts::PI).sqrt();

    // Gaussian-weighted centroid. For a Gaussian star, a Gaussian window of
    // its own size halves the measured shift, so each step doubles it.
    let sigma = (0.6 * fwhm).max(1.0);
    let reach = (3.0 * sigma).ceil().min(40.0) as i64;
    let inverse = 1.0 / (2.0 * sigma * sigma);
    let (mut cx, mut cy) = (mx as f64, my as f64);
    for _ in 0..8 {
        let (rx, ry) = (cx.round() as i64, cy.round() as i64);
        let (mut sum, mut sum_x, mut sum_y) = (0.0, 0.0, 0.0);
        for y in ry - reach..=ry + reach {
            let dy = y as f64 - cy;
            let weight_y = (-dy * dy * inverse).exp();
            for x in rx - reach..=rx + reach {
                let Some(value) = excess(x, y).filter(|value| *value > 0.0) else {
                    continue;
                };
                let dx = x as f64 - cx;
                let weight = value * weight_y * (-dx * dx * inverse).exp();
                sum += weight;
                sum_x += weight * dx;
                sum_y += weight * dy;
            }
        }
        if sum <= 0.0 {
            return None;
        }
        let (shift_x, shift_y) = (2.0 * sum_x / sum, 2.0 * sum_y / sum);
        cx += shift_x;
        cy += shift_y;
        if (cx - mx as f64).abs() > reach as f64 || (cy - my as f64).abs() > reach as f64 {
            return None;
        }
        if shift_x * shift_x + shift_y * shift_y < 1e-5 {
            break;
        }
    }
    Some((cx, cy))
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

/// Detect stars from a normalized, row-major f32 luma buffer.
///
/// This entry point is useful for image containers such as FITS whose native
/// precision cannot be represented by a grayscale [`image::DynamicImage`]
/// variant. Invalid dimensions return no detections.
pub fn detect_stars_luma_f32(
    pixels: &[f32],
    width: u32,
    height: u32,
    config: &DetectConfig,
) -> Vec<DetectedStar> {
    let expected = width
        .checked_mul(height)
        .and_then(|count| usize::try_from(count).ok());
    if width == 0 || height == 0 || expected != Some(pixels.len()) {
        return Vec::new();
    }
    detect_stars_luma(pixels, width, height, config)
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
    extract_stars(excess, width, height, config)
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

    /// The excess of a pixel below the threshold, and of one already
    /// gathered into a star.
    const ZERO: Self;

    fn is_positive(self) -> bool;
    fn normalized(self) -> f32;

    /// The first pixel at or after `from` in `row` whose excess is positive,
    /// or `row.len()`.
    fn next_positive(row: &[Self], from: usize) -> usize {
        row[from..]
            .iter()
            .position(|value| value.is_positive())
            .map_or(row.len(), |offset| from + offset)
    }
}

impl DetectionSample for f32 {
    type Background = f32;
    const ZERO: Self = 0.0;

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
    const ZERO: Self = 0;

    /// Most of a sky is below the threshold: blocks of 32 pixels are tested
    /// at once, which the compiler runs as a few vector instructions, and
    /// only a block with an excess in it is searched pixel by pixel.
    fn next_positive(row: &[Self], from: usize) -> usize {
        const BLOCK: usize = 32;
        let mut at = from;
        while at + BLOCK <= row.len() {
            if row[at..at + BLOCK]
                .iter()
                .fold(0u8, |any, &value| any | value)
                != 0
            {
                break;
            }
            at += BLOCK;
        }
        row[at..]
            .iter()
            .position(|&value| value != 0)
            .map_or(row.len(), |offset| at + offset)
    }

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
    mut excess: Vec<T>,
    width: u32,
    height: u32,
    config: &DetectConfig,
) -> Vec<DetectedStar> {
    // Connected components over the thresholded mask (8-connectivity),
    // iterative flood fill to keep the stack bounded. A pixel is zeroed as
    // it joins a star, so it is never gathered twice, and its excess rides
    // on the stack until it is added in.
    let mut stars = Vec::new();
    let mut stack: Vec<(u32, u32, T)> = Vec::new();
    let row_len = width as usize;

    for y in 0..height {
        let row_start = y as usize * row_len;
        let mut x = 0usize;
        loop {
            x = T::next_positive(&excess[row_start..row_start + row_len], x);
            if x >= row_len {
                break;
            }
            let idx = row_start + x;
            let mut moments = Moments::default();
            stack.push((x as u32, y, excess[idx]));
            excess[idx] = T::ZERO;
            while let Some((cx, cy, value)) = stack.pop() {
                moments.add(cx, cy, value.normalized());
                let x0 = cx.saturating_sub(1);
                let y0 = cy.saturating_sub(1);
                for ny in y0..=(cy + 1).min(height - 1) {
                    for nx in x0..=(cx + 1).min(width - 1) {
                        let nidx = (ny * width + nx) as usize;
                        if excess[nidx].is_positive() {
                            stack.push((nx, ny, excess[nidx]));
                            excess[nidx] = T::ZERO;
                        }
                    }
                }
            }
            x += 1;
            stars.extend(moments.star(width, height, config));
        }
    }

    stars.sort_by(|a, b| b.flux.total_cmp(&a.flux));
    stars.truncate(config.max_stars);
    stars
}

/// Stars from a thresholded excess image given only by its positive pixels:
/// `positives` holds each one's index in the `width` × `height` image and
/// its excess above the background, in row-major order. The same stars, in
/// the same order and to the bit, that detection extracts from the whole
/// excess image, for a detector that thresholds elsewhere, such as on a GPU,
/// and hands back only what passed. The flood fill visits the pixels in the
/// same order, so every sum adds the same values in the same order.
pub fn detect_stars_from_excess(
    width: u32,
    height: u32,
    positives: &[(u32, f32)],
    config: &DetectConfig,
) -> Vec<DetectedStar> {
    if width == 0 || height == 0 {
        return Vec::new();
    }
    // Each row's positives, as a range of `positives`; a pixel is found by
    // a binary search of its row's.
    let mut rows = vec![0usize; height as usize + 1];
    for &(index, _) in positives {
        let row = (index / width) as usize;
        if row < height as usize {
            rows[row + 1] += 1;
        }
    }
    for row in 0..height as usize {
        rows[row + 1] += rows[row];
    }
    let mut taken = vec![false; positives.len()];
    let find = |x: u32, y: u32, taken: &[bool]| -> Option<usize> {
        let range = rows[y as usize]..rows[y as usize + 1];
        let index = y * width + x;
        let at = range.start
            + positives[range]
                .binary_search_by_key(&index, |&(index, _)| index)
                .ok()?;
        (!taken[at] && positives[at].1 > 0.0).then_some(at)
    };
    let mut stars = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    for seed in 0..positives.len() {
        let (index, value) = positives[seed];
        if taken[seed] || value <= 0.0 || index / width >= height {
            continue;
        }
        let mut moments = Moments::default();
        stack.push(seed);
        taken[seed] = true;
        while let Some(at) = stack.pop() {
            let (index, value) = positives[at];
            let (cx, cy) = (index % width, index / width);
            moments.add(cx, cy, value);
            let x0 = cx.saturating_sub(1);
            let y0 = cy.saturating_sub(1);
            for ny in y0..=(cy + 1).min(height - 1) {
                for nx in x0..=(cx + 1).min(width - 1) {
                    if let Some(neighbour) = find(nx, ny, &taken) {
                        stack.push(neighbour);
                        taken[neighbour] = true;
                    }
                }
            }
        }
        stars.extend(moments.star(width, height, config));
    }
    stars.sort_by(|a, b| b.flux.total_cmp(&a.flux));
    stars.truncate(config.max_stars);
    stars
}

/// A component's flux-weighted moments as its pixels join it.
#[derive(Default)]
struct Moments {
    flux: f64,
    sx: f64,
    sy: f64,
    sxx: f64,
    syy: f64,
    sxy: f64,
    peak: f32,
    area: u32,
}

impl Moments {
    fn add(&mut self, cx: u32, cy: u32, value: f32) {
        let v = value as f64;
        self.flux += v;
        self.sx += cx as f64 * v;
        self.sy += cy as f64 * v;
        self.sxx += cx as f64 * cx as f64 * v;
        self.syy += cy as f64 * cy as f64 * v;
        self.sxy += cx as f64 * cy as f64 * v;
        self.peak = self.peak.max(value);
        self.area += 1;
    }

    /// The component as a star, unless its area, place or shape rules it
    /// out.
    fn star(&self, width: u32, height: u32, config: &DetectConfig) -> Option<DetectedStar> {
        let Self {
            flux,
            sx,
            sy,
            sxx,
            syy,
            sxy,
            peak,
            area,
        } = *self;
        if area < config.min_area || area > config.max_area || flux <= 0.0 {
            return None;
        }
        let (cx, cy) = (sx / flux, sy / flux);
        let border = config.ignore_border as f64;
        if cx < border || cy < border || cx >= width as f64 - border || cy >= height as f64 - border
        {
            return None;
        }
        (elongation(
            sxx / flux - cx * cx,
            syy / flux - cy * cy,
            sxy / flux - cx * cy,
        ) <= config.max_elongation as f64)
            .then_some(DetectedStar {
                x: cx,
                y: cy,
                flux,
                peak,
                area,
            })
    }
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
            let median = median_in_place(&mut tile).unwrap_or(0.0);
            let sigma = robust_sigma_in_place(&mut tile, median).unwrap_or(0.0);
            // floor avoids a zero threshold on perfectly flat synthetic tiles
            (median, sigma.max(1e-6))
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
            // Four histograms, summed at the end: runs of equal pixels, the
            // usual case in a flat sky, then increment different counters
            // instead of each waiting on the last.
            let mut lanes = [[0u32; 256]; 4];
            for y in y0..y1 {
                let row = (y * width) as usize;
                let values = &pixels[row + x0 as usize..row + x1 as usize];
                let (quads, rest) = values.as_chunks::<4>();
                for quad in quads {
                    lanes[0][quad[0] as usize] += 1;
                    lanes[1][quad[1] as usize] += 1;
                    lanes[2][quad[2] as usize] += 1;
                    lanes[3][quad[3] as usize] += 1;
                }
                for &value in rest {
                    lanes[0][value as usize] += 1;
                }
            }
            let histogram: [u32; 256] =
                std::array::from_fn(|bin| lanes.iter().map(|lane| lane[bin]).sum());
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

/// The largest 8-bit excess that does not clear `threshold`: a whole
/// number clears it exactly when it exceeds its floor, so the comparison
/// stays in `u8`, which the compiler runs many pixels at a time. A negative
/// threshold passes every excess, but an excess of 0 is written as 0 either
/// way; one of 255 or more, or NaN, passes none.
fn u8_threshold_floor(threshold: f32) -> u8 {
    if threshold.is_nan() || threshold >= 255.0 {
        u8::MAX
    } else {
        threshold.max(0.0).floor() as u8
    }
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
                let floor = u8_threshold_floor(sigma * noise[tile]);
                let background = background[tile];
                for (value, excess) in input[x0..x1].iter().zip(&mut output[x0..x1]) {
                    let difference = value.saturating_sub(background);
                    *excess = if difference > floor { difference } else { 0 };
                }
            }
        });
    excess
}

/// `max(pixel - background, 0)` where the excess clears the sigma
/// threshold, else 0. Rows run in parallel; each row's tile segments go
/// through [`threshold_row`].
fn threshold_excess(
    pixels: &[f32],
    width: u32,
    height: u32,
    tile_size: u32,
    background: &[f32],
    noise: &[f32],
    sigma: f32,
) -> Vec<f32> {
    use rayon::prelude::*;

    let tiles_x = width.div_ceil(tile_size);
    let mut excess = vec![0.0f32; pixels.len()];
    excess
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, output)| {
            let ty = y as u32 / tile_size;
            let tiles = (ty * tiles_x) as usize..((ty + 1) * tiles_x) as usize;
            threshold_row(
                &pixels[y * width as usize..(y + 1) * width as usize],
                output,
                tile_size as usize,
                &background[tiles.clone()],
                &noise[tiles],
                sigma,
            );
        });
    debug_assert_eq!(excess.len(), width as usize * height as usize);
    excess
}

/// One row of [`threshold_excess`]: the background and threshold are
/// constant within each `tile_size` segment.
///
/// Plain loops the compiler vectorizes for each dispatched target. The
/// multiversioned function must do the work itself: a closure it hands to
/// rayon is compiled once without the clone's target features, and `wide`
/// picks its SIMD width when the crate is built, not at run time. Baseline
/// x86-64 (SSE2) already vectorizes the compare-and-select 4 wide, so only
/// AVX2 earns a clone; on aarch64 NEON is the baseline and the default
/// build uses it.
#[multiversion::multiversion(targets("x86_64+avx2"))]
fn threshold_row(
    input: &[f32],
    output: &mut [f32],
    tile_size: usize,
    background: &[f32],
    noise: &[f32],
    sigma: f32,
) {
    for (((seg, out), &bg), &noise) in input
        .chunks(tile_size)
        .zip(output.chunks_mut(tile_size))
        .zip(background)
        .zip(noise)
    {
        let threshold = sigma * noise;
        for (out, &value) in out.iter_mut().zip(seg) {
            let excess = value - bg;
            // NaN fails the comparison and becomes 0.
            *out = if excess > threshold { excess } else { 0.0 };
        }
    }
}

/// Ratio of the principal axes of the flux distribution (≥ 1).
fn elongation(mxx: f64, myy: f64, mxy: f64) -> f64 {
    let trace = mxx + myy;
    let root = ((mxx - myy).powi(2) + 4.0 * mxy * mxy).sqrt();
    let l1 = ((trace + root) / 2.0).max(0.0);
    let l2 = ((trace - root) / 2.0).max(1e-12);
    (l1 / l2).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_rows_match_a_scalar_reference_on_every_dispatch() {
        // A ragged last tile, NaNs and values on the threshold.
        let (width, height, tile) = (37u32, 5u32, 8u32);
        let tiles_x = width.div_ceil(tile);
        let tiles = (tiles_x * height.div_ceil(tile)) as usize;
        let background = (0..tiles).map(|t| 10.0 + t as f32).collect::<Vec<_>>();
        let noise = (0..tiles).map(|t| 1.0 + 0.5 * t as f32).collect::<Vec<_>>();
        let pixels = (0..width * height)
            .map(|i| match i % 11 {
                3 => f32::NAN,
                _ => (i * 7 % 23) as f32,
            })
            .collect::<Vec<_>>();
        let sigma = 1.5;
        let got = threshold_excess(&pixels, width, height, tile, &background, &noise, sigma);
        for (i, (&value, &out)) in pixels.iter().zip(&got).enumerate() {
            let (x, y) = (i as u32 % width, i as u32 / width);
            let t = ((y / tile) * tiles_x + x / tile) as usize;
            let excess = value - background[t];
            let expected = if excess > sigma * noise[t] {
                excess
            } else {
                0.0
            };
            assert_eq!(out.to_bits(), expected.to_bits(), "pixel {i}");
        }
    }

    /// Extracting stars from an excess image's positive pixels alone gives
    /// what extracting them from the whole image does, to the bit: stars of
    /// every size touching each other and the edges, single hot pixels,
    /// long trails, and a cap on the stars kept.
    #[test]
    fn stars_from_positive_excess_match_the_whole_image() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for (width, height) in [(1, 1), (7, 3), (64, 48), (301, 207)] {
            for round in 0..20 {
                let mut excess = vec![0.0f32; (width * height) as usize];
                for _ in 0..(width * height / 40).max(1) {
                    let (cx, cy) = (next() % u64::from(width), next() % u64::from(height));
                    let radius = (next() % 5) as i64;
                    for dy in -radius..=radius {
                        for dx in -radius..=radius {
                            let (x, y) = (cx as i64 + dx, cy as i64 + dy);
                            if x >= 0 && y >= 0 && x < i64::from(width) && y < i64::from(height) {
                                excess[(y as u32 * width + x as u32) as usize] =
                                    ((next() >> 40) as f32 + 1.0) / (1 << 20) as f32;
                            }
                        }
                    }
                }
                if round % 4 == 0 {
                    // A trail across the frame.
                    for x in 0..width {
                        excess
                            [((x * height / width.max(1)).min(height - 1) * width + x) as usize] =
                            0.5;
                    }
                }
                let positives: Vec<(u32, f32)> = excess
                    .iter()
                    .enumerate()
                    .filter(|(_, value)| **value > 0.0)
                    .map(|(index, &value)| (index as u32, value))
                    .collect();
                let config = DetectConfig {
                    min_area: 1 + round % 3,
                    max_area: 5 + round * 7,
                    ignore_border: round % 2,
                    max_stars: 3 + round as usize * 5,
                    ..DetectConfig::default()
                };
                let whole = extract_stars(excess, width, height, &config);
                let sparse = detect_stars_from_excess(width, height, &positives, &config);
                assert_eq!(whole.len(), sparse.len(), "{width}x{height} round {round}");
                for (a, b) in whole.iter().zip(&sparse) {
                    assert_eq!(
                        (
                            a.x.to_bits(),
                            a.y.to_bits(),
                            a.flux.to_bits(),
                            a.peak.to_bits(),
                            a.area
                        ),
                        (
                            b.x.to_bits(),
                            b.y.to_bits(),
                            b.flux.to_bits(),
                            b.peak.to_bits(),
                            b.area
                        ),
                    );
                }
            }
        }
    }

    /// A Gaussian star of `sigma` pixels at `truth` with `peak` counts, on
    /// a sky of 1000 rising `slope` counts a pixel to the right.
    fn star_on_a_gradient(
        width: usize,
        height: usize,
        truth: (f64, f64),
        sigma: f64,
        peak: f64,
        slope: f64,
    ) -> Vec<f32> {
        let mut pixels = vec![0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let d2 = (x as f64 - truth.0).powi(2) + (y as f64 - truth.1).powi(2);
                let star = peak * (-d2 / (2.0 * sigma * sigma)).exp();
                pixels[y * width + x] = (1000.0 + slope * x as f64 + star) as f32;
            }
        }
        pixels
    }

    fn seeded(x: f64, y: f64) -> DetectedStar {
        DetectedStar {
            x,
            y,
            flux: 1.0,
            peak: 1.0,
            area: 30,
        }
    }

    #[test]
    fn refined_centroids_find_a_star_on_a_sky_gradient() {
        let (width, height) = (96, 80);
        let truth = (47.3, 38.6);
        // A gradient of 4 counts a pixel against a 2,000-count star: the
        // background changes by a tenth of the peak across the star.
        let pixels = star_on_a_gradient(width, height, truth, 1.5, 2000.0, 4.0);
        let mut stars = [seeded(truth.0 + 0.7, truth.1 - 0.6)];
        refine_centroids(&mut stars, width, height, 1, &|x, y| pixels[y * width + x]);
        let off = (stars[0].x - truth.0).hypot(stars[0].y - truth.1);
        assert!(off < 0.03, "{off} px from the truth: {:?}", stars[0]);
    }

    #[test]
    fn refined_centroids_on_binned_samples_return_full_resolution_positions() {
        let (width, height) = (192, 160);
        let truth = (95.8, 77.2);
        let pixels = star_on_a_gradient(width, height, truth, 3.0, 2000.0, 2.0);
        // 2 x 2 sums, as a raw colour mosaic is read.
        let binned = |x: usize, y: usize| {
            let at = |dx: usize, dy: usize| pixels[(2 * y + dy) * width + 2 * x + dx];
            at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)
        };
        let mut stars = [seeded(truth.0 - 0.9, truth.1 + 0.8)];
        stars[0].area = 120;
        refine_centroids(&mut stars, width / 2, height / 2, 2, &binned);
        let off = (stars[0].x - truth.0).hypot(stars[0].y - truth.1);
        assert!(off < 0.05, "{off} px from the truth: {:?}", stars[0]);
    }

    #[test]
    fn a_star_that_cannot_be_measured_keeps_its_position() {
        let (width, height) = (64, 64);
        let flat = vec![500f32; width * height];
        let mut stars = [seeded(30.4, 31.7)];
        refine_centroids(&mut stars, width, height, 1, &|x, y| flat[y * width + x]);
        assert_eq!((stars[0].x, stars[0].y), (30.4, 31.7));
        // Too close to the edge for an annulus.
        let mut edge = [seeded(1.0, 1.0)];
        refine_centroids(&mut edge, width, height, 1, &|x, y| flat[y * width + x]);
        assert_eq!((edge[0].x, edge[0].y), (1.0, 1.0));
    }

    #[test]
    fn the_u8_threshold_floor_passes_exactly_the_excesses_f32_would() {
        let mut thresholds = vec![f32::NAN, f32::NEG_INFINITY, f32::INFINITY, -0.0, 0.0];
        thresholds.extend((-40..=1100).map(|step| step as f32 * 0.25));
        thresholds.extend([0.999_999, 1.000_001, 254.999_98, 255.000_02, 1e9]);
        for threshold in thresholds {
            let floor = u8_threshold_floor(threshold);
            for difference in 0..=u8::MAX {
                let f32_excess = if difference as f32 > threshold {
                    difference
                } else {
                    0
                };
                let u8_excess = if difference > floor { difference } else { 0 };
                assert_eq!(u8_excess, f32_excess, "{difference} against {threshold}");
            }
        }
    }

    #[test]
    fn u8_background_maps_match_a_single_histogram() {
        // Tiles whose widths are not multiples of four, a ragged last tile
        // each way, and a flat region.
        let (width, height, tile) = (71u32, 45u32, 17u32);
        let mut state = 0x2545_f491u32;
        let pixels: Vec<u8> = (0..width * height)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                if i % width < 20 {
                    40
                } else {
                    (state % 90) as u8 + 20
                }
            })
            .collect();
        let (background, noise) = estimate_background_u8(&pixels, width, height, tile);
        let tiles_x = width.div_ceil(tile);
        for t in 0..background.len() as u32 {
            let (x0, y0) = ((t % tiles_x) * tile, (t / tiles_x) * tile);
            let (x1, y1) = ((x0 + tile).min(width), (y0 + tile).min(height));
            let mut histogram = [0u32; 256];
            for y in y0..y1 {
                for x in x0..x1 {
                    histogram[pixels[(y * width + x) as usize] as usize] += 1;
                }
            }
            let target = ((x1 - x0) * (y1 - y0)) as usize / 2 + 1;
            let mut cumulative = 0usize;
            let median = histogram
                .iter()
                .position(|&bin| {
                    cumulative += bin as usize;
                    cumulative >= target
                })
                .unwrap();
            let mut inside = histogram[median] as usize;
            let mut mad = 0usize;
            while inside < target && mad < 255 {
                mad += 1;
                if median + mad < 256 {
                    inside += histogram[median + mad] as usize;
                }
                if mad <= median {
                    inside += histogram[median - mad] as usize;
                }
            }
            assert_eq!(background[t as usize], median as u8, "tile {t}");
            assert_eq!(
                noise[t as usize].to_bits(),
                (1.4826 * mad as f32).max(255.0e-6).to_bits(),
                "tile {t}"
            );
        }
    }
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
