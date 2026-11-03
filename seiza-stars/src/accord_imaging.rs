//! The Accord.NET (AForge) imaging filters N.I.N.A.'s star detector runs,
//! ported from the copy of Accord.Imaging in the N.I.N.A. repository.
//!
//! Each filter follows its C# source step by step, quirks included: the
//! integer Gaussian kernel, Canny's thresholds taken on gradients scaled to
//! the image's strongest edge, bicubic resizing that samples without
//! smoothing first. N.I.N.A.'s star counts depend on all of them, so a
//! "better" filter here gives counts N.I.N.A. never recorded.

/// Detection utility functions
pub struct DetectionUtility;

impl DetectionUtility {
    /// N.I.N.A.'s `DetectionUtility.ResizeForDetection`: an image wider than
    /// `max_width` is resized to `floor(width * resize_factor)` by
    /// `floor(height * resize_factor)` with Accord's `ResizeBicubic`; a
    /// narrower one is returned unchanged.
    pub fn resize_for_detection(
        image: &[u8],
        width: usize,
        height: usize,
        max_width: usize,
        resize_factor: f64,
    ) -> (Vec<u8>, usize, usize) {
        if width <= max_width {
            return (image.to_vec(), width, height);
        }
        let new_width = (width as f64 * resize_factor).floor() as usize;
        let new_height = (height as f64 * resize_factor).floor() as usize;
        let resized = resize_bicubic(image, width, height, new_width, new_height);
        (resized, new_width, new_height)
    }

    /// N.I.N.A.'s `DetectionUtility.InROI`: whether `blob`, in an image of
    /// `width` by `height`, lies inside the outer crop and not wholly inside
    /// the inner one. A ratio of 1 turns its crop off.
    pub fn in_roi(
        width: usize,
        height: usize,
        blob: &Rectangle,
        outer_crop_ratio: f64,
        inner_crop_ratio: f64,
    ) -> bool {
        let (width, height) = (width as i32, height as i32);
        let crop = |ratio: f64| {
            let start = (1.0 - ratio) / 2.0;
            Rectangle {
                x: (width as f64 * start).floor() as i32,
                y: (height as f64 * start).floor() as i32,
                width: (width as f64 * ratio) as i32,
                height: (height as f64 * ratio) as i32,
            }
        };
        let outside = if inner_crop_ratio >= 1.0 || outer_crop_ratio <= 0.0 {
            Rectangle {
                x: 0,
                y: 0,
                width,
                height,
            }
        } else if outer_crop_ratio >= 1.0 {
            crop(inner_crop_ratio)
        } else {
            crop(outer_crop_ratio)
        };
        let inside = if outer_crop_ratio >= 1.0 {
            Rectangle {
                x: width / 2,
                y: height / 2,
                width: 0,
                height: 0,
            }
        } else {
            crop(inner_crop_ratio)
        };
        fully_inside(blob, &outside) && !fully_inside(blob, &inside)
    }
}

fn fully_inside(lhs: &Rectangle, rhs: &Rectangle) -> bool {
    let (right, bottom) = (rhs.x + rhs.width, rhs.y + rhs.height);
    if lhs.x < rhs.x || lhs.y < rhs.y || lhs.x >= right || lhs.y >= bottom {
        return false;
    }
    lhs.x + lhs.width <= right && lhs.y + lhs.height <= bottom
}

/// Accord's `ResizeBicubic` for 8-bit grayscale.
///
/// Each output pixel samples the 4 × 4 source pixels around
/// `(x * width / new_width - 0.5, y * height / new_height - 0.5)` with the
/// a = −0.5 bicubic kernel, so a large reduction skips most source pixels
/// rather than averaging them. Values are clamped and truncated.
pub fn resize_bicubic(
    image: &[u8],
    width: usize,
    height: usize,
    new_width: usize,
    new_height: usize,
) -> Vec<u8> {
    use rayon::prelude::*;

    let mut out = vec![0u8; new_width * new_height];
    if width == 0 || height == 0 || new_width == 0 {
        return out;
    }
    let x_factor = width as f64 / new_width as f64;
    let y_factor = height as f64 / new_height as f64;
    let (xmax, ymax) = (width as i64 - 1, height as i64 - 1);
    // Each column's source pixels and weights, the same on every row.
    let columns = (0..new_width)
        .map(|x| {
            let ox = x as f64 * x_factor - 0.5;
            let ox1 = ox as i64;
            let dx = ox - ox1 as f64;
            let mut taps = [(0usize, 0.0f64); 4];
            for (tap, m) in taps.iter_mut().zip(-1i64..3) {
                *tap = (
                    (ox1 + m).clamp(0, xmax) as usize,
                    bicubic_kernel(m as f64 - dx),
                );
            }
            taps
        })
        .collect::<Vec<_>>();
    out.par_chunks_mut(new_width)
        .enumerate()
        .for_each(|(y, row)| {
            let oy = y as f64 * y_factor - 0.5;
            let oy1 = oy as i64;
            let dy = oy - oy1 as f64;
            let mut rows = [(0usize, 0.0f64); 4];
            for (tap, n) in rows.iter_mut().zip(-1i64..3) {
                *tap = (
                    (oy1 + n).clamp(0, ymax) as usize,
                    bicubic_kernel(dy - n as f64),
                );
            }
            for (pixel, taps) in row.iter_mut().zip(&columns) {
                let mut g = 0.0f64;
                for &(sy, k1) in &rows {
                    let line = &image[sy * width..(sy + 1) * width];
                    for &(sx, kx) in taps {
                        g += k1 * kx * line[sx] as f64;
                    }
                }
                *pixel = g.clamp(0.0, 255.0) as u8;
            }
        });
    out
}

/// Accord's `Interpolation.BiCubicKernel`.
fn bicubic_kernel(x: f64) -> f64 {
    let x = x.abs();
    if x <= 1.0 {
        (1.5 * x - 2.5) * x * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

/// Accord's `GaussianBlur` for 8-bit grayscale; [`CannyEdgeDetector`] uses
/// it with σ = 1.4 and a 5 × 5 kernel.
///
/// The kernel is the sampled Gaussian divided by its corner value and
/// truncated to integers. Near the image edges only the kernel taps inside
/// the image count, and the sum is divided by their total.
pub fn gaussian_blur(
    image: &[u8],
    width: usize,
    height: usize,
    sigma: f64,
    size: usize,
) -> Vec<u8> {
    let sigma = sigma.clamp(0.5, 5.0);
    let size = (size | 1).clamp(3, 21);
    let radius = (size / 2) as i64;
    let sigma_squared = sigma * sigma;
    let gaussian = |x: i64, y: i64| {
        (-((x * x + y * y) as f64) / (2.0 * sigma_squared)).exp()
            / (2.0 * std::f64::consts::PI * sigma_squared)
    };
    let corner = gaussian(-radius, -radius);
    let mut kernel = vec![0i64; size * size];
    let mut divisor = 0i64;
    for i in 0..size {
        for j in 0..size {
            let value = (gaussian(j as i64 - radius, i as i64 - radius) / corner).min(65535.0);
            kernel[i * size + j] = value as i64;
            divisor += value as i64;
        }
    }
    convolve(image, width, height, &kernel, size, divisor.max(1))
}

/// Accord's `Convolution` for 8-bit grayscale with a dynamic divisor at the
/// edges.
fn convolve(
    image: &[u8],
    width: usize,
    height: usize,
    kernel: &[i64],
    size: usize,
    divisor: i64,
) -> Vec<u8> {
    use rayon::prelude::*;

    let radius = (size / 2) as i64;
    let (w, h) = (width as i64, height as i64);
    let mut out = vec![0u8; width * height];
    if width == 0 {
        return out;
    }
    out.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let y = y as i64;
        for (x, pixel) in row.iter_mut().enumerate() {
            let x = x as i64;
            let (mut g, mut div, mut taps) = (0i64, 0i64, 0usize);
            for i in 0..size as i64 {
                let t = y + i - radius;
                if t < 0 {
                    continue;
                }
                if t >= h {
                    break;
                }
                for j in 0..size as i64 {
                    let s = x + j - radius;
                    if s < 0 || s >= w {
                        continue;
                    }
                    let k = kernel[i as usize * size + j as usize];
                    div += k;
                    g += k * image[(t * w + s) as usize] as i64;
                    taps += 1;
                }
            }
            if taps == size * size {
                div = divisor;
            }
            if div != 0 {
                g /= div;
            }
            *pixel = g.clamp(0, 255) as u8;
        }
    });
    out
}

/// Accord's `CannyEdgeDetector`, or N.I.N.A.'s `NoBlurCannyEdgeDetector`
/// when `blur` is off.
///
/// Unlike the textbook detector, edge strengths are scaled so the image's
/// strongest gradient is 255 before the thresholds apply, hysteresis looks
/// only at the 8 neighbours, and surviving edges keep their scaled strength
/// rather than becoming 255.
#[derive(Debug, Clone, Copy)]
pub struct CannyEdgeDetector {
    /// Scaled strengths below this are never edges.
    pub low_threshold: u8,
    /// Scaled strengths at or above this are always edges; those between the
    /// thresholds are edges only beside one.
    pub high_threshold: u8,
    /// Blur with Accord's 5 × 5, σ = 1.4 Gaussian first.
    pub blur: bool,
}

impl CannyEdgeDetector {
    /// Find edges in `image` in place, as Accord's `ApplyInPlace` does.
    pub fn apply_in_place(&self, image: &mut [u8], width: usize, height: usize) {
        if width < 3 || height < 3 {
            image.fill(0);
            return;
        }
        let blurred;
        let source: &[u8] = if self.blur {
            blurred = gaussian_blur(image, width, height, 1.4, 5);
            &blurred
        } else {
            image
        };

        // Gradients and their directions on the interior; the border keeps
        // a gradient of zero.
        let mut gradients = vec![0f32; width * height];
        let mut orients = vec![0u8; width * height];
        let mut max_gradient = f32::NEG_INFINITY;
        let to_angle = 180.0 / std::f64::consts::PI;
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let p = y * width + x;
                let (up, down) = (p - width, p + width);
                let at = |q: usize| source[q] as i32;
                let gx = at(up + 1) + at(down + 1) - at(up - 1) - at(down - 1)
                    + 2 * (at(p + 1) - at(p - 1));
                let gy =
                    at(up - 1) + at(up + 1) - at(down - 1) - at(down + 1) + 2 * (at(up) - at(down));
                let gradient = ((gx * gx + gy * gy) as f64).sqrt() as f32;
                gradients[p] = gradient;
                if gradient > max_gradient {
                    max_gradient = gradient;
                }
                orients[p] = if gx == 0 {
                    if gy == 0 { 0 } else { 90 }
                } else {
                    let div = gy as f64 / gx as f64;
                    let orientation = if div < 0.0 {
                        180.0 - (-div).atan() * to_angle
                    } else {
                        div.atan() * to_angle
                    };
                    if orientation < 22.5 {
                        0
                    } else if orientation < 67.5 {
                        45
                    } else if orientation < 112.5 {
                        90
                    } else if orientation < 157.5 {
                        135
                    } else {
                        0
                    }
                };
            }
        }

        // Non-maximum suppression. In place, Accord writes the interior over
        // the source and draws the border black only at the end, so
        // hysteresis still sees the source's border pixels.
        let mut out = image.to_vec();
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let p = y * width + x;
                let (left, right) = match orients[p] {
                    0 => (gradients[p - 1], gradients[p + 1]),
                    45 => (gradients[p + width - 1], gradients[p - width + 1]),
                    90 => (gradients[p + width], gradients[p - width]),
                    _ => (gradients[p + width + 1], gradients[p - width - 1]),
                };
                out[p] = if gradients[p] < left || gradients[p] < right {
                    0
                } else {
                    (gradients[p] / max_gradient * 255.0) as u8
                };
            }
        }

        // Hysteresis against the 8 neighbours. Only pixels below the high
        // threshold change, so the order of the scan does not matter.
        let (low, high) = (self.low_threshold, self.high_threshold);
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let p = y * width + x;
                if out[p] >= high {
                    continue;
                }
                if out[p] < low {
                    out[p] = 0;
                    continue;
                }
                let neighbours = [
                    p - 1,
                    p + 1,
                    p - width - 1,
                    p - width,
                    p - width + 1,
                    p + width - 1,
                    p + width,
                    p + width + 1,
                ];
                if neighbours.iter().all(|&q| out[q] < high) {
                    out[p] = 0;
                }
            }
        }

        for x in 0..width {
            out[x] = 0;
            out[(height - 1) * width + x] = 0;
        }
        for y in 0..height {
            out[y * width] = 0;
            out[y * width + width - 1] = 0;
        }
        image.copy_from_slice(&out);
    }
}

/// Accord's `SISThreshold` in place: the threshold is the mean of the
/// interior pixels weighted by their larger central difference, and pixels
/// at or above it become 255, the rest 0. Returns the threshold.
pub fn sis_threshold(image: &mut [u8], width: usize, height: usize) -> u8 {
    let (mut weight_total, mut total) = (0.0f64, 0.0f64);
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let p = y * width + x;
            let ex = (image[p + 1] as f64 - image[p - 1] as f64).abs();
            let ey = (image[p + width] as f64 - image[p - width] as f64).abs();
            let weight = if ex > ey { ex } else { ey };
            weight_total += weight;
            total += weight * image[p] as f64;
        }
    }
    let threshold = if weight_total == 0.0 {
        0
    } else {
        (total / weight_total) as u8
    };
    for pixel in image.iter_mut() {
        *pixel = if *pixel >= threshold { 255 } else { 0 };
    }
    threshold
}

/// Accord's `BinaryDilation3x3`: each pixel becomes the OR of its 3 × 3
/// neighbourhood, clipped at the image edges.
pub fn binary_dilation_3x3(image: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![0u8; width * height];
    for y in 0..height {
        let (top, bottom) = (y.saturating_sub(1), (y + 1).min(height - 1));
        for x in 0..width {
            let (left, right) = (x.saturating_sub(1), (x + 1).min(width - 1));
            let mut value = 0u8;
            for row in top..=bottom {
                for column in left..=right {
                    value |= image[row * width + column];
                }
            }
            out[y * width + x] = value;
        }
    }
    out
}

/// Accord's `Median` with its default 3 × 3 window: near the edges the
/// median is taken over the window pixels inside the image, choosing the
/// upper middle of an even count.
pub fn median_3x3(image: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![0u8; width * height];
    let mut window = [0u8; 9];
    for y in 0..height {
        for x in 0..width {
            let mut count = 0;
            for row in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for column in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    window[count] = image[row * width + column];
                    count += 1;
                }
            }
            window[..count].sort_unstable();
            out[y * width + x] = window[count >> 1];
        }
    }
    out
}

/// N.I.N.A.'s `FastGaussianBlur.Process`: three box blurs approximating a
/// Gaussian of the given whole-pixel `sigma`.
///
/// The horizontal pass reproduces N.I.N.A.'s, which subtracts the pixel
/// leaving the window from its output rather than its input, so its sum
/// drifts upward after each bright pixel; values past 255 wrap as they do in
/// C#.
pub fn fast_gaussian_blur(image: &[u8], width: usize, height: usize, sigma: i32) -> Vec<u8> {
    let mut source = image.to_vec();
    let mut dest = vec![0u8; image.len()];
    if width == 0 || height == 0 {
        return dest;
    }
    let boxes = boxes_for_gauss(sigma, 3);
    box_blur(&mut source, &mut dest, width, height, (boxes[0] - 1) / 2);
    box_blur(&mut dest, &mut source, width, height, (boxes[1] - 1) / 2);
    box_blur(&mut source, &mut dest, width, height, (boxes[2] - 1) / 2);
    dest
}

fn boxes_for_gauss(sigma: i32, n: i32) -> [i32; 3] {
    let ideal = ((12 * sigma * sigma / n) as f64 + 1.0).sqrt();
    let mut lower = ideal.floor() as i32;
    if lower % 2 == 0 {
        lower -= 1;
    }
    let upper = lower + 2;
    let m_ideal = (12 * sigma * sigma - n * lower * lower - 4 * n * lower - 3 * n) as f64
        / (-4 * lower - 4) as f64;
    let m = m_ideal.round_ties_even();
    let mut sizes = [0; 3];
    for (i, size) in sizes.iter_mut().enumerate() {
        *size = if (i as f64) < m { lower } else { upper };
    }
    sizes
}

/// C#'s `(byte)Math.Floor(x)`: through a 32-bit integer, so a value outside
/// 0 to 255 wraps rather than saturating. N.I.N.A.'s horizontal pass drifts
/// above 255 beside bright stars, and the wrapped values are what it detects
/// on.
fn to_byte(x: f64) -> u8 {
    x.floor() as i32 as u8
}

/// `boxBlur_4(source, dest)`: copies `source` into `dest`, blurs rows from
/// `dest` into `source`, then columns from `source` into `dest`.
fn box_blur(source: &mut [u8], dest: &mut [u8], width: usize, height: usize, r: i32) {
    dest.copy_from_slice(source);
    box_blur_horizontal(dest, source, width, height, r as usize);
    box_blur_vertical(source, dest, width, height, r as usize);
}

fn box_blur_horizontal(source: &[u8], dest: &mut [u8], w: usize, h: usize, r: usize) {
    let iar = 1.0 / (r + r + 1) as f64;
    for i in 0..h {
        let mut ti = i * w;
        let mut li = ti;
        let mut ri = ti + r;
        let first = source[ti] as i32;
        let last = source[ti + w - 1] as i32;
        let mut value = (r as i32 + 1) * first;
        for j in 0..r {
            value += source[ti + j] as i32;
        }
        for _ in 0..=r {
            value += source[ri] as i32 - first;
            ri += 1;
            dest[ti] = to_byte(value as f64 * iar);
            ti += 1;
        }
        for _ in r + 1..w.saturating_sub(r) {
            value += source[ri] as i32 - dest[li] as i32;
            ri += 1;
            li += 1;
            dest[ti] = to_byte(value as f64 * iar);
            ti += 1;
        }
        for _ in w.saturating_sub(r)..w {
            value += last - source[li] as i32;
            li += 1;
            dest[ti] = to_byte(value as f64 * iar);
            ti += 1;
        }
    }
}

fn box_blur_vertical(source: &[u8], dest: &mut [u8], w: usize, h: usize, r: usize) {
    let iar = 1.0 / (r + r + 1) as f64;
    for i in 0..w {
        let mut ti = i;
        let mut li = ti;
        let mut ri = ti + r * w;
        let first = source[ti] as i32;
        let last = source[ti + w * (h - 1)] as i32;
        let mut value = (r as i32 + 1) * first;
        for j in 0..r {
            value += source[ti + j * w] as i32;
        }
        for _ in 0..=r {
            value += source[ri] as i32 - first;
            dest[ti] = to_byte(value as f64 * iar);
            ri += w;
            ti += w;
        }
        for _ in r + 1..h.saturating_sub(r) {
            value += source[ri] as i32 - source[li] as i32;
            dest[ti] = to_byte(value as f64 * iar);
            li += w;
            ri += w;
            ti += w;
        }
        for _ in h.saturating_sub(r)..h {
            value += last - source[li] as i32;
            dest[ti] = to_byte(value as f64 * iar);
            li += w;
            ti += w;
        }
    }
}

/// Blob representation
#[derive(Debug, Clone)]
pub struct Blob {
    pub rectangle: Rectangle,
}

#[derive(Debug, Clone, Copy)]
pub struct Rectangle {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

/// Accord's `BlobCounter`: 8-connected regions of the pixels above 0.
///
/// Blobs are listed in the order of their first pixel in a row-by-row scan.
#[derive(Default)]
pub struct BlobCounter {
    blobs: Vec<Blob>,
    /// Each pixel's blob, counting from 1; 0 is background.
    labels: Vec<u32>,
    width: usize,
}

impl BlobCounter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Label the connected regions of `image`.
    pub fn process_image(&mut self, image: &[u8], width: usize, height: usize) {
        fn root(parent: &mut [u32], mut label: u32) -> u32 {
            while parent[label as usize] != label {
                let next = parent[label as usize];
                parent[label as usize] = parent[next as usize];
                label = next;
            }
            label
        }

        self.blobs.clear();
        self.width = width;
        self.labels = vec![0u32; width * height];
        let mut parent: Vec<u32> = vec![0];
        for y in 0..height {
            for x in 0..width {
                let p = y * width + x;
                if image[p] == 0 {
                    continue;
                }
                let mut neighbours = [0u32; 4];
                if x > 0 {
                    neighbours[0] = self.labels[p - 1];
                }
                if y > 0 {
                    neighbours[1] = self.labels[p - width];
                    if x > 0 {
                        neighbours[2] = self.labels[p - width - 1];
                    }
                    if x + 1 < width {
                        neighbours[3] = self.labels[p - width + 1];
                    }
                }
                let mut label = 0u32;
                for &neighbour in neighbours.iter().filter(|&&n| n != 0) {
                    let other = root(&mut parent, neighbour);
                    if label == 0 {
                        label = other;
                    } else if other != label {
                        let (keep, merge) = (label.min(other), label.max(other));
                        parent[merge as usize] = keep;
                        label = keep;
                    }
                }
                if label == 0 {
                    label = parent.len() as u32;
                    parent.push(label);
                }
                self.labels[p] = label;
            }
        }

        // Number the regions in scan order of their first pixel and collect
        // each one's bounding box.
        let mut number = vec![0u32; parent.len()];
        let mut boxes: Vec<(i32, i32, i32, i32)> = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let p = y * width + x;
                if self.labels[p] == 0 {
                    continue;
                }
                let r = root(&mut parent, self.labels[p]) as usize;
                if number[r] == 0 {
                    boxes.push((x as i32, y as i32, x as i32, y as i32));
                    number[r] = boxes.len() as u32;
                }
                let id = number[r];
                self.labels[p] = id;
                let b = &mut boxes[id as usize - 1];
                b.0 = b.0.min(x as i32);
                b.2 = b.2.max(x as i32);
                b.3 = y as i32;
            }
        }
        self.blobs = boxes
            .into_iter()
            .map(|(min_x, min_y, max_x, max_y)| Blob {
                rectangle: Rectangle {
                    x: min_x,
                    y: min_y,
                    width: max_x - min_x + 1,
                    height: max_y - min_y + 1,
                },
            })
            .collect();
    }

    pub fn get_objects_information(&self) -> Vec<Blob> {
        self.blobs.clone()
    }

    /// Accord's `GetBlobsEdgePoints` for the blob at `index` in
    /// [`Self::get_objects_information`]: the first and last pixel of the
    /// blob in each row, then the first and last in each column, leaving out
    /// points the rows already gave.
    pub fn edge_points(&self, index: usize) -> Vec<(i32, i32)> {
        let Some(blob) = self.blobs.get(index) else {
            return Vec::new();
        };
        let label = index as u32 + 1;
        let r = blob.rectangle;
        let (xmin, ymin) = (r.x, r.y);
        let (xmax, ymax) = (r.x + r.width - 1, r.y + r.height - 1);
        let at = |x: i32, y: i32| self.labels[y as usize * self.width + x as usize] == label;
        let mut points = Vec::new();
        // Accord starts these at 0, which a blob in column 0 can match; that
        // is kept.
        let mut left = vec![0i32; r.height as usize];
        let mut right = vec![0i32; r.height as usize];
        for y in ymin..=ymax {
            let row = (y - ymin) as usize;
            if let Some(x) = (xmin..=xmax).find(|&x| at(x, y)) {
                points.push((x, y));
                left[row] = x;
            }
            if let Some(x) = (xmin..=xmax).rev().find(|&x| at(x, y)) {
                if left[row] != x {
                    points.push((x, y));
                }
                right[row] = x;
            }
        }
        for x in xmin..=xmax {
            let known = |y: i32| left[(y - ymin) as usize] == x || right[(y - ymin) as usize] == x;
            if let Some(y) = (ymin..=ymax).find(|&y| at(x, y))
                && !known(y)
            {
                points.push((x, y));
            }
            if let Some(y) = (ymin..=ymax).rev().find(|&y| at(x, y))
                && !known(y)
            {
                points.push((x, y));
            }
        }
        points
    }
}

/// Accord's `SimpleShapeChecker`, with its default distortion limits.
pub struct SimpleShapeChecker;

impl SimpleShapeChecker {
    /// Accord's `IsCircle(edgePoints, out center, out radius)`.
    ///
    /// The centre is the middle of the points' bounding box and the radius a
    /// quarter of its width plus height. The shape is a circle when the
    /// points' mean distance from that circle is at most
    /// `max(0.5, 0.03 * (width + height) / 2)`. The centre and radius are
    /// written whether or not the shape is a circle, and are zero for fewer
    /// than 8 points.
    pub fn is_circle(
        &self,
        points: &[(i32, i32)],
        center_x: &mut f32,
        center_y: &mut f32,
        radius: &mut f32,
    ) -> bool {
        if points.len() < 8 {
            *center_x = 0.0;
            *center_y = 0.0;
            *radius = 0.0;
            return false;
        }
        let (mut min_x, mut max_x) = (i32::MAX, i32::MIN);
        let (mut min_y, mut max_y) = (i32::MAX, i32::MIN);
        for &(x, y) in points {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        let (size_x, size_y) = (max_x - min_x, max_y - min_y);
        let cx = min_x as f32 + size_x as f32 / 2.0;
        let cy = min_y as f32 + size_y as f32 / 2.0;
        let r = (size_x as f32 + size_y as f32) / 4.0;
        let mut total = 0.0f32;
        for &(x, y) in points {
            let (dx, dy) = (cx - x as f32, cy - y as f32);
            let distance = ((dx * dx + dy * dy) as f64).sqrt() as f32;
            total += (distance - r).abs();
        }
        let mean_distance = total / points.len() as f32;
        let limit = 0.5f32.max((size_x as f32 + size_y as f32) / 2.0 * 0.03);
        *center_x = cx;
        *center_y = cy;
        *radius = r;
        mean_distance <= limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_counter_joins_diagonal_neighbours() {
        #[rustfmt::skip]
        let image = vec![
            0, 0, 0, 0, 0, 0, 0,
            0, 255, 255, 0, 0, 0, 0,
            0, 255, 255, 0, 0, 0, 0,
            0, 0, 0, 255, 255, 0, 0,
            0, 0, 0, 255, 255, 0, 0,
            0, 0, 0, 0, 0, 255, 0,
            0, 0, 0, 0, 0, 0, 0,
        ];
        let mut counter = BlobCounter::new();
        counter.process_image(&image, 7, 7);
        let blobs = counter.get_objects_information();
        assert_eq!(blobs.len(), 1);
        let r = blobs[0].rectangle;
        assert_eq!((r.x, r.y, r.width, r.height), (1, 1, 5, 5));
    }

    #[test]
    fn blob_counter_merges_u_shapes_and_lists_blobs_in_scan_order() {
        #[rustfmt::skip]
        let image = vec![
            255, 0, 255, 0, 0, 255,
            255, 0, 255, 0, 0, 0,
            255, 255, 255, 0, 0, 0,
            0, 0, 0, 0, 255, 0,
        ];
        let mut counter = BlobCounter::new();
        counter.process_image(&image, 6, 4);
        let rects = counter
            .get_objects_information()
            .iter()
            .map(|b| {
                let r = b.rectangle;
                (r.x, r.y, r.width, r.height)
            })
            .collect::<Vec<_>>();
        assert_eq!(rects, vec![(0, 0, 3, 3), (5, 0, 1, 1), (4, 3, 1, 1)]);
    }

    #[test]
    fn edge_points_and_circle_follow_accord() {
        // A 4 × 4 square gives each row's first and last pixel, then each
        // column's top and bottom where the rows did not.
        let mut image = vec![0u8; 36];
        for y in 1..5 {
            for x in 1..5 {
                image[y * 6 + x] = 255;
            }
        }
        let mut counter = BlobCounter::new();
        counter.process_image(&image, 6, 6);
        let points = counter.edge_points(0);
        assert_eq!(
            points,
            vec![
                (1, 1),
                (4, 1),
                (1, 2),
                (4, 2),
                (1, 3),
                (4, 3),
                (1, 4),
                (4, 4),
                (2, 1),
                (2, 4),
                (3, 1),
                (3, 4)
            ]
        );
        let (mut cx, mut cy, mut r) = (0.0, 0.0, 0.0);
        assert!(SimpleShapeChecker.is_circle(&points, &mut cx, &mut cy, &mut r));
        assert_eq!((cx, cy, r), (2.5, 2.5, 1.5));
    }

    #[test]
    fn shape_checker_needs_eight_points() {
        let (mut cx, mut cy, mut r) = (1.0, 1.0, 1.0);
        let points = [(0, 0), (1, 0), (0, 1), (1, 1)];
        assert!(!SimpleShapeChecker.is_circle(&points, &mut cx, &mut cy, &mut r));
        assert_eq!((cx, cy, r), (0.0, 0.0, 0.0));
    }

    #[test]
    fn gaussian_kernel_is_accords_integer_kernel() {
        // A lone pixel of 71 spreads into the kernel itself: Accord's
        // 5 × 5, σ = 1.4 kernel is [1 2 2 2 1; 2 4 5 4 2; 2 5 7 5 2; ...]
        // over 71.
        let mut image = vec![0u8; 81];
        image[40] = 71;
        let blurred = gaussian_blur(&image, 9, 9, 1.4, 5);
        let row = |y: usize| blurred[y * 9 + 2..y * 9 + 7].to_vec();
        assert_eq!(row(2), vec![1, 2, 2, 2, 1]);
        assert_eq!(row(3), vec![2, 4, 5, 4, 2]);
        assert_eq!(row(4), vec![2, 5, 7, 5, 2]);
    }

    #[test]
    fn resize_samples_without_smoothing() {
        // At an eighth of the size, output column 0 samples source columns
        // 0 to 2 and column 1 samples 6 to 9, so a bright source column 4
        // vanishes; an area-averaging resize would keep an eighth of it.
        let image = (0..64 * 16)
            .map(|i| if i % 64 == 4 { 255 } else { 0 })
            .collect::<Vec<u8>>();
        assert!(resize_bicubic(&image, 64, 16, 8, 2).iter().all(|&v| v == 0));
        let image = (0..64 * 16)
            .map(|i| if i % 64 == 7 { 255 } else { 0 })
            .collect::<Vec<u8>>();
        assert!(
            resize_bicubic(&image, 64, 16, 8, 2)
                .iter()
                .any(|&v| v > 100)
        );
    }

    #[test]
    fn sis_threshold_binarizes_at_the_weighted_mean() {
        // A lone bright pixel: its four neighbours carry all the weight and
        // are 0, so the threshold is 0 and every pixel passes, as in Accord.
        let mut image = vec![0u8; 25];
        image[12] = 100;
        assert_eq!(sis_threshold(&mut image, 5, 5), 0);
        assert!(image.iter().all(|&v| v == 255));

        let mut image = vec![0u8; 25];
        for (i, pixel) in image.iter_mut().enumerate() {
            *pixel = if i % 5 >= 2 { 200 } else { 10 };
        }
        // Weighted by the edge between columns 1 and 2: (10 + 200) / 2.
        assert_eq!(sis_threshold(&mut image, 5, 5), 105);
        assert_eq!(image[1], 0);
        assert_eq!(image[2], 255);
    }

    #[test]
    fn fast_gaussian_blur_box_sizes() {
        assert_eq!(boxes_for_gauss(1, 3), [1, 1, 3]);
        assert_eq!(boxes_for_gauss(2, 3), [3, 3, 5]);
        assert_eq!(boxes_for_gauss(3, 3), [5, 5, 7]);
    }
}
