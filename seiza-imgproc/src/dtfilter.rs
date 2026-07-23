//! Domain transform filter (Gastal & Oliveira 2011), normalized convolution
//! variant — the algorithm behind `cv::ximgproc::dtFilter` with `DTF_NC`.
//!
//! Edge-aware smoothing: distances between neighboring pixels are stretched
//! by the guide image's gradient, so averaging windows never cross strong
//! edges.

/// Apply the NC domain transform filter to a single-channel f32 image.
///
/// `guide` supplies the edge structure (pass `src` itself for self-guided
/// filtering). `sigma_spatial`/`sigma_color` control the spatial support and
/// the edge sensitivity; `num_iters` is the iteration count (OpenCV default
/// 3). Each iteration runs a horizontal then a vertical normalized box
/// filter in the transformed domain.
pub fn dt_filter_nc(
    guide: &[f32],
    src: &[f32],
    width: usize,
    height: usize,
    sigma_spatial: f64,
    sigma_color: f64,
    num_iters: usize,
) -> Vec<f32> {
    assert_eq!(guide.len(), width * height);
    assert_eq!(src.len(), width * height);
    assert!(num_iters >= 1);
    if width == 0 || height == 0 {
        return Vec::new();
    }

    let ratio = sigma_spatial / sigma_color;

    // Domain-transform positions (ct[0] = 0, ct[k] = ct[k-1] + 1 + ratio *
    // |g[k] - g[k-1]|) are computed per row/column on the fly: materializing
    // them for the whole image costs two full-resolution f64 planes of
    // allocation and memory traffic (about 1 GiB for a 61 MP frame) for
    // values each pass reads exactly once, in order.
    let mut res: Vec<f32> = src.to_vec();
    let n = num_iters as i32;
    for iter in 1..=n {
        // Per-iteration standard deviation from the paper (eq. 14).
        let sigma_h = sigma_spatial * (3.0f64).sqrt() * (2.0f64).powi(n - iter)
            / ((4.0f64).powi(n) - 1.0).sqrt();
        let radius = sigma_h * (3.0f64).sqrt();

        #[cfg(feature = "parallel")]
        if width * height >= crate::blur::PAR_MIN_PIXELS {
            horizontal_pass_par(&mut res, guide, ratio, width, height, radius);
            vertical_pass_par(&mut res, guide, ratio, width, height, radius);
            continue;
        }
        horizontal_pass(&mut res, guide, ratio, width, height, radius);
        vertical_pass(&mut res, guide, ratio, width, height, radius);
    }
    res
}

/// One row of the horizontal pass; shared by the serial and parallel
/// drivers so both run identical arithmetic.
#[inline(always)]
fn horizontal_row(
    res_row: &mut [f32],
    guide_row: &[f32],
    ratio: f64,
    width: usize,
    radius: f64,
    prefix: &mut [f32],
    ct_row: &mut [f64],
) {
    for x in 0..width {
        prefix[x + 1] = prefix[x] + res_row[x];
    }
    ct_row[0] = 0.0;
    for x in 1..width {
        let d = (guide_row[x] - guide_row[x - 1]).abs() as f64;
        ct_row[x] = ct_row[x - 1] + 1.0 + ratio * d;
    }
    let mut lo = 0usize;
    let mut hi = 0usize;
    for (x, r) in res_row.iter_mut().enumerate() {
        let left = ct_row[x] - radius;
        let right = ct_row[x] + radius;
        while ct_row[lo] < left {
            lo += 1;
        }
        if hi < x {
            hi = x;
        }
        while hi + 1 < width && ct_row[hi + 1] <= right {
            hi += 1;
        }
        let count = (hi - lo + 1) as f32;
        *r = (prefix[hi + 1] - prefix[lo]) / count;
    }
}

/// One column of the vertical pass, reading the strided column and writing
/// the filtered result into `ocol` (contiguous, transposed layout).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn vertical_column(
    res: &[f32],
    guide: &[f32],
    x: usize,
    ratio: f64,
    width: usize,
    height: usize,
    radius: f64,
    prefix: &mut [f32],
    ct_col: &mut [f64],
    ocol: &mut [f32],
) {
    ct_col[0] = 0.0;
    prefix[1] = prefix[0] + res[x];
    for y in 1..height {
        let d = (guide[y * width + x] - guide[(y - 1) * width + x]).abs() as f64;
        ct_col[y] = ct_col[y - 1] + 1.0 + ratio * d;
        prefix[y + 1] = prefix[y] + res[y * width + x];
    }
    let mut lo = 0usize;
    let mut hi = 0usize;
    for (y, o) in ocol.iter_mut().enumerate() {
        let left = ct_col[y] - radius;
        let right = ct_col[y] + radius;
        while ct_col[lo] < left {
            lo += 1;
        }
        if hi < y {
            hi = y;
        }
        while hi + 1 < height && ct_col[hi + 1] <= right {
            hi += 1;
        }
        let count = (hi - lo + 1) as f32;
        *o = (prefix[hi + 1] - prefix[lo]) / count;
    }
}

#[cfg(feature = "parallel")]
fn horizontal_pass_par(
    res: &mut [f32],
    guide: &[f32],
    ratio: f64,
    width: usize,
    _height: usize,
    radius: f64,
) {
    use rayon::prelude::*;
    res.par_chunks_exact_mut(width).enumerate().for_each_init(
        || (vec![0f32; width + 1], vec![0f64; width]),
        |(prefix, ct_row), (y, res_row)| {
            let guide_row = &guide[y * width..(y + 1) * width];
            horizontal_row(res_row, guide_row, ratio, width, radius, prefix, ct_row);
        },
    );
}

/// Column-parallel vertical pass: columns are computed independently into a
/// transposed buffer (contiguous per column), then transposed back — same
/// per-column arithmetic as the serial pass, without concurrent strided
/// writes into `res`.
#[cfg(feature = "parallel")]
fn vertical_pass_par(
    res: &mut [f32],
    guide: &[f32],
    ratio: f64,
    width: usize,
    height: usize,
    radius: f64,
) {
    use rayon::prelude::*;
    let mut transposed = vec![0f32; width * height];
    {
        let res_ref: &[f32] = res;
        transposed
            .par_chunks_exact_mut(height)
            .enumerate()
            .for_each_init(
                || (vec![0f32; height + 1], vec![0f64; height]),
                |(prefix, ct_col), (x, ocol)| {
                    vertical_column(
                        res_ref, guide, x, ratio, width, height, radius, prefix, ct_col, ocol,
                    );
                },
            );
    }
    res.par_chunks_exact_mut(width)
        .enumerate()
        .for_each(|(y, res_row)| {
            for (x, r) in res_row.iter_mut().enumerate() {
                *r = transposed[x * height + y];
            }
        });
}

fn horizontal_pass(
    res: &mut [f32],
    guide: &[f32],
    ratio: f64,
    width: usize,
    _height: usize,
    radius: f64,
) {
    // f32 prefix sums, matching OpenCV's box accumulation precision.
    let mut prefix = vec![0f32; width + 1];
    let mut ct_row = vec![0f64; width];
    for (y, res_row) in res.chunks_exact_mut(width).enumerate() {
        let guide_row = &guide[y * width..(y + 1) * width];
        horizontal_row(
            res_row,
            guide_row,
            ratio,
            width,
            radius,
            &mut prefix,
            &mut ct_row,
        );
    }
}

fn vertical_pass(
    res: &mut [f32],
    guide: &[f32],
    ratio: f64,
    width: usize,
    height: usize,
    radius: f64,
) {
    let mut ct_col = vec![0f64; height];
    let mut prefix = vec![0f32; height + 1];
    let mut ocol = vec![0f32; height];
    for x in 0..width {
        vertical_column(
            res,
            guide,
            x,
            ratio,
            width,
            height,
            radius,
            &mut prefix,
            &mut ct_col,
            &mut ocol,
        );
        for (y, &o) in ocol.iter().enumerate() {
            res[y * width + x] = o;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_image_unchanged() {
        let img = vec![5.0f32; 64];
        let out = dt_filter_nc(&img, &img, 8, 8, 10.0, 0.1, 1);
        for v in out {
            assert!((v - 5.0).abs() < 1e-6);
        }
    }

    #[test]
    fn smooths_noise_but_preserves_strong_edge() {
        // Two flat regions with small noise and a big step between them.
        let w = 32;
        let h = 4;
        let mut img = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let base = if x < 16 { 10.0 } else { 200.0 };
                let noise = if (x + y) % 2 == 0 { 0.5 } else { -0.5 };
                img[y * w + x] = base + noise;
            }
        }
        let guide = img.clone();
        let out = dt_filter_nc(&guide, &img, w, h, 8.0, 30.0, 3);
        // Noise reduced inside regions.
        assert!((out[w + 4] - 10.0).abs() < 0.4);
        assert!((out[w + 24] - 200.0).abs() < 0.4);
        // Step preserved: left side stays near 10, right near 200.
        assert!(out[w + 15] < 30.0);
        assert!(out[w + 16] > 180.0);
    }

    #[test]
    fn tiny_sigma_color_means_identity_on_noisy_data() {
        // With sigma_color far below the pixel differences, every window
        // collapses to the pixel itself.
        let w = 16;
        let img: Vec<f32> = (0..w * 4).map(|i| (i * 37 % 101) as f32 * 50.0).collect();
        let out = dt_filter_nc(&img, &img, w, 4, 10.0, 0.1, 1);
        for (a, b) in out.iter().zip(img.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
