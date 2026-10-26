use seiza_stacking::{LinearImage, RegistrationResult};
use std::collections::HashSet;
pub(crate) const PIXELS: usize = 256;

// Stratify whole non-overlapping 16x16 patches at two-pixel pitch over the
// central 80%, independent of image values. No preference for quiet sky.
pub(crate) fn patches(width: usize, height: usize, samples: usize) -> Vec<(usize, usize)> {
    let (mx, my) = (width / 10, height / 10);
    let (nx, ny) = ((width - 2 * mx) / 32, (height - 2 * my) / 32);
    let available = nx * ny;
    let count = available.min(samples / PIXELS);
    (0..count)
        .map(|i| {
            let cell = (2 * i + 1) * available / (2 * count);
            (mx + (cell % nx) * 32, my + (cell / nx) * 32)
        })
        .collect()
}
pub(crate) fn gather(
    image: &LinearImage,
    patches: &[(usize, usize)],
    registration: Option<&RegistrationResult>,
) -> (Vec<f32>, Vec<bool>) {
    let mut values = Vec::with_capacity(patches.len() * PIXELS);
    let mut valid = Vec::with_capacity(patches.len());
    for &(x0, y0) in patches {
        let mut good = true;
        let mut sites = HashSet::with_capacity(PIXELS);
        for y in 0..16 {
            for x in 0..16 {
                let (x, y) = ((x0 + 2 * x) as f64, (y0 + 2 * y) as f64);
                let (x, y) = registration.map_or((x, y), |r| {
                    r.warp
                        .as_ref()
                        .map_or_else(|| r.transform.inverse_apply(x, y), |w| w.apply(x, y))
                });
                let (x, y) = (x.round(), y.round());
                let v = if x.is_finite()
                    && y.is_finite()
                    && x >= 0.0
                    && y >= 0.0
                    && x < image.width as f64
                    && y < image.height as f64
                {
                    let site = y as usize * image.width + x as usize;
                    good &= sites.insert(site);
                    image.data[site]
                } else {
                    f32::NAN
                };
                good &= v.is_finite();
                values.push(v);
            }
        }
        valid.push(good);
    }
    (values, valid)
}
pub(crate) fn median(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut values: Vec<_> = values.into_iter().filter(|v| v.is_finite()).collect();
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f64::total_cmp);
    let n = values.len();
    Some(if n % 2 == 0 {
        values[n / 2 - 1] * 0.5 + values[n / 2] * 0.5
    } else {
        values[n / 2]
    })
}
pub(crate) fn sigma(values: &[f64]) -> f64 {
    let center = median(values.iter().copied()).unwrap_or(0.0);
    seiza_stats::NORMAL_MAD_SCALE * median(values.iter().map(|v| (v - center).abs())).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn patches_are_bounded_distinct_and_data_independent() {
        assert!(patches(10, 10, 8192).is_empty());
        let tiles = patches(1024, 768, 8192);
        assert_eq!(tiles.len(), 32);
        assert_eq!(tiles.iter().collect::<HashSet<_>>().len(), 32);
        for (x, y) in tiles {
            assert!(x + 30 < 1024 && y + 30 < 768);
        }
    }
    #[test]
    fn inverse_mapping_and_nonfinite_mask() {
        let mut image =
            LinearImage::new(128, 128, 1, (0..16384).map(|v| v as f32).collect()).unwrap();
        let transform = seiza_stacking::SimilarityTransform {
            translation_x: 4.0,
            translation_y: 2.0,
            scale: 1.0,
            rotation_radians: 0.0,
        };
        let r = RegistrationResult {
            transform,
            matched_stars: 10,
            rms_error_pixels: 0.0,
            drift_pixels: 4.5,
            warp: None,
        };
        let (v, valid) = gather(&image, &[(32, 32)], Some(&r));
        assert!(valid[0]);
        assert_eq!(v[0], image.data[30 * 128 + 28]);
        image.data[30 * 128 + 28] = f32::NAN;
        assert!(!gather(&image, &[(32, 32)], Some(&r)).1[0]);
    }
}
