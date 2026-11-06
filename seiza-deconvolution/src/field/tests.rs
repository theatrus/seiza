use super::*;
use seiza_stars::{PsfFieldOptions, SparseFieldPolicy};

/// A xorshift generator with Gaussian samples.
struct Random(u64);

impl Random {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (a, b) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    }
}

/// The second moment of a centred kernel along `x`.
fn variance_x(kernel: &[f32], size: usize) -> f64 {
    let centre = (size / 2) as f64;
    kernel
        .iter()
        .enumerate()
        .map(|(index, &v)| f64::from(v) * ((index % size) as f64 - centre).powi(2))
        .sum()
}

#[test]
fn the_remaining_kernel_of_two_gaussians_is_a_gaussian() {
    let size = 33;
    let divider = Divider::new(size);
    let psf = gaussian_stamp(4.0, size);
    let remaining = divider.remaining(&psf, 4.0, 2.0);
    assert!((remaining.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    let sigma = |fwhm: f64| fwhm / 2.354_820_045;
    let expected = sigma(4.0).powi(2) - sigma(2.0).powi(2);
    let measured = variance_x(&remaining, size);
    assert!(
        (measured - expected).abs() < 0.03 * expected,
        "{measured} against {expected}"
    );
}

#[test]
fn a_target_as_wide_as_the_psf_leaves_a_point() {
    let size = 33;
    let divider = Divider::new(size);
    let psf = gaussian_stamp(2.5, size);
    let remaining = divider.remaining(&psf, 2.5, 2.5);
    assert!(variance_x(&remaining, size) < 0.05);
    assert!(remaining[16 * size + 16] > 0.9);
}

/// Stars on a grid with a FWHM growing from `left` to `right`, on a sky
/// of 100 with noise of σ 2, and the per-pixel noise.
fn star_field(width: usize, height: usize, left: f64, right: f64, seed: u64) -> Vec<f32> {
    let mut random = Random(seed);
    let mut image = vec![100.0_f64; width * height];
    for cy in (20..height - 20).step_by(24) {
        for cx in (20..width - 20).step_by(24) {
            let (sx, sy) = (
                cx as f64 + random.uniform() - 0.5,
                cy as f64 + random.uniform() - 0.5,
            );
            let fwhm = left + (right - left) * sx / (width - 1) as f64;
            let sigma = fwhm / 2.354_820_045;
            let flux = 2.0e4 + 6.0e4 * random.uniform();
            for y in cy - 12..=cy + 12 {
                for x in cx - 12..=cx + 12 {
                    let r2 = (x as f64 - sx).powi(2) + (y as f64 - sy).powi(2);
                    image[y * width + x] += flux / (2.0 * std::f64::consts::PI * sigma * sigma)
                        * (-0.5 * r2 / (sigma * sigma)).exp();
                }
            }
        }
    }
    image
        .into_iter()
        .map(|value| (value + 2.0 * random.normal()) as f32)
        .collect()
}

fn field_of(image: &[f32], width: usize, height: usize) -> PsfField {
    PsfField::measure(
        image,
        width,
        height,
        1,
        &PsfFieldOptions {
            sparse: SparseFieldPolicy::Global,
            ..PsfFieldOptions::default()
        },
    )
    .unwrap()
}

/// A scene at the target PSF: Gaussian stars of FWHM `target` and a smooth
/// nebula, on a sky of 100.
fn scene(width: usize, height: usize, target: f64, seed: u64) -> Vec<f32> {
    let mut random = Random(seed);
    let mut image: Vec<f64> = (0..width * height)
        .map(|index| {
            let (x, y) = ((index % width) as f64, (index / width) as f64);
            let (dx, dy) = (x - width as f64 * 0.4, y - height as f64 * 0.6);
            100.0 + 60.0 * (-(dx * dx + dy * dy) / (2.0 * 30.0_f64.powi(2))).exp()
        })
        .collect();
    let sigma = target / 2.354_820_045;
    for _ in 0..60 {
        let (sx, sy) = (
            10.0 + random.uniform() * (width - 20) as f64,
            10.0 + random.uniform() * (height - 20) as f64,
        );
        let flux = 1.0e3 + 2.0e4 * random.uniform();
        let (cx, cy) = (sx.round() as isize, sy.round() as isize);
        for y in (cy - 8).max(0)..(cy + 9).min(height as isize) {
            for x in (cx - 8).max(0)..(cx + 9).min(width as isize) {
                let r2 = (x as f64 - sx).powi(2) + (y as f64 - sy).powi(2);
                image[y as usize * width + x as usize] += flux
                    / (2.0 * std::f64::consts::PI * sigma * sigma)
                    * (-0.5 * r2 / (sigma * sigma)).exp();
            }
        }
    }
    image.into_iter().map(|value| value as f32).collect()
}

#[test]
fn the_varying_blur_keeps_flux_and_a_flat_sky() {
    let (width, height) = (240, 200);
    let image = star_field(width, height, 3.0, 4.0, 3);
    let field = field_of(&image, width, height);
    let kernel = RemainingKernel::new(&field, TargetPsf::Fwhm(2.0), 64).unwrap();
    assert_eq!(kernel.node_count(), 5 * 5);
    let flat = vec![7.0_f32; width * height];
    let blurred = kernel.blur(&flat).unwrap();
    // Kernels that differ from node to node keep a flat sky flat to a
    // fraction of a percent: exactly where they agree, least well at the
    // edges, where reflection supplies the sky.
    let worst = |margin: usize| {
        (margin..height - margin)
            .flat_map(|y| (margin..width - margin).map(move |x| (x, y)))
            .map(|(x, y)| (blurred[y * width + x] - 7.0).abs() / 7.0)
            .fold(0.0_f32, f32::max)
    };
    assert!(worst(0) < 3e-3, "{}", worst(0));
    assert!(
        worst(kernel.kernel_size()) < 1e-3,
        "{}",
        worst(kernel.kernel_size())
    );
    // A point well inside keeps its light.
    let mut point = vec![0.0_f32; width * height];
    point[100 * width + 120] = 1000.0;
    let blurred = kernel.blur(&point).unwrap();
    assert!((blurred.iter().sum::<f32>() - 1000.0).abs() < 0.5);
    assert!(blurred[100 * width + 120] < 500.0);
}

#[test]
fn the_adjoint_matches_the_forward_blur() {
    let (width, height) = (240, 200);
    let image = star_field(width, height, 3.0, 4.0, 5);
    let field = field_of(&image, width, height);
    let kernel = RemainingKernel::new(&field, TargetPsf::Fwhm(2.0), 64).unwrap();
    let blur = VaryingBlur::new(&kernel, 0);
    let mut random = Random(11);
    // Zero within a stamp of the edges, where reflection breaks symmetry.
    let margin = kernel.kernel_size();
    let noise = |random: &mut Random| -> Vec<f32> {
        (0..width * height)
            .map(|index| {
                let (x, y) = (index % width, index / width);
                if x < margin || y < margin || x >= width - margin || y >= height - margin {
                    0.0
                } else {
                    random.normal() as f32
                }
            })
            .collect()
    };
    let (a, b) = (noise(&mut random), noise(&mut random));
    let (mut ha, mut htb) = (vec![0.0; a.len()], vec![0.0; a.len()]);
    blur.forward_into(&a, &mut ha);
    blur.adjoint_into(&b, &mut htb);
    let left: f64 = ha.iter().zip(&b).map(|(x, y)| f64::from(x * y)).sum();
    let right: f64 = a.iter().zip(&htb).map(|(x, y)| f64::from(x * y)).sum();
    assert!(
        (left - right).abs() < 1e-3 * left.abs().max(1.0),
        "{left} against {right}"
    );
}

#[test]
fn a_global_psf_blurs_like_one_kernel() {
    let (width, height) = (200, 160);
    let image = star_field(width, height, 3.0, 3.0, 7);
    let field = field_of(&image, width, height);
    let kernel = RemainingKernel::new(&field, TargetPsf::Fwhm(2.0), 48).unwrap();
    // The field still varies a little, so compare with the blend at each
    // point, applied directly.
    let mut random = Random(13);
    let input: Vec<f32> = (0..width * height)
        .map(|_| 10.0 + random.normal() as f32)
        .collect();
    let blurred = kernel.blur(&input).unwrap();
    let size = kernel.kernel_size();
    let half = (size / 2) as isize;
    for &(x, y) in &[(100usize, 80usize), (40, 40), (150, 120)] {
        // Gather form: the light at (x, y) from each source, through the
        // source's own kernel.
        let mut expected = 0.0_f64;
        for dy in -half..=half {
            for dx in -half..=half {
                let (sx, sy) = (x as isize - dx, y as isize - dy);
                let source = kernel.kernel_at(0, sx as f64, sy as f64).unwrap();
                let weight = source[((dy + half) * size as isize + dx + half) as usize];
                expected += f64::from(weight) * f64::from(input[sy as usize * width + sx as usize]);
            }
        }
        let got = f64::from(blurred[y * width + x]);
        assert!(
            (got - expected).abs() < 1e-3,
            "({x}, {y}): {got} against {expected}"
        );
    }
}

#[test]
fn deconvolution_sharpens_towards_the_target_and_keeps_flux() {
    let (width, height) = (240, 200);
    let image = star_field(width, height, 3.2, 4.2, 17);
    let field = field_of(&image, width, height);
    let kernel = RemainingKernel::new(&field, TargetPsf::Fwhm(2.2), 64).unwrap();
    let (low, high) = kernel.psf_fwhm_range(0).unwrap();
    assert!(low > 3.0 && high < 4.5, "{low} {high}");
    assert_eq!(kernel.target_fwhm_range(0), Some((2.2, 2.2)));
    let config = FieldDeconvolutionConfig {
        iterations: 30,
        amount: 1.0,
        noise_fraction: 0.0,
        max_correction: 3.0,
    };
    let restored = deconvolve_field(&image, &kernel, &config).unwrap();
    let input_flux: f64 = image.iter().map(|&v| f64::from(v)).sum();
    let output_flux: f64 = restored.data.iter().map(|&v| f64::from(v)).sum();
    assert!((output_flux - input_flux).abs() < 1e-4 * input_flux);
    let sharpened = field_of(&restored.data, width, height);
    for &x in &[40.0, 120.0, 200.0] {
        let before = field.shape_at(0, x, 100.0).unwrap().fwhm;
        let after = sharpened.shape_at(0, x, 100.0).unwrap().fwhm;
        assert!(
            after < before - 0.6 && after > 1.8,
            "at x {x}: {before} to {after}"
        );
    }
    // Every part of the field moves towards one target.
    let spread = |field: &PsfField| {
        let left = field.shape_at(0, 30.0, 100.0).unwrap().fwhm;
        let right = field.shape_at(0, 210.0, 100.0).unwrap().fwhm;
        right - left
    };
    assert!(spread(&sharpened) < 0.6 * spread(&field));
}

#[test]
fn reblur_consistency_holds_for_the_truth_and_fails_for_the_input() {
    let (width, height) = (256, 256);
    let stars = star_field(width, height, 3.0, 3.6, 19);
    let field = field_of(&stars, width, height);
    let kernel = RemainingKernel::new(&field, TargetPsf::BestRound, 64).unwrap();
    let truth = scene(width, height, 2.0, 23);
    let mut random = Random(29);
    let sigma = 2.0_f32;
    let observed: Vec<f32> = kernel
        .blur(&truth)
        .unwrap()
        .into_iter()
        .map(|v| v + sigma * random.normal() as f32)
        .collect();
    let noise = vec![sigma; width * height];
    let good = reblur_consistency(&truth, &observed, &noise, &kernel).unwrap();
    let channel = &good.channels[0];
    assert_eq!(channel.samples, width * height);
    assert!(
        (channel.reduced_chi_squared - 1.0).abs() < 0.05,
        "{channel:?}"
    );
    assert_eq!(channel.band_power.len(), SPECTRUM_BANDS);
    assert!(channel.spectral_flatness.unwrap() > 0.95, "{channel:?}");

    // The input itself, offered as the sharpened image, is too soft: its
    // reblur misses the stars' cores.
    let bad = reblur_consistency(&observed, &observed, &noise, &kernel).unwrap();
    let channel = &bad.channels[0];
    assert!(channel.reduced_chi_squared > 1.5, "{channel:?}");
    assert!(channel.spectral_flatness.unwrap() < 0.95, "{channel:?}");

    // Masked samples are left out.
    let mut masked = noise.clone();
    masked[..width].fill(f32::NAN);
    let partial = reblur_consistency(&truth, &observed, &masked, &kernel).unwrap();
    assert_eq!(partial.channels[0].samples, width * (height - 1));
}

#[test]
fn rejects_bad_targets_and_sizes() {
    assert!(TargetPsf::Fwhm(1.0).validate().is_err());
    assert!(TargetPsf::Ratio(1.5).validate().is_err());
    assert!(TargetPsf::Ratio(0.7).validate().is_ok());
    let (width, height) = (200, 160);
    let image = star_field(width, height, 3.0, 3.0, 31);
    let field = field_of(&image, width, height);
    assert!(RemainingKernel::new(&field, TargetPsf::BestRound, 8).is_err());
    let kernel = RemainingKernel::new(&field, TargetPsf::Ratio(0.7), 64).unwrap();
    let (low, high) = kernel.target_fwhm_range(0).unwrap();
    assert!(low >= MINIMUM_TARGET_FWHM && high < 2.5, "{low} {high}");
    assert!(kernel.blur(&[0.0; 10]).is_err());
    assert!(
        deconvolve_field(
            &[0.0; 10],
            &kernel,
            &FieldDeconvolutionConfig::conservative()
        )
        .is_err()
    );
}
