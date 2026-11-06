use super::*;
use crate::{PsfFieldOptions, SparseFieldPolicy};

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

const SIGMA: f64 = 2.0;

/// A round Moffat with β 3 whose FWHM grows from 2.8 px on the left to 3.6
/// on the right, per unit flux, pixel-integrated from 4 × 4 samples.
fn psf(dx: f64, dy: f64, x: f64, width: usize) -> f64 {
    let fwhm = 2.8 + 0.8 * x / (width - 1) as f64;
    let beta = 3.0;
    let alpha = fwhm / (2.0 * (2.0_f64.powf(1.0 / beta) - 1.0).sqrt());
    let norm = (beta - 1.0) / (std::f64::consts::PI * alpha * alpha);
    let mut sum = 0.0;
    for j in 0..4 {
        for i in 0..4 {
            let (u, v) = (dx - 0.375 + 0.25 * i as f64, dy - 0.375 + 0.25 * j as f64);
            sum += norm * (1.0 + (u * u + v * v) / (alpha * alpha)).powf(-beta);
        }
    }
    sum / 16.0
}

struct Scene {
    width: usize,
    height: usize,
    /// The sky alone: a plane and a soft nebula.
    sky: Vec<f64>,
    stars: Vec<(f64, f64, f64)>,
    image: Vec<f32>,
    noise: Vec<f32>,
}

/// Bright stars on a jittered grid, for the PSF field, and faint stars
/// between them, all on a sky with a gradient and a nebula.
fn scene(width: usize, height: usize, seed: u64, extra: &[(f64, f64, f64)]) -> Scene {
    let mut random = Random(seed);
    let mut stars = Vec::new();
    for cy in (24..height - 24).step_by(30) {
        for cx in (24..width - 24).step_by(30) {
            let (x, y) = (
                cx as f64 + random.uniform() - 0.5,
                cy as f64 + random.uniform() - 0.5,
            );
            stars.push((x, y, 1.0e4 + 6.0e4 * random.uniform()));
            // A faint companion 15 px away, inside the bright star's stamp.
            stars.push((
                x + 10.5 + random.uniform(),
                y + 11.0,
                800.0 + 1200.0 * random.uniform(),
            ));
        }
    }
    stars.extend_from_slice(extra);
    let sky: Vec<f64> = (0..width * height)
        .map(|index| {
            let (x, y) = ((index % width) as f64, (index / width) as f64);
            let (dx, dy) = (x - 0.6 * width as f64, y - 0.4 * height as f64);
            100.0 + 0.05 * x - 0.03 * y
                + 40.0 * (-(dx * dx + dy * dy) / (2.0 * 40.0_f64.powi(2))).exp()
        })
        .collect();
    let mut values = sky.clone();
    for &(sx, sy, flux) in &stars {
        let (cx, cy) = (sx.round() as isize, sy.round() as isize);
        for py in (cy - 20).max(0)..(cy + 21).min(height as isize) {
            for px in (cx - 20).max(0)..(cx + 21).min(width as isize) {
                values[py as usize * width + px as usize] +=
                    flux * psf(px as f64 - sx, py as f64 - sy, sx, width);
            }
        }
    }
    let image = values
        .iter()
        .map(|v| (v + SIGMA * random.normal()) as f32)
        .collect();
    Scene {
        width,
        height,
        sky,
        stars,
        image,
        noise: vec![SIGMA as f32; width * height],
    }
}

fn field_of(scene: &Scene) -> PsfField {
    PsfField::measure(
        &scene.image,
        scene.width,
        scene.height,
        1,
        &PsfFieldOptions {
            sparse: SparseFieldPolicy::Global,
            ..PsfFieldOptions::default()
        },
    )
    .unwrap()
}

fn catalogue(stars: &[(f64, f64, f64)]) -> Vec<CatalogueStar> {
    stars
        .iter()
        .map(|&(x, y, flux)| CatalogueStar {
            x,
            y,
            magnitude: -2.5 * flux.log10(),
        })
        .collect()
}

#[test]
fn fits_fluxes_at_catalogue_positions() {
    let scene = scene(300, 270, 3, &[]);
    let field = field_of(&scene);
    let fit = StarFit::fit(
        &scene.image,
        &field,
        &catalogue(&scene.stars),
        Some(&scene.noise),
        &StarFitOptions::default(),
    )
    .unwrap();
    assert_eq!(fit.stars().len(), scene.stars.len());
    let mut pulls = Vec::new();
    let mut chi = Vec::new();
    for (star, &(_, _, truth)) in fit.stars().iter().zip(&scene.stars) {
        let (flux, error) = (star.flux[0], star.flux_error[0]);
        assert!(flux.is_finite() && error > 0.0, "{star:?}");
        // Bright stars come out about 2% low: bright neighbours every 30 px
        // put a little of their wings into each star's sky ring, so the
        // measured wings run a few percent low. A faint star on the
        // nebula's curve takes the background plane's error, a couple of
        // units, which is about 100 in flux.
        let tolerance = (0.03 * truth).max(5.0 * error).max(100.0);
        assert!(
            (flux - truth).abs() < tolerance,
            "flux {flux} ± {error} against {truth}"
        );
        if truth < 3000.0 {
            pulls.push((flux - truth) / error);
        }
        // Bright stars' χ² carries the PSF model's percent-level error.
        assert!(star.reduced_chi_squared < 30.0, "{star:?}");
        chi.push(star.reduced_chi_squared);
        assert!(!star.saturated);
    }
    chi.sort_by(f64::total_cmp);
    assert!(chi[chi.len() / 2] < 2.0, "median χ² {}", chi[chi.len() / 2]);
    // The faint stars' errors are the noise's share of their scatter; the
    // sky's curve under the plane adds to it here.
    let spread = (pulls.iter().map(|p| p * p).sum::<f64>() / pulls.len() as f64).sqrt();
    assert!(spread > 0.5 && spread < 4.0, "pull spread {spread}");
}

#[test]
fn subtracting_leaves_the_sky_and_rendering_gives_the_stars_at_the_target() {
    let scene = scene(300, 270, 5, &[]);
    let field = field_of(&scene);
    let fit = StarFit::fit(
        &scene.image,
        &field,
        &catalogue(&scene.stars),
        Some(&scene.noise),
        &StarFitOptions::default(),
    )
    .unwrap();
    let removed = fit.subtract(&scene.image).unwrap();
    // Away from the edges and the bright stars' inner 8 px, what is left is
    // the sky and its noise; nearer, the PSF model's percent-level error.
    let (width, height) = (scene.width, scene.height);
    let near_core = |x: usize, y: usize| {
        scene
            .stars
            .iter()
            .any(|s| s.2 > 1.0e4 && (x as f64 - s.0).hypot(y as f64 - s.1) < 8.0)
    };
    let (mut clear, mut cores, mut before) = ((0.0, 0), (0.0, 0), 0.0);
    for y in 20..height - 20 {
        for x in 20..width - 20 {
            let index = y * width + x;
            let r2 = ((f64::from(removed[index]) - scene.sky[index]) / SIGMA).powi(2);
            before += ((f64::from(scene.image[index]) - scene.sky[index]) / SIGMA).powi(2);
            let sum = if near_core(x, y) {
                &mut cores
            } else {
                &mut clear
            };
            sum.0 += r2;
            sum.1 += 1;
        }
    }
    let clear_chi = clear.0 / clear.1 as f64;
    assert!(clear_chi < 1.35, "residual χ² {clear_chi} away from cores");
    assert!(
        cores.0 + clear.0 < 0.01 * before,
        "{} left of {before}",
        cores.0 + clear.0
    );

    let target = 2.0;
    let rendered = fit.render(target).unwrap();
    let total: f64 = rendered.iter().map(|&v| f64::from(v)).sum();
    let truth: f64 = scene.stars.iter().map(|s| s.2).sum();
    assert!(
        (total - truth).abs() < 0.03 * truth,
        "{total} against {truth}"
    );
    // The sharpened image's stars are as narrow as the target.
    let sharpened = fit.sharpen(&scene.image, target).unwrap();
    let after = PsfField::measure(
        &sharpened,
        width,
        height,
        1,
        &PsfFieldOptions {
            sparse: SparseFieldPolicy::Global,
            ..PsfFieldOptions::default()
        },
    )
    .unwrap();
    let shape = after.shape_at(0, 150.0, 135.0).unwrap();
    assert!((shape.fwhm - target).abs() < 0.25, "{shape:?}");
    assert!(shape.ellipticity() < 0.05, "{shape:?}");
}

#[test]
fn overlapping_stars_are_fitted_jointly() {
    // A pair 3 px apart, unresolved at this PSF, and a third star nearby.
    let pair = [
        (150.4, 140.2, 2.0e4),
        (153.4, 140.6, 1.2e4),
        (147.0, 146.0, 6.0e3),
    ];
    let scene = scene(300, 270, 7, &pair);
    let field = field_of(&scene);
    let fit = StarFit::fit(
        &scene.image,
        &field,
        &catalogue(&scene.stars),
        Some(&scene.noise),
        &StarFitOptions::default(),
    )
    .unwrap();
    let n = scene.stars.len();
    for (star, truth) in fit.stars()[n - 3..].iter().zip(&pair) {
        assert!(
            (star.flux[0] - truth.2).abs() < 0.08 * truth.2,
            "{} against {}",
            star.flux[0],
            truth.2
        );
    }
}

#[test]
fn refined_positions_follow_the_image() {
    let scene = scene(300, 270, 11, &[]);
    let field = field_of(&scene);
    // A solution 0.3 px off in x and 0.2 px in y.
    let shifted: Vec<(f64, f64, f64)> = scene
        .stars
        .iter()
        .map(|&(x, y, flux)| (x + 0.3, y - 0.2, flux))
        .collect();
    let fixed = StarFit::fit(
        &scene.image,
        &field,
        &catalogue(&shifted),
        Some(&scene.noise),
        &StarFitOptions::default(),
    )
    .unwrap();
    let refined = StarFit::fit(
        &scene.image,
        &field,
        &catalogue(&shifted),
        Some(&scene.noise),
        &StarFitOptions {
            positions: StarPositions::Refined,
            ..StarFitOptions::default()
        },
    )
    .unwrap();
    let bright = |fit: &StarFit| -> Vec<FittedStar> {
        fit.stars()
            .iter()
            .zip(&scene.stars)
            .filter(|(_, truth)| truth.2 > 1.0e4)
            .map(|(star, _)| star.clone())
            .collect()
    };
    let median = |mut values: Vec<f64>| {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    let fixed_chi = median(
        bright(&fixed)
            .iter()
            .map(|s| s.reduced_chi_squared)
            .collect(),
    );
    let refined_chi = median(
        bright(&refined)
            .iter()
            .map(|s| s.reduced_chi_squared)
            .collect(),
    );
    assert!(
        refined_chi < 0.5 * fixed_chi,
        "{refined_chi} against {fixed_chi}"
    );
    for (star, truth) in refined
        .stars()
        .iter()
        .zip(&scene.stars)
        .filter(|(_, truth)| truth.2 > 1.0e4)
    {
        assert!(
            (star.x - truth.0).abs() < 0.05 && (star.y - truth.1).abs() < 0.05,
            "({}, {}) against ({}, {})",
            star.x,
            star.y,
            truth.0,
            truth.1
        );
    }
}

#[test]
fn saturated_cores_are_left_out_and_the_wings_give_the_flux() {
    let bright = [(151.0, 139.7, 2.0e6)];
    let mut scene = scene(300, 270, 13, &bright);
    let level = 40_000.0_f32;
    for value in &mut scene.image {
        *value = value.min(level);
    }
    let field = PsfField::measure(
        &scene.image,
        scene.width,
        scene.height,
        1,
        &PsfFieldOptions {
            sparse: SparseFieldPolicy::Global,
            saturation: Some(0.8 * level),
            ..PsfFieldOptions::default()
        },
    )
    .unwrap();
    let fit = StarFit::fit(
        &scene.image,
        &field,
        &catalogue(&scene.stars),
        Some(&scene.noise),
        &StarFitOptions {
            saturation: Some(0.99 * level),
            ..StarFitOptions::default()
        },
    )
    .unwrap();
    let star = fit.stars().last().unwrap();
    assert!(star.saturated);
    assert!(
        (star.flux[0] - 2.0e6).abs() < 0.1 * 2.0e6,
        "{} against 2e6",
        star.flux[0]
    );
}

#[test]
fn rejects_mismatched_images_and_bad_options() {
    let scene = scene(200, 180, 17, &[]);
    let field = field_of(&scene);
    let stars = catalogue(&scene.stars);
    assert!(matches!(
        StarFit::fit(
            &scene.image[..100],
            &field,
            &stars,
            None,
            &StarFitOptions::default()
        ),
        Err(StarFitError::InvalidImage(_))
    ));
    assert!(matches!(
        StarFit::fit(
            &scene.image,
            &field,
            &stars,
            None,
            &StarFitOptions {
                fit_radius: Some(0.5),
                ..StarFitOptions::default()
            }
        ),
        Err(StarFitError::InvalidOptions(_))
    ));
    let fit = StarFit::fit(
        &scene.image,
        &field,
        &stars,
        None,
        &StarFitOptions::default(),
    )
    .unwrap();
    assert!(fit.subtract(&[0.0; 4]).is_err());
    assert!(fit.render(0.0).is_err());
    // Without a noise map the errors come from the residual scatter.
    assert!(fit.stars().iter().all(|s| s.flux_error[0] > 0.0));
    for positions in [StarPositions::Fixed, StarPositions::Refined] {
        assert_eq!(
            positions.to_string().parse::<StarPositions>(),
            Ok(positions)
        );
    }
}
