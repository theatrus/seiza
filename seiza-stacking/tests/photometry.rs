//! Forced aperture photometry on synthetic stars whose flux is known.
use seiza_stacking::{
    ApertureGeometry, ApertureMaps, ApertureMeasurement, AperturePhotometryOptions, LinearImage,
    aperture_photometry,
};
use std::f64::consts::PI;

const WIDTH: usize = 96;
const HEIGHT: usize = 80;

/// Unit normals from a fixed seed.
struct Noise(u64);
impl Noise {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.uniform().ln()).sqrt() * (2.0 * PI * self.uniform()).cos()
    }
}

/// A Gaussian star of total `flux` and width `sigma` at each of `stars`, on
/// the sky `background(x, y)`.
fn field(stars: &[([f64; 2], f64)], sigma: f64, background: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    let mut data = vec![0.0; WIDTH * HEIGHT];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let (fx, fy) = (x as f64, y as f64);
            let mut value = background(fx, fy);
            for ([sx, sy], flux) in stars {
                let r2 = (fx - sx).powi(2) + (fy - sy).powi(2);
                value += flux / (2.0 * PI * sigma * sigma) * (-r2 / (2.0 * sigma * sigma)).exp();
            }
            data[y * WIDTH + x] = value;
        }
    }
    data
}

fn image(data: &[f64]) -> LinearImage {
    LinearImage::new(
        WIDTH,
        HEIGHT,
        1,
        data.iter().map(|value| *value as f32).collect(),
    )
    .unwrap()
}

fn options(aperture: f64, inner: f64, outer: f64) -> AperturePhotometryOptions {
    AperturePhotometryOptions::new(ApertureGeometry::new(aperture, inner, outer))
}

fn measure(
    data: &[f64],
    positions: &[[f64; 2]],
    options: &AperturePhotometryOptions,
) -> Vec<ApertureMeasurement> {
    aperture_photometry(&image(data), positions, options, &ApertureMaps::default()).unwrap()
}

/// The share of a Gaussian's flux inside a circle of `radius` about it.
fn enclosed(radius: f64, sigma: f64) -> f64 {
    1.0 - (-radius * radius / (2.0 * sigma * sigma)).exp()
}

#[test]
fn flux_is_recovered_across_a_background_gradient() {
    let stars = [([30.0, 28.0], 5000.0), ([62.4, 47.7], 800.0)];
    let positions = [stars[0].0, stars[1].0];
    let geometry = options(9.0, 12.0, 18.0);
    for (slope_x, slope_y) in [(0.0, 0.0), (0.8, -0.5), (-2.0, 3.0)] {
        let sky = |x: f64, y: f64| 400.0 + slope_x * x + slope_y * y;
        let measured = measure(&field(&stars, 1.8, sky), &positions, &geometry);
        assert_eq!(measured.len(), 2);
        for (result, ([x, y], flux)) in measured.iter().zip(stars) {
            assert_eq!((result.x, result.y), (x, y));
            assert_eq!(result.flags, Default::default());
            let expected = flux * enclosed(9.0, 1.8);
            let got = result.flux.unwrap();
            assert!(
                (got - expected).abs() < 2e-4 * flux,
                "{got} against {expected}"
            );
            let background = result.background.unwrap();
            assert!((background - sky(x, y)).abs() < 2e-3, "{background}");
            assert!((result.aperture_area - PI * 81.0).abs() < 1e-9);
            assert_eq!(
                result.magnitude.unwrap(),
                -2.5 * result.flux.unwrap().log10()
            );
        }
    }
}

#[test]
fn flux_holds_steady_across_subpixel_shifts() {
    // A tight aperture, where a whole-pixel edge would move the sum.
    let geometry = options(3.0, 8.0, 14.0);
    let expected = 1000.0 * enclosed(3.0, 1.6);
    let (mut fluxes, mut whole) = (Vec::new(), Vec::new());
    for step in 0..25 {
        let position = [
            40.0 + f64::from(step % 5) * 0.2,
            36.0 + f64::from(step / 5) * 0.2,
        ];
        let sky = |x: f64, y: f64| 50.0 + 0.3 * x - 0.2 * y;
        let data = field(&[(position, 1000.0)], 1.6, sky);
        let result = &measure(&data, &[position], &geometry)[0];
        assert!((result.aperture_area - PI * 9.0).abs() < 1e-10);
        fluxes.push(result.flux.unwrap());
        // The same star summed over the pixels whose centres are inside.
        let mut sum = 0.0;
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                if (x as f64 - position[0]).hypot(y as f64 - position[1]) <= 3.0 {
                    sum += data[y * WIDTH + x] - sky(x as f64, y as f64);
                }
            }
        }
        whole.push(sum);
    }
    let range = |values: &[f64]| {
        let low = values.iter().copied().fold(f64::MAX, f64::min);
        let high = values.iter().copied().fold(f64::MIN, f64::max);
        (low, high)
    };
    let (low, high) = range(&fluxes);
    let (whole_low, whole_high) = range(&whole);
    // An edge pixel counts by its area at its centre's value, so on a
    // profile this steep the sum still moves by half a percent, and sits
    // within two of the continuous integral. Whole pixels move it by far more.
    assert!((high - low) / expected < 6e-3, "{low}..{high}");
    assert!((low - expected).abs() / expected < 2e-2, "{low}");
    assert!(
        whole_high - whole_low > 8.0 * (high - low),
        "{whole_low}..{whole_high} against {low}..{high}"
    );
}

#[test]
fn faint_sources_keep_their_sign_and_empty_sky_scatters_about_zero() {
    let mut noise = Noise(0x9e37_79b9_7f4a_7c15);
    let geometry = options(4.0, 7.0, 12.0);
    let position = [48.0, 40.0];
    let (mut negative, mut pulls) = (0, Vec::new());
    for _ in 0..400 {
        let data: Vec<_> = field(&[], 1.5, |_, _| 200.0)
            .iter()
            .map(|value| value + 3.0 * noise.normal())
            .collect();
        let result = &measure(&data, &[position], &geometry)[0];
        assert_eq!(result.flags, Default::default());
        let flux = result.flux.unwrap();
        let error = result.flux_error.unwrap();
        if flux <= 0.0 {
            negative += 1;
            assert_eq!(result.magnitude, None);
            assert_eq!(result.magnitude_error, None);
        } else {
            let expected = 2.5 / std::f64::consts::LN_10 * error / flux;
            assert_eq!(result.magnitude_error.unwrap(), expected);
        }
        assert!((result.background_noise.unwrap() - 3.0).abs() < 0.8);
        pulls.push(flux / error);
    }
    // No source: about half the fluxes fall below zero and are reported so,
    // and the reported error is the scatter the fluxes show.
    assert!((140..=260).contains(&negative), "{negative}");
    let mean = pulls.iter().sum::<f64>() / pulls.len() as f64;
    let spread =
        (pulls.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / pulls.len() as f64).sqrt();
    assert!(mean.abs() < 0.15, "{mean}");
    assert!((spread - 1.0).abs() < 0.12, "{spread}");
}

#[test]
fn the_flux_error_follows_the_documented_noise_terms() {
    let mut noise = Noise(0x1234_5678_9abc_def1);
    let star = ([48.0, 40.0], 20_000.0);
    let data: Vec<_> = field(&[star], 1.7, |_, _| 300.0)
        .iter()
        .map(|value| value + 2.0 * noise.normal())
        .collect();
    let mut settings = options(6.0, 10.0, 16.0);
    let sky_only = measure(&data, &[star.0], &settings)[0].clone();
    let (area, sky_noise) = (sky_only.aperture_area, sky_only.background_noise.unwrap());

    // A supplied map stands for the annulus's scatter: each pixel's variance
    // by its weight squared, which is a little less than the area.
    let mapped = |sigma: f32, settings: &AperturePhotometryOptions| {
        let uncertainty = vec![sigma; WIDTH * HEIGHT];
        let maps = ApertureMaps {
            uncertainty: Some(&uncertainty),
            ..Default::default()
        };
        aperture_photometry(&image(&data), &[star.0], settings, &maps).unwrap()[0].clone()
    };
    let sky_fit = mapped(0.0, &settings).flux_error.unwrap().powi(2);
    let squared_weights = (mapped(5.0, &settings).flux_error.unwrap().powi(2) - sky_fit) / 25.0;
    assert!(
        squared_weights < area && squared_weights > area - 0.5 * PI * 6.0,
        "{squared_weights} against {area}"
    );
    let doubled = mapped(10.0, &settings).flux_error.unwrap().powi(2) - sky_fit;
    assert!((doubled - 100.0 * squared_weights).abs() < 1e-6);
    let sky_variance = squared_weights * sky_noise * sky_noise + sky_fit;
    assert!((sky_only.flux_error.unwrap().powi(2) - sky_variance).abs() < 1e-6);

    // A gain adds the source's photons, and a map replaces them.
    settings.gain = Some(0.5);
    let with_gain = measure(&data, &[star.0], &settings)[0].clone();
    let flux = with_gain.flux.unwrap();
    assert_eq!(flux, sky_only.flux.unwrap());
    let photons = with_gain.flux_error.unwrap().powi(2) - sky_variance;
    assert!((photons / (flux / 0.5) - 1.0).abs() < 1e-3, "{photons}");
    let without_gain = AperturePhotometryOptions {
        gain: None,
        ..settings.clone()
    };
    assert_eq!(
        mapped(5.0, &settings).flux_error,
        mapped(5.0, &without_gain).flux_error
    );
    let uncertainty = vec![5.0_f32; WIDTH * HEIGHT];

    // A map with no value where the star is gives a flux and no error.
    let mut broken = uncertainty.clone();
    broken[40 * WIDTH + 48] = f32::NAN;
    let maps = ApertureMaps {
        uncertainty: Some(&broken),
        ..Default::default()
    };
    let result = &aperture_photometry(&image(&data), &[star.0], &settings, &maps).unwrap()[0];
    assert_eq!(result.flux, with_gain.flux);
    assert_eq!((result.flux_error, result.magnitude_error), (None, None));
    assert!(result.magnitude.is_some());
}

#[test]
fn photon_noise_with_a_gain_matches_the_scatter_of_repeated_frames() {
    let mut noise = Noise(0xfeed_face_cafe_beef);
    let star = ([47.3, 39.6], 6000.0);
    let (gain, read) = (2.0, 1.5);
    let truth = field(&[star], 1.7, |_, _| 100.0);
    let mut settings = options(6.0, 10.0, 16.0);
    settings.gain = Some(gain);
    let (mut fluxes, mut errors) = (Vec::new(), Vec::new());
    for _ in 0..300 {
        let data: Vec<_> = truth
            .iter()
            .map(|value| value + (value / gain + read * read).sqrt() * noise.normal())
            .collect();
        let result = &measure(&data, &[star.0], &settings)[0];
        fluxes.push(result.flux.unwrap());
        errors.push(result.flux_error.unwrap());
    }
    let mean = fluxes.iter().sum::<f64>() / fluxes.len() as f64;
    let scatter =
        (fluxes.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / fluxes.len() as f64).sqrt();
    let reported = errors.iter().sum::<f64>() / errors.len() as f64;
    assert!(
        (mean - star.1 * enclosed(6.0, 1.7)).abs() < 4.0 * scatter / 300_f64.sqrt() + 2.0,
        "{mean}"
    );
    assert!(
        (reported / scatter - 1.0).abs() < 0.12,
        "{reported} against {scatter}"
    );
}

#[test]
fn masks_remove_neighbours_and_report_lost_aperture() {
    let target = ([40.0, 40.0], 3000.0);
    let neighbour = ([53.0, 43.0], 9000.0);
    let data = field(&[target, neighbour], 1.6, |x, _| 120.0 + 0.2 * x);
    let geometry = options(6.0, 9.0, 16.0);
    let expected = target.1 * enclosed(6.0, 1.6);

    // Clipping alone keeps most of the neighbour out of the sky; masking it
    // keeps all of it out, and the aperture is whole either way.
    let mut mask = vec![0_u8; WIDTH * HEIGHT];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let distance = (x as f64 - neighbour.0[0]).hypot(y as f64 - neighbour.0[1]);
            mask[y * WIDTH + x] = u8::from(distance < 6.5);
        }
    }
    let maps = ApertureMaps {
        mask: Some(&mask),
        ..Default::default()
    };
    let unmasked = &measure(&data, &[target.0], &geometry)[0];
    let masked = &aperture_photometry(&image(&data), &[target.0], &geometry, &maps).unwrap()[0];
    assert_eq!(masked.flags, Default::default());
    assert!(masked.annulus_pixels < unmasked.annulus_pixels + 40);
    assert!((masked.flux.unwrap() - expected).abs() < 1.0, "{masked:?}");
    assert!(
        (unmasked.flux.unwrap() - expected).abs() < 0.02 * expected,
        "{unmasked:?}"
    );

    // A masked pixel inside the aperture is left out and said to be.
    let mut hole = vec![0_u8; WIDTH * HEIGHT];
    hole[40 * WIDTH + 41] = 1;
    let maps = ApertureMaps {
        mask: Some(&hole),
        ..Default::default()
    };
    let holed = &aperture_photometry(&image(&data), &[target.0], &geometry, &maps).unwrap()[0];
    assert!(holed.flags.missing_coverage && !holed.flags.unusable && !holed.flags.saturated);
    assert!((holed.aperture_area - (PI * 36.0 - 1.0)).abs() < 1e-9);
    assert!(holed.flux.unwrap() < 0.95 * expected);

    // So is a sample that is not a number.
    let mut broken = data.clone();
    broken[40 * WIDTH + 41] = f64::NAN;
    let result = &measure(&broken, &[target.0], &geometry)[0];
    assert!(result.flags.missing_coverage);
    assert_eq!(result.flux, holed.flux);
}

#[test]
fn saturation_is_flagged_by_level_and_by_map_and_the_flux_kept() {
    let star = ([44.0, 38.0], 60_000.0);
    let ceiling = 3000.0;
    let data: Vec<_> = field(&[star, ([70.0, 60.0], 500.0)], 1.6, |_, _| 150.0)
        .iter()
        .map(|value| value.min(ceiling))
        .collect();
    let positions = [star.0, [70.0, 60.0]];
    let mut settings = options(6.0, 9.0, 15.0);
    let unflagged = measure(&data, &positions, &settings);
    assert!(!unflagged[0].flags.saturated);

    settings.saturation = Some(ceiling);
    let by_level = measure(&data, &positions, &settings);
    assert!(by_level[0].flags.saturated && !by_level[0].flags.unusable);
    assert_eq!(by_level[0].peak, Some(ceiling));
    assert_eq!(by_level[0].flux, unflagged[0].flux);
    assert!(by_level[0].flux.unwrap() < star.1);
    assert_eq!(by_level[1].flags, Default::default());

    settings.saturation = None;
    let mut clipped = vec![0_u8; WIDTH * HEIGHT];
    clipped[38 * WIDTH + 44] = 1;
    let maps = ApertureMaps {
        saturation: Some(&clipped),
        ..Default::default()
    };
    let by_map = aperture_photometry(&image(&data), &positions, &settings, &maps).unwrap();
    assert!(by_map[0].flags.saturated && !by_map[1].flags.saturated);
    assert_eq!(by_map[0].flux, unflagged[0].flux);
}

#[test]
fn every_position_gets_a_result_on_the_edge_and_off_the_image() {
    let star = ([3.0, 40.0], 2000.0);
    // Nonzero sky noise prevents negligible analytic PSF wings from being
    // clipped out of this already partial annulus.
    let mut noise = Noise(0x1234_5678_9abc_def1);
    let data: Vec<_> = field(&[star], 1.6, |x, y| 80.0 + 0.1 * x + 0.05 * y)
        .into_iter()
        .map(|v| v + 0.01 * noise.normal())
        .collect();
    let geometry = options(5.0, 8.0, 14.0);
    let positions = [
        star.0,
        [-40.0, 20.0],
        [48.0, 40.0],
        [0.0, 0.0],
        [1e12, -1e12],
    ];
    let results = measure(&data, &positions, &geometry);
    assert_eq!(results.len(), positions.len());
    for (result, position) in results.iter().zip(positions) {
        assert_eq!([result.x, result.y], position);
    }
    // Over the left edge: part of the aperture is gone, the rest is summed.
    let edge = &results[0];
    assert!(edge.flags.missing_coverage && !edge.flags.unusable);
    assert!(edge.aperture_area < PI * 25.0 - 5.0);
    assert!(edge.flux.unwrap() > 0.8 * star.1 && edge.flux.unwrap() < star.1);
    assert!((edge.background.unwrap() - (80.0 + 0.3 + 2.0)).abs() < 0.05);
    // Off the image: nothing to measure, and no number pretending otherwise.
    for index in [1, 4] {
        let off = &results[index];
        assert!(off.flags.unusable && off.flags.missing_coverage);
        assert_eq!(
            (off.flux, off.flux_error, off.magnitude),
            (None, None, None)
        );
        assert_eq!((off.aperture_area, off.aperture_sum), (0.0, None));
    }
    assert_eq!(results[2].flags, Default::default());
    // A corner keeps a quarter of its annulus: too little for a background.
    let corner = &results[3];
    assert!(corner.flags.unusable && corner.flags.missing_coverage);
    assert_eq!(corner.flux, None);
    assert!(corner.aperture_sum.is_some());
}

#[test]
fn input_that_cannot_be_measured_is_refused() {
    let data = field(&[], 1.5, |_, _| 10.0);
    let mono = image(&data);
    let maps = ApertureMaps::default();
    let good = options(4.0, 6.0, 10.0);
    let refused = |options: &AperturePhotometryOptions, positions: &[[f64; 2]]| {
        aperture_photometry(&mono, positions, options, &maps).is_err()
    };
    assert!(!refused(&good, &[[20.0, 20.0]]));
    assert!(
        aperture_photometry(&mono, &[], &good, &maps)
            .unwrap()
            .is_empty()
    );
    for (aperture, inner, outer) in [
        (0.0, 6.0, 10.0),
        (4.0, 3.0, 10.0),
        (4.0, 6.0, 6.0),
        (f64::NAN, 6.0, 10.0),
        (4.0, 6.0, f64::INFINITY),
    ] {
        assert!(refused(&options(aperture, inner, outer), &[[20.0, 20.0]]));
    }
    assert!(refused(&good, &[[f64::NAN, 20.0]]));
    let mut bad_gain = good.clone();
    bad_gain.gain = Some(0.0);
    assert!(refused(&bad_gain, &[[20.0, 20.0]]));

    let rgb = LinearImage::new(4, 4, 3, vec![0.0; 48]).unwrap();
    let error = aperture_photometry(&rgb, &[[2.0, 2.0]], &good, &maps).unwrap_err();
    assert!(error.to_string().contains("mono"), "{error}");
    let short = vec![0_u8; 7];
    let maps = ApertureMaps {
        mask: Some(&short),
        ..Default::default()
    };
    assert!(aperture_photometry(&mono, &[[20.0, 20.0]], &good, &maps).is_err());
}

#[test]
fn asymmetric_sky_errors_match_repeated_frames() {
    let mut noise = Noise(0x1234_5678_9abc_def1);
    let settings = options(8.0, 10.0, 14.0);
    let position = [48.0, 40.0];
    let mask: Vec<u8> = (0..WIDTH * HEIGHT)
        .map(|i| {
            let dx = (i % WIDTH) as f64 - position[0];
            let dy = (i / WIDTH) as f64 - position[1];
            u8::from(dx < -1.0 && dx.hypot(dy) >= 10.0)
        })
        .collect();
    let maps = ApertureMaps {
        mask: Some(&mask),
        ..Default::default()
    };
    let mut fluxes = Vec::new();
    let mut errors = Vec::new();
    for _ in 0..500 {
        let data: Vec<f64> = (0..WIDTH * HEIGHT)
            .map(|_| 100.0 + 3.0 * noise.normal())
            .collect();
        let result = aperture_photometry(&image(&data), &[position], &settings, &maps)
            .unwrap()
            .remove(0);
        assert_eq!(result.flags, Default::default());
        fluxes.push(result.flux.unwrap());
        errors.push(result.flux_error.unwrap());
    }
    let mean = fluxes.iter().sum::<f64>() / fluxes.len() as f64;
    let scatter =
        (fluxes.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / fluxes.len() as f64).sqrt();
    let error = errors.iter().sum::<f64>() / errors.len() as f64;
    assert!(
        (scatter / error - 1.0).abs() < 0.15,
        "reported sigma {error} does not match empirical sigma {scatter}"
    );
}

#[test]
fn sky_coverage_is_checked_after_clipping() {
    let mut noise = Noise(0x1234_5678_9abc_def1);
    let settings = options(6.0, 10.0, 16.0);
    let mut count = 0;
    let mut retained = 0;
    let mut mask = vec![0u8; WIDTH * HEIGHT];
    let data: Vec<f64> = (0..WIDTH * HEIGHT)
        .map(|i| {
            let dx = (i % WIDTH) as f64 - 48.0;
            let dy = (i / WIDTH) as f64 - 40.0;
            let r = dx.hypot(dy);
            let mut v = 100.0 + noise.normal();
            if (10.0..=16.0).contains(&r) {
                count += 1;
                if dx < -2.0 {
                    mask[i] = 1;
                } else {
                    retained += 1;
                    if i % 5 == 0 {
                        v += 1000.0;
                    }
                }
            }
            v
        })
        .collect();
    let maps = ApertureMaps {
        mask: Some(&mask),
        ..Default::default()
    };
    let result = aperture_photometry(&image(&data), &[[48.0, 40.0]], &settings, &maps)
        .unwrap()
        .remove(0);
    assert!(retained * 2 > count, "coverage must pass before clipping");
    assert!(
        result.flags.unusable,
        "less than half sky accepted without flag"
    );
    assert_eq!(
        (result.flux, result.flux_error, result.magnitude),
        (None, None, None)
    );
}

#[test]
fn sampled_photometry_reads_only_local_pixels_and_owns_its_snapshot() {
    use seiza_stacking::{AperturePhotometry, AperturePixel};
    let mut image = image(&field(&[([40.3, 30.7], 4000.0)], 1.6, |x, y| {
        100.0 + 0.2 * x - 0.1 * y
    }));
    let settings = options(6.0, 10.0, 16.0);
    let positions = [[40.3, 30.7], [2.3, 50.4], [-40.0, 20.0]];
    let mut mask = vec![0; WIDTH * HEIGHT];
    mask[31 * WIDTH + 41] = 1;
    let mut clipped = vec![0; WIDTH * HEIGHT];
    clipped[30 * WIDTH + 40] = 1;
    let uncertainty = vec![3.0; WIDTH * HEIGHT];
    let maps = ApertureMaps {
        mask: Some(&mask),
        saturation: Some(&clipped),
        uncertainty: Some(&uncertainty),
    };
    let expected = aperture_photometry(&image, &positions, &settings, &maps).unwrap();
    let mut reads = 0;
    let sampled = AperturePhotometry::sample(WIDTH, HEIGHT, &positions, &settings, |x, y| {
        assert!(x < WIDTH && y < HEIGHT);
        assert!(positions.iter().any(|p| {
            let d = (x as f64 - p[0]).hypot(y as f64 - p[1]);
            d < 7.0 || (10.0..=16.0).contains(&d)
        }));
        reads += 1;
        let index = y * WIDTH + x;
        AperturePixel {
            value: image.data[index],
            masked: mask[index] != 0,
            saturated: clipped[index] != 0,
            uncertainty: Some(uncertainty[index]),
        }
    })
    .unwrap();
    assert!(
        reads > 0 && reads < 1500,
        "read {reads} pixels for two overlapping image regions"
    );
    image.data.fill(f32::NAN);
    assert_eq!(sampled.measure(), expected);
    assert!(
        AperturePhotometry::sample(WIDTH, HEIGHT, &[[f64::NAN, 0.0]], &settings, |_, _| panic!(
            "invalid requests must not read pixels"
        ))
        .is_err()
    );
}
