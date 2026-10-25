use seiza_stacking::*;
fn image(channels: usize, data: Vec<f32>) -> LinearImage {
    LinearImage::new(data.len() / channels, 1, channels, data).unwrap()
}
fn line() -> EmissionBlendLine {
    let source = |name: &str| ContinuumSource {
        name: name.into(),
        revision: "exact-v1".into(),
        transfer: ContinuumTransfer::Linear,
        saturation: None,
        clipping_revision: None,
    };
    EmissionBlendLine {
        plan: ContinuumPlan {
            psf_matching: None,
            schema_version: 1,
            sources: ContinuumSources {
                width: 2,
                height: 1,
                narrowband: source("oiii"),
                references: vec![source("green"), source("blue")],
            },
            coefficients: vec![0.2, 0.1],
            reference_flux_ratios: vec![1.0, 2.0],
            origin: ContinuumOrigin::Manual,
            diagnostics: None,
        },
        reference_channels: vec![1, 2],
        amount: 0.5,
        background_offset: 0.0,
    }
}
#[test]
fn projection_commutes_with_independent_physical_input_gains() {
    let base = image(3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    let residual = image(1, vec![0.04, -0.02]);
    let initial = line();
    let out = blend_emission_rgb(
        &base,
        &[&residual],
        std::slice::from_ref(&initial),
        &Default::default(),
    )
    .unwrap();
    assert!((out.data[1] - 0.25).abs() < 1e-6);
    assert!((out.data[2] - 0.4).abs() < 1e-6);
    assert!(out.data[4] < base.data[4]);
    let gains = [3.0_f32, 7.0, 11.0];
    let nb_gain = 5.0;
    let scaled_base = image(
        3,
        base.data
            .iter()
            .enumerate()
            .map(|(i, v)| v * gains[i % 3])
            .collect(),
    );
    let scaled_residual = image(1, residual.data.iter().map(|v| v * nb_gain).collect());
    let mut scaled = initial.clone();
    scaled.plan.coefficients = vec![
        0.2 * f64::from(nb_gain / gains[1]),
        0.1 * f64::from(nb_gain / gains[2]),
    ];
    scaled.plan.reference_flux_ratios = vec![1.0, 2.0 * f64::from(gains[2] / gains[1])];
    let actual = blend_emission_rgb(
        &scaled_base,
        &[&scaled_residual],
        &[scaled],
        &Default::default(),
    )
    .unwrap();
    for (i, (a, b)) in actual.data.iter().zip(out.data).enumerate() {
        assert!((a - b * gains[i % 3]).abs() < 2e-6);
    }
    let positive = blend_emission_rgb(
        &base,
        &[&residual],
        &[initial],
        &EmissionBlendOptions {
            negative: EmissionNegativePolicy::PositiveContribution,
        },
    )
    .unwrap();
    assert_eq!(&positive.data[3..], &base.data[3..]);
    assert!(residual.data[1] < 0.0);
}
#[test]
fn coverage_endpoints_and_calibration_offsets_are_explicit() {
    let n = image(3, vec![0.1, 0.2, 0.3, f32::NAN, 0.1, 0.2]);
    let b = image(3, vec![f32::NAN, 0.4, 0.5, 0.6, 0.7, 0.8]);
    let zero = blend_display_structure(&n, &b, 0.0).unwrap();
    assert_eq!(&zero.data[..3], &n.data[..3]);
    let one = blend_display_structure(&n, &b, 1.0).unwrap();
    assert_eq!(&one.data[3..], &b.data[3..]);
    assert!(
        blend_display_structure(&n, &b, 0.5)
            .unwrap()
            .data
            .iter()
            .all(|v| v.is_nan())
    );
    let base = image(3, vec![0.1, 0.2, 0.3]);
    let delta = image(3, vec![0.01, 0.02, 0.03]);
    let out = add_rgb_contribution(&base, &delta, [2.0, 3.0, 4.0]).unwrap();
    for (a, e) in out.data.iter().zip([0.12, 0.26, 0.42]) {
        assert!((a - e).abs() < 1e-6);
    }
    assert!(add_rgb_contribution(&base, &delta, [0.0, 1.0, 1.0]).is_err());
}
#[test]
fn invalid_units_and_empty_scales_do_not_produce_a_blend() {
    let base = image(3, vec![0.1; 6]);
    let residual = image(1, vec![0.01; 2]);
    let mut bad = line();
    bad.plan.reference_flux_ratios.clear();
    assert!(blend_emission_rgb(&base, &[&residual], &[bad], &Default::default()).is_err());
    let mut bad = line();
    bad.reference_channels = vec![1, 1];
    assert!(blend_emission_rgb(&base, &[&residual], &[bad], &Default::default()).is_err());
    let mut off = line();
    off.amount = 0.0;
    let missing = image(1, vec![f32::NAN; 2]);
    assert_eq!(
        blend_emission_rgb(&base, &[&missing], &[off], &Default::default())
            .unwrap()
            .data,
        base.data
    );
}

#[test]
fn green_reduction_preserves_neutrals_red_blue_and_is_transactional() {
    let mut image = LinearImage::new(
        4,
        1,
        3,
        vec![
            0.2,
            0.8,
            0.3,
            0.4,
            0.4,
            0.4,
            0.7,
            0.3,
            0.2,
            f32::NAN,
            0.2,
            0.3,
        ],
    )
    .unwrap();
    let original = image.clone();
    assert!(reduce_green_excess(&mut image, 1.1).is_err());
    assert!(
        image
            .data
            .iter()
            .zip(&original.data)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    reduce_green_excess(&mut image, 0.5).unwrap();
    assert!((image.data[1] - 0.55).abs() < 1e-6);
    assert_eq!(&image.data[3..9], &original.data[3..9]);
    assert!(image.data[9..].iter().all(|v| v.is_nan()));
}

fn flux_fixture() -> (Vec<ContinuumPlan>, EmissionFluxCalibration) {
    let mut a = line().plan;
    a.sources.narrowband.name = "ha".into();
    let b = line().plan;
    let cal = EmissionFluxCalibration {
        revision: "laboratory-response-and-gain-v1".into(),
        unit: "erg/s/cm2/pixel".into(),
        lines: vec!["ha".into(), "oiii".into()],
        sources: vec![a.sources.clone(), b.sources.clone()],
        narrowband_response: vec![vec![2.3, 0.25], vec![0.4, 3.3]],
        continuum_coefficients: vec![vec![0.2, 0.1]; 2],
        residual_zero_levels: vec![0.0; 2],
        reference_response: vec![
            vec![vec![1.0, 0.5], vec![1.0, 0.5]],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        ],
        residual_covariance: Some(vec![vec![4.0, 1.0], vec![1.0, 9.0]]),
    };
    (vec![a, b], cal)
}

#[test]
fn coupled_flux_inversion_retains_sign_coverage_and_gain_units() {
    let (plans, cal) = flux_fixture();
    // Effective M = [[2,.1],[.1,3]], true flux = [5,-2].
    let a = image(1, vec![9.8, f32::NAN]);
    let b = image(1, vec![-5.5, 1.0]);
    let (flux, report) = solve_emission_flux(&[&a, &b], &plans, &cal).unwrap();
    assert!((flux[0].data[0] - 5.0).abs() < 1e-5);
    assert!((flux[1].data[0] + 2.0).abs() < 1e-5);
    assert!(flux.iter().all(|image| image.data[1].is_nan()));
    assert!(report.flux_covariance.as_ref().unwrap()[0][0] > 0.0);
    // Multiply NB and reference sample units independently; physical flux stays fixed.
    let mut scaled = plans.clone();
    let mut calibration = cal.clone();
    let gains = [7.0, 0.25];
    let ref_gains = [[2.0, 11.0], [4.0, 3.0]];
    for i in 0..2 {
        for (j, reference_gain) in ref_gains[i].iter().enumerate() {
            scaled[i].coefficients[j] *= gains[i] / reference_gain;
            for k in 0..2 {
                calibration.reference_response[i][j][k] *= ref_gains[i][j];
            }
        }
        for k in 0..2 {
            calibration.narrowband_response[i][k] *= gains[i];
        }
    }
    for (i, plan) in scaled.iter_mut().enumerate() {
        plan.sources.narrowband.revision = format!("scaled-nb-{i}");
        for (j, source) in plan.sources.references.iter_mut().enumerate() {
            source.revision = format!("scaled-ref-{i}-{j}");
        }
        calibration.sources[i] = plan.sources.clone();
    }
    calibration.continuum_coefficients = scaled
        .iter()
        .map(|plan| plan.coefficients.clone())
        .collect();
    for i in 0..2 {
        for j in 0..2 {
            calibration.residual_covariance.as_mut().unwrap()[i][j] *= gains[i] * gains[j];
        }
    }
    let a = image(1, vec![9.8 * gains[0] as f32, f32::NAN]);
    let b = image(1, vec![-5.5 * gains[1] as f32, 1.0]);
    let (other, other_report) = solve_emission_flux(&[&a, &b], &scaled, &calibration).unwrap();
    for i in 0..2 {
        assert!((other[i].data[0] - flux[i].data[0]).abs() < 1e-5);
        for j in 0..2 {
            assert!(
                (report.flux_covariance.as_ref().unwrap()[i][j]
                    - other_report.flux_covariance.as_ref().unwrap()[i][j])
                    .abs()
                    < 1e-10
            );
        }
    }
}

#[test]
fn flux_calibration_refuses_ambiguous_unstable_and_mismatched_models() {
    let (plans, cal) = flux_fixture();
    let a = image(1, vec![1.0, 2.0]);
    let mut bad = cal.clone();
    bad.sources[0].narrowband.revision = "stale".into();
    assert!(solve_emission_flux(&[&a, &a], &plans, &bad).is_err());
    let mut bad = cal.clone();
    bad.narrowband_response[0][0] = 0.3;
    assert!(solve_emission_flux(&[&a, &a], &plans, &bad).is_err());
    let mut bad = cal.clone();
    bad.narrowband_response = vec![vec![1.3, 1.15], vec![1.3, 1.3]];
    assert!(solve_emission_flux(&[&a, &a], &plans, &bad).is_err());
    let mut bad = cal.clone();
    bad.residual_covariance = Some(vec![vec![1.0, 2.0], vec![2.0, 1.0]]);
    assert!(solve_emission_flux(&[&a, &a], &plans, &bad).is_err());
    let mut unknown = cal;
    unknown.residual_covariance = None;
    assert!(
        solve_emission_flux(&[&a, &a], &plans, &unknown)
            .unwrap()
            .1
            .flux_covariance
            .is_none()
    );
}

#[test]
fn flux_calibration_rejects_overflow_and_negative_tiny_variances() {
    let (plans, cal) = flux_fixture();
    let a = image(1, vec![1.0, 2.0]);
    let mut overflow = cal.clone();
    overflow.reference_response[0][0][1] = f64::MAX;
    overflow.reference_response[0][1][1] = f64::MAX;
    let mut extreme = plans.clone();
    extreme[0].coefficients = vec![2.0, 2.0];
    overflow.continuum_coefficients = extreme
        .iter()
        .map(|plan| plan.coefficients.clone())
        .collect();
    assert!(solve_emission_flux(&[&a, &a], &extreme, &overflow).is_err());
    let mut tiny = cal.clone();
    tiny.residual_covariance = Some(vec![vec![-1e-12, 0.0], vec![0.0, 1e-12]]);
    assert!(solve_emission_flux(&[&a, &a], &plans, &tiny).is_err());
    let mut overflowing_cov = cal;
    overflowing_cov.narrowband_response = vec![vec![0.3001, 0.15], vec![0.3, 0.3001]];
    overflowing_cov.residual_covariance = Some(vec![vec![f64::MAX, 0.0], vec![0.0, f64::MAX]]);
    assert!(solve_emission_flux(&[&a, &a], &plans, &overflowing_cov).is_err());
}

#[test]
fn malformed_public_images_return_errors_instead_of_panicking() {
    let (plans, cal) = flux_fixture();
    let malformed = LinearImage {
        width: 2,
        height: 1,
        channels: 1,
        data: vec![1.0],
    };
    assert!(solve_emission_flux(&[&malformed, &malformed], &plans, &cal).is_err());
    let mut malformed = LinearImage {
        width: 1,
        height: 1,
        channels: 3,
        data: vec![0.2, 0.3],
    };
    assert!(reduce_green_excess(&mut malformed, 0.5).is_err());
}

#[test]
fn response_inversion_survives_extreme_finite_independent_gain_units() {
    let (plans, mut cal) = flux_fixture();
    let a = image(1, vec![1.0, 2.0]);
    cal.reference_response = vec![vec![vec![0.0; 2]; 2]; 2];
    cal.narrowband_response = vec![vec![1e160, 0.0], vec![0.0, 1e160]];
    cal.residual_covariance = Some(vec![vec![1e308, 0.0], vec![0.0, 1e308]]);
    let (_, report) = solve_emission_flux(&[&a, &a], &plans, &cal).unwrap();
    assert!((report.flux_covariance.as_ref().unwrap()[0][0] / 1e-12 - 1.0).abs() < 1e-10);
    cal.narrowband_response = vec![vec![1e-160, 0.0], vec![0.0, 1e-160]];
    cal.residual_covariance = None;
    // The inverse is valid; overflowing the requested float32 flux is refused explicitly.
    let error = solve_emission_flux(&[&a, &a], &plans, &cal)
        .unwrap_err()
        .to_string();
    assert!(error.contains("float32"), "{error}");
}

#[test]
fn calibrated_zero_levels_remove_only_explicit_instrumental_pedestals() {
    let (plans, mut calibration) = flux_fixture();
    calibration.residual_zero_levels = vec![10.0, -4.0];
    let a = image(1, vec![19.8, 10.0]);
    let b = image(1, vec![-9.5, -4.0]);
    let (flux, report) = solve_emission_flux(&[&a, &b], &plans, &calibration).unwrap();
    assert!((flux[0].data[0] - 5.0).abs() < 1e-5);
    assert!((flux[1].data[0] + 2.0).abs() < 1e-5);
    assert_eq!(flux[0].data[1], 0.0);
    assert_eq!(flux[1].data[1], 0.0);
    assert_eq!(report.calibration.residual_zero_levels, vec![10.0, -4.0]);
    calibration.residual_zero_levels.clear();
    assert!(solve_emission_flux(&[&a, &b], &plans, &calibration).is_err());
}

#[test]
fn linked_star_reduction_preserves_color_coverage_and_the_zero_path() {
    let (width, height) = (7, 7);
    let base = [0.1_f32, 0.2, 0.3];
    let nebula = LinearImage::new(width, height, 3, base.repeat(width * height)).unwrap();
    let starless = LinearImage::new(width, height, 3, vec![0.2; width * height * 3]).unwrap();
    let mut data = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let value = 0.8 * (-((x as f32 - 3.0).powi(2) + (y as f32 - 3.0).powi(2)) / 4.0).exp();
            for color in [1.0, 0.5, 0.25] {
                data.push(1.0 - 0.8 * (1.0 - value * color));
            }
        }
    }
    let original = LinearImage::new(width, height, 3, data).unwrap();
    let mut legacy = nebula.clone();
    screen_rgb_stars(&mut legacy, &original, &starless).unwrap();
    let mut zero = nebula.clone();
    screen_rgb_stars_reduced(&mut zero, &original, &starless, 0.0).unwrap();
    assert_eq!(legacy.data, zero.data);
    let mut reduced = nebula.clone();
    screen_rgb_stars_reduced(&mut reduced, &original, &starless, 0.5).unwrap();
    for (i, (old, new)) in legacy.data.iter().zip(&reduced.data).enumerate() {
        assert!(*new <= *old + 1e-6 && *new >= base[i % 3] - 1e-6);
    }
    let center = (3 * width + 3) * 3;
    assert!(reduced.data[center] < legacy.data[center]);
    let layer: Vec<_> = (0..3)
        .map(|c| (reduced.data[center + c] - base[c]) / (1.0 - base[c]))
        .collect();
    assert!((layer[1] / layer[0] - 0.5).abs() < 1e-6);
    assert!((layer[2] / layer[0] - 0.25).abs() < 1e-6);
    assert!(screen_rgb_stars_reduced(&mut reduced, &original, &starless, 1.1).is_err());
    let mut original = original;
    original.data[(3 * width + 2) * 3] = f32::NAN;
    let mut masked = nebula;
    screen_rgb_stars_reduced(&mut masked, &original, &starless, 1.0).unwrap();
    assert!(
        masked.data[(3 * width + 2) * 3..(3 * width + 2) * 3 + 3]
            .iter()
            .all(|v| v.is_nan())
    );
    assert_eq!(
        &masked.data[center..center + 3],
        &legacy.data[center..center + 3]
    );
}

#[test]
fn residual_zero_calibration_cannot_survive_an_unacknowledged_model_edit() {
    let (mut plans, mut calibration) = flux_fixture();
    let a = image(1, vec![1.0, 2.0]);
    plans[0].coefficients[0] = 0.21;
    let error = solve_emission_flux(&[&a, &a], &plans, &calibration)
        .unwrap_err()
        .to_string();
    assert!(error.contains("coefficients changed"), "{error}");
    calibration.continuum_coefficients[0] = plans[0].coefficients.clone();
    calibration.residual_zero_levels[0] = 0.1;
    assert!(solve_emission_flux(&[&a, &a], &plans, &calibration).is_ok());
}
