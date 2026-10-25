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
