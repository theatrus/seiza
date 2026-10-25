use seiza_stacking::*;

fn sources(count: usize) -> ContinuumSources {
    let source = |name: &str| ContinuumSource {
        name: name.into(),
        revision: format!("{name}:linear-v1"),
        transfer: ContinuumTransfer::Linear,
        saturation: Some(1e6),
        clipping_revision: None,
    };
    ContinuumSources {
        width: 256,
        height: 256,
        narrowband: source("ha"),
        references: ["red", "green", "blue"]
            .iter()
            .take(count)
            .map(|n| source(n))
            .collect(),
    }
}
fn photometry(flux: f64, x: f64, y: f64) -> ContinuumPhotometry {
    ContinuumPhotometry {
        x,
        y,
        flux,
        peak: flux / 20.0 + 100.0,
        fwhm: 3.0,
        snr: 100.0,
    }
}
fn stars(coefficients: &[f64], count: usize) -> Vec<ContinuumStar> {
    (0..count)
        .map(|i| {
            let x = 25.0 + (i % 7) as f64 * 30.0;
            let y = 25.0 + (i / 7) as f64 * 15.0;
            let fluxes: Vec<_> = (0..coefficients.len())
                .map(|j| 1000.0 + ((i * (17 + 11 * j) + j * 83) % 157) as f64 * 50.0)
                .collect();
            ContinuumStar {
                narrowband: photometry(
                    fluxes.iter().zip(coefficients).map(|(f, a)| f * a).sum(),
                    x,
                    y,
                ),
                references: fluxes.into_iter().map(|f| photometry(f, x, y)).collect(),
            }
        })
        .collect()
}
fn plan(outcome: ContinuumFitOutcome) -> ContinuumPlan {
    match outcome {
        ContinuumFitOutcome::Fitted { plan } => plan,
        other => panic!("expected a fit: {other:?}"),
    }
}
fn manual(sources: &ContinuumSources, coefficients: Vec<f64>) -> ContinuumPlan {
    ContinuumPlan {
        psf_matching: None,
        schema_version: 1,
        sources: sources.clone(),
        coefficients,
        reference_flux_ratios: Vec::new(),
        origin: ContinuumOrigin::Manual,
        diagnostics: None,
    }
}
fn mono(values: Vec<f32>) -> LinearImage {
    LinearImage::new(values.len(), 1, 1, values).unwrap()
}

#[test]
fn robust_positive_fit_recovers_two_weights_and_rejects_stellar_outliers() {
    let context = sources(2);
    let mut data = stars(&[0.4, 0.2], 80);
    for (i, star) in data.iter_mut().enumerate() {
        if i % 17 == 0 {
            star.narrowband.flux *= 4.0;
        }
    }
    let result = plan(fit_continuum(&data, &context, &Default::default()).unwrap());
    assert_eq!(result.origin, ContinuumOrigin::Fitted);
    for (actual, expected) in result.coefficients.iter().zip([0.4, 0.2]) {
        assert!((actual - expected).abs() < 1e-10, "{result:?}");
    }
    let diagnostics = result.diagnostics.unwrap();
    assert!(diagnostics.used_stars >= 12);
    assert!(diagnostics.validation_stars >= 3);
    assert!(diagnostics.rejected[&ContinuumSkipReason::Outlier] >= 4);
    assert!(diagnostics.validation_fractional_rms.unwrap() < 1e-10);
}

#[test]
fn input_scale_changes_are_carried_by_the_fitted_weights_not_hidden_normalization() {
    let mut context = sources(1);
    let mut data = stars(&[0.35], 40);
    let initial = plan(fit_continuum(&data, &context, &Default::default()).unwrap());
    for star in &mut data {
        star.narrowband.flux *= 3.0;
        star.narrowband.peak *= 3.0;
        star.references[0].flux *= 7.0;
        star.references[0].peak *= 7.0;
    }
    context.narrowband.saturation = context.narrowband.saturation.map(|v| v * 3.0);
    context.references[0].saturation = context.references[0].saturation.map(|v| v * 7.0);
    context.narrowband.revision = "ha:scaled".into();
    context.references[0].revision = "red:scaled".into();
    let scaled = plan(fit_continuum(&data, &context, &Default::default()).unwrap());
    assert!((scaled.coefficients[0] - initial.coefficients[0] * 3.0 / 7.0).abs() < 1e-10);
    let n = mono(vec![0.2 * 3.0, 0.1 * 3.0, 0.5 * 3.0]);
    let b = mono(vec![0.5 * 7.0, 0.3 * 7.0, 0.4 * 7.0]);
    let mut grid = context.clone();
    grid.width = 3;
    grid.height = 1;
    let mut p = scaled.clone();
    p.sources = grid.clone();
    let out = subtract_continuum(&n, &[&b], &grid, &p).unwrap();
    for (v, e) in out.data.iter().zip([0.075, -0.015, 1.08]) {
        assert!((*v - e).abs() < 1e-6);
    }
    assert!(subtract_continuum(&n, &[&b], &grid, &initial).is_err());
}

#[test]
fn collinear_references_need_a_valid_single_reference_fallback() {
    let context = sources(2);
    let mut data = stars(&[0.35], 40);
    for star in &mut data {
        let mut second = star.references[0].clone();
        second.flux *= 2.0;
        second.peak *= 2.0;
        star.references.push(second);
    }
    let result = plan(fit_continuum(&data, &context, &Default::default()).unwrap());
    assert_eq!(result.origin, ContinuumOrigin::SingleReferenceFallback);
    assert_eq!(result.coefficients.iter().filter(|a| **a > 0.0).count(), 1);
    assert!((result.coefficients[0] + 2.0 * result.coefficients[1] - 0.35).abs() < 1e-10);
    assert!(matches!(
        fit_continuum(
            &data,
            &context,
            &ContinuumFitOptions {
                fallback_to_single: false,
                ..Default::default()
            }
        )
        .unwrap(),
        ContinuumFitOutcome::Skipped {
            reason: ContinuumSkipReason::RankDeficient,
            ..
        }
    ));
}

#[test]
fn bad_psf_alignment_saturation_and_missing_stars_never_produce_a_plan() {
    let context = sources(1);
    for (kind, reason) in [
        (0, ContinuumSkipReason::PsfMismatch),
        (1, ContinuumSkipReason::RegistrationMismatch),
        (2, ContinuumSkipReason::TooFewStars),
        (3, ContinuumSkipReason::TooFewStars),
    ] {
        let mut data = stars(&[0.35], 40);
        for star in &mut data {
            match kind {
                0 => star.references[0].fwhm *= 2.0,
                1 => star.references[0].x += 3.0,
                2 => star.narrowband.peak = 1e6,
                _ => star.narrowband.snr = 1.0,
            }
        }
        assert!(
            matches!(fit_continuum(&data,&context,&Default::default()).unwrap(),ContinuumFitOutcome::Skipped{reason:r,..} if r==reason)
        );
    }
    assert!(matches!(
        fit_continuum(&[], &context, &Default::default()).unwrap(),
        ContinuumFitOutcome::Skipped {
            reason: ContinuumSkipReason::NoStars,
            ..
        }
    ));
    let mut data = stars(&[0.35], 40);
    data[1].narrowband.flux = f64::NAN;
    let outcome = fit_continuum(&data, &context, &Default::default()).unwrap();
    assert!(serde_json::to_string(&outcome).is_ok());
}

#[test]
fn manual_subtraction_preserves_negative_samples_and_contributing_coverage() {
    let mut context = sources(2);
    context.width = 5;
    context.height = 1;
    let n = mono(vec![0.2, 0.1, 0.5, f32::NAN, -0.02]);
    let b = mono(vec![0.5, 0.3, 0.4, 1.0, 0.02]);
    let unused = mono(vec![f32::NAN; 5]);
    let output = subtract_continuum(
        &n,
        &[&b, &unused],
        &context,
        &manual(&context, vec![0.35, 0.0]),
    )
    .unwrap();
    for (i, e) in [(0, 0.025), (1, -0.005), (2, 0.36), (4, -0.027)] {
        assert!((output.data[i] - e).abs() < 1e-7);
    }
    assert!(output.data[3].is_nan());
    let missing = mono(vec![f32::NAN; 5]);
    let out = subtract_continuum(
        &n,
        &[&missing, &b],
        &context,
        &manual(&context, vec![0.35, 0.0]),
    )
    .unwrap();
    assert!(out.data.iter().all(|v| v.is_nan()));
}

#[test]
fn plans_reject_display_data_changed_revisions_reference_order_and_bad_coefficients() {
    let mut context = sources(2);
    context.width = 3;
    context.height = 1;
    let n = mono(vec![0.2, 0.1, 0.5]);
    let b = mono(vec![0.5, 0.3, 0.4]);
    let p = manual(&context, vec![0.3, 0.2]);
    let mut changed = context.clone();
    changed.references.swap(0, 1);
    assert!(subtract_continuum(&n, &[&b, &b], &changed, &p).is_err());
    changed = context.clone();
    changed.references[0].revision = "new-background".into();
    assert!(subtract_continuum(&n, &[&b, &b], &changed, &p).is_err());
    changed = context.clone();
    changed.references[0].transfer = ContinuumTransfer::Display;
    assert!(subtract_continuum(&n, &[&b, &b], &changed, &p).is_err());
    for coefficients in [vec![-1.0, 0.0], vec![f64::NAN, 0.0], vec![1.0]] {
        assert!(
            subtract_continuum(&n, &[&b, &b], &context, &manual(&context, coefficients)).is_err()
        );
    }
    let bad = ContinuumFitOptions {
        minimum_stars: 1,
        ..Default::default()
    };
    assert!(fit_continuum(&stars(&[0.3, 0.2], 40), &context, &bad).is_err());
}

#[test]
fn broadband_line_transmission_attenuates_the_signed_residual() {
    let mut context = sources(1);
    context.width = 3;
    context.height = 1;
    let continuum = [20.0, 40.0, 10.0];
    let emission = [5.0, -1.0, 3.0];
    let scale = 0.35;
    let beta = 0.2;
    let n = mono(
        continuum
            .iter()
            .zip(emission)
            .map(|(c, e)| scale * c + e)
            .collect(),
    );
    let b = mono(
        continuum
            .iter()
            .zip(emission)
            .map(|(c, e)| c + beta * e)
            .collect(),
    );
    let output = subtract_continuum(
        &n,
        &[&b],
        &context,
        &manual(&context, vec![f64::from(scale)]),
    )
    .unwrap();
    for (actual, e) in output.data.iter().zip(emission) {
        assert!((*actual - (1.0 - scale * beta) * e).abs() < 1e-6);
    }
}

fn field(sigma_n: f64, sigma_b: f64) -> (LinearImage, LinearImage, Vec<[f64; 2]>) {
    let positions: Vec<_> = (0..4)
        .flat_map(|y| (0..4).map(move |x| [32.0 + 48.0 * x as f64, 32.0 + 48.0 * y as f64]))
        .collect();
    let mut n = vec![0.0_f32; 256 * 256];
    let mut b = n.clone();
    for y in 0..256 {
        for x in 0..256 {
            let background = 300.0 + 0.15 * x as f64 + 0.1 * y as f64;
            let mut nb = 0.35 * background;
            let mut bb = background;
            for (i, p) in positions.iter().enumerate() {
                let radius = (x as f64 - p[0]).powi(2) + (y as f64 - p[1]).powi(2);
                let amplitude = 80.0 + 13.0 * i as f64;
                nb += 0.35 * amplitude * (-radius / (2.0 * sigma_n * sigma_n)).exp();
                bb += amplitude * (-radius / (2.0 * sigma_b * sigma_b)).exp();
            }
            let line = 100.0
                * (-((x as f64 - 220.0).powi(2) + (y as f64 - 220.0).powi(2)) / (2.0 * 7.0 * 7.0))
                    .exp();
            n[y * 256 + x] = (nb + line) as f32;
            b[y * 256 + x] = bb as f32;
        }
    }
    (
        LinearImage::new(256, 256, 1, n).unwrap(),
        LinearImage::new(256, 256, 1, b).unwrap(),
        positions,
    )
}

#[test]
fn aperture_photometry_fits_stars_across_gradients_and_retains_nebula() {
    let (n, b, positions) = field(1.4, 1.4);
    let context = sources(1);
    let result = plan(
        measure_and_fit_continuum(
            &n,
            &[&b],
            &context,
            &positions,
            &Default::default(),
            &Default::default(),
        )
        .unwrap(),
    );
    assert!((result.coefficients[0] - 0.35).abs() < 2e-5, "{result:?}");
    let output = subtract_continuum(&n, &[&b], &context, &result).unwrap();
    assert!((output.data[220 * 256 + 220] - 100.0).abs() < 0.02);
    assert!(output.data[32 * 256 + 32].abs() < 0.02);
    let automatic = plan(
        measure_and_fit_continuum(
            &n,
            &[&b],
            &context,
            &[],
            &Default::default(),
            &Default::default(),
        )
        .unwrap(),
    );
    assert!((automatic.coefficients[0] - 0.35).abs() < 2e-4);
}

#[test]
fn measured_psf_mismatch_and_partial_edge_apertures_are_refused() {
    let (n, b, positions) = field(1.4, 2.8);
    let context = sources(1);
    assert!(matches!(
        measure_and_fit_continuum(
            &n,
            &[&b],
            &context,
            &positions,
            &Default::default(),
            &Default::default()
        )
        .unwrap(),
        ContinuumFitOutcome::Skipped {
            reason: ContinuumSkipReason::PsfMismatch,
            ..
        }
    ));
    let outcome = measure_and_fit_continuum(
        &n,
        &[&b],
        &context,
        &[[235.9, 235.9]],
        &Default::default(),
        &Default::default(),
    )
    .unwrap();
    assert!(matches!(outcome, ContinuumFitOutcome::Skipped { .. }));
}

#[test]
fn a_fit_cannot_validate_itself_by_discarding_most_held_out_stars() {
    let context = sources(1);
    let mut data = stars(&[0.35], 80);
    for (i, star) in data.iter_mut().enumerate() {
        if i % 4 == 0 && i % 16 != 0 {
            star.narrowband.flux *= 2.0;
        }
    }
    assert!(matches!(
        fit_continuum(&data, &context, &Default::default()).unwrap(),
        ContinuumFitOutcome::Skipped {
            reason: ContinuumSkipReason::PoorFit,
            ..
        }
    ));
}

#[test]
fn automatic_fitting_refuses_brightness_dependent_and_extreme_flux_models() {
    let context = sources(1);
    let mut data = stars(&[0.35], 80);
    for (i, star) in data.iter_mut().enumerate() {
        star.narrowband.flux *= if i % 2 == 0 { 0.5 } else { 2.0 };
    }
    assert!(matches!(
        fit_continuum(&data, &context, &Default::default()).unwrap(),
        ContinuumFitOutcome::Skipped { .. }
    ));
    for star in &mut data {
        star.narrowband.flux = f64::MIN_POSITIVE;
        star.references[0].flux = f64::MAX;
    }
    assert!(matches!(
        fit_continuum(&data, &context, &Default::default()).unwrap(),
        ContinuumFitOutcome::Skipped { .. }
    ));
}

#[path = "common/continuum.rs"]
mod measurement_fixture;

#[test]
fn rejected_bright_stars_do_not_spend_the_usable_photometry_budget() {
    let summary = measurement_fixture::measurement_summary();
    assert_eq!(summary[0], 1.0);
    assert!((summary[1] - 0.35).abs() < 1e-6);
    let (n, b, positions, sources, options) = measurement_fixture::saturated_field();
    let result = plan(
        measure_and_fit_continuum(
            &n,
            &[&b],
            &sources,
            &positions,
            &options,
            &Default::default(),
        )
        .unwrap(),
    );
    let d = result.diagnostics.unwrap();
    assert_eq!(d.candidate_stars, 32);
    assert_eq!(d.eligible_stars, 16);
    assert_eq!(d.sampled_stars, 16);
    assert_eq!(d.rejected[&ContinuumSkipReason::Saturated], 16);
}

#[test]
fn candidate_search_reaches_valid_fainter_stars_independently_of_usable_budget() {
    let actual = measurement_fixture::deeper_candidate_summary();
    for (actual, expected) in actual.iter().zip([1.0, 0.35, 12.0, 4.0, 100.0]) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

use measurement_fixture as profile_fixture;

#[test]
fn profile_gate_rejects_aperture_success_and_matching_replays_exactly() {
    let (n, b, positions, sources, options) = profile_fixture::profile_field();
    let outcome = measure_and_fit_continuum(
        &n,
        &[&b],
        &sources,
        &positions,
        &options,
        &Default::default(),
    )
    .unwrap();
    assert!(matches!(
        outcome,
        ContinuumFitOutcome::Skipped {
            reason: ContinuumSkipReason::PixelResidual,
            ..
        }
    ));
    let photo = ContinuumPhotometryOptions {
        psf_matching: Some(ContinuumPsfOptions {
            target_fwhm: Some(5.0),
        }),
        ..Default::default()
    };
    let fitted =
        plan(measure_and_fit_continuum(&n, &[&b], &sources, &positions, &options, &photo).unwrap());
    let validation = fitted
        .diagnostics
        .as_ref()
        .unwrap()
        .pixel_validation
        .as_ref()
        .unwrap();
    assert!(validation.accepted, "{validation:?}");
    assert!(validation.p90_absolute_fraction < 0.05, "{validation:?}");
    let report = fitted.psf_matching.as_ref().unwrap();
    assert_eq!(report.target_fwhm, 5.0);
    assert!(report.kernel_sigma[1] > report.kernel_sigma[0]);
    assert_ne!(report.prepared_sources, sources);
    let residual = subtract_continuum(&n, &[&b], &sources, &fitted).unwrap();
    assert!(residual.data[0].is_nan());
    assert!((residual.data[30 * n.width + 30] - 6.5).abs() < 2.0);
    let serialized = serde_json::to_string(&fitted).unwrap();
    let restored: ContinuumPlan = serde_json::from_str(&serialized).unwrap();
    let replay = subtract_continuum(&n, &[&b], &sources, &restored).unwrap();
    assert!(
        residual
            .data
            .iter()
            .zip(replay.data)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    let mut inaccurate = restored.clone();
    inaccurate
        .psf_matching
        .as_mut()
        .unwrap()
        .measured_fwhm
        .fill(5.0);
    assert!(subtract_continuum(&n, &[&b], &sources, &inaccurate).is_err());
    let mut malformed = restored;
    malformed
        .psf_matching
        .as_mut()
        .unwrap()
        .kernel_radius
        .clear();
    assert!(subtract_continuum(&n, &[&b], &sources, &malformed).is_err());
}

#[test]
fn matching_never_fills_missing_coverage_or_accepts_sharpening() {
    let (mut n, b, positions, sources, options) = profile_fixture::profile_field();
    n.data[110 * n.width + 110] = f32::NAN;
    let photo = ContinuumPhotometryOptions {
        psf_matching: Some(ContinuumPsfOptions {
            target_fwhm: Some(5.0),
        }),
        ..Default::default()
    };
    let fitted =
        plan(measure_and_fit_continuum(&n, &[&b], &sources, &positions, &options, &photo).unwrap());
    let residual = subtract_continuum(&n, &[&b], &sources, &fitted).unwrap();
    assert!(residual.data[110 * n.width + 111].is_nan());
    let sharpen = ContinuumPhotometryOptions {
        psf_matching: Some(ContinuumPsfOptions {
            target_fwhm: Some(1.0),
        }),
        ..Default::default()
    };
    assert!(
        measure_and_fit_continuum(&n, &[&b], &sources, &positions, &options, &sharpen).is_err()
    );
}

#[test]
fn matching_cannot_fabricate_saturation_evidence() {
    let (n, b, positions, mut sources, options) = profile_fixture::profile_field();
    sources.references[0].saturation = None;
    let photo = ContinuumPhotometryOptions {
        psf_matching: Some(Default::default()),
        ..Default::default()
    };
    let error = measure_and_fit_continuum(&n, &[&b], &sources, &positions, &options, &photo)
        .unwrap_err()
        .to_string();
    assert!(error.contains("every original source"), "{error}");
}

#[test]
fn matching_dilates_producer_clipping_across_kernel_support() {
    let (n, b, positions, mut sources, options) = profile_fixture::saturated_field();
    let mut masks: Vec<Vec<u8>> = [&n, &b]
        .iter()
        .map(|image| image.data.iter().map(|v| u8::from(*v >= 1000.0)).collect())
        .collect();
    // Outside the fit aperture originally, but inside its broadened support.
    masks[0][positions[16][1] as usize * n.width + positions[16][0] as usize + 10] = 1;
    for source in std::iter::once(&mut sources.narrowband).chain(&mut sources.references) {
        source.saturation = None;
        source.clipping_revision = Some("producer-v1".into());
    }
    let photo = ContinuumPhotometryOptions {
        psf_matching: Some(ContinuumPsfOptions {
            target_fwhm: Some(5.0),
        }),
        ..Default::default()
    };
    let outcome = measure_and_fit_continuum_masked(
        &n,
        &[&b],
        &sources,
        &masks.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        &positions,
        &options,
        &photo,
    )
    .unwrap();
    let diagnostics = match outcome {
        ContinuumFitOutcome::Fitted { plan } => plan.diagnostics.unwrap(),
        ContinuumFitOutcome::Skipped { diagnostics, .. } => diagnostics,
    };
    assert!(
        diagnostics.rejected[&ContinuumSkipReason::Saturated] >= 17,
        "{diagnostics:?}"
    );
}

#[test]
fn gaussian_matching_does_not_excuse_extended_wing_residuals() {
    let (n, mut b, positions, sources, options) = profile_fixture::profile_field();
    b.data.fill(10.0);
    for (i, p) in positions.iter().enumerate() {
        let flux = (1500.0 + 50.0 * i as f64) / 0.35;
        for y in (p[1] as usize - 20)..=(p[1] as usize + 20) {
            for x in (p[0] as usize - 20)..=(p[0] as usize + 20) {
                let r2 = (x as f64 - p[0]).powi(2) + (y as f64 - p[1]).powi(2);
                let signal = flux
                    * (0.65 / (2.0 * std::f64::consts::PI * 1.4_f64.powi(2))
                        * (-r2 / (2.0 * 1.4_f64.powi(2))).exp()
                        + 0.35 / (2.0 * std::f64::consts::PI * 4.0_f64.powi(2))
                            * (-r2 / (2.0 * 4.0_f64.powi(2))).exp());
                b.data[y * b.width + x] += signal as f32;
            }
        }
    }
    let photo = ContinuumPhotometryOptions {
        psf_matching: Some(Default::default()),
        ..Default::default()
    };
    let outcome =
        measure_and_fit_continuum(&n, &[&b], &sources, &positions, &options, &photo).unwrap();
    assert!(
        matches!(
            outcome,
            ContinuumFitOutcome::Skipped {
                reason: ContinuumSkipReason::PixelResidual,
                ..
            }
        ),
        "{outcome:?}"
    );
}

#[test]
fn clipped_core_artifacts_cannot_hide_behind_a_clean_unsaturated_fit() {
    let (n, mut b, positions, sources, options) = profile_fixture::saturated_field();
    for value in &mut b.data {
        *value = value.min(1000.0);
    }
    let outcome = measure_and_fit_continuum(
        &n,
        &[&b],
        &sources,
        &positions,
        &options,
        &Default::default(),
    )
    .unwrap();
    let ContinuumFitOutcome::Skipped {
        reason,
        diagnostics,
        ..
    } = outcome
    else {
        panic!("clipped cores were accepted")
    };
    assert_eq!(reason, ContinuumSkipReason::PixelResidual);
    let report = diagnostics.pixel_validation.unwrap();
    assert!(
        report.p90_absolute_fraction < 0.01,
        "unsaturated stars should fit: {report:?}"
    );
    assert!(
        report.clipped_stars > 0 && report.clipped_max_absolute_fraction.unwrap() > report.limit,
        "{report:?}"
    );
}

#[test]
fn a_faint_nb_measurement_cannot_conceal_a_clipped_reference() {
    let (mut n, mut b, positions, sources, options) = profile_fixture::saturated_field();
    for p in positions.iter().take(16) {
        for y in (p[1] as usize - 12)..=(p[1] as usize + 12) {
            for x in (p[0] as usize - 12)..=(p[0] as usize + 12) {
                n.data[y * n.width + x] = 10.0;
            }
        }
    }
    for value in &mut b.data {
        *value = value.min(1000.0);
    }
    let outcome = measure_and_fit_continuum(
        &n,
        &[&b],
        &sources,
        &positions,
        &options,
        &Default::default(),
    )
    .unwrap();
    let ContinuumFitOutcome::Skipped {
        reason,
        diagnostics,
        ..
    } = outcome
    else {
        panic!("Unmeasurable clipped reference was accepted")
    };
    assert_eq!(reason, ContinuumSkipReason::PixelResidual);
    let report = diagnostics.pixel_validation.unwrap();
    assert_eq!(report.clipped_stars, 16);
    assert_eq!(report.clipped_profiles, 0);
}
