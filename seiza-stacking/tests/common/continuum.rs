use seiza_stacking::*;

pub fn saturated_field() -> (
    LinearImage,
    LinearImage,
    Vec<[f64; 2]>,
    ContinuumSources,
    ContinuumFitOptions,
) {
    let (width, height) = (400, 240);
    let positions: Vec<_> = (0..32)
        .map(|i| [30.0 + (i % 8) as f64 * 45.0, 30.0 + (i / 8) as f64 * 50.0])
        .collect();
    let mut n = vec![10.0f32; width * height];
    let mut b = n.clone();
    for (i, p) in positions.iter().enumerate() {
        let amplitude = if i < 16 { 5000.0 } else { 70.0 + i as f64 };
        for y in (p[1] as usize - 12)..=(p[1] as usize + 12) {
            for x in (p[0] as usize - 12)..=(p[0] as usize + 12) {
                let v = amplitude
                    * (-((x as f64 - p[0]).powi(2) + (y as f64 - p[1]).powi(2))
                        / (2.0 * 1.4_f64.powi(2)))
                    .exp();
                n[y * width + x] += v as f32;
                b[y * width + x] += (v / 0.35) as f32;
            }
        }
    }
    let source = |name: &str| ContinuumSource {
        name: name.into(),
        revision: "fixture-v1".into(),
        transfer: ContinuumTransfer::Linear,
        saturation: Some(1000.0),
        clipping_revision: None,
    };
    let sources = ContinuumSources {
        width,
        height,
        narrowband: source("ha"),
        references: vec![source("red")],
    };
    (
        LinearImage::new(width, height, 1, n).unwrap(),
        LinearImage::new(width, height, 1, b).unwrap(),
        positions,
        sources,
        ContinuumFitOptions {
            max_stars: 16,
            ..Default::default()
        },
    )
}

pub fn measurement_summary() -> Vec<f64> {
    let (n, b, positions, sources, options) = saturated_field();
    match measure_and_fit_continuum(
        &n,
        &[&b],
        &sources,
        &positions,
        &options,
        &Default::default(),
    )
    .unwrap()
    {
        ContinuumFitOutcome::Fitted { plan } => {
            let d = plan.diagnostics.unwrap();
            vec![
                1.0,
                plan.coefficients[0],
                d.used_stars as f64,
                d.validation_stars as f64,
                d.rejected
                    .get(&ContinuumSkipReason::Saturated)
                    .copied()
                    .unwrap_or_default() as f64,
            ]
        }
        ContinuumFitOutcome::Skipped { diagnostics: d, .. } => vec![
            0.0,
            0.0,
            0.0,
            0.0,
            d.rejected
                .get(&ContinuumSkipReason::Saturated)
                .copied()
                .unwrap_or_default() as f64,
        ],
    }
}

/// Valid stars lie past the former four-times-usable-budget search cutoff.
pub fn deeper_candidate_summary() -> Vec<f64> {
    let width = 480;
    let height = 520;
    let positions: Vec<_> = (0..100)
        .map(|i| [30.0 + (i % 10) as f64 * 45.0, 30.0 + (i / 10) as f64 * 50.0])
        .collect();
    let mut n = vec![10.0_f32; width * height];
    let mut b = n.clone();
    for (i, p) in positions.iter().enumerate() {
        let amplitude = if i < 80 { 5000.0 } else { 70.0 + i as f64 };
        for y in (p[1] as usize - 12)..=(p[1] as usize + 12) {
            for x in (p[0] as usize - 12)..=(p[0] as usize + 12) {
                let v = amplitude
                    * (-((x as f64 - p[0]).powi(2) + (y as f64 - p[1]).powi(2))
                        / (2.0 * 1.4_f64.powi(2)))
                    .exp();
                n[y * width + x] += v as f32;
                b[y * width + x] += (v / 0.35) as f32;
            }
        }
    }
    let (_, _, _, mut sources, options) = saturated_field();
    sources.width = width;
    sources.height = height;
    let n = LinearImage::new(width, height, 1, n).unwrap();
    let b = LinearImage::new(width, height, 1, b).unwrap();
    match measure_and_fit_continuum(
        &n,
        &[&b],
        &sources,
        &positions,
        &options,
        &Default::default(),
    )
    .unwrap()
    {
        ContinuumFitOutcome::Fitted { plan } => {
            let d = plan.diagnostics.unwrap();
            vec![
                1.0,
                plan.coefficients[0],
                d.used_stars as f64,
                d.validation_stars as f64,
                d.candidate_stars as f64,
            ]
        }
        ContinuumFitOutcome::Skipped { diagnostics: d, .. } => {
            vec![0.0, 0.0, 0.0, 0.0, d.candidate_stars as f64]
        }
    }
}
