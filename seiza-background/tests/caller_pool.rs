//! Background models render and correct in the Rayon pool their caller
//! installs.
//!
//! Every entry point here runs inside a one-thread and a two-thread pool,
//! and must give the same bits in both. The global pool must never start:
//! Rayon work that left the caller's pool would start it. That can be
//! checked once per process, and each file under `tests/` runs as a process
//! of its own, so this is one test.

use rayon::ThreadPoolBuilder;
use seiza_background::{
    BackgroundConfig, BackgroundFit, CorrectionMode, FittedModel, ModelConfig, fit_background,
    select_shared_model,
};

const WIDTH: usize = 256;
const HEIGHT: usize = 192;

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn bits64(values: impl IntoIterator<Item = f64>) -> Vec<u32> {
    values
        .into_iter()
        .flat_map(|value| {
            let bits = value.to_bits();
            [bits as u32, (bits >> 32) as u32]
        })
        .collect()
}

/// A tilted, vignetted sky per channel, with stars and a little noise,
/// interleaved.
fn sky() -> Vec<f32> {
    (0..WIDTH * HEIGHT * 3)
        .map(|sample| {
            let (pixel, channel) = (sample / 3, sample % 3);
            let (x, y) = (
                (pixel % WIDTH) as f32 / WIDTH as f32,
                (pixel / WIDTH) as f32 / HEIGHT as f32,
            );
            let gradient = 0.3 * x + (0.1 + 0.05 * channel as f32) * y;
            let vignette = 0.2 * ((x - 0.5).powi(2) + (y - 0.5).powi(2));
            let noise = ((pixel * 37 + channel * 11) % 23) as f32 * 0.002;
            let star = if (pixel * 7919).is_multiple_of(1009) {
                2.0
            } else {
                0.0
            };
            1.0 + gradient - vignette + noise + star
        })
        .collect()
}

fn fit_bits(fit: &BackgroundFit) -> Vec<u32> {
    let mut out = bits64(fit.reference.iter().copied());
    match &fit.model {
        FittedModel::Polynomial { coefficients, .. } => {
            out.extend(bits64(coefficients.iter().flatten().copied()));
        }
        FittedModel::RadialBasis { coefficients, .. } => {
            out.extend(bits64(coefficients.iter().flatten().copied()));
        }
        _ => unreachable!("an unknown model family"),
    }
    out.push(fit.diagnostics.accepted_samples as u32);
    out
}

/// Everything run inside `pool`, as the bits of each result.
fn run(pool: &rayon::ThreadPool) -> Vec<(&'static str, Vec<u32>)> {
    pool.install(|| {
        let mut results = Vec::new();
        let image = sky();

        // One model for all three channels, as a colour image is fitted.
        let fit = fit_background(&image, WIDTH, HEIGHT, 3, &BackgroundConfig::default()).unwrap();
        let mut out = fit_bits(&fit);
        out.extend(bits(&fit.render_model().unwrap()));
        out.extend(bits(
            &fit.correct(&image, CorrectionMode::Subtract).unwrap(),
        ));
        out.extend(bits(
            &fit.correct_with_strength(&image, CorrectionMode::Divide, 0.5)
                .unwrap(),
        ));
        results.push(("colour", out));

        // Each channel fitted alone, then refitted with the model they share.
        let automatic = ModelConfig::Automatic {
            max_degree: 2,
            ridge: 1.0e-8,
            rbf_smoothing: 0.01,
            max_control_points: 192,
            allow_radial_basis: true,
            minimum_improvement: 0.12,
        };
        let channels = (0..3)
            .map(|channel| {
                image
                    .iter()
                    .skip(channel)
                    .step_by(3)
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let config = BackgroundConfig {
            model: automatic.clone(),
            ..BackgroundConfig::default()
        };
        let fits = channels
            .iter()
            .map(|channel| fit_background(channel, WIDTH, HEIGHT, 1, &config).unwrap())
            .collect::<Vec<_>>();
        let shared = select_shared_model(&automatic, &fits.iter().collect::<Vec<_>>())
            .expect("every channel scored the same candidates");
        let config = BackgroundConfig {
            model: shared,
            ..BackgroundConfig::default()
        };
        let mut out = Vec::new();
        for (channel, data) in channels.iter().enumerate() {
            out.extend(fit_bits(&fits[channel]));
            let refit = fit_background(data, WIDTH, HEIGHT, 1, &config).unwrap();
            let mut corrected = data.clone();
            refit
                .correct_in_place(&mut corrected, CorrectionMode::Subtract)
                .unwrap();
            out.extend(fit_bits(&refit));
            out.extend(bits(&corrected));
        }
        results.push(("shared model", out));
        results
    })
}

#[test]
fn background_models_stay_in_the_callers_pool() {
    let pools = [1, 2].map(|threads| {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(move |index| format!("caller-{threads}-{index}"))
            .build()
            .unwrap()
    });
    let one = run(&pools[0]);
    let two = run(&pools[1]);
    assert_eq!(one.len(), two.len());
    for ((name, one), (_, two)) in one.iter().zip(&two) {
        assert!(one == two, "{name} differs between one and two threads");
    }
    ThreadPoolBuilder::new()
        .build_global()
        .expect("work escaped the caller's pool and started the global pool");
}
