//! Holds [`seiza_pipeline::OUTPUT_VERSION`] to the output it promises.
//!
//! Each case combines fixed synthetic partial stacks and compares a summary
//! of the result with the one recorded here. A change that moves a summary
//! changes what callers' stored tiles would hold: set `OUTPUT_VERSION` to
//! the release that ships the change, and record the new summary, which the
//! failure prints.

use seiza_pipeline::{Digest, Estimator, Identity, PartialStack, StackKind, WeightSource};
use seiza_stacking::LinearImage;

const WIDTH: usize = 48;
const HEIGHT: usize = 40;

/// Relative to the larger of the recorded value and 1.
const TOLERANCE: f64 = 1e-5;

#[test]
fn output_version_is_no_newer_than_the_next_minor_version() {
    let parse = |version: &str| {
        version
            .split('.')
            .map(|part| part.parse::<u64>().unwrap())
            .collect::<Vec<_>>()
    };
    let output = parse(seiza_pipeline::OUTPUT_VERSION);
    assert_eq!(output.len(), 3, "{}", seiza_pipeline::OUTPUT_VERSION);
    // A PR that changes output sets the crate's next minor version, and the
    // release that ships the change bumps the crate to it.
    let version = parse(env!("CARGO_PKG_VERSION"));
    let next_minor = vec![version[0], version[1] + 1, 0];
    assert!(
        output <= next_minor,
        "{} is past {}'s next minor",
        seiza_pipeline::OUTPUT_VERSION,
        env!("CARGO_PKG_VERSION")
    );
}

#[test]
fn combined_tiles_keep_their_output() {
    let mut failures = Vec::new();
    for (case, channels) in [("COMBINE_MONO", 1), ("COMBINE_RGB", 3)] {
        let stacks = (0..4)
            .map(|index| partial(index, channels))
            .collect::<Vec<_>>();
        let tile = PartialStack::combine(&stacks).unwrap();
        let mut actual = summary(&tile.mean.data, channels);
        actual.extend(summary(&tile.weight, channels));
        actual.extend(summary(tile.variance.as_deref().unwrap(), channels));
        let coverage = tile
            .coverage
            .iter()
            .map(|&count| count as f32)
            .collect::<Vec<_>>();
        actual.extend(summary(&coverage, channels));
        if let Some(failure) = compare(case, &actual, expected(case)) {
            failures.push(failure);
        }
    }
    assert!(
        failures.is_empty(),
        "combined output changed; set OUTPUT_VERSION to the release that ships the change \
         and record the new output:\n{}",
        failures.join("\n")
    );
}

#[rustfmt::skip]
fn expected(case: &str) -> &'static [f64] {
    match case {
        "COMBINE_MONO" => &[
            1.2311143e2, 1.3805135e2, 1.0129019e3, 1.0048203e2, 1.0844661e2, 1.1328358e2,
            1.1935239e2, 1.0525615e2, 1.5479333e2, 1.1632328e2, 1.2126694e2, 1.0681071e2,
            1.1321704e2, 1.1926091e2, 1.2421773e2, 1.0898203e2, 1.1694661e2, 1.2178358e2,
            1.2785239e2, 2.8040740e0, 2.8162258e0, 3.0987654e0, 2.4054322e0, 2.7511110e0,
            2.9609876e0, 3.0987654e0, 2.4054322e0, 2.7511110e0, 2.9609876e0, 3.0987654e0,
            2.4054322e0, 2.7511110e0, 2.9609876e0, 3.0987654e0, 2.4054322e0, 2.7511110e0,
            2.9609876e0, 3.0987654e0, 9.1581474e0, 9.3242447e0, 1.1314543e1, 1.0674321e1,
            1.0886021e1, 9.1826353e0, 5.7793241e0, 9.7030497e0, 1.0417053e1, 1.0134030e1,
            6.8354754e0, 1.0945094e1, 1.0253194e1, 9.0953083e0, 5.9397693e0, 1.0674321e1,
            1.0886021e1, 9.1826353e0, 5.7793241e0, 1.8000000e1, 1.8138357e1, 2.1000000e1,
            2.1000000e1, 1.9000000e1, 1.7000000e1, 1.5000000e1, 2.1000000e1, 1.9000000e1,
            1.7000000e1, 1.5000000e1, 2.1000000e1, 1.9000000e1, 1.7000000e1, 1.5000000e1,
            2.1000000e1, 1.9000000e1, 1.7000000e1, 1.5000000e1,
        ],
        "COMBINE_RGB" => &[
            1.2311143e2, 1.3805135e2, 1.0129019e3, 1.5311008e2, 1.6536095e2, 1.0412007e3,
            1.8311012e2, 1.9347065e2, 1.0724244e3, 1.0048203e2, 1.3192783e2, 1.6337363e2,
            1.0844661e2, 1.3674548e2, 1.6796922e2, 1.1328358e2, 1.4312598e2, 1.7286632e2,
            1.1935239e2, 1.4931296e2, 1.7787152e2, 1.0525615e2, 1.3513142e2, 1.6323203e2,
            1.5479333e2, 1.8478122e2, 2.1600497e2, 1.1632328e2, 1.4606361e2, 1.7603360e2,
            1.2126694e2, 1.5104462e2, 1.8210239e2, 1.0681071e2, 1.3656035e2, 1.6800615e2,
            1.1321704e2, 1.4299895e2, 1.7273965e2, 1.1926091e2, 1.4795503e2, 1.7907327e2,
            1.2421773e2, 1.5527550e2, 1.8401694e2, 1.0898203e2, 1.4042783e2, 1.7187363e2,
            1.1694661e2, 1.4524548e2, 1.7646922e2, 1.2178358e2, 1.5162598e2, 1.8136632e2,
            1.2785239e2, 1.5781296e2, 1.8637152e2, 2.8040740e0, 2.8162258e0, 3.0987654e0,
            2.8040740e0, 2.8162258e0, 3.0987654e0, 2.8040740e0, 2.8162258e0, 3.0987654e0,
            2.4054322e0, 2.4054322e0, 2.4054322e0, 2.7511110e0, 2.7511110e0, 2.7511110e0,
            2.9609876e0, 2.9609876e0, 2.9609876e0, 3.0987654e0, 3.0987654e0, 3.0987654e0,
            2.4054322e0, 2.4054322e0, 2.4054322e0, 2.7511110e0, 2.7511110e0, 2.7511110e0,
            2.9609876e0, 2.9609876e0, 2.9609876e0, 3.0987654e0, 3.0987654e0, 3.0987654e0,
            2.4054322e0, 2.4054322e0, 2.4054322e0, 2.7511110e0, 2.7511110e0, 2.7511110e0,
            2.9609876e0, 2.9609876e0, 2.9609876e0, 3.0987654e0, 3.0987654e0, 3.0987654e0,
            2.4054322e0, 2.4054322e0, 2.4054322e0, 2.7511110e0, 2.7511110e0, 2.7511110e0,
            2.9609876e0, 2.9609876e0, 2.9609876e0, 3.0987654e0, 3.0987654e0, 3.0987654e0,
            9.1581474e0, 9.3242447e0, 1.1314543e1, 9.1585032e0, 9.3245901e0, 1.1314540e1,
            9.1582117e0, 9.3243091e0, 1.1314548e1, 1.0674321e1, 9.7986908e0, 1.0016243e1,
            1.0886021e1, 1.0839963e1, 9.9236946e0, 9.1826353e0, 9.8477049e0, 9.6432323e0,
            5.7793241e0, 6.6412821e0, 6.4206138e0, 9.7030497e0, 1.0650864e1, 1.0674321e1,
            1.0417053e1, 9.8453102e0, 1.0886022e1, 1.0134030e1, 9.3637962e0, 9.1826344e0,
            6.8354754e0, 6.1705284e0, 5.7793241e0, 1.0945094e1, 1.0272807e1, 9.7030497e0,
            1.0253194e1, 1.1236749e1, 1.0417053e1, 9.0953083e0, 9.4387436e0, 1.0134028e1,
            5.9397693e0, 5.9890132e0, 6.8354783e0, 1.0674321e1, 9.7986908e0, 1.0016243e1,
            1.0886021e1, 1.0839963e1, 9.9236946e0, 9.1826353e0, 9.8477049e0, 9.6432323e0,
            5.7793241e0, 6.6412821e0, 6.4206138e0, 1.8000000e1, 1.8138357e1, 2.1000000e1,
            1.8000000e1, 1.8138357e1, 2.1000000e1, 1.8000000e1, 1.8138357e1, 2.1000000e1,
            2.1000000e1, 2.1000000e1, 2.1000000e1, 1.9000000e1, 1.9000000e1, 1.9000000e1,
            1.7000000e1, 1.7000000e1, 1.7000000e1, 1.5000000e1, 1.5000000e1, 1.5000000e1,
            2.1000000e1, 2.1000000e1, 2.1000000e1, 1.9000000e1, 1.9000000e1, 1.9000000e1,
            1.7000000e1, 1.7000000e1, 1.7000000e1, 1.5000000e1, 1.5000000e1, 1.5000000e1,
            2.1000000e1, 2.1000000e1, 2.1000000e1, 1.9000000e1, 1.9000000e1, 1.9000000e1,
            1.7000000e1, 1.7000000e1, 1.7000000e1, 1.5000000e1, 1.5000000e1, 1.5000000e1,
            2.1000000e1, 2.1000000e1, 2.1000000e1, 1.9000000e1, 1.9000000e1, 1.9000000e1,
            1.7000000e1, 1.7000000e1, 1.7000000e1, 1.5000000e1, 1.5000000e1, 1.5000000e1,
        ],
        _ => unreachable!(),
    }
}

/// Partial stack `index` of a smooth field with stars, its frames weighted
/// by a pattern of their own, its noise its own, and a band of the grid it
/// does not reach.
fn partial(index: usize, channels: usize) -> PartialStack {
    let samples = WIDTH * HEIGHT * channels;
    let mut mean = Vec::with_capacity(samples);
    let mut weight = Vec::with_capacity(samples);
    let mut coverage = Vec::with_capacity(samples);
    let mut variance = Vec::with_capacity(samples);
    let frames = 3 + index as u32 * 2;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            // Each stack misses its own band of columns.
            let reached = x / 12 != index;
            for channel in 0..channels {
                let star = (-(((x as f64 - 20.0).powi(2) + (y as f64 - 17.0).powi(2)) / 6.0)).exp()
                    * 900.0;
                let truth = 100.0 + 0.4 * x as f64 + 0.25 * y as f64 + 30.0 * channel as f64 + star;
                let wobble = ((x * 7 + y * 13 + index * 31 + channel * 5) % 17) as f64 - 8.0;
                let noise = 1.5 + index as f64 * 0.75;
                if reached {
                    mean.push((truth + wobble * noise * 0.1) as f32);
                    weight.push(frames as f32 / (noise * noise) as f32);
                    coverage.push(frames);
                    variance.push((noise * noise * (1.0 + wobble.abs() * 0.02)) as f32);
                } else {
                    mean.push(f32::NAN);
                    weight.push(0.0);
                    coverage.push(0);
                    variance.push(0.0);
                }
            }
        }
    }
    PartialStack {
        identity: Identity {
            kind: StackKind::Partial,
            stacking_version: seiza_stacking::OUTPUT_VERSION.into(),
            pipeline_version: seiza_pipeline::OUTPUT_VERSION.into(),
            step_key: Digest::of_bytes(&(index as u64).to_le_bytes()),
            inputs: Digest::of_bytes(b"inputs"),
            input_count: frames,
            combined: frames,
            calibration: None,
            options: Digest::of_bytes(b"options"),
            host: None,
            estimator: Estimator::Clipped,
            context: None,
            contributor: None,
            weight_source: WeightSource::Stack,
            grid: Digest::of_bytes(b"grid"),
        },
        mean: LinearImage::new(WIDTH, HEIGHT, channels, mean).unwrap(),
        weight,
        coverage,
        rejected: None,
        variance: Some(variance),
        headers: Vec::new(),
    }
}

/// Each channel's mean, root mean square and maximum over finite samples,
/// then the samples on a 4 by 4 grid of interior points, `NaN` as -1.
fn summary(data: &[f32], channels: usize) -> Vec<f64> {
    let mut values = Vec::new();
    for channel in 0..channels {
        let (mut sum, mut squares, mut maximum, mut count) = (0.0_f64, 0.0_f64, f64::MIN, 0);
        for &sample in data.iter().skip(channel).step_by(channels) {
            if sample.is_finite() {
                let sample = f64::from(sample);
                sum += sample;
                squares += sample * sample;
                maximum = maximum.max(sample);
                count += 1;
            }
        }
        let count = f64::from(count);
        values.extend([sum / count, (squares / count).sqrt(), maximum]);
    }
    for row in 0..4 {
        for column in 0..4 {
            let x = 3 + column * (WIDTH - 6) / 3;
            let y = 3 + row * (HEIGHT - 6) / 3;
            for channel in 0..channels {
                let sample = data[(y * WIDTH + x) * channels + channel];
                values.push(if sample.is_finite() {
                    f64::from(sample)
                } else {
                    -1.0
                });
            }
        }
    }
    values
}

fn compare(case: &str, actual: &[f64], expected: &[f64]) -> Option<String> {
    let same = actual.len() == expected.len()
        && actual.iter().zip(expected).all(|(actual, expected)| {
            (actual - expected).abs() <= TOLERANCE * expected.abs().max(1.0)
        });
    (!same).then(|| {
        let recorded = actual
            .iter()
            .map(|value| format!("{value:.7e}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("\"{case}\" => &[{recorded}],")
    })
}
