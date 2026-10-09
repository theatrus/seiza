//! Deconvolution runs in the Rayon pool its caller installs.
//!
//! Every entry point here runs inside a one-thread and a two-thread pool,
//! and must give the same bits in both. The global pool must never start:
//! Rayon work that left the caller's pool would start it. That can be
//! checked once per process, and each file under `tests/` runs as a process
//! of its own, so this is one test.

use rayon::ThreadPoolBuilder;
use seiza_deconvolution::{
    DeconvolutionConfig, DeconvolutionResult, deconvolve, deconvolve_masked,
};

const WIDTH: usize = 160;
const HEIGHT: usize = 128;

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn result_bits(result: &DeconvolutionResult) -> Vec<u32> {
    let mut out = bits(&result.data);
    for channel in &result.channels {
        for value in [channel.input_flux, channel.output_flux] {
            let bits = value.to_bits();
            out.extend([bits as u32, (bits >> 32) as u32]);
        }
        out.extend([channel.input_peak, channel.output_peak].map(f32::to_bits));
    }
    out
}

/// Soft stars on a flat sky, `channels` interleaved, with a registration
/// border of missing samples when `masked`.
fn star_field(channels: usize, masked: bool) -> Vec<f32> {
    (0..WIDTH * HEIGHT * channels)
        .map(|sample| {
            let (pixel, channel) = (sample / channels, sample % channels);
            let (x, y) = (pixel % WIDTH, pixel / WIDTH);
            if masked && (x < 4 || y >= HEIGHT - 3) {
                return f32::NAN;
            }
            let noise = ((pixel * 37 + channel * 11) % 23) as f32;
            let mut value = 500.0 + 20.0 * channel as f32 + noise;
            for star in 0..20 {
                let sx = ((star * 7919) % WIDTH) as f32;
                let sy = ((star * 6271) % HEIGHT) as f32;
                let r2 = (x as f32 - sx).powi(2) + (y as f32 - sy).powi(2);
                if r2 < 36.0 {
                    value += 6_000.0 * (-r2 / 4.5).exp();
                }
            }
            value
        })
        .collect()
}

/// Everything run inside `pool`, as the bits of each result.
fn run(pool: &rayon::ThreadPool) -> Vec<(&'static str, Vec<u32>)> {
    pool.install(|| {
        let config = DeconvolutionConfig::conservative(2.5);
        let mono = star_field(1, false);
        let rgb = star_field(3, true);
        vec![
            (
                "mono",
                result_bits(&deconvolve(&mono, WIDTH, HEIGHT, 1, &config).unwrap()),
            ),
            (
                "masked colour",
                result_bits(&deconvolve_masked(&rgb, WIDTH, HEIGHT, 3, &config).unwrap()),
            ),
        ]
    })
}

#[test]
fn deconvolution_stays_in_the_callers_pool() {
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
