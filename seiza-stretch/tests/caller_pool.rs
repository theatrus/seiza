//! Stretches run in the Rayon pool their caller installs.
//!
//! Every entry point here runs inside a one-thread and a two-thread pool,
//! and must give the same bits in both. The global pool must never start:
//! Rayon work that left the caller's pool would start it. That can be
//! checked once per process, and each file under `tests/` runs as a process
//! of its own, so this is one test.

use rayon::ThreadPoolBuilder;
use seiza_stretch::{
    ColorStrategy, RobustStatistics, SampleDomain, SampleNormalization, StretchAnalysis,
    StretchConfig, StretchParams, StretchStack, statistics_u16, stretch_u16_to_u8,
    stretch_u16_to_u16,
};

/// More than one of the 262,144-sample chunks the `u16` paths split into.
const U16_SAMPLES: usize = 640 * 480;
/// Four of the 16,384-pixel chunks the `f32` curves split into.
const PIXELS: usize = 256 * 256;

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn bits64(value: f64) -> [u32; 2] {
    let bits = value.to_bits();
    [bits as u32, (bits >> 32) as u32]
}

fn widen<T: Copy + Into<u32>>(values: &[T]) -> Vec<u32> {
    values.iter().map(|&value| value.into()).collect()
}

fn robust(statistics: &RobustStatistics) -> Vec<u32> {
    let mut out = Vec::new();
    for value in [
        statistics.min,
        statistics.max,
        statistics.median,
        statistics.mad,
    ] {
        out.extend(bits64(value));
    }
    out.push(statistics.count as u32);
    out
}

/// Faint sky with a few bright stars and a little noise.
fn sky(index: usize, channel: usize) -> f32 {
    let noise = ((index * 37 + channel * 11) % 23) as f32;
    let star = if (index * 7919).is_multiple_of(997) {
        30_000.0
    } else {
        0.0
    };
    1_000.0 + 40.0 * channel as f32 + noise * 3.0 + star
}

/// Interleaved RGB in camera units, with a few missing samples.
fn rgb() -> Vec<f32> {
    (0..PIXELS * 3)
        .map(|sample| {
            if sample.is_multiple_of(4099) {
                f32::NAN
            } else {
                sky(sample / 3, sample % 3)
            }
        })
        .collect()
}

/// Everything run inside `pool`, as the bits of each result.
fn run(pool: &rayon::ThreadPool) -> Vec<(&'static str, Vec<u32>)> {
    pool.install(|| {
        let mut results = Vec::new();
        let params = StretchParams::default();

        let camera = (0..U16_SAMPLES)
            .map(|index| sky(index, 0) as u16)
            .collect::<Vec<_>>();
        let statistics = statistics_u16(&camera);
        let mut out = widen(&stretch_u16_to_u8(&camera, &statistics, &params));
        out.extend(widen(&stretch_u16_to_u16(&camera, &statistics, &params)));
        out.extend([statistics.min, statistics.max, statistics.median].map(u32::from));
        out.extend(bits64(statistics.mean));
        out.extend(bits64(statistics.std_dev));
        out.extend(bits64(statistics.mad));
        results.push(("u16", out));

        // Camera units onto the unit range the curves expect.
        let mut data = rgb();
        let domain = SampleDomain::PhysicalLinear {
            normalization: SampleNormalization::default(),
        }
        .resolve(&data, 3)
        .unwrap();
        domain.apply_in_place(&mut data, 3).unwrap();
        results.push(("domain", bits(&data)));

        let analysis = StretchAnalysis::analyze(&data, 3, 200_000).unwrap();
        let mut out = robust(&analysis.linked_statistics());
        for channel in analysis.channel_statistics().iter().flatten() {
            out.extend(robust(channel));
        }
        out.extend(robust(&analysis.luminance_statistics().unwrap()));
        results.push(("analysis", out));

        for (name, strategy) in [
            ("linked", ColorStrategy::Linked),
            ("unlinked", ColorStrategy::Unlinked),
            ("luminance", ColorStrategy::LuminancePreserving),
        ] {
            let config = StretchConfig {
                color_strategy: strategy,
                ..StretchConfig::auto_mtf(params, 200_000)
            };
            let plan = config.resolve(&analysis).unwrap();
            let mut out = bits(&plan.apply_f32(&data, 3).unwrap());
            out.extend(widen(&plan.apply_u8(&data, 3).unwrap()));
            out.extend(widen(&plan.apply_u16(&data, 3).unwrap()));
            results.push((name, out));
        }

        let stack = StretchStack::new(vec![
            StretchConfig::percentile_asinh(0.001, 0.999, 50.0, 200_000),
            StretchConfig {
                color_strategy: ColorStrategy::LuminancePreserving,
                ..StretchConfig::auto_mtf(params, 200_000)
            },
        ])
        .unwrap();
        let mut stages = 0;
        let staged = stack
            .apply_f32_with_progress(&data, 3, |_| stages += 1)
            .unwrap();
        let mut out = bits(&staged.data);
        out.extend(widen(&stack.apply_u8(&data, 3).unwrap().data));
        out.extend(widen(&stack.apply_u16(&data, 3).unwrap().data));
        out.push(stages);
        results.push(("stack", out));

        let mono = data.iter().step_by(3).copied().collect::<Vec<_>>();
        let single = StretchStack::single(StretchConfig::auto_mtf(params, 200_000));
        results.push(("mono", bits(&single.apply_f32(&mono, 1).unwrap().data)));
        results
    })
}

#[test]
fn stretches_stay_in_the_callers_pool() {
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
