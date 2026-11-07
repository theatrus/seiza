//! Compare the public f64 statistics with their original sorting reference.
//!
//! `cargo run --release -p seiza-stats --example robust_profile`
//! prints interleaved median timings in CSV, after checking exact results.
//! For isolated peak memory measurements, run e.g.
//! `/usr/bin/time -v target/release/examples/robust_profile sigma native 10000000`.

use std::hint::black_box;
use std::time::Instant;

fn sorted_median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Some(if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) * 0.5
    } else {
        sorted[middle]
    })
}

fn sorted_sigma(values: &[f64]) -> Option<f64> {
    let deviations: Vec<_> = values.iter().map(|value| (value - 1000.0).abs()).collect();
    sorted_median(&deviations).map(|mad| mad * seiza_stats::NORMAL_MAD_SCALE)
}

fn native_sigma(values: &[f64]) -> Option<f64> {
    seiza_stats::robust_sigma_f64(values, 1000.0)
}

type Statistic = fn(&[f64]) -> Option<f64>;

fn samples(length: usize, shape: &str) -> Vec<f64> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut values: Vec<_> = (0..length)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Sky noise with a sparse bright tail; deterministic and finite.
            let noise = (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
            1000.0 + 30.0 * noise + if index % 101 == 0 { 500.0 } else { 0.0 }
        })
        .collect();
    match shape {
        "sky" => {}
        "sorted" => values.sort_unstable_by(f64::total_cmp),
        "reverse" => values.sort_unstable_by(|a, b| b.total_cmp(a)),
        "ties" => values.iter_mut().for_each(|value| *value = value.round()),
        _ => panic!("shape must be sky, sorted, reverse or ties"),
    }
    values
}

fn measure(statistic: Statistic, values: &[f64], iterations: usize) -> f64 {
    let started = Instant::now();
    for _ in 0..iterations {
        black_box(statistic(black_box(values)));
    }
    started.elapsed().as_secs_f64() * 1.0e6 / iterations as f64
}

fn main() {
    let operations: [(&str, Statistic, Statistic); 2] = [
        ("median", sorted_median, seiza_stats::median_f64),
        ("sigma", sorted_sigma, native_sigma),
    ];
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        assert_eq!(
            args.len(),
            3,
            "usage: robust_profile [median|sigma native|sorted length]"
        );
        let (_, sorted, native) = operations
            .iter()
            .find(|op| op.0 == args[0])
            .expect("median or sigma");
        let statistic = match args[1].as_str() {
            "native" => native,
            "sorted" => sorted,
            _ => panic!("native or sorted"),
        };
        let length: usize = args[2].parse().expect("sample count");
        let values = samples(length, "sky");
        println!("{:?}", black_box(statistic(black_box(&values))));
        return;
    }

    println!("operation,shape,samples,iterations,sorted_us,native_us,speedup");
    for shape in ["sky", "sorted", "reverse", "ties"] {
        for length in [31, 256, 4096, 65536, 1048576] {
            let values = samples(length, shape);
            let iterations = (1048576 / length).clamp(1, 10000);
            for (name, sorted, native) in operations {
                assert_eq!(
                    sorted(&values).map(f64::to_bits),
                    native(&values).map(f64::to_bits)
                );
                let mut sorting = Vec::new();
                let mut current = Vec::new();
                for trial in 0..7 {
                    // Alternate order to share frequency and thermal changes.
                    if trial % 2 == 0 {
                        sorting.push(measure(sorted, &values, iterations));
                        current.push(measure(native, &values, iterations));
                    } else {
                        current.push(measure(native, &values, iterations));
                        sorting.push(measure(sorted, &values, iterations));
                    }
                }
                sorting.sort_unstable_by(f64::total_cmp);
                current.sort_unstable_by(f64::total_cmp);
                println!(
                    "{name},{shape},{length},{iterations},{:.3},{:.3},{:.3}",
                    sorting[3],
                    current[3],
                    sorting[3] / current[3]
                );
            }
        }
    }
}
