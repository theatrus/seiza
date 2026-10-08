//! With the `parallel` feature, filters run in the Rayon pool their caller
//! installs.
//!
//! Every entry point here runs inside a one-thread and a two-thread pool,
//! and must give the same bits in both. The global pool must never start:
//! Rayon work that left the caller's pool would start it. That can be
//! checked once per process, and each file under `tests/` runs as a process
//! of its own, so this is one test.

#![cfg(feature = "parallel")]

use rayon::ThreadPoolBuilder;
use seiza_imgproc::blur::gaussian_blur_f32;
use seiza_imgproc::border::BorderMode;
use seiza_imgproc::components::{Connectivity, largest_connected_component};
use seiza_imgproc::dtfilter::dt_filter_nc;
use seiza_imgproc::wavelets::StructureRemover;

/// Four times the 16,384 pixels at which the filters split into rows.
const WIDTH: usize = 256;
const HEIGHT: usize = 256;

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Stars on a sloped background with a little noise.
fn star_field() -> Vec<f32> {
    (0..WIDTH * HEIGHT)
        .map(|index| {
            let (x, y) = ((index % WIDTH) as f32, (index / WIDTH) as f32);
            let noise = ((index * 37) % 23) as f32;
            let mut value = 1_000.0 + 0.5 * x + 0.25 * y + noise;
            for star in 0..30 {
                let sx = ((star * 7919) % WIDTH) as f32;
                let sy = ((star * 6271) % HEIGHT) as f32;
                let r2 = (x - sx).powi(2) + (y - sy).powi(2);
                if r2 < 25.0 {
                    value += 8_000.0 / (1.0 + r2);
                }
            }
            value
        })
        .collect()
}

/// Everything run inside `pool`, as the bits of each result.
fn run(pool: &rayon::ThreadPool) -> Vec<(&'static str, Vec<u32>)> {
    pool.install(|| {
        let image = star_field();
        let mut results = vec![
            (
                "blur",
                bits(&gaussian_blur_f32(
                    &image,
                    WIDTH,
                    HEIGHT,
                    5,
                    1.2,
                    BorderMode::Reflect101,
                )),
            ),
            (
                "domain transform",
                bits(&dt_filter_nc(&image, &image, WIDTH, HEIGHT, 20.0, 0.1, 3)),
            ),
        ];
        // Five layers: three Gaussian, then two domain transform.
        let remover = StructureRemover::new(5);
        let residual = remover.remove_structures_filtered_f32(&image, WIDTH, HEIGHT);
        results.push(("filtered residual", bits(&residual)));
        let map = remover.structure_map_atrous_chain(&image, WIDTH, HEIGHT);
        results.push(("structure map", bits(&map)));

        let threshold = residual.iter().copied().fold(f32::MIN, f32::max) * 0.2;
        let mask = residual
            .iter()
            .map(|&value| u8::from(value > threshold))
            .collect::<Vec<_>>();
        let component =
            largest_connected_component(&mask, WIDTH, HEIGHT, Connectivity::Eight).unwrap();
        let mut out = component
            .pixels
            .iter()
            .map(|&pixel| pixel as u32)
            .collect::<Vec<_>>();
        out.extend([component.centroid_x, component.centroid_y].map(f32::to_bits));
        results.push(("component", out));
        results
    })
}

#[test]
fn filters_stay_in_the_callers_pool() {
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
