//! Composition runs in the caller's Rayon pool and gives the same samples
//! whatever its size.

use rayon::ThreadPoolBuilder;
use seiza_compose::{
    ColorOptions, ForaxxOptions, NarrowbandPalette, combine_narrowband, combine_rgb,
};
use seiza_stacking::LinearImage;

fn channel(seed: u32) -> LinearImage {
    let (width, height) = (97, 61);
    let data = (0..width * height)
        .map(|index| {
            let mixed = (index as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed);
            0.02 + (mixed % 10_000) as f32 / 12_000.0
        })
        .collect();
    LinearImage::new(width, height, 1, data).unwrap()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn composition_stays_in_the_callers_pool() {
    let channels = [channel(1), channel(2), channel(3)];
    let run = |threads: usize| {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(move |index| format!("caller-{threads}-{index}"))
            .build()
            .unwrap();
        pool.install(|| {
            let rgb = combine_rgb(
                &channels[0],
                &channels[1],
                &channels[2],
                &ColorOptions::default(),
            )
            .unwrap();
            let sho = combine_narrowband(
                &channels[1],
                &channels[2],
                Some(&channels[0]),
                NarrowbandPalette::Sho,
                &ColorOptions::default(),
                &ForaxxOptions::default(),
            )
            .unwrap();
            (bits(&rgb.image.data), bits(&sho.image.data))
        })
    };
    assert_eq!(run(1), run(2));
    ThreadPoolBuilder::new()
        .build_global()
        .expect("work escaped the caller's pool and started the global pool");
}
