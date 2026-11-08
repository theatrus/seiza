//! Isolate the native preprocessing kernel without loading ONNX Runtime.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

mod bundle {
    pub use seiza_model::{InputPlane, NoiseInput, NoisePlane, Reading, SkyInput};
}
#[allow(dead_code)]
#[path = "@ROOT@/seiza-model/src/noise.rs"]
mod noise;
#[allow(dead_code)]
#[path = "@ROOT@/seiza-model/src/sky.rs"]
mod sky;
#[allow(dead_code)]
#[path = "@ROOT@/seiza-model/src/stats.rs"]
mod stats;

const COUNT: bool = @COUNT@;
struct Measured;
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Measured {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNT {
            CALLS.fetch_add(1, Relaxed);
            BYTES.fetch_add(layout.size(), Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if COUNT {
            CALLS.fetch_add(1, Relaxed);
            BYTES.fetch_add(size, Relaxed);
        }
        unsafe { System.realloc(pointer, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Measured = Measured;

fn checksum(values: &[f32]) -> u64 {
    values.iter().fold(0xcbf29ce484222325, |hash, value| {
        (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100000001b3)
    })
}

fn main() {
    let side: usize = std::env::args().nth(1).unwrap_or_else(|| "2048".into()).parse().unwrap();
    let mut state = 7_u64;
    let image: Vec<f32> = (0..side * side).map(|index| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let (x, y) = (index % side, index / side);
        if x < 32 && y < 64 { return f32::NAN; }
        let background = 100.0 + x as f32 * 0.003 + y as f32 * 0.002;
        let nebula = if x > side * 2 / 3 { 500.0 } else { 0.0 };
        background + nebula + ((state >> 40) as f32 / (1_u32 << 24) as f32 - 0.5) * 12.0
    }).collect();
    println!("side,threads,mode,trial,milliseconds,allocation_calls,allocated_bytes,checksum");
    for threads in [1, 8] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        for mode in [bundle::NoiseInput::Image, bundle::NoiseInput::Constant] {
            let reading = bundle::Reading {
                noise_map: mode, noise_plane: bundle::NoisePlane::Asinh, sky: bundle::SkyInput::Flat,
                sky_block: 32, flat_block: 256, image_map_scale: 1.13,
            };
            pool.install(|| {
                std::hint::black_box(noise::input_sigma(&image, side, side, &reading));
            });
            let mut expected = None;
            for trial in 0..9 {
                let calls = CALLS.load(Relaxed);
                let bytes = BYTES.load(Relaxed);
                let start = Instant::now();
                let result = pool.install(|| noise::input_sigma(&image, side, side, &reading));
                let milliseconds = start.elapsed().as_secs_f64() * 1000.0;
                let allocated_bytes = BYTES.load(Relaxed) - bytes;
                let allocation_calls = CALLS.load(Relaxed) - calls;
                let digest = checksum(&result);
                assert_eq!(*expected.get_or_insert(digest), digest);
                println!("{side},{threads},{mode:?},{trial},{milliseconds:.6},{allocation_calls},{allocated_bytes},{digest:016x}");
            }
        }
    }
}
