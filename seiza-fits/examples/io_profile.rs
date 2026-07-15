//! Compare complete FITS decode strategies in separate release processes.
//!
//! Run under a platform memory profiler, for example:
//! `/usr/bin/time -l cargo run --release -p seiza-fits --example io_profile -- stream image.fits 50 --load-only`

use std::path::Path;
use std::time::Instant;

use seiza_fits::{FitsImage, Pixels};

fn pixel_checksum(pixels: &Pixels) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    let mut update = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    };
    match pixels {
        Pixels::U8(values) => update(values),
        Pixels::U16(values) => values.iter().for_each(|value| update(&value.to_le_bytes())),
        Pixels::I32(values) => values.iter().for_each(|value| update(&value.to_le_bytes())),
        Pixels::F32(values) => values
            .iter()
            .for_each(|value| update(&value.to_bits().to_le_bytes())),
        Pixels::F64(values) => values
            .iter()
            .for_each(|value| update(&value.to_bits().to_le_bytes())),
    }
    hash
}

fn load_image(mode: &str, path: &Path) -> Result<FitsImage, Box<dyn std::error::Error>> {
    match mode {
        "buffered" => {
            let bytes = std::fs::read(path)?;
            Ok(FitsImage::from_bytes(&bytes)?)
        }
        "mmap" => {
            let file = std::fs::File::open(path)?;
            // SAFETY: the file is opened read-only and the map is used only
            // for the duration of the synchronous owned-pixel decode.
            let map = unsafe { memmap2::MmapOptions::new().map(&file)? };
            Ok(FitsImage::from_bytes(&map)?)
        }
        "stream" => Ok(FitsImage::open(path)?),
        _ => Err("mode must be one of: buffered, mmap, stream".into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let mode = args.next().ok_or(
        "usage: io_profile <buffered|mmap|stream> <image.fit[s]> [iterations] [--load-only]",
    )?;
    let path = args.next().ok_or(
        "usage: io_profile <buffered|mmap|stream> <image.fit[s]> [iterations] [--load-only]",
    )?;
    let iterations = match args.next() {
        Some(value) => value
            .to_str()
            .ok_or("iterations must be valid UTF-8")?
            .parse::<usize>()?,
        None => 1,
    };
    let load_only = match args.next() {
        Some(value) if value == "--load-only" => true,
        Some(_) => return Err("the only supported option is --load-only".into()),
        None => false,
    };
    if args.next().is_some() {
        return Err(
            "usage: io_profile <buffered|mmap|stream> <image.fit[s]> [iterations] [--load-only]"
                .into(),
        );
    }
    if iterations == 0 {
        return Err("iterations must be greater than zero".into());
    }
    let mode = mode.to_str().ok_or("mode must be valid UTF-8")?;
    let path = Path::new(&path);

    let started = Instant::now();
    let mut final_image = None;
    for iteration in 0..iterations {
        let image = load_image(mode, path)?;
        std::hint::black_box(&image.pixels);
        if iteration + 1 == iterations {
            final_image = Some(image);
        }
    }
    let load_time = started.elapsed().as_secs_f64() / iterations as f64;
    let image = final_image.expect("iterations is non-zero");

    let storage = match &image.pixels {
        Pixels::U8(_) => "u8",
        Pixels::U16(_) => "u16",
        Pixels::I32(_) => "i32",
        Pixels::F32(_) => "f32",
        Pixels::F64(_) => "f64",
    };
    if load_only {
        println!(
            "mode={} iterations={} image={}x{}x{} storage={} load_ms={:.3}",
            mode,
            iterations,
            image.width,
            image.height,
            image.planes,
            storage,
            load_time * 1_000.0,
        );
        return Ok(());
    }

    let started = Instant::now();
    let stats = image.statistics();
    let stats_time = started.elapsed();

    println!(
        "mode={} iterations={} image={}x{}x{} storage={} load_ms={:.3} stats_ms={:.3} median={} mean={:.3} range={}..{} checksum={:016x}",
        mode,
        iterations,
        image.width,
        image.height,
        image.planes,
        storage,
        load_time * 1_000.0,
        stats_time.as_secs_f64() * 1_000.0,
        stats.median,
        stats.mean,
        stats.min,
        stats.max,
        pixel_checksum(&image.pixels),
    );
    Ok(())
}
