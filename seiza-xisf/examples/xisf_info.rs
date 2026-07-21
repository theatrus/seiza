use seiza_fits::Pixels;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let path = PathBuf::from(
        arguments
            .next()
            .ok_or("usage: xisf_info <path> [--decode]")?,
    );
    let decode = arguments.any(|argument| argument == "--decode");

    let info = seiza_xisf::inspect(&path)?;
    println!("{}: {} image(s)", path.display(), info.images.len());
    for image in &info.images {
        println!(
            "  #{} id={:?} {}x{}x{} {:?} {} {:?}",
            image.index,
            image.id,
            image.width,
            image.height,
            image.planes,
            image.sample_format,
            image.color_space,
            image.compression
        );
    }

    if decode {
        let image = seiza_xisf::open(&path)?;
        let sample_count = match &image.pixels {
            Pixels::U8(values) => values.len(),
            Pixels::U16(values) => values.len(),
            Pixels::I32(values) => values.len(),
            Pixels::F32(values) => values.len(),
            Pixels::F64(values) => values.len(),
        };
        let statistics = image.statistics();
        println!(
            "  decoded first image: {} samples, median={}, MAD={}",
            sample_count, statistics.median, statistics.mad
        );
    }

    Ok(())
}
