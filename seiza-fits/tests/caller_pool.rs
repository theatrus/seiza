//! FITS conversions run in the Rayon pool their caller installs.
//!
//! Every entry point here runs inside a one-thread and a two-thread pool,
//! and must give the same bits in both. The global pool must never start:
//! Rayon work that left the caller's pool would start it. That can be
//! checked once per process, and each file under `tests/` runs as a process
//! of its own, so this is one test.

use rayon::ThreadPoolBuilder;
use seiza_fits::{
    BayerPattern, F32ImageData, FitsImage, HeaderValue, Pixels, StretchParams, WriteHeaderCard,
    debayer_rgb_f32, read_header, read_header_with_commentary, write_f32_image,
};

/// More than four of the crate's 65,536-sample chunks per plane, and more
/// than one of the stretch's 262,144-sample chunks.
const WIDTH: usize = 640;
const HEIGHT: usize = 480;

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

/// A sky with a gradient, a few stars and a little noise, in `[0, 1]`.
fn sky(plane: usize) -> Vec<f64> {
    (0..WIDTH * HEIGHT)
        .map(|index| {
            let (x, y) = ((index % WIDTH) as f64, (index / WIDTH) as f64);
            let noise = ((index * 37 + plane * 11) % 23) as f64 / 23.0;
            let mut value = 0.08 + 0.04 * x / WIDTH as f64 + 0.01 * noise;
            for star in 0..40 {
                let sx = ((star * 7919) % WIDTH) as f64;
                let sy = ((star * 6271) % HEIGHT) as f64;
                let r2 = (x - sx).powi(2) + (y - sy).powi(2);
                if r2 < 30.0 {
                    value += 0.6 * (-r2 / 4.0).exp();
                }
            }
            value.min(1.0) + plane as f64 * 0.02
        })
        .collect()
}

fn image(planes: usize, pixels: Pixels, headers: Vec<(String, HeaderValue)>) -> FitsImage {
    FitsImage {
        width: WIDTH,
        height: HEIGHT,
        planes,
        pixels,
        headers,
    }
}

struct Inputs {
    directory: tempfile::TempDir,
    mono: FitsImage,
    mosaic: FitsImage,
    planar: FitsImage,
    integer: FitsImage,
    double: FitsImage,
}

fn inputs() -> Inputs {
    let to_u16 = |values: Vec<f64>| -> Vec<u16> {
        values
            .into_iter()
            .map(|value| (value * 60_000.0) as u16)
            .collect()
    };
    let bayer = vec![("BAYERPAT".into(), HeaderValue::String("RGGB".into()))];
    let planar = (0..3).flat_map(sky).map(|value| value as f32).collect();
    Inputs {
        directory: tempfile::tempdir().unwrap(),
        mono: image(1, Pixels::U16(to_u16(sky(0))), Vec::new()),
        mosaic: image(1, Pixels::U16(to_u16(sky(1))), bayer),
        planar: image(3, Pixels::F32(planar), Vec::new()),
        integer: image(
            1,
            Pixels::I32(
                sky(2)
                    .into_iter()
                    .map(|value| (value * 1e6) as i32)
                    .collect(),
            ),
            Vec::new(),
        ),
        double: image(1, Pixels::F64(sky(0)), Vec::new()),
    }
}

/// Everything run inside `pool`, as the bits of each result.
fn run(pool: &rayon::ThreadPool, inputs: &Inputs) -> Vec<(&'static str, Vec<u32>)> {
    pool.install(|| {
        let params = StretchParams::default();
        let mut results = Vec::new();
        for (name, image) in [
            ("mono", &inputs.mono),
            ("mosaic", &inputs.mosaic),
            ("planar", &inputs.planar),
            ("integer", &inputs.integer),
            ("double", &inputs.double),
        ] {
            let statistics = image.statistics();
            let mut out = widen(&image.to_u16());
            out.extend(widen(&image.stretch_to_u8(&params)));
            out.extend(widen(&image.stretch_to_u16(&params)));
            out.extend(bits(&image.to_luma_f32()));
            out.extend([statistics.min, statistics.max, statistics.median].map(u32::from));
            out.extend(bits64(statistics.mean));
            out.extend(bits64(statistics.std_dev));
            out.extend(bits64(statistics.mad));
            if let Some(rgb) = image.rgb_planes() {
                out.extend(widen(&rgb.data));
            }
            if let Some(rgb) = image.debayer() {
                out.extend(widen(&rgb.data));
                out.extend(widen(&rgb.to_luma_u16()));
            }
            results.push((name, out));
        }

        let Pixels::U16(mosaic) = &inputs.mosaic.pixels else {
            unreachable!("the mosaic is 16-bit");
        };
        let mosaic = mosaic
            .iter()
            .map(|&value| f32::from(value))
            .collect::<Vec<_>>();
        let debayered = debayer_rgb_f32(&mosaic, WIDTH, HEIGHT, BayerPattern::Rggb, 0, 0);
        results.push(("debayer f32", bits(&debayered.data)));

        // Written, then read back from the file and from memory.
        let path = inputs
            .directory
            .path()
            .join(format!("planar-{}.fits", rayon::current_num_threads()));
        let Pixels::F32(planar) = &inputs.planar.pixels else {
            unreachable!("the planar image is floating point");
        };
        let cards = [WriteHeaderCard::new(
            "OBJECT",
            HeaderValue::String("sky".into()),
        )];
        write_f32_image(
            &path,
            WIDTH,
            HEIGHT,
            F32ImageData::RgbPlanar(planar),
            &cards,
        )
        .unwrap();
        let opened = FitsImage::open(&path).unwrap();
        let from_bytes = FitsImage::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
        let (Pixels::F32(opened), Pixels::F32(from_bytes)) = (&opened.pixels, &from_bytes.pixels)
        else {
            unreachable!("the writer stores 32-bit floats");
        };
        assert_eq!(bits(opened), bits(planar));
        assert_eq!(bits(from_bytes), bits(planar));
        assert_eq!(read_header(&path).unwrap().len(), {
            read_header_with_commentary(&path).unwrap().cards.len()
        });
        results.push(("written", bits(opened)));
        results
    })
}

#[test]
fn fits_conversions_stay_in_the_callers_pool() {
    let pools = [1, 2].map(|threads| {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(move |index| format!("caller-{threads}-{index}"))
            .build()
            .unwrap()
    });
    let inputs = pools[0].install(inputs);
    let one = run(&pools[0], &inputs);
    let two = run(&pools[1], &inputs);
    assert_eq!(one.len(), two.len());
    for ((name, one), (_, two)) in one.iter().zip(&two) {
        assert!(one == two, "{name} differs between one and two threads");
    }
    ThreadPoolBuilder::new()
        .build_global()
        .expect("work escaped the caller's pool and started the global pool");
}
