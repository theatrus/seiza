//! A tiny sharpening model through ONNX Runtime itself, against what the
//! Python harness makes of the same image. These need the runtime's library,
//! so they run only where `SEIZA_ONNXRUNTIME` names one, as CI's runtime job
//! does on each platform (`--test sharpen`); elsewhere each passes with a
//! note. The comparison on a real image needs a trained bundle as well, and
//! runs only when asked for.

use seiza_model::{
    Backend, BackendOptions, ImageView, ModelBundle, ModelError, NoiseModel, Precision,
    SharpenModel, SharpenOptions, Sharpened,
};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Deserialize)]
struct Expected {
    width: usize,
    height: usize,
    stride: usize,
    strengths: (f32, f32),
    rolloff: f32,
    background: f32,
    noise: f32,
    stars: usize,
    ys: Vec<f64>,
    xs: Vec<f64>,
    points: Vec<serde_json::Value>,
    #[serde(default)]
    output: Vec<Option<f32>>,
    #[serde(default)]
    rolled: Vec<Option<f32>>,
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sharpen")
}

fn floats(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

fn expected(directory: &Path) -> Expected {
    let text = std::fs::read_to_string(directory.join("expected.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// False, with a note, where no library is named.
fn runtime_present() -> bool {
    let present = std::env::var_os("SEIZA_ONNXRUNTIME").is_some();
    if !present {
        eprintln!("skipped: SEIZA_ONNXRUNTIME names no ONNX Runtime library");
    }
    present
}

fn run(
    model: &mut SharpenModel,
    pixels: &[f32],
    want: &Expected,
    rolloff: f32,
) -> Result<Sharpened, ModelError> {
    let view = ImageView::new(pixels, want.width, want.height, 1)?;
    let options = SharpenOptions {
        stars: Some(want.strengths.0),
        rest: Some(want.strengths.1),
        rolloff,
        ..Default::default()
    };
    model.sharpen(view, &options)
}

/// The largest and root mean square difference between `got` and the
/// harness's samples, in units of the image's noise.
fn difference(got: &[f32], samples: &[Option<f32>], want: &Expected) -> (f32, f32) {
    let (mut worst, mut squares, mut count) = (0.0_f32, 0.0_f64, 0_usize);
    let mut samples = samples.iter();
    for y in (0..want.height).step_by(want.stride) {
        for x in (0..want.width).step_by(want.stride) {
            let value = got[y * want.width + x];
            match samples.next().unwrap() {
                None => assert!(value.is_nan(), "({x}, {y}): {value} where nothing is"),
                Some(sample) => {
                    let off = (value - sample).abs() / want.noise;
                    assert!(off.is_finite(), "({x}, {y}): {value} against {sample}");
                    worst = worst.max(off);
                    squares += f64::from(off) * f64::from(off);
                    count += 1;
                }
            }
        }
    }
    assert!(samples.next().is_none());
    (worst, (squares / count as f64).sqrt() as f32)
}

#[test]
fn a_tiny_sharpening_model_matches_the_harness_on_the_cpu_provider() {
    if !runtime_present() {
        return;
    }
    let directory = fixtures();
    let want = expected(&directory);
    let pixels = floats(&directory.join("image.f32"));
    let bundle = ModelBundle::open(directory.join("tiny")).unwrap();
    let mut model = SharpenModel::open(&bundle, Backend::Cpu, &BackendOptions::default()).unwrap();
    assert_eq!(model.backend(), Backend::Cpu);
    assert_eq!(model.tiling().tile, 128);

    let tiles = AtomicUsize::new(0);
    let progress = |_| {
        tiles.fetch_add(1, Ordering::Relaxed);
    };
    let view = ImageView::new(&pixels, want.width, want.height, 1).unwrap();
    let options = SharpenOptions {
        stars: Some(want.strengths.0),
        rest: Some(want.strengths.1),
        progress: Some(&progress),
        ..Default::default()
    };
    let result = model.sharpen(view, &options).unwrap();
    assert_eq!(
        tiles.load(Ordering::Relaxed),
        model.tiling().tiles(want.width, want.height).unwrap()
    );
    let read = result.report.channels[0];
    assert_eq!(read.background, want.background);
    assert!((read.noise / want.noise - 1.0).abs() < 1e-5, "{read:?}");
    assert_eq!(read.psf_nodes, (want.ys.len(), want.xs.len()));
    assert_eq!(read.psf_stars, want.stars);
    assert_eq!(read.points, want.points.len());
    assert_eq!(
        (result.report.stars, result.report.rest),
        want.strengths,
        "the report says what was asked"
    );

    let (worst, rms) = difference(result.image(), &want.output, &want);
    println!("the output: max {worst} rms {rms} of the noise");
    assert!(
        worst < 2e-3 && rms < 2e-4,
        "the output differs from the harness's by up to {worst} of the noise, {rms} rms"
    );
    let changed = result
        .image()
        .iter()
        .zip(&pixels)
        .filter(|(out, pixel)| (*out - *pixel).abs() > 0.05 * want.noise)
        .count();
    assert!(
        changed > pixels.len() / 2,
        "the tiny model changes the image: {changed}"
    );

    let rolled = run(&mut model, &pixels, &want, want.rolloff).unwrap();
    let (worst, rms) = difference(rolled.image(), &want.rolled, &want);
    assert!(
        worst < 2e-3 && rms < 2e-4,
        "rolled off, by up to {worst} of the noise, {rms} rms"
    );
    assert_eq!(rolled.report.rolloff, want.rolloff);

    // Left as its card says, the run asks for the card's own strengths.
    let own = model.sharpen(view, &SharpenOptions::default()).unwrap();
    assert_eq!(
        (own.report.stars, own.report.rest),
        want.strengths,
        "the fixture's card holds the strengths the harness ran"
    );
    assert_eq!(own.image()[5000], result.image()[5000]);
}

#[test]
fn a_colour_image_s_channels_run_alone() {
    if !runtime_present() {
        return;
    }
    let directory = fixtures();
    let want = expected(&directory);
    let pixels = floats(&directory.join("image.f32"));
    let bundle = ModelBundle::open(directory.join("tiny")).unwrap();
    let mut model = SharpenModel::open(&bundle, Backend::Cpu, &BackendOptions::default()).unwrap();
    let mono = run(&mut model, &pixels, &want, 0.0).unwrap();
    // The second channel twice as bright: its scales follow, and its output.
    let colour: Vec<f32> = pixels
        .iter()
        .flat_map(|&value| [value, 2.0 * value])
        .collect();
    let view = ImageView::new(&colour, want.width, want.height, 2).unwrap();
    let options = SharpenOptions {
        stars: Some(want.strengths.0),
        rest: Some(want.strengths.1),
        ..Default::default()
    };
    let result = model.sharpen(view, &options).unwrap();
    assert_eq!(result.channels(), 2);
    assert_eq!(result.report.channels.len(), 2);
    let noise = want.noise;
    for (index, &single) in mono.image().iter().enumerate().step_by(97) {
        let (first, second) = (result.image()[2 * index], result.image()[2 * index + 1]);
        if single.is_nan() {
            assert!(first.is_nan() && second.is_nan());
            continue;
        }
        assert_eq!(first, single, "pixel {index}");
        assert!(
            (second - 2.0 * single).abs() < 0.05 * noise,
            "pixel {index}: {second} against twice {single}"
        );
    }
}

#[test]
fn a_sharpening_model_is_not_a_noise_model_and_stops_when_asked() {
    if !runtime_present() {
        return;
    }
    let directory = fixtures();
    let want = expected(&directory);
    let pixels = floats(&directory.join("image.f32"));
    let bundle = ModelBundle::open(directory.join("tiny")).unwrap();
    let options = BackendOptions::default();
    assert!(matches!(
        NoiseModel::open(&bundle, Backend::Cpu, &options),
        Err(ModelError::ModelContract(_))
    ));
    let noise = ModelBundle::open(directory.join("../tiny")).unwrap();
    assert!(matches!(
        SharpenModel::open(&noise, Backend::Cpu, &options),
        Err(ModelError::ModelContract(_))
    ));

    let mut model = SharpenModel::open(&bundle, Backend::Cpu, &options).unwrap();
    let view = ImageView::new(&pixels, want.width, want.height, 1).unwrap();
    let stop = || true;
    let stopped = SharpenOptions {
        cancel: Some(&stop),
        ..Default::default()
    };
    assert!(matches!(
        model.sharpen(view, &stopped),
        Err(ModelError::Cancelled)
    ));
    let blank = vec![100.0_f32; want.width * want.height];
    let view = ImageView::new(&blank, want.width, want.height, 1).unwrap();
    assert!(
        model.sharpen(view, &SharpenOptions::default()).is_err(),
        "an image with no noise and no stars has nothing to be read by"
    );
}

#[test]
fn an_image_too_bare_for_a_psf_is_refused_for_that() {
    if !runtime_present() {
        return;
    }
    let directory = fixtures();
    let bundle = ModelBundle::open(directory.join("tiny")).unwrap();
    let mut model = SharpenModel::open(&bundle, Backend::Cpu, &BackendOptions::default()).unwrap();
    // Noise and a gradient, and not one star.
    let (width, height) = (200, 160);
    let mut state = 9_u64;
    let sky: Vec<f32> = (0..width * height)
        .map(|index| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            100.0
                + 0.01 * (index % width) as f32
                + 6.0 * ((state >> 40) as f32 / (1_u32 << 24) as f32 - 0.5)
        })
        .collect();
    let view = ImageView::new(&sky, width, height, 1).unwrap();
    assert!(matches!(
        model.sharpen(view, &SharpenOptions::default()),
        Err(ModelError::PsfField(_))
    ));
}

/// A trained model on a real image, against the harness on the same: the
/// directory `make_sharpen_fixtures.py --image … --run … --out DIRECTORY`
/// wrote, named by `SEIZA_SHARPEN_PARITY`. `SEIZA_SHARPEN_BACKENDS` lists the
/// backends to run, by their words and a precision, as `cpu:f32,coreml-gpu:f16`
/// (default `cpu:f32`).
#[test]
#[ignore = "needs a trained bundle and the harness's answers, which are not in the repository"]
fn a_trained_model_matches_the_harness_on_a_real_image() {
    let directory = PathBuf::from(
        std::env::var_os("SEIZA_SHARPEN_PARITY").expect("SEIZA_SHARPEN_PARITY names a directory"),
    );
    let mut want = expected(&directory);
    want.stride = 1;
    let pixels = floats(&directory.join("image.f32"));
    let outputs = [
        (0.0, floats(&directory.join("output.f32"))),
        (want.rolloff, floats(&directory.join("rolled.f32"))),
    ];
    let bundle = ModelBundle::open(directory.join("bundle")).unwrap();
    let backends =
        std::env::var("SEIZA_SHARPEN_BACKENDS").unwrap_or_else(|_| "cpu:f32".to_string());
    for word in backends.split(',') {
        let (backend, precision) = word.split_once(':').expect("a backend and a precision");
        let backend: Backend = backend.parse().unwrap();
        let precision: Precision = precision.parse().unwrap();
        let options = BackendOptions {
            precision: Some(precision),
            ..Default::default()
        };
        let started = std::time::Instant::now();
        let mut model = SharpenModel::open(&bundle, backend, &options).unwrap();
        let opened = started.elapsed();
        for (rolloff, expected) in &outputs {
            let started = std::time::Instant::now();
            let result = run(&mut model, &pixels, &want, *rolloff).unwrap();
            let seconds = started.elapsed().as_secs_f32();
            let samples: Vec<Option<f32>> = expected
                .iter()
                .map(|&value| (!value.is_nan()).then_some(value))
                .collect();
            let (worst, rms) = difference(result.image(), &samples, &want);
            println!(
                "{backend} {precision} roll-off {rolloff}: max {worst:.5} rms {rms:.6} of the \
                 noise, in {seconds:.1} s (opened in {:.1} s), tile {}",
                opened.as_secs_f32(),
                result.report.tile
            );
            if backend == Backend::Cpu && precision == Precision::F32 {
                assert!(worst < 0.05 && rms < 0.002, "{worst} {rms}");
            }
        }
    }
}
