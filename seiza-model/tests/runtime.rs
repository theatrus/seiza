//! A tiny model through ONNX Runtime itself. These need its library, so they
//! run only where `SEIZA_ONNXRUNTIME` names one, as CI's runtime job does.

use seiza_model::{
    Backend, BackendOptions, CoreMlUnits, DenoiseOptions, ImageView, ModelBundle, ModelError,
    NoiseBands, NoiseModel, Precision, Tiling, runtime_info,
};
use serde::Deserialize;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Deserialize)]
struct Expected {
    width: usize,
    height: usize,
    stride: usize,
    noise: f32,
    residual: Vec<Option<f32>>,
}

fn fixtures() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"))
}

fn image() -> Vec<f32> {
    let bytes = std::fs::read(fixtures().join("image.f32")).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

fn expected() -> Expected {
    let text = std::fs::read_to_string(fixtures().join("tiny-expected.json")).unwrap();
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

fn open(backend: Backend, options: &BackendOptions) -> Result<NoiseModel, ModelError> {
    let bundle = ModelBundle::open(fixtures().join("tiny"))?;
    NoiseModel::open(&bundle, backend, options)
}

#[test]
fn the_runtime_reports_its_version_and_cpu_backend() {
    if !runtime_present() {
        return;
    }
    let options = BackendOptions::default();
    let info = runtime_info(&options).unwrap();
    assert!(info.version.starts_with("1."), "{}", info.version);
    let statuses = Backend::available(&options);
    let cpu = statuses
        .iter()
        .find(|status| status.backend == Backend::Cpu)
        .unwrap();
    assert_eq!(cpu.unavailable, None);
}

#[test]
fn a_tiny_model_matches_the_harness_on_the_cpu_provider() {
    if !runtime_present() {
        return;
    }
    let want = expected();
    let pixels = image();
    let mut model = open(Backend::Cpu, &BackendOptions::default()).unwrap();
    assert_eq!(model.backend(), Backend::Cpu);
    assert_eq!(
        model.tiling(),
        Tiling {
            tile: 64,
            overlap: 28,
            margin: 10,
            multiple: 2
        }
    );

    let tiles = AtomicUsize::new(0);
    let progress = |_| {
        tiles.fetch_add(1, Ordering::Relaxed);
    };
    let view = ImageView::new(&pixels, want.width, want.height, 1).unwrap();
    let options = DenoiseOptions {
        progress: Some(&progress),
        cancel: None,
    };
    let result = model.denoise(view, &options).unwrap();
    assert_eq!(
        tiles.load(Ordering::Relaxed),
        model.tiling().tiles(want.width, want.height).unwrap()
    );
    let scales = result.report.channels[0];
    assert!((scales.noise / want.noise - 1.0).abs() < 1e-5, "{scales:?}");

    let residual = result.residual(0).unwrap();
    let mut samples = want.residual.iter();
    for y in (0..want.height).step_by(want.stride) {
        for x in (0..want.width).step_by(want.stride) {
            let got = residual[y * want.width + x];
            let sample = samples
                .next()
                .unwrap()
                .expect("a residual is finite everywhere");
            let scale = want.noise.max(sample.abs());
            assert!(
                (got - sample).abs() <= 1e-4 * scale,
                "({x}, {y}): {got} against {sample}"
            );
        }
    }

    let whole = result.blend(&NoiseBands::default()).unwrap();
    assert!(whole[0].is_nan(), "a pixel with no data stays without");
}

#[test]
fn a_tiny_model_stops_when_asked() {
    if !runtime_present() {
        return;
    }
    let want = expected();
    let pixels = image();
    let mut model = open(Backend::Cpu, &BackendOptions::default()).unwrap();
    let view = ImageView::new(&pixels, want.width, want.height, 1).unwrap();
    let stop = || true;
    let options = DenoiseOptions {
        progress: None,
        cancel: Some(&stop),
    };
    assert!(matches!(
        model.denoise(view, &options),
        Err(ModelError::Cancelled)
    ));
}

/// The tiny model at `place`, with a card listing `parity` instead of its own.
fn listing(parity: &[&str], place: &Path) -> ModelBundle {
    let tiny = fixtures().join("tiny");
    for file in ["model-f32.onnx", "model-f16.onnx"] {
        std::fs::copy(tiny.join(file), place.join(file)).unwrap();
    }
    let text = std::fs::read_to_string(tiny.join("card.json")).unwrap();
    let mut card: serde_json::Value = serde_json::from_str(&text).unwrap();
    card["parity"] = serde_json::json!(parity);
    std::fs::write(place.join("card.json"), card.to_string()).unwrap();
    ModelBundle::open(place).unwrap()
}

#[test]
fn a_mac_picks_the_gpu_unless_the_card_leaves_it_out() {
    if !runtime_present() || !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return;
    }
    let options = BackendOptions::default();
    let detect = |parity: &[&str]| {
        let place = tempfile::tempdir().unwrap();
        Backend::detect(&listing(parity, place.path()), &options).unwrap()
    };
    let gpu = Backend::CoreMl(CoreMlUnits::Gpu);
    assert_eq!(detect(&[]), gpu);
    assert_eq!(detect(&["neural-engine", "coreml-gpu", "cpu"]), gpu);
    assert_eq!(
        detect(&["neural-engine", "cpu"]),
        Backend::CoreMl(CoreMlUnits::NeuralEngine)
    );
    assert_eq!(detect(&["cpu"]), Backend::Cpu);
    assert_eq!(gpu.default_precision(), Precision::F32);
}

#[test]
fn an_f32_graph_is_refused_on_the_neural_engine() {
    if !runtime_present() || !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return;
    }
    let options = BackendOptions {
        precision: Some(Precision::F32),
        ..Default::default()
    };
    let engine = Backend::CoreMl(CoreMlUnits::NeuralEngine);
    assert!(matches!(
        open(engine, &options),
        Err(ModelError::ModelContract(_))
    ));
}
