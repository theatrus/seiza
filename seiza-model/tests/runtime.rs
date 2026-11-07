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
    assert_eq!(
        std::env::var("ORT_DISABLE_TELEMETRY").as_deref(),
        Ok("1"),
        "Microsoft's builds report to Microsoft unless this is set first"
    );
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

/// A model from `fixtures/shapes`, on the CPU provider.
fn shape(name: &str, options: &BackendOptions) -> Result<NoiseModel, ModelError> {
    let bundle = ModelBundle::open(fixtures().join("shapes").join(name))?;
    NoiseModel::open(&bundle, Backend::Cpu, options)
}

/// Run `model` and hold it to its 1 × 1 convolution, `image_weight` on the
/// image and `noise_weight` on the noise plane, worked out at every pixel. A
/// constant map reads as `asinh(1)`.
fn matches_by_hand(mut model: NoiseModel, image_weight: f32, noise_weight: f32, tolerance: f32) {
    let Expected { width, height, .. } = expected();
    let pixels = image();
    let view = ImageView::new(&pixels, width, height, 1).unwrap();
    let result = model.denoise(view, &DenoiseOptions::default()).unwrap();
    let scales = result.report.channels[0];
    let (level, noise) = (scales.background, scales.noise);
    let found = result.residual(0).unwrap();
    let mut checked = 0;
    for (&pixel, &residual) in pixels.iter().zip(found) {
        if pixel.is_nan() {
            continue;
        }
        let picture = ((pixel - level) / noise).asinh();
        let wanted = (image_weight * picture + noise_weight * 1.0_f32.asinh()) * noise;
        assert!(
            (residual - wanted).abs() <= tolerance * noise,
            "{pixel}: {residual}, not {wanted}"
        );
        checked += 1;
    }
    assert!(checked > width * height / 2);
}

#[test]
fn the_card_says_what_a_graph_reads() {
    if !runtime_present() {
        return;
    }
    let options = BackendOptions::default();
    let (image, noise) = (0.25, 0.5);
    matches_by_hand(shape("two-planes", &options).unwrap(), image, noise, 1e-3);
    matches_by_hand(shape("one-plane", &options).unwrap(), image, 0.0, 1e-3);
    matches_by_hand(shape("reversed", &options).unwrap(), image, noise, 1e-3);
    matches_by_hand(shape("two-tensors", &options).unwrap(), image, noise, 1e-3);
}

#[test]
fn a_graph_of_a_fixed_size_takes_its_own_tile_and_no_other() {
    if !runtime_present() {
        return;
    }
    let model = shape("fixed", &BackendOptions::default()).unwrap();
    assert_eq!(model.tiling().tile, 64);
    matches_by_hand(model, 0.25, 0.5, 1e-3);

    let asked = |tile| BackendOptions {
        tile: Some(tile),
        ..Default::default()
    };
    assert!(shape("fixed", &asked(64)).is_ok());
    let Err(ModelError::ModelContract(reason)) = shape("fixed", &asked(128)) else {
        panic!("a tile the graph cannot take ran");
    };
    assert!(reason.contains("128"), "{reason}");
}

#[test]
fn a_graph_says_what_its_tensors_hold_whatever_its_precision() {
    if !runtime_present() {
        return;
    }
    let options = BackendOptions {
        precision: Some(Precision::F16),
        ..Default::default()
    };
    for name in ["half-inside", "half-tensors", "half-in"] {
        matches_by_hand(shape(name, &options).unwrap(), 0.25, 0.5, 0.02);
    }
}

#[test]
fn a_graph_that_differs_from_its_card_is_refused_when_it_opens() {
    if !runtime_present() {
        return;
    }
    let options = BackendOptions::default();
    for (name, says) in [
        ("three-planes", "holds 3 planes, not 2"),
        ("three-out", "holds 3 planes, not 1"),
        ("other-input", "reads planes"),
        ("missing-input", "names the input level"),
        ("other-output", "names the output clean"),
        ("extra-input", "reads extra"),
        ("fixed-small", "is 32 × 32"),
    ] {
        match shape(name, &options) {
            Err(ModelError::ModelContract(reason)) => {
                assert!(reason.contains(says), "{name}: {reason}");
            }
            Err(other) => panic!("{name}: {other}"),
            Ok(_) => panic!("{name} opened"),
        }
    }
}

/// ONNX Runtime's entries in the temp folder made by this process, which a
/// killed process would leave behind.
fn left_in_temp() -> Vec<String> {
    let mine = format!("-{}-", std::process::id());
    std::fs::read_dir(std::env::temp_dir())
        .unwrap()
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with("onnxruntime-") && name.contains(&mine))
        .collect()
}

#[test]
fn a_core_ml_session_keeps_out_of_the_temp_folder() {
    if !runtime_present() || !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return;
    }
    let cache = tempfile::tempdir().unwrap();
    let options = BackendOptions {
        cache_dir: Some(cache.path().to_path_buf()),
        ..Default::default()
    };
    let open_model = open(Backend::CoreMl(CoreMlUnits::Cpu), &options).unwrap();
    assert_eq!(left_in_temp(), Vec::<String>::new());
    let held = std::fs::read_dir(cache.path().join("models")).unwrap();
    assert_eq!(held.count(), 1, "the compiled model is in the cache");
    drop(open_model);
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
