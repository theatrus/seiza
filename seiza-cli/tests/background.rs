use seiza_fits::{F32ImageData, FitsImage, HeaderValue, WriteHeaderCard, write_f32_image};
use std::process::Command;

fn normalized(value: usize, extent: usize) -> f32 {
    2.0 * value as f32 / (extent - 1) as f32 - 1.0
}

#[test]
fn background_cli_writes_corrected_model_and_diagnostics() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("gradient.fits");
    let output = directory.path().join("corrected.fits");
    let model = directory.path().join("model.fits");
    let diagnostics = directory.path().join("fit.json");
    let (width, height) = (96, 72);
    let mut values = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            values.push(0.2 + 0.08 * normalized(x, width) - 0.04 * normalized(y, height));
        }
    }
    let wcs = [
        WriteHeaderCard::new("CRPIX1", HeaderValue::Float(48.0)),
        WriteHeaderCard::new("CRPIX2", HeaderValue::Float(36.0)),
        WriteHeaderCard::new("CRVAL1", HeaderValue::Float(180.0)),
        WriteHeaderCard::new("CRVAL2", HeaderValue::Float(20.0)),
        WriteHeaderCard::new("CTYPE1", HeaderValue::String("RA---TAN".into())),
        WriteHeaderCard::new("CTYPE2", HeaderValue::String("DEC--TAN".into())),
    ];
    write_f32_image(&input, width, height, F32ImageData::Mono(&values), &wcs).unwrap();

    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "background",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--model-output",
            model.to_str().unwrap(),
            "--diagnostics",
            diagnostics.to_str().unwrap(),
            "--degree",
            "1",
            "--sample-radius",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );

    let corrected = FitsImage::open(&output).unwrap();
    let corrected = match corrected.pixels {
        seiza_fits::Pixels::F32(values) => values,
        pixels => panic!("expected f32 output, got {pixels:?}"),
    };
    let left = corrected[height / 2 * width + 3];
    let right = corrected[height / 2 * width + width - 4];
    assert!((left - right).abs() < 0.003);
    let output_headers = FitsImage::open(&output).unwrap();
    assert_eq!(
        output_headers
            .header("SEIZABG")
            .and_then(HeaderValue::as_str),
        Some("SUBTRACT")
    );
    assert_eq!(output_headers.header_f64("CRVAL1"), Some(180.0));
    assert!(model.is_file());
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(diagnostics).unwrap()).unwrap();
    assert!(report["diagnostics"]["accepted_samples"].as_u64().unwrap() > 10);
}

#[test]
fn background_cli_refuses_to_overwrite_its_input() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("gradient.fits");
    let values = vec![0.2_f32; 32 * 32];
    write_f32_image(&input, 32, 32, F32ImageData::Mono(&values), &[]).unwrap();
    let original = std::fs::read(&input).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "background",
            input.to_str().unwrap(),
            "--output",
            input.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("same file"));
    assert_eq!(std::fs::read(input).unwrap(), original);
}

#[test]
fn background_cli_rejects_an_undebayered_cfa() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("raw-cfa.fits");
    let output = directory.path().join("incorrect.fits");
    let values = vec![0.2_f32; 32 * 32];
    let headers = [WriteHeaderCard::new(
        "BAYERPAT",
        HeaderValue::String("RGGB".into()),
    )];
    write_f32_image(&input, 32, 32, F32ImageData::Mono(&values), &headers).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "background",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("raw Bayer subchannels"));
    assert!(!output.exists());
}
