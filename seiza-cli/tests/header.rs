use seiza_fits::{F32ImageData, FitsImage, HeaderValue, WriteHeaderCard, write_f32_image};
use std::process::Command;

#[test]
fn header_cli_sets_and_gets_fits_header_cards() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("capture.fits");
    let initial_headers = [
        WriteHeaderCard::new("OBJECT", HeaderValue::String("Corrupted_Target".into())),
        WriteHeaderCard::new("EXPTIME", HeaderValue::Float(1.0)),
    ];
    write_f32_image(
        &input,
        4,
        4,
        F32ImageData::Mono(&[100.0; 16]),
        &initial_headers,
    )
    .unwrap();

    // 1. Get initial OBJECT
    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args(["header", "get", input.to_str().unwrap(), "OBJECT"])
        .output()
        .unwrap();
    assert!(result.status.success());
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(stdout.contains("OBJECT = String(\"Corrupted_Target\")"));

    // 2. Set OBJECT to Kite_Cluster with comment
    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "header",
            "set",
            input.to_str().unwrap(),
            "OBJECT",
            "Kite_Cluster",
            "--comment",
            "NGC 6819",
        ])
        .output()
        .unwrap();
    assert!(result.status.success());

    // 3. Set numerical GAIN header (new keyword)
    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args(["header", "set", input.to_str().unwrap(), "GAIN", "160"])
        .output()
        .unwrap();
    assert!(result.status.success());

    // 4. Set EXPTIME to float 0.8
    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args(["header", "set", input.to_str().unwrap(), "EXPTIME", "0.8"])
        .output()
        .unwrap();
    assert!(result.status.success());

    // 5. Verify through FitsImage reader
    let image = FitsImage::open(&input).unwrap();
    assert_eq!(
        image.header("OBJECT"),
        Some(&HeaderValue::String("Kite_Cluster".into()))
    );
    assert_eq!(image.header("GAIN"), Some(&HeaderValue::Integer(160)));
    assert_eq!(image.header("EXPTIME"), Some(&HeaderValue::Float(0.8)));

    // 6. Verify via header get
    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args(["header", "get", input.to_str().unwrap(), "GAIN"])
        .output()
        .unwrap();
    assert!(result.status.success());
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(stdout.contains("GAIN = Integer(160)"));
}

#[test]
fn header_cli_rejects_structural_cards() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("capture.fits");
    write_f32_image(&input, 2, 2, F32ImageData::Mono(&[10.0; 4]), &[]).unwrap();

    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args(["header", "set", input.to_str().unwrap(), "BITPIX", "32"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("cannot modify structural FITS card"));
}
