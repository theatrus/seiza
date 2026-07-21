use seiza_fits::{F32ImageData, write_f32_image};
use std::process::Command;

#[test]
fn stretch_cli_applies_parameterized_percentile_asinh() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("linear.fits");
    let output = directory.path().join("preview.png");
    let values = (0..100)
        .map(|index| index as f32 / 99.0)
        .collect::<Vec<_>>();
    write_f32_image(&input, 10, 10, F32ImageData::Mono(&values), &[]).unwrap();

    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "stretch",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "percentile-asinh",
            "--black-percentile",
            "0.01",
            "--white-percentile",
            "0.995",
            "--strength",
            "10",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let preview = image::open(output).unwrap().to_luma8();
    assert_eq!(preview.dimensions(), (10, 10));
    assert_eq!(preview.get_pixel(0, 0).0[0], 0);
    assert_eq!(preview.get_pixel(9, 9).0[0], 255);
    assert!(preview.get_pixel(5, 5).0[0] > 128);
}

#[test]
fn stretch_cli_accepts_manual_ghs_parameters() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("linear.fits");
    let output = directory.path().join("ghs.png");
    let values = (0..100)
        .map(|index| index as f32 / 99.0)
        .collect::<Vec<_>>();
    write_f32_image(&input, 10, 10, F32ImageData::Mono(&values), &[]).unwrap();

    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "stretch",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "ghs",
            "--stretch-factor",
            "2.3978952727983707",
            "--symmetry-point",
            "0",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let preview = image::open(output).unwrap().to_luma8();
    assert_eq!(preview.dimensions(), (10, 10));
    assert_eq!(preview.get_pixel(0, 0).0[0], 0);
    assert_eq!(preview.get_pixel(9, 9).0[0], 255);
}

#[test]
fn stretch_cli_refuses_to_overwrite_its_input() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("linear.fits");
    let values = vec![0.5_f32; 4];
    write_f32_image(&input, 2, 2, F32ImageData::Mono(&values), &[]).unwrap();
    let original = std::fs::read(&input).unwrap();

    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "stretch",
            input.to_str().unwrap(),
            "--output",
            input.to_str().unwrap(),
            "identity",
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("same file"));
    assert_eq!(std::fs::read(input).unwrap(), original);
}
