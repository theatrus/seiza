use seiza_fits::{HeaderValue, WriteHeaderCard};
use seiza_stacking::{FitsFrame, LinearImage, write_processed_image_fits_f32};
use std::process::Command;
fn write(path: &std::path::Path, channels: usize, data: Vec<f32>, transfer: &str) {
    write_processed_image_fits_f32(
        path,
        &LinearImage::new(1, 1, channels, data).unwrap(),
        &[],
        &[WriteHeaderCard::new(
            "SEIZATRF",
            HeaderValue::String(transfer.into()),
        )],
    )
    .unwrap();
}
#[test]
fn blend_commands_preserve_transfer_and_refuse_display_data_as_linear() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path();
    write(&dir.join("rgb.fits"), 3, vec![0.2, 0.3, 0.4], "LINEAR");
    write(&dir.join("delta.fits"), 3, vec![0.1, 0.2, 0.3], "LINEAR");
    write(&dir.join("line.fits"), 1, vec![0.04], "LINEAR");
    let source = |name: &str| serde_json::json!({"name":name,"revision":"v1","transfer":"linear"});
    let settings = serde_json::json!({"lines":[{"plan":{"schemaVersion":1,"sources":{"width":1,"height":1,"narrowband":source("ha"),"references":[source("red")]},"coefficients":[0.2],"origin":"manual","diagnostics":null},"referenceChannels":[0],"amount":0.25}]});
    std::fs::write(dir.join("settings.json"), settings.to_string()).unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_seiza"))
            .current_dir(dir)
            .args(["color", "blend"])
            .args(args)
            .output()
            .unwrap()
    };
    for args in [
        vec![
            "emission",
            "--broadband",
            "rgb.fits",
            "--residual",
            "line.fits",
            "--settings",
            "settings.json",
            "--output",
            "emission.fits",
        ],
        vec![
            "contribution",
            "--base",
            "rgb.fits",
            "--contribution",
            "delta.fits",
            "--gains",
            "2,1,0.5",
            "--output",
            "added.fits",
        ],
    ] {
        let out = run(&args);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    assert!(
        (FitsFrame::open(dir.join("emission.fits"))
            .unwrap()
            .image
            .data[0]
            - 0.25)
            .abs()
            < 1e-6
    );
    let added = FitsFrame::open(dir.join("added.fits")).unwrap();
    assert!((added.image.data[2] - 0.55).abs() < 1e-6);
    write(&dir.join("display.fits"), 3, vec![0.4, 0.5, 0.6], "DISPLAY");
    write(
        &dir.join("display2.fits"),
        3,
        vec![0.4, 0.5, 0.6],
        "DISPLAY",
    );
    let out = run(&[
        "structure",
        "--narrowband",
        "display.fits",
        "--broadband",
        "display2.fits",
        "--amount",
        "0.25",
        "--output",
        "structure.fits",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let output = FitsFrame::open(dir.join("structure.fits")).unwrap();
    assert!(
        output.headers.iter().any(|(key, value)| key == "SEIZATRF"
            && matches!(value,HeaderValue::String(s) if s=="DISPLAY"))
    );
    assert!(
        !run(&[
            "contribution",
            "--base",
            "display.fits",
            "--contribution",
            "delta.fits",
            "--output",
            "refused.fits"
        ])
        .status
        .success()
    );
    assert!(!dir.join("refused.fits").exists());
}

#[test]
fn calibrated_flux_cli_preserves_wcs_units_sign_and_file_provenance() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path();
    write_processed_image_fits_f32(
        dir.join("residual.fits"),
        &LinearImage::new(1, 1, 1, vec![-4.0]).unwrap(),
        &[
            ("CRVAL1".into(), HeaderValue::Float(123.0)),
            ("CRVAL2".into(), HeaderValue::Float(20.0)),
            ("CRPIX1".into(), HeaderValue::Float(1.0)),
            ("CRPIX2".into(), HeaderValue::Float(1.0)),
            ("CTYPE1".into(), HeaderValue::String("RA---TAN".into())),
            ("CTYPE2".into(), HeaderValue::String("DEC--TAN".into())),
            ("CD1_1".into(), HeaderValue::Float(-0.001)),
            ("CD1_2".into(), HeaderValue::Float(0.0)),
            ("CD2_1".into(), HeaderValue::Float(0.0)),
            ("CD2_2".into(), HeaderValue::Float(0.001)),
        ],
        &[WriteHeaderCard::new(
            "SEIZATRF",
            HeaderValue::String("LINEAR".into()),
        )],
    )
    .unwrap();
    let source =
        |name: &str| serde_json::json!({"name":name,"revision":"exact-v1","transfer":"linear"});
    let sources = serde_json::json!({"width":1,"height":1,"narrowband":source("ha"),"references":[source("red")]});
    let settings = serde_json::json!({"plans":[{"schemaVersion":1,"sources":sources,"coefficients":[0.2],"origin":"manual","diagnostics":null}],"calibration":{"revision":"lab-v1","unit":"test-flux/pixel","lines":["ha"],"sources":[sources],"narrowbandResponse":[[1.0]],"referenceResponse":[[[1.0]]],"continuumCoefficients":[[0.2]],"residualZeroLevels":[0.0],"residualCovariance":null}});
    std::fs::write(dir.join("response.json"), settings.to_string()).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .current_dir(dir)
        .args([
            "color",
            "blend",
            "flux",
            "--residual",
            "residual.fits",
            "--settings",
            "response.json",
            "--output",
            "flux.fits",
            "--report",
            "report.json",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let flux = FitsFrame::open(dir.join("flux.fits")).unwrap();
    assert_eq!(flux.image.data, vec![-5.0]);
    assert!(
        flux.headers
            .iter()
            .any(|(k, v)| k == "CRVAL1" && v.as_f64() == Some(123.0))
    );
    assert!(
        flux.headers
            .iter()
            .any(|(k, v)| k == "BUNIT" && v.as_str() == Some("test-flux/pixel"))
    );
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("report.json")).unwrap()).unwrap();
    assert!(report["residualInputs"][0]["sha256"].as_str().is_some());
    assert_eq!(report["calibration"]["revision"], "lab-v1");
}

#[test]
fn screen_stars_cli_can_reduce_the_rgb_star_layer() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path();
    let mut original = Vec::new();
    for value in [0.05, 0.2, 0.05, 0.2, 0.8, 0.2, 0.05, 0.2, 0.05] {
        for color in [1.0, 0.5, 0.25] {
            original.push(1.0 - 0.8 * (1.0 - value * color));
        }
    }
    for (name, data) in [
        ("nebula", [0.1, 0.2, 0.3].repeat(9)),
        ("original", original),
        ("starless", vec![0.2; 27]),
    ] {
        write_processed_image_fits_f32(
            dir.join(format!("{name}.fits")),
            &LinearImage::new(3, 3, 3, data).unwrap(),
            &[],
            &[WriteHeaderCard::new(
                "SEIZATRF",
                HeaderValue::String("DISPLAY".into()),
            )],
        )
        .unwrap();
    }
    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .current_dir(dir)
        .args([
            "color",
            "screen-stars",
            "--nebula",
            "nebula.fits",
            "--original-rgb",
            "original.fits",
            "--starless-rgb",
            "starless.fits",
            "--reduction",
            "0.5",
            "--output",
            "reduced.fits",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let image = FitsFrame::open(dir.join("reduced.fits")).unwrap().image;
    for (actual, expected) in image.data[12..15].iter().zip([0.55, 0.4, 0.3875]) {
        assert!((actual - expected).abs() < 1e-6);
    }
}
