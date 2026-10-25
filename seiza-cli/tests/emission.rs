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
