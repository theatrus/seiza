use std::process::Command;

#[test]
fn analysis_writes_failure_diagnostics_and_never_overwrites() {
    let dir = tempfile::tempdir().unwrap();
    let json = dir.path().join("report.json");
    let html = dir.path().join("report.html");
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_seiza"))
            .arg("analyze")
            .arg("--metrics-only")
            .arg(dir.path().join("missing.fits"))
            .arg("--json")
            .arg(&json)
            .arg("--html")
            .arg(&html)
            .output()
            .unwrap()
    };
    let first = run();
    assert!(!first.status.success());
    let bytes = std::fs::read(&json).unwrap();
    let report: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(report["frames"][0]["measurement_status"], "failed");
    assert_eq!(report["frames"][0]["depth_status"], "not_requested");
    assert_eq!(report["options"]["metrics_only"], true);
    assert!(
        std::fs::read_to_string(&html)
            .unwrap()
            .contains("Seiza project analysis")
    );
    let second = run();
    assert!(!second.status.success());
    assert!(String::from_utf8_lossy(&second.stderr).contains("must not already exist"));
    assert_eq!(std::fs::read(json).unwrap(), bytes);
}

#[test]
fn osc_cli_reports_components_without_counting_them_as_sources() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("osc.fits");
    let headers = [seiza_fits::WriteHeaderCard::new(
        "BAYERPAT",
        seiza_fits::HeaderValue::String("RGGB".into()),
    )];
    seiza_image_io::save_f32(
        &input,
        80,
        80,
        seiza_fits::F32ImageData::Mono(&vec![1000.; 6400]),
        &seiza_image_io::SaveOptions {
            headers: &headers,
            ..Default::default()
        },
    )
    .unwrap();
    let json = dir.path().join("osc.json");
    let output = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .arg("analyze")
        .arg(&input)
        .arg("--metrics-only")
        .arg("--json")
        .arg(&json)
        .arg("--html")
        .arg(dir.path().join("osc.html"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("1 sources (4 component records)"));
    let report: serde_json::Value = serde_json::from_slice(&std::fs::read(json).unwrap()).unwrap();
    assert_eq!(report["schema_version"], 3);
    assert_eq!(report["groups"][0]["component"], "R");
}
