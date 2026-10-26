use std::process::Command;

#[test]
fn analysis_writes_failure_diagnostics_and_never_overwrites() {
    let dir = tempfile::tempdir().unwrap();
    let json = dir.path().join("report.json");
    let html = dir.path().join("report.html");
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_seiza"))
            .arg("analyze")
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
    assert_eq!(report["frames"][0]["status"], "rejected");
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
