use seiza_fits::{F32ImageData, HeaderValue, WriteHeaderCard, write_f32_image};
use seiza_stacking::FitsFrame;
use std::path::Path;
use std::process::{Command, Output};
fn write(path: &Path, width: usize, height: usize, data: &[f32], tag: Option<&str>) {
    let cards: Vec<_> = tag
        .map(|v| WriteHeaderCard::new("SEIZATRF", HeaderValue::String(v.into())))
        .into_iter()
        .collect();
    write_f32_image(path, width, height, F32ImageData::Mono(data), &cards).unwrap();
}
fn run(dir: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_seiza"))
        .current_dir(dir)
        .args([
            "color",
            "continuum",
            "--narrowband",
            "ha.fits",
            "--reference",
            "r.fits",
            "--saturation",
            "65535",
        ])
        .args(extra)
        .output()
        .unwrap()
}
fn success(result: &Output) {
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
fn manual_signed_residual_and_exact_plan_reuse_preserve_units_and_inputs() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write(
        &dir.join("ha.fits"),
        2,
        2,
        &[0.2, 0.1, 0.5, -0.01],
        Some("LINEAR"),
    );
    write(
        &dir.join("r.fits"),
        2,
        2,
        &[0.5, 0.3, 0.4, 0.1],
        Some("LINEAR"),
    );
    let before = std::fs::read(dir.join("ha.fits")).unwrap();
    success(&run(
        dir,
        &[
            "--coefficients",
            "0.35",
            "--reference-flux-ratios",
            "1",
            "--output",
            "emission.fits",
            "--save-plan",
            "plan.json",
            "--report",
            "report.json",
        ],
    ));
    let saved: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("plan.json")).unwrap()).unwrap();
    assert_eq!(saved["referenceFluxRatios"], serde_json::json!([1.0]));
    let result = FitsFrame::open(dir.join("emission.fits")).unwrap();
    assert!(result.image.data[1] < 0.0);
    for (v, e) in result.image.data.iter().zip([0.025, -0.005, 0.36, -0.045]) {
        assert!((*v - e).abs() < 1e-6);
    }
    assert_eq!(std::fs::read(dir.join("ha.fits")).unwrap(), before);
    assert_eq!(
        result
            .headers
            .iter()
            .find(|(k, _)| k == "SEIZATRF")
            .unwrap()
            .1
            .as_str(),
        Some("LINEAR")
    );
    assert_eq!(
        result
            .headers
            .iter()
            .find(|(k, _)| k == "SEIZACON")
            .unwrap()
            .1
            .as_str(),
        Some("EMISSION-RESIDUAL")
    );
    std::fs::write(dir.join("reuse.fits"), b"previous output").unwrap();
    success(&run(
        dir,
        &["--plan", "plan.json", "--output", "reuse.fits"],
    ));
    assert_eq!(
        FitsFrame::open(dir.join("reuse.fits")).unwrap().image.data,
        result.image.data
    );
    assert!(std::fs::read_dir(dir).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".seiza-continuum-")
    }));
    write(&dir.join("r.fits"), 2, 2, &[0.6; 4], Some("LINEAR"));
    let bad = run(dir, &["--plan", "plan.json", "--output", "stale.fits"]);
    assert!(!bad.status.success());
    assert!(!dir.join("stale.fits").exists());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("revisions"));
}

#[test]
fn untagged_inputs_require_a_declaration_and_display_is_always_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write(&dir.join("ha.fits"), 2, 2, &[1.; 4], None);
    write(&dir.join("r.fits"), 2, 2, &[1.; 4], None);
    assert!(
        !run(dir, &["--coefficients", "0.35", "--output", "out.fits"])
            .status
            .success()
    );
    success(&run(
        dir,
        &[
            "--assume-linear",
            "--coefficients",
            "0.35",
            "--output",
            "out.fits",
        ],
    ));
    write(&dir.join("ha.fits"), 2, 2, &[1.; 4], Some("DISPLAY"));
    assert!(
        !run(
            dir,
            &[
                "--assume-linear",
                "--coefficients",
                "0.35",
                "--output",
                "bad.fits"
            ]
        )
        .status
        .success()
    );
    assert!(!dir.join("bad.fits").exists());
}

#[test]
fn failed_automatic_fit_writes_a_skip_report_without_a_new_output() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    for name in ["ha.fits", "r.fits"] {
        write(&dir.join(name), 64, 64, &vec![0.0; 64 * 64], Some("LINEAR"));
    }
    let result = run(
        dir,
        &[
            "--output",
            "bad.fits",
            "--save-plan",
            "bad-plan.json",
            "--report",
            "report.json",
        ],
    );
    assert!(!result.status.success());
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["status"], "skipped");
    assert_eq!(report["reason"], "no_stars");
    assert!(!dir.join("bad.fits").exists());
    assert!(!dir.join("bad-plan.json").exists());
}

#[test]
fn automatic_apertures_and_premeasured_stars_share_the_same_plan_contract() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let positions: Vec<_> = (0..16)
        .map(|i| [35.0 + (i % 4) as f64 * 60.0, 35.0 + (i / 4) as f64 * 60.0])
        .collect();
    let field = |factor: f64, background: f64| -> Vec<f32> {
        (0..256 * 256)
            .map(|i| {
                let x = (i % 256) as f64;
                let y = (i / 256) as f64;
                (background + 0.1 * x - 0.05 * y
                    + positions
                        .iter()
                        .enumerate()
                        .map(|(i, p)| {
                            factor
                                * (1000.0 + i as f64 * 50.0)
                                * (-((x - p[0]).powi(2) + (y - p[1]).powi(2))
                                    / (2.0 * 1.4_f64.powi(2)))
                                .exp()
                        })
                        .sum::<f64>()) as f32
            })
            .collect()
    };
    write(
        &dir.join("ha.fits"),
        256,
        256,
        &field(0.35, 100.0),
        Some("LINEAR"),
    );
    write(
        &dir.join("r.fits"),
        256,
        256,
        &field(1.0, 300.0),
        Some("LINEAR"),
    );
    success(&run(
        dir,
        &["--output", "automatic.fits", "--report", "automatic.json"],
    ));
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("automatic.json")).unwrap()).unwrap();
    assert!((report["plan"]["coefficients"][0].as_f64().unwrap() - 0.35).abs() < 2e-5);
    assert!(
        report["plan"]["diagnostics"]["validationStars"]
            .as_u64()
            .unwrap()
            >= 3
    );
    let stars:Vec<_>=(0..40).map(|i|{
        let measurement=|flux:f64|serde_json::json!({"x":25.+(i%7) as f64*30.,"y":25.+(i/7) as f64*20.,"flux":flux,"peak":100.+flux/20.,"fwhm":3.,"snr":100.});
        let flux=1000.+i as f64*100.;serde_json::json!({"narrowband":measurement(0.35*flux),"references":[measurement(flux)]})
    }).collect();
    std::fs::write(dir.join("stars.json"), serde_json::to_vec(&stars).unwrap()).unwrap();
    success(&run(
        dir,
        &["--stars", "stars.json", "--save-plan", "measured.json"],
    ));
    let plan: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("measured.json")).unwrap()).unwrap();
    assert!((plan["coefficients"][0].as_f64().unwrap() - 0.35).abs() < 1e-10);
}

#[test]
fn paths_and_manual_coefficients_are_checked_before_writing() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    for name in ["ha.fits", "r.fits"] {
        write(&dir.join(name), 2, 2, &[1.; 4], Some("LINEAR"));
    }
    let before = std::fs::read(dir.join("ha.fits")).unwrap();
    assert!(
        !run(dir, &["--coefficients", "0.35", "--output", "ha.fits"])
            .status
            .success()
    );
    assert_eq!(std::fs::read(dir.join("ha.fits")).unwrap(), before);
    assert!(
        !run(dir, &["--coefficients", "-0.35", "--output", "bad.fits"])
            .status
            .success()
    );
    assert!(!dir.join("bad.fits").exists());
    assert!(
        !run(
            dir,
            &["--coefficients", "0.3,0.1", "--save-plan", "bad.json"]
        )
        .status
        .success()
    );
    assert!(!dir.join("bad.json").exists());
}

#[test]
fn signed_xisf_output_and_raw_bayer_refusal_use_the_shared_image_contract() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write(
        &dir.join("ha.fits"),
        2,
        2,
        &[0.2, 0.1, 0.5, 0.01],
        Some("LINEAR"),
    );
    write(
        &dir.join("r.fits"),
        2,
        2,
        &[0.5, 0.3, 0.4, 0.1],
        Some("LINEAR"),
    );
    success(&run(
        dir,
        &["--coefficients", "0.35", "--output", "signed.xisf"],
    ));
    let output = FitsFrame::open(dir.join("signed.xisf")).unwrap();
    assert!(output.image.data[1] < 0.0);
    assert_eq!(
        output
            .headers
            .iter()
            .find(|(k, _)| k == "SEIZATRF")
            .unwrap()
            .1
            .as_str(),
        Some("LINEAR")
    );
    write_f32_image(
        dir.join("r.fits"),
        2,
        2,
        F32ImageData::Mono(&[1.; 4]),
        &[
            WriteHeaderCard::new("SEIZATRF", HeaderValue::String("LINEAR".into())),
            WriteHeaderCard::new("BAYERPAT", HeaderValue::String("RGGB".into())),
        ],
    )
    .unwrap();
    let bad = run(dir, &["--coefficients", "0.35", "--output", "bayer.fits"]);
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("prepared mono"));
    assert!(!dir.join("bayer.fits").exists());
}
