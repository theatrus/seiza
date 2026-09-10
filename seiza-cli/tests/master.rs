use seiza_fits::{F32ImageData, FitsImage, HeaderValue, Pixels, WriteHeaderCard, write_f32_image};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn write_raw(path: &Path, kind: &str, first: f32, filter: &str) {
    write_f32_image(
        path,
        2,
        2,
        F32ImageData::Mono(&[first, 1000.0, 1000.0, 1000.0]),
        &[
            WriteHeaderCard::new("IMAGETYP", HeaderValue::String(kind.to_uppercase())),
            WriteHeaderCard::new("EXPTIME", HeaderValue::Float(2.0)),
            WriteHeaderCard::new("FILTER", HeaderValue::String(filter.into())),
        ],
    )
    .unwrap();
}

fn run_master(
    kind: &str,
    paths: &[PathBuf],
    output: &Path,
    report: &Path,
    args: &[&str],
) -> Output {
    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args(["master", kind])
        .args(paths)
        .arg("--output")
        .arg(output)
        .arg("--report")
        .arg(report)
        .args(args)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    result
}

fn read_report(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn master_cli_reports_actual_integration_for_every_kind_and_input_count() {
    for kind in ["bias", "dark", "flat"] {
        for count in [2, 3] {
            let directory = tempfile::tempdir().unwrap();
            let paths = (0..count)
                .map(|index| directory.path().join(format!("{kind}-{index}.fits")))
                .collect::<Vec<_>>();
            for path in &paths {
                write_raw(path, kind, 1000.0, "R");
            }
            let output = directory.path().join("master.fits");
            let report = directory.path().join("report.json");
            let result = run_master(kind, &paths, &output, &report, &[]);
            let (integration, rejection) = if count == 2 {
                ("mean-without-rejection", "NONE")
            } else if kind == "flat" {
                ("median-mad-sigma-clipped-mean", "MEDIAN_MAD")
            } else {
                ("two-pass-leave-one-out-sigma-clipped-mean", "LEAVE_ONE_OUT")
            };
            let report = read_report(&report);
            let configuration = &report["configuration"];
            assert_eq!(configuration["integration"], integration);
            assert_eq!(configuration["rejection_method"], rejection);
            assert_eq!(configuration["rereads_inputs"], kind != "flat");
            assert_eq!(
                configuration["fallback_center"],
                if kind == "flat" {
                    "temporal median"
                } else {
                    "unclipped mean"
                }
            );
            assert_eq!(report["input_frames"], count);
            assert_eq!(report["skipped_inputs"].as_array().unwrap().len(), 0);
            let stdout = String::from_utf8_lossy(&result.stdout);
            assert!(stdout.contains(integration), "{stdout}");
            if kind == "flat" {
                assert!(stdout.contains("scratch-backed integration"), "{stdout}");
                assert!(!stdout.contains("two-pass"), "{stdout}");
            }
            let fits = FitsImage::open(&output).unwrap();
            assert_eq!(fits.header_str("REJMETH"), Some(rejection));
        }
    }
}

#[test]
fn master_cli_flat_and_bias_fallbacks_report_the_actual_center() {
    for kind in ["bias", "flat"] {
        let directory = tempfile::tempdir().unwrap();
        let paths = [800.0, 900.0, 1000.0, 1100.0, 1200.0, 10000.0]
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                let path = directory.path().join(format!("{kind}-{index}.fits"));
                write_raw(&path, kind, value, "R");
                path
            })
            .collect::<Vec<_>>();
        let output = directory.path().join("master.fits");
        let report = directory.path().join("report.json");
        let result = run_master(
            kind,
            &paths,
            &output,
            &report,
            &["--sigma-low", "0.1", "--sigma-high", "0.1"],
        );
        assert_eq!(read_report(&report)["fallback_pixels"], 1);
        let stderr = String::from_utf8_lossy(&result.stderr);
        let (expected, message) = if kind == "flat" {
            (1.05, "wrote their temporal median")
        } else {
            (2500.0, "wrote their unclipped mean")
        };
        assert!(stderr.contains(message), "{stderr}");
        let Pixels::F32(pixels) = FitsImage::open(&output).unwrap().pixels else {
            panic!("expected f32 master");
        };
        assert!((pixels[0] - expected).abs() < 1.0e-5);
    }
}

#[test]
fn master_cli_keeps_statistics_with_their_source_when_a_middle_flat_is_skipped() {
    let directory = tempfile::tempdir().unwrap();
    let paths = ["first", "wrong-filter", "third", "last"]
        .map(|name| directory.path().join(format!("{name}.fits")));
    for (index, path) in paths.iter().enumerate() {
        write_raw(
            path,
            "flat",
            if index == 3 { 5000.0 } else { 1000.0 },
            if index == 1 { "G" } else { "R" },
        );
    }
    for requested in [3, 4] {
        let output = directory.path().join(format!("master-{requested}.fits"));
        let report = directory.path().join(format!("report-{requested}.json"));
        run_master("flat", &paths[..requested], &output, &report, &[]);
        let report = read_report(&report);
        let inputs = report["inputs"].as_array().unwrap();
        assert_eq!(inputs.len(), requested - 1);
        assert_eq!(inputs[0]["source"]["path"], paths[0].display().to_string());
        assert_eq!(inputs[1]["source"]["path"], paths[2].display().to_string());
        assert_eq!(inputs[1]["accepted_samples"], 4);
        assert_eq!(inputs[1]["rejected_samples"], 0);
        if requested == 4 {
            assert_eq!(inputs[2]["source"]["path"], paths[3].display().to_string());
            assert_eq!(inputs[2]["accepted_samples"], 3);
            assert_eq!(inputs[2]["rejected_samples"], 1);
        }
        assert_eq!(
            report["configuration"]["rejection_method"],
            if requested == 3 { "NONE" } else { "MEDIAN_MAD" }
        );
        assert_eq!(report["input_frames"], requested - 1);
        let skipped = report["skipped_inputs"].as_array().unwrap();
        assert_eq!(skipped.len(), 1);
        assert_eq!(skipped[0]["source"]["path"], paths[1].display().to_string());
        assert_eq!(skipped[0]["source"]["sha256"].as_str().unwrap().len(), 64);
        assert!(
            skipped[0]["reason"].as_str().unwrap().contains("optical")
                || skipped[0]["reason"].as_str().unwrap().contains("FILTER")
        );
    }
}

#[test]
fn flat_help_does_not_advertise_leave_one_out_thresholds() {
    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args(["master", "flat", "--help"])
        .output()
        .unwrap();
    assert!(result.status.success());
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(stdout.contains("Low sigma rejection threshold"), "{stdout}");
    assert!(!stdout.contains("leave-one-out"), "{stdout}");
}
