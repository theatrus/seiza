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
            } else if kind == "flat" || kind == "dark" {
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
                if kind == "flat" || (kind == "dark" && count >= 3) {
                    "temporal median"
                } else {
                    "unclipped mean"
                }
            );
            assert_eq!(report["defect_pixels_replaced"], 0);
            assert!(configuration["defect_suppression"].is_null());
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
    assert!(stdout.contains("--star-mask"), "{stdout}");
}

#[test]
fn masked_flat_cli_reports_coverage_and_never_writes_an_unsupported_master() {
    let directory = tempfile::tempdir().unwrap();
    let paths = (0..4)
        .map(|index| directory.path().join(format!("flat-{index}.fits")))
        .collect::<Vec<_>>();
    for stationary in [false, true] {
        for (index, path) in paths.iter().enumerate() {
            let mut pixels = vec![1000.0; 128 * 64];
            let center = if stationary || index < 2 { 32 } else { 96 };
            for y in 30..34 {
                for x in center - 2..center + 2 {
                    pixels[y * 128 + x] = 60000.0;
                }
            }
            write_f32_image(
                path,
                128,
                64,
                F32ImageData::Mono(&pixels),
                &[WriteHeaderCard::new(
                    "IMAGETYP",
                    HeaderValue::String("FLAT".into()),
                )],
            )
            .unwrap();
        }
        let output = directory.path().join(format!("master-{stationary}.fits"));
        let report = directory.path().join(format!("report-{stationary}.json"));
        let args = ["--star-mask", "--saturation-level", "60000"];
        if stationary {
            let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
                .args(["master", "flat"])
                .args(&paths)
                .arg("--output")
                .arg(&output)
                .arg("--report")
                .arg(&report)
                .args(args)
                .output()
                .unwrap();
            assert!(!result.status.success());
            assert!(String::from_utf8_lossy(&result.stderr).contains("insufficient coverage"));
            assert!(!output.exists());
            assert!(!report.exists());
            let mask = directory.path().join("failed-mask.fits");
            let failed = Command::new(env!("CARGO_BIN_EXE_seiza"))
                .args(["master", "flat"])
                .args(&paths)
                .arg("--output")
                .arg(&output)
                .args(args)
                .arg("--defect-suppression")
                .arg("--defect-mask")
                .arg(&mask)
                .output()
                .unwrap();
            assert!(!failed.status.success());
            assert!(!mask.exists());
            assert!(!output.exists());
        } else {
            let result = run_master("flat", &paths, &output, &report, &args);
            let report = read_report(&report);
            let masking = &report["flat_star_masking"];
            assert_eq!(masking["minimum_clean_samples"], 2);
            assert_eq!(masking["maximum_clean_samples"], 4);
            assert_eq!(masking["unknown_saturation_inputs"], 0);
            assert!(masking["low_coverage_samples"].as_u64().unwrap() > 0);
            assert!(report["masked_samples"].as_u64().unwrap() > 0);
            assert_eq!(
                report["accepted_samples"].as_u64().unwrap()
                    + report["rejected_samples"].as_u64().unwrap()
                    + report["masked_samples"].as_u64().unwrap(),
                128 * 64 * 4
            );
            assert_eq!(
                report["configuration"]["fallback_center"],
                "none; insufficient coverage fails"
            );
            for input in report["inputs"].as_array().unwrap() {
                assert_eq!(
                    input["accepted_samples"].as_u64().unwrap()
                        + input["rejected_samples"].as_u64().unwrap()
                        + input["masked_samples"].as_u64().unwrap(),
                    128 * 64
                );
            }
            assert!(String::from_utf8_lossy(&result.stdout).contains("retained coverage 2..4"));
            assert!(String::from_utf8_lossy(&result.stderr).contains("fewer than three retained"));
            let image = FitsImage::open(&output).unwrap();
            assert_eq!(
                image.header("STARMASK").and_then(HeaderValue::as_bool),
                Some(true)
            );
        }
    }
}

#[test]
fn flat_defect_suppression_and_mask_report_actual_changes() {
    for extension in ["fits", "xisf"] {
        for contaminated in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let mut pixels = (0..32 * 32)
                .map(|i| {
                    1000.0 + (i % 32) as f32 + (i / 32) as f32 * 0.5 + ((i * 37) % 97) as f32 / 97.0
                })
                .collect::<Vec<_>>();
            if contaminated {
                // Edges have five neighbors; corners have only three and
                // deliberately cannot be corrected by the existing filter.
                for index in [
                    8,
                    8 * 32,
                    8 * 32 + 31,
                    31 * 32 + 8,
                    16 * 32 + 16,
                    0,
                    31,
                    31 * 32,
                    32 * 32 - 1,
                ] {
                    pixels[index] = 50000.0;
                }
            }
            let paths = (0..3)
                .map(|i| {
                    let path = directory.path().join(format!("flat-{i}.{extension}"));
                    seiza_image_io::save_f32(
                        &path,
                        32,
                        32,
                        F32ImageData::Mono(&pixels),
                        &seiza_image_io::SaveOptions::default(),
                    )
                    .unwrap();
                    path
                })
                .collect::<Vec<_>>();
            let mut results = Vec::new();
            for mode in ["disabled", "enabled", "mask", "repeat"] {
                let output = directory.path().join(format!("{mode}.fits"));
                let report_path = directory.path().join(format!("{mode}.json"));
                let mask_path = directory.path().join(format!("{mode}-mask.fits"));
                let mut args = Vec::new();
                if mode != "disabled" {
                    args.push("--defect-suppression");
                }
                if mode == "mask" || mode == "repeat" {
                    args.extend(["--defect-mask", mask_path.to_str().unwrap()]);
                }
                // Exercise the combination with explicit sigma settings and
                // native star masking on these unsaturated isolated impulses.
                args.extend(["--star-mask", "--sigma-low", "3", "--sigma-high", "3"]);
                let result = run_master("flat", &paths, &output, &report_path, &args);
                let report = read_report(&report_path);
                let expected = if contaminated && mode != "disabled" {
                    5
                } else {
                    0
                };
                assert_eq!(report["defect_pixels_replaced"], expected);
                assert_eq!(
                    report["configuration"]["defect_suppression"].is_null(),
                    mode == "disabled"
                );
                if mode != "disabled" {
                    assert_eq!(
                        report["configuration"]["defect_suppression"]["high_sigma"],
                        16.0
                    );
                    assert_eq!(
                        report["configuration"]["defect_suppression"]["low_sigma"],
                        16.0
                    );
                }
                assert!(
                    String::from_utf8_lossy(&result.stdout)
                        .contains(&format!("Master defects corrected: {expected}"))
                );
                let Pixels::F32(after) = seiza_image_io::read_astronomy(&output)
                    .unwrap()
                    .image
                    .pixels
                else {
                    panic!()
                };
                if mask_path.exists() {
                    let mask = seiza_image_io::read_astronomy(&mask_path).unwrap().image;
                    assert_eq!((mask.width, mask.height), (32, 32));
                    let Pixels::F32(mask) = mask.pixels else {
                        panic!()
                    };
                    let before: &Vec<f32> = &results[0];
                    assert_eq!(
                        mask.iter().filter(|&&v| v == 1.0).count(),
                        expected as usize
                    );
                    for ((&a, &b), &marked) in before.iter().zip(&after).zip(&mask) {
                        assert_eq!(marked, if a.to_bits() != b.to_bits() { 1.0 } else { 0.0 });
                    }
                    assert!(report["defect_mask"]["sha256"].is_string());
                }
                if mode == "mask" || mode == "repeat" {
                    assert_eq!(after, results[1]);
                }
                results.push(after);
            }
        }
    }
}

#[test]
fn defect_options_reject_invalid_commands_and_output_aliases() {
    for kind in ["bias", "dark", "flat"] {
        let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
            .args(["master", kind, "a.fits", "b.fits", "--output", "out.fits"])
            .args(if kind == "flat" {
                vec!["--defect-mask", "mask.fits"]
            } else {
                vec!["--defect-suppression"]
            })
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(
            String::from_utf8_lossy(&result.stderr).contains(if kind == "flat" {
                "--defect-suppression"
            } else {
                "unexpected argument"
            })
        );
    }
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input.fits");
    write_raw(&input, "flat", 1000.0, "H");
    let original = std::fs::read(&input).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args(["master", "flat"])
        .arg(&input)
        .arg(directory.path().join("other.fits"))
        .arg("--output")
        .arg(directory.path().join("out.fits"))
        .arg("--defect-suppression")
        .arg("--defect-mask")
        .arg(&input)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(std::fs::read(input).unwrap(), original);
    assert!(!directory.path().join("out.fits").exists());
}

#[test]
fn rgb_defect_mask_combines_channels_at_the_same_coordinate() {
    let directory = tempfile::tempdir().unwrap();
    let mut data = (0..32 * 32)
        .flat_map(|i| {
            let value =
                1000.0 + (i % 32) as f32 + (i / 32) as f32 * 0.5 + ((i * 37) % 97) as f32 / 97.0;
            [value, value + 100.0, value + 200.0]
        })
        .collect::<Vec<_>>();
    let first = 16 * 32 + 16;
    let second = 8 * 32 + 8;
    data[first * 3] = 50000.0;
    data[first * 3 + 1] = 50000.0;
    data[second * 3 + 2] = 50000.0;
    let paths = (0..3)
        .map(|i| {
            let path = directory.path().join(format!("rgb-{i}.fits"));
            seiza_image_io::save_f32(
                &path,
                32,
                32,
                F32ImageData::RgbInterleaved(&data),
                &seiza_image_io::SaveOptions::default(),
            )
            .unwrap();
            path
        })
        .collect::<Vec<_>>();
    let output = directory.path().join("master.fits");
    let report = directory.path().join("report.json");
    let mask = directory.path().join("mask.fits");
    run_master(
        "flat",
        &paths,
        &output,
        &report,
        &[
            "--defect-suppression",
            "--defect-mask",
            mask.to_str().unwrap(),
        ],
    );
    assert_eq!(read_report(&report)["defect_pixels_replaced"], 3);
    let image = seiza_image_io::read_astronomy(&mask).unwrap().image;
    assert_eq!(image.planes, 1);
    let Pixels::F32(data) = image.pixels else {
        panic!()
    };
    assert_eq!(data.iter().sum::<f32>(), 2.0);
    assert_eq!(data[first], 1.0);
    assert_eq!(data[second], 1.0);
}

#[test]
fn experimental_spatial_replacement_is_opt_in_flat_only_and_reported_separately() {
    for extension in ["fits", "xisf"] {
        let directory = tempfile::tempdir().unwrap();
        let mut data = vec![1000.0; 16 * 16];
        data[8 * 16 + 8] = 50000.0;
        let paths = (0..3)
            .map(|i| {
                let path = directory.path().join(format!("flat-{i}.{extension}"));
                seiza_image_io::save_f32(
                    &path,
                    16,
                    16,
                    F32ImageData::Mono(&data),
                    &seiza_image_io::SaveOptions::default(),
                )
                .unwrap();
                path
            })
            .collect::<Vec<_>>();
        for enabled in [false, true] {
            let output = directory.path().join(format!("{enabled}.fits"));
            let report = directory.path().join(format!("{enabled}.json"));
            let args = if enabled {
                vec![
                    "--experimental-spatial-replacement",
                    "--star-mask",
                    "--defect-suppression",
                ]
            } else {
                vec![]
            };
            let result = run_master("flat", &paths, &output, &report, &args);
            let report = read_report(&report);
            assert_eq!(
                report["experimental_flat_samples_replaced"],
                if enabled { 3 } else { 0 }
            );
            assert_eq!(report["defect_pixels_replaced"], 0);
            assert_eq!(
                report["configuration"]["experimental_flat_preprocessing"].is_null(),
                !enabled
            );
            if enabled {
                assert_eq!(
                    report["configuration"]["experimental_flat_preprocessing"]["sigma"],
                    10.0
                );
                assert!(
                    String::from_utf8_lossy(&result.stdout)
                        .contains("Experimental input samples replaced: 3")
                );
            }
        }
    }
    for (kind, extra) in [
        ("bias", vec!["--experimental-spatial-replacement"]),
        ("dark", vec!["--experimental-spatial-replacement"]),
        ("flat", vec!["--spatial-sigma", "6"]),
        (
            "flat",
            vec!["--experimental-spatial-replacement", "--spatial-sigma", "0"],
        ),
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
            .args(["master", kind, "a.fits", "b.fits", "--output", "out.fits"])
            .args(extra)
            .output()
            .unwrap();
        assert!(!result.status.success());
    }
}
