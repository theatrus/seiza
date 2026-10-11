use seiza_fits::{F32ImageData, HeaderValue, WriteHeaderCard, write_f32_image};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const WIDTH: usize = 128;
const HEIGHT: usize = 128;

fn star_field(shift_x: usize, shift_y: usize) -> Vec<f32> {
    let mut values = (0..WIDTH * HEIGHT)
        .map(|index| 0.01 + ((index * 37 + index / WIDTH * 19) % 29) as f32 * 1.0e-4)
        .collect::<Vec<_>>();
    let stars = [
        (18, 17),
        (43, 22),
        (77, 16),
        (108, 31),
        (29, 53),
        (61, 67),
        (96, 58),
        (19, 91),
        (55, 103),
        (88, 94),
        (111, 110),
    ];
    for (index, (x, y)) in stars.into_iter().enumerate() {
        let (x, y) = (x + shift_x, y + shift_y);
        let peak = 5.0 + index as f32 * 0.7;
        for (dx, dy, weight) in [
            (0, 0, 1.0),
            (-1, 0, 0.45),
            (1, 0, 0.45),
            (0, -1, 0.45),
            (0, 1, 0.45),
        ] {
            let sample_x = x.checked_add_signed(dx).unwrap();
            let sample_y = y.checked_add_signed(dy).unwrap();
            values[sample_y * WIDTH + sample_x] += peak * weight;
        }
    }
    values
}

fn sensor_cards(temperature: Option<f64>) -> Vec<WriteHeaderCard> {
    let mut cards = vec![
        WriteHeaderCard::new("EXPTIME", HeaderValue::Float(60.0)),
        WriteHeaderCard::new("GAIN", HeaderValue::Integer(100)),
    ];
    if let Some(temperature) = temperature {
        cards.push(WriteHeaderCard::new(
            "CCD-TEMP",
            HeaderValue::Float(temperature),
        ));
        cards.push(WriteHeaderCard::new(
            "SET-TEMP",
            HeaderValue::Float(temperature),
        ));
    }
    cards
}

/// Two cooled lights at -10C, a few pixels apart.
fn write_lights(directory: &Path) -> Vec<PathBuf> {
    [(0, 0), (3, 2)]
        .into_iter()
        .enumerate()
        .map(|(index, (shift_x, shift_y))| {
            let path = directory.join(format!("light-{index}.fits"));
            let mut cards = sensor_cards(Some(-10.0));
            cards.push(WriteHeaderCard::new(
                "IMAGETYP",
                HeaderValue::String("LIGHT".into()),
            ));
            write_f32_image(
                &path,
                WIDTH,
                HEIGHT,
                F32ImageData::Mono(&star_field(shift_x, shift_y)),
                &cards,
            )
            .unwrap();
            path
        })
        .collect()
}

fn write_master_dark(path: &Path, temperature: Option<f64>) {
    let mut cards = sensor_cards(temperature);
    cards.push(WriteHeaderCard::new(
        "SEIZAMST",
        HeaderValue::String("DARK".into()),
    ));
    write_f32_image(
        path,
        WIDTH,
        HEIGHT,
        F32ImageData::Mono(&vec![1.0e-3; WIDTH * HEIGHT]),
        &cards,
    )
    .unwrap();
}

fn stack(lights: &[PathBuf], dark: &Path, output: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_seiza"))
        .arg("stack")
        .args(lights)
        .arg("--dark")
        .arg(dark)
        .args(["--reference", "first", "--output"])
        .arg(output)
        .output()
        .unwrap()
}

#[test]
fn common_reference_aligns_channels_without_contributing_pixels() {
    let directory = tempfile::tempdir().unwrap();
    let reference = directory.path().join("common.fits");
    let width = WIDTH + 16;
    let height = HEIGHT + 8;
    let mut reference_pixels = vec![100.0; width * height];
    for (y, row) in star_field(5, 4).chunks(WIDTH).enumerate() {
        for (x, value) in row.iter().enumerate() {
            reference_pixels[y * width + x] += value * 2.0;
        }
    }
    write_f32_image(
        &reference,
        width,
        height,
        F32ImageData::Mono(&reference_pixels),
        &[
            WriteHeaderCard::new("CRPIX1", HeaderValue::Float(64.5)),
            WriteHeaderCard::new("CRPIX2", HeaderValue::Float(64.5)),
            WriteHeaderCard::new("CRVAL1", HeaderValue::Float(120.0)),
            WriteHeaderCard::new("CRVAL2", HeaderValue::Float(30.0)),
            WriteHeaderCard::new("CTYPE1", HeaderValue::String("RA---TAN".into())),
            WriteHeaderCard::new("CTYPE2", HeaderValue::String("DEC--TAN".into())),
            WriteHeaderCard::new("CD1_1", HeaderValue::Float(-0.001)),
            WriteHeaderCard::new("CD1_2", HeaderValue::Float(0.0)),
            WriteHeaderCard::new("CD2_1", HeaderValue::Float(0.0)),
            WriteHeaderCard::new("CD2_2", HeaderValue::Float(0.001)),
        ],
    )
    .unwrap();
    for reintegrate in [false, true] {
        for (channel, gain) in [("red", 1.0), ("green", 0.5)] {
            let lights = [(0, 0), (3, 2)]
                .into_iter()
                .enumerate()
                .map(|(i, (x, y))| {
                    let path = directory.path().join(format!("{channel}-{i}.fits"));
                    let pixels = star_field(x, y)
                        .into_iter()
                        // The second light needs real within-channel gain and
                        // background correction; disabling normalization fails.
                        .map(|v| {
                            if i == 0 {
                                v * gain
                            } else {
                                v * gain * 1.5 + 0.2
                            }
                        })
                        .collect::<Vec<_>>();
                    write_f32_image(&path, WIDTH, HEIGHT, F32ImageData::Mono(&pixels), &[])
                        .unwrap();
                    path
                })
                .collect::<Vec<_>>();
            let output = directory
                .path()
                .join(format!("{channel}-{reintegrate}.fits"));
            let report = output.with_extension("json");
            let mut command = Command::new(env!("CARGO_BIN_EXE_seiza"));
            command
                .arg("stack")
                .args(&lights)
                .arg("--reference-image")
                .arg(&reference)
                .args(["--reference", "first", "--rejection", "none"])
                .arg("--output")
                .arg(&output)
                .arg("--report")
                .arg(&report);
            if reintegrate {
                command.arg("--reintegrate");
            }
            let result = command.output().unwrap();
            assert!(
                result.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            let frame = seiza_stacking::FitsFrame::open(&output).unwrap();
            assert_eq!((frame.image.width, frame.image.height), (width, height));
            let peak = frame.image.data[(17 + 4) * width + 18 + 5];
            assert!(
                (peak - 5.01 * gain).abs() < 0.1,
                "{channel}, reintegrate={reintegrate}: {peak}"
            );
            assert!(
                frame.image.data[70 * width + 70] < 0.1,
                "reference background leaked"
            );
            assert!(
                frame
                    .headers
                    .iter()
                    .any(|(key, value)| key == "CRPIX1" && *value == HeaderValue::Float(64.5))
            );
            let report: serde_json::Value =
                serde_json::from_slice(&std::fs::read(report).unwrap()).unwrap();
            assert_eq!(report["accepted_frames"], 2);
            assert_eq!(report["frames"].as_array().unwrap().len(), 2);
            assert_eq!(report["reference_integrated"], false);
            assert_eq!(report["reference"]["path"], reference.display().to_string());
            assert_eq!(
                report["normalization_reference"]["path"],
                lights[0].display().to_string()
            );
            let frames = report["frames"].as_array().unwrap();
            assert!((frames[0]["diagnostics"]["weight"][0].as_f64().unwrap() - 1.0).abs() < 0.01);
            assert!(
                frames[1]["diagnostics"]["normalization_mean_gain"]
                    .as_f64()
                    .unwrap()
                    < 0.9
            );
        }
    }
}

#[test]
fn common_reference_can_also_be_an_explicit_light_once() {
    let directory = tempfile::tempdir().unwrap();
    let lights = write_lights(directory.path());
    for reintegrate in [false, true] {
        let output = directory.path().join(format!("stack-{reintegrate}.fits"));
        let mut command = Command::new(env!("CARGO_BIN_EXE_seiza"));
        command
            .arg("stack")
            .args(&lights)
            .arg("--reference-image")
            .arg(&lights[0])
            .arg("--output")
            .arg(output);
        if reintegrate {
            command.arg("--reintegrate");
        }
        let result = command.output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stdout).contains("2 accepted frame(s)"));
    }
}

#[test]
fn common_reference_is_not_calibrated_with_the_channel_masters() {
    let directory = tempfile::tempdir().unwrap();
    let lights = write_lights(directory.path());
    let reference = directory.path().join("reference.fits");
    write_f32_image(
        &reference,
        WIDTH,
        HEIGHT,
        F32ImageData::Mono(&star_field(5, 4)),
        &sensor_cards(Some(0.0)),
    )
    .unwrap();
    let dark = directory.path().join("dark.fits");
    write_master_dark(&dark, Some(-10.0));
    for reintegrate in [false, true] {
        let output = directory.path().join(format!("stack-{reintegrate}.fits"));
        let mut command = Command::new(env!("CARGO_BIN_EXE_seiza"));
        command
            .arg("stack")
            .args(&lights)
            .arg("--reference-image")
            .arg(&reference)
            .arg("--dark")
            .arg(&dark)
            .arg("--output")
            .arg(&output);
        if reintegrate {
            command.arg("--reintegrate");
        }
        let result = command.output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stdout).contains("2 accepted frame(s)"));
    }
    // The same incompatible master must still reject the actual lights.
    write_master_dark(&dark, Some(0.0));
    let output = directory.path().join("rejected.fits");
    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .arg("stack")
        .args(&lights)
        .arg("--reference-image")
        .arg(&reference)
        .arg("--dark")
        .arg(&dark)
        .arg("--output")
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("temperature light=-10C master=0C"));
    assert!(!output.exists());
}

#[test]
fn common_reference_rejects_output_aliases_with_either_channel_selection() {
    let directory = tempfile::tempdir().unwrap();
    let lights = write_lights(directory.path());
    let reference = directory.path().join("reference.fits");
    std::fs::copy(&lights[0], &reference).unwrap();
    let original = std::fs::read(&reference).unwrap();
    for selection in [None, Some("first"), Some("auto")] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_seiza"));
        command
            .arg("stack")
            .args(&lights)
            .arg("--reference-image")
            .arg(&reference)
            .arg("--output")
            .arg(&reference);
        if let Some(selection) = selection {
            command.args(["--reference", selection]);
        }
        let result = command.output().unwrap();
        assert!(!result.status.success());
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(stderr.contains("same file"), "{stderr}");
        assert_eq!(std::fs::read(&reference).unwrap(), original);
    }
}

#[test]
fn a_dark_master_at_the_lights_temperature_calibrates_them() {
    // The master's CCD-TEMP was dropped on load, so every cooled light was
    // refused: "temperature light=-10C, master did not record one".
    let directory = tempfile::tempdir().unwrap();
    let lights = write_lights(directory.path());
    let dark = directory.path().join("master-dark.fits");
    write_master_dark(&dark, Some(-10.0));
    let output = directory.path().join("stack.fits");

    let result = stack(&lights, &dark, &output);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(result.status.success(), "{stderr}");
    assert!(!stderr.contains("temperature"), "{stderr}");
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(stdout.contains("2 accepted frame(s)"), "{stdout}");
    assert!(output.exists());
}

#[test]
fn a_dark_master_at_another_temperature_is_refused() {
    let directory = tempfile::tempdir().unwrap();
    let lights = write_lights(directory.path());
    let dark = directory.path().join("master-dark.fits");
    write_master_dark(&dark, Some(0.0));
    let output = directory.path().join("stack.fits");

    let result = stack(&lights, &dark, &output);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(!result.status.success(), "{stderr}");
    assert!(
        stderr.contains("temperature light=-10C master=0C"),
        "{stderr}"
    );
    assert!(!output.exists());
}

#[test]
fn a_dark_master_without_a_temperature_is_used_with_a_warning() {
    let directory = tempfile::tempdir().unwrap();
    let lights = write_lights(directory.path());
    let dark = directory.path().join("master-dark.fits");
    write_master_dark(&dark, None);
    let output = directory.path().join("stack.fits");

    let result = stack(&lights, &dark, &output);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(result.status.success(), "{stderr}");
    assert!(
        stderr.contains("master dark records no sensor temperature"),
        "{stderr}"
    );
    assert!(output.exists());
}
