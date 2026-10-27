use seiza_fits::{F32ImageData, HeaderValue, WriteHeaderCard, write_f32_image};
use seiza_stacking::{
    CalibrationMasters, Continue, FitsFrame, FrameAcceptanceCriteria, FrameDisposition,
    LiveStacker, NormalizationMode, StackOptions,
};
use std::{
    path::{Path, PathBuf},
    process::Command,
};

const WIDTH: usize = 192;
const HEIGHT: usize = 160;
const STEP: usize = 112;

#[test]
fn separate_filters_reuse_a_saved_expanded_canvas() {
    let directory = tempfile::tempdir().unwrap();
    let paths = panels(directory.path());
    let canvas = directory.path().join("common.seiza-stack");
    let first = directory.path().join("ha.fits");
    let second = directory.path().join("oiii.fits");
    let run = |output: &Path, reuse: bool| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_seiza"));
        command
            .arg("stack")
            .args(&paths)
            .args(["--defects", "off", "--normalization", "none", "--output"])
            .arg(output);
        if reuse {
            command.arg("--reference-canvas").arg(&canvas);
        } else {
            command
                .args(["--reference", "expand", "--save-canvas"])
                .arg(&canvas);
        }
        let result = command.output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    };
    run(&first, false);
    run(&second, true);
    let a = FitsFrame::open(&first).unwrap();
    let b = FitsFrame::open(&second).unwrap();
    assert_eq!(
        (a.image.width, a.image.height),
        (b.image.width, b.image.height)
    );
    for key in ["CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CD1_1", "CD2_2"] {
        assert_eq!(
            a.headers.iter().find(|(k, _)| k == key),
            b.headers.iter().find(|(k, _)| k == key),
            "{key}"
        );
    }
    // A context is an input, and cannot be overwritten by any output.
    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .arg("stack")
        .args(&paths)
        .arg("--reference-canvas")
        .arg(&canvas)
        .arg("--output")
        .arg(&canvas)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(LiveStacker::open_context(&canvas).is_ok());
}

#[test]
fn joint_canvas_extends_beyond_the_integrated_filters_coverage() {
    let directory = tempfile::tempdir().unwrap();
    let paths = panels(directory.path());
    let manifest = directory.path().join("other-filters.json");
    std::fs::write(
        &manifest,
        serde_json::to_vec(&serde_json::json!([
            {"paths": [&paths[2]]}
        ]))
        .unwrap(),
    )
    .unwrap();
    let output = directory.path().join("ha.fits");
    let saved = directory.path().join("union.seiza-stack");
    let run = |target: &Path, conflicting: bool| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_seiza"));
        command
            .arg("stack")
            .args(&paths[..2])
            .args([
                "--defects",
                "off",
                "--normalization",
                "none",
                "--joint-canvas",
            ])
            .arg(&manifest)
            .arg("--save-canvas")
            .arg(&saved)
            .arg("--output")
            .arg(target);
        if conflicting {
            command.arg("--reference-canvas").arg(&saved);
        }
        command.output().unwrap()
    };
    let result = run(&output, false);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let image = FitsFrame::open(&output).unwrap().image;
    assert!(image.width >= WIDTH + 2 * STEP);
    assert!(image.data.iter().filter(|p| p.is_nan()).count() > image.data.len() / 5);
    let source = LiveStacker::open_context(&saved).unwrap();
    assert_eq!(source.view().width, image.width);
    let before = std::fs::read(&paths[2]).unwrap();
    assert!(!run(&paths[2], false).status.success());
    assert_eq!(std::fs::read(&paths[2]).unwrap(), before);
    assert!(!run(&output, true).status.success());
}

fn panels(directory: &Path) -> Vec<PathBuf> {
    let sky_width = WIDTH + 2 * STEP;
    let mut sky = (0..sky_width * HEIGHT)
        .map(|i| 0.01 + ((i * 37 + i / sky_width * 19) % 29) as f32 * 1.0e-4)
        .collect::<Vec<_>>();
    let mut seed = 721_u64;
    for row in 0..7 {
        for col in 0..19 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = 10 + col * 21 + ((seed >> 32) % 13) as usize;
            let y = 10 + row * 21 + ((seed >> 48) % 13) as usize;
            let peak = 4.0 + ((seed >> 20) % 100) as f32 / 25.0;
            for dy in -2_isize..=2 {
                for dx in -2_isize..=2 {
                    sky[(y as isize + dy) as usize * sky_width + (x as isize + dx) as usize] +=
                        peak * (-0.6 * (dx * dx + dy * dy) as f32).exp();
                }
            }
        }
    }
    (0..3)
        .map(|panel| {
            let path = directory.join(format!("panel-{panel}.fits"));
            let mut data = Vec::new();
            for y in 0..HEIGHT {
                data.extend_from_slice(
                    &sky[y * sky_width + panel * STEP..y * sky_width + panel * STEP + WIDTH],
                );
            }
            write_f32_image(
                &path,
                WIDTH,
                HEIGHT,
                F32ImageData::Mono(&data),
                &[
                    WriteHeaderCard::new(
                        "CRPIX1",
                        HeaderValue::Float(208.5 - (panel * STEP) as f64),
                    ),
                    WriteHeaderCard::new("CRPIX2", HeaderValue::Float(80.5)),
                    WriteHeaderCard::new("EXPTIME", HeaderValue::Float(60.0)),
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
            path
        })
        .collect()
}

/// A 3x3 mosaic of `panel_width` by `panel_height` panels overlapping by
/// about 12% on each side, over a crowded field whose star brightness follows
/// a steep power law, with two dithered frames per panel. The anchor is a
/// corner, so the far corner connects only through a chain of panels. Returns
/// each frame's path and origin on the sky.
fn dense_mosaic(
    directory: &Path,
    panel_width: usize,
    panel_height: usize,
) -> Vec<(PathBuf, (usize, usize))> {
    const DITHER: usize = 3;
    let step_x = panel_width * 88 / 100;
    let step_y = panel_height * 88 / 100;
    let sky_width = panel_width + 2 * step_x + DITHER;
    let sky_height = panel_height + 2 * step_y + DITHER;
    let mut sky = vec![0.01_f32; sky_width * sky_height];
    let mut seed = 0x5eed_u64;
    let mut random = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f64 / (1u64 << 31) as f64
    };
    for _ in 0..sky_width * sky_height / 160 {
        let x = 4.0 + random() * (sky_width - 8) as f64;
        let y = 4.0 + random() * (sky_height - 8) as f64;
        let peak = 0.02 + 0.0005 / (0.01 + random()).powf(1.5) as f32;
        for dy in -4_isize..=4 {
            for dx in -4_isize..=4 {
                let px = x.floor() as isize + dx;
                let py = y.floor() as isize + dy;
                let r2 = (px as f64 - x).powi(2) + (py as f64 - y).powi(2);
                sky[py as usize * sky_width + px as usize] += peak * (-r2 / 2.9).exp() as f32;
            }
        }
    }
    let mut frames = Vec::new();
    for (panel_x, panel_y) in [
        (0, 0),
        (2, 2),
        (1, 2),
        (2, 1),
        (0, 2),
        (1, 1),
        (2, 0),
        (0, 1),
        (1, 0),
    ] {
        for dither in [0, DITHER] {
            let (ox, oy) = (panel_x * step_x + dither, panel_y * step_y + dither / 3);
            let path = directory.join(format!("panel-{panel_x}{panel_y}-{dither}.fits"));
            let mut data = Vec::with_capacity(panel_width * panel_height);
            for y in 0..panel_height {
                let start = (y + oy) * sky_width + ox;
                data.extend_from_slice(&sky[start..start + panel_width]);
            }
            // Each exposure has its own read and sky noise.
            for value in &mut data {
                let normal = (0..4).map(|_| random()).sum::<f64>() - 2.0;
                *value += (normal * 0.0005) as f32;
            }
            write_f32_image(
                &path,
                panel_width,
                panel_height,
                F32ImageData::Mono(&data),
                &[],
            )
            .unwrap();
            frames.push((path, (ox, oy)));
        }
    }
    frames
}

#[test]
fn expansion_connects_a_crowded_mosaic_through_narrow_overlaps() {
    let directory = tempfile::tempdir().unwrap();
    let frames = dense_mosaic(directory.path(), 512, 384);
    let paths = frames
        .iter()
        .map(|(path, _)| path.clone())
        .collect::<Vec<_>>();
    let mut options = StackOptions {
        normalization: NormalizationMode::LocalBackground { tile_size: 256 },
        ..StackOptions::default()
    };
    options.acceptance.minimum_overlap_fraction = 0.1;
    let mut stacker = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        10_000_000,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    let transforms = paths
        .iter()
        .map(|path| match stacker.push_fits(path).unwrap() {
            FrameDisposition::Accepted(diagnostics) => diagnostics.transform,
            other => panic!("{}: {other:?}", path.display()),
        })
        .collect::<Vec<_>>();
    let snapshot = stacker.snapshot().unwrap();
    assert_eq!(snapshot.accepted_frames as usize, paths.len());
    // Every corner of every frame lands within half a pixel of where the sky
    // puts it, relative to the anchor, even three narrow overlaps from it.
    // Measured: 0.11px at worst.
    let anchor = (transforms[0].translation_x, transforms[0].translation_y);
    for ((path, (ox, oy)), transform) in frames.iter().zip(&transforms) {
        for (x, y) in [(0.0, 0.0), (511.0, 0.0), (0.0, 383.0), (511.0, 383.0)] {
            let (cx, cy) = transform.apply(x, y);
            let error = (cx - anchor.0 - (x + *ox as f64)).hypot(cy - anchor.1 - (y + *oy as f64));
            assert!(
                error < 0.5,
                "{}: corner ({x},{y}) off by {error:.2}px",
                path.display()
            );
        }
    }

    // A resumed stack has no plan for a new frame, which must find its own
    // panel: here a copy of the far corner, which shares no sky with the anchor.
    let context = directory.path().join("mosaic.context");
    stacker.save_context(&context).unwrap();
    let mut resumed = LiveStacker::open_context(&context).unwrap();
    let late = directory.path().join("late.fits");
    std::fs::copy(&paths[2], &late).unwrap();
    let disposition = resumed.push_fits(&late).unwrap();
    assert!(
        matches!(disposition, FrameDisposition::Accepted(_)),
        "{disposition:?}"
    );
    // The union of every frame's footprint, give or take the edge pixel.
    assert!(
        (1415..=1418).contains(&snapshot.image.width),
        "{}",
        snapshot.image.width
    );
    assert!(
        (1060..=1063).contains(&snapshot.image.height),
        "{}",
        snapshot.image.height
    );
}

#[test]
#[ignore = "a timing run on full-size mosaic frames"]
fn expansion_planning_timing_on_full_size_frames() {
    let directory = tempfile::tempdir().unwrap();
    let frames = dense_mosaic(directory.path(), 3008, 3008);
    let paths = frames
        .iter()
        .map(|(path, _)| path.clone())
        .collect::<Vec<_>>();
    let options = StackOptions {
        normalization: NormalizationMode::None,
        ..StackOptions::default()
    };
    // Baseline: an ordinary stack of the anchor panel's frames, each
    // registered as it is pushed.
    let started = std::time::Instant::now();
    let mut ordinary =
        LiveStacker::open_fits(&paths[0], None, None, None, None, options.clone()).unwrap();
    ordinary.push_fits(&paths[1]).unwrap();
    let ordinary = started.elapsed() / 2;
    let mut options = options;
    options.acceptance.minimum_overlap_fraction = 0.1;
    let started = std::time::Instant::now();
    let mut stacker = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        usize::MAX,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    let planning = started.elapsed() / paths.len() as u32;
    let started = std::time::Instant::now();
    for path in &paths {
        stacker.push_fits(path).unwrap();
    }
    let integration = started.elapsed() / paths.len() as u32;
    eprintln!(
        "per frame: ordinary stack {ordinary:?}; expansion planning {planning:?},          integration {integration:?} on a {}x{} canvas",
        stacker.reference_mapping().reference_width(),
        stacker.reference_mapping().reference_height()
    );
}

#[test]
fn expansion_refuses_a_chance_alignment_in_a_crowded_field() {
    let directory = tempfile::tempdir().unwrap();
    let frames = dense_mosaic(directory.path(), 512, 384);
    // Opposite corners share no sky, yet a dozen of their stars line up by
    // chance under some translation. Matching refuses it, not the minimum
    // overlap, so the low one suggested for panels admits no coincidence.
    let paths = [frames[0].0.clone(), frames[2].0.clone()];
    let mut options = StackOptions {
        normalization: NormalizationMode::None,
        ..StackOptions::default()
    };
    options.acceptance.minimum_overlap_fraction =
        FrameAcceptanceCriteria::EXPANDED_MINIMUM_OVERLAP_FRACTION;
    let error = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        10_000_000,
        |_, _, _, _| Continue::Yes,
    )
    .err()
    .unwrap()
    .to_string();
    assert!(error.contains("could not join"), "{error}");
    assert!(!error.contains("panel overlap"), "{error}");
}

#[test]
fn expansion_connects_out_of_order_panels_without_integrating_template() {
    let directory = tempfile::tempdir().unwrap();
    let paths = panels(directory.path());
    // Anchor at the right; the left panel can join only after the middle panel.
    let order = vec![paths[2].clone(), paths[0].clone(), paths[1].clone()];
    let mut options = StackOptions {
        normalization: NormalizationMode::None,
        ..StackOptions::default()
    };
    options.acceptance.minimum_overlap_fraction = 0.1;
    let mut stacker = LiveStacker::with_expanding_reference(
        &order,
        CalibrationMasters::default(),
        options,
        1_000_000,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    for path in &order {
        assert!(matches!(
            stacker.push_fits(path).unwrap(),
            FrameDisposition::Accepted(_)
        ));
    }
    let context = directory.path().join("expanded.context");
    stacker.save_context(&context).unwrap();
    let restored = LiveStacker::open_context(&context).unwrap();
    let replayed = restored
        .reintegrate(&seiza_stacking::BatchStackOptions::default(), |_, _, _| {})
        .unwrap();
    let snapshot = stacker.snapshot().unwrap();
    assert_eq!(replayed.snapshot.image.width, snapshot.image.width);
    assert_eq!(replayed.snapshot.image.height, snapshot.image.height);
    for (online, replay) in snapshot
        .image
        .data
        .iter()
        .zip(&replayed.snapshot.image.data)
    {
        if online.is_finite() {
            assert!((online - replay).abs() < 0.01);
        }
    }
    assert_eq!(snapshot.accepted_frames, 3);
    assert!((416..=420).contains(&snapshot.image.width));
    assert!((160..=164).contains(&snapshot.image.height));
    let crpix = stacker
        .reference_headers()
        .iter()
        .find(|(k, _)| k == "CRPIX1")
        .unwrap()
        .1
        .as_f64()
        .unwrap();
    assert!((208.5..=210.5).contains(&crpix));
    for x in [20, 200, 390] {
        assert!((snapshot.image.data[80 * snapshot.image.width + x] - 0.0114).abs() < 0.002);
    }
}

#[test]
fn cli_expand_preserves_all_panels_online_and_reintegrated() {
    let directory = tempfile::tempdir().unwrap();
    let paths = panels(directory.path());
    for reintegrate in [false, true] {
        for normalization in ["none", "local-background"] {
            let output = directory
                .path()
                .join(format!("stack-{reintegrate}-{normalization}.fits"));
            let report = output.with_extension("json");
            let mut command = Command::new(env!("CARGO_BIN_EXE_seiza"));
            command
                .arg("stack")
                .args(&paths)
                .args([
                    "--reference",
                    "expand",
                    "--normalization",
                    normalization,
                    "--weighting",
                    "equal",
                    "--output",
                ])
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
            let frame = FitsFrame::open(&output).unwrap();
            assert!((416..=420).contains(&frame.image.width));
            assert!((160..=164).contains(&frame.image.height));
            let origin = |key: &str, original: f64| {
                (frame
                    .headers
                    .iter()
                    .find(|(k, _)| k == key)
                    .unwrap()
                    .1
                    .as_f64()
                    .unwrap()
                    - original)
                    .round() as usize
            };
            let ox = origin("CRPIX1", 208.5);
            let oy = origin("CRPIX2", 80.5);
            let report: serde_json::Value =
                serde_json::from_slice(&std::fs::read(report).unwrap()).unwrap();
            assert_eq!(report["accepted_frames"], 3);
            assert_eq!(report["reference_integrated"], false);
            assert_eq!(report["configuration"]["expanded_canvas"], true);
            assert!(
                report["configuration"]["expansion_planning_notes"][0]
                    .as_str()
                    .unwrap()
                    .contains("could not be verified")
            );
            assert!(String::from_utf8_lossy(&result.stderr).contains("could not be verified"));
            assert_eq!(
                report["configuration"]["maximum_canvas_samples"],
                256_000_000
            );
            assert_eq!(report["frames"].as_array().unwrap().len(), 3);
            for panel in &paths {
                let original = FitsFrame::open(panel).unwrap();
                let offset = paths.iter().position(|p| p == panel).unwrap() * STEP;
                // Check every interior source pixel, including stars and single-panel coverage.
                for y in 4..HEIGHT - 4 {
                    for x in 4..WIDTH - 4 {
                        let actual =
                            frame.image.data[(y + oy) * frame.image.width + x + offset + ox];
                        let expected = original.image.data[y * WIDTH + x];
                        assert!(
                            (actual - expected).abs() < 0.03,
                            "{normalization}, replay={reintegrate}: ({x},{y}) {actual} != {expected}"
                        );
                    }
                }
            }
        }
    }
}

/// An expanded canvas normalizes globally unless asked otherwise, and
/// another stack keeps local-background, as the run report records.
#[test]
fn cli_expand_defaults_to_global_normalization() {
    let directory = tempfile::tempdir().unwrap();
    let paths = panels(directory.path());
    for (arguments, lights, expected) in [
        (vec!["--reference", "expand"], &paths[..], "global"),
        (
            vec![
                "--reference",
                "expand",
                "--normalization",
                "local-background",
            ],
            &paths[..],
            "local-background",
        ),
        (
            vec!["--reference", "first", "--min-overlap", "0.1"],
            &paths[..2],
            "local-background",
        ),
    ] {
        let output = directory.path().join("normalized.fits");
        let report = output.with_extension("json");
        let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
            .arg("stack")
            .args(lights)
            .args(&arguments)
            .arg("--output")
            .arg(&output)
            .arg("--report")
            .arg(&report)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{arguments:?}
{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&report).unwrap()).unwrap();
        assert_eq!(
            report["configuration"]["normalization"], expected,
            "{arguments:?}"
        );
    }
}

#[test]
fn expansion_refuses_disconnected_panels_and_canvas_overflow() {
    let directory = tempfile::tempdir().unwrap();
    let paths = panels(directory.path());
    let mut options = StackOptions {
        normalization: NormalizationMode::None,
        ..StackOptions::default()
    };
    options.acceptance.minimum_overlap_fraction = 0.1;
    let disconnected = LiveStacker::with_expanding_reference(
        &[paths[0].clone(), paths[2].clone()],
        CalibrationMasters::default(),
        options.clone(),
        1_000_000,
        |_, _, _, _| Continue::Yes,
    );
    assert!(
        disconnected
            .err()
            .unwrap()
            .to_string()
            .contains("could not join")
    );
    let limited = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        WIDTH * HEIGHT,
        |_, _, _, _| Continue::Yes,
    );
    assert!(
        limited
            .err()
            .unwrap()
            .to_string()
            .contains("exceeds the limit")
    );
}

#[test]
fn expansion_refuses_canvas_overflow_after_a_panel_has_joined() {
    let directory = tempfile::tempdir().unwrap();
    let paths = panels(directory.path());
    let mut options = StackOptions {
        normalization: NormalizationMode::None,
        ..StackOptions::default()
    };
    options.acceptance.minimum_overlap_fraction = 0.1;
    // Two panels fit, but adding the third must fail the whole plan.
    let limit = (WIDTH + STEP + 8) * (HEIGHT + 8);
    assert!(
        LiveStacker::with_expanding_reference(
            &paths[..2],
            CalibrationMasters::default(),
            options.clone(),
            limit,
            |_, _, _, _| Continue::Yes,
        )
        .is_ok()
    );
    let result = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        limit,
        |_, _, _, _| Continue::Yes,
    );
    assert!(
        result
            .err()
            .expect("canvas overflow must not reject just one frame")
            .to_string()
            .contains("exceeds the limit")
    );
}

#[test]
fn expansion_rejects_sources_changed_during_planning() {
    use std::io::Write;
    // Exercise cached detections on retry and an already accepted anchor
    // changed by the final progress callback.
    for change_on_retry in [true, false] {
        let directory = tempfile::tempdir().unwrap();
        let paths = panels(directory.path());
        let order = [paths[0].clone(), paths[2].clone(), paths[1].clone()];
        let mut options = StackOptions {
            normalization: NormalizationMode::None,
            ..StackOptions::default()
        };
        options.acceptance.minimum_overlap_fraction = 0.1;
        let mut visits = 0;
        let mut changed = false;
        let result = LiveStacker::with_expanding_reference(
            &order,
            CalibrationMasters::default(),
            options,
            1_000_000,
            |path, _, _, _| {
                if path == paths[2] {
                    visits += 1;
                }
                if !changed
                    && ((change_on_retry && path == paths[2] && visits == 2)
                        || (!change_on_retry && path == paths[0]))
                {
                    std::fs::OpenOptions::new()
                        .append(true)
                        .open(path)
                        .unwrap()
                        .write_all(&[0; 2880])
                        .unwrap();
                    changed = true;
                }
                Continue::Yes
            },
        );
        assert!(changed);
        assert!(
            result
                .err()
                .expect("changed pixels must not reuse a cached plan")
                .to_string()
                .contains("changed during expansion planning")
        );
    }
}

#[test]
fn cli_expand_rejects_incompatible_options_before_writing_output() {
    let directory = tempfile::tempdir().unwrap();
    let paths = panels(directory.path());
    let output = directory.path().join("invalid.fits");
    for arguments in [
        vec![
            "--reference",
            "expand",
            "--reference-image",
            paths[0].to_str().unwrap(),
        ],
        vec!["--reference", "expand", "--max-canvas-megasamples", "0"],
        vec!["--reference", "expand", "--drizzle", "2"],
        vec!["--reference", "expand", "--bayer-drizzle"],
        vec!["--max-canvas-megasamples", "100"],
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
            .arg("stack")
            .args(&paths)
            .arg("--output")
            .arg(&output)
            .args(&arguments)
            .output()
            .unwrap();
        assert!(!result.status.success(), "{arguments:?}");
        assert!(!output.exists());
    }
}

#[test]
fn expansion_rejects_sources_changed_after_planning_and_honors_cancellation() {
    let directory = tempfile::tempdir().unwrap();
    let paths = panels(directory.path());
    let mut options = StackOptions {
        normalization: NormalizationMode::None,
        ..StackOptions::default()
    };
    options.acceptance.minimum_overlap_fraction = 0.1;
    let cancelled = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options.clone(),
        1_000_000,
        |_, _, _, _| Continue::No,
    );
    assert!(matches!(
        cancelled.err(),
        Some(seiza_stacking::Error::Cancelled)
    ));
    let mut stacker = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        1_000_000,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    // Changing the length reliably changes the source stamp without a clock delay.
    use std::io::Write;
    std::fs::OpenOptions::new()
        .append(true)
        .open(&paths[0])
        .unwrap()
        .write_all(&[0; 2880])
        .unwrap();
    assert!(
        stacker
            .push_fits(&paths[0])
            .err()
            .unwrap()
            .to_string()
            .contains("changed after expansion planning")
    );
}

#[test]
fn expansion_matches_panel_brightness_through_chained_overlaps() {
    let directory = tempfile::tempdir().unwrap();
    let paths = panels(directory.path());
    let originals = paths
        .iter()
        .map(|p| FitsFrame::open(p).unwrap().image)
        .collect::<Vec<_>>();
    for (index, gain, offset) in [(1, 1.2, 0.03), (2, 0.7, 0.02)] {
        let data = originals[index]
            .data
            .iter()
            .map(|v| v * gain + offset)
            .collect::<Vec<_>>();
        write_f32_image(&paths[index], WIDTH, HEIGHT, F32ImageData::Mono(&data), &[]).unwrap();
    }
    let mut options = StackOptions {
        normalization: NormalizationMode::LocalBackground { tile_size: 256 },
        ..StackOptions::default()
    };
    options.acceptance.minimum_overlap_fraction = 0.1;
    let mut stacker = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        1_000_000,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    for path in &paths {
        assert!(matches!(
            stacker.push_fits(path).unwrap(),
            FrameDisposition::Accepted(_)
        ));
    }
    let origin = |key: &str, original: f64| {
        (stacker
            .reference_headers()
            .iter()
            .find(|(k, _)| k == key)
            .unwrap()
            .1
            .as_f64()
            .unwrap()
            - original)
            .round() as usize
    };
    let (ox, oy) = (origin("CRPIX1", 208.5), origin("CRPIX2", 80.5));
    let snapshot = stacker.snapshot().unwrap();
    for (index, original) in originals.iter().enumerate() {
        for y in 5..HEIGHT - 5 {
            for x in 5..WIDTH - 5 {
                let actual =
                    snapshot.image.data[(y + oy) * snapshot.image.width + x + index * STEP + ox];
                let expected = original.data[y * WIDTH + x];
                assert!(
                    (actual - expected).abs() < 1.0e-3,
                    "panel {index}, ({x},{y}): {actual} != {expected}"
                );
            }
        }
    }
}

/// A gradient within one panel cannot be told apart from sky, so it carries
/// into the panels matched through it, and steps appear where coverage
/// changes because the canvas does not feather. Bound both, so a change to
/// normalization that makes them worse is noticed.
#[test]
fn expansion_bounds_the_seams_a_panel_gradient_leaves() {
    let directory = tempfile::tempdir().unwrap();
    let paths = panels(directory.path());
    let originals = paths
        .iter()
        .map(|p| FitsFrame::open(p).unwrap().image)
        .collect::<Vec<_>>();
    // About a third of the sky level across the middle panel.
    let data = originals[1]
        .data
        .iter()
        .enumerate()
        .map(|(i, v)| v * 1.2 + 0.03 + 0.004 * (i % WIDTH) as f32 / WIDTH as f32)
        .collect::<Vec<_>>();
    write_f32_image(&paths[1], WIDTH, HEIGHT, F32ImageData::Mono(&data), &[]).unwrap();
    let mut options = StackOptions {
        normalization: NormalizationMode::LocalBackground { tile_size: 64 },
        ..StackOptions::default()
    };
    options.acceptance.minimum_overlap_fraction = 0.1;
    let mut stacker = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        1_000_000,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    for path in &paths {
        stacker.push_fits(path).unwrap();
    }
    let origin = |key: &str, original: f64| {
        (stacker
            .reference_headers()
            .iter()
            .find(|(k, _)| k == key)
            .unwrap()
            .1
            .as_f64()
            .unwrap()
            - original)
            .round() as usize
    };
    let (ox, oy) = (origin("CRPIX1", 208.5), origin("CRPIX2", 80.5));
    let snapshot = stacker.snapshot().unwrap();
    let width = snapshot.image.width;
    // The median departure from the true sky of every fourth column.
    let residuals = (4..WIDTH + 2 * STEP - 4)
        .step_by(4)
        .map(|x| {
            let panel = (0..3)
                .find(|p| x >= p * STEP && x < p * STEP + WIDTH)
                .unwrap();
            let mut column = (10..HEIGHT - 10)
                .map(|y| {
                    snapshot.image.data[(y + oy) * width + x + ox]
                        - originals[panel].data[y * WIDTH + x - panel * STEP]
                })
                .collect::<Vec<_>>();
            column.sort_by(f32::total_cmp);
            column[column.len() / 2]
        })
        .collect::<Vec<_>>();
    let worst_step = residuals
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .fold(0.0_f32, f32::max);
    let worst_offset = residuals
        .iter()
        .fold(0.0_f32, |worst, r| worst.max(r.abs()));
    // Measured: steps of 0.00053 and an offset of 0.0020. The live pass
    // matches a frame to the canvas with one offset (stack.rs, prepare_frame),
    // so a panel's gradient against its neighbour's is no longer absorbed
    // tile by tile: steps were 0.00034 while it was. Tile by tile, every
    // frame took on the sky of the frames that filled the canvas first, which
    // on a 362-frame mosaic left lumps three times as large as these steps.
    assert!(worst_step < 7.0e-4, "step {worst_step}");
    assert!(worst_offset < 2.5e-3, "offset {worst_offset}");
    // The anchor panel, away from the gradient's panel, is untouched.
    assert!(residuals[..25].iter().all(|r| r.abs() < 1.0e-5));
}

/// Two panels side by side, each with one frame from either side of the pier,
/// rendered from one star list through a decentred lens whose distortion
/// moves the corners about two pixels. The distortion is fixed to the sensor,
/// so a meridian flip turns it against the sky. Panel B's flipped frame comes
/// first. Returns paths, each with whether the frame is flipped.
fn flipped_panels(directory: &Path) -> Vec<(PathBuf, bool)> {
    const PANEL_WIDTH: usize = 512;
    const PANEL_HEIGHT: usize = 384;
    const STEP_X: usize = 430;
    let sky_width = PANEL_WIDTH + STEP_X;
    let mut seed = 0xf11e_u64;
    let mut random = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f64 / (1u64 << 31) as f64
    };
    let stars = (0..sky_width * PANEL_HEIGHT / 250)
        .map(|_| {
            (
                random() * sky_width as f64,
                random() * PANEL_HEIGHT as f64,
                (0.02 + 0.0005 / (0.01 + random()).powf(1.5)) as f32,
            )
        })
        .collect::<Vec<_>>();
    let (cx, cy) = (PANEL_WIDTH as f64 / 2.0, PANEL_HEIGHT as f64 / 2.0);
    let mut frames = Vec::new();
    for (panel, flipped) in [(0, false), (1, true), (0, true), (1, false)] {
        let mut data = vec![0.01_f32; PANEL_WIDTH * PANEL_HEIGHT];
        for &(sx, sy, peak) in &stars {
            let (mut x, mut y) = (sx - (panel * STEP_X) as f64, sy);
            if flipped {
                (x, y) = ((PANEL_WIDTH - 1) as f64 - x, (PANEL_HEIGHT - 1) as f64 - y);
            }
            let (u, v) = ((x - cx) / cx, (y - cy) / cy);
            let (x, y) = (x + 2.0 * u * u, y + 2.0 * u * v);
            for py in (y.floor() as isize - 4)..=(y.floor() as isize + 4) {
                for px in (x.floor() as isize - 4)..=(x.floor() as isize + 4) {
                    if px < 0 || py < 0 || px >= PANEL_WIDTH as isize || py >= PANEL_HEIGHT as isize
                    {
                        continue;
                    }
                    let d2 = (px as f64 - x).powi(2) + (py as f64 - y).powi(2);
                    data[py as usize * PANEL_WIDTH + px as usize] +=
                        peak * (-d2 / 2.9).exp() as f32;
                }
            }
        }
        for value in &mut data {
            *value += (((0..4).map(|_| random()).sum::<f64>() - 2.0) * 0.0005) as f32;
        }
        let path = directory.join(format!("panel-{panel}-{flipped}.fits"));
        write_f32_image(
            &path,
            PANEL_WIDTH,
            PANEL_HEIGHT,
            F32ImageData::Mono(&data),
            &[],
        )
        .unwrap();
        frames.push((path, flipped));
    }
    frames
}

#[test]
fn expansion_warps_frames_from_the_other_side_of_the_pier_onto_their_panel() {
    let directory = tempfile::tempdir().unwrap();
    let frames = flipped_panels(directory.path());
    let paths = frames
        .iter()
        .map(|(path, _)| path.clone())
        .collect::<Vec<_>>();
    let stack = |model| {
        let mut options = StackOptions {
            normalization: NormalizationMode::LocalBackground { tile_size: 128 },
            ..StackOptions::default()
        };
        options.registration.model = model;
        options.acceptance.minimum_overlap_fraction = 0.1;
        let mut stacker = LiveStacker::with_expanding_reference(
            &paths,
            CalibrationMasters::default(),
            options,
            10_000_000,
            |_, _, _, _| Continue::Yes,
        )
        .unwrap();
        paths
            .iter()
            .map(|path| match stacker.push_fits(path).unwrap() {
                FrameDisposition::Accepted(diagnostics) => diagnostics,
                other => panic!("{}: {other:?}", path.display()),
            })
            .collect::<Vec<_>>()
    };
    // A similarity cannot follow the distortion across the flip.
    let similarity = stack(seiza_stacking::RegistrationModel::Similarity)
        .iter()
        .map(|diagnostics| diagnostics.registration_rms_pixels)
        .fold(0.0, f64::max);
    assert!(similarity > 0.5, "{similarity}");
    let quadratic = stack(seiza_stacking::RegistrationModel::Quadratic);
    for ((path, flipped), diagnostics) in frames.iter().zip(&quadratic) {
        let warped = diagnostics.mapping.warp().is_some();
        let flipped_on_canvas = diagnostics.transform.rotation_radians.cos() < 0.0;
        assert_eq!(flipped_on_canvas, *flipped, "{}", path.display());
        // Panel B starts from its unflipped frame, though the flipped one
        // came first, and keeps its similarity; every flipped frame is warped
        // onto its panel.
        assert_eq!(warped, *flipped, "{}", path.display());
        if warped {
            assert!(
                diagnostics.registration_rms_pixels < 0.3,
                "{}: {}",
                path.display(),
                diagnostics.registration_rms_pixels
            );
        }
    }
}

/// A radial lens about an axis off the sensor's center: a star an ideal lens
/// images at `ideal` lands at `axis + (ideal - axis) (1 + k1 q + k2 q²)`, `q`
/// being its squared distance from the axis over the half diagonal squared.
struct Lens {
    axis: (f64, f64),
    radius: f64,
    k1: f64,
    k2: f64,
}

impl Lens {
    fn distort(&self, (x, y): (f64, f64)) -> (f64, f64) {
        let (u, v) = (x - self.axis.0, y - self.axis.1);
        let q = (u * u + v * v) / (self.radius * self.radius);
        let factor = 1.0 + self.k1 * q + self.k2 * q * q;
        (self.axis.0 + u * factor, self.axis.1 + v * factor)
    }

    fn undistort(&self, (x, y): (f64, f64)) -> (f64, f64) {
        let mut ideal = (x, y);
        for _ in 0..50 {
            let (dx, dy) = self.distort(ideal);
            ideal = (ideal.0 + x - dx, ideal.1 + y - dy);
        }
        ideal
    }
}

/// Three panels stacked down the sky, a quarter of each overlapping the
/// next, the middle one shot from the other side of the pier, two dithered
/// frames apiece, all through one decentred lens that moves stars up to ten
/// pixels. Returns each frame's path and the map from its sensor to the sky.
#[allow(clippy::type_complexity)]
fn lens_mosaic(directory: &Path) -> (Vec<(PathBuf, Box<dyn Fn((f64, f64)) -> (f64, f64)>)>, Lens) {
    const PANEL_WIDTH: usize = 720;
    const PANEL_HEIGHT: usize = 480;
    const STEP_Y: f64 = 360.0;
    let lens = Lens {
        axis: (359.5 + 25.0, 239.5 - 18.0),
        radius: 0.5 * (PANEL_WIDTH as f64).hypot(PANEL_HEIGHT as f64),
        k1: 0.02,
        k2: 0.004,
    };
    let mut seed = 0x1e45_u64;
    let mut random = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f64 / (1u64 << 31) as f64
    };
    let (sky_width, sky_height) = (
        PANEL_WIDTH as f64 + 40.0,
        2.0 * STEP_Y + PANEL_HEIGHT as f64 + 40.0,
    );
    let stars = (0..(sky_width * sky_height / 120.0) as usize)
        .map(|_| {
            (
                random() * sky_width - 20.0,
                random() * sky_height - 20.0,
                (0.02 + 0.0005 / (0.01 + random()).powf(1.5)) as f32,
            )
        })
        .collect::<Vec<_>>();
    let (right, bottom) = ((PANEL_WIDTH - 1) as f64, (PANEL_HEIGHT - 1) as f64);
    let mut frames: Vec<(PathBuf, Box<dyn Fn((f64, f64)) -> (f64, f64)>)> = Vec::new();
    for (panel, dither) in [(0, 0.0), (0, 3.3), (1, 0.0), (1, -2.7), (2, 0.0), (2, 4.1)] {
        let flipped = panel == 1;
        let (ox, oy) = (dither, panel as f64 * STEP_Y + dither * 0.6);
        // Sky to the frame's ideal grid, and back.
        let to_frame = move |(x, y): (f64, f64)| {
            let (x, y) = (x - ox, y - oy);
            if flipped {
                (right - x, bottom - y)
            } else {
                (x, y)
            }
        };
        let to_sky = move |(x, y): (f64, f64)| {
            let (x, y) = if flipped {
                (right - x, bottom - y)
            } else {
                (x, y)
            };
            (x + ox, y + oy)
        };
        let mut data = vec![0.01_f32; PANEL_WIDTH * PANEL_HEIGHT];
        for &(sx, sy, peak) in &stars {
            let (x, y) = lens.distort(to_frame((sx, sy)));
            for py in (y.floor() as isize - 4)..=(y.floor() as isize + 4) {
                for px in (x.floor() as isize - 4)..=(x.floor() as isize + 4) {
                    if px < 0 || py < 0 || px >= PANEL_WIDTH as isize || py >= PANEL_HEIGHT as isize
                    {
                        continue;
                    }
                    let d2 = (px as f64 - x).powi(2) + (py as f64 - y).powi(2);
                    data[py as usize * PANEL_WIDTH + px as usize] +=
                        peak * (-d2 / 2.9).exp() as f32;
                }
            }
        }
        for value in &mut data {
            *value += (((0..4).map(|_| random()).sum::<f64>() - 2.0) * 0.0005) as f32;
        }
        let path = directory.join(format!("lens-{panel}-{dither}.fits"));
        write_f32_image(
            &path,
            PANEL_WIDTH,
            PANEL_HEIGHT,
            F32ImageData::Mono(&data),
            &[
                WriteHeaderCard::new("CRPIX1", HeaderValue::Float(360.5)),
                WriteHeaderCard::new("CRPIX2", HeaderValue::Float(240.5)),
                WriteHeaderCard::new("CRVAL1", HeaderValue::Float(120.0)),
                WriteHeaderCard::new("CRVAL2", HeaderValue::Float(30.0)),
                WriteHeaderCard::new("CTYPE1", HeaderValue::String("RA---TAN-SIP".into())),
                WriteHeaderCard::new("CTYPE2", HeaderValue::String("DEC--TAN-SIP".into())),
                WriteHeaderCard::new("CD1_1", HeaderValue::Float(-0.001)),
                WriteHeaderCard::new("CD2_2", HeaderValue::Float(0.001)),
                WriteHeaderCard::new("A_ORDER", HeaderValue::Integer(2)),
                WriteHeaderCard::new("B_ORDER", HeaderValue::Integer(2)),
                WriteHeaderCard::new("A_2_0", HeaderValue::Float(1.0e-5)),
                WriteHeaderCard::new("EXPTIME", HeaderValue::Float(60.0)),
            ],
        )
        .unwrap();
        let undistort = Lens { ..lens };
        frames.push((path, Box::new(move |raw| to_sky(undistort.undistort(raw)))));
    }
    (frames, lens)
}

/// Plan and stack every frame of `paths` but `late`, save the context, then
/// register a copy of `late` on the live stack and another on the stack
/// reopened from the context. Returns both registrations and whether
/// planning measured a lens.
fn register_late_frame_after_reopening(
    directory: &Path,
    paths: &[PathBuf],
    late: usize,
) -> (
    seiza_stacking::FrameDiagnostics,
    seiza_stacking::FrameDiagnostics,
    bool,
) {
    let planned = paths
        .iter()
        .enumerate()
        .filter(|&(index, _)| index != late)
        .map(|(_, path)| path.clone())
        .collect::<Vec<_>>();
    let mut options = StackOptions {
        normalization: NormalizationMode::LocalBackground { tile_size: 128 },
        ..StackOptions::default()
    };
    options.registration.model = seiza_stacking::RegistrationModel::Quadratic;
    options.acceptance.minimum_overlap_fraction = 0.1;
    let mut stacker = LiveStacker::with_expanding_reference(
        &planned,
        CalibrationMasters::default(),
        options,
        10_000_000,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    for path in &planned {
        assert!(matches!(
            stacker.push_fits(path).unwrap(),
            FrameDisposition::Accepted(_)
        ));
    }
    let context = directory.join("late.context");
    stacker.save_context(&context).unwrap();
    let mut reopened = LiveStacker::open_context(&context).unwrap();
    assert_eq!(
        reopened.expanded_lens().is_some(),
        stacker.expanded_lens().is_some()
    );
    let push = |stacker: &mut LiveStacker, name: &str| {
        // Copies, since a stack refuses a file it has already integrated.
        let copy = directory.join(name);
        std::fs::copy(&paths[late], &copy).unwrap();
        match stacker.push_fits(&copy).unwrap() {
            FrameDisposition::Accepted(diagnostics) => diagnostics,
            other => panic!("{}: {other:?}", copy.display()),
        }
    };
    let live = push(&mut stacker, "late-live.fits");
    let restored = push(&mut reopened, "late-reopened.fits");
    (live, restored, stacker.expanded_lens().is_some())
}

/// Where a registration reads a canvas pixel from its source.
fn source_of(diagnostics: &seiza_stacking::FrameDiagnostics, x: f64, y: f64) -> (f64, f64) {
    match diagnostics.mapping.warp() {
        Some(warp) => warp.apply(x, y),
        None => diagnostics.transform.inverse_apply(x, y),
    }
}

/// The largest distance between where two registrations of one frame read
/// the canvas pixels the frame covers.
fn disagreement(
    a: &seiza_stacking::FrameDiagnostics,
    b: &seiza_stacking::FrameDiagnostics,
    width: f64,
    height: f64,
) -> f64 {
    let mut worst: f64 = 0.0;
    for row in 0..=8 {
        for column in 0..=8 {
            let (x, y) = a
                .transform
                .apply(width * column as f64 / 8.0, height * row as f64 / 8.0);
            let (p, q) = (source_of(a, x, y), source_of(b, x, y));
            worst = worst.max((p.0 - q.0).hypot(p.1 - q.1));
        }
    }
    worst
}

#[test]
fn a_frame_added_after_reopening_registers_as_before_without_a_lens() {
    let directory = tempfile::tempdir().unwrap();
    // Panel B's flipped frame is left out of planning and added late, to
    // the panel its unflipped frame started, whose footprint on the canvas
    // also holds panel A's edge.
    let frames = flipped_panels(directory.path());
    let paths = frames
        .iter()
        .map(|(path, _)| path.clone())
        .collect::<Vec<_>>();
    assert!(frames[1].1, "the late frame is flipped");
    let (live, reopened, lens) = register_late_frame_after_reopening(directory.path(), &paths, 1);
    assert!(!lens, "this mosaic's distortion is not radial");
    assert!(live.mapping.warp().is_some(), "the flipped frame is warped");
    assert!(live.registration_rms_pixels < 0.3, "{live:?}");
    let worst = disagreement(&live, &reopened, 511.0, 383.0);
    assert!(worst < 1.0e-6, "reopened stack reads it {worst:.4}px away");
}

#[test]
fn a_frame_added_after_reopening_registers_as_before_through_the_lens() {
    let directory = tempfile::tempdir().unwrap();
    let (frames, _) = lens_mosaic(directory.path());
    let paths = frames
        .iter()
        .map(|(path, _)| path.clone())
        .collect::<Vec<_>>();
    // The middle panel was shot from the other side of the pier; its second
    // frame comes late.
    let (live, reopened, lens) = register_late_frame_after_reopening(directory.path(), &paths, 3);
    assert!(lens);
    assert!(
        live.transform.rotation_radians.cos() < 0.0,
        "the frame is flipped"
    );
    assert!(live.registration_rms_pixels < 0.15, "{live:?}");
    let worst = disagreement(&live, &reopened, 719.0, 479.0);
    assert!(worst < 1.0e-6, "reopened stack reads it {worst:.4}px away");
}

#[test]
fn expansion_does_not_hide_a_source_change_in_the_lens_pass() {
    use std::io::Write;
    let directory = tempfile::tempdir().unwrap();
    let (frames, _) = lens_mosaic(directory.path());
    let paths = frames
        .iter()
        .map(|(path, _)| path.clone())
        .collect::<Vec<_>>();
    let mut options = StackOptions {
        normalization: NormalizationMode::LocalBackground { tile_size: 128 },
        ..StackOptions::default()
    };
    options.registration.model = seiza_stacking::RegistrationModel::Quadratic;
    options.acceptance.minimum_overlap_fraction = 0.1;
    let mut changed = false;
    let result = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        10_000_000,
        |path, planned, _, _| {
            if path == paths[0] && planned == 0 {
                std::fs::OpenOptions::new()
                    .append(true)
                    .open(path)
                    .unwrap()
                    .write_all(&[0; 2880])
                    .unwrap();
                changed = true;
            }
            Continue::Yes
        },
    );
    assert!(changed, "the fixture must reach the lens rebuild");
    assert!(
        result
            .err()
            .expect("a changed source must not fall back to the first plan")
            .to_string()
            .contains("changed during expansion planning")
    );
}

#[test]
fn a_conflicting_joined_frames_camera_disables_the_shared_lens() {
    let directory = tempfile::tempdir().unwrap();
    let (frames, _) = lens_mosaic(directory.path());
    let paths = frames
        .iter()
        .map(|(path, _)| path.clone())
        .collect::<Vec<_>>();
    for (index, path) in paths.iter().enumerate() {
        let frame = FitsFrame::open(path).unwrap();
        // This is a dithered sibling, not a panel start. Every joined frame
        // must participate in the compatibility check.
        let camera = if index == 1 { "Camera B" } else { "Camera A" };
        write_f32_image(
            path,
            frame.image.width,
            frame.image.height,
            F32ImageData::Mono(&frame.image.data),
            &[WriteHeaderCard::new(
                "INSTRUME",
                HeaderValue::String(camera.into()),
            )],
        )
        .unwrap();
    }
    let mut options = StackOptions {
        normalization: NormalizationMode::LocalBackground { tile_size: 128 },
        ..StackOptions::default()
    };
    options.registration.model = seiza_stacking::RegistrationModel::Quadratic;
    options.acceptance.minimum_overlap_fraction = 0.1;
    let stacker = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        10_000_000,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    assert!(stacker.expanded_lens().is_none());
    assert!(
        stacker
            .expansion_planning_notes()
            .iter()
            .any(|note| note.contains("shared lens fit skipped: camera")
                && note.contains("lens-0-3.3.fits"))
    );
}

#[test]
fn expansion_measures_one_lens_across_its_panels() {
    let directory = tempfile::tempdir().unwrap();
    let (frames, lens) = lens_mosaic(directory.path());
    let paths = frames
        .iter()
        .map(|(path, _)| path.clone())
        .collect::<Vec<_>>();
    let mut options = StackOptions {
        normalization: NormalizationMode::LocalBackground { tile_size: 128 },
        ..StackOptions::default()
    };
    options.registration.model = seiza_stacking::RegistrationModel::Quadratic;
    options.acceptance.minimum_overlap_fraction = 0.1;
    let mut stacker = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        10_000_000,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    let measured = stacker
        .expanded_lens()
        .expect("the panels' overlaps measure the lens");
    assert!(
        stacker
            .expansion_planning_notes()
            .iter()
            .any(|note| note.contains("could not be verified")
                && note.contains("lens fitting remains allowed"))
    );
    let decenter = (lens.axis.0 - measured.center_x).hypot(lens.axis.1 - measured.center_y);
    assert!(decenter < 8.0, "{measured:?}");
    assert!(
        measured.overlap_rms_pixels.unwrap() < 0.2
            && measured.undistorted_overlap_rms_pixels.unwrap() > 0.5,
        "{measured:?}"
    );
    let warps = paths
        .iter()
        .map(|path| match stacker.push_fits(path).unwrap() {
            FrameDisposition::Accepted(diagnostics) => diagnostics.mapping.warp().cloned().unwrap(),
            other => panic!("{}: {other:?}", path.display()),
        })
        .collect::<Vec<_>>();
    // Every frame covering a canvas pixel must read it from the same point
    // of the sky, seams included.
    let snapshot = stacker.snapshot().unwrap();
    let mut worst: f64 = 0.0;
    let mut shared = 0;
    for y in (0..snapshot.image.height).step_by(9) {
        for x in (0..snapshot.image.width).step_by(9) {
            let skies = warps
                .iter()
                .zip(&frames)
                .filter_map(|(warp, (_, to_sky))| {
                    let (u, v) = warp.apply(x as f64, y as f64);
                    (u >= 4.0 && v >= 4.0 && u <= 715.0 && v <= 475.0).then(|| to_sky((u, v)))
                })
                .collect::<Vec<_>>();
            if skies.len() > 2 {
                shared += 1;
            }
            for a in &skies {
                for b in &skies {
                    worst = worst.max((a.0 - b.0).hypot(a.1 - b.1));
                }
            }
        }
    }
    assert!(shared > 500, "{shared}");
    assert!(worst < 0.25, "frames disagree by {worst:.3}px");

    // A saved context keeps the lens: reintegration reads every frame back
    // through it, and a frame pushed after reopening registers through it.
    let context = directory.path().join("lens.context");
    stacker.save_context(&context).unwrap();
    let mut restored = LiveStacker::open_context(&context).unwrap();
    let kept = restored
        .expanded_lens()
        .expect("the context keeps the lens");
    assert_eq!(
        (kept.center_x, kept.center_y, kept.k1, kept.k2),
        (
            measured.center_x,
            measured.center_y,
            measured.k1,
            measured.k2
        )
    );
    let replayed = restored
        .reintegrate(&seiza_stacking::BatchStackOptions::default(), |_, _, _| {})
        .unwrap();
    // Raw TAN-SIP headers do not describe the undistorted canvas. Neither
    // online nor reopened/reintegrated output may publish that stale WCS.
    for (name, snapshot, headers) in [
        ("online", &snapshot, stacker.reference_headers()),
        ("replayed", &replayed.snapshot, restored.reference_headers()),
    ] {
        for key in ["CRPIX1", "CTYPE1", "CD1_1", "A_ORDER", "A_2_0"] {
            assert!(!headers.iter().any(|(k, _)| k == key), "{name}: {key}");
        }
        assert!(headers.iter().any(|(k, _)| k == "EXPTIME"));
        let output = directory.path().join(format!("{name}.fits"));
        seiza_stacking::write_fits_f32(&output, snapshot, headers).unwrap();
        let output = FitsFrame::open(output).unwrap();
        assert!(!output.headers.iter().any(|(k, _)| k == "CRPIX1"));
    }
    let (mut compared, mut differing) = (0, 0);
    for (online, replay) in snapshot
        .image
        .data
        .iter()
        .zip(&replayed.snapshot.image.data)
    {
        if online.is_finite() && replay.is_finite() {
            compared += 1;
            differing += usize::from((online - replay).abs() > 0.01);
        }
    }
    assert!(differing * 1000 < compared, "{differing} of {compared}");
    // A copy, since the stack refuses a file it has already integrated.
    let again = directory.path().join("again.fits");
    std::fs::copy(&paths[3], &again).unwrap();
    let FrameDisposition::Accepted(diagnostics) = restored.push_fits(&again).unwrap() else {
        panic!("a frame pushed after reopening joins the canvas");
    };
    let warp = diagnostics.mapping.warp().unwrap();
    for (x, y) in [(200.0, 500.0), (600.0, 700.0), (360.0, 900.0)] {
        let (a, b) = (warp.apply(x, y), warps[3].apply(x, y));
        assert!((a.0 - b.0).hypot(a.1 - b.1) < 0.1, "{a:?} vs {b:?}");
    }
}

#[test]
fn expansion_from_paths_keeps_masters_for_reintegration_after_reopening() {
    let directory = tempfile::tempdir().unwrap();
    let paths = panels(directory.path());
    let dark = directory.path().join("dark.fits");
    write_f32_image(
        &dark,
        WIDTH,
        HEIGHT,
        F32ImageData::Mono(&vec![1.0e-3; WIDTH * HEIGHT]),
        &[
            WriteHeaderCard::new("SEIZAMST", HeaderValue::String("DARK".into())),
            WriteHeaderCard::new("EXPTIME", HeaderValue::Float(60.0)),
        ],
    )
    .unwrap();
    let mut options = StackOptions {
        normalization: NormalizationMode::None,
        ..StackOptions::default()
    };
    options.acceptance.minimum_overlap_fraction = 0.1;
    let open = |lights: &[PathBuf]| {
        LiveStacker::open_fits_expanding(
            lights,
            None,
            Some(&dark),
            None,
            None,
            options.clone(),
            1_000_000,
            |_, _, _, _| Continue::Yes,
        )
    };
    // A master cannot also be a light.
    assert!(open(&[paths[0].clone(), dark.clone()]).is_err());
    let mut stacker = open(&paths).unwrap();
    assert!(
        stacker
            .protected_paths()
            .iter()
            .any(|path| path.ends_with("dark.fits"))
    );
    for path in &paths {
        assert!(matches!(
            stacker.push_fits(path).unwrap(),
            FrameDisposition::Accepted(_)
        ));
    }
    let context = directory.path().join("expanded.context");
    stacker.save_context(&context).unwrap();
    let reopened = LiveStacker::open_context(&context).unwrap();
    assert_eq!(reopened.reintegration_unavailable(), None);
    let replayed = reopened
        .reintegrate(&seiza_stacking::BatchStackOptions::default(), |_, _, _| {})
        .unwrap();
    assert_eq!(replayed.snapshot.accepted_frames, 3);
}

/// Real overlapping panels: the lens bends their shared edges apart, so the
/// similarity placing a panel misses the overlap's ends by many pixels. Before
/// photometry kept to stars the placement actually lands on, the panels of a
/// three-panel ASI6200MM Rosette mosaic came out 1.8 to 3.5 times too bright
/// in O-III, and S-II frames were rejected; the stars themselves differ by
/// under 25% between panels.
#[test]
#[ignore = "set SEIZA_STACKING_REAL_MOSAIC to a directory of one filter's overlapping lights"]
fn expansion_scales_real_panels_by_their_stars() {
    let directory =
        std::env::var("SEIZA_STACKING_REAL_MOSAIC").expect("fixture directory required");
    let mut paths = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("fits"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    let mut options = StackOptions {
        normalization: NormalizationMode::LocalBackground { tile_size: 256 },
        ..StackOptions::default()
    };
    options.registration.model = seiza_stacking::RegistrationModel::Quadratic;
    options.acceptance.minimum_overlap_fraction = 0.1;
    let mut stacker = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        usize::MAX,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    for path in &paths {
        match stacker.push_fits(path).unwrap() {
            FrameDisposition::Accepted(diagnostics) => assert!(
                (0.6..1.6).contains(&diagnostics.normalization_mean_gain),
                "{}: gain {}",
                path.display(),
                diagnostics.normalization_mean_gain
            ),
            other => panic!("{}: {other:?}", path.display()),
        }
    }
}

/// Frames of two side-by-side panels from sessions whose camera sat at
/// `rotations` degrees, rendered from one star list, followed by a frame of
/// unrelated sky. Returns the paths in that order.
fn rotated_sessions(directory: &Path, frames: &[(usize, f64)]) -> Vec<PathBuf> {
    const WIDTH: usize = 512;
    const HEIGHT: usize = 384;
    const STEP: f64 = 380.0;
    let render = |seed: u64, name: &str, panel: usize, degrees: f64| {
        let mut seed = seed;
        let mut random = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as f64 / (1u64 << 31) as f64
        };
        let sky = (0..1100)
            .map(|_| {
                (
                    -100.0 + random() * (WIDTH as f64 + STEP + 200.0),
                    -100.0 + random() * (HEIGHT as f64 + 200.0),
                    (0.02 + 0.0005 / (0.01 + random()).powf(1.5)) as f32,
                )
            })
            .collect::<Vec<_>>();
        let (cx, cy) = (WIDTH as f64 / 2.0, HEIGHT as f64 / 2.0);
        let (sin, cos) = degrees.to_radians().sin_cos();
        let mut data = vec![0.01_f32; WIDTH * HEIGHT];
        for &(sx, sy, peak) in &sky {
            // The sky about the panel's centre, turned into the camera's frame.
            let (u, v) = (sx - panel as f64 * STEP - cx, sy - cy);
            let (x, y) = (cx + cos * u + sin * v, cy - sin * u + cos * v);
            for py in (y.floor() as isize - 4)..=(y.floor() as isize + 4) {
                for px in (x.floor() as isize - 4)..=(x.floor() as isize + 4) {
                    if px < 0 || py < 0 || px >= WIDTH as isize || py >= HEIGHT as isize {
                        continue;
                    }
                    let d2 = (px as f64 - x).powi(2) + (py as f64 - y).powi(2);
                    data[py as usize * WIDTH + px as usize] += peak * (-d2 / 2.9).exp() as f32;
                }
            }
        }
        for value in &mut data {
            *value += (((0..4).map(|_| random()).sum::<f64>() - 2.0) * 0.0005) as f32;
        }
        let path = directory.join(name);
        write_f32_image(&path, WIDTH, HEIGHT, F32ImageData::Mono(&data), &[]).unwrap();
        path
    };
    let mut paths = frames
        .iter()
        .enumerate()
        .map(|(index, &(panel, degrees))| {
            render(0x5e55, &format!("session-{index}.fits"), panel, degrees)
        })
        .collect::<Vec<_>>();
    paths.push(render(0x0dd, "elsewhere.fits", 0, 0.0));
    paths
}

#[test]
fn expansion_joins_sessions_at_any_rotation_and_rejects_a_stray_frame() {
    let directory = tempfile::tempdir().unwrap();
    // The anchor's panel at 0 and 37 degrees, its neighbour at 10.8 and 95.
    let sessions = [(0, 0.0), (1, 10.8), (0, 37.0), (1, 95.0)];
    let paths = rotated_sessions(directory.path(), &sessions);
    for (index, path) in paths.iter().enumerate() {
        let frame = FitsFrame::open(path).unwrap();
        let camera = if index == sessions.len() {
            "Unrelated rig"
        } else {
            "Mosaic rig"
        };
        write_f32_image(
            path,
            frame.image.width,
            frame.image.height,
            F32ImageData::Mono(&frame.image.data),
            &[WriteHeaderCard::new(
                "INSTRUME",
                HeaderValue::String(camera.into()),
            )],
        )
        .unwrap();
    }
    let mut options = StackOptions {
        normalization: NormalizationMode::LocalBackground { tile_size: 128 },
        ..StackOptions::default()
    };
    options.registration.model = seiza_stacking::RegistrationModel::Quadratic;
    options.acceptance.minimum_overlap_fraction = 0.1;
    let mut stacker = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        10_000_000,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    for (path, &(_, degrees)) in paths.iter().zip(&sessions) {
        match stacker.push_fits(path).unwrap() {
            FrameDisposition::Accepted(diagnostics) => {
                // Mapping the frame back onto the anchor undoes the camera's turn.
                let rotation = diagnostics.transform.rotation_radians.to_degrees();
                let error = (rotation - degrees + 180.0).rem_euclid(360.0) - 180.0;
                assert!(error.abs() < 0.1, "{}: {rotation}", path.display());
            }
            other => panic!("{}: {other:?}", path.display()),
        }
    }
    match stacker.push_fits(paths.last().unwrap()).unwrap() {
        FrameDisposition::Rejected(reason) => {
            assert!(reason.to_string().contains("could not join"), "{reason}")
        }
        other => panic!("stray frame: {other:?}"),
    }
    assert_eq!(stacker.snapshot().unwrap().accepted_frames, 4);
    assert!(
        !stacker
            .expansion_planning_notes()
            .iter()
            .any(|note| note.contains("shared lens fit skipped"))
    );
}

#[test]
fn expansion_places_a_frame_on_its_own_panel_after_a_neighbours() {
    let directory = tempfile::tempdir().unwrap();
    let paths = panels(directory.path());
    // Another frame of the anchor's panel, taken after its neighbour's: the
    // neighbour is the panel the previous frame joined, and overlaps this
    // frame by 42%, but the frame belongs to the anchor's panel.
    let again = directory.path().join("panel-0-again.fits");
    std::fs::copy(&paths[0], &again).unwrap();
    let order = [paths[0].clone(), paths[1].clone(), again];
    let mut options = StackOptions {
        normalization: NormalizationMode::None,
        ..StackOptions::default()
    };
    options.registration.model = seiza_stacking::RegistrationModel::Quadratic;
    options.acceptance.minimum_overlap_fraction = 0.1;
    let mut stacker = LiveStacker::with_expanding_reference(
        &order,
        CalibrationMasters::default(),
        options,
        1_000_000,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    let diagnostics = order
        .iter()
        .map(|path| match stacker.push_fits(path).unwrap() {
            FrameDisposition::Accepted(diagnostics) => diagnostics,
            other => panic!("{}: {other:?}", path.display()),
        })
        .collect::<Vec<_>>();
    // Placed on its own panel, it is warped onto it, by many more stars than
    // the neighbour's overlap holds.
    assert!(diagnostics[2].mapping.warp().is_some());
    assert!(
        diagnostics[2].matched_stars > diagnostics[1].matched_stars,
        "{} vs {}",
        diagnostics[2].matched_stars,
        diagnostics[1].matched_stars
    );
}

/// Lights of another target, taken uncalibrated on the same camera: their hot
/// pixels sit where the anchor's do, so at no shift or turn at all they pair
/// far above chance. Two frames of M81 among a Rosette mosaic's H-alpha were
/// placed on the anchor that way, and only integration's gain check, near
/// 600, threw them out.
#[test]
#[ignore = "set SEIZA_STACKING_REAL_MOSAIC to overlapping lights, the first sorted the anchor, and SEIZA_STACKING_REAL_STRAYS to lights of another target"]
fn expansion_refuses_real_lights_of_another_target() {
    let read = |variable: &str| {
        let mut paths =
            std::fs::read_dir(std::env::var(variable).expect("fixture directory required"))
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .filter(|path| {
                    path.extension()
                        .is_some_and(|extension| extension.eq_ignore_ascii_case("fits"))
                })
                .collect::<Vec<_>>();
        paths.sort();
        paths
    };
    let mosaic = read("SEIZA_STACKING_REAL_MOSAIC");
    let strays = read("SEIZA_STACKING_REAL_STRAYS");
    let paths = mosaic.iter().chain(&strays).cloned().collect::<Vec<_>>();
    let mut options = StackOptions {
        normalization: NormalizationMode::LocalBackground { tile_size: 256 },
        ..StackOptions::default()
    };
    options.registration.model = seiza_stacking::RegistrationModel::Quadratic;
    options.acceptance.minimum_overlap_fraction = 0.1;
    let mut stacker = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        usize::MAX,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    for path in &mosaic {
        assert!(matches!(
            stacker.push_fits(path).unwrap(),
            FrameDisposition::Accepted(_)
        ));
    }
    for path in &strays {
        match stacker.push_fits(path).unwrap() {
            FrameDisposition::Rejected(reason) => assert!(
                reason.to_string().contains("could not join"),
                "{}: {reason}",
                path.display()
            ),
            other => panic!("{}: {other:?}", path.display()),
        }
    }
}
