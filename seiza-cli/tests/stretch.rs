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
fn stretch_cli_ht_clips_and_rescales() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("linear.fits");
    let output = directory.path().join("ht.png");
    let values = (0..=10).map(|step| step as f32 / 10.0).collect::<Vec<_>>();
    write_f32_image(&input, 11, 1, F32ImageData::Mono(&values), &[]).unwrap();

    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "stretch",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "ht",
            "--shadows",
            "0.2",
            "--midtone",
            "0.25",
            "--highlights",
            "0.8",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let preview = image::open(output).unwrap().to_luma8();
    let level = |x| preview.get_pixel(x, 0).0[0];
    assert_eq!(level(1), 0);
    assert_eq!(level(9), 255);
    // x = 0.5 of the rescaled range: 0.75 * 0.5 / (0.5 * 0.5 + 0.25) = 0.75.
    assert_eq!(level(5), 191);
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

/// A stack as `seiza stack` writes it: an RGB sky of about 3300 ADU with
/// noise, a few bright stars, and NaN where no frame covered the canvas.
fn physical_mosaic(path: &std::path::Path) {
    let (width, height) = (160_usize, 120_usize);
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            for channel in 0..3 {
                let value = if x + y < 40 {
                    f32::NAN
                } else if (x % 37 == 18) && (y % 29 == 14) {
                    50_000.0
                } else {
                    3300.0
                        + channel as f32 * 40.0
                        + ((x * 7919 + y * 104_729 + channel) % 41) as f32
                };
                data.push(value);
            }
        }
    }
    write_f32_image(
        path,
        width,
        height,
        F32ImageData::RgbInterleaved(&data),
        &[],
    )
    .unwrap();
}

/// The data-driven stretches place the sky from its own statistics, so a
/// stack in ADU comes out as a sky, not as a black or white sheet.
#[test]
fn stretch_cli_maps_physical_samples_for_data_driven_models() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("mosaic.fits");
    physical_mosaic(&input);
    for model in ["auto-mtf", "median-anchor", "layered"] {
        let output = directory.path().join(format!("{model}.png"));
        let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
            .args([
                "stretch",
                input.to_str().unwrap(),
                "--output",
                output.to_str().unwrap(),
                model,
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{model}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let preview = image::open(&output).unwrap().to_luma8();
        let mut sky = preview
            .enumerate_pixels()
            .filter(|(x, y, _)| x + y >= 40)
            .map(|(_, _, pixel)| pixel.0[0])
            .collect::<Vec<_>>();
        sky.sort_unstable();
        let median = sky[sky.len() / 2];
        assert!((15..=160).contains(&median), "{model}: sky median {median}");
        let white = sky.iter().filter(|&&value| value == 255).count();
        assert!(white * 20 < sky.len(), "{model}: {white} white pixels");
    }
}

/// Explicit levels are read in the samples' own units unless asked
/// otherwise, so a linear stretch between two ADU levels still works.
#[test]
fn stretch_cli_keeps_explicit_levels_in_sample_units() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("mosaic.fits");
    let output = directory.path().join("linear.png");
    physical_mosaic(&input);
    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "stretch",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "linear",
            "--black",
            "3300",
            "--white",
            "3400",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let preview = image::open(&output).unwrap().to_luma8();
    let middle = preview.get_pixel(100, 100).0[0];
    assert!((40..=250).contains(&middle), "{middle}");
}
