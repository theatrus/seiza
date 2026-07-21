use seiza_fits::{F32ImageData, FitsImage, HeaderValue, Pixels, write_f32_image};
use std::path::Path;
use std::process::Command;

fn write_mono(path: &Path, values: &[f32]) {
    write_f32_image(path, 2, 2, F32ImageData::Mono(values), &[]).unwrap();
}

fn synthetic_star_field(shift_x: isize, shift_y: isize, gain: f32) -> Vec<f32> {
    const WIDTH: usize = 128;
    const HEIGHT: usize = 128;
    let mut values = (0..WIDTH * HEIGHT)
        .map(|index| ((index * 37 + index / WIDTH * 19) % 29) as f32 * 1.0e-4)
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
        let x = (x as isize + shift_x) as usize;
        let y = (y as isize + shift_y) as usize;
        let peak = gain * (5.0 + index as f32 * 0.7);
        for (dx, dy, weight) in [
            (0, 0, 1.0),
            (-1, 0, 0.45),
            (1, 0, 0.45),
            (0, -1, 0.45),
            (0, 1, 0.45),
        ] {
            let sample_x = (x as isize + dx) as usize;
            let sample_y = (y as isize + dy) as usize;
            values[sample_y * WIDTH + sample_x] += peak * weight;
        }
    }
    values
}

#[test]
fn narrowband_cli_writes_rgb_fits_and_preview() {
    let directory = tempfile::tempdir().unwrap();
    let ha = directory.path().join("ha.fits");
    let oiii = directory.path().join("oiii.fits");
    let sii = directory.path().join("sii.fits");
    let output = directory.path().join("sho.fits");
    let preview = directory.path().join("sho.png");
    write_mono(&ha, &[0.1, 0.2, 0.3, 0.4]);
    write_mono(&oiii, &[0.2, 0.3, 0.4, 0.5]);
    write_mono(&sii, &[0.3, 0.4, 0.5, 0.6]);

    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "color",
            "narrowband",
            "--ha",
            ha.to_str().unwrap(),
            "--oiii",
            oiii.to_str().unwrap(),
            "--sii",
            sii.to_str().unwrap(),
            "--palette",
            "sho",
            "--normalization",
            "none",
            "--no-register",
            "--output",
            output.to_str().unwrap(),
            "--preview",
            preview.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );

    let fits = FitsImage::open(&output).unwrap();
    assert_eq!((fits.width, fits.height, fits.planes), (2, 2, 3));
    assert_eq!(
        fits.header("SEIZACLR").and_then(HeaderValue::as_str),
        Some("SHO")
    );
    assert_eq!(
        fits.header("SEIZATRF").and_then(HeaderValue::as_str),
        Some("LINEAR")
    );
    match fits.pixels {
        Pixels::F32(values) => assert_eq!(
            values,
            [
                0.3, 0.4, 0.5, 0.6, // red = SII
                0.1, 0.2, 0.3, 0.4, // green = H-alpha
                0.2, 0.3, 0.4, 0.5, // blue = OIII
            ]
        ),
        pixels => panic!("expected f32 output, got {pixels:?}"),
    }
    assert_eq!(image::open(preview).unwrap().to_rgb8().dimensions(), (2, 2));
}

#[test]
fn foraxx_cli_marks_display_referred_output() {
    let directory = tempfile::tempdir().unwrap();
    let ha = directory.path().join("ha.fits");
    let oiii = directory.path().join("oiii.fits");
    let output = directory.path().join("foraxx-hoo.fits");
    write_mono(&ha, &[0.1, 0.2, 0.3, 0.4]);
    write_mono(&oiii, &[0.2, 0.3, 0.4, 0.5]);

    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "color",
            "narrowband",
            "--ha",
            ha.to_str().unwrap(),
            "--oiii",
            oiii.to_str().unwrap(),
            "--palette",
            "foraxx-hoo",
            "--no-register",
            "--output",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let fits = FitsImage::open(&output).unwrap();
    assert_eq!(
        fits.header("SEIZATRF").and_then(HeaderValue::as_str),
        Some("DISPLAY")
    );
}

#[test]
fn foraxx_cli_rejects_sensor_units_when_normalization_is_disabled() {
    let directory = tempfile::tempdir().unwrap();
    let ha = directory.path().join("ha.fits");
    let oiii = directory.path().join("oiii.fits");
    let preview = directory.path().join("foraxx-hoo.png");
    write_mono(&ha, &[1_000.0, 1_100.0, 1_200.0, 1_300.0]);
    write_mono(&oiii, &[900.0, 1_000.0, 1_100.0, 1_200.0]);

    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "color",
            "narrowband",
            "--ha",
            ha.to_str().unwrap(),
            "--oiii",
            oiii.to_str().unwrap(),
            "--palette",
            "foraxx-hoo",
            "--normalization",
            "none",
            "--no-register",
            "--preview",
            preview.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("finite samples in [0, 1]"),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!preview.exists());
}

#[test]
fn rgb_cli_registers_shifted_filter_stacks_by_default() {
    let directory = tempfile::tempdir().unwrap();
    let red = directory.path().join("red.fits");
    let green = directory.path().join("green.fits");
    let blue = directory.path().join("blue.fits");
    let output = directory.path().join("rgb.fits");
    write_f32_image(
        &red,
        128,
        128,
        F32ImageData::Mono(&synthetic_star_field(0, 0, 1.0)),
        &[],
    )
    .unwrap();
    write_f32_image(
        &green,
        128,
        128,
        F32ImageData::Mono(&synthetic_star_field(4, -3, 1.5)),
        &[],
    )
    .unwrap();
    write_f32_image(
        &blue,
        128,
        128,
        F32ImageData::Mono(&synthetic_star_field(-5, 2, 2.0)),
        &[],
    )
    .unwrap();

    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "color",
            "rgb",
            "--red",
            red.to_str().unwrap(),
            "--green",
            green.to_str().unwrap(),
            "--blue",
            blue.to_str().unwrap(),
            "--normalization",
            "none",
            "--output",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(stdout.contains("registered green:"), "{stdout}");
    assert!(stdout.contains("registered blue:"), "{stdout}");

    let fits = FitsImage::open(&output).unwrap();
    assert_eq!((fits.width, fits.height, fits.planes), (128, 128, 3));
    assert_eq!(
        fits.header("SEIZATRF").and_then(HeaderValue::as_str),
        Some("LINEAR")
    );
    match fits.pixels {
        Pixels::F32(values) => {
            let pixels_per_plane = 128 * 128;
            let reference_star = 17 * 128 + 18;
            assert!((values[reference_star] - 5.0).abs() < 0.01);
            assert!((values[pixels_per_plane + reference_star] - 7.5).abs() < 0.01);
            assert!((values[pixels_per_plane * 2 + reference_star] - 10.0).abs() < 0.01);
        }
        pixels => panic!("expected f32 output, got {pixels:?}"),
    }
}

#[test]
fn narrowband_cli_validates_and_uses_only_palette_inputs() {
    let directory = tempfile::tempdir().unwrap();
    let missing_ha = directory.path().join("missing-ha.fits");
    let missing_oiii = directory.path().join("missing-oiii.fits");
    let missing_sii = directory.path().join("missing-sii.fits");
    let output = directory.path().join("color.fits");

    let missing_required = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "color",
            "narrowband",
            "--ha",
            missing_ha.to_str().unwrap(),
            "--oiii",
            missing_oiii.to_str().unwrap(),
            "--palette",
            "sho",
            "--output",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!missing_required.status.success());
    assert!(String::from_utf8_lossy(&missing_required.stderr).contains("SHO requires --sii"));

    let ha = directory.path().join("ha.fits");
    let oiii = directory.path().join("oiii.fits");
    write_mono(&ha, &[0.1, 0.2, 0.3, 0.4]);
    write_mono(&oiii, &[0.2, 0.3, 0.4, 0.5]);
    let ignores_unused = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "color",
            "narrowband",
            "--ha",
            ha.to_str().unwrap(),
            "--oiii",
            oiii.to_str().unwrap(),
            "--sii",
            missing_sii.to_str().unwrap(),
            "--palette",
            "hoo",
            "--no-register",
            "--output",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        ignores_unused.status.success(),
        "{}",
        String::from_utf8_lossy(&ignores_unused.stderr)
    );
}
