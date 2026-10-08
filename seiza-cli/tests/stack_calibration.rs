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
