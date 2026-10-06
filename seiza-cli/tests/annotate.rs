use seiza_fits::{F32ImageData, write_f32_image};
use std::process::Command;

/// A linear frame as a camera writes it: a dark sky at 1% of full scale
/// with a few faint stars.
fn linear_frame(path: &std::path::Path) {
    let (width, height) = (200usize, 150usize);
    let mut values = (0..width * height)
        .map(|index| 0.01 + ((index * 7919) % 17) as f32 * 0.0004)
        .collect::<Vec<_>>();
    for &(sx, sy) in &[(40.3, 30.7), (120.6, 90.2), (170.1, 40.8), (60.5, 120.4)] {
        for y in 0..height {
            for x in 0..width {
                let r2 = (x as f32 - sx).powi(2) + (y as f32 - sy).powi(2);
                values[y * width + x] += 0.05 * (-r2 / 4.0).exp();
            }
        }
    }
    write_f32_image(path, width, height, F32ImageData::Mono(&values), &[]).unwrap();
}

fn median_luma(path: &std::path::Path) -> u8 {
    let mut luma = image::open(path).unwrap().to_luma8().into_raw();
    luma.sort_unstable();
    luma[luma.len() / 2]
}

/// `--annotate` output is a picture to look at, so FITS input gets a display
/// stretch whatever the detection backend; the raw linear samples would
/// draw an almost black frame.
#[test]
fn annotated_fits_is_stretched_for_display_with_either_backend() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("linear.fits");
    linear_frame(&input);
    for backend in ["f32", "u8"] {
        let output = directory.path().join(format!("annotated-{backend}.png"));
        let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
            .args(["--detection-backend", backend, "detect"])
            .arg(&input)
            .arg("--annotate")
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        // Unstretched, a 1% sky is about level 3 of 255.
        let sky = median_luma(&output);
        assert!(sky > 40, "{backend}: sky median {sky}");
    }
}
