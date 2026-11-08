use seiza::Wcs;
use seiza::catalog::TileSetBuilder;
use seiza_fits::F32ImageData;
use seiza_image_io::{SaveOptions, save_f32};
use std::process::Command;

/// Exercise the printed summary through detection and a real catalog solve,
/// without an external catalog or network service.
#[test]
fn solve_summary_reconstructs_the_field_for_normal_and_mirrored_images() {
    let directory = tempfile::tempdir().unwrap();
    let (width, height) = (800, 600);
    let crpix = ((width - 1) as f64 / 2.0, (height - 1) as f64 / 2.0);
    for flipped in [false, true] {
        for angle in [33.5, -88.33] {
            let truth = Wcs::from_center_scale_rotation(
                (334.42417, 55.99702),
                crpix,
                0.6037,
                angle,
                flipped,
            );
            let mut catalog = TileSetBuilder::new(2, 2000.0, "rotation regression");
            let mut data = (0..width * height)
                .map(|i| 100.0 + ((i * 7919) % 13) as f32 * 0.01)
                .collect::<Vec<_>>();
            for index in 0..100 {
                let x = 25.0 + ((index * 7919) % 750) as f64 + (index % 7) as f64 * 0.13;
                let y = 25.0 + ((index * 6271) % 550) as f64 + (index % 5) as f64 * 0.17;
                let magnitude = 7.0 + index as f64 * 0.03;
                let (ra, dec) = truth.pixel_to_world(x, y);
                catalog.add(ra, dec, magnitude as f32);
                let peak = 5000.0 * 10.0_f64.powf(-0.4 * (magnitude - 7.0));
                for py in y as usize - 7..=y as usize + 7 {
                    for px in x as usize - 7..=x as usize + 7 {
                        let r2 = (px as f64 - x).powi(2) + (py as f64 - y).powi(2);
                        data[py * width + px] += (peak * (-r2 / 4.5).exp()) as f32;
                    }
                }
            }
            let stars = directory.path().join("stars.bin");
            catalog.write_to(&stars).unwrap();
            let image = directory.path().join("field.fits");
            save_f32(
                &image,
                width,
                height,
                F32ImageData::Mono(&data),
                &SaveOptions::default(),
            )
            .unwrap();
            let output = Command::new(env!("CARGO_BIN_EXE_seiza"))
                .arg("solve")
                .arg(&image)
                .arg("--data")
                .arg(&stars)
                .args([
                    "--ra",
                    "334.42417",
                    "--dec",
                    "55.99702",
                    "--scale",
                    "0.6037",
                    "--detection-backend",
                    "f32",
                ])
                .output()
                .unwrap();
            let text = String::from_utf8(output.stdout).unwrap();
            assert!(
                output.status.success(),
                "{text}\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let line = |key: &str| {
                text.lines()
                    .find(|line| line.trim_start().starts_with(key))
                    .unwrap()
            };
            let rotation = line("rotation");
            let reported: f64 = rotation
                .split_once(':')
                .unwrap()
                .1
                .split_once('°')
                .unwrap()
                .0
                .trim()
                .parse()
                .unwrap();
            assert!((reported - angle).abs() < 0.01, "{text}");
            assert!(rotation.ends_with("(CD matrix)"), "{rotation}");
            assert!(line("parity").ends_with(if flipped { "mirrored" } else { "normal" }));
            let center = line("center")
                .rsplit_once('(')
                .unwrap()
                .1
                .trim_end_matches(')');
            let (ra, dec) = center.split_once(',').unwrap();
            let degrees = |value: &str| value.trim().trim_end_matches('°').parse::<f64>().unwrap();
            let scale = line("pixel scale")
                .split_once(':')
                .unwrap()
                .1
                .split_once('"')
                .unwrap()
                .0
                .trim()
                .parse()
                .unwrap();
            let rebuilt = Wcs::from_center_scale_rotation(
                (degrees(ra), degrees(dec)),
                crpix,
                scale,
                reported,
                flipped,
            );
            let mut maximum_error = 0.0_f64;
            for (index, &(ra, dec)) in rebuilt
                .footprint(width as u32, height as u32)
                .iter()
                .enumerate()
            {
                let (x, y) = truth.world_to_pixel(ra, dec).unwrap();
                let expected = [
                    (0.0, 0.0),
                    ((width - 1) as f64, 0.0),
                    ((width - 1) as f64, (height - 1) as f64),
                    (0.0, (height - 1) as f64),
                ][index];
                maximum_error = maximum_error.max((x - expected.0).hypot(y - expected.1));
            }
            assert!(
                maximum_error < 0.5,
                "{text}\nmaximum corner displacement={maximum_error} px"
            );
            println!(
                "rotation={angle}, flipped={flipped}: maximum corner displacement={maximum_error:.6} px"
            );
        }
    }
}
