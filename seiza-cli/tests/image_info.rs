use seiza::raster::test_support::{Tag, Value, ascii, field, jpeg_with_exif};
use std::process::Command;

fn image_info(path: &std::path::Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args(["image-info", path.to_str().unwrap()])
        .output()
        .unwrap()
}

#[test]
fn image_info_reports_orientation_time_and_scale_hint() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("phone.jpg");
    std::fs::write(
        &path,
        jpeg_with_exif(
            &image::DynamicImage::new_rgb8(40, 30),
            &[
                field(Tag::Orientation, Value::Short(vec![6])),
                ascii(Tag::DateTimeOriginal, "2026:10:03 19:08:11"),
                ascii(Tag::GPSDateStamp, "2026:10:04"),
                field(
                    Tag::GPSTimeStamp,
                    Value::Rational(vec![(2, 1).into(), (8, 1).into(), (11, 1).into()]),
                ),
                field(Tag::FocalLengthIn35mmFilm, Value::Short(vec![24])),
            ],
        ),
    )
    .unwrap();
    let output = image_info(&path);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let info: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    // Orientation 6 turns the 40×30 stored frame upright as 30×40.
    assert_eq!(
        info["coordinates"]["original_dimensions"],
        serde_json::json!([40, 30])
    );
    assert_eq!(
        info["coordinates"]["oriented_dimensions"],
        serde_json::json!([30, 40])
    );
    assert_eq!(info["coordinates"]["orientation_applied"], 6);
    // No UTC offset, so the GPS stamp supplies the time.
    assert_eq!(info["metadata"]["capture_time_utc"], "2026-10-04T02:08:11Z");
    assert_eq!(info["metadata"]["capture_time_source"], "gps");
    assert!(
        info["scale_hint"]["nominal_arcsec_per_pixel"]
            .as_f64()
            .unwrap()
            > 1000.0
    );
}

#[test]
fn image_info_explains_that_it_does_not_read_fits() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("frame.fits");
    std::fs::write(&path, b"SIMPLE  =                    T").unwrap();
    let output = image_info(&path);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("FITS or XISF"));
}
