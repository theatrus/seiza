//! Display-oriented raster pixels and optional JPEG acquisition metadata.
//!
//! Orientation is applied before any detection or WCS fit. EXIF acquisition
//! timestamps and ExposureTime do not establish a continuous shutter interval.

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDateTime, SecondsFormat, Utc};
use exif::{Exif, In, Tag, Value};
use image::{DynamicImage, ImageDecoder, ImageReader};
use serde::Serialize;
use std::{fs::File, io::BufReader, path::Path};

#[derive(Debug, Serialize)]
pub(crate) struct PixelCoordinates {
    pub original_dimensions: (u32, u32),
    pub oriented_dimensions: (u32, u32),
    /// EXIF's 1..=8 transform from stored pixels to the working image.
    pub orientation_applied: u8,
    pub convention: &'static str,
}

pub(crate) struct RasterImage {
    pub pixels: DynamicImage,
    pub coordinates: PixelCoordinates,
}

pub(crate) fn open(path: &Path) -> Result<RasterImage> {
    let mut decoder = ImageReader::open(path)?
        .with_guessed_format()?
        .into_decoder()
        .with_context(|| format!("failed to open {}", path.display()))?;
    let original_dimensions = decoder.dimensions();
    let orientation = decoder.orientation()?;
    let mut pixels = DynamicImage::from_decoder(decoder)?;
    pixels.apply_orientation(orientation);
    let coordinates = PixelCoordinates {
        original_dimensions,
        oriented_dimensions: (pixels.width(), pixels.height()),
        orientation_applied: orientation.to_exif(),
        convention: "zero-based pixel centers in the EXIF-oriented image",
    };
    Ok(RasterImage {
        pixels,
        coordinates,
    })
}

#[derive(Debug, Default, Serialize)]
pub(crate) struct JpegMetadata {
    pub make: Option<String>,
    pub model: Option<String>,
    pub date_time_original: Option<String>,
    /// Keep the string, including leading zeros (e.g. "026").
    pub sub_sec_time_original: Option<String>,
    pub offset_time_original: Option<String>,
    /// Only populated when DateTimeOriginal has an explicit timezone offset.
    pub capture_time_utc: Option<String>,
    pub exposure_seconds: Option<f64>,
    pub gps_latitude_deg: Option<f64>,
    pub gps_longitude_deg: Option<f64>,
    /// EXIF altitude with its original reference; not an ellipsoid height.
    pub gps_altitude_m: Option<f64>,
    pub gps_altitude_ref: Option<u32>,
    pub gps_image_direction_deg: Option<f64>,
    /// "M" (magnetic north) or "T" (true north); never silently converted.
    pub gps_image_direction_ref: Option<String>,
    pub focal_length_mm: Option<f64>,
    pub focal_length_35mm: Option<f64>,
    pub exif_orientation: Option<u32>,
    pub warnings: Vec<String>,
}

/// Metadata is advisory: missing or malformed EXIF must not prevent solving.
pub(crate) fn read_metadata(path: &Path) -> JpegMetadata {
    if !path.extension().and_then(|s| s.to_str()).is_some_and(|s| {
        ["jpg", "jpeg", "jfif"]
            .iter()
            .any(|ext| s.eq_ignore_ascii_case(ext))
    }) {
        return JpegMetadata::default();
    }
    let result = File::open(path).map_err(exif::Error::Io).and_then(|file| {
        exif::Reader::new()
            .continue_on_error(true)
            .read_from_container(&mut BufReader::new(file))
    });
    match result {
        Ok(exif) => from_exif(&exif),
        Err(exif::Error::NotFound(_)) => JpegMetadata::default(),
        Err(exif::Error::PartialResult(partial)) => {
            let (exif, errors) = partial.into_inner();
            let mut metadata = from_exif(&exif);
            metadata
                .warnings
                .extend(errors.iter().map(|e| format!("EXIF: {e}")));
            metadata
        }
        Err(error) => JpegMetadata {
            warnings: vec![format!("EXIF metadata unavailable: {error}")],
            ..Default::default()
        },
    }
}

fn from_exif(exif: &Exif) -> JpegMetadata {
    let mut metadata = JpegMetadata {
        make: text(exif, Tag::Make),
        model: text(exif, Tag::Model),
        date_time_original: text(exif, Tag::DateTimeOriginal),
        sub_sec_time_original: text(exif, Tag::SubSecTimeOriginal),
        offset_time_original: text(exif, Tag::OffsetTimeOriginal),
        exposure_seconds: positive_number(exif, Tag::ExposureTime),
        focal_length_mm: positive_number(exif, Tag::FocalLength),
        focal_length_35mm: positive_number(exif, Tag::FocalLengthIn35mmFilm),
        exif_orientation: uint(exif, Tag::Orientation).filter(|n| (1..=8).contains(n)),
        ..Default::default()
    };
    if let Some(date) = &metadata.date_time_original {
        match capture_time(
            date,
            metadata.sub_sec_time_original.as_deref(),
            metadata.offset_time_original.as_deref(),
        ) {
            Ok(time) => metadata.capture_time_utc = time,
            Err(error) => metadata.warnings.push(error),
        }
    }
    metadata.gps_latitude_deg =
        coordinate(exif, Tag::GPSLatitude, Tag::GPSLatitudeRef, "N", "S", 90.0);
    metadata.gps_longitude_deg = coordinate(
        exif,
        Tag::GPSLongitude,
        Tag::GPSLongitudeRef,
        "E",
        "W",
        180.0,
    );
    if metadata.gps_latitude_deg.is_none() || metadata.gps_longitude_deg.is_none() {
        if exif.get_field(Tag::GPSLatitude, In::PRIMARY).is_some()
            || exif.get_field(Tag::GPSLongitude, In::PRIMARY).is_some()
        {
            metadata.warnings.push("GPS position is incomplete or invalid; latitude/longitude and hemisphere references are required".into());
        }
        metadata.gps_latitude_deg = None;
        metadata.gps_longitude_deg = None;
    }
    metadata.gps_altitude_ref = uint(exif, Tag::GPSAltitudeRef);
    metadata.gps_altitude_m = number(exif, Tag::GPSAltitude).filter(|v| *v >= 0.0);
    metadata.gps_image_direction_deg =
        number(exif, Tag::GPSImgDirection).filter(|v| (0.0..360.0).contains(v));
    metadata.gps_image_direction_ref =
        text(exif, Tag::GPSImgDirectionRef).filter(|s| s == "M" || s == "T");
    metadata
}

fn text(exif: &Exif, tag: Tag) -> Option<String> {
    let Value::Ascii(values) = &exif.get_field(tag, In::PRIMARY)?.value else {
        return None;
    };
    let value = std::str::from_utf8(values.first()?)
        .ok()?
        .trim_matches('\0')
        .trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn uint(exif: &Exif, tag: Tag) -> Option<u32> {
    exif.get_field(tag, In::PRIMARY)?.value.get_uint(0)
}

fn number(exif: &Exif, tag: Tag) -> Option<f64> {
    let value = &exif.get_field(tag, In::PRIMARY)?.value;
    let number = match value {
        Value::Rational(values) => values.first()?.to_f64(),
        Value::SRational(values) => values.first()?.to_f64(),
        _ => value.get_uint(0)? as f64,
    };
    number.is_finite().then_some(number)
}

fn positive_number(exif: &Exif, tag: Tag) -> Option<f64> {
    number(exif, tag).filter(|n| *n > 0.0)
}

fn coordinate(
    exif: &Exif,
    tag: Tag,
    reference: Tag,
    positive: &str,
    negative: &str,
    limit: f64,
) -> Option<f64> {
    let Value::Rational(parts) = &exif.get_field(tag, In::PRIMARY)?.value else {
        return None;
    };
    if parts.len() != 3 {
        return None;
    }
    let (degrees, minutes, seconds) = (parts[0].to_f64(), parts[1].to_f64(), parts[2].to_f64());
    if !degrees.is_finite() || !(0.0..60.0).contains(&minutes) || !(0.0..60.0).contains(&seconds) {
        return None;
    }
    let value = degrees + minutes / 60.0 + seconds / 3600.0;
    if !(0.0..=limit).contains(&value) {
        return None;
    }
    let sign = match text(exif, reference)?.as_str() {
        s if s == positive => 1.0,
        s if s == negative => -1.0,
        _ => return None,
    };
    Some(sign * value)
}

fn capture_time(
    date: &str,
    subsecond: Option<&str>,
    offset: Option<&str>,
) -> std::result::Result<Option<String>, String> {
    let base = NaiveDateTime::parse_from_str(date, "%Y:%m:%d %H:%M:%S")
        .map_err(|_| "invalid EXIF DateTimeOriginal".to_string())?;
    let Some(offset) = offset else {
        return Err("EXIF DateTimeOriginal has no OffsetTimeOriginal; UTC time is unknown (use --time to override)".into());
    };
    let fraction = match subsecond {
        Some(s) if !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit()) => {
            format!(".{}", &s[..s.len().min(9)])
        }
        None => String::new(),
        _ => return Err("invalid EXIF SubSecTimeOriginal".into()),
    };
    let text = format!("{}{fraction}{offset}", base.format("%Y-%m-%dT%H:%M:%S"));
    let time = DateTime::parse_from_rfc3339(&text)
        .map_err(|_| "invalid EXIF OffsetTimeOriginal".to_string())?;
    Ok(Some(
        time.with_timezone(&Utc)
            .to_rfc3339_opts(SecondsFormat::AutoSi, true),
    ))
}

#[derive(Debug, Serialize)]
pub(crate) struct ScaleHint {
    pub nominal_arcsec_per_pixel: f64,
    pub min_arcsec_per_pixel: f64,
    pub max_arcsec_per_pixel: f64,
}

impl JpegMetadata {
    /// 35mm-equivalent focal length describes a field of view, not a sensor
    /// pixel pitch. Use the full-frame diagonal and the *decoded* dimensions.
    /// Cropping, aspect ratio and lens correction make this only a broad prior.
    pub(crate) fn scale_hint(&self, dimensions: (u32, u32)) -> Option<ScaleHint> {
        let focal = self
            .focal_length_35mm
            .filter(|f| f.is_finite() && *f > 0.0)?;
        if dimensions.0 == 0 || dimensions.1 == 0 {
            return None;
        }
        let diagonal_pixels = (dimensions.0 as f64).hypot(dimensions.1 as f64);
        let focal_pixels = focal * diagonal_pixels / 36.0_f64.hypot(24.0);
        let nominal = (1.0 / focal_pixels).atan().to_degrees() * 3600.0;
        Some(ScaleHint {
            nominal_arcsec_per_pixel: nominal,
            min_arcsec_per_pixel: nominal / 2.0,
            max_arcsec_per_pixel: nominal * 2.0,
        })
    }
}

pub(crate) fn scale_bounds(
    metadata: &JpegMetadata,
    dimensions: (u32, u32),
    explicit_min: Option<f64>,
    explicit_max: Option<f64>,
) -> Result<(f64, f64)> {
    let hint = metadata.scale_hint(dimensions);
    let min = explicit_min.unwrap_or_else(|| hint.as_ref().map_or(0.1, |h| h.min_arcsec_per_pixel));
    let max =
        explicit_max.unwrap_or_else(|| hint.as_ref().map_or(20.0, |h| h.max_arcsec_per_pixel));
    anyhow::ensure!(
        min.is_finite() && max.is_finite() && min > 0.0 && max >= min,
        "pixel-scale bounds must be positive, finite and ordered; pass --min-scale and --max-scale to override EXIF"
    );
    Ok((min, max))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use exif::{Field, Rational};
    use std::io::Cursor;

    pub(crate) fn field(tag: Tag, value: Value) -> Field {
        Field {
            tag,
            ifd_num: In::PRIMARY,
            value,
        }
    }

    fn ascii(tag: Tag, text: &str) -> Field {
        field(tag, Value::Ascii(vec![text.as_bytes().to_vec()]))
    }

    fn encoded(fields: &[Field], little_endian: bool) -> Vec<u8> {
        let mut writer = exif::experimental::Writer::new();
        for field in fields {
            writer.push_field(field);
        }
        let mut cursor = Cursor::new(Vec::new());
        writer.write(&mut cursor, little_endian).unwrap();
        cursor.into_inner()
    }

    pub(crate) fn jpeg_with_exif(image: &DynamicImage, fields: &[Field]) -> Vec<u8> {
        let mut jpeg = Cursor::new(Vec::new());
        image.write_to(&mut jpeg, image::ImageFormat::Jpeg).unwrap();
        let jpeg = jpeg.into_inner();
        let mut payload = b"Exif\0\0".to_vec();
        payload.extend(encoded(fields, false));
        let mut result = jpeg[..2].to_vec();
        result.extend([0xff, 0xe1]);
        result.extend(((payload.len() + 2) as u16).to_be_bytes());
        result.extend(payload);
        result.extend_from_slice(&jpeg[2..]);
        result
    }

    fn parse(fields: &[Field], little_endian: bool) -> JpegMetadata {
        from_exif(
            &exif::Reader::new()
                .read_raw(encoded(fields, little_endian))
                .unwrap(),
        )
    }

    #[test]
    fn time_preserves_leading_zero_subseconds_and_crosses_utc_date() {
        for little_endian in [true, false] {
            let metadata = parse(
                &[
                    ascii(Tag::DateTimeOriginal, "2026:10:03 19:08:11"),
                    ascii(Tag::SubSecTimeOriginal, "026"),
                    ascii(Tag::OffsetTimeOriginal, "-07:00"),
                ],
                little_endian,
            );
            assert_eq!(
                metadata.capture_time_utc.as_deref(),
                Some("2026-10-04T02:08:11.026Z")
            );
            assert_eq!(metadata.sub_sec_time_original.as_deref(), Some("026"));
        }
        assert_eq!(
            capture_time("2026:10:04 00:00:00", Some("007"), Some("+09:00"))
                .unwrap()
                .as_deref(),
            Some("2026-10-03T15:00:00.007Z")
        );
    }

    #[test]
    fn missing_timezone_and_bad_times_do_not_invent_utc() {
        for (date, fraction, offset) in [
            ("2026:10:03 19:08:11", None, None),
            ("2026:02:30 19:08:11", None, Some("-07:00")),
            ("2026:10:03 19:08:11", Some("abc"), Some("-07:00")),
            ("2026:10:03 19:08:11", None, Some("nonsense")),
        ] {
            assert!(capture_time(date, fraction, offset).is_err());
        }
    }

    #[test]
    fn gps_requires_hemispheres_and_valid_dms() {
        let coords = |d, m, s| {
            Value::Rational(vec![
                Rational::from((d, 1)),
                Rational::from((m, 1)),
                Rational::from((s, 1)),
            ])
        };
        for little_endian in [true, false] {
            let metadata = parse(
                &[
                    field(Tag::GPSLatitude, coords(32, 22, 5)),
                    ascii(Tag::GPSLatitudeRef, "S"),
                    field(Tag::GPSLongitude, coords(110, 43, 6)),
                    ascii(Tag::GPSLongitudeRef, "W"),
                    field(Tag::GPSAltitude, Value::Rational(vec![(50, 1).into()])),
                    field(Tag::GPSAltitudeRef, Value::Byte(vec![1])),
                    field(Tag::GPSImgDirection, Value::Rational(vec![(350, 1).into()])),
                    ascii(Tag::GPSImgDirectionRef, "M"),
                ],
                little_endian,
            );
            assert!((metadata.gps_latitude_deg.unwrap() + 32.3680555556).abs() < 1e-8);
            assert!((metadata.gps_longitude_deg.unwrap() + 110.7183333333).abs() < 1e-8);
            assert_eq!(metadata.gps_altitude_m, Some(50.0));
            assert_eq!(metadata.gps_altitude_ref, Some(1));
            assert_eq!(metadata.gps_image_direction_ref.as_deref(), Some("M"));
        }
        let metadata = parse(
            &[
                field(Tag::GPSLatitude, coords(32, 60, 0)),
                ascii(Tag::GPSLatitudeRef, "N"),
            ],
            false,
        );
        assert!(metadata.gps_latitude_deg.is_none());
        assert!(!metadata.warnings.is_empty());
    }

    #[test]
    fn unknown_exposure_is_not_a_capture_interval() {
        let metadata = parse(
            &[field(
                Tag::ExposureTime,
                Value::Rational(vec![(16, 1).into()]),
            )],
            false,
        );
        assert_eq!(metadata.exposure_seconds, Some(16.0));
        assert!(metadata.capture_time_utc.is_none());
        let invalid = parse(
            &[field(
                Tag::ExposureTime,
                Value::Rational(vec![(1, 0).into()]),
            )],
            false,
        );
        assert!(invalid.exposure_seconds.is_none());
    }

    #[test]
    fn absent_or_damaged_exif_does_not_block_pixel_loading() {
        let dir = tempfile::tempdir().unwrap();
        let image = DynamicImage::new_rgb8(32, 24);
        let path = dir.path().join("no-exif.jpg");
        image.save(&path).unwrap();
        assert!(read_metadata(&path).warnings.is_empty());
        assert!(read_metadata(&path).capture_time_utc.is_none());
        let plain = std::fs::read(&path).unwrap();
        let payload = b"Exif\0\0broken-tiff";
        let mut damaged = plain[..2].to_vec();
        damaged.extend([0xff, 0xe1]);
        damaged.extend(((payload.len() + 2) as u16).to_be_bytes());
        damaged.extend(payload);
        damaged.extend_from_slice(&plain[2..]);
        std::fs::write(&path, damaged).unwrap();
        assert!(!read_metadata(&path).warnings.is_empty());
        assert_eq!(
            open(&path).unwrap().coordinates.oriented_dimensions,
            (32, 24)
        );
    }

    #[test]
    fn wide_field_prior_scales_with_dimensions_and_explicit_bounds_win() {
        let metadata = JpegMetadata {
            focal_length_35mm: Some(24.0),
            ..Default::default()
        };
        let native = metadata.scale_hint((4032, 3024)).unwrap();
        assert!((native.nominal_arcsec_per_pixel - 73.8).abs() < 0.1);
        let smaller = metadata.scale_hint((1600, 1200)).unwrap();
        assert!(
            (smaller.nominal_arcsec_per_pixel / native.nominal_arcsec_per_pixel - 2.52).abs()
                < 1e-5
        );
        assert_eq!(
            scale_bounds(&metadata, (4032, 3024), Some(0.1), Some(20.0)).unwrap(),
            (0.1, 20.0)
        );
        assert_eq!(
            scale_bounds(&JpegMetadata::default(), (4032, 3024), None, None).unwrap(),
            (0.1, 20.0)
        );
        assert!(scale_bounds(&metadata, (4032, 3024), Some(f64::NAN), None).is_err());
        assert!(scale_bounds(&metadata, (4032, 3024), Some(20.0), Some(1.0)).is_err());
        assert!(metadata.scale_hint((0, 3024)).is_none());
    }

    #[test]
    fn all_eight_exif_orientations_match_stored_pixel_coordinates() {
        // Each pixel has a distinct value; compare against independently
        // specified EXIF coordinate mappings, including both transposes.
        let source = DynamicImage::ImageRgb8(image::RgbImage::from_fn(3, 2, |x, y| {
            image::Rgb([((x + 3 * y) * 30) as u8, 0, 0])
        }));
        let dir = tempfile::tempdir().unwrap();
        for orientation in 1..=8 {
            let path = dir.path().join(format!("orientation-{orientation}.jpg"));
            std::fs::write(
                &path,
                jpeg_with_exif(
                    &source,
                    &[field(Tag::Orientation, Value::Short(vec![orientation]))],
                ),
            )
            .unwrap();
            let raw = image::open(&path).unwrap().to_rgb8();
            let loaded = open(&path).unwrap();
            let result = loaded.pixels.to_rgb8();
            assert_eq!(loaded.coordinates.orientation_applied, orientation as u8);
            assert_eq!(loaded.coordinates.original_dimensions, (3, 2));
            assert_eq!(
                loaded.coordinates.oriented_dimensions,
                if orientation >= 5 { (2, 3) } else { (3, 2) }
            );
            for y in 0..2 {
                for x in 0..3 {
                    let (nx, ny) = match orientation {
                        1 => (x, y),
                        2 => (2 - x, y),
                        3 => (2 - x, 1 - y),
                        4 => (x, 1 - y),
                        5 => (y, x),
                        6 => (1 - y, x),
                        7 => (1 - y, 2 - x),
                        8 => (y, 2 - x),
                        _ => unreachable!(),
                    };
                    assert_eq!(
                        result.get_pixel(nx, ny),
                        raw.get_pixel(x, y),
                        "orientation {orientation}"
                    );
                }
            }
        }
    }
}
