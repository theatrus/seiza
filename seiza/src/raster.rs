//! Ordinary raster images (JPEG, PNG, TIFF) as a camera or phone saved them.
//!
//! [`open_oriented`] and [`decode_oriented`] apply the EXIF Orientation tag
//! once, so star positions and any WCS fitted to them refer to the upright
//! image a viewer shows, not the stored pixel rows. Every reader of a raster
//! should go through them so detection, solving, overlays and previews share
//! one pixel frame.
//!
//! [`PhotoMetadata`] reads the EXIF fields that help a plate solve: capture
//! time, GPS position and the 35 mm-equivalent focal length, from which
//! [`ScaleSearch`] derives a pixel-scale range. All of it is advisory:
//! missing or damaged EXIF never stops an image from loading.

use crate::Error;
use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, SecondsFormat, Utc};
use exif::{Exif, In, Tag, Value};
use image::{DynamicImage, ImageDecoder, ImageReader};
use serde::Serialize;
use std::io::{BufRead, Cursor, Seek};
use std::path::Path;

/// How a decoded raster's pixels relate to the stored ones.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PixelCoordinates {
    /// Width and height as stored in the file.
    pub original_dimensions: (u32, u32),
    /// Width and height after the orientation was applied.
    pub oriented_dimensions: (u32, u32),
    /// The EXIF Orientation value (1–8) applied; 1 means none.
    pub orientation_applied: u8,
    pub convention: &'static str,
}

/// A decoded raster in its EXIF-oriented frame.
pub struct OrientedRaster {
    pub pixels: DynamicImage,
    pub coordinates: PixelCoordinates,
}

/// The most a local raster may take to decode. A large 16-bit TIFF passes
/// the decoder's default 512 MiB; a corrupt header claiming gigapixels is
/// refused rather than aborting the process for want of memory.
const LOCAL_DECODE_LIMIT: u64 = 16 << 30;

/// Open a raster file and apply its EXIF orientation.
///
/// A local file is the caller's own, so the decoder may take up to 16 GiB
/// rather than its default 512 MiB, which a large 16-bit TIFF exceeds;
/// [`decode_oriented`] keeps the default for bytes from elsewhere.
pub fn open_oriented(path: &Path) -> Result<OrientedRaster, Error> {
    let mut reader = ImageReader::open(path)
        .map_err(image::ImageError::IoError)?
        .with_guessed_format()
        .map_err(image::ImageError::IoError)?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(LOCAL_DECODE_LIMIT);
    reader.limits(limits);
    oriented(reader, Some(LOCAL_DECODE_LIMIT))
}

/// Decode raster bytes, such as an upload, and apply their EXIF orientation.
pub fn decode_oriented(bytes: &[u8]) -> Result<OrientedRaster, Error> {
    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(image::ImageError::IoError)?;
    oriented(reader, None)
}

/// The decoded raster, refused when its pixels would take more than
/// `most` bytes: the decoder checks only an image's sides once opened, so
/// a header claiming gigapixels would otherwise be allocated, and abort.
fn oriented<R: BufRead + Seek>(
    reader: ImageReader<R>,
    most: Option<u64>,
) -> Result<OrientedRaster, Error> {
    let mut decoder = reader.into_decoder()?;
    if let Some(most) = most
        && decoder.total_bytes() > most
    {
        return Err(
            image::ImageError::Limits(image::error::LimitError::from_kind(
                image::error::LimitErrorKind::InsufficientMemory,
            ))
            .into(),
        );
    }
    let original_dimensions = decoder.dimensions();
    let orientation = decoder.orientation()?;
    let mut pixels = DynamicImage::from_decoder(decoder)?;
    pixels.apply_orientation(orientation);
    Ok(OrientedRaster {
        coordinates: PixelCoordinates {
            original_dimensions,
            oriented_dimensions: (pixels.width(), pixels.height()),
            orientation_applied: orientation.to_exif(),
            convention: "zero-based pixel centers in the EXIF-oriented image",
        },
        pixels,
    })
}

/// EXIF fields that inform a plate solve.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct PhotoMetadata {
    pub make: Option<String>,
    pub model: Option<String>,
    pub date_time_original: Option<String>,
    /// Kept as text, with its leading zeros (e.g. "026").
    pub sub_sec_time_original: Option<String>,
    pub offset_time_original: Option<String>,
    /// UTC capture time, from DateTimeOriginal with its offset, or else
    /// from the GPS date and time stamps, which are UTC by definition.
    pub capture_time_utc: Option<String>,
    /// Where [`Self::capture_time_utc`] came from.
    pub capture_time_source: Option<CaptureTimeSource>,
    /// ExposureTime. It does not say when the shutter opened, so it is
    /// never combined with the capture time into an interval.
    pub exposure_seconds: Option<f64>,
    pub gps_latitude_deg: Option<f64>,
    pub gps_longitude_deg: Option<f64>,
    /// EXIF altitude with its original reference; not an ellipsoid height.
    pub gps_altitude_m: Option<f64>,
    pub gps_altitude_ref: Option<u32>,
    pub gps_image_direction_deg: Option<f64>,
    /// "M" (magnetic north) or "T" (true north); never converted.
    pub gps_image_direction_ref: Option<String>,
    pub focal_length_mm: Option<f64>,
    pub focal_length_35mm: Option<f64>,
    pub exif_orientation: Option<u32>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureTimeSource {
    /// DateTimeOriginal with OffsetTimeOriginal.
    DateTimeOriginal,
    /// GPSDateStamp with GPSTimeStamp.
    Gps,
}

impl PhotoMetadata {
    /// Read EXIF from a file. Files without EXIF give empty metadata.
    pub fn read(path: &Path) -> Self {
        match std::fs::File::open(path) {
            Ok(file) => Self::from_reader(&mut std::io::BufReader::new(file)),
            Err(error) => Self {
                warnings: vec![format!("EXIF metadata unavailable: {error}")],
                ..Default::default()
            },
        }
    }

    /// Read EXIF from file bytes of any container kamadak-exif knows
    /// (JPEG, TIFF, HEIF, PNG, WebP). The format is recognized from the
    /// content, not a file name.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self::from_reader(&mut Cursor::new(bytes))
    }

    fn from_reader<R: BufRead + Seek>(reader: &mut R) -> Self {
        let result = exif::Reader::new()
            .continue_on_error(true)
            .read_from_container(reader);
        match result {
            Ok(exif) => from_exif(&exif),
            // No EXIF, or not an image container kamadak-exif knows.
            Err(exif::Error::NotFound(_) | exif::Error::InvalidFormat("Unknown image format")) => {
                Self::default()
            }
            Err(exif::Error::PartialResult(partial)) => {
                let (exif, errors) = partial.into_inner();
                let mut metadata = from_exif(&exif);
                metadata
                    .warnings
                    .extend(errors.iter().map(|error| format!("EXIF: {error}")));
                metadata
            }
            Err(error) => Self {
                warnings: vec![format!("EXIF metadata unavailable: {error}")],
                ..Default::default()
            },
        }
    }

    /// The pixel scale the 35 mm-equivalent focal length implies for an
    /// image of these (decoded, oriented) dimensions.
    ///
    /// The equivalent focal length describes the field of view of the full
    /// frame, so this assumes the image is that whole frame, resized. A crop
    /// or an eyepiece breaks the assumption; [`ScaleSearch`] keeps the full
    /// range as a fallback for that reason.
    pub fn scale_hint(&self, dimensions: (u32, u32)) -> Option<ScaleHint> {
        let focal = self
            .focal_length_35mm
            .filter(|focal| focal.is_finite() && *focal > 0.0)?;
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

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ScaleHint {
    pub nominal_arcsec_per_pixel: f64,
    pub min_arcsec_per_pixel: f64,
    pub max_arcsec_per_pixel: f64,
}

/// The default blind-solve pixel-scale range, in arcseconds per pixel.
pub const DEFAULT_SCALE_RANGE: (f64, f64) = (0.1, 20.0);

/// Pixel-scale ranges for a blind solve, to be tried in order.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ScaleSearch {
    /// Each `(min, max)` in arcseconds per pixel.
    pub ranges: Vec<(f64, f64)>,
    /// Whether the first range came from the EXIF focal length.
    pub from_exif: bool,
}

impl ScaleSearch {
    /// Choose the ranges from explicit bounds and the EXIF hint.
    ///
    /// Explicit bounds always hold. A missing bound comes from the hint.
    /// Because a cropped photo or an eyepiece shot can lie outside the
    /// hint, a wide fallback follows it: the default range, stretched up to
    /// the hint's coarse end so a wide phone field stays inside, and still
    /// respecting any explicit bound. A hint bound that conflicts with an
    /// explicit one is dropped.
    pub fn new(
        metadata: &PhotoMetadata,
        dimensions: (u32, u32),
        explicit_min: Option<f64>,
        explicit_max: Option<f64>,
    ) -> Result<Self, Error> {
        let hint = match (explicit_min, explicit_max) {
            (Some(_), Some(_)) => None,
            _ => metadata.scale_hint(dimensions),
        };
        let Some(hint) = hint else {
            let range = (
                explicit_min.unwrap_or(DEFAULT_SCALE_RANGE.0),
                explicit_max.unwrap_or(DEFAULT_SCALE_RANGE.1),
            );
            check_range(range)?;
            return Ok(Self {
                ranges: vec![range],
                from_exif: false,
            });
        };
        let hinted = (
            explicit_min.unwrap_or(hint.min_arcsec_per_pixel),
            explicit_max.unwrap_or(hint.max_arcsec_per_pixel),
        );
        let fallback = (
            explicit_min.unwrap_or(DEFAULT_SCALE_RANGE.0),
            explicit_max.unwrap_or(DEFAULT_SCALE_RANGE.1.max(hint.max_arcsec_per_pixel)),
        );
        check_range(fallback)?;
        if check_range(hinted).is_err() {
            return Ok(Self {
                ranges: vec![fallback],
                from_exif: false,
            });
        }
        let mut ranges = vec![hinted];
        if fallback != hinted {
            ranges.push(fallback);
        }
        Ok(Self {
            ranges,
            from_exif: true,
        })
    }
}

fn check_range((min, max): (f64, f64)) -> Result<(), Error> {
    if min.is_finite() && max.is_finite() && min > 0.0 && max >= min {
        Ok(())
    } else {
        Err(Error::Solve(format!(
            "pixel-scale bounds must be positive, finite and ordered; got {min}–{max}\"/px"
        )))
    }
}

fn from_exif(exif: &Exif) -> PhotoMetadata {
    let mut metadata = PhotoMetadata {
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
    let mut time_warning = None;
    if let Some(date) = &metadata.date_time_original {
        match capture_time(
            date,
            metadata.sub_sec_time_original.as_deref(),
            metadata.offset_time_original.as_deref(),
        ) {
            Ok(time) => {
                metadata.capture_time_utc = Some(time);
                metadata.capture_time_source = Some(CaptureTimeSource::DateTimeOriginal);
            }
            Err(error) => time_warning = Some(error),
        }
    }
    if metadata.capture_time_utc.is_none()
        && let Some(time) = gps_time(exif)
    {
        metadata.capture_time_utc = Some(time);
        metadata.capture_time_source = Some(CaptureTimeSource::Gps);
        time_warning = None;
    }
    metadata.warnings.extend(time_warning);
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
            metadata.warnings.push(
                "GPS position is incomplete or invalid; latitude/longitude and hemisphere references are required".into(),
            );
        }
        metadata.gps_latitude_deg = None;
        metadata.gps_longitude_deg = None;
    } else if metadata.gps_latitude_deg == Some(0.0) && metadata.gps_longitude_deg == Some(0.0) {
        // Some apps write 0,0 when they have no fix.
        metadata
            .warnings
            .push("GPS position 0°, 0° looks like a missing fix; ignored".into());
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
) -> Result<String, String> {
    let base = NaiveDateTime::parse_from_str(date, "%Y:%m:%d %H:%M:%S")
        .map_err(|_| "invalid EXIF DateTimeOriginal".to_string())?;
    let Some(offset) = offset else {
        return Err("EXIF DateTimeOriginal has no OffsetTimeOriginal and there is no GPS time; UTC time is unknown (use --time to override)".into());
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
    Ok(time
        .with_timezone(&Utc)
        .to_rfc3339_opts(SecondsFormat::AutoSi, true))
}

/// GPSDateStamp ("YYYY:MM:DD") with GPSTimeStamp (three rationals), UTC.
fn gps_time(exif: &Exif) -> Option<String> {
    let date = NaiveDate::parse_from_str(&text(exif, Tag::GPSDateStamp)?, "%Y:%m:%d").ok()?;
    let Value::Rational(parts) = &exif.get_field(Tag::GPSTimeStamp, In::PRIMARY)?.value else {
        return None;
    };
    let [hours, minutes, seconds] = parts.as_slice() else {
        return None;
    };
    let (hours, minutes, seconds) = (hours.to_f64(), minutes.to_f64(), seconds.to_f64());
    if !(0.0..24.0).contains(&hours)
        || !(0.0..60.0).contains(&minutes)
        || !(0.0..61.0).contains(&seconds)
        || hours.fract() != 0.0
        || minutes.fract() != 0.0
    {
        return None;
    }
    let nanos = (seconds.fract() * 1e9).round() as u32;
    let time = NaiveTime::from_hms_nano_opt(
        hours as u32,
        minutes as u32,
        (seconds.trunc() as u32).min(59),
        nanos.min(999_999_999),
    )?;
    Some(
        date.and_time(time)
            .and_utc()
            .to_rfc3339_opts(SecondsFormat::AutoSi, true),
    )
}

/// Builders for JPEG fixtures carrying EXIF, shared by the crates' tests.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub mod test_support {
    pub use exif::{Field, In, Rational, Tag, Value};
    use image::DynamicImage;
    use std::io::Cursor;

    pub fn field(tag: Tag, value: Value) -> Field {
        Field {
            tag,
            ifd_num: In::PRIMARY,
            value,
        }
    }

    pub fn ascii(tag: Tag, text: &str) -> Field {
        field(tag, Value::Ascii(vec![text.as_bytes().to_vec()]))
    }

    pub fn rationals(values: &[(u32, u32)]) -> Value {
        Value::Rational(values.iter().map(|&value| Rational::from(value)).collect())
    }

    /// A bare TIFF-structured EXIF block, as an APP1 segment carries it.
    pub fn exif_block(fields: &[Field], little_endian: bool) -> Vec<u8> {
        let mut writer = exif::experimental::Writer::new();
        for field in fields {
            writer.push_field(field);
        }
        let mut cursor = Cursor::new(Vec::new());
        writer.write(&mut cursor, little_endian).unwrap();
        cursor.into_inner()
    }

    /// `image` encoded as JPEG with an APP1 segment holding `fields`.
    pub fn jpeg_with_exif(image: &DynamicImage, fields: &[Field]) -> Vec<u8> {
        let mut jpeg = Cursor::new(Vec::new());
        image.write_to(&mut jpeg, image::ImageFormat::Jpeg).unwrap();
        let jpeg = jpeg.into_inner();
        let mut payload = b"Exif\0\0".to_vec();
        payload.extend(exif_block(fields, false));
        let mut result = jpeg[..2].to_vec();
        result.extend([0xff, 0xe1]);
        result.extend(((payload.len() + 2) as u16).to_be_bytes());
        result.extend(payload);
        result.extend_from_slice(&jpeg[2..]);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{ascii, exif_block, field, jpeg_with_exif, rationals};
    use super::*;

    #[test]
    fn a_header_claiming_gigapixels_is_refused_not_allocated() {
        // A PNG whose header claims 200000 × 200000 16-bit RGBA: 320 GB.
        fn crc(bytes: &[u8]) -> u32 {
            let mut crc = !0_u32;
            for &byte in bytes {
                crc ^= u32::from(byte);
                for _ in 0..8 {
                    crc = if crc & 1 == 1 {
                        (crc >> 1) ^ 0xedb8_8320
                    } else {
                        crc >> 1
                    };
                }
            }
            !crc
        }
        let mut header = b"IHDR".to_vec();
        header.extend(200_000_u32.to_be_bytes());
        header.extend(200_000_u32.to_be_bytes());
        header.extend([16, 6, 0, 0, 0]);
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend(13_u32.to_be_bytes());
        png.extend(&header);
        png.extend(crc(&header).to_be_bytes());
        // A little image data, an empty zlib stream, so decoding begins.
        let mut data = b"IDAT".to_vec();
        data.extend([0x78, 0x9c, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01]);
        png.extend(8_u32.to_be_bytes());
        png.extend(&data);
        png.extend(crc(&data).to_be_bytes());
        png.extend(0_u32.to_be_bytes());
        png.extend(b"IEND");
        png.extend(crc(b"IEND").to_be_bytes());
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("huge.png");
        std::fs::write(&path, png).unwrap();
        assert!(open_oriented(&path).is_err());
    }

    fn parse(fields: &[exif::Field], little_endian: bool) -> PhotoMetadata {
        from_exif(
            &exif::Reader::new()
                .read_raw(exif_block(fields, little_endian))
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
            assert_eq!(
                metadata.capture_time_source,
                Some(CaptureTimeSource::DateTimeOriginal)
            );
            assert_eq!(metadata.sub_sec_time_original.as_deref(), Some("026"));
        }
        assert_eq!(
            capture_time("2026:10:04 00:00:00", Some("007"), Some("+09:00")).as_deref(),
            Ok("2026-10-03T15:00:00.007Z")
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
        let metadata = parse(
            &[ascii(Tag::DateTimeOriginal, "2026:10:03 19:08:11")],
            false,
        );
        assert!(metadata.capture_time_utc.is_none());
        assert_eq!(metadata.warnings.len(), 1);
    }

    #[test]
    fn gps_time_supplies_utc_when_the_offset_is_missing() {
        let metadata = parse(
            &[
                ascii(Tag::DateTimeOriginal, "2026:10:03 19:08:11"),
                ascii(Tag::GPSDateStamp, "2026:10:04"),
                field(Tag::GPSTimeStamp, rationals(&[(2, 1), (8, 1), (1150, 100)])),
            ],
            true,
        );
        assert_eq!(
            metadata.capture_time_utc.as_deref(),
            Some("2026-10-04T02:08:11.500Z")
        );
        assert_eq!(metadata.capture_time_source, Some(CaptureTimeSource::Gps));
        assert!(metadata.warnings.is_empty(), "{:?}", metadata.warnings);

        // The offset-bearing original time wins over GPS when both exist.
        let both = parse(
            &[
                ascii(Tag::DateTimeOriginal, "2026:10:03 19:08:11"),
                ascii(Tag::OffsetTimeOriginal, "-07:00"),
                ascii(Tag::GPSDateStamp, "2026:10:04"),
                field(Tag::GPSTimeStamp, rationals(&[(2, 1), (8, 1), (9, 1)])),
            ],
            false,
        );
        assert_eq!(
            both.capture_time_source,
            Some(CaptureTimeSource::DateTimeOriginal)
        );

        let bad = parse(
            &[
                ascii(Tag::GPSDateStamp, "2026:10:04"),
                field(Tag::GPSTimeStamp, rationals(&[(25, 1), (0, 1), (0, 1)])),
            ],
            false,
        );
        assert!(bad.capture_time_utc.is_none());
    }

    #[test]
    fn gps_requires_hemispheres_and_valid_dms() {
        let coords = |d, m, s| rationals(&[(d, 1), (m, 1), (s, 1)]);
        for little_endian in [true, false] {
            let metadata = parse(
                &[
                    field(Tag::GPSLatitude, coords(32, 22, 5)),
                    ascii(Tag::GPSLatitudeRef, "S"),
                    field(Tag::GPSLongitude, coords(110, 43, 6)),
                    ascii(Tag::GPSLongitudeRef, "W"),
                    field(Tag::GPSAltitude, rationals(&[(50, 1)])),
                    field(Tag::GPSAltitudeRef, Value::Byte(vec![1])),
                    field(Tag::GPSImgDirection, rationals(&[(350, 1)])),
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

        let null_island = parse(
            &[
                field(Tag::GPSLatitude, coords(0, 0, 0)),
                ascii(Tag::GPSLatitudeRef, "N"),
                field(Tag::GPSLongitude, coords(0, 0, 0)),
                ascii(Tag::GPSLongitudeRef, "E"),
            ],
            false,
        );
        assert!(null_island.gps_latitude_deg.is_none());
        assert_eq!(null_island.warnings.len(), 1);
    }

    #[test]
    fn unknown_exposure_is_not_a_capture_interval() {
        let metadata = parse(&[field(Tag::ExposureTime, rationals(&[(16, 1)]))], false);
        assert_eq!(metadata.exposure_seconds, Some(16.0));
        assert!(metadata.capture_time_utc.is_none());
        let invalid = parse(&[field(Tag::ExposureTime, rationals(&[(1, 0)]))], false);
        assert!(invalid.exposure_seconds.is_none());
    }

    #[test]
    fn absent_or_damaged_exif_does_not_block_pixel_loading() {
        let dir = tempfile::tempdir().unwrap();
        let image = DynamicImage::new_rgb8(32, 24);
        let path = dir.path().join("no-exif.jpg");
        image.save(&path).unwrap();
        let metadata = PhotoMetadata::read(&path);
        assert!(metadata.warnings.is_empty());
        assert!(metadata.capture_time_utc.is_none());
        let plain = std::fs::read(&path).unwrap();
        let payload = b"Exif\0\0broken-tiff";
        let mut damaged = plain[..2].to_vec();
        damaged.extend([0xff, 0xe1]);
        damaged.extend(((payload.len() + 2) as u16).to_be_bytes());
        damaged.extend(payload);
        damaged.extend_from_slice(&plain[2..]);
        std::fs::write(&path, &damaged).unwrap();
        assert!(!PhotoMetadata::read(&path).warnings.is_empty());
        assert_eq!(
            open_oriented(&path)
                .unwrap()
                .coordinates
                .oriented_dimensions,
            (32, 24)
        );
        assert_eq!(
            decode_oriented(&damaged)
                .unwrap()
                .coordinates
                .oriented_dimensions,
            (32, 24)
        );
    }

    #[test]
    fn metadata_is_found_by_content_not_file_name() {
        let dir = tempfile::tempdir().unwrap();
        let jpeg = jpeg_with_exif(
            &DynamicImage::new_rgb8(8, 8),
            &[field(Tag::FocalLengthIn35mmFilm, Value::Short(vec![26]))],
        );
        let path = dir.path().join("renamed.bin");
        std::fs::write(&path, &jpeg).unwrap();
        assert_eq!(PhotoMetadata::read(&path).focal_length_35mm, Some(26.0));
        assert_eq!(
            PhotoMetadata::from_bytes(&jpeg).focal_length_35mm,
            Some(26.0)
        );
        // Bytes that are no image at all are simply metadata-free.
        assert_eq!(
            PhotoMetadata::from_bytes(b"SIMPLE  =                    T"),
            PhotoMetadata::default()
        );
    }

    #[test]
    fn wide_field_prior_scales_with_dimensions() {
        let metadata = PhotoMetadata {
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
        assert!(metadata.scale_hint((0, 3024)).is_none());
    }

    #[test]
    fn the_exif_range_comes_first_and_the_full_range_follows() {
        let phone = PhotoMetadata {
            focal_length_35mm: Some(24.0),
            ..Default::default()
        };
        let dims = (4032, 3024);
        let hint = phone.scale_hint(dims).unwrap();

        let search = ScaleSearch::new(&phone, dims, None, None).unwrap();
        assert!(search.from_exif);
        assert_eq!(
            search.ranges,
            [
                (hint.min_arcsec_per_pixel, hint.max_arcsec_per_pixel),
                (DEFAULT_SCALE_RANGE.0, hint.max_arcsec_per_pixel)
            ]
        );

        // A cropped frame's true scale (here 5"/px, far finer than the
        // hint's 37–148"/px) still lies in the fallback range.
        assert!((search.ranges[1].0..=search.ranges[1].1).contains(&5.0));

        // Both bounds explicit: exactly those, no EXIF and no fallback.
        let explicit = ScaleSearch::new(&phone, dims, Some(1.0), Some(3.0)).unwrap();
        assert_eq!(explicit.ranges, [(1.0, 3.0)]);
        assert!(!explicit.from_exif);

        // An explicit bound that conflicts with the hint drops the hint
        // instead of failing.
        let conflicting = ScaleSearch::new(&phone, dims, None, Some(20.0)).unwrap();
        assert_eq!(conflicting.ranges, [(0.1, 20.0)]);
        assert!(!conflicting.from_exif);

        // A compatible explicit bound keeps the hint; the fallback keeps
        // the explicit bound too.
        let partial = ScaleSearch::new(&phone, dims, None, Some(100.0)).unwrap();
        assert_eq!(
            partial.ranges,
            [
                (hint.min_arcsec_per_pixel, 100.0),
                (DEFAULT_SCALE_RANGE.0, 100.0)
            ]
        );

        // A long lens's narrow hint inside the default range still falls
        // back to the whole default range.
        let tele = PhotoMetadata {
            focal_length_35mm: Some(600.0),
            ..Default::default()
        };
        let tele_hint = tele.scale_hint(dims).unwrap();
        assert_eq!(
            ScaleSearch::new(&tele, dims, None, None).unwrap().ranges,
            [
                (
                    tele_hint.min_arcsec_per_pixel,
                    tele_hint.max_arcsec_per_pixel
                ),
                DEFAULT_SCALE_RANGE
            ]
        );

        // No EXIF: the default range alone.
        let plain = ScaleSearch::new(&PhotoMetadata::default(), dims, None, None).unwrap();
        assert_eq!(plain.ranges, [DEFAULT_SCALE_RANGE]);

        assert!(ScaleSearch::new(&phone, dims, Some(f64::NAN), None).is_err());
        assert!(ScaleSearch::new(&phone, dims, Some(20.0), Some(1.0)).is_err());
    }

    /// Each pixel has a distinct value; compare against independently
    /// specified EXIF coordinate mappings, including both transposes.
    #[test]
    fn all_eight_exif_orientations_match_stored_pixel_coordinates() {
        let source = DynamicImage::ImageRgb8(image::RgbImage::from_fn(3, 2, |x, y| {
            image::Rgb([((x + 3 * y) * 30) as u8, 0, 0])
        }));
        let dir = tempfile::tempdir().unwrap();
        for orientation in 1..=8 {
            let bytes = jpeg_with_exif(
                &source,
                &[field(Tag::Orientation, Value::Short(vec![orientation]))],
            );
            let path = dir.path().join(format!("orientation-{orientation}.jpg"));
            std::fs::write(&path, &bytes).unwrap();
            let raw = image::open(&path).unwrap().to_rgb8();
            for loaded in [
                open_oriented(&path).unwrap(),
                decode_oriented(&bytes).unwrap(),
            ] {
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
}
