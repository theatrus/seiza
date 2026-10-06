use crate::{FitsError, HeaderValue};
use fitsio_pure::header::Card;
use fitsio_pure::image_writer::ImageWriter;
use fitsio_pure::io::AtomicFile;
use fitsio_pure::value::Value;
use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

/// Borrowed linear samples for a primary-HDU 32-bit floating-point image.
#[derive(Clone, Copy, Debug)]
pub enum F32ImageData<'a> {
    Mono(&'a [f32]),
    RgbInterleaved(&'a [f32]),
    RgbPlanar(&'a [f32]),
}

impl F32ImageData<'_> {
    /// Number of color planes this layout produces.
    pub fn planes(self) -> usize {
        match self {
            Self::Mono(_) => 1,
            Self::RgbInterleaved(_) | Self::RgbPlanar(_) => 3,
        }
    }

    /// The borrowed samples regardless of layout.
    pub fn samples(&self) -> &[f32] {
        match self {
            Self::Mono(samples) | Self::RgbInterleaved(samples) | Self::RgbPlanar(samples) => {
                samples
            }
        }
    }
}

/// One non-structural FITS header card supplied to the image writer.
#[derive(Clone, Debug, PartialEq)]
pub struct WriteHeaderCard {
    keyword: String,
    value: HeaderValue,
    comment: String,
}

impl WriteHeaderCard {
    pub fn new(keyword: impl Into<String>, value: HeaderValue) -> Self {
        Self {
            keyword: keyword.into(),
            value,
            comment: String::new(),
        }
    }

    pub fn with_comment(mut self, comment: impl Into<String>) -> Self {
        self.comment = comment.into();
        self
    }

    pub fn keyword(&self) -> &str {
        &self.keyword
    }

    pub fn value(&self) -> &HeaderValue {
        &self.value
    }

    pub fn comment(&self) -> &str {
        &self.comment
    }
}

/// Atomically write a primary-HDU 32-bit floating-point FITS image.
///
/// RGB input may be interleaved or planar in memory; FITS output is always
/// planar. The completed file is flushed and renamed over `path` only after
/// every header and pixel block has been written.
pub fn write_f32_image(
    path: impl AsRef<Path>,
    width: usize,
    height: usize,
    pixels: F32ImageData<'_>,
    headers: &[WriteHeaderCard],
) -> Result<(), FitsError> {
    validate_image(width, height, pixels, headers)?;
    let mut file = AtomicFile::new(path)?;
    write_f32_image_to(&mut file, width, height, pixels, headers)?;
    file.commit()?;
    Ok(())
}

/// Write a primary-HDU 32-bit floating-point FITS image to an existing stream.
///
/// The caller owns flushing and durability. Prefer [`write_f32_image`] for an
/// atomic on-disk file.
pub fn write_f32_image_to(
    writer: impl Write,
    width: usize,
    height: usize,
    pixels: F32ImageData<'_>,
    headers: &[WriteHeaderCard],
) -> Result<(), FitsError> {
    validate_image(width, height, pixels, headers)?;
    let planes = pixels.planes();
    let structural = |keyword: &str, value: &HeaderValue, comment: &str| -> Card {
        card(keyword, value, comment).expect("structural cards are valid")
    };
    let mut cards = vec![
        structural(
            "SIMPLE",
            &HeaderValue::Logical(true),
            "conforms to FITS standard",
        ),
        structural(
            "BITPIX",
            &HeaderValue::Integer(-32),
            "32-bit IEEE floating point",
        ),
        structural(
            "NAXIS",
            &HeaderValue::Integer(if planes == 3 { 3 } else { 2 }),
            "",
        ),
        structural("NAXIS1", &HeaderValue::Integer(width as i64), ""),
        structural("NAXIS2", &HeaderValue::Integer(height as i64), ""),
    ];
    if planes == 3 {
        cards.push(structural("NAXIS3", &HeaderValue::Integer(3), "RGB planes"));
    }
    cards.push(structural(
        "EXTEND",
        &HeaderValue::Logical(true),
        "extensions may be present",
    ));
    for header in headers {
        cards.push(card(header.keyword(), header.value(), header.comment())?);
    }

    let mut image = ImageWriter::new(writer, &cards).map_err(write_error)?;
    match pixels {
        F32ImageData::Mono(samples) | F32ImageData::RgbPlanar(samples) => {
            image.write_samples(samples).map_err(write_error)?;
        }
        F32ImageData::RgbInterleaved(samples) => {
            for channel in 0..3 {
                image
                    .write_iter(samples.iter().skip(channel).step_by(3).copied())
                    .map_err(write_error)?;
            }
        }
    }
    image.finish().map_err(write_error)?;
    Ok(())
}

fn write_error(error: fitsio_pure::Error) -> FitsError {
    match error {
        fitsio_pure::Error::Io(error) => FitsError::Io(error),
        other => FitsError::Malformed(other.to_string()),
    }
}

fn validate_image(
    width: usize,
    height: usize,
    pixels: F32ImageData<'_>,
    headers: &[WriteHeaderCard],
) -> Result<(), FitsError> {
    let expected = width
        .checked_mul(height)
        .and_then(|count| count.checked_mul(pixels.planes()))
        .ok_or_else(|| FitsError::Malformed("image dimensions overflow".into()))?;
    if width == 0 || height == 0 || expected > 2_000_000_000 {
        return Err(FitsError::Malformed("implausible image dimensions".into()));
    }
    if pixels.samples().len() != expected {
        return Err(FitsError::Malformed(format!(
            "pixel buffer has {} samples; expected {expected}",
            pixels.samples().len()
        )));
    }
    expected
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| FitsError::Malformed("image byte count overflows".into()))?;

    let mut keywords = HashSet::with_capacity(headers.len());
    for header in headers {
        validate_keyword(header.keyword())?;
        if is_structural_keyword(header.keyword()) {
            return Err(FitsError::Malformed(format!(
                "{} is managed by the FITS writer",
                header.keyword()
            )));
        }
        if !keywords.insert(header.keyword()) {
            return Err(FitsError::Malformed(format!(
                "duplicate FITS header {}",
                header.keyword()
            )));
        }
        card(header.keyword(), header.value(), header.comment())?;
    }
    Ok(())
}

fn validate_keyword(keyword: &str) -> Result<(), FitsError> {
    if keyword.is_empty()
        || keyword.len() > 8
        || !keyword.is_ascii()
        || !keyword
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || b"_-".contains(&byte))
    {
        return Err(FitsError::Malformed(format!(
            "invalid FITS keyword {keyword:?}"
        )));
    }
    Ok(())
}

fn is_structural_keyword(keyword: &str) -> bool {
    keyword == "NAXIS"
        || keyword.strip_prefix("NAXIS").is_some_and(|suffix| {
            !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
        })
        || matches!(
            keyword,
            "SIMPLE"
                | "BITPIX"
                | "EXTEND"
                | "END"
                | "BSCALE"
                | "BZERO"
                | "PCOUNT"
                | "GCOUNT"
                | "GROUPS"
                | "CHECKSUM"
                | "DATASUM"
        )
}

/// Build a validated single card. fitsio-pure formats it, truncating the
/// comment to fit.
fn card(keyword: &str, value: &HeaderValue, comment: &str) -> Result<Card, FitsError> {
    validate_keyword(keyword)?;
    if !comment.is_ascii() {
        return Err(FitsError::Malformed(format!(
            "FITS comment for {keyword} is not ASCII"
        )));
    }
    let value = match value {
        HeaderValue::Logical(value) => Value::Logical(*value),
        HeaderValue::Integer(value) => Value::Integer(*value),
        HeaderValue::Float(value) if value.is_finite() => Value::Float(*value),
        HeaderValue::Float(_) => {
            return Err(FitsError::Malformed(format!(
                "non-finite FITS header {keyword}"
            )));
        }
        HeaderValue::String(value) if value.is_ascii() => {
            // A value field holds 70 bytes, quotes included.
            if value.len() + value.matches('\'').count() + 2 > 70 {
                return Err(FitsError::Malformed(format!(
                    "FITS header {keyword} does not fit in one card"
                )));
            }
            Value::String(value.clone())
        }
        HeaderValue::String(_) => {
            return Err(FitsError::Malformed(format!(
                "FITS string {keyword} is not ASCII"
            )));
        }
        // An empty raw value is a valid FITS undefined value, not an empty string.
        HeaderValue::Raw(value) if value.trim().is_empty() => Value::Undefined,
        HeaderValue::Raw(value) => fitsio_pure::value::parse_value(value.as_bytes())
            .map(|(value, _)| value)
            .ok_or_else(|| {
                FitsError::Malformed(format!("raw FITS header {keyword} is not a FITS value"))
            })?,
    };
    let mut name = [b' '; 8];
    name[..keyword.len()].copy_from_slice(keyword.as_bytes());
    Ok(Card {
        keyword: name,
        value: Some(value),
        comment: (!comment.is_empty()).then(|| comment.to_string()),
    })
}

/// Update or insert a FITS header keyword in place without modifying or
/// rewriting the underlying image pixels.
///
/// If the keyword already exists in the header, its 80-byte card is overwritten
/// in place. If the keyword does not exist and the header block containing
/// `END` has spare card slots, the new card is inserted before `END`.
///
/// Modifying structural cards (`SIMPLE`, `BITPIX`, `NAXIS*`, `END`, etc.) is
/// rejected with an error to prevent corrupting the file layout.
pub fn update_header_in_place(
    path: &Path,
    keyword: &str,
    value: &HeaderValue,
    comment: Option<&str>,
) -> Result<bool, FitsError> {
    validate_keyword(keyword)?;
    if is_structural_keyword(keyword) {
        return Err(FitsError::Malformed(format!(
            "cannot modify structural FITS card {keyword} in place"
        )));
    }
    let card = card(keyword, value, comment.unwrap_or(""))?;
    fitsio_pure::edit::update_card_in_file(path, 0, &card).map_err(crate::header_error)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FitsImage, Pixels};

    const BLOCK: usize = 2880;
    const CARD: usize = 80;

    #[test]
    fn atomic_mono_writer_round_trips_pixels_and_headers() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("mono.fits");
        std::fs::write(&path, b"old complete file").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
        }
        let headers =
            [WriteHeaderCard::new("EXPTIME", HeaderValue::Float(30.0)).with_comment("seconds")];
        write_f32_image(
            &path,
            2,
            2,
            F32ImageData::Mono(&[-2.5, 0.25, 100.0, f32::NAN]),
            &headers,
        )
        .unwrap();

        let decoded = FitsImage::open(&path).unwrap();
        let Pixels::F32(ref values) = decoded.pixels else {
            panic!("writer must emit BITPIX=-32");
        };
        assert_eq!(values[..3], [-2.5, 0.25, 100.0]);
        assert!(values[3].is_nan());
        assert_eq!(decoded.header_f64("EXPTIME"), Some(30.0));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o640
            );
        }
    }

    #[test]
    fn rgb_layouts_are_written_as_fits_planes() {
        let interleaved = [1.0, 10.0, 100.0, 2.0, 20.0, 200.0];
        let planar = [1.0, 2.0, 10.0, 20.0, 100.0, 200.0];
        for pixels in [
            F32ImageData::RgbInterleaved(&interleaved),
            F32ImageData::RgbPlanar(&planar),
        ] {
            let mut encoded = Vec::new();
            write_f32_image_to(&mut encoded, 2, 1, pixels, &[]).unwrap();
            assert_eq!(encoded.len() % BLOCK, 0);
            let decoded = FitsImage::from_bytes(&encoded).unwrap();
            let Pixels::F32(values) = decoded.pixels else {
                panic!("writer must emit f32 pixels");
            };
            assert_eq!(values, planar);
            assert_eq!(decoded.planes, 3);
        }
    }

    #[test]
    fn writer_preserves_undefined_and_empty_string_values() {
        for raw in ["", "                    "] {
            for comment in ["", "no filter"] {
                let headers = [
                    WriteHeaderCard::new("FILTER", HeaderValue::Raw(raw.into()))
                        .with_comment(comment),
                    WriteHeaderCard::new("EMPTYSTR", HeaderValue::String(String::new())),
                    WriteHeaderCard::new("EXPTIME", HeaderValue::Float(30.0)),
                ];
                let mut encoded = Vec::new();
                write_f32_image_to(
                    &mut encoded,
                    2,
                    1,
                    F32ImageData::Mono(&[1.25, 2.5]),
                    &headers,
                )
                .unwrap();
                let filter_card = encoded[..BLOCK]
                    .chunks_exact(CARD)
                    .find(|card| card.starts_with(b"FILTER  = "))
                    .unwrap();
                assert!(filter_card[10..30].iter().all(|byte| *byte == b' '));
                if comment.is_empty() {
                    assert!(filter_card[30..].iter().all(|byte| *byte == b' '));
                } else {
                    assert!(filter_card[30..].starts_with(b" / no filter"));
                }

                let decoded = FitsImage::from_bytes(&encoded).unwrap();
                assert_eq!(
                    decoded.header("FILTER"),
                    Some(&HeaderValue::Raw(String::new()))
                );
                assert_eq!(
                    decoded.header("EMPTYSTR"),
                    Some(&HeaderValue::String(String::new()))
                );
                assert_eq!(decoded.header_f64("EXPTIME"), Some(30.0));
                assert!(matches!(decoded.pixels, Pixels::F32(values) if values == [1.25, 2.5]));
            }
        }
    }

    #[test]
    fn writer_rejects_invalid_shapes_and_headers() {
        let mut output = Vec::new();
        assert!(write_f32_image_to(&mut output, 2, 2, F32ImageData::Mono(&[1.0; 3]), &[]).is_err());
        let structural = [WriteHeaderCard::new("NAXIS4", HeaderValue::Integer(16))];
        assert!(
            write_f32_image_to(&mut output, 1, 1, F32ImageData::Mono(&[1.0]), &structural).is_err()
        );
        let duplicate = [
            WriteHeaderCard::new("FILTER", HeaderValue::String("R".into())),
            WriteHeaderCard::new("FILTER", HeaderValue::String("G".into())),
        ];
        assert!(
            write_f32_image_to(&mut output, 1, 1, F32ImageData::Mono(&[1.0]), &duplicate).is_err()
        );
        let non_ascii = [WriteHeaderCard::new(
            "FILTER",
            HeaderValue::Raw("\u{00e9}".into()),
        )];
        assert!(
            write_f32_image_to(&mut output, 1, 1, F32ImageData::Mono(&[1.0]), &non_ascii).is_err()
        );
        assert!(output.is_empty());
    }

    #[test]
    fn invalid_output_does_not_replace_an_existing_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("preserved.fits");
        std::fs::write(&path, b"previous complete output").unwrap();
        let invalid = [WriteHeaderCard::new(
            "TOOLONGKEY",
            HeaderValue::Logical(true),
        )];
        assert!(write_f32_image(&path, 1, 1, F32ImageData::Mono(&[1.0]), &invalid,).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"previous complete output");
    }

    #[test]
    fn in_place_header_updates_existing_and_adds_new() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("test_header.fits");
        let initial_headers = [
            WriteHeaderCard::new("OBJECT", HeaderValue::String("OldTarget".into())),
            WriteHeaderCard::new("EXPTIME", HeaderValue::Float(1.0)),
        ];
        write_f32_image(
            &path,
            2,
            2,
            F32ImageData::Mono(&[1.0, 2.0, 3.0, 4.0]),
            &initial_headers,
        )
        .unwrap();

        // 1. Update existing OBJECT header
        let updated = update_header_in_place(
            &path,
            "OBJECT",
            &HeaderValue::String("Kite Cluster".into()),
            Some("NGC 6819"),
        )
        .unwrap();
        assert!(updated);

        // 2. Insert new GAIN header
        let inserted = update_header_in_place(
            &path,
            "GAIN",
            &HeaderValue::Integer(160),
            Some("sensor gain"),
        )
        .unwrap();
        assert!(inserted);

        // 3. Verify via FitsImage
        let image = FitsImage::open(&path).unwrap();
        assert_eq!(
            image.header("OBJECT"),
            Some(&HeaderValue::String("Kite Cluster".into()))
        );
        assert_eq!(image.header("GAIN"), Some(&HeaderValue::Integer(160)));
        assert_eq!(image.header("EXPTIME"), Some(&HeaderValue::Float(1.0)));

        // 4. Verify pixel data remains intact
        let Pixels::F32(pixels) = image.pixels else {
            panic!("expected f32 pixels");
        };
        assert_eq!(pixels, vec![1.0, 2.0, 3.0, 4.0]);

        // 5. Reject modifying structural cards
        assert!(
            update_header_in_place(&path, "SIMPLE", &HeaderValue::Logical(true), None).is_err()
        );
        assert!(update_header_in_place(&path, "BITPIX", &HeaderValue::Integer(16), None).is_err());
        assert!(update_header_in_place(&path, "NAXIS1", &HeaderValue::Integer(100), None).is_err());
        assert!(
            update_header_in_place(&path, "END", &HeaderValue::String("".into()), None).is_err()
        );
    }
}
