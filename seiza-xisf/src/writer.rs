//! Monolithic XISF writer for 32-bit floating-point images.
//!
//! The output mirrors the layout the reader supports: an `XISF0100` preamble,
//! a UTF-8 XML header padded to a 4096-byte block boundary, and one attached
//! uncompressed little-endian `Float32` planar data block. FITS-compatible
//! metadata is stored as `FITSKeyword` elements whose values use FITS text
//! conventions, so files round-trip through this crate's reader and load in
//! PixInsight.

use crate::{MAX_HEADER_BYTES, MAX_SAMPLES, SIGNATURE, XisfError};
use quick_xml::escape::escape;
use seiza_fits::{F32ImageData, HeaderValue, WriteHeaderCard};
use std::collections::HashSet;
use std::fmt::Write as _;
use std::io::{BufWriter, Write};
use std::path::Path;

const BLOCK_ALIGNMENT: usize = 4096;
const PIXEL_CHUNK_BYTES: usize = 1024 * 1024;

/// Atomically write a one-image monolithic XISF file with `Float32` samples.
///
/// RGB input may be interleaved or planar in memory; XISF output is always
/// planar. The completed file is flushed and renamed over `path` only after
/// the header and every data block has been written.
pub fn write_f32_image(
    path: impl AsRef<Path>,
    width: usize,
    height: usize,
    pixels: F32ImageData<'_>,
    headers: &[WriteHeaderCard],
) -> Result<(), XisfError> {
    validate_image(width, height, pixels, headers)?;
    let path = path.as_ref();
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let prefix = format!(
        ".{}.",
        path.file_name().unwrap_or_default().to_string_lossy()
    );
    let mut builder = tempfile::Builder::new();
    builder.prefix(&prefix);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let permissions = std::fs::metadata(path)
            .map(|metadata| metadata.permissions())
            .unwrap_or_else(|_| std::fs::Permissions::from_mode(0o666));
        builder.permissions(permissions);
    }
    let mut temporary = builder.tempfile_in(parent)?;
    let mut writer = BufWriter::new(temporary.as_file_mut());
    write_f32_image_to(&mut writer, width, height, pixels, headers)?;
    writer.flush()?;
    drop(writer);
    temporary.as_file().sync_all()?;
    temporary.persist(path).map_err(|error| error.error)?;
    Ok(())
}

/// Write a one-image monolithic `Float32` XISF file to an existing stream.
///
/// The caller owns flushing and durability. Prefer [`write_f32_image`] for an
/// atomic on-disk file.
pub fn write_f32_image_to(
    mut writer: impl Write,
    width: usize,
    height: usize,
    pixels: F32ImageData<'_>,
    headers: &[WriteHeaderCard],
) -> Result<(), XisfError> {
    validate_image(width, height, pixels, headers)?;
    let data_bytes = std::mem::size_of_val(pixels.samples());
    let bounds = sample_bounds(pixels.samples());

    let mut data_offset = BLOCK_ALIGNMENT;
    let xml = loop {
        let xml = render_xml(
            width,
            height,
            pixels,
            headers,
            bounds,
            data_offset,
            data_bytes,
        );
        let end = PREAMBLE_LEN + xml.len();
        let needed = end.div_ceil(BLOCK_ALIGNMENT) * BLOCK_ALIGNMENT;
        if needed == data_offset {
            break xml;
        }
        data_offset = needed;
    };
    let header_bytes = data_offset - PREAMBLE_LEN;
    if header_bytes > MAX_HEADER_BYTES {
        return Err(XisfError::Malformed(format!(
            "XML header length {header_bytes} is outside the supported range"
        )));
    }

    writer.write_all(SIGNATURE)?;
    writer.write_all(&(header_bytes as u32).to_le_bytes())?;
    writer.write_all(&[0; 4])?;
    writer.write_all(xml.as_bytes())?;
    writer.write_all(&vec![b' '; data_offset - PREAMBLE_LEN - xml.len()])?;

    let mut byte_buffer = Vec::with_capacity(PIXEL_CHUNK_BYTES);
    match pixels {
        F32ImageData::Mono(samples) | F32ImageData::RgbPlanar(samples) => {
            write_float_values(&mut writer, samples.iter().copied(), &mut byte_buffer)?;
        }
        F32ImageData::RgbInterleaved(samples) => {
            let pixel_count = width * height;
            for channel in 0..3 {
                write_float_values(
                    &mut writer,
                    (0..pixel_count).map(|index| samples[index * 3 + channel]),
                    &mut byte_buffer,
                )?;
            }
        }
    }
    Ok(())
}

const PREAMBLE_LEN: usize = 16;

fn render_xml(
    width: usize,
    height: usize,
    pixels: F32ImageData<'_>,
    headers: &[WriteHeaderCard],
    bounds: (f32, f32),
    data_offset: usize,
    data_bytes: usize,
) -> String {
    let planes = pixels.planes();
    let color_space = if planes == 3 { "RGB" } else { "Gray" };
    let mut xml = String::with_capacity(1024);
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
    xml.push_str(
        "<xisf version=\"1.0\" xmlns=\"http://www.pixinsight.com/xisf\" \
         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" \
         xsi:schemaLocation=\"http://www.pixinsight.com/xisf \
         http://pixinsight.com/xisf/xisf-1.0.xsd\">",
    );
    let _ = write!(
        xml,
        "<Image geometry=\"{width}:{height}:{planes}\" sampleFormat=\"Float32\" \
         bounds=\"{}:{}\" colorSpace=\"{color_space}\" pixelStorage=\"Planar\" \
         location=\"attachment:{data_offset}:{data_bytes}\">",
        bounds.0, bounds.1
    );
    for header in headers {
        let _ = write!(
            xml,
            "<FITSKeyword name=\"{}\" value=\"{}\"",
            escape(header.keyword()),
            escape(fits_value_text(header.value()))
        );
        if header.comment().is_empty() {
            xml.push_str("/>");
        } else {
            let _ = write!(xml, " comment=\"{}\"/>", escape(header.comment()));
        }
    }
    xml.push_str("</Image>");
    let _ = write!(
        xml,
        "<Metadata><Property id=\"XISF:CreatorApplication\" type=\"String\">seiza-xisf {}\
         </Property></Metadata>",
        env!("CARGO_PKG_VERSION")
    );
    xml.push_str("</xisf>");
    xml
}

/// Serialize a header value with FITS text conventions so the reader's
/// `parse_header_value` recovers the same variant.
fn fits_value_text(value: &HeaderValue) -> String {
    match value {
        HeaderValue::Logical(true) => "T".into(),
        HeaderValue::Logical(false) => "F".into(),
        HeaderValue::Integer(value) => value.to_string(),
        HeaderValue::Float(value) => format!("{value:.12E}"),
        HeaderValue::String(value) => format!("'{}'", value.replace('\'', "''")),
        HeaderValue::Raw(value) => value.clone(),
    }
}

fn sample_bounds(samples: &[f32]) -> (f32, f32) {
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    for &sample in samples {
        if sample.is_finite() {
            minimum = minimum.min(sample);
            maximum = maximum.max(sample);
        }
    }
    if maximum > minimum {
        (minimum, maximum)
    } else if minimum.is_finite() {
        (minimum, minimum + 1.0)
    } else {
        (0.0, 1.0)
    }
}

fn validate_image(
    width: usize,
    height: usize,
    pixels: F32ImageData<'_>,
    headers: &[WriteHeaderCard],
) -> Result<(), XisfError> {
    let expected = width
        .checked_mul(height)
        .and_then(|count| count.checked_mul(pixels.planes()))
        .ok_or_else(|| XisfError::Malformed("image dimensions overflow".into()))?;
    if width == 0 || height == 0 || expected > MAX_SAMPLES {
        return Err(XisfError::Malformed("implausible image dimensions".into()));
    }
    if pixels.samples().len() != expected {
        return Err(XisfError::Malformed(format!(
            "pixel buffer has {} samples; expected {expected}",
            pixels.samples().len()
        )));
    }
    let mut keywords = HashSet::with_capacity(headers.len());
    for header in headers {
        validate_keyword(header.keyword())?;
        if is_structural_keyword(header.keyword()) {
            return Err(XisfError::Malformed(format!(
                "{} is managed by the XISF writer",
                header.keyword()
            )));
        }
        if !keywords.insert(header.keyword()) {
            return Err(XisfError::Malformed(format!(
                "duplicate FITS header {}",
                header.keyword()
            )));
        }
        if let HeaderValue::Float(value) = header.value()
            && !value.is_finite()
        {
            return Err(XisfError::Malformed(format!(
                "non-finite FITS header {}",
                header.keyword()
            )));
        }
        if let HeaderValue::Raw(value) = header.value()
            && value.is_empty()
        {
            return Err(XisfError::Malformed(format!(
                "empty raw FITS header {}",
                header.keyword()
            )));
        }
    }
    Ok(())
}

fn validate_keyword(keyword: &str) -> Result<(), XisfError> {
    if keyword.is_empty()
        || keyword.len() > 8
        || !keyword
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || b"_-".contains(&byte))
    {
        return Err(XisfError::Malformed(format!(
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

fn write_float_values(
    writer: &mut impl Write,
    values: impl Iterator<Item = f32>,
    buffer: &mut Vec<u8>,
) -> std::io::Result<()> {
    buffer.clear();
    for value in values {
        buffer.extend_from_slice(&value.to_le_bytes());
        if buffer.len() >= PIXEL_CHUNK_BYTES {
            writer.write_all(buffer)?;
            buffer.clear();
        }
    }
    writer.write_all(buffer)?;
    buffer.clear();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use seiza_fits::Pixels;

    #[test]
    fn atomic_mono_writer_round_trips_pixels_and_headers() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("mono.xisf");
        std::fs::write(&path, b"old complete file").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
        }
        let headers = [
            WriteHeaderCard::new("EXPTIME", HeaderValue::Float(30.0)).with_comment("seconds"),
            WriteHeaderCard::new("OBJECT", HeaderValue::String("M 31 & 'friends' <3".into())),
            WriteHeaderCard::new("GAINSET", HeaderValue::Logical(true)),
            WriteHeaderCard::new("OFFSET", HeaderValue::Integer(-30)),
        ];
        write_f32_image(
            &path,
            2,
            2,
            F32ImageData::Mono(&[-2.5, 0.25, 100.0, f32::NAN]),
            &headers,
        )
        .unwrap();

        let decoded = crate::open(&path).unwrap();
        let Pixels::F32(ref values) = decoded.pixels else {
            panic!("writer must emit Float32 samples");
        };
        assert_eq!(values[..3], [-2.5, 0.25, 100.0]);
        assert!(values[3].is_nan());
        assert_eq!(decoded.header_f64("EXPTIME"), Some(30.0));
        assert_eq!(decoded.header_str("OBJECT"), Some("M 31 & 'friends' <3"));
        assert_eq!(decoded.header("GAINSET"), Some(&HeaderValue::Logical(true)));
        assert_eq!(decoded.header("OFFSET"), Some(&HeaderValue::Integer(-30)));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o640
            );
        }

        let info = crate::inspect(&path).unwrap();
        assert_eq!(info.images.len(), 1);
        let image = &info.images[0];
        assert_eq!((image.width, image.height, image.planes), (2, 2, 1));
        assert_eq!(image.sample_format, crate::SampleFormat::Float32);
        assert!(image.compression.is_none());
        assert_eq!(image.attachment_offset % BLOCK_ALIGNMENT as u64, 0);
    }

    #[test]
    fn rgb_layouts_are_written_as_planar_planes() {
        let interleaved = [1.0, 10.0, 100.0, 2.0, 20.0, 200.0];
        let planar = [1.0, 2.0, 10.0, 20.0, 100.0, 200.0];
        for pixels in [
            F32ImageData::RgbInterleaved(&interleaved),
            F32ImageData::RgbPlanar(&planar),
        ] {
            let mut encoded = Vec::new();
            write_f32_image_to(&mut encoded, 2, 1, pixels, &[]).unwrap();
            let decoded = crate::from_bytes(&encoded).unwrap();
            let Pixels::F32(values) = decoded.pixels else {
                panic!("writer must emit f32 pixels");
            };
            assert_eq!(values, planar);
            assert_eq!(decoded.planes, 3);
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
        let non_finite = [WriteHeaderCard::new(
            "AIRMASS",
            HeaderValue::Float(f64::NAN),
        )];
        assert!(
            write_f32_image_to(&mut output, 1, 1, F32ImageData::Mono(&[1.0]), &non_finite).is_err()
        );
    }

    #[test]
    fn invalid_output_does_not_replace_an_existing_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("preserved.xisf");
        std::fs::write(&path, b"previous complete output").unwrap();
        let invalid = [WriteHeaderCard::new(
            "TOOLONGKEY",
            HeaderValue::Logical(true),
        )];
        assert!(write_f32_image(&path, 1, 1, F32ImageData::Mono(&[1.0]), &invalid).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"previous complete output");
    }

    #[test]
    fn header_growth_keeps_the_attachment_block_aligned() {
        // Enough keywords to push the XML header past one alignment block.
        let headers = (0..200)
            .map(|index| {
                WriteHeaderCard::new(
                    format!("KEY{index:05}"),
                    HeaderValue::String(format!("value number {index}")),
                )
                .with_comment("a reasonably long comment to grow the header")
            })
            .collect::<Vec<_>>();
        let mut encoded = Vec::new();
        write_f32_image_to(&mut encoded, 3, 2, F32ImageData::Mono(&[0.5; 6]), &headers).unwrap();

        let mut cursor = std::io::Cursor::new(encoded.as_slice());
        let parsed = crate::parse_file(&mut cursor, encoded.len() as u64).unwrap();
        let offset = parsed.images[0].info.attachment_offset;
        assert!(offset > BLOCK_ALIGNMENT as u64);
        assert_eq!(offset % BLOCK_ALIGNMENT as u64, 0);
        let decoded = crate::from_bytes(&encoded).unwrap();
        assert_eq!(decoded.header_str("KEY00199"), Some("value number 199"));
        let Pixels::F32(values) = decoded.pixels else {
            panic!("writer must emit f32 pixels");
        };
        assert_eq!(values, [0.5; 6]);
    }
}
