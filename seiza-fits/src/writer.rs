use crate::{FitsError, HeaderValue};
use std::collections::HashSet;
use std::io::{BufWriter, Write};
use std::path::Path;

const BLOCK: usize = 2880;
const CARD: usize = 80;
const PIXEL_CHUNK_BYTES: usize = 1024 * 1024;

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

/// Write a primary-HDU 32-bit floating-point FITS image to an existing stream.
///
/// The caller owns flushing and durability. Prefer [`write_f32_image`] for an
/// atomic on-disk file.
pub fn write_f32_image_to(
    mut writer: impl Write,
    width: usize,
    height: usize,
    pixels: F32ImageData<'_>,
    headers: &[WriteHeaderCard],
) -> Result<(), FitsError> {
    validate_image(width, height, pixels, headers)?;
    let planes = pixels.planes();
    let mut cards = vec![
        encode_card(
            "SIMPLE",
            &HeaderValue::Logical(true),
            "conforms to FITS standard",
        )?,
        encode_card(
            "BITPIX",
            &HeaderValue::Integer(-32),
            "32-bit IEEE floating point",
        )?,
        encode_card(
            "NAXIS",
            &HeaderValue::Integer(if planes == 3 { 3 } else { 2 }),
            "",
        )?,
        encode_card("NAXIS1", &HeaderValue::Integer(width as i64), "")?,
        encode_card("NAXIS2", &HeaderValue::Integer(height as i64), "")?,
    ];
    if planes == 3 {
        cards.push(encode_card(
            "NAXIS3",
            &HeaderValue::Integer(3),
            "RGB planes",
        )?);
    }
    cards.push(encode_card(
        "EXTEND",
        &HeaderValue::Logical(true),
        "extensions may be present",
    )?);
    for header in headers {
        cards.push(encode_card(
            header.keyword(),
            header.value(),
            header.comment(),
        )?);
    }
    cards.push(format!("{:<CARD$}", "END"));
    write_block_padded(&mut writer, cards.concat().as_bytes(), b' ')?;

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
    let byte_len = std::mem::size_of_val(pixels.samples());
    let padding = (BLOCK - byte_len % BLOCK) % BLOCK;
    writer.write_all(&vec![0; padding])?;
    Ok(())
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
        encode_card(header.keyword(), header.value(), header.comment())?;
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

fn encode_card(keyword: &str, value: &HeaderValue, comment: &str) -> Result<String, FitsError> {
    validate_keyword(keyword)?;
    if !comment.is_ascii() {
        return Err(FitsError::Malformed(format!(
            "FITS comment for {keyword} is not ASCII"
        )));
    }
    let value = match value {
        HeaderValue::Logical(value) => {
            if *value {
                "T".into()
            } else {
                "F".into()
            }
        }
        HeaderValue::Integer(value) => value.to_string(),
        HeaderValue::Float(value) if value.is_finite() => format!("{value:.12E}"),
        HeaderValue::Float(_) => {
            return Err(FitsError::Malformed(format!(
                "non-finite FITS header {keyword}"
            )));
        }
        HeaderValue::String(value) if value.is_ascii() => {
            format!("'{}'", value.replace('\'', "''"))
        }
        HeaderValue::String(_) => {
            return Err(FitsError::Malformed(format!(
                "FITS string {keyword} is not ASCII"
            )));
        }
        HeaderValue::Raw(value) if !value.is_empty() && value.is_ascii() => value.clone(),
        HeaderValue::Raw(_) => {
            return Err(FitsError::Malformed(format!(
                "empty or non-ASCII raw FITS header {keyword}"
            )));
        }
    };
    let base = format!("{keyword:<8}= {value:>20}");
    if base.len() > CARD {
        return Err(FitsError::Malformed(format!(
            "FITS header {keyword} does not fit in one card"
        )));
    }
    let mut text = base;
    if !comment.is_empty() && text.len() + 3 < CARD {
        text.push_str(" / ");
        let remaining = CARD - text.len();
        text.push_str(&comment[..comment.len().min(remaining)]);
    }
    Ok(format!("{text:<CARD$}"))
}

fn write_float_values(
    writer: &mut impl Write,
    values: impl Iterator<Item = f32>,
    buffer: &mut Vec<u8>,
) -> std::io::Result<()> {
    buffer.clear();
    for value in values {
        buffer.extend_from_slice(&value.to_be_bytes());
        if buffer.len() >= PIXEL_CHUNK_BYTES {
            writer.write_all(buffer)?;
            buffer.clear();
        }
    }
    writer.write_all(buffer)?;
    buffer.clear();
    Ok(())
}

fn write_block_padded(writer: &mut impl Write, bytes: &[u8], padding: u8) -> std::io::Result<()> {
    writer.write_all(bytes)?;
    let count = (BLOCK - bytes.len() % BLOCK) % BLOCK;
    writer.write_all(&vec![padding; count])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FitsImage, Pixels};

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
}
