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

/// Whether a keyword is written with the ESO `HIERARCH` convention: one too
/// long for the eight-column keyword field, or holding a space.
fn is_long_keyword(keyword: &str) -> bool {
    keyword.len() > 8 || keyword.contains(' ')
}

fn validate_keyword(keyword: &str) -> Result<(), FitsError> {
    let valid = if is_long_keyword(keyword) {
        crate::is_hierarch_keyword(keyword)
    } else {
        !keyword.is_empty()
            && keyword.bytes().all(|byte| {
                byte.is_ascii_uppercase() || byte.is_ascii_digit() || b"_-".contains(&byte)
            })
    };
    if !valid {
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
                | "XTENSION"
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

/// Encode one header card, or, for a string too long for one card, a run of
/// cards under the long-string convention. Keywords that do not fit the
/// standard eight columns get a `HIERARCH` card.
fn encode_card(keyword: &str, value: &HeaderValue, comment: &str) -> Result<String, FitsError> {
    validate_keyword(keyword)?;
    if !comment.is_ascii() {
        return Err(FitsError::Malformed(format!(
            "FITS comment for {keyword} is not ASCII"
        )));
    }
    let string = match value {
        HeaderValue::String(string) => Some(string),
        _ => None,
    };
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
        // An empty raw value is a valid FITS undefined value, not an empty string.
        HeaderValue::Raw(value) if value.is_ascii() => value.clone(),
        HeaderValue::Raw(_) => {
            return Err(FitsError::Malformed(format!(
                "non-ASCII raw FITS header {keyword}"
            )));
        }
    };
    let prefix = if is_long_keyword(keyword) {
        format!("HIERARCH {keyword} = ")
    } else {
        format!("{keyword:<8}= ")
    };
    let mut text = format!("{prefix}{value:>20}");
    if text.len() > CARD && prefix.len() + value.len() <= CARD {
        // A long HIERARCH keyword leaves no room to right-align the value.
        text = format!("{prefix}{value}");
    }
    if text.len() > CARD {
        return match string {
            Some(string) => encode_long_string(keyword, &prefix, string, comment),
            None => Err(FitsError::Malformed(format!(
                "FITS header {keyword} does not fit in one card"
            ))),
        };
    }
    Ok(with_comment(text, comment))
}

/// Pad a card's text to 80 columns, with as much of `comment` as fits.
fn with_comment(mut text: String, comment: &str) -> String {
    if !comment.is_empty() && text.len() + 3 < CARD {
        text.push_str(" / ");
        let remaining = CARD - text.len();
        text.push_str(&comment[..comment.len().min(remaining)]);
    }
    format!("{text:<CARD$}")
}

/// Split a string value across `CONTINUE` cards, the long-string convention
/// the reader joins: every part but the last ends in `&`. A doubled quote
/// stays within one part, and the comment goes on the last card.
fn encode_long_string(
    keyword: &str,
    prefix: &str,
    value: &str,
    comment: &str,
) -> Result<String, FitsError> {
    let mut cards = String::new();
    let mut lead = prefix.to_string();
    let mut part = String::new();
    for character in value.chars() {
        let length = if character == '\'' { 2 } else { 1 };
        // Room for the opening quote, the `&` and the closing quote.
        if lead.len() + part.len() + length + 3 > CARD {
            if part.is_empty() {
                return Err(FitsError::Malformed(format!(
                    "FITS header {keyword} does not fit in one card"
                )));
            }
            cards.push_str(&format!("{:<CARD$}", format!("{lead}'{part}&'")));
            lead = "CONTINUE  ".into();
            part.clear();
        }
        part.push(character);
        if character == '\'' {
            part.push('\'');
        }
    }
    cards.push_str(&with_comment(format!("{lead}'{part}'"), comment));
    Ok(cards)
}

/// Update or insert a FITS header keyword in place without modifying or
/// rewriting the underlying image pixels.
///
/// If the keyword already exists in the header, its 80-byte card is overwritten
/// in place. If the keyword does not exist and the header block containing
/// `END` has spare card slots, the new card is inserted before `END`. A
/// keyword too long for the standard eight columns, or holding a space, is
/// written as a `HIERARCH` card.
///
/// The header changed is that of the HDU [`crate::FitsImage::open`] reads:
/// the primary, or, when the primary holds no data, the image extension.
/// Reading the file again then finds the new value first. For a
/// tile-compressed image that is its table's header, where the table and
/// compression keywords are refused too.
///
/// Modifying structural cards (`SIMPLE`, `BITPIX`, `NAXIS*`, `END`, etc.) is
/// rejected with an error to prevent corrupting the file layout, and so is a
/// value too long for one card.
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
    let encoded_card = encode_card(keyword, value, comment.unwrap_or(""))?;
    if encoded_card.len() != CARD {
        return Err(FitsError::Malformed(format!(
            "FITS header {keyword} does not fit in one card"
        )));
    }

    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(FitsError::Io)?;
    let (header_start, compressed) = crate::image_header_offset(&mut file)?;
    if compressed && crate::compressed::is_reserved_keyword(keyword) {
        return Err(FitsError::Malformed(format!(
            "cannot modify {keyword} of a tile-compressed image in place"
        )));
    }

    let mut block = [0u8; BLOCK];
    let mut block_idx: u64 = 0;
    let mut end_card_location: Option<(u64, usize)> = None;

    loop {
        use std::io::{Read, Seek, SeekFrom, Write};
        let start_pos = header_start + block_idx * BLOCK as u64;
        file.seek(SeekFrom::Start(start_pos))
            .map_err(FitsError::Io)?;

        if let Err(error) = file.read_exact(&mut block) {
            if error.kind() == std::io::ErrorKind::UnexpectedEof {
                return Err(FitsError::Malformed("missing END card".into()));
            }
            return Err(FitsError::Io(error));
        }

        for (card_idx, card) in block.chunks_exact(CARD).enumerate() {
            if card_keyword(card).is_some_and(|found| found.eq_ignore_ascii_case(keyword)) {
                let target_offset = start_pos + (card_idx * CARD) as u64;
                file.seek(SeekFrom::Start(target_offset))
                    .map_err(FitsError::Io)?;
                file.write_all(encoded_card.as_bytes())
                    .map_err(FitsError::Io)?;
                file.flush().map_err(FitsError::Io)?;
                return Ok(true);
            }

            if card.starts_with(b"END") && (card.len() <= 3 || card[3] == b' ') {
                end_card_location = Some((block_idx, card_idx));
                break;
            }
        }

        if end_card_location.is_some() {
            break;
        }

        block_idx += 1;
    }

    let Some((end_block, end_card_idx)) = end_card_location else {
        return Err(FitsError::Malformed("missing END card".into()));
    };

    if end_card_idx + 1 < BLOCK / CARD {
        use std::io::{Seek, SeekFrom, Write};
        let insert_offset = header_start + end_block * BLOCK as u64 + (end_card_idx * CARD) as u64;
        let new_end_offset = insert_offset + CARD as u64;

        file.seek(SeekFrom::Start(insert_offset))
            .map_err(FitsError::Io)?;
        file.write_all(encoded_card.as_bytes())
            .map_err(FitsError::Io)?;

        let end_card_str = format!("{:<80}", "END");
        file.seek(SeekFrom::Start(new_end_offset))
            .map_err(FitsError::Io)?;
        file.write_all(end_card_str.as_bytes())
            .map_err(FitsError::Io)?;
        file.flush().map_err(FitsError::Io)?;
        Ok(true)
    } else {
        Err(FitsError::Malformed(
            "header block is full; inserting new keyword requires allocating new header block"
                .into(),
        ))
    }
}

/// The keyword of a valued card as the reader names it, `HIERARCH` cards
/// included.
fn card_keyword(card: &[u8]) -> Option<&str> {
    let card = std::str::from_utf8(card).ok()?;
    if card.as_bytes()[8] == b'=' {
        return Some(card[..8].trim_end());
    }
    let (keyword, _) = card.strip_prefix("HIERARCH ")?.split_once('=')?;
    Some(keyword.trim())
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
        let invalid = [WriteHeaderCard::new("BAD=KEY", HeaderValue::Logical(true))];
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

    fn header_cards(encoded: &[u8]) -> Vec<String> {
        let end = encoded
            .chunks_exact(CARD)
            .position(|card| card.starts_with(b"END "))
            .unwrap();
        encoded[..end * CARD]
            .chunks_exact(CARD)
            .map(|card| String::from_utf8(card.to_vec()).unwrap())
            .collect()
    }

    #[test]
    fn long_keywords_round_trip_as_hierarch_cards() {
        let long_text = format!("{}'quoted'{}", "a".repeat(60), "b".repeat(70));
        let headers = [
            WriteHeaderCard::new("LBTO LUCI DET ITIME", HeaderValue::Float(0.139764))
                .with_comment("[s] integration time"),
            WriteHeaderCard::new("FOCALLENGTH", HeaderValue::Integer(530)),
            WriteHeaderCard::new("ESO OBS NAME", HeaderValue::String(long_text.clone()))
                .with_comment("continued"),
            WriteHeaderCard::new("OBJECT", HeaderValue::String("short".into())),
            WriteHeaderCard::new(
                format!("ESO {}", "K".repeat(56)),
                HeaderValue::String("v".into()),
            ),
        ];
        let mut encoded = Vec::new();
        write_f32_image_to(&mut encoded, 1, 1, F32ImageData::Mono(&[1.0]), &headers).unwrap();
        let cards = header_cards(&encoded);
        assert!(cards.iter().any(|card| {
            card.starts_with("HIERARCH LBTO LUCI DET ITIME = ")
                && card.trim_end().ends_with("E-1 / [s] integration time")
        }));
        assert!(
            cards
                .iter()
                .any(|card| card.starts_with("HIERARCH FOCALLENGTH = "))
        );
        assert!(cards.iter().any(|card| card.starts_with("CONTINUE  '")));
        assert!(cards.iter().any(|card| card.starts_with("OBJECT  = ")));

        let decoded = FitsImage::from_bytes(&encoded).unwrap();
        for header in &headers {
            assert_eq!(decoded.header(header.keyword()), Some(header.value()));
        }
        assert_eq!(decoded.header_str("ESO OBS NAME"), Some(long_text.as_str()));
    }

    #[test]
    fn long_strings_split_without_breaking_quotes() {
        // A quote at every position crosses each card boundary once.
        for length in [68, 69, 70, 135, 136, 137, 300] {
            for quote_at in [0, 64, 65, 66, 67, 68, 131, 132, 133] {
                let mut text: String = (0..length)
                    .map(|index| char::from(b'A' + (index % 26) as u8))
                    .collect();
                if quote_at < length {
                    text.replace_range(quote_at..quote_at + 1, "'");
                }
                let encoded =
                    encode_card("LONGSTR", &HeaderValue::String(text.clone()), "note").unwrap();
                assert_eq!(encoded.len() % CARD, 0);
                let mut file = [
                    "SIMPLE  =                    T",
                    "BITPIX  =                    8",
                    "NAXIS   =                    2",
                    "NAXIS1  =                    1",
                    "NAXIS2  =                    1",
                ]
                .map(|text| format!("{text:<80}"))
                .concat();
                file.push_str(&encoded);
                file.push_str(&format!("{:<80}", "END"));
                let mut file = file.into_bytes();
                file.resize(file.len().next_multiple_of(BLOCK) + BLOCK, b' ');
                let header = FitsImage::from_bytes(&file).unwrap();
                assert_eq!(
                    header.header_str("LONGSTR"),
                    Some(text.as_str()),
                    "{length} {quote_at}"
                );
            }
        }
    }

    #[test]
    fn keywords_that_cannot_round_trip_are_rejected() {
        for keyword in [
            "lower",
            "BAD=KEY",
            "QUOTE'S KEY",
            "SLASH/KEY LONG",
            " LEADING SPACE",
            "TRAILING SPACE ",
            "NON\u{00e9}ASCII KEY",
            "TAB\tKEY LONG",
        ] {
            let headers = [WriteHeaderCard::new(keyword, HeaderValue::Integer(1))];
            let mut output = Vec::new();
            assert!(
                write_f32_image_to(&mut output, 1, 1, F32ImageData::Mono(&[1.0]), &headers)
                    .is_err(),
                "{keyword:?}"
            );
        }
        // A keyword that leaves no room for its value.
        let crowded = "K".repeat(67);
        assert!(encode_card(&crowded, &HeaderValue::Integer(12345), "").is_err());
        assert!(encode_card(&crowded, &HeaderValue::String("x".repeat(10)), "").is_err());
        assert!(encode_card(&crowded, &HeaderValue::Integer(1), "").is_ok());
    }

    #[test]
    fn in_place_updates_handle_hierarch_cards() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("hierarch.fits");
        let headers = [WriteHeaderCard::new("ESO DET DIT", HeaderValue::Float(1.0))];
        write_f32_image(&path, 1, 1, F32ImageData::Mono(&[1.0]), &headers).unwrap();
        let before = std::fs::metadata(&path).unwrap().len();

        update_header_in_place(&path, "ESO DET DIT", &HeaderValue::Float(2.5), None).unwrap();
        update_header_in_place(&path, "ESO DET NDIT", &HeaderValue::Integer(4), None).unwrap();
        let image = FitsImage::open(&path).unwrap();
        assert_eq!(image.header("ESO DET DIT"), Some(&HeaderValue::Float(2.5)));
        assert_eq!(image.header("ESO DET NDIT"), Some(&HeaderValue::Integer(4)));
        assert_eq!(
            image
                .headers
                .iter()
                .filter(|(key, _)| key == "ESO DET DIT")
                .count(),
            1
        );
        assert_eq!(std::fs::metadata(&path).unwrap().len(), before);

        // A value that needs CONTINUE cards cannot be written in place.
        let long = HeaderValue::String("x".repeat(100));
        assert!(update_header_in_place(&path, "OBJECT", &long, None).is_err());
        assert!(update_header_in_place(&path, "BAD=KEY", &HeaderValue::Integer(1), None).is_err());
    }
}
