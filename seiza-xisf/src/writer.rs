//! Monolithic XISF writer for 32-bit floating-point images.
//!
//! The output mirrors the layout the reader supports: an `XISF0100` preamble,
//! a UTF-8 XML header padded to a 4096-byte block boundary, and one attached
//! little-endian `Float32` planar data block, optionally compressed and
//! checksummed. FITS-compatible metadata is stored as `FITSKeyword` elements
//! whose values use FITS text conventions, so files round-trip through this
//! crate's reader and load in PixInsight.

use crate::{CompressionCodec, MAX_HEADER_BYTES, MAX_SAMPLES, SIGNATURE, XisfError};
use quick_xml::escape::escape;
use seiza_fits::{F32ImageData, HeaderValue, WriteHeaderCard};
use std::collections::HashSet;
use std::fmt::Write as _;
use std::io::{BufWriter, Write};
use std::path::Path;

const BLOCK_ALIGNMENT: usize = 4096;
const PIXEL_CHUNK_BYTES: usize = 1024 * 1024;
/// The largest block compressed as one unit. Larger blocks are split into
/// subblocks, which keeps every codec within its input limit.
const SUBBLOCK_BYTES: usize = 1 << 30;

/// Choices for [`write_f32_image_with_options`].
///
/// The default writes an uncompressed block without a checksum, which every
/// XISF reader can open.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WriteOptions {
    pub compression: Option<WriteCompression>,
    /// Hash the stored pixel block so readers can detect corruption.
    pub checksum: Option<ChecksumAlgorithm>,
}

/// How to compress the pixel block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WriteCompression {
    /// [`CompressionCodec::Lz4Hc`] is not available for writing.
    pub codec: CompressionCodec,
    /// Group the bytes of equal significance before compressing, which
    /// usually compresses floating-point pixels much better.
    pub byte_shuffle: bool,
    /// The codec's own level: 0 to 9 for zlib, 1 to 22 for zstd, and none
    /// for LZ4. `None` picks the codec's default.
    pub level: Option<i32>,
}

impl WriteCompression {
    /// Zstandard with byte shuffling at its default level, the codec the
    /// XISF specification recommends for pixel data.
    ///
    /// Readers built on the original 2017 specification cannot decode
    /// Zstandard; use zlib or LZ4 when they must open the file.
    pub fn recommended() -> Self {
        Self {
            codec: CompressionCodec::Zstd,
            byte_shuffle: true,
            level: None,
        }
    }
}

/// A hashing algorithm for data block checksums.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChecksumAlgorithm {
    Sha1,
    Sha256,
    Sha512,
    Sha3_256,
    Sha3_512,
}

impl ChecksumAlgorithm {
    fn name(self) -> &'static str {
        match self {
            Self::Sha1 => "sha-1",
            Self::Sha256 => "sha-256",
            Self::Sha512 => "sha-512",
            Self::Sha3_256 => "sha3-256",
            Self::Sha3_512 => "sha3-512",
        }
    }

    fn digest(self, bytes: &[u8]) -> String {
        fn hex<D: sha2::Digest>(bytes: &[u8]) -> String {
            crate::lowercase_hex(D::digest(bytes).as_ref())
        }
        match self {
            Self::Sha1 => hex::<sha1::Sha1>(bytes),
            Self::Sha256 => hex::<sha2::Sha256>(bytes),
            Self::Sha512 => hex::<sha2::Sha512>(bytes),
            Self::Sha3_256 => hex::<sha3::Sha3_256>(bytes),
            Self::Sha3_512 => hex::<sha3::Sha3_512>(bytes),
        }
    }
}

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
    write_f32_image_with_options(
        path,
        width,
        height,
        pixels,
        headers,
        &WriteOptions::default(),
    )
}

/// [`write_f32_image`] with compression and checksum choices.
pub fn write_f32_image_with_options(
    path: impl AsRef<Path>,
    width: usize,
    height: usize,
    pixels: F32ImageData<'_>,
    headers: &[WriteHeaderCard],
    options: &WriteOptions,
) -> Result<(), XisfError> {
    validate_image(width, height, pixels, headers)?;
    validate_options(options)?;
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
    write_f32_image_to_with_options(&mut writer, width, height, pixels, headers, options)?;
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
    writer: impl Write,
    width: usize,
    height: usize,
    pixels: F32ImageData<'_>,
    headers: &[WriteHeaderCard],
) -> Result<(), XisfError> {
    write_f32_image_to_with_options(
        writer,
        width,
        height,
        pixels,
        headers,
        &WriteOptions::default(),
    )
}

/// [`write_f32_image_to`] with compression and checksum choices.
pub fn write_f32_image_to_with_options(
    writer: impl Write,
    width: usize,
    height: usize,
    pixels: F32ImageData<'_>,
    headers: &[WriteHeaderCard],
    options: &WriteOptions,
) -> Result<(), XisfError> {
    write_image(
        writer,
        (width, height, pixels),
        headers,
        options,
        SUBBLOCK_BYTES,
    )
}

fn write_image(
    mut writer: impl Write,
    (width, height, pixels): (usize, usize, F32ImageData<'_>),
    headers: &[WriteHeaderCard],
    options: &WriteOptions,
    subblock_bytes: usize,
) -> Result<(), XisfError> {
    validate_image(width, height, pixels, headers)?;
    validate_options(options)?;
    let block = prepare_block(width, height, pixels, options, subblock_bytes);
    let bounds = sample_bounds(pixels.samples());

    let created = utc_timestamp(std::time::SystemTime::now());
    let mut data_offset = BLOCK_ALIGNMENT;
    let xml = loop {
        let xml = render_xml(
            width,
            height,
            pixels,
            headers,
            bounds,
            (data_offset, &block),
            &created,
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

    if let Some(stored) = &block.stored {
        writer.write_all(stored)?;
        return Ok(());
    }
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

/// The pixel block as it will be stored, with the `Image` attributes and
/// `Metadata` properties that describe it.
struct PreparedBlock {
    /// The stored bytes, or `None` to stream the samples uncompressed.
    stored: Option<Vec<u8>>,
    bytes: usize,
    attributes: String,
    metadata: Vec<(&'static str, &'static str, String)>,
}

fn validate_options(options: &WriteOptions) -> Result<(), XisfError> {
    let Some(compression) = options.compression else {
        return Ok(());
    };
    let level_range = match compression.codec {
        CompressionCodec::Zlib => Some(0..=9),
        CompressionCodec::Zstd => Some(1..=22),
        CompressionCodec::Lz4 => None,
        CompressionCodec::Lz4Hc => {
            return Err(XisfError::Unsupported(
                "LZ4HC compression is not available for writing".into(),
            ));
        }
    };
    match (compression.level, level_range) {
        (None, _) => Ok(()),
        (Some(level), Some(range)) if range.contains(&level) => Ok(()),
        (Some(level), _) => Err(XisfError::Malformed(format!(
            "compression level {level} is outside the range of {:?}",
            compression.codec
        ))),
    }
}

fn prepare_block(
    width: usize,
    height: usize,
    pixels: F32ImageData<'_>,
    options: &WriteOptions,
    subblock_bytes: usize,
) -> PreparedBlock {
    let raw_bytes = std::mem::size_of_val(pixels.samples());
    if options.compression.is_none() && options.checksum.is_none() {
        return PreparedBlock {
            stored: None,
            bytes: raw_bytes,
            attributes: String::new(),
            metadata: Vec::new(),
        };
    }
    let mut raw = Vec::with_capacity(raw_bytes);
    match pixels {
        F32ImageData::Mono(samples) | F32ImageData::RgbPlanar(samples) => {
            raw.extend(samples.iter().flat_map(|value| value.to_le_bytes()));
        }
        F32ImageData::RgbInterleaved(samples) => {
            for channel in 0..3 {
                raw.extend(
                    (0..width * height)
                        .flat_map(|index| samples[index * 3 + channel].to_le_bytes()),
                );
            }
        }
    }
    let mut block = PreparedBlock {
        stored: None,
        bytes: raw.len(),
        attributes: String::new(),
        metadata: Vec::new(),
    };
    let mut stored = raw;
    if let Some(compression) = options.compression {
        let item_bytes = std::mem::size_of::<f32>();
        let input = if compression.byte_shuffle {
            shuffle(&stored, item_bytes)
        } else {
            stored.clone()
        };
        let subblocks = input
            .chunks(subblock_bytes)
            .map(|chunk| (compress(chunk, compression), chunk.len()))
            .collect::<Vec<_>>();
        let compressed_bytes = subblocks.iter().map(|(data, _)| data.len()).sum::<usize>();
        // A block that does not shrink is stored as it is.
        if compressed_bytes < stored.len() {
            let name = codec_name(compression.codec);
            let _ = write!(block.attributes, " compression=\"{name}");
            if compression.byte_shuffle {
                let _ = write!(block.attributes, "+sh:{}:{item_bytes}\"", stored.len());
            } else {
                let _ = write!(block.attributes, ":{}\"", stored.len());
            }
            if subblocks.len() > 1 {
                let list = subblocks
                    .iter()
                    .map(|(data, uncompressed)| format!("{},{uncompressed}", data.len()))
                    .collect::<Vec<_>>()
                    .join(":");
                let _ = write!(block.attributes, " subblocks=\"{list}\"");
            }
            let shuffle_suffix = if compression.byte_shuffle { "+sh" } else { "" };
            block.metadata.push((
                "XISF:CompressionCodecs",
                "String",
                format!("{name}{shuffle_suffix}"),
            ));
            if let Some(level) = abstract_compression_level(compression) {
                block
                    .metadata
                    .push(("XISF:CompressionLevel", "Int32", level.to_string()));
            }
            stored = subblocks.into_iter().flat_map(|(data, _)| data).collect();
        }
    }
    if let Some(checksum) = options.checksum {
        let _ = write!(
            block.attributes,
            " checksum=\"{}:{}\"",
            checksum.name(),
            checksum.digest(&stored)
        );
        block
            .metadata
            .push(("XISF:ChecksumAlgorithms", "String", checksum.name().into()));
    }
    block.bytes = stored.len();
    block.stored = Some(stored);
    block
}

fn codec_name(codec: CompressionCodec) -> &'static str {
    match codec {
        CompressionCodec::Zlib => "zlib",
        CompressionCodec::Lz4 => "lz4",
        CompressionCodec::Lz4Hc => "lz4hc",
        CompressionCodec::Zstd => "zstd",
    }
}

const DEFAULT_ZLIB_LEVEL: i32 = 6;
const DEFAULT_ZSTD_LEVEL: i32 = 3;

fn compress(chunk: &[u8], compression: WriteCompression) -> Vec<u8> {
    match compression.codec {
        CompressionCodec::Zlib => {
            let level = compression.level.unwrap_or(DEFAULT_ZLIB_LEVEL) as u32;
            let mut encoder = flate2::write::ZlibEncoder::new(
                Vec::with_capacity(chunk.len() / 2),
                flate2::Compression::new(level),
            );
            encoder
                .write_all(chunk)
                .and_then(|()| encoder.finish())
                .expect("compressing into memory cannot fail")
        }
        CompressionCodec::Zstd => {
            let level = compression.level.unwrap_or(DEFAULT_ZSTD_LEVEL);
            zstd::bulk::compress(chunk, level).expect("compressing into memory cannot fail")
        }
        CompressionCodec::Lz4 | CompressionCodec::Lz4Hc => lz4_flex::block::compress(chunk),
    }
}

/// The codec-independent `XISF:CompressionLevel`: the codec's level range
/// mapped linearly onto 1 to 100.
fn abstract_compression_level(compression: WriteCompression) -> Option<i32> {
    let (level, low, high) = match compression.codec {
        CompressionCodec::Zlib => (compression.level.unwrap_or(DEFAULT_ZLIB_LEVEL), 0, 9),
        CompressionCodec::Zstd => (compression.level.unwrap_or(DEFAULT_ZSTD_LEVEL), 1, 22),
        CompressionCodec::Lz4 | CompressionCodec::Lz4Hc => return None,
    };
    Some(1 + ((level - low) * 99 + (high - low) / 2) / (high - low))
}

/// The XISF byte shuffle: all first bytes of each item, then all second
/// bytes, and so on. Trailing bytes that do not form a whole item stay last.
fn shuffle(bytes: &[u8], item_bytes: usize) -> Vec<u8> {
    let items = bytes.len() / item_bytes;
    let mut output = Vec::with_capacity(bytes.len());
    for lane in 0..item_bytes {
        output.extend((0..items).map(|item| bytes[item * item_bytes + lane]));
    }
    output.extend_from_slice(&bytes[items * item_bytes..]);
    output
}

/// The `XISF:CreatorOS` value for the operating system this build targets.
fn creator_os() -> Option<&'static str> {
    if cfg!(target_os = "linux") {
        Some("Linux")
    } else if cfg!(target_os = "macos") {
        Some("macOS")
    } else if cfg!(target_os = "windows") {
        Some("Windows")
    } else if cfg!(target_os = "freebsd") {
        Some("FreeBSD")
    } else {
        None
    }
}

const PREAMBLE_LEN: usize = 16;

fn render_xml(
    width: usize,
    height: usize,
    pixels: F32ImageData<'_>,
    headers: &[WriteHeaderCard],
    bounds: (f32, f32),
    (data_offset, block): (usize, &PreparedBlock),
    created: &str,
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
         location=\"attachment:{data_offset}:{}\"{}>",
        bounds.0, bounds.1, block.bytes, block.attributes
    );
    for header in headers {
        // name, value and comment are all mandatory attributes.
        let _ = write!(
            xml,
            "<FITSKeyword name=\"{}\" value=\"{}\" comment=\"{}\"/>",
            escape(header.keyword()),
            escape(fits_value_text(header.value())),
            escape(header.comment())
        );
    }
    xml.push_str("</Image>");
    let _ = write!(
        xml,
        "<Metadata><Property id=\"XISF:CreationTime\" type=\"TimePoint\" value=\"{created}\"/>\
         <Property id=\"XISF:CreatorApplication\" type=\"String\">seiza-xisf {}</Property>",
        env!("CARGO_PKG_VERSION")
    );
    if let Some(os) = creator_os() {
        let _ = write!(
            xml,
            "<Property id=\"XISF:CreatorOS\" type=\"String\">{os}</Property>"
        );
    }
    let _ = write!(
        xml,
        "<Property id=\"XISF:BlockAlignmentSize\" type=\"UInt16\" value=\"{BLOCK_ALIGNMENT}\"/>"
    );
    for (id, type_name, value) in &block.metadata {
        if *type_name == "String" {
            let _ = write!(
                xml,
                "<Property id=\"{id}\" type=\"String\">{}</Property>",
                escape(value.as_str())
            );
        } else {
            let _ = write!(
                xml,
                "<Property id=\"{id}\" type=\"{type_name}\" value=\"{value}\"/>"
            );
        }
    }
    xml.push_str("</Metadata>");
    xml.push_str("</xisf>");
    xml
}

/// Format a time as an ISO 8601 UTC timestamp, such as `2026-09-28T05:02:40Z`.
fn utc_timestamp(time: std::time::SystemTime) -> String {
    let seconds = time
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    let (days, second_of_day) = (seconds / 86_400, seconds % 86_400);
    // Civil date from days since 1970-01-01, after Howard Hinnant's
    // days_from_civil inverse, in 400-year eras starting on March 1.
    let days = days + 719_468;
    let era = days / 146_097;
    let day_of_era = days % 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + u64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        second_of_day / 3_600,
        second_of_day / 60 % 60,
        second_of_day % 60
    )
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

    fn smooth_rgb(width: usize, height: usize) -> Vec<f32> {
        (0..width * height * 3)
            .map(|index| 0.25 + (index % 97) as f32 / 1000.0)
            .collect()
    }

    fn write_to_memory(values: &[f32], options: &WriteOptions, subblock_bytes: usize) -> Vec<u8> {
        let mut encoded = Vec::new();
        write_image(
            &mut encoded,
            (16, 8, F32ImageData::RgbInterleaved(values)),
            &[],
            options,
            subblock_bytes,
        )
        .unwrap();
        encoded
    }

    fn planar_from(values: &[f32], pixels: usize) -> Vec<f32> {
        (0..3)
            .flat_map(|channel| (0..pixels).map(move |index| values[index * 3 + channel]))
            .collect()
    }

    #[test]
    fn compressed_and_checksummed_files_round_trip() {
        let values = smooth_rgb(16, 8);
        let expected = planar_from(&values, 16 * 8);
        let checksums = [
            ChecksumAlgorithm::Sha1,
            ChecksumAlgorithm::Sha256,
            ChecksumAlgorithm::Sha512,
            ChecksumAlgorithm::Sha3_256,
            ChecksumAlgorithm::Sha3_512,
        ];
        let mut cases = Vec::new();
        for codec in [
            CompressionCodec::Zlib,
            CompressionCodec::Lz4,
            CompressionCodec::Zstd,
        ] {
            for byte_shuffle in [false, true] {
                cases.push(Some(WriteCompression {
                    codec,
                    byte_shuffle,
                    level: None,
                }));
            }
        }
        cases.push(None);
        for (case, compression) in cases.into_iter().enumerate() {
            for subblock_bytes in [SUBBLOCK_BYTES, 500] {
                let options = WriteOptions {
                    compression,
                    checksum: Some(checksums[case % checksums.len()]),
                };
                let encoded = write_to_memory(&values, &options, subblock_bytes);
                let read = crate::read_image_from_bytes(&encoded, 0).unwrap();
                let Pixels::F32(actual) = &read.image.pixels else {
                    panic!("writer must emit f32 pixels");
                };
                assert_eq!(actual, &expected, "{options:?}");
                if let Some(compression) = compression {
                    let info = read.info.compression.expect("block must be compressed");
                    assert_eq!(info.codec, compression.codec);
                    assert_eq!(info.shuffled_item_bytes.is_some(), compression.byte_shuffle);
                    assert_eq!(info.subblocks.len() > 1, subblock_bytes == 500);
                }
                let header = String::from_utf8_lossy(&encoded[..4096]);
                assert!(header.contains("XISF:ChecksumAlgorithms"));
                assert_eq!(
                    header.contains("XISF:CompressionCodecs"),
                    compression.is_some()
                );

                // A flipped stored byte must fail verification before decoding.
                let mut corrupt = encoded.clone();
                let last = corrupt.len() - 1;
                corrupt[last] ^= 0x55;
                assert!(matches!(
                    crate::from_bytes(&corrupt),
                    Err(XisfError::Malformed(message)) if message.contains("checksum mismatch")
                ));
            }
        }
    }

    #[test]
    fn stores_a_block_that_does_not_shrink_uncompressed() {
        // Pseudo-random bits leave nothing for LZ4 to find.
        let mut state = 0x9e37_79b9_u32;
        let values = (0..16 * 8 * 3)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                f32::from_bits(state & 0x3fff_ffff)
            })
            .collect::<Vec<_>>();
        let options = WriteOptions {
            compression: Some(WriteCompression {
                codec: CompressionCodec::Lz4,
                byte_shuffle: false,
                level: None,
            }),
            checksum: None,
        };
        let encoded = write_to_memory(&values, &options, SUBBLOCK_BYTES);
        let read = crate::read_image_from_bytes(&encoded, 0).unwrap();
        assert!(read.info.compression.is_none());
        assert!(!String::from_utf8_lossy(&encoded[..4096]).contains("CompressionCodecs"));
    }

    #[test]
    fn rejects_unavailable_codecs_and_levels() {
        let values = smooth_rgb(16, 8);
        let write = |codec, level| {
            let options = WriteOptions {
                compression: Some(WriteCompression {
                    codec,
                    byte_shuffle: true,
                    level,
                }),
                checksum: None,
            };
            write_image(
                Vec::new(),
                (16, 8, F32ImageData::RgbInterleaved(&values)),
                &[],
                &options,
                SUBBLOCK_BYTES,
            )
        };
        assert!(matches!(
            write(CompressionCodec::Lz4Hc, None),
            Err(XisfError::Unsupported(_))
        ));
        assert!(write(CompressionCodec::Zlib, Some(10)).is_err());
        assert!(write(CompressionCodec::Zstd, Some(0)).is_err());
        assert!(write(CompressionCodec::Lz4, Some(1)).is_err());
        assert!(write(CompressionCodec::Zlib, Some(9)).is_ok());
        assert!(write(CompressionCodec::Zstd, Some(22)).is_ok());
    }

    #[test]
    fn maps_codec_levels_onto_the_abstract_range() {
        let level = |codec, level| {
            abstract_compression_level(WriteCompression {
                codec,
                byte_shuffle: false,
                level: Some(level),
            })
        };
        assert_eq!(level(CompressionCodec::Zlib, 0), Some(1));
        assert_eq!(level(CompressionCodec::Zlib, 9), Some(100));
        assert_eq!(level(CompressionCodec::Zstd, 1), Some(1));
        assert_eq!(level(CompressionCodec::Zstd, 22), Some(100));
        assert_eq!(level(CompressionCodec::Lz4, 0), None);
    }

    #[test]
    fn shuffle_matches_the_readers_unshuffle() {
        let bytes = (0_u8..23).collect::<Vec<_>>();
        for item_bytes in [1, 2, 4, 8] {
            assert_eq!(
                crate::unshuffle(&shuffle(&bytes, item_bytes), item_bytes),
                bytes
            );
        }
    }

    #[test]
    fn formats_utc_timestamps() {
        let at = |seconds| std::time::UNIX_EPOCH + std::time::Duration::from_secs(seconds);
        assert_eq!(utc_timestamp(at(0)), "1970-01-01T00:00:00Z");
        assert_eq!(utc_timestamp(at(951_782_400)), "2000-02-29T00:00:00Z");
        assert_eq!(utc_timestamp(at(1_790_571_760)), "2026-09-28T05:02:40Z");
        assert_eq!(utc_timestamp(at(4_107_542_399)), "2100-02-28T23:59:59Z");
    }

    #[test]
    fn writes_the_mandatory_metadata_and_keyword_attributes() {
        let headers = [WriteHeaderCard::new("EXPTIME", HeaderValue::Float(30.0))];
        let mut encoded = Vec::new();
        write_f32_image_to(&mut encoded, 1, 1, F32ImageData::Mono(&[0.5]), &headers).unwrap();
        let header = String::from_utf8_lossy(&encoded);
        assert!(header.contains("<Property id=\"XISF:CreationTime\" type=\"TimePoint\" value=\""));
        assert!(header.contains("<Property id=\"XISF:CreatorApplication\" type=\"String\">"));
        assert!(
            header.contains(
                "<FITSKeyword name=\"EXPTIME\" value=\"3.000000000000E1\" comment=\"\"/>"
            )
        );
    }

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
        let crate::BlockLocation::Attachment { offset, .. } = image.location else {
            panic!("writer must attach the pixel block");
        };
        assert_eq!(offset % BLOCK_ALIGNMENT as u64, 0);
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
        let parsed = crate::parse_file(&mut cursor, encoded.len() as u64, None).unwrap();
        let crate::BlockLocation::Attachment { offset, .. } = parsed.images[0].info.location else {
            panic!("writer must attach the pixel block");
        };
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
