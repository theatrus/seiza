//! Practical XISF 1.0 image reading for Seiza.
//!
//! This crate deliberately focuses on the monolithic, attached-image layout
//! produced by PixInsight for normal astrophotography workflows. Decoded
//! images use [`seiza_fits::FitsImage`] so downstream statistics, stretching,
//! Bayer handling, stacking, and solving do not depend on the source format.
//!
//! Sample values pass through unchanged: the XISF `bounds` attribute is
//! intentionally ignored, keeping linear data linear, and preserved FITS
//! scaling keywords (`BZERO`/`BSCALE`) are dropped because XISF samples are
//! already physical.

use flate2::read::ZlibDecoder;
use quick_xml::Reader;
use quick_xml::events::{BytesStart, Event};
use seiza_fits::{FitsImage, HeaderValue, Pixels, parse_header_value};
use sha1::Sha1;
use sha2::{Sha256, Sha512};
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const SIGNATURE: &[u8; 8] = b"XISF0100";
const PREAMBLE_BYTES: u64 = 16;
const MAX_HEADER_BYTES: usize = 16 * 1024 * 1024;
const MAX_SAMPLES: usize = 2_000_000_000;
const CHUNK_BYTES: usize = 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum XisfError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("not an XISF 1.0 file")]
    NotXisf,
    #[error("malformed XISF: {0}")]
    Malformed(String),
    #[error("unsupported XISF: {0}")]
    Unsupported(String),
    #[error("XISF image {0} does not exist")]
    ImageNotFound(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleFormat {
    UInt8,
    UInt16,
    UInt32,
    Float32,
    Float64,
}

impl SampleFormat {
    fn parse(value: &str) -> Result<Self, XisfError> {
        match value {
            "UInt8" | "Byte" => Ok(Self::UInt8),
            "UInt16" | "UShort" => Ok(Self::UInt16),
            "UInt32" | "UInt" => Ok(Self::UInt32),
            "Float32" | "Float" => Ok(Self::Float32),
            "Float64" | "Double" => Ok(Self::Float64),
            value => Err(XisfError::Unsupported(format!("sample format {value:?}"))),
        }
    }

    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::UInt8 => 1,
            Self::UInt16 => 2,
            Self::UInt32 | Self::Float32 => 4,
            Self::Float64 => 8,
        }
    }

    fn fits_bitpix(self) -> i64 {
        match self {
            Self::UInt8 => 8,
            Self::UInt16 => 16,
            Self::UInt32 => 32,
            Self::Float32 => -32,
            Self::Float64 => -64,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ByteOrder {
    #[default]
    Little,
    Big,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompressionCodec {
    Zlib,
    Lz4,
    Lz4Hc,
    Zstd,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressionInfo {
    pub codec: CompressionCodec,
    pub uncompressed_bytes: usize,
    pub shuffled_item_bytes: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XisfProperty {
    pub id: String,
    pub type_name: String,
    pub value: Option<String>,
    pub comment: Option<String>,
    pub format: Option<String>,
    pub location: Option<String>,
}

#[derive(Clone, Debug)]
pub struct XisfImageInfo {
    pub index: usize,
    pub id: Option<String>,
    pub image_type: Option<String>,
    pub width: usize,
    pub height: usize,
    pub planes: usize,
    pub sample_format: SampleFormat,
    pub color_space: String,
    pub byte_order: ByteOrder,
    pub attachment_offset: u64,
    pub attachment_bytes: u64,
    pub compression: Option<CompressionInfo>,
    pub headers: Vec<(String, HeaderValue)>,
    pub properties: Vec<XisfProperty>,
    pub cfa_pattern: Option<String>,
}

#[derive(Clone, Debug)]
pub struct XisfFileInfo {
    pub images: Vec<XisfImageInfo>,
}

/// Whether a path uses the conventional `.xisf` extension.
pub fn is_xisf_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("xisf"))
}

#[derive(Clone, Debug)]
struct ParsedImage {
    info: XisfImageInfo,
    checksum: Option<String>,
    expected_bytes: usize,
}

#[derive(Clone, Debug)]
struct ParsedFile {
    images: Vec<ParsedImage>,
}

/// Read the XISF header and describe every top-level image without loading
/// pixel attachments.
pub fn inspect(path: &Path) -> Result<XisfFileInfo, XisfError> {
    let mut file = std::fs::File::open(path)?;
    let file_bytes = file.metadata()?.len();
    let parsed = parse_file(&mut file, file_bytes)?;
    Ok(XisfFileInfo {
        images: parsed.images.into_iter().map(|image| image.info).collect(),
    })
}

/// Read the first image's FITS-compatible metadata without decoding pixels.
pub fn read_header(path: &Path) -> Result<Vec<(String, HeaderValue)>, XisfError> {
    let mut file = std::fs::File::open(path)?;
    let file_bytes = file.metadata()?.len();
    let parsed = parse_file(&mut file, file_bytes)?;
    let image = parsed
        .images
        .first()
        .ok_or_else(|| XisfError::Malformed("file contains no images".into()))?;
    let mut headers = image.info.headers.clone();
    add_structural_headers(&mut headers, &image.info);
    Ok(headers)
}

/// Open the first top-level image in a monolithic XISF file.
pub fn open(path: &Path) -> Result<FitsImage, XisfError> {
    open_image(path, 0)
}

/// Open a top-level image by zero-based index.
pub fn open_image(path: &Path, index: usize) -> Result<FitsImage, XisfError> {
    let mut file = std::fs::File::open(path)?;
    let file_bytes = file.metadata()?.len();
    open_image_from(&mut file, file_bytes, ImageSelection::Index(index))
}

/// Open a top-level image by its case-sensitive XISF `id` attribute.
pub fn open_image_by_id(path: &Path, id: &str) -> Result<FitsImage, XisfError> {
    let mut file = std::fs::File::open(path)?;
    let file_bytes = file.metadata()?.len();
    open_image_from(&mut file, file_bytes, ImageSelection::Id(id))
}

/// Decode the first image from a complete in-memory monolithic XISF file.
pub fn from_bytes(bytes: &[u8]) -> Result<FitsImage, XisfError> {
    image_from_bytes(bytes, 0)
}

/// Decode an indexed image from a complete in-memory monolithic XISF file.
pub fn image_from_bytes(bytes: &[u8], index: usize) -> Result<FitsImage, XisfError> {
    let mut reader = std::io::Cursor::new(bytes);
    open_image_from(
        &mut reader,
        bytes.len() as u64,
        ImageSelection::Index(index),
    )
}

enum ImageSelection<'a> {
    Index(usize),
    Id(&'a str),
}

fn open_image_from(
    reader: &mut (impl Read + Seek),
    file_bytes: u64,
    selection: ImageSelection<'_>,
) -> Result<FitsImage, XisfError> {
    let parsed = parse_file(reader, file_bytes)?;
    let image = match selection {
        ImageSelection::Index(index) => parsed
            .images
            .get(index)
            .ok_or_else(|| XisfError::ImageNotFound(format!("at zero-based index {index}")))?,
        ImageSelection::Id(id) => parsed
            .images
            .iter()
            .find(|image| image.info.id.as_deref() == Some(id))
            .ok_or_else(|| XisfError::ImageNotFound(format!("with id {id:?}")))?,
    };

    if let Some(checksum) = &image.checksum {
        verify_checksum(reader, image, checksum)?;
    }
    reader.seek(SeekFrom::Start(image.info.attachment_offset))?;
    let pixels = decode_attachment(reader, image)?;
    let mut headers = image.info.headers.clone();
    add_structural_headers(&mut headers, &image.info);

    Ok(FitsImage {
        width: image.info.width,
        height: image.info.height,
        planes: image.info.planes,
        pixels,
        headers,
    })
}

fn parse_file(reader: &mut (impl Read + Seek), file_bytes: u64) -> Result<ParsedFile, XisfError> {
    reader.seek(SeekFrom::Start(0))?;
    let mut preamble = [0_u8; PREAMBLE_BYTES as usize];
    reader.read_exact(&mut preamble).map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            XisfError::NotXisf
        } else {
            XisfError::Io(error)
        }
    })?;
    if &preamble[..8] != SIGNATURE {
        return Err(XisfError::NotXisf);
    }
    if preamble[12..16] != [0; 4] {
        return Err(XisfError::Malformed(
            "reserved preamble field is not zero".into(),
        ));
    }
    let header_bytes = u32::from_le_bytes(preamble[8..12].try_into().unwrap()) as usize;
    if header_bytes == 0 || header_bytes > MAX_HEADER_BYTES {
        return Err(XisfError::Malformed(format!(
            "XML header length {header_bytes} is outside the supported range"
        )));
    }
    let header_end = PREAMBLE_BYTES
        .checked_add(header_bytes as u64)
        .ok_or_else(|| XisfError::Malformed("XML header length overflows".into()))?;
    if header_end > file_bytes {
        return Err(XisfError::Malformed("XML header runs past EOF".into()));
    }
    let mut xml = vec![0_u8; header_bytes];
    reader.read_exact(&mut xml)?;
    parse_xml(&xml, header_end, file_bytes)
}

fn parse_xml(xml: &[u8], header_end: u64, file_bytes: u64) -> Result<ParsedFile, XisfError> {
    let mut reader = Reader::from_reader(xml);
    reader.config_mut().trim_text(false);
    let mut buffer = Vec::new();
    let mut images = Vec::<ParsedImage>::new();
    let mut current_image = None::<(usize, usize)>;
    let mut current_property = None::<(usize, usize, usize)>;
    let mut depth = 0_usize;
    let mut saw_root = false;

    loop {
        match reader
            .read_event_into(&mut buffer)
            .map_err(|error| XisfError::Malformed(format!("invalid XML header: {error}")))?
        {
            Event::Start(element) => {
                let element_depth = depth + 1;
                handle_element(
                    &reader,
                    &element,
                    element_depth,
                    header_end,
                    file_bytes,
                    &mut images,
                    &mut current_image,
                    &mut current_property,
                    &mut saw_root,
                    false,
                )?;
                depth = element_depth;
            }
            Event::Empty(element) => {
                handle_element(
                    &reader,
                    &element,
                    depth + 1,
                    header_end,
                    file_bytes,
                    &mut images,
                    &mut current_image,
                    &mut current_property,
                    &mut saw_root,
                    true,
                )?;
            }
            Event::Text(text) => {
                if let Some((image_index, property_index, property_depth)) = current_property
                    && depth == property_depth
                {
                    let decoded = text.decode().map_err(|error| {
                        XisfError::Malformed(format!("invalid property text: {error}"))
                    })?;
                    let value = quick_xml::escape::unescape(&decoded)
                        .map_err(|error| {
                            XisfError::Malformed(format!("invalid property text: {error}"))
                        })?
                        .into_owned();
                    let property = &mut images[image_index].info.properties[property_index];
                    property
                        .value
                        .get_or_insert_with(String::new)
                        .push_str(&value);
                }
            }
            Event::End(element) => {
                let name = element.local_name();
                if name.as_ref() == b"Property"
                    && current_property
                        .is_some_and(|(_, _, property_depth)| depth == property_depth)
                {
                    current_property = None;
                }
                if name.as_ref() == b"Image"
                    && current_image.is_some_and(|(_, image_depth)| depth == image_depth)
                {
                    current_image = None;
                    current_property = None;
                }
                depth = depth.saturating_sub(1);
            }
            Event::Eof => break,
            Event::DocType(_) => {
                return Err(XisfError::Unsupported("XML document types".into()));
            }
            _ => {}
        }
        buffer.clear();
    }

    if !saw_root {
        return Err(XisfError::Malformed("missing xisf root element".into()));
    }
    if images.is_empty() {
        return Err(XisfError::Malformed("file contains no images".into()));
    }
    Ok(ParsedFile { images })
}

#[allow(clippy::too_many_arguments)]
fn handle_element(
    reader: &Reader<&[u8]>,
    element: &BytesStart<'_>,
    depth: usize,
    header_end: u64,
    file_bytes: u64,
    images: &mut Vec<ParsedImage>,
    current_image: &mut Option<(usize, usize)>,
    current_property: &mut Option<(usize, usize, usize)>,
    saw_root: &mut bool,
    empty: bool,
) -> Result<(), XisfError> {
    let name = element.local_name();
    let attributes = attributes(reader, element)?;
    match name.as_ref() {
        b"xisf" if depth == 1 => {
            if attributes.get("version").map(String::as_str) != Some("1.0") {
                return Err(XisfError::Unsupported(format!(
                    "XISF version {:?}",
                    attributes.get("version")
                )));
            }
            *saw_root = true;
        }
        b"Image" if depth == 2 => {
            let index = images.len();
            images.push(parse_image(index, &attributes, header_end, file_bytes)?);
            if !empty {
                *current_image = Some((index, depth));
            }
        }
        b"FITSKeyword" => {
            if let Some((image_index, image_depth)) = *current_image
                && depth == image_depth + 1
            {
                let keyword = required(&attributes, "name", "FITSKeyword")?.to_string();
                // XISF samples are already physical and the XML geometry is
                // authoritative, so preserved FITS scaling and structure
                // keywords must not be re-applied by FITS-side consumers
                // such as into_physical_f32. COMMENT/HISTORY-style keywords
                // legitimately carry no value.
                if !structural_fits_keyword(&keyword) {
                    let raw = attributes.get("value").map(String::as_str).unwrap_or("");
                    images[image_index]
                        .info
                        .headers
                        .push((keyword, parse_header_value(raw)));
                }
            }
        }
        b"Property" => {
            if let Some((image_index, image_depth)) = *current_image
                && depth == image_depth + 1
            {
                let property = XisfProperty {
                    id: required(&attributes, "id", "Property")?.to_string(),
                    type_name: required(&attributes, "type", "Property")?.to_string(),
                    value: attributes.get("value").cloned(),
                    comment: attributes.get("comment").cloned(),
                    format: attributes.get("format").cloned(),
                    location: attributes.get("location").cloned(),
                };
                let property_index = images[image_index].info.properties.len();
                images[image_index].info.properties.push(property);
                if !empty {
                    *current_property = Some((image_index, property_index, depth));
                }
            }
        }
        b"ColorFilterArray" => {
            if let Some((image_index, image_depth)) = *current_image
                && depth == image_depth + 1
            {
                let width = parse_usize(required(&attributes, "width", "ColorFilterArray")?)?;
                let height = parse_usize(required(&attributes, "height", "ColorFilterArray")?)?;
                let pattern = required(&attributes, "pattern", "ColorFilterArray")?;
                if width == 2 && height == 2 && pattern.len() == 4 {
                    images[image_index].info.cfa_pattern = Some(pattern.to_string());
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn attributes(
    reader: &Reader<&[u8]>,
    element: &BytesStart<'_>,
) -> Result<BTreeMap<String, String>, XisfError> {
    let mut values = BTreeMap::new();
    for attribute in element.attributes() {
        let attribute = attribute
            .map_err(|error| XisfError::Malformed(format!("invalid XML attribute: {error}")))?;
        let key = std::str::from_utf8(attribute.key.as_ref())
            .map_err(|_| XisfError::Malformed("non-UTF-8 XML attribute name".into()))?;
        let value = attribute
            .decode_and_unescape_value(reader.decoder())
            .map_err(|error| XisfError::Malformed(format!("invalid XML attribute: {error}")))?;
        values.insert(key.to_string(), value.into_owned());
    }
    Ok(values)
}

fn parse_image(
    index: usize,
    attributes: &BTreeMap<String, String>,
    header_end: u64,
    file_bytes: u64,
) -> Result<ParsedImage, XisfError> {
    if attributes.contains_key("subblocks") {
        return Err(XisfError::Unsupported("compression subblocks".into()));
    }
    let geometry = required(attributes, "geometry", "Image")?
        .split(':')
        .collect::<Vec<_>>();
    if geometry.len() != 3 {
        return Err(XisfError::Unsupported(format!(
            "image geometry {:?}; only two-dimensional images are supported",
            attributes.get("geometry")
        )));
    }
    let width = parse_usize(geometry[0])?;
    let height = parse_usize(geometry[1])?;
    let planes = parse_usize(geometry[2])?;
    if width == 0 || height == 0 || !matches!(planes, 1 | 3) {
        return Err(XisfError::Unsupported(format!(
            "image geometry {width}:{height}:{planes}"
        )));
    }
    let count = width
        .checked_mul(height)
        .and_then(|count| count.checked_mul(planes))
        .ok_or_else(|| XisfError::Malformed("image dimensions overflow".into()))?;
    if count == 0 || count > MAX_SAMPLES {
        return Err(XisfError::Malformed("implausible image dimensions".into()));
    }

    let sample_format = SampleFormat::parse(required(attributes, "sampleFormat", "Image")?)?;
    let expected_bytes = count
        .checked_mul(sample_format.bytes_per_sample())
        .ok_or_else(|| XisfError::Malformed("image byte count overflows".into()))?;
    let pixel_storage = attributes
        .get("pixelStorage")
        .map(String::as_str)
        .unwrap_or("Planar");
    if pixel_storage != "Planar" {
        return Err(XisfError::Unsupported(format!(
            "pixel storage model {pixel_storage:?}"
        )));
    }
    let color_space = attributes
        .get("colorSpace")
        .cloned()
        .unwrap_or_else(|| "Gray".into());
    if !matches!((planes, color_space.as_str()), (1, "Gray") | (3, "RGB")) {
        return Err(XisfError::Unsupported(format!(
            "{planes}-channel {color_space} image"
        )));
    }
    let byte_order = match attributes.get("byteOrder").map(String::as_str) {
        None | Some("little") => ByteOrder::Little,
        Some("big") => ByteOrder::Big,
        Some(value) => {
            return Err(XisfError::Malformed(format!(
                "invalid byte order {value:?}"
            )));
        }
    };
    let (attachment_offset, attachment_bytes) =
        parse_attachment(required(attributes, "location", "Image")?)?;
    let attachment_end = attachment_offset
        .checked_add(attachment_bytes)
        .ok_or_else(|| XisfError::Malformed("attachment range overflows".into()))?;
    if attachment_offset < header_end || attachment_end > file_bytes {
        return Err(XisfError::Malformed(format!(
            "attachment {attachment_offset}:{attachment_bytes} is outside the file"
        )));
    }
    let compression = attributes
        .get("compression")
        .map(|value| parse_compression(value))
        .transpose()?;
    if let Some(compression) = &compression {
        if compression.uncompressed_bytes != expected_bytes {
            return Err(XisfError::Malformed(format!(
                "declared uncompressed size {} does not match image size {expected_bytes}",
                compression.uncompressed_bytes
            )));
        }
        if let Some(item_bytes) = compression.shuffled_item_bytes
            && item_bytes != sample_format.bytes_per_sample()
        {
            return Err(XisfError::Malformed(format!(
                "shuffle item size {item_bytes} does not match {:?} samples",
                sample_format
            )));
        }
    } else if attachment_bytes != expected_bytes as u64 {
        return Err(XisfError::Malformed(format!(
            "attachment size {attachment_bytes} does not match image size {expected_bytes}"
        )));
    }

    Ok(ParsedImage {
        info: XisfImageInfo {
            index,
            id: attributes.get("id").cloned(),
            image_type: attributes.get("imageType").cloned(),
            width,
            height,
            planes,
            sample_format,
            color_space,
            byte_order,
            attachment_offset,
            attachment_bytes,
            compression,
            headers: Vec::new(),
            properties: Vec::new(),
            cfa_pattern: None,
        },
        checksum: attributes.get("checksum").cloned(),
        expected_bytes,
    })
}

/// Whether a preserved FITS keyword describes storage scaling or geometry
/// that the XISF XML already resolved. Passing these through would make
/// FITS-side consumers re-apply scaling to already-physical samples.
fn structural_fits_keyword(keyword: &str) -> bool {
    keyword.eq_ignore_ascii_case("BZERO")
        || keyword.eq_ignore_ascii_case("BSCALE")
        || keyword.eq_ignore_ascii_case("BITPIX")
        || keyword.eq_ignore_ascii_case("SIMPLE")
        || keyword.eq_ignore_ascii_case("END")
        || keyword
            .get(..5)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("NAXIS"))
}

fn required<'a>(
    attributes: &'a BTreeMap<String, String>,
    name: &str,
    element: &str,
) -> Result<&'a str, XisfError> {
    attributes
        .get(name)
        .map(String::as_str)
        .ok_or_else(|| XisfError::Malformed(format!("{element} is missing {name}")))
}

fn parse_usize(value: &str) -> Result<usize, XisfError> {
    value
        .parse()
        .map_err(|_| XisfError::Malformed(format!("invalid unsigned integer {value:?}")))
}

fn parse_attachment(location: &str) -> Result<(u64, u64), XisfError> {
    let parts = location.split(':').collect::<Vec<_>>();
    if parts.len() != 3 || parts[0] != "attachment" {
        return Err(XisfError::Unsupported(format!(
            "image data location {location:?}"
        )));
    }
    let offset = parts[1]
        .parse()
        .map_err(|_| XisfError::Malformed(format!("invalid attachment offset in {location:?}")))?;
    let bytes = parts[2]
        .parse()
        .map_err(|_| XisfError::Malformed(format!("invalid attachment size in {location:?}")))?;
    Ok((offset, bytes))
}

fn parse_compression(value: &str) -> Result<CompressionInfo, XisfError> {
    let parts = value.split(':').collect::<Vec<_>>();
    if !matches!(parts.len(), 2 | 3) {
        return Err(XisfError::Malformed(format!(
            "invalid compression descriptor {value:?}"
        )));
    }
    let (codec_name, shuffled) = parts[0]
        .strip_suffix("+sh")
        .map_or((parts[0], false), |codec| (codec, true));
    let codec = match codec_name {
        "zlib" => CompressionCodec::Zlib,
        "lz4" => CompressionCodec::Lz4,
        "lz4hc" => CompressionCodec::Lz4Hc,
        "zstd" => CompressionCodec::Zstd,
        name => {
            return Err(XisfError::Unsupported(format!(
                "compression codec {name:?}"
            )));
        }
    };
    let uncompressed_bytes = parse_usize(parts[1])?;
    let shuffled_item_bytes = if shuffled {
        if parts.len() != 3 {
            return Err(XisfError::Malformed(format!(
                "missing shuffle item size in {value:?}"
            )));
        }
        Some(parse_usize(parts[2])?)
    } else {
        if parts.len() != 2 {
            return Err(XisfError::Malformed(format!(
                "unexpected compression parameter in {value:?}"
            )));
        }
        None
    };
    Ok(CompressionInfo {
        codec,
        uncompressed_bytes,
        shuffled_item_bytes,
    })
}

fn verify_checksum(
    reader: &mut (impl Read + Seek),
    image: &ParsedImage,
    checksum: &str,
) -> Result<(), XisfError> {
    let (algorithm, expected) = checksum
        .split_once(':')
        .ok_or_else(|| XisfError::Malformed(format!("invalid checksum {checksum:?}")))?;
    enum ChecksumDigest {
        Sha1(Sha1),
        Sha256(Sha256),
        Sha512(Sha512),
    }
    let mut digest = match algorithm {
        "sha1" | "sha-1" => ChecksumDigest::Sha1(Sha1::default()),
        "sha256" | "sha-256" => ChecksumDigest::Sha256(Sha256::default()),
        "sha512" | "sha-512" => ChecksumDigest::Sha512(Sha512::default()),
        algorithm => {
            return Err(XisfError::Unsupported(format!(
                "checksum algorithm {algorithm:?}"
            )));
        }
    };
    reader.seek(SeekFrom::Start(image.info.attachment_offset))?;
    let mut remaining = image.info.attachment_bytes;
    let mut buffer = vec![0_u8; CHUNK_BYTES];
    while remaining != 0 {
        let bytes = usize::try_from(remaining.min(buffer.len() as u64)).unwrap();
        reader.read_exact(&mut buffer[..bytes])?;
        match &mut digest {
            ChecksumDigest::Sha1(digest) => sha1::Digest::update(digest, &buffer[..bytes]),
            ChecksumDigest::Sha256(digest) => sha2::Digest::update(digest, &buffer[..bytes]),
            ChecksumDigest::Sha512(digest) => sha2::Digest::update(digest, &buffer[..bytes]),
        }
        remaining -= bytes as u64;
    }
    let actual = match digest {
        ChecksumDigest::Sha1(digest) => lowercase_hex(sha1::Digest::finalize(digest).as_ref()),
        ChecksumDigest::Sha256(digest) => lowercase_hex(sha2::Digest::finalize(digest).as_ref()),
        ChecksumDigest::Sha512(digest) => lowercase_hex(sha2::Digest::finalize(digest).as_ref()),
    };
    if actual != expected.to_ascii_lowercase() {
        return Err(XisfError::Malformed(format!(
            "checksum mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn lowercase_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;

    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

fn decode_attachment(
    reader: &mut (impl Read + Seek),
    image: &ParsedImage,
) -> Result<Pixels, XisfError> {
    let Some(compression) = &image.info.compression else {
        let mut attachment = reader.take(image.info.attachment_bytes);
        return decode_reader(&mut attachment, image);
    };

    let mut raw = Vec::new();
    raw.try_reserve_exact(image.expected_bytes)
        .map_err(|_| XisfError::Malformed("decompression allocation failed".into()))?;
    match compression.codec {
        CompressionCodec::Zlib => {
            let attachment = reader.take(image.info.attachment_bytes);
            read_decompressed(ZlibDecoder::new(attachment), image.expected_bytes, &mut raw)?;
        }
        CompressionCodec::Zstd => {
            let attachment = reader.take(image.info.attachment_bytes);
            let decoder = zstd::stream::read::Decoder::new(attachment)
                .map_err(|error| XisfError::Malformed(format!("invalid zstd stream: {error}")))?;
            read_decompressed(decoder, image.expected_bytes, &mut raw)?;
        }
        CompressionCodec::Lz4 | CompressionCodec::Lz4Hc => {
            let stored_bytes = usize::try_from(image.info.attachment_bytes)
                .map_err(|_| XisfError::Malformed("attachment is too large".into()))?;
            let mut stored = vec![0_u8; stored_bytes];
            reader.read_exact(&mut stored)?;
            raw = lz4_flex::block::decompress(&stored, image.expected_bytes)
                .map_err(|error| XisfError::Malformed(format!("invalid LZ4 block: {error}")))?;
        }
    }
    if raw.len() != image.expected_bytes {
        return Err(XisfError::Malformed(format!(
            "decompressed {} bytes; expected {}",
            raw.len(),
            image.expected_bytes
        )));
    }
    if compression.shuffled_item_bytes.is_some() {
        decode_shuffled(&raw, image)
    } else {
        decode_bytes(&raw, image)
    }
}

fn read_decompressed(
    reader: impl Read,
    expected_bytes: usize,
    output: &mut Vec<u8>,
) -> Result<(), XisfError> {
    let limit = expected_bytes
        .checked_add(1)
        .ok_or_else(|| XisfError::Malformed("decompressed size overflows".into()))?;
    reader.take(limit as u64).read_to_end(output)?;
    if output.len() != expected_bytes {
        return Err(XisfError::Malformed(format!(
            "decompressed {} bytes; expected {expected_bytes}",
            output.len()
        )));
    }
    Ok(())
}

fn decode_reader(reader: &mut impl Read, image: &ParsedImage) -> Result<Pixels, XisfError> {
    let item_bytes = image.info.sample_format.bytes_per_sample();
    let samples_per_chunk = (CHUNK_BYTES / item_bytes).max(1);
    let mut remaining = image.info.width * image.info.height * image.info.planes;
    let mut buffer = vec![0_u8; remaining.min(samples_per_chunk) * item_bytes];
    let mut pixels = empty_pixels(image.info.sample_format, remaining)?;
    while remaining != 0 {
        let samples = remaining.min(samples_per_chunk);
        let bytes = samples * item_bytes;
        reader.read_exact(&mut buffer[..bytes]).map_err(|error| {
            if error.kind() == std::io::ErrorKind::UnexpectedEof {
                XisfError::Malformed("image attachment is truncated".into())
            } else {
                XisfError::Io(error)
            }
        })?;
        append_decoded(
            &mut pixels,
            &buffer[..bytes],
            image.info.byte_order,
            image.info.sample_format,
        )?;
        remaining -= samples;
    }
    Ok(pixels)
}

fn empty_pixels(format: SampleFormat, count: usize) -> Result<Pixels, XisfError> {
    fn reserve<T>(count: usize) -> Result<Vec<T>, XisfError> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| XisfError::Malformed("pixel buffer allocation failed".into()))?;
        Ok(values)
    }
    match format {
        SampleFormat::UInt8 => reserve(count).map(Pixels::U8),
        SampleFormat::UInt16 => reserve(count).map(Pixels::U16),
        SampleFormat::UInt32 | SampleFormat::Float64 => reserve(count).map(Pixels::F64),
        SampleFormat::Float32 => reserve(count).map(Pixels::F32),
    }
}

fn append_decoded(
    pixels: &mut Pixels,
    bytes: &[u8],
    order: ByteOrder,
    format: SampleFormat,
) -> Result<(), XisfError> {
    match (pixels, format) {
        (Pixels::U8(values), SampleFormat::UInt8) => values.extend_from_slice(bytes),
        (Pixels::U16(values), SampleFormat::UInt16) => {
            values.extend(bytes.chunks_exact(2).map(|bytes| {
                let bytes = [bytes[0], bytes[1]];
                match order {
                    ByteOrder::Little => u16::from_le_bytes(bytes),
                    ByteOrder::Big => u16::from_be_bytes(bytes),
                }
            }))
        }
        (Pixels::F32(values), SampleFormat::Float32) => {
            values.extend(bytes.chunks_exact(4).map(|bytes| {
                let bytes = bytes.try_into().unwrap();
                match order {
                    ByteOrder::Little => f32::from_le_bytes(bytes),
                    ByteOrder::Big => f32::from_be_bytes(bytes),
                }
            }))
        }
        (Pixels::F64(values), SampleFormat::UInt32) => {
            values.extend(bytes.chunks_exact(4).map(|bytes| {
                let bytes = bytes.try_into().unwrap();
                match order {
                    ByteOrder::Little => u32::from_le_bytes(bytes) as f64,
                    ByteOrder::Big => u32::from_be_bytes(bytes) as f64,
                }
            }))
        }
        (Pixels::F64(values), SampleFormat::Float64) => {
            values.extend(bytes.chunks_exact(8).map(|bytes| {
                let bytes = bytes.try_into().unwrap();
                match order {
                    ByteOrder::Little => f64::from_le_bytes(bytes),
                    ByteOrder::Big => f64::from_be_bytes(bytes),
                }
            }))
        }
        _ => {
            return Err(XisfError::Malformed(
                "pixel buffer type does not match sample format".into(),
            ));
        }
    }
    Ok(())
}

fn decode_bytes(bytes: &[u8], image: &ParsedImage) -> Result<Pixels, XisfError> {
    let mut pixels = empty_pixels(
        image.info.sample_format,
        image.info.width * image.info.height * image.info.planes,
    )?;
    append_decoded(
        &mut pixels,
        bytes,
        image.info.byte_order,
        image.info.sample_format,
    )?;
    Ok(pixels)
}

fn decode_shuffled(bytes: &[u8], image: &ParsedImage) -> Result<Pixels, XisfError> {
    let item_bytes = image.info.sample_format.bytes_per_sample();
    let count = image.info.width * image.info.height * image.info.planes;
    if bytes.len() != count * item_bytes {
        return Err(XisfError::Malformed(
            "byte-shuffled image size is inconsistent".into(),
        ));
    }
    let byte_at = |sample: usize, significance: usize| -> u8 {
        let stored_lane = match image.info.byte_order {
            ByteOrder::Little => significance,
            ByteOrder::Big => item_bytes - significance - 1,
        };
        bytes[stored_lane * count + sample]
    };
    let mut pixels = empty_pixels(image.info.sample_format, count)?;
    match (&mut pixels, image.info.sample_format) {
        (Pixels::U8(values), SampleFormat::UInt8) => values.extend_from_slice(bytes),
        (Pixels::U16(values), SampleFormat::UInt16) => values.extend(
            (0..count).map(|sample| u16::from_le_bytes([byte_at(sample, 0), byte_at(sample, 1)])),
        ),
        (Pixels::F64(values), SampleFormat::UInt32) => values.extend((0..count).map(|sample| {
            u32::from_le_bytes(std::array::from_fn(|lane| byte_at(sample, lane))) as f64
        })),
        (Pixels::F32(values), SampleFormat::Float32) => {
            values.extend((0..count).map(|sample| {
                f32::from_le_bytes(std::array::from_fn(|lane| byte_at(sample, lane)))
            }))
        }
        (Pixels::F64(values), SampleFormat::Float64) => {
            values.extend((0..count).map(|sample| {
                f64::from_le_bytes(std::array::from_fn(|lane| byte_at(sample, lane)))
            }))
        }
        _ => unreachable!("pixel storage is selected from the sample format"),
    }
    Ok(pixels)
}

fn add_structural_headers(headers: &mut Vec<(String, HeaderValue)>, image: &XisfImageInfo) {
    let mut add = |name: &str, value: HeaderValue| {
        if !headers.iter().any(|(existing, _)| existing == name) {
            headers.push((name.to_string(), value));
        }
    };
    add(
        "BITPIX",
        HeaderValue::Integer(image.sample_format.fits_bitpix()),
    );
    add(
        "NAXIS",
        HeaderValue::Integer(if image.planes == 3 { 3 } else { 2 }),
    );
    add("NAXIS1", HeaderValue::Integer(image.width as i64));
    add("NAXIS2", HeaderValue::Integer(image.height as i64));
    if image.planes == 3 {
        add("NAXIS3", HeaderValue::Integer(3));
    }
    if let Some(image_type) = &image.image_type {
        add("IMAGETYP", HeaderValue::String(image_type.clone()));
    }
    if let Some(pattern) = &image.cfa_pattern {
        add("BAYERPAT", HeaderValue::String(pattern.clone()));
    }

    let property = |id: &str| {
        image
            .properties
            .iter()
            .find(|property| property.id == id)
            .and_then(|property| property.value.as_deref())
    };
    for (property_id, header) in [
        ("Observation:Object:Name", "OBJECT"),
        ("Instrument:Camera:Name", "INSTRUME"),
        ("Instrument:Telescope:Name", "TELESCOP"),
        ("Observation:Time:Start", "DATE-BEG"),
        ("Observation:Time:End", "DATE-END"),
    ] {
        if let Some(value) = property(property_id) {
            add(header, HeaderValue::String(value.to_string()));
        }
    }
    if let Some(value) = property("Observation:Time:Start") {
        add("DATE-OBS", HeaderValue::String(value.to_string()));
    }
    for (property_id, header) in [
        ("Observation:Center:RA", "RA"),
        ("Observation:Center:Dec", "DEC"),
        ("Observation:Location:Latitude", "SITELAT"),
        ("Observation:Location:Longitude", "SITELONG"),
        ("Observation:Location:Elevation", "ALT-OBS"),
    ] {
        if let Some(value) = property(property_id).and_then(|value| value.parse().ok()) {
            add(header, HeaderValue::Float(value));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn shuffled(bytes: &[u8], item_bytes: usize) -> Vec<u8> {
        let count = bytes.len() / item_bytes;
        let mut output = vec![0_u8; bytes.len()];
        for sample in 0..count {
            for lane in 0..item_bytes {
                output[lane * count + sample] = bytes[sample * item_bytes + lane];
            }
        }
        output
    }

    fn monolithic(image_template: String, attachments: &[&[u8]]) -> Vec<u8> {
        let mut offsets = vec![0_u64; attachments.len()];
        loop {
            let mut images = image_template.clone();
            for (index, offset) in offsets.iter().enumerate() {
                images = images.replace(&format!("@OFFSET{index}@"), &offset.to_string());
            }
            let header = format!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?><xisf version=\"1.0\" xmlns=\"http://www.pixinsight.com/xisf\">{images}</xisf>"
            );
            let mut next = 16_u64 + header.len() as u64;
            let new_offsets = attachments
                .iter()
                .map(|attachment| {
                    let offset = next;
                    next += attachment.len() as u64;
                    offset
                })
                .collect::<Vec<_>>();
            if new_offsets == offsets {
                let mut bytes = Vec::new();
                bytes.extend_from_slice(SIGNATURE);
                bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
                bytes.extend_from_slice(&[0; 4]);
                bytes.extend_from_slice(header.as_bytes());
                for attachment in attachments {
                    bytes.extend_from_slice(attachment);
                }
                return bytes;
            }
            offsets = new_offsets;
        }
    }

    fn image_element(
        index: usize,
        geometry: &str,
        sample_format: &str,
        data_bytes: usize,
        extra: &str,
        children: &str,
    ) -> String {
        format!(
            "<Image id=\"image{index}\" geometry=\"{geometry}\" sampleFormat=\"{sample_format}\" {extra} location=\"attachment:@OFFSET{index}@:{data_bytes}\">{children}</Image>"
        )
    }

    #[test]
    fn reads_uncompressed_float_and_fits_keywords() {
        let values = [0.25_f32, 0.5, 1.0, -0.5];
        let raw = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let xml = image_element(
            0,
            "2:2:1",
            "Float32",
            raw.len(),
            "bounds=\"0:1\" colorSpace=\"Gray\" imageType=\"Light\"",
            "<FITSKeyword name=\"EXPTIME\" value=\"300\" comment=\"seconds\"/><Property id=\"Observation:Object:Name\" type=\"String\">M42</Property><Property id=\"Observation:Center:RA\" type=\"Float64\" value=\"83.822\"/><Property id=\"Observation:Time:Start\" type=\"TimePoint\" value=\"2026-01-02T03:04:05Z\"/>",
        );
        let bytes = monolithic(xml, &[&raw]);
        let image = from_bytes(&bytes).unwrap();
        assert!(matches!(image.pixels, Pixels::F32(ref actual) if actual == &values));
        assert_eq!(image.header_f64("EXPTIME"), Some(300.0));
        assert_eq!(image.header_str("IMAGETYP"), Some("Light"));
        assert_eq!(image.header_str("OBJECT"), Some("M42"));
        assert_eq!(image.header_f64("RA"), Some(83.822));
        assert_eq!(image.header_str("DATE-OBS"), Some("2026-01-02T03:04:05Z"));
        let mut cursor = std::io::Cursor::new(&bytes);
        let parsed = parse_file(&mut cursor, bytes.len() as u64).unwrap();
        assert_eq!(
            parsed.images[0].info.properties[0].value.as_deref(),
            Some("M42")
        );
    }

    #[test]
    fn selects_auxiliary_images_by_index_and_id() {
        let first = [1_u8, 2, 3, 4];
        let second = [9_u8, 8, 7, 6];
        let xml = format!(
            "{}{}",
            image_element(0, "2:2:1", "UInt8", 4, "colorSpace=\"Gray\"", ""),
            image_element(1, "2:2:1", "UInt8", 4, "colorSpace=\"Gray\"", "")
        );
        let bytes = monolithic(xml, &[&first, &second]);
        let image = image_from_bytes(&bytes, 1).unwrap();
        assert!(matches!(image.pixels, Pixels::U8(ref actual) if actual == &second));
        let mut cursor = std::io::Cursor::new(&bytes);
        let image = open_image_from(
            &mut cursor,
            bytes.len() as u64,
            ImageSelection::Id("image1"),
        )
        .unwrap();
        assert!(matches!(image.pixels, Pixels::U8(ref actual) if actual == &second));
    }

    #[test]
    fn decodes_zstd_byte_shuffled_rgb_u16() {
        let values = [1_u16, 2, 3, 1000, 2000, 3000];
        let raw = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let compressed = zstd::bulk::compress(&shuffled(&raw, 2), 3).unwrap();
        let checksum = lowercase_hex(<Sha1 as sha1::Digest>::digest(&compressed).as_ref());
        let xml = image_element(
            0,
            "2:1:3",
            "UInt16",
            compressed.len(),
            &format!(
                "colorSpace=\"RGB\" compression=\"zstd+sh:{}:2\" checksum=\"sha-1:{checksum}\"",
                raw.len()
            ),
            "",
        );
        let bytes = monolithic(xml, &[&compressed]);
        let image = from_bytes(&bytes).unwrap();
        assert_eq!(image.planes, 3);
        assert!(matches!(image.pixels, Pixels::U16(ref actual) if actual == &values));
    }

    #[test]
    fn decodes_big_endian_byte_shuffled_samples() {
        let values = [1_u16, 2, 3, 1000, 2000, 3000];
        let raw = values
            .iter()
            .flat_map(|value| value.to_be_bytes())
            .collect::<Vec<_>>();
        let compressed = zstd::bulk::compress(&shuffled(&raw, 2), 3).unwrap();
        let xml = image_element(
            0,
            "2:1:3",
            "UInt16",
            compressed.len(),
            &format!(
                "colorSpace=\"RGB\" byteOrder=\"big\" compression=\"zstd+sh:{}:2\"",
                raw.len()
            ),
            "",
        );
        let image = from_bytes(&monolithic(xml, &[&compressed])).unwrap();
        assert!(matches!(image.pixels, Pixels::U16(ref actual) if actual == &values));
    }

    #[test]
    fn drops_structural_fits_keywords_and_accepts_valueless_ones() {
        let values = [0.25_f32, 0.5, 1.0, -0.5];
        let raw = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let xml = image_element(
            0,
            "2:2:1",
            "Float32",
            raw.len(),
            "colorSpace=\"Gray\"",
            "<FITSKeyword name=\"BZERO\" value=\"32768\"/>\
             <FITSKeyword name=\"BSCALE\" value=\"2\"/>\
             <FITSKeyword name=\"NAXIS1\" value=\"999\"/>\
             <FITSKeyword name=\"COMMENT\"/>\
             <FITSKeyword name=\"EXPTIME\" value=\"300\"/>",
        );
        let image = from_bytes(&monolithic(xml, &[&raw])).unwrap();
        assert!(image.header("BZERO").is_none());
        assert!(image.header("BSCALE").is_none());
        // The synthesized geometry card wins over the preserved keyword.
        assert_eq!(image.header("NAXIS1"), Some(&HeaderValue::Integer(2)));
        assert!(image.header("COMMENT").is_some());
        assert_eq!(image.header_f64("EXPTIME"), Some(300.0));
        // The poisonous case: preserved FITS scaling must not shift
        // already-physical XISF samples on the FITS-side decode path.
        assert_eq!(image.into_physical_f32(), values);
    }

    #[test]
    fn decodes_lz4_and_big_endian_zlib() {
        let lz4_values = [10_u16, 20, 30, 40];
        let lz4_raw = lz4_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let lz4 = lz4_flex::block::compress(&lz4_raw);
        let lz4_xml = image_element(
            0,
            "2:2:1",
            "UInt16",
            lz4.len(),
            &format!("colorSpace=\"Gray\" compression=\"lz4:{}\"", lz4_raw.len()),
            "",
        );
        let image = from_bytes(&monolithic(lz4_xml, &[&lz4])).unwrap();
        assert!(matches!(image.pixels, Pixels::U16(ref actual) if actual == &lz4_values));

        let values = [1.25_f64, -2.5];
        let raw = values
            .iter()
            .flat_map(|value| value.to_be_bytes())
            .collect::<Vec<_>>();
        let mut encoder =
            flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&raw).unwrap();
        let compressed = encoder.finish().unwrap();
        let xml = image_element(
            0,
            "2:1:1",
            "Float64",
            compressed.len(),
            &format!(
                "bounds=\"0:1\" colorSpace=\"Gray\" byteOrder=\"big\" compression=\"zlib:{}\"",
                raw.len()
            ),
            "",
        );
        let image = from_bytes(&monolithic(xml, &[&compressed])).unwrap();
        assert!(matches!(image.pixels, Pixels::F64(ref actual) if actual == &values));
    }

    #[test]
    fn preserves_uncompressed_uint32_samples_as_f64() {
        let values = [0_u32, 1, u32::MAX, 0x1234_5678];
        let raw = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let xml = image_element(0, "2:2:1", "UInt32", raw.len(), "colorSpace=\"Gray\"", "");
        let image = from_bytes(&monolithic(xml, &[&raw])).unwrap();
        let expected = values.map(f64::from);
        assert!(matches!(image.pixels, Pixels::F64(ref actual) if actual == &expected));
    }

    #[test]
    fn exposes_two_by_two_cfa_as_bayer_header() {
        let raw = [0_u8; 4];
        let xml = image_element(
            0,
            "2:2:1",
            "UInt8",
            raw.len(),
            "colorSpace=\"Gray\"",
            "<ColorFilterArray pattern=\"RGGB\" width=\"2\" height=\"2\"/>",
        );
        let image = from_bytes(&monolithic(xml, &[&raw])).unwrap();
        assert_eq!(image.header_str("BAYERPAT"), Some("RGGB"));
        assert!(image.bayer_pattern().is_some());
    }

    #[test]
    fn verifies_sha1_before_decoding() {
        let raw = [1_u8, 2, 3, 4];
        let digest = lowercase_hex(<Sha1 as sha1::Digest>::digest(raw).as_ref());
        let xml = image_element(
            0,
            "2:2:1",
            "UInt8",
            raw.len(),
            &format!("colorSpace=\"Gray\" checksum=\"sha1:{digest}\""),
            "",
        );
        assert!(from_bytes(&monolithic(xml.clone(), &[&raw])).is_ok());
        let mut corrupt = raw;
        corrupt[0] ^= 0xff;
        assert!(matches!(
            from_bytes(&monolithic(xml, &[&corrupt])),
            Err(XisfError::Malformed(message)) if message.contains("checksum mismatch")
        ));

        let digest = lowercase_hex(<Sha256 as sha2::Digest>::digest(raw).as_ref());
        let xml = image_element(
            0,
            "2:2:1",
            "UInt8",
            raw.len(),
            &format!("colorSpace=\"Gray\" checksum=\"sha-256:{digest}\""),
            "",
        );
        assert!(from_bytes(&monolithic(xml, &[&raw])).is_ok());
    }

    #[test]
    fn inspect_reads_only_header_and_rejects_bad_ranges() {
        let raw = [1_u8, 2, 3, 4];
        let xml = image_element(0, "2:2:1", "UInt8", raw.len(), "colorSpace=\"Gray\"", "");
        let bytes = monolithic(xml, &[&raw]);
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("test.xisf");
        std::fs::write(&path, &bytes).unwrap();
        let info = inspect(&path).unwrap();
        assert_eq!(info.images.len(), 1);
        assert_eq!(info.images[0].width, 2);
        let headers = read_header(&path).unwrap();
        assert!(headers.contains(&("NAXIS1".into(), HeaderValue::Integer(2))));

        let mut truncated = bytes;
        truncated.pop();
        assert!(matches!(
            from_bytes(&truncated),
            Err(XisfError::Malformed(message)) if message.contains("outside the file")
        ));
    }
}
