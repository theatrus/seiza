//! Monolithic XISF writer for 32-bit floating-point images, and for files
//! of several images.
//!
//! The output mirrors the layout the reader supports: an `XISF0100` preamble,
//! a UTF-8 XML header padded to a 4096-byte block boundary, and one attached
//! little-endian planar data block per image, `Float32` or `UInt16`, each
//! optionally compressed and checksummed. FITS-compatible metadata is stored
//! as `FITSKeyword` elements whose values use FITS text conventions, so files
//! round-trip through this crate's reader and load in PixInsight.

use crate::metadata::{ENCODING_PROPERTIES, PIXEL_ATTRIBUTES, SOURCE_ATTRIBUTES};
use crate::xml::escape_attribute;
use crate::{
    CompressionCodec, MAX_HEADER_BYTES, MAX_SAMPLES, SIGNATURE, XisfElement, XisfError,
    XisfMetadata,
};
use quick_xml::escape::escape;
use rayon::prelude::*;
use seiza_fits::{F32ImageData, HeaderValue, ImagePart, ImageSamples, WriteHeaderCard};
use sha2::digest::DynDigest;
use std::collections::BTreeSet;
use std::collections::HashSet;
use std::fmt::Write as _;
use std::io::{BufWriter, Write};
use std::path::Path;

const BLOCK_ALIGNMENT: usize = 4096;
const PIXEL_CHUNK_BYTES: usize = 1024 * 1024;
/// The length of each compressed subblock, as PixInsight uses. Subblocks
/// compress in parallel, and one that would not shrink, such as a byte
/// plane of noise, is stored uncompressed.
const SUBBLOCK_BYTES: usize = 4 << 20;

/// Choices for [`write_f32_image_with_options`] and [`write_images`].
///
/// The default writes an uncompressed block without a checksum, which every
/// XISF reader can open. In a file of several images, compression and the
/// checksum apply to every image's block, and `metadata` and `bounds` to the
/// first image only.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WriteOptions<'a> {
    pub compression: Option<WriteCompression>,
    /// Hash the stored pixel block so readers can detect corruption.
    pub checksum: Option<ChecksumAlgorithm>,
    /// Metadata read from a file, such as [`XisfImage::metadata`], to write
    /// back out along with the new pixels. See [`XisfMetadata`] for what the
    /// writer leaves out.
    ///
    /// [`XisfImage::metadata`]: crate::XisfImage::metadata
    pub metadata: Option<&'a XisfMetadata>,
    /// The representable range to declare, as `(low, high)`. By default the
    /// writer declares the range the samples span.
    ///
    /// Readers such as PixInsight map this range onto their display range
    /// when they load the file, so declare the range the samples are meant
    /// to have: `(0.0, 1.0)` for normalized samples, `(0.0, 65535.0)` for
    /// 16-bit camera counts.
    pub bounds: Option<(f64, f64)>,
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

    fn hasher(self) -> Box<dyn DynDigest> {
        match self {
            Self::Sha1 => Box::new(sha1::Sha1::default()),
            Self::Sha256 => Box::new(sha2::Sha256::default()),
            Self::Sha512 => Box::new(sha2::Sha512::default()),
            Self::Sha3_256 => Box::new(sha3::Sha3_256::default()),
            Self::Sha3_512 => Box::new(sha3::Sha3_512::default()),
        }
    }
}

/// A `Write` sink that hashes what it is given.
struct HashWriter(Box<dyn DynDigest>);

impl Write for HashWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
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
    options: &WriteOptions<'_>,
) -> Result<(), XisfError> {
    let image = ImagePart {
        name: None,
        width,
        height,
        samples: ImageSamples::F32(pixels),
        headers,
    };
    validate_image(&image)?;
    validate_options(options)?;
    write_atomically(path.as_ref(), |writer| {
        write_parts(writer, &[image], options, SUBBLOCK_BYTES)
    })
}

/// Atomically write a monolithic XISF file holding several images, in
/// order, the first being the one readers open by default. Each image's
/// `id` is its [`ImagePart::name`], by which [`crate::open_image_by_id`]
/// finds it, and its samples are `Float32` or `UInt16`. A `Float32` image
/// declares the range its samples span, or, for the first, the
/// [`WriteOptions::bounds`] given.
///
/// The names follow [`seiza_fits::write_images`]: every image after the
/// first needs one and no two may share one, whatever their case. Here a
/// name must also be an XISF identifier: a letter or underscore, then
/// letters, digits and underscores. The completed file is flushed and
/// renamed over `path` only after every block has been written.
pub fn write_images(
    path: impl AsRef<Path>,
    images: &[ImagePart<'_>],
    options: &WriteOptions<'_>,
) -> Result<(), XisfError> {
    validate_images(images)?;
    validate_options(options)?;
    write_atomically(path.as_ref(), |writer| {
        write_parts(writer, images, options, SUBBLOCK_BYTES)
    })
}

/// [`write_images`] to an existing stream. The caller owns flushing and
/// durability.
pub fn write_images_to(
    writer: impl Write,
    images: &[ImagePart<'_>],
    options: &WriteOptions<'_>,
) -> Result<(), XisfError> {
    validate_images(images)?;
    validate_options(options)?;
    write_parts(writer, images, options, SUBBLOCK_BYTES)
}

/// Write a file through `write` under a temporary name beside `path`, then
/// flush, sync and rename it over `path`.
fn write_atomically(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<&mut std::fs::File>) -> Result<(), XisfError>,
) -> Result<(), XisfError> {
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
    write(&mut writer)?;
    writer.flush()?;
    drop(writer);
    seiza_fits::sync_file(temporary.as_file())?;
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
    options: &WriteOptions<'_>,
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
    writer: impl Write,
    (width, height, pixels): (usize, usize, F32ImageData<'_>),
    headers: &[WriteHeaderCard],
    options: &WriteOptions<'_>,
    subblock_bytes: usize,
) -> Result<(), XisfError> {
    let image = ImagePart {
        name: None,
        width,
        height,
        samples: ImageSamples::F32(pixels),
        headers,
    };
    validate_image(&image)?;
    validate_options(options)?;
    write_parts(writer, &[image], options, subblock_bytes)
}

/// Write checked images: the header, then each image's block, then the
/// carried metadata's blocks, every block on an alignment boundary.
fn write_parts(
    mut writer: impl Write,
    images: &[ImagePart<'_>],
    options: &WriteOptions<'_>,
    subblock_bytes: usize,
) -> Result<(), XisfError> {
    let blocks = images
        .iter()
        .map(|image| prepare_block(image, options, subblock_bytes))
        .collect::<Result<Vec<_>, _>>()?;
    let bounds = images
        .iter()
        .enumerate()
        .map(|(index, image)| match image.samples {
            ImageSamples::F32(pixels) => {
                Some(options.bounds.filter(|_| index == 0).unwrap_or_else(|| {
                    let (low, high) = sample_bounds(pixels.samples());
                    (f64::from(low), f64::from(high))
                }))
            }
            // An integer image's range is its type's.
            ImageSamples::U16 { .. } => None,
        })
        .collect::<Vec<_>>();
    let first = &images[0];
    let carried = Carried::new(
        options.metadata,
        (first.width, first.height, first.samples.planes()),
        first.headers,
    )
    .named(first.name.is_some());

    let created = utc_timestamp(std::time::SystemTime::now());
    let mut data_offset = BLOCK_ALIGNMENT;
    let xml = loop {
        let xml = render_xml(images, &bounds, (data_offset, &blocks), &carried, &created);
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

    let mut position = data_offset;
    for (image, block) in images.iter().zip(&blocks) {
        let aligned = position.next_multiple_of(BLOCK_ALIGNMENT);
        writer.write_all(&vec![0; aligned - position])?;
        if let Some(stored) = &block.stored {
            for subblock in stored {
                writer.write_all(subblock)?;
            }
        } else {
            write_samples(&mut writer, image.width, image.height, image.samples)?;
        }
        position = aligned + block.bytes;
    }
    for bytes in carried.blocks() {
        let aligned = position.next_multiple_of(BLOCK_ALIGNMENT);
        writer.write_all(&vec![0; aligned - position])?;
        writer.write_all(bytes)?;
        position = aligned + bytes.len();
    }
    Ok(())
}

fn write_samples(
    mut writer: impl Write,
    width: usize,
    height: usize,
    samples: ImageSamples<'_>,
) -> Result<(), XisfError> {
    let mut byte_buffer = Vec::with_capacity(PIXEL_CHUNK_BYTES);
    match samples {
        ImageSamples::F32(F32ImageData::Mono(samples) | F32ImageData::RgbPlanar(samples)) => {
            write_float_values(&mut writer, samples.iter().copied(), &mut byte_buffer)?;
        }
        ImageSamples::F32(F32ImageData::RgbInterleaved(samples)) => {
            let pixel_count = width * height;
            for channel in 0..3 {
                write_float_values(
                    &mut writer,
                    (0..pixel_count).map(|index| samples[index * 3 + channel]),
                    &mut byte_buffer,
                )?;
            }
        }
        ImageSamples::U16 { samples, .. } => {
            byte_buffer.clear();
            for chunk in samples.chunks(PIXEL_CHUNK_BYTES / 2) {
                byte_buffer.extend(chunk.iter().flat_map(|value| value.to_le_bytes()));
                writer.write_all(&byte_buffer)?;
                byte_buffer.clear();
            }
        }
    }
    Ok(())
}

/// The number of samples `samples` holds, every plane counted.
fn sample_count(samples: ImageSamples<'_>) -> usize {
    match samples {
        ImageSamples::F32(pixels) => pixels.samples().len(),
        ImageSamples::U16 { samples, .. } => samples.len(),
    }
}

/// The bytes of one sample of `samples`.
fn item_bytes(samples: ImageSamples<'_>) -> usize {
    match samples {
        ImageSamples::F32(_) => std::mem::size_of::<f32>(),
        ImageSamples::U16 { .. } => std::mem::size_of::<u16>(),
    }
}

/// The metadata carried from a read into this write, less what the new
/// pixels make false.
#[derive(Default)]
struct Carried {
    image_attributes: Vec<(String, String)>,
    image_elements: Vec<XisfElement>,
    unit_properties: Vec<XisfElement>,
    root_elements: Vec<XisfElement>,
    root_attributes: Vec<(String, String)>,
    /// The source's creation time, to record as `XISF:OriginalCreationTime`.
    original_creation_time: Option<String>,
    /// The codec identifiers of the carried blocks, such as `zstd+sh`.
    codecs: BTreeSet<String>,
    /// The checksum algorithms of the carried blocks.
    checksums: BTreeSet<String>,
}

impl Carried {
    fn new(
        metadata: Option<&XisfMetadata>,
        (width, height, planes): (usize, usize, usize),
        headers: &[WriteHeaderCard],
    ) -> Self {
        let Some(metadata) = metadata else {
            return Self::default();
        };
        let (source_width, source_height, source_planes) = metadata.source_geometry;
        let resized = (width, height) != (source_width, source_height);
        let replanned = planes != source_planes;
        let image_elements = metadata
            .image_elements
            .iter()
            .filter(|element| match element.local_name() {
                "FITSKeyword" => element.attribute("name").is_some_and(|name| {
                    let name = name.trim();
                    // Scaling and structure cards describe the old samples,
                    // and the caller's cards replace carried ones of the same
                    // name, except commentary cards, which add to them.
                    !is_structural_keyword(name)
                        && (is_commentary_keyword(name)
                            || !headers.iter().any(|header| header.keyword() == name))
                        && !(resized && is_wcs_keyword(name))
                        && !(replanned && is_bayer_keyword(name))
                }),
                // A solution describes the old geometry; the specification
                // requires dropping it when the geometry changes.
                "Property" => {
                    !(resized
                        && element.attribute("id").is_some_and(|id| {
                            id.starts_with("AstrometricSolution:")
                                || id.starts_with("PCL:AstrometricSolution:")
                        }))
                }
                "ColorFilterArray" => !replanned,
                _ => true,
            })
            .cloned()
            .collect::<Vec<_>>();
        let property_id =
            |element: &XisfElement| element.attribute("id").unwrap_or("").trim().to_string();
        let original_creation_time = metadata
            .unit_properties
            .iter()
            .find(|element| property_id(element) == "XISF:OriginalCreationTime")
            .or_else(|| {
                metadata
                    .unit_properties
                    .iter()
                    .find(|element| property_id(element) == "XISF:CreationTime")
            })
            // PixInsight writes the time as String text rather than a
            // TimePoint value attribute.
            .and_then(|element| {
                element
                    .attribute("value")
                    .or(Some(element.text.as_str()))
                    .map(str::trim)
                    .filter(|time| !time.is_empty())
            })
            .map(str::to_string);
        let unit_properties = metadata
            .unit_properties
            .iter()
            .filter(|element| {
                let id = property_id(element);
                !ENCODING_PROPERTIES.contains(&id.as_str()) && id != "XISF:OriginalCreationTime"
            })
            .cloned()
            .collect();
        Self {
            image_attributes: metadata
                .image_attributes
                .iter()
                .filter(|(name, _)| {
                    !PIXEL_ATTRIBUTES.contains(&name.as_str())
                        && !SOURCE_ATTRIBUTES.contains(&name.as_str())
                })
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
            codecs: BTreeSet::new(),
            checksums: BTreeSet::new(),
            image_elements,
            unit_properties,
            root_elements: metadata.root_elements.clone(),
            root_attributes: metadata
                .root_attributes
                .iter()
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
            original_creation_time,
        }
        .with_block_encodings()
    }

    /// Leave out a carried `id` when the image is given a name of its own.
    fn named(mut self, named: bool) -> Self {
        if named {
            self.image_attributes.retain(|(name, _)| name != "id");
        }
        self
    }

    /// Record the codecs and checksum algorithms of every carried block,
    /// wherever it is stored.
    fn with_block_encodings(mut self) -> Self {
        fn visit(
            element: &XisfElement,
            codecs: &mut BTreeSet<String>,
            checksums: &mut BTreeSet<String>,
        ) {
            let first = |value: &str| value.split(':').next().unwrap_or("").trim().to_string();
            if let Some(compression) = element.attribute("compression") {
                codecs.insert(first(compression));
            }
            if let Some(checksum) = element.attribute("checksum") {
                checksums.insert(first(checksum));
            }
            for child in &element.children {
                visit(child, codecs, checksums);
            }
        }
        let (mut codecs, mut checksums) = (BTreeSet::new(), BTreeSet::new());
        for element in self
            .image_elements
            .iter()
            .chain(&self.unit_properties)
            .chain(&self.root_elements)
        {
            visit(element, &mut codecs, &mut checksums);
        }
        self.codecs = codecs;
        self.checksums = checksums;
        self
    }

    /// The loaded blocks of the carried elements, in document order.
    fn blocks(&self) -> Vec<&[u8]> {
        fn collect<'e>(element: &'e XisfElement, blocks: &mut Vec<&'e [u8]>) {
            if let Some(block) = &element.block {
                blocks.push(block);
            }
            for child in &element.children {
                collect(child, blocks);
            }
        }
        let mut blocks = Vec::new();
        for element in self
            .image_elements
            .iter()
            .chain(&self.unit_properties)
            .chain(&self.root_elements)
        {
            collect(element, &mut blocks);
        }
        blocks
    }
}

/// The pixel block as it will be stored, with the `Image` attributes and
/// `Metadata` properties that describe it.
struct PreparedBlock {
    /// The stored bytes of each subblock, or `None` to stream the samples
    /// uncompressed.
    stored: Option<Vec<Vec<u8>>>,
    bytes: usize,
    attributes: String,
    /// The codec identifier, such as `zstd+sh`, when the block is compressed.
    codec: Option<String>,
    /// The abstract compression level of `codec`.
    level: Option<i32>,
    checksum: Option<ChecksumAlgorithm>,
}

fn validate_options(options: &WriteOptions<'_>) -> Result<(), XisfError> {
    if let Some((low, high)) = options.bounds
        && !(low.is_finite() && high.is_finite() && low < high)
    {
        return Err(XisfError::Malformed(format!(
            "bounds {low}:{high} do not describe a range"
        )));
    }
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
    image: &ImagePart<'_>,
    options: &WriteOptions<'_>,
    subblock_bytes: usize,
) -> Result<PreparedBlock, XisfError> {
    let &ImagePart {
        width,
        height,
        samples,
        ..
    } = image;
    let item_bytes = item_bytes(samples);
    let mut block = PreparedBlock {
        stored: None,
        bytes: sample_count(samples) * item_bytes,
        attributes: String::new(),
        codec: None,
        level: None,
        checksum: options.checksum,
    };
    if let Some(compression) = options.compression {
        let input = block_bytes(width * height, samples, compression.byte_shuffle);
        let subblocks = input
            .par_chunks(subblock_bytes)
            .map(|chunk| {
                // Revision 2: a subblock that would not shrink is stored
                // uncompressed, and readers copy a subblock whose stored
                // and uncompressed lengths are equal.
                let compressed = compress(chunk, compression);
                let stored = if compressed.len() >= chunk.len() {
                    chunk.to_vec()
                } else {
                    compressed
                };
                (stored, chunk.len())
            })
            .collect::<Vec<_>>();
        let compressed_bytes = subblocks.iter().map(|(data, _)| data.len()).sum::<usize>();
        // A block that does not shrink is stored as it is.
        if compressed_bytes < input.len() {
            let name = codec_name(compression.codec);
            let codec = if compression.byte_shuffle {
                format!("{name}+sh")
            } else {
                name.to_string()
            };
            let _ = write!(block.attributes, " compression=\"{codec}:{}", input.len());
            if compression.byte_shuffle {
                let _ = write!(block.attributes, ":{item_bytes}");
            }
            block.attributes.push('"');
            if subblocks.len() > 1 {
                let list = subblocks
                    .iter()
                    .map(|(data, uncompressed)| format!("{},{uncompressed}", data.len()))
                    .collect::<Vec<_>>()
                    .join(":");
                let _ = write!(block.attributes, " subblocks=\"{list}\"");
            }
            block.codec = Some(codec);
            block.level = abstract_compression_level(compression);
            drop(input);
            block.stored = Some(subblocks.into_iter().map(|(data, _)| data).collect());
        } else {
            drop(subblocks);
            let raw = if compression.byte_shuffle {
                block_bytes(width * height, samples, false)
            } else {
                input
            };
            block.stored = Some(vec![raw]);
        }
    }
    if let Some(checksum) = options.checksum {
        let digest = match &block.stored {
            Some(stored) => {
                let mut hasher = checksum.hasher();
                for subblock in stored {
                    hasher.update(subblock);
                }
                crate::lowercase_hex(&hasher.finalize())
            }
            // Uncompressed samples are hashed as they stream, rather than
            // gathered into one buffer.
            None => {
                let mut hasher = HashWriter(checksum.hasher());
                write_samples(&mut hasher, width, height, samples)?;
                crate::lowercase_hex(&hasher.0.finalize())
            }
        };
        let _ = write!(
            block.attributes,
            " checksum=\"{}:{digest}\"",
            checksum.name()
        );
    }
    if let Some(stored) = &block.stored {
        block.bytes = stored.iter().map(Vec::len).sum();
    }
    Ok(block)
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

/// Bytes one task fills in [`block_bytes`].
const SHUFFLE_TASK_BYTES: usize = 1 << 20;

/// The samples of `samples` in planar order as little-endian bytes, built
/// in parallel. With `byte_shuffle`, they follow the XISF byte shuffle:
/// every sample's first byte, then every second byte, and so on.
fn block_bytes(pixel_count: usize, samples: ImageSamples<'_>, byte_shuffle: bool) -> Vec<u8> {
    match samples {
        ImageSamples::F32(pixels) => sample_bytes(
            pixels.samples().len(),
            |index| match pixels {
                F32ImageData::Mono(samples) | F32ImageData::RgbPlanar(samples) => {
                    samples[index].to_le_bytes()
                }
                F32ImageData::RgbInterleaved(samples) => {
                    samples[index % pixel_count * 3 + index / pixel_count].to_le_bytes()
                }
            },
            byte_shuffle,
        ),
        ImageSamples::U16 { samples, .. } => sample_bytes(
            samples.len(),
            |index| samples[index].to_le_bytes(),
            byte_shuffle,
        ),
    }
}

/// [`block_bytes`] for `count` samples of `N` bytes each, the bytes of the
/// sample at each planar index given by `sample`.
fn sample_bytes<const N: usize>(
    count: usize,
    sample: impl Fn(usize) -> [u8; N] + Sync,
    byte_shuffle: bool,
) -> Vec<u8> {
    let item_bytes = N;
    let mut bytes = vec![0_u8; count * N];
    if byte_shuffle {
        bytes
            .par_chunks_mut(count)
            .enumerate()
            .for_each(|(lane, lane_bytes)| {
                lane_bytes
                    .par_chunks_mut(SHUFFLE_TASK_BYTES)
                    .enumerate()
                    .for_each(|(task, chunk)| {
                        let first = task * SHUFFLE_TASK_BYTES;
                        for (offset, byte) in chunk.iter_mut().enumerate() {
                            *byte = sample(first + offset)[lane];
                        }
                    });
            });
    } else {
        bytes
            .par_chunks_mut(SHUFFLE_TASK_BYTES)
            .enumerate()
            .for_each(|(task, chunk)| {
                let first = task * SHUFFLE_TASK_BYTES / item_bytes;
                for (offset, item) in chunk.chunks_exact_mut(item_bytes).enumerate() {
                    item.copy_from_slice(&sample(first + offset));
                }
            });
    }
    bytes
}

/// The XISF byte shuffle: all first bytes of each item, then all second
/// bytes, and so on. Trailing bytes that do not form a whole item stay last.
#[cfg(test)]
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
    images: &[ImagePart<'_>],
    bounds: &[Option<(f64, f64)>],
    (data_offset, blocks): (usize, &[PreparedBlock]),
    carried: &Carried,
    created: &str,
) -> String {
    // Each image's block follows the last on an alignment boundary.
    let mut position = data_offset;
    let offsets = blocks
        .iter()
        .map(|block| {
            position = position.next_multiple_of(BLOCK_ALIGNMENT);
            let offset = position;
            position += block.bytes;
            offset
        })
        .collect::<Vec<_>>();
    // Carried blocks follow the pixel blocks, in document order.
    let mut locations = carried
        .blocks()
        .into_iter()
        .map(|bytes| {
            // Every attached block starts at a multiple of the declared
            // XISF:BlockAlignmentSize.
            position = position.next_multiple_of(BLOCK_ALIGNMENT);
            let location = format!("attachment:{position}:{}", bytes.len());
            position += bytes.len();
            location
        })
        .collect::<Vec<_>>()
        .into_iter();
    let mut xml = String::with_capacity(1024);
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
    xml.push_str(
        "<xisf version=\"1.0\" xmlns=\"http://www.pixinsight.com/xisf\" \
         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" \
         xsi:schemaLocation=\"http://www.pixinsight.com/xisf \
         http://pixinsight.com/xisf/xisf-1.0.xsd\"",
    );
    for (name, value) in &carried.root_attributes {
        let _ = write!(xml, " {name}=\"{}\"", escape_attribute(value));
    }
    xml.push('>');
    for (index, image) in images.iter().enumerate() {
        let (block, offset) = (&blocks[index], offsets[index]);
        let planes = image.samples.planes();
        let color_space = if planes == 3 { "RGB" } else { "Gray" };
        let sample_format = match image.samples {
            ImageSamples::F32(_) => "Float32",
            ImageSamples::U16 { .. } => "UInt16",
        };
        let _ = write!(
            xml,
            "<Image geometry=\"{}:{}:{planes}\" sampleFormat=\"{sample_format}\"",
            image.width, image.height
        );
        if let Some((low, high)) = bounds[index] {
            let _ = write!(xml, " bounds=\"{low}:{high}\"");
        }
        let _ = write!(
            xml,
            " colorSpace=\"{color_space}\" pixelStorage=\"Planar\" \
             location=\"attachment:{offset}:{}\"{}",
            block.bytes, block.attributes
        );
        if let Some(name) = image.name {
            let _ = write!(xml, " id=\"{}\"", escape_attribute(name));
        }
        if index == 0 {
            for (name, value) in &carried.image_attributes {
                let _ = write!(xml, " {name}=\"{}\"", escape_attribute(value));
            }
        }
        xml.push('>');
        if index == 0 {
            for element in &carried.image_elements {
                element.write(&mut xml, &mut locations);
            }
        }
        for header in image.headers {
            // name, value and comment are all mandatory attributes.
            let _ = write!(
                xml,
                "<FITSKeyword name=\"{}\" value=\"{}\" comment=\"{}\"/>",
                escape_attribute(header.keyword()),
                escape_attribute(&fits_value_text(header.value())),
                escape_attribute(header.comment())
            );
        }
        xml.push_str("</Image>");
    }
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
    // The codecs and checksum algorithms of every block in the unit, the
    // carried ones included.
    let codecs = blocks
        .iter()
        .filter_map(|block| block.codec.clone())
        .chain(carried.codecs.iter().cloned())
        .collect::<BTreeSet<_>>();
    if !codecs.is_empty() {
        let list = codecs.into_iter().collect::<Vec<_>>().join(",");
        let _ = write!(
            xml,
            "<Property id=\"XISF:CompressionCodecs\" type=\"String\">{}</Property>",
            escape(list.as_str())
        );
    }
    // One level only describes the unit when the pixel blocks, compressed
    // alike, are its only compressed blocks.
    if let Some(level) = blocks
        .iter()
        .find_map(|block| block.level)
        .filter(|_| carried.codecs.is_empty())
    {
        let _ = write!(
            xml,
            "<Property id=\"XISF:CompressionLevel\" type=\"Int32\" value=\"{level}\"/>"
        );
    }
    let checksums = blocks
        .iter()
        .filter_map(|block| block.checksum)
        .map(|checksum| checksum.name().to_string())
        .chain(carried.checksums.iter().cloned())
        .collect::<BTreeSet<_>>();
    if !checksums.is_empty() {
        let list = checksums.into_iter().collect::<Vec<_>>().join(",");
        let _ = write!(
            xml,
            "<Property id=\"XISF:ChecksumAlgorithms\" type=\"String\">{}</Property>",
            escape(list.as_str())
        );
    }
    if let Some(time) = &carried.original_creation_time {
        let _ = write!(
            xml,
            "<Property id=\"XISF:OriginalCreationTime\" type=\"TimePoint\" value=\"{}\"/>",
            escape_attribute(time)
        );
    }
    for element in &carried.unit_properties {
        element.write(&mut xml, &mut locations);
    }
    xml.push_str("</Metadata>");
    for element in &carried.root_elements {
        element.write(&mut xml, &mut locations);
    }
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

fn validate_image(image: &ImagePart<'_>) -> Result<(), XisfError> {
    let &ImagePart {
        width,
        height,
        samples,
        headers,
        ..
    } = image;
    if let ImageSamples::U16 { planes, .. } = samples
        && planes != 1
        && planes != 3
    {
        return Err(XisfError::Malformed(format!(
            "{planes} planes of 16-bit samples; write 1 or 3"
        )));
    }
    let expected = width
        .checked_mul(height)
        .and_then(|count| count.checked_mul(samples.planes()))
        .ok_or_else(|| XisfError::Malformed("image dimensions overflow".into()))?;
    if width == 0 || height == 0 || expected > MAX_SAMPLES {
        return Err(XisfError::Malformed("implausible image dimensions".into()));
    }
    if sample_count(samples) != expected {
        return Err(XisfError::Malformed(format!(
            "pixel buffer has {} samples; expected {expected}",
            sample_count(samples)
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
        if !is_commentary_keyword(header.keyword()) && !keywords.insert(header.keyword()) {
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

/// Check every image of a file of several, and their names: each image
/// after the first has one, none is repeated whatever its case, and each is
/// an XISF identifier.
fn validate_images(images: &[ImagePart<'_>]) -> Result<(), XisfError> {
    if images.is_empty() {
        return Err(XisfError::Malformed("no images to write".into()));
    }
    let mut names = HashSet::with_capacity(images.len());
    for (index, image) in images.iter().enumerate() {
        validate_image(image)?;
        if image
            .headers
            .iter()
            .any(|header| header.keyword() == "EXTNAME")
        {
            return Err(XisfError::Malformed(
                "EXTNAME is the image's name; set ImagePart::name".into(),
            ));
        }
        let Some(name) = image.name else {
            if index == 0 {
                continue;
            }
            return Err(XisfError::Malformed(format!(
                "image {index} has no name; every image after the first needs one"
            )));
        };
        let mut characters = name.chars();
        let identifier = characters
            .next()
            .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
            && characters.all(|rest| rest.is_ascii_alphanumeric() || rest == '_');
        if !identifier {
            return Err(XisfError::Malformed(format!(
                "image name {name:?} is not an XISF identifier"
            )));
        }
        if !names.insert(name.to_ascii_uppercase()) {
            return Err(XisfError::Malformed(format!(
                "two images are named {name:?}"
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

/// Keywords that may repeat, each card adding a line.
fn is_commentary_keyword(keyword: &str) -> bool {
    matches!(keyword, "HISTORY" | "COMMENT" | "")
}

/// FITS World Coordinate System keywords, including alternate descriptions
/// (a trailing letter) and SIP distortion coefficients.
fn is_wcs_keyword(keyword: &str) -> bool {
    let keyword = keyword.to_ascii_uppercase();
    let base = keyword
        .strip_suffix(|last: char| last.is_ascii_uppercase())
        .filter(|base| base.ends_with(|last: char| last.is_ascii_digit()))
        .unwrap_or(&keyword);
    let digits = |text: &str| !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit());
    let indexed = [
        "CRPIX", "CRVAL", "CDELT", "CROTA", "CTYPE", "CUNIT", "CRDER", "CSYER",
    ]
    .iter()
    .any(|prefix| base.strip_prefix(prefix).is_some_and(digits));
    let matrix = ["CD", "PC", "PV", "PS", "A_", "B_", "AP_", "BP_"]
        .iter()
        .any(|prefix| {
            base.strip_prefix(prefix)
                .and_then(|rest| rest.split_once('_'))
                .is_some_and(|(row, column)| digits(row) && digits(column))
        });
    indexed
        || matrix
        || matches!(
            base,
            "LONPOLE"
                | "LATPOLE"
                | "WCSAXES"
                | "WCSNAME"
                | "A_ORDER"
                | "B_ORDER"
                | "AP_ORDER"
                | "BP_ORDER"
                | "A_DMAX"
                | "B_DMAX"
                | "PLTSOLVD"
        )
}

/// Keywords that describe a color filter array mosaic.
fn is_bayer_keyword(keyword: &str) -> bool {
    matches!(
        keyword,
        "BAYERPAT" | "XBAYROFF" | "YBAYROFF" | "BAYOFFX" | "BAYOFFY" | "COLORTYP"
    )
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

    fn write_to_memory(
        values: &[f32],
        options: &WriteOptions<'_>,
        subblock_bytes: usize,
    ) -> Vec<u8> {
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
                    metadata: None,
                    bounds: None,
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
            metadata: None,
            bounds: None,
        };
        let encoded = write_to_memory(&values, &options, SUBBLOCK_BYTES);
        let read = crate::read_image_from_bytes(&encoded, 0).unwrap();
        assert!(read.info.compression.is_none());
        assert!(!String::from_utf8_lossy(&encoded[..4096]).contains("CompressionCodecs"));
    }

    #[test]
    fn stores_incompressible_subblocks_uncompressed() {
        // Smooth values with random low mantissa bits, as in a noisy image.
        // After the shuffle, the low byte planes are random and the high
        // ones constant.
        let mut state = 0x2545_f491_u32;
        let values = (0..16 * 8 * 3)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                f32::from_bits(0.25_f32.to_bits() | (state & 0xffff))
            })
            .collect::<Vec<_>>();
        let expected = planar_from(&values, 16 * 8);
        let raw = expected
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let input = shuffle(&raw, 4);
        for codec in [
            CompressionCodec::Zlib,
            CompressionCodec::Lz4,
            CompressionCodec::Zstd,
        ] {
            let options = WriteOptions {
                compression: Some(WriteCompression {
                    codec,
                    byte_shuffle: true,
                    level: None,
                }),
                checksum: Some(ChecksumAlgorithm::Sha256),
                metadata: None,
                bounds: None,
            };
            let encoded = write_to_memory(&values, &options, 128);
            let read = crate::read_image_from_bytes(&encoded, 0).unwrap();
            let Pixels::F32(actual) = &read.image.pixels else {
                panic!("writer must emit f32 pixels");
            };
            assert_eq!(actual, &expected, "{codec:?}");
            let crate::BlockLocation::Attachment { offset, .. } = read.info.location else {
                panic!("the block must be attached");
            };
            let info = read.info.compression.expect("block must be compressed");
            assert_eq!(info.subblocks.len(), input.len() / 128);
            let (mut position, mut uncompressed) = (offset as usize, 0);
            for &(stored, length) in &info.subblocks {
                let (stored, length) = (stored as usize, length as usize);
                // Every compressed subblock is shorter than its data, and an
                // uncompressed one holds the shuffled bytes as they are.
                assert!(stored <= length, "{codec:?}");
                if stored == length {
                    assert_eq!(
                        &encoded[position..position + stored],
                        &input[uncompressed..uncompressed + length]
                    );
                }
                position += stored;
                uncompressed += length;
            }
            let raw_subblocks = info
                .subblocks
                .iter()
                .filter(|&&(stored, length)| stored == length)
                .count();
            assert!(
                (1..info.subblocks.len()).contains(&raw_subblocks),
                "{codec:?}: {:?}",
                info.subblocks
            );
        }
    }

    #[test]
    fn splits_compressed_blocks_into_four_mib_subblocks() {
        let options = WriteOptions {
            compression: Some(WriteCompression::recommended()),
            checksum: Some(ChecksumAlgorithm::Sha1),
            metadata: None,
            bounds: None,
        };
        // Exactly one subblock's worth is compressed whole; more is split.
        for (height, lengths) in [
            (1024, Vec::new()),
            (1100, vec![SUBBLOCK_BYTES, 1024 * 76 * 4]),
        ] {
            let values = (0..1024 * height)
                .map(|index| 0.25 + (index % 251) as f32 / 1000.0)
                .collect::<Vec<_>>();
            let mut encoded = Vec::new();
            write_image(
                &mut encoded,
                (1024, height, F32ImageData::Mono(&values)),
                &[],
                &options,
                SUBBLOCK_BYTES,
            )
            .unwrap();
            let read = crate::read_image_from_bytes(&encoded, 0).unwrap();
            let Pixels::F32(actual) = &read.image.pixels else {
                panic!("writer must emit f32 pixels");
            };
            assert!(actual == &values, "{height} rows");
            let info = read.info.compression.expect("block must be compressed");
            let uncompressed = info
                .subblocks
                .iter()
                .map(|&(_, length)| length as usize)
                .collect::<Vec<_>>();
            assert_eq!(uncompressed, lengths);
        }
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
                metadata: None,
                bounds: None,
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
    fn block_bytes_match_the_streamed_samples_and_their_shuffle() {
        // More than one task's worth in each byte lane.
        let (width, height) = (1000, 400);
        let values = (0..width * height * 3)
            .map(|index| index as f32 * 0.37)
            .collect::<Vec<_>>();
        let pixel_count = width * height;
        let planar = planar_from(&values, pixel_count);
        for pixels in [
            F32ImageData::Mono(&values),
            F32ImageData::RgbPlanar(&values),
            F32ImageData::RgbInterleaved(&values),
        ] {
            let pixel_count = values.len() / pixels.planes();
            let height = pixel_count / width;
            let samples = ImageSamples::F32(pixels);
            let mut streamed = Vec::new();
            write_samples(&mut streamed, width, height, samples).unwrap();
            assert_eq!(block_bytes(pixel_count, samples, false), streamed);
            assert_eq!(
                block_bytes(pixel_count, samples, true),
                shuffle(&streamed, 4)
            );
        }
        let counts = (0..width * height * 3)
            .map(|index| (index * 7) as u16)
            .collect::<Vec<_>>();
        let samples = ImageSamples::U16 {
            samples: &counts,
            planes: 3,
        };
        let mut streamed = Vec::new();
        write_samples(&mut streamed, width, height, samples).unwrap();
        let little_endian = counts
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(streamed, little_endian);
        assert_eq!(block_bytes(width * height, samples, false), streamed);
        assert_eq!(
            block_bytes(width * height, samples, true),
            shuffle(&streamed, 2)
        );
        let mut interleaved = Vec::new();
        write_samples(
            &mut interleaved,
            width,
            height,
            ImageSamples::F32(F32ImageData::RgbInterleaved(&values)),
        )
        .unwrap();
        let planar_bytes = planar
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(interleaved, planar_bytes);
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

    /// A partial stack's images: a mean, its weight, and 16-bit coverage,
    /// mono and RGB.
    struct Partial {
        mean: Vec<f32>,
        weight: Vec<f32>,
        cover: Vec<u16>,
        rgb_cover: Vec<u16>,
        cards: [WriteHeaderCard; 1],
    }

    impl Partial {
        /// Repeating values, so every block compresses.
        fn new() -> Self {
            let pixels = 64 * 48;
            Self {
                mean: (0..pixels)
                    .map(|index| 100.0 + (index % 64) as f32)
                    .collect(),
                weight: (0..pixels).map(|index| 0.5 + (index % 5) as f32).collect(),
                cover: (0..pixels)
                    .map(|index| (index % 64 * 1000) as u16)
                    .collect(),
                rgb_cover: (0..3 * pixels)
                    .map(|index| (index % 64 * 1000) as u16)
                    .collect(),
                cards: [WriteHeaderCard::new("NCOMBINE", HeaderValue::Integer(40))],
            }
        }

        fn images(&self) -> [ImagePart<'_>; 4] {
            let image = |name, samples, headers| ImagePart {
                name,
                width: 64,
                height: 48,
                samples,
                headers,
            };
            [
                image(
                    None,
                    ImageSamples::F32(F32ImageData::Mono(&self.mean)),
                    &self.cards[..],
                ),
                image(
                    Some("WEIGHT"),
                    ImageSamples::F32(F32ImageData::Mono(&self.weight)),
                    &[],
                ),
                image(
                    Some("COVER"),
                    ImageSamples::U16 {
                        samples: &self.cover,
                        planes: 1,
                    },
                    &[],
                ),
                image(
                    Some("RGB_COVER"),
                    ImageSamples::U16 {
                        samples: &self.rgb_cover,
                        planes: 3,
                    },
                    &[],
                ),
            ]
        }
    }

    #[test]
    fn several_images_round_trip_by_id() {
        let partial = Partial::new();
        let compressed = WriteOptions {
            compression: Some(WriteCompression::recommended()),
            checksum: Some(ChecksumAlgorithm::Sha256),
            ..WriteOptions::default()
        };
        for (options, subblock_bytes) in [
            (WriteOptions::default(), SUBBLOCK_BYTES),
            (compressed, SUBBLOCK_BYTES),
            (compressed, 4096),
        ] {
            let mut encoded = Vec::new();
            write_parts(&mut encoded, &partial.images(), &options, subblock_bytes).unwrap();
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("partial.xisf");
            std::fs::write(&path, &encoded).unwrap();

            // Readers open the first image by default.
            let mean = crate::open(&path).unwrap();
            assert!(matches!(mean.pixels, Pixels::F32(ref values) if values == &partial.mean));
            assert_eq!(mean.header_f64("NCOMBINE"), Some(40.0));
            let weight = crate::open_image_by_id(&path, "WEIGHT").unwrap();
            assert!(matches!(weight.pixels, Pixels::F32(ref values) if values == &partial.weight));
            let cover = crate::open_image_by_id(&path, "COVER").unwrap();
            assert!(matches!(cover.pixels, Pixels::U16(ref values) if values == &partial.cover));
            let rgb = crate::open_image_by_id(&path, "RGB_COVER").unwrap();
            assert_eq!(rgb.planes, 3);
            assert!(matches!(rgb.pixels, Pixels::U16(ref values) if values == &partial.rgb_cover));

            let info = crate::inspect(&path).unwrap();
            let ids = info
                .images
                .iter()
                .map(|image| image.id.as_deref())
                .collect::<Vec<_>>();
            assert_eq!(
                ids,
                [None, Some("WEIGHT"), Some("COVER"), Some("RGB_COVER")]
            );
            for image in &info.images {
                let crate::BlockLocation::Attachment { offset, .. } = image.location else {
                    panic!("writer must attach every block");
                };
                assert_eq!(offset % BLOCK_ALIGNMENT as u64, 0);
                assert_eq!(
                    image.compression.is_some(),
                    options.compression.is_some(),
                    "{options:?}"
                );
            }
            assert_eq!(
                info.images[0]
                    .compression
                    .as_ref()
                    .map(|compression| compression.subblocks.len() > 1),
                options.compression.map(|_| subblock_bytes == 4096)
            );
            assert_eq!(info.images[2].sample_format, crate::SampleFormat::UInt16);
            // Each floating-point image declares the range its samples span.
            assert_eq!(info.images[0].bounds, Some((100.0, 163.0)));
            assert_eq!(info.images[1].bounds, Some((0.5, 4.5)));

            // Every block is checked, the last one's too.
            if options.checksum.is_some() {
                let mut corrupt = encoded.clone();
                let last = corrupt.len() - 1;
                corrupt[last] ^= 0x55;
                std::fs::write(&path, &corrupt).unwrap();
                assert!(crate::open(&path).is_ok());
                assert!(matches!(
                    crate::open_image_by_id(&path, "RGB_COVER"),
                    Err(XisfError::Malformed(message)) if message.contains("checksum mismatch")
                ));
            }
        }
    }

    #[test]
    fn bounds_and_carried_metadata_belong_to_the_first_image() {
        let partial = Partial::new();
        let mut images = partial.images();
        images[0].name = Some("MEAN");
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source.xisf");
        write_images(&source, &images, &WriteOptions::default()).unwrap();
        let metadata = crate::read_image(&source).unwrap().metadata;

        // Carried again under a new name, the source's id gives way to it.
        images[0].name = Some("AGAIN");
        let options = WriteOptions {
            metadata: Some(&metadata),
            bounds: Some((0.0, 65535.0)),
            ..WriteOptions::default()
        };
        let path = directory.path().join("again.xisf");
        write_images(&path, &images, &options).unwrap();
        let info = crate::inspect(&path).unwrap();
        assert_eq!(info.images[0].id.as_deref(), Some("AGAIN"));
        assert_eq!(info.images[0].bounds, Some((0.0, 65535.0)));
        assert_eq!(info.images[1].bounds, Some((0.5, 4.5)));

        // Unnamed, the first image keeps the id it carries.
        images[0].name = None;
        write_images(&path, &images, &options).unwrap();
        let info = crate::inspect(&path).unwrap();
        assert_eq!(info.images[0].id.as_deref(), Some("MEAN"));
    }

    #[test]
    fn several_images_need_distinct_identifiers() {
        let partial = Partial::new();
        let refused = |images: &[ImagePart<'_>]| {
            let mut output = Vec::new();
            let result = write_images_to(&mut output, images, &WriteOptions::default());
            assert_eq!(result.is_err(), output.is_empty());
            result.is_err()
        };
        let images = partial.images();
        assert!(!refused(&images));
        assert!(refused(&[]));
        for name in [
            None,
            Some(""),
            Some("1COVER"),
            Some("COVER-1"),
            Some("weight"),
        ] {
            let mut images = partial.images();
            images[2].name = name;
            assert!(refused(&images), "{name:?}");
        }
        let extname = [WriteHeaderCard::new(
            "EXTNAME",
            HeaderValue::String("X".into()),
        )];
        let mut images = partial.images();
        images[1].headers = &extname;
        assert!(refused(&images));
        let mut images = partial.images();
        images[2].samples = ImageSamples::U16 {
            samples: &partial.rgb_cover[..96],
            planes: 2,
        };
        assert!(refused(&images));
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
