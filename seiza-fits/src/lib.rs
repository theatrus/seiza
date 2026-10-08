//! Fast FITS image reading and linear `f32` writing for astrophotography.
//!
//! Scope: single-image FITS files as written by capture software
//! (N.I.N.A., SGP, ASIAIR, ...) — the primary HDU with a 2D image in
//! BITPIX 8/16/32/64/-32/-64, or, when the primary holds no data, the first
//! image extension, tile-compressed ones included. 16-bit data stays `u16`
//! end to end (no float
//! inflation), statistics come from histograms rather than sorts, and the
//! midtone-transfer-function autostretch matches N.I.N.A.'s. The writer emits
//! primary-HDU mono or RGB float images with validated typed headers and
//! atomic on-disk publication.

mod bayer;
mod compressed;
mod header;
mod writer;

pub use bayer::{
    BayerPattern, RgbImage16, RgbImageF32, RowOrder, debayer_rgb_f32, debayer_rgb_f32_rows,
    debayer_rgb16,
};
pub use header::{HeaderValue, parse_header_value};
pub use seiza_stretch::{
    Statistics, StretchParams, midtones_transfer_function, statistics_u16, stretch_u16_to_u8,
    stretch_u16_to_u16,
};
pub use writer::{
    F32ImageData, WriteHeaderCard, update_header_in_place, write_f32_image, write_f32_image_to,
};

use rayon::prelude::*;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const BLOCK: usize = 2880;
const CARD: usize = 80;
const PIXEL_CHUNK_BYTES: usize = 1024 * 1024;

#[derive(Debug)]
pub enum FitsError {
    Io(std::io::Error),
    NotFits,
    Malformed(String),
    Unsupported(String),
}

impl std::fmt::Display for FitsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::NotFits => write!(f, "not a FITS file"),
            Self::Malformed(what) => write!(f, "malformed FITS: {what}"),
            Self::Unsupported(what) => write!(f, "unsupported FITS: {what}"),
        }
    }
}

impl std::error::Error for FitsError {}

impl From<std::io::Error> for FitsError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Pixel data in its native representation.
///
/// The integer variants hold stored values and cannot mark a blank
/// (undefined) sample: one equal to the header's `BLANK` keeps its stored
/// value here, and [`FitsImage::into_physical_f32`] turns it into NaN. Where
/// decoding already produced floats, blank samples are NaN.
#[derive(Debug, Clone)]
pub enum Pixels {
    /// BITPIX 8 as stored; `BZERO` and `BSCALE` are not applied.
    U8(Vec<u8>),
    /// BITPIX 16 with `BSCALE` 1 and `BZERO` folded in: either the unsigned
    /// camera convention, `BZERO` 32768, or no `BZERO` when no sample is
    /// negative.
    U16(Vec<u16>),
    /// BITPIX 32 as stored; `BZERO` and `BSCALE` are not applied.
    I32(Vec<i32>),
    /// BITPIX -32 as stored, or BITPIX 16 that needs signed or scaled values,
    /// already in physical units.
    F32(Vec<f32>),
    /// BITPIX -64 as stored, or BITPIX 64 already in physical units. An
    /// integer beyond 2^53 does not fit an `f64` exactly and is rounded.
    F64(Vec<f64>),
}

/// A decoded FITS image: pixels plus the parsed header cards.
#[derive(Debug, Clone)]
pub struct FitsImage {
    pub width: usize,
    pub height: usize,
    /// Color planes: 1 for mono/CFA, 3 for planar RGB (NAXIS3 = 3)
    pub planes: usize,
    pub pixels: Pixels,
    /// Header cards in file order (keyword, value). For an image read from
    /// an extension, the extension's cards come first, then the primary
    /// header's; see [`read_header`].
    pub headers: Vec<(String, HeaderValue)>,
}

/// A FITS header with its commentary cards kept.
///
/// [`read_header`] keeps only valued cards. The `HISTORY` and `COMMENT`
/// cards it skips are where processing software records what it did to a
/// frame, so a caller asking "was this calibrated?" needs them too.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FitsHeader {
    /// Valued cards in file order (keyword, value), as [`read_header`]
    /// returns them.
    pub cards: Vec<(String, HeaderValue)>,
    /// The text of each `HISTORY` card in file order, trimmed.
    pub history: Vec<String>,
    /// The text of each `COMMENT` card in file order, trimmed.
    pub comments: Vec<String>,
}

/// Parse a primary header block by block until END. Returns the header and
/// the byte offset where the data section begins.
fn parse_headers(data: &[u8]) -> Result<(FitsHeader, usize), FitsError> {
    if data.len() < BLOCK || &data[0..6] != b"SIMPLE" {
        return Err(FitsError::NotFits);
    }
    parse_header_cards(data)
}

/// Parse header cards block by block until END. Returns the header and the
/// byte offset, from the start of `data`, where the data section begins.
fn parse_header_cards(data: &[u8]) -> Result<(FitsHeader, usize), FitsError> {
    let mut header = FitsHeader::default();
    let mut data_start = None;
    'blocks: for block in 0.. {
        let start = block * BLOCK;
        let Some(block_data) = data.get(start..start + BLOCK) else {
            return Err(FitsError::Malformed("header runs past EOF".into()));
        };
        for card in block_data.chunks_exact(CARD) {
            let keyword = std::str::from_utf8(&card[0..8])
                .map_err(|_| FitsError::Malformed("non-ASCII keyword".into()))?
                .trim_end()
                .to_string();
            if keyword == "END" {
                data_start = Some((block + 1) * BLOCK);
                break 'blocks;
            }
            let commentary = || String::from_utf8_lossy(&card[8..]).trim().to_string();
            match keyword.as_str() {
                "" => continue,
                "HISTORY" => header.history.push(commentary()),
                "COMMENT" => header.comments.push(commentary()),
                // The value starts after the `=` in column 9. The standard
                // puts a space in column 10, but some writers omit it
                // (`EXPTIME =30.5`), and cfitsio reads those too.
                _ if card[8] == b'=' => {
                    let raw = String::from_utf8_lossy(&card[9..]);
                    header.cards.push((keyword, parse_header_value(&raw)));
                }
                // The long-string convention: a string value ending in `&`
                // continues in the next `CONTINUE` card's string.
                "CONTINUE" => {
                    if let Some((_, HeaderValue::String(value))) = header.cards.last_mut()
                        && value.ends_with('&')
                        && let HeaderValue::String(more) =
                            parse_header_value(&String::from_utf8_lossy(&card[8..]))
                    {
                        value.pop();
                        value.push_str(&more);
                    }
                }
                // The ESO convention for keywords longer than eight
                // characters or holding spaces:
                // `HIERARCH ESO DET DIT = 1.5 / comment` names `ESO DET DIT`.
                "HIERARCH" => {
                    let text = String::from_utf8_lossy(&card[8..]);
                    if let Some((keyword, value)) = text.split_once('=')
                        && is_hierarch_keyword(keyword.trim())
                    {
                        header
                            .cards
                            .push((keyword.trim().to_string(), parse_header_value(value)));
                    }
                }
                _ => {}
            }
        }
    }
    let data_start = data_start.ok_or_else(|| FitsError::Malformed("missing END card".into()))?;
    Ok((header, data_start))
}

/// Whether `keyword` can be written as a `HIERARCH` card and read back
/// unchanged: printable ASCII without `=`, `'` or `/`, which would end it
/// early, and without spaces at either end, which a reader trims.
pub(crate) fn is_hierarch_keyword(keyword: &str) -> bool {
    !keyword.is_empty()
        && !keyword.starts_with(' ')
        && !keyword.ends_with(' ')
        && keyword
            .bytes()
            .all(|byte| (b' '..=b'~').contains(&byte) && !b"='/".contains(&byte))
}

/// Whether a header block holds the END card.
fn holds_end_card(block: &[u8]) -> bool {
    block
        .chunks_exact(CARD)
        .any(|card| card.starts_with(b"END") && card[3] == b' ')
}

/// Read complete FITS header blocks and leave the reader at the first byte of
/// the primary data unit. Only the normally small header is retained.
fn read_headers_from(
    reader: &mut impl Read,
    short_first_block_is_not_fits: bool,
) -> Result<(FitsHeader, usize), FitsError> {
    let mut data = Vec::new();
    loop {
        let start = data.len();
        data.resize(start + BLOCK, 0);
        if let Err(error) = reader.read_exact(&mut data[start..]) {
            return Err(match error.kind() {
                std::io::ErrorKind::UnexpectedEof
                    if start == 0 && short_first_block_is_not_fits =>
                {
                    FitsError::NotFits
                }
                std::io::ErrorKind::UnexpectedEof => {
                    FitsError::Malformed("header runs past EOF".into())
                }
                _ => FitsError::Io(error),
            });
        }
        // Stop at a first block that is not FITS rather than read the
        // whole file looking for an END card.
        if start == 0 && !data.starts_with(b"SIMPLE") {
            return Err(FitsError::NotFits);
        }
        if holds_end_card(&data[start..]) {
            break;
        }
    }
    parse_headers(&data)
}

/// Read an extension's header blocks and leave the reader at the first byte
/// of its data unit. Returns the header and its length in bytes, or `None`
/// when no complete, well-formed extension header starts here.
fn read_extension_header(reader: &mut impl Read) -> Result<Option<(FitsHeader, u64)>, FitsError> {
    let mut data = Vec::new();
    loop {
        let start = data.len();
        data.resize(start + BLOCK, 0);
        match reader.read_exact(&mut data[start..]) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(error) => return Err(FitsError::Io(error)),
        }
        if start == 0 && !data.starts_with(b"XTENSION") {
            return Ok(None);
        }
        if holds_end_card(&data[start..]) {
            break;
        }
    }
    Ok(parse_header_cards(&data)
        .ok()
        .map(|(header, length)| (header, length as u64)))
}

fn card_i64(cards: &[(String, HeaderValue)], key: &str) -> Option<i64> {
    cards
        .iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| v.as_i64())
}

/// Random groups mark themselves with `NAXIS1 = 0` and `GROUPS = T`.
fn is_random_groups(cards: &[(String, HeaderValue)]) -> bool {
    card_i64(cards, "NAXIS1") == Some(0)
        && cards
            .iter()
            .any(|(key, value)| key == "GROUPS" && value.as_bool() == Some(true))
}

/// The lengths of axes 1 to `NAXIS`, or `None` when one is missing or
/// negative.
fn axis_lengths(cards: &[(String, HeaderValue)]) -> Option<Vec<u64>> {
    let naxis = card_i64(cards, "NAXIS").unwrap_or(0).clamp(0, 999);
    (1..=naxis)
        .map(|axis| {
            card_i64(cards, &format!("NAXIS{axis}")).and_then(|length| u64::try_from(length).ok())
        })
        .collect()
}

/// Whether a primary header declares no data array: `NAXIS` 0 or an axis of
/// length 0, short of random groups.
fn declares_no_data(cards: &[(String, HeaderValue)]) -> bool {
    !is_random_groups(cards)
        && axis_lengths(cards).is_some_and(|axes| axes.is_empty() || axes.contains(&0))
}

/// Whether an extension header is an image extension with two or more axes,
/// none of them empty.
fn holds_image(cards: &[(String, HeaderValue)]) -> bool {
    cards
        .iter()
        .find(|(key, _)| key == "XTENSION")
        .and_then(|(_, value)| value.as_str())
        == Some("IMAGE")
        && axis_lengths(cards).is_some_and(|axes| axes.len() >= 2 && !axes.contains(&0))
}

/// The length of an HDU's data unit without its padding:
/// `|BITPIX| / 8 × GCOUNT × (PCOUNT + NAXIS1 × … × NAXISn)`, with `NAXIS1`
/// left out of the product for random groups.
fn data_unit_bytes(cards: &[(String, HeaderValue)]) -> Option<u64> {
    let axes = axis_lengths(cards)?;
    if axes.is_empty() {
        return Some(0);
    }
    let skip = usize::from(is_random_groups(cards));
    let elements = axes[skip..]
        .iter()
        .try_fold(1_u64, |product, &length| product.checked_mul(length))?;
    let bytes = card_i64(cards, "BITPIX")?.unsigned_abs() / 8;
    let pcount = u64::try_from(card_i64(cards, "PCOUNT").unwrap_or(0)).ok()?;
    let gcount = u64::try_from(card_i64(cards, "GCOUNT").unwrap_or(1)).ok()?;
    pcount
        .checked_add(elements)?
        .checked_mul(gcount)?
        .checked_mul(bytes)
}

/// Cards that describe the primary HDU's own data array, or its place in the
/// file, and so would mislead if read as the extension image's.
fn describes_primary_data(keyword: &str) -> bool {
    keyword.starts_with("NAXIS")
        || matches!(
            keyword,
            "SIMPLE"
                | "EXTEND"
                | "BITPIX"
                | "PCOUNT"
                | "GCOUNT"
                | "GROUPS"
                | "BZERO"
                | "BSCALE"
                | "BLANK"
                | "CHECKSUM"
                | "DATASUM"
        )
}

/// An image extension's header followed by the primary header's cards, as
/// in the `INHERIT` convention: a lookup finds the extension's own values
/// (its WCS, its scaling) first and still sees file-wide metadata such as
/// `DATE-OBS` that the primary holds.
///
/// The extension's `XTENSION`, `PCOUNT` and `GCOUNT` are left out, and so
/// are the primary's cards that describe its own (empty) data array and
/// those the extension already has. An extension with `INHERIT = F` gets
/// nothing from the primary.
fn inherit_primary(extension: FitsHeader, primary: FitsHeader) -> FitsHeader {
    let inherit = card_value(&extension.cards, "INHERIT").and_then(HeaderValue::as_bool);
    let mut header = FitsHeader {
        cards: extension
            .cards
            .into_iter()
            .filter(|(key, _)| !matches!(key.as_str(), "XTENSION" | "PCOUNT" | "GCOUNT"))
            .collect(),
        history: extension.history,
        comments: extension.comments,
    };
    if inherit == Some(false) {
        return header;
    }
    let own: std::collections::HashSet<String> =
        header.cards.iter().map(|(key, _)| key.clone()).collect();
    header.cards.extend(
        primary
            .cards
            .into_iter()
            .filter(|(key, _)| !describes_primary_data(key) && !own.contains(key)),
    );
    header.history.extend(primary.history);
    header.comments.extend(primary.comments);
    header
}

fn card_value<'a>(cards: &'a [(String, HeaderValue)], key: &str) -> Option<&'a HeaderValue> {
    cards.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

/// The header of the HDU whose image this crate reads, and where that HDU's
/// header and data start in the file.
struct ImageHdu {
    header: FitsHeader,
    header_start: u64,
    data_start: u64,
    /// For a tile-compressed image, the cards of the table that holds it.
    table: Option<Vec<(String, HeaderValue)>>,
}

/// Find the HDU whose image this crate reads and leave the reader at the
/// first byte of its data.
///
/// That is the primary HDU, unless its header declares no data. Then it is
/// the first image extension with two or more axes, plain or tile
/// compressed, its header combined with the primary's as
/// [`inherit_primary`] describes. Other tables are skipped.
/// When no such extension follows, or the search meets anything but a
/// well-formed extension header, the primary is returned, so callers report
/// on it as they did before extensions were read.
fn locate_image<R: Read + Seek>(
    reader: &mut R,
    short_first_block_is_not_fits: bool,
) -> Result<ImageHdu, FitsError> {
    let (primary, data_start) = read_headers_from(reader, short_first_block_is_not_fits)?;
    let data_start = data_start as u64;
    if !declares_no_data(&primary.cards) {
        return Ok(ImageHdu {
            header: primary,
            header_start: 0,
            data_start,
            table: None,
        });
    }
    // A primary that declares no data has an empty data unit.
    let mut offset = data_start;
    loop {
        reader.seek(SeekFrom::Start(offset))?;
        let Some((extension, header_bytes)) = read_extension_header(reader)? else {
            break;
        };
        let extension_data = offset + header_bytes;
        if holds_image(&extension.cards) {
            return Ok(ImageHdu {
                header: inherit_primary(extension, primary),
                header_start: offset,
                data_start: extension_data,
                table: None,
            });
        }
        if compressed::holds_compressed_image(&extension.cards) {
            let table = extension.cards.clone();
            return Ok(ImageHdu {
                header: inherit_primary(compressed::image_header(extension), primary),
                header_start: offset,
                data_start: extension_data,
                table: Some(table),
            });
        }
        let Some(next) = data_unit_bytes(&extension.cards)
            .and_then(|bytes| bytes.checked_next_multiple_of(BLOCK as u64))
            .and_then(|bytes| extension_data.checked_add(bytes))
        else {
            break;
        };
        offset = next;
    }
    reader.seek(SeekFrom::Start(data_start))?;
    Ok(ImageHdu {
        header: primary,
        header_start: 0,
        data_start,
        table: None,
    })
}

/// Read only the header cards of a FITS file, without touching the pixel
/// data — cheap metadata probes on large files.
///
/// When the primary HDU holds no data and an image extension follows, these
/// are the extension's cards, then the primary's, as [`FitsImage::open`]
/// reads them: the extension's `XTENSION`, `PCOUNT` and `GCOUNT` and the
/// primary's structural and scaling cards are left out, and a primary card
/// whose keyword the extension also has gives way to it. An extension with
/// `INHERIT = F` keeps the primary's cards out. A tile-compressed image's
/// cards are those of the image it holds, rebuilt from its table's `Z`
/// keywords as the FITS standard and astropy rebuild them.
pub fn read_header(path: &Path) -> Result<Vec<(String, HeaderValue)>, FitsError> {
    read_header_with_commentary(path).map(|header| header.cards)
}

/// Read a FITS file's header cards together with its `HISTORY` and
/// `COMMENT` text, without touching the pixel data. The cards are those of
/// [`read_header`], and the commentary is combined the same way.
pub fn read_header_with_commentary(path: &Path) -> Result<FitsHeader, FitsError> {
    let mut file = std::fs::File::open(path)?;
    locate_image(&mut file, false).map(|hdu| hdu.header)
}

/// Where the header of the HDU that [`FitsImage::open`] reads starts, and
/// whether that HDU is a tile-compressed image's table.
pub(crate) fn image_header_offset<R: Read + Seek>(
    reader: &mut R,
) -> Result<(u64, bool), FitsError> {
    locate_image(reader, true).map(|hdu| (hdu.header_start, hdu.table.is_some()))
}

/// The `BLANK` card of integer data. The standard defines it for integer
/// BITPIX only, with an integer value; astropy and cfitsio both ignore one
/// written as a float, such as `-32768.0`, and so does this crate.
fn integer_blank(cards: &[(String, HeaderValue)], bitpix: i64) -> Option<i64> {
    match card_value(cards, "BLANK") {
        Some(HeaderValue::Integer(blank)) if bitpix > 0 => Some(*blank),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
struct ImageSpec {
    width: usize,
    height: usize,
    planes: usize,
    count: usize,
    bitpix: i64,
    bzero: f64,
    bscale: f64,
    /// The stored value of an undefined sample, for integer data.
    blank: Option<i64>,
}

impl ImageSpec {
    fn from_headers(headers: &[(String, HeaderValue)]) -> Result<Self, FitsError> {
        let header_i64 = |key: &str| -> Option<i64> {
            headers
                .iter()
                .find(|(k, _)| k == key)
                .and_then(|(_, v)| v.as_i64())
        };
        let header_f64 = |key: &str| -> Option<f64> {
            headers
                .iter()
                .find(|(k, _)| k == key)
                .and_then(|(_, v)| v.as_f64())
        };

        let bitpix =
            header_i64("BITPIX").ok_or_else(|| FitsError::Malformed("missing BITPIX".into()))?;
        if !matches!(bitpix, 8 | 16 | 32 | 64 | -32 | -64) {
            return Err(FitsError::Unsupported(format!("BITPIX {bitpix}")));
        }
        let naxis = header_i64("NAXIS").unwrap_or(0);
        if naxis < 2 {
            return Err(FitsError::Unsupported(format!(
                "NAXIS {naxis} (need a 2D image)"
            )));
        }
        if header_i64("NAXIS1") == Some(0)
            && headers
                .iter()
                .any(|(key, value)| key == "GROUPS" && value.as_bool() == Some(true))
        {
            return Err(FitsError::Unsupported("random groups".into()));
        }
        let width = header_i64("NAXIS1")
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| FitsError::Malformed("missing or invalid NAXIS1".into()))?;
        let height = header_i64("NAXIS2")
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| FitsError::Malformed("missing or invalid NAXIS2".into()))?;
        // Planar color cubes (Siril and friends write RGB as NAXIS3 = 3);
        // planes beyond the third are ignored.
        let planes = if naxis >= 3 {
            header_i64("NAXIS3")
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(1)
                .clamp(1, 3)
        } else {
            1
        };
        let count = width
            .checked_mul(height)
            .and_then(|value| value.checked_mul(planes))
            .ok_or_else(|| FitsError::Malformed("implausible dimensions".into()))?;
        if count == 0 || count > 2_000_000_000 {
            return Err(FitsError::Malformed("implausible dimensions".into()));
        }

        Ok(Self {
            width,
            height,
            planes,
            count,
            bitpix,
            bzero: header_f64("BZERO").unwrap_or(0.0),
            bscale: header_f64("BSCALE").unwrap_or(1.0),
            blank: integer_blank(headers, bitpix),
        })
    }

    fn payload_bytes(self) -> u64 {
        let bytes_per_pixel = match self.bitpix {
            8 => 1,
            16 => 2,
            32 | -32 => 4,
            64 | -64 => 8,
            _ => unreachable!("ImageSpec validates BITPIX"),
        };
        self.count as u64 * bytes_per_pixel
    }
}

fn read_payload_exact(reader: &mut impl Read, buffer: &mut [u8]) -> Result<(), FitsError> {
    reader.read_exact(buffer).map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            FitsError::Malformed("data runs past EOF".into())
        } else {
            FitsError::Io(error)
        }
    })
}

fn read_payload_chunks(
    reader: &mut impl Read,
    count: usize,
    bytes_per_pixel: usize,
    mut decode: impl FnMut(&[u8]),
) -> Result<(), FitsError> {
    let samples_per_chunk = (PIXEL_CHUNK_BYTES / bytes_per_pixel).max(1);
    let mut buffer = vec![0; count.min(samples_per_chunk) * bytes_per_pixel];
    let mut remaining = count;
    while remaining != 0 {
        let samples = remaining.min(samples_per_chunk);
        let bytes = samples * bytes_per_pixel;
        read_payload_exact(reader, &mut buffer[..bytes])?;
        decode(&buffer[..bytes]);
        remaining -= samples;
    }
    Ok(())
}

fn allocate_pixel_vec<T>(count: usize) -> Result<Vec<T>, FitsError> {
    // Reserve the final address range fallibly, but initialize elements only
    // after their raw chunk has been read. A truncated stream therefore
    // cannot force the declared buffer's pages to be touched up front.
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|_| FitsError::Malformed("pixel buffer allocation failed".into()))?;
    Ok(out)
}

#[multiversion::multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
fn fold_be_u16(raw: &[u8], flip: u16, out: &mut Vec<u16>) {
    out.extend(
        raw.chunks_exact(2)
            .map(|chunk| u16::from_be_bytes([chunk[0], chunk[1]]) ^ flip),
    );
}

/// Signed 16-bit samples, given as their raw bits, as physical values with
/// blank samples as NaN.
fn signed_u16_bits_to_f32(bits: &[u16], blank: Option<i64>) -> Result<Vec<f32>, FitsError> {
    let mut out = allocate_pixel_vec(bits.len())?;
    out.extend(bits.iter().map(|&bits| {
        let stored = bits as i16;
        if blank == Some(i64::from(stored)) {
            f32::NAN
        } else {
            f32::from(stored)
        }
    }));
    Ok(out)
}

fn decode_pixels(reader: &mut impl Read, spec: ImageSpec) -> Result<Pixels, FitsError> {
    match spec.bitpix {
        8 => {
            let mut out = allocate_pixel_vec(spec.count)?;
            read_payload_chunks(reader, spec.count, 1, |raw| out.extend_from_slice(raw))?;
            Ok(Pixels::U8(out))
        }
        16 => {
            // The near-universal camera convention: unsigned data stored as
            // i16 with BZERO 32768. Fold BZERO in while staying u16.
            let offset = spec.bzero as i64;
            if spec.bscale == 1.0 && (offset == 32768 || offset == 0) {
                // Adding 32768 to an i16 is a sign-bit flip on the raw bits:
                // byteswap + XOR. With no offset the raw bits pass through,
                // which is the value unless the sample is negative.
                let flip = if offset == 32768 { 0x8000 } else { 0 };
                let mut out = allocate_pixel_vec(spec.count)?;
                read_payload_chunks(reader, spec.count, 2, |raw| {
                    fold_be_u16(raw, flip, &mut out)
                })?;
                if flip == 0 && out.iter().any(|&bits| bits & 0x8000 != 0) {
                    return signed_u16_bits_to_f32(&out, spec.blank).map(Pixels::F32);
                }
                Ok(Pixels::U16(out))
            } else {
                let mut out = allocate_pixel_vec(spec.count)?;
                read_payload_chunks(reader, spec.count, 2, |raw| {
                    out.extend(raw.chunks_exact(2).map(|chunk| {
                        let stored = i16::from_be_bytes([chunk[0], chunk[1]]);
                        if spec.blank == Some(i64::from(stored)) {
                            return f32::NAN;
                        }
                        (spec.bzero + spec.bscale * f64::from(stored)) as f32
                    }));
                })?;
                Ok(Pixels::F32(out))
            }
        }
        64 => {
            let mut out = allocate_pixel_vec(spec.count)?;
            // BZERO 2^63 is the unsigned convention, the 64-bit kin of
            // BZERO 32768: flipping the sign bit gives the u64 exactly.
            let unsigned = spec.bscale == 1.0 && spec.bzero == 9_223_372_036_854_775_808.0;
            read_payload_chunks(reader, spec.count, 8, |raw| {
                out.extend(raw.chunks_exact(8).map(|chunk| {
                    let stored = i64::from_be_bytes(chunk.try_into().unwrap());
                    if spec.blank == Some(stored) {
                        f64::NAN
                    } else if unsigned {
                        ((stored as u64) ^ (1 << 63)) as f64
                    } else {
                        spec.bzero + spec.bscale * stored as f64
                    }
                }));
            })?;
            Ok(Pixels::F64(out))
        }
        32 => {
            let mut out = allocate_pixel_vec(spec.count)?;
            read_payload_chunks(reader, spec.count, 4, |raw| {
                out.extend(
                    raw.chunks_exact(4)
                        .map(|chunk| i32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
                );
            })?;
            Ok(Pixels::I32(out))
        }
        -32 => {
            let mut out = allocate_pixel_vec(spec.count)?;
            read_payload_chunks(reader, spec.count, 4, |raw| {
                out.extend(
                    raw.chunks_exact(4)
                        .map(|chunk| f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
                );
            })?;
            Ok(Pixels::F32(out))
        }
        -64 => {
            let mut out = allocate_pixel_vec(spec.count)?;
            read_payload_chunks(reader, spec.count, 8, |raw| {
                out.extend(
                    raw.chunks_exact(8)
                        .map(|chunk| f64::from_be_bytes(chunk.try_into().unwrap())),
                );
            })?;
            Ok(Pixels::F64(out))
        }
        _ => unreachable!("ImageSpec validates BITPIX"),
    }
}

impl FitsImage {
    /// Open and decode the image while retaining only the parsed header, the
    /// final typed pixel vector, and a fixed-size conversion buffer. FITS
    /// data-unit padding and trailing HDUs are not read.
    ///
    /// The image is the primary HDU's, or, when the primary holds no data,
    /// the first image extension's, with the headers [`read_header`] gives.
    pub fn open(path: &Path) -> Result<FitsImage, FitsError> {
        let mut file = std::fs::File::open(path)?;
        Self::read_from(&mut file, None)
    }

    /// Decode an in-memory FITS image through the same bounded conversion
    /// pipeline used by [`Self::open`]. The caller retains ownership of the
    /// input slice, so this entry point does not reduce its memory footprint.
    pub fn from_bytes(data: &[u8]) -> Result<FitsImage, FitsError> {
        let mut reader = std::io::Cursor::new(data);
        Self::read_from(&mut reader, Some(data.len() as u64))
    }

    fn read_from<R: Read + Seek>(
        reader: &mut R,
        available_bytes: Option<u64>,
    ) -> Result<FitsImage, FitsError> {
        let ImageHdu {
            header,
            data_start,
            table,
            ..
        } = locate_image(reader, true)?;
        let headers = header.cards;
        let spec = ImageSpec::from_headers(&headers)?;
        let data_bytes = match &table {
            Some(cards) => compressed::table_bytes(cards)
                .ok_or_else(|| FitsError::Malformed("implausible table size".into()))?,
            None => spec.payload_bytes(),
        };
        if let Some(available_bytes) = available_bytes {
            let data_end = data_start
                .checked_add(data_bytes)
                .ok_or_else(|| FitsError::Malformed("implausible dimensions".into()))?;
            if data_end > available_bytes {
                return Err(FitsError::Malformed("data runs past EOF".into()));
            }
        }
        let pixels = match table {
            // Decompression rebuilds the uncompressed data unit, which then
            // decodes like any other.
            Some(cards) => {
                let shape = compressed::Shape {
                    width: spec.width,
                    height: spec.height,
                    planes: spec.planes,
                };
                let data = compressed::decompress(reader, &cards, &shape, spec.blank)?;
                decode_pixels(&mut data.as_slice(), spec)?
            }
            None => decode_pixels(reader, spec)?,
        };
        Ok(FitsImage {
            width: spec.width,
            height: spec.height,
            planes: spec.planes,
            pixels,
            headers,
        })
    }

    pub fn header(&self, key: &str) -> Option<&HeaderValue> {
        self.headers.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn header_f64(&self, key: &str) -> Option<f64> {
        self.header(key).and_then(|v| v.as_f64())
    }

    pub fn header_str(&self, key: &str) -> Option<&str> {
        self.header(key).and_then(|v| v.as_str())
    }

    /// Consume the image and return its pixels in physical units
    /// (`BZERO + BSCALE * stored`) as f32, still in plane order. Integer
    /// samples equal to the header's `BLANK` are undefined and become NaN.
    ///
    /// Three cases skip the header scaling because decoding already applied
    /// it: U16 pixels carry the standard unsigned-camera BZERO, a BITPIX=16
    /// image with negative samples or unusual scaling decodes straight to
    /// F32, and a BITPIX=64 image decodes straight to F64.
    pub fn into_physical_f32(self) -> Vec<f32> {
        let bitpix = self
            .header("BITPIX")
            .and_then(HeaderValue::as_i64)
            .unwrap_or(0);
        let bzero = self.header_f64("BZERO").unwrap_or(0.0);
        let bscale = self.header_f64("BSCALE").unwrap_or(1.0);
        let blank = self.blank();
        let physical = |stored: i64, value: f64| {
            if blank == Some(stored) {
                f32::NAN
            } else {
                (bzero + bscale * value) as f32
            }
        };
        match self.pixels {
            Pixels::U8(values) => values
                .into_iter()
                .map(|value| physical(i64::from(value), f64::from(value)))
                .collect(),
            // Decoding folded BZERO in, so it folds into BLANK too.
            Pixels::U16(values) => match blank
                .and_then(|blank| blank.checked_add(bzero as i64))
                .and_then(|blank| u16::try_from(blank).ok())
            {
                None => values.into_iter().map(f32::from).collect(),
                Some(blank) => values
                    .into_iter()
                    .map(|value| {
                        if value == blank {
                            f32::NAN
                        } else {
                            f32::from(value)
                        }
                    })
                    .collect(),
            },
            Pixels::I32(values) => values
                .into_iter()
                .map(|value| physical(i64::from(value), f64::from(value)))
                .collect(),
            Pixels::F32(values) if bitpix == 16 => values,
            Pixels::F32(values) => values
                .into_iter()
                .map(|value| (bzero + bscale * f64::from(value)) as f32)
                .collect(),
            Pixels::F64(values) if bitpix == 64 => {
                values.into_iter().map(|value| value as f32).collect()
            }
            Pixels::F64(values) => values
                .into_iter()
                .map(|value| (bzero + bscale * value) as f32)
                .collect(),
        }
    }

    /// The stored value that marks an undefined sample: the `BLANK` card of
    /// an integer image.
    fn blank(&self) -> Option<i64> {
        let bitpix = self.header("BITPIX").and_then(HeaderValue::as_i64)?;
        integer_blank(&self.headers, bitpix)
    }

    /// Pixels as u16, converting float/i32 data by min-max scaling.
    /// Planar RGB collapses to luminance; the mono u16 case is a borrow —
    /// no copy, no conversion.
    ///
    /// NaN, and an i32 sample equal to `BLANK`, stays out of the scaling
    /// range and maps to 0. A u8 or u16 sample equal to `BLANK` keeps its
    /// stored value, as nothing here rescales those.
    pub fn to_u16(&self) -> std::borrow::Cow<'_, [u16]> {
        if self.planes == 3 {
            let full = self.planes_u16();
            let n = self.width * self.height;
            let (red, rest) = full.split_at(n);
            let (green, blue) = rest.split_at(n);
            let mut luma = vec![0_u16; n];
            luma.par_chunks_mut(PARALLEL_CHUNK)
                .zip(red.par_chunks(PARALLEL_CHUNK))
                .zip(green.par_chunks(PARALLEL_CHUNK))
                .zip(blue.par_chunks(PARALLEL_CHUNK))
                .for_each(|(((luma, red), green), blue)| {
                    for (((luma, &red), &green), &blue) in
                        luma.iter_mut().zip(red).zip(green).zip(blue)
                    {
                        *luma = ((red as u32 + green as u32 + blue as u32) / 3) as u16;
                    }
                });
            return std::borrow::Cow::Owned(luma);
        }
        self.planes_u16()
    }

    /// Planar RGB as interleaved 16-bit RGB. `None` for mono images.
    pub fn rgb_planes(&self) -> Option<RgbImage16> {
        if self.planes != 3 {
            return None;
        }
        let full = self.planes_u16();
        let n = self.width * self.height;
        let (red, rest) = full.split_at(n);
        let (green, blue) = rest.split_at(n);
        let mut data = vec![0u16; n * 3];
        data.par_chunks_mut(PARALLEL_CHUNK * 3)
            .zip(red.par_chunks(PARALLEL_CHUNK))
            .zip(green.par_chunks(PARALLEL_CHUNK))
            .zip(blue.par_chunks(PARALLEL_CHUNK))
            .for_each(|(((data, red), green), blue)| {
                for (((pixel, &red), &green), &blue) in
                    data.chunks_exact_mut(3).zip(red).zip(green).zip(blue)
                {
                    pixel.copy_from_slice(&[red, green, blue]);
                }
            });
        Some(RgbImage16 {
            width: self.width,
            height: self.height,
            data,
        })
    }

    /// The full stored pixel buffer (all planes, planar order) as u16.
    fn planes_u16(&self) -> std::borrow::Cow<'_, [u16]> {
        match &self.pixels {
            Pixels::U16(data) => std::borrow::Cow::Borrowed(data),
            Pixels::U8(data) => {
                std::borrow::Cow::Owned(data.iter().map(|&v| (v as u16) << 8).collect())
            }
            Pixels::I32(data) => match self.i32_blank() {
                None => scale_to_u16(data, |v| v as f64),
                Some(blank) => scale_to_u16(data, move |v| blank_as_nan(v, blank)),
            },
            Pixels::F32(data) => scale_to_u16(data, |v| v as f64),
            Pixels::F64(data) => scale_to_u16(data, |v| v),
        }
    }

    /// Histogram-based image statistics on the u16 representation.
    pub fn statistics(&self) -> Statistics {
        statistics_u16(&self.to_u16())
    }

    /// N.I.N.A.-compatible MTF autostretch straight to 8-bit grayscale.
    /// Raw one-shot-color mosaics are debayered to luminance first.
    pub fn stretch_to_u8(&self, params: &StretchParams) -> Vec<u8> {
        let data = match self.debayer() {
            Some(rgb) => std::borrow::Cow::Owned(rgb.to_luma_u16()),
            None => self.to_u16(),
        };
        let stats = statistics_u16(&data);
        stretch_u16_to_u8(&data, &stats, params)
    }

    /// N.I.N.A.-compatible MTF autostretch to full-range 16-bit grayscale.
    /// Raw one-shot-color mosaics are debayered to luminance first.
    pub fn stretch_to_u16(&self, params: &StretchParams) -> Vec<u16> {
        let data = match self.debayer() {
            Some(rgb) => std::borrow::Cow::Owned(rgb.to_luma_u16()),
            None => self.to_u16(),
        };
        let stats = statistics_u16(&data);
        stretch_u16_to_u16(&data, &stats, params)
    }

    /// Linear grayscale samples normalized to `[0, 1]` for numeric processing.
    ///
    /// Unlike [`Self::stretch_to_u8`], this does not apply an MTF display
    /// stretch. Positive affine normalization preserves local sigma
    /// significance while retaining more-than-8-bit sample distinctions.
    /// Raw one-shot-color mosaics are debayered to luminance first.
    pub fn to_luma_f32(&self) -> Vec<f32> {
        if let Some(rgb) = self.debayer() {
            return rgb
                .to_luma_u16()
                .into_iter()
                .map(|value| value as f32 / u16::MAX as f32)
                .collect();
        }

        let full = self.planes_f32();
        if self.planes != 3 {
            return full;
        }

        let count = self.width * self.height;
        (0..count)
            .map(|index| (full[index] + full[count + index] + full[2 * count + index]) / 3.0)
            .collect()
    }

    /// The color filter array layout, when the `BAYERPAT` header marks
    /// this as a raw one-shot-color mosaic.
    pub fn bayer_pattern(&self) -> Option<BayerPattern> {
        BayerPattern::parse(self.header_str("BAYERPAT")?)
    }

    /// Declared row direction, or `None` when absent or unrecognized.
    /// Reading an image always preserves the stored pixel order.
    pub fn row_order(&self) -> Option<RowOrder> {
        RowOrder::parse(self.header_str("ROWORDER")?)
    }

    /// CFA pattern at the first stored pixel, before origin offsets.
    /// Missing or unknown `ROWORDER` retains the historical interpretation of
    /// `BAYERPAT` as already describing storage order.
    pub fn bayer_pattern_in_storage_order(&self) -> Option<BayerPattern> {
        let pattern = self.bayer_pattern()?;
        Some(match self.row_order() {
            Some(order) => pattern.in_row_order(order, self.height),
            None => pattern,
        })
    }

    /// Debayer a raw one-shot-color mosaic to interleaved RGB, honoring
    /// `ROWORDER` and `XBAYROFF`/`YBAYROFF` origin offsets. The result retains
    /// storage order, preserving calibration and astrometric coordinates.
    /// `None` for mono images.
    pub fn debayer(&self) -> Option<RgbImage16> {
        if self.planes != 1 {
            return None;
        }
        let pattern = self.bayer_pattern_in_storage_order()?;
        let x_off = self.header_f64("XBAYROFF").unwrap_or(0.0) as usize;
        let y_off = self.header_f64("YBAYROFF").unwrap_or(0.0) as usize;
        Some(debayer_rgb16(
            &self.to_u16(),
            self.width,
            self.height,
            pattern,
            x_off,
            y_off,
        ))
    }

    fn planes_f32(&self) -> Vec<f32> {
        match &self.pixels {
            Pixels::U8(data) => data.iter().map(|&value| value as f32 / 255.0).collect(),
            Pixels::U16(data) => data
                .iter()
                .map(|&value| value as f32 / u16::MAX as f32)
                .collect(),
            Pixels::I32(data) => match self.i32_blank() {
                None => scale_to_f32(data.iter().map(|&value| value as f64)),
                Some(blank) => scale_to_f32(data.iter().map(|&value| blank_as_nan(value, blank))),
            },
            Pixels::F32(data) => scale_to_f32(data.iter().map(|&value| value as f64)),
            Pixels::F64(data) => scale_to_f32(data.iter().copied()),
        }
    }

    /// [`Self::blank`] for i32 pixels, when an i32 can equal it.
    fn i32_blank(&self) -> Option<i32> {
        self.blank().and_then(|blank| i32::try_from(blank).ok())
    }
}

/// An i32 sample as f64, NaN when it is the blank value.
fn blank_as_nan(value: i32, blank: i32) -> f64 {
    if value == blank {
        f64::NAN
    } else {
        f64::from(value)
    }
}

/// Samples one rayon task takes in the whole-image conversions.
const PARALLEL_CHUNK: usize = 1 << 16;

/// The least and greatest finite sample, `(inf, -inf)` when none is.
///
/// Four running extremes, one per lane of every four samples, so the
/// comparisons run as vectors; any order finds the same extremes.
fn finite_range<T: Copy>(values: &[T], to_f64: impl Fn(T) -> f64) -> (f64, f64) {
    const LANES: usize = 4;
    let mut min = [f64::INFINITY; LANES];
    let mut max = [f64::NEG_INFINITY; LANES];
    let mut chunks = values.chunks_exact(LANES);
    for chunk in &mut chunks {
        for lane in 0..LANES {
            let v = to_f64(chunk[lane]);
            let finite = v.is_finite();
            min[lane] = if finite && v < min[lane] {
                v
            } else {
                min[lane]
            };
            max[lane] = if finite && v > max[lane] {
                v
            } else {
                max[lane]
            };
        }
    }
    for &v in chunks.remainder() {
        let v = to_f64(v);
        if v.is_finite() {
            min[0] = min[0].min(v);
            max[0] = max[0].max(v);
        }
    }
    (
        min.into_iter().fold(f64::INFINITY, f64::min),
        max.into_iter().fold(f64::NEG_INFINITY, f64::max),
    )
}

/// Min-max scale samples, as `f64`, onto the full `u16` range. Non-finite
/// samples are left out of the range; NaN maps to 0 and infinities clamp.
///
/// Chunks find their finite range and scale in parallel. The least and
/// greatest finite value do not depend on the order they are searched in,
/// save which zero wins when both signs are present, and that changes no
/// output: it can change only the sign of a zero `v - min` or `max - min`,
/// which the conversion to `u16` and the span's floor drop.
fn scale_to_u16<T: Copy + Sync>(
    values: &[T],
    to_f64: impl Fn(T) -> f64 + Copy + Sync,
) -> std::borrow::Cow<'static, [u16]> {
    let (min, max) = values
        .par_chunks(PARALLEL_CHUNK)
        .map(|chunk| finite_range(chunk, to_f64))
        .reduce(
            || (f64::INFINITY, f64::NEG_INFINITY),
            |(min_a, max_a), (min_b, max_b)| (min_a.min(min_b), max_a.max(max_b)),
        );
    let span = (max - min).max(1e-12);
    let mut out = vec![0_u16; values.len()];
    out.par_chunks_mut(PARALLEL_CHUNK)
        .zip(values.par_chunks(PARALLEL_CHUNK))
        .for_each(|(out, values)| {
            for (out, &v) in out.iter_mut().zip(values) {
                *out = (((to_f64(v) - min) / span).clamp(0.0, 1.0) * 65535.0) as u16;
            }
        });
    std::borrow::Cow::Owned(out)
}

fn scale_to_f32(values: impl Iterator<Item = f64> + Clone) -> Vec<f32> {
    let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
    for value in values.clone() {
        if value.is_finite() {
            min = min.min(value);
            max = max.max(value);
        }
    }
    if !min.is_finite() || !max.is_finite() {
        return values.map(|_| 0.0).collect();
    }

    let span = max - min;
    if span <= f64::EPSILON {
        return values.map(|_| 0.0).collect();
    }
    values
        .map(|value| {
            if value.is_finite() {
                ((value - min) / span).clamp(0.0, 1.0) as f32
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod conversion_tests {
    use super::*;

    /// The serial conversions as they were before they ran on rayon.
    fn previous_scale_to_u16(values: impl Iterator<Item = f64> + Clone) -> Vec<u16> {
        let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
        for v in values.clone() {
            if v.is_finite() {
                min = min.min(v);
                max = max.max(v);
            }
        }
        let span = (max - min).max(1e-12);
        values
            .map(|v| (((v - min) / span).clamp(0.0, 1.0) * 65535.0) as u16)
            .collect()
    }

    fn previous_planes_u16(image: &FitsImage) -> Vec<u16> {
        match &image.pixels {
            Pixels::U16(data) => data.clone(),
            Pixels::U8(data) => data.iter().map(|&v| (v as u16) << 8).collect(),
            Pixels::I32(data) => previous_scale_to_u16(data.iter().map(|&v| v as f64)),
            Pixels::F32(data) => previous_scale_to_u16(data.iter().map(|&v| v as f64)),
            Pixels::F64(data) => previous_scale_to_u16(data.iter().copied()),
        }
    }

    fn previous_to_u16(image: &FitsImage) -> Vec<u16> {
        let full = previous_planes_u16(image);
        if image.planes != 3 {
            return full;
        }
        let n = image.width * image.height;
        (0..n)
            .map(|i| ((full[i] as u32 + full[n + i] as u32 + full[2 * n + i] as u32) / 3) as u16)
            .collect()
    }

    fn previous_rgb_planes(image: &FitsImage) -> Vec<u16> {
        let full = previous_planes_u16(image);
        let n = image.width * image.height;
        let mut data = vec![0u16; n * 3];
        for i in 0..n {
            data[i * 3] = full[i];
            data[i * 3 + 1] = full[n + i];
            data[i * 3 + 2] = full[2 * n + i];
        }
        data
    }

    #[test]
    fn parallel_u16_conversions_match_the_serial_ones() {
        let mut state = 0x853c_49e6_748f_ea9b_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // Empty, single-sample, and chunk-crossing sizes, mono and RGB.
        let shapes = [
            (0, 0, 1),
            (1, 1, 1),
            (1, 1, 3),
            (7, 3, 3),
            (300, 219, 1),
            (512, 256, 3),
            (PARALLEL_CHUNK + 3, 2, 1),
            (PARALLEL_CHUNK / 2 + 5, 3, 3),
        ];
        for (width, height, planes) in shapes {
            let count = width * height * planes;
            // Finite values of both signs and zeros of both signs, with NaN
            // and infinities; some sets hold nothing finite, some only zeros.
            for kind in 0..5 {
                let mut sample = || -> f64 {
                    match (kind, next() % 41) {
                        (3, 0) => f64::NAN,
                        (3, _) => f64::INFINITY,
                        (4, roll) => {
                            if roll % 2 == 0 {
                                0.0
                            } else {
                                -0.0
                            }
                        }
                        (_, 0) => f64::NAN,
                        (_, 1) => f64::INFINITY,
                        (_, 2) => f64::NEG_INFINITY,
                        (_, 3) => -0.0,
                        (_, 4) => 0.0,
                        (2, _) => 3.0e38 * ((next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5),
                        _ => (next() >> 11) as f64 / (1u64 << 53) as f64 * 70_000.0 - 2_000.0,
                    }
                };
                let f64s = (0..count).map(|_| sample()).collect::<Vec<_>>();
                let pixel_sets = [
                    Pixels::F64(f64s.clone()),
                    Pixels::F32(f64s.iter().map(|&v| v as f32).collect()),
                    Pixels::I32(
                        f64s.iter()
                            .map(|&v| if v.is_finite() { v as i32 } else { i32::MIN })
                            .collect(),
                    ),
                    Pixels::U8((0..count).map(|_| next() as u8).collect()),
                    Pixels::U16((0..count).map(|_| next() as u16).collect()),
                ];
                for pixels in pixel_sets {
                    let image = FitsImage {
                        width,
                        height,
                        planes,
                        pixels,
                        headers: Vec::new(),
                    };
                    let label = format!("{width}x{height}x{planes} kind {kind}");
                    assert_eq!(*image.planes_u16(), previous_planes_u16(&image), "{label}");
                    assert_eq!(*image.to_u16(), previous_to_u16(&image), "{label}");
                    if planes == 3 {
                        assert_eq!(
                            image.rgb_planes().unwrap().data,
                            previous_rgb_planes(&image),
                            "{label}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod io_tests {
    use super::*;

    fn value_card(keyword: &str, value: &str) -> [u8; CARD] {
        assert!(keyword.len() <= 8);
        assert!(value.len() <= CARD - 10);
        let mut card = [b' '; CARD];
        card[..keyword.len()].copy_from_slice(keyword.as_bytes());
        card[8] = b'=';
        card[9] = b' ';
        card[10..10 + value.len()].copy_from_slice(value.as_bytes());
        card
    }

    fn image_bytes(
        bitpix: i64,
        axes: &[usize],
        extra_headers: &[(&str, &str)],
        payload: &[u8],
        pad_data: bool,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&value_card("SIMPLE", "T"));
        bytes.extend_from_slice(&value_card("BITPIX", &bitpix.to_string()));
        bytes.extend_from_slice(&value_card("NAXIS", &axes.len().to_string()));
        for (index, length) in axes.iter().enumerate() {
            bytes.extend_from_slice(&value_card(
                &format!("NAXIS{}", index + 1),
                &length.to_string(),
            ));
        }
        for (keyword, value) in extra_headers {
            bytes.extend_from_slice(&value_card(keyword, value));
        }
        let mut end = [b' '; CARD];
        end[..3].copy_from_slice(b"END");
        bytes.extend_from_slice(&end);
        bytes.resize(bytes.len().next_multiple_of(BLOCK), b' ');
        bytes.extend_from_slice(payload);
        if pad_data {
            bytes.resize(bytes.len().next_multiple_of(BLOCK), 0);
        }
        bytes
    }

    /// A header of literal 80-column cards (padded) for an 8-bit 2x2 image.
    fn literal_header_image(cards: &[&str]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for card in [
            "SIMPLE  = T",
            "BITPIX  = 8",
            "NAXIS   = 2",
            "NAXIS1  = 2",
            "NAXIS2  = 2",
        ]
        .iter()
        .chain(cards)
        .chain(&["END"])
        {
            bytes.extend_from_slice(format!("{card:<80}").as_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(BLOCK), b' ');
        bytes.resize(bytes.len() + BLOCK, 0);
        bytes
    }

    #[test]
    fn values_without_a_space_after_the_equals_sign_read_whole() {
        // The standard wants "= " in columns 9-10; reading from column 11
        // turned EXPTIME =30.5 into 0.5 and kept a stray quote.
        let image = FitsImage::from_bytes(&literal_header_image(&[
            "EXPTIME =30.5 / seconds",
            "CTYPE1  ='RA---TAN'",
            "GAIN    = 100",
        ]))
        .unwrap();
        assert_eq!(image.header_f64("EXPTIME"), Some(30.5));
        assert_eq!(image.header_str("CTYPE1"), Some("RA---TAN"));
        assert_eq!(image.header("GAIN"), Some(&HeaderValue::Integer(100)));
    }

    #[test]
    fn long_strings_continue_across_cards() {
        let image = FitsImage::from_bytes(&literal_header_image(&[
            "OBJECT  = 'NGC 7000 North America &'",
            "CONTINUE  'and Pelican &'",
            "CONTINUE  'Nebulae' / continued",
            "FILTER  = 'Ha&'",
            "TELESCOP= 'unrelated'",
            "CONTINUE  'orphan'",
        ]))
        .unwrap();
        assert_eq!(
            image.header_str("OBJECT"),
            Some("NGC 7000 North America and Pelican Nebulae")
        );
        // A trailing & with no CONTINUE card is part of the value, and a
        // CONTINUE after a value without & is ignored.
        assert_eq!(image.header_str("FILTER"), Some("Ha&"));
        assert_eq!(image.header_str("TELESCOP"), Some("unrelated"));
    }

    #[test]
    fn random_groups_are_named_as_unsupported() {
        let mut cards = literal_header_image(&["GROUPS  = T", "PCOUNT  = 0", "GCOUNT  = 1"]);
        // NAXIS1 = 0 marks random groups.
        let naxis1 = 3 * CARD;
        cards[naxis1..naxis1 + CARD].copy_from_slice(format!("{:<80}", "NAXIS1  = 0").as_bytes());
        let error = FitsImage::from_bytes(&cards).unwrap_err();
        assert!(
            matches!(error, FitsError::Unsupported(ref what) if what == "random groups"),
            "{error}"
        );
    }

    fn unsigned_u16_payload(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| (value ^ 0x8000).to_be_bytes())
            .collect()
    }

    #[test]
    fn decodes_integer_float_and_scaled_pixel_types() {
        let image = FitsImage::from_bytes(&image_bytes(
            8,
            &[3, 2],
            &[],
            &[0, 1, 2, 127, 254, 255],
            false,
        ))
        .unwrap();
        assert!(
            matches!(image.pixels, Pixels::U8(ref values) if values == &[0, 1, 2, 127, 254, 255])
        );

        let payload = unsigned_u16_payload(&[0, 1, 32768, 65535]);
        let image = FitsImage::from_bytes(&image_bytes(
            16,
            &[2, 2],
            &[("BZERO", "32768"), ("BSCALE", "1")],
            &payload,
            false,
        ))
        .unwrap();
        assert!(matches!(image.pixels, Pixels::U16(ref values) if values == &[0, 1, 32768, 65535]));

        let payload: Vec<_> = [-1i16, 0, 2, 10]
            .into_iter()
            .flat_map(i16::to_be_bytes)
            .collect();
        let image = FitsImage::from_bytes(&image_bytes(
            16,
            &[2, 2],
            &[("BZERO", "10"), ("BSCALE", "2")],
            &payload,
            false,
        ))
        .unwrap();
        assert!(
            matches!(image.pixels, Pixels::F32(ref values) if values == &[8.0, 10.0, 14.0, 30.0])
        );

        let payload: Vec<_> = [-2i32, 0, 1_234]
            .into_iter()
            .flat_map(i32::to_be_bytes)
            .collect();
        let image = FitsImage::from_bytes(&image_bytes(32, &[3, 1], &[], &payload, false)).unwrap();
        assert!(matches!(image.pixels, Pixels::I32(ref values) if values == &[-2, 0, 1_234]));

        let payload: Vec<_> = [1.5f32, -2.25]
            .into_iter()
            .flat_map(f32::to_be_bytes)
            .collect();
        let image =
            FitsImage::from_bytes(&image_bytes(-32, &[2, 1], &[], &payload, false)).unwrap();
        assert!(matches!(image.pixels, Pixels::F32(ref values) if values == &[1.5, -2.25]));

        let payload: Vec<_> = [1.5f64, -2.25]
            .into_iter()
            .flat_map(f64::to_be_bytes)
            .collect();
        let image =
            FitsImage::from_bytes(&image_bytes(-64, &[2, 1], &[], &payload, false)).unwrap();
        assert!(matches!(image.pixels, Pixels::F64(ref values) if values == &[1.5, -2.25]));
    }

    #[test]
    fn linear_f32_luma_preserves_more_than_eight_bits() {
        let values = [1000, 1001, 32768, 65535];
        let payload = unsigned_u16_payload(&values);
        let image = FitsImage::from_bytes(&image_bytes(
            16,
            &[2, 2],
            &[("BZERO", "32768")],
            &payload,
            false,
        ))
        .unwrap();

        let luma = image.to_luma_f32();
        for (actual, expected) in luma.iter().zip(values) {
            assert_eq!(*actual, expected as f32 / u16::MAX as f32);
        }
        assert_ne!(luma[0], luma[1]);
    }

    #[test]
    fn linear_f32_luma_affine_normalizes_float_data() {
        let image = FitsImage {
            width: 4,
            height: 1,
            planes: 1,
            pixels: Pixels::F32(vec![10.0, 15.0, 20.0, f32::NAN]),
            headers: Vec::new(),
        };
        assert_eq!(image.to_luma_f32(), [0.0, 0.5, 1.0, 0.0]);
    }

    #[test]
    fn streamed_decode_stops_before_fits_padding() {
        let payload = unsigned_u16_payload(&[10, 20, 30]);
        let bytes = image_bytes(16, &[3, 1], &[("BZERO", "32768")], &payload, true);
        let expected_position = BLOCK + payload.len();
        let mut reader = std::io::Cursor::new(bytes);
        let image = FitsImage::read_from(&mut reader, None).unwrap();
        assert_eq!(reader.position() as usize, expected_position);
        assert!(matches!(image.pixels, Pixels::U16(ref values) if values == &[10, 20, 30]));

        // The same non-block-aligned data unit is valid without physical
        // padding when the declared pixels are all present.
        let unpadded = image_bytes(16, &[3, 1], &[("BZERO", "32768")], &payload, false);
        assert!(FitsImage::from_bytes(&unpadded).is_ok());
    }

    #[test]
    fn streamed_decode_handles_a_partial_final_chunk() {
        let count = PIXEL_CHUNK_BYTES / 2 + 7;
        let payload: Vec<_> = (0..count)
            .flat_map(|index| ((index as u16) ^ 0x8000).to_be_bytes())
            .collect();
        let image = FitsImage::from_bytes(&image_bytes(
            16,
            &[count, 1],
            &[("BZERO", "32768")],
            &payload,
            false,
        ))
        .unwrap();
        let Pixels::U16(values) = image.pixels else {
            panic!("expected u16 storage");
        };
        assert_eq!(values.len(), count);
        for index in [0, count / 2, count - 8, count - 1] {
            assert_eq!(values[index], index as u16);
        }
    }

    #[test]
    fn rejects_truncated_headers_and_pixel_payloads() {
        let short_header = vec![b' '; BLOCK - 1];
        assert!(matches!(
            FitsImage::from_bytes(&short_header),
            Err(FitsError::NotFits)
        ));
        assert!(matches!(
            read_headers_from(&mut std::io::Cursor::new(&short_header), false),
            Err(FitsError::Malformed(message)) if message == "header runs past EOF"
        ));

        let mut incomplete_header = vec![b' '; BLOCK];
        incomplete_header[..6].copy_from_slice(b"SIMPLE");
        assert!(matches!(
            FitsImage::from_bytes(&incomplete_header),
            Err(FitsError::Malformed(message)) if message == "header runs past EOF"
        ));

        let payload = unsigned_u16_payload(&[10, 20, 30]);
        let mut truncated = image_bytes(16, &[3, 1], &[("BZERO", "32768")], &payload, false);
        truncated.pop();
        assert!(matches!(
            FitsImage::from_bytes(&truncated),
            Err(FitsError::Malformed(message)) if message == "data runs past EOF"
        ));

        // Dimension metadata is checked against known file/slice length
        // before attempting to reserve the declared final pixel vector.
        let huge_truncated =
            image_bytes(16, &[1_000_000, 1_000], &[("BZERO", "32768")], &[], false);
        assert!(matches!(
            FitsImage::from_bytes(&huge_truncated),
            Err(FitsError::Malformed(message)) if message == "data runs past EOF"
        ));
    }

    #[test]
    fn preserves_planar_rgb_and_bayer_metadata() {
        let payload = unsigned_u16_payload(&[10, 20, 30, 40, 50, 60]);
        let image = FitsImage::from_bytes(&image_bytes(
            16,
            &[2, 1, 3],
            &[("BZERO", "32768")],
            &payload,
            false,
        ))
        .unwrap();
        assert_eq!(image.planes, 3);
        assert_eq!(image.rgb_planes().unwrap().data, [10, 30, 50, 20, 40, 60]);

        let payload = unsigned_u16_payload(&[1000; 16]);
        let image = FitsImage::from_bytes(&image_bytes(
            16,
            &[4, 4],
            &[("BZERO", "32768"), ("BAYERPAT", "'RGGB'")],
            &payload,
            false,
        ))
        .unwrap();
        assert_eq!(image.bayer_pattern(), Some(BayerPattern::Rggb));
        let rgb = image.debayer().unwrap();
        assert_eq!((rgb.width, rgb.height), (4, 4));
        assert!(rgb.data.iter().all(|value| *value == 1000));
    }

    #[test]
    fn debayer_resolves_row_order_and_offsets_without_reordering_pixels() {
        // Bottom-up, even-height RGGB with X offset 1 is BG/GR in storage.
        let samples = [3000, 2000, 3002, 2002, 2004, 1004, 2006, 1006];
        let payload = unsigned_u16_payload(&samples);
        for (row_order, expected) in [
            (Some("' bottom-up '"), BayerPattern::Gbrg),
            (Some("'TOP-DOWN'"), BayerPattern::Rggb),
            (Some("'unknown'"), BayerPattern::Rggb),
            (None, BayerPattern::Rggb),
        ] {
            let mut headers = vec![
                ("BZERO", "32768"),
                ("BAYERPAT", "'RGGB'"),
                ("XBAYROFF", "1"),
                ("YBAYROFF", "0"),
            ];
            if let Some(value) = row_order {
                headers.push(("ROWORDER", value));
            }
            let image = FitsImage::from_bytes(&image_bytes(16, &[4, 2], &headers, &payload, false))
                .unwrap();
            assert_eq!(image.bayer_pattern(), Some(BayerPattern::Rggb));
            assert_eq!(image.bayer_pattern_in_storage_order(), Some(expected));
            assert_eq!(image.to_u16().as_ref(), samples);
            let rgb = image.debayer().unwrap();
            assert_eq!(rgb.data, debayer_rgb16(&samples, 4, 2, expected, 1, 0).data);
            if image.row_order() == Some(RowOrder::BottomUp) {
                assert_eq!(rgb.data[2], 3000);
                assert_eq!(rgb.data[5 * 3], 1004);
            }
        }
    }

    #[test]
    fn header_only_reader_does_not_require_or_touch_pixels() {
        let bytes = image_bytes(16, &[1000, 1000], &[], &[], false);
        let mut reader = std::io::Cursor::new(bytes);
        let (header, _) = read_headers_from(&mut reader, false).unwrap();
        assert_eq!(reader.position() as usize, BLOCK);
        assert!(header.cards.iter().any(|(key, _)| key == "NAXIS1"));
    }

    #[test]
    fn commentary_cards_are_kept_beside_the_valued_cards() {
        let card = |text: &str| format!("{text:<80}");
        let mut header = [
            "SIMPLE  =                    T",
            "BITPIX  =                  -32",
            "NAXIS   =                    0",
            "HISTORY Calibration: dark master-dark.fit",
            "COMMENT   written by a test",
            "HISTORY Registration: shift 1.5 -2.0",
            "IMAGETYP= 'LIGHT'",
            "END",
        ]
        .map(card)
        .concat()
        .into_bytes();
        header.resize(BLOCK, b' ');
        let (header, _) = read_headers_from(&mut std::io::Cursor::new(header), false).unwrap();
        assert_eq!(
            header.history,
            [
                "Calibration: dark master-dark.fit",
                "Registration: shift 1.5 -2.0"
            ]
        );
        assert_eq!(header.comments, ["written by a test"]);
        let keywords: Vec<&str> = header.cards.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(keywords, ["SIMPLE", "BITPIX", "NAXIS", "IMAGETYP"]);
    }

    #[test]
    fn open_streams_a_regular_file() {
        let path = std::env::temp_dir().join(format!(
            "seiza-fits-stream-open-{}.fits",
            std::process::id()
        ));
        let payload = unsigned_u16_payload(&[100, 200, 300, 400]);
        std::fs::write(
            &path,
            image_bytes(16, &[2, 2], &[("BZERO", "32768")], &payload, true),
        )
        .unwrap();
        let image = FitsImage::open(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!((image.width, image.height, image.planes), (2, 2, 1));
        assert!(matches!(image.pixels, Pixels::U16(ref values) if values == &[100, 200, 300, 400]));
    }
}

#[cfg(test)]
mod hdu_tests {
    use super::*;

    /// A valued card, `KEYWORD = value`, with the value right-aligned.
    fn card(keyword: &str, value: &str) -> String {
        format!("{keyword:<8}= {value:>20}")
    }

    /// One HDU: the cards and END, padded to a block, then the payload,
    /// padded with zeros.
    fn hdu(cards: &[String], payload: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for text in cards.iter().map(String::as_str).chain(["END"]) {
            assert!(text.len() <= CARD, "{text}");
            bytes.extend_from_slice(format!("{text:<80}").as_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(BLOCK), b' ');
        bytes.extend_from_slice(payload);
        bytes.resize(bytes.len().next_multiple_of(BLOCK), 0);
        bytes
    }

    fn primary(bitpix: i64, axes: &[usize], extra: &[String]) -> Vec<String> {
        let mut cards = vec![
            card("SIMPLE", "T"),
            card("BITPIX", &bitpix.to_string()),
            card("NAXIS", &axes.len().to_string()),
        ];
        for (index, length) in axes.iter().enumerate() {
            cards.push(card(&format!("NAXIS{}", index + 1), &length.to_string()));
        }
        cards.extend_from_slice(extra);
        cards
    }

    fn extension(kind: &str, bitpix: i64, axes: &[usize], extra: &[String]) -> Vec<String> {
        let mut cards = primary(bitpix, axes, &[]);
        cards[0] = format!("XTENSION= '{kind:<8}'");
        cards.push(card("PCOUNT", "0"));
        cards.push(card("GCOUNT", "1"));
        cards.extend_from_slice(extra);
        cards
    }

    fn be<T: Copy, const N: usize>(values: &[T], to_bytes: fn(T) -> [u8; N]) -> Vec<u8> {
        values.iter().flat_map(|&value| to_bytes(value)).collect()
    }

    /// Samples compared bit for bit, so NaN equals NaN.
    fn assert_samples(actual: &[f32], expected: &[f32]) {
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(actual), bits(expected), "{actual:?} vs {expected:?}");
    }

    fn keywords(image: &FitsImage) -> Vec<&str> {
        image.headers.iter().map(|(key, _)| key.as_str()).collect()
    }

    #[test]
    fn signed_16_bit_samples_keep_their_sign() {
        // A HiPS tile: BITPIX 16 with no BZERO and samples below zero, which
        // used to clamp to 0.
        let stored = [-2788_i16, -1, 0, 1, 32767, -32768];
        let file = hdu(&primary(16, &[3, 2], &[]), &be(&stored, i16::to_be_bytes));
        let image = FitsImage::from_bytes(&file).unwrap();
        assert!(
            matches!(image.pixels, Pixels::F32(ref values) if *values == stored.map(f32::from))
        );
        // The u16 view scales the full signed range.
        assert_eq!(image.to_u16()[..2], [29_980, 32_767]);
        assert_samples(&image.into_physical_f32(), &stored.map(f32::from));

        // Without a negative sample the data stays u16, as before; an
        // explicit BZERO 0 and BSCALE 1 change nothing.
        for extra in [vec![], vec![card("BZERO", "0"), card("BSCALE", "1")]] {
            let stored = [0_i16, 1, 32767];
            let file = hdu(
                &primary(16, &[3, 1], &extra),
                &be(&stored, i16::to_be_bytes),
            );
            let image = FitsImage::from_bytes(&file).unwrap();
            assert!(matches!(image.pixels, Pixels::U16(ref values) if values == &[0, 1, 32767]));
        }
    }

    #[test]
    fn blank_samples_become_nan_in_physical_values() {
        let nan = f32::NAN;
        let blank = |value: &str| card("BLANK", value);
        let decode = |cards: Vec<String>, payload: Vec<u8>| {
            FitsImage::from_bytes(&hdu(&cards, &payload)).unwrap()
        };

        // u8 storage keeps the stored blank; scaling skips it.
        let image = decode(
            primary(
                8,
                &[3, 1],
                &[blank("255"), card("BZERO", "1"), card("BSCALE", "2")],
            ),
            vec![0, 255, 7],
        );
        assert!(matches!(image.pixels, Pixels::U8(ref values) if values == &[0, 255, 7]));
        assert_samples(&image.into_physical_f32(), &[1.0, nan, 15.0]);

        // The camera convention folds BZERO into the stored blank as well:
        // stored -32768 is u16 0.
        let image = decode(
            primary(16, &[3, 1], &[card("BZERO", "32768"), blank("-32768")]),
            be(&[-32768_i16, 0, 100], i16::to_be_bytes),
        );
        assert!(matches!(image.pixels, Pixels::U16(ref values) if values == &[0, 32768, 32868]));
        assert_samples(&image.into_physical_f32(), &[nan, 32768.0, 32868.0]);

        // Signed and scaled 16-bit data decode to floats with NaN blanks.
        let image = decode(
            primary(16, &[3, 1], &[blank("-32768")]),
            be(&[-32768_i16, -5, 5], i16::to_be_bytes),
        );
        assert!(matches!(image.pixels, Pixels::F32(_)));
        assert_samples(&image.into_physical_f32(), &[nan, -5.0, 5.0]);
        // A HiPS writer gives BLANK as a float, which astropy and cfitsio
        // both ignore.
        let image = decode(
            primary(16, &[3, 1], &[blank("-32768.0")]),
            be(&[-32768_i16, -5, 5], i16::to_be_bytes),
        );
        assert_samples(&image.into_physical_f32(), &[-32768.0, -5.0, 5.0]);
        let image = decode(
            primary(
                16,
                &[3, 1],
                &[card("BZERO", "10"), card("BSCALE", "2"), blank("-1")],
            ),
            be(&[-1_i16, 0, 2], i16::to_be_bytes),
        );
        assert_samples(&image.into_physical_f32(), &[nan, 10.0, 14.0]);

        // i32 storage keeps the blank, but the u16 view leaves it out of the
        // scaling range rather than let it squeeze the real samples.
        let image = decode(
            primary(32, &[3, 1], &[blank("-2147483648")]),
            be(&[i32::MIN, 10, 20], i32::to_be_bytes),
        );
        assert!(matches!(image.pixels, Pixels::I32(ref values) if values == &[i32::MIN, 10, 20]));
        assert_eq!(*image.to_u16(), [0, 0, 65535]);
        assert_eq!(image.to_luma_f32(), [0.0, 0.0, 1.0]);
        assert_samples(&image.into_physical_f32(), &[nan, 10.0, 20.0]);

        let image = decode(
            primary(64, &[2, 1], &[blank("7")]),
            be(&[7_i64, 8], i64::to_be_bytes),
        );
        assert_samples(&image.into_physical_f32(), &[nan, 8.0]);

        // The standard defines BLANK for integer data only.
        let image = decode(
            primary(-32, &[2, 1], &[blank("0")]),
            be(&[0.0_f32, 1.0], f32::to_be_bytes),
        );
        assert_samples(&image.into_physical_f32(), &[0.0, 1.0]);
    }

    #[test]
    fn reads_64_bit_integers() {
        let stored = [-3_i64, 0, 5, 1 << 53, i64::MAX, i64::MIN];
        let file = hdu(&primary(64, &[3, 2], &[]), &be(&stored, i64::to_be_bytes));
        let image = FitsImage::from_bytes(&file).unwrap();
        let expected = stored.map(|value| value as f64);
        assert!(matches!(image.pixels, Pixels::F64(ref values) if *values == expected));
        assert_samples(
            &image.into_physical_f32(),
            &expected.map(|value| value as f32),
        );

        // BZERO 2^63 stores unsigned values; the sign-bit flip is exact.
        let unsigned = [card("BZERO", "9223372036854775808")];
        let file = hdu(
            &primary(64, &[3, 1], &unsigned),
            &be(&[i64::MIN, -1, i64::MAX], i64::to_be_bytes),
        );
        let image = FitsImage::from_bytes(&file).unwrap();
        let expected = [0.0, (i64::MAX as u64) as f64, u64::MAX as f64];
        assert!(matches!(image.pixels, Pixels::F64(ref values) if *values == expected));

        // Other scaling applies as written, and a cube keeps its planes.
        let scaled = [card("BZERO", "10"), card("BSCALE", "0.5")];
        let file = hdu(
            &primary(64, &[1, 1, 3], &scaled),
            &be(&[-4_i64, 0, 4], i64::to_be_bytes),
        );
        let image = FitsImage::from_bytes(&file).unwrap();
        assert_eq!((image.width, image.height, image.planes), (1, 1, 3));
        assert_samples(&image.into_physical_f32(), &[8.0, 10.0, 12.0]);

        let file = hdu(
            &primary(64, &[2, 1], &[]),
            &be(&[1_i64, 2], i64::to_be_bytes),
        );
        assert!(matches!(
            FitsImage::from_bytes(&file[..BLOCK + 8]),
            Err(FitsError::Malformed(message)) if message == "data runs past EOF"
        ));
    }

    #[test]
    fn hierarch_cards_name_their_full_keyword() {
        let cards = primary(
            8,
            &[1, 1],
            &[
                "HIERARCH LBTO LUCI DET ITIME = 0.139764 / [s] integration time".into(),
                "HIERARCH ESO OBS NAME = 'night one &'".into(),
                "CONTINUE  'and two' / continued".into(),
                "HIERARCH ESO TPL ID= 'a=b' / value holds an equals sign".into(),
                "HIERARCH GAIN = 3".into(),
                "HIERARCH Mixed  Case = T".into(),
                "HIERARCH no value / a=b".into(),
                "HIERARCH = 1".into(),
                card("HIERARCH", "5"),
            ],
        );
        let image = FitsImage::from_bytes(&hdu(&cards, &[0])).unwrap();
        assert_eq!(
            image.header("LBTO LUCI DET ITIME"),
            Some(&HeaderValue::Float(0.139764))
        );
        assert_eq!(image.header_str("ESO OBS NAME"), Some("night one and two"));
        assert_eq!(image.header_str("ESO TPL ID"), Some("a=b"));
        assert_eq!(image.header("GAIN"), Some(&HeaderValue::Integer(3)));
        assert_eq!(
            image.header("Mixed  Case"),
            Some(&HeaderValue::Logical(true))
        );
        // A card named HIERARCH with `=` in column 9 is an ordinary card.
        assert_eq!(image.header("HIERARCH"), Some(&HeaderValue::Integer(5)));
        assert_eq!(keywords(&image).len(), 5 + 6);
    }

    #[test]
    fn an_empty_primary_hands_over_to_the_first_image_extension() {
        // Shaped like fitsio-pure's test0.fits: a primary with BZERO 32768
        // but no data, then an unscaled 16-bit image extension.
        let mut file = hdu(
            &primary(
                16,
                &[],
                &[
                    card("EXTEND", "T"),
                    card("BZERO", "32768"),
                    card("BSCALE", "1"),
                    card("CHECKSUM", "'0000'"),
                    card("OBJECT", "'from the primary'"),
                    card("DATE-OBS", "'2024-05-01T00:00:00'"),
                    card("TELESCOP", "'scope'"),
                    "HISTORY primary history".into(),
                ],
            ),
            &[],
        );
        // A table with a heap, which the search must step over whole.
        file.extend(hdu(
            &extension(
                "BINTABLE",
                8,
                &[4, 3],
                &[card("TFIELDS", "1"), card("TFORM1", "'1J'")],
            )
            .into_iter()
            .map(|text| {
                if text.starts_with("PCOUNT") {
                    card("PCOUNT", "2900")
                } else {
                    text
                }
            })
            .collect::<Vec<_>>(),
            &[7; 12 + 2900],
        ));
        // A one-axis image does not count.
        file.extend(hdu(&extension("IMAGE", 8, &[4], &[]), &[1, 2, 3, 4]));
        file.extend(hdu(
            &extension(
                "IMAGE",
                16,
                &[3, 1],
                &[
                    card("INHERIT", "T"),
                    card("OBJECT", "'from the extension'"),
                    card("CRPIX1", "1.5"),
                    "HISTORY extension history".into(),
                ],
            ),
            &be(&[-5_i16, 0, 7], i16::to_be_bytes),
        ));
        // Only the first image extension is read.
        file.extend(hdu(&extension("IMAGE", 8, &[1, 1], &[]), &[9]));

        let image = FitsImage::from_bytes(&file).unwrap();
        assert_eq!((image.width, image.height, image.planes), (3, 1, 1));
        assert_eq!(
            keywords(&image),
            [
                "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "INHERIT", "OBJECT", "CRPIX1", "DATE-OBS",
                "TELESCOP",
            ]
        );
        assert_eq!(image.header_str("OBJECT"), Some("from the extension"));
        assert_eq!(image.header_str("TELESCOP"), Some("scope"));
        // The primary's BZERO describes its own empty array, not this one.
        assert_samples(&image.into_physical_f32(), &[-5.0, 0.0, 7.0]);

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("extension.fits");
        std::fs::write(&path, &file).unwrap();
        let opened = FitsImage::open(&path).unwrap();
        let header = read_header_with_commentary(&path).unwrap();
        assert_eq!(header.cards, opened.headers);
        assert_eq!(read_header(&path).unwrap(), opened.headers);
        assert_eq!(header.history, ["extension history", "primary history"]);
    }

    #[test]
    fn an_extension_with_inherit_false_stands_alone() {
        let mut file = hdu(&primary(8, &[], &[card("DATE-OBS", "'2024-05-01'")]), &[]);
        file.extend(hdu(
            &extension("IMAGE", 8, &[1, 1], &[card("INHERIT", "F")]),
            &[3],
        ));
        let image = FitsImage::from_bytes(&file).unwrap();
        assert_eq!(
            keywords(&image),
            ["BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "INHERIT"]
        );
        assert!(matches!(image.pixels, Pixels::U8(ref values) if values == &[3]));
    }

    #[test]
    fn files_without_an_image_extension_read_as_before() {
        let empty = hdu(&primary(8, &[], &[card("OBJECT", "'x'")]), &[]);
        let table = hdu(&extension("BINTABLE", 8, &[4, 1], &[]), &[0; 4]);
        let mut garbage = vec![b'x'; BLOCK];
        garbage[..8].copy_from_slice(b"XTENSION");
        let image_header = hdu(&extension("IMAGE", 16, &[1000, 1000], &[]), &[]);
        let cases = [
            [empty.clone(), table.clone()].concat(),
            [empty.clone(), vec![b' '; 100]].concat(),
            [empty.clone(), table, vec![b'?'; BLOCK]].concat(),
            // An extension header with no END card ends the search.
            [empty.clone(), garbage].concat(),
            // So does one cut short.
            [empty.clone(), image_header[..BLOCK / 2].to_vec()].concat(),
        ];
        for (index, file) in cases.iter().enumerate() {
            let error = FitsImage::from_bytes(file).unwrap_err();
            assert!(
                matches!(error, FitsError::Unsupported(ref what) if what == "NAXIS 0 (need a 2D image)"),
                "case {index}: {error}"
            );
        }
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("table.fits");
        std::fs::write(&path, &cases[0]).unwrap();
        let cards = read_header(&path).unwrap();
        assert_eq!(cards.len(), 4);
        assert_eq!(cards[3].0, "OBJECT");

        // An image extension whose data is cut short is an error of its own.
        let truncated = [empty, image_header].concat();
        assert!(matches!(
            FitsImage::from_bytes(&truncated),
            Err(FitsError::Malformed(message)) if message == "data runs past EOF"
        ));
    }

    #[test]
    fn header_updates_reach_the_extension_image() {
        let mut file = hdu(
            &primary(8, &[], &[card("OBJECT", "'old'"), card("OBSERVER", "'me'")]),
            &[],
        );
        file.extend(hdu(&extension("IMAGE", 8, &[2, 1], &[]), &[1, 2]));
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("update.fits");
        std::fs::write(&path, &file).unwrap();

        // OBJECT lives in the primary only; the new value goes to the
        // extension and is found first.
        update_header_in_place(&path, "OBJECT", &HeaderValue::String("new".into()), None).unwrap();
        update_header_in_place(&path, "OBJECT", &HeaderValue::String("newer".into()), None)
            .unwrap();
        let image = FitsImage::open(&path).unwrap();
        assert_eq!(image.header_str("OBJECT"), Some("newer"));
        assert_eq!(image.header_str("OBSERVER"), Some("me"));
        assert_eq!(
            image
                .headers
                .iter()
                .filter(|(key, _)| key == "OBJECT")
                .count(),
            1
        );
        assert!(matches!(image.pixels, Pixels::U8(ref values) if values == &[1, 2]));
        let written = std::fs::read(&path).unwrap();
        assert_eq!(written[..BLOCK], file[..BLOCK], "the primary is untouched");
        assert_eq!(written.len(), file.len());

        for structural in ["XTENSION", "PCOUNT", "NAXIS2"] {
            assert!(
                update_header_in_place(&path, structural, &HeaderValue::Integer(1), None).is_err()
            );
        }
    }

    /// One synthetic file for each kind of corpus file that used to fail,
    /// named after its model in OrbitalCommons' fits-test-cases.
    #[test]
    fn corpus_like_files_decode() {
        let empty_primary = |bitpix: i64, extra: &[String]| hdu(&primary(bitpix, &[], extra), &[]);
        // Name, file, (width, height, planes), physical samples, and a
        // keyword the header must hold.
        type Case = (
            &'static str,
            Vec<u8>,
            (usize, usize, usize),
            Vec<f32>,
            &'static str,
        );
        let cases: Vec<Case> = vec![
            (
                "hipsgen/Npix140.fits: signed 16-bit HiPS tile",
                hdu(
                    &primary(16, &[2, 2], &[card("BLANK", "-32768")]),
                    &be(&[-2788_i16, -32768, 0, 140], i16::to_be_bytes),
                ),
                (2, 2, 1),
                vec![-2788.0, f32::NAN, 0.0, 140.0],
                "BLANK",
            ),
            (
                "cfitsio/iter_image.fit: ESO HIERARCH cards",
                hdu(
                    &primary(
                        16,
                        &[2, 1],
                        &[
                            card("BZERO", "32768"),
                            "HIERARCH LBTO LUCI DET ITIME = 0.139764 / [s]".into(),
                        ],
                    ),
                    &be(&[i16::MIN, 0], i16::to_be_bytes),
                ),
                (2, 1, 1),
                vec![0.0, 32768.0],
                "LBTO LUCI DET ITIME",
            ),
            (
                "rust-fitsio/cube.fits: 64-bit integer cube",
                hdu(
                    &primary(64, &[1, 2, 2], &[]),
                    &be(&[0_i64, 1, 2, 3], i64::to_be_bytes),
                ),
                (1, 2, 2),
                vec![0.0, 1.0, 2.0, 3.0],
                "NAXIS3",
            ),
            (
                "meter-sim/mixed_types_u8_i32_f32.fits: image extensions after an empty primary",
                [
                    empty_primary(8, &[card("EXTEND", "T")]),
                    hdu(&extension("IMAGE", 8, &[2, 1], &[]), &[0, 255]),
                    hdu(
                        &extension("IMAGE", -32, &[1, 1], &[]),
                        &be(&[1.5_f32], f32::to_be_bytes),
                    ),
                ]
                .concat(),
                (2, 1, 1),
                vec![0.0, 255.0],
                "NAXIS2",
            ),
            (
                "meter-sim/roundtrip_f64_multi_hdu.fits: 64-bit float extension",
                [
                    empty_primary(8, &[]),
                    hdu(
                        &extension("IMAGE", -64, &[1, 2], &[]),
                        &be(&[0.25_f64, -0.5], f64::to_be_bytes),
                    ),
                ]
                .concat(),
                (1, 2, 1),
                vec![0.25, -0.5],
                "NAXIS2",
            ),
            (
                "nasa-samples/HST_NICMOS.fits: INHERIT = T extension with its own WCS",
                [
                    empty_primary(16, &[card("TELESCOP", "'HST'")]),
                    hdu(
                        &extension(
                            "IMAGE",
                            -32,
                            &[2, 1],
                            &[card("INHERIT", "T"), card("CRVAL1", "182.63")],
                        ),
                        &be(&[-1.0_f32, 2730.9], f32::to_be_bytes),
                    ),
                ]
                .concat(),
                (2, 1, 1),
                vec![-1.0, 2730.9],
                "TELESCOP",
            ),
            (
                "nasa-samples/EUVE.fits: INHERIT = F, BZERO 0, tables after",
                [
                    empty_primary(8, &[card("TELESCOP", "'EUVE'")]),
                    hdu(
                        &extension(
                            "IMAGE",
                            16,
                            &[2, 1],
                            &[
                                card("INHERIT", "F"),
                                card("BZERO", "0.0"),
                                card("BSCALE", "1.0"),
                            ],
                        ),
                        &be(&[-3_i16, 4], i16::to_be_bytes),
                    ),
                    hdu(&extension("BINTABLE", 8, &[16, 3], &[]), &[0; 48]),
                ]
                .concat(),
                (2, 1, 1),
                vec![-3.0, 4.0],
                "INHERIT",
            ),
            (
                "mwa/1297526432_gpubox_ch117.fits: i32 visibilities extension",
                [
                    empty_primary(8, &[card("TIME", "1297526432")]),
                    hdu(
                        &extension("IMAGE", 32, &[2, 2], &[]),
                        &be(&[-7_i32, 8, 9, 10], i32::to_be_bytes),
                    ),
                ]
                .concat(),
                (2, 2, 1),
                vec![-7.0, 8.0, 9.0, 10.0],
                "TIME",
            ),
        ];
        for (name, file, shape, physical, keyword) in cases {
            let image =
                FitsImage::from_bytes(&file).unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!((image.width, image.height, image.planes), shape, "{name}");
            assert!(image.header(keyword).is_some(), "{name}: {keyword}");
            assert_samples(&image.into_physical_f32(), &physical);
        }
    }
}
