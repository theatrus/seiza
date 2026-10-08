//! Tile-compressed images, as `fpack` and cfitsio write them (FITS 4.0
//! standard, section 10). The image is cut into tiles, and each row of a
//! binary table holds one tile, compressed with Rice, GZIP or PLIO. Floating
//! point tiles are usually quantized to integers first, with a per-tile
//! scale and zero point and, optionally, subtractive dithering.
//!
//! Decompression rebuilds the data unit the image would have had
//! uncompressed, as big-endian bytes, so the ordinary decoder applies
//! `BZERO`, `BSCALE` and `BLANK` to it exactly as to any other image.

use crate::{FitsError, FitsHeader, HeaderValue, card_i64, card_value, read_payload_exact};
use std::io::Read;
use std::sync::OnceLock;

/// The table keywords that give a column's layout, as `TFORMn` and the
/// rest name them.
const COLUMN_KEYWORDS: &[&str] = &[
    "TTYPE", "TFORM", "TUNIT", "TNULL", "TSCAL", "TZERO", "TDISP", "TBCOL", "TDIM", "TCTYP",
    "TCUNI", "TCRPX", "TCRVL", "TCDLT", "TRPOS",
];

/// Compression keywords numbered by axis or parameter.
const INDEXED_KEYWORDS: &[&str] = &["ZNAXIS", "ZTILE", "ZNAME", "ZVAL"];

/// Compression keywords that hold an original image keyword's value.
const RENAMED_KEYWORDS: &[(&str, &str)] = &[
    ("ZSIMPLE", "SIMPLE"),
    ("ZTENSION", "XTENSION"),
    ("ZBITPIX", "BITPIX"),
    ("ZNAXIS", "NAXIS"),
    ("ZEXTEND", "EXTEND"),
    ("ZBLOCKED", "BLOCKED"),
    ("ZPCOUNT", "PCOUNT"),
    ("ZGCOUNT", "GCOUNT"),
    ("ZHECKSUM", "CHECKSUM"),
    ("ZDATASUM", "DATASUM"),
];

/// A quantized sample that SUBTRACTIVE_DITHER_2 restores as exactly 0.
const ZERO_VALUE: i64 = -2_147_483_646;

/// Whether `keyword`, with a numeric suffix, is one of `labels` followed by
/// a number from 1 to 999.
fn is_indexed(keyword: &str, labels: &[&str]) -> bool {
    labels.iter().any(|label| {
        keyword.strip_prefix(label).is_some_and(|suffix| {
            !suffix.is_empty()
                && suffix.len() <= 3
                && !suffix.starts_with('0')
                && suffix.bytes().all(|byte| byte.is_ascii_digit())
        })
    })
}

/// Whether a keyword of a compressed image's table describes the table or
/// the compression rather than the image. Changing one in place would make
/// the image unreadable.
pub(crate) fn is_reserved_keyword(keyword: &str) -> bool {
    matches!(
        keyword,
        "TFIELDS"
            | "THEAP"
            | "ZIMAGE"
            | "ZCMPTYPE"
            | "ZMASKCMP"
            | "ZQUANTIZ"
            | "ZDITHER0"
            | "ZBLANK"
            | "ZSCALE"
            | "ZZERO"
    ) || RENAMED_KEYWORDS.iter().any(|(name, _)| *name == keyword)
        || is_indexed(keyword, COLUMN_KEYWORDS)
        || is_indexed(keyword, INDEXED_KEYWORDS)
}

/// The image's axis lengths, `ZNAXIS1` to `ZNAXISn`, when all are present
/// and none is negative.
fn image_axes(cards: &[(String, HeaderValue)]) -> Option<Vec<u64>> {
    let naxis = card_i64(cards, "ZNAXIS")?.clamp(0, 999);
    (1..=naxis)
        .map(|axis| {
            card_i64(cards, &format!("ZNAXIS{axis}")).and_then(|length| u64::try_from(length).ok())
        })
        .collect()
}

/// Whether an extension header is a binary table holding a tile-compressed
/// image with two or more axes, none of them empty.
pub(crate) fn holds_compressed_image(cards: &[(String, HeaderValue)]) -> bool {
    card_value(cards, "XTENSION").and_then(HeaderValue::as_str) == Some("BINTABLE")
        && card_value(cards, "ZIMAGE").and_then(HeaderValue::as_bool) == Some(true)
        && image_axes(cards).is_some_and(|axes| axes.len() >= 2 && !axes.contains(&0))
}

/// The header of the image a compressed table holds, rebuilt as the
/// standard and astropy rebuild it.
///
/// `ZBITPIX`, `ZNAXIS` and `ZNAXISn` become `BITPIX`, `NAXIS` and `NAXISn`
/// at the top, after `SIMPLE` or `XTENSION`. The other keywords that kept
/// an original card's value (`ZEXTEND`, `ZHECKSUM`, ...) take back its
/// name. The table's own layout and checksums, the compression keywords
/// and the default `EXTNAME = 'COMPRESSED_IMAGE'` are left out. An integer
/// image with a `ZBLANK` but no `BLANK` gets that value as its `BLANK`.
pub(crate) fn image_header(table: FitsHeader) -> FitsHeader {
    let cards = &table.cards;
    let renamed = |zkey: &str| card_value(cards, zkey).cloned();
    let mut image = Vec::new();
    match renamed("ZSIMPLE") {
        Some(simple) => image.push(("SIMPLE".to_string(), simple)),
        None => image.push(("XTENSION".to_string(), HeaderValue::String("IMAGE".into()))),
    }
    let bitpix = card_i64(cards, "ZBITPIX").unwrap_or(0);
    image.push(("BITPIX".to_string(), HeaderValue::Integer(bitpix)));
    let axes = image_axes(cards).unwrap_or_default();
    image.push(("NAXIS".to_string(), HeaderValue::Integer(axes.len() as i64)));
    for (index, length) in axes.iter().enumerate() {
        image.push((
            format!("NAXIS{}", index + 1),
            HeaderValue::Integer(*length as i64),
        ));
    }
    if renamed("ZSIMPLE").is_none() {
        image.push((
            "PCOUNT".to_string(),
            renamed("ZPCOUNT").unwrap_or(HeaderValue::Integer(0)),
        ));
        image.push((
            "GCOUNT".to_string(),
            renamed("ZGCOUNT").unwrap_or(HeaderValue::Integer(1)),
        ));
    }
    for (keyword, value) in cards {
        let table_structure = keyword == "XTENSION"
            || keyword == "BITPIX"
            || keyword.starts_with("NAXIS")
            || matches!(
                keyword.as_str(),
                "PCOUNT" | "GCOUNT" | "CHECKSUM" | "DATASUM"
            );
        let default_name = keyword == "EXTNAME" && value.as_str() == Some("COMPRESSED_IMAGE");
        if table_structure || default_name {
            continue;
        }
        match keyword.as_str() {
            "ZEXTEND" | "ZBLOCKED" | "ZHECKSUM" | "ZDATASUM" => {
                let name = RENAMED_KEYWORDS
                    .iter()
                    .find(|(zkey, _)| zkey == keyword)
                    .map(|(_, name)| *name)
                    .unwrap_or_default();
                image.push((name.to_string(), value.clone()));
            }
            _ if is_reserved_keyword(keyword) => {}
            _ => image.push((keyword.clone(), value.clone())),
        }
    }
    if bitpix > 0 && card_value(&image, "BLANK").is_none() {
        let blank_column = cards.iter().any(|(keyword, value)| {
            is_indexed(keyword, &["TTYPE"]) && value.as_str().map(str::trim) == Some("ZBLANK")
        });
        if let Some(blank) = card_value(cards, "ZBLANK") {
            image.push(("BLANK".to_string(), blank.clone()));
        } else if blank_column && bitpix > 8 {
            // Tiles name their own blank value; give them a shared one, the
            // least the sample type holds, as astropy does.
            image.push((
                "BLANK".to_string(),
                HeaderValue::Integer(i64::MIN >> (64 - bitpix)),
            ));
        }
    }
    FitsHeader {
        cards: image,
        history: table.history,
        comments: table.comments,
    }
}

/// How one table column stores its values.
#[derive(Clone, Copy, Debug)]
struct Column {
    /// Byte offset of the field within a row.
    offset: usize,
    /// The data type code: `B`, `I`, `J`, `K`, `E`, `D`, or `P`/`Q` for a
    /// variable-length array descriptor.
    kind: u8,
    /// For a descriptor, the type code of the array's elements.
    element: u8,
}

/// Bytes per element of a table data type code.
fn type_bytes(kind: u8) -> Option<usize> {
    Some(match kind {
        b'L' | b'B' | b'A' => 1,
        b'I' => 2,
        b'J' | b'E' => 4,
        b'K' | b'D' | b'C' | b'P' => 8,
        b'M' | b'Q' => 16,
        _ => return None,
    })
}

/// Parse a `TFORMn` value, `rT` or `rPt(max)`, into its repeat count, type
/// code and, for a descriptor, element type code.
fn parse_tform(tform: &str) -> Option<(usize, u8, u8)> {
    let tform = tform.trim();
    let digits = tform.bytes().take_while(u8::is_ascii_digit).count();
    let repeat = if digits == 0 {
        1
    } else {
        tform[..digits].parse().ok()?
    };
    let rest = &tform.as_bytes()[digits..];
    let kind = rest.first()?.to_ascii_uppercase();
    let element = match kind {
        b'P' | b'Q' => rest.get(1)?.to_ascii_uppercase(),
        _ => 0,
    };
    Some((repeat, kind, element))
}

/// The algorithm a table's tiles are compressed with.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Algorithm {
    Rice { blocksize: usize, bytepix: usize },
    Gzip1,
    Gzip2,
    Plio,
    None,
}

/// How quantized floating-point tiles are restored.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Dither {
    None,
    Subtractive1,
    Subtractive2,
}

/// The values one tile decodes to.
enum Tile {
    Int(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

/// A compressed image's table, read whole into memory, with what its
/// header says about the tiles.
struct Table<'a> {
    cards: &'a [(String, HeaderValue)],
    data: Vec<u8>,
    row_bytes: usize,
    rows: usize,
    heap: usize,
    compressed: Column,
    gzip: Option<Column>,
    uncompressed: Option<Column>,
    zscale: Option<Column>,
    zzero: Option<Column>,
    zblank: Option<Column>,
    algorithm: Algorithm,
    bitpix: i64,
    dither: Dither,
    dither_seed: i64,
}

fn malformed(what: impl Into<String>) -> FitsError {
    FitsError::Malformed(what.into())
}

/// A compression parameter, from the `ZNAMEi`/`ZVALi` pair that names it.
fn parameter(cards: &[(String, HeaderValue)], name: &str) -> Option<i64> {
    (1..1000)
        .map_while(|index| {
            card_value(cards, &format!("ZNAME{index}"))
                .and_then(HeaderValue::as_str)
                .map(|found| (index, found))
        })
        .find(|(_, found)| found.trim().eq_ignore_ascii_case(name))
        .and_then(|(index, _)| card_i64(cards, &format!("ZVAL{index}")))
}

/// The length of a compressed image's table data unit, heap included.
pub(crate) fn table_bytes(cards: &[(String, HeaderValue)]) -> Option<u64> {
    let row_bytes = u64::try_from(card_i64(cards, "NAXIS1")?).ok()?;
    let rows = u64::try_from(card_i64(cards, "NAXIS2")?).ok()?;
    let heap = u64::try_from(card_i64(cards, "PCOUNT").unwrap_or(0)).ok()?;
    row_bytes.checked_mul(rows)?.checked_add(heap)
}

impl<'a> Table<'a> {
    fn read(reader: &mut impl Read, cards: &'a [(String, HeaderValue)]) -> Result<Self, FitsError> {
        let bitpix = card_i64(cards, "ZBITPIX").ok_or_else(|| malformed("missing ZBITPIX"))?;
        if !matches!(bitpix, 8 | 16 | 32 | 64 | -32 | -64) {
            return Err(FitsError::Unsupported(format!("ZBITPIX {bitpix}")));
        }
        let algorithm = match card_value(cards, "ZCMPTYPE")
            .and_then(HeaderValue::as_str)
            .map(str::trim)
        {
            Some("RICE_1" | "RICE_ONE") => Algorithm::Rice {
                blocksize: usize::try_from(parameter(cards, "BLOCKSIZE").unwrap_or(32))
                    .ok()
                    .filter(|blocksize| *blocksize > 0)
                    .ok_or_else(|| malformed("invalid Rice BLOCKSIZE"))?,
                bytepix: match parameter(cards, "BYTEPIX").unwrap_or(4) {
                    1 => 1,
                    2 => 2,
                    4 => 4,
                    other => {
                        return Err(FitsError::Unsupported(format!("Rice BYTEPIX {other}")));
                    }
                },
            },
            Some("GZIP_1") => Algorithm::Gzip1,
            Some("GZIP_2") => Algorithm::Gzip2,
            Some("PLIO_1") => Algorithm::Plio,
            Some("NOCOMPRESS") => Algorithm::None,
            // HCOMPRESS_1 is a lossy wavelet coder written for the
            // Digitized Sky Survey. Its decoder is larger than all of the
            // above together, and capture software does not write it.
            Some(other) => {
                return Err(FitsError::Unsupported(format!("{other} tile compression")));
            }
            None => return Err(malformed("missing ZCMPTYPE")),
        };
        let dither = match card_value(cards, "ZQUANTIZ")
            .and_then(HeaderValue::as_str)
            .map(str::trim)
        {
            Some("SUBTRACTIVE_DITHER_1") => Dither::Subtractive1,
            Some("SUBTRACTIVE_DITHER_2") => Dither::Subtractive2,
            _ => Dither::None,
        };

        let row_bytes = card_i64(cards, "NAXIS1")
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| malformed("missing or invalid NAXIS1"))?;
        let rows = card_i64(cards, "NAXIS2")
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| malformed("missing or invalid NAXIS2"))?;
        let length = table_bytes(cards)
            .and_then(|length| usize::try_from(length).ok())
            .ok_or_else(|| malformed("implausible table size"))?;
        let heap = match card_i64(cards, "THEAP") {
            Some(heap) => usize::try_from(heap).map_err(|_| malformed("invalid THEAP"))?,
            None => row_bytes * rows,
        };

        let mut columns = Vec::new();
        let mut offset = 0_usize;
        let fields = card_i64(cards, "TFIELDS").unwrap_or(0).clamp(0, 999);
        for field in 1..=fields {
            let tform = card_value(cards, &format!("TFORM{field}"))
                .and_then(HeaderValue::as_str)
                .ok_or_else(|| malformed(format!("missing TFORM{field}")))?;
            let (repeat, kind, element) =
                parse_tform(tform).ok_or_else(|| malformed(format!("invalid TFORM{field}")))?;
            let width = if kind == b'X' {
                repeat.div_ceil(8)
            } else {
                type_bytes(kind)
                    .and_then(|bytes| bytes.checked_mul(repeat))
                    .ok_or_else(|| malformed(format!("invalid TFORM{field}")))?
            };
            let name = card_value(cards, &format!("TTYPE{field}"))
                .and_then(HeaderValue::as_str)
                .unwrap_or_default()
                .trim()
                .to_ascii_uppercase();
            if repeat > 0 {
                columns.push((
                    name,
                    Column {
                        offset,
                        kind,
                        element,
                    },
                ));
            }
            offset = offset
                .checked_add(width)
                .ok_or_else(|| malformed("implausible table row"))?;
        }
        if offset > row_bytes {
            return Err(malformed("table columns overrun NAXIS1"));
        }
        let column = |name: &str| {
            columns
                .iter()
                .find(|(found, _)| found == name)
                .map(|(_, column)| *column)
        };
        let compressed = column("COMPRESSED_DATA")
            .filter(|column| matches!(column.kind, b'P' | b'Q'))
            .ok_or_else(|| malformed("compressed image without a COMPRESSED_DATA column"))?;

        let mut data = Vec::new();
        data.try_reserve_exact(length)
            .map_err(|_| malformed("table buffer allocation failed"))?;
        data.resize(length, 0);
        read_payload_exact(reader, &mut data)?;
        Ok(Self {
            cards,
            data,
            row_bytes,
            rows,
            heap,
            compressed,
            gzip: column("GZIP_COMPRESSED_DATA")
                .filter(|column| matches!(column.kind, b'P' | b'Q')),
            uncompressed: column("UNCOMPRESSED_DATA")
                .filter(|column| matches!(column.kind, b'P' | b'Q')),
            zscale: column("ZSCALE"),
            zzero: column("ZZERO"),
            zblank: column("ZBLANK"),
            algorithm,
            bitpix,
            dither,
            dither_seed: card_i64(cards, "ZDITHER0").unwrap_or(1),
        })
    }

    /// The bytes of one row's field.
    fn field(&self, row: usize, column: Column, bytes: usize) -> Result<&[u8], FitsError> {
        let start = row * self.row_bytes + column.offset;
        self.data
            .get(start..start + bytes)
            .ok_or_else(|| malformed("table row runs past its data"))
    }

    /// A row's variable-length array in the heap, as raw bytes, with the
    /// byte length of its elements.
    fn array(&self, row: usize, column: Column) -> Result<(&[u8], usize), FitsError> {
        let (count, offset) = if column.kind == b'P' {
            let field = self.field(row, column, 8)?;
            let count = i32::from_be_bytes(field[..4].try_into().unwrap());
            let offset = i32::from_be_bytes(field[4..].try_into().unwrap());
            (i64::from(count), i64::from(offset))
        } else {
            let field = self.field(row, column, 16)?;
            let count = i64::from_be_bytes(field[..8].try_into().unwrap());
            let offset = i64::from_be_bytes(field[8..].try_into().unwrap());
            (count, offset)
        };
        let element = type_bytes(column.element).ok_or_else(|| malformed("invalid array type"))?;
        let bounds = usize::try_from(count)
            .ok()
            .zip(usize::try_from(offset).ok())
            .and_then(|(count, offset)| {
                let start = self.heap.checked_add(offset)?;
                Some(start..start.checked_add(count.checked_mul(element)?)?)
            })
            .ok_or_else(|| malformed("invalid heap descriptor"))?;
        let bytes = self
            .data
            .get(bounds)
            .ok_or_else(|| malformed("heap descriptor runs past the heap"))?;
        Ok((bytes, element))
    }

    /// A row's scalar numeric field, or the keyword of the same name.
    fn scalar(
        &self,
        row: usize,
        column: Option<Column>,
        keyword: &str,
    ) -> Result<Option<f64>, FitsError> {
        let Some(column) = column else {
            return Ok(card_value(self.cards, keyword).and_then(HeaderValue::as_f64));
        };
        let bytes = type_bytes(column.kind).ok_or_else(|| malformed("invalid column type"))?;
        let field = self.field(row, column, bytes)?;
        Ok(Some(match column.kind {
            b'B' => f64::from(field[0]),
            b'I' => f64::from(i16::from_be_bytes(field.try_into().unwrap())),
            b'J' => f64::from(i32::from_be_bytes(field.try_into().unwrap())),
            b'K' => i64::from_be_bytes(field.try_into().unwrap()) as f64,
            b'E' => f64::from(f32::from_be_bytes(field.try_into().unwrap())),
            b'D' => f64::from_be_bytes(field.try_into().unwrap()),
            _ => return Err(malformed(format!("unsupported {keyword} column type"))),
        }))
    }

    /// The integer that marks a blank pixel in a row's tile, if any.
    fn tile_blank(&self, row: usize, header_blank: Option<i64>) -> Result<Option<i64>, FitsError> {
        if self.zblank.is_some() {
            return Ok(self
                .scalar(row, self.zblank, "ZBLANK")?
                .map(|value| value as i64));
        }
        Ok(card_i64(self.cards, "ZBLANK").or(header_blank))
    }

    /// Decode one row's tile of `pixels` samples.
    fn tile(
        &self,
        row: usize,
        pixels: usize,
        header_blank: Option<i64>,
    ) -> Result<Tile, FitsError> {
        let quantized = self.bitpix < 0
            && (self.zscale.is_some() || card_value(self.cards, "ZSCALE").is_some());
        let (compressed, element) = self.array(row, self.compressed)?;
        if compressed.is_empty() {
            // A tile that would not quantize well is kept losslessly,
            // gzipped or raw.
            if let Some(column) = self.gzip {
                let (bytes, _) = self.array(row, column)?;
                return values(&inflate(bytes, pixels)?, pixels, self.bitpix < 0);
            }
            if let Some(column) = self.uncompressed {
                let (bytes, element) = self.array(row, column)?;
                return raw_values(bytes, element, column.element, pixels);
            }
            return Err(malformed(format!("tile {} holds no data", row + 1)));
        }
        let lossless_floats = self.bitpix < 0 && !quantized;
        let decoded = match self.algorithm {
            Algorithm::Rice { blocksize, bytepix } => {
                Tile::Int(rice_decode(compressed, blocksize, bytepix, pixels)?)
            }
            Algorithm::Gzip1 => values(&inflate(compressed, pixels)?, pixels, lossless_floats)?,
            Algorithm::Gzip2 => {
                let shuffled = inflate(compressed, pixels)?;
                let width = shuffled.len() / pixels.max(1);
                values(&unshuffle(&shuffled, width), pixels, lossless_floats)?
            }
            Algorithm::Plio => {
                if element != 2 {
                    return Err(malformed("PLIO_1 data must be 16-bit words"));
                }
                Tile::Int(plio_decode(compressed, pixels)?)
            }
            Algorithm::None => values(compressed, pixels, lossless_floats)?,
        };
        let blank = self.tile_blank(row, header_blank)?;
        match decoded {
            Tile::Int(ints) if quantized => {
                let scale = self.scalar(row, self.zscale, "ZSCALE")?.unwrap_or(1.0);
                let zero = self.scalar(row, self.zzero, "ZZERO")?.unwrap_or(0.0);
                Ok(self.unquantize(row, &ints, scale, zero, blank))
            }
            Tile::Int(mut ints) if self.bitpix > 0 => {
                // A per-tile ZBLANK marks blanks with its own value; give
                // them the image's BLANK.
                if let (Some(tile_blank), Some(image_blank)) = (blank, header_blank)
                    && tile_blank != image_blank
                {
                    for value in &mut ints {
                        if *value == tile_blank {
                            *value = image_blank;
                        }
                    }
                }
                Ok(Tile::Int(ints))
            }
            Tile::Int(_) => Err(malformed(
                "floating-point tile holds integers but no ZSCALE to restore them",
            )),
            floats if self.bitpix < 0 => Ok(floats),
            _ => Err(malformed("integer image tile holds floating-point data")),
        }
    }

    /// Restore quantized samples: `(q - r + 0.5) × scale + zero` with the
    /// dither `r` drawn from cfitsio's random sequence, or `q × scale +
    /// zero` without dithering. SUBTRACTIVE_DITHER_2 keeps exact zeros, and
    /// blank samples become NaN.
    fn unquantize(
        &self,
        row: usize,
        ints: &[i64],
        scale: f64,
        zero: f64,
        blank: Option<i64>,
    ) -> Tile {
        let random = random_sequence();
        // The tile's place in the sequence follows from its row and ZDITHER0.
        let mut seed = (row as i64 + self.dither_seed - 1).rem_euclid(RANDOM_COUNT as i64) as usize;
        let mut next = (random[seed] * 500.0) as usize;
        let mut restore = |value: i64| -> f64 {
            let restored = match self.dither {
                Dither::None => value as f64 * scale + zero,
                Dither::Subtractive2 if value == ZERO_VALUE => 0.0,
                _ => (value as f64 - f64::from(random[next]) + 0.5) * scale + zero,
            };
            if self.dither != Dither::None {
                next += 1;
                if next == RANDOM_COUNT {
                    seed = (seed + 1) % RANDOM_COUNT;
                    next = (random[seed] * 500.0) as usize;
                }
            }
            if Some(value) == blank {
                f64::NAN
            } else {
                restored
            }
        };
        if self.bitpix == -32 {
            Tile::F32(ints.iter().map(|&value| restore(value) as f32).collect())
        } else {
            Tile::F64(ints.iter().map(|&value| restore(value)).collect())
        }
    }
}

/// Inflate a gzip or zlib stream, expecting at most eight bytes a pixel.
fn inflate(bytes: &[u8], pixels: usize) -> Result<Vec<u8>, FitsError> {
    let limit = pixels.saturating_mul(8) as u64 + 1;
    let mut out = Vec::new();
    let result = if bytes.starts_with(&[0x1f, 0x8b]) {
        flate2::read::MultiGzDecoder::new(bytes)
            .take(limit)
            .read_to_end(&mut out)
    } else {
        flate2::read::ZlibDecoder::new(bytes)
            .take(limit)
            .read_to_end(&mut out)
    };
    result.map_err(|error| malformed(format!("GZIP tile: {error}")))?;
    Ok(out)
}

/// Undo GZIP_2's byte shuffle, which stores every value's first byte, then
/// every value's second byte, and so on.
fn unshuffle(shuffled: &[u8], width: usize) -> Vec<u8> {
    if width <= 1 {
        return shuffled.to_vec();
    }
    let count = shuffled.len() / width;
    let mut out = vec![0; count * width];
    for (byte, plane) in shuffled.chunks_exact(count.max(1)).take(width).enumerate() {
        for (index, &value) in plane.iter().enumerate() {
            out[index * width + byte] = value;
        }
    }
    out
}

/// Big-endian samples whose width the byte count gives away, as cfitsio
/// and astropy read GZIP tiles: 1 byte is unsigned, 2 bytes signed, and 4
/// or 8 bytes are floats for losslessly kept floating-point data, signed
/// integers otherwise.
fn values(bytes: &[u8], pixels: usize, floats: bool) -> Result<Tile, FitsError> {
    let width = bytes
        .len()
        .checked_div(pixels)
        .filter(|width| width * pixels == bytes.len())
        .ok_or_else(|| malformed("tile size does not match its pixel count"))?;
    Ok(match (width, floats) {
        (1, _) => Tile::Int(bytes.iter().map(|&byte| i64::from(byte)).collect()),
        (2, _) => Tile::Int(
            bytes
                .chunks_exact(2)
                .map(|chunk| i64::from(i16::from_be_bytes([chunk[0], chunk[1]])))
                .collect(),
        ),
        (4, false) => Tile::Int(
            bytes
                .chunks_exact(4)
                .map(|chunk| i64::from(i32::from_be_bytes(chunk.try_into().unwrap())))
                .collect(),
        ),
        (4, true) => Tile::F32(
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_be_bytes(chunk.try_into().unwrap()))
                .collect(),
        ),
        (8, false) => Tile::Int(
            bytes
                .chunks_exact(8)
                .map(|chunk| i64::from_be_bytes(chunk.try_into().unwrap()))
                .collect(),
        ),
        (8, true) => Tile::F64(
            bytes
                .chunks_exact(8)
                .map(|chunk| f64::from_be_bytes(chunk.try_into().unwrap()))
                .collect(),
        ),
        _ => return Err(malformed(format!("tile holds {width}-byte samples"))),
    })
}

/// An `UNCOMPRESSED_DATA` array in its column's element type.
fn raw_values(bytes: &[u8], element: usize, kind: u8, pixels: usize) -> Result<Tile, FitsError> {
    if bytes.len() != pixels * element {
        return Err(malformed("tile size does not match its pixel count"));
    }
    match kind {
        b'B' | b'I' | b'J' | b'K' => values(bytes, pixels, false),
        b'E' | b'D' => values(bytes, pixels, true),
        _ => Err(malformed("unsupported UNCOMPRESSED_DATA type")),
    }
}

/// Bits read most significant first.
struct Bits<'a> {
    data: &'a [u8],
    next: usize,
    /// Unread bits, left-aligned; the rest are zero.
    buffer: u64,
    available: u32,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            next: 0,
            buffer: 0,
            available: 0,
        }
    }

    fn refill(&mut self) {
        while self.available <= 56 && self.next < self.data.len() {
            self.buffer |= u64::from(self.data[self.next]) << (56 - self.available);
            self.available += 8;
            self.next += 1;
        }
    }

    fn consume(&mut self, count: u32) {
        self.buffer = self.buffer.checked_shl(count).unwrap_or(0);
        self.available -= count;
    }

    /// The next `count` bits, `count` at most 32.
    fn read(&mut self, count: u32) -> Option<u32> {
        if count == 0 {
            return Some(0);
        }
        if self.available < count {
            self.refill();
            if self.available < count {
                return None;
            }
        }
        let value = (self.buffer >> (64 - count)) as u32;
        self.consume(count);
        Some(value)
    }

    /// The number of zero bits before the next one bit, which is skipped.
    fn unary(&mut self) -> Option<u32> {
        let mut zeros = 0_u32;
        loop {
            if self.buffer == 0 {
                zeros = zeros.checked_add(self.available)?;
                self.buffer = 0;
                self.available = 0;
                self.refill();
                if self.available == 0 {
                    return None;
                }
                continue;
            }
            let leading = self.buffer.leading_zeros();
            self.consume(leading + 1);
            return zeros.checked_add(leading);
        }
    }
}

/// Decode a Rice-coded tile of `count` samples of `bytepix` bytes, as
/// cfitsio's `fits_rdecomp` and its 16- and 8-bit kin do.
///
/// The first sample is stored whole. Each block of `blocksize` samples then
/// opens with its split level: below zero means every difference in the
/// block is zero, the top level means the differences follow uncoded, and
/// the rest are Rice codes. A difference is mapped to unsigned (even for
/// positive, odd for negative) and added to the previous sample, wrapping
/// at the sample width. One-byte samples are unsigned, wider ones signed.
fn rice_decode(
    input: &[u8],
    blocksize: usize,
    bytepix: usize,
    count: usize,
) -> Result<Vec<i64>, FitsError> {
    let (fs_bits, fs_max, raw_bits) = match bytepix {
        1 => (3, 6, 8),
        2 => (4, 14, 16),
        _ => (5, 25, 32),
    };
    let mask = u32::MAX >> (32 - raw_bits);
    let truncated = || malformed("Rice tile ends early");
    let first = input.get(..bytepix).ok_or_else(truncated)?;
    let mut last = first
        .iter()
        .fold(0_u32, |value, &byte| (value << 8) | u32::from(byte));
    let mut bits = Bits::new(&input[bytepix..]);
    let mut out = Vec::with_capacity(count);
    let undo = |diff: u32| {
        if diff & 1 == 0 {
            diff >> 1
        } else {
            !(diff >> 1)
        }
    };
    while out.len() < count {
        let end = (out.len() + blocksize).min(count);
        let fs = bits.read(fs_bits).ok_or_else(truncated)? as i32 - 1;
        if fs < 0 {
            out.resize(end, last);
            continue;
        }
        while out.len() < end {
            let diff = if fs == fs_max {
                bits.read(raw_bits).ok_or_else(truncated)?
            } else {
                let zeros = bits.unary().ok_or_else(truncated)?;
                let low = bits.read(fs as u32).ok_or_else(truncated)?;
                zeros.wrapping_shl(fs as u32) | low
            };
            last = undo(diff).wrapping_add(last) & mask;
            out.push(last);
        }
    }
    Ok(out
        .into_iter()
        .map(|value| match bytepix {
            1 => i64::from(value as u8),
            2 => i64::from(value as u16 as i16),
            _ => i64::from(value as i32),
        })
        .collect())
}

/// Decode an IRAF PLIO line list of 16-bit big-endian words into `count`
/// samples, as cfitsio's `pl_l2pi` does.
///
/// A short header gives the list length and where the instructions start.
/// Each word holds a 4-bit opcode and 12 bits of data; the opcodes write
/// runs of zeros or of the current value, set or step the current value,
/// or step it and write one sample.
fn plio_decode(input: &[u8], count: usize) -> Result<Vec<i64>, FitsError> {
    let words: Vec<i32> = input
        .chunks_exact(2)
        .map(|chunk| i32::from(i16::from_be_bytes([chunk[0], chunk[1]])))
        .collect();
    let word = |index: usize| -> Result<i32, FitsError> {
        words
            .get(index)
            .copied()
            .ok_or_else(|| malformed("PLIO line list ends early"))
    };
    // Header words, counted from 1 as in the IRAF original.
    let (length, first) = if word(2)? > 0 {
        (word(2)? as i64, 4_i64)
    } else {
        (
            (i64::from(word(4)?) << 15) + i64::from(word(3)?),
            i64::from(word(1)?) + 1,
        )
    };
    let mut out = vec![0_i64; count];
    let mut written = 0_usize;
    let mut x = 1_i64;
    let end = count as i64;
    let mut value = 1_i64;
    let mut index = first;
    while index <= length && x <= end {
        let instruction = word(index as usize - 1)?;
        let opcode = instruction / 4096;
        let data = i64::from(instruction & 4095);
        match opcode {
            // A run of zeros, a run of the current value, or zeros ending
            // in one sample of the current value.
            0 | 4 | 5 => {
                let last = x + data - 1;
                let run = (last.min(end) - x + 1).max(0) as usize;
                if run > 0 {
                    if opcode == 4 {
                        out[written..written + run].fill(value);
                    } else if opcode == 5 && last <= end {
                        out[written + run - 1] = value;
                    }
                    written += run;
                }
                x = last + 1;
            }
            1 => {
                value = (i64::from(word(index as usize)?) << 12) + data;
                index += 1;
            }
            2 => value += data,
            3 => value -= data,
            6 | 7 => {
                value += if opcode == 6 { data } else { -data };
                out[written] = value;
                written += 1;
                x += 1;
            }
            _ => {}
        }
        index += 1;
    }
    Ok(out)
}

/// How many values cfitsio's dither sequence holds.
const RANDOM_COUNT: usize = 10_000;

/// cfitsio's sequence of uniform random numbers in (0, 1), from the Park
/// and Miller generator with seed 1, which subtractive dithering draws on.
fn random_sequence() -> &'static [f32] {
    static SEQUENCE: OnceLock<Vec<f32>> = OnceLock::new();
    SEQUENCE.get_or_init(|| {
        let (a, m) = (16_807.0_f64, 2_147_483_647.0_f64);
        let mut seed = 1.0_f64;
        (0..RANDOM_COUNT)
            .map(|_| {
                let product = a * seed;
                seed = product - m * (product / m).trunc();
                (seed / m) as f32
            })
            .collect()
    })
}

/// The image's sample count, as `ImageSpec` reads it: `width × height ×
/// planes`, with planes beyond the third left out.
pub(crate) struct Shape {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) planes: usize,
}

/// Decompress the tiles that cover the image `shape` describes and return
/// the data unit it would have uncompressed: `BITPIX`-sized big-endian
/// samples, planes in order. `header_blank` is the image's `BLANK`.
pub(crate) fn decompress(
    reader: &mut impl Read,
    cards: &[(String, HeaderValue)],
    shape: &Shape,
    header_blank: Option<i64>,
) -> Result<Vec<u8>, FitsError> {
    let table = Table::read(reader, cards)?;
    let axes = image_axes(cards).ok_or_else(|| malformed("invalid ZNAXISn"))?;
    let tiles: Vec<u64> = (1..=axes.len())
        .map(|axis| match card_i64(cards, &format!("ZTILE{axis}")) {
            Some(length) => u64::try_from(length).ok().filter(|length| *length > 0),
            // Without ZTILEn, each tile is one row.
            None => Some(if axis == 1 { axes[0] } else { 1 }),
        })
        .collect::<Option<_>>()
        .ok_or_else(|| malformed("invalid ZTILEn"))?;
    let counts: Vec<u64> = axes
        .iter()
        .zip(&tiles)
        .map(|(axis, tile)| axis.div_ceil(*tile))
        .collect();
    let total = counts
        .iter()
        .try_fold(1_u64, |product, &count| product.checked_mul(count))
        .ok_or_else(|| malformed("implausible tile count"))?;
    if (table.rows as u64) < total {
        return Err(malformed(
            "compressed image has fewer tiles than its size needs",
        ));
    }

    let bytes_per_sample = (table.bitpix.unsigned_abs() / 8) as usize;
    let length = shape.width * shape.height * shape.planes * bytes_per_sample;
    let mut out = Vec::new();
    out.try_reserve_exact(length)
        .map_err(|_| malformed("pixel buffer allocation failed"))?;
    out.resize(length, 0);

    for row in 0..total as usize {
        // The tile's index along each axis, the first axis varying fastest.
        let mut rest = row as u64;
        let mut start = Vec::with_capacity(axes.len());
        let mut extent = Vec::with_capacity(axes.len());
        for axis in 0..axes.len() {
            let index = rest % counts[axis];
            rest /= counts[axis];
            start.push(index * tiles[axis]);
            extent.push(tiles[axis].min(axes[axis] - index * tiles[axis]));
        }
        // Only the first three planes, and the first of any higher axis,
        // are kept.
        let plane_start = start.get(2).copied().unwrap_or(0) as usize;
        if plane_start >= shape.planes || start.iter().skip(3).any(|&start| start != 0) {
            continue;
        }
        let pixels = extent
            .iter()
            .try_fold(1_u64, |product, &length| product.checked_mul(length))
            .and_then(|pixels| usize::try_from(pixels).ok())
            .ok_or_else(|| malformed("implausible tile size"))?;
        let tile = table.tile(row, pixels, header_blank)?;
        let tile_width = extent[0] as usize;
        let tile_height = extent[1] as usize;
        let tile_planes = extent.get(2).copied().unwrap_or(1) as usize;
        let (x0, y0) = (start[0] as usize, start[1] as usize);
        for plane in 0..tile_planes.min(shape.planes - plane_start) {
            for y in 0..tile_height {
                let source = (plane * tile_height + y) * tile_width;
                let target = ((plane_start + plane) * shape.height + y0 + y) * shape.width + x0;
                let destination =
                    &mut out[target * bytes_per_sample..(target + tile_width) * bytes_per_sample];
                write_samples(&tile, source, tile_width, table.bitpix, destination);
            }
        }
    }
    Ok(out)
}

/// Write `count` of a tile's values from `source` as big-endian `bitpix`
/// samples.
fn write_samples(tile: &Tile, source: usize, count: usize, bitpix: i64, out: &mut [u8]) {
    let range = source..source + count;
    match (tile, bitpix) {
        (Tile::Int(values), _) => {
            for (value, out) in values[range]
                .iter()
                .zip(out.chunks_exact_mut((bitpix.unsigned_abs() / 8) as usize))
            {
                match bitpix {
                    8 => out.copy_from_slice(&[*value as u8]),
                    16 => out.copy_from_slice(&(*value as i16).to_be_bytes()),
                    32 => out.copy_from_slice(&(*value as i32).to_be_bytes()),
                    _ => out.copy_from_slice(&value.to_be_bytes()),
                }
            }
        }
        (Tile::F32(values), -32) => {
            for (value, out) in values[range].iter().zip(out.chunks_exact_mut(4)) {
                out.copy_from_slice(&value.to_be_bytes());
            }
        }
        (Tile::F32(values), _) => {
            for (value, out) in values[range].iter().zip(out.chunks_exact_mut(8)) {
                out.copy_from_slice(&f64::from(*value).to_be_bytes());
            }
        }
        (Tile::F64(values), -64) => {
            for (value, out) in values[range].iter().zip(out.chunks_exact_mut(8)) {
                out.copy_from_slice(&value.to_be_bytes());
            }
        }
        (Tile::F64(values), _) => {
            for (value, out) in values[range].iter().zip(out.chunks_exact_mut(4)) {
                out.copy_from_slice(&(*value as f32).to_be_bytes());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(text: &str) -> Vec<u8> {
        (0..text.len())
            .step_by(2)
            .map(|index| u8::from_str_radix(&text[index..index + 2], 16).unwrap())
            .collect()
    }

    /// Rice tiles of 96 samples, made by astropy with cfitsio's encoder:
    /// a block of one value, a block of noise across the whole range, and a
    /// ramp, one block for each kind of Rice block.
    const RICE_U8: &str = "641f13070f0f0f070f0f0f0f070f0f0f0f070f0f0f070f0f0f0f070f0f0f070f0f0d800333333333333333333333333333333330";
    const RICE_I16: &str = "fb2e0f687dc391c38fc391c38fc391c38fc391c38fc391c391c38fc391c38fc391c38fc391c38fc391c38fc391c38fc391c38fc391c38fc391c38fc391c391c38fc391900000000a841a0d068341a0d068341a0d068341a0d068341a0d068341a0d068341a0d068341a0d060";
    const RICE_I32: &str = "0001e240069c89560ff0e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e4432770e443277200000001b222b2800003400001a00000d000006800003400001a00000d000006800003400001a00000d000006800003400001a00000d000006800003400001a00000d000006800003400001a00000d000006800003400001a00000d000006800003400001a00000c";

    fn rice_samples(bytepix: usize) -> Vec<i64> {
        (0..96_u64)
            .map(|index| {
                let hash = (index * 2_654_435_761) % (1 << 32);
                match (index, bytepix) {
                    (0..32, 1) => 100,
                    (0..32, 2) => -1234,
                    (0..32, _) => 123_456,
                    (32..64, 1) => (hash >> 24) as i64,
                    (32..64, 2) => i64::from((hash >> 16) as u16 as i16),
                    (32..64, _) => i64::from(hash as u32 as i32),
                    (_, 1) => 10 + 3 * (index as i64 - 64),
                    (_, 2) => -50 + 3 * (index as i64 - 64),
                    _ => 1000 + 3 * (index as i64 - 64),
                }
            })
            .collect()
    }

    #[test]
    fn rice_tiles_from_cfitsio_decode_exactly() {
        for (bytepix, tile) in [(1, RICE_U8), (2, RICE_I16), (4, RICE_I32)] {
            let tile = unhex(tile);
            assert_eq!(
                rice_decode(&tile, 32, bytepix, 96).unwrap(),
                rice_samples(bytepix),
                "{bytepix} bytes a sample"
            );
            // A tile cut short is an error at every length, never a panic.
            for length in 0..tile.len() - 1 {
                assert!(rice_decode(&tile[..length], 32, bytepix, 96).is_err());
            }
        }
    }

    #[test]
    fn plio_lists_from_cfitsio_decode_exactly() {
        // Runs of zeros and of values, values past 4095 that need the
        // set-high instruction, and single-sample steps.
        let list = unhex(
            "00000007ff9c00140000000000002004000a401460021388000100134014700a117000110013400a",
        );
        let mut expected = vec![0; 100];
        expected[10..30].fill(5);
        expected[30] = 7;
        expected[50..70].fill(5000);
        expected[70] = 4990;
        expected[90..].fill(70_000);
        assert_eq!(plio_decode(&list, 100).unwrap(), expected);
        assert_eq!(plio_decode(&list, 40).unwrap(), expected[..40]);
        for length in 0..list.len() {
            let _ = plio_decode(&list[..length], 100);
        }
    }

    #[test]
    fn the_dither_sequence_matches_cfitsio() {
        let sequence = random_sequence();
        assert_eq!(sequence.len(), RANDOM_COUNT);
        assert_eq!(sequence[0], (16_807.0 / 2_147_483_647.0_f64) as f32);
        // cfitsio checks its generator by the seed it ends on.
        assert_eq!(
            sequence[9_999],
            (1_043_618_065.0 / 2_147_483_647.0_f64) as f32
        );
    }

    #[test]
    fn gzip_2_unshuffles_by_sample_width() {
        assert_eq!(unshuffle(&[1, 3, 5, 2, 4, 6], 2), [1, 2, 3, 4, 5, 6]);
        assert_eq!(
            unshuffle(&[1, 5, 2, 6, 3, 7, 4, 8], 4),
            [1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(unshuffle(&[9, 8, 7], 1), [9, 8, 7]);
    }

    #[test]
    fn table_formats_and_reserved_keywords_parse() {
        assert_eq!(parse_tform("1PB(176)"), Some((1, b'P', b'B')));
        assert_eq!(parse_tform("QI"), Some((1, b'Q', b'I')));
        assert_eq!(parse_tform(" 1D "), Some((1, b'D', 0)));
        assert_eq!(parse_tform("16X"), Some((16, b'X', 0)));
        assert_eq!(parse_tform(""), None);

        for keyword in [
            "TFORM1", "TTYPE12", "ZTILE2", "ZNAXIS1", "ZNAME3", "ZVAL1", "ZCMPTYPE", "ZBITPIX",
            "ZHECKSUM", "THEAP",
        ] {
            assert!(is_reserved_keyword(keyword), "{keyword}");
        }
        for keyword in [
            "TELESCOP", "TIMESYS", "ZD", "ZTILE", "TFORM0", "OBJECT", "ZENITH",
        ] {
            assert!(!is_reserved_keyword(keyword), "{keyword}");
        }
    }

    fn card(keyword: &str, value: &str) -> String {
        format!("{keyword:<8}= {value:>20}")
    }

    /// One HDU: the cards and END, then the data, each padded to a block.
    fn unit(cards: &[String], data: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for text in cards.iter().map(String::as_str).chain(["END"]) {
            bytes.extend_from_slice(format!("{text:<80}").as_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(2880), b' ');
        bytes.extend_from_slice(data);
        bytes.resize(bytes.len().next_multiple_of(2880), 0);
        bytes
    }

    /// A heap array column: name, element type code, element bytes, and
    /// each row's array as bytes.
    type Array<'a> = (&'a str, char, usize, Vec<Vec<u8>>);

    /// An empty primary, then a tile-compressed image's table holding
    /// `arrays` in its heap and `scalars` (name, TFORM, each row's field)
    /// in its rows.
    fn compressed_file(
        image: &[String],
        arrays: &[Array],
        scalars: &[(&str, &str, Vec<Vec<u8>>)],
    ) -> Vec<u8> {
        let rows = arrays[0].3.len();
        let mut fields = vec![Vec::new(); rows];
        let mut heap = Vec::new();
        let mut columns = Vec::new();
        for (name, kind, size, values) in arrays {
            let longest = values
                .iter()
                .map(|value| value.len() / size)
                .max()
                .unwrap_or(0);
            columns.push((name.to_string(), format!("'1P{kind}({longest})'")));
            for (row, value) in values.iter().enumerate() {
                fields[row].extend(((value.len() / size) as i32).to_be_bytes());
                fields[row].extend((heap.len() as i32).to_be_bytes());
                heap.extend_from_slice(value);
            }
        }
        for (name, tform, values) in scalars {
            columns.push((name.to_string(), format!("'{tform}'")));
            for (row, value) in values.iter().enumerate() {
                fields[row].extend_from_slice(value);
            }
        }
        let mut cards = vec![
            "XTENSION= 'BINTABLE'".to_string(),
            card("BITPIX", "8"),
            card("NAXIS", "2"),
            card("NAXIS1", &fields[0].len().to_string()),
            card("NAXIS2", &rows.to_string()),
            card("PCOUNT", &heap.len().to_string()),
            card("GCOUNT", "1"),
            card("TFIELDS", &columns.len().to_string()),
        ];
        for (index, (name, tform)) in columns.iter().enumerate() {
            cards.push(card(&format!("TTYPE{}", index + 1), &format!("'{name}'")));
            cards.push(card(&format!("TFORM{}", index + 1), tform));
        }
        cards.push(card("ZIMAGE", "T"));
        cards.extend_from_slice(image);
        let primary = [card("SIMPLE", "T"), card("BITPIX", "8"), card("NAXIS", "0")];
        [
            unit(&primary, &[]),
            unit(&cards, &[fields.concat(), heap].concat()),
        ]
        .concat()
    }

    fn image_cards(bitpix: i64, axes: &[usize], tiles: &[usize], algorithm: &str) -> Vec<String> {
        let mut cards = vec![
            card("ZBITPIX", &bitpix.to_string()),
            card("ZNAXIS", &axes.len().to_string()),
        ];
        for (index, length) in axes.iter().enumerate() {
            cards.push(card(&format!("ZNAXIS{}", index + 1), &length.to_string()));
        }
        for (index, length) in tiles.iter().enumerate() {
            cards.push(card(&format!("ZTILE{}", index + 1), &length.to_string()));
        }
        cards.push(card("ZCMPTYPE", &format!("'{algorithm}'")));
        cards
    }

    fn gzip(bytes: &[u8]) -> Vec<u8> {
        use std::io::Write;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(bytes).unwrap();
        encoder.finish().unwrap()
    }

    fn shuffle(bytes: &[u8], width: usize) -> Vec<u8> {
        (0..width)
            .flat_map(|byte| bytes.iter().skip(byte).step_by(width).copied())
            .collect()
    }

    fn compressed_column(tiles: Vec<Vec<u8>>) -> Array<'static> {
        ("COMPRESSED_DATA", 'B', 1, tiles)
    }

    fn assert_bits(actual: &[f32], expected: &[f32]) {
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(actual), bits(expected), "{actual:?} vs {expected:?}");
    }

    #[test]
    fn integer_tiles_decode_like_the_plain_image() {
        // 5x3 unsigned 16-bit samples in 2x2 tiles; the last column and the
        // last row of tiles are partial.
        let samples: Vec<u16> = (0..15).map(|index| (index * 4_099) as u16).collect();
        let tile_bytes = |tx: usize, ty: usize| -> Vec<u8> {
            (ty * 2..(ty * 2 + 2).min(3))
                .flat_map(|y| (tx * 2..(tx * 2 + 2).min(5)).map(move |x| (x, y)))
                .flat_map(|(x, y)| ((samples[y * 5 + x] ^ 0x8000) as i16).to_be_bytes())
                .collect()
        };
        for algorithm in ["GZIP_1", "GZIP_2", "NOCOMPRESS"] {
            let tiles = (0..2)
                .flat_map(|ty| (0..3).map(move |tx| (tx, ty)))
                .map(|(tx, ty)| {
                    let raw = tile_bytes(tx, ty);
                    match algorithm {
                        "GZIP_1" => gzip(&raw),
                        "GZIP_2" => gzip(&shuffle(&raw, 2)),
                        _ => raw,
                    }
                })
                .collect();
            let mut image = image_cards(16, &[5, 3], &[2, 2], algorithm);
            image.extend([card("BZERO", "32768"), card("BSCALE", "1")]);
            let file = compressed_file(&image, &[compressed_column(tiles)], &[]);
            let decoded = crate::FitsImage::from_bytes(&file).unwrap();
            assert_eq!((decoded.width, decoded.height, decoded.planes), (5, 3, 1));
            assert!(
                matches!(decoded.pixels, crate::Pixels::U16(ref values) if *values == samples),
                "{algorithm}"
            );
        }
    }

    #[test]
    fn rice_images_honor_bytepix_and_blocksize() {
        let mut image = image_cards(16, &[96, 1], &[96, 1], "RICE_1");
        image.extend([
            card("ZNAME1", "'BLOCKSIZE'"),
            card("ZVAL1", "32"),
            card("ZNAME2", "'BYTEPIX'"),
            card("ZVAL2", "2"),
        ]);
        let file = compressed_file(&image, &[compressed_column(vec![unhex(RICE_I16)])], &[]);
        let decoded = crate::FitsImage::from_bytes(&file).unwrap();
        let expected: Vec<f32> = rice_samples(2).iter().map(|&value| value as f32).collect();
        assert_eq!(decoded.into_physical_f32(), expected);
    }

    /// Floating-point tiles quantized by astropy (cfitsio's quantizer) with
    /// each dithering method: the RICE tile, ZSCALE, ZZERO, and the samples
    /// astropy restores. The samples hold a NaN and exact zeros.
    const QUANTIZED: &[(&str, f64, f64, &str, &str)] = &[
        (
            "NO_DITHER",
            0.16353738764362336,
            0.0,
            "00000266d0000000000000003000000047ffffd937ffffd888000000280000000800002728000000000000264000000048000000180000006800000058000000200000002000000008000000400000009000000060000000200000007000000070000000000000002800000010000000180000007800000068000000280000004800000078fa4820662043798a3bb8a2e9d82198609ab5",
            "42c8d2ec42c9ce1d42cb1d0a7fc0000042ce625a42cd672842cd136d000000000000000042c82b7642c688ce42c5e15742c3973942c1a0d642c2484c42c2efc342c29c0842c3eaf442c6dc8942c8d2ec42c97a6242cbc48042ce0e9f42ce0e9f42cd136d42cd672842ccbfb242ca21d942c7d7ba42c6dc8942c539e142c29c0842c1f49142c29c0842c29c0842c29c0842c4926b42c783ff42c87f3142c9ce1d42cc6bf742ce0e9f42cd672842cd672842cdbae342cc183c42c97a6242c82b7642c6dc8942c4926b42c2484c42c29c0842c29c0842c1f49142c2efc342c58d9c42c7304442c82b7642ca759442cd136d42cd672842cd136d42ce0e9f42cdbae3",
        ),
        (
            "SUBTRACTIVE_DITHER_1",
            0.16353738764362336,
            0.0,
            "00000266d0000000000000003000000037ffffd947ffffd878000000380000001800002718000000000000264000000038000000280000006800000058000000100000003000000000000000300000009000000050000000300000007000000070000000000000002800000010000000180000006800000078000000280000004800000078d5a418621021bcc51cbc73c9d82088304d1280",
            "42c8c92e42c9f7ee42caf31a7fc0000042cea30c42cd764c42ccc696bcf47cb9bcf1e60742c8442342c6c05e42c5cfd942c37bb342c1c2c942c2179742c2d9a842c2e4e942c402ea42c6f48542c8a24542c983b142cb9ea442cde92542ce17d842cd26c042cd571142ccd1c742ca5e6542c7bff642c6e30742c54c1f42c2b5a742c1c9c142c2b5d542c2739142c2b17642c4cb6042c74eb542c887b842c9c5a842cc870542cdf32042cd83d142cd3e4d42cdcf3942cc2e8a42c99bbd42c7ef3942c724fc42c4820142c2835942c28b9942c286da42c1e66042c2e49742c5b2c042c77bc642c8425142ca84c042cd02b242cd873142cd512c42ce0c6f42cdb660",
        ),
        (
            "SUBTRACTIVE_DITHER_2",
            0.16353738764362336,
            351193864.00240713,
            "800002714404343003b400868783007ac00036a1e16362e0a1a021a4a2a1a3a3a02160a0e363e16263c6ad20c310810de628e5e39e4ec10441826894",
            "42c8c92e42c9f7ee42caf31a7fc0000042cea30c42cd764c42ccc696000000000000000042c8442342c6c05e42c5cfd942c37bb342c1c2c942c2179742c2d9a842c2e4e942c402ea42c6f48542c8a24542c983b142cb9ea442cde92542ce17d842cd26c042cd571142ccd1c742ca5e6542c7bff642c6e30742c54c1f42c2b5a742c1c9c142c2b5d542c2739142c2b17642c4cb6042c74eb542c887b842c9c5a842cc870542cdf32042cd83d142cd3e4d42cdcf3942cc2e8a42c99bbd42c7ef3942c724fc42c4820142c2835942c28b9942c286da42c1e66042c2e49742c5b2c042c77bc642c8425142ca84c042cd02b242cd873142cd512c42ce0c6f42cdb660",
        ),
    ];

    #[test]
    fn quantized_tiles_restore_exactly_as_cfitsio_does() {
        for &(method, scale, zero, tile, restored) in QUANTIZED {
            let mut image = image_cards(-32, &[64, 1], &[64, 1], "RICE_1");
            image.extend([
                card("ZQUANTIZ", &format!("'{method}'")),
                card("ZDITHER0", "5000"),
                card("ZBLANK", "-2147483648"),
            ]);
            let file = compressed_file(
                &image,
                &[compressed_column(vec![unhex(tile)])],
                &[
                    ("ZSCALE", "1D", vec![scale.to_be_bytes().to_vec()]),
                    ("ZZERO", "1D", vec![zero.to_be_bytes().to_vec()]),
                ],
            );
            let decoded = crate::FitsImage::from_bytes(&file).unwrap();
            let expected: Vec<f32> = unhex(restored)
                .chunks_exact(4)
                .map(|chunk| f32::from_be_bytes(chunk.try_into().unwrap()))
                .collect();
            let crate::Pixels::F32(values) = &decoded.pixels else {
                panic!("{method}: expected f32 samples");
            };
            assert_bits(values, &expected);
            assert!(values[3].is_nan(), "{method}");
        }
    }

    #[test]
    fn lossless_tiles_fill_in_where_quantizing_failed() {
        // Row 0 is quantized without dithering; row 1 kept its floats,
        // gzipped in one file and raw in the other.
        let quantized = gzip(
            &[10_i32, -20, 30]
                .iter()
                .flat_map(|value| value.to_be_bytes())
                .collect::<Vec<_>>(),
        );
        let floats = [1.25_f32, f32::NAN, -3.5];
        let float_bytes: Vec<u8> = floats
            .iter()
            .flat_map(|value| value.to_be_bytes())
            .collect();
        let scalars = [
            (
                "ZSCALE",
                "1D",
                vec![
                    0.5_f64.to_be_bytes().to_vec(),
                    1.0_f64.to_be_bytes().to_vec(),
                ],
            ),
            (
                "ZZERO",
                "1D",
                vec![
                    100.0_f64.to_be_bytes().to_vec(),
                    0.0_f64.to_be_bytes().to_vec(),
                ],
            ),
        ];
        let image = image_cards(-32, &[3, 2], &[3, 1], "GZIP_1");
        let gzipped = compressed_file(
            &image,
            &[
                compressed_column(vec![quantized.clone(), Vec::new()]),
                (
                    "GZIP_COMPRESSED_DATA",
                    'B',
                    1,
                    vec![Vec::new(), gzip(&float_bytes)],
                ),
            ],
            &scalars,
        );
        let raw = compressed_file(
            &image,
            &[
                compressed_column(vec![quantized, Vec::new()]),
                (
                    "UNCOMPRESSED_DATA",
                    'E',
                    4,
                    vec![Vec::new(), float_bytes.clone()],
                ),
            ],
            &scalars,
        );
        for file in [gzipped, raw] {
            let decoded = crate::FitsImage::from_bytes(&file).unwrap();
            assert_bits(
                &decoded.into_physical_f32(),
                &[105.0, 90.0, 115.0, 1.25, f32::NAN, -3.5],
            );
        }
    }

    #[test]
    fn cubes_keep_three_planes_and_skip_the_tiles_beyond() {
        let plane = |value: i16| {
            gzip(
                &[value; 4]
                    .iter()
                    .flat_map(|value| value.to_be_bytes())
                    .collect::<Vec<_>>(),
            )
        };
        // The fourth and fifth planes' tiles are not even valid; they are
        // never read.
        let tiles = vec![plane(1), plane(2), plane(3), vec![0xde, 0xad], Vec::new()];
        let file = compressed_file(
            &image_cards(16, &[2, 2, 5], &[2, 2, 1], "GZIP_1"),
            &[compressed_column(tiles)],
            &[],
        );
        let decoded = crate::FitsImage::from_bytes(&file).unwrap();
        assert_eq!((decoded.width, decoded.height, decoded.planes), (2, 2, 3));
        assert_eq!(
            decoded.into_physical_f32(),
            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        );
    }

    #[test]
    fn the_image_header_is_rebuilt_from_the_table() {
        let mut image = image_cards(16, &[2, 1], &[2, 1], "GZIP_1");
        image.extend([
            card("ZSIMPLE", "T"),
            card("ZEXTEND", "T"),
            card("ZQUANTIZ", "'NO_DITHER'"),
            card("ZBLANK", "-7"),
            card("OBJECT", "'M13'"),
            card("EXTNAME", "'COMPRESSED_IMAGE'"),
            card("ZHECKSUM", "'image sum'"),
            card("CHECKSUM", "'table sum'"),
            card("ZD", "12.5"),
        ]);
        let samples = gzip(
            &[-7_i16, 300]
                .iter()
                .flat_map(|value| value.to_be_bytes())
                .collect::<Vec<_>>(),
        );
        let file = compressed_file(&image, &[compressed_column(vec![samples])], &[]);
        let decoded = crate::FitsImage::from_bytes(&file).unwrap();
        let keywords: Vec<&str> = decoded
            .headers
            .iter()
            .map(|(key, _)| key.as_str())
            .collect();
        assert_eq!(
            keywords,
            [
                "SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "EXTEND", "OBJECT", "CHECKSUM",
                "ZD", "BLANK"
            ]
        );
        assert_eq!(decoded.header_str("CHECKSUM"), Some("image sum"));
        // ZBLANK is the integer image's BLANK.
        let physical = decoded.into_physical_f32();
        assert!(physical[0].is_nan());
        assert_eq!(physical[1], 300.0);

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("compressed.fits");
        std::fs::write(&path, &file).unwrap();
        assert_eq!(
            crate::read_header(&path).unwrap(),
            crate::FitsImage::open(&path).unwrap().headers
        );
    }

    #[test]
    fn unsupported_and_damaged_tables_are_errors() {
        let tile = gzip(&[0_u8; 4]);
        let file = |algorithm: &str, tiles: Vec<Vec<u8>>, rows: usize| {
            compressed_file(
                &image_cards(16, &[2, rows], &[2, 1], algorithm),
                &[compressed_column(tiles)],
                &[],
            )
        };
        let error =
            crate::FitsImage::from_bytes(&file("HCOMPRESS_1", vec![tile.clone()], 1)).unwrap_err();
        assert!(
            matches!(error, FitsError::Unsupported(ref what) if what == "HCOMPRESS_1 tile compression"),
            "{error}"
        );
        // Two rows of image but one tile.
        let error =
            crate::FitsImage::from_bytes(&file("GZIP_1", vec![tile.clone()], 2)).unwrap_err();
        assert!(matches!(error, FitsError::Malformed(_)), "{error}");
        // A tile that inflates to the wrong size.
        let error =
            crate::FitsImage::from_bytes(&file("GZIP_1", vec![gzip(&[0; 3])], 1)).unwrap_err();
        assert!(matches!(error, FitsError::Malformed(_)), "{error}");
        // A cut-off file.
        let whole = file("GZIP_1", vec![tile], 1);
        let end = 2 * 2880 + 8;
        let error = crate::FitsImage::from_bytes(&whole[..end]).unwrap_err();
        assert!(
            matches!(error, FitsError::Malformed(ref what) if what == "data runs past EOF"),
            "{error}"
        );
    }

    #[test]
    fn header_updates_leave_the_compression_cards_alone() {
        let samples = gzip(
            &[5_i16, 6]
                .iter()
                .flat_map(|value| value.to_be_bytes())
                .collect::<Vec<_>>(),
        );
        let file = compressed_file(
            &image_cards(16, &[2, 1], &[2, 1], "GZIP_1"),
            &[compressed_column(vec![samples])],
            &[],
        );
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("update.fits");
        std::fs::write(&path, &file).unwrap();
        crate::update_header_in_place(&path, "OBJECT", &HeaderValue::String("M13".into()), None)
            .unwrap();
        crate::update_header_in_place(&path, "ZD", &HeaderValue::Float(30.0), None).unwrap();
        for reserved in [
            "ZCMPTYPE", "ZTILE1", "TFORM1", "ZQUANTIZ", "ZBITPIX", "THEAP",
        ] {
            assert!(
                crate::update_header_in_place(&path, reserved, &HeaderValue::Integer(1), None)
                    .is_err(),
                "{reserved}"
            );
        }
        let decoded = crate::FitsImage::open(&path).unwrap();
        assert_eq!(decoded.header_str("OBJECT"), Some("M13"));
        assert_eq!(decoded.header_f64("ZD"), Some(30.0));
        assert_eq!(decoded.into_physical_f32(), [5.0, 6.0]);
    }
}
