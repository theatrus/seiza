//! Fast FITS image reading and linear `f32` writing for astrophotography.
//!
//! Scope: single-image FITS files as written by capture software
//! (N.I.N.A., SGP, ASIAIR, ...) — the primary HDU with a 2D image in
//! BITPIX 8/16/32/-32/-64. 16-bit data stays `u16` end to end (no float
//! inflation), statistics come from histograms rather than sorts, and the
//! midtone-transfer-function autostretch matches N.I.N.A.'s. The writer emits
//! primary-HDU mono or RGB float images with validated typed headers and
//! atomic on-disk publication.

mod bayer;
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

use fitsio_pure::hdu::{Hdu, HduInfo};
use fitsio_pure::image::ImageData;
use fitsio_pure::stream::FitsReader;
use std::io::Read;
use std::path::Path;

const BLOCK: usize = 2880;

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
#[derive(Debug, Clone)]
pub enum Pixels {
    U8(Vec<u8>),
    /// BITPIX 16 with BZERO applied (the unsigned camera convention)
    U16(Vec<u16>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

/// A decoded FITS image: primary-HDU pixels plus the parsed header cards.
#[derive(Debug, Clone)]
pub struct FitsImage {
    pub width: usize,
    pub height: usize,
    /// Color planes: 1 for mono/CFA, 3 for planar RGB (NAXIS3 = 3)
    pub planes: usize,
    pub pixels: Pixels,
    /// Header cards in file order (keyword, value)
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

impl FitsHeader {
    fn from_cards(cards: &[fitsio_pure::header::Card]) -> Self {
        let mut header = Self::default();
        for card in cards {
            let commentary = || card.comment.as_deref().unwrap_or("").trim().to_string();
            match (card.keyword_str(), &card.value) {
                ("HISTORY", _) => header.history.push(commentary()),
                ("COMMENT", _) => header.comments.push(commentary()),
                (keyword, Some(value)) => header
                    .cards
                    .push((keyword.to_string(), header::header_value(value))),
                _ => {}
            }
        }
        header
    }
}

pub(crate) fn header_error(error: fitsio_pure::Error) -> FitsError {
    match error {
        fitsio_pure::Error::UnexpectedEof => FitsError::Malformed("header runs past EOF".into()),
        fitsio_pure::Error::InvalidHeader("first HDU must be primary") => FitsError::NotFits,
        fitsio_pure::Error::Io(error) => FitsError::Io(error),
        other => FitsError::Malformed(other.to_string()),
    }
}

fn data_error(error: fitsio_pure::Error) -> FitsError {
    match error {
        fitsio_pure::Error::UnexpectedEof => FitsError::Malformed("data runs past EOF".into()),
        fitsio_pure::Error::Io(error) => FitsError::Io(error),
        other => FitsError::Malformed(other.to_string()),
    }
}

/// Read the primary header and leave `stream` at the first byte of the
/// primary data unit. Only the header blocks are read.
fn read_primary_hdu<R: Read>(stream: &mut FitsReader<R>) -> Result<Hdu, FitsError> {
    stream
        .next_hdu()
        .map_err(header_error)?
        .cloned()
        .ok_or(FitsError::NotFits)
}

fn read_headers_from(reader: impl Read) -> Result<FitsHeader, FitsError> {
    let hdu = read_primary_hdu(&mut FitsReader::new(reader))?;
    Ok(FitsHeader::from_cards(&hdu.cards))
}

/// Read only the header cards of a FITS file, without touching the pixel
/// data — cheap metadata probes on large files.
pub fn read_header(path: &Path) -> Result<Vec<(String, HeaderValue)>, FitsError> {
    read_header_with_commentary(path).map(|header| header.cards)
}

/// Read a FITS file's header cards together with its `HISTORY` and
/// `COMMENT` text, without touching the pixel data.
pub fn read_header_with_commentary(path: &Path) -> Result<FitsHeader, FitsError> {
    read_headers_from(std::fs::File::open(path)?)
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
}

impl ImageSpec {
    fn from_hdu(info: &HduInfo, headers: &[(String, HeaderValue)]) -> Result<Self, FitsError> {
        let header_f64 = |key: &str| -> Option<f64> {
            headers
                .iter()
                .find(|(k, _)| k == key)
                .and_then(|(_, v)| v.as_f64())
        };

        let HduInfo::Primary { bitpix, naxes } = info else {
            return Err(FitsError::Unsupported("random groups".into()));
        };
        let bitpix = *bitpix;
        if !matches!(bitpix, 8 | 16 | 32 | -32 | -64) {
            return Err(FitsError::Unsupported(format!("BITPIX {bitpix}")));
        }
        let &[width, height, ref rest @ ..] = naxes.as_slice() else {
            return Err(FitsError::Unsupported(format!(
                "NAXIS {} (need a 2D image)",
                naxes.len()
            )));
        };
        // Planar color cubes (Siril and friends write RGB as NAXIS3 = 3);
        // planes beyond the third are ignored.
        let planes = rest.first().copied().unwrap_or(1).clamp(1, 3);
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
        })
    }
}

/// Decode the primary data unit in fitsio-pure's bounded chunks. Planes past
/// the third are decoded and then dropped.
fn decode_pixels<R: Read>(
    stream: &mut FitsReader<R>,
    hdu: &Hdu,
    spec: ImageSpec,
) -> Result<Pixels, FitsError> {
    // The near-universal camera convention: unsigned data stored as i16 with
    // BZERO 32768. Fold BZERO in while staying u16.
    let offset = spec.bzero as i64;
    let unsigned_u16 = spec.bscale == 1.0 && (offset == 32768 || offset == 0);
    if spec.bitpix == 16 && !unsigned_u16 {
        let mut out = vec![0.0f32; hdu.data_len / 2];
        stream.read_image_into_f32(&mut out).map_err(data_error)?;
        out.truncate(spec.count);
        for value in &mut out {
            *value = (spec.bzero + spec.bscale * *value as f64) as f32;
        }
        return Ok(Pixels::F32(out));
    }
    let take = |data: ImageData| match data {
        ImageData::U8(mut out) => {
            out.truncate(spec.count);
            Pixels::U8(out)
        }
        // Adding 32768 to an i16 is a sign-bit flip on the raw bits. With no
        // offset, negatives clamp to zero.
        ImageData::I16(mut out) => {
            out.truncate(spec.count);
            if offset == 32768 {
                out.iter_mut().for_each(|value| *value ^= i16::MIN);
            } else {
                out.iter_mut().for_each(|value| *value = (*value).max(0));
            }
            Pixels::U16(out.into_iter().map(|value| value as u16).collect())
        }
        ImageData::I32(mut out) => {
            out.truncate(spec.count);
            Pixels::I32(out)
        }
        ImageData::F32(mut out) => {
            out.truncate(spec.count);
            Pixels::F32(out)
        }
        ImageData::F64(mut out) => {
            out.truncate(spec.count);
            Pixels::F64(out)
        }
        ImageData::I64(_) => unreachable!("ImageSpec validates BITPIX"),
    };
    stream.read_image().map(take).map_err(data_error)
}

impl FitsImage {
    /// Open and decode the primary image while retaining only the parsed
    /// header, the final typed pixel vector, and a fixed-size conversion
    /// buffer. FITS data-unit padding and trailing HDUs are not read.
    pub fn open(path: &Path) -> Result<FitsImage, FitsError> {
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len();
        Self::read_from(file, Some(len))
    }

    /// Decode an in-memory FITS image through the same bounded conversion
    /// pipeline used by [`Self::open`]. The caller retains ownership of the
    /// input slice, so this entry point does not reduce its memory footprint.
    pub fn from_bytes(data: &[u8]) -> Result<FitsImage, FitsError> {
        Self::read_from(data, Some(data.len() as u64))
    }

    fn read_from(reader: impl Read, available_bytes: Option<u64>) -> Result<FitsImage, FitsError> {
        if available_bytes.is_some_and(|len| len < BLOCK as u64) {
            return Err(FitsError::NotFits);
        }
        let mut stream = FitsReader::new(reader);
        let hdu = read_primary_hdu(&mut stream)?;
        let headers = FitsHeader::from_cards(&hdu.cards).cards;
        let spec = ImageSpec::from_hdu(&hdu.info, &headers)?;
        // Dimension metadata is checked against a known input length before
        // the declared pixel vector is allocated.
        if let Some(available_bytes) = available_bytes {
            let data_end = (hdu.data_start as u64)
                .checked_add(hdu.data_len as u64)
                .ok_or_else(|| FitsError::Malformed("implausible dimensions".into()))?;
            if data_end > available_bytes {
                return Err(FitsError::Malformed("data runs past EOF".into()));
            }
        }
        let pixels = decode_pixels(&mut stream, &hdu, spec)?;
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
    /// (`BZERO + BSCALE * stored`) as f32, still in plane order.
    ///
    /// Two cases skip the header scaling because decoding already applied
    /// it: U16 pixels carry the standard unsigned-camera BZERO, and a
    /// BITPIX=16 image with unusual scaling decodes straight to F32.
    pub fn into_physical_f32(self) -> Vec<f32> {
        let bitpix = self
            .header("BITPIX")
            .and_then(HeaderValue::as_i64)
            .unwrap_or(0);
        let bzero = self.header_f64("BZERO").unwrap_or(0.0);
        let bscale = self.header_f64("BSCALE").unwrap_or(1.0);
        match self.pixels {
            Pixels::U8(values) => values
                .into_iter()
                .map(|value| (bzero + bscale * f64::from(value)) as f32)
                .collect(),
            Pixels::U16(values) => values.into_iter().map(f32::from).collect(),
            Pixels::I32(values) => values
                .into_iter()
                .map(|value| (bzero + bscale * f64::from(value)) as f32)
                .collect(),
            Pixels::F32(values) if bitpix == 16 => values,
            Pixels::F32(values) => values
                .into_iter()
                .map(|value| (bzero + bscale * f64::from(value)) as f32)
                .collect(),
            Pixels::F64(values) => values
                .into_iter()
                .map(|value| (bzero + bscale * value) as f32)
                .collect(),
        }
    }

    /// Pixels as u16, converting float/i32 data by min-max scaling.
    /// Planar RGB collapses to luminance; the mono u16 case is a borrow —
    /// no copy, no conversion.
    pub fn to_u16(&self) -> std::borrow::Cow<'_, [u16]> {
        if self.planes == 3 {
            let full = self.planes_u16();
            let n = self.width * self.height;
            return std::borrow::Cow::Owned(
                (0..n)
                    .map(|i| {
                        ((full[i] as u32 + full[n + i] as u32 + full[2 * n + i] as u32) / 3) as u16
                    })
                    .collect(),
            );
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
        let mut data = vec![0u16; n * 3];
        for i in 0..n {
            data[i * 3] = full[i];
            data[i * 3 + 1] = full[n + i];
            data[i * 3 + 2] = full[2 * n + i];
        }
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
            Pixels::I32(data) => scale_to_u16(data.iter().map(|&v| v as f64)),
            Pixels::F32(data) => scale_to_u16(data.iter().map(|&v| v as f64)),
            Pixels::F64(data) => scale_to_u16(data.iter().copied()),
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
            Pixels::I32(data) => scale_to_f32(data.iter().map(|&value| value as f64)),
            Pixels::F32(data) => scale_to_f32(data.iter().map(|&value| value as f64)),
            Pixels::F64(data) => scale_to_f32(data.iter().copied()),
        }
    }
}

fn scale_to_u16(values: impl Iterator<Item = f64> + Clone) -> std::borrow::Cow<'static, [u16]> {
    let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
    for v in values.clone() {
        if v.is_finite() {
            min = min.min(v);
            max = max.max(v);
        }
    }
    let span = (max - min).max(1e-12);
    std::borrow::Cow::Owned(
        values
            .map(|v| (((v - min) / span).clamp(0.0, 1.0) * 65535.0) as u16)
            .collect(),
    )
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
mod io_tests {
    use super::*;

    const CARD: usize = 80;
    /// fitsio-pure decodes the data unit in chunks of this size.
    const CHUNK_BYTES: usize = 1024 * 1024;

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
        let count = CHUNK_BYTES / 2 + 7;
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
            read_headers_from(short_header.as_slice()),
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
        let header = read_headers_from(&mut reader).unwrap();
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
        let header = read_headers_from(header.as_slice()).unwrap();
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
