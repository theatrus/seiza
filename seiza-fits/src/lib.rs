//! Fast, dependency-free FITS image reading for astrophotography.
//!
//! Scope: single-image FITS files as written by capture software
//! (N.I.N.A., SGP, ASIAIR, ...) — the primary HDU with a 2D image in
//! BITPIX 8/16/32/-32/-64. 16-bit data stays `u16` end to end (no float
//! inflation), statistics come from histograms rather than sorts, and the
//! midtone-transfer-function autostretch matches N.I.N.A.'s.

mod bayer;
mod header;
mod stats;
mod stretch;

pub use bayer::{BayerPattern, RgbImage16, debayer_rgb16};
pub use header::{HeaderValue, parse_header_value};
pub use stats::Statistics;
pub use stretch::{StretchParams, midtones_transfer_function};

use std::io::Read;
use std::path::Path;

const BLOCK: usize = 2880;
const CARD: usize = 80;

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
    pub pixels: Pixels,
    /// Header cards in file order (keyword, value)
    pub headers: Vec<(String, HeaderValue)>,
}

impl FitsImage {
    pub fn open(path: &Path) -> Result<FitsImage, FitsError> {
        let mut file = std::fs::File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Self::from_bytes(&data)
    }

    pub fn from_bytes(data: &[u8]) -> Result<FitsImage, FitsError> {
        if data.len() < BLOCK || &data[0..6] != b"SIMPLE" {
            return Err(FitsError::NotFits);
        }

        // Parse header cards block by block until END
        let mut headers = Vec::new();
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
                if keyword.is_empty() || keyword == "COMMENT" || keyword == "HISTORY" {
                    continue;
                }
                if card[8] == b'=' {
                    let raw = String::from_utf8_lossy(&card[10..]);
                    headers.push((keyword, parse_header_value(&raw)));
                }
            }
        }
        let data_start =
            data_start.ok_or_else(|| FitsError::Malformed("missing END card".into()))?;

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
        let naxis = header_i64("NAXIS").unwrap_or(0);
        if naxis < 2 {
            return Err(FitsError::Unsupported(format!(
                "NAXIS {naxis} (need a 2D image)"
            )));
        }
        let width = header_i64("NAXIS1")
            .ok_or_else(|| FitsError::Malformed("missing NAXIS1".into()))?
            as usize;
        let height = header_i64("NAXIS2")
            .ok_or_else(|| FitsError::Malformed("missing NAXIS2".into()))?
            as usize;
        // NAXIS3 color cubes: take the first plane
        let count = width
            .checked_mul(height)
            .ok_or_else(|| FitsError::Malformed("implausible dimensions".into()))?;
        if count == 0 || count > 2_000_000_000 {
            return Err(FitsError::Malformed("implausible dimensions".into()));
        }

        let bzero = header_f64("BZERO").unwrap_or(0.0);
        let bscale = header_f64("BSCALE").unwrap_or(1.0);

        let bytes_per_px = (bitpix.unsigned_abs() / 8) as usize;
        let needed = count * bytes_per_px;
        let raw = data
            .get(data_start..data_start + needed)
            .ok_or_else(|| FitsError::Malformed("data runs past EOF".into()))?;

        let pixels = match bitpix {
            8 => Pixels::U8(raw.to_vec()),
            16 => {
                // The near-universal camera convention: unsigned data stored
                // as i16 with BZERO 32768. Fold BZERO in while staying u16.
                let offset = bzero as i64;
                if bscale == 1.0 && (offset == 32768 || offset == 0) {
                    let shift = offset as i32;
                    let mut out = Vec::with_capacity(count);
                    for chunk in raw.chunks_exact(2) {
                        let v = i16::from_be_bytes([chunk[0], chunk[1]]) as i32 + shift;
                        out.push(v.clamp(0, 65535) as u16);
                    }
                    Pixels::U16(out)
                } else {
                    let mut out = Vec::with_capacity(count);
                    for chunk in raw.chunks_exact(2) {
                        let v = i16::from_be_bytes([chunk[0], chunk[1]]) as f64;
                        out.push((bzero + bscale * v) as f32);
                    }
                    Pixels::F32(out)
                }
            }
            32 => {
                let mut out = Vec::with_capacity(count);
                for chunk in raw.chunks_exact(4) {
                    out.push(i32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Pixels::I32(out)
            }
            -32 => {
                let mut out = Vec::with_capacity(count);
                for chunk in raw.chunks_exact(4) {
                    out.push(f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Pixels::F32(out)
            }
            -64 => {
                let mut out = Vec::with_capacity(count);
                for chunk in raw.chunks_exact(8) {
                    out.push(f64::from_be_bytes(chunk.try_into().unwrap()));
                }
                Pixels::F64(out)
            }
            other => {
                return Err(FitsError::Unsupported(format!("BITPIX {other}")));
            }
        };

        Ok(FitsImage {
            width,
            height,
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

    /// Pixels as u16, converting float/i32 data by min-max scaling.
    /// The u16 case is a borrow — no copy, no conversion.
    pub fn to_u16(&self) -> std::borrow::Cow<'_, [u16]> {
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
        stats::statistics_u16(&self.to_u16())
    }

    /// N.I.N.A.-compatible MTF autostretch straight to 8-bit grayscale.
    /// Raw one-shot-color mosaics are debayered to luminance first.
    pub fn stretch_to_u8(&self, params: &StretchParams) -> Vec<u8> {
        let data = match self.debayer() {
            Some(rgb) => std::borrow::Cow::Owned(rgb.to_luma_u16()),
            None => self.to_u16(),
        };
        let stats = stats::statistics_u16(&data);
        stretch::stretch_u16_to_u8(&data, &stats, params)
    }

    /// The color filter array layout, when the `BAYERPAT` header marks
    /// this as a raw one-shot-color mosaic.
    pub fn bayer_pattern(&self) -> Option<BayerPattern> {
        BayerPattern::parse(self.header_str("BAYERPAT")?)
    }

    /// Debayer a raw one-shot-color mosaic to interleaved RGB, honoring
    /// `XBAYROFF`/`YBAYROFF` origin offsets. `None` for mono images.
    pub fn debayer(&self) -> Option<RgbImage16> {
        let pattern = self.bayer_pattern()?;
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
