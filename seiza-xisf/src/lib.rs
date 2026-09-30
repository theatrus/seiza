//! Practical XISF 1.0 image reading and writing for Seiza.
//!
//! The reader meets the baseline decoder rules of XISF 1.0, Revision 1: it
//! reads monolithic files with attached, inline, or embedded pixel blocks,
//! planar or normal storage, and every standard codec, subblocks included.
//! An image it cannot decode is unavailable on its own and leaves the rest of
//! the file readable; see [`XisfFileInfo::unavailable`]. Decoded
//! images use [`seiza_fits::FitsImage`] so downstream statistics, stretching,
//! Bayer handling, stacking, and solving do not depend on the source format.
//! [`write_f32_image`] writes the same layout back out with `Float32` samples,
//! mirroring the `seiza_fits` writer API.
//!
//! Sample values pass through unchanged: decoding never applies the XISF
//! `bounds` attribute, keeping linear data linear, and preserved FITS scaling
//! keywords (`BZERO`/`BSCALE`) are dropped because XISF samples are already
//! physical.
//!
//! Bounds still matter, because PixInsight writes floating-point images
//! normalized to `0:1` and nothing in the samples says so. [`read_image`]
//! returns the declared range alongside the pixels, and
//! [`XisfImage::rescale_normalized_to`] converts such a frame onto a chosen
//! full scale. It acts only on a declared `0:1`, the one spelling whose
//! meaning is settled — writers disagree about the rest, this crate's own
//! writer among them — and [`XisfImage::rescale_from`] takes the source range
//! from a caller who knows better than the file. All of it is opt-in;
//! [`open`] still hands back exactly what is stored.

use flate2::read::ZlibDecoder;
use quick_xml::events::{BytesStart, Event};
use quick_xml::{Reader, XmlVersion};
use seiza_fits::{FitsImage, HeaderValue, Pixels, parse_header_value};
use sha1::Sha1;
use sha2::{Sha256, Sha512};
use sha3::{Sha3_256, Sha3_512};
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

mod writer;

pub use writer::{
    ChecksumAlgorithm, WriteCompression, WriteOptions, write_f32_image, write_f32_image_to,
    write_f32_image_to_with_options, write_f32_image_with_options,
};

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
    /// Decoded to `f64`, which holds integers exactly only up to 2^53.
    UInt64,
    Float32,
    Float64,
    /// Two `Float32` values, real then imaginary. Only
    /// [`read_complex_image`] decodes complex samples.
    Complex32,
    /// Two `Float64` values, real then imaginary.
    Complex64,
}

impl SampleFormat {
    fn parse(value: &str) -> Result<Self, XisfError> {
        match value {
            "UInt8" | "Byte" => Ok(Self::UInt8),
            "UInt16" | "UShort" => Ok(Self::UInt16),
            "UInt32" | "UInt" => Ok(Self::UInt32),
            "UInt64" => Ok(Self::UInt64),
            "Complex32" => Ok(Self::Complex32),
            "Complex64" => Ok(Self::Complex64),
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
            Self::UInt64 | Self::Float64 | Self::Complex32 => 8,
            Self::Complex64 => 16,
        }
    }

    pub fn is_complex(self) -> bool {
        matches!(self, Self::Complex32 | Self::Complex64)
    }

    fn fits_bitpix(self) -> i64 {
        match self {
            Self::UInt8 => 8,
            Self::UInt16 => 16,
            Self::UInt32 => 32,
            Self::UInt64 => 64,
            Self::Float32 | Self::Complex32 => -32,
            Self::Float64 | Self::Complex64 => -64,
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
    /// `(compressed, uncompressed)` byte lengths of each independently
    /// compressed subblock, in storage order. Empty when the block was
    /// compressed whole.
    pub subblocks: Vec<(u64, u64)>,
}

/// How the pixel samples of an image are ordered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PixelStorage {
    /// Channel by channel. Decoded images always use this order.
    #[default]
    Planar,
    /// Pixel by pixel, with the samples of each pixel together.
    Normal,
}

/// Where an image's pixel data block is stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlockLocation {
    /// Attached to a monolithic file at a byte offset.
    Attachment { offset: u64, bytes: u64 },
    /// Base64 or hex text inside the `Image` element; `bytes` is the decoded
    /// length.
    Inline { bytes: u64 },
    /// Base64 or hex text inside a child `Data` element; `bytes` is the
    /// decoded length.
    Embedded { bytes: u64 },
    /// A local file named by a distributed unit's header: all of it, or one
    /// block of an XISF data blocks (`.xisb`) file.
    External {
        path: PathBuf,
        offset: u64,
        bytes: u64,
    },
}

impl BlockLocation {
    /// The stored length of the block in bytes, after any text decoding and
    /// before decompression.
    pub fn bytes(&self) -> u64 {
        match *self {
            Self::Attachment { bytes, .. }
            | Self::Inline { bytes }
            | Self::Embedded { bytes }
            | Self::External { bytes, .. } => bytes,
        }
    }
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
    pub pixel_storage: PixelStorage,
    pub location: BlockLocation,
    pub compression: Option<CompressionInfo>,
    /// The declared `bounds` attribute, as `(low, high)` with `low < high`.
    ///
    /// PixInsight writes `bounds="0:1"` on floating-point images, so a frame
    /// that looks like it holds physical ADU may in fact be normalized.
    /// Samples decode exactly as stored, so this attribute is the only hint
    /// that they are.
    ///
    /// Read it as a hint, not a fact. Writers disagree about what the range
    /// means — this crate's own [`write_f32_image`] reports the observed
    /// sample minimum and maximum, not a nominal `0:1` — so `Some((0.0,
    /// 30000.0))` may describe a normalization range or may just describe the
    /// data. Only `Some((0.0, 1.0))` carries the settled meaning that
    /// [`XisfImage::rescale_normalized_to`] acts on.
    ///
    /// `None` means the file declared nothing usable: no attribute, or one
    /// this crate could not read as a `low:high` pair spanning a real range.
    /// The XISF default of `0:1` for floating-point formats is deliberately
    /// not filled in, because a writer that omits the attribute has told us
    /// nothing about its intent.
    pub bounds: Option<(f64, f64)>,
    pub headers: Vec<(String, HeaderValue)>,
    pub properties: Vec<XisfProperty>,
    pub cfa_pattern: Option<String>,
    /// The image's `RGBWorkingSpace`, when it declares a usable one. XISF
    /// takes sRGB when none is declared.
    pub rgb_working_space: Option<RgbWorkingSpace>,
}

/// How an RGB working space turns nominal components into linear ones.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gamma {
    /// The sRGB piecewise function.
    Srgb,
    /// A power law with this exponent.
    Exponent(f64),
}

/// A colorimetrically defined RGB working space, relative to the D50
/// reference white.
#[derive(Clone, Debug, PartialEq)]
pub struct RgbWorkingSpace {
    pub name: Option<String>,
    pub gamma: Gamma,
    /// Chromaticity x of the red, green and blue primaries.
    pub x: [f64; 3],
    /// Chromaticity y of the red, green and blue primaries.
    pub y: [f64; 3],
    /// Luminance coefficients of the red, green and blue primaries.
    pub luminance: [f64; 3],
}

impl RgbWorkingSpace {
    /// sRGB adapted to D50 with the Bradford transform, the default XISF
    /// working space.
    pub fn srgb() -> Self {
        Self {
            name: Some("sRGB IEC61966-2.1".into()),
            gamma: Gamma::Srgb,
            x: [0.648431, 0.321152, 0.155886],
            y: [0.330856, 0.597871, 0.066044],
            luminance: [0.222491, 0.716888, 0.060621],
        }
    }

    fn parse(attributes: &BTreeMap<String, String>) -> Result<Self, XisfError> {
        let triple = |name: &str| -> Result<[f64; 3], XisfError> {
            let value = required(attributes, name, "RGBWorkingSpace")?;
            let parts = value
                .split(':')
                .map(|part| part.trim().parse::<f64>().ok().filter(|v| v.is_finite()))
                .collect::<Option<Vec<_>>>()
                .filter(|parts| parts.len() == 3)
                .ok_or_else(|| {
                    XisfError::Malformed(format!("invalid RGBWorkingSpace {name}={value:?}"))
                })?;
            Ok([parts[0], parts[1], parts[2]])
        };
        let gamma = required(attributes, "gamma", "RGBWorkingSpace")?.trim();
        let gamma = if gamma.eq_ignore_ascii_case("srgb") {
            Gamma::Srgb
        } else {
            gamma
                .parse::<f64>()
                .ok()
                .filter(|gamma| gamma.is_finite() && *gamma > 0.0)
                .map(Gamma::Exponent)
                .ok_or_else(|| {
                    XisfError::Malformed(format!("invalid RGBWorkingSpace gamma {gamma:?}"))
                })?
        };
        let space = Self {
            name: attributes.get("name").cloned(),
            gamma,
            x: triple("x")?,
            y: triple("y")?,
            luminance: triple("Y")?,
        };
        if space.y.contains(&0.0) || space.xyz_from_linear_rgb().inverse().is_none() {
            return Err(XisfError::Malformed(
                "RGBWorkingSpace primaries do not define a valid space".into(),
            ));
        }
        Ok(space)
    }

    /// The matrix M of Annex B, taking linear RGB to CIE XYZ.
    fn xyz_from_linear_rgb(&self) -> Matrix3 {
        let column = |channel: usize| {
            let (x, y, luminance) = (self.x[channel], self.y[channel], self.luminance[channel]);
            [luminance * x / y, luminance, luminance * (1.0 - x - y) / y]
        };
        let (red, green, blue) = (column(0), column(1), column(2));
        Matrix3(std::array::from_fn(|row| [red[row], green[row], blue[row]]))
    }

    fn delinearize(&self, value: f64) -> f64 {
        match self.gamma {
            Gamma::Srgb if value <= 0.003_130_8 => 12.92 * value,
            Gamma::Srgb => 1.055 * value.powf(1.0 / 2.4) - 0.055,
            Gamma::Exponent(gamma) => value.powf(1.0 / gamma),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Matrix3([[f64; 3]; 3]);

impl Matrix3 {
    fn inverse(&self) -> Option<Self> {
        let m = &self.0;
        let cofactor = |row: usize, column: usize| {
            let (r0, r1) = ((row + 1) % 3, (row + 2) % 3);
            let (c0, c1) = ((column + 1) % 3, (column + 2) % 3);
            m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
        };
        let determinant = (0..3)
            .map(|column| m[0][column] * cofactor(0, column))
            .sum::<f64>();
        if !determinant.is_normal() {
            return None;
        }
        // The inverse is the transposed cofactor matrix over the determinant.
        Some(Self(std::array::from_fn(|row| {
            std::array::from_fn(|column| cofactor(column, row) / determinant)
        })))
    }

    fn apply(&self, vector: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|row| {
            (0..3)
                .map(|column| self.0[row][column] * vector[column])
                .sum()
        })
    }
}

/// Convert decoded CIE L*a*b* samples to RGB in place, with the equations of
/// Annex B and the image's working space. Samples are mapped to the nominal
/// `[0, 1]` range through the image's representable range and back, so the
/// RGB samples keep the stored sample format and range.
fn rgb_from_lab(pixels: &mut Pixels, info: &XisfImageInfo) -> Result<(), XisfError> {
    let (low, high) = match (info.bounds, info.sample_format) {
        (Some(bounds), _) => bounds,
        (None, SampleFormat::Float32 | SampleFormat::Float64) => {
            return Err(XisfError::Malformed(
                "floating-point CIELab image has no usable bounds".into(),
            ));
        }
        (None, format) => (0.0, 2_f64.powi(8 * format.bytes_per_sample() as i32) - 1.0),
    };
    let space = info
        .rgb_working_space
        .clone()
        .unwrap_or_else(RgbWorkingSpace::srgb);
    let linear_from_xyz = space
        .xyz_from_linear_rgb()
        .inverse()
        .expect("working spaces are validated when parsed");
    const WHITE_X: f64 = 0.96422;
    const WHITE_Z: f64 = 0.82521;
    const EPSILON: f64 = 216.0 / 24389.0;
    const KAPPA: f64 = 24389.0 / 27.0;
    let g = |t: f64| {
        let cube = t * t * t;
        if cube > EPSILON {
            cube
        } else {
            (116.0 * t - 16.0) / KAPPA
        }
    };
    let convert = |[l, a, b]: [f64; 3]| -> [f64; 3] {
        let f_y = (l + 0.16) / 1.16;
        let f_x = f_y + 50.0 / 29.0 * (a - 0.5);
        let f_z = f_y - 50.0 / 29.0 * (b - 0.5);
        let xyz = [WHITE_X * g(f_x), g(f_y), WHITE_Z * g(f_z)];
        linear_from_xyz
            .apply(xyz)
            .map(|linear| space.delinearize(linear.clamp(0.0, 1.0)).clamp(0.0, 1.0))
    };
    let pixel_count = info.width * info.height;
    let scale = high - low;
    fn convert_planes<T: Copy>(
        values: &mut [T],
        pixel_count: usize,
        to_f64: impl Fn(T) -> f64,
        from_f64: impl Fn(f64) -> T,
        convert: impl Fn([f64; 3]) -> [f64; 3],
    ) {
        for index in 0..pixel_count {
            let lab = std::array::from_fn(|plane| to_f64(values[plane * pixel_count + index]));
            let rgb = convert(lab);
            for (plane, value) in rgb.into_iter().enumerate() {
                values[plane * pixel_count + index] = from_f64(value);
            }
        }
    }
    let nominal = |value: f64| ((value - low) / scale).clamp(0.0, 1.0);
    let stored = |value: f64| low + value * scale;
    match pixels {
        Pixels::U8(values) => convert_planes(
            values,
            pixel_count,
            |value| nominal(f64::from(value)),
            |value| stored(value).round() as u8,
            convert,
        ),
        Pixels::U16(values) => convert_planes(
            values,
            pixel_count,
            |value| nominal(f64::from(value)),
            |value| stored(value).round() as u16,
            convert,
        ),
        Pixels::I32(values) => convert_planes(
            values,
            pixel_count,
            |value| nominal(f64::from(value)),
            |value| stored(value).round() as i32,
            convert,
        ),
        Pixels::F32(values) => convert_planes(
            values,
            pixel_count,
            |value| nominal(f64::from(value)),
            |value| stored(value) as f32,
            convert,
        ),
        Pixels::F64(values) => {
            let integer = !matches!(info.sample_format, SampleFormat::Float64);
            convert_planes(
                values,
                pixel_count,
                nominal,
                |value| {
                    if integer {
                        stored(value).round()
                    } else {
                        stored(value)
                    }
                },
                convert,
            )
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct XisfFileInfo {
    /// The images this crate can decode. Their `index` values count every
    /// `Image` element in the file, so they skip the unavailable ones.
    pub images: Vec<XisfImageInfo>,
    /// Images this crate cannot decode, such as CIELab or three-dimensional
    /// images. The XISF conformance rules make such an image unavailable
    /// without making the rest of the file unreadable.
    pub unavailable: Vec<UnavailableImage>,
}

/// An `Image` element that this crate cannot decode.
#[derive(Clone, Debug)]
pub struct UnavailableImage {
    /// Zero-based position among all `Image` elements in the file.
    pub index: usize,
    pub id: Option<String>,
    pub reason: String,
}

/// A decoded image together with the XISF metadata that describes it.
///
/// [`open`] answers with pixels alone, which is all most callers want. Use
/// this when the metadata decides how to read the pixels — above all
/// [`XisfImageInfo::bounds`], which hints whether the samples are normalized.
///
/// The two header lists differ on purpose. `info.headers` holds exactly the
/// keywords the file preserved, matching what [`inspect`] reports.
/// `image.headers` is that list plus the structural cards this crate
/// synthesizes from the image geometry — `BITPIX`, `NAXIS`, `NAXIS1`,
/// `NAXIS2`, `NAXIS3`, `IMAGETYP` — matching what [`read_header`] and
/// [`open`] return. Read geometry from `image`, not from `info`.
#[derive(Clone, Debug)]
pub struct XisfImage {
    pub image: FitsImage,
    pub info: XisfImageInfo,
}

impl XisfImage {
    /// Put a frame the file declares as normalized onto `0..=full_scale`.
    ///
    /// A PixInsight frame arrives normalized to `0:1`, which is not
    /// comparable with a camera frame's ADU. `rescale_normalized_to(65535.0)`
    /// puts it on a 16-bit scale so background and flux can be compared
    /// across the two.
    ///
    /// This acts only on a declared range of exactly `0:1`, the one spelling
    /// whose meaning is settled. Any other range is ambiguous — writers
    /// disagree, and this crate's own [`write_f32_image`] reports the observed
    /// sample minimum and maximum — so converting from it would as easily
    /// stretch an already-physical frame as normalize a normalized one. When a
    /// caller knows better than the file, [`rescale_from`] takes the source
    /// range directly.
    ///
    /// [`rescale_from`]: XisfImage::rescale_from
    ///
    /// Answers whether the conversion ran. It does not run, and the samples
    /// are untouched, when the declared range is absent or is anything but
    /// `0:1`, when `full_scale` is not finite and positive, or when the
    /// samples are integers — XISF integer formats already span their type's
    /// range, so rescaling them would destroy data rather than place it.
    /// A second call answers `false`, because the first rewrote `bounds`.
    pub fn rescale_normalized_to(&mut self, full_scale: f32) -> bool {
        if self.info.bounds != Some((0.0, 1.0)) {
            return false;
        }
        self.rescale_from((0.0, 1.0), full_scale)
    }

    /// Map floating-point samples from `source` onto `0..=full_scale`, taking
    /// the caller's word for the source range.
    ///
    /// The escape hatch from [`rescale_normalized_to`], for a caller who
    /// knows what a file's samples mean when the file itself does not say so
    /// usefully. Nothing is inferred from [`XisfImageInfo::bounds`], though
    /// it is rewritten to the new range on success so the metadata stays
    /// truthful.
    ///
    /// [`rescale_normalized_to`]: XisfImage::rescale_normalized_to
    ///
    /// The map is linear and does not clamp: a sample outside `source` lands
    /// outside `0..=full_scale`, which keeps unclipped highlights and
    /// negative background residuals intact rather than flattening them.
    ///
    /// Answers whether the conversion ran. It does not run when `source` is
    /// not a finite range with `low < high`, when `full_scale` is not finite
    /// and positive, or when the samples are integers.
    pub fn rescale_from(&mut self, source: (f64, f64), full_scale: f32) -> bool {
        let (low, high) = source;
        if !low.is_finite() || !high.is_finite() || high <= low {
            return false;
        }
        if !full_scale.is_finite() || full_scale <= 0.0 {
            return false;
        }
        let scale = f64::from(full_scale) / (high - low);
        match &mut self.image.pixels {
            Pixels::F32(samples) => {
                for sample in samples.iter_mut() {
                    *sample = ((f64::from(*sample) - low) * scale) as f32;
                }
            }
            Pixels::F64(samples) => {
                for sample in samples.iter_mut() {
                    *sample = (*sample - low) * scale;
                }
            }
            // Integer samples already span their format's range.
            Pixels::U8(_) | Pixels::U16(_) | Pixels::I32(_) => return false,
        }
        self.info.bounds = Some((0.0, f64::from(full_scale)));
        true
    }
}

/// Whether a path uses the conventional `.xisf` extension.
pub fn is_xisf_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("xisf") || extension.eq_ignore_ascii_case("xish")
        })
}

#[derive(Clone, Debug)]
struct ParsedImage {
    info: XisfImageInfo,
    checksum: Option<String>,
    expected_bytes: usize,
    /// The decoded contents of an inline or embedded block.
    inline_data: Option<Vec<u8>>,
}

#[derive(Debug)]
struct ParsedFile {
    images: Vec<ParsedImage>,
    unavailable: Vec<(usize, Option<String>, XisfError)>,
}

impl ParsedFile {
    /// Remove and return the image at a zero-based `Image` element index, or
    /// the reason it cannot be decoded.
    fn take(&mut self, index: usize) -> Result<ParsedImage, XisfError> {
        if let Some(position) = self
            .images
            .iter()
            .position(|image| image.info.index == index)
        {
            return Ok(self.images.swap_remove(position));
        }
        match self
            .unavailable
            .iter()
            .position(|(other, ..)| *other == index)
        {
            Some(position) => Err(self.unavailable.swap_remove(position).2),
            None => Err(XisfError::ImageNotFound(format!(
                "at zero-based index {index}"
            ))),
        }
    }

    fn index_of_id(&self, id: &str) -> Option<usize> {
        let available = self
            .images
            .iter()
            .map(|image| (image.info.index, image.info.id.as_deref()));
        let unavailable = self
            .unavailable
            .iter()
            .map(|(index, other, _)| (*index, other.as_deref()));
        available
            .chain(unavailable)
            .filter(|(_, other)| *other == Some(id))
            .map(|(index, _)| index)
            .min()
    }
}

/// Read the XISF header and describe every top-level image without loading
/// pixel attachments.
pub fn inspect(path: &Path) -> Result<XisfFileInfo, XisfError> {
    let mut file = std::fs::File::open(path)?;
    let file_bytes = file.metadata()?.len();
    let parsed = parse_file(&mut file, file_bytes, Some(&header_dir(path)))?;
    Ok(XisfFileInfo {
        images: parsed.images.into_iter().map(|image| image.info).collect(),
        unavailable: parsed
            .unavailable
            .into_iter()
            .map(|(index, id, reason)| UnavailableImage {
                index,
                id,
                reason: reason.to_string(),
            })
            .collect(),
    })
}

/// Read the first image's FITS-compatible metadata without decoding pixels.
pub fn read_header(path: &Path) -> Result<Vec<(String, HeaderValue)>, XisfError> {
    let mut file = std::fs::File::open(path)?;
    let file_bytes = file.metadata()?.len();
    let mut parsed = parse_file(&mut file, file_bytes, Some(&header_dir(path)))?;
    let image = parsed.take(0)?;
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
    read_image_at(path, index).map(|read| read.image)
}

/// Open a top-level image by its case-sensitive XISF `id` attribute.
pub fn open_image_by_id(path: &Path, id: &str) -> Result<FitsImage, XisfError> {
    read_image_by_id(path, id).map(|read| read.image)
}

/// Decode the first image from a complete in-memory monolithic XISF file.
pub fn from_bytes(bytes: &[u8]) -> Result<FitsImage, XisfError> {
    image_from_bytes(bytes, 0)
}

/// Decode an indexed image from a complete in-memory monolithic XISF file.
pub fn image_from_bytes(bytes: &[u8], index: usize) -> Result<FitsImage, XisfError> {
    read_image_from_bytes(bytes, index).map(|read| read.image)
}

/// Read the first top-level image and the metadata describing it, in one
/// pass over the file.
///
/// The paired form of [`open`]. Reach for it when the metadata decides how to
/// read the pixels — see [`XisfImageInfo::bounds`].
pub fn read_image(path: &Path) -> Result<XisfImage, XisfError> {
    read_image_at(path, 0)
}

/// [`read_image`] for a top-level image at a zero-based index.
pub fn read_image_at(path: &Path, index: usize) -> Result<XisfImage, XisfError> {
    let mut file = std::fs::File::open(path)?;
    let file_bytes = file.metadata()?.len();
    read_image_from(
        &mut file,
        file_bytes,
        Some(&header_dir(path)),
        ImageSelection::Index(index),
    )
}

/// [`read_image`] for a top-level image with a given case-sensitive XISF
/// `id` attribute.
pub fn read_image_by_id(path: &Path, id: &str) -> Result<XisfImage, XisfError> {
    let mut file = std::fs::File::open(path)?;
    let file_bytes = file.metadata()?.len();
    read_image_from(
        &mut file,
        file_bytes,
        Some(&header_dir(path)),
        ImageSelection::Id(id),
    )
}

/// [`read_image`] for a complete in-memory monolithic XISF file.
pub fn read_image_from_bytes(bytes: &[u8], index: usize) -> Result<XisfImage, XisfError> {
    let mut reader = std::io::Cursor::new(bytes);
    read_image_from(
        &mut reader,
        bytes.len() as u64,
        None,
        ImageSelection::Index(index),
    )
}

/// Complex pixel samples in planar order, as `[real, imaginary]` pairs.
#[derive(Clone, Debug, PartialEq)]
pub enum ComplexSamples {
    C32(Vec<[f32; 2]>),
    C64(Vec<[f64; 2]>),
}

/// A decoded complex image. `FitsImage` has no complex sample type, so these
/// images have a reader of their own.
#[derive(Clone, Debug)]
pub struct ComplexImage {
    pub width: usize,
    pub height: usize,
    pub planes: usize,
    pub samples: ComplexSamples,
    pub info: XisfImageInfo,
}

/// Read a complex image (`Complex32` or `Complex64` samples) by zero-based
/// index. Real images fail with [`XisfError::Unsupported`]; open those with
/// [`open_image`].
pub fn read_complex_image(path: &Path, index: usize) -> Result<ComplexImage, XisfError> {
    let mut file = std::fs::File::open(path)?;
    let file_bytes = file.metadata()?.len();
    read_complex_from(
        &mut file,
        file_bytes,
        Some(&header_dir(path)),
        ImageSelection::Index(index),
    )
}

/// [`read_complex_image`] by case-sensitive XISF `id` attribute.
pub fn read_complex_image_by_id(path: &Path, id: &str) -> Result<ComplexImage, XisfError> {
    let mut file = std::fs::File::open(path)?;
    let file_bytes = file.metadata()?.len();
    read_complex_from(
        &mut file,
        file_bytes,
        Some(&header_dir(path)),
        ImageSelection::Id(id),
    )
}

/// [`read_complex_image`] for a complete in-memory monolithic XISF file.
pub fn read_complex_image_from_bytes(
    bytes: &[u8],
    index: usize,
) -> Result<ComplexImage, XisfError> {
    let mut reader = std::io::Cursor::new(bytes);
    read_complex_from(
        &mut reader,
        bytes.len() as u64,
        None,
        ImageSelection::Index(index),
    )
}

fn read_complex_from(
    reader: &mut (impl Read + Seek),
    file_bytes: u64,
    header_dir: Option<&Path>,
    selection: ImageSelection<'_>,
) -> Result<ComplexImage, XisfError> {
    let mut parsed = parse_file(reader, file_bytes, header_dir)?;
    let index = match selection {
        ImageSelection::Index(index) => index,
        ImageSelection::Id(id) => parsed
            .index_of_id(id)
            .ok_or_else(|| XisfError::ImageNotFound(format!("with id {id:?}")))?,
    };
    let image = parsed.take(index)?;
    let format = image.info.sample_format;
    if !format.is_complex() {
        return Err(XisfError::Unsupported(format!(
            "{format:?} samples are real; open the image with open_image"
        )));
    }
    if let Some(checksum) = &image.checksum {
        verify_checksum(reader, &image, checksum)?;
    }
    let bytes = match &image.info.compression {
        None => with_block(reader, &image, |block| {
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(image.expected_bytes)
                .map_err(|_| XisfError::Malformed("pixel buffer allocation failed".into()))?;
            read_decompressed(block, image.expected_bytes, &mut bytes)?;
            Ok(bytes)
        })?,
        Some(compression) => {
            let raw = with_block(reader, &image, |block| {
                decompress(block, &image, compression)
            })?;
            match compression.shuffled_item_bytes {
                Some(item_bytes) => unshuffle(&raw, item_bytes),
                None => raw,
            }
        }
    };
    let order = image.info.byte_order;
    let component_bytes = format.bytes_per_sample() / 2;
    let components = bytes.chunks_exact(component_bytes);
    let (width, height, planes) = (image.info.width, image.info.height, image.info.planes);
    let mut samples = match format {
        SampleFormat::Complex32 => {
            let values = components
                .map(|bytes| {
                    let bytes = bytes.try_into().unwrap();
                    match order {
                        ByteOrder::Little => f32::from_le_bytes(bytes),
                        ByteOrder::Big => f32::from_be_bytes(bytes),
                    }
                })
                .collect::<Vec<_>>();
            ComplexSamples::C32(
                values
                    .chunks_exact(2)
                    .map(|pair| [pair[0], pair[1]])
                    .collect(),
            )
        }
        _ => {
            let values = components
                .map(|bytes| {
                    let bytes = bytes.try_into().unwrap();
                    match order {
                        ByteOrder::Little => f64::from_le_bytes(bytes),
                        ByteOrder::Big => f64::from_be_bytes(bytes),
                    }
                })
                .collect::<Vec<_>>();
            ComplexSamples::C64(
                values
                    .chunks_exact(2)
                    .map(|pair| [pair[0], pair[1]])
                    .collect(),
            )
        }
    };
    if image.info.pixel_storage == PixelStorage::Normal && planes > 1 {
        match &mut samples {
            ComplexSamples::C32(values) => reorder_normal(values, planes),
            ComplexSamples::C64(values) => reorder_normal(values, planes),
        }
    }
    Ok(ComplexImage {
        width,
        height,
        planes,
        samples,
        info: image.info,
    })
}

enum ImageSelection<'a> {
    Index(usize),
    Id(&'a str),
}

/// The directory that `@header_dir` block paths in a header file resolve
/// against.
fn header_dir(path: &Path) -> PathBuf {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

fn read_image_from(
    reader: &mut (impl Read + Seek),
    file_bytes: u64,
    header_dir: Option<&Path>,
    selection: ImageSelection<'_>,
) -> Result<XisfImage, XisfError> {
    let mut parsed = parse_file(reader, file_bytes, header_dir)?;
    // Take the chosen image out of the parse, so the metadata moves into the
    // result instead of being cloned for every caller of `open`.
    let index = match selection {
        ImageSelection::Index(index) => index,
        ImageSelection::Id(id) => parsed
            .index_of_id(id)
            .ok_or_else(|| XisfError::ImageNotFound(format!("with id {id:?}")))?,
    };
    let image = parsed.take(index)?;

    if let Some(checksum) = &image.checksum {
        verify_checksum(reader, &image, checksum)?;
    }
    let pixels = decode_block(reader, &image)?;
    let mut headers = image.info.headers.clone();
    add_structural_headers(&mut headers, &image.info);

    Ok(XisfImage {
        image: FitsImage {
            width: image.info.width,
            height: image.info.height,
            planes: image.info.planes,
            pixels,
            headers,
        },
        info: image.info,
    })
}

/// What kind of XISF unit a header belongs to, which decides where its data
/// blocks may live.
#[derive(Clone, Debug)]
enum Unit {
    Monolithic {
        header_end: u64,
        file_bytes: u64,
    },
    /// A header file (`.xish`); `header_dir` is `None` for one read from
    /// memory, which leaves `@header_dir` paths unresolvable.
    Distributed {
        header_dir: Option<PathBuf>,
    },
}

/// Parse a monolithic XISF file or the header file of a distributed unit,
/// told apart by their first bytes.
fn parse_file(
    reader: &mut (impl Read + Seek),
    file_bytes: u64,
    header_dir: Option<&Path>,
) -> Result<ParsedFile, XisfError> {
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
        let start = preamble.strip_prefix(b"\xef\xbb\xbf").unwrap_or(&preamble);
        let start = start.trim_ascii_start();
        if !start.starts_with(b"<?xml") {
            return Err(XisfError::NotXisf);
        }
        let header_bytes = usize::try_from(file_bytes)
            .ok()
            .filter(|bytes| *bytes <= MAX_HEADER_BYTES)
            .ok_or_else(|| XisfError::Malformed("XML header file is too large".into()))?;
        let mut xml = Vec::with_capacity(header_bytes);
        reader.seek(SeekFrom::Start(0))?;
        reader.take(header_bytes as u64).read_to_end(&mut xml)?;
        return parse_xml(
            &xml,
            &Unit::Distributed {
                header_dir: header_dir.map(Path::to_path_buf),
            },
        );
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
    parse_xml(
        &xml,
        &Unit::Monolithic {
            header_end,
            file_bytes,
        },
    )
}

fn parse_xml(xml: &[u8], unit: &Unit) -> Result<ParsedFile, XisfError> {
    let mut reader = Reader::from_reader(xml);
    reader.config_mut().trim_text(false);
    let mut buffer = Vec::new();
    let mut state = ParseState::default();
    let mut depth = 0_usize;

    loop {
        match reader
            .read_event_into(&mut buffer)
            .map_err(|error| XisfError::Malformed(format!("invalid XML header: {error}")))?
        {
            Event::Start(element) => {
                depth += 1;
                state.start(&reader, &element, depth, false)?;
            }
            Event::Empty(element) => {
                state.start(&reader, &element, depth + 1, true)?;
            }
            Event::Text(text) => {
                let text = text
                    .decode()
                    .map_err(|error| XisfError::Malformed(format!("invalid XML text: {error}")))?;
                state.text(&text, depth);
            }
            Event::CData(text) => {
                let text = text
                    .decode()
                    .map_err(|error| XisfError::Malformed(format!("invalid XML text: {error}")))?;
                state.text(&text, depth);
            }
            Event::GeneralRef(reference) => {
                let invalid = |error: &dyn std::fmt::Display| {
                    XisfError::Malformed(format!("invalid XML reference: {error}"))
                };
                let character = reference
                    .resolve_char_ref()
                    .map_err(|error| invalid(&error))?;
                let text = match character {
                    Some(character) => character.to_string(),
                    None => {
                        let name = reference.decode().map_err(|error| invalid(&error))?;
                        quick_xml::escape::resolve_predefined_entity(&name)
                            .ok_or_else(|| {
                                XisfError::Malformed(format!("undefined XML entity &{name};"))
                            })?
                            .to_string()
                    }
                };
                state.text(&text, depth);
            }
            Event::End(_) => {
                state.end(depth);
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

    if !state.saw_root {
        return Err(XisfError::Malformed("missing xisf root element".into()));
    }
    if state.images.is_empty() {
        return Err(XisfError::Malformed("file contains no images".into()));
    }
    // Shared elements may follow the images that reference them, so images
    // are only interpreted once the whole header has been read.
    let mut parsed = ParsedFile {
        images: Vec::new(),
        unavailable: Vec::new(),
    };
    for image in state.images {
        let image = match image {
            Ok(image) => image,
            Err((index, error)) => {
                parsed.unavailable.push((index, None, error));
                continue;
            }
        };
        let (index, id) = (image.index, image.attributes.get("id").cloned());
        match parse_image(image, &state.shared, unit) {
            Ok(image) => parsed.images.push(image),
            Err(error) => parsed.unavailable.push((index, id, error)),
        }
    }
    Ok(parsed)
}

/// An XML element with its attributes and character data. Elements nested
/// inside it are not kept.
#[derive(Clone, Debug, Default)]
struct Element {
    name: String,
    attributes: BTreeMap<String, String>,
    text: String,
}

/// An `Image` element, kept whole until the header has been read.
struct PendingImage {
    index: usize,
    attributes: BTreeMap<String, String>,
    /// Character data directly inside the `Image` element: an inline block.
    text: String,
    children: Vec<Element>,
}

#[derive(Default)]
struct ParseState {
    images: Vec<Result<PendingImage, (usize, XisfError)>>,
    /// Child elements of the root with a `uid`, which `Reference` elements
    /// inside images point to.
    shared: BTreeMap<String, Element>,
    /// Whether the last `images` entry is open.
    in_image: bool,
    /// A shared root element being read.
    open_shared: Option<(String, Element)>,
    /// A child element of the open image being read.
    open_child: Option<Element>,
    saw_root: bool,
}

impl ParseState {
    fn start(
        &mut self,
        reader: &Reader<&[u8]>,
        element: &BytesStart<'_>,
        depth: usize,
        empty: bool,
    ) -> Result<(), XisfError> {
        let name = element.local_name();
        let name = String::from_utf8_lossy(name.as_ref());
        match depth {
            1 if name == "xisf" => {
                let attributes = attributes(reader, element)?;
                if attributes.get("version").map(|value| value.trim()) != Some("1.0") {
                    return Err(XisfError::Unsupported(format!(
                        "XISF version {:?}",
                        attributes.get("version")
                    )));
                }
                self.saw_root = true;
            }
            2 if name == "Image" => {
                let index = self.images.len();
                let image = attributes(reader, element)
                    .map(|attributes| PendingImage {
                        index,
                        attributes,
                        text: String::new(),
                        children: Vec::new(),
                    })
                    .map_err(|error| (index, error));
                self.in_image = image.is_ok() && !empty;
                self.images.push(image);
            }
            2 => {
                let attributes = attributes(reader, element)?;
                if let Some(uid) = attributes.get("uid") {
                    let shared = Element {
                        name: name.into_owned(),
                        attributes: attributes.clone(),
                        text: String::new(),
                    };
                    let uid = uid.trim().to_string();
                    if empty {
                        self.shared.insert(uid, shared);
                    } else {
                        self.open_shared = Some((uid, shared));
                    }
                }
            }
            3 if self.in_image => {
                let child = Element {
                    name: name.into_owned(),
                    attributes: attributes(reader, element)?,
                    text: String::new(),
                };
                if empty {
                    self.push_child(child);
                } else {
                    self.open_child = Some(child);
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn push_child(&mut self, child: Element) {
        if let Some(Ok(image)) = self.images.last_mut() {
            image.children.push(child);
        }
    }

    fn text(&mut self, text: &str, depth: usize) {
        match depth {
            3 => {
                if let Some(child) = &mut self.open_child {
                    child.text.push_str(text);
                }
            }
            2 if self.in_image => {
                if let Some(Ok(image)) = self.images.last_mut() {
                    image.text.push_str(text);
                }
            }
            2 => {
                if let Some((_, shared)) = &mut self.open_shared {
                    shared.text.push_str(text);
                }
            }
            _ => {}
        }
    }

    fn end(&mut self, depth: usize) {
        match depth {
            3 => {
                if let Some(child) = self.open_child.take() {
                    self.push_child(child);
                }
            }
            2 => {
                self.in_image = false;
                if let Some((uid, shared)) = self.open_shared.take() {
                    self.shared.insert(uid, shared);
                }
            }
            _ => {}
        }
    }
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
            .decoded_and_normalized_value(XmlVersion::Implicit1_0, reader.decoder())
            .map_err(|error| XisfError::Malformed(format!("invalid XML attribute: {error}")))?;
        values.insert(key.to_string(), value.into_owned());
    }
    Ok(values)
}

fn parse_image(
    image: PendingImage,
    shared: &BTreeMap<String, Element>,
    unit: &Unit,
) -> Result<ParsedImage, XisfError> {
    let attributes = &image.attributes;
    let mut headers = Vec::new();
    let mut properties = Vec::new();
    let mut cfa_pattern = None;
    let mut data = None;
    let mut working_space = None;
    // A Reference child stands for the shared root element it names.
    // Chained references are not allowed, so one lookup suffices.
    let children = image.children.iter().filter_map(|child| {
        if child.name == "Reference" {
            child
                .attributes
                .get("ref")
                .and_then(|reference| shared.get(reference.trim()))
        } else {
            Some(child)
        }
    });
    for child in children {
        let child_attributes = &child.attributes;
        match child.name.as_str() {
            "FITSKeyword" => {
                let keyword = required(child_attributes, "name", "FITSKeyword")?
                    .trim()
                    .to_string();
                // XISF samples are already physical and the XML geometry is
                // authoritative, so preserved FITS scaling and structure
                // keywords must not be re-applied by FITS-side consumers
                // such as into_physical_f32. COMMENT/HISTORY-style keywords
                // legitimately carry no value.
                if !structural_fits_keyword(&keyword) {
                    let raw = child_attributes
                        .get("value")
                        .map(String::as_str)
                        .unwrap_or("");
                    headers.push((keyword, parse_header_value(raw)));
                }
            }
            "Property" => {
                let mut value = child_attributes.get("value").cloned();
                if !child.text.is_empty() {
                    value.get_or_insert_with(String::new).push_str(&child.text);
                }
                properties.push(XisfProperty {
                    id: required(child_attributes, "id", "Property")?.to_string(),
                    type_name: required(child_attributes, "type", "Property")?.to_string(),
                    value,
                    comment: child_attributes.get("comment").cloned(),
                    format: child_attributes.get("format").cloned(),
                    location: child_attributes.get("location").cloned(),
                });
            }
            "ColorFilterArray" => {
                let width = parse_usize(required(child_attributes, "width", "ColorFilterArray")?)?;
                let height =
                    parse_usize(required(child_attributes, "height", "ColorFilterArray")?)?;
                let pattern = required(child_attributes, "pattern", "ColorFilterArray")?.trim();
                if width == 2 && height == 2 && pattern.len() == 4 {
                    cfa_pattern = Some(pattern.to_string());
                }
            }
            "Data" => data = Some(child),
            "RGBWorkingSpace" => working_space = Some(RgbWorkingSpace::parse(child_attributes)),
            _ => {}
        }
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
    let pixel_storage = match attributes.get("pixelStorage").map(|value| value.trim()) {
        None | Some("Planar") => PixelStorage::Planar,
        Some("Normal") => PixelStorage::Normal,
        Some(value) => {
            return Err(XisfError::Malformed(format!(
                "invalid pixel storage model {value:?}"
            )));
        }
    };
    let color_space = attributes
        .get("colorSpace")
        .map(|value| value.trim().to_string())
        .unwrap_or_else(|| "Gray".into());
    if sample_format.is_complex() && color_space == "CIELab" {
        return Err(XisfError::Unsupported(
            "complex CIELab image; complex images have no representable range".into(),
        ));
    }
    if !matches!(
        (planes, color_space.as_str()),
        (1, "Gray") | (3, "RGB") | (3, "CIELab")
    ) {
        return Err(XisfError::Unsupported(format!(
            "{planes}-channel {color_space} image"
        )));
    }
    // A working space only matters to this crate for CIELab images, so a
    // broken one fails only those.
    let rgb_working_space = match working_space {
        Some(Ok(space)) => Some(space),
        Some(Err(error)) if color_space == "CIELab" => return Err(error),
        Some(Err(_)) | None => None,
    };

    // An embedded block carries its encoding, compression and subblocks on
    // the child Data element; every other location carries them on Image.
    let location_value = required(attributes, "location", "Image")?.trim();
    let (location, inline_data, block_attributes) = match location_value.split_once(':') {
        Some(("attachment", position)) => {
            let &Unit::Monolithic {
                header_end,
                file_bytes,
            } = unit
            else {
                return Err(XisfError::Malformed(
                    "attached data block in an XISF header file".into(),
                ));
            };
            let (offset, bytes) = parse_attachment(position, location_value)?;
            let end = offset
                .checked_add(bytes)
                .ok_or_else(|| XisfError::Malformed("attachment range overflows".into()))?;
            if offset < header_end || end > file_bytes {
                return Err(XisfError::Malformed(format!(
                    "attachment {offset}:{bytes} is outside the file"
                )));
            }
            (
                BlockLocation::Attachment { offset, bytes },
                None,
                attributes,
            )
        }
        Some(("inline", encoding)) => {
            let data = decode_text_block(&image.text, encoding)?;
            let bytes = data.len() as u64;
            (BlockLocation::Inline { bytes }, Some(data), attributes)
        }
        None if location_value == "embedded" => {
            let Element {
                attributes: data_attributes,
                text,
                ..
            } = data
                .ok_or_else(|| XisfError::Malformed("embedded image has no Data element".into()))?;
            let encoding = required(data_attributes, "encoding", "Data")?;
            let data = decode_text_block(text, encoding)?;
            let bytes = data.len() as u64;
            (
                BlockLocation::Embedded { bytes },
                Some(data),
                data_attributes,
            )
        }
        _ if location_value.starts_with("url(") || location_value.starts_with("path(") => {
            let Unit::Distributed { header_dir } = unit else {
                return Err(XisfError::Malformed(format!(
                    "external data block {location_value:?} in a monolithic XISF file"
                )));
            };
            let (path, offset, bytes) = resolve_external(location_value, header_dir.as_deref())?;
            (
                BlockLocation::External {
                    path,
                    offset,
                    bytes,
                },
                None,
                attributes,
            )
        }
        _ => {
            return Err(XisfError::Malformed(format!(
                "invalid data block location {location_value:?}"
            )));
        }
    };
    let attribute = |name: &str| {
        block_attributes
            .get(name)
            .or_else(|| attributes.get(name))
            .map(|value| value.trim())
    };
    let byte_order = match attribute("byteOrder") {
        None | Some("little") => ByteOrder::Little,
        Some("big") => ByteOrder::Big,
        Some(value) => {
            return Err(XisfError::Malformed(format!(
                "invalid byte order {value:?}"
            )));
        }
    };
    let bounds = attributes
        .get("bounds")
        .and_then(|value| parse_bounds(value));
    let mut compression = attribute("compression")
        .map(parse_compression)
        .transpose()?;
    let block_bytes = location.bytes();
    if let Some(compression) = &mut compression {
        if compression.uncompressed_bytes != expected_bytes {
            return Err(XisfError::Malformed(format!(
                "declared uncompressed size {} does not match image size {expected_bytes}",
                compression.uncompressed_bytes
            )));
        }
        if let Some(subblocks) = attribute("subblocks") {
            compression.subblocks = parse_subblocks(subblocks)?;
            let compressed = compression
                .subblocks
                .iter()
                .map(|(bytes, _)| bytes)
                .sum::<u64>();
            let uncompressed = compression
                .subblocks
                .iter()
                .map(|(_, bytes)| bytes)
                .sum::<u64>();
            if compressed != block_bytes || uncompressed != expected_bytes as u64 {
                return Err(XisfError::Malformed(format!(
                    "subblocks hold {compressed} compressed and {uncompressed} uncompressed \
                     bytes; the block holds {block_bytes} and the image {expected_bytes}"
                )));
            }
        }
        if compression.shuffled_item_bytes == Some(0) {
            return Err(XisfError::Malformed("shuffle item size is zero".into()));
        }
    } else if block_bytes != expected_bytes as u64 {
        return Err(XisfError::Malformed(format!(
            "data block size {block_bytes} does not match image size {expected_bytes}"
        )));
    }

    Ok(ParsedImage {
        checksum: attribute("checksum").map(str::to_string),
        info: XisfImageInfo {
            index: image.index,
            id: attributes.get("id").cloned(),
            image_type: attributes
                .get("imageType")
                .map(|value| value.trim().to_string()),
            width,
            height,
            planes,
            sample_format,
            color_space,
            byte_order,
            pixel_storage,
            location,
            compression,
            bounds,
            headers,
            properties,
            cfa_pattern,
            rgb_working_space,
        },
        expected_bytes,
        inline_data,
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

/// Parse a `bounds="low:high"` attribute.
/// Read a `bounds="low:high"` attribute, or `None` when it is not a usable
/// range.
///
/// Deliberately lenient. Nothing in this crate needs `bounds` to decode an
/// image, and releases before it existed ignored the attribute outright, so
/// refusing a file over a spelling this crate does not recognize would break
/// images that used to open. An unusable range reads as "the file said
/// nothing", which is what it amounts to.
fn parse_bounds(value: &str) -> Option<(f64, f64)> {
    let (low, high) = value.split_once(':')?;
    let parse = |part: &str| part.trim().parse::<f64>().ok().filter(|v| v.is_finite());
    let (low, high) = (parse(low)?, parse(high)?);
    (low < high).then_some((low, high))
}

/// Parse an unsigned integer, ignoring the surrounding white space that the
/// XISF scalar serialization rules tell decoders to ignore.
fn parse_u64(value: &str) -> Result<u64, XisfError> {
    value
        .trim()
        .parse()
        .map_err(|_| XisfError::Malformed(format!("invalid unsigned integer {value:?}")))
}

fn parse_usize(value: &str) -> Result<usize, XisfError> {
    usize::try_from(parse_u64(value)?)
        .map_err(|_| XisfError::Malformed(format!("unsigned integer {value:?} is too large")))
}

/// Parse the `position:size` part of an `attachment:position:size` location.
fn parse_attachment(position: &str, location: &str) -> Result<(u64, u64), XisfError> {
    let (offset, bytes) = position
        .split_once(':')
        .ok_or_else(|| XisfError::Malformed(format!("invalid attachment location {location:?}")))?;
    Ok((parse_u64(offset)?, parse_u64(bytes)?))
}

/// Resolve a `url(...)` or `path(...)` block location, with its optional
/// `:index-id` suffix, to a local file and a byte range in it.
fn resolve_external(
    location: &str,
    header_dir: Option<&Path>,
) -> Result<(PathBuf, u64, u64), XisfError> {
    let invalid = || XisfError::Malformed(format!("invalid data block location {location:?}"));
    // The URL or path extends to the last closing parenthesis, so it may
    // itself contain parentheses.
    let close = location.rfind(')').ok_or_else(invalid)?;
    let (specification, suffix) = (&location[..close], location[close + 1..].trim());
    let path = if let Some(url) = specification.strip_prefix("url(") {
        file_url_path(url.trim())?
    } else {
        let path = specification
            .strip_prefix("path(")
            .ok_or_else(invalid)?
            .trim();
        match path.strip_prefix("@header_dir/") {
            Some(relative) => header_dir
                .ok_or_else(|| {
                    XisfError::Unsupported(
                        "a relative block path needs the header file's directory; \
                         open the header file by path"
                            .into(),
                    )
                })?
                .join(relative),
            None if Path::new(path).is_absolute() => PathBuf::from(path),
            None => return Err(invalid()),
        }
    };
    let file_bytes = std::fs::metadata(&path)?.len();
    if suffix.is_empty() {
        return Ok((path, 0, file_bytes));
    }
    let index_id = suffix.strip_prefix(':').ok_or_else(invalid)?.trim();
    let index_id = match index_id
        .strip_prefix("0x")
        .or_else(|| index_id.strip_prefix("0X"))
    {
        Some(hex) => u64::from_str_radix(hex, 16).ok(),
        None => index_id.parse().ok(),
    }
    .ok_or_else(invalid)?;
    let (offset, bytes) = find_indexed_block(&path, file_bytes, index_id)?;
    Ok((path, offset, bytes))
}

/// The local path of a `file:` URL. Other schemes are remote and
/// unsupported.
fn file_url_path(url: &str) -> Result<PathBuf, XisfError> {
    let remote = || XisfError::Unsupported(format!("remote data block {url:?}"));
    let scheme_end = url.find(':').ok_or_else(remote)?;
    if !url[..scheme_end].eq_ignore_ascii_case("file") {
        return Err(remote());
    }
    let rest = url[scheme_end + 1..]
        .strip_prefix("//")
        .ok_or_else(|| XisfError::Malformed(format!("invalid file URL {url:?}")))?;
    let (host, path) = rest.split_at(rest.find('/').unwrap_or(rest.len()));
    if !(host.is_empty() || host.eq_ignore_ascii_case("localhost")) {
        return Err(remote());
    }
    let decoded = percent_decode(path)
        .ok_or_else(|| XisfError::Malformed(format!("invalid file URL {url:?}")))?;
    // file:///C:/dir names C:/dir on Windows.
    let decoded = match decoded.as_bytes() {
        [b'/', drive, b':', ..] if cfg!(windows) && drive.is_ascii_alphabetic() => {
            decoded[1..].to_string()
        }
        _ => decoded,
    };
    Ok(PathBuf::from(decoded))
}

fn percent_decode(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let hex = std::str::from_utf8(bytes.get(index + 1..index + 3)?).ok()?;
            output.push(u8::from_str_radix(hex, 16).ok()?);
            index += 3;
        } else {
            output.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(output).ok()
}

const BLOCKS_SIGNATURE: &[u8; 8] = b"XISB0100";
const BLOCK_INDEX_ELEMENT_BYTES: u64 = 40;

/// Find the position and length of the block with a given identifier in the
/// block index of an XISF data blocks file.
fn find_indexed_block(path: &Path, file_bytes: u64, id: u64) -> Result<(u64, u64), XisfError> {
    let malformed =
        |message: String| XisfError::Malformed(format!("{}: {message}", path.display()));
    let mut file = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut header = [0_u8; 16];
    file.read_exact(&mut header)
        .map_err(|_| malformed("not an XISF data blocks file".into()))?;
    if &header[..8] != BLOCKS_SIGNATURE {
        return Err(malformed("not an XISF data blocks file".into()));
    }
    let u64_at =
        |bytes: &[u8], at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
    let mut node = 16_u64;
    let mut visited = std::collections::BTreeSet::new();
    loop {
        if !visited.insert(node) {
            return Err(malformed("the block index loops".into()));
        }
        file.seek(SeekFrom::Start(node))?;
        let mut node_header = [0_u8; 16];
        file.read_exact(&mut node_header)
            .map_err(|_| malformed("the block index is truncated".into()))?;
        let length = u64::from(u32::from_le_bytes(node_header[..4].try_into().unwrap()));
        let next = u64_at(&node_header, 8);
        let elements_bytes = length * BLOCK_INDEX_ELEMENT_BYTES;
        if node.saturating_add(16).saturating_add(elements_bytes) > file_bytes {
            return Err(malformed("the block index is truncated".into()));
        }
        let mut elements = vec![0_u8; elements_bytes as usize];
        file.read_exact(&mut elements)?;
        for element in elements.chunks_exact(BLOCK_INDEX_ELEMENT_BYTES as usize) {
            let (unique_id, position, bytes) =
                (u64_at(element, 0), u64_at(element, 8), u64_at(element, 16));
            // A zero position marks a free element, which holds no block.
            if unique_id != id || position == 0 {
                continue;
            }
            if position
                .checked_add(bytes)
                .is_none_or(|end| end > file_bytes)
            {
                return Err(malformed(format!("block {id:#x} is outside the file")));
            }
            return Ok((position, bytes));
        }
        if next == 0 {
            return Err(malformed(format!("the block index has no block {id:#x}")));
        }
        node = next;
    }
}

/// Parse a `subblocks="c1,u1:c2,u2:..."` attribute.
fn parse_subblocks(value: &str) -> Result<Vec<(u64, u64)>, XisfError> {
    value
        .split(':')
        .map(|subblock| {
            let (compressed, uncompressed) = subblock.split_once(',').ok_or_else(|| {
                XisfError::Malformed(format!("invalid compression subblocks {value:?}"))
            })?;
            Ok((parse_u64(compressed)?, parse_u64(uncompressed)?))
        })
        .collect()
}

/// Decode the Base64 or Base16 text of an inline or embedded data block.
/// White space anywhere in the text is not significant.
fn decode_text_block(text: &str, encoding: &str) -> Result<Vec<u8>, XisfError> {
    let digits = text.bytes().filter(|byte| !byte.is_ascii_whitespace());
    match encoding.trim() {
        "base64" => decode_base64(digits),
        "hex" => decode_hex(digits),
        encoding => Err(XisfError::Unsupported(format!(
            "data block encoding {encoding:?}"
        ))),
    }
}

fn decode_base64(digits: impl Iterator<Item = u8>) -> Result<Vec<u8>, XisfError> {
    let invalid = || XisfError::Malformed("invalid Base64 data block".into());
    let mut output = Vec::new();
    let (mut accumulator, mut bits, mut padded) = (0_u32, 0_u32, false);
    for digit in digits {
        let value = match digit {
            b'A'..=b'Z' => digit - b'A',
            b'a'..=b'z' => digit - b'a' + 26,
            b'0'..=b'9' => digit - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => {
                padded = true;
                continue;
            }
            _ => return Err(invalid()),
        };
        if padded {
            return Err(invalid());
        }
        accumulator = (accumulator << 6 | u32::from(value)) & 0xfff;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push((accumulator >> bits) as u8);
        }
    }
    // Two to four leftover bits are the padding of a final group; six mean
    // a lone trailing digit.
    if bits >= 6 {
        return Err(invalid());
    }
    Ok(output)
}

fn decode_hex(mut digits: impl Iterator<Item = u8>) -> Result<Vec<u8>, XisfError> {
    let invalid = || XisfError::Malformed("invalid Base16 data block".into());
    let nibble = |digit: u8| (digit as char).to_digit(16).ok_or_else(invalid);
    let mut output = Vec::new();
    while let Some(high) = digits.next() {
        let low = digits.next().ok_or_else(invalid)?;
        output.push((nibble(high)? << 4 | nibble(low)?) as u8);
    }
    Ok(output)
}

fn parse_compression(value: &str) -> Result<CompressionInfo, XisfError> {
    let parts = value.split(':').collect::<Vec<_>>();
    if !matches!(parts.len(), 2 | 3) {
        return Err(XisfError::Malformed(format!(
            "invalid compression descriptor {value:?}"
        )));
    }
    let codec_name = parts[0].trim();
    let (codec_name, shuffled) = codec_name
        .strip_suffix("+sh")
        .map_or((codec_name, false), |codec| (codec, true));
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
        subblocks: Vec::new(),
    })
}

/// Run `read` over the stored bytes of an image's data block, wherever the
/// block lives.
fn with_block<T>(
    reader: &mut (impl Read + Seek),
    image: &ParsedImage,
    read: impl FnOnce(&mut dyn Read) -> Result<T, XisfError>,
) -> Result<T, XisfError> {
    match (&image.info.location, &image.inline_data) {
        (&BlockLocation::Attachment { offset, bytes }, _) => {
            reader.seek(SeekFrom::Start(offset))?;
            read(&mut reader.take(bytes))
        }
        (
            BlockLocation::External {
                path,
                offset,
                bytes,
            },
            _,
        ) => {
            let mut file = std::fs::File::open(path)?;
            file.seek(SeekFrom::Start(*offset))?;
            read(&mut file.take(*bytes))
        }
        (_, Some(data)) => read(&mut data.as_slice()),
        (_, None) => Err(XisfError::Malformed("inline data block is missing".into())),
    }
}

fn verify_checksum(
    reader: &mut (impl Read + Seek),
    image: &ParsedImage,
    checksum: &str,
) -> Result<(), XisfError> {
    let (algorithm, expected) = checksum
        .split_once(':')
        .ok_or_else(|| XisfError::Malformed(format!("invalid checksum {checksum:?}")))?;
    let actual = match algorithm.trim() {
        "sha1" | "sha-1" => block_digest::<Sha1>(reader, image)?,
        "sha256" | "sha-256" => block_digest::<Sha256>(reader, image)?,
        "sha512" | "sha-512" => block_digest::<Sha512>(reader, image)?,
        "sha3-256" => block_digest::<Sha3_256>(reader, image)?,
        "sha3-512" => block_digest::<Sha3_512>(reader, image)?,
        algorithm => {
            return Err(XisfError::Unsupported(format!(
                "checksum algorithm {algorithm:?}"
            )));
        }
    };
    let expected = expected.trim();
    if actual != expected.to_ascii_lowercase() {
        return Err(XisfError::Malformed(format!(
            "checksum mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

/// Hash the stored bytes of a data block: the compressed bytes of a
/// compressed block, and the decoded bytes of an inline or embedded one.
fn block_digest<D: sha2::Digest>(
    reader: &mut (impl Read + Seek),
    image: &ParsedImage,
) -> Result<String, XisfError> {
    with_block(reader, image, |block| {
        let mut digest = D::new();
        let mut buffer = vec![0_u8; CHUNK_BYTES];
        loop {
            let bytes = block.read(&mut buffer)?;
            if bytes == 0 {
                break;
            }
            digest.update(&buffer[..bytes]);
        }
        Ok(lowercase_hex(digest.finalize().as_ref()))
    })
}

fn lowercase_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;

    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

fn decode_block(reader: &mut (impl Read + Seek), image: &ParsedImage) -> Result<Pixels, XisfError> {
    if image.info.sample_format.is_complex() {
        return Err(complex_needs_its_own_reader());
    }
    let mut pixels = match &image.info.compression {
        None => with_block(reader, image, |block| decode_reader(block, image))?,
        Some(compression) => {
            let raw = with_block(reader, image, |block| decompress(block, image, compression))?;
            match compression.shuffled_item_bytes {
                None => decode_bytes(&raw, image)?,
                Some(item_bytes) if item_bytes == image.info.sample_format.bytes_per_sample() => {
                    decode_shuffled(&raw, image)?
                }
                Some(item_bytes) => decode_bytes(&unshuffle(&raw, item_bytes), image)?,
            }
        }
    };
    if image.info.pixel_storage == PixelStorage::Normal {
        planar_from_normal(&mut pixels, image.info.planes);
    }
    if image.info.color_space == "CIELab" {
        rgb_from_lab(&mut pixels, &image.info)?;
    }
    Ok(pixels)
}

/// Decompress a whole data block, one independently compressed subblock at a
/// time. A block without a `subblocks` attribute is a single subblock.
fn decompress(
    block: &mut dyn Read,
    image: &ParsedImage,
    compression: &CompressionInfo,
) -> Result<Vec<u8>, XisfError> {
    let whole = [(image.info.location.bytes(), image.expected_bytes as u64)];
    let subblocks = if compression.subblocks.is_empty() {
        &whole[..]
    } else {
        &compression.subblocks[..]
    };
    let mut raw = Vec::new();
    raw.try_reserve_exact(image.expected_bytes)
        .map_err(|_| XisfError::Malformed("decompression allocation failed".into()))?;
    for &(compressed, uncompressed) in subblocks {
        let uncompressed = usize::try_from(uncompressed)
            .map_err(|_| XisfError::Malformed("subblock is too large".into()))?;
        let mut stored = Read::take(&mut *block, compressed);
        match compression.codec {
            CompressionCodec::Zlib => {
                read_decompressed(ZlibDecoder::new(stored), uncompressed, &mut raw)?;
            }
            CompressionCodec::Zstd => {
                let decoder = zstd::stream::read::Decoder::new(stored).map_err(|error| {
                    XisfError::Malformed(format!("invalid zstd stream: {error}"))
                })?;
                read_decompressed(decoder, uncompressed, &mut raw)?;
            }
            CompressionCodec::Lz4 | CompressionCodec::Lz4Hc => {
                let compressed = usize::try_from(compressed)
                    .map_err(|_| XisfError::Malformed("subblock is too large".into()))?;
                let mut bytes = vec![0_u8; compressed];
                stored.read_exact(&mut bytes).map_err(|error| {
                    if error.kind() == std::io::ErrorKind::UnexpectedEof {
                        XisfError::Malformed("compressed data block is truncated".into())
                    } else {
                        XisfError::Io(error)
                    }
                })?;
                let decoded = lz4_flex::block::decompress(&bytes, uncompressed)
                    .map_err(|error| XisfError::Malformed(format!("invalid LZ4 block: {error}")))?;
                if decoded.len() != uncompressed {
                    return Err(XisfError::Malformed(format!(
                        "decompressed {} bytes; expected {uncompressed}",
                        decoded.len()
                    )));
                }
                raw.extend_from_slice(&decoded);
            }
        }
    }
    if raw.len() != image.expected_bytes {
        return Err(XisfError::Malformed(format!(
            "decompressed {} bytes; expected {}",
            raw.len(),
            image.expected_bytes
        )));
    }
    Ok(raw)
}

/// Append exactly `expected_bytes` decompressed bytes to `output`.
fn read_decompressed(
    reader: impl Read,
    expected_bytes: usize,
    output: &mut Vec<u8>,
) -> Result<(), XisfError> {
    let limit = expected_bytes
        .checked_add(1)
        .ok_or_else(|| XisfError::Malformed("decompressed size overflows".into()))?;
    let before = output.len();
    reader.take(limit as u64).read_to_end(output)?;
    let read = output.len() - before;
    if read != expected_bytes {
        return Err(XisfError::Malformed(format!(
            "decompressed {read} bytes; expected {expected_bytes}"
        )));
    }
    Ok(())
}

/// Reverse the XISF byte shuffle for an arbitrary item size. Trailing bytes
/// that do not form a whole item are stored unshuffled.
fn unshuffle(bytes: &[u8], item_bytes: usize) -> Vec<u8> {
    let items = bytes.len() / item_bytes;
    let mut output = vec![0_u8; bytes.len()];
    for (lane, stored) in bytes[..items * item_bytes]
        .chunks_exact(items.max(1))
        .enumerate()
    {
        for (item, &byte) in stored.iter().enumerate() {
            output[item * item_bytes + lane] = byte;
        }
    }
    output[items * item_bytes..].copy_from_slice(&bytes[items * item_bytes..]);
    output
}

fn reorder_normal<T: Copy>(values: &mut Vec<T>, planes: usize) {
    let mut planar = Vec::with_capacity(values.len());
    for channel in 0..planes {
        planar.extend(values.iter().skip(channel).step_by(planes).copied());
    }
    *values = planar;
}

/// Reorder pixel-by-pixel samples into the channel-by-channel order that
/// decoded images use.
fn planar_from_normal(pixels: &mut Pixels, planes: usize) {
    if planes < 2 {
        return;
    }
    match pixels {
        Pixels::U8(values) => reorder_normal(values, planes),
        Pixels::U16(values) => reorder_normal(values, planes),
        Pixels::I32(values) => reorder_normal(values, planes),
        Pixels::F32(values) => reorder_normal(values, planes),
        Pixels::F64(values) => reorder_normal(values, planes),
    }
}

fn decode_reader(reader: &mut dyn Read, image: &ParsedImage) -> Result<Pixels, XisfError> {
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
        SampleFormat::UInt32 | SampleFormat::UInt64 | SampleFormat::Float64 => {
            reserve(count).map(Pixels::F64)
        }
        SampleFormat::Float32 => reserve(count).map(Pixels::F32),
        SampleFormat::Complex32 | SampleFormat::Complex64 => Err(complex_needs_its_own_reader()),
    }
}

fn complex_needs_its_own_reader() -> XisfError {
    XisfError::Unsupported("complex samples; use read_complex_image".into())
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
        (Pixels::F64(values), SampleFormat::UInt64) => {
            values.extend(bytes.chunks_exact(8).map(|bytes| {
                let bytes = bytes.try_into().unwrap();
                match order {
                    ByteOrder::Little => u64::from_le_bytes(bytes) as f64,
                    ByteOrder::Big => u64::from_be_bytes(bytes) as f64,
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
        (Pixels::F64(values), SampleFormat::UInt64) => values.extend((0..count).map(|sample| {
            u64::from_le_bytes(std::array::from_fn(|lane| byte_at(sample, lane))) as f64
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
        if let Some(value) = property(property_id)
            .and_then(|value| value.trim().parse::<f64>().ok())
            .filter(|value| value.is_finite())
        {
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
        // Bytes past the last whole item stay in place.
        output[count * item_bytes..].copy_from_slice(&bytes[count * item_bytes..]);
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
    fn undefined_fits_keywords_round_trip_through_both_writers() {
        use seiza_fits::{F32ImageData, WriteHeaderCard};

        let samples = [0.25_f32, 0.5, 1.0, -0.5];
        let raw = samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        for value_attribute in ["", " value=\"\"", " value=\"        \"", " value=\"''\""] {
            let expected = if value_attribute.contains("''") {
                HeaderValue::String(String::new())
            } else {
                HeaderValue::Raw(String::new())
            };
            let xml = image_element(
                0,
                "2:2:1",
                "Float32",
                raw.len(),
                "bounds=\"0:1\" colorSpace=\"Gray\" imageType=\"Bias\"",
                &format!(
                    "<FITSKeyword name=\"FILTER\"{value_attribute}/><FITSKeyword name=\"EXPTIME\" value=\"0\"/>"
                ),
            );
            let decoded = from_bytes(&monolithic(xml, &[&raw])).unwrap();
            assert_eq!(decoded.header("FILTER"), Some(&expected));
            let headers = ["FILTER", "EXPTIME", "IMAGETYP"].map(|keyword| {
                WriteHeaderCard::new(keyword, decoded.header(keyword).unwrap().clone())
            });
            let Pixels::F32(pixels) = decoded.pixels else {
                panic!("Float32 samples must preserve their representation");
            };

            let mut fits = Vec::new();
            seiza_fits::write_f32_image_to(
                &mut fits,
                decoded.width,
                decoded.height,
                F32ImageData::Mono(&pixels),
                &headers,
            )
            .unwrap();
            let mut xisf = Vec::new();
            write_f32_image_to(
                &mut xisf,
                decoded.width,
                decoded.height,
                F32ImageData::Mono(&pixels),
                &headers,
            )
            .unwrap();

            for round_tripped in [
                FitsImage::from_bytes(&fits).unwrap(),
                from_bytes(&xisf).unwrap(),
            ] {
                assert_eq!(round_tripped.header("FILTER"), Some(&expected));
                assert_eq!(round_tripped.header_f64("EXPTIME"), Some(0.0));
                assert_eq!(round_tripped.header_str("IMAGETYP"), Some("Bias"));
                assert_eq!(
                    (
                        round_tripped.width,
                        round_tripped.height,
                        round_tripped.planes
                    ),
                    (2, 2, 1)
                );
                assert!(matches!(round_tripped.pixels, Pixels::F32(actual) if actual == samples));
            }
        }
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
        let parsed = parse_file(&mut cursor, bytes.len() as u64, None).unwrap();
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
        let image = read_image_from(
            &mut cursor,
            bytes.len() as u64,
            None,
            ImageSelection::Id("image1"),
        )
        .unwrap()
        .image;
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

    fn u8_samples(image: &FitsImage) -> Vec<u8> {
        match &image.pixels {
            Pixels::U8(values) => values.clone(),
            other => panic!("expected UInt8 samples, got {other:?}"),
        }
    }

    /// The 6x6 RGB image from the Revision 1 examples, serialized both
    /// uncompressed and zlib-compressed in an embedded Data element.
    #[test]
    fn decodes_the_specification_embedded_block_examples() {
        let plain = "<Image geometry=\"6:6:3\" sampleFormat=\"UInt8\" colorSpace=\"RGB\" location=\"embedded\">
           <Data encoding=\"base64\">
              AAAAAP8A/wD/AAAAAAAAAP8AAP8AAAAAAAAA/wD/AP8AAAAA/wD//wD/AP8AAP8A/wD//wD//
              wD//wD/AP8AAP8A/wD//wD/AP8AAAAAAAAA/wD/AP8AAAAAAAAAAP8A/wD/AAAAAAAAAP8A
           </Data>
        </Image>";
        let compressed = "<Image geometry=\"6:6:3\" sampleFormat=\"UInt8\" colorSpace=\"RGB\" location=\"embedded\">
           <Data compression=\"zlib:108\" encoding=\"base64\">
              eJxjYGBg+A+GEPCfAYkJFQZSUPZ/KBtTBFMXOuc/AwCjKyPd
           </Data>
        </Image>";
        let plain = from_bytes(&monolithic(plain.into(), &[])).unwrap();
        let compressed = from_bytes(&monolithic(compressed.into(), &[])).unwrap();
        assert_eq!((plain.width, plain.height, plain.planes), (6, 6, 3));
        assert_eq!(u8_samples(&plain).len(), 108);
        assert_eq!(u8_samples(&plain), u8_samples(&compressed));
    }

    #[test]
    fn decodes_inline_hex_blocks_and_checks_their_decoded_bytes() {
        let raw = [0x01_u8, 0xab, 0xff, 0x10];
        let digest = lowercase_hex(<Sha1 as sha1::Digest>::digest(raw).as_ref());
        let xml = format!(
            "<Image geometry=\"2:2:1\" sampleFormat=\"UInt8\" location=\"inline:hex\" checksum=\"sha1:{digest}\">\n 01ab\n ff10\n</Image>"
        );
        let image = from_bytes(&monolithic(xml, &[])).unwrap();
        assert_eq!(u8_samples(&image), raw);
    }

    #[test]
    fn reorders_normal_pixel_storage_into_planes() {
        // Three RGB pixels stored pixel by pixel.
        let values = [1_u16, 10, 100, 2, 20, 200, 3, 30, 300];
        let raw = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let xml = image_element(
            0,
            "3:1:3",
            "UInt16",
            raw.len(),
            "colorSpace=\"RGB\" pixelStorage=\"Normal\"",
            "",
        );
        let image = from_bytes(&monolithic(xml, &[&raw])).unwrap();
        assert!(matches!(
            image.pixels,
            Pixels::U16(ref actual) if actual == &[1, 2, 3, 10, 20, 30, 100, 200, 300]
        ));
    }

    #[test]
    fn decodes_shuffled_subblocks_compressed_independently() {
        let values = (0..64_u16).map(|value| value * 1000).collect::<Vec<_>>();
        let raw = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        // The shuffle spans the whole block; only then is it split.
        let shuffled = shuffled(&raw, 2);
        let (first, second) = shuffled.split_at(50);
        let first = zstd::bulk::compress(first, 3).unwrap();
        let second = zstd::bulk::compress(second, 3).unwrap();
        let stored = [first.as_slice(), second.as_slice()].concat();
        let xml = image_element(
            0,
            "8:8:1",
            "UInt16",
            stored.len(),
            &format!(
                "compression=\"zstd+sh:128:2\" subblocks=\"{},50:{},78\"",
                first.len(),
                second.len()
            ),
            "",
        );
        let image = from_bytes(&monolithic(xml, &[&stored])).unwrap();
        assert!(matches!(image.pixels, Pixels::U16(ref actual) if actual == &values));

        let lz4_first = lz4_flex::block::compress(&raw[..40]);
        let lz4_second = lz4_flex::block::compress(&raw[40..]);
        let stored = [lz4_first.as_slice(), lz4_second.as_slice()].concat();
        let xml = image_element(
            0,
            "8:8:1",
            "UInt16",
            stored.len(),
            &format!(
                "compression=\"lz4:128\" subblocks=\"{},40:{},88\"",
                lz4_first.len(),
                lz4_second.len()
            ),
            "",
        );
        let image = from_bytes(&monolithic(xml, &[&stored])).unwrap();
        assert!(matches!(image.pixels, Pixels::U16(ref actual) if actual == &values));
    }

    #[test]
    fn unshuffles_items_of_any_size() {
        // Four-byte items over UInt16 samples, with a trailing partial item.
        let raw = (0_u8..14).collect::<Vec<_>>();
        assert_eq!(unshuffle(&shuffled(&raw, 4), 4), raw);
        let values = (0..7_u16).collect::<Vec<_>>();
        let raw = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let compressed = zstd::bulk::compress(&shuffled(&raw, 4), 3).unwrap();
        let xml = image_element(
            0,
            "7:1:1",
            "UInt16",
            compressed.len(),
            "compression=\"zstd+sh:14:4\"",
            "",
        );
        let image = from_bytes(&monolithic(xml, &[&compressed])).unwrap();
        assert!(matches!(image.pixels, Pixels::U16(ref actual) if actual == &values));
    }

    #[test]
    fn an_unsupported_image_leaves_the_rest_of_the_file_readable() {
        let raw = [1_u8, 2, 3, 4];
        let xml = format!(
            "<Image id=\"alpha\" geometry=\"2:2:2\" sampleFormat=\"UInt8\" colorSpace=\"Gray\" location=\"attachment:@OFFSET0@:8\"/>\
             <Image id=\"cube\" geometry=\"2:2:2:1\" sampleFormat=\"UInt8\" location=\"attachment:@OFFSET0@:16\"/>\
             <Image id=\"remote\" geometry=\"2:2:1\" sampleFormat=\"UInt8\" location=\"url(https://example.com/a(1).bin)\"/>\
             {}",
            image_element(1, "2:2:1", "UInt8", raw.len(), "", "")
                .replace("id=\"image1\"", "id=\"gray\"")
        );
        let padding = [0_u8; 16];
        let bytes = monolithic(xml, &[&padding, &raw]);

        let image = image_from_bytes(&bytes, 3).unwrap();
        assert_eq!(u8_samples(&image), raw);
        assert!(matches!(from_bytes(&bytes), Err(XisfError::Unsupported(_))));
        assert!(matches!(
            image_from_bytes(&bytes, 1),
            Err(XisfError::Unsupported(_))
        ));
        // External blocks are not allowed in a monolithic file.
        assert!(matches!(
            image_from_bytes(&bytes, 2),
            Err(XisfError::Malformed(_))
        ));
        assert!(matches!(
            image_from_bytes(&bytes, 4),
            Err(XisfError::ImageNotFound(_))
        ));

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("mixed.xisf");
        std::fs::write(&path, &bytes).unwrap();
        let info = inspect(&path).unwrap();
        assert_eq!(info.images.len(), 1);
        assert_eq!(info.images[0].index, 3);
        let unavailable = info
            .unavailable
            .iter()
            .map(|image| (image.index, image.id.as_deref()))
            .collect::<Vec<_>>();
        assert_eq!(
            unavailable,
            [(0, Some("alpha")), (1, Some("cube")), (2, Some("remote"))]
        );
        assert!(open_image_by_id(&path, "gray").is_ok());
        assert!(matches!(
            open_image_by_id(&path, "alpha"),
            Err(XisfError::Unsupported(_))
        ));
    }

    #[test]
    fn ignores_white_space_around_scalar_attributes() {
        let raw = [1_u8, 2, 3, 4];
        let digest = lowercase_hex(<Sha1 as sha1::Digest>::digest(raw).as_ref());
        let xml = format!(
            "<Image geometry=\" 2 : 2 : 1 \" sampleFormat=\"UInt8\" colorSpace=\" Gray \" location=\" attachment: @OFFSET0@ : 4 \" checksum=\"sha1: {digest} \"/>"
        );
        let image = from_bytes(&monolithic(xml, &[&raw])).unwrap();
        assert_eq!(u8_samples(&image), raw);
    }

    #[test]
    fn verifies_sha3_checksums() {
        let raw = [1_u8, 2, 3, 4];
        for (name, digest) in [
            (
                "sha3-256",
                lowercase_hex(<Sha3_256 as sha3::Digest>::digest(raw).as_ref()),
            ),
            (
                "sha3-512",
                lowercase_hex(<Sha3_512 as sha3::Digest>::digest(raw).as_ref()),
            ),
        ] {
            let xml = image_element(
                0,
                "2:2:1",
                "UInt8",
                raw.len(),
                &format!("checksum=\"{name}:{digest}\""),
                "",
            );
            assert!(from_bytes(&monolithic(xml.clone(), &[&raw])).is_ok());
            assert!(matches!(
                from_bytes(&monolithic(xml, &[&[9, 9, 9, 9]])),
                Err(XisfError::Malformed(message)) if message.contains("checksum mismatch")
            ));
        }
        // FIPS 202 SHA3-256 of the empty message, not the Keccak one.
        assert_eq!(
            lowercase_hex(<Sha3_256 as sha3::Digest>::digest(b"").as_ref()),
            "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
        );
    }

    #[test]
    fn keeps_entity_references_in_property_text() {
        let raw = [1_u8, 2, 3, 4];
        let xml = image_element(
            0,
            "2:2:1",
            "UInt8",
            raw.len(),
            "",
            "<Property id=\"Observation:Object:Name\" type=\"String\">M 31 &amp; M 32 &#x2014; &lt;wide&gt;</Property>",
        );
        let image = from_bytes(&monolithic(xml, &[&raw])).unwrap();
        assert_eq!(
            image.header_str("OBJECT"),
            Some("M 31 & M 32 \u{2014} <wide>")
        );
    }

    #[test]
    fn reads_a_signed_header_with_a_detached_signature() {
        let raw = [1_u8, 2, 3, 4];
        let image = image_element(0, "2:2:1", "UInt8", raw.len(), "", "");
        let signed = |offset: usize| {
            format!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<xisf version=\"1.0\" id=\"XISFRootElement\">\n  {}\n</xisf>\n<Signature xmlns=\"http://www.w3.org/2000/09/xmldsig#\"><SignedInfo><Reference URI=\"#XISFRootElement\"/></SignedInfo><SignatureValue>AAAA</SignatureValue></Signature>",
                image.replace("@OFFSET0@", &offset.to_string())
            )
        };
        let mut offset = 0;
        let header = loop {
            let header = signed(offset);
            if 16 + header.len() == offset {
                break header;
            }
            offset = 16 + header.len();
        };
        let mut bytes = Vec::new();
        bytes.extend_from_slice(SIGNATURE);
        bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&[0; 4]);
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&raw);
        assert_eq!(u8_samples(&from_bytes(&bytes).unwrap()), raw);
    }

    #[test]
    fn projects_only_finite_observation_coordinates() {
        let raw = [1_u8, 2, 3, 4];
        let xml = image_element(
            0,
            "2:2:1",
            "UInt8",
            raw.len(),
            "",
            "<Property id=\"Observation:Center:RA\" type=\"Float64\" value=\" 83.822 \"/>\
             <Property id=\"Observation:Center:Dec\" type=\"Float64\" value=\"-nan\"/>",
        );
        let image = from_bytes(&monolithic(xml, &[&raw])).unwrap();
        assert!(
            image
                .headers
                .iter()
                .any(|(name, value)| name == "RA" && *value == HeaderValue::Float(83.822))
        );
        assert!(!image.headers.iter().any(|(name, _)| name == "DEC"));
    }

    #[test]
    fn decodes_uint64_samples_in_both_byte_orders() {
        let values = [0_u64, 1, 1 << 40, (1 << 53) + 2];
        let expected = values.map(|value| value as f64);
        for (order, raw) in [
            (
                "little",
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            (
                "big",
                values
                    .iter()
                    .flat_map(|value| value.to_be_bytes())
                    .collect::<Vec<_>>(),
            ),
        ] {
            let xml = image_element(
                0,
                "2:2:1",
                "UInt64",
                raw.len(),
                &format!("byteOrder=\"{order}\""),
                "",
            );
            let image = from_bytes(&monolithic(xml, &[&raw])).unwrap();
            assert!(matches!(image.pixels, Pixels::F64(ref actual) if actual == &expected));
        }
        let raw = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let compressed = zstd::bulk::compress(&shuffled(&raw, 8), 3).unwrap();
        let xml = image_element(
            0,
            "2:2:1",
            "UInt64",
            compressed.len(),
            "compression=\"zstd+sh:32:8\"",
            "",
        );
        let image = from_bytes(&monolithic(xml, &[&compressed])).unwrap();
        assert!(matches!(image.pixels, Pixels::F64(ref actual) if actual == &expected));
    }

    #[test]
    fn default_working_space_takes_linear_white_to_d50() {
        let white = RgbWorkingSpace::srgb()
            .xyz_from_linear_rgb()
            .apply([1.0, 1.0, 1.0]);
        for (actual, expected) in white.into_iter().zip([0.96422, 1.0, 0.82521]) {
            assert!((actual - expected).abs() < 1e-5, "{white:?}");
        }
    }

    /// Annex B forward transform, RGB to normalized L*a*b*.
    fn lab_from_rgb(space: &RgbWorkingSpace, rgb: [f64; 3]) -> [f64; 3] {
        let linear = rgb.map(|value| match space.gamma {
            Gamma::Srgb if value <= 0.04045 => value / 12.92,
            Gamma::Srgb => ((value + 0.055) / 1.055).powf(2.4),
            Gamma::Exponent(gamma) => value.powf(gamma),
        });
        let [x, y, z] = space.xyz_from_linear_rgb().apply(linear);
        let f = |t: f64| {
            if t > 216.0 / 24389.0 {
                t.cbrt()
            } else {
                (24389.0 / 27.0 * t + 16.0) / 116.0
            }
        };
        let (f_x, f_y, f_z) = (f(x / 0.96422), f(y), f(z / 0.82521));
        [
            1.16 * f_y - 0.16,
            0.5 + 29.0 / 50.0 * (f_x - f_y),
            0.5 + 29.0 / 50.0 * (f_y - f_z),
        ]
    }

    const COLORS: [[f64; 3]; 6] = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.8, 0.1, 0.2],
        [0.05, 0.6, 0.3],
        [0.2, 0.3, 0.9],
        [0.002, 0.001, 0.003],
    ];

    fn lab_planes(space: &RgbWorkingSpace) -> Vec<f64> {
        let lab = COLORS.map(|rgb| lab_from_rgb(space, rgb));
        (0..3)
            .flat_map(|plane| lab.iter().map(move |pixel| pixel[plane]))
            .collect()
    }

    fn rgb_planes() -> Vec<f64> {
        (0..3)
            .flat_map(|plane| COLORS.iter().map(move |pixel| pixel[plane]))
            .collect()
    }

    #[test]
    fn converts_cielab_to_rgb_in_the_default_space() {
        let lab = lab_planes(&RgbWorkingSpace::srgb());
        let raw = lab
            .iter()
            .flat_map(|value| (*value as f32).to_le_bytes())
            .collect::<Vec<_>>();
        let xml = image_element(
            0,
            "6:1:3",
            "Float32",
            raw.len(),
            "colorSpace=\"CIELab\" bounds=\"0:1\"",
            "",
        );
        let image = from_bytes(&monolithic(xml, &[&raw])).unwrap();
        let Pixels::F32(actual) = &image.pixels else {
            panic!("CIELab Float32 must decode to f32");
        };
        for (actual, expected) in actual.iter().zip(rgb_planes()) {
            assert!(
                (f64::from(*actual) - expected).abs() < 1e-5,
                "{actual} vs {expected}"
            );
        }

        // A float image without bounds cannot be mapped to nominal values.
        let xml = image_element(
            0,
            "6:1:3",
            "Float32",
            raw.len(),
            "colorSpace=\"CIELab\"",
            "",
        );
        assert!(matches!(
            from_bytes(&monolithic(xml, &[&raw])),
            Err(XisfError::Malformed(message)) if message.contains("bounds")
        ));
    }

    #[test]
    fn converts_cielab_with_a_referenced_working_space() {
        let linear = RgbWorkingSpace {
            name: None,
            gamma: Gamma::Exponent(1.0),
            ..RgbWorkingSpace::srgb()
        };
        let raw = lab_planes(&linear)
            .iter()
            .flat_map(|value| ((value * 65535.0).round() as u16).to_le_bytes())
            .collect::<Vec<_>>();
        let xml = format!(
            "{}<RGBWorkingSpace uid=\"linear\" x=\"0.648431:0.321152:0.155886\" y=\"0.330856:0.597871:0.066044\" Y=\"0.222491:0.716888:0.060621\" gamma=\"1\" name=\"Linear sRGB\"/>\
             <FITSKeyword uid=\"shared-object\" name=\"OBJECT\" value=\"'M 42'\" comment=\"\"/>",
            image_element(
                0,
                "6:1:3",
                "UInt16",
                raw.len(),
                "colorSpace=\"CIELab\"",
                "<Reference ref=\"linear\"/><Reference ref=\"shared-object\"/>",
            )
        );
        let read = read_image_from_bytes(&monolithic(xml, &[&raw]), 0).unwrap();
        assert_eq!(
            read.info
                .rgb_working_space
                .as_ref()
                .map(|space| space.gamma),
            Some(Gamma::Exponent(1.0))
        );
        assert_eq!(read.image.header_str("OBJECT"), Some("M 42"));
        let Pixels::U16(actual) = &read.image.pixels else {
            panic!("CIELab UInt16 must decode to u16");
        };
        for (actual, expected) in actual.iter().zip(rgb_planes()) {
            // 16-bit Lab quantization moves dark colors by a few counts.
            assert!(
                (f64::from(*actual) / 65535.0 - expected).abs() < 2e-3,
                "{actual} vs {expected}"
            );
        }
    }

    #[test]
    fn a_broken_working_space_only_fails_cielab_images() {
        let raw = [0_u8; 3];
        let broken =
            "<RGBWorkingSpace x=\"0.6:0.3\" y=\"0.3:0.6:0.06\" Y=\"0.2:0.7:0.1\" gamma=\"2.2\"/>";
        let rgb = image_element(0, "1:1:3", "UInt8", 3, "colorSpace=\"RGB\"", broken);
        let read = read_image_from_bytes(&monolithic(rgb, &[&raw]), 0).unwrap();
        assert!(read.info.rgb_working_space.is_none());
        let lab = image_element(0, "1:1:3", "UInt8", 3, "colorSpace=\"CIELab\"", broken);
        assert!(matches!(
            from_bytes(&monolithic(lab, &[&raw])),
            Err(XisfError::Malformed(_))
        ));
    }

    /// Build an XISF data blocks file whose index spans two linked nodes,
    /// with a free element in the first.
    fn data_blocks_file(blocks: &[(u64, &[u8])]) -> Vec<u8> {
        let (first, second) = blocks.split_at(blocks.len() / 2);
        let first_node_bytes = 16 + 40 * (first.len() as u64 + 1);
        let second_node = 16 + first_node_bytes;
        let mut position = second_node + 16 + 40 * second.len() as u64;
        let mut element = |id: u64, data: &[u8]| {
            let mut bytes = Vec::new();
            for value in [id, position, data.len() as u64, 0, 0] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            position += data.len() as u64;
            bytes
        };
        let mut file = BLOCKS_SIGNATURE.to_vec();
        file.extend_from_slice(&[0; 8]);
        file.extend_from_slice(&(first.len() as u32 + 1).to_le_bytes());
        file.extend_from_slice(&[0; 4]);
        file.extend_from_slice(&second_node.to_le_bytes());
        // A free element: any identifier, position zero.
        file.extend_from_slice(&[0xee; 8]);
        file.extend_from_slice(&[0; 32]);
        for (id, data) in first {
            let bytes = element(*id, data);
            file.extend_from_slice(&bytes);
        }
        file.extend_from_slice(&(second.len() as u32).to_le_bytes());
        file.extend_from_slice(&[0; 12]);
        for (id, data) in second {
            let bytes = element(*id, data);
            file.extend_from_slice(&bytes);
        }
        for (_, data) in blocks {
            file.extend_from_slice(data);
        }
        file
    }

    fn header_file(images: &str) -> String {
        format!(
            "\u{feff}<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<xisf version=\"1.0\" xmlns=\"http://www.pixinsight.com/xisf\">{images}</xisf>"
        )
    }

    #[test]
    fn reads_distributed_units_from_local_files() {
        let directory = tempfile::tempdir().unwrap();
        let nested = directory.path().join("raw data");
        std::fs::create_dir(&nested).unwrap();
        let gray = [1_u8, 2, 3, 4];
        std::fs::write(nested.join("frame(1).bin"), gray).unwrap();

        let values = [10_u16, 20, 30, 40];
        let raw = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let compressed = zstd::bulk::compress(&shuffled(&raw, 2), 3).unwrap();
        let digest = lowercase_hex(<Sha1 as sha1::Digest>::digest(&compressed).as_ref());
        let other = [9_u8; 4];
        let blocks = data_blocks_file(&[
            (0x1111, &other),
            (0x7a73_526b_584c_6167, &compressed),
            (0x2222, &gray),
        ]);
        std::fs::write(directory.path().join("blocks.xisb"), blocks).unwrap();

        let url_path = nested
            .join("frame(1).bin")
            .to_string_lossy()
            .replace('\\', "/")
            .replace(' ', "%20");
        let url_path = if url_path.starts_with('/') {
            url_path
        } else {
            format!("/{url_path}")
        };
        let header = header_file(&format!(
            "<Image id=\"relative\" geometry=\"2:2:1\" sampleFormat=\"UInt8\" location=\"path(@header_dir/raw data/frame(1).bin)\"/>\
             <Image id=\"indexed\" geometry=\"2:2:1\" sampleFormat=\"UInt16\" compression=\"zstd+sh:8:2\" checksum=\"sha1:{digest}\" location=\"path(@header_dir/blocks.xisb):0x7a73526b584c6167\"/>\
             <Image id=\"decimal\" geometry=\"2:2:1\" sampleFormat=\"UInt8\" location=\"path(@header_dir/blocks.xisb):8738\"/>\
             <Image id=\"url\" geometry=\"2:2:1\" sampleFormat=\"UInt8\" location=\"url(file://{url_path})\"/>\
             <Image id=\"remote\" geometry=\"2:2:1\" sampleFormat=\"UInt8\" location=\"url(https://example.com/frame.bin)\"/>\
             <Image id=\"missing\" geometry=\"2:2:1\" sampleFormat=\"UInt8\" location=\"path(@header_dir/blocks.xisb):0x3333\"/>\
             <Image id=\"attached\" geometry=\"2:2:1\" sampleFormat=\"UInt8\" location=\"attachment:0:4\"/>"
        ));
        let path = directory.path().join("unit.xish");
        std::fs::write(&path, &header).unwrap();
        assert!(is_xisf_path(&path));

        assert_eq!(
            u8_samples(&open_image_by_id(&path, "relative").unwrap()),
            gray
        );
        assert!(matches!(
            open_image_by_id(&path, "indexed").unwrap().pixels,
            Pixels::U16(ref actual) if actual == &values
        ));
        assert_eq!(
            u8_samples(&open_image_by_id(&path, "decimal").unwrap()),
            gray
        );
        assert_eq!(u8_samples(&open_image_by_id(&path, "url").unwrap()), gray);
        assert!(matches!(
            open_image_by_id(&path, "remote"),
            Err(XisfError::Unsupported(_))
        ));
        assert!(matches!(
            open_image_by_id(&path, "missing"),
            Err(XisfError::Malformed(message)) if message.contains("no block 0x3333")
        ));
        assert!(matches!(
            open_image_by_id(&path, "attached"),
            Err(XisfError::Malformed(_))
        ));
        let info = inspect(&path).unwrap();
        assert_eq!(info.images.len(), 4);
        assert!(matches!(
            &info.images[1].location,
            BlockLocation::External { bytes, .. } if *bytes == compressed.len() as u64
        ));

        // From memory there is no header directory to resolve against.
        assert!(matches!(
            from_bytes(header.as_bytes()),
            Err(XisfError::Unsupported(message)) if message.contains("header file")
        ));
    }

    #[test]
    fn reads_complex_samples_through_their_own_api() {
        let values = [[1.5_f32, -2.0], [0.0, 3.25], [-4.0, 0.5], [7.0, -8.0]];
        let raw = values
            .iter()
            .flatten()
            .flat_map(|value| value.to_be_bytes())
            .collect::<Vec<_>>();
        let xml = image_element(0, "2:2:1", "Complex32", raw.len(), "byteOrder=\"big\"", "");
        let bytes = monolithic(xml, &[&raw]);
        let image = read_complex_image_from_bytes(&bytes, 0).unwrap();
        assert_eq!((image.width, image.height, image.planes), (2, 2, 1));
        assert_eq!(image.samples, ComplexSamples::C32(values.to_vec()));
        assert!(matches!(
            from_bytes(&bytes),
            Err(XisfError::Unsupported(message)) if message.contains("read_complex_image")
        ));

        // Complex64 RGB, stored pixel by pixel and shuffled as 16-byte items.
        let pixel = |index: usize, plane: usize| [index as f64, plane as f64 * 10.0];
        let normal = (0..2)
            .flat_map(|index| (0..3).map(move |plane| pixel(index, plane)))
            .collect::<Vec<_>>();
        let raw = normal
            .iter()
            .flatten()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let compressed = zstd::bulk::compress(&shuffled(&raw, 16), 3).unwrap();
        let xml = image_element(
            0,
            "2:1:3",
            "Complex64",
            compressed.len(),
            &format!(
                "colorSpace=\"RGB\" pixelStorage=\"Normal\" compression=\"zstd+sh:{}:16\"",
                raw.len()
            ),
            "",
        );
        let image = read_complex_image_from_bytes(&monolithic(xml, &[&compressed]), 0).unwrap();
        let planar = (0..3)
            .flat_map(|plane| (0..2).map(move |index| pixel(index, plane)))
            .collect::<Vec<_>>();
        assert_eq!(image.samples, ComplexSamples::C64(planar));

        let raw = [1_u8, 2, 3, 4];
        let real = monolithic(image_element(0, "2:2:1", "UInt8", 4, "", ""), &[&raw]);
        assert!(matches!(
            read_complex_image_from_bytes(&real, 0),
            Err(XisfError::Unsupported(_))
        ));
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

    #[test]
    fn read_image_reports_the_declared_bounds_beside_the_pixels() {
        let values = [0.0_f32, 0.25, 0.5, 1.0];
        let raw = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let xml = image_element(
            0,
            "2:2:1",
            "Float32",
            raw.len(),
            "bounds=\"0:1\" colorSpace=\"Gray\"",
            "",
        );
        let bytes = monolithic(xml, &[&raw]);
        let read = read_image_from_bytes(&bytes, 0).unwrap();

        assert_eq!(read.info.bounds, Some((0.0, 1.0)));
        assert!(matches!(read.image.pixels, Pixels::F32(ref actual) if actual == &values));
        // The paired read must not change what `open` already returns.
        let opened = from_bytes(&bytes).unwrap();
        assert!(matches!(
            (&opened.pixels, &read.image.pixels),
            (Pixels::F32(opened), Pixels::F32(paired)) if opened == paired
        ));
    }

    #[test]
    fn an_undeclared_range_reads_as_none() {
        let raw = [1_u8, 2, 3, 4];
        let xml = image_element(0, "2:2:1", "UInt8", raw.len(), "colorSpace=\"Gray\"", "");
        let read = read_image_from_bytes(&monolithic(xml, &[&raw]), 0).unwrap();
        assert_eq!(read.info.bounds, None);
    }

    #[test]
    /// Releases before `bounds` existed ignored the attribute outright, so a
    /// spelling this crate cannot read must not cost anyone a file that used
    /// to open.
    fn an_unusable_range_reads_as_none_without_failing_the_file() {
        let raw = [1_u8, 2, 3, 4];
        for bad in [
            "1", "0:", "low:high", "0:nan", "0:inf", "0,1", "",
            "0 1", // Not a range: reversed, or spanning nothing.
            "1:0", "1:1",
        ] {
            let xml = image_element(
                0,
                "2:2:1",
                "UInt8",
                raw.len(),
                &format!("bounds=\"{bad}\" colorSpace=\"Gray\""),
                "",
            );
            let bytes = monolithic(xml, &[&raw]);
            assert!(
                from_bytes(&bytes).is_ok(),
                "bounds {bad:?} must not fail the file"
            );
            assert_eq!(
                read_image_from_bytes(&bytes, 0).unwrap().info.bounds,
                None,
                "bounds {bad:?} should read as none"
            );
        }
    }

    /// A helper for a 2x2 Float32 image with the given `bounds` spelling.
    fn float_frame(values: &[f32], bounds: Option<&str>) -> XisfImage {
        let raw = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let extra = match bounds {
            Some(bounds) => format!("bounds=\"{bounds}\" colorSpace=\"Gray\""),
            None => "colorSpace=\"Gray\"".into(),
        };
        let xml = image_element(0, "2:2:1", "Float32", raw.len(), &extra, "");
        read_image_from_bytes(&monolithic(xml, &[&raw]), 0).unwrap()
    }

    fn samples(image: &XisfImage) -> Vec<f32> {
        match &image.image.pixels {
            Pixels::F32(values) => values.clone(),
            other => panic!("expected float samples, got {other:?}"),
        }
    }

    #[test]
    fn rescale_normalized_to_puts_a_declared_unit_frame_on_a_chosen_scale() {
        let mut read = float_frame(&[0.0, 0.25, 0.5, 1.0], Some("0:1"));

        assert!(read.rescale_normalized_to(65535.0));
        assert_eq!(samples(&read), [0.0, 16383.75, 32767.5, 65535.0]);
        // Bounds describe the new range, so a second call declines.
        assert_eq!(read.info.bounds, Some((0.0, 65535.0)));
        assert!(!read.rescale_normalized_to(65535.0));
        assert_eq!(samples(&read), [0.0, 16383.75, 32767.5, 65535.0]);
    }

    /// The range this crate's own writer emits describes the data, not a
    /// normalization. Converting from it would stretch a physical frame, so
    /// only a declared `0:1` counts.
    #[test]
    fn rescale_normalized_to_declines_a_range_that_is_not_unit() {
        let physical = [100.0_f32, 200.0, 5000.0, 30000.0];
        let mut written = Vec::new();
        write_f32_image_to(
            &mut written,
            2,
            2,
            seiza_fits::F32ImageData::Mono(&physical),
            &[],
        )
        .unwrap();
        let mut read = read_image_from_bytes(&written, 0).unwrap();

        assert_eq!(read.info.bounds, Some((100.0, 30000.0)));
        assert!(!read.rescale_normalized_to(65535.0));
        assert_eq!(samples(&read), physical, "physical samples must survive");

        // A caller who knows the range can still ask for the conversion.
        assert!(read.rescale_from((100.0, 30000.0), 65535.0));
        assert_eq!(samples(&read)[0], 0.0);
        assert_eq!(samples(&read)[3], 65535.0);
    }

    #[test]
    fn rescale_normalized_to_declines_an_absent_or_non_unit_range() {
        for bounds in [None, Some("0:2"), Some("-1:1"), Some("0.5:1")] {
            let values = [0.0_f32, 0.25, 0.5, 1.0];
            let mut read = float_frame(&values, bounds);
            assert!(
                !read.rescale_normalized_to(65535.0),
                "bounds {bounds:?} is not a declared unit range"
            );
            assert_eq!(samples(&read), values);
        }
    }

    #[test]
    fn rescale_from_takes_the_callers_range_whatever_the_file_says() {
        let mut read = float_frame(&[-1.0, 0.0, 1.0, 3.0], None);
        assert!(read.rescale_from((-1.0, 3.0), 100.0));
        assert_eq!(samples(&read), [0.0, 25.0, 50.0, 100.0]);
        assert_eq!(read.info.bounds, Some((0.0, 100.0)));
    }

    /// Clamping would flatten unclipped highlights and negative background
    /// residuals, so the map stays linear past both ends.
    #[test]
    fn rescale_from_maps_linearly_past_the_declared_range() {
        let mut read = float_frame(&[-0.5, 0.0, 1.0, 2.0], Some("0:1"));
        assert!(read.rescale_normalized_to(100.0));
        assert_eq!(samples(&read), [-50.0, 0.0, 100.0, 200.0]);
    }

    #[test]
    fn rescale_declines_what_it_cannot_convert() {
        // A full scale that is not finite and positive would erase the frame.
        for full_scale in [0.0, -100.0, f32::NAN, f32::INFINITY] {
            let values = [0.0_f32, 0.25, 0.5, 1.0];
            let mut read = float_frame(&values, Some("0:1"));
            assert!(
                !read.rescale_normalized_to(full_scale),
                "full scale {full_scale} should be refused"
            );
            assert_eq!(samples(&read), values);
            assert_eq!(read.info.bounds, Some((0.0, 1.0)), "bounds must survive");
        }

        // A source range that spans nothing would divide by zero.
        let values = [0.0_f32, 0.25, 0.5, 1.0];
        let mut read = float_frame(&values, Some("0:1"));
        assert!(!read.rescale_from((1.0, 1.0), 65535.0));
        assert!(!read.rescale_from((3.0, 1.0), 65535.0));
        assert!(!read.rescale_from((f64::NAN, 1.0), 65535.0));
        assert_eq!(samples(&read), values);

        // Integer samples already span their format's range.
        let raw = [1_u8, 2, 3, 4];
        let xml = image_element(
            0,
            "2:2:1",
            "UInt8",
            raw.len(),
            "bounds=\"0:1\" colorSpace=\"Gray\"",
            "",
        );
        let mut read = read_image_from_bytes(&monolithic(xml, &[&raw]), 0).unwrap();
        assert!(!read.rescale_normalized_to(65535.0));
        assert!(!read.rescale_from((0.0, 1.0), 65535.0));
        assert!(matches!(read.image.pixels, Pixels::U8(ref actual) if actual == &raw));
    }

    /// `info` reports what the file preserved; `image` adds the structural
    /// cards this crate synthesizes. Both are deliberate, so pin the split.
    #[test]
    fn the_two_header_lists_differ_in_a_documented_way() {
        let raw = [1_u8, 2, 3, 4];
        let xml = image_element(
            0,
            "2:2:1",
            "UInt8",
            raw.len(),
            "colorSpace=\"Gray\"",
            "<FITSKeyword name=\"EXPTIME\" value=\"300\"/>",
        );
        let read = read_image_from_bytes(&monolithic(xml, &[&raw]), 0).unwrap();

        let names = |headers: &[(String, HeaderValue)]| {
            headers
                .iter()
                .map(|(name, _)| name.clone())
                .collect::<Vec<_>>()
        };
        assert!(names(&read.info.headers).contains(&"EXPTIME".to_string()));
        assert!(
            !names(&read.info.headers).contains(&"NAXIS1".to_string()),
            "info reports what the file preserved, matching inspect"
        );
        assert!(names(&read.image.headers).contains(&"NAXIS1".to_string()));
        assert!(names(&read.image.headers).contains(&"EXPTIME".to_string()));
    }

    #[test]
    fn read_image_selects_by_index_and_id_like_open_does() {
        let first = [1_u8, 2, 3, 4];
        let second = [9_u8, 8, 7, 6];
        let xml = format!(
            "{}{}",
            image_element(0, "2:2:1", "UInt8", 4, "colorSpace=\"Gray\"", ""),
            image_element(1, "2:2:1", "UInt8", 4, "colorSpace=\"Gray\"", "")
        );
        let bytes = monolithic(xml, &[&first, &second]);
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("pair.xisf");
        std::fs::write(&path, &bytes).unwrap();

        assert_eq!(read_image(&path).unwrap().info.index, 0);
        let by_index = read_image_at(&path, 1).unwrap();
        assert_eq!(by_index.info.id.as_deref(), Some("image1"));
        assert!(matches!(by_index.image.pixels, Pixels::U8(ref actual) if actual == &second));
        let by_id = read_image_by_id(&path, "image1").unwrap();
        assert_eq!(by_id.info.index, 1);
        assert!(read_image_by_id(&path, "nope").is_err());
    }
}
