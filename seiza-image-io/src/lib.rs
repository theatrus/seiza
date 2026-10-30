//! Open and save images by their file extension, the same way across Seiza.
//!
//! PNG, JPEG and TIFF go through the `image` crate, upright by their EXIF
//! orientation as [`raster`] reads them; FITS and XISF go through
//! Seiza's own readers and writers. Three ways in:
//!
//! - [`open`] gives any format as floating-point samples. Integer samples
//!   are scaled to 0 to 1 by their type's range, so an 8-bit PNG and a
//!   16-bit FITS read alike; floating-point samples are kept as stored.
//! - [`open_raster`] gives a PNG, JPEG or TIFF at its own depth, for callers
//!   that choose how to treat 8-bit pixels.
//! - [`read_astronomy`] gives a FITS or XISF file as Seiza's [`FitsImage`],
//!   with its headers and stored sample type, and an XISF file's
//!   description and metadata.
//!
//! And three ways out: [`save`] writes floating-point samples, PNG and TIFF
//! at 16 bits and FITS and XISF as 32-bit floats; [`save_f32`] writes
//! borrowed samples with FITS header cards, XISF bounds or a chosen format;
//! [`save_display`] writes a display image, 8 or 16 bits, as PNG, JPEG or
//! TIFF, or as floats in FITS or XISF. Exports replace the destination only
//! after the complete image has been written, flushed and synced.
//!
//! FITS rows are read in their stored order, as the rest of Seiza reads
//! them. [`save`] writes them top-down with a `ROWORDER` card saying so,
//! which PixInsight and Siril follow.

pub mod raster;

use image::{DynamicImage, ImageBuffer, ImageFormat, Luma, Rgb, Rgb32FImage};
use raster::OrientedRaster;
use seiza_core::prior::SolvePrior;
use seiza_fits::{F32ImageData, FitsError, FitsImage, HeaderValue, Pixels, WriteHeaderCard};
pub use seiza_fits::{FitsHeader, ImageWindow, MAX_WINDOW_EDGE};
use seiza_xisf::{XisfError, XisfImageInfo, XisfMetadata};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// An image file's format, as its extension names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Png,
    Jpeg,
    Tiff,
    Fits,
    /// A monolithic `.xisf` file, or a distributed unit's `.xish` header.
    Xisf,
}

impl Format {
    /// The format `path`'s extension names, ignoring case, if it is one of
    /// these.
    pub fn from_path(path: &Path) -> Option<Self> {
        if seiza_xisf::is_xisf_path(path) || seiza_xisf::is_xisf_header_path(path) {
            return Some(Self::Xisf);
        }
        let extension = path.extension()?.to_str()?.to_ascii_lowercase();
        match extension.as_str() {
            "png" => Some(Self::Png),
            "jpg" | "jpeg" => Some(Self::Jpeg),
            "tif" | "tiff" => Some(Self::Tiff),
            "fits" | "fit" | "fts" => Some(Self::Fits),
            _ => None,
        }
    }

    /// The container [`read_astronomy`] reads `path` as: XISF when the
    /// extension says so (`.xisf`, or a distributed unit's `.xish` header),
    /// FITS otherwise, whatever the extension, as a FITS reader of camera
    /// files must.
    pub fn astronomy(path: &Path) -> Self {
        match Self::from_path(path) {
            Some(Self::Xisf) => Self::Xisf,
            _ => Self::Fits,
        }
    }

    /// The format's usual name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
            Self::Tiff => "TIFF",
            Self::Fits => "FITS",
            Self::Xisf => "XISF",
        }
    }

    /// Whether it is an astronomical format, FITS or XISF.
    pub fn is_astronomy(self) -> bool {
        matches!(self, Self::Fits | Self::Xisf)
    }

    /// Whether [`save`] writes it. JPEG is lossy and 8-bit, so only
    /// [`save_display`] does.
    pub fn can_save(self) -> bool {
        !matches!(self, Self::Jpeg)
    }

    /// The `image` crate's format for a raster.
    fn raster(self) -> Option<ImageFormat> {
        match self {
            Self::Png => Some(ImageFormat::Png),
            Self::Jpeg => Some(ImageFormat::Jpeg),
            Self::Tiff => Some(ImageFormat::Tiff),
            Self::Fits | Self::Xisf => None,
        }
    }
}

/// The extensions [`save`] writes, for messages.
pub const SAVE_EXTENSIONS: &str = ".png, .tif, .fits or .xisf";

/// The extensions [`save_display`] writes, for messages.
pub const DISPLAY_EXTENSIONS: &str = ".png, .jpg, .tif, .fits or .xisf";

/// Why an image could not be opened or saved, and the file.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to open {}: {cause}", path.display())]
    Open { path: PathBuf, cause: Cause },
    #[error("failed to write {}: {cause}", path.display())]
    Save { path: PathBuf, cause: Cause },
}

impl Error {
    fn open(path: &Path, cause: impl Into<Cause>) -> Self {
        Self::Open {
            path: path.to_path_buf(),
            cause: cause.into(),
        }
    }

    fn save(path: &Path, cause: impl Into<Cause>) -> Self {
        Self::Save {
            path: path.to_path_buf(),
            cause: cause.into(),
        }
    }

    /// The file it concerns.
    pub fn path(&self) -> &Path {
        match self {
            Self::Open { path, .. } | Self::Save { path, .. } => path,
        }
    }

    /// What went wrong, without the file.
    pub fn cause(&self) -> &Cause {
        match self {
            Self::Open { cause, .. } | Self::Save { cause, .. } => cause,
        }
    }

    /// The file and what went wrong, taken apart.
    pub fn into_parts(self) -> (PathBuf, Cause) {
        match self {
            Self::Open { path, cause } | Self::Save { path, cause } => (path, cause),
        }
    }
}

/// What went wrong, from the reader or writer that found it.
#[derive(Debug, thiserror::Error)]
pub enum Cause {
    #[error(transparent)]
    Fits(#[from] FitsError),
    #[error(transparent)]
    Xisf(#[from] XisfError),
    #[error(transparent)]
    Image(#[from] image::ImageError),
    #[error(transparent)]
    Raster(#[from] raster::RasterError),
    #[error("{0}")]
    Invalid(String),
}

impl From<String> for Cause {
    fn from(message: String) -> Self {
        Self::Invalid(message)
    }
}

impl From<&str> for Cause {
    fn from(message: &str) -> Self {
        Self::Invalid(message.into())
    }
}

/// An image as floating-point samples, row by row from the top, its
/// channels (1 for gray, 3 for red, green and blue) interleaved.
#[derive(Clone, Debug, PartialEq)]
pub struct FloatImage {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub samples: Vec<f32>,
}

impl FloatImage {
    /// A gray image of `samples`, row by row.
    pub fn gray(width: usize, height: usize, samples: Vec<f32>) -> Self {
        Self {
            width,
            height,
            channels: 1,
            samples,
        }
    }

    /// An RGB image's samples.
    pub fn from_rgb(image: &Rgb32FImage) -> Self {
        Self {
            width: image.width() as usize,
            height: image.height() as usize,
            channels: 3,
            samples: image.as_raw().clone(),
        }
    }

    /// A display image's samples, 0 to 1: gray if it has no colour.
    pub fn from_display(image: &DynamicImage) -> Self {
        let (width, height) = (image.width() as usize, image.height() as usize);
        if image.color().has_color() {
            Self::from_rgb(&image.to_rgb32f())
        } else {
            Self::gray(width, height, image.to_luma32f().into_raw())
        }
    }

    /// The image as RGB, a gray one's sample in every channel.
    pub fn to_rgb(&self) -> Rgb32FImage {
        let samples = match self.channels {
            1 => self
                .samples
                .iter()
                .flat_map(|&sample| [sample; 3])
                .collect(),
            _ => self.samples.clone(),
        };
        Rgb32FImage::from_raw(self.width as u32, self.height as u32, samples)
            .expect("an image's samples fill its size")
    }

    /// The image as gray: each pixel's mean channel.
    pub fn to_gray(&self) -> Vec<f32> {
        match self.channels {
            1 => self.samples.clone(),
            channels => self
                .samples
                .chunks_exact(channels)
                .map(|pixel| pixel.iter().sum::<f32>() / channels as f32)
                .collect(),
        }
    }

    /// The samples as the FITS and XISF writers take them.
    pub fn data(&self) -> F32ImageData<'_> {
        if self.channels == 1 {
            F32ImageData::Mono(&self.samples)
        } else {
            F32ImageData::RgbInterleaved(&self.samples)
        }
    }

    /// Whether the samples fill the size, with 1 or 3 channels.
    fn check(&self) -> Result<(), String> {
        if !matches!(self.channels, 1 | 3) {
            return Err(format!(
                "an image has 1 or 3 channels, not {}",
                self.channels
            ));
        }
        check_size(self.width, self.height, self.channels, self.samples.len())
    }
}

/// Whether `samples` fill a `width` × `height` image of `channels`.
fn check_size(width: usize, height: usize, channels: usize, samples: usize) -> Result<(), String> {
    if width == 0 || height == 0 {
        return Err("the image is empty".into());
    }
    let expected = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(channels));
    if expected != Some(samples) {
        return Err(format!(
            "{samples} samples do not fill a {width}×{height} image of {channels} channel{}",
            if channels == 1 { "" } else { "s" }
        ));
    }
    Ok(())
}

/// A FITS or XISF file as Seiza's [`FitsImage`], and for XISF its
/// description and metadata.
#[derive(Debug, Clone)]
pub struct AstronomyImage {
    pub image: FitsImage,
    pub format: Format,
    /// An XISF image's description, such as its declared `bounds`.
    pub xisf_info: Option<XisfImageInfo>,
    /// Everything else an XISF file says about the image, such as a
    /// PixInsight astrometric solution.
    pub xisf_metadata: Option<XisfMetadata>,
}

/// Read a FITS or XISF file: XISF when the extension says so (`.xisf`, or a
/// distributed unit's `.xish` header), FITS otherwise, whatever the
/// extension, as a FITS reader of camera files must.
pub fn read_astronomy(path: &Path) -> Result<AstronomyImage, Error> {
    read_astronomy_decoding(path, None::<fn(usize) -> Vec<f32>>)
}

/// [`read_astronomy`], decoding a FITS camera's 16-bit data straight to
/// physical f32 into the vector `buffer` gives for the sample count, as
/// [`FitsImage::open_with`] does: for a reader that converts every frame to
/// f32 and can hand each the buffer a finished one leaves. Other data, and
/// XISF, read as [`read_astronomy`] reads them, without calling `buffer`.
pub fn read_astronomy_with(
    path: &Path,
    buffer: impl FnOnce(usize) -> Vec<f32>,
) -> Result<AstronomyImage, Error> {
    read_astronomy_decoding(path, Some(buffer))
}

fn read_astronomy_decoding(
    path: &Path,
    physical: Option<impl FnOnce(usize) -> Vec<f32>>,
) -> Result<AstronomyImage, Error> {
    if Format::astronomy(path) == Format::Xisf {
        let read =
            seiza_xisf::read_image_with_options(path, 0, &seiza_xisf::ReadOptions::default())
                .map_err(|error| Error::open(path, error))?;
        return Ok(AstronomyImage {
            image: read.image,
            format: Format::Xisf,
            xisf_info: Some(read.info),
            xisf_metadata: Some(read.metadata),
        });
    }
    let image = match physical {
        Some(buffer) => FitsImage::open_with(path, buffer),
        None => FitsImage::open(path),
    }
    .map_err(|error| Error::open(path, error))?;
    Ok(AstronomyImage {
        image,
        format: Format::Fits,
        xisf_info: None,
        xisf_metadata: None,
    })
}

/// Read the samples of the rectangle at `(x, y)`, `width` × `height`
/// pixels, of every plane of a FITS or XISF image, without decoding the
/// rest: XISF when the extension says so, FITS otherwise, as
/// [`read_astronomy`] chooses.
///
/// The samples are in physical units, as [`FitsImage::into_physical_f32`]
/// gives the whole image: FITS samples with `BZERO`, `BSCALE` and `BLANK`
/// applied, XISF samples as stored. Rows count in stored order. The
/// rectangle is clipped to the image, and a side longer than
/// [`MAX_WINDOW_EDGE`] is refused; see [`seiza_fits::read_window`] and
/// [`seiza_xisf::read_window`] for what each container can window.
///
/// `normalized_full_scale` puts an XISF image the file declares as
/// normalized (`bounds="0:1"`, as PixInsight writes) onto that scale, by the
/// rules of [`seiza_xisf::XisfWindow::rescale_normalized_to`]: 65535.0
/// reads it on a 16-bit camera's scale, as stacking's
/// `normalized_full_scale` does. `None` leaves every sample as stored. A
/// FITS image declares no bounds, so it reads the same either way. A scale
/// that is not finite and positive is refused.
pub fn read_window(
    path: &Path,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    normalized_full_scale: Option<f32>,
) -> Result<ImageWindow, Error> {
    if normalized_full_scale.is_some_and(|scale| !scale.is_finite() || scale <= 0.0) {
        return Err(Error::open(
            path,
            "normalized_full_scale must be finite and positive",
        ));
    }
    if Format::astronomy(path) == Format::Xisf {
        let mut read = seiza_xisf::read_window(path, x, y, width, height)
            .map_err(|error| Error::open(path, error))?;
        if let Some(scale) = normalized_full_scale {
            read.rescale_normalized_to(scale);
        }
        return Ok(read.window);
    }
    seiza_fits::read_window(path, x, y, width, height).map_err(|error| Error::open(path, error))
}

/// Read a FITS or XISF file's header without decoding its pixels: XISF
/// when the extension says so, FITS otherwise, as [`read_astronomy`]
/// chooses.
///
/// The cards are those [`read_astronomy`] gives the image: a FITS file's
/// from the HDU [`FitsImage::open`] reads, with its `HISTORY` and `COMMENT`
/// text, and an XISF file's keywords and structure, with a PixInsight
/// astrometric solution added as WCS cards when they carry none
/// ([`add_xisf_solution_wcs`]). An XISF file keeps its processing history
/// as properties, not commentary, so its `history` and `comments` are
/// empty.
pub fn read_header(path: &Path) -> Result<FitsHeader, Error> {
    read_header_as(path, Format::astronomy(path))
}

/// [`read_header`] for a file whose name does not say its container, such
/// as an upload streamed to a temporary name: the file at `path`, read as
/// `format`. Only FITS and XISF have headers to read.
pub fn read_header_as(path: &Path, format: Format) -> Result<FitsHeader, Error> {
    match format {
        Format::Xisf => {
            let (mut cards, metadata) = seiza_xisf::read_header_and_metadata(path)
                .map_err(|error| Error::open(path, error))?;
            add_xisf_solution_wcs(&mut cards, &metadata);
            Ok(FitsHeader {
                cards,
                ..FitsHeader::default()
            })
        }
        Format::Fits => {
            seiza_fits::read_header_with_commentary(path).map_err(|error| Error::open(path, error))
        }
        _ => Err(Error::open(path, "it is not a FITS or XISF file")),
    }
}

/// Add FITS WCS cards for an XISF image's astrometric solution, when its
/// headers carry no WCS of their own. PixInsight keeps a solution only as
/// `AstrometricSolution` (or, before XISF Revision 1,
/// `PCL:AstrometricSolution`) properties and drops the cards when it saves.
///
/// XISF image coordinates put the top-left corner of the first stored row
/// at (0, 0), so a pixel's centre is at index + 0.5, and rows are read in
/// stored order: `CRPIX` is the reference image point plus 0.5 and `CD` is
/// the linear transformation unchanged. Only a gnomonic solution in the WCS
/// default orientation (reference at native (0, 90), pole at native
/// longitude 180) and ICRS is carried over; its projective and distortion
/// layers, which have no TAN-SIP equivalent, are left out, so positions far
/// from the centre of a distorted field are approximate.
pub fn add_xisf_solution_wcs(headers: &mut Vec<(String, HeaderValue)>, metadata: &XisfMetadata) {
    if headers
        .iter()
        .any(|(key, _)| matches!(key.as_str(), "CTYPE1" | "CTYPE2" | "CRVAL1" | "CRVAL2"))
    {
        return;
    }
    let Ok(Some(solution)) = metadata.astrometric_solution() else {
        return;
    };
    let projection = &solution.projection;
    let pole_default = projection
        .celestial_pole_native
        .is_none_or(|pole| (pole[0] - 180.0).abs() < 1e-9);
    if projection.system != seiza_xisf::ProjectionSystem::Gnomonic
        || projection.reference_native != [0.0, 90.0]
        || !pole_default
        || projection.celestial_reference_system != "ICRS"
    {
        return;
    }
    let [ra, dec] = projection.reference_celestial;
    let [x, y] = projection.reference_image;
    let [[cd11, cd12], [cd21, cd22]] = projection.linear;
    for (key, value) in [
        ("CTYPE1", HeaderValue::String("RA---TAN".into())),
        ("CTYPE2", HeaderValue::String("DEC--TAN".into())),
        ("CRVAL1", HeaderValue::Float(ra)),
        ("CRVAL2", HeaderValue::Float(dec)),
        ("CRPIX1", HeaderValue::Float(x + 0.5)),
        ("CRPIX2", HeaderValue::Float(y + 0.5)),
        ("CD1_1", HeaderValue::Float(cd11)),
        ("CD1_2", HeaderValue::Float(cd12)),
        ("CD2_1", HeaderValue::Float(cd21)),
        ("CD2_2", HeaderValue::Float(cd22)),
        ("RADESYS", HeaderValue::String("ICRS".into())),
    ] {
        headers.push((key.to_owned(), value));
    }
}

impl AstronomyImage {
    /// What the file says about where the image points, for
    /// `seiza::prior::solve_with_prior`: its WCS cards, an XISF
    /// astrometric solution, its pointing (`RA`/`DEC`, `OBJCTRA`/`OBJCTDEC`,
    /// or XISF `Observation:Center`), and its pixel scale (`PIXSCALE`, or
    /// pixel size over focal length, from cards or XISF `Instrument`
    /// properties).
    pub fn solve_prior(&self) -> SolvePrior {
        let mut headers = self.image.headers.clone();
        if let Some(metadata) = &self.xisf_metadata {
            add_xisf_solution_wcs(&mut headers, metadata);
        }
        solve_prior_from_headers(&headers)
    }

    /// Put an XISF image the file declares as normalized (`bounds="0:1"`,
    /// as PixInsight writes) onto `0..=full_scale`, by the rules of
    /// [`seiza_xisf::XisfImage::rescale_normalized_to`]: only an exact
    /// `0:1`, only floating-point samples, and only a finite, positive
    /// `full_scale`. 65535.0 puts it on a 16-bit camera's scale, as
    /// stacking's `normalized_full_scale` does. Answers whether it ran; on
    /// success `xisf_info`'s bounds become `0:full_scale`. A FITS image
    /// declares no bounds and is left alone.
    pub fn rescale_normalized_to(&mut self, full_scale: f32) -> bool {
        self.xisf_info
            .as_mut()
            .is_some_and(|info| seiza_xisf::rescale_normalized(&mut self.image, info, full_scale))
    }
}

/// [`AstronomyImage::solve_prior`] from a FITS or XISF file's headers and
/// metadata alone, without decoding its pixels.
pub fn read_solve_prior(path: &Path) -> Result<SolvePrior, Error> {
    match Format::from_path(path) {
        Some(format @ (Format::Xisf | Format::Fits)) => {
            read_header_as(path, format).map(|header| solve_prior_from_headers(&header.cards))
        }
        _ => Err(Error::open(path, "it is not a FITS or XISF file")),
    }
}

/// [`AstronomyImage::solve_prior`] for header cards alone.
pub fn solve_prior_from_headers(headers: &[(String, HeaderValue)]) -> SolvePrior {
    let card = |key: &str| {
        headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(key))
            .map(|(_, value)| value)
    };
    SolvePrior::from_header(
        |key| card(key).and_then(HeaderValue::as_f64),
        |key| card(key).and_then(HeaderValue::as_str).map(str::to_owned),
    )
}

/// When an image was taken, as its header cards write it: `DATE-OBS`, else
/// `DATE-BEG`, else `DATE-AVG`, as written (ISO 8601, in UTC by the FITS
/// standard). An XISF file's `Observation:Time:Start` reaches the cards as
/// `DATE-OBS`.
pub fn capture_time_from_headers(headers: &[(String, HeaderValue)]) -> Option<String> {
    ["DATE-OBS", "DATE-BEG", "DATE-AVG"]
        .into_iter()
        .find_map(|key| {
            headers
                .iter()
                .find(|(name, _)| name == key)
                .and_then(|(_, value)| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        })
}

/// When the image at `path` was taken: from a FITS or XISF file's header
/// ([`capture_time_from_headers`]), read without decoding its pixels, or
/// from a photo's EXIF ([`raster::PhotoMetadata::capture_time_utc`]).
/// `None` when the file does not say; an error when it cannot be read.
pub fn read_capture_time(path: &Path) -> Result<Option<String>, Error> {
    match Format::from_path(path) {
        Some(format @ (Format::Xisf | Format::Fits)) => {
            read_header_as(path, format).map(|header| capture_time_from_headers(&header.cards))
        }
        _ => {
            std::fs::metadata(path).map_err(|error| Error::open(path, error.to_string()))?;
            Ok(raster::PhotoMetadata::read(path).capture_time_utc)
        }
    }
}

/// The first finite, positive value among `keys`, matched without regard
/// to case, as FITS-derived XISF keywords need not keep FITS casing.
fn first_positive_card(headers: &[(String, HeaderValue)], keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|key| {
        headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case(key))
            .filter_map(|(_, value)| value.as_f64())
            .find(|value| value.is_finite() && *value > 0.0)
    })
}

/// The telescope's focal length in millimetres, from `FOCALLEN`,
/// `FOCALLENGTH` or `FOCAL`: the first that holds a positive number.
pub fn header_focal_length_mm(headers: &[(String, HeaderValue)]) -> Option<f64> {
    first_positive_card(headers, &["FOCALLEN", "FOCALLENGTH", "FOCAL"])
}

/// The pixel size in micrometres, binning included, from `XPIXSZ`.
pub fn header_pixel_size_um(headers: &[(String, HeaderValue)]) -> Option<f64> {
    first_positive_card(headers, &["XPIXSZ"])
}

/// An image's linear 16-bit luminance, for measuring its stars: mono
/// 16-bit samples as stored, without a copy; planar RGB collapsed to
/// luminance; a raw Bayer mosaic debayered first; other sample types mapped
/// to 16 bits as [`FitsImage::to_u16`] maps them.
pub fn luma_u16(image: &FitsImage) -> std::borrow::Cow<'_, [u16]> {
    match image.debayer() {
        Some(rgb) => std::borrow::Cow::Owned(rgb.to_luma_u16()),
        None => image.to_u16(),
    }
}

/// Read a FITS or XISF file as Seiza's [`FitsImage`], with its headers and
/// stored sample type, and its format; an error for any other extension.
pub fn open_astronomy(path: &Path) -> Result<(FitsImage, Format), Error> {
    match Format::from_path(path) {
        Some(Format::Fits) => FitsImage::open(path)
            .map(|image| (image, Format::Fits))
            .map_err(|error| Error::open(path, error)),
        Some(Format::Xisf) => seiza_xisf::open(path)
            .map(|image| (image, Format::Xisf))
            .map_err(|error| Error::open(path, error)),
        _ => Err(Error::open(path, "it is not a FITS or XISF file")),
    }
}

/// Open a PNG, JPEG or TIFF, or any raster the `image` crate knows, at its
/// own depth and upright by its EXIF orientation; an error for FITS or
/// XISF.
pub fn open_raster(path: &Path) -> Result<OrientedRaster, Error> {
    if Format::from_path(path).is_some_and(Format::is_astronomy) {
        return Err(Error::open(
            path,
            "it is a FITS or XISF file, not a PNG, JPEG or TIFF",
        ));
    }
    raster::open_oriented(path).map_err(|error| Error::open(path, error))
}

/// Open the image at `path`, its format chosen by its extension: FITS and
/// XISF by their own readers, anything else as a raster the `image` crate
/// knows, upright by its EXIF orientation.
pub fn open(path: &Path) -> Result<FloatImage, Error> {
    match Format::from_path(path) {
        Some(Format::Fits | Format::Xisf) => {
            let (image, _) = open_astronomy(path)?;
            from_fits(image).map_err(|message| Error::open(path, message))
        }
        _ => Ok(FloatImage::from_display(&open_raster(path)?.pixels)),
    }
}

/// A FITS or XISF image's samples, integer ones scaled to 0 to 1.
fn from_fits(image: FitsImage) -> Result<FloatImage, String> {
    let (width, height, planes) = (image.width, image.height, image.planes);
    if !matches!(planes, 1 | 3) {
        return Err(format!(
            "it has {planes} planes; gray and RGB images have 1 or 3"
        ));
    }
    let planar: Vec<f32> = match image.pixels {
        Pixels::U8(samples) => samples
            .iter()
            .map(|&sample| f32::from(sample) / f32::from(u8::MAX))
            .collect(),
        Pixels::U16(samples) => samples
            .iter()
            .map(|&sample| f32::from(sample) / f32::from(u16::MAX))
            .collect(),
        _ => image.into_physical_f32(),
    };
    let samples = if planes == 1 {
        planar
    } else {
        let plane = width * height;
        (0..plane)
            .flat_map(|index| {
                [
                    planar[index],
                    planar[plane + index],
                    planar[2 * plane + index],
                ]
            })
            .collect()
    };
    Ok(FloatImage {
        width,
        height,
        channels: planes,
        samples,
    })
}

/// How [`save_f32`] writes.
#[derive(Clone, Debug, Default)]
pub struct SaveOptions<'a> {
    /// The format, in place of the one the extension names.
    pub format: Option<Format>,
    /// FITS header cards, written in FITS files and as FITS keywords in
    /// XISF files.
    pub headers: &'a [WriteHeaderCard],
    /// The range an XISF reader shows the samples on; `None` declares 0 to
    /// 1 when the samples lie in it, and nothing otherwise.
    pub xisf_bounds: Option<(f64, f64)>,
    /// Add a `ROWORDER` card saying FITS rows run top-down, unless the
    /// headers carry one.
    pub fits_top_down: bool,
}

/// Write `image` to `path` in the format its extension names: PNG or TIFF
/// at 16 bits, samples clamped to 0 to 1, or FITS or XISF as 32-bit
/// floats, as they are. FITS rows are declared top-down.
pub fn save(path: &Path, image: &FloatImage) -> Result<(), Error> {
    image
        .check()
        .map_err(|message| Error::save(path, message))?;
    save_f32(
        path,
        image.width,
        image.height,
        image.data(),
        &SaveOptions {
            fits_top_down: true,
            ..SaveOptions::default()
        },
    )
}

/// Write borrowed samples, gray or RGB, to `path` in the format `options`
/// or the extension names: PNG or TIFF at 16 bits, samples clamped to 0 to
/// 1, or FITS or XISF as 32-bit floats with `options`' header cards. FITS
/// and XISF files are written atomically.
pub fn save_f32(
    path: &Path,
    width: usize,
    height: usize,
    pixels: F32ImageData<'_>,
    options: &SaveOptions<'_>,
) -> Result<(), Error> {
    let samples = match pixels {
        F32ImageData::Mono(samples)
        | F32ImageData::RgbInterleaved(samples)
        | F32ImageData::RgbPlanar(samples) => samples,
    };
    check_size(width, height, pixels.planes(), samples.len())
        .map_err(|message| Error::save(path, message))?;
    let format = options
        .format
        .or_else(|| Format::from_path(path))
        .filter(|format| format.can_save())
        .ok_or_else(|| Error::save(path, format!("name it {SAVE_EXTENSIONS}")))?;
    match format {
        Format::Png | Format::Tiff => {
            let (width, height) = (
                u32::try_from(width).map_err(|_| Error::save(path, "it is too wide"))?,
                u32::try_from(height).map_err(|_| Error::save(path, "it is too tall"))?,
            );
            let sixteen =
                |sample: f32| (sample.clamp(0.0, 1.0) * f32::from(u16::MAX)).round() as u16;
            let image = match pixels {
                F32ImageData::Mono(samples) => DynamicImage::ImageLuma16(
                    ImageBuffer::<Luma<u16>, _>::from_raw(
                        width,
                        height,
                        samples.iter().copied().map(sixteen).collect(),
                    )
                    .expect("checked size"),
                ),
                F32ImageData::RgbInterleaved(samples) => DynamicImage::ImageRgb16(
                    ImageBuffer::<Rgb<u16>, _>::from_raw(
                        width,
                        height,
                        samples.iter().copied().map(sixteen).collect(),
                    )
                    .expect("checked size"),
                ),
                F32ImageData::RgbPlanar(samples) => {
                    let plane = samples.len() / 3;
                    let interleaved = (0..plane)
                        .flat_map(|index| {
                            [
                                samples[index],
                                samples[plane + index],
                                samples[2 * plane + index],
                            ]
                        })
                        .map(sixteen)
                        .collect();
                    DynamicImage::ImageRgb16(
                        ImageBuffer::<Rgb<u16>, _>::from_raw(width, height, interleaved)
                            .expect("checked size"),
                    )
                }
            };
            save_raster(path, &image, format.raster().expect("a raster format"))
        }
        Format::Fits => {
            let mut headers = options.headers.to_vec();
            if options.fits_top_down && !headers.iter().any(|card| card.keyword() == "ROWORDER") {
                headers.push(
                    WriteHeaderCard::new("ROWORDER", HeaderValue::String("TOP-DOWN".into()))
                        .with_comment("first row is the top of the image"),
                );
            }
            seiza_fits::write_f32_image(path, width, height, pixels, &headers)
                .map_err(|error| Error::save(path, error))
        }
        Format::Xisf => {
            if seiza_xisf::is_xisf_header_path(path) {
                return Err(Error::save(
                    path,
                    XisfError::Unsupported(
                        ".xish names a distributed unit's header file; write a .xisf file".into(),
                    ),
                ));
            }
            let bounds = options.xisf_bounds.or_else(|| {
                samples
                    .iter()
                    .all(|sample| (0.0..=1.0).contains(sample))
                    .then_some((0.0, 1.0))
            });
            let xisf = seiza_xisf::WriteOptions {
                bounds,
                ..seiza_xisf::WriteOptions::default()
            };
            seiza_xisf::write_f32_image_with_options(
                path,
                width,
                height,
                pixels,
                options.headers,
                &xisf,
            )
            .map_err(|error| Error::save(path, error))
        }
        Format::Jpeg => unreachable!("filtered out above"),
    }
}

/// Write a display image, such as a preview or an annotated photo, in the
/// format its extension names: PNG or TIFF at the image's own depth, JPEG
/// at 8 bits, or FITS or XISF as 32-bit floats from 0 to 1.
pub fn save_display(path: &Path, image: &DynamicImage) -> Result<(), Error> {
    let format = Format::from_path(path)
        .ok_or_else(|| Error::save(path, format!("name it {DISPLAY_EXTENSIONS}")))?;
    match format.raster() {
        Some(ImageFormat::Jpeg) => {
            // JPEG holds 8-bit gray or colour, without alpha.
            let eight = if image.color().has_color() {
                DynamicImage::ImageRgb8(image.to_rgb8())
            } else {
                DynamicImage::ImageLuma8(image.to_luma8())
            };
            save_raster(path, &eight, ImageFormat::Jpeg)
        }
        Some(raster) => save_raster(path, image, raster),
        None => save(path, &FloatImage::from_display(image)),
    }
}

// Keep the destination intact until encoding, flushing and syncing all succeed.
fn save_raster(path: &Path, image: &DynamicImage, format: ImageFormat) -> Result<(), Error> {
    write_raster_atomic(path, |writer| image.write_to(writer, format))
}

fn write_raster_atomic(
    path: &Path,
    encode: impl FnOnce(&mut BufWriter<&mut std::fs::File>) -> image::ImageResult<()>,
) -> Result<(), Error> {
    let write = || -> image::ImageResult<()> {
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
        encode(&mut writer)?;
        writer.flush()?;
        drop(writer);
        temporary.as_file().sync_all()?;
        temporary.persist(path).map_err(|error| error.error)?;
        Ok(())
    };
    write().map_err(|error| Error::save(path, error))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_reads_fits_and_xisf_alike() {
        let directory = tempfile::tempdir().unwrap();
        let (width, height) = (9, 7);
        let gray = FloatImage::gray(
            width,
            height,
            (0..width * height)
                .map(|index| index as f32 / 10.0)
                .collect(),
        );
        let rgb = FloatImage {
            width,
            height,
            channels: 3,
            samples: (0..width * height * 3).map(|index| index as f32).collect(),
        };
        for (name, image) in [("gray", &gray), ("rgb", &rgb)] {
            for extension in ["fits", "xisf"] {
                let path = directory.path().join(format!("{name}.{extension}"));
                save(&path, image).unwrap();
                let window = read_window(&path, 2, 3, 4, 2, None).unwrap();
                assert_eq!(
                    (
                        window.x,
                        window.y,
                        window.width,
                        window.height,
                        window.planes
                    ),
                    (2, 3, 4, 2, image.channels),
                    "{name}.{extension}"
                );
                // The saved samples are interleaved; the window's are planar.
                for plane in 0..image.channels {
                    for y in 3..5 {
                        for x in 2..6 {
                            assert_eq!(
                                window.get(plane, x, y),
                                Some(image.samples[(y * width + x) * image.channels + plane]),
                                "{name}.{extension} plane {plane} ({x}, {y})"
                            );
                        }
                    }
                }
            }
        }
        let missing = directory.path().join("missing.fits");
        assert_eq!(
            read_window(&missing, 0, 0, 1, 1, None).unwrap_err().path(),
            missing
        );
    }

    /// A monolithic 2x2 gray float XISF of `samples`, with `extra` image
    /// attributes and `properties` inside its Image element.
    fn xisf_file(path: &Path, samples: [f32; 4], extra: &str, properties: &str) {
        let pixels: Vec<u8> = samples
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        // The attachment's offset is written in the header it follows.
        let mut offset = 0;
        loop {
            let header = format!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?><xisf version=\"1.0\" xmlns=\"http://www.pixinsight.com/xisf\">\
                 <Image geometry=\"2:2:1\" sampleFormat=\"Float32\" colorSpace=\"Gray\" {extra} \
                 location=\"attachment:{offset}:{}\">{properties}</Image></xisf>",
                pixels.len()
            );
            let start = 16 + header.len();
            if start == offset {
                let mut bytes = b"XISF0100".to_vec();
                bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
                bytes.extend_from_slice(&[0; 4]);
                bytes.extend_from_slice(header.as_bytes());
                bytes.extend_from_slice(&pixels);
                std::fs::write(path, bytes).unwrap();
                return;
            }
            offset = start;
        }
    }

    /// PixInsight's gnomonic solution, as properties: reference at RA 83.8,
    /// Dec -5.4 on image point (100.5, 80.5).
    fn pixinsight_solution() -> String {
        let vector = |id: &str, dimensions: &str, values: &[f64]| {
            let kind = if dimensions.starts_with("rows") {
                "F64Matrix"
            } else {
                "F64Vector"
            };
            let hex: String = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .map(|byte| format!("{byte:02x}"))
                .collect();
            format!(
                "<Property id=\"AstrometricSolution:{id}\" type=\"{kind}\" {dimensions} location=\"inline:hex\">{hex}</Property>"
            )
        };
        let text = |id: &str, value: &str| {
            format!("<Property id=\"AstrometricSolution:{id}\" type=\"String\">{value}</Property>")
        };
        [
            text("Version", "1.3"),
            text("ProjectionSystem", "Gnomonic"),
            vector(
                "ReferenceCelestialCoordinates",
                "length=\"2\"",
                &[83.8, -5.4],
            ),
            vector("ReferenceImageCoordinates", "length=\"2\"", &[100.5, 80.5]),
            vector(
                "LinearTransformationMatrix",
                "rows=\"2\" columns=\"2\"",
                &[-2e-4, 0.0, 0.0, 2e-4],
            ),
        ]
        .concat()
    }

    #[test]
    fn a_header_is_read_with_its_commentary_and_an_xisf_solution() {
        let directory = tempfile::tempdir().unwrap();
        let fits = directory.path().join("light.fits");
        let mut header = Vec::new();
        for card in [
            "SIMPLE  =                    T".to_string(),
            "BITPIX  =                  -32".to_string(),
            "NAXIS   =                    2".to_string(),
            "NAXIS1  =                    2".to_string(),
            "NAXIS2  =                    2".to_string(),
            "EXPTIME =                 30.0".to_string(),
            "HISTORY Calibrated with a master dark".to_string(),
            "COMMENT Written by a test".to_string(),
            "END".to_string(),
        ] {
            header.extend_from_slice(format!("{card:<80}").as_bytes());
        }
        header.resize(2880 * 2, 0);
        std::fs::write(&fits, header).unwrap();

        let read = read_header(&fits).unwrap();
        assert_eq!(
            read.cards
                .iter()
                .find(|(key, _)| key == "EXPTIME")
                .map(|(_, value)| value.as_f64()),
            Some(Some(30.0))
        );
        assert_eq!(read.history, ["Calibrated with a master dark"]);
        assert_eq!(read.comments, ["Written by a test"]);

        // PixInsight keeps a solution only as properties; the header gains
        // its WCS cards, as solving reads them.
        let xisf = directory.path().join("solved.xisf");
        xisf_file(&xisf, [0.0; 4], "", &pixinsight_solution());
        let read = read_header(&xisf).unwrap();
        let card = |key: &str| {
            read.cards
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.clone())
        };
        assert_eq!(card("CTYPE1"), Some(HeaderValue::String("RA---TAN".into())));
        assert_eq!(card("CRVAL1"), Some(HeaderValue::Float(83.8)));
        assert_eq!(card("CRPIX1"), Some(HeaderValue::Float(101.0)));
        assert!(read.history.is_empty() && read.comments.is_empty());
        assert_eq!(
            read_solve_prior(&xisf).unwrap().wcs.unwrap().crval,
            (83.8, -5.4)
        );

        // XISF by its extension, FITS otherwise.
        for (name, format) in [
            ("a.XISF", Format::Xisf),
            ("unit.xish", Format::Xisf),
            ("a.fts", Format::Fits),
            ("upload.part", Format::Fits),
            ("photo.png", Format::Fits),
        ] {
            assert_eq!(Format::astronomy(Path::new(name)), format, "{name}");
        }

        // A file whose name does not say its container is read as asked.
        let upload = directory.path().join("upload.part");
        std::fs::copy(&xisf, &upload).unwrap();
        assert_eq!(read_header_as(&upload, Format::Xisf).unwrap(), read);
        assert!(read_header(&upload).is_err(), "read as FITS by its name");
        assert!(read_header_as(&upload, Format::Png).is_err());
    }

    #[test]
    fn a_whole_normalized_xisf_is_put_on_the_scale_asked_for() {
        let directory = tempfile::tempdir().unwrap();
        let normalized = directory.path().join("normalized.xisf");
        xisf_file(&normalized, [0.0, 0.25, 0.5, 1.0], "bounds=\"0:1\"", "");
        let mut read = read_astronomy(&normalized).unwrap();
        assert!(read.rescale_normalized_to(65535.0));
        assert_eq!(
            read.image.clone().into_physical_f32(),
            [0.0, 16383.75, 32767.5, 65535.0]
        );
        assert_eq!(
            read.xisf_info.as_ref().unwrap().bounds,
            Some((0.0, 65535.0))
        );
        // As the window of the same file reads it, and only once.
        assert_eq!(
            read_window(&normalized, 0, 0, 2, 2, Some(65535.0))
                .unwrap()
                .values,
            read.image.into_physical_f32()
        );

        // Other bounds, and FITS, are left alone.
        let physical = directory.path().join("physical.xisf");
        xisf_file(&physical, [0.0, 0.25, 0.5, 1.0], "bounds=\"0:30000\"", "");
        assert!(
            !read_astronomy(&physical)
                .unwrap()
                .rescale_normalized_to(65535.0)
        );
        let fits = directory.path().join("normalized.fits");
        save(&fits, &FloatImage::gray(2, 2, vec![0.0, 0.25, 0.5, 1.0])).unwrap();
        assert!(
            !read_astronomy(&fits)
                .unwrap()
                .rescale_normalized_to(65535.0)
        );
    }

    #[test]
    fn a_window_puts_a_normalized_xisf_on_the_scale_asked_for() {
        let directory = tempfile::tempdir().unwrap();
        // Samples within 0 to 1, which save declares as bounds="0:1".
        let normalized = FloatImage::gray(4, 3, (0..12).map(|index| index as f32 / 11.0).collect());
        let counts = FloatImage::gray(4, 3, (0..12).map(|index| index as f32 * 100.0).collect());
        let xisf = directory.path().join("normalized.xisf");
        let fits = directory.path().join("normalized.fits");
        let camera = directory.path().join("counts.xisf");
        save(&xisf, &normalized).unwrap();
        save(&fits, &normalized).unwrap();
        save(&camera, &counts).unwrap();

        // The samples a whole image rescaled to that scale has.
        let mut whole = seiza_xisf::read_image(&xisf).unwrap();
        assert!(whole.rescale_normalized_to(65535.0));
        let scaled = read_window(&xisf, 1, 1, 3, 2, Some(65535.0)).unwrap();
        let physical = whole.image.into_physical_f32();
        let expected: Vec<f32> = (1..3)
            .flat_map(|y| (1..4).map(move |x| y * 4 + x))
            .map(|index| physical[index])
            .collect();
        assert_eq!(scaled.values, expected);

        // As stored without a scale; a FITS file and an XISF on a camera's
        // scale declare no 0:1, so they read as stored either way.
        let whole_window =
            |path: &Path, scale| read_window(path, 0, 0, 4, 3, scale).unwrap().values;
        assert_eq!(whole_window(&xisf, None), normalized.samples);
        assert_eq!(whole_window(&fits, Some(65535.0)), normalized.samples);
        assert_eq!(whole_window(&camera, Some(65535.0)), counts.samples);

        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let error = read_window(&xisf, 0, 0, 1, 1, Some(scale)).unwrap_err();
            assert!(
                matches!(error.cause(), Cause::Invalid(_)),
                "{scale}: {error}"
            );
        }
    }

    #[test]
    fn failed_raster_writes_preserve_existing_files() {
        for extension in ["png", "tiff", "jpg"] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join(format!("existing.{extension}"));
            let original = b"previous complete file";
            std::fs::write(&path, original).unwrap();
            let result = write_raster_atomic(&path, |writer| {
                // Flush partial output to disk before failing, rather than merely
                // failing validation before the destination is opened.
                writer.write_all(b"partial encoded image")?;
                writer.flush()?;
                Err(std::io::Error::other("injected encoding failure").into())
            });
            assert!(result.is_err());
            assert_eq!(std::fs::read(&path).unwrap(), original);
            assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
        }
    }

    #[cfg(unix)]
    #[test]
    fn atomic_raster_replacement_preserves_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("existing.png");
        std::fs::write(&path, b"old destination").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
        save_display(&path, &DynamicImage::new_rgb8(2, 2)).unwrap();
        assert_eq!(
            std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o640
        );
    }

    #[test]
    fn failed_raster_publication_cleans_up_temporary_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("occupied.png");
        std::fs::create_dir(&path).unwrap();
        let sentinel = path.join("keep");
        std::fs::write(&sentinel, b"keep").unwrap();
        let image = DynamicImage::new_rgb8(2, 2);
        assert!(save_display(&path, &image).is_err());
        assert_eq!(std::fs::read(sentinel).unwrap(), b"keep");
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[test]
    fn atomic_display_exports_match_direct_encoding() {
        let directory = tempfile::tempdir().unwrap();
        let images = [
            DynamicImage::ImageRgb8(ImageBuffer::from_fn(7, 5, |x, y| {
                Rgb([(x * 31) as u8, (y * 47) as u8, 123])
            })),
            DynamicImage::ImageLuma16(ImageBuffer::from_fn(7, 5, |x, y| {
                Luma([(x * 1234 + y * 4321) as u16])
            })),
        ];
        for image in images {
            for (extension, format) in [
                ("png", ImageFormat::Png),
                ("tiff", ImageFormat::Tiff),
                ("jpg", ImageFormat::Jpeg),
            ] {
                let path = directory.path().join(format!("atomic.{extension}"));
                let direct = directory.path().join(format!("direct.{extension}"));
                let expected = if format == ImageFormat::Jpeg {
                    if image.color().has_color() {
                        DynamicImage::ImageRgb8(image.to_rgb8())
                    } else {
                        DynamicImage::ImageLuma8(image.to_luma8())
                    }
                } else {
                    image.clone()
                };
                expected.save_with_format(&direct, format).unwrap();
                std::fs::write(&path, b"old destination").unwrap();
                save_display(&path, &image).unwrap();
                assert_eq!(
                    std::fs::read(&path).unwrap(),
                    std::fs::read(&direct).unwrap()
                );
            }
        }
    }

    #[test]
    fn atomic_float_exports_match_direct_encoding() {
        let directory = tempfile::tempdir().unwrap();
        let samples = [-1.0, 0.0, 0.25, 0.5, 1.0, 2.0];
        let expected = DynamicImage::ImageLuma16(
            ImageBuffer::from_raw(3, 2, vec![0, 0, 16384, 32768, 65535, 65535]).unwrap(),
        );
        for (extension, format) in [("png", ImageFormat::Png), ("tiff", ImageFormat::Tiff)] {
            let path = directory.path().join(format!("atomic.{extension}"));
            let direct = directory.path().join(format!("direct.{extension}"));
            expected.save_with_format(&direct, format).unwrap();
            std::fs::write(&path, b"old destination").unwrap();
            save_f32(
                &path,
                3,
                2,
                F32ImageData::Mono(&samples),
                &SaveOptions::default(),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&path).unwrap(),
                std::fs::read(&direct).unwrap()
            );
        }
    }

    #[test]
    fn capture_time_and_optics_come_from_the_header() {
        let directory = tempfile::tempdir().unwrap();
        let card = |key: &str, value: HeaderValue| WriteHeaderCard::new(key, value);
        let pixels = [0.0_f32; 4];
        let fits = directory.path().join("dated.fits");
        seiza_fits::write_f32_image(
            &fits,
            2,
            2,
            F32ImageData::Mono(&pixels),
            &[
                card(
                    "DATE-BEG",
                    HeaderValue::String("2025-07-20T12:00:00".into()),
                ),
                card(
                    "DATE-OBS",
                    HeaderValue::String(" 2025-07-20T12:34:56.5Z ".into()),
                ),
            ],
        )
        .unwrap();
        assert_eq!(
            read_capture_time(&fits).unwrap().as_deref(),
            Some("2025-07-20T12:34:56.5Z")
        );

        let xisf = directory.path().join("dated.xisf");
        seiza_xisf::write_f32_image(
            &xisf,
            2,
            2,
            F32ImageData::Mono(&pixels),
            &[card(
                "DATE-AVG",
                HeaderValue::String("2026-01-02T03:04:05".into()),
            )],
        )
        .unwrap();
        assert_eq!(
            read_capture_time(&xisf).unwrap().as_deref(),
            Some("2026-01-02T03:04:05")
        );

        let undated = directory.path().join("undated.fits");
        seiza_fits::write_f32_image(&undated, 2, 2, F32ImageData::Mono(&pixels), &[]).unwrap();
        assert_eq!(read_capture_time(&undated).unwrap(), None);
        assert!(read_capture_time(&directory.path().join("missing.jpg")).is_err());

        let headers = vec![
            ("FOCALLEN".to_string(), HeaderValue::Float(-1.0)),
            ("focallength".to_string(), HeaderValue::Float(2000.0)),
            ("xpixsz".to_string(), HeaderValue::Float(3.76)),
        ];
        assert_eq!(header_focal_length_mm(&headers), Some(2000.0));
        assert_eq!(header_pixel_size_um(&headers), Some(3.76));
        assert_eq!(header_pixel_size_um(&[]), None);
    }

    #[test]
    fn a_solve_prior_is_read_from_headers_without_pixels() {
        let directory = tempfile::tempdir().unwrap();
        let card = |key: &str, value: HeaderValue| WriteHeaderCard::new(key, value);
        let pixels = [0.0_f32; 4];

        // A FITS mount position and optics scale.
        let fits = directory.path().join("light.fits");
        seiza_fits::write_f32_image(
            &fits,
            2,
            2,
            F32ImageData::Mono(&pixels),
            &[
                card("OBJCTRA", HeaderValue::String("05 35 17.3".into())),
                card("OBJCTDEC", HeaderValue::String("-05 23 28".into())),
                card("XPIXSZ", HeaderValue::Float(3.76)),
                card("FOCALLEN", HeaderValue::Float(530.0)),
            ],
        )
        .unwrap();
        let prior = read_solve_prior(&fits).unwrap();
        assert!(prior.wcs.is_none());
        assert!((prior.center.unwrap().0 - 83.822).abs() < 1e-3);
        assert!((prior.scale_arcsec_px.unwrap() - 1.4633).abs() < 1e-3);

        // An XISF carrying a WCS as FITS keywords.
        let xisf = directory.path().join("solved.xisf");
        seiza_xisf::write_f32_image(
            &xisf,
            2,
            2,
            F32ImageData::Mono(&pixels),
            &[
                card("CTYPE1", HeaderValue::String("RA---TAN".into())),
                card("CTYPE2", HeaderValue::String("DEC--TAN".into())),
                card("CRVAL1", HeaderValue::Float(150.5)),
                card("CRVAL2", HeaderValue::Float(33.2)),
                card("CRPIX1", HeaderValue::Float(1.5)),
                card("CRPIX2", HeaderValue::Float(1.5)),
                card("CD1_1", HeaderValue::Float(-5e-4)),
                card("CD1_2", HeaderValue::Float(0.0)),
                card("CD2_1", HeaderValue::Float(0.0)),
                card("CD2_2", HeaderValue::Float(5e-4)),
            ],
        )
        .unwrap();
        let prior = read_solve_prior(&xisf).unwrap();
        assert_eq!(prior.wcs.as_ref().unwrap().crval, (150.5, 33.2));
        assert!((prior.scale_arcsec_px.unwrap() - 1.8).abs() < 1e-9);
        // The same as reading the whole file.
        assert_eq!(read_astronomy(&xisf).unwrap().solve_prior(), prior);

        assert!(read_solve_prior(&directory.path().join("photo.jpg")).is_err());
    }

    /// A gray ramp across, and an RGB image whose channels differ.
    fn gray() -> FloatImage {
        FloatImage::gray(
            5,
            3,
            (0..15).map(|index| (index % 5) as f32 / 4.0).collect(),
        )
    }

    fn rgb() -> FloatImage {
        FloatImage {
            width: 4,
            height: 2,
            channels: 3,
            samples: (0..24).map(|index| index as f32 / 23.0).collect(),
        }
    }

    #[test]
    fn formats_come_from_the_extension_whatever_its_case() {
        for (name, format) in [
            ("a.png", Some(Format::Png)),
            ("a.JPG", Some(Format::Jpeg)),
            ("a.jpeg", Some(Format::Jpeg)),
            ("a.tif", Some(Format::Tiff)),
            ("a.TIFF", Some(Format::Tiff)),
            ("a.fits", Some(Format::Fits)),
            ("a.fit", Some(Format::Fits)),
            ("a.FTS", Some(Format::Fits)),
            ("a.xisf", Some(Format::Xisf)),
            ("a.xish", Some(Format::Xisf)),
            ("a.webp", None),
            ("a", None),
        ] {
            assert_eq!(Format::from_path(Path::new(name)), format, "{name}");
        }
        assert!(Format::Fits.is_astronomy() && !Format::Tiff.is_astronomy());
        assert!(!Format::Jpeg.can_save());
    }

    #[test]
    fn every_saved_format_reads_back_the_same() {
        let directory = tempfile::tempdir().unwrap();
        for image in [gray(), rgb()] {
            for extension in ["png", "tif", "fits", "xisf"] {
                let path = directory
                    .path()
                    .join(format!("image-{}.{extension}", image.channels));
                save(&path, &image).unwrap();
                let read = open(&path).unwrap();
                assert_eq!(
                    (read.width, read.height, read.channels),
                    (image.width, image.height, image.channels),
                    "{extension}"
                );
                // 16 bits for PNG and TIFF, exact for the float formats.
                let tolerance = if matches!(extension, "png" | "tif") {
                    1.0 / 65535.0
                } else {
                    0.0
                };
                for (read, written) in read.samples.iter().zip(&image.samples) {
                    assert!(
                        (read - written).abs() <= tolerance,
                        "{extension}: {read} for {written}"
                    );
                }
            }
        }
    }

    #[test]
    fn fits_says_its_rows_run_top_down() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("rows.fits");
        save(&path, &gray()).unwrap();
        let (image, format) = open_astronomy(&path).unwrap();
        assert_eq!(format, Format::Fits);
        assert_eq!(image.header_str("ROWORDER"), Some("TOP-DOWN"));
        // The first stored row is the image's top row.
        let read = open(&path).unwrap();
        assert_eq!(&read.samples[..5], &gray().samples[..5]);
    }

    #[test]
    fn borrowed_samples_keep_their_cards_bounds_and_planes() {
        let directory = tempfile::tempdir().unwrap();
        let counts: Vec<f32> = (0..15).map(|index| index as f32 * 1000.0).collect();
        let cards = [WriteHeaderCard::new("EXPTIME", HeaderValue::Float(30.0))];
        let options = SaveOptions {
            headers: &cards,
            xisf_bounds: Some((0.0, 65535.0)),
            ..SaveOptions::default()
        };
        // No ROWORDER unless asked, and counts kept as counts.
        let fits = directory.path().join("counts.fits");
        save_f32(&fits, 5, 3, F32ImageData::Mono(&counts), &options).unwrap();
        let read = read_astronomy(&fits).unwrap();
        assert_eq!(read.format, Format::Fits);
        assert_eq!(read.image.header_f64("EXPTIME"), Some(30.0));
        assert_eq!(read.image.header_str("ROWORDER"), None);
        assert_eq!(read.image.into_physical_f32(), counts);
        let xisf = directory.path().join("counts.xisf");
        save_f32(&xisf, 5, 3, F32ImageData::Mono(&counts), &options).unwrap();
        let read = read_astronomy(&xisf).unwrap();
        assert_eq!(read.format, Format::Xisf);
        assert_eq!(read.xisf_info.unwrap().bounds, Some((0.0, 65535.0)));
        assert!(read.xisf_metadata.is_some());
        // A chosen format wins over the extension.
        let other = directory.path().join("counts.dat");
        let fits_anyway = SaveOptions {
            format: Some(Format::Fits),
            ..options.clone()
        };
        save_f32(&other, 5, 3, F32ImageData::Mono(&counts), &fits_anyway).unwrap();
        assert_eq!(read_astronomy(&other).unwrap().format, Format::Fits);
        // Planar RGB reads back interleaved.
        let planar: Vec<f32> = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6].to_vec();
        let png = directory.path().join("planar.png");
        save_f32(
            &png,
            2,
            1,
            F32ImageData::RgbPlanar(&planar),
            &SaveOptions::default(),
        )
        .unwrap();
        let read = open(&png).unwrap();
        let expected = [0.1, 0.3, 0.5, 0.2, 0.4, 0.6];
        for (read, expected) in read.samples.iter().zip(expected) {
            assert!((read - expected).abs() < 1e-4, "{read} for {expected}");
        }
        // A distributed unit's header is not a file to write pixels to.
        let error = save_f32(
            &directory.path().join("unit.xish"),
            5,
            3,
            F32ImageData::Mono(&counts),
            &options,
        )
        .unwrap_err();
        assert!(matches!(
            error.cause(),
            Cause::Xisf(XisfError::Unsupported(_))
        ));
    }

    #[test]
    fn display_images_go_out_as_their_extension_says() {
        let directory = tempfile::tempdir().unwrap();
        let photo = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(3, 2, Rgb([255, 0, 51])));
        for extension in ["png", "jpg", "tif", "fits", "xisf"] {
            let path = directory.path().join(format!("photo.{extension}"));
            save_display(&path, &photo).unwrap();
            let read = open(&path).unwrap();
            assert_eq!(
                (read.width, read.height, read.channels),
                (3, 2, 3),
                "{extension}"
            );
            // JPEG is lossy; the rest keep the 8-bit values.
            let tolerance = if extension == "jpg" { 0.05 } else { 1e-6 };
            for (read, expected) in read.samples[..3].iter().zip([1.0, 0.0, 0.2]) {
                assert!(
                    (read - expected).abs() <= tolerance,
                    "{extension}: {read} for {expected}"
                );
            }
        }
        // A 16-bit gray preview stays 16-bit in PNG and becomes 8-bit JPEG.
        let deep = DynamicImage::ImageLuma16(ImageBuffer::from_pixel(2, 2, Luma([40_000_u16])));
        let png = directory.path().join("deep.png");
        save_display(&png, &deep).unwrap();
        assert_eq!(image::open(&png).unwrap().color(), image::ColorType::L16);
        let jpeg = directory.path().join("deep.jpg");
        save_display(&jpeg, &deep).unwrap();
        assert_eq!(image::open(&jpeg).unwrap().color(), image::ColorType::L8);
        let error = save_display(&directory.path().join("photo.bmp"), &photo).unwrap_err();
        assert!(error.to_string().contains(DISPLAY_EXTENSIONS), "{error}");
    }

    #[test]
    fn an_eight_bit_png_reads_as_zero_to_one() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("eight.png");
        image::GrayImage::from_raw(2, 1, vec![0, 255])
            .unwrap()
            .save(&path)
            .unwrap();
        let read = open(&path).unwrap();
        assert_eq!(read.channels, 1);
        assert_eq!(read.samples, vec![0.0, 1.0]);
        let colour = directory.path().join("colour.png");
        image::RgbImage::from_raw(1, 1, vec![255, 0, 51])
            .unwrap()
            .save(&colour)
            .unwrap();
        let read = open(&colour).unwrap();
        assert_eq!(read.channels, 3);
        assert_eq!(read.samples, vec![1.0, 0.0, 0.2]);
        assert_eq!(read.to_gray(), vec![0.4]);
        assert_eq!(read.to_rgb().get_pixel(0, 0).0, [1.0, 0.0, 0.2]);
        // The raster at its own depth, and not a FITS file.
        assert_eq!(
            open_raster(&path).unwrap().pixels.color(),
            image::ColorType::L8
        );
        let fits = directory.path().join("image.fits");
        save(&fits, &gray()).unwrap();
        assert!(open_raster(&fits).is_err());
    }

    #[test]
    fn a_jpeg_an_unknown_extension_or_a_short_image_is_not_written() {
        let directory = tempfile::tempdir().unwrap();
        for name in ["lossy.jpg", "unknown.bmp", "none"] {
            let error = save(&directory.path().join(name), &gray()).unwrap_err();
            assert!(error.to_string().contains(SAVE_EXTENSIONS), "{error}");
        }
        let mut short = gray();
        short.samples.pop();
        let error = save(&directory.path().join("short.png"), &short).unwrap_err();
        assert!(error.to_string().contains("do not fill"), "{error}");
        let error = open_astronomy(&directory.path().join("image.png")).unwrap_err();
        assert!(error.to_string().contains("not a FITS or XISF"), "{error}");
        assert_eq!(error.path(), directory.path().join("image.png"));
    }
}
