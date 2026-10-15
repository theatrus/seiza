//! Open and save images by their file extension, the same way across Seiza.
//!
//! PNG, JPEG and TIFF go through the `image` crate, upright by their EXIF
//! orientation as [`seiza::raster`] reads them; FITS and XISF go through
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
//! TIFF, or as floats in FITS or XISF.
//!
//! FITS rows are read in their stored order, as the rest of Seiza reads
//! them. [`save`] writes them top-down with a `ROWORDER` card saying so,
//! which PixInsight and Siril follow.

use image::{DynamicImage, ImageBuffer, ImageFormat, Luma, Rgb, Rgb32FImage};
use seiza::raster::OrientedRaster;
use seiza_fits::{F32ImageData, FitsError, FitsImage, HeaderValue, Pixels, WriteHeaderCard};
use seiza_xisf::{XisfError, XisfImageInfo, XisfMetadata};
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
    Raster(#[from] seiza::Error),
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
    if Format::from_path(path) == Some(Format::Xisf) {
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
    let image = FitsImage::open(path).map_err(|error| Error::open(path, error))?;
    Ok(AstronomyImage {
        image,
        format: Format::Fits,
        xisf_info: None,
        xisf_metadata: None,
    })
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
    /// [`seiza::prior::solve_with_prior`]: its WCS cards, an XISF
    /// astrometric solution, its pointing (`RA`/`DEC`, `OBJCTRA`/`OBJCTDEC`,
    /// or XISF `Observation:Center`), and its pixel scale (`PIXSCALE`, or
    /// pixel size over focal length, from cards or XISF `Instrument`
    /// properties).
    pub fn solve_prior(&self) -> seiza::prior::SolvePrior {
        let mut headers = self.image.headers.clone();
        if let Some(metadata) = &self.xisf_metadata {
            add_xisf_solution_wcs(&mut headers, metadata);
        }
        solve_prior_from_headers(&headers)
    }
}

/// [`AstronomyImage::solve_prior`] from a FITS or XISF file's headers and
/// metadata alone, without decoding its pixels.
pub fn read_solve_prior(path: &Path) -> Result<seiza::prior::SolvePrior, Error> {
    match Format::from_path(path) {
        Some(Format::Xisf) => {
            let (mut headers, metadata) = seiza_xisf::read_header_and_metadata(path)
                .map_err(|error| Error::open(path, error))?;
            add_xisf_solution_wcs(&mut headers, &metadata);
            Ok(solve_prior_from_headers(&headers))
        }
        Some(Format::Fits) => seiza_fits::read_header(path)
            .map(|headers| solve_prior_from_headers(&headers))
            .map_err(|error| Error::open(path, error)),
        _ => Err(Error::open(path, "it is not a FITS or XISF file")),
    }
}

/// [`AstronomyImage::solve_prior`] for header cards alone.
pub fn solve_prior_from_headers(headers: &[(String, HeaderValue)]) -> seiza::prior::SolvePrior {
    let card = |key: &str| {
        headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(key))
            .map(|(_, value)| value)
    };
    seiza::prior::SolvePrior::from_header(
        |key| card(key).and_then(HeaderValue::as_f64),
        |key| card(key).and_then(HeaderValue::as_str).map(str::to_owned),
    )
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
    seiza::raster::open_oriented(path).map_err(|error| Error::open(path, error))
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
            image
                .save_with_format(path, format.raster().expect("a raster format"))
                .map_err(|error| Error::save(path, error))
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
            eight
                .save_with_format(path, ImageFormat::Jpeg)
                .map_err(|error| Error::save(path, error))
        }
        Some(raster) => image
            .save_with_format(path, raster)
            .map_err(|error| Error::save(path, error)),
        None => save(path, &FloatImage::from_display(image)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
