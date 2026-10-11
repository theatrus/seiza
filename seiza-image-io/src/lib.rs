//! Open and save images by their file extension, the same way across Seiza.
//!
//! PNG, JPEG and TIFF go through the `image` crate, upright by their EXIF
//! orientation as [`seiza::raster`] reads them; FITS and XISF go through
//! Seiza's own readers and writers. [`open`] gives any of them as
//! floating-point samples. Integer samples are scaled to 0 to 1 by their
//! type's range, so an 8-bit PNG and a 16-bit FITS read alike;
//! floating-point samples are kept as stored. [`save`] writes PNG and TIFF
//! at 16 bits and FITS and XISF as 32-bit floats.
//!
//! FITS rows are read in their stored order, as the rest of Seiza reads
//! them, and written top-down with a `ROWORDER` card saying so, which
//! PixInsight and Siril follow.

use image::{ImageBuffer, ImageFormat, Luma, Rgb, Rgb32FImage};
use seiza_fits::{F32ImageData, FitsImage, HeaderValue, Pixels, WriteHeaderCard};
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

    /// Whether [`save`] writes it. JPEG is lossy and 8-bit, so it does not.
    pub fn can_save(self) -> bool {
        !matches!(self, Self::Jpeg)
    }
}

/// The extensions [`save`] writes, for messages.
pub const SAVE_EXTENSIONS: &str = ".png, .tif, .fits or .xisf";

/// Why an image could not be opened or saved.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to open {}: {message}", path.display())]
    Open { path: PathBuf, message: String },
    #[error("failed to write {}: {message}", path.display())]
    Save { path: PathBuf, message: String },
}

impl Error {
    fn open(path: &Path, message: impl ToString) -> Self {
        Self::Open {
            path: path.to_path_buf(),
            message: message.to_string(),
        }
    }

    fn save(path: &Path, message: impl ToString) -> Self {
        Self::Save {
            path: path.to_path_buf(),
            message: message.to_string(),
        }
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

    /// Whether the samples fill the size, with 1 or 3 channels.
    fn check(&self) -> Result<(), String> {
        if !matches!(self.channels, 1 | 3) {
            return Err(format!(
                "an image has 1 or 3 channels, not {}",
                self.channels
            ));
        }
        if self.width == 0 || self.height == 0 {
            return Err("the image is empty".into());
        }
        let expected = self
            .width
            .checked_mul(self.height)
            .and_then(|pixels| pixels.checked_mul(self.channels));
        if expected != Some(self.samples.len()) {
            return Err(format!(
                "{} samples do not fill a {}×{} image of {} channel{}",
                self.samples.len(),
                self.width,
                self.height,
                self.channels,
                if self.channels == 1 { "" } else { "s" }
            ));
        }
        Ok(())
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

/// Open the image at `path`, its format chosen by its extension: FITS and
/// XISF by their own readers, anything else as a raster the `image` crate
/// knows, upright by its EXIF orientation.
pub fn open(path: &Path) -> Result<FloatImage, Error> {
    match Format::from_path(path) {
        Some(Format::Fits | Format::Xisf) => {
            let (image, _) = open_astronomy(path)?;
            from_fits(image).map_err(|message| Error::open(path, message))
        }
        _ => {
            let raster = seiza::raster::open_oriented(path)
                .map_err(|error| Error::open(path, error))?
                .pixels;
            let (width, height) = (raster.width() as usize, raster.height() as usize);
            Ok(if raster.color().has_color() {
                FloatImage::from_rgb(&raster.to_rgb32f())
            } else {
                FloatImage::gray(width, height, raster.to_luma32f().into_raw())
            })
        }
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

/// Write `image` to `path` in the format its extension names: PNG or TIFF
/// at 16 bits, samples clamped to 0 to 1, or FITS or XISF as 32-bit
/// floats, as they are.
pub fn save(path: &Path, image: &FloatImage) -> Result<(), Error> {
    image
        .check()
        .map_err(|message| Error::save(path, message))?;
    let format = Format::from_path(path)
        .filter(|format| format.can_save())
        .ok_or_else(|| Error::save(path, format!("name it {SAVE_EXTENSIONS}")))?;
    let (width, height) = (image.width, image.height);
    match format {
        Format::Png | Format::Tiff => {
            let encoding = if format == Format::Png {
                ImageFormat::Png
            } else {
                ImageFormat::Tiff
            };
            let samples: Vec<u16> = image
                .samples
                .iter()
                .map(|&sample| (sample.clamp(0.0, 1.0) * f32::from(u16::MAX)).round() as u16)
                .collect();
            let (width, height) = (
                u32::try_from(width).map_err(|_| Error::save(path, "it is too wide"))?,
                u32::try_from(height).map_err(|_| Error::save(path, "it is too tall"))?,
            );
            let saved = if image.channels == 1 {
                ImageBuffer::<Luma<u16>, _>::from_raw(width, height, samples)
                    .expect("checked size")
                    .save_with_format(path, encoding)
            } else {
                ImageBuffer::<Rgb<u16>, _>::from_raw(width, height, samples)
                    .expect("checked size")
                    .save_with_format(path, encoding)
            };
            saved.map_err(|error| Error::save(path, error))
        }
        Format::Fits => {
            let row_order =
                WriteHeaderCard::new("ROWORDER", HeaderValue::String("TOP-DOWN".into()))
                    .with_comment("first row is the top of the image");
            seiza_fits::write_f32_image(path, width, height, f32_data(image), &[row_order])
                .map_err(|error| Error::save(path, error))
        }
        Format::Xisf => {
            let normalized = image
                .samples
                .iter()
                .all(|sample| (0.0..=1.0).contains(sample));
            let options = seiza_xisf::WriteOptions {
                bounds: normalized.then_some((0.0, 1.0)),
                ..seiza_xisf::WriteOptions::default()
            };
            seiza_xisf::write_f32_image_with_options(
                path,
                width,
                height,
                f32_data(image),
                &[],
                &options,
            )
            .map_err(|error| Error::save(path, error))
        }
        Format::Jpeg => unreachable!("filtered out above"),
    }
}

/// `image`'s samples as the FITS and XISF writers take them.
fn f32_data(image: &FloatImage) -> F32ImageData<'_> {
    if image.channels == 1 {
        F32ImageData::Mono(&image.samples)
    } else {
        F32ImageData::RgbInterleaved(&image.samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }
}
