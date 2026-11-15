//! File-level learned restoration shared by the CLI and application bindings.

use crate::{
    DenoiseOptions, DenoiseReport, Denoised, ImageView, ModelError, NoiseBands, NoiseModel,
};
use seiza_fits::{BayerPattern, FitsError, HeaderValue, WriteHeaderCard};
use seiza_image_io::{Cause, read_header, read_window};
use seiza_stacking::{FitsFrame, LinearImage, ReferenceRegion, write_processed_image_fits_f32};
use seiza_stretch::{SampleDomain, StretchConfig, StretchError, StretchParams};
use std::path::Path;

const PREVIEW_EDGE: usize = 768;

/// Why a file could not be restored or written.
#[derive(Debug, thiserror::Error)]
pub enum FileError {
    /// The file could not be read.
    #[error(transparent)]
    Image(#[from] seiza_image_io::Error),
    /// The frame could not be constructed or saved.
    #[error(transparent)]
    Frame(#[from] seiza_stacking::Error),
    /// A comparison could not be stretched.
    #[error(transparent)]
    Stretch(#[from] StretchError),
    /// Inference or blending failed.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// Bayer data needs debayering first.
    #[error("choose a prepared mono or RGB image; debayer raw images first")]
    Bayer,
    /// The image is already display-referred.
    #[error("choose a linear FITS or XISF image for restoration")]
    Display,
    /// The header has no usable value for a dimension.
    #[error("missing or invalid image dimension {0}")]
    Dimension(&'static str),
    /// The center has no coverage in at least one channel.
    #[error("the center crop has no usable data in at least one channel")]
    EmptyCenter,
    /// The requested output belongs to a different input region.
    #[error("a center comparison and a full image require separate model runs")]
    OutputArea,
}

/// The pixels to read and restore from a file.
#[derive(Clone, Copy, Debug)]
pub enum RestorationArea {
    /// The full linear image with source metadata.
    Full,
    /// The center crop, up to 768 by 768 pixels.
    Center,
}

/// The decoded linear file or center crop, ready for inference.
pub struct RestorationImage {
    image: LinearImage,
    headers: Option<Vec<(String, HeaderValue)>>,
}

impl RestorationImage {
    /// Open a FITS or XISF stack. Compressed XISF is decoded before cropping.
    pub fn open(path: &Path, area: RestorationArea) -> Result<Self, FileError> {
        let (mut image, headers) = match area {
            RestorationArea::Center => (open_center(path)?, None),
            RestorationArea::Full => {
                let frame = open_full(path)?;
                (frame.image, Some(frame.headers))
            }
        };
        if headers.is_none()
            && (0..image.channels).any(|channel| {
                !image
                    .data
                    .iter()
                    .skip(channel)
                    .step_by(image.channels)
                    .any(|sample| sample.is_finite())
            })
        {
            return Err(FileError::EmptyCenter);
        }
        for sample in &mut image.data {
            if !sample.is_finite() {
                *sample = f32::NAN;
            }
        }
        Ok(Self { image, headers })
    }

    /// Run inference, retaining the residual for subsequent blends.
    pub fn denoise(
        self,
        model: &mut NoiseModel,
        options: &DenoiseOptions<'_>,
    ) -> Result<FileRestoration, FileError> {
        let image = &self.image;
        let view = ImageView::new(&image.data, image.width, image.height, image.channels)?;
        let denoised = model.denoise(view, options)?;
        let source = match self.headers {
            Some(headers) => OutputSource::Full(headers),
            None => OutputSource::Center(self.image),
        };
        Ok(FileRestoration { source, denoised })
    }
}

enum OutputSource {
    Full(Vec<(String, HeaderValue)>),
    Center(LinearImage),
}

/// The result of restoring a whole file or its center.
pub struct FileRestoration {
    source: OutputSource,
    denoised: Denoised,
}

impl FileRestoration {
    /// The model, graph and backend used for this run.
    pub fn report(&self) -> &DenoiseReport {
        &self.denoised.report
    }

    /// Write the full linear result with the original geometry and provenance.
    pub fn write_linear(&self, path: &Path, bands: &NoiseBands) -> Result<(), FileError> {
        let OutputSource::Full(headers) = &self.source else {
            return Err(FileError::OutputArea);
        };
        let image = self.blended_image(bands)?;
        write_processed_image_fits_f32(
            path,
            &image,
            headers,
            &operation_cards(self.report(), bands),
        )?;
        Ok(())
    }

    /// Write the original and restored center crops with one display stretch.
    pub fn write_comparison(
        &self,
        original_path: &Path,
        restored_path: &Path,
        bands: &NoiseBands,
    ) -> Result<(), FileError> {
        let OutputSource::Center(original) = &self.source else {
            return Err(FileError::OutputArea);
        };
        write_comparison(
            original,
            &self.blended_image(bands)?,
            original_path,
            restored_path,
        )
    }

    fn blended_image(&self, bands: &NoiseBands) -> Result<LinearImage, FileError> {
        let result = &self.denoised;
        Ok(LinearImage::new(
            result.width(),
            result.height(),
            result.channels(),
            result.blend(bands)?,
        )?)
    }
}

fn validate_linear(headers: &[(String, HeaderValue)]) -> Result<(), FileError> {
    if headers.iter().any(|(key, value)| {
        key.eq_ignore_ascii_case("SEIZATRF") && value.as_str() == Some("DISPLAY")
    }) {
        return Err(FileError::Display);
    }
    Ok(())
}

fn open_full(path: &Path) -> Result<FitsFrame, FileError> {
    let frame = FitsFrame::open(path)?;
    if frame.bayer.is_some() {
        return Err(FileError::Bayer);
    }
    validate_linear(&frame.headers)?;
    Ok(frame)
}

fn center_region(width: usize, height: usize) -> ReferenceRegion {
    let crop_width = width.min(PREVIEW_EDGE);
    let crop_height = height.min(PREVIEW_EDGE);
    ReferenceRegion {
        x: (width - crop_width) / 2,
        y: (height - crop_height) / 2,
        width: crop_width,
        height: crop_height,
    }
}

fn open_center(path: &Path) -> Result<LinearImage, FileError> {
    let header = read_header(path)?;
    validate_linear(&header.cards)?;
    let dimension = |name: &'static str| {
        header
            .cards
            .iter()
            .find(|(key, _)| key == name)
            .and_then(|(_, value)| value.as_i64())
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value > 0)
            .ok_or(FileError::Dimension(name))
    };
    let region = center_region(dimension("NAXIS1")?, dimension("NAXIS2")?);
    let window = match read_window(path, region.x, region.y, region.width, region.height, None) {
        Ok(window) => window,
        Err(error)
            if matches!(
                error.cause(),
                Cause::Xisf(_) | Cause::Fits(FitsError::Unsupported(_))
            ) =>
        {
            // Some containers, such as compressed XISF, need a full decode.
            // Crop before running the model even on this path.
            let image = open_full(path)?.image;
            return image
                .crop(center_region(image.width, image.height))
                .map_err(Into::into);
        }
        Err(error) => return Err(error.into()),
    };
    if window.planes == 1
        && header.cards.iter().any(|(key, value)| {
            key.eq_ignore_ascii_case("BAYERPAT")
                && value.as_str().and_then(BayerPattern::parse).is_some()
        })
    {
        return Err(FileError::Bayer);
    }
    let data = if window.planes == 3 {
        let (red, green_blue) = window.values.split_at(window.width * window.height);
        let (green, blue) = green_blue.split_at(red.len());
        red.iter()
            .zip(green)
            .zip(blue)
            .flat_map(|((&r, &g), &b)| [r, g, b])
            .collect()
    } else {
        window.values
    };
    LinearImage::new(window.width, window.height, window.planes, data).map_err(Into::into)
}

fn operation_cards(report: &DenoiseReport, bands: &NoiseBands) -> Vec<WriteHeaderCard> {
    let text = |key: &str, value: String, comment: &str| {
        WriteHeaderCard::new(key, HeaderValue::String(value)).with_comment(comment)
    };
    let number = |key: &str, value: f32, comment: &str| {
        WriteHeaderCard::new(key, HeaderValue::Float(f64::from(value))).with_comment(comment)
    };
    let mut cards = vec![
        text("SEIZAMDL", report.model.clone(), "Seiza learned model"),
        text("SEIZATRF", "LINEAR".to_string(), "linear sample transfer"),
        text(
            "MDHASH",
            report.sha256.chars().take(16).collect(),
            "model graph SHA-256, first 16",
        ),
        text(
            "MDCARD",
            report.card_sha256.chars().take(16).collect(),
            "model card SHA-256, first 16",
        ),
        text("MDBACKND", report.backend.to_string(), "backend requested"),
        text("MDPREC", report.precision.to_string(), "graph precision"),
        text(
            "MDNOISE",
            report.noise_map.to_string(),
            "noise input, made from the image",
        ),
        WriteHeaderCard::new(
            "MDTILE",
            HeaderValue::Integer(i64::try_from(report.tile).unwrap_or(i64::MAX)),
        )
        .with_comment("tile side in pixels"),
        number("MDHIGH", bands.high, "share of the grain taken out"),
        number("MDLOW", bands.low, "share of the mottle taken out"),
        number("MDSCALE", bands.scale_px, "band scale in pixels"),
        text(
            "MDVERS",
            crate::OUTPUT_VERSION.to_string(),
            "seiza-model output version",
        ),
    ];
    if let Some(guard) = report.holes {
        cards.push(
            WriteHeaderCard::new("MDHOLES", HeaderValue::Float(guard.depth))
                .with_comment("holes this many sigma deep kept from the model"),
        );
    }
    cards
}

fn write_comparison(
    original: &LinearImage,
    restored: &LinearImage,
    original_path: &Path,
    restored_path: &Path,
) -> Result<(), FileError> {
    let channels = original.channels;
    let domain = SampleDomain::PhysicalLinear {
        normalization: Default::default(),
    }
    .resolve(&original.data, channels)?;
    let original_data = domain.apply(&original.data, channels)?;
    let restored_data = domain.apply(&restored.data, channels)?;
    let plan = StretchConfig::auto_mtf(StretchParams::default(), 200_000)
        .resolve_for(&original_data, channels)?;
    let cards = [WriteHeaderCard::new(
        "SEIZATRF",
        HeaderValue::String("DISPLAY".into()),
    )];
    for (data, path) in [
        (original_data, original_path),
        (restored_data, restored_path),
    ] {
        let pixels = plan.apply_f32(&data, channels)?;
        let display = LinearImage::new(original.width, original.height, channels, pixels)?;
        write_processed_image_fits_f32(path, &display, &[], &cards)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, Precision};

    #[test]
    fn preview_reads_the_center_and_preserves_rgb_order() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("rgb.fits");
        let width = 1001;
        let height = 4;
        let data = (0..width * height)
            .flat_map(|index| [index as f32, (index + 1_000) as f32, -(index as f32)])
            .collect();
        let image = LinearImage::new(width, height, 3, data).unwrap();
        write_processed_image_fits_f32(&path, &image, &[], &[]).unwrap();

        let preview = open_center(&path).unwrap();
        assert_eq!(
            (preview.width, preview.height, preview.channels),
            (768, height, 3)
        );
        assert_eq!(
            preview.data,
            image.crop(center_region(width, height)).unwrap().data
        );
    }
    #[test]
    fn the_cards_record_what_ran() {
        let report = DenoiseReport {
            model: "noise-test".to_string(),
            sha256: "0123456789abcdef0123".to_string(),
            card_sha256: "fedcba9876543210fedc".to_string(),
            backend: Backend::Cpu,
            precision: Precision::F32,
            tile: 512,
            noise_map: crate::NoiseInput::Constant,
            holes: None,
            channels: Vec::new(),
        };
        let cards = operation_cards(&report, &NoiseBands::default());
        let keys: Vec<&str> = cards.iter().map(WriteHeaderCard::keyword).collect();
        assert!(keys.contains(&"MDHASH") && keys.contains(&"MDCARD") && keys.contains(&"MDBACKND"));
        assert!(
            !keys.contains(&"MDHOLES"),
            "a run without the guard says nothing of it"
        );
        assert!(keys.iter().all(|key| key.len() <= 8), "{keys:?}");
        let guarded = DenoiseReport {
            holes: Some(crate::HoleGuard {
                depth: 30.0,
                margin: 3,
                fill_radius: 12,
                restore: 5.0,
            }),
            ..report
        };
        let cards = operation_cards(&guarded, &NoiseBands::default());
        assert!(cards.iter().any(|card| card.keyword() == "MDHOLES"));
    }
}
