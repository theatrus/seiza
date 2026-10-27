//! Host-neutral calibration for linear image buffers.
//!
//! Four things live here: the pixel kernels that apply a calibration
//! response; the rules for deciding which frames belong together in the first
//! place — whether a dark suits a light, whether a flat still corrects it,
//! which candidates can be averaged into one master; grades for the bias,
//! dark, and flat frames that go into a master; [`flat_response`], which
//! measures whether flat sets describe the same light path and which one a
//! light's own sky follows; and [`dust`], the one definition of the dust a
//! flat holds and of how it changes between sessions, which the flat grades
//! and the comparisons share.
//!
//! All are pure. Hosts retain file loading, detector evidence, cache policy,
//! and user confirmation; they hand this crate values and act on what comes
//! back.
//!
//! # Grading calibration frames
//!
//! [`measure_flat`] and [`measure_dark`] read one raw frame each and return a
//! small, serializable measurement. [`screen_flats`], [`screen_darks`], and
//! [`screen_bias`] judge the measurements of one set — frames that should
//! match — and give each frame a [`FrameVerdict`]: a [`Grade`] and the
//! [`Finding`]s behind it. Most checks compare a frame with its peers and need
//! three or more frames. The screens say what the pixels show, never why, and
//! never judge a flat by its temperature.

pub mod dust;
pub mod flat_response;
pub mod library;
mod matching;
mod pedestal;
mod quality;
mod residual_flat;

pub use matching::{
    FrameRole, FrameSignature, LightRotation, MatchTolerances, coherent_subset,
    coherent_subset_indices, describe_optics_mismatch, describe_sensor_mismatch, describe_text,
    describe_value, exposure_matches, exposure_tolerance, optics_consistent, optics_match,
    rotation_matches, sensor_consistent, sensor_matches, sort_by_proximity, temperature_matches,
};
pub use pedestal::fit_flat_pedestal;
pub use quality::{
    BiasScreening, DarkMeasureOptions, DarkMeasurement, DarkScreening,
    FLAT_QUALITY_ALGORITHM_VERSION, FRAME_QUALITY_ALGORITHM_VERSION, Finding, FindingCode,
    FlatFeature, FlatMeasureOptions, FlatMeasurement, FlatScreening, FlatSetReport, FrameVerdict,
    Grade, IlluminationProfile, Limits, measure_dark, measure_flat, screen_bias, screen_darks,
    screen_flats,
};

pub use residual_flat::{
    RESIDUAL_FLAT_ALGORITHM_VERSION, ResidualFlatBuild, ResidualFlatDiagnostics,
    ResidualFlatOptions, ResidualFlatPatch, apply_residual_flat_response_at,
    build_residual_flat_patch, validate_residual_flat_response,
};

/// A validated, contiguous, row-major view of linear `f32` samples.
#[derive(Clone, Copy, Debug)]
pub struct LinearImageRef<'a> {
    data: &'a [f32],
    width: usize,
    height: usize,
    channels: usize,
}

/// A validated mutable view of a contiguous, row-major linear image.
#[derive(Debug)]
pub struct LinearImageMut<'a> {
    data: &'a mut [f32],
    width: usize,
    height: usize,
    channels: usize,
}

impl<'a> LinearImageMut<'a> {
    /// Borrow a mutable, contiguous, channel-interleaved image buffer.
    pub fn new(data: &'a mut [f32], width: usize, height: usize, channels: usize) -> Result<Self> {
        validate_dimensions(data.len(), width, height, channels)?;
        Ok(Self {
            data,
            width,
            height,
            channels,
        })
    }

    /// Row-major, channel-interleaved samples.
    pub fn data(&self) -> &[f32] {
        self.data
    }

    /// Mutable row-major, channel-interleaved samples.
    pub fn data_mut(&mut self) -> &mut [f32] {
        self.data
    }

    /// Image width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Number of interleaved channels per pixel.
    pub fn channels(&self) -> usize {
        self.channels
    }
}

impl<'a> LinearImageRef<'a> {
    /// Borrow a contiguous, channel-interleaved image buffer.
    pub fn new(data: &'a [f32], width: usize, height: usize, channels: usize) -> Result<Self> {
        validate_dimensions(data.len(), width, height, channels)?;
        Ok(Self {
            data,
            width,
            height,
            channels,
        })
    }

    /// Row-major, channel-interleaved samples.
    pub fn data(&self) -> &'a [f32] {
        self.data
    }

    /// Image width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Number of interleaved channels per pixel.
    pub fn channels(&self) -> usize {
        self.channels
    }

    pub(crate) fn sample_count(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    pub(crate) fn dimensions_match(&self, other: &Self) -> bool {
        self.width == other.width && self.height == other.height && self.channels == other.channels
    }
}

/// Errors returned by calibration kernels.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Invalid immutable calibration library snapshot.
    #[error("calibration library: {0}")]
    CalibrationLibrary(String),
    /// A measured flat-group request has inconsistent identities or limits.
    #[error("flat groups: {0}")]
    FlatGroups(String),
    /// Image dimensions, channel count, or sample buffer are inconsistent.
    #[error("invalid image: {0}")]
    InvalidImage(String),
    /// A residual response could not be fitted or applied safely.
    #[error("residual flat: {0}")]
    ResidualFlat(String),
    /// Calibration frame measurement or screening settings were refused, or
    /// the measurements given cannot form one set.
    #[error("frame quality: {0}")]
    FrameQuality(String),
    /// A [`MatchTolerances`] value no comparison can use; see
    /// [`MatchTolerances::validate`].
    #[error("invalid match tolerance: {0}")]
    InvalidTolerance(String),
}

/// Result specialized to this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn validate_dimensions(
    sample_count: usize,
    width: usize,
    height: usize,
    channels: usize,
) -> Result<()> {
    if width == 0 || height == 0 || channels == 0 {
        return Err(Error::InvalidImage(
            "dimensions and channel count must be non-zero".into(),
        ));
    }
    let expected = width
        .checked_mul(height)
        .and_then(|value| value.checked_mul(channels))
        .ok_or_else(|| Error::InvalidImage("image dimensions overflow".into()))?;
    if sample_count != expected {
        return Err(Error::InvalidImage(format!(
            "pixel buffer has {sample_count} samples; expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_view_rejects_invalid_buffers() {
        assert!(LinearImageRef::new(&[], 0, 1, 1).is_err());
        assert!(LinearImageRef::new(&[0.0; 3], 2, 2, 1).is_err());
        assert!(LinearImageRef::new(&[0.0; 4], usize::MAX, 2, 1).is_err());
    }
}
