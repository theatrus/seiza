//! Linear calibration, local registration, normalization, and incremental
//! image stacking for astrophotography.

mod calibration;
mod color;
mod fits;
mod image;
mod master;
mod normalization;
mod paths;
mod registration;
mod stack;

pub use calibration::{CalibrationMasters, MasterDark, MasterFlat};
pub use color::{
    ColorComposition, ColorNormalization, ColorOptions, ColorTransfer, ForaxxOptions,
    NarrowbandMatrix, NarrowbandMix, NarrowbandPalette, combine_lrgb, combine_narrowband,
    combine_narrowband_matrix, combine_rgb,
};
pub use fits::{
    FitsFrame, write_color_fits_f32, write_fits_f32, write_linear_image_fits_f32,
    write_master_fits_f32, write_processed_image_fits_f32,
};
pub use image::{BayerLayout, LinearImage};
pub use master::{
    MasterBuildOptions, MasterFrame, MasterFrameKind, MasterInputStatistics,
    MasterRejectionOptions, build_master_from_fits,
};
pub use normalization::{NormalizationMap, NormalizationMode};
pub use paths::{path_identity, paths_refer_to_same_file};
pub use registration::{
    Registrar, RegistrationOptions, RegistrationResult, SimilarityTransform, resample_to_reference,
};
pub use stack::{
    DeltaSigmaOptions, FrameAcceptanceCriteria, FrameDiagnostics, FrameDisposition,
    FrameRejectionReason, LiveStacker, RejectionMode, StackOptions, StackSnapshot, StackView,
};

use std::path::PathBuf;

/// Anything that can go wrong while calibrating, registering, normalizing, or
/// stacking frames.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Image dimensions, channel count, or sample buffer are inconsistent.
    #[error("invalid image: {0}")]
    InvalidImage(String),
    /// A calibration master or its metadata could not be applied.
    #[error("calibration error: {0}")]
    Calibration(String),
    /// No star match reached the registration thresholds.
    #[error("registration failed: {0}")]
    Registration(String),
    /// Background matching between reference and source failed.
    #[error("normalization failed: {0}")]
    Normalization(String),
    /// A frame could not be integrated into the stack.
    #[error("stacking error: {0}")]
    Stack(String),
    /// Color composition inputs or options were rejected.
    #[error("color composition error: {0}")]
    Color(String),
    /// Reading a FITS frame from disk failed.
    #[error("failed to read FITS frame {}: {source}", path.display())]
    FitsRead {
        /// Path that could not be read.
        path: PathBuf,
        /// Underlying FITS decode error.
        #[source]
        source: seiza_fits::FitsError,
    },
    /// Reading an XISF frame from disk failed.
    #[error("failed to read XISF frame {}: {source}", path.display())]
    XisfRead {
        /// Path that could not be read.
        path: PathBuf,
        /// Underlying XISF decode error.
        #[source]
        source: seiza_xisf::XisfError,
    },
    /// Writing a FITS frame to disk failed.
    #[error("failed to write FITS frame {}: {source}", path.display())]
    FitsWrite {
        /// Path that could not be written.
        path: PathBuf,
        /// Underlying FITS encode error.
        #[source]
        source: seiza_fits::FitsError,
    },
}

/// Result specialized to this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
