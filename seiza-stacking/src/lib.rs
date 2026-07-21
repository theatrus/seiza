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
pub use paths::paths_refer_to_same_file;
pub use registration::{
    Registrar, RegistrationOptions, RegistrationResult, SimilarityTransform, resample_to_reference,
};
pub use stack::{
    DeltaSigmaOptions, FrameAcceptanceCriteria, FrameDiagnostics, FrameDisposition,
    FrameRejectionReason, LiveStacker, RejectionMode, StackOptions, StackSnapshot, StackView,
};

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("invalid image: {0}")]
    InvalidImage(String),
    #[error("calibration error: {0}")]
    Calibration(String),
    #[error("registration failed: {0}")]
    Registration(String),
    #[error("normalization failed: {0}")]
    Normalization(String),
    #[error("stacking error: {0}")]
    Stack(String),
    #[error("color composition error: {0}")]
    Color(String),
    #[error("failed to read FITS frame {}: {source}", path.display())]
    FitsRead {
        path: PathBuf,
        #[source]
        source: seiza_fits::FitsError,
    },
    #[error("failed to write FITS frame {}: {source}", path.display())]
    FitsWrite {
        path: PathBuf,
        #[source]
        source: seiza_fits::FitsError,
    },
}

pub type Result<T> = std::result::Result<T, Error>;
