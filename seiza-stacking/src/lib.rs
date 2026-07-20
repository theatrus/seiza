//! Linear calibration, local registration, normalization, and incremental
//! image stacking for astrophotography.

mod calibration;
mod fits;
mod image;
mod master;
mod normalization;
mod registration;
mod stack;

pub use calibration::{CalibrationMasters, MasterDark, MasterFlat};
pub use fits::{FitsFrame, write_fits_f32, write_master_fits_f32};
pub use image::{BayerLayout, LinearImage};
pub use master::{
    MasterBuildOptions, MasterFrame, MasterFrameKind, MasterInputStatistics,
    MasterRejectionOptions, build_master_from_fits,
};
pub use normalization::{NormalizationMap, NormalizationMode};
pub use registration::{Registrar, RegistrationOptions, RegistrationResult, SimilarityTransform};
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
    #[error("failed to read FITS frame {}: {source}", path.display())]
    FitsRead {
        path: PathBuf,
        #[source]
        source: seiza_fits::FitsError,
    },
    #[error("I/O error for {}: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

pub type Result<T> = std::result::Result<T, Error>;
