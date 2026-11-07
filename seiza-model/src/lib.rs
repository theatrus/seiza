//! Learned restoration of linear astrophotography images.
//!
//! A model is a graph and a card. Everything around the graph is here, in
//! Rust, and is the same on every backend: the sky, the noise input, tiling,
//! blending and the bands of noise kept. A backend only runs tiles, through
//! ONNX Runtime, which [`Backend`] finds and opens at run time, so a build of
//! this crate links nothing native.
//!
//! A shipped model takes an image and nothing else, from any stacker. Whatever
//! else it reads is made from that image, as its [`ModelCard`] says.
//!
//! ```no_run
//! use seiza_model::{
//!     Backend, BackendOptions, DenoiseOptions, ImageView, ModelBundle, NoiseBands, NoiseModel,
//! };
//!
//! # fn run(pixels: &[f32], width: usize, height: usize) -> Result<(), seiza_model::ModelError> {
//! let bundle = ModelBundle::open("models/noise-v1")?;
//! let options = BackendOptions::default();
//! let backend = Backend::detect(&bundle, &options)?;
//! let mut model = NoiseModel::open(&bundle, backend, &options)?;
//! let image = ImageView::new(pixels, width, height, 1)?;
//! let result = model.denoise(image, &DenoiseOptions::default())?;
//! // A second blend costs no inference.
//! let kept = result.blend(&NoiseBands { high: 0.7, low: 0.9, scale_px: 5.0 })?;
//! # Ok(())
//! # }
//! ```

mod backend;
mod bands;
mod bundle;
mod error;
mod model;
mod noise;
#[cfg(test)]
mod parity;
mod session;
mod sky;
mod stats;
mod store;
mod tiles;

pub use backend::{
    Backend, BackendOptions, BackendStatus, CoreMlUnits, Precision, RuntimeInfo, UnknownBackend,
    UnknownPrecision, runtime_info,
};
pub use bands::NoiseBands;
pub use bundle::{
    GraphFile, ModelBundle, ModelCard, NoiseInput, NoisePlane, Reading, SkyInput, Task, TilingCard,
};
pub use error::ModelError;
pub use model::{
    ChannelScales, DenoiseOptions, DenoiseReport, Denoised, ImageView, NoiseModel, Progress,
};
pub use store::ModelStore;
pub use tiles::Tiling;

/// The version of this crate whose output a build reproduces. A host keys
/// cached results on it, with the model's hash, the backend and the precision.
pub const OUTPUT_VERSION: &str = "0.1.0";
