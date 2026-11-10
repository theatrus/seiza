use crate::backend::Backend;
use std::path::PathBuf;

/// Why a model could not be found, opened or run.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// No ONNX Runtime library was found, or the one found did not load.
    #[error("ONNX Runtime is not available: {0}")]
    RuntimeNotFound(String),
    /// This machine or this ONNX Runtime cannot run the backend.
    #[error("backend {backend} is not available: {reason}")]
    BackendUnavailable {
        /// The backend asked for.
        backend: Backend,
        /// Why it cannot run here.
        reason: String,
    },
    /// A bundle's card asks for something this build does not do.
    #[error("model contract: {0}")]
    ModelContract(String),
    /// A bundle's file does not match the hash its card records.
    #[error("{} does not match its card: expected {expected}, found {found}", file.display())]
    HashMismatch {
        /// The file that differs.
        file: PathBuf,
        /// The hash the card records.
        expected: String,
        /// The file's own hash.
        found: String,
    },
    /// No installed model has the name asked for.
    #[error("no model named {0} is installed")]
    ModelNotFound(String),
    /// No model was named, and none is installed to take its place.
    #[error("no noise model is installed in {}", store.display())]
    NoModelInstalled {
        /// The store searched.
        store: PathBuf,
    },
    /// No model was named, and more than one could take its place.
    #[error("several noise models are installed, so name one: {}", names.join(", "))]
    SeveralModels {
        /// The models installed.
        names: Vec<String>,
    },
    /// The image cannot be run: empty, the wrong length, or with no finite pixels.
    #[error("{0}")]
    InvalidImage(String),
    /// An option's value is out of range.
    #[error("{0}")]
    InvalidOption(String),
    /// A tile came back holding a value that is not finite.
    #[error("the model returned a non-finite value in the tile at ({x}, {y})")]
    NonFiniteOutput {
        /// The tile's left edge.
        x: usize,
        /// The tile's top edge.
        y: usize,
    },
    /// The caller asked the run to stop.
    #[error("the run was cancelled")]
    Cancelled,
    /// ONNX Runtime refused or failed a call.
    #[error("ONNX Runtime: {0}")]
    Runtime(String),
    /// A file could not be read or written.
    #[error("{}: {source}", path.display())]
    Io {
        /// The path concerned.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// A hosted model could not be fetched: its manifest or its files, or a
    /// name the manifest does not offer
    /// ([`ModelNotOffered`](seiza_download::Error::ModelNotOffered)).
    #[cfg(feature = "download")]
    #[error("{0}")]
    Download(#[from] seiza_download::Error),
}

impl<R> From<ort::Error<R>> for ModelError {
    fn from(error: ort::Error<R>) -> Self {
        Self::Runtime(error.to_string())
    }
}

impl ModelError {
    pub(crate) fn io(path: impl Into<PathBuf>) -> impl FnOnce(std::io::Error) -> Self {
        let path = path.into();
        move |source| Self::Io { path, source }
    }
}
