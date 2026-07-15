use std::path::PathBuf;

/// Errors returned by catalog bundle installation and cache operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("could not determine a Seiza cache directory; configure one explicitly")]
    CacheDirectoryUnavailable,

    #[error("{action} {}: {source}", path.display())]
    Io {
        action: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to fetch {url}: {source}")]
    Http {
        url: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("{url} returned HTTP {status}")]
    HttpStatus { url: String, status: u16 },

    #[error("catalog bundle manifest is invalid: {0}")]
    Manifest(String),

    #[error("catalog bundle manifest is not valid JSON: {0}")]
    ManifestJson(#[from] serde_json::Error),

    #[error("offline mode requires a cached bundle manifest at {}", .0.display())]
    CachedManifestUnavailable(PathBuf),

    #[error("{name} has {actual} bytes; expected {expected}")]
    Size {
        name: String,
        expected: u64,
        actual: u64,
    },

    #[error("{name} failed SHA-256 verification: expected {expected}, got {actual}")]
    Checksum {
        name: String,
        expected: String,
        actual: String,
    },

    #[error("catalog bundle does not contain {0}")]
    MissingArtifact(String),

    #[error("background cache task failed: {0}")]
    BackgroundTask(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn io(action: &'static str, path: impl Into<PathBuf>, source: std::io::Error) -> Error {
    Error::Io {
        action,
        path: path.into(),
        source,
    }
}
