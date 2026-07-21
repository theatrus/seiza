//! Generic content-addressed artifact transfer and verification.
//!
//! This is the substrate shared by every hosted-bundle consumer: catalog
//! bundles today, the satellite mirror next. It owns HTTP acquisition, the
//! encoded/decoded SHA-256 and size verification, decompression with a bounded
//! output, and atomic publication of one immutable artifact. *Selection* (which
//! artifacts, from which manifest) and *retention* (how the local cache is laid
//! out and evicted) are deliberately left to the caller — those differ between
//! a frozen named catalog set and a rolling time series of satellite snapshots.

use crate::Error;
use crate::error::io;
use async_compression::tokio::bufread::ZstdDecoder;
use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_util::io::StreamReader;

use crate::Result;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
pub(crate) const IO_BUFFER_BYTES: usize = 1024 * 1024;

/// Build an HTTP client for bundle transfers. `user_agent` identifies the
/// calling crate and version (e.g. `seiza-download/0.4.0`). `read_timeout`
/// bounds inactivity while reading the response body — callers pick a value
/// matched to their artifact sizes rather than inheriting a hidden default.
pub fn http_client(
    user_agent: impl Into<String>,
    read_timeout: std::time::Duration,
) -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(read_timeout)
        .user_agent(user_agent.into())
        .build()
        .map_err(|source| Error::Http {
            url: "HTTP client initialization".into(),
            source,
        })
}

/// How the bytes on the wire are encoded relative to the canonical artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Encoding {
    /// The transferred body is zstd-compressed; the canonical artifact is its
    /// decompression.
    Zstd,
}

impl Encoding {
    fn label(self) -> &'static str {
        match self {
            Encoding::Zstd => "zstd",
        }
    }
}

/// The encoded transport for an artifact: what is verified on the wire before
/// it is decoded to the canonical form.
#[derive(Clone, Copy, Debug)]
pub struct EncodedSpec<'a> {
    pub encoding: Encoding,
    /// Encoded (on-wire) byte count.
    pub bytes: u64,
    /// SHA-256 of the encoded bytes.
    pub sha256: &'a str,
}

/// One immutable artifact to fetch and verify. When `encoded` is `None` the
/// body is the canonical artifact itself; when `Some`, the body is verified in
/// its encoded form, then decoded and re-verified against `decoded_*`.
#[derive(Clone, Copy, Debug)]
pub struct ArtifactSpec<'a> {
    /// Human-facing name used in progress events and error messages.
    pub label: &'a str,
    /// Absolute URL of the object to GET.
    pub url: &'a str,
    /// Canonical (decoded) byte count.
    pub decoded_bytes: u64,
    /// SHA-256 of the canonical (decoded) bytes.
    pub decoded_sha256: &'a str,
    pub encoded: Option<EncodedSpec<'a>>,
}

/// Byte-level progress for a single transfer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransferProgress {
    /// Bytes received from the network (encoded bytes for a compressed body).
    pub downloaded: u64,
    /// Total transfer size in bytes (the on-wire size).
    pub total: u64,
    /// Bytes written to the local file so far. Equal to `downloaded` for an
    /// identity transfer; ahead of it while decompressing.
    pub written: u64,
}

/// Observable events from a single transfer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransferEvent {
    /// Emitted once, after the response status and any `Content-Length` are
    /// validated and before the body is streamed. `total` is the on-wire
    /// transfer size (encoded bytes for a compressed transport). A transfer
    /// that fails at the request or validation stage emits no `Started`.
    Started {
        total: u64,
    },
    Progress(TransferProgress),
}

/// Stream `spec` into `temp`, verifying the encoded body then the decoded
/// artifact by size and SHA-256, bounding decoded output to `decoded_bytes`. On
/// success the verified canonical bytes are left at `temp`; the caller installs
/// them into its cache. On failure `temp` is left for the caller to clean up.
pub async fn stream_to_temp(
    client: &reqwest::Client,
    spec: &ArtifactSpec<'_>,
    temp: &Path,
    report: impl Fn(TransferEvent),
) -> Result<()> {
    match spec.encoded {
        None => stream_identity(client, spec, temp, &report).await,
        Some(encoded) => stream_encoded(client, spec, &encoded, temp, &report).await,
    }
}

async fn stream_identity(
    client: &reqwest::Client,
    spec: &ArtifactSpec<'_>,
    temp: &Path,
    report: &impl Fn(TransferEvent),
) -> Result<()> {
    let response = get(client, spec.url).await?;
    if let Some(bytes) = response.content_length()
        && bytes != spec.decoded_bytes
    {
        return Err(Error::Size {
            name: spec.label.into(),
            expected: spec.decoded_bytes,
            actual: bytes,
        });
    }

    report(TransferEvent::Started {
        total: spec.decoded_bytes,
    });
    let mut output = tokio::fs::File::create(temp)
        .await
        .map_err(|source| io("create", temp, source))?;
    let mut stream = response.bytes_stream();
    let mut hasher = Sha256::new();
    let mut downloaded = 0u64;
    let mut reported = 0u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|source| Error::Http {
            url: spec.url.into(),
            source,
        })?;
        // Reject before writing when the body overruns the declared size, so a
        // lying or absent Content-Length (chunked/HTTP-2) cannot make us write
        // an unbounded temp file. Mirrors the decoded bound in the zstd path.
        let next = downloaded + chunk.len() as u64;
        if next > spec.decoded_bytes {
            return Err(Error::Size {
                name: spec.label.into(),
                expected: spec.decoded_bytes,
                actual: next,
            });
        }
        output
            .write_all(&chunk)
            .await
            .map_err(|source| io("write", temp, source))?;
        hasher.update(&chunk);
        downloaded = next;
        if downloaded == spec.decoded_bytes
            || downloaded.saturating_sub(reported) >= 4 * 1024 * 1024
        {
            report(TransferEvent::Progress(TransferProgress {
                downloaded,
                total: spec.decoded_bytes,
                written: downloaded,
            }));
            reported = downloaded;
        }
    }
    output
        .sync_all()
        .await
        .map_err(|source| io("sync", temp, source))?;
    drop(output);

    if downloaded != spec.decoded_bytes {
        return Err(Error::Size {
            name: spec.label.into(),
            expected: spec.decoded_bytes,
            actual: downloaded,
        });
    }
    let actual = lowercase_hex(&hasher.finalize());
    if actual != spec.decoded_sha256 {
        return Err(Error::Checksum {
            name: spec.label.into(),
            expected: spec.decoded_sha256.into(),
            actual,
        });
    }
    Ok(())
}

async fn stream_encoded(
    client: &reqwest::Client,
    spec: &ArtifactSpec<'_>,
    encoded: &EncodedSpec<'_>,
    temp: &Path,
    report: &impl Fn(TransferEvent),
) -> Result<()> {
    let encoded_label = || format!("{} ({} transport)", spec.label, encoded.encoding.label());
    let response = get(client, spec.url).await?;
    if let Some(bytes) = response.content_length()
        && bytes != encoded.bytes
    {
        return Err(Error::Size {
            name: encoded_label(),
            expected: encoded.bytes,
            actual: bytes,
        });
    }

    report(TransferEvent::Started {
        total: encoded.bytes,
    });
    let downloaded = Arc::new(AtomicU64::new(0));
    let stream_downloaded = Arc::clone(&downloaded);
    let encoded_hasher = Arc::new(Mutex::new(Sha256::new()));
    let stream_hasher = Arc::clone(&encoded_hasher);
    let stream = response.bytes_stream().map(move |chunk| match chunk {
        Ok(chunk) => {
            stream_hasher
                .lock()
                .expect("encoded hash mutex poisoned")
                .update(&chunk);
            stream_downloaded.fetch_add(chunk.len() as u64, Ordering::Relaxed);
            Ok(chunk)
        }
        Err(source) => Err(std::io::Error::other(source)),
    });
    let reader = StreamReader::new(stream);
    let mut decoder = match encoded.encoding {
        Encoding::Zstd => ZstdDecoder::new(tokio::io::BufReader::new(reader)),
    };
    let mut output = tokio::fs::File::create(temp)
        .await
        .map_err(|source| io("create", temp, source))?;
    let mut decoded_hasher = Sha256::new();
    let mut decoded = 0u64;
    let mut reported = 0u64;
    let mut buffer = vec![0u8; IO_BUFFER_BYTES];
    loop {
        let read = decoder
            .read(&mut buffer)
            .await
            .map_err(|source| io("decompress into", temp, source))?;
        if read == 0 {
            break;
        }
        let next_decoded = decoded
            .checked_add(read as u64)
            .ok_or_else(|| Error::Size {
                name: spec.label.into(),
                expected: spec.decoded_bytes,
                actual: u64::MAX,
            })?;
        // Reject the decoded chunk before writing it when it would exceed the
        // canonical size. This bounds temporary disk usage as well as the work
        // performed on a wrong or malicious frame.
        if next_decoded > spec.decoded_bytes {
            return Err(Error::Size {
                name: spec.label.into(),
                expected: spec.decoded_bytes,
                actual: next_decoded,
            });
        }
        output
            .write_all(&buffer[..read])
            .await
            .map_err(|source| io("write", temp, source))?;
        decoded_hasher.update(&buffer[..read]);
        decoded = next_decoded;
        if decoded.saturating_sub(reported) >= 4 * 1024 * 1024 {
            report(TransferEvent::Progress(TransferProgress {
                downloaded: downloaded.load(Ordering::Relaxed),
                total: encoded.bytes,
                written: decoded,
            }));
            reported = decoded;
        }
    }
    drop(decoder);
    report(TransferEvent::Progress(TransferProgress {
        downloaded: downloaded.load(Ordering::Relaxed),
        total: encoded.bytes,
        written: decoded,
    }));
    output
        .sync_all()
        .await
        .map_err(|source| io("sync", temp, source))?;
    drop(output);

    let downloaded = downloaded.load(Ordering::Relaxed);
    if downloaded != encoded.bytes {
        return Err(Error::Size {
            name: encoded_label(),
            expected: encoded.bytes,
            actual: downloaded,
        });
    }
    let encoded_actual = lowercase_hex(
        &encoded_hasher
            .lock()
            .expect("encoded hash mutex poisoned")
            .clone()
            .finalize(),
    );
    if encoded_actual != encoded.sha256 {
        return Err(Error::Checksum {
            name: encoded_label(),
            expected: encoded.sha256.into(),
            actual: encoded_actual,
        });
    }
    if decoded != spec.decoded_bytes {
        return Err(Error::Size {
            name: spec.label.into(),
            expected: spec.decoded_bytes,
            actual: decoded,
        });
    }
    let decoded_actual = lowercase_hex(&decoded_hasher.finalize());
    if decoded_actual != spec.decoded_sha256 {
        return Err(Error::Checksum {
            name: spec.label.into(),
            expected: spec.decoded_sha256.into(),
            actual: decoded_actual,
        });
    }
    Ok(())
}

async fn get(client: &reqwest::Client, url: &str) -> Result<reqwest::Response> {
    let response = client.get(url).send().await.map_err(|source| Error::Http {
        url: url.into(),
        source,
    })?;
    if !response.status().is_success() {
        return Err(Error::HttpStatus {
            url: url.into(),
            status: response.status().as_u16(),
        });
    }
    Ok(response)
}

/// Verify that the file at `path` has exactly `bytes` length and hashes to
/// `sha256`. `label` names the artifact in any resulting error.
///
/// SHA-256 over a multi-gigabyte artifact is CPU-bound, so the read-and-hash
/// loop runs on the blocking pool rather than the async runtime thread. This
/// keeps the caller's runtime free to report progress and to hash other
/// artifacts concurrently.
pub async fn verify_file(path: &Path, bytes: u64, sha256: &str, label: &str) -> Result<()> {
    // Fail fast on a size mismatch without reading a whole wrong-sized file.
    let metadata = tokio::fs::metadata(path)
        .await
        .map_err(|source| io("read metadata for", path, source))?;
    if metadata.len() != bytes {
        return Err(Error::Size {
            name: label.into(),
            expected: bytes,
            actual: metadata.len(),
        });
    }

    let owned = path.to_path_buf();
    let digest = tokio::task::spawn_blocking(move || hash_file_blocking(&owned))
        .await
        .map_err(|error| Error::BackgroundTask(error.to_string()))??;
    let actual = lowercase_hex(&digest);
    if actual != sha256 {
        return Err(Error::Checksum {
            name: label.into(),
            expected: sha256.into(),
            actual,
        });
    }
    Ok(())
}

/// Synchronously read `path` in `IO_BUFFER_BYTES` chunks and return its raw
/// SHA-256 digest. Runs inside `spawn_blocking`; uses `std::fs` so the blocking
/// pool thread — not an async worker — bears the file I/O and hashing.
fn hash_file_blocking(path: &Path) -> Result<[u8; 32]> {
    use std::io::Read;
    let mut input = std::fs::File::open(path).map_err(|source| io("open", path, source))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; IO_BUFFER_BYTES];
    loop {
        let read = input
            .read(&mut buffer)
            .map_err(|source| io("read", path, source))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().into())
}

/// Cheap cache-hit probe: is `path` a regular file of exactly `bytes` length?
/// Immutable content-addressed artifacts need no re-hash on the hot path.
pub async fn size_matches(path: &Path, bytes: u64) -> Result<bool> {
    match tokio::fs::metadata(path).await {
        Ok(metadata) => Ok(metadata.is_file() && metadata.len() == bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(source) => Err(io("read metadata for", path, source)),
    }
}

/// Atomically move a verified temporary file over its final path.
pub async fn replace_file(temp: &Path, target: &Path) -> Result<()> {
    tokio::fs::rename(temp, target)
        .await
        .map_err(|source| io("rename", temp, source))
}

/// Whether `a` and `b` resolve to the same underlying file (one inode reached
/// through two names). For an immutable, previously verified cache object this
/// proves the other name holds identical bytes without reading either file.
///
/// Only implemented on Unix, where `dev`+`ino` are available on stable. On other
/// platforms (notably Windows, whose file-identity metadata is still unstable —
/// `windows_by_handle`, rust-lang/rust#63010) it conservatively returns `false`,
/// so callers reinstall rather than skip. That reinstall re-links the verified
/// cache object without re-hashing, so the cost is a cheap unlink + hard link.
pub async fn is_same_file(a: &Path, b: &Path) -> Result<bool> {
    let a_meta = tokio::fs::metadata(a)
        .await
        .map_err(|source| io("read metadata for", a, source))?;
    let b_meta = tokio::fs::metadata(b)
        .await
        .map_err(|source| io("read metadata for", b, source))?;
    Ok(same_identity(&a_meta, &b_meta))
}

#[cfg(unix)]
fn same_identity(a: &std::fs::Metadata, b: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    a.dev() == b.dev() && a.ino() == b.ino()
}

#[cfg(not(unix))]
fn same_identity(_a: &std::fs::Metadata, _b: &std::fs::Metadata) -> bool {
    false
}

/// How [`hardlink_or_copy`] installed an artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Installed {
    /// A hard link was created. The installed name shares the source's inode,
    /// so it holds the source's already-verified bytes with nothing to re-check.
    Linked,
    /// The bytes were copied into an independent file. The copy is fresh and
    /// unverified — the caller should re-hash it if integrity matters.
    Copied,
}

/// Install an already-verified cache object at `target`, reporting whether it
/// was linked or copied. Prefers a hard link — no bytes copied and no re-hash —
/// and falls back to a byte copy when linking fails for any reason (a different
/// filesystem, or one that does not support hard links). A hard link requires
/// `target` to be absent; the copy fallback overwrites it. Because the source
/// is a content-addressed, previously verified artifact, a [`Installed::Linked`]
/// result inherits its integrity, while a [`Installed::Copied`] result does not.
pub async fn hardlink_or_copy(source: &Path, target: &Path) -> Result<Installed> {
    if tokio::fs::hard_link(source, target).await.is_ok() {
        return Ok(Installed::Linked);
    }
    tokio::fs::copy(source, target)
        .await
        .map(|_| Installed::Copied)
        .map_err(|error| io("copy cached artifact to", target, error))
}

/// Monotonic per-process counter for unique temporary filenames.
pub fn next_sequence() -> u64 {
    TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
}

/// Lowercase hex encoding of `bytes`.
pub fn lowercase_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

/// Lowercase hex SHA-256 of an in-memory payload.
pub fn sha256_hex(bytes: &[u8]) -> String {
    lowercase_hex(&Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn hardlink_or_copy_links_a_clean_target() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("src");
        let target = dir.path().join("dst");
        std::fs::write(&source, b"payload").unwrap();

        assert_eq!(
            hardlink_or_copy(&source, &target).await.unwrap(),
            Installed::Linked
        );
        assert_eq!(std::fs::read(&target).unwrap(), b"payload");
        // The link shares the source inode and therefore its bytes. Identity is
        // only observable on Unix; see `is_same_file`.
        #[cfg(unix)]
        assert!(is_same_file(&source, &target).await.unwrap());
    }

    #[tokio::test]
    async fn hardlink_or_copy_falls_back_to_copy_when_link_fails() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("src");
        let target = dir.path().join("dst");
        std::fs::write(&source, b"payload").unwrap();
        // A pre-existing target makes `hard_link` fail; the copy fallback
        // overwrites it and reports an independent, unshared file.
        std::fs::write(&target, b"stale contents").unwrap();

        assert_eq!(
            hardlink_or_copy(&source, &target).await.unwrap(),
            Installed::Copied
        );
        assert!(!is_same_file(&source, &target).await.unwrap());
        assert_eq!(std::fs::read(&target).unwrap(), b"payload");
    }
}
