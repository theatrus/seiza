//! SHA-256 digests: of a file's bytes, which name the file wherever it is,
//! and of what a step reads, which is the step's key.

use crate::{Error, Result};
use sha2::{Digest as _, Sha256};
use std::io::Read;
use std::path::Path;

/// A SHA-256 digest, written as 64 lowercase hexadecimal digits.
///
/// A file is named by the digest of its bytes, not by its path, size and
/// modification time, because two machines can hold the same frame at
/// different paths. A step is named by the digest of what it reads, so
/// anyone with the same inputs computes the same key.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Digest([u8; 32]);

impl Digest {
    /// The digest of `bytes`.
    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    /// The digest of the file at `path`'s bytes, read in 1 MiB pieces.
    pub fn of_file(path: &Path) -> Result<Self> {
        let io = |source| Error::Io {
            path: path.to_path_buf(),
            source,
        };
        let mut file = std::fs::File::open(path).map_err(io)?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0_u8; 1 << 20];
        loop {
            let read = file.read(&mut buffer).map_err(io)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        Ok(Self(hasher.finalize().into()))
    }

    /// The digest's 32 bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl std::fmt::Display for Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Digest({self})")
    }
}

/// Why a string is not a [`Digest`].
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("a digest is 64 hexadecimal digits")]
pub struct DigestParseError;

impl std::str::FromStr for Digest {
    type Err = DigestParseError;

    fn from_str(text: &str) -> std::result::Result<Self, Self::Err> {
        let text = text.as_bytes();
        if text.len() != 64 {
            return Err(DigestParseError);
        }
        let digit = |byte: u8| (byte as char).to_digit(16).ok_or(DigestParseError);
        let mut bytes = [0_u8; 32];
        for (byte, &[high, low]) in bytes.iter_mut().zip(text.as_chunks::<2>().0) {
            *byte = (digit(high)? << 4 | digit(low)?) as u8;
        }
        Ok(Self(bytes))
    }
}

impl serde::Serialize for Digest {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> serde::Deserialize<'de> for Digest {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

/// Builds a digest from parts, each written with its length or tag, so
/// that no two different sequences of parts hash alike.
pub(crate) struct KeyHasher(Sha256);

impl KeyHasher {
    /// A hasher for one kind of key, named by `domain`.
    pub(crate) fn new(domain: &str) -> Self {
        let mut hasher = Self(Sha256::new());
        hasher.text(domain);
        hasher
    }

    pub(crate) fn bytes(&mut self, bytes: &[u8]) -> &mut Self {
        self.0.update((bytes.len() as u64).to_le_bytes());
        self.0.update(bytes);
        self
    }

    pub(crate) fn text(&mut self, text: &str) -> &mut Self {
        self.bytes(text.as_bytes())
    }

    pub(crate) fn digest(&mut self, digest: &Digest) -> &mut Self {
        self.0.update(digest.0);
        self
    }

    pub(crate) fn optional_digest(&mut self, digest: Option<&Digest>) -> &mut Self {
        match digest {
            Some(digest) => {
                self.0.update([1]);
                self.digest(digest)
            }
            None => {
                self.0.update([0]);
                self
            }
        }
    }

    pub(crate) fn number(&mut self, value: u64) -> &mut Self {
        self.0.update(value.to_le_bytes());
        self
    }

    /// An optional float, by its bits, so `0.0` and `-0.0` differ.
    pub(crate) fn optional_float(&mut self, value: Option<f64>) -> &mut Self {
        match value {
            Some(value) => {
                self.0.update([1]);
                self.0.update(value.to_bits().to_le_bytes());
            }
            None => self.0.update([0]),
        }
        self
    }

    pub(crate) fn finish(&mut self) -> Digest {
        Digest(std::mem::take(&mut self.0).finalize().into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digests_print_and_parse_as_hex() {
        let digest = Digest::of_bytes(b"abc");
        let text = digest.to_string();
        assert_eq!(
            text,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(text.parse::<Digest>(), Ok(digest));
        assert_eq!(text.to_uppercase().parse::<Digest>(), Ok(digest));
        assert!("ba78".parse::<Digest>().is_err());
        assert!(text.replace('b', "g").parse::<Digest>().is_err());
        assert_eq!(
            serde_json::to_string(&digest).unwrap(),
            format!("\"{text}\"")
        );
    }

    #[test]
    fn a_file_digest_is_its_bytes_digest() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("frame.fits");
        let bytes = (0..3_000_000_u32)
            .map(|value| value as u8)
            .collect::<Vec<_>>();
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(Digest::of_file(&path).unwrap(), Digest::of_bytes(&bytes));
        assert!(matches!(
            Digest::of_file(&directory.path().join("missing")),
            Err(Error::Io { .. })
        ));
    }

    #[test]
    fn parts_are_delimited() {
        let key = |parts: &[&str]| {
            let mut hasher = KeyHasher::new("test");
            for part in parts {
                hasher.text(part);
            }
            hasher.finish()
        };
        assert_ne!(key(&["ab", "c"]), key(&["a", "bc"]));
        assert_eq!(key(&["ab", "c"]), key(&["ab", "c"]));
    }
}
