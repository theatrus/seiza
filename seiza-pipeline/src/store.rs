//! Where step outputs live, each under its step's key.

use crate::{Digest, Error, PartialStack, Result};
use std::path::{Path, PathBuf};

/// A directory of step outputs, each under its step's key:
/// `steps/<first two digits>/<key>/<name>`.
///
/// A file is written once and never replaced: a key already present is
/// kept, so two machines that run one step write the same key and the first
/// copy stays. Nothing indexes the store; what it holds is read from its
/// files' cards. A local directory, a NAS or a synced folder serve alike,
/// and merging two stores is copying across the keys one lacks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Store {
    root: PathBuf,
}

impl Store {
    /// The store at `root`, which [`Self::put`] creates as it needs.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The store's directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Where the step with `key` keeps the file it calls `name`.
    pub fn path(&self, key: &Digest, name: &str) -> PathBuf {
        let key = key.to_string();
        self.root
            .join("steps")
            .join(&key[..2])
            .join(&key)
            .join(name)
    }

    /// The file the step with `key` wrote as `name`, when the store holds
    /// it. Only the header is read. A file there whose cards carry another
    /// key is [`Error::Invalid`]: a store never holds one.
    pub fn find(&self, key: &Digest, name: &str) -> Result<Option<PathBuf>> {
        let path = self.path(key, name);
        if !path.is_file() {
            return Ok(None);
        }
        let identity = PartialStack::read_identity(&path)?;
        if identity.step_key != *key {
            return Err(Error::Invalid(format!(
                "{} carries step key {}, not the key it is stored under",
                path.display(),
                identity.step_key
            )));
        }
        Ok(Some(path))
    }

    /// Write `stack` as `name` under its own step key, unless the store
    /// holds it already, and return its path.
    ///
    /// The file is written beside its place under a hidden name, then linked
    /// into place, which fails if another writer got there first; that copy
    /// is kept. Where links are not supported the file is renamed into
    /// place, which another writer racing this one could replace with its
    /// own copy of the same step.
    pub fn put(&self, stack: &PartialStack, name: &str) -> Result<PathBuf> {
        let key = &stack.identity.step_key;
        if let Some(path) = self.find(key, name)? {
            return Ok(path);
        }
        self.put_file(key, name, |path| stack.write(path))
    }

    /// Write a file of the step with `key` as `name`, by `write` given the
    /// path to write, as [`Self::put`] writes a stack: under a hidden name,
    /// then linked into place, keeping a copy already there. Returns its
    /// path.
    pub(crate) fn put_file(
        &self,
        key: &Digest,
        name: &str,
        write: impl FnOnce(&Path) -> Result<()>,
    ) -> Result<PathBuf> {
        let path = self.path(key, name);
        if path.is_file() {
            return Ok(path);
        }
        let directory = path.parent().expect("a step's file has a directory");
        std::fs::create_dir_all(directory).map_err(|source| Error::Io {
            path: directory.to_path_buf(),
            source,
        })?;
        let temporary = directory.join(format!(
            ".{}-{}.{name}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_nanos())
        ));
        if let Err(error) = write(&temporary) {
            let _ = std::fs::remove_file(&temporary);
            return Err(error);
        }
        let placed = match std::fs::hard_link(&temporary, &path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
            Err(_) if !path.exists() => std::fs::rename(&temporary, &path),
            Err(_) => Ok(()),
        };
        let _ = std::fs::remove_file(&temporary);
        placed.map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stack_is_kept_under_its_key_and_never_replaced() {
        let directory = tempfile::tempdir().unwrap();
        let store = Store::new(directory.path().join("store"));
        let stack = crate::partial::tests::stack(1, 4);
        let key = stack.identity.step_key;
        assert_eq!(store.find(&key, "partial.fits").unwrap(), None);
        let path = store.put(&stack, "partial.fits").unwrap();
        assert!(path.ends_with(format!(
            "steps/{}/{key}/partial.fits",
            &key.to_string()[..2]
        )));
        assert_eq!(
            store.find(&key, "partial.fits").unwrap(),
            Some(path.clone())
        );

        // The first copy stays.
        let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
        let mut changed = stack.clone();
        changed.mean.data[0] += 1.0;
        assert_eq!(store.put(&changed, "partial.fits").unwrap(), path);
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            modified
        );
        crate::partial::tests::assert_same(&PartialStack::open(&path).unwrap(), &stack);
        // No temporary file is left behind.
        assert_eq!(
            std::fs::read_dir(path.parent().unwrap()).unwrap().count(),
            1
        );

        // A file under the wrong key is an error, not a hit.
        let other = Digest::of_bytes(b"another step");
        let misplaced = store.path(&other, "partial.fits");
        std::fs::create_dir_all(misplaced.parent().unwrap()).unwrap();
        std::fs::copy(&path, &misplaced).unwrap();
        assert!(matches!(
            store.find(&other, "partial.fits"),
            Err(Error::Invalid(_))
        ));
    }
}
