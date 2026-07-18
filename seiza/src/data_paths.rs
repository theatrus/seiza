//! Catalog and index path resolution shared by the CLI, the
//! compatibility modes, and embedding applications.
//!
//! Each resolver accepts what the caller has:
//!
//! - **A file** — used as given.
//! - **A directory** — the right file inside is picked: the deepest
//!   star catalog present, `objects.bin`, `minor-bodies.bin`,
//!   `transients.bin`, any `.idx` blind index, any `.ids.bin` star
//!   identifier sidecar.
//! - **Nothing** — the standard places are checked in order: the
//!   kind's environment variable (`SEIZA_STAR_DATA`,
//!   `SEIZA_BLIND_INDEX`; each takes a file or a directory, and a set
//!   variable that resolves to nothing is an error, not a fallback), a
//!   `seiza.toml` or data file next to the executable, then the shared
//!   catalog directories `seiza setup` installs into ([`CATALOG_DIR_ENV`]
//!   and the platform data dirs).
//!
//! An application that keeps every catalog in one directory needs a
//! single configured path:
//!
//! ```no_run
//! use std::path::Path;
//!
//! let dir = Some(Path::new("/var/lib/seiza/catalogs"));
//! let stars = seiza::data_paths::star_data(dir)?;
//! let objects = seiza::data_paths::objects(dir)?;
//! let transients = seiza::data_paths::transients(dir)?;
//! # Ok::<(), seiza::data_paths::DataPathError>(())
//! ```

use directories::ProjectDirs;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Environment variable naming the shared catalog directory that
/// `seiza setup` installs into.
pub const CATALOG_DIR_ENV: &str = "SEIZA_CATALOG_DIR";

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DataPathError {
    #[error("no {kind} found in {} (expected one of: {expected})", path.display())]
    NotFoundInDirectory {
        kind: &'static str,
        path: PathBuf,
        expected: String,
    },
    #[error("{kind} {} does not exist", path.display())]
    Missing { kind: &'static str, path: PathBuf },
    #[error("{var} is set to {} but no {kind} was found there", path.display())]
    EnvVar {
        kind: &'static str,
        var: &'static str,
        path: PathBuf,
    },
    #[error(
        "no {kind} found; pass a file or a directory, run `seiza setup`, \
         or run: seiza download-data prebuilt --output <dir> (https://downloads.seiza.fyi)"
    )]
    NoDefault { kind: &'static str },
}

/// Solver star tile catalogs, preferring the deepest available.
const STAR_DATA_NAMES: &[&str] = &[
    "stars-deep-gaia17.bin",
    "stars-gaia.bin",
    "stars-lite-tycho2.bin",
    "stars.bin",
];

pub fn star_data(arg: Option<&Path>) -> Result<PathBuf, DataPathError> {
    resolve(
        arg,
        "star catalog",
        Some("SEIZA_STAR_DATA"),
        Some("star_data"),
        STAR_DATA_NAMES,
        None,
    )
}

/// Prebuilt blind pattern index. Optional: `Ok(None)` means "none found,
/// build in memory"; an explicitly given path that resolves to nothing is
/// an error.
pub fn blind_index(arg: Option<&Path>) -> Result<Option<PathBuf>, DataPathError> {
    let result = resolve(
        arg,
        "blind index",
        Some("SEIZA_BLIND_INDEX"),
        Some("blind_index"),
        &["blind-gaia16.idx"],
        Some("idx"),
    );
    match arg {
        Some(_) => result.map(Some),
        // Omitted and absent means "build the index in memory" — but a set
        // env var that resolves nothing is still an error, never a fallback.
        None => match result {
            Err(DataPathError::NoDefault { .. }) => Ok(None),
            other => other.map(Some),
        },
    }
}

pub fn objects(arg: Option<&Path>) -> Result<PathBuf, DataPathError> {
    resolve(arg, "object catalog", None, None, &["objects.bin"], None)
}

pub fn star_identifiers(arg: Option<&Path>) -> Result<PathBuf, DataPathError> {
    resolve(
        arg,
        "star identifier sidecar",
        None,
        None,
        &["stars-lite-tycho2.ids.bin"],
        Some("ids.bin"),
    )
}

pub fn minor_bodies(arg: Option<&Path>) -> Result<PathBuf, DataPathError> {
    resolve(
        arg,
        "minor-body catalog",
        None,
        None,
        &["minor-bodies.bin"],
        None,
    )
}

pub fn transients(arg: Option<&Path>) -> Result<PathBuf, DataPathError> {
    resolve(
        arg,
        "transient catalog",
        None,
        None,
        &["transients.bin"],
        None,
    )
}

fn resolve(
    arg: Option<&Path>,
    kind: &'static str,
    env: Option<&'static str>,
    toml_key: Option<&str>,
    names: &[&str],
    extension: Option<&str>,
) -> Result<PathBuf, DataPathError> {
    if let Some(path) = arg {
        if path.is_dir() {
            return find_in_dir(path, names, extension).ok_or_else(|| {
                DataPathError::NotFoundInDirectory {
                    kind,
                    path: path.to_path_buf(),
                    expected: names.join(", "),
                }
            });
        }
        if path.exists() {
            return Ok(path.to_path_buf());
        }
        return Err(DataPathError::Missing {
            kind,
            path: path.to_path_buf(),
        });
    }

    if let Some(var) = env
        && let Some(value) = std::env::var_os(var).filter(|value| !value.is_empty())
    {
        return env_candidate(kind, var, &PathBuf::from(value), names, extension);
    }

    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        if let Some(key) = toml_key
            && let Ok(content) = std::fs::read_to_string(dir.join("seiza.toml"))
        {
            for line in content.lines() {
                if let Some((name, value)) = line.split_once('=')
                    && name.trim() == key
                {
                    let path = PathBuf::from(value.trim().trim_matches('"'));
                    if path.exists() {
                        return Ok(path);
                    }
                }
            }
        }
        if let Some(found) = find_in_dir(dir, names, extension) {
            return Ok(found);
        }
    }

    for base in search_dirs() {
        if let Some(found) = find_in_dir(&base, names, extension) {
            return Ok(found);
        }
    }
    Err(DataPathError::NoDefault { kind })
}

/// A set environment variable is a pinned choice: when it resolves to
/// nothing, that is an error — never a silent fallback to other catalogs.
fn env_candidate(
    kind: &'static str,
    var: &'static str,
    path: &Path,
    names: &[&str],
    extension: Option<&str>,
) -> Result<PathBuf, DataPathError> {
    if path.is_dir() {
        if let Some(found) = find_in_dir(path, names, extension) {
            return Ok(found);
        }
    } else if path.exists() {
        return Ok(path.to_path_buf());
    }
    Err(DataPathError::EnvVar {
        kind,
        var,
        path: path.to_path_buf(),
    })
}

/// Search one directory: exact names in priority order, then any file
/// with the fallback extension (e.g. `.idx`, `.ids.bin`).
fn find_in_dir(dir: &Path, names: &[&str], extension: Option<&str>) -> Option<PathBuf> {
    for name in names {
        let path = dir.join(name);
        if path.exists() {
            return Some(path);
        }
    }
    let suffix = format!(".{}", extension?);
    let mut matches: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.ends_with(&suffix))
        })
        .collect();
    matches.sort();
    matches.into_iter().next()
}

/// The shared catalog directories `seiza setup` installs into.
pub fn search_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![default_catalog_dir()];
    for base in [
        std::env::var("LOCALAPPDATA").ok().map(PathBuf::from),
        dirs_data_dir(),
    ]
    .into_iter()
    .flatten()
    .map(|base| base.join("seiza"))
    {
        if !dirs.contains(&base) {
            dirs.push(base);
        }
    }
    dirs
}

/// The directory `seiza setup` installs into by default:
/// [`CATALOG_DIR_ENV`] when set, else the platform data directory.
pub fn default_catalog_dir() -> PathBuf {
    configured_catalog_dir(std::env::var_os(CATALOG_DIR_ENV)).unwrap_or_else(|| {
        ProjectDirs::from("fyi", "Seiza", "seiza")
            .map(|dirs| dirs.data_local_dir().join("catalogs"))
            .unwrap_or_else(|| PathBuf::from("seiza-data"))
    })
}

fn configured_catalog_dir(value: Option<OsString>) -> Option<PathBuf> {
    value.filter(|value| !value.is_empty()).map(PathBuf::from)
}

fn dirs_data_dir() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|dirs| dirs.data_local_dir().to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_search_starts_with_setup_directory() {
        assert_eq!(search_dirs().first(), Some(&default_catalog_dir()));
    }

    #[test]
    fn configured_catalog_directory_ignores_empty_values() {
        assert_eq!(configured_catalog_dir(None), None);
        assert_eq!(configured_catalog_dir(Some(OsString::new())), None);
        assert_eq!(
            configured_catalog_dir(Some(OsString::from("shared-catalogs"))),
            Some(PathBuf::from("shared-catalogs"))
        );
    }

    #[test]
    fn directory_argument_picks_the_best_catalog() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("stars-lite-tycho2.bin"), b"lite").unwrap();
        let picked = star_data(Some(dir.path())).unwrap();
        assert!(picked.ends_with("stars-lite-tycho2.bin"));

        std::fs::write(dir.path().join("stars-gaia.bin"), b"gaia").unwrap();
        let picked = star_data(Some(dir.path())).unwrap();
        assert!(picked.ends_with("stars-gaia.bin"), "prefers deeper catalog");
    }

    #[test]
    fn directory_argument_resolves_index_objects_and_identifiers() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("custom-blind.idx"), b"idx").unwrap();
        std::fs::write(dir.path().join("objects.bin"), b"objects").unwrap();
        std::fs::write(dir.path().join("stars-lite-tycho2.ids.bin"), b"ids").unwrap();

        let index = blind_index(Some(dir.path())).unwrap().unwrap();
        assert!(index.ends_with("custom-blind.idx"), "extension fallback");
        assert!(objects(Some(dir.path())).unwrap().ends_with("objects.bin"));
        assert!(
            star_identifiers(Some(dir.path()))
                .unwrap()
                .ends_with("stars-lite-tycho2.ids.bin")
        );
    }

    #[test]
    fn directory_argument_resolves_sky_annotation_catalogs() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("minor-bodies.bin"), b"mb").unwrap();
        std::fs::write(dir.path().join("transients.bin"), b"tr").unwrap();

        assert!(
            minor_bodies(Some(dir.path()))
                .unwrap()
                .ends_with("minor-bodies.bin")
        );
        assert!(
            transients(Some(dir.path()))
                .unwrap()
                .ends_with("transients.bin")
        );
    }

    #[test]
    fn set_env_var_that_resolves_nothing_is_an_error_not_a_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let error = env_candidate(
            "star catalog",
            "SEIZA_STAR_DATA",
            &dir.path().join("typo.bin"),
            STAR_DATA_NAMES,
            None,
        )
        .unwrap_err();
        assert!(matches!(error, DataPathError::EnvVar { .. }));
        assert!(error.to_string().contains("SEIZA_STAR_DATA"), "{error}");

        // An empty directory behind the variable is just as much an error
        let error = env_candidate(
            "star catalog",
            "SEIZA_STAR_DATA",
            dir.path(),
            STAR_DATA_NAMES,
            None,
        )
        .unwrap_err();
        assert!(matches!(error, DataPathError::EnvVar { .. }));

        std::fs::write(dir.path().join("stars-gaia.bin"), b"gaia").unwrap();
        let found = env_candidate(
            "star catalog",
            "SEIZA_STAR_DATA",
            dir.path(),
            STAR_DATA_NAMES,
            None,
        )
        .unwrap();
        assert!(found.ends_with("stars-gaia.bin"));
    }

    #[test]
    fn missing_paths_error_and_empty_directories_explain_expectations() {
        let dir = tempfile::tempdir().unwrap();
        let error = star_data(Some(&dir.path().join("nope.bin"))).unwrap_err();
        assert!(error.to_string().contains("does not exist"));
        let error = objects(Some(dir.path())).unwrap_err();
        assert!(error.to_string().contains("objects.bin"), "{error}");
        let error = transients(Some(dir.path())).unwrap_err();
        assert!(error.to_string().contains("transients.bin"), "{error}");
        // An explicit missing index errors; an omitted one is simply absent.
        assert!(blind_index(Some(&dir.path().join("no.idx"))).is_err());
    }
}
