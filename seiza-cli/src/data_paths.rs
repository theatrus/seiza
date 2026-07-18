//! Catalog and index path resolution shared by every command and
//! compatibility mode.
//!
//! Each resolver accepts what the user gave it: an explicit file, a
//! directory to search (`--data <dir>` picks the right file inside), or
//! nothing at all — in which case the standard locations are searched:
//! the kind's environment variable, a `seiza.toml` or data file next to
//! the executable, then the shared catalog directories used by
//! `seiza setup` (`SEIZA_CATALOG_DIR` and the platform data dirs).

use anyhow::Result;
use std::path::{Path, PathBuf};

/// Solver star tile catalogs, preferring the deepest available.
const STAR_DATA_NAMES: &[&str] = &[
    "stars-deep-gaia17.bin",
    "stars-gaia.bin",
    "stars-lite-tycho2.bin",
    "stars.bin",
];

pub(crate) fn star_data(arg: Option<&Path>) -> Result<PathBuf> {
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
pub(crate) fn blind_index(arg: Option<&Path>) -> Result<Option<PathBuf>> {
    match arg {
        Some(_) => resolve(
            arg,
            "blind index",
            Some("SEIZA_BLIND_INDEX"),
            Some("blind_index"),
            &["blind-gaia16.idx"],
            Some("idx"),
        )
        .map(Some),
        None => Ok(resolve(
            None,
            "blind index",
            Some("SEIZA_BLIND_INDEX"),
            Some("blind_index"),
            &["blind-gaia16.idx"],
            Some("idx"),
        )
        .ok()),
    }
}

pub(crate) fn objects(arg: Option<&Path>) -> Result<PathBuf> {
    resolve(arg, "object catalog", None, None, &["objects.bin"], None)
}

pub(crate) fn star_identifiers(arg: Option<&Path>) -> Result<PathBuf> {
    resolve(
        arg,
        "star identifier sidecar",
        None,
        None,
        &["stars-lite-tycho2.ids.bin"],
        Some("ids.bin"),
    )
}

pub(crate) fn minor_bodies(arg: Option<&Path>) -> Result<PathBuf> {
    resolve(
        arg,
        "minor-body catalog",
        None,
        None,
        &["minor-bodies.bin"],
        None,
    )
}

fn resolve(
    arg: Option<&Path>,
    kind: &str,
    env: Option<&str>,
    toml_key: Option<&str>,
    names: &[&str],
    extension: Option<&str>,
) -> Result<PathBuf> {
    if let Some(path) = arg {
        if path.is_dir() {
            return find_in_dir(path, names, extension).ok_or_else(|| {
                anyhow::anyhow!(
                    "no {kind} found in {} (expected one of: {})",
                    path.display(),
                    names.join(", ")
                )
            });
        }
        if path.exists() {
            return Ok(path.to_path_buf());
        }
        anyhow::bail!("{kind} {} does not exist", path.display());
    }

    if let Some(env) = env
        && let Ok(path) = std::env::var(env)
    {
        let path = PathBuf::from(path);
        if path.is_dir() {
            if let Some(found) = find_in_dir(&path, names, extension) {
                return Ok(found);
            }
        } else if path.exists() {
            return Ok(path);
        }
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
    anyhow::bail!(
        "no {kind} found; pass --data (a file or a directory), run `seiza setup`, \
         or run: seiza download-data prebuilt --output <dir> (https://downloads.seiza.fyi)"
    )
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
pub(crate) fn search_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![crate::setup::default_catalog_dir()];
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

fn dirs_data_dir() -> Option<PathBuf> {
    std::env::var("XDG_DATA_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|home| PathBuf::from(home).join(".local/share"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_search_starts_with_setup_directory() {
        assert_eq!(
            search_dirs().first(),
            Some(&crate::setup::default_catalog_dir())
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
    fn missing_paths_error_and_empty_directories_explain_expectations() {
        let dir = tempfile::tempdir().unwrap();
        let error = star_data(Some(&dir.path().join("nope.bin"))).unwrap_err();
        assert!(error.to_string().contains("does not exist"));
        let error = objects(Some(dir.path())).unwrap_err();
        assert!(error.to_string().contains("objects.bin"), "{error}");
        // An explicit missing index errors; an omitted one is simply absent.
        assert!(blind_index(Some(&dir.path().join("no.idx"))).is_err());
    }
}
