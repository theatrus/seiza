use std::path::{Path, PathBuf};

/// Return whether two paths resolve to the same filesystem entry or target.
///
/// Existing files are compared by filesystem identity, including hard links.
/// If either identity cannot be read, canonical paths are compared instead.
/// For a not-yet-created output, the parent directory is canonicalized so
/// aliases such as `child/../output.fits` are still detected.
pub fn paths_refer_to_same_file(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }
    if let Ok(same) = same_file::is_same_file(left, right) {
        return same;
    }
    match (comparable_path(left), comparable_path(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

/// Return a canonical path key, falling back to the literal path when
/// canonicalization fails. Callers can dedup path spellings with one
/// canonicalization each instead of one per pair. Hard links retain distinct
/// keys; use [`paths_refer_to_same_file`] to compare existing files' identities.
pub fn path_identity(path: &Path) -> PathBuf {
    comparable_path(path).unwrap_or_else(|_| path.to_path_buf())
}

fn comparable_path(path: &Path) -> std::io::Result<PathBuf> {
    if let Ok(path) = std::fs::canonicalize(path) {
        return Ok(path);
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = std::fs::canonicalize(parent)?;
    Ok(parent.join(path.file_name().unwrap_or_default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_links_refer_to_the_same_file_but_identical_copies_do_not() {
        let directory = tempfile::tempdir().unwrap();
        let original = directory.path().join("original.txt");
        let linked = directory.path().join("linked.txt");
        let copied = directory.path().join("copied.txt");
        std::fs::write(&original, b"same contents").unwrap();
        std::fs::hard_link(&original, &linked).unwrap();
        std::fs::copy(&original, &copied).unwrap();
        assert!(paths_refer_to_same_file(&original, &linked));
        assert!(paths_refer_to_same_file(&linked, &original));
        assert!(!paths_refer_to_same_file(&original, &copied));
        assert!(!paths_refer_to_same_file(&linked, &copied));
    }

    #[test]
    fn resolves_parent_components_for_a_future_output() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir(directory.path().join("child")).unwrap();
        let direct = directory.path().join("output.fits");
        let aliased = directory.path().join("child/../output.fits");
        assert!(paths_refer_to_same_file(&direct, &aliased));
    }
}
