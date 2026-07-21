use std::path::{Path, PathBuf};

/// Return whether two paths resolve to the same filesystem entry or target.
///
/// Existing files are canonicalized. For a not-yet-created output, the parent
/// directory is canonicalized so aliases such as `child/../output.fits` are
/// still detected.
pub fn paths_refer_to_same_file(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }
    match (comparable_path(left), comparable_path(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
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
    fn resolves_parent_components_for_a_future_output() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir(directory.path().join("child")).unwrap();
        let direct = directory.path().join("output.fits");
        let aliased = directory.path().join("child/../output.fits");
        assert!(paths_refer_to_same_file(&direct, &aliased));
    }
}
