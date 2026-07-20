use anyhow::{Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Serialize)]
pub(crate) struct FileIdentity {
    pub(crate) path: String,
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
}

pub(crate) fn paths_refer_to_same_file(left: &Path, right: &Path) -> bool {
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

pub(crate) fn file_identity(path: &Path) -> Result<FileIdentity> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("failed to fingerprint {}", path.display()))?;
    let bytes = file.metadata()?.len();
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    Ok(FileIdentity {
        path: path.display().to_string(),
        bytes,
        sha256: digest.iter().map(|byte| format!("{byte:02x}")).collect(),
    })
}

pub(crate) fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let prefix = format!(
        ".{}.",
        path.file_name().unwrap_or_default().to_string_lossy()
    );
    let mut builder = tempfile::Builder::new();
    builder.prefix(&prefix);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let permissions = std::fs::metadata(path)
            .map(|metadata| metadata.permissions())
            .unwrap_or_else(|_| std::fs::Permissions::from_mode(0o666));
        builder.permissions(permissions);
    }
    let mut temporary = builder
        .tempfile_in(parent)
        .with_context(|| format!("failed to create report beside {}", path.display()))?;
    serde_json::to_writer_pretty(temporary.as_file_mut(), value)?;
    temporary.as_file_mut().write_all(b"\n")?;
    temporary.as_file_mut().flush()?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("failed to publish report {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collision_detection_resolves_parent_components() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir(directory.path().join("child")).unwrap();
        let direct = directory.path().join("output.fits");
        let aliased = directory.path().join("child/../output.fits");
        assert!(paths_refer_to_same_file(&direct, &aliased));
    }
}
