use anyhow::{Context, Result};
use seiza_stacking::paths_refer_to_same_file;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::Path;

#[derive(Serialize)]
pub(crate) struct FileIdentity {
    pub(crate) path: String,
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
}

pub(crate) fn validate_path_roles<'a>(
    entries: impl IntoIterator<Item = (String, &'a Path)>,
) -> Result<()> {
    let entries = entries.into_iter().collect::<Vec<_>>();
    for (index, (role, path)) in entries.iter().enumerate() {
        if let Some((other_role, other_path)) = entries[..index]
            .iter()
            .find(|(_, other_path)| paths_refer_to_same_file(path, other_path))
        {
            anyhow::bail!(
                "{role} {} refers to the same file as {other_role} {}",
                path.display(),
                other_path.display()
            );
        }
    }
    Ok(())
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

    #[test]
    fn path_roles_reject_aliases() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("light.fits");
        std::fs::write(&path, b"pixels").unwrap();
        let error = validate_path_roles([
            ("light frame 1".into(), path.as_path()),
            ("light frame 2".into(), path.as_path()),
        ])
        .unwrap_err();
        assert!(error.to_string().contains("light frame 2"));
    }
}
