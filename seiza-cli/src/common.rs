use anyhow::{Context, Result};
use seiza_stacking::FitsFrame;
use std::fmt::Arguments;
use std::path::Path;

/// Compare the final path component of `program` (without `.exe`) with
/// `expected`, case-insensitively, accepting both separator styles
/// regardless of host platform. Compatibility modes use this to answer to
/// a copy of the binary installed under the name the host tool expects.
pub(crate) fn binary_name_matches(program: &str, expected: &str) -> bool {
    let name = program.rsplit(['/', '\\']).next().unwrap_or(program);
    let stem = name
        .strip_suffix(".exe")
        .or_else(|| name.strip_suffix(".EXE"))
        .unwrap_or(name);
    stem.eq_ignore_ascii_case(expected)
}

/// Open a FITS frame, tagging any read error with the caller's role.
pub(crate) fn open_frame(path: &Path, role: &str) -> Result<FitsFrame> {
    FitsFrame::open(path).with_context(|| format!("failed to read {role} {}", path.display()))
}

/// Report a written artifact on stdout as `wrote {path}: {summary}`.
pub(crate) fn wrote(path: &Path, summary: Arguments<'_>) {
    println!("wrote {}: {summary}", path.display());
}

/// Validate the shared `--max-registration-*` flags with friendly messages.
pub(crate) fn validate_registration_flags(
    max_registration_rms: f64,
    max_registration_drift: f64,
    max_registration_drift_fraction: f64,
) -> Result<()> {
    if !max_registration_rms.is_finite() || max_registration_rms <= 0.0 {
        anyhow::bail!("--max-registration-rms must be a positive finite number");
    }
    if !max_registration_drift.is_finite() || max_registration_drift <= 0.0 {
        anyhow::bail!("--max-registration-drift must be a positive finite number");
    }
    if !max_registration_drift_fraction.is_finite()
        || !(0.0..=1.0).contains(&max_registration_drift_fraction)
    {
        anyhow::bail!("--max-registration-drift-fraction must be between zero and one");
    }
    Ok(())
}
