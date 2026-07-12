//! Source catalog downloaders.
//!
//! Each downloader fetches the primary distribution files into a directory,
//! verifies integrity where the format allows it, and skips files that are
//! already present and valid — safe to re-run after interruptions.

use anyhow::{Context, Result, bail};
use std::io::Read;
use std::path::Path;

const CDS_TYCHO2: &str = "https://cdsarc.cds.unistra.fr/ftp/I/259";
const OPENNGC: &str = "https://raw.githubusercontent.com/mattiaverga/OpenNGC/master/database_files";

/// Tycho-2 (CDS I/259): the ReadMe plus the 20 main-catalogue parts.
pub fn download_tycho2(output: &Path) -> Result<()> {
    std::fs::create_dir_all(output)?;
    fetch(
        &format!("{CDS_TYCHO2}/ReadMe"),
        &output.join("ReadMe"),
        Verify::None,
    )?;
    for part in 0..20 {
        let name = format!("tyc2.dat.{part:02}.gz");
        fetch(
            &format!("{CDS_TYCHO2}/{name}"),
            &output.join(&name),
            Verify::Gzip,
        )?;
    }
    // The supplement holds most stars brighter than magnitude ~2
    fetch(
        &format!("{CDS_TYCHO2}/suppl_1.dat.gz"),
        &output.join("suppl_1.dat.gz"),
        Verify::Gzip,
    )?;
    println!("Tycho-2 catalogue ready in {}", output.display());
    Ok(())
}

/// OpenNGC: the NGC/IC object list and its addendum.
pub fn download_openngc(output: &Path) -> Result<()> {
    std::fs::create_dir_all(output)?;
    for name in ["NGC.csv", "addendum.csv"] {
        fetch(
            &format!("{OPENNGC}/{name}"),
            &output.join(name),
            Verify::None,
        )?;
    }
    println!("OpenNGC catalog ready in {}", output.display());
    Ok(())
}

enum Verify {
    None,
    /// The file must decompress fully as gzip
    Gzip,
}

fn verify(path: &Path, how: &Verify) -> bool {
    match how {
        Verify::None => path.exists(),
        Verify::Gzip => {
            let Ok(file) = std::fs::File::open(path) else {
                return false;
            };
            let mut decoder = flate2::read::GzDecoder::new(file);
            let mut sink = [0u8; 64 * 1024];
            loop {
                match decoder.read(&mut sink) {
                    Ok(0) => return true,
                    Ok(_) => continue,
                    Err(_) => return false,
                }
            }
        }
    }
}

/// Download `url` to `target` unless a valid copy is already there.
/// Downloads land in a temp file first, are verified, then renamed.
fn fetch(url: &str, target: &Path, how: Verify) -> Result<()> {
    if verify(target, &how) {
        println!("  {} already present", target.display());
        return Ok(());
    }

    println!("  fetching {url}");
    let response = ureq::get(url)
        .timeout(std::time::Duration::from_secs(300))
        .call()
        .with_context(|| format!("failed to fetch {url}"))?;

    let temp = target.with_extension("part");
    let mut out = std::fs::File::create(&temp)?;
    std::io::copy(&mut response.into_reader(), &mut out)
        .with_context(|| format!("failed to download {url}"))?;
    drop(out);

    if !verify(&temp, &how) && !matches!(how, Verify::None) {
        std::fs::remove_file(&temp).ok();
        bail!("{url} downloaded but failed integrity verification");
    }
    std::fs::rename(&temp, target)?;
    Ok(())
}
