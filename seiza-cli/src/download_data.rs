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

/// All object-overlay sources: OpenNGC, VizieR Sharpless and Barnard (with
/// VizieR-computed J2000 positions), and the IAU star-name list.
pub fn download_objects(output: &Path) -> Result<()> {
    download_openngc(output)?;
    fetch(
        "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=VII/20/catalog&-out=_RAJ2000,_DEJ2000,Sh2,Diam&-out.max=unlimited",
        &output.join("sh2.tsv"),
        Verify::None,
    )?;
    fetch(
        "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=VII/220A/barnard&-out=_RAJ2000,_DEJ2000,Barn,Diam&-out.max=unlimited",
        &output.join("barnard.tsv"),
        Verify::None,
    )?;
    fetch(
        "https://www.pas.rochester.edu/~emamajek/WGSN/IAU-CSN.txt",
        &output.join("IAU-CSN.txt"),
        Verify::None,
    )?;
    let vizier = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=";
    for (name, source, columns) in [
        (
            "ugc.tsv",
            "VII/26D/catalog",
            "_RAJ2000,_DEJ2000,UGC,MajAxis,MinAxis,PA",
        ),
        ("ldn.tsv", "VII/7A/ldn", "_RAJ2000,_DEJ2000,LDN,Area"),
        (
            "vdb.tsv",
            "VII/21/catalog",
            "_RAJ2000,_DEJ2000,VdB,BRadMax,Vmag",
        ),
        ("bsc.tsv", "V/50/catalog", "_RAJ2000,_DEJ2000,HD,Name,Vmag"),
        // ~1M galaxies; the builder keeps the ones large enough to matter
        (
            "pgc.tsv",
            "VII/237/pgc",
            "_RAJ2000,_DEJ2000,PGC,logD25,logR25,PA",
        ),
        // Green's catalogue of Galactic supernova remnants
        (
            "snr.tsv",
            "VII/284/snrs",
            "_RAJ2000,_DEJ2000,SNR,MajDiam,MinDiam,Names",
        ),
        // van der Hucht's VIIth catalogue of Galactic Wolf-Rayet stars
        (
            "wr.tsv",
            "III/215/table13",
            "_RAJ2000,_DEJ2000,WR,Name,GCVS,OName",
        ),
    ] {
        fetch(
            &format!("{vizier}{source}&-out={columns}&-out.max=unlimited"),
            &output.join(name),
            Verify::None,
        )?;
    }
    println!("object catalogs ready in {}", output.display());
    Ok(())
}

enum Verify {
    None,
    /// The file must decompress fully as gzip
    Gzip,
    /// The file must hash to this SHA-256 (lowercase hex)
    Sha256(String),
}

fn verify(path: &Path, how: &Verify) -> bool {
    match how {
        Verify::None => path.exists(),
        Verify::Sha256(expected) => {
            let Ok(mut file) = std::fs::File::open(path) else {
                return false;
            };
            use sha2::Digest;
            let mut hasher = sha2::Sha256::new();
            if std::io::copy(&mut file, &mut hasher).is_err() {
                return false;
            }
            format!("{:x}", hasher.finalize()) == *expected
        }
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

/// Rochester Astronomy "Latest Supernovae" active list — no registration
/// required, updated daily. The canonical source (IAU TNS) needs a bot
/// account for its dumps; this covers the practical overlay use.
pub fn download_transients(output: &Path) -> Result<()> {
    std::fs::create_dir_all(output)?;
    let target = output.join("snactive.html");
    // Always refetch: the whole point is freshness
    std::fs::remove_file(&target).ok();
    fetch(
        "https://www.rochesterastronomy.org/snimages/snactive.html",
        &target,
        Verify::None,
    )?;
    println!("transient list ready in {}", output.display());
    Ok(())
}

const GAIA_TAP_SYNC: &str = "https://gea.esac.esa.int/tap-server/tap/sync";
/// Gaia DR3 source_id encodes the HEALPix level-12 cell in the high bits;
/// this spans the whole sky.
const GAIA_SOURCE_ID_MAX: u64 = 201_326_592 << 35;
const GAIA_MAXREC: u64 = 3_000_000;

/// Gaia DR3 star positions via ESA TAP, chunked by source_id so the download
/// resumes cleanly. Roughly 25M rows / 1.5 GB at the default magnitude limit;
/// expect a couple of hours.
pub fn download_gaia(output: &Path, max_mag: f32, chunks: u64) -> Result<()> {
    std::fs::create_dir_all(output)?;
    let mut done = 0u64;
    for chunk in 0..chunks {
        let target = output.join(format!("gaia-{chunk:04}.csv"));
        if chunk_complete(&target) {
            done += 1;
            continue;
        }
        let lo = GAIA_SOURCE_ID_MAX / chunks * chunk;
        let hi = if chunk + 1 == chunks {
            GAIA_SOURCE_ID_MAX
        } else {
            GAIA_SOURCE_ID_MAX / chunks * (chunk + 1) - 1
        };
        let query = format!(
            "SELECT ra, dec, pmra, pmdec, phot_g_mean_mag FROM gaiadr3.gaia_source \
             WHERE phot_g_mean_mag <= {max_mag} AND source_id BETWEEN {lo} AND {hi}"
        );

        let mut attempts = 0;
        loop {
            attempts += 1;
            match fetch_gaia_chunk(&query, &target) {
                Ok(rows) => {
                    if rows >= GAIA_MAXREC {
                        bail!(
                            "chunk {chunk} hit the {GAIA_MAXREC}-row cap; rerun with \
                             --chunks {}",
                            chunks * 4
                        );
                    }
                    done += 1;
                    println!("  chunk {chunk:04}: {rows} rows ({done}/{chunks})");
                    break;
                }
                Err(e) if attempts < 4 => {
                    eprintln!("  chunk {chunk:04} attempt {attempts} failed: {e}; retrying");
                    std::thread::sleep(std::time::Duration::from_secs(5 * attempts));
                }
                Err(e) => return Err(e.context(format!("chunk {chunk} failed"))),
            }
        }
    }
    println!("Gaia download complete in {}", output.display());
    Ok(())
}

/// A chunk is complete when it exists, ends with a newline, and has a header.
fn chunk_complete(path: &Path) -> bool {
    let Ok(data) = std::fs::read(path) else {
        return false;
    };
    data.len() > 10 && data.ends_with(b"\n")
}

fn fetch_gaia_chunk(query: &str, target: &Path) -> Result<u64> {
    let response = ureq::post(GAIA_TAP_SYNC)
        .timeout(std::time::Duration::from_secs(600))
        .send_form(&[
            ("REQUEST", "doQuery"),
            ("LANG", "ADQL"),
            ("FORMAT", "csv"),
            ("MAXREC", &GAIA_MAXREC.to_string()),
            ("QUERY", query),
        ])
        .context("TAP request failed")?;

    let temp = target.with_extension("part");
    let mut out = std::fs::File::create(&temp)?;
    std::io::copy(&mut response.into_reader(), &mut out)?;
    drop(out);

    let data = std::fs::read(&temp)?;
    if !data.starts_with(b"ra,dec") || !data.ends_with(b"\n") {
        std::fs::remove_file(&temp).ok();
        bail!("chunk response malformed or truncated");
    }
    let rows = data.iter().filter(|&&b| b == b'\n').count() as u64 - 1;
    std::fs::rename(&temp, target)?;
    Ok(rows)
}

/// Base URL for the hosted, prebuilt seiza datasets.
const HOSTED_DATA_URL: &str = "https://downloads.seiza.fyi/data";

/// Prebuilt datasets from downloads.seiza.fyi: reads the manifest, then
/// downloads every listed file (or just `names`) with SHA-256
/// verification. The quickest route to a working solver — no catalog
/// building required.
pub fn download_prebuilt(output: &Path, names: &[String]) -> Result<()> {
    std::fs::create_dir_all(output)?;

    let manifest_url = format!("{HOSTED_DATA_URL}/manifest.json");
    println!("  fetching {manifest_url}");
    let manifest: serde_json::Value = ureq::get(&manifest_url)
        .timeout(std::time::Duration::from_secs(60))
        .call()
        .with_context(|| format!("failed to fetch {manifest_url}"))?
        .into_json()
        .context("manifest is not valid JSON")?;

    let files = manifest["files"]
        .as_array()
        .context("manifest has no files list")?;
    let mut fetched = 0;
    for entry in files {
        let (Some(name), Some(sha)) = (entry["name"].as_str(), entry["sha256"].as_str()) else {
            continue;
        };
        if !names.is_empty() && !names.iter().any(|n| n == name) {
            continue;
        }
        fetch(
            &format!("{HOSTED_DATA_URL}/{name}"),
            &output.join(name),
            Verify::Sha256(sha.to_string()),
        )?;
        fetched += 1;
    }
    if fetched == 0 {
        bail!(
            "nothing matched; the manifest offers: {}",
            files
                .iter()
                .filter_map(|e| e["name"].as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    println!("{fetched} dataset(s) ready in {}", output.display());
    Ok(())
}

/// Minor Planet Center orbital element sets: current comets plus the
/// full MPCORB asteroid file (gzip kept on disk; the builder streams it).
pub fn download_mpc(output: &Path) -> Result<()> {
    std::fs::create_dir_all(output)?;
    // Comet elements change often; always refetch
    let comets = output.join("CometEls.txt");
    std::fs::remove_file(&comets).ok();
    fetch(
        "https://www.minorplanetcenter.net/iau/MPCORB/CometEls.txt",
        &comets,
        Verify::None,
    )?;
    fetch(
        "https://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz",
        &output.join("MPCORB.DAT.gz"),
        Verify::Gzip,
    )?;
    println!("MPC element sets ready in {}", output.display());
    Ok(())
}
