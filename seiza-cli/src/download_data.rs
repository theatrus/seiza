//! Source catalog downloaders.
//!
//! Each downloader fetches the primary distribution files into a directory,
//! verifies integrity where the format allows it, and skips files that are
//! already present and valid — safe to re-run after interruptions.

use anyhow::{Context, Result, bail};
use std::collections::BTreeSet;
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

/// Textual and bright-star identity sources used to enrich a Tycho identifier
/// sidecar: Bright Star Catalogue, GCVS, WDS, and the IAU Catalog of Star
/// Names. VizieR supplies computed J2000 positions in tab-separated form.
pub fn download_star_identifiers(output: &Path) -> Result<()> {
    std::fs::create_dir_all(output)?;
    let vizier = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=";
    for (name, source, columns) in [
        (
            "bsc-identifiers.tsv",
            "V/50/catalog",
            "_RAJ2000,_DEJ2000,HR,Name,HD,SAO,FK5,ADS,ADScomp,VarID,Vmag,pmRA,pmDE",
        ),
        (
            "gcvs.tsv",
            "B/gcvs/gcvs_cat",
            "_RAJ2000,_DEJ2000,GCVS,VarType,magMax,l_Min1,Min1,n_Min1,flt,Period,pmRA,pmDE,Ep-coor,Exists",
        ),
        (
            "wds.tsv",
            "B/wds/wds",
            "_RAJ2000,_DEJ2000,WDS,Disc,Comp,mag1,mag2,pa2,sep2,pmRA1,pmDE1",
        ),
    ] {
        fetch(
            &format!("{vizier}{source}&-out={columns}&-out.max=unlimited"),
            &output.join(name),
            Verify::None,
        )?;
    }
    fetch(
        "https://www.pas.rochester.edu/~emamajek/WGSN/IAU-CSN.txt",
        &output.join("IAU-CSN.txt"),
        Verify::None,
    )?;
    println!("stellar identifier sources ready in {}", output.display());
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

/// All object-overlay sources: OpenNGC, selected VizieR catalogs (with
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
            "_RAJ2000,_DEJ2000,UGC,A,MajAxis,MinAxis,PA",
        ),
        ("ldn.tsv", "VII/7A/ldn", "_RAJ2000,_DEJ2000,LDN,Area"),
        (
            "vdb.tsv",
            "VII/21/catalog",
            "_RAJ2000,_DEJ2000,VdB,BRadMax,Vmag",
        ),
        (
            "ced.tsv",
            "VII/231/catalog",
            "_RAJ2000,_DEJ2000,Ced,m_Ced,Name,Dim1,Dim2,Class,SpNeb",
        ),
        (
            "lbn.tsv",
            "VII/9/catalog",
            "_RAJ2000,_DEJ2000,Seq,Diam1,Diam2,Name,ID",
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

/// Current complete catalog bundle. The unversioned `/data` prefix remains a
/// legacy v1 compatibility surface and is not consulted by new clients.
const HOSTED_DATA_V2_URL: &str = "https://downloads.seiza.fyi/data/v2";
const HOSTED_DATA_V2_REQUIRED: &[&str] = &[
    "blind-gaia16.idx",
    "minor-bodies.bin",
    "objects.bin",
    "stars-deep-gaia17.bin",
    "stars-gaia.bin",
    "stars-lite-tycho2.bin",
    "stars-lite-tycho2.ids.bin",
    "transients.bin",
];

#[derive(Debug, Eq, PartialEq)]
struct HostedFile {
    name: String,
    sha256: String,
}

fn manifest_entries(manifest: &serde_json::Value) -> Result<Vec<(String, String)>> {
    let files = manifest["files"]
        .as_array()
        .context("manifest has no files list")?;
    files
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let name = entry["name"]
                .as_str()
                .with_context(|| format!("manifest file {index} has no name"))?;
            let sha256 = entry["sha256"]
                .as_str()
                .with_context(|| format!("manifest file {index} has no sha256"))?;
            Ok((name.to_string(), sha256.to_string()))
        })
        .collect()
}

/// Select files from the complete v2 catalog bundle. The required-file check
/// turns a missed publication into an explicit error instead of silently
/// shipping a partial default download.
fn hosted_download_plan(manifest: &serde_json::Value, names: &[String]) -> Result<Vec<HostedFile>> {
    let version = manifest["version"]
        .as_str()
        .context("manifest has no version")?;
    if !version.starts_with("catalog-bundle-v2-") {
        bail!("unsupported catalog bundle manifest version: {version}");
    }

    let entries = manifest_entries(manifest)?;
    let requested = names.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let mut offered = BTreeSet::new();
    let mut matched = BTreeSet::new();
    let mut plan = Vec::new();

    for (name, sha256) in entries {
        if !offered.insert(name.clone()) {
            bail!("catalog bundle v2 manifest has duplicate file: {name}");
        }
        if requested.is_empty() || requested.contains(name.as_str()) {
            matched.insert(name.clone());
            plan.push(HostedFile { name, sha256 });
        }
    }

    let missing_bundle_files = HOSTED_DATA_V2_REQUIRED
        .iter()
        .filter(|name| !offered.contains(**name))
        .copied()
        .collect::<Vec<_>>();
    if !missing_bundle_files.is_empty() {
        bail!(
            "catalog bundle v2 manifest is incomplete; missing: {}",
            missing_bundle_files.join(", ")
        );
    }

    let missing = requested
        .iter()
        .filter(|name| !matched.contains(**name))
        .copied()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "requested file(s) unavailable: {}; the manifests offer: {}",
            missing.join(", "),
            offered.into_iter().collect::<Vec<_>>().join(", ")
        );
    }
    if plan.is_empty() {
        bail!(
            "nothing matched; the manifests offer: {}",
            offered.into_iter().collect::<Vec<_>>().join(", ")
        );
    }
    Ok(plan)
}

fn fetch_hosted_manifest(base_url: &str) -> Result<serde_json::Value> {
    let manifest_url = format!("{base_url}/manifest.json");
    println!("  fetching {manifest_url}");
    let response = ureq::get(&manifest_url)
        .timeout(std::time::Duration::from_secs(60))
        .call()
        .with_context(|| format!("failed to fetch {manifest_url}"))?;
    response
        .into_json()
        .with_context(|| format!("{manifest_url} is not valid JSON"))
}

/// Prebuilt datasets from the complete v2 bundle at downloads.seiza.fyi.
/// Downloads every listed file (or just `names`) with SHA-256 verification.
/// The quickest route to a working solver — no catalog building required.
pub fn download_prebuilt(output: &Path, names: &[String]) -> Result<()> {
    std::fs::create_dir_all(output)?;

    let manifest = fetch_hosted_manifest(HOSTED_DATA_V2_URL)?;
    let plan = hosted_download_plan(&manifest, names)?;

    let fetched = plan.len();
    for file in plan {
        fetch(
            &format!("{HOSTED_DATA_V2_URL}/{}", file.name),
            &output.join(file.name),
            Verify::Sha256(file.sha256),
        )?;
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
    // JPL SBDB: every catalogued comet with apparition-specific elements —
    // this is what makes PAST acquisition dates work (CometEls only lists
    // currently observable comets at current epochs). Always refetched.
    let sbdb = output.join("sbdb-comets.json");
    std::fs::remove_file(&sbdb).ok();
    fetch(
        "https://ssd-api.jpl.nasa.gov/sbdb_query.api?fields=full_name,epoch,q,e,i,om,w,tp,M1,K1&sb-kind=c",
        &sbdb,
        Verify::None,
    )?;
    println!("MPC + SBDB element sets ready in {}", output.display());
    Ok(())
}

#[cfg(test)]
mod prebuilt_tests {
    use super::*;
    use serde_json::json;

    fn v2_manifest() -> serde_json::Value {
        json!({
            "version": "catalog-bundle-v2-test",
            "files": [
                {"name": "blind-gaia16.idx", "sha256": "blind-v2"},
                {"name": "minor-bodies.bin", "sha256": "minor-v2"},
                {"name": "objects.bin", "sha256": "objects-v3"},
                {"name": "stars-deep-gaia17.bin", "sha256": "deep-v2"},
                {"name": "stars-gaia.bin", "sha256": "gaia-v2"},
                {"name": "stars-lite-tycho2.bin", "sha256": "tycho-v2"},
                {"name": "stars-lite-tycho2.ids.bin", "sha256": "identifiers-v1"},
                {"name": "transients.bin", "sha256": "transients-v3"}
            ]
        })
    }

    #[test]
    fn default_plan_contains_the_complete_v2_bundle() {
        let plan = hosted_download_plan(&v2_manifest(), &[]).unwrap();
        assert_eq!(plan.len(), HOSTED_DATA_V2_REQUIRED.len());
        assert!(plan.contains(&HostedFile {
            name: "objects.bin".into(),
            sha256: "objects-v3".into(),
        }));
        assert!(plan.contains(&HostedFile {
            name: "stars-lite-tycho2.ids.bin".into(),
            sha256: "identifiers-v1".into(),
        }));
    }

    #[test]
    fn explicit_download_selects_only_requested_v2_files() {
        let plan = hosted_download_plan(
            &v2_manifest(),
            &["objects.bin".into(), "stars-lite-tycho2.ids.bin".into()],
        )
        .unwrap();
        assert_eq!(
            plan,
            vec![
                HostedFile {
                    name: "objects.bin".into(),
                    sha256: "objects-v3".into(),
                },
                HostedFile {
                    name: "stars-lite-tycho2.ids.bin".into(),
                    sha256: "identifiers-v1".into(),
                },
            ]
        );
    }

    #[test]
    fn incomplete_v2_bundle_is_rejected_even_for_an_explicit_file() {
        let mut manifest = v2_manifest();
        manifest["files"]
            .as_array_mut()
            .unwrap()
            .retain(|entry| entry["name"] != "stars-lite-tycho2.ids.bin");
        let error = hosted_download_plan(&manifest, &["objects.bin".into()]).unwrap_err();
        assert!(error.to_string().contains(
            "catalog bundle v2 manifest is incomplete; missing: stars-lite-tycho2.ids.bin"
        ));
    }

    #[test]
    fn unavailable_requested_file_lists_the_bundle() {
        let error = hosted_download_plan(&v2_manifest(), &["missing.bin".into()]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requested file(s) unavailable: missing.bin")
        );
        assert!(error.to_string().contains("objects.bin"));
    }

    #[test]
    fn wrong_bundle_version_is_rejected() {
        let mut manifest = v2_manifest();
        manifest["version"] = json!("object-v3-test");
        let error = hosted_download_plan(&manifest, &[]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unsupported catalog bundle manifest version")
        );
    }

    #[test]
    fn duplicate_bundle_file_is_rejected() {
        let mut manifest = v2_manifest();
        manifest["files"]
            .as_array_mut()
            .unwrap()
            .push(json!({"name": "objects.bin", "sha256": "other"}));
        let error = hosted_download_plan(&manifest, &[]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("catalog bundle v2 manifest has duplicate file: objects.bin")
        );
    }

    #[test]
    fn malformed_bundle_entry_is_rejected() {
        let mut manifest = v2_manifest();
        manifest["files"][0]
            .as_object_mut()
            .unwrap()
            .remove("sha256");
        let error = hosted_download_plan(&manifest, &[]).unwrap_err();
        assert!(error.to_string().contains("manifest file 0 has no sha256"));
    }
}
