//! Async acquisition of the upstream catalogs used by Seiza's builders.
//!
//! These are raw source distributions, not runtime catalog bundles. For
//! application-facing installation of published `.bin` and `.idx` artifacts,
//! use `seiza-download` instead.

use flate2::read::GzDecoder;
use futures_util::StreamExt;
use std::io::{Read, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

const CDS_TYCHO2: &str = "https://cdsarc.cds.unistra.fr/ftp/I/259";
const OPENNGC: &str = "https://raw.githubusercontent.com/mattiaverga/OpenNGC/master/database_files";
const GAIA_TAP_SYNC: &str = "https://gea.esac.esa.int/tap-server/tap/sync";
/// Gaia DR3 source_id encodes the HEALPix level-12 cell in the high bits.
const GAIA_SOURCE_ID_MAX: u64 = 201_326_592 << 35;
const GAIA_MAXREC: u64 = 3_000_000;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{action} {}: {source}", path.display())]
    Io {
        action: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to fetch {url}: {source}")]
    Http {
        url: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("{url} returned HTTP {status}")]
    HttpStatus { url: String, status: u16 },

    #[error("{0} downloaded but failed integrity verification")]
    Integrity(String),

    #[error("Gaia TAP chunk response was malformed or truncated")]
    MalformedGaiaChunk,

    #[error("Gaia chunk {chunk} hit the {limit}-row cap; rerun with --chunks {suggested_chunks}")]
    GaiaRowCap {
        chunk: u64,
        limit: u64,
        suggested_chunks: u64,
    },

    #[error("background verification task failed: {0}")]
    BackgroundTask(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Source acquisition events suitable for CLI output or application progress.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SourceEvent {
    AlreadyPresent {
        path: PathBuf,
    },
    Fetching {
        url: String,
        path: PathBuf,
    },
    Progress {
        path: PathBuf,
        downloaded: u64,
        total: Option<u64>,
    },
    Retry {
        label: String,
        attempt: u32,
        delay: Duration,
        error: String,
    },
    GaiaChunkComplete {
        chunk: u64,
        rows: u64,
        completed: u64,
        total: u64,
    },
    Ready {
        source: &'static str,
        directory: PathBuf,
    },
}

type Reporter = Arc<dyn Fn(SourceEvent) + Send + Sync>;

/// Reusable asynchronous client for upstream astronomy sources.
#[derive(Clone)]
pub struct SourceDownloader {
    client: reqwest::Client,
    reporter: Reporter,
}

impl std::fmt::Debug for SourceDownloader {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SourceDownloader")
            .finish_non_exhaustive()
    }
}

impl SourceDownloader {
    pub fn new() -> Result<Self> {
        Self::with_reporter(|_| {})
    }

    pub fn with_reporter<F>(reporter: F) -> Result<Self>
    where
        F: Fn(SourceEvent) + Send + Sync + 'static,
    {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .user_agent(format!("seiza-sources/{}", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|source| Error::Http {
                url: "HTTP client initialization".into(),
                source,
            })?;
        Ok(Self {
            client,
            reporter: Arc::new(reporter),
        })
    }

    /// Tycho-2 (CDS I/259): the ReadMe, 20 main-catalog parts, and the
    /// bright-star supplement.
    pub async fn download_tycho2(&self, output: impl AsRef<Path>) -> Result<()> {
        let output = output.as_ref();
        create_dir_all(output).await?;
        self.fetch(
            &format!("{CDS_TYCHO2}/ReadMe"),
            &output.join("ReadMe"),
            Verify::None,
        )
        .await?;
        for part in 0..20 {
            let name = format!("tyc2.dat.{part:02}.gz");
            self.fetch(
                &format!("{CDS_TYCHO2}/{name}"),
                &output.join(&name),
                Verify::Gzip,
            )
            .await?;
        }
        self.fetch(
            &format!("{CDS_TYCHO2}/suppl_1.dat.gz"),
            &output.join("suppl_1.dat.gz"),
            Verify::Gzip,
        )
        .await?;
        self.ready("Tycho-2", output);
        Ok(())
    }

    /// Bright Star Catalogue, GCVS, WDS, and IAU star-name sources used to
    /// build the optional stellar identifier sidecar.
    pub async fn download_star_identifiers(&self, output: impl AsRef<Path>) -> Result<()> {
        let output = output.as_ref();
        create_dir_all(output).await?;
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
            self.fetch(
                &format!("{vizier}{source}&-out={columns}&-out.max=unlimited"),
                &output.join(name),
                Verify::None,
            )
            .await?;
        }
        self.fetch(
            "https://www.pas.rochester.edu/~emamajek/WGSN/IAU-CSN.txt",
            &output.join("IAU-CSN.txt"),
            Verify::None,
        )
        .await?;
        self.ready("stellar identifier sources", output);
        Ok(())
    }

    pub async fn download_openngc(&self, output: impl AsRef<Path>) -> Result<()> {
        let output = output.as_ref();
        create_dir_all(output).await?;
        for name in ["NGC.csv", "addendum.csv"] {
            self.fetch(
                &format!("{OPENNGC}/{name}"),
                &output.join(name),
                Verify::None,
            )
            .await?;
        }
        self.ready("OpenNGC", output);
        Ok(())
    }

    /// All object-overlay sources consumed by the object catalog builder.
    pub async fn download_objects(&self, output: impl AsRef<Path>) -> Result<()> {
        let output = output.as_ref();
        self.download_openngc(output).await?;
        self.fetch(
            "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=VII/20/catalog&-out=_RAJ2000,_DEJ2000,Sh2,Diam&-out.max=unlimited",
            &output.join("sh2.tsv"),
            Verify::None,
        )
        .await?;
        self.fetch(
            "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=VII/220A/barnard&-out=_RAJ2000,_DEJ2000,Barn,Diam&-out.max=unlimited",
            &output.join("barnard.tsv"),
            Verify::None,
        )
        .await?;
        self.fetch(
            "https://www.pas.rochester.edu/~emamajek/WGSN/IAU-CSN.txt",
            &output.join("IAU-CSN.txt"),
            Verify::None,
        )
        .await?;
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
            (
                "pgc.tsv",
                "VII/237/pgc",
                "_RAJ2000,_DEJ2000,PGC,logD25,logR25,PA",
            ),
            (
                "snr.tsv",
                "VII/284/snrs",
                "_RAJ2000,_DEJ2000,SNR,MajDiam,MinDiam,Names",
            ),
            (
                "wr.tsv",
                "III/215/table13",
                "_RAJ2000,_DEJ2000,WR,Name,GCVS,OName",
            ),
        ] {
            self.fetch(
                &format!("{vizier}{source}&-out={columns}&-out.max=unlimited"),
                &output.join(name),
                Verify::None,
            )
            .await?;
        }
        self.ready("object catalogs", output);
        Ok(())
    }

    /// Rochester Astronomy's active supernova list. Always refreshed.
    pub async fn download_transients(&self, output: impl AsRef<Path>) -> Result<()> {
        let output = output.as_ref();
        create_dir_all(output).await?;
        let target = output.join("snactive.html");
        self.refresh(
            "https://www.rochesterastronomy.org/snimages/snactive.html",
            &target,
            Verify::None,
        )
        .await?;
        self.ready("transient list", output);
        Ok(())
    }

    /// Gaia DR3 positions via ESA TAP, split by source_id for resumability.
    pub async fn download_gaia(
        &self,
        output: impl AsRef<Path>,
        max_mag: f32,
        chunks: u64,
    ) -> Result<()> {
        let output = output.as_ref();
        create_dir_all(output).await?;
        let mut completed = 0u64;

        for chunk in 0..chunks {
            let target = output.join(format!("gaia-{chunk:04}.csv"));
            if chunk_complete(&target).await? {
                completed += 1;
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

            let mut attempts = 0u32;
            loop {
                attempts += 1;
                match self.fetch_gaia_chunk(&query, &target).await {
                    Ok(rows) => {
                        if rows >= GAIA_MAXREC {
                            return Err(Error::GaiaRowCap {
                                chunk,
                                limit: GAIA_MAXREC,
                                suggested_chunks: chunks * 4,
                            });
                        }
                        completed += 1;
                        (self.reporter)(SourceEvent::GaiaChunkComplete {
                            chunk,
                            rows,
                            completed,
                            total: chunks,
                        });
                        break;
                    }
                    Err(error) if attempts < 4 => {
                        let delay = Duration::from_secs(5 * attempts as u64);
                        (self.reporter)(SourceEvent::Retry {
                            label: format!("Gaia chunk {chunk:04}"),
                            attempt: attempts,
                            delay,
                            error: error.to_string(),
                        });
                        tokio::time::sleep(delay).await;
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        self.ready("Gaia", output);
        Ok(())
    }

    /// Minor Planet Center comet/asteroid elements plus JPL SBDB historical
    /// comet apparitions.
    pub async fn download_mpc(&self, output: impl AsRef<Path>) -> Result<()> {
        let output = output.as_ref();
        create_dir_all(output).await?;
        let comets = output.join("CometEls.txt");
        self.refresh(
            "https://www.minorplanetcenter.net/iau/MPCORB/CometEls.txt",
            &comets,
            Verify::None,
        )
        .await?;
        self.fetch(
            "https://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz",
            &output.join("MPCORB.DAT.gz"),
            Verify::Gzip,
        )
        .await?;
        let sbdb = output.join("sbdb-comets.json");
        self.refresh(
            "https://ssd-api.jpl.nasa.gov/sbdb_query.api?fields=full_name,epoch,q,e,i,om,w,tp,M1,K1&sb-kind=c",
            &sbdb,
            Verify::None,
        )
        .await?;
        self.ready("MPC + SBDB element sets", output);
        Ok(())
    }

    async fn fetch(&self, url: &str, target: &Path, verify: Verify) -> Result<()> {
        self.fetch_with_policy(url, target, verify, false).await
    }

    async fn refresh(&self, url: &str, target: &Path, verify: Verify) -> Result<()> {
        self.fetch_with_policy(url, target, verify, true).await
    }

    async fn fetch_with_policy(
        &self,
        url: &str,
        target: &Path,
        verify: Verify,
        force: bool,
    ) -> Result<()> {
        if !force && verify_file(target, verify).await? {
            (self.reporter)(SourceEvent::AlreadyPresent {
                path: target.to_path_buf(),
            });
            return Ok(());
        }

        (self.reporter)(SourceEvent::Fetching {
            url: url.into(),
            path: target.to_path_buf(),
        });
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|source| Error::Http {
                url: url.into(),
                source,
            })?;
        if !response.status().is_success() {
            return Err(Error::HttpStatus {
                url: url.into(),
                status: response.status().as_u16(),
            });
        }
        let total = response.content_length();
        let temp = partial_path(target);
        let mut output = tokio::fs::File::create(&temp)
            .await
            .map_err(|source| io("create", &temp, source))?;
        let mut stream = response.bytes_stream();
        let mut downloaded = 0u64;
        let mut reported = 0u64;
        let transfer = async {
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|source| Error::Http {
                    url: url.into(),
                    source,
                })?;
                output
                    .write_all(&chunk)
                    .await
                    .map_err(|source| io("write", &temp, source))?;
                downloaded += chunk.len() as u64;
                if total == Some(downloaded)
                    || downloaded.saturating_sub(reported) >= 4 * 1024 * 1024
                {
                    (self.reporter)(SourceEvent::Progress {
                        path: target.to_path_buf(),
                        downloaded,
                        total,
                    });
                    reported = downloaded;
                }
            }
            if downloaded != reported {
                (self.reporter)(SourceEvent::Progress {
                    path: target.to_path_buf(),
                    downloaded,
                    total,
                });
            }
            output
                .sync_all()
                .await
                .map_err(|source| io("sync", &temp, source))?;
            Ok(())
        }
        .await;
        drop(output);
        if let Err(error) = transfer {
            let _ = tokio::fs::remove_file(&temp).await;
            return Err(error);
        }
        if !verify_file(&temp, verify).await? {
            let _ = tokio::fs::remove_file(&temp).await;
            return Err(Error::Integrity(url.into()));
        }
        replace_file(&temp, target).await
    }

    async fn fetch_gaia_chunk(&self, query: &str, target: &Path) -> Result<u64> {
        let form = [
            ("REQUEST", "doQuery".to_string()),
            ("LANG", "ADQL".to_string()),
            ("FORMAT", "csv".to_string()),
            ("MAXREC", GAIA_MAXREC.to_string()),
            ("QUERY", query.to_string()),
        ];
        let response = self
            .client
            .post(GAIA_TAP_SYNC)
            .timeout(Duration::from_secs(600))
            .form(&form)
            .send()
            .await
            .map_err(|source| Error::Http {
                url: GAIA_TAP_SYNC.into(),
                source,
            })?;
        if !response.status().is_success() {
            return Err(Error::HttpStatus {
                url: GAIA_TAP_SYNC.into(),
                status: response.status().as_u16(),
            });
        }

        let temp = partial_path(target);
        let mut output = tokio::fs::File::create(&temp)
            .await
            .map_err(|source| io("create", &temp, source))?;
        let mut stream = response.bytes_stream();
        let mut prefix = Vec::with_capacity(6);
        let mut last = None;
        let mut newline_count = 0u64;
        let transfer = async {
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|source| Error::Http {
                    url: GAIA_TAP_SYNC.into(),
                    source,
                })?;
                if prefix.len() < 6 {
                    let needed = 6 - prefix.len();
                    prefix.extend_from_slice(&chunk[..chunk.len().min(needed)]);
                }
                last = chunk.last().copied().or(last);
                newline_count += chunk.iter().filter(|&&byte| byte == b'\n').count() as u64;
                output
                    .write_all(&chunk)
                    .await
                    .map_err(|source| io("write", &temp, source))?;
            }
            output
                .sync_all()
                .await
                .map_err(|source| io("sync", &temp, source))?;
            Ok(())
        }
        .await;
        drop(output);
        if let Err(error) = transfer {
            let _ = tokio::fs::remove_file(&temp).await;
            return Err(error);
        }
        if !prefix.starts_with(b"ra,dec") || last != Some(b'\n') || newline_count == 0 {
            let _ = tokio::fs::remove_file(&temp).await;
            return Err(Error::MalformedGaiaChunk);
        }
        replace_file(&temp, target).await?;
        Ok(newline_count - 1)
    }

    fn ready(&self, source: &'static str, output: &Path) {
        (self.reporter)(SourceEvent::Ready {
            source,
            directory: output.to_path_buf(),
        });
    }
}

#[derive(Clone, Copy)]
enum Verify {
    None,
    Gzip,
}

async fn verify_file(path: &Path, verify: Verify) -> Result<bool> {
    match verify {
        Verify::None => match tokio::fs::metadata(path).await {
            Ok(metadata) => Ok(metadata.is_file() && metadata.len() > 0),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(source) => Err(io("read metadata for", path, source)),
        },
        Verify::Gzip => {
            let path = path.to_path_buf();
            let task_path = path.clone();
            tokio::task::spawn_blocking(move || -> std::io::Result<bool> {
                let input = match std::fs::File::open(&task_path) {
                    Ok(input) => input,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
                    Err(error) => return Err(error),
                };
                let mut decoder = GzDecoder::new(input);
                let mut sink = [0u8; 64 * 1024];
                loop {
                    match decoder.read(&mut sink) {
                        Ok(0) => return Ok(true),
                        Ok(_) => {}
                        Err(_) => return Ok(false),
                    }
                }
            })
            .await
            .map_err(|error| Error::BackgroundTask(error.to_string()))?
            .map_err(|source| io("verify gzip file", path, source))
        }
    }
}

async fn chunk_complete(path: &Path) -> Result<bool> {
    let mut input = match tokio::fs::File::open(path).await {
        Ok(input) => input,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(source) => return Err(io("open", path, source)),
    };
    let metadata = input
        .metadata()
        .await
        .map_err(|source| io("read metadata for", path, source))?;
    if metadata.len() <= 10 {
        return Ok(false);
    }
    let mut prefix = [0u8; 6];
    input
        .read_exact(&mut prefix)
        .await
        .map_err(|source| io("read", path, source))?;
    input
        .seek(SeekFrom::End(-1))
        .await
        .map_err(|source| io("seek", path, source))?;
    let mut last = [0u8; 1];
    input
        .read_exact(&mut last)
        .await
        .map_err(|source| io("read", path, source))?;
    Ok(&prefix == b"ra,dec" && last[0] == b'\n')
}

async fn create_dir_all(path: &Path) -> Result<()> {
    tokio::fs::create_dir_all(path)
        .await
        .map_err(|source| io("create directory", path, source))
}

#[cfg(windows)]
async fn remove_if_exists(path: &Path) -> Result<()> {
    match tokio::fs::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(io("remove", path, source)),
    }
}

async fn replace_file(temp: &Path, target: &Path) -> Result<()> {
    #[cfg(windows)]
    remove_if_exists(target).await?;
    tokio::fs::rename(temp, target)
        .await
        .map_err(|source| io("rename", temp, source))
}

fn partial_path(target: &Path) -> PathBuf {
    target.with_extension(format!(
        "part-{}-{}",
        std::process::id(),
        TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ))
}

fn io(action: &'static str, path: impl Into<PathBuf>, source: std::io::Error) -> Error {
    Error::Io {
        action,
        path: path.into(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn gaia_completion_check_reads_only_boundaries() {
        let temp = tempfile::tempdir().unwrap();
        let complete = temp.path().join("complete.csv");
        let truncated = temp.path().join("truncated.csv");
        tokio::fs::write(&complete, b"ra,dec,pmra\n1,2,3\n")
            .await
            .unwrap();
        tokio::fs::write(&truncated, b"ra,dec,pmra\n1,2,3")
            .await
            .unwrap();
        assert!(chunk_complete(&complete).await.unwrap());
        assert!(!chunk_complete(&truncated).await.unwrap());
    }

    #[tokio::test]
    async fn gzip_verification_detects_truncation() {
        use std::io::Write;

        let temp = tempfile::tempdir().unwrap();
        let valid = temp.path().join("valid.gz");
        let mut encoder = flate2::write::GzEncoder::new(
            std::fs::File::create(&valid).unwrap(),
            flate2::Compression::default(),
        );
        encoder.write_all(b"catalog").unwrap();
        encoder.finish().unwrap();
        assert!(verify_file(&valid, Verify::Gzip).await.unwrap());

        let bytes = std::fs::read(&valid).unwrap();
        let truncated = temp.path().join("truncated.gz");
        std::fs::write(&truncated, &bytes[..bytes.len() / 2]).unwrap();
        assert!(!verify_file(&truncated, Verify::Gzip).await.unwrap());
    }
}
