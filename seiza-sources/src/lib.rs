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
const OPENNGC_ARCHIVE: &str =
    "https://github.com/mattiaverga/OpenNGC/archive/refs/heads/master.tar.gz";
const VIZIER_TSV: &str = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=";
const SIMBAD_TAP: &str = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync";
/// Row cap asked of SIMBAD; a result this long may be cut short.
const SIMBAD_MAXREC: u64 = 3_000_000;

/// VizieR tables behind the object distance builder: file name, table, and
/// the columns kept. The first column names the header line that marks a
/// complete download.
const DISTANCE_TABLES: &[(&str, &str, &str)] = &[
    (
        "hunt-reffert-2024.tsv",
        "J/A+A/686/A42/clusters",
        "Name,AllNames,Type,dist16,dist50,dist84,_RAJ2000,_DEJ2000",
    ),
    ("cosmicflows-4.tsv", "J/ApJ/944/94/table2", "PGC,DM,e_DM"),
    (
        "chornay-walton-2021.tsv",
        "J/A+A/656/A110/tablea1",
        "PNG,Name,rcomb,b_rcomb,B_rcomb,pnRAdeg,pnDEdeg",
    ),
    (
        "gonzalez-santamaria-2021-names.tsv",
        "J/A+A/656/A51/tablea1",
        "PNG,OName,_RAJ2000,_DEJ2000",
    ),
    (
        "gonzalez-santamaria-2021.tsv",
        "J/A+A/656/A51/tablea2",
        "PNG,Dist,b_Dist,B_Dist",
    ),
    (
        "stanghellini-haywood-2010.tsv",
        "J/ApJ/714/1096/table1",
        "PNG,d,e_d,_RA,_DE",
    ),
    (
        "foster-brunt-2015.tsv",
        "J/AJ/150/147/table2",
        "HII,r,dr,SimbadName,_RAJ2000,_DEJ2000",
    ),
    (
        "ranasinghe-leahy-2023.tsv",
        "J/ApJS/265/53/table1",
        "SNR,l_X,X,_RAJ2000,_DEJ2000",
    ),
    (
        "harris-1997.tsv",
        "VII/202/catalog",
        "ID,Rsun,_RAJ2000,_DEJ2000",
    ),
    (
        "wise-hii-2014.tsv",
        "J/ApJS/212/1/wisecat",
        "WISE,Rad,Dist,_RAJ2000,_DEJ2000",
    ),
    (
        "zucker-2020.tsv",
        "J/A+A/633/A51/handbook",
        "Name,d50,_RAJ2000,_DEJ2000",
    ),
    (
        "hilton-lahulla-1995.tsv",
        "J/A+AS/113/325/table1b",
        "LDN,n_Dist2,Dist,Refs",
    ),
    ("vdb-stars.tsv", "VII/21/catalog", "VdB,DM,HD"),
];

/// SIMBAD identifier patterns for the catalogues `objects.bin` draws on.
const SIMBAD_DESIGNATIONS: &[&str] = &[
    "NGC %",
    "IC %",
    "M %",
    "SH %",
    "LBN %",
    "LDN %",
    "Barnard %",
    "VdB %",
    "Ced %",
    "UGC %",
    "LEDA %",
    "SNR G%",
    "PN G%",
    "HCG %",
    "Cl Melotte %",
    "Cl Collinder %",
];
/// A TAP service carrying Gaia DR3 under the archive's column names.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GaiaArchive {
    /// ESA's Gaia archive (`gaiadr3.gaia_source`), the primary source.
    #[default]
    Esa,
    /// The GAVO data centre's mirror in Heidelberg (`gaia.dr3lite`), which
    /// carries every column Seiza reads.
    Gavo,
}

impl GaiaArchive {
    fn sync_url(self) -> &'static str {
        match self {
            Self::Esa => "https://gea.esac.esa.int/tap-server/tap/sync",
            Self::Gavo => "https://dc.g-vo.org/tap/sync",
        }
    }

    fn async_url(self) -> &'static str {
        match self {
            Self::Esa => "https://gea.esac.esa.int/tap-server/tap/async",
            Self::Gavo => "https://dc.g-vo.org/tap/async",
        }
    }

    fn table(self) -> &'static str {
        match self {
            Self::Esa => "gaiadr3.gaia_source",
            Self::Gavo => "gaia.dr3lite",
        }
    }

    /// The archive's copy of Bailer-Jones et al. (2021), distances to
    /// Gaia EDR3 sources, which keep their source IDs in DR3.
    fn distance_table(self) -> &'static str {
        match self {
            Self::Esa => "external.gaiaedr3_distance",
            Self::Gavo => "gedr3dist.main",
        }
    }

    /// How many source_id ranges each bulk chunk is fetched in. GAVO stops a
    /// synchronous query after about 25 s, which a whole chunk near the
    /// galactic plane exceeds.
    fn chunk_pieces(self) -> u64 {
        match self {
            Self::Esa => 1,
            Self::Gavo => 32,
        }
    }
}
const GAIA_JOB_TIMEOUT: Duration = Duration::from_secs(900);
/// Gaia DR3 source_id encodes the HEALPix level-12 cell in the high bits.
const GAIA_SOURCE_ID_MAX: u64 = 201_326_592 << 35;
const GAIA_MAXREC: u64 = 3_000_000;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const READ_TIMEOUT: Duration = Duration::from_secs(300);

/// The ADQL for Gaia DR3 sources in a cone with their Bailer-Jones distances.
fn gaia_distance_query(
    archive: GaiaArchive,
    ra: f64,
    dec: f64,
    radius_deg: f64,
    max_mag: f32,
) -> String {
    format!(
        "SELECT g.ra, g.dec, g.pmra, g.pmdec, g.phot_g_mean_mag, g.phot_bp_mean_mag, \
         g.phot_rp_mean_mag, g.parallax, g.parallax_error, \
         d.r_med_geo, d.r_lo_geo, d.r_hi_geo \
         FROM {table} AS g LEFT OUTER JOIN {distances} AS d ON g.source_id = d.source_id \
         WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), \
         CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) \
         AND g.phot_g_mean_mag <= {max_mag} ORDER BY g.phot_g_mean_mag",
        table = archive.table(),
        distances = archive.distance_table(),
    )
}

/// VizieR's synchronous TAP endpoint.
const VIZIER_TAP_SYNC: &str = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync";

/// A cone needs a finite centre, a radius in (0, 90] degrees and a finite
/// magnitude limit.
fn check_cone(ra: f64, dec: f64, radius_deg: f64, max_mag: f32) -> Result<()> {
    if !ra.is_finite()
        || !dec.is_finite()
        || !(radius_deg.is_finite() && radius_deg > 0.0 && radius_deg <= 90.0)
        || !max_mag.is_finite()
    {
        return Err(Error::InvalidGaiaCone);
    }
    Ok(())
}

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

    #[error(
        "Gaia cone search needs a finite centre, a radius in (0, 90] degrees, and a finite magnitude limit"
    )]
    InvalidGaiaCone,

    #[error("Gaia archive query {0}")]
    GaiaJobFailed(String),

    #[error("Hipparcos query response was malformed")]
    MalformedHipparcos,

    #[error("Gaia magnitude limit must be finite; got {0}")]
    InvalidGaiaMagnitude(f32),

    #[error("Gaia chunk count must be between 1 and {max}; got {chunks}")]
    InvalidGaiaChunks { chunks: u64, max: u64 },

    #[error("Gaia chunk {chunk} hit the {limit}-row cap; rerun with --chunks {suggested_chunks}")]
    GaiaRowCap {
        chunk: u64,
        limit: u64,
        suggested_chunks: u64,
    },

    #[error("background verification task failed: {0}")]
    BackgroundTask(String),

    #[error("invalid GitHub repository name: {0}")]
    InvalidRepository(String),

    #[error("invalid pinned Git commit: {0}")]
    InvalidRevision(String),
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

/// The most stars a Gaia cone search returns, brightest first.
pub const GAIA_CONE_MAXREC: u64 = 200_000;

/// One Gaia DR3 source from [`SourceDownloader::gaia_photometry_cone`]:
/// its ICRS position at epoch J2016.0, proper motion, and mean photometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaiaPhotometry {
    /// Right ascension, degrees.
    pub ra: f64,
    /// Declination, degrees.
    pub dec: f64,
    /// Proper motion in right ascension times cos(dec), mas/yr.
    pub pmra: Option<f64>,
    /// Proper motion in declination, mas/yr.
    pub pmdec: Option<f64>,
    /// G-band mean magnitude.
    pub g: f32,
    /// BP mean magnitude.
    pub bp: Option<f32>,
    /// RP mean magnitude.
    pub rp: Option<f32>,
    /// Renormalized unit weight error; above about 1.4 the source is likely
    /// a binary or otherwise poorly fitted.
    pub ruwe: Option<f32>,
}

impl GaiaPhotometry {
    /// The BP − RP colour, when both magnitudes are measured.
    pub fn bp_rp(&self) -> Option<f32> {
        Some(self.bp? - self.rp?)
    }
}

/// Parse the CSV [`SourceDownloader::gaia_photometry_cone_csv`] returns.
/// Rows with a missing position or G magnitude are skipped.
pub fn parse_gaia_photometry(csv: &str) -> Result<Vec<GaiaPhotometry>> {
    let mut lines = csv.lines();
    let header = lines.next().ok_or(Error::MalformedGaiaChunk)?;
    let columns = header.split(',').map(str::trim).collect::<Vec<_>>();
    let column = |name: &str| {
        columns
            .iter()
            .position(|column| *column == name)
            .ok_or(Error::MalformedGaiaChunk)
    };
    let indices = [
        column("ra")?,
        column("dec")?,
        column("pmra")?,
        column("pmdec")?,
        column("phot_g_mean_mag")?,
        column("phot_bp_mean_mag")?,
        column("phot_rp_mean_mag")?,
        column("ruwe")?,
    ];
    let mut stars = Vec::new();
    for line in lines.filter(|line| !line.trim().is_empty()) {
        let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
        let value = |index: usize| {
            fields
                .get(indices[index])
                .and_then(|field| field.parse::<f64>().ok())
                .filter(|value| value.is_finite())
        };
        let (Some(ra), Some(dec), Some(g)) = (value(0), value(1), value(4)) else {
            continue;
        };
        stars.push(GaiaPhotometry {
            ra,
            dec,
            pmra: value(2),
            pmdec: value(3),
            g: g as f32,
            bp: value(5).map(|value| value as f32),
            rp: value(6).map(|value| value as f32),
            ruwe: value(7).map(|value| value as f32),
        });
    }
    Ok(stars)
}

/// One Gaia DR3 source from [`SourceDownloader::gaia_distance_cone`]: its
/// position at epoch J2016.0, proper motion, G magnitude and colour, and how
/// far away it is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaiaDistance {
    /// Right ascension, degrees.
    pub ra: f64,
    /// Declination, degrees.
    pub dec: f64,
    /// Proper motion in right ascension times cos(dec), mas/yr.
    pub pmra: Option<f64>,
    /// Proper motion in declination, mas/yr.
    pub pmdec: Option<f64>,
    /// G-band mean magnitude.
    pub g: f32,
    /// BP − RP colour, when both are measured.
    pub bp_rp: Option<f32>,
    /// Parallax and its standard error, mas. Gaia has none for the
    /// brightest stars, and a faint star's can be negative.
    pub parallax: Option<f64>,
    pub parallax_error: Option<f64>,
    /// Bailer-Jones et al. (2021) geometric distance in parsecs: the median
    /// and the 16th and 84th percentiles. It stays sensible where 1/parallax
    /// does not, for faint stars and small or negative parallaxes.
    pub distance: Option<f64>,
    pub distance_low: Option<f64>,
    pub distance_high: Option<f64>,
}

impl GaiaDistance {
    /// The best distance in parsecs: Bailer-Jones's, or else 1/parallax when
    /// the parallax is at least five times its error.
    pub fn best_distance(&self) -> Option<f64> {
        self.distance.or_else(|| {
            let parallax = self.parallax?;
            let error = self.parallax_error?;
            (parallax > 0.0 && parallax >= 5.0 * error).then(|| 1000.0 / parallax)
        })
    }
}

/// Parse the CSV [`SourceDownloader::gaia_distance_cone_csv`] returns. Rows
/// with a missing position or G magnitude are skipped.
pub fn parse_gaia_distances(csv: &str) -> Result<Vec<GaiaDistance>> {
    let table = CsvTable::new(csv, || Error::MalformedGaiaChunk)?;
    let [
        ra,
        dec,
        pmra,
        pmdec,
        g,
        bp,
        rp,
        parallax,
        parallax_error,
        r_med,
        r_lo,
        r_hi,
    ] = [
        "ra",
        "dec",
        "pmra",
        "pmdec",
        "phot_g_mean_mag",
        "phot_bp_mean_mag",
        "phot_rp_mean_mag",
        "parallax",
        "parallax_error",
        "r_med_geo",
        "r_lo_geo",
        "r_hi_geo",
    ]
    .map(|name| table.column(name));
    let indices = [
        ra?,
        dec?,
        pmra?,
        pmdec?,
        g?,
        bp?,
        rp?,
        parallax?,
        parallax_error?,
        r_med?,
        r_lo?,
        r_hi?,
    ];
    Ok(table
        .rows()
        .filter_map(|row| {
            let value = |index: usize| row.number(indices[index]);
            let (ra, dec, g) = (value(0)?, value(1)?, value(4)?);
            let bp_rp = value(5).zip(value(6)).map(|(bp, rp)| (bp - rp) as f32);
            Some(GaiaDistance {
                ra,
                dec,
                pmra: value(2),
                pmdec: value(3),
                g: g as f32,
                bp_rp,
                parallax: value(7),
                parallax_error: value(8),
                distance: value(9),
                distance_low: value(10),
                distance_high: value(11),
            })
        })
        .collect())
}

/// One star of the Hipparcos new reduction (van Leeuwen 2007), from
/// [`SourceDownloader::hipparcos_cone`]. Gaia leaves the brightest stars
/// without parallaxes; Hipparcos measured them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HipparcosStar {
    /// Hipparcos catalogue number.
    pub hip: u32,
    /// ICRS position at epoch J1991.25, degrees.
    pub ra: f64,
    pub dec: f64,
    /// Parallax and its standard error, mas.
    pub parallax: Option<f64>,
    pub parallax_error: Option<f64>,
    /// Hipparcos magnitude.
    pub hp_mag: Option<f32>,
}

impl HipparcosStar {
    /// 1/parallax in parsecs, when the parallax is at least five times its
    /// error.
    pub fn distance(&self) -> Option<f64> {
        let parallax = self.parallax?;
        let error = self.parallax_error?;
        (parallax > 0.0 && parallax >= 5.0 * error).then(|| 1000.0 / parallax)
    }
}

/// Parse the CSV [`SourceDownloader::hipparcos_cone`] fetches.
pub fn parse_hipparcos(csv: &str) -> Result<Vec<HipparcosStar>> {
    let table = CsvTable::new(csv, || Error::MalformedHipparcos)?;
    let indices = ["HIP", "RArad", "DErad", "Plx", "e_Plx", "Hpmag"].map(|name| table.column(name));
    let [hip, ra, dec, plx, e_plx, hp] = indices;
    let indices = [hip?, ra?, dec?, plx?, e_plx?, hp?];
    Ok(table
        .rows()
        .filter_map(|row| {
            let value = |index: usize| row.number(indices[index]);
            Some(HipparcosStar {
                hip: u32::try_from(value(0)? as i64).ok()?,
                ra: value(1)?,
                dec: value(2)?,
                parallax: value(3),
                parallax_error: value(4),
                hp_mag: value(5).map(|value| value as f32),
            })
        })
        .collect())
}

/// A comma-separated table with a header row, as TAP services return.
struct CsvTable<'a> {
    columns: Vec<&'a str>,
    body: std::str::Lines<'a>,
    malformed: fn() -> Error,
}

impl<'a> CsvTable<'a> {
    fn new(csv: &'a str, malformed: fn() -> Error) -> Result<Self> {
        let mut body = csv.lines();
        let header = body.next().ok_or_else(malformed)?;
        Ok(Self {
            columns: header
                .split(',')
                .map(|column| column.trim().trim_matches('"'))
                .collect(),
            body,
            malformed,
        })
    }

    fn column(&self, name: &str) -> Result<usize> {
        self.columns
            .iter()
            .position(|column| *column == name)
            .ok_or_else(self.malformed)
    }

    fn rows(self) -> impl Iterator<Item = CsvRow<'a>> {
        self.body
            .filter(|line| !line.trim().is_empty())
            .map(|line| CsvRow {
                fields: line.split(',').map(str::trim).collect(),
            })
    }
}

struct CsvRow<'a> {
    fields: Vec<&'a str>,
}

impl CsvRow<'_> {
    fn number(&self, index: usize) -> Option<f64> {
        self.fields
            .get(index)
            .and_then(|field| field.parse::<f64>().ok())
            .filter(|value| value.is_finite())
    }
}

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
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(READ_TIMEOUT)
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
        let marker = output.join("outlines").join(".complete");
        if tokio::fs::metadata(&marker).await.is_err() {
            let archive = output.join("openngc-master.tar.gz");
            self.fetch(OPENNGC_ARCHIVE, &archive, Verify::Gzip).await?;
            let archive_for_task = archive.clone();
            let output_for_task = output.to_path_buf();
            tokio::task::spawn_blocking(move || {
                extract_openngc_outlines(&archive_for_task, &output_for_task)
            })
            .await
            .map_err(|error| Error::BackgroundTask(error.to_string()))??;
            tokio::fs::write(&marker, b"OpenNGC master outlines extracted\n")
                .await
                .map_err(|source| io("write", &marker, source))?;
        }
        self.ready("OpenNGC", output);
        Ok(())
    }

    /// Download a pinned GitHub curation repository snapshot without invoking
    /// Git. Existing snapshots are reused only when their recorded commit
    /// matches exactly.
    pub async fn download_curation(
        &self,
        repository: &str,
        commit: &str,
        output: impl AsRef<Path>,
    ) -> Result<()> {
        if !valid_repository(repository) {
            return Err(Error::InvalidRepository(repository.into()));
        }
        if !(7..=40).contains(&commit.len()) || !commit.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(Error::InvalidRevision(commit.into()));
        }
        let output = output.as_ref();
        let marker = output.join(".seiza-revision");
        if tokio::fs::read_to_string(&marker)
            .await
            .is_ok_and(|value| value.trim() == commit)
        {
            self.ready("catalog curation", output);
            return Ok(());
        }
        if let Ok(mut entries) = tokio::fs::read_dir(output).await
            && entries
                .next_entry()
                .await
                .map_err(|source| io("read directory", output, source))?
                .is_some()
        {
            return Err(Error::Integrity(format!(
                "{} contains a different or unpinned curation snapshot",
                output.display()
            )));
        }
        create_dir_all(output).await?;
        let archive = output.join("curation.tar.gz");
        let url = format!("https://github.com/{repository}/archive/{commit}.tar.gz");
        self.fetch(&url, &archive, Verify::Gzip).await?;
        let archive_for_task = archive.clone();
        let output_for_task = output.to_path_buf();
        tokio::task::spawn_blocking(move || {
            extract_github_snapshot(&archive_for_task, &output_for_task)
        })
        .await
        .map_err(|error| Error::BackgroundTask(error.to_string()))??;
        tokio::fs::remove_file(&archive)
            .await
            .map_err(|source| io("remove", &archive, source))?;
        tokio::fs::write(&marker, format!("{commit}\n"))
            .await
            .map_err(|source| io("write", &marker, source))?;
        self.ready("catalog curation", output);
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

    /// The catalogues behind the object distance file: VizieR tables of
    /// cluster, nebula, supernova-remnant, molecular-cloud and galaxy
    /// distances, then SIMBAD's distance measurements and galaxy redshifts
    /// for the designations in `objects.bin`, and the parallaxes of the
    /// stars that light van den Bergh's reflection nebulae. Files already
    /// downloaded are kept, so an interrupted download resumes.
    pub async fn download_object_distances(&self, output: impl AsRef<Path>) -> Result<()> {
        let output = output.as_ref();
        create_dir_all(output).await?;
        for (name, table, columns) in DISTANCE_TABLES {
            let header = columns.split(',').next().unwrap_or_default();
            self.fetch(
                &format!("{VIZIER_TSV}{table}&-out={columns}&-out.max=unlimited"),
                &output.join(name),
                Verify::Text {
                    header: header.to_string(),
                },
            )
            .await?;
        }
        let designations = SIMBAD_DESIGNATIONS
            .iter()
            .map(|pattern| format!("i.id LIKE '{pattern}'"))
            .collect::<Vec<_>>()
            .join(" OR ");
        self.fetch_simbad(
            &format!(
                "SELECT i.id, d.oidref, b.ra, b.dec, d.dist, d.unit, d.minus_err, \
                 d.plus_err, d.method, d.bibcode FROM mesDistance AS d \
                 JOIN ident AS i ON i.oidref = d.oidref JOIN basic AS b ON b.oid = d.oidref \
                 WHERE {designations}"
            ),
            &output.join("simbad-distances.csv"),
        )
        .await?;
        self.fetch_simbad(
            "SELECT i.id, b.rvz_redshift, b.rvz_qual, b.rvz_bibcode FROM ident AS i \
             JOIN basic AS b ON b.oid = i.oidref WHERE (i.id LIKE 'LEDA %' \
             OR i.id LIKE 'UGC %' OR i.id LIKE 'NGC %' OR i.id LIKE 'IC %' \
             OR i.id LIKE 'M %' OR i.id LIKE 'HCG %') AND b.rvz_redshift IS NOT NULL",
            &output.join("simbad-redshifts.csv"),
        )
        .await?;
        let vdb = output.join("vdb-stars.tsv");
        let stars = tokio::fs::read_to_string(&vdb)
            .await
            .map_err(|source| io("read", &vdb, source))?;
        let identifiers = vdb_star_identifiers(&stars)
            .into_iter()
            .map(|identifier| format!("'{identifier}'"))
            .collect::<Vec<_>>();
        if identifiers.is_empty() {
            return Err(Error::Integrity(format!(
                "{} lists no illuminating stars",
                vdb.display()
            )));
        }
        self.fetch_simbad(
            &format!(
                "SELECT i.id, b.plx_value, b.plx_err, b.plx_bibcode FROM ident AS i \
                 JOIN basic AS b ON b.oid = i.oidref WHERE i.id IN ({})",
                identifiers.join(",")
            ),
            &output.join("simbad-vdb-stars.csv"),
        )
        .await?;
        self.ready("object distance sources", output);
        Ok(())
    }

    /// Run an ADQL query on SIMBAD's TAP service into a CSV file, unless an
    /// earlier run left a complete one.
    async fn fetch_simbad(&self, query: &str, target: &Path) -> Result<()> {
        let form = [
            ("REQUEST", "doQuery".to_string()),
            ("LANG", "ADQL".to_string()),
            ("FORMAT", "csv".to_string()),
            ("MAXREC", SIMBAD_MAXREC.to_string()),
            ("QUERY", query.to_string()),
        ];
        let header = query
            .trim_start_matches("SELECT ")
            .split([',', ' '])
            .next()
            .unwrap_or_default()
            .rsplit('.')
            .next()
            .unwrap_or_default()
            .to_string();
        self.fetch_request(
            SIMBAD_TAP,
            || self.client.post(SIMBAD_TAP).form(&form),
            target,
            Verify::Text { header },
            false,
        )
        .await?;
        if count_rows(target).await? >= SIMBAD_MAXREC {
            let _ = tokio::fs::remove_file(target).await;
            return Err(Error::Integrity(format!(
                "SIMBAD query for {} reached its {SIMBAD_MAXREC}-row cap",
                target.display()
            )));
        }
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
        self.download_gaia_columns(
            output.as_ref(),
            max_mag,
            chunks,
            "gaia-",
            "ra, dec, pmra, pmdec, phot_g_mean_mag",
            GaiaArchive::Esa,
        )
        .await
    }

    /// [`Self::download_gaia`] with BP and RP photometry and RUWE, for a
    /// colour-calibration catalogue: chunks `gaiaphot-NNNN.csv` with columns
    /// ra, dec, pmra, pmdec, phot_g_mean_mag, phot_bp_mean_mag,
    /// phot_rp_mean_mag and ruwe. Completed chunks are kept, so an
    /// interrupted download resumes, from either archive.
    pub async fn download_gaia_photometry(
        &self,
        output: impl AsRef<Path>,
        max_mag: f32,
        chunks: u64,
        archive: GaiaArchive,
    ) -> Result<()> {
        self.download_gaia_columns(
            output.as_ref(),
            max_mag,
            chunks,
            "gaiaphot-",
            "ra, dec, pmra, pmdec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, ruwe",
            archive,
        )
        .await
    }

    async fn download_gaia_columns(
        &self,
        output: &Path,
        max_mag: f32,
        chunks: u64,
        prefix: &str,
        columns: &str,
        archive: GaiaArchive,
    ) -> Result<()> {
        if !max_mag.is_finite() {
            return Err(Error::InvalidGaiaMagnitude(max_mag));
        }
        if chunks == 0 || chunks > GAIA_SOURCE_ID_MAX {
            return Err(Error::InvalidGaiaChunks {
                chunks,
                max: GAIA_SOURCE_ID_MAX,
            });
        }

        create_dir_all(output).await?;
        let mut completed = 0u64;

        for chunk in 0..chunks {
            let target = output.join(format!("{prefix}{chunk:04}.csv"));
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
            let pieces = archive.chunk_pieces().min(hi - lo + 1);
            let mut rows = 0u64;
            let mut piece_paths = Vec::new();
            for piece in 0..pieces {
                let span = (hi - lo + 1) / pieces;
                let piece_lo = lo + span * piece;
                let piece_hi = if piece + 1 == pieces {
                    hi
                } else {
                    piece_lo + span - 1
                };
                let piece_target = if pieces == 1 {
                    target.clone()
                } else {
                    piece_path(&target, piece)
                };
                piece_paths.push(piece_target.clone());
                rows += self
                    .fetch_gaia_range(
                        columns,
                        max_mag,
                        (piece_lo, piece_hi),
                        &piece_target,
                        archive,
                        &format!("Gaia chunk {chunk:04}"),
                    )
                    .await?;
            }
            if pieces > 1 {
                join_gaia_pieces(&piece_paths, &target).await?;
            } else {
                // Another archive may have left pieces of this chunk behind.
                for piece in 0..GaiaArchive::Gavo.chunk_pieces() {
                    let _ = tokio::fs::remove_file(piece_path(&target, piece)).await;
                }
            }
            if rows >= GAIA_MAXREC {
                return Err(Error::GaiaRowCap {
                    chunk,
                    limit: GAIA_MAXREC,
                    suggested_chunks: chunks.saturating_mul(4).min(GAIA_SOURCE_ID_MAX),
                });
            }
            completed += 1;
            (self.reporter)(SourceEvent::GaiaChunkComplete {
                chunk,
                rows,
                completed,
                total: chunks,
            });
        }
        self.ready("Gaia", output);
        Ok(())
    }

    /// Fetch one source_id range into `target`, retrying, unless an earlier
    /// run already finished it. Returns the row count.
    async fn fetch_gaia_range(
        &self,
        columns: &str,
        max_mag: f32,
        (lo, hi): (u64, u64),
        target: &Path,
        archive: GaiaArchive,
        label: &str,
    ) -> Result<u64> {
        if chunk_complete(target).await? {
            return count_rows(target).await;
        }
        let query = format!(
            "SELECT {columns} FROM {table} \
             WHERE phot_g_mean_mag <= {max_mag} AND source_id BETWEEN {lo} AND {hi}",
            table = archive.table()
        );
        let mut attempts = 0u32;
        loop {
            attempts += 1;
            match self.fetch_gaia_chunk(&query, target, archive).await {
                Ok(rows) => return Ok(rows),
                Err(error) if attempts < 4 => {
                    let delay = Duration::from_secs(5 * attempts as u64);
                    (self.reporter)(SourceEvent::Retry {
                        label: label.to_owned(),
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
        self.fetch_request(url, || self.client.get(url), target, verify, force)
            .await
    }

    /// Download what `request` returns into `target` through a temporary
    /// file, unless `target` already passes `verify` and `force` is false.
    /// `url` labels progress events and errors.
    async fn fetch_request(
        &self,
        url: &str,
        request: impl FnOnce() -> reqwest::RequestBuilder,
        target: &Path,
        verify: Verify,
        force: bool,
    ) -> Result<()> {
        if !force && verify_file(target, verify.clone()).await? {
            (self.reporter)(SourceEvent::AlreadyPresent {
                path: target.to_path_buf(),
            });
            return Ok(());
        }

        (self.reporter)(SourceEvent::Fetching {
            url: url.into(),
            path: target.to_path_buf(),
        });
        let response = request().send().await.map_err(|source| Error::Http {
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
        let transfer = async {
            let mut output = tokio::fs::File::create(&temp)
                .await
                .map_err(|source| io("create", &temp, source))?;
            let mut stream = response.bytes_stream();
            let mut downloaded = 0u64;
            let mut reported = 0u64;
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
            drop(output);
            if !verify_file(&temp, verify).await? {
                return Err(Error::Integrity(url.into()));
            }
            replace_file(&temp, target).await
        }
        .await;
        if transfer.is_err() {
            let _ = tokio::fs::remove_file(&temp).await;
        }
        transfer
    }

    /// Gaia DR3 sources within `radius_deg` of `(ra, dec)` down to G
    /// `max_mag`, brightest first, with their BP and RP photometry, from the
    /// ESA Gaia archive. A field holding [`GAIA_CONE_MAXREC`] stars or more
    /// is refused rather than cut short: use a brighter limit.
    pub async fn gaia_photometry_cone(
        &self,
        ra: f64,
        dec: f64,
        radius_deg: f64,
        max_mag: f32,
    ) -> Result<Vec<GaiaPhotometry>> {
        parse_gaia_photometry(
            &self
                .gaia_photometry_cone_csv(ra, dec, radius_deg, max_mag)
                .await?,
        )
    }

    /// [`Self::gaia_photometry_cone`] as the archive's CSV, for a caller
    /// that caches it; [`parse_gaia_photometry`] reads it.
    /// ESA's archive answers first; when it fails, GAVO's mirror does.
    pub async fn gaia_photometry_cone_csv(
        &self,
        ra: f64,
        dec: f64,
        radius_deg: f64,
        max_mag: f32,
    ) -> Result<String> {
        match self
            .gaia_photometry_cone_csv_from(GaiaArchive::Esa, ra, dec, radius_deg, max_mag)
            .await
        {
            Ok(csv) => Ok(csv),
            Err(esa) => {
                (self.reporter)(SourceEvent::Retry {
                    label: "Gaia cone search on the GAVO mirror".into(),
                    attempt: 1,
                    delay: Duration::ZERO,
                    error: esa.to_string(),
                });
                self.gaia_photometry_cone_csv_from(GaiaArchive::Gavo, ra, dec, radius_deg, max_mag)
                    .await
                    .map_err(|_| esa)
            }
        }
    }

    /// [`Self::gaia_photometry_cone_csv`] from one archive.
    pub async fn gaia_photometry_cone_csv_from(
        &self,
        archive: GaiaArchive,
        ra: f64,
        dec: f64,
        radius_deg: f64,
        max_mag: f32,
    ) -> Result<String> {
        check_cone(ra, dec, radius_deg, max_mag)?;
        let query = format!(
            "SELECT ra, dec, pmra, pmdec, phot_g_mean_mag, phot_bp_mean_mag, \
             phot_rp_mean_mag, ruwe FROM {table} \
             WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) \
             AND phot_g_mean_mag <= {max_mag} ORDER BY phot_g_mean_mag",
            table = archive.table()
        );
        let body = self.gaia_cone_job(archive, query).await?;
        // Check it parses before a caller caches it.
        parse_gaia_photometry(&body)?;
        Ok(body)
    }

    /// Gaia DR3 sources within `radius_deg` of `(ra, dec)` down to G
    /// `max_mag`, brightest first, with their parallaxes and Bailer-Jones
    /// distances. ESA's archive answers first; when it fails, GAVO's mirror
    /// does. A field holding [`GAIA_CONE_MAXREC`] stars or more is refused.
    pub async fn gaia_distance_cone(
        &self,
        ra: f64,
        dec: f64,
        radius_deg: f64,
        max_mag: f32,
    ) -> Result<Vec<GaiaDistance>> {
        parse_gaia_distances(
            &self
                .gaia_distance_cone_csv(ra, dec, radius_deg, max_mag)
                .await?,
        )
    }

    /// [`Self::gaia_distance_cone`] as the archive's CSV, for a caller that
    /// caches it; [`parse_gaia_distances`] reads it.
    pub async fn gaia_distance_cone_csv(
        &self,
        ra: f64,
        dec: f64,
        radius_deg: f64,
        max_mag: f32,
    ) -> Result<String> {
        check_cone(ra, dec, radius_deg, max_mag)?;
        let query = |archive| gaia_distance_query(archive, ra, dec, radius_deg, max_mag);
        // Each archive's synchronous endpoint answers a cone in seconds; a
        // queued job can wait many minutes to start. So both archives are
        // asked directly before either queue.
        for archive in [GaiaArchive::Esa, GaiaArchive::Gavo] {
            if let Ok(body) = self.gaia_cone_sync(archive, &query(archive)).await
                && parse_gaia_distances(&body).is_ok()
            {
                return Ok(body);
            }
        }
        let queued = |archive: GaiaArchive| async move {
            let body = self.gaia_cone_job(archive, query(archive)).await?;
            parse_gaia_distances(&body)?;
            Ok::<_, Error>(body)
        };
        match queued(GaiaArchive::Esa).await {
            Ok(body) => Ok(body),
            Err(esa) => {
                (self.reporter)(SourceEvent::Retry {
                    label: "Gaia distance search on the GAVO mirror".into(),
                    attempt: 1,
                    delay: Duration::ZERO,
                    error: esa.to_string(),
                });
                queued(GaiaArchive::Gavo).await.map_err(|_| esa)
            }
        }
    }

    /// [`Self::gaia_distance_cone_csv`] from one archive.
    pub async fn gaia_distance_cone_csv_from(
        &self,
        archive: GaiaArchive,
        ra: f64,
        dec: f64,
        radius_deg: f64,
        max_mag: f32,
    ) -> Result<String> {
        check_cone(ra, dec, radius_deg, max_mag)?;
        let query = gaia_distance_query(archive, ra, dec, radius_deg, max_mag);
        // A cone of a few thousand stars answers in seconds on the
        // synchronous endpoint, while a queued job can wait minutes for the
        // archive to start it. A wide cone the synchronous endpoint times
        // out on goes through the job queue instead.
        let body = match self.gaia_cone_sync(archive, &query).await {
            Ok(body) if parse_gaia_distances(&body).is_ok() => body,
            _ => self.gaia_cone_job(archive, query).await?,
        };
        // Check it parses before a caller caches it.
        parse_gaia_distances(&body)?;
        Ok(body)
    }

    /// Run a Gaia cone `query` on `archive`'s synchronous endpoint. An
    /// answer that is not CSV, such as a timeout reported as VOTable, is an
    /// error, and so is one that may have been cut at the row limit.
    async fn gaia_cone_sync(&self, archive: GaiaArchive, query: &str) -> Result<String> {
        let url = archive.sync_url();
        let http = |source| Error::Http {
            url: url.into(),
            source,
        };
        let response = self
            .client
            .post(url)
            .form(&[
                ("REQUEST", "doQuery"),
                ("LANG", "ADQL"),
                ("FORMAT", "csv"),
                ("MAXREC", &GAIA_CONE_MAXREC.to_string()),
                ("QUERY", query),
            ])
            .send()
            .await
            .map_err(http)?;
        if !response.status().is_success() {
            return Err(Error::HttpStatus {
                url: url.into(),
                status: response.status().as_u16(),
            });
        }
        let body = response.text().await.map_err(http)?;
        if body.trim_start().starts_with('<') {
            return Err(Error::GaiaJobFailed(
                "answered with an error document".into(),
            ));
        }
        let rows = body
            .lines()
            .skip(1)
            .filter(|line| !line.trim().is_empty())
            .count();
        if rows as u64 >= GAIA_CONE_MAXREC {
            return Err(Error::GaiaJobFailed(format!(
                "returned its {GAIA_CONE_MAXREC}-star limit; use a brighter magnitude limit"
            )));
        }
        Ok(body)
    }

    /// Hipparcos stars (van Leeuwen 2007, VizieR I/311) within `radius_deg`
    /// of `(ra, dec)`, brightest first, from VizieR's TAP service. Gaia has
    /// no parallax for the brightest stars; these do.
    pub async fn hipparcos_cone(
        &self,
        ra: f64,
        dec: f64,
        radius_deg: f64,
    ) -> Result<Vec<HipparcosStar>> {
        check_cone(ra, dec, radius_deg, 0.0)?;
        let query = format!(
            "SELECT HIP, RArad, DErad, Plx, e_Plx, Hpmag FROM \"I/311/hip2\" \
             WHERE 1 = CONTAINS(POINT('ICRS', RArad, DErad), \
             CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) ORDER BY Hpmag"
        );
        let url = VIZIER_TAP_SYNC;
        let http = |source| Error::Http {
            url: url.into(),
            source,
        };
        let response = self
            .client
            .post(url)
            .form(&[
                ("REQUEST", "doQuery"),
                ("LANG", "ADQL"),
                ("FORMAT", "csv"),
                ("QUERY", query.as_str()),
            ])
            .send()
            .await
            .map_err(http)?;
        if !response.status().is_success() {
            return Err(Error::HttpStatus {
                url: url.into(),
                status: response.status().as_u16(),
            });
        }
        parse_hipparcos(&response.text().await.map_err(http)?)
    }

    /// Run a Gaia cone `query` as an asynchronous job on `archive` and return
    /// its CSV. A result holding [`GAIA_CONE_MAXREC`] rows or more is refused
    /// rather than cut short.
    async fn gaia_cone_job(&self, archive: GaiaArchive, query: String) -> Result<String> {
        // A wide field holds hundreds of thousands of stars, more than the
        // synchronous endpoint returns before it times out, so the query
        // runs as an asynchronous job: submit, poll, then fetch.
        let form = [
            ("REQUEST", "doQuery".to_string()),
            ("LANG", "ADQL".to_string()),
            ("FORMAT", "csv".to_string()),
            ("MAXREC", GAIA_CONE_MAXREC.to_string()),
            ("PHASE", "RUN".to_string()),
            ("QUERY", query),
        ];
        let http = |source| Error::Http {
            url: archive.async_url().into(),
            source,
        };
        let submitted = self
            .client
            .post(archive.async_url())
            .form(&form)
            .send()
            .await
            .map_err(http)?;
        if !submitted.status().is_success() {
            return Err(Error::HttpStatus {
                url: archive.async_url().into(),
                status: submitted.status().as_u16(),
            });
        }
        // The archive answers with a redirect to the job, which the client
        // follows: the final URL is the job's. Archives may keep jobs under
        // another path (GAVO's are under /__system__/tap/run/async/), so the
        // job must be on the archive's host, and somewhere other than where
        // the query was submitted.
        let job = submitted.url().to_string();
        let host = |url: &str| {
            url.split_once("://")
                .map(|(_, rest)| rest.split('/').next().unwrap_or_default().to_owned())
        };
        if host(&job) != host(archive.async_url())
            || job.trim_end_matches('/') == archive.async_url()
        {
            return Err(Error::GaiaJobFailed(format!(
                "was not queued: the archive answered from {job}"
            )));
        }
        let result = self.finish_gaia_job(&job).await;
        // Finished jobs only take space on the archive; clean up either way.
        let _ = self
            .client
            .post(&job)
            .form(&[("ACTION", "DELETE")])
            .send()
            .await;
        let body = result?;
        let rows = body
            .lines()
            .skip(1)
            .filter(|line| !line.trim().is_empty())
            .count();
        if rows as u64 >= GAIA_CONE_MAXREC {
            return Err(Error::GaiaJobFailed(format!(
                "returned its {GAIA_CONE_MAXREC}-star limit; use a brighter magnitude limit"
            )));
        }
        Ok(body)
    }

    /// Poll a queued TAP job until it completes, and return its result.
    async fn finish_gaia_job(&self, job: &str) -> Result<String> {
        let http = |source| Error::Http {
            url: job.into(),
            source,
        };
        let started = std::time::Instant::now();
        loop {
            let response = self
                .client
                .get(format!("{job}/phase"))
                .send()
                .await
                .map_err(http)?;
            if !response.status().is_success() {
                return Err(Error::HttpStatus {
                    url: format!("{job}/phase"),
                    status: response.status().as_u16(),
                });
            }
            let phase = response.text().await.map_err(http)?;
            match phase.trim() {
                "COMPLETED" => break,
                "PENDING" | "QUEUED" | "EXECUTING" => {
                    if started.elapsed() > GAIA_JOB_TIMEOUT {
                        return Err(Error::GaiaJobFailed("timed out".into()));
                    }
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
                "ERROR" => {
                    let reason = match self.client.get(format!("{job}/error")).send().await {
                        Ok(response) => response.text().await.unwrap_or_default(),
                        Err(_) => String::new(),
                    };
                    let reason = reason.trim();
                    return Err(Error::GaiaJobFailed(if reason.is_empty() {
                        "failed".into()
                    } else {
                        format!("failed: {}", reason.chars().take(500).collect::<String>())
                    }));
                }
                other => {
                    return Err(Error::GaiaJobFailed(format!(
                        "stopped in phase {}",
                        other.chars().take(40).collect::<String>()
                    )));
                }
            }
        }
        let response = self
            .client
            .get(format!("{job}/results/result"))
            .send()
            .await
            .map_err(http)?;
        if !response.status().is_success() {
            return Err(Error::HttpStatus {
                url: format!("{job}/results/result"),
                status: response.status().as_u16(),
            });
        }
        response.text().await.map_err(http)
    }

    async fn fetch_gaia_chunk(
        &self,
        query: &str,
        target: &Path,
        archive: GaiaArchive,
    ) -> Result<u64> {
        let url = archive.sync_url();
        let form = [
            ("REQUEST", "doQuery".to_string()),
            ("LANG", "ADQL".to_string()),
            ("FORMAT", "csv".to_string()),
            ("MAXREC", GAIA_MAXREC.to_string()),
            ("QUERY", query.to_string()),
        ];
        let response = self
            .client
            .post(url)
            .form(&form)
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

        let temp = partial_path(target);
        let transfer = async {
            let mut output = tokio::fs::File::create(&temp)
                .await
                .map_err(|source| io("create", &temp, source))?;
            let mut stream = response.bytes_stream();
            let mut prefix = Vec::with_capacity(6);
            let mut last = None;
            let mut newline_count = 0u64;
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|source| Error::Http {
                    url: url.into(),
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
            drop(output);
            if !prefix.starts_with(b"ra,dec") || last != Some(b'\n') || newline_count == 0 {
                return Err(Error::MalformedGaiaChunk);
            }
            replace_file(&temp, target).await?;
            Ok(newline_count - 1)
        }
        .await;
        if transfer.is_err() {
            let _ = tokio::fs::remove_file(&temp).await;
        }
        transfer
    }

    fn ready(&self, source: &'static str, output: &Path) {
        (self.reporter)(SourceEvent::Ready {
            source,
            directory: output.to_path_buf(),
        });
    }
}

fn extract_openngc_outlines(archive_path: &Path, output: &Path) -> Result<()> {
    let file =
        std::fs::File::open(archive_path).map_err(|source| io("open", archive_path, source))?;
    let decoder = GzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);
    let target = output.join("outlines").join("objects");
    std::fs::create_dir_all(&target).map_err(|source| io("create", &target, source))?;
    let entries = archive
        .entries()
        .map_err(|source| io("read archive", archive_path, source))?;
    let mut extracted = 0usize;
    for entry in entries {
        let mut entry = entry.map_err(|source| io("read archive entry", archive_path, source))?;
        let path = entry
            .path()
            .map_err(|source| io("read archive entry path", archive_path, source))?;
        let components = path.components().collect::<Vec<_>>();
        let Some(index) = components
            .windows(2)
            .position(|pair| pair[0].as_os_str() == "outlines" && pair[1].as_os_str() == "objects")
        else {
            continue;
        };
        if components.len() != index + 3 {
            continue;
        }
        let file_name = components[index + 2].as_os_str();
        if !file_name.to_string_lossy().ends_with(".txt") {
            continue;
        }
        let destination = target.join(file_name);
        let mut output_file = std::fs::File::create(&destination)
            .map_err(|source| io("create", &destination, source))?;
        std::io::copy(&mut entry, &mut output_file)
            .map_err(|source| io("extract", &destination, source))?;
        extracted += 1;
    }
    if extracted == 0 {
        return Err(Error::Integrity(
            "OpenNGC archive contained no outline files".into(),
        ));
    }
    Ok(())
}

fn extract_github_snapshot(archive_path: &Path, output: &Path) -> Result<()> {
    let file =
        std::fs::File::open(archive_path).map_err(|source| io("open", archive_path, source))?;
    let decoder = GzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);
    let entries = archive
        .entries()
        .map_err(|source| io("read archive", archive_path, source))?;
    let mut extracted = 0usize;
    for entry in entries {
        let mut entry = entry.map_err(|source| io("read archive entry", archive_path, source))?;
        let path = entry
            .path()
            .map_err(|source| io("read archive entry path", archive_path, source))?;
        let relative = path.components().skip(1).collect::<PathBuf>();
        if relative.as_os_str().is_empty()
            || relative
                .components()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            continue;
        }
        let destination = output.join(&relative);
        if entry.header().entry_type().is_dir() {
            std::fs::create_dir_all(&destination)
                .map_err(|source| io("create", &destination, source))?;
            continue;
        }
        if !entry.header().entry_type().is_file() {
            continue;
        }
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent).map_err(|source| io("create", parent, source))?;
        }
        let mut output_file = std::fs::File::create(&destination)
            .map_err(|source| io("create", &destination, source))?;
        std::io::copy(&mut entry, &mut output_file)
            .map_err(|source| io("extract", &destination, source))?;
        extracted += 1;
    }
    if extracted == 0 {
        return Err(Error::Integrity(
            "curation archive contained no regular files".into(),
        ));
    }
    Ok(())
}

fn valid_repository(value: &str) -> bool {
    let mut parts = value.split('/');
    let valid_part = |part: &str| {
        !part.is_empty()
            && part
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    };
    matches!((parts.next(), parts.next(), parts.next()), (Some(owner), Some(repo), None) if valid_part(owner) && valid_part(repo))
}

#[derive(Clone)]
enum Verify {
    None,
    Gzip,
    /// A text table: some line starts with `header`, and the file ends with
    /// a newline.
    Text {
        header: String,
    },
}

async fn verify_file(path: &Path, verify: Verify) -> Result<bool> {
    match verify {
        Verify::Text { header } => match tokio::fs::read(path).await {
            Ok(bytes) => Ok(text_table_complete(&bytes, &header)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(source) => Err(io("read", path, source)),
        },
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

/// A downloaded table is whole when a line starts with its header and the
/// last line is finished.
fn text_table_complete(bytes: &[u8], header: &str) -> bool {
    !header.is_empty()
        && bytes.last() == Some(&b'\n')
        && bytes
            .split(|&byte| byte == b'\n')
            .any(|line| line.starts_with(header.as_bytes()))
}

/// SIMBAD identifiers of the stars that light van den Bergh's reflection
/// nebulae, from VizieR's VII/21 table with columns VdB, DM and HD: the HD
/// number padded as SIMBAD writes it, else the Durchmusterung number.
/// Anything else is dropped, so nothing from the file reaches a query
/// unchecked.
fn vdb_star_identifiers(table: &str) -> Vec<String> {
    let mut identifiers = Vec::new();
    for line in table.lines().filter(|line| !line.starts_with('#')) {
        let fields = line.split('\t').map(str::trim).collect::<Vec<_>>();
        let [number, durchmusterung, hd, ..] = fields[..] else {
            continue;
        };
        if number.is_empty() || !number.bytes().all(|byte| byte.is_ascii_digit()) {
            continue;
        }
        if !hd.is_empty() && hd.len() <= 6 && hd.bytes().all(|byte| byte.is_ascii_digit()) {
            identifiers.push(format!("HD{hd:>7}"));
            continue;
        }
        let zone_ok = durchmusterung.len() > 6
            && ["BD", "CD", "CP"].contains(&&durchmusterung[..2])
            && matches!(durchmusterung.as_bytes()[2], b'+' | b'-')
            && durchmusterung.as_bytes()[3..5]
                .iter()
                .all(u8::is_ascii_digit);
        let number_ok = durchmusterung.get(5..).is_some_and(|rest| {
            let rest = rest.trim_start_matches(' ');
            !rest.is_empty() && rest.bytes().all(|byte| byte.is_ascii_digit())
        });
        if zone_ok && number_ok {
            identifiers.push(durchmusterung.to_string());
        }
    }
    identifiers
}

/// Where one source_id range of a chunk fetched in pieces is kept. The name
/// does not end in `.csv`, so a catalog build never reads it.
fn piece_path(target: &Path, piece: u64) -> PathBuf {
    let mut name = target.as_os_str().to_owned();
    name.push(format!(".piece{piece:02}"));
    PathBuf::from(name)
}

/// Rows in a finished CSV chunk, not counting its header.
async fn count_rows(path: &Path) -> Result<u64> {
    let bytes = tokio::fs::read(path)
        .await
        .map_err(|source| io("read", path, source))?;
    Ok(bytes.iter().filter(|&&byte| byte == b'\n').count() as u64 - 1)
}

/// Concatenate finished piece files under the first one's header into
/// `target`, then remove the pieces.
async fn join_gaia_pieces(pieces: &[PathBuf], target: &Path) -> Result<()> {
    let temp = partial_path(target);
    let mut output = tokio::fs::File::create(&temp)
        .await
        .map_err(|source| io("create", &temp, source))?;
    for (index, piece) in pieces.iter().enumerate() {
        let bytes = tokio::fs::read(piece)
            .await
            .map_err(|source| io("read", piece, source))?;
        let body = if index == 0 {
            &bytes[..]
        } else {
            let header_end = bytes
                .iter()
                .position(|&byte| byte == b'\n')
                .ok_or(Error::MalformedGaiaChunk)?;
            &bytes[header_end + 1..]
        };
        output
            .write_all(body)
            .await
            .map_err(|source| io("write", &temp, source))?;
    }
    output
        .sync_all()
        .await
        .map_err(|source| io("sync", &temp, source))?;
    drop(output);
    replace_file(&temp, target).await?;
    for piece in pieces {
        let _ = tokio::fs::remove_file(piece).await;
    }
    Ok(())
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

async fn replace_file(temp: &Path, target: &Path) -> Result<()> {
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

    #[test]
    fn gaia_distances_parse_with_and_without_parallaxes() {
        // Rows as ESA's archive returned them for the Pleiades: Alcyone has
        // no parallax in DR3, Atlas does.
        let csv = "ra,dec,pmra,pmdec,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag,\
                   parallax,parallax_error,r_med_geo,r_lo_geo,r_hi_geo\n\
                   56.87125,24.10493,,,2.896132,,,,,,,\n\
                   57.29068,24.05321,17.6,-45.1,3.6157947,3.55,3.70,8.118448,0.47909123,\
                   125.42968,117.72701,134.53589\n\
                   ,24.0,,,9.0,,,,,,,\n";
        let stars = parse_gaia_distances(csv).unwrap();
        assert_eq!(stars.len(), 2, "a row without a position is skipped");
        assert_eq!(stars[0].parallax, None);
        assert_eq!(stars[0].best_distance(), None);
        assert_eq!(stars[1].distance, Some(125.42968));
        assert_eq!(stars[1].distance_low, Some(117.72701));
        assert!((stars[1].bp_rp.unwrap() - -0.15).abs() < 1e-5);
        assert_eq!(stars[1].best_distance(), Some(125.42968));
        assert!(parse_gaia_distances("ra,dec\n1,2\n").is_err());
    }

    #[test]
    fn best_distance_falls_back_to_a_precise_parallax_only() {
        let star = |parallax: f64, error: f64| GaiaDistance {
            ra: 0.0,
            dec: 0.0,
            pmra: None,
            pmdec: None,
            g: 10.0,
            bp_rp: None,
            parallax: Some(parallax),
            parallax_error: Some(error),
            distance: None,
            distance_low: None,
            distance_high: None,
        };
        assert_eq!(star(10.0, 1.0).best_distance(), Some(100.0));
        assert_eq!(star(1.0, 0.5).best_distance(), None);
        assert_eq!(star(-1.0, 0.1).best_distance(), None);
    }

    #[test]
    fn hipparcos_rows_parse() {
        let csv = "HIP,RArad,DErad,Plx,e_Plx,Hpmag,B-V\n\
                   17702,56.87110081,24.10524179,8.09,0.42,2.848,-0.086\n\
                   17847,57.29054699,24.05352413,0.5,0.9,3.6084,-0.07\n";
        let stars = parse_hipparcos(csv).unwrap();
        assert_eq!(stars.len(), 2);
        assert_eq!(stars[0].hip, 17702);
        assert!((stars[0].distance().unwrap() - 1000.0 / 8.09).abs() < 1e-9);
        assert_eq!(
            stars[1].distance(),
            None,
            "an imprecise parallax gives none"
        );
        assert!(matches!(
            parse_hipparcos("HIP,RArad\n1,2\n"),
            Err(Error::MalformedHipparcos)
        ));
    }

    #[test]
    fn gaia_photometry_rows_parse_with_missing_colours() {
        let csv = "ra,dec,pmra,pmdec,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag,ruwe\n\
                   56.75,24.11,19.9,-45.5,2.86,2.84,2.89,1.1\n\
                   56.80,24.20,,,12.5,,,\n\
                   ,24.3,1,1,13,13,12,1\n";
        let stars = parse_gaia_photometry(csv).unwrap();
        assert_eq!(stars.len(), 2);
        assert!((stars[0].bp_rp().unwrap() + 0.05).abs() < 1e-6);
        assert_eq!(stars[1].pmra, None);
        assert_eq!(stars[1].bp_rp(), None);
        assert!(parse_gaia_photometry("ra,dec\n1,2\n").is_err());
    }

    #[test]
    fn text_tables_are_complete_with_header_and_final_newline() {
        let vizier = b"#RESOURCE=yCat\n#Name: VII/21\n\nVdB\tDM\tHD\n \t \t \n---\t---\t---\n  1\tBD+57   22\t   627\n";
        assert!(text_table_complete(vizier, "VdB"));
        assert!(!text_table_complete(&vizier[..vizier.len() - 1], "VdB"));
        assert!(!text_table_complete(vizier, "Name,"));
        assert!(text_table_complete(
            b"id,plx_value\n\"HD    627\",1.0\n",
            "id"
        ));
        assert!(!text_table_complete(b"<VOTABLE>error</VOTABLE>\n", "id"));
        assert!(!text_table_complete(b"", ""));
    }

    #[test]
    fn vdb_star_identifiers_are_padded_and_sanitized() {
        let table = "#Column\tVdB\nVdB\tDM\tHD\n \t \t \n---\t----------\t------\n\
                     \x20 1\tBD+57   22\t   627\n\
                     \x20 2\tBD+64   13\t      \n\
                     \x20 3\tCD-27 3174\t143018\n\
                     \x20 4\tBD+64 13') OR 1=1 --\t\n\
                     \x20 5\tXX+01    1\t\n\
                     \x20 6\tBD+61  154\t12a\n";
        assert_eq!(
            vdb_star_identifiers(table),
            ["HD    627", "BD+64   13", "HD 143018", "BD+61  154"]
        );
    }

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
    async fn gaia_pieces_join_under_one_header() {
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("gaiaphot-0007.csv");
        let pieces = [
            (piece_path(&target, 0), &b"ra,dec\n1,2\n3,4\n"[..]),
            (piece_path(&target, 1), &b"ra,dec\n"[..]),
            (piece_path(&target, 2), &b"ra,dec\n5,6\n"[..]),
        ];
        for (path, bytes) in &pieces {
            tokio::fs::write(path, bytes).await.unwrap();
        }
        assert_eq!(
            pieces[1].0.file_name().unwrap(),
            "gaiaphot-0007.csv.piece01"
        );
        let paths = pieces
            .iter()
            .map(|(path, _)| path.clone())
            .collect::<Vec<_>>();
        join_gaia_pieces(&paths, &target).await.unwrap();
        assert_eq!(
            tokio::fs::read(&target).await.unwrap(),
            b"ra,dec\n1,2\n3,4\n5,6\n"
        );
        assert_eq!(count_rows(&target).await.unwrap(), 3);
        assert!(paths.iter().all(|path| !path.exists()));
    }

    #[tokio::test]
    async fn gaia_rejects_invalid_arguments_before_creating_output() {
        let directory = tempfile::tempdir().unwrap();
        let downloader = SourceDownloader::new().unwrap();

        let zero = directory.path().join("zero");
        assert!(matches!(
            downloader.download_gaia(&zero, 15.0, 0).await,
            Err(Error::InvalidGaiaChunks { chunks: 0, .. })
        ));
        assert!(!zero.exists());

        let excessive = directory.path().join("excessive");
        assert!(matches!(
            downloader
                .download_gaia(&excessive, 15.0, GAIA_SOURCE_ID_MAX + 1)
                .await,
            Err(Error::InvalidGaiaChunks { .. })
        ));
        assert!(!excessive.exists());

        let non_finite = directory.path().join("non-finite");
        assert!(matches!(
            downloader.download_gaia(&non_finite, f32::NAN, 1).await,
            Err(Error::InvalidGaiaMagnitude(value)) if value.is_nan()
        ));
        assert!(!non_finite.exists());
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

    #[tokio::test]
    async fn replace_file_overwrites_existing_target() {
        let directory = tempfile::tempdir().unwrap();
        let temp = directory.path().join("download.part");
        let target = directory.path().join("catalog.dat");
        tokio::fs::write(&temp, b"new catalog").await.unwrap();
        tokio::fs::write(&target, b"old catalog").await.unwrap();

        replace_file(&temp, &target).await.unwrap();

        assert_eq!(tokio::fs::read(&target).await.unwrap(), b"new catalog");
        assert!(!temp.exists());
    }

    #[test]
    fn extracts_only_openngc_outline_objects() {
        use std::io::Write;

        let directory = tempfile::tempdir().unwrap();
        let archive_path = directory.path().join("openngc.tar.gz");
        let file = std::fs::File::create(&archive_path).unwrap();
        let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        let mut archive = tar::Builder::new(encoder);
        for (path, bytes) in [
            (
                "OpenNGC-master/outlines/objects/NGC7000_lv1.txt",
                b"outline".as_slice(),
            ),
            (
                "OpenNGC-master/database_files/NGC.csv",
                b"catalog".as_slice(),
            ),
        ] {
            let mut header = tar::Header::new_gnu();
            header.set_size(bytes.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            archive.append_data(&mut header, path, bytes).unwrap();
        }
        let mut encoder = archive.into_inner().unwrap();
        encoder.flush().unwrap();
        encoder.finish().unwrap();

        let output = directory.path().join("output");
        extract_openngc_outlines(&archive_path, &output).unwrap();
        assert_eq!(
            std::fs::read(output.join("outlines/objects/NGC7000_lv1.txt")).unwrap(),
            b"outline"
        );
        assert!(!output.join("database_files/NGC.csv").exists());
    }

    #[test]
    fn validates_github_repository_names() {
        assert!(valid_repository("theatrus/seiza-catalog-curation"));
        assert!(!valid_repository("theatrus"));
        assert!(!valid_repository("https://github.com/theatrus/seiza"));
        assert!(!valid_repository("owner/repo/extra"));
    }
}
