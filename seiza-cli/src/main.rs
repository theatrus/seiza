use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use seiza::catalog::{StarCatalog, TileCatalog};
use seiza::objects::{ObjectCatalog, ObjectKind, ObjectQuery, ObjectSort, SkyRegion};
use seiza::solve::{SolveHint, solve};
use seiza::{DetectConfig, detect_stars};
use std::path::PathBuf;

mod astap;
mod build_data;
mod download_data;

/// Open an image file; FITS files are MTF-autostretched to 8-bit grayscale.
pub(crate) fn load_image(path: &std::path::Path) -> Result<image::DynamicImage> {
    let is_fits = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("fits") || e.eq_ignore_ascii_case("fit"));
    if is_fits {
        let fits = seiza_fits::FitsImage::open(path)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        let stretched = fits.stretch_to_u8(&seiza_fits::StretchParams::default());
        let buffer = image::GrayImage::from_raw(fits.width as u32, fits.height as u32, stretched)
            .ok_or_else(|| anyhow::anyhow!("FITS dimensions mismatch"))?;
        return Ok(image::DynamicImage::ImageLuma8(buffer));
    }
    image::open(path).with_context(|| format!("failed to open {}", path.display()))
}

/// RA/Dec hint from FITS headers (N.I.N.A. writes RA/DEC in degrees).
fn fits_hint(path: &std::path::Path) -> Option<(f64, f64)> {
    let fits = seiza_fits::FitsImage::open(path).ok()?;
    let ra = fits
        .header_f64("RA")
        .or_else(|| fits.header_f64("OBJCTRA"))?;
    let dec = fits
        .header_f64("DEC")
        .or_else(|| fits.header_f64("OBJCTDEC"))?;
    Some((ra, dec))
}

#[derive(Parser)]
#[command(name = "seiza", about = "Star detection and plate solving", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Detect stars in an image and print their positions
    Detect {
        /// Image file (PNG, JPEG, TIFF)
        image: PathBuf,
        /// Detection threshold in noise sigmas
        #[arg(long, default_value_t = 4.0)]
        sigma: f32,
        /// Maximum number of stars to report
        #[arg(long, default_value_t = 100)]
        max_stars: usize,
        /// Write a copy of the image with detections circled
        #[arg(long)]
        annotate: Option<PathBuf>,
    },
    /// Plate-solve an image against a star catalog
    Solve {
        /// Image file (PNG, JPEG, TIFF)
        image: PathBuf,
        /// Star tile file built by build-data
        #[arg(long)]
        data: PathBuf,
        /// Approximate field center RA, degrees (default: FITS RA header)
        #[arg(long, allow_negative_numbers = true)]
        ra: Option<f64>,
        /// Approximate field center Dec, degrees (default: FITS DEC header)
        #[arg(long, allow_negative_numbers = true)]
        dec: Option<f64>,
        /// Search radius around the hinted center, degrees
        #[arg(long, default_value_t = 2.0)]
        radius: f64,
        /// Approximate pixel scale, arcseconds per pixel
        #[arg(long)]
        scale: f64,
        /// Allowed relative scale error
        #[arg(long, default_value_t = 0.2)]
        scale_tolerance: f64,
        /// Detection threshold in noise sigmas
        #[arg(long, default_value_t = 4.0)]
        sigma: f32,
        /// Ignore detections within this many pixels of the image edges
        /// (captions and watermarks)
        #[arg(long, default_value_t = 0)]
        ignore_border: u32,
        /// Write a copy with detections (green) and catalog stars projected
        /// through the solution (red)
        #[arg(long)]
        annotate: Option<PathBuf>,
        /// Object catalog file: list objects in the field and draw them
        /// (cyan) on the annotated copy
        #[arg(long)]
        objects: Option<PathBuf>,
        /// Minor-body element file (comets + asteroids); positions are
        /// propagated to the acquisition time
        #[arg(long)]
        minor_bodies: Option<PathBuf>,
        /// Acquisition time, ISO 8601 UTC (default: the FITS DATE-OBS
        /// header). Required for accurate minor-body positions
        #[arg(long)]
        time: Option<String>,
    },
    /// Download source catalogs from their archives
    DownloadData {
        #[command(subcommand)]
        source: DownloadSource,
    },
    /// Build catalog data bundles from primary sources
    BuildData {
        #[command(subcommand)]
        source: BuildDataSource,
    },
    /// Plate-solve with no position hint (wide fields, pixel-scale range)
    SolveBlind {
        /// Image file (PNG, JPEG, TIFF, FITS)
        image: PathBuf,
        /// Star tile file built by build-data
        #[arg(long)]
        data: PathBuf,
        /// Minimum plausible pixel scale, arcseconds per pixel
        #[arg(long, default_value_t = 0.5)]
        min_scale: f64,
        /// Maximum plausible pixel scale, arcseconds per pixel
        #[arg(long, default_value_t = 20.0)]
        max_scale: f64,
        /// Detection threshold in noise sigmas
        #[arg(long, default_value_t = 4.0)]
        sigma: f32,
        /// Ignore detections within this many pixels of the image edges
        #[arg(long, default_value_t = 0)]
        ignore_border: u32,
    },
    /// Inspect a FITS file: headers, statistics, optional stretched PNG
    FitsInfo {
        /// FITS file
        image: PathBuf,
        /// Write an autostretched 8-bit preview
        #[arg(long)]
        stretch: Option<PathBuf>,
    },
    /// Query a star tile file: list stars around a sky position
    Cone {
        /// Star tile file built by build-data
        #[arg(long)]
        data: PathBuf,
        #[arg(long, allow_negative_numbers = true)]
        ra: f64,
        #[arg(long, allow_negative_numbers = true)]
        dec: f64,
        /// Search radius in degrees
        #[arg(long, default_value_t = 1.0)]
        radius: f64,
        #[arg(long, default_value_t = 25)]
        limit: usize,
    },
    /// Query astronomical catalogs without plate-solving an image
    Catalog {
        #[command(subcommand)]
        query: CatalogCommand,
    },
}

#[derive(Subcommand)]
enum CatalogCommand {
    /// List named and deep-sky objects in a known sky region
    Objects {
        #[command(flatten)]
        args: CatalogObjectsArgs,
    },
}

#[derive(Args)]
struct CatalogObjectsArgs {
    /// Object catalog file built by build-data objects
    #[arg(long)]
    data: PathBuf,
    /// Cone center RA, ICRS degrees; use with --dec and --radius
    #[arg(long, allow_negative_numbers = true)]
    ra: Option<f64>,
    /// Cone center Dec, ICRS degrees; use with --ra and --radius
    #[arg(long, allow_negative_numbers = true)]
    dec: Option<f64>,
    /// Cone radius in degrees; use with --ra and --dec
    #[arg(long)]
    radius: Option<f64>,
    /// Convex image-footprint vertex as RA,DEC in boundary order; repeat 3+
    /// times. Conflicts with the cone arguments
    #[arg(
        long,
        value_name = "RA,DEC",
        allow_hyphen_values = true,
        value_parser = parse_sky_coordinate
    )]
    corner: Vec<SkyCoordinate>,
    /// Include only these object kinds; repeat or comma-separate values
    #[arg(long, value_delimiter = ',', value_parser = parse_object_kind)]
    kind: Vec<ObjectKind>,
    /// Faintest integrated visual magnitude to include; unknowns are excluded
    #[arg(long, allow_negative_numbers = true)]
    max_mag: Option<f32>,
    /// Minimum catalog major axis in arcminutes; unknowns are excluded
    #[arg(long)]
    min_size: Option<f32>,
    /// Require a common/popular name in addition to a catalog designation
    #[arg(long)]
    common_name_only: bool,
    /// Exclude objects whose extent overlaps the region but center is outside
    #[arg(long)]
    center_only: bool,
    /// Maximum results; zero means unlimited
    #[arg(long, default_value_t = 25)]
    limit: usize,
    #[arg(long, value_enum, default_value_t = CatalogSortArg::Prominence)]
    sort: CatalogSortArg,
    #[arg(long, value_enum, default_value_t = CatalogOutputFormat::Table)]
    format: CatalogOutputFormat,
}

#[derive(Debug, Clone, Copy)]
struct SkyCoordinate {
    ra: f64,
    dec: f64,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CatalogSortArg {
    Prominence,
    Size,
    Magnitude,
    Distance,
    Name,
}

impl From<CatalogSortArg> for ObjectSort {
    fn from(value: CatalogSortArg) -> Self {
        match value {
            CatalogSortArg::Prominence => Self::Prominence,
            CatalogSortArg::Size => Self::Size,
            CatalogSortArg::Magnitude => Self::Magnitude,
            CatalogSortArg::Distance => Self::Distance,
            CatalogSortArg::Name => Self::Name,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CatalogOutputFormat {
    Table,
    Json,
    Csv,
}

#[derive(Subcommand)]
enum DownloadSource {
    /// Tycho-2 distribution files from CDS (~150 MB)
    Tycho2 {
        /// Directory to download into
        #[arg(long)]
        output: PathBuf,
    },
    /// The OpenNGC deep-sky object catalog (~4 MB)
    Openngc {
        /// Directory to download into
        #[arg(long)]
        output: PathBuf,
    },
    /// All object-overlay sources: OpenNGC, Sh2, Barnard, UGC, LDN, vdB,
    /// IAU + Bright Star Catalogue star names
    Objects {
        /// Directory to download into
        #[arg(long)]
        output: PathBuf,
    },
    /// Gaia DR3 star positions via ESA TAP (resumable; can take hours —
    /// prefer `download-data prebuilt` unless you need a custom depth)
    Gaia {
        /// Directory to download into
        #[arg(long)]
        output: PathBuf,
        /// Magnitude limit for the download
        #[arg(long, default_value_t = 15.0)]
        max_mag: f32,
        /// Sky chunks (multiples of 192; 768 = HEALPix level 3). Deeper
        /// magnitude limits need more chunks to stay under the TAP row cap
        #[arg(long, default_value_t = 768)]
        chunks: u64,
    },
    /// The Rochester Astronomy active supernova/transient list (refetched)
    Transients {
        /// Directory to download into
        #[arg(long)]
        output: PathBuf,
    },
    /// Minor Planet Center element sets: comets + numbered asteroids
    Mpc {
        /// Directory to download into
        #[arg(long)]
        output: PathBuf,
    },
    /// Prebuilt datasets from downloads.seiza.fyi (SHA-256 verified) —
    /// the quickest route to a working solver
    Prebuilt {
        /// Directory to download into
        #[arg(long)]
        output: PathBuf,
        /// Specific files (default: everything in the manifest)
        #[arg(long)]
        file: Vec<String>,
    },
}

#[derive(Subcommand)]
enum BuildDataSource {
    /// An ASTAP .1476 star database directory (e.g. D80, Gaia DR3)
    Astap {
        /// Directory containing *.1476 files
        #[arg(long)]
        input: PathBuf,
        /// Output tile file
        #[arg(long)]
        output: PathBuf,
        /// Epoch of the source positions, Julian year (D80 is epoch 2025)
        #[arg(long, default_value_t = 2025.0)]
        epoch: f64,
        /// Drop stars fainter than this magnitude
        #[arg(long, default_value_t = 21.0)]
        max_mag: f32,
        /// Declination bands (tile granularity); 180 = 1° tiles
        #[arg(long, default_value_t = 180)]
        bands: u32,
    },
    /// Tycho-2 (CDS I/259): the ~2.5M star lite tier
    Tycho2 {
        /// Directory containing tyc2.dat.NN[.gz]
        #[arg(long)]
        input: PathBuf,
        /// Output tile file
        #[arg(long)]
        output: PathBuf,
        /// Epoch to apply proper motions to, Julian year
        #[arg(long, default_value_t = 2025.5)]
        epoch: f64,
        /// Drop stars fainter than this magnitude
        #[arg(long, default_value_t = 13.0)]
        max_mag: f32,
    },
    /// Object catalog from the downloaded object sources
    Objects {
        /// Directory containing the source files
        #[arg(long)]
        input: PathBuf,
        /// Output object catalog file
        #[arg(long)]
        output: PathBuf,
    },
    /// Star tiles from Gaia DR3 TAP chunks (download-data gaia)
    Gaia {
        /// Directory containing gaia-*.csv
        #[arg(long)]
        input: PathBuf,
        /// Output tile file
        #[arg(long)]
        output: PathBuf,
        /// Epoch to apply proper motions to, Julian year
        #[arg(long, default_value_t = 2025.5)]
        epoch: f64,
        /// Drop stars fainter than this magnitude
        #[arg(long, default_value_t = 15.0)]
        max_mag: f32,
        /// Declination bands (tile granularity); 180 = 1° tiles
        #[arg(long, default_value_t = 180)]
        bands: u32,
    },
    /// Transient catalog from the downloaded Rochester active list
    Transients {
        /// Directory containing snactive.html
        #[arg(long)]
        input: PathBuf,
        /// Output transient catalog file
        #[arg(long)]
        output: PathBuf,
    },
    /// Comets + bright numbered asteroids from MPC element sets
    MinorBodies {
        /// Directory containing CometEls.txt and MPCORB.DAT.gz
        #[arg(long)]
        input: PathBuf,
        /// Output element file
        #[arg(long)]
        output: PathBuf,
        /// Drop asteroids with absolute magnitude H above this
        #[arg(long, default_value_t = 16.0)]
        max_h: f32,
    },
    /// Bundle manifest (sizes + sha256) for hosting data files
    Manifest {
        /// Directory containing built .bin data files
        #[arg(long)]
        dir: PathBuf,
        /// Version string recorded in the manifest
        #[arg(long)]
        version: String,
        /// Output manifest path
        #[arg(long)]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    // A raw ASTAP-style command line (or a copy of the binary named
    // astap) routes to the ASTAP-compatible mode before clap sees it
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if astap::looks_like_astap(&raw) {
        return astap::run(&raw);
    }

    match Cli::parse().command {
        Command::Detect {
            image,
            sigma,
            max_stars,
            annotate,
        } => detect(&image, sigma, max_stars, annotate.as_deref()),
        Command::Solve {
            image,
            data,
            ra,
            dec,
            radius,
            scale,
            scale_tolerance,
            sigma,
            ignore_border,
            annotate,
            objects,
            minor_bodies,
            time,
        } => {
            let hint = match (ra, dec) {
                (Some(ra), Some(dec)) => (ra, dec),
                _ => fits_hint(&image).ok_or_else(|| {
                    anyhow::anyhow!("--ra/--dec required (no RA/DEC headers found in the image)")
                })?,
            };
            let acquisition_jd = resolve_acquisition_jd(&image, time.as_deref())?;
            solve_command(
                &image,
                &data,
                hint,
                radius,
                scale,
                scale_tolerance,
                sigma,
                ignore_border,
                annotate.as_deref(),
                objects.as_deref(),
                minor_bodies.as_deref(),
                acquisition_jd,
            )
        }
        Command::DownloadData { source } => match source {
            DownloadSource::Tycho2 { output } => download_data::download_tycho2(&output),
            DownloadSource::Openngc { output } => download_data::download_openngc(&output),
            DownloadSource::Objects { output } => download_data::download_objects(&output),
            DownloadSource::Gaia {
                output,
                max_mag,
                chunks,
            } => download_data::download_gaia(&output, max_mag, chunks),
            DownloadSource::Transients { output } => download_data::download_transients(&output),
            DownloadSource::Mpc { output } => download_data::download_mpc(&output),
            DownloadSource::Prebuilt { output, file } => {
                download_data::download_prebuilt(&output, &file)
            }
        },
        Command::BuildData { source } => match source {
            BuildDataSource::Astap {
                input,
                output,
                epoch,
                max_mag,
                bands,
            } => build_data::build_astap(&input, &output, epoch, max_mag, bands),
            BuildDataSource::Tycho2 {
                input,
                output,
                epoch,
                max_mag,
            } => build_data::build_tycho2(&input, &output, epoch, max_mag),
            BuildDataSource::Objects { input, output } => {
                build_data::build_objects(&input, &output)
            }
            BuildDataSource::Gaia {
                input,
                output,
                epoch,
                max_mag,
                bands,
            } => build_data::build_gaia(&input, &output, epoch, max_mag, bands),
            BuildDataSource::Transients { input, output } => {
                build_data::build_transients(&input, &output)
            }
            BuildDataSource::MinorBodies {
                input,
                output,
                max_h,
            } => build_data::build_minor_bodies(&input, &output, max_h),
            BuildDataSource::Manifest {
                dir,
                version,
                output,
            } => build_data::build_manifest(&dir, &version, &output),
        },
        Command::SolveBlind {
            image,
            data,
            min_scale,
            max_scale,
            sigma,
            ignore_border,
        } => solve_blind_command(&image, &data, min_scale, max_scale, sigma, ignore_border),
        Command::FitsInfo { image, stretch } => fits_info(&image, stretch.as_deref()),
        Command::Cone {
            data,
            ra,
            dec,
            radius,
            limit,
        } => cone(&data, ra, dec, radius, limit),
        Command::Catalog { query } => match query {
            CatalogCommand::Objects { args } => catalog_objects(args),
        },
    }
}

fn fits_info(path: &std::path::Path, stretch: Option<&std::path::Path>) -> Result<()> {
    let started = std::time::Instant::now();
    let fits = seiza_fits::FitsImage::open(path)
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let load_time = started.elapsed();
    println!(
        "{}: {}x{} ({:?}-type pixels), loaded in {:.0}ms",
        path.display(),
        fits.width,
        fits.height,
        match fits.pixels {
            seiza_fits::Pixels::U8(_) => "u8",
            seiza_fits::Pixels::U16(_) => "u16",
            seiza_fits::Pixels::I32(_) => "i32",
            seiza_fits::Pixels::F32(_) => "f32",
            seiza_fits::Pixels::F64(_) => "f64",
        },
        load_time.as_secs_f64() * 1000.0
    );
    for key in [
        "OBJECT", "FILTER", "EXPOSURE", "EXPTIME", "GAIN", "CCD-TEMP", "RA", "DEC", "INSTRUME",
        "TELESCOP", "DATE-OBS", "BAYERPAT",
    ] {
        if let Some(value) = fits.header(key) {
            println!("  {key:<10} {value:?}");
        }
    }
    let started = std::time::Instant::now();
    let stats = fits.statistics();
    println!(
        "  stats: median {} mad {:.0} mean {:.0} range {}..{} ({:.0}ms)",
        stats.median,
        stats.mad,
        stats.mean,
        stats.min,
        stats.max,
        started.elapsed().as_secs_f64() * 1000.0
    );
    if let Some(out) = stretch {
        let started = std::time::Instant::now();
        let data = fits.stretch_to_u8(&seiza_fits::StretchParams::default());
        let elapsed = started.elapsed();
        image::GrayImage::from_raw(fits.width as u32, fits.height as u32, data)
            .ok_or_else(|| anyhow::anyhow!("dimension mismatch"))?
            .save(out)?;
        println!(
            "  stretched preview written to {} ({:.0}ms stretch)",
            out.display(),
            elapsed.as_secs_f64() * 1000.0
        );
    }
    Ok(())
}

fn detect(
    path: &std::path::Path,
    sigma: f32,
    max_stars: usize,
    annotate: Option<&std::path::Path>,
) -> Result<()> {
    let img = load_image(path)?;
    let config = DetectConfig {
        sigma,
        max_stars,
        ..Default::default()
    };
    let stars = detect_stars(&img, &config);

    println!("{} stars detected in {}", stars.len(), path.display());
    println!(
        "{:>10} {:>10} {:>12} {:>8} {:>6}",
        "x", "y", "flux", "peak", "area"
    );
    for star in &stars {
        println!(
            "{:>10.2} {:>10.2} {:>12.1} {:>8.3} {:>6}",
            star.x, star.y, star.flux, star.peak, star.area
        );
    }

    if let Some(out) = annotate {
        let mut canvas = img.to_rgb8();
        for star in &stars {
            let radius = (star.area as f32).sqrt().max(6.0) as i32 + 4;
            imageproc::drawing::draw_hollow_circle_mut(
                &mut canvas,
                (star.x.round() as i32, star.y.round() as i32),
                radius,
                image::Rgb([64, 255, 64]),
            );
        }
        canvas
            .save(out)
            .with_context(|| format!("failed to write {}", out.display()))?;
        println!("annotated image written to {}", out.display());
    }

    Ok(())
}

fn cone(data: &std::path::Path, ra: f64, dec: f64, radius: f64, limit: usize) -> Result<()> {
    let catalog =
        TileCatalog::open(data).with_context(|| format!("failed to open {}", data.display()))?;
    println!(
        "{} stars in catalog, epoch {}",
        catalog.star_count(),
        catalog.epoch()
    );
    let stars = catalog.cone_search(ra, dec, radius, limit);
    println!("{} stars within {radius}° of ({ra}, {dec}):", stars.len());
    println!("{:>12} {:>12} {:>7}", "ra", "dec", "mag");
    for star in stars {
        println!("{:>12.6} {:>12.6} {:>7.3}", star.ra, star.dec, star.mag);
    }
    Ok(())
}

fn parse_sky_coordinate(value: &str) -> std::result::Result<SkyCoordinate, String> {
    let (ra, dec) = value
        .split_once(',')
        .ok_or_else(|| "expected RA,DEC in decimal degrees".to_string())?;
    let ra = ra
        .trim()
        .parse::<f64>()
        .map_err(|_| format!("invalid RA in {value:?}"))?;
    let dec = dec
        .trim()
        .parse::<f64>()
        .map_err(|_| format!("invalid Dec in {value:?}"))?;
    Ok(SkyCoordinate { ra, dec })
}

fn parse_object_kind(value: &str) -> std::result::Result<ObjectKind, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "galaxy" => Ok(ObjectKind::Galaxy),
        "open-cluster" => Ok(ObjectKind::OpenCluster),
        "globular-cluster" => Ok(ObjectKind::GlobularCluster),
        "nebula" => Ok(ObjectKind::Nebula),
        "planetary-nebula" => Ok(ObjectKind::PlanetaryNebula),
        "hii" | "hii-region" => Ok(ObjectKind::HiiRegion),
        "snr" | "supernova-remnant" => Ok(ObjectKind::SupernovaRemnant),
        "dark-nebula" => Ok(ObjectKind::DarkNebula),
        "cluster-nebula" => Ok(ObjectKind::ClusterWithNebula),
        "star" => Ok(ObjectKind::Star),
        "double-star" => Ok(ObjectKind::DoubleStar),
        "association" => Ok(ObjectKind::Association),
        "other" => Ok(ObjectKind::Other),
        "transient" => Ok(ObjectKind::Transient),
        _ => Err(format!(
            "unknown kind {value:?}; expected galaxy, open-cluster, globular-cluster, \
             nebula, planetary-nebula, hii-region, supernova-remnant, dark-nebula, \
             cluster-nebula, star, double-star, association, other, or transient"
        )),
    }
}

fn catalog_objects(args: CatalogObjectsArgs) -> Result<()> {
    if args.max_mag.is_some_and(|value| !value.is_finite()) {
        anyhow::bail!("--max-mag must be finite");
    }
    if args
        .min_size
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        anyhow::bail!("--min-size must be finite and non-negative");
    }

    let cone_requested = args.ra.is_some() || args.dec.is_some() || args.radius.is_some();
    let region = if !args.corner.is_empty() {
        if cone_requested {
            anyhow::bail!("--corner conflicts with --ra, --dec, and --radius");
        }
        SkyRegion::Polygon {
            vertices: args
                .corner
                .iter()
                .map(|corner| (corner.ra, corner.dec))
                .collect(),
        }
    } else {
        let (Some(ra), Some(dec), Some(radius_deg)) = (args.ra, args.dec, args.radius) else {
            anyhow::bail!(
                "specify either --ra/--dec/--radius or at least three ordered --corner RA,DEC values"
            );
        };
        SkyRegion::Cone {
            center: (ra, dec),
            radius_deg,
        }
    };

    let catalog = ObjectCatalog::open(&args.data)
        .with_context(|| format!("failed to open {}", args.data.display()))?;
    let query = ObjectQuery {
        kinds: args.kind,
        max_mag: args.max_mag,
        min_major_arcmin: args.min_size,
        common_name_only: args.common_name_only,
        include_extent_overlaps: !args.center_only,
        limit: (args.limit > 0).then_some(args.limit),
        sort: args.sort.into(),
    };
    let hits = catalog
        .query_region(&region, &query)
        .map_err(|error| anyhow::anyhow!(error))?;

    match args.format {
        CatalogOutputFormat::Table => {
            println!(
                "{} of {} catalog objects matched:",
                hits.len(),
                catalog.len()
            );
            println!(
                "{:>6} {:<7} {:<20} {:<30} {:>10} {:>10} {:>7} {:>8}",
                "score", "match", "kind", "object", "ra", "dec", "mag", "size'"
            );
            for hit in &hits {
                let object = hit.object;
                let name = if object.common_name.is_empty() {
                    object.name.clone()
                } else {
                    format!("{} ({})", object.name, object.common_name)
                };
                let mag = object
                    .mag
                    .map(|value| format!("{value:.2}"))
                    .unwrap_or_else(|| "-".to_string());
                let size = object
                    .major_arcmin
                    .map(|value| format!("{value:.1}"))
                    .unwrap_or_else(|| "-".to_string());
                println!(
                    "{:>5.1}% {:<7} {:<20} {:<30} {:>10.5} {:>10.5} {:>7} {:>8}",
                    hit.predicted_prominence * 100.0,
                    if hit.extent_only { "extent" } else { "center" },
                    object.kind.as_str(),
                    name,
                    object.ra,
                    object.dec,
                    mag,
                    size
                );
            }
        }
        CatalogOutputFormat::Json => {
            let region_json = match &region {
                SkyRegion::Cone { center, radius_deg } => serde_json::json!({
                    "type": "cone",
                    "center": { "ra_deg": center.0, "dec_deg": center.1 },
                    "radius_deg": radius_deg,
                }),
                SkyRegion::Polygon { vertices } => serde_json::json!({
                    "type": "polygon",
                    "vertices": vertices.iter().map(|&(ra, dec)| {
                        serde_json::json!({ "ra_deg": ra, "dec_deg": dec })
                    }).collect::<Vec<_>>(),
                }),
            };
            let objects = hits
                .iter()
                .map(|hit| {
                    let object = hit.object;
                    serde_json::json!({
                        "kind": object.kind.as_str(),
                        "name": object.name,
                        "common_name": object.common_name,
                        "ra_deg": object.ra,
                        "dec_deg": object.dec,
                        "mag": object.mag,
                        "major_arcmin": object.major_arcmin,
                        "minor_arcmin": object.minor_arcmin,
                        "position_angle_deg": object.position_angle_deg,
                        "center_inside": hit.center_inside,
                        "extent_only": hit.extent_only,
                        "distance_from_center_deg": hit.distance_from_center_deg,
                        "predicted_prominence": hit.predicted_prominence,
                    })
                })
                .collect::<Vec<_>>();
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "catalog": args.data.display().to_string(),
                    "catalog_objects": catalog.len(),
                    "region": region_json,
                    "returned": objects.len(),
                    "objects": objects,
                }))?
            );
        }
        CatalogOutputFormat::Csv => {
            println!(
                "kind,name,common_name,ra_deg,dec_deg,mag,major_arcmin,minor_arcmin,position_angle_deg,match,distance_from_center_deg,predicted_prominence"
            );
            for hit in &hits {
                let object = hit.object;
                println!(
                    "{},{},{},{:.8},{:.8},{},{},{},{},{},{:.8},{:.8}",
                    object.kind.as_str(),
                    csv_field(&object.name),
                    csv_field(&object.common_name),
                    object.ra,
                    object.dec,
                    csv_optional(object.mag),
                    csv_optional(object.major_arcmin),
                    csv_optional(object.minor_arcmin),
                    csv_optional(object.position_angle_deg),
                    if hit.extent_only { "extent" } else { "center" },
                    hit.distance_from_center_deg,
                    hit.predicted_prominence,
                );
            }
        }
    }
    Ok(())
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn csv_optional(value: Option<f32>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

#[allow(clippy::too_many_arguments)]
/// Acquisition time as a JD: an explicit ISO 8601 argument wins, else the
/// FITS DATE-OBS header. `None` when neither is available.
fn resolve_acquisition_jd(image: &std::path::Path, time: Option<&str>) -> Result<Option<f64>> {
    let text = match time {
        Some(text) => Some(text.to_string()),
        None => {
            let is_fits = image
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("fits") || e.eq_ignore_ascii_case("fit"));
            if is_fits {
                seiza_fits::read_header(image).ok().and_then(|headers| {
                    headers
                        .iter()
                        .find(|(k, _)| k == "DATE-OBS")
                        .and_then(|(_, v)| v.as_str().map(str::to_string))
                })
            } else {
                None
            }
        }
    };
    let Some(text) = text else { return Ok(None) };
    parse_iso_jd(&text)
        .map(Some)
        .ok_or_else(|| anyhow::anyhow!("cannot parse acquisition time {text:?} as ISO 8601"))
}

/// "2025-10-12T08:30:00(.frac)(Z)" to a Julian date.
fn parse_iso_jd(text: &str) -> Option<f64> {
    let text = text.trim().trim_end_matches('Z');
    let (date, clock) = match text.split_once('T') {
        Some((d, t)) => (d, t),
        None => (text, "0:0:0"),
    };
    let mut date_parts = date.split('-');
    let year: i32 = date_parts.next()?.parse().ok()?;
    let month: u32 = date_parts.next()?.parse().ok()?;
    let day: u32 = date_parts.next()?.parse().ok()?;
    let mut clock_parts = clock.split(':');
    let hour: f64 = clock_parts.next()?.parse().ok()?;
    let minute: f64 = clock_parts.next().unwrap_or("0").parse().ok()?;
    let second: f64 = clock_parts.next().unwrap_or("0").parse().ok()?;
    let day_fraction = day as f64 + (hour + minute / 60.0 + second / 3600.0) / 24.0;
    Some(seiza::minor_bodies::julian_date(year, month, day_fraction))
}

#[allow(clippy::too_many_arguments)]
fn solve_command(
    path: &std::path::Path,
    data: &std::path::Path,
    center: (f64, f64),
    radius: f64,
    scale: f64,
    scale_tolerance: f64,
    sigma: f32,
    ignore_border: u32,
    annotate: Option<&std::path::Path>,
    objects: Option<&std::path::Path>,
    minor_bodies: Option<&std::path::Path>,
    acquisition_jd: Option<f64>,
) -> Result<()> {
    let img = load_image(path)?;
    let dims = (img.width(), img.height());

    let config = DetectConfig {
        sigma,
        ignore_border,
        max_stars: 200,
        ..Default::default()
    };
    let stars = detect_stars(&img, &config);
    println!(
        "{} stars detected in {}x{} image",
        stars.len(),
        dims.0,
        dims.1
    );

    let catalog =
        TileCatalog::open(data).with_context(|| format!("failed to open {}", data.display()))?;
    let hint = SolveHint {
        center,
        radius_deg: radius,
        scale_arcsec_px: scale,
        scale_tolerance,
    };
    let started = std::time::Instant::now();
    let solution = solve(&stars, &catalog, &hint, dims).map_err(|e| anyhow::anyhow!("{e}"))?;
    let elapsed = started.elapsed();

    let wcs = &solution.wcs;
    let (ra, dec) = wcs.pixel_to_world(dims.0 as f64 / 2.0, dims.1 as f64 / 2.0);
    let det = wcs.cd[0][0] * wcs.cd[1][1] - wcs.cd[0][1] * wcs.cd[1][0];
    // Position angle of north, measured in the image
    let north = wcs.world_to_pixel(wcs.crval.0, wcs.crval.1 + 0.01);
    let rotation = north
        .map(|(x, y)| (x - wcs.crpix.0).atan2(-(y - wcs.crpix.1)).to_degrees())
        .unwrap_or(f64::NAN);

    println!("Solved in {:.2}s:", elapsed.as_secs_f64());
    println!(
        "  center     : {} {}  ({ra:.5}°, {dec:.5}°)",
        hms(ra),
        dms(dec)
    );
    println!("  pixel scale: {:.4}\"/px", wcs.scale_arcsec_per_px());
    println!("  rotation   : {rotation:.2}° (north angle in image)");
    println!(
        "  parity     : {}",
        if det > 0.0 { "normal" } else { "mirrored" }
    );
    println!(
        "  quality    : {} stars matched, RMS {:.3}\"",
        solution.matched_stars, solution.rms_arcsec
    );
    let footprint = wcs.footprint(dims.0, dims.1);
    println!(
        "  footprint  : {:.4},{:.4} / {:.4},{:.4} / {:.4},{:.4} / {:.4},{:.4}",
        footprint[0].0,
        footprint[0].1,
        footprint[1].0,
        footprint[1].1,
        footprint[2].0,
        footprint[2].1,
        footprint[3].0,
        footprint[3].1
    );

    let placed = match objects {
        Some(path) => {
            let object_catalog = seiza::objects::ObjectCatalog::open(path)
                .with_context(|| format!("failed to open {}", path.display()))?;
            let placed = object_catalog.objects_in_footprint(wcs, dims);
            println!("{} catalog objects in the field:", placed.len());
            for p in &placed {
                let size = match p.object.major_arcmin {
                    Some(major) => format!("{major:.1}'"),
                    None => "-".to_string(),
                };
                let common = if p.object.common_name.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", p.object.common_name)
                };
                println!(
                    "  {:<12} {:>18} {:>8} at ({:.0}, {:.0}){common}",
                    p.object.kind.as_str(),
                    p.object.name,
                    size,
                    p.x,
                    p.y
                );
            }
            placed
        }
        None => Vec::new(),
    };

    if let Some(path) = minor_bodies {
        match acquisition_jd {
            Some(jd) => {
                let catalog = seiza::minor_bodies::MinorBodyCatalog::open(path)
                    .with_context(|| format!("failed to open {}", path.display()))?;
                let moving = catalog.objects_in_footprint(wcs, dims, jd, 20.0);
                println!("{} minor bodies in the field at JD {jd:.4}:", moving.len());
                for m in &moving {
                    let kind = match m.body.kind {
                        seiza::minor_bodies::MinorBodyKind::Comet => "comet",
                        seiza::minor_bodies::MinorBodyKind::Asteroid => "asteroid",
                    };
                    println!(
                        "  {:<9} {:<32} V~{:>4.1} at ({:.0}, {:.0})  {:.3} AU",
                        kind, m.body.name, m.mag, m.x, m.y, m.delta_au
                    );
                }
            }
            None => println!(
                "minor bodies skipped: no acquisition time (pass --time or use a FITS with DATE-OBS)"
            ),
        }
    }

    if let Some(out) = annotate {
        let mut canvas = img.to_rgb8();
        for star in &stars {
            imageproc::drawing::draw_hollow_circle_mut(
                &mut canvas,
                (star.x.round() as i32, star.y.round() as i32),
                10,
                image::Rgb([64, 255, 64]),
            );
        }
        let fov = (dims.0 as f64).hypot(dims.1 as f64) / 2.0 * wcs.scale_arcsec_per_px() / 3600.0;
        for cat_star in catalog.cone_search(ra, dec, fov, 300) {
            if let Some((x, y)) = wcs.world_to_pixel(cat_star.ra, cat_star.dec)
                && x >= 0.0
                && y >= 0.0
                && x < dims.0 as f64
                && y < dims.1 as f64
            {
                imageproc::drawing::draw_hollow_circle_mut(
                    &mut canvas,
                    (x.round() as i32, y.round() as i32),
                    6,
                    image::Rgb([255, 64, 64]),
                );
            }
        }
        for p in &placed {
            draw_rotated_ellipse(
                &mut canvas,
                (p.x, p.y),
                p.semi_major_px.max(12.0),
                p.semi_minor_px.max(12.0),
                p.angle_deg,
                image::Rgb([64, 220, 255]),
            );
        }
        canvas
            .save(out)
            .with_context(|| format!("failed to write {}", out.display()))?;
        println!("annotated image written to {}", out.display());
    }
    Ok(())
}

fn draw_rotated_ellipse(
    canvas: &mut image::RgbImage,
    center: (f64, f64),
    semi_major: f64,
    semi_minor: f64,
    angle_deg: f64,
    color: image::Rgb<u8>,
) {
    let (sin_r, cos_r) = angle_deg.to_radians().sin_cos();
    let segments = 72;
    let point = |i: usize| -> (f32, f32) {
        let t = i as f64 / segments as f64 * std::f64::consts::TAU;
        let (lx, ly) = (semi_major * t.cos(), semi_minor * t.sin());
        (
            (center.0 + lx * cos_r - ly * sin_r) as f32,
            (center.1 + lx * sin_r + ly * cos_r) as f32,
        )
    };
    for i in 0..segments {
        imageproc::drawing::draw_line_segment_mut(canvas, point(i), point(i + 1), color);
    }
}

fn hms(ra: f64) -> String {
    let hours = ra.rem_euclid(360.0) / 15.0;
    let h = hours.floor();
    let m = ((hours - h) * 60.0).floor();
    let sec = (hours - h - m / 60.0) * 3600.0;
    format!("{:02}h {:02}m {:05.2}s", h as u32, m as u32, sec)
}

fn dms(dec: f64) -> String {
    let sign = if dec < 0.0 { '-' } else { '+' };
    let a = dec.abs();
    let d = a.floor();
    let m = ((a - d) * 60.0).floor();
    let sec = (a - d - m / 60.0) * 3600.0;
    format!("{sign}{:02}° {:02}′ {:04.1}″", d as u32, m as u32, sec)
}

fn solve_blind_command(
    path: &std::path::Path,
    data: &std::path::Path,
    min_scale: f64,
    max_scale: f64,
    sigma: f32,
    ignore_border: u32,
) -> Result<()> {
    use seiza::blind::{BlindIndex, BlindParams, solve_blind};

    let img = load_image(path)?;
    let dims = (img.width(), img.height());
    let config = DetectConfig {
        sigma,
        ignore_border,
        max_stars: 600,
        ..Default::default()
    };
    let stars = detect_stars(&img, &config);
    println!(
        "{} stars detected in {}x{} image",
        stars.len(),
        dims.0,
        dims.1
    );

    let catalog =
        TileCatalog::open(data).with_context(|| format!("failed to open {}", data.display()))?;
    let params = BlindParams {
        min_scale_arcsec_px: min_scale,
        max_scale_arcsec_px: max_scale,
        ..Default::default()
    };
    let started = std::time::Instant::now();
    let index = BlindIndex::build(&catalog, &params);
    println!(
        "pattern index: {} patterns in {:.2}s",
        index.pattern_count(),
        started.elapsed().as_secs_f64()
    );

    let started = std::time::Instant::now();
    let solution =
        solve_blind(&stars, &catalog, &index, &params, dims).map_err(|e| anyhow::anyhow!("{e}"))?;
    let wcs = &solution.wcs;
    let (ra, dec) = wcs.pixel_to_world(dims.0 as f64 / 2.0, dims.1 as f64 / 2.0);
    println!("Blind-solved in {:.2}s:", started.elapsed().as_secs_f64());
    println!(
        "  center     : {} {}  ({ra:.5}°, {dec:.5}°)",
        hms(ra),
        dms(dec)
    );
    println!("  pixel scale: {:.4}\"/px", wcs.scale_arcsec_per_px());
    println!(
        "  quality    : {} stars matched, RMS {:.3}\"",
        solution.matched_stars, solution.rms_arcsec
    );
    Ok(())
}
