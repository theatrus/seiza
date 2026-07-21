use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use seiza::blind::BlindIndex;
use seiza::catalog::{StarCatalog, TileCatalog};
use seiza::data_paths;
use seiza::minor_bodies::MinorBodyCatalog;
use seiza::objects::{ObjectCatalog, ObjectKind, ObjectQuery, ObjectSort, SkyRegion};
use seiza::solve::{SolveHint, solve};
use seiza::star_ids::{StarIdentifierCatalog, StarLookupMatch};
use seiza::{DetectBackend, DetectConfig, DetectedStar};
use seiza_satellites::{
    CacheState, CelesTrakSource, ExposureProvenance, ObserverLocation, OrbitalCatalogSource,
    SatelliteCatalog, SatelliteTrack, SingleExposure, TrackOptions, UtcTimestamp,
};
use std::path::PathBuf;

mod astap;
mod background;
mod build_data;
mod color;
mod deconvolution;
mod master;
mod preview;
mod provenance;
mod setup;
mod solve_field;
mod stack;
mod stretch_command;
mod worker;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AstronomyImageFormat {
    Fits,
    Xisf,
}

fn astronomy_image_format(path: &std::path::Path) -> Option<AstronomyImageFormat> {
    if seiza_xisf::is_xisf_path(path) {
        return Some(AstronomyImageFormat::Xisf);
    }
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("fits")
                || extension.eq_ignore_ascii_case("fit")
                || extension.eq_ignore_ascii_case("fts")
        })
        .then_some(AstronomyImageFormat::Fits)
}

fn is_fits_path(path: &std::path::Path) -> bool {
    astronomy_image_format(path) == Some(AstronomyImageFormat::Fits)
}

fn is_astronomy_image_path(path: &std::path::Path) -> bool {
    astronomy_image_format(path).is_some()
}

fn open_astronomy_image(path: &std::path::Path) -> Result<seiza_fits::FitsImage> {
    match astronomy_image_format(path) {
        Some(AstronomyImageFormat::Fits) => seiza_fits::FitsImage::open(path)
            .map_err(|error| anyhow::anyhow!("{}: {error}", path.display())),
        Some(AstronomyImageFormat::Xisf) => {
            seiza_xisf::open(path).map_err(|error| anyhow::anyhow!("{}: {error}", path.display()))
        }
        None => anyhow::bail!("{} is not a FITS or XISF path", path.display()),
    }
}

fn read_astronomy_headers(
    path: &std::path::Path,
) -> Result<Vec<(String, seiza_fits::HeaderValue)>> {
    match astronomy_image_format(path) {
        Some(AstronomyImageFormat::Fits) => seiza_fits::read_header(path)
            .map_err(|error| anyhow::anyhow!("{}: {error}", path.display())),
        Some(AstronomyImageFormat::Xisf) => seiza_xisf::read_header(path)
            .map_err(|error| anyhow::anyhow!("{}: {error}", path.display())),
        None => anyhow::bail!("{} is not a FITS or XISF path", path.display()),
    }
}

pub(crate) enum LoadedImage {
    Dynamic(image::DynamicImage),
    LinearF32 {
        luma: Vec<f32>,
        width: u32,
        height: u32,
    },
}

impl LoadedImage {
    pub(crate) fn dimensions(&self) -> (u32, u32) {
        match self {
            Self::Dynamic(image) => (image.width(), image.height()),
            Self::LinearF32 { width, height, .. } => (*width, *height),
        }
    }

    pub(crate) fn detect_stars(&self, config: &DetectConfig) -> Vec<DetectedStar> {
        match self {
            Self::Dynamic(image) => seiza::detect_stars(image, config),
            Self::LinearF32 {
                luma,
                width,
                height,
            } => seiza::detect_stars_luma_f32(luma, *width, *height, config),
        }
    }

    fn is_converted_8bit_color(&self) -> bool {
        matches!(
            self,
            Self::Dynamic(
                image::DynamicImage::ImageLumaA8(_)
                    | image::DynamicImage::ImageRgb8(_)
                    | image::DynamicImage::ImageRgba8(_)
            )
        )
    }

    fn to_rgb8(&self) -> image::RgbImage {
        match self {
            Self::Dynamic(image) => image.to_rgb8(),
            Self::LinearF32 {
                luma,
                width,
                height,
            } => {
                let mut rgb = Vec::with_capacity(luma.len() * 3);
                for &value in luma {
                    let value = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
                    rgb.extend_from_slice(&[value; 3]);
                }
                image::RgbImage::from_raw(*width, *height, rgb)
                    .expect("linear luma dimensions were validated when loaded")
            }
        }
    }

    fn to_luma8(&self) -> image::GrayImage {
        match self {
            Self::Dynamic(image) => image.to_luma8(),
            Self::LinearF32 {
                luma,
                width,
                height,
            } => {
                let luma = luma
                    .iter()
                    .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
                    .collect();
                image::GrayImage::from_raw(*width, *height, luma)
                    .expect("linear luma dimensions were validated when loaded")
            }
        }
    }
}

/// Open an image in the representation selected for detection. FITS and XISF
/// use an MTF-compressed u8 buffer for Auto/U8 and native-precision linear f32
/// for F32, which also lets a compatibility retry genuinely reload the source.
pub(crate) fn load_image(path: &std::path::Path, backend: DetectBackend) -> Result<LoadedImage> {
    if is_astronomy_image_path(path) {
        let image = open_astronomy_image(path)?;
        let width = u32::try_from(image.width)
            .map_err(|_| anyhow::anyhow!("image width exceeds supported dimensions"))?;
        let height = u32::try_from(image.height)
            .map_err(|_| anyhow::anyhow!("image height exceeds supported dimensions"))?;
        if backend == DetectBackend::F32 {
            return Ok(LoadedImage::LinearF32 {
                luma: image.to_luma_f32(),
                width,
                height,
            });
        }
        let stretched = image.stretch_to_u8(&seiza_fits::StretchParams::default());
        let buffer = image::GrayImage::from_raw(width, height, stretched)
            .ok_or_else(|| anyhow::anyhow!("image dimensions mismatch"))?;
        return Ok(LoadedImage::Dynamic(image::DynamicImage::ImageLuma8(
            buffer,
        )));
    }
    image::open(path)
        .map(LoadedImage::Dynamic)
        .with_context(|| format!("failed to open {}", path.display()))
}

/// Auto FITS/XISF begins with its compact MTF/u8 representation, while
/// converted 8-bit color can change sub-level ordering relative to f32 luma.
/// Either may benefit from a compatibility retry after a solve miss.
fn auto_can_retry_f32(path: &std::path::Path, image: &LoadedImage, backend: DetectBackend) -> bool {
    backend == DetectBackend::Auto
        && (is_astronomy_image_path(path) || image.is_converted_8bit_color())
}

fn redetect_f32(
    path: &std::path::Path,
    image: &LoadedImage,
    config: &DetectConfig,
) -> Result<Vec<DetectedStar>> {
    let config = DetectConfig {
        backend: DetectBackend::F32,
        ..config.clone()
    };
    if is_astronomy_image_path(path) {
        return Ok(load_image(path, DetectBackend::F32)?.detect_stars(&config));
    }
    Ok(image.detect_stars(&config))
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum DetectionFallback {
    None,
    #[default]
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DetectionPass {
    Primary,
    F32Fallback,
}

/// Common detection and retry state for every hinted or blind solve
/// invocation. The f32 representation is loaded lazily after a solve miss and
/// cached when a command tries more than one solving strategy.
pub(crate) struct SolveInvocation<'a> {
    path: &'a std::path::Path,
    image: &'a LoadedImage,
    config: DetectConfig,
    fallback: DetectionFallback,
    primary_stars: Vec<DetectedStar>,
    f32_stars: Option<Vec<DetectedStar>>,
    active_f32: bool,
}

impl<'a> SolveInvocation<'a> {
    pub(crate) fn new(
        path: &'a std::path::Path,
        image: &'a LoadedImage,
        config: DetectConfig,
        fallback: DetectionFallback,
    ) -> Self {
        let primary_stars = image.detect_stars(&config);
        Self {
            path,
            image,
            config,
            fallback,
            primary_stars,
            f32_stars: None,
            active_f32: false,
        }
    }

    pub(crate) fn stars(&self) -> &[DetectedStar] {
        if self.active_f32 {
            self.f32_stars
                .as_deref()
                .expect("active f32 solve has f32 detections")
        } else {
            &self.primary_stars
        }
    }

    pub(crate) fn solve<T>(
        &mut self,
        mut solver: impl FnMut(&[DetectedStar]) -> std::result::Result<T, seiza::Error>,
    ) -> Result<T> {
        self.solve_with_pass(|stars, _| solver(stars))
    }

    pub(crate) fn solve_with_pass<T>(
        &mut self,
        mut solver: impl FnMut(&[DetectedStar], DetectionPass) -> std::result::Result<T, seiza::Error>,
    ) -> Result<T> {
        match solver(&self.primary_stars, DetectionPass::Primary) {
            Ok(solution) => Ok(solution),
            Err(seiza::Error::Solve(_))
                if self.fallback == DetectionFallback::F32
                    && auto_can_retry_f32(self.path, self.image, self.config.backend) =>
            {
                eprintln!("u8 detection did not solve; retrying with f32 detection");
                if self.f32_stars.is_none() {
                    self.f32_stars = Some(redetect_f32(self.path, self.image, &self.config)?);
                }
                let result = solver(
                    self.f32_stars.as_deref().expect("f32 stars were populated"),
                    DetectionPass::F32Fallback,
                );
                if result.is_ok() {
                    self.active_f32 = true;
                }
                result.map_err(anyhow::Error::from)
            }
            Err(error) => Err(anyhow::Error::from(error)),
        }
    }
}

fn blind_params_for_detection_pass(
    params: &seiza::blind::BlindParams,
    can_retry_f32: bool,
    fallback_hypotheses: usize,
    pass: DetectionPass,
) -> seiza::blind::BlindParams {
    let mut attempt = params.clone();
    if can_retry_f32 && pass == DetectionPass::Primary && fallback_hypotheses > 0 {
        attempt.max_hypotheses = attempt.max_hypotheses.min(fallback_hypotheses);
    }
    attempt
}

/// RA/Dec hint from FITS-compatible headers (N.I.N.A. writes RA/DEC in degrees).
fn fits_hint(path: &std::path::Path) -> Option<(f64, f64)> {
    let image = open_astronomy_image(path).ok()?;
    let ra = image
        .header_f64("RA")
        .or_else(|| image.header_f64("OBJCTRA"))?;
    let dec = image
        .header_f64("DEC")
        .or_else(|| image.header_f64("OBJCTDEC"))?;
    Some((ra, dec))
}

#[derive(Parser)]
#[command(name = "seiza", about = "Star detection and plate solving", version)]
struct Cli {
    /// Detector numeric representation. Auto uses compact u8 for decoded
    /// 8-bit images and FITS, and f32 for other higher-precision inputs.
    #[arg(long, value_enum, default_value = "auto", global = true)]
    detection_backend: DetectionBackendArg,
    /// Retry a failed Auto/u8 solve using f32 detection. FITS is reloaded from
    /// its native-precision linear samples.
    #[arg(long, value_enum, default_value = "f32", global = true)]
    detection_fallback: DetectionFallback,
    /// Blind hypotheses to verify with Auto/u8 before an available f32 retry.
    /// Zero gives u8 the full --max-hypotheses budget.
    #[arg(long, default_value_t = 64, global = true)]
    detection_fallback_hypotheses: usize,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DetectionBackendArg {
    Auto,
    U8,
    F32,
}

impl From<DetectionBackendArg> for DetectBackend {
    fn from(value: DetectionBackendArg) -> Self {
        match value {
            DetectionBackendArg::Auto => Self::Auto,
            DetectionBackendArg::U8 => Self::U8,
            DetectionBackendArg::F32 => Self::F32,
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Guided installation of published star and object catalogs
    Setup(setup::SetupArgs),
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
        /// Star tile file or catalog directory (default: standard catalog
        /// locations)
        #[arg(long)]
        data: Option<PathBuf>,
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
        /// SIP distortion polynomial order to fit (0 = linear solution,
        /// 2-5 = fit when enough matched stars support it)
        #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u8).range(0..=5))]
        sip_order: u8,
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
        /// Offline satellite OMM JSON or TLE file. Predicts tracks only for a
        /// single exposure and does not claim the trail was detected
        #[arg(long, conflicts_with = "satellites_celestrak")]
        satellites: Option<PathBuf>,
        /// Fetch and cache CelesTrak's current active-satellite OMM set
        #[arg(long, conflicts_with = "satellites")]
        satellites_celestrak: bool,
        /// CelesTrak cache directory (default: platform Seiza cache)
        #[arg(long, requires = "satellites_celestrak")]
        satellite_cache: Option<PathBuf>,
        /// UTC start of this single exposure (default: FITS DATE-BEG or
        /// DATE-OBS). Also supplies the minor-body acquisition time
        #[arg(long)]
        time: Option<String>,
        /// Shutter-open duration for this single exposure in seconds
        /// (default: FITS DATE-END or XPOSURE/EXPTIME/EXPOSURE)
        #[arg(long, value_parser = clap::value_parser!(f64))]
        exposure_seconds: Option<f64>,
        /// Observer geodetic latitude, degrees north
        #[arg(long, allow_negative_numbers = true, requires = "observer_lon")]
        observer_lat: Option<f64>,
        /// Observer geodetic longitude, degrees east
        #[arg(long, allow_negative_numbers = true, requires = "observer_lat")]
        observer_lon: Option<f64>,
        /// Observer height above the reference ellipsoid, meters
        #[arg(long, allow_negative_numbers = true)]
        observer_alt_m: Option<f64>,
        /// Fine satellite track sampling interval in seconds
        #[arg(long, default_value_t = 1.0, value_parser = clap::value_parser!(f64))]
        satellite_sample_seconds: f64,
        /// Maximum absolute age of orbital elements at exposure midpoint,
        /// hours. Prevents current elements being applied to old images
        #[arg(long, default_value_t = 168.0, value_parser = clap::value_parser!(f64))]
        satellite_max_element_age_hours: f64,
    },
    /// Download ready-to-use catalogs (recommended) or advanced source data
    #[command(
        after_help = "RECOMMENDED:\n  seiza download-data prebuilt --output <directory>\n      Download the complete ready-to-use, SHA-256-verified catalog bundle.\n\n  seiza setup\n      Choose a smaller use-case-based preset interactively.\n\nRun `seiza download-data prebuilt --help` to select individual ready-to-use files.\nThe other subcommands fetch upstream source data for custom catalog builds and are not needed to use Seiza."
    )]
    DownloadData {
        #[command(subcommand)]
        source: DownloadSource,
    },
    /// Build catalog data bundles from primary sources
    BuildData {
        #[command(subcommand)]
        source: BuildDataSource,
    },
    /// Build a reusable, memory-mapped blind pattern index
    BuildBlindIndex {
        /// Star tile file used to build the index
        #[arg(long)]
        data: PathBuf,
        /// Output blind index (use blind-gaia16.idx for the hosted G<=16 set)
        #[arg(long)]
        output: PathBuf,
        /// Catalog magnitude limit used to build the blind pattern index
        #[arg(long, default_value_t = 16.0)]
        index_mag_limit: f32,
        /// Longest allowed catalog pattern edge, degrees
        #[arg(long, default_value_t = 6.0)]
        max_pattern_deg: f64,
    },
    /// Plate-solve with no position hint (wide fields, pixel-scale range)
    SolveBlind {
        /// Image file (PNG, JPEG, TIFF, FITS, XISF)
        image: PathBuf,
        /// Star tile file or catalog directory (default: standard catalog
        /// locations)
        #[arg(long)]
        data: Option<PathBuf>,
        /// Prebuilt blind pattern index file or directory (default: found
        /// in the standard locations; built in memory when absent)
        #[arg(long)]
        index: Option<PathBuf>,
        /// Minimum plausible pixel scale, arcseconds per pixel
        #[arg(long, default_value_t = 0.1)]
        min_scale: f64,
        /// Maximum plausible pixel scale, arcseconds per pixel
        #[arg(long, default_value_t = 20.0)]
        max_scale: f64,
        /// Catalog magnitude limit used to build the blind pattern index.
        /// Use 16 with the deep Gaia catalog for small, fine-scale fields.
        #[arg(long, default_value_t = 12.7)]
        index_mag_limit: f32,
        /// Field hypotheses to verify before giving up
        #[arg(long, default_value_t = 400)]
        max_hypotheses: usize,
        /// Field hypotheses to pre-rank by coarse catalog projection;
        /// bounds the cost of an image that never solves
        #[arg(long, default_value_t = 20_000)]
        max_coarse_hypotheses: usize,
        /// SIP distortion polynomial order to fit on the accepted solution
        /// (0 = linear solution, 2-5 = fit when enough matched stars
        /// support it)
        #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u8).range(0..=5))]
        sip_order: u8,
        /// Detection threshold in noise sigmas
        #[arg(long, default_value_t = 4.0)]
        sigma: f32,
        /// Ignore detections within this many pixels of the image edges
        #[arg(long, default_value_t = 0)]
        ignore_border: u32,
    },
    /// Install the astrometry.net drop-in layout for Siril: copies of this
    /// binary named solve-field and bin/bash(.exe), plus the tmp directory
    /// Windows Siril writes its launch script into
    InstallSolveField {
        /// Directory to point Siril's astrometry.net preference at
        #[arg(long)]
        dir: PathBuf,
    },
    /// Serve versioned JSON-RPC plate-solve requests over stdin/stdout
    Worker {
        /// Star tile file or catalog directory kept open for the lifetime
        /// of the worker (default: standard catalog locations)
        #[arg(long, conflicts_with = "server")]
        data: Option<PathBuf>,
        /// Optional prebuilt blind pattern index file or directory kept
        /// open by the worker
        #[arg(long, conflicts_with = "server")]
        index: Option<PathBuf>,
        /// Remote seiza-server base URL; local images are converted according to --server-upload
        #[arg(long)]
        server: Option<String>,
        /// Remote bearer token; defaults to SEIZA_SERVER_TOKEN when set
        #[arg(long, requires = "server")]
        server_token: Option<String>,
        /// Remote upload representation (PNG is usually smaller and lossless for star detection)
        #[arg(long, value_enum, requires = "server")]
        server_upload: Option<worker::ServerUploadFormat>,
        /// Maximum remote upload, queue, and solve time in seconds (default: 300)
        #[arg(
            long,
            value_name = "SECONDS",
            requires = "server",
            value_parser = clap::value_parser!(u64).range(1..)
        )]
        server_timeout: Option<u64>,
    },
    /// Inspect a FITS or XISF file: headers, statistics, optional stretched PNG
    FitsInfo {
        /// FITS or XISF file
        image: PathBuf,
        /// Write an autostretched 8-bit preview
        #[arg(long)]
        stretch: Option<PathBuf>,
    },
    /// Apply an explicit display-stretch model to a linear FITS image
    Stretch(stretch_command::StretchArgs),
    /// Estimate and remove a smooth background gradient from linear FITS
    Background(background::BackgroundArgs),
    /// Experimentally restore mild blur in a linear FITS using a measured PSF
    Deconvolve(deconvolution::DeconvolutionArgs),
    /// Register and incrementally stack linear FITS light frames
    Stack(stack::StackArgs),
    /// Register and compose mono stacks into RGB, LRGB, or narrowband color
    Color(color::ColorArgs),
    /// Build sigma-clipped bias, dark, and flat masters from raw FITS frames
    Master(master::MasterArgs),
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
    /// Resolve an object name, alias, or stable ID from objects.bin
    Object {
        #[command(flatten)]
        args: CatalogObjectArgs,
    },
    /// List named and deep-sky objects in a known sky region
    Objects {
        #[command(flatten)]
        args: CatalogObjectsArgs,
    },
    /// Resolve an exact stellar catalog designation without a network query
    Star {
        #[command(flatten)]
        args: CatalogStarArgs,
    },
    /// Exhaustively validate a catalog or index file
    Validate {
        /// Catalog, sidecar, or blind-index file; the format is auto-detected
        #[arg(long)]
        data: PathBuf,
    },
}

#[derive(Args)]
struct CatalogObjectArgs {
    /// Object catalog file or catalog directory (default: standard
    /// catalog locations)
    #[arg(long)]
    data: Option<PathBuf>,
    /// Object designation, common name, alias, or stable ID
    query: String,
    /// Return prefix completions instead of an exact lookup
    #[arg(long)]
    prefix: bool,
    /// Include every contributing source record, selection, and geometry
    #[arg(long)]
    all_sources: bool,
    /// Maximum prefix completions; ignored for exact lookup
    #[arg(long, default_value_t = 25)]
    limit: usize,
    #[arg(long, value_enum, default_value_t = CatalogOutputFormat::Table)]
    format: CatalogOutputFormat,
}

#[derive(Args)]
struct CatalogStarArgs {
    /// Star identifier sidecar file or catalog directory (default:
    /// standard catalog locations)
    #[arg(long)]
    data: Option<PathBuf>,
    /// Catalog designation or stellar name, for example HIP 32349 or RR Lyr
    query: String,
    /// Return textual-name prefix completions instead of an exact lookup
    #[arg(long)]
    prefix: bool,
    /// Maximum prefix completions; ignored for exact lookup
    #[arg(long, default_value_t = 25)]
    limit: usize,
    #[arg(long, value_enum, default_value_t = CatalogOutputFormat::Table)]
    format: CatalogOutputFormat,
}

#[derive(Args)]
struct CatalogObjectsArgs {
    /// Object catalog file or catalog directory (default: standard
    /// catalog locations)
    #[arg(long)]
    data: Option<PathBuf>,
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
    /// Ready-to-use catalog bundle (recommended; SHA-256 verified)
    Prebuilt {
        /// Directory to download into
        #[arg(long)]
        output: PathBuf,
        /// Specific files (default: the complete standard bundle). Optional
        /// extras hosted outside the bundle, such as stars-deep-gaia20.bin, are
        /// fetched only when named here.
        #[arg(long)]
        file: Vec<String>,
    },
    /// Historical satellite TLE snapshots from the Seiza mirror or IAU fallback
    SatelliteHistory {
        /// UTC epoch(s) to prewarm, normally exposure midpoints. Nearby epochs
        /// reuse one validated observing-night snapshot.
        #[arg(long, required = true, num_args = 1.., value_name = "RFC3339")]
        epoch: Vec<String>,
        /// Shared satellite cache directory (default: platform Seiza cache)
        #[arg(long)]
        cache: Option<PathBuf>,
        /// Fetch each exact epoch from the public origin, bypassing the Seiza
        /// mirror and nearby-cache reuse. Required when publishing the mirror.
        #[arg(long)]
        origin: bool,
    },
    /// Advanced source: Tycho-2 distribution files from CDS (~150 MB)
    Tycho2 {
        /// Directory to download into
        #[arg(long)]
        output: PathBuf,
    },
    /// Advanced source: bright, variable, double, and IAU star identities
    StarIdentifiers {
        /// Directory to download into
        #[arg(long)]
        output: PathBuf,
    },
    /// Advanced source: the OpenNGC deep-sky object catalog (~4 MB)
    Openngc {
        /// Directory to download into
        #[arg(long)]
        output: PathBuf,
    },
    /// Advanced source: a pinned Seiza catalog-curation GitHub snapshot
    Curation {
        /// Directory to download into; must be empty or the same pinned snapshot
        #[arg(long)]
        output: PathBuf,
        /// GitHub owner/repository
        #[arg(long, default_value = "theatrus/seiza-catalog-curation")]
        repository: String,
        /// Exact Git commit (7-40 hexadecimal characters)
        #[arg(long)]
        commit: String,
    },
    /// Advanced source: OpenNGC, Sh2, Barnard, UGC, LDN, vdB,
    /// LBN, Cederblad, IAU + Bright Star Catalogue star names
    Objects {
        /// Directory to download into
        #[arg(long)]
        output: PathBuf,
    },
    /// Advanced source: Gaia DR3 via ESA TAP (resumable; can take hours)
    Gaia {
        /// Directory to download into
        #[arg(long)]
        output: PathBuf,
        /// Magnitude limit for the download
        #[arg(long, default_value_t = 15.0, allow_negative_numbers = true)]
        max_mag: f32,
        /// Sky chunks (multiples of 192; 768 = HEALPix level 3). Deeper
        /// magnitude limits need more chunks to stay under the TAP row cap
        #[arg(
            long,
            default_value_t = 768,
            value_parser = clap::value_parser!(u64).range(1..)
        )]
        chunks: u64,
    },
    /// Advanced source: Rochester active supernova/transient list
    Transients {
        /// Directory to download into
        #[arg(long)]
        output: PathBuf,
    },
    /// Advanced source: MPC comet and numbered-asteroid elements
    Mpc {
        /// Directory to download into
        #[arg(long)]
        output: PathBuf,
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
        /// Optional exact TYC/HIP lookup sidecar
        #[arg(long)]
        identifier_index: Option<PathBuf>,
        /// Optional directory from download-data star-identifiers
        #[arg(long, requires = "identifier_index")]
        identifier_sources: Option<PathBuf>,
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
        /// Pinned seiza-catalog-curation checkout; no network access is used
        #[arg(long)]
        curation_dir: Option<PathBuf>,
        /// Output object catalog file
        #[arg(long)]
        output: PathBuf,
        /// Optional deterministic provenance manifest with source-file hashes
        #[arg(long)]
        source_manifest: Option<PathBuf>,
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
    /// Bundle manifest and optional upload-ready artifacts for hosting
    Manifest {
        /// Directory containing new or replacement .bin and .idx data files
        #[arg(long)]
        dir: PathBuf,
        /// Existing complete v2 or v4 manifest to roll forward. Files in
        /// `dir` replace matching entries; all other entries are retained.
        #[arg(long)]
        base_manifest: Option<PathBuf>,
        /// Version string and output layout: catalog-bundle-v2-* writes the
        /// flat compatibility bundle; catalog-bundle-v4-* writes immutable
        /// content-addressed artifact keys
        #[arg(long)]
        version: String,
        /// Output manifest path
        #[arg(long)]
        output: PathBuf,
        /// Stage both uncompressed and zstd artifacts below this directory.
        /// V4 only; the resulting content-addressed tree can be uploaded as
        /// one compatibility-safe publication step.
        #[arg(long)]
        artifact_dir: Option<PathBuf>,
        /// Zstd compression level (1-22). Defaults to maximum compression
        /// when --artifact-dir is supplied.
        #[arg(
            long,
            requires = "artifact_dir",
            value_parser = clap::value_parser!(i32).range(1..=22)
        )]
        zstd_level: Option<i32>,
    },
    /// Rolling satellite mirror manifest and upload-ready immutable TLEs
    SatelliteManifest {
        /// Shared orbital cache populated by download-data satellite-history
        #[arg(long)]
        cache: PathBuf,
        /// Previously published manifest to roll forward
        #[arg(long)]
        base_manifest: Option<PathBuf>,
        /// New manifest pointer; upload this to manifest.json last
        #[arg(long)]
        output: PathBuf,
        /// Stage content-addressed artifacts for S3 sync
        #[arg(long)]
        artifact_dir: PathBuf,
    },
}

/// Falling through every standard location earns the CLI flag hint the
/// library error cannot give.
fn with_data_flag_hint<T>(result: Result<T, data_paths::DataPathError>) -> Result<T> {
    result.map_err(|error| match error {
        data_paths::DataPathError::NoDefault { kind } => anyhow::anyhow!(
            "no {kind} found; pass --data (a file or a directory), run `seiza setup`, \
             or run: seiza download-data prebuilt --output <dir> (https://downloads.seiza.fyi)"
        ),
        other => anyhow::Error::new(other),
    })
}

fn main() -> Result<()> {
    // A raw ASTAP-style command line (or a copy of the binary named
    // astap) routes to the ASTAP-compatible mode before clap sees it
    let raw: Vec<String> = std::env::args().skip(1).collect();
    // A copy of the binary named solve-field (or a solve-field-shaped
    // command line) routes to the astrometry.net-compatible mode used by
    // Siril; ASTAP-style invocations route to the ASTAP-compatible mode.
    let program = std::env::args().next().unwrap_or_default();
    if solve_field::invoked_as_bash(&program) {
        return solve_field::run_as_bash(&raw);
    }
    if solve_field::invoked_as_solve_field(&program) || solve_field::looks_like_solve_field(&raw) {
        return solve_field::run(&raw);
    }
    if astap::looks_like_astap(&raw) {
        return astap::run(&raw);
    }

    let cli = Cli::parse();
    let detection_backend = cli.detection_backend.into();
    let detection_fallback = cli.detection_fallback;
    let detection_fallback_hypotheses = cli.detection_fallback_hypotheses;
    match cli.command {
        Command::Setup(args) => setup::run(args),
        Command::InstallSolveField { dir } => solve_field::install_layout(&dir),
        Command::Detect {
            image,
            sigma,
            max_stars,
            annotate,
        } => detect(
            &image,
            sigma,
            max_stars,
            detection_backend,
            annotate.as_deref(),
        ),
        Command::Solve {
            image,
            data,
            ra,
            dec,
            radius,
            scale,
            scale_tolerance,
            sip_order,
            sigma,
            ignore_border,
            annotate,
            objects,
            minor_bodies,
            satellites,
            satellites_celestrak,
            satellite_cache,
            time,
            exposure_seconds,
            observer_lat,
            observer_lon,
            observer_alt_m,
            satellite_sample_seconds,
            satellite_max_element_age_hours,
        } => {
            let hint = match (ra, dec) {
                (Some(ra), Some(dec)) => (ra, dec),
                _ => fits_hint(&image).ok_or_else(|| {
                    anyhow::anyhow!("--ra/--dec required (no RA/DEC headers found in the image)")
                })?,
            };
            let acquisition_jd = resolve_acquisition_jd(&image, time.as_deref())?;
            let satellite_input = match (satellites, satellites_celestrak) {
                (Some(path), false) => Some(SatelliteInput::File(path)),
                (None, true) => Some(SatelliteInput::CelesTrak { satellite_cache }),
                (None, false) => None,
                (Some(_), true) => unreachable!("clap rejects conflicting satellite sources"),
            };
            let satellite_request = satellite_input
                .map(|input| {
                    if !satellite_sample_seconds.is_finite() || satellite_sample_seconds <= 0.0 {
                        anyhow::bail!(
                            "--satellite-sample-seconds must be a positive finite number"
                        );
                    }
                    if !satellite_max_element_age_hours.is_finite()
                        || satellite_max_element_age_hours <= 0.0
                    {
                        anyhow::bail!(
                            "--satellite-max-element-age-hours must be a positive finite number"
                        );
                    }
                    let exposure = resolve_single_exposure(
                        &image,
                        time.as_deref(),
                        exposure_seconds,
                        observer_lat,
                        observer_lon,
                        observer_alt_m,
                    )?;
                    Ok::<_, anyhow::Error>(SatelliteSolveRequest {
                        input,
                        exposure,
                        sample_interval_seconds: satellite_sample_seconds,
                        maximum_element_age_seconds: satellite_max_element_age_hours * 3600.0,
                    })
                })
                .transpose()?;
            let data = with_data_flag_hint(data_paths::star_data(data.as_deref()))?;
            let objects = objects
                .map(|path| data_paths::objects(Some(&path)))
                .transpose()?;
            let minor_bodies = minor_bodies
                .map(|path| data_paths::minor_bodies(Some(&path)))
                .transpose()?;
            solve_command(
                &image,
                &data,
                hint,
                radius,
                scale,
                scale_tolerance,
                sip_order,
                sigma,
                ignore_border,
                detection_backend,
                detection_fallback,
                annotate.as_deref(),
                objects.as_deref(),
                minor_bodies.as_deref(),
                acquisition_jd,
                satellite_request,
            )
        }
        Command::DownloadData { source } => run_download_command(source),
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
                identifier_index,
                identifier_sources,
                epoch,
                max_mag,
            } => build_data::build_tycho2(
                &input,
                &output,
                identifier_index.as_deref(),
                identifier_sources.as_deref(),
                epoch,
                max_mag,
            ),
            BuildDataSource::Objects {
                input,
                curation_dir,
                output,
                source_manifest,
            } => build_data::build_objects(
                &input,
                &output,
                source_manifest.as_deref(),
                curation_dir.as_deref(),
            ),
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
                base_manifest,
                version,
                output,
                artifact_dir,
                zstd_level,
            } => build_data::build_manifest(
                &dir,
                base_manifest.as_deref(),
                &version,
                &output,
                artifact_dir.as_deref(),
                zstd_level.unwrap_or(22),
            ),
            BuildDataSource::SatelliteManifest {
                cache,
                base_manifest,
                output,
                artifact_dir,
            } => build_data::build_satellite_manifest(
                &cache,
                base_manifest.as_deref(),
                &output,
                &artifact_dir,
            ),
        },
        Command::BuildBlindIndex {
            data,
            output,
            index_mag_limit,
            max_pattern_deg,
        } => build_blind_index_command(&data, &output, index_mag_limit, max_pattern_deg),
        Command::SolveBlind {
            image,
            data,
            index,
            min_scale,
            max_scale,
            index_mag_limit,
            max_hypotheses,
            max_coarse_hypotheses,
            sip_order,
            sigma,
            ignore_border,
        } => {
            let data = with_data_flag_hint(data_paths::star_data(data.as_deref()))?;
            let index = data_paths::blind_index(index.as_deref())?;
            solve_blind_command(
                &image,
                &data,
                SolveBlindOptions {
                    index_path: index.as_deref(),
                    min_scale,
                    max_scale,
                    index_mag_limit,
                    max_hypotheses,
                    max_coarse_hypotheses,
                    sip_order,
                    sigma,
                    ignore_border,
                    detection_backend,
                    detection_fallback,
                    detection_fallback_hypotheses,
                },
            )
        }
        Command::Worker {
            data,
            index,
            server,
            server_token,
            server_upload,
            server_timeout,
        } => {
            let server_token = server_token.or_else(|| std::env::var("SEIZA_SERVER_TOKEN").ok());
            let data = match &server {
                Some(_) => None,
                None => Some(with_data_flag_hint(data_paths::star_data(data.as_deref()))?),
            };
            let index = index
                .map(|path| data_paths::blind_index(Some(&path)))
                .transpose()?
                .flatten();
            worker::run(worker::WorkerOptions {
                data_path: data.as_deref(),
                index_path: index.as_deref(),
                server: server.as_deref(),
                server_token: server_token.as_deref(),
                server_upload: server_upload.unwrap_or(worker::ServerUploadFormat::Png),
                server_timeout: std::time::Duration::from_secs(server_timeout.unwrap_or(300)),
                detection_backend,
                detection_fallback,
                detection_fallback_hypotheses,
            })
        }
        Command::FitsInfo { image, stretch } => fits_info(&image, stretch.as_deref()),
        Command::Stretch(options) => stretch_command::run(options),
        Command::Background(options) => background::run(options),
        Command::Deconvolve(options) => deconvolution::run(options),
        Command::Stack(options) => stack::run(options),
        Command::Color(options) => color::run(options),
        Command::Master(options) => master::run(options),
        Command::Cone {
            data,
            ra,
            dec,
            radius,
            limit,
        } => cone(&data, ra, dec, radius, limit),
        Command::Catalog { query } => match query {
            CatalogCommand::Object { args } => catalog_object(args),
            CatalogCommand::Objects { args } => catalog_objects(args),
            CatalogCommand::Star { args } => catalog_star(args),
            CatalogCommand::Validate { data } => catalog_validate(&data),
        },
    }
}

fn run_download_command(source: DownloadSource) -> Result<()> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to start the download runtime")?
        .block_on(download_command(source))
}

pub(crate) fn run_prebuilt_download(output: PathBuf, file: Vec<String>) -> Result<()> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to start the download runtime")?
        .block_on(download_prebuilt(output, file))
}

async fn download_prebuilt(output: PathBuf, file: Vec<String>) -> Result<()> {
    let selection = seiza_download::CatalogSet::from_names(file)?;
    let manager = seiza_download::CatalogManager::builder()
        .policy(seiza_download::CachePolicy::ForceRefresh)
        .build()?;
    let bundle = manager
        .ensure_with(&selection, report_download_event)
        .await?;
    let paths = bundle
        .materialize_with(&output, report_download_event)
        .await?;
    println!(
        "{} dataset(s) from {} ready in {}",
        paths.len(),
        bundle.version,
        output.display()
    );
    Ok(())
}

async fn download_command(source: DownloadSource) -> Result<()> {
    match source {
        DownloadSource::Prebuilt { output, file } => {
            download_prebuilt(output, file).await?;
        }
        DownloadSource::SatelliteHistory {
            epoch,
            cache,
            origin,
        } => {
            download_satellite_history(epoch, cache, origin).await?;
        }
        source => download_source(source).await?,
    }
    Ok(())
}

async fn download_satellite_history(
    epoch: Vec<String>,
    cache: Option<PathBuf>,
    origin: bool,
) -> Result<()> {
    let mut epochs = epoch
        .into_iter()
        .map(|value| {
            UtcTimestamp::parse(&value)
                .with_context(|| format!("invalid satellite history epoch {value:?}"))
        })
        .collect::<Result<Vec<_>>>()?;
    epochs.sort_by(|left, right| left.unix_seconds().total_cmp(&right.unix_seconds()));
    epochs.dedup_by(|left, right| left.unix_seconds() == right.unix_seconds());

    let source = match cache {
        Some(cache) => OrbitalCatalogSource::new(cache)?,
        None => OrbitalCatalogSource::platform_default()?,
    };
    for epoch in epochs {
        let load = if origin {
            source.fetch_historical_origin(epoch).await?
        } else {
            source.prewarm_historical(epoch).await?
        };
        let state = match load.state {
            CacheState::Downloaded => "downloaded",
            CacheState::Cached => "cached",
            CacheState::Fresh => "fresh cache",
            CacheState::StaleFallback => "stale cache fallback",
        };
        println!(
            "satellite history for {}: {} records ({}, {state}, {})",
            load.snapshot
                .query_time
                .expect("historical prewarming always records its query epoch")
                .to_rfc3339(),
            load.catalog.len(),
            match load.snapshot.provider {
                seiza_satellites::OrbitalCatalogProvider::CelesTrakActive => "CelesTrak active",
                seiza_satellites::OrbitalCatalogProvider::SeizaMirror => "Seiza mirror",
                seiza_satellites::OrbitalCatalogProvider::IauSatChecker => "IAU SatChecker",
            },
            load.cache_path.display()
        );
    }
    Ok(())
}

async fn download_source(source: DownloadSource) -> Result<()> {
    let downloader = seiza_sources::SourceDownloader::with_reporter(report_source_event)?;
    match source {
        DownloadSource::Tycho2 { output } => downloader.download_tycho2(output).await,
        DownloadSource::StarIdentifiers { output } => {
            downloader.download_star_identifiers(output).await
        }
        DownloadSource::Openngc { output } => downloader.download_openngc(output).await,
        DownloadSource::Curation {
            output,
            repository,
            commit,
        } => {
            downloader
                .download_curation(&repository, &commit, output)
                .await
        }
        DownloadSource::Objects { output } => downloader.download_objects(output).await,
        DownloadSource::Gaia {
            output,
            max_mag,
            chunks,
        } => downloader.download_gaia(output, max_mag, chunks).await,
        DownloadSource::Transients { output } => downloader.download_transients(output).await,
        DownloadSource::Mpc { output } => downloader.download_mpc(output).await,
        DownloadSource::Prebuilt { .. } | DownloadSource::SatelliteHistory { .. } => {
            unreachable!("handled by download_command")
        }
    }?;
    Ok(())
}

fn report_download_event(event: seiza_download::DownloadEvent) {
    use seiza_download::DownloadEvent;
    use std::io::{IsTerminal, Write};
    use std::sync::Mutex;
    use std::time::Instant;

    struct ActiveDownload {
        name: String,
        started: Instant,
        downloaded: u64,
        written: u64,
    }
    // Downloads are sequential; remember the active one so progress and
    // completion lines can report elapsed-time transfer speed.
    static ACTIVE: Mutex<Option<ActiveDownload>> = Mutex::new(None);
    let mib = |bytes: u64| bytes as f64 / 1_048_576.0;

    match event {
        DownloadEvent::FetchingManifest { url } => println!("  fetching {url}"),
        DownloadEvent::UsingCachedManifest { version, stale } => {
            let qualifier = if stale { "stale " } else { "" };
            println!("  using {qualifier}cached manifest {version}");
        }
        DownloadEvent::CacheHit { name, .. } => println!("  {name} already cached"),
        DownloadEvent::DownloadStarted { name, bytes } => {
            *ACTIVE.lock().unwrap() = Some(ActiveDownload {
                name: name.clone(),
                started: Instant::now(),
                downloaded: 0,
                written: 0,
            });
            println!("  downloading {name} ({:.1} MiB)", mib(bytes))
        }
        DownloadEvent::DownloadProgress {
            name,
            downloaded,
            total,
            written,
        } => {
            let mut active = ACTIVE.lock().unwrap();
            let elapsed = match active.as_mut() {
                Some(active) if active.name == name => {
                    active.downloaded = downloaded;
                    active.written = written;
                    active.started.elapsed().as_secs_f64()
                }
                _ => 0.0,
            };
            drop(active);
            if !std::io::stdout().is_terminal() {
                return;
            }
            let percent = downloaded.saturating_mul(100) / total.max(1);
            let speed = if elapsed > 0.0 {
                mib(downloaded) / elapsed
            } else {
                0.0
            };
            let unpacked = if written > downloaded {
                format!(", unpacked {:.1} MiB", mib(written))
            } else {
                String::new()
            };
            print!(
                "\r  {name}: {percent:>3}% ({:.1}/{:.1} MiB, {speed:.1} MiB/s{unpacked})   ",
                mib(downloaded),
                mib(total),
            );
            let _ = std::io::stdout().flush();
        }
        DownloadEvent::DownloadComplete { name, .. } => {
            let active = ACTIVE.lock().unwrap().take();
            match active.filter(|active| active.name == name && active.downloaded > 0) {
                Some(active) => {
                    let elapsed = active
                        .started
                        .elapsed()
                        .as_secs_f64()
                        .max(f64::MIN_POSITIVE);
                    let speed = mib(active.downloaded) / elapsed;
                    let unpacked = if active.written > active.downloaded {
                        format!(", unpacked to {:.1} MiB", mib(active.written))
                    } else {
                        String::new()
                    };
                    println!(
                        "\r  cached {name}: {:.1} MiB in {elapsed:.1}s ({speed:.1} MiB/s{unpacked})      ",
                        mib(active.downloaded),
                    );
                }
                None => println!("\r  cached {name}                         "),
            }
        }
        DownloadEvent::Verifying { name } => println!("  verifying {name}"),
        DownloadEvent::Installing { name, .. } => println!("  installing {name}"),
        DownloadEvent::InstallComplete { .. } => {}
    }
}

fn report_source_event(event: seiza_sources::SourceEvent) {
    use seiza_sources::SourceEvent;
    match event {
        SourceEvent::AlreadyPresent { path } => println!("  {} already present", path.display()),
        SourceEvent::Fetching { url, .. } => println!("  fetching {url}"),
        SourceEvent::Progress { .. } => {}
        SourceEvent::Retry {
            label,
            attempt,
            delay,
            error,
        } => eprintln!(
            "  {label} attempt {attempt} failed: {error}; retrying in {}s",
            delay.as_secs()
        ),
        SourceEvent::GaiaChunkComplete {
            chunk,
            rows,
            completed,
            total,
        } => println!("  chunk {chunk:04}: {rows} rows ({completed}/{total})"),
        SourceEvent::Ready { source, directory } => {
            println!("{source} ready in {}", directory.display())
        }
    }
}

fn catalog_object(args: CatalogObjectArgs) -> Result<()> {
    let data = with_data_flag_hint(data_paths::objects(args.data.as_deref()))?;
    let catalog =
        ObjectCatalog::open(&data).with_context(|| format!("failed to open {}", data.display()))?;
    let matches = if args.prefix {
        catalog.search_names(&args.query, args.limit)?
    } else {
        catalog.lookup_name(&args.query)?
    };
    if args.all_sources {
        return catalog_object_all_sources(&catalog, &matches, args.format);
    }

    match args.format {
        CatalogOutputFormat::Table => {
            println!(
                "{} object name match{}:",
                matches.len(),
                if matches.len() == 1 { "" } else { "es" }
            );
            println!(
                "{:<24} {:<18} {:<28} {:>10} {:>10}  id",
                "matched", "kind", "object", "ra", "dec"
            );
            for item in &matches {
                println!(
                    "{:<24} {:<18} {:<28} {:>10.5} {:>10.5}  {}",
                    item.matched_name,
                    item.object.kind.as_str(),
                    item.object.name,
                    item.object.ra,
                    item.object.dec,
                    item.object.metadata.id,
                );
            }
        }
        CatalogOutputFormat::Json => {
            let values = matches
                .iter()
                .map(|item| {
                    let object = &item.object;
                    serde_json::json!({
                        "matched_name": item.matched_name,
                        "kind": object.kind.as_str(),
                        "name": object.name,
                        "common_name": object.common_name,
                        "id": object.metadata.id,
                        "source": object.metadata.source,
                        "aliases": object.metadata.aliases,
                        "parent_ids": object.metadata.parent_ids,
                        "alternate_ids": object.metadata.alternate_ids,
                        "alternate_sources": object.metadata.alternate_sources,
                        "ra_deg": object.ra,
                        "dec_deg": object.dec,
                        "mag": object.mag,
                        "major_arcmin": object.major_arcmin,
                        "minor_arcmin": object.minor_arcmin,
                        "position_angle_deg": object.position_angle_deg,
                    })
                })
                .collect::<Vec<_>>();
            println!("{}", serde_json::to_string_pretty(&values)?);
        }
        CatalogOutputFormat::Csv => {
            println!(
                "matched_name,kind,name,common_name,ra_deg,dec_deg,mag,major_arcmin,id,source"
            );
            for item in &matches {
                let object = &item.object;
                println!(
                    "{},{},{},{},{:.8},{:.8},{},{},{},{}",
                    csv_field(&item.matched_name),
                    object.kind.as_str(),
                    csv_field(&object.name),
                    csv_field(&object.common_name),
                    object.ra,
                    object.dec,
                    csv_optional(object.mag),
                    csv_optional(object.major_arcmin),
                    csv_field(&object.metadata.id),
                    csv_field(&object.metadata.source),
                );
            }
        }
    }
    Ok(())
}

fn catalog_object_all_sources(
    catalog: &ObjectCatalog,
    matches: &[seiza::objects::ObjectNameMatch],
    format: CatalogOutputFormat,
) -> Result<()> {
    match format {
        CatalogOutputFormat::Json => {
            let values = matches
                .iter()
                .map(|item| {
                    let details = catalog.object_details(&item.object.metadata.id)?;
                    Ok(serde_json::json!({
                        "matched_name": item.matched_name,
                        "canonical": item.object,
                        "details": details,
                    }))
                })
                .collect::<Result<Vec<_>>>()?;
            println!("{}", serde_json::to_string_pretty(&values)?);
        }
        CatalogOutputFormat::Csv => {
            println!(
                "canonical_id,record_id,source,name,ra_deg,dec_deg,mag,major_arcmin,minor_arcmin,position_angle_deg"
            );
            for item in matches {
                for record in catalog.catalog_records(&item.object.metadata.id)? {
                    println!(
                        "{},{},{},{},{:.8},{:.8},{},{},{},{}",
                        csv_field(&item.object.metadata.id),
                        csv_field(&record.id),
                        csv_field(&record.source),
                        csv_field(&record.object.name),
                        record.object.ra,
                        record.object.dec,
                        csv_optional(record.object.mag),
                        csv_optional(record.object.major_arcmin),
                        csv_optional(record.object.minor_arcmin),
                        csv_optional(record.object.position_angle_deg),
                    );
                }
            }
        }
        CatalogOutputFormat::Table => {
            println!(
                "{} object name match{} (format v{}):",
                matches.len(),
                if matches.len() == 1 { "" } else { "es" },
                catalog.format_version(),
            );
            for item in matches {
                println!(
                    "\n{} [{}]  {}  {:.5}, {:+.5}",
                    item.object.name,
                    item.object.metadata.id,
                    item.object.kind.as_str(),
                    item.object.ra,
                    item.object.dec,
                );
                let Some(details) = catalog.object_details(&item.object.metadata.id)? else {
                    println!("  no source detail section");
                    continue;
                };
                println!("  selections:");
                for selection in &details.selections {
                    println!(
                        "    {:?}: source={} geometry={}{}",
                        selection.facet,
                        selection.source_record_id.as_deref().unwrap_or("-"),
                        selection.geometry_id.as_deref().unwrap_or("-"),
                        if selection.reason.is_empty() {
                            String::new()
                        } else {
                            format!(" ({})", selection.reason)
                        },
                    );
                }
                println!("  source records:");
                for record in &details.source_records {
                    println!(
                        "    {}  {}  {}  {:.5}, {:+.5}  size={}x{} pa={}",
                        record.id,
                        record.source,
                        record.object.name,
                        record.object.ra,
                        record.object.dec,
                        display_optional(record.object.major_arcmin),
                        display_optional(record.object.minor_arcmin),
                        display_optional(record.object.position_angle_deg),
                    );
                }
                println!("  geometries:");
                for geometry in &details.geometries {
                    let description = match &geometry.data {
                        seiza::objects::GeometryData::Point { .. } => "point".to_string(),
                        seiza::objects::GeometryData::Ellipse {
                            major_arcmin,
                            minor_arcmin,
                            position_angle_deg,
                            ..
                        } => format!(
                            "ellipse {}x{} arcmin pa={}",
                            major_arcmin,
                            display_optional(*minor_arcmin),
                            display_optional(*position_angle_deg),
                        ),
                        seiza::objects::GeometryData::OutlineSet { level, contours } => format!(
                            "outline-set {} contours, {} vertices, {}",
                            contours.len(),
                            contours
                                .iter()
                                .map(|contour| contour.vertices.len())
                                .sum::<usize>(),
                            level.as_deref().unwrap_or("no level"),
                        ),
                    };
                    println!(
                        "    {}  {:?}/{:?}  {}",
                        geometry.id, geometry.role, geometry.quality, description
                    );
                }
            }
        }
    }
    Ok(())
}

fn display_optional(value: Option<f32>) -> String {
    value.map_or_else(|| "-".into(), |value| format!("{value}"))
}

fn fits_info(path: &std::path::Path, stretch: Option<&std::path::Path>) -> Result<()> {
    let started = std::time::Instant::now();
    let image = open_astronomy_image(path)?;
    let load_time = started.elapsed();
    println!(
        "{}: {}x{} ({:?}-type pixels), loaded in {:.0}ms",
        path.display(),
        image.width,
        image.height,
        match image.pixels {
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
        if let Some(value) = image.header(key) {
            println!("  {key:<10} {value:?}");
        }
    }
    let started = std::time::Instant::now();
    let stats = image.statistics();
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
        let data = image.stretch_to_u8(&seiza_fits::StretchParams::default());
        let elapsed = started.elapsed();
        image::GrayImage::from_raw(image.width as u32, image.height as u32, data)
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
    detection_backend: DetectBackend,
    annotate: Option<&std::path::Path>,
) -> Result<()> {
    let img = load_image(path, detection_backend)?;
    let config = DetectConfig {
        backend: detection_backend,
        sigma,
        max_stars,
        ..Default::default()
    };
    let stars = img.detect_stars(&config);

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

    let data = with_data_flag_hint(data_paths::objects(args.data.as_deref()))?;
    let catalog =
        ObjectCatalog::open(&data).with_context(|| format!("failed to open {}", data.display()))?;
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
                let object = &hit.object;
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
                    let object = &hit.object;
                    serde_json::json!({
                        "kind": object.kind.as_str(),
                        "name": object.name,
                        "common_name": object.common_name,
                        "id": object.metadata.id,
                        "source": object.metadata.source,
                        "aliases": object.metadata.aliases,
                        "parent_ids": object.metadata.parent_ids,
                        "alternate_ids": object.metadata.alternate_ids,
                        "alternate_sources": object.metadata.alternate_sources,
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
                    "catalog": data.display().to_string(),
                    "catalog_objects": catalog.len(),
                    "region": region_json,
                    "returned": objects.len(),
                    "objects": objects,
                }))?
            );
        }
        CatalogOutputFormat::Csv => {
            println!(
                "kind,name,common_name,ra_deg,dec_deg,mag,major_arcmin,minor_arcmin,position_angle_deg,match,distance_from_center_deg,predicted_prominence,id,source,aliases,parent_ids,alternate_ids,alternate_sources"
            );
            for hit in &hits {
                let object = &hit.object;
                println!(
                    "{},{},{},{:.8},{:.8},{},{},{},{},{},{:.8},{:.8},{},{},{},{},{},{}",
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
                    csv_field(&object.metadata.id),
                    csv_field(&object.metadata.source),
                    csv_field(&object.metadata.aliases.join("|")),
                    csv_field(&object.metadata.parent_ids.join("|")),
                    csv_field(&object.metadata.alternate_ids.join("|")),
                    csv_field(&object.metadata.alternate_sources.join("|")),
                );
            }
        }
    }
    Ok(())
}

fn catalog_star(args: CatalogStarArgs) -> Result<()> {
    let data = with_data_flag_hint(data_paths::star_identifiers(args.data.as_deref()))?;
    let catalog = StarIdentifierCatalog::open(&data)
        .with_context(|| format!("failed to open {}", data.display()))?;
    let matches = if args.prefix {
        catalog
            .search_names(&args.query, args.limit)?
            .into_iter()
            .map(StarLookupMatch::Name)
            .collect::<Vec<_>>()
    } else {
        catalog.lookup_query(&args.query)?
    };
    let rows = matches.iter().map(catalog_star_row).collect::<Vec<_>>();

    match args.format {
        CatalogOutputFormat::Table => {
            println!(
                "{} {} match{} for {:?} in {} (epoch J{}):",
                rows.len(),
                if args.prefix { "prefix" } else { "exact" },
                if rows.len() == 1 { "" } else { "es" },
                args.query,
                catalog.attribution(),
                catalog.epoch()
            );
            if !rows.is_empty() {
                println!(
                    "{:<22} {:<18} {:<17} {:<10} {:>11} {:>11} {:>7}  stable ID",
                    "designation", "catalog", "kind", "detail", "ra", "dec", "mag"
                );
                for row in &rows {
                    println!(
                        "{:<22} {:<18} {:<17} {:<10} {:>11.6} {:>11.6} {:>7}  {}",
                        row.designation,
                        row.catalog,
                        row.kind,
                        row.detail,
                        row.ra,
                        row.dec,
                        row.mag
                            .map(|value| format!("{value:.3}"))
                            .unwrap_or_else(|| "-".to_string()),
                        row.stable_id,
                    );
                }
            }
        }
        CatalogOutputFormat::Json => {
            let matches = rows
                .iter()
                .map(|row| {
                    serde_json::json!({
                        "designation": row.designation,
                        "stable_id": row.stable_id,
                        "catalog": row.catalog,
                        "kind": row.kind,
                        "detail": row.detail,
                        "ra_deg": row.ra,
                        "dec_deg": row.dec,
                        "mag": row.mag,
                    })
                })
                .collect::<Vec<_>>();
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "query": args.query,
                    "mode": if args.prefix { "prefix" } else { "exact" },
                    "source": catalog.attribution(),
                    "epoch": catalog.epoch(),
                    "numeric_entries": catalog.numeric_len(),
                    "name_entries": catalog.name_len(),
                    "returned": matches.len(),
                    "matches": matches,
                }))?
            );
        }
        CatalogOutputFormat::Csv => {
            println!("designation,stable_id,catalog,kind,detail,ra_deg,dec_deg,mag,epoch,source");
            for row in &rows {
                println!(
                    "{},{},{},{},{},{:.8},{:.8},{},{},{}",
                    csv_field(&row.designation),
                    csv_field(&row.stable_id),
                    csv_field(&row.catalog),
                    csv_field(&row.kind),
                    csv_field(&row.detail),
                    row.ra,
                    row.dec,
                    row.mag
                        .map(|value| format!("{value:.3}"))
                        .unwrap_or_default(),
                    catalog.epoch(),
                    csv_field(catalog.attribution()),
                );
            }
        }
    }
    Ok(())
}

fn catalog_validate(path: &std::path::Path) -> Result<()> {
    use std::io::Read;

    let mut file =
        std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)
        .with_context(|| format!("failed to read catalog header from {}", path.display()))?;

    let summary = match &magic {
        b"SEIZASI1" => {
            let catalog = StarIdentifierCatalog::open(path)?;
            catalog.validate()?;
            format!(
                "stellar identifier sidecar: {} numeric identifiers, {} names",
                catalog.numeric_len(),
                catalog.name_len()
            )
        }
        b"SEIZAST1" | b"SEIZAST2" => {
            let catalog = TileCatalog::open(path)?;
            catalog.validate()?;
            format!("star tile catalog: {} stars", catalog.star_count())
        }
        b"SEIZABI1" => {
            let index = BlindIndex::open(path)?;
            index.validate()?;
            format!("blind-pattern index: {} patterns", index.pattern_count())
        }
        b"SEIZAOB1" | b"SEIZAOB3" | b"SEIZAOB\0" => {
            let catalog = ObjectCatalog::open(path)?;
            catalog.validate()?;
            format!(
                "object catalog v{}: {} objects",
                catalog.format_version(),
                catalog.len()
            )
        }
        b"SEIZAMB1" => {
            let catalog = MinorBodyCatalog::open(path)?;
            catalog.validate()?;
            format!("minor-body catalog: {} bodies", catalog.len())
        }
        _ => anyhow::bail!("{} is not a recognized seiza catalog", path.display()),
    };
    println!("{}: valid {summary}", path.display());
    Ok(())
}

struct CatalogStarRow {
    designation: String,
    stable_id: String,
    catalog: String,
    kind: String,
    detail: String,
    ra: f64,
    dec: f64,
    mag: Option<f32>,
}

fn catalog_star_row(value: &StarLookupMatch<'_>) -> CatalogStarRow {
    match value {
        StarLookupMatch::Identifier(star) => CatalogStarRow {
            designation: star.identifier.to_string(),
            stable_id: star.identifier.stable_id(),
            catalog: star.identifier.namespace().as_str().to_string(),
            kind: "catalog-identifier".to_string(),
            detail: String::new(),
            ra: star.ra,
            dec: star.dec,
            mag: Some(star.mag),
        },
        StarLookupMatch::Name(star) => CatalogStarRow {
            designation: star.designation.to_string(),
            stable_id: star.stable_id.to_string(),
            catalog: star.catalog.as_str().to_string(),
            kind: star.kind.as_str().to_string(),
            detail: star.detail.to_string(),
            ra: star.ra,
            dec: star.dec,
            mag: star.mag,
        },
    }
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
/// Acquisition time as a JD: an explicit ISO 8601 argument wins, else a
/// FITS-compatible DATE-OBS header. `None` when neither is available.
fn resolve_acquisition_jd(image: &std::path::Path, time: Option<&str>) -> Result<Option<f64>> {
    let text = match time {
        Some(text) => Some(text.to_string()),
        None => {
            if is_astronomy_image_path(image) {
                read_astronomy_headers(image).ok().and_then(|headers| {
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

#[derive(Debug)]
enum SatelliteInput {
    File(PathBuf),
    CelesTrak { satellite_cache: Option<PathBuf> },
}

#[derive(Debug)]
struct SatelliteSolveRequest {
    input: SatelliteInput,
    exposure: SingleExposure,
    sample_interval_seconds: f64,
    maximum_element_age_seconds: f64,
}

fn resolve_single_exposure(
    image: &std::path::Path,
    explicit_start: Option<&str>,
    explicit_duration: Option<f64>,
    explicit_latitude: Option<f64>,
    explicit_longitude: Option<f64>,
    explicit_altitude: Option<f64>,
) -> Result<SingleExposure> {
    let headers = if is_astronomy_image_path(image) {
        read_astronomy_headers(image).with_context(|| {
            format!("failed to read FITS/XISF metadata from {}", image.display())
        })?
    } else {
        Vec::new()
    };
    resolve_single_exposure_from_headers(
        &headers,
        explicit_start,
        explicit_duration,
        explicit_latitude,
        explicit_longitude,
        explicit_altitude,
    )
}

fn resolve_single_exposure_from_headers(
    headers: &[(String, seiza_fits::HeaderValue)],
    explicit_start: Option<&str>,
    explicit_duration: Option<f64>,
    explicit_latitude: Option<f64>,
    explicit_longitude: Option<f64>,
    explicit_altitude: Option<f64>,
) -> Result<SingleExposure> {
    if explicit_start.is_none() {
        if let Some(timesys) = header_text(headers, "TIMESYS")
            && !timesys.eq_ignore_ascii_case("UTC")
        {
            anyhow::bail!(
                "satellite tracks currently require UTC timestamps; FITS TIMESYS is {timesys:?}"
            );
        }
        if let Some(reference) = header_text(headers, "TREFPOS")
            && !reference.to_ascii_uppercase().starts_with("TOP")
        {
            anyhow::bail!(
                "satellite tracks require topocentric acquisition times; FITS TREFPOS is {reference:?}"
            );
        }
    }

    let observer = match (explicit_latitude, explicit_longitude) {
        (Some(latitude), Some(longitude)) => {
            ObserverLocation::geodetic(latitude, longitude, explicit_altitude.unwrap_or(0.0))?
        }
        (Some(_), None) | (None, Some(_)) => {
            anyhow::bail!("--observer-lat and --observer-lon must be supplied together")
        }
        (None, None) if explicit_altitude.is_some() => {
            anyhow::bail!("--observer-alt-m requires --observer-lat and --observer-lon")
        }
        (None, None) => {
            let itrf = (
                header_number(headers, "OBSGEO-X"),
                header_number(headers, "OBSGEO-Y"),
                header_number(headers, "OBSGEO-Z"),
            );
            let geodetic = (
                header_number(headers, "OBSGEO-B"),
                header_number(headers, "OBSGEO-L"),
                header_number(headers, "OBSGEO-H"),
            );
            let site_alias = (
                header_number(headers, "SITELAT"),
                header_number(headers, "SITELONG"),
                header_number(headers, "SITEALT").unwrap_or(0.0),
            );
            match (itrf, geodetic, site_alias) {
                ((Some(x), Some(y), Some(z)), _, _) => ObserverLocation::itrf_meters(x, y, z)?,
                (_, (Some(latitude), Some(longitude), Some(altitude)), _) => {
                    ObserverLocation::geodetic(latitude, longitude, altitude)?
                }
                (_, _, (Some(latitude), Some(longitude), altitude)) => {
                    ObserverLocation::geodetic(latitude, longitude, altitude)?
                }
                _ => anyhow::bail!(
                    "satellite tracks require an observer location; pass --observer-lat/--observer-lon or provide FITS OBSGEO-X/Y/Z, OBSGEO-B/L/H, or SITELAT/SITELONG"
                ),
            }
        }
    };

    let date_beg = header_text(headers, "DATE-BEG");
    let date_end = header_text(headers, "DATE-END");
    let date_avg = header_text(headers, "DATE-AVG");
    let date_obs = header_text(headers, "DATE-OBS");
    let header_duration = ["XPOSURE", "EXPTIME", "EXPOSURE"]
        .into_iter()
        .find_map(|key| header_number(headers, key));

    if let Some(start) = explicit_start {
        let start = UtcTimestamp::parse(start)?;
        let duration = explicit_duration.or(header_duration).or_else(|| {
            let begin = UtcTimestamp::parse(date_beg?).ok()?;
            let end = UtcTimestamp::parse(date_end?).ok()?;
            Some(end.seconds_since(begin))
        });
        let duration = duration.ok_or_else(|| {
            anyhow::anyhow!(
                "satellite tracks require one shutter-open duration; pass --exposure-seconds or provide FITS DATE-BEG/DATE-END or XPOSURE/EXPTIME"
            )
        })?;
        return Ok(SingleExposure::from_start_and_duration(
            start,
            duration,
            observer,
            ExposureProvenance::Explicit,
        )?);
    }

    if let Some(duration) = explicit_duration {
        if let Some(start) = date_beg.or(date_obs) {
            return Ok(SingleExposure::from_start_and_duration(
                UtcTimestamp::parse(start)?,
                duration,
                observer,
                ExposureProvenance::Explicit,
            )?);
        }
        if let Some(midpoint) = date_avg {
            return Ok(SingleExposure::from_midpoint_and_duration(
                UtcTimestamp::parse(midpoint)?,
                duration,
                observer,
                ExposureProvenance::FitsDateAvgAndExposure,
            )?);
        }
        if let Some(end) = date_end {
            return Ok(SingleExposure::from_end_and_duration(
                UtcTimestamp::parse(end)?,
                duration,
                observer,
                ExposureProvenance::FitsEndAndExposure,
            )?);
        }
        anyhow::bail!(
            "--exposure-seconds also requires --time or a FITS DATE-BEG/DATE-OBS, DATE-AVG, or DATE-END timestamp"
        );
    }

    if let (Some(start), Some(end)) = (date_beg, date_end) {
        return Ok(SingleExposure::new(
            UtcTimestamp::parse(start)?,
            UtcTimestamp::parse(end)?,
            observer,
            ExposureProvenance::FitsBounds,
        )?);
    }

    if let (Some(midpoint), Some(duration)) = (date_avg, header_duration) {
        return Ok(SingleExposure::from_midpoint_and_duration(
            UtcTimestamp::parse(midpoint)?,
            duration,
            observer,
            ExposureProvenance::FitsDateAvgAndExposure,
        )?);
    }

    let duration = header_duration.ok_or_else(|| {
        anyhow::anyhow!(
            "satellite tracks require one shutter-open duration; pass --exposure-seconds or provide FITS XPOSURE/EXPTIME/EXPOSURE (stack integration totals are not accepted)"
        )
    })?;
    if let Some(start) = date_obs.or(date_beg) {
        return Ok(SingleExposure::from_start_and_duration(
            UtcTimestamp::parse(start)?,
            duration,
            observer,
            ExposureProvenance::FitsDateObsAndExposure,
        )?);
    }
    if let Some(end) = date_end {
        return Ok(SingleExposure::from_end_and_duration(
            UtcTimestamp::parse(end)?,
            duration,
            observer,
            ExposureProvenance::FitsEndAndExposure,
        )?);
    }
    anyhow::bail!(
        "satellite tracks require one exposure timestamp; pass --time or provide FITS DATE-BEG/DATE-OBS, DATE-AVG, or DATE-END"
    )
}

fn header_text<'a>(headers: &'a [(String, seiza_fits::HeaderValue)], key: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(name, _)| name == key)
        .and_then(|(_, value)| value.as_str())
}

fn header_number(headers: &[(String, seiza_fits::HeaderValue)], key: &str) -> Option<f64> {
    headers
        .iter()
        .find(|(name, _)| name == key)
        .and_then(|(_, value)| value.as_f64())
}

fn load_satellite_catalog(input: SatelliteInput) -> Result<SatelliteCatalog> {
    match input {
        SatelliteInput::File(path) => SatelliteCatalog::open(&path)
            .with_context(|| format!("failed to open satellite elements from {}", path.display())),
        SatelliteInput::CelesTrak { satellite_cache } => {
            let source = match satellite_cache {
                Some(cache) => CelesTrakSource::new(cache)?,
                None => CelesTrakSource::platform_default()?,
            };
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .context("failed to initialize satellite download runtime")?;
            let load = runtime.block_on(source.load_active())?;
            let state = match load.state {
                CacheState::Fresh => "fresh cache",
                CacheState::Downloaded => "downloaded",
                CacheState::StaleFallback => "stale cache fallback",
                CacheState::Cached => "cache-only",
            };
            println!(
                "satellite elements: {} records ({state}, {})",
                load.catalog.len(),
                load.cache_path.display()
            );
            if let Some(warning) = load.warning {
                eprintln!("warning: {warning}");
            }
            Ok(load.catalog)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_command(
    path: &std::path::Path,
    data: &std::path::Path,
    center: (f64, f64),
    radius: f64,
    scale: f64,
    scale_tolerance: f64,
    sip_order: u8,
    sigma: f32,
    ignore_border: u32,
    detection_backend: DetectBackend,
    detection_fallback: DetectionFallback,
    annotate: Option<&std::path::Path>,
    objects: Option<&std::path::Path>,
    minor_bodies: Option<&std::path::Path>,
    acquisition_jd: Option<f64>,
    satellite_request: Option<SatelliteSolveRequest>,
) -> Result<()> {
    let img = load_image(path, detection_backend)?;
    let dims = img.dimensions();

    let config = DetectConfig {
        backend: detection_backend,
        sigma,
        ignore_border,
        max_stars: 200,
        ..Default::default()
    };
    let mut invocation = SolveInvocation::new(path, &img, config, detection_fallback);
    println!(
        "{} stars detected in {}x{} image",
        invocation.stars().len(),
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
        sip_order,
    };
    let started = std::time::Instant::now();
    let solution = invocation.solve(|stars| solve(stars, &catalog, &hint, dims))?;
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
    if let Some(sip) = &wcs.sip {
        println!(
            "  distortion : SIP order {} (forward and inverse)",
            sip.order
        );
    }
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
            let placed = object_catalog
                .objects_in_footprint(wcs, dims)
                .map_err(|error| anyhow::anyhow!(error))?;
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

    let moving = if let Some(path) = minor_bodies {
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
                    let rate = match m.motion_arcsec_per_hour {
                        Some(rate) => format!("{rate:.1}\"/hr"),
                        None => "-".to_string(),
                    };
                    println!(
                        "  {:<9} {:<32} V~{:>4.1} at ({:.0}, {:.0})  {:.3} AU  {rate}",
                        kind, m.body.name, m.mag, m.x, m.y, m.delta_au
                    );
                }
                moving
            }
            None => {
                println!(
                    "minor bodies skipped: no acquisition time (pass --time or use a FITS with DATE-OBS)"
                );
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };

    let satellite_tracks = if let Some(request) = satellite_request {
        let satellite_catalog = load_satellite_catalog(request.input)?;
        if satellite_catalog.retrieved_at().is_none() {
            println!(
                "satellite elements: {} records ({})",
                satellite_catalog.len(),
                satellite_catalog.source()
            );
        }
        println!(
            "single exposure: {} to {} ({:.3}s, {:?})",
            request.exposure.start_utc.to_rfc3339(),
            request.exposure.end_utc.to_rfc3339(),
            request.exposure.duration_seconds(),
            request.exposure.provenance
        );
        let result = satellite_catalog.tracks_in_footprint(
            wcs,
            dims,
            &request.exposure,
            &TrackOptions {
                sample_interval_seconds: request.sample_interval_seconds,
                maximum_element_age_seconds: Some(request.maximum_element_age_seconds),
                ..TrackOptions::default()
            },
        )?;
        println!(
            "{} predicted satellite tracks in the field ({} elements considered):",
            result.tracks.len(),
            result.elements_considered
        );
        let mut undersampled = 0usize;
        for track in &result.tracks {
            let minimum_range = track
                .samples
                .iter()
                .map(|sample| sample.range_km)
                .fold(f64::INFINITY, f64::min);
            let illumination = match track.maximum_sunlight_fraction() {
                value if value <= 0.01 => "eclipsed",
                value if value >= 0.99 => "sunlit",
                _ => "partly sunlit",
            };
            let rate = track
                .maximum_apparent_rate_arcsec_per_second()
                .map(|rate| format!("{:>6.0}\"/s", rate))
                .unwrap_or_else(|| "      —".into());
            println!(
                "  {:<36} max el {:>5.1}°  {:>7.0} km  {rate}  {:<13} elements {:+.1}h",
                track.identity.display_label(),
                track.maximum_elevation_deg(),
                minimum_range,
                illumination,
                track.element_age_seconds / 3600.0
            );
            if let Some(pixel_rate) = track.maximum_pixel_rate_px_per_second()
                && pixel_rate > 0.0
                && track.clipped_length_px() / pixel_rate < 2.0 * track.sample_interval_seconds
            {
                undersampled += 1;
            }
        }
        if undersampled != 0 {
            eprintln!(
                "warning: {undersampled} track(s) cross the field in under two sample intervals; pass a smaller --satellite-sample-seconds for time-resolved samples"
            );
        }
        if result.propagation_failures != 0 {
            println!(
                "  {} element records could not be propagated",
                result.propagation_failures
            );
        }
        if result.stale_elements != 0 {
            eprintln!(
                "warning: {} element records were outside the {:.1}h maximum age",
                result.stale_elements,
                request.maximum_element_age_seconds / 3600.0
            );
            if result.stale_elements == result.elements_considered {
                eprintln!(
                    "warning: no element set is close enough to this exposure; use historical OMM/TLE data rather than current CelesTrak elements"
                );
            }
        }
        result.tracks
    } else {
        Vec::new()
    };

    if let Some(out) = annotate {
        let mut canvas = img.to_rgb8();
        for star in invocation.stars() {
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
            // An asymmetric extent with an unknown position angle must not be
            // drawn at a guessed orientation; fall back to the conservative
            // major-axis circle.
            let (semi_minor_px, angle_deg) = match p.angle_deg {
                Some(angle) => (p.semi_minor_px, angle),
                None => (p.semi_major_px, 0.0),
            };
            draw_rotated_ellipse(
                &mut canvas,
                (p.x, p.y),
                p.semi_major_px.max(12.0),
                semi_minor_px.max(12.0),
                angle_deg,
                image::Rgb([64, 220, 255]),
            );
        }
        for m in &moving {
            let center = (m.x, m.y);
            imageproc::drawing::draw_hollow_circle_mut(
                &mut canvas,
                (m.x.round() as i32, m.y.round() as i32),
                8,
                image::Rgb([255, 200, 64]),
            );
            // Arrow along the characteristic direction, one hour of
            // apparent motion long, clamped so slow movers stay legible
            // and near-Earth flybys don't shoot off the frame.
            if let (Some(pa), Some(rate)) = (m.direction_pa_deg, m.motion_arcsec_per_hour) {
                let max_px = (dims.0 as f64).hypot(dims.1 as f64) * 0.06;
                let length = (rate / wcs.scale_arcsec_per_px()).clamp(14.0, max_px);
                draw_motion_arrow(&mut canvas, center, (m.ra, m.dec), wcs, pa, length);
            }
        }
        for track in &satellite_tracks {
            draw_satellite_track(&mut canvas, track);
        }
        canvas
            .save(out)
            .with_context(|| format!("failed to write {}", out.display()))?;
        println!("annotated image written to {}", out.display());
    }
    Ok(())
}

fn draw_satellite_track(canvas: &mut image::RgbImage, track: &SatelliteTrack) {
    let color = image::Rgb([255, 72, 220]);
    for segment in &track.clipped_segments {
        imageproc::drawing::draw_line_segment_mut(
            canvas,
            (segment.start.x as f32, segment.start.y as f32),
            (segment.end.x as f32, segment.end.y as f32),
            color,
        );
    }
    let Some(first) = track.clipped_segments.first() else {
        return;
    };
    let last = track
        .clipped_segments
        .last()
        .expect("a first satellite segment implies a last segment");
    imageproc::drawing::draw_hollow_circle_mut(
        canvas,
        (first.start.x.round() as i32, first.start.y.round() as i32),
        4,
        color,
    );
    draw_pixel_arrowhead(canvas, last.start, last.end, color);

    let label = track.identity.display_label();
    let label = label.chars().take(28).collect::<String>();
    let width = (label.chars().count() * 8 * 2) as i32;
    let x = (first.start.x.round() as i32 + 7).clamp(1, (canvas.width() as i32 - width - 1).max(1));
    let y = (first.start.y.round() as i32 + 7).clamp(1, (canvas.height() as i32 - 17).max(1));
    draw_bitmap_text(canvas, x + 1, y + 1, &label, image::Rgb([0, 0, 0]), 2);
    draw_bitmap_text(canvas, x, y, &label, color, 2);
}

fn draw_pixel_arrowhead(
    canvas: &mut image::RgbImage,
    from: seiza_satellites::PixelPoint,
    tip: seiza_satellites::PixelPoint,
    color: image::Rgb<u8>,
) {
    let dx = tip.x - from.x;
    let dy = tip.y - from.y;
    if dx.hypot(dy) < 1.0e-6 {
        return;
    }
    let angle = dy.atan2(dx);
    for offset in [-150.0_f64, 150.0] {
        let wing = angle + offset.to_radians();
        imageproc::drawing::draw_line_segment_mut(
            canvas,
            (tip.x as f32, tip.y as f32),
            (
                (tip.x + wing.cos() * 7.0) as f32,
                (tip.y + wing.sin() * 7.0) as f32,
            ),
            color,
        );
    }
}

fn draw_bitmap_text(
    canvas: &mut image::RgbImage,
    x: i32,
    y: i32,
    text: &str,
    color: image::Rgb<u8>,
    scale: i32,
) {
    use font8x8::UnicodeFonts;

    for (character_index, character) in text.chars().enumerate() {
        let character = if character.is_ascii() { character } else { '?' };
        let Some(glyph) = font8x8::BASIC_FONTS.get(character) else {
            continue;
        };
        let origin_x = x + character_index as i32 * 8 * scale;
        for (row, bits) in glyph.into_iter().enumerate() {
            for column in 0..8 {
                if bits & (1 << column) == 0 {
                    continue;
                }
                for offset_y in 0..scale {
                    for offset_x in 0..scale {
                        let pixel_x = origin_x + column * scale + offset_x;
                        let pixel_y = y + row as i32 * scale + offset_y;
                        if pixel_x >= 0
                            && pixel_y >= 0
                            && pixel_x < canvas.width() as i32
                            && pixel_y < canvas.height() as i32
                        {
                            canvas.put_pixel(pixel_x as u32, pixel_y as u32, color);
                        }
                    }
                }
            }
        }
    }
}

/// Draw an arrow from a body's pixel position along a sky position angle
/// (degrees east of north). `length_px` is the *displayed* shaft length —
/// the caller derives it from the apparent rate and clamps it; the sky
/// PA is converted to a pixel direction through the WCS so flips and
/// rotation come out right.
fn draw_motion_arrow(
    canvas: &mut image::RgbImage,
    from: (f64, f64),
    sky: (f64, f64),
    wcs: &seiza::wcs::Wcs,
    pa_deg: f64,
    length_px: f64,
) {
    // Step a small angle along the position angle on the sphere and
    // project it: the great-circle destination formula, with the step
    // small enough that the pixel direction is exact at any latitude.
    let (sin_pa, cos_pa) = pa_deg.to_radians().sin_cos();
    let dec1 = sky.1.to_radians();
    let delta = (30.0_f64 / 3600.0).to_radians();
    let dec2 = (dec1.sin() * delta.cos() + dec1.cos() * delta.sin() * cos_pa).asin();
    let ra2 = sky.0.to_radians()
        + (sin_pa * delta.sin() * dec1.cos()).atan2(delta.cos() - dec1.sin() * dec2.sin());
    let Some((px, py)) = wcs.world_to_pixel(ra2.to_degrees(), dec2.to_degrees()) else {
        return;
    };
    let (dx, dy) = (px - from.0, py - from.1);
    let norm = dx.hypot(dy);
    if norm < 1e-9 {
        return;
    }
    let (ux, uy) = (dx / norm, dy / norm);
    let tip = (from.0 + ux * length_px, from.1 + uy * length_px);
    let color = image::Rgb([255, 200, 64]);
    imageproc::drawing::draw_line_segment_mut(
        canvas,
        (from.0 as f32, from.1 as f32),
        (tip.0 as f32, tip.1 as f32),
        color,
    );
    let head = 6.0_f64.min(length_px * 0.4);
    for sign in [1.0_f64, -1.0] {
        let angle = uy.atan2(ux) + sign * 150.0_f64.to_radians();
        imageproc::drawing::draw_line_segment_mut(
            canvas,
            (tip.0 as f32, tip.1 as f32),
            (
                (tip.0 + angle.cos() * head) as f32,
                (tip.1 + angle.sin() * head) as f32,
            ),
            color,
        );
    }
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

struct SolveBlindOptions<'a> {
    index_path: Option<&'a std::path::Path>,
    min_scale: f64,
    max_scale: f64,
    index_mag_limit: f32,
    max_hypotheses: usize,
    max_coarse_hypotheses: usize,
    sip_order: u8,
    sigma: f32,
    ignore_border: u32,
    detection_backend: DetectBackend,
    detection_fallback: DetectionFallback,
    detection_fallback_hypotheses: usize,
}

fn solve_blind_command(
    path: &std::path::Path,
    data: &std::path::Path,
    options: SolveBlindOptions<'_>,
) -> Result<()> {
    use seiza::blind::{BlindIndex, BlindParams, solve_blind};

    let img = load_image(path, options.detection_backend)?;
    let dims = img.dimensions();
    let config = DetectConfig {
        backend: options.detection_backend,
        sigma: options.sigma,
        ignore_border: options.ignore_border,
        max_stars: 600,
        ..Default::default()
    };
    let can_retry_f32 = options.detection_fallback == DetectionFallback::F32
        && auto_can_retry_f32(path, &img, options.detection_backend);
    let mut invocation = SolveInvocation::new(path, &img, config, options.detection_fallback);
    println!(
        "{} stars detected in {}x{} image",
        invocation.stars().len(),
        dims.0,
        dims.1
    );

    let catalog =
        TileCatalog::open(data).with_context(|| format!("failed to open {}", data.display()))?;
    let mut params = BlindParams {
        min_scale_arcsec_px: options.min_scale,
        max_scale_arcsec_px: options.max_scale,
        index_mag_limit: options.index_mag_limit,
        max_hypotheses: options.max_hypotheses,
        max_coarse_hypotheses: options.max_coarse_hypotheses,
        sip_order: options.sip_order,
        ..Default::default()
    };
    let started = std::time::Instant::now();
    let index = if let Some(path) = options.index_path {
        let index = BlindIndex::open(path)
            .map_err(anyhow::Error::from)
            .with_context(|| format!("failed to open {}", path.display()))?;
        params.index_mag_limit = index.index_mag_limit();
        params.max_pattern_deg = index.max_pattern_deg();
        warn_on_index_catalog_mismatch(&index, &catalog);
        println!(
            "pattern index: {} patterns mapped from {} in {:.2}s (G<={:.1})",
            index.pattern_count(),
            path.display(),
            started.elapsed().as_secs_f64(),
            index.index_mag_limit()
        );
        index
    } else {
        let index = BlindIndex::build(&catalog, &params);
        println!(
            "pattern index: {} patterns built in {:.2}s",
            index.pattern_count(),
            started.elapsed().as_secs_f64()
        );
        index
    };

    let started = std::time::Instant::now();
    let solution = invocation.solve_with_pass(|stars, pass| {
        let attempt_params = blind_params_for_detection_pass(
            &params,
            can_retry_f32,
            options.detection_fallback_hypotheses,
            pass,
        );
        solve_blind(stars, &catalog, &index, &attempt_params, dims)
    })?;
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

/// A blind index only produces hypotheses its build catalog supports; a
/// much shallower runtime catalog cannot verify the deep tiers and every
/// solve pays the full failure path with no diagnostic.
fn warn_on_index_catalog_mismatch(
    index: &seiza::blind::BlindIndex,
    catalog: &seiza::catalog::TileCatalog,
) {
    let built_from = index.source_star_count();
    let runtime = catalog.star_count();
    if built_from > 0 && runtime > 0 && built_from.max(runtime) > 2 * built_from.min(runtime) {
        eprintln!(
            "warning: blind index was built from a {built_from}-star catalog but solving \
             against {runtime} stars; deep-tier hypotheses may never verify"
        );
    }
}

fn build_blind_index_command(
    data: &std::path::Path,
    output: &std::path::Path,
    index_mag_limit: f32,
    max_pattern_deg: f64,
) -> Result<()> {
    use seiza::blind::{BlindIndex, BlindParams};

    let catalog =
        TileCatalog::open(data).with_context(|| format!("failed to open {}", data.display()))?;
    let params = BlindParams {
        index_mag_limit,
        max_pattern_deg,
        ..Default::default()
    };
    let started = std::time::Instant::now();
    let index = BlindIndex::build(&catalog, &params);
    println!(
        "built {} G<={index_mag_limit:.1} patterns in {:.2}s",
        index.pattern_count(),
        started.elapsed().as_secs_f64()
    );
    let started = std::time::Instant::now();
    index
        .write_to(output)
        .with_context(|| format!("failed to write {}", output.display()))?;
    println!(
        "wrote {} in {:.2}s",
        output.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn stack_requires_multiple_lights_and_linear_output() {
        assert!(
            Cli::try_parse_from(["seiza", "stack", "one.fits", "--output", "stack.fits"]).is_err()
        );
        let cli = Cli::try_parse_from([
            "seiza",
            "stack",
            "one.fits",
            "two.fits",
            "--output",
            "stack.fits",
            "--preview",
            "stack.png",
            "--report",
            "stack-report.json",
            "--normalization",
            "local",
            "--max-registration-drift",
            "512",
            "--max-registration-drift-fraction",
            "0.2",
        ])
        .unwrap();
        assert!(matches!(cli.command, Command::Stack(_)));
    }

    #[test]
    fn color_commands_make_the_composition_mode_explicit() {
        let lrgb = Cli::try_parse_from([
            "seiza",
            "color",
            "lrgb",
            "--luminance",
            "l.fits",
            "--red",
            "r.fits",
            "--green",
            "g.fits",
            "--blue",
            "b.fits",
            "--output",
            "lrgb.fits",
            "--preview",
            "lrgb.png",
        ])
        .unwrap();
        assert!(matches!(lrgb.command, Command::Color(_)));

        let narrowband = Cli::try_parse_from([
            "seiza",
            "color",
            "narrowband",
            "--ha",
            "ha.fits",
            "--oiii",
            "oiii.fits",
            "--palette",
            "foraxx-hoo",
            "--preview",
            "hoo.png",
        ])
        .unwrap();
        assert!(matches!(narrowband.command, Command::Color(_)));
    }

    #[test]
    fn master_commands_require_multiple_calibration_frames() {
        assert!(
            Cli::try_parse_from([
                "seiza",
                "master",
                "bias",
                "one.fits",
                "--output",
                "master-bias.fits",
            ])
            .is_err()
        );
        let cli = Cli::try_parse_from([
            "seiza",
            "master",
            "flat",
            "flat-1.fits",
            "flat-2.fits",
            "--output",
            "master-flat.fits",
            "--bias",
            "master-bias.fits",
            "--dark-flat",
            "master-dark-flat.fits",
            "--report",
            "master-flat.json",
        ])
        .unwrap();
        assert!(matches!(cli.command, Command::Master(_)));
    }

    #[test]
    fn download_data_help_leads_with_the_ready_to_use_route() {
        let error = match Cli::try_parse_from(["seiza", "download-data"]) {
            Ok(_) => panic!("download-data without a selection should show guidance"),
            Err(error) => error,
        };
        let help = error.to_string();
        let prebuilt = help.find("\n  prebuilt").expect("prebuilt command in help");
        let tycho2 = help.find("\n  tycho2").expect("advanced command in help");

        assert!(
            help.starts_with(
                "Download ready-to-use catalogs (recommended) or advanced source data"
            )
        );
        assert!(
            prebuilt < tycho2,
            "prebuilt should be listed first:\n{help}"
        );
        assert!(help.contains("RECOMMENDED:"));
        assert!(help.contains("seiza download-data prebuilt --output <directory>"));
        assert!(help.contains("seiza setup"));
        assert!(help.contains("are not needed to use Seiza"));
    }

    #[test]
    fn detection_backend_is_global_and_selectable() {
        let automatic = Cli::try_parse_from(["seiza", "detect", "image.jpg"]).unwrap();
        let before =
            Cli::try_parse_from(["seiza", "--detection-backend", "f32", "detect", "image.jpg"])
                .unwrap();
        let after =
            Cli::try_parse_from(["seiza", "detect", "image.jpg", "--detection-backend", "u8"])
                .unwrap();
        assert!(matches!(
            automatic.detection_backend,
            DetectionBackendArg::Auto
        ));
        assert!(matches!(before.detection_backend, DetectionBackendArg::F32));
        assert!(matches!(after.detection_backend, DetectionBackendArg::U8));
        assert_eq!(automatic.detection_fallback, DetectionFallback::F32);
        assert_eq!(automatic.detection_fallback_hypotheses, 64);
        let no_fallback = Cli::try_parse_from([
            "seiza",
            "solve-blind",
            "image.fits",
            "--data",
            "stars.bin",
            "--detection-fallback",
            "none",
            "--detection-fallback-hypotheses",
            "0",
        ])
        .unwrap();
        assert_eq!(no_fallback.detection_fallback, DetectionFallback::None);
        assert_eq!(no_fallback.detection_fallback_hypotheses, 0);
        assert!(
            Cli::try_parse_from([
                "seiza",
                "detect",
                "image.jpg",
                "--detection-backend",
                "other",
            ])
            .is_err()
        );
    }

    #[test]
    fn auto_retries_astronomy_and_converted_color_inputs() {
        let rgb = LoadedImage::Dynamic(image::DynamicImage::ImageRgb8(image::RgbImage::new(1, 1)));
        let luma =
            LoadedImage::Dynamic(image::DynamicImage::ImageLuma8(image::GrayImage::new(1, 1)));
        let jpeg = std::path::Path::new("image.jpg");
        let fits = std::path::Path::new("image.fits");
        let xisf = std::path::Path::new("image.XISF");
        assert!(auto_can_retry_f32(jpeg, &rgb, DetectBackend::Auto));
        assert!(!auto_can_retry_f32(jpeg, &rgb, DetectBackend::U8));
        assert!(!auto_can_retry_f32(jpeg, &luma, DetectBackend::Auto));
        assert!(auto_can_retry_f32(fits, &luma, DetectBackend::Auto));
        assert!(!auto_can_retry_f32(fits, &luma, DetectBackend::U8));
        assert!(auto_can_retry_f32(xisf, &luma, DetectBackend::Auto));
        assert_eq!(
            astronomy_image_format(xisf),
            Some(AstronomyImageFormat::Xisf)
        );
        assert!(!is_fits_path(xisf));
    }

    #[test]
    fn fits_loader_keeps_u8_mtf_and_linear_f32_paths_distinct() {
        const CARD: usize = 80;
        let card = |keyword: &str, value: &str| {
            let mut card = [b' '; CARD];
            card[..keyword.len()].copy_from_slice(keyword.as_bytes());
            card[8] = b'=';
            card[9] = b' ';
            card[10..10 + value.len()].copy_from_slice(value.as_bytes());
            card
        };
        let mut bytes = Vec::new();
        for (keyword, value) in [
            ("SIMPLE", "T"),
            ("BITPIX", "16"),
            ("NAXIS", "2"),
            ("NAXIS1", "2"),
            ("NAXIS2", "2"),
            ("BZERO", "32768"),
            ("BSCALE", "1"),
        ] {
            bytes.extend_from_slice(&card(keyword, value));
        }
        let mut end = [b' '; CARD];
        end[..3].copy_from_slice(b"END");
        bytes.extend_from_slice(&end);
        bytes.resize(bytes.len().next_multiple_of(2880), b' ');
        for value in [1000u16, 1001, 32768, 65535] {
            bytes.extend_from_slice(&(value ^ 0x8000).to_be_bytes());
        }

        let path = std::env::temp_dir().join(format!(
            "seiza-cli-detection-backends-{}-{}.fits",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, bytes).unwrap();
        let compact = load_image(&path, DetectBackend::Auto).unwrap();
        let precise = load_image(&path, DetectBackend::F32).unwrap();

        assert!(matches!(
            compact,
            LoadedImage::Dynamic(image::DynamicImage::ImageLuma8(_))
        ));
        let LoadedImage::LinearF32 { luma, .. } = precise else {
            panic!("forced f32 astronomy image must retain a linear f32 buffer");
        };
        assert_eq!(luma[0], 1000.0 / u16::MAX as f32);
        assert_eq!(luma[1], 1001.0 / u16::MAX as f32);
        assert_ne!(luma[0], luma[1]);

        let mut invocation = SolveInvocation::new(
            &path,
            &compact,
            DetectConfig::default(),
            DetectionFallback::F32,
        );
        let mut passes = Vec::new();
        invocation
            .solve_with_pass(|_, pass| {
                passes.push(pass);
                if pass == DetectionPass::Primary {
                    Err(seiza::Error::Solve("exercise f32 reload".into()))
                } else {
                    Ok(())
                }
            })
            .unwrap();
        assert_eq!(passes, [DetectionPass::Primary, DetectionPass::F32Fallback]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn solve_invocation_honors_disabled_fallback() {
        let image =
            LoadedImage::Dynamic(image::DynamicImage::ImageRgb8(image::RgbImage::new(2, 2)));
        let mut invocation = SolveInvocation::new(
            std::path::Path::new("image.jpg"),
            &image,
            DetectConfig::default(),
            DetectionFallback::None,
        );
        let mut passes = Vec::new();
        let result: Result<()> = invocation.solve_with_pass(|_, pass| {
            passes.push(pass);
            Err(seiza::Error::Solve("expected miss".into()))
        });
        assert!(result.is_err());
        assert_eq!(passes, [DetectionPass::Primary]);
    }

    #[test]
    fn blind_fallback_budget_only_caps_the_primary_pass() {
        let params = seiza::blind::BlindParams {
            max_hypotheses: 400,
            ..Default::default()
        };
        let primary = blind_params_for_detection_pass(&params, true, 64, DetectionPass::Primary);
        let fallback =
            blind_params_for_detection_pass(&params, true, 64, DetectionPass::F32Fallback);
        let unavailable =
            blind_params_for_detection_pass(&params, false, 64, DetectionPass::Primary);
        let unlimited = blind_params_for_detection_pass(&params, true, 0, DetectionPass::Primary);

        assert_eq!(primary.max_hypotheses, 64);
        assert_eq!(fallback.max_hypotheses, 400);
        assert_eq!(unavailable.max_hypotheses, 400);
        assert_eq!(unlimited.max_hypotheses, 400);
    }

    #[test]
    fn gaia_accepts_a_separated_negative_magnitude() {
        let cli = Cli::try_parse_from([
            "seiza",
            "download-data",
            "gaia",
            "--output",
            "gaia",
            "--max-mag",
            "-1.5",
            "--chunks",
            "1",
        ])
        .unwrap();

        let Command::DownloadData {
            source: DownloadSource::Gaia {
                max_mag, chunks, ..
            },
        } = cli.command
        else {
            panic!("expected Gaia download command");
        };
        assert_eq!(max_mag, -1.5);
        assert_eq!(chunks, 1);
    }

    #[test]
    fn satellite_history_accepts_multiple_explicit_epochs_and_cache() {
        let cli = Cli::try_parse_from([
            "seiza",
            "download-data",
            "satellite-history",
            "--epoch",
            "2025-10-17T12:50:21Z",
            "2025-10-18T12:51:42Z",
            "--cache",
            "satellite-cache",
            "--origin",
        ])
        .unwrap();

        let Command::DownloadData {
            source:
                DownloadSource::SatelliteHistory {
                    epoch,
                    cache,
                    origin,
                },
        } = cli.command
        else {
            panic!("expected satellite history download command");
        };
        assert_eq!(epoch.len(), 2);
        assert_eq!(cache, Some(PathBuf::from("satellite-cache")));
        assert!(origin);
    }

    #[test]
    fn gaia_rejects_zero_chunks_at_parse_time() {
        assert!(
            Cli::try_parse_from([
                "seiza",
                "download-data",
                "gaia",
                "--output",
                "gaia",
                "--chunks",
                "0",
            ])
            .is_err()
        );
    }

    #[test]
    fn satellite_cli_sources_are_explicit_and_mutually_exclusive() {
        assert!(
            Cli::try_parse_from([
                "seiza",
                "solve",
                "image.fits",
                "--scale",
                "1.5",
                "--satellites-celestrak",
            ])
            .is_ok()
        );
        assert!(
            Cli::try_parse_from([
                "seiza",
                "solve",
                "image.fits",
                "--scale",
                "1.5",
                "--satellites",
                "elements.json",
                "--satellites-celestrak",
            ])
            .is_err()
        );
    }

    #[test]
    fn satellite_exposure_prefers_unambiguous_fits_bounds() {
        use seiza_fits::HeaderValue::{Float, String as Text};

        let headers = vec![
            ("DATE-OBS".into(), Text("2024-05-02T11:59:00".into())),
            ("DATE-BEG".into(), Text("2024-05-02T12:00:00".into())),
            ("DATE-END".into(), Text("2024-05-02T12:00:30".into())),
            ("EXPTIME".into(), Float(999.0)),
            ("OBSGEO-B".into(), Float(37.3)),
            ("OBSGEO-L".into(), Float(-122.0)),
            ("OBSGEO-H".into(), Float(50.0)),
        ];
        let exposure =
            resolve_single_exposure_from_headers(&headers, None, None, None, None, None).unwrap();
        assert_eq!(exposure.provenance, ExposureProvenance::FitsBounds);
        assert!((exposure.duration_seconds() - 30.0).abs() < 1.0e-6);
        assert_eq!(
            exposure.start_utc,
            UtcTimestamp::parse("2024-05-02T12:00:00Z").unwrap()
        );
    }

    #[test]
    fn satellite_exposure_falls_back_to_date_obs_and_exptime() {
        use seiza_fits::HeaderValue::{Float, String as Text};

        let headers = vec![
            ("DATE-OBS".into(), Text("2024-05-02T12:00:00".into())),
            ("EXPTIME".into(), Float(45.5)),
            ("OBSGEO-X".into(), Float(-2_700_000.0)),
            ("OBSGEO-Y".into(), Float(-4_300_000.0)),
            ("OBSGEO-Z".into(), Float(3_850_000.0)),
        ];
        let exposure =
            resolve_single_exposure_from_headers(&headers, None, None, None, None, None).unwrap();
        assert_eq!(
            exposure.provenance,
            ExposureProvenance::FitsDateObsAndExposure
        );
        assert!((exposure.duration_seconds() - 45.5).abs() < 1.0e-6);
    }

    #[test]
    fn satellite_exposure_supports_date_avg_as_the_midpoint() {
        use seiza_fits::HeaderValue::{Float, String as Text};

        let headers = vec![
            ("DATE-AVG".into(), Text("2024-05-02T12:00:30".into())),
            ("EXPTIME".into(), Float(60.0)),
            ("OBSGEO-B".into(), Float(37.3)),
            ("OBSGEO-L".into(), Float(-122.0)),
            ("OBSGEO-H".into(), Float(50.0)),
        ];
        let exposure =
            resolve_single_exposure_from_headers(&headers, None, None, None, None, None).unwrap();
        assert_eq!(
            exposure.provenance,
            ExposureProvenance::FitsDateAvgAndExposure
        );
        assert_eq!(
            exposure.start_utc,
            UtcTimestamp::parse("2024-05-02T12:00:00Z").unwrap()
        );
        assert_eq!(
            exposure.end_utc,
            UtcTimestamp::parse("2024-05-02T12:01:00Z").unwrap()
        );
    }

    #[test]
    fn satellite_exposure_supports_date_end_as_shutter_close() {
        use seiza_fits::HeaderValue::{Float, String as Text};

        let headers = vec![
            ("DATE-END".into(), Text("2024-05-02T12:01:00".into())),
            ("EXPTIME".into(), Float(60.0)),
            ("OBSGEO-B".into(), Float(37.3)),
            ("OBSGEO-L".into(), Float(-122.0)),
            ("OBSGEO-H".into(), Float(50.0)),
        ];
        let exposure =
            resolve_single_exposure_from_headers(&headers, None, None, None, None, None).unwrap();
        assert_eq!(exposure.provenance, ExposureProvenance::FitsEndAndExposure);
        assert_eq!(
            exposure.start_utc,
            UtcTimestamp::parse("2024-05-02T12:00:00Z").unwrap()
        );
        assert_eq!(
            exposure.end_utc,
            UtcTimestamp::parse("2024-05-02T12:01:00Z").unwrap()
        );
    }

    #[test]
    fn explicit_satellite_metadata_overrides_fits_time_frame() {
        use seiza_fits::HeaderValue::String as Text;

        let headers = vec![
            ("TIMESYS".into(), Text("TDB".into())),
            ("TREFPOS".into(), Text("BARYCENTER".into())),
        ];
        let exposure = resolve_single_exposure_from_headers(
            &headers,
            Some("2024-05-02T12:00:00Z"),
            Some(10.0),
            Some(37.3),
            Some(-122.0),
            Some(50.0),
        )
        .unwrap();
        assert_eq!(exposure.provenance, ExposureProvenance::Explicit);
        assert!((exposure.duration_seconds() - 10.0).abs() < 1.0e-6);
    }

    #[test]
    fn satellite_exposure_requires_observer_and_one_interval() {
        use seiza_fits::HeaderValue::{Float, String as Text};

        let no_observer = vec![
            ("DATE-OBS".into(), Text("2024-05-02T12:00:00".into())),
            ("EXPTIME".into(), Float(30.0)),
        ];
        assert!(
            resolve_single_exposure_from_headers(&no_observer, None, None, None, None, None)
                .is_err()
        );

        let no_duration = vec![
            ("DATE-OBS".into(), Text("2024-05-02T12:00:00".into())),
            ("OBSGEO-B".into(), Float(37.3)),
            ("OBSGEO-L".into(), Float(-122.0)),
            ("OBSGEO-H".into(), Float(50.0)),
        ];
        assert!(
            resolve_single_exposure_from_headers(&no_duration, None, None, None, None, None)
                .is_err()
        );
    }
}
