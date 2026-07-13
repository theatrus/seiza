use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use seiza::catalog::{StarCatalog, TileCatalog};
use seiza::solve::{SolveHint, solve};
use seiza::{DetectConfig, detect_stars};
use std::path::PathBuf;

mod build_data;
mod download_data;

/// Open an image file; FITS files are MTF-autostretched to 8-bit grayscale.
fn load_image(path: &std::path::Path) -> Result<image::DynamicImage> {
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
    /// Gaia DR3 star positions via ESA TAP (~1.5 GB, resumable)
    Gaia {
        /// Directory to download into
        #[arg(long)]
        output: PathBuf,
        /// Magnitude limit for the download
        #[arg(long, default_value_t = 15.0)]
        max_mag: f32,
    },
    /// The Rochester Astronomy active supernova/transient list (refetched)
    Transients {
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
        } => {
            let hint = match (ra, dec) {
                (Some(ra), Some(dec)) => (ra, dec),
                _ => fits_hint(&image).ok_or_else(|| {
                    anyhow::anyhow!("--ra/--dec required (no RA/DEC headers found in the image)")
                })?,
            };
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
            )
        }
        Command::DownloadData { source } => match source {
            DownloadSource::Tycho2 { output } => download_data::download_tycho2(&output),
            DownloadSource::Openngc { output } => download_data::download_openngc(&output),
            DownloadSource::Objects { output } => download_data::download_objects(&output),
            DownloadSource::Gaia { output, max_mag } => {
                download_data::download_gaia(&output, max_mag)
            }
            DownloadSource::Transients { output } => download_data::download_transients(&output),
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
