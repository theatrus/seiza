use anyhow::{Context, Result};
use clap::Args;
use seiza_fits::{HeaderValue, WriteHeaderCard};
use seiza_stacking::{
    ColorCalibration, ColorCalibrationOptions, ColorCalibrationOutcome, ColorCalibrationSkip,
    ColorCalibrationSkipReason, GaiaColorSource, GaiaColors, SOLAR_BP_RP, calibrate_and_stretch,
    calibrate_color_image, write_processed_image_fits_f32,
};
use seiza_stretch::{ImageShape, StretchConfig, StretchParams, StretchStack};
use std::path::{Path, PathBuf};

#[derive(Args)]
pub(crate) struct ColorCalibrateArgs {
    /// Linear RGB FITS or XISF input with an astrometric solution in its
    /// headers
    input: PathBuf,
    /// Calibrated linear 32-bit floating-point FITS output
    #[arg(short, long)]
    output: PathBuf,
    /// JSON report: gains, offsets, colour fits, and every measured star
    #[arg(long)]
    report: Option<PathBuf>,
    /// Also write a display-referred PNG, JPEG or TIFF, stretched after
    /// calibration. When calibration cannot run (no solution in the
    /// headers, no Gaia colours to be had, too few stars), both images are
    /// written uncalibrated, the reason is printed and written to --report,
    /// and the command succeeds
    #[arg(long)]
    display: Option<PathBuf>,
    /// The display stretch: a stretch configuration as JSON, or a file of
    /// it, with one stage or an array of stages applied in order (default:
    /// linked Auto-MTF)
    #[arg(long, requires = "display")]
    stretch: Option<String>,
    /// Gaia BP-RP colour to render neutral; the default is the Sun's (G2V)
    #[arg(long, default_value_t = SOLAR_BP_RP)]
    white_bp_rp: f32,
    /// Faintest Gaia G magnitude to fetch from the archive or read from a
    /// CSV; the offline catalog holds every star to G 15
    #[arg(long, default_value_t = 15.0)]
    gaia_max_mag: f32,
    /// Aperture radius in pixels (default: twice the measured FWHM)
    #[arg(long)]
    aperture: Option<f64>,
    /// Leave each channel's background level as it is
    #[arg(long)]
    no_background_neutralization: bool,
    /// Where fetched Gaia fields are kept for reuse (default: Seiza's data
    /// directory)
    #[arg(long, conflicts_with = "gaia_csv")]
    gaia_cache: Option<PathBuf>,
    /// Read Gaia DR3 photometry from this CSV instead of the ESA archive:
    /// columns ra, dec, pmra, pmdec, phot_g_mean_mag, phot_bp_mean_mag,
    /// phot_rp_mean_mag and ruwe, as the archive returns them
    #[arg(long, conflicts_with = "gaia_catalog")]
    gaia_csv: Option<PathBuf>,
    /// Offline Gaia photometry catalog, or a directory holding
    /// stars-gaia-photometry.bin (built by `seiza build-data
    /// gaia-photometry`). Seiza's catalog directories are searched when
    /// omitted, and the ESA archive is queried when none is found
    #[arg(long)]
    gaia_catalog: Option<PathBuf>,
}

pub(crate) fn run(args: ColorCalibrateArgs) -> Result<()> {
    let mut frame = crate::common::open_frame(&args.input, "colour calibration input")?;
    if frame.image.channels != 3 {
        anyhow::bail!(
            "{} has {} channel(s); colour calibration needs a debayered RGB image",
            args.input.display(),
            frame.image.channels
        );
    }
    let header = |key: &str| {
        frame
            .headers
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    };
    let wcs = seiza::Wcs::from_fits_values(
        |key| header(key).and_then(HeaderValue::as_f64),
        |key| header(key).and_then(|value| value.as_str().map(str::to_owned)),
    );
    if wcs.is_none() && args.display.is_none() {
        anyhow::bail!(
            "{} has no TAN astrometric solution in its headers; plate-solve it first",
            args.input.display()
        );
    }
    // Observed at DATE-OBS when the headers say; Gaia positions are J2016.0.
    let epoch = ["DATE-AVG", "DATE-OBS", "DATE-BEG"]
        .iter()
        .find_map(|key| header(key).and_then(|value| value.as_str()))
        .and_then(crate::parse_iso_jd)
        .map(|jd| 2000.0 + (jd - 2_451_545.0) / 365.25);
    let (width, height) = (frame.image.width, frame.image.height);

    // Where the Gaia colours come from. With --display, colours that cannot
    // be had are reported and the image is still stretched.
    let gaia = match &wcs {
        Some(wcs) => match gaia_input(&args, wcs, width, height, epoch) {
            Ok(gaia) => Ok(gaia),
            Err(error) if args.display.is_some() => Err(ColorCalibrationSkip {
                reason: ColorCalibrationSkipReason::NoCatalog,
                detail: format!("{error:#}"),
            }),
            Err(error) => return Err(error),
        },
        None => Err(ColorCalibrationSkip {
            reason: ColorCalibrationSkipReason::NoSolution,
            detail: "the image has no TAN astrometric solution in its headers".into(),
        }),
    };
    let options = ColorCalibrationOptions {
        white_bp_rp: args.white_bp_rp,
        aperture_radius: args.aperture,
        neutralize_background: !args.no_background_neutralization,
        ..ColorCalibrationOptions::default()
    };

    let outcome = match &args.display {
        None => {
            let gaia = gaia.expect("only --display goes on without Gaia colours");
            let mut image = frame.image.clone();
            let outcome = calibrate_color_image(&mut image, wcs.as_ref(), gaia.colors(), &options);
            if let ColorCalibrationOutcome::Skipped(skip) = &outcome {
                anyhow::bail!("colour calibration failed: {}", skip.detail);
            }
            outcome
        }
        Some(display) => {
            let stretch = display_stretch(args.stretch.as_deref())?;
            let (pixels, outcome) = match &gaia {
                Ok(gaia) => {
                    let result = calibrate_and_stretch(
                        frame.image.clone(),
                        wcs.as_ref(),
                        gaia.colors(),
                        &options,
                        &stretch,
                    )
                    .context("display stretch failed")?;
                    (result.image.data, result.calibration)
                }
                Err(skip) => {
                    let shape = ImageShape::new(width, height, 3);
                    let output = stretch
                        .apply_f32_image(&frame.image.data, shape)
                        .context("display stretch failed")?;
                    (output.data, ColorCalibrationOutcome::Skipped(skip.clone()))
                }
            };
            let pixels = pixels
                .iter()
                .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect();
            crate::preview::write_display_image(display, width, height, 3, pixels)?;
            let summary = match &outcome {
                ColorCalibrationOutcome::Calibrated(_) => "colour-calibrated, stretched",
                ColorCalibrationOutcome::Skipped(_) => "uncalibrated, stretched",
            };
            crate::common::wrote(display, format_args!("{summary}"));
            outcome
        }
    };
    match outcome {
        ColorCalibrationOutcome::Calibrated(calibration) => {
            report_calibration(&calibration);
            calibration.apply(&mut frame.image)?;
            write_calibrated(&args, &frame, &calibration)
        }
        ColorCalibrationOutcome::Skipped(skip) => {
            println!("not colour calibrated: {}", skip.detail);
            write_processed_image_fits_f32(&args.output, &frame.image, &frame.headers, &[])
                .with_context(|| format!("could not write {}", args.output.display()))?;
            if let Some(path) = &args.report {
                crate::provenance::write_json_atomic(
                    path,
                    &ColorCalibrationOutcome::Skipped(skip.clone()),
                )
                .with_context(|| format!("could not write {}", path.display()))?;
            }
            crate::common::wrote(&args.output, format_args!("uncalibrated, linear f32"));
            Ok(())
        }
    }
}

/// Gaia colours for an image: a CSV, the offline catalog, or ESA's archive.
enum GaiaInput {
    Catalog(seiza::catalog::PhotometryCatalog),
    Sources {
        sources: Vec<GaiaColorSource>,
        epoch: Option<f64>,
    },
}

impl GaiaInput {
    fn colors(&self) -> GaiaColors<'_> {
        match self {
            Self::Catalog(catalog) => GaiaColors::Catalog(catalog),
            Self::Sources { sources, epoch } => GaiaColors::Sources {
                sources,
                epoch: *epoch,
            },
        }
    }
}

fn gaia_input(
    args: &ColorCalibrateArgs,
    wcs: &seiza::Wcs,
    width: usize,
    height: usize,
    epoch: Option<f64>,
) -> Result<GaiaInput> {
    let from_rows = |rows: Vec<seiza_sources::GaiaPhotometry>| GaiaInput::Sources {
        sources: rows
            .iter()
            .filter(|star| star.g <= args.gaia_max_mag)
            .map(|star| GaiaColorSource {
                ra: star.ra,
                dec: star.dec,
                pmra: star.pmra,
                pmdec: star.pmdec,
                g: star.g,
                bp_rp: star.bp_rp(),
                ruwe: star.ruwe,
            })
            .collect(),
        epoch,
    };
    if let Some(path) = &args.gaia_csv {
        let csv = std::fs::read_to_string(path)
            .with_context(|| format!("could not read {}", path.display()))?;
        let rows = seiza_sources::parse_gaia_photometry(&csv)
            .with_context(|| format!("{} is not a Gaia photometry CSV", path.display()))?;
        return Ok(from_rows(rows));
    }
    if let Some(path) = seiza::data_paths::gaia_photometry(args.gaia_catalog.as_deref())? {
        let catalog = seiza::catalog::PhotometryCatalog::open(&path)
            .with_context(|| format!("could not open {}", path.display()))?;
        println!(
            "Gaia photometry from {} (epoch {})",
            path.display(),
            catalog.epoch()
        );
        return Ok(GaiaInput::Catalog(catalog));
    }
    // Image sides fit u32: decoders and detection use u32 dimensions.
    let middle = seiza::image_center(width as u32, height as u32);
    let center = wcs.pixel_to_world(middle.0, middle.1);
    let (right, bottom) = (width as f64 - 0.5, height as f64 - 0.5);
    let radius = [(-0.5, -0.5), (right, -0.5), (-0.5, bottom), (right, bottom)]
        .iter()
        .map(|&(x, y)| angular_separation(center, wcs.pixel_to_world(x, y)))
        .fold(0.0_f64, f64::max)
        * 1.02;
    let cache = args.gaia_cache.clone().unwrap_or_else(default_gaia_cache);
    Ok(from_rows(gaia_field(
        &cache,
        center,
        radius,
        args.gaia_max_mag,
    )?))
}

/// The stretch `--stretch` gives, as JSON or a JSON file, or linked
/// Auto-MTF.
fn display_stretch(stretch: Option<&str>) -> Result<StretchStack> {
    let Some(stretch) = stretch else {
        return Ok(StretchStack::single(StretchConfig::auto_mtf(
            StretchParams::default(),
            crate::preview::MAXIMUM_SAMPLES,
        )));
    };
    let text = if stretch.trim_start().starts_with(['{', '[']) {
        stretch.to_owned()
    } else {
        std::fs::read_to_string(stretch).with_context(|| format!("could not read {stretch}"))?
    };
    let value: serde_json::Value =
        serde_json::from_str(&text).context("the stretch is not JSON")?;
    if value.is_array() {
        serde_json::from_value(value).context("invalid stretch stages")
    } else {
        serde_json::from_value(value)
            .map(StretchStack::single)
            .context("invalid stretch")
    }
}

fn report_calibration(calibration: &ColorCalibration) {
    for (name, fit) in [
        ("R/G", &calibration.red_fit),
        ("B/G", &calibration.blue_fit),
    ] {
        println!(
            "{name}: {:+.3} {:+.3} x (BP-RP) mag, scatter {:.3} mag over {} stars",
            fit.intercept, fit.slope, fit.scatter, fit.stars
        );
    }
    println!(
        "gains R {:.4} G {:.4} B {:.4}, offsets R {:+.3} G {:+.3} B {:+.3} (white BP-RP {:.2}, aperture {:.1} px, {} of {} stars measured)",
        calibration.gains[0],
        calibration.gains[1],
        calibration.gains[2],
        calibration.offsets[0],
        calibration.offsets[1],
        calibration.offsets[2],
        calibration.white_bp_rp,
        calibration.aperture_radius,
        calibration.stars_measured,
        calibration.stars_offered,
    );
}

/// Write the calibrated linear image and the report.
fn write_calibrated(
    args: &ColorCalibrateArgs,
    frame: &seiza_stacking::FitsFrame,
    calibration: &ColorCalibration,
) -> Result<()> {
    let cards = [
        WriteHeaderCard::new("COLORCAL", HeaderValue::String("Gaia DR3 BP-RP".into()))
            .with_comment("photometric colour calibration"),
        WriteHeaderCard::new(
            "CCWHITE",
            HeaderValue::Float(f64::from(calibration.white_bp_rp)),
        )
        .with_comment("white reference Gaia BP-RP"),
        WriteHeaderCard::new(
            "CCGAINR",
            HeaderValue::Float(f64::from(calibration.gains[0])),
        )
        .with_comment("red gain"),
        WriteHeaderCard::new(
            "CCGAING",
            HeaderValue::Float(f64::from(calibration.gains[1])),
        )
        .with_comment("green gain"),
        WriteHeaderCard::new(
            "CCGAINB",
            HeaderValue::Float(f64::from(calibration.gains[2])),
        )
        .with_comment("blue gain"),
        WriteHeaderCard::new(
            "CCSTARS",
            HeaderValue::Integer(calibration.red_fit.stars as i64),
        )
        .with_comment("stars in the red colour fit"),
    ];
    write_processed_image_fits_f32(&args.output, &frame.image, &frame.headers, &cards)
        .with_context(|| format!("could not write {}", args.output.display()))?;
    if let Some(path) = &args.report {
        crate::provenance::write_json_atomic(path, calibration)
            .with_context(|| format!("could not write {}", path.display()))?;
    }
    crate::common::wrote(
        &args.output,
        format_args!(
            "colour-calibrated against {} Gaia stars, linear f32",
            calibration.red_fit.stars.min(calibration.blue_fit.stars)
        ),
    );
    Ok(())
}

fn default_gaia_cache() -> PathBuf {
    let catalogs = seiza::data_paths::default_catalog_dir();
    catalogs
        .parent()
        .map_or_else(|| catalogs.clone(), Path::to_path_buf)
        .join("gaia-fields")
}

/// Gaia DR3 photometry for a field, from the cache when this field was
/// fetched before and from the ESA archive otherwise.
fn gaia_field(
    cache: &Path,
    center: (f64, f64),
    radius: f64,
    max_mag: f32,
) -> Result<Vec<seiza_sources::GaiaPhotometry>> {
    // A re-solved field moves its centre slightly: round the centre to
    // 0.01 degrees and widen the radius past the rounding, so it finds the
    // same cached field. The magnitude limit is kept exactly.
    let center = (
        (center.0 * 100.0).round() / 100.0,
        (center.1 * 100.0).round() / 100.0,
    );
    let radius = ((radius + 0.01) / 0.05).ceil() * 0.05;
    let name = format!(
        "gaia-dr3-{:.2}{:+.2}-r{:.2}-g{}.csv",
        center.0, center.1, radius, max_mag
    );
    let path = cache.join(name);
    if let Ok(csv) = std::fs::read_to_string(&path)
        && let Ok(stars) = seiza_sources::parse_gaia_photometry(&csv)
    {
        println!("Gaia field from {}", path.display());
        return Ok(stars);
    }
    println!(
        "fetching Gaia DR3 within {radius:.2} deg of ({:.4}, {:+.4}) from the ESA archive",
        center.0, center.1
    );
    let downloader = seiza_sources::SourceDownloader::new()?;
    let csv = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(downloader.gaia_photometry_cone_csv(center.0, center.1, radius, max_mag))
        .context("Gaia archive query failed")?;
    let stars = seiza_sources::parse_gaia_photometry(&csv)?;
    if std::fs::create_dir_all(cache).is_ok() {
        let partial = path.with_extension("csv.partial");
        if std::fs::write(&partial, &csv).is_ok() {
            let _ = std::fs::rename(&partial, &path);
        }
    }
    Ok(stars)
}

fn angular_separation(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (ra1, dec1) = (a.0.to_radians(), a.1.to_radians());
    let (ra2, dec2) = (b.0.to_radians(), b.1.to_radians());
    let cos = dec1.sin() * dec2.sin() + dec1.cos() * dec2.cos() * (ra1 - ra2).cos();
    cos.clamp(-1.0, 1.0).acos().to_degrees()
}
