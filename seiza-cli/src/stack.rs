use crate::preview::{PreviewTransfer, write_preview};
use crate::provenance::{FileIdentity, file_identity, validate_path_roles, write_json_atomic};
use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use rayon::prelude::*;
use seiza_stacking::{
    CalibrationMasters, DeltaSigmaOptions, FitsFrame, FrameDisposition, NormalizationMode,
    RegistrationOptions, RejectionMode, StackOptions,
};
use seiza_stacking::{Continue, PipelineOptions};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum NormalizationArg {
    None,
    Global,
    Local,
    /// The global gain, with local background offsets that remove the
    /// seams frames' edges leave where their sky gradients differ
    LocalBackground,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RejectionArg {
    None,
    DeltaSigma,
}

#[derive(Clone, Copy, ValueEnum)]
enum RegistrationModelArg {
    Similarity,
    Affine,
    Quadratic,
}

#[derive(Clone, Copy, ValueEnum)]
enum ReferenceArg {
    Auto,
    First,
}

#[derive(Clone, Copy, ValueEnum)]
enum WeightingArg {
    Equal,
    InverseNoise,
}

#[derive(Clone, Copy, ValueEnum)]
enum DemosaicArg {
    Vng,
    Mhc,
    Bilinear,
}

#[derive(Clone, Copy, ValueEnum)]
enum InterpolationArg {
    Bilinear,
    Lanczos3,
}

#[derive(Args)]
pub(crate) struct StackArgs {
    /// FITS or XISF light frames, in acquisition order; see --reference
    #[arg(required = true, num_args = 2..)]
    images: Vec<PathBuf>,
    /// Which frame is the reference: the clearest and sharpest by a quick
    /// star count over every frame, or the first frame given
    #[arg(long, value_enum, default_value = "auto")]
    reference: ReferenceArg,
    /// Linear 32-bit floating-point FITS stack
    #[arg(short, long)]
    output: PathBuf,
    /// Optional display-stretched PNG; never used by stacking math
    #[arg(long)]
    preview: Option<PathBuf>,
    /// JSON admission/provenance report with SHA-256 input identities
    #[arg(long)]
    report: Option<PathBuf>,
    /// Integrated master bias FITS or XISF
    #[arg(long)]
    bias: Option<PathBuf>,
    /// Integrated master dark FITS or XISF
    #[arg(long)]
    dark: Option<PathBuf>,
    /// Integrated master flat FITS or XISF in the light frame's raw sampling
    #[arg(long)]
    flat: Option<PathBuf>,
    /// Override master-dark exposure time in seconds
    #[arg(long, requires = "dark")]
    dark_exposure_seconds: Option<f64>,
    /// Background normalization applied after registration
    #[arg(long, value_enum, default_value = "local-background")]
    normalization: NormalizationArg,
    /// Tile size for --normalization local
    #[arg(long, default_value_t = 256)]
    local_tile_size: usize,
    /// Online sample-rejection estimator
    #[arg(long, value_enum, default_value = "delta-sigma")]
    rejection: RejectionArg,
    /// Low residual rejection threshold
    #[arg(long, default_value_t = 3.0)]
    sigma_low: f32,
    /// High residual rejection threshold
    #[arg(long, default_value_t = 3.0)]
    sigma_high: f32,
    /// Accepted observations before online rejection begins
    #[arg(long, default_value_t = 5)]
    rejection_warmup: u32,
    /// After stacking, read every admitted frame back and integrate them again
    /// with three-pass, leave-one-out rejection at --sigma-low and
    /// --sigma-high. This removes satellite trails and other transients from
    /// the reference and warm-up frames, which online rejection cannot
    /// revisit.
    #[arg(long)]
    reintegrate: bool,
    /// Drizzle the stack onto a grid this many times finer than the
    /// reference (1 to 4), as PixInsight's DrizzleIntegration does after
    /// ImageIntegration: implies --reintegrate, whose rejection decides which
    /// pixels each frame drops. The output is the drizzled image, with the
    /// reference WCS scaled to it. Bayer frames drizzle their photosites,
    /// each into its own color, without demosaicing.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=4))]
    drizzle: Option<u32>,
    /// Each drizzle drop's side as a fraction of a source pixel; smaller
    /// drops keep more detail and need more dithered frames to fill the
    /// grid. Defaults to WBPP's: 0.9 for monochrome frames, 1.0 for Bayer
    #[arg(long, requires = "drizzle")]
    drizzle_drop_shrink: Option<f32>,
    /// Where --reintegrate keeps each admitted frame's registered image,
    /// written as the frame is stacked and read back by each of its passes
    /// (about four bytes per output sample per frame); defaults to the
    /// output's directory
    #[arg(long)]
    scratch_directory: Option<PathBuf>,
    /// Integrate each Bayer frame's photosites without demosaicing: every
    /// registered pixel takes the nearest photosite in its own color, and
    /// the stack fills the colors in. Sharper colors when frames are
    /// dithered by several pixels; lower SNR when they barely move.
    /// Registration still uses the demosaiced frame
    #[arg(long)]
    bayer_drizzle: bool,
    /// How much each frame counts: equally, or by the inverse of its noise
    /// variance relative to the reference, so hazy frames count for less
    #[arg(long, value_enum, default_value = "inverse-noise")]
    weighting: WeightingArg,
    /// How registration resamples each frame: bilinear, or Lanczos-3 clamped
    /// against ringing, which is sharper and slower
    #[arg(long, value_enum, default_value = "lanczos3")]
    interpolation: InterpolationArg,
    /// How Bayer frames are demosaiced: VNG keeps stars' colour across their
    /// profile; MHC is sharper but rings around small stars; bilinear is
    /// fastest and softest
    #[arg(long, value_enum, default_value = "vng")]
    demosaic: DemosaicArg,
    /// Geometry fitted to each frame after the similarity match: similarity
    /// (shift, rotation, scale), affine, or quadratic, which follows lens
    /// distortion, for example after a meridian flip turns it against the
    /// sky. A frame with too few stars for its model keeps the similarity
    #[arg(long, value_enum, default_value = "quadratic")]
    registration_model: RegistrationModelArg,
    /// Maximum registration residual for additive admission
    #[arg(long, default_value_t = 2.0)]
    max_registration_rms: f64,
    /// Pixel floor for maximum frame-to-reference drift
    #[arg(
        long,
        default_value_t = RegistrationOptions::DEFAULT_MAXIMUM_DRIFT_PIXELS
    )]
    max_registration_drift: f64,
    /// Fraction of the larger image dimension allowed for registration drift
    #[arg(
        long,
        default_value_t = RegistrationOptions::DEFAULT_MAXIMUM_DRIFT_FRACTION
    )]
    max_registration_drift_fraction: f64,
    /// Minimum fraction of samples overlapping the reference
    #[arg(long, default_value_t = 0.60)]
    min_overlap: f32,
    /// Frames read and prepared at once. By default this follows
    /// --pipeline-memory-mib and the machine's cores
    #[arg(long, value_parser = clap::value_parser!(usize))]
    workers: Option<usize>,
    /// Memory for frames prepared ahead of integration, and for the bands of
    /// rows of every frame --reintegrate reads back at once, in MiB
    #[arg(long, default_value_t = 4096)]
    pipeline_memory_mib: usize,
}

#[derive(Serialize)]
struct CalibrationReport {
    bias: Option<FileIdentity>,
    dark: Option<FileIdentity>,
    flat: Option<FileIdentity>,
    dark_exposure_seconds: Option<f64>,
}

#[derive(Serialize)]
struct ConfigurationReport {
    registration_detection_sigma: f32,
    registration_maximum_stars: usize,
    registration_triangle_stars: usize,
    registration_descriptor_tolerance: f64,
    registration_scale_tolerance: f64,
    registration_match_tolerance_pixels: f64,
    registration_maximum_drift_floor_pixels: f64,
    registration_maximum_drift_fraction: f64,
    registration_effective_maximum_drift_pixels: f64,
    registration_minimum_matches: usize,
    registration_maximum_candidates: usize,
    normalization: &'static str,
    local_tile_size: usize,
    rejection: &'static str,
    sigma_low: f32,
    sigma_high: f32,
    rejection_warmup: u32,
    rejection_minimum_sigma: f32,
    reintegrate: bool,
    drizzle_scale: Option<u32>,
    drizzle_drop_shrink: Option<f32>,
    bayer_drizzle: bool,
    weighting: &'static str,
    interpolation: &'static str,
    demosaic: &'static str,
    registration_model: &'static str,
    maximum_registration_rms_pixels: f64,
    maximum_scale_deviation: f64,
    maximum_rotation_degrees: f64,
    minimum_overlap_fraction: f32,
    minimum_normalization_gain: f32,
    maximum_normalization_gain: f32,
    minimum_integrated_fraction: f32,
}

#[derive(Serialize)]
struct DiagnosticReport {
    matched_stars: usize,
    registration_rms_pixels: f64,
    registration_drift_pixels: f64,
    scale: f64,
    rotation_degrees: f64,
    translation_x: f64,
    translation_y: f64,
    normalization_mean_gain: f32,
    normalization_mean_offset: f32,
    overlap_fraction: f32,
    integrated_fraction: f32,
    accepted_samples: usize,
    rejected_samples: usize,
    /// Per-channel weight relative to the reference; absent when frames
    /// count equally.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    weight: Vec<f32>,
}

#[derive(Serialize)]
struct AdmissionRecord {
    source: FileIdentity,
    disposition: &'static str,
    reason: Option<String>,
    diagnostics: Option<DiagnosticReport>,
}

#[derive(Serialize)]
struct StackReport {
    schema_version: u32,
    output: FileIdentity,
    preview: Option<String>,
    reference: FileIdentity,
    calibration: CalibrationReport,
    configuration: ConfigurationReport,
    frames: Vec<AdmissionRecord>,
    accepted_frames: u32,
    rejected_frames: u32,
}

/// End a run stopped by Ctrl-C or SIGTERM once its scratch files are gone.
fn stop_interrupted() -> ! {
    eprintln!("interrupted: stopped and removed scratch files");
    std::process::exit(crate::interrupt::INTERRUPTED_EXIT_CODE);
}

/// The sensor temperature a frame recorded (CCD-TEMP, else SET-TEMP).
fn known_temperature(frame: &FitsFrame) -> Option<f64> {
    frame
        .metadata()
        .signature
        .camera_temp_c
        .filter(|value| value.is_finite())
}

pub(crate) fn run(options: StackArgs) -> Result<()> {
    let drizzle = options
        .drizzle
        .map(|scale| {
            let drizzle = seiza_stacking::DrizzleOptions {
                scale,
                drop_shrink: options.drizzle_drop_shrink,
            };
            drizzle.validate().map(|()| drizzle)
        })
        .transpose()?;
    // The drizzle follows reintegration's rejection.
    let reintegrate = options.reintegrate || drizzle.is_some();
    let report_path = options.report.clone();
    let preview_path = options.preview.clone();
    let mut path_roles = options
        .images
        .iter()
        .enumerate()
        .map(|(index, path)| (format!("light frame {}", index + 1), path.as_path()))
        .collect::<Vec<_>>();
    for (role, path) in [
        ("master bias", options.bias.as_deref()),
        ("master dark", options.dark.as_deref()),
        ("master flat", options.flat.as_deref()),
        ("stack output", Some(options.output.as_path())),
        ("preview output", options.preview.as_deref()),
        ("report output", options.report.as_deref()),
    ] {
        if let Some(path) = path {
            path_roles.push((role.into(), path));
        }
    }
    validate_path_roles(path_roles)?;
    if matches!(options.rejection, RejectionArg::DeltaSigma) {
        if !options.sigma_low.is_finite()
            || options.sigma_low <= 0.0
            || !options.sigma_high.is_finite()
            || options.sigma_high <= 0.0
        {
            anyhow::bail!("--sigma-low and --sigma-high must be positive finite numbers");
        }
        if options.rejection_warmup < 2 {
            anyhow::bail!("--rejection-warmup must be at least 2");
        }
    }
    if options
        .dark_exposure_seconds
        .is_some_and(|seconds| !seconds.is_finite() || seconds <= 0.0)
    {
        anyhow::bail!("--dark-exposure-seconds must be a positive finite number");
    }
    crate::common::validate_registration_flags(
        options.max_registration_rms,
        options.max_registration_drift,
        options.max_registration_drift_fraction,
    )?;
    if !options.min_overlap.is_finite() || !(0.0..=1.0).contains(&options.min_overlap) {
        anyhow::bail!("--min-overlap must be between zero and one");
    }
    if matches!(
        options.normalization,
        NormalizationArg::Local | NormalizationArg::LocalBackground
    ) && options.local_tile_size < 16
    {
        anyhow::bail!("--local-tile-size must be at least 16 pixels");
    }

    let load_master = |path: Option<&PathBuf>| -> Result<Option<FitsFrame>> {
        path.map(|path| crate::common::open_frame(path, "calibration master"))
            .transpose()
    };
    let mut calibration_report = report_path
        .as_ref()
        .map(|_| {
            Ok::<_, anyhow::Error>(CalibrationReport {
                bias: options.bias.as_deref().map(file_identity).transpose()?,
                dark: options.dark.as_deref().map(file_identity).transpose()?,
                flat: options.flat.as_deref().map(file_identity).transpose()?,
                dark_exposure_seconds: options.dark_exposure_seconds,
            })
        })
        .transpose()?;
    let bias = load_master(options.bias.as_ref())?;
    let dark = load_master(options.dark.as_ref())?;
    let flat = load_master(options.flat.as_ref())?;
    if let (Some(report), Some(dark)) = (&mut calibration_report, &dark) {
        report.dark_exposure_seconds = options.dark_exposure_seconds.or(dark.exposure_seconds);
    }
    let dark_records_temperature = dark.as_ref().map(|dark| known_temperature(dark).is_some());
    // From the decoded frames, not their pixels: each master's headers say
    // what it was shot at, and every light is checked against that.
    let calibration =
        CalibrationMasters::from_fits_frames(bias, dark, flat, options.dark_exposure_seconds)?;

    let normalization = match options.normalization {
        NormalizationArg::None => NormalizationMode::None,
        NormalizationArg::Global => NormalizationMode::Global,
        NormalizationArg::Local => NormalizationMode::Local {
            tile_size: options.local_tile_size,
        },
        NormalizationArg::LocalBackground => NormalizationMode::LocalBackground {
            tile_size: options.local_tile_size,
        },
    };
    let rejection = match options.rejection {
        RejectionArg::None => RejectionMode::None,
        RejectionArg::DeltaSigma => RejectionMode::DeltaSigma(DeltaSigmaOptions {
            low_sigma: options.sigma_low,
            high_sigma: options.sigma_high,
            warmup_samples: options.rejection_warmup,
            ..DeltaSigmaOptions::default()
        }),
    };
    let mut stack_options = StackOptions {
        normalization,
        rejection,
        demosaic: match options.demosaic {
            DemosaicArg::Vng => seiza_stacking::Demosaic::Vng,
            DemosaicArg::Mhc => seiza_stacking::Demosaic::Mhc,
            DemosaicArg::Bilinear => seiza_stacking::Demosaic::Bilinear,
        },
        interpolation: match options.interpolation {
            InterpolationArg::Bilinear => seiza_stacking::Interpolation::Bilinear,
            InterpolationArg::Lanczos3 => seiza_stacking::Interpolation::Lanczos3,
        },
        weighting: match options.weighting {
            WeightingArg::Equal => seiza_stacking::FrameWeighting::Equal,
            WeightingArg::InverseNoise => seiza_stacking::FrameWeighting::inverse_noise_variance(),
        },
        cfa_integration: if options.bayer_drizzle {
            seiza_stacking::CfaIntegration::BayerDrizzle
        } else {
            seiza_stacking::CfaIntegration::Demosaic
        },
        ..StackOptions::default()
    };
    stack_options.acceptance.maximum_registration_rms_pixels = options.max_registration_rms;
    stack_options.acceptance.minimum_overlap_fraction = options.min_overlap;
    stack_options.registration.maximum_drift_pixels = options.max_registration_drift;
    stack_options.registration.model = match options.registration_model {
        RegistrationModelArg::Similarity => seiza_stacking::RegistrationModel::Similarity,
        RegistrationModelArg::Affine => seiza_stacking::RegistrationModel::Affine,
        RegistrationModelArg::Quadratic => seiza_stacking::RegistrationModel::Quadratic,
    };
    stack_options.registration.maximum_drift_fraction = options.max_registration_drift_fraction;

    // The reference fixes the stack's grid and, under local background
    // normalization, the background every frame is matched to; the first
    // frame of a night is often low in the sky and in twilight.
    let mut ordered = options.images.clone();
    if matches!(options.reference, ReferenceArg::Auto) {
        let (best, scores) = seiza_stacking::choose_reference(&ordered, 8)?;
        let chosen = ordered.remove(best);
        if let Some(score) = &scores[best] {
            println!(
                "chose      {} as reference: {} stars, median area {:.1}px, sky variation {:.1}",
                chosen.display(),
                score.stars,
                score.median_star_area,
                score.background_variation,
            );
        }
        ordered.insert(0, chosen);
    }
    let mut images = ordered.iter();
    let reference_path = images.next().expect("clap requires at least two images");
    let reference_identity = report_path
        .as_ref()
        .map(|_| file_identity(reference_path))
        .transpose()?;
    let reference = crate::common::open_frame(reference_path, "reference frame")?;
    // A dark master that never recorded its temperature is used, since
    // nothing says it is wrong, but nothing says it is right either.
    if dark_records_temperature == Some(false)
        && let Some(temperature) = known_temperature(&reference)
    {
        eprintln!(
            "warning: the master dark records no sensor temperature, so it cannot be checked against the lights ({temperature}C in {})",
            reference_path.display()
        );
    }
    let effective_maximum_drift_pixels = stack_options
        .registration
        .effective_maximum_drift_pixels(reference.image.width, reference.image.height);

    let configuration_report = ConfigurationReport {
        registration_detection_sigma: stack_options.registration.detection_sigma,
        registration_maximum_stars: stack_options.registration.maximum_stars,
        registration_triangle_stars: stack_options.registration.triangle_stars,
        registration_descriptor_tolerance: stack_options.registration.descriptor_tolerance,
        registration_scale_tolerance: stack_options.registration.scale_tolerance,
        registration_match_tolerance_pixels: stack_options.registration.match_tolerance_pixels,
        registration_maximum_drift_floor_pixels: stack_options.registration.maximum_drift_pixels,
        registration_maximum_drift_fraction: stack_options.registration.maximum_drift_fraction,
        registration_effective_maximum_drift_pixels: effective_maximum_drift_pixels,
        registration_minimum_matches: stack_options.registration.minimum_matches,
        registration_maximum_candidates: stack_options.registration.maximum_candidates,
        normalization: match options.normalization {
            NormalizationArg::None => "none",
            NormalizationArg::Global => "global",
            NormalizationArg::Local => "local",
            NormalizationArg::LocalBackground => "local-background",
        },
        local_tile_size: options.local_tile_size,
        rejection: match options.rejection {
            RejectionArg::None => "none",
            RejectionArg::DeltaSigma => "delta-sigma",
        },
        sigma_low: options.sigma_low,
        sigma_high: options.sigma_high,
        rejection_warmup: options.rejection_warmup,
        reintegrate,
        drizzle_scale: options.drizzle,
        drizzle_drop_shrink: options.drizzle.and(options.drizzle_drop_shrink),
        bayer_drizzle: options.bayer_drizzle,
        registration_model: match options.registration_model {
            RegistrationModelArg::Similarity => "similarity",
            RegistrationModelArg::Affine => "affine",
            RegistrationModelArg::Quadratic => "quadratic",
        },
        weighting: match options.weighting {
            WeightingArg::Equal => "equal",
            WeightingArg::InverseNoise => "inverse-noise",
        },
        demosaic: match options.demosaic {
            DemosaicArg::Vng => "vng",
            DemosaicArg::Mhc => "mhc",
            DemosaicArg::Bilinear => "bilinear",
        },
        interpolation: match options.interpolation {
            InterpolationArg::Bilinear => "bilinear",
            InterpolationArg::Lanczos3 => "lanczos3",
        },
        rejection_minimum_sigma: match stack_options.rejection {
            RejectionMode::None => DeltaSigmaOptions::default().minimum_sigma,
            RejectionMode::DeltaSigma(options) => options.minimum_sigma,
        },
        maximum_registration_rms_pixels: options.max_registration_rms,
        maximum_scale_deviation: stack_options.acceptance.maximum_scale_deviation,
        maximum_rotation_degrees: stack_options.acceptance.maximum_rotation_degrees,
        minimum_overlap_fraction: options.min_overlap,
        minimum_normalization_gain: stack_options.acceptance.minimum_normalization_gain,
        maximum_normalization_gain: stack_options.acceptance.maximum_normalization_gain,
        minimum_integrated_fraction: stack_options.acceptance.minimum_integrated_fraction,
    };

    let mut stacker = seiza_stacking::LiveStacker::new(reference, calibration, stack_options)
        .with_context(|| {
            format!(
                "failed to initialize stack from {}",
                reference_path.display()
            )
        })?;
    // Each admitted frame's registered image waits here for reintegration:
    // about four bytes per output sample per frame, beside the output rather
    // than in a temporary directory that may live in memory. The live pass
    // writes it, so reintegration need not prepare the frame again.
    let scratch_directory = options.scratch_directory.clone().unwrap_or_else(|| {
        options
            .output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .map_or_else(|| PathBuf::from("."), PathBuf::from)
    });
    if reintegrate {
        // An interrupted run removes its scratch files, and a run left here
        // by a killed process has its files removed now.
        crate::interrupt::install();
        crate::interrupt::watch_scratch_parent(&scratch_directory);
    }
    if reintegrate
        && let Err(error) = stacker.retain_frames_for_reintegration(Some(&scratch_directory))
    {
        eprintln!("warning: {error}; reintegration will prepare each frame again");
    }
    println!(
        "reference  {} ({:.1}px registration drift limit)",
        reference_path.display(),
        effective_maximum_drift_pixels,
    );

    // Fingerprint every frame up front, on all cores, so the report costs the
    // stacking loop nothing.
    let paths = images.cloned().collect::<Vec<_>>();
    let mut source_identities = match &report_path {
        Some(_) => paths
            .par_iter()
            .map(|path| file_identity(path).map(Some))
            .collect::<Result<Vec<_>>>()?,
        None => paths.iter().map(|_| None).collect(),
    }
    .into_iter();
    let pipeline = PipelineOptions {
        workers: options.workers,
        ..PipelineOptions::with_budget(options.pipeline_memory_mib.saturating_mul(1024 * 1024))
    };
    let mut unreadable_frames = 0_u32;
    let mut admission_records = Vec::new();
    let _counted_above = stacker.push_fits_pipelined(&paths, &pipeline, |path, outcome| {
        let source_identity = source_identities.next().flatten();
        match outcome {
            Err(error) => {
                eprintln!("rejected   {}: {error}", path.display());
                unreadable_frames = unreadable_frames.saturating_add(1);
                if let Some(source) = source_identity {
                    admission_records.push(AdmissionRecord {
                        source,
                        disposition: "rejected",
                        reason: Some(error.to_string()),
                        diagnostics: None,
                    });
                }
            }
            Ok(FrameDisposition::Accepted(diagnostics)) => {
                println!(
                    "accepted   {}: {} stars, {:.3}px RMS, {:.1}px drift, {:+.3}deg, {:.1}% samples",
                    path.display(),
                    diagnostics.matched_stars,
                    diagnostics.registration_rms_pixels,
                    diagnostics.registration_drift_pixels,
                    diagnostics.transform.rotation_radians.to_degrees(),
                    diagnostics.integrated_fraction * 100.0,
                );
                if let Some(source) = source_identity {
                    admission_records.push(AdmissionRecord {
                        source,
                        disposition: "accepted",
                        reason: None,
                        diagnostics: Some(DiagnosticReport {
                            matched_stars: diagnostics.matched_stars,
                            registration_rms_pixels: diagnostics.registration_rms_pixels,
                            registration_drift_pixels: diagnostics.registration_drift_pixels,
                            scale: diagnostics.transform.scale,
                            rotation_degrees: diagnostics.transform.rotation_radians.to_degrees(),
                            translation_x: diagnostics.transform.translation_x,
                            translation_y: diagnostics.transform.translation_y,
                            normalization_mean_gain: diagnostics.normalization_mean_gain,
                            normalization_mean_offset: diagnostics.normalization_mean_offset,
                            overlap_fraction: diagnostics.overlap_fraction,
                            integrated_fraction: diagnostics.integrated_fraction,
                            accepted_samples: diagnostics.accepted_samples,
                            rejected_samples: diagnostics.rejected_samples,
                            weight: diagnostics.weight.clone(),
                        }),
                    });
                }
            }
            Ok(FrameDisposition::Rejected(reason)) => {
                eprintln!("rejected   {}: {reason}", path.display());
                if let Some(source) = source_identity {
                    admission_records.push(AdmissionRecord {
                        source,
                        disposition: "rejected",
                        reason: Some(reason.to_string()),
                        diagnostics: None,
                    });
                }
            }
        }
        if crate::interrupt::interrupted() {
            Continue::No
        } else {
            Continue::Yes
        }
    })?;
    if crate::interrupt::interrupted() {
        // Dropping the stacker removes its scratch directory.
        drop(stacker);
        stop_interrupted();
    }

    let reference_headers = stacker.reference_headers().to_vec();
    let mut drizzled = None;
    let mut snapshot = if reintegrate {
        if let Some(reason) = stacker.reintegration_unavailable() {
            anyhow::bail!("cannot reintegrate: {reason}");
        }
        let batch = seiza_stacking::BatchStackOptions {
            rejection: seiza_stacking::MasterRejectionOptions {
                low_sigma: options.sigma_low,
                high_sigma: options.sigma_high,
            },
            // For any frame the live pass could not keep.
            scratch_directory: Some(scratch_directory),
            cancel: Some(crate::interrupt::cancel_signal()),
            band_memory_bytes: options.pipeline_memory_mib.saturating_mul(1024 * 1024),
            ..seiza_stacking::BatchStackOptions::default()
        };
        let progress = |pass, index, count| {
            if index == 0 {
                let what = match pass {
                    seiza_stacking::BatchStackPass::Estimate => "estimating",
                    seiza_stacking::BatchStackPass::Refine => "refining",
                    seiza_stacking::BatchStackPass::Integrate if drizzle.is_some() => {
                        "integrating and drizzling"
                    }
                    seiza_stacking::BatchStackPass::Integrate => "integrating",
                };
                println!("reintegrate {what} {count} admitted frame(s)");
            }
        };
        let result =
            match &drizzle {
                Some(drizzle) => stacker.reintegrate_drizzled(&batch, drizzle, progress).map(
                    |(result, image)| {
                        drizzled = Some(image);
                        result
                    },
                ),
                None => stacker.reintegrate(&batch, progress),
            };
        if crate::interrupt::interrupted() {
            drop(stacker);
            stop_interrupted();
        }
        let result = result?;
        // Free the frames' scratch files before writing the outputs.
        drop(stacker);
        let rejected = result
            .snapshot
            .rejected_samples
            .iter()
            .map(|&count| u64::from(count))
            .sum::<u64>();
        println!("reintegrate rejected {rejected} sample(s)");
        result.snapshot
    } else {
        stacker.into_snapshot()?
    };
    snapshot.rejected_frames = snapshot.rejected_frames.saturating_add(unreadable_frames);
    match &drizzled {
        Some(drizzled) => {
            seiza_stacking::write_drizzle_fits_f32(
                &options.output,
                drizzled,
                snapshot.accepted_frames,
                snapshot.rejected_frames,
                &reference_headers,
            )?;
            crate::common::wrote(
                &options.output,
                format_args!(
                    "{} accepted frame(s), {} rejected frame(s), drizzled {}x ({}x{}), linear f32",
                    snapshot.accepted_frames,
                    snapshot.rejected_frames,
                    drizzled.scale,
                    drizzled.image.width,
                    drizzled.image.height,
                ),
            );
        }
        None => {
            seiza_stacking::write_fits_f32(&options.output, &snapshot, &reference_headers)?;
            crate::common::wrote(
                &options.output,
                format_args!(
                    "{} accepted frame(s), {} rejected frame(s), linear f32",
                    snapshot.accepted_frames, snapshot.rejected_frames,
                ),
            );
        }
    }
    if let Some(preview) = preview_path.as_ref() {
        let image = drizzled
            .as_ref()
            .map_or(&snapshot.image, |drizzled| &drizzled.image);
        write_preview(image, preview, PreviewTransfer::LinearLight)?;
        crate::common::wrote(
            preview,
            format_args!("display stretch only (not used by the stack)"),
        );
    }
    if let Some(report_path) = report_path {
        let report = StackReport {
            schema_version: 1,
            output: file_identity(&options.output)?,
            preview: preview_path.map(|path| path.display().to_string()),
            reference: reference_identity.expect("report identity was prepared"),
            calibration: calibration_report.expect("report calibration was prepared"),
            configuration: configuration_report,
            frames: admission_records,
            accepted_frames: snapshot.accepted_frames,
            rejected_frames: snapshot.rejected_frames,
        };
        write_json_atomic(&report_path, &report)?;
        crate::common::wrote(
            &report_path,
            format_args!("admission and provenance report"),
        );
    }
    Ok(())
}
