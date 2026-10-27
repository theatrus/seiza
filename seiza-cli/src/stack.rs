use crate::preview::{PreviewTransfer, write_preview};
use crate::provenance::{FileIdentity, file_identity, validate_path_roles, write_json_atomic};
use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use rayon::prelude::*;
use seiza_stacking::{
    CalibrationMasters, DeltaSigmaOptions, FitsFrame, FrameAcceptanceCriteria, FrameDisposition,
    NormalizationMode, RegistrationOptions, RejectionMode, StackOptions,
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
    /// Auto-select an anchor and expand to include all connected panels
    Expand,
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

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum DefectsArg {
    /// Only without --dark, whose master subtracts them
    Auto,
    /// Also with --dark, for the defects its master leaves
    On,
    Off,
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
    /// Which light supplies normalization and noise weighting: the clearest
    /// and sharpest, or the first given. Also sets alignment unless
    /// --reference-image supplies a common grid. Expand anchors on the
    /// clearest light and grows the canvas to hold every connected,
    /// overlapping frame, as for a mosaic
    #[arg(long, value_enum, default_value = "auto")]
    reference: ReferenceArg,
    /// Common FITS or XISF alignment reference for channel
    /// stacks. Prepared without the lights' calibration masters; contributes
    /// no pixels unless also listed as a light. Works with --reintegrate
    #[arg(long)]
    reference_image: Option<PathBuf>,
    /// Reuse an expanded canvas saved with --save-canvas for another filter
    #[arg(long, conflicts_with = "reference_image")]
    reference_canvas: Option<PathBuf>,
    /// Plan a union with additional filter groups from a JSON file. Each group
    /// has paths and optional bias, dark, flat, darkExposureSeconds; paths are
    /// relative to the current directory. This command's lights remain the
    /// only integrated filter. Use --reference-canvas instead to reuse a grid
    #[arg(long, conflicts_with_all = ["reference_canvas", "reference_image"])]
    joint_canvas: Option<PathBuf>,
    /// Save the planned expanded canvas for subsequent per-filter runs
    #[arg(long)]
    save_canvas: Option<PathBuf>,
    /// Largest canvas --reference expand may grow, in millions of samples:
    /// pixels times color channels. A run peaks at about 40 bytes per sample
    /// [default: 256]
    #[arg(long)]
    max_canvas_megasamples: Option<usize>,
    /// Linear 32-bit floating-point FITS stack
    #[arg(short, long)]
    output: PathBuf,
    /// Optional display-stretched PNG; never used by stacking math
    #[arg(long)]
    preview: Option<PathBuf>,
    /// Also write, as a mono 32-bit FITS on the stack's grid, how much of each
    /// pixel came from frames whose raw pixels clipped at the sensor's
    /// saturation level, from 0 to 1, for `color-calibrate --clipped`, which
    /// renders those pixels neutral rather than tinted. The level comes from
    /// SATURATE or SATLEVEL, or the frames' integer encoding
    #[arg(long)]
    clipped: Option<PathBuf>,
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
    /// Learn the sensor's hot and dead pixels from up to 16 of the lights
    /// and repair them in every light before debayering and registration.
    /// A defect stays on its pixel while the sky moves, so the lights must
    /// be dithered or have drifted
    #[arg(long, value_enum, default_value = "auto")]
    defects: DefectsArg,
    /// Background normalization applied after registration [default:
    /// local-background, or global with --reference expand]
    #[arg(long, value_enum)]
    normalization: Option<NormalizationArg>,
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
    /// sky. A frame with too few stars for its model keeps the similarity, as
    /// does the first frame of each panel under --reference expand
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
    /// Minimum fraction of samples overlapping the reference. With
    /// --reference expand, the fraction of the smaller footprint a frame must
    /// share with the canvas to join it; panels planned for 10% overlap land
    /// nearer 9% [default: 0.60, or 0.05 with --reference expand]
    #[arg(long)]
    min_overlap: Option<f32>,
    /// Frames read and prepared at once. By default this follows
    /// --pipeline-memory-mib and the threads RAYON_NUM_THREADS allows, or
    /// the machine's cores
    #[arg(long, value_parser = clap::value_parser!(usize))]
    workers: Option<usize>,
    /// Memory for frames prepared ahead of integration, and for the bands of
    /// rows of every frame --reintegrate reads back at once, in MiB
    #[arg(long, default_value_t = 4096)]
    pipeline_memory_mib: usize,
    #[command(flatten)]
    gpu: crate::common::GpuArgs,
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
    /// Sensor defects learned from the lights and repaired in each, and the
    /// lights they were learned from; present only when --defects looked.
    #[serde(skip_serializing_if = "Option::is_none")]
    defect_pixels: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    defect_frames: Option<u32>,
    /// Present only for --reference expand, so other reports are unchanged.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    expanded_canvas: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    maximum_canvas_samples: Option<usize>,
    /// The lens distortion the expanded canvas was planned through, when its
    /// panels' overlaps measured one.
    #[serde(skip_serializing_if = "Option::is_none")]
    expanded_lens: Option<seiza_stacking::ExpandedLens>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    expansion_planning_notes: Vec<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    normalization_reference: Option<FileIdentity>,
    reference_integrated: bool,
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

/// With `--reference auto` and `--reference-image`, how many of the best
/// lights to try in turn until one aligns to the shared image.
const CHANNEL_REFERENCE_ATTEMPTS: usize = 5;

/// The sensor temperature a frame recorded (CCD-TEMP, else SET-TEMP).
fn known_temperature(metadata: &seiza_stacking::FrameMetadata) -> Option<f64> {
    metadata
        .signature
        .camera_temp_c
        .filter(|value| value.is_finite())
}

/// The normalization a stack uses when the caller names none: global on an
/// expanded canvas, local-background otherwise.
///
/// On a mosaic local-background matches each frame's background to the
/// canvas under it, whose panels each keep their first frame's sky, and
/// its photometric gains scale the hazier, brighter-sky frames up, so the
/// stack keeps a large smooth gradient. On a 362-frame Seestar S50 mosaic,
/// global left a gradient two to three times smaller (3.0 / 2.4 / 3.4
/// sky-noise units against 8.1 / 6.2 / 5.1), a quarter to a half less lumpy
/// background, and the same seams, for 0.7% more noise. The library's own
/// default, which the C ABI and Python take, is global already.
fn normalization_or_default(chosen: Option<NormalizationArg>, expand: bool) -> NormalizationArg {
    chosen.unwrap_or(if expand {
        NormalizationArg::Global
    } else {
        NormalizationArg::LocalBackground
    })
}

pub(crate) fn run(options: StackArgs) -> Result<()> {
    let expand = matches!(options.reference, ReferenceArg::Expand)
        || options.reference_canvas.is_some()
        || options.joint_canvas.is_some();
    let normalization_arg = normalization_or_default(options.normalization, expand);
    let registration_model = options.registration_model;
    let min_overlap = options.min_overlap.unwrap_or(if expand {
        FrameAcceptanceCriteria::EXPANDED_MINIMUM_OVERLAP_FRACTION
    } else {
        FrameAcceptanceCriteria::default().minimum_overlap_fraction
    });
    if !expand && options.save_canvas.is_some() {
        anyhow::bail!(
            "--save-canvas requires --reference expand or --reference-canvas or --joint-canvas"
        );
    }
    if expand && options.reference_image.is_some() {
        anyhow::bail!("--reference expand cannot be combined with --reference-image");
    }
    if expand && (options.drizzle.is_some() || options.bayer_drizzle) {
        anyhow::bail!("--reference expand cannot be combined with --drizzle or --bayer-drizzle");
    }
    if !expand && options.max_canvas_megasamples.is_some() {
        anyhow::bail!("--max-canvas-megasamples applies only to --reference expand");
    }
    let maximum_samples = options
        .max_canvas_megasamples
        .unwrap_or(256)
        .checked_mul(1_000_000)
        .filter(|n| *n > 0)
        .context("--max-canvas-megasamples must be positive and fit in memory addressing")?;

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
        ("clipped map output", options.clipped.as_deref()),
        ("reference canvas", options.reference_canvas.as_deref()),
        ("joint canvas manifest", options.joint_canvas.as_deref()),
        ("saved canvas", options.save_canvas.as_deref()),
    ] {
        if let Some(path) = path {
            path_roles.push((role.into(), path));
        }
    }
    validate_path_roles(path_roles)?;
    let extra_channels: Vec<seiza_stacking::CanvasChannel> = options
        .joint_canvas
        .as_ref()
        .map(|path| -> Result<_> {
            let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
            serde_json::from_slice(&bytes).context("invalid joint canvas channel groups")
        })
        .transpose()?
        .unwrap_or_default();
    for path in extra_channels.iter().flat_map(|group| {
        group
            .paths
            .iter()
            .chain(group.bias.iter())
            .chain(group.dark.iter())
            .chain(group.flat.iter())
    }) {
        let mut roles = vec![("joint canvas input".into(), path.as_path())];
        for (role, output) in [
            ("stack output", Some(&options.output)),
            ("preview output", options.preview.as_ref()),
            ("report output", options.report.as_ref()),
            ("clipped map output", options.clipped.as_ref()),
            ("saved canvas", options.save_canvas.as_ref()),
        ] {
            if let Some(output) = output {
                roles.push((role.into(), output.as_path()));
            }
        }
        validate_path_roles(roles)?;
    }
    if let Some(reference) = options.reference_image.as_deref() {
        // Sharing the reference with a light is intentional; outputs and
        // calibration masters must still be distinct from it.
        let mut roles = vec![("reference image".into(), reference)];
        for (role, path) in [
            ("master bias", options.bias.as_deref()),
            ("master dark", options.dark.as_deref()),
            ("master flat", options.flat.as_deref()),
            ("stack output", Some(options.output.as_path())),
            ("preview output", options.preview.as_deref()),
            ("report output", options.report.as_deref()),
        ] {
            if let Some(path) = path {
                roles.push((role.into(), path));
            }
        }
        validate_path_roles(roles)?;
    }
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
    if !min_overlap.is_finite() || !(0.0..=1.0).contains(&min_overlap) {
        anyhow::bail!("--min-overlap must be between zero and one");
    }
    if matches!(
        normalization_arg,
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
    let dark_records_temperature = dark
        .as_ref()
        .map(|dark| known_temperature(&dark.metadata()).is_some());
    // From the decoded frames, not their pixels: each master's headers say
    // what it was shot at, and every light is checked against that.
    let calibration =
        CalibrationMasters::from_fits_frames(bias, dark, flat, options.dark_exposure_seconds)?;

    let normalization = match normalization_arg {
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
        gpu: options.gpu.options(),
        ..StackOptions::default()
    };
    stack_options.acceptance.maximum_registration_rms_pixels = options.max_registration_rms;
    stack_options.acceptance.minimum_overlap_fraction = min_overlap;
    stack_options.registration.maximum_drift_pixels = options.max_registration_drift;
    stack_options.registration.model = match registration_model {
        RegistrationModelArg::Similarity => seiza_stacking::RegistrationModel::Similarity,
        RegistrationModelArg::Affine => seiza_stacking::RegistrationModel::Affine,
        RegistrationModelArg::Quadratic => seiza_stacking::RegistrationModel::Quadratic,
    };
    stack_options.registration.maximum_drift_fraction = options.max_registration_drift_fraction;

    // Pick this channel's photometric target even when an external image
    // supplies the alignment grid. The first light of a night can be low
    // in the sky and in twilight.
    let mut ordered = options.images.clone();
    // The lights to try in turn as the channel reference of a stack on
    // `--reference-image`, best first.
    let mut channel_candidates = vec![ordered[0].clone()];
    if matches!(options.reference, ReferenceArg::Auto | ReferenceArg::Expand) {
        let (best, scores) = seiza_stacking::choose_reference(&ordered, 8)?;
        channel_candidates = seiza_stacking::rank_references(&scores)
            .into_iter()
            .take(CHANNEL_REFERENCE_ATTEMPTS)
            .map(|index| ordered[index].clone())
            .collect();
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
    // Without a master dark the lights keep their hot pixels: learn them
    // from the lights themselves, beneath which the sky moves.
    let mut defect_report = None;
    if options.defects == DefectsArg::On
        || (options.defects == DefectsArg::Auto && !calibration.has_dark())
    {
        let map = seiza_stacking::detect_defects(
            &ordered,
            &calibration,
            &seiza_stacking::DefectDetectionOptions {
                gpu: stack_options.gpu,
                ..seiza_stacking::DefectDetectionOptions::default()
            },
        )?;
        println!(
            "defects    {} sensor pixel(s) from {} light(s)",
            map.len(),
            map.frames()
        );
        defect_report = Some((map.len(), map.frames()));
        stack_options.defects = (!map.is_empty()).then_some(map);
    }
    let reference_path = options
        .reference_image
        .clone()
        .unwrap_or_else(|| ordered[0].clone());
    let reference_identity = report_path
        .as_ref()
        .map(|_| file_identity(options.reference_canvas.as_ref().unwrap_or(&reference_path)))
        .transpose()?;
    let reference = crate::common::open_frame(&reference_path, "reference frame")?;
    let effective_maximum_drift_pixels = stack_options
        .registration
        .effective_maximum_drift_pixels(reference.image.width, reference.image.height);

    let mut configuration_report = ConfigurationReport {
        defect_pixels: defect_report.map(|(pixels, _)| pixels),
        defect_frames: defect_report.map(|(_, frames)| frames),
        expanded_canvas: expand,
        maximum_canvas_samples: expand.then_some(maximum_samples),
        expanded_lens: None,
        expansion_planning_notes: Vec::new(),
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
        normalization: match normalization_arg {
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
        registration_model: match registration_model {
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
        minimum_overlap_fraction: min_overlap,
        minimum_normalization_gain: stack_options.acceptance.minimum_normalization_gain,
        maximum_normalization_gain: stack_options.acceptance.maximum_normalization_gain,
        minimum_integrated_fraction: stack_options.acceptance.minimum_integrated_fraction,
    };

    let initialize = || {
        format!(
            "failed to initialize stack from {}",
            reference_path.display()
        )
    };
    let mut stacker = if options.joint_canvas.is_some() {
        crate::interrupt::install();
        let mut channels = vec![seiza_stacking::CanvasChannel {
            paths: ordered.clone(),
            bias: options.bias.clone(),
            dark: options.dark.clone(),
            flat: options.flat.clone(),
            dark_exposure_seconds: options.dark_exposure_seconds,
        }];
        channels.extend(extra_channels);
        seiza_stacking::LiveStacker::open_fits_joint_canvas(
            &channels,
            0,
            stack_options,
            maximum_samples,
            |path, count, width, height| {
                println!(
                    "plan       {count} connected, {width}x{height}: {}",
                    path.display()
                );
                if crate::interrupt::interrupted() {
                    Continue::No
                } else {
                    Continue::Yes
                }
            },
        )
        .with_context(initialize)?
    } else if let Some(path) = &options.reference_canvas {
        crate::interrupt::install();
        let source = seiza_stacking::LiveStacker::open_context_with_gpu(path, stack_options.gpu)?;
        let view = source.view();
        if view.width * view.height * view.channels > maximum_samples {
            anyhow::bail!("shared canvas exceeds --max-canvas-megasamples");
        }
        source
            .open_fits_on_canvas(
                &ordered,
                options.bias.as_deref(),
                options.dark.as_deref(),
                options.flat.as_deref(),
                options.dark_exposure_seconds,
                stack_options,
                |path, count, width, height| {
                    println!(
                        "plan       {count}/{} connected, {width}x{height}: {}",
                        ordered.len(),
                        path.display()
                    );
                    if crate::interrupt::interrupted() {
                        Continue::No
                    } else {
                        Continue::Yes
                    }
                },
            )
            .with_context(initialize)?
    } else if expand {
        crate::interrupt::install();
        seiza_stacking::LiveStacker::with_expanding_reference(
            &ordered,
            calibration,
            stack_options,
            maximum_samples,
            |path, count, width, height| {
                println!(
                    "plan       {count}/{} connected, {width}x{height}: {}",
                    ordered.len(),
                    path.display()
                );
                if crate::interrupt::interrupted() {
                    Continue::No
                } else {
                    Continue::Yes
                }
            },
        )
        .with_context(initialize)?
    } else if options.reference_image.is_some() {
        let (stacker, used) = seiza_stacking::LiveStacker::with_reference_image_candidates(
            reference,
            channel_candidates
                .iter()
                .map(|path| crate::common::open_frame(path, "channel normalization reference")),
            calibration,
            stack_options,
            |index, reason| {
                eprintln!(
                    "warning: {} does not align to the reference image: {reason}",
                    channel_candidates[index].display()
                );
            },
        )
        .with_context(initialize)?;
        // The channel reference is pushed first, like an integrated reference.
        let used = &channel_candidates[used];
        let position = ordered
            .iter()
            .position(|path| path == used)
            .expect("channel candidates are listed lights");
        let chosen = ordered.remove(position);
        ordered.insert(0, chosen);
        println!("normalize  {} (channel reference)", ordered[0].display());
        stacker
    } else {
        seiza_stacking::LiveStacker::new(reference, calibration, stack_options)
            .with_context(initialize)?
    };
    let normalization_identity = options
        .reference_image
        .as_ref()
        .or(options.reference_canvas.as_ref())
        .and(report_path.as_ref())
        .map(|_| file_identity(&ordered[0]))
        .transpose()?;
    // A dark master that never recorded its temperature is used, since
    // nothing says it is wrong, but nothing says it is right either.
    if dark_records_temperature == Some(false)
        && let Some(temperature) = known_temperature(stacker.reference_metadata())
    {
        eprintln!(
            "warning: the master dark records no sensor temperature, so it cannot be checked against the lights ({temperature}C in {})",
            ordered[0].display()
        );
    }
    if expand {
        // Frames register through the panels they overlap; the canvas
        // diagonal bounds how far any of them lies from its center.
        let drift = stacker.effective_maximum_drift_pixels();
        configuration_report.registration_maximum_drift_floor_pixels = drift;
        configuration_report.registration_effective_maximum_drift_pixels = drift;
        // Sessions may differ in rotation; expansion admits any.
        configuration_report.maximum_rotation_degrees = 180.0;
        configuration_report.expanded_lens = stacker.expanded_lens();
        configuration_report.expansion_planning_notes = stacker.expansion_planning_notes().to_vec();
        for note in &configuration_report.expansion_planning_notes {
            eprintln!("note: {note}");
        }
        if configuration_report.expanded_lens.is_some() {
            eprintln!(
                "note: lens correction changes the pixel grid; original WCS is omitted. Plate-solve the output mosaic for astrometry."
            );
        }
    }
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
        configuration_report.registration_effective_maximum_drift_pixels,
    );

    // Fingerprint every frame up front, on all cores, so the report costs the
    // stacking loop nothing.
    // An integrated reference is already in the stack.
    let paths = ordered[usize::from(options.reference_image.is_none() && !expand)..].to_vec();
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

    if stacker.view().accepted_frames == 0 {
        anyhow::bail!("no light frames were accepted against the reference image");
    }

    if let Some(path) = &options.save_canvas {
        stacker.save_context(path)?;
    }
    let reference_headers = stacker.reference_headers().to_vec();
    // Read before reintegration drops the stacker: the live pass recorded it.
    let clipped = options.clipped.as_ref().map(|_| stacker.clipped_fraction());
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
    if let (Some(path), Some(clipped)) = (options.clipped.as_ref(), clipped) {
        match clipped {
            Some(map) => {
                seiza_stacking::write_linear_image_fits_f32(path, &map, &[], &[])?;
                crate::common::wrote(
                    path,
                    format_args!(
                        "clipped share of each pixel, {}x{}, {} pixel(s) at least half clipped",
                        map.width,
                        map.height,
                        map.data.iter().filter(|&&value| value >= 0.5).count()
                    ),
                );
            }
            None => println!(
                "no clipped map: no accepted frame clipped, or their saturation level is unknown"
            ),
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
            normalization_reference: normalization_identity,
            reference_integrated: options.reference_image.is_none() && !expand,
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
