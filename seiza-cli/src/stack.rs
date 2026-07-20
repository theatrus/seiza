use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use seiza_stacking::{
    CalibrationMasters, DeltaSigmaOptions, FitsFrame, FrameDisposition, MasterDark,
    NormalizationMode, RejectionMode, StackOptions,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum NormalizationArg {
    None,
    Global,
    Local,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RejectionArg {
    None,
    DeltaSigma,
}

#[derive(Args)]
pub(crate) struct StackArgs {
    /// Light frames, in acquisition order; the first is the reference
    #[arg(required = true, num_args = 2..)]
    images: Vec<PathBuf>,
    /// Linear 32-bit floating-point FITS stack
    #[arg(short, long)]
    output: PathBuf,
    /// Optional display-stretched PNG; never used by stacking math
    #[arg(long)]
    preview: Option<PathBuf>,
    /// JSON admission/provenance report with SHA-256 input identities
    #[arg(long)]
    report: Option<PathBuf>,
    /// Integrated master bias FITS
    #[arg(long)]
    bias: Option<PathBuf>,
    /// Integrated master dark FITS
    #[arg(long)]
    dark: Option<PathBuf>,
    /// Integrated master flat FITS in the light frame's raw sampling
    #[arg(long)]
    flat: Option<PathBuf>,
    /// Override master-dark exposure time in seconds
    #[arg(long, requires = "dark")]
    dark_exposure_seconds: Option<f64>,
    /// Background normalization applied after registration
    #[arg(long, value_enum, default_value = "global")]
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
    /// Maximum registration residual for additive admission
    #[arg(long, default_value_t = 2.0)]
    max_registration_rms: f64,
    /// Minimum fraction of samples overlapping the reference
    #[arg(long, default_value_t = 0.60)]
    min_overlap: f32,
}

#[derive(Serialize)]
struct FileIdentity {
    path: String,
    bytes: u64,
    sha256: String,
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
    registration_minimum_matches: usize,
    registration_maximum_candidates: usize,
    normalization: &'static str,
    local_tile_size: usize,
    rejection: &'static str,
    sigma_low: f32,
    sigma_high: f32,
    rejection_warmup: u32,
    rejection_minimum_sigma: f32,
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

pub(crate) fn run(options: StackArgs) -> Result<()> {
    let report_path = options.report.clone();
    let preview_path = options.preview.clone();
    let calibration_paths = [
        options.bias.as_ref(),
        options.dark.as_ref(),
        options.flat.as_ref(),
    ];
    if options
        .images
        .iter()
        .any(|path| paths_refer_to_same_file(path, &options.output))
        || calibration_paths
            .iter()
            .flatten()
            .any(|path| paths_refer_to_same_file(path, &options.output))
    {
        anyhow::bail!("--output must not overwrite an input or calibration frame");
    }
    if options.preview.as_ref().is_some_and(|preview| {
        paths_refer_to_same_file(preview, &options.output)
            || options
                .images
                .iter()
                .any(|path| paths_refer_to_same_file(path, preview))
            || calibration_paths
                .iter()
                .flatten()
                .any(|path| paths_refer_to_same_file(path, preview))
    }) {
        anyhow::bail!("--preview must not overwrite the stack or an input frame");
    }
    if options.report.as_ref().is_some_and(|report| {
        paths_refer_to_same_file(report, &options.output)
            || options
                .preview
                .as_ref()
                .is_some_and(|preview| paths_refer_to_same_file(report, preview))
            || options
                .images
                .iter()
                .any(|path| paths_refer_to_same_file(path, report))
            || calibration_paths
                .iter()
                .flatten()
                .any(|path| paths_refer_to_same_file(path, report))
    }) {
        anyhow::bail!("--report must not overwrite the stack, preview, or an input frame");
    }
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
    if options
        .dark_exposure_seconds
        .is_some_and(|seconds| !seconds.is_finite() || seconds <= 0.0)
    {
        anyhow::bail!("--dark-exposure-seconds must be a positive finite number");
    }
    if !options.max_registration_rms.is_finite() || options.max_registration_rms <= 0.0 {
        anyhow::bail!("--max-registration-rms must be a positive finite number");
    }
    if !options.min_overlap.is_finite() || !(0.0..=1.0).contains(&options.min_overlap) {
        anyhow::bail!("--min-overlap must be between zero and one");
    }
    if matches!(options.normalization, NormalizationArg::Local) && options.local_tile_size < 16 {
        anyhow::bail!("--local-tile-size must be at least 16 pixels");
    }

    let load_master = |path: Option<&PathBuf>| -> Result<Option<FitsFrame>> {
        path.map(|path| {
            FitsFrame::open(path)
                .with_context(|| format!("failed to load calibration master {}", path.display()))
        })
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
    let bias = load_master(options.bias.as_ref())?.map(|frame| frame.image);
    let dark = load_master(options.dark.as_ref())?.map(|frame| {
        let exposure_seconds = options.dark_exposure_seconds.or(frame.exposure_seconds);
        if let Some(report) = &mut calibration_report {
            report.dark_exposure_seconds = exposure_seconds;
        }
        MasterDark {
            exposure_seconds,
            image: frame.image,
        }
    });
    let flat = load_master(options.flat.as_ref())?.map(|frame| frame.image);
    let calibration = CalibrationMasters::new(bias, dark, flat)?;

    let normalization = match options.normalization {
        NormalizationArg::None => NormalizationMode::None,
        NormalizationArg::Global => NormalizationMode::Global,
        NormalizationArg::Local => NormalizationMode::Local {
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
        ..StackOptions::default()
    };
    stack_options.acceptance.maximum_registration_rms_pixels = options.max_registration_rms;
    stack_options.acceptance.minimum_overlap_fraction = options.min_overlap;

    let configuration_report = ConfigurationReport {
        registration_detection_sigma: stack_options.registration.detection_sigma,
        registration_maximum_stars: stack_options.registration.maximum_stars,
        registration_triangle_stars: stack_options.registration.triangle_stars,
        registration_descriptor_tolerance: stack_options.registration.descriptor_tolerance,
        registration_scale_tolerance: stack_options.registration.scale_tolerance,
        registration_match_tolerance_pixels: stack_options.registration.match_tolerance_pixels,
        registration_minimum_matches: stack_options.registration.minimum_matches,
        registration_maximum_candidates: stack_options.registration.maximum_candidates,
        normalization: match options.normalization {
            NormalizationArg::None => "none",
            NormalizationArg::Global => "global",
            NormalizationArg::Local => "local",
        },
        local_tile_size: options.local_tile_size,
        rejection: match options.rejection {
            RejectionArg::None => "none",
            RejectionArg::DeltaSigma => "delta-sigma",
        },
        sigma_low: options.sigma_low,
        sigma_high: options.sigma_high,
        rejection_warmup: options.rejection_warmup,
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

    let mut images = options.images.iter();
    let reference_path = images.next().expect("clap requires at least two images");
    let reference_identity = report_path
        .as_ref()
        .map(|_| file_identity(reference_path))
        .transpose()?;
    let reference = FitsFrame::open(reference_path)
        .with_context(|| format!("failed to load reference {}", reference_path.display()))?;
    let mut stacker = seiza_stacking::LiveStacker::new(reference, calibration, stack_options)
        .with_context(|| {
            format!(
                "failed to initialize stack from {}",
                reference_path.display()
            )
        })?;
    println!("reference  {}", reference_path.display());

    let mut unreadable_frames = 0_u32;
    let mut admission_records = Vec::new();
    for path in images {
        let source_identity = report_path
            .as_ref()
            .map(|_| file_identity(path))
            .transpose()?;
        let frame = match FitsFrame::open(path) {
            Ok(frame) => frame,
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
                continue;
            }
        };
        match stacker.push(frame)? {
            FrameDisposition::Accepted(diagnostics) => {
                println!(
                    "accepted   {}: {} stars, {:.3}px RMS, {:+.3}deg, {:.1}% samples",
                    path.display(),
                    diagnostics.matched_stars,
                    diagnostics.registration_rms_pixels,
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
                        }),
                    });
                }
            }
            FrameDisposition::Rejected(reason) => {
                println!("rejected   {}: {reason}", path.display());
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
    }

    let reference_headers = stacker.reference_headers().to_vec();
    let mut snapshot = stacker.into_snapshot()?;
    snapshot.rejected_frames = snapshot.rejected_frames.saturating_add(unreadable_frames);
    seiza_stacking::write_fits_f32(&options.output, &snapshot, &reference_headers)?;
    println!(
        "wrote {}: {} accepted frame(s), {} rejected frame(s), linear f32",
        options.output.display(),
        snapshot.accepted_frames,
        snapshot.rejected_frames,
    );
    if let Some(preview) = preview_path.as_ref() {
        write_preview(&snapshot.image, preview)?;
        println!(
            "wrote {}: display stretch only (not used by the stack)",
            preview.display()
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
        println!(
            "wrote {}: admission and provenance report",
            report_path.display()
        );
    }
    Ok(())
}

fn paths_refer_to_same_file(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }
    match (comparable_path(left), comparable_path(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn comparable_path(path: &Path) -> std::io::Result<PathBuf> {
    if let Ok(path) = std::fs::canonicalize(path) {
        return Ok(path);
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = std::fs::canonicalize(parent)?;
    Ok(parent.join(path.file_name().unwrap_or_default()))
}

fn file_identity(path: &Path) -> Result<FileIdentity> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("failed to fingerprint {}", path.display()))?;
    let bytes = file.metadata()?.len();
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    Ok(FileIdentity {
        path: path.display().to_string(),
        bytes,
        sha256: digest.iter().map(|byte| format!("{byte:02x}")).collect(),
    })
}

fn write_json_atomic(path: &Path, report: &StackReport) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let prefix = format!(
        ".{}.",
        path.file_name().unwrap_or_default().to_string_lossy()
    );
    let mut builder = tempfile::Builder::new();
    builder.prefix(&prefix);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let permissions = std::fs::metadata(path)
            .map(|metadata| metadata.permissions())
            .unwrap_or_else(|_| std::fs::Permissions::from_mode(0o666));
        builder.permissions(permissions);
    }
    let mut temporary = builder
        .tempfile_in(parent)
        .with_context(|| format!("failed to create report beside {}", path.display()))?;
    serde_json::to_writer_pretty(temporary.as_file_mut(), report)?;
    temporary.as_file_mut().write_all(b"\n")?;
    temporary.as_file_mut().flush()?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("failed to publish report {}", path.display()))?;
    Ok(())
}

fn write_preview(image: &seiza_stacking::LinearImage, path: &Path) -> Result<()> {
    let stride = (image.data.len() / 200_000).max(1);
    let mut sample = image
        .data
        .iter()
        .step_by(stride)
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if sample.is_empty() {
        anyhow::bail!("stack has no finite samples to preview");
    }
    sample.sort_unstable_by(f32::total_cmp);
    let black = sample[sample.len() / 100];
    let white = sample[sample.len() * 995 / 1000].max(black + f32::EPSILON);
    let stretch = |value: f32| {
        if !value.is_finite() {
            return 0;
        }
        let linear = ((value - black) / (white - black)).max(0.0);
        let display = (10.0 * linear).asinh() / 10.0_f32.asinh();
        (display.clamp(0.0, 1.0) * 255.0).round() as u8
    };
    if image.channels == 1 {
        let pixels = image.data.iter().copied().map(stretch).collect();
        image::GrayImage::from_raw(image.width as u32, image.height as u32, pixels)
            .ok_or_else(|| anyhow::anyhow!("preview dimension mismatch"))?
            .save(path)?;
    } else {
        let pixels = image.data.iter().copied().map(stretch).collect();
        image::RgbImage::from_raw(image.width as u32, image.height as u32, pixels)
            .ok_or_else(|| anyhow::anyhow!("preview dimension mismatch"))?
            .save(path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_collision_detection_resolves_parent_components() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir(directory.path().join("child")).unwrap();
        let direct = directory.path().join("stack.fits");
        let aliased = directory.path().join("child/../stack.fits");
        assert!(paths_refer_to_same_file(&direct, &aliased));
    }
}
