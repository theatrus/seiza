use crate::provenance::{FileIdentity, file_identity, validate_path_roles, write_json_atomic};
use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use seiza_stacking::{ExperimentalFlatPreprocessingOptions, ImpulseFilterOptions};
use seiza_stacking::{
    FitsFrame, FlatStarMaskingOptions, FlatStarMaskingStatistics, MasterBuildOptions,
    MasterBuildStage, MasterDark, MasterFrame, MasterFrameKind, MasterProgress,
    MasterRejectionMethod, MasterRejectionOptions, build_master_from_fits, write_master_fits_f32,
};
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Args)]
pub(crate) struct MasterArgs {
    #[command(subcommand)]
    command: MasterCommand,
}

#[derive(Subcommand)]
enum MasterCommand {
    /// Integrate raw bias exposures
    Bias(CommonArgs),
    /// Integrate raw dark exposures, optionally after bias subtraction
    Dark(DarkArgs),
    /// Calibrate, normalize, and integrate raw flat exposures
    Flat(FlatArgs),
}

#[derive(Args)]
struct CommonArgs {
    /// Raw calibration FITS or XISF frames
    #[arg(required = true, num_args = 2..)]
    images: Vec<PathBuf>,
    /// Linear 32-bit floating-point master FITS
    #[arg(short, long)]
    output: PathBuf,
    /// JSON integration/provenance report with SHA-256 input identities
    #[arg(long)]
    report: Option<PathBuf>,
    /// Low sigma rejection threshold
    #[arg(long, default_value_t = 3.0)]
    sigma_low: f32,
    /// High sigma rejection threshold
    #[arg(long, default_value_t = 3.0)]
    sigma_high: f32,
}

#[derive(Args)]
struct DarkArgs {
    #[command(flatten)]
    common: CommonArgs,
    /// Master bias used to remove the bias pedestal before integration
    #[arg(long)]
    bias: Option<PathBuf>,
    /// Assert the common dark exposure when headers omit or misreport it
    #[arg(long)]
    exposure_seconds: Option<f64>,
}

#[derive(Args)]
struct FlatArgs {
    #[command(flatten)]
    common: CommonArgs,
    /// Master bias used to calibrate each flat before normalization
    #[arg(long)]
    bias: Option<PathBuf>,
    /// Master dark-flat used to calibrate each flat before normalization
    #[arg(long)]
    dark_flat: Option<PathBuf>,
    /// Override the dark-flat exposure time in seconds
    #[arg(long, requires = "dark_flat")]
    dark_flat_exposure_seconds: Option<f64>,
    /// Assert the flat exposure when headers omit or misreport it
    #[arg(long)]
    exposure_seconds: Option<f64>,
    /// Exclude native star and extended saturation footprints before flat integration
    #[arg(long)]
    star_mask: bool,
    /// Replace isolated anomalous master samples using the core default thresholds
    #[arg(long)]
    defect_suppression: bool,
    /// EXPERIMENTAL: replace positive spatial outliers in calibrated flat inputs
    #[arg(long)]
    experimental_spatial_replacement: bool,
    /// Experimental local MAD sigma threshold (default 10; 6 is more aggressive)
    #[arg(long, requires = "experimental_spatial_replacement")]
    spatial_sigma: Option<f32>,
    /// Experimental minimum fractional excess over local median (default 0.01)
    #[arg(long, requires = "experimental_spatial_replacement")]
    spatial_minimum_relative_excess: Option<f32>,
    /// Experimental local noise floor in calibrated ADU (default 1)
    #[arg(long, requires = "experimental_spatial_replacement")]
    spatial_noise_floor_adu: Option<f32>,
    /// Diagnostic FITS mask: 1 where any channel was corrected, 0 elsewhere
    #[arg(long, requires = "defect_suppression")]
    defect_mask: Option<PathBuf>,
    /// Minimum unmasked samples retained at every pixel after clipping
    #[arg(long, default_value_t = 2, requires = "star_mask")]
    minimum_clean_samples: usize,
    /// Raw-input saturation ceiling; otherwise use explicit metadata/integer encoding
    #[arg(long, requires = "star_mask")]
    saturation_level: Option<f32>,
}

#[derive(Serialize)]
struct MasterCalibrationReport {
    bias: Option<FileIdentity>,
    dark_flat: Option<FileIdentity>,
    dark_flat_exposure_seconds: Option<f64>,
}

#[derive(Serialize)]
struct MasterConfigurationReport {
    low_sigma: f32,
    high_sigma: f32,
    exposure_seconds_override: Option<f64>,
    integration: &'static str,
    rejection_method: &'static str,
    fallback_center: &'static str,
    rereads_inputs: bool,
    defect_suppression: Option<ImpulseFilterOptions>,
    experimental_flat_preprocessing: Option<ExperimentalFlatPreprocessingOptions>,
}

impl MasterConfigurationReport {
    fn from_master(master: &MasterFrame, options: &MasterBuildOptions) -> Self {
        Self {
            defect_suppression: options.defect_suppression,
            experimental_flat_preprocessing: options.experimental_flat_preprocessing,
            low_sigma: options.rejection.low_sigma,
            high_sigma: options.rejection.high_sigma,
            exposure_seconds_override: options.exposure_seconds,
            integration: integration_name(master.rejection_method),
            rejection_method: master.rejection_method.as_str(),
            fallback_center: if master.flat_star_masking.is_some() {
                "none; insufficient coverage fails"
            } else {
                fallback_center(master.kind)
            },
            rereads_inputs: master.kind != MasterFrameKind::Flat,
        }
    }
}

fn integration_name(method: MasterRejectionMethod) -> &'static str {
    match method {
        MasterRejectionMethod::None => "mean-without-rejection",
        MasterRejectionMethod::MedianMad => "median-mad-sigma-clipped-mean",
        MasterRejectionMethod::LeaveOneOut => "two-pass-leave-one-out-sigma-clipped-mean",
    }
}

fn fallback_center(kind: MasterFrameKind) -> &'static str {
    match kind {
        MasterFrameKind::Flat => "temporal median",
        MasterFrameKind::Bias | MasterFrameKind::Dark => "unclipped mean",
    }
}

#[derive(Serialize)]
struct MasterInputReport {
    source: FileIdentity,
    accepted_samples: u64,
    rejected_samples: u64,
    masked_samples: u64,
}

#[derive(Serialize)]
struct MasterSkippedInputReport {
    source: FileIdentity,
    reason: String,
}

#[derive(Serialize)]
struct MasterReport {
    schema_version: u32,
    kind: &'static str,
    output: FileIdentity,
    calibration: MasterCalibrationReport,
    configuration: MasterConfigurationReport,
    inputs: Vec<MasterInputReport>,
    skipped_inputs: Vec<MasterSkippedInputReport>,
    input_frames: usize,
    accepted_samples: u64,
    rejected_samples: u64,
    masked_samples: u64,
    flat_star_masking: Option<FlatStarMaskingStatistics>,
    fallback_pixels: u64,
    defect_pixels_replaced: u64,
    experimental_flat_samples_replaced: u64,
    defect_mask: Option<FileIdentity>,
    bias_subtracted: bool,
    dark_subtracted: bool,
    normalized: bool,
    output_exposure_seconds: Option<f64>,
}

pub(crate) fn run(args: MasterArgs) -> Result<()> {
    match args.command {
        MasterCommand::Bias(common) => build(
            common,
            MasterFrameKind::Bias,
            None,
            None,
            None,
            None,
            FlatProcessing::default(),
        ),
        MasterCommand::Dark(args) => build(
            args.common,
            MasterFrameKind::Dark,
            args.bias,
            None,
            None,
            args.exposure_seconds,
            FlatProcessing::default(),
        ),
        MasterCommand::Flat(args) => build(
            args.common,
            MasterFrameKind::Flat,
            args.bias,
            args.dark_flat,
            args.dark_flat_exposure_seconds,
            args.exposure_seconds,
            FlatProcessing {
                defect_suppression: args.defect_suppression.then(ImpulseFilterOptions::default),
                defect_mask: args.defect_mask,
                experimental_flat_preprocessing: args.experimental_spatial_replacement.then(|| {
                    let defaults = ExperimentalFlatPreprocessingOptions::default();
                    ExperimentalFlatPreprocessingOptions {
                        sigma: args.spatial_sigma.unwrap_or(defaults.sigma),
                        minimum_relative_excess: args
                            .spatial_minimum_relative_excess
                            .unwrap_or(defaults.minimum_relative_excess),
                        noise_floor_adu: args
                            .spatial_noise_floor_adu
                            .unwrap_or(defaults.noise_floor_adu),
                    }
                }),
                star_masking: args.star_mask.then_some(FlatStarMaskingOptions {
                    minimum_clean_samples: args.minimum_clean_samples,
                    saturation_level: args.saturation_level,
                    ..Default::default()
                }),
            },
        ),
    }
}

#[derive(Default)]
struct FlatProcessing {
    star_masking: Option<FlatStarMaskingOptions>,
    defect_suppression: Option<ImpulseFilterOptions>,
    experimental_flat_preprocessing: Option<ExperimentalFlatPreprocessingOptions>,
    defect_mask: Option<PathBuf>,
}

fn build(
    common: CommonArgs,
    kind: MasterFrameKind,
    bias_path: Option<PathBuf>,
    dark_path: Option<PathBuf>,
    dark_exposure_seconds: Option<f64>,
    exposure_seconds: Option<f64>,
    flat: FlatProcessing,
) -> Result<()> {
    validate_input_paths(
        &common,
        bias_path.as_deref(),
        dark_path.as_deref(),
        flat.defect_mask.as_deref(),
    )?;
    if let Some(path) = &flat.defect_mask {
        anyhow::ensure!(
            seiza_image_io::Format::from_path(path) == Some(seiza_image_io::Format::Fits),
            "--defect-mask must name a FITS file"
        );
    }
    let input_identities = common
        .report
        .as_ref()
        .map(|_| {
            common
                .images
                .iter()
                .map(|path| file_identity(path))
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let mut calibration_report = common
        .report
        .as_ref()
        .map(|_| {
            Ok::<_, anyhow::Error>(MasterCalibrationReport {
                bias: bias_path.as_deref().map(file_identity).transpose()?,
                dark_flat: dark_path.as_deref().map(file_identity).transpose()?,
                dark_flat_exposure_seconds: dark_exposure_seconds,
            })
        })
        .transpose()?;

    let bias = bias_path
        .as_deref()
        .map(load_bias)
        .transpose()?
        .map(|frame| frame.image);
    let dark = dark_path
        .as_deref()
        .map(|path| {
            let frame = crate::common::open_frame(path, "master dark-flat")?;
            MasterDark::from_fits_frame(frame, dark_exposure_seconds).map_err(anyhow::Error::from)
        })
        .transpose()?;
    if let Some(report) = &mut calibration_report {
        report.dark_flat_exposure_seconds = dark
            .as_ref()
            .and_then(|dark| dark.exposure_seconds)
            .or(dark_exposure_seconds);
    }
    let mut options = MasterBuildOptions {
        rejection: MasterRejectionOptions {
            low_sigma: common.sigma_low,
            high_sigma: common.sigma_high,
        },
        exposure_seconds,
        bias,
        dark,
        cancel: None,
        progress: Some(print_progress()),
        defect_suppression: flat.defect_suppression,
        experimental_flat_preprocessing: flat.experimental_flat_preprocessing,
        flat_star_masking: flat.star_masking,
        dark_level_screening: None,
    };
    if kind == MasterFrameKind::Flat && options.bias.is_none() && options.dark.is_none() {
        eprintln!(
            "warning: building an uncalibrated normalized flat; supply --bias or --dark-flat when available"
        );
    }
    println!(
        "building {} master from {} frame(s): {}",
        kind.as_str(),
        common.images.len(),
        if kind == MasterFrameKind::Flat {
            "calibration, normalization, and scratch-backed integration"
        } else {
            "two-pass integration"
        }
    );
    if let Some(preprocessing) = &options.experimental_flat_preprocessing {
        eprintln!(
            "experimental spatial replacement: sigma {}, minimum relative excess {}, noise floor {} ADU; may remove genuine pixel response",
            preprocessing.sigma,
            preprocessing.minimum_relative_excess,
            preprocessing.noise_floor_adu
        );
    }
    // Only mask requests need a pre-correction snapshot. Use the same core
    // filter once, at the same final-master stage, then compare actual writes.
    let filter = options.defect_suppression;
    if flat.defect_mask.is_some() {
        options.defect_suppression = None;
    }
    let mut master = build_master_from_fits(&common.images, kind, &options)?;
    options.defect_suppression = filter;
    let defect_mask = if flat.defect_mask.is_some() {
        let mut pixels = master.image.data.clone();
        master.defect_pixels_replaced = seiza_stacking::suppress_impulses(
            &mut master.image,
            master.bayer,
            options
                .defect_suppression
                .as_ref()
                .expect("mask requires suppression"),
        )? as u64;
        // Reuse the snapshot as the output mask. Compressing RGB forward
        // cannot overwrite a sample that a later coordinate still needs.
        let channels = master.image.channels;
        for (index, after) in master.image.data.chunks(channels).enumerate() {
            let offset = index * channels;
            let changed = pixels[offset..offset + channels]
                .iter()
                .zip(after)
                .any(|(a, b)| a.to_bits() != b.to_bits());
            pixels[index] = f32::from(changed);
        }
        pixels.truncate(master.image.width * master.image.height);
        Some(pixels)
    } else {
        None
    };
    println!(
        "Experimental input samples replaced: {}",
        master.experimental_flat_samples_replaced
    );
    println!(
        "Master defects corrected: {}",
        master.defect_pixels_replaced
    );
    if let Some(masking) = &master.flat_star_masking {
        println!(
            "star masking: {} excluded sample(s), {} affected pixel(s), retained coverage {}..{}",
            masking.masked_samples,
            masking.masked_pixels,
            masking.minimum_clean_samples,
            masking.maximum_clean_samples
        );
        if masking.low_coverage_samples > 0 {
            eprintln!(
                "warning: {} output sample(s) have fewer than three retained inputs; robust rejection is limited",
                masking.low_coverage_samples
            );
        }
        if masking.unknown_saturation_inputs > 0 {
            eprintln!(
                "warning: saturation ceiling is unknown for {} input(s); supply --saturation-level in raw units when known",
                masking.unknown_saturation_inputs
            );
        }
        if masking.unmasked_saturation_samples > 0 {
            eprintln!(
                "warning: {} isolated saturated sample(s) were not expanded into star masks; review detector defects",
                masking.unmasked_saturation_samples
            );
        }
    }
    write_master_fits_f32(&common.output, &master)?;
    if let (Some(path), Some(pixels)) = (&flat.defect_mask, &defect_mask) {
        let headers = master
            .reference_headers
            .iter()
            .filter(|(key, _)| key.eq_ignore_ascii_case("ROWORDER"))
            .map(|(key, value)| seiza_fits::WriteHeaderCard::new(key, value.clone()))
            .collect::<Vec<_>>();
        seiza_image_io::save_f32(
            path,
            master.image.width,
            master.image.height,
            seiza_fits::F32ImageData::Mono(pixels),
            &seiza_image_io::SaveOptions {
                headers: &headers,
                fits_top_down: false,
                ..Default::default()
            },
        )?;
        crate::common::wrote(path, format_args!("master correction mask"));
    }
    if master.fallback_pixels > 0 {
        eprintln!(
            "warning: rejection removed every sample at {} pixel(s); wrote their {}",
            master.fallback_pixels,
            fallback_center(master.kind)
        );
    }
    crate::common::wrote(
        &common.output,
        format_args!(
            "{} {} frame(s), {}, {} rejected sample(s), linear f32",
            master.input_frames,
            kind.as_str(),
            integration_name(master.rejection_method),
            master.rejected_samples
        ),
    );

    if let Some(report_path) = common.report {
        let identities = input_identities
            .expect("input identities were prepared for the report")
            .into_iter();
        let skipped_by_path = master
            .skipped_inputs
            .iter()
            .map(|skipped| (skipped.path.as_path(), skipped.reason.as_str()))
            .collect::<std::collections::HashMap<_, _>>();
        let mut statistics = master.input_statistics.iter();
        let mut inputs = Vec::with_capacity(master.input_frames);
        let mut skipped_inputs = Vec::with_capacity(master.skipped_inputs.len());
        for (path, source) in common.images.iter().zip(identities) {
            if let Some(reason) = skipped_by_path.get(path.as_path()) {
                skipped_inputs.push(MasterSkippedInputReport {
                    source,
                    reason: (*reason).to_owned(),
                });
                continue;
            }
            let statistics = statistics
                .next()
                .context("master report is missing statistics for an accepted input")?;
            inputs.push(MasterInputReport {
                source,
                accepted_samples: statistics.accepted_samples,
                rejected_samples: statistics.rejected_samples,
                masked_samples: statistics.masked_samples,
            });
        }
        anyhow::ensure!(
            statistics.next().is_none(),
            "master report has more statistics than accepted inputs"
        );
        let report = MasterReport {
            schema_version: 1,
            kind: kind.as_str(),
            output: file_identity(&common.output)?,
            calibration: calibration_report.expect("calibration report was prepared"),
            configuration: MasterConfigurationReport::from_master(&master, &options),
            inputs,
            skipped_inputs,
            input_frames: master.input_frames,
            accepted_samples: master.accepted_samples,
            rejected_samples: master.rejected_samples,
            masked_samples: master.masked_samples,
            flat_star_masking: master.flat_star_masking.clone(),
            fallback_pixels: master.fallback_pixels,
            defect_pixels_replaced: master.defect_pixels_replaced,
            experimental_flat_samples_replaced: master.experimental_flat_samples_replaced,
            defect_mask: flat.defect_mask.as_deref().map(file_identity).transpose()?,
            bias_subtracted: master.bias_subtracted,
            dark_subtracted: master.dark_subtracted,
            normalized: master.normalized,
            output_exposure_seconds: master.exposure_seconds,
        };
        write_json_atomic(&report_path, &report)?;
        crate::common::wrote(&report_path, format_args!("master provenance report"));
    }
    Ok(())
}

fn validate_input_paths(
    common: &CommonArgs,
    bias: Option<&Path>,
    dark: Option<&Path>,
    defect_mask: Option<&Path>,
) -> Result<()> {
    let mut path_roles = common
        .images
        .iter()
        .enumerate()
        .map(|(index, path)| (format!("calibration frame {}", index + 1), path.as_path()))
        .collect::<Vec<_>>();
    for (role, path) in [
        ("defect mask output", defect_mask),
        ("master bias", bias),
        ("master dark-flat", dark),
        ("master output", Some(common.output.as_path())),
        ("report output", common.report.as_deref()),
    ] {
        if let Some(path) = path {
            path_roles.push((role.into(), path));
        }
    }
    validate_path_roles(path_roles)
}

fn load_bias(path: &Path) -> Result<FitsFrame> {
    let frame = crate::common::open_frame(path, "master bias")?;
    frame.validate_master_kind("BIAS")?;
    Ok(frame)
}

/// Build progress on stderr. On a terminal one line per stage counts up in
/// place; in a pipe or log each stage writes one line when it ends, so the
/// log is not filled with carriage returns.
fn print_progress() -> MasterProgress {
    use std::io::IsTerminal;
    let terminal = std::io::stderr().is_terminal();
    MasterProgress::new(move |progress| {
        let what = match progress.stage {
            MasterBuildStage::Read => "reading",
            MasterBuildStage::Reread => "rereading kept darks",
            MasterBuildStage::Integrate => "integrating",
            MasterBuildStage::Combine => "combining tiles",
        };
        let unit = if progress.stage == MasterBuildStage::Combine {
            "tile"
        } else {
            "frame"
        };
        if progress.done < progress.total {
            if terminal {
                eprint!(
                    "\r{what}: {unit} {}/{}   ",
                    progress.done + 1,
                    progress.total
                );
            }
        } else if terminal {
            eprintln!("\r{what}: {} {unit}(s) done   ", progress.total);
        } else {
            eprintln!("{what}: {} {unit}(s) done", progress.total);
        }
    })
}
