use crate::provenance::{FileIdentity, file_identity, validate_path_roles, write_json_atomic};
use anyhow::Result;
use clap::{Args, Subcommand};
use seiza_stacking::{
    FitsFrame, MasterBuildOptions, MasterDark, MasterFrameKind, MasterRejectionOptions,
    build_master_from_fits, write_master_fits_f32,
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
    /// Low leave-one-out sigma rejection threshold
    #[arg(long, default_value_t = 3.0)]
    sigma_low: f32,
    /// High leave-one-out sigma rejection threshold
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
    rereads_inputs: bool,
}

#[derive(Serialize)]
struct MasterInputReport {
    source: FileIdentity,
    accepted_samples: u64,
    rejected_samples: u64,
}

#[derive(Serialize)]
struct MasterReport {
    schema_version: u32,
    kind: &'static str,
    output: FileIdentity,
    calibration: MasterCalibrationReport,
    configuration: MasterConfigurationReport,
    inputs: Vec<MasterInputReport>,
    input_frames: usize,
    accepted_samples: u64,
    rejected_samples: u64,
    fallback_pixels: u64,
    bias_subtracted: bool,
    dark_subtracted: bool,
    normalized: bool,
    output_exposure_seconds: Option<f64>,
}

pub(crate) fn run(args: MasterArgs) -> Result<()> {
    match args.command {
        MasterCommand::Bias(common) => build(common, MasterFrameKind::Bias, None, None, None, None),
        MasterCommand::Dark(args) => build(
            args.common,
            MasterFrameKind::Dark,
            args.bias,
            None,
            None,
            args.exposure_seconds,
        ),
        MasterCommand::Flat(args) => build(
            args.common,
            MasterFrameKind::Flat,
            args.bias,
            args.dark_flat,
            args.dark_flat_exposure_seconds,
            args.exposure_seconds,
        ),
    }
}

fn build(
    common: CommonArgs,
    kind: MasterFrameKind,
    bias_path: Option<PathBuf>,
    dark_path: Option<PathBuf>,
    dark_exposure_seconds: Option<f64>,
    exposure_seconds: Option<f64>,
) -> Result<()> {
    validate_input_paths(&common, bias_path.as_deref(), dark_path.as_deref())?;
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
    let options = MasterBuildOptions {
        rejection: MasterRejectionOptions {
            low_sigma: common.sigma_low,
            high_sigma: common.sigma_high,
        },
        exposure_seconds,
        bias,
        dark,
    };
    if kind == MasterFrameKind::Flat && options.bias.is_none() && options.dark.is_none() {
        eprintln!(
            "warning: building an uncalibrated normalized flat; supply --bias or --dark-flat when available"
        );
    }
    println!(
        "building {} master from {} frame(s): two-pass sigma-clipped mean",
        kind.as_str(),
        common.images.len()
    );
    let master = build_master_from_fits(&common.images, kind, &options)?;
    write_master_fits_f32(&common.output, &master)?;
    if master.fallback_pixels > 0 {
        eprintln!(
            "warning: rejection removed every sample at {} pixel(s); wrote their unclipped mean",
            master.fallback_pixels
        );
    }
    crate::common::wrote(
        &common.output,
        format_args!(
            "{} {} frame(s), {} rejected sample(s), linear f32",
            master.input_frames,
            kind.as_str(),
            master.rejected_samples
        ),
    );

    if let Some(report_path) = common.report {
        let inputs = input_identities
            .expect("input identities were prepared for the report")
            .into_iter()
            .zip(&master.input_statistics)
            .map(|(source, statistics)| MasterInputReport {
                source,
                accepted_samples: statistics.accepted_samples,
                rejected_samples: statistics.rejected_samples,
            })
            .collect();
        let report = MasterReport {
            schema_version: 1,
            kind: kind.as_str(),
            output: file_identity(&common.output)?,
            calibration: calibration_report.expect("calibration report was prepared"),
            configuration: MasterConfigurationReport {
                low_sigma: common.sigma_low,
                high_sigma: common.sigma_high,
                exposure_seconds_override: exposure_seconds,
                integration: "two-pass-leave-one-out-sigma-clipped-mean",
                rereads_inputs: true,
            },
            inputs,
            input_frames: master.input_frames,
            accepted_samples: master.accepted_samples,
            rejected_samples: master.rejected_samples,
            fallback_pixels: master.fallback_pixels,
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
) -> Result<()> {
    let mut path_roles = common
        .images
        .iter()
        .enumerate()
        .map(|(index, path)| (format!("calibration frame {}", index + 1), path.as_path()))
        .collect::<Vec<_>>();
    for (role, path) in [
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
