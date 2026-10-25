//! File and transfer validation around the shared emission blend functions.
use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};
use seiza_fits::{HeaderValue, WriteHeaderCard};
use seiza_stacking::{EmissionBlendLine, EmissionBlendOptions, FitsFrame};
use std::path::PathBuf;

#[derive(Args)]
pub(crate) struct BlendArgs {
    #[command(subcommand)]
    command: BlendCommand,
}
#[derive(Subcommand)]
enum BlendCommand {
    /// Add signed mono line residuals to linear RGB using exact plans
    Emission {
        #[arg(long)]
        broadband: PathBuf,
        #[arg(long, num_args=1..=3)]
        residual: Vec<PathBuf>,
        /// JSON object with lines and optional options (same as the C ABI)
        #[arg(long)]
        settings: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Mix prepared, bounded, starless display RGB before restoring stars
    Structure {
        #[arg(long)]
        narrowband: PathBuf,
        #[arg(long)]
        broadband: PathBuf,
        #[arg(long)]
        amount: f64,
        #[arg(long)]
        output: PathBuf,
    },
    /// Add normalized RGB emission after broadband calibration
    Contribution {
        #[arg(long)]
        base: PathBuf,
        #[arg(long)]
        contribution: PathBuf,
        #[arg(long, value_delimiter = ',', num_args = 1..=3, default_value = "1,1,1")]
        gains: Vec<f32>,
        #[arg(long)]
        output: PathBuf,
    },
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    lines: Vec<EmissionBlendLine>,
    #[serde(default)]
    options: EmissionBlendOptions,
}
fn open(path: &std::path::Path, transfer: &str) -> Result<FitsFrame> {
    let frame = FitsFrame::open(path)?;
    if frame.image.channels != 3 || !frame.headers.iter().any(|(key, value)| {
        key.eq_ignore_ascii_case("SEIZATRF")
            && matches!(value,HeaderValue::String(tag) if tag.trim().eq_ignore_ascii_case(transfer))
    }) {
        bail!("{} must be RGB with SEIZATRF={transfer}", path.display());
    }
    Ok(frame)
}
pub(crate) fn run(args: BlendArgs) -> Result<()> {
    let (base, image, output, transfer) = match args.command {
        BlendCommand::Emission {
            broadband,
            residual,
            settings,
            output,
        } => {
            let mut roles = vec![
                ("broadband".into(), broadband.as_path()),
                ("settings".into(), settings.as_path()),
                ("output".into(), output.as_path()),
            ];
            roles.extend(
                residual
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (format!("residual {i}"), p.as_path())),
            );
            crate::provenance::validate_path_roles(roles)?;
            let base = open(&broadband, "LINEAR")?;
            let config: Settings = serde_json::from_slice(&std::fs::read(&settings)?)
                .context("Invalid emission settings")?;
            let residuals = residual
                .iter()
                .map(FitsFrame::open)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            if residuals.iter().any(|frame| !frame.headers.iter().any(|(key,value)|key.eq_ignore_ascii_case("SEIZATRF")&&matches!(value,HeaderValue::String(tag) if tag.trim().eq_ignore_ascii_case("LINEAR")))) {bail!("Residuals require SEIZATRF=LINEAR");}
            let images: Vec<_> = residuals.iter().map(|f| &f.image).collect();
            let image = seiza_stacking::blend_emission_rgb(
                &base.image,
                &images,
                &config.lines,
                &config.options,
            )?;
            (base, image, output, "LINEAR")
        }
        BlendCommand::Structure {
            narrowband,
            broadband,
            amount,
            output,
        } => {
            crate::provenance::validate_path_roles([
                ("narrowband".into(), narrowband.as_path()),
                ("broadband".into(), broadband.as_path()),
                ("output".into(), output.as_path()),
            ])?;
            let base = open(&narrowband, "DISPLAY")?;
            let broadband = open(&broadband, "DISPLAY")?;
            let image =
                seiza_stacking::blend_display_structure(&base.image, &broadband.image, amount)?;
            (base, image, output, "DISPLAY")
        }
        BlendCommand::Contribution {
            base,
            contribution,
            gains,
            output,
        } => {
            crate::provenance::validate_path_roles([
                ("base".into(), base.as_path()),
                ("contribution".into(), contribution.as_path()),
                ("output".into(), output.as_path()),
            ])?;
            let base = open(&base, "LINEAR")?;
            let contribution = open(&contribution, "LINEAR")?;
            let image = seiza_stacking::add_rgb_contribution(
                &base.image,
                &contribution.image,
                gains
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("Supply three RGB gains"))?,
            )?;
            (base, image, output, "LINEAR")
        }
    };
    seiza_stacking::write_processed_image_fits_f32(
        output,
        &image,
        &base.headers,
        &[WriteHeaderCard::new(
            "SEIZATRF",
            HeaderValue::String(transfer.into()),
        )],
    )?;
    Ok(())
}
