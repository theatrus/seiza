use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use seiza_background::{BackgroundConfig, CorrectionMode, ModelConfig, fit_background};
use seiza_fits::{HeaderValue, WriteHeaderCard};
use seiza_stacking::{LinearImage, write_processed_image_fits_f32};
use std::path::PathBuf;

#[derive(Args)]
pub(crate) struct BackgroundArgs {
    /// Linear mono or RGB FITS or XISF input
    input: PathBuf,
    /// Corrected linear 32-bit floating-point FITS output
    #[arg(short, long)]
    output: PathBuf,
    /// Optional linear 32-bit floating-point FITS background model
    #[arg(long)]
    model_output: Option<PathBuf>,
    /// Optional JSON fit, sample, and rejection diagnostics
    #[arg(long)]
    diagnostics: Option<PathBuf>,
    /// Additive gradient subtraction or multiplicative field correction
    #[arg(long, value_enum, default_value = "subtract")]
    mode: CorrectionModeArg,
    /// Total degree of the fitted polynomial surface
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u8).range(0..=4))]
    degree: u8,
    /// Scale-independent regularization of non-constant coefficients
    #[arg(long, default_value_t = 1.0e-8)]
    ridge: f64,
    /// Deterministic sample seeds along the longest image axis
    #[arg(long, default_value_t = 12)]
    samples_per_axis: usize,
    /// Sample-window radius in pixels (default: image-size dependent)
    #[arg(long)]
    sample_radius: Option<usize>,
    /// Local low-background search moves per seed
    #[arg(long, default_value_t = 4)]
    search_steps: usize,
    /// Robust sigma threshold for rejecting locally noisy windows
    #[arg(long, default_value_t = 3.5)]
    sample_rejection_sigma: f64,
    /// Robust sigma threshold for rejecting fit residuals
    #[arg(long, default_value_t = 3.0)]
    fit_rejection_sigma: f64,
    /// Maximum residual rejection and refit passes
    #[arg(long, default_value_t = 3)]
    fit_rejection_iterations: usize,
    /// Fractional image border excluded from sampling
    #[arg(long, default_value_t = 0.03)]
    border_fraction: f64,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CorrectionModeArg {
    Subtract,
    Divide,
}

impl From<CorrectionModeArg> for CorrectionMode {
    fn from(value: CorrectionModeArg) -> Self {
        match value {
            CorrectionModeArg::Subtract => Self::Subtract,
            CorrectionModeArg::Divide => Self::Divide,
        }
    }
}

impl CorrectionModeArg {
    fn fits_name(self) -> &'static str {
        match self {
            Self::Subtract => "SUBTRACT",
            Self::Divide => "DIVIDE",
        }
    }
}

pub(crate) fn run(args: BackgroundArgs) -> Result<()> {
    let mut roles = vec![
        ("background input".into(), args.input.as_path()),
        ("background output".into(), args.output.as_path()),
    ];
    if let Some(path) = &args.model_output {
        roles.push(("background model output".into(), path.as_path()));
    }
    if let Some(path) = &args.diagnostics {
        roles.push(("background diagnostics".into(), path.as_path()));
    }
    crate::provenance::validate_path_roles(roles)?;

    let mut frame = crate::common::open_frame(&args.input, "background input")?;
    if frame.bayer.is_some() {
        anyhow::bail!(
            "background extraction does not mix raw Bayer subchannels; debayer or stack {} first",
            args.input.display()
        );
    }
    let config = BackgroundConfig {
        model: ModelConfig::Polynomial {
            degree: args.degree,
            ridge: args.ridge,
        },
        samples_per_axis: args.samples_per_axis,
        sample_radius: args.sample_radius,
        search_steps: args.search_steps,
        sample_rejection_sigma: args.sample_rejection_sigma,
        fit_rejection_sigma: args.fit_rejection_sigma,
        fit_rejection_iterations: args.fit_rejection_iterations,
        border_fraction: args.border_fraction,
    };
    let fit = fit_background(
        &frame.image.data,
        frame.image.width,
        frame.image.height,
        frame.image.channels,
        &config,
    )
    .context("could not fit background model")?;

    if let Some(path) = &args.model_output {
        let image = LinearImage::new(
            frame.image.width,
            frame.image.height,
            frame.image.channels,
            fit.render_model()
                .context("could not render background model")?,
        )?;
        let cards = operation_cards("MODEL", args.degree, &fit);
        write_processed_image_fits_f32(path, &image, &frame.headers, &cards)
            .with_context(|| format!("could not write {}", path.display()))?;
    }
    fit.correct_in_place(&mut frame.image.data, args.mode.into())
        .context("could not apply background correction")?;
    let cards = operation_cards(args.mode.fits_name(), args.degree, &fit);
    write_processed_image_fits_f32(&args.output, &frame.image, &frame.headers, &cards)
        .with_context(|| format!("could not write {}", args.output.display()))?;
    if let Some(path) = &args.diagnostics {
        crate::provenance::write_json_atomic(path, &fit)
            .with_context(|| format!("could not write {}", path.display()))?;
    }

    crate::common::wrote(
        &args.output,
        format_args!(
            "polynomial degree {}, {} of {} samples accepted, {} correction",
            args.degree,
            fit.diagnostics.accepted_samples,
            fit.diagnostics.candidate_samples,
            args.mode.fits_name().to_ascii_lowercase(),
        ),
    );
    Ok(())
}

fn operation_cards(
    operation: &str,
    degree: u8,
    fit: &seiza_background::BackgroundFit,
) -> Vec<WriteHeaderCard> {
    vec![
        WriteHeaderCard::new("SEIZABG", HeaderValue::String(operation.into()))
            .with_comment("Seiza background operation"),
        WriteHeaderCard::new("SEIZATRF", HeaderValue::String("LINEAR".into()))
            .with_comment("linear sample transfer"),
        WriteHeaderCard::new("BGMODEL", HeaderValue::String("POLY".into()))
            .with_comment("background surface family"),
        WriteHeaderCard::new("BGDEG", HeaderValue::Integer(i64::from(degree)))
            .with_comment("background polynomial degree"),
        WriteHeaderCard::new(
            "BGSAMP",
            HeaderValue::Integer(
                i64::try_from(fit.diagnostics.accepted_samples).unwrap_or(i64::MAX),
            ),
        )
        .with_comment("accepted background samples"),
    ]
}
