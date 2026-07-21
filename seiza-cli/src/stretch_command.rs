use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use seiza_stacking::FitsFrame;
use seiza_stretch::{ColorStrategy, GhsParams, StretchConfig, StretchModel, StretchParams};
use std::path::PathBuf;

#[derive(Args)]
pub(crate) struct StretchArgs {
    /// Linear mono or RGB FITS or XISF input
    input: PathBuf,
    /// Display-referred PNG, JPEG, or TIFF output
    #[arg(short, long)]
    output: PathBuf,
    /// How RGB channels share statistics and transfer curves
    #[arg(long, value_enum, default_value = "linked")]
    color_strategy: ColorStrategyArg,
    /// Maximum pooled scalar samples retained by data-driven models
    #[arg(long, default_value_t = 200_000)]
    max_analysis_samples: usize,
    #[command(subcommand)]
    model: StretchModelArgs,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ColorStrategyArg {
    /// One distribution and curve shared by all channels
    Linked,
    /// Independent distribution and curve for each channel
    Unlinked,
    /// Stretch Rec.709 luminance and preserve RGB chromaticity
    LuminancePreserving,
}

impl From<ColorStrategyArg> for ColorStrategy {
    fn from(value: ColorStrategyArg) -> Self {
        match value {
            ColorStrategyArg::Linked => Self::Linked,
            ColorStrategyArg::Unlinked => Self::Unlinked,
            ColorStrategyArg::LuminancePreserving => Self::LuminancePreserving,
        }
    }
}

#[derive(Subcommand)]
enum StretchModelArgs {
    /// Clamp already display-referred samples to [0, 1]
    Identity,
    /// Affinely map an explicit black and white point
    Linear {
        #[arg(long)]
        black: f64,
        #[arg(long)]
        white: f64,
    },
    /// Apply asinh after an explicit black/white mapping
    Asinh {
        #[arg(long)]
        black: f64,
        #[arg(long)]
        white: f64,
        #[arg(long, default_value_t = 10.0)]
        strength: f64,
    },
    /// Resolve black/white percentiles and apply asinh
    PercentileAsinh {
        #[arg(long, default_value_t = 0.01)]
        black_percentile: f64,
        #[arg(long, default_value_t = 0.995)]
        white_percentile: f64,
        #[arg(long, default_value_t = 10.0)]
        strength: f64,
    },
    /// Apply explicit PixInsight/N.I.N.A.-family MTF parameters
    Mtf {
        #[arg(long)]
        shadows: f64,
        #[arg(long)]
        midtone: f64,
        #[arg(long)]
        highlights: f64,
    },
    /// Apply a manual Generalized Hyperbolic Stretch
    Ghs {
        /// Logarithmic stretch factor S, from 0 through 20
        #[arg(long, default_value_t = 1.0)]
        stretch_factor: f64,
        /// Local intensity b, from -5 through 15
        #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
        local_intensity: f64,
        /// Symmetry point SP, from 0 through 1
        #[arg(long, default_value_t = 0.0)]
        symmetry_point: f64,
        /// Shadow-protection boundary LP, no greater than SP
        #[arg(long, default_value_t = 0.0)]
        protect_shadows: f64,
        /// Highlight-protection boundary HP, no less than SP
        #[arg(long, default_value_t = 1.0)]
        protect_highlights: f64,
        /// Explicit input black point
        #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
        black: f64,
        /// Explicit input white point
        #[arg(long, default_value_t = 1.0, allow_negative_numbers = true)]
        white: f64,
    },
    /// Resolve the existing median/MAD Auto-MTF model
    AutoMtf {
        #[arg(long, default_value_t = 0.2)]
        target_median: f64,
        #[arg(long, default_value_t = -2.8, allow_negative_numbers = true)]
        shadows_clip: f64,
    },
}

impl StretchModelArgs {
    fn model(&self) -> StretchModel {
        match *self {
            Self::Identity => StretchModel::Identity,
            Self::Linear { black, white } => StretchModel::Linear { black, white },
            Self::Asinh {
                black,
                white,
                strength,
            } => StretchModel::Asinh {
                black,
                white,
                strength,
            },
            Self::PercentileAsinh {
                black_percentile,
                white_percentile,
                strength,
            } => StretchModel::PercentileAsinh {
                black_percentile,
                white_percentile,
                strength,
            },
            Self::Mtf {
                shadows,
                midtone,
                highlights,
            } => StretchModel::Mtf {
                shadows,
                midtone,
                highlights,
            },
            Self::Ghs {
                stretch_factor,
                local_intensity,
                symmetry_point,
                protect_shadows,
                protect_highlights,
                black,
                white,
            } => StretchModel::Ghs(GhsParams {
                stretch_factor,
                local_intensity,
                symmetry_point,
                protect_shadows,
                protect_highlights,
                black,
                white,
            }),
            Self::AutoMtf {
                target_median,
                shadows_clip,
            } => StretchModel::AutoMtf(StretchParams {
                target_median,
                shadows_clip,
            }),
        }
    }
}

pub(crate) fn run(args: StretchArgs) -> Result<()> {
    crate::provenance::validate_path_roles([
        ("stretch input".into(), args.input.as_path()),
        ("stretch output".into(), args.output.as_path()),
    ])?;
    let frame = FitsFrame::open(&args.input)
        .with_context(|| format!("could not read {}", args.input.display()))?;
    let config = StretchConfig {
        model: args.model.model(),
        color_strategy: args.color_strategy.into(),
        max_analysis_samples: args.max_analysis_samples,
    };
    let plan = config
        .resolve_for(&frame.image.data, frame.image.channels)
        .context("could not resolve stretch model")?;
    let pixels = plan
        .apply_u8(&frame.image.data, frame.image.channels)
        .context("could not apply stretch model")?;
    crate::preview::write_display_image(
        &args.output,
        frame.image.width,
        frame.image.height,
        frame.image.channels,
        pixels,
    )?;
    println!(
        "wrote {}: {:?}, {:?}, display-referred",
        args.output.display(),
        config.model,
        config.color_strategy
    );
    Ok(())
}
