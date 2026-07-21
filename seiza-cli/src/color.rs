use crate::preview::{PreviewTransfer, write_preview};
use crate::provenance::validate_path_roles;
use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use seiza_stacking::{
    ColorComposition, ColorNormalization, ColorOptions, ColorTransfer, FitsFrame, ForaxxOptions,
    LinearImage, NarrowbandPalette, Registrar, RegistrationOptions, combine_lrgb,
    combine_narrowband, combine_rgb, resample_to_reference, write_color_fits_f32,
};
use std::path::{Path, PathBuf};

#[derive(Args)]
pub(crate) struct ColorArgs {
    #[command(subcommand)]
    command: ColorCommand,
}

#[derive(Subcommand)]
enum ColorCommand {
    /// Combine red, green, and blue mono stacks
    Rgb(RgbArgs),
    /// Combine luminance, red, green, and blue mono stacks
    Lrgb(LrgbArgs),
    /// Map H-alpha, OIII, and optional SII stacks into a false-color palette
    Narrowband(NarrowbandArgs),
}

#[derive(Args)]
struct RgbArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long)]
    red: PathBuf,
    #[arg(long)]
    green: PathBuf,
    #[arg(long)]
    blue: PathBuf,
}

#[derive(Args)]
struct LrgbArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long)]
    luminance: PathBuf,
    #[arg(long)]
    red: PathBuf,
    #[arg(long)]
    green: PathBuf,
    #[arg(long)]
    blue: PathBuf,
    /// Fraction of output luminance supplied by the L stack
    #[arg(long, default_value_t = 1.0)]
    luminance_weight: f32,
}

#[derive(Args)]
struct NarrowbandArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long)]
    ha: PathBuf,
    #[arg(long)]
    oiii: PathBuf,
    /// Required by three-filter palettes; unnecessary for HOO/Foraxx-HOO
    #[arg(long)]
    sii: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "sho")]
    palette: PaletteArg,
    /// Display level assigned to each channel median before Foraxx composition
    #[arg(long, default_value_t = 0.2)]
    foraxx_target_median: f32,
    /// Normal-equivalent MADs below the median used as the Foraxx black point
    #[arg(long, default_value_t = -2.8, allow_negative_numbers = true)]
    foraxx_shadows_clip: f32,
}

#[derive(Args)]
struct CommonArgs {
    /// RGB 32-bit floating-point FITS output
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Quick-look PNG output
    #[arg(long)]
    preview: Option<PathBuf>,
    /// Per-channel scaling before composition
    #[arg(long, value_enum, default_value = "percentile")]
    normalization: ColorNormalizationArg,
    /// Robust black point in the range 0..1
    #[arg(long, default_value_t = 0.001)]
    black_percentile: f32,
    /// Robust white point in the range 0..1
    #[arg(long, default_value_t = 0.995)]
    white_percentile: f32,
    /// Maximum samples per channel used to estimate percentile levels
    #[arg(long, default_value_t = 1_000_000)]
    normalization_samples: usize,
    /// Trust that input channel stacks already share an identical pixel grid
    #[arg(long)]
    no_register: bool,
    /// Maximum RMS residual for cross-filter registration
    #[arg(long, default_value_t = 2.0)]
    max_registration_rms: f64,
    /// Pixel floor for maximum cross-filter drift
    #[arg(
        long,
        default_value_t = RegistrationOptions::DEFAULT_MAXIMUM_DRIFT_PIXELS
    )]
    max_registration_drift: f64,
    /// Fraction of the larger reference dimension allowed for cross-filter drift
    #[arg(
        long,
        default_value_t = RegistrationOptions::DEFAULT_MAXIMUM_DRIFT_FRACTION
    )]
    max_registration_drift_fraction: f64,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ColorNormalizationArg {
    None,
    Percentile,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PaletteArg {
    Sho,
    Soh,
    Hso,
    Hos,
    Osh,
    Ohs,
    Hoo,
    ForaxxSho,
    ForaxxHoo,
}

impl From<PaletteArg> for NarrowbandPalette {
    fn from(value: PaletteArg) -> Self {
        match value {
            PaletteArg::Sho => Self::Sho,
            PaletteArg::Soh => Self::Soh,
            PaletteArg::Hso => Self::Hso,
            PaletteArg::Hos => Self::Hos,
            PaletteArg::Osh => Self::Osh,
            PaletteArg::Ohs => Self::Ohs,
            PaletteArg::Hoo => Self::Hoo,
            PaletteArg::ForaxxSho => Self::ForaxxSho,
            PaletteArg::ForaxxHoo => Self::ForaxxHoo,
        }
    }
}

pub(crate) fn run(args: ColorArgs) -> Result<()> {
    match args.command {
        ColorCommand::Rgb(args) => run_rgb(args),
        ColorCommand::Lrgb(args) => run_lrgb(args),
        ColorCommand::Narrowband(args) => run_narrowband(args),
    }
}

fn run_rgb(args: RgbArgs) -> Result<()> {
    validate_paths(
        &args.common,
        &[
            ("red", &args.red),
            ("green", &args.green),
            ("blue", &args.blue),
        ],
    )?;
    let red = open(&args.red, "red")?;
    let green = open(&args.green, "green")?;
    let blue = open(&args.blue, "blue")?;
    let registrar = args.common.registrar(&red.image)?;
    let green = args
        .common
        .align(registrar.as_ref(), green.image, &red.image, "green")?;
    let blue = args
        .common
        .align(registrar.as_ref(), blue.image, &red.image, "blue")?;
    let composition = combine_rgb(&red.image, &green, &blue, &args.common.options())?;
    write_outputs(&args.common, &composition, &red.headers, "RGB")
}

fn run_lrgb(args: LrgbArgs) -> Result<()> {
    validate_paths(
        &args.common,
        &[
            ("luminance", &args.luminance),
            ("red", &args.red),
            ("green", &args.green),
            ("blue", &args.blue),
        ],
    )?;
    let luminance = open(&args.luminance, "luminance")?;
    let red = open(&args.red, "red")?;
    let green = open(&args.green, "green")?;
    let blue = open(&args.blue, "blue")?;
    let registrar = args.common.registrar(&luminance.image)?;
    let red = args
        .common
        .align(registrar.as_ref(), red.image, &luminance.image, "red")?;
    let green = args
        .common
        .align(registrar.as_ref(), green.image, &luminance.image, "green")?;
    let blue = args
        .common
        .align(registrar.as_ref(), blue.image, &luminance.image, "blue")?;
    let composition = combine_lrgb(
        &luminance.image,
        &red,
        &green,
        &blue,
        args.luminance_weight,
        &args.common.options(),
    )?;
    write_outputs(&args.common, &composition, &luminance.headers, "LRGB")
}

fn run_narrowband(args: NarrowbandArgs) -> Result<()> {
    let palette = NarrowbandPalette::from(args.palette);
    if palette.requires_sii() && args.sii.is_none() {
        anyhow::bail!("{} requires --sii", palette.name());
    }
    let sii_path = if palette.requires_sii() {
        args.sii.as_ref()
    } else {
        None
    };
    let mut inputs = vec![("H-alpha", &args.ha), ("OIII", &args.oiii)];
    if let Some(sii) = sii_path {
        inputs.push(("SII", sii));
    }
    validate_paths(&args.common, &inputs)?;
    let ha = open(&args.ha, "H-alpha")?;
    let oiii = open(&args.oiii, "OIII")?;
    let sii = sii_path
        .map(PathBuf::as_path)
        .map(|path| open(path, "SII"))
        .transpose()?;
    let registrar = args.common.registrar(&ha.image)?;
    let oiii = args
        .common
        .align(registrar.as_ref(), oiii.image, &ha.image, "OIII")?;
    let sii = sii
        .map(|frame| {
            args.common
                .align(registrar.as_ref(), frame.image, &ha.image, "SII")
        })
        .transpose()?;
    let composition = combine_narrowband(
        &ha.image,
        &oiii,
        sii.as_ref(),
        palette,
        &args.common.options(),
        &ForaxxOptions {
            target_median: args.foraxx_target_median,
            shadows_clip: args.foraxx_shadows_clip,
        },
    )?;
    write_outputs(&args.common, &composition, &ha.headers, palette.name())
}

impl CommonArgs {
    fn options(&self) -> ColorOptions {
        let normalization = match self.normalization {
            ColorNormalizationArg::None => ColorNormalization::None,
            ColorNormalizationArg::Percentile => ColorNormalization::Percentile {
                black_percentile: self.black_percentile,
                white_percentile: self.white_percentile,
                max_samples: self.normalization_samples,
            },
        };
        ColorOptions { normalization }
    }

    fn registrar(&self, reference: &LinearImage) -> Result<Option<Registrar>> {
        if self.no_register {
            return Ok(None);
        }
        if !self.max_registration_rms.is_finite() || self.max_registration_rms <= 0.0 {
            anyhow::bail!("--max-registration-rms must be a positive finite number");
        }
        let options = RegistrationOptions {
            maximum_drift_pixels: self.max_registration_drift,
            maximum_drift_fraction: self.max_registration_drift_fraction,
            ..RegistrationOptions::default()
        };
        Ok(Some(Registrar::new(reference, options)?))
    }

    fn align(
        &self,
        registrar: Option<&Registrar>,
        source: LinearImage,
        reference: &LinearImage,
        role: &str,
    ) -> Result<LinearImage> {
        let Some(registrar) = registrar else {
            return Ok(source);
        };
        let registration = registrar
            .register(&source)
            .with_context(|| format!("failed to register {role} to the reference channel"))?;
        if registration.rms_error_pixels > self.max_registration_rms {
            anyhow::bail!(
                "{role} registration RMS {:.3}px exceeds {:.3}px",
                registration.rms_error_pixels,
                self.max_registration_rms
            );
        }
        println!(
            "registered {role}: {:.3}px RMS, {:.1}px drift, {:.3}deg rotation, {} stars",
            registration.rms_error_pixels,
            registration.drift_pixels,
            registration.transform.rotation_radians.to_degrees(),
            registration.matched_stars,
        );
        Ok(resample_to_reference(
            &source,
            reference.width,
            reference.height,
            registration.transform,
        )?)
    }
}

fn open(path: &Path, role: &str) -> Result<FitsFrame> {
    FitsFrame::open(path).with_context(|| format!("failed to read {role} stack {}", path.display()))
}

fn validate_paths(common: &CommonArgs, inputs: &[(&str, &PathBuf)]) -> Result<()> {
    if common.output.is_none() && common.preview.is_none() {
        anyhow::bail!("pass --output, --preview, or both");
    }
    let mut roles = inputs
        .iter()
        .map(|(role, path)| ((*role).to_owned(), path.as_path()))
        .collect::<Vec<_>>();
    if let Some(output) = common.output.as_deref() {
        roles.push(("color FITS output".into(), output));
    }
    if let Some(preview) = common.preview.as_deref() {
        roles.push(("color preview output".into(), preview));
    }
    validate_path_roles(roles)
}

fn write_outputs(
    common: &CommonArgs,
    composition: &ColorComposition,
    reference_headers: &[(String, seiza_fits::HeaderValue)],
    label: &str,
) -> Result<()> {
    if let Some(output) = common.output.as_deref() {
        write_color_fits_f32(output, composition, reference_headers, label)?;
        println!(
            "wrote {}: {label} RGB f32 ({})",
            output.display(),
            composition.transfer.fits_name().to_ascii_lowercase()
        );
    }
    if let Some(preview) = common.preview.as_deref() {
        let transfer = match composition.transfer {
            ColorTransfer::LinearLight => PreviewTransfer::LinearLight,
            ColorTransfer::DisplayReferred => PreviewTransfer::DisplayReferred,
        };
        write_preview(&composition.image, preview, transfer)?;
        println!("wrote {}: {label} quick-look PNG", preview.display());
    }
    Ok(())
}
