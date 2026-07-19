use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use seiza_stacking::{
    CalibrationMasters, DeltaSigmaOptions, FitsFrame, FrameDisposition, MasterDark,
    NormalizationMode, RejectionMode, StackOptions,
};
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
    /// Integrated master bias FITS
    #[arg(long)]
    bias: Option<PathBuf>,
    /// Integrated master dark FITS
    #[arg(long)]
    dark: Option<PathBuf>,
    /// Integrated master flat FITS
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

pub(crate) fn run(options: StackArgs) -> Result<()> {
    let calibration_paths = [
        options.bias.as_ref(),
        options.dark.as_ref(),
        options.flat.as_ref(),
    ];
    if options.images.iter().any(|path| path == &options.output)
        || calibration_paths
            .iter()
            .flatten()
            .any(|path| *path == &options.output)
    {
        anyhow::bail!("--output must not overwrite an input or calibration frame");
    }
    if options.preview.as_ref().is_some_and(|preview| {
        preview == &options.output
            || options.images.contains(preview)
            || calibration_paths
                .iter()
                .flatten()
                .any(|path| *path == preview)
    }) {
        anyhow::bail!("--preview must not overwrite the stack or an input frame");
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
    let bias = load_master(options.bias.as_ref())?.map(|frame| frame.image);
    let dark = load_master(options.dark.as_ref())?.map(|frame| MasterDark {
        exposure_seconds: options.dark_exposure_seconds.or(frame.exposure_seconds),
        image: frame.image,
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

    let mut images = options.images.iter();
    let reference_path = images.next().expect("clap requires at least two images");
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
    for path in images {
        let frame = match FitsFrame::open(path) {
            Ok(frame) => frame,
            Err(error) => {
                eprintln!("rejected   {}: {error}", path.display());
                unreadable_frames = unreadable_frames.saturating_add(1);
                continue;
            }
        };
        match stacker.push(frame)? {
            FrameDisposition::Accepted(diagnostics) => println!(
                "accepted   {}: {} stars, {:.3}px RMS, {:+.3}deg, {:.1}% samples",
                path.display(),
                diagnostics.matched_stars,
                diagnostics.registration_rms_pixels,
                diagnostics.transform.rotation_radians.to_degrees(),
                diagnostics.integrated_fraction * 100.0,
            ),
            FrameDisposition::Rejected(reason) => {
                println!("rejected   {}: {reason}", path.display());
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
    if let Some(preview) = options.preview {
        write_preview(&snapshot.image, &preview)?;
        println!(
            "wrote {}: display stretch only (not used by the stack)",
            preview.display()
        );
    }
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
