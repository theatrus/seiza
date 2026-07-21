use anyhow::{Context, Result};
use clap::Args;
use seiza_deconvolution::{DeconvolutionConfig, deconvolve};
use seiza_fits::{HeaderValue, WriteHeaderCard};
use seiza_stacking::{FitsFrame, write_processed_image_fits_f32};
use std::path::PathBuf;

#[derive(Args)]
pub(crate) struct DeconvolutionArgs {
    /// Calibrated/stacked linear mono or RGB FITS or XISF input
    input: PathBuf,
    /// Restored linear 32-bit floating-point FITS output
    #[arg(short, long)]
    output: PathBuf,
    /// Measured unsaturated-star PSF FWHM, in pixels
    #[arg(long)]
    psf_fwhm: f32,
    /// Richardson-Lucy iterations (start with 3-5)
    #[arg(long, default_value_t = 4)]
    iterations: usize,
    /// Fraction of the restored estimate blended into the input
    #[arg(long, default_value_t = 0.35)]
    amount: f32,
    /// Residual damping floor as a fraction of each channel's range
    #[arg(long, default_value_t = 0.001)]
    noise_fraction: f32,
    /// Maximum multiplicative correction in one iteration
    #[arg(long, default_value_t = 2.0)]
    max_correction: f32,
}

pub(crate) fn run(args: DeconvolutionArgs) -> Result<()> {
    crate::provenance::validate_path_roles([
        ("deconvolution input".into(), args.input.as_path()),
        ("deconvolution output".into(), args.output.as_path()),
    ])?;
    let mut frame = FitsFrame::open(&args.input)
        .with_context(|| format!("could not read {}", args.input.display()))?;
    if frame.bayer.is_some() {
        anyhow::bail!(
            "deconvolution cannot operate on the raw Bayer mosaic in {}; debayer or stack it first",
            args.input.display()
        );
    }
    let config = DeconvolutionConfig {
        psf_fwhm_pixels: args.psf_fwhm,
        iterations: args.iterations,
        amount: args.amount,
        noise_fraction: args.noise_fraction,
        max_correction: args.max_correction,
    };
    let result = deconvolve(
        &frame.image.data,
        frame.image.width,
        frame.image.height,
        frame.image.channels,
        &config,
    )
    .context("could not deconvolve image")?;
    frame.image.data = result.data;

    let cards = operation_cards(&config);
    write_processed_image_fits_f32(&args.output, &frame.image, &frame.headers, &cards)
        .with_context(|| format!("could not write {}", args.output.display()))?;
    println!(
        "wrote {}: damped Richardson-Lucy, {:.3}px FWHM, {} iterations, {:.0}% blend",
        args.output.display(),
        config.psf_fwhm_pixels,
        config.iterations,
        config.amount * 100.0,
    );
    for (channel, diagnostics) in result.channels.iter().enumerate() {
        println!(
            "channel {}: peak {:.6} -> {:.6}, flux delta {:+.3e}",
            channel + 1,
            diagnostics.input_peak,
            diagnostics.output_peak,
            diagnostics.output_flux - diagnostics.input_flux,
        );
    }
    Ok(())
}

fn operation_cards(config: &DeconvolutionConfig) -> Vec<WriteHeaderCard> {
    vec![
        WriteHeaderCard::new("SEIZADC", HeaderValue::String("RL-GAUSS".into()))
            .with_comment("Seiza deconvolution method"),
        WriteHeaderCard::new("SEIZATRF", HeaderValue::String("LINEAR".into()))
            .with_comment("linear sample transfer"),
        WriteHeaderCard::new(
            "DCFWHM",
            HeaderValue::Float(f64::from(config.psf_fwhm_pixels)),
        )
        .with_comment("Gaussian PSF FWHM in pixels"),
        WriteHeaderCard::new(
            "DCITER",
            HeaderValue::Integer(i64::try_from(config.iterations).unwrap_or(i64::MAX)),
        )
        .with_comment("Richardson-Lucy iterations"),
        WriteHeaderCard::new("DCAMT", HeaderValue::Float(f64::from(config.amount)))
            .with_comment("restored estimate blend"),
        WriteHeaderCard::new(
            "DCNOISE",
            HeaderValue::Float(f64::from(config.noise_fraction)),
        )
        .with_comment("channel-relative damping floor"),
        WriteHeaderCard::new(
            "DCMAXCOR",
            HeaderValue::Float(f64::from(config.max_correction)),
        )
        .with_comment("per-iteration correction limit"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use seiza_fits::{F32ImageData, FitsImage, write_f32_image};

    #[test]
    fn command_restores_linear_fits_and_records_parameters() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("blurred.fits");
        let output = temp.path().join("restored.fits");
        let size = 31;
        let center = size / 2;
        let sigma = 2.8_f32 / 2.354_82;
        let mut pixels = vec![0.0; size * size];
        for y in 0..size {
            for x in 0..size {
                let radius_squared = ((x as isize - center as isize).pow(2)
                    + (y as isize - center as isize).pow(2))
                    as f32;
                pixels[y * size + x] = (-0.5 * radius_squared / sigma.powi(2)).exp();
            }
        }
        write_f32_image(&input, size, size, F32ImageData::Mono(&pixels), &[]).unwrap();

        run(DeconvolutionArgs {
            input,
            output: output.clone(),
            psf_fwhm: 2.8,
            iterations: 4,
            amount: 0.35,
            noise_fraction: 0.0,
            max_correction: 2.0,
        })
        .unwrap();

        let restored = FitsImage::open(&output).unwrap();
        let restored_pixels = match &restored.pixels {
            seiza_fits::Pixels::F32(pixels) => pixels,
            other => panic!("expected f32 output, got {other:?}"),
        };
        assert!(restored_pixels[center * size + center] > pixels[center * size + center]);
        assert_eq!(restored.header_str("SEIZADC"), Some("RL-GAUSS"));
        assert!((restored.header_f64("DCFWHM").unwrap() - 2.8).abs() < 1.0e-6);
        assert_eq!(restored.header_f64("DCITER"), Some(4.0));
    }
}
