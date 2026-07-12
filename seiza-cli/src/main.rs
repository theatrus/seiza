use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use seiza::{DetectConfig, detect_stars};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "seiza", about = "Star detection and plate solving", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Detect stars in an image and print their positions
    Detect {
        /// Image file (PNG, JPEG, TIFF)
        image: PathBuf,
        /// Detection threshold in noise sigmas
        #[arg(long, default_value_t = 4.0)]
        sigma: f32,
        /// Maximum number of stars to report
        #[arg(long, default_value_t = 100)]
        max_stars: usize,
        /// Write a copy of the image with detections circled
        #[arg(long)]
        annotate: Option<PathBuf>,
    },
    /// Solve an image against a star catalog (not implemented yet)
    Solve { image: PathBuf },
    /// Download prebuilt catalog data bundles (not implemented yet)
    DownloadData,
    /// Build catalog data bundles from primary sources (not implemented yet)
    BuildData,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Detect {
            image,
            sigma,
            max_stars,
            annotate,
        } => detect(&image, sigma, max_stars, annotate.as_deref()),
        Command::Solve { .. } => anyhow::bail!("solving is not implemented yet"),
        Command::DownloadData => anyhow::bail!("data download is not implemented yet"),
        Command::BuildData => anyhow::bail!("data building is not implemented yet"),
    }
}

fn detect(
    path: &std::path::Path,
    sigma: f32,
    max_stars: usize,
    annotate: Option<&std::path::Path>,
) -> Result<()> {
    let img = image::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let config = DetectConfig {
        sigma,
        max_stars,
        ..Default::default()
    };
    let stars = detect_stars(&img, &config);

    println!("{} stars detected in {}", stars.len(), path.display());
    println!(
        "{:>10} {:>10} {:>12} {:>8} {:>6}",
        "x", "y", "flux", "peak", "area"
    );
    for star in &stars {
        println!(
            "{:>10.2} {:>10.2} {:>12.1} {:>8.3} {:>6}",
            star.x, star.y, star.flux, star.peak, star.area
        );
    }

    if let Some(out) = annotate {
        let mut canvas = img.to_rgb8();
        for star in &stars {
            let radius = (star.area as f32).sqrt().max(6.0) as i32 + 4;
            imageproc::drawing::draw_hollow_circle_mut(
                &mut canvas,
                (star.x.round() as i32, star.y.round() as i32),
                radius,
                image::Rgb([64, 255, 64]),
            );
        }
        canvas
            .save(out)
            .with_context(|| format!("failed to write {}", out.display()))?;
        println!("annotated image written to {}", out.display());
    }

    Ok(())
}
