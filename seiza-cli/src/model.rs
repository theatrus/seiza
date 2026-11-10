//! `seiza model`: run and manage learned models.

use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};
use seiza_fits::{HeaderValue, WriteHeaderCard};
use seiza_model::{
    Backend, BackendOptions, DenoiseOptions, DenoiseReport, ImageView, ModelStore, NoiseBands,
    NoiseModel, Precision, Progress, runtime_info,
};
use seiza_stacking::write_processed_image_fits_f32;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Args)]
pub(crate) struct ModelArgs {
    #[command(subcommand)]
    command: ModelCommand,
}

#[derive(Subcommand)]
enum ModelCommand {
    /// Take the noise out of a linear FITS image, keeping a share of it
    Denoise(Box<DenoiseArgs>),
    /// Show which backends run here, and why the others do not
    Backends(RuntimeArgs),
    /// List the installed models
    List,
    /// Install a published model by name, or a bundle from a directory, once
    /// its hashes match
    Install {
        /// A published model's name, such as noise-v1-d33, which is
        /// downloaded from downloads.seiza.fyi; or a directory holding a
        /// bundle's card.json and graphs
        #[arg(value_name = "NAME|PATH")]
        model: String,
    },
    /// Check installed models against their cards
    Verify {
        /// Only this model (default: all of them)
        name: Option<String>,
    },
    /// Remove an installed model
    Remove {
        /// The model's name
        name: String,
    },
}

#[derive(Args)]
struct RuntimeArgs {
    /// ONNX Runtime's library, in place of searching for it
    #[arg(long, value_name = "PATH")]
    onnxruntime: Option<PathBuf>,
}

#[derive(Args)]
struct DenoiseArgs {
    /// Calibrated, stacked linear mono or RGB FITS or XISF input, from any
    /// stacker
    input: PathBuf,
    /// Denoised linear 32-bit floating-point FITS output
    #[arg(short, long)]
    output: PathBuf,
    /// An installed model's name, or a bundle's directory (default: the one
    /// installed noise model)
    #[arg(long, value_name = "NAME|PATH")]
    model: Option<String>,
    /// What runs the model (default: the best this machine has)
    #[arg(long)]
    backend: Option<Backend>,
    /// A tile's side in pixels (default: the backend's); it does not change
    /// the result
    #[arg(long)]
    tile: Option<usize>,
    /// The graph's precision (default: the backend's)
    #[arg(long)]
    precision: Option<Precision>,
    /// The share of the pixel-scale grain the model finds that is taken out
    #[arg(long, default_value_t = 0.7)]
    high_amount: f32,
    /// The share of the wider mottle the model finds that is taken out
    #[arg(long, default_value_t = 0.9)]
    low_amount: f32,
    /// The Gaussian scale in pixels dividing the grain from the mottle
    #[arg(long, default_value_t = 5.0)]
    band_scale: f32,
    #[command(flatten)]
    runtime: RuntimeArgs,
}

pub(crate) fn run(args: ModelArgs) -> Result<()> {
    match args.command {
        ModelCommand::Denoise(args) => denoise(*args),
        ModelCommand::Backends(args) => backends(&args),
        ModelCommand::List => list(),
        ModelCommand::Install { model } => install(&model),
        ModelCommand::Verify { name } => verify(name.as_deref()),
        ModelCommand::Remove { name } => remove(&name),
    }
}

fn denoise(args: DenoiseArgs) -> Result<()> {
    crate::provenance::validate_path_roles([
        ("model input".into(), args.input.as_path()),
        ("model output".into(), args.output.as_path()),
    ])?;
    let bands = NoiseBands {
        high: args.high_amount,
        low: args.low_amount,
        scale_px: args.band_scale,
    };
    bands.validate()?;
    let bundle = ModelStore::default_store()?
        .resolve(args.model.as_deref())
        .context("give --model, or install one: `seiza model install noise-v1-d33`")?;
    let options = BackendOptions {
        library: args.runtime.onnxruntime,
        tile: args.tile,
        precision: args.precision,
        ..Default::default()
    };
    let backend = match args.backend {
        Some(backend) => backend,
        None => Backend::detect(&bundle, &options)?,
    };

    let mut frame = crate::common::open_frame(&args.input, "model input")?;
    if frame.bayer.is_some() {
        bail!(
            "a model cannot read the raw Bayer mosaic in {}; debayer or stack it first",
            args.input.display()
        );
    }

    let opening = Instant::now();
    eprintln!("opening {} on {backend}", bundle.card().name);
    let mut model = NoiseModel::open(&bundle, backend, &options)
        .with_context(|| format!("could not open {} on {backend}", bundle.card().name))?;
    eprintln!("opened in {:.1} s", opening.elapsed().as_secs_f64());

    crate::interrupt::install();
    let image = &frame.image;
    let view = ImageView::new(&image.data, image.width, image.height, image.channels)?;
    let running = Instant::now();
    let progress = |step: Progress| {
        eprint!(
            "\rchannel {}/{}: tile {}/{}",
            step.channel + 1,
            step.channels,
            step.tile,
            step.tiles
        );
        let _ = std::io::stderr().flush();
    };
    let result = model.denoise(
        view,
        &DenoiseOptions {
            progress: Some(&progress),
            cancel: Some(&crate::interrupt::interrupted),
        },
    );
    eprintln!();
    if matches!(result, Err(seiza_model::ModelError::Cancelled)) {
        std::process::exit(crate::interrupt::INTERRUPTED_EXIT_CODE);
    }
    let result = result.context("could not run the model")?;
    let seconds = running.elapsed().as_secs_f64();

    frame.image.data = result.blend(&bands)?;
    let cards = operation_cards(&result.report, &bands);
    write_processed_image_fits_f32(&args.output, &frame.image, &frame.headers, &cards)
        .with_context(|| format!("could not write {}", args.output.display()))?;
    crate::common::wrote(
        &args.output,
        format_args!(
            "{} on {backend}, {}, tile {}, in {seconds:.1} s, taking out {:.0}% of the grain and {:.0}% of the mottle",
            result.report.model,
            result.report.precision,
            result.report.tile,
            bands.high * 100.0,
            bands.low * 100.0,
        ),
    );
    for (channel, scales) in result.report.channels.iter().enumerate() {
        println!(
            "channel {}: noise {:.6}, background {:.6}",
            channel + 1,
            scales.noise,
            scales.background
        );
    }
    Ok(())
}

fn operation_cards(report: &DenoiseReport, bands: &NoiseBands) -> Vec<WriteHeaderCard> {
    let text = |key: &str, value: String, comment: &str| {
        WriteHeaderCard::new(key, HeaderValue::String(value)).with_comment(comment)
    };
    let number = |key: &str, value: f32, comment: &str| {
        WriteHeaderCard::new(key, HeaderValue::Float(f64::from(value))).with_comment(comment)
    };
    let mut cards = vec![
        text("SEIZAMDL", report.model.clone(), "Seiza learned model"),
        text("SEIZATRF", "LINEAR".to_string(), "linear sample transfer"),
        text(
            "MDHASH",
            report.sha256.chars().take(16).collect(),
            "model graph SHA-256, first 16",
        ),
        text(
            "MDCARD",
            report.card_sha256.chars().take(16).collect(),
            "model card SHA-256, first 16",
        ),
        text("MDBACKND", report.backend.to_string(), "backend requested"),
        text("MDPREC", report.precision.to_string(), "graph precision"),
        text(
            "MDNOISE",
            report.noise_map.to_string(),
            "noise input, made from the image",
        ),
        WriteHeaderCard::new(
            "MDTILE",
            HeaderValue::Integer(i64::try_from(report.tile).unwrap_or(i64::MAX)),
        )
        .with_comment("tile side in pixels"),
        number("MDHIGH", bands.high, "share of the grain taken out"),
        number("MDLOW", bands.low, "share of the mottle taken out"),
        number("MDSCALE", bands.scale_px, "band scale in pixels"),
        text(
            "MDVERS",
            seiza_model::OUTPUT_VERSION.to_string(),
            "seiza-model output version",
        ),
    ];
    if let Some(guard) = report.holes {
        cards.push(
            WriteHeaderCard::new("MDHOLES", HeaderValue::Float(guard.depth))
                .with_comment("holes this many sigma deep kept from the model"),
        );
    }
    cards
}

fn backends(args: &RuntimeArgs) -> Result<()> {
    let options = BackendOptions {
        library: args.onnxruntime.clone(),
        ..Default::default()
    };
    let info = runtime_info(&options)?;
    println!("ONNX Runtime {} at {}", info.version, info.path.display());
    for status in Backend::available(&options) {
        let word = status.backend.to_string();
        match status.unavailable {
            None if status.automatic => println!("  {word:<14} available"),
            None => {
                println!("  {word:<14} available when asked for: not measured on this hardware")
            }
            Some(reason) => println!("  {word:<14} {reason}"),
        }
    }
    Ok(())
}

fn list() -> Result<()> {
    let store = ModelStore::default_store()?;
    let models = store.list()?;
    if models.is_empty() {
        println!("no models are installed in {}", store.root().display());
    }
    for bundle in models {
        let card = bundle.card();
        let graphs: Vec<String> = card
            .graphs
            .iter()
            .map(|graph| graph.precision.to_string())
            .collect();
        println!(
            "{}: {}, reads {} noise{}, {}",
            card.name,
            card.task,
            card.reading.noise_map,
            if card.reading.holes.is_some() {
                ", keeps holes from the network"
            } else {
                ""
            },
            graphs.join(" and ")
        );
    }
    Ok(())
}

fn install(model: &str) -> Result<()> {
    let store = ModelStore::default_store()?;
    let bundle = if names_a_path(model) {
        store
            .install(model)
            .with_context(|| format!("could not install {model}"))?
    } else {
        let manager = seiza_download::ModelManager::builder()
            .policy(seiza_download::CachePolicy::ForceRefresh)
            .build()?;
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("failed to start the download runtime")?
            .block_on(store.download(&manager, model, crate::report_download_event))
            .with_context(|| format!("could not install {model}"))?
    };
    println!(
        "installed {} in {}",
        bundle.card().name,
        bundle.root().display()
    );
    Ok(())
}

/// Whether `install`'s argument is a bundle's directory rather than a
/// published model's name, which is never a path.
fn names_a_path(model: &str) -> bool {
    model.contains(['/', '\\']) || Path::new(model).exists()
}

fn verify(name: Option<&str>) -> Result<()> {
    let store = ModelStore::default_store()?;
    let models = match name {
        Some(name) => vec![store.find(name)?],
        None => store.list()?,
    };
    if models.is_empty() {
        println!("no models are installed in {}", store.root().display());
    }
    for bundle in models {
        bundle.verify()?;
        println!("{}: matches its card", bundle.card().name);
    }
    Ok(())
}

fn remove(name: &str) -> Result<()> {
    ModelStore::default_store()?.remove(name)?;
    println!("removed {name}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Harness {
        #[command(subcommand)]
        command: ModelCommand,
    }

    #[test]
    fn denoise_takes_its_amounts_and_a_backend_by_name() {
        let parsed = Harness::try_parse_from([
            "model",
            "denoise",
            "in.fits",
            "-o",
            "out.fits",
            "--backend",
            "neural-engine",
            "--high-amount",
            "0.5",
            "--tile",
            "768",
            "--precision",
            "f16",
        ])
        .unwrap();
        let ModelCommand::Denoise(args) = parsed.command else {
            panic!("not denoise")
        };
        assert_eq!(args.backend.unwrap().to_string(), "neural-engine");
        assert_eq!(args.high_amount, 0.5);
        assert_eq!(args.low_amount, 0.9);
        assert_eq!(args.tile, Some(768));
    }

    #[test]
    fn install_takes_a_published_name_or_a_bundle_s_directory() {
        let parsed = Harness::try_parse_from(["model", "install", "noise-v1-d33"]).unwrap();
        let ModelCommand::Install { model } = parsed.command else {
            panic!("not install")
        };
        assert!(!names_a_path(&model), "a name is downloaded");
        assert!(names_a_path("bundles/noise-v1-d33"));
        assert!(names_a_path("bundles\\noise-v1-d33"));
        let temp = tempfile::tempdir().unwrap();
        assert!(
            names_a_path(temp.path().to_str().unwrap()),
            "a directory is installed from"
        );
    }

    #[test]
    fn an_unknown_backend_is_refused_at_the_command_line() {
        let parsed = Harness::try_parse_from([
            "model",
            "denoise",
            "in.fits",
            "-o",
            "out.fits",
            "--backend",
            "metal",
        ]);
        assert!(parsed.is_err());
    }

    #[test]
    fn the_cards_record_what_ran() {
        let report = DenoiseReport {
            model: "noise-test".to_string(),
            sha256: "0123456789abcdef0123".to_string(),
            card_sha256: "fedcba9876543210fedc".to_string(),
            backend: Backend::Cpu,
            precision: Precision::F32,
            tile: 512,
            noise_map: seiza_model::NoiseInput::Constant,
            holes: None,
            channels: Vec::new(),
        };
        let cards = operation_cards(&report, &NoiseBands::default());
        let keys: Vec<&str> = cards.iter().map(WriteHeaderCard::keyword).collect();
        assert!(keys.contains(&"MDHASH") && keys.contains(&"MDCARD") && keys.contains(&"MDBACKND"));
        assert!(
            !keys.contains(&"MDHOLES"),
            "a run without the guard says nothing of it"
        );
        assert!(keys.iter().all(|key| key.len() <= 8), "{keys:?}");
        let guarded = DenoiseReport {
            holes: Some(seiza_model::HoleGuard {
                depth: 30.0,
                margin: 3,
                fill_radius: 12,
                restore: 5.0,
            }),
            ..report
        };
        let cards = operation_cards(&guarded, &NoiseBands::default());
        assert!(cards.iter().any(|card| card.keyword() == "MDHOLES"));
    }
}
