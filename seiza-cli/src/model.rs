//! `seiza model`: run and manage learned models.

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use seiza_model::{
    Backend, BackendOptions, DenoiseOptions, FileError, ModelBundle, ModelError, ModelStore,
    NoiseBands, NoiseModel, Precision, Progress, Published, RestorationArea, RestorationImage,
    SharpenModel, SharpenOptions, Task, runtime_info,
};
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
    /// Sharpen a linear FITS image: bring its stars, and everything else, to
    /// a share of their width
    Sharpen(Box<SharpenArgs>),
    /// Show which backends run here, and why the others do not
    Backends(RuntimeArgs),
    /// List the installed models
    List,
    /// Install a published model, or a bundle from a directory, once its
    /// hashes match
    Install {
        /// A published model's name, such as noise-v1-d33, which is
        /// downloaded from downloads.seiza.fyi; or a directory holding a
        /// bundle's card.json and graphs (default: the latest published noise
        /// model)
        #[arg(value_name = "NAME|PATH")]
        model: Option<String>,
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
    /// installed noise model, or, with none installed, the latest published
    /// one, which is fetched first)
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
    /// DXGI adapter index for DirectML.
    #[arg(long, value_name = "INDEX")]
    device_id: Option<i32>,
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

#[derive(Args)]
struct SharpenArgs {
    /// Calibrated, stacked linear mono or RGB FITS or XISF input, from any
    /// stacker
    input: PathBuf,
    /// Sharpened linear 32-bit floating-point FITS output
    #[arg(short, long)]
    output: PathBuf,
    /// An installed model's name, or a bundle's directory (default: the one
    /// installed sharpening model)
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
    /// DXGI adapter index for DirectML.
    #[arg(long, value_name = "INDEX")]
    device_id: Option<i32>,
    /// The share of its width every star is brought to, above 0 and at most
    /// 1, never below the model's floor in pixels (default: the model's)
    #[arg(long)]
    stars: Option<f32>,
    /// The share asked of everything that is not a star (default: the
    /// model's)
    #[arg(long)]
    rest: Option<f32>,
    /// How hard the finest part of the change is held back where it lies
    /// under the noise, in multiples of the image's noise; 0 keeps all of it
    #[arg(long, default_value_t = 0.0)]
    rolloff: f32,
    #[command(flatten)]
    runtime: RuntimeArgs,
}

pub(crate) fn run(args: ModelArgs) -> Result<()> {
    match args.command {
        ModelCommand::Denoise(args) => denoise(*args),
        ModelCommand::Sharpen(args) => sharpen(*args),
        ModelCommand::Backends(args) => backends(&args),
        ModelCommand::List => list(),
        ModelCommand::Install { model } => install(model.as_deref()),
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
    let options = BackendOptions {
        library: args.runtime.onnxruntime,
        tile: args.tile,
        precision: args.precision,
        device_id: args.device_id,
        ..Default::default()
    };

    // The input is read first, so a run that cannot start never waits on a
    // model's download.
    let input = RestorationImage::open(&args.input, RestorationArea::Full)?;
    let bundle = denoise_model(args.model.as_deref(), &options)?;
    let backend = match args.backend {
        Some(backend) => backend,
        None => Backend::detect(&bundle, &options)?,
    };

    let opening = Instant::now();
    eprintln!("opening {} on {backend}", bundle.card().name);
    let mut model = NoiseModel::open(&bundle, backend, &options)
        .with_context(|| format!("could not open {} on {backend}", bundle.card().name))?;
    eprintln!("opened in {:.1} s", opening.elapsed().as_secs_f64());

    crate::interrupt::install();
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
    let result = input.denoise(
        &mut model,
        &DenoiseOptions {
            progress: Some(&progress),
            cancel: Some(&crate::interrupt::interrupted),
        },
    );
    eprintln!();
    if matches!(
        result,
        Err(FileError::Model(seiza_model::ModelError::Cancelled))
    ) {
        std::process::exit(crate::interrupt::INTERRUPTED_EXIT_CODE);
    }
    let result = result.context("could not run the model")?;
    let seconds = running.elapsed().as_secs_f64();

    result
        .write_linear(&args.output, &bands)
        .with_context(|| format!("could not write {}", args.output.display()))?;
    let report = result.report();
    crate::common::wrote(
        &args.output,
        format_args!(
            "{} on {backend}, {}, tile {}, in {seconds:.1} s, taking out {:.0}% of the grain and {:.0}% of the mottle",
            report.model,
            report.precision,
            report.tile,
            bands.high * 100.0,
            bands.low * 100.0,
        ),
    );
    for (channel, scales) in report.channels.iter().enumerate() {
        println!(
            "channel {}: noise {:.6}, background {:.6}",
            channel + 1,
            scales.noise,
            scales.background
        );
    }
    Ok(())
}

fn sharpen(args: SharpenArgs) -> Result<()> {
    crate::provenance::validate_path_roles([
        ("model input".into(), args.input.as_path()),
        ("model output".into(), args.output.as_path()),
    ])?;
    let options = BackendOptions {
        library: args.runtime.onnxruntime,
        tile: args.tile,
        precision: args.precision,
        device_id: args.device_id,
        ..Default::default()
    };

    let input = RestorationImage::open(&args.input, RestorationArea::Full)?;
    // No sharpening model is published, so none is fetched: one is named, or
    // installed from a bundle's directory.
    let bundle = ModelStore::default_store()?
        .resolve(args.model.as_deref(), Task::Sharpen)
        .context("give --model, or install a sharpening bundle with `seiza model install DIR`")?;
    let backend = match args.backend {
        Some(backend) => backend,
        None => Backend::detect(&bundle, &options)?,
    };

    let opening = Instant::now();
    eprintln!("opening {} on {backend}", bundle.card().name);
    let mut model = SharpenModel::open(&bundle, backend, &options)
        .with_context(|| format!("could not open {} on {backend}", bundle.card().name))?;
    eprintln!("opened in {:.1} s", opening.elapsed().as_secs_f64());

    crate::interrupt::install();
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
    let result = input.sharpen(
        &mut model,
        &SharpenOptions {
            stars: args.stars,
            rest: args.rest,
            rolloff: args.rolloff,
            progress: Some(&progress),
            cancel: Some(&crate::interrupt::interrupted),
        },
    );
    eprintln!();
    if matches!(
        result,
        Err(FileError::Model(seiza_model::ModelError::Cancelled))
    ) {
        std::process::exit(crate::interrupt::INTERRUPTED_EXIT_CODE);
    }
    let result = result.context("could not run the model")?;
    let seconds = running.elapsed().as_secs_f64();

    result
        .write_linear(&args.output)
        .with_context(|| format!("could not write {}", args.output.display()))?;
    let report = result.report();
    crate::common::wrote(
        &args.output,
        format_args!(
            "{} on {backend}, {}, tile {}, in {seconds:.1} s, stars to {:.0}% of their width and the rest to {:.0}%",
            report.model,
            report.precision,
            report.tile,
            report.stars * 100.0,
            report.rest * 100.0,
        ),
    );
    for (channel, read) in report.channels.iter().enumerate() {
        println!(
            "channel {}: stars {:.2} px wide, from {} stars at {} × {} nodes{}, {} point sources, noise {:.6}, background {:.6}",
            channel + 1,
            read.psf_width_px,
            read.psf_stars,
            read.psf_nodes.1,
            read.psf_nodes.0,
            if read.psf_global {
                " (one PSF for the whole image)"
            } else {
                ""
            },
            read.points,
            read.noise,
            read.background
        );
    }
    Ok(())
}

/// The model `denoise` runs. A model named with `--model` always wins, and is
/// never fetched. Without one, the one installed noise model runs, with no
/// network at all; with none installed, the latest published noise model is
/// fetched and installed first. Several installed is an error naming them.
fn denoise_model(chosen: Option<&str>, options: &BackendOptions) -> Result<ModelBundle> {
    let store = ModelStore::default_store()?;
    if let Some(chosen) = chosen {
        return store.resolve(Some(chosen), Task::Denoise).with_context(|| {
            format!("install it with `seiza model install {chosen}`, or give a bundle's directory")
        });
    }
    match store.resolve(None, Task::Denoise) {
        Err(ModelError::NoModelInstalled { .. }) => {
            // Without ONNX Runtime the model could not run: say so before
            // fetching hundreds of megabytes.
            runtime_info(options)?;
            println!("No noise model is installed; fetching the latest published one.");
            download(&store, Published::Latest(Task::Denoise)).context(
                "could not fetch the latest noise model; install it with `seiza model install` \
                 once the network is reachable, or give --model",
            )
        }
        resolved => resolved.context("give --model, or remove the models you do not want"),
    }
}

/// Fetch `model` from the published models into `store`, with the manifest
/// fresh from the server and progress shown as catalog downloads show it.
pub(crate) fn download(store: &ModelStore, model: Published<'_>) -> Result<ModelBundle> {
    let manager = seiza_download::ModelManager::builder()
        .policy(seiza_download::CachePolicy::ForceRefresh)
        .build()?;
    let bundle = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to start the download runtime")?
        .block_on(store.download(&manager, model, crate::report_download_event))?;
    Ok(bundle)
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
        let reads = match &card.sharpen {
            Some(sharpen) => format!(
                "stars to {:.0}% of their width and the rest to {:.0}% unless told, no star under {} px",
                sharpen.strengths.stars * 100.0,
                sharpen.strengths.rest * 100.0,
                sharpen.floor_px
            ),
            None => format!(
                "reads {} noise{}",
                card.reading.noise_map,
                if card.reading.holes.is_some() {
                    ", keeps holes from the network"
                } else {
                    ""
                }
            ),
        };
        println!(
            "{}: {}, {reads}, {}",
            card.name,
            card.task,
            graphs.join(" and ")
        );
    }
    Ok(())
}

fn install(model: Option<&str>) -> Result<()> {
    let store = ModelStore::default_store()?;
    let bundle = match model {
        Some(path) if names_a_path(path) => store
            .install(path)
            .with_context(|| format!("could not install {path}"))?,
        Some(name) => download(&store, Published::Named(name))
            .with_context(|| format!("could not install {name}"))?,
        None => download(&store, Published::Latest(Task::Denoise))
            .context("could not install the latest noise model")?,
    };
    installed(&store, &bundle)
}

/// Say where `bundle` went, and which other models of its task remain, since
/// `seiza model denoise` then needs `--model` to choose.
pub(crate) fn installed(store: &ModelStore, bundle: &ModelBundle) -> Result<()> {
    println!(
        "installed {} in {}",
        bundle.card().name,
        bundle.root().display()
    );
    let others: Vec<String> = store
        .list()?
        .into_iter()
        .filter(|other| {
            other.card().task == bundle.card().task && other.card().name != bundle.card().name
        })
        .map(|other| other.card().name.clone())
        .collect();
    if !others.is_empty() {
        println!(
            "also installed: {}; `seiza model denoise` takes --model while more than one is \
             installed, or remove the others with `seiza model remove NAME`",
            others.join(", ")
        );
    }
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
        let ModelCommand::Install { model: Some(model) } = parsed.command else {
            panic!("not install with a name")
        };
        assert!(!names_a_path(&model), "a name is downloaded");
        let parsed = Harness::try_parse_from(["model", "install"]).unwrap();
        assert!(
            matches!(parsed.command, ModelCommand::Install { model: None }),
            "with nothing named, the latest published noise model"
        );
        assert!(names_a_path("bundles/noise-v1-d33"));
        assert!(names_a_path("bundles\\noise-v1-d33"));
        let temp = tempfile::tempdir().unwrap();
        assert!(
            names_a_path(temp.path().to_str().unwrap()),
            "a directory is installed from"
        );
    }

    #[test]
    fn sharpen_takes_its_two_shares_and_leaves_them_to_the_model_unasked() {
        let parsed = Harness::try_parse_from([
            "model",
            "sharpen",
            "in.fits",
            "-o",
            "out.fits",
            "--stars",
            "0.46",
            "--rolloff",
            "1",
            "--backend",
            "coreml-gpu",
        ])
        .unwrap();
        let ModelCommand::Sharpen(args) = parsed.command else {
            panic!("not sharpen")
        };
        assert_eq!(args.stars, Some(0.46));
        assert_eq!(args.rest, None, "the model's own share, unless told");
        assert_eq!(args.rolloff, 1.0);
        assert_eq!(args.backend.unwrap().to_string(), "coreml-gpu");
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
}
