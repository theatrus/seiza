use anyhow::{Context, Result, ensure};
use clap::{Args, Subcommand};
use seiza_pipeline::{ChannelPlan, ChannelProgress, PartialStack, Store};
use std::path::{Path, PathBuf};

#[derive(Args)]
pub struct PipelineArgs {
    #[command(subcommand)]
    command: PipelineCommand,
}

#[derive(Subcommand)]
enum PipelineCommand {
    /// Stack a channel as partial stacks, one per session, into a store, and
    /// combine them; only the steps whose outputs the store lacks run
    Run {
        /// The channel plan, JSON: {"reference": {"image", "light",
        /// "calibration"}, "sessions": [{"lights", "calibration"}],
        /// "options": {"stack", "reintegration", "contributor", "host"}}
        plan: PathBuf,
        /// The store's directory, made if missing
        #[arg(long)]
        store: PathBuf,
        /// Also write the combined stack here, FITS or XISF by its extension
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Combine stack files on one grid into one, weighing each by its weight
    Combine {
        /// Partial or tile stack files
        #[arg(required = true)]
        stacks: Vec<PathBuf>,
        /// The combined stack, FITS or XISF by its extension
        #[arg(long)]
        output: PathBuf,
    },
    /// Print a stack file's identity cards as JSON, from its header alone
    Show {
        /// A partial or tile stack file
        stack: PathBuf,
    },
}

pub fn run(args: PipelineArgs) -> Result<()> {
    match args.command {
        PipelineCommand::Run {
            plan,
            store,
            output,
        } => run_plan(&plan, store, output.as_deref()),
        PipelineCommand::Combine { stacks, output } => combine(&stacks, &output),
        PipelineCommand::Show { stack } => {
            let identity = PartialStack::read_identity(&stack)
                .with_context(|| format!("reading {}", stack.display()))?;
            println!("{}", serde_json::to_string_pretty(&identity)?);
            Ok(())
        }
    }
}

fn run_plan(plan: &Path, store: PathBuf, output: Option<&Path>) -> Result<()> {
    let plan: ChannelPlan = serde_json::from_reader(
        std::fs::File::open(plan).with_context(|| format!("reading {}", plan.display()))?,
    )
    .with_context(|| format!("parsing {}", plan.display()))?;
    let store = Store::new(store);
    let outcome = seiza_pipeline::run_channel(&plan, &store, |event| match event {
        ChannelProgress::Planning { session } => {
            eprintln!("session {}: hashing its files", session + 1)
        }
        ChannelProgress::Reused { session, key } => {
            eprintln!("session {}: stored as {key}", session + 1)
        }
        ChannelProgress::Stacking { session, key } => {
            eprintln!("session {}: stacking as {key}", session + 1)
        }
        ChannelProgress::Combining { stacks } => eprintln!("combining {stacks} partial stacks"),
    })?;
    for (index, session) in outcome.sessions.iter().enumerate() {
        for (path, reason) in &session.rejected {
            eprintln!(
                "session {}: {} turned away: {reason}",
                index + 1,
                path.display()
            );
        }
        if session.path.is_none() {
            eprintln!("session {}: no light was accepted", index + 1);
        }
    }
    let identity = &outcome.stack.identity;
    println!(
        "{} {} frames from {} partial stacks: {}",
        if outcome.reused { "stored" } else { "combined" },
        identity.combined,
        identity.input_count,
        outcome.path.display()
    );
    if let Some(output) = output {
        outcome.stack.write(output)?;
        println!("wrote {}", output.display());
    }
    Ok(())
}

fn combine(stacks: &[PathBuf], output: &Path) -> Result<()> {
    ensure!(
        !stacks.iter().any(|stack| stack == output),
        "the output must not be one of the stacks"
    );
    let stacks = stacks
        .iter()
        .map(|path| PartialStack::open(path).with_context(|| format!("reading {}", path.display())))
        .collect::<Result<Vec<_>>>()?;
    let tile = PartialStack::combine(&stacks)?;
    tile.write(output)?;
    println!(
        "combined {} frames from {} stacks: {}",
        tile.identity.combined,
        tile.identity.input_count,
        output.display()
    );
    Ok(())
}
