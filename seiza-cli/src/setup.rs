use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use directories::ProjectDirs;
use seiza_download::Dataset;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;

#[derive(Args)]
pub(crate) struct SetupArgs {
    /// Catalog package to install without prompting for a selection
    #[arg(long, value_enum)]
    preset: Option<SetupPreset>,
    /// Directory that receives the selected catalog files
    #[arg(long)]
    output: Option<PathBuf>,
    /// Accept the displayed selection without confirmation
    #[arg(long)]
    yes: bool,
    /// Adjust the welcome text for setup launched by the Windows installer
    #[arg(long, hide = true)]
    from_installer: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum SetupPreset {
    Objects,
    SolverLite,
    SolverGaia,
    BlindDeep,
    All,
}

impl SetupPreset {
    fn files(self) -> Vec<String> {
        let datasets: &[Dataset] = match self {
            Self::Objects => &[Dataset::Objects],
            Self::SolverLite => &[Dataset::Objects, Dataset::StarsLiteTycho2],
            Self::SolverGaia => &[Dataset::Objects, Dataset::StarsGaia],
            Self::BlindDeep => &[
                Dataset::Objects,
                Dataset::StarsDeepGaia17,
                Dataset::BlindGaia16,
            ],
            Self::All => return Vec::new(),
        };
        datasets
            .iter()
            .map(|dataset| dataset.file_name().to_string())
            .collect()
    }

    fn description(self) -> &'static str {
        match self {
            Self::Objects => "Object catalog for search and image overlays",
            Self::SolverLite => "Object catalog and compact Tycho-2 solver catalog",
            Self::SolverGaia => "Object catalog and Gaia solver catalog",
            Self::BlindDeep => "Object catalog, deep Gaia catalog, and blind index",
            Self::All => "Every catalog in the published bundle",
        }
    }
}

pub(crate) fn run(args: SetupArgs) -> Result<()> {
    let interactive = io::stdin().is_terminal() && io::stdout().is_terminal();
    if args.preset.is_none() && !interactive {
        anyhow::bail!(
            "interactive setup needs a terminal; pass --preset and --yes for unattended setup"
        );
    }

    if args.from_installer {
        println!("Seiza is installed. Let's add the catalogs you want to use.\n");
    } else {
        println!("Seiza catalog setup\n");
    }

    let output = args.output.unwrap_or_else(default_catalog_dir);
    let preset = match args.preset {
        Some(preset) => preset,
        None => {
            let mut input = io::stdin().lock();
            let mut output = io::stdout().lock();
            prompt_preset(&mut input, &mut output)?
        }
    };

    println!("Selection : {}", preset.description());
    println!("Directory : {}", output.display());
    println!("Downloads are SHA-256 verified and safe to retry.\n");

    if !args.yes {
        if !interactive {
            anyhow::bail!("confirmation requires a terminal; pass --yes to continue unattended");
        }
        let mut input = io::stdin().lock();
        let mut output_stream = io::stdout().lock();
        if !prompt_confirm(&mut input, &mut output_stream)? {
            println!("Setup cancelled; Seiza remains installed.");
            return Ok(());
        }
    }

    crate::run_prebuilt_download(output, preset.files())?;
    println!("\nCatalog setup complete. You can run `seiza setup` again at any time.");
    Ok(())
}

fn default_catalog_dir() -> PathBuf {
    ProjectDirs::from("fyi", "Seiza", "seiza")
        .map(|dirs| dirs.data_local_dir().join("catalogs"))
        .unwrap_or_else(|| PathBuf::from("seiza-data"))
}

fn prompt_preset<R: BufRead, W: Write>(input: &mut R, output: &mut W) -> Result<SetupPreset> {
    writeln!(output, "  1. Object catalog")?;
    writeln!(output, "  2. Object catalog + lightweight plate solver")?;
    writeln!(output, "  3. Object catalog + Gaia plate solver")?;
    writeln!(output, "  4. Object catalog + deep blind solver")?;
    writeln!(output, "  5. Complete catalog bundle")?;
    write!(output, "\nChoose a package [1]: ")?;
    output.flush()?;

    let mut answer = String::new();
    input
        .read_line(&mut answer)
        .context("failed to read catalog selection")?;
    match answer.trim() {
        "" | "1" => Ok(SetupPreset::Objects),
        "2" => Ok(SetupPreset::SolverLite),
        "3" => Ok(SetupPreset::SolverGaia),
        "4" => Ok(SetupPreset::BlindDeep),
        "5" => Ok(SetupPreset::All),
        other => anyhow::bail!("invalid selection {other:?}; expected a number from 1 to 5"),
    }
}

fn prompt_confirm<R: BufRead, W: Write>(input: &mut R, output: &mut W) -> Result<bool> {
    write!(output, "Download this package now? [Y/n] ")?;
    output.flush()?;
    let mut answer = String::new();
    input
        .read_line(&mut answer)
        .context("failed to read confirmation")?;
    match answer.trim().to_ascii_lowercase().as_str() {
        "" | "y" | "yes" => Ok(true),
        "n" | "no" => Ok(false),
        other => anyhow::bail!("invalid confirmation {other:?}; enter yes or no"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn preset_prompt_defaults_to_objects() {
        let mut output = Vec::new();
        assert_eq!(
            prompt_preset(&mut Cursor::new("\n"), &mut output).unwrap(),
            SetupPreset::Objects
        );
    }

    #[test]
    fn preset_prompt_selects_deep_blind_package() {
        let mut output = Vec::new();
        let preset = prompt_preset(&mut Cursor::new("4\n"), &mut output).unwrap();
        assert_eq!(preset, SetupPreset::BlindDeep);
        assert_eq!(
            preset.files(),
            ["objects.bin", "stars-deep-gaia17.bin", "blind-gaia16.idx"]
        );
    }

    #[test]
    fn confirmation_accepts_default_and_no() {
        assert!(prompt_confirm(&mut Cursor::new("\n"), &mut Vec::new()).unwrap());
        assert!(!prompt_confirm(&mut Cursor::new("no\n"), &mut Vec::new()).unwrap());
    }
}
