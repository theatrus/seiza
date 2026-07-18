use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use seiza::data_paths::default_catalog_dir;
use seiza_download::Dataset;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;

#[cfg(windows)]
use std::ffi::{OsStr, OsString};
#[cfg(windows)]
use std::os::windows::ffi::OsStrExt;

#[derive(Args)]
pub(crate) struct SetupArgs {
    /// Catalog package to install without prompting for a selection
    #[arg(long, value_enum)]
    preset: Option<SetupPreset>,
    /// Directory that receives the selected files (defaults to SEIZA_CATALOG_DIR when set)
    #[arg(long)]
    output: Option<PathBuf>,
    /// Accept the displayed selection without confirmation
    #[arg(long)]
    yes: bool,
    /// Adjust the welcome text for setup launched by the Windows installer
    #[arg(long, hide = true)]
    from_installer: bool,
    /// Relaunch catalog setup with administrator privileges (Windows installer use only)
    #[arg(long, hide = true)]
    elevate: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum SetupPreset {
    SolverLite,
    SolverGaia,
    BlindDeep,
    All,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NextAction {
    Menu,
    Quit,
}

impl SetupPreset {
    #[cfg(windows)]
    fn cli_name(self) -> &'static str {
        match self {
            Self::SolverLite => "solver-lite",
            Self::SolverGaia => "solver-gaia",
            Self::BlindDeep => "blind-deep",
            Self::All => "all",
        }
    }

    fn files(self) -> Vec<String> {
        let datasets: &[Dataset] = match self {
            Self::SolverLite => &[
                Dataset::Objects,
                Dataset::MinorBodies,
                Dataset::Transients,
                Dataset::StarsLiteTycho2,
            ],
            Self::SolverGaia => &[
                Dataset::Objects,
                Dataset::MinorBodies,
                Dataset::Transients,
                Dataset::StarsGaia,
            ],
            Self::BlindDeep => &[
                Dataset::Objects,
                Dataset::MinorBodies,
                Dataset::Transients,
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
            Self::SolverLite => {
                "Telescope control and hinted solves: objects + Solar System + transients + compact Tycho-2 catalog"
            }
            Self::SolverGaia => {
                "Narrow or crowded fields: objects + Solar System + transients + denser Gaia solver catalog"
            }
            Self::BlindDeep => {
                "Unknown sky position: objects + Solar System + transients + deep Gaia catalog + blind index"
            }
            Self::All => "Development and offline use: every published catalog",
        }
    }
}

pub(crate) fn run(args: SetupArgs) -> Result<()> {
    let from_installer = args.from_installer;

    #[cfg(windows)]
    let result = if args.elevate {
        launch_elevated(&args)
    } else {
        run_setup(args)
    };

    #[cfg(not(windows))]
    let result = run_setup(args);

    if let Err(error) = &result
        && from_installer
    {
        eprintln!("\nSeiza catalog setup failed:\n{error:#}");
        if io::stdin().is_terminal() {
            eprintln!("\nPress Enter to close this window.");
            let mut answer = String::new();
            let _ = io::stdin().read_line(&mut answer);
        }
    }

    result
}

fn run_setup(args: SetupArgs) -> Result<()> {
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
    loop {
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
                anyhow::bail!(
                    "confirmation requires a terminal; pass --yes to continue unattended"
                );
            }
            let mut input = io::stdin().lock();
            let mut output_stream = io::stdout().lock();
            if !prompt_confirm(&mut input, &mut output_stream)? {
                println!("Setup cancelled; Seiza remains installed.");
                return Ok(());
            }
        }

        crate::run_prebuilt_download(output.clone(), preset.files())?;
        println!("\nSuccess! Catalog setup is complete.");
        println!("Catalogs are ready in {}.", output.display());
        println!("Seiza's ASTAP-compatible solver mode will discover them automatically.");

        if args.preset.is_some() || !interactive {
            return Ok(());
        }

        let mut input = io::stdin().lock();
        let mut output_stream = io::stdout().lock();
        match prompt_next_action(&mut input, &mut output_stream)? {
            NextAction::Menu => println!(),
            NextAction::Quit => return Ok(()),
        }
    }
}

#[cfg(windows)]
fn launch_elevated(args: &SetupArgs) -> Result<()> {
    use windows_sys::Win32::UI::Shell::ShellExecuteW;
    use windows_sys::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;

    let executable = std::env::current_exe().context("failed to locate seiza.exe")?;
    let executable = wide_null(executable.as_os_str());
    let verb = wide_null(OsStr::new("runas"));
    let parameters = elevated_parameters(args);

    // SAFETY: All pointers refer to nul-terminated UTF-16 buffers that remain
    // alive for the duration of ShellExecuteW. Null window/directory handles
    // request the shell defaults. The elevated child owns its new process.
    let result = unsafe {
        ShellExecuteW(
            std::ptr::null_mut(),
            verb.as_ptr(),
            executable.as_ptr(),
            parameters.as_ptr(),
            std::ptr::null(),
            SW_SHOWNORMAL,
        )
    };
    if result as usize <= 32 {
        anyhow::bail!(
            "administrator approval was declined or Windows could not start elevated setup (ShellExecute error {})",
            result as usize
        );
    }

    Ok(())
}

#[cfg(windows)]
fn elevated_parameters(args: &SetupArgs) -> Vec<u16> {
    let mut arguments = vec![OsString::from("setup")];
    if args.from_installer {
        arguments.push(OsString::from("--from-installer"));
    }
    if let Some(output) = &args.output {
        arguments.push(OsString::from("--output"));
        arguments.push(output.as_os_str().to_os_string());
    }
    if let Some(preset) = args.preset {
        arguments.push(OsString::from("--preset"));
        arguments.push(OsString::from(preset.cli_name()));
    }
    if args.yes {
        arguments.push(OsString::from("--yes"));
    }

    quote_windows_arguments(&arguments)
}

#[cfg(windows)]
fn quote_windows_arguments(arguments: &[OsString]) -> Vec<u16> {
    let mut command_line = Vec::new();
    for argument in arguments {
        if !command_line.is_empty() {
            command_line.push(u16::from(b' '));
        }
        append_quoted_windows_argument(&mut command_line, argument.as_os_str());
    }
    command_line.push(0);
    command_line
}

#[cfg(windows)]
fn append_quoted_windows_argument(command_line: &mut Vec<u16>, argument: &OsStr) {
    let units: Vec<u16> = argument.encode_wide().collect();
    let requires_quotes = units.is_empty()
        || units.iter().any(|unit| {
            *unit == u16::from(b' ') || *unit == u16::from(b'\t') || *unit == u16::from(b'"')
        });
    if !requires_quotes {
        command_line.extend_from_slice(&units);
        return;
    }

    command_line.push(u16::from(b'"'));
    let mut backslashes = 0;
    for unit in units {
        if unit == u16::from(b'\\') {
            backslashes += 1;
        } else if unit == u16::from(b'"') {
            command_line.extend(std::iter::repeat_n(u16::from(b'\\'), backslashes * 2 + 1));
            command_line.push(unit);
            backslashes = 0;
        } else {
            command_line.extend(std::iter::repeat_n(u16::from(b'\\'), backslashes));
            command_line.push(unit);
            backslashes = 0;
        }
    }
    command_line.extend(std::iter::repeat_n(u16::from(b'\\'), backslashes * 2));
    command_line.push(u16::from(b'"'));
}

#[cfg(windows)]
fn wide_null(value: &OsStr) -> Vec<u16> {
    value.encode_wide().chain(std::iter::once(0)).collect()
}

fn prompt_next_action<R: BufRead, W: Write>(input: &mut R, output: &mut W) -> Result<NextAction> {
    writeln!(output, "\nWhat would you like to do next?")?;
    writeln!(output, "  1. Return to the catalog menu")?;
    writeln!(output, "  2. Quit")?;
    write!(output, "\nChoose an option [2]: ")?;
    output.flush()?;

    let mut answer = String::new();
    input
        .read_line(&mut answer)
        .context("failed to read next action")?;
    match answer.trim().to_ascii_lowercase().as_str() {
        "1" | "m" | "menu" => Ok(NextAction::Menu),
        "" | "2" | "q" | "quit" => Ok(NextAction::Quit),
        other => anyhow::bail!("invalid selection {other:?}; enter 1 for the menu or 2 to quit"),
    }
}

fn prompt_preset<R: BufRead, W: Write>(input: &mut R, output: &mut W) -> Result<SetupPreset> {
    writeln!(
        output,
        "Every option includes object search, Solar System objects, active transients, and at least one plate-solving catalog.\n"
    )?;
    writeln!(
        output,
        "  1. Telescope control / N.I.N.A.: lightweight hinted solving (recommended)"
    )?;
    writeln!(output, "  2. Narrow or crowded fields: denser Gaia solving")?;
    writeln!(output, "  3. Unknown sky position: deep blind solving")?;
    writeln!(
        output,
        "  4. Development / offline use: complete catalog bundle"
    )?;
    write!(output, "\nChoose a package [1]: ")?;
    output.flush()?;

    let mut answer = String::new();
    input
        .read_line(&mut answer)
        .context("failed to read catalog selection")?;
    match answer.trim() {
        "" | "1" => Ok(SetupPreset::SolverLite),
        "2" => Ok(SetupPreset::SolverGaia),
        "3" => Ok(SetupPreset::BlindDeep),
        "4" => Ok(SetupPreset::All),
        other => anyhow::bail!("invalid selection {other:?}; expected a number from 1 to 4"),
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
    fn preset_prompt_defaults_to_lightweight_solver() {
        let mut output = Vec::new();
        assert_eq!(
            prompt_preset(&mut Cursor::new("\n"), &mut output).unwrap(),
            SetupPreset::SolverLite
        );
        assert!(String::from_utf8(output).unwrap().contains("N.I.N.A."));
    }

    #[test]
    fn preset_prompt_selects_deep_blind_package() {
        let mut output = Vec::new();
        let preset = prompt_preset(&mut Cursor::new("3\n"), &mut output).unwrap();
        assert_eq!(preset, SetupPreset::BlindDeep);
        assert_eq!(
            preset.files(),
            [
                "objects.bin",
                "minor-bodies.bin",
                "transients.bin",
                "stars-deep-gaia17.bin",
                "blind-gaia16.idx",
            ]
        );
    }

    #[test]
    fn every_selective_preset_includes_sky_objects_and_plate_solving() {
        for preset in [
            SetupPreset::SolverLite,
            SetupPreset::SolverGaia,
            SetupPreset::BlindDeep,
        ] {
            let files = preset.files();
            assert!(files.iter().any(|file| file == "objects.bin"));
            assert!(files.iter().any(|file| file == "minor-bodies.bin"));
            assert!(files.iter().any(|file| file == "transients.bin"));
            assert!(files.iter().any(|file| file.starts_with("stars-")));
        }
    }

    #[test]
    fn confirmation_accepts_default_and_no() {
        assert!(prompt_confirm(&mut Cursor::new("\n"), &mut Vec::new()).unwrap());
        assert!(!prompt_confirm(&mut Cursor::new("no\n"), &mut Vec::new()).unwrap());
    }

    #[test]
    fn next_action_returns_to_menu_or_quits() {
        assert_eq!(
            prompt_next_action(&mut Cursor::new("1\n"), &mut Vec::new()).unwrap(),
            NextAction::Menu
        );
        assert_eq!(
            prompt_next_action(&mut Cursor::new("\n"), &mut Vec::new()).unwrap(),
            NextAction::Quit
        );
        assert_eq!(
            prompt_next_action(&mut Cursor::new("quit\n"), &mut Vec::new()).unwrap(),
            NextAction::Quit
        );
    }

    #[cfg(windows)]
    #[test]
    fn elevated_command_line_quotes_spaces_and_trailing_backslashes() {
        let arguments = [
            OsString::from("setup"),
            OsString::from("--output"),
            OsString::from("C:\\Program Data\\Seiza\\catalogs\\"),
        ];
        let encoded = quote_windows_arguments(&arguments);
        let command_line = String::from_utf16(&encoded[..encoded.len() - 1]).unwrap();
        assert_eq!(
            command_line,
            "setup --output \"C:\\Program Data\\Seiza\\catalogs\\\\\""
        );
    }
}
