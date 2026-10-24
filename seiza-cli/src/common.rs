use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use seiza_stacking::FitsFrame;
use std::fmt::Arguments;
use std::path::Path;

/// A pass that `--cpu-pass` keeps on the CPU.
#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
pub(crate) enum GpuPassArg {
    /// Defect learning's screen of each light
    Defects,
    /// VNG debayering
    Debayer,
    /// Star detection
    Detection,
    /// Resampling onto the reference grid
    Resampling,
    /// The live stack's accumulator
    Accumulator,
    /// --reintegrate's rejection passes
    Integration,
}

/// The GPU flags of every subcommand whose work can run on the GPU.
#[derive(Args, Debug)]
pub(crate) struct GpuArgs {
    /// Keep every pass on the CPU and never open the GPU. Otherwise
    /// debayering, star detection, resampling and stacking's other heavy
    /// passes run on the GPU, when there is one, in single precision
    #[arg(long)]
    no_gpu: bool,
    /// Keep these passes on the CPU and the rest on the GPU, comma-separated
    #[arg(long, value_enum, value_delimiter = ',')]
    cpu_pass: Vec<GpuPassArg>,
    /// Frame calls (debayering, detection, resampling, screening) the GPU
    /// takes at once; the rest wait without holding its memory
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u64).range(1..))]
    gpu_concurrency: u64,
    /// Device memory to plan the GPU's work in, in MiB; by default what the
    /// device has. With less, a live stack's accumulator stays on the CPU
    /// sooner, leaving the rest to other programs
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    gpu_memory_mib: Option<u64>,
    /// Print each GPU call's stages and their times to stderr
    #[arg(long)]
    gpu_trace: bool,
}

impl GpuArgs {
    /// The passes the GPU may run, as the flags say.
    pub(crate) fn options(&self) -> seiza_stacking::GpuOptions {
        let on_gpu = |pass| !self.cpu_pass.contains(&pass);
        seiza_stacking::GpuOptions {
            enabled: !self.no_gpu,
            defects: on_gpu(GpuPassArg::Defects),
            debayer: on_gpu(GpuPassArg::Debayer),
            detection: on_gpu(GpuPassArg::Detection),
            resampling: on_gpu(GpuPassArg::Resampling),
            accumulator: on_gpu(GpuPassArg::Accumulator),
            integration: on_gpu(GpuPassArg::Integration),
            concurrency: usize::try_from(self.gpu_concurrency).unwrap_or(usize::MAX),
            memory_bytes: self
                .gpu_memory_mib
                .map(|mib| mib.saturating_mul(1024 * 1024)),
            trace: self.gpu_trace,
        }
    }
}

/// Compare the final path component of `program` (without `.exe`) with
/// `expected`, case-insensitively, accepting both separator styles
/// regardless of host platform. Compatibility modes use this to answer to
/// a copy of the binary installed under the name the host tool expects.
pub(crate) fn binary_name_matches(program: &str, expected: &str) -> bool {
    let name = program.rsplit(['/', '\\']).next().unwrap_or(program);
    let stem = name
        .strip_suffix(".exe")
        .or_else(|| name.strip_suffix(".EXE"))
        .unwrap_or(name);
    stem.eq_ignore_ascii_case(expected)
}

/// Open a FITS frame, tagging any read error with the caller's role.
pub(crate) fn open_frame(path: &Path, role: &str) -> Result<FitsFrame> {
    FitsFrame::open(path).with_context(|| format!("failed to read {role} {}", path.display()))
}

/// Report a written artifact on stdout as `wrote {path}: {summary}`.
pub(crate) fn wrote(path: &Path, summary: Arguments<'_>) {
    println!("wrote {}: {summary}", path.display());
}

/// Validate the shared `--max-registration-*` flags with friendly messages.
pub(crate) fn validate_registration_flags(
    max_registration_rms: f64,
    max_registration_drift: f64,
    max_registration_drift_fraction: f64,
) -> Result<()> {
    if !max_registration_rms.is_finite() || max_registration_rms <= 0.0 {
        anyhow::bail!("--max-registration-rms must be a positive finite number");
    }
    if !max_registration_drift.is_finite() || max_registration_drift <= 0.0 {
        anyhow::bail!("--max-registration-drift must be a positive finite number");
    }
    if !max_registration_drift_fraction.is_finite()
        || !(0.0..=1.0).contains(&max_registration_drift_fraction)
    {
        anyhow::bail!("--max-registration-drift-fraction must be between zero and one");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Command {
        #[command(flatten)]
        gpu: GpuArgs,
    }

    fn options(args: &[&str]) -> seiza_stacking::GpuOptions {
        Command::try_parse_from(std::iter::once("seiza").chain(args.iter().copied()))
            .unwrap()
            .gpu
            .options()
    }

    #[test]
    fn gpu_flags_give_the_options_they_name() {
        assert_eq!(options(&[]), seiza_stacking::GpuOptions::default());
        assert_eq!(options(&["--no-gpu"]), seiza_stacking::GpuOptions::cpu());
        assert_eq!(
            options(&[
                "--cpu-pass",
                "debayer,resampling",
                "--gpu-concurrency",
                "2",
                "--gpu-memory-mib",
                "512",
                "--gpu-trace",
            ]),
            seiza_stacking::GpuOptions {
                debayer: false,
                resampling: false,
                concurrency: 2,
                memory_bytes: Some(512 << 20),
                trace: true,
                ..seiza_stacking::GpuOptions::default()
            }
        );
        assert!(Command::try_parse_from(["seiza", "--gpu-concurrency", "0"]).is_err());
    }
}
