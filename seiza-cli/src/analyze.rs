use anyhow::{Context, Result, bail};
use clap::Args;
use std::io::Write;
use std::path::PathBuf;

#[derive(Args)]
pub struct AnalyzeArgs {
    /// Files, directories, or .txt lists (relative entries resolve beside the list)
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    /// Filter header to include (case insensitive)
    #[arg(long)]
    filter: Option<String>,
    /// Detector samples per component, rounded down to 256-sample patches
    #[arg(long, default_value_t = 262_144)]
    samples: usize,
    /// Measure frame quality without registration or noise-depth analysis
    #[arg(long)]
    metrics_only: bool,
    /// First N sorted, deduplicated paths before header filtering
    #[arg(long)]
    max_frames: Option<usize>,
    /// JSON destination (default: project-analysis.json)
    #[arg(long, default_value = "project-analysis.json")]
    json: PathBuf,
    /// Portable HTML destination (default: project-analysis.html)
    #[arg(long, default_value = "project-analysis.html")]
    html: PathBuf,
}
pub fn run(args: AnalyzeArgs) -> Result<()> {
    // Create-new semantics protect inputs and existing reports, including aliases.
    if args.json.exists() || args.html.exists() {
        bail!("report destinations must not already exist; choose new --json and --html paths");
    }
    if seiza_stacking::paths_refer_to_same_file(&args.json, &args.html) {
        bail!("JSON and HTML destinations must differ");
    }
    let report = seiza_analysis::analyze(
        &args.inputs,
        &seiza_analysis::AnalysisOptions {
            samples: args.samples,
            metrics_only: args.metrics_only,
            filter: args.filter,
            max_frames: args.max_frames,
        },
    )?;
    let json = serde_json::to_vec_pretty(&report)?;
    let html = seiza_analysis::render_html(&report)?;
    for (path, data) in [(&args.json, json.as_slice()), (&args.html, html.as_bytes())] {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(std::path::Path::new("."));
        let mut file = tempfile::NamedTempFile::new_in(parent)?;
        file.write_all(data)?;
        file.as_file().sync_all()?;
        file.persist_noclobber(path)
            .with_context(|| format!("writing {}", path.display()))?;
    }
    eprintln!(
        "Analyzed {} sources ({} component records) in {} filter/component groups. Reports: {}, {}",
        report
            .frames
            .iter()
            .map(|f| &f.path)
            .collect::<std::collections::HashSet<_>>()
            .len(),
        report.frames.len(),
        report.groups.len(),
        args.json.display(),
        args.html.display()
    );
    if report.groups.is_empty() {
        bail!("no usable frames; reports contain the input errors");
    }
    Ok(())
}
