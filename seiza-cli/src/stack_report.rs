use anyhow::{Context, Result, ensure};
use clap::Args;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Args)]
pub struct StackReportArgs {
    /// Ordered labeled JSON files, e.g. --stack S=s.json --stack H=h.json
    #[arg(long = "stack", required = true, value_name = "LABEL=PATH")]
    stacks: Vec<String>,
    /// Collection title
    #[arg(long, default_value = "Completed stacks")]
    title: String,
    /// New self-contained HTML file
    #[arg(long)]
    html: PathBuf,
    /// Optional new combined JSON file
    #[arg(long)]
    json: Option<PathBuf>,
}

pub fn run(args: StackReportArgs) -> Result<()> {
    let mut stacks = Vec::new();
    for input in &args.stacks {
        let (label, path) = input
            .split_once('=')
            .context("--stack must be LABEL=PATH")?;
        let mut value: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(path).with_context(|| format!("reading {path}"))?,
        )
        .with_context(|| format!("parsing {path}"))?;
        // The stack CLI preserves its legacy provenance envelope. Bindings emit the report directly.
        if value.get("post_stack").is_some() {
            value = value["post_stack"].take();
        }
        let report =
            serde_json::from_value(value).with_context(|| format!("post-stack report {path}"))?;
        stacks.push(seiza_analysis::NamedStackReport {
            label: label.into(),
            report,
        });
    }
    let collection = seiza_analysis::StackReportCollection {
        schema_version: 1,
        title: args.title,
        stacks,
    };
    let html = seiza_analysis::render_stack_collection_html(&collection)?;
    let json = serde_json::to_vec_pretty(&collection)?;
    let mut outputs = vec![(&args.html, html.as_bytes())];
    if let Some(path) = &args.json {
        ensure!(
            !seiza_stacking::paths_refer_to_same_file(path, &args.html),
            "HTML and JSON destinations must differ"
        );
        outputs.push((path, json.as_slice()));
    }
    // Refuse existing files, including input aliases. Persist without clobbering even in a race.
    for (path, _) in &outputs {
        ensure!(
            !path.exists(),
            "report destination already exists: {}",
            path.display()
        );
    }
    let mut prepared = Vec::new();
    for (path, bytes) in outputs {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let mut file = tempfile::NamedTempFile::new_in(parent)?;
        file.write_all(bytes)?;
        seiza_fits::sync_file(file.as_file())?;
        prepared.push((path, file));
    }
    for (path, file) in prepared {
        file.persist_noclobber(path)
            .with_context(|| format!("writing {}", path.display()))?;
    }
    eprintln!(
        "Rendered {} independent stacks: {}",
        collection.stacks.len(),
        args.html.display()
    );
    Ok(())
}
