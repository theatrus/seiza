use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "seiza-server",
    about = "Warm HTTP plate-solving service for Seiza clients",
    version
)]
struct Cli {
    /// Star tile file kept open for the lifetime of the server
    #[arg(long)]
    data: PathBuf,
    /// Optional prebuilt blind pattern index kept open by the server
    #[arg(long)]
    index: Option<PathBuf>,
    /// HTTP listen address
    #[arg(long, default_value = "127.0.0.1:7878")]
    listen: String,
    /// Bearer token; defaults to SEIZA_SERVER_TOKEN when set
    #[arg(long)]
    token: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let token = cli
        .token
        .or_else(|| std::env::var("SEIZA_SERVER_TOKEN").ok());
    seiza_cli::worker::run_server(
        &cli.listen,
        &cli.data,
        cli.index.as_deref(),
        token.as_deref(),
    )
}
