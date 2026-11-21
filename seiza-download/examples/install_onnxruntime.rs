//! Install the runtime through the same native API used by C and Python hosts.
use seiza_download::OnnxRuntimeInstaller;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::args_os()
        .nth(1)
        .ok_or("usage: install_onnxruntime DIRECTORY")?;
    let installer = OnnxRuntimeInstaller::new(directory)?;
    let runtime = installer.install(|_| {}, || false).await?;
    println!("{}", runtime.library.display());
    Ok(())
}
