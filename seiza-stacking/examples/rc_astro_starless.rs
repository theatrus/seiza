//! Round-trip a real frame through RC-Astro's StarXTerminator.
//!
//! Usage: `cargo run -p seiza-stacking --example rc_astro_starless -- <frame> [out-dir]`
//!
//! Needs an installed, licensed `rc-astro` on PATH. Writes `starless.fits`
//! and `stars.fits` into the output directory (default: the current one).

use seiza_stacking::{
    ExternalParameterValue, ExternalToolRequest, FitsFrame, RcAstroCli,
    write_processed_image_fits_f32,
};

fn main() {
    let mut arguments = std::env::args().skip(1);
    let input = arguments
        .next()
        .expect("usage: rc_astro_starless <frame> [out-dir]");
    let out_dir = std::path::PathBuf::from(arguments.next().unwrap_or_else(|| ".".into()));

    let cli = RcAstroCli::locate().expect("rc-astro not found on PATH");
    let schema = cli.tool_schema("sxt").expect("sxt schema");
    println!(
        "{} {} (contract v{}, ML{}): {}",
        schema.name,
        schema.cli_version,
        schema.schema_version,
        schema.ml_version.unwrap_or(0),
        schema
            .license_message
            .as_deref()
            .unwrap_or("license state unknown"),
    );

    let frame = FitsFrame::open(std::path::Path::new(&input)).expect("open frame");
    println!(
        "frame: {}x{}x{}",
        frame.image.width, frame.image.height, frame.image.channels
    );

    let request = ExternalToolRequest {
        tool: "sxt".into(),
        parameters: vec![("stars".into(), ExternalParameterValue::Bool(true))],
        device: None,
    };
    let started = std::time::Instant::now();
    let processed = cli
        .process_image(
            &schema,
            &request,
            &frame.image,
            &frame.headers,
            None,
            &mut |f| {
                print!("\rprocessing {:3.0}%", f * 100.0);
                use std::io::Write;
                let _ = std::io::stdout().flush();
            },
        )
        .expect("sxt run");
    println!(
        "\ndone in {:.1}s on {}",
        started.elapsed().as_secs_f32(),
        processed.device.as_deref().unwrap_or("?")
    );

    write_processed_image_fits_f32(
        out_dir.join("starless.fits"),
        &processed.image,
        &frame.headers,
        &[],
    )
    .expect("write starless");
    if let Some(stars) = &processed.stars {
        write_processed_image_fits_f32(out_dir.join("stars.fits"), stars, &frame.headers, &[])
            .expect("write stars");
    }
    println!("wrote {}", out_dir.join("starless.fits").display());
}
