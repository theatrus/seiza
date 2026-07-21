//! Generate the C header (`include/seiza_cabi.h`) from the crate's FFI surface
//! with cbindgen, so the header can never drift from the Rust source.
//!
//! The header is committed and consumed by the Windows (.NET) and macOS (Swift)
//! apps, which do not build this crate. `write_to_file` only rewrites the file
//! when the generated content changes, so ordinary builds leave a clean working
//! tree; CI fails if a source change was not accompanied by a regenerated header
//! (see the `lint` job's header drift check).

use std::path::Path;

fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set");
    // Regenerate when the FFI surface or the cbindgen config changes.
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=build.rs");

    let config = cbindgen::Config::from_root_or_default(&crate_dir);
    match cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(Path::new(&crate_dir).join("include/seiza_cabi.h"));
        }
        // Do not fail the build on a generation error; the committed header
        // remains in place and the CI drift check will surface any mismatch.
        Err(error) => println!("cargo:warning=cbindgen header generation failed: {error}"),
    }
}
