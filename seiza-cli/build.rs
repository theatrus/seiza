fn main() {
    // Windows gives the main thread a 1 MB stack, and clap plus the large
    // command dispatch in an unoptimized build overflow it at startup. Ask
    // the MSVC linker for the 8 MB Linux and macOS give by default.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
    {
        println!("cargo:rustc-link-arg-bins=/STACK:8388608");
    }

    // N.I.N.A. reads the Windows FileVersion to gate ASTAP capabilities
    // (anything below 0.9.1.0 loses auto-downsample); report a high file
    // version and the real crate version as ProductVersion. Windows
    // binaries are built natively on Windows, so a host gate suffices.
    #[cfg(windows)]
    {
        let mut resource = winresource::WindowsResource::new();
        resource.set("FileVersion", "1.0.0.0");
        resource.set("ProductVersion", env!("CARGO_PKG_VERSION"));
        resource.set("ProductName", "seiza");
        resource.set(
            "FileDescription",
            "seiza plate solver (ASTAP-compatible mode)",
        );
        resource
            .set_version_info(winresource::VersionInfo::FILEVERSION, 0x0001_0000_0000_0000)
            .compile()
            .expect("windows resource");
    }
}
