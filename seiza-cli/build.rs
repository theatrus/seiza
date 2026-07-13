fn main() {
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
