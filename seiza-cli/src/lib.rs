use anyhow::{Context, Result};

pub mod worker;

/// Open an image file; FITS files are MTF-autostretched to 8-bit grayscale.
pub fn load_image(path: &std::path::Path) -> Result<image::DynamicImage> {
    let is_fits = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("fits") || extension.eq_ignore_ascii_case("fit")
        });
    if is_fits {
        let fits = seiza_fits::FitsImage::open(path)
            .map_err(|error| anyhow::anyhow!("{}: {error}", path.display()))?;
        let stretched = fits.stretch_to_u8(&seiza_fits::StretchParams::default());
        let buffer = image::GrayImage::from_raw(fits.width as u32, fits.height as u32, stretched)
            .ok_or_else(|| anyhow::anyhow!("FITS dimensions mismatch"))?;
        return Ok(image::DynamicImage::ImageLuma8(buffer));
    }
    image::open(path).with_context(|| format!("failed to open {}", path.display()))
}
