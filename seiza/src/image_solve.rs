//! Plate-solving an image file: reading it, finding its stars, and solving
//! with what it says about itself.
//!
//! [`solve_image`] is what the C ABI's `seiza_solve_image_json` and
//! Python's `solve_image` do with a path. FITS and XISF images go through
//! `seiza_fits::FitsImage::detection_u8` and are measured again on their own
//! samples ([`crate::detect::refine_centroids`]); PNG, JPEG and TIFF images
//! are detected as decoded. The file's prior ([`crate::prior`]) and capture
//! time steer the solve.

use crate::blind::{BlindIndex, BlindParams};
use crate::catalog::{DatedCatalog, ProperMotions, TileCatalog};
use crate::detect::{
    DetectBackend, DetectConfig, DetectedStar, detect_stars, detect_stars_luma_f32,
    refine_centroids,
};
use crate::prior::{PriorSolution, SolvePrior, SolveSource, solve_with_prior};
use crate::solve::{Solution, SolveLimits};
use image::DynamicImage;
use seiza_fits::{FitsImage, StretchParams};
use std::path::Path;

/// How [`solve_image`] solves.
#[derive(Clone, Debug)]
pub struct ImageSolveOptions<'a> {
    /// Where the catalogs are: a directory, or `None` for the standard
    /// places (see [`crate::data_paths`]). The deepest star catalog there is
    /// used, with the blind index beside it.
    pub catalog_directory: Option<&'a Path>,
    /// The pixel scales a blind solve tries, arcseconds per pixel.
    pub min_scale_arcsec_px: f64,
    pub max_scale_arcsec_px: f64,
    /// SIP distortion order of the solution, 0 to 5.
    pub sip_order: u8,
}

/// An image [`solve_image`] solved.
pub struct SolvedImage {
    pub width: u32,
    pub height: u32,
    /// The stars the solution came from, brightest first, in the image's
    /// pixels.
    pub stars: Vec<DetectedStar>,
    /// The star catalog it was solved against, as written.
    pub catalog: TileCatalog,
    /// When the image was taken, ISO 8601 in UTC, if its headers or EXIF say.
    pub capture_time: Option<String>,
    pub solution: Solution,
    /// Where the solution came from.
    pub source: SolveSource,
}

/// Blind hypotheses the first detection of an image spends when a second
/// detection follows, as the CLI's `--detection-fallback-hypotheses`.
const FIRST_DETECTION_HYPOTHESES: usize = 64;

/// Solve the image at `path`, a FITS, XISF or raster file, with what the
/// file says about itself (see [`crate::prior`]): its own WCS checked, else
/// a hinted solve around its pointing and scale, else blind; and, when
/// nothing solves, its own WCS unchecked.
///
/// FITS and XISF images are detected first on an MTF stretch, as the CLI
/// does, which suits processed images, binned 2 × 2 for a raw colour mosaic
/// or a mono frame over 20 megapixels, and again on their full-resolution
/// linear samples when that does not solve. Either way the stars found are
/// measured again on the linear samples, for the positions the solve fits.
/// A raster converted from 8-bit colour is detected again in float. A
/// raster's EXIF focal length gives the first scale to try. Only when the
/// second detection also fails does an unchecked header WCS stand. When the
/// star file has a proper-motion sidecar (`stars-gaia.pm.bin` beside
/// `stars-gaia.bin`) and the image a capture time, the solve searches the
/// catalog moved to that date.
pub fn solve_image(
    path: &Path,
    options: &ImageSolveOptions<'_>,
    limits: &SolveLimits,
) -> Result<SolvedImage, crate::Error> {
    /// How to detect again when the first detection does not solve.
    enum Retry {
        None,
        Linear(FitsImage),
        Float(DynamicImage),
    }
    let detection_config = DetectConfig {
        max_stars: 600,
        ..Default::default()
    };
    let read_error = |error: seiza_image_io::Error| crate::Error::Solve(error.to_string());
    let (width, height, first_stars, retry, capture_time, prior) = if is_astronomy_image(path) {
        let read = seiza_image_io::read_astronomy(path).map_err(read_error)?;
        let prior = read.solve_prior();
        let image = read.image;
        let width = u32::try_from(image.width)
            .map_err(|_| crate::Error::Solve("image width is too large".into()))?;
        let height = u32::try_from(image.height)
            .map_err(|_| crate::Error::Solve("image height is too large".into()))?;
        let capture_time = seiza_image_io::capture_time_from_headers(&image.headers);
        let stars = detect_on_stretch(&image, &detection_config);
        (
            width,
            height,
            stars,
            Retry::Linear(image),
            capture_time,
            prior,
        )
    } else {
        let image = seiza_image_io::open_raster(path)
            .map_err(read_error)?
            .pixels;
        let (width, height) = (image.width(), image.height());
        let stars = detect_stars(&image, &detection_config);
        let metadata = seiza_image_io::raster::PhotoMetadata::read(path);
        let prior = SolvePrior {
            scale_arcsec_px: metadata
                .scale_hint((width, height))
                .map(|hint| hint.nominal_arcsec_per_pixel),
            ..SolvePrior::default()
        };
        let retry = if is_converted_8bit_color(&image) {
            Retry::Float(image)
        } else {
            Retry::None
        };
        (
            width,
            height,
            stars,
            retry,
            metadata.capture_time_utc,
            prior,
        )
    };

    let data_error =
        |error: crate::data_paths::DataPathError| crate::Error::Catalog(error.to_string());
    let star_path = crate::data_paths::star_data(options.catalog_directory).map_err(data_error)?;
    let index_path = crate::data_paths::blind_index_beside(options.catalog_directory, &star_path)
        .map_err(data_error)?
        .ok_or_else(|| {
            crate::Error::Catalog(
                "no blind index found; install a complete Seiza catalog bundle first".into(),
            )
        })?;
    let catalog = TileCatalog::open(&star_path).map_err(|error| {
        crate::Error::Catalog(format!("failed to open {}: {error}", star_path.display()))
    })?;
    // With proper motions beside the star file and a capture time, the
    // solve searches the catalog as it was when the image was taken.
    let motions = ProperMotions::open_beside(&star_path, &catalog).map_err(|error| {
        crate::Error::Catalog(format!(
            "failed to open the proper motions beside {}: {error}",
            star_path.display()
        ))
    })?;
    let epoch = capture_time
        .as_deref()
        .and_then(crate::field_objects::capture_julian_date)
        .map(crate::catalog::julian_year);
    let dated = DatedCatalog::new(&catalog, motions.as_ref(), epoch);
    let index = BlindIndex::open(&index_path).map_err(|error| {
        crate::Error::Catalog(format!("failed to open {}: {error}", index_path.display()))
    })?;

    let params = BlindParams {
        min_scale_arcsec_px: options.min_scale_arcsec_px.max(0.01),
        max_scale_arcsec_px: options
            .max_scale_arcsec_px
            .max(options.min_scale_arcsec_px.max(0.01)),
        index_mag_limit: index.index_mag_limit(),
        max_pattern_deg: index.max_pattern_deg(),
        sip_order: options.sip_order.min(5),
        ..Default::default()
    };
    let dimensions = (width, height);
    let solve = |stars: &[DetectedStar], params: &BlindParams| {
        solve_with_prior(
            stars,
            &dated,
            Some(&index),
            &prior,
            params,
            dimensions,
            limits,
        )
    };
    let unchecked = |found: &PriorSolution| found.source == SolveSource::HeaderUnchecked;

    // With a second detection to come, the first spends fewer blind
    // hypotheses, as the CLI's does.
    let first_params = match retry {
        Retry::None => params.clone(),
        _ => BlindParams {
            max_hypotheses: params.max_hypotheses.min(FIRST_DETECTION_HYPOTHESES),
            ..params.clone()
        },
    };
    let first = solve(&first_stars, &first_params);
    let (found, stars) = match (first, retry) {
        (Err(error), _) if crate::solve::is_stop(&error) => return Err(error),
        (Ok(found), _) if !unchecked(&found) => (found, first_stars),
        (first, Retry::None) => (first?, first_stars),
        (first, retry) => {
            let stars = match retry {
                Retry::Linear(image) => refine_on_linear_samples(
                    &image,
                    detect_stars_luma_f32(&image.to_luma_f32(), width, height, &detection_config),
                ),
                Retry::Float(image) => detect_stars(
                    &image,
                    &DetectConfig {
                        backend: DetectBackend::F32,
                        ..detection_config
                    },
                ),
                Retry::None => unreachable!("handled above"),
            };
            match (solve(&stars, &params), first) {
                (Err(error), _) if crate::solve::is_stop(&error) => return Err(error),
                (Ok(found), _) if !unchecked(&found) => (found, stars),
                // Neither detection solved: the header's own WCS stands.
                (Ok(found), _) | (Err(_), Ok(found)) => (found, stars),
                (Err(error), Err(_)) => return Err(error),
            }
        }
    };
    Ok(SolvedImage {
        width,
        height,
        stars,
        catalog,
        capture_time,
        solution: found.solution,
        source: found.source,
    })
}

/// Whether `path` names a FITS or XISF file, by its extension.
fn is_astronomy_image(path: &Path) -> bool {
    seiza_image_io::Format::from_path(path).is_some_and(seiza_image_io::Format::is_astronomy)
}

/// A raster converted from 8-bit colour, whose float luma can order faint
/// stars differently from its 8-bit luma.
fn is_converted_8bit_color(image: &DynamicImage) -> bool {
    matches!(
        image,
        DynamicImage::ImageLumaA8(_) | DynamicImage::ImageRgb8(_) | DynamicImage::ImageRgba8(_)
    )
}

/// Stars found in `image`'s 8-bit stretch, binned 2 × 2 for a raw colour
/// mosaic or a large mono frame (`FitsImage::detection_u8`), placed in the
/// image's own pixels and measured again on its samples.
pub fn detect_on_stretch(image: &FitsImage, config: &DetectConfig) -> Vec<DetectedStar> {
    let (pixels, detection_width, detection_height, binning) =
        image.detection_u8(&StretchParams::default());
    let config = DetectConfig {
        ignore_border: config.ignore_border.div_ceil(binning as u32),
        ..config.clone()
    };
    let mut stars =
        image::GrayImage::from_raw(detection_width as u32, detection_height as u32, pixels)
            .map(|gray| detect_stars(&DynamicImage::ImageLuma8(gray), &config))
            .unwrap_or_default();
    // From the binned grid to image pixels: a sample covers `binning`
    // pixels each way, centred half a block in.
    let (bin, centre) = (binning as f64, (binning as f64 - 1.0) / 2.0);
    for star in &mut stars {
        star.x = star.x * bin + centre;
        star.y = star.y * bin + centre;
        star.area *= (binning * binning) as u32;
    }
    refine_on_linear_samples(image, stars)
}

/// Measure `stars` again on `image`'s linear samples, for the positions a
/// solve fits; see [`refine_centroids`].
pub fn refine_on_linear_samples(
    image: &FitsImage,
    mut stars: Vec<DetectedStar>,
) -> Vec<DetectedStar> {
    let sampler = image.linear_sampler();
    refine_centroids(
        &mut stars,
        sampler.width(),
        sampler.height(),
        sampler.binning(),
        &|x, y| sampler.get(x, y),
    );
    stars
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::TileSetBuilder;
    use crate::wcs::Wcs;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// A 16-bit FITS image, `pixels` in storage order, with no position or
    /// scale in its header.
    fn fits_bytes(width: usize, height: usize, pixels: &[u16]) -> Vec<u8> {
        let card = |text: String| {
            let mut card = text.into_bytes();
            card.resize(80, b' ');
            card
        };
        let mut bytes = Vec::new();
        for (keyword, value) in [
            ("SIMPLE", "T".to_string()),
            ("BITPIX", "16".to_string()),
            ("NAXIS", "2".to_string()),
            ("NAXIS1", width.to_string()),
            ("NAXIS2", height.to_string()),
            ("BZERO", "32768".to_string()),
            ("BSCALE", "1".to_string()),
        ] {
            bytes.extend(card(format!("{keyword:<8}= {value:>20}")));
        }
        bytes.extend(card("END".to_string()));
        bytes.resize(bytes.len().next_multiple_of(2880), b' ');
        for &value in pixels {
            bytes.extend_from_slice(&(value ^ 0x8000).to_be_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(2880), 0);
        bytes
    }

    #[test]
    fn an_image_file_solves_against_the_catalogs_in_a_directory() {
        let dir = tempfile::tempdir().unwrap();
        let mut rng = Lcg(0x501E);
        let center = (150.3, 33.4);
        let mut builder = TileSetBuilder::new(180, 2025.5, "test");
        let mut stars = Vec::new();
        for _ in 0..6000 {
            let ra = center.0 + (rng.next() - 0.5) * 16.0;
            let dec = center.1 + (rng.next() - 0.5) * 12.0;
            let mag = 4.0 + rng.next() * 8.5;
            builder.add(ra, dec, mag as f32);
            stars.push((ra, dec, mag));
        }
        let star_path = dir.path().join("stars-gaia.bin");
        builder.write_to(&star_path).unwrap();
        let catalog = TileCatalog::open(&star_path).unwrap();
        BlindIndex::build(&catalog, &BlindParams::default())
            .write_to(&dir.path().join("blind-gaia16.idx"))
            .unwrap();
        assert!(catalog.star_count() > 0);

        // A 1200 x 900 frame at 12"/px, rotated 25°, its stars Gaussian.
        let (width, height) = (1200usize, 900usize);
        let truth = Wcs::from_center_scale_rotation(center, (599.5, 449.5), 12.0, 25.0, false);
        let mut sky = vec![1000.0f64; width * height];
        for &(ra, dec, mag) in &stars {
            let Some((x, y)) = truth.world_to_pixel(ra, dec) else {
                continue;
            };
            if x < -5.0 || y < -5.0 || x > width as f64 + 5.0 || y > height as f64 + 5.0 {
                continue;
            }
            let peak = 40_000.0 * 10f64.powf(-0.4 * (mag - 4.0));
            let (x0, y0) = (x.round() as i64, y.round() as i64);
            for py in (y0 - 6).max(0)..(y0 + 7).min(height as i64) {
                for px in (x0 - 6).max(0)..(x0 + 7).min(width as i64) {
                    let d2 = (px as f64 - x).powi(2) + (py as f64 - y).powi(2);
                    sky[py as usize * width + px as usize] += peak * (-d2 / 4.5).exp();
                }
            }
        }
        let pixels: Vec<u16> = sky
            .iter()
            .map(|&value| value.min(65_535.0) as u16)
            .collect();
        let image_path = dir.path().join("field.fits");
        std::fs::write(&image_path, fits_bytes(width, height, &pixels)).unwrap();

        let solved = solve_image(
            &image_path,
            &ImageSolveOptions {
                catalog_directory: Some(dir.path()),
                min_scale_arcsec_px: 5.0,
                max_scale_arcsec_px: 30.0,
                sip_order: 0,
            },
            &SolveLimits::default(),
        )
        .unwrap();
        assert_eq!((solved.width, solved.height), (1200, 900));
        assert_eq!(solved.source, SolveSource::Blind);
        assert!(
            solved.solution.matched_stars >= 20,
            "{}",
            solved.solution.matched_stars
        );
        for (x, y) in [(0.0, 0.0), (1199.0, 0.0), (599.5, 449.5), (0.0, 899.0)] {
            let (ra, dec) = solved.solution.wcs.pixel_to_world(x, y);
            let (true_ra, true_dec) = truth.pixel_to_world(x, y);
            let off = crate::catalog::angular_separation_deg(ra, dec, true_ra, true_dec) * 3600.0;
            assert!(off < 3.0, "({x}, {y}) is {off:.2}\" off");
        }
    }
}
