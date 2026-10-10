//! Parallax fly-through videos of astrophotographs.
//!
//! A stretched image is split into its starless image and its stars (for
//! example with StarXTerminator). The starless image becomes a plane at the
//! target's distance, and each star is cut out and placed at its own Gaia
//! distance. A camera then trucks and dollies toward a point of the plane:
//! foreground stars slide past, background stars barely move, and stars the
//! camera flies by grow, brighten and fade.
//!
//! Layers combine with a screen blend, done as addition of "light" (see
//! [`light`]), so the first frame reproduces the original image.
//!
//! ```no_run
//! use seiza_parallax::{CutOptions, LightImage, Scene, Shot, Star};
//! # let (starless, stars) = (LightImage::new(64, 64), LightImage::new(64, 64));
//! let found = [Star { x: 30.0, y: 20.0, distance_pc: Some(136.0) }];
//! let scene = Scene::new(&starless, &stars, &found, 410.0, 1500.0, 3000.0, &CutOptions::default());
//! let shot = Shot { focus: (32.0, 32.0), ..Shot::default() };
//! let first = shot.render(&scene, 0).to_display_rgb8();
//! ```

pub mod dust;
pub mod encode;
pub mod lift;
pub mod light;
pub mod render;
pub mod scene;

pub use dust::Dust;
#[cfg(feature = "openh264")]
pub use encode::OpenH264Sink;
pub use encode::{FfmpegSink, FrameSink, PngSequence, VideoSettings};
pub use lift::{Extent, lift_object};
pub use light::{LightImage, Pyramid};
pub use render::{Easing, Shot, Start, View};
pub use scene::{CutOptions, Scene, SmallStars, Sprite, Star};
pub use seiza_stars::PeakStar;

use rayon::prelude::*;

/// Stars in a star image, brightest first, as the peaks of its light summed
/// over channels: [`seiza_stars::find_peak_stars`]. A peak must stand
/// `sigma` noise levels above the background.
pub fn find_stars(stars: &LightImage, sigma: f32) -> Vec<PeakStar> {
    let sum: Vec<f32> = stars
        .pixels
        .par_iter()
        .map(|pixel| pixel[0] + pixel[1] + pixel[2])
        .collect();
    seiza_stars::find_peak_stars(&sum, stars.width, stars.height, sigma)
}
