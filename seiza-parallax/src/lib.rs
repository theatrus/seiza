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

pub mod encode;
pub mod find;
pub mod light;
pub mod render;
pub mod scene;

#[cfg(feature = "openh264")]
pub use encode::OpenH264Sink;
pub use encode::{FfmpegSink, FrameSink, PngSequence, VideoSettings};
pub use find::{FoundStar, find_stars};
pub use light::{LightImage, Pyramid};
pub use render::{Easing, Shot};
pub use scene::{CutOptions, Scene, Sprite, Star};
