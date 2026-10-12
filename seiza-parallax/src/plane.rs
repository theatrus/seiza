//! Depth planes: regions of the starless image at distances of their own.
//!
//! The starless image is one plane at the target's distance, but a field
//! often holds clouds at other depths: a bright nebula in front of the
//! target, or one far behind it. A depth plane lifts such a region off the
//! starless image, as a galaxy is lifted, and the camera moves it at its
//! own depth. What lay behind it is filled in from the light around it, and
//! the plane and the filled image add back to the starless image, so the
//! first frame is still the photograph.
//!
//! A plane can block light as well as give it: where its region is darker
//! than what fills in behind it, a dark cloud lies there, and the plane
//! keeps the share of light it lets through. Everything behind it then
//! dims or brightens as the cloud slides across it.
//!
//! A [`PlaneMask`] marks a plane's region as an image a person can paint:
//! white where the image belongs to the plane, black where it does not,
//! and grays for a soft edge.

use crate::dust::{OPAQUE, open_unit_image, same_shape};
use crate::lift::{Extent, recentred, ring_median};
use crate::light::{LightImage, Pyramid, floor64};
use crate::scene::Sprite;
use rayon::prelude::*;
use std::path::Path;

/// Light along the image's edge above this share of a plane's mean light
/// over its body cuts the plane off there; below it, the plane has faded
/// out before the edge.
const EDGE_LIGHT: f32 = 0.05;

/// The dark pixels a region keeps about the pixels it takes, so a plane
/// cut from it fades to nothing before its edges.
const BORDER: usize = 2;

/// The scale, in catalogue ellipses, an object's light is followed out to
/// for its region.
const FARTHEST: f64 = 3.0;

/// Past its edge, a region drawn from an ellipse fades out over this much
/// more of its size.
const TAPER: f64 = 1.3;

/// A region's light within this share of what fills in behind it, either
/// way, is the background's own variation; below it, a cloud blocks light,
/// and above it, the plane gives light of its own.
const NEAR_FILL: f32 = 0.95;

/// The least share of light a cloud lets through: black dust lets through
/// next to nothing, and the light restored behind it stays near the fill.
const LEAST_PASSED: f32 = 0.01;

/// A depth plane's region as an image: its share of each pixel, 1 where the
/// image belongs to the plane and 0 where it does not, row by row. It spans
/// the whole image at any size of the same shape.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneMask {
    pub width: usize,
    pub height: usize,
    pub coverage: Vec<f32>,
}

impl PlaneMask {
    /// The mask in the image at `path`, white in the plane, its format
    /// chosen by its extension: PNG, JPEG, TIFF, FITS or XISF. A colour
    /// image gives each pixel's mean channel. Integer samples read as 0 to
    /// 1; float ones must already lie there, give or take a rounding.
    pub fn open(path: &Path) -> Result<Self, String> {
        let (width, height, coverage) =
            open_unit_image(path, "plane mask", "0 (not in the plane) and 1 (in it)")?;
        Ok(Self {
            width,
            height,
            coverage,
        })
    }

    /// Whether it is a mask: a size, a share for every pixel, each 0 to 1.
    pub fn check(&self) -> Result<(), String> {
        if self.width == 0 || self.height == 0 {
            return Err("the plane mask is empty".into());
        }
        if self.coverage.len() != self.width * self.height {
            return Err(format!(
                "the plane mask has {} values for {}×{} pixels",
                self.coverage.len(),
                self.width,
                self.height
            ));
        }
        if !self
            .coverage
            .iter()
            .all(|share| (0.0..=1.0).contains(share))
        {
            return Err("the plane mask's values must lie between 0 and 1".into());
        }
        Ok(())
    }

    /// Write the mask to `path`, white in the plane, in the format its
    /// extension names: a 16-bit grayscale PNG or TIFF, or a 32-bit float
    /// FITS or XISF.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        self.check()?;
        let image =
            seiza_image_io::FloatImage::gray(self.width, self.height, self.coverage.clone());
        seiza_image_io::save(path, &image).map_err(|error| error.to_string())
    }

    /// The share at mask point `(x, y)`, between pixel centres, and the
    /// edge pixels' beyond them.
    fn sample(&self, x: f64, y: f64) -> f32 {
        let x = x.clamp(0.0, (self.width - 1) as f64);
        let y = y.clamp(0.0, (self.height - 1) as f64);
        // Both are clamped at zero, so truncating rounds them down.
        let (column, row) = (x as usize, y as usize);
        let (next_column, next_row) = (
            (column + 1).min(self.width - 1),
            (row + 1).min(self.height - 1),
        );
        let (tx, ty) = ((x - column as f64) as f32, (y - row as f64) as f32);
        let value = |column: usize, row: usize| self.coverage[row * self.width + column];
        let top = value(column, row) * (1.0 - tx) + value(next_column, row) * tx;
        let bottom = value(column, next_row) * (1.0 - tx) + value(next_column, next_row) * tx;
        top * (1.0 - ty) + bottom * ty
    }
}

/// A depth plane as a video's options give it: its region, its distance
/// and a name to report it by.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthPlane {
    pub mask: PlaneMask,
    /// Parsecs.
    pub distance_pc: f64,
    pub name: String,
}

/// The pixels a plane takes, at the image's size: a box, and the plane's
/// share of each pixel in it, row by row.
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    /// The box's top-left pixel in the image, and its size.
    pub left: usize,
    pub top: usize,
    pub width: usize,
    pub height: usize,
    pub coverage: Vec<f32>,
}

impl Region {
    /// The region `mask` marks over a `width` × `height` image, `None`
    /// when it marks nothing, or why it cannot: a mask of another shape.
    /// Each pixel takes the mask's share at its centre, between the mask's
    /// own pixels.
    pub fn from_mask(
        mask: &PlaneMask,
        width: usize,
        height: usize,
    ) -> Result<Option<Self>, String> {
        mask.check()?;
        if width == 0 || height == 0 {
            return Err("the image is empty".into());
        }
        if !same_shape((mask.width, mask.height), (width, height)) {
            return Err(format!(
                "the plane mask is {}×{}, not the shape of the {width}×{height} image",
                mask.width, mask.height
            ));
        }
        // The mask pixels that mark anything.
        let (mut low, mut high) = ((usize::MAX, usize::MAX), (0, 0));
        for (index, &share) in mask.coverage.iter().enumerate() {
            if share > 0.0 {
                let (x, y) = (index % mask.width, index / mask.width);
                low = (low.0.min(x), low.1.min(y));
                high = (high.0.max(x), high.1.max(y));
            }
        }
        if low.0 == usize::MAX {
            return Ok(None);
        }
        // Mask pixels per image pixel, each way. Image pixel `x` takes the
        // mask's share at `(x + 0.5) · across − 0.5`, so the image pixels
        // within a mask pixel of those marked, and a pixel more, hold all
        // of the plane.
        let (across, down) = (
            mask.width as f64 / width as f64,
            mask.height as f64 / height as f64,
        );
        let first = |marked: usize, scale: f64| {
            (((marked as f64 - 1.0) / scale).floor() - 1.0).max(0.0) as usize
        };
        let last = |marked: usize, scale: f64, size: usize| {
            ((((marked as f64 + 2.0) / scale).ceil() + 1.0) as usize).min(size)
        };
        let (left, right) = (first(low.0, across), last(high.0, across, width));
        let (top, bottom) = (first(low.1, down), last(high.1, down, height));
        let coverage: Vec<f32> = (top..bottom)
            .into_par_iter()
            .flat_map_iter(|y| {
                (left..right).map(move |x| {
                    mask.sample(
                        (x as f64 + 0.5) * across - 0.5,
                        (y as f64 + 0.5) * down - 0.5,
                    )
                })
            })
            .collect();
        let region = Self {
            left,
            top,
            width: right - left,
            height: bottom - top,
            coverage,
        };
        Ok(region.trimmed((width, height)))
    }

    /// The region of the extended object whose catalogue ellipse is
    /// `extent`, sized by the light of `starless`: moved onto the light,
    /// then out to where its light, ring by elliptical ring, comes near its
    /// surroundings' or stops falling, at most [`FARTHEST`] times the
    /// ellipse, and fading out over [`TAPER`] times that. A catalogue's
    /// ellipse often marks only a nebula's bright body. `None` when the
    /// object does not stand out from its surroundings, or lies mostly off
    /// the image.
    pub fn from_extent(starless: &LightImage, extent: &Extent) -> Option<Self> {
        Self::from_profile(starless, &recentred(starless, extent)?, 1.0)
    }

    /// [`Self::from_extent`] for a dark cloud: out to where the image,
    /// darkest at its middle, brightens to near its surroundings or stops
    /// brightening. The ellipse stays where the catalogue puts it, as
    /// moving it onto the light would take it off the cloud.
    pub fn from_dark_extent(starless: &LightImage, extent: &Extent) -> Option<Self> {
        Self::from_profile(starless, extent, -1.0)
    }

    /// The region about `extent` out to where its light, times `sign`,
    /// ring by ring, comes near its surroundings' or levels off.
    fn from_profile(starless: &LightImage, extent: &Extent, sign: f32) -> Option<Self> {
        let scales: Vec<f64> = (1..=(FARTHEST * 4.0) as usize)
            .map(|step| step as f64 / 4.0)
            .collect();
        let profile: Vec<Option<f32>> = scales
            .iter()
            .map(|&scale| ring_median(starless, extent, scale).map(|light| sign * light))
            .collect();
        let core = profile[0]?;
        let floor = profile
            .iter()
            .zip(&scales)
            .filter(|(_, scale)| **scale >= 1.0)
            .filter_map(|(light, _)| *light)
            .fold(f32::INFINITY, f32::min);
        if !floor.is_finite() || core - floor < 0.02 {
            return None;
        }
        let rise = core - floor;
        // From the ellipse out, the first ring near the floor, or where the
        // light has levelled off once it has fallen half way: a nebula
        // within a wider glow stops there, not at the darker sky beyond the
        // glow, and a flat-topped one does not stop on its top.
        let edge = scales
            .iter()
            .enumerate()
            .filter(|(_, scale)| **scale >= 1.0)
            .find(|&(index, _)| {
                let Some(light) = profile[index] else {
                    return true;
                };
                let next = profile.get(index + 1).copied().flatten();
                light - floor <= 0.1 * rise
                    || (light - floor <= 0.5 * rise
                        && next.is_some_and(|next| light - next <= 0.02 * rise))
            })
            .map_or(FARTHEST, |(_, scale)| *scale);
        let (width, height) = (starless.width, starless.height);
        let reach = edge * TAPER * extent.semi_major.max(extent.semi_minor).max(1.0);
        let left = (extent.x - reach).floor().max(0.0) as usize;
        let top = (extent.y - reach).floor().max(0.0) as usize;
        let right = (((extent.x + reach).ceil() + 1.0).max(0.0) as usize).min(width);
        let bottom = (((extent.y + reach).ceil() + 1.0).max(0.0) as usize).min(height);
        if left >= right || top >= bottom {
            return None;
        }
        let coverage = (top..bottom)
            .into_par_iter()
            .flat_map_iter(|y| {
                (left..right).map(move |x| {
                    let out = extent.reach(x as f64, y as f64) / edge;
                    if out <= 1.0 {
                        1.0
                    } else if out >= TAPER {
                        0.0
                    } else {
                        let t = (TAPER - out) / (TAPER - 1.0);
                        (t * t * (3.0 - 2.0 * t)) as f32
                    }
                })
            })
            .collect();
        Self {
            left,
            top,
            width: right - left,
            height: bottom - top,
            coverage,
        }
        .trimmed((width, height))
    }

    /// The region inside `outlines`, closed polygons in image pixels of a
    /// `width` × `height` image, a point being inside when an odd number of
    /// them hold it, with its edge softened over a thirtieth of its size,
    /// at least 3 pixels; `None` when it covers none of the image.
    pub fn from_outlines(
        outlines: &[Vec<(f64, f64)>],
        width: usize,
        height: usize,
    ) -> Option<Self> {
        let (mut low, mut high) = (
            (f64::INFINITY, f64::INFINITY),
            (f64::NEG_INFINITY, f64::NEG_INFINITY),
        );
        for &(x, y) in outlines.iter().flatten() {
            low = (low.0.min(x), low.1.min(y));
            high = (high.0.max(x), high.1.max(y));
        }
        if !(low.0.is_finite() && high.0.is_finite() && low.1.is_finite() && high.1.is_finite()) {
            return None;
        }
        let soften = ((high.0 - low.0).max(high.1 - low.1) / 30.0).max(3.0);
        let pad = 2.0 * soften + 2.0;
        let clamp = |value: f64, size: usize| (value.max(0.0) as usize).min(size);
        let (left, right) = (
            clamp((low.0 - pad).floor(), width),
            clamp((high.0 + pad).ceil() + 1.0, width),
        );
        let (top, bottom) = (
            clamp((low.1 - pad).floor(), height),
            clamp((high.1 + pad).ceil() + 1.0, height),
        );
        if left >= right || top >= bottom {
            return None;
        }
        let (columns, rows) = (right - left, bottom - top);
        let mut coverage = vec![0.0_f32; columns * rows];
        // Each row's crossings with the outlines' sides, in order: the
        // pixels whose centres lie between alternate pairs are inside.
        coverage
            .par_chunks_mut(columns)
            .enumerate()
            .for_each(|(row, pixels)| {
                let y = (top + row) as f64;
                let mut crossings: Vec<f64> = Vec::new();
                for outline in outlines {
                    let closing = outline.last().zip(outline.first());
                    for (a, b) in outline.iter().zip(outline.iter().skip(1)).chain(closing) {
                        if (a.1 <= y) != (b.1 <= y) {
                            crossings.push(a.0 + (y - a.1) * (b.0 - a.0) / (b.1 - a.1));
                        }
                    }
                }
                crossings.sort_by(f64::total_cmp);
                for pair in crossings.chunks_exact(2) {
                    let from = ((pair[0] - left as f64).ceil().max(0.0) as usize).min(columns);
                    let to = ((pair[1] - left as f64).ceil().max(0.0) as usize).min(columns);
                    for pixel in &mut pixels[from..to.max(from)] {
                        *pixel = 1.0;
                    }
                }
            });
        // Three box blurs come close to a Gaussian.
        let radius = (soften / 2.0).round().max(1.0) as usize;
        for _ in 0..3 {
            box_blur(&mut coverage, columns, rows, radius);
        }
        Self {
            left,
            top,
            width: columns,
            height: rows,
            coverage,
        }
        .trimmed((width, height))
    }

    /// The region as a mask the size of a `width` × `height` image, as
    /// [`Self::from_mask`] reads it back.
    pub fn to_mask(&self, width: usize, height: usize) -> PlaneMask {
        let coverage = (0..width * height)
            .into_par_iter()
            .map(|index| self.at(index % width, index / width))
            .collect();
        PlaneMask {
            width,
            height,
            coverage,
        }
    }

    /// The region cut down to the pixels it takes a share of, with a dark
    /// border of [`BORDER`] pixels inside a `size` image, or `None` when it
    /// takes none.
    fn trimmed(&self, (width, height): (usize, usize)) -> Option<Self> {
        let (mut low, mut high) = ((usize::MAX, usize::MAX), (0, 0));
        for (index, &share) in self.coverage.iter().enumerate() {
            if share > 0.0 {
                let (x, y) = (index % self.width, index / self.width);
                low = (low.0.min(x), low.1.min(y));
                high = (high.0.max(x), high.1.max(y));
            }
        }
        if low.0 == usize::MAX {
            return None;
        }
        let left = (self.left + low.0).saturating_sub(BORDER);
        let top = (self.top + low.1).saturating_sub(BORDER);
        let right = (self.left + high.0 + 1 + BORDER).min(width);
        let bottom = (self.top + high.1 + 1 + BORDER).min(height);
        let coverage = (top..bottom)
            .flat_map(|y| (left..right).map(move |x| self.at(x, y)))
            .collect();
        Some(Self {
            left,
            top,
            width: right - left,
            height: bottom - top,
            coverage,
        })
    }

    /// The plane's share of image pixel `(x, y)`, 0 outside the box.
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        if x < self.left || y < self.top {
            return 0.0;
        }
        let (column, row) = (x - self.left, y - self.top);
        if column >= self.width || row >= self.height {
            return 0.0;
        }
        self.coverage[row * self.width + column]
    }

    /// The plane's share of the pixel nearest image point `(x, y)`.
    pub fn covers(&self, x: f64, y: f64) -> f32 {
        if !(x >= -0.5 && y >= -0.5) {
            return 0.0;
        }
        self.at(x.round() as usize, y.round() as usize)
    }

    /// How many pixels' worth the plane takes in all.
    pub fn area(&self) -> f64 {
        self.coverage.iter().map(|&share| share as f64).sum()
    }
}

/// The share of the light behind it a plane lets through, each channel's,
/// over a box of the image, row by row; all of it outside the box.
#[derive(Clone, Debug, PartialEq)]
pub struct Transmission {
    /// The box's top-left pixel in the image, and its size.
    pub left: usize,
    pub top: usize,
    pub width: usize,
    pub height: usize,
    pub values: Vec<[f32; 3]>,
}

impl Transmission {
    /// The share let through at image point `(x, y)`, between pixel
    /// centres; all of it outside the box.
    #[inline]
    pub fn at(&self, x: f64, y: f64) -> [f32; 3] {
        let (fx, fy) = (x - self.left as f64, y - self.top as f64);
        if !(fx > -1.0 && fy > -1.0 && fx < self.width as f64 && fy < self.height as f64) {
            return [1.0; 3];
        }
        let (x0, y0) = (floor64(fx), floor64(fy));
        let (tx, ty) = ((fx - x0 as f64) as f32, (fy - y0 as f64) as f32);
        let value = |dx: isize, dy: isize| {
            let (column, row) = (x0 + dx, y0 + dy);
            if column < 0 || row < 0 || column as usize >= self.width || row as usize >= self.height
            {
                [1.0; 3]
            } else {
                self.values[row as usize * self.width + column as usize]
            }
        };
        let (a, b, c, d) = (value(0, 0), value(1, 0), value(0, 1), value(1, 1));
        std::array::from_fn(|channel| {
            let top = a[channel] + (b[channel] - a[channel]) * tx;
            let bottom = c[channel] + (d[channel] - c[channel]) * tx;
            top + (bottom - top) * ty
        })
    }

    /// Whether it hides everything behind it at image point `(x, y)`, every
    /// channel letting through [`OPAQUE`] or less.
    pub fn is_opaque(&self, x: f64, y: f64) -> bool {
        let passed = self.at(x, y);
        passed[0].max(passed[1]).max(passed[2]) <= OPAQUE
    }
}

/// A region lifted off the starless image onto a plane at a distance of its
/// own, which grows and slides as the camera moves, as the background does.
#[derive(Clone, Debug)]
pub struct Plane {
    pub name: String,
    /// The plane image's top-left pixel in the image.
    pub left: usize,
    pub top: usize,
    pub image: Pyramid,
    /// Parsecs.
    pub distance_pc: f64,
    /// The plane's share of each pixel: what a label or a star there sits
    /// on.
    pub region: Region,
    /// The light the plane lets through from behind it, where it is a dark
    /// cloud; `None` where it blocks nothing.
    pub transmission: Option<Transmission>,
    /// Whether it gives any light of its own, rather than only blocking it.
    pub lit: bool,
}

impl Plane {
    /// The plane `image` makes with its top-left pixel at image pixel
    /// `(left, top)`.
    pub fn new(
        name: impl Into<String>,
        (left, top): (usize, usize),
        image: LightImage,
        distance_pc: f64,
        region: Region,
    ) -> Self {
        let lit = image
            .pixels
            .iter()
            .any(|pixel| pixel.iter().any(|&value| value > 0.0));
        Self {
            name: name.into(),
            left,
            top,
            image: Pyramid::down_to_a_pixel(image),
            distance_pc,
            region,
            transmission: None,
            lit,
        }
    }

    /// The plane's light at full size.
    pub fn base(&self) -> &LightImage {
        self.image.base()
    }

    /// Which edges of a `width` × `height` image cut the plane's light
    /// off, left, top, right and bottom: those along which it holds more
    /// than [`EDGE_LIGHT`] of its mean light over its body, the pixels its
    /// region takes half or more of. Elsewhere its light fades out before
    /// the edge, and moving it uncovers nothing.
    pub fn cut_edges(&self, width: usize, height: usize) -> [bool; 4] {
        let base = self.base();
        let light = |x: usize, y: usize| -> f32 {
            if x < self.left || y < self.top {
                return 0.0;
            }
            let (column, row) = (x - self.left, y - self.top);
            if column >= base.width || row >= base.height {
                return 0.0;
            }
            let pixel = base.at(column, row);
            pixel[0] + pixel[1] + pixel[2]
        };
        let region = &self.region;
        let (mut sum, mut count) = (0.0_f64, 0_usize);
        for y in region.top..region.top + region.height {
            for x in region.left..region.left + region.width {
                if region.at(x, y) >= 0.5 {
                    sum += light(x, y) as f64;
                    count += 1;
                }
            }
        }
        let floor = EDGE_LIGHT * (sum / count.max(1) as f64) as f32;
        let (right, bottom) = (self.left + base.width, self.top + base.height);
        let column = |x: usize| (self.top..bottom).any(|y| light(x, y) > floor);
        let row = |y: usize| (self.left..right).any(|x| light(x, y) > floor);
        [
            column(0),
            row(0),
            column(width.saturating_sub(1)),
            row(height.saturating_sub(1)),
        ]
    }

    /// Add `sprites`' light to the plane, growing its image to hold them.
    pub(crate) fn take(&mut self, sprites: &[&Sprite]) {
        let base = self.base();
        let (mut left, mut top) = (self.left, self.top);
        let (mut right, mut bottom) = (self.left + base.width, self.top + base.height);
        for sprite in sprites {
            left = left.min(sprite.left);
            top = top.min(sprite.top);
            right = right.max(sprite.left + sprite.image.width);
            bottom = bottom.max(sprite.top + sprite.image.height);
        }
        let mut image = LightImage::new(right - left, bottom - top);
        add(&mut image, (left, top), base, (self.left, self.top));
        for sprite in sprites {
            add(
                &mut image,
                (left, top),
                &sprite.image,
                (sprite.left, sprite.top),
            );
        }
        (self.left, self.top) = (left, top);
        self.image = Pyramid::down_to_a_pixel(image);
        self.lit = true;
    }
}

/// Add `light`, whose top-left pixel is image pixel `from`, onto `image`,
/// whose top-left pixel is image pixel `at`; `image` must hold it.
fn add(image: &mut LightImage, at: (usize, usize), light: &LightImage, from: (usize, usize)) {
    for y in 0..light.height {
        for x in 0..light.width {
            let index = (from.1 + y - at.1) * image.width + from.0 + x - at.0;
            for (target, light) in image.pixels[index].iter_mut().zip(light.at(x, y)) {
                *target += light;
            }
        }
    }
}

/// Lift the light `region` marks off `starless` onto a plane named `name`
/// at `distance_pc`, or `None` when nothing there stands apart from what
/// fills in behind it.
///
/// What lies behind the region is filled in from the light around it.
/// Where the region is brighter than the fill, past [`NEAR_FILL`] of it,
/// the plane takes, in its share of each pixel, the light above that.
/// Where it is darker, a cloud blocks the light behind: the plane keeps the
/// share it lets through, each channel's, and `starless` keeps the light
/// behind, restored. Either way the plane drawn over what is left gives
/// back `starless`.
pub fn lift_region(
    starless: &mut LightImage,
    region: &Region,
    distance_pc: f64,
    name: &str,
) -> Option<Plane> {
    let (width, height) = (starless.width, starless.height);
    // The fill draws on the light about the region, out to half its size
    // again.
    let margin = (region.width.max(region.height) / 2).max(16);
    let left = region.left.saturating_sub(margin);
    let top = region.top.saturating_sub(margin);
    let right = (region.left + region.width + margin).min(width);
    let bottom = (region.top + region.height + margin).min(height);
    let fill = filled(starless, (left, top, right - left, bottom - top), |x, y| {
        1.0 - region.at(x, y)
    });
    let mut image = LightImage::new(region.width, region.height);
    let mut passed = vec![[1.0_f32; 3]; region.width * region.height];
    {
        let starless = &mut *starless;
        let columns = region.width;
        starless
            .pixels
            .par_chunks_mut(width)
            .skip(region.top)
            .take(region.height)
            .zip(image.pixels.par_chunks_mut(columns))
            .zip(passed.par_chunks_mut(columns))
            .enumerate()
            .for_each(|(row, ((pixels, light), passed))| {
                let y = region.top + row;
                for column in 0..columns {
                    let share = region.coverage[row * columns + column];
                    if share <= 0.0 {
                        continue;
                    }
                    let x = region.left + column;
                    let here = &mut pixels[x];
                    let behind = fill.at(x - left, y - top);
                    for channel in 0..3 {
                        let bright = behind[channel] / NEAR_FILL;
                        if here[channel] > bright {
                            let above = share * (here[channel] - bright);
                            light[column][channel] = above;
                            here[channel] -= above;
                        } else if behind[channel] > 0.0 {
                            let ratio = here[channel] / behind[channel] / NEAR_FILL;
                            if ratio < 1.0 {
                                let through = 1.0 - share * (1.0 - ratio.max(LEAST_PASSED));
                                passed[column][channel] = through;
                                here[channel] /= through;
                            }
                        }
                    }
                }
            });
    }
    let lit = image
        .pixels
        .iter()
        .any(|pixel| pixel.iter().any(|&value| value > 0.0));
    let dark = passed
        .iter()
        .any(|through| through.iter().any(|&share| share < 1.0));
    if !lit && !dark {
        return None;
    }
    let mut plane = Plane::new(
        name,
        (region.left, region.top),
        image,
        distance_pc,
        region.clone(),
    );
    plane.transmission = dark.then_some(Transmission {
        left: region.left,
        top: region.top,
        width: region.width,
        height: region.height,
        values: passed,
    });
    Some(plane)
}

/// The light of `image` over the box `(left, top, width, height)`, each
/// pixel taken as `known(x, y)` known, 0 to 1, and what is not known filled
/// in from the known light about it: averaged into coarser and coarser
/// levels, each weighted by how much of it is known, then blended back
/// down, each level taking from the coarser one what it does not know
/// itself. This is push-pull (Gortler et al., 1996). Deep in a hole the
/// fill is smooth; near its edge, it keeps close to the light just outside.
fn filled(
    image: &LightImage,
    (left, top, width, height): (usize, usize, usize, usize),
    known: impl Fn(usize, usize) -> f32 + Sync,
) -> LightImage {
    let mut light = LightImage::new(width, height);
    let mut weight = vec![0.0_f32; width * height];
    light
        .pixels
        .par_chunks_mut(width)
        .zip(weight.par_chunks_mut(width))
        .enumerate()
        .for_each(|(row, (pixels, weights))| {
            for column in 0..width {
                let (x, y) = (left + column, top + row);
                pixels[column] = image.at(x, y);
                weights[column] = known(x, y).clamp(0.0, 1.0);
            }
        });
    // Each level's light where it is known, and how much of it is.
    let mut levels = vec![(light, weight)];
    // Push: each coarser pixel holds the light of the four beneath it,
    // weighted by how much of each is known, and is known as far as they
    // hold a whole pixel's worth between them.
    while let Some((light, weight)) = levels.last()
        && (light.width > 1 || light.height > 1)
    {
        let (columns, rows) = (light.width.div_ceil(2), light.height.div_ceil(2));
        let mut coarse = LightImage::new(columns, rows);
        let mut coarse_weight = vec![0.0_f32; columns * rows];
        coarse
            .pixels
            .par_chunks_mut(columns)
            .zip(coarse_weight.par_chunks_mut(columns))
            .enumerate()
            .for_each(|(row, (pixels, weights))| {
                for column in 0..columns {
                    let (mut sum, mut total) = ([0.0_f32; 3], 0.0_f32);
                    for (x, y) in [
                        (2 * column, 2 * row),
                        (2 * column + 1, 2 * row),
                        (2 * column, 2 * row + 1),
                        (2 * column + 1, 2 * row + 1),
                    ] {
                        if x < light.width && y < light.height {
                            let known = weight[y * light.width + x];
                            let value = light.at(x, y);
                            for channel in 0..3 {
                                sum[channel] += known * value[channel];
                            }
                            total += known;
                        }
                    }
                    if total > 0.0 {
                        pixels[column] = sum.map(|value| value / total);
                    }
                    weights[column] = total.min(1.0);
                }
            });
        levels.push((coarse, coarse_weight));
    }
    // Pull: from the coarsest down, each level takes the share of a pixel
    // it does not know from the level above, drawn up to its size.
    for index in (0..levels.len() - 1).rev() {
        let (finer, coarser) = levels.split_at_mut(index + 1);
        let (light, weight) = &mut finer[index];
        let above = &coarser[0].0;
        let columns = light.width;
        light
            .pixels
            .par_chunks_mut(columns)
            .zip(weight.par_chunks(columns))
            .enumerate()
            .for_each(|(row, (pixels, weights))| {
                for (column, pixel) in pixels.iter_mut().enumerate() {
                    let known = weights[column];
                    if known >= 1.0 {
                        continue;
                    }
                    // Pixel `i` of a level lies at `(i − 0.5) / 2` on the
                    // level above.
                    let from = sample_clamped(
                        above,
                        (column as f32 - 0.5) / 2.0,
                        (row as f32 - 0.5) / 2.0,
                    );
                    for channel in 0..3 {
                        pixel[channel] = known * pixel[channel] + (1.0 - known) * from[channel];
                    }
                }
            });
    }
    levels.swap_remove(0).0
}

/// Blur the `columns` × `rows` grid `values` by the mean of the
/// `2 · radius + 1` values about each, across then down, of those on the
/// grid.
fn box_blur(values: &mut [f32], columns: usize, rows: usize, radius: usize) {
    let pass = |line: &mut [f32], scratch: &mut Vec<f32>| {
        scratch.clear();
        scratch.extend_from_slice(line);
        let mut sum: f64 = scratch[..radius.min(line.len())]
            .iter()
            .map(|&value| value as f64)
            .sum();
        for index in 0..line.len() {
            if index + radius < line.len() {
                sum += scratch[index + radius] as f64;
            }
            if index > radius {
                sum -= scratch[index - radius - 1] as f64;
            }
            let first = index.saturating_sub(radius);
            let last = (index + radius).min(line.len() - 1);
            line[index] = (sum / (last - first + 1) as f64) as f32;
        }
    };
    values
        .par_chunks_mut(columns)
        .for_each_init(Vec::new, |scratch, row| pass(row, scratch));
    let mut column = Vec::with_capacity(rows);
    let mut scratch = Vec::with_capacity(rows);
    for x in 0..columns {
        column.clear();
        column.extend((0..rows).map(|y| values[y * columns + x]));
        pass(&mut column, &mut scratch);
        for (y, &value) in column.iter().enumerate() {
            values[y * columns + x] = value;
        }
    }
}

/// Bilinear light of `image` at `(x, y)`, and the edge pixels' beyond them.
fn sample_clamped(image: &LightImage, x: f32, y: f32) -> [f32; 3] {
    image.sample(
        x.clamp(0.0, (image.width - 1) as f32),
        y.clamp(0.0, (image.height - 1) as f32),
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A `width` × `height` mask, 1 within `radius` of `(x, y)`.
    pub(crate) fn disc(width: usize, height: usize, (x, y): (f64, f64), radius: f64) -> PlaneMask {
        PlaneMask {
            width,
            height,
            coverage: (0..width * height)
                .map(|index| {
                    let (column, row) = ((index % width) as f64, (index / width) as f64);
                    if (column - x).hypot(row - y) <= radius {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
        }
    }

    #[test]
    fn a_cloud_lifts_off_its_nebula_and_adds_back() {
        // A sloping nebula with a soft cloud on it, and a mask about the
        // cloud.
        let (width, height) = (200, 160);
        let nebula = |x: usize, y: usize| 0.2 + 0.001 * x as f32 + 0.0005 * y as f32;
        let cloud = |x: usize, y: usize| {
            let r2 = (x as f64 - 90.0).powi(2) + (y as f64 - 70.0).powi(2);
            (0.8 * (-r2 / (2.0 * 10.0_f64.powi(2))).exp()) as f32
        };
        let mut image = LightImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let light = nebula(x, y) + cloud(x, y);
                image.pixels[y * width + x] = [light, 0.9 * light, 0.8 * light];
            }
        }
        let original = image.clone();
        let region = Region::from_mask(&disc(width, height, (90.0, 70.0), 35.0), width, height)
            .unwrap()
            .unwrap();
        let plane = lift_region(&mut image, &region, 150.0, "cloud").unwrap();
        assert_eq!((plane.name.as_str(), plane.distance_pc), ("cloud", 150.0));
        // The plane and what is left add back to the image.
        let mut rebuilt = image.clone();
        add(&mut rebuilt, (0, 0), plane.base(), (plane.left, plane.top));
        for (rebuilt, original) in rebuilt.pixels.iter().zip(&original.pixels) {
            for channel in 0..3 {
                assert!((rebuilt[channel] - original[channel]).abs() < 1e-5);
            }
        }
        // What is left under the cloud is the nebula, filled in from around
        // it, and the plane holds nearly all of the cloud: all but the
        // twentieth of the nebula's light the background may vary by.
        let centre = image.at(90, 70)[0];
        assert!((centre - nebula(90, 70)).abs() < 0.03, "{centre}");
        let lifted: f32 = plane.base().pixels.iter().map(|pixel| pixel[0]).sum();
        let total: f32 = (0..height)
            .flat_map(|y| (0..width).map(move |x| cloud(x, y)))
            .sum();
        assert!(
            lifted > 0.85 * total && lifted < 1.05 * total,
            "{lifted} of {total}"
        );
        // Nothing outside the mask moved.
        assert_eq!(image.at(10, 10), original.at(10, 10));
    }

    #[test]
    fn a_mask_spans_the_image_at_any_size_of_its_shape() {
        let (width, height) = (200, 160);
        // A fifth of the image's size: a disc of 6 mask pixels is one of 30
        // image pixels.
        let small = disc(40, 32, (19.5, 15.5), 6.0);
        let region = Region::from_mask(&small, width, height).unwrap().unwrap();
        assert_eq!(region.covers(99.5, 79.5), 1.0);
        assert_eq!(region.at(99, 120), 0.0);
        assert!((region.area() / (std::f64::consts::PI * 900.0) - 1.0).abs() < 0.1);
        // The box holds the disc and a dark border, and no more.
        assert!(region.left < 70 && region.left > 55, "{}", region.left);
        assert!(region.width < 75, "{}", region.width);
        // Another shape is refused, and a mask marking nothing gives none.
        let error = Region::from_mask(&disc(100, 100, (5.0, 5.0), 3.0), width, height);
        assert!(error.unwrap_err().contains("100×100"));
        let empty = disc(40, 32, (-50.0, -50.0), 3.0);
        assert_eq!(Region::from_mask(&empty, width, height), Ok(None));
    }

    #[test]
    fn a_plane_tells_which_image_edges_cut_its_light() {
        let (width, height) = (200, 160);
        // A glow running off the left edge, and one that fades out short
        // of the right.
        let glow = |cx: f64, spread: f64| {
            let mut image = LightImage::new(width, height);
            for (index, pixel) in image.pixels.iter_mut().enumerate() {
                let (x, y) = ((index % width) as f64, (index / width) as f64);
                let r2 = (x - cx).powi(2) + (y - 80.0).powi(2);
                *pixel = [0.1 + (0.8 * (-r2 / (2.0 * spread * spread)).exp()) as f32; 3];
            }
            image
        };
        let mut left = glow(5.0, 20.0);
        let region = Region::from_mask(&disc(width, height, (0.0, 80.0), 50.0), width, height)
            .unwrap()
            .unwrap();
        assert_eq!(region.left, 0);
        let plane = lift_region(&mut left, &region, 150.0, "left").unwrap();
        assert_eq!(plane.cut_edges(width, height), [true, false, false, false]);
        // The mask reaches the right edge, but the light is gone by then.
        let mut right = glow(165.0, 6.0);
        let region = Region::from_mask(&disc(width, height, (175.0, 80.0), 40.0), width, height)
            .unwrap()
            .unwrap();
        assert_eq!(region.left + region.width, width);
        let plane = lift_region(&mut right, &region, 150.0, "right").unwrap();
        assert_eq!(plane.cut_edges(width, height), [false; 4]);
    }

    #[test]
    fn an_ellipse_region_reaches_as_far_as_the_light_and_no_farther() {
        // A nebula with a halo well past its catalogued ellipse, sitting in
        // a wide, even glow with dark sky beyond it.
        let (width, height) = (400, 300);
        let mut image = LightImage::new(width, height);
        for (index, pixel) in image.pixels.iter_mut().enumerate() {
            let (x, y) = ((index % width) as f64, (index / width) as f64);
            let r = (x - 200.0).hypot(y - 150.0);
            let nebula = 1.0 * (-(r * r) / (2.0 * 18.0_f64.powi(2))).exp();
            let glow = if r < 120.0 { 0.2 } else { 0.05 };
            *pixel = [(nebula + glow) as f32; 3];
        }
        let extent = Extent {
            x: 200.0,
            y: 150.0,
            semi_major: 20.0,
            semi_minor: 20.0,
            angle: 0.0,
        };
        let region = Region::from_extent(&image, &extent).unwrap();
        // It takes the halo, out past the ellipse...
        assert_eq!(region.covers(200.0, 150.0), 1.0);
        assert!(region.covers(235.0, 150.0) > 0.9, "{region:?}");
        // ...but stops in the glow, not at the dark sky past it.
        assert_eq!(region.covers(290.0, 150.0), 0.0);
        assert!(region.width < 160, "{}", region.width);
        // A flat image gives none.
        let mut flat = LightImage::new(width, height);
        flat.pixels.fill([0.3; 3]);
        assert!(Region::from_extent(&flat, &extent).is_none());
    }

    #[test]
    fn an_outline_region_fills_its_polygons_with_a_soft_edge() {
        let (width, height) = (300, 200);
        // A square with a square hole, the hole drawn the other way round.
        let outer = vec![(50.0, 50.0), (150.0, 50.0), (150.0, 150.0), (50.0, 150.0)];
        let hole = vec![(80.0, 80.0), (80.0, 120.0), (120.0, 120.0), (120.0, 80.0)];
        let region = Region::from_outlines(&[outer.clone(), hole], width, height).unwrap();
        assert!(region.at(60, 60) > 0.99, "{}", region.at(60, 60));
        assert!(region.at(100, 100) < 0.01, "{}", region.at(100, 100));
        assert!(
            (region.at(50, 100) - 0.5).abs() < 0.1,
            "{}",
            region.at(50, 100)
        );
        assert_eq!(region.at(200, 100), 0.0);
        // Off the image, or touching it only past its edge, gives none.
        let away: Vec<(f64, f64)> = outer.iter().map(|&(x, y)| (x + 1000.0, y)).collect();
        assert!(Region::from_outlines(&[away], width, height).is_none());
        // Written as a mask and read back, it is the same region.
        let mask = region.to_mask(width, height);
        assert_eq!(Region::from_mask(&mask, width, height), Ok(Some(region)));
    }

    #[test]
    fn a_flat_region_lifts_nothing_and_a_dark_cloud_blocks_light() {
        let (width, height) = (100, 100);
        let mut image = LightImage::new(width, height);
        image.pixels.fill([0.3; 3]);
        let region = Region::from_mask(&disc(width, height, (50.0, 50.0), 20.0), width, height)
            .unwrap()
            .unwrap();
        assert!(lift_region(&mut image, &region, 150.0, "flat").is_none());
        // A dark cloud, redder than the glow it hides, with a black core.
        for (index, pixel) in image.pixels.iter_mut().enumerate() {
            let (x, y) = ((index % width) as f64, (index / width) as f64);
            let r = (x - 50.0).hypot(y - 50.0);
            if r < 3.0 {
                *pixel = [0.0; 3];
            } else if r < 10.0 {
                *pixel = [0.15, 0.1, 0.06];
            }
        }
        let original = image.clone();
        let plane = lift_region(&mut image, &region, 150.0, "dark").unwrap();
        assert!(!plane.lit);
        let passed = plane.transmission.clone().expect("it blocks light");
        // Behind it, the glow about it, near enough.
        let behind = image.at(56, 50);
        assert!(
            behind.iter().all(|&light| (light - 0.285).abs() < 0.01),
            "{behind:?}"
        );
        // It lets through more red than blue, none at its core, and all of
        // the light outside it.
        let at = passed.at(56.0, 50.0);
        assert!(at[0] > at[2] && at[2] < 0.3, "{at:?}");
        assert!(
            passed
                .at(50.0, 50.0)
                .iter()
                .all(|&share| (share - LEAST_PASSED).abs() < 1e-6)
        );
        assert!(passed.is_opaque(50.0, 50.0) && !passed.is_opaque(56.0, 50.0));
        assert_eq!(passed.at(80.0, 80.0), [1.0; 3]);
        // The light behind, dimmed by what it lets through, is the image.
        for y in 0..height {
            for x in 0..width {
                let through = passed.at(x as f64, y as f64);
                let (behind, photographed) = (image.at(x, y), original.at(x, y));
                for ((behind, through), photographed) in
                    behind.iter().zip(through).zip(photographed)
                {
                    assert!((behind * through - photographed).abs() < 1e-5, "({x}, {y})");
                }
            }
        }
        // The cloud's darkness, not light, sizes its region.
        let extent = Extent {
            x: 50.0,
            y: 50.0,
            semi_major: 4.0,
            semi_minor: 4.0,
            angle: 0.0,
        };
        let region = Region::from_dark_extent(&original, &extent).unwrap();
        assert_eq!(region.covers(50.0, 50.0), 1.0);
        assert!(region.covers(57.0, 50.0) > 0.9, "{region:?}");
        assert_eq!(region.covers(80.0, 50.0), 0.0);
        assert!(Region::from_extent(&original, &extent).is_none());
    }
}
