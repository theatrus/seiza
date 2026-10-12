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
//! A [`PlaneMask`] marks a plane's region as an image a person can paint:
//! white where the image belongs to the plane, black where it does not,
//! and grays for a soft edge.

use crate::dust::{open_unit_image, same_shape};
use crate::light::{LightImage, Pyramid};
use crate::scene::Sprite;
use rayon::prelude::*;
use std::path::Path;

/// A region's share of a pixel on the image's edge above this cuts the
/// plane off there; below it, the mask has faded the plane out first.
const EDGE_SHARE: f32 = 0.01;

/// The dark pixels a region keeps about the pixels it takes, so a plane
/// cut from it fades to nothing before its edges.
const BORDER: usize = 2;

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

    /// Which edges of a `width` × `height` image the plane reaches, left,
    /// top, right and bottom: where its light ends at the image's edge
    /// rather than fading out within it.
    pub fn edges(&self, width: usize, height: usize) -> [bool; 4] {
        let column =
            |x: usize| (self.top..self.top + self.height).any(|y| self.at(x, y) > EDGE_SHARE);
        let row =
            |y: usize| (self.left..self.left + self.width).any(|x| self.at(x, y) > EDGE_SHARE);
        [
            column(0),
            row(0),
            column(width.saturating_sub(1)),
            row(height.saturating_sub(1)),
        ]
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
        Self {
            name: name.into(),
            left,
            top,
            image: Pyramid::down_to_a_pixel(image),
            distance_pc,
            region,
        }
    }

    /// The plane's light at full size.
    pub fn base(&self) -> &LightImage {
        self.image.base()
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
/// at `distance_pc`, or `None` when nothing there stands above what fills
/// in behind it.
///
/// What lies behind the region is filled in from the light around it, and
/// the plane takes, in its share of each pixel, the light above the fill:
/// the plane and what is left add back to `starless`. A dark cloud gives
/// the plane nothing, as the light it blocks cannot be lifted off.
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
    {
        let starless = &*starless;
        image
            .pixels
            .par_chunks_mut(region.width)
            .enumerate()
            .for_each(|(row, pixels)| {
                let y = region.top + row;
                for (column, pixel) in pixels.iter_mut().enumerate() {
                    let share = region.coverage[row * region.width + column];
                    if share <= 0.0 {
                        continue;
                    }
                    let x = region.left + column;
                    let here = starless.at(x, y);
                    let behind = fill.at(x - left, y - top);
                    for channel in 0..3 {
                        pixel[channel] = share * (here[channel] - behind[channel]).max(0.0);
                    }
                }
            });
    }
    let lifted: f64 = image
        .pixels
        .par_iter()
        .map(|pixel| (pixel[0] + pixel[1] + pixel[2]) as f64)
        .sum();
    if lifted <= 0.0 {
        return None;
    }
    add_scaled(starless, &image, (region.left, region.top), -1.0);
    Some(Plane::new(
        name,
        (region.left, region.top),
        image,
        distance_pc,
        region.clone(),
    ))
}

/// Add `scale` times `light`, whose top-left pixel is image pixel `at`, to
/// `image`.
fn add_scaled(image: &mut LightImage, light: &LightImage, at: (usize, usize), scale: f32) {
    let width = image.width;
    image
        .pixels
        .par_chunks_mut(width)
        .skip(at.1)
        .take(light.height)
        .enumerate()
        .for_each(|(row, pixels)| {
            for (column, pixel) in pixels[at.0..at.0 + light.width].iter_mut().enumerate() {
                let light = light.at(column, row);
                for channel in 0..3 {
                    pixel[channel] += scale * light[channel];
                }
            }
        });
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
        // it, and the plane holds nearly all of the cloud.
        let centre = image.at(90, 70)[0];
        assert!((centre - nebula(90, 70)).abs() < 0.03, "{centre}");
        let lifted: f32 = plane.base().pixels.iter().map(|pixel| pixel[0]).sum();
        let total: f32 = (0..height)
            .flat_map(|y| (0..width).map(move |x| cloud(x, y)))
            .sum();
        assert!(
            lifted > 0.9 * total && lifted < 1.1 * total,
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
        assert_eq!(region.edges(width, height), [false; 4]);
        // Another shape is refused, and a mask marking nothing gives none.
        let error = Region::from_mask(&disc(100, 100, (5.0, 5.0), 3.0), width, height);
        assert!(error.unwrap_err().contains("100×100"));
        let empty = disc(40, 32, (-50.0, -50.0), 3.0);
        assert_eq!(Region::from_mask(&empty, width, height), Ok(None));
    }

    #[test]
    fn a_region_tells_which_image_edges_cut_it() {
        let (width, height) = (200, 160);
        let mask = disc(width, height, (0.0, 80.0), 30.0);
        let region = Region::from_mask(&mask, width, height).unwrap().unwrap();
        assert_eq!(region.left, 0);
        assert_eq!(region.edges(width, height), [true, false, false, false]);
        // A soft edge that fades out before the image's edge does not cut.
        let mut soft = disc(width, height, (195.0, 80.0), 30.0);
        for (index, share) in soft.coverage.iter_mut().enumerate() {
            if index % width >= 198 {
                *share = 0.0;
            }
        }
        let region = Region::from_mask(&soft, width, height).unwrap().unwrap();
        assert_eq!(region.edges(width, height), [false; 4]);
    }

    #[test]
    fn a_flat_or_dark_region_lifts_nothing() {
        let (width, height) = (100, 100);
        let mut image = LightImage::new(width, height);
        image.pixels.fill([0.3; 3]);
        let region = Region::from_mask(&disc(width, height, (50.0, 50.0), 20.0), width, height)
            .unwrap()
            .unwrap();
        assert!(lift_region(&mut image, &region, 150.0, "flat").is_none());
        // A dark cloud: nothing stands above what fills in behind it.
        for (index, pixel) in image.pixels.iter_mut().enumerate() {
            let (x, y) = ((index % width) as f64, (index / width) as f64);
            if (x - 50.0).hypot(y - 50.0) < 10.0 {
                *pixel = [0.1; 3];
            }
        }
        assert!(lift_region(&mut image, &region, 150.0, "dark").is_none());
        assert_eq!(image.at(50, 50), [0.1; 3]);
    }
}
