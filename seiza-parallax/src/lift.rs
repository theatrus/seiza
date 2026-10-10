//! Lifting extended objects, such as galaxies, out of the starless image.
//!
//! A galaxy in the field lies millions of parsecs beyond everything else,
//! but a star remover leaves it in the starless image, where it would grow
//! with the nebula as the camera closes in. Lifted onto a sprite of its own
//! it holds still like the farthest stars, and the nebula behind it is
//! filled in from around it.

use crate::light::LightImage;
use crate::scene::Sprite;
use rayon::prelude::*;

/// An object's ellipse in image pixels, as a catalog places it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Extent {
    pub x: f64,
    pub y: f64,
    pub semi_major: f64,
    pub semi_minor: f64,
    /// Rotation of the major axis from +x toward +y, radians.
    pub angle: f64,
}

impl Extent {
    /// How many times the ellipse's size `(px, py)` lies from its centre:
    /// 1 on the ellipse.
    pub fn reach(&self, px: f64, py: f64) -> f64 {
        let (dx, dy) = (px - self.x, py - self.y);
        let (sin, cos) = self.angle.sin_cos();
        let (u, v) = (dx * cos + dy * sin, -dx * sin + dy * cos);
        ((u / self.semi_major.max(1.0)).powi(2) + (v / self.semi_minor.max(1.0)).powi(2)).sqrt()
    }

    /// The point `scale` times the ellipse's size out at parameter `t`.
    fn point(&self, scale: f64, t: f64) -> (f64, f64) {
        let (sin, cos) = self.angle.sin_cos();
        let (u, v) = (
            scale * self.semi_major.max(1.0) * t.cos(),
            scale * self.semi_minor.max(1.0) * t.sin(),
        );
        (self.x + u * cos - v * sin, self.y + u * sin + v * cos)
    }
}

/// The scale, in catalog ellipses, the profile is followed out to.
const FARTHEST: f64 = 6.0;

/// Lift the object `extent` outlines out of `starless` onto a sprite at
/// `distance_pc`, or `None` when it does not stand out from its
/// surroundings.
///
/// A catalog's ellipse marks a galaxy's bright body, and a stretched image
/// shows it further out, so its edge is taken where the light, ring by
/// elliptical ring, has fallen nearly to its surroundings, and the lift
/// tapers off beyond it. There the nebula is filled in from just outside,
/// weighted by nearness, and the object's light above the fill becomes the
/// sprite: the sprite and the filled image add back to `starless`.
pub fn lift_object(starless: &mut LightImage, extent: &Extent, distance_pc: f64) -> Option<Sprite> {
    let (width, height) = (starless.width, starless.height);
    // Catalog places can be tens of pixels out, more than enough to leave
    // half a galaxy behind, so the ellipse first moves onto the light.
    let extent = &recentred(starless, extent);
    let ring = |scale: f64| ring_median(starless, extent, scale);
    let scales: Vec<f64> = (1..=(FARTHEST * 4.0) as usize)
        .map(|step| step as f64 / 4.0)
        .collect();
    let profile: Vec<Option<f32>> = scales.iter().map(|&scale| ring(scale)).collect();
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
    // The edge: where the light stands within a tenth of the surroundings'
    // own light above them. A stretched image carries a bright galaxy's
    // halo far out, so the edge is set by the surroundings, not the core.
    let edge = scales
        .iter()
        .zip(&profile)
        .find(|(scale, light)| {
            **scale >= 1.0 && light.is_some_and(|light| light - floor <= 0.1 * floor + 0.01)
        })
        .map_or(FARTHEST, |(scale, _)| *scale);
    // The lift tapers off beyond the edge, where little is left to take.
    const TAPER: f64 = 1.3;

    // The fill comes from a ring just outside the taper.
    let around = 2.0 * std::f64::consts::PI * edge * 1.4 * extent.semi_major.max(1.0);
    let steps = (around as usize).clamp(32, 512);
    let samples: Vec<(f64, f64, [f32; 3])> = (0..steps)
        .filter_map(|step| {
            let t = step as f64 / steps as f64 * std::f64::consts::TAU;
            let (px, py) = extent.point(edge * 1.4, t);
            (px >= 0.0 && py >= 0.0 && px <= width as f64 - 1.0 && py <= height as f64 - 1.0).then(
                || {
                    (
                        px,
                        py,
                        starless.at(px.round() as usize, py.round() as usize),
                    )
                },
            )
        })
        .collect();
    if samples.is_empty() {
        return None;
    }

    // The ellipse's box at the taper's end, inside the image.
    let reach = edge * TAPER * extent.semi_major.max(extent.semi_minor).max(1.0);
    let left = (extent.x - reach).floor().max(0.0) as usize;
    let top = (extent.y - reach).floor().max(0.0) as usize;
    let right = ((extent.x + reach).ceil() as usize + 1).min(width);
    let bottom = ((extent.y + reach).ceil() as usize + 1).min(height);
    if left >= right || top >= bottom {
        return None;
    }
    let mut image = LightImage::new(right - left, bottom - top);
    let rows: Vec<Vec<[f32; 3]>> = (top..bottom)
        .into_par_iter()
        .map(|y| {
            (left..right)
                .map(|x| {
                    let out = extent.reach(x as f64, y as f64) / edge;
                    if out >= TAPER {
                        return [0.0; 3];
                    }
                    // Full weight to the edge, easing to none past it.
                    let weight = if out <= 1.0 {
                        1.0
                    } else {
                        let s = (TAPER - out) / (TAPER - 1.0);
                        (s * s * (3.0 - 2.0 * s)) as f32
                    };
                    let (mut sum, mut total) = ([0.0_f64; 3], 0.0_f64);
                    for &(sx, sy, light) in &samples {
                        let near = 1.0 / ((sx - x as f64).powi(2) + (sy - y as f64).powi(2) + 1.0);
                        for (sum, light) in sum.iter_mut().zip(light) {
                            *sum += near * light as f64;
                        }
                        total += near;
                    }
                    let here = starless.at(x, y);
                    let mut lifted = [0.0_f32; 3];
                    for channel in 0..3 {
                        let fill = (sum[channel] / total) as f32;
                        lifted[channel] = weight * (here[channel] - fill).max(0.0);
                    }
                    lifted
                })
                .collect()
        })
        .collect();
    for (row, y) in rows.into_iter().zip(top..bottom) {
        for (lifted, x) in row.into_iter().zip(left..right) {
            image.pixels[(y - top) * image.width + (x - left)] = lifted;
            let pixel = &mut starless.pixels[y * width + x];
            for (value, lifted) in pixel.iter_mut().zip(lifted) {
                *value -= lifted;
            }
        }
    }
    Some(Sprite::new(
        left,
        top,
        image,
        extent.x,
        extent.y,
        distance_pc,
    ))
}

/// The median light, summed over channels, on the ellipse `scale` times
/// `extent`'s size, or `None` when the ring lies mostly off the image.
fn ring_median(light: &LightImage, extent: &Extent, scale: f64) -> Option<f32> {
    let (width, height) = (light.width as f64, light.height as f64);
    let around = 2.0 * std::f64::consts::PI * scale * extent.semi_major.max(1.0);
    let steps = (around as usize).clamp(16, 2048);
    let mut values: Vec<f32> = (0..steps)
        .filter_map(|step| {
            let t = step as f64 / steps as f64 * std::f64::consts::TAU;
            let (px, py) = extent.point(scale, t);
            (px >= 0.0 && py >= 0.0 && px <= width - 1.0 && py <= height - 1.0).then(|| {
                let pixel = light.at(px.round() as usize, py.round() as usize);
                pixel[0] + pixel[1] + pixel[2]
            })
        })
        .collect();
    if values.len() < steps / 3 {
        return None;
    }
    let middle = values.len() / 2;
    Some(*values.select_nth_unstable_by(middle, f32::total_cmp).1)
}

/// `extent` moved to the centroid of the light standing above its
/// surroundings within one and a half times its size, a few times over.
fn recentred(light: &LightImage, extent: &Extent) -> Extent {
    let mut extent = *extent;
    for _ in 0..4 {
        let Some(floor) = ring_median(light, &extent, 3.0) else {
            break;
        };
        let reach = 1.5 * extent.semi_major.max(extent.semi_minor).max(1.0);
        let (left, right) = (
            (extent.x - reach).floor().max(0.0) as usize,
            ((extent.x + reach).ceil().max(0.0) as usize).min(light.width - 1),
        );
        let (top, bottom) = (
            (extent.y - reach).floor().max(0.0) as usize,
            ((extent.y + reach).ceil().max(0.0) as usize).min(light.height - 1),
        );
        let (mut sx, mut sy, mut total) = (0.0_f64, 0.0_f64, 0.0_f64);
        for y in top..=bottom {
            for x in left..=right {
                if extent.reach(x as f64, y as f64) > 1.5 {
                    continue;
                }
                let pixel = light.at(x, y);
                let above = (pixel[0] + pixel[1] + pixel[2] - floor).max(0.0) as f64;
                sx += above * x as f64;
                sy += above * y as f64;
                total += above;
            }
        }
        if total <= 0.0 {
            break;
        }
        let moved = (sx / total - extent.x, sy / total - extent.y);
        extent.x += moved.0;
        extent.y += moved.1;
        if moved.0.hypot(moved.1) < 0.5 {
            break;
        }
    }
    extent
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_galaxy_lifts_off_its_nebula_and_adds_back() {
        // A sloping nebula with an elliptical galaxy on it, wider than its
        // catalog ellipse.
        let (width, height) = (200, 160);
        let extent = Extent {
            x: 90.0,
            y: 70.0,
            semi_major: 10.0,
            semi_minor: 6.0,
            angle: 0.5,
        };
        let nebula = |x: usize, y: usize| 0.2 + 0.001 * x as f32 + 0.0005 * y as f32;
        let mut image = LightImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let r = extent.reach(x as f64, y as f64);
                let galaxy = 1.5 * (-(r * r) / 2.0).exp() as f32;
                let light = nebula(x, y) + galaxy;
                image.pixels[y * width + x] = [light, 0.9 * light, 0.8 * light];
            }
        }
        let original = image.clone();
        let sprite = lift_object(&mut image, &extent, 1e8).expect("the galaxy stands out");
        // The sprite and what is left add back to the image.
        let mut rebuilt = image.clone();
        for y in 0..sprite.image.height {
            for x in 0..sprite.image.width {
                let index = (sprite.top + y) * width + sprite.left + x;
                for (value, lifted) in rebuilt.pixels[index].iter_mut().zip(sprite.image.at(x, y)) {
                    *value += lifted;
                }
            }
        }
        for (rebuilt, original) in rebuilt.pixels.iter().zip(&original.pixels) {
            assert!((rebuilt[0] - original[0]).abs() < 1e-5);
        }
        // The galaxy is gone from what is left: its centre is the nebula.
        let centre = image.at(90, 70)[0];
        assert!((centre - nebula(90, 70)).abs() < 0.05, "{centre}");
        // And the sprite holds nearly all of its light.
        let lifted: f32 = sprite.image.pixels.iter().map(|pixel| pixel[0]).sum();
        let total: f32 = (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .map(|(x, y)| {
                let r = extent.reach(x as f64, y as f64);
                1.5 * (-(r * r) / 2.0).exp() as f32
            })
            .sum();
        assert!(
            lifted > 0.9 * total && lifted < 1.1 * total,
            "{lifted} of {total}"
        );
    }

    #[test]
    fn a_misplaced_catalog_ellipse_moves_onto_the_galaxy() {
        let (width, height) = (200, 160);
        let galaxy = Extent {
            x: 100.0,
            y: 80.0,
            semi_major: 10.0,
            semi_minor: 9.0,
            angle: 0.0,
        };
        let mut image = LightImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let r = galaxy.reach(x as f64, y as f64);
                image.pixels[y * width + x] = [0.2 + (-(r * r) / 2.0).exp() as f32; 3];
            }
        }
        // The catalog has it eight pixels off.
        let catalog = Extent {
            x: 108.0,
            y: 74.0,
            ..galaxy
        };
        let sprite = lift_object(&mut image, &catalog, 1e8).expect("the galaxy stands out");
        assert!((sprite.x - 100.0).abs() < 1.0 && (sprite.y - 80.0).abs() < 1.0);
        // Nothing of it is left behind on the far side.
        let left = image.at(88, 86)[0];
        assert!((left - 0.2).abs() < 0.03, "{left}");
    }

    #[test]
    fn flat_ground_lifts_nothing() {
        let mut image = LightImage::new(100, 100);
        image.pixels.iter_mut().for_each(|pixel| *pixel = [0.3; 3]);
        let extent = Extent {
            x: 50.0,
            y: 50.0,
            semi_major: 8.0,
            semi_minor: 8.0,
            angle: 0.0,
        };
        assert!(lift_object(&mut image, &extent, 1e8).is_none());
    }
}
