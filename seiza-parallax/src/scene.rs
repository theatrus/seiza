//! The layers a fly-through moves: the starless image as a plane at the
//! target's distance, and every star cut out of the star image at its own.

use crate::light::{LightImage, Pyramid};
use rayon::prelude::*;

/// A star found in the star image, and how far away it is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Star {
    /// Centroid in image pixels, pixel-centre coordinates.
    pub x: f64,
    pub y: f64,
    /// Distance in parsecs, or `None` to make it part of the background
    /// plane.
    pub distance_pc: Option<f64>,
}

/// One star's light, cut from the star image.
#[derive(Clone, Debug)]
pub struct Sprite {
    /// The sprite's top-left pixel in the image.
    pub left: usize,
    pub top: usize,
    pub image: LightImage,
    /// The star's centroid in image pixels.
    pub x: f64,
    pub y: f64,
    pub distance_pc: f64,
}

/// How [`Scene::new`] cuts stars out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CutOptions {
    /// A star's footprint grows ring by ring until the ring's median light
    /// is within this much of the star image's background (or four times
    /// its noise, if that is more), so faint stars cut small and bright ones
    /// take their halo.
    pub edge_light: f32,
    /// The smallest and largest footprint radius, pixels.
    pub min_radius: usize,
    pub max_radius: usize,
    /// How many stars, brightest first, fly at their own distances; `None`
    /// for all of them. A deep image holds so many faint stars that, each
    /// moving on its own, they crowd the view.
    pub max_stars: Option<usize>,
    /// What becomes of the stars past `max_stars`.
    pub small_stars: SmallStars,
    /// A star within this fraction of the background's distance is part of
    /// it, and grows with it rather than staying a point: most often the
    /// star lighting a reflection nebula, whose light the star image took
    /// from bright nebula and which only looks right over the same patch.
    pub embedded: f64,
}

impl Default for CutOptions {
    fn default() -> Self {
        Self {
            edge_light: 0.004,
            min_radius: 3,
            max_radius: 200,
            max_stars: None,
            small_stars: SmallStars::Drop,
            embedded: 0.02,
        }
    }
}

/// What becomes of the stars past [`CutOptions::max_stars`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmallStars {
    /// Their light stays on the leftover plane, which moves as the distant
    /// star field does.
    Field,
    /// Their light is removed from the video.
    Drop,
}

/// Everything a frame is rendered from.
#[derive(Clone, Debug)]
pub struct Scene {
    /// The starless image, on the background plane.
    pub background: Pyramid,
    /// Star light no sprite took: stars too faint to find, and the edges of
    /// halos. Most of it belongs to the field's distant stars, so it is a
    /// plane of its own at their distance rather than part of the
    /// background, where it would grow with a nearby nebula.
    pub leftover: Pyramid,
    pub sprites: Vec<Sprite>,
    /// Distance of the background plane, parsecs.
    pub background_distance_pc: f64,
    /// Distance of the leftover star light, parsecs.
    pub leftover_distance_pc: f64,
    /// The image's focal length in pixels: one radian across the field is
    /// this many pixels.
    pub focal_px: f64,
    /// A distance most sprites lie within, at least the background's: the
    /// far star field a camera move must not uncover the edge of.
    pub far_distance_pc: f64,
}

impl Scene {
    /// Cut `stars` out of `star_light`, keeping `starless` as the background
    /// plane at `background_distance_pc` and the light no star took as a
    /// plane at `leftover_distance_pc`.
    ///
    /// Each star takes a soft round footprint. Where footprints overlap the
    /// light is shared out in proportion to their weights, and the weights
    /// never sum past one, so the sprites plus the leftover add back up to
    /// the star image exactly, less any stars `options` drops.
    pub fn new(
        starless: &LightImage,
        star_light: &LightImage,
        stars: &[Star],
        background_distance_pc: f64,
        leftover_distance_pc: f64,
        focal_px: f64,
        options: &CutOptions,
    ) -> Self {
        assert_eq!(
            (starless.width, starless.height),
            (star_light.width, star_light.height),
            "the starless and star images must be the same size"
        );
        let (width, height) = (star_light.width, star_light.height);
        // A star image keeps a little light between stars; a footprint ends
        // where its ring reaches that level, not at zero.
        let (background, noise) = background_and_noise(star_light);
        let edge = background + options.edge_light.max(4.0 * noise);
        let measured: Vec<Footprint> = stars
            .par_iter()
            .map(|star| Footprint::measure(star, star_light, edge, options))
            .collect();
        // `stars` come brightest first. A fainter star inside a brighter
        // one's footprint is most often a piece of its halo or spikes, and
        // even a real neighbour there shares its light: either way it moves
        // as part of the brighter star rather than flying off on its own.
        let kept = outside_brighter(&measured);
        let footprints: Vec<Footprint> = kept.iter().map(|&index| measured[index]).collect();
        let stars: Vec<Star> = kept.iter().map(|&index| stars[index]).collect();

        // The sum of every footprint's weight at each pixel, which says how
        // much of the light the stars take, and the sum of their claims,
        // which says how they share it.
        let peaks: Vec<f32> = footprints
            .iter()
            .map(|footprint| footprint.peak(star_light))
            .collect();
        let mut total = vec![0.0_f32; width * height];
        let mut claims = vec![0.0_f32; width * height];
        for (footprint, &peak) in footprints.iter().zip(&peaks) {
            footprint.for_each(width, height, |x, y, weight| {
                total[y * width + x] += weight;
                claims[y * width + x] += weight * footprint.model(x, y, peak);
            });
        }
        // Every star is cut, so a small star keeps its own light rather
        // than a bright neighbour taking it, but only the brightest fly.
        let flying = options.max_stars.unwrap_or(usize::MAX).min(stars.len());
        // The leftover plane gives up only the flying stars' light when the
        // small stars stay on it: their share of what the stars take.
        let flying_share =
            (flying < stars.len() && options.small_stars == SmallStars::Field).then(|| {
                let mut flying_claims = vec![0.0_f32; width * height];
                for (footprint, &peak) in footprints.iter().zip(&peaks).take(flying) {
                    footprint.for_each(width, height, |x, y, weight| {
                        flying_claims[y * width + x] += weight * footprint.model(x, y, peak);
                    });
                }
                flying_claims
                    .par_iter_mut()
                    .zip(&claims)
                    .for_each(|(flying, &all)| {
                        *flying = if all > 0.0 { *flying / all } else { 0.0 };
                    });
                flying_claims
            });

        let mut sprites: Vec<Sprite> = footprints[..flying]
            .par_iter()
            .zip(&stars)
            .zip(&peaks)
            .map(|((footprint, star), &peak)| {
                let (left, top, right, bottom) = footprint.bounds(width, height);
                let mut image = LightImage::new(right - left, bottom - top);
                footprint.for_each(width, height, |x, y, weight| {
                    // Overlapping stars share the light they take in
                    // proportion to how much each would put there, so a
                    // bright star's halo stays with it rather than going to
                    // the faint stars inside it.
                    let index = y * width + x;
                    let claim = weight * footprint.model(x, y, peak);
                    let share = if claims[index] > 0.0 {
                        total[index].min(1.0) * claim / claims[index]
                    } else {
                        0.0
                    };
                    let light = star_light.at(x, y);
                    image.pixels[(y - top) * image.width + (x - left)] =
                        light.map(|value| value * share);
                });
                Sprite {
                    left,
                    top,
                    image,
                    x: star.x,
                    y: star.y,
                    distance_pc: star.distance_pc.unwrap_or(background_distance_pc),
                }
            })
            .collect();

        let leftover = LightImage {
            width,
            height,
            pixels: star_light
                .pixels
                .par_iter()
                .zip(total.par_iter())
                .enumerate()
                .map(|(index, (stars, taken))| {
                    let taken = match &flying_share {
                        Some(share) => taken.min(1.0) * share[index],
                        None => taken.min(1.0),
                    };
                    stars.map(|value| value * (1.0 - taken))
                })
                .collect(),
        };
        // Stars at the background's distance become part of it.
        let mut background = starless.clone();
        sprites.retain(|sprite| {
            let embedded = (sprite.distance_pc - background_distance_pc).abs()
                <= options.embedded * background_distance_pc;
            if embedded {
                for y in 0..sprite.image.height {
                    for x in 0..sprite.image.width {
                        let index = (sprite.top + y) * width + sprite.left + x;
                        let light = sprite.image.at(x, y);
                        for (target, light) in background.pixels[index].iter_mut().zip(light) {
                            *target += light;
                        }
                    }
                }
            }
            !embedded
        });
        let mut distances: Vec<f64> = sprites.iter().map(|sprite| sprite.distance_pc).collect();
        let far_distance_pc = if distances.is_empty() {
            background_distance_pc
        } else {
            let index = distances.len() * 9 / 10;
            let (_, &mut far, _) = distances.select_nth_unstable_by(index, f64::total_cmp);
            far.max(background_distance_pc)
        };
        Self {
            background: Pyramid::new(background),
            leftover: Pyramid::new(leftover),
            sprites,
            background_distance_pc,
            leftover_distance_pc,
            focal_px,
            far_distance_pc,
        }
    }

    pub fn width(&self) -> usize {
        self.background.base().width
    }

    pub fn height(&self) -> usize {
        self.background.base().height
    }
}

/// A star's soft round footprint: weight one out to `inner`, falling to zero
/// at `outer`.
#[derive(Clone, Copy, Debug)]
struct Footprint {
    x: f64,
    y: f64,
    inner: f64,
    outer: f64,
}

impl Footprint {
    /// The star's light at its centre, summed over channels.
    fn peak(&self, light: &LightImage) -> f32 {
        let x = (self.x.round().max(0.0) as usize).min(light.width - 1);
        let y = (self.y.round().max(0.0) as usize).min(light.height - 1);
        let pixel = light.at(x, y);
        (pixel[0] + pixel[1] + pixel[2]).max(1e-6)
    }

    /// The radius of the star's bright core, the width of its profile.
    fn core(&self) -> f64 {
        (self.inner / 3.0).max(1.0)
    }

    /// The light this star would put at `(x, y)`: a Moffat-like profile of
    /// its peak, as wide as its core. A bright star's footprint, and so its
    /// profile, reaches far into its halo.
    fn model(&self, x: usize, y: usize, peak: f32) -> f32 {
        let scale = self.core();
        let r2 = ((x as f64 - self.x).powi(2) + (y as f64 - self.y).powi(2)) / (scale * scale);
        peak * (1.0 / (1.0 + r2)).powi(2) as f32
    }

    fn measure(star: &Star, light: &LightImage, edge: f32, options: &CutOptions) -> Self {
        let mut radius = options.min_radius;
        while radius < options.max_radius && ring_light(light, star.x, star.y, radius) >= edge {
            radius += 1;
        }
        let outer = radius as f64 + 1.5;
        Self {
            x: star.x,
            y: star.y,
            inner: outer * 0.6,
            outer,
        }
    }

    fn weight(&self, x: usize, y: usize) -> f32 {
        let distance = ((x as f64 - self.x).powi(2) + (y as f64 - self.y).powi(2)).sqrt();
        if distance <= self.inner {
            1.0
        } else if distance >= self.outer {
            0.0
        } else {
            let t = (distance - self.inner) / (self.outer - self.inner);
            // Smoothstep down to zero.
            (1.0 - t * t * (3.0 - 2.0 * t)) as f32
        }
    }

    fn bounds(&self, width: usize, height: usize) -> (usize, usize, usize, usize) {
        let clamp = |value: f64, limit: usize| (value.max(0.0) as usize).min(limit);
        (
            clamp((self.x - self.outer).floor(), width),
            clamp((self.y - self.outer).floor(), height),
            clamp((self.x + self.outer).ceil() + 1.0, width),
            clamp((self.y + self.outer).ceil() + 1.0, height),
        )
    }

    fn for_each(&self, width: usize, height: usize, mut visit: impl FnMut(usize, usize, f32)) {
        let (left, top, right, bottom) = self.bounds(width, height);
        for y in top..bottom {
            for x in left..right {
                let weight = self.weight(x, y);
                if weight > 0.0 {
                    visit(x, y, weight);
                }
            }
        }
    }
}

/// Indices of the footprints whose centres lie outside the core of every
/// earlier (so brighter) kept star. Only a bright star's core swallows what
/// lies in it; a star in its halo travels on its own, and the halo's light
/// stays with the bright star by the shares the profiles set.
fn outside_brighter(footprints: &[Footprint]) -> Vec<usize> {
    const CELL: f64 = 64.0;
    let cell = |x: f64, y: f64| ((x / CELL).floor() as i64, (y / CELL).floor() as i64);
    let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> =
        std::collections::HashMap::new();
    let mut widest = 0.0_f64;
    let mut kept = Vec::with_capacity(footprints.len());
    for (index, footprint) in footprints.iter().enumerate() {
        let span = (widest / CELL).ceil() as i64;
        let (cx, cy) = cell(footprint.x, footprint.y);
        let inside = (cy - span..=cy + span).any(|row| {
            (cx - span..=cx + span).any(|column| {
                grid.get(&(column, row)).is_some_and(|indices| {
                    indices.iter().any(|&other| {
                        let brighter: &Footprint = &footprints[other];
                        (brighter.x - footprint.x).hypot(brighter.y - footprint.y) < brighter.core()
                    })
                })
            })
        });
        if inside {
            continue;
        }
        widest = widest.max(footprint.core());
        grid.entry(cell(footprint.x, footprint.y))
            .or_default()
            .push(index);
        kept.push(index);
    }
    kept
}

/// Median light, summed over channels, on the ring `radius` pixels from
/// `(x, y)`. The median ignores the few ring pixels a neighbouring star
/// covers.
fn ring_light(light: &LightImage, x: f64, y: f64, radius: usize) -> f32 {
    let steps = (radius * 8).max(8);
    let mut samples = Vec::with_capacity(steps);
    for step in 0..steps {
        let angle = step as f64 / steps as f64 * std::f64::consts::TAU;
        let (px, py) = (
            (x + radius as f64 * angle.cos()).round(),
            (y + radius as f64 * angle.sin()).round(),
        );
        if px < 0.0 || py < 0.0 || px >= light.width as f64 || py >= light.height as f64 {
            continue;
        }
        let pixel = light.at(px as usize, py as usize);
        samples.push(pixel[0] + pixel[1] + pixel[2]);
    }
    if samples.is_empty() {
        return 0.0;
    }
    let middle = samples.len() / 2;
    *samples.select_nth_unstable_by(middle, f32::total_cmp).1
}

/// The median light, summed over channels, of a sample of the image, and
/// its noise as a scaled median absolute deviation.
fn background_and_noise(light: &LightImage) -> (f32, f32) {
    let step = (light.pixels.len() / 200_000).max(1);
    let mut samples: Vec<f32> = light
        .pixels
        .iter()
        .step_by(step)
        .map(|pixel| pixel[0] + pixel[1] + pixel[2])
        .collect();
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let middle = samples.len() / 2;
    let median = *samples.select_nth_unstable_by(middle, f32::total_cmp).1;
    let mut deviations: Vec<f32> = samples.iter().map(|value| (value - median).abs()).collect();
    let mad = *deviations.select_nth_unstable_by(middle, f32::total_cmp).1;
    (median, 1.4826 * mad)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian_stars(width: usize, height: usize, stars: &[(f64, f64, f32)]) -> LightImage {
        let mut image = LightImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let mut light = 0.0;
                for (sx, sy, peak) in stars {
                    let r2 = (x as f64 - sx).powi(2) + (y as f64 - sy).powi(2);
                    light += peak * (-r2 / 4.0).exp() as f32;
                }
                image.pixels[y * width + x] = [light, light * 0.8, light * 0.6];
            }
        }
        image
    }

    #[test]
    fn a_second_peak_in_a_bright_core_joins_it() {
        // A bright star and a second peak on its core, found as a star of
        // its own.
        let light = gaussian_stars(64, 48, &[(30.0, 24.0, 8.0), (31.0, 24.0, 0.2)]);
        let starless = LightImage::new(64, 48);
        let stars = [(30.0, 24.0, Some(136.0)), (31.0, 24.0, Some(2000.0))]
            .map(|(x, y, distance_pc)| Star { x, y, distance_pc });
        let scene = Scene::new(
            &starless,
            &light,
            &stars,
            410.0,
            900.0,
            500.0,
            &CutOptions::default(),
        );
        assert_eq!(scene.sprites.len(), 1);
        assert_eq!(scene.sprites[0].distance_pc, 136.0);
    }

    #[test]
    fn a_bright_halo_stays_with_its_star_not_with_faint_neighbours() {
        // A bright star with a wide halo, and a faint star well out in it.
        let mut light = gaussian_stars(200, 160, &[(70.0, 80.0, 6.0), (125.0, 80.0, 0.6)]);
        for (index, pixel) in light.pixels.iter_mut().enumerate() {
            let (x, y) = ((index % 200) as f64, (index / 200) as f64);
            let halo =
                0.4 * (-((x - 70.0).powi(2) + (y - 80.0).powi(2)) / (2.0 * 30.0_f64.powi(2))).exp();
            for channel in pixel.iter_mut() {
                *channel += halo as f32;
            }
        }
        let starless = LightImage::new(200, 160);
        let stars = [(70.0, 80.0, Some(130.0)), (125.0, 80.0, Some(900.0))]
            .map(|(x, y, distance_pc)| Star { x, y, distance_pc });
        let scene = Scene::new(
            &starless,
            &light,
            &stars,
            400.0,
            900.0,
            500.0,
            &CutOptions::default(),
        );
        assert_eq!(scene.sprites.len(), 2);
        let total =
            |sprite: &Sprite| -> f32 { sprite.image.pixels.iter().map(|pixel| pixel[0]).sum() };
        // The faint star's own light: a Gaussian of peak 0.6 and variance 2.
        let own = 0.6 * 2.0 * std::f32::consts::PI * 2.0;
        let faint = total(&scene.sprites[1]);
        assert!(
            faint < own * 1.5,
            "the faint sprite took {faint}, its own light is {own}"
        );
    }

    #[test]
    fn stars_at_the_edges_measure_inside_the_image() {
        let light = gaussian_stars(32, 24, &[(31.4, 23.4, 3.0), (0.0, 0.0, 3.0)]);
        let starless = LightImage::new(32, 24);
        let stars = [(31.4, 23.4), (0.0, 0.0)].map(|(x, y)| Star {
            x,
            y,
            distance_pc: Some(50.0),
        });
        let scene = Scene::new(
            &starless,
            &light,
            &stars,
            100.0,
            900.0,
            500.0,
            &CutOptions::default(),
        );
        assert_eq!(scene.sprites.len(), 2);
    }

    #[test]
    fn sprites_and_leftover_add_back_to_the_star_image() {
        // Two stars whose footprints overlap, and one alone.
        let stars_light = gaussian_stars(
            64,
            48,
            &[(20.0, 20.0, 2.0), (31.0, 21.0, 1.0), (50.0, 34.0, 0.5)],
        );
        let mut starless = LightImage::new(64, 48);
        starless
            .pixels
            .iter_mut()
            .for_each(|pixel| *pixel = [0.1, 0.2, 0.3]);
        let stars = [
            (20.0, 20.0, Some(100.0)),
            (31.0, 21.0, None),
            (50.0, 34.0, Some(5000.0)),
        ]
        .map(|(x, y, distance_pc)| Star { x, y, distance_pc });
        let scene = Scene::new(
            &starless,
            &stars_light,
            &stars,
            400.0,
            900.0,
            1000.0,
            &CutOptions::default(),
        );
        assert_eq!(
            scene.sprites.len(),
            2,
            "the star with no distance joins the background"
        );
        let background = scene.background.base().at(31, 21);
        assert!(background[0] > 0.1 + 0.5, "{background:?}");

        let rebuilt = rebuilt(&scene);
        for (index, (rebuilt, (base, stars))) in rebuilt
            .pixels
            .iter()
            .zip(starless.pixels.iter().zip(&stars_light.pixels))
            .enumerate()
        {
            for channel in 0..3 {
                let expected = base[channel] + stars[channel];
                assert!(
                    (rebuilt[channel] - expected).abs() < 1e-5,
                    "pixel {index} channel {channel}"
                );
            }
        }
        // A bright star's footprint reaches further than a faint one's.
        let size = |sprite: &Sprite| sprite.image.width;
        assert!(size(&scene.sprites[0]) > size(&scene.sprites[1]));
    }

    /// The background, leftover and sprites of `scene` added back together.
    fn rebuilt(scene: &Scene) -> LightImage {
        let mut rebuilt = scene.background.base().clone();
        for (rebuilt, leftover) in rebuilt.pixels.iter_mut().zip(&scene.leftover.base().pixels) {
            for channel in 0..3 {
                rebuilt[channel] += leftover[channel];
            }
        }
        for sprite in &scene.sprites {
            for y in 0..sprite.image.height {
                for x in 0..sprite.image.width {
                    let index = (sprite.top + y) * rebuilt.width + sprite.left + x;
                    let light = sprite.image.at(x, y);
                    for (target, light) in rebuilt.pixels[index].iter_mut().zip(light) {
                        *target += light;
                    }
                }
            }
        }
        rebuilt
    }

    #[test]
    fn small_stars_stay_on_the_field_or_are_dropped() {
        // A bright star with a faint one in its halo, and a faint one alone.
        let stars_light = gaussian_stars(
            64,
            48,
            &[(20.0, 20.0, 2.0), (27.0, 20.0, 0.3), (50.0, 34.0, 0.3)],
        );
        let starless = LightImage::new(64, 48);
        let stars = [(20.0, 20.0), (27.0, 20.0), (50.0, 34.0)].map(|(x, y)| Star {
            x,
            y,
            distance_pc: Some(100.0),
        });
        let scene = |small_stars| {
            let options = CutOptions {
                max_stars: Some(1),
                small_stars,
                ..CutOptions::default()
            };
            Scene::new(
                &starless,
                &stars_light,
                &stars,
                400.0,
                900.0,
                1000.0,
                &options,
            )
        };
        let light = |image: &LightImage, (x, y): (usize, usize)| image.at(x, y)[0];

        let field = scene(SmallStars::Field);
        assert_eq!(field.sprites.len(), 1);
        // The small stars' light is all still there, on the leftover plane.
        let whole = rebuilt(&field);
        for (whole, stars) in whole.pixels.iter().zip(&stars_light.pixels) {
            assert!((whole[0] - stars[0]).abs() < 1e-5);
        }
        for point in [(27, 20), (50, 34)] {
            let expected = light(&stars_light, point);
            let leftover = light(field.leftover.base(), point);
            assert!(
                leftover > 0.5 * expected,
                "{point:?}: {leftover} of {expected}"
            );
        }

        let dropped = scene(SmallStars::Drop);
        assert_eq!(dropped.sprites.len(), 1);
        // The small stars are gone, and the bright star keeps no more than
        // it would have with them flying.
        for point in [(27, 20), (50, 34)] {
            let left = light(&rebuilt(&dropped), point);
            let expected = light(&stars_light, point);
            assert!(left < 0.5 * expected, "{point:?}: {left} of {expected}");
        }
        assert_eq!(
            dropped.sprites[0].image.pixels,
            field.sprites[0].image.pixels
        );
    }
}
