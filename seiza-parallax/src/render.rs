//! A camera that trucks and dollies toward a point of the background plane,
//! and the frames it sees.
//!
//! The image is treated as a pinhole view from Earth with its optical axis
//! through the image centre: a pixel `Δ` pixels off centre at distance `d`
//! sits at `Δ · d / f` across, where `f` is the image's focal length in
//! pixels. The camera never turns: its lens keeps the angle that framed the
//! first frame, and every change of view comes from moving it. It flies
//! `dolly` of the way to the background plane and moves sideways until the
//! focus point is ahead of it, at the centre of the frame. Started on the
//! focus point it simply flies along the line of sight to it, and every
//! depth only grows about that point. Moving sideways, toward the focus
//! point or in a `truck`, slides nearer depths across farther ones, as
//! from a moving car the near trees race by and the hills barely move.

use crate::light::{LightImage, Pyramid};
use crate::scene::{Scene, Sprite};
use rayon::prelude::*;

/// How the camera's progress follows time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Easing {
    /// Constant speed.
    Linear,
    /// Starts and ends at rest (smoothstep).
    #[default]
    InOut,
}

impl Easing {
    fn apply(self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::InOut => t * t * (3.0 - 2.0 * t),
        }
    }
}

/// What a shot's first frame shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Start {
    /// The widest view centred on the focus point, which stays centred.
    #[default]
    Focus,
    /// The widest view of the whole image. The camera moves sideways as it
    /// flies in, without turning, until the focus point is ahead of it.
    Whole,
}

/// A camera move and how it is filmed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shot {
    /// The point of the background plane the camera keeps centred, image
    /// pixels.
    pub focus: (f64, f64),
    /// The fraction of the way to the background plane the camera travels.
    pub dolly: f64,
    /// Sideways travel at the middle of the shot, as a fraction of the
    /// background distance, along the image's x and y axes. The camera
    /// swings out and back without turning, so near stars slide across the
    /// far ones.
    pub truck: (f64, f64),
    /// What the first frame shows, and so where the camera's aim starts.
    pub start: Start,
    /// How far the first frame zooms in: 1 shows the widest view the start
    /// allows that stays inside the image, 2 half as wide.
    pub zoom: f64,
    /// How much more the last frame is magnified than the first, by
    /// lengthening the lens rather than moving the camera: it enlarges every
    /// depth alike, so it pushes in without sliding layers apart.
    pub zoom_end: f64,
    /// How early the camera makes its sideways travel toward the focus
    /// point from the opening view, 0 to 1: at 1 the focus point closes on
    /// the frame's centre in step with the shot, at 0 mostly at the end.
    /// [`Self::fitted`] lowers it if a far layer's edge would show.
    pub lead: f64,
    /// Output frame size, pixels.
    pub width: usize,
    pub height: usize,
    /// Number of frames.
    pub frames: usize,
    pub easing: Easing,
    /// How much nearer a star has come, as its first-frame distance over its
    /// distance now, is its growth. A star is a point, so it keeps its
    /// first-frame size as the background grows around it; it swells as the
    /// square root of its growth, capped at this.
    pub growth_limit: f64,
    /// A star brightens as its growth (capped) to this power.
    pub brightening: f64,
    /// Past this growth a star starts fading out, and is gone by twice it,
    /// so the camera passes through stars instead of into a blinding disc.
    pub fade_from: f64,
}

impl Default for Shot {
    fn default() -> Self {
        Self {
            focus: (0.0, 0.0),
            dolly: 0.5,
            truck: (0.05, 0.0),
            start: Start::Focus,
            zoom: 1.0,
            zoom_end: 1.0,
            lead: 1.0,
            width: 1920,
            height: 1080,
            frames: 240,
            easing: Easing::default(),
            growth_limit: 4.0,
            brightening: 0.5,
            fade_from: 6.0,
        }
    }
}

/// The camera at one moment.
#[derive(Clone, Copy, Debug)]
struct View {
    /// Image centre, pixels.
    centre: (f64, f64),
    focal_px: f64,
    /// Camera position: across (parsecs, image axes) and toward the plane.
    across: (f64, f64),
    along: f64,
    /// Output focal length, pixels.
    focal_out: f64,
    /// The lens shift that frames the first frame's view, output pixels.
    focus_shift: (f64, f64),
    /// Output centre, pixels.
    out_centre: (f64, f64),
}

impl View {
    /// The image pixel at `distance` that output pixel `(u, v)` shows.
    fn unproject(&self, u: f64, v: f64, distance: f64) -> (f64, f64) {
        let depth = distance - self.along;
        let scale = self.focal_out * distance / (self.focal_px * depth);
        let back = |out: f64, out_centre: f64, across: f64, shift: f64, centre: f64| {
            centre + (out - out_centre + self.focal_out * across / depth + shift) / scale
        };
        (
            back(
                u,
                self.out_centre.0,
                self.across.0,
                self.focus_shift.0,
                self.centre.0,
            ),
            back(
                v,
                self.out_centre.1,
                self.across.1,
                self.focus_shift.1,
                self.centre.1,
            ),
        )
    }

    /// Output position of image pixel `(x, y)` at `distance`, and output
    /// pixels per image pixel there, or `None` when it is behind the camera.
    fn project(&self, x: f64, y: f64, distance: f64) -> Option<(f64, f64, f64)> {
        let depth = distance - self.along;
        if depth <= distance * 1e-3 {
            return None;
        }
        let scale = self.focal_out * distance / (self.focal_px * depth);
        let offset = |delta: f64, across: f64, shift: f64| {
            scale * delta - self.focal_out * across / depth - shift
        };
        Some((
            self.out_centre.0 + offset(x - self.centre.0, self.across.0, self.focus_shift.0),
            self.out_centre.1 + offset(y - self.centre.1, self.across.1, self.focus_shift.1),
            scale,
        ))
    }
}

impl Shot {
    /// The widest footprint (image pixels per output pixel) of a frame
    /// centred on `(x, y)` that stays inside a `width` × `height` image.
    fn widest_footprint(&self, (x, y): (f64, f64), width: usize, height: usize) -> f64 {
        let half_x = x.min(width as f64 - 1.0 - x).max(1.0);
        let half_y = y.min(height as f64 - 1.0 - y).max(1.0);
        (2.0 * half_x / self.width as f64).min(2.0 * half_y / self.height as f64)
    }

    /// The first frame's centre on the background plane and its footprint.
    fn opening(&self, width: usize, height: usize) -> ((f64, f64), f64) {
        let zoom = self.zoom.max(1.0);
        match self.start {
            Start::Focus => (
                self.focus,
                self.widest_footprint(self.focus, width, height) / zoom,
            ),
            Start::Whole => {
                let centre = ((width as f64 - 1.0) / 2.0, (height as f64 - 1.0) / 2.0);
                let footprint = self.widest_footprint(centre, width, height) / zoom;
                // As near the focus point as a view that size allows.
                let toward = |focus: f64, out: usize, size: usize, centre: f64| {
                    let half = footprint * (out as f64 - 1.0) / 2.0;
                    let (low, high) = (half, size as f64 - 1.0 - half);
                    if low <= high {
                        focus.clamp(low, high)
                    } else {
                        centre
                    }
                };
                (
                    (
                        toward(self.focus.0, self.width, width, centre.0),
                        toward(self.focus.1, self.height, height, centre.1),
                    ),
                    footprint,
                )
            }
        }
    }

    /// The depths whose edges must stay out of view: the background, the
    /// leftover star light and the far sprites.
    fn guarded_depths(&self, scene: &Scene) -> [f64; 3] {
        [
            scene.background_distance_pc,
            scene.leftover_distance_pc,
            scene.far_distance_pc,
        ]
    }

    /// Whether every corner of `view` lies inside the image at `depths`.
    fn inside(&self, scene: &Scene, view: &View, depths: &[f64]) -> bool {
        let (width, height) = (scene.width() as f64 - 1.0, scene.height() as f64 - 1.0);
        let corners = [
            (0.0, 0.0),
            (self.width as f64 - 1.0, 0.0),
            (0.0, self.height as f64 - 1.0),
            (self.width as f64 - 1.0, self.height as f64 - 1.0),
        ];
        depths.iter().all(|&distance| {
            corners.iter().all(|&(u, v)| {
                let (x, y) = view.unproject(u, v, distance);
                (0.0..=width).contains(&x) && (0.0..=height).contains(&y)
            })
        })
    }

    fn view(&self, scene: &Scene, frame: usize) -> View {
        let t = if self.frames > 1 {
            frame as f64 / (self.frames - 1) as f64
        } else {
            0.0
        };
        let progress = self.easing.apply(t);
        let distance = scene.background_distance_pc;
        let centre = (
            (scene.width() as f64 - 1.0) / 2.0,
            (scene.height() as f64 - 1.0) / 2.0,
        );
        let (opening, footprint) = self.opening(scene.width(), scene.height());
        let focal_out = scene.focal_px / footprint * self.zoom_end.max(1.0).powf(progress);
        let dolly = self.dolly.clamp(0.0, 0.99);
        let along = dolly * distance * progress;
        // The background's distance from the camera, as a fraction of its
        // distance from where the image was taken.
        let near = 1.0 - dolly * progress;
        // The camera never turns: its lens keeps the angle that framed the
        // opening view, and it reaches the focus point by moving. The
        // background point at the centre of the frame goes from the opening
        // view's centre to the focus point, and the focus point's place in
        // the frame closes on the centre in step with the shot's progress
        // (`lead` 1) or, as a straight line from where the image was taken
        // would have it, mostly at the end (`lead` 0). A truck swings the
        // camera sideways and back.
        let lead = self.lead.clamp(0.0, 1.0);
        let remaining = (1.0 - progress) * (lead + (1.0 - lead) / near);
        let swing = 4.0 * progress * (1.0 - progress);
        let across = |focus: f64, opening: f64, centre: f64, truck: f64| {
            let aimed = focus - (focus - opening) * remaining * near;
            (aimed - centre - (opening - centre) * near) * distance / scene.focal_px
                + truck * distance * swing
        };
        View {
            centre,
            focal_px: scene.focal_px,
            across: (
                across(self.focus.0, opening.0, centre.0, self.truck.0),
                across(self.focus.1, opening.1, centre.1, self.truck.1),
            ),
            along,
            focal_out,
            focus_shift: (
                focal_out * (opening.0 - centre.0) / scene.focal_px,
                focal_out * (opening.1 - centre.1) / scene.focal_px,
            ),
            out_centre: (
                (self.width as f64 - 1.0) / 2.0,
                (self.height as f64 - 1.0) / 2.0,
            ),
        }
    }

    /// This shot made to keep every layer's edge out of view, and the
    /// factor its truck was scaled by. The camera's sideways travel toward
    /// the focus point comes as early as `lead` allows, and its truck swings
    /// as wide as the image allows: moving sideways slides layers at
    /// different depths against each other, and the far ones would uncover
    /// ground the image never showed. Moving along a straight line from
    /// where the image was taken never does.
    pub fn fitted(&self, scene: &Scene) -> (Self, f64) {
        let depths = self.guarded_depths(scene);
        let fits = |shot: &Self| {
            (0..shot.frames).all(|frame| shot.inside(scene, &shot.view(scene, frame), &depths))
        };
        // The largest factor in [0, 1] the shot `make` builds still fits at.
        let largest = |make: &dyn Fn(f64) -> Self| -> f64 {
            if fits(&make(1.0)) {
                return 1.0;
            }
            let (mut low, mut high) = (0.0, 1.0);
            for _ in 0..30 {
                let middle = (low + high) / 2.0;
                if fits(&make(middle)) {
                    low = middle;
                } else {
                    high = middle;
                }
            }
            low
        };
        let lead = largest(&|factor| Self {
            lead: self.lead * factor,
            truck: (0.0, 0.0),
            ..*self
        });
        let led = Self {
            lead: self.lead * lead,
            ..*self
        };
        let factor = largest(&|factor| Self {
            truck: (self.truck.0 * factor, self.truck.1 * factor),
            ..led
        });
        (
            Self {
                truck: (self.truck.0 * factor, self.truck.1 * factor),
                ..led
            },
            factor,
        )
    }

    /// Render frame `frame` of `scene`.
    pub fn render(&self, scene: &Scene, frame: usize) -> LightImage {
        let view = self.view(scene, frame);
        let mut out = LightImage::new(self.width, self.height);
        draw_plane(
            &mut out,
            &scene.background,
            &view,
            scene.background_distance_pc,
        );
        draw_plane(&mut out, &scene.leftover, &view, scene.leftover_distance_pc);
        draw_sprites(&mut out, scene, &view, self);
        out
    }
}

/// Rows each parallel band of the frame covers.
const BAND_ROWS: usize = 16;

/// Add the light of the plane `image` at `distance` to `out`.
fn draw_plane(out: &mut LightImage, image: &Pyramid, view: &View, distance: f64) {
    // Invert `project` at the plane's distance, where it is affine.
    let Some((x0, y0, scale)) = view.project(view.centre.0, view.centre.1, distance) else {
        return;
    };
    let (level, level_scale) = image.level_for((1.0 / scale) as f32);
    let width = out.width;
    out.pixels
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, pixels)| {
            let y = view.centre.1 + (row as f64 - y0) / scale;
            for (column, pixel) in pixels.iter_mut().enumerate() {
                let x = view.centre.0 + (column as f64 - x0) / scale;
                let light = Pyramid::sample_level(level, level_scale, x as f32, y as f32);
                for channel in 0..3 {
                    pixel[channel] += light[channel];
                }
            }
        });
}

/// A sprite placed in one frame.
struct Placed<'a> {
    sprite: &'a Sprite,
    /// Output position of the star's centroid.
    x: f64,
    y: f64,
    /// Output pixels per sprite pixel.
    scale: f64,
    /// Light multiplier.
    gain: f32,
    /// Output rows it covers.
    top: f64,
    bottom: f64,
}

fn draw_sprites(out: &mut LightImage, scene: &Scene, view: &View, shot: &Shot) {
    // Every layer's scale with the lens as it is now and the camera not yet
    // moved: a star's growth is how much nearer it has come.
    let lens_scale = view.focal_out / view.focal_px;
    let (width, height) = (out.width as f64, out.height as f64);
    let placed: Vec<Placed> = scene
        .sprites
        .iter()
        .filter_map(|sprite| {
            let (x, y, scale) = view.project(sprite.x, sprite.y, sprite.distance_pc)?;
            let growth = scale / lens_scale;
            let fade = if growth <= shot.fade_from {
                1.0
            } else {
                (2.0 - growth / shot.fade_from).max(0.0)
            };
            if fade <= 0.0 {
                return None;
            }
            let grown = growth.clamp(1.0, shot.growth_limit);
            let size = lens_scale * grown.sqrt();
            let gain = (grown.powf(shot.brightening) * fade) as f32;
            // Sprite pixel (i, j) sits at image (left + i, top + j).
            let reach_left = (sprite.x - sprite.left as f64 + 1.0) * size;
            let reach_right =
                (sprite.left as f64 + sprite.image.width as f64 - sprite.x + 1.0) * size;
            let reach_up = (sprite.y - sprite.top as f64 + 1.0) * size;
            let reach_down =
                (sprite.top as f64 + sprite.image.height as f64 - sprite.y + 1.0) * size;
            if x + reach_right < 0.0
                || x - reach_left > width
                || y + reach_down < 0.0
                || y - reach_up > height
            {
                return None;
            }
            Some(Placed {
                sprite,
                x,
                y,
                scale: size,
                gain,
                top: y - reach_up,
                bottom: y + reach_down,
            })
        })
        .collect();
    let out_width = out.width;
    out.pixels
        .par_chunks_mut(out_width * BAND_ROWS)
        .enumerate()
        .for_each(|(band, pixels)| {
            let first = band * BAND_ROWS;
            let rows = pixels.len() / out_width;
            for placed in placed.iter().filter(|placed| {
                placed.bottom >= first as f64 && placed.top < (first + rows) as f64
            }) {
                if placed.scale >= 1.0 {
                    sample_sprite(pixels, out_width, first, rows, placed);
                } else {
                    splat_sprite(pixels, out_width, first, rows, placed);
                }
            }
        });
}

/// Draw a sprite at least as large as its pixels by sampling it at each
/// output pixel it covers.
fn sample_sprite(
    pixels: &mut [[f32; 3]],
    width: usize,
    first: usize,
    rows: usize,
    placed: &Placed,
) {
    let sprite = placed.sprite;
    let left = (placed.x - (sprite.x - sprite.left as f64 + 1.0) * placed.scale)
        .floor()
        .max(0.0) as usize;
    let right = ((placed.x
        + (sprite.left as f64 + sprite.image.width as f64 - sprite.x + 1.0) * placed.scale)
        .ceil()
        .max(0.0) as usize)
        .min(width);
    let top = (placed.top.floor().max(first as f64) as usize).max(first);
    let bottom = (placed.bottom.ceil().max(0.0) as usize).min(first + rows);
    for row in top..bottom {
        let sy = (row as f64 - placed.y) / placed.scale + sprite.y - sprite.top as f64;
        for column in left..right {
            let sx = (column as f64 - placed.x) / placed.scale + sprite.x - sprite.left as f64;
            let light = sprite.image.sample(sx as f32, sy as f32);
            let pixel = &mut pixels[(row - first) * width + column];
            for channel in 0..3 {
                pixel[channel] += light[channel] * placed.gain;
            }
        }
    }
}

/// Draw a sprite smaller than its pixels by spreading each pixel's light
/// over the output pixels it lands between, so a shrunk star keeps its
/// total light instead of flickering.
fn splat_sprite(pixels: &mut [[f32; 3]], width: usize, first: usize, rows: usize, placed: &Placed) {
    let sprite = placed.sprite;
    let area = (placed.scale * placed.scale) as f32 * placed.gain;
    for j in 0..sprite.image.height {
        let y = placed.y + (sprite.top as f64 + j as f64 - sprite.y) * placed.scale;
        for i in 0..sprite.image.width {
            let light = sprite.image.at(i, j);
            if light == [0.0; 3] {
                continue;
            }
            let x = placed.x + (sprite.left as f64 + i as f64 - sprite.x) * placed.scale;
            let (fx, fy) = (x.floor(), y.floor());
            let (tx, ty) = ((x - fx) as f32, (y - fy) as f32);
            for (dy, wy) in [(0, 1.0 - ty), (1, ty)] {
                let row = fy as isize + dy;
                if row < first as isize || row >= (first + rows) as isize {
                    continue;
                }
                for (dx, wx) in [(0, 1.0 - tx), (1, tx)] {
                    let column = fx as isize + dx;
                    if column < 0 || column >= width as isize {
                        continue;
                    }
                    let weight = wx * wy * area;
                    let pixel = &mut pixels[(row as usize - first) * width + column as usize];
                    for channel in 0..3 {
                        pixel[channel] += light[channel] * weight;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{CutOptions, Star};

    fn scene_with(stars: &[Star]) -> Scene {
        let (width, height) = (400, 300);
        let mut starless = LightImage::new(width, height);
        for (index, pixel) in starless.pixels.iter_mut().enumerate() {
            let (x, y) = (index % width, index / width);
            *pixel = [x as f32 / width as f32, y as f32 / height as f32, 0.2];
        }
        let mut star_light = LightImage::new(width, height);
        for star in stars {
            for y in 0..height {
                for x in 0..width {
                    let r2 = (x as f64 - star.x).powi(2) + (y as f64 - star.y).powi(2);
                    let value = 3.0 * (-r2 / 3.0).exp() as f32;
                    let pixel = &mut star_light.pixels[y * width + x];
                    for channel in pixel.iter_mut() {
                        *channel += value;
                    }
                }
            }
        }
        Scene::new(
            &starless,
            &star_light,
            stars,
            400.0,
            400.0,
            2000.0,
            &CutOptions::default(),
        )
    }

    fn brightest(image: &LightImage) -> (usize, usize) {
        let index = (0..image.pixels.len())
            .max_by(|a, b| image.pixels[*a][2].total_cmp(&image.pixels[*b][2]))
            .unwrap();
        (index % image.width, index / image.width)
    }

    #[test]
    fn the_first_frame_shows_the_image_around_the_focus_point() {
        let scene = scene_with(&[Star {
            x: 200.0,
            y: 150.0,
            distance_pc: Some(100.0),
        }]);
        let shot = Shot {
            focus: (199.5, 149.5),
            width: 200,
            height: 150,
            frames: 10,
            ..Shot::default()
        };
        let frame = shot.render(&scene, 0);
        // At rest, footprint 2: output pixel (i, j) shows image (2i, 2j)
        // roughly, and the star lands at the centre.
        let (x, y) = brightest(&frame);
        assert!(
            (x as i64 - 100).abs() <= 1 && (y as i64 - 75).abs() <= 1,
            "{x} {y}"
        );
    }

    #[test]
    fn trucking_moves_near_stars_against_far_ones() {
        let stars = [
            Star {
                x: 150.0,
                y: 150.0,
                distance_pc: Some(50.0),
            },
            Star {
                x: 250.0,
                y: 150.0,
                distance_pc: Some(5000.0),
            },
        ];
        let scene = scene_with(&stars);
        let shot = Shot {
            focus: (199.5, 149.5),
            dolly: 0.0,
            truck: (0.01, 0.0),
            width: 200,
            height: 150,
            frames: 3,
            easing: Easing::Linear,
            ..Shot::default()
        };
        // The middle frame is the truck's widest swing.
        let start = shot.view(&scene, 0);
        let middle = shot.view(&scene, 1);
        let moved = |x: f64, y: f64, distance: f64| {
            middle.project(x, y, distance).unwrap().0 - start.project(x, y, distance).unwrap().0
        };
        // The camera moves +x without turning: everything slides −x, the
        // near star most, the background less, the far star least.
        let near = moved(150.0, 150.0, 50.0);
        let background = moved(199.5, 149.5, 400.0);
        let far = moved(250.0, 150.0, 5000.0);
        assert!(
            near < background && background < far && far < 0.0,
            "{near} {background} {far}"
        );
        // And the swing returns: the last frame is the first.
        let last = shot.view(&scene, 2);
        let (a, b) = (
            start.project(150.0, 150.0, 50.0).unwrap(),
            last.project(150.0, 150.0, 50.0).unwrap(),
        );
        assert!((a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9);
    }

    #[test]
    fn the_camera_never_turns() {
        let scene = scene_with(&[]);
        let shot = Shot {
            focus: (320.0, 60.0),
            dolly: 0.8,
            truck: (0.002, 0.001),
            start: Start::Whole,
            zoom_end: 1.5,
            width: 200,
            height: 150,
            frames: 12,
            ..Shot::default()
        };
        // The lens shift, as an angle, is the same in every frame.
        let angle = |view: &View| {
            (
                view.focus_shift.0 / view.focal_out,
                view.focus_shift.1 / view.focal_out,
            )
        };
        let first = angle(&shot.view(&scene, 0));
        for frame in 1..shot.frames {
            let now = angle(&shot.view(&scene, frame));
            assert!((now.0 - first.0).abs() < 1e-12 && (now.1 - first.1).abs() < 1e-12);
        }
    }

    #[test]
    fn dollying_in_grows_the_background_about_the_focus_point() {
        let scene = scene_with(&[]);
        let shot = Shot {
            focus: (199.5, 149.5),
            dolly: 0.5,
            truck: (0.0, 0.0),
            width: 200,
            height: 150,
            frames: 2,
            easing: Easing::Linear,
            ..Shot::default()
        };
        let start = shot.view(&scene, 0).project(299.5, 149.5, 400.0).unwrap();
        let end = shot.view(&scene, 1).project(299.5, 149.5, 400.0).unwrap();
        // Halfway to the plane, everything on it is twice the size.
        assert!((end.2 / start.2 - 2.0).abs() < 1e-9);
        assert!(((end.0 - 99.5) / (start.0 - 99.5) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn dollying_toward_an_off_centre_focus_slides_no_depth() {
        let scene = scene_with(&[]);
        let shot = Shot {
            focus: (320.0, 60.0),
            dolly: 0.8,
            truck: (0.0, 0.0),
            width: 200,
            height: 150,
            frames: 2,
            easing: Easing::Linear,
            ..Shot::default()
        };
        let (start, end) = (shot.view(&scene, 0), shot.view(&scene, 1));
        // The focus point stays at the centre at every depth, nearer than
        // the plane or far beyond it.
        for distance in [350.0, 400.0, 1200.0, 5000.0] {
            for view in [&start, &end] {
                let (x, y, _) = view.project(320.0, 60.0, distance).unwrap();
                assert!(
                    (x - 99.5).abs() < 1e-6 && (y - 74.5).abs() < 1e-6,
                    "{distance} pc: ({x}, {y})"
                );
            }
        }
        // So a far layer only grows about it, and the frame's corners stay
        // inside the image there as on the plane.
        for distance in [400.0, 1200.0] {
            for (u, v) in [(0.0, 0.0), (199.0, 0.0), (0.0, 149.0), (199.0, 149.0)] {
                let (x, y) = end.unproject(u, v, distance);
                let (x0, y0) = start.unproject(u, v, distance);
                assert!(
                    (0.0..400.0).contains(&x) && (0.0..300.0).contains(&y),
                    "{distance} pc: ({u}, {v}) shows ({x}, {y})"
                );
                // Nearer the focus point than in the first frame.
                assert!((x - 320.0).abs() <= (x0 - 320.0).abs() + 1e-9);
                assert!((y - 60.0).abs() <= (y0 - 60.0).abs() + 1e-9);
            }
        }
    }

    #[test]
    fn a_whole_start_shows_the_image_then_closes_on_the_focus() {
        let scene = scene_with(&[]);
        let shot = Shot {
            focus: (320.0, 60.0),
            dolly: 0.8,
            truck: (0.0, 0.0),
            start: Start::Whole,
            width: 200,
            height: 150,
            frames: 30,
            easing: Easing::Linear,
            ..Shot::default()
        };
        // The first frame spans nearly the whole image, which is the
        // frame's shape.
        let first = shot.view(&scene, 0);
        let (left, top) = first.unproject(0.0, 0.0, 400.0);
        let (right, bottom) = first.unproject(199.0, 149.0, 400.0);
        assert!(left < 3.0 && right > 396.0, "{left}..{right}");
        assert!(top < 3.0 && bottom > 296.0, "{top}..{bottom}");
        // Fitted, no frame shows past the image at the background or the far
        // field, and the last one is centred on the focus point.
        let (shot, _) = shot.fitted(&scene);
        let depths = shot.guarded_depths(&scene);
        for frame in 0..shot.frames {
            let view = shot.view(&scene, frame);
            assert!(shot.inside(&scene, &view, &depths), "frame {frame}");
        }
        let last = shot.view(&scene, shot.frames - 1);
        let (x, y, _) = last.project(320.0, 60.0, 400.0).unwrap();
        assert!(
            (x - 99.5).abs() < 0.5 && (y - 74.5).abs() < 0.5,
            "({x}, {y})"
        );
    }

    #[test]
    fn a_truck_too_wide_for_the_image_is_scaled_down() {
        let scene = scene_with(&[Star {
            x: 100.0,
            y: 100.0,
            distance_pc: Some(4000.0),
        }]);
        let shot = Shot {
            focus: (199.5, 149.5),
            dolly: 0.5,
            truck: (0.2, 0.0),
            zoom: 1.5,
            width: 200,
            height: 150,
            frames: 20,
            ..Shot::default()
        };
        let (fitted, factor) = shot.fitted(&scene);
        assert!(factor > 0.0 && factor < 1.0, "{factor}");
        // Every frame's corners, on the far layer, stay inside.
        for frame in 0..fitted.frames {
            let view = fitted.view(&scene, frame);
            assert!(fitted.inside(&scene, &view, &[4000.0]), "frame {frame}");
        }
        let gentle = Shot {
            truck: (0.0001, 0.0),
            ..shot
        };
        assert_eq!(gentle.fitted(&scene).1, 1.0);
    }

    #[test]
    fn a_star_the_camera_passes_fades_out() {
        let star = Star {
            x: 210.0,
            y: 150.0,
            distance_pc: Some(250.0),
        };
        let scene = scene_with(&[star]);
        let shot = Shot {
            focus: (199.5, 149.5),
            dolly: 0.62,
            truck: (0.0, 0.0),
            width: 200,
            height: 150,
            frames: 2,
            easing: Easing::Linear,
            fade_from: 3.0,
            ..Shot::default()
        };
        // At the end the camera is 248 pc in: the star is 2 pc ahead, grown
        // far past twice the fade point, so it is gone.
        let frame = shot.render(&scene, 1);
        let background = {
            let mut without = scene.clone();
            without.sprites.clear();
            shot.render(&without, 1)
        };
        let excess: f32 = frame
            .pixels
            .iter()
            .zip(&background.pixels)
            .map(|(a, b)| (a[2] - b[2]).abs())
            .sum();
        assert_eq!(excess, 0.0);
    }
}
