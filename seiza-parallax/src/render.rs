//! A camera that trucks and dollies toward a point of the background plane,
//! and the frames it sees.
//!
//! The image is treated as a pinhole view from Earth with its optical axis
//! through the image centre: a pixel `Δ` pixels off centre at distance `d`
//! sits at `Δ · d / f` across, where `f` is the image's focal length in
//! pixels. The camera never turns aside: its lens keeps the angle that
//! framed the first frame, and every change of view comes from moving it,
//! save that it may roll about its line of sight, which turns the frame and
//! every depth in it alike. It flies
//! `dolly` of the way to the background plane and moves sideways until the
//! focus point is ahead of it, at the centre of the frame. Started on the
//! focus point it simply flies along the line of sight to it, and every
//! depth only grows about that point. Moving sideways, toward the focus
//! point or in a `truck`, slides nearer depths across farther ones, as
//! from a moving car the near trees race by and the hills barely move.

use crate::dust::Dust;
use crate::light::{LightImage, Pyramid, floor64};
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

/// How carefully frames are drawn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Quality {
    /// One sample per output pixel, from the level of detail nearest the
    /// view.
    #[default]
    Standard,
    /// Drawn at twice the size and averaged down, with the levels of
    /// detail either side of the view blended, so fine detail neither
    /// shimmers nor steps in sharpness as the camera moves. About four
    /// times the work.
    High,
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

/// A stop on a tour: where the camera looks, how near it has come, and
/// how long it takes to come here and stays.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stop {
    /// The point of the background plane at the frame's centre, image
    /// pixels.
    pub focus: (f64, f64),
    /// The fraction of the way to the background plane the camera has
    /// flown, 0 to below 1.
    pub dolly: f64,
    /// The lens's magnification over the opening view's.
    pub zoom: f64,
    /// The frame's turn about its centre, radians anticlockwise.
    pub rotation: f64,
    /// How much of its way to the focus point the camera turns rather than
    /// moves, 0 to 1. Turning sweeps the far star field with the nebula.
    pub pan: f64,
    /// Seconds to come here from the stop before (unused for the first),
    /// and to stay here.
    pub travel: f64,
    pub hold: f64,
    /// How far the frame turns while the camera holds here, radians
    /// anticlockwise, easing in and out. Later stops' turns count from the
    /// frame as it is left, so a whole turn does not unwind on the way on.
    pub spin: f64,
}

impl Default for Stop {
    fn default() -> Self {
        Self {
            focus: (0.0, 0.0),
            dolly: 0.0,
            zoom: 1.0,
            rotation: 0.0,
            pan: 0.0,
            travel: 5.0,
            hold: 0.0,
            spin: 0.0,
        }
    }
}

/// A camera move and how it is filmed.
#[derive(Clone, Debug, PartialEq)]
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
    /// How much of its way to the focus point the camera turns rather than
    /// moves, 0 to 1. Turning sweeps every depth alike, the distant star
    /// field with the nebula, so a little keeps the target nearer the centre
    /// early on and much of it looks like the sky spinning.
    /// [`Self::fitted`] lowers it if a far layer's edge would show.
    pub pan: f64,
    /// The frame's turn about its centre at the first and last frames,
    /// radians anticlockwise, reached as the camera moves. A turned frame
    /// needs more of the image than a square one, so [`Self::fitted`] zooms
    /// the first frame in until every frame fits.
    pub rotation: (f64, f64),
    /// A tour instead of the single move: the camera glides through the
    /// stops in turn, the first being the opening view, easing to a halt
    /// at each stop it holds at. Empty for the single move, which the
    /// fields above describe; a tour uses only `zoom` of them.
    pub tour: Vec<Stop>,
    /// Output frame size, pixels.
    pub width: usize,
    pub height: usize,
    /// Number of frames.
    pub frames: usize,
    pub easing: Easing,
    pub quality: Quality,
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
            pan: 0.0,
            rotation: (0.0, 0.0),
            tour: Vec::new(),
            width: 1920,
            height: 1080,
            frames: 240,
            easing: Easing::default(),
            quality: Quality::default(),
            growth_limit: 4.0,
            brightening: 0.5,
            fade_from: 6.0,
        }
    }
}

/// The camera at one moment, from [`Shot::view`]: where it puts the
/// points of each depth in the frame.
#[derive(Clone, Copy, Debug)]
pub struct View {
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
    /// Cosine and sine of the frame's turn about its centre, anticlockwise.
    turn: (f64, f64),
}

/// Which point of a plane each output pixel shows: output pixel
/// `(column, row)` shows image point `origin + column · across + row · down`.
#[derive(Clone, Copy, Debug)]
struct PlaneMap {
    origin: (f64, f64),
    across: (f64, f64),
    down: (f64, f64),
}

impl PlaneMap {
    #[inline]
    fn at(&self, column: f64, row: f64) -> (f64, f64) {
        (
            self.origin.0 + column * self.across.0 + row * self.down.0,
            self.origin.1 + column * self.across.1 + row * self.down.1,
        )
    }
}

impl View {
    /// The image pixel at `distance` that output pixel `(u, v)` shows.
    pub fn unproject(&self, u: f64, v: f64, distance: f64) -> (f64, f64) {
        let depth = distance - self.along;
        let scale = self.focal_out * distance / (self.focal_px * depth);
        // Turn the frame back first.
        let (cos, sin) = self.turn;
        let (du, dv) = (u - self.out_centre.0, v - self.out_centre.1);
        let (du, dv) = (cos * du - sin * dv, sin * du + cos * dv);
        let back = |out: f64, across: f64, shift: f64, centre: f64| {
            centre + (out + self.focal_out * across / depth + shift) / scale
        };
        (
            back(du, self.across.0, self.focus_shift.0, self.centre.0),
            back(dv, self.across.1, self.focus_shift.1, self.centre.1),
        )
    }

    /// The frame's turn about its centre, radians anticlockwise.
    pub fn turn(&self) -> f64 {
        self.turn.1.atan2(self.turn.0)
    }

    /// How much nearer a point drawn at `scale` output pixels per image
    /// pixel has come since the first frame, with the lens as it is now.
    fn growth(&self, scale: f64) -> f64 {
        scale * self.focal_px / self.focal_out
    }

    /// The plane at `distance` as the frame shows it, or `None` when it is
    /// behind the camera.
    fn plane_map(&self, distance: f64) -> Option<PlaneMap> {
        if distance - self.along <= distance * 1e-3 {
            return None;
        }
        let origin = self.unproject(0.0, 0.0, distance);
        let right = self.unproject(1.0, 0.0, distance);
        let below = self.unproject(0.0, 1.0, distance);
        Some(PlaneMap {
            origin,
            across: (right.0 - origin.0, right.1 - origin.1),
            down: (below.0 - origin.0, below.1 - origin.1),
        })
    }

    /// Output position of image pixel `(x, y)` at `distance`, and output
    /// pixels per image pixel there, or `None` when it is behind the camera.
    pub fn project(&self, x: f64, y: f64, distance: f64) -> Option<(f64, f64, f64)> {
        let depth = distance - self.along;
        if depth <= distance * 1e-3 {
            return None;
        }
        let scale = self.focal_out * distance / (self.focal_px * depth);
        let offset = |delta: f64, across: f64, shift: f64| {
            scale * delta - self.focal_out * across / depth - shift
        };
        let du = offset(x - self.centre.0, self.across.0, self.focus_shift.0);
        let dv = offset(y - self.centre.1, self.across.1, self.focus_shift.1);
        let (cos, sin) = self.turn;
        Some((
            self.out_centre.0 + cos * du + sin * dv,
            self.out_centre.1 - sin * du + cos * dv,
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

    /// Whether every corner of `view` lies inside the image at `depths`. A
    /// whole-image start puts the corners on the image's edges, so rounding
    /// a millionth of a pixel past them still counts as inside.
    fn inside(&self, scene: &Scene, view: &View, depths: &[f64]) -> bool {
        const SLACK: f64 = 1e-6;
        let (width, height) = (
            scene.width() as f64 - 1.0 + SLACK,
            scene.height() as f64 - 1.0 + SLACK,
        );
        let corners = [
            (0.0, 0.0),
            (self.width as f64 - 1.0, 0.0),
            (0.0, self.height as f64 - 1.0),
            (self.width as f64 - 1.0, self.height as f64 - 1.0),
        ];
        depths.iter().all(|&distance| {
            corners.iter().all(|&(u, v)| {
                let (x, y) = view.unproject(u, v, distance);
                (-SLACK..=width).contains(&x) && (-SLACK..=height).contains(&y)
            })
        })
    }

    /// The camera at frame `frame` of `scene`.
    pub fn view(&self, scene: &Scene, frame: usize) -> View {
        let t = if self.frames > 1 {
            frame as f64 / (self.frames - 1) as f64
        } else {
            0.0
        };
        if self.tour.len() >= 2 {
            return self.tour_view(scene, t);
        }
        let progress = self.easing.apply(t);
        let (opening, footprint) = self.opening(scene.width(), scene.height());
        let dolly = self.dolly.clamp(0.0, 0.99);
        // The background's distance from the camera, as a fraction of its
        // distance from where the image was taken.
        let near = 1.0 - dolly * progress;
        // The background point at the centre of the frame goes from the
        // opening view's centre to the focus point, and the focus point's
        // place in the frame closes on the centre in step with the shot's
        // progress (`lead` 1) or, as a straight line from where the image
        // was taken would have it, mostly at the end (`lead` 0). A truck
        // swings the camera sideways and back.
        let lead = self.lead.clamp(0.0, 1.0);
        let remaining = (1.0 - progress) * (lead + (1.0 - lead) / near);
        let aimed = |focus: f64, opening: f64| focus - (focus - opening) * remaining * near;
        let swing = 4.0 * progress * (1.0 - progress) * scene.background_distance_pc;
        self.place(
            scene,
            (opening, footprint),
            Placement {
                aimed: (
                    aimed(self.focus.0, opening.0),
                    aimed(self.focus.1, opening.1),
                ),
                near,
                pan: self.pan,
                swing: (self.truck.0 * swing, self.truck.1 * swing),
                magnification: self.zoom_end.max(1.0).powf(progress),
                turn: self.rotation.0 + (self.rotation.1 - self.rotation.0) * progress,
            },
        )
    }

    /// The camera `t` of the way through a tour.
    fn tour_view(&self, scene: &Scene, t: f64) -> View {
        let first = self.tour[0];
        let footprint =
            self.widest_footprint(first.focus, scene.width(), scene.height()) / self.zoom.max(1.0);
        let [x, y, log_near, log_zoom, turn, pan] = tour_state(&self.tour, t);
        self.place(
            scene,
            (first.focus, footprint),
            Placement {
                aimed: (x, y),
                near: log_near.exp().clamp(0.01, 1.0),
                pan,
                swing: (0.0, 0.0),
                magnification: log_zoom.exp() / first.zoom.max(1e-6),
                turn,
            },
        )
    }

    /// The camera that frames `placement`, its lens keeping the angle that
    /// framed the opening view of `footprint` centred on `opening`.
    fn place(
        &self,
        scene: &Scene,
        (opening, footprint): ((f64, f64), f64),
        placement: Placement,
    ) -> View {
        let distance = scene.background_distance_pc;
        let centre = (
            (scene.width() as f64 - 1.0) / 2.0,
            (scene.height() as f64 - 1.0) / 2.0,
        );
        let focal_out = scene.focal_px / footprint * placement.magnification;
        let near = placement.near;
        let pan = placement.pan.clamp(0.0, 1.0);
        // The lens keeps the angle that framed the opening view, and the
        // camera moves sideways to bring the aimed point to the centre, or
        // turns for `pan` of the way. Per axis: the camera's sideways
        // place, parsecs, and its lens angle as image pixels at the image's
        // focal length.
        let axis = |aimed: f64, opening: f64, centre: f64, swing: f64| {
            // Where the opening angle meets the background from here, and
            // how far the centre of the frame still has to go.
            let to_go = aimed - (centre + (opening - centre) * near);
            let across = (1.0 - pan) * to_go * distance / scene.focal_px + swing;
            let angle = opening - centre + pan * to_go / near;
            (across, angle)
        };
        let (across_x, angle_x) = axis(placement.aimed.0, opening.0, centre.0, placement.swing.0);
        let (across_y, angle_y) = axis(placement.aimed.1, opening.1, centre.1, placement.swing.1);
        let (sin, cos) = placement.turn.sin_cos();
        View {
            centre,
            focal_px: scene.focal_px,
            across: (across_x, across_y),
            along: (1.0 - near) * distance,
            focal_out,
            focus_shift: (
                focal_out * angle_x / scene.focal_px,
                focal_out * angle_y / scene.focal_px,
            ),
            out_centre: (
                (self.width as f64 - 1.0) / 2.0,
                (self.height as f64 - 1.0) / 2.0,
            ),
            turn: (cos, sin),
        }
    }

    /// A tour's length in seconds.
    pub fn tour_seconds(&self) -> f64 {
        tour_times(&self.tour)
            .last()
            .map_or(0.0, |&(_, leave)| leave)
    }

    /// This shot made to keep every layer's edge out of view, and the
    /// factor its truck was scaled by (for a tour, its stops' pan). A turning frame's first frame zooms
    /// in as far as it must. The camera's sideways travel toward
    /// the focus point comes as early as `lead` allows, it turns as much of
    /// the way as `pan` allows, and its truck swings as wide as the image
    /// allows: moving sideways slides layers at
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
        // The least zoom from the shot's own that the shot `make` builds
        // fits at.
        let least_zoom = |make: &dyn Fn(f64) -> Self| -> f64 {
            if fits(&make(self.zoom)) {
                return self.zoom;
            }
            let base = self.zoom.max(1.0);
            let (mut low, mut high) = (base, base * 2.0);
            while !fits(&make(high)) && high < base * 64.0 {
                (low, high) = (high, high * 2.0);
            }
            for _ in 0..30 {
                let middle = (low + high) / 2.0;
                if fits(&make(middle)) {
                    high = middle;
                } else {
                    low = middle;
                }
            }
            high
        };
        // A tour keeps its stops, so it is fitted by turning less and
        // zooming in. Turning toward a stop far to the side from near the
        // nebula sweeps the far star field off the image, which no zoom
        // mends, so the stops keep the largest share of their pan that fits
        // with each stop zoomed in by at most half again. Each stop zooms
        // only as far as the views about it need, so an opening on the
        // whole image stays whole.
        if self.tour.len() >= 2 {
            let panned = |share: f64| Self {
                tour: self
                    .tour
                    .iter()
                    .map(|stop| Stop {
                        pan: stop.pan * share,
                        ..*stop
                    })
                    .collect(),
                ..self.clone()
            };
            let zoomed = |share: f64| self.zoom_stops(scene, &depths, panned(share), 1.5);
            let share = if zoomed(1.0).is_some() {
                1.0
            } else {
                let (mut low, mut high) = (0.0, 1.0);
                for _ in 0..20 {
                    let middle = (low + high) / 2.0;
                    if zoomed(middle).is_some() {
                        low = middle;
                    } else {
                        high = middle;
                    }
                }
                low
            };
            // Past half again even without turning, the whole tour zooms in
            // together.
            let fitted = match zoomed(share) {
                Some(fitted) => fitted,
                None => {
                    let zoom = least_zoom(&|zoom| Self {
                        zoom,
                        ..panned(share)
                    });
                    Self {
                        zoom,
                        ..panned(share)
                    }
                }
            };
            return (fitted, share);
        }
        // A turned frame reaches past a square one's corners, so the first
        // frame zooms in until flying straight in, turning, fits.
        let straight = |zoom: f64| Self {
            zoom,
            lead: 0.0,
            pan: 0.0,
            truck: (0.0, 0.0),
            ..self.clone()
        };
        let zoom = if self.rotation != (0.0, 0.0) {
            least_zoom(&straight)
        } else {
            self.zoom
        };
        // Then the sideways travel, then turning, then the truck: each
        // takes what room the ones before it leave.
        let lead = largest(&|factor| Self {
            lead: self.lead * factor,
            ..straight(zoom)
        });
        let led = Self {
            lead: self.lead * lead,
            ..straight(zoom)
        };
        let pan = largest(&|factor| Self {
            pan: self.pan * factor,
            ..led.clone()
        });
        let led = Self {
            pan: self.pan * pan,
            truck: self.truck,
            ..led
        };
        let factor = largest(&|factor| Self {
            truck: (self.truck.0 * factor, self.truck.1 * factor),
            ..led.clone()
        });
        (
            Self {
                truck: (self.truck.0 * factor, self.truck.1 * factor),
                ..led
            },
            factor,
        )
    }

    /// `shot`, a tour, with each stop zoomed in by at most `most` times its
    /// own zoom, no further than keeps every view inside the image at
    /// `depths`, or `None` if that is not enough. A view that shows an edge
    /// zooms in the stop it is at, or the nearer of the two it lies
    /// between, a little at a time.
    fn zoom_stops(&self, scene: &Scene, depths: &[f64], mut shot: Self, most: f64) -> Option<Self> {
        let asked: Vec<f64> = shot.tour.iter().map(|stop| stop.zoom).collect();
        let times = tour_times(&shot.tour);
        let total = times.last().map_or(0.0, |&(_, leave)| leave);
        for _ in 0..400 {
            let mut bump = vec![false; shot.tour.len()];
            for frame in 0..shot.frames {
                if shot.inside(scene, &shot.view(scene, frame), depths) {
                    continue;
                }
                let now = frame as f64 / (shot.frames - 1).max(1) as f64 * total;
                let next = times
                    .iter()
                    .position(|&(arrive, _)| arrive >= now)
                    .unwrap_or(times.len() - 1);
                // The stop it is held at, or the nearer of the two it
                // travels between.
                let index = if next == 0 || now >= times[next].0 {
                    next
                } else if now - times[next - 1].1 < times[next].0 - now {
                    next - 1
                } else {
                    next
                };
                bump[index] = true;
            }
            if !bump.contains(&true) {
                return Some(shot);
            }
            for (index, stop) in shot.tour.iter_mut().enumerate() {
                if bump[index] {
                    if stop.zoom >= asked[index] * most {
                        return None;
                    }
                    stop.zoom = (stop.zoom * 1.02).min(asked[index] * most);
                }
            }
        }
        None
    }

    /// How much of a star drawn at `scale` output pixels per image pixel in
    /// `view` is left, 0 to 1: it fades as the camera passes it.
    pub fn star_fade(&self, view: &View, scale: f64) -> f64 {
        let growth = view.growth(scale);
        if growth <= self.fade_from {
            1.0
        } else {
            (2.0 - growth / self.fade_from).max(0.0)
        }
    }

    /// Render frame `frame` of `scene`.
    pub fn render(&self, scene: &Scene, frame: usize) -> LightImage {
        match self.quality {
            Quality::Standard => self.draw(scene, frame, false),
            Quality::High => {
                // The same view at twice the size; light adds, so the mean
                // of each 2×2 block is what the pixel it becomes would see.
                let finer = Self {
                    width: self.width * 2,
                    height: self.height * 2,
                    ..self.clone()
                };
                finer.draw(scene, frame, true).halved()
            }
        }
    }

    /// Draw frame `frame` at this shot's size, blending levels of detail
    /// if `blend`.
    fn draw(&self, scene: &Scene, frame: usize, blend: bool) -> LightImage {
        let view = self.view(scene, frame);
        let mut out = LightImage::new(self.width, self.height);
        draw_plane(
            &mut out,
            &scene.background,
            &view,
            scene.background_distance_pc,
            None,
            blend,
        );
        // The leftover star light lies behind the dust when it lies beyond
        // the background.
        let behind = scene
            .dust
            .as_ref()
            .filter(|_| scene.leftover_distance_pc > scene.background_distance_pc)
            .map(|dust| (dust, scene.background_distance_pc));
        draw_plane(
            &mut out,
            &scene.leftover,
            &view,
            scene.leftover_distance_pc,
            behind,
            blend,
        );
        draw_sprites(&mut out, scene, &view, self);
        out
    }
}

/// Where a camera is: the background point at the frame's centre, the
/// background's distance as a fraction of its distance from where the image
/// was taken, how much of the way to that point it turned, its sideways
/// swing in parsecs, its lens's magnification over the opening view's, and
/// the frame's turn, radians anticlockwise.
#[derive(Clone, Copy, Debug)]
struct Placement {
    aimed: (f64, f64),
    near: f64,
    pan: f64,
    swing: (f64, f64),
    magnification: f64,
    turn: f64,
}

/// When the camera reaches each stop of a tour and leaves it, seconds.
fn tour_times(tour: &[Stop]) -> Vec<(f64, f64)> {
    let mut times = Vec::with_capacity(tour.len());
    let mut clock = 0.0;
    for (index, stop) in tour.iter().enumerate() {
        if index > 0 {
            clock += stop.travel.max(0.0);
        }
        let arrive = clock;
        clock += stop.hold.max(0.0);
        times.push((arrive, clock));
    }
    times
}

/// A stop as numbers that blend smoothly: its focus, the logarithms of the
/// background's nearness and of the lens's zoom (so a flight in or a zoom
/// keeps an even pace), its turn and its pan.
fn stop_state(stop: &Stop) -> [f64; 6] {
    [
        stop.focus.0,
        stop.focus.1,
        (1.0 - stop.dolly.clamp(0.0, 0.99)).ln(),
        stop.zoom.max(1e-6).ln(),
        stop.rotation,
        stop.pan.clamp(0.0, 1.0),
    ]
}

/// The camera's state `t` of the way through `tour`: still at a stop it
/// holds at, and between stops a smooth curve through them that eases to a
/// halt at the ends and at each held stop and glides through the others.
fn tour_state(tour: &[Stop], t: f64) -> [f64; 6] {
    let times = tour_times(tour);
    let total = times.last().map_or(0.0, |&(_, leave)| leave);
    let now = t.clamp(0.0, 1.0) * total;
    // Each stop as the camera reaches it, its turn counted from the frame
    // the spins before it left.
    let mut spun = 0.0;
    let states: Vec<[f64; 6]> = tour
        .iter()
        .map(|stop| {
            let mut state = stop_state(stop);
            state[4] += spun;
            if stop.hold > 0.0 {
                spun += stop.spin;
            }
            state
        })
        .collect();
    // And as it leaves.
    let left = |index: usize| {
        let mut state = states[index];
        if tour[index].hold > 0.0 {
            state[4] += tour[index].spin;
        }
        state
    };
    let last = tour.len() - 1;
    // The rate of change through each stop, per second: none at the ends
    // and where the camera holds, else Catmull-Rom's.
    let rate = |index: usize| -> [f64; 6] {
        if index == 0 || index == last || tour[index].hold > 0.0 {
            return [0.0; 6];
        }
        let span = (times[index + 1].0 - times[index - 1].1).max(1e-9);
        std::array::from_fn(|k| (states[index + 1][k] - left(index - 1)[k]) / span)
    };
    for index in 0..=last {
        let (arrive, leave) = times[index];
        if now <= leave || index == last {
            if now >= arrive || index == 0 {
                // Held here, spinning as the stop asks.
                let mut state = states[index];
                if leave > arrive {
                    state[4] += tour[index].spin * spin_share(now - arrive, leave - arrive);
                }
                return state;
            }
            // On the way here from the stop before.
            let from = times[index - 1].1;
            let span = (arrive - from).max(1e-9);
            let s = ((now - from) / span).clamp(0.0, 1.0);
            // The turn and the pan need room the camera only has nearer
            // the nebula, so flying in they begin slowly and catch up, and
            // flying out they finish early.
            let (before, after) = (states[index - 1][2], states[index][2]);
            let late = if after < before {
                s * s
            } else if after > before {
                1.0 - (1.0 - s) * (1.0 - s)
            } else {
                s
            };
            let hermite = |s: f64| {
                (
                    2.0 * s.powi(3) - 3.0 * s * s + 1.0,
                    s.powi(3) - 2.0 * s * s + s,
                    -2.0 * s.powi(3) + 3.0 * s * s,
                    s.powi(3) - s * s,
                )
            };
            let (start, end) = (rate(index - 1), rate(index));
            let from_state = left(index - 1);
            return std::array::from_fn(|k| {
                let (h00, h10, h01, h11) = hermite(if k >= 4 { late } else { s });
                h00 * from_state[k]
                    + h10 * span * start[k]
                    + h01 * states[index][k]
                    + h11 * span * end[k]
            });
        }
    }
    left(last)
}

/// How much of a spin is done `elapsed` seconds into a hold of `hold`
/// seconds: at an even pace, easing up to it over the first second or
/// quarter of the hold and down from it over the last.
fn spin_share(elapsed: f64, hold: f64) -> f64 {
    let ramp = (hold / 4.0).min(1.0);
    let t = elapsed.clamp(0.0, hold);
    // The even pace that covers the whole spin, ramps included.
    let pace = 1.0 / (hold - ramp);
    if t < ramp {
        pace * t * t / (2.0 * ramp)
    } else if t <= hold - ramp {
        pace * (t - ramp / 2.0)
    } else {
        1.0 - pace * (hold - t) * (hold - t) / (2.0 * ramp)
    }
}

/// Rows each parallel band of the frame covers.
const BAND_ROWS: usize = 16;

/// Draw the plane `image` at `distance`, and if `dust` gives the dust and
/// the distance of the plane it lies on, in front, dim it by the change in
/// the dust it shows through. With `blend`, sample the levels of detail
/// either side of the view and blend them.
fn draw_plane(
    out: &mut LightImage,
    image: &Pyramid,
    view: &View,
    distance: f64,
    dust: Option<(&Dust, f64)>,
    blend: bool,
) {
    let Some(map) = view.plane_map(distance) else {
        return;
    };
    let dust = dust.and_then(|(dust, at)| view.plane_map(at).map(|front| (dust, front)));
    let footprint = map.across.0.hypot(map.across.1) as f32;
    let (finer, coarser, toward) = image.levels_between(footprint);
    let levels = if blend && toward > 0.0 {
        (finer, Some((coarser, toward)))
    } else {
        (image.level_for(footprint), None)
    };
    let width = out.width;
    out.pixels
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, pixels)| {
            plane_row(pixels, row, map, levels, dust);
        });
}

/// A level of detail and its scale relative to the base.
type Level<'a> = (&'a LightImage, f32);

/// One row of [`draw_plane`]: output row `row` shows the plane as `map`
/// has it, sampled from a level of detail, blended `toward` a coarser one
/// if given, and the dust as its map has it. Built for wider vector units too, and picked at run
/// time; it must do the work itself, as a closure it handed on would be
/// built without them.
#[multiversion::multiversion(targets("x86_64+avx2+fma", "x86_64+sse4.1"))]
fn plane_row(
    pixels: &mut [[f32; 3]],
    row: usize,
    map: PlaneMap,
    ((level, level_scale), coarser): (Level, Option<(Level, f32)>),
    dust: Option<(&Dust, PlaneMap)>,
) {
    let row = row as f64;
    for (column, pixel) in pixels.iter_mut().enumerate() {
        let (x, y) = map.at(column as f64, row);
        let mut light = Pyramid::sample_level(level, level_scale, x as f32, y as f32);
        if let Some(((coarse, coarse_scale), toward)) = coarser {
            let other = Pyramid::sample_level(coarse, coarse_scale, x as f32, y as f32);
            for channel in 0..3 {
                light[channel] += (other[channel] - light[channel]) * toward;
            }
        }
        if let Some((dust, front)) = dust {
            let change = dust.change((x, y), front.at(column as f64, row));
            light = light.map(|value| value * change);
        }
        for channel in 0..3 {
            pixel[channel] += light[channel];
        }
    }
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
    /// Cosine and sine of the frame's turn, which turns the sprite too.
    turn: (f64, f64),
    /// Output columns and rows it covers.
    left: f64,
    right: f64,
    top: f64,
    bottom: f64,
}

/// The least and greatest of `k · a` and `k · b`.
#[inline]
fn spread(k: f64, a: f64, b: f64) -> (f64, f64) {
    let (a, b) = (k * a, k * b);
    (a.min(b), a.max(b))
}

/// The indices `k` below `count` for which `start + k · step` lies from
/// `low` to `high`.
#[inline]
fn within(start: f64, step: f64, (low, high): (f64, f64), count: usize) -> std::ops::Range<usize> {
    if step == 0.0 {
        return if (low..=high).contains(&start) {
            0..count
        } else {
            0..0
        };
    }
    let (a, b) = ((low - start) / step, (high - start) / step);
    let first = a.min(b).ceil().max(0.0);
    let last = (a.max(b).floor() + 1.0).min(count as f64);
    if first < last {
        first as usize..last as usize
    } else {
        0..0
    }
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
            let growth = view.growth(scale);
            let fade = shot.star_fade(view, scale);
            if fade <= 0.0 {
                return None;
            }
            let grown = growth.clamp(1.0, shot.growth_limit);
            let size = lens_scale * grown.sqrt();
            // Behind the dust, the change in the dust it shows through.
            let dimming = match &scene.dust {
                Some(dust) if sprite.distance_pc > scene.background_distance_pc => {
                    let now = view.unproject(x, y, scene.background_distance_pc);
                    dust.change((sprite.x, sprite.y), now)
                }
                _ => 1.0,
            };
            let gain = (grown.powf(shot.brightening) * fade) as f32 * dimming;
            // Sprite pixel (i, j) sits at image (left + i, top + j), and
            // its corners from the centroid, in output pixels unturned, at:
            let west = (sprite.left as f64 - 1.0 - sprite.x) * size;
            let east = (sprite.left as f64 + sprite.image.width as f64 + 1.0 - sprite.x) * size;
            let north = (sprite.top as f64 - 1.0 - sprite.y) * size;
            let south = (sprite.top as f64 + sprite.image.height as f64 + 1.0 - sprite.y) * size;
            let (cos, sin) = view.turn;
            let (across, down) = (spread(cos, west, east), spread(sin, north, south));
            let (left, right) = (x + across.0 + down.0, x + across.1 + down.1);
            let (across, down) = (spread(-sin, west, east), spread(cos, north, south));
            let (top, bottom) = (y + across.0 + down.0, y + across.1 + down.1);
            if right < 0.0 || left > width || bottom < 0.0 || top > height {
                return None;
            }
            Some(Placed {
                sprite,
                x,
                y,
                scale: size,
                gain,
                turn: view.turn,
                left,
                right,
                top,
                bottom,
            })
        })
        .collect();
    // Which sprites each band of rows holds, so a band need not look
    // through them all.
    let bands = out.height.div_ceil(BAND_ROWS);
    let mut by_band: Vec<Vec<u32>> = vec![Vec::new(); bands];
    for (index, placed) in placed.iter().enumerate() {
        let first = (placed.top.max(0.0) as usize / BAND_ROWS).min(bands - 1);
        let last = (placed.bottom.max(0.0) as usize / BAND_ROWS).min(bands - 1);
        for band in &mut by_band[first..=last] {
            band.push(index as u32);
        }
    }
    let out_width = out.width;
    out.pixels
        .par_chunks_mut(out_width * BAND_ROWS)
        .zip(by_band.par_iter())
        .enumerate()
        .for_each(|(band, (pixels, indices))| {
            draw_band(pixels, out_width, band * BAND_ROWS, &placed, indices);
        });
}

/// The sprites `indices` of `placed` in the band of output rows starting
/// at `first`. Built for wider vector units too, as [`plane_row`] is.
#[multiversion::multiversion(targets("x86_64+avx2+fma", "x86_64+sse4.1"))]
fn draw_band(
    pixels: &mut [[f32; 3]],
    width: usize,
    first: usize,
    placed: &[Placed<'_>],
    indices: &[u32],
) {
    let rows = pixels.len() / width;
    for placed in indices
        .iter()
        .map(|&index| &placed[index as usize])
        .filter(|placed| placed.bottom >= first as f64 && placed.top < (first + rows) as f64)
    {
        if placed.scale >= 1.0 {
            sample_sprite(pixels, width, first, rows, placed);
        } else {
            splat_sprite(pixels, width, first, rows, placed);
        }
    }
}

/// Draw a sprite at least as large as its pixels by sampling it at each
/// output pixel it covers.
#[inline]
fn sample_sprite(
    pixels: &mut [[f32; 3]],
    width: usize,
    first: usize,
    rows: usize,
    placed: &Placed,
) {
    let sprite = placed.sprite;
    let left = placed.left.floor().max(0.0) as usize;
    let right = (placed.right.ceil().max(0.0) as usize).min(width);
    let top = (placed.top.floor().max(first as f64) as usize).max(first);
    let bottom = (placed.bottom.ceil().max(0.0) as usize).min(first + rows);
    // Output pixel (column, row) shows sprite pixel (sx, sy), turned back:
    // a step along the row moves it by `(cos, sin) / scale`.
    let (cos, sin) = placed.turn;
    let (step_x, step_y) = (cos / placed.scale, sin / placed.scale);
    for row in top..bottom {
        let (du, dv) = (left as f64 - placed.x, row as f64 - placed.y);
        let start_x = (cos * du - sin * dv) / placed.scale + sprite.x - sprite.left as f64;
        let start_y = (sin * du + cos * dv) / placed.scale + sprite.y - sprite.top as f64;
        for column in left..right {
            let k = (column - left) as f64;
            let (sx, sy) = (start_x + k * step_x, start_y + k * step_y);
            let light = sprite.image.sample(sx as f32, sy as f32);
            let pixel = &mut pixels[(row - first) * width + column];
            for channel in 0..3 {
                pixel[channel] += light[channel] * placed.gain;
            }
        }
    }
}

/// Draw a sprite smaller than its pixels by spreading each pixel's light
/// over the output pixels it covers, so a shrunk star keeps its total light
/// instead of flickering. Each pixel is a square shared by area, not a
/// point shared by distance: the squares tile the frame, so a smooth sprite
/// stays smooth however its pixels fall between the frame's, where points
/// would beat against the frame's grid in faint cross-hatching. It spreads
/// the halving of the sprite
/// whose pixels come nearest an output pixel without passing it, which
/// keeps that light with a fraction of the work, and only the rows that
/// land in this band.
#[inline]
fn splat_sprite(pixels: &mut [[f32; 3]], width: usize, first: usize, rows: usize, placed: &Placed) {
    let sprite = placed.sprite;
    let wanted = floor64((1.0 / placed.scale).log2()).max(0) as usize;
    let (image, level) = sprite.level(wanted);
    let step = (1_usize << level) as f64;
    // A halved pixel holds the mean of `step` × `step` pixels and is
    // centred on them.
    let offset = (step - 1.0) / 2.0;
    let scale = placed.scale * step;
    // A halved pixel's square, `scale` output pixels a side (at most one),
    // from `centre`: the first output pixel it covers, and how much of that
    // one and the next.
    let half = scale / 2.0;
    let shares = |centre: f64| {
        let low = centre - half;
        let first = floor64(low + 0.5);
        let edge = first as f64 + 0.5;
        (
            first,
            (edge.min(centre + half) - low) as f32,
            (centre + half - edge).max(0.0) as f32,
        )
    };
    // Halved pixel (i, j) lands at output `start + i · along + j · down`.
    let (cos, sin) = placed.turn;
    let west = (sprite.left as f64 + offset - sprite.x) * placed.scale;
    let north = (sprite.top as f64 + offset - sprite.y) * placed.scale;
    let start = (
        placed.x + cos * west + sin * north,
        placed.y - sin * west + cos * north,
    );
    let (along, down) = ((cos * scale, -sin * scale), (sin * scale, cos * scale));
    // A pixel lands on the rows either side of it, so those from just
    // above the band to its end; and the rows of halved pixels some of
    // whose pixels land there.
    let band = (first as f64 - 1.0, (first + rows) as f64);
    let reach = spread(along.1, 0.0, image.width as f64 - 1.0);
    let lines = within(
        start.1,
        down.1,
        (band.0 - reach.1, band.1 - reach.0),
        image.height,
    );
    for j in lines {
        let line = (start.0 + j as f64 * down.0, start.1 + j as f64 * down.1);
        for i in within(line.1, along.1, band, image.width) {
            let light = image.at(i, j);
            if light == [0.0; 3] {
                continue;
            }
            let (x, y) = (line.0 + i as f64 * along.0, line.1 + i as f64 * along.1);
            let (fx, left, right) = shares(x);
            let (fy, top, bottom) = shares(y);
            for (dy, wy) in [(0, top), (1, bottom)] {
                let row = fy + dy;
                if wy == 0.0 || row < first as isize || row >= (first + rows) as isize {
                    continue;
                }
                for (dx, wx) in [(0, left), (1, right)] {
                    let column = fx + dx;
                    if column < 0 || column >= width as isize {
                        continue;
                    }
                    let weight = wx * wy * placed.gain;
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
    fn a_shrunk_star_keeps_its_light_and_place_in_every_band() {
        // A 41-pixel star drawn at a fifth of its size, so from a halving,
        // straddling two bands.
        let mut image = LightImage::new(41, 41);
        for (index, pixel) in image.pixels.iter_mut().enumerate() {
            let (x, y) = ((index % 41) as f64 - 20.0, (index / 41) as f64 - 20.0);
            *pixel = [(-(x * x + y * y) / 40.0).exp() as f32; 3];
        }
        let total: f32 = image.pixels.iter().map(|pixel| pixel[0]).sum();
        let sprite = Sprite::new(100, 200, image, 120.0, 220.0, 50.0);
        let scale = 0.2;
        // Unturned, and turned so its square reaches furthest into the
        // bands either side.
        for angle in [0.0_f64, 0.7] {
            let (sin, cos) = angle.sin_cos();
            let reach = 21.0 * scale * (cos + sin);
            let placed = Placed {
                sprite: &sprite,
                x: 30.3,
                y: 15.6,
                scale,
                gain: 1.0,
                turn: (cos, sin),
                left: 30.3 - reach,
                right: 30.3 + reach,
                top: 15.6 - reach,
                bottom: 15.6 + reach,
            };
            let width = 64;
            let mut out = vec![[0.0_f32; 3]; width * 32];
            for (band, pixels) in out.chunks_mut(width * BAND_ROWS).enumerate() {
                let rows = pixels.len() / width;
                splat_sprite(pixels, width, band * BAND_ROWS, rows, &placed);
            }
            let drawn: f32 = out.iter().map(|pixel| pixel[0]).sum();
            let expected = total * (scale * scale) as f32;
            assert!(
                (drawn / expected - 1.0).abs() < 0.01,
                "{angle}: {drawn} of {expected}"
            );
            let (mut cx, mut cy) = (0.0_f64, 0.0_f64);
            for (index, pixel) in out.iter().enumerate() {
                cx += (index % width) as f64 * pixel[0] as f64;
                cy += (index / width) as f64 * pixel[0] as f64;
            }
            let (cx, cy) = (cx / drawn as f64, cy / drawn as f64);
            assert!(
                (cx - 30.3).abs() < 0.1 && (cy - 15.6).abs() < 0.1,
                "{angle}: ({cx}, {cy})"
            );
        }
    }

    #[test]
    fn a_high_quality_frame_is_the_same_view_drawn_finer() {
        let scene = scene_with(&[Star {
            x: 230.0,
            y: 160.0,
            distance_pc: Some(300.0),
        }]);
        let shot = Shot {
            focus: (199.5, 149.5),
            dolly: 0.3,
            width: 200,
            height: 150,
            frames: 3,
            ..Shot::default()
        };
        let high = Shot {
            quality: Quality::High,
            ..shot.clone()
        };
        for frame in 0..3 {
            let (standard, fine) = (shot.render(&scene, frame), high.render(&scene, frame));
            assert_eq!((fine.width, fine.height), (200, 150));
            // The same light in the same places: the gradient background
            // and the star agree pixel by pixel to within the finer
            // sampling.
            let total = |image: &LightImage| -> f64 {
                image.pixels.iter().map(|pixel| pixel[0] as f64).sum()
            };
            assert!(
                (total(&fine) / total(&standard) - 1.0).abs() < 0.01,
                "frame {frame}"
            );
            let (a, b) = (brightest(&fine), brightest(&standard));
            assert!(
                a.0.abs_diff(b.0) <= 1 && a.1.abs_diff(b.1) <= 1,
                "frame {frame}: {a:?} {b:?}"
            );
        }
    }

    #[test]
    fn levels_between_blend_toward_the_coarser_as_the_view_widens() {
        let pyramid = Pyramid::new(LightImage::new(512, 512));
        let ((_, finer), (_, coarser), toward) = pyramid.levels_between(2.0);
        assert_eq!((finer, coarser, toward), (2.0, 4.0, 0.0));
        let ((_, finer), (_, coarser), toward) = pyramid.levels_between(2.0_f32.powf(1.5));
        assert_eq!((finer, coarser), (2.0, 4.0));
        assert!((toward - 0.5).abs() < 1e-6);
        let (_, _, toward) = pyramid.levels_between(0.7);
        assert_eq!(toward, 0.0);
    }

    #[test]
    fn a_smooth_sprite_drawn_small_stays_smooth() {
        // An even patch of light, drawn at scales whose halvings land a
        // little under an output pixel apart, where points shared by
        // distance beat against the frame's grid.
        let image = LightImage {
            width: 400,
            height: 400,
            pixels: vec![[1.0; 3]; 400 * 400],
        };
        let sprite = Sprite::new(0, 0, image, 200.0, 200.0, 50.0);
        for scale in [0.37, 0.29, 0.6, 0.153] {
            let reach = 201.0 * scale;
            let placed = Placed {
                sprite: &sprite,
                x: 100.3,
                y: 100.6,
                scale,
                gain: 1.0,
                turn: (1.0, 0.0),
                left: 100.3 - reach,
                right: 100.3 + reach,
                top: 100.6 - reach,
                bottom: 100.6 + reach,
            };
            let width = 200;
            let mut out = vec![[0.0_f32; 3]; width * 200];
            for (band, pixels) in out.chunks_mut(width * BAND_ROWS).enumerate() {
                let rows = pixels.len() / width;
                splat_sprite(pixels, width, band * BAND_ROWS, rows, &placed);
            }
            // Well inside the patch every pixel holds the same light.
            let inner = (reach * 0.8) as usize;
            let (low, high) = (100 - inner, 100 + inner);
            let values: Vec<f32> = (low..high)
                .flat_map(|y| (low..high).map(move |x| (x, y)))
                .map(|(x, y)| out[y * width + x][0])
                .collect();
            let (least, most) = values
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            assert!(
                (least - 1.0).abs() < 1e-3 && (most - 1.0).abs() < 1e-3,
                "scale {scale}: {least}..{most}"
            );
        }
    }

    #[test]
    fn a_turned_star_turns_its_shape_with_the_frame() {
        // A star twice as long across as down, drawn by sampling and by
        // spreading, turned a quarter: it lies twice as long down.
        let mut image = LightImage::new(41, 41);
        for (index, pixel) in image.pixels.iter_mut().enumerate() {
            let (x, y) = ((index % 41) as f64 - 20.0, (index / 41) as f64 - 20.0);
            *pixel = [(-(x * x / 4.0 + y * y) / 8.0).exp() as f32; 3];
        }
        let sprite = Sprite::new(0, 0, image, 20.0, 20.0, 50.0);
        let (sin, cos) = std::f64::consts::FRAC_PI_2.sin_cos();
        // Spread at 0.8, sampled at 1.5.
        for scale in [0.8, 1.5] {
            let reach = 21.0 * scale * (cos.abs() + sin.abs());
            let placed = Placed {
                sprite: &sprite,
                x: 40.0,
                y: 40.0,
                scale,
                gain: 1.0,
                turn: (cos, sin),
                left: 40.0 - reach,
                right: 40.0 + reach,
                top: 40.0 - reach,
                bottom: 40.0 + reach,
            };
            let width = 80;
            let mut out = vec![[0.0_f32; 3]; width * 80];
            for (band, pixels) in out.chunks_mut(width * BAND_ROWS).enumerate() {
                draw_band(
                    pixels,
                    width,
                    band * BAND_ROWS,
                    std::slice::from_ref(&placed),
                    &[0],
                );
            }
            let (mut across, mut down, mut total) = (0.0_f64, 0.0_f64, 0.0_f64);
            for (index, pixel) in out.iter().enumerate() {
                let (x, y) = ((index % width) as f64 - 40.0, (index / width) as f64 - 40.0);
                let light = pixel[0] as f64;
                across += x * x * light;
                down += y * y * light;
                total += light;
            }
            let ratio = (down / total).sqrt() / (across / total).sqrt();
            assert!((ratio - 2.0).abs() < 0.15, "scale {scale}: {ratio}");
        }
    }

    #[test]
    fn a_turning_frame_turns_every_depth_about_its_centre() {
        let scene = scene_with(&[]);
        let shot = Shot {
            focus: (199.5, 149.5),
            dolly: 0.0,
            truck: (0.0, 0.0),
            rotation: (0.0, std::f64::consts::FRAC_PI_2),
            width: 200,
            height: 150,
            frames: 2,
            easing: Easing::Linear,
            ..Shot::default()
        };
        let (first, last) = (shot.view(&scene, 0), shot.view(&scene, 1));
        // A point right of the focus point, at any depth, ends a quarter
        // turn anticlockwise: above the centre, as far from it.
        for distance in [50.0, 400.0, 5000.0] {
            let (x0, y0, _) = first.project(219.5, 149.5, distance).unwrap();
            let (x1, y1, _) = last.project(219.5, 149.5, distance).unwrap();
            assert!((y0 - 74.5).abs() < 1e-9 && x0 > 99.5, "({x0}, {y0})");
            assert!((x1 - 99.5).abs() < 1e-9, "({x1}, {y1})");
            assert!(((74.5 - y1) - (x0 - 99.5)).abs() < 1e-9, "({x1}, {y1})");
            // And the frame shows it there.
            let (x, y) = last.unproject(x1, y1, distance);
            assert!((x - 219.5).abs() < 1e-9 && (y - 149.5).abs() < 1e-9);
        }
    }

    #[test]
    fn a_turning_shot_zooms_in_to_stay_inside_the_image() {
        let scene = scene_with(&[]);
        let shot = Shot {
            focus: (199.5, 149.5),
            dolly: 0.3,
            truck: (0.0, 0.0),
            rotation: (0.0, std::f64::consts::FRAC_PI_4),
            width: 200,
            height: 150,
            frames: 20,
            ..Shot::default()
        };
        let depths = shot.guarded_depths(&scene);
        assert!(!(0..shot.frames).all(|frame| shot.inside(
            &scene,
            &shot.view(&scene, frame),
            &depths
        )));
        let (fitted, _) = shot.fitted(&scene);
        assert!(fitted.zoom > 1.0, "{}", fitted.zoom);
        for frame in 0..fitted.frames {
            let view = fitted.view(&scene, frame);
            assert!(fitted.inside(&scene, &view, &depths), "frame {frame}");
        }
        // No further in than it must: a little less and a frame shows past
        // the image.
        let wider = Shot {
            zoom: fitted.zoom * 0.99,
            ..fitted.clone()
        };
        assert!(!(0..wider.frames).all(|frame| wider.inside(
            &scene,
            &wider.view(&scene, frame),
            &depths
        )));
        // An unturned shot is left as it was.
        let still = Shot {
            rotation: (0.0, 0.0),
            ..shot.clone()
        };
        assert_eq!(still.fitted(&scene).0.zoom, 1.0);
    }

    #[test]
    fn a_pan_turns_part_of_the_way_and_still_ends_on_the_focus() {
        let scene = scene_with(&[]);
        let shot = Shot {
            focus: (320.0, 60.0),
            dolly: 0.5,
            truck: (0.0, 0.0),
            start: Start::Whole,
            pan: 0.3,
            width: 200,
            height: 150,
            frames: 10,
            ..Shot::default()
        };
        let angle = |view: &View| view.focus_shift.0 / view.focal_out;
        let (first, last) = (shot.view(&scene, 0), shot.view(&scene, 9));
        // The lens turns toward the focus point, and the camera moves less.
        assert!(angle(&last) > angle(&first));
        let unpanned = Shot {
            pan: 0.0,
            ..shot.clone()
        }
        .view(&scene, 9);
        assert!(last.across.0.abs() < unpanned.across.0.abs());
        let (x, y, _) = last.project(320.0, 60.0, 400.0).unwrap();
        assert!(
            (x - 99.5).abs() < 1e-6 && (y - 74.5).abs() < 1e-6,
            "({x}, {y})"
        );
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
            ..shot.clone()
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

#[cfg(test)]
mod edge_tests {
    use super::*;
    use crate::scene::CutOptions;

    #[test]
    fn a_whole_start_first_frame_counts_as_inside_whatever_the_scale() {
        // The first frame spans the image edge to edge; rounding must not
        // put its corners a hair outside, which would leave no room for
        // any sideways travel at all.
        let mut outside = Vec::new();
        for (width, height) in [(9595, 6346), (4000, 2671), (6248, 4176), (3001, 1999)] {
            let image = LightImage::new(width, height);
            let mut scene = Scene::new(
                &image,
                &image,
                &[],
                355.0,
                1134.0,
                150_000.0,
                &CutOptions::default(),
            );
            for step in 0..40 {
                let focal_px = 150_000.0 + step as f64 * 1234.567;
                scene.focal_px = focal_px;
                let shot = Shot {
                    focus: (width as f64 * 0.625, height as f64 * 0.37),
                    start: Start::Whole,
                    dolly: 0.8,
                    width: 1920,
                    height: 1080,
                    frames: 2,
                    ..Shot::default()
                };
                let view = shot.view(&scene, 0);
                if !shot.inside(&scene, &view, &shot.guarded_depths(&scene)) {
                    outside.push((width, focal_px));
                }
            }
        }
        assert!(outside.is_empty(), "{outside:?}");
    }
}

#[cfg(test)]
mod tour_tests {
    use super::*;
    use crate::scene::CutOptions;

    fn scene() -> Scene {
        let image = LightImage::new(600, 400);
        Scene::new(
            &image,
            &image,
            &[],
            400.0,
            1500.0,
            2000.0,
            &CutOptions::default(),
        )
    }

    fn tour() -> Shot {
        Shot {
            tour: vec![
                Stop {
                    focus: (299.5, 199.5),
                    hold: 1.0,
                    ..Stop::default()
                },
                Stop {
                    focus: (200.0, 150.0),
                    dolly: 0.6,
                    rotation: 0.3,
                    pan: 0.2,
                    travel: 3.0,
                    hold: 1.0,
                    ..Stop::default()
                },
                Stop {
                    focus: (420.0, 260.0),
                    dolly: 0.4,
                    zoom: 1.5,
                    rotation: -0.2,
                    travel: 2.0,
                    ..Stop::default()
                },
                Stop {
                    focus: (299.5, 199.5),
                    travel: 3.0,
                    ..Stop::default()
                },
            ],
            width: 300,
            height: 200,
            // Ten seconds at ten frames a second, a frame on each tenth.
            frames: 101,
            ..Shot::default()
        }
    }

    #[test]
    fn a_tour_arrives_at_each_stop_framed_as_asked() {
        let scene = scene();
        let shot = tour();
        assert_eq!(shot.tour_seconds(), 10.0);
        // Frame 10k is k/10 of the way, so second k.
        for (frame, stop) in [(0, 0), (40, 1), (70, 2), (100, 3)] {
            let view = shot.view(&scene, frame);
            let stop = shot.tour[stop];
            let (x, y, _) = view.project(stop.focus.0, stop.focus.1, 400.0).unwrap();
            assert!(
                (x - 149.5).abs() < 1e-6 && (y - 99.5).abs() < 1e-6,
                "frame {frame}: ({x}, {y})"
            );
            assert!((view.turn() - stop.rotation).abs() < 1e-9, "frame {frame}");
            assert!(
                (view.along - stop.dolly * 400.0).abs() < 1e-6,
                "frame {frame}"
            );
        }
    }

    #[test]
    fn a_tour_holds_still_and_moves_smoothly_between_stops() {
        let scene = scene();
        let shot = tour();
        // Held at the second stop from second 4 to 5.
        let (a, b) = (shot.view(&scene, 40), shot.view(&scene, 50));
        assert_eq!(
            a.project(100.0, 100.0, 900.0),
            b.project(100.0, 100.0, 900.0)
        );
        // No jumps: a point's place changes by a bounded step each frame.
        let place = |frame| {
            shot.view(&scene, frame)
                .project(299.5, 199.5, 400.0)
                .unwrap()
        };
        let steps: Vec<f64> = (1..=100)
            .map(|frame| {
                let (p, q) = (place(frame - 1), place(frame));
                (q.0 - p.0).hypot(q.1 - p.1)
            })
            .collect();
        let biggest = steps.iter().cloned().fold(0.0, f64::max);
        for pair in steps.windows(2) {
            assert!(
                (pair[1] - pair[0]).abs() <= biggest * 0.35 + 1e-9,
                "a jerk: {pair:?} of at most {biggest}"
            );
        }
    }

    #[test]
    fn a_spin_turns_the_held_frame_at_an_even_pace_and_later_stops_follow_on() {
        // The share of a spin done eases in, runs evenly and eases out.
        assert_eq!(spin_share(0.0, 8.0), 0.0);
        assert!((spin_share(8.0, 8.0) - 1.0).abs() < 1e-12);
        let pace = |t: f64| (spin_share(t + 0.01, 8.0) - spin_share(t, 8.0)) / 0.01;
        assert!((pace(3.0) - pace(5.0)).abs() < 1e-9 && pace(0.1) < pace(3.0));
        let mut last = 0.0;
        for step in 1..=80 {
            let share = spin_share(step as f64 * 0.1, 8.0);
            assert!(share >= last);
            last = share;
        }
        // A whole turn while holding at the second stop, then back to the
        // whole view: the turn grows through the hold and is not undone.
        let scene = scene();
        let shot = Shot {
            tour: vec![
                Stop {
                    focus: (299.5, 199.5),
                    ..Stop::default()
                },
                Stop {
                    focus: (299.5, 199.5),
                    dolly: 0.6,
                    travel: 2.0,
                    hold: 8.0,
                    spin: std::f64::consts::TAU,
                    ..Stop::default()
                },
                Stop {
                    focus: (299.5, 199.5),
                    travel: 2.0,
                    ..Stop::default()
                },
            ],
            width: 300,
            height: 200,
            // Twelve seconds, a frame each tenth.
            frames: 121,
            ..Shot::default()
        };
        let turn = |frame: usize| tour_state(&shot.tour, frame as f64 / 120.0)[4];
        assert_eq!(turn(20), 0.0);
        assert!(
            (turn(60) - std::f64::consts::PI).abs() < 1e-9,
            "{}",
            turn(60)
        );
        assert!((turn(100) - std::f64::consts::TAU).abs() < 1e-9);
        assert!((turn(120) - std::f64::consts::TAU).abs() < 1e-9);
        let view = shot.view(&scene, 120);
        assert!(view.turn().abs() < 1e-9);
    }

    #[test]
    fn a_tour_that_would_show_an_edge_zooms_in_until_it_does_not() {
        let scene = scene();
        let shot = tour();
        let depths = shot.guarded_depths(&scene);
        let fits = |shot: &Shot| {
            (0..shot.frames).all(|frame| shot.inside(&scene, &shot.view(&scene, frame), &depths))
        };
        assert!(!fits(&shot), "the turning and panning tour shows an edge");
        let (fitted, share) = shot.fitted(&scene);
        assert!(fits(&fitted));
        for (fitted, stop) in fitted.tour.iter().zip(&shot.tour) {
            assert_eq!(fitted.focus, stop.focus);
            assert!(fitted.zoom >= stop.zoom && fitted.zoom <= stop.zoom * 1.5 + 1e-9);
            assert!((fitted.pan - stop.pan * share).abs() < 1e-12);
        }
        // Only the stops whose views need it zoom in: the opening, on the
        // whole image, stays as it was.
        assert_eq!(fitted.zoom, shot.zoom);
        assert_eq!(fitted.tour[0].zoom, 1.0);
        assert!(
            fitted
                .tour
                .iter()
                .zip(&shot.tour)
                .any(|(a, b)| a.zoom > b.zoom)
        );
        // Turning hard toward a stop at the image's edge from near the
        // nebula cannot fit at any zoom; the pan gives way.
        let mut hard = tour();
        hard.tour[1] = Stop {
            focus: (300.0, 30.0),
            dolly: 0.8,
            pan: 1.0,
            ..hard.tour[1]
        };
        let (fitted, share) = hard.fitted(&scene);
        assert!(share < 1.0 && fits(&fitted), "{share} {}", fitted.zoom);
    }
}
