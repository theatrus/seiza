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

use crate::dust::{Dust, MAX_BRIGHTENING};
use crate::light::{LightImage, Pyramid, floor64};
use crate::plane::{Plane, Transmission};
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

/// The most of a frame a tour's fit lets show the smooth fill left behind
/// a depth plane, as [`Shot::fill_shown`] measures it. Past it the fill reads
/// as a smear where the photograph had texture.
pub const FILL_SHOWN: f64 = 0.15;

/// Each step of the fit flies a stop showing too much fill this share of
/// its way in, until it flies no farther than [`LEAST_FLATTENED`].
const FLATTEN_STEP: f64 = 0.9;
const LEAST_FLATTENED: f64 = 0.05;

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
    /// The lens's magnification over the widest view of the first stop's
    /// place (as [`Shot::zoom`] narrows it).
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
    /// anticlockwise. It eases in over the last second of the way here, as
    /// the camera slows, and out over the first second of the way on, as it
    /// gathers pace, so the view never stands still. Later stops' turns
    /// count from the frame as it is left, so a whole turn does not unwind
    /// on the way on.
    pub spin: f64,
    /// How much of its remaining way to the nebula the camera flies on
    /// while it holds here, 0 to below 1: a slow push in, which with a
    /// spin turns the near stars in a spiral past the far ones.
    pub push: f64,
    /// The distance of the plane the focus point lies on, as a share of
    /// the background's: the camera centres the focus point on that plane.
    /// A camera that has moved sideways sees a nearer or farther plane
    /// shifted against the background, so it aims at the background point
    /// that puts the focus point at the centre. 1 on the background.
    pub depth: f64,
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
            push: 0.0,
            depth: 1.0,
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
    /// stops in turn, the first being the opening view, slowing through
    /// each stop it holds at. Empty for the single move, which the fields
    /// above describe; a tour uses only `zoom` of them.
    pub tour: Vec<Stop>,
    /// How fast a tour moves on through a stop it holds at, as a share of
    /// its pace between stops: it never quite stops. At 0 it comes to rest.
    pub glide: f64,
    /// A tour that loops: its last stop shows its first stop's view, the
    /// frame after the last is the first, and the camera drifts through
    /// that join at one steady pace. Unlooped, the last stop's hold is a
    /// final drift that slows to rest.
    pub looped: bool,
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
            glide: 0.2,
            looped: false,
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
    /// leftover star light and the far sprites, and the depth planes that
    /// reach the image's edges, on those edges.
    fn guarded_depths(&self, scene: &Scene) -> Vec<Guard> {
        let mut guards = vec![
            Guard::whole(scene.background_distance_pc),
            Guard::whole(scene.leftover_distance_pc),
            Guard::whole(scene.far_distance_pc),
        ];
        for plane in scene.planes.iter() {
            let edges = plane.cut_edges(scene.width(), scene.height());
            if edges.contains(&true) {
                guards.push(Guard {
                    distance: plane.distance_pc,
                    edges,
                    fades: true,
                });
            }
        }
        guards
    }

    /// Whether every frame's view keeps every layer's edge out of sight, as
    /// the fit aims for. A fit can fall short, such as for a tour stop
    /// too near the image's edge to turn there.
    pub fn stays_inside(&self, scene: &Scene) -> bool {
        self.inside_throughout(scene, &self.guarded_depths(scene))
    }

    /// Whether every frame's view lies inside the image at `depths`.
    fn inside_throughout(&self, scene: &Scene, depths: &[Guard]) -> bool {
        (0..self.frames)
            .into_par_iter()
            .all(|frame| self.inside(scene, &self.view(scene, frame), depths))
    }

    /// Whether every corner of `view` lies inside the guarded edges of the
    /// image at `depths`, but for a plane gone from view. A whole-image
    /// start puts the corners on the image's edges, so rounding a millionth
    /// of a pixel past them still counts as inside.
    fn inside(&self, scene: &Scene, view: &View, depths: &[Guard]) -> bool {
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
        depths.iter().all(|guard| {
            if guard.fades && self.plane_fade(view, guard.distance) <= 0.0 {
                return true;
            }
            let [left, top, right, bottom] = guard.edges;
            corners.iter().all(|&(u, v)| {
                let (x, y) = view.unproject(u, v, guard.distance);
                (!left || x >= -SLACK)
                    && (!top || y >= -SLACK)
                    && (!right || x <= width)
                    && (!bottom || y <= height)
            })
        })
    }

    /// How far through the shot frame `frame` is, 0 to 1. A looped tour's
    /// last frame comes a frame before the end, which is the first again.
    pub fn progress(&self, frame: usize) -> f64 {
        if self.looped && self.tour.len() >= 2 {
            frame as f64 / self.frames.max(1) as f64
        } else if self.frames > 1 {
            frame as f64 / (self.frames - 1) as f64
        } else {
            0.0
        }
    }

    /// The camera at frame `frame` of `scene`.
    pub fn view(&self, scene: &Scene, frame: usize) -> View {
        let t = self.progress(frame);
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
        // Each stop as the background point it aims at to centre its focus
        // point on its plane. The camera glides between those, so it keeps
        // to the image between stops, as on the background alone.
        let centre = (
            (scene.width() as f64 - 1.0) / 2.0,
            (scene.height() as f64 - 1.0) / 2.0,
        );
        let aimed: Vec<Stop> = self
            .tour
            .iter()
            .map(|stop| {
                let camera = (1.0 - stop.dolly.clamp(0.0, 0.99), stop.pan.clamp(0.0, 1.0));
                let through = |target: f64, opening: f64, centre: f64| {
                    aim_through(target, stop.depth, camera, (opening, centre))
                };
                Stop {
                    focus: (
                        through(stop.focus.0, first.focus.0, centre.0),
                        through(stop.focus.1, first.focus.1, centre.1),
                    ),
                    ..*stop
                }
            })
            .collect();
        let [x, y, log_near, log_zoom, turn, pan] = tour_state(&aimed, self.glide, self.looped, t);
        self.place(
            scene,
            (first.focus, footprint),
            Placement {
                aimed: (x, y),
                near: log_near.exp().clamp(0.01, 1.0),
                pan,
                swing: (0.0, 0.0),
                magnification: log_zoom.exp(),
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

    /// When the camera reaches each stop of the tour and leaves it,
    /// seconds from the start.
    pub fn stop_times(&self) -> Vec<(f64, f64)> {
        tour_times(&self.tour)
    }

    /// A tour's length in seconds.
    pub fn tour_seconds(&self) -> f64 {
        tour_times(&self.tour)
            .last()
            .map_or(0.0, |&(_, leave)| leave)
    }

    /// This shot made to keep every layer's edge out of view, and the
    /// factor its truck was scaled by (for a tour, its stops' pan), as
    /// [`Self::fitted_until`] finds them with nothing to stop it.
    pub fn fitted(&self, scene: &Scene) -> (Self, f64) {
        self.fitted_until(scene, &|| false)
            .expect("a fit nothing stops finishes")
    }

    /// [`Self::fitted`], asking `stop` as it goes and giving `None` once it
    /// answers true. A turning frame's first frame zooms
    /// in as far as it must. The camera's sideways travel toward
    /// the focus point comes as early as `lead` allows, it turns as much of
    /// the way as `pan` allows, and its truck swings as wide as the image
    /// allows: moving sideways slides layers at
    /// different depths against each other, and the far ones would uncover
    /// ground the image never showed. Moving along a straight line from
    /// where the image was taken never does.
    pub fn fitted_until(&self, scene: &Scene, stop: &dyn Fn() -> bool) -> Option<(Self, f64)> {
        // Once asked to stop, every check fails at once, so each search
        // ends quickly, and the result is thrown away.
        let stopped = std::cell::Cell::new(false);
        let halted = || {
            if !stopped.get() && stop() {
                stopped.set(true);
            }
            stopped.get()
        };
        let fitted = self.fit(scene, &halted);
        (!stopped.get()).then_some(fitted)
    }

    /// The fit itself, every check failing once `halted` answers true.
    fn fit(&self, scene: &Scene, halted: &dyn Fn() -> bool) -> (Self, f64) {
        let depths = self.guarded_depths(scene);
        let fits = |shot: &Self| !halted() && shot.inside_throughout(scene, &depths);
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
            // A stop on a depth plane keeps its pan: the camera turns to
            // see a plane beyond the nebula, which moving sideways cannot
            // centre.
            let panned = |share: f64| Self {
                tour: self
                    .tour
                    .iter()
                    .map(|stop| Stop {
                        pan: if (stop.depth - 1.0).abs() > 1e-9 {
                            stop.pan
                        } else {
                            stop.pan * share
                        },
                        ..*stop
                    })
                    .collect(),
                ..self.clone()
            };
            let zoomed = |share: f64| self.zoom_stops(scene, &depths, panned(share), 1.5, halted);
            let share = if zoomed(1.0).is_some() {
                1.0
            } else if zoomed(0.0).is_none() {
                // Even without turning some stop needs more than half
                // again, which the pan is not to blame for: the whole tour
                // zooms in together, keeping what pan fits at that zoom.
                let zoom = least_zoom(&|zoom| Self {
                    zoom,
                    ..panned(0.0)
                });
                let share = largest(&|share| Self {
                    zoom,
                    ..panned(share)
                });
                return (
                    Self {
                        zoom,
                        ..panned(share)
                    },
                    share,
                );
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
                Some(fitted) => self.with_less_fill(scene, &depths, fitted, &panned(share), halted),
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
        // A whole-image opening whose focus lies near an edge also ends on
        // a view past the edge unless zoomed; the search keeps the shot's
        // own zoom when it fits.
        let zoom = least_zoom(&straight);
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

    /// `fitted`, the tour `asked` with its stops zoomed to keep inside the
    /// image at `depths`, or one showing less of the fill behind its
    /// planes: while a frame shows more than [`FILL_SHOWN`] of it, the stop
    /// it belongs to flies less far and zooms in to keep its plane's size,
    /// a little at a time, as far as the stops can still be zoomed to keep
    /// inside the image.
    fn with_less_fill(
        &self,
        scene: &Scene,
        depths: &[Guard],
        fitted: Self,
        asked: &Self,
        halted: &dyn Fn() -> bool,
    ) -> Self {
        if scene.planes.is_empty() {
            return fitted;
        }
        let times = tour_times(&asked.tour);
        let total = times.last().map_or(0.0, |&(_, leave)| leave);
        let mut shares = vec![1.0; asked.tour.len()];
        let mut best = fitted;
        for _ in 0..200 {
            if halted() {
                break;
            }
            // The frame showing the most fill, if too much, flies less
            // far at the stop it is held at, or of the two it travels
            // between, the nearer that still can.
            let fill = best.fill_by_frame(scene);
            let Some((frame, _)) = fill
                .iter()
                .enumerate()
                .filter(|&(_, &shown)| shown > FILL_SHOWN)
                .max_by(|a, b| a.1.total_cmp(b.1))
            else {
                break;
            };
            let (nearer, other) = stops_at(&times, best.progress(frame) * total);
            let can = |index: usize| asked.tour[index].dolly * shares[index] > LEAST_FLATTENED;
            let Some(index) = [nearer, other].into_iter().find(|&index| can(index)) else {
                break;
            };
            let mut flatten = vec![false; shares.len()];
            flatten[index] = true;
            if best.looped {
                joined(&mut flatten);
            }
            for (index, flat) in flatten.into_iter().enumerate() {
                if flat {
                    shares[index] *= FLATTEN_STEP;
                }
            }
            let flatter = Self {
                tour: asked
                    .tour
                    .iter()
                    .zip(&shares)
                    .map(|(stop, &share)| flattened(stop, share))
                    .collect(),
                ..asked.clone()
            };
            match self.zoom_stops(scene, depths, flatter, 1.5, halted) {
                Some(shot) => best = shot,
                // The image's edges come first.
                None => break,
            }
        }
        best
    }

    /// `shot`, a tour, with each stop zoomed in by at most `most` times its
    /// own zoom, no further than keeps every view inside the image at
    /// `depths`, or `None` if that is not enough. A view that shows an edge
    /// zooms in the stop it is at, or the nearer of the two it lies
    /// between, a little at a time.
    fn zoom_stops(
        &self,
        scene: &Scene,
        depths: &[Guard],
        mut shot: Self,
        most: f64,
        halted: &dyn Fn() -> bool,
    ) -> Option<Self> {
        let asked: Vec<f64> = shot.tour.iter().map(|stop| stop.zoom).collect();
        let times = tour_times(&shot.tour);
        let total = times.last().map_or(0.0, |&(_, leave)| leave);
        for _ in 0..400 {
            if halted() {
                return None;
            }
            let mut bump = vec![false; shot.tour.len()];
            let outside: Vec<usize> = (0..shot.frames)
                .into_par_iter()
                .filter(|&frame| !shot.inside(scene, &shot.view(scene, frame), depths))
                .collect();
            for frame in outside {
                bump[stops_at(&times, shot.progress(frame) * total).0] = true;
            }
            if shot.looped {
                joined(&mut bump);
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
        self.fade(view.growth(scale))
    }

    /// How much of a depth plane at `distance` is left in `view`, 0 to 1:
    /// it fades as the camera passes through it, as a star does, and is
    /// gone once behind the camera.
    pub fn plane_fade(&self, view: &View, distance: f64) -> f64 {
        let depth = distance - view.along;
        if depth <= distance * 1e-3 {
            return 0.0;
        }
        self.fade(distance / depth)
    }

    /// How much of `view`'s frame shows the fill left where a depth plane
    /// was lifted, 0 to 1. Behind a plane the background keeps only the
    /// light filled in from about it, which is smooth: where the plane took
    /// the light the image had, or restored the light its dark cloud hid.
    /// At each of a grid of points, the share a plane took where the
    /// background shows counts, less what is left of the plane times the
    /// share it took where the plane shows: a plane in its own place hides
    /// its fill, and one slid aside, outgrown by the background or flown
    /// through uncovers it.
    pub fn fill_shown(&self, scene: &Scene, view: &View) -> f64 {
        const COLUMNS: usize = 64;
        if scene.planes.is_empty() {
            return 0.0;
        }
        let rows = ((COLUMNS * self.height) as f64 / self.width.max(1) as f64)
            .round()
            .max(1.0) as usize;
        let (step_u, step_v) = (
            self.width as f64 / COLUMNS as f64,
            self.height as f64 / rows as f64,
        );
        let distance = scene.background_distance_pc;
        // A grid cell's span on the background, image pixels: light is
        // read at the level of detail that wide.
        let (x0, y0) = view.unproject(0.0, 0.0, distance);
        let (x1, y1) = view.unproject(step_u, 0.0, distance);
        let footprint = (x1 - x0).hypot(y1 - y0) as f32;
        let light = |(level, scale): (&LightImage, f32), x: f64, y: f64| -> f32 {
            let pixel = Pyramid::sample_level(level, scale, x as f32, y as f32);
            pixel[0] + pixel[1] + pixel[2]
        };
        let background = scene.background.level_for(footprint);
        let planes: Vec<_> = scene
            .planes
            .iter()
            .map(|plane| {
                (
                    plane,
                    plane.image.level_for(footprint),
                    self.plane_fade(view, plane.distance_pc),
                )
            })
            .collect();
        // The share of the light photographed at image point `(x, y)` that
        // `plane` took, or that its dark cloud hid.
        let taken = |&(plane, level, _): &(&Plane, (&LightImage, f32), f64), x: f64, y: f64| {
            let lifted = if plane.lit {
                let own = light(level, x - plane.left as f64, y - plane.top as f64);
                if own > 0.0 {
                    own / (own + light(background, x, y).max(0.0))
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let hidden = plane.transmission.as_ref().map_or(0.0, |cloud| {
                let passed = cloud.at(x, y);
                1.0 - (passed[0] + passed[1] + passed[2]) / 3.0
            });
            lifted.max(hidden) as f64
        };
        let mut total = 0.0;
        for row in 0..rows {
            let v = (row as f64 + 0.5) * step_v;
            for column in 0..COLUMNS {
                let u = (column as f64 + 0.5) * step_u;
                let (bx, by) = view.unproject(u, v, distance);
                let shown: f64 = planes
                    .iter()
                    .map(|plane| {
                        let under = taken(plane, bx, by);
                        let fade = plane.2;
                        if under <= 0.0 {
                            return 0.0;
                        }
                        let over = if fade > 0.0 {
                            let (px, py) = view.unproject(u, v, plane.0.distance_pc);
                            fade * taken(plane, px, py)
                        } else {
                            0.0
                        };
                        (under - over).max(0.0)
                    })
                    .sum();
                total += shown.min(1.0);
            }
        }
        total / (COLUMNS * rows) as f64
    }

    /// How much of each frame shows the fill behind the depth planes, as
    /// [`Self::fill_shown`] finds it.
    pub fn fill_by_frame(&self, scene: &Scene) -> Vec<f64> {
        (0..self.frames)
            .into_par_iter()
            .map(|frame| self.fill_shown(scene, &self.view(scene, frame)))
            .collect()
    }

    /// How much of a layer that has come `growth` times nearer is left.
    fn fade(&self, growth: f64) -> f64 {
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
        let absorbers = self.absorbers(scene, &view);
        // The dark clouds in front of a layer at `distance`: each dims a
        // layer lifted after it, or the background, by the light it lets
        // through, as the light behind it was restored; and the stars and
        // the layers lifted before it, photographed through it, by the
        // change in that light.
        let shading = |distance: f64, restored: &dyn Fn(usize) -> bool| -> Vec<Shade> {
            absorbers
                .iter()
                .filter(|absorber| absorber.distance < distance)
                .map(|&absorber| Shade {
                    absorber,
                    restored: restored(absorber.order),
                })
                .collect()
        };
        let whole = |image, distance| Layer {
            image,
            at: (0, 0),
            distance,
            gain: 1.0,
        };
        draw_plane(
            &mut out,
            whole(&scene.background, scene.background_distance_pc),
            &view,
            (None, &shading(scene.background_distance_pc, &|_| true)),
            blend,
        );
        // What lies beyond the background lies behind its dust.
        let behind = |distance: f64| {
            scene
                .dust
                .as_ref()
                .filter(|_| distance > scene.background_distance_pc)
                .map(|dust| (&**dust, scene.background_distance_pc))
        };
        draw_plane(
            &mut out,
            whole(&scene.leftover, scene.leftover_distance_pc),
            &view,
            (
                behind(scene.leftover_distance_pc),
                &shading(scene.leftover_distance_pc, &|_| false),
            ),
            blend,
        );
        for (order, plane) in scene.planes.iter().enumerate() {
            let gain = self.plane_fade(&view, plane.distance_pc) as f32;
            if gain <= 0.0 || !plane.lit {
                continue;
            }
            draw_plane(
                &mut out,
                Layer {
                    image: &plane.image,
                    at: (plane.left, plane.top),
                    distance: plane.distance_pc,
                    gain,
                },
                &view,
                (
                    behind(plane.distance_pc),
                    &shading(plane.distance_pc, &|before| before < order),
                ),
                blend,
            );
        }
        draw_sprites(&mut out, scene, &view, self, &absorbers);
        out
    }

    /// The planes of `scene` that block light, as `view` sees them.
    fn absorbers<'a>(&self, scene: &'a Scene, view: &View) -> Vec<Absorber<'a>> {
        scene
            .planes
            .iter()
            .enumerate()
            .filter_map(|(order, plane)| {
                let transmission = plane.transmission.as_ref()?;
                // Once the camera has flown through a cloud, it blocks
                // nothing in front of it.
                let fade = self.plane_fade(view, plane.distance_pc) as f32;
                Some(Absorber {
                    map: view.plane_map(plane.distance_pc).filter(|_| fade > 0.0),
                    transmission,
                    distance: plane.distance_pc,
                    order,
                    fade,
                })
            })
            .collect()
    }
}

/// A plane that blocks light, as a frame sees it: which of its points each
/// output pixel shows, if it lies ahead, the light it lets through, its
/// distance, its place in the order the planes were lifted, and how much
/// of it is left as the camera passes through it.
#[derive(Clone, Copy)]
struct Absorber<'a> {
    map: Option<PlaneMap>,
    transmission: &'a Transmission,
    distance: f64,
    order: usize,
    fade: f32,
}

impl Absorber<'_> {
    /// The share of light it lets through at output pixel `(column, row)`.
    #[inline]
    fn now(&self, column: f64, row: f64) -> [f32; 3] {
        let Some(map) = self.map else {
            return [1.0; 3];
        };
        let (x, y) = map.at(column, row);
        self.transmission
            .at(x, y)
            .map(|passed| 1.0 - self.fade * (1.0 - passed))
    }

    /// How light photographed through it at image point `seen` changes as
    /// it shows through it at output pixel `(column, row)`, each channel.
    #[inline]
    fn change(&self, seen: (f64, f64), (column, row): (f64, f64)) -> [f32; 3] {
        let now = self.now(column, row);
        let then = self.transmission.at(seen.0, seen.1);
        std::array::from_fn(|channel| (now[channel] / then[channel]).min(MAX_BRIGHTENING))
    }
}

/// A dark cloud in front of a layer, and whether the light behind it was
/// restored as the layer was lifted, so that the cloud dims it by the light
/// it lets through rather than by the change in it.
#[derive(Clone, Copy)]
struct Shade<'a> {
    absorber: Absorber<'a>,
    restored: bool,
}

/// A depth whose layer's image edges must stay out of view: which of them,
/// left, top, right and bottom, and whether the layer fades as the camera
/// passes through it, and so needs no guard once gone.
#[derive(Clone, Copy, Debug)]
struct Guard {
    distance: f64,
    edges: [bool; 4],
    fades: bool,
}

impl Guard {
    /// A layer at `distance` that spans the image and never fades.
    fn whole(distance: f64) -> Self {
        Self {
            distance,
            edges: [true; 4],
            fades: false,
        }
    }
}

/// A plane to draw: its image, the image pixel its top-left pixel lies on,
/// its distance and the gain on its light.
#[derive(Clone, Copy)]
struct Layer<'a> {
    image: &'a Pyramid,
    at: (usize, usize),
    distance: f64,
    gain: f32,
}

impl Layer<'_> {
    /// The output columns and rows, each from and to below, that the layer
    /// can reach in `view` on a `width` × `height` frame: the box about its
    /// projected corners, a pixel wider for the sampling.
    fn reach(&self, view: &View, width: usize, height: usize) -> ((usize, usize), (usize, usize)) {
        let base = self.image.base();
        let (left, top) = (self.at.0 as f64 - 1.5, self.at.1 as f64 - 1.5);
        let (right, bottom) = (
            left + base.width as f64 + 2.0,
            top + base.height as f64 + 2.0,
        );
        let (mut low, mut high) = (
            (f64::INFINITY, f64::INFINITY),
            (f64::NEG_INFINITY, f64::NEG_INFINITY),
        );
        for (x, y) in [(left, top), (right, top), (left, bottom), (right, bottom)] {
            let Some((u, v, _)) = view.project(x, y, self.distance) else {
                return ((0, 0), (0, 0));
            };
            low = (low.0.min(u), low.1.min(v));
            high = (high.0.max(u), high.1.max(v));
        }
        let span = |low: f64, high: f64, size: usize| {
            let first = low.floor().clamp(0.0, size as f64) as usize;
            let last = (high.ceil() + 1.0).clamp(0.0, size as f64) as usize;
            (first, last.max(first))
        };
        (span(low.0, high.0, width), span(low.1, high.1, height))
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

/// The stops of a tour reaching and leaving its stops at `times` that a
/// frame `now` seconds in belongs to: the one it is held at, twice, or the
/// nearer of the two it travels between and then the other.
fn stops_at(times: &[(f64, f64)], now: f64) -> (usize, usize) {
    let next = times
        .iter()
        .position(|&(arrive, _)| arrive >= now)
        .unwrap_or(times.len() - 1);
    if next == 0 || now >= times[next].0 {
        (next, next)
    } else if now - times[next - 1].1 < times[next].0 - now {
        (next - 1, next)
    } else {
        (next, next - 1)
    }
}

/// A looped tour's first and last stops are one view: either marked marks
/// both.
fn joined(marked: &mut [bool]) {
    let last = marked.len() - 1;
    let either = marked[0] || marked[last];
    (marked[0], marked[last]) = (either, either);
}

/// `stop` flown `share` of its way in, its lens zoomed so the plane it
/// frames keeps its size, and the camera kept as near that plane's line of
/// sight from Earth as it was.
fn flattened(stop: &Stop, share: f64) -> Stop {
    let dolly = stop.dolly * share;
    // A plane at `depth` grows as `depth / (depth − dolly)`.
    let growth = |dolly: f64| stop.depth / (stop.depth - dolly);
    let on_plane = (stop.depth - 1.0).abs() > 1e-9;
    Stop {
        dolly,
        zoom: stop.zoom * growth(stop.dolly) / growth(dolly),
        // Turning `1 − dolly` of the way puts the camera on the line of
        // sight through a stop on a plane; flying less far, it turns more.
        pan: if on_plane {
            (stop.pan + stop.dolly - dolly).clamp(0.0, 1.0)
        } else {
            stop.pan
        },
        ..*stop
    }
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

/// The numbers a tour's camera blends between stops.
const STATE: usize = 6;
type TourState = [f64; STATE];

/// A stop as numbers that blend smoothly: its focus, the logarithms of the
/// background's nearness and of the lens's zoom (so a flight in or a zoom
/// keeps an even pace), its turn and its pan.
fn stop_state(stop: &Stop) -> TourState {
    [
        stop.focus.0,
        stop.focus.1,
        (1.0 - stop.dolly.clamp(0.0, 0.99)).ln(),
        stop.zoom.max(1e-6).ln(),
        stop.rotation,
        stop.pan.clamp(0.0, 1.0),
    ]
}

/// Along one axis, the point of the background plane a camera must aim at
/// for image point `target`, on a plane at `depth` (as a share of the
/// background's first distance), to lie at the frame's centre, when the
/// camera has `near` of that distance left to the background and turns
/// `pan` of its way from the opening view's angle on `opening`. `centre` is
/// the image's centre. A camera that has not moved sideways sees every
/// depth alike; one that has sees nearer planes shifted further. A plane
/// the camera has all but reached is aimed at as the background.
pub(crate) fn aim_through(
    target: f64,
    depth: f64,
    (near, pan): (f64, f64),
    (opening, centre): (f64, f64),
) -> f64 {
    let ahead = depth - (1.0 - near);
    if (depth - 1.0).abs() < 1e-9 || ahead <= 0.05 * depth {
        return target;
    }
    // Where the opening angle meets the background, and the share of the
    // way the camera moves rather than turns: the background point `aim`
    // is centred from `across = (1 − pan) · (aim − start)` to the side,
    // looking `opening − centre + pan · (aim − start) / near` aside.
    let start = centre + (opening - centre) * near;
    start
        + ((target - centre) * depth - (opening - centre) * ahead)
            / ((1.0 - pan) + pan * ahead / near)
}

/// The camera's state `t` of the way through `tour`. It moves on through
/// a stop it holds at at `glide` of its pace between stops, centred on the
/// stop, plus any push and spin the stop asks for; it passes the others at
/// full pace. Between them it follows a smooth curve whose speed meets
/// those at either end, so it never jerks. The last stop's hold is a final
/// drift that slows to rest, unless the tour is `looped`: then the last
/// stop is the first again, and the camera drifts through the join at one
/// pace.
fn tour_state(tour: &[Stop], glide: f64, looped: bool, t: f64) -> TourState {
    let times = tour_times(tour);
    let total = times.last().map_or(0.0, |&(_, leave)| leave);
    let now = t.clamp(0.0, 1.0) * total;
    let mut state = tour_path(tour, &times, glide, looped, now);
    state[4] += tour_spin(tour, &times, now);
    state
}

/// The turn the stops' spins have added by `now` seconds into `tour`. Each
/// spin runs from up to a second before the camera reaches its stop to up
/// to a second after it leaves (at most half the way either side), so it
/// gathers pace as the camera slows on arrival and loses it as the camera
/// gathers pace again.
fn tour_spin(tour: &[Stop], times: &[(f64, f64)], now: f64) -> f64 {
    let last = tour.len() - 1;
    let mut turn = 0.0;
    for (index, stop) in tour.iter().enumerate() {
        if stop.spin == 0.0 || stop.hold <= 0.0 {
            continue;
        }
        let (arrive, depart) = times[index];
        let before = if index == 0 {
            0.0
        } else {
            ((arrive - times[index - 1].1) / 2.0).min(1.0)
        };
        let after = if index == last {
            0.0
        } else {
            ((times[index + 1].0 - depart) / 2.0).min(1.0)
        };
        let from = arrive - before;
        turn += stop.spin * spin_share(now - from, depart + after - from);
    }
    turn
}

/// The share of the glide's pace a looped tour drifts through its join at.
const JOIN_DRIFT: f64 = 0.5;

/// The least the lens lengthens through a looped tour's join, per second
/// as a logarithm: 2% a second. The stops either side can leave the join
/// next to nothing to drift by (on a whole view, a wide image in a tall
/// frame moved 0.1 pixels a frame), and the view must never stand still.
const JOIN_ZOOM: f64 = 0.02;

/// The camera's state `now` seconds into `tour`, but for its spins.
fn tour_path(tour: &[Stop], times: &[(f64, f64)], glide: f64, looped: bool, now: f64) -> TourState {
    let last = tour.len() - 1;
    let total = times[last].1;
    let states: Vec<TourState> = tour.iter().map(stop_state).collect();
    // A looped tour joins its last stop to its first, which shows the same
    // view turned by whole turns (less the spins, which tour_spin adds).
    let join = looped && last >= 1;
    // The pace through each stop, per second: Catmull-Rom's, from the
    // stops either side (or one side at the ends, or across the join).
    let through = |index: usize| -> TourState {
        // Stops with no time between them (a zero travel, a cut) give no
        // pace rather than an endless one.
        if join && (index == 0 || index == last) {
            let span = total - times[last - 1].1 + times[1].0;
            if span < 1e-6 {
                return [0.0; STATE];
            }
            return std::array::from_fn(|k| {
                let before = states[last - 1][k] - (states[last][k] - states[0][k]);
                (states[1][k] - before) / span
            });
        }
        let (before, after) = (index.saturating_sub(1), (index + 1).min(last));
        let span = times[after].0 - times[before].1;
        if span < 1e-6 {
            return [0.0; STATE];
        }
        std::array::from_fn(|k| (states[after][k] - states[before][k]) / span)
    };
    // No turning or panning on the way through a view near the whole
    // image, which has no room for it.
    let roomy = |index: usize, mut rate: TourState| {
        let room = ((1.0 - states[index][2].exp()) / 0.5).clamp(0.0, 1.0);
        rate[4] *= room;
        rate[5] *= room;
        rate
    };
    // The drift through the join, slower than a glide, and never further
    // back than the whole image: the camera's distance only holds there.
    let join_drift = join.then(|| {
        let mut drift = roomy(
            0,
            through(0).map(|value| value * glide.max(0.0) * JOIN_DRIFT),
        );
        let longest = tour[0].hold.max(tour[last].hold).max(1e-9);
        let room = (-states[0][2]).max(0.0) / longest;
        drift[2] = drift[2].clamp(-room, room);
        // Zooming in, never out: the fit zooms the join's stops in to give
        // the frames before it room.
        drift[3] = drift[3].max(JOIN_ZOOM);
        drift
    });
    // Where the camera is as it reaches and leaves each stop, its pace
    // there, and its drift and push while it holds.
    let mut reach = Vec::with_capacity(tour.len());
    let mut leave = Vec::with_capacity(tour.len());
    let mut pace = Vec::with_capacity(tour.len());
    let mut drifts = Vec::with_capacity(tour.len());
    let mut pushes = Vec::with_capacity(tour.len());
    for (index, stop) in tour.iter().enumerate() {
        let hold = stop.hold.max(0.0);
        let end_stop = index == 0 || index == last;
        let joined = join_drift.filter(|_| end_stop);
        if hold <= 0.0 {
            let rate = match joined {
                Some(drift) => drift,
                None if end_stop => [0.0; STATE],
                None => through(index),
            };
            reach.push(states[index]);
            leave.push(states[index]);
            pace.push(rate);
            drifts.push([0.0; STATE]);
            pushes.push(0.0);
            continue;
        }
        // The slow glide on through the stop, centred on it (but the first
        // starts there and the last ends there), and the push in from the
        // stop's own nearness.
        let mut drift = match joined {
            Some(drift) => drift,
            None => roomy(index, through(index).map(|value| value * glide.max(0.0))),
        };
        // A spin turns about the stop itself, its motion the spin and any
        // push (tour_spin adds the spin).
        if stop.spin != 0.0 && joined.is_none() {
            drift = [0.0; STATE];
        }
        // The push in, eased in and out over the hold as a spin is, so the
        // camera neither backs off before a short hold nor ends a final
        // drift still flying.
        let push = if joined.is_some() {
            0.0
        } else {
            (1.0 - stop.push.clamp(0.0, 0.95)).ln()
        };
        // An unlooped tour's last hold slows to rest, covering half the way
        // a steady drift would.
        let covered = if index == last && !join {
            hold / 2.0
        } else {
            hold
        };
        let before = if index == 0 {
            0.0
        } else if index == last {
            1.0
        } else {
            0.5
        };
        let mut start: TourState =
            std::array::from_fn(|k| states[index][k] - drift[k] * covered * before);
        let mut end: TourState = std::array::from_fn(|k| start[k] + drift[k] * covered);
        end[2] += push;
        // A stop on a depth plane keeps its own share of the turn through
        // its hold, as the hold below draws it, so the camera reaches and
        // leaves it where the hold begins and ends. The drift alone would
        // leave from another share, and close to the plane a little more
        // or less turn swings the view across the frame.
        if on_plane(stop) {
            let share = pan_share(&states[index]);
            start[5] = (share * start[2].exp()).clamp(0.0, 1.0);
            end[5] = (share * end[2].exp()).clamp(0.0, 1.0);
        }
        reach.push(start);
        leave.push(end);
        pace.push(drift);
        drifts.push(drift);
        pushes.push(push);
    }
    for index in 0..=last {
        let (arrive, depart) = times[index];
        if now <= depart || index == last {
            if now >= arrive || index == 0 {
                // Moving on through the stop at its slow pace, slowing to
                // rest through an unlooped tour's last.
                let held = (now - arrive).clamp(0.0, depart - arrive);
                let hold = depart - arrive;
                let moved = if index == last && !join && hold > 0.0 {
                    held - held * held / (2.0 * hold)
                } else {
                    held
                };
                let mut state: TourState =
                    std::array::from_fn(|k| reach[index][k] + drifts[index][k] * moved);
                if hold > 0.0 {
                    state[2] += pushes[index] * spin_share(held, hold);
                }
                // A stop on a depth plane turns in step with its push, so
                // the camera keeps to its line of sight.
                if on_plane(&tour[index]) {
                    state[5] = (pan_share(&states[index]) * state[2].exp()).clamp(0.0, 1.0);
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
            let (before, after) = (leave[index - 1][2], reach[index][2]);
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
            // The turn and pan blend late, but their speeds at either end
            // stay the stops' own, so they never jerk.
            let (start, end) = (pace[index - 1], pace[index]);
            let (_, h10, _, h11) = hermite(s);
            let mut state: TourState = std::array::from_fn(|k| {
                let (h00, _, h01, _) = hermite(if k >= 4 { late } else { s });
                h00 * leave[index - 1][k]
                    + h10 * span * start[k]
                    + h01 * reach[index][k]
                    + h11 * span * end[k]
            });
            // To or from a stop on a depth plane, the pan keeps in step with
            // the camera's nearness: a stop there turns as much of its way
            // as it has left, to look along the line of sight through its
            // target, and a pan blended on its own would turn the camera
            // past it as it nears, sweeping the far star field off the
            // image.
            if on_plane(&tour[index - 1]) || on_plane(&tour[index]) {
                let (h00, _, h01, _) = hermite(s);
                let share = h00 * pan_share(&leave[index - 1]) + h01 * pan_share(&reach[index]);
                state[5] = (share * state[2].exp()).clamp(0.0, 1.0);
            }
            return state;
        }
    }
    leave[last]
}

/// Whether `stop`'s focus point lies on a depth plane.
fn on_plane(stop: &Stop) -> bool {
    (stop.depth - 1.0).abs() > 1e-9
}

/// A camera state's pan as a share of the way it has left to the nebula.
fn pan_share(state: &TourState) -> f64 {
    state[5] / state[2].exp()
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

/// Draw `layer`, and if `dust` gives the dust and the distance of the
/// plane it lies on, in front, dim it by the change in the dust it shows
/// through. With `blend`, sample the levels of detail either side of the
/// view and blend them.
fn draw_plane(
    out: &mut LightImage,
    layer: Layer,
    view: &View,
    (dust, shades): (Option<(&Dust, f64)>, &[Shade]),
    blend: bool,
) {
    let Some(map) = view.plane_map(layer.distance) else {
        return;
    };
    let dust = dust.and_then(|(dust, at)| view.plane_map(at).map(|front| (dust, front)));
    let image = layer.image;
    let footprint = map.across.0.hypot(map.across.1) as f32;
    let (finer, coarser, toward) = image.levels_between(footprint);
    let levels = if blend && toward > 0.0 {
        (finer, Some((coarser, toward)))
    } else {
        (image.level_for(footprint), None)
    };
    let (columns, rows) = layer.reach(view, out.width, out.height);
    if columns.0 >= columns.1 || rows.0 >= rows.1 {
        return;
    }
    let plan = RowPlan {
        map,
        at: (layer.at.0 as f64, layer.at.1 as f64),
        columns,
        gain: layer.gain,
    };
    let width = out.width;
    out.pixels
        .par_chunks_mut(width)
        .enumerate()
        .skip(rows.0)
        .take(rows.1 - rows.0)
        .for_each(|(row, pixels)| {
            plane_row(pixels, row, plan, levels, (dust, shades));
        });
}

/// A level of detail and its scale relative to the base.
type Level<'a> = (&'a LightImage, f32);

/// How [`plane_row`] draws a plane: which point of it each output pixel
/// shows, the image pixel its image's top-left pixel lies on, the output
/// columns it can reach, from and to below, and the gain on its light.
#[derive(Clone, Copy)]
struct RowPlan {
    map: PlaneMap,
    at: (f64, f64),
    columns: (usize, usize),
    gain: f32,
}

/// One row of [`draw_plane`]: output row `row` shows the plane as `plan`
/// has it, sampled from a level of detail, blended `toward` a coarser one
/// if given, and the dust as its map has it. Built for wider vector units
/// too, and picked at run time; it must do the work itself, as a closure
/// it handed on would be built without them.
#[multiversion::multiversion(targets("x86_64+avx2+fma", "x86_64+sse4.1"))]
fn plane_row(
    pixels: &mut [[f32; 3]],
    row: usize,
    plan: RowPlan,
    ((level, level_scale), coarser): (Level, Option<(Level, f32)>),
    (dust, shades): (Option<(&Dust, PlaneMap)>, &[Shade]),
) {
    let row = row as f64;
    let (first, last) = plan.columns;
    for (column, pixel) in (first..last).zip(&mut pixels[first..last]) {
        let (x, y) = plan.map.at(column as f64, row);
        let (px, py) = ((x - plan.at.0) as f32, (y - plan.at.1) as f32);
        let mut light = Pyramid::sample_level(level, level_scale, px, py);
        if let Some(((coarse, coarse_scale), toward)) = coarser {
            let other = Pyramid::sample_level(coarse, coarse_scale, px, py);
            for channel in 0..3 {
                light[channel] += (other[channel] - light[channel]) * toward;
            }
        }
        let mut gain = [plan.gain; 3];
        if let Some((dust, front)) = dust {
            let change = dust.change((x, y), front.at(column as f64, row));
            gain = gain.map(|gain| gain * change);
        }
        for shade in shades {
            let factor = if shade.restored {
                shade.absorber.now(column as f64, row)
            } else {
                shade.absorber.change((x, y), (column as f64, row))
            };
            for channel in 0..3 {
                gain[channel] *= factor[channel];
            }
        }
        for channel in 0..3 {
            pixel[channel] += light[channel] * gain[channel];
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

fn draw_sprites(
    out: &mut LightImage,
    scene: &Scene,
    view: &View,
    shot: &Shot,
    absorbers: &[Absorber],
) {
    // Every layer's scale with the lens as it is now and the camera not yet
    // moved: a star's growth is how much nearer it has come.
    let lens_scale = view.focal_out / view.focal_px;
    let (width, height) = (out.width as f64, out.height as f64);
    let placed: Vec<Placed> = scene
        .sprites
        .iter()
        .filter_map(|sprite| {
            let distance = scene.flies_at(sprite);
            let (x, y, scale) = view.project(sprite.x, sprite.y, distance)?;
            let growth = view.growth(scale);
            let fade = shot.star_fade(view, scale);
            if fade <= 0.0 {
                return None;
            }
            let grown = growth.clamp(1.0, shot.growth_limit);
            let size = lens_scale * grown.sqrt();
            // Behind the dust, the change in the dust it shows through.
            let mut dimming = match &scene.dust {
                Some(dust) if distance > scene.background_distance_pc => {
                    let now = view.unproject(x, y, scene.background_distance_pc);
                    dust.change((sprite.x, sprite.y), now)
                }
                _ => 1.0,
            };
            // And the change in each dark cloud in front of it, as the star
            // was photographed through it.
            for absorber in absorbers
                .iter()
                .filter(|absorber| absorber.distance < distance)
            {
                let change = absorber.change((sprite.x, sprite.y), (x, y));
                dimming *= (change[0] + change[1] + change[2]) / 3.0;
            }
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
    use std::sync::Arc;

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

    /// A scene without stars whose starless image, a gradient with a soft
    /// cloud at `(150, 150)`, has the cloud lifted onto a plane at
    /// `distance` pc, or left on the background if `None`.
    fn clouded(distance: Option<f64>) -> Scene {
        let (width, height) = (400, 300);
        let mut starless = LightImage::new(width, height);
        for (index, pixel) in starless.pixels.iter_mut().enumerate() {
            let (x, y) = (index % width, index / width);
            let r2 = (x as f64 - 150.0).powi(2) + (y as f64 - 150.0).powi(2);
            let cloud = (0.8 * (-r2 / 200.0).exp()) as f32;
            *pixel = [
                x as f32 / width as f32 + cloud,
                y as f32 / height as f32 + cloud,
                0.2 + cloud,
            ];
        }
        let planes = distance.map(|distance| {
            let mask = crate::plane::tests::disc(width, height, (150.0, 150.0), 45.0);
            let region = crate::Region::from_mask(&mask, width, height)
                .unwrap()
                .unwrap();
            crate::lift_region(&mut starless, &region, distance, "cloud", true, &[]).unwrap()
        });
        let scene = Scene::new(
            &starless,
            &LightImage::new(width, height),
            &[],
            400.0,
            400.0,
            2000.0,
            &CutOptions::default(),
        );
        scene.with_planes(planes.into_iter().collect(), 0.02)
    }

    #[test]
    fn a_plane_keeps_the_first_frame_and_moves_at_its_own_depth() {
        let (flat, near) = (clouded(None), clouded(Some(100.0)));
        // Full size, so both draw from their finest level.
        let shot = Shot {
            focus: (199.5, 149.5),
            dolly: 0.0,
            truck: (0.002, 0.0),
            width: 400,
            height: 300,
            frames: 3,
            easing: Easing::Linear,
            ..Shot::default()
        };
        let (a, b) = (shot.render(&flat, 0), shot.render(&near, 0));
        for (a, b) in a.pixels.iter().zip(&b.pixels) {
            for channel in 0..3 {
                assert!((a[channel] - b[channel]).abs() < 1e-4, "{a:?} {b:?}");
            }
        }
        // At the truck's widest swing the cloud is where its own depth puts
        // it, not where the background's would.
        let view = shot.view(&near, 1);
        let at_plane = view.project(150.0, 150.0, 100.0).unwrap();
        let at_background = view.project(150.0, 150.0, 400.0).unwrap();
        assert!((at_plane.0 - at_background.0).abs() > 8.0);
        let (x, y) = brightest(&shot.render(&near, 1));
        assert!(
            (x as f64 - at_plane.0).abs() <= 1.0 && (y as f64 - at_plane.1).abs() <= 1.0,
            "{x} {y} {at_plane:?}"
        );
    }

    #[test]
    fn light_lifted_in_front_of_a_dark_cloud_keeps_the_first_frame() {
        // An even glow behind a dark cloud at 200 pc, and a glowing
        // nebula at 100 pc over both, larger than the cloud, so lifted
        // after it.
        let (width, height) = (400, 300);
        let mut starless = LightImage::new(width, height);
        for (index, pixel) in starless.pixels.iter_mut().enumerate() {
            let (x, y) = ((index % width) as f64, (index / width) as f64);
            let r = (x - 150.0).hypot(y - 150.0);
            let t = ((r - 4.0) / 16.0).clamp(0.0, 1.0);
            let passed = (0.05 + 0.95 * t * t * (3.0 - 2.0 * t)) as f32;
            let glow = (0.3 * (-(r / 35.0).powi(2)).exp()) as f32;
            *pixel = [0.3 * passed + glow, 0.3 * passed + glow, 0.4 * passed];
        }
        let scene = |starless: &LightImage, planes: Vec<crate::Plane>| {
            Scene::new(
                starless,
                &LightImage::new(width, height),
                &[],
                400.0,
                400.0,
                2000.0,
                &CutOptions::default(),
            )
            .with_planes(planes, 0.02)
        };
        let region = |radius: f64| {
            let mask = crate::plane::tests::disc(width, height, (150.0, 150.0), radius);
            crate::Region::from_mask(&mask, width, height)
                .unwrap()
                .unwrap()
        };
        let mut lifted = starless.clone();
        let cloud =
            crate::lift_region(&mut lifted, &region(30.0), 200.0, "cloud", true, &[]).unwrap();
        assert!(cloud.transmission.is_some());
        let earlier = [cloud];
        let nebula =
            crate::lift_region(&mut lifted, &region(70.0), 100.0, "nebula", true, &earlier)
                .unwrap();
        assert!(nebula.lit);
        let [cloud] = earlier;
        let shot = Shot {
            focus: (199.5, 149.5),
            dolly: 0.0,
            width: 400,
            height: 300,
            frames: 2,
            ..Shot::default()
        };
        let (a, b) = (
            shot.render(&scene(&starless, Vec::new()), 0)
                .to_display_rgb8(),
            shot.render(&scene(&lifted, vec![cloud, nebula]), 0)
                .to_display_rgb8(),
        );
        for (a, b) in a.pixels().zip(b.pixels()) {
            for channel in 0..3 {
                assert!(a.0[channel].abs_diff(b.0[channel]) <= 1, "{a:?} {b:?}");
            }
        }
    }

    #[test]
    fn a_dark_cloud_slides_across_what_it_hides_at_its_own_depth() {
        // An even glow, and a dark cloud in front of it at a quarter of its
        // distance, letting through 2% at its core and thinning out to
        // nothing 20 pixels out.
        let (width, height) = (400, 300);
        let mut starless = LightImage::new(width, height);
        for (index, pixel) in starless.pixels.iter_mut().enumerate() {
            let (x, y) = ((index % width) as f64, (index / width) as f64);
            let t = (((x - 150.0).hypot(y - 150.0) - 4.0) / 16.0).clamp(0.0, 1.0);
            let passed = (0.02 + 0.98 * t * t * (3.0 - 2.0 * t)) as f32;
            *pixel = [0.3 * passed, 0.3 * passed, 0.4 * passed];
        }
        let photo = Scene::new(
            &starless,
            &LightImage::new(width, height),
            &[],
            400.0,
            400.0,
            2000.0,
            &CutOptions::default(),
        );
        let mask = crate::plane::tests::disc(width, height, (150.0, 150.0), 30.0);
        let region = crate::Region::from_mask(&mask, width, height)
            .unwrap()
            .unwrap();
        let mut lifted = starless.clone();
        let cloud = crate::lift_region(&mut lifted, &region, 100.0, "cloud", true, &[]).unwrap();
        assert!(cloud.transmission.is_some() && !cloud.lit);
        // A star behind the cloud, seen through its black core.
        let behind = Sprite::new(148, 148, LightImage::new(5, 5), 150.0, 150.0, 1000.0);
        let mut scene = Scene::new(
            &lifted,
            &LightImage::new(width, height),
            &[],
            400.0,
            400.0,
            2000.0,
            &CutOptions::default(),
        )
        .with_planes(vec![cloud], 0.02);
        Arc::make_mut(&mut scene.sprites).push(behind);
        assert_eq!(scene.flies_at(&scene.sprites[0]), 90.0);
        let shot = Shot {
            focus: (199.5, 149.5),
            dolly: 0.0,
            truck: (0.002, 0.0),
            width: 400,
            height: 300,
            frames: 3,
            easing: Easing::Linear,
            ..Shot::default()
        };
        // The first frame is the image. It samples it between pixels'
        // centres, where the light behind and the cloud, sampled apart,
        // multiply to a little more or less than the image's own light at
        // the cloud's hard edges: at most a step of an 8-bit frame.
        let (a, b) = (
            shot.render(&photo, 0).to_display_rgb8(),
            shot.render(&scene, 0).to_display_rgb8(),
        );
        for (a, b) in a.pixels().zip(b.pixels()) {
            for channel in 0..3 {
                assert!(a.0[channel].abs_diff(b.0[channel]) <= 1, "{a:?} {b:?}");
            }
        }
        // At the truck's widest swing the cloud's core is where its own
        // depth puts it, and where it was, the glow shows again.
        let view = shot.view(&scene, 1);
        let frame = shot.render(&scene, 1);
        let at = |(x, y): (f64, f64)| frame.at(x.round() as usize, y.round() as usize)[2];
        let (core, _) = {
            let (x, y, _) = view.project(150.0, 150.0, 100.0).unwrap();
            ((x, y), ())
        };
        let (was, _) = {
            let (x, y, _) = view.project(150.0, 150.0, 400.0).unwrap();
            ((x, y), ())
        };
        assert!((core.0 - was.0).abs() > 12.0, "{core:?} {was:?}");
        assert!(at(core) < 0.01, "{}", at(core));
        // Behind the cloud as it moves off, the glow it hid shows again, as
        // bright as the glow about it.
        let uncovered = (was.0 + 8.0, was.1);
        let first = shot.render(&scene, 0);
        assert!(first.at(154, 150)[2] < 0.05, "{}", first.at(154, 150)[2]);
        assert!(at(uncovered) > 0.35, "{}", at(uncovered));
    }

    #[test]
    fn a_plane_fades_as_the_camera_passes_through_it() {
        let near = clouded(Some(100.0));
        let mut bare = near.clone();
        bare.planes = Arc::new(Vec::new());
        // The camera flies to 96 pc, 4 short of the plane: 25 times nearer.
        let shot = Shot {
            focus: (199.5, 149.5),
            dolly: 0.24,
            truck: (0.0, 0.0),
            width: 200,
            height: 150,
            frames: 3,
            ..Shot::default()
        };
        assert_eq!(shot.plane_fade(&shot.view(&near, 0), 100.0), 1.0);
        let last = shot.view(&near, 2);
        assert_eq!(shot.plane_fade(&last, 100.0), 0.0);
        assert_eq!(shot.render(&near, 2), shot.render(&bare, 2));
        // Past it, it is behind the camera.
        let past = Shot {
            dolly: 0.3,
            ..shot.clone()
        };
        assert_eq!(past.plane_fade(&past.view(&near, 2), 100.0), 0.0);
    }

    /// A 400 × 300 image of a nebula at 400 pc, with a broad glowing cloud
    /// at its centre lifted onto a plane at `distance`.
    fn glowing(distance: f64) -> Scene {
        let (width, height) = (400, 300);
        let mut starless = LightImage::new(width, height);
        for (index, pixel) in starless.pixels.iter_mut().enumerate() {
            let (x, y) = (index % width, index / width);
            let r2 = (x as f64 - 199.5).powi(2) + (y as f64 - 149.5).powi(2);
            let cloud = (2.0 * (-r2 / 3200.0).exp()) as f32;
            *pixel = [0.2 + cloud; 3];
        }
        let mask = crate::plane::tests::disc(width, height, (199.5, 149.5), 110.0);
        let region = crate::Region::from_mask(&mask, width, height)
            .unwrap()
            .unwrap();
        let plane =
            crate::lift_region(&mut starless, &region, distance, "cloud", false, &[]).unwrap();
        Scene::new(
            &starless,
            &LightImage::new(width, height),
            &[],
            400.0,
            400.0,
            2000.0,
            &CutOptions::default(),
        )
        .with_planes(vec![plane], 0.02)
    }

    #[test]
    fn a_plane_outgrown_slid_aside_or_flown_through_uncovers_its_fill() {
        // The last frame of a move flying `dolly` of the way in toward the
        // cloud, its lens magnified `zoom_end` by then.
        let fill = |scene: &Scene, dolly: f64, zoom_end: f64| {
            let shot = Shot {
                focus: (199.5, 149.5),
                dolly,
                truck: (0.0, 0.0),
                zoom_end,
                width: 160,
                height: 120,
                frames: 3,
                ..Shot::default()
            };
            assert_eq!(shot.fill_shown(scene, &shot.view(scene, 0)), 0.0);
            shot.fill_shown(scene, &shot.view(scene, 2))
        };
        let (far, near) = (glowing(1200.0), glowing(200.0));
        // Flying in, the nebula outgrows a cloud far behind it, and the
        // fill where the cloud was lifted rings it.
        let deep = fill(&far, 0.85, 1.0);
        assert!(deep > FILL_SHOWN, "{deep}");
        // Flying less far and zooming the lens to show the cloud as large
        // shows less of it.
        let growth = |dolly: f64| 3.0 / (3.0 - dolly);
        let shallow = fill(&far, 0.3, growth(0.85) / growth(0.3));
        assert!(shallow < deep / 2.0, "{shallow} {deep}");
        // A cloud in front outgrows its own place, and hides its fill.
        assert!(fill(&near, 0.4, 1.0) < 0.01);
        // Swinging aside, at the middle of the move, slides it off its
        // place.
        let shot = Shot {
            focus: (199.5, 149.5),
            dolly: 0.0,
            truck: (0.1, 0.0),
            width: 160,
            height: 120,
            frames: 3,
            ..Shot::default()
        };
        let aside = shot.fill_shown(&near, &shot.view(&near, 1));
        assert!(aside > 0.02, "{aside}");
        // Flown through, it is gone, and all its fill shows.
        let through = fill(&near, 0.6, 1.0);
        assert!(through > 0.3, "{through}");
    }

    #[test]
    fn a_tour_stop_showing_much_fill_flies_less_far_and_keeps_its_size() {
        let scene = glowing(1200.0);
        let tour = |dolly: f64| Shot {
            tour: vec![
                Stop {
                    focus: (199.5, 149.5),
                    hold: 1.0,
                    ..Stop::default()
                },
                // On the cloud's plane, from its line of sight from Earth.
                Stop {
                    focus: (199.5, 149.5),
                    dolly,
                    zoom: 1.5,
                    pan: 1.0 - dolly,
                    depth: 3.0,
                    travel: 2.0,
                    hold: 2.0,
                    ..Stop::default()
                },
            ],
            glide: 0.0,
            width: 160,
            height: 120,
            frames: 60,
            ..Shot::default()
        };
        let most = |shot: &Shot| shot.fill_by_frame(&scene).into_iter().fold(0.0, f64::max);
        let asked = tour(0.85);
        assert!(most(&asked) > FILL_SHOWN);
        let (fitted, _) = asked.fitted(&scene);
        assert!(most(&fitted) <= FILL_SHOWN, "{}", most(&fitted));
        let (stop, wanted) = (fitted.tour[1], asked.tour[1]);
        assert!(stop.dolly < 0.85, "{stop:?}");
        // The cloud shows as large as asked, and the camera stays on its
        // line of sight.
        let size = |stop: &Stop| stop.zoom * stop.depth / (stop.depth - stop.dolly);
        assert!((size(&stop) / size(&wanted) - 1.0).abs() < 1e-9, "{stop:?}");
        assert!((stop.pan - (1.0 - stop.dolly)).abs() < 1e-9, "{stop:?}");
        // A tour that shows little fill keeps its stops.
        let shallow = tour(0.1);
        assert!(most(&shallow) <= FILL_SHOWN, "{}", most(&shallow));
        assert_eq!(shallow.fitted(&scene).0.tour, shallow.tour);
    }

    #[test]
    fn a_plane_cut_by_the_image_edge_keeps_that_edge_out_of_view() {
        // A near cloud running off the image's right edge.
        let (width, height) = (400, 300);
        let mut starless = LightImage::new(width, height);
        for (index, pixel) in starless.pixels.iter_mut().enumerate() {
            let x = (index % width) as f32;
            *pixel = [0.2 + 0.3 * (x / width as f32).powi(4); 3];
        }
        let mask = crate::plane::tests::disc(width, height, (399.0, 150.0), 80.0);
        let region = crate::Region::from_mask(&mask, width, height)
            .unwrap()
            .unwrap();
        let plane = crate::lift_region(&mut starless, &region, 100.0, "cloud", true, &[]).unwrap();
        let bare = Scene::new(
            &starless,
            &LightImage::new(width, height),
            &[],
            400.0,
            450.0,
            2000.0,
            &CutOptions::default(),
        );
        let clouded = bare.clone().with_planes(vec![plane], 0.02);
        let shot = Shot {
            focus: (199.5, 149.5),
            dolly: 0.0,
            truck: (0.05, 0.0),
            zoom: 2.0,
            width: 200,
            height: 150,
            frames: 9,
            ..Shot::default()
        };
        let guards = shot.guarded_depths(&clouded);
        assert_eq!(guards.len(), 4);
        assert_eq!(guards[3].edges, [false, false, true, false]);
        // Moving right, the near cloud's cut edge would slide into view
        // well before the background's.
        let (_, free) = shot.fitted(&bare);
        let (fitted, held) = shot.fitted(&clouded);
        assert!(held < 0.5 * free, "{held} {free}");
        assert!(fitted.stays_inside(&clouded));
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
    fn a_whole_start_toward_an_edge_zooms_in_until_its_last_frame_fits() {
        // At dolly 0.4 the last view is 0.6 of the whole one, centred on a
        // focus 60 pixels from the edge: past it unless zoomed.
        let scene = scene_with(&[]);
        let shot = Shot {
            focus: (60.0, 150.0),
            dolly: 0.4,
            truck: (0.0, 0.0),
            start: Start::Whole,
            width: 200,
            height: 150,
            frames: 20,
            ..Shot::default()
        };
        assert!(!shot.stays_inside(&scene));
        let (fitted, _) = shot.fitted(&scene);
        assert!(fitted.stays_inside(&scene));
        assert!(fitted.zoom > 1.0);
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
            assert!(
                fitted.inside(&scene, &view, &[Guard::whole(4000.0)]),
                "frame {frame}"
            );
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
            without.sprites = std::sync::Arc::new(Vec::new());
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

    #[test]
    fn a_stop_on_a_depth_plane_centres_its_point_there() {
        let scene = scene();
        let shot = Shot {
            tour: vec![
                Stop {
                    focus: (299.5, 199.5),
                    hold: 1.0,
                    ..Stop::default()
                },
                Stop {
                    focus: (100.0, 120.0),
                    dolly: 0.3,
                    pan: 0.2,
                    depth: 0.5,
                    travel: 2.0,
                    hold: 2.0,
                    ..Stop::default()
                },
            ],
            glide: 0.0,
            width: 160,
            height: 90,
            frames: 60,
            ..Shot::default()
        };
        // A second into the second stop's hold, of five seconds in all.
        let view = shot.tour_view(&scene, 0.8);
        let (u, v, _) = view.project(100.0, 120.0, 200.0).unwrap();
        assert!(
            (u - 79.5).abs() < 1e-6 && (v - 44.5).abs() < 1e-6,
            "{u} {v}"
        );
        // The background point behind it lies well off the centre.
        let (u, v, _) = view.project(100.0, 120.0, 400.0).unwrap();
        assert!((u - 79.5).hypot(v - 44.5) > 5.0, "{u} {v}");
        // Turning as much of the way as it has left puts the camera on the
        // line of sight from Earth through the point: at every depth, the
        // background point to aim at is the point itself.
        for depth in [0.5, 3.0] {
            let aim = aim_through(100.0, depth, (0.4, 0.4), (299.5, 299.5));
            assert!((aim - 100.0).abs() < 1e-9, "{depth}: {aim}");
        }
    }

    #[test]
    fn a_stop_on_a_depth_plane_leaves_where_its_hold_ends() {
        // A close stop on a plane far behind the nebula, turning all of the
        // way it has left, between a nearer stop and the whole view; a
        // gliding tour drifts through the hold, backing off and turning
        // less, and must leave from where the drift ends. Arriving at
        // second 5, leaving at 7.25, of 12.25.
        let tour = vec![
            Stop {
                focus: (200.0, 150.0),
                dolly: 0.5,
                pan: 0.5,
                hold: 1.0,
                ..Stop::default()
            },
            Stop {
                focus: (100.0, 300.0),
                dolly: 0.88,
                zoom: 3.0,
                pan: 0.12,
                depth: 3.0,
                travel: 4.0,
                hold: 2.25,
                ..Stop::default()
            },
            Stop {
                focus: (299.5, 199.5),
                travel: 4.0,
                hold: 1.0,
                ..Stop::default()
            },
        ];
        let state = |seconds: f64| tour_state(&tour, 0.2, true, seconds / 12.25);
        for seconds in [5.0, 7.25] {
            let (before, after) = (state(seconds - 1e-6), state(seconds + 1e-6));
            for k in 0..STATE {
                assert!(
                    (after[k] - before[k]).abs() < 1e-4,
                    "at second {seconds}, state {k}: {} to {}",
                    before[k],
                    after[k]
                );
            }
        }
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
            // Coming to rest at each held stop, for exact checks.
            glide: 0.0,
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
    fn a_gliding_tour_never_stops_and_slows_through_its_stops() {
        let scene = scene();
        let shot = Shot {
            glide: 0.2,
            ..tour()
        };
        // Through the middle of the second stop's hold, seconds 4 to 5, it
        // is where the stop asks; the first frame is the opening view and
        // the last the closing one.
        for (frame, stop) in [(0, 0), (45, 1), (100, 3)] {
            let view = shot.view(&scene, frame);
            let stop = shot.tour[stop];
            let (x, y, _) = view.project(stop.focus.0, stop.focus.1, 400.0).unwrap();
            assert!(
                (x - 149.5).abs() < 1e-6 && (y - 99.5).abs() < 1e-6,
                "frame {frame}: ({x}, {y})"
            );
        }
        // Always moving, slowest through the held stops.
        let place = |frame| {
            shot.view(&scene, frame)
                .project(299.5, 199.5, 400.0)
                .unwrap()
        };
        let steps: Vec<f64> = (1..=100)
            .map(|frame| {
                let (p, q) = (place(frame - 1), place(frame));
                (q.0 - p.0).hypot(q.1 - p.1) + (q.2 - p.2).abs()
            })
            .collect();
        assert!(steps.iter().all(|&step| step > 0.0), "{steps:?}");
        let fastest = steps.iter().cloned().fold(0.0, f64::max);
        assert!(steps[44] < 0.35 * fastest, "{} of {fastest}", steps[44]);
        // A push flies on in while holding.
        let mut pushed = tour();
        pushed.tour[1].push = 0.5;
        let (reach, leave) = (pushed.view(&scene, 40), pushed.view(&scene, 50));
        assert!(
            leave.along > reach.along + 1.0,
            "{} {}",
            reach.along,
            leave.along
        );
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
        let turn = |frame: usize| tour_state(&shot.tour, 0.0, false, frame as f64 / 120.0)[4];
        // It starts turning a second before the camera arrives, as it
        // slows, and is done a second after it leaves.
        assert_eq!(turn(10), 0.0);
        assert!(turn(20) > 0.0);
        assert!(
            (turn(60) - std::f64::consts::PI).abs() < 1e-9,
            "{}",
            turn(60)
        );
        assert!(turn(100) < std::f64::consts::TAU);
        assert!((turn(110) - std::f64::consts::TAU).abs() < 1e-9);
        assert!((turn(120) - std::f64::consts::TAU).abs() < 1e-9);
        // Once under way, the view never slows: the spin gathers pace as
        // the camera slows on arrival, and loses it as the camera leaves.
        // (Without the overlap it came to rest at either end of the hold.)
        let motion = |frame: usize| {
            let (a, b) = (shot.view(&scene, frame), shot.view(&scene, frame + 1));
            let (x0, y0, _) = a.project(250.0, 150.0, 900.0).unwrap();
            let (x1, y1, _) = b.project(250.0, 150.0, 900.0).unwrap();
            (x1 - x0).hypot(y1 - y0)
        };
        let under_way = motion(8);
        for frame in 8..=111 {
            assert!(
                motion(frame) > 0.9 * under_way,
                "frame {frame}: {} against {under_way}",
                motion(frame)
            );
        }
        let view = shot.view(&scene, 120);
        assert!(view.turn().abs() < 1e-9);
    }

    #[test]
    fn a_push_flies_on_in_through_the_hold_at_an_even_pace() {
        // Half the remaining way over a four-second hold, from 0.6 of the
        // way to 0.8, with no glide.
        let tour = vec![
            Stop {
                focus: (299.5, 199.5),
                ..Stop::default()
            },
            Stop {
                focus: (299.5, 199.5),
                dolly: 0.6,
                travel: 2.0,
                hold: 4.0,
                push: 0.5,
                ..Stop::default()
            },
            Stop {
                focus: (299.5, 199.5),
                travel: 2.0,
                ..Stop::default()
            },
        ];
        // Eight seconds; the hold runs from second 2 to second 6.
        let left = |seconds: f64| tour_state(&tour, 0.0, false, seconds / 8.0)[2].exp();
        assert!((left(2.0) - 0.4).abs() < 1e-9);
        assert!((left(6.0) - 0.2).abs() < 1e-9);
        // Evenly on a log scale through the middle of the hold, each second
        // taking the same share, easing in and out over the first and last.
        let (a, b) = (left(4.0) / left(3.0), left(5.0) / left(4.0));
        assert!((a - b).abs() < 1e-9 && a < 1.0);
        assert!(left(2.5) / left(2.0) > a);
        // The camera never backs away, even flying in to a short hold.
        let short = vec![
            tour[0],
            Stop {
                hold: 0.5,
                ..tour[1]
            },
        ];
        let mut last = f64::INFINITY;
        for step in 0..=70 {
            let now = tour_state(&short, 0.0, false, step as f64 / 70.0)[2].exp();
            assert!(now <= last + 1e-12, "step {step}: {now} after {last}");
            last = now;
        }
    }

    #[test]
    fn a_looped_tour_drifts_through_its_join_into_its_first_frame() {
        let scene = scene();
        let mut shot = tour();
        shot.glide = 0.2;
        shot.looped = true;
        shot.tour[0].hold = 2.0;
        shot.tour[3].hold = 3.0;
        // 2 + 3 + 1 + 2 + 3 + 3 = 14 seconds, at ten frames a second.
        shot.frames = 140;
        let state = |t: f64| tour_state(&shot.tour, shot.glide, true, t);
        // The frame after the last is the first.
        assert_eq!(shot.progress(shot.frames), 1.0);
        let (first, end) = (state(0.0), state(1.0));
        for k in 0..STATE {
            assert!((first[k] - end[k]).abs() < 1e-9, "{k}: {first:?} {end:?}");
        }
        // The step into the join matches the step out of it: the camera
        // neither stops nor jumps there.
        let step = 1.0 / shot.frames as f64;
        let into: TourState = std::array::from_fn(|k| state(1.0)[k] - state(1.0 - step)[k]);
        let out: TourState = std::array::from_fn(|k| state(step)[k] - state(0.0)[k]);
        assert!(into[0].hypot(into[1]) > 0.1, "{into:?}");
        for k in 0..STATE {
            assert!((into[k] - out[k]).abs() < 1e-9, "{k}: {into:?} {out:?}");
        }
        // Never further back than the whole image, and the lens drifts in
        // at least JOIN_ZOOM a second through the join.
        for frame in 0..shot.frames {
            assert!(state(shot.progress(frame))[2] <= 1e-12);
        }
        assert!(into[3] >= JOIN_ZOOM * 14.0 * step - 1e-12, "{into:?}");
        // Fitted, the join's two stops zoom in together, so its views still
        // meet.
        let (fitted, _) = shot.fitted(&scene);
        let depths = fitted.guarded_depths(&scene);
        assert!((0..fitted.frames).all(|frame| fitted.inside(
            &scene,
            &fitted.view(&scene, frame),
            &depths
        )));
        assert_eq!(fitted.tour[0].zoom, fitted.tour[3].zoom);
        let (a, b) = (fitted.view(&scene, 0), fitted.view(&scene, fitted.frames));
        assert_eq!(
            a.project(100.0, 100.0, 900.0),
            b.project(100.0, 100.0, 900.0)
        );
    }

    #[test]
    fn an_unlooped_tour_slows_to_rest_through_its_last_hold() {
        let mut shot = tour();
        shot.glide = 0.2;
        shot.tour[3].hold = 4.0;
        // 1 + 3 + 1 + 2 + 3 + 4 = 14 seconds; the last hold from second 10.
        let total = 14.0;
        let at = |seconds: f64| tour_state(&shot.tour, shot.glide, false, seconds / total);
        let pace = |seconds: f64| {
            let (a, b) = (at(seconds), at(seconds + 0.01));
            (b[0] - a[0]).hypot(b[1] - a[1]) / 0.01
        };
        // It ends at the last stop, and slows all the way there, to rest.
        let end = at(total);
        assert!((end[0] - 299.5).abs() < 1e-9 && (end[1] - 199.5).abs() < 1e-9);
        assert!(pace(10.0) > 1.0);
        assert!(pace(total - 0.01) < 0.01 * pace(10.0));
        let mut last = f64::INFINITY;
        for step in 0..40 {
            let now = pace(10.0 + step as f64 * 0.1);
            assert!(now <= last + 1e-9);
            last = now;
        }
    }

    #[test]
    fn turning_and_panning_keep_their_pace_through_a_stop_passed_by() {
        // A stop not held between two legs in: the turn arrives and leaves
        // at the same pace.
        let tour = vec![
            Stop {
                focus: (299.5, 199.5),
                ..Stop::default()
            },
            Stop {
                focus: (299.5, 199.5),
                dolly: 0.3,
                rotation: 0.5,
                pan: 0.2,
                travel: 3.0,
                ..Stop::default()
            },
            Stop {
                focus: (299.5, 199.5),
                dolly: 0.6,
                rotation: 1.0,
                pan: 0.4,
                travel: 3.0,
                ..Stop::default()
            },
        ];
        let at = |seconds: f64| tour_state(&tour, 0.0, false, seconds / 6.0);
        for k in [4, 5] {
            let arriving = (at(3.0)[k] - at(3.0 - 1e-5)[k]) / 1e-5;
            let leaving = (at(3.0 + 1e-5)[k] - at(3.0)[k]) / 1e-5;
            assert!(
                (arriving - leaving).abs() < 1e-3 * arriving.abs().max(1e-3),
                "{k}: {arriving} then {leaving}"
            );
        }
    }

    #[test]
    fn a_zero_travel_cuts_without_flinging_the_camera() {
        let tour = vec![
            Stop {
                focus: (299.5, 199.5),
                hold: 2.0,
                ..Stop::default()
            },
            Stop {
                focus: (200.0, 150.0),
                dolly: 0.5,
                travel: 0.0,
                hold: 2.0,
                ..Stop::default()
            },
            Stop {
                focus: (299.5, 199.5),
                travel: 3.0,
                ..Stop::default()
            },
        ];
        for step in 0..=70 {
            let state = tour_state(&tour, 0.2, false, step as f64 / 70.0);
            assert!(
                state
                    .iter()
                    .all(|value| value.is_finite() && value.abs() < 1e3),
                "step {step}: {state:?}"
            );
        }
    }

    #[test]
    fn a_last_stop_with_a_push_still_comes_to_rest() {
        let mut tour = tour().tour;
        tour[3].hold = 4.0;
        tour[3].push = 0.5;
        // 1 + 3 + 1 + 2 + 3 + 4 = 14 seconds.
        let near = |seconds: f64| tour_state(&tour, 0.2, false, seconds / 14.0)[2];
        let ending = (near(14.0) - near(14.0 - 1e-4)) / 1e-4;
        let arriving = (near(10.0 + 1e-4) - near(10.0)) / 1e-4;
        assert!(ending.abs() < 0.01 * arriving.abs(), "{ending} {arriving}");
        assert!((near(14.0) - 0.5_f64.ln()).abs() < 1e-9);
    }

    #[test]
    fn a_two_stop_loop_drifts_through_its_join() {
        // Round the same place once, the last stop the first a turn on.
        let tour = vec![
            Stop {
                focus: (299.5, 199.5),
                dolly: 0.5,
                hold: 2.0,
                ..Stop::default()
            },
            Stop {
                focus: (299.5, 199.5),
                dolly: 0.5,
                rotation: std::f64::consts::TAU,
                travel: 4.0,
                hold: 2.0,
                ..Stop::default()
            },
        ];
        let state = |t: f64| tour_state(&tour, 0.2, true, t);
        let step = 0.001;
        let into = state(1.0)[4] - state(1.0 - step)[4];
        let out = state(step)[4] - state(0.0)[4];
        assert!(into > 0.0 && (into - out).abs() < 1e-6, "{into} {out}");
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
