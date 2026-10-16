//! A whole parallax video from a starless image, its stars and a plate
//! solution: the stars' distances from Gaia and Hipparcos, the scene cut
//! into depths, the dust, lifted galaxies, depth planes and labels, then
//! the frames.
//!
//! [`Parallax::prepare`] does the work that comes once. [`Parallax::frame`]
//! then draws any frame on request, and [`Parallax::render`] hands every
//! frame in turn to a [`FrameSink`]: one of this crate's encoders, or the
//! caller's own, such as a platform video encoder.

use crate::cloud::{Clouds, CountedDistance, counted_distance};
use crate::dust::{Dust, DustMap};
use crate::field::{self, FieldSource};
use crate::lift::Extent;
use crate::overlay::{self, Corner, CustomLabel, Decor, Mark, Overlay, Placed};
use crate::picture::{Artwork, Picture};
use crate::plane::{DepthPlane, Region};
use crate::render::{Easing, Framing, Quality, Shot, Start, Stop};
use crate::scene::{CutOptions, Scene, SmallStars, Star};
use crate::tour::{self, AutoTour, PlannedStop};
use crate::{FrameSink, LightImage, VideoSettings};
use image::{Rgb, Rgb32FImage, RgbImage};
use seiza::Wcs;
use seiza::catalog::{DistanceBasis, ObjectDistance};
use seiza::objects::{GeometryData, SkyObject};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

/// Where lifted galaxies fly: so far beyond the stars that they hold still.
pub const GALAXY_DISTANCE_PC: f64 = 1e8;

/// How much nearer or farther than the nebula, as a share of its distance,
/// a catalogued nebula must lie for [`SceneOptions::auto_planes`] to give
/// it a plane of its own.
pub const AUTO_PLANE_DEPTH: f64 = 0.15;

/// The largest share of the image a plane [`SceneOptions::auto_planes`]
/// finds may cover. A nebula or dust cloud over more is the scene's
/// backdrop, and stays with the nebula unless given a distance.
pub const AUTO_PLANE_COVERAGE: f64 = 1.0 / 3.0;

/// A distance to use for the catalogued object named `object` in place of
/// the catalogue's own, parsecs.
#[derive(Clone, Debug, PartialEq)]
pub struct DistanceOverride {
    pub object: String,
    pub distance_pc: f64,
}

/// A depth plane lifted, as [`Summary`] tells it.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneSummary {
    pub name: String,
    pub distance_pc: f64,
    /// How the distance was found: given, or as the catalogue found it.
    pub basis: String,
    /// The share of the image it covers, 0 to 1.
    pub coverage: f64,
    /// Whether it blocks light behind it: a dark cloud.
    pub absorbs: bool,
    /// The catalogued object whose distance places it, by name, which a
    /// distance override moves it by; `None` for a given mask or a cloud
    /// of dust no catalogued object names.
    pub object: Option<String>,
}

/// What happens as a video is prepared and rendered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event<'a> {
    /// A step done or a choice made, worth telling the user.
    Note(&'a str),
    /// Something missing that the video goes on without.
    Warning(&'a str),
    /// `done` of `total` frames rendered.
    Frame { done: usize, total: usize },
}

/// Why a video could not be made.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The options or images cannot make a video.
    #[error("{0}")]
    Invalid(String),
    #[error("failed to open {}: {message}", path.display())]
    Catalog { path: PathBuf, message: String },
    #[error("Gaia archive query failed: {0}")]
    Gaia(String),
    #[error(
        "no catalogued object or Gaia distances near the focus point to place the background at; \
         give its distance"
    )]
    NoDistance,
    #[error("the labels' font: {0}")]
    Fonts(String),
    #[error(transparent)]
    Encode(#[from] crate::encode::Error),
    #[error("stopped before finishing")]
    Stopped,
}

/// A stop on a tour, as the options give it.
#[derive(Clone, Debug, PartialEq)]
pub struct TourStop {
    /// The image point the camera centres, or the image's centre if
    /// `None`.
    pub focus: Option<(f64, f64)>,
    /// Fraction of the way to the nebula the camera has flown, 0 to below 1.
    pub dolly: f64,
    /// The lens's magnification over the widest view of the first stop's
    /// place.
    pub zoom: f64,
    /// The frame's turn, degrees anticlockwise.
    pub rotate_deg: f64,
    /// How much of its way here the camera turns rather than moves, 0 to 1.
    pub pan: f64,
    /// Seconds to come here from the stop before, and to stay.
    pub travel: f64,
    pub hold: f64,
    /// Degrees the frame turns anticlockwise while the camera holds here,
    /// easing in and out; later stops' turns count from where it ends.
    pub spin_deg: f64,
    /// How much of its remaining way to the nebula the camera flies on
    /// while it holds here, 0 to below 1.
    pub push: f64,
    /// The title shown while the camera drifts through here, with
    /// [`VideoOptions::tour_titles`]: a planned tour names its target.
    pub title: Option<String>,
    /// An image, its transparency kept, shown while the camera drifts
    /// through here as a title is, in place of the title, whether or not
    /// titles are shown. It is fitted inside the frame and centred.
    pub card: Option<PathBuf>,
    /// The distance of the plane the camera centres the focus point on,
    /// parsecs: a planned tour gives a target's on a depth plane. If
    /// `None`, that of the plane the image shows at the focus point.
    pub depth_pc: Option<f64>,
}

impl Default for TourStop {
    fn default() -> Self {
        Self {
            focus: None,
            dolly: 0.0,
            zoom: 1.0,
            rotate_deg: 0.0,
            pan: 0.0,
            travel: 5.0,
            hold: 0.0,
            spin_deg: 0.0,
            push: 0.0,
            title: None,
            card: None,
            depth_pc: None,
        }
    }
}

/// A [`TourStop`] written `X,Y` or `whole` (the image's centre), then any
/// of `dolly=`, `zoom=`, `rotate=` (degrees), `pan=`, `travel=` and
/// `hold=` (seconds), `spin=` (degrees turned while holding), `push=`
/// (the share of the remaining way flown in while holding), `depth=`
/// (parsecs, the plane the focus point is centred on) and `title=`,
/// separated by spaces: `2700,3400 dolly=0.85 rotate=-20 travel=6
/// hold=1.5 title="Sh2-170"`. A value in double quotes may hold spaces,
/// and `\"` or `\\` within them a quote or backslash.
pub fn parse_stop(text: &str) -> Result<TourStop, String> {
    let parts = words(text)?;
    let mut parts = parts.iter().map(String::as_str);
    let place = parts
        .next()
        .ok_or_else(|| "a stop needs a place, X,Y or whole".to_string())?;
    let mut stop = TourStop {
        focus: if place.eq_ignore_ascii_case("whole") {
            None
        } else {
            let (x, y) = place
                .split_once(',')
                .ok_or_else(|| format!("a stop's place is X,Y or whole; got {place}"))?;
            let number = |part: &str| {
                part.trim()
                    .parse::<f64>()
                    .map_err(|error| format!("{part}: {error}"))
            };
            Some((number(x)?, number(y)?))
        },
        ..TourStop::default()
    };
    for part in parts {
        let (key, value) = part
            .split_once('=')
            .ok_or_else(|| format!("expected key=value in a stop; got {part}"))?;
        if key == "title" {
            stop.title = (!value.is_empty()).then(|| value.to_string());
            continue;
        }
        if key == "card" {
            stop.card = (!value.is_empty()).then(|| PathBuf::from(value));
            continue;
        }
        let value: f64 = value.parse().map_err(|error| format!("{part}: {error}"))?;
        match key {
            "dolly" => stop.dolly = value,
            "zoom" => stop.zoom = value,
            "rotate" => stop.rotate_deg = value,
            "pan" => stop.pan = value,
            "travel" => stop.travel = value,
            "hold" => stop.hold = value,
            "spin" => stop.spin_deg = value,
            "push" => stop.push = value,
            "depth" => stop.depth_pc = Some(value),
            other => {
                return Err(format!(
                    "a stop takes dolly, zoom, rotate, pan, travel, hold, spin, push, depth, \
                     title and card; got {other}"
                ));
            }
        }
    }
    Ok(stop)
}

/// `text` split at spaces, but not those within double quotes, which go.
fn words(text: &str) -> Result<Vec<String>, String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let (mut quoted, mut started) = (false, false);
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => {
                quoted = !quoted;
                started = true;
            }
            '\\' if quoted => match chars.next() {
                Some(next @ ('"' | '\\')) => word.push(next),
                Some(next) => {
                    word.push('\\');
                    word.push(next);
                }
                None => word.push('\\'),
            },
            c if c.is_whitespace() && !quoted => {
                if started {
                    words.push(std::mem::take(&mut word));
                    started = false;
                }
            }
            c => {
                word.push(c);
                started = true;
            }
        }
    }
    if quoted {
        return Err(format!("a stop has an unclosed quote: {text}"));
    }
    if started {
        words.push(word);
    }
    Ok(words)
}

/// What preparing a video's scene takes beyond its images and plate
/// solution: where the distances come from and how the stars are cut.
/// Changing any of these needs a new [`Parallax::prepare`]. The defaults are
/// `seiza parallax-video`'s.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneOptions {
    /// Distance to the nebula or galaxy behind the stars, parsecs; if
    /// `None`, that of the catalogued object at `distance_focus`, else the
    /// median Gaia distance of the stars near it.
    pub distance_pc: Option<f64>,
    /// The image point whose object, or nearby stars, give the nebula's
    /// distance when `distance_pc` is `None`, image pixels. If `None`, the
    /// camera's destination: [`VideoOptions::focus`], a tour's first stop
    /// flown in toward, or an automatic tour's target most worth a visit;
    /// else the image's centre. The scene keeps that depth whatever later
    /// camera moves aim at.
    pub distance_focus: Option<(f64, f64)>,
    /// Distance for stars without one, parsecs; if `None`, the matched
    /// stars' median when the nebula lies nearer than it, else the star
    /// field's far end (the matched stars' 90th percentile), and never
    /// within 3% of the nebula, where they would join its plane.
    pub unmatched_distance_pc: Option<f64>,
    /// The object catalog and object distance files, or the standard places
    /// if `None`.
    pub objects: Option<PathBuf>,
    pub object_distances: Option<PathBuf>,
    /// The offline star distance file, or the standard places if `None`.
    pub star_distances: Option<PathBuf>,
    /// The faintest Gaia G magnitude to match.
    pub gaia_max_mag: f32,
    /// Where fetched Gaia and Hipparcos fields are kept, or Seiza's data
    /// directory if `None`.
    pub gaia_cache: Option<PathBuf>,
    /// Whether to ask the archives when the offline star distance file is
    /// missing or too shallow. Without either, every star sits at one
    /// distance.
    pub online: bool,
    /// How many stars, brightest first, fly at their own distances; all if
    /// `None`. The rest go as `small_stars` says.
    pub max_stars: Option<usize>,
    pub small_stars: SmallStars,
    /// Leave catalogued galaxies on the nebula's plane rather than lifting
    /// them onto the far field.
    pub keep_galaxies: bool,
    /// Dim what lies behind the nebula by the dust it slides behind,
    /// mapped from how few stars show through.
    pub dust: bool,
    /// The light the dust lets through is the share of stars seen to this
    /// power.
    pub dust_opacity: f32,
    /// Dust drawn as a map, in place of the one mapped from the stars;
    /// [`SceneOptions::dust`] must be on. Where it is opaque, the stars seen
    /// through it fly in front of the background, which the counted map
    /// never does.
    pub dust_map: Option<DustMap>,
    /// Regions of the starless image lifted onto planes at their own
    /// distances: clouds in front of the nebula or behind it. What lay
    /// behind each is filled in from around it. Each costs a draw over as
    /// much of the frame as it covers.
    pub planes: Vec<DepthPlane>,
    /// Lift the catalogued nebulae in the field that lie at distances of
    /// their own onto planes, besides `planes`: those whose distance was
    /// measured, or borrowed from an object about them (never one typical
    /// of their kind), that lie at least [`AUTO_PLANE_DEPTH`] nearer or
    /// farther than the nebula, and whose range of distances leaves the
    /// nebula's out. Each takes its catalogued outline, or else its ellipse
    /// sized by the image's light, and only its light. One holding the
    /// distance focus is the nebula, or holds it, and keeps to the nebula's
    /// plane. So does one spanning most of the image, or that would cover
    /// more than [`AUTO_PLANE_COVERAGE`] of it unless given a distance, as
    /// the scene's backdrop. A nebula inside one of these that takes its
    /// distance stays with it.
    ///
    /// A dark nebula moves with the cloud of dust it lies in, as the star
    /// counts find the clouds. The cloud holding the distance focus stays
    /// on the nebula's plane, its dark nebulae with it. Another cloud at
    /// least [`AUTO_PLANE_DEPTH`] in front of the nebula moves as one
    /// plane, at the distance of its largest dark nebula that the
    /// catalogue places, also from the nearest cloud measured, else where
    /// the Gaia stars seen through it run short, unless it would cover more
    /// than [`AUTO_PLANE_COVERAGE`] of the image and no distance was given
    /// for it. A dark nebula in clear sky stands alone, likewise placed.
    pub auto_planes: bool,
    /// Distances to use for catalogued objects in place of the catalogue's:
    /// for the nebula's distance and the automatic planes.
    pub distance_overrides: Vec<DistanceOverride>,
    /// The part of the image to film, image pixels: left, top, width,
    /// height. The starless and stars images, the plate solution, and any
    /// plane masks and dust map (at their own sizes) are cut to it before
    /// anything else, so every image point the other options give, such as
    /// focus points and tour stops, counts in the cropped image.
    pub crop: Option<(usize, usize, usize, usize)>,
}

impl Default for SceneOptions {
    fn default() -> Self {
        Self {
            distance_pc: None,
            distance_focus: None,
            unmatched_distance_pc: None,
            objects: None,
            object_distances: None,
            star_distances: None,
            gaia_max_mag: 16.0,
            gaia_cache: None,
            online: true,
            max_stars: None,
            small_stars: SmallStars::Drop,
            keep_galaxies: false,
            dust: true,
            dust_opacity: 3.0,
            dust_map: None,
            planes: Vec::new(),
            auto_planes: false,
            distance_overrides: Vec::new(),
            crop: None,
        }
    }
}

impl SceneOptions {
    /// Whether these options can prepare a scene.
    pub fn check(&self) -> Result<(), Error> {
        let invalid = |message: &str| Err(Error::Invalid(message.into()));
        let distance = |distance: f64| distance.is_finite() && distance > 0.0;
        if !(self.dust_opacity.is_finite() && self.dust_opacity >= 0.0) {
            return invalid("the dust opacity must be at least 0");
        }
        if let Some(map) = &self.dust_map {
            if !self.dust {
                return invalid("a dust map needs the dust on");
            }
            map.check().map_err(Error::Invalid)?;
        }
        if self.distance_pc.is_some_and(|parsecs| !distance(parsecs))
            || self
                .unmatched_distance_pc
                .is_some_and(|parsecs| !distance(parsecs))
        {
            return invalid("the distances must be positive numbers of parsecs");
        }
        if !self.gaia_max_mag.is_finite() {
            return invalid("the faintest Gaia magnitude must be a number");
        }
        if self
            .distance_focus
            .is_some_and(|(x, y)| !x.is_finite() || !y.is_finite())
        {
            return invalid("the distance focus must be a point");
        }
        for plane in &self.planes {
            if !distance(plane.distance_pc) {
                return Err(Error::Invalid(format!(
                    "the depth plane {}'s distance must be a positive number of parsecs",
                    plane.name
                )));
            }
            plane
                .mask
                .check()
                .map_err(|error| Error::Invalid(format!("{}: {error}", plane.name)))?;
        }
        for given in &self.distance_overrides {
            if given.object.trim().is_empty() || !distance(given.distance_pc) {
                return invalid(
                    "a distance given for an object needs the object's name and a positive \
                     number of parsecs",
                );
            }
        }
        Ok(())
    }
}

/// How a prepared scene is filmed: the camera, the output and the labels.
/// [`Parallax::reconfigure`] films the same scene anew with other
/// `VideoOptions`, without preparing it again. The defaults are `seiza
/// parallax-video`'s.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoOptions {
    /// The image point the camera flies toward, image pixels; the image's
    /// centre if `None`. It moves the camera only; the nebula's depth is
    /// [`SceneOptions`]'s.
    pub focus: Option<(f64, f64)>,
    pub start: Start,
    /// Fraction of the way to the nebula the camera flies, 0 to below 1.
    pub dolly: f64,
    /// Sideways swing at the middle of the shot, as a fraction of the
    /// nebula's distance, and its direction, degrees anticlockwise from the
    /// image's rightward axis.
    pub truck: f64,
    pub truck_angle_deg: f64,
    /// How much of its way to the focus point the camera turns, 0 to 1.
    pub pan: f64,
    /// How far the first frame zooms in, and how much more the last frame
    /// is magnified by lengthening the lens.
    pub zoom: f64,
    pub zoom_end: f64,
    /// How the widest view meets an image of another shape: filling the
    /// frame and cutting the image's longer ends, or showing all of it with
    /// black beside it. An automatic tour is planned for it.
    pub framing: Framing,
    /// The frame's turn at the first and last frames, degrees
    /// anticlockwise, taken as given: 0 to 360 turns once anticlockwise and
    /// 0 to -360 once clockwise.
    pub rotate_deg: (f64, f64),
    pub easing: Easing,
    pub quality: Quality,
    /// How much nearer a star has come, its first-frame distance over its
    /// distance now, is its growth. It swells as the square root of its
    /// growth capped at `growth_limit` (the default 4 doubles its size),
    /// and fades out from `fade_from` until gone at twice it.
    pub growth_limit: f64,
    pub fade_from: f64,
    /// A tour of stops instead of the single move toward `focus`: the
    /// camera glides through them, slowing through each it holds at. The
    /// first is the opening view. Its length replaces `seconds`, and the
    /// move's own settings (`start`, `dolly`, `truck`, `pan`, `rotate_deg`,
    /// `zoom_end`, `easing`) go unused. Stops' turns are taken as given,
    /// unwrapped.
    pub tour: Vec<TourStop>,
    /// Plan a tour of the catalogued objects in the field, when `tour` is
    /// empty: the most prominent, visited in a short round from the whole
    /// image and back.
    pub auto_tour: Option<AutoTour>,
    /// How fast a tour moves on through a stop it holds at, as a share of
    /// its pace between stops, so it never quite stops; 0 comes to rest.
    pub tour_glide: f64,
    /// Show each tour stop's title low in the frame while the camera
    /// drifts through the stop, fading in and out.
    pub tour_titles: bool,
    /// End the tour where it began, so the video loops: the last frame
    /// leads into the first. The tour gains a last stop at the first's
    /// view unless it ends there, and turns back to the first's frame by
    /// the nearest whole turns, spins included. The camera drifts through
    /// the join at half a glide's pace; the first and last stops' pushes
    /// go unused.
    pub tour_loop: bool,
    /// Frame size, even sides.
    pub size: (usize, usize),
    /// The image's shape with this longer side, each side even, in place
    /// of `size`, from a size such as `4k-image`: a 4:3 image at 3840 is
    /// filmed at 3840 × 2880, all of it. The image's size is known only
    /// once it is read (and cropped), so this waits until then.
    pub image_shaped_size: Option<usize>,
    pub seconds: f64,
    pub fps: u32,
    /// Label the catalogued objects in the field, the most prominent
    /// `overlay_density` share of them.
    pub overlay: bool,
    pub overlay_density: f64,
    /// The caller's own labels, in `label_color`.
    pub labels: Vec<CustomLabel>,
    pub label_color: [u8; 3],
    /// A line of text in a corner of every frame.
    pub watermark: Option<String>,
    /// An image drawn in that corner of every frame, such as a logo, its
    /// transparency kept; with text too, it sits just inside the text.
    pub watermark_image: Option<PathBuf>,
    /// The corner the watermark sits in.
    pub watermark_corner: Corner,
    /// The watermark image's height as a share of the frame's shorter
    /// side; it is at most a third of the frame's width.
    pub watermark_size: f64,
    /// The watermark's opacity, 0 to 1, times the image's own alpha.
    pub watermark_opacity: f64,
    /// Cards laid over the start and the end of the video: images, their
    /// transparency kept, fitted inside the frame and centred. The opening
    /// card shows whole from the first frame and then fades out; the
    /// closing card fades in and shows whole at the last. A looping tour
    /// shows both across its join instead, so it still loops.
    pub intro_card: Option<PathBuf>,
    pub outro_card: Option<PathBuf>,
    /// How long the opening and closing cards each show, seconds, fades
    /// included.
    pub card_seconds: f64,
}

impl Default for VideoOptions {
    fn default() -> Self {
        Self {
            focus: None,
            start: Start::Focus,
            dolly: 0.4,
            truck: 0.0,
            truck_angle_deg: 0.0,
            pan: 0.0,
            zoom: 1.0,
            zoom_end: 1.0,
            framing: Framing::Fill,
            rotate_deg: (0.0, 0.0),
            easing: Easing::InOut,
            image_shaped_size: None,
            quality: Quality::Standard,
            growth_limit: 4.0,
            fade_from: 6.0,
            tour: Vec::new(),
            auto_tour: None,
            tour_glide: 0.2,
            tour_titles: false,
            tour_loop: false,
            size: (1920, 1080),
            seconds: 8.0,
            fps: 30,
            overlay: false,
            overlay_density: overlay::DEFAULT_DENSITY,
            labels: Vec::new(),
            label_color: [0xf0, 0xf4, 0xf8],
            watermark: None,
            watermark_image: None,
            watermark_corner: Corner::BottomRight,
            watermark_size: 0.06,
            watermark_opacity: 0.8,
            intro_card: None,
            outro_card: None,
            card_seconds: 3.0,
        }
    }
}

/// The longest video, seconds, and the most frames: past these a typo
/// (`--seconds 1e9`, `hold=inf`) would fit and render for ever.
const LONGEST_SECONDS: f64 = 24.0 * 3600.0;
const MOST_FRAMES: f64 = 5_000_000.0;

/// Whether a frame of `size` pixels can be encoded: even sides, at least 16.
fn check_frame((width, height): (usize, usize)) -> Result<(), Error> {
    if width < 16 || height < 16 || width % 2 != 0 || height % 2 != 0 {
        return Err(Error::Invalid(
            "the frame's sides must be even and at least 16".into(),
        ));
    }
    Ok(())
}

impl VideoOptions {
    /// Film at `size`: a fixed `size`, or an [`Self::image_shaped_size`].
    pub fn set_frame_size(&mut self, size: crate::encode::FrameSize) {
        match size {
            crate::encode::FrameSize::Fixed(width, height) => {
                self.size = (width, height);
                self.image_shaped_size = None;
            }
            crate::encode::FrameSize::ImageShaped(long) => self.image_shaped_size = Some(long),
        }
    }

    /// These options with any [`Self::image_shaped_size`] made a `size`
    /// for an image of `image` pixels.
    pub fn sized_for(&self, image: (usize, usize)) -> Self {
        match self.image_shaped_size {
            Some(long) => Self {
                size: crate::encode::FrameSize::ImageShaped(long).for_image(image),
                image_shaped_size: None,
                ..self.clone()
            },
            None => self.clone(),
        }
    }

    /// Whether these options can film a video.
    pub fn check(&self) -> Result<(), Error> {
        let invalid = |message: &str| Err(Error::Invalid(message.into()));
        let point = |(x, y): (f64, f64)| x.is_finite() && y.is_finite();
        if self
            .image_shaped_size
            .is_some_and(|long| long < 16 || long % 2 != 0)
        {
            return invalid("an image-shaped frame's longer side must be even and at least 16");
        }
        if !(0.0..1.0).contains(&self.dolly) {
            return invalid("the dolly must be at least 0 and below 1");
        }
        if !(0.0..=1.0).contains(&self.pan) {
            return invalid("the pan must be from 0 to 1");
        }
        if !(0.0..=1.0).contains(&self.overlay_density) {
            return invalid("the overlay density must be from 0 to 1");
        }
        if !self.rotate_deg.0.is_finite() || !self.rotate_deg.1.is_finite() {
            return invalid("the rotation must be numbers of degrees");
        }
        if !(self.zoom.is_finite() && self.zoom >= 1.0)
            || !(self.zoom_end.is_finite() && self.zoom_end >= 1.0)
        {
            return invalid("the zoom and the end zoom must be at least 1");
        }
        if !self.truck.is_finite() || !self.truck_angle_deg.is_finite() {
            return invalid("the truck must be a number");
        }
        if !(self.growth_limit.is_finite() && self.growth_limit >= 1.0) {
            return invalid("the growth limit must be at least 1");
        }
        if !(self.fade_from.is_finite() && self.fade_from > 0.0) {
            return invalid("stars must start fading at a positive growth");
        }
        if !(self.seconds.is_finite() && self.seconds > 0.0) || self.fps == 0 {
            return invalid("the length and frame rate must be positive");
        }
        check_frame(self.size)?;
        if self.focus.is_some_and(|focus| !point(focus)) {
            return invalid("the focus must be a point");
        }
        if self
            .labels
            .iter()
            .any(|label| !point((label.x, label.y)) || !label.radius.is_finite())
        {
            return invalid("a label's place and radius must be numbers");
        }
        if self.tour.len() == 1 {
            return invalid("a tour needs at least two stops");
        }
        for stop in &self.tour {
            if stop.focus.is_some_and(|focus| !point(focus)) {
                return invalid("a stop's focus must be a point");
            }
            if !(0.0..1.0).contains(&stop.dolly)
                || !(0.0..=1.0).contains(&stop.pan)
                || !(0.0..1.0).contains(&stop.push)
            {
                return invalid(
                    "a stop's dolly and push must be from 0 to below 1, and its pan 0 to 1",
                );
            }
            if !(stop.zoom.is_finite() && stop.zoom > 0.0)
                || !stop.rotate_deg.is_finite()
                || !stop.spin_deg.is_finite()
            {
                return invalid("a stop's zoom must be positive, and its turn and spin numbers");
            }
            if !(stop.travel.is_finite()
                && stop.travel >= 0.0
                && stop.hold.is_finite()
                && stop.hold >= 0.0)
            {
                return invalid("a stop's travel and hold must be at least 0 seconds");
            }
            if stop
                .depth_pc
                .is_some_and(|depth| !(depth.is_finite() && depth > 0.0))
            {
                return invalid("a stop's depth must be a positive number of parsecs");
            }
        }
        if let Some(auto) = &self.auto_tour {
            auto.check().map_err(Error::Invalid)?;
        }
        if !(self.tour_glide.is_finite() && self.tour_glide >= 0.0) {
            return invalid("a tour's glide must be at least 0");
        }
        if (self.tour_titles || self.tour_loop) && self.tour.is_empty() && self.auto_tour.is_none()
        {
            return invalid("tour titles and a looping tour need a tour");
        }
        if !self.tour.is_empty() && self.seconds() <= 0.0 {
            return invalid("a tour must take some time");
        }
        if !(self.watermark_size > 0.0 && self.watermark_size <= 1.0) {
            return invalid("the watermark's size must be above 0 and at most 1");
        }
        if !(0.0..=1.0).contains(&self.watermark_opacity) {
            return invalid("the watermark's opacity must be from 0 to 1");
        }
        if !(self.card_seconds.is_finite() && self.card_seconds > 0.0) {
            return invalid("the cards must show for some seconds");
        }
        if self.seconds() > LONGEST_SECONDS || self.seconds() * self.fps as f64 > MOST_FRAMES {
            return invalid("the video must be under a day long and five million frames");
        }
        Ok(())
    }

    /// The tour's stops, ending at the first's view when the tour loops.
    pub fn stops(&self) -> Vec<TourStop> {
        if self.tour_loop {
            close_loop(&self.tour)
        } else {
            self.tour.clone()
        }
    }

    /// The video's length, seconds: the tour's, if there is one.
    pub fn seconds(&self) -> f64 {
        if self.tour.is_empty() {
            return self.seconds;
        }
        self.stops()
            .iter()
            .enumerate()
            .map(|(index, stop)| if index > 0 { stop.travel } else { 0.0 } + stop.hold)
            .sum()
    }

    /// The number of frames the video runs to.
    pub fn frames(&self) -> usize {
        ((self.seconds() * self.fps as f64).round() as usize).max(2)
    }
}

/// Everything about a video but its images and plate solution.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParallaxOptions {
    pub scene: SceneOptions,
    pub video: VideoOptions,
}

impl ParallaxOptions {
    /// Whether these options can make a video.
    pub fn check(&self) -> Result<(), Error> {
        self.scene.check()?;
        self.video.check()
    }
}

/// How the fit changed the framing asked for, so a caller can explain it.
/// Each pair is what was asked and what the video uses.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FitSummary {
    /// The first frame's zoom.
    pub zoom: (f64, f64),
    /// The single move's pan.
    pub pan: (f64, f64),
    /// How early the single move's sideways travel toward the focus comes,
    /// 1 as asked.
    pub lead: (f64, f64),
    /// The single move's truck, as a fraction of the nebula's distance.
    pub truck: (f64, f64),
    /// A tour's stops, in order.
    pub stops: Vec<StopFit>,
    /// Whether every frame keeps the layers' edges out of view. When it
    /// does not (a stop too near the image's edge to turn there, say),
    /// some frames show past the image.
    pub inside: bool,
}

/// How the fit changed one tour stop: its zoom, pan and dolly, asked and
/// used.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StopFit {
    pub zoom: (f64, f64),
    pub pan: (f64, f64),
    pub dolly: (f64, f64),
}

/// Tell what `scene`'s dust is and does, and note it in `summary`.
fn describe_dust(scene: &Scene, summary: &mut Summary, report: &mut dyn FnMut(Event)) {
    let Some(dust) = &scene.dust else {
        return;
    };
    let (cells, _) = dust.cells();
    let mut sorted = cells.to_vec();
    sorted.sort_by(f32::total_cmp);
    let (middle, thickest) = (sorted[sorted.len() / 2], sorted[0]);
    let opaque = sorted.partition_point(|&transmission| transmission <= crate::dust::OPAQUE);
    summary.dust_transmission = Some((middle, thickest));
    summary.dust_given = dust.given();
    summary.stars_in_front_of_dust = scene.stars_brought_forward();
    if dust.given() {
        report(Event::Note(&format!(
            "dust from the given map: median transmission {:.0}%, thickest {:.0}%, opaque \
             over {:.0}% of the image; {} star{} seen where it is opaque fl{} in front of \
             the nebula",
            100.0 * middle,
            100.0 * thickest,
            100.0 * opaque as f32 / sorted.len() as f32,
            summary.stars_in_front_of_dust,
            if summary.stars_in_front_of_dust == 1 {
                ""
            } else {
                "s"
            },
            if summary.stars_in_front_of_dust == 1 {
                "ies"
            } else {
                "y"
            },
        )));
    } else {
        report(Event::Note(&format!(
            "dust from the star counts: median transmission {:.0}%, thickest {:.0}%",
            100.0 * middle,
            100.0 * thickest,
        )));
    }
}

/// What preparing a video found, and how its camera was fitted.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Summary {
    /// Stars found in the stars image.
    pub detected_stars: usize,
    /// Catalogue stars in the field.
    pub gaia_stars: usize,
    pub hipparcos_stars: usize,
    /// Detections matched to a Gaia star, and those given a distance.
    pub gaia_matches: usize,
    pub with_distance: usize,
    /// The nebula's distance, how it was found, and the image point it was
    /// taken at.
    pub background_distance_pc: f64,
    pub background_basis: String,
    pub background_focus: (f64, f64),
    /// Where stars without a distance were put.
    pub unmatched_distance_pc: f64,
    /// Stars cut out to fly at their own distances.
    pub flying_stars: usize,
    /// Galaxies lifted onto the far field.
    pub galaxies_lifted: Vec<String>,
    /// The depth planes lifted, smallest first.
    pub planes: Vec<PlaneSummary>,
    /// The dust map's median and least transmission, 0 to 1, if one was
    /// made.
    pub dust_transmission: Option<(f32, f32)>,
    /// Whether the dust came from a given map rather than the star counts.
    pub dust_given: bool,
    /// Stars seen where a given map is opaque, and so brought in front of
    /// the background.
    pub stars_in_front_of_dust: usize,
    /// Catalogued objects in the field to label.
    pub labelled_objects: usize,
    /// How the camera was fitted.
    pub fit: FitSummary,
}

/// What preparation made once, shared by every video of the scene.
struct Prepared {
    scene: Scene,
    /// The dust mapped from the star counts, kept when a given map takes
    /// its place so that it can come back.
    counted_dust: Option<Arc<Dust>>,
    /// The plate solution of the image filmed, after any crop.
    wcs: Wcs,
    /// The plate solution of the image as given, and the crop cut from it.
    source_wcs: Wcs,
    crop: Option<(usize, usize, usize, usize)>,
    source_size: (usize, usize),
    image_size: (usize, usize),
    /// Galaxies lifted onto the far field.
    lifted: Vec<overlay::Lifted>,
    objects: Option<PathBuf>,
    summary: Summary,
    /// The catalogued objects' labels, read on first need and kept for
    /// every video of the scene.
    catalog_marks: OnceLock<Arc<Vec<Mark>>>,
}

impl Prepared {
    /// The catalogued objects' labels, read once they can be: a failure,
    /// such as a catalogue not yet installed, is tried again next time.
    fn catalog_marks(&self) -> Result<Arc<Vec<Mark>>, Error> {
        if let Some(marks) = self.catalog_marks.get() {
            return Ok(Arc::clone(marks));
        }
        let path = seiza::data_paths::objects(self.objects.as_deref()).map_err(|error| {
            Error::Invalid(format!(
                "labelling objects needs the object catalog (seiza setup): {error}"
            ))
        })?;
        let catalog = open_objects(&path)?;
        let (width, height) = self.image_size;
        let marks = overlay::catalog_marks(
            &catalog,
            &self.wcs,
            (width as u32, height as u32),
            &self.scene,
            &self.lifted,
        )?;
        Ok(Arc::clone(
            self.catalog_marks.get_or_init(|| Arc::new(marks)),
        ))
    }
}

/// A prepared parallax video: a scene cut into depths, filmed by a fitted
/// camera. Its frames may be drawn from several threads at once.
/// [`Parallax::reconfigure`] films the same scene anew, sharing it: the
/// scene is held once, however many videos film it, and lives until the
/// last of them is dropped.
pub struct Parallax {
    prepared: Arc<Prepared>,
    video: VideoOptions,
    shot: Shot,
    overlay: Option<Overlay>,
    summary: Summary,
}

impl std::fmt::Debug for Parallax {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Parallax")
            .field("shot", &self.shot)
            .field("summary", &self.summary)
            .finish_non_exhaustive()
    }
}

impl Parallax {
    /// Prepare a video of `starless` and `stars`, a stretched image split
    /// into its starless image and its unscreened stars (values 0 to 1),
    /// whose sky `wcs` gives. `report` hears each step; `stop` is asked
    /// between steps and through the fit, and when it answers true the
    /// preparation ends with [`Error::Stopped`].
    pub fn prepare(
        starless: &Rgb32FImage,
        stars: &Rgb32FImage,
        wcs: &Wcs,
        options: &ParallaxOptions,
        report: &mut dyn FnMut(Event),
        stop: &dyn Fn() -> bool,
    ) -> Result<Self, Error> {
        options.check()?;
        let (source_wcs, crop) = (wcs.clone(), options.scene.crop);
        let source_size = (starless.width() as usize, starless.height() as usize);
        let cropped;
        let (starless, stars, wcs, options) = match options.scene.crop {
            Some(crop) => {
                cropped = crop_inputs(starless, stars, wcs, options, crop)?;
                (&cropped.0, &cropped.1, &cropped.2, &cropped.3)
            }
            None => (starless, stars, wcs, options),
        };
        let sized;
        let options = match options.video.image_shaped_size {
            Some(_) => {
                let image = (starless.width() as usize, starless.height() as usize);
                sized = ParallaxOptions {
                    video: options.video.sized_for(image),
                    ..options.clone()
                };
                &sized
            }
            None => options,
        };
        let halt = || if stop() { Err(Error::Stopped) } else { Ok(()) };
        let scene_options = &options.scene;
        // An automatic tour's best target may give the nebula's distance.
        // The tour itself is planned once the scene is cut into depths, so
        // each stop frames its target at the depth of its plane.
        let video = &options.video;
        let planned_focus = match &video.auto_tour {
            Some(auto) if video.tour.is_empty() => Some(
                plan_tour(
                    wcs,
                    starless.dimensions(),
                    scene_options.objects.as_deref(),
                    video.size,
                    &AutoTour {
                        framing: video.framing,
                        ..*auto
                    },
                )?
                .focus,
            ),
            _ => None,
        };
        if starless.dimensions() != stars.dimensions() {
            return Err(Error::Invalid(format!(
                "the starless image is {:?} but the stars image is {:?}",
                starless.dimensions(),
                stars.dimensions()
            )));
        }
        let (width, height) = (stars.width() as usize, stars.height() as usize);
        check_places(video, (width, height))?;
        let given_distances = given_distances(scene_options)?;
        // A map or mask of the wrong shape is refused before the slow work.
        let given_dust = scene_options
            .dust_map
            .as_ref()
            .map(|map| Dust::from_source_map(map, source_size, crop).map(Arc::new))
            .transpose()
            .map_err(Error::Invalid)?;
        let mut regions = Vec::new();
        for plane in &scene_options.planes {
            match Region::from_mask(&plane.mask, width, height)
                .map_err(|error| Error::Invalid(format!("{}: {error}", plane.name)))?
            {
                Some(region) => regions.push(PlaneToLift {
                    name: plane.name.clone(),
                    region,
                    distance_pc: plane.distance_pc,
                    basis: "given".into(),
                    blocks: true,
                    object: None,
                }),
                None => report(Event::Warning(&format!(
                    "the depth plane {}'s mask marks nothing; it is left out",
                    plane.name
                ))),
            }
        }
        let scale = wcs.scale_arcsec_per_px();
        if !(scale.is_finite() && scale > 0.0) {
            return Err(Error::Invalid(format!(
                "the plate solution's scale ({scale} arcsec a pixel) is unusable"
            )));
        }
        let shallow: Vec<&str> = [("starless", starless), ("stars", stars)]
            .into_iter()
            .filter(|(_, image)| eight_bit(image))
            .map(|(name, _)| name)
            .collect();
        if !shallow.is_empty() {
            report(Event::Warning(&format!(
                "the {} image{} only 8 bits a channel: a bright star's core flattens, which \
                 can misplace it and lose its distance, and smooth nebula bands as the camera \
                 nears it; split and export at 16 bits (TIFF or PNG) for the best video",
                shallow.join(" and "),
                if shallow.len() > 1 { "s have" } else { " has" }
            )));
        }
        let mut summary = Summary::default();
        let focal_px = 206_264.806_247 / scale;

        let stars_light = LightImage::from_display(stars);
        let detections = seiza_stars::fold_core_fragments(crate::find_stars(&stars_light, 5.0));
        summary.detected_stars = detections.len();
        report(Event::Note(&format!(
            "{} stars found in the stars image",
            detections.len()
        )));
        halt()?;
        let source = FieldSource {
            star_distances: scene_options.star_distances.as_deref(),
            gaia_max_mag: scene_options.gaia_max_mag,
            cache: scene_options.gaia_cache.as_deref(),
            online: scene_options.online,
        };
        let (gaia, hipparcos) = field::catalogue_stars(&source, wcs, (width, height), report)?;
        (summary.gaia_stars, summary.hipparcos_stars) = (gaia.len(), hipparcos.len());
        let (matched, gaia_matches) =
            field::match_distances(&detections, &gaia, &hipparcos, wcs, scale);
        let with_distance = matched
            .iter()
            .filter(|star| star.distance_pc.is_some())
            .count();
        (summary.gaia_matches, summary.with_distance) = (gaia_matches, with_distance);
        report(Event::Note(&format!(
            "{gaia_matches} matched to Gaia, {with_distance} with a distance"
        )));
        halt()?;

        let centre = ((width as f64 - 1.0) / 2.0, (height as f64 - 1.0) / 2.0);
        // The nebula's depth is taken where the scene options say, else at
        // the camera's destination: its focus, a tour's first stop flown in
        // toward, or an automatic tour's target most worth a visit.
        let first_visited = video
            .tour
            .iter()
            .find(|stop| stop.dolly > 0.0)
            .map(|stop| stop.focus.unwrap_or(centre));
        let focus = scene_options
            .distance_focus
            .or(video.focus)
            .or(planned_focus)
            .or(first_visited)
            .unwrap_or(centre);
        if focus.0 < 0.0 || focus.1 < 0.0 || focus.0 >= width as f64 || focus.1 >= height as f64 {
            return Err(Error::Invalid(format!(
                "the distance focus {focus:?} is outside the {width}x{height} image"
            )));
        }
        let (distance, basis) = match scene_options.distance_pc {
            Some(distance) => (distance, "given".to_string()),
            None => {
                match target_distance(scene_options, wcs, (width, height), focus, &given_distances)
                {
                    Ok(Some((name, distance, basis))) => {
                        (distance, format!("the distance of {name} ({basis})"))
                    }
                    found => {
                        if let Err(error) = found {
                            report(Event::Warning(&format!("no object distance: {error}")));
                        }
                        let distance = field::median_distance_near(&matched, focus, width, height)
                            .ok_or(Error::NoDistance)?;
                        (
                            distance,
                            "the median distance of the stars near the focus point (no catalogued \
                         object there)"
                                .to_string(),
                        )
                    }
                }
            }
        };
        report(Event::Note(&format!(
            "background at {distance:.0} pc, {basis}"
        )));
        (
            summary.background_distance_pc,
            summary.background_basis,
            summary.background_focus,
        ) = (distance, basis, focus);

        // Most stars too faint to match are field stars: well beyond a
        // nearby target, so at the matched stars' median distance, and in
        // front of a target beyond most stars (a galaxy above all), so at
        // the star field's far end. Never on the nebula's plane, where they
        // would join it and swell with it rather than fly as points.
        let unmatched = scene_options.unmatched_distance_pc.unwrap_or_else(|| {
            let mut known: Vec<f64> = matched.iter().filter_map(|star| star.distance_pc).collect();
            known.sort_by(f64::total_cmp);
            let share = |part: f64| {
                known
                    .get(((known.len() as f64 * part) as usize).min(known.len().saturating_sub(1)))
                    .copied()
            };
            let placed = match (share(0.5), share(0.9)) {
                (Some(median), _) if distance < median => median,
                (_, Some(far)) => far,
                _ => distance,
            };
            if (placed - distance).abs() <= 0.03 * distance {
                distance * if placed >= distance { 1.03 } else { 0.97 }
            } else {
                placed
            }
        });
        summary.unmatched_distance_pc = unmatched;
        report(Event::Note(&format!(
            "stars without a distance at {unmatched:.0} pc"
        )));
        let placed: Vec<Star> = matched
            .iter()
            .map(|star| Star {
                distance_pc: Some(star.distance_pc.unwrap_or(unmatched)),
                ..*star
            })
            .collect();
        // The stars seen through the dust: all but the matched ones in
        // front of it. The unmatched ones count however near the dust they
        // were placed, as faint stars are mostly far. Their counts map the
        // dust, and find its clouds.
        let behind: Vec<(f64, f64)> = matched
            .iter()
            .filter(|star| star.distance_pc.is_none_or(|pc| pc > distance))
            .map(|star| (star.x, star.y))
            .collect();
        let counts = (scene_options.dust || scene_options.auto_planes)
            .then(|| Dust::from_star_counts(&behind, width, height, scene_options.dust_opacity))
            .flatten();

        let mut starless = LightImage::from_display(starless);
        // Galaxies lie far beyond everything else, but the star remover
        // leaves them on the nebula's plane.
        let mut galaxies = Vec::new();
        if !scene_options.keep_galaxies {
            match galaxies_in_image(scene_options, wcs, (width, height), focus) {
                Ok(found) => {
                    for (name, extent) in found {
                        if let Some(sprite) =
                            crate::lift_object(&mut starless, &extent, GALAXY_DISTANCE_PC)
                        {
                            galaxies.push((name, extent, sprite));
                        }
                    }
                }
                Err(error) => report(Event::Warning(&format!("no galaxies lifted: {error}"))),
            }
            if !galaxies.is_empty() {
                let names: Vec<&str> = galaxies
                    .iter()
                    .take(5)
                    .map(|(name, _, _)| name.as_str())
                    .collect();
                report(Event::Note(&format!(
                    "{} galaxies lifted onto the far field: {}{}",
                    galaxies.len(),
                    names.join(", "),
                    if galaxies.len() > names.len() {
                        ", ..."
                    } else {
                        ""
                    }
                )));
            }
        }
        if scene_options.auto_planes {
            let (glowing, dark) = catalogued_planes(
                scene_options,
                wcs,
                &starless,
                (focus, distance),
                &given_distances,
                report,
            )
            .unwrap_or_else(|error| {
                report(Event::Warning(&format!(
                    "no catalogued nebulae as depth planes: {error}"
                )));
                Default::default()
            });
            // Gaia's stars with distances, where they lie in the image: the
            // dust clouds' stars.
            let gaia_stars: Vec<(f64, f64, f64)> = gaia
                .iter()
                .filter_map(|star| {
                    let star_distance = star.best_distance()?;
                    let (x, y) = wcs.world_to_pixel(star.ra, star.dec)?;
                    (x >= 0.0 && y >= 0.0 && x < width as f64 && y < height as f64).then_some((
                        x,
                        y,
                        star_distance,
                    ))
                })
                .collect();
            let clouds = counts.as_ref().map(Clouds::find);
            let dusty = cloud_planes(
                clouds.as_ref(),
                dark,
                &gaia_stars,
                (focus, distance),
                (width, height),
                report,
            );
            if glowing.is_empty() && dusty.is_empty() {
                report(Event::Note(
                    "no catalogued nebula or dust cloud in the field lies at a distance of its own",
                ));
            }
            regions.extend(glowing);
            regions.extend(dusty);
        }
        // The depth planes, smallest first: where one lies within another,
        // the smaller takes its own light, and the larger the fill left
        // behind it.
        regions.sort_by(|a, b| a.region.area().total_cmp(&b.region.area()));
        let mut planes = Vec::new();
        for plane in &regions {
            let lifted = crate::lift_region(
                &mut starless,
                &plane.region,
                plane.distance_pc,
                &plane.name,
                plane.blocks && plane.distance_pc < distance,
                &planes,
            );
            match lifted {
                Some(lifted) => {
                    let coverage = plane.region.area() / (width * height) as f64;
                    let absorbs = lifted.transmission.is_some();
                    report(Event::Note(&format!(
                        "depth plane {} at {:.0} pc ({}){}, over {:.1}% of the image",
                        plane.name,
                        plane.distance_pc,
                        plane.basis,
                        match (lifted.lit, absorbs) {
                            (true, true) => ", glowing and dark",
                            (false, true) => ", a dark cloud",
                            _ => "",
                        },
                        100.0 * coverage
                    )));
                    summary.planes.push(PlaneSummary {
                        name: plane.name.clone(),
                        distance_pc: plane.distance_pc,
                        basis: plane.basis.clone(),
                        coverage,
                        absorbs,
                        object: plane.object.clone(),
                    });
                    planes.push(lifted);
                }
                None => report(Event::Warning(&format!(
                    "nothing in the depth plane {} stands apart from the light around it; it \
                     is left out",
                    plane.name
                ))),
            }
        }
        halt()?;
        let cut = CutOptions {
            max_stars: scene_options.max_stars,
            small_stars: scene_options.small_stars,
            ..CutOptions::default()
        };
        let mut scene = Scene::new(
            &starless,
            &stars_light,
            &placed,
            distance,
            unmatched,
            focal_px,
            &cut,
        );
        scene = scene.with_planes(planes, cut.embedded);
        summary.flying_stars = scene.sprites.len();
        summary.galaxies_lifted = galaxies.iter().map(|(name, _, _)| name.clone()).collect();
        let lifted: Vec<overlay::Lifted> = galaxies
            .iter()
            .map(|(name, extent, sprite)| overlay::Lifted {
                name: name.clone(),
                extent: *extent,
                at: (sprite.x, sprite.y),
                distance_pc: sprite.distance_pc,
            })
            .collect();
        Arc::make_mut(&mut scene.sprites).extend(galaxies.into_iter().map(|(_, _, sprite)| sprite));
        let mut counted_dust = None;
        if scene_options.dust {
            // A dark cloud on a plane in front of the nebula blocks the
            // light behind it itself, so the counted dust clears there.
            let clouds: Vec<&crate::plane::Transmission> = scene
                .planes
                .iter()
                .filter(|plane| plane.distance_pc < distance)
                .filter_map(|plane| plane.transmission.as_ref())
                .collect();
            counted_dust = counts
                .map(|dust| Arc::new(dust.with_darkness(&starless, &behind).cleared(&clouds)));
            match (&counted_dust, &given_dust) {
                (Some(_), None) => report(Event::Note(&format!(
                    "dust mapped from {} stars behind it and the nebula's dark places",
                    behind.len()
                ))),
                (None, None) => report(Event::Note(
                    "too few stars to map the dust; nothing dims behind it",
                )),
                (_, Some(_)) => {}
            }
            scene.dust = given_dust.or_else(|| counted_dust.clone());
            describe_dust(&scene, &mut summary, report);
        }
        let behind_clouds = scene
            .sprites
            .iter()
            .filter(|sprite| {
                scene.planes.iter().any(|plane| {
                    plane.transmission.as_ref().is_some_and(|transmission| {
                        !sprite.lifted
                            && sprite.distance_pc > plane.distance_pc
                            && transmission.is_opaque(sprite.x, sprite.y)
                    })
                })
            })
            .count();
        if behind_clouds > 0 {
            report(Event::Note(&format!(
                "{behind_clouds} star{} seen where a dark cloud's plane is opaque fl{} in front \
                 of it",
                if behind_clouds == 1 { "" } else { "s" },
                if behind_clouds == 1 { "ies" } else { "y" },
            )));
        }
        if scene_options.max_stars.is_some() {
            report(Event::Note(&format!(
                "the {} brightest stars fly; the rest {}",
                summary.flying_stars,
                match scene_options.small_stars {
                    SmallStars::Drop => "are dropped",
                    SmallStars::Field => "stay on the star field",
                }
            )));
        }
        halt()?;
        let prepared = Arc::new(Prepared {
            scene,
            counted_dust,
            wcs: wcs.clone(),
            source_wcs,
            crop,
            source_size,
            image_size: (width, height),
            lifted,
            objects: scene_options.objects.clone(),
            summary,
            catalog_marks: OnceLock::new(),
        });
        Self::film(prepared, video, report, stop)
    }

    /// The same scene filmed anew with `video`: another camera move, frame
    /// size, length, frame rate, quality or labels. Nothing of the scene is
    /// prepared again, and this video is left as it was, free to keep
    /// drawing frames while the new one is fitted and after. The catalogued
    /// objects' labels are read once for every video of the scene. `stop`
    /// is asked through the fit; when it answers true the result is
    /// [`Error::Stopped`].
    pub fn reconfigure(
        &self,
        video: &VideoOptions,
        report: &mut dyn FnMut(Event),
        stop: &dyn Fn() -> bool,
    ) -> Result<Self, Error> {
        Self::film(Arc::clone(&self.prepared), video, report, stop)
    }

    /// The same scene and camera with the dust `map` draws, or the dust
    /// mapped from the star counts again if `None`: for trying a map
    /// painted over this one. Only the dust changes; the planes and sprites
    /// are shared, and this video is left as it was. A map gives dust even
    /// to a video prepared without it; `None` then leaves it without.
    /// Maps span the original source image, even when only a crop is filmed.
    pub fn with_dust_map(
        &self,
        map: Option<&DustMap>,
        report: &mut dyn FnMut(Event),
        stop: &dyn Fn() -> bool,
    ) -> Result<Self, Error> {
        let prepared = &self.prepared;
        let dust = match map {
            Some(map) => Some(Arc::new(
                Dust::from_source_map(map, prepared.source_size, prepared.crop)
                    .map_err(Error::Invalid)?,
            )),
            None => prepared.counted_dust.clone(),
        };
        let mut scene = prepared.scene.clone();
        scene.dust = dust;
        let mut summary = prepared.summary.clone();
        summary.dust_transmission = None;
        summary.dust_given = false;
        summary.stars_in_front_of_dust = 0;
        describe_dust(&scene, &mut summary, report);
        let catalog_marks = OnceLock::new();
        if let Some(marks) = prepared.catalog_marks.get() {
            let _ = catalog_marks.set(Arc::clone(marks));
        }
        let prepared = Arc::new(Prepared {
            scene,
            counted_dust: prepared.counted_dust.clone(),
            wcs: prepared.wcs.clone(),
            source_wcs: prepared.source_wcs.clone(),
            crop: prepared.crop,
            source_size: prepared.source_size,
            image_size: prepared.image_size,
            lifted: prepared.lifted.clone(),
            objects: prepared.objects.clone(),
            summary,
            catalog_marks,
        });
        Self::film(prepared, &self.video, report, stop)
    }

    /// The dust the video dims and brings stars forward by, if any.
    pub fn dust(&self) -> Option<&Dust> {
        self.prepared.scene.dust.as_deref()
    }

    /// The dust mapped from the star counts, whether or not a given map
    /// took its place: `None` when the dust is off or too few stars show.
    pub fn counted_dust(&self) -> Option<&Dust> {
        self.prepared.counted_dust.as_deref()
    }

    /// The original image's dimensions, before any crop.
    pub fn source_size(&self) -> (usize, usize) {
        self.prepared.source_size
    }

    /// Dust as a map over the original source image, at the requested size.
    /// Set `counted` to read the star-count estimate rather than the dust in
    /// use. Pixels outside the filmed crop are clear (no estimate exists).
    /// This map can be painted and passed back to [`Self::with_dust_map`].
    pub fn source_dust_map(&self, counted: bool, width: usize, height: usize) -> Option<DustMap> {
        let dust = if counted {
            self.counted_dust()
        } else {
            self.dust()
        }?;
        Some(dust.to_source_map(width, height, self.prepared.source_size, self.prepared.crop))
    }

    /// Plan a tour of the catalogued objects in the image as `auto` asks,
    /// for frames of `frame` output pixels: [`plan_tour`], each stop framing
    /// its target at the depth of the plane this scene shows it on.
    pub fn plan_tour(&self, auto: &AutoTour, frame: (usize, usize)) -> Result<PlannedTour, Error> {
        let (width, height) = self.prepared.image_size;
        plan_tour_in(
            &self.prepared.wcs,
            (width as u32, height as u32),
            self.prepared.objects.as_deref(),
            frame,
            auto,
            Some(&self.prepared.scene),
        )
    }

    /// Film `prepared` as `video` asks: plan any automatic tour, fit the
    /// camera to the scene and plan the labels.
    fn film(
        prepared: Arc<Prepared>,
        video: &VideoOptions,
        report: &mut dyn FnMut(Event),
        stop: &dyn Fn() -> bool,
    ) -> Result<Self, Error> {
        let video = &video.sized_for(prepared.image_size);
        video.check()?;
        let (width, height) = prepared.image_size;
        let video = match &video.auto_tour {
            Some(auto) if video.tour.is_empty() => plan_video(
                video,
                auto,
                &prepared.wcs,
                (width as u32, height as u32),
                prepared.objects.as_deref(),
                Some(&prepared.scene),
                report,
            )?,
            _ => video.clone(),
        };
        let scene = &prepared.scene;
        let centre = ((width as f64 - 1.0) / 2.0, (height as f64 - 1.0) / 2.0);
        check_places(&video, (width, height))?;
        let focus = video.focus.unwrap_or(centre);
        let (sin, cos) = video.truck_angle_deg.to_radians().sin_cos();
        let asked = Shot {
            focus,
            dolly: video.dolly,
            truck: (video.truck * cos, -video.truck * sin),
            start: video.start,
            framing: video.framing,
            pan: video.pan,
            rotation: (
                video.rotate_deg.0.to_radians(),
                video.rotate_deg.1.to_radians(),
            ),
            zoom: video.zoom,
            zoom_end: video.zoom_end,
            width: video.size.0,
            height: video.size.1,
            frames: video.frames(),
            easing: video.easing,
            quality: video.quality,
            growth_limit: video.growth_limit,
            fade_from: video.fade_from,
            tour: video
                .stops()
                .iter()
                .map(|stop| Stop {
                    focus: stop.focus.unwrap_or(centre),
                    dolly: stop.dolly,
                    zoom: stop.zoom,
                    rotation: stop.rotate_deg.to_radians(),
                    pan: stop.pan,
                    travel: stop.travel,
                    hold: stop.hold,
                    spin: stop.spin_deg.to_radians(),
                    push: stop.push,
                    // A stop on a depth plane centres its focus point there,
                    // at the depth it gives or the image shows there; the
                    // whole view centres the image.
                    depth: match (stop.depth_pc, stop.focus) {
                        (Some(depth), _) => depth / scene.background_distance_pc,
                        (None, Some((x, y))) => scene.depth_at(x, y) / scene.background_distance_pc,
                        (None, None) => 1.0,
                    },
                })
                .collect(),
            glide: video.tour_glide,
            looped: video.tour_loop,
            ..Shot::default()
        };
        // A stop framed at a depth the image does not show at its point,
        // such as one from a tour planned with planes since changed, would
        // frame its point off the centre of its view.
        for (index, stop) in video.stops().iter().enumerate() {
            let (Some(depth), Some((x, y))) = (stop.depth_pc, stop.focus) else {
                continue;
            };
            let shown = scene.depth_at(x, y);
            if (depth - shown).abs() > AUTO_PLANE_DEPTH * shown {
                let title = stop
                    .title
                    .as_ref()
                    .map_or(String::new(), |title| format!(" ({title})"));
                report(Event::Warning(&format!(
                    "stop {}{title} is framed at {depth:.0} pc, but the image there lies at \
                     {shown:.0} pc: drop its depth, or plan the tour again",
                    index + 1
                )));
            }
        }
        if !asked.tour.is_empty() {
            report(Event::Note(&format!(
                "a tour of {} stops over {:.1} s",
                asked.tour.len(),
                video.seconds()
            )));
        }
        let (shot, share) = asked.fitted_until(scene, stop).ok_or(Error::Stopped)?;
        let fit = FitSummary {
            zoom: (video.zoom, shot.zoom),
            pan: (video.pan, shot.pan),
            lead: (1.0, shot.lead),
            truck: (video.truck, video.truck * share),
            stops: asked
                .tour
                .iter()
                .zip(&shot.tour)
                .map(|(asked, fitted)| StopFit {
                    zoom: (asked.zoom, fitted.zoom),
                    pan: (asked.pan, fitted.pan),
                    dolly: (asked.dolly, fitted.dolly),
                })
                .collect(),
            inside: shot.stays_inside(scene),
        };
        report_fit(&fit, &shot, report);
        if stop() {
            return Err(Error::Stopped);
        }

        // The labels: the caller's own, the catalogued objects (read once
        // for every video of the scene), and the watermark.
        let mut summary = prepared.summary.clone();
        let mut marks = overlay::custom_marks(&video.labels, Rgb(video.label_color), scene);
        // Without the catalogue the video goes on unlabelled, as it does
        // without galaxies or object distances, rather than throwing away
        // the preparation; a later video tries the catalogue again.
        if video.overlay {
            match prepared.catalog_marks() {
                Ok(catalogued) => {
                    summary.labelled_objects = catalogued.len();
                    report(Event::Note(&format!(
                        "{} catalogued objects in the field to label",
                        catalogued.len()
                    )));
                    marks.extend(catalogued.iter().cloned());
                }
                Err(error) => report(Event::Warning(&format!(
                    "no catalogued objects labelled: {error}"
                ))),
            }
        }
        let stops = video.stops();
        let titles: Vec<Option<String>> = stops
            .iter()
            .map(|stop| stop.title.clone().filter(|_| video.tour_titles))
            .collect();
        let decor = decor(&video, &stops, report)?;
        let overlay = if marks.is_empty() && titles.iter().all(Option::is_none) && decor.is_empty()
        {
            None
        } else {
            let mut overlay = Overlay::new(
                marks,
                titles,
                decor,
                video.overlay_density,
                prepared.wcs.scale_arcsec_per_px(),
                video.size,
            )?;
            let missing = overlay.missing_glyphs();
            if !missing.is_empty() {
                report(Event::Warning(&format!(
                    "the labels' typeface (Inter, Latin and Greek letters) has no {}; they are \
                     left blank",
                    missing
                        .iter()
                        .map(|c| format!("\"{c}\""))
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            }
            if !overlay.plan(&shot, scene, video.fps as f64, stop) {
                return Err(Error::Stopped);
            }
            Some(overlay)
        };
        if stop() {
            return Err(Error::Stopped);
        }
        summary.fit = fit;
        Ok(Self {
            prepared,
            video,
            shot,
            overlay,
            summary,
        })
    }

    /// The number of frames.
    pub fn frames(&self) -> usize {
        self.shot.frames
    }

    pub fn fps(&self) -> u32 {
        self.video.fps
    }

    /// The frame size, pixels.
    pub fn size(&self) -> (usize, usize) {
        (self.shot.width, self.shot.height)
    }

    /// The size of the image filmed, pixels, after any crop.
    pub fn image_size(&self) -> (usize, usize) {
        self.prepared.image_size
    }

    /// The camera move as fitted to the image.
    pub fn shot(&self) -> &Shot {
        &self.shot
    }

    /// The scene cut into depths.
    pub fn scene(&self) -> &Scene {
        &self.prepared.scene
    }

    /// The plate solution of the image filmed: with a crop, the crop's,
    /// its pixels counting from the crop's corner.
    pub fn wcs(&self) -> &Wcs {
        &self.prepared.wcs
    }

    /// The plate solution of the image as given, before any crop: the one
    /// to pass back to [`Self::prepare`] with the same images.
    pub fn source_wcs(&self) -> &Wcs {
        &self.prepared.source_wcs
    }

    /// The part of the image filmed, (left, top, width, height) in the
    /// image as given, if cropped.
    pub fn crop(&self) -> Option<(usize, usize, usize, usize)> {
        self.prepared.crop
    }

    /// The video options this video was filmed with, an automatic tour as
    /// planned.
    pub fn video_options(&self) -> &VideoOptions {
        &self.video
    }

    pub fn summary(&self) -> &Summary {
        &self.summary
    }

    /// Settings for an encoder of this video, with a bitrate of half a bit
    /// per pixel per frame.
    pub fn video_settings(&self) -> VideoSettings {
        let (width, height) = self.size();
        let fps = self.fps();
        VideoSettings {
            width: width as u32,
            height: height as u32,
            fps,
            bitrate: (width * height * fps as usize / 2).min(u32::MAX as usize) as u32,
        }
    }

    /// Frame `index`, with its labels, as display RGB.
    pub fn frame(&self, index: usize) -> RgbImage {
        let scene = self.scene();
        let mut image = self.shot.render(scene, index).to_display_rgb8();
        if let Some(overlay) = &self.overlay {
            overlay.draw(index, &self.shot.view(scene, index), &mut image);
        }
        image
    }

    /// Render every frame in turn into `sink` and finish it. `stop` is
    /// asked before each frame; when it answers true the sink is dropped
    /// unfinished and [`Error::Stopped`] returned.
    pub fn render(
        &self,
        mut sink: Box<dyn FrameSink + '_>,
        report: &mut dyn FnMut(Event),
        stop: &dyn Fn() -> bool,
    ) -> Result<(), Error> {
        let total = self.frames();
        for index in 0..total {
            if stop() {
                return Err(Error::Stopped);
            }
            sink.push(&self.frame(index))?;
            report(Event::Frame {
                done: index + 1,
                total,
            });
        }
        sink.finish()?;
        Ok(())
    }
}

/// Tell the caller how the fit changed the framing asked for.
fn report_fit(fit: &FitSummary, shot: &Shot, report: &mut dyn FnMut(Event)) {
    if !fit.inside {
        report(Event::Warning(
            "the camera could not be kept inside the image: some frames show past its edge; \
             move the focus or stops in from the edge, or ask for less turn and pan",
        ));
    } else if fit.zoom.1 > fit.zoom.0.max(1.0) * 1.001 {
        report(Event::Note(&format!(
            "first frame zoomed in to {:.2} so every {} stays inside the image",
            fit.zoom.1,
            if shot.tour.is_empty() {
                "frame"
            } else {
                "view of the tour"
            }
        )));
    }
    if shot.tour.is_empty() {
        if fit.pan.1 < fit.pan.0 {
            report(Event::Note(&format!(
                "pan reduced to {:.2} so the far stars stay inside the image",
                fit.pan.1
            )));
        }
        if fit.lead.1 < 1.0 {
            report(Event::Note(&format!(
                "sideways travel toward the focus point comes later (lead {:.2}) so the far \
                 stars stay inside the image",
                fit.lead.1
            )));
        }
        if fit.truck.1 < fit.truck.0 {
            report(Event::Note(&format!(
                "truck reduced to {:.4} of the distance so the far stars stay inside the image",
                fit.truck.1
            )));
        }
        return;
    }
    // A stop flown less far zooms in to keep its plane's size; the rest of
    // its zoom keeps it inside the image.
    let kept_size = |index: usize, stop: &StopFit| {
        let depth = shot.tour[index].depth;
        let growth = |dolly: f64| depth / (depth - dolly);
        stop.zoom.0 * growth(stop.dolly.0) / growth(stop.dolly.1)
    };
    let flattened: Vec<String> = fit
        .stops
        .iter()
        .enumerate()
        .filter(|(_, stop)| stop.dolly.1 < stop.dolly.0 - 1e-9)
        .map(|(index, stop)| {
            format!(
                "stop {} {:.2} of the way rather than {:.2}",
                index + 1,
                stop.dolly.1,
                stop.dolly.0
            )
        })
        .collect();
    if !flattened.is_empty() {
        report(Event::Note(&format!(
            "flew less far in, zoomed to keep each target's size, so less of the smooth fill \
             behind the depth planes shows: {}",
            flattened.join(", ")
        )));
    }
    let zoomed: Vec<String> = fit
        .stops
        .iter()
        .enumerate()
        .filter(|&(index, stop)| stop.zoom.1 > kept_size(index, stop) * 1.001)
        .map(|(index, stop)| {
            format!(
                "stop {} by {:.2}",
                index + 1,
                stop.zoom.1 / kept_size(index, stop)
            )
        })
        .collect();
    if !zoomed.is_empty() && fit.inside {
        report(Event::Note(&format!(
            "zoomed in so every view stays inside the image: {}",
            zoomed.join(", ")
        )));
    }
    if let Some(stop) = fit.stops.iter().find(|stop| stop.pan.1 < stop.pan.0) {
        report(Event::Note(&format!(
            "each stop's pan reduced to {:.2} of what it asked so the far stars stay inside \
             the image",
            stop.pan.1 / stop.pan.0
        )));
    }
}

/// A tour a planner chose: its stops, each with the name of the target it
/// visits, and the target most worth a visit, at whose place the nebula's
/// distance is best taken.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedTour {
    pub stops: Vec<PlannedStop>,
    pub focus: (f64, f64),
    pub focus_name: String,
}

impl PlannedTour {
    /// The stops alone, for [`VideoOptions::tour`].
    pub fn tour(&self) -> Vec<TourStop> {
        self.stops
            .iter()
            .map(|planned| planned.stop.clone())
            .collect()
    }
}

/// The plate solution and size of the part of a `dimensions`-pixel image
/// that `crop` (left, top, width, height, image pixels) keeps, as
/// [`SceneOptions::crop`] cuts it: for planning a tour of a crop before
/// [`prepare`] reads the images.
pub fn crop_wcs(
    wcs: &Wcs,
    (image_width, image_height): (u32, u32),
    (left, top, width, height): (usize, usize, usize, usize),
) -> Result<(Wcs, (u32, u32)), Error> {
    let (image_width, image_height) = (image_width as usize, image_height as usize);
    if width < 16
        || height < 16
        || left
            .checked_add(width)
            .is_none_or(|right| right > image_width)
        || top
            .checked_add(height)
            .is_none_or(|bottom| bottom > image_height)
    {
        return Err(Error::Invalid(format!(
            "the crop {width}x{height} at ({left}, {top}) must be at least 16 pixels each way and \
             lie inside the {image_width}x{image_height} image"
        )));
    }
    // SIP terms count from the reference pixel, which moves with the crop.
    let wcs = Wcs {
        crpix: (wcs.crpix.0 - left as f64, wcs.crpix.1 - top as f64),
        ..wcs.clone()
    };
    Ok((wcs, (width as u32, height as u32)))
}

/// The images, plate solution and options of a video cut to `crop` (left,
/// top, width, height, image pixels), with the crop then cleared.
fn crop_inputs(
    starless: &Rgb32FImage,
    stars: &Rgb32FImage,
    wcs: &Wcs,
    options: &ParallaxOptions,
    crop: (usize, usize, usize, usize),
) -> Result<(Rgb32FImage, Rgb32FImage, Wcs, ParallaxOptions), Error> {
    let (wcs, _) = crop_wcs(wcs, starless.dimensions(), crop)?;
    let (left, top, width, height) = crop;
    let (image_width, image_height) = (starless.width() as usize, starless.height() as usize);
    let cut = |image: &Rgb32FImage| {
        image::imageops::crop_imm(image, left as u32, top as u32, width as u32, height as u32)
            .to_image()
    };
    // A grid with the image's shape at its own size, cut to the same part.
    let cut_grid = |grid_width: usize, grid_height: usize, values: &[f32]| {
        let along = |at: usize, size: usize, grid: usize| {
            ((at as f64 * grid as f64 / size as f64).round() as usize).min(grid)
        };
        let (x0, x1) = (
            along(left, image_width, grid_width),
            along(left + width, image_width, grid_width),
        );
        let (y0, y1) = (
            along(top, image_height, grid_height),
            along(top + height, image_height, grid_height),
        );
        let (x1, y1) = (
            x1.max(x0 + 1).min(grid_width),
            y1.max(y0 + 1).min(grid_height),
        );
        let x0 = x0.min(x1 - 1);
        let y0 = y0.min(y1 - 1);
        let kept = (y0..y1)
            .flat_map(|row| {
                values[row * grid_width + x0..row * grid_width + x1]
                    .iter()
                    .copied()
            })
            .collect::<Vec<f32>>();
        (x1 - x0, y1 - y0, kept)
    };
    let mut options = options.clone();
    options.scene.crop = None;
    // Dust stays source-relative and is sampled directly over the crop in
    // Dust::from_source_map, including maps too coarse for integer cropping.
    for plane in &mut options.scene.planes {
        let mask = &plane.mask;
        let (width, height, coverage) = cut_grid(mask.width, mask.height, &mask.coverage);
        plane.mask = crate::plane::PlaneMask {
            width,
            height,
            coverage,
        };
    }
    Ok((cut(starless), cut(stars), wcs, options))
}

/// Plan a tour of the catalogued objects in a `width` × `height` image
/// whose sky `wcs` gives, as `auto` asks, for frames of `frame` output
/// pixels, from the object catalog at `objects` or the standard places.
/// The plan is the caller's to edit, dropping, moving or changing stops,
/// before passing its stops as [`VideoOptions::tour`] and its `focus` as
/// [`SceneOptions::distance_focus`].
pub fn plan_tour(
    wcs: &Wcs,
    dimensions: (u32, u32),
    objects: Option<&Path>,
    frame: (usize, usize),
    auto: &AutoTour,
) -> Result<PlannedTour, Error> {
    plan_tour_in(wcs, dimensions, objects, frame, auto, None)
}

/// [`plan_tour`], each stop framing its target at the depth of the plane
/// `scene` shows it on, when given.
fn plan_tour_in(
    wcs: &Wcs,
    (width, height): (u32, u32),
    objects: Option<&Path>,
    frame: (usize, usize),
    auto: &AutoTour,
    scene: Option<&Scene>,
) -> Result<PlannedTour, Error> {
    auto.check().map_err(Error::Invalid)?;
    check_frame(frame)?;
    let path = seiza::data_paths::objects(objects).map_err(|error| {
        Error::Invalid(format!(
            "planning a tour needs the object catalog (seiza setup): {error}"
        ))
    })?;
    let placed = open_objects(&path)?
        .objects_in_footprint(wcs, (width, height))
        .map_err(|error| Error::Invalid(error.to_string()))?;
    let size = (width as usize, height as usize);
    let mut targets = tour::targets(&placed, size, frame, auto.targets, auto.framing);
    if let Some(scene) = scene {
        for target in &mut targets {
            let depth = scene.depth_at(target.x, target.y);
            target.depth = depth / scene.background_distance_pc;
            target.distance_pc = (depth != scene.background_distance_pc).then_some(depth);
        }
    }
    let Some(best) = targets.first() else {
        return Err(Error::Invalid(
            "no catalogued objects in the field to tour: none has a view of its own".into(),
        ));
    };
    Ok(PlannedTour {
        stops: tour::plan(
            &targets,
            size,
            frame,
            auto,
            scene.map_or(1.0, |scene| {
                scene.leftover_distance_pc.max(scene.far_distance_pc) / scene.background_distance_pc
            }),
        ),
        focus: (best.x, best.y),
        focus_name: best.name.clone(),
    })
}

/// `tour` ending where it begins, for a video that loops: a last stop at
/// the first's view, unless the last is already there, turned back to the
/// first's frame by the nearest whole turns, spins included. An added last
/// stop comes the tour's mean travel after the one before and holds three
/// times as long as the first, at least three seconds: the final drift.
fn close_loop(tour: &[TourStop]) -> Vec<TourStop> {
    let mut closed = tour.to_vec();
    let Some(first) = tour.first() else {
        return closed;
    };
    let last = tour.last().expect("a first stop is a last");
    let home = tour.len() >= 2
        && last.focus == first.focus
        && last.dolly == first.dolly
        && last.zoom == first.zoom
        && last.pan == first.pan;
    if !home {
        let travels: Vec<f64> = tour.iter().skip(1).map(|stop| stop.travel).collect();
        let travel = if travels.is_empty() {
            TourStop::default().travel
        } else {
            travels.iter().sum::<f64>() / travels.len() as f64
        };
        closed.push(TourStop {
            travel,
            hold: (first.hold * 3.0).max(3.0),
            spin_deg: 0.0,
            push: 0.0,
            ..first.clone()
        });
    }
    // The turn the tour ends on, spins included, comes back to the first
    // stop's by whole turns: as few as may be from the turn the last stop
    // asked, or from the stop before an added one.
    let spun: f64 = closed
        .iter()
        .filter(|stop| stop.hold > 0.0)
        .map(|stop| stop.spin_deg)
        .sum();
    let count = closed.len();
    let near = if home {
        closed[count - 1].rotate_deg
    } else {
        closed[count - 2].rotate_deg
    };
    let back = first.rotate_deg - spun;
    let closing = &mut closed[count - 1];
    closing.rotate_deg = back + ((near - back) / 360.0).round() * 360.0;
    closing.push = 0.0;
    closed
}

/// Whether the camera's focus and every tour stop's lie inside a `width` ×
/// `height` image.
fn check_places(video: &VideoOptions, (width, height): (usize, usize)) -> Result<(), Error> {
    let inside =
        |(x, y): (f64, f64)| (0.0..width as f64).contains(&x) && (0.0..height as f64).contains(&y);
    if let Some(focus) = video.focus.filter(|&focus| !inside(focus)) {
        return Err(Error::Invalid(format!(
            "the focus point {focus:?} is outside the {width}x{height} image"
        )));
    }
    for (index, stop) in video.tour.iter().enumerate() {
        if let Some(focus) = stop.focus.filter(|&focus| !inside(focus)) {
            return Err(Error::Invalid(format!(
                "tour stop {}'s place {focus:?} is outside the {width}x{height} image",
                index + 1
            )));
        }
    }
    Ok(())
}

/// `value` to `places` decimals, trailing zeros dropped, or to more where
/// rounding would carry a value below 1 up to 1, past a dolly's or push's
/// limit.
fn decimal(value: f64, places: usize) -> String {
    let mut places = places.max(1);
    loop {
        let text = format!("{value:.places$}");
        let text = text.trim_end_matches('0').trim_end_matches('.').to_string();
        let carried = value < 1.0 && text.parse::<f64>().is_ok_and(|read| read >= 1.0);
        if !carried || places >= 12 {
            return text;
        }
        places += 1;
    }
}

/// A stop written as [`parse_stop`] reads it: the focus to a hundredth of a
/// pixel, the rest to three decimals.
pub fn format_stop(stop: &TourStop) -> String {
    let mut text = match stop.focus {
        Some((x, y)) => format!("{},{}", decimal(x, 2), decimal(y, 2)),
        None => "whole".to_string(),
    };
    let defaults = TourStop::default();
    for (key, value, default) in [
        ("dolly", stop.dolly, defaults.dolly),
        ("zoom", stop.zoom, defaults.zoom),
        ("rotate", stop.rotate_deg, defaults.rotate_deg),
        ("pan", stop.pan, defaults.pan),
        ("travel", stop.travel, f64::NAN),
        ("hold", stop.hold, defaults.hold),
        ("spin", stop.spin_deg, defaults.spin_deg),
        ("push", stop.push, defaults.push),
    ] {
        if value != default {
            text.push_str(&format!(" {key}={}", decimal(value, 3)));
        }
    }
    if let Some(depth) = stop.depth_pc {
        text.push_str(&format!(" depth={}", decimal(depth, 1)));
    }
    let quoted = |text: &str| text.replace('\\', "\\\\").replace('"', "\\\"");
    if let Some(title) = stop.title.as_ref().filter(|title| !title.is_empty()) {
        text.push_str(&format!(" title=\"{}\"", quoted(title)));
    }
    if let Some(card) = &stop.card {
        text.push_str(&format!(" card=\"{}\"", quoted(&card.to_string_lossy())));
    }
    text
}

/// The watermark and cards `video` asks for, its images read and scaled to
/// its frame, each tour stop's card among `stops`. An image read for
/// several cards is read once. A card without transparency, or of another
/// shape than the frame, is drawn but warned of.
fn decor(
    video: &VideoOptions,
    stops: &[TourStop],
    report: &mut dyn FnMut(Event),
) -> Result<Decor, Error> {
    let (width, height) = video.size;
    let read = |path: &Path, what: &str| {
        Artwork::open(path).map_err(|error| Error::Invalid(format!("the {what}: {error}")))
    };
    let watermark_image = match &video.watermark_image {
        Some(path) => {
            let artwork = read(path, "watermark image")?;
            // Its height a share of the frame's shorter side, its width at
            // most a third of the frame's.
            let tall = video.watermark_size * width.min(height) as f64;
            let wide = (tall * artwork.aspect()).min(width as f64 / 3.0);
            let tall = wide / artwork.aspect();
            Some(Picture::new(
                &artwork,
                (wide.round().max(1.0) as u32, tall.round().max(1.0) as u32),
            ))
        }
        None => None,
    };
    let mut read_cards: Vec<(PathBuf, Placed)> = Vec::new();
    let mut card = |path: &Path, what: &str, report: &mut dyn FnMut(Event)| {
        if let Some((_, placed)) = read_cards.iter().find(|(read, _)| read == path) {
            return Ok(placed.clone());
        }
        let artwork = read(path, what)?;
        if !artwork.transparent {
            report(Event::Warning(&format!(
                "the {what} ({}) has no transparency, so it hides the whole frame while it shows",
                path.display()
            )));
        }
        let frame = width as f64 / height as f64;
        if (artwork.aspect() / frame - 1.0).abs() > 0.02 {
            report(Event::Warning(&format!(
                "the {what} ({}) is {}×{}, another shape than the {width}×{height} frame: it is \
                 scaled to fit inside it and centred; make it the frame's shape to fill the frame",
                path.display(),
                artwork.image.width(),
                artwork.image.height()
            )));
        }
        let (size, at) = crate::picture::fitted(artwork.aspect(), video.size);
        let placed = Placed {
            picture: Arc::new(Picture::new(&artwork, size)),
            at,
        };
        read_cards.push((path.to_path_buf(), placed.clone()));
        Ok::<_, Error>(placed)
    };
    let intro = match &video.intro_card {
        Some(path) => Some(card(path, "opening card", report)?),
        None => None,
    };
    let outro = match &video.outro_card {
        Some(path) => Some(card(path, "closing card", report)?),
        None => None,
    };
    let cards = stops
        .iter()
        .enumerate()
        .map(|(index, stop)| match &stop.card {
            Some(path) => card(path, &format!("card of stop {}", index + 1), report).map(Some),
            None => Ok(None),
        })
        .collect::<Result<Vec<_>, Error>>()?;
    Ok(Decor {
        watermark: video.watermark.clone(),
        watermark_image,
        corner: video.watermark_corner,
        opacity: video.watermark_opacity as f32,
        intro,
        outro,
        card_seconds: video.card_seconds,
        cards,
    })
}

/// `video` with a tour of the catalogued objects in a `dimensions` image
/// planned as `auto` asks, each target framed at the depth of its plane in
/// `scene`.
fn plan_video(
    video: &VideoOptions,
    auto: &AutoTour,
    wcs: &Wcs,
    dimensions: (u32, u32),
    objects: Option<&Path>,
    scene: Option<&Scene>,
    report: &mut dyn FnMut(Event),
) -> Result<VideoOptions, Error> {
    let auto = AutoTour {
        framing: video.framing,
        ..*auto
    };
    let plan = plan_tour_in(wcs, dimensions, objects, video.size, &auto, scene)?;
    let planned = VideoOptions {
        tour: plan.tour(),
        auto_tour: None,
        ..video.clone()
    };
    let names: Vec<&str> = plan
        .stops
        .iter()
        .filter_map(|stop| stop.name.as_deref())
        .collect();
    report(Event::Note(&format!(
        "an automatic tour of {} targets over {:.0} s: {}",
        names.len(),
        planned.seconds(),
        names.join(" → ")
    )));
    planned.check()?;
    Ok(planned)
}

/// Whether every value of `image` sits on one of 256 levels, as an 8-bit
/// file's do; a 16-bit image's hardly ever all fall there. Looks at a
/// spread of up to 100,000 values.
fn eight_bit(image: &Rgb32FImage) -> bool {
    let values = image.as_raw();
    let step = (values.len() / 100_000).max(1);
    values
        .iter()
        .step_by(step)
        .all(|&value| ((value * 255.0) - (value * 255.0).round()).abs() < 1e-3)
}

fn open_objects(path: &Path) -> Result<seiza::objects::ObjectCatalog, Error> {
    seiza::objects::ObjectCatalog::open(path).map_err(|error| Error::Catalog {
        path: path.to_path_buf(),
        message: error.to_string(),
    })
}

/// The catalogued object at the focus point, its distance and how that
/// distance was found, or `None` without an object catalog, a distance file
/// or an object there.
fn target_distance(
    options: &SceneOptions,
    wcs: &Wcs,
    (width, height): (usize, usize),
    focus: (f64, f64),
    given: &HashMap<String, f64>,
) -> Result<Option<(String, f64, &'static str)>, Error> {
    let Ok(objects_path) = seiza::data_paths::objects(options.objects.as_deref()) else {
        return Ok(None);
    };
    let Some(distances_path) = seiza::data_paths::object_distances_beside(
        options.object_distances.as_deref(),
        &objects_path,
    )
    .map_err(|error| Error::Invalid(error.to_string()))?
    else {
        return Ok(None);
    };
    let catalog = open_objects(&objects_path)?;
    let distances =
        seiza::catalog::ObjectDistances::open(&distances_path, &catalog).map_err(|error| {
            Error::Catalog {
                path: distances_path.clone(),
                message: error.to_string(),
            }
        })?;
    // Search a twentieth of the image's diagonal past the nearest edge.
    let reach = (width as f64).hypot(height as f64) / 20.0;
    let Some(found) = distances
        .object_at_pixel(&catalog, wcs, (width as u32, height as u32), focus, reach)
        .map_err(|error| Error::Invalid(error.to_string()))?
    else {
        return Ok(None);
    };
    let object = &found.placed.object;
    let name = if object.common_name.is_empty() {
        object.name.clone()
    } else {
        format!("{} ({})", object.name, object.common_name)
    };
    if let Some(distance) = given_distance(given, object) {
        return Ok(Some((name, distance, "given")));
    }
    Ok(found
        .distance()
        .map(|distance| match given_around(given, &distance) {
            Some(around) => (name, around, "given"),
            None => (name, distance.distance_pc, distance.basis.as_str()),
        }))
}

/// The distances `options` gives for catalogued objects, by catalogue ID:
/// every object each given name resolves to. A name the catalogue does not
/// know is an error.
fn given_distances(options: &SceneOptions) -> Result<HashMap<String, f64>, Error> {
    let mut given = HashMap::new();
    if options.distance_overrides.is_empty() {
        return Ok(given);
    }
    let path = seiza::data_paths::objects(options.objects.as_deref()).map_err(|error| {
        Error::Invalid(format!(
            "distances given by object name need the object catalog (seiza setup): {error}"
        ))
    })?;
    let catalog = open_objects(&path)?;
    for each in &options.distance_overrides {
        let found = catalog
            .lookup_name(each.object.trim())
            .map_err(|error| Error::Catalog {
                path: path.clone(),
                message: error.to_string(),
            })?;
        if found.is_empty() {
            return Err(Error::Invalid(format!(
                "no catalogued object is named {}",
                each.object
            )));
        }
        for found in found {
            given.insert(found.object.metadata.id, each.distance_pc);
        }
    }
    Ok(given)
}

/// The distance `given` holds for `object`, under its ID or any of its
/// alternate IDs.
fn given_distance(given: &HashMap<String, f64>, object: &SkyObject) -> Option<f64> {
    std::iter::once(&object.metadata.id)
        .chain(&object.metadata.alternate_ids)
        .find_map(|id| given.get(id).copied())
}

/// The distance `given` holds for the object around one whose catalogued
/// distance, `found`, is that object's: a nebula inside another lies where
/// the other is said to, so the two keep one depth.
fn given_around(given: &HashMap<String, f64>, found: &ObjectDistance) -> Option<f64> {
    match (found.basis, found.via) {
        (DistanceBasis::Contained, Some(via)) => given.get(via).copied(),
        _ => None,
    }
}

/// The name an ID such as `vizier:VII/20:Sh2-171` ends in.
fn id_name(id: &str) -> &str {
    id.rsplit(':').next().unwrap_or(id)
}

/// A depth plane to lift: its name, region, distance and how that was
/// found, and whether it may block light. A glowing nebula found in the
/// catalogue takes only its light: the darkness about it is dust, which
/// stays with its cloud.
struct PlaneToLift {
    name: String,
    region: Region,
    distance_pc: f64,
    basis: String,
    blocks: bool,
    /// The catalogued object a distance override moves it by.
    object: Option<String>,
}

/// How the catalogue found `distance`, in a few words.
fn basis_of(distance: &ObjectDistance) -> String {
    match (distance.basis, distance.via) {
        (DistanceBasis::Contained, Some(via)) => format!("that of {}, around it", id_name(via)),
        (basis, _) => match distance.method.as_str() {
            "unknown" => basis.as_str().to_string(),
            method => format!("{}, {method}", basis.as_str()),
        },
    }
}

/// A catalogued dark nebula in the field.
struct DarkNebula {
    name: String,
    region: Region,
    /// Its distance and how that was found, if the catalogue places it: at
    /// the nebula's own distance when its range takes that in.
    distance: Option<(f64, String)>,
    /// Whether that distance was given, for it or for one around it.
    given: bool,
    /// Whether its ellipse holds the point the nebula's distance was taken
    /// at: it is the nebula, or holds it.
    at_focus: bool,
}

/// The catalogued nebulae in the field, as [`SceneOptions::auto_planes`]
/// chooses them, against a nebula at `distance` whose depth was taken at
/// `focus`: the glowing ones lying at distances of their own, largest
/// first, and every dark one, each with its outermost catalogued outline as
/// its region, or else its ellipse sized by the light of `starless`. A dark
/// nebula lies in a cloud of dust and moves with it, so [`cloud_planes`]
/// places the dark ones. `given` distances take the catalogue's place.
///
/// A nebula that holds the focus, spans most of the image or cannot be
/// cut out of it stays with the nebula, as does one that would cover more
/// than [`AUTO_PLANE_COVERAGE`] of it unless given a distance; and so do
/// the nebulae inside each that take its distance, as lifting them would
/// tear one structure apart.
fn catalogued_planes(
    options: &SceneOptions,
    wcs: &Wcs,
    starless: &LightImage,
    (focus, distance): ((f64, f64), f64),
    given: &HashMap<String, f64>,
    report: &mut dyn FnMut(Event),
) -> Result<(Vec<PlaneToLift>, Vec<DarkNebula>), Error> {
    let (width, height) = (starless.width, starless.height);
    let objects_path = seiza::data_paths::objects(options.objects.as_deref())
        .map_err(|error| Error::Invalid(format!("they need the object catalog: {error}")))?;
    let distances_path = seiza::data_paths::object_distances_beside(
        options.object_distances.as_deref(),
        &objects_path,
    )
    .map_err(|error| Error::Invalid(error.to_string()))?
    .ok_or_else(|| Error::Invalid("they need the object distance file (seiza setup)".into()))?;
    let catalog = open_objects(&objects_path)?;
    let distances =
        seiza::catalog::ObjectDistances::open(&distances_path, &catalog).map_err(|error| {
            Error::Catalog {
                path: distances_path.clone(),
                message: error.to_string(),
            }
        })?;
    let placed = catalog
        .objects_in_footprint(wcs, (width as u32, height as u32))
        .map_err(|error| Error::Invalid(error.to_string()))?;
    let short = width.min(height) as f64;
    let mut kept: Vec<(Extent, Option<f64>)> = Vec::new();
    // The nebulae staying with the nebula, largest first as the catalogue
    // gives them, so before any inside them: each one's ID and name, why
    // it stays, if the notes should say, and the nebulae staying with it.
    let mut staying: Vec<(String, String, Option<String>, Vec<String>)> = Vec::new();
    let (mut planes, mut dark_nebulae) = (Vec::new(), Vec::new());
    for object in placed {
        use seiza::objects::ObjectKind as Kind;
        let dark = object.object.kind == Kind::DarkNebula;
        let nebula = dark
            || matches!(
                object.object.kind,
                Kind::Nebula
                    | Kind::PlanetaryNebula
                    | Kind::HiiRegion
                    | Kind::SupernovaRemnant
                    | Kind::ClusterWithNebula
            );
        // Too small to see move.
        if !nebula || object.semi_major_px < 12.0 {
            continue;
        }
        let mut given_here = false;
        let found = match given_distance(given, &object.object) {
            Some(depth) => {
                given_here = true;
                Some((depth, "given".to_string()))
            }
            None => distances.for_object(&object.object).and_then(|found| {
                if let Some(depth) = given_around(given, &found) {
                    given_here = true;
                    let around = found.via.map_or("", id_name);
                    return Some((depth, format!("given for {around}, around it")));
                }
                // One inside a nebula that stays with the nebula stays too.
                let around = match (found.basis, found.via) {
                    (DistanceBasis::Contained, Some(via)) => {
                        staying.iter_mut().find(|(id, ..)| id == via)
                    }
                    _ => None,
                };
                if let Some((_, around, _, inside)) = around {
                    // A dark one's cloud says where it goes.
                    if !dark {
                        inside.push(object.object.name.clone());
                    }
                    return Some((distance, format!("that of {around}, which stays")));
                }
                // A dark cloud's distance may come from the nearest cloud
                // measured along a sightline beside it.
                let measured = match found.basis {
                    DistanceBasis::Measured | DistanceBasis::Contained => true,
                    DistanceBasis::NearbyCloud => dark,
                    _ => false,
                };
                // A range that takes in the nebula's distance cannot say
                // which lies nearer.
                let overlaps = found.lower_pc.is_some_and(|lower| lower <= distance)
                    && found.upper_pc.is_some_and(|upper| upper >= distance);
                match (measured, overlaps) {
                    (false, _) => None,
                    (true, true) => Some((distance, "its range takes in the nebula's".into())),
                    (true, false) => Some((found.distance_pc, basis_of(&found))),
                }
            }),
        };
        let apart = found
            .as_ref()
            .is_some_and(|(depth, _)| (depth - distance).abs() >= AUTO_PLANE_DEPTH * distance);
        let id = object.object.metadata.id.clone();
        let name = object.object.name.clone();
        // So large it spans most of the image: the scene's backdrop.
        if object.semi_major_px > 0.6 * short {
            let why =
                apart.then(|| format!("{name} spans most of the image and stays with the nebula"));
            staying.push((id, name, why, Vec::new()));
            continue;
        }
        if !dark && !apart {
            continue;
        }
        let (semi_minor, angle) = match object.angle_deg {
            Some(angle) => (object.semi_minor_px, angle),
            None => (object.semi_major_px, 0.0),
        };
        let extent = Extent {
            x: object.x,
            y: object.y,
            semi_major: object.semi_major_px,
            semi_minor: semi_minor.max(1.0),
            angle: angle.to_radians(),
        };
        let at_focus = extent.reach(focus.0, focus.1) <= 1.0;
        // The nebula itself, or a region about it, which would carry it
        // off.
        if at_focus && !dark {
            staying.push((id, name, None, Vec::new()));
            continue;
        }
        // A second listing of one already taken.
        let depth = found.as_ref().map(|(depth, _)| *depth);
        let same_depth = |other: Option<f64>| match (other, depth) {
            (Some(other), Some(depth)) => (other - depth).abs() < AUTO_PLANE_DEPTH * other,
            (None, None) => true,
            _ => false,
        };
        if kept.iter().any(|(other, other_depth)| {
            other.reach(extent.x, extent.y) <= 1.0 && same_depth(*other_depth)
        }) {
            continue;
        }
        let outline = outermost_outline(&catalog, &object.object.metadata.id, wcs);
        let region = if !outline.is_empty() {
            Region::from_outlines(&outline, width, height)
        } else if dark {
            Region::from_dark_extent(starless, &extent)
        } else {
            Region::from_extent(starless, &extent)
        };
        // One that cannot be cut out leaves its light with the nebula's.
        let Some(region) = region else {
            if !dark {
                staying.push((id, name, None, Vec::new()));
            }
            continue;
        };
        kept.push((extent, depth));
        let share = region.area() / (width * height) as f64;
        if !dark && !given_here && share > AUTO_PLANE_COVERAGE {
            let why = format!(
                "{name} would cover {:.0}% of the image and stays with the nebula",
                100.0 * share
            );
            staying.push((id, name, Some(why), Vec::new()));
            continue;
        }
        match (dark, found) {
            (true, distance) => dark_nebulae.push(DarkNebula {
                name,
                region,
                distance,
                given: given_here,
                at_focus,
            }),
            (false, Some((distance_pc, basis))) => planes.push(PlaneToLift {
                object: Some(name.clone()),
                name,
                region,
                distance_pc,
                basis,
                blocks: false,
            }),
            (false, None) => {}
        }
    }
    for (_, name, why, inside) in staying {
        if let Some(why) = why {
            report(Event::Note(&why));
        }
        if !inside.is_empty() {
            let one = inside.len() == 1;
            report(Event::Note(&format!(
                "{} lie{} in {name} and stay{} with it",
                in_words(&inside),
                if one { "s" } else { "" },
                if one { "s" } else { "" },
            )));
        }
    }
    Ok((planes, dark_nebulae))
}

/// `names` as a list in words: "A, B and C".
fn in_words<S: AsRef<str>>(names: &[S]) -> String {
    match names {
        [] => String::new(),
        [one] => one.as_ref().to_string(),
        [rest @ .., last] => {
            let rest: Vec<&str> = rest.iter().map(AsRef::as_ref).collect();
            format!("{} and {}", rest.join(", "), last.as_ref())
        }
    }
}

/// The dust clouds lying in front of a nebula at `distance`, whose depth
/// was taken at `focus`, as planes to lift: each of `clouds` that does not
/// hold the nebula, with the catalogued `dark` nebulae in it, placed at its
/// largest dark nebula's distance, else where the Gaia `stars` (image
/// place and parsecs) seen through it run short; and the dark nebulae that
/// lie in no cloud, each at its own distance. Dark nebulae in the nebula's
/// own cloud stay with it, as does a cloud or dark nebula that would cover
/// more than [`AUTO_PLANE_COVERAGE`] of the `width` × `height` image,
/// unless given a distance. Without `clouds`, every dark nebula stands
/// alone.
fn cloud_planes(
    clouds: Option<&Clouds>,
    dark: Vec<DarkNebula>,
    stars: &[(f64, f64, f64)],
    (focus, distance): ((f64, f64), f64),
    (width, height): (usize, usize),
    report: &mut dyn FnMut(Event),
) -> Vec<PlaneToLift> {
    let in_front = |depth: f64| depth <= (1.0 - AUTO_PLANE_DEPTH) * distance;
    let counted_basis = |found: &CountedDistance| {
        format!(
            "Gaia star counts, {} stars in front, {:.0}-{:.0} pc",
            found.in_front, found.lower_pc, found.upper_pc
        )
    };
    let names = |nebulae: &[DarkNebula]| {
        let names: Vec<&str> = nebulae.iter().map(|nebula| nebula.name.as_str()).collect();
        in_words(&names)
    };
    let share = |region: &Region| region.area() / (width * height) as f64;
    let mut planes = Vec::new();
    let mut alone = Vec::new();
    let mut clear = Vec::new();
    if let Some(clouds) = clouds {
        let mut members: Vec<Vec<DarkNebula>> = (0..clouds.len()).map(|_| Vec::new()).collect();
        for nebula in dark {
            match clouds.holding(&nebula.region) {
                Some(label) => members[label].push(nebula),
                None => alone.push(nebula),
            }
        }
        // The stars seen through each cloud's core, and over clear sky.
        let mut seen: Vec<Vec<f64>> = vec![Vec::new(); clouds.len()];
        for &(x, y, star_distance) in stars {
            if let Some(label) = clouds.core_at(x, y) {
                seen[label].push(star_distance);
            } else if clouds.is_clear(x, y) {
                clear.push(star_distance);
            }
        }
        let clear_area = clouds.clear_area();
        let home = clouds.near(focus, 2);
        let mut labels: Vec<(usize, f64)> = (0..clouds.len())
            .map(|label| (label, clouds.area(label)))
            .collect();
        labels.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut unnamed = 0;
        for (label, _) in labels {
            let mut nebulae = std::mem::take(&mut members[label]);
            nebulae.sort_by(|a, b| b.region.area().total_cmp(&a.region.area()));
            let counted = || {
                let core = clouds.core_area(label);
                counted_distance(&seen[label], &clear, clear_area / core)
            };
            if home == Some(label) || nebulae.iter().any(|nebula| nebula.at_focus) {
                if let Some(found) = counted() {
                    report(Event::Note(&format!(
                        "the Gaia star counts put the nebula's dust at {:.0} pc ({:.0}-{:.0}), \
                         {} stars in front of it",
                        found.distance_pc, found.lower_pc, found.upper_pc, found.in_front
                    )));
                }
                if !nebulae.is_empty() {
                    report(Event::Note(&format!(
                        "{} lie{} in the nebula's own dust and stay{} with it",
                        names(&nebulae),
                        if nebulae.len() == 1 { "s" } else { "" },
                        if nebulae.len() == 1 { "s" } else { "" },
                    )));
                }
                continue;
            }
            let placed = nebulae.iter().find(|nebula| nebula.distance.is_some());
            let given = placed.is_some_and(|nebula| nebula.given);
            let (depth, basis, upper) = match placed.and_then(|nebula| nebula.distance.clone()) {
                Some((depth, basis)) => (depth, basis, depth),
                None => match counted() {
                    Some(found) => (found.distance_pc, counted_basis(&found), found.upper_pc),
                    None => continue,
                },
            };
            if !(in_front(depth) && upper < distance) {
                continue;
            }
            let Some(mut region) = clouds.region(label) else {
                continue;
            };
            for nebula in &nebulae {
                region = region.union(&nebula.region);
            }
            // So much of the image is the scene's backdrop.
            let covers = share(&region);
            if covers > AUTO_PLANE_COVERAGE && !given {
                report(Event::Note(&if nebulae.is_empty() {
                    format!(
                        "a dust cloud that would cover {:.0}% of the image stays with the nebula",
                        100.0 * covers
                    )
                } else {
                    let one = nebulae.len() == 1;
                    format!(
                        "{} lie{} in dust that would cover {:.0}% of the image and stay{} with \
                         the nebula",
                        names(&nebulae),
                        if one { "s" } else { "" },
                        100.0 * covers,
                        if one { "s" } else { "" },
                    )
                }));
                continue;
            }
            let name = match nebulae.first() {
                Some(largest) => largest.name.clone(),
                None => {
                    unnamed += 1;
                    format!("dust cloud {unnamed}")
                }
            };
            if nebulae.len() > 1 {
                report(Event::Note(&format!(
                    "{} lie{} in {name}'s dust and move{} with it",
                    names(&nebulae[1..]),
                    if nebulae.len() == 2 { "s" } else { "" },
                    if nebulae.len() == 2 { "s" } else { "" },
                )));
            }
            planes.push(PlaneToLift {
                object: nebulae.first().map(|largest| largest.name.clone()),
                name,
                region,
                distance_pc: depth,
                basis,
                blocks: true,
            });
        }
    } else {
        alone = dark;
    }
    // A dark nebula in clear sky stands alone.
    for nebula in alone {
        if nebula.at_focus {
            continue;
        }
        let found = nebula.distance.or_else(|| {
            let seen: Vec<f64> = stars
                .iter()
                .filter(|&&(x, y, _)| nebula.region.covers(x, y) >= 0.5)
                .map(|star| star.2)
                .collect();
            let area = clouds.map(|clouds| clouds.clear_area() / nebula.region.area());
            area.and_then(|area| counted_distance(&seen, &clear, area))
                .filter(|found| found.upper_pc < distance)
                .map(|found| (found.distance_pc, counted_basis(&found)))
        });
        if let Some((depth, basis)) = found.filter(|(depth, _)| in_front(*depth)) {
            let covers = share(&nebula.region);
            if covers > AUTO_PLANE_COVERAGE && !nebula.given {
                report(Event::Note(&format!(
                    "{} would cover {:.0}% of the image and stays with the nebula",
                    nebula.name,
                    100.0 * covers
                )));
                continue;
            }
            planes.push(PlaneToLift {
                object: Some(nebula.name.clone()),
                name: nebula.name,
                region: nebula.region,
                distance_pc: depth,
                basis,
                blocks: true,
            });
        }
    }
    planes
}

/// The closed contours of the outermost brightness level catalogued for
/// the object `id`, in image pixels: inner levels lie within it.
fn outermost_outline(
    catalog: &seiza::objects::ObjectCatalog,
    id: &str,
    wcs: &Wcs,
) -> Vec<Vec<(f64, f64)>> {
    if id.is_empty() {
        return Vec::new();
    }
    let Ok(geometries) = catalog.geometries(id) else {
        return Vec::new();
    };
    let outermost = geometries
        .into_iter()
        .filter_map(|geometry| match geometry.data {
            GeometryData::OutlineSet { level, contours } => Some((level, contours)),
            _ => None,
        })
        .min_by(|a, b| a.0.cmp(&b.0));
    let Some((_, contours)) = outermost else {
        return Vec::new();
    };
    contours
        .into_iter()
        .filter(|contour| contour.closed && contour.vertices.len() >= 3)
        .filter_map(|contour| {
            contour
                .vertices
                .iter()
                .map(|&(ra, dec)| wcs.world_to_pixel(ra, dec))
                .collect::<Option<Vec<_>>>()
        })
        .collect()
}

/// Catalogued galaxies in the image large enough to see, largest first,
/// leaving out one at the focus point, which is the target, and each only
/// once: catalogs list some galaxies twice, a little apart.
fn galaxies_in_image(
    options: &SceneOptions,
    wcs: &Wcs,
    (width, height): (usize, usize),
    focus: (f64, f64),
) -> Result<Vec<(String, Extent)>, Error> {
    let Ok(objects_path) = seiza::data_paths::objects(options.objects.as_deref()) else {
        return Ok(Vec::new());
    };
    let catalog = open_objects(&objects_path)?;
    let placed = catalog
        .objects_in_footprint(wcs, (width as u32, height as u32))
        .map_err(|error| Error::Invalid(error.to_string()))?;
    let mut kept: Vec<(String, Extent)> = Vec::new();
    for object in placed {
        if object.object.kind != seiza::objects::ObjectKind::Galaxy || object.semi_major_px < 4.0 {
            continue;
        }
        // An asymmetric extent without a position angle must not be drawn
        // at a guessed orientation, so it becomes a circle.
        let (semi_minor, angle) = match object.angle_deg {
            Some(angle) => (object.semi_minor_px, angle),
            None => (object.semi_major_px, 0.0),
        };
        let extent = Extent {
            x: object.x,
            y: object.y,
            semi_major: object.semi_major_px,
            semi_minor: semi_minor.max(1.0),
            angle: angle.to_radians(),
        };
        if extent.reach(focus.0, focus.1) <= 2.0
            || kept
                .iter()
                .any(|(_, other)| other.reach(extent.x, extent.y) <= 1.5)
        {
            continue;
        }
        kept.push((object.object.name.clone(), extent));
        if kept.len() == 200 {
            break;
        }
    }
    Ok(kept)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameFn;
    use crate::plane::PlaneMask;
    use crate::tour::AutoTour;
    use seiza::catalog::{DistanceStar, StarDistanceCatalogBuilder};

    /// A small solved field: a smooth starless image, a stars image with a
    /// few Gaussian stars, and an offline star distance file placing them.
    fn field(directory: &Path) -> (Rgb32FImage, Rgb32FImage, Wcs, PathBuf) {
        let (width, height) = (320_u32, 240_u32);
        let wcs = Wcs::from_center_scale_rotation((56.75, 24.12), (160.0, 120.0), 2.0, 0.0, false);
        let starless = Rgb32FImage::from_fn(width, height, |x, y| {
            image::Rgb([
                0.1 + 0.2 * x as f32 / width as f32,
                0.15,
                0.1 + 0.1 * y as f32 / height as f32,
            ])
        });
        let places = [
            (60.0, 50.0, 120.0),
            (250.0, 70.0, 900.0),
            (140.0, 180.0, 300.0),
            (200.0, 110.0, 2500.0),
        ];
        let stars = Rgb32FImage::from_fn(width, height, |x, y| {
            let light: f32 = places
                .iter()
                .map(|&(sx, sy, _)| {
                    let r2 = (x as f64 - sx).powi(2) + (y as f64 - sy).powi(2);
                    0.9 * (-r2 / 4.0).exp() as f32
                })
                .sum();
            image::Rgb([light.min(1.0); 3])
        });
        let mut builder = StarDistanceCatalogBuilder::new(18, 2016.0, 16.0, "test");
        for &(x, y, distance) in &places {
            let (ra, dec) = wcs.pixel_to_world(x, y);
            builder.add(DistanceStar {
                ra,
                dec,
                mag: 9.0,
                distance_pc: Some(distance),
                hipparcos: false,
            });
        }
        let path = directory.join("star-distances.bin");
        builder.write_to(&path).unwrap();
        (starless, stars, wcs, path)
    }

    fn options(star_distances: PathBuf, directory: &Path) -> ParallaxOptions {
        ParallaxOptions {
            scene: SceneOptions {
                distance_pc: Some(400.0),
                objects: Some(directory.join("no-objects.bin")),
                star_distances: Some(star_distances),
                online: false,
                ..SceneOptions::default()
            },
            video: VideoOptions {
                size: (160, 120),
                seconds: 1.0,
                fps: 6,
                labels: vec![CustomLabel {
                    x: 100.0,
                    y: 100.0,
                    radius: 20.0,
                    text: "here".into(),
                }],
                watermark: Some(overlay::DEFAULT_WATERMARK.into()),
                ..VideoOptions::default()
            },
        }
    }

    /// A `width` × `height` dust map of one opacity.
    fn even_map(width: usize, height: usize, opacity: f32) -> DustMap {
        DustMap {
            width,
            height,
            opacity: vec![opacity; width * height],
        }
    }

    #[test]
    fn a_given_dust_map_brings_stars_forward_and_another_can_take_its_place() {
        let directory = tempfile::tempdir().unwrap();
        let (starless, stars, wcs, distances) = field(directory.path());
        let mut options = options(distances, directory.path());
        // Opaque everywhere, at a quarter of the image's size.
        options.scene.dust_map = Some(even_map(80, 60, 1.0));
        let mut notes = Vec::new();
        let opaque = Parallax::prepare(
            &starless,
            &stars,
            &wcs,
            &options,
            &mut |event| {
                if let Event::Note(note) = event {
                    notes.push(note.to_string());
                }
            },
            &|| false,
        )
        .unwrap();
        let summary = opaque.summary();
        assert!(summary.dust_given);
        // The two stars behind the nebula at 400 pc, at 900 and 2500 pc.
        assert_eq!(summary.stars_in_front_of_dust, 2, "{notes:?}");
        assert!(
            notes
                .iter()
                .any(|note| note.contains("dust from the given map")),
            "{notes:?}"
        );
        // Four stars are too few to count dust from.
        assert!(opaque.counted_dust().is_none());
        let before = opaque.frame(2);
        // A clear map brings none forward, and the first video is untouched.
        let clear = opaque
            .with_dust_map(Some(&even_map(320, 240, 0.0)), &mut |_| {}, &|| false)
            .unwrap();
        assert!(Arc::ptr_eq(
            &opaque.prepared.scene.background,
            &clear.prepared.scene.background
        ));
        assert_eq!(clear.summary().stars_in_front_of_dust, 0);
        assert!(clear.summary().dust_given);
        assert_ne!(clear.frame(2), before);
        assert_eq!(opaque.frame(2), before);
        assert_eq!(opaque.summary().stars_in_front_of_dust, 2);
        // No map, and none counted: no dust at all.
        let none = opaque.with_dust_map(None, &mut |_| {}, &|| false).unwrap();
        assert!(none.dust().is_none());
        assert_eq!(none.summary().dust_transmission, None);
        // The map in use, written out, is the one given.
        let map = clear.dust().unwrap().to_map(32, 24);
        assert!(map.opacity.iter().all(|&opacity| opacity == 0.0));
    }

    #[test]
    fn cropped_dust_keeps_source_coordinates_through_prepare_paint_and_reload() {
        let directory = tempfile::tempdir().unwrap();
        let (starless, stars, wcs, distances) = field(directory.path());
        let mut options = options(distances, directory.path());
        options.scene.crop = Some((40, 30, 160, 160));
        options.video.focus = None;
        options.video.tour.clear();
        // A coarse, source-shaped map, half clear. A crop of this grid would
        // be 4x5, the wrong shape for a square scene; sample source coordinates.
        let map = DustMap {
            width: 8,
            height: 6,
            opacity: (0..48).map(|i| if i % 8 < 4 { 0.0 } else { 1.0 }).collect(),
        };
        options.scene.dust_map = Some(map.clone());
        let video =
            Parallax::prepare(&starless, &stars, &wcs, &options, &mut |_| {}, &|| false).unwrap();
        let swapped = video
            .with_dust_map(Some(&map), &mut |_| {}, &|| false)
            .unwrap();
        assert_eq!(video.frame(2), swapped.frame(2));
        let dust = swapped.dust().unwrap();
        assert!(!dust.is_opaque(40.0, 80.0));
        assert!(dust.is_opaque(155.0, 80.0));
        assert_eq!(video.source_size(), (320, 240));
        let read = video.source_dust_map(false, 320, 240).unwrap();
        assert_eq!(read.opacity[0], 0.0); // unfilmed, not a stretched crop
        assert_eq!(read.opacity[100 * 320 + 80], 0.0);
        assert!(read.opacity[100 * 320 + 195] > 0.99);
        let repainted = video
            .with_dust_map(Some(&read), &mut |_| {}, &|| false)
            .unwrap();
        assert!(repainted.dust().unwrap().is_opaque(155.0, 80.0));
        assert!(!repainted.dust().unwrap().is_opaque(40.0, 80.0));
        // Crop-shaped maps cannot silently replace source-shaped maps.
        assert!(
            video
                .with_dust_map(Some(&even_map(32, 32, 0.5)), &mut |_| {}, &|| false)
                .is_err()
        );
        let removed = video.with_dust_map(None, &mut |_| {}, &|| false).unwrap();
        assert!(removed.dust().is_none());
        assert!(video.dust().is_some());
    }

    #[test]
    fn a_crop_films_part_of_the_image_at_a_size_of_its_shape() {
        let directory = tempfile::tempdir().unwrap();
        let (starless, stars, wcs, distances) = field(directory.path());
        let mut options = options(distances, directory.path());
        options.scene.crop = Some((40, 30, 200, 150));
        // A plane mask at half the image's size is cut at its own scale.
        options.scene.planes = vec![DepthPlane {
            mask: crate::plane::tests::disc(160, 120, (102.25, 57.25), 8.0),
            distance_pc: 150.0,
            name: "knot".into(),
        }];
        options.video.image_shaped_size = Some(400);
        let video =
            Parallax::prepare(&starless, &stars, &wcs, &options, &mut |_| {}, &|| false).unwrap();
        // The frame takes the cropped image's 4:3 shape.
        assert_eq!(video.size(), (400, 300));
        assert_eq!((video.scene().width(), video.scene().height()), (200, 150));
        // The sky stays where it was: the crop's corner is the image's
        // pixel (40, 30).
        let (ra, dec) = video.wcs().pixel_to_world(0.0, 0.0);
        let (expected_ra, expected_dec) = wcs.pixel_to_world(40.0, 30.0);
        assert!((ra - expected_ra).abs() < 1e-9 && (dec - expected_dec).abs() < 1e-9);
        // The solution given and the crop come back, to prepare again with.
        assert_eq!(video.source_wcs().crpix, wcs.crpix);
        assert_eq!(video.crop(), Some((40, 30, 200, 150)));
        // The half-size mask is cut to the same part at its own scale: the
        // disc at (102.25, 57.25) is at (82.25, 42.25) of a 100 × 75 mask.
        let (_, _, _, cut) =
            crop_inputs(&starless, &stars, &wcs, &options, (40, 30, 200, 150)).unwrap();
        let mask = &cut.scene.planes[0].mask;
        assert_eq!((mask.width, mask.height), (100, 75));
        let (mut sum, mut x, mut y) = (0.0, 0.0, 0.0);
        for (index, &cover) in mask.coverage.iter().enumerate() {
            sum += cover as f64;
            x += cover as f64 * (index % mask.width) as f64;
            y += cover as f64 * (index / mask.width) as f64;
        }
        let centre = (x / sum, y / sum);
        assert!(
            (centre.0 - 82.25).abs() < 1.0 && (centre.1 - 42.25).abs() < 1.0,
            "{centre:?}"
        );
        assert_eq!(cut.scene.crop, None);

        // A crop must lie inside the image.
        options.scene.crop = Some((300, 0, 100, 100));
        assert!(
            Parallax::prepare(&starless, &stars, &wcs, &options, &mut |_| {}, &|| false).is_err()
        );
    }

    #[test]
    fn depth_planes_lift_smallest_first_and_keep_the_first_frame() {
        let directory = tempfile::tempdir().unwrap();
        let (mut starless, stars, wcs, distances) = field(directory.path());
        // A wide cloud with a bright knot in it.
        for (x, y, pixel) in starless.enumerate_pixels_mut() {
            let r2 = |cx: f64, cy: f64| (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2);
            let glow =
                0.3 * (-r2(220.0, 120.0) / 1800.0).exp() + 0.4 * (-r2(205.0, 115.0) / 40.0).exp();
            for value in pixel.0.iter_mut() {
                *value = (*value + glow as f32).min(1.0);
            }
        }
        let mut options = options(distances, directory.path());
        // At full size, so the planes and the background draw from their
        // finest levels.
        options.video.size = (320, 240);
        let plain =
            Parallax::prepare(&starless, &stars, &wcs, &options, &mut |_| {}, &|| false).unwrap();
        let mask = crate::plane::tests::disc;
        options.scene.planes = vec![
            DepthPlane {
                mask: mask(320, 240, (220.0, 120.0), 75.0),
                distance_pc: 1200.0,
                name: "cloud".into(),
            },
            DepthPlane {
                mask: mask(160, 120, (102.25, 57.25), 8.0),
                distance_pc: 150.0,
                name: "knot".into(),
            },
        ];
        let mut notes = Vec::new();
        let planed = Parallax::prepare(
            &starless,
            &stars,
            &wcs,
            &options,
            &mut |event| {
                if let Event::Note(note) = event {
                    notes.push(note.to_string());
                }
            },
            &|| false,
        )
        .unwrap();
        let planes: Vec<(&str, f64, &str)> = planed
            .summary()
            .planes
            .iter()
            .map(|plane| (plane.name.as_str(), plane.distance_pc, plane.basis.as_str()))
            .collect();
        assert_eq!(
            planes,
            [("knot", 150.0, "given"), ("cloud", 1200.0, "given")]
        );
        assert!(
            notes
                .iter()
                .any(|note| note.starts_with("depth plane knot at 150 pc (given)")),
            "{notes:?}"
        );
        // The first frame is the photograph either way; the last is not.
        let (first, also) = (plain.frame(0), planed.frame(0));
        for (a, b) in first.pixels().zip(also.pixels()) {
            for channel in 0..3 {
                assert!(a.0[channel].abs_diff(b.0[channel]) <= 1, "{a:?} {b:?}");
            }
        }
        let last = plain.frames() - 1;
        assert_ne!(plain.frame(last), planed.frame(last));
        // The knot's label would sit on its plane.
        assert_eq!(planed.scene().depth_at(205.0, 115.0), 150.0);
        assert_eq!(planed.scene().depth_at(250.0, 120.0), 1200.0);
        assert_eq!(planed.scene().depth_at(20.0, 20.0), 400.0);
    }

    /// A nebula for [`nebulae`]: its name, place and radius in pixels,
    /// and its distance, range and basis, if it has one.
    type Listed = (
        &'static str,
        (f64, f64),
        f64,
        Option<(f64, Option<(f64, f64)>, seiza::catalog::DistanceBasis)>,
    );

    /// An object catalogue of the nebulae `listed`, placed in `wcs`'s
    /// image, and their distances, written to `directory`; the starless
    /// image brightened by each one's glow, or darkened by a dark cloud's,
    /// whose name starts `LDN`. A distance `Contained` is that of the
    /// nebula listed before it.
    fn nebulae(
        directory: &Path,
        wcs: &Wcs,
        starless: &mut Rgb32FImage,
        listed: &[Listed],
    ) -> (PathBuf, PathBuf) {
        use seiza::catalog::{
            DistanceEntry, DistanceMethod, DistanceSource, ObjectDistancesBuilder,
        };
        use seiza::objects::{ObjectCatalog, ObjectKind, ObjectMetadata, SkyObject};
        let objects: Vec<SkyObject> = listed
            .iter()
            .map(|&(name, (x, y), radius, _)| {
                let (ra, dec) = wcs.pixel_to_world(x, y);
                SkyObject {
                    kind: if name.starts_with("LDN") {
                        ObjectKind::DarkNebula
                    } else {
                        ObjectKind::HiiRegion
                    },
                    ra,
                    dec,
                    mag: None,
                    // Two arcseconds a pixel.
                    major_arcmin: Some((4.0 * radius / 60.0) as f32),
                    minor_arcmin: None,
                    position_angle_deg: None,
                    name: name.into(),
                    common_name: String::new(),
                    metadata: ObjectMetadata {
                        id: format!("test:{name}"),
                        ..ObjectMetadata::default()
                    },
                }
            })
            .collect();
        let objects_path = directory.join("nebulae.bin");
        ObjectCatalog::new(objects).write_to(&objects_path).unwrap();
        let catalog = ObjectCatalog::open(&objects_path).unwrap();
        let mut builder = ObjectDistancesBuilder::new("test", catalog.fingerprint().unwrap());
        let source = builder
            .add_source(DistanceSource {
                key: "test".into(),
                ..DistanceSource::default()
            })
            .unwrap();
        for (index, &(name, _, _, distance)) in listed.iter().enumerate() {
            if let Some((distance_pc, range, basis)) = distance {
                builder.add(DistanceEntry {
                    object_id: format!("test:{name}"),
                    distance_pc,
                    lower_pc: range.map(|(lower, _)| lower),
                    upper_pc: range.map(|(_, upper)| upper),
                    source,
                    method: DistanceMethod::Photometric,
                    basis,
                    reference: None,
                    via: (basis == seiza::catalog::DistanceBasis::Contained)
                        .then(|| format!("test:{}", listed[index - 1].0)),
                });
            }
        }
        let distances_path = directory.join("nebula-distances.bin");
        builder.write_to(&distances_path).unwrap();
        for (x, y, pixel) in starless.enumerate_pixels_mut() {
            let (mut glow, mut passed) = (0.0_f64, 1.0_f64);
            for &(name, (cx, cy), radius, _) in listed {
                let r2 = (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2);
                let profile = (-r2 / (2.0 * (0.7 * radius).powi(2))).exp();
                if name.starts_with("LDN") {
                    passed *= 1.0 - 0.9 * profile;
                } else {
                    glow += 0.4 * profile;
                }
            }
            for value in pixel.0.iter_mut() {
                *value = ((*value as f64 * passed + glow) as f32).min(1.0);
            }
        }
        (objects_path, distances_path)
    }

    #[test]
    fn catalogued_nebulae_at_their_own_distances_get_planes() {
        use seiza::catalog::DistanceBasis::Measured;
        let directory = tempfile::tempdir().unwrap();
        let (mut starless, stars, wcs, distances) = field(directory.path());
        // The nebula is at 400 pc, its depth taken at the image's centre.
        let listed: [Listed; 5] = [
            (
                "NGC 9001",
                (240.0, 70.0),
                18.0,
                Some((2500.0, Some((2200.0, 2800.0)), Measured)),
            ),
            (
                "NGC 9002",
                (80.0, 70.0),
                18.0,
                Some((430.0, None, Measured)),
            ),
            ("NGC 9003", (80.0, 175.0), 18.0, None),
            (
                "NGC 9004",
                (240.0, 175.0),
                18.0,
                Some((700.0, Some((300.0, 1100.0)), Measured)),
            ),
            (
                "NGC 9005",
                (160.0, 120.0),
                40.0,
                Some((2000.0, None, Measured)),
            ),
        ];
        let (objects, object_distances) = nebulae(directory.path(), &wcs, &mut starless, &listed);
        let mut options = options(distances, directory.path());
        options.scene.objects = Some(objects);
        options.scene.object_distances = Some(object_distances);
        options.scene.auto_planes = true;
        let prepare = |options: &ParallaxOptions| {
            Parallax::prepare(&starless, &stars, &wcs, options, &mut |_| {}, &|| false).unwrap()
        };
        let planes = |video: &Parallax| -> Vec<(String, f64, String)> {
            video
                .summary()
                .planes
                .iter()
                .map(|plane| (plane.name.clone(), plane.distance_pc, plane.basis.clone()))
                .collect()
        };
        // Only the far one: the next lies within 15% of the nebula, the
        // third has no distance, the fourth's range takes in the nebula's,
        // and the last holds the point the nebula's depth was taken at.
        let video = prepare(&options);
        assert_eq!(
            planes(&video),
            [("NGC 9001".into(), 2500.0, "measured, photometric".into())]
        );
        assert_eq!(video.scene().depth_at(240.0, 70.0), 2500.0);
        // Its object names it for a distance given in place of the
        // catalogue's.
        assert_eq!(
            video.summary().planes[0].object.as_deref(),
            Some("NGC 9001")
        );
        // A distance given by name makes the second a near plane.
        options.scene.distance_overrides = vec![DistanceOverride {
            object: "ngc 9002".into(),
            distance_pc: 150.0,
        }];
        let video = prepare(&options);
        assert_eq!(
            planes(&video),
            [
                ("NGC 9002".into(), 150.0, "given".into()),
                ("NGC 9001".into(), 2500.0, "measured, photometric".into())
            ]
        );
        // An automatic tour frames the far one with the lens, the camera
        // unable to fly to it, and the near one by flying less far; each
        // turns to look along the line of sight from Earth through it.
        options.video.auto_tour = Some(AutoTour {
            targets: Some(5),
            ..AutoTour::default()
        });
        let video = prepare(&options);
        let stop = |name: &str| {
            video
                .video_options()
                .tour
                .iter()
                .find(|stop| stop.title.as_deref() == Some(name))
                .cloned()
                .unwrap()
        };
        let (far, near, flat) = (stop("NGC 9001"), stop("NGC 9002"), stop("NGC 9003"));
        assert!(far.zoom > 1.5 && far.dolly == 0.88, "{far:?}");
        assert!(near.dolly < 0.5 * flat.dolly, "{near:?} {flat:?}");
        for stop in [&far, &near] {
            assert!((stop.pan - (1.0 - stop.dolly)).abs() < 1e-9, "{stop:?}");
        }
        // A name the catalogue does not know is refused.
        options.scene.distance_overrides[0].object = "NGC 404404".into();
        let error = Parallax::prepare(&starless, &stars, &wcs, &options, &mut |_| {}, &|| false)
            .unwrap_err();
        assert!(error.to_string().contains("NGC 404404"), "{error}");
    }

    #[test]
    fn a_nebula_inside_another_moves_with_a_distance_given_for_the_other() {
        use seiza::catalog::DistanceBasis::{Contained, Measured};
        let directory = tempfile::tempdir().unwrap();
        let (mut starless, stars, wcs, distances) = field(directory.path());
        // A nebula at 700 pc, and one inside it the catalogue places there
        // too, as the nebula around it.
        let listed: [Listed; 2] = [
            (
                "NGC 9101",
                (110.0, 60.0),
                18.0,
                Some((700.0, None, Measured)),
            ),
            (
                "NGC 9102",
                (113.0, 63.0),
                14.0,
                Some((700.0, None, Contained)),
            ),
        ];
        let (objects, object_distances) = nebulae(directory.path(), &wcs, &mut starless, &listed);
        let mut options = options(distances, directory.path());
        options.scene.objects = Some(objects);
        options.scene.object_distances = Some(object_distances);
        options.scene.auto_planes = true;
        let planes = |options: &ParallaxOptions| -> Vec<(String, f64, String)> {
            Parallax::prepare(&starless, &stars, &wcs, options, &mut |_| {}, &|| false)
                .unwrap()
                .summary()
                .planes
                .iter()
                .map(|plane| (plane.name.clone(), plane.distance_pc, plane.basis.clone()))
                .collect()
        };
        // One depth, so one plane.
        let found = planes(&options);
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].1, 700.0);
        // A distance given for the outer one takes the inner one with it,
        // rather than leaving it behind on a plane of its own.
        options.scene.distance_overrides = vec![DistanceOverride {
            object: "NGC 9101".into(),
            distance_pc: 150.0,
        }];
        let moved = planes(&options);
        assert_eq!(moved.len(), 1, "{moved:?}");
        assert_eq!(moved[0].1, 150.0);
    }

    #[test]
    fn a_nebula_over_most_of_the_image_stays_with_the_nebula_and_its_parts_too() {
        use seiza::catalog::DistanceBasis::{Contained, Measured};
        let directory = tempfile::tempdir().unwrap();
        let (mut starless, stars, wcs, distances) = field(directory.path());
        // A wide nebula far behind the nebula at 400 pc, one inside it the
        // catalogue places there too, and a small one farther still; and
        // where the nebula's depth is taken, a nebula the catalogue puts
        // at 1200 pc, and one inside it.
        let listed: [Listed; 5] = [
            (
                "NGC 9301",
                (120.0, 100.0),
                110.0,
                Some((2000.0, None, Measured)),
            ),
            (
                "NGC 9302",
                (90.0, 80.0),
                14.0,
                Some((2000.0, None, Contained)),
            ),
            (
                "NGC 9303",
                (280.0, 40.0),
                16.0,
                Some((3000.0, None, Measured)),
            ),
            (
                "NGC 9304",
                (290.0, 210.0),
                30.0,
                Some((1200.0, None, Measured)),
            ),
            (
                "NGC 9305",
                (280.0, 200.0),
                13.0,
                Some((1200.0, None, Contained)),
            ),
        ];
        let (objects, object_distances) = nebulae(directory.path(), &wcs, &mut starless, &listed);
        let mut options = options(distances, directory.path());
        options.scene.objects = Some(objects);
        options.scene.object_distances = Some(object_distances);
        options.scene.auto_planes = true;
        // The nebula's depth is taken in a corner the wide one leaves out.
        options.scene.distance_focus = Some((300.0, 220.0));
        let prepare = |options: &ParallaxOptions| {
            let mut notes = Vec::new();
            let video = Parallax::prepare(
                &starless,
                &stars,
                &wcs,
                options,
                &mut |event| {
                    if let Event::Note(note) = event {
                        notes.push(note.to_string());
                    }
                },
                &|| false,
            )
            .unwrap();
            let planes: Vec<(String, f64)> = video
                .summary()
                .planes
                .iter()
                .map(|plane| (plane.name.clone(), plane.distance_pc))
                .collect();
            (planes, notes)
        };
        // Lifting the wide one would move most of the image; the one inside
        // it stays too, rather than flying off alone.
        let (planes, notes) = prepare(&options);
        assert_eq!(planes, [("NGC 9303".into(), 3000.0)], "{notes:?}");
        assert!(
            notes
                .iter()
                .any(|note| note.starts_with("NGC 9301 would cover")
                    && note.ends_with("of the image and stays with the nebula")),
            "{notes:?}"
        );
        for staying in [
            "NGC 9302 lies in NGC 9301 and stays with it",
            "NGC 9305 lies in NGC 9304 and stays with it",
        ] {
            assert!(notes.iter().any(|note| note == staying), "{notes:?}");
        }
        // A distance given for the wide one lifts it after all, the one
        // inside it with it.
        options.scene.distance_overrides = vec![DistanceOverride {
            object: "NGC 9301".into(),
            distance_pc: 1500.0,
        }];
        let (planes, notes) = prepare(&options);
        assert_eq!(
            planes,
            [("NGC 9303".into(), 3000.0), ("NGC 9301".into(), 1500.0)],
            "{notes:?}"
        );
    }

    #[test]
    fn a_catalogued_dark_cloud_gets_a_plane_that_blocks_light() {
        use seiza::catalog::DistanceBasis::NearbyCloud;
        let directory = tempfile::tempdir().unwrap();
        let (mut starless, stars, wcs, distances) = field(directory.path());
        // A dark cloud at 150 pc, in front of the nebula at 400, its
        // distance that of the nearest cloud measured.
        let listed: [Listed; 1] = [(
            "LDN 9001",
            (230.0, 170.0),
            16.0,
            Some((150.0, None, NearbyCloud)),
        )];
        let (objects, object_distances) = nebulae(directory.path(), &wcs, &mut starless, &listed);
        let mut options = options(distances, directory.path());
        options.scene.objects = Some(objects);
        options.scene.object_distances = Some(object_distances);
        options.video.size = (320, 240);
        let plain =
            Parallax::prepare(&starless, &stars, &wcs, &options, &mut |_| {}, &|| false).unwrap();
        options.scene.auto_planes = true;
        let mut notes = Vec::new();
        let clouded = Parallax::prepare(
            &starless,
            &stars,
            &wcs,
            &options,
            &mut |event| {
                if let Event::Note(note) = event {
                    notes.push(note.to_string());
                }
            },
            &|| false,
        )
        .unwrap();
        let planes = &clouded.summary().planes;
        assert_eq!(planes.len(), 1, "{notes:?}");
        assert!(planes[0].absorbs && planes[0].basis.starts_with("nearby-cloud"));
        assert!(
            notes
                .iter()
                .any(|note| note.contains("LDN 9001 at 150 pc") && note.contains("a dark cloud")),
            "{notes:?}"
        );
        assert_eq!(clouded.scene().depth_at(230.0, 170.0), 150.0);
        // The first frame is the photograph, to a step of 8 bits; the last
        // is not.
        let (first, also) = (plain.frame(0), clouded.frame(0));
        for (a, b) in first.pixels().zip(also.pixels()) {
            for channel in 0..3 {
                assert!(a.0[channel].abs_diff(b.0[channel]) <= 1, "{a:?} {b:?}");
            }
        }
        let last = plain.frames() - 1;
        assert_ne!(plain.frame(last), clouded.frame(last));
    }

    /// Stars with distances on a jittered grid over a `width` × `height`
    /// image, evenly spread from 50 to 1500 pc, but for most of those
    /// behind each of the clouds of dust `dusty` (centre, radius and
    /// distance); and the clouds the star counts find, for a nebula at
    /// 600 pc.
    fn dusty_field(
        (width, height): (usize, usize),
        dusty: &[((f64, f64), f64, f64)],
    ) -> (Vec<(f64, f64, f64)>, Clouds) {
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut random = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1_u64 << 53) as f64
        };
        let mut stars = Vec::new();
        for row in 0..height / 8 {
            for column in 0..width / 8 {
                let x = column as f64 * 8.0 + 8.0 * random();
                let y = row as f64 * 8.0 + 8.0 * random();
                let distance = 50.0 + 1450.0 * random();
                let hidden = dusty.iter().any(|&((cx, cy), radius, at)| {
                    (x - cx).hypot(y - cy) <= radius && distance > at
                }) && random() < 0.9;
                if !hidden {
                    stars.push((x, y, distance));
                }
            }
        }
        let behind: Vec<(f64, f64)> = stars
            .iter()
            .filter(|star| star.2 > 600.0)
            .map(|star| (star.0, star.1))
            .collect();
        let dust = Dust::from_star_counts(&behind, width, height, 3.0).unwrap();
        let clouds = Clouds::find(&dust);
        (stars, clouds)
    }

    /// A catalogued dark nebula for [`cloud_planes`]: a disc of radius 25
    /// at `centre` in a `width` × `height` image, `distance` parsecs away.
    fn dark_disc(
        name: &str,
        (width, height): (usize, usize),
        centre: (f64, f64),
        distance: f64,
    ) -> DarkNebula {
        let mask = crate::plane::tests::disc(width, height, centre, 25.0);
        DarkNebula {
            name: name.into(),
            region: Region::from_mask(&mask, width, height).unwrap().unwrap(),
            distance: Some((distance, "nearby-cloud".into())),
            given: false,
            at_focus: false,
        }
    }

    #[test]
    fn dark_nebulae_move_with_the_dust_clouds_they_lie_in() {
        // Three clouds of dust far apart: one about the nebula at 600 pc,
        // one at 150 pc to its upper right, and one at 300 pc to its lower
        // right.
        let (width, height) = (1600, 900);
        let (stars, clouds) = dusty_field(
            (width, height),
            &[
                ((400.0, 450.0), 110.0, 550.0),
                ((1350.0, 170.0), 90.0, 150.0),
                ((1150.0, 730.0), 90.0, 300.0),
            ],
        );
        let disc = |centre: (f64, f64)| dark_disc("", (width, height), centre, 0.0).region;
        let nebula = |name: &str, centre: (f64, f64), distance: f64| {
            dark_disc(name, (width, height), centre, distance)
        };
        let dark = vec![
            nebula("LDN 9001", (420.0, 430.0), 150.0),
            nebula("LDN 9002", (1360.0, 180.0), 150.0),
            nebula("LDN 9003", (800.0, 120.0), 300.0),
        ];
        let mut notes = Vec::new();
        let planes = cloud_planes(
            Some(&clouds),
            dark,
            &stars,
            ((400.0, 450.0), 600.0),
            (width, height),
            &mut |event| {
                if let Event::Note(note) = event {
                    notes.push(note.to_string());
                }
            },
        );
        let found = |name: &str| {
            planes
                .iter()
                .find(|plane| plane.name == name)
                .unwrap_or_else(|| {
                    let found: Vec<(&str, f64, f64)> = planes
                        .iter()
                        .map(|plane| (plane.name.as_str(), plane.distance_pc, plane.region.area()))
                        .collect();
                    panic!("no plane {name}: {found:?} {notes:?}")
                })
        };
        // The dark nebula in the nebula's own dust stays with it.
        assert!(planes.iter().all(|plane| plane.name != "LDN 9001"));
        assert!(
            notes
                .iter()
                .any(|note| note
                    .contains("LDN 9001 lies in the nebula's own dust and stays with it")),
            "{notes:?}"
        );
        // The one in the other cloud takes the whole cloud with it.
        let carried = found("LDN 9002");
        assert_eq!(carried.distance_pc, 150.0);
        assert!(carried.blocks);
        assert!(
            carried.region.area() > 5.0 * disc((1360.0, 180.0)).area(),
            "{}",
            carried.region.area()
        );
        assert!(carried.region.covers(1320.0, 140.0) > 0.9);
        // A distance given for the dark nebula moves the cloud.
        assert_eq!(carried.object.as_deref(), Some("LDN 9002"));
        // The cloud no catalogue lists lies where its stars run short.
        let counted = found("dust cloud 1");
        assert!(
            (counted.distance_pc - 300.0).abs() < 30.0,
            "{} {}",
            counted.distance_pc,
            counted.basis
        );
        assert!(counted.basis.starts_with("Gaia star counts"));
        assert!(counted.region.covers(1150.0, 730.0) > 0.9);
        assert_eq!(counted.object, None);
        // The one in clear sky stands alone.
        let alone = found("LDN 9003");
        assert_eq!(alone.region, disc((800.0, 120.0)));
        assert_eq!(alone.object.as_deref(), Some("LDN 9003"));
        assert_eq!(planes.len(), 3, "{notes:?}");
    }

    #[test]
    fn a_dust_cloud_over_most_of_the_image_stays_with_the_nebula() {
        // A small cloud about the nebula at 600 pc, and one at 150 pc over
        // more than a third of the image, a dark nebula in it.
        let (width, height) = (1600, 900);
        let (stars, clouds) = dusty_field(
            (width, height),
            &[
                ((150.0, 150.0), 90.0, 550.0),
                ((1120.0, 450.0), 450.0, 150.0),
            ],
        );
        let lift = |given: bool| {
            let mut notes = Vec::new();
            let dark = DarkNebula {
                given,
                ..dark_disc("LDN 9011", (width, height), (1120.0, 450.0), 150.0)
            };
            let planes = cloud_planes(
                Some(&clouds),
                vec![dark],
                &stars,
                ((150.0, 150.0), 600.0),
                (width, height),
                &mut |event| {
                    if let Event::Note(note) = event {
                        notes.push(note.to_string());
                    }
                },
            );
            (planes, notes)
        };
        let (planes, notes) = lift(false);
        assert!(planes.is_empty(), "{notes:?}");
        assert!(
            notes.iter().any(
                |note| note.starts_with("LDN 9011 lies in dust that would cover")
                    && note.ends_with("of the image and stays with the nebula")
            ),
            "{notes:?}"
        );
        // A distance given for its dark nebula moves it after all.
        let (planes, notes) = lift(true);
        assert_eq!(planes.len(), 1, "{notes:?}");
        let covers = planes[0].region.area() / (width * height) as f64;
        assert!(covers > AUTO_PLANE_COVERAGE, "{covers}");
    }

    #[test]
    fn a_plane_mask_of_the_wrong_shape_is_refused_and_an_empty_one_left_out() {
        let directory = tempfile::tempdir().unwrap();
        let (starless, stars, wcs, distances) = field(directory.path());
        let mut options = options(distances, directory.path());
        let plane = |mask: PlaneMask| DepthPlane {
            mask,
            distance_pc: 150.0,
            name: "cloud".into(),
        };
        options.scene.planes = vec![plane(crate::plane::tests::disc(
            100,
            100,
            (50.0, 50.0),
            9.0,
        ))];
        let error = Parallax::prepare(&starless, &stars, &wcs, &options, &mut |_| {}, &|| false)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cloud: the plane mask is 100×100"),
            "{error}"
        );
        options.scene.planes = vec![plane(crate::plane::tests::disc(32, 24, (-9.0, -9.0), 2.0))];
        let mut warnings = Vec::new();
        let video = Parallax::prepare(
            &starless,
            &stars,
            &wcs,
            &options,
            &mut |event| {
                if let Event::Warning(warning) = event {
                    warnings.push(warning.to_string());
                }
            },
            &|| false,
        )
        .unwrap();
        assert!(video.summary().planes.is_empty());
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("marks nothing")),
            "{warnings:?}"
        );
        options.scene.planes[0].distance_pc = -1.0;
        assert!(options.check().is_err());
    }

    #[test]
    fn a_dust_map_of_the_wrong_shape_or_without_dust_is_refused() {
        let directory = tempfile::tempdir().unwrap();
        let (starless, stars, wcs, distances) = field(directory.path());
        let mut options = options(distances, directory.path());
        options.scene.dust_map = Some(even_map(100, 100, 0.5));
        let error = Parallax::prepare(&starless, &stars, &wcs, &options, &mut |_| {}, &|| false)
            .unwrap_err();
        assert!(error.to_string().contains("100×100"), "{error}");
        options.scene.dust_map = Some(even_map(32, 24, 0.5));
        options.scene.dust = false;
        let error = Parallax::prepare(&starless, &stars, &wcs, &options, &mut |_| {}, &|| false)
            .unwrap_err();
        assert!(error.to_string().contains("needs the dust on"), "{error}");
    }

    #[test]
    fn a_prepared_video_hands_every_frame_to_the_callers_sink() {
        let directory = tempfile::tempdir().unwrap();
        let (starless, stars, wcs, distances) = field(directory.path());
        let options = options(distances, directory.path());
        let mut notes = Vec::new();
        let parallax = Parallax::prepare(
            &starless,
            &stars,
            &wcs,
            &options,
            &mut |event| {
                if let Event::Note(note) = event {
                    notes.push(note.to_string());
                }
            },
            &|| false,
        )
        .unwrap();
        let summary = parallax.summary();
        assert_eq!(summary.detected_stars, 4, "{notes:?}");
        assert_eq!(summary.gaia_matches, 4);
        assert_eq!(summary.background_distance_pc, 400.0);
        assert_eq!(parallax.frames(), 6);
        assert_eq!(parallax.size(), (160, 120));
        let mut seen = Vec::new();
        let mut frames_reported = 0;
        parallax
            .render(
                Box::new(FrameFn::new(|frame: &RgbImage| {
                    seen.push(frame.dimensions());
                    Ok(())
                })),
                &mut |event| {
                    if let Event::Frame { .. } = event {
                        frames_reported += 1;
                    }
                },
                &|| false,
            )
            .unwrap();
        assert_eq!(seen, vec![(160, 120); 6]);
        assert_eq!(frames_reported, 6);
        // A frame on request is the one the sink saw.
        assert_eq!(parallax.frame(0).dimensions(), (160, 120));
    }

    #[test]
    fn a_stop_or_a_refusing_sink_ends_the_video_early() {
        let directory = tempfile::tempdir().unwrap();
        let (starless, stars, wcs, distances) = field(directory.path());
        let options = options(distances, directory.path());
        let parallax =
            Parallax::prepare(&starless, &stars, &wcs, &options, &mut |_| {}, &|| false).unwrap();
        let pushed = std::cell::Cell::new(0);
        let stopped = parallax.render(
            Box::new(FrameFn::new(|_: &RgbImage| {
                pushed.set(pushed.get() + 1);
                Ok(())
            })),
            &mut |_| {},
            &|| pushed.get() == 2,
        );
        assert!(matches!(stopped, Err(Error::Stopped)), "{stopped:?}");
        assert_eq!(pushed.get(), 2);
        let refused = parallax.render(
            Box::new(FrameFn::new(|_: &RgbImage| Err("disk full".to_string()))),
            &mut |_| {},
            &|| false,
        );
        assert!(
            matches!(&refused, Err(Error::Encode(crate::encode::Error::Sink(message))) if message == "disk full"),
            "{refused:?}"
        );
    }

    #[test]
    fn a_stop_framed_at_a_depth_the_image_does_not_show_is_reported() {
        let directory = tempfile::tempdir().unwrap();
        let (mut starless, stars, wcs, distances) = field(directory.path());
        // A glowing cloud for the plane to lift.
        for (x, y, pixel) in starless.enumerate_pixels_mut() {
            let r2 = (x as f64 - 220.0).powi(2) + (y as f64 - 120.0).powi(2);
            let glow = 0.4 * (-r2 / 800.0).exp();
            for value in pixel.0.iter_mut() {
                *value = (*value + glow as f32).min(1.0);
            }
        }
        let mut options = options(distances, directory.path());
        options.video.size = (320, 240);
        options.scene.planes = vec![DepthPlane {
            mask: crate::plane::tests::disc(320, 240, (220.0, 120.0), 40.0),
            distance_pc: 1200.0,
            name: "cloud".into(),
        }];
        // On the plane at its depth; on it at another; and on the
        // background, at 400 pc, at a depth from a plane no longer there.
        let stop = |focus: (f64, f64), depth_pc: f64, title: &str| TourStop {
            focus: Some(focus),
            dolly: 0.3,
            travel: 1.0,
            hold: 0.5,
            title: Some(title.into()),
            depth_pc: Some(depth_pc),
            ..TourStop::default()
        };
        options.video.tour = vec![
            TourStop::default(),
            stop((220.0, 120.0), 1200.0, "cloud"),
            stop((220.0, 120.0), 600.0, "cloud again"),
            stop((40.0, 40.0), 288.0, "LDN 9001"),
        ];
        let mut warnings = Vec::new();
        Parallax::prepare(
            &starless,
            &stars,
            &wcs,
            &options,
            &mut |event| {
                if let Event::Warning(warning) = event
                    && warning.contains("is framed at")
                {
                    warnings.push(warning.to_string());
                }
            },
            &|| false,
        )
        .unwrap();
        assert_eq!(
            warnings,
            [
                "stop 3 (cloud again) is framed at 600 pc, but the image there lies at 1200 pc: \
                 drop its depth, or plan the tour again",
                "stop 4 (LDN 9001) is framed at 288 pc, but the image there lies at 400 pc: \
                 drop its depth, or plan the tour again",
            ]
        );
    }

    #[test]
    fn overlay_images_are_read_at_setup_and_odd_ones_warned_of() {
        let folder = tempfile::tempdir().unwrap();
        let png = |name: &str, (width, height): (u32, u32), alpha: u8| {
            let path = folder.path().join(name);
            image::RgbaImage::from_pixel(width, height, image::Rgba([200, 100, 50, alpha]))
                .save(&path)
                .unwrap();
            path
        };
        let clear = png("clear.png", (320, 180), 128);
        let opaque = png("opaque.png", (320, 180), 255);
        let square = png("square.png", (100, 100), 128);
        let logo = png("logo.png", (400, 100), 255);
        let decorated = |video: VideoOptions| {
            let mut warnings = Vec::new();
            let stops = video.stops();
            let decor = decor(&video, &stops, &mut |event| {
                if let Event::Warning(warning) = event {
                    warnings.push(warning.to_string());
                }
            });
            (decor, warnings)
        };
        let video = VideoOptions {
            size: (640, 360),
            ..VideoOptions::default()
        };
        // A card the frame's shape, with transparency, passes quietly and
        // fills the frame; one read twice is read once.
        let (decor, warnings) = decorated(VideoOptions {
            intro_card: Some(clear.clone()),
            outro_card: Some(clear.clone()),
            ..video.clone()
        });
        let decor = decor.unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        let intro = decor.intro.unwrap();
        assert_eq!((intro.picture.size(), intro.at), ((640, 360), (0, 0)));
        assert!(Arc::ptr_eq(&intro.picture, &decor.outro.unwrap().picture));
        // An opaque card, or one of another shape, is drawn but warned of.
        let (decor, warnings) = decorated(VideoOptions {
            tour: vec![
                TourStop {
                    card: Some(opaque),
                    ..TourStop::default()
                },
                TourStop {
                    card: Some(square),
                    ..TourStop::default()
                },
            ],
            ..video.clone()
        });
        let decor = decor.unwrap();
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert!(warnings[0].contains("card of stop 1") && warnings[0].contains("transparency"));
        assert!(warnings[1].contains("card of stop 2") && warnings[1].contains("100×100"));
        let square = decor.cards[1].as_ref().unwrap();
        assert_eq!((square.picture.size(), square.at), ((360, 360), (140, 0)));
        // The watermark image takes its share of the shorter side, at most
        // a third of the width.
        let (decor, _) = decorated(VideoOptions {
            watermark_image: Some(logo.clone()),
            ..video.clone()
        });
        let watermark = decor.unwrap().watermark_image.unwrap();
        assert_eq!(watermark.size(), (86, 22));
        let (decor, _) = decorated(VideoOptions {
            watermark_image: Some(logo),
            watermark_size: 0.5,
            ..video.clone()
        });
        assert_eq!(decor.unwrap().watermark_image.unwrap().size(), (213, 53));
        // A missing file fails, naming it.
        let missing = folder.path().join("missing.png");
        let (decor, _) = decorated(VideoOptions {
            outro_card: Some(missing.clone()),
            ..video
        });
        let error = decor.unwrap_err().to_string();
        assert!(
            error.contains("closing card") && error.contains("missing.png"),
            "{error}"
        );
    }

    #[test]
    fn a_stop_written_out_reads_back() {
        for stop in [
            TourStop::default(),
            TourStop {
                focus: Some((2700.0, 3400.0)),
                dolly: 0.85,
                zoom: 1.2,
                rotate_deg: -12.5,
                pan: 0.3,
                travel: 6.0,
                hold: 1.5,
                spin_deg: 360.0,
                push: 0.25,
                title: Some(r#"Sh2-170, the "core" \ centre"#.into()),
                card: Some(PathBuf::from(r#"cards\Sh2-170 "east".png"#)),
                depth_pc: Some(2790.0),
            },
        ] {
            let text = format_stop(&stop);
            assert_eq!(parse_stop(&text).unwrap(), stop, "{text}");
        }
        assert_eq!(format_stop(&TourStop::default()), "whole travel=5");
        // Fractions of a pixel survive, and a dolly just short of 1 is not
        // rounded up to it.
        let fine = TourStop {
            focus: Some((1999.5, 431.25)),
            dolly: 0.9996,
            push: 0.99995,
            title: Some(String::new()),
            ..TourStop::default()
        };
        let read = parse_stop(&format_stop(&fine)).unwrap();
        assert_eq!(read.focus, fine.focus);
        assert!(
            read.dolly < 1.0 && read.push < 1.0,
            "{}",
            format_stop(&fine)
        );
        assert_eq!(read.title, None);
        let titled = parse_stop("whole hold=2 title=Plain").unwrap();
        assert_eq!(titled.title.as_deref(), Some("Plain"));
        let spaced = parse_stop(r#"10,20 title="NGC 7822 \"east\"" hold=1"#).unwrap();
        assert_eq!(spaced.title.as_deref(), Some(r#"NGC 7822 "east""#));
        assert_eq!(spaced.hold, 1.0);
        assert!(parse_stop(r#"whole title="open"#).is_err());
    }

    #[test]
    fn a_looping_tour_ends_at_its_first_view_and_frame() {
        let stops = |text: &[&str]| -> Vec<TourStop> {
            text.iter().map(|stop| parse_stop(stop).unwrap()).collect()
        };
        // Ending elsewhere, it gains a last stop at the first's view, the
        // tour's mean travel on, with a long final drift; a half-turn spin
        // is made up to a whole turn by the nearest way round.
        let tour = stops(&[
            "whole hold=1.5 title=Start",
            "100,100 dolly=0.5 travel=4 hold=4 spin=180 push=0.2",
            "300,200 dolly=0.3 rotate=30 travel=6",
        ]);
        let closed = close_loop(&tour);
        assert_eq!(closed.len(), 4);
        let last = &closed[3];
        assert_eq!((last.focus, last.dolly, last.zoom), (None, 0.0, 1.0));
        assert_eq!((last.travel, last.hold), (5.0, 4.5));
        assert_eq!(last.title.as_deref(), Some("Start"));
        assert_eq!(last.rotate_deg, 180.0);
        assert_eq!((last.rotate_deg + 180.0) % 360.0, 0.0);
        // A whole-turn spin needs no unwinding.
        let mut spun = tour.clone();
        spun[1].spin_deg = -360.0;
        assert_eq!(close_loop(&spun)[3].rotate_deg, 0.0);
        // Ending at the first's view already, it gains nothing.
        let home = stops(&[
            "whole hold=1",
            "100,100 dolly=0.5 travel=4 hold=1",
            "whole travel=5 hold=4",
        ]);
        assert_eq!(close_loop(&home), home);
        // The video's length counts the added stop.
        let video = VideoOptions {
            tour,
            tour_loop: true,
            ..VideoOptions::default()
        };
        assert_eq!(video.seconds(), 1.5 + 8.0 + 6.0 + 9.5);
        assert!(
            VideoOptions {
                tour_loop: true,
                ..VideoOptions::default()
            }
            .check()
            .is_err()
        );
    }

    #[test]
    fn eight_bit_images_are_told_from_deeper_ones() {
        let eight = Rgb32FImage::from_fn(40, 30, |x, y| {
            image::Rgb([((x * 7 + y) % 256) as f32 / 255.0, 0.0, 1.0])
        });
        assert!(eight_bit(&eight));
        let sixteen = Rgb32FImage::from_fn(40, 30, |x, y| {
            image::Rgb([((x * 977 + y * 31) % 65_536) as f32 / 65_535.0, 0.25, 1.0])
        });
        assert!(!eight_bit(&sixteen));
    }

    #[test]
    fn options_that_cannot_make_a_video_are_refused() {
        let video = |video: VideoOptions| ParallaxOptions {
            video,
            ..ParallaxOptions::default()
        };
        for options in [
            video(VideoOptions {
                dolly: 1.0,
                ..VideoOptions::default()
            }),
            video(VideoOptions {
                size: (161, 120),
                ..VideoOptions::default()
            }),
            video(VideoOptions {
                fps: 0,
                ..VideoOptions::default()
            }),
            video(VideoOptions {
                overlay_density: 2.0,
                ..VideoOptions::default()
            }),
            ParallaxOptions {
                scene: SceneOptions {
                    distance_pc: Some(-1.0),
                    ..SceneOptions::default()
                },
                ..ParallaxOptions::default()
            },
            // Values that once passed and then panicked, hung, or drew
            // black frames.
            video(VideoOptions {
                growth_limit: 0.5,
                ..VideoOptions::default()
            }),
            video(VideoOptions {
                fade_from: 0.0,
                ..VideoOptions::default()
            }),
            video(VideoOptions {
                seconds: f64::INFINITY,
                ..VideoOptions::default()
            }),
            video(VideoOptions {
                zoom: 0.5,
                ..VideoOptions::default()
            }),
            video(VideoOptions {
                tour: vec![
                    parse_stop("whole hold=1").unwrap(),
                    parse_stop("100,100 dolly=0.3 hold=inf").unwrap(),
                ],
                ..VideoOptions::default()
            }),
            video(VideoOptions {
                tour: vec![
                    parse_stop("whole hold=1").unwrap(),
                    parse_stop("nan,50 dolly=0.3 travel=1").unwrap(),
                ],
                ..VideoOptions::default()
            }),
            video(VideoOptions {
                labels: vec![CustomLabel {
                    x: f64::NAN,
                    y: 1.0,
                    radius: 0.0,
                    text: "lost".into(),
                }],
                ..VideoOptions::default()
            }),
            ParallaxOptions {
                scene: SceneOptions {
                    distance_pc: Some(f64::INFINITY),
                    ..SceneOptions::default()
                },
                ..ParallaxOptions::default()
            },
            ParallaxOptions {
                scene: SceneOptions {
                    unmatched_distance_pc: Some(f64::NAN),
                    ..SceneOptions::default()
                },
                ..ParallaxOptions::default()
            },
        ] {
            assert!(options.check().is_err(), "{options:?}");
        }
        assert!(ParallaxOptions::default().check().is_ok());
    }

    #[test]
    fn stars_without_a_distance_fly_in_front_of_a_target_beyond_them() {
        let directory = tempfile::tempdir().unwrap();
        let (starless, mut stars, wcs, distances) = field(directory.path());
        // Three stars the catalogue does not have.
        for (sx, sy) in [(30.0, 200.0), (290.0, 200.0), (100.0, 120.0)] {
            for (x, y, pixel) in stars.enumerate_pixels_mut() {
                let r2 = (x as f64 - sx).powi(2) + (y as f64 - sy).powi(2);
                let light = 0.9 * (-r2 / 4.0).exp() as f32;
                for channel in pixel.0.iter_mut() {
                    *channel = (*channel + light).min(1.0);
                }
            }
        }
        for (target, flying) in [(400.0, 7), (5000.0, 7)] {
            let mut options = options(distances.clone(), directory.path());
            options.scene.distance_pc = Some(target);
            let video =
                Parallax::prepare(&starless, &stars, &wcs, &options, &mut |_| {}, &|| false)
                    .unwrap();
            let summary = video.summary();
            assert_eq!(summary.flying_stars, flying, "target {target}: {summary:?}");
            let unmatched = summary.unmatched_distance_pc;
            assert!(
                (unmatched - target).abs() > 0.02 * target,
                "{unmatched} {target}"
            );
            if target > 2500.0 {
                assert!(unmatched < target, "{unmatched}");
            }
        }
    }

    #[test]
    fn a_tour_stop_off_the_image_is_refused_before_the_work() {
        let directory = tempfile::tempdir().unwrap();
        let (starless, stars, wcs, distances) = field(directory.path());
        let mut options = options(distances, directory.path());
        options.video.tour = vec![
            parse_stop("whole hold=1").unwrap(),
            parse_stop("100,100 dolly=0.3 travel=1").unwrap(),
            parse_stop("3000,-2000 dolly=0.5 travel=1 hold=1").unwrap(),
        ];
        let mut notes = Vec::new();
        let refused = Parallax::prepare(
            &starless,
            &stars,
            &wcs,
            &options,
            &mut |event| notes.push(format!("{event:?}")),
            &|| false,
        );
        let Err(Error::Invalid(message)) = refused else {
            panic!("{:?}", refused.map(|_| ()));
        };
        assert!(message.contains("tour stop 3"), "{message}");
        // Refused before any of the work.
        assert!(notes.is_empty(), "{notes:?}");
    }

    #[test]
    fn labels_wait_for_the_catalogue_and_fit_a_short_frame() {
        let directory = tempfile::tempdir().unwrap();
        let (starless, stars, wcs, distances) = field(directory.path());
        let mut options = options(distances, directory.path());
        options.video.overlay = true;
        // A frame shorter than a line of labels.
        options.video.size = (64, 18);
        let mut warnings = Vec::new();
        let video = Parallax::prepare(
            &starless,
            &stars,
            &wcs,
            &options,
            &mut |event| {
                if let Event::Warning(message) = event {
                    warnings.push(message.to_string());
                }
            },
            &|| false,
        )
        .expect("an unlabelled video, not a failure");
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("no catalogued objects labelled")),
            "{warnings:?}"
        );
        assert_eq!(video.summary().labelled_objects, 0);
        video.frame(0);
        // Installed later, the catalogue labels the next video.
        let place = |x: f64, y: f64| {
            let (ra, dec) = wcs.pixel_to_world(x, y);
            seiza::objects::SkyObject {
                kind: seiza::objects::ObjectKind::Nebula,
                ra,
                dec,
                mag: None,
                major_arcmin: Some(1.0),
                minor_arcmin: None,
                position_angle_deg: None,
                name: "NGC 9001".into(),
                common_name: String::new(),
                metadata: Default::default(),
            }
        };
        seiza::objects::ObjectCatalog::new(vec![place(160.0, 120.0)])
            .write_to(&directory.path().join("no-objects.bin"))
            .unwrap();
        let again = video
            .reconfigure(&options.video, &mut |_| {}, &|| false)
            .unwrap();
        assert_eq!(again.summary().labelled_objects, 1);
        again.frame(0);
    }

    #[test]
    fn a_reconfigured_video_shares_the_scene_and_leaves_the_first_alone() {
        let directory = tempfile::tempdir().unwrap();
        let (starless, stars, wcs, distances) = field(directory.path());
        let options = options(distances, directory.path());
        let first =
            Parallax::prepare(&starless, &stars, &wcs, &options, &mut |_| {}, &|| false).unwrap();
        let before = first.frame(3);
        // Another camera, size, length, rate and quality on the same scene.
        let video = VideoOptions {
            focus: Some((150.0, 110.0)),
            dolly: 0.6,
            rotate_deg: (0.0, -360.0),
            size: (200, 150),
            seconds: 2.0,
            fps: 5,
            quality: Quality::High,
            watermark: None,
            ..options.video.clone()
        };
        let second = std::thread::scope(|scope| {
            // The first video keeps drawing while the second is fitted.
            let drawing = scope.spawn(|| first.frame(3));
            let second = first.reconfigure(&video, &mut |_| {}, &|| false).unwrap();
            assert_eq!(drawing.join().unwrap(), before);
            second
        });
        assert!(Arc::ptr_eq(&first.prepared, &second.prepared));
        assert_eq!(
            (second.frames(), second.size(), second.fps()),
            (10, (200, 150), 5)
        );
        assert_eq!(first.frames(), 6);
        assert_eq!(first.frame(3), before);
        // The scene's depth is the first's, wherever the camera now aims.
        assert_eq!(
            second.summary().background_distance_pc,
            first.summary().background_distance_pc
        );
        assert_eq!(
            second.summary().detected_stars,
            first.summary().detected_stars
        );
        // A whole turn clockwise is kept as asked: early on the frame has
        // turned clockwise by as much of the turn as the move has gone.
        let t = 2.0 / 9.0;
        let progress = t * t * (3.0 - 2.0 * t);
        let early = second.shot().view(second.scene(), 2).turn();
        assert!(
            (early + std::f64::consts::TAU * progress).abs() < 1e-9,
            "{early}"
        );
        assert_eq!(second.shot().rotation, (0.0, -std::f64::consts::TAU));
        // A stop asked for ends the reconfiguration.
        let stopped = first.reconfigure(&video, &mut |_| {}, &|| true);
        assert!(matches!(stopped, Err(Error::Stopped)), "{stopped:?}");
    }

    #[test]
    fn the_fit_is_reported_as_asked_and_used() {
        let directory = tempfile::tempdir().unwrap();
        let (starless, stars, wcs, distances) = field(directory.path());
        let mut options = options(distances, directory.path());
        // A turned frame from the whole image must zoom in.
        options.video.rotate_deg = (0.0, 45.0);
        let video =
            Parallax::prepare(&starless, &stars, &wcs, &options, &mut |_| {}, &|| false).unwrap();
        let fit = &video.summary().fit;
        assert_eq!(fit.zoom.0, 1.0);
        assert!(fit.zoom.1 > 1.0, "{fit:?}");
        assert_eq!(fit.lead, (1.0, video.shot().lead));
        assert!(fit.stops.is_empty());
    }
}
