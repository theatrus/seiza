//! A whole parallax video from a starless image, its stars and a plate
//! solution: the stars' distances from Gaia and Hipparcos, the scene cut
//! into depths, the dust, lifted galaxies and labels, then the frames.
//!
//! [`Parallax::prepare`] does the work that comes once. [`Parallax::frame`]
//! then draws any frame on request, and [`Parallax::render`] hands every
//! frame in turn to a [`FrameSink`]: one of this crate's encoders, or the
//! caller's own, such as a platform video encoder.

use crate::field::{self, FieldSource};
use crate::lift::Extent;
use crate::overlay::{self, CustomLabel, Overlay};
use crate::render::{Easing, Quality, Shot, Start, Stop};
use crate::scene::{CutOptions, Scene, SmallStars, Star};
use crate::tour::{self, AutoTour, PlannedStop};
use crate::{FrameSink, LightImage, VideoSettings};
use image::{Rgb, Rgb32FImage, RgbImage};
use seiza::Wcs;
use std::path::{Path, PathBuf};

/// Where lifted galaxies fly: so far beyond the stars that they hold still.
pub const GALAXY_DISTANCE_PC: f64 = 1e8;

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
    #[error("stopped before the last frame")]
    Stopped,
}

/// A stop on a tour, as the options give it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TourStop {
    /// The image point the camera centres, or the image's centre if
    /// `None`.
    pub focus: Option<(f64, f64)>,
    /// Fraction of the way to the nebula the camera has flown, 0 to below 1.
    pub dolly: f64,
    /// The lens's magnification over the opening view's.
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
        }
    }
}

/// A [`TourStop`] written `X,Y` or `whole` (the image's centre), then any
/// of `dolly=`, `zoom=`, `rotate=` (degrees), `pan=`, `travel=` and
/// `hold=` (seconds), `spin=` (degrees turned while holding) and `push=`
/// (the share of the remaining way flown in while holding),
/// separated by spaces: `2700,3400 dolly=0.85 rotate=-20 travel=6
/// hold=1.5`.
pub fn parse_stop(text: &str) -> Result<TourStop, String> {
    let mut parts = text.split_whitespace();
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
            other => {
                return Err(format!(
                    "a stop takes dolly, zoom, rotate, pan, travel, hold, spin and push; got \
                     {other}"
                ));
            }
        }
    }
    Ok(stop)
}

/// Everything about a video but its images and plate solution. The
/// defaults are `seiza parallax-video`'s.
#[derive(Clone, Debug, PartialEq)]
pub struct ParallaxOptions {
    /// The image point the camera flies toward, image pixels; the image's
    /// centre if `None`.
    pub focus: Option<(f64, f64)>,
    /// Distance to the nebula or galaxy behind the stars, parsecs; if
    /// `None`, the catalogued object's at the focus point, else the median
    /// Gaia distance of the stars near it.
    pub distance_pc: Option<f64>,
    /// Distance for stars without one, parsecs; if `None`, the matched
    /// stars' median, and never nearer than the nebula.
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
    /// The frame's turn at the first and last frames, degrees
    /// anticlockwise.
    pub rotate_deg: (f64, f64),
    pub easing: Easing,
    pub quality: Quality,
    /// A star the camera nears grows up to this many times its size, and
    /// fades out past this growth.
    pub growth_limit: f64,
    pub fade_from: f64,
    /// A tour of stops instead of the single move toward `focus`: the
    /// camera glides through them, easing to a halt where it holds. The
    /// first is the opening view. The nebula takes the distance of the
    /// object at `focus` if given, else at the first stop flown in toward. Its length replaces `seconds`, and the
    /// move's own settings (`start`, `dolly`, `truck`, `pan`, `rotate_deg`,
    /// `zoom_end`, `easing`) go unused.
    pub tour: Vec<TourStop>,
    /// Plan a tour of the catalogued objects in the field, when `tour` is
    /// empty: the most prominent, visited in a short round from the whole
    /// image and back.
    pub auto_tour: Option<AutoTour>,
    /// How fast a tour moves on through a stop it holds at, as a share of
    /// its pace between stops, so it never quite stops; 0 comes to rest.
    pub tour_glide: f64,
    /// Frame size, even sides.
    pub size: (usize, usize),
    pub seconds: f64,
    pub fps: u32,
    /// Label the catalogued objects in the field, the most prominent
    /// `overlay_density` share of them.
    pub overlay: bool,
    pub overlay_density: f64,
    /// The caller's own labels, in `label_color`.
    pub labels: Vec<CustomLabel>,
    pub label_color: [u8; 3],
    /// A line of text in the bottom-right corner of every frame.
    pub watermark: Option<String>,
}

impl Default for ParallaxOptions {
    fn default() -> Self {
        Self {
            focus: None,
            distance_pc: None,
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
            start: Start::Focus,
            dolly: 0.4,
            truck: 0.0,
            truck_angle_deg: 0.0,
            pan: 0.0,
            zoom: 1.0,
            zoom_end: 1.0,
            rotate_deg: (0.0, 0.0),
            easing: Easing::InOut,
            quality: Quality::Standard,
            growth_limit: 4.0,
            fade_from: 6.0,
            tour: Vec::new(),
            auto_tour: None,
            tour_glide: 0.2,
            size: (1920, 1080),
            seconds: 8.0,
            fps: 30,
            overlay: false,
            overlay_density: overlay::DEFAULT_DENSITY,
            labels: Vec::new(),
            label_color: [0xf0, 0xf4, 0xf8],
            watermark: None,
        }
    }
}

impl ParallaxOptions {
    /// Whether these options can make a video.
    pub fn check(&self) -> Result<(), Error> {
        let invalid = |message: &str| Err(Error::Invalid(message.into()));
        if !(0.0..1.0).contains(&self.dolly) {
            return invalid("the dolly must be at least 0 and below 1");
        }
        if !(0.0..=1.0).contains(&self.pan) {
            return invalid("the pan must be from 0 to 1");
        }
        if !(0.0..=1.0).contains(&self.overlay_density) {
            return invalid("the overlay density must be from 0 to 1");
        }
        if self.dust_opacity.is_nan() || self.dust_opacity < 0.0 {
            return invalid("the dust opacity must be at least 0");
        }
        if !self.rotate_deg.0.is_finite() || !self.rotate_deg.1.is_finite() {
            return invalid("the rotation must be numbers of degrees");
        }
        if !(self.zoom.is_finite() && self.zoom > 0.0) || !(1.0..).contains(&self.zoom_end) {
            return invalid("the zoom must be positive and the end zoom at least 1");
        }
        if !self.truck.is_finite() || !self.truck_angle_deg.is_finite() {
            return invalid("the truck must be a number");
        }
        if self.seconds.is_nan() || self.seconds <= 0.0 || self.fps == 0 {
            return invalid("the length and frame rate must be positive");
        }
        let (width, height) = self.size;
        if width < 16 || height < 16 || width % 2 != 0 || height % 2 != 0 {
            return invalid("the frame's sides must be even and at least 16");
        }
        if self
            .distance_pc
            .is_some_and(|distance| distance.is_nan() || distance <= 0.0)
        {
            return invalid("the distance must be positive");
        }
        if self.tour.len() == 1 {
            return invalid("a tour needs at least two stops");
        }
        for stop in &self.tour {
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
            if !(stop.travel >= 0.0 && stop.hold >= 0.0) {
                return invalid("a stop's travel and hold must be at least 0 seconds");
            }
        }
        if let Some(auto) = &self.auto_tour
            && (auto.targets == Some(0)
                || auto.hold.is_nan()
                || auto.hold < 0.0
                || auto.motion.is_nan()
                || auto.motion < 0.0)
        {
            return invalid(
                "an automatic tour needs a target, and a hold and motion of at least 0",
            );
        }
        if self.tour_glide.is_nan() || self.tour_glide < 0.0 {
            return invalid("a tour's glide must be at least 0");
        }
        if !self.tour.is_empty() && self.seconds() <= 0.0 {
            return invalid("a tour must take some time");
        }
        Ok(())
    }

    /// The video's length, seconds: the tour's, if there is one.
    pub fn seconds(&self) -> f64 {
        if self.tour.is_empty() {
            return self.seconds;
        }
        self.tour
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

/// What preparing a video found.
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
    /// The nebula's distance and how it was found.
    pub background_distance_pc: f64,
    pub background_basis: String,
    /// Where stars without a distance were put.
    pub unmatched_distance_pc: f64,
    /// Stars cut out to fly at their own distances.
    pub flying_stars: usize,
    /// Galaxies lifted onto the far field.
    pub galaxies_lifted: Vec<String>,
    /// The dust map's median and least transmission, 0 to 1, if one was
    /// made.
    pub dust_transmission: Option<(f32, f32)>,
    /// Catalogued objects in the field to label.
    pub labelled_objects: usize,
}

/// A prepared parallax video.
pub struct Parallax {
    scene: Scene,
    shot: Shot,
    overlay: Option<Overlay>,
    summary: Summary,
    fps: u32,
}

impl std::fmt::Debug for Parallax {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Parallax")
            .field("shot", &self.shot)
            .field("summary", &self.summary)
            .field("fps", &self.fps)
            .finish_non_exhaustive()
    }
}

impl Parallax {
    /// Prepare a video of `starless` and `stars`, a stretched image split
    /// into its starless image and its unscreened stars (values 0 to 1),
    /// whose sky `wcs` gives. `report` hears each step.
    pub fn prepare(
        starless: &Rgb32FImage,
        stars: &Rgb32FImage,
        wcs: &Wcs,
        options: &ParallaxOptions,
        report: &mut dyn FnMut(Event),
    ) -> Result<Self, Error> {
        options.check()?;
        let planned;
        let options = match &options.auto_tour {
            Some(auto) if options.tour.is_empty() => {
                planned = plan_options(options, auto, wcs, starless.dimensions(), report)?;
                &planned
            }
            _ => options,
        };
        if starless.dimensions() != stars.dimensions() {
            return Err(Error::Invalid(format!(
                "the starless image is {:?} but the stars image is {:?}",
                starless.dimensions(),
                stars.dimensions()
            )));
        }
        let (width, height) = (stars.width() as usize, stars.height() as usize);
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
        let scale = wcs.scale_arcsec_per_px();
        let focal_px = 206_264.806_247 / scale;

        let stars_light = LightImage::from_display(stars);
        let detections = seiza_stars::fold_core_fragments(crate::find_stars(&stars_light, 5.0));
        summary.detected_stars = detections.len();
        report(Event::Note(&format!(
            "{} stars found in the stars image",
            detections.len()
        )));
        let source = FieldSource {
            star_distances: options.star_distances.as_deref(),
            gaia_max_mag: options.gaia_max_mag,
            cache: options.gaia_cache.as_deref(),
            online: options.online,
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

        let centre = ((width as f64 - 1.0) / 2.0, (height as f64 - 1.0) / 2.0);
        // On a tour, the first stop the camera flies in toward stands for
        // the target whose distance the nebula takes, unless `focus` says.
        let first_visited = options
            .tour
            .iter()
            .find(|stop| stop.dolly > 0.0)
            .map(|stop| stop.focus.unwrap_or(centre));
        let focus = options.focus.or(first_visited).unwrap_or(centre);
        if focus.0 < 0.0 || focus.1 < 0.0 || focus.0 >= width as f64 || focus.1 >= height as f64 {
            return Err(Error::Invalid(format!(
                "the focus point {focus:?} is outside the {width}x{height} image"
            )));
        }
        let (distance, basis) = match options.distance_pc {
            Some(distance) => (distance, "given".to_string()),
            None => match target_distance(options, wcs, (width, height), focus) {
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
            },
        };
        report(Event::Note(&format!(
            "background at {distance:.0} pc, {basis}"
        )));
        (summary.background_distance_pc, summary.background_basis) = (distance, basis);

        // Most stars too faint to match are field stars well beyond a
        // nearby target, so they go to the matched stars' median distance
        // rather than onto the nebula.
        let unmatched = options.unmatched_distance_pc.unwrap_or_else(|| {
            field::median(matched.iter().filter_map(|star| star.distance_pc))
                .unwrap_or(distance)
                .max(distance)
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

        let mut starless = LightImage::from_display(starless);
        // Galaxies lie far beyond everything else, but the star remover
        // leaves them on the nebula's plane.
        let mut galaxies = Vec::new();
        if !options.keep_galaxies {
            match galaxies_in_image(options, wcs, (width, height), focus) {
                Ok(found) => {
                    for (name, extent) in found {
                        if let Some(sprite) =
                            crate::lift_object(&mut starless, &extent, GALAXY_DISTANCE_PC)
                        {
                            galaxies.push((name, sprite));
                        }
                    }
                }
                Err(error) => report(Event::Warning(&format!("no galaxies lifted: {error}"))),
            }
            if !galaxies.is_empty() {
                let names: Vec<&str> = galaxies
                    .iter()
                    .take(5)
                    .map(|(name, _)| name.as_str())
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
        let mut scene = Scene::new(
            &starless,
            &stars_light,
            &placed,
            distance,
            unmatched,
            focal_px,
            &CutOptions {
                max_stars: options.max_stars,
                small_stars: options.small_stars,
                ..CutOptions::default()
            },
        );
        summary.flying_stars = scene.sprites.len();
        summary.galaxies_lifted = galaxies.iter().map(|(name, _)| name.clone()).collect();
        let lifted: Vec<(String, (f64, f64), f64)> = galaxies
            .iter()
            .map(|(name, sprite)| (name.clone(), (sprite.x, sprite.y), sprite.distance_pc))
            .collect();
        scene
            .sprites
            .extend(galaxies.into_iter().map(|(_, sprite)| sprite));
        if options.dust {
            // The stars seen through the dust: all but the matched ones in
            // front of it. The unmatched ones count however near the dust
            // they were placed, as faint stars are mostly far.
            let behind: Vec<(f64, f64)> = matched
                .iter()
                .filter(|star| star.distance_pc.is_none_or(|pc| pc > distance))
                .map(|star| (star.x, star.y))
                .collect();
            scene.dust =
                crate::Dust::from_star_counts(&behind, width, height, options.dust_opacity)
                    .map(|dust| dust.with_darkness(&starless, &behind));
            match &scene.dust {
                Some(dust) => {
                    let (cells, _) = dust.cells();
                    let mut sorted = cells.to_vec();
                    sorted.sort_by(f32::total_cmp);
                    let (middle, thickest) = (sorted[sorted.len() / 2], sorted[0]);
                    summary.dust_transmission = Some((middle, thickest));
                    report(Event::Note(&format!(
                        "dust mapped from {} stars behind it and the nebula's dark places: \
                         median transmission {:.0}%, thickest {:.0}%",
                        behind.len(),
                        100.0 * middle,
                        100.0 * thickest
                    )));
                }
                None => report(Event::Note(
                    "too few stars to map the dust; nothing dims behind it",
                )),
            }
        }
        if options.max_stars.is_some() {
            report(Event::Note(&format!(
                "the {} brightest stars fly; the rest {}",
                summary.flying_stars,
                match options.small_stars {
                    SmallStars::Drop => "are dropped",
                    SmallStars::Field => "stay on the star field",
                }
            )));
        }

        let (sin, cos) = options.truck_angle_deg.to_radians().sin_cos();
        let shot = Shot {
            focus,
            dolly: options.dolly,
            truck: (options.truck * cos, -options.truck * sin),
            start: options.start,
            pan: options.pan,
            rotation: (
                options.rotate_deg.0.to_radians(),
                options.rotate_deg.1.to_radians(),
            ),
            zoom: options.zoom,
            zoom_end: options.zoom_end,
            width: options.size.0,
            height: options.size.1,
            frames: options.frames(),
            easing: options.easing,
            quality: options.quality,
            growth_limit: options.growth_limit,
            fade_from: options.fade_from,
            tour: options
                .tour
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
                })
                .collect(),
            glide: options.tour_glide,
            ..Shot::default()
        };
        if !shot.tour.is_empty() {
            report(Event::Note(&format!(
                "a tour of {} stops over {:.1} s",
                shot.tour.len(),
                options.seconds()
            )));
        }
        let (shot, fitted) = shot.fitted(&scene);
        if shot.zoom > options.zoom.max(1.0) {
            report(Event::Note(&format!(
                "first frame zoomed in to {:.2} so {} stays inside the image",
                shot.zoom,
                if shot.tour.is_empty() {
                    "the turned frame"
                } else {
                    "every view of the tour"
                }
            )));
        }
        if shot.pan < options.pan {
            report(Event::Note(&format!(
                "pan reduced to {:.2} so the far stars stay inside the image",
                shot.pan
            )));
        }
        if shot.lead < 1.0 {
            report(Event::Note(&format!(
                "sideways travel toward the focus point comes later (lead {:.2}) so the far \
                 stars stay inside the image",
                shot.lead
            )));
        }
        let zoomed: Vec<String> = shot
            .tour
            .iter()
            .zip(&options.tour)
            .enumerate()
            .filter(|(_, (fitted, asked))| fitted.zoom > asked.zoom * 1.001)
            .map(|(index, (fitted, asked))| {
                format!("stop {} by {:.2}", index + 1, fitted.zoom / asked.zoom)
            })
            .collect();
        if !zoomed.is_empty() {
            report(Event::Note(&format!(
                "zoomed in so every view stays inside the image: {}",
                zoomed.join(", ")
            )));
        }
        if !shot.tour.is_empty() && fitted < 1.0 && options.tour.iter().any(|stop| stop.pan > 0.0) {
            report(Event::Note(&format!(
                "each stop's pan reduced to {:.2} of what it asked so the far stars stay inside \
                 the image",
                fitted
            )));
        }
        if shot.tour.is_empty() && fitted < 1.0 && options.truck != 0.0 {
            report(Event::Note(&format!(
                "truck reduced to {:.4} of the distance so the far stars stay inside the image",
                options.truck * fitted
            )));
        }

        let overlay = prepare_overlay(options, wcs, &scene, &lifted, &shot, &mut summary)?;
        if summary.labelled_objects > 0 {
            report(Event::Note(&format!(
                "{} catalogued objects in the field to label",
                summary.labelled_objects
            )));
        }
        Ok(Self {
            scene,
            shot,
            overlay,
            summary,
            fps: options.fps,
        })
    }

    /// The number of frames.
    pub fn frames(&self) -> usize {
        self.shot.frames
    }

    pub fn fps(&self) -> u32 {
        self.fps
    }

    /// The frame size, pixels.
    pub fn size(&self) -> (usize, usize) {
        (self.shot.width, self.shot.height)
    }

    /// The camera move as fitted to the image.
    pub fn shot(&self) -> &Shot {
        &self.shot
    }

    /// The scene cut into depths.
    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    pub fn summary(&self) -> &Summary {
        &self.summary
    }

    /// Settings for an encoder of this video, with a bitrate of half a bit
    /// per pixel per frame.
    pub fn video_settings(&self) -> VideoSettings {
        let (width, height) = self.size();
        VideoSettings {
            width: width as u32,
            height: height as u32,
            fps: self.fps,
            bitrate: (width * height * self.fps as usize / 2).min(u32::MAX as usize) as u32,
        }
    }

    /// Frame `index`, with its labels, as display RGB.
    pub fn frame(&self, index: usize) -> RgbImage {
        let mut image = self.shot.render(&self.scene, index).to_display_rgb8();
        if let Some(overlay) = &self.overlay {
            overlay.draw(index, &self.shot.view(&self.scene, index), &mut image);
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
    /// The stops alone, for [`ParallaxOptions::tour`].
    pub fn tour(&self) -> Vec<TourStop> {
        self.stops.iter().map(|planned| planned.stop).collect()
    }
}

/// Plan a tour of the catalogued objects in a `width` × `height` image
/// whose sky `wcs` gives, as `auto` asks, for frames of `frame` output
/// pixels, from the object catalog at `objects` or the standard places.
/// The plan is the caller's to edit, dropping, moving or changing stops,
/// before passing its stops as [`ParallaxOptions::tour`] and its `focus`
/// as [`ParallaxOptions::focus`].
pub fn plan_tour(
    wcs: &Wcs,
    (width, height): (u32, u32),
    objects: Option<&Path>,
    frame: (usize, usize),
    auto: &AutoTour,
) -> Result<PlannedTour, Error> {
    let path = seiza::data_paths::objects(objects).map_err(|error| {
        Error::Invalid(format!(
            "planning a tour needs the object catalog (seiza setup): {error}"
        ))
    })?;
    let placed = open_objects(&path)?
        .objects_in_footprint(wcs, (width, height))
        .map_err(|error| Error::Invalid(error.to_string()))?;
    let size = (width as usize, height as usize);
    let targets = tour::targets(&placed, size, auto.targets);
    let Some(best) = targets.first() else {
        return Err(Error::Invalid(
            "no catalogued objects in the field to tour".into(),
        ));
    };
    Ok(PlannedTour {
        stops: tour::plan(&targets, size, frame, auto),
        focus: (best.x, best.y),
        focus_name: best.name.clone(),
    })
}

/// A stop written as [`parse_stop`] reads it.
pub fn format_stop(stop: &TourStop) -> String {
    let mut text = match stop.focus {
        Some((x, y)) => format!("{x:.0},{y:.0}"),
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
            let value = format!("{value:.3}");
            let value = value.trim_end_matches('0').trim_end_matches('.');
            text.push_str(&format!(" {key}={value}"));
        }
    }
    text
}

/// `options` with a tour of the catalogued objects in a `dimensions` image
/// planned as `auto` asks, and the nebula's distance taken at the target
/// most worth a visit unless `focus` says otherwise.
fn plan_options(
    options: &ParallaxOptions,
    auto: &AutoTour,
    wcs: &Wcs,
    dimensions: (u32, u32),
    report: &mut dyn FnMut(Event),
) -> Result<ParallaxOptions, Error> {
    let plan = plan_tour(
        wcs,
        dimensions,
        options.objects.as_deref(),
        options.size,
        auto,
    )?;
    let mut planned = ParallaxOptions {
        tour: plan.tour(),
        focus: options.focus.or(Some(plan.focus)),
        ..options.clone()
    };
    planned.auto_tour = None;
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

/// The labels over the video, if any were asked for: the caller's own, the
/// catalogued objects, and the watermark.
fn prepare_overlay(
    options: &ParallaxOptions,
    wcs: &Wcs,
    scene: &Scene,
    lifted: &[(String, (f64, f64), f64)],
    shot: &Shot,
    summary: &mut Summary,
) -> Result<Option<Overlay>, Error> {
    let color = Rgb(options.label_color);
    let mut marks = overlay::custom_marks(&options.labels, color, scene);
    if options.overlay {
        let path = seiza::data_paths::objects(options.objects.as_deref()).map_err(|error| {
            Error::Invalid(format!(
                "labelling objects needs the object catalog (seiza setup): {error}"
            ))
        })?;
        let catalog = open_objects(&path)?;
        let dimensions = (scene.width() as u32, scene.height() as u32);
        let found = overlay::catalog_marks(&catalog, wcs, dimensions, scene, lifted)?;
        summary.labelled_objects = found.len();
        marks.extend(found);
    }
    if marks.is_empty() && options.watermark.is_none() {
        return Ok(None);
    }
    let mut overlay = Overlay::new(
        marks,
        options.watermark.clone(),
        options.overlay_density,
        wcs.scale_arcsec_per_px(),
        options.size,
    )?;
    overlay.plan(shot, scene, options.fps as f64);
    Ok(Some(overlay))
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
    options: &ParallaxOptions,
    wcs: &Wcs,
    (width, height): (usize, usize),
    focus: (f64, f64),
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
    Ok(found
        .distance()
        .map(|distance| (name, distance.distance_pc, distance.basis.as_str())))
}

/// Catalogued galaxies in the image large enough to see, largest first,
/// leaving out one at the focus point, which is the target, and each only
/// once: catalogs list some galaxies twice, a little apart.
fn galaxies_in_image(
    options: &ParallaxOptions,
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
            distance_pc: Some(400.0),
            objects: Some(directory.join("no-objects.bin")),
            star_distances: Some(star_distances),
            online: false,
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
            ..ParallaxOptions::default()
        }
    }

    #[test]
    fn a_prepared_video_hands_every_frame_to_the_callers_sink() {
        let directory = tempfile::tempdir().unwrap();
        let (starless, stars, wcs, distances) = field(directory.path());
        let options = options(distances, directory.path());
        let mut notes = Vec::new();
        let parallax = Parallax::prepare(&starless, &stars, &wcs, &options, &mut |event| {
            if let Event::Note(note) = event {
                notes.push(note.to_string());
            }
        })
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
        let parallax = Parallax::prepare(&starless, &stars, &wcs, &options, &mut |_| {}).unwrap();
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
            },
        ] {
            let text = format_stop(&stop);
            assert_eq!(parse_stop(&text).unwrap(), stop, "{text}");
        }
        assert_eq!(format_stop(&TourStop::default()), "whole travel=5");
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
        for options in [
            ParallaxOptions {
                dolly: 1.0,
                ..ParallaxOptions::default()
            },
            ParallaxOptions {
                size: (161, 120),
                ..ParallaxOptions::default()
            },
            ParallaxOptions {
                fps: 0,
                ..ParallaxOptions::default()
            },
            ParallaxOptions {
                overlay_density: 2.0,
                ..ParallaxOptions::default()
            },
        ] {
            assert!(options.check().is_err(), "{options:?}");
        }
        assert!(ParallaxOptions::default().check().is_ok());
    }
}
