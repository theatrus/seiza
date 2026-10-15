//! Parallax fly-through videos (`seiza-parallax`), as `seiza parallax-video`
//! makes them.
//!
//! [`seiza_parallax_prepare_json`] does the work that comes once: it splits
//! the image with StarXTerminator if no split is given, plate-solves it if
//! no WCS is given, finds the stars' distances, and cuts the scene into
//! depths. The prepared video then draws frames into the caller's own
//! buffers ([`seiza_parallax_render_frame`]), hands every frame in turn to
//! a callback, the hook for a platform video encoder
//! ([`seiza_parallax_render_frames`]), or writes a video file
//! ([`seiza_parallax_write_video_json`]).

use super::{
    SeizaCancelSignal, SipResponse, WcsResponse, blind_solve_path, clear_error, ffi_result,
    owned_json, rc_astro_cli, required_str,
};
use image::{Rgb32FImage, RgbImage};
use seiza::wcs::{Sip, Wcs};
use seiza_parallax::{
    Corner, CustomLabel, DepthPlane, DistanceOverride, Dust, DustMap, Easing, Event, FfmpegSink,
    FrameFn, FrameSink, Parallax, ParallaxOptions, PlaneMask, PngSequence, Quality, SmallStars,
    Start,
};
use seiza_parallax::{FitSummary, SceneOptions, VideoOptions};
use serde::{Deserialize, Serialize};
use std::ffi::{CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

/// Frame pixels as 3 bytes each, red, green, blue.
pub const SEIZA_PIXEL_FORMAT_RGB8: u32 = 0;
/// Frame pixels as 4 bytes each, red, green, blue and an opaque alpha.
pub const SEIZA_PIXEL_FORMAT_RGBA8: u32 = 1;
/// Frame pixels as 4 bytes each, blue, green, red and an opaque alpha: the
/// order of Core Video's `kCVPixelFormatType_32BGRA`, Direct2D and Media
/// Foundation's RGB32.
pub const SEIZA_PIXEL_FORMAT_BGRA8: u32 = 2;

/// An opaque prepared parallax video. Release it with
/// [`seiza_parallax_free`]. Its frames may be drawn from several threads at
/// once; do not free it while any call using it is running. Videos made
/// from it by [`seiza_parallax_reconfigure_json`] and
/// [`seiza_parallax_with_dust_map`] share its prepared scene, which lives
/// until the last of them is freed.
pub struct SeizaParallax {
    video: Parallax,
}

/// Progress for the parallax functions: one event as JSON, valid only for
/// the call, plus the caller's context pointer. Events are
/// `{"kind":"note","message":...}` for a step done or a choice made,
/// `{"kind":"warning","message":...}` for something the video goes on
/// without, `{"kind":"split","fraction":...}` as StarXTerminator works, and
/// `{"kind":"frame","done":...,"total":...}` as frames are drawn. Called on
/// the thread that made the call.
pub type SeizaParallaxEventCallback = Option<unsafe extern "C" fn(*const c_char, *mut c_void)>;

/// One frame for [`seiza_parallax_render_frames`]: its pixels in the
/// requested format, `stride` bytes a row, borrowed for the call only;
/// its width and height; its index and the number of frames; its time in
/// seconds from the start; and the caller's context. Return 0 to go on,
/// anything else to stop the video. Called on the thread that made the
/// call, frames in order.
pub type SeizaParallaxFrameCallback =
    Option<unsafe extern "C" fn(*const u8, usize, u32, u32, u32, u32, f64, *mut c_void) -> i32>;

/// A parallax request's fields that make the scene: the images, the plate
/// solution, the catalogs and how the stars are placed. Changing any of
/// them needs a new [`seiza_parallax_prepare_json`];
/// [`seiza_parallax_reconfigure_json`] refuses them.
const SCENE_FIELDS: [&str; 27] = [
    "image",
    "starless",
    "stars",
    "rcAstroExecutable",
    "rcAstroHost",
    "wcs",
    "catalogDirectory",
    "minimumScaleArcsecPerPixel",
    "maximumScaleArcsecPerPixel",
    "distanceParsecs",
    "distanceFocus",
    "unmatchedDistanceParsecs",
    "objects",
    "objectDistances",
    "starDistances",
    "gaiaMaxMagnitude",
    "gaiaCache",
    "online",
    "maxStars",
    "smallStars",
    "keepGalaxies",
    "dust",
    "dustOpacity",
    "dustMap",
    "planes",
    "autoPlanes",
    "distanceOverrides",
];

/// Everything a parallax video takes, as one JSON object: the scene's
/// fields ([`SCENE_FIELDS`]) and the video's. Fields left out keep `seiza
/// parallax-video`'s defaults; a field not known is an error, so a typo
/// cannot silently run the defaults.
#[derive(Debug, Default)]
struct ParallaxRequest {
    scene: SceneRequest,
    video: VideoRequest,
}

/// The fields that make the scene.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SceneRequest {
    /// The stretched image. Split with StarXTerminator when `starless` and
    /// `stars` are absent, and plate-solved when `wcs` is absent.
    image: Option<PathBuf>,
    /// The image split into its starless image and unscreened stars.
    starless: Option<PathBuf>,
    stars: Option<PathBuf>,
    rc_astro_executable: Option<PathBuf>,
    rc_astro_host: Option<String>,
    /// The image's plate solution, as `seiza_solve_image_json` returns it.
    wcs: Option<WcsRequest>,
    /// Where Seiza's catalogs are: the star tiles and blind index for
    /// solving, the object catalog, and the star distance file.
    catalog_directory: Option<PathBuf>,
    minimum_scale_arcsec_per_pixel: Option<f64>,
    maximum_scale_arcsec_per_pixel: Option<f64>,
    distance_parsecs: Option<f64>,
    /// The image point whose object or nearby stars give the nebula's
    /// distance; the camera's destination if absent.
    distance_focus: Option<[f64; 2]>,
    unmatched_distance_parsecs: Option<f64>,
    objects: Option<PathBuf>,
    object_distances: Option<PathBuf>,
    star_distances: Option<PathBuf>,
    gaia_max_magnitude: Option<f32>,
    gaia_cache: Option<PathBuf>,
    online: Option<bool>,
    max_stars: Option<usize>,
    /// "drop" or "field".
    small_stars: Option<String>,
    keep_galaxies: Option<bool>,
    dust: Option<bool>,
    dust_opacity: Option<f32>,
    /// Dust drawn as an image, white where opaque, in place of the dust
    /// mapped from the star counts.
    dust_map: Option<PathBuf>,
    /// Regions of the image lifted onto planes at their own distances.
    planes: Option<Vec<PlaneRequest>>,
    /// Lift the catalogued nebulae at distances of their own onto planes.
    auto_planes: Option<bool>,
    /// Distances for catalogued objects in place of the catalogue's.
    distance_overrides: Option<Vec<DistanceOverrideRequest>>,
}

/// A distance for a catalogued object, by name, in place of the
/// catalogue's.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DistanceOverrideRequest {
    object: String,
    distance_parsecs: f64,
}

/// A depth plane: a region of the image at a distance of its own.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct PlaneRequest {
    /// An image, white where the image belongs to the plane and black
    /// elsewhere, at any size of the image's shape.
    mask: PathBuf,
    distance_parsecs: f64,
    /// What to call it; the mask's file name if absent.
    name: Option<String>,
}

/// The fields that film the scene: the camera, the output and the labels.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct VideoRequest {
    /// The image point the camera flies toward; it moves the camera only.
    focus: Option<[f64; 2]>,
    /// "focus" or "whole".
    start: Option<String>,
    dolly: Option<f64>,
    truck: Option<f64>,
    truck_angle_degrees: Option<f64>,
    pan: Option<f64>,
    zoom: Option<f64>,
    zoom_end: Option<f64>,
    /// The frame's turn at the first and last frames, degrees
    /// anticlockwise, taken as given: [0, 360] turns once anticlockwise and
    /// [0, -360] once clockwise.
    rotate_degrees: Option<[f64; 2]>,
    /// "inOut" or "linear".
    easing: Option<String>,
    /// "standard" or "high".
    quality: Option<String>,
    growth_limit: Option<f64>,
    fade_from: Option<f64>,
    /// A tour of stops instead of the single move.
    tour: Option<Vec<StopRequest>>,
    /// Plan a tour of the catalogued objects in the field when `tour` is
    /// empty: `{targets, hold, motion}`, each optional.
    auto_tour: Option<AutoTourRequest>,
    /// How fast a tour moves on through a stop it holds at, as a share of
    /// its pace between stops; 0 comes to rest.
    tour_glide: Option<f64>,
    /// Show each stop's title low in the frame as the camera drifts
    /// through it.
    tour_titles: Option<bool>,
    /// End the tour where it began, so the video loops.
    tour_loop: Option<bool>,
    /// "720p", "1080p", "1440p" or "4k", each with "-portrait" for the tall
    /// form, or "WIDTHxHEIGHT".
    size: Option<String>,
    seconds: Option<f64>,
    fps: Option<u32>,
    overlay: Option<bool>,
    overlay_density: Option<f64>,
    labels: Option<Vec<LabelRequest>>,
    /// "#RRGGBB".
    label_color: Option<String>,
    /// true for "Rendered with seiza.fyi", the text to write, or
    /// `{text, image, corner, size, opacity}`.
    watermark: Option<WatermarkRequest>,
    /// Cards laid over the start and end of the video, and how long each
    /// shows, seconds.
    intro_card: Option<PathBuf>,
    outro_card: Option<PathBuf>,
    card_seconds: Option<f64>,
}

/// A JSON object's fields.
type Fields = serde_json::Map<String, serde_json::Value>;

/// A JSON object's fields, the scene's and the video's apart.
fn split_fields(json: &str, what: &str) -> Result<(Fields, Fields), String> {
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|error| format!("invalid {what}: {error}"))?;
    let serde_json::Value::Object(fields) = value else {
        return Err(format!("the {what} must be a JSON object"));
    };
    let (mut scene, mut video) = (Fields::new(), Fields::new());
    for (key, value) in fields {
        if SCENE_FIELDS.contains(&key.as_str()) {
            // A null scene field is left at its default, as it would be
            // left out, so a reconfiguration may carry it.
            if !value.is_null() {
                scene.insert(key, value);
            }
        } else {
            video.insert(key, value);
        }
    }
    Ok((scene, video))
}

/// A prepare request, its scene's fields and its video's apart.
fn parse_request(json: &str) -> Result<ParallaxRequest, String> {
    let what = "parallax request JSON";
    let (scene, video) = split_fields(json, what)?;
    Ok(ParallaxRequest {
        scene: serde_json::from_value(serde_json::Value::Object(scene))
            .map_err(|error| format!("invalid {what}: {error}"))?,
        video: serde_json::from_value(serde_json::Value::Object(video))
            .map_err(|error| format!("invalid {what}: {error}"))?,
    })
}

/// A reconfiguration's video fields, refusing any that would change the
/// prepared scene.
fn parse_video(json: &str) -> Result<VideoRequest, String> {
    let what = "parallax settings JSON";
    let (scene, video) = split_fields(json, what)?;
    if let Some(field) = scene.keys().next() {
        return Err(format!(
            "\"{field}\" changes the prepared scene (its images, plate solution, catalogs, \
             distances or star placement); prepare a new video for it"
        ));
    }
    serde_json::from_value(serde_json::Value::Object(video))
        .map_err(|error| format!("invalid {what}: {error}"))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct LabelRequest {
    x: f64,
    y: f64,
    #[serde(default)]
    radius: f64,
    text: String,
}

/// A stop on a tour: `focus` `[x, y]` (absent for the image's centre),
/// `dolly`, `zoom`, `rotateDegrees`, `pan`, and `travel` and `hold` in
/// seconds.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct StopRequest {
    /// The target the stop visits, as a plan names it; not used.
    #[serde(default)]
    #[allow(dead_code)]
    name: Option<String>,
    focus: Option<[f64; 2]>,
    dolly: Option<f64>,
    zoom: Option<f64>,
    rotate_degrees: Option<f64>,
    pan: Option<f64>,
    travel: Option<f64>,
    hold: Option<f64>,
    spin_degrees: Option<f64>,
    push: Option<f64>,
    /// Shown while the camera drifts through the stop, with `tourTitles`.
    title: Option<String>,
    /// An image shown as the title would be, in its place, whether or not
    /// titles are shown.
    card: Option<PathBuf>,
    /// The distance of the plane the focus point is centred on, parsecs;
    /// that of the plane the image shows there if absent.
    depth_parsecs: Option<f64>,
}

/// How to plan a tour: how many targets (every one worth a visit if
/// absent), the seconds at each, and how much it turns and pans.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AutoTourRequest {
    targets: Option<usize>,
    hold: Option<f64>,
    motion: Option<f64>,
}

impl AutoTourRequest {
    fn auto_tour(&self) -> seiza_parallax::AutoTour {
        let defaults = seiza_parallax::AutoTour::default();
        seiza_parallax::AutoTour {
            targets: self.targets,
            hold: self.hold.unwrap_or(defaults.hold),
            motion: self.motion.unwrap_or(defaults.motion),
        }
    }
}

/// A watermark's text: true for "Rendered with seiza.fyi", or the text.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum WatermarkText {
    Shown(bool),
    Text(String),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum WatermarkRequest {
    Plain(WatermarkText),
    Styled(WatermarkStyle),
}

/// A watermark of text, an image or both, each optional.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct WatermarkStyle {
    text: Option<WatermarkText>,
    image: Option<PathBuf>,
    /// "bottomRight", "bottomLeft", "topRight" or "topLeft".
    corner: Option<String>,
    /// The image's height as a share of the frame's shorter side.
    size: Option<f64>,
    opacity: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WcsRequest {
    crval: [f64; 2],
    crpix: [f64; 2],
    cd: [[f64; 2]; 2],
    #[serde(default)]
    sip: Option<SipRequest>,
    /// What `crpix` counts from: 0, as the C ABI returns a WCS (and the
    /// default), or 1, as a FITS header does.
    #[serde(default, rename = "pixelOrigin")]
    pixel_origin: Option<u8>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SipRequest {
    order: u8,
    a: Vec<f64>,
    b: Vec<f64>,
    ap: Vec<f64>,
    bp: Vec<f64>,
}

/// How to write a prepared video to a file.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct WriteRequest {
    /// The video (.mp4), or a directory of PNG frames with the "png"
    /// encoder.
    output: PathBuf,
    /// "auto" (ffmpeg), "ffmpeg" or "png".
    #[serde(default)]
    encoder: Option<String>,
    /// The ffmpeg program; "ffmpeg" on PATH if absent.
    #[serde(default)]
    ffmpeg: Option<PathBuf>,
    /// "h264" (the default) or "hevc".
    #[serde(default)]
    codec: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SummaryResponse {
    schema_version: u32,
    frames: usize,
    fps: u32,
    width: usize,
    height: usize,
    detected_stars: usize,
    gaia_stars: usize,
    hipparcos_stars: usize,
    gaia_matches: usize,
    with_distance: usize,
    background_distance_parsecs: f64,
    background_basis: String,
    /// The image point the nebula's distance was taken at.
    background_focus: [f64; 2],
    unmatched_distance_parsecs: f64,
    flying_stars: usize,
    galaxies_lifted: Vec<String>,
    /// The depth planes lifted.
    planes: Vec<PlaneResponse>,
    dust_transmission: Option<DustResponse>,
    /// Whether the dust came from a given map rather than the star counts.
    dust_given: bool,
    /// Stars seen where the dust is opaque, flying in front of the nebula.
    stars_in_front_of_dust: usize,
    labelled_objects: usize,
    /// How the fit changed the framing asked for.
    fit: FitResponse,
    wcs: WcsResponse,
}

/// A depth plane lifted: its name, distance, how that was found, the
/// share of the image it covers, and whether it blocks light behind it.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PlaneResponse {
    name: String,
    distance_parsecs: f64,
    basis: String,
    coverage: f64,
    absorbs: bool,
    /// The catalogued object whose distance places it, which
    /// `distanceOverrides` moves it by; null for a given mask or a cloud of
    /// dust no catalogued object names.
    object: Option<String>,
}

/// A setting as asked and as the video uses it.
#[derive(Serialize)]
struct AskedUsed {
    asked: f64,
    used: f64,
}

impl From<(f64, f64)> for AskedUsed {
    fn from((asked, used): (f64, f64)) -> Self {
        Self { asked, used }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FitResponse {
    zoom: AskedUsed,
    pan: AskedUsed,
    lead: AskedUsed,
    truck: AskedUsed,
    stops: Vec<StopFitResponse>,
    /// Whether every frame stays inside the image.
    inside: bool,
}

#[derive(Serialize)]
struct StopFitResponse {
    zoom: AskedUsed,
    pan: AskedUsed,
    dolly: AskedUsed,
}

impl From<&FitSummary> for FitResponse {
    fn from(fit: &FitSummary) -> Self {
        Self {
            zoom: fit.zoom.into(),
            pan: fit.pan.into(),
            lead: fit.lead.into(),
            truck: fit.truck.into(),
            stops: fit
                .stops
                .iter()
                .map(|stop| StopFitResponse {
                    zoom: stop.zoom.into(),
                    pan: stop.pan.into(),
                    dolly: stop.dolly.into(),
                })
                .collect(),
            inside: fit.inside,
        }
    }
}

#[derive(Serialize)]
struct DustResponse {
    median: f32,
    thickest: f32,
}

/// Send one event, as JSON, to `events`.
fn send(events: SeizaParallaxEventCallback, context: usize, event: serde_json::Value) {
    if let Some(callback) = events
        && let Ok(json) = CString::new(event.to_string())
    {
        unsafe { callback(json.as_ptr(), context as *mut c_void) };
    }
}

fn event_json(event: Event) -> serde_json::Value {
    match event {
        Event::Note(message) => serde_json::json!({ "kind": "note", "message": message }),
        Event::Warning(message) => serde_json::json!({ "kind": "warning", "message": message }),
        Event::Frame { done, total } => {
            serde_json::json!({ "kind": "frame", "done": done, "total": total })
        }
    }
}

/// A stretched image as RGB with values 0 to 1: PNG, JPEG or TIFF in its
/// EXIF orientation, or FITS or XISF.
fn open_display(path: &Path) -> Result<Rgb32FImage, String> {
    Ok(seiza_image_io::open(path)
        .map_err(|error| error.to_string())?
        .to_rgb())
}

fn choice<T: Copy>(
    value: &Option<String>,
    name: &str,
    choices: &[(&str, T)],
    default: T,
) -> Result<T, String> {
    let Some(value) = value else {
        return Ok(default);
    };
    choices
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(value))
        .map(|(_, choice)| *choice)
        .ok_or_else(|| {
            let keys: Vec<&str> = choices.iter().map(|(key, _)| *key).collect();
            format!("{name} must be one of {}; got {value:?}", keys.join(", "))
        })
}

/// The video's options from the request, over the defaults.
fn options(request: &ParallaxRequest) -> Result<ParallaxOptions, String> {
    Ok(ParallaxOptions {
        scene: scene_options(&request.scene)?,
        video: video_options(&request.video)?,
    })
}

/// The scene's options from the request, over the defaults.
fn scene_options(request: &SceneRequest) -> Result<SceneOptions, String> {
    let defaults = SceneOptions::default();
    let catalogs = request.catalog_directory.as_deref();
    // A star distance file the request names must be there; one beside the
    // other catalogs is taken only if it is.
    let star_distances = request.star_distances.clone().or_else(|| {
        catalogs
            .map(|directory| directory.join("star-distances.bin"))
            .filter(|path| path.is_file())
    });
    Ok(SceneOptions {
        distance_pc: request.distance_parsecs,
        distance_focus: request.distance_focus.map(|[x, y]| (x, y)),
        unmatched_distance_pc: request.unmatched_distance_parsecs,
        objects: request
            .objects
            .clone()
            .or_else(|| catalogs.map(Path::to_path_buf)),
        object_distances: request.object_distances.clone(),
        star_distances,
        gaia_max_mag: request.gaia_max_magnitude.unwrap_or(defaults.gaia_max_mag),
        gaia_cache: request.gaia_cache.clone(),
        online: request.online.unwrap_or(defaults.online),
        max_stars: request.max_stars,
        small_stars: choice(
            &request.small_stars,
            "smallStars",
            &[("drop", SmallStars::Drop), ("field", SmallStars::Field)],
            defaults.small_stars,
        )?,
        keep_galaxies: request.keep_galaxies.unwrap_or(defaults.keep_galaxies),
        dust: request.dust.unwrap_or(defaults.dust),
        dust_opacity: request.dust_opacity.unwrap_or(defaults.dust_opacity),
        dust_map: request.dust_map.as_deref().map(DustMap::open).transpose()?,
        planes: request
            .planes
            .iter()
            .flatten()
            .map(|plane| {
                Ok(DepthPlane {
                    mask: PlaneMask::open(&plane.mask)?,
                    distance_pc: plane.distance_parsecs,
                    name: plane.name.clone().unwrap_or_else(|| {
                        plane.mask.file_stem().map_or_else(
                            || plane.mask.display().to_string(),
                            |stem| stem.to_string_lossy().into_owned(),
                        )
                    }),
                })
            })
            .collect::<Result<_, String>>()?,
        auto_planes: request.auto_planes.unwrap_or(defaults.auto_planes),
        distance_overrides: request
            .distance_overrides
            .iter()
            .flatten()
            .map(|given| DistanceOverride {
                object: given.object.clone(),
                distance_pc: given.distance_parsecs,
            })
            .collect(),
    })
}

/// The video's options from the request, over the defaults.
fn video_options(request: &VideoRequest) -> Result<VideoOptions, String> {
    let defaults = VideoOptions::default();
    let unstyled = WatermarkStyle::default();
    let (watermark_text, style) = match &request.watermark {
        None => (None, &unstyled),
        Some(WatermarkRequest::Plain(text)) => (Some(text), &unstyled),
        Some(WatermarkRequest::Styled(style)) => (style.text.as_ref(), style),
    };
    Ok(VideoOptions {
        focus: request.focus.map(|[x, y]| (x, y)),
        start: choice(
            &request.start,
            "start",
            &[("focus", Start::Focus), ("whole", Start::Whole)],
            defaults.start,
        )?,
        dolly: request.dolly.unwrap_or(defaults.dolly),
        truck: request.truck.unwrap_or(defaults.truck),
        truck_angle_deg: request
            .truck_angle_degrees
            .unwrap_or(defaults.truck_angle_deg),
        pan: request.pan.unwrap_or(defaults.pan),
        zoom: request.zoom.unwrap_or(defaults.zoom),
        zoom_end: request.zoom_end.unwrap_or(defaults.zoom_end),
        rotate_deg: request
            .rotate_degrees
            .map_or(defaults.rotate_deg, |[first, last]| (first, last)),
        easing: choice(
            &request.easing,
            "easing",
            &[("inOut", Easing::InOut), ("linear", Easing::Linear)],
            defaults.easing,
        )?,
        quality: choice(
            &request.quality,
            "quality",
            &[("standard", Quality::Standard), ("high", Quality::High)],
            defaults.quality,
        )?,
        growth_limit: request.growth_limit.unwrap_or(defaults.growth_limit),
        fade_from: request.fade_from.unwrap_or(defaults.fade_from),
        tour: request
            .tour
            .iter()
            .flatten()
            .map(|stop| {
                let base = seiza_parallax::TourStop::default();
                seiza_parallax::TourStop {
                    focus: stop.focus.map(|[x, y]| (x, y)),
                    dolly: stop.dolly.unwrap_or(base.dolly),
                    zoom: stop.zoom.unwrap_or(base.zoom),
                    rotate_deg: stop.rotate_degrees.unwrap_or(base.rotate_deg),
                    pan: stop.pan.unwrap_or(base.pan),
                    travel: stop.travel.unwrap_or(base.travel),
                    hold: stop.hold.unwrap_or(base.hold),
                    spin_deg: stop.spin_degrees.unwrap_or(base.spin_deg),
                    push: stop.push.unwrap_or(base.push),
                    title: stop.title.clone().filter(|title| !title.is_empty()),
                    card: stop.card.clone(),
                    depth_pc: stop.depth_parsecs,
                }
            })
            .collect(),
        auto_tour: request.auto_tour.as_ref().map(AutoTourRequest::auto_tour),
        tour_glide: request.tour_glide.unwrap_or(defaults.tour_glide),
        tour_titles: request.tour_titles.unwrap_or(defaults.tour_titles),
        tour_loop: request.tour_loop.unwrap_or(defaults.tour_loop),
        size: match &request.size {
            Some(size) => seiza_parallax::parse_frame_size(size)?,
            None => defaults.size,
        },
        seconds: request.seconds.unwrap_or(defaults.seconds),
        fps: request.fps.unwrap_or(defaults.fps),
        overlay: request.overlay.unwrap_or(defaults.overlay),
        overlay_density: request.overlay_density.unwrap_or(defaults.overlay_density),
        labels: request
            .labels
            .iter()
            .flatten()
            .map(|label| CustomLabel {
                x: label.x,
                y: label.y,
                radius: label.radius.max(0.0),
                text: label.text.clone(),
            })
            .collect(),
        label_color: match &request.label_color {
            Some(color) => seiza_parallax::parse_color(color)?.0,
            None => defaults.label_color,
        },
        watermark: match watermark_text {
            None | Some(WatermarkText::Shown(false)) => None,
            Some(WatermarkText::Shown(true)) => Some(seiza_parallax::DEFAULT_WATERMARK.into()),
            Some(WatermarkText::Text(text)) => Some(text.clone()),
        },
        watermark_image: style.image.clone(),
        watermark_corner: choice(
            &style.corner,
            "watermark corner",
            &[
                ("bottomRight", Corner::BottomRight),
                ("bottomLeft", Corner::BottomLeft),
                ("topRight", Corner::TopRight),
                ("topLeft", Corner::TopLeft),
            ],
            defaults.watermark_corner,
        )?,
        watermark_size: style.size.unwrap_or(defaults.watermark_size),
        watermark_opacity: style.opacity.unwrap_or(defaults.watermark_opacity),
        intro_card: request.intro_card.clone(),
        outro_card: request.outro_card.clone(),
        card_seconds: request.card_seconds.unwrap_or(defaults.card_seconds),
    })
}

impl WcsRequest {
    fn wcs(&self) -> Result<Wcs, String> {
        let origin = match self.pixel_origin.unwrap_or(0) {
            origin @ (0 | 1) => f64::from(origin),
            other => return Err(format!("wcs pixelOrigin must be 0 or 1, not {other}")),
        };
        Ok(Wcs {
            crval: (self.crval[0], self.crval[1]),
            crpix: (self.crpix[0] - origin, self.crpix[1] - origin),
            cd: self.cd,
            sip: self.sip.as_ref().map(|sip| Sip {
                order: sip.order,
                a: sip.a.clone(),
                b: sip.b.clone(),
                ap: sip.ap.clone(),
                bp: sip.bp.clone(),
            }),
        })
    }
}

/// The starless and stars images, from the request's split or split here
/// with StarXTerminator, and the image to plate-solve.
fn split(
    request: &ParallaxRequest,
    cancel: Option<&seiza_stacking::CancelSignal>,
    events: SeizaParallaxEventCallback,
    context: usize,
) -> Result<(Rgb32FImage, Rgb32FImage, PathBuf), String> {
    if let (Some(starless), Some(stars)) = (&request.scene.starless, &request.scene.stars) {
        let solve = request.scene.image.clone().unwrap_or_else(|| stars.clone());
        return Ok((open_display(starless)?, open_display(stars)?, solve));
    }
    if request.scene.starless.is_some() || request.scene.stars.is_some() {
        return Err("give both starless and stars, or neither".into());
    }
    let path = request
        .scene
        .image
        .clone()
        .ok_or("give an image to split, or starless and stars")?;
    let image = open_display(&path)?;
    let cli = rc_astro_cli(
        request.scene.rc_astro_executable.clone(),
        request.scene.rc_astro_host.clone(),
    )?;
    let linear = seiza_stacking::LinearImage::new(
        image.width() as usize,
        image.height() as usize,
        3,
        image.into_raw(),
    )
    .map_err(|error| error.to_string())?;
    send(
        events,
        context,
        serde_json::json!({ "kind": "note", "message": format!("splitting {} with StarXTerminator", path.display()) }),
    );
    let (starless, stars) = cli
        .split_stars(&linear, cancel, &mut |fraction| {
            send(
                events,
                context,
                serde_json::json!({ "kind": "split", "fraction": fraction }),
            );
        })
        .map_err(|error| error.to_string())?;
    let to_display = |image: seiza_stacking::LinearImage| {
        let data = image
            .data
            .iter()
            .map(|value| value.clamp(0.0, 1.0))
            .collect();
        Rgb32FImage::from_raw(image.width as u32, image.height as u32, data)
            .ok_or_else(|| "StarXTerminator returned an image of the wrong size".to_string())
    };
    Ok((to_display(starless)?, to_display(stars)?, path))
}

/// Prepare a parallax video from a JSON request (see the fields below) and
/// return it, released with [`seiza_parallax_free`], or null with
/// `error_out` set.
///
/// The request gives either `image` (a stretched PNG, JPEG, TIFF, FITS or
/// XISF), which StarXTerminator splits through the `rc-astro` CLI, or
/// `starless` and `stars` (the split, with the stars unscreened, in any of
/// those formats), plus `image` to solve
/// when it differs from `stars`; `rcAstroExecutable` and `rcAstroHost`
/// point at the `rc-astro` CLI and its host when they are not the
/// defaults. Without `wcs` (`{crval, crpix, cd, sip}`
/// as `seiza_solve_image_json` returns it) the image is blind-solved
/// against the catalogs in `catalogDirectory`, between
/// `minimumScaleArcsecPerPixel` and `maximumScaleArcsecPerPixel` (0.1 and
/// 1000 if absent). The other fields are `seiza parallax-video`'s options
/// in camelCase: `focus` `[x, y]` (where the camera flies), `distanceParsecs`,
/// `distanceFocus` `[x, y]` (where the nebula's distance is measured; by
/// default the camera's destination), `unmatchedDistanceParsecs`, `objects`, `objectDistances`,
/// `starDistances`, `gaiaMaxMagnitude`, `gaiaCache`, `online`, `maxStars`,
/// `smallStars` ("drop", "field"), `keepGalaxies`, `dust`, `dustOpacity`,
/// `dustMap` (an image, white where the dust is opaque, black where clear,
/// at any size of the image's shape, its format by its extension: PNG,
/// JPEG, TIFF, FITS or XISF; it takes the place of the dust mapped from the
/// star counts, and a star seen where it is opaque flies in front of the
/// nebula), `planes` (`[{mask, distanceParsecs, name}]`: each
/// `mask` an image, white where the image belongs to the plane and black
/// elsewhere, at any size of the image's shape, whose region is lifted off
/// the starless image onto a plane at `distanceParsecs`, what lay behind it
/// filled in from around it; `name`, the mask's file name if absent, names
/// it in events and the summary), `autoPlanes` (lift the catalogued nebulae
/// that lie at least 15% nearer or farther than the nebula, by a measured
/// distance or one borrowed from an object around them, onto planes of
/// their own; a dark nebula moves with the cloud of dust it lies in, which
/// stays with the nebula when it holds it, and moves as one plane when it
/// lies at least 15% in front, placed by its dark nebulae or the Gaia star
/// counts; a nebula or cloud that would cover more than a third of the
/// image stays with the nebula unless given a distance, and so do the
/// nebulae inside a nebula that stays), `distanceOverrides` (`[{object, distanceParsecs}]`:
/// distances for catalogued objects, by name, in place of the catalogue's,
/// for the nebula's distance and the automatic planes), `start` ("focus",
/// "whole"), `dolly`, `truck`, `truckAngleDegrees`,
/// `pan`, `zoom`, `zoomEnd`, `rotateDegrees` `[first, last]`, `easing`
/// ("inOut", "linear"), `quality` ("standard", "high"), `growthLimit`,
/// `fadeFrom`, `tour` (stops `[{focus, dolly, zoom, rotateDegrees, pan,
/// travel, hold, spinDegrees, push, title, card, depthParsecs}]`, `card`
/// an image shown as the title would be, in its place, each centring
/// its focus point on the plane at `depthParsecs`, or on the plane the
/// image shows there if absent, gliding through held stops
/// at `tourGlide` of the pace between them, the first the opening view and
/// the last's hold a final drift that slows to rest, which replace the
/// single move and set the length; [`seiza_parallax_plan_tour_json`] plans
/// one), `autoTour` (`{targets, hold, motion}`: plan a tour of the
/// catalogued objects and render it), `tourTitles` (show each stop's
/// title low in the frame while the camera drifts through it, fading in
/// and out), `tourLoop` (end where the tour began, adding a last stop at
/// the first's view unless it ends there, so the last frame leads into the
/// first), `size` ("720p", "1080p", "1440p", "4k", each with
/// "-portrait", or "WIDTHxHEIGHT"), `seconds`, `fps`, `overlay`,
/// `overlayDensity`, `labels` (`[{x, y, radius, text}]`), `labelColor`
/// ("#RRGGBB"), `watermark` (true, the text, or `{text, image, corner,
/// size, opacity}`: `text` true or the text, `image` a picture such as a
/// logo, its transparency kept, `corner` "bottomRight" (the default),
/// "bottomLeft", "topRight" or "topLeft", `size` the image's height as a
/// share of the frame's shorter side, 0.06 by default, and `opacity`, 0.8
/// by default), and `introCard` and `outroCard` (images laid over the
/// start and the end, fitted inside the frame, each shown for
/// `cardSeconds`, 3 by default; a looping tour shows them across its
/// join). An image that cannot be read fails here, naming it. An unknown
/// field is an error; a null field takes its default.
///
/// Stars' distances come from the star distance file when installed, else
/// from the Gaia and VizieR archives unless `online` is false. `cancel`
/// stops the preparation, StarXTerminator's split included, between steps
/// and while the camera is fitted; it then returns null with an error
/// saying it stopped (check the signal to tell a cancel from a failure).
/// `events` (nullable) hears each step on the calling thread, with
/// `context` passed through.
///
/// # Safety
///
/// `request_json` must be a NUL-terminated UTF-8 string. `cancel` must be
/// null or a live [`SeizaCancelSignal`] retained until this call returns.
/// When non-null, `error_out` must point to writable storage for one
/// pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_parallax_prepare_json(
    request_json: *const c_char,
    cancel: *const SeizaCancelSignal,
    events: SeizaParallaxEventCallback,
    context: *mut c_void,
    error_out: *mut *mut c_char,
) -> *mut SeizaParallax {
    clear_error(error_out);
    // The callback runs on this thread and the context's lifetime is the
    // caller's promise, so the address crosses catch_unwind as an integer.
    let context = context as usize;
    let cancel_address = cancel as usize;
    ffi_result(error_out, || {
        let request_json = required_str(request_json, "parallax request JSON")?;
        let request = parse_request(&request_json)?;
        let options = options(&request)?;
        options.check().map_err(|error| error.to_string())?;
        let cancellation = unsafe { cancel.as_ref() }
            .map(|signal| seiza_stacking::CancelSignal::from(signal.cancelled.clone()));
        let (starless, stars, solve_path) =
            split(&request, cancellation.as_ref(), events, context)?;
        let wcs = solution(&request, &solve_path, events, context)?;
        let video = Parallax::prepare(
            &starless,
            &stars,
            &wcs,
            &options,
            &mut |event| send(events, context, event_json(event)),
            &|| cancelled(cancel_address),
        )
        .map_err(|error| error.to_string())?;
        Ok(Box::into_raw(Box::new(SeizaParallax { video })))
    })
    .unwrap_or(ptr::null_mut())
}

/// The request's WCS, or a blind solve of the image at `solve_path`
/// against the catalogs in `catalogDirectory`.
fn solution(
    request: &ParallaxRequest,
    solve_path: &Path,
    events: SeizaParallaxEventCallback,
    context: usize,
) -> Result<Wcs, String> {
    if let Some(wcs) = &request.scene.wcs {
        return wcs.wcs();
    }
    let minimum = request.scene.minimum_scale_arcsec_per_pixel.unwrap_or(0.1);
    let maximum = request
        .scene
        .maximum_scale_arcsec_per_pixel
        .unwrap_or(1000.0);
    let solved = blind_solve_path(
        solve_path,
        request.scene.catalog_directory.as_deref(),
        minimum,
        maximum,
        2,
        &seiza::SolveLimits::default(),
    )?;
    let wcs = solved.solution.wcs;
    send(
        events,
        context,
        serde_json::json!({
            "kind": "note",
            "message": format!(
                "solved: {:.3}\"/px, {} stars matched",
                wcs.scale_arcsec_per_px(),
                solved.solution.matched_stars
            ),
        }),
    );
    Ok(wcs)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PlannedStopResponse {
    name: Option<String>,
    focus: Option<[f64; 2]>,
    dolly: f64,
    zoom: f64,
    rotate_degrees: f64,
    pan: f64,
    travel: f64,
    hold: f64,
    spin_degrees: f64,
    push: f64,
    title: Option<String>,
    depth_parsecs: Option<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PlannedTourResponse {
    schema_version: u32,
    focus: [f64; 2],
    focus_name: String,
    seconds: f64,
    tour: Vec<PlannedStopResponse>,
}

impl PlannedTourResponse {
    fn new(plan: &seiza_parallax::PlannedTour) -> Self {
        let seconds = VideoOptions {
            tour: plan.tour(),
            ..VideoOptions::default()
        }
        .seconds();
        Self {
            schema_version: 1,
            focus: [plan.focus.0, plan.focus.1],
            focus_name: plan.focus_name.clone(),
            seconds,
            tour: plan
                .stops
                .iter()
                .map(|planned| PlannedStopResponse {
                    name: planned.name.clone(),
                    focus: planned.stop.focus.map(|(x, y)| [x, y]),
                    dolly: planned.stop.dolly,
                    zoom: planned.stop.zoom,
                    rotate_degrees: planned.stop.rotate_deg,
                    pan: planned.stop.pan,
                    travel: planned.stop.travel,
                    hold: planned.stop.hold,
                    spin_degrees: planned.stop.spin_deg,
                    push: planned.stop.push,
                    title: planned.stop.title.clone(),
                    depth_parsecs: planned.stop.depth_pc,
                })
                .collect(),
        }
    }
}

/// Plan a tour of the catalogued objects in an image, for the caller to
/// edit before making the video. The request is
/// [`seiza_parallax_prepare_json`]'s: the image (`image`, else `stars` or
/// `starless`) gives the size, and is blind-solved without `wcs`; `size`
/// is the video's frame; `objects` or `catalogDirectory` the object
/// catalog; and `autoTour` (`{targets, hold, motion}`, each optional) how
/// to plan. Returns JSON, released with `seiza_string_free`:
/// `{"focus": [x, y], "focusName", "seconds", "tour": [{name, focus,
/// dolly, zoom, rotateDegrees, pan, travel, hold, spinDegrees, push,
/// title, depthParsecs}]}`, each target's stop titled with its name. Its `tour`, with any
/// stops or titles dropped, moved or changed, goes back into a prepare
/// request as it is, and its `focus` as `distanceFocus`. Null with `error_out` set on failure; `events`
/// (nullable) hears the solve. It plans from the catalogue alone; once a
/// scene is prepared, [`seiza_parallax_plan_scene_tour_json`] plans on it,
/// framing each target at the depth of its plane.
///
/// # Safety
///
/// `request_json` must be a NUL-terminated UTF-8 string. When non-null,
/// `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_parallax_plan_tour_json(
    request_json: *const c_char,
    events: SeizaParallaxEventCallback,
    context: *mut c_void,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    let context = context as usize;
    ffi_result(error_out, || {
        let request_json = required_str(request_json, "parallax request JSON")?;
        let request = parse_request(&request_json)?;
        let options = options(&request)?;
        let path = request
            .scene
            .image
            .clone()
            .or_else(|| request.scene.stars.clone())
            .or_else(|| request.scene.starless.clone())
            .ok_or("give the image to plan a tour of")?;
        let dimensions = open_display(&path)?.dimensions();
        let wcs = solution(&request, &path, events, context)?;
        let auto = request
            .video
            .auto_tour
            .as_ref()
            .map(AutoTourRequest::auto_tour)
            .unwrap_or_default();
        let plan = seiza_parallax::plan_tour(
            &wcs,
            dimensions,
            options.scene.objects.as_deref(),
            options.video.size,
            &auto,
        )
        .map_err(|error| error.to_string())?;
        owned_json(&PlannedTourResponse::new(&plan))
    })
    .unwrap_or(ptr::null_mut())
}

/// How to plan a tour on a prepared scene: the frame it is planned for, and
/// how many targets, how long at each and how much motion.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ScenePlanRequest {
    /// "720p", "1080p", "1440p" or "4k", each with "-portrait", or
    /// "WIDTHxHEIGHT"; the video's own frame if absent.
    size: Option<String>,
    auto_tour: Option<AutoTourRequest>,
}

/// Plan a tour of the catalogued objects in a prepared video's image, on
/// its scene: each stop frames its target at the depth of the plane the
/// scene shows it on, so a target on a depth plane is centred there and
/// framed to the size it shows at that depth, as an automatic tour of the
/// same scene would be. [`seiza_parallax_plan_tour_json`] plans from the
/// catalogue alone, without the scene's planes.
///
/// `settings_json` is `{"size", "autoTour": {targets, hold, motion}}`, each
/// optional: `size` the frame the tour is for ("720p", "1080p", "1440p",
/// "4k", each with "-portrait", or "WIDTHxHEIGHT"; the video's own if
/// absent), and `autoTour` how to plan. Returns
/// [`seiza_parallax_plan_tour_json`]'s JSON, released with
/// `seiza_string_free`, its stops carrying their planes' `depthParsecs`.
/// Its `tour`, with any stops or titles dropped, moved or changed, goes
/// into [`seiza_parallax_reconfigure_json`] as it is. The scene keeps the
/// distance it was prepared with, so the plan's `focus`, where an
/// automatic plan would take the nebula's distance, matters only to a new
/// preparation. Null with `error_out` set on failure.
///
/// # Safety
///
/// `video` must be a live pointer from [`seiza_parallax_prepare_json`],
/// [`seiza_parallax_reconfigure_json`], [`seiza_parallax_with_dust_map`] or
/// [`seiza_parallax_with_dust_map_json`]. `settings_json` must be a
/// NUL-terminated UTF-8 string. When non-null, `error_out` must point to
/// writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_parallax_plan_scene_tour_json(
    video: *const SeizaParallax,
    settings_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let video = &unsafe { video.as_ref() }.ok_or("video is null")?.video;
        let settings_json = required_str(settings_json, "tour plan settings JSON")?;
        let request: ScenePlanRequest = serde_json::from_str(&settings_json)
            .map_err(|error| format!("invalid tour plan settings JSON: {error}"))?;
        let frame = match &request.size {
            Some(size) => seiza_parallax::parse_frame_size(size)?,
            None => video.size(),
        };
        let auto = request
            .auto_tour
            .as_ref()
            .map(AutoTourRequest::auto_tour)
            .unwrap_or_default();
        auto.check()?;
        let plan = video
            .plan_tour(&auto, frame)
            .map_err(|error| error.to_string())?;
        owned_json(&PlannedTourResponse::new(&plan))
    })
    .unwrap_or(ptr::null_mut())
}

/// A new video of the same prepared scene, filmed as `settings_json`
/// asks, released with [`seiza_parallax_free`]; null with `error_out` set
/// on failure, and when `cancel` stopped it.
///
/// `settings_json` is a complete set of the video's settings, as in
/// [`seiza_parallax_prepare_json`] but for the scene's: the single move
/// (`focus`, `start`, `dolly`, `truck`, `truckAngleDegrees`, `pan`, `zoom`,
/// `zoomEnd`, `rotateDegrees`, `easing`), a tour (`tour`, `autoTour`,
/// `tourGlide`, `tourTitles`, `tourLoop`), the output (`size`, `seconds`, `fps`, `quality`,
/// `growthLimit`, `fadeFrom`), the labels (`overlay`, `overlayDensity`,
/// `labels`, `labelColor`, `watermark`) and the cards (`introCard`,
/// `outroCard`, `cardSeconds`). Fields left out take their
/// defaults, not the first video's. Turns are taken as given, unwrapped:
/// `rotateDegrees` [0, 360] turns once anticlockwise and [0, -360] once
/// clockwise, and a stop's `rotateDegrees` likewise. A field that would
/// change the scene (the images, plate solution, catalogs, distances,
/// `distanceFocus`, star placement, galaxies, depth planes or dust) is
/// refused: those
/// need a new preparation, except a dust map, which
/// [`seiza_parallax_with_dust_map`] changes. The camera's `focus` only moves the camera; the
/// nebula keeps the depth it was prepared at.
///
/// Nothing of the scene is prepared again: the image is not reloaded,
/// split or solved, the stars are not found again and their distances not
/// looked up. The camera is fitted to the scene anew, and
/// [`seiza_parallax_summary_json`] on the new video gives its frame count,
/// size and the fit's adjustments. The catalogued objects' labels are read
/// once for every video of the scene.
///
/// Ownership: the new video and `video` share the prepared scene, which
/// lives until the last video using it is freed; free each with
/// [`seiza_parallax_free`], in any order. `video` is not changed: it may
/// keep drawing frames on other threads while this runs and after, and
/// both videos may draw frames at once. `cancel` stops the fit and the
/// label planning; `events` (nullable) hears each step on the calling
/// thread, with `context` passed through.
///
/// # Safety
///
/// `video` must be a live pointer from [`seiza_parallax_prepare_json`] or
/// this function, kept alive until this call returns. `settings_json` must
/// be a NUL-terminated UTF-8 string. `cancel` must be null or a live
/// [`SeizaCancelSignal`] retained until this call returns. When non-null,
/// `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_parallax_reconfigure_json(
    video: *const SeizaParallax,
    settings_json: *const c_char,
    cancel: *const SeizaCancelSignal,
    events: SeizaParallaxEventCallback,
    context: *mut c_void,
    error_out: *mut *mut c_char,
) -> *mut SeizaParallax {
    clear_error(error_out);
    let context = context as usize;
    let cancel = cancel as usize;
    ffi_result(error_out, || {
        let video = &unsafe { video.as_ref() }.ok_or("video is null")?.video;
        let settings_json = required_str(settings_json, "parallax settings JSON")?;
        let settings = video_options(&parse_video(&settings_json)?)?;
        let refilmed = video
            .reconfigure(
                &settings,
                &mut |event| send(events, context, event_json(event)),
                &|| cancelled(cancel),
            )
            .map_err(|error| error.to_string())?;
        Ok(Box::into_raw(Box::new(SeizaParallax { video: refilmed })))
    })
    .unwrap_or(ptr::null_mut())
}

/// What preparing the video found, as JSON: its frame count, frame rate
/// and size, the stars found and matched, the nebula's distance with how
/// and where (`backgroundFocus`) it was found, galaxies lifted, the depth
/// planes lifted (`planes`, `[{name, distanceParsecs, basis, coverage,
/// absorbs, object}]`, `basis` saying how the distance was found,
/// `coverage` the share of the image, `absorbs` whether it is a dark cloud
/// that blocks light behind it, and `object` the catalogued object whose
/// distance places it, by name, which `distanceOverrides` moves it by, or
/// null for a given mask or a cloud of dust no catalogued object names),
/// the dust
/// (`dustTransmission` `{median, thickest}`, `dustGiven` when a map gave
/// it, and `starsInFrontOfDust`, the stars it brought in front), objects
/// labelled, the WCS used, and `fit`, how fitting the camera to the
/// image changed the framing asked for: `zoom`, `pan`, `lead` and `truck`
/// for the single move and `stops[].zoom`, `stops[].pan` and
/// `stops[].dolly` for a tour, each `{asked, used}`, and `inside`, false
/// when some frames still show past the image's edge. Returns a string
/// released with `seiza_string_free`, or null with `error_out` set.
///
/// # Safety
///
/// `video` must be a live pointer from [`seiza_parallax_prepare_json`],
/// [`seiza_parallax_reconfigure_json`], [`seiza_parallax_with_dust_map`] or
/// [`seiza_parallax_with_dust_map_json`].
/// When non-null, `error_out` must point to writable storage for one
/// pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_parallax_summary_json(
    video: *const SeizaParallax,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let owner = unsafe { video.as_ref() }.ok_or("video is null")?;
        let video = &owner.video;
        let summary = video.summary();
        let (width, height) = video.size();
        let wcs = video.wcs();
        owned_json(&SummaryResponse {
            schema_version: 1,
            frames: video.frames(),
            fps: video.fps(),
            width,
            height,
            detected_stars: summary.detected_stars,
            gaia_stars: summary.gaia_stars,
            hipparcos_stars: summary.hipparcos_stars,
            gaia_matches: summary.gaia_matches,
            with_distance: summary.with_distance,
            background_distance_parsecs: summary.background_distance_pc,
            background_basis: summary.background_basis.clone(),
            background_focus: [summary.background_focus.0, summary.background_focus.1],
            unmatched_distance_parsecs: summary.unmatched_distance_pc,
            flying_stars: summary.flying_stars,
            galaxies_lifted: summary.galaxies_lifted.clone(),
            planes: summary
                .planes
                .iter()
                .map(|plane| PlaneResponse {
                    name: plane.name.clone(),
                    distance_parsecs: plane.distance_pc,
                    basis: plane.basis.clone(),
                    coverage: plane.coverage,
                    absorbs: plane.absorbs,
                    object: plane.object.clone(),
                })
                .collect(),
            dust_transmission: summary
                .dust_transmission
                .map(|(median, thickest)| DustResponse { median, thickest }),
            dust_given: summary.dust_given,
            stars_in_front_of_dust: summary.stars_in_front_of_dust,
            labelled_objects: summary.labelled_objects,
            fit: (&summary.fit).into(),
            wcs: WcsResponse {
                crval: [wcs.crval.0, wcs.crval.1],
                crpix: [wcs.crpix.0, wcs.crpix.1],
                cd: wcs.cd,
                sip: wcs.sip.as_ref().map(|sip| SipResponse {
                    order: sip.order,
                    a: sip.a.clone(),
                    b: sip.b.clone(),
                    ap: sip.ap.clone(),
                    bp: sip.bp.clone(),
                }),
                pixel_origin: 0,
            },
        })
    })
    .unwrap_or(ptr::null_mut())
}

/// [`seiza_parallax_dust_map`] reads the dust the video uses: a given
/// map's, or the dust mapped from the star counts.
pub const SEIZA_DUST_MAP_IN_USE: u32 = 0;
/// [`seiza_parallax_dust_map`] reads the dust mapped from the star counts,
/// even while a given map takes its place.
pub const SEIZA_DUST_MAP_COUNTED: u32 = 1;

/// The dust `which` names, or `None` when the video has none.
fn which_dust(video: &Parallax, which: u32) -> Result<Option<&Dust>, String> {
    match which {
        SEIZA_DUST_MAP_IN_USE => Ok(video.dust()),
        SEIZA_DUST_MAP_COUNTED => Ok(video.counted_dust()),
        other => Err(format!("{other} is not a SEIZA_DUST_MAP_* value")),
    }
}

/// Write `video`'s dust into `opacity` as a `width` × `height` map over the
/// whole image, row by row: 0 where the sky is clear to 1 where the dust is
/// opaque, the values [`seiza_parallax_with_dust_map`] takes back, so a
/// desktop app can show it and paint over it. `which` is
/// [`SEIZA_DUST_MAP_IN_USE`] or [`SEIZA_DUST_MAP_COUNTED`]. Returns 1 when
/// the map was written, 0 when the video has no such dust (the dust is off,
/// or too few stars show to count it), and -1 with `error_out` set on
/// failure. Any size works; the image's own shape keeps the map square to
/// it.
///
/// # Safety
///
/// `video` must be a live pointer from [`seiza_parallax_prepare_json`],
/// [`seiza_parallax_reconfigure_json`], [`seiza_parallax_with_dust_map`] or
/// [`seiza_parallax_with_dust_map_json`].
/// `opacity` must point to `length` writable floats. When non-null,
/// `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_parallax_dust_map(
    video: *const SeizaParallax,
    which: u32,
    width: u32,
    height: u32,
    opacity: *mut f32,
    length: usize,
    error_out: *mut *mut c_char,
) -> i32 {
    clear_error(error_out);
    let opacity = opacity as usize;
    ffi_result(error_out, || {
        let video = &unsafe { video.as_ref() }.ok_or("video is null")?.video;
        if opacity == 0 {
            return Err("opacity is null".into());
        }
        let (width, height) = (width as usize, height as usize);
        if width == 0 || height == 0 {
            return Err("the map needs a width and height".into());
        }
        let needed = width.checked_mul(height).ok_or("the map is too large")?;
        if length < needed {
            return Err(format!(
                "the map needs {needed} values; the buffer holds {length}"
            ));
        }
        let Some(dust) = which_dust(video, which)? else {
            return Ok(0);
        };
        let map = dust.to_map(width, height);
        let out = unsafe { std::slice::from_raw_parts_mut(opacity as *mut f32, needed) };
        out.copy_from_slice(&map.opacity);
        Ok(1)
    })
    .unwrap_or(-1)
}

/// A new video of `video`'s scene and camera whose dust is the `width` ×
/// `height` map `opacity`, row by row, 0 where the sky is clear to 1 where
/// the dust is opaque, spanning the whole image at any size of its shape;
/// or, when `opacity` is null, the dust mapped from the star counts again.
/// What lies behind the nebula dims by the map, and a star seen where it
/// is opaque flies in front of the nebula. This is for painting: draw over
/// [`seiza_parallax_dust_map`]'s map, pass it here, and draw frames from
/// the new video; [`seiza_parallax_with_dust_map_json`] reads the map from
/// a file instead. A map gives dust even to a video prepared without it.
///
/// Only the dust changes. The planes and stars are shared, not copied, so
/// this takes about as long as fitting the camera again
/// ([`seiza_parallax_reconfigure_json`]). Ownership is as there: free each
/// video with [`seiza_parallax_free`], in any order; `video` is not changed
/// and may keep drawing frames meanwhile. `cancel` stops the fit; `events`
/// (nullable) hears each step on the calling thread, with `context` passed
/// through.
///
/// # Safety
///
/// `video` must be a live pointer from [`seiza_parallax_prepare_json`],
/// [`seiza_parallax_reconfigure_json`] or this function, kept alive until
/// this call returns. `opacity` must be null or point to `width` × `height`
/// readable floats. `cancel` must be null or a live [`SeizaCancelSignal`]
/// retained until this call returns. When non-null, `error_out` must point
/// to writable storage for one pointer.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn seiza_parallax_with_dust_map(
    video: *const SeizaParallax,
    opacity: *const f32,
    width: u32,
    height: u32,
    cancel: *const SeizaCancelSignal,
    events: SeizaParallaxEventCallback,
    context: *mut c_void,
    error_out: *mut *mut c_char,
) -> *mut SeizaParallax {
    clear_error(error_out);
    let context = context as usize;
    let cancel = cancel as usize;
    let opacity = opacity as usize;
    ffi_result(error_out, || {
        let video = &unsafe { video.as_ref() }.ok_or("video is null")?.video;
        let map = if opacity == 0 {
            None
        } else {
            let (width, height) = (width as usize, height as usize);
            let length = width.checked_mul(height).ok_or("the map is too large")?;
            if length == 0 {
                return Err("the map needs a width and height".into());
            }
            let values = unsafe { std::slice::from_raw_parts(opacity as *const f32, length) };
            Some(DustMap {
                width,
                height,
                opacity: values.to_vec(),
            })
        };
        let painted = video
            .with_dust_map(
                map.as_ref(),
                &mut |event| send(events, context, event_json(event)),
                &|| cancelled(cancel),
            )
            .map_err(|error| error.to_string())?;
        Ok(Box::into_raw(Box::new(SeizaParallax { video: painted })))
    })
    .unwrap_or(ptr::null_mut())
}

/// What [`seiza_parallax_with_dust_map_json`] reads.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DustMapReadRequest {
    /// The map file, its format by its extension; the counted dust if absent.
    path: Option<PathBuf>,
}

/// [`seiza_parallax_with_dust_map`] with the map read from a file:
/// `request_json` is `{"path": ...}`, an image white where the dust is
/// opaque, at any size of the image's shape, as PNG, JPEG, TIFF, FITS or
/// XISF by its extension; `{}` or a null `path` goes back to the dust
/// mapped from the star counts. For a map painted in another program and
/// read again after each change. Ownership, sharing and the other arguments
/// are as for [`seiza_parallax_with_dust_map`].
///
/// # Safety
///
/// `video` must be a live pointer from [`seiza_parallax_prepare_json`],
/// [`seiza_parallax_reconfigure_json`] or [`seiza_parallax_with_dust_map`],
/// kept alive until this call returns. `request_json` must be a
/// NUL-terminated UTF-8 string. `cancel` must be null or a live
/// [`SeizaCancelSignal`] retained until this call returns. When non-null,
/// `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_parallax_with_dust_map_json(
    video: *const SeizaParallax,
    request_json: *const c_char,
    cancel: *const SeizaCancelSignal,
    events: SeizaParallaxEventCallback,
    context: *mut c_void,
    error_out: *mut *mut c_char,
) -> *mut SeizaParallax {
    clear_error(error_out);
    let context = context as usize;
    let cancel = cancel as usize;
    ffi_result(error_out, || {
        let video = &unsafe { video.as_ref() }.ok_or("video is null")?.video;
        let request_json = required_str(request_json, "dust map request JSON")?;
        let request: DustMapReadRequest = serde_json::from_str(&request_json)
            .map_err(|error| format!("invalid dust map request JSON: {error}"))?;
        let map = request.path.as_deref().map(DustMap::open).transpose()?;
        let painted = video
            .with_dust_map(
                map.as_ref(),
                &mut |event| send(events, context, event_json(event)),
                &|| cancelled(cancel),
            )
            .map_err(|error| error.to_string())?;
        Ok(Box::into_raw(Box::new(SeizaParallax { video: painted })))
    })
    .unwrap_or(ptr::null_mut())
}

/// What [`seiza_parallax_write_dust_map_json`] writes.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DustMapWriteRequest {
    /// The file to write, its format by its extension.
    path: PathBuf,
    /// The map's size; the image's if absent.
    width: Option<u32>,
    height: Option<u32>,
    /// The dust mapped from the star counts, rather than the dust in use.
    #[serde(default)]
    counted: bool,
}

/// Write `video`'s dust, white (1) where opaque, as `seiza parallax-video
/// --export-dust-map` does, in the format the path's extension names: a
/// 16-bit grayscale PNG or TIFF (`.png`, `.tif`), or a 32-bit float FITS or
/// XISF (`.fits`, `.xisf`). `request_json` is `{"path": ..., "width": ...,
/// "height": ..., "counted": false}`, the size the image's when absent, and
/// `counted` true for the dust mapped from the star counts rather than the
/// dust in use. Returns 1 when written, 0 when
/// the video has no such dust, and -1 with `error_out` set on failure.
///
/// # Safety
///
/// `video` must be a live pointer from [`seiza_parallax_prepare_json`],
/// [`seiza_parallax_reconfigure_json`], [`seiza_parallax_with_dust_map`] or
/// [`seiza_parallax_with_dust_map_json`].
/// `request_json` must be a NUL-terminated UTF-8 string. When non-null,
/// `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_parallax_write_dust_map_json(
    video: *const SeizaParallax,
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> i32 {
    clear_error(error_out);
    ffi_result(error_out, || {
        let video = &unsafe { video.as_ref() }.ok_or("video is null")?.video;
        let request_json = required_str(request_json, "dust map request JSON")?;
        let request: DustMapWriteRequest = serde_json::from_str(&request_json)
            .map_err(|error| format!("invalid dust map request JSON: {error}"))?;
        let which = if request.counted {
            SEIZA_DUST_MAP_COUNTED
        } else {
            SEIZA_DUST_MAP_IN_USE
        };
        let Some(dust) = which_dust(video, which)? else {
            return Ok(0);
        };
        let (image_width, image_height) = dust.size();
        let width = request.width.map_or(image_width, |width| width as usize);
        let height = request
            .height
            .map_or(image_height, |height| height as usize);
        if width == 0 || height == 0 {
            return Err("the map needs a width and height".into());
        }
        dust.to_map(width, height).save(&request.path)?;
        Ok(1)
    })
    .unwrap_or(-1)
}

/// Bytes a pixel of `format` takes.
fn pixel_bytes(format: u32) -> Result<usize, String> {
    match format {
        SEIZA_PIXEL_FORMAT_RGB8 => Ok(3),
        SEIZA_PIXEL_FORMAT_RGBA8 | SEIZA_PIXEL_FORMAT_BGRA8 => Ok(4),
        _ => Err(format!("unknown pixel format {format}")),
    }
}

/// Write `frame` into `out`, `stride` bytes a row, in `format`.
fn write_pixels(frame: &RgbImage, format: u32, out: &mut [u8], stride: usize) {
    let width = frame.width() as usize;
    for (y, row) in frame.as_raw().chunks_exact(width * 3).enumerate() {
        let target = &mut out[y * stride..];
        match format {
            SEIZA_PIXEL_FORMAT_RGB8 => target[..width * 3].copy_from_slice(row),
            SEIZA_PIXEL_FORMAT_RGBA8 => {
                for (pixel, rgb) in target.chunks_exact_mut(4).zip(row.chunks_exact(3)) {
                    pixel.copy_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
                }
            }
            _ => {
                for (pixel, rgb) in target.chunks_exact_mut(4).zip(row.chunks_exact(3)) {
                    pixel.copy_from_slice(&[rgb[2], rgb[1], rgb[0], 255]);
                }
            }
        }
    }
}

/// Draw frame `index` into the caller's `buffer` of `buffer_length` bytes,
/// `stride` bytes a row (at least the frame's width times the pixel's
/// bytes), in `format` (a `SEIZA_PIXEL_FORMAT_*` value): a platform
/// encoder's own pixel buffer can take the frame directly. Returns false
/// with `error_out` set when the index, format or buffer will not do.
/// Frames may be drawn in any order, from several threads at once.
///
/// # Safety
///
/// `video` must be a live pointer from [`seiza_parallax_prepare_json`],
/// [`seiza_parallax_reconfigure_json`], [`seiza_parallax_with_dust_map`] or
/// [`seiza_parallax_with_dust_map_json`].
/// `buffer` must point to `buffer_length` writable bytes. When non-null,
/// `error_out` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_parallax_render_frame(
    video: *const SeizaParallax,
    index: u32,
    format: u32,
    buffer: *mut u8,
    buffer_length: usize,
    stride: usize,
    error_out: *mut *mut c_char,
) -> bool {
    clear_error(error_out);
    let buffer = buffer as usize;
    ffi_result(error_out, || {
        let video = &unsafe { video.as_ref() }.ok_or("video is null")?.video;
        if buffer == 0 {
            return Err("buffer is null".into());
        }
        let (width, height) = video.size();
        let bytes = pixel_bytes(format)?;
        let row = width * bytes;
        if stride < row {
            return Err(format!("a row needs {row} bytes; the stride is {stride}"));
        }
        let needed = stride
            .checked_mul(height - 1)
            .and_then(|rows| rows.checked_add(row))
            .ok_or_else(|| format!("a stride of {stride} bytes is too large"))?;
        if buffer_length < needed {
            return Err(format!(
                "the frame needs {needed} bytes; the buffer holds {buffer_length}"
            ));
        }
        if index as usize >= video.frames() {
            return Err(format!(
                "frame {index} is past the video's {} frames",
                video.frames()
            ));
        }
        let out = unsafe { std::slice::from_raw_parts_mut(buffer as *mut u8, buffer_length) };
        write_pixels(&video.frame(index as usize), format, out, stride);
        Ok(true)
    })
    .unwrap_or(false)
}

/// Draw every frame in order and hand each to `frame` in `format` (a
/// `SEIZA_PIXEL_FORMAT_*` value), rows packed with no padding: the hook for
/// a platform or other external video encoder. Returns 1 when every frame
/// was delivered, 0 when `cancel` or a nonzero return from `frame` stopped
/// the video, and -1 with `error_out` set on failure. `events` (nullable)
/// hears each frame drawn, with `context` passed through to both
/// callbacks.
///
/// # Safety
///
/// `video` must be a live pointer from [`seiza_parallax_prepare_json`],
/// [`seiza_parallax_reconfigure_json`], [`seiza_parallax_with_dust_map`] or
/// [`seiza_parallax_with_dust_map_json`].
/// `cancel` must be null or a live [`SeizaCancelSignal`] retained until
/// this call returns. When non-null, `error_out` must point to writable
/// storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_parallax_render_frames(
    video: *const SeizaParallax,
    format: u32,
    cancel: *const SeizaCancelSignal,
    frame: SeizaParallaxFrameCallback,
    events: SeizaParallaxEventCallback,
    context: *mut c_void,
    error_out: *mut *mut c_char,
) -> i32 {
    clear_error(error_out);
    let context = context as usize;
    let cancel = cancel as usize;
    ffi_result(error_out, || {
        let video = &unsafe { video.as_ref() }.ok_or("video is null")?.video;
        let callback = frame.ok_or("the frame callback is null")?;
        let bytes = pixel_bytes(format)?;
        let (width, height) = video.size();
        let stride = width * bytes;
        let fps = video.fps() as f64;
        let total = video.frames();
        let refused = AtomicBool::new(false);
        let mut pixels = vec![0_u8; stride * height];
        let mut index = 0_usize;
        let sink = FrameFn::new(|image: &RgbImage| {
            write_pixels(image, format, &mut pixels, stride);
            let answer = unsafe {
                callback(
                    pixels.as_ptr(),
                    stride,
                    width as u32,
                    height as u32,
                    index as u32,
                    total as u32,
                    index as f64 / fps,
                    context as *mut c_void,
                )
            };
            index += 1;
            if answer != 0 {
                refused.store(true, Ordering::Relaxed);
                return Err("the frame callback stopped the video".into());
            }
            Ok(())
        });
        finish(
            video.render(
                Box::new(sink),
                &mut |event| send(events, context, event_json(event)),
                &|| cancelled(cancel),
            ),
            refused.load(Ordering::Relaxed),
        )
    })
    .unwrap_or(-1)
}

/// Write the video to a file: `{"output": path, "encoder": "auto" |
/// "ffmpeg" | "png", "ffmpeg": program, "codec": "h264" | "hevc"}`.
/// "auto" and "ffmpeg" encode with ffmpeg, H.264 unless `codec` says HEVC; "png" writes numbered PNG frames into the `output`
/// directory. Returns 1 when written, 0 when `cancel` stopped it, and -1
/// with `error_out` set on failure. `events` (nullable) hears each frame
/// drawn, with `context` passed through.
///
/// # Safety
///
/// `video` must be a live pointer from [`seiza_parallax_prepare_json`],
/// [`seiza_parallax_reconfigure_json`], [`seiza_parallax_with_dust_map`] or
/// [`seiza_parallax_with_dust_map_json`].
/// `request_json` must be a NUL-terminated UTF-8 string. `cancel` must be
/// null or a live [`SeizaCancelSignal`] retained until this call returns.
/// When non-null, `error_out` must point to writable storage for one
/// pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_parallax_write_video_json(
    video: *const SeizaParallax,
    request_json: *const c_char,
    cancel: *const SeizaCancelSignal,
    events: SeizaParallaxEventCallback,
    context: *mut c_void,
    error_out: *mut *mut c_char,
) -> i32 {
    clear_error(error_out);
    let context = context as usize;
    let cancel = cancel as usize;
    ffi_result(error_out, || {
        let video = &unsafe { video.as_ref() }.ok_or("video is null")?.video;
        let request_json = required_str(request_json, "write request JSON")?;
        let request: WriteRequest = serde_json::from_str(&request_json)
            .map_err(|error| format!("invalid write request JSON: {error}"))?;
        let settings = video.video_settings();
        let ffmpeg = request.ffmpeg.unwrap_or_else(|| PathBuf::from("ffmpeg"));
        let sink: Box<dyn FrameSink> = match request.encoder.as_deref().unwrap_or("auto") {
            "auto" | "ffmpeg" => Box::new(
                FfmpegSink::start(
                    &ffmpeg,
                    &request.output,
                    settings,
                    match &request.codec {
                        Some(codec) => seiza_parallax::Codec::parse(codec)?,
                        None => seiza_parallax::Codec::H264,
                    },
                )
                .map_err(|error| error.to_string())?,
            ),
            "png" => Box::new(
                PngSequence::new(&request.output, settings).map_err(|error| error.to_string())?,
            ),
            other => {
                return Err(format!(
                    "encoder must be \"auto\", \"ffmpeg\" or \"png\"; got {other:?}"
                ));
            }
        };
        finish(
            video.render(
                sink,
                &mut |event| send(events, context, event_json(event)),
                &|| cancelled(cancel),
            ),
            false,
        )
    })
    .unwrap_or(-1)
}

/// Whether the cancel signal at address `cancel`, if any, has fired.
fn cancelled(cancel: usize) -> bool {
    unsafe { (cancel as *const SeizaCancelSignal).as_ref() }
        .is_some_and(|signal| signal.cancelled.load(Ordering::Relaxed))
}

/// 1 for a finished render, 0 for one stopped by a cancel or a refusing
/// callback, and the error otherwise.
fn finish(
    rendered: Result<(), seiza_parallax::pipeline::Error>,
    refused: bool,
) -> Result<i32, String> {
    match rendered {
        Ok(()) => Ok(1),
        Err(seiza_parallax::pipeline::Error::Stopped) => Ok(0),
        Err(_) if refused => Ok(0),
        Err(error) => Err(error.to_string()),
    }
}

/// Release a video from [`seiza_parallax_prepare_json`].
///
/// # Safety
///
/// `video` must be null or a pointer from [`seiza_parallax_prepare_json`],
/// [`seiza_parallax_reconfigure_json`] or [`seiza_parallax_with_dust_map`]
/// that has not already been freed, with no call using it still running.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_parallax_free(video: *mut SeizaParallax) {
    if !video.is_null() {
        unsafe { drop(Box::from_raw(video)) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        seiza_cancel_signal_cancel, seiza_cancel_signal_create, seiza_cancel_signal_free,
        seiza_string_free,
    };
    use seiza::catalog::{DistanceStar, StarDistanceCatalogBuilder};
    use std::ffi::CStr;

    #[test]
    fn a_wcs_reads_its_pixel_origin() {
        let read = |json: &str| serde_json::from_str::<WcsRequest>(json).unwrap().wcs();
        let base = r#""crval":[10.0,20.0],"crpix":[101.0,51.0],"cd":[[0.001,0.0],[0.0,0.001]]"#;
        // As this ABI returns it, counted from 0, by default or by name.
        assert_eq!(read(&format!("{{{base}}}")).unwrap().crpix, (101.0, 51.0));
        assert_eq!(
            read(&format!("{{{base},\"pixelOrigin\":0}}"))
                .unwrap()
                .crpix,
            (101.0, 51.0)
        );
        // As a FITS header gives it, counted from 1.
        assert_eq!(
            read(&format!("{{{base},\"pixelOrigin\":1}}"))
                .unwrap()
                .crpix,
            (100.0, 50.0)
        );
        assert!(read(&format!("{{{base},\"pixelOrigin\":2}}")).is_err());
    }

    #[test]
    fn a_watermark_and_cards_read_from_their_fields() {
        let video = |json: &str| video_options(&parse_video(json).unwrap());
        // The plain forms keep their meaning.
        let shown = video(r#"{"watermark": true}"#).unwrap();
        assert_eq!(
            shown.watermark.as_deref(),
            Some(seiza_parallax::DEFAULT_WATERMARK)
        );
        assert_eq!(shown.watermark_image, None);
        let text = video(r#"{"watermark": "© me"}"#).unwrap();
        assert_eq!(text.watermark.as_deref(), Some("© me"));
        // The object form adds an image, its corner, size and opacity.
        let styled = video(
            r#"{"watermark": {"image": "logo.png", "corner": "topLeft", "size": 0.1,
                "opacity": 0.5},
               "introCard": "start.png", "outroCard": "end.png", "cardSeconds": 4,
               "tour": [{"hold": 1}, {"focus": [10, 20], "card": "stop.png"}]}"#,
        )
        .unwrap();
        assert_eq!(styled.watermark, None);
        assert_eq!(styled.watermark_image, Some(PathBuf::from("logo.png")));
        assert_eq!(styled.watermark_corner, Corner::TopLeft);
        assert_eq!(
            (styled.watermark_size, styled.watermark_opacity),
            (0.1, 0.5)
        );
        assert_eq!(styled.intro_card, Some(PathBuf::from("start.png")));
        assert_eq!(styled.outro_card, Some(PathBuf::from("end.png")));
        assert_eq!(styled.card_seconds, 4.0);
        assert_eq!(styled.tour[1].card, Some(PathBuf::from("stop.png")));
        let both = video(r#"{"watermark": {"text": true, "image": "logo.png"}}"#).unwrap();
        assert!(both.watermark.is_some() && both.watermark_image.is_some());
        assert_eq!(both.watermark_corner, Corner::BottomRight);
        // An unknown corner is refused.
        let error = video(r#"{"watermark": {"corner": "middle"}}"#).unwrap_err();
        assert!(error.contains("watermark corner"), "{error}");
    }

    /// A small solved field written to `directory`: starless and stars
    /// PNGs and a star distance file placing the stars, and the request
    /// JSON naming them with the field's WCS.
    fn request(directory: &Path, extra: &str) -> CString {
        let (width, height) = (320_u32, 240_u32);
        let wcs = Wcs::from_center_scale_rotation((56.75, 24.12), (160.0, 120.0), 2.0, 0.0, false);
        image::RgbImage::from_fn(width, height, |x, _| {
            image::Rgb([20 + (x / 8) as u8, 30, 25])
        })
        .save(directory.join("starless.png"))
        .unwrap();
        let places = [
            (60.0, 50.0, 120.0),
            (250.0, 70.0, 900.0),
            (140.0, 180.0, 300.0),
        ];
        image::RgbImage::from_fn(width, height, |x, y| {
            let light: f64 = places
                .iter()
                .map(|&(sx, sy, _)| {
                    0.9 * (-((x as f64 - sx).powi(2) + (y as f64 - sy).powi(2)) / 4.0).exp()
                })
                .sum();
            image::Rgb([(light.min(1.0) * 255.0) as u8; 3])
        })
        .save(directory.join("stars.png"))
        .unwrap();
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
        builder
            .write_to(&directory.join("star-distances.bin"))
            .unwrap();
        let mut json = serde_json::json!({
            "starless": directory.join("starless.png"),
            "stars": directory.join("stars.png"),
            "wcs": {
                "crval": [wcs.crval.0, wcs.crval.1],
                "crpix": [wcs.crpix.0, wcs.crpix.1],
                "cd": wcs.cd,
            },
            "catalogDirectory": directory,
            "online": false,
            "distanceParsecs": 400.0,
            "size": "160x120",
            "seconds": 1.0,
            "fps": 5,
            "labels": [{"x": 100.0, "y": 90.0, "radius": 15.0, "text": "here"}],
            "watermark": true,
        });
        // Fields to add or replace, written as the inside of an object.
        if !extra.is_empty() {
            let more: serde_json::Map<String, serde_json::Value> =
                serde_json::from_str(&format!("{{{extra}}}")).unwrap();
            json.as_object_mut().unwrap().extend(more);
        }
        let json = json.to_string();
        CString::new(json).unwrap()
    }

    fn take_error(error: *mut c_char) -> String {
        assert!(!error.is_null());
        let message = unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned();
        unsafe { seiza_string_free(error) };
        message
    }

    unsafe extern "C" fn collect_event(json: *const c_char, context: *mut c_void) {
        let events = unsafe { &mut *(context as *mut Vec<String>) };
        events.push(
            unsafe { CStr::from_ptr(json) }
                .to_string_lossy()
                .into_owned(),
        );
    }

    /// Frames seen and the frame to stop at, for the frame callback.
    struct Seen {
        frames: Vec<(u32, u32, u32, f64)>,
        stop_at: u32,
    }

    unsafe extern "C" fn take_frame(
        pixels: *const u8,
        stride: usize,
        width: u32,
        height: u32,
        index: u32,
        total: u32,
        seconds: f64,
        context: *mut c_void,
    ) -> i32 {
        let seen = unsafe { &mut *(context as *mut Seen) };
        assert!(!pixels.is_null());
        assert_eq!(stride, width as usize * 4);
        let _ = height;
        seen.frames.push((index, total, width, seconds));
        i32::from(index + 1 == seen.stop_at)
    }

    #[test]
    fn a_prepared_video_draws_frames_for_the_caller_and_its_encoder() {
        let directory = tempfile::tempdir().unwrap();
        let request = request(directory.path(), "");
        let mut events: Vec<String> = Vec::new();
        let mut error = ptr::null_mut();
        let video = unsafe {
            seiza_parallax_prepare_json(
                request.as_ptr(),
                ptr::null(),
                Some(collect_event),
                &mut events as *mut Vec<String> as *mut c_void,
                &mut error,
            )
        };
        assert!(!video.is_null(), "{}", take_error(error));
        assert!(
            events.iter().any(|event| event.contains("matched to Gaia")),
            "{events:?}"
        );

        let summary = unsafe { seiza_parallax_summary_json(video, &mut error) };
        let parsed: serde_json::Value =
            serde_json::from_str(&unsafe { CStr::from_ptr(summary) }.to_string_lossy()).unwrap();
        unsafe { seiza_string_free(summary) };
        assert_eq!(parsed["frames"], 5);
        assert_eq!(parsed["width"], 160);
        assert_eq!(parsed["detectedStars"], 3);
        assert_eq!(parsed["gaiaMatches"], 3);
        assert_eq!(parsed["backgroundDistanceParsecs"], 400.0);

        // One frame into a padded BGRA buffer, matching the RGB frame.
        let stride = 160 * 4 + 64;
        let mut bgra = vec![7_u8; stride * 120];
        assert!(unsafe {
            seiza_parallax_render_frame(
                video,
                2,
                SEIZA_PIXEL_FORMAT_BGRA8,
                bgra.as_mut_ptr(),
                bgra.len(),
                stride,
                &mut error,
            )
        });
        let mut rgb = vec![0_u8; 160 * 3 * 120];
        assert!(unsafe {
            seiza_parallax_render_frame(
                video,
                2,
                SEIZA_PIXEL_FORMAT_RGB8,
                rgb.as_mut_ptr(),
                rgb.len(),
                160 * 3,
                &mut error,
            )
        });
        for (x, y) in [(0, 0), (80, 60), (159, 119)] {
            let b = &bgra[y * stride + x * 4..][..4];
            let r = &rgb[(y * 160 + x) * 3..][..3];
            assert_eq!([b[2], b[1], b[0], b[3]], [r[0], r[1], r[2], 255]);
        }
        assert_eq!(bgra[160 * 4], 7, "the row's padding is left alone");

        // A buffer too small, a frame past the end and an unknown format.
        for (index, format, length) in [
            (0, SEIZA_PIXEL_FORMAT_RGB8, 100),
            (5, SEIZA_PIXEL_FORMAT_RGB8, rgb.len()),
            (0, 9, rgb.len()),
        ] {
            assert!(!unsafe {
                seiza_parallax_render_frame(
                    video,
                    index,
                    format,
                    rgb.as_mut_ptr(),
                    length,
                    160 * 3,
                    &mut error,
                )
            });
            take_error(error);
        }

        // Every frame to the callback in order, and a callback that stops.
        let mut seen = Seen {
            frames: Vec::new(),
            stop_at: 0,
        };
        let done = unsafe {
            seiza_parallax_render_frames(
                video,
                SEIZA_PIXEL_FORMAT_BGRA8,
                ptr::null(),
                Some(take_frame),
                None,
                &mut seen as *mut Seen as *mut c_void,
                &mut error,
            )
        };
        assert_eq!(done, 1);
        let indices: Vec<u32> = seen.frames.iter().map(|frame| frame.0).collect();
        assert_eq!(indices, [0, 1, 2, 3, 4]);
        assert!((seen.frames[4].3 - 0.8).abs() < 1e-9);
        seen = Seen {
            frames: Vec::new(),
            stop_at: 2,
        };
        let stopped = unsafe {
            seiza_parallax_render_frames(
                video,
                SEIZA_PIXEL_FORMAT_BGRA8,
                ptr::null(),
                Some(take_frame),
                None,
                &mut seen as *mut Seen as *mut c_void,
                &mut error,
            )
        };
        assert_eq!((stopped, seen.frames.len()), (0, 2));
        assert!(error.is_null());

        // A cancelled signal stops before the first frame.
        let signal = seiza_cancel_signal_create();
        unsafe { seiza_cancel_signal_cancel(signal) };
        seen.frames.clear();
        let cancelled = unsafe {
            seiza_parallax_render_frames(
                video,
                SEIZA_PIXEL_FORMAT_RGB8,
                signal,
                Some(take_frame),
                None,
                &mut seen as *mut Seen as *mut c_void,
                &mut error,
            )
        };
        assert_eq!((cancelled, seen.frames.len()), (0, 0));
        unsafe { seiza_cancel_signal_free(signal) };

        // PNG frames written to a directory.
        let frames = directory.path().join("frames");
        let write =
            CString::new(serde_json::json!({ "output": frames, "encoder": "png" }).to_string())
                .unwrap();
        let written = unsafe {
            seiza_parallax_write_video_json(
                video,
                write.as_ptr(),
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        assert_eq!(
            written,
            1,
            "{}",
            if error.is_null() {
                String::new()
            } else {
                take_error(error)
            }
        );
        assert_eq!(std::fs::read_dir(&frames).unwrap().count(), 5);
        unsafe { seiza_parallax_free(video) };

        // A tour of three stops sets the length: 1 + 1 + 1 + 1 seconds at
        // 5 frames a second.
        let tour = self::request(
            directory.path(),
            "\"tour\": [{\"hold\": 1}, {\"focus\": [150, 110], \"dolly\": 0.5, \"rotateDegrees\": 10, \"travel\": 1, \"hold\": 1}, {\"travel\": 1}]",
        );
        let toured = unsafe {
            seiza_parallax_prepare_json(
                tour.as_ptr(),
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        assert!(!toured.is_null(), "{}", take_error(error));
        let summary = unsafe { seiza_parallax_summary_json(toured, &mut error) };
        let parsed: serde_json::Value =
            serde_json::from_str(&unsafe { CStr::from_ptr(summary) }.to_string_lossy()).unwrap();
        unsafe { seiza_string_free(summary) };
        assert_eq!(parsed["frames"], 20);
        unsafe { seiza_parallax_free(toured) };
    }

    #[test]
    fn a_tour_planned_on_a_prepared_scene_frames_targets_at_their_planes() {
        let directory = tempfile::tempdir().unwrap();
        let wcs = Wcs::from_center_scale_rotation((56.75, 24.12), (160.0, 120.0), 2.0, 0.0, false);
        let object = |name: &str, x: f64, y: f64| {
            let (ra, dec) = wcs.pixel_to_world(x, y);
            seiza::objects::SkyObject {
                kind: seiza::objects::ObjectKind::Nebula,
                ra,
                dec,
                mag: None,
                major_arcmin: Some(1.0),
                minor_arcmin: None,
                position_angle_deg: None,
                name: name.into(),
                common_name: String::new(),
                metadata: Default::default(),
            }
        };
        seiza::objects::ObjectCatalog::new(vec![
            object("NGC 9001", 100.0, 80.0),
            object("IC 9002", 230.0, 170.0),
        ])
        .write_to(&directory.path().join("objects.bin"))
        .unwrap();
        // A glowing knot about IC 9002, given its own plane at 1200 pc,
        // behind the nebula at 400.
        let mask = directory.path().join("knot.png");
        image::GrayImage::from_fn(320, 240, |x, y| {
            let inside = (x as f64 - 230.0).hypot(y as f64 - 170.0) < 30.0;
            image::Luma([if inside { 255 } else { 0 }])
        })
        .save(&mask)
        .unwrap();
        let asked = request(
            directory.path(),
            &format!(
                "\"planes\": [{{\"mask\": {}, \"distanceParsecs\": 1200}}]",
                serde_json::json!(mask)
            ),
        );
        image::RgbImage::from_fn(320, 240, |x, y| {
            let r2 = (x as f64 - 230.0).powi(2) + (y as f64 - 170.0).powi(2);
            let knot = (120.0 * (-r2 / 300.0).exp()) as u8;
            image::Rgb([20 + (x / 8) as u8 + knot, 30 + knot, 25 + knot])
        })
        .save(directory.path().join("starless.png"))
        .unwrap();
        let mut error = ptr::null_mut();
        let video = unsafe {
            seiza_parallax_prepare_json(
                asked.as_ptr(),
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        assert!(!video.is_null(), "{}", take_error(error));
        assert_eq!(summary(video)["planes"][0]["distanceParsecs"], 1200.0);
        let plan = |settings: &str| -> Result<serde_json::Value, String> {
            let settings = CString::new(settings).unwrap();
            let mut error = ptr::null_mut();
            let planned = unsafe {
                seiza_parallax_plan_scene_tour_json(video, settings.as_ptr(), &mut error)
            };
            if planned.is_null() {
                return Err(take_error(error));
            }
            let plan = serde_json::from_str(&unsafe { CStr::from_ptr(planned) }.to_string_lossy());
            unsafe { seiza_string_free(planned) };
            Ok(plan.unwrap())
        };
        let depth_of = |plan: &serde_json::Value, name: &str| {
            plan["tour"]
                .as_array()
                .unwrap()
                .iter()
                .find(|stop| stop["name"] == name)
                .map(|stop| stop["depthParsecs"].clone())
                .unwrap()
        };
        // On the scene, the knot's target is framed at its plane, and the
        // other on the nebula's.
        let on_scene = plan(r#"{"autoTour": {"hold": 1.0}}"#).unwrap();
        assert_eq!(depth_of(&on_scene, "IC 9002"), 1200.0);
        assert!(depth_of(&on_scene, "NGC 9001").is_null());
        // From the catalogue alone, neither is.
        let mut error = ptr::null_mut();
        let planned = unsafe {
            seiza_parallax_plan_tour_json(asked.as_ptr(), None, ptr::null_mut(), &mut error)
        };
        assert!(!planned.is_null(), "{}", take_error(error));
        let catalogued: serde_json::Value =
            serde_json::from_str(&unsafe { CStr::from_ptr(planned) }.to_string_lossy()).unwrap();
        unsafe { seiza_string_free(planned) };
        assert!(depth_of(&catalogued, "IC 9002").is_null());
        // The plan goes back in as the video's tour.
        let tour = serde_json::to_string(&on_scene["tour"]).unwrap();
        let settings = CString::new(format!(
            r#"{{"size": "160x120", "fps": 5, "tour": {tour}}}"#
        ))
        .unwrap();
        let mut error = ptr::null_mut();
        let toured = unsafe {
            seiza_parallax_reconfigure_json(
                video,
                settings.as_ptr(),
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        assert!(!toured.is_null(), "{}", take_error(error));
        // A frame of its own size, and bad settings, are answered.
        let portrait = plan(r#"{"size": "1080p-portrait"}"#).unwrap();
        assert!(portrait["tour"].as_array().unwrap().len() >= 2);
        assert!(
            plan(r#"{"autoTour": {"targets": 0}}"#)
                .unwrap_err()
                .contains("target")
        );
        assert!(plan(r#"{"frame": 1}"#).unwrap_err().contains("frame"));
        let mut error = ptr::null_mut();
        let settings = CString::new("{}").unwrap();
        assert!(
            unsafe {
                seiza_parallax_plan_scene_tour_json(ptr::null(), settings.as_ptr(), &mut error)
            }
            .is_null()
        );
        assert!(take_error(error).contains("null"));
        unsafe {
            seiza_parallax_free(toured);
            seiza_parallax_free(video);
        }
    }

    #[test]
    fn a_planned_tour_names_its_stops_and_goes_back_into_a_request() {
        let directory = tempfile::tempdir().unwrap();
        // Two catalogued objects in the test field, written beside the
        // other catalogs.
        let wcs = Wcs::from_center_scale_rotation((56.75, 24.12), (160.0, 120.0), 2.0, 0.0, false);
        let object = |name: &str, x: f64, y: f64, major: f32| {
            let (ra, dec) = wcs.pixel_to_world(x, y);
            seiza::objects::SkyObject {
                kind: seiza::objects::ObjectKind::Nebula,
                ra,
                dec,
                mag: None,
                major_arcmin: Some(major),
                minor_arcmin: None,
                position_angle_deg: None,
                name: name.into(),
                common_name: String::new(),
                metadata: Default::default(),
            }
        };
        seiza::objects::ObjectCatalog::new(vec![
            object("NGC 9001", 100.0, 80.0, 1.0),
            object("IC 9002", 230.0, 170.0, 0.8),
        ])
        .write_to(&directory.path().join("objects.bin"))
        .unwrap();
        let request = self::request(directory.path(), "\"autoTour\": {\"hold\": 1.0}");
        let mut error = ptr::null_mut();
        let planned = unsafe {
            seiza_parallax_plan_tour_json(request.as_ptr(), None, ptr::null_mut(), &mut error)
        };
        assert!(!planned.is_null(), "{}", take_error(error));
        let plan: serde_json::Value =
            serde_json::from_str(&unsafe { CStr::from_ptr(planned) }.to_string_lossy()).unwrap();
        unsafe { seiza_string_free(planned) };
        let tour = plan["tour"].as_array().unwrap();
        assert!(
            tour.first().unwrap()["focus"].is_null() && tour.last().unwrap()["focus"].is_null()
        );
        let names: Vec<&str> = tour
            .iter()
            .filter_map(|stop| stop["name"].as_str())
            .collect();
        assert_eq!(names.len(), 2, "{names:?}");
        assert!(names.contains(&"NGC 9001") && names.contains(&"IC 9002"));
        assert_eq!(plan["focusName"], "NGC 9001");
        // Each target's stop is titled with its name.
        for stop in tour {
            assert_eq!(stop["title"], stop["name"]);
        }
        // Drop the second target, retitle the first, and make a looping,
        // titled video from what is left.
        let kept: Vec<serde_json::Value> = tour
            .iter()
            .filter(|stop| stop["name"] != "IC 9002")
            .cloned()
            .map(|mut stop| {
                if stop["name"] == "NGC 9001" {
                    stop["title"] = "The first".into();
                }
                stop
            })
            .collect();
        let edited = self::request(
            directory.path(),
            &format!(
                "\"tour\": {}, \"distanceFocus\": {}, \"tourTitles\": true, \"tourLoop\": true",
                serde_json::to_string(&kept).unwrap(),
                plan["focus"]
            ),
        );
        let video = unsafe {
            seiza_parallax_prepare_json(
                edited.as_ptr(),
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        assert!(!video.is_null(), "{}", take_error(error));
        let summary = unsafe { seiza_parallax_summary_json(video, &mut error) };
        let parsed: serde_json::Value =
            serde_json::from_str(&unsafe { CStr::from_ptr(summary) }.to_string_lossy()).unwrap();
        unsafe { seiza_string_free(summary) };
        let seconds: f64 = kept
            .iter()
            .enumerate()
            .map(|(index, stop)| {
                stop["hold"].as_f64().unwrap()
                    + if index > 0 {
                        stop["travel"].as_f64().unwrap()
                    } else {
                        0.0
                    }
            })
            .sum();
        assert_eq!(parsed["frames"], (seconds * 5.0).round() as u64);
        unsafe { seiza_parallax_free(video) };
    }

    fn summary(video: *const SeizaParallax) -> serde_json::Value {
        let mut error = ptr::null_mut();
        let json = unsafe { seiza_parallax_summary_json(video, &mut error) };
        assert!(!json.is_null(), "{}", take_error(error));
        let parsed =
            serde_json::from_str(&unsafe { CStr::from_ptr(json) }.to_string_lossy()).unwrap();
        unsafe { seiza_string_free(json) };
        parsed
    }

    #[test]
    fn null_fields_take_their_defaults_and_a_huge_stride_is_refused() {
        let directory = tempfile::tempdir().unwrap();
        // As a .NET or Swift encoder writes options it has no value for.
        let request = self::request(
            directory.path(),
            "\"tour\": null, \"labels\": null, \"autoTour\": null, \"distanceFocus\": null",
        );
        let mut error = ptr::null_mut();
        let video = unsafe {
            seiza_parallax_prepare_json(
                request.as_ptr(),
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        assert!(!video.is_null(), "{}", take_error(error));
        let settings = CString::new(
            "{\"distanceParsecs\": null, \"maxStars\": null, \"tour\": null, \"dolly\": 0.3}",
        )
        .unwrap();
        let again = unsafe {
            seiza_parallax_reconfigure_json(
                video,
                settings.as_ptr(),
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        assert!(!again.is_null(), "{}", take_error(error));
        // A stride whose rows overflow is an error, not a panic.
        let mut pixels = vec![0_u8; 160 * 120 * 3];
        for stride in [usize::MAX, usize::MAX / 119 + 1] {
            let drawn = unsafe {
                seiza_parallax_render_frame(
                    video,
                    0,
                    SEIZA_PIXEL_FORMAT_RGB8,
                    pixels.as_mut_ptr(),
                    pixels.len(),
                    stride,
                    &mut error,
                )
            };
            assert!(!drawn);
            let message = take_error(error);
            assert!(message.contains("too large"), "{message}");
        }
        unsafe {
            seiza_parallax_free(again);
            seiza_parallax_free(video);
        }
    }

    /// Read `video`'s dust (`which`) as a `width` × `height` map, or
    /// `None` when it has none.
    fn dust_map(
        video: *const SeizaParallax,
        which: u32,
        width: u32,
        height: u32,
    ) -> Option<Vec<f32>> {
        let mut opacity = vec![-1.0_f32; (width * height) as usize];
        let mut error = ptr::null_mut();
        let written = unsafe {
            seiza_parallax_dust_map(
                video,
                which,
                width,
                height,
                opacity.as_mut_ptr(),
                opacity.len(),
                &mut error,
            )
        };
        assert!(written >= 0, "{}", take_error(error));
        (written == 1).then_some(opacity)
    }

    #[test]
    fn a_painted_dust_map_goes_in_comes_back_out_and_can_be_swapped() {
        let directory = tempfile::tempdir().unwrap();
        // Opaque everywhere, at a fifth of the image's size.
        let painted = directory.path().join("painted.png");
        image::GrayImage::from_pixel(64, 48, image::Luma([255]))
            .save(&painted)
            .unwrap();
        let request = request(
            directory.path(),
            &format!("\"dustMap\": {}", serde_json::json!(painted)),
        );
        let mut error = ptr::null_mut();
        let video = unsafe {
            seiza_parallax_prepare_json(
                request.as_ptr(),
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        assert!(!video.is_null(), "{}", take_error(error));
        let opaque = summary(video);
        assert_eq!(opaque["dustGiven"], true);
        // The one star behind the nebula at 400 pc, at 900 pc.
        assert_eq!(opaque["starsInFrontOfDust"], 1, "{opaque}");
        let map = dust_map(video, SEIZA_DUST_MAP_IN_USE, 16, 12).unwrap();
        assert!(map.iter().all(|&opacity| opacity == 1.0), "{map:?}");
        // Three stars are too few to count dust from.
        assert!(dust_map(video, SEIZA_DUST_MAP_COUNTED, 16, 12).is_none());

        // Painted clear over the left half: the star at x 250 stays behind.
        let half: Vec<f32> = (0..32 * 24)
            .map(|index| if index % 32 < 16 { 0.0 } else { 1.0 })
            .collect();
        let mut events: Vec<String> = Vec::new();
        let repainted = unsafe {
            seiza_parallax_with_dust_map(
                video,
                half.as_ptr(),
                32,
                24,
                ptr::null(),
                Some(collect_event),
                (&mut events as *mut Vec<String>).cast(),
                &mut error,
            )
        };
        assert!(!repainted.is_null(), "{}", take_error(error));
        assert!(
            events
                .iter()
                .any(|event| event.contains("dust from the given map")),
            "{events:?}"
        );
        assert_eq!(summary(repainted)["starsInFrontOfDust"], 1);
        let clear: Vec<f32> = vec![0.0; 32 * 24];
        let cleared = unsafe {
            seiza_parallax_with_dust_map(
                video,
                clear.as_ptr(),
                32,
                24,
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        assert!(!cleared.is_null(), "{}", take_error(error));
        assert_eq!(summary(cleared)["starsInFrontOfDust"], 0);
        // The first video keeps its own dust.
        assert_eq!(summary(video)["starsInFrontOfDust"], 1);

        // Back to the counted dust: none here.
        let counted = unsafe {
            seiza_parallax_with_dust_map(
                video,
                ptr::null(),
                0,
                0,
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        assert!(!counted.is_null(), "{}", take_error(error));
        assert!(dust_map(counted, SEIZA_DUST_MAP_IN_USE, 16, 12).is_none());
        assert!(summary(counted)["dustTransmission"].is_null());

        // Written out as a 16-bit PNG the image's size, white opaque.
        let written = directory.path().join("written.png");
        let request = CString::new(serde_json::json!({ "path": written }).to_string()).unwrap();
        assert_eq!(
            unsafe { seiza_parallax_write_dust_map_json(video, request.as_ptr(), &mut error) },
            1,
            "{}",
            take_error(error)
        );
        let read = image::open(&written).unwrap();
        assert_eq!(read.color(), image::ColorType::L16);
        assert_eq!((read.width(), read.height()), (320, 240));
        assert!(read.to_luma16().pixels().all(|pixel| pixel[0] == u16::MAX));
        assert_eq!(
            unsafe { seiza_parallax_write_dust_map_json(counted, request.as_ptr(), &mut error) },
            0
        );

        // A painted file read in, clear: no star in front; then back.
        let clear_file = directory.path().join("clear.tif");
        seiza_image_io::save(
            &clear_file,
            &seiza_image_io::FloatImage::gray(32, 24, vec![0.0; 32 * 24]),
        )
        .unwrap();
        let request = CString::new(serde_json::json!({ "path": clear_file }).to_string()).unwrap();
        let from_file = unsafe {
            seiza_parallax_with_dust_map_json(
                video,
                request.as_ptr(),
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        assert!(!from_file.is_null(), "{}", take_error(error));
        assert_eq!(summary(from_file)["dustGiven"], true);
        assert_eq!(summary(from_file)["starsInFrontOfDust"], 0);
        let back = CString::new("{}").unwrap();
        let counted_again = unsafe {
            seiza_parallax_with_dust_map_json(
                from_file,
                back.as_ptr(),
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        assert!(!counted_again.is_null(), "{}", take_error(error));
        assert!(dust_map(counted_again, SEIZA_DUST_MAP_IN_USE, 4, 3).is_none());
        let missing = CString::new(
            serde_json::json!({ "path": directory.path().join("missing.png") }).to_string(),
        )
        .unwrap();
        let refused = unsafe {
            seiza_parallax_with_dust_map_json(
                video,
                missing.as_ptr(),
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        assert!(refused.is_null());
        assert!(take_error(error).contains("missing.png"));
        for video in [from_file, counted_again] {
            unsafe { seiza_parallax_free(video) };
        }

        // A map of another shape, a short buffer and an unknown choice.
        let square: Vec<f32> = vec![0.5; 20 * 20];
        let refused = unsafe {
            seiza_parallax_with_dust_map(
                video,
                square.as_ptr(),
                20,
                20,
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        assert!(refused.is_null());
        assert!(take_error(error).contains("20×20"));
        let mut short = vec![0.0_f32; 10];
        let result = unsafe {
            seiza_parallax_dust_map(
                video,
                SEIZA_DUST_MAP_IN_USE,
                16,
                12,
                short.as_mut_ptr(),
                short.len(),
                &mut error,
            )
        };
        assert_eq!(result, -1);
        assert!(take_error(error).contains("192 values"));
        let result = unsafe {
            seiza_parallax_dust_map(video, 7, 1, 1, short.as_mut_ptr(), short.len(), &mut error)
        };
        assert_eq!(result, -1);
        assert!(take_error(error).contains("SEIZA_DUST_MAP_"));

        for video in [video, repainted, cleared, counted] {
            unsafe { seiza_parallax_free(video) };
        }
    }

    #[test]
    fn a_reconfigured_video_shares_the_scene_and_both_draw_at_once() {
        let directory = tempfile::tempdir().unwrap();
        let request = self::request(directory.path(), "");
        let mut error = ptr::null_mut();
        let first = unsafe {
            seiza_parallax_prepare_json(
                request.as_ptr(),
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        assert!(!first.is_null(), "{}", take_error(error));
        let settings = CString::new(
            serde_json::json!({
                "focus": [150, 110],
                "dolly": 0.6,
                "rotateDegrees": [0, -360],
                "size": "200x150",
                "seconds": 2,
                "fps": 5,
                "quality": "high",
            })
            .to_string(),
        )
        .unwrap();
        let second = unsafe {
            seiza_parallax_reconfigure_json(
                first,
                settings.as_ptr(),
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        assert!(!second.is_null(), "{}", take_error(error));
        let (before, after) = (summary(first), summary(second));
        assert_eq!(
            (before["frames"].as_u64(), before["width"].as_u64()),
            (Some(5), Some(160))
        );
        assert_eq!(
            (after["frames"].as_u64(), after["width"].as_u64()),
            (Some(10), Some(200))
        );
        assert_eq!(
            after["backgroundDistanceParsecs"],
            before["backgroundDistanceParsecs"]
        );
        assert_eq!(after["detectedStars"], before["detectedStars"]);
        assert_eq!(after["fit"]["zoom"]["asked"], 1.0);
        assert!(after["fit"]["zoom"]["used"].as_f64().unwrap() >= 1.0);
        // Both draw frames at once, on threads of their own.
        let (first_at, second_at) = (first as usize, second as usize);
        std::thread::scope(|scope| {
            for (address, (width, height)) in [(first_at, (160, 120)), (second_at, (200, 150))] {
                scope.spawn(move || {
                    let mut pixels = vec![0_u8; width * height * 3];
                    for index in 0..4 {
                        let mut error = ptr::null_mut();
                        assert!(unsafe {
                            seiza_parallax_render_frame(
                                address as *const SeizaParallax,
                                index,
                                SEIZA_PIXEL_FORMAT_RGB8,
                                pixels.as_mut_ptr(),
                                pixels.len(),
                                width * 3,
                                &mut error,
                            )
                        });
                    }
                });
            }
        });
        // Fields that would change the scene are refused by name, and
        // unknown ones as before.
        for (settings, expected) in [
            (
                "{\"distanceParsecs\": 500}",
                "\"distanceParsecs\" changes the prepared scene",
            ),
            (
                "{\"maxStars\": 10}",
                "\"maxStars\" changes the prepared scene",
            ),
            (
                "{\"distanceFocus\": [1, 2]}",
                "\"distanceFocus\" changes the prepared scene",
            ),
            ("{\"dollly\": 0.5}", "unknown field"),
        ] {
            let settings = CString::new(settings).unwrap();
            let refused = unsafe {
                seiza_parallax_reconfigure_json(
                    first,
                    settings.as_ptr(),
                    ptr::null(),
                    None,
                    ptr::null_mut(),
                    &mut error,
                )
            };
            assert!(refused.is_null());
            let message = take_error(error);
            assert!(message.contains(expected), "{message}");
        }
        // A cancelled reconfiguration gives nothing.
        let signal = seiza_cancel_signal_create();
        unsafe { seiza_cancel_signal_cancel(signal) };
        let cancelled = unsafe {
            seiza_parallax_reconfigure_json(
                first,
                settings.as_ptr(),
                signal,
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        assert!(cancelled.is_null());
        assert!(take_error(error).contains("stopped"));
        unsafe { seiza_cancel_signal_free(signal) };
        // The second outlives the first: the scene is theirs together.
        unsafe { seiza_parallax_free(first) };
        let mut pixels = vec![0_u8; 200 * 150 * 3];
        assert!(unsafe {
            seiza_parallax_render_frame(
                second,
                9,
                SEIZA_PIXEL_FORMAT_RGB8,
                pixels.as_mut_ptr(),
                pixels.len(),
                600,
                &mut error,
            )
        });
        unsafe { seiza_parallax_free(second) };
    }

    #[test]
    fn depth_planes_go_in_as_masks_and_come_back_in_the_summary() {
        let directory = tempfile::tempdir().unwrap();
        // A mask at half the image's size about a bright cloud.
        let mask = directory.path().join("knot.png");
        image::GrayImage::from_fn(160, 120, |x, y| {
            let inside = (x as f64 - 100.0).hypot(y as f64 - 60.0) < 20.0;
            image::Luma([if inside { 255 } else { 0 }])
        })
        .save(&mask)
        .unwrap();
        let asked = request(
            directory.path(),
            &format!(
                "\"planes\": [{{\"mask\": {}, \"distanceParsecs\": 150}}]",
                serde_json::json!(mask)
            ),
        );
        image::RgbImage::from_fn(320, 240, |x, y| {
            let r2 = (x as f64 - 200.0).powi(2) + (y as f64 - 120.0).powi(2);
            let cloud = (120.0 * (-r2 / 300.0).exp()) as u8;
            image::Rgb([20 + (x / 8) as u8 + cloud, 30 + cloud, 25 + cloud])
        })
        .save(directory.path().join("starless.png"))
        .unwrap();
        let mut events: Vec<String> = Vec::new();
        let mut error = ptr::null_mut();
        let video = unsafe {
            seiza_parallax_prepare_json(
                asked.as_ptr(),
                ptr::null(),
                Some(collect_event),
                (&mut events as *mut Vec<String>).cast(),
                &mut error,
            )
        };
        assert!(!video.is_null(), "{}", take_error(error));
        let planes = &summary(video)["planes"];
        assert_eq!(planes[0]["name"], "knot");
        assert_eq!(planes[0]["distanceParsecs"], 150.0);
        assert_eq!(planes[0]["basis"], "given");
        assert!(planes[0]["object"].is_null());
        let coverage = planes[0]["coverage"].as_f64().unwrap();
        // A disc of 40 pixels on the 320 by 240 image.
        let disc = std::f64::consts::PI * 40.0 * 40.0 / (320.0 * 240.0);
        assert!((coverage / disc - 1.0).abs() < 0.05, "{planes}");
        assert!(
            events
                .iter()
                .any(|event| event.contains("depth plane knot at 150 pc")),
            "{events:?}"
        );
        // Planes make the scene, so a new video of it cannot change them.
        let settings = CString::new(format!(
            "{{\"planes\": [{{\"mask\": {}, \"distanceParsecs\": 90}}]}}",
            serde_json::json!(mask)
        ))
        .unwrap();
        let refused = unsafe {
            seiza_parallax_reconfigure_json(
                video,
                settings.as_ptr(),
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        assert!(refused.is_null());
        assert!(take_error(error).contains("planes"));
        unsafe { seiza_parallax_free(video) };
        // A mask of another shape is refused, naming the plane.
        let square = directory.path().join("square.png");
        image::GrayImage::from_pixel(50, 50, image::Luma([255]))
            .save(&square)
            .unwrap();
        let wrong = request(
            directory.path(),
            &format!(
                "\"planes\": [{{\"mask\": {}, \"distanceParsecs\": 150, \"name\": \"sq\"}}]",
                serde_json::json!(square)
            ),
        );
        let video = unsafe {
            seiza_parallax_prepare_json(
                wrong.as_ptr(),
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        assert!(video.is_null());
        let message = take_error(error);
        assert!(message.contains("sq: the plane mask is 50×50"), "{message}");
    }

    #[test]
    fn a_request_with_an_unknown_field_or_bad_choice_is_refused() {
        let directory = tempfile::tempdir().unwrap();
        for (extra, expected) in [
            ("\"dollly\": 0.5", "unknown field"),
            ("\"start\": \"middle\"", "start must be one of"),
            ("\"size\": \"8k\"", "expected WIDTHxHEIGHT"),
            ("\"dolly\": 1.5", "dolly"),
            ("\"tour\": [{\"dolly\": 0.5}]", "at least two stops"),
            ("\"tour\": [{}, {\"dolli\": 0.5}]", "unknown field"),
        ] {
            let request = request(directory.path(), extra);
            let mut error = ptr::null_mut();
            let video = unsafe {
                seiza_parallax_prepare_json(
                    request.as_ptr(),
                    ptr::null(),
                    None,
                    ptr::null_mut(),
                    &mut error,
                )
            };
            assert!(video.is_null());
            let message = take_error(error);
            assert!(message.contains(expected), "{extra}: {message}");
        }
    }
}
