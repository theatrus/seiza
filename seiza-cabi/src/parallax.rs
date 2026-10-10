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
    CustomLabel, Easing, Event, FfmpegSink, FrameFn, FrameSink, Parallax, ParallaxOptions,
    PngSequence, Quality, SmallStars, Start,
};
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
/// once; do not free it while any call using it is running.
pub struct SeizaParallax {
    video: Parallax,
    wcs: Wcs,
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

/// Everything a parallax video takes. Fields left out keep `seiza
/// parallax-video`'s defaults; a field not listed here is an error, so a
/// typo cannot silently run the defaults.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ParallaxRequest {
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
    focus: Option<[f64; 2]>,
    distance_parsecs: Option<f64>,
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
    /// "focus" or "whole".
    start: Option<String>,
    dolly: Option<f64>,
    truck: Option<f64>,
    truck_angle_degrees: Option<f64>,
    pan: Option<f64>,
    zoom: Option<f64>,
    zoom_end: Option<f64>,
    /// The frame's turn at the first and last frames, degrees
    /// anticlockwise.
    rotate_degrees: Option<[f64; 2]>,
    /// "inOut" or "linear".
    easing: Option<String>,
    /// "standard" or "high".
    quality: Option<String>,
    growth_limit: Option<f64>,
    fade_from: Option<f64>,
    /// A tour of stops instead of the single move.
    #[serde(default)]
    tour: Vec<StopRequest>,
    /// Plan a tour of the catalogued objects in the field when `tour` is
    /// empty: `{targets, hold, motion}`, each optional.
    auto_tour: Option<AutoTourRequest>,
    /// "720p", "1080p", "1440p" or "4k", each with "-portrait" for the tall
    /// form, or "WIDTHxHEIGHT".
    size: Option<String>,
    seconds: Option<f64>,
    fps: Option<u32>,
    overlay: Option<bool>,
    overlay_density: Option<f64>,
    #[serde(default)]
    labels: Vec<LabelRequest>,
    /// "#RRGGBB".
    label_color: Option<String>,
    /// true for "Rendered with seiza.fyi", or the text to write.
    watermark: Option<WatermarkRequest>,
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

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum WatermarkRequest {
    Shown(bool),
    Text(String),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WcsRequest {
    crval: [f64; 2],
    crpix: [f64; 2],
    cd: [[f64; 2]; 2],
    #[serde(default)]
    sip: Option<SipRequest>,
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
    unmatched_distance_parsecs: f64,
    flying_stars: usize,
    galaxies_lifted: Vec<String>,
    dust_transmission: Option<DustResponse>,
    labelled_objects: usize,
    wcs: WcsResponse,
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

fn open_display(path: &Path) -> Result<Rgb32FImage, String> {
    Ok(seiza::raster::open_oriented(path)
        .map_err(|error| format!("failed to open {}: {error}", path.display()))?
        .pixels
        .to_rgb32f())
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
    let defaults = ParallaxOptions::default();
    let catalogs = request.catalog_directory.as_deref();
    // A star distance file the request names must be there; one beside the
    // other catalogs is taken only if it is.
    let star_distances = request.star_distances.clone().or_else(|| {
        catalogs
            .map(|directory| directory.join("star-distances.bin"))
            .filter(|path| path.is_file())
    });
    Ok(ParallaxOptions {
        focus: request.focus.map(|[x, y]| (x, y)),
        distance_pc: request.distance_parsecs,
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
                }
            })
            .collect(),
        auto_tour: request.auto_tour.as_ref().map(AutoTourRequest::auto_tour),
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
        watermark: match &request.watermark {
            None | Some(WatermarkRequest::Shown(false)) => None,
            Some(WatermarkRequest::Shown(true)) => Some(seiza_parallax::DEFAULT_WATERMARK.into()),
            Some(WatermarkRequest::Text(text)) => Some(text.clone()),
        },
    })
}

impl WcsRequest {
    fn wcs(&self) -> Wcs {
        Wcs {
            crval: (self.crval[0], self.crval[1]),
            crpix: (self.crpix[0], self.crpix[1]),
            cd: self.cd,
            sip: self.sip.as_ref().map(|sip| Sip {
                order: sip.order,
                a: sip.a.clone(),
                b: sip.b.clone(),
                ap: sip.ap.clone(),
                bp: sip.bp.clone(),
            }),
        }
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
    if let (Some(starless), Some(stars)) = (&request.starless, &request.stars) {
        let solve = request.image.clone().unwrap_or_else(|| stars.clone());
        return Ok((open_display(starless)?, open_display(stars)?, solve));
    }
    if request.starless.is_some() || request.stars.is_some() {
        return Err("give both starless and stars, or neither".into());
    }
    let path = request
        .image
        .clone()
        .ok_or("give an image to split, or starless and stars")?;
    let image = open_display(&path)?;
    let cli = rc_astro_cli(
        request.rc_astro_executable.clone(),
        request.rc_astro_host.clone(),
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
/// The request gives either `image` (a stretched PNG, JPEG or TIFF), which
/// StarXTerminator splits through the `rc-astro` CLI, or `starless` and
/// `stars` (the split, with the stars unscreened), plus `image` to solve
/// when it differs from `stars`. Without `wcs` (`{crval, crpix, cd, sip}`
/// as `seiza_solve_image_json` returns it) the image is blind-solved
/// against the catalogs in `catalogDirectory`, between
/// `minimumScaleArcsecPerPixel` and `maximumScaleArcsecPerPixel` (0.1 and
/// 1000 if absent). The other fields are `seiza parallax-video`'s options
/// in camelCase: `focus` `[x, y]`, `distanceParsecs`,
/// `unmatchedDistanceParsecs`, `objects`, `objectDistances`,
/// `starDistances`, `gaiaMaxMagnitude`, `gaiaCache`, `online`, `maxStars`,
/// `smallStars` ("drop", "field"), `keepGalaxies`, `dust`, `dustOpacity`,
/// `start` ("focus", "whole"), `dolly`, `truck`, `truckAngleDegrees`,
/// `pan`, `zoom`, `zoomEnd`, `rotateDegrees` `[first, last]`, `easing`
/// ("inOut", "linear"), `quality` ("standard", "high"), `growthLimit`,
/// `fadeFrom`, `tour` (stops `[{focus, dolly, zoom, rotateDegrees, pan,
/// travel, hold, spinDegrees}]`, the first the opening view, which replace the single
/// move and set the length; [`seiza_parallax_plan_tour_json`] plans
/// one), `autoTour` (`{targets, hold, motion}`: plan a tour of the
/// catalogued objects and render it), `size` ("720p", "1080p", "1440p", "4k", each with
/// "-portrait", or "WIDTHxHEIGHT"), `seconds`, `fps`, `overlay`,
/// `overlayDensity`, `labels` (`[{x, y, radius, text}]`), `labelColor`
/// ("#RRGGBB") and `watermark` (true, or the text). An unknown field is an
/// error.
///
/// Stars' distances come from the star distance file when installed, else
/// from the Gaia and VizieR archives unless `online` is false. `cancel`
/// stops StarXTerminator's split; the rest of the preparation runs to the
/// end. `events` (nullable) hears each step on the calling thread, with
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
    ffi_result(error_out, || {
        let request_json = required_str(request_json, "parallax request JSON")?;
        let request: ParallaxRequest = serde_json::from_str(&request_json)
            .map_err(|error| format!("invalid parallax request JSON: {error}"))?;
        let options = options(&request)?;
        options.check().map_err(|error| error.to_string())?;
        let cancellation = unsafe { cancel.as_ref() }
            .map(|signal| seiza_stacking::CancelSignal::from(signal.cancelled.clone()));
        let (starless, stars, solve_path) =
            split(&request, cancellation.as_ref(), events, context)?;
        let wcs = solution(&request, &solve_path, events, context)?;
        let video = Parallax::prepare(&starless, &stars, &wcs, &options, &mut |event| {
            send(events, context, event_json(event));
        })
        .map_err(|error| error.to_string())?;
        Ok(Box::into_raw(Box::new(SeizaParallax { video, wcs })))
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
    if let Some(wcs) = &request.wcs {
        return Ok(wcs.wcs());
    }
    let minimum = request.minimum_scale_arcsec_per_pixel.unwrap_or(0.1);
    let maximum = request.maximum_scale_arcsec_per_pixel.unwrap_or(1000.0);
    let solved = blind_solve_path(
        solve_path,
        request.catalog_directory.as_deref(),
        minimum,
        maximum,
        2,
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

/// Plan a tour of the catalogued objects in an image, for the caller to
/// edit before making the video. The request is
/// [`seiza_parallax_prepare_json`]'s: the image (`image`, else `stars` or
/// `starless`) gives the size, and is blind-solved without `wcs`; `size`
/// is the video's frame; `objects` or `catalogDirectory` the object
/// catalog; and `autoTour` (`{targets, hold, motion}`, each optional) how
/// to plan. Returns JSON, released with `seiza_string_free`:
/// `{"focus": [x, y], "focusName", "seconds", "tour": [{name, focus,
/// dolly, zoom, rotateDegrees, pan, travel, hold}]}`. Its `tour`, with any
/// stops dropped, moved or changed, and its `focus` go back into a prepare
/// request as they are. Null with `error_out` set on failure; `events`
/// (nullable) hears the solve.
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
        let request: ParallaxRequest = serde_json::from_str(&request_json)
            .map_err(|error| format!("invalid parallax request JSON: {error}"))?;
        let options = options(&request)?;
        let path = request
            .image
            .clone()
            .or_else(|| request.stars.clone())
            .or_else(|| request.starless.clone())
            .ok_or("give the image to plan a tour of")?;
        let dimensions = open_display(&path)?.dimensions();
        let wcs = solution(&request, &path, events, context)?;
        let auto = request
            .auto_tour
            .as_ref()
            .map(AutoTourRequest::auto_tour)
            .unwrap_or_default();
        let plan = seiza_parallax::plan_tour(
            &wcs,
            dimensions,
            options.objects.as_deref(),
            options.size,
            &auto,
        )
        .map_err(|error| error.to_string())?;
        let seconds = ParallaxOptions {
            tour: plan.tour(),
            ..ParallaxOptions::default()
        }
        .seconds();
        owned_json(&PlannedTourResponse {
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
                })
                .collect(),
        })
    })
    .unwrap_or(ptr::null_mut())
}

/// What preparing the video found, as JSON: its frame count, frame rate
/// and size, the stars found and matched, the nebula's distance and how it
/// was found, galaxies lifted, the dust, objects labelled, and the WCS
/// used. Returns a string released with `seiza_string_free`, or null with
/// `error_out` set.
///
/// # Safety
///
/// `video` must be a live pointer from [`seiza_parallax_prepare_json`].
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
        let wcs = &owner.wcs;
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
            unmatched_distance_parsecs: summary.unmatched_distance_pc,
            flying_stars: summary.flying_stars,
            galaxies_lifted: summary.galaxies_lifted.clone(),
            dust_transmission: summary
                .dust_transmission
                .map(|(median, thickest)| DustResponse { median, thickest }),
            labelled_objects: summary.labelled_objects,
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
            },
        })
    })
    .unwrap_or(ptr::null_mut())
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
/// `video` must be a live pointer from [`seiza_parallax_prepare_json`].
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
        if stride < width * bytes {
            return Err(format!(
                "a row needs {} bytes; the stride is {stride}",
                width * bytes
            ));
        }
        let needed = stride * (height - 1) + width * bytes;
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
/// `video` must be a live pointer from [`seiza_parallax_prepare_json`].
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
/// `video` must be a live pointer from [`seiza_parallax_prepare_json`].
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
/// `video` must be null or a pointer from [`seiza_parallax_prepare_json`]
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
        // Drop the second target and make the video from what is left.
        let kept: Vec<&serde_json::Value> = tour
            .iter()
            .filter(|stop| stop["name"] != "IC 9002")
            .collect();
        let edited = self::request(
            directory.path(),
            &format!(
                "\"tour\": {}, \"focus\": {}",
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
