//! Parallax fly-through videos (`seiza-parallax`), as `seiza parallax-video`
//! makes them: [`PyParallaxVideo`] prepares a video from a starless image,
//! its stars and a plate solution, then draws frames as numpy arrays one at
//! a time, in turn to a callback (any encoder: PyAV, OpenCV, a platform
//! writer), or into a video file.

use crate::PyWcs;
use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use seiza_parallax::{
    Corner, CustomLabel, DepthPlane, DistanceOverride, DustMap, Easing, Event, FfmpegSink,
    FrameSink, FrameSize, Framing, Parallax, ParallaxOptions, PlaneMask, PlannedTour, PngSequence,
    Quality, SmallStars, Start,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyParallaxVideo>()?;
    module.add_class::<PyParallaxFrames>()?;
    module.add_function(wrap_pyfunction!(plan_parallax_tour, module)?)?;
    Ok(())
}

/// How to plan a tour, from `auto_tour=`: True for every target worth a
/// visit, a number for that many, or a dict of `targets`, `hold` and
/// `motion`.
fn auto_tour_argument(value: &Bound<'_, PyAny>) -> PyResult<Option<seiza_parallax::AutoTour>> {
    let mut auto = seiza_parallax::AutoTour::default();
    if let Ok(flag) = value.extract::<bool>() {
        return Ok(flag.then_some(auto));
    }
    if let Ok(count) = value.extract::<usize>() {
        auto.targets = Some(count);
        return Ok(Some(auto));
    }
    let dict = value.cast::<PyDict>()?;
    for (key, value) in dict.iter() {
        let key: String = key.extract()?;
        match key.as_str() {
            "targets" => auto.targets = value.extract()?,
            "hold" => auto.hold = value.extract()?,
            "motion" => auto.motion = value.extract()?,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown auto_tour field {other:?}"
                )));
            }
        }
    }
    Ok(Some(auto))
}

/// Plan a tour of the catalogued objects in a `width` × `height` image
/// whose sky `wcs` gives, for frames of `size`, to edit before making the
/// video: every target worth a visit (or the `targets` most worth it),
/// visited in a short round from the whole image and back to a final
/// drift, staying `hold` seconds at each and turning and panning as much
/// as `motion` says. Returns a dict: `tour`, the stops as dicts (`name`,
/// `focus`, `dolly`, `zoom`, `rotate_deg`, `pan`, `travel`, `hold`,
/// `spin_deg`, `push`, and `title`, the target's name, which
/// `tour_titles=True` shows), and `focus` with
/// `focus_name`, the target most worth a visit, where the nebula's distance
/// is best taken. Drop, move or change stops, then pass `tour=plan["tour"],
/// distance_focus=plan["focus"]` to `ParallaxVideo`. With `crop` (x, y,
/// width, height), the plan is of that part of the image, as
/// `ParallaxVideo(crop=...)` films it; `framing` is the video's.
#[pyfunction]
#[pyo3(signature = (width, height, wcs, *, objects=None, size=None, targets=None, hold=1.5, motion=1.0, crop=None, framing="fill"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_parallax_tour<'py>(
    py: Python<'py>,
    width: u32,
    height: u32,
    wcs: &Bound<'py, PyWcs>,
    objects: Option<PathBuf>,
    size: Option<&Bound<'py, PyAny>>,
    targets: Option<usize>,
    hold: f64,
    motion: f64,
    crop: Option<(usize, usize, usize, usize)>,
    framing: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let wcs = wcs.get().wcs.clone();
    let (wcs, (width, height)) = match crop {
        Some(crop) => seiza_parallax::crop_wcs(&wcs, (width, height), crop).map_err(video_error)?,
        None => (wcs, (width, height)),
    };
    let frame = match size {
        None => seiza_parallax::VideoOptions::default().size,
        Some(size) => frame_size_argument(size)?.for_image((width as usize, height as usize)),
    };
    let auto = seiza_parallax::AutoTour {
        targets,
        hold,
        motion,
        framing: framing_argument(framing)?,
    };
    let plan = py
        .detach(|| {
            seiza_parallax::plan_tour(&wcs, (width, height), objects.as_deref(), frame, &auto)
        })
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    plan_dict(py, &plan)
}

/// A frame size from `size=`: a name such as "1080p", "4k-portrait" or
/// "4k-image", "WIDTHxHEIGHT", or (width, height).
fn frame_size_argument(value: &Bound<'_, PyAny>) -> PyResult<FrameSize> {
    match value.extract::<String>() {
        Ok(text) => seiza_parallax::parse_frame_size_or_shape(&text).map_err(PyValueError::new_err),
        Err(_) => {
            let (width, height) = pair(value)?;
            Ok(FrameSize::Fixed(width, height))
        }
    }
}

/// "fill" or "fit", from `framing=`.
fn framing_argument(value: &str) -> PyResult<Framing> {
    choice(
        value,
        "framing",
        &[("fill", Framing::Fill), ("fit", Framing::Fit)],
    )
}

/// A planned tour as Python sees it: its stops as dicts `tour=` takes back,
/// each named for its target, and the focus where the nebula's distance is
/// best taken.
fn plan_dict<'py>(py: Python<'py>, plan: &PlannedTour) -> PyResult<Bound<'py, PyDict>> {
    let stops = plan
        .stops
        .iter()
        .map(|planned| {
            let stop = PyDict::new(py);
            stop.set_item("name", &planned.name)?;
            stop.set_item("focus", planned.stop.focus)?;
            stop.set_item("dolly", planned.stop.dolly)?;
            stop.set_item("zoom", planned.stop.zoom)?;
            stop.set_item("rotate_deg", planned.stop.rotate_deg)?;
            stop.set_item("pan", planned.stop.pan)?;
            stop.set_item("travel", planned.stop.travel)?;
            stop.set_item("hold", planned.stop.hold)?;
            stop.set_item("spin_deg", planned.stop.spin_deg)?;
            stop.set_item("push", planned.stop.push)?;
            stop.set_item("title", &planned.stop.title)?;
            stop.set_item("depth_pc", planned.stop.depth_pc)?;
            Ok(stop)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let seconds = seiza_parallax::VideoOptions {
        tour: plan.tour(),
        ..seiza_parallax::VideoOptions::default()
    }
    .seconds();
    let result = PyDict::new(py);
    result.set_item("tour", stops)?;
    result.set_item("focus", plan.focus)?;
    result.set_item("focus_name", &plan.focus_name)?;
    result.set_item("seconds", seconds)?;
    Ok(result)
}

/// A prepared parallax video: the stars of a starless image and its stars
/// placed at their Gaia distances, galaxies lifted onto the far field, the
/// dust mapped, and a camera move fitted to the image.
///
/// `starless` and `stars` are the stretched image split into its starless
/// image and unscreened stars, each a path to a PNG, JPEG, TIFF, FITS or
/// XISF, or a numpy array of shape (height, width, 3), float32 from 0 to 1
/// or uint8;
/// `wcs` is the image's plate solution (`seiza.solve_blind(...).wcs`).
/// The other keyword arguments are `seiza parallax-video`'s options, where
/// a pair is a tuple or a list:
/// `focus` (x, y), `distance_pc`, `distance_focus` (x, y),
/// `unmatched_distance_pc`, `objects`,
/// `object_distances`, `star_distances`, `gaia_max_mag`, `gaia_cache`,
/// `online`, `max_stars`, `small_stars` ("drop", "field"), `keep_galaxies`,
/// `dust`, `dust_opacity`, `crop` (x, y, width, height in image pixels:
/// film that part of the image, every other image point counting in it),
/// `start` ("focus", "whole"), `framing` ("fill", the default, fills the
/// frame and cuts the ends of the image's longer side when the image and
/// frame differ in shape; "fit" shows all the image at the widest, with
/// black beside it), `dolly`, `truck`,
/// `truck_angle_deg`, `pan`, `zoom`, `zoom_end`, `rotate_deg` (first,
/// last), `easing` ("in_out", "linear"), `quality` ("standard", "high"),
/// `growth_limit`, `fade_from`, `tour` (stops, each a string as `seiza
/// parallax-video --stop` takes, "X,Y dolly=0.8 rotate=-10 travel=6
/// hold=1" or "whole ...", or a dict of `focus`, `dolly`, `zoom`,
/// `rotate_deg`, `pan`, `travel`, `hold`, `spin_deg`, `push`, `title` and
/// `card` (an image shown as the title would be, in its place), as
/// `plan_parallax_tour` gives them, with its `name`), `auto_tour`
/// (True, a count, or a dict of `targets`, `hold` and `motion`: plan a
/// tour of the catalogued objects and render it), `tour_glide`,
/// `tour_titles`, `tour_loop`, `size` ("720p", "1080p", "1440p", "4k",
/// each with "-portrait", or with "-image" for the image's own shape with
/// that longer side, so "4k-image" films a 4:3 image at 3840x2880, all of
/// it; "WIDTHxHEIGHT", or (width, height)), `seconds`,
/// `fps`, `overlay`, `overlay_density`, `labels` ((x, y, text) or (x, y,
/// radius, text) tuples), `label_color` ("#RRGGBB"), `watermark` (True,
/// the text, or a dict of `text`, `image` (a picture such as a logo, its
/// transparency kept), `corner` ("bottom_right", "bottom_left", "top_right"
/// or "top_left"), `size` (the image's height as a share of the frame's
/// shorter side, 0.06) and `opacity` (0.8)), and `intro_card`,
/// `outro_card` and `card_seconds` (images laid over the start and the
/// end, each shown that many seconds, 3 by default). `progress` hears
/// each step as a dict with a "kind" of "note" or "warning" and a
/// "message". `cancel()` returning True, or Ctrl-C, stops preparing; a
/// `progress` that raises stops it too, and its exception is raised.
#[pyclass(name = "ParallaxVideo", module = "seiza", frozen)]
pub(crate) struct PyParallaxVideo {
    video: Parallax,
}

/// The frames of a [`PyParallaxVideo`], in order, as RGB numpy arrays.
#[pyclass(name = "ParallaxFrames", module = "seiza")]
pub(crate) struct PyParallaxFrames {
    video: Py<PyParallaxVideo>,
    next: usize,
}

#[pymethods]
impl PyParallaxFrames {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyArray3<u8>>>> {
        let video = self.video.get();
        if self.next >= video.video.frames() {
            return Ok(None);
        }
        let index = self.next;
        self.next += 1;
        video.frame(py, index, "rgb").map(Some)
    }
}

/// An image argument as given: a file to open, or its pixels.
enum ImageArgument {
    Path(PathBuf),
    Pixels(image::Rgb32FImage),
}

impl ImageArgument {
    /// The pixels, opening a file if need be; call it without the GIL, as
    /// a large file takes a while to decode.
    fn pixels(self) -> PyResult<image::Rgb32FImage> {
        match self {
            Self::Pixels(pixels) => Ok(pixels),
            Self::Path(path) => Ok(seiza_image_io::open(&path)
                .map_err(|error| PyValueError::new_err(error.to_string()))?
                .to_rgb()),
        }
    }
}

/// An image argument: a path, or a numpy array of float32 0 to 1 or uint8,
/// in any memory layout.
fn image_argument(value: &Bound<'_, PyAny>, name: &str) -> PyResult<ImageArgument> {
    if let Ok(path) = value.extract::<PathBuf>() {
        return Ok(ImageArgument::Path(path));
    }
    let shape_error =
        || PyValueError::new_err(format!("{name} must have shape (height, width, 3)"));
    let raw = |shape: &[usize]| match shape {
        [height, width, 3] => Ok((*width as u32, *height as u32)),
        _ => Err(shape_error()),
    };
    // Read in index order, whatever the array's strides.
    let pixels = if let Ok(array) = value.extract::<PyReadonlyArrayDyn<'_, f32>>() {
        let (width, height) = raw(array.shape())?;
        let data = array.as_array().iter().copied().collect();
        image::Rgb32FImage::from_raw(width, height, data)
    } else if let Ok(array) = value.extract::<PyReadonlyArrayDyn<'_, u8>>() {
        let (width, height) = raw(array.shape())?;
        let data = array
            .as_array()
            .iter()
            .map(|&value| value as f32 / 255.0)
            .collect();
        image::Rgb32FImage::from_raw(width, height, data)
    } else {
        return Err(PyValueError::new_err(format!(
            "{name} must be a path or a numpy array of float32 or uint8"
        )));
    };
    pixels.map(ImageArgument::Pixels).ok_or_else(shape_error)
}

/// A one-channel image of values 0 to 1, from a 2-D numpy array of float32
/// 0 to 1 or uint8, as (width, height, values row by row).
fn unit_array(value: &Bound<'_, PyAny>, name: &str) -> PyResult<(usize, usize, Vec<f32>)> {
    let shape = |shape: &[usize]| match shape {
        [height, width] => Ok((*width, *height)),
        _ => Err(PyValueError::new_err(format!(
            "{name} must have shape (height, width)"
        ))),
    };
    if let Ok(array) = value.extract::<PyReadonlyArrayDyn<'_, f32>>() {
        let (width, height) = shape(array.shape())?;
        return Ok((width, height, array.as_array().iter().copied().collect()));
    }
    if let Ok(array) = value.extract::<PyReadonlyArrayDyn<'_, u8>>() {
        let (width, height) = shape(array.shape())?;
        let values = array.as_array().iter().map(|&v| v as f32 / 255.0).collect();
        return Ok((width, height, values));
    }
    Err(PyValueError::new_err(format!(
        "{name} must be a path or a 2-D numpy array of float32 or uint8"
    )))
}

/// A dust map from a path (PNG, JPEG, TIFF, FITS or XISF) or a 2-D array,
/// white where the dust is opaque.
fn dust_map_argument(value: &Bound<'_, PyAny>) -> PyResult<DustMap> {
    let map = if let Ok(path) = value.extract::<PathBuf>() {
        DustMap::open(&path).map_err(PyValueError::new_err)?
    } else {
        let (width, height, opacity) = unit_array(value, "dust_map")?;
        DustMap {
            width,
            height,
            opacity,
        }
    };
    map.check().map_err(PyValueError::new_err)?;
    Ok(map)
}

/// Depth planes from `planes=`: (mask, distance_pc) or (mask, distance_pc,
/// name), each mask a path or a 2-D array, white where the image belongs to
/// the plane, at any size of the image's shape. A plane takes its mask
/// file's name, or "plane N", unless named.
fn planes_argument(value: &Bound<'_, PyAny>) -> PyResult<Vec<DepthPlane>> {
    value
        .try_iter()?
        .enumerate()
        .map(|(index, plane)| {
            let plane = plane?;
            let parts: Vec<Bound<'_, PyAny>> = plane.try_iter()?.collect::<PyResult<_>>()?;
            let (mask, distance_pc, name) = match parts.as_slice() {
                [mask, distance] => (mask, distance.extract::<f64>()?, None),
                [mask, distance, name] => (mask, distance.extract()?, Some(name.extract()?)),
                _ => {
                    return Err(PyValueError::new_err(
                        "a plane is (mask, distance_pc) or (mask, distance_pc, name)",
                    ));
                }
            };
            let (mask, default_name) = if let Ok(path) = mask.extract::<PathBuf>() {
                let stem = path.file_stem().map_or_else(
                    || path.display().to_string(),
                    |stem| stem.to_string_lossy().into_owned(),
                );
                (PlaneMask::open(&path).map_err(PyValueError::new_err)?, stem)
            } else {
                let (width, height, coverage) = unit_array(mask, "a plane's mask")?;
                let mask = PlaneMask {
                    width,
                    height,
                    coverage,
                };
                mask.check().map_err(PyValueError::new_err)?;
                (mask, format!("plane {}", index + 1))
            };
            Ok(DepthPlane {
                mask,
                distance_pc,
                name: name.unwrap_or(default_name),
            })
        })
        .collect()
}

/// Distances for catalogued objects from `distance_overrides=`: a dict of
/// name to parsecs, or (name, parsecs) pairs.
fn distance_overrides_argument(value: &Bound<'_, PyAny>) -> PyResult<Vec<DistanceOverride>> {
    let pairs: Vec<(String, f64)> = match value.cast::<PyDict>() {
        Ok(dict) => dict
            .iter()
            .map(|(name, distance)| Ok((name.extract()?, distance.extract()?)))
            .collect::<PyResult<_>>()?,
        Err(_) => value
            .try_iter()?
            .map(|pair| pair?.extract())
            .collect::<PyResult<_>>()?,
    };
    Ok(pairs
        .into_iter()
        .map(|(object, distance_pc)| DistanceOverride {
            object,
            distance_pc,
        })
        .collect())
}

/// Two numbers from a two-item tuple or list.
fn pair<'py, T: pyo3::conversion::FromPyObjectOwned<'py>>(
    value: &Bound<'py, PyAny>,
) -> PyResult<(T, T)> {
    let [a, b]: [T; 2] = value.extract()?;
    Ok((a, b))
}

/// A video error as Python sees it: bad options are a `ValueError`.
fn video_error(error: seiza_parallax::pipeline::Error) -> PyErr {
    match error {
        seiza_parallax::pipeline::Error::Invalid(message) => PyValueError::new_err(message),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

fn choice<T: Copy>(value: &str, name: &str, choices: &[(&str, T)]) -> PyResult<T> {
    choices
        .iter()
        .find(|(key, _)| *key == value)
        .map(|(_, choice)| *choice)
        .ok_or_else(|| {
            let keys: Vec<&str> = choices.iter().map(|(key, _)| *key).collect();
            PyValueError::new_err(format!(
                "{name} must be one of {}; got {value:?}",
                keys.join(", ")
            ))
        })
}

/// The keyword arguments that make the scene: changing any needs a new
/// `ParallaxVideo`, and `reconfigure` refuses them.
const SCENE_KEYS: [&str; 19] = [
    "distance_pc",
    "distance_focus",
    "unmatched_distance_pc",
    "objects",
    "object_distances",
    "star_distances",
    "gaia_max_mag",
    "gaia_cache",
    "online",
    "max_stars",
    "small_stars",
    "keep_galaxies",
    "dust",
    "dust_opacity",
    "dust_map",
    "planes",
    "auto_planes",
    "distance_overrides",
    "crop",
];

/// The video's options from keyword arguments, over the defaults.
fn options(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<ParallaxOptions> {
    let mut options = ParallaxOptions::default();
    let Some(kwargs) = kwargs else {
        return Ok(options);
    };
    for (key, value) in kwargs.iter() {
        let key: String = key.extract()?;
        if value.is_none() {
            continue;
        }
        match key.as_str() {
            "focus" => options.video.focus = Some(pair(&value)?),
            "distance_pc" => options.scene.distance_pc = Some(value.extract()?),
            "distance_focus" => options.scene.distance_focus = Some(pair(&value)?),
            "unmatched_distance_pc" => options.scene.unmatched_distance_pc = Some(value.extract()?),
            "objects" => options.scene.objects = Some(value.extract()?),
            "object_distances" => options.scene.object_distances = Some(value.extract()?),
            "star_distances" => options.scene.star_distances = Some(value.extract()?),
            "gaia_max_mag" => options.scene.gaia_max_mag = value.extract()?,
            "gaia_cache" => options.scene.gaia_cache = Some(value.extract()?),
            "online" => options.scene.online = value.extract()?,
            "max_stars" => options.scene.max_stars = Some(value.extract()?),
            "small_stars" => {
                options.scene.small_stars = choice(
                    &value.extract::<String>()?,
                    "small_stars",
                    &[("drop", SmallStars::Drop), ("field", SmallStars::Field)],
                )?
            }
            "keep_galaxies" => options.scene.keep_galaxies = value.extract()?,
            "dust" => options.scene.dust = value.extract()?,
            "dust_opacity" => options.scene.dust_opacity = value.extract()?,
            "dust_map" => options.scene.dust_map = Some(dust_map_argument(&value)?),
            "planes" => options.scene.planes = planes_argument(&value)?,
            "auto_planes" => options.scene.auto_planes = value.extract()?,
            "distance_overrides" => {
                options.scene.distance_overrides = distance_overrides_argument(&value)?
            }
            "crop" => options.scene.crop = Some(value.extract()?),
            "framing" => options.video.framing = framing_argument(&value.extract::<String>()?)?,
            "start" => {
                options.video.start = choice(
                    &value.extract::<String>()?,
                    "start",
                    &[("focus", Start::Focus), ("whole", Start::Whole)],
                )?
            }
            "dolly" => options.video.dolly = value.extract()?,
            "truck" => options.video.truck = value.extract()?,
            "truck_angle_deg" => options.video.truck_angle_deg = value.extract()?,
            "pan" => options.video.pan = value.extract()?,
            "zoom" => options.video.zoom = value.extract()?,
            "zoom_end" => options.video.zoom_end = value.extract()?,
            "rotate_deg" => options.video.rotate_deg = pair(&value)?,
            "easing" => {
                options.video.easing = choice(
                    &value.extract::<String>()?,
                    "easing",
                    &[("in_out", Easing::InOut), ("linear", Easing::Linear)],
                )?
            }
            "quality" => {
                options.video.quality = choice(
                    &value.extract::<String>()?,
                    "quality",
                    &[("standard", Quality::Standard), ("high", Quality::High)],
                )?
            }
            "growth_limit" => options.video.growth_limit = value.extract()?,
            "auto_tour" => options.video.auto_tour = auto_tour_argument(&value)?,
            "tour_glide" => options.video.tour_glide = value.extract()?,
            "tour_titles" => options.video.tour_titles = value.extract()?,
            "tour_loop" => options.video.tour_loop = value.extract()?,
            "tour" => {
                options.video.tour = value
                    .try_iter()?
                    .map(|stop| {
                        let stop = stop?;
                        if let Ok(text) = stop.extract::<String>() {
                            return seiza_parallax::parse_stop(&text)
                                .map_err(PyValueError::new_err);
                        }
                        let stop = stop.cast::<PyDict>()?;
                        let mut parsed = seiza_parallax::TourStop::default();
                        for (key, value) in stop.iter() {
                            let key: String = key.extract()?;
                            match key.as_str() {
                                "focus" => {
                                    parsed.focus = if value.is_none() {
                                        None
                                    } else {
                                        Some(pair(&value)?)
                                    }
                                }
                                "dolly" => parsed.dolly = value.extract()?,
                                "zoom" => parsed.zoom = value.extract()?,
                                "rotate_deg" => parsed.rotate_deg = value.extract()?,
                                "pan" => parsed.pan = value.extract()?,
                                "travel" => parsed.travel = value.extract()?,
                                "hold" => parsed.hold = value.extract()?,
                                "spin_deg" => parsed.spin_deg = value.extract()?,
                                "push" => parsed.push = value.extract()?,
                                "title" => {
                                    parsed.title = value
                                        .extract::<Option<String>>()?
                                        .filter(|title| !title.is_empty())
                                }
                                "card" => parsed.card = value.extract()?,
                                "depth_pc" => parsed.depth_pc = value.extract()?,
                                // A plan's name for the stop's target.
                                "name" => {}
                                other => {
                                    return Err(PyValueError::new_err(format!(
                                        "unknown tour stop field {other:?}"
                                    )));
                                }
                            }
                        }
                        Ok(parsed)
                    })
                    .collect::<PyResult<_>>()?
            }
            "fade_from" => options.video.fade_from = value.extract()?,
            "size" => options.video.set_frame_size(frame_size_argument(&value)?),
            "seconds" => options.video.seconds = value.extract()?,
            "fps" => options.video.fps = value.extract()?,
            "overlay" => options.video.overlay = value.extract()?,
            "overlay_density" => options.video.overlay_density = value.extract()?,
            "labels" => {
                options.video.labels = value
                    .try_iter()?
                    .map(|label| {
                        let label = label?;
                        let label: Vec<Bound<'_, PyAny>> = if label.is_instance_of::<PyString>() {
                            Vec::new()
                        } else {
                            label.try_iter()?.collect::<PyResult<_>>()?
                        };
                        match label.as_slice() {
                            [x, y, text] => Ok(CustomLabel {
                                x: x.extract()?,
                                y: y.extract()?,
                                radius: 0.0,
                                text: text.extract()?,
                            }),
                            [x, y, radius, text] => Ok(CustomLabel {
                                x: x.extract()?,
                                y: y.extract()?,
                                radius: radius.extract()?,
                                text: text.extract()?,
                            }),
                            _ => Err(PyValueError::new_err(
                                "a label is (x, y, text) or (x, y, radius, text)",
                            )),
                        }
                    })
                    .collect::<PyResult<_>>()?
            }
            "label_color" => {
                options.video.label_color = seiza_parallax::parse_color(&value.extract::<String>()?)
                    .map_err(PyValueError::new_err)?
                    .0
            }
            "watermark" => match value.cast::<PyDict>() {
                Ok(style) => {
                    options.video.watermark = None;
                    for (key, value) in style.iter() {
                        match key.extract::<String>()?.as_str() {
                            "text" => options.video.watermark = watermark_text(&value)?,
                            "image" => options.video.watermark_image = value.extract()?,
                            "corner" => {
                                options.video.watermark_corner = choice(
                                    &value.extract::<String>()?,
                                    "watermark corner",
                                    &[
                                        ("bottom_right", Corner::BottomRight),
                                        ("bottom_left", Corner::BottomLeft),
                                        ("top_right", Corner::TopRight),
                                        ("top_left", Corner::TopLeft),
                                    ],
                                )?
                            }
                            "size" => options.video.watermark_size = value.extract()?,
                            "opacity" => options.video.watermark_opacity = value.extract()?,
                            other => {
                                return Err(PyValueError::new_err(format!(
                                    "unknown watermark field {other:?}"
                                )));
                            }
                        }
                    }
                }
                Err(_) => options.video.watermark = watermark_text(&value)?,
            },
            "intro_card" => options.video.intro_card = value.extract()?,
            "outro_card" => options.video.outro_card = value.extract()?,
            "card_seconds" => options.video.card_seconds = value.extract()?,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown parallax video option {other:?}"
                )));
            }
        }
    }
    Ok(options)
}

/// A watermark's text: True for "Rendered with seiza.fyi", False for none,
/// or the text.
fn watermark_text(value: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    Ok(match value.extract::<bool>() {
        Ok(true) => Some(seiza_parallax::DEFAULT_WATERMARK.into()),
        Ok(false) => None,
        Err(_) => Some(value.extract()?),
    })
}

/// An event as the dict `progress` callbacks receive.
fn event_dict<'py>(py: Python<'py>, event: Event) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    match event {
        Event::Note(message) => {
            dict.set_item("kind", "note")?;
            dict.set_item("message", message)?;
        }
        Event::Warning(message) => {
            dict.set_item("kind", "warning")?;
            dict.set_item("message", message)?;
        }
        Event::Frame { done, total } => {
            dict.set_item("kind", "frame")?;
            dict.set_item("done", done)?;
            dict.set_item("total", total)?;
        }
    }
    Ok(dict)
}

/// A reporter that hands events to `progress` and keeps its first error,
/// re-raised once the work settles.
fn reporter(progress: Option<Py<PyAny>>, raised: Arc<Mutex<Option<PyErr>>>) -> impl FnMut(Event) {
    move |event| {
        if let Some(callback) = &progress {
            Python::attach(|py| {
                if let Err(error) =
                    event_dict(py, event).and_then(|dict| callback.call1(py, (dict,)))
                    && let Ok(mut slot) = raised.lock()
                    && slot.is_none()
                {
                    *slot = Some(error);
                }
            });
        }
    }
}

fn reraise(raised: &Arc<Mutex<Option<PyErr>>>) -> PyResult<()> {
    match raised.lock().ok().and_then(|mut slot| slot.take()) {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Whether to stop: Ctrl-C, `cancel` returning true, or a callback that has
/// raised. A raising `cancel` stops too, its error kept for re-raising.
fn should_stop(cancel: &Option<Py<PyAny>>, raised: &Arc<Mutex<Option<PyErr>>>) -> bool {
    // A callback has raised already, most often `progress` taking Ctrl-C's
    // KeyboardInterrupt: stop, and let it be raised.
    if raised.lock().is_ok_and(|slot| slot.is_some()) {
        return true;
    }
    Python::attach(|py| {
        let asked = py.check_signals().and_then(|()| match cancel {
            Some(cancel) => cancel
                .call0(py)
                .and_then(|value| value.bind(py).is_truthy()),
            None => Ok(false),
        });
        match asked {
            Ok(stop) => stop,
            Err(error) => {
                if let Ok(mut slot) = raised.lock()
                    && slot.is_none()
                {
                    *slot = Some(error);
                }
                true
            }
        }
    })
}

/// Frame bytes in `format` as a (height, width, channels) array.
fn frame_array(py: Python<'_>, frame: image::RgbImage, format: &str) -> PyResult<Py<PyArray3<u8>>> {
    let (width, height) = (frame.width() as usize, frame.height() as usize);
    let (channels, data) = match format {
        "rgb" => (3, frame.into_raw()),
        "rgba" | "bgra" => {
            let bgra = format == "bgra";
            let data = frame
                .as_raw()
                .chunks_exact(3)
                .flat_map(|rgb| {
                    if bgra {
                        [rgb[2], rgb[1], rgb[0], 255]
                    } else {
                        [rgb[0], rgb[1], rgb[2], 255]
                    }
                })
                .collect();
            (4, data)
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "format must be \"rgb\", \"rgba\" or \"bgra\"; got {other:?}"
            )));
        }
    };
    let array = Array3::from_shape_vec((height, width, channels), data)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py).unbind())
}

#[pymethods]
impl PyParallaxVideo {
    #[new]
    #[pyo3(signature = (starless, stars, wcs, *, progress=None, cancel=None, **options))]
    fn new(
        py: Python<'_>,
        starless: &Bound<'_, PyAny>,
        stars: &Bound<'_, PyAny>,
        wcs: &Bound<'_, PyWcs>,
        progress: Option<Py<PyAny>>,
        cancel: Option<Py<PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let options = self::options(options)?;
        options.check().map_err(video_error)?;
        let starless = image_argument(starless, "starless")?;
        let stars = image_argument(stars, "stars")?;
        let wcs = wcs.get().wcs.clone();
        let raised = Arc::new(Mutex::new(None));
        let mut report = reporter(progress, Arc::clone(&raised));
        let stop_raised = Arc::clone(&raised);
        let prepared = py.detach(|| {
            let (starless, stars) = (starless.pixels()?, stars.pixels()?);
            Ok::<_, PyErr>(Parallax::prepare(
                &starless,
                &stars,
                &wcs,
                &options,
                &mut report,
                &|| should_stop(&cancel, &stop_raised),
            ))
        })?;
        reraise(&raised)?;
        let video = prepared.map_err(video_error)?;
        Ok(Self { video })
    }

    /// A new video of the same prepared scene, filmed with `options`: the
    /// camera, tour, output and labels, as the constructor takes them.
    /// Options left out take their defaults, not this video's. Nothing of
    /// the scene is prepared again, and options that would change it (the
    /// distances and their sources, `distance_focus`, star placement,
    /// galaxies or dust) are refused. This video is left as it was; both
    /// may draw frames at once, and the scene lives as long as either.
    /// `cancel()` returning True, or Ctrl-C, stops it.
    #[pyo3(signature = (*, progress=None, cancel=None, **options))]
    fn reconfigure(
        &self,
        py: Python<'_>,
        progress: Option<Py<PyAny>>,
        cancel: Option<Py<PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if let Some(options) = options {
            for (key, value) in options.iter() {
                let key: String = key.extract()?;
                if SCENE_KEYS.contains(&key.as_str()) && !value.is_none() {
                    return Err(PyValueError::new_err(format!(
                        "{key} changes the prepared scene; make a new ParallaxVideo for it"
                    )));
                }
            }
        }
        let video = self::options(options)?.video;
        let raised = Arc::new(Mutex::new(None));
        let mut report = reporter(progress, Arc::clone(&raised));
        let stop_raised = Arc::clone(&raised);
        let refilmed = py.detach(|| {
            self.video
                .reconfigure(&video, &mut report, &|| should_stop(&cancel, &stop_raised))
        });
        reraise(&raised)?;
        let video = refilmed.map_err(video_error)?;
        Ok(Self { video })
    }

    /// A tour of the catalogued objects in the image planned on this
    /// video's scene, as `auto_tour=` would plan it: each stop frames its
    /// target at the depth of the plane the scene shows it on.
    /// `plan_parallax_tour` plans from the catalogue alone. `auto_tour` is
    /// True, a count, or a dict of `targets`, `hold` and `motion`; `size` is
    /// the frame the tour is for and `framing` "fill" or "fit", this
    /// video's own by default. Returns
    /// `plan_parallax_tour`'s dict, its stops carrying `depth_pc`; its
    /// `tour`, edited, goes back in through `reconfigure(tour=...)`. The
    /// scene keeps the distance it was prepared with, so the plan's `focus`
    /// matters only to a new `ParallaxVideo`.
    #[pyo3(signature = (auto_tour=None, *, size=None, framing=None))]
    fn plan_tour<'py>(
        &self,
        py: Python<'py>,
        auto_tour: Option<&Bound<'py, PyAny>>,
        size: Option<&Bound<'py, PyAny>>,
        framing: Option<&str>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut auto = match auto_tour {
            None => seiza_parallax::AutoTour::default(),
            Some(value) => auto_tour_argument(value)?
                .ok_or_else(|| PyValueError::new_err("auto_tour=False plans no tour"))?,
        };
        auto.framing = match framing {
            Some(framing) => framing_argument(framing)?,
            None => self.video.shot().framing,
        };
        auto.check().map_err(PyValueError::new_err)?;
        let frame = match size {
            None => self.video.size(),
            Some(size) => frame_size_argument(size)?.for_image(self.video.image_size()),
        };
        let plan = py
            .detach(|| self.video.plan_tour(&auto, frame))
            .map_err(video_error)?;
        plan_dict(py, &plan)
    }

    /// The same scene and camera with `dust_map` (a path or a 2-D array,
    /// white where the dust is opaque), or with the dust counted from the
    /// stars again when None. Only the dust changes: nothing is prepared
    /// again, and this video is left as it was. A map gives dust even to a
    /// video prepared without it. `cancel()` returning True, or Ctrl-C,
    /// stops it.
    #[pyo3(signature = (dust_map, *, progress=None, cancel=None))]
    fn with_dust_map(
        &self,
        py: Python<'_>,
        dust_map: Option<&Bound<'_, PyAny>>,
        progress: Option<Py<PyAny>>,
        cancel: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let map = dust_map.map(dust_map_argument).transpose()?;
        let raised = Arc::new(Mutex::new(None));
        let mut report = reporter(progress, Arc::clone(&raised));
        let stop_raised = Arc::clone(&raised);
        let swapped = py.detach(|| {
            self.video.with_dust_map(map.as_ref(), &mut report, &|| {
                should_stop(&cancel, &stop_raised)
            })
        });
        reraise(&raised)?;
        let video = swapped.map_err(video_error)?;
        Ok(Self { video })
    }

    /// The dust as a (height, width) float32 array over the whole image, 0
    /// where the sky is clear to 1 where the dust is opaque, the values
    /// `with_dust_map` takes back: the dust in use, or with `counted=True`
    /// the dust counted from the stars even while a map takes its place.
    /// `size` (width, height) defaults to the image's. None when there is
    /// no such dust: it is off, or too few stars show to count it.
    #[pyo3(signature = (*, counted=false, size=None))]
    fn dust_map<'py>(
        &self,
        py: Python<'py>,
        counted: bool,
        size: Option<(usize, usize)>,
    ) -> PyResult<Option<Bound<'py, PyArray2<f32>>>> {
        let dust = if counted {
            self.video.counted_dust()
        } else {
            self.video.dust()
        };
        let Some(dust) = dust else {
            return Ok(None);
        };
        let (width, height) = size.unwrap_or_else(|| dust.size());
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("the map needs a width and height"));
        }
        let map = py.detach(|| dust.to_map(width, height));
        let array = Array2::from_shape_vec((height, width), map.opacity)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(Some(array.into_pyarray(py)))
    }

    /// Write the dust to `path`, white where it is opaque, in the format its
    /// extension names: a 16-bit PNG or TIFF, or a 32-bit float FITS or
    /// XISF, the size of the image: the dust in use, or with `counted=True`
    /// the dust counted from the stars. False when there is no such dust.
    #[pyo3(signature = (path, *, counted=false))]
    fn write_dust_map(&self, py: Python<'_>, path: PathBuf, counted: bool) -> PyResult<bool> {
        let dust = if counted {
            self.video.counted_dust()
        } else {
            self.video.dust()
        };
        let Some(dust) = dust else {
            return Ok(false);
        };
        let (width, height) = dust.size();
        py.detach(|| dust.to_map(width, height).save(&path))
            .map_err(PyRuntimeError::new_err)?;
        Ok(true)
    }

    /// The number of frames.
    #[getter]
    fn frame_count(&self) -> usize {
        self.video.frames()
    }

    #[getter]
    fn fps(&self) -> u32 {
        self.video.fps()
    }

    /// (width, height) of a frame.
    #[getter]
    fn size(&self) -> (usize, usize) {
        self.video.size()
    }

    /// What preparing the video found, as a dict. Its `wcs` is the plate
    /// solution of the image as given, before any crop, and `crop` the
    /// part filmed, (x, y, width, height), or None.
    #[getter]
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let summary = self.video.summary();
        let dict = PyDict::new(py);
        dict.set_item("detected_stars", summary.detected_stars)?;
        dict.set_item("gaia_stars", summary.gaia_stars)?;
        dict.set_item("hipparcos_stars", summary.hipparcos_stars)?;
        dict.set_item("gaia_matches", summary.gaia_matches)?;
        dict.set_item("with_distance", summary.with_distance)?;
        dict.set_item("background_distance_pc", summary.background_distance_pc)?;
        dict.set_item("background_basis", &summary.background_basis)?;
        dict.set_item("background_focus", summary.background_focus)?;
        dict.set_item("unmatched_distance_pc", summary.unmatched_distance_pc)?;
        dict.set_item("flying_stars", summary.flying_stars)?;
        dict.set_item("galaxies_lifted", summary.galaxies_lifted.clone())?;
        dict.set_item("dust_transmission", summary.dust_transmission)?;
        dict.set_item("dust_given", summary.dust_given)?;
        dict.set_item("stars_in_front_of_dust", summary.stars_in_front_of_dust)?;
        let planes = summary
            .planes
            .iter()
            .map(|plane| {
                let entry = PyDict::new(py);
                entry.set_item("name", &plane.name)?;
                entry.set_item("distance_pc", plane.distance_pc)?;
                entry.set_item("basis", &plane.basis)?;
                entry.set_item("coverage", plane.coverage)?;
                entry.set_item("absorbs", plane.absorbs)?;
                entry.set_item("object", &plane.object)?;
                Ok(entry)
            })
            .collect::<PyResult<Vec<_>>>()?;
        dict.set_item("planes", planes)?;
        dict.set_item("labelled_objects", summary.labelled_objects)?;
        dict.set_item(
            "wcs",
            Py::new(
                py,
                PyWcs {
                    wcs: self.video.source_wcs().clone(),
                },
            )?,
        )?;
        dict.set_item("crop", self.video.crop())?;
        let fit = PyDict::new(py);
        fit.set_item("zoom", summary.fit.zoom)?;
        fit.set_item("pan", summary.fit.pan)?;
        fit.set_item("lead", summary.fit.lead)?;
        fit.set_item("truck", summary.fit.truck)?;
        let stops = summary
            .fit
            .stops
            .iter()
            .map(|stop| {
                let entry = PyDict::new(py);
                entry.set_item("zoom", stop.zoom)?;
                entry.set_item("pan", stop.pan)?;
                entry.set_item("dolly", stop.dolly)?;
                Ok(entry)
            })
            .collect::<PyResult<Vec<_>>>()?;
        fit.set_item("stops", stops)?;
        fit.set_item("inside", summary.fit.inside)?;
        dict.set_item("fit", fit)?;
        Ok(dict)
    }

    fn __len__(&self) -> usize {
        self.video.frames()
    }

    /// Every frame in order, as RGB arrays.
    fn __iter__(slf: Py<Self>) -> PyParallaxFrames {
        PyParallaxFrames {
            video: slf,
            next: 0,
        }
    }

    /// Frame `index` as a (height, width, channels) uint8 array, in
    /// `format` "rgb", "rgba" or "bgra".
    #[pyo3(signature = (index, format="rgb"))]
    fn frame(&self, py: Python<'_>, index: usize, format: &str) -> PyResult<Py<PyArray3<u8>>> {
        if index >= self.video.frames() {
            return Err(PyValueError::new_err(format!(
                "frame {index} is past the video's {} frames",
                self.video.frames()
            )));
        }
        let frame = py.detach(|| self.video.frame(index));
        frame_array(py, frame, format)
    }

    /// Hand every frame in order to `on_frame(array, index)`, the hook for
    /// any encoder. Returns True when every frame was delivered and False
    /// when `on_frame` returned False, `cancel()` returned True, or Ctrl-C
    /// stopped it. `progress` hears each frame drawn.
    #[pyo3(signature = (on_frame, *, format="rgb", progress=None, cancel=None))]
    fn render(
        &self,
        py: Python<'_>,
        on_frame: Py<PyAny>,
        format: &str,
        progress: Option<Py<PyAny>>,
        cancel: Option<Py<PyAny>>,
    ) -> PyResult<bool> {
        let total = self.video.frames();
        let raised = Arc::new(Mutex::new(None));
        let mut report = reporter(progress, Arc::clone(&raised));
        for index in 0..total {
            if should_stop(&cancel, &raised) {
                reraise(&raised)?;
                return Ok(false);
            }
            let frame = py.detach(|| self.video.frame(index));
            let array = frame_array(py, frame, format)?;
            let answer = on_frame.call1(py, (array, index))?;
            report(Event::Frame {
                done: index + 1,
                total,
            });
            reraise(&raised)?;
            if answer.bind(py).is(&*pyo3::types::PyBool::new(py, false)) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Write the video to `output`: an MP4 through ffmpeg ("auto",
    /// "ffmpeg") in `codec` "h264" or "hevc", or numbered PNG frames in the
    /// `output` directory ("png").
    /// Returns True when written and False when `cancel()` returned True or
    /// Ctrl-C stopped it.
    #[pyo3(signature = (output, *, encoder="auto", codec="h264", ffmpeg=None, progress=None, cancel=None))]
    #[allow(clippy::too_many_arguments)]
    fn write(
        &self,
        py: Python<'_>,
        output: PathBuf,
        encoder: &str,
        codec: &str,
        ffmpeg: Option<PathBuf>,
        progress: Option<Py<PyAny>>,
        cancel: Option<Py<PyAny>>,
    ) -> PyResult<bool> {
        let settings = self.video.video_settings();
        let ffmpeg = ffmpeg.unwrap_or_else(|| PathBuf::from("ffmpeg"));
        let sink: Box<dyn FrameSink + Send> = match encoder {
            "auto" | "ffmpeg" => Box::new(
                FfmpegSink::start(
                    &ffmpeg,
                    &output,
                    settings,
                    seiza_parallax::Codec::parse(codec).map_err(PyValueError::new_err)?,
                )
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
            ),
            "png" => Box::new(
                PngSequence::new(&output, settings)
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
            ),
            other => {
                return Err(PyValueError::new_err(format!(
                    "encoder must be \"auto\", \"ffmpeg\" or \"png\"; got {other:?}"
                )));
            }
        };
        let raised = Arc::new(Mutex::new(None));
        let mut report = reporter(progress, Arc::clone(&raised));
        let stop_raised = Arc::clone(&raised);
        let written = py.detach(|| {
            self.video
                .render(sink, &mut report, &|| should_stop(&cancel, &stop_raised))
        });
        reraise(&raised)?;
        match written {
            Ok(()) => Ok(true),
            Err(seiza_parallax::pipeline::Error::Stopped) => Ok(false),
            Err(error) => Err(PyRuntimeError::new_err(error.to_string())),
        }
    }
}
