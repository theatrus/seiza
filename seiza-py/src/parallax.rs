//! Parallax fly-through videos (`seiza-parallax`), as `seiza parallax-video`
//! makes them: [`PyParallaxVideo`] prepares a video from a starless image,
//! its stars and a plate solution, then draws frames as numpy arrays one at
//! a time, in turn to a callback (any encoder: PyAV, OpenCV, a platform
//! writer), or into a video file.

use crate::PyWcs;
use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use seiza_parallax::{
    CustomLabel, Easing, Event, FfmpegSink, FrameSink, Parallax, ParallaxOptions, PngSequence,
    Quality, SmallStars, Start,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyParallaxVideo>()?;
    module.add_class::<PyParallaxFrames>()?;
    Ok(())
}

/// A prepared parallax video: the stars of a starless image and its stars
/// placed at their Gaia distances, galaxies lifted onto the far field, the
/// dust mapped, and a camera move fitted to the image.
///
/// `starless` and `stars` are the stretched image split into its starless
/// image and unscreened stars, each a path to a PNG, JPEG or TIFF or a
/// numpy array of shape (height, width, 3), float32 from 0 to 1 or uint8;
/// `wcs` is the image's plate solution (`seiza.solve_blind(...).wcs`).
/// The other keyword arguments are `seiza parallax-video`'s options:
/// `focus` (x, y), `distance_pc`, `unmatched_distance_pc`, `objects`,
/// `object_distances`, `star_distances`, `gaia_max_mag`, `gaia_cache`,
/// `online`, `max_stars`, `small_stars` ("drop", "field"), `keep_galaxies`,
/// `dust`, `dust_opacity`, `start` ("focus", "whole"), `dolly`, `truck`,
/// `truck_angle_deg`, `pan`, `zoom`, `zoom_end`, `rotate_deg` (first,
/// last), `easing` ("in_out", "linear"), `quality` ("standard", "high"),
/// `growth_limit`, `fade_from`, `size` ("720p", "1080p", "1440p", "4k",
/// each with "-portrait", "WIDTHxHEIGHT", or (width, height)), `seconds`,
/// `fps`, `overlay`, `overlay_density`, `labels` ((x, y, text) or (x, y,
/// radius, text) tuples), `label_color` ("#RRGGBB") and `watermark` (True,
/// or the text). `progress` hears each step as a dict with a "kind" of
/// "note" or "warning" and a "message".
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

/// An image argument: a path, or a numpy array of float32 0 to 1 or uint8.
fn image_argument(value: &Bound<'_, PyAny>, name: &str) -> PyResult<image::Rgb32FImage> {
    if let Ok(path) = value.extract::<PathBuf>() {
        return Ok(seiza::raster::open_oriented(&path)
            .map_err(|error| {
                PyValueError::new_err(format!("failed to open {}: {error}", path.display()))
            })?
            .pixels
            .to_rgb32f());
    }
    let shape_error =
        || PyValueError::new_err(format!("{name} must have shape (height, width, 3)"));
    let raw = |shape: &[usize]| match shape {
        [height, width, 3] => Ok((*width as u32, *height as u32)),
        _ => Err(shape_error()),
    };
    if let Ok(array) = value.extract::<PyReadonlyArrayDyn<'_, f32>>() {
        let (width, height) = raw(array.shape())?;
        let data = array
            .as_slice()
            .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))?
            .to_vec();
        return image::Rgb32FImage::from_raw(width, height, data).ok_or_else(shape_error);
    }
    if let Ok(array) = value.extract::<PyReadonlyArrayDyn<'_, u8>>() {
        let (width, height) = raw(array.shape())?;
        let data = array
            .as_slice()
            .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))?
            .iter()
            .map(|&value| value as f32 / 255.0)
            .collect();
        return image::Rgb32FImage::from_raw(width, height, data).ok_or_else(shape_error);
    }
    Err(PyValueError::new_err(format!(
        "{name} must be a path or a numpy array of float32 or uint8"
    )))
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
            "focus" => options.focus = Some(value.extract()?),
            "distance_pc" => options.distance_pc = Some(value.extract()?),
            "unmatched_distance_pc" => options.unmatched_distance_pc = Some(value.extract()?),
            "objects" => options.objects = Some(value.extract()?),
            "object_distances" => options.object_distances = Some(value.extract()?),
            "star_distances" => options.star_distances = Some(value.extract()?),
            "gaia_max_mag" => options.gaia_max_mag = value.extract()?,
            "gaia_cache" => options.gaia_cache = Some(value.extract()?),
            "online" => options.online = value.extract()?,
            "max_stars" => options.max_stars = Some(value.extract()?),
            "small_stars" => {
                options.small_stars = choice(
                    &value.extract::<String>()?,
                    "small_stars",
                    &[("drop", SmallStars::Drop), ("field", SmallStars::Field)],
                )?
            }
            "keep_galaxies" => options.keep_galaxies = value.extract()?,
            "dust" => options.dust = value.extract()?,
            "dust_opacity" => options.dust_opacity = value.extract()?,
            "start" => {
                options.start = choice(
                    &value.extract::<String>()?,
                    "start",
                    &[("focus", Start::Focus), ("whole", Start::Whole)],
                )?
            }
            "dolly" => options.dolly = value.extract()?,
            "truck" => options.truck = value.extract()?,
            "truck_angle_deg" => options.truck_angle_deg = value.extract()?,
            "pan" => options.pan = value.extract()?,
            "zoom" => options.zoom = value.extract()?,
            "zoom_end" => options.zoom_end = value.extract()?,
            "rotate_deg" => options.rotate_deg = value.extract()?,
            "easing" => {
                options.easing = choice(
                    &value.extract::<String>()?,
                    "easing",
                    &[("in_out", Easing::InOut), ("linear", Easing::Linear)],
                )?
            }
            "quality" => {
                options.quality = choice(
                    &value.extract::<String>()?,
                    "quality",
                    &[("standard", Quality::Standard), ("high", Quality::High)],
                )?
            }
            "growth_limit" => options.growth_limit = value.extract()?,
            "fade_from" => options.fade_from = value.extract()?,
            "size" => {
                options.size = match value.extract::<String>() {
                    Ok(size) => {
                        seiza_parallax::parse_frame_size(&size).map_err(PyValueError::new_err)?
                    }
                    Err(_) => value.extract()?,
                }
            }
            "seconds" => options.seconds = value.extract()?,
            "fps" => options.fps = value.extract()?,
            "overlay" => options.overlay = value.extract()?,
            "overlay_density" => options.overlay_density = value.extract()?,
            "labels" => {
                options.labels = value
                    .try_iter()?
                    .map(|label| {
                        let label = label?;
                        let label = label.downcast::<PyTuple>()?;
                        match label.len() {
                            3 => Ok(CustomLabel {
                                x: label.get_item(0)?.extract()?,
                                y: label.get_item(1)?.extract()?,
                                radius: 0.0,
                                text: label.get_item(2)?.extract()?,
                            }),
                            4 => Ok(CustomLabel {
                                x: label.get_item(0)?.extract()?,
                                y: label.get_item(1)?.extract()?,
                                radius: label.get_item(2)?.extract()?,
                                text: label.get_item(3)?.extract()?,
                            }),
                            _ => Err(PyValueError::new_err(
                                "a label is (x, y, text) or (x, y, radius, text)",
                            )),
                        }
                    })
                    .collect::<PyResult<_>>()?
            }
            "label_color" => {
                options.label_color = seiza_parallax::parse_color(&value.extract::<String>()?)
                    .map_err(PyValueError::new_err)?
                    .0
            }
            "watermark" => {
                options.watermark = match value.extract::<bool>() {
                    Ok(true) => Some(seiza_parallax::DEFAULT_WATERMARK.into()),
                    Ok(false) => None,
                    Err(_) => Some(value.extract()?),
                }
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown parallax video option {other:?}"
                )));
            }
        }
    }
    Ok(options)
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
fn reporter(progress: Option<PyObject>, raised: Arc<Mutex<Option<PyErr>>>) -> impl FnMut(Event) {
    move |event| {
        if let Some(callback) = &progress {
            Python::with_gil(|py| {
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

/// Whether to stop: Ctrl-C, or `cancel` returning true. A raising `cancel`
/// stops too, its error kept for re-raising.
fn should_stop(cancel: &Option<PyObject>, raised: &Arc<Mutex<Option<PyErr>>>) -> bool {
    Python::with_gil(|py| {
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
    #[pyo3(signature = (starless, stars, wcs, *, progress=None, **options))]
    fn new(
        py: Python<'_>,
        starless: &Bound<'_, PyAny>,
        stars: &Bound<'_, PyAny>,
        wcs: &Bound<'_, PyWcs>,
        progress: Option<PyObject>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let options = self::options(options)?;
        options
            .check()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let starless = image_argument(starless, "starless")?;
        let stars = image_argument(stars, "stars")?;
        let wcs = wcs.get().wcs.clone();
        let raised = Arc::new(Mutex::new(None));
        let mut report = reporter(progress, Arc::clone(&raised));
        let prepared =
            py.allow_threads(|| Parallax::prepare(&starless, &stars, &wcs, &options, &mut report));
        reraise(&raised)?;
        let video = prepared.map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self { video })
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

    /// What preparing the video found, as a dict.
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
        dict.set_item("unmatched_distance_pc", summary.unmatched_distance_pc)?;
        dict.set_item("flying_stars", summary.flying_stars)?;
        dict.set_item("galaxies_lifted", summary.galaxies_lifted.clone())?;
        dict.set_item("dust_transmission", summary.dust_transmission)?;
        dict.set_item("labelled_objects", summary.labelled_objects)?;
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
        let frame = py.allow_threads(|| self.video.frame(index));
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
        on_frame: PyObject,
        format: &str,
        progress: Option<PyObject>,
        cancel: Option<PyObject>,
    ) -> PyResult<bool> {
        let total = self.video.frames();
        let raised = Arc::new(Mutex::new(None));
        let mut report = reporter(progress, Arc::clone(&raised));
        for index in 0..total {
            if should_stop(&cancel, &raised) {
                reraise(&raised)?;
                return Ok(false);
            }
            let frame = py.allow_threads(|| self.video.frame(index));
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
    /// "ffmpeg"), or numbered PNG frames in the `output` directory ("png").
    /// Returns True when written and False when `cancel()` returned True or
    /// Ctrl-C stopped it.
    #[pyo3(signature = (output, *, encoder="auto", ffmpeg=None, progress=None, cancel=None))]
    fn write(
        &self,
        py: Python<'_>,
        output: PathBuf,
        encoder: &str,
        ffmpeg: Option<PathBuf>,
        progress: Option<PyObject>,
        cancel: Option<PyObject>,
    ) -> PyResult<bool> {
        let settings = self.video.video_settings();
        let ffmpeg = ffmpeg.unwrap_or_else(|| PathBuf::from("ffmpeg"));
        let sink: Box<dyn FrameSink + Send> = match encoder {
            "auto" | "ffmpeg" => Box::new(
                FfmpegSink::start(&ffmpeg, &output, settings)
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
        let written = py.allow_threads(|| {
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
