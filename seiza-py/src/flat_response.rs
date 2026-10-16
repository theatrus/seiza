//! Flat sessions and the dust they hold (`seiza_calibration::flat_response`
//! and `seiza_calibration::dust`), as the C ABI's `seiza_flat_*` and
//! `seiza_dust_*` calls offer them.
//!
//! Planes are 2-D float32 arrays. Results are dicts parsed from the core's
//! own JSON, the form the C calls return, so both bindings give the same
//! keys. Options are a dict of `DustOptions` fields, any left out taking
//! their defaults.

use crate::arrays::{float_array, float_image_view};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use seiza_calibration::dust::{self, DustFeature, DustMap, DustOptions};
use seiza_calibration::flat_response::{Plane, Side, Verdict, choice, evidence, metric};

/// A 2-D float32 array as a plane.
fn as_plane(array: &PyReadonlyArrayDyn<'_, f32>, name: &str) -> PyResult<Plane> {
    let view = float_image_view(array)?;
    if view.channels != 1 {
        return Err(PyValueError::new_err(format!(
            "{name} must be a 2-D array (height, width)"
        )));
    }
    if view.width == 0 || view.height == 0 {
        return Err(PyValueError::new_err(format!("{name} is empty")));
    }
    Ok(Plane {
        width: view.width,
        height: view.height,
        data: view.data.to_vec(),
    })
}

/// Several 2-D arrays of one size as planes.
fn as_planes(arrays: Vec<PyReadonlyArrayDyn<'_, f32>>, name: &str) -> PyResult<Vec<Plane>> {
    let planes = arrays
        .iter()
        .map(|array| as_plane(array, name))
        .collect::<PyResult<Vec<_>>>()?;
    if let Some(first) = planes.first()
        && planes
            .iter()
            .any(|plane| (plane.width, plane.height) != (first.width, first.height))
    {
        return Err(PyValueError::new_err(format!(
            "{name} must all be one size"
        )));
    }
    Ok(planes)
}

fn plane_array(py: Python<'_>, plane: Plane) -> PyResult<Bound<'_, PyArrayDyn<f32>>> {
    float_array(py, plane.width, plane.height, 1, plane.data)
}

/// The core's JSON for `value` as Python objects.
pub(crate) fn to_python<'py>(
    py: Python<'py>,
    value: &impl serde::Serialize,
) -> PyResult<Bound<'py, PyAny>> {
    let json =
        serde_json::to_string(value).map_err(|error| PyValueError::new_err(error.to_string()))?;
    py.import("json")?.call_method1("loads", (json,))
}

/// A Python value (a dict, a list, a string) as the core type `T`.
fn from_python<T: serde::de::DeserializeOwned>(
    value: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<T> {
    let json: String = value
        .py()
        .import("json")?
        .call_method1("dumps", (value,))?
        .extract()?;
    serde_json::from_str(&json)
        .map_err(|error| PyValueError::new_err(format!("invalid {name}: {error}")))
}

fn dust_options(options: Option<&Bound<'_, PyAny>>) -> PyResult<DustOptions> {
    match options {
        None => Ok(DustOptions::default()),
        Some(options) if options.is_none() => Ok(DustOptions::default()),
        Some(options) => from_python(options, "dust options"),
    }
}

/// The block factor that brings a `width` × `height` frame's long side
/// within the response size; with `cfa=True` the blocks cover whole 2×2
/// colour cells.
#[pyfunction]
#[pyo3(signature = (width, height, cfa=false))]
fn flat_bin_factor(width: usize, height: usize, cfa: bool) -> usize {
    metric::bin_factor(width, height, cfa)
}

/// Bin a frame for a response: block means of `factor` × `factor` pixels,
/// its channels averaged. `image` is (height, width) or (height, width, 3)
/// float32; the result is (height // factor, width // factor).
#[pyfunction]
fn flat_bin_frame<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, f32>,
    factor: usize,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    if factor == 0 {
        return Err(PyValueError::new_err("factor must be at least 1"));
    }
    let view = float_image_view(&image)?;
    let binned = metric::bin_frame(view.data, view.width, view.height, view.channels, factor);
    plane_array(py, binned)
}

/// Combine binned flats of one session (raw ADU with their pedestals) into
/// one response. Returns `{"plane", "bin", "level", "noise", "frames"}`,
/// `plane` a 2-D array, or None when the flats are too few or unusable.
#[pyfunction]
fn flat_build_response<'py>(
    py: Python<'py>,
    planes: Vec<PyReadonlyArrayDyn<'py, f32>>,
    bin: usize,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let binned = as_planes(planes, "planes")?;
    let Some(response) = py.detach(|| metric::build_response(&binned, bin)) else {
        return Ok(None);
    };
    let result = PyDict::new(py);
    result.set_item("bin", response.bin)?;
    result.set_item("level", response.level)?;
    result.set_item("noise", response.noise)?;
    result.set_item("frames", response.frames)?;
    result.set_item("plane", plane_array(py, response.plane)?)?;
    Ok(Some(result))
}

/// Compare two responses of one light path (pedestals removed), binned by
/// `bin`: the broad change, the dust, and the spots that came, went or
/// moved. Returns the comparison as a dict, or None when the responses hold
/// too little to compare.
#[pyfunction]
#[pyo3(signature = (earlier, later, bin, options=None))]
fn flat_compare<'py>(
    py: Python<'py>,
    earlier: PyReadonlyArrayDyn<'py, f32>,
    later: PyReadonlyArrayDyn<'py, f32>,
    bin: usize,
    options: Option<&Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let earlier = as_plane(&earlier, "earlier")?;
    let later = as_plane(&later, "later")?;
    let options = dust_options(options)?;
    py.detach(|| metric::compare_with(&earlier, &later, bin, &options))
        .map(|comparison| to_python(py, &comparison))
        .transpose()
}

/// The compact change between a session before an event and one after it
/// (`after / before`), as a 2-D array, or None when they hold no change map.
#[pyfunction]
fn flat_change_map<'py>(
    py: Python<'py>,
    before: PyReadonlyArrayDyn<'py, f32>,
    after: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
    let before = as_plane(&before, "before")?;
    let after = as_plane(&after, "after")?;
    py.detach(|| metric::change_map(&before, &after))
        .map(|map| plane_array(py, map))
        .transpose()
}

/// Carry a flat across a change in place: divide `image` ((height, width) or
/// (height, width, 3) float32, sensor-aligned, C-contiguous) by the change
/// `map` (binned by `bin`) for a light shot before the change
/// (`divide=True`), or multiply for one shot after it.
#[pyfunction]
fn flat_apply_change_map(
    mut image: PyReadwriteArrayDyn<'_, f32>,
    map: PyReadonlyArrayDyn<'_, f32>,
    bin: usize,
    divide: bool,
) -> PyResult<()> {
    if bin == 0 {
        return Err(PyValueError::new_err("bin must be at least 1"));
    }
    let map = as_plane(&map, "map")?;
    let (width, channels) = match image.shape() {
        [_, width] => (*width, 1),
        [_, width, 3] => (*width, 3),
        _ => {
            return Err(PyValueError::new_err(
                "image must have shape (height, width) or (height, width, 3)",
            ));
        }
    };
    if !image.is_c_contiguous() {
        return Err(PyValueError::new_err("image must be C-contiguous"));
    }
    let data = image
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("image must be C-contiguous and writable"))?;
    metric::apply_change_map(data, width, channels, &map, bin, divide);
    Ok(())
}

/// The sky background of a few binned lights, each less `pedestal` and
/// divided by its own median, combined: a 2-D array, or None when the
/// lights hold too little.
#[pyfunction]
fn flat_lights_background<'py>(
    py: Python<'py>,
    planes: Vec<PyReadonlyArrayDyn<'py, f32>>,
    pedestal: f32,
    bin: usize,
) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
    let binned = as_planes(planes, "planes")?;
    py.detach(|| evidence::background(&binned, pedestal, bin))
        .map(|background| plane_array(py, background))
        .transpose()
}

/// Which side of a change a few lights were shot on: their `background`
/// weighed against the `reference` response, which lies on side
/// `reference_side` ("before" or "after") of the change `map`. Returns a
/// dict: `{"outcome": "fit", "carried", "spread", "verdict"}`,
/// `{"outcome": "no_shadows"}` or `{"outcome": "unfitted"}`.
#[pyfunction]
fn flat_fit_change<'py>(
    py: Python<'py>,
    background: PyReadonlyArrayDyn<'py, f32>,
    reference: PyReadonlyArrayDyn<'py, f32>,
    map: PyReadonlyArrayDyn<'py, f32>,
    reference_side: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let background = as_plane(&background, "background")?;
    let reference = as_plane(&reference, "reference")?;
    let map = as_plane(&map, "map")?;
    let side: Side = serde_json::from_value(serde_json::Value::String(reference_side.into()))
        .map_err(|_| {
            PyValueError::new_err(format!(
                "reference_side must be \"before\" or \"after\", not {reference_side:?}"
            ))
        })?;
    let outcome = py.detach(|| evidence::fit_outcome(&background, &reference, side, &map));
    to_python(py, &outcome)
}

/// One verdict for an imaging session from its filters' verdicts
/// ("before", "after" or "undecided"): a side some fit decides and none
/// contradicts.
#[pyfunction]
fn flat_combine_verdicts(verdicts: &Bound<'_, PyAny>) -> PyResult<String> {
    let verdicts: Vec<Verdict> = from_python(verdicts, "verdicts")?;
    let combined = evidence::combine(&verdicts);
    serde_json::to_value(combined)
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned))
        .ok_or_else(|| PyValueError::new_err("the verdict has no text form"))
}

/// Which of the candidate responses (such as one rig's flats at several
/// rotator angles) the lights' `background` follows. `alike` lists
/// `(first, second)` index pairs already known to be alike, which are not
/// weighed. Returns `{"winners": [...], "pairs": [...]}`.
#[pyfunction]
#[pyo3(signature = (background, candidates, alike=None))]
fn flat_choose_response<'py>(
    py: Python<'py>,
    background: PyReadonlyArrayDyn<'py, f32>,
    candidates: Vec<PyReadonlyArrayDyn<'py, f32>>,
    alike: Option<Vec<(usize, usize)>>,
) -> PyResult<Bound<'py, PyAny>> {
    let background = as_plane(&background, "background")?;
    let candidates = as_planes(candidates, "candidates")?;
    let alike = alike.unwrap_or_default();
    let choice = py.detach(|| {
        choice::choose_response(&background, &candidates, |first, second| {
            alike
                .iter()
                .any(|&(a, b)| (a, b) == (first, second) || (a, b) == (second, first))
        })
    });
    to_python(py, &choice)
}

/// Encode a plane for a cache: an 8-byte size, then 16-bit fixed point
/// covering 0 to 2.
#[pyfunction]
fn flat_encode_plane<'py>(
    py: Python<'py>,
    plane: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyBytes>> {
    let plane = as_plane(&plane, "plane")?;
    Ok(PyBytes::new(py, &metric::encode_plane(&plane)))
}

/// Read a plane `flat_encode_plane` wrote, as a 2-D array; None when the
/// bytes are not one.
#[pyfunction]
fn flat_decode_plane<'py>(
    py: Python<'py>,
    data: &[u8],
) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
    metric::decode_plane(data)
        .map(|plane| plane_array(py, plane))
        .transpose()
}

/// Find the dust in one plane, binned by `bin`: a flat session's response
/// or one binned flat. Returns the dust map as a dict, or None when the
/// plane holds too little.
#[pyfunction]
#[pyo3(signature = (plane, bin, options=None))]
fn dust_find<'py>(
    py: Python<'py>,
    plane: PyReadonlyArrayDyn<'py, f32>,
    bin: usize,
    options: Option<&Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let plane = as_plane(&plane, "plane")?;
    let options = dust_options(options)?;
    py.detach(|| dust::find_dust(&plane, bin, &options))
        .map(|map| to_python(py, &map))
        .transpose()
}

/// Find a whole flat session's dust: binned flats (raw ADU, binned by
/// `bin`, with pedestal `pedestal`) stacked, so faint disks and donuts clear
/// the noise, and each feature counted across the flats that show it.
/// Returns the session's dust map as a dict, or None with fewer than two
/// usable flats.
#[pyfunction]
#[pyo3(signature = (planes, bin, pedestal, options=None))]
fn dust_find_session<'py>(
    py: Python<'py>,
    planes: Vec<PyReadonlyArrayDyn<'py, f32>>,
    bin: usize,
    pedestal: f32,
    options: Option<&Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let binned = as_planes(planes, "planes")?;
    let options = dust_options(options)?;
    py.detach(|| dust::find_session_dust(&binned, bin, pedestal, &options))
        .map(|map| to_python(py, &map))
        .transpose()
}

/// One flat of a session against the session's dust: which of the
/// session's features it shows, and what it holds that the session does
/// not. `frame` is binned like the session's flats (raw ADU); `session` is
/// the dict `dust_find_session` returned. Returns `{"seen", "transients"}`,
/// or None when the frame holds too little.
#[pyfunction]
#[pyo3(signature = (frame, pedestal, session, options=None))]
fn dust_frame<'py>(
    py: Python<'py>,
    frame: PyReadonlyArrayDyn<'py, f32>,
    pedestal: f32,
    session: &Bound<'py, PyAny>,
    options: Option<&Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let frame = as_plane(&frame, "frame")?;
    let session: DustMap = from_python(session, "session dust map")?;
    let options = dust_options(options)?;
    py.detach(|| dust::frame_dust(&frame, pedestal, &session, &options))
        .map(|found| to_python(py, &found))
        .transpose()
}

/// How the dust of two sessions of one light path differs: `earlier` and
/// `later` are responses on one grid (binned by `bin`, pedestals removed).
/// Returns the changes as a dict (what came, went or moved, with both
/// sessions' dust maps), or None when the planes hold too little.
#[pyfunction]
#[pyo3(signature = (earlier, later, bin, options=None))]
fn dust_find_changes<'py>(
    py: Python<'py>,
    earlier: PyReadonlyArrayDyn<'py, f32>,
    later: PyReadonlyArrayDyn<'py, f32>,
    bin: usize,
    options: Option<&Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let earlier = as_plane(&earlier, "earlier")?;
    let later = as_plane(&later, "later")?;
    let options = dust_options(options)?;
    py.detach(|| dust::find_changes(&earlier, &later, bin, &options))
        .map(|changes| to_python(py, &changes))
        .transpose()
}

/// Match two lists of dust features, as two dust maps' `features` hold
/// them. Returns a list of `{"earlier", "later", "offset",
/// "within_tolerance"}`.
#[pyfunction]
#[pyo3(signature = (earlier, later, options=None))]
fn dust_match_features<'py>(
    py: Python<'py>,
    earlier: &Bound<'py, PyAny>,
    later: &Bound<'py, PyAny>,
    options: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let earlier: Vec<DustFeature> = from_python(earlier, "earlier features")?;
    let later: Vec<DustFeature> = from_python(later, "later features")?;
    let options = dust_options(options)?;
    to_python(py, &dust::match_features(&earlier, &later, &options))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(flat_bin_factor, module)?)?;
    module.add_function(wrap_pyfunction!(flat_bin_frame, module)?)?;
    module.add_function(wrap_pyfunction!(flat_build_response, module)?)?;
    module.add_function(wrap_pyfunction!(flat_compare, module)?)?;
    module.add_function(wrap_pyfunction!(flat_change_map, module)?)?;
    module.add_function(wrap_pyfunction!(flat_apply_change_map, module)?)?;
    module.add_function(wrap_pyfunction!(flat_lights_background, module)?)?;
    module.add_function(wrap_pyfunction!(flat_fit_change, module)?)?;
    module.add_function(wrap_pyfunction!(flat_combine_verdicts, module)?)?;
    module.add_function(wrap_pyfunction!(flat_choose_response, module)?)?;
    module.add_function(wrap_pyfunction!(flat_encode_plane, module)?)?;
    module.add_function(wrap_pyfunction!(flat_decode_plane, module)?)?;
    module.add_function(wrap_pyfunction!(dust_find, module)?)?;
    module.add_function(wrap_pyfunction!(dust_find_session, module)?)?;
    module.add_function(wrap_pyfunction!(dust_frame, module)?)?;
    module.add_function(wrap_pyfunction!(dust_find_changes, module)?)?;
    module.add_function(wrap_pyfunction!(dust_match_features, module)?)?;
    Ok(())
}
