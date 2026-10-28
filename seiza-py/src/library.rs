//! Python calls the same immutable PSF Guard library policy as the C ABI.
use crate::flat_response::{from_python, to_python};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use seiza_calibration::library;
use std::sync::Arc;

/// Reusable measured grouping book. Construction and planning perform no I/O.
#[pyclass(module = "seiza", frozen)]
struct FlatGroups {
    book: Arc<seiza_calibration::flat_response::groups::Book>,
}
#[pymethods]
impl FlatGroups {
    #[new]
    fn new(request: &Bound<'_, PyAny>) -> PyResult<Self> {
        let request = from_python(request, "flat grouping request")?;
        let book = seiza_calibration::flat_response::groups::prepare(request)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            book: Arc::new(book),
        })
    }
}
fn records<T: serde::de::DeserializeOwned>(
    value: Option<&Bound<'_, PyAny>>,
    name: &str,
) -> PyResult<Vec<T>> {
    match value {
        None => Ok(Vec::new()),
        Some(v) if v.is_none() => Ok(Vec::new()),
        Some(v) => from_python(v, name),
    }
}
/// Grade and select an immutable calibration library; no file or catalog I/O.
#[pyfunction]
#[pyo3(signature=(*,frames=None,lights=None,quality=None,global_settings=None,rigs=None,grouping=None,available_responses=None,night_shifts=None,regrade=false, master_failures=None, pedestal_evidence=None, force_flat_without_pedestal=false, flat_star_masking=false, flat_stability=None, prepared_grouping=None, flat_association="automatic", dark_association="automatic"))]
#[allow(clippy::too_many_arguments)]
fn calibration_library_plan<'py>(
    py: Python<'py>,
    frames: Option<&Bound<'py, PyAny>>,
    lights: Option<&Bound<'py, PyAny>>,
    quality: Option<&Bound<'py, PyAny>>,
    global_settings: Option<&Bound<'py, PyAny>>,
    rigs: Option<&Bound<'py, PyAny>>,
    grouping: Option<&Bound<'py, PyAny>>,
    available_responses: Option<Vec<String>>,
    night_shifts: Option<&Bound<'py, PyAny>>,
    regrade: bool,
    master_failures: Option<&Bound<'py, PyAny>>,
    pedestal_evidence: Option<&Bound<'py, PyAny>>,
    force_flat_without_pedestal: bool,
    flat_star_masking: bool,
    flat_stability: Option<&Bound<'py, PyAny>>,
    prepared_grouping: Option<PyRef<'py, FlatGroups>>,
    flat_association: &str,
    dark_association: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let mode = |value: &str| {
        serde_json::from_value(serde_json::Value::String(value.into()))
            .map_err(|e| PyValueError::new_err(format!("invalid association mode: {e}")))
    };
    let request = library::Request {
        flat_association: mode(flat_association)?,
        dark_association: mode(dark_association)?,
        frames: records(frames, "frames")?,
        lights: records(lights, "lights")?,
        quality: records(quality, "quality")?,
        global_settings: records(global_settings, "global settings")?,
        rigs: records(rigs, "rig settings")?,
        grouping: match grouping {
            None => Default::default(),
            Some(v) if v.is_none() => Default::default(),
            Some(v) => from_python(v, "grouping")?,
        },
        available_responses: available_responses.unwrap_or_default(),
        night_shifts: records(night_shifts, "night shifts")?,
        regrade,
        master_failures: records(master_failures, "master failures")?,
        pedestal_evidence: records(pedestal_evidence, "pedestal evidence")?,
        force_flat_without_pedestal,
        flat_star_masking,
        flat_stability: match flat_stability {
            None => Default::default(),
            Some(v) if v.is_none() => Default::default(),
            Some(v) => from_python(v, "flat stability policy")?,
        },
    };
    let book = prepared_grouping.map(|groups| Arc::clone(&groups.book));
    let result = py
        .detach(|| match book {
            Some(book) => library::evaluate_with_book(request, book),
            None => library::evaluate(request),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    to_python(py, &result)
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let code = std::ffi::CString::new(include_str!("library_types.py"))
        .expect("type definitions contain no NUL");
    module.py().run(&code, Some(&module.dict()), None)?;
    for name in module
        .getattr("_calibration_exports")?
        .extract::<Vec<String>>()?
    {
        module.add(name.as_str(), module.getattr(name.as_str())?)?;
    }
    module.add_class::<FlatGroups>()?;
    module.add_function(wrap_pyfunction!(calibration_library_plan, module)?)?;
    module.add_function(wrap_pyfunction!(calibration_flat_response_plan, module)?)?;
    module.add_function(wrap_pyfunction!(calibration_flat_work_plan, module)?)?;
    module.add_function(wrap_pyfunction!(calibration_flat_work_finish, module)?)?;
    module.add_function(wrap_pyfunction!(master_flat_stability, module)?)?;
    module.add_function(wrap_pyfunction!(master_flat_stability_images, module)?)?;
    module.add_function(wrap_pyfunction!(calibration_master_plan, module)?)?;
    module.add_function(wrap_pyfunction!(calibration_master_cohort, module)?)?;
    module.add_function(wrap_pyfunction!(calibration_flat_retry, module)?)?;
    Ok(())
}

// Both call styles use this one decoder. Request objects are retained for
// existing clients; keyword calls expose the same fields to interactive users.
fn request_or_keywords<T: serde::de::DeserializeOwned>(
    py: Python<'_>,
    request: Option<&Bound<'_, PyAny>>,
    fields: &[(&str, Option<&Bound<'_, PyAny>>)],
    name: &str,
) -> PyResult<T> {
    let keywords = pyo3::types::PyDict::new(py);
    for (key, value) in fields {
        if let Some(value) = value {
            keywords.set_item(key, value)?;
        }
    }
    if let Some(request) = request {
        if !keywords.is_empty() {
            return Err(PyValueError::new_err(
                "use either a request mapping or keyword fields, not both",
            ));
        }
        from_python(request, name)
    } else {
        from_python(keywords.as_any(), name)
    }
}

/// Choose raw-flat response sessions and samples with shared grading/settings.
#[pyfunction]
#[pyo3(signature=(request=None,*,frames=None,quality=None,global_settings=None,rigs=None,night_shifts=None,regrade=None,flat_association=None))]
#[allow(clippy::too_many_arguments)]
fn calibration_flat_response_plan<'py>(
    py: Python<'py>,
    request: Option<&Bound<'py, PyAny>>,
    frames: Option<&Bound<'py, PyAny>>,
    quality: Option<&Bound<'py, PyAny>>,
    global_settings: Option<&Bound<'py, PyAny>>,
    rigs: Option<&Bound<'py, PyAny>>,
    night_shifts: Option<&Bound<'py, PyAny>>,
    regrade: Option<&Bound<'py, PyAny>>,
    flat_association: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let request: library::Request = request_or_keywords(
        py,
        request,
        &[
            ("frames", frames),
            ("quality", quality),
            ("global_settings", global_settings),
            ("rigs", rigs),
            ("night_shifts", night_shifts),
            ("regrade", regrade),
            ("flat_association", flat_association),
        ],
        "calibration_flat_response_plan request",
    )?;
    let result = py
        .detach(|| library::plan_flat_responses(request))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    to_python(py, &result)
}

/// Resolve one selected session without repeating grouping, matching or file I/O.
#[pyfunction]
#[pyo3(signature = (request=None, *, frames=None,lights=None,quality=None,global_settings=None,rigs=None,flat_basis=None,flat_family=None,flat_held_back=None,flat_star_masking=None,flat_stability=None,master_failures=None,pedestal_evidence=None,force_flat_without_pedestal=None,flat_association=None,dark_association=None))]
#[allow(clippy::too_many_arguments)]
fn calibration_master_plan<'py>(
    py: Python<'py>,
    request: Option<&Bound<'py, PyAny>>,
    frames: Option<&Bound<'py, PyAny>>,
    lights: Option<&Bound<'py, PyAny>>,
    quality: Option<&Bound<'py, PyAny>>,
    global_settings: Option<&Bound<'py, PyAny>>,
    rigs: Option<&Bound<'py, PyAny>>,
    flat_basis: Option<&Bound<'py, PyAny>>,
    flat_family: Option<&Bound<'py, PyAny>>,
    flat_held_back: Option<&Bound<'py, PyAny>>,
    flat_star_masking: Option<&Bound<'py, PyAny>>,
    flat_stability: Option<&Bound<'py, PyAny>>,
    master_failures: Option<&Bound<'py, PyAny>>,
    pedestal_evidence: Option<&Bound<'py, PyAny>>,
    force_flat_without_pedestal: Option<&Bound<'py, PyAny>>,
    flat_association: Option<&Bound<'py, PyAny>>,
    dark_association: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let request: library::MasterPlanRequest = request_or_keywords(
        py,
        request,
        &[
            ("frames", frames),
            ("lights", lights),
            ("quality", quality),
            ("global_settings", global_settings),
            ("rigs", rigs),
            ("flat_association", flat_association),
            ("dark_association", dark_association),
            ("flat_basis", flat_basis),
            ("flat_family", flat_family),
            ("flat_held_back", flat_held_back),
            ("flat_star_masking", flat_star_masking),
            ("flat_stability", flat_stability),
            ("master_failures", master_failures),
            ("pedestal_evidence", pedestal_evidence),
            ("force_flat_without_pedestal", force_flat_without_pedestal),
        ],
        "calibration_master_plan request",
    )?;
    let result = py
        .detach(|| library::plan_masters(request))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    to_python(py, &result)
}

/// Preview a coherent cohort. Invalid settings raise ValueError.
#[pyfunction]
#[pyo3(signature = (request=None, *, kind=None,frames=None,settings=None))]
fn calibration_master_cohort<'py>(
    py: Python<'py>,
    request: Option<&Bound<'py, PyAny>>,
    kind: Option<&Bound<'py, PyAny>>,
    frames: Option<&Bound<'py, PyAny>>,
    settings: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let request: library::MasterCohortRequest = request_or_keywords(
        py,
        request,
        &[("kind", kind), ("frames", frames), ("settings", settings)],
        "calibration_master_cohort request",
    )?;
    let result = py
        .detach(|| library::master_cohort(request))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    to_python(py, &result)
}

/// Decide the next flat action. Pass session=plan to preserve the resolved policy.
#[pyfunction]
#[pyo3(signature = (request=None, *, session=None,candidates=None,held_back=None,settings=None,policy=None,first=None,trimmed_attempted=None,trimmed=None,alternate_attempted=None,alternate=None))]
#[allow(clippy::too_many_arguments)]
fn calibration_flat_retry<'py>(
    py: Python<'py>,
    request: Option<&Bound<'py, PyAny>>,
    session: Option<&Bound<'py, PyAny>>,
    candidates: Option<&Bound<'py, PyAny>>,
    held_back: Option<&Bound<'py, PyAny>>,
    settings: Option<&Bound<'py, PyAny>>,
    policy: Option<&Bound<'py, PyAny>>,
    first: Option<&Bound<'py, PyAny>>,
    trimmed_attempted: Option<&Bound<'py, PyAny>>,
    trimmed: Option<&Bound<'py, PyAny>>,
    alternate_attempted: Option<&Bound<'py, PyAny>>,
    alternate: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let request: library::FlatRetryRequest = request_or_keywords(
        py,
        request,
        &[
            ("session", session),
            ("candidates", candidates),
            ("held_back", held_back),
            ("settings", settings),
            ("policy", policy),
            ("first", first),
            ("trimmed_attempted", trimmed_attempted),
            ("trimmed", trimmed),
            ("alternate_attempted", alternate_attempted),
            ("alternate", alternate),
        ],
        "calibration_flat_retry request",
    )?;
    let result = py
        .detach(|| library::flat_retry(request))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    to_python(py, &result)
}

/// Measure sources from files. For cached pixels use master_flat_stability_images.
#[pyfunction]
#[pyo3(signature = (request=None, *, master=None,inputs=None,bias=None,dark=None,dark_exposure_seconds=None,normalized_full_scale=None))]
#[allow(clippy::too_many_arguments)]
fn master_flat_stability<'py>(
    py: Python<'py>,
    request: Option<&Bound<'py, PyAny>>,
    master: Option<&Bound<'py, PyAny>>,
    inputs: Option<&Bound<'py, PyAny>>,
    bias: Option<&Bound<'py, PyAny>>,
    dark: Option<&Bound<'py, PyAny>>,
    dark_exposure_seconds: Option<&Bound<'py, PyAny>>,
    normalized_full_scale: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let request: seiza_stacking::FlatStabilityFileRequest = request_or_keywords(
        py,
        request,
        &[
            ("master", master),
            ("inputs", inputs),
            ("bias", bias),
            ("dark", dark),
            ("dark_exposure_seconds", dark_exposure_seconds),
            ("normalized_full_scale", normalized_full_scale),
        ],
        "master_flat_stability request",
    )?;
    let result = py
        .detach(|| seiza_stacking::measure_flat_stability_files(request))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    to_python(py, &result)
}

/// Measure cached NumPy images without file I/O. All pixels use the same units.
#[pyfunction]
#[pyo3(signature = (master, inputs, *, bias=None, dark=None, input_metadata=None, bayer=None, dark_exposure_seconds=None, dark_bias_subtracted=false))]
#[allow(clippy::too_many_arguments)]
fn master_flat_stability_images<'py>(
    py: Python<'py>,
    master: numpy::PyReadonlyArrayDyn<'py, f32>,
    inputs: Vec<numpy::PyReadonlyArrayDyn<'py, f32>>,
    bias: Option<numpy::PyReadonlyArrayDyn<'py, f32>>,
    dark: Option<numpy::PyReadonlyArrayDyn<'py, f32>>,
    input_metadata: Option<&Bound<'py, PyAny>>,
    bayer: Option<&Bound<'py, PyAny>>,
    dark_exposure_seconds: Option<f64>,
    dark_bias_subtracted: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let master = crate::arrays::linear_image(master)?;
    let inputs = inputs
        .into_iter()
        .map(crate::arrays::linear_image)
        .collect::<PyResult<Vec<_>>>()?;
    let bias = bias.map(crate::arrays::linear_image).transpose()?;
    let dark = dark.map(crate::arrays::linear_image).transpose()?;
    let options = seiza_stacking::FlatStabilityImageOptions {
        inputs: records(input_metadata, "input metadata")?,
        bayer: match bayer {
            None => None,
            Some(value) => Some(from_python(value, "Bayer layout")?),
        },
        dark_exposure_seconds,
        dark_bias_subtracted,
    };
    let result = py
        .detach(|| {
            seiza_stacking::measure_flat_stability_images(
                &master,
                &inputs,
                bias.as_ref(),
                dark.as_ref(),
                options,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    to_python(py, &result)
}

#[pyfunction]
#[pyo3(signature=(request=None,*,library=None,light_context=None))]
fn calibration_flat_work_plan<'py>(
    py: Python<'py>,
    request: Option<&Bound<'py, PyAny>>,
    library: Option<&Bound<'py, PyAny>>,
    light_context: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let request: library::FlatWorkRequest = request_or_keywords(
        py,
        request,
        &[("library", library), ("light_context", light_context)],
        "flat workflow request",
    )?;
    let result = py
        .detach(|| library::plan_flat_work(request))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    to_python(py, &result)
}
#[pyfunction]
#[pyo3(signature=(request=None,*,kind=None,question=None,choice=None,lights=None,fits=None))]
#[allow(clippy::too_many_arguments)]
fn calibration_flat_work_finish<'py>(
    py: Python<'py>,
    request: Option<&Bound<'py, PyAny>>,
    kind: Option<&Bound<'py, PyAny>>,
    question: Option<&Bound<'py, PyAny>>,
    choice: Option<&Bound<'py, PyAny>>,
    lights: Option<&Bound<'py, PyAny>>,
    fits: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let request: library::FlatWorkAnswer = request_or_keywords(
        py,
        request,
        &[
            ("kind", kind),
            ("question", question),
            ("choice", choice),
            ("lights", lights),
            ("fits", fits),
        ],
        "flat workflow feedback",
    )?;
    let result = py
        .detach(|| library::finish_flat_work(request))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    to_python(py, &result)
}
