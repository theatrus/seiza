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
#[pyo3(signature=(*,frames=None,lights=None,quality=None,global_settings=None,rigs=None,grouping=None,available_responses=None,night_shifts=None,regrade=false, master_failures=None, pedestal_evidence=None, force_flat_without_pedestal=false, flat_star_masking=false, prepared_grouping=None))]
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
    prepared_grouping: Option<PyRef<'py, FlatGroups>>,
) -> PyResult<Bound<'py, PyAny>> {
    let request = library::Request {
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
    module.add_class::<FlatGroups>()?;
    module.add_function(wrap_pyfunction!(calibration_library_plan, module)?)?;
    Ok(())
}
