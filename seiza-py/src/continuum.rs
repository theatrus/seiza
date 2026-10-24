//! Thin translations of Seiza's linear continuum model and diagnostics.
use crate::arrays::{into_image_array, linear_image};
use crate::flat_response::to_python;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use seiza_stacking::{
    ContinuumFitOptions, ContinuumPhotometryOptions, ContinuumPlan, ContinuumSources, ContinuumStar,
};

fn from_python<T: serde::de::DeserializeOwned>(value: &Bound<'_, PyAny>) -> PyResult<T> {
    let json: String = value
        .py()
        .import("json")?
        .call_method1("dumps", (value,))?
        .extract()?;
    serde_json::from_str(&json).map_err(|e| PyValueError::new_err(e.to_string()))
}
fn core_error(error: seiza_stacking::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Fit premeasured, isolated common stars on exact prepared linear revisions.
/// Returns the same tagged fitted/skipped outcome and diagnostics as the C ABI.
#[pyfunction]
#[pyo3(signature = (stars, sources, *, options=None))]
fn fit_continuum<'py>(
    py: Python<'py>,
    stars: &Bound<'_, PyAny>,
    sources: &Bound<'_, PyAny>,
    options: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let stars: Vec<ContinuumStar> = from_python(stars)?;
    let sources: ContinuumSources = from_python(sources)?;
    let options: ContinuumFitOptions = options.map(from_python).transpose()?.unwrap_or_default();
    let result = py
        .detach(move || seiza_stacking::fit_continuum(&stars, &sources, &options))
        .map_err(core_error)?;
    to_python(py, &result)
}

/// Detect or measure common stars, then fit the core's nonnegative model.
/// Arrays are C-contiguous mono float32 planes, aligned and linear with stars.
#[pyfunction]
#[pyo3(signature = (narrowband, references, sources, *, positions=None, fit_options=None, photometry_options=None))]
fn measure_and_fit_continuum<'py>(
    py: Python<'py>,
    narrowband: PyReadonlyArrayDyn<'_, f32>,
    references: Vec<PyReadonlyArrayDyn<'_, f32>>,
    sources: &Bound<'_, PyAny>,
    positions: Option<Vec<[f64; 2]>>,
    fit_options: Option<&Bound<'_, PyAny>>,
    photometry_options: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let narrowband = linear_image(narrowband)?;
    let references = references
        .into_iter()
        .map(linear_image)
        .collect::<PyResult<Vec<_>>>()?;
    let sources: ContinuumSources = from_python(sources)?;
    let fit_options: ContinuumFitOptions = fit_options
        .map(from_python)
        .transpose()?
        .unwrap_or_default();
    let photometry_options: ContinuumPhotometryOptions = photometry_options
        .map(from_python)
        .transpose()?
        .unwrap_or_default();
    let positions = positions.unwrap_or_default();
    let result = py
        .detach(move || {
            let references: Vec<_> = references.iter().collect();
            seiza_stacking::measure_and_fit_continuum(
                &narrowband,
                &references,
                &sources,
                &positions,
                &fit_options,
                &photometry_options,
            )
        })
        .map_err(core_error)?;
    to_python(py, &result)
}

/// Apply an exact-source continuum plan into a new signed mono float32 array.
/// Negative samples and missing contributing coverage are retained; no stretch
/// or normalization is applied and none of the input arrays is modified.
#[pyfunction]
fn subtract_continuum<'py>(
    py: Python<'py>,
    narrowband: PyReadonlyArrayDyn<'_, f32>,
    references: Vec<PyReadonlyArrayDyn<'_, f32>>,
    sources: &Bound<'_, PyAny>,
    plan: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let narrowband = linear_image(narrowband)?;
    let references = references
        .into_iter()
        .map(linear_image)
        .collect::<PyResult<Vec<_>>>()?;
    let sources: ContinuumSources = from_python(sources)?;
    let plan: ContinuumPlan = from_python(plan)?;
    let result = py
        .detach(move || {
            let references: Vec<_> = references.iter().collect();
            seiza_stacking::subtract_continuum(&narrowband, &references, &sources, &plan)
        })
        .map_err(core_error)?;
    into_image_array(py, result)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_continuum, m)?)?;
    m.add_function(wrap_pyfunction!(measure_and_fit_continuum, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_continuum, m)?)?;
    Ok(())
}
