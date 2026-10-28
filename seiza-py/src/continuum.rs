//! Thin translations of Seiza's linear continuum model and diagnostics.
use crate::arrays::{into_image_array, linear_image};
use crate::convert::{from_python, to_python};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use seiza_stacking::{
    ContinuumFitOptions, ContinuumPhotometryOptions, ContinuumPlan, ContinuumSources, ContinuumStar,
};

pub(crate) fn core_error(error: seiza_stacking::Error) -> PyErr {
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
    let stars: Vec<ContinuumStar> = from_python(stars, "stars")?;
    let sources: ContinuumSources = from_python(sources, "sources")?;
    let options: ContinuumFitOptions = options
        .map(|value| from_python(value, "fit options"))
        .transpose()?
        .unwrap_or_default();
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
    let sources: ContinuumSources = from_python(sources, "sources")?;
    let fit_options: ContinuumFitOptions = fit_options
        .map(|value| from_python(value, "fit options"))
        .transpose()?
        .unwrap_or_default();
    let photometry_options: ContinuumPhotometryOptions = photometry_options
        .map(|value| from_python(value, "photometry options"))
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

/// Measure linear planes with producer-derived clipping masks, one uint8 mono
/// array per input (NB then references). Zero is usable, one is clipped.
#[pyfunction]
#[pyo3(signature = (narrowband, references, sources, clipping_masks, *, positions=None, fit_options=None, photometry_options=None))]
#[allow(clippy::too_many_arguments)]
fn measure_and_fit_continuum_masked<'py>(
    py: Python<'py>,
    narrowband: PyReadonlyArrayDyn<'_, f32>,
    references: Vec<PyReadonlyArrayDyn<'_, f32>>,
    sources: &Bound<'_, PyAny>,
    clipping_masks: Vec<PyReadonlyArrayDyn<'_, u8>>,
    positions: Option<Vec<[f64; 2]>>,
    fit_options: Option<&Bound<'_, PyAny>>,
    photometry_options: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let narrowband = linear_image(narrowband)?;
    let references = references
        .into_iter()
        .map(linear_image)
        .collect::<PyResult<Vec<_>>>()?;
    let masks = clipping_masks
        .into_iter()
        .map(|array| {
            if array.shape() != [narrowband.height, narrowband.width] || !array.is_c_contiguous() {
                return Err(PyValueError::new_err(
                    "Clipping masks must be C-contiguous mono uint8 arrays on the input grid",
                ));
            }
            Ok(array.as_slice()?.to_vec())
        })
        .collect::<PyResult<Vec<_>>>()?;
    let sources: ContinuumSources = from_python(sources, "sources")?;
    let options: ContinuumFitOptions = fit_options
        .map(|value| from_python(value, "fit options"))
        .transpose()?
        .unwrap_or_default();
    let photometry: ContinuumPhotometryOptions = photometry_options
        .map(|value| from_python(value, "photometry options"))
        .transpose()?
        .unwrap_or_default();
    let positions = positions.unwrap_or_default();
    let result = py
        .detach(move || {
            let references: Vec<_> = references.iter().collect();
            let masks: Vec<_> = masks.iter().map(Vec::as_slice).collect();
            seiza_stacking::measure_and_fit_continuum_masked(
                &narrowband,
                &references,
                &sources,
                &masks,
                &positions,
                &options,
                &photometry,
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
    let sources: ContinuumSources = from_python(sources, "sources")?;
    let plan: ContinuumPlan = from_python(plan, "continuum plan")?;
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
    m.add_function(wrap_pyfunction!(measure_and_fit_continuum_masked, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_continuum, m)?)?;
    Ok(())
}
