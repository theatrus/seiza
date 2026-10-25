//! Thin translations of emission projection and display structure blending.
use crate::arrays::{into_image_array, linear_image};
use crate::continuum::{core_error, from_python};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use seiza_stacking::{EmissionBlendLine, EmissionBlendOptions, EmissionNegativePolicy};

#[pyfunction]
#[pyo3(signature=(broadband,residuals,lines,*,negative="preserve"))]
fn blend_emission_rgb<'py>(
    py: Python<'py>,
    broadband: PyReadonlyArrayDyn<'_, f32>,
    residuals: Vec<PyReadonlyArrayDyn<'_, f32>>,
    lines: &Bound<'_, PyAny>,
    negative: &str,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let base = linear_image(broadband)?;
    let residuals = residuals
        .into_iter()
        .map(linear_image)
        .collect::<PyResult<Vec<_>>>()?;
    let lines: Vec<EmissionBlendLine> = from_python(lines)?;
    let negative = match negative {
        "preserve" => EmissionNegativePolicy::Preserve,
        "positive_contribution" => EmissionNegativePolicy::PositiveContribution,
        _ => {
            return Err(PyValueError::new_err(
                "negative must be preserve or positive_contribution",
            ));
        }
    };
    let image = py
        .detach(move || {
            let residuals: Vec<_> = residuals.iter().collect();
            seiza_stacking::blend_emission_rgb(
                &base,
                &residuals,
                &lines,
                &EmissionBlendOptions { negative },
            )
        })
        .map_err(core_error)?;
    into_image_array(py, image)
}
#[pyfunction]
fn blend_display_structure<'py>(
    py: Python<'py>,
    narrowband: PyReadonlyArrayDyn<'_, f32>,
    broadband: PyReadonlyArrayDyn<'_, f32>,
    amount: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let n = linear_image(narrowband)?;
    let b = linear_image(broadband)?;
    let image = py
        .detach(move || seiza_stacking::blend_display_structure(&n, &b, amount))
        .map_err(core_error)?;
    into_image_array(py, image)
}
#[pyfunction]
#[pyo3(signature=(base,contribution,*,gains=[1.0,1.0,1.0]))]
fn add_rgb_contribution<'py>(
    py: Python<'py>,
    base: PyReadonlyArrayDyn<'_, f32>,
    contribution: PyReadonlyArrayDyn<'_, f32>,
    gains: [f32; 3],
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let base = linear_image(base)?;
    let contribution = linear_image(contribution)?;
    let result = py
        .detach(move || seiza_stacking::add_rgb_contribution(&base, &contribution, gains))
        .map_err(core_error)?;
    into_image_array(py, result)
}
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(blend_emission_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(add_rgb_contribution, m)?)?;
    m.add_function(wrap_pyfunction!(blend_display_structure, m)?)?;
    Ok(())
}
