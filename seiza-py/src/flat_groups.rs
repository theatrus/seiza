//! Python's batch interface to the shared flat grouping policy.
use crate::flat_response::{from_python, to_python};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use seiza_calibration::flat_response::groups;

fn records<T: serde::de::DeserializeOwned>(
    value: Option<&Bound<'_, PyAny>>,
    name: &str,
) -> PyResult<Vec<T>> {
    match value {
        None => Ok(Vec::new()),
        Some(value) if value.is_none() => Ok(Vec::new()),
        Some(value) => from_python(value, name),
    }
}

/// Measured flat families, events and light placements from values alone.
/// Results and input records use snake_case; times are Unix seconds, angles
/// degrees, dust/broad fractions of background. Automatic choices and resolved
/// personal choices are distinct. Compatible families must pass hard matching
/// gates before the host submits them. This does not grade or build masters.
#[pyfunction]
#[pyo3(signature = (*, responses=None, comparisons=None, evidence=None, imaging=None, imaging_lights=None, profile_rigs=None,
    limits=None, group_overrides=None, night_boundaries=None, epoch_samples=None,
    family_evidence=None, light_choices=None, lights=None))]
#[allow(clippy::too_many_arguments)]
fn flat_group_plan<'py>(
    py: Python<'py>,
    responses: Option<&Bound<'py, PyAny>>,
    comparisons: Option<&Bound<'py, PyAny>>,
    evidence: Option<&Bound<'py, PyAny>>,
    imaging: Option<&Bound<'py, PyAny>>,
    imaging_lights: Option<&Bound<'py, PyAny>>,
    profile_rigs: Option<&Bound<'py, PyAny>>,
    limits: Option<&Bound<'py, PyAny>>,
    group_overrides: Option<&Bound<'py, PyAny>>,
    night_boundaries: Option<&Bound<'py, PyAny>>,
    epoch_samples: Option<&Bound<'py, PyAny>>,
    family_evidence: Option<&Bound<'py, PyAny>>,
    light_choices: Option<&Bound<'py, PyAny>>,
    lights: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let request = groups::Request {
        responses: records(responses, "responses")?,
        comparisons: records(comparisons, "comparisons")?,
        evidence: records(evidence, "evidence")?,
        imaging: records(imaging, "imaging")?,
        imaging_lights: records(imaging_lights, "imaging lights")?,
        profile_rigs: records(profile_rigs, "profile rigs")?,
        limits: records(limits, "limits")?,
        group_overrides: records(group_overrides, "group overrides")?,
        night_boundaries: records(night_boundaries, "night boundaries")?,
        epoch_samples: records(epoch_samples, "epoch samples")?,
        family_evidence: records(family_evidence, "family evidence")?,
        light_choices: records(light_choices, "light choices")?,
        lights: records(lights, "lights")?,
    };
    let result = py
        .detach(|| groups::evaluate(request))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    to_python(py, &result)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(flat_group_plan, module)?)?;
    Ok(())
}
