use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::PathBuf;

/// Return frame measurements and accepted-frame group baselines, including stellar peak SNR.
#[pyfunction]
#[pyo3(signature = (inputs, *, samples=262_144, filter=None, max_frames=None))]
fn analyze_project(
    py: Python<'_>,
    inputs: Vec<PathBuf>,
    samples: usize,
    filter: Option<String>,
    max_frames: Option<usize>,
) -> PyResult<Bound<'_, PyAny>> {
    let report = py
        .detach(|| {
            seiza_analysis::analyze(
                &inputs,
                &seiza_analysis::AnalysisOptions {
                    samples,
                    filter,
                    max_frames,
                },
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    crate::flat_response::to_python(py, &report)
}
#[pyfunction]
fn analysis_html(report: &Bound<'_, PyAny>) -> PyResult<String> {
    let json: String = report
        .py()
        .import("json")?
        .call_method1("dumps", (report,))?
        .extract()?;
    let report = serde_json::from_str(&json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    seiza_analysis::render_html(&report).map_err(|e| PyValueError::new_err(e.to_string()))
}
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_project, m)?)?;
    m.add_function(wrap_pyfunction!(analysis_html, m)?)?;
    Ok(())
}
