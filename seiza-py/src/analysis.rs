use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::PathBuf;

/// Return schema-3 mono/CFA/RGB component measurements and filter/component baselines.
/// Sampling geometry maps CFA sizes to original-image pixels without debayering.
#[pyfunction]
#[pyo3(signature = (inputs, *, samples=262_144, filter=None, max_frames=None, metrics_only=false))]
fn analyze_project(
    py: Python<'_>,
    inputs: Vec<PathBuf>,
    samples: usize,
    filter: Option<String>,
    max_frames: Option<usize>,
    metrics_only: bool,
) -> PyResult<Bound<'_, PyAny>> {
    let report = py
        .detach(|| {
            seiza_analysis::analyze(
                &inputs,
                &seiza_analysis::AnalysisOptions {
                    samples,
                    filter,
                    max_frames,
                    metrics_only,
                },
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    crate::convert::to_python(py, &report)
}
#[pyfunction]
fn analysis_html(report: &Bound<'_, PyAny>) -> PyResult<String> {
    let report = crate::convert::from_python(report, "report")?;
    seiza_analysis::render_html(&report).map_err(|e| PyValueError::new_err(e.to_string()))
}
#[pyfunction]
fn stack_report_html(report: &Bound<'_, PyAny>) -> PyResult<String> {
    let report = crate::convert::from_python(report, "report")?;
    seiza_analysis::render_stack_html(&report).map_err(|e| PyValueError::new_err(e.to_string()))
}
/// Render a schema-1 collection of labeled completed-stack reports.
#[pyfunction]
fn stack_collection_html(report: &Bound<'_, PyAny>) -> PyResult<String> {
    let report = crate::convert::from_python(report, "report")?;
    seiza_analysis::render_stack_collection_html(&report)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}
/// Select cheap retained diagnostics or rich source analysis, independently of stacking.
#[pyfunction]
#[pyo3(signature = (report=None, *, level="cheap", inputs=Vec::new()))]
fn prepare_stack_report<'py>(
    py: Python<'py>,
    report: Option<&Bound<'py, PyAny>>,
    level: &str,
    inputs: Vec<PathBuf>,
) -> PyResult<Bound<'py, PyAny>> {
    let report = report
        .map(|r| crate::convert::from_python(r, "report"))
        .transpose()?;
    let result = py
        .detach(|| seiza_analysis::prepare_stack_report(report, level, &inputs))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    crate::convert::to_python(py, &result)
}
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_project, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_stack_report, m)?)?;
    m.add_function(wrap_pyfunction!(analysis_html, m)?)?;
    m.add_function(wrap_pyfunction!(stack_report_html, m)?)?;
    m.add_function(wrap_pyfunction!(stack_collection_html, m)?)?;
    Ok(())
}
