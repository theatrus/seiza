//! RC-Astro external tools (BlurXTerminator, StarXTerminator,
//! NoiseXTerminator) through their standalone `rc-astro` CLI —
//! `seiza-stacking`'s `external` module.
//!
//! The CLI publishes a machine-readable contract: [`rc_astro_tool_schema`]
//! reads a tool's parameters (flags change between CLI builds, so requests
//! should be built from the schema), and [`rc_astro_process_file`] runs a
//! tool on an image file, returning every written file — StarXTerminator's
//! stars sidecar included.

use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use seiza_stacking::{
    CancelSignal, ExternalParameterKind, ExternalParameterValue, ExternalToolRequest,
    ExternalToolSchema, RcAstroCli,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// One schema parameter: its request name, CLI flag (absent for GUI-only
/// parameters), display strings, type ("float", "bool", or "int"), default,
/// and range.
#[pyclass(name = "RcAstroParameter", module = "seiza", frozen)]
#[derive(Clone)]
pub(crate) struct PyRcAstroParameter {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    flag: Option<String>,
    #[pyo3(get)]
    label: String,
    #[pyo3(get)]
    description: String,
    kind: ExternalParameterKind,
}

#[pymethods]
impl PyRcAstroParameter {
    #[getter]
    fn r#type(&self) -> &'static str {
        match self.kind {
            ExternalParameterKind::Float { .. } => "float",
            ExternalParameterKind::Bool { .. } => "bool",
            ExternalParameterKind::Int { .. } => "int",
        }
    }

    #[getter]
    fn default(&self, py: Python<'_>) -> PyObject {
        match self.kind {
            ExternalParameterKind::Float { default, .. } => {
                default.into_pyobject(py).unwrap().into()
            }
            ExternalParameterKind::Bool { default } => default
                .into_pyobject(py)
                .unwrap()
                .to_owned()
                .into_any()
                .unbind(),
            ExternalParameterKind::Int { default, .. } => default.into_pyobject(py).unwrap().into(),
        }
    }

    #[getter]
    fn min(&self) -> Option<f64> {
        match self.kind {
            ExternalParameterKind::Float { min, .. } => Some(min).filter(|value| value.is_finite()),
            ExternalParameterKind::Int { min, .. } => (min != i64::MIN).then_some(min as f64),
            ExternalParameterKind::Bool { .. } => None,
        }
    }

    #[getter]
    fn max(&self) -> Option<f64> {
        match self.kind {
            ExternalParameterKind::Float { max, .. } => Some(max).filter(|value| value.is_finite()),
            ExternalParameterKind::Int { max, .. } => (max != i64::MAX).then_some(max as f64),
            ExternalParameterKind::Bool { .. } => None,
        }
    }
}

/// One tool's live contract from `rc-astro <tool> --json`: versions,
/// license state, and its parameters. `cli_version` and `ml_version` belong
/// in any cache key a caller builds — an upgrade changes the output for
/// identical inputs.
#[pyclass(name = "RcAstroSchema", module = "seiza", frozen)]
pub(crate) struct PyRcAstroSchema {
    #[pyo3(get)]
    contract_version: u32,
    #[pyo3(get)]
    cli_version: String,
    #[pyo3(get)]
    key: String,
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    ml_version: Option<i64>,
    #[pyo3(get)]
    licensed: bool,
    #[pyo3(get)]
    license_message: Option<String>,
    #[pyo3(get)]
    parameters: Vec<PyRcAstroParameter>,
}

/// What a run wrote: the requested output, any sidecars (StarXTerminator's
/// stars image), the compute device the tool reported, and its warnings.
#[pyclass(name = "RcAstroRun", module = "seiza", frozen)]
pub(crate) struct PyRcAstroRun {
    #[pyo3(get)]
    primary: PathBuf,
    #[pyo3(get)]
    sidecars: Vec<PathBuf>,
    #[pyo3(get)]
    device: Option<String>,
    #[pyo3(get)]
    warnings: Vec<String>,
    #[pyo3(get)]
    cli_version: String,
    #[pyo3(get)]
    ml_version: Option<i64>,
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRcAstroParameter>()?;
    module.add_class::<PyRcAstroSchema>()?;
    module.add_class::<PyRcAstroRun>()?;
    module.add_function(wrap_pyfunction!(rc_astro_locate, module)?)?;
    module.add_function(wrap_pyfunction!(rc_astro_tool_schema, module)?)?;
    module.add_function(wrap_pyfunction!(rc_astro_process_file, module)?)?;
    Ok(())
}

fn cli(executable: Option<PathBuf>, host: Option<String>) -> PyResult<RcAstroCli> {
    let cli = match executable {
        Some(path) => RcAstroCli::with_executable(path),
        None => RcAstroCli::locate()
            .ok_or_else(|| PyFileNotFoundError::new_err("rc-astro was not found on PATH"))?,
    };
    Ok(match host {
        Some(host) => cli.with_host(host),
        None => cli.with_host(format!("seiza-py-{}", env!("CARGO_PKG_VERSION"))),
    })
}

fn schema_class(schema: &ExternalToolSchema) -> PyRcAstroSchema {
    PyRcAstroSchema {
        contract_version: schema.schema_version,
        cli_version: schema.cli_version.clone(),
        key: schema.key.clone(),
        name: schema.name.clone(),
        ml_version: schema.ml_version,
        licensed: schema.licensed,
        license_message: schema.license_message.clone(),
        parameters: schema
            .parameters
            .iter()
            .map(|parameter| PyRcAstroParameter {
                name: parameter.name.clone(),
                flag: parameter.flag.clone(),
                label: parameter.label.clone(),
                description: parameter.description.clone(),
                kind: parameter.kind.clone(),
            })
            .collect(),
    }
}

/// The path of the `rc-astro` executable on `PATH`, or `None` when the CLI
/// is not installed.
#[pyfunction]
pub(crate) fn rc_astro_locate() -> Option<String> {
    RcAstroCli::locate().map(|cli| cli.executable().display().to_string())
}

/// Read one tool's live contract: `rc-astro <tool> --json`. `tool` is
/// "bxt", "sxt", or "nxt"; `executable` overrides the `PATH` search.
#[pyfunction]
#[pyo3(signature = (tool, *, executable=None))]
pub(crate) fn rc_astro_tool_schema(
    py: Python<'_>,
    tool: &str,
    executable: Option<PathBuf>,
) -> PyResult<PyRcAstroSchema> {
    let cli = cli(executable, None)?;
    let tool = tool.to_string();
    let schema = py
        .allow_threads(move || cli.tool_schema(&tool))
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(schema_class(&schema))
}

/// Run one RC-Astro tool on an image file (FITS, XISF, or TIFF — whatever
/// the installed CLI accepts), writing `output` and possibly sidecars
/// beside it. `parameters` maps schema names to values (bools for switches
/// — `False` emits the CLI's `--no-` negation, overriding a true default —
/// and numbers for the rest; whole numbers are fine for float parameters
/// and vice versa). `progress` is called with a fraction in 0..1 as the
/// tool reports it. Ctrl-C cancels within half a second, as does `cancel`
/// returning true; a child completely silent for ten minutes is killed —
/// a first run downloads ML models, which `rc-astro download-models`
/// handles ahead of time. StarXTerminator's "stars" parameter puts the
/// stars image in `sidecars`.
#[pyfunction]
#[pyo3(signature = (tool, input, output, *, parameters=None, device=None, executable=None, host=None, progress=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn rc_astro_process_file(
    py: Python<'_>,
    tool: &str,
    input: PathBuf,
    output: PathBuf,
    parameters: Option<&Bound<'_, PyDict>>,
    device: Option<String>,
    executable: Option<PathBuf>,
    host: Option<String>,
    progress: Option<PyObject>,
    cancel: Option<PyObject>,
) -> PyResult<PyRcAstroRun> {
    let cli = cli(executable, host)?;
    let mut values = Vec::new();
    if let Some(parameters) = parameters {
        for (name, value) in parameters.iter() {
            let name: String = name.extract()?;
            let value = if let Ok(flag) = value.extract::<bool>() {
                ExternalParameterValue::Bool(flag)
            } else if let Ok(whole) = value.extract::<i64>() {
                ExternalParameterValue::Int(whole)
            } else if let Ok(real) = value.extract::<f64>() {
                ExternalParameterValue::Float(real)
            } else {
                return Err(PyValueError::new_err(format!(
                    "parameter {name:?} must be a bool, int, or float"
                )));
            };
            values.push((name, value));
        }
    }
    let request = ExternalToolRequest {
        tool: tool.to_string(),
        parameters: values,
        device,
    };
    let tool_name = tool.to_string();
    // A raising progress callback must not vanish: the first exception is
    // kept and re-raised once the run settles (the run itself is not
    // interrupted — progress is advisory). A raising cancel predicate stops
    // the run and re-raises the same way, so Ctrl-C surfaces as
    // KeyboardInterrupt rather than a generic run error.
    let raised: Arc<Mutex<Option<PyErr>>> = Arc::new(Mutex::new(None));
    let signal = {
        let raised = Arc::clone(&raised);
        CancelSignal::new(move || {
            Python::with_gil(|py| {
                let asked = py.check_signals().and_then(|()| match &cancel {
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
        })
    };
    let outcome = py.allow_threads(|| {
        let schema = cli.tool_schema(&tool_name)?;
        let mut report = |fraction: f32| {
            if let Some(callback) = &progress {
                Python::with_gil(|py| {
                    if let Err(error) = callback.call1(py, (fraction,))
                        && let Ok(mut slot) = raised.lock()
                        && slot.is_none()
                    {
                        *slot = Some(error);
                    }
                });
            }
        };
        let run = cli.run_on_file(
            &schema,
            &request,
            &input,
            &output,
            Some(&signal),
            &mut report,
        )?;
        Ok::<_, seiza_stacking::Error>((schema, run))
    });
    if let Ok(mut slot) = raised.lock()
        && let Some(error) = slot.take()
    {
        return Err(error);
    }
    let (schema, run) = outcome.map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(PyRcAstroRun {
        primary: run.primary,
        sidecars: run.sidecars,
        device: run.device,
        warnings: run.warnings,
        cli_version: schema.cli_version.clone(),
        ml_version: schema.ml_version,
    })
}
