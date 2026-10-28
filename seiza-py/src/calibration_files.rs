//! Python's file coordinator binds native methods, independently of C's JSON.
use crate::convert::{from_python, to_python};
use crate::library::PyCalibrationRules;
use crate::stacking::{keep_first, python_cancel_signal};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use seiza_calibration::library;
use seiza_stacking::calibration_files::{
    FileCalibration, PreparationOptions, PreparationProgress, Source,
};
use std::path::PathBuf;
use std::sync::Arc;

fn invalid(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}
fn records<T: serde::de::DeserializeOwned + Default>(
    value: Option<&Bound<'_, PyAny>>,
    name: &str,
) -> PyResult<T> {
    match value {
        Some(v) if !v.is_none() => from_python(v, name),
        _ => Ok(T::default()),
    }
}
/// File-oriented grading/matching/master generation. Database hosts retain the
/// existing CalibrationLibrary, quality and evidence classes.
#[pyclass(frozen, name = "CalibrationFiles", module = "seiza")]
struct PyCalibrationFiles {
    inner: FileCalibration,
}
#[pymethods]
impl PyCalibrationFiles {
    #[new]
    #[pyo3(signature=(lights,cache,*,library=None,provided=None,rules=None,incomplete="stop",flat_association="automatic",dark_association="automatic",flat_star_masking=false,flat_stability=None,choices=None,group_overrides=None,light_choices=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        lights: Vec<PathBuf>,
        cache: PathBuf,
        library: Option<PathBuf>,
        provided: Option<Vec<PathBuf>>,
        rules: Option<PyRef<'_, PyCalibrationRules>>,
        incomplete: &str,
        flat_association: &str,
        dark_association: &str,
        flat_star_masking: bool,
        flat_stability: Option<&Bound<'_, PyAny>>,
        choices: Option<&Bound<'_, PyAny>>,
        group_overrides: Option<&Bound<'_, PyAny>>,
        light_choices: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut inner = FileCalibration::new(lights, cache);
        inner.library = library;
        inner.provided = provided.unwrap_or_default();
        if let Some(rules) = rules {
            inner.rules = rules.book.clone();
        }
        inner.options = PreparationOptions {
            incomplete: incomplete.parse().map_err(invalid)?,
            associations: library::Associations {
                flat: flat_association.parse().map_err(invalid)?,
                dark: dark_association.parse().map_err(invalid)?,
            },
            masters: library::SessionOptions {
                flat_star_masking,
                flat_stability: records(flat_stability, "flat stability")?,
            },
            choices: records(choices, "quality choices")?,
            group_overrides: records(group_overrides, "group overrides")?,
            light_choices: records(light_choices, "light choices")?,
        };
        Ok(Self { inner })
    }
    /// Analyze without integrating masters; returns native snake_case records.
    #[pyo3(signature=(*,cancel=None,progress=None))]
    fn analyze<'py>(
        &self,
        py: Python<'py>,
        cancel: Option<Py<PyAny>>,
        progress: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.perform(py, None, cancel, progress)
    }
    /// Revalidate a review, build dependencies, and atomically publish corrected masters.
    #[pyo3(signature=(review_key,*,cancel=None,progress=None))]
    fn prepare<'py>(
        &self,
        py: Python<'py>,
        review_key: &str,
        cancel: Option<Py<PyAny>>,
        progress: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.perform(py, Some(review_key), cancel, progress)
    }
    /// Validate the source/master stamps from a report before using its files.
    #[staticmethod]
    fn verify(sources: &Bound<'_, PyAny>) -> PyResult<()> {
        let sources: Vec<Source> = from_python(sources, "calibration sources")?;
        seiza_stacking::calibration_files::verify_sources(&sources).map_err(invalid)
    }
}
impl PyCalibrationFiles {
    fn perform<'py>(
        &self,
        py: Python<'py>,
        key: Option<&str>,
        cancel: Option<Py<PyAny>>,
        progress: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (signal, raised) = python_cancel_signal(cancel);
        let errors = Arc::clone(&raised);
        let report = py.detach(|| {
            let notify = |update: &PreparationProgress| {
                if let Some(callback) = &progress {
                    Python::attach(|py| {
                        if errors.lock().is_ok_and(|slot| slot.is_some()) {
                            return;
                        }
                        let result = to_python(py, update).and_then(|v| callback.call1(py, (v,)));
                        if let Err(e) = result {
                            keep_first(&errors, e);
                        }
                    });
                }
            };
            match key {
                Some(key) => self.inner.prepare(key, signal, notify),
                None => self.inner.analyze(signal, notify),
            }
        });
        if let Some(error) = raised.lock().ok().and_then(|mut slot| slot.take()) {
            return Err(error);
        }
        to_python(py, &report.map_err(invalid)?)
    }
}
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCalibrationFiles>()
}
