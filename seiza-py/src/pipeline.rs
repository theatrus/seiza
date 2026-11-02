//! Distributed stacking: partial stacks, one session's step, and a channel
//! run into a store, bound from `seiza_pipeline`'s native types.

use crate::arrays::{image_array, u32_array};
use crate::stacking::StackError;
use numpy::PyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use seiza_pipeline::{
    ChannelPlan, ChannelReference, PartialStack, Session, SessionOptions, SessionOutcome,
    SessionStep, Store,
};
use std::path::PathBuf;

fn pipeline_error(error: seiza_pipeline::Error) -> PyErr {
    StackError::new_err(error.to_string())
}

/// A native value from a Python dict, its paths as ``str`` or ``Path``.
fn native<T: serde::de::DeserializeOwned>(value: &Bound<'_, PyAny>, name: &str) -> PyResult<T> {
    crate::convert::from_python(&crate::convert::with_fspaths(value)?, name)
}

fn session_outcome<'py>(py: Python<'py>, outcome: &SessionOutcome) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("key", outcome.key.to_string())?;
    dict.set_item("path", &outcome.path)?;
    dict.set_item("reused", outcome.reused)?;
    dict.set_item(
        "extended_from",
        outcome.extended_from.as_ref().map(ToString::to_string),
    )?;
    let rejected = PyList::empty(py);
    for (path, reason) in &outcome.rejected {
        rejected.append((path, reason.to_string()))?;
    }
    dict.set_item("rejected", rejected)?;
    Ok(dict)
}

/// One stack on one grid, as distributed stacking moves it between steps:
/// the weighted mean and the sums that let it be combined with others, and
/// the identity cards that say what made it.
#[pyclass(skip_from_py_object, frozen, name = "PartialStack", module = "seiza")]
pub(crate) struct PyPartialStack {
    inner: PartialStack,
}

#[pymethods]
impl PyPartialStack {
    /// Read a stack file a pipeline wrote, FITS or XISF.
    #[staticmethod]
    fn open(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = py
            .detach(|| PartialStack::open(&path))
            .map_err(pipeline_error)?;
        Ok(Self { inner })
    }

    /// Read a stack file's identity cards from its header alone, as a dict:
    /// how a host learns whether a step is done.
    #[staticmethod]
    fn read_identity<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyAny>> {
        let identity = PartialStack::read_identity(&path).map_err(pipeline_error)?;
        crate::convert::to_python(py, &identity)
    }

    /// Combine stacks on one grid into a tile stack, weighing each by its
    /// weight; the order they are given in does not matter.
    #[staticmethod]
    fn combine(py: Python<'_>, stacks: Vec<PyRef<'_, PyPartialStack>>) -> PyResult<Self> {
        let stacks = stacks
            .iter()
            .map(|stack| stack.inner.clone())
            .collect::<Vec<_>>();
        let inner = py
            .detach(|| PartialStack::combine(&stacks))
            .map_err(pipeline_error)?;
        Ok(Self { inner })
    }

    /// Combine the stack files at ``paths`` as :meth:`combine` combines
    /// stacks, reading one at a time, so memory does not grow with their
    /// number.
    #[staticmethod]
    fn combine_files(py: Python<'_>, paths: Vec<PathBuf>) -> PyResult<Self> {
        let inner = py
            .detach(|| PartialStack::combine_files(&paths))
            .map_err(pipeline_error)?;
        Ok(Self { inner })
    }

    /// Write the stack to a FITS or XISF file, as the path's extension
    /// names: the mean first, then WEIGHT, COVER, REJECT and VARIANCE.
    fn write(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        py.detach(|| self.inner.write(&path))
            .map_err(pipeline_error)
    }

    /// The identity cards, as a dict.
    #[getter]
    fn identity<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::convert::to_python(py, &self.inner.identity)
    }

    /// The weighted mean, ``NaN`` where no frame reached.
    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        image_array(py, &self.inner.mean)
    }

    /// Each sample's sum of frame weights.
    #[getter]
    fn weight<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        self.floats(py, &self.inner.weight)
    }

    /// Each sample's integrated frames.
    #[getter]
    fn coverage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<u32>>> {
        self.counts(py, &self.inner.coverage)
    }

    /// Each sample's rejected frames, or ``None``.
    #[getter]
    fn rejected<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArrayDyn<u32>>>> {
        self.inner
            .rejected
            .as_deref()
            .map(|rejected| self.counts(py, rejected))
            .transpose()
    }

    /// Each sample's variance of one frame of weight 1, or ``None``.
    #[getter]
    fn variance<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
        self.inner
            .variance
            .as_deref()
            .map(|variance| self.floats(py, variance))
            .transpose()
    }
}

impl PyPartialStack {
    fn floats<'py>(
        &self,
        py: Python<'py>,
        values: &[f32],
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let mean = &self.inner.mean;
        crate::arrays::float_array(py, mean.width, mean.height, mean.channels, values.to_vec())
    }

    fn counts<'py>(
        &self,
        py: Python<'py>,
        values: &[u32],
    ) -> PyResult<Bound<'py, PyArrayDyn<u32>>> {
        let mean = &self.inner.mean;
        u32_array(py, values, mean.width, mean.height, mean.channels)
    }
}

/// One session's step, planned: its files hashed and its key computed,
/// ready to run.
#[pyclass(skip_from_py_object, frozen, name = "SessionStep", module = "seiza")]
pub(crate) struct PySessionStep {
    inner: SessionStep,
}

#[pymethods]
impl PySessionStep {
    /// Plan ``session``'s step on ``reference``: hash its lights and masters
    /// and the reference's, and compute its key. ``reference`` is
    /// ``{"image", "light", "calibration"}``, ``session`` ``{"lights",
    /// "calibration"}``, each ``calibration`` a session record of
    /// :meth:`LiveStacker.open_fits_expanding_sessions`, and ``options``
    /// ``{"stack", "reintegration", "contributor", "host", "keep_context"}``.
    #[staticmethod]
    #[pyo3(signature = (reference, session, options=None))]
    fn plan(
        py: Python<'_>,
        reference: &Bound<'_, PyAny>,
        session: &Bound<'_, PyAny>,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let reference: ChannelReference = native(reference, "channel reference")?;
        let session: Session = native(session, "session")?;
        let options: SessionOptions = options
            .map(|options| native(options, "session options"))
            .transpose()?
            .unwrap_or_default();
        let inner = py
            .detach(|| SessionStep::plan(&reference, &session, &options))
            .map_err(pipeline_error)?;
        Ok(Self { inner })
    }

    /// The step's key, 64 hexadecimal digits.
    #[getter]
    fn key(&self) -> String {
        self.inner.key.to_string()
    }

    /// The digest of the lights, in order.
    #[getter]
    fn inputs(&self) -> String {
        self.inner.inputs.to_string()
    }

    /// Each light's digest, in order.
    #[getter]
    fn frames(&self) -> Vec<String> {
        self.inner.frames.iter().map(ToString::to_string).collect()
    }

    /// The digest of the session's calibration recipe.
    #[getter]
    fn calibration(&self) -> String {
        self.inner.calibration.to_string()
    }

    /// The digest of the options that change the pixels.
    #[getter]
    fn stack_options(&self) -> String {
        self.inner.stack_options.to_string()
    }

    /// The digest of the grid.
    #[getter]
    fn grid(&self) -> String {
        self.inner.grid.to_string()
    }

    /// The session's lights.
    #[getter]
    fn lights(&self) -> Vec<PathBuf> {
        self.inner.lights().to_vec()
    }

    /// Stack the session: its partial stack, and each light turned away
    /// with why, as ``(path, reason)``.
    fn run(&self, py: Python<'_>) -> PyResult<(PyPartialStack, Vec<(PathBuf, String)>)> {
        let stacked = py.detach(|| self.inner.run()).map_err(pipeline_error)?;
        let rejected = stacked
            .rejected
            .into_iter()
            .map(|(path, reason)| (path, reason.to_string()))
            .collect();
        Ok((
            PyPartialStack {
                inner: stacked.stack,
            },
            rejected,
        ))
    }

    /// Do the step against the store at ``store``: the stored partial stack
    /// when the store holds one under the key, or else the session stacked
    /// and stored. With ``keep_context``, a session that has gained lights
    /// since its context was stored is extended from it, only the new
    /// lights pushed. Returns ``{"key", "path", "reused", "extended_from",
    /// "rejected"}``, ``path`` ``None`` when no light was accepted and
    /// ``extended_from`` the key it was extended from, or ``None``.
    fn run_into<'py>(&self, py: Python<'py>, store: PathBuf) -> PyResult<Bound<'py, PyDict>> {
        let store = Store::new(store);
        let outcome = py
            .detach(|| self.inner.run_into(&store))
            .map_err(pipeline_error)?;
        session_outcome(py, &outcome)
    }
}

/// Stack a channel as partial stacks, one per session, into the store at
/// ``store``, and combine them into its tile stack, doing only the steps
/// whose outputs the store lacks. ``plan`` is ``{"reference", "sessions",
/// "options"}`` as :meth:`SessionStep.plan` takes them. Returns ``{"tile":
/// {"path", "reused", "identity"}, "sessions": [...]}``, each session as
/// :meth:`SessionStep.run_into` returns it.
#[pyfunction]
fn run_channel<'py>(
    py: Python<'py>,
    plan: &Bound<'_, PyAny>,
    store: PathBuf,
) -> PyResult<Bound<'py, PyDict>> {
    let plan: ChannelPlan = native(plan, "channel plan")?;
    let store = Store::new(store);
    let outcome = py
        .detach(|| seiza_pipeline::run_channel(&plan, &store, |_| {}))
        .map_err(pipeline_error)?;
    let tile = PyDict::new(py);
    tile.set_item("path", &outcome.path)?;
    tile.set_item("reused", outcome.reused)?;
    tile.set_item(
        "identity",
        crate::convert::to_python(py, &outcome.stack.identity)?,
    )?;
    let sessions = PyList::empty(py);
    for session in &outcome.sessions {
        sessions.append(session_outcome(py, session)?)?;
    }
    let answer = PyDict::new(py);
    answer.set_item("tile", tile)?;
    answer.set_item("sessions", sessions)?;
    Ok(answer)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPartialStack>()?;
    module.add_class::<PySessionStep>()?;
    module.add_function(wrap_pyfunction!(run_channel, module)?)?;
    Ok(())
}
