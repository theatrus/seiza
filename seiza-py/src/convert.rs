//! Python values to and from the core's types, converted directly: no JSON
//! text between them.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// `value` as Python objects: structs and maps become dicts, sequences and
/// tuples lists, unit enum variants their names, and a float that is not
/// finite `None`. The values are those of the C ABI's JSON for the same
/// value: an `f32` comes through at its exact value (0.1 as
/// 0.10000000149011612), as the C ABI writes it, where a JSON text round
/// trip printed it shortest.
pub(crate) fn to_python<'py>(
    py: Python<'py>,
    value: &impl serde::Serialize,
) -> PyResult<Bound<'py, PyAny>> {
    let value = pythonize::pythonize(py, value)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    as_in_json(value)
}

/// What pythonize gives that JSON, and so the C ABI, does not: a Rust tuple
/// is a Python tuple, where JSON has a list, and a NaN or infinity a float,
/// where JSON has null.
fn as_in_json(value: Bound<'_, PyAny>) -> PyResult<Bound<'_, PyAny>> {
    use pyo3::types::{PyDict, PyFloat, PyList, PyTuple};
    let nested = |item: &Bound<'_, PyAny>| {
        item.is_instance_of::<PyDict>()
            || item.is_instance_of::<PyList>()
            || item.is_instance_of::<PyTuple>()
            || item.is_instance_of::<PyFloat>()
    };
    if let Ok(dict) = value.cast::<PyDict>() {
        for (key, item) in dict.iter() {
            if nested(&item) {
                dict.set_item(key, as_in_json(item)?)?;
            }
        }
        return Ok(value);
    }
    if let Ok(list) = value.cast::<PyList>() {
        for index in 0..list.len() {
            let item = list.get_item(index)?;
            if nested(&item) {
                list.set_item(index, as_in_json(item)?)?;
            }
        }
        return Ok(value);
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        let items = tuple.iter().map(as_in_json).collect::<PyResult<Vec<_>>>()?;
        return Ok(PyList::new(value.py(), items)?.into_any());
    }
    if let Ok(float) = value.cast::<PyFloat>()
        && !float.value().is_finite()
    {
        return Ok(value.py().None().into_bound(value.py()));
    }
    Ok(value)
}

/// A Python value (a dict, a list, a string) as the core type `T`. `name`
/// says what it is in the error.
pub(crate) fn from_python<T: serde::de::DeserializeOwned>(
    value: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<T> {
    pythonize::depythonize(value)
        .map_err(|error| PyValueError::new_err(format!("invalid {name}: {error}")))
}

/// `value` with every path-like object in it (a `pathlib.Path`, anything
/// with `__fspath__`) replaced by its string, through dicts, lists and
/// tuples, for records whose paths a caller may give either way.
pub(crate) fn with_fspaths<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::types::{PyDict, PyList, PyString, PyTuple};
    let py = value.py();
    if let Ok(dict) = value.cast::<PyDict>() {
        let out = PyDict::new(py);
        for (key, item) in dict.iter() {
            out.set_item(key, with_fspaths(&item)?)?;
        }
        return Ok(out.into_any());
    }
    if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        let items = value
            .try_iter()?
            .map(|item| with_fspaths(&item?))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(PyList::new(py, items)?.into_any());
    }
    if !value.is_instance_of::<PyString>() && value.hasattr("__fspath__")? {
        return py.import("os")?.call_method1("fspath", (value,));
    }
    Ok(value.clone())
}
