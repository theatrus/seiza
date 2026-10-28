//! Constellation stick figures (`seiza::constellations`), as the C ABI's
//! `seiza_constellation_figures_json` and `seiza_constellations_project_json`
//! give them: dicts of the core's own JSON.
//!
//! The figures come from the Constellation Lines dataset by Marc van der
//! Sluys (CC BY 4.0). Anything that shows them must carry the `attribution`
//! both functions return.

use crate::PyWcs;
use crate::convert::to_python;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use seiza::constellations;

/// Every constellation figure and the stars its lines use: `{"attribution",
/// "epoch", "figures": [{"abbr", "name", "segments", "labels"}], "stars":
/// [{"hr", "ra", "dec", "mag"}]}`, positions in ICRS degrees at the Julian
/// `epoch`, each segment a pair of HR numbers.
#[pyfunction]
fn constellation_figures(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    to_python(py, &constellations::dataset())
}

/// The constellation figures of a solved `width` × `height` image, as pixel
/// polylines clipped to it: `{"attribution", "figures": [{"abbr", "name",
/// "polylines", "label"}]}`, pixels counted from 0.
#[pyfunction]
fn project_constellations<'py>(
    py: Python<'py>,
    wcs: &Bound<'py, PyWcs>,
    width: u32,
    height: u32,
) -> PyResult<Bound<'py, PyAny>> {
    if width == 0 || height == 0 {
        return Err(PyValueError::new_err(
            "the image needs a width and a height",
        ));
    }
    let wcs = wcs.get().wcs.clone();
    let projection = py.detach(|| constellations::project(&wcs, (width, height)));
    to_python(py, &projection)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(constellation_figures, module)?)?;
    module.add_function(wrap_pyfunction!(project_constellations, module)?)?;
    Ok(())
}
