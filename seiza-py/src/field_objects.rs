//! The catalogued objects in a solved field — `seiza::field_objects`.

use crate::PyWcs;
use pyo3::prelude::*;
use seiza::field_objects::{FieldContour, FieldObject, FieldObjects, FieldOutline};
use std::path::PathBuf;

/// One line of an outline, in pixels counted from 0.
#[pyclass(skip_from_py_object, name = "ObjectContour", module = "seiza", frozen)]
#[derive(Clone)]
pub(crate) struct PyObjectContour {
    #[pyo3(get)]
    closed: bool,
    #[pyo3(get)]
    points: Vec<(f64, f64)>,
}

/// One of an object's outlines from the catalog, projected into the image.
#[pyclass(skip_from_py_object, name = "ObjectOutline", module = "seiza", frozen)]
#[derive(Clone)]
pub(crate) struct PyObjectOutline {
    #[pyo3(get)]
    geometry_id: String,
    #[pyo3(get)]
    source_record_id: String,
    #[pyo3(get)]
    role: &'static str,
    #[pyo3(get)]
    quality: &'static str,
    #[pyo3(get)]
    method: String,
    #[pyo3(get)]
    evidence: String,
    #[pyo3(get)]
    level: Option<String>,
    #[pyo3(get)]
    contours: Vec<PyObjectContour>,
}

/// A catalogued object placed in a solved image.
#[pyclass(skip_from_py_object, name = "FieldObject", module = "seiza", frozen)]
#[derive(Clone)]
pub(crate) struct PyFieldObject {
    #[pyo3(get)]
    stable_id: Option<String>,
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    common_name: String,
    #[pyo3(get)]
    kind: &'static str,
    #[pyo3(get)]
    source: &'static str,
    #[pyo3(get)]
    catalog_source: Option<String>,
    #[pyo3(get)]
    x: f64,
    #[pyo3(get)]
    y: f64,
    #[pyo3(get)]
    semi_major_pixels: f64,
    #[pyo3(get)]
    semi_minor_pixels: f64,
    #[pyo3(get)]
    angle_deg: Option<f64>,
    #[pyo3(get)]
    prominence: Option<f64>,
    #[pyo3(get)]
    ra_deg: f64,
    #[pyo3(get)]
    dec_deg: f64,
    #[pyo3(get)]
    discovered: Option<String>,
    #[pyo3(get)]
    near_capture: Option<bool>,
    #[pyo3(get)]
    distance_au: Option<f64>,
    #[pyo3(get)]
    motion_arcsec_per_hour: Option<f64>,
    #[pyo3(get)]
    direction_position_angle_deg: Option<f64>,
    #[pyo3(get)]
    direction_image_angle_deg: Option<f64>,
    #[pyo3(get)]
    outlines: Vec<PyObjectOutline>,
    layer: &'static str,
}

#[pymethods]
impl PyFieldObject {
    /// The overlay layer the object belongs to: ``"deep_sky"``,
    /// ``"named_stars"``, ``"transients"`` or ``"minor_bodies"``.
    #[getter]
    fn layer(&self) -> &'static str {
        self.layer
    }

    fn __repr__(&self) -> String {
        format!(
            "FieldObject(name={:?}, kind={:?}, x={:.1}, y={:.1})",
            self.name, self.kind, self.x, self.y
        )
    }
}

/// What :func:`field_objects` found, and why any catalog gave nothing.
#[pyclass(name = "FieldObjects", module = "seiza", frozen)]
pub(crate) struct PyFieldObjects {
    #[pyo3(get)]
    objects: Vec<PyFieldObject>,
    #[pyo3(get)]
    object_catalog_error: Option<String>,
    #[pyo3(get)]
    transient_catalog_error: Option<String>,
    #[pyo3(get)]
    minor_body_error: Option<String>,
}

#[pymethods]
impl PyFieldObjects {
    fn __repr__(&self) -> String {
        format!("FieldObjects({} objects)", self.objects.len())
    }
}

fn contour(contour: FieldContour) -> PyObjectContour {
    PyObjectContour {
        closed: contour.closed,
        points: contour.points,
    }
}

fn outline(outline: FieldOutline) -> PyObjectOutline {
    PyObjectOutline {
        geometry_id: outline.geometry_id,
        source_record_id: outline.source_record_id,
        role: outline.role.as_str(),
        quality: outline.quality.as_str(),
        method: outline.method,
        evidence: outline.evidence,
        level: outline.level,
        contours: outline.contours.into_iter().map(contour).collect(),
    }
}

fn object(object: FieldObject) -> PyFieldObject {
    PyFieldObject {
        layer: object.layer(),
        stable_id: object.stable_id,
        name: object.name,
        common_name: object.common_name,
        kind: object.kind,
        source: object.source,
        catalog_source: object.catalog_source,
        x: object.x,
        y: object.y,
        semi_major_pixels: object.semi_major_px,
        semi_minor_pixels: object.semi_minor_px,
        angle_deg: object.angle_deg,
        prominence: object.prominence,
        ra_deg: object.ra_deg,
        dec_deg: object.dec_deg,
        discovered: object.discovered,
        near_capture: object.near_capture,
        distance_au: object.distance_au,
        motion_arcsec_per_hour: object.motion_arcsec_per_hour,
        direction_position_angle_deg: object.direction_position_angle_deg,
        direction_image_angle_deg: object.direction_image_angle_deg,
        outlines: object.outlines.into_iter().map(outline).collect(),
    }
}

/// The catalogued objects in an image ``width`` by ``height`` pixels that
/// ``wcs`` solves: deep-sky objects and named stars, transients, and comets
/// and asteroids, placed in its pixels with their outlines, from the
/// catalogs :func:`fetch_catalogs` installs in ``catalog_dir`` (or the
/// standard places when it is ``None``).
///
/// ``capture_time`` (ISO 8601, UTC; see :func:`read_capture_time`) places
/// comets and asteroids and dates transients; without it none of the
/// first are placed. A catalog that is missing or unreadable gives no
/// objects, and the result's ``object_catalog_error``,
/// ``transient_catalog_error`` or ``minor_body_error`` says why.
#[pyfunction]
#[pyo3(signature = (wcs, width, height, *, catalog_dir=None, capture_time=None))]
fn field_objects(
    py: Python<'_>,
    wcs: PyRef<'_, PyWcs>,
    width: u32,
    height: u32,
    catalog_dir: Option<PathBuf>,
    capture_time: Option<String>,
) -> PyFieldObjects {
    let wcs = wcs.wcs.clone();
    let FieldObjects {
        objects,
        object_catalog_error,
        transient_catalog_error,
        minor_body_error,
    } = py.detach(|| {
        seiza::field_objects::field_objects(
            catalog_dir.as_deref(),
            &wcs,
            (width, height),
            capture_time.as_deref(),
        )
    });
    PyFieldObjects {
        objects: objects.into_iter().map(object).collect(),
        object_catalog_error,
        transient_catalog_error,
        minor_body_error,
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyObjectContour>()?;
    module.add_class::<PyObjectOutline>()?;
    module.add_class::<PyFieldObject>()?;
    module.add_class::<PyFieldObjects>()?;
    module.add_function(wrap_pyfunction!(field_objects, module)?)?;
    Ok(())
}
