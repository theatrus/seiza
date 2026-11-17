//! Thin translation of the core's forced aperture photometry.
use numpy::ndarray::{ArrayView2, Ix2};
use numpy::{PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use seiza_stacking::{
    ApertureFlags, ApertureGeometry, ApertureMeasurement, AperturePhotometry,
    AperturePhotometryOptions, AperturePixel,
};

/// What is wrong with a measurement, if anything.
#[pyclass(skip_from_py_object, name = "ApertureFlags", module = "seiza", frozen)]
#[derive(Clone)]
pub(crate) struct PyApertureFlags {
    /// A pixel of the aperture is saturated; the flux is a lower bound.
    #[pyo3(get)]
    saturated: bool,
    /// Part of the aperture is off the image, masked or not finite.
    #[pyo3(get)]
    missing_coverage: bool,
    /// No flux could be measured.
    #[pyo3(get)]
    unusable: bool,
}

impl From<ApertureFlags> for PyApertureFlags {
    fn from(flags: ApertureFlags) -> Self {
        Self {
            saturated: flags.saturated,
            missing_coverage: flags.missing_coverage,
            unusable: flags.unusable,
        }
    }
}

#[pymethods]
impl PyApertureFlags {
    /// True when any flag is set.
    fn __bool__(&self) -> bool {
        self.saturated || self.missing_coverage || self.unusable
    }

    fn __repr__(&self) -> String {
        let flag = |value| if value { "True" } else { "False" };
        format!(
            "ApertureFlags(saturated={}, missing_coverage={}, unusable={})",
            flag(self.saturated),
            flag(self.missing_coverage),
            flag(self.unusable)
        )
    }
}

/// One position's forced aperture measurement, in the image's units.
#[pyclass(
    skip_from_py_object,
    name = "ApertureMeasurement",
    module = "seiza",
    frozen
)]
pub(crate) struct PyApertureMeasurement {
    #[pyo3(get)]
    x: f64,
    #[pyo3(get)]
    y: f64,
    #[pyo3(get)]
    flux: Option<f64>,
    #[pyo3(get)]
    flux_error: Option<f64>,
    #[pyo3(get)]
    magnitude: Option<f64>,
    #[pyo3(get)]
    magnitude_error: Option<f64>,
    #[pyo3(get)]
    background: Option<f64>,
    #[pyo3(get)]
    background_noise: Option<f64>,
    #[pyo3(get)]
    aperture_sum: Option<f64>,
    #[pyo3(get)]
    aperture_area: f64,
    #[pyo3(get)]
    annulus_pixels: usize,
    #[pyo3(get)]
    peak: Option<f64>,
    #[pyo3(get)]
    flags: PyApertureFlags,
}

impl From<ApertureMeasurement> for PyApertureMeasurement {
    fn from(measurement: ApertureMeasurement) -> Self {
        Self {
            x: measurement.x,
            y: measurement.y,
            flux: measurement.flux,
            flux_error: measurement.flux_error,
            magnitude: measurement.magnitude,
            magnitude_error: measurement.magnitude_error,
            background: measurement.background,
            background_noise: measurement.background_noise,
            aperture_sum: measurement.aperture_sum,
            aperture_area: measurement.aperture_area,
            annulus_pixels: measurement.annulus_pixels,
            peak: measurement.peak,
            flags: measurement.flags.into(),
        }
    }
}

#[pymethods]
impl PyApertureMeasurement {
    fn __repr__(&self) -> String {
        let number = |value: Option<f64>| value.map_or("None".into(), |value| format!("{value}"));
        format!(
            "ApertureMeasurement(x={}, y={}, flux={}, flux_error={}, magnitude={}, flags={})",
            self.x,
            self.y,
            number(self.flux),
            number(self.flux_error),
            number(self.magnitude),
            self.flags.__repr__()
        )
    }
}

/// Validate a plane without materializing its pixels.
fn mono_shape(array: &PyReadonlyArrayDyn<'_, f32>, name: &str) -> PyResult<(usize, usize)> {
    let [height, width] = array.shape() else {
        return Err(PyValueError::new_err(format!(
            "{name} must be one mono plane of shape (height, width); extract the channel to measure first, such as rgb[..., 1]"
        )));
    };
    Ok((*width, *height))
}

enum BytePlane<'py> {
    Bool(PyReadonlyArrayDyn<'py, bool>),
    Byte(PyReadonlyArrayDyn<'py, u8>),
}
impl BytePlane<'_> {
    fn shape(&self) -> &[usize] {
        match self {
            Self::Bool(a) => a.shape(),
            Self::Byte(a) => a.shape(),
        }
    }
    fn view(&self) -> ByteView<'_> {
        match self {
            Self::Bool(a) => ByteView::Bool(
                a.as_array()
                    .into_dimensionality()
                    .expect("validated map shape"),
            ),
            Self::Byte(a) => ByteView::Byte(
                a.as_array()
                    .into_dimensionality()
                    .expect("validated map shape"),
            ),
        }
    }
}

enum ByteView<'a> {
    Bool(ArrayView2<'a, bool>),
    Byte(ArrayView2<'a, u8>),
}
impl ByteView<'_> {
    fn at(&self, x: usize, y: usize) -> bool {
        match self {
            Self::Bool(a) => a[[y, x]],
            Self::Byte(a) => a[[y, x]] != 0,
        }
    }
}

/// Borrow a bool or uint8 map; sparse sampling reads through its strides.
fn byte_plane<'py>(
    value: Option<&Bound<'py, PyAny>>,
    width: usize,
    height: usize,
    name: &str,
) -> PyResult<Option<BytePlane<'py>>> {
    let Some(value) = value else { return Ok(None) };
    let plane = if let Ok(array) = value.extract::<PyReadonlyArrayDyn<'py, bool>>() {
        BytePlane::Bool(array)
    } else if let Ok(array) = value.extract::<PyReadonlyArrayDyn<'py, u8>>() {
        BytePlane::Byte(array)
    } else {
        return Err(PyValueError::new_err(format!(
            "{name} must be a bool or uint8 array"
        )));
    };
    if plane.shape() != [height, width] {
        return Err(PyValueError::new_err(format!(
            "{name} must have the image's shape ({height}, {width})"
        )));
    }
    Ok(Some(plane))
}

/// Positions as a sequence of (x, y) pairs or an (n, 2) float array.
fn positions(value: &Bound<'_, PyAny>) -> PyResult<Vec<[f64; 2]>> {
    if let Ok(array) = value.extract::<PyReadonlyArray2<'_, f64>>() {
        if array.shape()[1] != 2 {
            return Err(PyValueError::new_err(
                "positions must have shape (n, 2): one (x, y) per row",
            ));
        }
        return Ok(array
            .as_array()
            .rows()
            .into_iter()
            .map(|row| [row[0], row[1]])
            .collect());
    }
    value.extract().map_err(|_| {
        PyValueError::new_err("positions must be (x, y) pairs or a float64 array of shape (n, 2)")
    })
}

/// Measure a circular aperture at each of `positions` on a linear mono
/// `image`, a float32 array of shape (height, width), and return one
/// `ApertureMeasurement` per position, in order. Positions are (x, y),
/// zero-based with pixel centres on whole numbers, and are measured where
/// given: nothing is detected, centred or moved, and no colour is converted.
/// The core's `aperture_photometry` documents the weighting, the background
/// and the noise model.
#[pyfunction]
#[pyo3(signature = (image, positions, *, aperture_radius, annulus_inner, annulus_outer, saturation=None, gain=None, mask=None, saturation_mask=None, uncertainty=None))]
#[allow(clippy::too_many_arguments)]
fn aperture_photometry(
    py: Python<'_>,
    image: PyReadonlyArrayDyn<'_, f32>,
    #[pyo3(from_py_with = positions)] positions: Vec<[f64; 2]>,
    aperture_radius: f64,
    annulus_inner: f64,
    annulus_outer: f64,
    saturation: Option<f64>,
    gain: Option<f64>,
    mask: Option<&Bound<'_, PyAny>>,
    saturation_mask: Option<&Bound<'_, PyAny>>,
    uncertainty: Option<PyReadonlyArrayDyn<'_, f32>>,
) -> PyResult<Vec<PyApertureMeasurement>> {
    let (width, height) = mono_shape(&image, "image")?;
    let mask = byte_plane(mask, width, height, "mask")?;
    let saturation_mask = byte_plane(saturation_mask, width, height, "saturation_mask")?;
    if let Some(array) = &uncertainty
        && mono_shape(array, "uncertainty")? != (width, height)
    {
        return Err(PyValueError::new_err(format!(
            "uncertainty must have the image's shape ({height}, {width})"
        )));
    }
    let options = AperturePhotometryOptions {
        geometry: ApertureGeometry::new(aperture_radius, annulus_inner, annulus_outer),
        saturation,
        gain,
    };
    let image_view = image
        .as_array()
        .into_dimensionality::<Ix2>()
        .expect("validated image shape");
    let uncertainty_view = uncertainty.as_ref().map(|array| {
        array
            .as_array()
            .into_dimensionality::<Ix2>()
            .expect("validated uncertainty shape")
    });
    let mask_view = mask.as_ref().map(BytePlane::view);
    let saturation_view = saturation_mask.as_ref().map(BytePlane::view);
    // Snapshot local pixels while holding the GIL, then fit owned samples.
    // No borrowed NumPy memory escapes into detached Rust execution.
    let read = |x: usize, y: usize| AperturePixel {
        value: image_view[[y, x]],
        masked: mask_view.as_ref().is_some_and(|map| map.at(x, y)),
        saturated: saturation_view.as_ref().is_some_and(|map| map.at(x, y)),
        uncertainty: uncertainty_view.as_ref().map(|map| map[[y, x]]),
    };
    // Validate even an empty request, then bound snapshot memory for catalogues.
    AperturePhotometry::sample(width, height, &[], &options, read)
        .map_err(crate::continuum::core_error)?;
    let mut measurements = Vec::with_capacity(positions.len());
    for positions in positions.chunks(64) {
        let sampled = AperturePhotometry::sample(width, height, positions, &options, read)
            .map_err(crate::continuum::core_error)?;
        measurements.extend(py.detach(move || sampled.measure()));
    }
    Ok(measurements.into_iter().map(Into::into).collect())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyApertureFlags>()?;
    m.add_class::<PyApertureMeasurement>()?;
    m.add_function(wrap_pyfunction!(aperture_photometry, m)?)?;
    Ok(())
}
