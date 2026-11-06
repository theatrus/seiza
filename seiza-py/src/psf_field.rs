//! Empirical PSF fields — `seiza-stars`'s `PsfField`.
//!
//! A mean PSF stamp plus principal components whose coefficients vary as
//! quadratic polynomials across the field, measured from an image's
//! isolated stars, with the wings from its saturated stars.

use crate::arrays::{float_array, float_image_view};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use seiza_stars::{
    CatalogueStar, PsfChannelFit, PsfField, PsfFieldError, PsfFieldOptions, PsfShape, PsfStar,
    SparseFieldPolicy,
};

pyo3::create_exception!(
    seiza,
    TooFewStarsError,
    crate::EngineError,
    "Too few isolated, unsaturated stars for a varying PSF field. Measure \
     again with sparse=\"global\" to accept one PSF for the whole image."
);

/// The shape of a PSF at one point, from an elliptical Moffat profile with
/// free β fitted to its stamp. Widths are in pixels; `angle` is the major
/// axis's direction in radians over `[0, π)`, from +x towards +y; `e1` and
/// `e2` are `χ·cos 2φ` and `χ·sin 2φ` with `χ = (a² − b²)/(a² + b²)`.
#[pyclass(skip_from_py_object, name = "PsfShape", module = "seiza", frozen)]
#[derive(Clone)]
pub(crate) struct PyPsfShape {
    #[pyo3(get)]
    fwhm: f64,
    #[pyo3(get)]
    fwhm_major: f64,
    #[pyo3(get)]
    fwhm_minor: f64,
    #[pyo3(get)]
    e1: f64,
    #[pyo3(get)]
    e2: f64,
    #[pyo3(get)]
    angle: f64,
    #[pyo3(get)]
    beta: f64,
}

impl From<PsfShape> for PyPsfShape {
    fn from(shape: PsfShape) -> Self {
        Self {
            fwhm: shape.fwhm,
            fwhm_major: shape.fwhm_major,
            fwhm_minor: shape.fwhm_minor,
            e1: shape.e1,
            e2: shape.e2,
            angle: shape.angle,
            beta: shape.beta,
        }
    }
}

#[pymethods]
impl PyPsfShape {
    /// The ellipticity's magnitude, `hypot(e1, e2)`.
    #[getter]
    fn ellipticity(&self) -> f64 {
        self.e1.hypot(self.e2)
    }

    fn __repr__(&self) -> String {
        format!(
            "PsfShape(fwhm={:.3}, e1={:+.4}, e2={:+.4}, beta={:.2})",
            self.fwhm, self.e1, self.e2, self.beta
        )
    }
}

/// A star a PSF field was fitted from. `role` is `"core"` for an
/// unsaturated star or `"wing"` for a saturated one.
#[pyclass(skip_from_py_object, name = "PsfStar", module = "seiza", frozen)]
#[derive(Clone)]
pub(crate) struct PyPsfStar {
    #[pyo3(get)]
    x: f64,
    #[pyo3(get)]
    y: f64,
    #[pyo3(get)]
    flux: f64,
    #[pyo3(get)]
    snr: f64,
    #[pyo3(get)]
    fwhm: f64,
    #[pyo3(get)]
    role: &'static str,
}

impl From<&PsfStar> for PyPsfStar {
    fn from(star: &PsfStar) -> Self {
        Self {
            x: star.x,
            y: star.y,
            flux: star.flux,
            snr: star.snr,
            fwhm: star.fwhm,
            role: star.role.as_str(),
        }
    }
}

/// How well one channel's PSF field fits its stars, and the `degree` of the
/// polynomials its coefficients follow (0 for a global PSF). `wing_overlap` gives
/// the radii in pixels where the core hands over to the saturated stars'
/// wings, or is None when no saturated star gave any.
#[pyclass(skip_from_py_object, name = "PsfChannelFit", module = "seiza", frozen)]
#[derive(Clone)]
pub(crate) struct PyPsfChannelFit {
    #[pyo3(get)]
    reduced_chi_squared: f64,
    #[pyo3(get)]
    relative_residual: f64,
    #[pyo3(get)]
    degree: usize,
    #[pyo3(get)]
    wing_stars: usize,
    #[pyo3(get)]
    wing_overlap: Option<(f64, f64)>,
}

impl From<&PsfChannelFit> for PyPsfChannelFit {
    fn from(fit: &PsfChannelFit) -> Self {
        Self {
            reduced_chi_squared: fit.reduced_chi_squared,
            relative_residual: fit.relative_residual,
            degree: fit.degree,
            wing_stars: fit.wing_stars,
            wing_overlap: fit.wing_overlap,
        }
    }
}

/// An empirical PSF that varies across an image, measured from its stars.
#[pyclass(name = "PsfField", module = "seiza", frozen)]
pub(crate) struct PyPsfField {
    pub(crate) field: PsfField,
}

#[pymethods]
impl PyPsfField {
    /// Measure the PSF field of a linear image, `(height, width)` or
    /// `(height, width, 3)` float32. Each channel of an RGB image gets its own
    /// field. `catalogue` lists `(x, y, magnitude)` catalogue stars at their
    /// image positions: a star then gives a stamp only when it matches one,
    /// and catalogue neighbours within `isolation_magnitudes` of it are
    /// masked out of its stamp, or disqualify it close to its core. `degree` (1 to 3) fixes the
    /// polynomials the components' coefficients follow across the field;
    /// None chooses it by cross-validation. With fewer than `minimum_stars`
    /// usable stars this raises `TooFewStarsError`, unless
    /// `sparse="global"` accepts one PSF for the whole image.
    #[staticmethod]
    #[pyo3(signature = (
        image, *, catalogue=None, stamp_size=None, components=3, minimum_stars=30,
        maximum_stars=400, minimum_snr=30.0, detection_sigma=5.0, saturation=None,
        isolation_magnitudes=5.0, sparse="refuse", degree=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn measure(
        py: Python<'_>,
        image: PyReadonlyArrayDyn<'_, f32>,
        catalogue: Option<Vec<(f64, f64, f64)>>,
        stamp_size: Option<usize>,
        components: usize,
        minimum_stars: usize,
        maximum_stars: usize,
        minimum_snr: f64,
        detection_sigma: f64,
        saturation: Option<f32>,
        isolation_magnitudes: f64,
        sparse: &str,
        degree: Option<usize>,
    ) -> PyResult<Self> {
        let image = float_image_view(&image)?;
        let sparse: SparseFieldPolicy = sparse
            .parse()
            .map_err(|error: PsfFieldError| PyValueError::new_err(error.to_string()))?;
        let options = PsfFieldOptions {
            stamp_size,
            components,
            minimum_stars,
            maximum_stars,
            minimum_snr,
            detection_sigma,
            saturation,
            isolation_magnitudes,
            sparse,
            degree,
        };
        let catalogue: Vec<CatalogueStar> = catalogue
            .unwrap_or_default()
            .into_iter()
            .map(|(x, y, magnitude)| CatalogueStar { x, y, magnitude })
            .collect();
        let field = py
            .detach(|| {
                if catalogue.is_empty() {
                    PsfField::measure(
                        image.data,
                        image.width,
                        image.height,
                        image.channels,
                        &options,
                    )
                } else {
                    PsfField::measure_with_catalogue(
                        image.data,
                        image.width,
                        image.height,
                        image.channels,
                        &catalogue,
                        &options,
                    )
                }
            })
            .map_err(|error| match error {
                PsfFieldError::TooFewStars { .. } => TooFewStarsError::new_err(error.to_string()),
                PsfFieldError::InvalidImage(_) | PsfFieldError::InvalidOptions(_) => {
                    PyValueError::new_err(error.to_string())
                }
                other => crate::EngineError::new_err(other.to_string()),
            })?;
        Ok(Self { field })
    }

    #[getter]
    fn width(&self) -> usize {
        self.field.width()
    }

    #[getter]
    fn height(&self) -> usize {
        self.field.height()
    }

    #[getter]
    fn channels(&self) -> usize {
        self.field.channels()
    }

    /// The side of every stamp in pixels; odd.
    #[getter]
    fn stamp_size(&self) -> usize {
        self.field.stamp_size()
    }

    /// Principal components beyond the mean PSF; 0 for a global PSF.
    #[getter]
    fn component_count(&self) -> usize {
        self.field.component_count()
    }

    /// Unsaturated stars the core was fitted from.
    #[getter]
    fn star_count(&self) -> usize {
        self.field.star_count()
    }

    /// Whether one PSF serves the whole image.
    #[getter]
    fn is_global(&self) -> bool {
        self.field.is_global()
    }

    /// What a caller should know about the fit, as messages.
    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.field
            .warnings()
            .iter()
            .map(|warning| warning.to_string())
            .collect()
    }

    /// Every star the field was fitted from: the core's, then the wings'.
    #[getter]
    fn stars(&self) -> Vec<PyPsfStar> {
        self.field.stars().iter().map(Into::into).collect()
    }

    /// How well each channel's field fits its stars.
    #[getter]
    fn channel_fits(&self) -> Vec<PyPsfChannelFit> {
        self.field.channel_fits().iter().map(Into::into).collect()
    }

    /// The local PSF at pixel `(x, y)`: `(size, size)` float32, or
    /// `(size, size, 3)` for RGB, each channel summing to one and centred on
    /// the middle pixel. Points outside the image take the PSF at the
    /// nearest edge.
    fn stamp_at<'py>(
        &self,
        py: Python<'py>,
        x: f64,
        y: f64,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let stamp = self.field.stamp_at(x, y);
        float_array(py, stamp.size, stamp.size, stamp.channels, stamp.data)
    }

    /// The shape of one channel's local PSF at pixel `(x, y)`, or None
    /// when the profile cannot be fitted. Raises `ValueError` for a channel
    /// out of range.
    #[pyo3(signature = (x, y, channel=0))]
    fn shape_at(&self, x: f64, y: f64, channel: usize) -> PyResult<Option<PyPsfShape>> {
        if channel >= self.field.channels() {
            return Err(PyValueError::new_err(format!(
                "channel {channel} is out of range for a {}-channel PSF field",
                self.field.channels()
            )));
        }
        Ok(self.field.shape_at(channel, x, y).map(Into::into))
    }

    fn __repr__(&self) -> String {
        format!(
            "PsfField(width={}, height={}, channels={}, stars={}, stamp_size={}, global={})",
            self.field.width(),
            self.field.height(),
            self.field.channels(),
            self.field.star_count(),
            self.field.stamp_size(),
            self.field.is_global(),
        )
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPsfShape>()?;
    module.add_class::<PyPsfStar>()?;
    module.add_class::<PyPsfChannelFit>()?;
    module.add_class::<PyPsfField>()?;
    module.add(
        "TooFewStarsError",
        module.py().get_type::<TooFewStarsError>(),
    )?;
    Ok(())
}
