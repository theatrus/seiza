//! Catalogue-guided star photometry, removal and rendering —
//! `seiza-stars`'s `StarFit`.

use crate::arrays::{float_array, float_image_view};
use crate::psf_field::PyPsfField;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use seiza_stars::{
    CatalogueStar, FittedStar, StarFit, StarFitError, StarFitOptions, StarPositions,
};

fn star_fit_error(error: StarFitError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// One catalogue star's fit: where it was fitted, and per channel its
/// `flux`, `flux_error` and `background` (NaN where nothing constrained
/// it).
#[pyclass(skip_from_py_object, name = "FittedStar", module = "seiza", frozen)]
#[derive(Clone)]
pub(crate) struct PyFittedStar {
    #[pyo3(get)]
    x: f64,
    #[pyo3(get)]
    y: f64,
    #[pyo3(get)]
    magnitude: f64,
    #[pyo3(get)]
    flux: Vec<f64>,
    #[pyo3(get)]
    flux_error: Vec<f64>,
    #[pyo3(get)]
    background: Vec<f64>,
    #[pyo3(get)]
    reduced_chi_squared: f64,
    #[pyo3(get)]
    saturated: bool,
}

impl From<&FittedStar> for PyFittedStar {
    fn from(star: &FittedStar) -> Self {
        Self {
            x: star.x,
            y: star.y,
            magnitude: star.magnitude,
            flux: star.flux.clone(),
            flux_error: star.flux_error.clone(),
            background: star.background.clone(),
            reduced_chi_squared: star.reduced_chi_squared,
            saturated: star.saturated,
        }
    }
}

#[pymethods]
impl PyFittedStar {
    fn __repr__(&self) -> String {
        format!(
            "FittedStar(x={:.2}, y={:.2}, flux={:?}, saturated={})",
            self.x, self.y, self.flux, self.saturated
        )
    }
}

/// Catalogue stars fitted with a PSF field, ready to remove and render.
#[pyclass(name = "StarFit", module = "seiza", frozen)]
pub(crate) struct PyStarFit {
    fit: StarFit,
}

#[pymethods]
impl PyStarFit {
    /// Fit the flux of each catalogue star, `(x, y, magnitude)` at its image
    /// position, in a linear image of `field`'s shape, with the field's local
    /// PSF: stars whose fit windows overlap jointly, each group with its own
    /// background plane. `noise` is the per-pixel σ (non-finite or zero
    /// leaves a sample out). Samples at or above `saturation` are left out,
    /// so a saturated star's flux comes from its wings. `positions="refined"`
    /// lets each star move by up to `max_shift` pixels to where the image
    /// puts it. The call releases the GIL.
    #[staticmethod]
    #[pyo3(signature = (image, field, stars, *, noise=None, fit_radius=None, saturation=None, positions="fixed", max_shift=1.0))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        py: Python<'_>,
        image: PyReadonlyArrayDyn<'_, f32>,
        field: PyRef<'_, PyPsfField>,
        stars: Vec<(f64, f64, f64)>,
        noise: Option<PyReadonlyArrayDyn<'_, f32>>,
        fit_radius: Option<f64>,
        saturation: Option<f32>,
        positions: &str,
        max_shift: f64,
    ) -> PyResult<Self> {
        let image = float_image_view(&image)?;
        let noise_view = match &noise {
            Some(array) => Some(float_image_view(array)?),
            None => None,
        };
        let positions: StarPositions = positions.parse().map_err(star_fit_error)?;
        let options = StarFitOptions {
            fit_radius,
            saturation,
            positions,
            max_shift,
        };
        let stars: Vec<CatalogueStar> = stars
            .into_iter()
            .map(|(x, y, magnitude)| CatalogueStar { x, y, magnitude })
            .collect();
        let field = &field.field;
        let fit = py
            .detach(|| {
                StarFit::fit(
                    image.data,
                    field,
                    &stars,
                    noise_view.map(|view| view.data),
                    &options,
                )
            })
            .map_err(star_fit_error)?;
        Ok(Self { fit })
    }

    #[getter]
    fn width(&self) -> usize {
        self.fit.width()
    }

    #[getter]
    fn height(&self) -> usize {
        self.fit.height()
    }

    #[getter]
    fn channels(&self) -> usize {
        self.fit.channels()
    }

    /// The fitted stars, in the order given.
    #[getter]
    fn stars(&self) -> Vec<PyFittedStar> {
        self.fit.stars().iter().map(Into::into).collect()
    }

    /// The fitted stars through the measured PSF, on black.
    fn model<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let model = py.detach(|| self.fit.model());
        float_array(
            py,
            self.fit.width(),
            self.fit.height(),
            self.fit.channels(),
            model,
        )
    }

    /// The image with the fitted stars removed.
    fn subtract<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArrayDyn<'_, f32>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let image = float_image_view(&image)?;
        let removed = py
            .detach(|| self.fit.subtract(image.data))
            .map_err(star_fit_error)?;
        float_array(py, image.width, image.height, image.channels, removed)
    }

    /// The fitted stars through a round Gaussian of `target_fwhm` pixels,
    /// on black.
    fn render<'py>(
        &self,
        py: Python<'py>,
        target_fwhm: f64,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let rendered = py
            .detach(|| self.fit.render(target_fwhm))
            .map_err(star_fit_error)?;
        float_array(
            py,
            self.fit.width(),
            self.fit.height(),
            self.fit.channels(),
            rendered,
        )
    }

    /// The classical star sharpener: the image with the fitted stars
    /// removed and rendered again through a round Gaussian of
    /// `target_fwhm` pixels.
    fn sharpen<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArrayDyn<'_, f32>,
        target_fwhm: f64,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let image = float_image_view(&image)?;
        let sharpened = py
            .detach(|| self.fit.sharpen(image.data, target_fwhm))
            .map_err(star_fit_error)?;
        float_array(py, image.width, image.height, image.channels, sharpened)
    }

    fn __repr__(&self) -> String {
        format!(
            "StarFit(width={}, height={}, channels={}, stars={})",
            self.fit.width(),
            self.fit.height(),
            self.fit.channels(),
            self.fit.stars().len()
        )
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyFittedStar>()?;
    module.add_class::<PyStarFit>()?;
    Ok(())
}
