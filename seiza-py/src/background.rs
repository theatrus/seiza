use crate::arrays::{float_array, float_image_view};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use seiza_background::{
    BackgroundConfig, BackgroundFit, CorrectionMode, ModelConfig, SampleStatus,
    fit_background_masked,
};

type PyBackgroundSample = (usize, usize, Vec<f32>, f32, f32, &'static str);

#[pyclass(frozen, name = "BackgroundModel", module = "seiza")]
struct PyBackgroundModel {
    fit: BackgroundFit,
}

#[pymethods]
impl PyBackgroundModel {
    #[getter]
    fn width(&self) -> usize {
        self.fit.width
    }

    #[getter]
    fn height(&self) -> usize {
        self.fit.height
    }

    #[getter]
    fn channels(&self) -> usize {
        self.fit.channels
    }

    #[getter]
    fn reference(&self) -> Vec<f64> {
        self.fit.reference.clone()
    }

    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let diagnostics = PyDict::new(py);
        diagnostics.set_item("candidate_samples", self.fit.diagnostics.candidate_samples)?;
        diagnostics.set_item("accepted_samples", self.fit.diagnostics.accepted_samples)?;
        diagnostics.set_item("rejected_noise", self.fit.diagnostics.rejected_noise)?;
        diagnostics.set_item("rejected_residual", self.fit.diagnostics.rejected_residual)?;
        diagnostics.set_item(
            "rejection_iterations",
            self.fit.diagnostics.rejection_iterations,
        )?;
        diagnostics.set_item("sample_radius", self.fit.diagnostics.sample_radius)?;
        Ok(diagnostics)
    }

    /// `(x, y, values, dispersion, weight, status)` for model diagnostics.
    fn samples(&self) -> Vec<PyBackgroundSample> {
        self.fit
            .samples
            .iter()
            .map(|sample| {
                (
                    sample.x,
                    sample.y,
                    sample.values.clone(),
                    sample.dispersion,
                    sample.weight,
                    match sample.status {
                        SampleStatus::Accepted => "accepted",
                        SampleStatus::RejectedNoise => "rejected_noise",
                        SampleStatus::RejectedResidual => "rejected_residual",
                    },
                )
            })
            .collect()
    }

    /// Render the fitted background. This is the only operation that allocates
    /// a full-size model image.
    fn render<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let values = py
            .allow_threads(|| self.fit.render_model())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        float_array(
            py,
            self.fit.width,
            self.fit.height,
            self.fit.channels,
            values,
        )
    }

    /// Correct an image with the fitted model while preserving NaNs.
    #[pyo3(signature = (image, *, mode="subtract"))]
    fn correct<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArrayDyn<'_, f32>,
        mode: &str,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let image = float_image_view(&image)?;
        if (image.width, image.height, image.channels)
            != (self.fit.width, self.fit.height, self.fit.channels)
        {
            return Err(PyValueError::new_err(
                "image shape differs from the fitted background model",
            ));
        }
        let mode = correction_mode(mode)?;
        let corrected = py
            .allow_threads(|| self.fit.correct(image.data, mode))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        float_array(py, image.width, image.height, image.channels, corrected)
    }

    fn __repr__(&self) -> String {
        format!(
            "BackgroundModel(width={}, height={}, channels={}, accepted_samples={})",
            self.fit.width,
            self.fit.height,
            self.fit.channels,
            self.fit.diagnostics.accepted_samples,
        )
    }
}

/// Fit a deterministic robust polynomial background to a linear image.
#[pyfunction]
#[pyo3(signature = (image, *, mask=None, degree=2, ridge=1.0e-8, samples_per_axis=12, sample_radius=None, search_steps=4, sample_rejection_sigma=3.5, fit_rejection_sigma=3.0, fit_rejection_iterations=3, border_fraction=0.03))]
#[allow(clippy::too_many_arguments)]
fn fit_background(
    py: Python<'_>,
    image: PyReadonlyArrayDyn<'_, f32>,
    mask: Option<PyReadonlyArrayDyn<'_, bool>>,
    degree: u8,
    ridge: f64,
    samples_per_axis: usize,
    sample_radius: Option<usize>,
    search_steps: usize,
    sample_rejection_sigma: f64,
    fit_rejection_sigma: f64,
    fit_rejection_iterations: usize,
    border_fraction: f64,
) -> PyResult<PyBackgroundModel> {
    let image = float_image_view(&image)?;
    let mask = match &mask {
        Some(mask) => {
            if mask.shape() != [image.height, image.width] {
                return Err(PyValueError::new_err(
                    "background mask must have shape (height, width)",
                ));
            }
            Some(
                mask.as_slice()
                    .map_err(|_| PyValueError::new_err("background mask must be C-contiguous"))?,
            )
        }
        None => None,
    };
    let config = BackgroundConfig {
        model: ModelConfig::Polynomial { degree, ridge },
        samples_per_axis,
        sample_radius,
        search_steps,
        sample_rejection_sigma,
        fit_rejection_sigma,
        fit_rejection_iterations,
        border_fraction,
    };
    let fit = py
        .allow_threads(|| {
            fit_background_masked(
                image.data,
                image.width,
                image.height,
                image.channels,
                mask,
                &config,
            )
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyBackgroundModel { fit })
}

fn correction_mode(mode: &str) -> PyResult<CorrectionMode> {
    match mode.to_ascii_lowercase().as_str() {
        "subtract" => Ok(CorrectionMode::Subtract),
        "divide" => Ok(CorrectionMode::Divide),
        _ => Err(PyValueError::new_err(
            "background correction mode must be subtract or divide",
        )),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyBackgroundModel>()?;
    module.add_function(wrap_pyfunction!(fit_background, module)?)?;
    Ok(())
}
