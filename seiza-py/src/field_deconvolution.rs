//! Deconvolution towards a round target with a field-varying PSF —
//! `seiza-deconvolution`'s `RemainingKernel`, `deconvolve_field` and
//! `reblur_consistency`.

use crate::arrays::{float_array, float_image_view};
use crate::psf_field::PyPsfField;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use seiza_deconvolution::{
    ChannelConsistency, DEFAULT_NODE_SPACING, DeconvolutionError, FieldDeconvolutionConfig,
    RemainingKernel, TargetPsf, deconvolve_field as restore_field, deconvolve_field_masked,
    reblur_consistency as check_reblur,
};

fn engine_error(error: DeconvolutionError) -> PyErr {
    crate::EngineError::new_err(error.to_string())
}

/// The remaining kernel `R`, with `P = T ∗ R`, between a measured PSF field
/// `P` and a round Gaussian target `T`, on a grid of nodes
/// at most `node_spacing` pixels apart. Give `target_fwhm` for one FWHM in pixels
/// (at least 1.5), `target_ratio` for a fraction of the local FWHM, or
/// neither for the field's best round PSF everywhere.
#[pyclass(name = "RemainingKernel", module = "seiza", frozen)]
pub(crate) struct PyRemainingKernel {
    kernel: RemainingKernel,
}

#[pymethods]
impl PyRemainingKernel {
    #[new]
    #[pyo3(signature = (field, *, target_fwhm=None, target_ratio=None, node_spacing=DEFAULT_NODE_SPACING))]
    fn new(
        py: Python<'_>,
        field: PyRef<'_, PyPsfField>,
        target_fwhm: Option<f32>,
        target_ratio: Option<f32>,
        node_spacing: usize,
    ) -> PyResult<Self> {
        let target = match (target_fwhm, target_ratio) {
            (Some(fwhm), None) => TargetPsf::Fwhm(fwhm),
            (None, Some(ratio)) => TargetPsf::Ratio(ratio),
            (None, None) => TargetPsf::BestRound,
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "give target_fwhm or target_ratio, not both",
                ));
            }
        };
        let field = &field.field;
        let kernel = py
            .detach(|| RemainingKernel::new(field, target, node_spacing))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { kernel })
    }

    #[getter]
    fn width(&self) -> usize {
        self.kernel.width()
    }

    #[getter]
    fn height(&self) -> usize {
        self.kernel.height()
    }

    #[getter]
    fn channels(&self) -> usize {
        self.kernel.channels()
    }

    /// The side of each node's kernel in pixels.
    #[getter]
    fn kernel_size(&self) -> usize {
        self.kernel.kernel_size()
    }

    /// The most pixels between nodes.
    #[getter]
    fn node_spacing(&self) -> usize {
        self.kernel.node_spacing()
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.kernel.node_count()
    }

    /// The narrowest and widest local PSF FWHM over the nodes.
    #[pyo3(signature = (channel=0))]
    fn psf_fwhm_range(&self, channel: usize) -> PyResult<(f32, f32)> {
        self.kernel
            .psf_fwhm_range(channel)
            .ok_or_else(|| PyValueError::new_err(format!("channel {channel} is out of range")))
    }

    /// The narrowest and widest target FWHM over the nodes.
    #[pyo3(signature = (channel=0))]
    fn target_fwhm_range(&self, channel: usize) -> PyResult<(f32, f32)> {
        self.kernel
            .target_fwhm_range(channel)
            .ok_or_else(|| PyValueError::new_err(format!("channel {channel} is out of range")))
    }

    /// The kernel at pixel `(x, y)`, `(size, size)` float32 summing to one.
    #[pyo3(signature = (x, y, channel=0))]
    fn kernel_at<'py>(
        &self,
        py: Python<'py>,
        x: f64,
        y: f64,
        channel: usize,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let values = self
            .kernel
            .kernel_at(channel, x, y)
            .ok_or_else(|| PyValueError::new_err(format!("channel {channel} is out of range")))?;
        let size = self.kernel.kernel_size();
        float_array(py, size, size, 1, values)
    }

    /// The image blurred by the remaining kernel, each source pixel by its
    /// own: what an image at the target looks like with the measured PSF.
    fn blur<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArrayDyn<'_, f32>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let image = float_image_view(&image)?;
        self.check_shape(image.width, image.height, image.channels)?;
        let blurred = py
            .detach(|| self.kernel.blur(image.data))
            .map_err(engine_error)?;
        float_array(py, image.width, image.height, image.channels, blurred)
    }

    fn __repr__(&self) -> String {
        format!(
            "RemainingKernel(width={}, height={}, channels={}, nodes={}, node_spacing={})",
            self.kernel.width(),
            self.kernel.height(),
            self.kernel.channels(),
            self.kernel.node_count(),
            self.kernel.node_spacing(),
        )
    }
}

impl PyRemainingKernel {
    fn check_shape(&self, width: usize, height: usize, channels: usize) -> PyResult<()> {
        if (width, height, channels)
            != (
                self.kernel.width(),
                self.kernel.height(),
                self.kernel.channels(),
            )
        {
            return Err(PyValueError::new_err(
                "image shape differs from the remaining kernel's",
            ));
        }
        Ok(())
    }
}

/// Deconvolve a linear image towards `kernel`'s target with damped
/// Richardson–Lucy, the remaining kernel as its forward model.
///
/// With `masked=True`, non-finite samples are a fixed missing-data mask
/// that stays in the output; otherwise they raise `EngineError`. The input
/// array is read in place while the GIL is released.
#[pyfunction]
#[pyo3(signature = (image, kernel, *, iterations=4, amount=0.35, noise_fraction=0.001, max_correction=2.0, masked=false))]
#[allow(clippy::too_many_arguments)]
fn deconvolve_field<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'_, f32>,
    kernel: PyRef<'_, PyRemainingKernel>,
    iterations: usize,
    amount: f32,
    noise_fraction: f32,
    max_correction: f32,
    masked: bool,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let image = float_image_view(&image)?;
    kernel.check_shape(image.width, image.height, image.channels)?;
    let config = FieldDeconvolutionConfig {
        iterations,
        amount,
        noise_fraction,
        max_correction,
    };
    let restore = if masked {
        deconvolve_field_masked
    } else {
        restore_field
    };
    let kernel = &kernel.kernel;
    let restored = py
        .detach(|| restore(image.data, kernel, &config))
        .map_err(engine_error)?;
    float_array(py, image.width, image.height, image.channels, restored.data)
}

/// How well one channel of an estimate, blurred by the remaining kernel,
/// gives back its input.
#[pyclass(
    skip_from_py_object,
    name = "ChannelConsistency",
    module = "seiza",
    frozen
)]
#[derive(Clone)]
pub(crate) struct PyChannelConsistency {
    #[pyo3(get)]
    reduced_chi_squared: f64,
    #[pyo3(get)]
    samples: usize,
    #[pyo3(get)]
    band_power: Vec<f64>,
    #[pyo3(get)]
    spectral_flatness: Option<f64>,
}

impl From<&ChannelConsistency> for PyChannelConsistency {
    fn from(channel: &ChannelConsistency) -> Self {
        Self {
            reduced_chi_squared: channel.reduced_chi_squared,
            samples: channel.samples,
            band_power: channel.band_power.clone(),
            spectral_flatness: channel.spectral_flatness,
        }
    }
}

#[pymethods]
impl PyChannelConsistency {
    fn __repr__(&self) -> String {
        format!(
            "ChannelConsistency(reduced_chi_squared={:.3}, spectral_flatness={:?})",
            self.reduced_chi_squared, self.spectral_flatness
        )
    }
}

/// The reblur-consistency check of an estimate at the target PSF.
#[pyclass(
    skip_from_py_object,
    name = "ReblurConsistency",
    module = "seiza",
    frozen
)]
pub(crate) struct PyReblurConsistency {
    #[pyo3(get)]
    channels: Vec<PyChannelConsistency>,
}

/// Check `output`, an estimate at `kernel`'s target, against `input`, the
/// image it came from, with `noise` the input's per-sample noise as a
/// standard deviation (non-finite or zero leaves a sample out). Blurred by
/// the remaining kernel, the estimate must give the input back to within
/// its noise: a reduced χ² near 1 and a flat residual spectrum.
#[pyfunction]
fn reblur_consistency(
    py: Python<'_>,
    output: PyReadonlyArrayDyn<'_, f32>,
    input: PyReadonlyArrayDyn<'_, f32>,
    noise: PyReadonlyArrayDyn<'_, f32>,
    kernel: PyRef<'_, PyRemainingKernel>,
) -> PyResult<PyReblurConsistency> {
    let (output, input, noise) = (
        float_image_view(&output)?,
        float_image_view(&input)?,
        float_image_view(&noise)?,
    );
    for view in [&output, &input, &noise] {
        kernel.check_shape(view.width, view.height, view.channels)?;
    }
    let kernel = &kernel.kernel;
    let consistency = py
        .detach(|| check_reblur(output.data, input.data, noise.data, kernel))
        .map_err(engine_error)?;
    Ok(PyReblurConsistency {
        channels: consistency.channels.iter().map(Into::into).collect(),
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRemainingKernel>()?;
    module.add_class::<PyChannelConsistency>()?;
    module.add_class::<PyReblurConsistency>()?;
    module.add_function(wrap_pyfunction!(deconvolve_field, module)?)?;
    module.add_function(wrap_pyfunction!(reblur_consistency, module)?)?;
    Ok(())
}
