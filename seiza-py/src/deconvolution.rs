use crate::arrays::{float_array, float_image_view};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use seiza_deconvolution::{DeconvolutionConfig, deconvolve as restore};

/// Apply conservative damped Richardson-Lucy restoration to a linear image.
///
/// The input array is read in place while the GIL is released; do not mutate
/// it from another thread until the call returns.
#[pyfunction]
#[pyo3(signature = (image, *, psf_fwhm, iterations=4, amount=0.35, noise_fraction=0.001, max_correction=2.0))]
fn deconvolve<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'_, f32>,
    psf_fwhm: f32,
    iterations: usize,
    amount: f32,
    noise_fraction: f32,
    max_correction: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let image = float_image_view(&image)?;
    let config = DeconvolutionConfig {
        psf_fwhm_pixels: psf_fwhm,
        iterations,
        amount,
        noise_fraction,
        max_correction,
    };
    let restored = py
        .allow_threads(|| {
            restore(
                image.data,
                image.width,
                image.height,
                image.channels,
                &config,
            )
        })
        .map_err(|error| crate::EngineError::new_err(error.to_string()))?;
    float_array(py, image.width, image.height, image.channels, restored.data)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(deconvolve, module)?)?;
    Ok(())
}
