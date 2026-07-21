use crate::arrays::{into_image_array, linear_image};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use seiza_stacking::LinearImage;
use seiza_stretch::{ColorStrategy, GhsParams, StretchConfig, StretchModel, StretchParams};

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyfunction]
#[pyo3(signature = (image, *, model="percentile-asinh", color_strategy="linked", max_analysis_samples=200_000, black=0.0, white=1.0, strength=10.0, black_percentile=0.01, white_percentile=0.995, shadows=0.0, midtone=0.5, highlights=1.0, stretch_factor=1.0, local_intensity=0.0, symmetry_point=0.0, protect_shadows=0.0, protect_highlights=1.0, target_median=0.2, shadows_clip=-2.8))]
#[allow(clippy::too_many_arguments)]
fn stretch<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'_, f32>,
    model: &str,
    color_strategy: &str,
    max_analysis_samples: usize,
    black: f64,
    white: f64,
    strength: f64,
    black_percentile: f64,
    white_percentile: f64,
    shadows: f64,
    midtone: f64,
    highlights: f64,
    stretch_factor: f64,
    local_intensity: f64,
    symmetry_point: f64,
    protect_shadows: f64,
    protect_highlights: f64,
    target_median: f64,
    shadows_clip: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let model = match model.to_ascii_lowercase().as_str() {
        "identity" => StretchModel::Identity,
        "linear" => StretchModel::Linear { black, white },
        "asinh" => StretchModel::Asinh {
            black,
            white,
            strength,
        },
        "percentile-asinh" => StretchModel::PercentileAsinh {
            black_percentile,
            white_percentile,
            strength,
        },
        "mtf" => StretchModel::Mtf {
            shadows,
            midtone,
            highlights,
        },
        "ghs" => StretchModel::Ghs(GhsParams {
            stretch_factor,
            local_intensity,
            symmetry_point,
            protect_shadows,
            protect_highlights,
            black,
            white,
        }),
        "auto-mtf" => StretchModel::AutoMtf(StretchParams {
            target_median,
            shadows_clip,
        }),
        _ => {
            return Err(PyValueError::new_err(
                "model must be identity, linear, asinh, percentile-asinh, mtf, ghs, or auto-mtf",
            ));
        }
    };
    let color_strategy = match color_strategy.to_ascii_lowercase().as_str() {
        "linked" => ColorStrategy::Linked,
        "unlinked" => ColorStrategy::Unlinked,
        "luminance-preserving" => ColorStrategy::LuminancePreserving,
        _ => {
            return Err(PyValueError::new_err(
                "color_strategy must be linked, unlinked, or luminance-preserving",
            ));
        }
    };
    let image = linear_image(image)?;
    let (width, height, channels) = (image.width, image.height, image.channels);
    let config = StretchConfig {
        model,
        color_strategy,
        max_analysis_samples,
    };
    let data = py
        .allow_threads(move || {
            let plan = if config.model.requires_analysis() {
                let analysis = config.analyze(&image.data, channels)?;
                config.resolve(&analysis)?
            } else {
                config.resolve_explicit(channels)?
            };
            plan.apply_f32(&image.data, channels)
        })
        .map_err(value_error)?;
    let output = LinearImage::new(width, height, channels, data).map_err(value_error)?;
    into_image_array(py, output)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(stretch, module)?)?;
    Ok(())
}
