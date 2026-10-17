use crate::arrays::{float_array, float_image_view};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use seiza_stretch::{
    ColorStrategy, GhsParams, MedianAnchorParams, StretchConfig, StretchModel, StretchParams,
};

/// Stretch a linear image with the selected model and color strategy.
///
/// The input array is read in place while the GIL is released; do not mutate
/// it from another thread until the call returns.
#[pyfunction]
#[pyo3(signature = (image, *, model="percentile-asinh", color_strategy="linked", max_analysis_samples=200_000, black=0.0, white=1.0, strength=10.0, black_percentile=0.01, white_percentile=0.995, shadows=0.0, midtone=0.5, highlights=1.0, stretch_factor=1.0, local_intensity=0.0, symmetry_point=0.0, protect_shadows=0.0, protect_highlights=1.0, target_median=None, shadows_clip=-2.8, black_point_sigma=Some(2.7), curves_boost=0.0, hdr_amount=0.0, hdr_knee=0.75, normalize=false))]
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
    target_median: Option<f64>,
    shadows_clip: f64,
    black_point_sigma: Option<f64>,
    curves_boost: f64,
    hdr_amount: f64,
    hdr_knee: f64,
    normalize: bool,
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
            target_median: target_median.unwrap_or(StretchParams::default().target_median),
            shadows_clip,
        }),
        "median-anchor" => StretchModel::MedianAnchor(MedianAnchorParams {
            target_median: target_median.unwrap_or(MedianAnchorParams::default().target_median),
            black_point_sigma,
            curves_boost,
            hdr_amount,
            hdr_knee,
            normalize,
        }),
        _ => {
            return Err(PyValueError::new_err(
                "model must be identity, linear, asinh, percentile-asinh, mtf, ghs, auto-mtf, or median-anchor",
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
    let image = float_image_view(&image)?;
    let config = StretchConfig {
        model,
        color_strategy,
        max_analysis_samples,
    };
    let data = py
        .detach(move || {
            let plan = config.resolve_for(image.data, image.channels)?;
            plan.apply_f32(image.data, image.channels)
        })
        .map_err(|error| crate::EngineError::new_err(error.to_string()))?;
    float_array(py, image.width, image.height, image.channels, data)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(stretch, module)?)?;
    Ok(())
}
