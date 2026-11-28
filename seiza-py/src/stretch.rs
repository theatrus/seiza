use crate::arrays::{float_array, float_image_view};
use crate::convert::{from_python, to_python};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use seiza_stretch::{
    ColorStrategy, GhsParams, ImageShape, LayeredParams, MedianAnchorParams, StretchConfig,
    StretchModel, StretchParams, StretchPlan,
};

/// The stretch the keywords of [`stretch`] and [`resolve_stretch`] ask for.
#[allow(clippy::too_many_arguments)]
fn stretch_config(
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
    target_background: f64,
    aggressiveness: f64,
    dynamic_range_compression: f64,
    scale_separation: u32,
    contrast_recovery: f64,
    saturation_amount: f64,
    saturation_boost: f64,
    lightness_mask: bool,
) -> PyResult<StretchConfig> {
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
        "ht" => StretchModel::Ht {
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
        "layered" => StretchModel::Layered(LayeredParams {
            target_background,
            aggressiveness,
            dynamic_range_compression,
            scale_separation,
            contrast_recovery,
            saturation_amount,
            saturation_boost,
            lightness_mask,
            // The C ABI does not take a sky floor either; both keep none.
            ..LayeredParams::default()
        }),
        _ => {
            return Err(PyValueError::new_err(
                "model must be identity, linear, asinh, percentile-asinh, mtf, ht, ghs, auto-mtf, median-anchor, or layered",
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
    Ok(StretchConfig {
        model,
        color_strategy,
        max_analysis_samples,
    })
}

/// Stretch a linear image with the selected model and color strategy.
///
/// ``shadows``, ``midtone`` and ``highlights`` set two models. ``"mtf"``
/// shifts samples by the clipping points and does not rescale them, as
/// N.I.N.A. does. ``"ht"`` clips to them and rescales that range to 0 to 1
/// before the curve, as PixInsight's HistogramTransformation does, and reads
/// the curve from a 24-bit table.
///
/// The input array is read in place while the GIL is released; do not mutate
/// it from another thread until the call returns.
#[pyfunction]
#[pyo3(signature = (image, *, model="percentile-asinh", color_strategy="linked", max_analysis_samples=200_000, black=0.0, white=1.0, strength=10.0, black_percentile=0.01, white_percentile=0.995, shadows=0.0, midtone=0.5, highlights=1.0, stretch_factor=1.0, local_intensity=0.0, symmetry_point=0.0, protect_shadows=0.0, protect_highlights=1.0, target_median=None, shadows_clip=-2.8, black_point_sigma=Some(2.7), curves_boost=0.0, hdr_amount=0.0, hdr_knee=0.75, normalize=false, target_background=0.15, aggressiveness=0.5, dynamic_range_compression=0.0, scale_separation=256, contrast_recovery=0.0, saturation_amount=0.0, saturation_boost=0.5, lightness_mask=true))]
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
    target_background: f64,
    aggressiveness: f64,
    dynamic_range_compression: f64,
    scale_separation: u32,
    contrast_recovery: f64,
    saturation_amount: f64,
    saturation_boost: f64,
    lightness_mask: bool,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let config = stretch_config(
        model,
        color_strategy,
        max_analysis_samples,
        black,
        white,
        strength,
        black_percentile,
        white_percentile,
        shadows,
        midtone,
        highlights,
        stretch_factor,
        local_intensity,
        symmetry_point,
        protect_shadows,
        protect_highlights,
        target_median,
        shadows_clip,
        black_point_sigma,
        curves_boost,
        hdr_amount,
        hdr_knee,
        normalize,
        target_background,
        aggressiveness,
        dynamic_range_compression,
        scale_separation,
        contrast_recovery,
        saturation_amount,
        saturation_boost,
        lightness_mask,
    )?;
    let image = float_image_view(&image)?;
    let data = py
        .detach(move || {
            let plan = config.resolve_for(image.data, image.channels)?;
            let shape = ImageShape::new(image.width, image.height, image.channels);
            plan.apply_f32_image(image.data, shape)
        })
        .map_err(|error| crate::EngineError::new_err(error.to_string()))?;
    float_array(py, image.width, image.height, image.channels, data)
}

/// A stretch with its curves resolved, from :func:`resolve_stretch`. It
/// depends on no image, so every image it is applied to takes the same curve.
#[pyclass(name = "ResolvedStretch", module = "seiza", frozen)]
pub(crate) struct PyResolvedStretch {
    plan: StretchPlan,
}

#[pymethods]
impl PyResolvedStretch {
    /// The plan a dict from :meth:`to_dict`, or one of
    /// ``calibrate_and_stretch``'s ``stretch_plans``, describes. Raises
    /// ``ValueError`` when it is not a plan.
    #[staticmethod]
    fn from_dict(plan: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            plan: from_python(plan, "stretch plan")?,
        })
    }

    /// The plan as a dict: ``channels``, ``color_strategy`` and ``curves``,
    /// each curve a dict whose ``type`` says which it is. A plan stored this
    /// way gives the same samples when read back.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.plan)
    }

    /// Channels of the images it applies to.
    #[getter]
    fn channels(&self) -> usize {
        self.plan.channels()
    }

    /// Stretch a linear image with this plan. The input array is read in
    /// place while the GIL is released; do not mutate it from another thread
    /// until the call returns.
    fn apply<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArrayDyn<'_, f32>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let image = float_image_view(&image)?;
        let shape = ImageShape::new(image.width, image.height, image.channels);
        let data = py
            .detach(|| self.plan.apply_f32_image(image.data, shape))
            .map_err(|error| crate::EngineError::new_err(error.to_string()))?;
        float_array(py, image.width, image.height, image.channels, data)
    }

    fn __repr__(&self) -> String {
        format!(
            "ResolvedStretch(channels={}, curves={})",
            self.plan.channels(),
            self.plan.curves().len()
        )
    }
}

/// Resolve a stretch on a reference image without stretching it, so that the
/// result's ``apply`` gives any image the same curve. Automatic models
/// otherwise take their curve from each image, which hides the differences
/// between them. The keywords are :func:`stretch`'s.
#[pyfunction]
#[pyo3(signature = (image, *, model="percentile-asinh", color_strategy="linked", max_analysis_samples=200_000, black=0.0, white=1.0, strength=10.0, black_percentile=0.01, white_percentile=0.995, shadows=0.0, midtone=0.5, highlights=1.0, stretch_factor=1.0, local_intensity=0.0, symmetry_point=0.0, protect_shadows=0.0, protect_highlights=1.0, target_median=None, shadows_clip=-2.8, black_point_sigma=Some(2.7), curves_boost=0.0, hdr_amount=0.0, hdr_knee=0.75, normalize=false, target_background=0.15, aggressiveness=0.5, dynamic_range_compression=0.0, scale_separation=256, contrast_recovery=0.0, saturation_amount=0.0, saturation_boost=0.5, lightness_mask=true))]
#[allow(clippy::too_many_arguments)]
fn resolve_stretch<'py>(
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
    target_background: f64,
    aggressiveness: f64,
    dynamic_range_compression: f64,
    scale_separation: u32,
    contrast_recovery: f64,
    saturation_amount: f64,
    saturation_boost: f64,
    lightness_mask: bool,
) -> PyResult<PyResolvedStretch> {
    let config = stretch_config(
        model,
        color_strategy,
        max_analysis_samples,
        black,
        white,
        strength,
        black_percentile,
        white_percentile,
        shadows,
        midtone,
        highlights,
        stretch_factor,
        local_intensity,
        symmetry_point,
        protect_shadows,
        protect_highlights,
        target_median,
        shadows_clip,
        black_point_sigma,
        curves_boost,
        hdr_amount,
        hdr_knee,
        normalize,
        target_background,
        aggressiveness,
        dynamic_range_compression,
        scale_separation,
        contrast_recovery,
        saturation_amount,
        saturation_boost,
        lightness_mask,
    )?;
    let image = float_image_view(&image)?;
    let plan = py
        .detach(move || config.resolve_for(image.data, image.channels))
        .map_err(|error| crate::EngineError::new_err(error.to_string()))?;
    Ok(PyResolvedStretch { plan })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyResolvedStretch>()?;
    module.add_function(wrap_pyfunction!(stretch, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_stretch, module)?)?;
    Ok(())
}
