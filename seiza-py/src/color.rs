use crate::arrays::{into_image_array, linear_image};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use seiza_stacking::{
    ColorNormalization, ColorOptions, ForaxxOptions, NarrowbandPalette,
    combine_lrgb as compose_lrgb, combine_narrowband as compose_narrowband,
    combine_rgb as compose_rgb,
};

fn color_error(error: seiza_stacking::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn options(
    normalization: &str,
    black_percentile: f32,
    white_percentile: f32,
    normalization_samples: usize,
) -> PyResult<ColorOptions> {
    let normalization = match normalization.to_ascii_lowercase().as_str() {
        "none" => ColorNormalization::None,
        "percentile" => ColorNormalization::Percentile {
            black_percentile,
            white_percentile,
            max_samples: normalization_samples,
        },
        _ => {
            return Err(PyValueError::new_err(
                "normalization must be 'none' or 'percentile'",
            ));
        }
    };
    Ok(ColorOptions { normalization })
}

#[pyfunction]
#[pyo3(signature = (red, green, blue, *, normalization="percentile", black_percentile=0.001, white_percentile=0.995, normalization_samples=1_000_000))]
#[allow(clippy::too_many_arguments)]
fn combine_rgb<'py>(
    py: Python<'py>,
    red: PyReadonlyArrayDyn<'_, f32>,
    green: PyReadonlyArrayDyn<'_, f32>,
    blue: PyReadonlyArrayDyn<'_, f32>,
    normalization: &str,
    black_percentile: f32,
    white_percentile: f32,
    normalization_samples: usize,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let red = linear_image(red)?;
    let green = linear_image(green)?;
    let blue = linear_image(blue)?;
    let options = options(
        normalization,
        black_percentile,
        white_percentile,
        normalization_samples,
    )?;
    let result = py
        .allow_threads(move || compose_rgb(&red, &green, &blue, &options))
        .map_err(color_error)?;
    into_image_array(py, result.image)
}

#[pyfunction]
#[pyo3(signature = (luminance, red, green, blue, *, luminance_weight=1.0, normalization="percentile", black_percentile=0.001, white_percentile=0.995, normalization_samples=1_000_000))]
#[allow(clippy::too_many_arguments)]
fn combine_lrgb<'py>(
    py: Python<'py>,
    luminance: PyReadonlyArrayDyn<'_, f32>,
    red: PyReadonlyArrayDyn<'_, f32>,
    green: PyReadonlyArrayDyn<'_, f32>,
    blue: PyReadonlyArrayDyn<'_, f32>,
    luminance_weight: f32,
    normalization: &str,
    black_percentile: f32,
    white_percentile: f32,
    normalization_samples: usize,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let luminance = linear_image(luminance)?;
    let red = linear_image(red)?;
    let green = linear_image(green)?;
    let blue = linear_image(blue)?;
    let options = options(
        normalization,
        black_percentile,
        white_percentile,
        normalization_samples,
    )?;
    let result = py
        .allow_threads(move || {
            compose_lrgb(&luminance, &red, &green, &blue, luminance_weight, &options)
        })
        .map_err(color_error)?;
    into_image_array(py, result.image)
}

#[pyfunction]
#[pyo3(signature = (ha, oiii, sii=None, *, palette="sho", normalization="percentile", black_percentile=0.001, white_percentile=0.995, normalization_samples=1_000_000, foraxx_target_median=0.2, foraxx_shadows_clip=-2.8))]
#[allow(clippy::too_many_arguments)]
fn combine_narrowband<'py>(
    py: Python<'py>,
    ha: PyReadonlyArrayDyn<'_, f32>,
    oiii: PyReadonlyArrayDyn<'_, f32>,
    sii: Option<PyReadonlyArrayDyn<'_, f32>>,
    palette: &str,
    normalization: &str,
    black_percentile: f32,
    white_percentile: f32,
    normalization_samples: usize,
    foraxx_target_median: f32,
    foraxx_shadows_clip: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let palette = parse_palette(palette)?;
    if palette.requires_sii() && sii.is_none() {
        return Err(PyValueError::new_err(format!(
            "{} requires an SII channel",
            palette.name()
        )));
    }
    let options = options(
        normalization,
        black_percentile,
        white_percentile,
        normalization_samples,
    )?;
    let foraxx = ForaxxOptions {
        target_median: foraxx_target_median,
        shadows_clip: foraxx_shadows_clip,
    };
    let ha = linear_image(ha)?;
    let oiii = linear_image(oiii)?;
    let sii = if palette.requires_sii() {
        sii.map(linear_image).transpose()?
    } else {
        None
    };
    let result = py
        .allow_threads(move || {
            compose_narrowband(&ha, &oiii, sii.as_ref(), palette, &options, &foraxx)
        })
        .map_err(color_error)?;
    into_image_array(py, result.image)
}

fn parse_palette(value: &str) -> PyResult<NarrowbandPalette> {
    value.parse().map_err(|_| {
        PyValueError::new_err(
            "palette must be SHO, SOH, HSO, HOS, OSH, OHS, HOO, Foraxx-SHO, or Foraxx-HOO",
        )
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(combine_rgb, module)?)?;
    module.add_function(wrap_pyfunction!(combine_lrgb, module)?)?;
    module.add_function(wrap_pyfunction!(combine_narrowband, module)?)?;
    Ok(())
}
