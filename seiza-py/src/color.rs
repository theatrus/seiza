use crate::arrays::{into_image_array, linear_image};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList};
use seiza_fits::HeaderValue;
use seiza_stacking::{
    ChannelSamples, ColorComposition, ColorCrop, ColorNormalization, ColorOptions, CropReport,
    ForaxxOptions, NarrowbandMatrix, NarrowbandMix, NarrowbandPalette, ReferenceRegion,
    combine_lrgb as compose_lrgb, combine_narrowband as compose_narrowband,
    combine_narrowband_matrix as compose_narrowband_matrix, combine_rgb as compose_rgb,
    combine_super_lrgb as compose_super_lrgb, combine_super_rgb as compose_super_rgb,
    crop_report as report_crop, write_color_fits_f32,
};
use std::path::PathBuf;

fn color_error(error: seiza_stacking::Error) -> PyErr {
    crate::EngineError::new_err(error.to_string())
}

fn parse_crop(crop: &str) -> PyResult<ColorCrop> {
    crop.parse()
        .map_err(|_| PyValueError::new_err("crop must be 'none', 'bounds', or 'inscribed'"))
}

fn options(
    normalization: &str,
    black_percentile: f32,
    white_percentile: f32,
    normalization_samples: usize,
    crop: &str,
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
    Ok(ColorOptions {
        normalization,
        crop: parse_crop(crop)?,
        ..ColorOptions::default()
    })
}

/// Where to write a composition, and the FITS cards of the reference
/// header whose WCS the output keeps.
struct ColorOutput {
    path: PathBuf,
    header: Vec<(String, HeaderValue)>,
}

impl ColorOutput {
    fn new(output: Option<PathBuf>, header: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Self>> {
        match (output, header) {
            (None, Some(_)) => Err(PyValueError::new_err("header only applies with output")),
            (None, None) => Ok(None),
            (Some(path), header) => Ok(Some(Self {
                path,
                header: header.map(header_cards).transpose()?.unwrap_or_default(),
            })),
        }
    }
}

/// A FITS header (an astropy `Header` or a dict) as cards. Values that are
/// not a bool, int, float or string are left out.
fn header_cards(header: &Bound<'_, PyAny>) -> PyResult<Vec<(String, HeaderValue)>> {
    let mut cards = Vec::new();
    for item in header.call_method0("items")?.try_iter()? {
        let (key, value): (String, Bound<'_, PyAny>) = item?.extract()?;
        let value = if let Ok(value) = value.cast::<PyBool>() {
            HeaderValue::Logical(value.is_true())
        } else if let Ok(value) = value.extract::<i64>() {
            HeaderValue::Integer(value)
        } else if let Ok(value) = value.extract::<f64>() {
            HeaderValue::Float(value)
        } else if let Ok(value) = value.extract::<String>() {
            HeaderValue::String(value)
        } else {
            continue;
        };
        cards.push((key.to_ascii_uppercase(), value));
    }
    Ok(cards)
}

/// Compose with the GIL released, write the result when asked, and return
/// its image as an array.
fn finish_composition<'py>(
    py: Python<'py>,
    output: Option<ColorOutput>,
    label: &str,
    compose: impl FnOnce() -> seiza_stacking::Result<ColorComposition> + Send,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let result = py
        .detach(move || {
            let composition = compose()?;
            if let Some(output) = &output {
                write_color_fits_f32(&output.path, &composition, &output.header, label)?;
            }
            Ok(composition)
        })
        .map_err(color_error)?;
    into_image_array(py, result.image)
}

/// Combine mono red, green, and blue stacks into one RGB image.
///
/// `luminance_mode="native"` keeps the composed channels as they are;
/// `"super"` scales the triplet to a synthetic luminance of `R + G + B`,
/// which may exceed one. With `output`, the result is also written there
/// (XISF for a `.xisf` name, FITS otherwise), keeping the WCS of `header`,
/// the red stack's FITS header.
#[pyfunction]
#[pyo3(signature = (red, green, blue, *, luminance_mode="native", normalization="percentile", black_percentile=0.001, white_percentile=0.995, normalization_samples=1_000_000, crop="none", output=None, header=None))]
#[allow(clippy::too_many_arguments)]
fn combine_rgb<'py>(
    py: Python<'py>,
    red: PyReadonlyArrayDyn<'_, f32>,
    green: PyReadonlyArrayDyn<'_, f32>,
    blue: PyReadonlyArrayDyn<'_, f32>,
    luminance_mode: &str,
    normalization: &str,
    black_percentile: f32,
    white_percentile: f32,
    normalization_samples: usize,
    crop: &str,
    output: Option<PathBuf>,
    header: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let output = ColorOutput::new(output, header)?;
    let super_luminance = match luminance_mode.to_ascii_lowercase().as_str() {
        "native" => false,
        "super" => true,
        _ => {
            return Err(PyValueError::new_err(
                "luminance_mode must be 'native' or 'super'",
            ));
        }
    };
    let red = linear_image(red)?;
    let green = linear_image(green)?;
    let blue = linear_image(blue)?;
    let options = options(
        normalization,
        black_percentile,
        white_percentile,
        normalization_samples,
        crop,
    )?;
    let label = if super_luminance { "SUPER-RGB" } else { "RGB" };
    finish_composition(py, output, label, move || {
        if super_luminance {
            compose_super_rgb(&red, &green, &blue, &options)
        } else {
            compose_rgb(&red, &green, &blue, &options)
        }
    })
}

/// Combine a luminance stack with RGB channels into an LRGB image.
///
/// `luminance_mode="replace"` swaps in a weighted L as the output luminance;
/// `"super"` targets the additive `L + R + G + B`, which may exceed one.
/// `output` and `header` (the luminance stack's) are those of `combine_rgb`.
#[pyfunction]
#[pyo3(signature = (luminance, red, green, blue, *, luminance_weight=1.0, luminance_mode="replace", normalization="percentile", black_percentile=0.001, white_percentile=0.995, normalization_samples=1_000_000, crop="none", output=None, header=None))]
#[allow(clippy::too_many_arguments)]
fn combine_lrgb<'py>(
    py: Python<'py>,
    luminance: PyReadonlyArrayDyn<'_, f32>,
    red: PyReadonlyArrayDyn<'_, f32>,
    green: PyReadonlyArrayDyn<'_, f32>,
    blue: PyReadonlyArrayDyn<'_, f32>,
    luminance_weight: f32,
    luminance_mode: &str,
    normalization: &str,
    black_percentile: f32,
    white_percentile: f32,
    normalization_samples: usize,
    crop: &str,
    output: Option<PathBuf>,
    header: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let output = ColorOutput::new(output, header)?;
    let super_luminance = match luminance_mode.to_ascii_lowercase().as_str() {
        "replace" => false,
        "super" => {
            if luminance_weight != 1.0 {
                return Err(PyValueError::new_err(
                    "luminance_weight only applies when luminance_mode='replace'",
                ));
            }
            true
        }
        _ => {
            return Err(PyValueError::new_err(
                "luminance_mode must be 'replace' or 'super'",
            ));
        }
    };
    let luminance = linear_image(luminance)?;
    let red = linear_image(red)?;
    let green = linear_image(green)?;
    let blue = linear_image(blue)?;
    let options = options(
        normalization,
        black_percentile,
        white_percentile,
        normalization_samples,
        crop,
    )?;
    let label = if super_luminance {
        "SUPER-LRGB"
    } else {
        "LRGB"
    };
    finish_composition(py, output, label, move || {
        if super_luminance {
            compose_super_lrgb(&luminance, &red, &green, &blue, &options)
        } else {
            compose_lrgb(&luminance, &red, &green, &blue, luminance_weight, &options)
        }
    })
}

/// Map narrowband stacks (Ha/OIII, optionally SII) onto an RGB palette.
///
/// `output` and `header` (the H-alpha stack's) are those of `combine_rgb`.
#[pyfunction]
#[pyo3(signature = (ha, oiii, sii=None, *, palette="sho", normalization="percentile", black_percentile=0.001, white_percentile=0.995, normalization_samples=1_000_000, foraxx_target_median=0.2, foraxx_shadows_clip=-2.8, crop="none", output=None, header=None))]
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
    crop: &str,
    output: Option<PathBuf>,
    header: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let output = ColorOutput::new(output, header)?;
    let palette = parse_palette(palette)?;
    if palette.requires_sii() && sii.is_none() {
        return Err(PyValueError::new_err(format!(
            "{} requires an SII channel",
            palette.name()
        )));
    }
    if !palette.requires_sii() && sii.is_some() {
        return Err(PyValueError::new_err(format!(
            "{} does not use an SII channel",
            palette.name()
        )));
    }
    let options = options(
        normalization,
        black_percentile,
        white_percentile,
        normalization_samples,
        crop,
    )?;
    let foraxx = ForaxxOptions {
        target_median: foraxx_target_median,
        shadows_clip: foraxx_shadows_clip,
    };
    let ha = linear_image(ha)?;
    let oiii = linear_image(oiii)?;
    let sii = sii.map(linear_image).transpose()?;
    finish_composition(py, output, palette.name(), move || {
        compose_narrowband(&ha, &oiii, sii.as_ref(), palette, &options, &foraxx)
    })
}

/// Mix narrowband stacks (Ha/OIII, optionally SII) into RGB through your
/// own linear matrix.
///
/// `matrix` maps each output channel (`"red"`, `"green"`, `"blue"`) to its
/// filter weights (`"sii"`, `"ha"`, `"oiii"`); a weight left out is zero.
/// A matrix that weights SII needs `sii`. The result is linear light.
/// `output` and `header` (the H-alpha stack's) are those of `combine_rgb`.
#[pyfunction]
#[pyo3(signature = (ha, oiii, sii=None, *, matrix, normalization="percentile", black_percentile=0.001, white_percentile=0.995, normalization_samples=1_000_000, crop="none", output=None, header=None))]
#[allow(clippy::too_many_arguments)]
fn combine_narrowband_matrix<'py>(
    py: Python<'py>,
    ha: PyReadonlyArrayDyn<'_, f32>,
    oiii: PyReadonlyArrayDyn<'_, f32>,
    sii: Option<PyReadonlyArrayDyn<'_, f32>>,
    matrix: &Bound<'_, PyDict>,
    normalization: &str,
    black_percentile: f32,
    white_percentile: f32,
    normalization_samples: usize,
    crop: &str,
    output: Option<PathBuf>,
    header: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let output = ColorOutput::new(output, header)?;
    let matrix = parse_matrix(matrix)?;
    let options = options(
        normalization,
        black_percentile,
        white_percentile,
        normalization_samples,
        crop,
    )?;
    let ha = linear_image(ha)?;
    let oiii = linear_image(oiii)?;
    let sii = sii.map(linear_image).transpose()?;
    finish_composition(py, output, NarrowbandMatrix::LABEL, move || {
        compose_narrowband_matrix(&ha, &oiii, sii.as_ref(), matrix, &options)
    })
}

fn parse_matrix(matrix: &Bound<'_, PyDict>) -> PyResult<NarrowbandMatrix> {
    let mut mixes = [NarrowbandMix::default(); 3];
    for (channel, weights) in matrix.iter() {
        let channel = channel.extract::<String>()?;
        let mix = match channel.as_str() {
            "red" => &mut mixes[0],
            "green" => &mut mixes[1],
            "blue" => &mut mixes[2],
            _ => {
                return Err(PyValueError::new_err(format!(
                    "matrix channels are 'red', 'green' and 'blue', not {channel:?}"
                )));
            }
        };
        let weights = weights.cast::<PyDict>().map_err(|_| {
            PyValueError::new_err(format!(
                "matrix[{channel:?}] must be a dict of filter weights"
            ))
        })?;
        for (filter, weight) in weights.iter() {
            let filter = filter.extract::<String>()?;
            let weight = weight.extract::<f32>()?;
            match filter.as_str() {
                "sii" => mix.sii = weight,
                "ha" => mix.ha = weight,
                "oiii" => mix.oiii = weight,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "matrix filters are 'sii', 'ha' and 'oiii', not {filter:?}"
                    )));
                }
            }
        }
    }
    let [red, green, blue] = mixes;
    Ok(NarrowbandMatrix { red, green, blue })
}

/// Measure what each aligned mono channel covers, and the region a crop keeps.
///
/// `channels` maps a name to a `float32` array; every array must share one
/// grid. The returned dict carries the kept `region` as `(x, y, width,
/// height)` on that grid, the fraction of it retained, and one entry per
/// channel including whether the channel sits far enough from the others to
/// look like a pointing error.
#[pyfunction]
#[pyo3(signature = (channels, *, crop="inscribed"))]
fn crop_report<'py>(
    py: Python<'py>,
    channels: &Bound<'_, PyDict>,
    crop: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let crop = parse_crop(crop)?;
    let mut names = Vec::with_capacity(channels.len());
    let mut images = Vec::with_capacity(channels.len());
    for (name, array) in channels.iter() {
        names.push(name.extract::<String>()?);
        images.push(linear_image(
            array.extract::<PyReadonlyArrayDyn<'_, f32>>()?,
        )?);
    }
    let named = names
        .iter()
        .zip(&images)
        .map(|(name, image)| ChannelSamples::new(name, image))
        .collect::<Vec<_>>();
    let report = py
        .detach(|| report_crop(&named, crop))
        .map_err(color_error)?;
    report_dict(py, &report)
}

fn report_dict<'py>(py: Python<'py>, report: &CropReport) -> PyResult<Bound<'py, PyDict>> {
    let channels = PyList::empty(py);
    for channel in &report.channels {
        let entry = PyDict::new(py);
        entry.set_item("name", &channel.name)?;
        entry.set_item("region", region_tuple(channel.region))?;
        entry.set_item("covered_pixels", channel.covered_pixels)?;
        entry.set_item(
            "center_offset",
            (channel.center_offset_x, channel.center_offset_y),
        )?;
        entry.set_item("center_offset_pixels", channel.center_offset_pixels())?;
        entry.set_item("off_center", channel.off_center)?;
        channels.append(entry)?;
    }
    let dict = PyDict::new(py);
    dict.set_item("grid", (report.grid_width, report.grid_height))?;
    dict.set_item("region", region_tuple(report.region))?;
    dict.set_item("retained_fraction", report.retained_fraction())?;
    dict.set_item("channels", channels)?;
    Ok(dict)
}

fn region_tuple(region: ReferenceRegion) -> (usize, usize, usize, usize) {
    (region.x, region.y, region.width, region.height)
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
    module.add_function(wrap_pyfunction!(combine_narrowband_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(crop_report, module)?)?;
    Ok(())
}
