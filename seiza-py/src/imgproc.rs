use crate::arrays::float_array;
use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use seiza_imgproc::blur::{auto_ksize_f32, gaussian_blur_f32, gaussian_blur_u8, median_blur3_u8};
use seiza_imgproc::border::BorderMode;
use seiza_imgproc::canny::canny as canny_rs;
use seiza_imgproc::contours::{Point, contour_area as contour_area_rs, find_external_contours};
use seiza_imgproc::dtfilter::dt_filter_nc;
use seiza_imgproc::morphology::{KernelShape, MorphBorder, StructuringElement, dilate, erode};
use seiza_imgproc::threshold::{otsu_binary as otsu_binary_rs, otsu_threshold as otsu_rs};
use seiza_imgproc::wavelets::StructureRemover;

struct GrayView<'a, T> {
    data: &'a [T],
    width: usize,
    height: usize,
}

fn gray_view<'a, T: numpy::Element>(
    array: &'a PyReadonlyArrayDyn<'_, T>,
) -> PyResult<GrayView<'a, T>> {
    let [height, width] = array.shape() else {
        return Err(PyValueError::new_err(
            "image arrays must have shape (height, width)",
        ));
    };
    let (height, width) = (*height, *width);
    let data = array
        .as_slice()
        .map_err(|_| PyValueError::new_err("image arrays must be C-contiguous"))?;
    Ok(GrayView {
        data,
        width,
        height,
    })
}

fn gray_array<T: numpy::Element>(
    py: Python<'_>,
    width: usize,
    height: usize,
    values: Vec<T>,
) -> PyResult<Bound<'_, PyArrayDyn<T>>> {
    let array = ArrayD::from_shape_vec(IxDyn(&[height, width]), values)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py))
}

fn parse_border(border: &str) -> PyResult<BorderMode> {
    match border.to_ascii_lowercase().as_str() {
        "replicate" => Ok(BorderMode::Replicate),
        "reflect" => Ok(BorderMode::Reflect),
        "reflect101" => Ok(BorderMode::Reflect101),
        _ => Err(PyValueError::new_err(
            "border must be replicate, reflect, or reflect101",
        )),
    }
}

fn parse_morph_border(border: &str) -> PyResult<MorphBorder> {
    match border.to_ascii_lowercase().as_str() {
        "ignore" => Ok(MorphBorder::Ignore),
        "replicate" => Ok(MorphBorder::Replicate),
        "reflect" => Ok(MorphBorder::Reflect),
        "reflect101" => Ok(MorphBorder::Reflect101),
        _ => Err(PyValueError::new_err(
            "border must be ignore, replicate, reflect, or reflect101",
        )),
    }
}

fn parse_shape(shape: &str) -> PyResult<KernelShape> {
    match shape.to_ascii_lowercase().as_str() {
        "rect" => Ok(KernelShape::Rect),
        "ellipse" => Ok(KernelShape::Ellipse),
        "cross" => Ok(KernelShape::Cross),
        _ => Err(PyValueError::new_err(
            "shape must be rect, ellipse, or cross",
        )),
    }
}

fn require_odd(ksize: usize) -> PyResult<()> {
    if ksize == 0 || ksize % 2 == 1 {
        Ok(())
    } else {
        Err(PyValueError::new_err("ksize must be odd (or 0 for auto)"))
    }
}

/// Gaussian blur with OpenCV's exact kernels and arithmetic.
///
/// Accepts a uint8 array (OpenCV's fixed-point `CV_8U` path) or a float32
/// array (the `CV_32F` path). `ksize=0` derives the kernel size from sigma
/// using OpenCV's per-depth rule. Returns an array of the input dtype.
#[pyfunction]
#[pyo3(signature = (image, sigma, *, ksize=0, border="reflect101"))]
fn gaussian_blur<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    sigma: f64,
    ksize: usize,
    border: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let border = parse_border(border)?;
    require_odd(ksize)?;
    if let Ok(array) = image.extract::<PyReadonlyArrayDyn<'_, u8>>() {
        let view = gray_view(&array)?;
        let (data, width, height) = (view.data, view.width, view.height);
        let out =
            py.allow_threads(move || gaussian_blur_u8(data, width, height, ksize, sigma, border));
        return Ok(gray_array(py, width, height, out)?.into_any());
    }
    if let Ok(array) = image.extract::<PyReadonlyArrayDyn<'_, f32>>() {
        let view = gray_view(&array)?;
        let (data, width, height) = (view.data, view.width, view.height);
        let ksize = if ksize == 0 {
            auto_ksize_f32(sigma)
        } else {
            ksize
        };
        let out =
            py.allow_threads(move || gaussian_blur_f32(data, width, height, ksize, sigma, border));
        return Ok(gray_array(py, width, height, out)?.into_any());
    }
    Err(PyValueError::new_err(
        "image must be a uint8 or float32 array of shape (height, width)",
    ))
}

/// 3x3 median blur on a uint8 image (OpenCV `medianBlur` with ksize 3).
#[pyfunction]
fn median_blur3<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'_, u8>,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let view = gray_view(&image)?;
    let (data, width, height) = (view.data, view.width, view.height);
    let out = py.allow_threads(move || median_blur3_u8(data, width, height));
    gray_array(py, width, height, out)
}

/// Canny edge detection (Sobel aperture 3, L1 gradient), as in `cv::Canny`.
/// Returns a uint8 edge map with edges at 255.
#[pyfunction]
#[pyo3(signature = (image, low, high))]
fn canny<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'_, u8>,
    low: i32,
    high: i32,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let view = gray_view(&image)?;
    let (data, width, height) = (view.data, view.width, view.height);
    let out = py.allow_threads(move || canny_rs(data, width, height, low, high));
    gray_array(py, width, height, out)
}

/// Otsu threshold of a uint8 image, as in `cv::threshold` with `THRESH_OTSU`.
#[pyfunction]
fn otsu_threshold(py: Python<'_>, image: PyReadonlyArrayDyn<'_, u8>) -> PyResult<u8> {
    let view = gray_view(&image)?;
    let data = view.data;
    Ok(py.allow_threads(move || otsu_rs(data)))
}

/// Binarize a uint8 image with the Otsu threshold: values above the
/// threshold become 255, the rest 0.
#[pyfunction]
fn otsu_binary<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'_, u8>,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let view = gray_view(&image)?;
    let (data, width, height) = (view.data, view.width, view.height);
    let out = py.allow_threads(move || otsu_binary_rs(data, width, height));
    gray_array(py, width, height, out)
}

fn morph<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'_, u8>,
    shape: &str,
    ksize: usize,
    border: &str,
    op: fn(&[u8], usize, usize, &StructuringElement, MorphBorder) -> Vec<u8>,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let shape = parse_shape(shape)?;
    let border = parse_morph_border(border)?;
    if ksize == 0 || ksize.is_multiple_of(2) {
        return Err(PyValueError::new_err("ksize must be odd"));
    }
    let view = gray_view(&image)?;
    let (data, width, height) = (view.data, view.width, view.height);
    let out = py.allow_threads(move || {
        let element = StructuringElement::new(shape, ksize);
        op(data, width, height, &element, border)
    });
    gray_array(py, width, height, out)
}

/// Binary erosion with an OpenCV structuring element. `border="ignore"`
/// reproduces OpenCV's default constant border for erosion.
#[pyfunction]
#[pyo3(name = "erode", signature = (image, *, shape="ellipse", ksize=3, border="ignore"))]
fn erode_py<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'_, u8>,
    shape: &str,
    ksize: usize,
    border: &str,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    morph(py, image, shape, ksize, border, erode)
}

/// Binary dilation with an OpenCV structuring element. `border="ignore"`
/// reproduces OpenCV's default constant border for dilation.
#[pyfunction]
#[pyo3(name = "dilate", signature = (image, *, shape="ellipse", ksize=3, border="ignore"))]
fn dilate_py<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'_, u8>,
    shape: &str,
    ksize: usize,
    border: &str,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    morph(py, image, shape, ksize, border, dilate)
}

/// External contours of the non-zero regions of a uint8 image, as in
/// `cv::findContours` with `RETR_EXTERNAL` and `CHAIN_APPROX_SIMPLE`.
/// Returns a list of int32 arrays of shape (n, 2) in (x, y) order.
#[pyfunction]
fn find_contours<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'_, u8>,
) -> PyResult<Vec<Bound<'py, PyArrayDyn<i32>>>> {
    let view = gray_view(&image)?;
    let (data, width, height) = (view.data, view.width, view.height);
    let contours = py.allow_threads(move || find_external_contours(data, width, height));
    contours
        .into_iter()
        .map(|contour| {
            let n = contour.len();
            let flat: Vec<i32> = contour.into_iter().flat_map(|(x, y)| [x, y]).collect();
            let array = ArrayD::from_shape_vec(IxDyn(&[n, 2]), flat)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            Ok(array.into_pyarray(py))
        })
        .collect()
}

fn contour_points(contour: &PyReadonlyArrayDyn<'_, i32>) -> PyResult<Vec<Point>> {
    let [_, 2] = contour.shape() else {
        return Err(PyValueError::new_err(
            "contour must be an int32 array of shape (n, 2)",
        ));
    };
    let flat = contour
        .as_slice()
        .map_err(|_| PyValueError::new_err("contour arrays must be C-contiguous"))?;
    Ok(flat.chunks_exact(2).map(|p| (p[0], p[1])).collect())
}

/// Polygon area of a contour, as `cv::contourArea` computes it (Green's
/// theorem over pixel centers, not a pixel count).
#[pyfunction]
fn contour_area(contour: PyReadonlyArrayDyn<'_, i32>) -> PyResult<f64> {
    Ok(contour_area_rs(&contour_points(&contour)?))
}

/// Edge-aware smoothing with the domain transform filter (normalized
/// convolution), as in `cv::ximgproc::dtFilter` with `DTF_NC`.
#[pyfunction]
#[pyo3(signature = (guide, src, sigma_spatial, sigma_color, *, num_iters=3))]
fn dt_filter<'py>(
    py: Python<'py>,
    guide: PyReadonlyArrayDyn<'_, f32>,
    src: PyReadonlyArrayDyn<'_, f32>,
    sigma_spatial: f64,
    sigma_color: f64,
    num_iters: usize,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    if num_iters == 0 {
        return Err(PyValueError::new_err("num_iters must be at least 1"));
    }
    let guide = gray_view(&guide)?;
    let src = gray_view(&src)?;
    if guide.width != src.width || guide.height != src.height {
        return Err(PyValueError::new_err(
            "guide and src must have the same shape",
        ));
    }
    let (guide_data, src_data) = (guide.data, src.data);
    let (width, height) = (src.width, src.height);
    let out = py.allow_threads(move || {
        dt_filter_nc(
            guide_data,
            src_data,
            width,
            height,
            sigma_spatial,
            sigma_color,
            num_iters,
        )
    });
    float_array(py, width, height, 1, out)
}

/// Remove large-scale structure from a float64 image, returning the
/// small-scale residual (stars plus noise).
///
/// `method="filtered"` uses a Gaussian pyramid for the first three layers
/// and edge-aware domain transform filtering for deeper layers;
/// `method="atrous"` uses the à trous B3-spline wavelet transform.
#[pyfunction]
#[pyo3(signature = (image, *, layers=4, method="filtered"))]
fn remove_structures<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'_, f64>,
    layers: usize,
    method: &str,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let atrous = match method.to_ascii_lowercase().as_str() {
        "filtered" => false,
        "atrous" => true,
        _ => return Err(PyValueError::new_err("method must be filtered or atrous")),
    };
    let view = gray_view(&image)?;
    let (data, width, height) = (view.data, view.width, view.height);
    let out = py.allow_threads(move || {
        let remover = StructureRemover::new(layers);
        if atrous {
            remover.remove_structures_atrous(data, width, height)
        } else {
            remover.remove_structures_filtered(data, width, height)
        }
    });
    gray_array(py, width, height, out)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(gaussian_blur, module)?)?;
    module.add_function(wrap_pyfunction!(median_blur3, module)?)?;
    module.add_function(wrap_pyfunction!(canny, module)?)?;
    module.add_function(wrap_pyfunction!(otsu_threshold, module)?)?;
    module.add_function(wrap_pyfunction!(otsu_binary, module)?)?;
    module.add_function(wrap_pyfunction!(erode_py, module)?)?;
    module.add_function(wrap_pyfunction!(dilate_py, module)?)?;
    module.add_function(wrap_pyfunction!(find_contours, module)?)?;
    module.add_function(wrap_pyfunction!(contour_area, module)?)?;
    module.add_function(wrap_pyfunction!(dt_filter, module)?)?;
    module.add_function(wrap_pyfunction!(remove_structures, module)?)?;
    Ok(())
}
