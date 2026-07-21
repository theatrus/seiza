use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use seiza_stacking::LinearImage;

#[derive(Clone, Copy)]
pub(crate) struct FloatImageView<'a> {
    pub(crate) data: &'a [f32],
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) channels: usize,
}

pub(crate) fn float_image_view<'a>(
    array: &'a PyReadonlyArrayDyn<'_, f32>,
) -> PyResult<FloatImageView<'a>> {
    let (height, width, channels) = match array.shape() {
        [height, width] => (*height, *width, 1),
        [height, width, 3] => (*height, *width, 3),
        _ => {
            return Err(PyValueError::new_err(
                "image arrays must have shape (height, width) or (height, width, 3)",
            ));
        }
    };
    let data = array
        .as_slice()
        .map_err(|_| PyValueError::new_err("image arrays must be C-contiguous"))?;
    Ok(FloatImageView {
        data,
        width,
        height,
        channels,
    })
}

pub(crate) fn linear_image(array: PyReadonlyArrayDyn<'_, f32>) -> PyResult<LinearImage> {
    let image = float_image_view(&array)?;
    LinearImage::new(
        image.width,
        image.height,
        image.channels,
        image.data.to_vec(),
    )
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn array_shape(width: usize, height: usize, channels: usize) -> Vec<usize> {
    if channels == 1 {
        vec![height, width]
    } else {
        vec![height, width, channels]
    }
}

pub(crate) fn image_array<'py>(
    py: Python<'py>,
    image: &LinearImage,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    float_array(
        py,
        image.width,
        image.height,
        image.channels,
        image.data.clone(),
    )
}

pub(crate) fn into_image_array(
    py: Python<'_>,
    image: LinearImage,
) -> PyResult<Bound<'_, PyArrayDyn<f32>>> {
    float_array(py, image.width, image.height, image.channels, image.data)
}

pub(crate) fn float_array<'py>(
    py: Python<'py>,
    width: usize,
    height: usize,
    channels: usize,
    values: Vec<f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let shape = array_shape(width, height, channels);
    let array = ArrayD::from_shape_vec(IxDyn(&shape), values)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py))
}

pub(crate) fn u32_array<'py>(
    py: Python<'py>,
    values: &[u32],
    width: usize,
    height: usize,
    channels: usize,
) -> PyResult<Bound<'py, PyArrayDyn<u32>>> {
    let shape = array_shape(width, height, channels);
    let array = ArrayD::from_shape_vec(IxDyn(&shape), values.to_vec())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py))
}
