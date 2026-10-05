use crate::PyWcs;
use crate::arrays::{image_array, linear_image};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use seiza_stacking::{
    ColorCalibration, ColorCalibrationOptions, ColorFit, GaiaColorSource, SOLAR_BP_RP,
    calibrate_color as rust_calibrate_color, place_gaia_sources,
};

/// Fetch Gaia DR3 photometry within `radius_deg` of `(ra, dec)` (degrees)
/// down to G `max_mag` from the ESA Gaia archive.
///
/// Returns a list of dicts with `ra`, `dec` (J2016.0), `pmra`, `pmdec`
/// (mas/yr), `g`, `bp`, `rp` and `ruwe`; absent values are None. A wide
/// field can take several minutes; keep the result for reuse.
#[pyfunction]
#[pyo3(signature = (ra, dec, radius_deg, max_mag=15.0))]
fn gaia_photometry_cone(
    py: Python<'_>,
    ra: f64,
    dec: f64,
    radius_deg: f64,
    max_mag: f32,
) -> PyResult<Vec<Py<PyDict>>> {
    let stars = py
        .allow_threads(|| {
            let downloader = seiza_sources::SourceDownloader::new()?;
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|error| seiza_sources::Error::GaiaJobFailed(error.to_string()))?;
            runtime.block_on(downloader.gaia_photometry_cone(ra, dec, radius_deg, max_mag))
        })
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    stars
        .into_iter()
        .map(|star| {
            let row = PyDict::new(py);
            row.set_item("ra", star.ra)?;
            row.set_item("dec", star.dec)?;
            row.set_item("pmra", star.pmra)?;
            row.set_item("pmdec", star.pmdec)?;
            row.set_item("g", star.g)?;
            row.set_item("bp", star.bp)?;
            row.set_item("rp", star.rp)?;
            row.set_item("ruwe", star.ruwe)?;
            Ok(row.unbind())
        })
        .collect()
}

/// Read Gaia DR3 photometry within `radius_deg` of `(ra, dec)` (degrees)
/// down to G `max_mag` from an offline catalog (`stars-gaia-photometry.bin`,
/// from :func:`fetch_catalogs`).
///
/// Returns dicts like :func:`gaia_photometry_cone`'s, but with positions
/// already at the catalog's epoch, so `pmra` and `pmdec` are None, and the
/// colour as `bp_rp` (None when Gaia's is missing or its astrometry is
/// unreliable).
#[pyfunction]
#[pyo3(signature = (path, ra, dec, radius_deg, max_mag=15.0))]
fn gaia_photometry_catalog_cone(
    py: Python<'_>,
    path: std::path::PathBuf,
    ra: f64,
    dec: f64,
    radius_deg: f64,
    max_mag: f32,
) -> PyResult<Vec<Py<PyDict>>> {
    let stars = py
        .allow_threads(|| {
            let catalog = seiza::catalog::PhotometryCatalog::open(&path)?;
            Ok::<_, std::io::Error>(catalog.cone_search(ra, dec, radius_deg, max_mag))
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    stars
        .into_iter()
        .map(|star| {
            let row = PyDict::new(py);
            row.set_item("ra", star.ra)?;
            row.set_item("dec", star.dec)?;
            row.set_item("pmra", py.None())?;
            row.set_item("pmdec", py.None())?;
            row.set_item("g", star.g)?;
            row.set_item("bp_rp", star.bp_rp.filter(|_| star.reliable))?;
            Ok(row.unbind())
        })
        .collect()
}

/// Photometric colour calibration fitted by :func:`calibrate_color`.
#[pyclass(frozen, name = "ColorCalibration", module = "seiza")]
pub(crate) struct PyColorCalibration {
    inner: ColorCalibration,
}

fn fit_dict<'py>(py: Python<'py>, fit: &ColorFit) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("intercept", fit.intercept)?;
    dict.set_item("slope", fit.slope)?;
    dict.set_item("scatter", fit.scatter)?;
    dict.set_item("stars", fit.stars)?;
    Ok(dict)
}

#[pymethods]
impl PyColorCalibration {
    /// Multipliers for R, G, B; green's is 1.
    #[getter]
    fn gains(&self) -> (f32, f32, f32) {
        let [r, g, b] = self.inner.gains;
        (r, g, b)
    }

    /// Offsets added after the gains, neutralizing the background.
    #[getter]
    fn offsets(&self) -> (f32, f32, f32) {
        let [r, g, b] = self.inner.offsets;
        (r, g, b)
    }

    /// Each channel's sky median before calibration.
    #[getter]
    fn background(&self) -> (f32, f32, f32) {
        let [r, g, b] = self.inner.background;
        (r, g, b)
    }

    /// The line of `-2.5 log10(R / G)` against Gaia BP - RP.
    #[getter]
    fn red_fit<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        fit_dict(py, &self.inner.red_fit)
    }

    /// The line of `-2.5 log10(B / G)` against Gaia BP - RP.
    #[getter]
    fn blue_fit<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        fit_dict(py, &self.inner.blue_fit)
    }

    #[getter]
    fn white_bp_rp(&self) -> f32 {
        self.inner.white_bp_rp
    }

    #[getter]
    fn aperture_radius(&self) -> f64 {
        self.inner.aperture_radius
    }

    #[getter]
    fn aperture_correction(&self) -> (f64, f64, f64) {
        let [r, g, b] = self.inner.aperture_correction;
        (r, g, b)
    }

    #[getter]
    fn stars_offered(&self) -> usize {
        self.inner.stars_offered
    }

    #[getter]
    fn stars_measured(&self) -> usize {
        self.inner.stars_measured
    }

    /// Return a calibrated copy of a linear `(height, width, 3)` image.
    fn apply<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArrayDyn<'_, f32>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let mut image = linear_image(image)?;
        py.allow_threads(|| self.inner.apply(&mut image))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        image_array(py, &image)
    }

    fn __repr__(&self) -> String {
        let [r, g, b] = self.inner.gains;
        format!(
            "ColorCalibration(gains=({r:.4}, {g:.4}, {b:.4}), stars={}, white_bp_rp={})",
            self.inner.red_fit.stars.min(self.inner.blue_fit.stars),
            self.inner.white_bp_rp
        )
    }
}

/// Fit photometric colour calibration for a linear `(height, width, 3)`
/// image against Gaia DR3 star colours.
///
/// `wcs` is the image's astrometric solution, and `gaia` a list of dicts as
/// :func:`gaia_photometry_cone` or :func:`gaia_photometry_catalog_cone`
/// returns; a `bp_rp` value is used in place of `bp` and `rp`. `epoch` is the observation's Julian
/// year, for proper motion. The fit renders a star of Gaia BP - RP
/// `white_bp_rp` (the Sun's by default) neutral; use the result's
/// :meth:`ColorCalibration.apply` to calibrate the image.
#[pyfunction]
#[pyo3(signature = (
    image,
    wcs,
    gaia,
    *,
    epoch=None,
    white_bp_rp=SOLAR_BP_RP,
    aperture_radius=None,
    neutralize_background=true
))]
#[allow(clippy::too_many_arguments)]
fn calibrate_color(
    py: Python<'_>,
    image: PyReadonlyArrayDyn<'_, f32>,
    wcs: PyRef<'_, PyWcs>,
    gaia: Vec<Bound<'_, PyDict>>,
    epoch: Option<f64>,
    white_bp_rp: f32,
    aperture_radius: Option<f64>,
    neutralize_background: bool,
) -> PyResult<PyColorCalibration> {
    let image = linear_image(image)?;
    let number = |row: &Bound<'_, PyDict>, key: &str| -> PyResult<Option<f64>> {
        match row.get_item(key)? {
            Some(value) if !value.is_none() => Ok(Some(value.extract::<f64>()?)),
            _ => Ok(None),
        }
    };
    let sources = gaia
        .iter()
        .map(|row| {
            let required = |key: &str| {
                number(row, key)?.ok_or_else(|| {
                    PyValueError::new_err(format!("every Gaia row needs a {key} value"))
                })
            };
            let bp_rp = match row.get_item("bp_rp")? {
                Some(value) => (!value.is_none())
                    .then(|| value.extract::<f64>())
                    .transpose()?,
                None => {
                    let (bp, rp) = (number(row, "bp")?, number(row, "rp")?);
                    bp.zip(rp).map(|(bp, rp)| bp - rp)
                }
            };
            Ok(GaiaColorSource {
                ra: required("ra")?,
                dec: required("dec")?,
                pmra: number(row, "pmra")?,
                pmdec: number(row, "pmdec")?,
                g: required("g")? as f32,
                bp_rp: bp_rp.map(|value| value as f32),
                ruwe: number(row, "ruwe")?.map(|ruwe| ruwe as f32),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let wcs = wcs.wcs.clone();
    let options = ColorCalibrationOptions {
        white_bp_rp,
        aperture_radius,
        neutralize_background,
        ..ColorCalibrationOptions::default()
    };
    let inner = py
        .allow_threads(|| {
            let stars = place_gaia_sources(&wcs, image.width, image.height, epoch, &sources);
            rust_calibrate_color(&image, &stars, &options)
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyColorCalibration { inner })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyColorCalibration>()?;
    module.add_function(wrap_pyfunction!(gaia_photometry_cone, module)?)?;
    module.add_function(wrap_pyfunction!(gaia_photometry_catalog_cone, module)?)?;
    module.add_function(wrap_pyfunction!(calibrate_color, module)?)?;
    module.add("SOLAR_BP_RP", SOLAR_BP_RP)?;
    Ok(())
}
