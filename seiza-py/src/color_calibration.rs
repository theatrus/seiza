use crate::PyWcs;
use crate::arrays::{image_array, linear_image};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use seiza_stacking::{
    ColorCalibration, ColorCalibrationOptions, ColorCalibrationOutcome, ColorFit, GaiaColorSource,
    GaiaColors, SOLAR_BP_RP, calibrate_and_stretch as rust_calibrate_and_stretch,
    calibrate_color as rust_calibrate_color, calibrate_color_image as rust_calibrate_color_image,
    place_gaia_sources,
};
use seiza_stretch::{StretchConfig, StretchStack};

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
        .detach(|| {
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
/// from :func:`fetch_catalogs`). `path` is the catalog file or the
/// directory that holds it.
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
        .detach(|| {
            let path = seiza::data_paths::gaia_photometry(Some(&path))
                .map_err(|error| error.to_string())?
                .ok_or("no Gaia photometry catalog is installed")?;
            let catalog = seiza::catalog::PhotometryCatalog::open(&path)
                .map_err(|error| format!("failed to open {}: {error}", path.display()))?;
            Ok::<_, String>(catalog.cone_search(ra, dec, radius_deg, max_mag))
        })
        .map_err(PyValueError::new_err)?;
    stars
        .into_iter()
        .map(|star| {
            let row = PyDict::new(py);
            row.set_item("ra", star.ra)?;
            row.set_item("dec", star.dec)?;
            row.set_item("pmra", py.None())?;
            row.set_item("pmdec", py.None())?;
            row.set_item("g", star.g)?;
            row.set_item("bp_rp", star.calibration_bp_rp())?;
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

    /// Every cleanly measured star: a dict with `x`, `y`, `g`, `bp_rp`,
    /// `flux` (background-subtracted, aperture-corrected R, G, B) and `used`
    /// (whether the red and the blue fit kept it).
    #[getter]
    fn measurements<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .measurements
            .iter()
            .map(|star| {
                let row = PyDict::new(py);
                row.set_item("x", star.x)?;
                row.set_item("y", star.y)?;
                row.set_item("g", star.g)?;
                row.set_item("bp_rp", star.bp_rp)?;
                let [r, g, b] = star.flux;
                row.set_item("flux", (r, g, b))?;
                let [red, blue] = star.used;
                row.set_item("used", (red, blue))?;
                Ok(row)
            })
            .collect()
    }

    /// Return a calibrated copy of a linear `(height, width, 3)` image.
    fn apply<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArrayDyn<'_, f32>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let mut image = linear_image(image)?;
        py.detach(|| self.inner.apply(&mut image))
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

/// Gaia rows as :func:`gaia_photometry_cone` or
/// :func:`gaia_photometry_catalog_cone` returns them; a `bp_rp` value is
/// used in place of `bp` and `rp`.
fn gaia_sources(gaia: &[Bound<'_, PyDict>]) -> PyResult<Vec<GaiaColorSource>> {
    let number = |row: &Bound<'_, PyDict>, key: &str| -> PyResult<Option<f64>> {
        match row.get_item(key)? {
            Some(value) if !value.is_none() => Ok(Some(value.extract::<f64>()?)),
            _ => Ok(None),
        }
    };
    gaia.iter()
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
        .collect()
}

/// What :func:`calibrate_color_image` or :func:`calibrate_and_stretch` did:
/// `status` is ``"calibrated"``, with the :class:`ColorCalibration` applied,
/// or ``"skipped"``, with the `reason` (``"no_solution"``, ``"no_catalog"``,
/// ``"no_stars"``, ``"too_few_stars"``, ``"not_rgb"``, ``"invalid_options"``
/// or ``"fit_failed"``) and its `detail`.
#[pyclass(frozen, name = "ColorCalibrationOutcome", module = "seiza")]
pub(crate) struct PyColorCalibrationOutcome {
    inner: ColorCalibrationOutcome,
}

#[pymethods]
impl PyColorCalibrationOutcome {
    #[getter]
    fn status(&self) -> &'static str {
        match self.inner {
            ColorCalibrationOutcome::Calibrated(_) => "calibrated",
            ColorCalibrationOutcome::Skipped(_) => "skipped",
        }
    }

    /// Why the image was not calibrated, or None.
    #[getter]
    fn reason(&self) -> Option<String> {
        match &self.inner {
            ColorCalibrationOutcome::Skipped(skip) => serde_json::to_value(skip.reason)
                .ok()
                .and_then(|value| value.as_str().map(str::to_owned)),
            ColorCalibrationOutcome::Calibrated(_) => None,
        }
    }

    /// The reason in words, or None.
    #[getter]
    fn detail(&self) -> Option<String> {
        match &self.inner {
            ColorCalibrationOutcome::Skipped(skip) => Some(skip.detail.clone()),
            ColorCalibrationOutcome::Calibrated(_) => None,
        }
    }

    /// The calibration applied, or None.
    #[getter]
    fn calibration(&self) -> Option<PyColorCalibration> {
        self.inner
            .calibration()
            .map(|calibration| PyColorCalibration {
                inner: calibration.clone(),
            })
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            ColorCalibrationOutcome::Calibrated(calibration) => {
                let [r, g, b] = calibration.gains;
                format!("ColorCalibrationOutcome(calibrated, gains=({r:.4}, {g:.4}, {b:.4}))")
            }
            ColorCalibrationOutcome::Skipped(skip) => {
                format!("ColorCalibrationOutcome(skipped, {:?})", skip.detail)
            }
        }
    }
}

/// A colour-calibrated, stretched image, from :func:`calibrate_and_stretch`.
#[pyclass(frozen, name = "CalibratedStretch", module = "seiza")]
pub(crate) struct PyCalibratedStretch {
    image: Py<PyArrayDyn<f32>>,
    calibration: Py<PyColorCalibrationOutcome>,
    stretch_plans: Py<PyAny>,
    linked_stages: Vec<usize>,
}

#[pymethods]
impl PyCalibratedStretch {
    /// The display-referred `(height, width, 3)` result.
    #[getter]
    fn image(&self, py: Python<'_>) -> Py<PyArrayDyn<f32>> {
        self.image.clone_ref(py)
    }

    /// The calibration applied before the stretch, or why there was none.
    #[getter]
    fn calibration(&self, py: Python<'_>) -> Py<PyColorCalibrationOutcome> {
        self.calibration.clone_ref(py)
    }

    /// Each stretch stage's resolved plan, as dicts.
    #[getter]
    fn stretch_plans(&self, py: Python<'_>) -> Py<PyAny> {
        self.stretch_plans.clone_ref(py)
    }

    /// The stages, counted from 0, that asked to stretch each channel on
    /// its own and were linked because calibration ran.
    #[getter]
    fn linked_stages(&self) -> Vec<usize> {
        self.linked_stages.clone()
    }
}

/// Where the Gaia colours come from: the rows given, or else the offline
/// catalog at `catalog` (the file or the directory that holds it; None
/// looks wherever Seiza's catalogs are installed).
struct GaiaInput {
    sources: Option<Vec<GaiaColorSource>>,
    catalog: Option<std::path::PathBuf>,
    epoch: Option<f64>,
}

impl GaiaInput {
    fn colors(&self) -> GaiaColors<'_> {
        match &self.sources {
            Some(sources) => GaiaColors::Sources {
                sources,
                epoch: self.epoch,
            },
            None => GaiaColors::CatalogPath(self.catalog.as_deref()),
        }
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
///
/// The fits use stars whose BP - RP lies in `fit_bp_rp_range`, reject stars
/// more than `rejection_sigma` robust sigmas off the line, and fail with
/// fewer than `minimum_stars` left. `calibrator_max_g` is the faintest G
/// magnitude a calibrator may have; by default, two magnitudes brighter than
/// the faintest source given.
#[pyfunction]
#[pyo3(signature = (
    image,
    wcs,
    gaia,
    *,
    epoch=None,
    white_bp_rp=SOLAR_BP_RP,
    fit_bp_rp_range=(0.0, 1.8),
    aperture_radius=None,
    rejection_sigma=3.0,
    minimum_stars=20,
    neutralize_background=true,
    calibrator_max_g=None
))]
#[allow(clippy::too_many_arguments)]
fn calibrate_color(
    py: Python<'_>,
    image: PyReadonlyArrayDyn<'_, f32>,
    wcs: PyRef<'_, PyWcs>,
    gaia: Vec<Bound<'_, PyDict>>,
    epoch: Option<f64>,
    white_bp_rp: f32,
    fit_bp_rp_range: (f32, f32),
    aperture_radius: Option<f64>,
    rejection_sigma: f64,
    minimum_stars: usize,
    neutralize_background: bool,
    calibrator_max_g: Option<f32>,
) -> PyResult<PyColorCalibration> {
    let image = linear_image(image)?;
    let sources = gaia_sources(&gaia)?;
    let wcs = wcs.wcs.clone();
    let options = ColorCalibrationOptions {
        white_bp_rp,
        fit_bp_rp_range,
        aperture_radius,
        rejection_sigma,
        minimum_stars,
        neutralize_background,
        calibrator_max_g,
    };
    let inner = py
        .detach(|| {
            let stars = place_gaia_sources(&wcs, image.width, image.height, epoch, &sources);
            rust_calibrate_color(&image, &stars, &options)
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyColorCalibration { inner })
}

/// Colour calibrate a linear `(height, width, 3)` image against Gaia DR3
/// star colours and return the calibrated copy with the outcome.
///
/// `wcs` is the image's own solution (for an image made from a solved
/// frame, see :meth:`Wcs.transformed`); None means there is none. `gaia` is
/// a list of rows as :func:`gaia_photometry_cone` returns, with `epoch` the
/// observation's Julian year for proper motion; without it the colours come
/// from the offline catalog at `catalog`, or wherever Seiza's catalogs are
/// installed. The fit's options are :func:`calibrate_color`'s.
///
/// When calibration cannot run, the copy is the image unchanged and the
/// outcome says why; this raises only for a malformed argument.
#[pyfunction]
#[pyo3(signature = (
    image,
    wcs=None,
    *,
    gaia=None,
    catalog=None,
    epoch=None,
    white_bp_rp=SOLAR_BP_RP,
    fit_bp_rp_range=(0.0, 1.8),
    aperture_radius=None,
    rejection_sigma=3.0,
    minimum_stars=20,
    neutralize_background=true,
    calibrator_max_g=None
))]
#[allow(clippy::too_many_arguments)]
fn calibrate_color_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'_, f32>,
    wcs: Option<PyRef<'_, PyWcs>>,
    gaia: Option<Vec<Bound<'_, PyDict>>>,
    catalog: Option<std::path::PathBuf>,
    epoch: Option<f64>,
    white_bp_rp: f32,
    fit_bp_rp_range: (f32, f32),
    aperture_radius: Option<f64>,
    rejection_sigma: f64,
    minimum_stars: usize,
    neutralize_background: bool,
    calibrator_max_g: Option<f32>,
) -> PyResult<(Bound<'py, PyArrayDyn<f32>>, PyColorCalibrationOutcome)> {
    let mut image = linear_image(image)?;
    let input = GaiaInput {
        sources: gaia.as_deref().map(gaia_sources).transpose()?,
        catalog,
        epoch,
    };
    let wcs = wcs.map(|wcs| wcs.wcs.clone());
    let options = ColorCalibrationOptions {
        white_bp_rp,
        fit_bp_rp_range,
        aperture_radius,
        rejection_sigma,
        minimum_stars,
        neutralize_background,
        calibrator_max_g,
    };
    let inner = py
        .detach(|| rust_calibrate_color_image(&mut image, wcs.as_ref(), input.colors(), &options));
    Ok((
        image_array(py, &image)?,
        PyColorCalibrationOutcome { inner },
    ))
}

/// Colour calibrate a linear `(height, width, 3)` image against Gaia, as
/// :func:`calibrate_color_image` does, then display-stretch it.
///
/// `stretch` is one stretch configuration or a list of them applied in
/// order, as dicts in the form the C ABI takes: for example
/// ``{"model": {"type": "auto-mtf", "target_median": 0.25, "shadows_clip":
/// -2.8}, "color_strategy": "linked", "max_analysis_samples": 200000}``. A
/// statistics-driven stretch sees the calibrated colours, so a linked one
/// keeps them. Once calibration has run, a stage with ``"color_strategy":
/// "unlinked"`` is linked instead and listed in `linked_stages`. When
/// calibration cannot run, every stage runs as given, the image is still
/// stretched, and the result's outcome says why. With `clipped`, the map
/// `LiveStacker.clipped_fraction()` gives, the pixels a stack's clipped raw
/// samples reached come out neutral rather than tinted, as
/// `neutralize_clipped` renders them, before the stretch.
#[pyfunction]
#[pyo3(signature = (
    image,
    wcs=None,
    *,
    stretch,
    clipped=None,
    gaia=None,
    catalog=None,
    epoch=None,
    white_bp_rp=SOLAR_BP_RP,
    fit_bp_rp_range=(0.0, 1.8),
    aperture_radius=None,
    rejection_sigma=3.0,
    minimum_stars=20,
    neutralize_background=true,
    calibrator_max_g=None
))]
#[allow(clippy::too_many_arguments)]
fn calibrate_and_stretch(
    py: Python<'_>,
    image: PyReadonlyArrayDyn<'_, f32>,
    wcs: Option<PyRef<'_, PyWcs>>,
    stretch: Bound<'_, PyAny>,
    clipped: Option<PyReadonlyArrayDyn<'_, f32>>,
    gaia: Option<Vec<Bound<'_, PyDict>>>,
    catalog: Option<std::path::PathBuf>,
    epoch: Option<f64>,
    white_bp_rp: f32,
    fit_bp_rp_range: (f32, f32),
    aperture_radius: Option<f64>,
    rejection_sigma: f64,
    minimum_stars: usize,
    neutralize_background: bool,
    calibrator_max_g: Option<f32>,
) -> PyResult<PyCalibratedStretch> {
    let image = linear_image(image)?;
    let clipped = clipped.map(linear_image).transpose()?;
    // A list of stages, or one stage on its own.
    let stack = if stretch.is_instance_of::<pyo3::types::PyList>()
        || stretch.is_instance_of::<pyo3::types::PyTuple>()
    {
        crate::convert::from_python::<StretchStack>(&stretch, "stretch")?
    } else {
        StretchStack::single(crate::convert::from_python::<StretchConfig>(
            &stretch, "stretch",
        )?)
    };
    let input = GaiaInput {
        sources: gaia.as_deref().map(gaia_sources).transpose()?,
        catalog,
        epoch,
    };
    let wcs = wcs.map(|wcs| wcs.wcs.clone());
    let options = ColorCalibrationOptions {
        white_bp_rp,
        fit_bp_rp_range,
        aperture_radius,
        rejection_sigma,
        minimum_stars,
        neutralize_background,
        calibrator_max_g,
    };
    let result = py
        .detach(|| {
            rust_calibrate_and_stretch(
                image,
                wcs.as_ref(),
                input.colors(),
                &options,
                &stack,
                clipped.as_ref(),
            )
        })
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let plans = crate::convert::to_python(py, &result.stretch_plans)?;
    Ok(PyCalibratedStretch {
        image: image_array(py, &result.image)?.unbind(),
        calibration: Py::new(
            py,
            PyColorCalibrationOutcome {
                inner: result.calibration,
            },
        )?,
        stretch_plans: plans.unbind(),
        linked_stages: result.linked_stages,
    })
}

/// Render neutral the pixels of a linear `(height, width, 3)` image a
/// stack's clipped raw samples reached, and return the result: each pixel's
/// channels rise toward its brightest channel by a smooth step of its
/// clipped fraction, none at 5% and all the way from 50%, the map feathered
/// over a few pixels so a neutral core fades into its halo. Run it after
/// colour calibration. `clipped` is the `(height, width)` map
/// `LiveStacker.clipped_fraction()` gives, on the image's grid or one it is
/// a whole multiple of, such as a drizzled stack's.
#[pyfunction]
fn neutralize_clipped<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'_, f32>,
    clipped: PyReadonlyArrayDyn<'_, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let mut image = linear_image(image)?;
    let clipped = linear_image(clipped)?;
    py.detach(|| seiza_stacking::neutralize_clipped(&mut image, &clipped))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    image_array(py, &image)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyColorCalibration>()?;
    module.add_function(wrap_pyfunction!(gaia_photometry_cone, module)?)?;
    module.add_function(wrap_pyfunction!(gaia_photometry_catalog_cone, module)?)?;
    module.add_function(wrap_pyfunction!(calibrate_color, module)?)?;
    module.add_class::<PyColorCalibrationOutcome>()?;
    module.add_class::<PyCalibratedStretch>()?;
    module.add_function(wrap_pyfunction!(calibrate_color_image, module)?)?;
    module.add_function(wrap_pyfunction!(calibrate_and_stretch, module)?)?;
    module.add_function(wrap_pyfunction!(neutralize_clipped, module)?)?;
    module.add("SOLAR_BP_RP", SOLAR_BP_RP)?;
    Ok(())
}
