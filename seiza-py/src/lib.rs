//! Python bindings for Seiza astrometry, satellite prediction, calibration,
//! restoration, and image stacking. The Python module is named `seiza`.

mod arrays;
mod background;
mod calibration;
mod color;
mod color_calibration;
mod constellations;
mod continuum;
mod deconvolution;
mod emission;
mod field_objects;
mod flat_groups;
mod flat_response;
mod imgproc;
mod parallax;
mod rc_astro;
mod satellites;
mod screening;
mod stacking;
mod stars;
mod stretch;

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyFileNotFoundError, PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use seiza::catalog::{CatalogStar, MemoryCatalog, StarCatalog as StarCatalogTrait};
use seiza::detect::{DetectConfig, DetectedStar, detect_stars, detect_stars_luma_f32};
use seiza::wcs::Wcs as SeizaWcs;
use std::path::PathBuf;

pyo3::create_exception!(
    seiza,
    SolveError,
    PyRuntimeError,
    "The field could not be solved (too few stars, no catalog match, or no \
     verified hypothesis)."
);

pyo3::create_exception!(
    seiza,
    EngineError,
    PyRuntimeError,
    "Stretching, color composition, background fitting, or deconvolution \
     failed during computation."
);

/// A detected or externally supplied star in 0-indexed pixel coordinates.
#[pyclass(from_py_object, frozen, name = "Star", module = "seiza")]
#[derive(Clone)]
struct Star {
    #[pyo3(get)]
    x: f64,
    #[pyo3(get)]
    y: f64,
    #[pyo3(get)]
    flux: f64,
    #[pyo3(get)]
    peak: f32,
    #[pyo3(get)]
    area: u32,
}

#[pymethods]
impl Star {
    #[new]
    #[pyo3(signature = (x, y, flux, peak=None, area=1))]
    fn new(x: f64, y: f64, flux: f64, peak: Option<f32>, area: u32) -> Self {
        Self {
            x,
            y,
            flux,
            peak: peak.unwrap_or(flux as f32),
            area,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Star(x={:.2}, y={:.2}, flux={:.1})",
            self.x, self.y, self.flux
        )
    }
}

impl From<&Star> for DetectedStar {
    fn from(star: &Star) -> Self {
        Self {
            x: star.x,
            y: star.y,
            flux: star.flux,
            peak: star.peak,
            area: star.area,
        }
    }
}

impl From<DetectedStar> for Star {
    fn from(star: DetectedStar) -> Self {
        Self {
            x: star.x,
            y: star.y,
            flux: star.flux,
            peak: star.peak,
            area: star.area,
        }
    }
}

enum CatalogBackend {
    /// A star tile file, and the proper motions beside it when it has them;
    /// boxed, as the tile file's index dwarfs an in-memory catalog's handle.
    Tiles(
        Box<seiza::catalog::TileCatalog>,
        Option<Box<seiza::catalog::ProperMotions>>,
    ),
    Memory(MemoryCatalog),
}

impl CatalogBackend {
    fn as_trait(&self) -> &(dyn StarCatalogTrait + Sync) {
        match self {
            Self::Tiles(catalog, _) => catalog.as_ref(),
            Self::Memory(catalog) => catalog,
        }
    }

    /// The catalog a solve searches: moved to `epoch` (a Julian year) when
    /// the star file has proper motions and an epoch is given.
    fn at_epoch(&self, epoch: Option<f64>) -> SearchedCatalog<'_> {
        match self {
            Self::Tiles(catalog, motions) => SearchedCatalog::Dated(
                seiza::catalog::DatedCatalog::new(catalog, motions.as_deref(), epoch),
            ),
            Self::Memory(catalog) => SearchedCatalog::Memory(catalog),
        }
    }
}

/// [`CatalogBackend::at_epoch`]'s catalog.
enum SearchedCatalog<'a> {
    Dated(seiza::catalog::DatedCatalog<'a>),
    Memory(&'a MemoryCatalog),
}

impl StarCatalogTrait for SearchedCatalog<'_> {
    fn cone_search(&self, ra: f64, dec: f64, radius_deg: f64, limit: usize) -> Vec<CatalogStar> {
        match self {
            Self::Dated(catalog) => catalog.cone_search(ra, dec, radius_deg, limit),
            Self::Memory(catalog) => catalog.cone_search(ra, dec, radius_deg, limit),
        }
    }

    fn star_count(&self) -> u64 {
        match self {
            Self::Dated(catalog) => catalog.star_count(),
            Self::Memory(catalog) => catalog.star_count(),
        }
    }

    fn all_brighter_than(&self, mag_limit: f32) -> Vec<CatalogStar> {
        match self {
            Self::Dated(catalog) => catalog.all_brighter_than(mag_limit),
            Self::Memory(catalog) => catalog.all_brighter_than(mag_limit),
        }
    }
}

/// A star catalog used as the reference for plate solving.
#[pyclass(frozen, name = "StarCatalog", module = "seiza")]
struct PyStarCatalog {
    backend: CatalogBackend,
}

#[pymethods]
impl PyStarCatalog {
    /// Open a memory-mapped seiza star tile catalog (`stars-*.bin`).
    /// `path` may be a file, a directory holding catalogs (the deepest star
    /// catalog inside is picked), or omitted to search the standard
    /// locations (`SEIZA_STAR_DATA`, files next to the program, and the
    /// `seiza setup` directories) — the same rules as the CLI's `--data`.
    /// The proper motions beside the file (`stars-gaia.pm.bin` beside
    /// `stars-gaia.bin`) are opened too when it has them; see
    /// :attr:`has_proper_motions`.
    #[staticmethod]
    #[pyo3(signature = (path=None))]
    fn open(path: Option<PathBuf>) -> PyResult<Self> {
        let path = seiza::data_paths::star_data(path.as_deref())
            .map_err(|error| PyFileNotFoundError::new_err(error.to_string()))?;
        let catalog = seiza::catalog::TileCatalog::open(&path)
            .map_err(|error| PyIOError::new_err(format!("{}: {error}", path.display())))?;
        let motions =
            seiza::catalog::ProperMotions::open_beside(&path, &catalog).map_err(|error| {
                PyIOError::new_err(format!(
                    "the proper motions beside {}: {error}",
                    path.display()
                ))
            })?;
        Ok(Self {
            backend: CatalogBackend::Tiles(Box::new(catalog), motions.map(Box::new)),
        })
    }

    /// Whether the catalog has proper motions, so that a solve given an
    /// ``epoch`` searches it moved to that date.
    #[getter]
    fn has_proper_motions(&self) -> bool {
        matches!(self.backend, CatalogBackend::Tiles(_, Some(_)))
    }

    /// Build an in-memory catalog from `(ra_deg, dec_deg, mag)` tuples.
    /// Intended for tests and synthetic fields, not full-sky solving.
    #[staticmethod]
    fn from_stars(stars: Vec<(f64, f64, f32)>) -> Self {
        let stars = stars
            .into_iter()
            .map(|(ra, dec, mag)| CatalogStar { ra, dec, mag })
            .collect();
        Self {
            backend: CatalogBackend::Memory(MemoryCatalog::new(stars)),
        }
    }

    /// Total stars in the catalog (0 when unknown).
    fn star_count(&self) -> u64 {
        self.backend.as_trait().star_count()
    }

    /// Stars within `radius_deg` of `(ra, dec)`, brightest first, as
    /// `(ra_deg, dec_deg, mag)` tuples.
    #[pyo3(signature = (ra, dec, radius_deg, limit=100))]
    fn cone_search(
        &self,
        py: Python<'_>,
        ra: f64,
        dec: f64,
        radius_deg: f64,
        limit: usize,
    ) -> Vec<(f64, f64, f32)> {
        py.detach(|| {
            self.backend
                .as_trait()
                .cone_search(ra, dec, radius_deg, limit)
                .into_iter()
                .map(|star| (star.ra, star.dec, star.mag))
                .collect()
        })
    }
}

/// A memory-mapped whole-sky blind-solving pattern index (`blind-*.idx`).
#[pyclass(frozen, name = "BlindIndex", module = "seiza")]
struct PyBlindIndex {
    index: seiza::blind::BlindIndex,
}

#[pymethods]
impl PyBlindIndex {
    /// Open a prebuilt blind pattern index (`*.idx`). `path` may be a file,
    /// a directory holding one, or omitted to search the standard locations
    /// (`SEIZA_BLIND_INDEX`, files next to the program, and the
    /// `seiza setup` directories).
    #[staticmethod]
    #[pyo3(signature = (path=None))]
    fn open(path: Option<PathBuf>) -> PyResult<Self> {
        let path = seiza::data_paths::blind_index(path.as_deref())
            .map_err(|error| PyFileNotFoundError::new_err(error.to_string()))?
            .ok_or_else(|| {
                PyFileNotFoundError::new_err(
                    seiza::data_paths::DataPathError::NoDefault {
                        kind: "blind index",
                    }
                    .to_string(),
                )
            })?;
        let index = seiza::blind::BlindIndex::open(&path)
            .map_err(|error| PyIOError::new_err(error.to_string()))?;
        Ok(Self { index })
    }

    fn pattern_count(&self) -> usize {
        self.index.pattern_count()
    }
}

/// A TAN-projection WCS. `crpix` is 0-indexed; FITS keyword output converts
/// to the 1-indexed FITS convention.
#[pyclass(skip_from_py_object, frozen, name = "Wcs", module = "seiza")]
#[derive(Clone)]
pub(crate) struct PyWcs {
    pub(crate) wcs: SeizaWcs,
}

#[pymethods]
impl PyWcs {
    /// Construct from a field center, 0-indexed reference pixel, pixel scale
    /// in arcsec/px, rotation in degrees east of north, and parity.
    #[staticmethod]
    #[pyo3(signature = (center, crpix, scale_arcsec_px, rotation_deg=0.0, flipped=false))]
    fn from_center_scale_rotation(
        center: (f64, f64),
        crpix: (f64, f64),
        scale_arcsec_px: f64,
        rotation_deg: f64,
        flipped: bool,
    ) -> Self {
        Self {
            wcs: SeizaWcs::from_center_scale_rotation(
                center,
                crpix,
                scale_arcsec_px,
                rotation_deg,
                flipped,
            ),
        }
    }

    /// Read a TAN or TAN-SIP solution from FITS WCS keywords, such as an
    /// astropy header or a plain dict; None when the keywords do not
    /// describe one.
    #[staticmethod]
    fn from_header(header: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Option<Self>> {
        let lookup = |key: &str| -> Option<Bound<'_, pyo3::types::PyAny>> {
            header.get_item(key).ok().filter(|value| !value.is_none())
        };
        Ok(SeizaWcs::from_fits_values(
            |key| {
                let value = lookup(key)?;
                value.extract::<f64>().ok().or_else(|| {
                    value
                        .extract::<String>()
                        .ok()
                        .and_then(|text| text.trim().parse().ok())
                })
            },
            |key| lookup(key)?.extract::<String>().ok(),
        )
        .map(|wcs| Self { wcs }))
    }

    #[getter]
    fn crval(&self) -> (f64, f64) {
        self.wcs.crval
    }

    #[getter]
    fn crpix(&self) -> (f64, f64) {
        self.wcs.crpix
    }

    /// CD matrix in degrees per pixel: ((cd1_1, cd1_2), (cd2_1, cd2_2)).
    #[getter]
    fn cd(&self) -> ((f64, f64), (f64, f64)) {
        let cd = self.wcs.cd;
        ((cd[0][0], cd[0][1]), (cd[1][0], cd[1][1]))
    }

    #[getter]
    fn scale_arcsec_px(&self) -> f64 {
        self.wcs.scale_arcsec_per_px()
    }

    /// Position angle of north in the image, degrees east of north.
    #[getter]
    fn rotation_deg(&self) -> f64 {
        let cd = self.wcs.cd;
        cd[0][1].atan2(-cd[1][1]).to_degrees()
    }

    /// True when the image parity is mirrored.
    #[getter]
    fn flipped(&self) -> bool {
        let cd = self.wcs.cd;
        cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0] < 0.0
    }

    /// Map a 0-indexed pixel coordinate to `(ra_deg, dec_deg)`.
    fn pixel_to_world(&self, x: f64, y: f64) -> (f64, f64) {
        self.wcs.pixel_to_world(x, y)
    }

    /// Map `(ra_deg, dec_deg)` to a 0-indexed pixel coordinate, or `None`
    /// when the position is on the far hemisphere.
    fn world_to_pixel(&self, ra: f64, dec: f64) -> Option<(f64, f64)> {
        self.wcs.world_to_pixel(ra, dec)
    }

    /// The sky positions of the four image corners for `(width, height)`.
    fn footprint(&self, width: u32, height: u32) -> Vec<(f64, f64)> {
        self.wcs.footprint(width, height).to_vec()
    }

    /// The solution of an image made from this one by a pixel mapping, such
    /// as registration, a turn, binning or a crop: a pixel `(x, y)` here
    /// lands at `(a x + b y + translation_x, c x + d y + translation_y)` on
    /// the new image, with `matrix` `((a, b), (c, d))`. Pixels count from 0
    /// with centres on whole numbers. A crop at `(x, y)` is the identity
    /// matrix with translation `(-x, -y)`; a half turn of a `w` × `h` image is
    /// `-1` on the diagonal with translation `(w - 1, h - 1)`; drizzle or
    /// binning by a scale `s` is `s` on the diagonal with translation
    /// `((s - 1) / 2, (s - 1) / 2)`. Raises when the mapping cannot be
    /// inverted, or when the solution has SIP terms and the mapping is more
    /// than a translation.
    #[pyo3(signature = (matrix, translation_x=0.0, translation_y=0.0))]
    fn transformed(
        &self,
        matrix: ((f64, f64), (f64, f64)),
        translation_x: f64,
        translation_y: f64,
    ) -> PyResult<Self> {
        let ((a, b), (c, d)) = matrix;
        seiza_stacking::transform_wcs(
            &self.wcs,
            seiza_stacking::AffineTransform {
                matrix: [[a, b], [c, d]],
                translation_x,
                translation_y,
            },
        )
        .map(|wcs| Self { wcs })
        .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// SIP distortion order, or `None` for a purely linear solution.
    #[getter]
    fn sip_order(&self) -> Option<u8> {
        self.wcs.sip.as_ref().map(|sip| sip.order)
    }

    /// SIP coefficients as `{"A": {(p, q): value, ...}, "B": ..., "AP": ...,
    /// "BP": ...}`, or `None` for a purely linear solution.
    fn sip_coefficients<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(sip) = &self.wcs.sip else {
            return Ok(None);
        };
        let cards = PyDict::new(py);
        for (name, terms, values) in [
            ("A", seiza::Sip::forward_terms(sip.order), &sip.a),
            ("B", seiza::Sip::forward_terms(sip.order), &sip.b),
            ("AP", seiza::Sip::inverse_terms(sip.order), &sip.ap),
            ("BP", seiza::Sip::inverse_terms(sip.order), &sip.bp),
        ] {
            let inner = PyDict::new(py);
            for ((p, q), value) in terms.iter().zip(values) {
                inner.set_item((*p, *q), *value)?;
            }
            cards.set_item(name, inner)?;
        }
        Ok(Some(cards))
    }

    /// FITS WCS keywords as a dict: 1-indexed `CRPIX`, TAN projection, and
    /// the complete SIP keyword set when distortion was fitted.
    fn fits_header_cards<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let cards = PyDict::new(py);
        for (keyword, value) in self.wcs.fits_header_cards() {
            match value {
                seiza::FitsCardValue::Text(text) => cards.set_item(keyword, text)?,
                seiza::FitsCardValue::Integer(value) => cards.set_item(keyword, value)?,
                seiza::FitsCardValue::Number(value) => cards.set_item(keyword, value)?,
            }
        }
        Ok(cards)
    }

    fn __repr__(&self) -> String {
        format!(
            "Wcs(crval=({:.5}, {:+.5}), scale={:.3}\"/px, rotation={:.2}°)",
            self.wcs.crval.0,
            self.wcs.crval.1,
            self.wcs.scale_arcsec_per_px(),
            self.rotation_deg(),
        )
    }
}

/// A successful plate solution.
#[pyclass(frozen, name = "Solution", module = "seiza")]
struct PySolution {
    #[pyo3(get)]
    wcs: PyWcs,
    /// Number of detected stars matched to catalog stars.
    #[pyo3(get)]
    matched_stars: usize,
    /// RMS of match residuals, arcseconds.
    #[pyo3(get)]
    rms_arcsec: f64,
    /// The chance that a wrong field would match this well, across every
    /// candidate the search tried. Near 0 for a sure solution.
    #[pyo3(get)]
    false_match_probability: f64,
    /// Where :func:`solve_with_prior` or :func:`solve_image` found the solution:
    /// ``"header-checked"``, ``"header-hint"``, ``"blind"``, or
    /// ``"header-unchecked"`` when nothing solved and the image's own WCS
    /// stands unchecked. ``None`` from the other solves.
    #[pyo3(get)]
    source: Option<&'static str>,
    width: u32,
    height: u32,
}

#[pymethods]
impl PySolution {
    /// Sky coordinates of the image center, `(ra_deg, dec_deg)`.
    #[getter]
    fn center(&self) -> (f64, f64) {
        let middle = seiza::image_center(self.width, self.height);
        self.wcs.wcs.pixel_to_world(middle.0, middle.1)
    }

    #[getter]
    fn ra(&self) -> f64 {
        self.center().0
    }

    #[getter]
    fn dec(&self) -> f64 {
        self.center().1
    }

    #[getter]
    fn scale_arcsec_px(&self) -> f64 {
        self.wcs.scale_arcsec_px()
    }

    #[getter]
    fn rotation_deg(&self) -> f64 {
        self.wcs.rotation_deg()
    }

    #[getter]
    fn flipped(&self) -> bool {
        self.wcs.flipped()
    }

    /// FITS WCS keywords as a dict (1-indexed `CRPIX`, TAN projection).
    fn fits_header_cards<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.wcs.fits_header_cards(py)
    }

    /// A minimal FITS header string (80-column cards ending with `END`)
    /// carrying the WCS keywords, including SIP keywords when distortion
    /// was fitted. Suitable for header-injection APIs such as sirilpy's
    /// `set_image_header`.
    fn fits_header_text(&self) -> String {
        self.wcs.wcs.fits_header_text()
    }

    fn __repr__(&self) -> String {
        let (ra, dec) = self.center();
        format!(
            "Solution(ra={ra:.5}, dec={dec:+.5}, scale={:.3}\"/px, matched={}, rms={:.2}\")",
            self.wcs.scale_arcsec_px(),
            self.matched_stars,
            self.rms_arcsec,
        )
    }
}

fn solution_to_py(solution: seiza::solve::Solution, dimensions: (u32, u32)) -> PySolution {
    PySolution {
        wcs: PyWcs { wcs: solution.wcs },
        matched_stars: solution.matched_stars,
        rms_arcsec: solution.rms_arcsec,
        false_match_probability: solution.false_match_probability,
        source: None,
        width: dimensions.0,
        height: dimensions.1,
    }
}

/// How often a solve asks Python whether to stop. Every check would take
/// the GIL, and a blind solve checks from every core at once.
const CANCEL_POLL: std::time::Duration = std::time::Duration::from_millis(50);

/// The limits a Python solve runs under, and any exception its `cancel`
/// predicate (or Ctrl-C) raised, kept to re-raise once the solve returns.
struct PySolveLimits {
    limits: seiza::SolveLimits,
    raised: std::sync::Arc<std::sync::Mutex<Option<PyErr>>>,
}

impl PySolveLimits {
    /// `time_limit` in seconds; `cancel` a callable returning true to stop.
    /// Without `cancel` the solve still stops on Ctrl-C.
    fn new(time_limit: Option<f64>, cancel: Option<Py<PyAny>>) -> PyResult<Self> {
        let limits = match time_limit {
            None => seiza::SolveLimits::default(),
            Some(seconds) if seconds.is_finite() && seconds > 0.0 => {
                seiza::SolveLimits::within(std::time::Duration::from_secs_f64(seconds))
            }
            Some(_) => {
                return Err(PyValueError::new_err(
                    "time_limit must be a positive number of seconds",
                ));
            }
        };
        let raised = std::sync::Arc::new(std::sync::Mutex::new(None));
        let signal = {
            let raised = std::sync::Arc::clone(&raised);
            // The last time Python was asked, and whether it said stop.
            let asked: std::sync::Mutex<(Option<std::time::Instant>, bool)> =
                std::sync::Mutex::new((None, false));
            seiza::CancelSignal::new(move || {
                let Ok(mut asked) = asked.lock() else {
                    return true;
                };
                if asked.1 {
                    return true;
                }
                if asked.0.is_some_and(|last| last.elapsed() < CANCEL_POLL) {
                    return false;
                }
                asked.0 = Some(std::time::Instant::now());
                asked.1 = Python::attach(|py| {
                    let stop = py.check_signals().and_then(|()| match &cancel {
                        Some(cancel) => cancel
                            .call0(py)
                            .and_then(|value| value.bind(py).is_truthy()),
                        None => Ok(false),
                    });
                    stop.unwrap_or_else(|error| {
                        if let Ok(mut slot) = raised.lock() {
                            slot.get_or_insert(error);
                        }
                        true
                    })
                });
                asked.1
            })
        };
        Ok(Self {
            limits: limits.with_cancel(signal),
            raised,
        })
    }

    /// The solve's result for Python: an exception the predicate raised,
    /// `TimeoutError` at the time limit, or `SolveError`.
    fn finish(
        self,
        result: Result<seiza::solve::Solution, seiza::Error>,
        dimensions: (u32, u32),
    ) -> PyResult<PySolution> {
        if let Some(error) = self.raised.lock().ok().and_then(|mut slot| slot.take()) {
            return Err(error);
        }
        result
            .map(|solution| solution_to_py(solution, dimensions))
            .map_err(|error| match error {
                seiza::Error::TimedOut => {
                    pyo3::exceptions::PyTimeoutError::new_err(error.to_string())
                }
                other => SolveError::new_err(other.to_string()),
            })
    }
}

/// Accept `Star` objects or `(x, y, flux)` tuples; return brightest-first.
fn extract_stars(stars: &Bound<'_, PyAny>) -> PyResult<Vec<DetectedStar>> {
    let mut out = Vec::new();
    for item in stars.try_iter()? {
        let item = item?;
        if let Ok(star) = item.extract::<Star>() {
            out.push(DetectedStar::from(&star));
        } else {
            let (x, y, flux): (f64, f64, f64) = item.extract().map_err(|_| {
                PyValueError::new_err("stars must be seiza.Star objects or (x, y, flux) tuples")
            })?;
            out.push(DetectedStar {
                x,
                y,
                flux,
                peak: flux as f32,
                area: 1,
            });
        }
    }
    out.sort_by(|a, b| b.flux.total_cmp(&a.flux));
    Ok(out)
}

fn detect_config(
    sigma: f32,
    max_stars: usize,
    tile_size: u32,
    min_area: u32,
    max_area: u32,
    max_elongation: f32,
    ignore_border: u32,
) -> DetectConfig {
    DetectConfig {
        sigma,
        max_stars,
        tile_size,
        min_area,
        max_area,
        max_elongation,
        ignore_border,
        ..DetectConfig::default()
    }
}

/// Detect stars in a 2D image array (rows are `y`). Accepts `float32`
/// (linear luma in any scale) or `uint8` arrays; returns brightest-first.
#[pyfunction]
#[pyo3(signature = (image, *, sigma=4.0, max_stars=500, tile_size=64, min_area=3,
    max_area=20_000, max_elongation=2.5, ignore_border=0))]
#[allow(clippy::too_many_arguments)]
fn detect(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    sigma: f32,
    max_stars: usize,
    tile_size: u32,
    min_area: u32,
    max_area: u32,
    max_elongation: f32,
    ignore_border: u32,
) -> PyResult<Vec<Star>> {
    let config = detect_config(
        sigma,
        max_stars,
        tile_size,
        min_area,
        max_area,
        max_elongation,
        ignore_border,
    );
    let stars = if let Ok(array) = image.extract::<PyReadonlyArray2<'_, f32>>() {
        let dims = array.shape();
        let (height, width) = (dims[0] as u32, dims[1] as u32);
        let pixels = array
            .as_slice()
            .map_err(|_| PyValueError::new_err("image array is not contiguous"))?;
        py.detach(|| detect_stars_luma_f32(pixels, width, height, &config))
    } else if let Ok(array) = image.extract::<PyReadonlyArray2<'_, u8>>() {
        let dims = array.shape();
        let (height, width) = (dims[0] as u32, dims[1] as u32);
        let pixels = array
            .as_slice()
            .map_err(|_| PyValueError::new_err("image array is not contiguous"))?
            .to_vec();
        let buffer = image::GrayImage::from_raw(width, height, pixels)
            .ok_or_else(|| PyValueError::new_err("image dimensions are inconsistent"))?;
        let dynamic = image::DynamicImage::ImageLuma8(buffer);
        py.detach(|| detect_stars(&dynamic, &config))
    } else {
        return Err(PyValueError::new_err(
            "image must be a 2D numpy array of float32 or uint8",
        ));
    };
    Ok(stars.into_iter().map(Star::from).collect())
}

/// Measure detected stars again on an image's linear samples, for the
/// positions a plate solve fits. ``image`` is a 2D ``float32`` or ``uint16``
/// array (rows are ``y``) of linear samples covering the image in blocks of
/// ``binning`` x ``binning`` pixels: the raw pixels of a mono frame, or a raw
/// colour mosaic summed in 2 x 2 blocks with ``binning=2``. Each star's
/// background is a plane fitted to an annulus around it and its centroid is
/// iterated with a Gaussian weight, which removes the bias that a stretched
/// detection image and a sky gradient give. Stars that cannot be measured
/// keep their positions; flux, peak and area stay as detected.
#[pyfunction]
#[pyo3(signature = (stars, image, *, binning=1))]
fn refine_centroids(
    py: Python<'_>,
    stars: &Bound<'_, PyAny>,
    image: &Bound<'_, PyAny>,
    binning: usize,
) -> PyResult<Vec<Star>> {
    if binning == 0 {
        return Err(PyValueError::new_err("binning must be at least 1"));
    }
    let mut stars = extract_stars(stars)?;
    if let Ok(array) = image.extract::<PyReadonlyArray2<'_, f32>>() {
        let dims = array.shape();
        let (height, width) = (dims[0], dims[1]);
        let pixels = array
            .as_slice()
            .map_err(|_| PyValueError::new_err("image array is not contiguous"))?;
        py.detach(|| {
            seiza::detect::refine_centroids(&mut stars, width, height, binning, &|x, y| {
                pixels[y * width + x]
            })
        });
    } else if let Ok(array) = image.extract::<PyReadonlyArray2<'_, u16>>() {
        let dims = array.shape();
        let (height, width) = (dims[0], dims[1]);
        let pixels = array
            .as_slice()
            .map_err(|_| PyValueError::new_err("image array is not contiguous"))?;
        py.detach(|| {
            seiza::detect::refine_centroids(&mut stars, width, height, binning, &|x, y| {
                f32::from(pixels[y * width + x])
            })
        });
    } else {
        return Err(PyValueError::new_err(
            "image must be a 2D numpy array of float32 or uint16",
        ));
    }
    Ok(stars.into_iter().map(Star::from).collect())
}

/// Plate solve with an approximate position and pixel-scale hint.
/// ``epoch``, a Julian year such as ``2024.92``, moves a catalog with proper
/// motions (:attr:`StarCatalog.has_proper_motions`) to that date before it
/// is searched; leave it ``None`` to search the catalog as written. The
/// same keyword does the same for :func:`solve_blind` and
/// :func:`solve_with_prior`.
#[pyfunction]
#[pyo3(signature = (stars, catalog, width, height, *, ra, dec, scale_arcsec_px,
    radius_deg=2.0, scale_tolerance=0.2, sip_order=0, time_limit=None, cancel=None, epoch=None))]
#[allow(clippy::too_many_arguments)]
fn solve(
    py: Python<'_>,
    stars: &Bound<'_, PyAny>,
    catalog: &Bound<'_, PyStarCatalog>,
    width: u32,
    height: u32,
    ra: f64,
    dec: f64,
    scale_arcsec_px: f64,
    radius_deg: f64,
    scale_tolerance: f64,
    sip_order: u8,
    time_limit: Option<f64>,
    cancel: Option<Py<PyAny>>,
    epoch: Option<f64>,
) -> PyResult<PySolution> {
    if sip_order > 5 {
        return Err(PyValueError::new_err("sip_order must be between 0 and 5"));
    }
    let stars = extract_stars(stars)?;
    let catalog = catalog.get();
    let hint = seiza::solve::SolveHint {
        center: (ra, dec),
        radius_deg,
        scale_arcsec_px,
        scale_tolerance,
        sip_order,
    };
    let limits = PySolveLimits::new(time_limit, cancel)?;
    let solution = py.detach(|| {
        seiza::solve::solve_with_limits(
            &stars,
            &catalog.backend.at_epoch(epoch),
            &hint,
            (width, height),
            &limits.limits,
        )
    });
    limits.finish(solution, (width, height))
}

/// Plate solve with no position hint using a whole-sky pattern index.
#[pyfunction]
#[pyo3(signature = (stars, catalog, index, width, height, *,
    min_scale_arcsec_px=0.1, max_scale_arcsec_px=20.0, sip_order=0, time_limit=None,
    cancel=None, epoch=None))]
#[allow(clippy::too_many_arguments)]
fn solve_blind(
    py: Python<'_>,
    stars: &Bound<'_, PyAny>,
    catalog: &Bound<'_, PyStarCatalog>,
    index: &Bound<'_, PyBlindIndex>,
    width: u32,
    height: u32,
    min_scale_arcsec_px: f64,
    max_scale_arcsec_px: f64,
    sip_order: u8,
    time_limit: Option<f64>,
    cancel: Option<Py<PyAny>>,
    epoch: Option<f64>,
) -> PyResult<PySolution> {
    if sip_order > 5 {
        return Err(PyValueError::new_err("sip_order must be between 0 and 5"));
    }
    let stars = extract_stars(stars)?;
    let catalog = catalog.get();
    let index = index.get();
    let params = seiza::blind::BlindParams {
        min_scale_arcsec_px,
        max_scale_arcsec_px,
        sip_order,
        ..seiza::blind::BlindParams::default()
    };
    let limits = PySolveLimits::new(time_limit, cancel)?;
    let solution = py.detach(|| {
        seiza::blind::solve_blind_with_limits(
            &stars,
            &catalog.backend.at_epoch(epoch),
            &index.index,
            &params,
            (width, height),
            &limits.limits,
        )
    });
    limits.finish(solution, (width, height))
}

fn dataset_by_file_name(name: &str) -> PyResult<seiza::downloads::Dataset> {
    use seiza::downloads::Dataset;
    const ALL: [Dataset; 12] = [
        Dataset::BlindGaia16,
        Dataset::GaiaPhotometry,
        Dataset::MinorBodies,
        Dataset::ObjectDistances,
        Dataset::Objects,
        Dataset::StarDistances,
        Dataset::StarsDeepGaia17,
        Dataset::StarsDeepGaia20,
        Dataset::StarsGaia,
        Dataset::StarsLiteTycho2,
        Dataset::StarsLiteTycho2Identifiers,
        Dataset::Transients,
    ];
    ALL.into_iter()
        .find(|dataset| dataset.file_name() == name)
        .ok_or_else(|| {
            let known = ALL.map(|dataset| dataset.file_name()).join(", ");
            PyValueError::new_err(format!("unknown dataset {name:?}; known datasets: {known}"))
        })
}

/// Download and cache published seiza catalogs, returning a dict of dataset
/// file name to verified local path. `datasets` lists file names such as
/// `"stars-lite-tycho2.bin"`; the default is the lightweight solver set plus
/// the object, Solar System, and transient annotation data.
/// Pass `datasets="all"` for the complete standard bundle. The optional,
/// much larger `"stars-deep-gaia20.bin"` catalog must be named explicitly.
/// Files are SHA-256 verified and cached; repeated calls are cheap.
#[pyfunction]
#[pyo3(signature = (datasets=None, *, cache_dir=None))]
fn fetch_catalogs(
    py: Python<'_>,
    datasets: Option<&Bound<'_, PyAny>>,
    cache_dir: Option<PathBuf>,
) -> PyResult<std::collections::BTreeMap<String, PathBuf>> {
    use seiza::downloads::{CatalogManager, CatalogSet};

    let set = match datasets {
        None => CatalogSet::solver_lite(),
        Some(any) => {
            if let Ok(keyword) = any.extract::<String>() {
                if keyword == "all" {
                    CatalogSet::all()
                } else {
                    CatalogSet::dataset(dataset_by_file_name(&keyword)?)
                }
            } else {
                let names: Vec<String> = any.extract().map_err(|_| {
                    PyValueError::new_err(
                        "datasets must be \"all\", a dataset file name, or a list of file names",
                    )
                })?;
                if names.is_empty() {
                    return Err(PyValueError::new_err("datasets list is empty"));
                }
                let mut set = CatalogSet::dataset(dataset_by_file_name(&names[0])?);
                for name in &names[1..] {
                    set = set.with(dataset_by_file_name(name)?);
                }
                set
            }
        }
    };

    py.detach(|| {
        let mut builder = CatalogManager::builder();
        if let Some(cache_dir) = cache_dir {
            builder = builder.cache_dir(cache_dir);
        }
        let manager = builder
            .build()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let bundle = runtime
            .block_on(manager.ensure(&set))
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(bundle
            .artifacts()
            .map(|artifact| (artifact.name.clone(), artifact.path.clone()))
            .collect())
    })
}

#[cfg(test)]
mod tests {
    use super::dataset_by_file_name;
    use seiza::downloads::Dataset;

    #[test]
    fn deep_gaia20_is_selectable_by_python_filename() {
        assert_eq!(
            dataset_by_file_name("stars-deep-gaia20.bin").unwrap(),
            Dataset::StarsDeepGaia20
        );
    }
}

/// Read the EXIF fields of a JPEG, TIFF, HEIF, PNG or WebP file that help a
/// plate solve: capture time (UTC, from the original time and its offset or
/// from the GPS stamps), GPS position, and the 35 mm-equivalent focal
/// length. Missing or damaged EXIF gives empty fields and `warnings`, never
/// an error.
///
/// Pass the decoded image's `width` and `height` (after applying its EXIF
/// orientation, as `PIL.ImageOps.exif_transpose` does) to also get
/// `scale_hint` and `scale_ranges`: the pixel-scale ranges, in arcseconds
/// per pixel, to try in order with :func:`solve_blind`.
#[pyfunction]
#[pyo3(signature = (path, width=None, height=None))]
fn read_photo_metadata(
    py: Python<'_>,
    path: PathBuf,
    width: Option<u32>,
    height: Option<u32>,
) -> PyResult<Py<PyAny>> {
    use seiza_image_io::raster::{PhotoMetadata, ScaleSearch};
    let metadata = PhotoMetadata::read(&path);
    let mut value = serde_json::to_value(&metadata)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    if let (Some(width), Some(height)) = (width, height) {
        let search = ScaleSearch::new(&metadata, (width, height), None, None)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        value["scale_hint"] = serde_json::to_value(metadata.scale_hint((width, height)))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        value["scale_ranges"] = serde_json::to_value(search.ranges)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
    }
    let json = py.import("json")?;
    Ok(json.call_method1("loads", (value.to_string(),))?.unbind())
}

/// When the image at `path` was taken, ISO 8601 in UTC: from a FITS or
/// XISF header's ``DATE-OBS``, ``DATE-BEG`` or ``DATE-AVG``, read without
/// decoding the pixels, or from a photo's EXIF. ``None`` when the file does
/// not say. Pass it to :func:`field_objects` to place comets and asteroids
/// and date transients.
#[pyfunction]
fn read_capture_time(path: PathBuf) -> PyResult<Option<String>> {
    seiza_image_io::read_capture_time(&path)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// What a FITS or XISF file says about where it points, read without
/// decoding its pixels: ``{"wcs": Wcs or None, "center": (ra, dec) or None,
/// "scale_arcsec_px": float or None}``, from its WCS cards or PixInsight
/// solution, its RA/DEC or OBJCTRA/OBJCTDEC, and its PIXSCALE or pixel size
/// over focal length. Pass it to :func:`solve_with_prior`.
#[pyfunction]
fn read_solve_prior(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyDict>> {
    let prior = seiza_image_io::read_solve_prior(&path)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("wcs", prior.wcs.map(|wcs| PyWcs { wcs }))?;
    dict.set_item("center", prior.center)?;
    dict.set_item("scale_arcsec_px", prior.scale_arcsec_px)?;
    Ok(dict.unbind())
}

/// Solve with what the image says about itself: check its own WCS, else
/// search around its pointing and scale, else (with an ``index``) solve
/// blind, trying near its scale first; and when nothing solves, return its
/// own WCS unchecked, with ``source == "header-unchecked"``. ``prior`` is a
/// dict like :func:`read_solve_prior` returns; any key may be missing or
/// ``None``. The other arguments are as for :func:`solve_blind`.
#[pyfunction]
#[pyo3(signature = (stars, catalog, width, height, prior, *, index=None,
    min_scale_arcsec_px=0.1, max_scale_arcsec_px=20.0, sip_order=0, time_limit=None,
    cancel=None, epoch=None))]
#[allow(clippy::too_many_arguments)]
fn solve_with_prior(
    py: Python<'_>,
    stars: &Bound<'_, PyAny>,
    catalog: &Bound<'_, PyStarCatalog>,
    width: u32,
    height: u32,
    prior: &Bound<'_, PyDict>,
    index: Option<&Bound<'_, PyBlindIndex>>,
    min_scale_arcsec_px: f64,
    max_scale_arcsec_px: f64,
    sip_order: u8,
    time_limit: Option<f64>,
    cancel: Option<Py<PyAny>>,
    epoch: Option<f64>,
) -> PyResult<PySolution> {
    if sip_order > 5 {
        return Err(PyValueError::new_err("sip_order must be between 0 and 5"));
    }
    let stars = extract_stars(stars)?;
    let item = |key: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
        Ok(prior.get_item(key)?.filter(|value| !value.is_none()))
    };
    let prior = seiza::prior::SolvePrior {
        wcs: match item("wcs")? {
            Some(value) => Some(
                value
                    .cast::<PyWcs>()
                    .map_err(|_| PyValueError::new_err("prior[\"wcs\"] must be a seiza.Wcs"))?
                    .get()
                    .wcs
                    .clone(),
            ),
            None => None,
        },
        center: item("center")?
            .map(|center| center.extract::<(f64, f64)>())
            .transpose()?,
        scale_arcsec_px: item("scale_arcsec_px")?
            .map(|scale| scale.extract::<f64>())
            .transpose()?,
    };
    let catalog = catalog.get();
    let index = index.map(|index| index.get());
    let params = seiza::blind::BlindParams {
        min_scale_arcsec_px,
        max_scale_arcsec_px,
        sip_order,
        ..seiza::blind::BlindParams::default()
    };
    let limits = PySolveLimits::new(time_limit, cancel)?;
    let found = py.detach(|| {
        seiza::prior::solve_with_prior(
            &stars,
            &catalog.backend.at_epoch(epoch),
            index.map(|index| &index.index),
            &prior,
            &params,
            (width, height),
            &limits.limits,
        )
    });
    let source = found.as_ref().ok().map(|found| found.source.as_str());
    let mut solution = limits.finish(found.map(|found| found.solution), (width, height))?;
    solution.source = source;
    Ok(solution)
}

/// Plate-solve the image file at ``path``, a FITS, XISF, PNG, JPEG or TIFF
/// file, with what it says about itself, as C hosts do with
/// ``seiza_solve_image_json``: its own WCS checked, else a hinted solve
/// around its pointing and scale, else blind; and, when nothing solves, its
/// own WCS unchecked (``source == "header-unchecked"``).
///
/// FITS and XISF images are detected on an MTF stretch, binned 2 x 2 for a
/// raw colour mosaic or a large mono frame, with the stars measured again on
/// the image's own samples, and detected again at full resolution when that
/// does not solve. A photo's EXIF focal length gives the first scale to try.
/// ``catalog_dir`` is a directory of catalogs, or ``None`` for the standard
/// places; the deepest star catalog directly there is used (no recursive
/// search), or pass a star catalog file. The blind index beside it is opened
/// only if the header/hinted solve fails. A proper-motion sidecar, when
/// present, moves the catalog to the image's capture time. The other
/// arguments are as for :func:`solve_blind`.
#[pyfunction]
#[pyo3(signature = (path, *, catalog_dir=None, min_scale_arcsec_px=0.1,
    max_scale_arcsec_px=20.0, sip_order=0, time_limit=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn solve_image(
    py: Python<'_>,
    path: PathBuf,
    catalog_dir: Option<PathBuf>,
    min_scale_arcsec_px: f64,
    max_scale_arcsec_px: f64,
    sip_order: u8,
    time_limit: Option<f64>,
    cancel: Option<Py<PyAny>>,
) -> PyResult<PySolution> {
    if sip_order > 5 {
        return Err(PyValueError::new_err("sip_order must be between 0 and 5"));
    }
    let limits = PySolveLimits::new(time_limit, cancel)?;
    let solved = py.detach(|| {
        seiza::image_solve::solve_image(
            &path,
            &seiza::image_solve::ImageSolveOptions {
                catalog_directory: catalog_dir.as_deref(),
                min_scale_arcsec_px,
                max_scale_arcsec_px,
                sip_order,
            },
            &limits.limits,
        )
    });
    let (dimensions, source, result) = match solved {
        Ok(solved) => (
            (solved.width, solved.height),
            Some(solved.source.as_str()),
            Ok(solved.solution),
        ),
        Err(error) => ((0, 0), None, Err(error)),
    };
    let mut solution = limits.finish(result, dimensions)?;
    solution.source = source;
    Ok(solution)
}

#[pymodule]
#[pyo3(name = "seiza")]
fn seiza_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Star>()?;
    m.add_class::<PyStarCatalog>()?;
    m.add_class::<PyBlindIndex>()?;
    m.add_class::<PyWcs>()?;
    m.add_class::<PySolution>()?;
    background::register(m)?;
    m.add_class::<satellites::PySatelliteCatalog>()?;
    m.add_class::<satellites::PySatelliteTrack>()?;
    m.add_class::<satellites::PyTrackSample>()?;
    m.add_class::<satellites::PyTrackSearchResult>()?;
    m.add_function(wrap_pyfunction!(detect, m)?)?;
    m.add_function(wrap_pyfunction!(refine_centroids, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(solve_blind, m)?)?;
    m.add_function(wrap_pyfunction!(fetch_catalogs, m)?)?;
    m.add_function(wrap_pyfunction!(read_photo_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(read_capture_time, m)?)?;
    m.add_function(wrap_pyfunction!(read_solve_prior, m)?)?;
    m.add_function(wrap_pyfunction!(solve_with_prior, m)?)?;
    m.add_function(wrap_pyfunction!(solve_image, m)?)?;
    color::register(m)?;
    continuum::register(m)?;
    emission::register(m)?;
    color_calibration::register(m)?;
    deconvolution::register(m)?;
    field_objects::register(m)?;
    imgproc::register(m)?;
    stacking::register(m)?;
    stars::register(m)?;
    rc_astro::register(m)?;
    parallax::register(m)?;
    calibration::register(m)?;
    constellations::register(m)?;
    flat_response::register(m)?;
    flat_groups::register(m)?;
    screening::register(m)?;
    stretch::register(m)?;
    m.add("SolveError", m.py().get_type::<SolveError>())?;
    m.add("EngineError", m.py().get_type::<EngineError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // The releases whose stacking and stretch output this build reproduces:
    // what a cache of stacks or stretched images keys on.
    m.add("STACKING_OUTPUT_VERSION", seiza_stacking::OUTPUT_VERSION)?;
    m.add("STRETCH_OUTPUT_VERSION", seiza_stretch::OUTPUT_VERSION)?;
    Ok(())
}
