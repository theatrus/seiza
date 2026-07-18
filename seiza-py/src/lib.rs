//! Python bindings for seiza: star detection, WCS fitting, and hinted/blind
//! plate solving. The Python module is named `seiza`.

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
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

/// A detected or externally supplied star in 0-indexed pixel coordinates.
#[pyclass(frozen, name = "Star", module = "seiza")]
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
    Tiles(seiza::catalog::TileCatalog),
    Memory(MemoryCatalog),
}

impl CatalogBackend {
    fn as_trait(&self) -> &(dyn StarCatalogTrait + Sync) {
        match self {
            Self::Tiles(catalog) => catalog,
            Self::Memory(catalog) => catalog,
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
    #[staticmethod]
    fn open(path: PathBuf) -> PyResult<Self> {
        let catalog = seiza::catalog::TileCatalog::open(&path)
            .map_err(|error| PyIOError::new_err(format!("{}: {error}", path.display())))?;
        Ok(Self {
            backend: CatalogBackend::Tiles(catalog),
        })
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
        ra: f64,
        dec: f64,
        radius_deg: f64,
        limit: usize,
    ) -> Vec<(f64, f64, f32)> {
        self.backend
            .as_trait()
            .cone_search(ra, dec, radius_deg, limit)
            .into_iter()
            .map(|star| (star.ra, star.dec, star.mag))
            .collect()
    }
}

/// A memory-mapped whole-sky blind-solving pattern index (`blind-*.idx`).
#[pyclass(frozen, name = "BlindIndex", module = "seiza")]
struct PyBlindIndex {
    index: seiza::blind::BlindIndex,
}

#[pymethods]
impl PyBlindIndex {
    #[staticmethod]
    fn open(path: PathBuf) -> PyResult<Self> {
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
#[pyclass(frozen, name = "Wcs", module = "seiza")]
#[derive(Clone)]
struct PyWcs {
    wcs: SeizaWcs,
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
    width: u32,
    height: u32,
}

#[pymethods]
impl PySolution {
    /// Sky coordinates of the image center, `(ra_deg, dec_deg)`.
    #[getter]
    fn center(&self) -> (f64, f64) {
        self.wcs.wcs.pixel_to_world(
            (self.width as f64 - 1.0) / 2.0,
            (self.height as f64 - 1.0) / 2.0,
        )
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
        let mut header = String::new();
        for (keyword, value) in self.wcs.wcs.fits_header_cards() {
            let formatted = match value {
                seiza::FitsCardValue::Text(text) => format!("'{text}'"),
                seiza::FitsCardValue::Integer(value) => value.to_string(),
                seiza::FitsCardValue::Number(value) => format!("{value:.13E}"),
            };
            header.push_str(&format!("{keyword:<8}= {formatted:>20}"));
            header.push_str(&" ".repeat(80 - 30));
        }
        header.push_str(&format!("{:<80}", "END"));
        header
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
        width: dimensions.0,
        height: dimensions.1,
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
        let pixels = array.as_array().to_owned();
        let pixels = pixels
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("image array is not contiguous"))?;
        py.allow_threads(|| detect_stars_luma_f32(pixels, width, height, &config))
    } else if let Ok(array) = image.extract::<PyReadonlyArray2<'_, u8>>() {
        let dims = array.shape();
        let (height, width) = (dims[0] as u32, dims[1] as u32);
        let pixels = array.as_array().to_owned();
        let pixels = pixels
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("image array is not contiguous"))?
            .to_vec();
        let buffer = image::GrayImage::from_raw(width, height, pixels)
            .ok_or_else(|| PyValueError::new_err("image dimensions are inconsistent"))?;
        let dynamic = image::DynamicImage::ImageLuma8(buffer);
        py.allow_threads(|| detect_stars(&dynamic, &config))
    } else {
        return Err(PyValueError::new_err(
            "image must be a 2D numpy array of float32 or uint8",
        ));
    };
    Ok(stars.into_iter().map(Star::from).collect())
}

/// Plate solve with an approximate position and pixel-scale hint.
#[pyfunction]
#[pyo3(signature = (stars, catalog, width, height, *, ra, dec, scale_arcsec_px,
    radius_deg=2.0, scale_tolerance=0.2, sip_order=0))]
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
    let solution = py.allow_threads(|| {
        seiza::solve::solve(&stars, catalog.backend.as_trait(), &hint, (width, height))
    });
    solution
        .map(|solution| solution_to_py(solution, (width, height)))
        .map_err(|error| SolveError::new_err(error.to_string()))
}

/// Plate solve with no position hint using a whole-sky pattern index.
#[pyfunction]
#[pyo3(signature = (stars, catalog, index, width, height, *,
    min_scale_arcsec_px=0.1, max_scale_arcsec_px=20.0, sip_order=0))]
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
    let solution = py.allow_threads(|| {
        seiza::blind::solve_blind(
            &stars,
            catalog.backend.as_trait(),
            &index.index,
            &params,
            (width, height),
        )
    });
    solution
        .map(|solution| solution_to_py(solution, (width, height)))
        .map_err(|error| SolveError::new_err(error.to_string()))
}

fn dataset_by_file_name(name: &str) -> PyResult<seiza::downloads::Dataset> {
    use seiza::downloads::Dataset;
    const ALL: [Dataset; 8] = [
        Dataset::BlindGaia16,
        Dataset::MinorBodies,
        Dataset::Objects,
        Dataset::StarsDeepGaia17,
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
/// `"stars-lite-tycho2.bin"`; the default is the lightweight solver set.
/// Pass `datasets="all"` for the complete bundle. Files are SHA-256 verified
/// and cached; repeated calls are cheap.
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

    py.allow_threads(|| {
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

#[pymodule]
#[pyo3(name = "seiza")]
fn seiza_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Star>()?;
    m.add_class::<PyStarCatalog>()?;
    m.add_class::<PyBlindIndex>()?;
    m.add_class::<PyWcs>()?;
    m.add_class::<PySolution>()?;
    m.add_function(wrap_pyfunction!(detect, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(solve_blind, m)?)?;
    m.add_function(wrap_pyfunction!(fetch_catalogs, m)?)?;
    m.add("SolveError", m.py().get_type::<SolveError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
