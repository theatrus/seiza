//! Satellite track prediction bindings over `seiza-satellites`.
//!
//! Results are predictions from orbital elements, not pixel detections; the
//! shapes mirror the Rust crate with Python-friendly scalars (Unix seconds,
//! tuples for pixel geometry).

use crate::PyWcs;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use seiza_satellites as sat;
use std::path::PathBuf;

fn map_error(error: sat::Error) -> PyErr {
    match &error {
        sat::Error::InvalidTimestamp { .. }
        | sat::Error::InvalidExposure(_)
        | sat::Error::InvalidObserver(_)
        | sat::Error::InvalidOptions(_)
        | sat::Error::Elements { .. }
        | sat::Error::EmptyElements(_) => PyValueError::new_err(error.to_string()),
        sat::Error::Io { .. } => PyIOError::new_err(error.to_string()),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

/// `start` accepts Unix seconds, an RFC 3339 string, or a timezone-aware
/// `datetime.datetime`.
fn parse_start(start: &Bound<'_, PyAny>) -> PyResult<sat::UtcTimestamp> {
    if let Ok(seconds) = start.extract::<f64>() {
        return sat::UtcTimestamp::from_unix_seconds(seconds).map_err(map_error);
    }
    if let Ok(text) = start.extract::<String>() {
        return sat::UtcTimestamp::parse(&text).map_err(map_error);
    }
    if let Ok(tzinfo) = start.getattr("tzinfo") {
        if tzinfo.is_none() {
            return Err(PyValueError::new_err(
                "naive datetime is ambiguous; pass a timezone-aware datetime, an \
                 RFC 3339 string, or Unix seconds",
            ));
        }
        let seconds: f64 = start.call_method0("timestamp")?.extract()?;
        return sat::UtcTimestamp::from_unix_seconds(seconds).map_err(map_error);
    }
    Err(PyValueError::new_err(
        "start must be Unix seconds, an RFC 3339 string, or a timezone-aware datetime",
    ))
}

/// One time-tagged topocentric prediction sample.
#[pyclass(frozen, name = "TrackSample", module = "seiza")]
#[derive(Clone)]
pub struct PyTrackSample {
    /// UTC time as Unix seconds.
    #[pyo3(get)]
    time_unix: f64,
    #[pyo3(get)]
    ra_deg: f64,
    #[pyo3(get)]
    dec_deg: f64,
    /// 0-indexed pixel position, or `None` below the elevation cutoff or on
    /// the far hemisphere of the projection.
    #[pyo3(get)]
    pixel: Option<(f64, f64)>,
    #[pyo3(get)]
    elevation_deg: f64,
    #[pyo3(get)]
    range_km: f64,
    /// Fraction of the solar disk visible from the satellite, 0 to 1.
    #[pyo3(get)]
    sunlight_fraction: f64,
}

#[pymethods]
impl PyTrackSample {
    fn __repr__(&self) -> String {
        format!(
            "TrackSample(t={:.3}, ra={:.5}, dec={:+.5}, el={:.1}°)",
            self.time_unix, self.ra_deg, self.dec_deg, self.elevation_deg
        )
    }
}

/// A predicted satellite crossing. This is a prediction from orbital
/// elements, never a pixel detection.
#[pyclass(frozen, name = "SatelliteTrack", module = "seiza")]
#[derive(Clone)]
pub struct PySatelliteTrack {
    #[pyo3(get)]
    norad_id: Option<u32>,
    #[pyo3(get)]
    cospar_id: Option<String>,
    #[pyo3(get)]
    name: String,
    /// Human-readable identity, e.g. `"ISS (ZARYA) [25544]"`.
    #[pyo3(get)]
    label: String,
    #[pyo3(get)]
    element_epoch_unix: f64,
    /// Signed seconds between element epoch and exposure midpoint.
    #[pyo3(get)]
    element_age_s: f64,
    #[pyo3(get)]
    source: String,
    #[pyo3(get)]
    sample_interval_s: f64,
    #[pyo3(get)]
    samples: Vec<PyTrackSample>,
    /// In-image path as `((x0, y0), (x1, y1))` segments.
    #[pyo3(get)]
    clipped_segments: Vec<((f64, f64), (f64, f64))>,
    #[pyo3(get)]
    max_elevation_deg: f64,
    #[pyo3(get)]
    max_sunlight_fraction: f64,
    #[pyo3(get)]
    max_apparent_rate_arcsec_per_s: Option<f64>,
    #[pyo3(get)]
    max_pixel_rate_px_per_s: Option<f64>,
    #[pyo3(get)]
    clipped_length_px: f64,
}

#[pymethods]
impl PySatelliteTrack {
    fn __repr__(&self) -> String {
        format!(
            "SatelliteTrack({}, max_el={:.1}°, {} samples, {:.0}px in image)",
            self.label,
            self.max_elevation_deg,
            self.samples.len(),
            self.clipped_length_px,
        )
    }
}

fn track_to_py(track: sat::SatelliteTrack) -> PySatelliteTrack {
    PySatelliteTrack {
        label: track.identity.display_label(),
        max_elevation_deg: track.maximum_elevation_deg(),
        max_sunlight_fraction: track.maximum_sunlight_fraction(),
        max_apparent_rate_arcsec_per_s: track.maximum_apparent_rate_arcsec_per_second(),
        max_pixel_rate_px_per_s: track.maximum_pixel_rate_px_per_second(),
        clipped_length_px: track.clipped_length_px(),
        norad_id: track.identity.norad_id,
        cospar_id: track.identity.cospar_id,
        name: track.identity.name,
        element_epoch_unix: track.element_epoch_utc.unix_seconds(),
        element_age_s: track.element_age_seconds,
        source: track.source,
        sample_interval_s: track.sample_interval_seconds,
        samples: track
            .samples
            .into_iter()
            .map(|sample| PyTrackSample {
                time_unix: sample.time_utc.unix_seconds(),
                ra_deg: sample.ra_deg,
                dec_deg: sample.dec_deg,
                pixel: sample.pixel.map(|point| (point.x, point.y)),
                elevation_deg: sample.elevation_deg,
                range_km: sample.range_km,
                sunlight_fraction: sample.sunlight_fraction,
            })
            .collect(),
        clipped_segments: track
            .clipped_segments
            .into_iter()
            .map(|segment| {
                (
                    (segment.start.x, segment.start.y),
                    (segment.end.x, segment.end.y),
                )
            })
            .collect(),
    }
}

/// Predicted tracks plus catalog accounting for one exposure.
#[pyclass(frozen, name = "TrackSearchResult", module = "seiza")]
pub struct PyTrackSearchResult {
    /// Highest-elevation track first.
    #[pyo3(get)]
    tracks: Vec<PySatelliteTrack>,
    #[pyo3(get)]
    elements_considered: usize,
    #[pyo3(get)]
    propagation_failures: usize,
    /// Records outside `max_element_age_s`, reported instead of silently
    /// extrapolated.
    #[pyo3(get)]
    stale_elements: usize,
}

#[pymethods]
impl PyTrackSearchResult {
    fn __repr__(&self) -> String {
        format!(
            "TrackSearchResult({} tracks, {} considered, {} stale)",
            self.tracks.len(),
            self.elements_considered,
            self.stale_elements
        )
    }
}

struct CatalogMeta {
    provider: &'static str,
    state: &'static str,
    cache_path: PathBuf,
    warning: Option<String>,
}

fn cache_state_label(state: sat::CacheState) -> &'static str {
    match state {
        sat::CacheState::Fresh => "fresh",
        sat::CacheState::Downloaded => "downloaded",
        sat::CacheState::StaleFallback => "stale-fallback",
        sat::CacheState::Cached => "cache-only",
    }
}

/// Parsed OMM or TLE satellite orbital elements with source provenance.
#[pyclass(frozen, name = "SatelliteCatalog", module = "seiza")]
pub struct PySatelliteCatalog {
    catalog: sat::SatelliteCatalog,
    meta: Option<CatalogMeta>,
}

#[pymethods]
impl PySatelliteCatalog {
    /// Open a local element file: CCSDS OMM JSON or two-/three-line TLE.
    #[staticmethod]
    fn open(path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            catalog: sat::SatelliteCatalog::open(path).map_err(map_error)?,
            meta: None,
        })
    }

    /// Parse OMM JSON text (an array or a single object).
    #[staticmethod]
    #[pyo3(signature = (text, source="inline"))]
    fn from_omm_json(text: &str, source: &str) -> PyResult<Self> {
        Ok(Self {
            catalog: sat::SatelliteCatalog::from_omm_json(text, source).map_err(map_error)?,
            meta: None,
        })
    }

    /// Parse two- or three-line TLE text.
    #[staticmethod]
    #[pyo3(signature = (text, source="inline"))]
    fn from_tle_text(text: &str, source: &str) -> PyResult<Self> {
        Ok(Self {
            catalog: sat::SatelliteCatalog::from_tle_text(text, source).map_err(map_error)?,
            meta: None,
        })
    }

    /// Download (or reuse a cached copy of) CelesTrak's current
    /// active-satellite OMM set. The cache refreshes at most every two hours
    /// and falls back to the newest previously validated snapshot when a
    /// refresh fails; CelesTrak rate-limits repeated downloads, so keep
    /// reusing one cache directory. Inspect `cache_state` and `warning`
    /// afterwards. Not suitable for historical images; open a file of
    /// epoch-appropriate elements instead.
    #[staticmethod]
    #[pyo3(signature = (cache_dir=None))]
    fn fetch_celestrak(py: Python<'_>, cache_dir: Option<PathBuf>) -> PyResult<Self> {
        py.allow_threads(|| {
            let source = match cache_dir {
                Some(directory) => sat::CelesTrakSource::new(directory),
                None => sat::CelesTrakSource::platform_default(),
            }
            .map_err(map_error)?;
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            let load = runtime.block_on(source.load_active()).map_err(map_error)?;
            Ok(Self {
                catalog: load.catalog,
                meta: Some(CatalogMeta {
                    provider: "celestrak-active",
                    state: cache_state_label(load.state),
                    cache_path: load.cache_path,
                    warning: load.warning,
                }),
            })
        })
    }

    /// Resolve the epoch-appropriate catalog for an image time, cascading
    /// across providers so callers do not orchestrate them: recent times use
    /// CelesTrak's current active set; older times use the Seiza satellite
    /// mirror, falling back to IAU SatChecker. `time` is Unix seconds, an
    /// RFC 3339 string, or a timezone-aware datetime — pass the exposure
    /// midpoint. Inspect `provider`, `cache_state`, and `warning` afterwards.
    #[staticmethod]
    #[pyo3(signature = (time, cache_dir=None))]
    fn resolve(
        py: Python<'_>,
        time: &Bound<'_, PyAny>,
        cache_dir: Option<PathBuf>,
    ) -> PyResult<Self> {
        let time = parse_start(time)?;
        py.allow_threads(|| {
            let source = match cache_dir {
                Some(directory) => sat::OrbitalCatalogSource::new(directory),
                None => sat::OrbitalCatalogSource::platform_default(),
            }
            .map_err(map_error)?;
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            let load = runtime.block_on(source.load_at(time)).map_err(map_error)?;
            let provider = match load.snapshot.provider {
                sat::OrbitalCatalogProvider::CelesTrakActive => "celestrak-active",
                sat::OrbitalCatalogProvider::SeizaMirror => "seiza-mirror",
                sat::OrbitalCatalogProvider::IauSatChecker => "iau-satchecker",
            };
            Ok(Self {
                catalog: load.catalog,
                meta: Some(CatalogMeta {
                    provider,
                    state: cache_state_label(load.state),
                    cache_path: load.cache_path,
                    warning: load.warning,
                }),
            })
        })
    }

    fn __len__(&self) -> usize {
        self.catalog.len()
    }

    /// Element source provenance (file path, URL, or caller-supplied name).
    #[getter]
    fn source(&self) -> &str {
        self.catalog.source()
    }

    /// When the element set was retrieved, as Unix seconds, if known.
    #[getter]
    fn retrieved_at_unix(&self) -> Option<f64> {
        self.catalog
            .retrieved_at()
            .map(sat::UtcTimestamp::unix_seconds)
    }

    /// Provider that served the elements after `resolve` or `fetch_celestrak`:
    /// `"celestrak-active"`, `"seiza-mirror"`, or `"iau-satchecker"`; `None`
    /// for file- or text-backed catalogs.
    #[getter]
    fn provider(&self) -> Option<&'static str> {
        self.meta.as_ref().map(|meta| meta.provider)
    }

    /// `"fresh"`, `"downloaded"`, `"stale-fallback"`, or `"cache-only"` after
    /// `resolve`/`fetch_celestrak`; `None` for file- or text-backed catalogs.
    #[getter]
    fn cache_state(&self) -> Option<&'static str> {
        self.meta.as_ref().map(|meta| meta.state)
    }

    #[getter]
    fn cache_path(&self) -> Option<PathBuf> {
        self.meta.as_ref().map(|meta| meta.cache_path.clone())
    }

    /// Non-fatal problem from the last refresh (e.g. the reason a stale
    /// fallback was used).
    #[getter]
    fn warning(&self) -> Option<String> {
        self.meta.as_ref().and_then(|meta| meta.warning.clone())
    }

    /// Predict which satellites crossed the solved image while the shutter
    /// was open. `start` is the exposure start as Unix seconds, an RFC 3339
    /// string, or a timezone-aware datetime; the exposure must be one
    /// continuous shutter-open interval (never a stack's total integration).
    /// Give the observer as `latitude`/`longitude` (degrees, east-positive)
    /// with optional `altitude_m`, or as an ITRF `observer_itrf=(x, y, z)`
    /// in meters. `max_element_age_s=None` permits unlimited extrapolation.
    #[pyo3(signature = (wcs, width, height, *, start, duration_s,
        latitude=None, longitude=None, altitude_m=0.0, observer_itrf=None,
        sample_interval_s=1.0, coarse_interval_s=10.0, min_elevation_deg=0.0,
        max_element_age_s=Some(604_800.0)))]
    #[allow(clippy::too_many_arguments)]
    fn tracks_in_footprint(
        &self,
        py: Python<'_>,
        wcs: &Bound<'_, PyWcs>,
        width: u32,
        height: u32,
        start: &Bound<'_, PyAny>,
        duration_s: f64,
        latitude: Option<f64>,
        longitude: Option<f64>,
        altitude_m: f64,
        observer_itrf: Option<(f64, f64, f64)>,
        sample_interval_s: f64,
        coarse_interval_s: f64,
        min_elevation_deg: f64,
        max_element_age_s: Option<f64>,
    ) -> PyResult<PyTrackSearchResult> {
        let observer = match (latitude, longitude, observer_itrf) {
            (None, None, Some((x, y, z))) => {
                sat::ObserverLocation::itrf_meters(x, y, z).map_err(map_error)?
            }
            (Some(latitude), Some(longitude), None) => {
                sat::ObserverLocation::geodetic(latitude, longitude, altitude_m)
                    .map_err(map_error)?
            }
            (None, None, None) => {
                return Err(PyValueError::new_err(
                    "an observer location is required: pass latitude and longitude, \
                     or observer_itrf=(x, y, z) meters",
                ));
            }
            _ => {
                return Err(PyValueError::new_err(
                    "pass either latitude and longitude together, or observer_itrf, \
                     not a mixture",
                ));
            }
        };
        let exposure = sat::SingleExposure::from_start_and_duration(
            parse_start(start)?,
            duration_s,
            observer,
            sat::ExposureProvenance::Explicit,
        )
        .map_err(map_error)?;
        let options = sat::TrackOptions {
            sample_interval_seconds: sample_interval_s,
            coarse_interval_seconds: coarse_interval_s,
            minimum_elevation_deg: min_elevation_deg,
            maximum_element_age_seconds: max_element_age_s,
        };
        let wcs = wcs.get().wcs.clone();
        let result = py
            .allow_threads(|| {
                self.catalog
                    .tracks_in_footprint(&wcs, (width, height), &exposure, &options)
            })
            .map_err(map_error)?;
        Ok(PyTrackSearchResult {
            tracks: result.tracks.into_iter().map(track_to_py).collect(),
            elements_considered: result.elements_considered,
            propagation_failures: result.propagation_failures,
            stale_elements: result.stale_elements,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SatelliteCatalog({} records, source={:?})",
            self.catalog.len(),
            self.catalog.source()
        )
    }
}
