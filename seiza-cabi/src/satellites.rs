//! Satellite tracks across a solved image (`seiza-satellites`), as Python's
//! `SatelliteCatalog` predicts them.
//!
//! A [`SeizaSatelliteCatalog`] holds orbital elements: from a file
//! ([`seiza_satellite_catalog_open`]), from OMM JSON or TLE text, from
//! CelesTrak's current set ([`seiza_satellite_catalog_fetch_celestrak`]) or
//! from the provider that suits an image's time
//! ([`seiza_satellite_catalog_resolve_json`]).
//! [`seiza_satellite_catalog_tracks_json`] then predicts which satellites
//! crossed the image while the shutter was open. The tracks are predictions
//! from the elements, never detections in the pixels.

use super::{
    SeizaCancelSignal, clear_error, ffi_result, optional_path, owned_json, required_path,
    required_str,
};
use crate::parallax::WcsRequest;
use seiza_satellites as sat;
use serde::{Deserialize, Serialize};
use std::ffi::c_char;
use std::path::PathBuf;
use std::ptr;

/// An opaque set of satellite orbital elements, with where they came from.
/// Release it with [`seiza_satellite_catalog_free`]. It is read-only once
/// made, so several threads may predict tracks from it at once; do not free
/// it while any call using it is running.
pub struct SeizaSatelliteCatalog {
    catalog: sat::SatelliteCatalog,
    load: Option<LoadInfo>,
}

/// What a network load reports beside the elements.
struct LoadInfo {
    provider: &'static str,
    cache_state: &'static str,
    cache_path: PathBuf,
    warning: Option<String>,
}

/// A time as Unix seconds or as RFC 3339 text. Text with no offset is UTC.
#[derive(Deserialize)]
#[serde(untagged)]
enum TimeRequest {
    UnixSeconds(f64),
    Text(String),
}

impl TimeRequest {
    fn timestamp(&self) -> Result<sat::UtcTimestamp, sat::Error> {
        match self {
            Self::UnixSeconds(seconds) => sat::UtcTimestamp::from_unix_seconds(*seconds),
            Self::Text(text) => sat::UtcTimestamp::parse(text),
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SatelliteResolveRequest {
    /// The image's time; give the exposure's midpoint.
    time: TimeRequest,
    /// The cache directory; the platform's satellite cache if absent.
    #[serde(default)]
    cache_dir: Option<PathBuf>,
}

/// One exposure and the prediction settings. The last four fields are
/// `seiza-satellites::TrackOptions`, with its defaults.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SatelliteTracksRequest {
    /// When the shutter opened.
    start: TimeRequest,
    /// The exposure's length; give this or `end`.
    #[serde(default)]
    duration_seconds: Option<f64>,
    /// When the shutter closed; give this or `durationSeconds`.
    #[serde(default)]
    end: Option<TimeRequest>,
    /// Geodetic latitude in degrees; give it with `longitude`, or give
    /// `observerItrf`.
    #[serde(default)]
    latitude: Option<f64>,
    /// East-positive longitude in degrees.
    #[serde(default)]
    longitude: Option<f64>,
    /// Height above the ellipsoid in metres.
    #[serde(default)]
    altitude_m: f64,
    /// The observer as ITRF x, y and z in metres, in place of latitude and
    /// longitude.
    #[serde(default)]
    observer_itrf: Option<[f64; 3]>,
    #[serde(default = "default_sample_interval_seconds")]
    sample_interval_seconds: f64,
    #[serde(default = "default_coarse_interval_seconds")]
    coarse_interval_seconds: f64,
    #[serde(default = "default_minimum_elevation_deg")]
    minimum_elevation_deg: f64,
    /// Elements further than this from the exposure's midpoint are counted
    /// as stale and skipped; null allows any age.
    #[serde(default = "default_maximum_element_age_seconds")]
    maximum_element_age_seconds: Option<f64>,
}

fn default_sample_interval_seconds() -> f64 {
    sat::TrackOptions::default().sample_interval_seconds
}

fn default_coarse_interval_seconds() -> f64 {
    sat::TrackOptions::default().coarse_interval_seconds
}

fn default_minimum_elevation_deg() -> f64 {
    sat::TrackOptions::default().minimum_elevation_deg
}

fn default_maximum_element_age_seconds() -> Option<f64> {
    sat::TrackOptions::default().maximum_element_age_seconds
}

impl SatelliteTracksRequest {
    fn observer(&self) -> Result<sat::ObserverLocation, String> {
        match (self.latitude, self.longitude, self.observer_itrf) {
            (None, None, Some([x, y, z])) => {
                sat::ObserverLocation::itrf_meters(x, y, z).map_err(|error| error.to_string())
            }
            (Some(latitude), Some(longitude), None) => {
                sat::ObserverLocation::geodetic(latitude, longitude, self.altitude_m)
                    .map_err(|error| error.to_string())
            }
            (None, None, None) => Err("an observer location is required: give latitude and \
                 longitude, or observerItrf in metres"
                .into()),
            _ => Err("give latitude and longitude together, or observerItrf, not a mixture".into()),
        }
    }

    fn exposure(&self) -> Result<sat::SingleExposure, String> {
        let observer = self.observer()?;
        let start = self.start.timestamp().map_err(|error| error.to_string())?;
        match (self.duration_seconds, &self.end) {
            (Some(duration), None) => sat::SingleExposure::from_start_and_duration(
                start,
                duration,
                observer,
                sat::ExposureProvenance::Explicit,
            ),
            (None, Some(end)) => sat::SingleExposure::new(
                start,
                end.timestamp().map_err(|error| error.to_string())?,
                observer,
                sat::ExposureProvenance::Explicit,
            ),
            _ => {
                return Err(
                    "give exactly one of durationSeconds (the exposure's length) \
                     and end (its close)"
                        .into(),
                );
            }
        }
        .map_err(|error| error.to_string())
    }

    fn options(&self) -> sat::TrackOptions {
        sat::TrackOptions {
            sample_interval_seconds: self.sample_interval_seconds,
            coarse_interval_seconds: self.coarse_interval_seconds,
            minimum_elevation_deg: self.minimum_elevation_deg,
            maximum_element_age_seconds: self.maximum_element_age_seconds,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SatelliteCatalogInfoResponse {
    /// How many element records the catalog holds.
    records: usize,
    /// A file path, a URL, or the caller's own source text.
    source: String,
    retrieved_at_unix: Option<f64>,
    /// `"celestrak-active"`, `"seiza-mirror"` or `"iau-satchecker"` after a
    /// fetch or resolve; null for a file or text.
    provider: Option<&'static str>,
    /// `"fresh"`, `"downloaded"`, `"stale-fallback"` or `"cache-only"` after
    /// a fetch or resolve; null for a file or text.
    cache_state: Option<&'static str>,
    cache_path: Option<String>,
    /// A problem that did not stop the load, such as why a stale copy was
    /// used.
    warning: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SatelliteTrackSearchResponse {
    /// Highest elevation first.
    tracks: Vec<SatelliteTrackResponse>,
    elements_considered: usize,
    propagation_failures: usize,
    stale_elements: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SatelliteTrackResponse {
    norad_id: Option<u32>,
    cospar_id: Option<String>,
    name: String,
    /// The satellite as a person reads it, such as `"ISS (ZARYA) [25544]"`.
    label: String,
    element_epoch_unix: f64,
    /// Seconds from the element epoch to the exposure's midpoint; positive
    /// when the exposure comes after the epoch.
    element_age_seconds: f64,
    source: String,
    sample_interval_seconds: f64,
    samples: Vec<SatelliteTrackSampleResponse>,
    /// The path inside the image.
    clipped_segments: Vec<PixelSegmentResponse>,
    max_elevation_deg: f64,
    max_sunlight_fraction: f64,
    max_apparent_rate_arcsec_per_s: Option<f64>,
    max_pixel_rate_px_per_s: Option<f64>,
    clipped_length_px: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SatelliteTrackSampleResponse {
    time_unix: f64,
    ra_deg: f64,
    dec_deg: f64,
    /// Counted from 0; null below the elevation cutoff or on the far side
    /// of the projection.
    pixel: Option<PixelPointResponse>,
    elevation_deg: f64,
    range_km: f64,
    /// How much of the Sun's disk the satellite sees, 0 to 1.
    sunlight_fraction: f64,
}

#[derive(Serialize)]
struct PixelPointResponse {
    x: f64,
    y: f64,
}

#[derive(Serialize)]
struct PixelSegmentResponse {
    start: PixelPointResponse,
    end: PixelPointResponse,
}

impl From<sat::PixelPoint> for PixelPointResponse {
    fn from(point: sat::PixelPoint) -> Self {
        Self {
            x: point.x,
            y: point.y,
        }
    }
}

impl From<sat::SatelliteTrack> for SatelliteTrackResponse {
    fn from(track: sat::SatelliteTrack) -> Self {
        Self {
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
            element_age_seconds: track.element_age_seconds,
            source: track.source,
            sample_interval_seconds: track.sample_interval_seconds,
            samples: track
                .samples
                .into_iter()
                .map(|sample| SatelliteTrackSampleResponse {
                    time_unix: sample.time_utc.unix_seconds(),
                    ra_deg: sample.ra_deg,
                    dec_deg: sample.dec_deg,
                    pixel: sample.pixel.map(PixelPointResponse::from),
                    elevation_deg: sample.elevation_deg,
                    range_km: sample.range_km,
                    sunlight_fraction: sample.sunlight_fraction,
                })
                .collect(),
            clipped_segments: track
                .clipped_segments
                .into_iter()
                .map(|segment| PixelSegmentResponse {
                    start: segment.start.into(),
                    end: segment.end.into(),
                })
                .collect(),
        }
    }
}

fn catalog_handle(
    error_out: *mut *mut c_char,
    make: impl FnOnce() -> Result<SeizaSatelliteCatalog, String>,
) -> *mut SeizaSatelliteCatalog {
    clear_error(error_out);
    ffi_result(error_out, make).map_or(ptr::null_mut(), |catalog| Box::into_raw(Box::new(catalog)))
}

fn from_elements(
    catalog: sat::Result<sat::SatelliteCatalog>,
) -> Result<SeizaSatelliteCatalog, String> {
    Ok(SeizaSatelliteCatalog {
        catalog: catalog.map_err(|error| error.to_string())?,
        load: None,
    })
}

/// The caller's source text, or `"inline"` when it gives none.
fn source_name(source: *const c_char) -> Result<String, String> {
    if source.is_null() {
        return Ok("inline".into());
    }
    required_str(source, "source")
}

fn cancel_flag<'a>(cancel: *const SeizaCancelSignal) -> Option<&'a std::sync::atomic::AtomicBool> {
    unsafe { cancel.as_ref() }.map(|signal| signal.cancelled.as_ref())
}

#[unsafe(no_mangle)]
/// Open a file of satellite orbital elements: CCSDS OMM JSON (an array or
/// one object) or two- or three-line TLE text. Use it for images older than
/// a few days, with elements from near their time. Returns an owned catalog
/// to release with [`seiza_satellite_catalog_free`], or null with an owned
/// `error_out` string to free with `seiza_string_free`.
///
/// # Safety
/// `path` must be a NUL-terminated UTF-8 string. When non-null, `error_out`
/// must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_satellite_catalog_open(
    path: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut SeizaSatelliteCatalog {
    catalog_handle(error_out, || {
        from_elements(sat::SatelliteCatalog::open(required_path(path, "path")?))
    })
}

#[unsafe(no_mangle)]
/// Parse OMM JSON text (an array or one object). `source`, which the
/// catalog and its tracks report, may be null for `"inline"`. Returns an
/// owned catalog to release with [`seiza_satellite_catalog_free`], or null
/// with an owned `error_out` string.
///
/// # Safety
/// `text` must be a NUL-terminated UTF-8 string, and `source` null or one.
/// When non-null, `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_satellite_catalog_from_omm_json(
    text: *const c_char,
    source: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut SeizaSatelliteCatalog {
    catalog_handle(error_out, || {
        let text = required_str(text, "OMM JSON")?;
        from_elements(sat::SatelliteCatalog::from_omm_json(
            &text,
            source_name(source)?,
        ))
    })
}

#[unsafe(no_mangle)]
/// Parse two- or three-line TLE text. Records that do not parse are
/// skipped; text with none that do fails. `source` may be null for
/// `"inline"`. Returns an owned catalog to release with
/// [`seiza_satellite_catalog_free`], or null with an owned `error_out`
/// string.
///
/// # Safety
/// `text` must be a NUL-terminated UTF-8 string, and `source` null or one.
/// When non-null, `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_satellite_catalog_from_tle_text(
    text: *const c_char,
    source: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut SeizaSatelliteCatalog {
    catalog_handle(error_out, || {
        let text = required_str(text, "TLE text")?;
        from_elements(sat::SatelliteCatalog::from_tle_text(
            &text,
            source_name(source)?,
        ))
    })
}

#[unsafe(no_mangle)]
/// Download CelesTrak's current set of active satellites, or reuse a cached
/// copy. The cache refreshes at most every two hours and falls back to the
/// newest good copy when a refresh fails; CelesTrak blocks clients that
/// download too often, so keep to one cache directory. `cache_dir` may be
/// null or empty for the platform's satellite cache. The call blocks on the
/// network; run it off the UI thread. `cancel` may be null; raising it stops
/// the download at once, failing the call, and leaves no half-written file
/// in the cache. Read the provider, cache state and any warning with
/// [`seiza_satellite_catalog_info_json`]. Current elements suit only recent
/// images; for an older one use [`seiza_satellite_catalog_resolve_json`] or
/// open a file of elements from its time.
///
/// Returns an owned catalog to release with [`seiza_satellite_catalog_free`],
/// or null with an owned `error_out` string.
///
/// # Safety
/// `cache_dir` must be null or a NUL-terminated UTF-8 string. `cancel` must
/// be null or a live [`SeizaCancelSignal`] retained until this call returns.
/// When non-null, `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_satellite_catalog_fetch_celestrak(
    cache_dir: *const c_char,
    cancel: *const SeizaCancelSignal,
    error_out: *mut *mut c_char,
) -> *mut SeizaSatelliteCatalog {
    catalog_handle(error_out, || {
        let source = match optional_path(cache_dir)? {
            Some(directory) => sat::CelesTrakSource::new(directory),
            None => sat::CelesTrakSource::platform_default(),
        }
        .map_err(|error| error.to_string())?;
        let load = source
            .load_active_blocking(cancel_flag(cancel))
            .map_err(|error| error.to_string())?;
        Ok(SeizaSatelliteCatalog {
            catalog: load.catalog,
            load: Some(LoadInfo {
                provider: sat::OrbitalCatalogProvider::CelesTrakActive.as_str(),
                cache_state: load.state.as_str(),
                cache_path: load.cache_path,
                warning: load.warning,
            }),
        })
    })
}

#[unsafe(no_mangle)]
/// Find the elements that suit an image's time, choosing the provider so
/// the caller need not: a recent time uses CelesTrak's current set; an
/// older one a nearby cached copy, then the Seiza satellite mirror, then
/// IAU SatChecker.
///
/// `request_json` is `{"time": ..., "cacheDir": ...}`: `time` is Unix
/// seconds or RFC 3339 text (UTC when it has no offset), best the
/// exposure's midpoint; `cacheDir` may be absent for the platform's
/// satellite cache. A recent time may refresh CelesTrak, which blocks
/// clients that download too often, so keep to one cache directory. The
/// call blocks on the network; `cancel` may be null, and raising it stops
/// the load at once, failing the call. Read the provider, cache state and
/// any warning with [`seiza_satellite_catalog_info_json`].
///
/// Returns an owned catalog to release with [`seiza_satellite_catalog_free`],
/// or null with an owned `error_out` string.
///
/// # Safety
/// `request_json` must be a NUL-terminated UTF-8 string. `cancel` must be
/// null or a live [`SeizaCancelSignal`] retained until this call returns.
/// When non-null, `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_satellite_catalog_resolve_json(
    request_json: *const c_char,
    cancel: *const SeizaCancelSignal,
    error_out: *mut *mut c_char,
) -> *mut SeizaSatelliteCatalog {
    catalog_handle(error_out, || {
        let request: SatelliteResolveRequest =
            serde_json::from_str(&required_str(request_json, "resolve request JSON")?)
                .map_err(|error| format!("invalid satellite resolve request JSON: {error}"))?;
        let time = request
            .time
            .timestamp()
            .map_err(|error| error.to_string())?;
        let source = match request.cache_dir {
            Some(directory) => sat::OrbitalCatalogSource::new(directory),
            None => sat::OrbitalCatalogSource::platform_default(),
        }
        .map_err(|error| error.to_string())?;
        let load = source
            .load_at_blocking(time, cancel_flag(cancel))
            .map_err(|error| error.to_string())?;
        Ok(SeizaSatelliteCatalog {
            catalog: load.catalog,
            load: Some(LoadInfo {
                provider: load.snapshot.provider.as_str(),
                cache_state: load.state.as_str(),
                cache_path: load.cache_path,
                warning: load.warning,
            }),
        })
    })
}

#[unsafe(no_mangle)]
/// Describe a catalog as owned JSON: `records` (how many), `source` (a path,
/// a URL or the caller's text), `retrievedAtUnix`, and after a fetch or
/// resolve `provider` (`"celestrak-active"`, `"seiza-mirror"` or
/// `"iau-satchecker"`), `cacheState` (`"fresh"`, `"downloaded"`,
/// `"stale-fallback"` or `"cache-only"`), `cachePath` and `warning`, a
/// problem that did not stop the load. Those four are null for a file or
/// text. Free the result, or an `error_out` string, with
/// `seiza_string_free`.
///
/// # Safety
/// `catalog` must be a live pointer from a `seiza_satellite_catalog_*`
/// constructor. When non-null, `error_out` must point to writable storage
/// for one pointer.
pub unsafe extern "C" fn seiza_satellite_catalog_info_json(
    catalog: *const SeizaSatelliteCatalog,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let handle = unsafe { catalog.as_ref() }.ok_or("satellite catalog is required")?;
        let load = handle.load.as_ref();
        owned_json(&SatelliteCatalogInfoResponse {
            records: handle.catalog.len(),
            source: handle.catalog.source().to_owned(),
            retrieved_at_unix: handle
                .catalog
                .retrieved_at()
                .map(sat::UtcTimestamp::unix_seconds),
            provider: load.map(|load| load.provider),
            cache_state: load.map(|load| load.cache_state),
            cache_path: load.map(|load| load.cache_path.to_string_lossy().into_owned()),
            warning: load.and_then(|load| load.warning.clone()),
        })
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Predict which satellites crossed a solved `width` by `height` image while
/// the shutter was open. The tracks are predictions from orbital elements,
/// never detections in the pixels.
///
/// `wcs_json` is the image's WCS as the C ABI's solve returns it (`crval`,
/// `crpix`, `cd`, `sip`, and `pixelOrigin`, 0 or 1, for what `crpix` counts
/// from). `request_json` gives one exposure, one continuous time with the
/// shutter open (never a stack's total): `start`, as Unix seconds or RFC 3339
/// text, and either `durationSeconds` or `end`; the observer as `latitude`
/// and `longitude` in degrees (east positive) with `altitudeM` above the
/// ellipsoid, or as `observerItrf`, `[x, y, z]` in metres; and optionally
/// `sampleIntervalSeconds` (1), `coarseIntervalSeconds` (10),
/// `minimumElevationDeg` (0) and `maximumElementAgeSeconds` (seven days;
/// null allows any age).
///
/// Returns owned JSON: `tracks`, highest elevation first, each with its
/// satellite (`noradId`, `cosparId`, `name`, `label`), `elementEpochUnix`,
/// `elementAgeSeconds`, `source`, `sampleIntervalSeconds`, `samples` (each
/// `timeUnix`, `raDeg`, `decDeg`, `pixel` as `{x, y}` counted from 0 or null,
/// `elevationDeg`, `rangeKm`, `sunlightFraction`), `clippedSegments` (the
/// path inside the image, each `{start, end}`), `maxElevationDeg`,
/// `maxSunlightFraction`, `maxApparentRateArcsecPerS`,
/// `maxPixelRatePxPerS` and `clippedLengthPx`; then `elementsConsidered`,
/// `propagationFailures` and `staleElements`, the records skipped for age.
/// Free it, or an `error_out` string, with `seiza_string_free`.
///
/// # Safety
/// `catalog` must be a live pointer from a `seiza_satellite_catalog_*`
/// constructor; `wcs_json` and `request_json` NUL-terminated UTF-8 strings.
/// When non-null, `error_out` must point to writable storage for one pointer.
pub unsafe extern "C" fn seiza_satellite_catalog_tracks_json(
    catalog: *const SeizaSatelliteCatalog,
    wcs_json: *const c_char,
    width: u32,
    height: u32,
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    clear_error(error_out);
    ffi_result(error_out, || {
        let handle = unsafe { catalog.as_ref() }.ok_or("satellite catalog is required")?;
        let wcs = serde_json::from_str::<WcsRequest>(&required_str(wcs_json, "WCS JSON")?)
            .map_err(|error| format!("invalid WCS JSON: {error}"))?
            .wcs()?;
        let request: SatelliteTracksRequest =
            serde_json::from_str(&required_str(request_json, "tracks request JSON")?)
                .map_err(|error| format!("invalid satellite tracks request JSON: {error}"))?;
        let result = handle
            .catalog
            .tracks_in_footprint(
                |ra, dec| wcs.world_to_pixel(ra, dec),
                (width, height),
                &request.exposure()?,
                &request.options(),
            )
            .map_err(|error| error.to_string())?;
        owned_json(&SatelliteTrackSearchResponse {
            tracks: result.tracks.into_iter().map(Into::into).collect(),
            elements_considered: result.elements_considered,
            propagation_failures: result.propagation_failures,
            stale_elements: result.stale_elements,
        })
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
/// Release a catalog from a `seiza_satellite_catalog_*` constructor.
///
/// # Safety
/// `catalog` must be null or a pointer from a `seiza_satellite_catalog_*`
/// constructor that has not already been freed, with no call using it
/// still running.
pub unsafe extern "C" fn seiza_satellite_catalog_free(catalog: *mut SeizaSatelliteCatalog) {
    if !catalog.is_null() {
        unsafe { drop(Box::from_raw(catalog)) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        seiza_cancel_signal_cancel, seiza_cancel_signal_create, seiza_cancel_signal_free,
        seiza_string_free,
    };
    use seiza::wcs::Wcs;
    use serde_json::{Value, json};
    use std::ffi::{CStr, CString};

    const ISS_TLE: &str = "ISS (ZARYA)
1 25544U 98067A   24123.50000000  .00016717  00000-0  30126-3 0  9990
2 25544  51.6400 160.0000 0005000  80.0000 280.0000 15.50000000450000
";

    const ISS_OMM: &str = r#"[{
        "OBJECT_NAME":"ISS (ZARYA)", "OBJECT_ID":"1998-067A",
        "EPOCH":"2024-05-02T12:00:00.000000", "MEAN_MOTION":15.5,
        "ECCENTRICITY":0.0005, "INCLINATION":51.64, "RA_OF_ASC_NODE":160.0,
        "ARG_OF_PERICENTER":80.0, "MEAN_ANOMALY":280.0, "NORAD_CAT_ID":25544,
        "BSTAR":0.0001
    }]"#;

    /// The elements' epoch, 2024 day 123.5: 2024-05-02T12:00:00Z.
    const EPOCH_UNIX: f64 = 1_714_651_200.0;

    fn text(value: &str) -> CString {
        CString::new(value).unwrap()
    }

    fn owned_value(json: *mut c_char) -> Value {
        assert!(!json.is_null());
        let value = serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap());
        unsafe { seiza_string_free(json) };
        value.unwrap()
    }

    fn take_error(error: *mut c_char) -> String {
        assert!(!error.is_null());
        let message = unsafe { CStr::from_ptr(error) }
            .to_str()
            .unwrap()
            .to_owned();
        unsafe { seiza_string_free(error) };
        message
    }

    fn info(catalog: *const SeizaSatelliteCatalog) -> Value {
        let mut error = ptr::null_mut();
        let json = unsafe { seiza_satellite_catalog_info_json(catalog, &mut error) };
        assert!(error.is_null());
        owned_value(json)
    }

    fn iss() -> *mut SeizaSatelliteCatalog {
        let (tle, source) = (text(ISS_TLE), text("test"));
        let mut error = ptr::null_mut();
        let catalog = unsafe {
            seiza_satellite_catalog_from_tle_text(tle.as_ptr(), source.as_ptr(), &mut error)
        };
        assert!(error.is_null());
        assert!(!catalog.is_null());
        catalog
    }

    fn wcs_json(wcs: &Wcs) -> CString {
        text(
            &json!({
                "crval": [wcs.crval.0, wcs.crval.1],
                "crpix": [wcs.crpix.0, wcs.crpix.1],
                "cd": wcs.cd,
                "pixelOrigin": 0,
            })
            .to_string(),
        )
    }

    fn tracks(
        catalog: *const SeizaSatelliteCatalog,
        wcs: &Wcs,
        request: Value,
    ) -> Result<Value, String> {
        let (wcs, request) = (wcs_json(wcs), text(&request.to_string()));
        let mut error = ptr::null_mut();
        let json = unsafe {
            seiza_satellite_catalog_tracks_json(
                catalog,
                wcs.as_ptr(),
                1024,
                1024,
                request.as_ptr(),
                &mut error,
            )
        };
        if json.is_null() {
            Err(take_error(error))
        } else {
            assert!(error.is_null());
            Ok(owned_value(json))
        }
    }

    fn observer(mut request: Value) -> Value {
        let fields = request.as_object_mut().unwrap();
        fields.insert("latitude".into(), json!(42.466));
        fields.insert("longitude".into(), json!(-71.1516));
        fields.insert("altitudeM".into(), json!(150.0));
        request
    }

    /// The ISS's sky position at the epoch, found through whole-sky WCSs.
    fn locate_iss(catalog: *const SeizaSatelliteCatalog) -> (f64, f64) {
        for center_ra in [0.0, 90.0, 180.0, 270.0] {
            for center_dec in [-45.0, 45.0] {
                let wide = Wcs::from_center_scale_rotation(
                    (center_ra, center_dec),
                    (512.0, 512.0),
                    3600.0,
                    0.0,
                    false,
                );
                let request = observer(json!({
                    "start": EPOCH_UNIX,
                    "durationSeconds": 2.0,
                    "minimumElevationDeg": -90.0,
                }));
                let result = tracks(catalog, &wide, request).unwrap();
                if let Some(track) = result["tracks"].as_array().unwrap().first() {
                    let sample = &track["samples"][0];
                    return (
                        sample["raDeg"].as_f64().unwrap(),
                        sample["decDeg"].as_f64().unwrap(),
                    );
                }
            }
        }
        panic!("the ISS never fell inside a whole-sky WCS");
    }

    #[test]
    fn text_and_files_make_catalogs_with_no_load_details() {
        let catalog = iss();
        assert_eq!(
            info(catalog),
            json!({
                "records": 1, "source": "test", "retrievedAtUnix": null,
                "provider": null, "cacheState": null, "cachePath": null, "warning": null,
            })
        );
        unsafe { seiza_satellite_catalog_free(catalog) };

        let omm = text(ISS_OMM);
        let mut error = ptr::null_mut();
        let catalog =
            unsafe { seiza_satellite_catalog_from_omm_json(omm.as_ptr(), ptr::null(), &mut error) };
        assert!(error.is_null());
        assert_eq!(info(catalog)["source"], "inline");
        unsafe { seiza_satellite_catalog_free(catalog) };

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("elements.tle");
        std::fs::write(&path, ISS_TLE).unwrap();
        let path = text(path.to_str().unwrap());
        let catalog = unsafe { seiza_satellite_catalog_open(path.as_ptr(), &mut error) };
        assert!(error.is_null());
        let described = info(catalog);
        assert_eq!(described["records"], 1);
        assert_eq!(described["source"], path.to_str().unwrap());
        unsafe { seiza_satellite_catalog_free(catalog) };

        let empty = text("not elements");
        let catalog = unsafe {
            seiza_satellite_catalog_from_tle_text(empty.as_ptr(), ptr::null(), &mut error)
        };
        assert!(catalog.is_null());
        assert!(take_error(error).contains("no records"));
        unsafe { seiza_satellite_catalog_free(ptr::null_mut()) };
    }

    #[test]
    fn fetch_and_resolve_reuse_a_fresh_cache_without_the_network() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("celestrak-active-manual.json"),
            ISS_OMM,
        )
        .unwrap();
        let cache_dir = text(directory.path().to_str().unwrap());
        let mut error = ptr::null_mut();

        let catalog = unsafe {
            seiza_satellite_catalog_fetch_celestrak(cache_dir.as_ptr(), ptr::null(), &mut error)
        };
        assert!(error.is_null());
        let described = info(catalog);
        assert_eq!(described["records"], 1);
        assert_eq!(described["provider"], "celestrak-active");
        assert_eq!(described["cacheState"], "fresh");
        assert!(
            described["cachePath"]
                .as_str()
                .unwrap()
                .ends_with("manual.json")
        );
        assert!(described["retrievedAtUnix"].is_f64());
        unsafe { seiza_satellite_catalog_free(catalog) };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        let request = text(&json!({"time": now, "cacheDir": directory.path()}).to_string());
        let catalog = unsafe {
            seiza_satellite_catalog_resolve_json(request.as_ptr(), ptr::null(), &mut error)
        };
        assert!(error.is_null());
        let described = info(catalog);
        assert_eq!(described["provider"], "celestrak-active");
        assert_eq!(described["cacheState"], "fresh");
        unsafe { seiza_satellite_catalog_free(catalog) };

        let request = text(r#"{"time": "not a time"}"#);
        let catalog = unsafe {
            seiza_satellite_catalog_resolve_json(request.as_ptr(), ptr::null(), &mut error)
        };
        assert!(catalog.is_null());
        assert!(take_error(error).contains("invalid UTC timestamp"));
    }

    #[test]
    fn a_raised_cancel_signal_stops_a_fetch() {
        let directory = tempfile::tempdir().unwrap();
        let cache_dir = text(directory.path().to_str().unwrap());
        let cancel = seiza_cancel_signal_create();
        unsafe { seiza_cancel_signal_cancel(cancel) };
        let mut error = ptr::null_mut();
        let catalog = unsafe {
            seiza_satellite_catalog_fetch_celestrak(cache_dir.as_ptr(), cancel, &mut error)
        };
        assert!(catalog.is_null());
        assert!(take_error(error).contains("cancelled"));
        unsafe { seiza_cancel_signal_free(cancel) };
    }

    #[test]
    fn predicts_a_crossing_through_a_matching_wcs() {
        let catalog = iss();
        let (ra, dec) = locate_iss(catalog);
        let wcs = Wcs::from_center_scale_rotation((ra, dec), (512.0, 512.0), 10.0, 0.0, false);
        let request = observer(json!({
            "start": EPOCH_UNIX,
            "durationSeconds": 2.0,
            "minimumElevationDeg": -90.0,
        }));
        let result = tracks(catalog, &wcs, request.clone()).unwrap();
        assert_eq!(result["elementsConsidered"], 1);
        assert_eq!(result["propagationFailures"], 0);
        assert_eq!(result["staleElements"], 0);
        let [track] = result["tracks"].as_array().unwrap().as_slice() else {
            panic!("expected one track: {result}");
        };
        assert_eq!(track["noradId"], 25544);
        assert_eq!(track["cosparId"], "1998-067A");
        assert_eq!(track["name"], "ISS (ZARYA)");
        assert_eq!(track["label"], "ISS (ZARYA) [25544]");
        assert_eq!(track["source"], "test");
        assert!((track["elementEpochUnix"].as_f64().unwrap() - EPOCH_UNIX).abs() < 1.0);
        assert!(track["clippedLengthPx"].as_f64().unwrap() > 0.0);
        assert!(track["maxApparentRateArcsecPerS"].as_f64().unwrap() > 0.0);
        let segment = &track["clippedSegments"][0];
        assert!(segment["start"]["x"].is_f64() && segment["end"]["y"].is_f64());
        let samples = track["samples"].as_array().unwrap();
        assert!(samples.len() >= 2);
        assert!(samples.iter().all(|sample| {
            let sunlight = sample["sunlightFraction"].as_f64().unwrap();
            (0.0..=1.0).contains(&sunlight) && sample["pixel"]["x"].is_f64()
        }));

        // The shutter's close in place of its length gives the same tracks.
        let mut by_end = request.clone();
        by_end.as_object_mut().unwrap().remove("durationSeconds");
        by_end["end"] = json!("2024-05-02T12:00:02Z");
        assert_eq!(tracks(catalog, &wcs, by_end).unwrap(), result);

        // A month later the elements are stale unless any age is allowed.
        let mut later = request;
        later["start"] = json!(EPOCH_UNIX + 30.0 * 86_400.0);
        let stale = tracks(catalog, &wcs, later.clone()).unwrap();
        assert_eq!(stale["staleElements"], 1);
        later["maximumElementAgeSeconds"] = Value::Null;
        assert_eq!(tracks(catalog, &wcs, later).unwrap()["staleElements"], 0);

        unsafe { seiza_satellite_catalog_free(catalog) };
    }

    #[test]
    fn a_tracks_request_needs_one_observer_and_one_exposure_length() {
        let catalog = iss();
        let wcs = Wcs::from_center_scale_rotation((0.0, 0.0), (512.0, 512.0), 10.0, 0.0, false);
        let no_observer = json!({"start": EPOCH_UNIX, "durationSeconds": 2.0});
        assert!(
            tracks(catalog, &wcs, no_observer)
                .unwrap_err()
                .contains("observer")
        );
        let mixed = json!({
            "start": EPOCH_UNIX, "durationSeconds": 2.0,
            "latitude": 42.0, "observerItrf": [1.0, 2.0, 3.0],
        });
        assert!(
            tracks(catalog, &wcs, mixed)
                .unwrap_err()
                .contains("not a mixture")
        );
        let both = observer(json!({
            "start": EPOCH_UNIX, "durationSeconds": 2.0, "end": EPOCH_UNIX + 2.0,
        }));
        assert!(
            tracks(catalog, &wcs, both)
                .unwrap_err()
                .contains("exactly one")
        );
        let unknown = observer(json!({
            "start": EPOCH_UNIX, "durationSeconds": 2.0, "elevation": 3,
        }));
        assert!(
            tracks(catalog, &wcs, unknown)
                .unwrap_err()
                .contains("unknown field")
        );
        let itrf = json!({
            "start": EPOCH_UNIX, "durationSeconds": 2.0,
            "observerItrf": [1_529_000.0, -4_465_000.0, 4_275_000.0],
        });
        assert_eq!(
            tracks(catalog, &wcs, itrf).unwrap()["elementsConsidered"],
            1
        );
        unsafe { seiza_satellite_catalog_free(catalog) };
    }
}
