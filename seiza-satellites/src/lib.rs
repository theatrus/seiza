//! Optional, provenance-bearing satellite track prediction for solved images.
//!
//! This crate consumes one [`SingleExposure`], an observer location, orbital
//! elements, and a Seiza WCS solution. Results are predictions only; callers
//! must not present them as pixel detections without a separate evidence
//! matcher.

mod source;

pub use source::{
    CELESTRAK_ACTIVE_OMM_URL, CELESTRAK_MIN_REFRESH, CacheState, CelesTrakLoad, CelesTrakSource,
};

use satkit::frametransform::{qitrf2gcrf_approx, qteme2gcrf, qteme2itrf};
use satkit::lpephem::sun::{pos_gcrf as sun_position_gcrf, shadowfunc};
use satkit::omm::OMM;
use satkit::sgp4::{SGP4Error, SGP4State, sgp4};
use satkit::{ITRFCoord, Instant, TLE, Vector3};
use seiza::wcs::Wcs;
use std::path::{Path, PathBuf};

const MAX_SAMPLE_SEGMENTS: usize = 1_000_000;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("invalid UTC timestamp {value:?}: {message}")]
    InvalidTimestamp { value: String, message: String },
    #[error("invalid single exposure: {0}")]
    InvalidExposure(&'static str),
    #[error("invalid observer location: {0}")]
    InvalidObserver(&'static str),
    #[error("invalid track options: {0}")]
    InvalidOptions(&'static str),
    #[error("failed to read {}: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse satellite elements from {source_name}: {message}")]
    Elements {
        source_name: String,
        message: String,
    },
    #[error("satellite element source contains no records: {0}")]
    EmptyElements(String),
    #[error("satellite propagation failed: {0}")]
    Propagation(String),
    #[error("failed to initialize HTTP client: {0}")]
    HttpClient(#[source] reqwest::Error),
    #[error("failed to fetch {url}: {source}")]
    Http {
        url: String,
        #[source]
        source: reqwest::Error,
    },
    #[error("{url} returned HTTP {status}")]
    HttpStatus { url: String, status: u16 },
    #[error("no platform cache directory is available")]
    NoCacheDirectory,
}

pub type Result<T> = std::result::Result<T, Error>;

/// A UTC timestamp represented as Unix seconds, including a fractional part.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct UtcTimestamp(f64);

impl UtcTimestamp {
    pub fn from_unix_seconds(seconds: f64) -> Result<Self> {
        if !seconds.is_finite() {
            return Err(Error::InvalidTimestamp {
                value: seconds.to_string(),
                message: "Unix seconds must be finite".into(),
            });
        }
        Ok(Self(seconds))
    }

    /// Parse an RFC 3339 timestamp. A FITS-style timestamp without a suffix is
    /// interpreted as UTC; explicit non-zero offsets are normalized to UTC.
    pub fn parse(value: &str) -> Result<Self> {
        let instant = Instant::from_rfc3339(value).map_err(|error| Error::InvalidTimestamp {
            value: value.into(),
            message: error.to_string(),
        })?;
        Self::from_unix_seconds(instant.as_unixtime())
    }

    pub fn unix_seconds(self) -> f64 {
        self.0
    }

    pub fn to_rfc3339(self) -> String {
        self.instant().as_rfc3339()
    }

    pub fn add_seconds(self, seconds: f64) -> Result<Self> {
        Self::from_unix_seconds(self.0 + seconds)
    }

    pub fn seconds_since(self, earlier: Self) -> f64 {
        self.0 - earlier.0
    }

    fn instant(self) -> Instant {
        Instant::from_unixtime(self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExposureProvenance {
    Explicit,
    FitsBounds,
    FitsDateObsAndExposure,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ObserverLocation {
    /// Geodetic latitude, east-positive longitude, and height above ellipsoid.
    Geodetic {
        latitude_deg: f64,
        longitude_deg: f64,
        altitude_m: f64,
    },
    /// ITRF Cartesian coordinates in meters.
    ItrfMeters { x: f64, y: f64, z: f64 },
}

impl ObserverLocation {
    pub fn geodetic(latitude_deg: f64, longitude_deg: f64, altitude_m: f64) -> Result<Self> {
        if !latitude_deg.is_finite() || !(-90.0..=90.0).contains(&latitude_deg) {
            return Err(Error::InvalidObserver(
                "latitude must be finite and between -90 and 90 degrees",
            ));
        }
        if !longitude_deg.is_finite() || !altitude_m.is_finite() {
            return Err(Error::InvalidObserver(
                "longitude and altitude must be finite",
            ));
        }
        Ok(Self::Geodetic {
            latitude_deg,
            longitude_deg: normalize_longitude(longitude_deg),
            altitude_m,
        })
    }

    pub fn itrf_meters(x: f64, y: f64, z: f64) -> Result<Self> {
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return Err(Error::InvalidObserver("ITRF coordinates must be finite"));
        }
        if x == 0.0 && y == 0.0 && z == 0.0 {
            return Err(Error::InvalidObserver(
                "ITRF coordinates cannot be the center of Earth",
            ));
        }
        Ok(Self::ItrfMeters { x, y, z })
    }

    fn itrf(self) -> ITRFCoord {
        match self {
            Self::Geodetic {
                latitude_deg,
                longitude_deg,
                altitude_m,
            } => ITRFCoord::from_geodetic_deg(latitude_deg, longitude_deg, altitude_m),
            Self::ItrfMeters { x, y, z } => ITRFCoord::from([x, y, z]),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SingleExposure {
    pub start_utc: UtcTimestamp,
    pub end_utc: UtcTimestamp,
    pub observer: ObserverLocation,
    pub provenance: ExposureProvenance,
}

impl SingleExposure {
    pub fn new(
        start_utc: UtcTimestamp,
        end_utc: UtcTimestamp,
        observer: ObserverLocation,
        provenance: ExposureProvenance,
    ) -> Result<Self> {
        if end_utc <= start_utc {
            return Err(Error::InvalidExposure(
                "end time must be later than start time",
            ));
        }
        Ok(Self {
            start_utc,
            end_utc,
            observer,
            provenance,
        })
    }

    pub fn from_start_and_duration(
        start_utc: UtcTimestamp,
        duration_seconds: f64,
        observer: ObserverLocation,
        provenance: ExposureProvenance,
    ) -> Result<Self> {
        if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
            return Err(Error::InvalidExposure(
                "duration must be a positive finite number of seconds",
            ));
        }
        Self::new(
            start_utc,
            start_utc.add_seconds(duration_seconds)?,
            observer,
            provenance,
        )
    }

    pub fn duration_seconds(self) -> f64 {
        self.end_utc.seconds_since(self.start_utc)
    }

    pub fn midpoint(self) -> UtcTimestamp {
        UtcTimestamp(self.start_utc.0 + self.duration_seconds() / 2.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SatelliteIdentity {
    pub norad_id: Option<u32>,
    pub cospar_id: Option<String>,
    pub name: String,
}

impl SatelliteIdentity {
    pub fn display_label(&self) -> String {
        match self.norad_id {
            Some(id) if self.name.is_empty() => format!("NORAD {id}"),
            Some(id) => format!("{} [{id}]", self.name),
            None if !self.name.is_empty() => self.name.clone(),
            None => self
                .cospar_id
                .clone()
                .unwrap_or_else(|| "unidentified satellite".into()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackAssociation {
    Predicted,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PixelPoint {
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PixelSegment {
    pub start: PixelPoint,
    pub end: PixelPoint,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrackSample {
    pub time_utc: UtcTimestamp,
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub pixel: Option<PixelPoint>,
    pub elevation_deg: f64,
    pub range_km: f64,
    /// Fraction of the solar disk visible from the satellite, from 0 to 1.
    pub sunlight_fraction: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SatelliteTrack {
    pub identity: SatelliteIdentity,
    pub association: TrackAssociation,
    pub element_epoch_utc: UtcTimestamp,
    /// Signed age at exposure midpoint; positive means the exposure follows
    /// the element epoch.
    pub element_age_seconds: f64,
    pub source: String,
    pub sample_interval_seconds: f64,
    pub samples: Vec<TrackSample>,
    pub clipped_segments: Vec<PixelSegment>,
}

impl SatelliteTrack {
    pub fn maximum_elevation_deg(&self) -> f64 {
        self.samples
            .iter()
            .map(|sample| sample.elevation_deg)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn maximum_sunlight_fraction(&self) -> f64 {
        self.samples
            .iter()
            .map(|sample| sample.sunlight_fraction)
            .fold(0.0, f64::max)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackOptions {
    /// Final sampling interval for tracks intersecting the image.
    pub sample_interval_seconds: f64,
    /// Initial interval used to reject non-intersecting tracks cheaply.
    pub coarse_interval_seconds: f64,
    /// Samples below this apparent elevation are excluded from image paths.
    pub minimum_elevation_deg: f64,
    /// Maximum absolute separation between element epoch and exposure
    /// midpoint. `None` deliberately allows unrestricted extrapolation.
    pub maximum_element_age_seconds: Option<f64>,
}

impl Default for TrackOptions {
    fn default() -> Self {
        Self {
            sample_interval_seconds: 1.0,
            coarse_interval_seconds: 10.0,
            minimum_elevation_deg: 0.0,
            maximum_element_age_seconds: Some(7.0 * 24.0 * 60.0 * 60.0),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrackSearchResult {
    pub tracks: Vec<SatelliteTrack>,
    pub elements_considered: usize,
    pub propagation_failures: usize,
    pub stale_elements: usize,
}

enum ElementRecord {
    Omm(Box<OMM>),
    Tle(Box<TLE>),
}

impl ElementRecord {
    fn identity(&self) -> SatelliteIdentity {
        match self {
            Self::Omm(omm) => SatelliteIdentity {
                norad_id: omm.norad_cat_id,
                cospar_id: nonempty(&omm.object_id),
                name: omm.object_name.trim().to_string(),
            },
            Self::Tle(tle) => SatelliteIdentity {
                norad_id: u32::try_from(tle.sat_num).ok(),
                cospar_id: tle_cospar_id(tle),
                name: tle.name.trim().trim_start_matches("0 ").to_string(),
            },
        }
    }

    fn epoch(&self) -> UtcTimestamp {
        let instant = match self {
            Self::Omm(omm) => omm
                .epoch_instant()
                .expect("OMM epoch was validated while opening the catalog"),
            Self::Tle(tle) => tle.epoch,
        };
        UtcTimestamp(instant.as_unixtime())
    }

    fn propagate(&mut self, times: &[Instant]) -> std::result::Result<SGP4State, String> {
        match self {
            Self::Omm(omm) => sgp4(omm.as_mut(), times).map_err(|error| error.to_string()),
            Self::Tle(tle) => sgp4(tle.as_mut(), times).map_err(|error| error.to_string()),
        }
    }
}

/// Parsed OMM or TLE records plus their source provenance.
pub struct SatelliteCatalog {
    elements: Vec<ElementRecord>,
    source: String,
    retrieved_at: Option<UtcTimestamp>,
}

impl std::fmt::Debug for SatelliteCatalog {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SatelliteCatalog")
            .field("elements", &self.elements.len())
            .field("source", &self.source)
            .field("retrieved_at", &self.retrieved_at)
            .finish()
    }
}

impl SatelliteCatalog {
    pub fn from_omm_json(payload: &str, source: impl Into<String>) -> Result<Self> {
        let source = source.into();
        let trimmed = payload.trim();
        let wrapped;
        let payload = if trimmed.starts_with('{') {
            wrapped = format!("[{trimmed}]");
            wrapped.as_str()
        } else {
            payload
        };
        let omms = OMM::from_json_string(payload).map_err(|error| Error::Elements {
            source_name: source.clone(),
            message: error.to_string(),
        })?;
        for omm in &omms {
            omm.epoch_instant().map_err(|error| Error::Elements {
                source_name: source.clone(),
                message: format!("invalid OMM epoch {:?}: {error}", omm.epoch),
            })?;
        }
        Self::from_records(
            omms.into_iter()
                .map(|omm| ElementRecord::Omm(Box::new(omm)))
                .collect(),
            source,
        )
    }

    pub fn from_tle_text(payload: &str, source: impl Into<String>) -> Result<Self> {
        let source = source.into();
        let lines = payload.lines().map(str::to_string).collect::<Vec<_>>();
        let tles = TLE::from_lines(&lines).map_err(|error| Error::Elements {
            source_name: source.clone(),
            message: error.to_string(),
        })?;
        Self::from_records(
            tles.into_iter()
                .map(|tle| ElementRecord::Tle(Box::new(tle)))
                .collect(),
            source,
        )
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let payload = std::fs::read_to_string(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let source = path.display().to_string();
        match payload.trim_start().chars().next() {
            Some('[' | '{') => Self::from_omm_json(&payload, source),
            _ => Self::from_tle_text(&payload, source),
        }
    }

    fn from_records(elements: Vec<ElementRecord>, source: String) -> Result<Self> {
        if elements.is_empty() {
            return Err(Error::EmptyElements(source));
        }
        Ok(Self {
            elements,
            source,
            retrieved_at: None,
        })
    }

    pub fn with_retrieved_at(mut self, retrieved_at: UtcTimestamp) -> Self {
        self.retrieved_at = Some(retrieved_at);
        self
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn retrieved_at(&self) -> Option<UtcTimestamp> {
        self.retrieved_at
    }

    pub fn tracks_in_footprint(
        &mut self,
        wcs: &Wcs,
        dimensions: (u32, u32),
        exposure: &SingleExposure,
        options: &TrackOptions,
    ) -> Result<TrackSearchResult> {
        validate_options(options)?;
        if dimensions.0 == 0 || dimensions.1 == 0 {
            return Err(Error::InvalidOptions("image dimensions must be non-zero"));
        }

        let observer = exposure.observer.itrf();
        let coarse_times = sample_instants(exposure, options.coarse_interval_seconds)?;
        let fine_times = sample_instants(exposure, options.sample_interval_seconds)?;
        let considered = self.elements.len();
        let mut propagation_failures = 0usize;
        let mut stale_elements = 0usize;
        let mut tracks = Vec::new();

        for element in &mut self.elements {
            if options.maximum_element_age_seconds.is_some_and(|maximum| {
                exposure.midpoint().seconds_since(element.epoch()).abs() > maximum
            }) {
                stale_elements += 1;
                continue;
            }
            let coarse_state = match element.propagate(&coarse_times) {
                Ok(state) => state,
                Err(_) => {
                    propagation_failures += 1;
                    continue;
                }
            };
            let coarse_samples = project_samples(
                &coarse_state,
                &coarse_times,
                &observer,
                wcs,
                options.minimum_elevation_deg,
            );
            if !path_intersects_image(&coarse_samples, dimensions) {
                continue;
            }

            let fine_state = match element.propagate(&fine_times) {
                Ok(state) => state,
                Err(_) => {
                    propagation_failures += 1;
                    continue;
                }
            };
            let samples = project_samples(
                &fine_state,
                &fine_times,
                &observer,
                wcs,
                options.minimum_elevation_deg,
            );
            let clipped_segments = clipped_path(&samples, dimensions);
            if clipped_segments.is_empty() {
                continue;
            }
            let epoch = element.epoch();
            tracks.push(SatelliteTrack {
                identity: element.identity(),
                association: TrackAssociation::Predicted,
                element_epoch_utc: epoch,
                element_age_seconds: exposure.midpoint().seconds_since(epoch),
                source: self.source.clone(),
                sample_interval_seconds: actual_interval(exposure, fine_times.len()),
                samples,
                clipped_segments,
            });
        }

        tracks.sort_by(|left, right| {
            right
                .maximum_elevation_deg()
                .total_cmp(&left.maximum_elevation_deg())
                .then_with(|| left.identity.norad_id.cmp(&right.identity.norad_id))
        });
        Ok(TrackSearchResult {
            tracks,
            elements_considered: considered,
            propagation_failures,
            stale_elements,
        })
    }
}

fn validate_options(options: &TrackOptions) -> Result<()> {
    if !options.sample_interval_seconds.is_finite() || options.sample_interval_seconds <= 0.0 {
        return Err(Error::InvalidOptions(
            "sample interval must be a positive finite number of seconds",
        ));
    }
    if !options.coarse_interval_seconds.is_finite() || options.coarse_interval_seconds <= 0.0 {
        return Err(Error::InvalidOptions(
            "coarse interval must be a positive finite number of seconds",
        ));
    }
    if !options.minimum_elevation_deg.is_finite()
        || !(-90.0..=90.0).contains(&options.minimum_elevation_deg)
    {
        return Err(Error::InvalidOptions(
            "minimum elevation must be between -90 and 90 degrees",
        ));
    }
    if options
        .maximum_element_age_seconds
        .is_some_and(|seconds| !seconds.is_finite() || seconds <= 0.0)
    {
        return Err(Error::InvalidOptions(
            "maximum element age must be a positive finite number of seconds",
        ));
    }
    Ok(())
}

fn sample_instants(exposure: &SingleExposure, maximum_interval: f64) -> Result<Vec<Instant>> {
    let duration = exposure.duration_seconds();
    let requested_segments = (duration / maximum_interval).ceil().max(1.0);
    if requested_segments > MAX_SAMPLE_SEGMENTS as f64 {
        return Err(Error::InvalidOptions(
            "sample interval would create more than 1,000,001 samples for one exposure",
        ));
    }
    let segments = requested_segments as usize;
    Ok((0..=segments)
        .map(|index| {
            Instant::from_unixtime(
                exposure.start_utc.unix_seconds() + duration * index as f64 / segments as f64,
            )
        })
        .collect())
}

fn actual_interval(exposure: &SingleExposure, sample_count: usize) -> f64 {
    exposure.duration_seconds() / (sample_count.saturating_sub(1).max(1) as f64)
}

fn project_samples(
    states: &SGP4State,
    times: &[Instant],
    observer: &ITRFCoord,
    wcs: &Wcs,
    minimum_elevation_deg: f64,
) -> Vec<TrackSample> {
    times
        .iter()
        .enumerate()
        .filter_map(|(index, time)| {
            if states.errcode.get(index) != Some(&SGP4Error::SGP4Success) {
                return None;
            }
            let teme = Vector3::from_array([
                states.pos[(0, index)],
                states.pos[(1, index)],
                states.pos[(2, index)],
            ]);
            let satellite_itrf_vector = qteme2itrf(time) * teme;
            let satellite_itrf = ITRFCoord::from(satellite_itrf_vector);
            let relative_itrf = satellite_itrf - *observer;
            let range_m = relative_itrf.norm();
            if !range_m.is_finite() || range_m <= 0.0 {
                return None;
            }
            let enu = observer.q_enu2itrf().conjugate() * relative_itrf;
            let elevation_deg = (enu[2] / range_m).clamp(-1.0, 1.0).asin().to_degrees();
            let relative_gcrf = qitrf2gcrf_approx(time) * relative_itrf;
            let ra_deg = relative_gcrf[1]
                .atan2(relative_gcrf[0])
                .to_degrees()
                .rem_euclid(360.0);
            let dec_deg = (relative_gcrf[2] / range_m)
                .clamp(-1.0, 1.0)
                .asin()
                .to_degrees();
            let satellite_gcrf = qteme2gcrf(time) * teme;
            let sunlight_fraction =
                shadowfunc(&sun_position_gcrf(time), &satellite_gcrf).clamp(0.0, 1.0);
            let pixel = (elevation_deg >= minimum_elevation_deg)
                .then(|| wcs.world_to_pixel(ra_deg, dec_deg))
                .flatten()
                .map(|(x, y)| PixelPoint { x, y });
            Some(TrackSample {
                time_utc: UtcTimestamp(time.as_unixtime()),
                ra_deg,
                dec_deg,
                pixel,
                elevation_deg,
                range_km: range_m / 1000.0,
                sunlight_fraction,
            })
        })
        .collect()
}

fn path_intersects_image(samples: &[TrackSample], dimensions: (u32, u32)) -> bool {
    samples.iter().any(|sample| {
        sample
            .pixel
            .is_some_and(|point| point_inside(point, dimensions))
    }) || samples
        .windows(2)
        .any(|pair| match (pair[0].pixel, pair[1].pixel) {
            (Some(start), Some(end)) => clip_segment(start, end, dimensions).is_some(),
            _ => false,
        })
}

fn clipped_path(samples: &[TrackSample], dimensions: (u32, u32)) -> Vec<PixelSegment> {
    samples
        .windows(2)
        .filter_map(|pair| match (pair[0].pixel, pair[1].pixel) {
            (Some(start), Some(end)) => clip_segment(start, end, dimensions),
            _ => None,
        })
        .collect()
}

fn point_inside(point: PixelPoint, dimensions: (u32, u32)) -> bool {
    point.x >= 0.0
        && point.y >= 0.0
        && point.x <= f64::from(dimensions.0.saturating_sub(1))
        && point.y <= f64::from(dimensions.1.saturating_sub(1))
}

/// Liang-Barsky clipping against the inclusive image rectangle.
fn clip_segment(
    start: PixelPoint,
    end: PixelPoint,
    dimensions: (u32, u32),
) -> Option<PixelSegment> {
    if !start.x.is_finite() || !start.y.is_finite() || !end.x.is_finite() || !end.y.is_finite() {
        return None;
    }
    let max_x = f64::from(dimensions.0.saturating_sub(1));
    let max_y = f64::from(dimensions.1.saturating_sub(1));
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let mut low: f64 = 0.0;
    let mut high: f64 = 1.0;
    for (p, q) in [
        (-dx, start.x),
        (dx, max_x - start.x),
        (-dy, start.y),
        (dy, max_y - start.y),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
            continue;
        }
        let ratio = q / p;
        if p < 0.0 {
            low = low.max(ratio);
        } else {
            high = high.min(ratio);
        }
        if low > high {
            return None;
        }
    }
    Some(PixelSegment {
        start: PixelPoint {
            x: dx.mul_add(low, start.x),
            y: dy.mul_add(low, start.y),
        },
        end: PixelPoint {
            x: dx.mul_add(high, start.x),
            y: dy.mul_add(high, start.y),
        },
    })
}

fn normalize_longitude(longitude: f64) -> f64 {
    (longitude + 180.0).rem_euclid(360.0) - 180.0
}

fn nonempty(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn tle_cospar_id(tle: &TLE) -> Option<String> {
    nonempty(&tle.intl_desig)?;
    let year = if tle.desig_year >= 57 {
        1900 + tle.desig_year
    } else {
        2000 + tle.desig_year
    };
    Some(format!(
        "{year:04}-{:03}{}",
        tle.desig_launch, tle.desig_piece
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ISS_TLE: &str = include_str!("../tests/data/iss-2024.tle");

    #[test]
    fn rejects_non_positive_exposures() {
        let time = UtcTimestamp::parse("2024-05-02T12:00:00Z").unwrap();
        let observer = ObserverLocation::geodetic(37.0, -122.0, 10.0).unwrap();
        assert!(SingleExposure::new(time, time, observer, ExposureProvenance::Explicit).is_err());
        assert!(
            SingleExposure::from_start_and_duration(
                time,
                -1.0,
                observer,
                ExposureProvenance::Explicit
            )
            .is_err()
        );
    }

    #[test]
    fn parses_fits_style_utc_without_z() {
        let fits = UtcTimestamp::parse("2024-05-02T12:00:00.125").unwrap();
        let rfc = UtcTimestamp::parse("2024-05-02T12:00:00.125Z").unwrap();
        assert!((fits.seconds_since(rfc)).abs() < 1.0e-6);
    }

    #[test]
    fn opens_tle_and_preserves_identity() {
        let catalog = SatelliteCatalog::from_tle_text(ISS_TLE, "test").unwrap();
        assert_eq!(catalog.len(), 1);
        let identity = catalog.elements[0].identity();
        assert_eq!(identity.norad_id, Some(25544));
        assert_eq!(identity.cospar_id.as_deref(), Some("1998-067A"));
        assert_eq!(identity.name, "ISS (ZARYA)");
    }

    #[test]
    fn predicts_a_topocentric_track_through_a_matching_wcs() {
        let mut catalog = SatelliteCatalog::from_tle_text(ISS_TLE, "test").unwrap();
        let observer = ObserverLocation::geodetic(42.466, -71.1516, 150.0).unwrap();
        let start = catalog.elements[0].epoch();
        let exposure = SingleExposure::from_start_and_duration(
            start,
            2.0,
            observer,
            ExposureProvenance::Explicit,
        )
        .unwrap();
        let midpoint = exposure.midpoint().instant();
        let state = catalog.elements[0].propagate(&[midpoint]).unwrap();
        let seed_wcs =
            Wcs::from_center_scale_rotation((0.0, 0.0), (512.0, 512.0), 3600.0, 0.0, false);
        let sample =
            project_samples(&state, &[midpoint], &observer.itrf(), &seed_wcs, -90.0).remove(0);
        let wcs = Wcs::from_center_scale_rotation(
            (sample.ra_deg, sample.dec_deg),
            (512.0, 512.0),
            10.0,
            0.0,
            false,
        );
        let result = catalog
            .tracks_in_footprint(
                &wcs,
                (1024, 1024),
                &exposure,
                &TrackOptions {
                    minimum_elevation_deg: -90.0,
                    ..TrackOptions::default()
                },
            )
            .unwrap();
        assert_eq!(result.elements_considered, 1);
        assert_eq!(result.propagation_failures, 0);
        assert_eq!(result.stale_elements, 0);
        assert_eq!(result.tracks.len(), 1);
        assert_eq!(result.tracks[0].association, TrackAssociation::Predicted);
        assert!(!result.tracks[0].clipped_segments.is_empty());
    }

    #[test]
    fn clips_crossing_segments_to_the_image() {
        let clipped = clip_segment(
            PixelPoint { x: -10.0, y: 5.0 },
            PixelPoint { x: 20.0, y: 5.0 },
            (11, 11),
        )
        .unwrap();
        assert!(clipped.start.x.abs() < 1.0e-12);
        assert_eq!(clipped.start.y, 5.0);
        assert!((clipped.end.x - 10.0).abs() < 1.0e-12);
        assert_eq!(clipped.end.y, 5.0);
    }

    #[test]
    fn refuses_to_extrapolate_stale_elements_by_default() {
        let mut catalog = SatelliteCatalog::from_tle_text(ISS_TLE, "test").unwrap();
        let observer = ObserverLocation::geodetic(42.466, -71.1516, 150.0).unwrap();
        let start = catalog.elements[0]
            .epoch()
            .add_seconds(8.0 * 24.0 * 60.0 * 60.0)
            .unwrap();
        let exposure = SingleExposure::from_start_and_duration(
            start,
            10.0,
            observer,
            ExposureProvenance::Explicit,
        )
        .unwrap();
        let wcs = Wcs::from_center_scale_rotation((0.0, 0.0), (50.0, 50.0), 3600.0, 0.0, false);
        let result = catalog
            .tracks_in_footprint(&wcs, (100, 100), &exposure, &TrackOptions::default())
            .unwrap();
        assert_eq!(result.elements_considered, 1);
        assert_eq!(result.stale_elements, 1);
        assert!(result.tracks.is_empty());
    }

    #[test]
    fn rejects_sampling_requests_that_would_allocate_unbounded_vectors() {
        let mut catalog = SatelliteCatalog::from_tle_text(ISS_TLE, "test").unwrap();
        let observer = ObserverLocation::geodetic(42.466, -71.1516, 150.0).unwrap();
        let exposure = SingleExposure::from_start_and_duration(
            catalog.elements[0].epoch(),
            10.0,
            observer,
            ExposureProvenance::Explicit,
        )
        .unwrap();
        let wcs = Wcs::from_center_scale_rotation((0.0, 0.0), (50.0, 50.0), 3600.0, 0.0, false);
        let result = catalog.tracks_in_footprint(
            &wcs,
            (100, 100),
            &exposure,
            &TrackOptions {
                sample_interval_seconds: 1.0e-9,
                ..TrackOptions::default()
            },
        );
        assert!(matches!(result, Err(Error::InvalidOptions(_))));
    }
}
