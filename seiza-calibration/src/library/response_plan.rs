//! Raw flat sessions, their response identities and pedestals: what a host
//! measures before flat grouping can compare anything. The planning opens
//! no file; hosts read the flats, build each response with
//! [`metric::build_response`], and keep the row [`FlatSessionPlan::measured_row`]
//! gives.
//!
//! Extracted from theatrus/psf-guard at 79f5e31d, `calibration/flat_groups/job.rs`.
//! Copyright the PSF Guard contributors; Apache-2.0.
use super::*;
use crate::flat_response::{groups, metric};

/// One flat session about to be measured: one night of one family, frames
/// whose rotator angle lies within [`groups::ROTATION_CLUSTER_DEG`] of the
/// session's first.
#[derive(Clone, Debug)]
pub struct FlatSessionPlan {
    pub session_key: String,
    pub family_key: String,
    pub rig_uuid: String,
    pub night: String,
    pub filter: Option<String>,
    pub rotation: Option<f64>,
    /// In capture order.
    pub frames: Vec<CalibrationFrame>,
}

impl FlatSessionPlan {
    /// The frames the response combines: at most [`metric::MAX_FRAMES`],
    /// evenly spaced.
    pub fn combined(&self) -> Vec<&CalibrationFrame> {
        let n = self.frames.len();
        if n <= metric::MAX_FRAMES {
            return self.frames.iter().collect();
        }
        (0..metric::MAX_FRAMES)
            .map(|i| &self.frames[i * (n - 1) / (metric::MAX_FRAMES - 1)])
            .collect()
    }

    /// The response's identity: the frames it combines and their sources,
    /// with the response algorithm and binning that read them.
    pub fn response_key(&self) -> String {
        let mut parts: Vec<String> = self
            .combined()
            .iter()
            .map(|frame| format!("{}:{}", frame.frame_uuid, frame.source_fingerprint))
            .collect();
        parts.sort();
        short_hex(&format!(
            "{}\u{1f}{}\u{1f}{}",
            metric::RESPONSE_ALGORITHM_VERSION,
            metric::MAX_SIDE,
            parts.join(",")
        ))
    }

    /// Whether the session holds enough flats to make a response.
    pub fn measurable(&self) -> bool {
        self.frames.len() >= metric::MIN_FRAMES
    }

    /// The row a planner shows for this session before any flat is read:
    /// its identity and pedestal, no measurement and no error.
    pub fn pending_row(&self, pedestal: Option<f64>) -> groups::ResponseRow {
        groups::ResponseRow {
            bin: None,
            error: None,
            ..self.measured_row(pedestal, ResponseRead::default())
        }
    }

    /// The row a host keeps for this session once it has read its
    /// [`FlatSessionPlan::combined`] flats.
    pub fn measured_row(&self, pedestal: Option<f64>, read: ResponseRead<'_>) -> groups::ResponseRow {
        let ResponseRead {
            read,
            bin,
            response,
            unread,
        } = read;
        let combined = self.combined().len();
        let error = match response {
            Some(_) => None,
            None if read < metric::MIN_FRAMES && !unread.is_empty() => Some(format!(
                "{} of {combined} flats could not be read: {}",
                unread.len(),
                unread[0].reason
            )),
            None => Some("the flats could not be combined".into()),
        };
        groups::ResponseRow {
            response_key: self.response_key(),
            session_key: self.session_key.clone(),
            rig_uuid: self.rig_uuid.clone(),
            family_key: self.family_key.clone(),
            night: self.night.clone(),
            filter: self.filter.clone(),
            rotation: self.rotation,
            frame_uuids: self.frames.iter().map(|f| f.frame_uuid.clone()).collect(),
            frames: read,
            captured_first: self.frames.first().and_then(|f| f.captured_at),
            captured_last: self.frames.last().and_then(|f| f.captured_at),
            level_adu: response.map(|r| f64::from(r.level)),
            pedestal_adu: pedestal,
            noise: response.map(|r| f64::from(r.noise)),
            width: response.map(|r| r.plane.width),
            height: response.map(|r| r.plane.height),
            bin: Some(bin),
            algorithm_version: metric::RESPONSE_ALGORITHM_VERSION,
            error,
            unread,
        }
    }
}

/// What a host read of a session's flats.
#[derive(Clone, Debug)]
pub struct ResponseRead<'a> {
    /// The planes read.
    pub read: usize,
    /// The block factor they were binned by.
    pub bin: usize,
    /// The response they built, or `None` when they would not combine.
    pub response: Option<&'a metric::FlatResponse>,
    /// The flats that would not read, and why.
    pub unread: Vec<groups::UnreadFlat>,
}
impl Default for ResponseRead<'_> {
    fn default() -> Self {
        Self {
            read: 0,
            bin: 1,
            response: None,
            unread: Vec::new(),
        }
    }
}

/// The first 16 bytes of a SHA-256, as hex: the length every flat
/// grouping key has.
fn short_hex(text: &str) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(text.as_bytes())
        .iter()
        .take(16)
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The family a flat belongs to, rotation aside: rig, filter, focal length
/// class (a reducer swapped in at the same gain is another light path, not
/// new dust; see [`focal_classes`]), frame size, binning and sensor mode.
pub fn family_key(frame: &CalibrationFrame, focal: Option<f64>) -> String {
    short_hex(&format!(
        "{}\u{1f}{}\u{1f}{:?}\u{1f}{:?}\u{1f}{:?}\u{1f}{:?}\u{1f}{:?}\u{1f}{:?}\u{1f}{:?}\u{1f}{}",
        frame.rig_uuid,
        frame
            .filter
            .as_deref()
            .map(groups::model::canonical_filter)
            .unwrap_or_default(),
        focal.map(|mm| (mm * 10.0).round() as i64),
        frame.width,
        frame.height,
        frame.binning_x,
        frame.binning_y,
        frame.gain,
        frame.offset,
        frame
            .readout_mode_name
            .as_deref()
            .map(|name| name.trim().to_ascii_lowercase())
            .unwrap_or_default(),
    ))
}

/// Each flat's focal length class, by frame uuid: per rig, recorded focal
/// lengths chained while each is within `tolerance(rig)` of the next (what
/// selection holds one master's flats to), named by the class's shortest.
/// A flat that records none joins the rig's only class when it has
/// exactly one, and is a class of its own otherwise.
pub fn focal_classes<'a>(
    flats: impl IntoIterator<Item = &'a CalibrationFrame>,
    tolerance: impl Fn(&str) -> f64,
) -> HashMap<String, Option<f64>> {
    let mut by_rig: BTreeMap<&str, Vec<&CalibrationFrame>> = BTreeMap::new();
    for frame in flats {
        by_rig
            .entry(frame.rig_uuid.as_str())
            .or_default()
            .push(frame);
    }
    let mut out = HashMap::new();
    for (rig, frames) in by_rig {
        let tolerance = tolerance(rig);
        let mut known: Vec<f64> = frames
            .iter()
            .filter_map(|frame| frame.focal_length_mm.filter(|mm| mm.is_finite()))
            .collect();
        known.sort_by(f64::total_cmp);
        known.dedup();
        // The shortest of each chain.
        let mut classes: Vec<f64> = Vec::new();
        for (i, mm) in known.iter().enumerate() {
            if i == 0 || mm - known[i - 1] > tolerance {
                classes.push(*mm);
            }
        }
        let class_of = |mm: f64| classes.iter().rev().find(|start| **start <= mm).copied();
        for frame in frames {
            let class = match frame.focal_length_mm.filter(|mm| mm.is_finite()) {
                Some(mm) => class_of(mm),
                None if classes.len() == 1 => Some(classes[0]),
                None => None,
            };
            out.insert(frame.frame_uuid.clone(), class);
        }
    }
    out
}

/// The flat sessions in use: raw flats that serve and record a capture
/// time, one night of one family each, frames whose rotator angle lies
/// within [`groups::ROTATION_CLUSTER_DEG`] of the session's first.
pub fn plan_flat_sessions(
    frames: &[CalibrationFrame],
    nights: &dyn NightSplit,
    rules: &RuleBook,
) -> Vec<FlatSessionPlan> {
    let mut flats: Vec<&CalibrationFrame> = frames
        .iter()
        .filter(|frame| {
            frame.kind == CalibrationKind::Flat
                && !frame.is_master
                && frame.kept_out.is_none()
                && frame.captured_at.is_some()
        })
        .collect();
    flats.sort_by_key(|frame| (frame.captured_at, frame.id));
    let focal = focal_classes(flats.iter().copied(), |rig| {
        rules.for_rig(rig).coherence_tolerances().focal_length_mm
    });
    let mut sessions: Vec<FlatSessionPlan> = Vec::new();
    for frame in flats {
        let night = nights.night_of_frame(
            &frame.rig_uuid,
            &frame.frame_uuid,
            frame.captured_at.unwrap_or_default(),
        );
        let family = family_key(frame, focal.get(&frame.frame_uuid).copied().flatten());
        let joins = |session: &FlatSessionPlan| {
            session.family_key == family
                && session.night == night
                && match (session.rotation, frame.rotation) {
                    (None, None) => true,
                    (Some(a), Some(b)) => rotation_gap(a, b) <= groups::ROTATION_CLUSTER_DEG,
                    _ => false,
                }
        };
        match sessions.iter_mut().find(|session| joins(session)) {
            Some(session) => session.frames.push(frame.clone()),
            None => sessions.push(FlatSessionPlan {
                session_key: short_hex(&format!("{family}\u{1f}{night}\u{1f}{}", frame.frame_uuid)),
                family_key: family,
                rig_uuid: frame.rig_uuid.clone(),
                night,
                filter: frame.filter.clone(),
                rotation: frame.rotation,
                frames: vec![frame.clone()],
            }),
        }
    }
    sessions
}

/// A rig's sensor setting: its UUID, gain and offset.
type SensorSetting = (String, Option<i64>, Option<i64>);
/// One measured level: the frame's kind, exposure and level in ADU.
type Level = (CalibrationKind, Option<f64>, f64);

/// One measured bias or dark-flat level, for [`PedestalTable`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PedestalLevel<'a> {
    pub rig_uuid: &'a str,
    pub gain: Option<i64>,
    pub offset: Option<i64>,
    pub kind: CalibrationKind,
    pub exposure_s: Option<f64>,
    /// In ADU.
    pub level: f64,
}

/// Bias and dark-flat levels by rig, gain and offset: what lies under a
/// session's flats, and under the lights whose evidence is fitted.
#[derive(Clone, Debug, Default)]
pub struct PedestalTable {
    rows: HashMap<SensorSetting, Vec<Level>>,
}

impl PedestalTable {
    /// The measured level of every raw bias and dark-flat.
    pub fn new(frames: &[CalibrationFrame], quality: &QualityIndex) -> Self {
        let mut table = Self::default();
        for frame in frames.iter().filter(|frame| {
            matches!(
                frame.kind,
                CalibrationKind::Bias | CalibrationKind::DarkFlat
            ) && !frame.is_master
        }) {
            if let Some(StoredMeasurement::Dark(measurement)) = quality
                .get(&frame.frame_uuid)
                .and_then(|row| row.measurement.as_ref())
            {
                table.insert(PedestalLevel {
                    rig_uuid: &frame.rig_uuid,
                    gain: frame.gain,
                    offset: frame.offset,
                    kind: frame.kind,
                    exposure_s: frame.exposure_s,
                    level: f64::from(measurement.level),
                });
            }
        }
        table
    }

    /// Add one measured bias or dark-flat level.
    pub fn insert(&mut self, level: PedestalLevel<'_>) {
        self.rows
            .entry((level.rig_uuid.to_string(), level.gain, level.offset))
            .or_default()
            .push((level.kind, level.exposure_s, level.level));
    }

    /// The pedestal under lights of a rig at this gain and offset: the
    /// median of the rig's biases there, or nothing. A dark-flat measures
    /// a flat's exposure, not a light's.
    pub fn light_pedestal(&self, rig: &str, gain: Option<i64>, offset: Option<i64>) -> Option<f64> {
        self.flat_pedestal(rig, gain, offset, None)
    }

    /// The pedestal under flats of a rig at this gain, offset and
    /// exposure: the median of the dark-flats within 5% of the exposure,
    /// else the median of the rig's biases there, else nothing.
    pub fn flat_pedestal(
        &self,
        rig: &str,
        gain: Option<i64>,
        offset: Option<i64>,
        exposure: Option<f64>,
    ) -> Option<f64> {
        let rows = self.rows.get(&(rig.to_string(), gain, offset))?;
        let dark_flats: Vec<f64> = rows
            .iter()
            .filter(|(kind, e, _)| {
                *kind == CalibrationKind::DarkFlat
                    && match (e, exposure) {
                        (Some(a), Some(b)) => (a - b).abs() <= 0.05 * a.max(b).max(0.1),
                        _ => false,
                    }
            })
            .map(|(_, _, level)| *level)
            .collect();
        median(dark_flats).or_else(|| {
            median(
                rows.iter()
                    .filter(|(kind, _, _)| *kind == CalibrationKind::Bias)
                    .map(|(_, _, level)| *level)
                    .collect(),
            )
        })
    }

    /// The pedestal under one session's flats, by its first.
    pub fn for_session(&self, plan: &FlatSessionPlan) -> Option<f64> {
        let first = plan.frames.first()?;
        self.flat_pedestal(&plan.rig_uuid, first.gain, first.offset, first.exposure_s)
    }
}

/// Whether a kept response's pedestal moved enough to measure it again:
/// a dark-flat or bias that arrived since, or went, or a level more than
/// half an ADU away. Its comparisons and dust were made with the old one.
pub fn pedestal_moved(kept: Option<f64>, now: Option<f64>) -> bool {
    match (kept, now) {
        (Some(a), Some(b)) => (a - b).abs() > 0.5,
        (a, b) => a.is_some() != b.is_some(),
    }
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[values.len() / 2])
}
