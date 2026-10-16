//! Reading a frame's role from its header, and choosing the calibration
//! frames that suit a set of lights.
//!
//! Both bindings call these, so a host gets the whole matching policy rather
//! than rebuilding it from the single tests in `seiza-calibration`.

use crate::{
    Error, FrameMetadata, FrameSourceRole, Result, path_identity, paths_refer_to_same_file,
};
use seiza_calibration::{FrameRole, FrameSignature, MatchTolerances};
use seiza_fits::HeaderValue;
use seiza_image_io::Format;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// What a FITS or XISF header says about its frame, read without decoding
/// the pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameProbe {
    /// The file that was read.
    pub path: PathBuf,
    /// [`Format::Fits`] or [`Format::Xisf`].
    pub format: Format,
    /// The frame type as the file wrote it (`IMAGETYP`, `OBSTYPE` or
    /// `FRAME`), before it was read as a role.
    pub raw_image_type: Option<String>,
    /// The frame's role, master flag, signature and calibration state.
    pub metadata: FrameMetadata,
}

/// Read a FITS or XISF header into a [`FrameProbe`]. No pixels are decoded,
/// so a file whose header is whole but whose data is cut short still
/// answers.
pub fn probe_frame(path: &Path) -> Result<FrameProbe> {
    let format = Format::from_path(path)
        .filter(|format| format.is_astronomy())
        .ok_or_else(|| {
            Error::Calibration(format!("{} is not a FITS or XISF path", path.display()))
        })?;
    let headers = match format {
        Format::Fits => seiza_fits::read_header(path).map_err(|error| {
            Error::Calibration(format!(
                "failed to read FITS header {}: {error}",
                path.display()
            ))
        })?,
        _ => seiza_xisf::read_header(path).map_err(|error| {
            Error::Calibration(format!(
                "failed to read XISF header {}: {error}",
                path.display()
            ))
        })?,
    };
    Ok(FrameProbe {
        path: path.to_path_buf(),
        format,
        raw_image_type: header_text(&headers, &["IMAGETYP", "OBSTYPE", "FRAME"]),
        metadata: FrameMetadata::from_headers(&headers),
    })
}

fn header_text(headers: &[(String, HeaderValue)], keys: &[&str]) -> Option<String> {
    let value = keys.iter().find_map(|key| {
        headers
            .iter()
            .find(|(candidate, _)| candidate.eq_ignore_ascii_case(key))
            .map(|(_, value)| value)
    })?;
    let value = match value {
        HeaderValue::String(value) | HeaderValue::Raw(value) => value.trim(),
        _ => return None,
    };
    (!value.is_empty()).then(|| value.to_owned())
}

/// Which kind of calibration frame a plan chooses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationPlanKind {
    /// Bias frames, matched to lights by sensor.
    Bias,
    /// Darks, matched to lights by sensor, exposure and temperature.
    Dark,
    /// Dark flats, matched to the flat they calibrate as darks are to lights.
    DarkFlat,
    /// Flats, matched to lights by sensor and optics.
    Flat,
}

impl CalibrationPlanKind {
    /// The role a candidate must have.
    pub fn role(self) -> FrameSourceRole {
        match self {
            Self::Bias => FrameSourceRole::Bias,
            Self::Dark => FrameSourceRole::Dark,
            Self::DarkFlat => FrameSourceRole::DarkFlat,
            Self::Flat => FrameSourceRole::Flat,
        }
    }

    /// `bias`, `dark`, `dark-flat` or `flat`.
    pub fn as_str(self) -> &'static str {
        self.role().as_str()
    }

    fn uses_dark_matching(self) -> bool {
        matches!(self, Self::Dark | Self::DarkFlat)
    }
}

impl FromStr for CalibrationPlanKind {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "bias" => Ok(Self::Bias),
            "dark" => Ok(Self::Dark),
            "dark-flat" => Ok(Self::DarkFlat),
            "flat" => Ok(Self::Flat),
            other => Err(Error::Calibration(format!(
                "calibration plan kind must be bias, dark, dark-flat or flat, not {other:?}"
            ))),
        }
    }
}

/// One frame a plan weighs: a light or flat it serves, or a candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationRecord {
    /// The frame's file, returned as given.
    pub path: PathBuf,
    /// What the frame is.
    pub role: FrameSourceRole,
    /// What it was shot with.
    pub signature: FrameSignature,
}

/// What [`plan_calibration`] is asked.
#[derive(Clone, Debug)]
pub struct CalibrationPlanRequest {
    /// Which kind of frame to choose.
    pub kind: CalibrationPlanKind,
    /// The frame the choice serves, and whose capture time orders the
    /// candidates: a light, or for dark flats the chosen flat.
    pub reference: CalibrationRecord,
    /// Every frame the choice must serve, `reference` among them; empty
    /// means `reference` alone.
    pub references: Vec<CalibrationRecord>,
    /// The frames to choose from.
    pub candidates: Vec<CalibrationRecord>,
    /// The fewest frames a usable set holds; zero counts as one.
    pub minimum: usize,
    /// How close readings must be.
    pub tolerances: MatchTolerances,
    /// Whether a usable bias master has been built. Darks of another
    /// exposure can then be scaled, so exposure no longer has to match.
    pub bias_available: bool,
}

/// Why [`plan_calibration`] left a candidate out.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationExclusionReason {
    /// The candidate is not the kind of frame asked for.
    RoleMismatch,
    /// It came off another sensor, or the same one in another mode.
    SensorMismatch,
    /// A dark or a frame it serves does not record its exposure.
    MissingExposure,
    /// A dark's exposure differs, and no bias allows scaling it.
    ExposureMismatch,
    /// A dark's temperature differs.
    TemperatureMismatch,
    /// A flat's filter, telescope, focal length or angle differs.
    OpticsMismatch,
    /// It suits the frames served, but not the set chosen to combine.
    OutsideCoherentSet,
}

impl CalibrationExclusionReason {
    /// A stable kebab-case spelling, such as `exposure-mismatch`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RoleMismatch => "role-mismatch",
            Self::SensorMismatch => "sensor-mismatch",
            Self::MissingExposure => "missing-exposure",
            Self::ExposureMismatch => "exposure-mismatch",
            Self::TemperatureMismatch => "temperature-mismatch",
            Self::OpticsMismatch => "optics-mismatch",
            Self::OutsideCoherentSet => "outside-coherent-set",
        }
    }
}

/// A candidate [`plan_calibration`] left out, and why.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationExclusion {
    /// The candidate's path, as given.
    pub path: PathBuf,
    /// Why it was left out.
    pub reason: CalibrationExclusionReason,
}

/// The calibration frames [`plan_calibration`] chose.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationPlan {
    /// Which kind of frame was chosen.
    pub kind: CalibrationPlanKind,
    /// The fewest frames a usable set holds, at least one.
    pub minimum: usize,
    /// Whether the chosen set holds at least `minimum` frames.
    pub ready: bool,
    /// Every candidate that suits all the frames served, nearest in time
    /// first.
    pub matched_paths: Vec<PathBuf>,
    /// The matched candidates that can be combined into one master.
    pub selected_paths: Vec<PathBuf>,
    /// Each candidate left out, once, with the reason.
    pub excluded: Vec<CalibrationExclusion>,
}

/// Choose the calibration frames that suit every frame a request serves,
/// and can be combined with each other.
///
/// Each candidate is tested against every reference: its role, then its
/// sensor, then for darks and dark flats its exposure and temperature, and
/// for flats its optics. Those that pass are ordered by capture time,
/// nearest the primary reference first, and each in turn anchors a set of
/// the others that agree with it; the first set of `minimum` frames wins.
/// Two candidates whose recorded settings contradict each other never share
/// a set, even where the references record too little to rule either out.
pub fn plan_calibration(request: CalibrationPlanRequest) -> Result<CalibrationPlan> {
    let CalibrationPlanRequest {
        kind,
        reference: primary,
        references,
        candidates,
        minimum,
        tolerances,
        bias_available,
    } = request;
    let minimum = minimum.max(1);
    validate_tolerances(&tolerances)?;
    if primary.path.to_string_lossy().trim().is_empty() {
        return Err(plan_error(
            "primary calibration reference path must not be empty",
        ));
    }
    let expected_reference_role = if kind == CalibrationPlanKind::DarkFlat {
        FrameSourceRole::Flat
    } else {
        FrameSourceRole::Light
    };
    let references = if references.is_empty() {
        vec![primary.clone()]
    } else {
        references
    };
    validate_paths(&references, "calibration reference")?;
    let Some(primary_in_references) = references
        .iter()
        .find(|record| paths_refer_to_same_file(&record.path, &primary.path))
    else {
        return Err(plan_error(
            "calibration references must include the primary reference path",
        ));
    };
    if primary_in_references.role != primary.role
        || primary_in_references.signature != primary.signature
    {
        return Err(plan_error(
            "the primary calibration reference must match its record in references",
        ));
    }
    if primary.role != expected_reference_role
        || references
            .iter()
            .any(|record| record.role != expected_reference_role)
    {
        return Err(Error::Calibration(format!(
            "{} calibration references must all have role {}",
            kind.as_str(),
            expected_reference_role.as_str()
        )));
    }
    let reference_signatures = references
        .iter()
        .map(|record| &record.signature)
        .collect::<Vec<_>>();
    let scalable_dark = kind.uses_dark_matching() && bias_available;
    let dark_targets_have_exposure = !kind.uses_dark_matching()
        || reference_signatures
            .iter()
            .all(|signature| has_positive_exposure(signature));
    validate_paths(&candidates, "calibration candidate")?;

    let every_reference = |test: &dyn Fn(&FrameSignature) -> bool| {
        reference_signatures.iter().all(|reference| test(reference))
    };
    let mut excluded = Vec::new();
    let mut matched = Vec::new();
    for record in candidates {
        let candidate = &record.signature;
        let reason = if record.role != kind.role() {
            Some(CalibrationExclusionReason::RoleMismatch)
        } else if !every_reference(&|light| seiza_calibration::sensor_matches(light, candidate)) {
            Some(CalibrationExclusionReason::SensorMismatch)
        } else if kind.uses_dark_matching()
            && (!dark_targets_have_exposure || !has_positive_exposure(candidate))
        {
            Some(CalibrationExclusionReason::MissingExposure)
        } else if kind.uses_dark_matching()
            && !scalable_dark
            && !every_reference(&|light| {
                seiza_calibration::exposure_matches(light, candidate, &tolerances)
            })
        {
            Some(CalibrationExclusionReason::ExposureMismatch)
        } else if kind.uses_dark_matching()
            && !every_reference(&|light| {
                seiza_calibration::temperature_matches(light, candidate, &tolerances)
            })
        {
            Some(CalibrationExclusionReason::TemperatureMismatch)
        } else if kind == CalibrationPlanKind::Flat
            && !every_reference(&|light| {
                seiza_calibration::optics_match(light, candidate, &tolerances)
            })
        {
            Some(CalibrationExclusionReason::OpticsMismatch)
        } else {
            None
        };
        match reason {
            Some(reason) => excluded.push(CalibrationExclusion {
                path: record.path,
                reason,
            }),
            None => matched.push(record),
        }
    }

    // Order by proximity through the shared rule, then pair each sorted
    // signature with the first unplaced record that carries it. The sort is
    // stable, so records with equal signatures keep their order.
    let mut ordered_signatures = matched
        .iter()
        .map(|record| record.signature.clone())
        .collect::<Vec<_>>();
    seiza_calibration::sort_by_proximity(
        &mut ordered_signatures,
        primary.signature.captured_at_unix,
    );
    let mut ordered = Vec::with_capacity(matched.len());
    for signature in &ordered_signatures {
        let index = matched
            .iter()
            .position(|record| record.signature == *signature)
            .ok_or_else(|| plan_error("calibration plan ordering lost a candidate"))?;
        ordered.push(matched.remove(index));
    }
    let matched_paths = ordered
        .iter()
        .map(|record| record.path.clone())
        .collect::<Vec<_>>();
    let coherent = coherent_signatures(&ordered_signatures, kind, minimum, &tolerances);
    let mut selected = vec![false; ordered.len()];
    for signature in coherent {
        if let Some(index) = ordered
            .iter()
            .enumerate()
            .position(|(index, record)| !selected[index] && record.signature == signature)
        {
            selected[index] = true;
        }
    }
    let mut selected_paths = Vec::new();
    for (index, record) in ordered.into_iter().enumerate() {
        if selected[index] {
            selected_paths.push(record.path);
        } else {
            excluded.push(CalibrationExclusion {
                path: record.path,
                reason: CalibrationExclusionReason::OutsideCoherentSet,
            });
        }
    }

    Ok(CalibrationPlan {
        kind,
        minimum,
        ready: selected_paths.len() >= minimum,
        matched_paths,
        selected_paths,
        excluded,
    })
}

fn plan_error(message: &str) -> Error {
    Error::Calibration(message.into())
}

fn validate_tolerances(tolerances: &MatchTolerances) -> Result<()> {
    for (name, value) in [
        ("exposure-seconds", tolerances.exposure_seconds),
        ("exposure-fraction", tolerances.exposure_fraction),
        ("dark-temperature", tolerances.dark_temperature_c),
        ("master-temperature", tolerances.master_temperature_c),
        ("rotation", tolerances.rotation_deg),
        ("focal-length", tolerances.focal_length_mm),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(Error::Calibration(format!(
                "calibration {name} tolerance must be finite and non-negative"
            )));
        }
    }
    Ok(())
}

fn validate_paths(records: &[CalibrationRecord], what: &str) -> Result<()> {
    if records
        .iter()
        .any(|record| record.path.as_os_str().is_empty())
    {
        return Err(Error::Calibration(format!(
            "{what} paths must not be empty"
        )));
    }
    let mut seen = std::collections::HashSet::new();
    for record in records {
        if !seen.insert(path_identity(&record.path)) {
            return Err(Error::Calibration(format!(
                "duplicate {what} path {}",
                record.path.display()
            )));
        }
    }
    Ok(())
}

fn has_positive_exposure(signature: &FrameSignature) -> bool {
    signature
        .exposure_seconds
        .is_some_and(|value| value.is_finite() && value > 0.0)
}

/// The first set, anchored on each matched candidate in turn, that agrees
/// with itself and holds `minimum` frames; failing that, the first set.
fn coherent_signatures(
    signatures: &[FrameSignature],
    kind: CalibrationPlanKind,
    minimum: usize,
    tolerances: &MatchTolerances,
) -> Vec<FrameSignature> {
    let mut first = None;
    for anchor in signatures {
        let compatible = signatures
            .iter()
            .filter(|candidate| internally_compatible(anchor, candidate, kind, tolerances))
            .cloned()
            .collect::<Vec<_>>();
        let cluster = seiza_calibration::coherent_subset(
            &compatible,
            if kind == CalibrationPlanKind::Flat {
                FrameRole::Flat
            } else {
                FrameRole::Other
            },
            minimum,
            tolerances,
        );
        if cluster.len() >= minimum.max(1) {
            return cluster;
        }
        if first.is_none() {
            first = Some(cluster);
        }
    }
    first.unwrap_or_default()
}

fn internally_compatible(
    left: &FrameSignature,
    right: &FrameSignature,
    kind: CalibrationPlanKind,
    tolerances: &MatchTolerances,
) -> bool {
    // Light-to-calibration matching is deliberately asymmetric: unknown
    // metadata on a light cannot rule a candidate out. A master set needs the
    // stricter answer, because two candidates with conflicting known settings
    // must never be averaged merely because every target omitted that field.
    seiza_calibration::sensor_consistent(left, right)
        && (!kind.uses_dark_matching()
            || seiza_calibration::exposure_matches(left, right, tolerances))
        && (kind != CalibrationPlanKind::Flat
            || seiza_calibration::optics_consistent(left, right, tolerances))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signature(
        exposure_seconds: Option<f64>,
        camera_temp_c: Option<f64>,
        captured_at_unix: Option<i64>,
    ) -> FrameSignature {
        let mut signature = FrameSignature::default();
        signature.camera = Some("ASI2600MM".into());
        signature.width = Some(100);
        signature.height = Some(80);
        signature.channels = Some(1);
        signature.exposure_seconds = exposure_seconds;
        signature.camera_temp_c = camera_temp_c;
        signature.captured_at_unix = captured_at_unix;
        signature
    }

    fn record(path: &str, role: FrameSourceRole, signature: FrameSignature) -> CalibrationRecord {
        CalibrationRecord {
            path: path.into(),
            role,
            signature,
        }
    }

    fn dark_request(candidates: Vec<CalibrationRecord>) -> CalibrationPlanRequest {
        CalibrationPlanRequest {
            kind: CalibrationPlanKind::Dark,
            reference: record(
                "light.fits",
                FrameSourceRole::Light,
                signature(Some(60.0), Some(-10.0), Some(1000)),
            ),
            references: Vec::new(),
            candidates,
            minimum: 2,
            tolerances: MatchTolerances::default(),
            bias_available: false,
        }
    }

    #[test]
    fn darks_are_ordered_by_time_and_each_exclusion_says_why() {
        let plan = plan_calibration(dark_request(vec![
            record(
                "dark-later.fits",
                FrameSourceRole::Dark,
                signature(Some(60.0), Some(-9.8), Some(1100)),
            ),
            record(
                "dark-nearest.fits",
                FrameSourceRole::Dark,
                signature(Some(60.0), Some(-10.1), Some(1005)),
            ),
            record(
                "wrong-exposure.fits",
                FrameSourceRole::Dark,
                signature(Some(120.0), Some(-10.0), Some(1001)),
            ),
            record(
                "a-flat.fits",
                FrameSourceRole::Flat,
                signature(Some(60.0), Some(-10.0), Some(1001)),
            ),
        ]))
        .unwrap();
        assert!(plan.ready);
        assert_eq!(plan.kind.as_str(), "dark");
        assert_eq!(
            plan.matched_paths,
            [
                PathBuf::from("dark-nearest.fits"),
                PathBuf::from("dark-later.fits")
            ]
        );
        assert_eq!(plan.selected_paths.len(), 2);
        let reasons = plan
            .excluded
            .iter()
            .map(|exclusion| (exclusion.path.to_str().unwrap(), exclusion.reason.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(
            reasons,
            [
                ("wrong-exposure.fits", "exposure-mismatch"),
                ("a-flat.fits", "role-mismatch")
            ]
        );
    }

    #[test]
    fn a_bias_lets_darks_of_another_exposure_match() {
        let candidates = vec![record(
            "dark-120.fits",
            FrameSourceRole::Dark,
            signature(Some(120.0), Some(-10.0), Some(1001)),
        )];
        let unscaled = plan_calibration(dark_request(candidates.clone())).unwrap();
        assert!(unscaled.matched_paths.is_empty());
        let mut request = dark_request(candidates);
        request.bias_available = true;
        request.minimum = 1;
        let scaled = plan_calibration(request).unwrap();
        assert!(scaled.ready);
        assert_eq!(scaled.selected_paths, [PathBuf::from("dark-120.fits")]);
    }

    #[test]
    fn bad_requests_are_refused() {
        let mut request = dark_request(Vec::new());
        request.tolerances.rotation_deg = f64::NAN;
        assert!(
            plan_calibration(request)
                .unwrap_err()
                .to_string()
                .contains("rotation tolerance")
        );

        let mut request = dark_request(Vec::new());
        request.reference.role = FrameSourceRole::Flat;
        assert!(
            plan_calibration(request)
                .unwrap_err()
                .to_string()
                .contains("role light")
        );

        let twice = record(
            "dark.fits",
            FrameSourceRole::Dark,
            signature(Some(60.0), None, None),
        );
        let request = dark_request(vec![twice.clone(), twice]);
        assert!(
            plan_calibration(request)
                .unwrap_err()
                .to_string()
                .contains("duplicate calibration candidate path")
        );
    }

    #[test]
    fn kinds_read_their_own_spelling() {
        for kind in [
            CalibrationPlanKind::Bias,
            CalibrationPlanKind::Dark,
            CalibrationPlanKind::DarkFlat,
            CalibrationPlanKind::Flat,
        ] {
            assert_eq!(kind.as_str().parse::<CalibrationPlanKind>().unwrap(), kind);
        }
        assert!("light".parse::<CalibrationPlanKind>().is_err());
    }

    #[test]
    fn a_probe_reads_a_header_without_its_pixels() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("dark.fits");
        let image = crate::LinearImage::new(2, 2, 1, vec![1.0; 4]).unwrap();
        let cards = [
            seiza_fits::WriteHeaderCard::new(
                "IMAGETYP",
                HeaderValue::String("Master Dark Frame".into()),
            ),
            seiza_fits::WriteHeaderCard::new("GAIN", HeaderValue::Integer(100)),
        ];
        crate::write_processed_image_fits_f32(&path, &image, &[], &cards).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(2880)
            .unwrap();

        let probe = probe_frame(&path).unwrap();
        assert_eq!(probe.format, Format::Fits);
        assert_eq!(probe.raw_image_type.as_deref(), Some("Master Dark Frame"));
        assert_eq!(probe.metadata.role, FrameSourceRole::Dark);
        assert!(probe.metadata.is_master);
        assert_eq!(probe.metadata.signature.gain, Some(100));

        let error = probe_frame(&directory.path().join("dark.png")).unwrap_err();
        assert!(error.to_string().contains("not a FITS or XISF path"));
    }
}
