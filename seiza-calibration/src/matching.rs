//! Deciding which frames belong together.
//!
//! Building a master is a pixel operation; choosing what goes into it is not.
//! These are the rules for that choice — whether a dark suits a light, whether
//! a flat still corrects it, which of a pile of candidates can share one
//! master — as plain functions over what a frame was shot with.
//!
//! Nothing here touches a file, a catalog, or a cache. A host supplies
//! [`FrameSignature`] values from wherever it keeps them and decides what to
//! do with the answer.
//!
//! # Unknown matches
//!
//! The general rule: a missing value on the *candidate* side disqualifies it,
//! and a missing value on the *reference* side is compatible. The asymmetry is
//! deliberate. A light that does not say what gain it used cannot rule
//! anything out, so it accepts what it is offered; a calibration frame that
//! does not say cannot prove it belongs, so it is not offered.
//!
//! Three places depart from it, each for a reason:
//!
//! - [`rotation_matches`] accepts unknown on *either* side. A missing angle
//!   means the rig had no rotator, or the record predates keeping one, and
//!   treating that as a mismatch would strip flats from every frame shot
//!   before anyone wrote the angle down.
//! - [`sensor_matches`] needs positive identity evidence, so an all-unknown
//!   *reference* matches nothing. Without that floor a frame that recorded
//!   nothing would match everything.
//! - [`coherent_subset`] treats unknown temperatures and capture times as
//!   compatible on either side: it is asking whether frames can be averaged
//!   together, and an absent reading cannot prove they cannot.

use serde::{Deserialize, Serialize};

/// Which rules a set of frames is being judged under.
///
/// Flats are the strict case: they record one session's dust at one rotator
/// angle, so a flat from another night describes a different optical train
/// however well its settings agree. Everything else — bias, dark, dark flat —
/// only has to agree on the sensor.
///
/// Dark flats are deliberately [`Self::Other`], matching the rule this was
/// extracted from: they are darks that happen to be the length of a flat, and
/// carry none of a flat's optical meaning.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameRole {
    /// Bias, dark, or dark flat.
    #[default]
    Other,
    /// A flat, which additionally has to share a session and an angle.
    Flat,
}

/// How close two readings have to be to count as the same.
///
/// The defaults are the ones a rig's own scatter needs rather than what a
/// specification promises: sensors report temperature to about a degree, and
/// a set-point hunts by more than that over a night.
///
/// Deliberately *not* `#[non_exhaustive]`, unlike [`FrameSignature`]. A new
/// tolerance changes what matches, so a consumer being made to look at it when
/// one is added is the point, not a cost — and it keeps
/// `MatchTolerances { rotation_deg: 2.0, ..Default::default() }` working,
/// which is how a config struct wants to be written.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MatchTolerances {
    /// Dark exposure against light exposure: the floor, in seconds.
    ///
    /// The comparison takes whichever of this and
    /// [`Self::exposure_fraction`] is larger, because neither works alone. A
    /// fixed 0.05 s is a tenth of a half-second flat and a six-thousandth of a
    /// five-minute sub — far too loose at one end and tighter than a shutter
    /// can be trusted at the other.
    pub exposure_seconds: f64,
    /// Dark exposure against light exposure: the proportional part, as a
    /// fraction of the longer of the two. Timing error scales with exposure,
    /// so past about a minute this is what decides.
    pub exposure_fraction: f64,
    /// Dark sensor temperature against light sensor temperature, in Celsius.
    pub dark_temperature_c: f64,
    /// Sensor temperature within one master's input set, in Celsius. Tighter
    /// than [`Self::dark_temperature_c`]: matching decides what may be used,
    /// this decides what may be averaged together.
    pub master_temperature_c: f64,
    /// Rotator angle between a flat and what it corrects, in degrees.
    pub rotation_deg: f64,
    /// Focal length between a flat and what it corrects, in millimetres.
    pub focal_length_mm: f64,
    /// How far apart flats in one master may have been shot, in seconds.
    /// Flats are a session's dust and vignetting; a week-old flat describes a
    /// different optical train even when every setting agrees.
    pub flat_session_seconds: u64,
}

impl Default for MatchTolerances {
    fn default() -> Self {
        Self {
            exposure_seconds: 0.05,
            exposure_fraction: 1.0e-3,
            dark_temperature_c: 3.0,
            master_temperature_c: 1.0,
            rotation_deg: 1.0,
            focal_length_mm: 1.0,
            flat_session_seconds: 24 * 60 * 60,
        }
    }
}

/// What a frame was shot with, as far as matching cares.
///
/// A host's own record will hold more — identity, paths, checksums, grades.
/// This is the part that decides whether two frames belong together.
///
/// Construct with [`Default::default`] — every field is "unknown" — and set
/// what the frame actually recorded; more fields may be added without a major
/// version.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FrameSignature {
    pub camera: Option<String>,
    pub telescope: Option<String>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub channels: Option<i64>,
    pub binning_x: Option<i64>,
    pub binning_y: Option<i64>,
    pub gain: Option<i64>,
    pub offset: Option<i64>,
    pub readout_mode: Option<i64>,
    pub bayer_pattern: Option<String>,
    pub filter: Option<String>,
    pub focal_length_mm: Option<f64>,
    /// Rotator angle in degrees, when the rig recorded one.
    pub rotation_deg: Option<f64>,
    pub exposure_seconds: Option<f64>,
    pub camera_temp_c: Option<f64>,
    /// Capture time in Unix seconds.
    pub captured_at_unix: Option<i64>,
}

/// Whether two frames came off the same sensor in the same mode.
///
/// This is the floor every kind of calibration frame has to clear: a bias, a
/// dark, a dark flat and a flat are all useless against a light read out
/// differently.
///
/// Identity needs positive evidence: agreeing camera names, or known and
/// agreeing dimensions on both sides. Without that floor, a frame that
/// recorded nothing at all would match everything, because every other rule
/// treats an unknown reference as compatible.
///
/// Note what this does *not* do. Two frames that both name a camera must agree
/// on that name — geometry does not rescue a rename, because a camera called
/// something new is, as far as this can tell, a different camera.
pub fn sensor_matches(reference: &FrameSignature, candidate: &FrameSignature) -> bool {
    sensor_identity_matches(reference, candidate)
        && text_equal_if_known(reference.camera.as_deref(), candidate.camera.as_deref())
        && equal_if_known(reference.width, candidate.width)
        && equal_if_known(reference.height, candidate.height)
        && equal_if_known(reference.channels, candidate.channels)
        && equal_if_known(reference.binning_x, candidate.binning_x)
        && equal_if_known(reference.binning_y, candidate.binning_y)
        && equal_if_known(reference.gain, candidate.gain)
        && equal_if_known(reference.offset, candidate.offset)
        && equal_if_known(reference.readout_mode, candidate.readout_mode)
        && text_equal_if_known(
            reference.bayer_pattern.as_deref(),
            candidate.bayer_pattern.as_deref(),
        )
}

/// Whether two calibration frames have no contradictory sensor settings.
///
/// This is deliberately symmetric and differs from [`sensor_matches`]. A
/// missing setting on either side is tolerated because absence does not prove
/// two calibration frames disagree; two known values must agree. Positive
/// identity evidence is still required through an agreeing camera name or
/// agreeing dimensions, so two empty signatures are not called consistent.
pub fn sensor_consistent(left: &FrameSignature, right: &FrameSignature) -> bool {
    sensor_identity_matches(left, right)
        && text_equal_if_both_known(left.camera.as_deref(), right.camera.as_deref())
        && equal_if_both_known(left.width, right.width)
        && equal_if_both_known(left.height, right.height)
        && equal_if_both_known(left.channels, right.channels)
        && equal_if_both_known(left.binning_x, right.binning_x)
        && equal_if_both_known(left.binning_y, right.binning_y)
        && equal_if_both_known(left.gain, right.gain)
        && equal_if_both_known(left.offset, right.offset)
        && equal_if_both_known(left.readout_mode, right.readout_mode)
        && text_equal_if_both_known(
            left.bayer_pattern.as_deref(),
            right.bayer_pattern.as_deref(),
        )
}

/// Whether a flat describes the same optical path as what it would correct.
///
/// A flat records the dust and vignetting of one train at one angle. Change
/// the filter, the telescope, the focal length or the rotator and it is
/// describing something else.
pub fn optics_match(
    reference: &FrameSignature,
    candidate: &FrameSignature,
    tolerances: &MatchTolerances,
) -> bool {
    text_equal_if_known(reference.filter.as_deref(), candidate.filter.as_deref())
        && text_equal_if_known(
            reference.telescope.as_deref(),
            candidate.telescope.as_deref(),
        )
        && option_near(
            reference.focal_length_mm,
            candidate.focal_length_mm,
            tolerances.focal_length_mm,
        )
        && rotation_matches(
            reference.rotation_deg,
            candidate.rotation_deg,
            tolerances.rotation_deg,
        )
}

/// Whether two calibration frames have no contradictory optical settings.
///
/// Unlike [`optics_match`], neither side is privileged as the candidate: a
/// missing value is tolerated, while two recorded values must agree. This is
/// the appropriate question before averaging flat candidates into one master.
pub fn optics_consistent(
    left: &FrameSignature,
    right: &FrameSignature,
    tolerances: &MatchTolerances,
) -> bool {
    text_equal_if_both_known(left.filter.as_deref(), right.filter.as_deref())
        && text_equal_if_both_known(left.telescope.as_deref(), right.telescope.as_deref())
        && option_near_if_both_known(
            left.focal_length_mm,
            right.focal_length_mm,
            tolerances.focal_length_mm,
        )
        && rotation_matches(
            left.rotation_deg,
            right.rotation_deg,
            tolerances.rotation_deg,
        )
}

/// Whether two rotator angles are close enough to share a flat.
///
/// The rotator sits between the telescope and the camera, so vignetting from
/// the optics ahead of it turns relative to the sensor as the rotator moves: a
/// flat only corrects frames shot at nearly the same angle. The comparison
/// wraps, so 359 degrees and 1 degree are two degrees apart.
///
/// Unknown on *either* side matches, unlike every other rule here. A missing
/// angle means the rig had no rotator, or the host's record predates keeping
/// one; treating that as a mismatch would strip flats from every frame shot
/// before anyone thought to write the angle down.
pub fn rotation_matches(
    reference: Option<f64>,
    candidate: Option<f64>,
    tolerance_deg: f64,
) -> bool {
    match (known(reference), known(candidate)) {
        (Some(reference), Some(candidate)) => {
            let difference = (reference - candidate).rem_euclid(360.0);
            difference.min(360.0 - difference) <= tolerance_deg
        }
        _ => true,
    }
}

/// Whether a dark's exposure suits the frame it would be subtracted from.
pub fn exposure_matches(
    reference: &FrameSignature,
    candidate: &FrameSignature,
    tolerances: &MatchTolerances,
) -> bool {
    option_near(
        reference.exposure_seconds,
        candidate.exposure_seconds,
        exposure_tolerance(
            reference.exposure_seconds,
            candidate.exposure_seconds,
            tolerances,
        ),
    )
}

/// How far apart two exposures may be and still count as the same one.
///
/// The floor or the proportional part, whichever is larger. This is the single
/// answer to "are these the same exposure" — the master builder asks it of
/// frames going into one master, and selection asks it of a dark against a
/// light. Two answers to one question is how a set that builds cleanly comes
/// to contain a frame selection would have refused.
pub fn exposure_tolerance(
    reference: Option<f64>,
    candidate: Option<f64>,
    tolerances: &MatchTolerances,
) -> f64 {
    let longer = known(reference)
        .unwrap_or(0.0)
        .abs()
        .max(known(candidate).unwrap_or(0.0).abs());
    tolerances
        .exposure_seconds
        .max(longer * tolerances.exposure_fraction)
}

/// Whether a dark's sensor temperature suits the frame it would be subtracted
/// from. Dark current roughly doubles every six degrees, so this is the
/// tolerance that decides whether the subtraction helps or hurts.
pub fn temperature_matches(
    reference: &FrameSignature,
    candidate: &FrameSignature,
    tolerances: &MatchTolerances,
) -> bool {
    option_near(
        reference.camera_temp_c,
        candidate.camera_temp_c,
        tolerances.dark_temperature_c,
    )
}

/// The subset of `candidates` that can actually be averaged into one master.
///
/// Matching says what *may* be used; this says what may be combined. Frames
/// that each suit the light can still disagree with each other — two darks a
/// month and three degrees apart average into a master that describes neither
/// night.
///
/// Every candidate in turn anchors a cluster of the frames coherent with it,
/// and the first cluster large enough to build wins. Taking the first rather
/// than the largest keeps a stray frame shot near the lights from orphaning a
/// complete session from a week earlier, given the caller has ordered
/// candidates by preference — see [`sort_by_proximity`]. With no cluster big
/// enough, the first is returned and the caller can decline to build.
///
/// [`FrameRole::Flat`] tightens the rule: flats additionally have to share a
/// session and a rotator angle, because they describe one night's dust at one
/// angle. Dark flats are [`FrameRole::Other`] — see [`FrameRole`].
///
/// `minimum` is treated as at least one. A zero would make every cluster large
/// enough and return whatever the first candidate happened to agree with,
/// which is not an answer anyone means to ask for.
pub fn coherent_subset(
    candidates: &[FrameSignature],
    role: FrameRole,
    minimum: usize,
    tolerances: &MatchTolerances,
) -> Vec<FrameSignature> {
    coherent_subset_indices(candidates, role, minimum, tolerances)
        .into_iter()
        .map(|index| candidates[index].clone())
        .collect()
}

/// [`coherent_subset`], as positions into `candidates` rather than copies.
///
/// A signature is only the part of a frame that decides whether it belongs
/// with another; a host's own record carries the rest — a path, a checksum, a
/// catalog row — and copies cannot be traced back to it. Positions can, so a
/// caller that needs its own frames back asks for these and indexes its own
/// list.
///
/// This is also the cheaper call. It compares signatures without copying any,
/// where [`coherent_subset`] clones the cluster it settles on.
pub fn coherent_subset_indices(
    candidates: &[FrameSignature],
    role: FrameRole,
    minimum: usize,
    tolerances: &MatchTolerances,
) -> Vec<usize> {
    let minimum = minimum.max(1);
    let flats = role == FrameRole::Flat;
    let coherent = |anchor: &FrameSignature, frame: &FrameSignature| -> bool {
        let temperature = match (known(anchor.camera_temp_c), known(frame.camera_temp_c)) {
            (Some(anchor), Some(frame)) => {
                (anchor - frame).abs() <= tolerances.master_temperature_c
            }
            _ => true,
        };
        let session = if flats {
            match (anchor.captured_at_unix, frame.captured_at_unix) {
                (Some(anchor), Some(frame)) => {
                    anchor.abs_diff(frame) <= tolerances.flat_session_seconds
                }
                _ => true,
            }
        } else {
            true
        };
        let rotation = !flats
            || rotation_matches(
                anchor.rotation_deg,
                frame.rotation_deg,
                tolerances.rotation_deg,
            );
        temperature && session && rotation
    };

    let mut first: Option<Vec<usize>> = None;
    for anchor in candidates {
        let cluster: Vec<usize> = candidates
            .iter()
            .enumerate()
            .filter(|(_, frame)| coherent(anchor, frame))
            .map(|(index, _)| index)
            .collect();
        if cluster.len() >= minimum {
            return cluster;
        }
        if first.is_none() {
            first = Some(cluster);
        }
    }
    first.unwrap_or_default()
}

/// Order candidates by how close in time they were shot to `reference_unix`,
/// nearest first.
///
/// Calibration ages: a sensor's dark current drifts, dust arrives and leaves.
/// Where several sets match equally well, the one shot nearest the light is
/// the one that describes it. Frames with no capture time sort last when a
/// reference is known, and keep their order when it is not.
pub fn sort_by_proximity(frames: &mut [FrameSignature], reference_unix: Option<i64>) {
    frames.sort_by_key(|frame| match (reference_unix, frame.captured_at_unix) {
        (Some(reference), Some(captured)) => reference.abs_diff(captured),
        (Some(_), None) => u64::MAX,
        _ => 0,
    });
}

/// Camera name agreement, or failing that, identical dimensions.
fn sensor_identity_matches(left: &FrameSignature, right: &FrameSignature) -> bool {
    matches!(
        (left.camera.as_deref(), right.camera.as_deref()),
        (Some(left), Some(right)) if left.trim().eq_ignore_ascii_case(right.trim())
    ) || matches!(
        (left.width, right.width, left.height, right.height),
        (Some(lw), Some(rw), Some(lh), Some(rh)) if lw == rw && lh == rh
    )
}

fn equal_if_known<T: PartialEq>(reference: Option<T>, candidate: Option<T>) -> bool {
    match (reference, candidate) {
        (Some(reference), Some(candidate)) => reference == candidate,
        (Some(_), None) => false,
        (None, _) => true,
    }
}

fn equal_if_both_known<T: PartialEq>(left: Option<T>, right: Option<T>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left == right,
        _ => true,
    }
}

fn text_equal_if_known(reference: Option<&str>, candidate: Option<&str>) -> bool {
    match (reference, candidate) {
        (Some(reference), Some(candidate)) => {
            reference.trim().eq_ignore_ascii_case(candidate.trim())
        }
        (Some(_), None) => false,
        (None, _) => true,
    }
}

fn text_equal_if_both_known(left: Option<&str>, right: Option<&str>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left.trim().eq_ignore_ascii_case(right.trim()),
        _ => true,
    }
}

fn option_near(reference: Option<f64>, candidate: Option<f64>, tolerance: f64) -> bool {
    match (known(reference), known(candidate)) {
        (Some(reference), Some(candidate)) => (reference - candidate).abs() <= tolerance,
        (Some(_), None) => false,
        (None, _) => true,
    }
}

fn option_near_if_both_known(left: Option<f64>, right: Option<f64>, tolerance: f64) -> bool {
    match (known(left), known(right)) {
        (Some(left), Some(right)) => (left - right).abs() <= tolerance,
        _ => true,
    }
}

/// A reading that is not finite is not a reading.
///
/// `NaN` compares false against everything including itself, so left alone it
/// would mean "matches nothing" — while the C ABI uses it as the *unknown*
/// sentinel, which means the opposite. Worse, a frame with a `NaN` temperature
/// would fail to be coherent with its own self and drop out of the cluster it
/// anchors. Reading non-finite as unknown makes every surface agree and keeps
/// a frame comparable with itself.
fn known(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signature() -> FrameSignature {
        FrameSignature {
            camera: Some("ASI2600MM".into()),
            telescope: Some("Askar107PHQ".into()),
            width: Some(6248),
            height: Some(4176),
            channels: Some(1),
            binning_x: Some(1),
            binning_y: Some(1),
            gain: Some(100),
            offset: Some(50),
            readout_mode: Some(0),
            bayer_pattern: None,
            filter: Some("Ha".into()),
            focal_length_mm: Some(749.0),
            rotation_deg: Some(120.0),
            exposure_seconds: Some(300.0),
            camera_temp_c: Some(-10.0),
            captured_at_unix: Some(1_700_000_000),
        }
    }

    #[test]
    fn a_different_sensor_mode_never_matches() {
        let light = signature();
        for change in [
            |f: &mut FrameSignature| f.gain = Some(200),
            |f: &mut FrameSignature| f.offset = Some(10),
            |f: &mut FrameSignature| f.binning_x = Some(2),
            |f: &mut FrameSignature| f.readout_mode = Some(1),
            |f: &mut FrameSignature| f.width = Some(4144),
            |f: &mut FrameSignature| f.camera = Some("ASI6200MM".into()),
        ] {
            let mut candidate = signature();
            change(&mut candidate);
            assert!(!sensor_matches(&light, &candidate), "{candidate:?}");
        }
        assert!(sensor_matches(&light, &signature()));
    }

    #[test]
    fn a_camera_that_never_recorded_its_name_matches_on_geometry() {
        // Identity falls back to dimensions, so frames from a rig that never
        // wrote a camera name still find each other.
        let mut light = signature();
        let mut candidate = signature();
        light.camera = None;
        candidate.camera = None;
        assert!(sensor_matches(&light, &candidate));
    }

    #[test]
    fn two_names_that_disagree_do_not_match_however_alike_the_sensors() {
        // Geometry does not rescue a rename: a camera called something new is,
        // as far as this can tell, a different camera. Documented rather than
        // fixed, because the alternative — trusting dimensions over a stated
        // disagreement — would pair frames from two identical bodies.
        let light = signature();
        let mut candidate = signature();
        candidate.camera = Some("ZWO ASI2600MM Pro".into());
        assert!(!sensor_matches(&light, &candidate));
    }

    #[test]
    fn a_reference_that_recorded_nothing_matches_nothing() {
        // Every other rule treats an unknown reference as compatible, so
        // without an identity floor an empty signature would match anything.
        assert!(!sensor_matches(&FrameSignature::default(), &signature()));
    }

    #[test]
    fn an_unknown_setting_disqualifies_the_candidate_not_the_light() {
        let light = signature();
        let mut candidate = signature();
        candidate.gain = None;
        assert!(
            !sensor_matches(&light, &candidate),
            "a frame that cannot prove its gain is not offered"
        );

        let mut light = signature();
        light.gain = None;
        assert!(
            sensor_matches(&light, &signature()),
            "a light that does not say cannot rule anything out"
        );
    }

    #[test]
    fn calibration_sensor_consistency_tolerates_missing_but_not_conflicting_settings() {
        let mut left = signature();
        let mut right = signature();
        left.gain = None;
        assert!(sensor_consistent(&left, &right));
        right.gain = Some(200);
        left.gain = Some(100);
        assert!(!sensor_consistent(&left, &right));

        let unidentified = FrameSignature {
            gain: Some(100),
            ..FrameSignature::default()
        };
        assert!(!sensor_consistent(
            &unidentified,
            &FrameSignature::default()
        ));
    }

    #[test]
    fn a_flat_only_corrects_its_own_optical_path() {
        let tolerances = MatchTolerances::default();
        let light = signature();
        assert!(optics_match(&light, &signature(), &tolerances));

        for change in [
            |f: &mut FrameSignature| f.filter = Some("OIII".into()),
            |f: &mut FrameSignature| f.telescope = Some("SpaceCat61".into()),
            |f: &mut FrameSignature| f.focal_length_mm = Some(300.0),
            |f: &mut FrameSignature| f.rotation_deg = Some(45.0),
        ] {
            let mut candidate = signature();
            change(&mut candidate);
            assert!(
                !optics_match(&light, &candidate, &tolerances),
                "{candidate:?}"
            );
        }
    }

    #[test]
    fn flat_optics_consistency_tolerates_missing_but_not_conflicting_settings() {
        let mut left = signature();
        let mut right = signature();
        left.filter = None;
        assert!(optics_consistent(
            &left,
            &right,
            &MatchTolerances::default()
        ));
        left.filter = Some("Ha".into());
        right.filter = Some("OIII".into());
        assert!(!optics_consistent(
            &left,
            &right,
            &MatchTolerances::default()
        ));
    }

    #[test]
    fn rotation_wraps_and_forgives_what_was_never_recorded() {
        assert!(rotation_matches(Some(120.0), Some(120.6), 1.0));
        assert!(!rotation_matches(Some(120.0), Some(124.0), 1.0));
        // Across the wrap: 359.5 and 0.2 are 0.7 degrees apart.
        assert!(rotation_matches(Some(359.5), Some(0.2), 1.0));
        assert!(!rotation_matches(Some(359.5), Some(3.0), 1.0));
        // A rig with no rotator, or a record that predates keeping one.
        assert!(rotation_matches(None, Some(120.0), 1.0));
        assert!(rotation_matches(Some(120.0), None, 1.0));
        assert!(rotation_matches(None, None, 1.0));
    }

    #[test]
    fn darks_match_on_exposure_and_temperature_within_tolerance() {
        let tolerances = MatchTolerances::default();
        let with = |exposure, temp| FrameSignature {
            exposure_seconds: Some(exposure),
            camera_temp_c: Some(temp),
            ..signature()
        };
        let light = with(300.0, -10.0);
        assert!(exposure_matches(&light, &with(300.02, -10.0), &tolerances));
        assert!(!exposure_matches(&light, &with(180.0, -10.0), &tolerances));

        // Past a minute the proportional part decides: 0.1% of 300 s is 0.3 s,
        // which a fixed 0.05 s floor would have refused.
        assert!(exposure_matches(&light, &with(300.25, -10.0), &tolerances));
        assert!(!exposure_matches(&light, &with(301.0, -10.0), &tolerances));

        // Below it the floor decides, because 0.1% of half a second is half a
        // millisecond and no header is written that precisely.
        let short = with(0.5, -10.0);
        assert!(exposure_matches(&short, &with(0.52, -10.0), &tolerances));
        assert!(!exposure_matches(&short, &with(0.7, -10.0), &tolerances));
        assert!(temperature_matches(
            &light,
            &with(300.0, -12.5),
            &tolerances
        ));
        assert!(!temperature_matches(&light, &with(300.0, 0.0), &tolerances));
    }

    #[test]
    fn a_coherent_subset_will_not_average_frames_that_disagree() {
        let tolerances = MatchTolerances::default();
        let at = |seconds: i64, temp: f64| FrameSignature {
            captured_at_unix: Some(seconds),
            camera_temp_c: Some(temp),
            ..signature()
        };
        // Three from one night, two from a month later and four degrees warmer.
        let candidates = vec![
            at(1_700_000_000, -10.0),
            at(1_700_000_600, -10.2),
            at(1_700_001_200, -10.1),
            at(1_702_600_000, -6.0),
            at(1_702_600_600, -6.1),
        ];
        let chosen = coherent_subset(&candidates, FrameRole::Other, 2, &tolerances);
        assert_eq!(chosen.len(), 3, "the warm month-later pair must not join");
        assert!(
            chosen
                .iter()
                .all(|frame| frame.camera_temp_c.unwrap() < -9.0)
        );
    }

    #[test]
    fn flats_additionally_have_to_share_a_session() {
        let tolerances = MatchTolerances::default();
        let at = |seconds: i64| FrameSignature {
            captured_at_unix: Some(seconds),
            ..signature()
        };
        // Same temperature and angle throughout, but two nights apart.
        let candidates = vec![
            at(1_700_000_000),
            at(1_700_000_600),
            at(1_700_400_000),
            at(1_700_400_600),
        ];
        assert_eq!(
            coherent_subset(&candidates, FrameRole::Other, 2, &tolerances).len(),
            4
        );
        assert_eq!(
            coherent_subset(&candidates, FrameRole::Flat, 2, &tolerances).len(),
            2,
            "a flat describes one session's dust"
        );
    }

    #[test]
    fn a_stray_frame_does_not_orphan_a_complete_session() {
        let tolerances = MatchTolerances::default();
        let at = |seconds: i64| FrameSignature {
            captured_at_unix: Some(seconds),
            ..signature()
        };
        // The nearest frame is alone; the real session is a week earlier.
        let candidates = vec![
            at(1_700_600_000),
            at(1_700_000_000),
            at(1_700_000_600),
            at(1_700_001_200),
        ];
        let chosen = coherent_subset(&candidates, FrameRole::Flat, 2, &tolerances);
        assert_eq!(chosen.len(), 3, "the complete session wins over the stray");
    }

    #[test]
    fn a_reading_that_is_not_finite_is_read_as_unknown() {
        // The C ABI uses NaN as its unknown sentinel. Left alone, NaN compares
        // false against everything — including itself — so the same value would
        // mean "unknown, accepts anything" through C and "matches nothing"
        // through Rust.
        let tolerances = MatchTolerances::default();
        let nan = FrameSignature {
            camera_temp_c: Some(f64::NAN),
            exposure_seconds: Some(f64::NAN),
            rotation_deg: Some(f64::NAN),
            ..signature()
        };
        assert!(temperature_matches(&nan, &signature(), &tolerances));
        assert!(exposure_matches(&nan, &signature(), &tolerances));
        assert!(rotation_matches(Some(f64::NAN), Some(120.0), 1.0));
        assert!(optics_match(&nan, &signature(), &tolerances));

        // And a frame stays comparable with itself, so it cannot drop out of
        // the cluster it anchors.
        assert_eq!(
            coherent_subset(&[nan.clone(), nan], FrameRole::Flat, 1, &tolerances).len(),
            2
        );
    }

    #[test]
    fn indices_point_back_at_the_caller_s_own_frames() {
        // The reason this exists: a signature cannot be traced to the record
        // it came from, so a host that needs its own frames back asks where
        // they were rather than for copies.
        let tolerances = MatchTolerances::default();
        let at = |seconds: i64, temp: f64| FrameSignature {
            captured_at_unix: Some(seconds),
            camera_temp_c: Some(temp),
            ..signature()
        };
        // Three from one night, two a month later and four degrees warmer.
        let candidates = vec![
            at(1_700_000_000, -10.0),
            at(1_702_600_000, -6.0),
            at(1_700_000_600, -10.2),
            at(1_702_600_600, -6.1),
            at(1_700_001_200, -10.1),
        ];
        let chosen = coherent_subset_indices(&candidates, FrameRole::Other, 2, &tolerances);
        assert_eq!(chosen, vec![0, 2, 4], "positions, in candidate order");

        // And they agree with what the copying form returns.
        let copied = coherent_subset(&candidates, FrameRole::Other, 2, &tolerances);
        assert_eq!(
            copied,
            chosen
                .iter()
                .map(|index| candidates[*index].clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn nothing_coherent_enough_returns_the_first_cluster_rather_than_guessing() {
        let tolerances = MatchTolerances::default();
        let at = |seconds: i64| FrameSignature {
            captured_at_unix: Some(seconds),
            ..signature()
        };
        let candidates = vec![at(1_700_000_000), at(1_800_000_000)];
        let chosen = coherent_subset(&candidates, FrameRole::Flat, 3, &tolerances);
        assert_eq!(chosen.len(), 1, "the caller decides whether to build");
    }

    #[test]
    fn proximity_ordering_puts_the_nearest_calibration_first() {
        let at = |seconds: Option<i64>| FrameSignature {
            captured_at_unix: seconds,
            ..signature()
        };
        let mut frames = vec![
            at(Some(1_700_900_000)),
            at(None),
            at(Some(1_700_000_100)),
            at(Some(1_700_400_000)),
        ];
        sort_by_proximity(&mut frames, Some(1_700_000_000));
        assert_eq!(
            frames
                .iter()
                .map(|frame| frame.captured_at_unix)
                .collect::<Vec<_>>(),
            vec![
                Some(1_700_000_100),
                Some(1_700_400_000),
                Some(1_700_900_000),
                None
            ]
        );
    }
}
