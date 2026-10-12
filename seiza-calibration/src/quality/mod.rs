//! Grading bias, dark, and flat frames before they go into a master.
//!
//! Measuring and judging are separate steps. A `measure_*` function reads one
//! frame and returns numbers small enough to keep, so a host can measure each
//! frame once, store the result, and judge it again whenever the set it
//! belongs to changes. A `screen_*` function takes the measurements of one
//! set — frames that should match: one camera mode, one exposure and
//! temperature for darks, and for flats one filter, session, and rotation —
//! and gives every frame a [`FrameVerdict`].
//!
//! Most judgements compare a frame with its peers, so they need at least
//! three frames; with fewer, only the checks that stand alone run. Grouping
//! frames into sets is the host's job, as is deciding what a verdict means for
//! a master: these functions say what the pixels show, never why.
//!
//! The temperature a flat was shot at is not judged. A flat taken warm with
//! the cooler off can be a good flat.

mod dark;
mod flat;

pub use dark::{
    BiasScreening, DarkMeasureOptions, DarkMeasurement, DarkScreening, measure_dark, screen_bias,
    screen_darks,
};
pub use flat::{
    FlatFeature, FlatMeasureOptions, FlatMeasurement, FlatScreening, FlatSetReport,
    IlluminationProfile, measure_flat, screen_flats,
};

use serde::{Deserialize, Serialize};

/// Bump when equal frames and options may measure differently, so a host can
/// tell its stored measurements are stale.
pub const FRAME_QUALITY_ALGORITHM_VERSION: u32 = 1;

const fn frame_quality_algorithm_version() -> u32 {
    FRAME_QUALITY_ALGORITHM_VERSION
}

/// Fewest frames a set needs before a frame is compared with its peers.
/// Below three there is no majority to say which frame is the odd one out.
const MIN_PEERS: usize = 3;

/// How a frame came out of screening, from best to worst.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Grade {
    /// Nothing crossed a limit.
    #[default]
    Ok,
    /// Something crossed the first limit: worth a look before use.
    Suspect,
    /// Something crossed the second limit: likely to harm a master.
    Poor,
}

impl Grade {
    /// Stable lowercase name for logs and reports.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Suspect => "suspect",
            Self::Poor => "poor",
        }
    }
}

/// What a finding measured. The name is stable and machine-readable; each
/// variant says what [`Finding::value`] holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FindingCode {
    /// Flat: share of samples at full scale.
    Clipped,
    /// Flat: median of the brightest channel as a share of full scale.
    Bright,
    /// Flat: median of the dimmest channel as a share of full scale.
    Dim,
    /// Flat: largest difference between this frame's illumination pattern
    /// and the set's, as a fraction of the response.
    Illumination,
    /// Flat: compact dips in this frame, steady or not.
    Features,
    /// Flat: compact dips in this frame that do not hold steady across the
    /// set.
    ChangingFeatures,
    /// Flat set: changing dips present in more than half the frames, which a
    /// median-based master keeps.
    CarriedFeatures,
    /// Bias: distance of this frame's level from the set median, in units of
    /// the set's median pixel noise.
    Level,
    /// Dark: how far this frame's level sits above the quietest frame's, in
    /// units of that frame's robust pixel spread. Light reaching the sensor
    /// lifts the level.
    LevelExcess,
    /// Bias or dark: pixel noise as a ratio to the set median.
    Noise,
    /// Bias or dark: spread of row levels as a ratio to the pixel noise.
    RowBanding,
    /// Bias or dark: spread of column levels as a ratio to the pixel noise.
    ColumnBanding,
    /// Dark: hot pixel count as a ratio to the set median.
    HotPixels,
}

impl FindingCode {
    /// Stable snake_case name, the same one serde writes.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Clipped => "clipped",
            Self::Bright => "bright",
            Self::Dim => "dim",
            Self::Illumination => "illumination",
            Self::Features => "features",
            Self::ChangingFeatures => "changing_features",
            Self::CarriedFeatures => "carried_features",
            Self::Level => "level",
            Self::LevelExcess => "level_excess",
            Self::Noise => "noise",
            Self::RowBanding => "row_banding",
            Self::ColumnBanding => "column_banding",
            Self::HotPixels => "hot_pixels",
        }
    }
}

/// One measurement that crossed a limit, with the numbers behind it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Finding {
    /// What was measured.
    pub code: FindingCode,
    /// [`Grade::Suspect`] or [`Grade::Poor`].
    pub grade: Grade,
    /// The measured value, in the unit [`Self::code`] names.
    pub value: f64,
    /// The limit it reached.
    pub limit: f64,
}

/// One frame's result: its worst grade and every limit it crossed.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FrameVerdict {
    /// The worst grade among the findings; [`Grade::Ok`] when there are none.
    pub grade: Grade,
    /// Every limit crossed, in the order the checks ran.
    pub findings: Vec<Finding>,
}

impl FrameVerdict {
    fn from_findings(findings: Vec<Finding>) -> Self {
        let grade = findings
            .iter()
            .map(|finding| finding.grade)
            .max()
            .unwrap_or_default();
        Self { grade, findings }
    }
}

/// The two values of one measurement at which a frame turns suspect and then
/// poor. Each check says whether a higher or a lower value is worse. An
/// infinite limit on the bad side turns that step off.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Limits {
    /// The value at which a frame becomes suspect.
    pub suspect: f64,
    /// The value at which a frame becomes poor.
    pub poor: f64,
}

impl Limits {
    /// Limits for one check.
    pub const fn new(suspect: f64, poor: f64) -> Self {
        Self { suspect, poor }
    }

    /// Record a finding when `value` reaches a limit, a higher value being
    /// worse. A non-finite value says nothing and is skipped.
    fn check_above(self, findings: &mut Vec<Finding>, code: FindingCode, value: f64) {
        if !value.is_finite() {
            return;
        }
        let (grade, limit) = if value >= self.poor {
            (Grade::Poor, self.poor)
        } else if value >= self.suspect {
            (Grade::Suspect, self.suspect)
        } else {
            return;
        };
        findings.push(Finding {
            code,
            grade,
            value,
            limit,
        });
    }

    /// Record a finding when `value` reaches a limit, a lower value being
    /// worse.
    fn check_below(self, findings: &mut Vec<Finding>, code: FindingCode, value: f64) {
        if !value.is_finite() {
            return;
        }
        let (grade, limit) = if value <= self.poor {
            (Grade::Poor, self.poor)
        } else if value <= self.suspect {
            (Grade::Suspect, self.suspect)
        } else {
            return;
        };
        findings.push(Finding {
            code,
            grade,
            value,
            limit,
        });
    }
}

/// Median of finite values, sorting a copy. `NaN` when there are none.
fn median_of(values: impl IntoIterator<Item = f64>) -> f64 {
    let finite: Vec<f64> = values
        .into_iter()
        .filter(|value| value.is_finite())
        .collect();
    seiza_stats::median_f64(&finite).unwrap_or(f64::NAN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_grade_in_the_direction_asked() {
        let limits = Limits::new(2.0, 5.0);
        let mut findings = Vec::new();
        limits.check_above(&mut findings, FindingCode::Noise, 1.9);
        limits.check_above(&mut findings, FindingCode::Noise, 2.0);
        limits.check_above(&mut findings, FindingCode::Noise, 7.0);
        limits.check_above(&mut findings, FindingCode::Noise, f64::NAN);
        assert_eq!(
            findings
                .iter()
                .map(|finding| (finding.grade, finding.limit))
                .collect::<Vec<_>>(),
            vec![(Grade::Suspect, 2.0), (Grade::Poor, 5.0)]
        );

        let limits = Limits::new(0.1, 0.03);
        let mut findings = Vec::new();
        limits.check_below(&mut findings, FindingCode::Dim, 0.2);
        limits.check_below(&mut findings, FindingCode::Dim, 0.05);
        limits.check_below(&mut findings, FindingCode::Dim, 0.01);
        assert_eq!(
            findings
                .iter()
                .map(|finding| finding.grade)
                .collect::<Vec<_>>(),
            vec![Grade::Suspect, Grade::Poor]
        );
    }

    #[test]
    fn a_verdict_takes_its_worst_finding() {
        assert_eq!(FrameVerdict::from_findings(Vec::new()).grade, Grade::Ok);
        let finding = |grade| Finding {
            code: FindingCode::Clipped,
            grade,
            value: 1.0,
            limit: 1.0,
        };
        let verdict =
            FrameVerdict::from_findings(vec![finding(Grade::Suspect), finding(Grade::Poor)]);
        assert_eq!(verdict.grade, Grade::Poor);
    }

    #[test]
    fn codes_serialize_as_their_names() {
        for code in [
            FindingCode::Clipped,
            FindingCode::ChangingFeatures,
            FindingCode::CarriedFeatures,
            FindingCode::LevelExcess,
            FindingCode::RowBanding,
            FindingCode::HotPixels,
        ] {
            assert_eq!(
                serde_json::to_string(&code).unwrap(),
                format!("\"{}\"", code.as_str())
            );
        }
        assert_eq!(
            serde_json::to_string(&Grade::Suspect).unwrap(),
            "\"suspect\""
        );
    }
}
