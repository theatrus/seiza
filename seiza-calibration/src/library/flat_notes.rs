// Extracted from theatrus/psf-guard at 79f5e31d53b8371e8fcd0b22a9b69692412cd8b4.
// Copyright the PSF Guard contributors; Apache-2.0.
use super::*;
pub(crate) use crate::LightRotation;
use crate::flat_response::evidence::Side;
use crate::flat_response::groups::{AngleTerm, Event, Placement};
pub(crate) use crate::flat_response::groups::{lights, model, overrides};
use crate::flat_response::{groups::Book, metric};
use sha2::{Digest, Sha256};
/// A change as a light's account names it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventNote {
    pub key: String,
    /// The window it happened in, Unix seconds.
    pub start: i64,
    pub end: i64,
    /// Fractions of the background.
    pub dust: f64,
    pub broad: f64,
    pub camera_side: bool,
    pub filters: Vec<String>,
    /// "new dust: 2 shadows near (1,460, 3,570)".
    pub what: String,
    /// A split a person made where the flats measure alike, not a
    /// measured change.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub by_hand: bool,
}

impl EventNote {
    pub(super) fn of(event: &Event, rules: &Rules) -> Self {
        Self {
            key: event.key.clone(),
            start: event.start,
            end: event.end,
            dust: event.dust,
            broad: event.broad,
            camera_side: event.camera_side,
            filters: event.filters.iter().cloned().collect(),
            what: describe_what(event, rules),
            by_hand: event.by_hand,
        }
    }
}

/// What a change is, in a few words: that a person split the flats there,
/// a change of vignetting where that leads, or its spots.
pub fn describe_what(event: &Event, rules: &Rules) -> String {
    if event.by_hand {
        "split by you where the flats measure alike".into()
    } else if vignetting_leads(
        event.dust,
        event.broad,
        event.vignetting,
        &rules.grouping_limits(),
    ) {
        format!(
            "vignetting changed at the {}",
            event
                .vignetting_changes
                .first()
                .map_or("border", |change| edge_words(change.vignetting.edge))
        )
    } else {
        describe_spots(&event.spots, event.dust, event.broad)
    }
}

/// Whether a change of vignetting, rather than dust or the light pattern,
/// is what a change is named for: held to the light-pattern limit, as
/// [`model::Limits::distance`] holds it, it is further past that than the
/// dust is past its own, and larger than the light pattern.
pub fn vignetting_leads(dust: f64, broad: f64, vignetting: f64, limits: &model::Limits) -> bool {
    vignetting > broad && vignetting / limits.broad.max(1e-9) >= dust / limits.dust.max(1e-9)
}

/// An edge or corner in words.
pub fn edge_words(edge: crate::dust::Edge) -> &'static str {
    use crate::dust::Edge;
    match edge {
        Edge::Top => "top edge",
        Edge::Bottom => "bottom edge",
        Edge::Left => "left edge",
        Edge::Right => "right edge",
        Edge::TopLeft => "top-left corner",
        Edge::TopRight => "top-right corner",
        Edge::BottomLeft => "bottom-left corner",
        Edge::BottomRight => "bottom-right corner",
    }
}

/// The spots of a change in a few words.
pub fn describe_spots(spots: &[metric::Spot], dust: f64, broad: f64) -> String {
    use metric::SpotKind;
    // Dust that moved within focus travel is drawn but not counted.
    let strong: Vec<&metric::Spot> = spots
        .iter()
        .filter(|spot| !spot.within_tolerance)
        .filter(|spot| f64::from(spot.peak) >= dust * 0.5)
        .collect();
    if strong.is_empty() {
        return format!("the light pattern changed by {:.1}%", broad * 100.0);
    }
    let kind = if strong.iter().any(|s| s.kind == SpotKind::Moved) {
        "dust moved"
    } else if strong.iter().all(|s| s.kind == SpotKind::Gone) {
        "dust gone"
    } else {
        "new dust"
    };
    let top = strong[0];
    // Large shadows are named by their shape: a filter's dust casts disks,
    // an SCT's donuts.
    let (disks, donuts) = (
        strong
            .iter()
            .filter(|s| s.shape == crate::dust::DustShape::Disk)
            .count(),
        strong
            .iter()
            .filter(|s| s.shape == crate::dust::DustShape::Donut)
            .count(),
    );
    let noun = if donuts == strong.len() {
        "donut"
    } else if disks == strong.len() {
        "disk"
    } else {
        "shadow"
    };
    let shadows = if strong.len() == 1 {
        format!("1 {noun}")
    } else {
        format!("{} {noun}s", strong.len())
    };
    format!("{kind}: {shadows} near ({:.0}, {:.0})", top.x, top.y)
}

/// The correction a light's flat master takes: the flats at its angle,
/// from the other side of a change, carried across it by the change's map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlatCorrection {
    pub event: String,
    /// The before and after sessions the map is measured from.
    pub before_response: String,
    pub after_response: String,
    /// The side the light is on: `before` divides the flats by the map,
    /// `after` multiplies.
    pub light_side: Side,
    /// Key of the correction, for master and stack identities.
    pub key: String,
}

/// Why a light's flats are the ones they are.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "basis", rename_all = "snake_case")]
pub enum FlatBasis {
    /// No measured change touches the light: the nearest flats, as always.
    #[default]
    Nearest,
    /// Flats from the light's own side of each change near it.
    SameSide {
        event: EventNote,
        placement: Placement,
        /// Matching flats left out for being on the other side.
        left_out: usize,
    },
    /// None at its angle on its side; flats from the nearest other angle on
    /// its side, which this rig's measurements allow.
    OtherAngle {
        event: EventNote,
        placement: Placement,
        rotation: Option<f64>,
        angle_term: AngleTerm,
        /// The rotation tolerance the stacker must hold this light to so
        /// those flats pass: the angle between them and the light, plus the
        /// rig's own tolerance. Absent when either angle is unknown.
        rotation_tolerance_deg: Option<f64>,
    },
    /// None on its side at any allowed angle; flats at its angle from the
    /// other side, carried across by the change's map.
    Corrected {
        event: EventNote,
        placement: Placement,
        correction: FlatCorrection,
    },
    /// Its flats are all from the other side of a change, and nothing can
    /// stand in.
    NoSameSide {
        event: EventNote,
        placement: Placement,
    },
    /// Shot while a change happened; nothing tells which side it is on.
    Undetermined {
        event: EventNote,
        placement: Placement,
        /// What another change it does stand clear of chose, such as
        /// another angle's flats or a correction: its flats and gate are
        /// that one's, and the light keeps its Check for this one.
        #[serde(skip_serializing_if = "Option::is_none")]
        then: Option<Box<FlatBasis>>,
    },
    /// A person chose the group its flats come from, over every change.
    Group { group: GroupNote },
}

/// A group of one family, as a light's account names it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupNote {
    /// The family's label: "R at 104.0°".
    pub family: String,
    pub first_night: String,
    pub last_night: String,
    pub sessions: usize,
}

impl GroupNote {
    pub(super) fn of(book: &Book, family: usize, group: usize) -> Option<Self> {
        let sessions = book.families.get(family)?.groups.get(group)?;
        let night = |index: usize| book.sessions[index].row.night.clone();
        Some(Self {
            family: family_label(book, family),
            first_night: night(*sessions.first()?),
            last_night: night(*sessions.last()?),
            sessions: sessions.len(),
        })
    }

    /// "R at 104.0°, 2026-08-12 to 2026-08-24".
    pub fn label(&self) -> String {
        if self.first_night == self.last_night {
            format!("{}, {}", self.family, self.first_night)
        } else {
            format!(
                "{}, {} to {}",
                self.family, self.first_night, self.last_night
            )
        }
    }
}

/// A family's filter and measured angle range: "OIII at 72.6° & 82.2°".
pub fn family_label(book: &Book, family: usize) -> String {
    format!(
        "{} at {}",
        book.families[family]
            .filter
            .as_deref()
            .map(str::trim)
            .unwrap_or("flats"),
        book.angle_label(family)
    )
}

/// A person's choice for a light, as its account shows it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceNote {
    pub choice: overrides::LightChoice,
    /// The change it is about, as recorded; none for a group.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub change_key: Option<String>,
    /// What it was about, in words.
    pub label: String,
    pub chosen_at: i64,
    /// Why it does not apply, when it finds nothing to apply to. It stays
    /// recorded until a person resets it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orphaned: Option<String>,
}

impl ChoiceNote {
    pub(super) fn of(row: &overrides::LightChoiceRow, orphaned: Option<String>) -> Option<Self> {
        Some(Self {
            choice: row.choice?,
            change_key: (row.change_key != overrides::GROUP_CHOICE_KEY)
                .then(|| row.change_key.clone()),
            label: row.label.clone(),
            chosen_at: row.chosen_at,
            orphaned,
        })
    }
}

impl FlatBasis {
    /// Whether the light is worth a look for it.
    pub fn flags(&self) -> bool {
        matches!(
            self,
            FlatBasis::NoSameSide { .. } | FlatBasis::Undetermined { .. }
        )
    }

    /// The change it chose around, when one touches the light.
    pub fn event(&self) -> Option<&EventNote> {
        match self {
            FlatBasis::Nearest | FlatBasis::Group { .. } => None,
            FlatBasis::SameSide { event, .. }
            | FlatBasis::OtherAngle { event, .. }
            | FlatBasis::Corrected { event, .. }
            | FlatBasis::NoSameSide { event, .. }
            | FlatBasis::Undetermined { event, .. } => Some(event),
        }
    }

    /// Its name, as the light's explanation gives it.
    pub fn name(&self) -> &'static str {
        match self {
            FlatBasis::Nearest => "nearest",
            FlatBasis::SameSide { .. } => "same_side",
            FlatBasis::OtherAngle { .. } => "other_angle",
            FlatBasis::Corrected { .. } => "corrected",
            FlatBasis::NoSameSide { .. } => "no_same_side",
            FlatBasis::Undetermined { .. } => "undetermined",
            FlatBasis::Group { .. } => "group",
        }
    }

    /// The rotation tolerance this light's masters must carry, when its
    /// flats are from another angle.
    pub fn rotation_tolerance(&self) -> Option<f64> {
        match self {
            FlatBasis::OtherAngle {
                rotation_tolerance_deg,
                ..
            } => *rotation_tolerance_deg,
            FlatBasis::Undetermined {
                then: Some(then), ..
            } => then.rotation_tolerance(),
            _ => None,
        }
    }

    /// Whether an earlier change already left the light with flats from
    /// its other side and nothing to stand in.
    pub fn lacks_same_side(&self) -> bool {
        match self {
            FlatBasis::NoSameSide { .. } => true,
            FlatBasis::Undetermined {
                then: Some(then), ..
            } => then.lacks_same_side(),
            _ => false,
        }
    }

    pub fn correction(&self) -> Option<&FlatCorrection> {
        match self {
            FlatBasis::Corrected { correction, .. } => Some(correction),
            FlatBasis::Undetermined {
                then: Some(then), ..
            } => then.correction(),
            _ => None,
        }
    }

    /// `next`, chosen around a later change, in place of this basis. An
    /// undetermined placement keeps its Check whatever comes after it: the
    /// choice rides along inside it.
    pub fn followed_by(self, next: FlatBasis) -> FlatBasis {
        match (self, next) {
            (
                FlatBasis::Undetermined {
                    event,
                    placement,
                    then,
                },
                next,
            ) if !matches!(next, FlatBasis::Undetermined { .. }) => FlatBasis::Undetermined {
                event,
                placement,
                then: Some(Box::new(match then {
                    Some(earlier) => earlier.followed_by(next),
                    None => next,
                })),
            },
            // A second undetermined change, after another choice: the
            // light is flagged, and keeps what was chosen.
            (
                earlier,
                FlatBasis::Undetermined {
                    event, placement, ..
                },
            ) => FlatBasis::Undetermined {
                event,
                placement,
                then: match earlier {
                    FlatBasis::Nearest => None,
                    FlatBasis::Undetermined { then, .. } => then,
                    other => Some(Box::new(other)),
                },
            },
            (_, next) => next,
        }
    }
}

/// The response family a light's flats come from, when grouping measured
/// it: which family, how it was chosen, and the gate a stack holds the
/// light to for it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlatFamily {
    /// The family's key (`model::Family::key`).
    pub key: String,
    /// Its filter, as the flats name it, and measured angle range: "Ha at
    /// 72.6° & 82.2°".
    pub label: String,
    /// How it was chosen.
    #[serde(flatten)]
    pub basis: lights::FamilyBasis,
    /// The family its lights pointed to, when that move was not confirmed
    /// (`disputed`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pointed: Option<String>,
    /// Whether the family has flats at the light's recorded angle.
    pub at_recorded: bool,
    /// The light's recorded angle, and how far the rotator's epochs turn
    /// it into the family's nearest session.
    pub light_angle: f64,
    pub shift: f64,
    /// The candidates within the angle prior, nearest first, as labels.
    pub candidates: Vec<String>,
    /// How the stacker's gate reads the light's angle: as recorded, or as
    /// the angle of the flats its master is built from when measurement
    /// gave it flats at another angle (seiza-calibration's
    /// `LightRotation`).
    pub light_rotation: LightRotation,
    /// When those flats lie at another angle inside a span the family's
    /// flats measure alike across: the span and its evidence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alike_range: Option<AlikeRangeNote>,
}

/// Where a light stands against a change, as selection reads it: the one
/// place choosing flats asks. A side a person chose for the light comes
/// first, over its time, its imaging session and its pixels
/// (`Placement::Person`); otherwise the book places it.
pub fn place_light(
    book: &Book,
    choices: &overrides::ResolvedChoices,
    event: &model::Event,
    at: i64,
) -> model::Placement {
    let chosen = book
        .events
        .iter()
        .position(|candidate| std::ptr::eq(candidate, event))
        .and_then(|index| choices.side_of(index));
    match chosen {
        Some(side) => model::Placement::Person { side },
        None => book.place(event, at),
    }
}

/// A family's [`model::AlikeRange`] as a light's account names it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlikeRangeNote {
    /// The angle of the flats the light's master is built from.
    pub flats_angle: f64,
    pub low: f64,
    pub high: f64,
    /// "101.9–102.2° and 104.2°": the angles at its two ends.
    pub label: String,
    pub pairs: usize,
    pub nights: usize,
    /// Its pairs were all measured in other groups of the family than the
    /// flats', across camera-side dust changes only.
    pub carried: bool,
}

impl AlikeRangeNote {
    pub(super) fn of(
        range: &model::AlikeRange,
        flats_angle: f64,
        flats_group: Option<usize>,
    ) -> Self {
        Self {
            flats_angle,
            low: range.span.low,
            high: range.span.high,
            label: range.label(),
            pairs: range.pairs,
            nights: range.nights,
            carried: flats_group.is_some_and(|group| !range.groups.contains(&group)),
        }
    }
}

impl FlatFamily {
    pub(super) fn of(book: &Book, choice: &lights::FamilyChoice) -> Self {
        let label = |family: usize| family_label(book, family);
        Self {
            key: book.families[choice.family].key.clone(),
            label: label(choice.family),
            basis: choice.basis.clone(),
            pointed: match &choice.basis {
                lights::FamilyBasis::Disputed { pointed, .. } => Some(label(*pointed)),
                _ => None,
            },
            at_recorded: choice.at_recorded,
            light_angle: choice.light_angle,
            shift: choice.shift,
            candidates: choice.candidates.iter().map(|f| label(*f)).collect(),
            light_rotation: LightRotation::Recorded,
            alike_range: None,
        }
    }

    /// Whether the light's own pixels chose it.
    pub fn by_pixels(&self) -> bool {
        matches!(self.basis, lights::FamilyBasis::Pixels { .. })
    }

    /// Whether the light is worth a look for it: its pixels pointed to
    /// another family and nothing confirmed it.
    pub fn flags(&self) -> bool {
        matches!(self.basis, lights::FamilyBasis::Disputed { .. })
    }
}

pub fn correction_key(
    before: &str,
    after: &str,
    side: Side,
    pedestals: (Option<f64>, Option<f64>),
) -> String {
    let digest = Sha256::digest(
        format!(
            "{}.{}\u{1f}{before}\u{1f}{after}\u{1f}{side:?}\u{1f}{:?}\u{1f}{:?}",
            metric::COMPARE_ALGORITHM_VERSION,
            metric::CHANGE_MAP_ALGORITHM_VERSION,
            pedestals.0,
            pedestals.1
        )
        .as_bytes(),
    );
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}
