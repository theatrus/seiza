//! Explicit supplied-set policy. Automatic defaults retain the existing library
//! contract. OneSet removes temporal/cohort selection, never physical gates.
use super::*;
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Association {
    /// Each light's frames chosen by time, cohort and validity.
    #[default]
    Automatic,
    /// Every compatible frame of the kind taken whole, as one supplied set.
    OneSet,
}
words!(Association {
    Automatic => "automatic",
    OneSet => "one_set",
});

/// How a light's flats and darks are chosen: each by time, or as one
/// supplied set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Associations {
    pub flat: Association,
    pub dark: Association,
}
impl Associations {
    /// The kinds taken as one supplied set.
    pub fn one_set(self) -> BTreeSet<CalibrationKind> {
        [
            (CalibrationKind::Flat, self.flat),
            (CalibrationKind::Dark, self.dark),
        ]
        .into_iter()
        .filter(|(_, association)| *association == Association::OneSet)
        .map(|(kind, _)| kind)
        .collect()
    }

    /// Whether `frames` can stand as one supplied set for each kind taken
    /// whole: raw frames that integrate together, or one existing master.
    /// Masters a rig ignores take no part.
    pub fn check(self, frames: &[CalibrationFrame], rules: &RuleBook) -> crate::Result<()> {
        validate(frames, rules, self)
    }
}
fn validate(
    frames: &[CalibrationFrame],
    rules: &RuleBook,
    associations: Associations,
) -> crate::Result<()> {
    for (kind, mode) in [
        (CalibrationKind::Flat, associations.flat),
        (CalibrationKind::Dark, associations.dark),
    ] {
        if mode != Association::OneSet {
            continue;
        }
        // Masters the rig ignores take no part in its set.
        let kept: Vec<_> = frames
            .iter()
            .filter(|f| {
                f.kind == kind
                    && f.kept_out.is_none()
                    && !(f.is_master
                        && rules.for_rig(&f.rig_uuid).external_masters
                            == ExternalMasterPolicy::Ignore)
            })
            .collect();
        if kept.iter().filter(|f| f.is_master).count() > 1
            || (kept.iter().any(|f| f.is_master) && kept.iter().any(|f| !f.is_master))
        {
            return Err(invalid(format!(
                "one supplied {kind:?} set needs raw frames or one existing master, not several masters or a mixture"
            )));
        }
        for (index, a) in kept.iter().enumerate() {
            let tolerance = rules.for_rig(&a.rig_uuid).coherence_tolerances();
            for b in kept.iter().skip(index + 1) {
                let (sa, sb) = (frame_signature(a), frame_signature(b));
                let temperatures = match (a.camera_temp, b.camera_temp) {
                    (Some(a), Some(b)) => (a - b).abs() <= tolerance.master_temperature_c,
                    _ => true,
                };
                if a.rig_uuid != b.rig_uuid
                    || !integrates_with(kind, &sa, &sb, &tolerance)
                    || !temperatures
                    || (kind == CalibrationKind::Dark
                        && !exposure_matches(a.exposure_s, b.exposure_s))
                {
                    return Err(invalid(format!(
                        "{} and {} cannot form one compatible {kind:?} set; use automatic grouping or choose matching frames",
                        a.frame_uuid, b.frame_uuid
                    )));
                }
            }
        }
    }
    Ok(())
}
pub(super) fn apply(
    library: &Library,
    light: &Light,
    selected: &mut CalibrationSelection,
    associations: Associations,
) {
    let frames = library.frames();
    let all = |kind| {
        frames
            .iter()
            .filter(|f| f.kind == kind && f.kept_out.is_none())
            .cloned()
            .collect::<Vec<_>>()
    };
    let rules = &selected.rules;
    if associations.dark == Association::OneSet && rules.use_darks {
        let candidates = all(CalibrationKind::Dark);
        selected.library.dark = frames.iter().any(|f| f.kind == CalibrationKind::Dark);
        selected.dark = if candidates.iter().all(|f| {
            sensor_matches(light, f)
                && exposure_matches(light.exposure_s, f.exposure_s)
                && temperature_admits(light.camera_temp, f, rules)
        }) {
            candidates
        } else {
            Vec::new()
        };
        selected.one_set.insert(CalibrationKind::Dark);
    }
    if associations.flat == Association::OneSet {
        let candidates = all(CalibrationKind::Flat);
        selected.library.flat = frames.iter().any(|f| f.kind == CalibrationKind::Flat);
        selected.flat = if candidates
            .iter()
            .all(|f| sensor_matches(light, f) && flat_matches(light, f, rules))
        {
            candidates
        } else {
            Vec::new()
        };
        selected.flat_basis = Default::default();
        selected.flat_family = None;
        selected.flat_held_back.clear();
        selected.flat_choices.clear();
        selected.one_set.insert(CalibrationKind::Flat);
        library.select_dark_flats(light, &Arc::clone(&selected.rules), selected);
    }
}
