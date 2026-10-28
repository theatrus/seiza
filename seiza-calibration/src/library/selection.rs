//! Matching lights to calibration frames, and the cohort a master takes.
//!
//! Extracted from theatrus/psf-guard at 79f5e31d53b8371e8fcd0b22a9b69692412cd8b4.
//! Copyright the PSF Guard contributors; Apache-2.0.
use super::*;

/// A frame's settings as matching compares them: the crate's
/// [`FrameSignature`], and the readout mode's name, matched apart from its
/// numeric index.
#[derive(Clone, Debug, PartialEq)]
pub struct Signature {
    pub seiza: FrameSignature,
    pub readout_mode_name: Option<String>,
}

impl Light {
    /// Its settings as matching compares them.
    pub fn signature(&self) -> Signature {
        light_signature(self)
    }
}

impl CalibrationFrame {
    /// Its settings as matching compares them.
    pub fn signature(&self) -> Signature {
        frame_signature(self)
    }
}

/// The frames a master integrates from a set of candidates, and those that
/// clustered with them but would not integrate with its first.
#[derive(Clone, Debug, Default)]
pub struct Cohort {
    pub kept: Vec<CalibrationFrame>,
    pub dropped: Vec<CalibrationFrame>,
}

impl Library {
    pub(super) fn of_kind(&self, kind: CalibrationKind) -> impl Iterator<Item = &CalibrationFrame> {
        self.frames.iter().filter(move |frame| frame.kind == kind)
    }
    /// A person's choices for this light's flats, resolved against the
    /// book. Grouping by time keeps every choice and applies none.
    pub fn light_choices(
        &self,
        light: &Light,
        rules: &Rules,
    ) -> flat_groups::overrides::ResolvedChoices {
        let book = &self.flat_book;
        let Some(name) = light
            .path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
        else {
            return Default::default();
        };
        // A path-scoped host can distinguish identical basenames captured at
        // the same time. Existing catalog clients retain basename lookup.
        let full_path = light.path.to_string_lossy();
        let name = if book.light_choices.contains_key(full_path.as_ref()) {
            full_path.as_ref()
        } else {
            &name
        };
        let mut choices = book.light_choices(
            name,
            light.filter.as_deref(),
            &light.rig_uuid.clone(),
            light.timestamp,
        );
        if rules.flat_grouping != rules::FlatGrouping::Measured {
            let reason = "this rig groups its flats by time, so no change or group is measured";
            let mut kept = std::mem::take(&mut choices.orphaned);
            kept.extend(
                choices
                    .sides
                    .drain(..)
                    .map(|(_, _, row)| (row, reason.to_string())),
            );
            if let Some(pick) = choices.group.take() {
                kept.push((pick.row, reason.to_string()));
            }
            choices.orphaned = kept;
        }
        choices
    }
    pub(super) fn select_choosing(
        &self,
        light: &Light,
        rules: Arc<Rules>,
        choices: flat_groups::overrides::ResolvedChoices,
    ) -> CalibrationSelection {
        let mut selected = CalibrationSelection {
            rules: Arc::clone(&rules),
            ..Default::default()
        };
        let policy = rules.external_masters;
        // A group a person chose names the family too.
        let family = match &choices.group {
            Some(pick) => Some(flat_groups::lights::FamilyChoice {
                family: pick.family,
                basis: flat_groups::lights::FamilyBasis::Person,
                candidates: vec![pick.family],
                light_angle: light
                    .rotator_position
                    .filter(|a| a.is_finite())
                    .or(self.flat_book.families[pick.family].rotation)
                    .unwrap_or(0.0),
                shift: 0.0,
                at_recorded: false,
            }),
            None => self.flat_family(light, &rules),
        };
        // The family serves the light's flats when it has flats at the
        // angle the light recorded, its own pixels chose it, or a person
        // did. A family its lights only pointed to is a Check, not flats.
        let serving = family.as_ref().filter(|choice| {
            choice.at_recorded
                || choice.by_pixels()
                || choice.basis == flat_groups::lights::FamilyBasis::Person
        });
        // Flats of the serving family at another angle than the light's,
        // admitted by measurement: they stay candidates only where nothing
        // at the light's own angle can serve it (below), so a light that
        // never needs them keeps the candidates, and the stack key, it had.
        let mut off_angle: HashSet<String> = HashSet::new();
        for candidate in &self.frames {
            if candidate.is_master && policy == ExternalMasterPolicy::Ignore {
                continue;
            }
            let sensor = sensor_matches(light, candidate);
            // A rig that does without darks takes none, and lacks none.
            if candidate.kind == CalibrationKind::Dark && !rules.use_darks {
                selected.library.unused_dark |= sensor;
                continue;
            }
            // A frame kept out still shows the library holds this kind, so a
            // light whose only flats are poor is reported as one that cannot
            // be calibrated rather than as one whose rig shoots none.
            if sensor {
                selected.library.note(candidate.kind);
            }
            // A frame marked usable only forward or backward of its capture —
            // shot around an optics change or cleaning — never serves a light
            // on the other side of that boundary.
            if !validity_admits(candidate, light.timestamp) {
                continue;
            }
            if !sensor {
                continue;
            }
            let matches = match candidate.kind {
                CalibrationKind::Bias => true,
                CalibrationKind::Dark => {
                    exposure_matches(light.exposure_s, candidate.exposure_s)
                        && temperature_admits(light.camera_temp, candidate, &rules)
                        && dark_within_reach(candidate, light.timestamp, &rules)
                }
                CalibrationKind::DarkFlat => false,
                CalibrationKind::Flat => {
                    match self.family_admits(serving, light, candidate, &rules, &choices) {
                        Some(admitted) => {
                            if admitted && !self.at_lights_angle(light, candidate, &rules) {
                                off_angle.insert(candidate.frame_uuid.clone());
                            }
                            admitted && {
                                // Its family was chosen for the light: matched at
                                // the light's angle, every other gate as always.
                                let mut at_angle = candidate.clone();
                                at_angle.rotation = light.rotator_position;
                                flat_matches(light, &at_angle, &rules)
                            }
                        }
                        None => match self.measured_angle(candidate, &rules) {
                            // Flats that record no angle, measured alike with a
                            // rotation cluster: matched at its angle.
                            Some(angle) if candidate.rotation.is_none() => {
                                let mut at_angle = candidate.clone();
                                at_angle.rotation = Some(angle);
                                flat_matches(light, &at_angle, &rules)
                            }
                            _ => flat_matches(light, candidate, &rules),
                        },
                    }
                }
            };
            if matches && candidate.kept_out.is_some() {
                selected.kept_out.push(candidate.clone());
                continue;
            }
            if matches {
                match candidate.kind {
                    CalibrationKind::Bias => selected.bias.push(candidate.clone()),
                    CalibrationKind::Dark => selected.dark.push(candidate.clone()),
                    CalibrationKind::Flat => selected.flat.push(candidate.clone()),
                    CalibrationKind::DarkFlat => {}
                }
            }
        }

        // A group a person chose keeps its flats at any angle of its family.
        if !off_angle.is_empty() && choices.group.is_none() {
            // One at the light's own angle that can reach it leaves them
            // nothing to do; one across a known change from the light
            // could serve it only as its own angle's flats would, carried
            // or flagged, which they are not.
            let own_angle_serves = selected.flat.iter().any(|frame| {
                !off_angle.contains(&frame.frame_uuid)
                    && self.at_lights_angle(light, frame, &rules)
                    && self.not_across_a_known_change(light, frame, &choices)
            });
            let keep = |frame: &CalibrationFrame| {
                !off_angle.contains(&frame.frame_uuid)
                    || (!own_angle_serves && self.not_across_a_known_change(light, frame, &choices))
            };
            selected.flat.retain(|frame| keep(frame));
            selected.kept_out.retain(|frame| keep(frame));
        }
        sort_candidates(&mut selected.bias, light.timestamp);
        sort_candidates(&mut selected.dark, light.timestamp);
        sort_candidates(&mut selected.flat, light.timestamp);
        if let Some(family) = serving {
            // Within its family, the flats at the angle the light recorded
            // first, as before families: measured alike is not identical.
            // Then those within the angle prior, then the rest, each in
            // time order: the family's far angles serve where nothing
            // nearer lies on the light's side of a change. The epochs widen
            // which families are candidates; they do not reorder a
            // family's flats, since a sync's degree or two would.
            let tier = |frame: &CalibrationFrame| match self.measured_angle(frame, &rules) {
                Some(flat) if rotation_gap(family.light_angle, flat) <= rules.rotation_deg => 0,
                Some(flat)
                    if rotation_gap(family.light_angle, flat)
                        <= flat_groups::lights::ANGLE_PRIOR_DEG =>
                {
                    1
                }
                _ => 2,
            };
            selected.flat.sort_by_key(tier);
        }
        if let Some(pick) = &choices.group {
            // The group a person chose, over every change: its flats only.
            let book = &self.flat_book;
            let sessions = &book.families[pick.family].groups[pick.group];
            let kept: Vec<CalibrationFrame> = selected
                .flat
                .iter()
                .filter(|frame| {
                    book.session_of_frame(&frame.frame_uuid)
                        .is_some_and(|session| sessions.contains(&session))
                })
                .cloned()
                .collect();
            if kept.is_empty() {
                // None of its flats match this light: matched as though
                // nobody chose, and the choice shows why it is not used.
                let mut fallback = choices.clone();
                fallback.group = None;
                fallback.orphaned.push((
                    pick.row.clone(),
                    "none of that group's flats match this light".into(),
                ));
                return self.select_choosing(light, rules, fallback);
            }
            selected.flat = kept;
            if let Some(group) = flat_groups::GroupNote::of(book, pick.family, pick.group) {
                selected.flat_basis = flat_groups::FlatBasis::Group { group };
            }
        } else {
            self.keep_unknown_angles_last(light, &rules, &mut selected.flat, &choices);
            self.choose_flats_around_changes(light, &rules, &mut selected, &choices);
        }
        selected.flat_choices = choice_notes(&choices);
        selected.flat_family = family.map(|choice| {
            let mut family = flat_groups::FlatFamily::of(&self.flat_book, &choice);
            family.light_rotation = family_light_rotation(light, &selected.flat, &rules);
            // Its flats lie at another angle that the family measures
            // alike with the light's: say over which span, and on what.
            if let flat_groups::LightRotation::Measured { degrees } = family.light_rotation {
                let book = &self.flat_book;
                family.alike_range = selected
                    .flat
                    .iter()
                    .filter(|frame| frame.rotation.is_some_and(|r| (r - degrees).abs() < 1e-6))
                    .find_map(|frame| book.session_of_frame(&frame.frame_uuid))
                    .and_then(|session| {
                        let range = book.alike_range_for(
                            choice.family,
                            session,
                            choice.light_angle,
                            rules.rotation_deg,
                        )?;
                        let group = book.families[choice.family]
                            .groups
                            .iter()
                            .position(|group| group.contains(&session));
                        Some(flat_groups::AlikeRangeNote::of(range, degrees, group))
                    });
            }
            family
        });
        self.select_dark_flats(light, &rules, &mut selected);
        selected
    }
    pub(super) fn select_dark_flats(
        &self,
        light: &Light,
        rules: &Rules,
        selected: &mut CalibrationSelection,
    ) {
        selected.dark_flat.clear();
        if let Some(flat) = selected.flat.first() {
            for candidate in self.of_kind(CalibrationKind::DarkFlat) {
                // Dark-flats pair with the flat, but the boundary is judged
                // against the light: the mark protects the imaging on the
                // other side of the change.
                if validity_admits(candidate, light.timestamp)
                    && frame_pair_matches(flat, candidate)
                    && exposure_matches(flat.exposure_s, candidate.exposure_s)
                    && temperature_matches(flat.camera_temp, candidate.camera_temp, rules)
                {
                    if candidate.kept_out.is_some() {
                        selected.kept_out.push(candidate.clone());
                    } else {
                        selected.dark_flat.push(candidate.clone());
                    }
                }
            }
        }
        sort_candidates(&mut selected.dark_flat, light.timestamp);
    }
    pub(super) fn flat_family(
        &self,
        light: &Light,
        rules: &Rules,
    ) -> Option<flat_groups::lights::FamilyChoice> {
        if rules.flat_grouping != rules::FlatGrouping::Measured
            || self.flat_book.families.is_empty()
        {
            return None;
        }
        let angle = light.rotator_position.filter(|a| a.is_finite())?;
        let compatible = self.flat_family_compatible(light, rules);
        self.flat_book.choose_family(
            &light.rig_uuid.clone(),
            light.filter.as_deref(),
            light.timestamp,
            Some(angle),
            &compatible,
            rules.rotation_deg,
        )
    }
    /// The response families whose flats could serve `light` by sensor,
    /// optics and validity, at any angle: the candidates its own pixels may
    /// choose among.
    pub fn flat_family_compatible(&self, light: &Light, rules: &Rules) -> Vec<usize> {
        let book = &self.flat_book;
        let light_signature = light_signature(light);
        let mut tolerances = rules.tolerances();
        tolerances.rotation_deg = crate::MatchTolerances::MAX_ROTATION_DEG;
        let mut compatible: Vec<usize> = self
            .of_kind(CalibrationKind::Flat)
            .filter(|frame| {
                !frame.is_master
                    && frame.kept_out.is_none()
                    && validity_admits(frame, light.timestamp)
                    && sensor_matches(light, frame)
                    && crate::optics_match(
                        &light_signature.seiza,
                        &frame_signature(frame).seiza,
                        &tolerances,
                    )
            })
            .filter_map(|frame| book.family_of_frame(&frame.frame_uuid))
            .collect();
        compatible.sort_unstable();
        compatible.dedup();
        compatible
    }
    pub(super) fn family_admits(
        &self,
        family: Option<&flat_groups::lights::FamilyChoice>,
        light: &Light,
        candidate: &CalibrationFrame,
        rules: &Rules,
        choices: &flat_groups::overrides::ResolvedChoices,
    ) -> Option<bool> {
        let family = family?;
        let book = &self.flat_book;
        match book.family_of_frame(&candidate.frame_uuid) {
            // A person chose a group of this family: any of its flats, the
            // group picked out after.
            Some(of)
                if of == family.family
                    && family.basis == flat_groups::lights::FamilyBasis::Person =>
            {
                Some(true)
            }
            Some(of) if of == family.family => {
                let angle = self.measured_angle(candidate, rules);
                let at_angle = angle.is_some_and(|flat| {
                    rotation_gap(family.light_angle, flat) <= rules.rotation_deg
                });
                let in_range = || {
                    book.session_of_frame(&candidate.frame_uuid)
                        .and_then(|session| {
                            book.alike_range_for(
                                of,
                                session,
                                family.light_angle,
                                rules.rotation_deg,
                            )
                        })
                        .is_some()
                        && self.on_lights_side_of_changes(light, candidate, choices)
                };
                Some(
                    at_angle
                        || family.by_pixels()
                        || in_range()
                        || book
                            .session_of_frame(&candidate.frame_uuid)
                            .is_some_and(|session| {
                                book.alike_with_angle(
                                    of,
                                    session,
                                    family.light_angle,
                                    rules.rotation_deg,
                                )
                            }),
                )
            }
            Some(of) if book.families[of].angles.is_empty() => None,
            Some(_) => Some(false),
            None if family.moved() => Some(false),
            None => None,
        }
    }
    pub(super) fn at_lights_angle(
        &self,
        light: &Light,
        frame: &CalibrationFrame,
        rules: &Rules,
    ) -> bool {
        match (light.rotator_position, self.measured_angle(frame, rules)) {
            (Some(light), Some(flat)) => rotation_gap(light, flat) <= rules.rotation_deg,
            _ => false,
        }
    }
    pub(super) fn not_across_a_known_change(
        &self,
        light: &Light,
        candidate: &CalibrationFrame,
        choices: &flat_groups::overrides::ResolvedChoices,
    ) -> bool {
        let book = &self.flat_book;
        let rig = light.rig_uuid.clone();
        let Some(at) = light.timestamp else {
            return true;
        };
        book.events
            .iter()
            .filter(|event| event.rig_uuid == rig && event.applies_to(light.filter.as_deref()))
            .all(
                |event| match flat_groups::place_light(book, choices, event, at).side() {
                    Some(side) => candidate
                        .captured_at
                        .is_none_or(|captured| event.side_of(captured) == Some(side)),
                    None => true,
                },
            )
    }
    pub(super) fn on_lights_side_of_changes(
        &self,
        light: &Light,
        candidate: &CalibrationFrame,
        choices: &flat_groups::overrides::ResolvedChoices,
    ) -> bool {
        let book = &self.flat_book;
        let rig = light.rig_uuid.clone();
        let (Some(at), Some(captured)) = (light.timestamp, candidate.captured_at) else {
            return false;
        };
        book.events
            .iter()
            .filter(|event| event.rig_uuid == rig && event.applies_to(light.filter.as_deref()))
            .all(
                |event| match flat_groups::place_light(book, choices, event, at).side() {
                    Some(side) => event.side_of(captured) == Some(side),
                    None => false,
                },
            )
    }
    pub(super) fn measured_angle(&self, frame: &CalibrationFrame, rules: &Rules) -> Option<f64> {
        frame
            .rotation
            .filter(|angle| angle.is_finite())
            .or_else(|| {
                (rules.flat_grouping == rules::FlatGrouping::Measured)
                    .then(|| self.flat_book.inferred_rotation(&frame.frame_uuid))
                    .flatten()
            })
    }
    pub(super) fn keep_unknown_angles_last(
        &self,
        light: &Light,
        rules: &Rules,
        flats: &mut Vec<CalibrationFrame>,
        choices: &flat_groups::overrides::ResolvedChoices,
    ) {
        if rules.flat_grouping != rules::FlatGrouping::Measured
            || light
                .rotator_position
                .is_none_or(|angle| !angle.is_finite())
        {
            return;
        }
        let unknown = |frame: &CalibrationFrame| {
            !frame.is_master && self.measured_angle(frame, rules).is_none()
        };
        // The light's side of every measured change it stands clear of.
        let book = &self.flat_book;
        let rig = light.rig_uuid.clone();
        let sides: Vec<(&flat_groups::model::Event, flat_groups::model::Side)> =
            match light.timestamp {
                Some(at) => book
                    .events_for(&rig, light.filter.as_deref())
                    .filter_map(|event| {
                        Some((
                            event,
                            flat_groups::place_light(book, choices, event, at).side()?,
                        ))
                    })
                    .collect(),
                None => Vec::new(),
            };
        let on_side = |frame: &CalibrationFrame| {
            sides.iter().all(|(event, side)| {
                frame
                    .captured_at
                    .and_then(|captured| event.side_of(captured))
                    == Some(*side)
            })
        };
        let known: Vec<&CalibrationFrame> = flats
            .iter()
            .filter(|frame| !frame.is_master && !unknown(frame))
            .collect();
        // Unknown-angle flats that are alone on the light's side of a
        // change are its own side's flats: the change logic weighs them.
        let alone_on_side = !known.iter().any(|frame| on_side(frame))
            && flats.iter().any(|frame| unknown(frame) && on_side(frame));
        if !known.is_empty() && !alone_on_side {
            flats.retain(|frame| !unknown(frame));
        }
    }
    pub(super) fn choose_flats_around_changes(
        &self,
        light: &Light,
        rules: &Rules,
        selected: &mut CalibrationSelection,
        choices: &flat_groups::overrides::ResolvedChoices,
    ) {
        use flat_groups::{EventNote, FlatBasis};
        if rules.flat_grouping != rules::FlatGrouping::Measured {
            return;
        }
        let Some(at) = light.timestamp else {
            return;
        };
        let book = Arc::clone(&self.flat_book);
        let rig = light.rig_uuid.clone();
        let events: Vec<(usize, &flat_groups::model::Event)> = book
            .events
            .iter()
            .enumerate()
            .filter(|(_, event)| event.rig_uuid == rig && event.applies_to(light.filter.as_deref()))
            .collect();
        if events.is_empty() || selected.flat.is_empty() {
            return;
        }
        let placements: Vec<(usize, flat_groups::model::Placement)> = events
            .iter()
            .map(|(index, event)| (*index, flat_groups::place_light(&book, choices, event, at)))
            .collect();
        let side_of = |frame: &CalibrationFrame, event: &flat_groups::model::Event| {
            frame
                .captured_at
                .and_then(|captured| event.side_of(captured))
        };
        // Same side of every event whose side is known.
        let on_side = |frame: &CalibrationFrame| {
            placements
                .iter()
                .all(|(index, placement)| match placement.side() {
                    Some(side) => side_of(frame, &book.events[*index]) == Some(side),
                    None => true,
                })
        };
        for (index, placement) in &placements {
            let event = &book.events[*index];
            let note = EventNote::of(event, rules);
            let Some(side) = placement.side() else {
                // Shot while it happened: whichever flats it takes may be
                // the wrong ones, unless they all lie on one side and that
                // side is ruled out by nothing. Say so either way.
                if selected
                    .flat
                    .iter()
                    .any(|frame| side_of(frame, event).is_some())
                {
                    selected.flat_basis = std::mem::take(&mut selected.flat_basis).followed_by(
                        FlatBasis::Undetermined {
                            event: note,
                            placement: placement.clone(),
                            then: None,
                        },
                    );
                }
                continue;
            };
            let before = selected.flat.len();
            let kept: Vec<CalibrationFrame> = selected
                .flat
                .iter()
                .filter(|frame| on_side(frame))
                .cloned()
                .collect();
            if kept.len() == before {
                continue;
            }
            if !kept.is_empty() {
                // Flats from the other side that no master of this light
                // could reach stay in the list, so its stacks keep their
                // keys: the nearest flat is on its side with another
                // beside it, and every flat dropped is further than twice
                // the flat session from it.
                let reach = book.limits(&rig).map_or(0, |limits| 2 * limits.window);
                let apart = |a: &CalibrationFrame, b: &CalibrationFrame| match (
                    a.captured_at,
                    b.captured_at,
                ) {
                    (Some(a), Some(b)) => (a - b).abs(),
                    _ => 0,
                };
                let anchor = &kept[0];
                let unreachable = on_side(&selected.flat[0])
                    && kept
                        .iter()
                        .filter(|frame| apart(frame, anchor) <= reach / 2)
                        .count()
                        >= 2
                    && selected
                        .flat
                        .iter()
                        .filter(|frame| !on_side(frame))
                        .all(|frame| apart(frame, anchor) > reach);
                if unreachable {
                    for frame in selected.flat.iter().filter(|frame| !on_side(frame)) {
                        if !selected.flat_held_back.contains(&frame.frame_uuid) {
                            selected.flat_held_back.push(frame.frame_uuid.clone());
                        }
                    }
                    continue;
                }
                selected.flat = kept;
                if !selected.flat_basis.flags() {
                    selected.flat_basis = FlatBasis::SameSide {
                        event: note,
                        placement: placement.clone(),
                        left_out: before - selected.flat.len(),
                    };
                }
                continue;
            }
            // The nearest flat, which its master is built around, lies on
            // its side of this change: another change keeps it from the
            // light, and the fallback is that one's to choose.
            if selected
                .flat
                .first()
                .is_some_and(|frame| side_of(frame, event) == Some(side))
            {
                continue;
            }
            // Nothing at its angle on its side. First: its side at the
            // nearest other angle, when the rig's angle term is measured
            // and within the limits.
            let limits = book.limits(&rig);
            let angle_term = book.angle_terms.get(&rig).copied();
            let angle_ok = match (angle_term, limits) {
                (Some(term), Some(limits)) => {
                    term.pairs > 0 && term.dust <= limits.dust && term.broad <= limits.broad
                }
                _ => false,
            };
            if angle_ok {
                let other = self.other_angle_flats(light, rules, &on_side);
                if let Some((rotation, frames)) = other {
                    selected.flat = frames;
                    // Wide enough to admit exactly these flats: their angle
                    // from the light, plus the rig's own tolerance.
                    let rotation_tolerance_deg = match (light.rotator_position, rotation) {
                        (Some(light_at), Some(flats_at)) => Some(
                            (rotation_gap(light_at, flats_at) + rules.rotation_deg)
                                .clamp(rules.rotation_deg, 180.0),
                        ),
                        _ => None,
                    };
                    selected.flat_basis = std::mem::take(&mut selected.flat_basis).followed_by(
                        FlatBasis::OtherAngle {
                            event: note,
                            placement: placement.clone(),
                            rotation,
                            angle_term: angle_term.unwrap_or_default(),
                            rotation_tolerance_deg,
                        },
                    );
                    continue;
                }
            }
            // Then: the flats at its angle from the other side, carried
            // across by the change's map, when the change is on the camera
            // side and so the same at every angle. One map only: flats
            // across a second change would need it carried too, so a light
            // already corrected, or already left with the other side's
            // flats, is left with them and keeps its Check.
            if selected.flat_basis.lacks_same_side() {
                continue;
            }
            if event.camera_side
                && selected.flat_basis.correction().is_none()
                && let Some((before_key, after_key)) = book.map_source_for(
                    *index,
                    light.filter.as_deref(),
                    selected.flat.first().map(|frame| frame.frame_uuid.as_str()),
                )
                && self.can_correct(&before_key, &after_key)
            {
                let pedestal = |key: &str| {
                    book.session_by_response(key)
                        .and_then(|index| book.sessions[index].row.pedestal_adu)
                };
                let key = flat_groups::correction_key(
                    &before_key,
                    &after_key,
                    side,
                    (pedestal(&before_key), pedestal(&after_key)),
                );
                selected.flat_basis =
                    std::mem::take(&mut selected.flat_basis).followed_by(FlatBasis::Corrected {
                        event: note,
                        placement: placement.clone(),
                        correction: flat_groups::FlatCorrection {
                            event: event.key.clone(),
                            before_response: before_key,
                            after_response: after_key,
                            light_side: side,
                            key,
                        },
                    });
                continue;
            }
            selected.flat_basis =
                std::mem::take(&mut selected.flat_basis).followed_by(FlatBasis::NoSameSide {
                    event: note,
                    placement: placement.clone(),
                });
        }
    }
    pub(super) fn other_angle_flats(
        &self,
        light: &Light,
        rules: &Rules,
        keep: &dyn Fn(&CalibrationFrame) -> bool,
    ) -> Option<(Option<f64>, Vec<CalibrationFrame>)> {
        let light_signature = light_signature(light);
        let mut tolerances = rules.tolerances();
        tolerances.rotation_deg = 360.0;
        let candidates: Vec<&CalibrationFrame> = self
            .of_kind(CalibrationKind::Flat)
            .filter(|frame| {
                !frame.is_master
                    && frame.kept_out.is_none()
                    && validity_admits(frame, light.timestamp)
                    && sensor_matches(light, frame)
                    && crate::optics_match(
                        &light_signature.seiza,
                        &frame_signature(frame).seiza,
                        &tolerances,
                    )
                    && keep(frame)
            })
            .collect();
        // A flat of unknown angle is no nearer one angle than another.
        let candidates: Vec<&CalibrationFrame> = candidates
            .into_iter()
            .filter(|frame| {
                light.rotator_position.is_none() || self.measured_angle(frame, rules).is_some()
            })
            .collect();
        let gap = |frame: &CalibrationFrame| match (
            light.rotator_position,
            self.measured_angle(frame, rules),
        ) {
            (Some(a), Some(b)) => rotation_gap(a, b),
            _ => 0.0,
        };
        let nearest = candidates
            .iter()
            .map(|frame| gap(frame))
            .fold(f64::INFINITY, f64::min);
        if !nearest.is_finite() {
            return None;
        }
        let rotation = candidates
            .iter()
            .find(|frame| gap(frame) == nearest)
            .and_then(|frame| self.measured_angle(frame, rules));
        let mut frames: Vec<CalibrationFrame> = candidates
            .into_iter()
            .filter(
                |frame| match (rotation, self.measured_angle(frame, rules)) {
                    (Some(a), Some(b)) => {
                        rotation_gap(a, b) <= rules.coherence_tolerances().rotation_deg
                    }
                    (None, None) => true,
                    _ => false,
                },
            )
            .cloned()
            .collect();
        if frames.is_empty() {
            return None;
        }
        sort_candidates(&mut frames, light.timestamp);
        Some((rotation, frames))
    }
}
pub(super) fn choice_notes(
    choices: &flat_groups::overrides::ResolvedChoices,
) -> Vec<flat_groups::ChoiceNote> {
    choices
        .group
        .iter()
        .map(|pick| &pick.row)
        .chain(choices.sides.iter().map(|(_, _, row)| row))
        .filter_map(|row| flat_groups::ChoiceNote::of(row, None))
        .chain(
            choices
                .orphaned
                .iter()
                .filter_map(|(row, reason)| flat_groups::ChoiceNote::of(row, Some(reason.clone()))),
        )
        .collect()
}
pub(super) fn family_light_rotation(
    light: &Light,
    flats: &[CalibrationFrame],
    rules: &Rules,
) -> flat_groups::LightRotation {
    use flat_groups::LightRotation;
    let subset = coherent_master_subset(CalibrationKind::Flat, flats, rules);
    let Some(angle) = subset
        .iter()
        .find_map(|frame| frame.rotation.filter(|a| a.is_finite()))
    else {
        return LightRotation::Recorded;
    };
    match light.rotator_position.filter(|a| a.is_finite()) {
        Some(recorded)
            if subset.iter().all(|frame| {
                frame
                    .rotation
                    .is_none_or(|flat| rotation_gap(recorded, flat) <= rules.rotation_deg)
            }) =>
        {
            LightRotation::Recorded
        }
        Some(_) => LightRotation::Measured { degrees: angle },
        None => LightRotation::Recorded,
    }
}
/// The angle between two rotator positions, in degrees, the short way
/// round: 0 to 180.
pub fn rotation_gap(left: f64, right: f64) -> f64 {
    let difference = (left - right).rem_euclid(360.0);
    difference.min(360.0 - difference)
}
/// Whether a raw `candidate` was shot with the `reference`'s sensor
/// settings: camera, size, binning, gain, offset and readout mode.
pub fn signature_sensor_matches(reference: &Signature, candidate: &Signature) -> bool {
    crate::sensor_matches(&reference.seiza, &candidate.seiza)
        && readout_names_agree(reference, candidate)
}
/// Whether two signatures do not contradict each other's sensor settings:
/// the lenient test an external master takes, its producer having
/// stripped most of them.
pub fn signature_sensor_consistent(left: &Signature, right: &Signature) -> bool {
    crate::sensor_consistent(&left.seiza, &right.seiza) && readout_names_agree(left, right)
}
/// Whether two readout mode names agree, or either is unknown.
pub fn readout_names_agree(left: &Signature, right: &Signature) -> bool {
    match (
        named_mode(&left.readout_mode_name),
        named_mode(&right.readout_mode_name),
    ) {
        (Some(left), Some(right)) => readout_names_equal(left, right),
        _ => true,
    }
}
/// A readout mode name, or `None` for none or a blank one: a header that
/// spelled no name names no mode.
fn named_mode(name: &Option<String>) -> Option<&str> {
    name.as_deref().filter(|name| !name.trim().is_empty())
}
/// Whether two readout mode names are one, ignoring case and surrounding
/// space.
pub fn readout_names_equal(left: &str, right: &str) -> bool {
    left.trim().eq_ignore_ascii_case(right.trim())
}
/// A light's settings as matching compares them: [`Light::signature`].
pub fn light_signature(light: &Light) -> Signature {
    let signature = crate::FrameSignature {
        camera: light.camera.clone(),
        telescope: light.telescope.clone(),
        bayer_pattern: light.bayer_pattern.clone(),
        filter: light.filter.clone(),
        width: light.width,
        height: light.height,
        channels: light.channels,
        binning_x: light.binning_x,
        binning_y: light.binning_y,
        gain: light.gain,
        offset: light.offset,
        readout_mode: light.readout_mode,
        focal_length_mm: light.focal_length_mm,
        rotation_deg: light.rotator_position,
        exposure_seconds: light.exposure_s,
        camera_temp_c: light.camera_temp,
        captured_at_unix: light.timestamp,
        ..Default::default()
    };
    Signature {
        seiza: signature,
        readout_mode_name: named_mode(&light.readout_mode_name).map(str::to_string),
    }
}
/// A frame's settings as matching compares them:
/// [`CalibrationFrame::signature`].
pub fn frame_signature(frame: &CalibrationFrame) -> Signature {
    let signature = crate::FrameSignature {
        camera: frame.camera.clone(),
        telescope: frame.telescope.clone(),
        bayer_pattern: frame.bayer_pattern.clone(),
        filter: frame.filter.clone(),
        width: frame.width,
        height: frame.height,
        channels: frame.channels,
        binning_x: frame.binning_x,
        binning_y: frame.binning_y,
        gain: frame.gain,
        offset: frame.offset,
        readout_mode: frame.readout_mode,
        focal_length_mm: frame.focal_length_mm,
        rotation_deg: frame.rotation,
        exposure_seconds: frame.exposure_s,
        camera_temp_c: frame.camera_temp,
        captured_at_unix: frame.captured_at,
        ..Default::default()
    };
    Signature {
        seiza: signature,
        readout_mode_name: named_mode(&frame.readout_mode_name).map(str::to_string),
    }
}
/// Whether a dark is within the rules' reach of a light shot at
/// `light_timestamp`: days apart, unless a person marked its validity.
pub fn dark_within_reach(
    candidate: &CalibrationFrame,
    light_timestamp: Option<i64>,
    rules: &Rules,
) -> bool {
    if candidate.valid_direction.is_some() {
        return true;
    }
    match (candidate.captured_at, light_timestamp) {
        (Some(captured), Some(light)) => {
            captured.abs_diff(light) as f64 <= rules.dark_reach_days * 86_400.0
        }
        _ => true,
    }
}
pub(super) fn nearest_complete_dark_session(
    kept: Vec<CalibrationFrame>,
    rules: &Rules,
) -> Vec<CalibrationFrame> {
    let Some(nearest) = kept.first().and_then(|frame| frame.captured_at) else {
        return kept;
    };
    let session: Vec<CalibrationFrame> = kept
        .iter()
        .filter(|frame| {
            frame
                .captured_at
                .is_some_and(|at| at.abs_diff(nearest) <= DARK_SESSION_SECONDS)
        })
        .cloned()
        .collect();
    if session.len() >= rules.complete_dark_frames {
        session
    } else {
        kept
    }
}
/// Whether `candidate` shares the light's sensor settings: exactly for a
/// raw frame, leniently for an external master. Unlike the crate's
/// [`crate::sensor_matches`], it takes a library light and frame.
pub fn sensor_matches(light: &Light, candidate: &CalibrationFrame) -> bool {
    let (light, candidate_signature) = (light_signature(light), frame_signature(candidate));
    if candidate.is_master {
        signature_sensor_consistent(&light, &candidate_signature)
    } else {
        signature_sensor_matches(&light, &candidate_signature)
    }
}
/// Whether a flat `candidate` shares the light's optics within the rules'
/// tolerances: rotation, focal length and filter.
pub fn flat_matches(light: &Light, candidate: &CalibrationFrame, rules: &Rules) -> bool {
    let (light, candidate_signature) = (light_signature(light), frame_signature(candidate));
    let tolerances = rules.tolerances();
    if candidate.is_master {
        crate::optics_consistent(&light.seiza, &candidate_signature.seiza, &tolerances)
    } else {
        crate::optics_match(&light.seiza, &candidate_signature.seiza, &tolerances)
    }
}
/// Whether a dark's temperature admits it for a light at `light_temp`; an
/// external master that recorded none is admitted.
pub fn temperature_admits(
    light_temp: Option<f64>,
    candidate: &CalibrationFrame,
    rules: &Rules,
) -> bool {
    if candidate.is_master && candidate.camera_temp.is_none() {
        return true;
    }
    temperature_matches(light_temp, candidate.camera_temp, rules)
}
/// Whether two raw frames share their sensor settings.
pub fn frame_pair_matches(left: &CalibrationFrame, right: &CalibrationFrame) -> bool {
    signature_sensor_matches(&frame_signature(left), &frame_signature(right))
}
/// Whether two exposures, in seconds, match as a dark and light must;
/// either unknown matches. Unlike the crate's [`crate::exposure_matches`],
/// it takes the exposures themselves.
pub fn exposure_matches(left: Option<f64>, right: Option<f64>) -> bool {
    let signature = |exposure| crate::FrameSignature {
        exposure_seconds: exposure,
        ..Default::default()
    };
    crate::exposure_matches(
        &signature(left),
        &signature(right),
        &crate::MatchTolerances::default(),
    )
}
/// Whether two sensor temperatures, in °C, are within the rules' dark
/// tolerance; either unknown matches.
pub fn temperature_matches(left: Option<f64>, right: Option<f64>, rules: &Rules) -> bool {
    let signature = |temperature| crate::FrameSignature {
        camera_temp_c: temperature,
        ..Default::default()
    };
    crate::temperature_matches(&signature(left), &signature(right), &rules.tolerances())
}
pub(super) fn coherent_master_subset(
    kind: CalibrationKind,
    frames: &[CalibrationFrame],
    rules: &Rules,
) -> Vec<CalibrationFrame> {
    coherent_cohort(kind, frames, rules).kept
}
/// The frames one master of `kind` integrates from `frames`, given nearest
/// first: one readout mode, then the nearest cluster of agreeing frames (a
/// complete set first for darks and biases unless the rules take the
/// nearest), anchored on the first as the integrator anchors it. A dark
/// master keeps to its nearest complete night.
pub fn coherent_cohort(
    kind: CalibrationKind,
    frames: &[CalibrationFrame],
    rules: &Rules,
) -> Cohort {
    let mut subset = master_subset_by_mode(kind, frames, rules);
    if kind == CalibrationKind::Dark {
        subset.kept = nearest_complete_dark_session(subset.kept, rules);
    }
    subset
}
fn master_subset_by_mode(
    kind: CalibrationKind,
    frames: &[CalibrationFrame],
    rules: &Rules,
) -> Cohort {
    // Seiza clusters by temperature, session and angle, and knows nothing
    // of the readout mode's name. Left to the anchor test below, a nearest
    // stray in one mode would set aside a whole set in the other and blame
    // the filter. So the name partitions first, the way Seiza's own session
    // rule works: nearest-first, the first name group with enough frames
    // wins, and frames that never named a mode ride along with any group.
    let mut tried: Vec<&str> = Vec::new();
    for frame in frames {
        let Some(name) = named_mode(&frame.readout_mode_name) else {
            continue;
        };
        if tried.iter().any(|seen| readout_names_equal(seen, name)) {
            continue;
        }
        tried.push(name);
        let group: Vec<CalibrationFrame> = frames
            .iter()
            .filter(|candidate| {
                named_mode(&candidate.readout_mode_name)
                    .is_none_or(|other| readout_names_equal(other, name))
            })
            .cloned()
            .collect();
        let subset = master_subset_of_one_mode(kind, &group, rules);
        if subset.kept.len() >= MIN_MASTER_FRAMES {
            return subset;
        }
    }
    master_subset_of_one_mode(kind, frames, rules)
}
fn master_subset_of_one_mode(
    kind: CalibrationKind,
    frames: &[CalibrationFrame],
    rules: &Rules,
) -> Cohort {
    let signatures: Vec<_> = frames.iter().map(frame_signature).collect();
    let seiza_signatures: Vec<_> = signatures
        .iter()
        .map(|signature| signature.seiza.clone())
        .collect();
    let role = if kind == CalibrationKind::Flat {
        crate::FrameRole::Flat
    } else {
        crate::FrameRole::Other
    };
    let tolerances = rules.coherence_tolerances();
    let cluster =
        |minimum| crate::coherent_subset_indices(&seiza_signatures, role, minimum, &tolerances);
    // Seiza takes the nearest anchor whose cluster reaches the minimum. With
    // a minimum of two, a warm pair of darks shot near the lights beat eighty
    // at the set point a few weeks away, and the pair's read noise went into
    // every light as banding. Bias had the same flaw: five frames at -4 °C
    // beat thirty at the set point. So by default a dark or bias master looks
    // for a complete set first and settles for a short one only when no
    // cluster has one.
    let clustered = if matches!(kind, CalibrationKind::Dark | CalibrationKind::Bias)
        && rules.dark_choice == DarkChoice::Complete
    {
        let complete = cluster(rules.complete_dark_frames);
        if complete.len() >= rules.complete_dark_frames {
            complete
        } else {
            cluster(MIN_MASTER_FRAMES)
        }
    } else {
        cluster(MIN_MASTER_FRAMES)
    };

    // The integrator takes its first frame as the reference and measures every
    // later one against it, so anchor on the same frame it will. Candidates
    // arrive nearest-first, which makes that the frame closest to the lights.
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    let mut anchor: Option<usize> = None;
    for index in clustered {
        match anchor {
            None => {
                anchor = Some(index);
                kept.push(frames[index].clone());
            }
            Some(anchor)
                if integrates_with(kind, &signatures[anchor], &signatures[index], &tolerances) =>
            {
                kept.push(frames[index].clone());
            }
            Some(_) => dropped.push(frames[index].clone()),
        }
    }
    Cohort { kept, dropped }
}
pub(super) fn integrates_with(
    kind: CalibrationKind,
    reference: &Signature,
    candidate: &Signature,
    tolerances: &crate::MatchTolerances,
) -> bool {
    if !signature_sensor_consistent(reference, candidate) {
        return false;
    }
    // Only a flat records an optical path; a bias or a dark does not care
    // which filter was in the way.
    kind != CalibrationKind::Flat
        || crate::optics_consistent(&reference.seiza, &candidate.seiza, tolerances)
}
pub(super) fn sort_candidates(frames: &mut [CalibrationFrame], reference_at: Option<i64>) {
    frames.sort_by_key(|frame| match (reference_at, frame.captured_at) {
        (Some(reference), Some(captured)) => reference.abs_diff(captured),
        (Some(_), None) => u64::MAX,
        _ => 0,
    });
}
pub fn validity_admits(candidate: &CalibrationFrame, light_timestamp: Option<i64>) -> bool {
    let (Some(direction), Some(light), Some(frame)) = (
        candidate.valid_direction,
        light_timestamp,
        candidate.captured_at,
    ) else {
        return true;
    };
    match direction {
        ValidDirection::Forward => light >= frame,
        ValidDirection::Backward => light <= frame,
    }
}
