// Extracted from theatrus/psf-guard at 79f5e31d53b8371e8fcd0b22a9b69692412cd8b4.
// Copyright the PSF Guard contributors; Apache-2.0.
use super::*;
pub(super) fn kept_out_reason(
    stray_light: Option<String>,
    grade: Option<&str>,
    reason: Option<String>,
    choice: Option<QualityOverride>,
) -> Option<String> {
    match choice {
        Some(QualityOverride::Use) => None,
        Some(QualityOverride::Exclude) => Some("by hand".into()),
        None => match grade {
            Some("poor") => Some(reason.unwrap_or_else(|| "graded poor".into())),
            Some(_) => None,
            None => stray_light,
        },
    }
}
pub(super) fn judge_darks(
    measured: &[Measured],
    judged: &mut HashMap<String, Judged>,
    book: &rules::RuleBook,
) {
    let mut buckets: HashMap<(CalibrationKind, &str), Vec<(&Measured, &DarkMeasurement)>> =
        HashMap::new();
    for entry in measured {
        if let StoredMeasurement::Dark(dark) = &entry.measurement
            && entry.frame.kind != CalibrationKind::Flat
        {
            buckets
                .entry((entry.frame.kind, entry.frame.rig_uuid.as_str()))
                .or_default()
                .push((entry, dark));
        }
    }
    for ((kind, rig), frames) in buckets {
        // Each rig's frames are graded by its own limits.
        let rules = book.for_rig(rig);
        let (dark_screening, bias_screening): (&DarkScreening, &BiasScreening) =
            (&rules.dark, &rules.bias);
        for (entry, _) in &frames {
            let peers: Vec<&(&Measured, &DarkMeasurement)> = frames
                .iter()
                .filter(|(other, _)| {
                    signature_sensor_matches(&entry.signature, &other.signature)
                        && exposure_matches(entry.frame.exposure_s, other.frame.exposure_s)
                        && temperature_matches(
                            entry.frame.camera_temp,
                            other.frame.camera_temp,
                            &rules,
                        )
                })
                .collect();
            let Some(index) = peers
                .iter()
                .position(|(other, _)| other.frame.frame_uuid == entry.frame.frame_uuid)
            else {
                continue;
            };
            let set: Vec<DarkMeasurement> = peers.iter().map(|(_, dark)| (*dark).clone()).collect();
            let verdicts = if kind == CalibrationKind::Bias {
                crate::screen_bias(&set, bias_screening)
            } else {
                crate::screen_darks(&set, dark_screening)
            };
            let Some(verdict) = verdicts.into_iter().nth(index) else {
                continue;
            };
            let stray_light = (kind != CalibrationKind::Bias)
                .then(|| stray_light_text(&verdict, &set, index))
                .flatten();
            judged.insert(
                entry.frame.frame_uuid.clone(),
                judge(verdict, stray_light, kind),
            );
        }
    }
}
fn stray_light_text(
    verdict: &FrameVerdict,
    set: &[DarkMeasurement],
    index: usize,
) -> Option<String> {
    let finding = verdict
        .findings
        .iter()
        .find(|finding| finding.code == FindingCode::LevelExcess && finding.grade == Grade::Poor)?;
    let quietest = set.iter().min_by(|a, b| a.level.total_cmp(&b.level))?;
    let frame = set.get(index)?;
    Some(format!(
        "stray light: level {:.1} sits {:.1} above the quietest dark, {:.1} times its pixel \
         spread of {:.1}",
        frame.level,
        frame.level - quietest.level,
        finding.value,
        quietest.sigma
    ))
}
pub(super) fn judge_flats(
    measured: &[Measured],
    nights: &Nights,
    judged: &mut HashMap<String, Judged>,
    book: &rules::RuleBook,
) {
    for set in flat_sessions(measured, nights, book) {
        let measurements: Vec<FlatMeasurement> = set
            .iter()
            .filter_map(|entry| match &entry.measurement {
                StoredMeasurement::Flat(flat) => Some(flat.measurement.clone()),
                StoredMeasurement::Dark(_) => None,
            })
            .collect();
        // A session is one rig's, so its flats are graded by that rig's limits.
        let Some(rig) = set.first().map(|entry| entry.frame.rig_uuid.as_str()) else {
            continue;
        };
        let rules = book.for_rig(rig);
        let report = match crate::screen_flats(&measurements, &rules.flat) {
            Ok(report) => report,
            Err(error) => {
                let _ = error;
                continue;
            }
        };
        for (entry, verdict) in set.iter().zip(report.frames) {
            judged.insert(
                entry.frame.frame_uuid.clone(),
                judge(verdict, None, CalibrationKind::Flat),
            );
        }
    }
}
fn flat_sessions<'a>(
    measured: &'a [Measured],
    nights: &Nights,
    book: &rules::RuleBook,
) -> Vec<Vec<&'a Measured>> {
    let mut sessions: Vec<(FlatSessionKey, Vec<&Measured>)> = Vec::new();
    for entry in measured {
        let StoredMeasurement::Flat(flat) = &entry.measurement else {
            continue;
        };
        if entry.frame.kind != CalibrationKind::Flat {
            continue;
        }
        let key = FlatSessionKey {
            rig: entry.frame.rig_uuid.clone(),
            night: entry
                .frame
                .captured_at
                .map(|at| nights.night_of_frame(&entry.frame.rig_uuid, &entry.frame.frame_uuid, at))
                .unwrap_or_else(|| "unknown".into()),
            filter: entry
                .frame
                .filter
                .as_deref()
                .map(|filter| filter.trim().to_ascii_lowercase()),
            width: flat.width,
            height: flat.height,
        };
        let tolerances = book.for_rig(&entry.frame.rig_uuid).tolerances();
        let joins = |(session_key, frames): &&mut (FlatSessionKey, Vec<&Measured>)| {
            *session_key == key
                && frames.first().is_some_and(|first| {
                    signature_sensor_consistent(&first.signature, &entry.signature)
                        && crate::optics_consistent(
                            &first.signature.seiza,
                            &entry.signature.seiza,
                            &tolerances,
                        )
                })
        };
        match sessions.iter_mut().find(|session| joins(session)) {
            Some((_, frames)) => frames.push(entry),
            None => sessions.push((key, vec![entry])),
        }
    }
    sessions.into_iter().map(|(_, frames)| frames).collect()
}
fn judge(verdict: FrameVerdict, stray_light: Option<String>, kind: CalibrationKind) -> Judged {
    // The worst finding speaks for the frame; the first of equal grade.
    let main = verdict
        .findings
        .iter()
        .find(|finding| finding.grade == verdict.grade)
        .filter(|_| verdict.grade != Grade::Ok);
    let reason = main.map(|finding| match (&stray_light, finding.code) {
        (Some(text), FindingCode::LevelExcess) if finding.grade == Grade::Poor => text.clone(),
        _ => describe_finding(finding, kind),
    });
    Judged {
        grade: verdict.grade,
        code: main.map(|finding| finding.code),
        reason,
        findings: verdict.findings,
        stray_light,
    }
}
pub fn describe_finding(finding: &Finding, kind: CalibrationKind) -> String {
    let value = finding.value;
    let peers = if kind == CalibrationKind::Flat {
        "its session"
    } else {
        "its peers"
    };
    match finding.code {
        FindingCode::Clipped => format!("{:.2}% of samples clipped", value * 100.0),
        FindingCode::Bright => format!("brightest channel at {:.0}% of full scale", value * 100.0),
        FindingCode::Dim => format!("dimmest channel at {:.1}% of full scale", value * 100.0),
        FindingCode::Illumination => {
            format!(
                "light falls {:.1}% off the pattern of {peers}",
                value * 100.0
            )
        }
        FindingCode::Features => format!("{value:.0} clear spots"),
        FindingCode::ChangingFeatures => format!("{value:.0} spots that change across {peers}"),
        FindingCode::CarriedFeatures => format!("{value:.0} changing spots in most frames"),
        FindingCode::Level => format!("level {value:.1} times the noise from {peers}"),
        FindingCode::LevelExcess => {
            format!("level {value:.2} times its spread above the quietest dark")
        }
        FindingCode::Noise => format!("noise {value:.2} times that of {peers}"),
        FindingCode::RowBanding => format!("row banding {value:.2} of the noise"),
        FindingCode::ColumnBanding => format!("column banding {value:.2} of the noise"),
        FindingCode::HotPixels => format!("{value:.1} times the hot pixels of {peers}"),
    }
}
