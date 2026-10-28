// The C ABI's JSON calls: built with the `json` feature.
#![cfg(feature = "json")]

// Compiled as an external crate, like a PSF Guard consumer. Private-module
// unit tests cannot catch an otherwise public result model being unnameable.
use seiza_calibration::library::{
    AlikeRangeNote, ChoiceNote, EventNote, FlatBasis, FlatCorrection, FlatFamily, GroupNote,
    json::{Request, evaluate},
    rules::{ChoiceOption, Effect, FieldKind, FieldSpec, Group, RuleNote, RuleSource},
};

#[test]
fn native_hosts_can_name_and_match_result_models() {
    let result = evaluate(Request::default()).unwrap();
    for field in &result.settings_schema {
        let _: &FieldSpec = field;
        let _: Effect = field.effect;
        let _: Group = field.group;
        match &field.kind {
            FieldKind::Choice { options } => {
                let _: &[ChoiceOption] = options;
            }
            FieldKind::Toggle
            | FieldKind::Number { .. }
            | FieldKind::Count { .. }
            | FieldKind::Check { .. }
            | FieldKind::Limits { .. } => {}
        }
    }
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/library/psf-guard.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let result = evaluate(serde_json::from_value(case["request"].clone()).unwrap()).unwrap();
        for selection in &result.selections {
            let _: Option<&FlatFamily> = selection.flat_family.as_ref();
            let _: &[ChoiceNote] = &selection.flat_choices;
            for note in &selection.rule_notes {
                let _: &RuleNote = note;
                let _: RuleSource = note.source;
            }
            match &selection.flat_basis {
                FlatBasis::Nearest => {}
                FlatBasis::Group { group } => {
                    let _: &GroupNote = group;
                }
                FlatBasis::SameSide { event, .. }
                | FlatBasis::OtherAngle { event, .. }
                | FlatBasis::Corrected { event, .. }
                | FlatBasis::NoSameSide { event, .. }
                | FlatBasis::Undetermined { event, .. } => {
                    let _: &EventNote = event;
                }
            }
            let _: Option<&FlatCorrection> = selection.flat_basis.correction();
            let _: Option<&AlikeRangeNote> = selection
                .flat_family
                .as_ref()
                .and_then(|f| f.alike_range.as_ref());
        }
    }
}
