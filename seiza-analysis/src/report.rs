use crate::{AnalysisReport, Error};

/// Render a portable, network-free HTML report. Embedded data cannot terminate
/// its script element; filenames and metadata are inserted as text, never HTML.
pub fn render_html(report: &AnalysisReport) -> Result<String, Error> {
    if !matches!(report.schema_version, 1..=3) {
        return Err(Error::Invalid("unsupported analysis report schema".into()));
    }
    let json = serde_json::to_string(report)?
        .replace('&', "\\u0026")
        .replace('<', "\\u003c")
        .replace('>', "\\u003e");
    Ok(include_str!("report.html")
        .replace("__REPORT_STYLE__", include_str!("report.css"))
        .replace("__CHART_SCRIPT__", include_str!("chart.js"))
        .replace("__REPORT_DATA__", &json))
}
/// Render a completed-stack report with bounded support maps and embedded JSON.
pub fn render_stack_html(report: &seiza_stacking::PostStackReport) -> Result<String, Error> {
    validate_stack(report)?;
    stack_document(report)
}

/// Select report detail without changing numerical stack results.
/// `none` returns immediately without reading inputs. `cheap` uses retained data.
/// `rich` performs a separate, uncalibrated source-analysis pass; its depth curve
/// is an equal-weight diagnostic, not a reconstruction of final integration.
/// Inputs are explicit so hosts can include unsuccessful admission attempts too.
pub fn prepare_stack_report(
    report: Option<seiza_stacking::PostStackReport>,
    level: &str,
    inputs: &[std::path::PathBuf],
) -> Result<Option<seiza_stacking::PostStackReport>, Error> {
    if !matches!(level, "none" | "cheap" | "rich") {
        return Err(Error::Invalid(
            "report level must be none, cheap or rich".into(),
        ));
    }
    if level == "none" {
        return Ok(None);
    }
    let mut report =
        report.ok_or_else(|| Error::Invalid("a completed stack report is required".into()))?;
    validate_stack(&report)?;
    if report.provenance.as_ref().is_some_and(|p| !p.is_object()) {
        return Err(Error::Invalid("report provenance must be an object".into()));
    }
    let analysis = if level == "rich" {
        if inputs.is_empty() {
            return Err(Error::Invalid(
                "rich reporting requires explicit source inputs".into(),
            ));
        }
        Some(crate::analyze(inputs, &crate::AnalysisOptions::default())?)
    } else {
        None
    };
    let provenance = report
        .provenance
        .get_or_insert_with(|| serde_json::json!({}));
    let provenance = provenance
        .as_object_mut()
        .ok_or_else(|| Error::Invalid("report provenance must be an object".into()))?;
    provenance.insert(
        "reporting".into(),
        serde_json::json!({"level":level,"analysis":analysis}),
    );
    Ok(Some(report))
}

/// Named independent stack. Labels are unique within a collection; order is preserved.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NamedStackReport {
    pub label: String,
    pub report: seiza_stacking::PostStackReport,
}

/// Version 1 collection of independent results, not a shared canvas or color composite.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StackReportCollection {
    pub schema_version: u32,
    pub title: String,
    pub stacks: Vec<NamedStackReport>,
}

/// Validate and render independent stacks using the same view as a single-stack report.
/// The first listed stack is selected. No source images are opened.
pub fn render_stack_collection_html(report: &StackReportCollection) -> Result<String, Error> {
    if report.schema_version != 1 || report.title.trim().is_empty() || report.stacks.is_empty() {
        return Err(Error::Invalid(
            "collection requires schema 1, a title and at least one stack".into(),
        ));
    }
    let mut labels = std::collections::HashSet::new();
    for stack in &report.stacks {
        if stack.label.trim().is_empty() || !labels.insert(stack.label.trim().to_lowercase()) {
            return Err(Error::Invalid(
                "stack labels must be nonempty and unique".into(),
            ));
        }
        validate_stack(&stack.report)?;
    }
    stack_document(report)
}

fn stack_document(report: &impl serde::Serialize) -> Result<String, Error> {
    let json = serde_json::to_string(report)?
        .replace('&', "\\u0026")
        .replace('<', "\\u003c")
        .replace('>', "\\u003e");
    Ok(include_str!("stack-report.html")
        .replace("__REPORT_STYLE__", include_str!("report.css"))
        .replace("__CHART_SCRIPT__", include_str!("chart.js"))
        .replace("__STACK_SCRIPT__", include_str!("stack-report.js"))
        .replace("__STACK_CHARTS__", include_str!("stack-charts.js"))
        .replace("__REPORT_DATA__", &json))
}

fn validate_stack(r: &seiza_stacking::PostStackReport) -> Result<(), Error> {
    if let Some(analysis) = r
        .provenance
        .as_ref()
        .and_then(|p| p.get("reporting"))
        .and_then(|p| p.get("analysis"))
        .filter(|a| !a.is_null())
    {
        let analysis: AnalysisReport = serde_json::from_value(analysis.clone())?;
        if !matches!(analysis.schema_version, 1..=3) {
            return Err(Error::Invalid("unsupported rich analysis schema".into()));
        }
    }
    let invalid =
        || Error::Invalid("invalid post-stack report dimensions, maps, counts or schema".into());
    let pixels = r
        .width
        .checked_mul(r.height)
        .filter(|n| *n > 0)
        .ok_or_else(invalid)?;
    if r.schema_version != 1
        || !matches!(r.stage.as_str(), "online" | "batch" | "drizzled")
        || !matches!(r.channels, 1 | 3)
        || r.support.len() != r.channels
    {
        return Err(invalid());
    }
    let expected_quantity = if r.stage == "drizzled" {
        "drop_area_times_frame_weight"
    } else {
        "accepted_observations"
    };
    if r.support_map.quantity != expected_quantity
        || r.rejection_map
            .as_ref()
            .is_some_and(|map| map.quantity != "rejected_observations")
        || (r.stage == "drizzled" && r.rejection_map.is_some())
    {
        return Err(invalid());
    }
    for map in std::iter::once(&r.support_map).chain(r.rejection_map.iter()) {
        if map.width != r.width.min(128)
            || map.height != r.height.min(128)
            || map.channels != r.channels
            || map.values.len() != map.width * map.height * map.channels
            || map.values.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(invalid());
        }
    }
    for s in &r.support {
        if s.total_pixels != pixels
            || s.supported_pixels > pixels
            || [s.support_minimum, s.support_maximum, s.support_mean]
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
            || s.support_minimum > s.support_mean
            || s.support_mean > s.support_maximum
        {
            return Err(invalid());
        }
    }
    if r.context.as_ref().is_some_and(|ctx| {
        ctx.accepted_frames != r.accepted_frames
            || ctx.rejected_frames != r.rejected_frames
            || (!ctx.frames.is_empty() && ctx.frames.len() != r.accepted_frames as usize)
            || ctx.frames.iter().any(|f| {
                f.exposure_seconds
                    .is_some_and(|v| !v.is_finite() || v <= 0.0)
            })
    }) || r.final_samples.as_ref().is_some_and(|samples| {
        samples.len() != r.accepted_frames as usize
            || samples
                .iter()
                .any(|s| s.integrated_samples > s.finite_samples)
    }) {
        return Err(invalid());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn metadata_cannot_escape_script() {
        let report = crate::AnalysisReport {
            timings: Default::default(),
            schema_version: 1,
            algorithm_version: 1,
            options: Default::default(),
            frames: vec![],
            groups: vec![],
            warnings: vec!["</script><img src=x onerror=alert(1)>".into()],
        };
        let html = super::render_html(&report).unwrap();
        assert!(!html.contains("<img src=x"));
        assert!(html.contains("\\u003c/script\\u003e"));
    }
    #[test]
    fn legacy_report_remains_renderable_without_inventing_new_outcomes() {
        let report: crate::AnalysisReport = serde_json::from_value(serde_json::json!({
            "schema_version":1,"algorithm_version":1,"options":{},"warnings":[],
            "frames":[{"path":"old.fits","status":"rejected","star_count":42,"warnings":[]}],
            "groups":[{"filter":"Ha","reference":"ref.fits","candidate_patches":32,
                "common_patches":32,"accepted_frames":1,"integration_seconds":60,
                "unknown_exposures":0,"curve":[],"noise_trend":null,"warnings":[]}]
        }))
        .unwrap();
        assert_eq!(report.frames[0].measurement_status, "rejected");
        assert!(report.frames[0].depth_status.is_empty());
        assert_eq!(report.groups[0].depth_frames, 1);
        let html = super::render_html(&report).unwrap();
        assert!(html.contains("Legacy depth-included group median"));
        assert!(html.contains("old.fits"));
    }
}

#[cfg(test)]
mod collection_tests {
    use super::*;
    fn fixture() -> StackReportCollection {
        let report = seiza_stacking::stack_report(
            &seiza_stacking::StackSnapshot {
                image: seiza_stacking::LinearImage {
                    width: 2,
                    height: 1,
                    channels: 1,
                    data: vec![1.0; 2],
                },
                variance: seiza_stacking::LinearImage {
                    width: 2,
                    height: 1,
                    channels: 1,
                    data: vec![0.0; 2],
                },
                coverage: vec![2, 1],
                rejected_samples: vec![0, 1],
                accepted_frames: 2,
                rejected_frames: 1,
            },
            None,
            None,
        )
        .unwrap();
        StackReportCollection {
            schema_version: 1,
            title: "SHO </script> & __STACK_SCRIPT__".into(),
            stacks: vec![NamedStackReport {
                label: "S".into(),
                report,
            }],
        }
    }
    #[test]
    fn report_levels_skip_source_io_and_preserve_final_statistics() {
        let report = fixture().stacks.remove(0).report;
        let missing = vec![std::path::PathBuf::from("missing.fits")];
        assert!(
            prepare_stack_report(None, "none", &missing)
                .unwrap()
                .is_none()
        );
        assert!(prepare_stack_report(None, "cheap", &missing).is_err());
        assert!(prepare_stack_report(Some(report.clone()), "invalid", &missing).is_err());
        let cheap = prepare_stack_report(Some(report.clone()), "cheap", &missing)
            .unwrap()
            .unwrap();
        let rich = prepare_stack_report(Some(report), "rich", &missing)
            .unwrap()
            .unwrap();
        assert!(cheap.provenance.as_ref().unwrap()["reporting"]["analysis"].is_null());
        assert_eq!(
            rich.provenance.as_ref().unwrap()["reporting"]["analysis"]["frames"][0]["measurement_status"],
            "failed"
        );
        assert_eq!(
            serde_json::to_value(&cheap.support).unwrap(),
            serde_json::to_value(&rich.support).unwrap()
        );
        let html = render_stack_html(&rich).unwrap();
        assert!(html.contains("function renderStackCharts"));
        assert!(!html.contains("__CHART_SCRIPT__"));
    }
    #[test]
    fn collection_embeds_exact_data_and_reuses_single_renderer() {
        let collection = fixture();
        let html = render_stack_collection_html(&collection).unwrap();
        let data = html
            .split("<script id=\"data\" type=\"application/json\">")
            .nth(1)
            .unwrap()
            .split("</script>")
            .next()
            .unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(data).unwrap(),
            serde_json::to_value(&collection).unwrap()
        );
        assert!(!html.contains("SHO </script>"));
        assert!(!html.contains("<iframe"));
        assert!(html.contains("select(0)"));
        assert!(
            render_stack_html(&collection.stacks[0].report)
                .unwrap()
                .contains("function renderReport")
        );
    }
    #[test]
    fn invalid_collections_and_nested_reports_are_rejected() {
        let original = fixture();
        let mut c = original.clone();
        c.stacks.clear();
        assert!(render_stack_collection_html(&c).is_err());
        let mut c = original.clone();
        c.schema_version = 2;
        assert!(render_stack_collection_html(&c).is_err());
        let mut c = original.clone();
        c.stacks.push(c.stacks[0].clone());
        c.stacks[1].label = " s ".into();
        assert!(render_stack_collection_html(&c).is_err());
        let mut c = original.clone();
        c.stacks[0].label = " ".into();
        assert!(render_stack_collection_html(&c).is_err());
        let mut c = original.clone();
        c.stacks[0].report.support_map.values.pop();
        assert!(render_stack_collection_html(&c).is_err());
        let mut c = original.clone();
        c.stacks[0].report.support[0].supported_pixels = 3;
        assert!(render_stack_collection_html(&c).is_err());
        let mut c = original.clone();
        c.stacks[0].report.support_map.values[0] = f64::NAN;
        assert!(render_stack_collection_html(&c).is_err());
        let mut c = original;
        c.stacks[0].report.final_samples = Some(vec![]);
        assert!(render_stack_collection_html(&c).is_err());
    }
}
