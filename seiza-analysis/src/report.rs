use crate::{AnalysisReport, Error};

/// Render a portable, network-free HTML report. Embedded data cannot terminate
/// its script element; filenames and metadata are inserted as text, never HTML.
pub fn render_html(report: &AnalysisReport) -> Result<String, Error> {
    if !matches!(report.schema_version, 1 | 2) {
        return Err(Error::Invalid("unsupported analysis report schema".into()));
    }
    let json = serde_json::to_string(report)?
        .replace('&', "\\u0026")
        .replace('<', "\\u003c")
        .replace('>', "\\u003e");
    Ok(include_str!("report.html").replace("__REPORT_DATA__", &json))
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
