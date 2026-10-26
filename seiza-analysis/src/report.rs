use crate::{AnalysisReport, Error};

/// Render a portable, network-free HTML report. Embedded data cannot terminate
/// its script element; filenames and metadata are inserted as text, never HTML.
pub fn render_html(report: &AnalysisReport) -> Result<String, Error> {
    if report.schema_version != 1 {
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
}
