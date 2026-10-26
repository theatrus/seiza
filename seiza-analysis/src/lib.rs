//! Project diagnostics without producing a stack. Measurements use native
//! detector samples, a common footprint, and balanced odd/even differences.
//! This is an equal-weight diagnostic, not a prediction of production rejection.

mod report;
mod sampling;
mod trend;
pub use report::render_html;

use seiza_stacking::{FitsFrame, GpuOptions, Registrar, RegistrationOptions, RegistrationResult};
use seiza_stars::{HocusFocusDetectionResult, HocusFocusParams, PSFType};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Errors affecting the request as a whole; individual unreadable frames stay in the report.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

/// Analysis settings shared by all hosts. Sample counts round down to whole 256-sample patches.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnalysisOptions {
    pub samples: usize,
    /// Case-insensitive header filter selection; missing filters are never guessed.
    pub filter: Option<String>,
    /// First N paths after sorting and deduplication, before header filtering.
    pub max_frames: Option<usize>,
}
impl Default for AnalysisOptions {
    fn default() -> Self {
        Self {
            samples: 262_144,
            filter: None,
            max_frames: None,
        }
    }
}

/// Schema 1. Values are in each filter reference's stored physical sample units.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnalysisReport {
    pub schema_version: u32,
    pub algorithm_version: u32,
    pub options: AnalysisOptions,
    pub frames: Vec<FrameReport>,
    pub groups: Vec<FilterReport>,
    pub warnings: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameReport {
    pub path: PathBuf,
    pub filter: Option<String>,
    pub capture_time: Option<String>,
    pub exposure_seconds: Option<f64>,
    /// accepted, rejected, or excluded (by the requested filter).
    pub status: String,
    pub error: Option<String>,
    pub star_count: Option<usize>,
    pub median_hfr: Option<f64>,
    pub median_fwhm: Option<f64>,
    pub median_eccentricity: Option<f64>,
    pub background: Option<f64>,
    pub noise: Option<f64>,
    pub matched_stars: Option<usize>,
    pub registration_rms_pixels: Option<f64>,
    pub gain: Option<f64>,
    pub offset: Option<f64>,
    pub normalization: Option<String>,
    pub warnings: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FilterReport {
    pub filter: Option<String>,
    pub reference: PathBuf,
    pub candidate_patches: usize,
    pub common_patches: usize,
    pub accepted_frames: usize,
    pub integration_seconds: f64,
    pub unknown_exposures: usize,
    pub curve: Vec<DepthPoint>,
    pub noise_trend: Option<NoiseTrend>,
    pub warnings: Vec<String>,
}
/// Descriptive log-log fit on geometric even-depth checkpoints in the last
/// fourfold window. Cumulative points are correlated: r_squared is not confidence.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NoiseTrend {
    pub exponent: f64,
    pub r_squared: f64,
    pub checkpoints: usize,
    pub first_frame_count: usize,
    pub last_frame_count: usize,
    /// Conditional factor if this trend persists to twice the frame count.
    /// Suppressed unless -1 <= exponent < 0 and r_squared >= 0.8.
    pub conditional_double_depth_noise_factor: Option<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DepthPoint {
    /// Index into AnalysisReport.frames.
    pub frame_index: usize,
    pub frame_count: usize,
    pub integration_seconds: f64,
    pub unknown_exposures: usize,
    /// Only even depths have balanced split-noise measurements.
    pub noise: Option<f64>,
    /// 1/sqrt(N), anchored at the measured N=2 noise. A comparison, not a forecast.
    pub ideal_noise: Option<f64>,
}

struct Group {
    report: FilterReport,
    registrar: Registrar,
    measured: HocusFocusDetectionResult,
    width: usize,
    height: usize,
    exposure: Option<f64>,
    patches: Vec<(usize, usize)>,
    reference_samples: Vec<f32>,
    common: Vec<bool>,
    spool: std::fs::File,
    indices: Vec<usize>,
}

/// Analyze files, directories (nonrecursive), or UTF-8 .txt lists. Relative list
/// entries resolve against their list. Paths are sorted and deduplicated by identity.
/// Source images are never written. Compact samples spill to temporary disk and
/// are deleted on return, including errors; image memory does not grow with frame count.
pub fn analyze(inputs: &[PathBuf], options: &AnalysisOptions) -> Result<AnalysisReport, Error> {
    if !(8192..=2_097_152).contains(&options.samples) {
        return Err(Error::Invalid(
            "samples must be between 8192 and 2097152".into(),
        ));
    }
    if options.max_frames == Some(0) || options.filter.as_ref().is_some_and(|s| s.trim().is_empty())
    {
        return Err(Error::Invalid(
            "max_frames must be positive and filter must not be blank".into(),
        ));
    }
    let mut paths = Vec::new();
    for input in inputs {
        discover(input, &mut paths)?;
    }
    paths.sort();
    let mut seen = HashSet::new();
    paths.retain(|p| seen.insert(seiza_stacking::path_identity(p)));
    if let Some(n) = options.max_frames {
        paths.truncate(n);
    }
    if paths.is_empty() {
        return Err(Error::Invalid("no FITS/XISF inputs found".into()));
    }
    let mut result = AnalysisReport { schema_version: 1, algorithm_version: 1,
        options: options.clone(), frames: Vec::new(), groups: Vec::new(),
        warnings: vec!["Equal-weight nearest-detector diagnostic. Shared scene structure and shared systematic errors cancel in split noise; seeing, registration and illumination changes may remain. This does not reproduce production weighting or rejection.".into(),
        "Curves follow sorted path order, not capture-time or quality order. Ideal noise assumes independent equal-quality frames; it is not an improvement forecast.".into()] };
    let mut groups: BTreeMap<Option<String>, Group> = BTreeMap::new();
    for path in paths {
        let mut row = FrameReport {
            path: path.clone(),
            filter: None,
            capture_time: None,
            exposure_seconds: None,
            status: "rejected".into(),
            error: None,
            star_count: None,
            median_hfr: None,
            median_fwhm: None,
            median_eccentricity: None,
            background: None,
            noise: None,
            matched_stars: None,
            registration_rms_pixels: None,
            gain: None,
            offset: None,
            normalization: None,
            warnings: Vec::new(),
        };
        let index = result.frames.len();
        match prepare(&path, options, &mut row) {
            Ok(Some((frame, measured))) => {
                let key = row.filter.as_ref().map(|s| s.to_lowercase());
                if let Some(group) = groups.get_mut(&key) {
                    match sample_frame(group, &frame, &measured, &mut row) {
                        Ok(values) => {
                            append(group, &values, index)?;
                            row.status = "accepted".into();
                        }
                        Err(error) => row.error = Some(error),
                    }
                } else {
                    match new_group(&frame, measured, &row, options) {
                        Ok(mut group) => {
                            let values = group.reference_samples.clone();
                            append(&mut group, &values, index)?;
                            row.status = "accepted".into();
                            row.gain = Some(1.0);
                            row.offset = Some(0.0);
                            row.normalization = Some("reference".into());
                            row.warnings.push("Reference frame: identity placement is not an independent registration measurement.".into());
                            groups.insert(key, group);
                        }
                        Err(error) => row.error = Some(error),
                    }
                }
            }
            Ok(None) => {}
            Err(error) => row.error = Some(error),
        }
        result.frames.push(row);
    }
    for (_, group) in groups {
        annotate_health(&group.indices, &mut result.frames);
        result.groups.push(finish(group, &result.frames)?);
    }
    if result.groups.is_empty() {
        result
            .warnings
            .push("No usable reference frames; inspect frame errors and filter selection.".into());
    }
    Ok(result)
}

fn discover(input: &Path, paths: &mut Vec<PathBuf>) -> Result<(), Error> {
    if input.is_dir() {
        for entry in std::fs::read_dir(input)? {
            let path = entry?.path();
            if path.is_file() && astronomy(&path) {
                paths.push(path);
            }
        }
    } else if input
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("txt"))
    {
        for line in std::fs::read_to_string(input)?.lines() {
            let line = line.trim();
            if !line.is_empty() && !line.starts_with('#') {
                let path = input.parent().unwrap_or(Path::new(".")).join(line);
                if !astronomy(&path) {
                    return Err(Error::Invalid(format!(
                        "list entry is not FITS/XISF: {}",
                        path.display()
                    )));
                }
                paths.push(path);
            }
        }
    } else {
        paths.push(input.to_path_buf());
    }
    Ok(())
}
fn astronomy(path: &Path) -> bool {
    seiza_image_io::Format::from_path(path).is_some_and(|f| f.is_astronomy())
}
fn prepare(
    path: &Path,
    options: &AnalysisOptions,
    row: &mut FrameReport,
) -> Result<Option<(FitsFrame, HocusFocusDetectionResult)>, String> {
    let raw = seiza_image_io::read_astronomy(path).map_err(|e| e.to_string())?;
    row.filter = raw
        .image
        .header_str("FILTER")
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    row.capture_time = seiza_image_io::capture_time_from_headers(&raw.image.headers);
    if options.filter.as_ref().is_some_and(|f| {
        !row.filter
            .as_ref()
            .is_some_and(|v| v.eq_ignore_ascii_case(f))
    }) {
        row.status = "excluded".into();
        return Ok(None);
    }
    if raw.image.planes != 1 || raw.image.bayer_pattern_in_storage_order().is_some() {
        return Err("analysis currently requires monochrome, non-Bayer frames".into());
    }
    if raw.image.width < 40 || raw.image.height < 40 {
        return Err("image too small for analysis (minimum 40 pixels per axis)".into());
    }
    let luma = seiza_image_io::luma_u16(&raw.image);
    let mut measured = seiza_stars::detect_stars_hocus_focus(
        &luma,
        raw.image.width,
        raw.image.height,
        &HocusFocusParams {
            psf_type: PSFType::Gaussian,
            ..Default::default()
        },
    );
    let luma = luma.into_owned();
    let saturation = raw.image.header_f64("SATURATE").filter(|v| v.is_finite());
    let mut frame =
        FitsFrame::from_fits(raw.image, Some(path.to_path_buf())).map_err(|e| e.to_string())?;
    // The measurement detector works on u16. Its flux must be converted back
    // to physical units: per-frame min/max scaling otherwise erases transparency.
    let endpoints = luma
        .iter()
        .zip(&frame.image.data)
        .filter(|(_, v)| v.is_finite());
    let low = endpoints.clone().min_by_key(|(v, _)| **v);
    let high = endpoints.max_by_key(|(v, _)| **v);
    if let (Some((lo, physical_lo)), Some((hi, physical_hi))) = (low, high) {
        let scale =
            (f64::from(*physical_hi) - f64::from(*physical_lo)) / f64::from(hi.saturating_sub(*lo));
        for star in &mut measured.stars {
            star.flux *= scale;
        }
    }
    row.exposure_seconds = frame.exposure_seconds;
    row.star_count = Some(measured.stars.len());
    row.median_hfr = sampling::median(measured.stars.iter().map(|s| s.hfr));
    row.median_fwhm = sampling::median(measured.stars.iter().map(|s| s.fwhm));
    row.median_eccentricity = sampling::median(
        measured
            .stars
            .iter()
            .filter_map(|s| s.psf_model.as_ref().map(|m| m.eccentricity)),
    );
    row.noise = seiza_stacking::frame_noise(&frame.image)
        .and_then(|v| v.first().copied())
        .map(f64::from);
    row.background = sampling::median(frame.image.data.iter().step_by(16).map(|v| f64::from(*v)));
    if row.filter.is_none() {
        row.warnings
            .push("Missing FILTER: unknown frames may represent different passbands.".into());
    }
    if row.exposure_seconds.is_none() {
        row.warnings
            .push("Missing positive exposure duration.".into());
    }
    if measured.stars.len() < 10 {
        row.warnings
            .push("Few measured stars; shape and photometry may be unreliable.".into());
    }
    if let Some(limit) = saturation {
        for v in &mut frame.image.data {
            if f64::from(*v) >= limit {
                *v = f32::NAN;
            }
        }
    } else {
        row.warnings
            .push("No SATURATE header; detector-sample saturation masking is unavailable.".into());
    }
    Ok(Some((frame, measured)))
}

fn new_group(
    frame: &FitsFrame,
    measured: HocusFocusDetectionResult,
    row: &FrameReport,
    options: &AnalysisOptions,
) -> Result<Group, String> {
    let registrar = Registrar::new_with_gpu(
        &frame.image,
        RegistrationOptions::default(),
        &GpuOptions::cpu(),
    )
    .map_err(|e| e.to_string())?;
    let patches = sampling::patches(frame.image.width, frame.image.height, options.samples);
    if patches.is_empty() {
        return Err("image too small for sampling patches".into());
    }
    let (reference_samples, common) = sampling::gather(&frame.image, &patches, None);
    let report = FilterReport {
        filter: row.filter.clone(),
        reference: row.path.clone(),
        candidate_patches: patches.len(),
        common_patches: 0,
        accepted_frames: 0,
        integration_seconds: 0.0,
        unknown_exposures: 0,
        curve: Vec::new(),
        noise_trend: None,
        warnings: Vec::new(),
    };
    Ok(Group {
        report,
        registrar,
        measured,
        width: frame.image.width,
        height: frame.image.height,
        exposure: frame.exposure_seconds,
        patches,
        reference_samples,
        common,
        spool: tempfile::tempfile().map_err(|e| e.to_string())?,
        indices: Vec::new(),
    })
}
fn sample_frame(
    group: &mut Group,
    frame: &FitsFrame,
    measured: &HocusFocusDetectionResult,
    row: &mut FrameReport,
) -> Result<Vec<f32>, String> {
    if frame.image.width != group.width || frame.image.height != group.height {
        return Err("dimensions differ from this filter's reference".into());
    }
    let registration = group
        .registrar
        .register(&frame.image)
        .map_err(|e| e.to_string())?;
    row.matched_stars = Some(registration.matched_stars);
    row.registration_rms_pixels = Some(registration.rms_error_pixels);
    let (mut values, valid) = sampling::gather(&frame.image, &group.patches, Some(&registration));
    let ratios = stellar_ratios(&group.measured, measured, &registration);
    let gain = if ratios.len() >= 5 {
        row.normalization = Some("stellar_flux".into());
        sampling::median(ratios).unwrap()
    } else {
        row.warnings.push("Insufficient matched photometry: exposure-ratio (or unity) gain cannot measure transparency changes.".into());
        row.normalization = Some("exposure_fallback".into());
        match (group.exposure, frame.exposure_seconds) {
            (Some(a), Some(b)) => a / b,
            _ => 1.0,
        }
    };
    if !gain.is_finite() || gain <= 0.0 {
        return Err("unusable photometric gain".into());
    }
    let offset = sampling::median(
        values
            .iter()
            .zip(&group.reference_samples)
            .enumerate()
            .filter(|(i, _)| valid[i / sampling::PIXELS] && group.common[i / sampling::PIXELS])
            .map(|(_, (v, r))| f64::from(*r) - gain * f64::from(*v)),
    )
    .ok_or("no valid sample overlap")?;
    for v in &mut values {
        *v = (f64::from(*v) * gain + offset) as f32;
    }
    for (i, common) in group.common.iter_mut().enumerate() {
        *common &= valid[i]
            && values[i * sampling::PIXELS..(i + 1) * sampling::PIXELS]
                .iter()
                .all(|v| v.is_finite());
    }
    row.gain = Some(gain);
    row.offset = Some(offset);
    Ok(values)
}
fn stellar_ratios(
    reference: &HocusFocusDetectionResult,
    source: &HocusFocusDetectionResult,
    registration: &RegistrationResult,
) -> Vec<f64> {
    let mut used = HashSet::new();
    let mut ratios = Vec::new();
    for star in source.stars.iter().filter(|s| !s.saturated && s.flux > 0.0) {
        let (x, y) = registration
            .transform
            .apply(star.position.0, star.position.1);
        let nearest = reference
            .stars
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.saturated && s.flux > 0.0)
            .map(|(i, s)| (i, (s.position.0 - x).hypot(s.position.1 - y)))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((i, d)) = nearest.filter(|(_, d)| *d < 2.0) {
            let _ = d;
            if used.insert(i) {
                ratios.push(reference.stars[i].flux / star.flux);
            }
        }
    }
    ratios
}
fn append(group: &mut Group, values: &[f32], index: usize) -> Result<(), Error> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    group.spool.write_all(&bytes)?;
    group.indices.push(index);
    Ok(())
}
fn finish(mut group: Group, frames: &[FrameReport]) -> Result<FilterReport, Error> {
    group.report.common_patches = group.common.iter().filter(|v| **v).count();
    group.report.accepted_frames = group.indices.len();
    let enough = group.report.common_patches >= 32;
    if !enough {
        group
            .report
            .warnings
            .push("Fewer than 32 common patches: noise curve suppressed.".into());
    }
    if group.indices.len() < 2 {
        group
            .report
            .warnings
            .push("At least two accepted frames are required for split noise.".into());
    }
    if group.report.filter.is_none() {
        group
            .report
            .warnings
            .push("Unknown passband group: do not assume these frames share a filter.".into());
    }
    group.spool.seek(SeekFrom::Start(0))?;
    let mut differences = vec![0.0f64; group.reference_samples.len()];
    let mut bytes = vec![0u8; differences.len() * 4];
    let mut anchor = None;
    for (i, index) in group.indices.into_iter().enumerate() {
        let n = i + 1;
        match frames[index].exposure_seconds {
            Some(t) => group.report.integration_seconds += t,
            None => group.report.unknown_exposures += 1,
        }
        group.spool.read_exact(&mut bytes)?;
        for (sum, bytes) in differences.iter_mut().zip(bytes.as_chunks::<4>().0) {
            let v = f64::from(f32::from_le_bytes(*bytes));
            *sum += if n % 2 == 1 { v } else { -v };
        }
        let noise = if enough && n % 2 == 0 {
            sampling::median(
                differences
                    .as_chunks::<{ sampling::PIXELS }>()
                    .0
                    .iter()
                    .zip(&group.common)
                    .filter(|(_, valid)| **valid)
                    .map(|(tile, _)| sampling::sigma(tile) / n as f64),
            )
        } else {
            None
        };
        if n == 2 {
            anchor = noise;
        }
        group.report.curve.push(DepthPoint {
            frame_index: index,
            frame_count: n,
            integration_seconds: group.report.integration_seconds,
            unknown_exposures: group.report.unknown_exposures,
            noise,
            ideal_noise: anchor.map(|v| v * (2.0 / n as f64).sqrt()),
        });
    }
    group.report.noise_trend = trend::fit(&group.report.curve);
    Ok(group.report)
}

fn annotate_health(indices: &[usize], frames: &mut [FrameReport]) {
    if indices.len() < 3 {
        return;
    }
    let hfr = sampling::median(indices.iter().filter_map(|i| frames[*i].median_hfr));
    let stars = sampling::median(
        indices
            .iter()
            .filter_map(|i| frames[*i].star_count.map(|v| v as f64)),
    );
    let noise = sampling::median(
        indices
            .iter()
            .filter_map(|i| Some(frames[*i].noise? * frames[*i].gain?)),
    );
    for &i in indices {
        let f = &mut frames[i];
        if hfr
            .zip(f.median_hfr)
            .is_some_and(|(m, v)| m > 0.0 && v > 1.5 * m)
        {
            f.warnings
                .push("HFR exceeds 1.5 times this filter group's median (advisory).".into());
        }
        if stars
            .zip(f.star_count)
            .is_some_and(|(m, v)| (v as f64) < 0.5 * m)
        {
            f.warnings.push(
                "Measured star count is below half this filter group's median (advisory).".into(),
            );
        }
        if noise
            .zip(f.noise.zip(f.gain))
            .is_some_and(|(m, (v, g))| m > 0.0 && v * g > 2.0 * m)
        {
            f.warnings.push(
                "Normalized frame noise exceeds twice this filter group's median (advisory)."
                    .into(),
            );
        }
    }
}
