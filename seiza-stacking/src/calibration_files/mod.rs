//! File-backed graded calibration: discovery, measurement caches, native evidence,
//! master generation and correction publication. Hosts retain platform file
//! access grants and UI. Database hosts may continue to drive the independent
//! `seiza_calibration::library` stages using their own caches and pixels.
use crate::{CancelSignal, FileCalibrationSession, FitsFrame, FrameSourceRole};
use seiza_calibration::library::{self as policy, CalibrationKind as Kind, RecipeAction};
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(test)]
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};

#[cfg(feature = "schema")]
pub mod json;
mod planning;
mod responses;
pub use planning::{GroupingReview, Selection, Snapshot};
use planning::{Planner, RigSettings, Setting};
pub use responses::{
    ChangeOption, FlatCorrectionPlan, FlatWorkReview, GroupEdge, GroupOption, LightOptions,
};

type Result<T> = std::result::Result<T, String>;
#[derive(Default)]
struct Request {
    rules: Option<policy::rules::RuleBook>,
    operation: String,
    library: PathBuf,
    lights: Vec<PathBuf>,
    cache: PathBuf,
    settings: Vec<Setting>,
    rigs: Vec<RigSettings>,
    choices: Vec<Choice>,
    review_key: Option<String>,
    incomplete: String,
    flat_star_masking: bool,
    flat_stability: policy::FlatRetryPolicy,
    flat_association: policy::Association,
    dark_association: policy::Association,
    provided: Vec<PathBuf>,
    group_overrides: Vec<seiza_calibration::flat_response::groups::GroupOverrideRow>,
    light_choices: Vec<LightChoice>,
}
/// A quality override tied to the exact reviewed source.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Choice {
    /// Canonical source path.
    pub path: PathBuf,
    /// Metadata identity captured when this file was inspected.
    pub fingerprint: String,
    /// A user quality decision; physical gates still apply.
    pub choice: policy::QualityOverride,
}
/// An immutable path and metadata stamp used to reject stale inputs.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    /// Canonical source path.
    pub path: PathBuf,
    /// Metadata identity captured when this file was inspected.
    pub fingerprint: String,
}
/// A personal flat choice tied to one exact light source.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LightChoice {
    /// Canonical source path.
    pub path: PathBuf,
    /// Metadata identity captured when this file was inspected.
    pub fingerprint: String,
    /// The unchanged native command row, resolved by Seiza.
    pub row: seiza_calibration::flat_response::groups::LightChoiceRow,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
struct Cached {
    source: Source,
    frame: policy::CalibrationFrame,
    light: policy::Light,
    quality: policy::QualityRecord,
    processed: PathBuf,
    processed_source: Source,
    #[serde(default)]
    target: Option<String>,
}
/// The native quality decision shown for one calibration file.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Debug, Serialize)]
pub struct Frame {
    /// Canonical source path.
    pub path: PathBuf,
    /// Metadata identity captured when this file was inspected.
    pub fingerprint: String,
    /// Stable native identity.
    pub id: String,
    /// Calibration frame kind identified from its metadata.
    pub kind: Kind,
    /// Whether metadata identifies an existing external master.
    pub is_master: bool,
    /// Native grade, absent for unmeasured/external records.
    pub grade: Option<seiza_calibration::Grade>,
    /// Native grading or exclusion explanation.
    pub reason: Option<String>,
    /// Whether the cached measurement is current, stale, missing, failed or external.
    pub measurement_state: policy::MeasurementState,
    /// Input measurement failure, when present.
    pub error: Option<String>,
}
/// A review or prepared master plan preserving the original light order.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Debug, Serialize)]
pub struct PreparationReport {
    /// Identity of ordered sources, rules, choices and algorithm versions.
    pub review_key: String,
    /// Calibration file quality records in discovered order.
    pub frames: Vec<Frame>,
    /// Camera/telescope rig identities available for settings.
    pub rigs: Vec<Rig>,
    /// Native per-light selections, shared master sessions and assignments.
    pub snapshot: Snapshot,
    /// The measured native grouping used for this review.
    pub grouping: GroupingReview,
    /// Evidence rows, required corrections and manual edge choices.
    pub flat_work: FlatWorkReview,
    /// Applicable group/side commands for each original light.
    pub light_options: Vec<LightOptions>,
    /// Concrete skipped inputs and native preparation notes.
    pub warnings: Vec<String>,
    /// Incomplete calibration flags in original light order.
    pub missing: Vec<bool>,
    /// Physical-sample input paths, including staged declared-unit XISF sources.
    pub paths: Vec<PathBuf>,
    /// Original canonical light paths in the requested order.
    pub light_paths: Vec<PathBuf>,
    /// Published master paths and complete gates; empty during analysis.
    pub file_sessions: Vec<FileCalibrationSession>,
    /// Reviewed input, staged, response, evidence and master stamps.
    pub sources: Vec<Source>,
}

/// The camera/telescope rig named in review.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Debug, Serialize)]
pub struct Rig {
    /// Stable native rig identity.
    pub id: String,
    /// Capture metadata label.
    pub name: String,
}

fn digest(value: impl AsRef<[u8]>) -> String {
    Sha256::digest(value.as_ref())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn check(cancel: &CancelSignal) -> Result<()> {
    if cancel.is_cancelled() {
        Err("Calibration preparation was cancelled.".into())
    } else {
        Ok(())
    }
}
fn stamp(path: &Path) -> Result<Source> {
    let path = path.canonicalize().map_err(error)?;
    let m = std::fs::metadata(&path).map_err(error)?;
    if !m.is_file() {
        return Err(format!("{} is not a file", path.display()));
    }
    #[cfg(unix)]
    let fingerprint = {
        use std::os::unix::fs::MetadataExt;
        digest(format!(
            "{}:{}:{}:{}:{}:{}",
            path.display(),
            m.dev(),
            m.ino(),
            m.len(),
            m.mtime(),
            m.mtime_nsec()
        ))
    };
    #[cfg(not(unix))]
    let fingerprint = digest(format!(
        "{}:{}:{:?}:{:?}",
        path.display(),
        m.len(),
        m.modified().map_err(error)?,
        m.created().ok()
    ));
    Ok(Source { path, fingerprint })
}
fn verify(sources: &[Source]) -> Result<()> {
    for source in sources {
        if stamp(&source.path)?.fingerprint != source.fingerprint {
            return Err(format!(
                "{} changed. Analyze calibration again.",
                source.path.display()
            ));
        }
    }
    Ok(())
}
fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut file = tempfile::NamedTempFile::new_in(path.parent().ok_or("Missing cache directory")?)
        .map_err(error)?;
    serde_json::to_writer(file.as_file_mut(), value).map_err(error)?;
    file.as_file().sync_all().map_err(error)?;
    file.persist(path).map_err(error)?;
    Ok(())
}
fn publish_frame(path: &Path, frame: &FitsFrame) -> Result<()> {
    let temp = tempfile::Builder::new()
        .suffix(".fits")
        .tempfile_in(path.parent().unwrap())
        .map_err(error)?;
    let mut cards: Vec<_> = frame
        .headers
        .iter()
        .filter(|(k, _)| {
            [
                "IMAGETYP", "OBSTYPE", "FRAME", "SEIZAMST", "BIASSUB", "DARKSUB", "FLATNORM",
            ]
            .contains(&k.as_str())
        })
        .map(|(k, v)| seiza_fits::WriteHeaderCard::new(k.clone(), v.clone()))
        .collect();
    if let Some(mode) = named_mode(frame) {
        cards.push(seiza_fits::WriteHeaderCard::new(
            "READOUTM",
            seiza_fits::HeaderValue::String(mode),
        ));
    }
    crate::write_processed_image_fits_f32(temp.path(), &frame.image, &frame.headers, &cards)
        .map_err(error)?;
    temp.as_file().sync_all().map_err(error)?;
    temp.persist(path).map_err(error)?;
    Ok(())
}
fn image_ref(frame: &FitsFrame) -> Result<seiza_calibration::LinearImageRef<'_>> {
    seiza_calibration::LinearImageRef::new(
        &frame.image.data,
        frame.image.width,
        frame.image.height,
        frame.image.channels,
    )
    .map_err(error)
}
fn named_mode(frame: &FitsFrame) -> Option<String> {
    ["READOUTM", "READMODE", "READOUT", "READOUTMODE"]
        .iter()
        .find_map(|name| {
            frame.headers.iter().find_map(|(key, value)| match value {
                seiza_fits::HeaderValue::String(v)
                    if key.eq_ignore_ascii_case(name)
                        && v.parse::<i64>().is_err()
                        && !v.trim().is_empty() =>
                {
                    Some(v.trim().to_owned())
                }
                _ => None,
            })
        })
}
fn frame_target(frame: &FitsFrame) -> String {
    frame
        .headers
        .iter()
        .find_map(|(key, value)| match value {
            seiza_fits::HeaderValue::String(value) if key.eq_ignore_ascii_case("OBJECT") => {
                Some(value.trim().to_owned())
            }
            _ => None,
        })
        .unwrap_or_default()
}
fn inspect(path: &Path, cache: &Path, light: bool, cancel: &CancelSignal) -> Result<Cached> {
    check(cancel)?;
    let source = stamp(path)?;
    // The measurement generation participates in cache identity. Settings only
    // regrade cached measurements, never force another pixel measurement.
    let cache_key = digest(format!(
        "measurement-1-{light}-{}-{}-{}",
        source.fingerprint,
        seiza_calibration::FRAME_QUALITY_ALGORITHM_VERSION,
        seiza_calibration::FLAT_QUALITY_ALGORITHM_VERSION
    ));
    let memo = cache.join(format!("{cache_key}.json"));
    if let Ok(bytes) = std::fs::read(&memo)
        && let Ok(mut row) = serde_json::from_slice::<Cached>(&bytes)
        && row.source.fingerprint == source.fingerprint
        && verify(std::slice::from_ref(&row.processed_source)).is_ok()
    {
        if row.target.is_none() {
            let input = FitsFrame::open(&row.processed).map_err(error)?;
            row.target = Some(frame_target(&input));
            verify(&[row.source.clone(), row.processed_source.clone()])?;
            atomic_json(&memo, &row)?;
        }
        return Ok(row);
    }
    let mut input = FitsFrame::open(&source.path).map_err(error)?;
    let meta = input.metadata();
    let preprocessed = meta.calibration_state.bias_subtracted
        || meta.calibration_state.dark_subtracted
        || meta.calibration_state.flat_normalized;
    if light
        && (meta.is_master
            || preprocessed
            || !matches!(meta.role, FrameSourceRole::Light | FrameSourceRole::Unknown))
    {
        return Err(format!(
            "{} is not a raw light frame",
            source.path.display()
        ));
    }
    if !light && !meta.is_master && preprocessed {
        return Err(format!("{} is already calibrated", source.path.display()));
    }
    let kind = match meta.role {
        FrameSourceRole::Bias => Kind::Bias,
        FrameSourceRole::Dark => Kind::Dark,
        FrameSourceRole::DarkFlat => Kind::DarkFlat,
        FrameSourceRole::Flat => Kind::Flat,
        _ if light => Kind::Bias, // Not used as a calibration row.
        _ => {
            return Err(format!(
                "{} is not a calibration frame",
                source.path.display()
            ));
        }
    };
    let s = meta.signature;
    let mode = named_mode(&input);
    let rig_uuid = digest(serde_json::to_vec(&(&s.camera, &s.telescope)).map_err(error)?);
    // Declared unit XISF inputs are staged once in physical ADU. The original
    // source remains protected and verified; cached staging is not remeasured.
    let scaled = input.rescale_declared_unit_bounds(65535.0);
    let processed = if scaled {
        let target = cache.join(format!("source-{}.fits", source.fingerprint));
        publish_frame(&target, &input)?;
        target
    } else {
        source.path.clone()
    };
    let frame = policy::CalibrationFrame {
        frame_uuid: digest(source.path.to_string_lossy().as_bytes()),
        rig_uuid: rig_uuid.clone(),
        kind,
        source_path: source.path.clone(),
        source_fingerprint: source.fingerprint.clone(),
        source_verified: true,
        is_master: meta.is_master,
        captured_at: s.captured_at_unix,
        camera: s.camera.clone(),
        telescope: s.telescope.clone(),
        width: s.width,
        height: s.height,
        channels: s.channels,
        binning_x: s.binning_x,
        binning_y: s.binning_y,
        gain: s.gain,
        offset: s.offset,
        readout_mode: s.readout_mode,
        readout_mode_name: mode.clone(),
        bayer_pattern: s.bayer_pattern.clone(),
        exposure_s: s.exposure_seconds,
        camera_temp: s.camera_temp_c,
        filter: s.filter.clone(),
        focal_length_mm: s.focal_length_mm,
        rotation: s.rotation_deg,
        ..Default::default()
    };
    let target = policy::Light {
        path: source.path.clone(),
        rig_uuid,
        camera: s.camera,
        telescope: s.telescope,
        bayer_pattern: s.bayer_pattern,
        filter: s.filter,
        width: s.width,
        height: s.height,
        channels: s.channels,
        binning_x: s.binning_x,
        binning_y: s.binning_y,
        gain: s.gain,
        offset: s.offset,
        readout_mode: s.readout_mode,
        readout_mode_name: mode,
        focal_length_mm: s.focal_length_mm,
        rotator_position: s.rotation_deg,
        exposure_s: s.exposure_seconds,
        camera_temp: s.camera_temp_c,
        timestamp: s.captured_at_unix,
    };
    let mut quality = policy::QualityRecord {
        frame_uuid: frame.frame_uuid.clone(),
        source_fingerprint: Some(source.fingerprint.clone()),
        ..Default::default()
    };
    if !light && !frame.is_master {
        let measured = if kind == Kind::Flat {
            quality.algorithm_version = Some(seiza_calibration::FLAT_QUALITY_ALGORITHM_VERSION);
            seiza_calibration::measure_flat(
                image_ref(&input)?,
                input.bayer.is_some(),
                &Default::default(),
            )
            .map(|measurement| {
                policy::StoredMeasurement::Flat(policy::FlatRecord {
                    measurement,
                    optics: None,
                })
            })
        } else {
            quality.algorithm_version = Some(seiza_calibration::FRAME_QUALITY_ALGORITHM_VERSION);
            seiza_calibration::measure_dark(image_ref(&input)?, &Default::default())
                .map(policy::StoredMeasurement::Dark)
        };
        match measured {
            Ok(m) => quality.measurement = Some(m),
            Err(e) => quality.measure_error = Some(e.to_string()),
        }
    }
    verify(std::slice::from_ref(&source))?;
    let processed_source = stamp(&processed)?;
    let row = Cached {
        source,
        frame,
        light: target,
        quality,
        processed,
        processed_source,
        target: Some(frame_target(&input)),
    };
    atomic_json(&memo, &row)?;
    Ok(row)
}
fn discover(
    root: &Path,
    cache: &Path,
    files: &mut Vec<PathBuf>,
    cancel: &CancelSignal,
) -> Result<()> {
    check(cancel)?;
    if root.starts_with(cache) {
        return Ok(());
    }
    let metadata = std::fs::symlink_metadata(root).map_err(error)?;
    if metadata.file_type().is_symlink() {
        return Ok(());
    }
    if metadata.is_dir() {
        let mut entries = std::fs::read_dir(root)
            .map_err(error)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(error)?;
        entries.sort_by_key(|e| e.path());
        for entry in entries {
            discover(&entry.path(), cache, files, cancel)?;
        }
    } else if root.extension().and_then(|e| e.to_str()).is_some_and(|e| {
        ["fit", "fits", "fts", "xisf", "xish"]
            .iter()
            .any(|s| s.eq_ignore_ascii_case(e))
    }) {
        files.push(root.to_owned());
    }
    Ok(())
}

fn run(
    request: Request,
    cancel: CancelSignal,
    progress: &dyn Fn(&PreparationProgress),
) -> Result<PreparationReport> {
    if !["analyze", "execute"].contains(&request.operation.as_str()) {
        return Err("Unknown calibration operation".into());
    }
    if request.lights.is_empty() || request.cache.as_os_str().is_empty() {
        return Err("Choose lights and a calibration cache".into());
    }
    // Validate all native settings before touching image files.
    if request.rules.is_none() {
        planning::rules(&request.settings, &request.rigs)?;
    }
    std::fs::create_dir_all(&request.cache).map_err(error)?;
    let cache = request.cache.canonicalize().map_err(error)?;
    let library = if request.library.as_os_str().is_empty() {
        None
    } else {
        Some(request.library.canonicalize().map_err(error)?)
    };
    if library.as_ref().is_some_and(|p| p.starts_with(&cache)) {
        return Err("The calibration library cannot be inside its cache".into());
    }
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(cache.join("library.lock"))
        .map_err(error)?;
    loop {
        check(&cancel)?;
        match fs2::FileExt::try_lock_exclusive(&lock) {
            Ok(()) => break,
            Err(e) if e.raw_os_error() == fs2::lock_contended_error().raw_os_error() => {
                std::thread::sleep(std::time::Duration::from_millis(50))
            }
            Err(e) => return Err(e.to_string()),
        }
    }
    progress(&PreparationProgress {
        phase: "discover",
        completed: 0,
        total: 0,
    });
    let mut found = request.provided.clone();
    if let Some(library) = &library {
        discover(library, &cache, &mut found, &cancel)?;
    }
    if found.is_empty() {
        return Err("Choose calibration files or a library containing calibration frames".into());
    }
    let mut identities = HashSet::new();
    let mut visited = HashSet::new();
    let mut rows = Vec::new();
    let mut warnings = Vec::new();
    let count = found.len();
    for (index, path) in found.into_iter().enumerate() {
        progress(&PreparationProgress {
            phase: "measure_calibration",
            completed: index,
            total: count,
        });
        let id = same_file::Handle::from_path(&path).map_err(error)?;
        if !visited.insert(same_file::Handle::from_path(&path).map_err(error)?) {
            continue;
        }
        match inspect(&path, &cache, false, &cancel) {
            Ok(row) => {
                identities.insert(id);
                rows.push(row);
            }
            Err(e) => {
                check(&cancel)?;
                warnings.push(e);
            }
        }
    }
    let mut lights = Vec::new();
    let mut light_ids = HashSet::new();
    for (index, path) in request.lights.iter().enumerate() {
        progress(&PreparationProgress {
            phase: "measure_lights",
            completed: index,
            total: request.lights.len(),
        });
        let id = same_file::Handle::from_path(path).map_err(error)?;
        if identities.contains(&id) || !light_ids.insert(id) {
            return Err("Lights and calibration files must be distinct physical files".into());
        }
        lights.push(inspect(path, &cache, true, &cancel)?);
    }
    for row in &mut rows {
        if let Some(choice) = request
            .choices
            .iter()
            .find(|c| c.path == row.source.path && c.fingerprint == row.source.fingerprint)
        {
            row.quality.override_choice = Some(choice.choice);
        }
    }
    let mut planner = Planner::new(&request, &rows, &lights)?;
    planner.grouping.group_overrides = request.group_overrides.clone();
    for choice in request
        .light_choices
        .iter()
        .filter(|choice| lights.iter().any(|light| light.source.path == choice.path))
    {
        if let Some((index, light)) = lights.iter().enumerate().find(|(_, l)| {
            l.source.path == choice.path && l.source.fingerprint == choice.fingerprint
        }) {
            planner.grouping.light_choices.push(
                seiza_calibration::flat_response::groups::LightChoiceEntry {
                    light_id: index as i64,
                    file_name: light.source.path.to_string_lossy().into_owned(),
                    filter: light.light.filter.clone(),
                    at: light.light.timestamp,
                    row: choice.row.clone(),
                },
            );
        } else {
            warnings.push(format!(
                "A saved flat choice no longer matches {}. Choose again for the current source.",
                choice.path.display()
            ));
        }
    }
    progress(&PreparationProgress {
        phase: "evidence",
        completed: 0,
        total: request.lights.len(),
    });
    let mut responses = responses::prepare(&planner, &rows, &cache, &cancel, &mut warnings)?;
    let work = responses.connect(&mut planner, &lights, &cache, &cancel, &mut warnings)?;
    let grouping_snapshot = planning::grouping(planner.library()?.flat_book());
    let light_options = responses.light_options(&lights)?;
    let flat_work = work;
    let snapshot = planner.snapshot()?;
    let mut sources: Vec<_> = rows
        .iter()
        .chain(&lights)
        .map(|r| r.source.clone())
        .collect();
    for row in rows
        .iter()
        .chain(&lights)
        .filter(|r| r.processed != r.source.path)
    {
        sources.push(stamp(&row.processed)?);
    }
    sources.extend(responses.sources.clone());
    verify(&sources)?;
    let review_key = digest(
        serde_json::to_vec(&(
            &sources,
            &request.settings,
            &request.rigs,
            &request.choices,
            &request.incomplete,
            request.flat_star_masking,
            &request.flat_stability,
            request.flat_association,
            request.dark_association,
            &grouping_snapshot,
            &flat_work,
            &request.group_overrides,
            &request.light_choices,
            planner.rules.digest(),
            policy::POLICY_ALGORITHM_VERSION,
            crate::OUTPUT_VERSION,
        ))
        .map_err(error)?,
    );
    if request.operation == "execute" && request.review_key.as_deref() != Some(&review_key) {
        return Err("The library, inputs, choices or settings changed. Analyze calibration again before stacking.".into());
    }
    let by_id: BTreeMap<_, _> = rows
        .iter()
        .map(|r| (r.frame.frame_uuid.clone(), r))
        .collect();
    let mut missing = Vec::new();
    for (index, selection) in snapshot.selections.iter().enumerate() {
        let session = &snapshot.sessions[snapshot.assignments[index]];
        let available = |kind| {
            session.recipes.iter().any(|r| {
                r.kind == kind
                    && matches!(r.action, RecipeAction::External | RecipeAction::Integrate)
            })
        };
        let ungraded = selection
            .bias
            .iter()
            .chain(&selection.dark)
            .chain(&selection.dark_flat)
            .chain(&selection.flat)
            .any(|id| {
                by_id.get(id).is_some_and(|r| {
                    !r.frame.is_master
                        && r.quality.measurement.is_none()
                        && !matches!(
                            r.quality.override_choice,
                            Some(policy::QualityOverride::Use)
                        )
                })
            });
        missing.push(
            ungraded
                || selection.flat_basis.flags()
                || selection.flat_family.as_ref().is_some_and(|f| f.flags())
                || (!available(Kind::Bias) && !available(Kind::Dark) && !available(Kind::Flat))
                || (selection.library.bias && !available(Kind::Bias))
                || (selection.library.dark && !available(Kind::Dark))
                || (selection.library.flat && !available(Kind::Flat)),
        );
    }
    if missing.iter().any(|v| *v) {
        warnings.push(format!("{} lights have incomplete or ungraded calibration. Review Light matches and choose how to handle incomplete lights.",missing.iter().filter(|v|**v).count()));
    }
    let mut frames = Vec::new();
    let mut rigs = BTreeMap::new();
    for row in rows.iter().chain(&lights) {
        rigs.insert(
            row.frame.rig_uuid.clone(),
            format!(
                "{} · {}",
                row.frame.camera.as_deref().unwrap_or("Unknown camera"),
                row.frame
                    .telescope
                    .as_deref()
                    .unwrap_or("Unknown telescope")
            ),
        );
    }
    // Present the exact judgments that kept frames in/out of the library.
    let quality = &planner.quality;
    let judged = &planner.judged;
    for row in &rows {
        let grade = judged.get(&row.frame.frame_uuid);
        frames.push(Frame {
            path: row.source.path.clone(),
            fingerprint: row.source.fingerprint.clone(),
            id: row.frame.frame_uuid.clone(),
            kind: row.frame.kind,
            is_master: row.frame.is_master,
            grade: grade.map(|g| g.grade),
            reason: quality
                .kept_out(&row.frame, grade)
                .or_else(|| grade.and_then(|g| g.reason.clone())),
            measurement_state: quality.measurement_state(&row.frame),
            error: row.quality.measure_error.clone(),
        });
    }
    let mut file_sessions = Vec::new();
    if request.operation == "execute" {
        if missing.iter().any(|v| *v) && request.incomplete == "stop" {
            return Err("Some lights lack usable graded calibration. Review the plan or change Incomplete lights.".into());
        }
        if !["stop", "keep", "exclude"].contains(&request.incomplete.as_str()) {
            return Err("Choose an incomplete-light policy".into());
        }
        for (session_index, session) in snapshot.sessions.iter().enumerate() {
            check(&cancel)?;
            progress(&PreparationProgress {
                phase: "masters",
                completed: session_index,
                total: snapshot.sessions.len(),
            });
            let mut context = planner.clone();
            context.lights = planner
                .lights
                .iter()
                .zip(&snapshot.assignments)
                .filter(|(_, a)| **a == session_index)
                .map(|(l, _)| l.clone())
                .collect();
            let file = execute(
                session,
                &context,
                &by_id,
                &lights,
                &responses,
                &cache,
                &cancel,
                &mut warnings,
            )?;
            // A retry or pedestal fit can withhold a flat after the preview.
            // Reapply the user's incomplete-light decision to the executed plan.
            if session.recipes.iter().any(|r| {
                r.kind == Kind::Flat
                    && matches!(r.action, RecipeAction::External | RecipeAction::Integrate)
            }) && file.flat.is_none()
            {
                for (index, assignment) in snapshot.assignments.iter().enumerate() {
                    if *assignment == session_index {
                        missing[index] = true;
                    }
                }
            }
            file_sessions.push(file);
        }
        if request.incomplete == "stop" && missing.iter().any(|v| *v) {
            return Err("A flat was withheld during preparation. Review calibration or choose how to handle incomplete lights.".into());
        }
        for file in &file_sessions {
            for path in [&file.bias, &file.dark, &file.flat].into_iter().flatten() {
                sources.push(stamp(path)?);
            }
        }
    }
    check(&cancel)?;
    verify(&sources)?;
    progress(&PreparationProgress {
        phase: "complete",
        completed: request.lights.len(),
        total: request.lights.len(),
    });
    Ok(PreparationReport {
        review_key,
        frames,
        rigs: rigs
            .into_iter()
            .map(|(id, name)| Rig { id, name })
            .collect(),
        snapshot,
        grouping: grouping_snapshot,
        flat_work,
        light_options,
        warnings,
        missing,
        paths: lights.iter().map(|l| l.processed.clone()).collect(),
        light_paths: lights.iter().map(|l| l.source.path.clone()).collect(),
        file_sessions,
        sources,
    })
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Built {
    source: Source,
    used: Vec<String>,
    warnings: Vec<String>,
}
fn build_set(
    session: &policy::Session,
    rows: &BTreeMap<String, &Cached>,
    cache: &Path,
    cancel: &CancelSignal,
    warnings: &mut Vec<String>,
) -> Result<BTreeMap<Kind, Built>> {
    use crate::{MasterBuildOptions, MasterFrameKind};
    let mut outputs: BTreeMap<Kind, Built> = BTreeMap::new();
    for recipe in &session.recipes {
        check(cancel)?;
        let inputs = recipe
            .inputs
            .iter()
            .map(|id| {
                rows.get(id)
                    .copied()
                    .ok_or("Unknown master input".to_owned())
            })
            .collect::<Result<Vec<_>>>()?;
        if recipe.action == RecipeAction::External {
            if inputs.len() != 1 {
                return Err("External master needs exactly one source".into());
            }
            outputs.insert(
                recipe.kind,
                Built {
                    source: stamp(&inputs[0].processed)?,
                    used: recipe.inputs.clone(),
                    warnings: Vec::new(),
                },
            );
            continue;
        }
        if recipe.action != RecipeAction::Integrate {
            continue;
        }
        if recipe.requires_source_verification {
            return Err("A master recipe still requires source verification".into());
        }
        for kind in &recipe.dependencies {
            if !outputs.contains_key(kind) {
                return Err("A declared master dependency was not built".into());
            }
        }
        let dependencies: Vec<_> = recipe
            .dependencies
            .iter()
            .filter_map(|k| outputs.get(k))
            .map(|p| stamp(&p.source.path))
            .collect::<Result<_>>()?;
        let key = digest(
            serde_json::to_vec(&(
                recipe,
                inputs.iter().map(|r| &r.source).collect::<Vec<_>>(),
                &dependencies,
                crate::OUTPUT_VERSION,
            ))
            .map_err(error)?,
        );
        let path = cache.join(format!("master-{key}.fits"));
        let memo = cache.join(format!("master-{key}.json"));
        if let Some(built) = std::fs::read(&memo)
            .ok()
            .and_then(|b| serde_json::from_slice::<Built>(&b).ok())
            .filter(|b| b.source.path == path && verify(std::slice::from_ref(&b.source)).is_ok())
        {
            warnings.extend(built.warnings.clone());
            outputs.insert(recipe.kind, built);
            continue;
        }
        let load = |kind| {
            outputs
                .get(&kind)
                .map(|p| FitsFrame::open(&p.source.path).map_err(error))
                .transpose()
        };
        let bias = if recipe.dependencies.contains(&Kind::Bias) {
            load(Kind::Bias)?.map(|f| f.image)
        } else {
            None
        };
        let dark_kind = if recipe.dependencies.contains(&Kind::DarkFlat) {
            Some(Kind::DarkFlat)
        } else if recipe.dependencies.contains(&Kind::Dark) {
            Some(Kind::Dark)
        } else {
            None
        };
        let dark = dark_kind
            .map(load)
            .transpose()?
            .flatten()
            .map(|f| crate::MasterDark::from_fits_frame(f, None).map_err(error))
            .transpose()?;
        let options = MasterBuildOptions {
            bias,
            dark,
            exposure_seconds: recipe.exposure_seconds,
            cancel: Some(cancel.clone()),
            defect_suppression: recipe.defect_suppression.then(Default::default),
            flat_star_masking: recipe.flat_star_masking.then(Default::default),
            dark_level_screening: recipe.dark_level_screening.then(Default::default),
            ..Default::default()
        };
        let kind = match recipe.kind {
            Kind::Bias => MasterFrameKind::Bias,
            Kind::Flat => MasterFrameKind::Flat,
            _ => MasterFrameKind::Dark,
        };
        let master = crate::build_master_from_fits_with_scratch(
            &inputs
                .iter()
                .map(|r| r.processed.clone())
                .collect::<Vec<_>>(),
            kind,
            &options,
            cache,
        )
        .map_err(error)?;
        let notes: Vec<_> = master
            .skipped_inputs
            .iter()
            .map(|s| format!("Skipped {}: {}", s.path.display(), s.reason))
            .collect();
        warnings.extend(notes.clone());
        let used = inputs
            .iter()
            .filter(|r| !master.skipped_inputs.iter().any(|s| s.path == r.processed))
            .map(|r| r.frame.frame_uuid.clone())
            .collect();
        check(cancel)?;
        let temp = tempfile::Builder::new()
            .suffix(".fits")
            .tempfile_in(cache)
            .map_err(error)?;
        crate::write_master_fits_f32(temp.path(), &master).map_err(error)?;
        verify(
            &inputs
                .iter()
                .flat_map(|r| [r.source.clone(), r.processed_source.clone()])
                .collect::<Vec<_>>(),
        )?;
        temp.as_file().sync_all().map_err(error)?;
        temp.persist(&path).map_err(error)?;
        let built = Built {
            source: stamp(&path)?,
            used,
            warnings: notes,
        };
        atomic_json(&memo, &built)?;
        outputs.insert(recipe.kind, built);
    }
    Ok(outputs)
}
fn flat_attempt(
    session: &policy::Session,
    outputs: &BTreeMap<Kind, Built>,
    rows: &BTreeMap<String, &Cached>,
    cancel: &CancelSignal,
    warnings: &mut Vec<String>,
) -> Result<policy::FlatAttempt> {
    check(cancel)?;
    let flat = outputs.get(&Kind::Flat).ok_or("No flat master")?;
    let recipe = session
        .recipes
        .iter()
        .find(|r| r.kind == Kind::Flat)
        .ok_or("No flat recipe")?;
    let path = |kind| outputs.get(&kind).map(|b| b.source.path.clone());
    let measurement = crate::measure_flat_stability_files(crate::FlatStabilityFileRequest {
        master: flat.source.path.clone(),
        inputs: flat
            .used
            .iter()
            .map(|id| crate::FlatStabilityInput {
                path: rows[id].processed.clone(),
                exposure_seconds: rows[id].frame.exposure_s,
                skipped: false,
            })
            .collect(),
        bias: recipe
            .dependencies
            .contains(&Kind::Bias)
            .then(|| path(Kind::Bias))
            .flatten(),
        dark: recipe
            .dependencies
            .contains(&Kind::DarkFlat)
            .then(|| path(Kind::DarkFlat))
            .flatten(),
        dark_exposure_seconds: None,
        normalized_full_scale: None,
    })
    .map_err(error)?;
    check(cancel)?;
    for (index, reason) in measurement.errors {
        warnings.push(format!("Flat stability input {}: {}", index + 1, reason));
    }
    Ok(policy::FlatAttempt {
        used: flat.used.clone(),
        disagreement: measurement.disagreement,
        drift_threshold: measurement.drift_threshold,
        feature_pixels: measurement.feature_pixels,
        ..Default::default()
    })
}
// A retry is rematched through the complete shared planner. In particular, an
// alternate flat exposure must select its own dark-flat rather than inherit the
// first flat's dependencies. Only the permitted flat candidates are narrowed.
fn retry_context(context: &Planner, ids: &[String]) -> Result<(Planner, policy::Session)> {
    let mut request = context.clone();
    request
        .frames
        .retain(|f| f.kind != Kind::Flat || ids.contains(&f.frame_uuid));
    let snapshot = request.snapshot()?;
    if snapshot.sessions.len() != 1 {
        return Err("The retry no longer has a shared calibration session".into());
    }
    Ok((request, snapshot.sessions.into_iter().next().unwrap()))
}
#[allow(clippy::too_many_arguments)]
fn execute(
    session: &policy::Session,
    context: &Planner,
    rows: &BTreeMap<String, &Cached>,
    lights: &[Cached],
    responses: &responses::Prepared,
    cache: &Path,
    cancel: &CancelSignal,
    warnings: &mut Vec<String>,
) -> Result<FileCalibrationSession> {
    let mut effective = session.clone();
    let mut effective_context = context.clone();
    let mut outputs = build_set(session, rows, cache, cancel, warnings)?;
    if session.flat_stability.enabled
        && session
            .recipes
            .iter()
            .any(|r| r.kind == Kind::Flat && r.action == RecipeAction::Integrate)
    {
        // Native candidate selection order is significant for alternate retries.
        let library = context.library()?;
        let light = context.lights.first().ok_or("Missing flat selection")?;
        let selection = context.select(&library, light);
        let mut feedback = policy::FlatRetry::for_session(
            flat_attempt(session, &outputs, rows, cancel, warnings)?,
            &selection.flat,
            session,
            &selection.rules,
        )
        .map_err(error)?;
        let mut trimmed = None;
        let mut alternate = None;
        loop {
            check(cancel)?;
            let decision = feedback.next();
            if let Some(note) = decision.note {
                warnings.push(note);
            }
            use policy::FlatRetryAction as A;
            match decision.action {
                A::PublishFirst => break,
                A::Withhold => {
                    outputs.remove(&Kind::Flat);
                    break;
                }
                A::PublishTrimmed => {
                    (effective_context, effective, outputs) =
                        trimmed.take().ok_or("Missing trimmed flat")?;
                    break;
                }
                A::PublishAlternate => {
                    (effective_context, effective, outputs) =
                        alternate.take().ok_or("Missing alternate flat")?;
                    break;
                }
                action => {
                    let result = (|| {
                        let (request, retry) = retry_context(context, &decision.inputs)?;
                        let built = build_set(&retry, rows, cache, cancel, warnings)?;
                        let attempt = flat_attempt(&retry, &built, rows, cancel, warnings)?;
                        Ok::<_, String>((attempt, (request, retry, built)))
                    })();
                    match action {
                        A::TryTrimmed => match result {
                            Ok((attempt, built)) => {
                                feedback.trimmed(Some(attempt)).map_err(error)?;
                                trimmed = Some(built);
                            }
                            Err(e) => {
                                check(cancel)?;
                                warnings.push(format!("Trimmed flat retry: {e}"));
                                feedback.trimmed(None).map_err(error)?;
                            }
                        },
                        A::TryAlternate => match result {
                            Ok((attempt, built)) => {
                                feedback.alternate(Some(attempt)).map_err(error)?;
                                alternate = Some(built);
                            }
                            Err(e) => {
                                check(cancel)?;
                                warnings.push(format!("Alternate flat retry: {e}"));
                                feedback.alternate(None).map_err(error)?;
                            }
                        },
                        _ => unreachable!(),
                    }
                }
            }
        }
    }
    if outputs.contains_key(&Kind::Flat)
        && matches!(
            effective.application.flat_use,
            policy::FlatUse::NeedsPedestalFit
        )
    {
        let flat = FitsFrame::open(&outputs[&Kind::Flat].source.path).map_err(error)?;
        let mut estimates = Vec::new();
        for path in &effective.application.sample_lights {
            check(cancel)?;
            let light = lights
                .iter()
                .find(|l| l.source.path == *path)
                .ok_or("Missing pedestal sample")?;
            let input = FitsFrame::open(&light.processed).map_err(error)?;
            // CFA colour planes do not share the mono fit's response model.
            if input.bayer.is_none() && input.image.channels == 1 {
                match seiza_calibration::fit_flat_pedestal(image_ref(&input)?, image_ref(&flat)?) {
                    Ok(Some(value)) => estimates.push(value),
                    Ok(None) => {}
                    Err(e) => warnings.push(format!("Pedestal fit {}: {e}", path.display())),
                }
            }
        }
        effective.accept_pedestal(&estimates, None).map_err(error)?;
        if let Some(value) = effective.application.pedestal {
            // The fixed C opener has no scalar-pedestal slot. Materialize the
            // shared policy's accepted constant as a bias master for both paths.
            let source = lights
                .iter()
                .find(|l| effective.application.sample_lights.contains(&l.source.path))
                .ok_or("No pedestal sample")?;
            let mut input = FitsFrame::open(&source.processed).map_err(error)?;
            input.image.data.fill(value);
            input.headers.retain(|(k, _)| {
                !["IMAGETYP", "OBSTYPE", "FRAME", "SEIZAMST"].contains(&k.as_str())
            });
            input.headers.push((
                "IMAGETYP".into(),
                seiza_fits::HeaderValue::String("MASTER BIAS".into()),
            ));
            input.headers.push((
                "SEIZAMST".into(),
                seiza_fits::HeaderValue::String("BIAS".into()),
            ));
            let path = cache.join(format!(
                "pedestal-{}.fits",
                digest(
                    serde_json::to_vec(&(&effective.key, value, &source.source)).map_err(error)?
                )
            ));
            let memo = path.with_extension("json");
            let reusable = std::fs::read(&memo)
                .ok()
                .and_then(|b| serde_json::from_slice::<Source>(&b).ok())
                .is_some_and(|s| s.path == path && verify(&[s]).is_ok());
            if !reusable {
                publish_frame(&path, &input)?;
                atomic_json(&memo, &stamp(&path)?)?;
            }
            outputs.insert(
                Kind::Bias,
                Built {
                    source: stamp(&path)?,
                    used: Vec::new(),
                    warnings: Vec::new(),
                },
            );
        }
    }
    if matches!(effective.application.flat_use, policy::FlatUse::Withheld) {
        outputs.remove(&Kind::Flat);
        warnings.push("Seiza withheld a flat because a consistent pedestal could not be fitted. Supply matching bias/dark calibration.".into());
    }
    if let Some(flat) = outputs.get(&Kind::Flat) {
        let corrected =
            responses.correct_master(flat, &effective_context, lights, cache, cancel, warnings)?;
        outputs.insert(Kind::Flat, corrected);
    }
    let path = |kind| outputs.get(&kind).map(|b| b.source.path.clone());
    let dark_exposure_seconds = effective
        .recipes
        .iter()
        .find(|r| r.kind == Kind::Dark)
        .and_then(|r| r.exposure_seconds);
    Ok(FileCalibrationSession {
        bias: path(Kind::Bias),
        dark: path(Kind::Dark),
        flat: path(Kind::Flat),
        dark_exposure_seconds,
        tolerances: effective.tolerances,
        ..Default::default()
    })
}

/// How preparation treats lights with missing, ungraded or undecided calibration.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IncompleteLights {
    /// Refuse preparation until the host reviews incomplete lights.
    #[default]
    Stop,
    /// Prepare the available calibration and retain incomplete lights.
    Keep,
    /// Mark incomplete lights for the host to omit from its original input list.
    Exclude,
}
impl IncompleteLights {
    /// Every supported handling mode.
    pub const ALL: &'static [Self] = &[Self::Stop, Self::Keep, Self::Exclude];
}
impl std::fmt::Display for IncompleteLights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Stop => "stop",
            Self::Keep => "keep",
            Self::Exclude => "exclude",
        })
    }
}
impl std::str::FromStr for IncompleteLights {
    type Err = String;
    fn from_str(word: &str) -> std::result::Result<Self, Self::Err> {
        match word {
            "stop" => Ok(Self::Stop),
            "keep" => Ok(Self::Keep),
            "exclude" => Ok(Self::Exclude),
            _ => Err(format!("Unknown incomplete-light mode: {word}")),
        }
    }
}
/// Host choices and builder options. Matching/grade settings belong in the
/// native `RuleBook`, which database clients can also use independently.
#[derive(Clone, Debug, Default)]
pub struct PreparationOptions {
    /// Treatment of incomplete lights; the default stops before master building.
    pub incomplete: IncompleteLights,
    /// Native flat masking and stability policy.
    pub masters: policy::SessionOptions,
    /// Choose flat and dark sets independently; automatic by default.
    pub associations: policy::Associations,
    /// Quality overrides bound to the exact measured source.
    pub choices: Vec<Choice>,
    /// Measured flat-group edges a person split or joined.
    pub group_overrides: Vec<seiza_calibration::flat_response::groups::GroupOverrideRow>,
    /// Per-light group/side choices, bound to exact source paths and stamps.
    pub light_choices: Vec<LightChoice>,
}
/// A bounded progress update, emitted synchronously by the coordinator.
/// `total == 0` denotes discovery whose total is not known yet.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize)]
pub struct PreparationProgress {
    /// `discover`, `measure_calibration`, `measure_lights`, `evidence`,
    /// `masters`, or `complete`. No callback or context is retained.
    pub phase: &'static str,
    /// Work units already reached in this phase.
    pub completed: usize,
    /// Units in this phase, or zero while discovering.
    pub total: usize,
}
/// A preparation refusal callers can distinguish from a pixel/build failure.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Debug, thiserror::Error)]
pub enum PreparationError {
    /// Cancellation was observed; completed immutable cache entries remain reusable.
    #[error("Calibration preparation was cancelled")]
    Cancelled,
    /// Sources, settings or choices changed since the review being executed.
    #[error("Calibration review changed; analyze again before preparation")]
    ReviewChanged,
    /// File, configuration or master preparation failed, with a concrete reason.
    #[error("{0}")]
    Failed(String),
}
/// File-backed calibration coordinator shared by desktop/CLI hosts. It does
/// not hold security bookmarks, UI preferences or database state. Rust callers
/// set native rules/options and call methods directly; C/Python translate here.
#[derive(Clone, Debug)]
pub struct FileCalibration {
    /// Original ordered light paths. Review/prepared sessions preserve order.
    pub lights: Vec<PathBuf>,
    /// Explicit cache directory, owned and managed by the host. Never auto-pruned.
    pub cache: PathBuf,
    /// Optional recursively discovered calibration library; directory links are skipped.
    pub library: Option<PathBuf>,
    /// Explicit calibration files, in addition to the optional directory.
    pub provided: Vec<PathBuf>,
    /// Validated native matching/grading rules, global and per rig.
    pub rules: policy::rules::RuleBook,
    /// Independent association modes, builder options and persisted choices.
    pub options: PreparationOptions,
}
impl FileCalibration {
    /// A coordinator for ordered lights and an explicit cache. Add provided files
    /// or a library directory before analysis. Defaults are Seiza's native rules.
    pub fn new(lights: Vec<PathBuf>, cache: PathBuf) -> Self {
        Self {
            lights,
            cache,
            library: None,
            provided: Vec::new(),
            rules: Default::default(),
            options: Default::default(),
        }
    }
    /// Measure/grade/group/match and return a review without integrating masters.
    /// Calls are serialized per cache; cancellation also interrupts lock waits.
    pub fn analyze(
        &self,
        cancel: CancelSignal,
        progress: impl Fn(&PreparationProgress),
    ) -> std::result::Result<PreparationReport, PreparationError> {
        self.perform(None, cancel, &progress)
    }
    /// Revalidate the reviewed inputs, then publish each unique master's exact
    /// dependencies and corrections atomically. A required correction failure
    /// aborts; it never silently returns an uncorrected master. Apply `missing`
    /// in original input order when the chosen policy is `Exclude`.
    pub fn prepare(
        &self,
        review_key: &str,
        cancel: CancelSignal,
        progress: impl Fn(&PreparationProgress),
    ) -> std::result::Result<PreparationReport, PreparationError> {
        self.perform(Some(review_key), cancel, &progress)
    }
    fn perform(
        &self,
        key: Option<&str>,
        cancel: CancelSignal,
        progress: &dyn Fn(&PreparationProgress),
    ) -> std::result::Result<PreparationReport, PreparationError> {
        let request = Request {
            rules: Some(self.rules.clone()),
            operation: if key.is_some() { "execute" } else { "analyze" }.into(),
            lights: self.lights.clone(),
            cache: self.cache.clone(),
            library: self.library.clone().unwrap_or_default(),
            provided: self.provided.clone(),
            review_key: key.map(str::to_owned),
            incomplete: self.options.incomplete.to_string(),
            choices: self.options.choices.clone(),
            group_overrides: self.options.group_overrides.clone(),
            light_choices: self.options.light_choices.clone(),
            flat_star_masking: self.options.masters.flat_star_masking,
            flat_stability: self.options.masters.flat_stability.clone(),
            flat_association: self.options.associations.flat,
            dark_association: self.options.associations.dark,
            ..Default::default()
        };
        run(request, cancel.clone(), progress).map_err(|reason| {
            if cancel.is_cancelled() {
                PreparationError::Cancelled
            } else if reason.starts_with("The library, inputs, choices or settings changed.") {
                PreparationError::ReviewChanged
            } else {
                PreparationError::Failed(reason)
            }
        })
    }
}
/// Verify immutable source/master stamps immediately before a stack uses them.
/// Unix stamps include device/inode; other platforms use canonical path, size,
/// modification and creation times. These are metadata identities, not pixel hashes.
pub fn verify_sources(sources: &[Source]) -> std::result::Result<(), PreparationError> {
    verify(sources).map_err(|_| PreparationError::ReviewChanged)
}

#[cfg(test)]
mod tests {
    use super::*;
    // Keep original #92 file fixtures while asserting the native records.
    fn presentation(value: Value) -> Value {
        match value {
            Value::Array(v) => Value::Array(v.into_iter().map(presentation).collect()),
            Value::Object(v) => Value::Object(
                v.into_iter()
                    .map(|(k, v)| {
                        let k = match k.as_str() {
                            "review_key" => "reviewKey",
                            "file_sessions" => "fileSessions",
                            "light_options" => "lightOptions",
                            "flat_work" => "flatWork",
                            other => other,
                        }
                        .to_owned();
                        (k, presentation(v))
                    })
                    .collect(),
            ),
            other => other,
        }
    }
    #[test]
    fn native_coordinator_progress_review_binding_and_cache_lock_cancellation() {
        use std::cell::RefCell;
        let (_dir, request) = fixture();
        let mut files = FileCalibration::new(request.lights.clone(), request.cache.clone());
        files.library = Some(request.library.clone());
        let phases = RefCell::new(Vec::new());
        let review = files
            .analyze(CancelSignal::new(|| false), |p| {
                phases.borrow_mut().push(p.phase)
            })
            .unwrap();
        assert!(phases.borrow().contains(&"evidence"));
        assert!(review.file_sessions.is_empty());
        let built = files
            .prepare(&review.review_key, CancelSignal::new(|| false), |_| {})
            .unwrap();
        assert_eq!(built.snapshot.assignments, vec![0, 1, 0]);
        assert_eq!(built.file_sessions.len(), 2);
        verify_sources(&built.sources).unwrap();
        files.rules = policy::rules::RuleBook::validated(
            BTreeMap::from([("use_darks".into(), json!(false))]),
            BTreeMap::new(),
        )
        .unwrap();
        assert!(matches!(
            files.prepare(&review.review_key, CancelSignal::new(|| false), |_| {}),
            Err(PreparationError::ReviewChanged)
        ));
        let lock = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(request.cache.join("library.lock"))
            .unwrap();
        fs2::FileExt::lock_exclusive(&lock).unwrap();
        let polls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count = std::sync::Arc::clone(&polls);
        let cancel =
            CancelSignal::new(move || count.fetch_add(1, std::sync::atomic::Ordering::SeqCst) > 4);
        assert!(matches!(
            files.analyze(cancel, |_| {}),
            Err(PreparationError::Cancelled)
        ));
    }
    fn write_test_image(
        path: &Path,
        width: usize,
        height: usize,
        pixels: seiza_fits::F32ImageData<'_>,
        headers: &[seiza_fits::WriteHeaderCard],
    ) -> std::result::Result<(), seiza_image_io::Error> {
        seiza_image_io::save_f32(
            path,
            width,
            height,
            pixels,
            &seiza_image_io::SaveOptions {
                headers,
                ..Default::default()
            },
        )
    }
    fn fixture() -> (tempfile::TempDir, Request) {
        let dir = tempfile::tempdir().unwrap();
        let library = dir.path().join("library");
        std::fs::create_dir(&library).unwrap();
        let cache = dir.path().join("cache");
        let mut lights = Vec::new();
        for (index, gain) in [100, 200, 100].into_iter().enumerate() {
            let path = dir.path().join(format!("light-{index}.fits"));
            write(&path, "LIGHT", gain, 1000.0);
            lights.push(path);
        }
        for gain in [100, 200] {
            for index in 0..3 {
                write(
                    &library.join(format!("bias-{gain}-{index}.fits")),
                    "BIAS",
                    gain,
                    500.0,
                );
            }
        }
        (
            dir,
            Request {
                operation: "analyze".into(),
                library,
                cache,
                lights,
                incomplete: "stop".into(),
                ..Default::default()
            },
        )
    }
    fn write(path: &Path, kind: &str, gain: i64, level: f32) {
        use seiza_fits::HeaderValue as H;
        let headers: Vec<(String, H)> = vec![
            ("IMAGETYP".into(), H::String(kind.into())),
            ("INSTRUME".into(), H::String("TestCam".into())),
            ("GAIN".into(), H::Integer(gain)),
            ("OFFSET".into(), H::Integer(50)),
            ("XBINNING".into(), H::Integer(1)),
            ("YBINNING".into(), H::Integer(1)),
            (
                "EXPTIME".into(),
                H::Float(if kind == "BIAS" { 0.01 } else { 30.0 }),
            ),
            ("DATE-OBS".into(), H::String("2026-10-20T20:00:00Z".into())),
        ];
        let image = crate::LinearImage::new(
            64,
            64,
            1,
            (0..4096).map(|i| level + (i % 7) as f32 - 3.0).collect(),
        )
        .unwrap();
        write_test_image(
            path,
            64,
            64,
            seiza_fits::F32ImageData::Mono(&image.data),
            &headers
                .into_iter()
                .map(|(k, v)| seiza_fits::WriteHeaderCard::new(k, v))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
    fn call(request: Request) -> Value {
        presentation(
            serde_json::to_value(run(request, CancelSignal::new(|| false), &|_| {}).unwrap())
                .unwrap(),
        )
    }

    fn response_fixture(correction: bool, families: bool) -> (tempfile::TempDir, Request) {
        use seiza_fits::HeaderValue as H;
        let (dir, mut request) = fixture();
        for path in std::fs::read_dir(&request.library).unwrap() {
            std::fs::remove_file(path.unwrap().path()).unwrap();
        }
        request.lights.clear();
        let write = |path: &Path, kind: &str, filter: &str, angle: f64, day: u8, dust: usize| {
            let data = response_pixels(dust)
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    let noise = ((i * 17 + i / 240 * 13) % 101) as f32 / 100.0 - 0.5;
                    let v = v * if correction && angle == 90.0 {
                        1.0 - 0.04
                            * ((14.0
                                - (((i % 240) as f32 - 170.0).powi(2)
                                    + ((i / 240) as f32 - 40.0).powi(2))
                                .sqrt())
                                / 2.0)
                                .clamp(0.0, 1.0)
                    } else {
                        1.0
                    };
                    match kind {
                        "BIAS" => 500.0 + noise * 2.0,
                        "FLAT" => 500.0 + 20000.0 * v + noise * 3.0,
                        _ => 500.0 + 1000.0 * v + noise * 2.0,
                    }
                })
                .collect::<Vec<_>>();
            let cards = [
                ("IMAGETYP", H::String(kind.into())),
                ("FILTER", H::String(filter.into())),
                ("ROTATANG", H::Float(angle)),
                ("INSTRUME", H::String("TestCam".into())),
                ("GAIN", H::Integer(100)),
                ("OFFSET", H::Integer(50)),
                ("XBINNING", H::Integer(1)),
                ("YBINNING", H::Integer(1)),
                (
                    "EXPTIME",
                    H::Float(if kind == "BIAS" { 0.01 } else { 30.0 }),
                ),
                ("DATE-OBS", H::String(format!("2026-10-{day}T20:00:00Z"))),
                ("OBJECT", H::String("M42".into())),
            ]
            .into_iter()
            .map(|(k, v)| seiza_fits::WriteHeaderCard::new(k, v))
            .collect::<Vec<_>>();
            write_test_image(
                path,
                240,
                160,
                seiza_fits::F32ImageData::Mono(&data),
                &cards,
            )
            .unwrap();
        };
        for i in 0..3 {
            write(
                &request.library.join(format!("bias-{i}.fits")),
                "BIAS",
                "Ha",
                0.0,
                20,
                0,
            );
        }
        let sets = if correction {
            vec![
                ("Ha", 0.0, 20, 0),
                ("OIII", 0.0, 20, 0),
                ("Ha", 0.0, 26, 1),
                ("OIII", 0.0, 26, 1),
                ("Ha", 90.0, 26, 1),
            ]
        } else if families {
            vec![("Ha", 86.0, 20, 2), ("Ha", 90.0, 20, 1)]
        } else {
            vec![("Ha", 0.0, 20, 0), ("Ha", 0.0, 26, 1)]
        };
        for (set, (filter, angle, day, dust)) in sets.into_iter().enumerate() {
            for i in 0..3 {
                write(
                    &request.library.join(format!("flat-{set}-{i}.fits")),
                    "FLAT",
                    filter,
                    angle,
                    day,
                    dust,
                );
            }
        }
        for i in 0..5 {
            let path = dir.path().join(format!("sky-{i}.fits"));
            write(
                &path,
                "LIGHT",
                "Ha",
                if correction {
                    90.0
                } else if families {
                    89.0
                } else {
                    0.0
                },
                if correction {
                    19
                } else if families {
                    21
                } else {
                    23
                },
                usize::from(!correction),
            );
            request.lights.push(path);
        }
        (dir, request)
    }
    fn response_pixels(dust: usize) -> Vec<f32> {
        (0..240 * 160)
            .map(|i| {
                let x = (i % 240) as f32;
                let y = (i / 240) as f32;
                let mut v = 1.0
                    - 0.3 * ((x - 120.0).powi(2) + (y - 80.0).powi(2))
                        / (120.0f32.powi(2) + 80.0f32.powi(2));
                if dust != 0 {
                    let (cx, cy) = if dust == 1 {
                        (60.0, 90.0)
                    } else {
                        (170.0, 40.0)
                    };
                    let distance = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
                    v *= 1.0 - 0.03 * ((14.0 - distance) / 2.0).clamp(0.0, 1.0);
                }
                v
            })
            .collect()
    }
    #[test]
    fn light_change_evidence_is_measured_cached_and_used_in_selection() {
        let (_dir, mut request) = response_fixture(false, false);
        let review = call(request_from(&request, "analyze", None));
        assert_eq!(
            review["flatWork"]["grouping"]["evidence"][0]["verdict"], "after",
            "{review}"
        );
        let evidence = std::fs::read_dir(&request.cache)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name().to_string_lossy().contains("evidence-")
                    || e.file_name().to_string_lossy().starts_with("light-bin-")
            })
            .map(|e| stamp(&e.path()).unwrap())
            .collect::<Vec<_>>();
        assert!(!evidence.is_empty());
        request.operation = "execute".into();
        request.review_key = review["reviewKey"].as_str().map(str::to_owned);
        let built = call(request);
        assert_eq!(built["reviewKey"], review["reviewKey"]);
        assert_eq!(built["missing"], json!([false, false, false, false, false]));
        assert_eq!(
            built["snapshot"]["selections"][0]["flat_held_back"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        assert_eq!(
            built["snapshot"]["sessions"][0]["recipes"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["kind"] == "flat")
                .unwrap()["inputs"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        verify(&evidence).unwrap();
    }
    fn request_from(request: &Request, operation: &str, key: Option<String>) -> Request {
        Request {
            operation: operation.into(),
            library: request.library.clone(),
            cache: request.cache.clone(),
            lights: request.lights.clone(),
            settings: request.settings.clone(),
            review_key: key,
            incomplete: request.incomplete.clone(),
            ..Default::default()
        }
    }
    #[test]
    fn light_family_pixels_select_the_matching_response() {
        let (_dir, request) = response_fixture(false, true);
        let review = call(request);
        assert!(
            review["flatWork"]["grouping"]["family_evidence"][0]["chosen"].is_string(),
            "{review}"
        );
        assert_eq!(
            review["snapshot"]["selections"][0]["flat_family"]["by"], "pixels",
            "{review}"
        );
    }

    #[test]
    fn manual_group_and_path_scoped_side_choices_reach_native_policy() {
        use seiza_calibration::flat_response::groups as G;
        let (_dir, mut request) = response_fixture(false, false);
        // Identical basenames/timestamps are common across acquisition folders.
        let second = request.cache.parent().unwrap().join("second");
        std::fs::create_dir(&second).unwrap();
        let copy = second.join(request.lights[0].file_name().unwrap());
        std::fs::copy(&request.lights[0], &copy).unwrap();
        request.lights.push(copy);
        let review = call(request_from(&request, "analyze", None));
        let option = &review["lightOptions"][0];
        let change = &option["changes"][0];
        let row:G::LightChoiceRow=serde_json::from_value(json!({
            "light_guid":option["guid"],"change_key":change["key"],"rig_uuid":option["rig_uuid"],
            "change_start":change["start"],"change_end":change["end"],"choice":"before",
            "group_frame":null,"group_session_key":null,"label":"Before camera cleaning","chosen_at":1
        })).unwrap();
        request.light_choices = vec![LightChoice {
            path: PathBuf::from(option["path"].as_str().unwrap()),
            fingerprint: option["fingerprint"].as_str().unwrap().into(),
            row,
        }];
        let picked = call(Request {
            light_choices: request.light_choices.clone(),
            ..request_from(&request, "analyze", None)
        });
        assert!(
            !picked["snapshot"]["selections"][0]["flat_choices"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert!(
            picked["snapshot"]["selections"][5]["flat_choices"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert_ne!(
            picked["snapshot"]["assignments"][0],
            picked["snapshot"]["assignments"][5]
        );
        let mut stale = request.light_choices.clone();
        stale[0].fingerprint = "old-source".into();
        let ignored = call(Request {
            light_choices: stale,
            ..request_from(&request, "analyze", None)
        });
        assert!(
            ignored["snapshot"]["selections"][0]["flat_choices"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert!(
            ignored["warnings"]
                .as_array()
                .unwrap()
                .iter()
                .any(|w| w.as_str().unwrap().contains("saved flat choice"))
        );
        let mut edge: G::GroupOverrideRow =
            serde_json::from_value(review["flatWork"]["group_edges"][0]["row"].clone()).unwrap();
        edge.choice = Some(G::GroupChoice::Join);
        let joined = call(Request {
            group_overrides: vec![edge],
            ..request_from(&request, "analyze", None)
        });
        assert_eq!(
            joined["grouping"]["families"][0]["groups"]
                .as_array()
                .unwrap()
                .len(),
            1,
            "{}",
            joined["flatWork"]["overrides"]
        );
        assert_eq!(
            joined["flatWork"]["overrides"][0]["status"]["status"],
            "applied"
        );
    }
    #[test]
    fn unavailable_light_background_stays_undetermined() {
        let (_dir, request) = response_fixture(false, false);
        for path in &request.lights {
            let mut frame = FitsFrame::open(path).unwrap();
            frame.image.data.fill(500.0);
            crate::write_processed_image_fits_f32(path, &frame.image, &frame.headers, &[]).unwrap();
        }
        let review = call(request);
        assert_eq!(
            review["flatWork"]["grouping"]["evidence"][0]["verdict"],
            "undecided"
        );
        assert_eq!(review["missing"], json!([true, true, true, true, true]));
    }
    #[test]
    fn native_correction_is_applied_to_the_published_flat_and_reused() {
        let (_dir, request) = response_fixture(true, false);
        let review = call(request_from(&request, "analyze", None));
        assert_eq!(
            review["flatWork"]["corrections"].as_array().unwrap().len(),
            1,
            "{review}"
        );
        let built = call(request_from(
            &request,
            "execute",
            review["reviewKey"].as_str().map(str::to_owned),
        ));
        let flat = PathBuf::from(built["fileSessions"][0]["flat"].as_str().unwrap());
        assert!(
            flat.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("corrected-flat-")
        );
        let frame = FitsFrame::open(&flat).unwrap();
        let truth = response_pixels(0)
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                v * (1.0
                    - 0.04
                        * ((14.0
                            - (((i % 240) as f32 - 170.0).powi(2)
                                + ((i / 240) as f32 - 40.0).powi(2))
                            .sqrt())
                            / 2.0)
                            .clamp(0.0, 1.0))
            })
            .collect::<Vec<_>>();
        // The clean before-side response has no dust depression; compare the
        // dust centre to its symmetric location to cancel radial vignetting.
        assert!(
            (frame.image.data[90 * 240 + 60] / frame.image.data[90 * 240 + 180]
                - truth[90 * 240 + 60] / truth[90 * 240 + 180])
                .abs()
                < 0.005
        );
        let source = stamp(&flat).unwrap();
        call(request_from(
            &request,
            "execute",
            review["reviewKey"].as_str().map(str::to_owned),
        ));
        verify(&[source]).unwrap();
    }
    #[test]
    fn native_sessions_share_master_sets_despite_candidate_order_and_unrelated_options() {
        let (_dir, mut request) = fixture();
        request.lights = vec![request.lights[0].clone(), request.lights[2].clone()];
        let redated = |path: &Path, day: u8| {
            let mut frame = FitsFrame::open(path).unwrap();
            frame.headers.retain(|(key, _)| key != "DATE-OBS");
            frame.headers.push((
                "DATE-OBS".into(),
                seiza_fits::HeaderValue::String(format!("2026-10-{day}T20:00:00Z")),
            ));
            publish_frame(path, &frame).unwrap();
        };
        redated(&request.lights[1], 22);
        for index in 0..3 {
            redated(
                &request.library.join(format!("bias-100-{index}.fits")),
                20 + index,
            );
        }
        let review = call(request_from(&request, "analyze", None));
        assert_ne!(
            review["snapshot"]["selections"][0]["bias"][0],
            review["snapshot"]["selections"][1]["bias"][0]
        );
        assert_eq!(review["snapshot"]["assignments"], json!([0, 0]));
        let built = call(request_from(
            &request,
            "execute",
            review["reviewKey"].as_str().map(str::to_owned),
        ));
        assert_eq!(built["fileSessions"].as_array().unwrap().len(), 1);
        request.flat_star_masking = true;
        let changed = call(Request {
            flat_star_masking: true,
            ..request_from(&request, "analyze", None)
        });
        assert_ne!(review["reviewKey"], changed["reviewKey"]);
        assert_eq!(
            review["snapshot"]["sessions"][0]["key"],
            changed["snapshot"]["sessions"][0]["key"]
        );
        let rebuilt = call(Request {
            flat_star_masking: true,
            ..request_from(
                &request,
                "execute",
                changed["reviewKey"].as_str().map(str::to_owned),
            )
        });
        assert_eq!(built["fileSessions"], rebuilt["fileSessions"]);
    }
    #[test]
    fn measured_response_groups_reach_association_and_reuse_cached_pixels() {
        let (dir, mut request) = fixture();
        request.lights = vec![request.lights[0].clone()];
        request.settings = vec![Setting {
            key: "flat_session_hours".into(),
            value: json!(240.0),
        }];
        let input = FitsFrame::open(&request.lights[0]).unwrap();
        crate::write_processed_image_fits_f32(
            &request.lights[0],
            &input.image,
            &input.headers,
            &[
                seiza_fits::WriteHeaderCard::new(
                    "FILTER",
                    seiza_fits::HeaderValue::String("Ha".into()),
                ),
                seiza_fits::WriteHeaderCard::new("ROTATANG", seiza_fits::HeaderValue::Float(0.0)),
                seiza_fits::WriteHeaderCard::new(
                    "IMAGETYP",
                    seiza_fits::HeaderValue::String("LIGHT".into()),
                ),
            ],
        )
        .unwrap();
        // Same detector and angle, three nights. The last response has a
        // different vignette; the first two should form one measured group.
        for (day, depth) in [(20, 0.1_f32), (23, 0.1), (26, 0.35)] {
            for index in 0..3 {
                let path = request.library.join(format!("flat-{day}-{index}.fits"));
                use seiza_fits::HeaderValue as H;
                let cards = vec![
                    ("IMAGETYP", H::String("FLAT".into())),
                    ("FILTER", H::String("Ha".into())),
                    ("ROTATANG", H::Float(0.0)),
                    ("INSTRUME", H::String("TestCam".into())),
                    ("GAIN", H::Integer(100)),
                    ("OFFSET", H::Integer(50)),
                    ("EXPTIME", H::Float(30.0)),
                    ("XBINNING", H::Integer(1)),
                    ("YBINNING", H::Integer(1)),
                    ("DATE-OBS", H::String(format!("2026-10-{day}T20:00:00Z"))),
                ];
                let data: Vec<_> = (0..4096)
                    .map(|i| {
                        let x = (i % 64) as f32 / 63.0 - 0.5;
                        let y = (i / 64) as f32 / 63.0 - 0.5;
                        20000.0 * (1.0 - depth * (x * x + y * y)) + (i % 7) as f32 - 3.0
                    })
                    .collect();
                write_test_image(
                    &path,
                    64,
                    64,
                    seiza_fits::F32ImageData::Mono(&data),
                    &cards
                        .into_iter()
                        .map(|(k, v)| seiza_fits::WriteHeaderCard::new(k, v))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            }
        }
        let before = call(Request {
            operation: "analyze".into(),
            library: request.library.clone(),
            settings: request.settings.clone(),
            lights: request.lights.clone(),
            cache: request.cache.clone(),
            ..Default::default()
        });
        assert_eq!(
            before["grouping"]["sessions"].as_array().unwrap().len(),
            3,
            "{before}"
        );
        assert!(
            !before["grouping"]["events"].as_array().unwrap().is_empty(),
            "{before}"
        );
        assert!(
            !before["warnings"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v.as_str().unwrap().contains("not measured"))
        );
        let measurements: Vec<_> = std::fs::read_dir(&request.cache)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name();
                let name = name.to_string_lossy();
                name.starts_with("response-") || name.starts_with("comparison-")
            })
            .map(|e| stamp(&e.path()).unwrap())
            .collect();
        assert!(measurements.iter().any(|s| {
            s.path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("comparison-")
        }));
        let first_inputs = before["snapshot"]["selections"][0]["flat"]
            .as_array()
            .unwrap();
        assert_eq!(first_inputs.len(), 6);
        let held = before["snapshot"]["selections"][0]["flat_held_back"]
            .as_array()
            .unwrap();
        assert!(held.is_empty());
        let flat_recipe = before["snapshot"]["sessions"][0]["recipes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["kind"] == "flat")
            .unwrap();
        assert_eq!(flat_recipe["inputs"].as_array().unwrap().len(), 6);
        let again = call(Request {
            operation: "analyze".into(),
            library: request.library.clone(),
            settings: request.settings.clone(),
            lights: request.lights.clone(),
            cache: request.cache.clone(),
            ..Default::default()
        });
        assert_eq!(before["reviewKey"], again["reviewKey"]);
        verify(&measurements).unwrap();
        // One-set changes only association; the measured caches survive.
        request.flat_association = policy::Association::OneSet;
        let explicit = call(request);
        assert_eq!(
            explicit["snapshot"]["selections"][0]["flat"]
                .as_array()
                .unwrap()
                .len(),
            9
        );
        assert!(
            explicit["grouping"]["sessions"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert_ne!(before["reviewKey"], explicit["reviewKey"]);
        verify(&measurements).unwrap();
        drop(dir);
    }
    #[test]
    fn one_dark_set_builds_all_supplied_inputs_across_nights() {
        let (_dir, mut request) = fixture();
        request.lights.truncate(1);
        request.dark_association = policy::Association::OneSet;
        for i in 0..80 {
            let path = request.library.join(format!("dark-{i}.fits"));
            write(&path, "DARK", 100, 800.0);
            if i >= 40 {
                let bytes = std::fs::read(&path).unwrap();
                let old = b"2026-10-20T20:00:00Z";
                let offset = bytes.windows(old.len()).position(|s| s == old).unwrap();
                let mut bytes = bytes;
                bytes[offset..offset + old.len()].copy_from_slice(b"2026-10-23T20:00:00Z");
                std::fs::write(path, bytes).unwrap();
            }
        }
        let review = call(Request {
            operation: "analyze".into(),
            library: request.library.clone(),
            cache: request.cache.clone(),
            lights: request.lights.clone(),
            dark_association: request.dark_association,
            incomplete: request.incomplete.clone(),
            ..Default::default()
        });
        let dark = review["snapshot"]["sessions"][0]["recipes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["kind"] == "dark")
            .unwrap();
        assert_eq!(dark["inputs"].as_array().unwrap().len(), 80);
        request.operation = "execute".into();
        request.review_key = Some(review["reviewKey"].as_str().unwrap().into());
        let report = call(request);
        let dark_path = report["fileSessions"][0]["dark"].as_str().unwrap();
        let memo = Path::new(dark_path).with_extension("json");
        let built: Built = serde_json::from_slice(&std::fs::read(memo).unwrap()).unwrap();
        assert_eq!(built.used.len(), 80);
        let image = FitsFrame::open(dark_path).unwrap();
        assert!((image.image.data[0] - 300.0).abs() < 1.0);
    }
    #[test]
    fn explicit_files_need_no_library() {
        let (_dir, mut request) = fixture();
        request.lights.truncate(1);
        request.provided = std::fs::read_dir(&request.library)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .collect();
        request.library = PathBuf::new();
        let report = call(request);
        assert_eq!(report["frames"].as_array().unwrap().len(), 6);
        assert_eq!(report["missing"], json!([false]));
    }
    #[test]
    fn library_grades_and_builds_shared_aba_assignments() {
        let (_dir, mut request) = fixture();
        let report = call(Request {
            operation: "analyze".into(),
            library: request.library.clone(),
            cache: request.cache.clone(),
            lights: request.lights.clone(),
            incomplete: "stop".into(),
            ..Default::default()
        });
        assert_eq!(report["snapshot"]["assignments"], json!([0, 1, 0]));
        assert_eq!(report["missing"], json!([false, false, false]));
        assert!(
            report["frames"]
                .as_array()
                .unwrap()
                .iter()
                .all(|f| f["grade"] == "ok")
        );
        assert_eq!(
            std::fs::read_dir(&request.cache)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().starts_with("master-"))
                .count(),
            0
        );
        request.operation = "execute".into();
        request.review_key = Some(report["reviewKey"].as_str().unwrap().into());
        let built = call(request);
        assert_eq!(built["fileSessions"].as_array().unwrap().len(), 2);
        for session in built["fileSessions"].as_array().unwrap() {
            let frame = FitsFrame::open(session["bias"].as_str().unwrap()).unwrap();
            assert!((frame.image.data[0] - 497.0).abs() < 0.01);
        }
    }
    #[test]
    fn settings_regrade_memos_and_changed_review_is_refused() {
        let (_dir, mut request) = fixture();
        let original = call(Request {
            operation: "analyze".into(),
            library: request.library.clone(),
            cache: request.cache.clone(),
            lights: request.lights.clone(),
            incomplete: "stop".into(),
            ..Default::default()
        });
        let before: Vec<_> = std::fs::read_dir(&request.cache)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| stamp(&e.path()).unwrap())
            .collect();
        request.settings = vec![Setting {
            key: "grade_bias_level".into(),
            value: json!({"suspect":null,"poor":null}),
        }];
        request.operation = "execute".into();
        request.review_key = Some(original["reviewKey"].as_str().unwrap().into());
        assert!(
            run(request, CancelSignal::new(|| false), &|_| {})
                .unwrap_err()
                .contains("changed")
        );
        verify(&before).unwrap(); // Settings reused measured pixels without rewriting their memos.
    }
    #[test]
    fn source_changes_and_invalid_settings_fail_closed() {
        let (_dir, mut request) = fixture();
        let source = stamp(&request.lights[0]).unwrap();
        write(&request.lights[0], "LIGHT", 100, 1200.0);
        assert!(verify(&[source]).unwrap_err().contains("changed"));
        request.settings = vec![Setting {
            key: "unknown_setting".into(),
            value: json!(true),
        }];
        let cache = request.cache.clone();
        assert!(
            run(request, CancelSignal::new(|| false), &|_| {})
                .unwrap_err()
                .contains("unknown setting")
        );
        assert!(!cache.exists());
    }
    #[test]
    #[cfg(unix)]
    fn physical_aliases_are_not_separate_light_inputs() {
        let (dir, mut request) = fixture();
        let alias = dir.path().join("alias.fits");
        std::fs::hard_link(&request.lights[0], &alias).unwrap();
        request.lights.push(alias);
        assert!(
            run(request, CancelSignal::new(|| false), &|_| {})
                .unwrap_err()
                .contains("distinct physical files")
        );
    }
    #[test]
    fn cancellation_and_staged_cache_changes_are_observed() {
        let (_dir, request) = fixture();
        assert!(
            run(request, CancelSignal::new(|| true), &|_| {})
                .unwrap_err()
                .contains("cancelled")
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.fits");
        write(&path, "LIGHT", 100, 2000.0);
        let original = stamp(&path).unwrap();
        write(&path, "LIGHT", 100, 3000.0);
        assert!(verify(&[original]).is_err());
    }
    #[test]
    fn manual_use_does_not_relax_sensor_gates() {
        let (_dir, mut request) = fixture();
        let path = request.library.join("bias-100-0.fits");
        let source = stamp(&path).unwrap();
        request.choices.push(Choice {
            path: source.path,
            fingerprint: source.fingerprint,
            choice: policy::QualityOverride::Use,
        });
        let report = call(request);
        let frames = report["frames"].as_array().unwrap();
        let wrong = frames
            .iter()
            .find(|f| f["path"].as_str().unwrap().ends_with("bias-100-0.fits"))
            .unwrap()["id"]
            .clone();
        assert!(
            !report["snapshot"]["selections"][1]["bias"]
                .as_array()
                .unwrap()
                .contains(&wrong)
        );
    }
    #[test]
    fn poor_grades_stay_out_until_explicit_use_and_regrade_is_native() {
        let (_dir, mut request) = fixture();
        let bad = request.library.join("bias-bad.fits");
        write(&bad, "BIAS", 100, 5000.0);
        let first = Request {
            operation: "analyze".into(),
            library: request.library.clone(),
            cache: request.cache.clone(),
            lights: request.lights.clone(),
            incomplete: "stop".into(),
            ..Default::default()
        };
        let report = call(first);
        let bad_frame = report["frames"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["path"].as_str().unwrap().ends_with("bias-bad.fits"))
            .unwrap();
        assert_eq!(bad_frame["grade"], "poor");
        assert!(
            !report["snapshot"]["selections"][0]["bias"]
                .as_array()
                .unwrap()
                .contains(&bad_frame["id"])
        );
        let source = stamp(&bad).unwrap();
        request.choices.push(Choice {
            path: source.path,
            fingerprint: source.fingerprint,
            choice: policy::QualityOverride::Use,
        });
        let chosen = call(request);
        assert!(
            chosen["snapshot"]["selections"][0]["bias"]
                .as_array()
                .unwrap()
                .contains(&bad_frame["id"])
        );
    }
    #[test]
    fn raw_flats_build_after_bias_and_run_shared_stability_policy() {
        let (_dir, mut request) = fixture();
        for gain in [100, 200] {
            for index in 0..3 {
                write(
                    &request.library.join(format!("flat-{gain}-{index}.fits")),
                    "FLAT",
                    gain,
                    20000.0,
                );
            }
        }
        let report = call(Request {
            operation: "analyze".into(),
            library: request.library.clone(),
            cache: request.cache.clone(),
            lights: request.lights.clone(),
            incomplete: "stop".into(),
            ..Default::default()
        });
        assert_eq!(report["missing"], json!([false, false, false]));
        request.operation = "execute".into();
        request.review_key = Some(report["reviewKey"].as_str().unwrap().into());
        let built = call(request);
        for session in built["fileSessions"].as_array().unwrap() {
            let flat = FitsFrame::open(session["flat"].as_str().unwrap()).unwrap();
            assert!(flat.metadata().calibration_state.bias_subtracted);
            assert!(flat.metadata().calibration_state.flat_normalized);
            assert!((flat.image.data[0] - 1.0).abs() < 0.002);
        }
    }
    #[test]
    fn tampered_staging_is_regenerated_instead_of_reusing_measurement_row() {
        let (_dir, request) = fixture();
        std::fs::create_dir_all(&request.cache).unwrap();
        let row = inspect(
            &request.lights[0],
            &request.cache,
            true,
            &CancelSignal::new(|| false),
        )
        .unwrap();
        let fake = request.cache.join("stage.fits");
        std::fs::copy(&row.processed, &fake).unwrap();
        let mut altered = row.clone();
        altered.processed = fake.clone();
        altered.processed_source = stamp(&fake).unwrap();
        let memo = std::fs::read_dir(&request.cache)
            .unwrap()
            .filter_map(|e| e.ok())
            .find(|e| e.path().extension().is_some_and(|e| e == "json"))
            .unwrap()
            .path();
        atomic_json(&memo, &altered).unwrap();
        write(&fake, "LIGHT", 100, 3333.0);
        let fresh = inspect(
            &request.lights[0],
            &request.cache,
            true,
            &CancelSignal::new(|| false),
        )
        .unwrap();
        assert_eq!(fresh.processed, row.source.path);
    }
    #[test]
    fn drifting_flat_feedback_runs_native_trim_and_rebuild() {
        let (_dir, mut request) = fixture();
        request.lights = vec![request.lights[0].clone(), request.lights[2].clone()];
        for index in 0..11 {
            let path = request.library.join(format!("flat-{index:02}.fits"));
            write(&path, "FLAT", 100, 20000.0);
            if index < 10 {
                let mut frame = FitsFrame::open(&path).unwrap();
                for y in 27..37 {
                    for x in 27..37 {
                        frame.image.data[y * 64 + x] = 10000.0;
                    }
                }
                publish_frame(&path, &frame).unwrap();
            }
            let source = stamp(&path).unwrap();
            request.choices.push(Choice {
                path: source.path,
                fingerprint: source.fingerprint,
                choice: policy::QualityOverride::Use,
            });
        }
        let preview = call(Request {
            operation: "analyze".into(),
            library: request.library.clone(),
            cache: request.cache.clone(),
            lights: request.lights.clone(),
            choices: request.choices.clone(),
            incomplete: "stop".into(),
            ..Default::default()
        });
        request.operation = "execute".into();
        request.review_key = Some(preview["reviewKey"].as_str().unwrap().into());
        let built = call(request);
        assert!(
            built["warnings"]
                .as_array()
                .unwrap()
                .iter()
                .any(|w| w.as_str().unwrap().contains("Left out")),
            "{built}"
        );
        let master = FitsFrame::open(built["fileSessions"][0]["flat"].as_str().unwrap()).unwrap();
        assert!(master.image.data[32 * 64 + 32] < 0.55);
    }
    #[test]
    fn flat_only_fits_pedestal_through_shared_policy_and_materializes_a_bias() {
        let (dir, mut request) = fixture();
        for file in std::fs::read_dir(&request.library).unwrap() {
            std::fs::remove_file(file.unwrap().path()).unwrap();
        }
        let width = 512;
        let height = 512;
        let response: Vec<f32> = (0..width * height)
            .map(|i| {
                let x = (i % width) as f32 - 256.0;
                let y = (i / width) as f32 - 256.0;
                1.0 - 0.35 * (x * x + y * y) / (2.0 * 256.0 * 256.0)
            })
            .collect();
        let write_large = |path: &Path, kind: &str, data: Vec<f32>| {
            use seiza_fits::{HeaderValue as H, WriteHeaderCard as C};
            let cards = vec![
                C::new("IMAGETYP", H::String(kind.into())),
                C::new("INSTRUME", H::String("TestCam".into())),
                C::new("GAIN", H::Integer(100)),
                C::new("OFFSET", H::Integer(50)),
                C::new("XBINNING", H::Integer(1)),
                C::new("YBINNING", H::Integer(1)),
                C::new("EXPTIME", H::Float(30.0)),
                C::new("DATE-OBS", H::String("2026-10-20T20:00:00Z".into())),
            ];
            write_test_image(
                path,
                width,
                height,
                seiza_fits::F32ImageData::Mono(&data),
                &cards,
            )
            .unwrap();
        };
        for i in 0..2 {
            write_large(
                &request.library.join(format!("flat-{i}.fits")),
                "FLAT",
                response.iter().map(|r| r * 20000.0).collect(),
            );
        }
        request.lights.clear();
        for i in 0..2 {
            let path = dir.path().join(format!("fit-light-{i}.fits"));
            write_large(
                &path,
                "LIGHT",
                response.iter().map(|r| r * 900.0 + 500.0).collect(),
            );
            request.lights.push(path);
        }
        let preview = call(Request {
            operation: "analyze".into(),
            library: request.library.clone(),
            cache: request.cache.clone(),
            lights: request.lights.clone(),
            incomplete: "stop".into(),
            ..Default::default()
        });
        assert_eq!(
            preview["snapshot"]["sessions"][0]["application"]["flat_use"],
            "needs_pedestal_fit"
        );
        request.operation = "execute".into();
        request.review_key = Some(preview["reviewKey"].as_str().unwrap().into());
        let built = call(request);
        let bias = FitsFrame::open(built["fileSessions"][0]["bias"].as_str().unwrap()).unwrap();
        assert!(bias.metadata().is_master);
        assert_eq!(bias.metadata().role, FrameSourceRole::Bias);
        let flat = FitsFrame::open(built["fileSessions"][0]["flat"].as_str().unwrap()).unwrap();
        let light = FitsFrame::open(dir.path().join("fit-light-0.fits")).unwrap();
        let expected = seiza_calibration::fit_flat_pedestal(
            image_ref(&light).unwrap(),
            image_ref(&flat).unwrap(),
        )
        .unwrap()
        .unwrap();
        assert!((bias.image.data[0] - expected).abs() < 0.001);
        assert!(built["fileSessions"][0]["flat"].is_string());
    }
    #[test]
    fn normalized_xisf_staging_preserves_role_and_recovers_from_cache_tampering() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("unit.xisf");
        let cache = dir.path().join("cache");
        std::fs::create_dir(&cache).unwrap();
        use seiza_fits::{HeaderValue as H, WriteHeaderCard as C};
        let headers = [
            C::new("IMAGETYP", H::String("MASTER FLAT".into())),
            C::new("SEIZAMST", H::String("FLAT".into())),
            C::new("FLATNORM", H::Logical(true)),
        ];
        let data = vec![0.5; 4096];
        seiza_image_io::save_f32(
            &source,
            64,
            64,
            seiza_fits::F32ImageData::Mono(&data),
            &seiza_image_io::SaveOptions {
                headers: &headers,
                xisf_bounds: Some((0.0, 1.0)),
                ..Default::default()
            },
        )
        .unwrap();
        let first = inspect(&source, &cache, false, &CancelSignal::new(|| false)).unwrap();
        assert_ne!(first.processed, first.source.path);
        let scaled = FitsFrame::open(&first.processed).unwrap();
        assert!((scaled.image.data[0] - 32767.5).abs() < 0.01);
        assert!(scaled.metadata().is_master);
        assert!(scaled.metadata().calibration_state.flat_normalized);
        assert_eq!(scaled.metadata().role, FrameSourceRole::Flat);
        std::fs::write(&first.processed, b"tampered").unwrap();
        let repaired = inspect(&source, &cache, false, &CancelSignal::new(|| false)).unwrap();
        assert_eq!(repaired.source.fingerprint, first.source.fingerprint);
        assert!(
            (FitsFrame::open(&repaired.processed).unwrap().image.data[0] - 32767.5).abs() < 0.01
        );
    }
}
