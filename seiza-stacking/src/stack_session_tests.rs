//! A session-planned expanded mosaic checks each file when it uses it, as an
//! ordinary stack does, so a context reopens without files it never needs.

use super::*;
use crate::sessions::tests::{recreate, with_device_identities};
use crate::{
    BatchStackOptions, Continue, FileCalibrationSession, PipelineExecution, PipelineOptions,
};

/// Eight lights of a two-panel mosaic, four per panel, under three sessions
/// with bias masters of their own. The third session's lights come last.
struct Night {
    directory: tempfile::TempDir,
    paths: Vec<PathBuf>,
    sessions: Vec<FileCalibrationSession>,
    assignments: Vec<usize>,
    options: StackOptions,
}

impl Night {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let (width, height) = (384, 256);
        let paths = super::canvas_tests::mosaic(directory.path(), (2, 1), (width, height), 4, 1);
        let sessions = (0..3)
            .map(|index| {
                let bias = directory.path().join(format!("bias-{index}.fits"));
                let image = LinearImage::new(width, height, 1, vec![0.0; width * height]).unwrap();
                crate::write_processed_image_fits_f32(&bias, &image, &[], &[]).unwrap();
                FileCalibrationSession {
                    bias: Some(bias),
                    ..Default::default()
                }
            })
            .collect();
        let mut options = StackOptions {
            normalization: NormalizationMode::None,
            ..StackOptions::default()
        };
        options.acceptance.minimum_overlap_fraction = 0.1;
        Self {
            directory,
            paths,
            sessions,
            assignments: vec![0, 1, 0, 1, 0, 1, 2, 2],
            options,
        }
    }

    fn open(&self, paths: &[PathBuf], assignments: &[usize]) -> Result<LiveStacker> {
        LiveStacker::open_fits_expanding_sessions(
            paths,
            &self.sessions,
            assignments,
            self.options.clone(),
            usize::MAX,
            |_, _, _, _| Continue::Yes,
        )
    }

    /// A stack that took the first four lights, saved to a context.
    fn saved(&self) -> (LiveStacker, PathBuf) {
        let mut stack = self.open(&self.paths, &self.assignments).unwrap();
        for path in &self.paths[..4] {
            accepted(stack.push_fits(path).unwrap());
        }
        let context = self.directory.path().join("night.seiza-stack");
        stack.save_context(&context).unwrap();
        (stack, context)
    }

    fn masters(&self) -> impl Iterator<Item = &Path> {
        self.sessions.iter().flat_map(FileCalibrationSession::paths)
    }
}

fn accepted(disposition: FrameDisposition) {
    assert!(
        matches!(disposition, FrameDisposition::Accepted(_)),
        "{disposition:?}"
    );
}

fn reintegrate(stack: &LiveStacker) -> Result<crate::BatchStackResult> {
    stack.reintegrate(&BatchStackOptions::default(), |_, _, _| {})
}

/// Each path the pipeline offered, with its error when it failed.
fn push_pipelined(stack: &mut LiveStacker, paths: &[PathBuf]) -> Vec<(PathBuf, Option<String>)> {
    let mut outcomes = Vec::new();
    let report = stack
        .push_fits_pipelined(paths, &PipelineOptions::default(), |path, outcome| {
            outcomes.push((
                path.to_path_buf(),
                match outcome {
                    Ok(disposition) => {
                        accepted(disposition);
                        None
                    }
                    Err(error) => Some(error.to_string()),
                },
            ));
            Continue::Yes
        })
        .unwrap();
    let failed = outcomes.iter().filter(|(_, error)| error.is_some()).count();
    assert_eq!(
        (report.integrated, report.failed),
        (paths.len() - failed, failed)
    );
    outcomes
}

/// A volume that remounts presents every file with a new identity but the
/// same size and modification time. The context resumes and replays, while
/// the process that made the plan still tells its files were replaced.
#[test]
fn a_session_context_reopens_after_its_files_come_back_as_new_files() {
    let night = Night::new();
    let (mut stack, context) = night.saved();
    for path in night
        .paths
        .iter()
        .map(PathBuf::as_path)
        .chain(night.masters())
    {
        recreate(path);
    }
    let mut resumed = LiveStacker::open_context(&context).unwrap();
    assert_eq!(
        resumed.configuration_fingerprint(),
        stack.configuration_fingerprint()
    );
    for path in &night.paths[4..] {
        accepted(resumed.push_fits(path).unwrap());
    }
    assert_eq!(reintegrate(&resumed).unwrap().snapshot.accepted_frames, 8);
    let refused = stack.push_fits(&night.paths[4]).unwrap_err().to_string();
    assert!(refused.contains("changed or is missing"), "{refused}");
}

/// A light the stack never took is deleted and a session none of its frames
/// used gets a new master: the context opens, those lights fail one by one,
/// and the rest are taken and replay.
#[test]
fn a_session_context_reopens_without_a_light_or_master_it_never_used() {
    let night = Night::new();
    let (_, context) = night.saved();
    std::fs::remove_file(&night.paths[4]).unwrap();
    let rebuilt = night.sessions[2].bias.as_ref().unwrap();
    let modified = std::fs::metadata(rebuilt).unwrap().modified().unwrap();
    std::fs::File::options()
        .write(true)
        .open(rebuilt)
        .unwrap()
        .set_times(
            std::fs::FileTimes::new().set_modified(modified + std::time::Duration::from_secs(60)),
        )
        .unwrap();
    let mut resumed = LiveStacker::open_context(&context).unwrap();
    let outcomes = push_pipelined(&mut resumed, &night.paths[4..]);
    assert_eq!(outcomes.len(), 4);
    assert!(outcomes[0].1.is_some(), "the deleted light fails");
    assert_eq!(outcomes[1].1, None, "its session's other light is taken");
    for (path, error) in &outcomes[2..] {
        let error = error.as_deref().unwrap_or_default();
        assert!(
            error.contains("bias-2") && error.contains("changed"),
            "{}: {error}",
            path.display()
        );
    }
    assert_eq!(reintegrate(&resumed).unwrap().snapshot.accepted_frames, 5);
    // A historical master that changed refuses the replay that needs it.
    std::fs::write(night.sessions[0].bias.as_ref().unwrap(), b"rebuilt").unwrap();
    assert!(reintegrate(&resumed).is_err());
    LiveStacker::open_context(&context).unwrap();
}

/// A context 0.46 wrote, its plan holding each source's device identity,
/// opens with the same configuration and verifies its files by size and
/// modification time.
#[test]
fn a_session_context_saved_with_device_identities_still_opens() {
    let night = Night::new();
    let (stack, context) = night.saved();
    rewrite_metadata(&context, |metadata| {
        with_device_identities(&mut metadata["file_sessions"]);
    });
    let mut resumed = LiveStacker::open_context(&context).unwrap();
    assert_eq!(
        resumed.configuration_fingerprint(),
        stack.configuration_fingerprint()
    );
    recreate(&night.paths[4]);
    for path in &night.paths[4..] {
        accepted(resumed.push_fits(path).unwrap());
    }
    assert_eq!(reintegrate(&resumed).unwrap().snapshot.accepted_frames, 8);
}

/// Twin catalog rows and a hard-linked copy of a light no longer refuse the
/// plan: each repeat is a duplicate frame, as in an ordinary stack, before
/// and after a resume. A light that is a master's file still refuses it.
#[test]
fn a_repeated_or_linked_light_is_a_duplicate_frame_not_a_refused_plan() {
    let night = Night::new();
    let link = night.directory.path().join("linked.fits");
    std::fs::hard_link(&night.paths[5], &link).unwrap();
    let paths = [&night.paths[..], &[night.paths[3].clone(), link.clone()]].concat();
    let assignments = [&night.assignments[..], &[0, 1]].concat();
    let mut stack = night.open(&paths, &assignments).unwrap();
    let outcomes = push_pipelined(&mut stack, &paths);
    for (path, error) in &outcomes[..8] {
        assert_eq!(error, &None, "{}", path.display());
    }
    for (_, error) in &outcomes[8..] {
        let error = error.as_deref().unwrap_or_default();
        assert!(error.contains("already been used"), "{error}");
    }
    assert!(stack.push_fits(&link).is_err());
    let frame = FitsFrame::open(&link).unwrap();
    let refused = stack.push(frame).unwrap_err().to_string();
    assert!(
        refused.contains("same file as an earlier light"),
        "{refused}"
    );
    let context = night.directory.path().join("twins.seiza-stack");
    stack.save_context(&context).unwrap();
    let mut resumed = LiveStacker::open_context(&context).unwrap();
    assert!(resumed.push_fits(&link).is_err());
    assert_eq!(resumed.snapshot().unwrap().accepted_frames, 8);

    let master = night.directory.path().join("bias-alias.fits");
    std::fs::hard_link(night.sessions[0].bias.as_ref().unwrap(), &master).unwrap();
    let paths = [&night.paths[..], &[master]].concat();
    let assignments = [&night.assignments[..], &[1]].concat();
    let Err(refused) = night.open(&paths, &assignments) else {
        panic!("a light that is a master's file refuses the plan");
    };
    let refused = refused.to_string();
    assert!(
        refused.contains("aliases a calibration master"),
        "{refused}"
    );
}

/// One frame at a time on a supplied pool, assigned lights report the
/// calibration sessions' ordered execution, not a plain sequential request.
#[test]
fn sequential_pushes_on_a_pool_report_calibration_sessions() {
    let night = Night::new();
    let mut stack = night.open(&night.paths, &night.assignments).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let report = stack
        .push_fits_sequential_with_pool(&night.paths, None, &pool, |_, outcome| {
            accepted(outcome.unwrap());
            Continue::Yes
        })
        .unwrap();
    assert_eq!(
        report.execution,
        PipelineExecution::SequentialCalibrationSessions
    );
    assert_eq!(report.frames.integrated, 8);
    let empty = stack
        .push_fits_sequential_with_pool(&[], None, &pool, |_, _| Continue::Yes)
        .unwrap();
    assert_eq!(
        empty.execution,
        PipelineExecution::SequentialCalibrationSessions
    );
}

/// Rewrite a context's JSON metadata in place, keeping every other section.
fn rewrite_metadata(path: &Path, edit: impl FnOnce(&mut serde_json::Value)) {
    let bytes = std::fs::read(path).unwrap();
    let (header, compressed) = bytes.split_at(12);
    let body = zstd::decode_all(compressed).unwrap();
    let length = u64::from_le_bytes(body[..8].try_into().unwrap()) as usize;
    let mut metadata: serde_json::Value = serde_json::from_slice(&body[8..8 + length]).unwrap();
    edit(&mut metadata);
    let metadata = serde_json::to_vec(&metadata).unwrap();
    let mut rewritten = (metadata.len() as u64).to_le_bytes().to_vec();
    rewritten.extend_from_slice(&metadata);
    rewritten.extend_from_slice(&body[8 + length..]);
    let mut file = header.to_vec();
    file.extend(zstd::encode_all(&rewritten[..], 1).unwrap());
    std::fs::write(path, file).unwrap();
}
