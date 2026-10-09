//! A ledger of the frames a live stack admitted, and the completed-stack pass
//! that replays them with leave-one-out rejection.
//!
//! Live integration can never revisit a sample: the reference frame and the
//! warm-up frames are admitted before rejection has any statistics, so a
//! satellite trail in one of them stays in the mean. The ledger records, for
//! every admitted frame, the file it came from, the calibration masters that
//! applied, and the registration and normalization mapping the live pass
//! chose. [`LiveStacker::reintegrate`] reads each frame again, repeats the
//! same calibration and preparation, and maps it through the recorded
//! transform, so neither star detection nor registration runs a second time.
//! A stack told to [`LiveStacker::retain_frames_for_reintegration`] keeps
//! each frame's registered image from the live pass instead, so the replay
//! prepares nothing.

use crate::CancelSignal;
use crate::batch::PackedFates;
use crate::drizzle::{DrizzleAccumulator, DrizzleFrame, DrizzleOptions, DrizzleResult};
use crate::normalization::{BackgroundSamples, RowCoefficients, RowNormalizer};
use crate::{
    BatchStackOptions, BatchStackPass, BatchStackResult, CalibrationMasters, Error, FitsFrame,
    LinearImage, LiveStacker, ReferenceRegion, RegisteredFrameMapping, Result,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The file a frame was read from, and what was done to it before
/// calibration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct FrameSource {
    path: PathBuf,
    /// The full scale a declared `0:1` range was mapped onto before
    /// calibration, when the source declared one.
    rescale_to: Option<f32>,
    /// Size and modification time when the frame was read, so a replay can
    /// refuse a file that changed since.
    stamp: Option<SourceStamp>,
}

impl FrameSource {
    /// The provenance of a frame opened from disk and not changed since,
    /// apart from a declared-range rescale. `None` for a frame built from
    /// pixels.
    pub(crate) fn of(frame: &FitsFrame) -> Option<Self> {
        let path = frame.source.clone()?;
        let rescale_to = frame
            .bounds
            .filter(|(low, high)| *low == 0.0 && high.is_finite() && *high > 0.0)
            .map(|(_, high)| high as f32);
        Some(Self {
            stamp: SourceStamp::of(&path),
            path,
            rescale_to,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct SourceStamp {
    bytes: u64,
    modified_seconds: u64,
    modified_nanos: u32,
}

impl SourceStamp {
    fn of(path: &Path) -> Option<Self> {
        let metadata = std::fs::metadata(path).ok()?;
        let modified = metadata
            .modified()
            .ok()?
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?;
        Some(Self {
            bytes: metadata.len(),
            modified_seconds: modified.as_secs(),
            modified_nanos: modified.subsec_nanos(),
        })
    }
}

/// Where one set of calibration masters came from.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct CalibrationRecord {
    /// The master paths, when the masters were loaded from files. Masters
    /// supplied in memory can only be replayed while they are still active.
    paths: Option<CalibrationPaths>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct CalibrationPaths {
    bias: Option<PathBuf>,
    dark: Option<PathBuf>,
    flat: Option<PathBuf>,
    dark_exposure_seconds: Option<f64>,
}

impl CalibrationRecord {
    pub(crate) fn from_paths(
        bias: Option<&Path>,
        dark: Option<&Path>,
        flat: Option<&Path>,
        dark_exposure_seconds: Option<f64>,
    ) -> Self {
        Self {
            paths: Some(CalibrationPaths {
                bias: bias.map(Path::to_path_buf),
                dark: dark.map(Path::to_path_buf),
                flat: flat.map(Path::to_path_buf),
                dark_exposure_seconds,
            }),
        }
    }

    fn load(&self) -> Option<Result<CalibrationMasters>> {
        let paths = self.paths.as_ref()?;
        Some(CalibrationMasters::from_fits_paths(
            paths.bias.as_deref(),
            paths.dark.as_deref(),
            paths.flat.as_deref(),
            paths.dark_exposure_seconds,
        ))
    }
}

/// One admitted frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct AdmittedFrame {
    source: Option<FrameSource>,
    /// Index into [`Ledger::calibrations`].
    calibration: u32,
    mapping: RegisteredFrameMapping,
    /// The frame's weight and noise in a weighted stack.
    ///
    /// Skipped by serde: the postcard ledger section keeps the layout of
    /// context format 3, and a weighted context (format 4) stores these in a
    /// section of its own, which [`Ledger::weights`] and
    /// [`Ledger::attach_weights`] move in and out.
    #[serde(skip)]
    weighting: FrameWeightRecord,
}

/// Per-channel noise and weight recorded for one admitted frame. Both are
/// empty in an equally weighted stack.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct FrameWeightRecord {
    pub(crate) noise: Vec<f32>,
    pub(crate) weight: Vec<f32>,
}

/// Every admitted frame, in admission order, and every calibration set the
/// stack used.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Ledger {
    frames: Vec<AdmittedFrame>,
    calibrations: Vec<CalibrationRecord>,
}

impl Ledger {
    /// A ledger holding the reference frame, admitted under the first
    /// calibration set.
    pub(crate) fn new(reference: &LinearImage) -> Self {
        Self {
            frames: vec![AdmittedFrame {
                source: None,
                calibration: 0,
                mapping: RegisteredFrameMapping::identity(reference),
                weighting: FrameWeightRecord::default(),
            }],
            calibrations: vec![CalibrationRecord::default()],
        }
    }

    /// The ledger of a context saved before ledgers existed: it cannot
    /// replay anything, but records later calibration sets and frames.
    pub(crate) fn legacy() -> Self {
        Self {
            frames: Vec::new(),
            calibrations: vec![CalibrationRecord::default()],
        }
    }

    pub(crate) fn set_reference_source(&mut self, source: Option<FrameSource>) {
        if let Some(reference) = self.frames.first_mut() {
            reference.source = source;
        }
    }

    /// Describe the calibration set in use now.
    pub(crate) fn set_current_calibration(&mut self, record: CalibrationRecord) {
        if let Some(current) = self.calibrations.last_mut() {
            *current = record;
        }
    }

    /// Start a new calibration set for the frames admitted from now on.
    pub(crate) fn begin_calibration(&mut self, record: CalibrationRecord) {
        self.calibrations.push(record);
    }

    /// Record the reference frame's weight record: its noise, and weight 1
    /// in every channel.
    pub(crate) fn set_reference_weighting(&mut self, noise: &[f32]) {
        if let Some(reference) = self.frames.first_mut() {
            reference.weighting = FrameWeightRecord {
                noise: noise.to_vec(),
                weight: vec![1.0; noise.len()],
            };
        }
    }

    pub(crate) fn admit(
        &mut self,
        source: Option<FrameSource>,
        mapping: RegisteredFrameMapping,
        weighting: FrameWeightRecord,
    ) {
        let calibration = self.current_calibration();
        self.frames.push(AdmittedFrame {
            source,
            calibration,
            mapping,
            weighting,
        });
    }

    /// Every frame's weight record, in admission order, for a weighted
    /// context.
    pub(crate) fn weights(&self) -> Vec<FrameWeightRecord> {
        self.frames
            .iter()
            .map(|frame| frame.weighting.clone())
            .collect()
    }

    /// Whether any admitted frame carries a polynomial warp.
    pub(crate) fn has_warps(&self) -> bool {
        self.frames
            .iter()
            .any(|frame| frame.mapping.warp().is_some())
    }

    /// Each admitted frame's polynomial warp, which the ledger's own compact
    /// encoding leaves out.
    pub(crate) fn warps(&self) -> Vec<Option<crate::PolynomialWarp>> {
        self.frames
            .iter()
            .map(|frame| frame.mapping.warp().cloned())
            .collect()
    }

    /// Restore the warps a context saved beside the ledger, one per frame.
    pub(crate) fn attach_warps(
        &mut self,
        warps: Vec<Option<crate::PolynomialWarp>>,
    ) -> std::result::Result<(), String> {
        if warps.len() != self.frames.len() {
            return Err(format!(
                "context holds {} frame warps for {} admitted frames",
                warps.len(),
                self.frames.len()
            ));
        }
        for (frame, warp) in self.frames.iter_mut().zip(warps) {
            frame
                .mapping
                .set_warp(warp)
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    /// Restore the weight records a weighted context saved beside the
    /// ledger, checking one record per frame, one value per channel, and
    /// weight 1 for the reference.
    pub(crate) fn attach_weights(
        &mut self,
        records: Vec<FrameWeightRecord>,
        channels: usize,
    ) -> std::result::Result<(), String> {
        if records.len() != self.frames.len() {
            return Err(format!(
                "context has {} frame weight records for {} admitted frames",
                records.len(),
                self.frames.len()
            ));
        }
        for (index, record) in records.iter().enumerate() {
            let valid = record.noise.len() == channels
                && record.weight.len() == channels
                && record
                    .noise
                    .iter()
                    .all(|noise| noise.is_finite() && *noise >= 0.0)
                && record
                    .weight
                    .iter()
                    .all(|weight| weight.is_finite() && *weight > 0.0)
                && (index > 0 || record.weight.iter().all(|weight| *weight == 1.0));
            if !valid {
                return Err(format!(
                    "context has an invalid weight record for admitted frame {}",
                    index + 1
                ));
            }
        }
        for (frame, record) in self.frames.iter_mut().zip(records) {
            frame.weighting = record;
        }
        Ok(())
    }

    fn current_calibration(&self) -> u32 {
        self.calibrations.len().saturating_sub(1) as u32
    }

    pub(crate) fn validate(&self, accepted_frames: u32) -> std::result::Result<(), String> {
        if self.calibrations.is_empty() {
            return Err("admitted-frame ledger has no calibration sets".into());
        }
        if self.frames.len() > accepted_frames as usize {
            return Err(format!(
                "admitted-frame ledger lists {} frames but the stack admitted {accepted_frames}",
                self.frames.len()
            ));
        }
        for frame in &self.frames {
            if frame.calibration as usize >= self.calibrations.len() {
                return Err("admitted-frame ledger names an unknown calibration set".into());
            }
            frame
                .mapping
                .validate()
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}

/// Frames averaged into the reference a spatially normalized replay matches
/// backgrounds to; PixInsight's WBPP integrates its twenty best for local
/// normalization.
const NORMALIZATION_REFERENCE_FRAMES: usize = 20;

/// The integrated background reference and each frame's normalization
/// fitted against it, filled on first use.
struct Renormalizer {
    /// The samples the background fit reads.
    layout: BackgroundSamples,
    /// The integrated reference at those samples, all the fit reads of it.
    reference: Vec<f32>,
    maps: std::sync::Mutex<Vec<Option<crate::NormalizationMap>>>,
}

impl Renormalizer {
    /// Frame `index`'s background offsets refitted against the integrated
    /// reference on `interpolated`, its unnormalized registered image,
    /// keeping its recorded gain; fitted once and then reused, so every pass
    /// sees the same samples.
    fn map_for(
        &self,
        admitted: &AdmittedFrame,
        index: usize,
        interpolated: &LinearImage,
    ) -> Result<crate::NormalizationMap> {
        if let Some(map) = self.fitted(index) {
            return Ok(map);
        }
        self.map_from_samples(admitted, index, &self.layout.of(interpolated)?)
    }

    /// [`Self::map_for`] from the frame's background samples alone.
    fn map_from_samples(
        &self,
        admitted: &AdmittedFrame,
        index: usize,
        samples: &[f32],
    ) -> Result<crate::NormalizationMap> {
        if let Some(map) = self.fitted(index) {
            return Ok(map);
        }
        let map = admitted.mapping.normalization().refit_background_samples(
            &self.layout,
            &self.reference,
            samples,
        )?;
        self.maps
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())[index] = Some(map.clone());
        Ok(map)
    }

    fn fitted(&self, index: usize) -> Option<crate::NormalizationMap> {
        self.maps
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())[index]
            .clone()
    }
}

/// Each sample's mean from its sum and count, `NaN` where nothing counted.
fn mean(sum: Vec<f32>, count: Vec<u16>) -> Vec<f32> {
    sum.into_par_iter()
        .zip(count.into_par_iter())
        .map(|(sum, count)| {
            if count == 0 {
                f32::NAN
            } else {
                sum / f32::from(count)
            }
        })
        .collect()
}

/// One integrated frame waiting for the drizzle thread.
struct DrizzleJob {
    index: usize,
    image: LinearImage,
    layout: Option<crate::BayerLayout>,
    normalization: crate::NormalizationMap,
    fates: PackedFates,
    weight: Vec<f32>,
}

/// Frames prepared ahead of the one a reintegration pass is integrating.
const REPLAY_LOOKAHEAD: usize = 2;

/// The most frames a banded reintegration reads whole at once, to fit their
/// normalization before the bands.
const WHOLE_FRAME_READS: usize = 4;

/// The most frames a banded reintegration drizzles together.
const DRIZZLE_FRAMES: usize = 4;

/// [`REPLAY_LOOKAHEAD`], or fewer in a Rayon pool of few threads.
fn replay_lookahead() -> usize {
    (rayon::current_num_threads() / 6).clamp(1, REPLAY_LOOKAHEAD)
}

/// Calibration masters for replay. The current set is the stacker's own; an
/// earlier set is loaded from its paths when first needed and kept with the
/// one before it, so frames prepared ahead across a set boundary do not load
/// either set twice.
///
/// A pool thread loads a set without the lock held, because loading runs
/// Rayon work, and a pool thread waiting on that work can take up another
/// frame's preparation, which would then wait on a lock its own thread
/// holds. So that frames prepared at once in a pool do not each load the
/// same set, the thread that starts them loads it first with
/// [`MastersCache::warm`].
#[derive(Default)]
struct MastersCache {
    loaded: std::sync::Mutex<Vec<(u32, std::sync::Arc<CalibrationMasters>)>>,
}

/// Masters for one frame: borrowed from the stacker or shared from the cache.
enum ReplayMasters<'a> {
    Current(&'a CalibrationMasters),
    Loaded(std::sync::Arc<CalibrationMasters>),
}

impl std::ops::Deref for ReplayMasters<'_> {
    type Target = CalibrationMasters;

    fn deref(&self) -> &CalibrationMasters {
        match self {
            Self::Current(masters) => masters,
            Self::Loaded(masters) => masters,
        }
    }
}

/// Whether a kept frame of a stack with `options`, debayered from `cfa`,
/// holds its registered samples as they are, rather than through the
/// identity normalization, which turns a negative zero positive: so that
/// normalizing it as it is read gives, to the bit, what the whole-frame
/// passes give.
///
/// Those normalize a Bayer-drizzled frame's photosites as resampled. A frame
/// without a Bayer layout in such a stack is normalized as resampled with
/// its recorded map, but through the identity first when its background is
/// fitted again, as every frame of any other stack is.
pub(crate) fn keeps_zeros(options: &crate::StackOptions, cfa: Option<crate::BayerLayout>) -> bool {
    options.cfa_integration == crate::CfaIntegration::BayerDrizzle
        && (cfa.is_some()
            || !matches!(
                options.normalization,
                crate::NormalizationMode::LocalBackground { .. }
            ))
}

/// Each admitted frame's prepared image, registered but not normalized,
/// kept in a scratch file so replay's passes and its integrated reference
/// read it back instead of reading, calibrating, demosaicing and resampling
/// the source again.
///
/// A replay makes one for itself and fills it as it first prepares each
/// frame. A stack told to [`LiveStacker::retain_frames_for_reintegration`]
/// keeps one beside the live pass, which writes each frame's file while it
/// has the registered image in hand, so the replay prepares nothing. A frame
/// whose file could not be written is prepared again when it is next needed.
///
/// Under local background normalization each frame's file has a smaller one
/// beside it, the samples the background fit reads (see
/// [`BackgroundSamples`]), so a replay refits the frame from those rather
/// than reading it whole.
pub(crate) struct FrameCache {
    files: CacheFiles,
    slots: Vec<std::sync::Mutex<CacheSlot>>,
}

/// The directory a [`FrameCache`] keeps its files in, which frames being
/// prepared on other threads write into.
pub(crate) struct CacheFiles {
    directory: tempfile::TempDir,
    /// Names the files of frames written before their admission is decided.
    staged: std::sync::atomic::AtomicU64,
    /// The samples a local background fit reads, kept beside each frame
    /// when the stack is normalized that way.
    background: Option<BackgroundSamples>,
}

#[derive(Clone, Copy, PartialEq)]
enum CacheSlot {
    Empty,
    /// The frame's image is kept, and its background samples when
    /// `samples` says so.
    Stored {
        samples: bool,
    },
    Unavailable,
}

/// A frame's registered image, and its background samples, written while
/// the frame waits to learn whether it is admitted. Dropping it, as a
/// rejection, an error or a cancelled run does, deletes the files.
pub(crate) struct StagedFrame {
    path: Option<PathBuf>,
    samples: Option<PathBuf>,
}

impl StagedFrame {
    /// Move the files to frame `index`'s, reporting what is kept.
    fn commit(mut self, files: &CacheFiles, index: usize) -> CacheSlot {
        let Some(path) = self.path.take() else {
            return CacheSlot::Empty;
        };
        if std::fs::rename(&path, files.path(index)).is_err() {
            let _ = std::fs::remove_file(path);
            return CacheSlot::Empty;
        }
        let samples = self.samples.take().is_some_and(|samples| {
            let kept = std::fs::rename(&samples, files.samples_path(index)).is_ok();
            if !kept {
                let _ = std::fs::remove_file(samples);
            }
            kept
        });
        CacheSlot::Stored { samples }
    }
}

impl Drop for StagedFrame {
    fn drop(&mut self) {
        for path in [self.path.take(), self.samples.take()]
            .into_iter()
            .flatten()
        {
            let _ = std::fs::remove_file(path);
        }
    }
}

impl CacheFiles {
    fn path(&self, index: usize) -> PathBuf {
        self.directory.path().join(format!("frame-{index}.f32"))
    }

    fn samples_path(&self, index: usize) -> PathBuf {
        self.directory.path().join(format!("frame-{index}.samples"))
    }

    /// The samples of `fitted`, a frame's registered, unnormalized and
    /// interpolated image, that a background fit reads, when the stack
    /// keeps them.
    pub(crate) fn background_samples(&self, fitted: &LinearImage) -> Option<Vec<f32>> {
        self.background.as_ref()?.of(fitted).ok()
    }

    /// Write a frame's registered, unnormalized image before its admission
    /// is decided, with its background `samples` if any, or `None` when the
    /// image could not be written. `keep_zeros` writes the image's samples
    /// as they are (see [`keeps_zeros`]).
    pub(crate) fn stage(
        &self,
        image: &LinearImage,
        samples: Option<Vec<f32>>,
        keep_zeros: bool,
    ) -> Option<StagedFrame> {
        let number = self
            .staged
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let staged = |extension: &str| {
            self.directory
                .path()
                .join(format!("staged-{number}.{extension}"))
        };
        // Owning the paths from here on deletes a partial file on failure.
        let mut frame = StagedFrame {
            path: Some(staged("f32")),
            samples: None,
        };
        FrameCache::write_image(frame.path.as_deref()?, image, keep_zeros).ok()?;
        if let Some(samples) = samples {
            let path = staged("samples");
            if FrameCache::write_samples(&path, &samples).is_ok() {
                frame.samples = Some(path);
            } else {
                let _ = std::fs::remove_file(path);
            }
        }
        Some(frame)
    }
}

/// The part of a [`FrameCache`] the integration side of a live pass changes:
/// one slot for each frame it admits.
pub(crate) struct CacheAdmissions<'a> {
    files: &'a CacheFiles,
    slots: &'a mut Vec<std::sync::Mutex<CacheSlot>>,
}

impl CacheAdmissions<'_> {
    /// Give the frame just admitted its slot, keeping the file its
    /// preparation staged, if any.
    pub(crate) fn admit(&mut self, staged: Option<StagedFrame>) {
        let index = self.slots.len();
        let slot = staged.map_or(CacheSlot::Empty, |staged| staged.commit(self.files, index));
        self.slots.push(std::sync::Mutex::new(slot));
    }
}

impl FrameCache {
    /// A cache for `frames` frames in a new directory, keeping the samples
    /// of `background` beside each.
    fn new(
        directory: Option<&Path>,
        frames: usize,
        background: Option<BackgroundSamples>,
    ) -> std::io::Result<Self> {
        // The owning process ID in the name lets a later run recognise, and
        // remove, a directory left behind by a process that was killed.
        let pid = std::process::id();
        let directory = match directory {
            Some(directory) => tempfile::Builder::new()
                .prefix(&format!(".seiza-reintegrate-{pid}-"))
                .tempdir_in(directory),
            None => tempfile::Builder::new()
                .prefix(&format!("seiza-reintegrate-{pid}-"))
                .tempdir(),
        }?;
        Ok(Self {
            files: CacheFiles {
                directory,
                staged: std::sync::atomic::AtomicU64::new(0),
                background,
            },
            slots: (0..frames)
                .map(|_| std::sync::Mutex::new(CacheSlot::Empty))
                .collect(),
        })
    }

    fn slot(&self, index: usize) -> CacheSlot {
        *self.slots[index]
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// Whether frame `index`'s file is kept.
    fn stored(&self, index: usize) -> bool {
        matches!(self.slot(index), CacheSlot::Stored { .. })
    }

    /// Whether frame `index`'s background samples are kept.
    fn sampled(&self, index: usize) -> bool {
        self.slot(index) == CacheSlot::Stored { samples: true }
    }

    /// Frame `index`'s background samples, read back, or `None` when they
    /// are not kept or cannot be read.
    fn samples(&self, index: usize) -> Option<Vec<f32>> {
        let layout = self.files.background.as_ref()?;
        if !self.sampled(index) {
            return None;
        }
        Self::read_samples(&self.files.samples_path(index), layout.len())
    }

    /// Where preparation threads stage frames.
    pub(crate) fn files(&self) -> &CacheFiles {
        &self.files
    }

    /// The files, for preparation threads to stage frames in, and the slots,
    /// for the integration thread to admit them to.
    pub(crate) fn split(&mut self) -> (&CacheFiles, CacheAdmissions<'_>) {
        let files = &self.files;
        (
            files,
            CacheAdmissions {
                files,
                slots: &mut self.slots,
            },
        )
    }

    fn path(&self, index: usize) -> PathBuf {
        self.files.path(index)
    }

    /// The cached image of frame `index`, read into `buffer` when it is
    /// big enough, and made with `prepare` the first time.
    ///
    /// The slot is not held while preparing, because in a pool preparing
    /// runs Rayon work, and a pool thread waiting on that work can take up
    /// the same frame's next pass, which would then wait on a slot its own
    /// thread holds. Two passes that reach the same frame at once, which
    /// only a stack of one frame does, both prepare it, and the first keeps
    /// it.
    fn get_or_prepare(
        &self,
        index: usize,
        buffer: Vec<f32>,
        prepare: impl FnOnce() -> Result<LinearImage>,
    ) -> Result<LinearImage> {
        {
            let mut slot = self.slots[index]
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if let CacheSlot::Stored { .. } = *slot {
                if let Some(image) = Self::read(&self.path(index), buffer) {
                    return Ok(image);
                }
                *slot = CacheSlot::Unavailable;
            }
        }
        let image = prepare()?;
        let mut slot = self.slots[index]
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if *slot == CacheSlot::Empty {
            *slot = if Self::write(&self.path(index), &image).is_ok() {
                let samples = self.files.background.as_ref().is_some_and(|layout| {
                    let path = self.files.samples_path(index);
                    let written = layout
                        .of(&image)
                        .is_ok_and(|samples| Self::write_samples(&path, &samples).is_ok());
                    if !written {
                        let _ = std::fs::remove_file(path);
                    }
                    written
                });
                CacheSlot::Stored { samples }
            } else {
                let _ = std::fs::remove_file(self.path(index));
                CacheSlot::Unavailable
            };
        }
        Ok(image)
    }

    /// Keep frame `index`'s `image`, written as its samples are when
    /// `keep_zeros`, and its background `samples`, unless they are kept
    /// already; a frame whose image cannot be written is marked so.
    fn keep(&self, index: usize, image: &LinearImage, samples: Option<&[f32]>, keep_zeros: bool) {
        let mut slot = self.slots[index]
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let stored = match *slot {
            CacheSlot::Stored { samples: true } | CacheSlot::Unavailable => return,
            CacheSlot::Stored { samples: false } => true,
            CacheSlot::Empty => Self::write_image(&self.path(index), image, keep_zeros).is_ok(),
        };
        if !stored {
            let _ = std::fs::remove_file(self.path(index));
            *slot = CacheSlot::Unavailable;
            return;
        }
        let samples = samples.is_some_and(|samples| {
            let path = self.files.samples_path(index);
            let written = Self::write_samples(&path, samples).is_ok();
            if !written {
                let _ = std::fs::remove_file(path);
            }
            written
        });
        *slot = CacheSlot::Stored { samples };
    }

    /// Write `image` as replay keeps it: registered, and passed through the
    /// identity normalization, which leaves every sample as it is except a
    /// negative zero, which `mul_add(1, 0)` turns positive. The live pass
    /// writes its registered image before any normalization, so making the
    /// same change here keeps its files bit-identical to a replay's own.
    fn write(path: &Path, image: &LinearImage) -> std::io::Result<()> {
        Self::write_image(path, image, false)
    }

    /// [`Self::write`], or with every sample as it is when `keep_zeros`.
    fn write_image(path: &Path, image: &LinearImage, keep_zeros: bool) -> std::io::Result<()> {
        Self::write_values(
            path,
            &[image.width, image.height, image.channels],
            &image.data,
            keep_zeros,
        )
    }

    /// Write a frame's background samples, changed as [`Self::write`]
    /// changes an image, since the fit reads them through the identity
    /// normalization.
    fn write_samples(path: &Path, samples: &[f32]) -> std::io::Result<()> {
        Self::write_values(path, &[samples.len()], samples, false)
    }

    /// Write `header`, then `values`, with negative zeros made positive
    /// unless `keep_zeros`.
    fn write_values(
        path: &Path,
        header: &[usize],
        values: &[f32],
        keep_zeros: bool,
    ) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(path)?);
        for &dimension in header {
            file.write_all(&(dimension as u64).to_le_bytes())?;
        }
        let mut bytes = vec![0_u8; 1 << 20];
        for chunk in values.chunks(bytes.len() / 4) {
            for (&sample, out) in chunk.iter().zip(bytes.chunks_exact_mut(4)) {
                let sample = if sample == 0.0 && !keep_zeros {
                    0.0_f32
                } else {
                    sample
                };
                out.copy_from_slice(&sample.to_le_bytes());
            }
            file.write_all(&bytes[..chunk.len() * 4])?;
        }
        file.into_inner().map_err(|error| error.into_error())?;
        Ok(())
    }

    /// Read an image [`Self::write`] wrote, a megabyte at a time, so reading
    /// holds no second copy of it, into `buffer`: one of the same size is
    /// reused as it is, saving the cost of mapping fresh memory.
    fn read(path: &Path, buffer: Vec<f32>) -> Option<LinearImage> {
        use std::io::Read;
        let mut file = std::fs::File::open(path).ok()?;
        let mut header = [0_u8; 24];
        file.read_exact(&mut header).ok()?;
        let dimension = |at: usize| -> Option<usize> {
            u64::from_le_bytes(header[at..at + 8].try_into().ok()?)
                .try_into()
                .ok()
        };
        let (width, height, channels) = (dimension(0)?, dimension(8)?, dimension(16)?);
        let samples = width.checked_mul(height)?.checked_mul(channels)?;
        let length = u64::try_from(samples.checked_mul(4)?.checked_add(header.len())?).ok()?;
        if file.metadata().ok()?.len() != length {
            return None;
        }
        let mut data = buffer;
        data.resize(samples, 0.0);
        let mut bytes = vec![0_u8; 1 << 20];
        for chunk in data.chunks_mut(bytes.len() / 4) {
            let raw = &mut bytes[..chunk.len() * 4];
            file.read_exact(raw).ok()?;
            for (sample, raw) in chunk.iter_mut().zip(raw.chunks_exact(4)) {
                *sample = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
            }
        }
        LinearImage::new(width, height, channels, data).ok()
    }

    /// Read `count` samples [`Self::write_samples`] wrote.
    fn read_samples(path: &Path, count: usize) -> Option<Vec<f32>> {
        use std::io::Read;
        let mut file = std::fs::File::open(path).ok()?;
        let mut header = [0_u8; 8];
        file.read_exact(&mut header).ok()?;
        let length = u64::try_from(count.checked_mul(4)?.checked_add(header.len())?).ok()?;
        if u64::from_le_bytes(header) != count as u64 || file.metadata().ok()?.len() != length {
            return None;
        }
        let mut bytes = vec![0_u8; count * 4];
        file.read_exact(&mut bytes).ok()?;
        Some(
            bytes
                .chunks_exact(4)
                .map(|raw| f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
                .collect(),
        )
    }

    /// Fill `band`, whole rows of an image of `shape`, with rows `top..`
    /// of frame `index`'s file, normalizing them with `normalizer` as each
    /// megabyte arrives, while it is still in the processor's cache.
    fn read_band(
        &self,
        index: usize,
        shape: (usize, usize, usize),
        top: usize,
        band: &mut [f32],
        normalizer: &RowNormalizer<'_>,
    ) -> Result<()> {
        use std::io::{Read, Seek};
        let path = self.path(index);
        let failed = |reason: String| {
            Error::Stack(format!(
                "cannot read admitted frame {} back from {}: {reason}",
                index + 1,
                path.display()
            ))
        };
        let (width, height, channels) = shape;
        let row_samples = width * channels;
        let row_bytes = row_samples * 4;
        let mut file = std::fs::File::open(&path).map_err(|error| failed(error.to_string()))?;
        let mut header = [0_u8; 24];
        file.read_exact(&mut header)
            .map_err(|error| failed(error.to_string()))?;
        let dimension =
            |at: usize| u64::from_le_bytes(header[at..at + 8].try_into().expect("eight bytes"));
        let length = file
            .metadata()
            .map_err(|error| failed(error.to_string()))?
            .len();
        if [dimension(0), dimension(8), dimension(16)]
            != [width as u64, height as u64, channels as u64]
            || length != (header.len() + height * row_bytes) as u64
            || !band.len().is_multiple_of(row_samples)
            || top + band.len() / row_samples > height
        {
            return Err(failed("it does not match the stack's shape".into()));
        }
        file.seek(std::io::SeekFrom::Start(
            (header.len() + top * row_bytes) as u64,
        ))
        .map_err(|error| failed(error.to_string()))?;
        let rows_per_read = ((1 << 20) / row_bytes).max(1);
        let mut bytes = vec![0_u8; rows_per_read * row_bytes];
        let mut coefficients = RowCoefficients::default();
        for (read, rows) in band.chunks_mut(rows_per_read * row_samples).enumerate() {
            let raw = &mut bytes[..rows.len() * 4];
            file.read_exact(raw)
                .map_err(|error| failed(error.to_string()))?;
            for (sample, raw) in rows.iter_mut().zip(raw.chunks_exact(4)) {
                *sample = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
            }
            normalizer.apply(&mut coefficients, rows, top + read * rows_per_read);
        }
        Ok(())
    }
}

/// What one reintegration shares between its steps.
struct Replay<'a> {
    stacker: &'a LiveStacker,
    options: &'a BatchStackOptions,
    masters: &'a MastersCache,
    renormalizer: Option<&'a Renormalizer>,
}

/// Each frame's band of rows from its scratch file, normalized as it is
/// read.
struct CacheBands<'a> {
    cache: &'a FrameCache,
    shape: (usize, usize, usize),
    /// The admitted frame each band source index reads.
    frames: Vec<usize>,
    normalizers: Vec<RowNormalizer<'a>>,
}

impl crate::batch::BandSource for CacheBands<'_> {
    fn read(&self, index: usize, top: usize, band: &mut [f32]) -> Result<()> {
        self.cache.read_band(
            self.frames[index],
            self.shape,
            top,
            band,
            &self.normalizers[index],
        )
    }
}

/// An admitted frame's source file, for a helper to read ahead.
fn source_files(admitted: &AdmittedFrame) -> Vec<PathBuf> {
    admitted
        .source
        .iter()
        .map(|source| source.path.clone())
        .collect()
}

impl Replay<'_> {
    /// Each frame's weights, one per channel.
    fn frame_weights(&self, index: usize) -> Vec<f32> {
        match &self.options.frame_weights {
            Some(weights) => weights[index].clone(),
            None => vec![1.0; self.stacker.reference.channels],
        }
    }

    /// The map that normalizes frame `index`: refitted against the
    /// integrated reference, or as the live pass recorded it.
    fn normalization(&self, index: usize) -> Result<crate::NormalizationMap> {
        match self.renormalizer {
            Some(renormalizer) => renormalizer
                .maps
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())[index]
                .clone()
                .ok_or_else(|| Error::Stack("a frame's normalization was never fitted".into())),
            None => Ok(self.stacker.ledger.frames[index]
                .mapping
                .normalization()
                .clone()),
        }
    }

    /// Integrate every frame from its scratch file, a band of rows at a
    /// time, or `None` when some frame could not be kept there, which the
    /// whole-frame passes handle by preparing it again.
    ///
    /// Each frame is read once in full first if its normalization must be
    /// fitted again or it is not in the cache yet, then once more in bands.
    /// A drizzle follows, frame by frame, from the fates the integration
    /// recorded.
    #[allow(clippy::type_complexity)]
    fn bands(
        &self,
        cache: &FrameCache,
        drizzle: Option<&DrizzleOptions>,
        progress: &mut impl FnMut(BatchStackPass, usize, usize),
    ) -> Result<Option<(BatchStackResult, Option<DrizzleResult>)>> {
        let stacker = self.stacker;
        let frames = &stacker.ledger.frames;
        let count = frames.len();
        let stored = |index: usize| cache.stored(index);
        for frame in frames {
            stacker.source_unchanged(frame)?;
        }
        // Fit each frame's normalization, and keep any frame the cache does
        // not hold yet. A frame whose background samples are kept is fitted
        // from those alone; any other is read in one piece, mostly on one
        // core, so several are read at once, as many as half the band
        // memory holds, from the replay's lookahead to four; their buffers
        // are reused, saving the cost of mapping fresh memory.
        let pending = (0..count)
            .filter(|&index| self.renormalizer.is_some() || !stored(index))
            .collect::<Vec<_>>();
        let frame_bytes = stacker.reference.sample_count() * std::mem::size_of::<f32>();
        let lookahead = (self.options.band_memory_bytes / 2 / frame_bytes.max(1))
            .clamp(replay_lookahead(), WHOLE_FRAME_READS);
        let buffers = std::sync::Mutex::new(Vec::<Vec<f32>>::new());
        crate::tasks::ahead(|ahead| {
            std::thread::scope(|scope| {
                let prepare = |index: usize| {
                    let admitted = &frames[index];
                    if let Some(renormalizer) = self.renormalizer
                        && let Some(samples) = cache.samples(index)
                    {
                        renormalizer.map_from_samples(admitted, index, &samples)?;
                        return Ok(());
                    }
                    let masters = self.masters.get(stacker, admitted.calibration)?;
                    // A Bayer-drizzled frame keeps its photosites, but its
                    // background is fitted on its interpolated image.
                    if stacker.options.cfa_integration == crate::CfaIntegration::BayerDrizzle {
                        stacker.source_unchanged(admitted)?;
                        let (kept, interpolated, keep_zeros) =
                            stacker.kept_frame(admitted, &masters)?;
                        let samples = cache.files.background_samples(&interpolated);
                        if let (Some(renormalizer), Some(samples)) = (self.renormalizer, &samples) {
                            renormalizer.map_from_samples(admitted, index, samples)?;
                        }
                        cache.keep(index, &kept, samples.as_deref(), keep_zeros);
                        return Ok(());
                    }
                    let buffer = buffers
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner())
                        .pop()
                        .unwrap_or_default();
                    let image =
                        stacker.prepared_into(index, admitted, &masters, Some(cache), buffer)?;
                    if let Some(renormalizer) = self.renormalizer {
                        renormalizer.map_for(admitted, index, &image)?;
                    }
                    buffers
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner())
                        .push(image.data);
                    Ok::<_, Error>(())
                };
                let mut pending = pending.into_iter();
                let mut in_flight = std::collections::VecDeque::new();
                loop {
                    crate::batch::check_cancelled(self.options)?;
                    while in_flight.len() < lookahead
                        && let Some(index) = pending.next()
                    {
                        progress(BatchStackPass::Estimate, index, count);
                        // A frame fitted from its background samples reads
                        // only their small file, so nothing is read ahead.
                        let fitted_from_samples =
                            self.renormalizer.is_some() && cache.sampled(index);
                        if !fitted_from_samples {
                            self.masters.warm(stacker, frames[index].calibration);
                        }
                        in_flight.push_back(ahead.spawn(
                            scope,
                            move || {
                                if fitted_from_samples {
                                    Vec::new()
                                } else {
                                    stacker.replay_files(index, &frames[index], Some(cache))
                                }
                            },
                            move || prepare(index),
                        ));
                    }
                    let Some(handle) = in_flight.pop_front() else {
                        return Ok::<_, Error>(());
                    };
                    ahead.join(handle)?;
                }
            })
        })?;
        drop(buffers);
        if !(0..count).all(stored) {
            return Ok(None);
        }

        let maps = (0..count)
            .map(|index| self.normalization(index))
            .collect::<Result<Vec<_>>>()?;
        let reference = &stacker.reference;
        let source = CacheBands {
            cache,
            shape: (reference.width, reference.height, reference.channels),
            frames: (0..count).collect(),
            normalizers: maps
                .iter()
                .map(|map| RowNormalizer::new(map, 0, reference.width))
                .collect(),
        };
        // The frame indices are spread over the bands, and the drizzle's
        // frames, so that they follow the work done.
        let mut reported = 0;
        let mut report = |unit: usize, units: usize| {
            let until = ((unit + 1) * count).div_ceil(units).min(count);
            while reported < until {
                progress(BatchStackPass::Integrate, reported, count);
                reported += 1;
            }
        };
        let drizzle_frames = if drizzle.is_some() { count } else { 0 };
        let mut total_units = 0;
        let (result, fates) = crate::batch::integrate_bands(
            source.shape,
            count,
            self.options,
            &source,
            drizzle.is_some(),
            &mut |band, bands| {
                total_units = bands + drizzle_frames;
                report(band, total_units);
            },
        )?;
        drop(source);
        let Some((drizzle, mut fates)) = drizzle.zip(fates) else {
            return Ok(Some((result, None)));
        };

        // Drizzle the frames a few at a time, every output tile taking each
        // of them in turn, so the tile's sums stay in the processor's cache
        // between them, while the next frames' calibrated sources are read
        // on threads of their own.
        let mut accumulator = DrizzleAccumulator::new(
            *drizzle,
            reference.width,
            reference.height,
            reference.channels,
        )?;
        let bands = total_units - count;
        let together =
            DRIZZLE_FRAMES.min((self.options.band_memory_bytes / 2 / frame_bytes.max(1)).max(1));
        crate::tasks::ahead(|ahead| {
            std::thread::scope(|scope| {
                let read = |index: usize| {
                    let admitted = &frames[index];
                    let masters = self.masters.get(stacker, admitted.calibration)?;
                    stacker.read_calibrated(admitted, &masters)
                };
                let mut next = 0;
                let mut in_flight = std::collections::VecDeque::new();
                let mut first = 0;
                while first < count {
                    crate::batch::check_cancelled(self.options)?;
                    let last = (first + together).min(count);
                    // These frames, and as many after them as the lookahead
                    // allows, which are read while these are drizzled.
                    while next < count.min(last + lookahead) {
                        let frame = next;
                        self.masters.warm(stacker, frames[frame].calibration);
                        in_flight.push_back(ahead.spawn(
                            scope,
                            move || source_files(&frames[frame]),
                            move || read(frame),
                        ));
                        next += 1;
                    }
                    let sources = (first..last)
                        .map(|index| {
                            let source = ahead.join(
                                in_flight
                                    .pop_front()
                                    .expect("the frame being drizzled was read ahead"),
                            );
                            report(bands + index, total_units);
                            source
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let fates = (first..last)
                        .map(|index| std::mem::take(&mut fates[index]))
                        .collect::<Vec<_>>();
                    let weights = (first..last)
                        .map(|index| self.frame_weights(index))
                        .collect::<Vec<_>>();
                    let batch = (first..last)
                        .zip(&sources)
                        .zip(fates.iter().zip(&weights))
                        .map(|((index, (image, layout)), (fates, weight))| DrizzleFrame {
                            image,
                            layout: *layout,
                            mapping: &frames[index].mapping,
                            normalization: &maps[index],
                            fates,
                            weight,
                        })
                        .collect::<Vec<_>>();
                    accumulator.add_frames(&batch)?;
                    first = last;
                }
                Ok::<_, Error>(())
            })
        })?;
        crate::batch::check_cancelled(self.options)?;
        Ok(Some((result, Some(accumulator.finish()?))))
    }

    /// Integrate every frame with the whole-frame passes, reading each one
    /// once per pass: from the cache when it holds the frame, and otherwise
    /// from its source. A drizzle runs on a thread of its own, a frame behind
    /// the final pass.
    fn frames(
        &self,
        cache: Option<&FrameCache>,
        drizzle: Option<&DrizzleOptions>,
        progress: &mut impl FnMut(BatchStackPass, usize, usize),
    ) -> Result<(BatchStackResult, Option<DrizzleResult>)> {
        let stacker = self.stacker;
        let ledger = &stacker.ledger;
        let count = ledger.frames.len();
        let options = self.options;
        let masters = self.masters;
        let renormalizer = self.renormalizer;
        let accumulator = drizzle
            .map(|drizzle| {
                DrizzleAccumulator::new(
                    *drizzle,
                    stacker.reference.width,
                    stacker.reference.height,
                    stacker.reference.channels,
                )
            })
            .transpose()?;
        // Frames are requested in a fixed order: every frame for each pass in
        // turn. Preparing the next few on their own threads while the batch
        // integrates the current one overlaps reading, debayering and
        // resampling with the rejection arithmetic, which leaves cores idle
        // when they alternate. Each in-flight frame holds about one prepared
        // image, so the lookahead stays small.
        let order = [
            BatchStackPass::Estimate,
            BatchStackPass::Refine,
            BatchStackPass::Integrate,
        ]
        .into_iter()
        .flat_map(|pass| (0..count).map(move |index| (pass, index)))
        .collect::<Vec<_>>();
        let lookahead = replay_lookahead();
        // The calibrated source of the frame the final pass is integrating,
        // read alongside its registered image for the drizzle.
        let drizzle_source = std::cell::RefCell::new(None);
        let (result, accumulator) = crate::tasks::ahead(|ahead| {
            std::thread::scope(|scope| {
                let prepare = |pass: BatchStackPass, index: usize| {
                    let frame = &ledger.frames[index];
                    let masters = masters.get(stacker, frame.calibration)?;
                    let image =
                        stacker.replay_frame(frame, &masters, index, renormalizer, cache)?;
                    let source = if drizzle.is_some() && pass == BatchStackPass::Integrate {
                        Some(stacker.read_calibrated(frame, &masters)?)
                    } else {
                        None
                    };
                    Ok::<_, Error>((image, source))
                };
                let mut in_flight = std::collections::VecDeque::new();
                let mut next = 0;
                let load = |pass, index| {
                    progress(pass, index, count);
                    while next < order.len() && in_flight.len() < lookahead {
                        let (frame_pass, frame_index) = order[next];
                        let frame = &ledger.frames[frame_index];
                        masters.warm(stacker, frame.calibration);
                        let files = move || {
                            let mut files = stacker.replay_files(frame_index, frame, cache);
                            if drizzle.is_some() && frame_pass == BatchStackPass::Integrate {
                                files.extend(source_files(frame));
                            }
                            files
                        };
                        in_flight.push_back((
                            next,
                            ahead.spawn(scope, files, move || prepare(frame_pass, frame_index)),
                        ));
                        next += 1;
                    }
                    let (image, source) = match in_flight.pop_front() {
                        Some((position, handle)) if order[position] == (pass, index) => {
                            ahead.join(handle)?
                        }
                        // A request out of the expected order is served directly.
                        other => {
                            if let Some(entry) = other {
                                in_flight.push_front(entry);
                            }
                            prepare(pass, index)?
                        }
                    };
                    *drizzle_source.borrow_mut() = source.map(|source| (index, source));
                    Ok(image)
                };
                // The drizzle runs on a thread of its own, a frame behind the
                // integration, so dropping one frame overlaps integrating the
                // next. The channel holds one frame, which bounds the memory.
                let (jobs, worker) = match accumulator {
                    Some(mut accumulator) => {
                        let (sender, receiver) = std::sync::mpsc::sync_channel::<DrizzleJob>(1);
                        let worker = scope.spawn(move || {
                            for job in receiver {
                                // A cancelled integration stops at its next frame;
                                // there is no use drizzling the ones queued here.
                                if options
                                    .cancel
                                    .as_ref()
                                    .is_some_and(CancelSignal::is_cancelled)
                                {
                                    return Err(Error::Cancelled);
                                }
                                // In the caller's pool when it runs in one.
                                accumulator = ahead.run(move || {
                                    accumulator.add(&DrizzleFrame {
                                        image: &job.image,
                                        layout: job.layout,
                                        mapping: &ledger.frames[job.index].mapping,
                                        normalization: &job.normalization,
                                        fates: &job.fates,
                                        weight: &job.weight,
                                    })?;
                                    Ok::<_, Error>(accumulator)
                                })?;
                            }
                            Ok::<_, Error>(accumulator)
                        });
                        (Some(sender), Some(worker))
                    }
                    None => (None, None),
                };
                let mut observe = |index: usize, fates: PackedFates| -> Result<()> {
                    let Some(jobs) = &jobs else {
                        return Ok(());
                    };
                    let frame = &ledger.frames[index];
                    let (image, layout) = match drizzle_source.borrow_mut().take() {
                        Some((source_index, source)) if source_index == index => source,
                        _ => stacker
                            .read_calibrated(frame, &*masters.get(stacker, frame.calibration)?)?,
                    };
                    let normalization = match renormalizer {
                        Some(_) => self.normalization(index).map_err(|_| {
                            Error::Stack(
                                "a frame reached the drizzle without its normalization".into(),
                            )
                        })?,
                        None => frame.mapping.normalization().clone(),
                    };
                    // A closed channel means the drizzle failed; its error is
                    // reported when the worker is joined.
                    ahead
                        .send(
                            jobs,
                            DrizzleJob {
                                index,
                                image,
                                layout,
                                normalization,
                                fates,
                                weight: self.frame_weights(index),
                            },
                        )
                        .map_err(|_| Error::Stack("the drizzle stopped early".into()))
                };
                let integrated = crate::batch::integrate_registered_frames_observed(
                    count,
                    options,
                    load,
                    drizzle.is_some().then_some(&mut observe as _),
                );
                drop(jobs);
                let drizzled = worker.map(|worker| ahead.join_helper(worker)).transpose()?;
                Ok::<_, Error>((integrated?, drizzled))
            })
        })?;
        let drizzled = accumulator.map(DrizzleAccumulator::finish).transpose()?;
        Ok((result, drizzled))
    }
}

impl MastersCache {
    fn get<'a>(&self, stacker: &'a LiveStacker, calibration: u32) -> Result<ReplayMasters<'a>> {
        if calibration == stacker.ledger.current_calibration() {
            return Ok(ReplayMasters::Current(&stacker.calibration));
        }
        let find = |loaded: &[(u32, std::sync::Arc<CalibrationMasters>)]| {
            loaded
                .iter()
                .find(|(set, _)| *set == calibration)
                .map(|(_, masters)| ReplayMasters::Loaded(std::sync::Arc::clone(masters)))
        };
        // Only a thread outside any pool loads with the lock held, which
        // spares another frame on the same set a second load.
        let held = {
            let loaded = self.lock();
            if let Some(masters) = find(&loaded) {
                return Ok(masters);
            }
            rayon::current_thread_index().is_none().then_some(loaded)
        };
        let masters = std::sync::Arc::new(
            stacker.ledger.calibrations[calibration as usize]
                .load()
                .ok_or_else(|| {
                    Error::Stack("calibration masters are no longer available".into())
                })??,
        );
        let mut loaded = match held {
            Some(loaded) => loaded,
            None => {
                let loaded = self.lock();
                // Another frame may have loaded the same set meanwhile.
                if let Some(masters) = find(&loaded) {
                    return Ok(masters);
                }
                loaded
            }
        };
        if loaded.len() == 2 {
            loaded.remove(0);
        }
        loaded.push((calibration, std::sync::Arc::clone(&masters)));
        Ok(ReplayMasters::Loaded(masters))
    }

    /// From a pool thread, load the set `calibration` names, if it is not
    /// loaded, before frames that use it are prepared in the pool. A set that
    /// cannot be loaded is left for each frame to report.
    fn warm(&self, stacker: &LiveStacker, calibration: u32) {
        if rayon::current_thread_index().is_some() {
            let _ = self.get(stacker, calibration);
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<(u32, std::sync::Arc<CalibrationMasters>)>> {
        self.loaded
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

impl LiveStacker {
    /// Keep each frame this stack admits from now on in a scratch file,
    /// registered but not normalized, so [`Self::reintegrate`] and
    /// [`Self::reintegrate_drizzled`] read it back instead of reading,
    /// calibrating, demosaicing and resampling the frame's source again.
    /// The result is bit-identical either way.
    ///
    /// Call this before pushing frames to a stack that will be reintegrated.
    /// The threads preparing frames write each file while they hold the
    /// registered image, and a frame that is turned away has its file
    /// deleted. The reference is written now. Frames admitted before the
    /// call, such as those of a stack reopened from a saved context, are
    /// prepared from their sources when replayed, as they are without it.
    /// A stack that integrates Bayer photosites keeps each frame's
    /// photosites, carried onto the reference grid; a reopened one prepares
    /// its reference again too.
    ///
    /// The files take about four bytes per output sample per admitted frame,
    /// 313 MB for each frame of a 26 MP colour sensor, and a sixteenth as
    /// much again under [`crate::NormalizationMode::LocalBackground`], for
    /// the samples its background fit reads: the space a reintegration's own
    /// scratch files take, held from now until the stacker is dropped rather
    /// than only while it replays. They go in a new
    /// directory inside `scratch_directory`, or the system temporary
    /// directory, which a replay then uses in place of
    /// [`BatchStackOptions::scratch_directory`]. Dropping the stacker removes
    /// it. A frame whose file cannot be written is prepared again when it is
    /// replayed.
    ///
    /// Calling this again keeps the directory already made. A stack that
    /// cannot be reintegrated (see [`Self::reintegration_unavailable`])
    /// keeps nothing.
    pub fn retain_frames_for_reintegration(
        &mut self,
        scratch_directory: Option<&Path>,
    ) -> Result<()> {
        if self.frame_cache.is_some() || self.reintegration_unavailable().is_some() {
            return Ok(());
        }
        let cache = FrameCache::new(
            scratch_directory,
            self.ledger.frames.len(),
            self.background_samples(),
        )
        .map_err(|error| {
            Error::Stack(format!(
                "cannot make a directory for frames kept for reintegration: {error}"
            ))
        })?;
        // The reference was prepared from its source already; carry it
        // through its identity mapping, as a replay would after preparing it
        // again. A Bayer-drizzled stack keeps its photosites, so needs the
        // layout it was demosaiced from; without it, the replay prepares the
        // reference again.
        let reference = &self.ledger.frames[0];
        if self.options.cfa_integration != crate::CfaIntegration::BayerDrizzle {
            cache.get_or_prepare(0, Vec::new(), || {
                self.register_unnormalized(reference, &self.reference, self.options.interpolation)
            })?;
        } else if let Some(layout) = self.reference_cfa {
            let (kept, interpolated) =
                self.kept_photosites(reference, &self.reference, Some(layout))?;
            cache.keep(
                0,
                &kept,
                cache.files.background_samples(&interpolated).as_deref(),
                keeps_zeros(&self.options, Some(layout)),
            );
        }
        self.frame_cache = Some(cache);
        Ok(())
    }

    /// The samples a local background fit reads from a registered frame, for
    /// a stack normalized that way.
    fn background_samples(&self) -> Option<BackgroundSamples> {
        match self.options.normalization {
            crate::NormalizationMode::LocalBackground { tile_size } => {
                Some(BackgroundSamples::for_image(&self.reference, tile_size))
            }
            _ => None,
        }
    }

    /// Why [`Self::reintegrate`] cannot replay this stack, or `None` when it
    /// can.
    ///
    /// Replaying needs every admitted frame's source file and the
    /// calibration masters it was calibrated with. That rules out stacks
    /// built from prepared pixels, frames pushed as pixels, frames admitted
    /// under masters supplied in memory that have since been replaced, and
    /// contexts saved before the admitted-frame ledger existed.
    pub fn reintegration_unavailable(&self) -> Option<String> {
        let ledger = &self.ledger;
        if ledger.frames.len() != self.accepted_frames as usize {
            return Some(
                "this stack was resumed from a context saved before admitted frames were \
                 recorded; rebuild it to reintegrate"
                    .into(),
            );
        }
        let current = ledger.current_calibration();
        for (index, frame) in ledger.frames.iter().enumerate() {
            if frame.source.is_none() {
                return Some(format!(
                    "admitted frame {} was supplied as pixels rather than read from a file",
                    index + 1
                ));
            }
            if frame.calibration != current
                && ledger.calibrations[frame.calibration as usize]
                    .paths
                    .is_none()
            {
                return Some(format!(
                    "admitted frame {} was calibrated with masters supplied in memory that \
                     have since been replaced",
                    index + 1
                ));
            }
        }
        None
    }

    /// Integrate every admitted frame again with three-pass, leave-one-out
    /// rejection, so a transient in the reference or a warm-up frame is
    /// rejected like any other.
    ///
    /// Each frame is read from the file it came from and prepared exactly as
    /// the live pass prepared it: the same declared-range rescale,
    /// calibration masters, cosmetic filter, debayering, and the recorded
    /// registration and normalization mapping. The prepared image waits in
    /// [`BatchStackOptions::scratch_directory`] for the passes, and a frame
    /// [`Self::retain_frames_for_reintegration`] kept is not read at all.
    /// Star detection and registration do not run again, and no admission
    /// decision is revisited: every admitted frame takes part. A source file
    /// that changed since it was stacked is refused.
    ///
    /// The passes read the scratch files a band of rows at a time and run
    /// all three on one band of every frame before the next (see
    /// [`BatchStackOptions::band_memory_bytes`]), so each file is read once
    /// for them. A stack normalized with
    /// [`crate::NormalizationMode::LocalBackground`] first fits each frame's
    /// background offsets again from the samples kept beside its file, about
    /// a sixteenth of it, or, for a frame kept without them, from the file
    /// read whole. The result is exactly that
    /// of [`crate::integrate_registered_frames`], whose rejection this is. Memory
    /// is 16 bytes per output sample (20 for a weighted stack) plus the
    /// bands. Where a frame cannot be kept in a scratch file, every frame is
    /// instead read whole on each of the three passes, with the memory
    /// [`crate::integrate_registered_frames`] describes.
    ///
    /// The live stack is left as it was. `progress` receives the pass, a
    /// zero-based frame index and the frame count: [`BatchStackPass::Estimate`]
    /// for each frame as its normalization is fitted or it is read whole, then
    /// [`BatchStackPass::Integrate`] for the frame indices in turn, spread
    /// over the bands so that they follow the work done. When frames are
    /// read whole for each pass, it hears each pass and frame before the
    /// read.
    ///
    /// A weighted stack replays each frame with the weight the live pass
    /// recorded, so noise is not measured again. It fills
    /// [`BatchStackOptions::frame_weights`] itself and refuses options that
    /// already carry weights.
    pub fn reintegrate(
        &self,
        options: &BatchStackOptions,
        progress: impl FnMut(BatchStackPass, usize, usize),
    ) -> Result<BatchStackResult> {
        Ok(self.replay(options, None, progress)?.0)
    }

    /// [`Self::reintegrate`], also drizzling every admitted frame onto a
    /// grid `drizzle.scale` times the reference's.
    ///
    /// The drizzle follows the integration, so each frame is read once more,
    /// from its source file, and calibrated but not debayered or resampled.
    /// A source pixel whose registered sample the integration rejected is
    /// left out, and every pixel is normalized and weighted as the
    /// integration did its registered sample. A Bayer frame drizzles its
    /// photosites, each into its own colour.
    ///
    /// The drizzle needs eight bytes per output sample on top of
    /// reintegration's memory, 2.5 GB for a 26 MP colour frame at twice the
    /// scale, and the integration keeps every sample's fate for it, a quarter
    /// of a byte per sample per frame. When the frames were integrated band
    /// by band, up to four frames' calibrated sources, within half of
    /// [`BatchStackOptions::band_memory_bytes`], are drizzled together, each
    /// output tile taking them in turn. Returns the reintegrated stack and
    /// the drizzled one.
    pub fn reintegrate_drizzled(
        &self,
        options: &BatchStackOptions,
        drizzle: &DrizzleOptions,
        progress: impl FnMut(BatchStackPass, usize, usize),
    ) -> Result<(BatchStackResult, DrizzleResult)> {
        drizzle.validate()?;
        let (result, drizzled) = self.replay(options, Some(drizzle), progress)?;
        Ok((result, drizzled.expect("a drizzle was requested")))
    }

    fn replay(
        &self,
        options: &BatchStackOptions,
        drizzle: Option<&DrizzleOptions>,
        progress: impl FnMut(BatchStackPass, usize, usize),
    ) -> Result<(BatchStackResult, Option<DrizzleResult>)> {
        self.replay_with(options, drizzle, progress, true)
    }

    /// [`Self::replay`], reading cached frames in bands when `banded`, and
    /// otherwise a whole frame for each pass.
    pub(crate) fn replay_with(
        &self,
        options: &BatchStackOptions,
        drizzle: Option<&DrizzleOptions>,
        mut progress: impl FnMut(BatchStackPass, usize, usize),
        banded: bool,
    ) -> Result<(BatchStackResult, Option<DrizzleResult>)> {
        if let Some(reason) = self.reintegration_unavailable() {
            return Err(Error::Stack(reason));
        }
        let ledger = &self.ledger;
        let weighted_options;
        let options = if self.options.weighting.is_equal() {
            options
        } else {
            if options.frame_weights.is_some() {
                return Err(Error::Stack(
                    "a weighted stack replays with its recorded frame weights; \
                     do not supply frame_weights"
                        .into(),
                ));
            }
            weighted_options = BatchStackOptions {
                frame_weights: Some(
                    ledger
                        .frames
                        .iter()
                        .map(|frame| frame.weighting.weight.clone())
                        .collect(),
                ),
                ..options.clone()
            };
            &weighted_options
        };
        let count = ledger.frames.len();
        let masters = MastersCache::default();
        // The frames the live pass kept, when it was asked to; otherwise a
        // cache of this replay's own.
        let own_cache;
        let cache = if let Some(retained) = self
            .frame_cache
            .as_ref()
            .filter(|retained| retained.slots.len() == count)
        {
            Some(retained)
        } else {
            own_cache = FrameCache::new(
                options.scratch_directory.as_deref(),
                count,
                self.background_samples(),
            )
            .ok();
            own_cache.as_ref()
        };
        let renormalizer = match self.background_samples() {
            Some(layout) => Some(Renormalizer {
                reference: self.normalization_reference(&layout, &masters, cache, options)?,
                layout,
                maps: std::sync::Mutex::new(vec![None; count]),
            }),
            None => None,
        };
        let replay = Replay {
            stacker: self,
            options,
            masters: &masters,
            renormalizer: renormalizer.as_ref(),
        };
        let banded_result = match cache.filter(|_| banded) {
            Some(cache) => replay.bands(cache, drizzle, &mut progress)?,
            None => None,
        };
        let (mut result, drizzled) = match banded_result {
            Some(result) => result,
            None => replay.frames(cache, drizzle, &mut progress)?,
        };
        result.snapshot.rejected_frames = self.rejected_frames;
        self.fill_photosite_gaps(&mut result.snapshot.image.data, &result.snapshot.coverage);
        Ok((result, drizzled))
    }

    /// The reference a spatially normalized replay matches backgrounds to:
    /// the mean of the [`NORMALIZATION_REFERENCE_FRAMES`] best admitted
    /// frames, each registered and scaled by its recorded gain with its mean
    /// recorded offset. The best are those with the highest recorded weight
    /// in a weighted stack, and otherwise the lowest photometric gain, the
    /// most transparent frames; averaging frames chosen without regard to
    /// quality copied a hazy frame's cloud into every frame.
    ///
    /// Local normalization against a single frame copies that frame's
    /// large-scale background, its gradient, vignetting and banding, into
    /// every frame of the stack. In the mean of frames taken at different
    /// times and orientations those patterns largely cancel, as in the
    /// integrated reference PixInsight's WBPP builds for local normalization.
    ///
    /// The background fit reads only the reference's samples in `layout`,
    /// so only those are returned. When the cache keeps every one of the
    /// best frames' samples, only those are read and averaged; otherwise the
    /// frames are averaged whole.
    fn normalization_reference(
        &self,
        layout: &BackgroundSamples,
        masters: &MastersCache,
        cache: Option<&FrameCache>,
        options: &BatchStackOptions,
    ) -> Result<Vec<f32>> {
        let frames = &self.ledger.frames;
        // Higher is better: the recorded weight, or else the inverse of the
        // photometric gain, which rises as haze dims a frame's stars.
        let quality = |admitted: &AdmittedFrame| {
            let weight = &admitted.weighting.weight;
            if weight.is_empty() {
                1.0 / admitted.mapping.normalization().mean_gain()
            } else {
                weight.iter().sum::<f32>() / weight.len() as f32
            }
        };
        let mut ranked = (0..frames.len()).collect::<Vec<_>>();
        ranked.sort_by(|&left, &right| quality(&frames[right]).total_cmp(&quality(&frames[left])));
        ranked.truncate(NORMALIZATION_REFERENCE_FRAMES);
        if let Some(cache) = cache.filter(|cache| ranked.iter().all(|&index| cache.sampled(index)))
            && let Some(reference) = self.sampled_reference(layout, cache, &ranked)?
        {
            return Ok(reference);
        }
        layout.of(&self.normalization_reference_image(&ranked, masters, cache, options)?)
    }

    /// [`Self::normalization_reference`] at the samples of `layout` alone,
    /// from the samples the cache keeps of each of the `ranked` frames, or
    /// `None` when one cannot be read. Each sample sees the arithmetic it
    /// sees in a whole image, frame by frame in the same order.
    fn sampled_reference(
        &self,
        layout: &BackgroundSamples,
        cache: &FrameCache,
        ranked: &[usize],
    ) -> Result<Option<Vec<f32>>> {
        let frames = &self.ledger.frames;
        for &index in ranked {
            self.source_unchanged(&frames[index])?;
        }
        let mut sum = vec![0.0_f32; layout.len()];
        let mut count = vec![0_u16; layout.len()];
        for &index in ranked {
            let Some(mut samples) = cache.samples(index) else {
                return Ok(None);
            };
            frames[index]
                .mapping
                .normalization()
                .global_equivalent()
                .apply_global_samples(layout, &mut samples)?;
            sum.par_iter_mut()
                .zip(count.par_iter_mut())
                .zip(samples.par_iter())
                .for_each(|((sum, count), &value)| {
                    if value.is_finite() {
                        *sum += value;
                        *count += 1;
                    }
                });
        }
        Ok(Some(mean(sum, count)))
    }

    /// [`Self::normalization_reference`] as a whole image, from the
    /// `ranked` frames read whole or in bands.
    fn normalization_reference_image(
        &self,
        ranked: &[usize],
        masters: &MastersCache,
        cache: Option<&FrameCache>,
        options: &BatchStackOptions,
    ) -> Result<LinearImage> {
        let frames = &self.ledger.frames;
        let ranked = ranked.to_vec();
        let samples = self.reference.sample_count();
        // A Bayer-drizzled stack keeps photosites, not the interpolated
        // frames the reference averages.
        let cache =
            cache.filter(|_| self.options.cfa_integration != crate::CfaIntegration::BayerDrizzle);
        let mut sum = vec![0.0_f32; samples];
        let mut count = vec![0_u16; samples];
        // Frames kept in scratch files are read a band at a time and summed
        // in the same ranked order, without a whole frame in memory.
        if let Some(cache) = cache.filter(|cache| ranked.iter().all(|&index| cache.stored(index))) {
            for &index in &ranked {
                self.source_unchanged(&frames[index])?;
            }
            let maps = ranked
                .iter()
                .map(|&index| frames[index].mapping.normalization().global_equivalent())
                .collect::<Vec<_>>();
            let reference = &self.reference;
            let shape = (reference.width, reference.height, reference.channels);
            let source = CacheBands {
                cache,
                shape,
                frames: ranked.clone(),
                normalizers: maps
                    .iter()
                    .map(|map| RowNormalizer::new(map, 0, reference.width))
                    .collect(),
            };
            const CHUNK: usize = 1 << 14;
            crate::batch::for_each_band(
                shape,
                ranked.len(),
                crate::batch::band_rows(shape, ranked.len(), options.band_memory_bytes),
                &source,
                options.cancel.as_ref(),
                |_, top, frames| {
                    let start = top * reference.width * reference.channels;
                    let end = start + frames[0].len();
                    sum[start..end]
                        .par_chunks_mut(CHUNK)
                        .zip(count[start..end].par_chunks_mut(CHUNK))
                        .enumerate()
                        .for_each(|(chunk, (sum, count))| {
                            let (at, len) = (chunk * CHUNK, sum.len());
                            for frame in frames {
                                for ((sum, count), &value) in sum
                                    .iter_mut()
                                    .zip(count.iter_mut())
                                    .zip(&frame[at..at + len])
                                {
                                    if value.is_finite() {
                                        *sum += value;
                                        *count += 1;
                                    }
                                }
                            }
                        });
                    Ok(())
                },
            )?;
            return LinearImage::new(
                reference.width,
                reference.height,
                reference.channels,
                mean(sum, count),
            );
        }
        let prepare = |index: usize| {
            let admitted = &frames[index];
            let masters = masters.get(self, admitted.calibration)?;
            let mut image = self.prepared(index, admitted, &masters, cache)?;
            // The frame's recorded gain and mean offset: its background stays
            // its own, while its stars match the reference's.
            admitted
                .mapping
                .normalization()
                .global_equivalent()
                .apply_global(&mut image)?;
            Ok::<_, Error>(image)
        };
        // The next frames are read, or prepared, on threads of their own
        // while this one is summed; the sum keeps the ranked order.
        let lookahead = replay_lookahead();
        crate::tasks::ahead(|ahead| {
            std::thread::scope(|scope| {
                let mut ranked = ranked.into_iter();
                let mut in_flight = std::collections::VecDeque::new();
                loop {
                    while in_flight.len() < lookahead
                        && let Some(index) = ranked.next()
                    {
                        masters.warm(self, frames[index].calibration);
                        in_flight.push_back(ahead.spawn(
                            scope,
                            move || self.replay_files(index, &frames[index], cache),
                            move || prepare(index),
                        ));
                    }
                    let Some(handle) = in_flight.pop_front() else {
                        return Ok::<_, Error>(());
                    };
                    let image = ahead.join(handle)?;
                    sum.par_iter_mut()
                        .zip(count.par_iter_mut())
                        .zip(image.data.par_iter())
                        .for_each(|((sum, count), &value)| {
                            if value.is_finite() {
                                *sum += value;
                                *count += 1;
                            }
                        });
                }
            })
        })?;
        let image = LinearImage::new(
            self.reference.width,
            self.reference.height,
            self.reference.channels,
            mean(sum, count),
        )?;
        Ok(image)
    }

    /// An admitted frame of a Bayer-drizzled stack, prepared from its source
    /// as the cache keeps it: see [`Self::kept_photosites`].
    fn kept_frame(
        &self,
        admitted: &AdmittedFrame,
        masters: &CalibrationMasters,
    ) -> Result<(LinearImage, LinearImage, bool)> {
        let (frame, cfa) = self.read_admitted(admitted, masters)?;
        let (kept, interpolated) = self.kept_photosites(admitted, &frame.image, cfa)?;
        Ok((kept, interpolated, keeps_zeros(&self.options, cfa)))
    }

    /// An admitted frame of a Bayer-drizzled stack as the cache keeps it,
    /// from `prepared`, its debayered source, and `cfa`, the layout it was
    /// debayered from: its photosites, or without a layout its interpolated
    /// image, carried onto the reference grid through its recorded
    /// registration and not normalized, with negative zeros as
    /// [`keeps_zeros`] says. Also the interpolated image the frame's
    /// background fit reads, through the identity normalization.
    fn kept_photosites(
        &self,
        admitted: &AdmittedFrame,
        prepared: &LinearImage,
        cfa: Option<crate::BayerLayout>,
    ) -> Result<(LinearImage, LinearImage)> {
        let interpolated =
            self.register_unnormalized(admitted, prepared, self.options.interpolation)?;
        let region = self.full_region(admitted);
        let kept = match cfa {
            Some(layout) => admitted.mapping.resample_region(
                prepared,
                region,
                crate::registration::Sampling::NearestPhotosite(layout),
            )?,
            None if keeps_zeros(&self.options, cfa) => admitted.mapping.resample_region(
                prepared,
                region,
                self.options.interpolation.into(),
            )?,
            None => interpolated.clone(),
        };
        Ok((kept, interpolated))
    }

    /// Read, calibrate and prepare an admitted frame's source and carry it
    /// onto the reference grid through its recorded registration, without
    /// any normalization.
    fn replay_frame_unnormalized(
        &self,
        admitted: &AdmittedFrame,
        masters: &CalibrationMasters,
        interpolation: crate::Interpolation,
    ) -> Result<LinearImage> {
        let (frame, _) = self.read_admitted(admitted, masters)?;
        self.register_unnormalized(admitted, &frame.image, interpolation)
    }

    /// Carry an admitted frame's prepared source onto the reference grid
    /// through its recorded registration, applying the identity in place of
    /// its normalization.
    fn register_unnormalized(
        &self,
        admitted: &AdmittedFrame,
        prepared: &LinearImage,
        interpolation: crate::Interpolation,
    ) -> Result<LinearImage> {
        let identity = admitted
            .mapping
            .with_normalization(crate::NormalizationMap::identity(&self.reference))?;
        identity.extract_region_with(prepared, self.full_region(admitted), interpolation)
    }

    fn full_region(&self, admitted: &AdmittedFrame) -> ReferenceRegion {
        ReferenceRegion {
            x: 0,
            y: 0,
            width: admitted.mapping.reference_width(),
            height: admitted.mapping.reference_height(),
        }
    }

    /// Read, calibrate and filter an admitted frame's source, refusing a
    /// file that changed since it was stacked.
    fn read_source(
        &self,
        admitted: &AdmittedFrame,
        masters: &CalibrationMasters,
    ) -> Result<FitsFrame> {
        let source = admitted
            .source
            .as_ref()
            .ok_or_else(|| Error::Stack("an admitted frame has no source file".into()))?;
        let changed = || {
            Error::Stack(format!(
                "{} changed since it was stacked; rebuild the stack",
                source.path.display()
            ))
        };
        if source.stamp.is_some() && SourceStamp::of(&source.path) != source.stamp {
            return Err(changed());
        }
        let mut frame = FitsFrame::open(&source.path)?;
        if let Some(full_scale) = source.rescale_to {
            frame.rescale_declared_unit_bounds(full_scale);
        }
        masters.validate_light_frame(&frame)?;
        masters.apply(&mut frame.image, frame.exposure_seconds, frame.bayer)?;
        if let Some(filter) = &self.options.cosmetic {
            crate::cosmetic::suppress_impulses(&mut frame.image, frame.bayer, filter)?;
        }
        if source.stamp.is_some() && SourceStamp::of(&source.path) != source.stamp {
            return Err(changed());
        }
        Ok(frame)
    }

    /// Read, calibrate, filter and debayer an admitted frame's source,
    /// refusing a file that changed since it was stacked.
    fn read_admitted(
        &self,
        admitted: &AdmittedFrame,
        masters: &CalibrationMasters,
    ) -> Result<(FitsFrame, Option<crate::BayerLayout>)> {
        self.read_source(admitted, masters)?
            .into_prepared_with_layout(self.options.demosaic)
    }

    /// An admitted frame's calibrated and filtered source pixels, not
    /// debayered, with the Bayer layout of a colour sensor's mosaic.
    fn read_calibrated(
        &self,
        admitted: &AdmittedFrame,
        masters: &CalibrationMasters,
    ) -> Result<(LinearImage, Option<crate::BayerLayout>)> {
        let frame = self.read_source(admitted, masters)?;
        Ok((frame.image, frame.bayer))
    }

    /// An admitted frame registered through its recorded mapping, not yet
    /// normalized: from the cache when there is one, and made (then cached)
    /// otherwise. A source file that changed since it was stacked is
    /// refused either way.
    fn prepared(
        &self,
        index: usize,
        admitted: &AdmittedFrame,
        masters: &CalibrationMasters,
        cache: Option<&FrameCache>,
    ) -> Result<LinearImage> {
        self.prepared_into(index, admitted, masters, cache, Vec::new())
    }

    /// [`Self::prepared`], reading a cached frame into `buffer`.
    fn prepared_into(
        &self,
        index: usize,
        admitted: &AdmittedFrame,
        masters: &CalibrationMasters,
        cache: Option<&FrameCache>,
        buffer: Vec<f32>,
    ) -> Result<LinearImage> {
        let make = || self.replay_frame_unnormalized(admitted, masters, self.options.interpolation);
        match cache {
            Some(cache) => {
                self.source_unchanged(admitted)?;
                cache.get_or_prepare(index, buffer, make)
            }
            None => make(),
        }
    }

    /// The file preparing admitted frame `index` reads, for a helper to read
    /// ahead: its kept image when the cache holds one the preparation uses,
    /// and its source otherwise.
    fn replay_files(
        &self,
        index: usize,
        admitted: &AdmittedFrame,
        cache: Option<&FrameCache>,
    ) -> Vec<PathBuf> {
        match cache.filter(|cache| {
            self.options.cfa_integration != crate::CfaIntegration::BayerDrizzle
                && cache.stored(index)
        }) {
            Some(cache) => vec![cache.path(index)],
            None => source_files(admitted),
        }
    }

    fn source_unchanged(&self, admitted: &AdmittedFrame) -> Result<()> {
        let source = admitted
            .source
            .as_ref()
            .ok_or_else(|| Error::Stack("an admitted frame has no source file".into()))?;
        if source.stamp.is_some() && SourceStamp::of(&source.path) != source.stamp {
            return Err(Error::Stack(format!(
                "{} changed since it was stacked; rebuild the stack",
                source.path.display()
            )));
        }
        Ok(())
    }

    fn replay_frame(
        &self,
        admitted: &AdmittedFrame,
        masters: &CalibrationMasters,
        index: usize,
        renormalizer: Option<&Renormalizer>,
        cache: Option<&FrameCache>,
    ) -> Result<LinearImage> {
        if self.options.cfa_integration == crate::CfaIntegration::BayerDrizzle {
            if let Some(renormalizer) = renormalizer {
                return self.replay_frame_renormalized(admitted, masters, renormalizer, index);
            }
            let (frame, cfa) = self.read_admitted(admitted, masters)?;
            let region = self.full_region(admitted);
            return match cfa {
                Some(layout) => {
                    admitted
                        .mapping
                        .extract_region_photosites(&frame.image, region, layout)
                }
                None => admitted.mapping.extract_region_with(
                    &frame.image,
                    region,
                    self.options.interpolation,
                ),
            };
        }
        let mut image = self.prepared(index, admitted, masters, cache)?;
        match renormalizer {
            Some(renormalizer) => renormalizer
                .map_for(admitted, index, &image)?
                .apply(&mut image)?,
            None => admitted.mapping.normalization().apply(&mut image)?,
        }
        Ok(image)
    }

    /// [`Self::replay_frame`] with the frame's background offsets fitted
    /// again against the integrated reference, keeping its recorded gain. The fit runs on the
    /// interpolated frame, which samples every channel evenly, and is cached
    /// so every pass sees the same samples.
    fn replay_frame_renormalized(
        &self,
        admitted: &AdmittedFrame,
        masters: &CalibrationMasters,
        renormalizer: &Renormalizer,
        index: usize,
    ) -> Result<LinearImage> {
        let (frame, cfa) = self.read_admitted(admitted, masters)?;
        let region = self.full_region(admitted);
        let identity = admitted
            .mapping
            .with_normalization(crate::NormalizationMap::identity(&self.reference))?;
        let mut interpolated =
            identity.extract_region_with(&frame.image, region, self.options.interpolation)?;
        let map = renormalizer.map_for(admitted, index, &interpolated)?;
        match cfa.filter(|_| self.options.cfa_integration == crate::CfaIntegration::BayerDrizzle) {
            Some(layout) => admitted
                .mapping
                .with_normalization(map)?
                .extract_region_photosites(&frame.image, region, layout),
            None => {
                map.apply(&mut interpolated)?;
                Ok(interpolated)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(noise: f32, weight: f32) -> FrameWeightRecord {
        FrameWeightRecord {
            noise: vec![noise],
            weight: vec![weight],
        }
    }

    #[test]
    fn kept_frames_read_back_as_replay_makes_them_and_staged_files_go_with_their_frames() {
        let bits = |image: &LinearImage| {
            image
                .data
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        let directory = tempfile::tempdir().unwrap();
        // A 2 by 2 image is one tile, every pixel sampled.
        let layout = BackgroundSamples::new(2, 2, 1, 16);
        let mut cache = FrameCache::new(Some(directory.path()), 0, Some(layout.clone())).unwrap();
        let kept_in = cache.files.directory.path().to_path_buf();
        let files = || {
            let mut names = std::fs::read_dir(&kept_in)
                .unwrap()
                .map(|entry| entry.unwrap().file_name().into_string().unwrap())
                .collect::<Vec<_>>();
            names.sort();
            names
        };
        let payload = f32::from_bits(0x7fc0_1234);
        let image = LinearImage::new(2, 2, 1, vec![-0.0, 0.0, payload, -1.5e-40]).unwrap();
        let (staging, mut admissions) = cache.split();
        let stage = || {
            staging
                .stage(&image, staging.background_samples(&image), false)
                .unwrap()
        };
        let turned_away = stage();
        let admitted = stage();
        drop(turned_away);
        admissions.admit(Some(admitted));
        admissions.admit(None);
        admissions.admit(None);
        assert_eq!(files(), ["frame-0.f32", "frame-0.samples"]);
        // The identity normalization a replay applies turns only a negative
        // zero positive; the file holds what it would have made.
        let mut normalized = image.clone();
        crate::NormalizationMap::identity(&image)
            .apply(&mut normalized)
            .unwrap();
        assert_eq!(normalized.data[0].to_bits(), 0);
        let kept = cache
            .get_or_prepare(0, Vec::new(), || panic!("frame 0 was kept"))
            .unwrap();
        assert_eq!(bits(&kept), bits(&normalized));
        // The background samples kept are those of the kept image.
        let sample_bits = |samples: Vec<f32>| {
            samples
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            sample_bits(cache.samples(0).unwrap()),
            sample_bits(layout.of(&normalized).unwrap())
        );
        assert!(cache.samples(1).is_none());
        // A frame admitted without a file is prepared once, then read back.
        cache
            .get_or_prepare(1, Vec::new(), || Ok(normalized.clone()))
            .unwrap();
        let read = cache
            .get_or_prepare(1, vec![7.0; 4], || {
                panic!("frame 1 was kept when first prepared")
            })
            .unwrap();
        assert_eq!(bits(&read), bits(&normalized));
        assert_eq!(
            sample_bits(cache.samples(1).unwrap()),
            sample_bits(cache.samples(0).unwrap())
        );
        assert_eq!(
            files(),
            [
                "frame-0.f32",
                "frame-0.samples",
                "frame-1.f32",
                "frame-1.samples"
            ]
        );
        // A Bayer-drizzled frame's photosites are kept as they are, negative
        // zero and all, without samples when it has none to give.
        cache.keep(2, &image, None, true);
        let kept = cache
            .get_or_prepare(2, Vec::new(), || panic!("frame 2 was kept"))
            .unwrap();
        assert_eq!(bits(&kept), bits(&image));
        assert!(cache.samples(2).is_none());
        assert_eq!(files().len(), 5);
    }

    /// A Bayer-drizzled frame's photosites are normalized as resampled in
    /// the whole-frame passes, so they are kept as they are: read back in
    /// bands and normalized, a negative zero meets a negative offset as it
    /// does there. Turned positive, as other frames are kept, it would not.
    #[test]
    fn kept_photosites_normalize_as_the_whole_frame_passes_do() {
        let (width, height) = (4, 3);
        let data = (0..width * height * 3)
            .map(|index| match index % 5 {
                0 => -0.0,
                1 => f32::NAN,
                2 => 0.0,
                _ => index as f32 - 20.5,
            })
            .collect();
        let image = LinearImage::new(width, height, 3, data).unwrap();
        let map: crate::NormalizationMap = serde_json::from_value(serde_json::json!({
            "schema_version": 1,
            "width": width,
            "height": height,
            "channels": 3,
            "tile_size": width,
            "columns": 1,
            "rows": 1,
            "gains": [1.0, 2.0, 0.5],
            "offsets": [-0.0, 1.0, -0.0],
        }))
        .unwrap();
        let mut expected = image.clone();
        map.apply(&mut expected).unwrap();
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        let directory = tempfile::tempdir().unwrap();
        let mut cache = FrameCache::new(Some(directory.path()), 0, None).unwrap();
        let (_, mut admissions) = cache.split();
        admissions.admit(None);
        admissions.admit(None);
        cache.keep(0, &image, None, true);
        cache.keep(1, &image, None, false);
        let read = |index: usize| {
            let mut band = vec![0.0; image.sample_count()];
            cache
                .read_band(
                    index,
                    (width, height, 3),
                    0,
                    &mut band,
                    &RowNormalizer::new(&map, 0, width),
                )
                .unwrap();
            band
        };
        assert_eq!(bits(&read(0)), bits(&expected.data));
        assert_ne!(bits(&read(1)), bits(&expected.data));
    }

    #[test]
    fn weight_records_must_cover_every_frame_and_weigh_the_reference_as_one() {
        let reference = LinearImage::new(4, 4, 1, vec![1.0; 16]).unwrap();
        let mut ledger = Ledger::new(&reference);
        ledger.admit(
            None,
            RegisteredFrameMapping::identity(&reference),
            record(4.0, 0.5),
        );
        let valid = vec![record(2.0, 1.0), record(4.0, 0.25)];
        for invalid in [
            vec![record(2.0, 1.0)],
            vec![record(2.0, 0.5), record(4.0, 0.25)],
            vec![record(2.0, 1.0), record(4.0, 0.0)],
            vec![record(2.0, 1.0), record(f32::NAN, 0.25)],
            vec![
                record(2.0, 1.0),
                FrameWeightRecord {
                    noise: vec![4.0; 3],
                    weight: vec![0.25; 3],
                },
            ],
        ] {
            assert!(ledger.clone().attach_weights(invalid, 1).is_err());
        }
        ledger.attach_weights(valid.clone(), 1).unwrap();
        assert_eq!(ledger.weights(), valid);
        // The postcard ledger layout does not carry the records.
        let bytes = postcard::to_stdvec(&ledger).unwrap();
        let restored: Ledger = postcard::from_bytes(&bytes).unwrap();
        assert!(
            restored
                .weights()
                .iter()
                .all(|record| record.weight.is_empty())
        );
    }
}
