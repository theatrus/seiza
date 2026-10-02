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

use crate::batch::SampleFate;
use crate::drizzle::{DrizzleAccumulator, DrizzleFrame, DrizzleOptions, DrizzleResult};
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
    reference: LinearImage,
    maps: std::sync::Mutex<Vec<Option<crate::NormalizationMap>>>,
}

impl Renormalizer {
    /// Frame `index`'s background offsets refitted against the integrated
    /// reference on `interpolated`, its unnormalized registered image,
    /// keeping its recorded gain; fitted once and then reused, so every pass
    /// sees the same samples.
    fn map_for(
        &self,
        stacker: &LiveStacker,
        admitted: &AdmittedFrame,
        index: usize,
        interpolated: &LinearImage,
    ) -> Result<crate::NormalizationMap> {
        if let Some(map) = self
            .maps
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())[index]
            .clone()
        {
            return Ok(map);
        }
        let crate::NormalizationMode::LocalBackground { tile_size } = stacker.options.normalization
        else {
            unreachable!("only local background normalization is refitted");
        };
        let map = admitted.mapping.normalization().refit_background(
            &self.reference,
            interpolated,
            tile_size,
        )?;
        self.maps
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())[index] = Some(map.clone());
        Ok(map)
    }
}

/// One integrated frame waiting for the drizzle thread.
struct DrizzleJob {
    index: usize,
    image: LinearImage,
    layout: Option<crate::BayerLayout>,
    normalization: crate::NormalizationMap,
    fates: Vec<SampleFate>,
    weight: Vec<f32>,
}

/// Frames prepared ahead of the one a reintegration pass is integrating.
const REPLAY_LOOKAHEAD: usize = 2;

/// Calibration masters for replay. The current set is the stacker's own; an
/// earlier set is loaded from its paths when first needed and kept with the
/// one before it, so frames prepared ahead across a set boundary do not load
/// either set twice.
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

/// Each admitted frame's prepared image, registered but not normalized,
/// kept in a scratch file after it is first made, so replay's passes and
/// its integrated reference read it back instead of reading, calibrating,
/// demosaicing and resampling the source again. A frame whose file could
/// not be written is prepared again when it is next needed.
struct FrameCache {
    directory: tempfile::TempDir,
    slots: Vec<std::sync::Mutex<CacheSlot>>,
}

#[derive(Clone, Copy, PartialEq)]
enum CacheSlot {
    Empty,
    Stored,
    Unavailable,
}

impl FrameCache {
    fn new(directory: Option<&Path>, frames: usize) -> Option<Self> {
        let directory = match directory {
            Some(directory) => tempfile::Builder::new()
                .prefix(".seiza-reintegrate-")
                .tempdir_in(directory),
            None => tempfile::Builder::new()
                .prefix("seiza-reintegrate-")
                .tempdir(),
        }
        .ok()?;
        Some(Self {
            directory,
            slots: (0..frames)
                .map(|_| std::sync::Mutex::new(CacheSlot::Empty))
                .collect(),
        })
    }

    fn path(&self, index: usize) -> PathBuf {
        self.directory.path().join(format!("frame-{index}.f32"))
    }

    /// The cached image of frame `index`, made with `prepare` the first
    /// time. Holding the slot while preparing keeps two passes that reach
    /// the same frame at once from preparing it twice.
    fn get_or_prepare(
        &self,
        index: usize,
        prepare: impl FnOnce() -> Result<LinearImage>,
    ) -> Result<LinearImage> {
        let mut slot = self.slots[index]
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if *slot == CacheSlot::Stored {
            if let Some(image) = Self::read(&self.path(index)) {
                return Ok(image);
            }
            *slot = CacheSlot::Unavailable;
        }
        let image = prepare()?;
        if *slot == CacheSlot::Empty {
            *slot = if Self::write(&self.path(index), &image).is_ok() {
                CacheSlot::Stored
            } else {
                let _ = std::fs::remove_file(self.path(index));
                CacheSlot::Unavailable
            };
        }
        Ok(image)
    }

    fn write(path: &Path, image: &LinearImage) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(path)?);
        for dimension in [image.width, image.height, image.channels] {
            file.write_all(&(dimension as u64).to_le_bytes())?;
        }
        let mut bytes = vec![0_u8; 1 << 20];
        for chunk in image.data.chunks(bytes.len() / 4) {
            for (sample, out) in chunk.iter().zip(bytes.chunks_exact_mut(4)) {
                out.copy_from_slice(&sample.to_le_bytes());
            }
            file.write_all(&bytes[..chunk.len() * 4])?;
        }
        file.into_inner().map_err(|error| error.into_error())?;
        Ok(())
    }

    fn read(path: &Path) -> Option<LinearImage> {
        let bytes = std::fs::read(path).ok()?;
        let dimension = |at: usize| {
            u64::from_le_bytes(bytes.get(at..at + 8)?.try_into().ok()?)
                .try_into()
                .ok()
        };
        let (width, height, channels): (usize, usize, usize) =
            (dimension(0)?, dimension(8)?, dimension(16)?);
        let samples = &bytes[24..];
        if samples.len()
            != width
                .checked_mul(height)?
                .checked_mul(channels)?
                .checked_mul(4)?
        {
            return None;
        }
        let data = samples
            .par_chunks_exact(4)
            .map(|sample| f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]))
            .collect();
        LinearImage::new(width, height, channels, data).ok()
    }
}

impl MastersCache {
    fn get<'a>(&self, stacker: &'a LiveStacker, calibration: u32) -> Result<ReplayMasters<'a>> {
        if calibration == stacker.ledger.current_calibration() {
            return Ok(ReplayMasters::Current(&stacker.calibration));
        }
        let mut loaded = self
            .loaded
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some((_, masters)) = loaded.iter().find(|(set, _)| *set == calibration) {
            return Ok(ReplayMasters::Loaded(std::sync::Arc::clone(masters)));
        }
        let masters = std::sync::Arc::new(
            stacker.ledger.calibrations[calibration as usize]
                .load()
                .ok_or_else(|| {
                    Error::Stack("calibration masters are no longer available".into())
                })??,
        );
        if loaded.len() == 2 {
            loaded.remove(0);
        }
        loaded.push((calibration, std::sync::Arc::clone(&masters)));
        Ok(ReplayMasters::Loaded(masters))
    }
}

impl LiveStacker {
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
    /// Each frame is read three times, from the file it came from, and prepared
    /// exactly as the live pass prepared it: the same declared-range
    /// rescale, calibration masters, cosmetic filter, debayering, and the
    /// recorded registration and normalization mapping. Star detection and
    /// registration do not run again, and no admission decision is revisited:
    /// every admitted frame takes part. A source file that changed since it
    /// was stacked is refused.
    ///
    /// The live stack is left as it was. `progress` receives the pass, the
    /// zero-based frame index, and the frame count before each read. See
    /// [`integrate_registered_frames`] for the rejection and its memory use.
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
    /// The drizzle runs alongside the final pass, so each frame is read once
    /// more, from its source file, and calibrated but not debayered or
    /// resampled. A source pixel whose registered sample the integration
    /// rejected is left out, and every pixel is normalized and weighted as
    /// the integration did its registered sample. A Bayer frame drizzles its
    /// photosites, each into its own colour.
    ///
    /// The drizzle needs eight bytes per output sample on top of
    /// reintegration's memory: 2.5 GB for a 26 MP colour frame at twice the
    /// scale. Returns the reintegrated stack and the drizzled one.
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
        mut progress: impl FnMut(BatchStackPass, usize, usize),
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
        // Bayer drizzle resamples photosites, which the cache does not hold.
        let cache = (self.options.cfa_integration != crate::CfaIntegration::BayerDrizzle)
            .then(|| FrameCache::new(options.scratch_directory.as_deref(), count))
            .flatten();
        let cache = cache.as_ref();
        let renormalizer =
            if let crate::NormalizationMode::LocalBackground { .. } = self.options.normalization {
                Some(Renormalizer {
                    reference: self.normalization_reference(&masters, cache)?,
                    maps: std::sync::Mutex::new(vec![None; count]),
                })
            } else {
                None
            };
        let accumulator = drizzle
            .map(|drizzle| {
                DrizzleAccumulator::new(
                    *drizzle,
                    self.reference.width,
                    self.reference.height,
                    self.reference.channels,
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
        let lookahead = std::thread::available_parallelism()
            .map_or(1, |cores| (cores.get() / 6).clamp(1, REPLAY_LOOKAHEAD));
        // The calibrated source of the frame the final pass is integrating,
        // read alongside its registered image for the drizzle.
        let drizzle_source = std::cell::RefCell::new(None);
        let (mut result, accumulator) = std::thread::scope(|scope| {
            let prepare = |pass: BatchStackPass, index: usize| {
                let frame = &ledger.frames[index];
                let masters = masters.get(self, frame.calibration)?;
                let image =
                    self.replay_frame(frame, &masters, index, renormalizer.as_ref(), cache)?;
                let source = if drizzle.is_some() && pass == BatchStackPass::Integrate {
                    Some(self.read_calibrated(frame, &masters)?)
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
                    in_flight
                        .push_back((next, scope.spawn(move || prepare(frame_pass, frame_index))));
                    next += 1;
                }
                let (image, source) =
                    match in_flight.pop_front() {
                        Some((position, handle)) if order[position] == (pass, index) => handle
                            .join()
                            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))?,
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
                            accumulator.add(&DrizzleFrame {
                                image: &job.image,
                                layout: job.layout,
                                mapping: &ledger.frames[job.index].mapping,
                                normalization: &job.normalization,
                                fates: &job.fates,
                                weight: &job.weight,
                            })?;
                        }
                        Ok::<_, Error>(accumulator)
                    });
                    (Some(sender), Some(worker))
                }
                None => (None, None),
            };
            let mut observe = |index: usize, fates: Vec<SampleFate>| -> Result<()> {
                let Some(jobs) = &jobs else {
                    return Ok(());
                };
                let frame = &ledger.frames[index];
                let (image, layout) = match drizzle_source.borrow_mut().take() {
                    Some((source_index, source)) if source_index == index => source,
                    _ => self.read_calibrated(frame, &*masters.get(self, frame.calibration)?)?,
                };
                let normalization = match &renormalizer {
                    Some(renormalizer) => renormalizer
                        .maps
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner())[index]
                        .clone()
                        .ok_or_else(|| {
                            Error::Stack(
                                "a frame reached the drizzle without its normalization".into(),
                            )
                        })?,
                    None => frame.mapping.normalization().clone(),
                };
                let weight = match &options.frame_weights {
                    Some(weights) => weights[index].clone(),
                    None => vec![1.0; self.reference.channels],
                };
                // A closed channel means the drizzle failed; its error is
                // reported when the worker is joined.
                jobs.send(DrizzleJob {
                    index,
                    image,
                    layout,
                    normalization,
                    fates,
                    weight,
                })
                .map_err(|_| Error::Stack("the drizzle stopped early".into()))
            };
            let integrated = crate::batch::integrate_registered_frames_observed(
                count,
                options,
                load,
                drizzle.is_some().then_some(&mut observe as _),
            );
            drop(jobs);
            let drizzled = worker
                .map(|worker| {
                    worker
                        .join()
                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                })
                .transpose()?;
            Ok::<_, Error>((integrated?, drizzled))
        })?;
        result.snapshot.rejected_frames = self.rejected_frames;
        self.fill_photosite_gaps(&mut result.snapshot.image.data, &result.snapshot.coverage);
        let drizzled = accumulator.map(DrizzleAccumulator::finish).transpose()?;
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
    fn normalization_reference(
        &self,
        masters: &MastersCache,
        cache: Option<&FrameCache>,
    ) -> Result<LinearImage> {
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
        let samples = self.reference.sample_count();
        let mut sum = vec![0.0_f32; samples];
        let mut count = vec![0_u16; samples];
        for index in ranked {
            let admitted = &frames[index];
            let masters = masters.get(self, admitted.calibration)?;
            let raw = self.prepared(index, admitted, &masters, cache)?;
            // The frame's recorded gain and mean offset: its background stays
            // its own, while its stars match the reference's.
            let mut image = raw;
            admitted
                .mapping
                .normalization()
                .global_equivalent()
                .apply_global(&mut image)?;
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
        let mean = sum
            .into_par_iter()
            .zip(count.into_par_iter())
            .map(|(sum, count)| {
                if count == 0 {
                    f32::NAN
                } else {
                    sum / f32::from(count)
                }
            })
            .collect();
        let image = LinearImage::new(
            self.reference.width,
            self.reference.height,
            self.reference.channels,
            mean,
        )?;
        Ok(image)
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
        let identity = admitted
            .mapping
            .with_normalization(crate::NormalizationMap::identity(&self.reference))?;
        identity.extract_region_with(&frame.image, self.full_region(admitted), interpolation)
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
        let make = || self.replay_frame_unnormalized(admitted, masters, self.options.interpolation);
        match cache {
            Some(cache) => {
                self.source_unchanged(admitted)?;
                cache.get_or_prepare(index, make)
            }
            None => make(),
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
                .map_for(self, admitted, index, &image)?
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
        let map = renormalizer.map_for(self, admitted, index, &interpolated)?;
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
