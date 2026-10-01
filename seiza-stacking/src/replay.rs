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

use crate::{
    BatchStackOptions, BatchStackPass, BatchStackResult, CalibrationMasters, Error, FitsFrame,
    LinearImage, LiveStacker, ReferenceRegion, RegisteredFrameMapping, Result,
    integrate_registered_frames,
};
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

    /// Integrate every admitted frame again with two-pass, leave-one-out
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
        mut progress: impl FnMut(BatchStackPass, usize, usize),
    ) -> Result<BatchStackResult> {
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
        let current = ledger.current_calibration();
        let count = ledger.frames.len();
        // Masters of earlier calibration sets, loaded once per switch.
        let mut loaded: Option<(u32, CalibrationMasters)> = None;
        let mut result = integrate_registered_frames(count, options, |pass, index| {
            progress(pass, index, count);
            let frame = &ledger.frames[index];
            let masters = if frame.calibration == current {
                &self.calibration
            } else {
                if loaded
                    .as_ref()
                    .is_none_or(|(generation, _)| *generation != frame.calibration)
                {
                    let masters = ledger.calibrations[frame.calibration as usize]
                        .load()
                        .ok_or_else(|| {
                            Error::Stack("calibration masters are no longer available".into())
                        })??;
                    loaded = Some((frame.calibration, masters));
                }
                &loaded.as_ref().expect("loaded just above").1
            };
            self.replay_frame(frame, masters)
        })?;
        result.snapshot.rejected_frames = self.rejected_frames;
        Ok(result)
    }

    fn replay_frame(
        &self,
        admitted: &AdmittedFrame,
        masters: &CalibrationMasters,
    ) -> Result<LinearImage> {
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
        let frame = frame.into_prepared()?;
        let image = admitted.mapping.extract_region(
            &frame.image,
            ReferenceRegion {
                x: 0,
                y: 0,
                width: admitted.mapping.reference_width(),
                height: admitted.mapping.reference_height(),
            },
        )?;
        if source.stamp.is_some() && SourceStamp::of(&source.path) != source.stamp {
            return Err(changed());
        }
        Ok(image)
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
