//! One session's step: its lights stacked on the channel's reference grid
//! into a partial stack, named by a key computed before it runs.

use crate::digest::KeyHasher;
use crate::partial::reference_grid;
use crate::{
    Digest, Error, Estimator, Identity, PartialStack, Result, SessionOutcome, StackKind, Store,
    WeightSource,
};
use rayon::prelude::*;
use seiza_stacking::{
    BatchStackOptions, FileCalibrationSession, FrameDisposition, FrameRejectionReason, LiveStacker,
    StackOptions,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// What every session of one channel is stacked against: the image they
/// align to, which sets the grid, and the light they are normalized and
/// weighted against, calibrated by its own recipe. Sessions that share it
/// put their frames on one grid, on one scale and with one weight, so their
/// partial stacks combine into what one stack of every frame would give.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChannelReference {
    /// The image every session aligns to: the reference light itself, or an
    /// image several filters share so their stacks share one grid.
    pub image: PathBuf,
    /// The light that sets normalization and noise weighting.
    pub light: PathBuf,
    /// The reference light's own calibration recipe.
    #[serde(default)]
    pub calibration: FileCalibrationSession,
}

/// One session: lights of one target through one optical train and filter,
/// calibrated by one recipe, usually in one night.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Session {
    /// The lights, in the order they are stacked.
    pub lights: Vec<PathBuf>,
    /// Their calibration recipe and gate.
    #[serde(default)]
    pub calibration: FileCalibrationSession,
}

/// Reintegrating a session's frames with three-pass leave-one-out
/// rejection, as `seiza_stacking::LiveStacker::reintegrate` does, so a
/// transient in any frame is rejected. The partial stack is then
/// [`Estimator::Clipped`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Reintegration {
    /// Reject a sample this many sigma below the others.
    pub low_sigma: f32,
    /// Reject a sample this many sigma above the others.
    pub high_sigma: f32,
    /// The noise floor, in the frames' units.
    pub minimum_sigma: f32,
    /// Where frames wait between passes; the system's temporary directory
    /// when `None`. It changes no pixel, so no key holds it.
    pub scratch_directory: Option<PathBuf>,
}

impl Default for Reintegration {
    fn default() -> Self {
        let options = BatchStackOptions::default();
        Self {
            low_sigma: options.rejection.low_sigma,
            high_sigma: options.rejection.high_sigma,
            minimum_sigma: options.minimum_sigma,
            scratch_directory: None,
        }
    }
}

impl Reintegration {
    fn batch_options(&self) -> BatchStackOptions {
        let mut options = BatchStackOptions {
            minimum_sigma: self.minimum_sigma,
            scratch_directory: self.scratch_directory.clone(),
            ..BatchStackOptions::default()
        };
        options.rejection.low_sigma = self.low_sigma;
        options.rejection.high_sigma = self.high_sigma;
        options
    }
}

/// How every session of a channel is stacked.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SessionOptions {
    /// How frames are registered, normalized, weighted, rejected live and
    /// admitted. Each session's recipe sets its own match tolerances, so
    /// these options' are not used; where the arithmetic runs is no part of
    /// any key.
    pub stack: StackOptions,
    /// Reintegrate each session's frames with leave-one-out rejection, for a
    /// clipped partial stack; `None` keeps the live stack's running mean, an
    /// additive one.
    pub reintegration: Option<Reintegration>,
    /// Who contributed the frames, written as `CONTRIB`.
    pub contributor: Option<String>,
    /// A fingerprint of the host's own decisions, written as `HOSTSHA` and
    /// held in every key.
    pub host: Option<Digest>,
}

/// A session's step, planned: every file it reads hashed, and its key
/// computed from those digests, before any frame is stacked. A store that
/// holds a file under [`Self::key`] holds this step's output.
#[derive(Clone, Debug)]
pub struct SessionStep {
    reference: ChannelReference,
    session: Session,
    options: SessionOptions,
    /// The step's key, `STEPKEY`: a digest of everything below, the
    /// reference light and its recipe, and the output versions.
    pub key: Digest,
    /// `INPUTSHA`: the lights' digests, in order.
    pub inputs: Digest,
    /// Each light's digest, in order.
    pub frames: Vec<Digest>,
    /// `CALIBSHA`: the session's recipe, its masters by their digests.
    pub calibration: Digest,
    /// `OPTSSHA`: the options that change the pixels.
    pub stack_options: Digest,
    /// `GRIDSHA`: the reference image and bin factor.
    pub grid: Digest,
}

/// A session's partial stack, and the lights the stack turned away.
#[derive(Clone, Debug)]
pub struct SessionStack {
    /// The partial stack.
    pub stack: PartialStack,
    /// Each light turned away, and why.
    pub rejected: Vec<(PathBuf, FrameRejectionReason)>,
}

impl SessionStep {
    /// Plan `session`'s step on `reference`: hash its lights, its masters,
    /// the reference image and light and the reference's masters, and
    /// compute its key. The lights are hashed in parallel; nothing is
    /// stacked.
    pub fn plan(
        reference: &ChannelReference,
        session: &Session,
        options: &SessionOptions,
    ) -> Result<Self> {
        if session.lights.is_empty() {
            return Err(Error::Invalid("a session needs at least one light".into()));
        }
        let frames = session
            .lights
            .par_iter()
            .map(|light| Digest::of_file(light))
            .collect::<Result<Vec<_>>>()?;
        let mut inputs = KeyHasher::new("seiza-pipeline-frames-v1");
        inputs.number(frames.len() as u64);
        for frame in &frames {
            inputs.digest(frame);
        }
        let inputs = inputs.finish();
        let calibration = recipe_digest(&session.calibration)?;
        let reference_recipe = recipe_digest(&reference.calibration)?;
        let reference_light = Digest::of_file(&reference.light)?;
        let grid = reference_grid(
            &Digest::of_file(&reference.image)?,
            options.stack.bin_factor,
        );
        let stack_options = options_digest(options)?;
        let key = KeyHasher::new("seiza-pipeline-partial-v1")
            .text(seiza_stacking::OUTPUT_VERSION)
            .text(crate::OUTPUT_VERSION)
            .digest(&grid)
            .digest(&reference_light)
            .digest(&reference_recipe)
            .digest(&calibration)
            .digest(&stack_options)
            .digest(&inputs)
            .optional_digest(options.host.as_ref())
            .finish();
        Ok(Self {
            reference: reference.clone(),
            session: session.clone(),
            options: options.clone(),
            key,
            inputs,
            frames,
            calibration,
            stack_options,
            grid,
        })
    }

    /// Stack the session: open a stack aligned to the reference image and
    /// normalized against the reference light, calibrated by its own recipe
    /// (`LiveStacker::open_fits_sessions_with_reference_image`), push the
    /// session's lights under theirs, and reintegrate them when the options
    /// say so. A session none of whose lights is accepted is
    /// [`Error::Empty`].
    pub fn run(&self) -> Result<SessionStack> {
        let mut recipes = vec![self.reference.calibration.clone()];
        let session_recipe = if self.session.calibration == self.reference.calibration {
            0
        } else {
            recipes.push(self.session.calibration.clone());
            1
        };
        let paths = std::iter::once(self.reference.light.clone())
            .chain(self.session.lights.iter().cloned())
            .collect::<Vec<_>>();
        let assignments = std::iter::once(0)
            .chain(self.session.lights.iter().map(|_| session_recipe))
            .collect::<Vec<_>>();
        let mut stacker = LiveStacker::open_fits_sessions_with_reference_image(
            &self.reference.image,
            &paths,
            &recipes,
            &assignments,
            0,
            self.options.stack.clone(),
        )?;
        let mut rejected = Vec::new();
        for light in &self.session.lights {
            if let FrameDisposition::Rejected(reason) = stacker.push_fits(light)? {
                rejected.push((light.clone(), reason));
            }
        }
        let accepted = stacker.view().accepted_frames;
        if accepted == 0 {
            return Err(Error::Empty);
        }
        let headers = stacker.reference_headers().to_vec();
        let (snapshot, estimator) = match &self.options.reintegration {
            Some(reintegration) => (
                stacker
                    .reintegrate(&reintegration.batch_options(), |_, _, _| {})?
                    .snapshot,
                Estimator::Clipped,
            ),
            None => (stacker.into_snapshot()?, Estimator::Additive),
        };
        let identity = Identity {
            kind: StackKind::Partial,
            stacking_version: seiza_stacking::OUTPUT_VERSION.into(),
            pipeline_version: crate::OUTPUT_VERSION.into(),
            step_key: self.key,
            inputs: self.inputs,
            input_count: self.frames.len() as u32,
            combined: accepted,
            calibration: Some(self.calibration),
            options: self.stack_options,
            host: self.options.host,
            estimator,
            context: None,
            contributor: self.options.contributor.clone(),
            weight_source: WeightSource::Stack,
            grid: self.grid,
        };
        Ok(SessionStack {
            stack: PartialStack::from_snapshot(snapshot, &headers, identity)?,
            rejected,
        })
    }

    /// Do the step against `store`: its partial stack when the store holds
    /// one under [`Self::key`], or else [`Self::run`] stored there. A
    /// session that accepts no light stores nothing, and its outcome has no
    /// path.
    pub fn run_into(&self, store: &Store) -> Result<SessionOutcome> {
        if let Some(path) = store.find(&self.key, PARTIAL)? {
            return Ok(SessionOutcome {
                key: self.key,
                path: Some(path),
                reused: true,
                rejected: Vec::new(),
            });
        }
        match self.run() {
            Ok(stacked) => Ok(SessionOutcome {
                key: self.key,
                path: Some(store.put(&stacked.stack, PARTIAL)?),
                reused: false,
                rejected: stacked.rejected,
            }),
            Err(Error::Empty) => Ok(SessionOutcome {
                key: self.key,
                path: None,
                reused: false,
                rejected: Vec::new(),
            }),
            Err(error) => Err(error),
        }
    }

    /// The session's lights.
    pub fn lights(&self) -> &[PathBuf] {
        &self.session.lights
    }
}

/// The file name a partial stack is stored under.
pub(crate) const PARTIAL: &str = "partial.fits";

/// The digest of a calibration recipe: its masters by their digests, its
/// dark exposure, pedestal and declared scale, and its gate.
fn recipe_digest(recipe: &FileCalibrationSession) -> Result<Digest> {
    let mut hasher = KeyHasher::new("seiza-pipeline-calibration-v1");
    for master in [&recipe.bias, &recipe.dark, &recipe.flat] {
        let digest = master.as_deref().map(Digest::of_file).transpose()?;
        hasher.optional_digest(digest.as_ref());
    }
    hasher
        .optional_float(recipe.dark_exposure_seconds)
        .optional_float(recipe.pedestal.map(f64::from))
        .optional_float(recipe.normalized_full_scale.map(f64::from));
    hasher.bytes(&json(&recipe.tolerances)?);
    Ok(hasher.finish())
}

/// The digest of the options that change a partial stack's pixels.
fn options_digest(options: &SessionOptions) -> Result<Digest> {
    // Each session's recipe sets its own gate, and where the arithmetic
    // runs moves a stack by rounding at most.
    let stack = StackOptions {
        gpu: Default::default(),
        match_tolerances: Default::default(),
        ..options.stack.clone()
    };
    let mut hasher = KeyHasher::new("seiza-pipeline-options-v1");
    hasher.bytes(&json(&stack)?);
    match &options.reintegration {
        Some(reintegration) => hasher
            .number(1)
            .optional_float(Some(f64::from(reintegration.low_sigma)))
            .optional_float(Some(f64::from(reintegration.high_sigma)))
            .optional_float(Some(f64::from(reintegration.minimum_sigma))),
        None => hasher.number(0),
    };
    Ok(hasher.finish())
}

fn json(value: &impl Serialize) -> Result<Vec<u8>> {
    serde_json::to_vec(value).map_err(|error| Error::Invalid(error.to_string()))
}
