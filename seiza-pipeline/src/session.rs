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
    /// Keep each session's live-stack context in the store beside its
    /// partial stack, written as `CONTEXT`, so a session that gains lights
    /// later is extended from it: only the new lights are pushed (see
    /// [`SessionStep::run_into`]). A context holds the stack's moments and
    /// reference, more than the partial stack itself. It changes no pixel,
    /// so no key holds it.
    pub keep_context: bool,
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
    reference_light: Digest,
    reference_recipe: Digest,
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
        let mut step = Self {
            reference: reference.clone(),
            session: session.clone(),
            options: options.clone(),
            key: Digest::of_bytes(&[]),
            inputs: inputs_digest(&frames),
            frames,
            calibration: recipe_digest(&session.calibration)?,
            stack_options: options_digest(options)?,
            grid: reference_grid(
                &Digest::of_file(&reference.image)?,
                options.stack.bin_factor,
            ),
            reference_light: Digest::of_file(&reference.light)?,
            reference_recipe: recipe_digest(&reference.calibration)?,
        };
        step.key = step.key_for(&step.inputs);
        Ok(step)
    }

    /// The key of this session's step over lights whose digests give
    /// `inputs`, everything else the same.
    fn key_for(&self, inputs: &Digest) -> Digest {
        KeyHasher::new("seiza-pipeline-partial-v1")
            .text(seiza_stacking::OUTPUT_VERSION)
            .text(crate::OUTPUT_VERSION)
            .digest(&self.grid)
            .digest(&self.reference_light)
            .digest(&self.reference_recipe)
            .digest(&self.calibration)
            .digest(&self.stack_options)
            .digest(inputs)
            .optional_digest(self.options.host.as_ref())
            .finish()
    }

    /// Stack the session: open a stack aligned to the reference image and
    /// normalized against the reference light, calibrated by its own recipe
    /// (`LiveStacker::open_fits_sessions_with_reference_image`), push the
    /// session's lights under theirs, and reintegrate them when the options
    /// say so. A session none of whose lights is accepted is
    /// [`Error::Empty`].
    pub fn run(&self) -> Result<SessionStack> {
        let mut stacker = self.open()?;
        let rejected = push(&mut stacker, &self.session.lights)?;
        Ok(SessionStack {
            stack: self.finish(stacker, None)?,
            rejected,
        })
    }

    /// Do the step against `store`: its partial stack when the store holds
    /// one under [`Self::key`], or else the session stacked and stored there.
    /// A session that accepts no light stores nothing, and its outcome has
    /// no path.
    ///
    /// With [`SessionOptions::keep_context`], the live stack's context is
    /// stored beside the partial stack, and a session that has gained lights
    /// since is extended rather than stacked again: when the store holds
    /// this step over the session's first lights, with its context, that
    /// context is reopened and only the lights after them are pushed. The
    /// partial stack is then, to the bit, the one stacking every light would
    /// give. A reintegrated session reintegrates every light again, from
    /// their files.
    pub fn run_into(&self, store: &Store) -> Result<SessionOutcome> {
        if let Some(path) = store.find(&self.key, PARTIAL)? {
            return Ok(SessionOutcome {
                key: self.key,
                path: Some(path),
                reused: true,
                extended_from: None,
                rejected: Vec::new(),
            });
        }
        let prefix = if self.options.keep_context {
            self.stored_prefix(store)?
        } else {
            None
        };
        let (mut stacker, taken, extended_from) = match prefix {
            Some((key, context, taken)) => {
                let mut stacker = LiveStacker::open_context(&context)?;
                stacker.add_session_lights(&self.session.lights[taken..], self.session_recipe())?;
                (stacker, taken, Some(key))
            }
            None => (self.open()?, 0, None),
        };
        let rejected = push(&mut stacker, &self.session.lights[taken..])?;
        if stacker.view().accepted_frames == 0 {
            return Ok(SessionOutcome {
                key: self.key,
                path: None,
                reused: false,
                extended_from,
                rejected,
            });
        }
        let context = if self.options.keep_context {
            let path = store.put_file(&self.key, CONTEXT, |path| {
                stacker.save_context(path).map_err(Error::from)
            })?;
            Some(Digest::of_file(&path)?)
        } else {
            None
        };
        let stack = self.finish(stacker, context)?;
        Ok(SessionOutcome {
            key: self.key,
            path: Some(store.put(&stack, PARTIAL)?),
            reused: false,
            extended_from,
            rejected,
        })
    }

    /// The step over the session's first lights that `store` holds with its
    /// context, the longest first: its key, its context and how many lights
    /// it took. A context whose digest is not the one its partial stack
    /// records is passed over.
    fn stored_prefix(&self, store: &Store) -> Result<Option<(Digest, PathBuf, usize)>> {
        for taken in (1..self.frames.len()).rev() {
            let key = self.key_for(&inputs_digest(&self.frames[..taken]));
            let Some(partial) = store.find(&key, PARTIAL)? else {
                continue;
            };
            let Some(recorded) = PartialStack::read_identity(&partial)?.context else {
                continue;
            };
            let context = store.path(&key, CONTEXT);
            if context.is_file() && Digest::of_file(&context)? == recorded {
                return Ok(Some((key, context, taken)));
            }
        }
        Ok(None)
    }

    /// The index of the session's recipe among those [`Self::open`] gives
    /// the stack: 0 when it is the reference's own.
    fn session_recipe(&self) -> usize {
        usize::from(self.session.calibration != self.reference.calibration)
    }

    /// An empty stack on the reference, with the reference's recipe and the
    /// session's, and the reference light and the session's lights in its
    /// plan.
    fn open(&self) -> Result<LiveStacker> {
        let mut recipes = vec![self.reference.calibration.clone()];
        if self.session_recipe() == 1 {
            recipes.push(self.session.calibration.clone());
        }
        let paths = std::iter::once(self.reference.light.clone())
            .chain(self.session.lights.iter().cloned())
            .collect::<Vec<_>>();
        let assignments = std::iter::once(0)
            .chain(self.session.lights.iter().map(|_| self.session_recipe()))
            .collect::<Vec<_>>();
        Ok(LiveStacker::open_fits_sessions_with_reference_image(
            &self.reference.image,
            &paths,
            &recipes,
            &assignments,
            0,
            self.options.stack.clone(),
        )?)
    }

    /// The partial stack of `stacker`, which holds every light of the
    /// session it accepted: its live mean, or every light reintegrated.
    fn finish(&self, stacker: LiveStacker, context: Option<Digest>) -> Result<PartialStack> {
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
            context,
            contributor: self.options.contributor.clone(),
            weight_source: WeightSource::Stack,
            grid: self.grid,
        };
        PartialStack::from_snapshot(snapshot, &headers, identity)
    }

    /// The session's lights.
    pub fn lights(&self) -> &[PathBuf] {
        &self.session.lights
    }
}

/// The file name a partial stack is stored under.
pub(crate) const PARTIAL: &str = "partial.fits";
/// The file name a partial stack's live-stack context is stored under.
const CONTEXT: &str = "context.seizastack";

/// Push `lights`, returning each one turned away and why.
fn push(
    stacker: &mut LiveStacker,
    lights: &[PathBuf],
) -> Result<Vec<(PathBuf, FrameRejectionReason)>> {
    let mut rejected = Vec::new();
    for light in lights {
        if let FrameDisposition::Rejected(reason) = stacker.push_fits(light)? {
            rejected.push((light.clone(), reason));
        }
    }
    Ok(rejected)
}

/// `INPUTSHA` of lights with these digests, in order.
fn inputs_digest(frames: &[Digest]) -> Digest {
    let mut inputs = KeyHasher::new("seiza-pipeline-frames-v1");
    inputs.number(frames.len() as u64);
    for frame in frames {
        inputs.digest(frame);
    }
    inputs.finish()
}

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
