//! The description of a color picture that [`crate::render`] renders.
//!
//! A [`Picture`] lists layers, bottom first, each showing one of the
//! picture's inputs through its own [`Transfer`], in a color, blended over
//! the layers beneath. Masks limit where a layer applies, heals copy pixels
//! from elsewhere, and [`FinishStep`]s change the combined picture. The
//! `combine_*` functions build pictures of this kind; a host with a document
//! of its own, such as PSF Studio, builds its own.

use seiza_stretch::{
    HdrTail, LayeredMaps, ResolvedCurve, ResolvedLayered, ResolvedSampleDomain, StretchPlan,
};
use std::sync::Arc;

/// One stage of a [`Transfer`].
#[derive(Clone, Debug, PartialEq)]
pub enum TransferStage {
    /// Multiplies the sample.
    Scale(f32),
    /// Keeps the sample within `low` to `high`.
    Clamp { low: f32, high: f32 },
    /// Maps `black` to 0 and `black + 1 / reciprocal_range` to 1, and keeps
    /// the result within 0 to 1: percentile normalization, as
    /// [`crate::ColorNormalization::Percentile`] resolves it.
    Normalize { black: f32, reciprocal_range: f32 },
    /// Seiza's mapping of physical samples onto the unit range a stretch
    /// works on.
    Domain(ResolvedSampleDomain),
    /// One of Seiza's resolved stretch curves.
    Curve(ResolvedCurve),
    /// A resolved curve over 0 to 1, with values above SDR white passing as
    /// they are: a stretch after an HDR one shapes the range up to white
    /// and keeps the headroom, where [`Self::Curve`] would clip it.
    BelowWhite(ResolvedCurve),
    /// A resolved curve continued into an HDR picture's headroom above its
    /// knee: values above 1 are on the extended sRGB curve, up to the
    /// tail's peak.
    TailedCurve { curve: ResolvedCurve, tail: HdrTail },
    /// A narrowband channel matched to H-alpha, as
    /// [`crate::balance_narrowband`] resolves it: above `background`, the
    /// midtones curve of `midtone` over the range from there to 1; below,
    /// the sample as it is.
    Balance { background: f32, midtone: f32 },
    /// Seiza's layered stretch of the sample, read as luminance: its curve,
    /// with the core compression, HDR tail and contrast recovery its maps
    /// call for. It reads the maps at the sample's place in the picture.
    Layered(LayeredStage),
}

/// A layered stretch as a transfer stage: the resolved stretch, its maps
/// measured from the whole picture, and the HDR tail they give it.
#[derive(Clone, Debug, PartialEq)]
pub struct LayeredStage {
    pub layered: ResolvedLayered,
    pub maps: Arc<LayeredMaps>,
    /// The input that holds the luminance the stretch reads, as the stretch
    /// reads it (0 to 1 over its range): for red, green and blue stretched
    /// together, so each channel follows their luminance. `None` reads the
    /// sample itself as luminance.
    pub luminance: Option<usize>,
    /// The input that holds that luminance blurred by
    /// `seiza_stretch::fine_luminance`, for star-core compression. `None`
    /// compresses no star cores.
    pub fine: Option<usize>,
    tail: Option<HdrTail>,
}

impl LayeredStage {
    /// The stage of `layered` with `maps`, measured by
    /// [`ResolvedLayered::measure`] from the whole picture.
    pub fn new(layered: ResolvedLayered, maps: Arc<LayeredMaps>) -> Self {
        let tail = layered.tail(&maps);
        Self {
            layered,
            maps,
            luminance: None,
            fine: None,
            tail,
        }
    }

    /// The stage reading its luminance from input `luminance`, and the
    /// star-core compression's blur of it from input `fine`.
    pub fn with_guides(self, luminance: Option<usize>, fine: Option<usize>) -> Self {
        Self {
            luminance,
            fine,
            ..self
        }
    }

    /// The HDR tail the stage continues its curve with, if any.
    pub fn tail(&self) -> Option<&HdrTail> {
        self.tail.as_ref()
    }

    /// The brightest value the stage gives.
    pub fn peak(&self) -> f32 {
        self.layered.peak(self.tail.as_ref())
    }

    /// The stage of a sample at picture pixel `(x, y)`, with the values of
    /// its guide inputs there; with no place, of a sample alone, which
    /// reads no maps.
    pub fn map(
        &self,
        value: f32,
        at: Option<(usize, usize)>,
        luminance: Option<f32>,
        fine: Option<f32>,
    ) -> f32 {
        let sample = at.map_or_else(Default::default, |(x, y)| self.maps.sample(x, y));
        let read = luminance.unwrap_or(value);
        let mut stretched = self
            .layered
            .stretch_luminance(read, sample, self.tail.as_ref());
        if let Some(fine) = fine {
            stretched = self.layered.star_cores(stretched, read, fine);
        }
        let peak = f64::from(self.peak());
        let stretched = self.layered.sky_floor(stretched, peak);
        let out = match luminance {
            Some(luminance) => self
                .layered
                .colour_channel(value, luminance, stretched, peak),
            None => stretched.clamp(0.0, peak),
        };
        out as f32
    }
}

/// How an input's samples become a layer's values: its stages, in order. No
/// stage leaves the samples as they are.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Transfer {
    pub stages: Vec<TransferStage>,
}

impl Transfer {
    /// The transfer that leaves samples as they are.
    pub fn identity() -> Self {
        Self::default()
    }

    /// A transfer of these stages.
    pub fn of(stages: impl IntoIterator<Item = TransferStage>) -> Self {
        Self {
            stages: stages.into_iter().collect(),
        }
    }

    /// Applies the transfer to every sample of `samples`. A non-finite
    /// sample stays non-finite. A layered stage reads no maps, as for a
    /// sample alone; [`Self::apply_at`] gives the samples their places.
    pub fn apply(&self, samples: &mut [f32]) {
        self.apply_placed(samples, None, &|_| None);
    }

    /// Applies the transfer to `samples`, the rows of a rectangle `width`
    /// samples wide whose first sample is at picture pixel `origin`; a place
    /// outside the picture reads the maps at its nearest edge.
    pub fn apply_at(&self, samples: &mut [f32], origin: [i64; 2], width: usize) {
        self.apply_placed(samples, Some((origin, width)), &|_| None);
    }

    /// [`Self::apply_at`], with `planes` giving another input's samples
    /// over the same rectangle, for a layered stage's guides.
    pub fn apply_guided<'a>(
        &self,
        samples: &mut [f32],
        origin: [i64; 2],
        width: usize,
        planes: &dyn Fn(usize) -> Option<&'a [f32]>,
    ) {
        self.apply_placed(samples, Some((origin, width)), planes);
    }

    /// The inputs a layered stage reads besides the sample.
    pub fn guides(&self) -> impl Iterator<Item = usize> + '_ {
        self.stages
            .iter()
            .flat_map(|stage| match stage {
                TransferStage::Layered(stage) => [stage.luminance, stage.fine],
                _ => [None, None],
            })
            .flatten()
    }

    fn apply_placed<'a>(
        &self,
        samples: &mut [f32],
        placed: Option<([i64; 2], usize)>,
        planes: &dyn Fn(usize) -> Option<&'a [f32]>,
    ) {
        for stage in &self.stages {
            match stage {
                TransferStage::Scale(factor) => {
                    samples.iter_mut().for_each(|sample| *sample *= factor)
                }
                TransferStage::Clamp { low, high } => samples.iter_mut().for_each(|sample| {
                    if sample.is_finite() {
                        *sample = sample.clamp(*low, *high);
                    }
                }),
                TransferStage::Normalize {
                    black,
                    reciprocal_range,
                } => samples.iter_mut().for_each(|sample| {
                    if sample.is_finite() {
                        *sample = ((*sample - black) * reciprocal_range).clamp(0.0, 1.0);
                    }
                }),
                TransferStage::Domain(domain) => {
                    // A mono plane cannot be refused for its channel count.
                    let _ = domain.apply_in_place(samples, 1);
                }
                TransferStage::Curve(curve) => samples.iter_mut().for_each(|sample| {
                    if sample.is_finite() {
                        *sample = curve.map(*sample);
                    }
                }),
                TransferStage::BelowWhite(curve) => samples.iter_mut().for_each(|sample| {
                    if sample.is_finite() && *sample <= 1.0 {
                        *sample = curve.map(*sample);
                    }
                }),
                TransferStage::Balance {
                    background,
                    midtone,
                } => samples.iter_mut().for_each(|sample| {
                    if sample.is_finite() && *sample > *background {
                        let range = (1.0 - background).max(f32::MIN_POSITIVE);
                        let along = f64::from(((*sample - background) / range).min(1.0));
                        *sample = background
                            + range
                                * seiza_stretch::midtones_transfer_function(
                                    f64::from(*midtone),
                                    along,
                                ) as f32;
                    }
                }),
                TransferStage::TailedCurve { curve, tail } => {
                    samples.iter_mut().for_each(|sample| {
                        if sample.is_finite() {
                            *sample = tail.map(*curve, *sample);
                        }
                    })
                }
                TransferStage::Layered(stage) => {
                    let luminance = stage.luminance.and_then(planes);
                    let fine = stage.fine.and_then(planes);
                    for (index, sample) in samples.iter_mut().enumerate() {
                        if !sample.is_finite() {
                            continue;
                        }
                        let at = placed.map(|([x, y], width)| {
                            (
                                (x + (index % width.max(1)) as i64).max(0) as usize,
                                (y + (index / width.max(1)) as i64).max(0) as usize,
                            )
                        });
                        let guide =
                            |plane: Option<&[f32]>| plane.and_then(|p| p.get(index).copied());
                        *sample = stage.map(*sample, at, guide(luminance), guide(fine));
                    }
                }
            }
        }
    }

    /// The transfer of one sample.
    pub fn map(&self, sample: f32) -> f32 {
        let mut one = [sample];
        self.apply(&mut one);
        one[0]
    }
}

/// An input a layer or a computed mask reads: an input of the picture, with
/// a second input blended into it before the transfer, and how much of it.
#[derive(Clone, Debug, PartialEq)]
pub struct InputRead {
    pub input: usize,
    /// A second input and its share, from 0 to 1. A host uses it for a
    /// correction whose strength a person moves: the input uncorrected and
    /// corrected in full.
    pub blended: Option<(usize, f32)>,
    pub transfer: Transfer,
}

impl InputRead {
    /// `input` read through `transfer`, with nothing blended in.
    pub fn new(input: usize, transfer: Transfer) -> Self {
        Self {
            input,
            blended: None,
            transfer,
        }
    }
}

/// How a layer combines with the layers beneath it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Blend {
    #[default]
    Normal,
    /// `1 − (1 − a)(1 − b)`, over the range 0 to the picture's peak, so a
    /// layer screened over values above SDR white keeps them in order.
    Screen,
    Add,
    Multiply,
    Lighten,
    Darken,
    /// The layer's Rec. 709 luminance added to the colors beneath, less
    /// theirs: a display-space luminance swap.
    Luminance,
}

/// The luminance a [`LayerKind::Luminance`] layer gives the picture beneath
/// it, keeping each pixel's chromaticity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LuminanceTarget {
    /// The input's samples, weighted by the layer's cover against the
    /// picture's own luminance: LRGB's replacement.
    Replace,
    /// The input plus `R + G + B`; with no input, `R + G + B` alone.
    Super,
}

/// What an adjustment layer does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Adjustment {
    /// Brightens or darkens by this many stops.
    Exposure(f32),
    /// A contrast curve, from -1 to 1.
    Curves(f32),
    /// Saturation; 1 leaves the color as it is.
    Saturation(f32),
    /// Takes green down toward the brighter of red and blue where it is
    /// above both, by this much from 0 to 1: the green excess a
    /// narrowband palette leaves, as `seiza_stacking::reduce_green_excess`
    /// takes it out.
    GreenReduction(f32),
}

/// What one entry in a picture's list of layers is.
#[derive(Clone, Debug, PartialEq)]
pub enum LayerKind {
    /// An input in a color. `curve` is the layer's own contrast curve, from
    /// -1 to 1, applied after the transfer.
    Input {
        read: InputRead,
        color: [f32; 3],
        curve: f32,
    },
    /// One color everywhere.
    Solid { color: [f32; 3] },
    /// The start of a group: the entries up to its end blend together on
    /// their own, starting from black. Groups nest up to
    /// [`MAX_GROUP_DEPTH`] deep; a group deeper than that is flattened, its
    /// members blending into the group around it.
    GroupBegin,
    /// The end of a group: the members' blend becomes one layer with this
    /// entry's opacity, blend and masks.
    GroupEnd,
    /// A change to what is beneath. With [`Layer::own`], to the layer or
    /// group just before it only, before it is blended.
    Adjustment(Adjustment),
    /// A new luminance for the picture beneath, with its chromaticity kept,
    /// as LRGB combines.
    Luminance {
        read: Option<InputRead>,
        target: LuminanceTarget,
    },
    /// What a Finish step does, here in the stack: to the group's members
    /// beneath it inside a group, and to every layer beneath it at the
    /// top. A step that reads neighboring pixels, such as a local Tone step
    /// or a heal, reads the picture beneath as the layers below it leave
    /// it. The layer's opacity and masks limit where it applies, as an
    /// adjustment's do; a heal's stroke is its mask.
    Step(FinishKind),
}

/// How deep groups nest. A group inside this many others is flattened.
pub const MAX_GROUP_DEPTH: usize = 4;

/// One layer of a picture.
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub kind: LayerKind,
    /// From 0 to 1. A hidden layer has 0.
    pub opacity: f32,
    pub blend: Blend,
    /// A painted mask that limits the layer, as an index into
    /// [`Picture::masks`].
    pub mask: Option<usize>,
    /// The painted mask of the layer it is clipped to.
    pub clip: Option<usize>,
    /// A computed mask that limits the layer, as an index into
    /// [`Picture::computed`].
    pub computed: Option<usize>,
    /// For an adjustment: whether it changes only the layer or group just
    /// before it.
    pub own: bool,
}

impl Layer {
    /// A layer of this kind, fully opaque, blended normally, unmasked.
    pub fn new(kind: LayerKind) -> Self {
        Self {
            kind,
            opacity: 1.0,
            blend: Blend::Normal,
            mask: None,
            clip: None,
            computed: None,
            own: false,
        }
    }

    /// `read` in `color`, added to what is beneath.
    pub fn added(read: InputRead, color: [f32; 3]) -> Self {
        Self {
            blend: Blend::Add,
            ..Self::new(LayerKind::Input {
                read,
                color,
                curve: 0.0,
            })
        }
    }
}

/// A mask computed at each pixel from one or two inputs: the product of
/// their transferred samples, kept within 0 to 1, shaped, smoothed, with
/// painted strokes added or taken away.
#[derive(Clone, Debug, PartialEq)]
pub struct ComputedMask {
    pub reads: Vec<InputRead>,
    /// Whether the product `p` becomes `p^(1-p)`, as Foraxx's blend factors
    /// are.
    pub power_of_inverse: bool,
    /// The power the shaped product is raised to. 1 leaves it as it is.
    pub reach: f32,
    /// The standard deviation, in pixels, of the Gaussian blur the mask is
    /// smoothed with before the strokes, weighted by where its inputs cover
    /// the picture and reflected at the picture's edge, as Foraxx's
    /// smoothing is. 0 is not smoothed.
    pub smoothing: f32,
    /// Painted strokes that set the mask to 1 under them.
    pub added: Option<usize>,
    /// Painted strokes that set the mask to 0 under them.
    pub removed: Option<usize>,
    /// Whether the mask is 1 minus all of the above.
    pub inverted: bool,
}

impl ComputedMask {
    /// The mask of `reads`, shaped as Foraxx shapes a blend factor.
    pub fn foraxx(reads: Vec<InputRead>, smoothing: f32) -> Self {
        Self {
            reads,
            power_of_inverse: true,
            reach: 1.0,
            smoothing,
            added: None,
            removed: None,
            inverted: false,
        }
    }
}

/// One segment of a brush stroke, in the picture's pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    pub from: [f32; 2],
    pub to: [f32; 2],
    pub radius: f32,
    /// Where the stroke's edge starts to soften, as a share of `radius`.
    pub hardness: f32,
    /// 1 adds to the mask, 0 takes away from it.
    pub value: f32,
}

/// A heal of one layer: where its mask is painted, the layer takes its own
/// samples from `offset` pixels away.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerHeal {
    /// The layer healed, as an index into [`Picture::layers`].
    pub layer: usize,
    /// Whether the heal copies the input's samples before the transfer, as
    /// a heal of an input does, or the layer's values after it.
    pub before_transfer: bool,
    pub offset: [i64; 2],
    /// The brightness added to what is copied.
    pub shift: f32,
    /// The heal's stroke, as an index into [`Picture::masks`].
    pub mask: usize,
}

/// What a step of the finished picture does.
#[derive(Clone, Debug, PartialEq)]
pub enum FinishKind {
    /// One of Seiza's stretches of the combined picture.
    Stretch(StretchPlan),
    /// Tone and color: a curve of the luminance applied with each pixel's
    /// color kept, then vibrance and saturation.
    Tone(crate::tone::Tone),
    /// A contrast curve, from -1 to 1.
    Curves(f32),
    /// Saturation; 1 leaves the color as it is.
    Saturation(f32),
    /// Where the step's mask is painted, the picture from `offset` pixels
    /// away, with `shift` added.
    Heal { offset: [i64; 2], shift: [f32; 3] },
}

/// One step of the finished picture.
#[derive(Clone, Debug, PartialEq)]
pub struct FinishStep {
    pub kind: FinishKind,
    /// How strongly the step applies, from 0 to 1.
    pub opacity: f32,
    /// A painted mask that limits the step. A heal's is its stroke.
    pub mask: Option<usize>,
}

/// A color picture: what [`crate::render`] renders.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Picture {
    /// The layers, bottom first.
    pub layers: Vec<Layer>,
    pub computed: Vec<ComputedMask>,
    /// Painted masks: each a stroke's segments, in the order painted.
    pub masks: Vec<Vec<Segment>>,
    pub heals: Vec<LayerHeal>,
    pub finish: Vec<FinishStep>,
    /// The primaries of the picture's colors: its layers' colors, the
    /// combined picture and every Finish step's result. Luminance is
    /// measured with its weights. sRGB unless a host chooses a wider one.
    pub space: crate::space::WorkingSpace,
    /// How far above SDR white the picture reaches, in stops, from 0 (an SDR
    /// picture) to [`crate::tone::MAX_HEADROOM`]. With headroom the picture
    /// is HDR all the way through: every layer and step may give values up
    /// to [`Self::peak`], and every Tone step is an HDR one with this
    /// headroom.
    pub headroom: f32,
}

impl Picture {
    /// The brightest value the picture shows. With [`Self::headroom`], the
    /// value on the extended sRGB curve of 2 to the headroom, for the
    /// combined picture and every step alike. Otherwise 1, SDR white, unless
    /// an HDR Tone step lifts it (see [`crate::tone::Tone::peak`]): then a
    /// later Curves or Stretch step over the whole picture brings it back to
    /// 1, and one limited by a mask or opacity leaves the brighter areas
    /// outside it as they were. A host presents and saves values only up to
    /// it.
    pub fn peak(&self) -> f32 {
        if self.headroom > 0.0 {
            let stops = self.headroom.min(crate::tone::MAX_HEADROOM);
            return crate::tone::extended_srgb_encode(stops.exp2());
        }
        let mut peak = 1.0_f32;
        // The steps in the stack at the top level, then the Finish steps;
        // one in a group changes only the group.
        let mut depth = 0_usize;
        let mut stacked = Vec::new();
        for layer in &self.layers {
            match &layer.kind {
                LayerKind::GroupBegin => depth += 1,
                LayerKind::GroupEnd => depth = depth.saturating_sub(1),
                LayerKind::Step(kind) if depth == 0 => stacked.push(FinishStep {
                    kind: kind.clone(),
                    opacity: layer.opacity,
                    mask: layer
                        .mask
                        .or(layer.clip)
                        .or(layer.computed.map(|_| usize::MAX)),
                }),
                _ => {}
            }
        }
        for step in stacked.iter().chain(&self.finish) {
            let whole = step.mask.is_none() && step.opacity >= 1.0;
            match &step.kind {
                FinishKind::Tone(tone) if whole => peak = tone.peak(),
                FinishKind::Tone(tone) => peak = peak.max(tone.peak()),
                FinishKind::Curves(_) | FinishKind::Stretch(_) if whole => peak = 1.0,
                _ => {}
            }
        }
        peak
    }

    /// A step as the picture renders it: with headroom, a Tone step is an
    /// HDR one with the picture's headroom.
    pub(crate) fn rendered_kind(&self, kind: &FinishKind) -> FinishKind {
        match kind {
            FinishKind::Tone(tone) if self.headroom > 0.0 => FinishKind::Tone(crate::tone::Tone {
                hdr: true,
                headroom: self.headroom,
                ..*tone
            }),
            other => other.clone(),
        }
    }

    /// The Finish steps as the picture renders them: with headroom, each
    /// Tone step is an HDR one with the picture's headroom.
    pub(crate) fn finish_steps(&self) -> std::borrow::Cow<'_, [FinishStep]> {
        if self.headroom <= 0.0 {
            return std::borrow::Cow::Borrowed(&self.finish);
        }
        std::borrow::Cow::Owned(
            self.finish
                .iter()
                .map(|step| match &step.kind {
                    FinishKind::Tone(tone) => FinishStep {
                        kind: FinishKind::Tone(crate::tone::Tone {
                            hdr: true,
                            headroom: self.headroom,
                            ..*tone
                        }),
                        ..step.clone()
                    },
                    _ => step.clone(),
                })
                .collect(),
        )
    }

    /// The inputs the picture reads, each once, in no particular order.
    pub fn inputs(&self) -> Vec<usize> {
        let mut inputs = Vec::new();
        let mut note = |read: &InputRead| {
            inputs.push(read.input);
            if let Some((other, _)) = read.blended {
                inputs.push(other);
            }
            inputs.extend(read.transfer.guides());
        };
        for layer in &self.layers {
            match &layer.kind {
                LayerKind::Input { read, .. } => note(read),
                LayerKind::Luminance {
                    read: Some(read), ..
                } => note(read),
                _ => {}
            }
        }
        for mask in &self.computed {
            mask.reads.iter().for_each(&mut note);
        }
        inputs.sort_unstable();
        inputs.dedup();
        inputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tone::{Tone, extended_srgb_encode};

    fn step(kind: FinishKind, opacity: f32, mask: Option<usize>) -> FinishStep {
        FinishStep {
            kind,
            opacity,
            mask,
        }
    }

    #[test]
    fn the_peak_follows_the_last_step_that_sets_the_range() {
        let hdr = FinishKind::Tone(Tone {
            hdr: true,
            headroom: 2.0,
            ..Tone::default()
        });
        let lifted = extended_srgb_encode(4.0);
        let mut picture = Picture::default();
        assert_eq!(picture.peak(), 1.0);
        picture.finish.push(step(hdr.clone(), 1.0, None));
        assert_eq!(picture.peak(), lifted);
        // A heal or a saturation step leaves the range as it is.
        picture
            .finish
            .push(step(FinishKind::Saturation(1.2), 1.0, None));
        assert_eq!(picture.peak(), lifted);
        // Curves under a mask leaves the bright areas outside it.
        picture
            .finish
            .push(step(FinishKind::Curves(0.3), 1.0, Some(0)));
        assert_eq!(picture.peak(), lifted);
        // Curves over the whole picture clips it at SDR white.
        picture
            .finish
            .push(step(FinishKind::Curves(0.3), 1.0, None));
        assert_eq!(picture.peak(), 1.0);
        // An SDR Tone step over the whole picture does too.
        picture.finish[3] = step(FinishKind::Tone(Tone::default()), 1.0, None);
        assert_eq!(picture.peak(), 1.0);
        // An HDR step at half opacity lifts the peak.
        picture.finish.push(step(hdr, 0.5, None));
        assert_eq!(picture.peak(), lifted);
        // A picture with headroom of its own reaches it all the way through,
        // whatever its steps.
        let own = Picture {
            headroom: 2.0,
            ..Picture::default()
        };
        assert_eq!(own.peak(), lifted);
        picture.headroom = 2.0;
        picture
            .finish
            .push(step(FinishKind::Curves(0.3), 1.0, None));
        assert_eq!(picture.peak(), lifted);
    }
}
