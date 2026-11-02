//! A stack's statistics on one grid, in a file that describes itself, and
//! their combination.

use crate::digest::KeyHasher;
use crate::{Digest, Error, Estimator, Identity, Result, StackKind, WeightSource};
use rayon::prelude::*;
use seiza_fits::{HeaderValue, ImagePart, ImageSamples, Pixels, WriteHeaderCard};
use seiza_stacking::{LinearImage, StackSnapshot};
use std::path::Path;

/// The names of a stack file's images after the mean, which is the image
/// any viewer opens.
const WEIGHT: &str = "WEIGHT";
const COVER: &str = "COVER";
const REJECT: &str = "REJECT";
const VARIANCE: &str = "VARIANCE";

/// One stack on one grid, as distributed stacking moves it between steps:
/// the weighted mean and the sums that let it be combined with others, with
/// the [`Identity`] that says what made it.
///
/// A weighted mean is a ratio of two sums. A stack that keeps its weight
/// beside its mean can be combined with another on the same grid into
/// exactly the mean one stack of both sets of frames would give, apart from
/// rejection, which [`Self::combine`] does not repeat. A partial stack holds
/// one session's frames; a tile stack the partial stacks of one grid,
/// combined.
///
/// Samples are interleaved as [`LinearImage`] holds them: one value per
/// channel per pixel, row by row.
#[derive(Clone, Debug)]
pub struct PartialStack {
    /// What the stack is and what made it.
    pub identity: Identity,
    /// The weighted mean, `NaN` where no frame reached.
    pub mean: LinearImage,
    /// Each sample's sum of frame weights, each relative to the reference
    /// frame; where frames weigh the same, the coverage.
    pub weight: Vec<f32>,
    /// Each sample's integrated frames.
    pub coverage: Vec<u32>,
    /// Each sample's rejected frames, when known.
    pub rejected: Option<Vec<u32>>,
    /// Each sample's variance of one frame of weight 1 about the mean, as
    /// the stack measured it from its own frames; 0 where fewer than two
    /// reached.
    pub variance: Option<Vec<f32>>,
    /// Cards carried from the reference frame: its WCS, and the few that
    /// describe the instrument and filter ([`carried_cards`]).
    pub headers: Vec<(String, HeaderValue)>,
}

impl PartialStack {
    /// A stack from a finished [`StackSnapshot`], live or reintegrated, with
    /// `reference_headers`' WCS and instrument cards, and `identity`.
    pub fn from_snapshot(
        snapshot: StackSnapshot,
        reference_headers: &[(String, HeaderValue)],
        identity: Identity,
    ) -> Result<Self> {
        identity.check()?;
        let weight = snapshot.weight_sum.unwrap_or_else(|| {
            snapshot
                .coverage
                .iter()
                .map(|&count| count as f32)
                .collect()
        });
        let stack = Self {
            identity,
            mean: snapshot.image,
            weight,
            coverage: snapshot.coverage,
            rejected: Some(snapshot.rejected_samples),
            variance: Some(snapshot.variance.data),
            headers: carried_cards(reference_headers),
        };
        stack.check_shape()?;
        Ok(stack)
    }

    /// Write the stack to a FITS or XISF file, as `path`'s extension names,
    /// atomically: the mean first, which any viewer opens as an ordinary
    /// stack, with the identity and carried cards, then `WEIGHT` (`f32`),
    /// `COVER`, and `REJECT` and `VARIANCE` when held. Counts are unsigned
    /// 16-bit, or `f32` where one passes 65535, as a deep tile's can; a count
    /// past 2²⁴, which `f32` cannot hold exactly, is refused.
    pub fn write(&self, path: &Path) -> Result<()> {
        self.identity.check()?;
        self.check_shape()?;
        let LinearImage {
            width,
            height,
            channels,
            ..
        } = self.mean;
        let floats = |samples| {
            ImageSamples::F32(if channels == 3 {
                seiza_fits::F32ImageData::RgbInterleaved(samples)
            } else {
                seiza_fits::F32ImageData::Mono(samples)
            })
        };
        let mut cards = self.identity.cards();
        for (key, value) in &self.headers {
            if let Some(card) = carried_card(key, value) {
                cards.push(card);
            }
        }
        let cover = PlanarCounts::of(&self.coverage, channels, COVER)?;
        let reject = self
            .rejected
            .as_ref()
            .map(|rejected| PlanarCounts::of(rejected, channels, REJECT))
            .transpose()?;
        let image = |name, samples| ImagePart {
            name,
            width,
            height,
            samples,
            headers: &[],
        };
        let mut images = vec![
            ImagePart {
                headers: &cards,
                ..image(None, floats(&self.mean.data))
            },
            image(Some(WEIGHT), floats(&self.weight)),
            image(Some(COVER), cover.samples(channels)),
        ];
        if let Some(reject) = &reject {
            images.push(image(Some(REJECT), reject.samples(channels)));
        }
        if let Some(variance) = &self.variance {
            images.push(image(Some(VARIANCE), floats(variance)));
        }
        seiza_image_io::save_images(path, &images)?;
        Ok(())
    }

    /// Read a stack [`Self::write`] wrote.
    pub fn open(path: &Path) -> Result<Self> {
        let primary = seiza_image_io::read_astronomy(path)?.image;
        let identity = Identity::from_cards(&primary.headers)?;
        let headers = carried_cards(&primary.headers);
        let (width, height, channels) = (primary.width, primary.height, primary.planes);
        let mean = LinearImage::new(
            width,
            height,
            channels,
            interleaved(primary.into_physical_f32(), channels),
        )?;
        let named = |name: &str| -> Result<Option<seiza_fits::FitsImage>> {
            match seiza_image_io::read_astronomy_named(path, name) {
                Ok(read) => {
                    let image = read.image;
                    if (image.width, image.height, image.planes) != (width, height, channels) {
                        return Err(Error::Invalid(format!(
                            "{} holds a {name} image of another shape than its mean",
                            path.display()
                        )));
                    }
                    Ok(Some(image))
                }
                Err(error) if is_missing(&error) => Ok(None),
                Err(error) => Err(error.into()),
            }
        };
        let required = |name: &str| {
            named(name)?
                .ok_or_else(|| Error::Invalid(format!("{} has no {name} image", path.display())))
        };
        let floats =
            |image: seiza_fits::FitsImage| interleaved(image.into_physical_f32(), channels);
        let counts = |image: seiza_fits::FitsImage| {
            let values = match image.pixels {
                Pixels::U16(values) => values.into_iter().map(u32::from).collect(),
                _ => image
                    .into_physical_f32()
                    .into_iter()
                    .map(|value| value.max(0.0) as u32)
                    .collect(),
            };
            interleaved(values, channels)
        };
        let stack = Self {
            identity,
            mean,
            weight: floats(required(WEIGHT)?),
            coverage: counts(required(COVER)?),
            rejected: named(REJECT)?.map(counts),
            variance: named(VARIANCE)?.map(floats),
            headers,
        };
        stack.check_shape()?;
        Ok(stack)
    }

    /// Read only the identity of the stack file at `path`, from its header,
    /// without decoding pixels: how a host learns whether a step is done.
    pub fn read_identity(path: &Path) -> Result<Identity> {
        Identity::from_cards(&seiza_image_io::read_header(path)?.cards)
    }

    /// Combine stacks on one grid into a tile stack: at each sample, the
    /// mean of their means weighted by their weights, Σwx / Σw, with the
    /// weights, coverage and rejection summed and the variances pooled.
    ///
    /// Without rejection that is the mean one stack of all their frames
    /// would give. Nothing is rejected here: a stack that disagrees with the
    /// others is a whole-session fault, to be left out, not clipped pixel
    /// by pixel. The stacks are taken in the order of their step keys, so
    /// the result does not depend on the order they are given in, and its
    /// key is a digest of theirs. Stacks on another grid, made with other
    /// options or of another shape are refused: their weights and scales are
    /// not comparable.
    pub fn combine(stacks: &[PartialStack]) -> Result<Self> {
        let mut ordered = stacks.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|stack| stack.identity.step_key);
        let identities = ordered
            .iter()
            .map(|stack| stack.identity.clone())
            .collect::<Vec<_>>();
        check_combinable(&identities)?;
        let mut combiner = Combiner::new(ordered[0])?;
        for stack in ordered {
            combiner.add(stack)?;
        }
        Ok(combiner.finish(&identities))
    }

    /// [`Self::combine`] the stack files at `paths`, reading one at a time:
    /// their headers first, to take them in step-key order, then each whole
    /// in turn, so memory does not grow with their number. Gives exactly
    /// what [`Self::combine`] gives the same stacks.
    pub fn combine_files(paths: &[impl AsRef<Path>]) -> Result<Self> {
        let mut ordered = paths
            .iter()
            .map(|path| Ok((Self::read_identity(path.as_ref())?, path.as_ref())))
            .collect::<Result<Vec<_>>>()?;
        ordered.sort_by_key(|(identity, _)| identity.step_key);
        let identities = ordered
            .iter()
            .map(|(identity, _)| identity.clone())
            .collect::<Vec<_>>();
        check_combinable(&identities)?;
        let mut combiner = None;
        for (identity, path) in &ordered {
            let stack = Self::open(path)?;
            if stack.identity != *identity {
                return Err(Error::Invalid(format!(
                    "{} changed while it was being combined",
                    path.display()
                )));
            }
            match &mut combiner {
                None => combiner.insert(Combiner::new(&stack)?).add(&stack)?,
                Some(combiner) => combiner.add(&stack)?,
            }
        }
        Ok(combiner.expect("checked non-empty").finish(&identities))
    }

    /// An error unless every per-sample layer matches the mean.
    fn check_shape(&self) -> Result<()> {
        let samples = self.mean.data.len();
        let lengths = [
            Some(self.weight.len()),
            Some(self.coverage.len()),
            self.rejected.as_ref().map(Vec::len),
            self.variance.as_ref().map(Vec::len),
        ];
        if lengths.iter().flatten().any(|&length| length != samples) {
            return Err(Error::Invalid(
                "a stack's layers must each hold one value per sample of its mean".into(),
            ));
        }
        Ok(())
    }
}

/// An error unless stacks with these identities, in step-key order, can be
/// combined: at least one, none given twice, all on one grid and made with
/// the same options.
fn check_combinable(identities: &[Identity]) -> Result<()> {
    let Some(first) = identities.first() else {
        return Err(Error::Invalid("there are no stacks to combine".into()));
    };
    if identities
        .windows(2)
        .any(|pair| pair[0].step_key == pair[1].step_key)
    {
        return Err(Error::Invalid(
            "a stack is given twice: two stacks share a step key".into(),
        ));
    }
    for identity in identities {
        if identity.grid != first.grid {
            return Err(Error::Incompatible(format!(
                "stack {} is not on the grid of stack {}",
                identity.step_key, first.step_key
            )));
        }
        if identity.options != first.options {
            return Err(Error::Incompatible(format!(
                "stack {} was made with other options than stack {}",
                identity.step_key, first.step_key
            )));
        }
    }
    Ok(())
}

/// Running sums over stacks added one at a time, in step-key order: per
/// sample, the summed weight, the weighted mean and the weighted sum of
/// squared deviations, each stack merged in by Chan's pairwise update in
/// `f64`, and the summed counts.
struct Combiner {
    width: usize,
    height: usize,
    channels: usize,
    weight: Vec<f64>,
    mean: Vec<f64>,
    squares: Vec<f64>,
    coverage: Vec<u32>,
    /// `None` once a stack without rejection counts is added.
    rejected: Option<Vec<u32>>,
    /// Whether every stack added so far holds its variance.
    pooled: bool,
    headers: Vec<(String, HeaderValue)>,
}

impl Combiner {
    fn new(first: &PartialStack) -> Result<Self> {
        let samples = first.mean.data.len();
        Ok(Self {
            width: first.mean.width,
            height: first.mean.height,
            channels: first.mean.channels,
            weight: vec![0.0; samples],
            mean: vec![0.0; samples],
            squares: vec![0.0; samples],
            coverage: vec![0; samples],
            rejected: Some(vec![0; samples]),
            pooled: true,
            headers: first.headers.clone(),
        })
    }

    fn add(&mut self, stack: &PartialStack) -> Result<()> {
        stack.check_shape()?;
        if (stack.mean.width, stack.mean.height, stack.mean.channels)
            != (self.width, self.height, self.channels)
        {
            return Err(Error::Incompatible(format!(
                "stack {} is not of the others' shape",
                stack.identity.step_key
            )));
        }
        self.pooled &= stack.variance.is_some();
        let variance = stack.variance.as_deref();
        self.weight
            .par_iter_mut()
            .zip(self.mean.par_iter_mut())
            .zip(self.squares.par_iter_mut())
            .zip(self.coverage.par_iter_mut())
            .enumerate()
            .for_each(|(index, (((weight, mean), squares), coverage))| {
                let part_weight = f64::from(stack.weight[index]);
                let part_count = stack.coverage[index];
                if part_count == 0 || part_weight <= 0.0 {
                    return;
                }
                let part_mean = f64::from(stack.mean.data[index]);
                // The stack's own weighted sum of squared deviations.
                let part_squares = variance.map_or(0.0, |variance| {
                    f64::from(variance[index]) * f64::from(part_count - 1)
                });
                let total = *weight + part_weight;
                let delta = part_mean - *mean;
                *squares += part_squares + delta * delta * *weight * part_weight / total;
                *mean += delta * part_weight / total;
                *weight = total;
                *coverage = coverage.saturating_add(part_count);
            });
        self.rejected = match (self.rejected.take(), &stack.rejected) {
            (Some(mut sums), Some(rejected)) => {
                for (sum, &count) in sums.iter_mut().zip(rejected) {
                    *sum = sum.saturating_add(count);
                }
                Some(sums)
            }
            _ => None,
        };
        Ok(())
    }

    fn finish(self, identities: &[Identity]) -> PartialStack {
        let references = identities.iter().collect::<Vec<_>>();
        let mean = self
            .mean
            .iter()
            .zip(&self.coverage)
            .map(|(&mean, &count)| if count == 0 { f32::NAN } else { mean as f32 })
            .collect();
        let variance = self.pooled.then(|| {
            self.squares
                .iter()
                .zip(&self.coverage)
                .map(|(&squares, &count)| {
                    if count > 1 {
                        (squares / f64::from(count - 1)) as f32
                    } else {
                        0.0
                    }
                })
                .collect()
        });
        PartialStack {
            identity: tile_identity(&references),
            mean: LinearImage {
                width: self.width,
                height: self.height,
                channels: self.channels,
                data: mean,
            },
            weight: self.weight.iter().map(|&weight| weight as f32).collect(),
            coverage: self.coverage,
            rejected: self.rejected,
            variance,
            headers: self.headers,
        }
    }
}

/// The identity of the tile [`PartialStack::combine`] makes of stacks with
/// these identities, given in step-key order: what a store finds the tile
/// under before any pixel is read. Its key is a digest of theirs.
pub(crate) fn tile_identity(identities: &[&Identity]) -> Identity {
    let first = identities[0];
    let stacking_version = if identities
        .iter()
        .all(|identity| identity.stacking_version == first.stacking_version)
    {
        first.stacking_version.clone()
    } else {
        "mixed".into()
    };
    let mut inputs = KeyHasher::new("seiza-pipeline-tile-inputs-v1");
    inputs.number(identities.len() as u64);
    for identity in identities {
        inputs.digest(&identity.step_key);
    }
    let inputs = inputs.finish();
    let options = KeyHasher::new("seiza-pipeline-combine-v1")
        .text("inverse-variance")
        .finish();
    let step_key = KeyHasher::new("seiza-pipeline-tile-v1")
        .text(crate::OUTPUT_VERSION)
        .digest(&first.grid)
        .digest(&options)
        .digest(&inputs)
        .finish();
    Identity {
        kind: StackKind::Tile,
        stacking_version,
        pipeline_version: crate::OUTPUT_VERSION.into(),
        step_key,
        inputs,
        input_count: identities.len() as u32,
        combined: identities
            .iter()
            .fold(0_u32, |sum, identity| sum.saturating_add(identity.combined)),
        calibration: None,
        options,
        host: None,
        estimator: if identities
            .iter()
            .all(|identity| identity.estimator == Estimator::Additive)
        {
            Estimator::Additive
        } else {
            Estimator::Clipped
        },
        context: None,
        contributor: None,
        weight_source: if identities
            .iter()
            .all(|identity| identity.weight_source == WeightSource::Stack)
        {
            WeightSource::Stack
        } else {
            WeightSource::Measured
        },
        grid: first.grid,
    }
}

/// The cards a stack file keeps from its reference frame: the WCS, SIP
/// distortion included, and the cards that say what took the frames,
/// through which filter, at what binning and row order.
///
/// Observer names, site coordinates and file paths are left out: partial
/// stacks are made to be shared, and capture software writes the observer's
/// location into every frame.
pub fn carried_cards(headers: &[(String, HeaderValue)]) -> Vec<(String, HeaderValue)> {
    headers
        .iter()
        .filter(|(key, value)| is_carried(key) && carried_card(key, value).is_some())
        .cloned()
        .collect()
}

fn is_carried(key: &str) -> bool {
    let indexed = |prefixes: &[&str]| {
        prefixes.iter().any(|prefix| {
            key.strip_prefix(prefix).is_some_and(|rest| {
                !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit() || b == b'_')
            })
        })
    };
    !crate::identity::KEYWORDS.contains(&key)
        && (matches!(
            key,
            "WCSAXES"
                | "RADESYS"
                | "EQUINOX"
                | "LONPOLE"
                | "LATPOLE"
                | "A_ORDER"
                | "B_ORDER"
                | "AP_ORDER"
                | "BP_ORDER"
                | "FILTER"
                | "INSTRUME"
                | "TELESCOP"
                | "XBINNING"
                | "YBINNING"
                | "XPIXSZ"
                | "YPIXSZ"
                | "FOCALLEN"
                | "ROWORDER"
                | "SEIZABIN"
        ) || indexed(&[
            "CRPIX", "CRVAL", "CTYPE", "CUNIT", "CDELT", "CROTA", "CD", "PC", "PV", "A_", "B_",
            "AP_", "BP_",
        ]))
}

/// The card writing `value` under `key`, unless it cannot be written.
fn carried_card(key: &str, value: &HeaderValue) -> Option<WriteHeaderCard> {
    match value {
        HeaderValue::Float(value) if !value.is_finite() => None,
        HeaderValue::Raw(value) if value.is_empty() => None,
        _ if key.len() > 8 => None,
        _ => Some(WriteHeaderCard::new(key, value.clone())),
    }
}

/// Interleaved counts as planar samples to write: unsigned 16-bit when
/// every count fits, `f32` otherwise.
enum PlanarCounts {
    U16(Vec<u16>),
    F32(Vec<f32>),
}

impl PlanarCounts {
    /// `counts` planar, refusing a count `f32` cannot hold exactly.
    fn of(counts: &[u32], channels: usize, name: &str) -> Result<Self> {
        const EXACT_IN_F32: u32 = 1 << 24;
        let largest = counts.iter().copied().max().unwrap_or(0);
        if largest > EXACT_IN_F32 {
            return Err(Error::Invalid(format!(
                "{name} holds {largest} frames at a sample, more than a stack file can"
            )));
        }
        let pixels = counts.len() / channels;
        let place = |index: usize| (index % channels) * pixels + index / channels;
        Ok(if largest <= u32::from(u16::MAX) {
            let mut planar = vec![0_u16; counts.len()];
            for (index, &count) in counts.iter().enumerate() {
                planar[place(index)] = count as u16;
            }
            Self::U16(planar)
        } else {
            let mut planar = vec![0.0_f32; counts.len()];
            for (index, &count) in counts.iter().enumerate() {
                planar[place(index)] = count as f32;
            }
            Self::F32(planar)
        })
    }

    fn samples(&self, channels: usize) -> ImageSamples<'_> {
        match self {
            Self::U16(samples) => ImageSamples::U16 {
                samples,
                planes: channels,
            },
            Self::F32(samples) => ImageSamples::F32(if channels == 3 {
                seiza_fits::F32ImageData::RgbPlanar(samples)
            } else {
                seiza_fits::F32ImageData::Mono(samples)
            }),
        }
    }
}

/// Planar samples, as FITS and XISF hold them, interleaved.
fn interleaved<T: Copy + Default>(planar: Vec<T>, channels: usize) -> Vec<T> {
    if channels == 1 {
        return planar;
    }
    let pixels = planar.len() / channels;
    let mut values = vec![T::default(); planar.len()];
    for (index, value) in values.iter_mut().enumerate() {
        *value = planar[(index % channels) * pixels + index / channels];
    }
    values
}

/// Whether reading a named image failed because the file holds none.
fn is_missing(error: &seiza_image_io::Error) -> bool {
    matches!(
        error.cause(),
        seiza_image_io::Cause::Fits(seiza_fits::FitsError::ImageNotFound(_))
            | seiza_image_io::Cause::Xisf(seiza_xisf::XisfError::ImageNotFound(_))
    )
}

/// The digest of a stack's grid on today's single reference: the image it
/// aligns to, the bin factor, and the light it is normalized and weighted
/// against with that light's recipe. Weights and scales are relative to
/// that light, so stacks made against another are not on this grid.
pub(crate) fn reference_grid(
    image: &Digest,
    bin_factor: u32,
    light: &Digest,
    recipe: &Digest,
) -> Digest {
    KeyHasher::new("seiza-pipeline-grid-v1")
        .text("reference")
        .digest(image)
        .number(u64::from(bin_factor))
        .digest(light)
        .digest(recipe)
        .finish()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::identity::tests::identity;

    /// A stack of `channels` channels over a 4 by 3 grid, its values set by
    /// `seed`, with a sample no frame reached.
    pub(crate) fn stack(channels: usize, seed: u32) -> PartialStack {
        let samples = 4 * 3 * channels;
        let value = |index: usize, scale: f32| {
            (((index as u32).wrapping_mul(2654435761) ^ seed.wrapping_mul(40503)) % 1000) as f32
                * scale
        };
        let mut coverage = (0..samples)
            .map(|index| 2 + (index as u32 + seed) % 5)
            .collect::<Vec<_>>();
        coverage[5] = 0;
        let weight = coverage
            .iter()
            .enumerate()
            .map(|(index, &count)| count as f32 * (0.5 + value(index, 0.001)))
            .collect::<Vec<_>>();
        let mean = (0..samples)
            .map(|index| {
                if coverage[index] == 0 {
                    f32::NAN
                } else {
                    100.0 + value(index, 0.01)
                }
            })
            .collect();
        PartialStack {
            identity: Identity {
                step_key: Digest::of_bytes(&seed.to_le_bytes()),
                ..identity()
            },
            mean: LinearImage::new(4, 3, channels, mean).unwrap(),
            weight,
            rejected: Some(coverage.iter().map(|count| count % 2).collect()),
            variance: Some((0..samples).map(|index| value(index, 0.003)).collect()),
            coverage,
            headers: vec![
                ("CTYPE1".into(), HeaderValue::String("RA---TAN".into())),
                ("FILTER".into(), HeaderValue::String("Ha".into())),
            ],
        }
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// Two stacks hold the same, bit for bit, `NaN`s included.
    pub(crate) fn assert_same(actual: &PartialStack, expected: &PartialStack) {
        assert_eq!(actual.identity, expected.identity);
        assert_eq!(
            (actual.mean.width, actual.mean.height, actual.mean.channels),
            (
                expected.mean.width,
                expected.mean.height,
                expected.mean.channels
            )
        );
        assert_eq!(bits(&actual.mean.data), bits(&expected.mean.data));
        assert_eq!(bits(&actual.weight), bits(&expected.weight));
        assert_eq!(actual.coverage, expected.coverage);
        assert_eq!(actual.rejected, expected.rejected);
        assert_eq!(
            actual.variance.as_deref().map(bits),
            expected.variance.as_deref().map(bits)
        );
        assert_eq!(actual.headers, expected.headers);
    }

    #[test]
    fn stacks_round_trip_through_fits_and_xisf() {
        let directory = tempfile::tempdir().unwrap();
        for channels in [1, 3] {
            for name in ["partial.fits", "partial.xisf"] {
                let path = directory.path().join(format!("{channels}-{name}"));
                let written = stack(channels, 7);
                written.write(&path).unwrap();
                assert_same(&PartialStack::open(&path).unwrap(), &written);
                assert_eq!(
                    PartialStack::read_identity(&path).unwrap(),
                    written.identity
                );
            }
        }
        // Rejection and variance are optional.
        let path = directory.path().join("lean.fits");
        let lean = PartialStack {
            rejected: None,
            variance: None,
            ..stack(1, 3)
        };
        lean.write(&path).unwrap();
        assert_same(&PartialStack::open(&path).unwrap(), &lean);
    }

    #[test]
    fn deep_counts_are_written_as_floats_and_a_short_layer_is_refused() {
        let directory = tempfile::tempdir().unwrap();
        for (channels, name) in [(1, "deep.fits"), (3, "deep.xisf")] {
            let path = directory.path().join(name);
            let mut deep = stack(channels, 1);
            deep.coverage[0] = 70_000;
            deep.rejected.as_mut().unwrap()[1] = 16_777_216;
            deep.write(&path).unwrap();
            assert_same(&PartialStack::open(&path).unwrap(), &deep);
        }
        let path = directory.path().join("refused.fits");
        let mut deeper = stack(1, 1);
        deeper.coverage[0] = 16_777_217;
        assert!(matches!(deeper.write(&path), Err(Error::Invalid(_))));
        let mut short = stack(1, 1);
        short.weight.pop();
        assert!(matches!(short.write(&path), Err(Error::Invalid(_))));
        assert!(!path.exists());
    }

    #[test]
    fn carried_cards_keep_the_wcs_and_instrument_but_not_the_observer() {
        let headers = [
            ("CRPIX1", HeaderValue::Float(10.5)),
            ("CD1_2", HeaderValue::Float(1e-4)),
            ("A_ORDER", HeaderValue::Integer(2)),
            ("A_0_2", HeaderValue::Float(1e-6)),
            ("FILTER", HeaderValue::String("Ha".into())),
            ("OBSERVER", HeaderValue::String("me".into())),
            ("SITELAT", HeaderValue::Float(45.0)),
            ("CRVAL", HeaderValue::Float(1.0)),
            ("STEPKEY", HeaderValue::String("stale".into())),
            ("CRVAL2", HeaderValue::Float(f64::NAN)),
        ]
        .map(|(key, value)| (key.to_string(), value));
        let kept = carried_cards(&headers)
            .into_iter()
            .map(|(key, _)| key)
            .collect::<Vec<_>>();
        assert_eq!(kept, ["CRPIX1", "CD1_2", "A_ORDER", "A_0_2", "FILTER"]);
    }

    #[test]
    fn combining_weighs_each_stack_by_its_weight() {
        for channels in [1, 3] {
            let stacks = [stack(channels, 1), stack(channels, 2), stack(channels, 3)];
            let tile = PartialStack::combine(&stacks).unwrap();
            assert_eq!(tile.identity.kind, StackKind::Tile);
            assert_eq!(tile.identity.input_count, 3);
            assert_eq!(tile.identity.combined, 33);
            for index in 0..tile.mean.data.len() {
                let reached = stacks.iter().filter(|stack| stack.coverage[index] > 0);
                let weight = reached
                    .clone()
                    .map(|stack| f64::from(stack.weight[index]))
                    .sum::<f64>();
                let sum = reached
                    .clone()
                    .map(|stack| f64::from(stack.weight[index]) * f64::from(stack.mean.data[index]))
                    .sum::<f64>();
                if weight == 0.0 {
                    assert!(tile.mean.data[index].is_nan());
                    assert_eq!(tile.coverage[index], 0);
                    continue;
                }
                assert!((f64::from(tile.mean.data[index]) - sum / weight).abs() < 1e-3);
                assert!((f64::from(tile.weight[index]) - weight).abs() < 1e-3);
                assert_eq!(
                    tile.coverage[index],
                    reached.map(|stack| stack.coverage[index]).sum::<u32>()
                );
            }
            // The order the stacks arrive in does not matter, to the bit.
            let reversed =
                PartialStack::combine(&[stacks[2].clone(), stacks[0].clone(), stacks[1].clone()])
                    .unwrap();
            assert_eq!(bits(&reversed.mean.data), bits(&tile.mean.data));
            assert_eq!(
                bits(reversed.variance.as_deref().unwrap()),
                bits(tile.variance.as_deref().unwrap())
            );
            assert_eq!(reversed.identity, tile.identity);
        }
    }

    /// Pooled moments are those of every frame at once: two stacks of
    /// equal-weight frames combine into the mean and sample variance of
    /// their frames together.
    #[test]
    fn combining_pools_variances_as_one_stack_of_every_frame() {
        let frames = [[1.0_f64, 2.0, 4.0], [7.0, 8.0, 12.0]];
        let moments = |values: &[f64]| {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let squares = values
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>();
            (mean, squares / (values.len() - 1) as f64)
        };
        let part = |seed: u32, values: &[f64]| {
            let (mean, variance) = moments(values);
            PartialStack {
                identity: Identity {
                    step_key: Digest::of_bytes(&seed.to_le_bytes()),
                    ..identity()
                },
                mean: LinearImage::new(1, 1, 1, vec![mean as f32]).unwrap(),
                weight: vec![values.len() as f32],
                coverage: vec![values.len() as u32],
                rejected: None,
                variance: Some(vec![variance as f32]),
                headers: Vec::new(),
            }
        };
        let tile = PartialStack::combine(&[part(1, &frames[0]), part(2, &frames[1])]).unwrap();
        let (mean, variance) = moments(&frames.concat());
        assert!((f64::from(tile.mean.data[0]) - mean).abs() < 1e-6);
        assert!((f64::from(tile.variance.unwrap()[0]) - variance).abs() < 1e-5);
        assert!(tile.rejected.is_none());
    }

    #[test]
    fn stacks_on_another_grid_or_given_twice_are_refused() {
        let other = PartialStack {
            identity: Identity {
                grid: Digest::of_bytes(b"another grid"),
                ..stack(1, 2).identity
            },
            ..stack(1, 2)
        };
        assert!(matches!(
            PartialStack::combine(&[stack(1, 1), other]),
            Err(Error::Incompatible(_))
        ));
        assert!(matches!(
            PartialStack::combine(&[stack(1, 1), stack(3, 2)]),
            Err(Error::Incompatible(_))
        ));
        // Weights made under other options are not comparable.
        let reweighted = PartialStack {
            identity: Identity {
                options: Digest::of_bytes(b"other weighting"),
                ..stack(1, 2).identity
            },
            ..stack(1, 2)
        };
        assert!(matches!(
            PartialStack::combine(&[stack(1, 1), reweighted]),
            Err(Error::Incompatible(_))
        ));
        assert!(PartialStack::combine(&[stack(1, 1), stack(1, 1)]).is_err());
        assert!(PartialStack::combine(&[]).is_err());
        assert!(PartialStack::combine_files(&[] as &[&Path]).is_err());
    }

    /// Combining files, one read at a time, gives what combining the same
    /// stacks in memory gives, to the bit.
    #[test]
    fn combining_files_matches_combining_in_memory() {
        let directory = tempfile::tempdir().unwrap();
        for channels in [1, 3] {
            let stacks = (1..=4)
                .map(|seed| stack(channels, seed))
                .collect::<Vec<_>>();
            let paths = stacks
                .iter()
                .enumerate()
                .map(|(index, stack)| {
                    let path = directory.path().join(format!("{channels}-{index}.fits"));
                    stack.write(&path).unwrap();
                    path
                })
                .collect::<Vec<_>>();
            let mut reversed = paths.clone();
            reversed.reverse();
            assert_same(
                &PartialStack::combine_files(&reversed).unwrap(),
                &PartialStack::combine(&stacks).unwrap(),
            );
        }
    }
}
