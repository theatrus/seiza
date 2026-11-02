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
    /// `COVER` (unsigned 16-bit), and `REJECT` and `VARIANCE` when held.
    /// Coverage or rejection above 65535 frames is refused.
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
        let cover = planar_counts(&self.coverage, channels, COVER)?;
        let reject = self
            .rejected
            .as_ref()
            .map(|rejected| planar_counts(rejected, channels, REJECT))
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
            image(
                Some(COVER),
                ImageSamples::U16 {
                    samples: &cover,
                    planes: channels,
                },
            ),
        ];
        if let Some(reject) = &reject {
            images.push(image(
                Some(REJECT),
                ImageSamples::U16 {
                    samples: reject,
                    planes: channels,
                },
            ));
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
    /// key is a digest of theirs. Stacks on another grid or of another
    /// shape are refused.
    pub fn combine(stacks: &[PartialStack]) -> Result<Self> {
        let Some(first) = stacks.first() else {
            return Err(Error::Invalid("there are no stacks to combine".into()));
        };
        for stack in stacks {
            stack.check_shape()?;
            if stack.identity.grid != first.identity.grid
                || (stack.mean.width, stack.mean.height, stack.mean.channels)
                    != (first.mean.width, first.mean.height, first.mean.channels)
            {
                return Err(Error::Incompatible(format!(
                    "stack {} is not on the grid of stack {}",
                    stack.identity.step_key, first.identity.step_key
                )));
            }
        }
        let mut ordered = stacks.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|stack| stack.identity.step_key);
        if ordered
            .windows(2)
            .any(|pair| pair[0].identity.step_key == pair[1].identity.step_key)
        {
            return Err(Error::Invalid(
                "a stack is given twice: two stacks share a step key".into(),
            ));
        }
        let pooled = ordered.iter().all(|stack| stack.variance.is_some());
        let samples = first.mean.data.len();
        let mut mean = vec![0.0_f32; samples];
        let mut weight = vec![0.0_f32; samples];
        let mut coverage = vec![0_u32; samples];
        let mut variance = vec![0.0_f32; samples];
        mean.par_iter_mut()
            .zip(weight.par_iter_mut())
            .zip(coverage.par_iter_mut())
            .zip(variance.par_iter_mut())
            .enumerate()
            .for_each(|(index, (((mean, weight), coverage), variance))| {
                let (mut sum, mut total, mut count) = (0.0_f64, 0.0_f64, 0_u32);
                for stack in &ordered {
                    let part_weight = f64::from(stack.weight[index]);
                    if stack.coverage[index] > 0 && part_weight > 0.0 {
                        sum += part_weight * f64::from(stack.mean.data[index]);
                        total += part_weight;
                        count = count.saturating_add(stack.coverage[index]);
                    }
                }
                *coverage = count;
                *weight = total as f32;
                if count == 0 {
                    *mean = f32::NAN;
                    return;
                }
                let combined = sum / total;
                *mean = combined as f32;
                if !pooled || count < 2 {
                    return;
                }
                // Each stack's sum of weighted squared deviations about its
                // own mean, and its weight times its mean's squared
                // deviation from the whole's.
                let mut squares = 0.0_f64;
                for stack in &ordered {
                    let part_count = stack.coverage[index];
                    let part_weight = f64::from(stack.weight[index]);
                    if part_count == 0 || part_weight <= 0.0 {
                        continue;
                    }
                    let part_variance = stack.variance.as_ref().expect("pooled")[index];
                    squares += f64::from(part_variance) * f64::from(part_count - 1);
                    let deviation = f64::from(stack.mean.data[index]) - combined;
                    squares += part_weight * deviation * deviation;
                }
                *variance = (squares / f64::from(count - 1)) as f32;
            });
        let rejected = ordered
            .iter()
            .map(|stack| stack.rejected.as_deref())
            .collect::<Option<Vec<_>>>()
            .map(|parts| {
                (0..samples)
                    .map(|index| {
                        parts
                            .iter()
                            .fold(0_u32, |sum, part| sum.saturating_add(part[index]))
                    })
                    .collect()
            });
        let identities = ordered
            .iter()
            .map(|stack| &stack.identity)
            .collect::<Vec<_>>();
        Ok(Self {
            identity: tile_identity(&identities),
            mean: LinearImage::new(
                first.mean.width,
                first.mean.height,
                first.mean.channels,
                mean,
            )?,
            weight,
            coverage,
            rejected,
            variance: pooled.then_some(variance),
            headers: ordered[0].headers.clone(),
        })
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

/// Interleaved counts as planar 16-bit samples, refusing a count above
/// 65535.
fn planar_counts(counts: &[u32], channels: usize, name: &str) -> Result<Vec<u16>> {
    let pixels = counts.len() / channels;
    let mut planar = vec![0_u16; counts.len()];
    for (index, &count) in counts.iter().enumerate() {
        let value = u16::try_from(count).map_err(|_| {
            Error::Invalid(format!(
                "{name} holds {count} frames at a sample, more than a 16-bit image can"
            ))
        })?;
        planar[(index % channels) * pixels + index / channels] = value;
    }
    Ok(planar)
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
/// aligns to and the bin factor.
pub(crate) fn reference_grid(image: &Digest, bin_factor: u32) -> Digest {
    KeyHasher::new("seiza-pipeline-grid-v1")
        .text("reference")
        .digest(image)
        .number(u64::from(bin_factor))
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
    fn a_count_above_sixteen_bits_or_a_short_layer_is_refused() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("deep.fits");
        let mut deep = stack(1, 1);
        deep.coverage[0] = 70_000;
        assert!(matches!(deep.write(&path), Err(Error::Invalid(_))));
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
        assert!(PartialStack::combine(&[stack(1, 1), stack(1, 1)]).is_err());
        assert!(PartialStack::combine(&[]).is_err());
    }
}
