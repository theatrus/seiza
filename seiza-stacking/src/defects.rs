//! Sensor defects learned from the lights themselves.
//!
//! A master dark subtracts a sensor's hot pixels. Without one, every light
//! keeps them, and they do more harm than a few bright dots: on a raw colour
//! frame demosaicing turns each into a small coloured blob that star
//! detection takes for a star, so registration can lock onto the sensor's
//! fixed pattern instead of the sky, and the online rejection lets the
//! reference and warm-up frames' hot pixels into the stack whatever the
//! dither.
//!
//! Dithering is what tells a defect from the sky: between frames the sky
//! moves across the sensor, and a defect does not. [`detect_defects`] looks
//! at a sample of the lights and flags a pixel only when it stands far from
//! its same-plane neighbours in most of them. A star's core lands on any one
//! sensor pixel only in the frames the dither puts it there, so stars stay
//! unflagged; and a pixel whose neighbours stand above the background around
//! them, as a star's do, is not flagged even when it stood out in every
//! frame. The resulting [`DefectMap`], carried by
//! [`crate::StackOptions::defects`], is repaired in every light after
//! calibration and before debayering, so registration, rejection and the
//! stack never see the defects.
//!
//! A steady hot pixel is repaired as a dark would repair it: the map keeps
//! what it reads above its neighbours, learned across the lights, and each
//! light has that subtracted, so a star the dither carries over the pixel
//! keeps its core. A dead pixel, or one whose excess wanders, takes the
//! median of its neighbours instead.

use crate::{BayerLayout, CalibrationMasters, Error, FitsFrame, LinearImage, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Consistency factor between the median absolute deviation and the
/// standard deviation of a normal distribution.
const MAD_TO_SIGMA: f32 = 1.4826;

/// The standard deviation, in units of the frame's deviation sigma, of the
/// difference between the median of a pixel's eight nearest same-plane
/// neighbours and the median of the sixteen beyond them, on flat noise.
const STRUCTURE_SIGMA: f32 = 0.5;

/// How far, in its own noise, the median over the frames that flagged a
/// pixel of its neighbours' height above the background beyond them may
/// lie from zero. A defect's neighbours are ordinary sky; a star that sat
/// still lifts them in every frame, while one the dither moved past lifts
/// them in too few to move the median.
const MAXIMUM_STRUCTURE: f32 = 4.0;

/// The standard deviation of the median of `n` normal samples is about
/// this many times theirs over the square root of `n`.
const MEDIAN_EFFICIENCY: f32 = 1.2533;

/// Fewest lights that can tell a defect from the sky.
const MINIMUM_DEFECT_FRAMES: u32 = 3;

/// How far a hot pixel's excess may wander between lights, beyond their
/// noise, as a fraction of the excess, for it to be subtracted rather than
/// replaced: dark current is steady; a flickering pixel is not.
const EXCESS_WANDER: f32 = 0.1;

/// How [`detect_defects`] decides which pixels are defects.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DefectDetectionOptions {
    /// How far a pixel must stand above or below the median of its eight
    /// same-plane neighbours to count as deviant in a frame, in both the
    /// frame's noise and the spread of the 24 samples around it.
    pub sigma: f32,
    /// Fraction of the examined lights in which a pixel must be deviant, in
    /// the same direction, to be a defect. The sky moves between dithered
    /// frames, so only the sensor's own defects are deviant in most of them.
    pub persistence: f32,
    /// Most lights examined; a longer list is sampled evenly across it. At
    /// least three.
    pub maximum_frames: usize,
    /// Whether each light's screen may run on the GPU, as
    /// [`crate::GpuOptions::defects`] says. The default is not serialized.
    #[serde(default, skip_serializing_if = "crate::GpuOptions::is_default")]
    pub gpu: crate::GpuOptions,
}

impl Default for DefectDetectionOptions {
    fn default() -> Self {
        Self {
            sigma: 5.0,
            persistence: 0.6,
            maximum_frames: 16,
            gpu: crate::GpuOptions::default(),
        }
    }
}

impl DefectDetectionOptions {
    fn validate(&self) -> Result<()> {
        if !self.sigma.is_finite()
            || self.sigma <= 0.0
            || !self.persistence.is_finite()
            || self.persistence <= 0.0
            || self.persistence > 1.0
            || self.maximum_frames < MINIMUM_DEFECT_FRAMES as usize
        {
            return Err(Error::Calibration(format!(
                "defect detection needs a positive finite sigma, a persistence in (0, 1], \
                 and at least {MINIMUM_DEFECT_FRAMES} frames"
            )));
        }
        self.gpu.validate()
    }
}

/// Defective pixels of one sensor geometry, which each light has repaired
/// before debayering.
///
/// Pixels are raw-frame indices `y * width + x`, ascending. A steady hot
/// pixel has its excess over its neighbours subtracted; every other pixel
/// takes the median of its same-plane neighbours. A frame of another size
/// is left as it is.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DefectMap {
    width: usize,
    height: usize,
    /// Lights the map was learned from; zero for a map given directly.
    #[serde(default)]
    frames: u32,
    pixels: Vec<u32>,
    /// Each pixel's steady excess over its neighbours, in the lights'
    /// calibrated units at [`Self::exposure_seconds`], or zero to take the
    /// neighbours' median; empty when every pixel takes the median.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    excess: Vec<f32>,
    /// The exposure the excess was learned at, which scales it to each
    /// light's own; unknown when a light had none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exposure_seconds: Option<f64>,
}

impl DefectMap {
    /// A map of the given raw-frame pixel indices, `y * width + x`, in any
    /// order; duplicates are dropped.
    pub fn new(width: usize, height: usize, mut pixels: Vec<u32>) -> Result<Self> {
        pixels.sort_unstable();
        pixels.dedup();
        let map = Self {
            width,
            height,
            frames: 0,
            pixels,
            excess: Vec::new(),
            exposure_seconds: None,
        };
        map.validate()?;
        Ok(map)
    }

    /// Raw frame width the map describes.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Raw frame height the map describes.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Lights the map was learned from; zero for a map given directly.
    pub fn frames(&self) -> u32 {
        self.frames
    }

    /// Defective pixels, as ascending raw-frame indices `y * width + x`.
    pub fn pixels(&self) -> &[u32] {
        &self.pixels
    }

    /// Each pixel's steady excess over its neighbours, subtracted from every
    /// light (scaled to its exposure), or zero where the neighbours' median
    /// replaces the pixel; empty when every pixel takes the median.
    pub fn excess(&self) -> &[f32] {
        &self.excess
    }

    /// The exposure, in seconds, the excess was learned at.
    pub fn exposure_seconds(&self) -> Option<f64> {
        self.exposure_seconds
    }

    /// Number of defective pixels.
    pub fn len(&self) -> usize {
        self.pixels.len()
    }

    /// Whether no pixel is defective.
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }

    /// Check that every pixel lies inside the frame, once, in order. A map
    /// with no pixels repairs nothing, whatever its frame.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.pixels.is_empty() {
            return Ok(());
        }
        let samples = self.width.checked_mul(self.height);
        if self.width == 0
            || self.height == 0
            || samples.is_none_or(|samples| samples > u32::MAX as usize)
        {
            return Err(Error::Calibration(
                "a defect map needs a nonempty frame of at most 2^32 pixels".into(),
            ));
        }
        let samples = samples.unwrap_or_default();
        if self.pixels.windows(2).any(|pair| pair[0] >= pair[1])
            || self
                .pixels
                .last()
                .is_some_and(|&last| last as usize >= samples)
        {
            return Err(Error::Calibration(
                "defect map pixels must be distinct indices inside the frame, ascending".into(),
            ));
        }
        if !(self.excess.is_empty() || self.excess.len() == self.pixels.len())
            || self
                .excess
                .iter()
                .any(|excess| !excess.is_finite() || *excess < 0.0)
            || self
                .exposure_seconds
                .is_some_and(|seconds| !seconds.is_finite() || seconds <= 0.0)
        {
            return Err(Error::Calibration(
                "a defect map's excess needs one finite, nonnegative value per pixel, and a \
                 positive finite exposure"
                    .into(),
            ));
        }
        Ok(())
    }

    fn contains(&self, pixel: usize) -> bool {
        self.pixels.binary_search(&(pixel as u32)).is_ok()
    }

    /// Repair each defective pixel. A steady hot pixel has its excess,
    /// scaled from [`Self::exposure_seconds`] to `exposure_seconds`,
    /// subtracted, unless that would leave it far below its neighbours, as a
    /// light of another temperature might; every other pixel takes the
    /// median of its nearest same-plane neighbours that are not defects
    /// themselves: the eight around it, or the sixteen beyond them when fewer
    /// than three of those are usable. A raw colour frame steps by two, so a
    /// red photosite takes only red neighbours. Returns the samples
    /// repaired; a frame of another size is left as it is.
    pub fn repair(
        &self,
        image: &mut LinearImage,
        bayer: Option<BayerLayout>,
        exposure_seconds: Option<f64>,
    ) -> Result<usize> {
        if self.pixels.is_empty() || image.width != self.width || image.height != self.height {
            return Ok(0);
        }
        if bayer.is_some() && image.channels != 1 {
            return Err(Error::Calibration(
                "a CFA defect repair needs a one-channel image".into(),
            ));
        }
        let step = if bayer.is_some() { 2 } else { 1 };
        let (width, height, channels) = (image.width, image.height, image.channels);
        let scale = match (self.exposure_seconds, exposure_seconds) {
            (Some(learned), Some(light)) if light.is_finite() && light > 0.0 => {
                (light / learned) as f32
            }
            _ => 1.0,
        };
        let mut replacements = Vec::with_capacity(self.pixels.len() * channels);
        let mut neighbors = Vec::with_capacity(16);
        let mut spread = Vec::with_capacity(16);
        for (index, &pixel) in self.pixels.iter().enumerate() {
            let (x, y) = (pixel as usize % width, pixel as usize / width);
            let excess = match channels {
                1 => self.excess.get(index).copied().unwrap_or(0.0) * scale,
                _ => 0.0,
            };
            for channel in 0..channels {
                for ring in 1..=2 {
                    neighbors.clear();
                    for (dx, dy) in ring_offsets(ring) {
                        let Some((nx, ny)) = offset(x, y, dx * step, dy * step, width, height)
                        else {
                            continue;
                        };
                        let neighbor = ny * width + nx;
                        let value = image.data[neighbor * channels + channel];
                        if value.is_finite() && !self.contains(neighbor) {
                            neighbors.push(value);
                        }
                    }
                    if neighbors.len() < 3 {
                        continue;
                    }
                    let sample = pixel as usize * channels + channel;
                    let median = median_in_place(&mut neighbors);
                    let subtracted = image.data[sample] - excess;
                    let repaired = if excess > 0.0 && subtracted.is_finite() {
                        // The median left the neighbours sorted, lowest first.
                        spread.clear();
                        spread.extend(neighbors.iter().map(|value| (value - median).abs()));
                        let scatter = MAD_TO_SIGMA * median_in_place(&mut spread);
                        if subtracted >= neighbors[0] - 3.0 * scatter {
                            subtracted
                        } else {
                            median
                        }
                    } else {
                        median
                    };
                    replacements.push((sample, repaired));
                    break;
                }
            }
        }
        // Every replacement reads only pixels that are not defects, so none
        // reads another's result.
        for &(index, value) in &replacements {
            image.data[index] = value;
        }
        Ok(replacements.len())
    }
}

/// Learn a sensor's hot and dead pixels from its lights.
///
/// Up to [`DefectDetectionOptions::maximum_frames`] of `paths`, evenly
/// spaced, are read, calibrated with `calibration` and examined. A light
/// that cannot be read or calibrated, or whose geometry differs from the
/// first one examined, is passed over; fewer than three examined lights
/// cannot tell a defect from the sky and give an empty map.
///
/// A pixel is a defect when it stands [`DefectDetectionOptions::sigma`]
/// from its same-plane neighbours, the same way, in
/// [`DefectDetectionOptions::persistence`] of the examined lights, and its
/// neighbours lie at the background around them. The lights must move on
/// the sensor between frames, by dithering or drift, for the map to hold
/// every defect and no star: a star that sat still is kept out by its
/// neighbours, unless it is so undersampled that it fills one pixel, and a
/// defect under one is kept in.
pub fn detect_defects(
    paths: &[PathBuf],
    calibration: &CalibrationMasters,
    options: &DefectDetectionOptions,
) -> Result<DefectMap> {
    options.validate()?;
    let chosen = evenly_spaced(paths.len(), options.maximum_frames);
    let mut detector = DefectDetector::new(*options);
    // A few frames at once keeps the cores busy while each is read.
    let concurrency = rayon::current_num_threads().clamp(1, 4);
    for batch in chosen.chunks(concurrency) {
        let found = batch
            .par_iter()
            .map(|&index| {
                let mut frame = FitsFrame::open(&paths[index]).ok()?;
                calibration
                    .apply(&mut frame.image, frame.exposure_seconds, frame.bayer)
                    .ok()?;
                let deviations =
                    frame_deviations(&frame.image, frame.bayer, options.sigma, &options.gpu)?;
                Some((
                    (frame.image.width, frame.image.height, frame.image.channels),
                    frame.exposure_seconds,
                    deviations,
                ))
            })
            .collect::<Vec<_>>();
        for (geometry, exposure, deviations) in found.into_iter().flatten() {
            detector.add(geometry, exposure, deviations);
        }
    }
    Ok(detector.finish())
}

/// `count` indices spread evenly over `0..length`, ends included.
fn evenly_spaced(length: usize, count: usize) -> Vec<usize> {
    if length <= count {
        return (0..length).collect();
    }
    let mut chosen = (0..count)
        .map(|step| step * (length - 1) / (count - 1).max(1))
        .collect::<Vec<_>>();
    chosen.dedup();
    chosen
}

/// One sample's deviation in one frame.
#[derive(Clone, Copy, Debug)]
struct Deviation {
    sample: u32,
    /// Above its neighbours rather than below.
    hot: bool,
    /// How far above the median of its eight neighbours it stands.
    excess: f32,
    /// The frame's deviation sigma in the sample's plane.
    noise: f32,
    /// How far the sample's surroundings rise above their own background,
    /// in their noise: on a raw colour frame, the median of the eight
    /// adjacent photosites' deviations from their own colour's neighbours;
    /// otherwise the median of the eight nearest neighbours less the median
    /// of the sixteen beyond them, over the frame's deviation sigma.
    structure: f32,
}

/// What the examined frames said about one sample.
#[derive(Clone, Debug, Default)]
struct Evidence {
    hot: u32,
    cold: u32,
    /// Each flagging frame's [`Deviation::structure`].
    structure: Vec<f32>,
    /// Each frame that found it hot: its excess and noise, per second of
    /// exposure when every frame had one.
    excess: Vec<(f32, f32, Option<f64>)>,
}

/// Collects each examined frame's deviant samples.
struct DefectDetector {
    options: DefectDetectionOptions,
    geometry: Option<(usize, usize, usize)>,
    frames: u32,
    /// Each examined frame's exposure.
    exposures: Vec<Option<f64>>,
    evidence: HashMap<u32, Evidence>,
}

impl DefectDetector {
    fn new(options: DefectDetectionOptions) -> Self {
        Self {
            options,
            geometry: None,
            frames: 0,
            exposures: Vec::new(),
            evidence: HashMap::new(),
        }
    }

    /// Count one frame's deviant samples, unless its geometry differs from
    /// the first frame's.
    fn add(
        &mut self,
        geometry: (usize, usize, usize),
        exposure: Option<f64>,
        deviations: Vec<Deviation>,
    ) {
        if *self.geometry.get_or_insert(geometry) != geometry {
            return;
        }
        self.frames += 1;
        let exposure = exposure.filter(|seconds| seconds.is_finite() && *seconds > 0.0);
        self.exposures.push(exposure);
        for deviation in deviations {
            let evidence = self.evidence.entry(deviation.sample).or_default();
            if deviation.hot {
                evidence.hot += 1;
                evidence
                    .excess
                    .push((deviation.excess, deviation.noise, exposure));
            } else {
                evidence.cold += 1;
            }
            evidence.structure.push(deviation.structure);
        }
    }

    fn finish(self) -> DefectMap {
        let Some((width, height, channels)) = self.geometry else {
            return DefectMap::default();
        };
        // The excess is kept at the examined lights' median exposure, and
        // scaled to it from each light's, when every light had one.
        let mut exposures = self.exposures.iter().copied().collect::<Option<Vec<f64>>>();
        let exposure_seconds = exposures.as_mut().and_then(|exposures| {
            exposures.sort_by(f64::total_cmp);
            exposures.get(exposures.len() / 2).copied()
        });
        let mut map = DefectMap {
            width,
            height,
            frames: self.frames,
            pixels: Vec::new(),
            excess: Vec::new(),
            exposure_seconds,
        };
        if self.frames < MINIMUM_DEFECT_FRAMES {
            return map;
        }
        let needed = ((self.options.persistence * self.frames as f32).ceil() as u32).max(2);
        let mut found = self
            .evidence
            .into_iter()
            .filter_map(|(sample, mut evidence)| {
                if evidence.hot < needed && evidence.cold < needed {
                    return None;
                }
                let flagged = evidence.structure.len() as f32;
                let limit =
                    MAXIMUM_STRUCTURE * STRUCTURE_SIGMA * MEDIAN_EFFICIENCY / flagged.sqrt();
                if median_in_place(&mut evidence.structure).abs() > limit {
                    return None;
                }
                let excess = if channels == 1 && evidence.hot >= needed {
                    steady_excess(&evidence.excess, exposure_seconds)
                } else {
                    0.0
                };
                Some((sample / channels as u32, excess))
            })
            .collect::<Vec<_>>();
        found.sort_unstable_by_key(|&(pixel, _)| pixel);
        found.dedup_by_key(|&mut (pixel, _)| pixel);
        map.excess = if found.iter().any(|&(_, excess)| excess > 0.0) {
            found.iter().map(|&(_, excess)| excess).collect()
        } else {
            Vec::new()
        };
        map.pixels = found.into_iter().map(|(pixel, _)| pixel).collect();
        map
    }
}

/// A hot pixel's excess over its neighbours at `exposure_seconds`, from the
/// frames that found it hot, when it holds steady across them; otherwise
/// zero, so its neighbours replace it.
fn steady_excess(found: &[(f32, f32, Option<f64>)], exposure_seconds: Option<f64>) -> f32 {
    if found.len() < MINIMUM_DEFECT_FRAMES as usize {
        return 0.0;
    }
    let scaled = |value: f32, exposure: Option<f64>| match (exposure_seconds, exposure) {
        (Some(reference), Some(own)) => value * (reference / own) as f32,
        _ => value,
    };
    let mut excess = found
        .iter()
        .map(|&(excess, _, exposure)| scaled(excess, exposure))
        .collect::<Vec<_>>();
    let mut noise = found
        .iter()
        .map(|&(_, noise, exposure)| scaled(noise, exposure))
        .collect::<Vec<_>>();
    let typical = median_in_place(&mut excess);
    let noise = median_in_place(&mut noise);
    for value in &mut excess {
        *value = (*value - typical).abs();
    }
    let wander = MAD_TO_SIGMA * median_in_place(&mut excess);
    if typical > 0.0 && wander <= 2.0 * noise + EXCESS_WANDER * typical {
        typical
    } else {
        0.0
    }
}

/// The samples of one calibrated frame that stand more than `sigma` from
/// the median of their eight same-plane neighbours, both in the frame's
/// noise and in the spread of the 24 samples around them. `None` when the
/// frame is too small or its noise cannot be measured. Screened on the GPU
/// when `gpu` allows it and one is in use, or else by examining every
/// sample: the same deviations either way.
fn frame_deviations(
    image: &LinearImage,
    bayer: Option<BayerLayout>,
    sigma: f32,
    gpu: &crate::GpuOptions,
) -> Option<Vec<Deviation>> {
    let step = if bayer.is_some() && image.channels == 1 {
        2
    } else {
        1
    };
    let (width, height, channels) = (image.width, image.height, image.channels);
    if width <= 4 * step || height <= 4 * step || image.data.len() > u32::MAX as usize {
        return None;
    }
    let planes = step * step;
    let plane = |x: usize, y: usize| (y % step) * step + x % step;
    let noise = plane_noise(image, step)?;
    let data = &image.data;
    // Whether the sample at `(x, y, channel)` stands out, as a deviation.
    let examine = |x: usize, y: usize, channel: usize| -> Option<Deviation> {
        let mut ring = [0.0f32; 8];
        let mut outer = [0.0f32; 16];
        let mut around = [0.0f32; 24];
        let value = data[(y * width + x) * channels + channel];
        if !value.is_finite() {
            return None;
        }
        let count = gather(
            data, x, y, channel, step, 1, width, height, channels, &mut ring,
        );
        if count < 5 {
            return None;
        }
        let frame_sigma = noise[channel * planes + plane(x, y)];
        let threshold = sigma * frame_sigma;
        // The median lies between the extremes, so a sample within the
        // threshold of both cannot deviate beyond it.
        let (low, high) = ring[..count]
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), &v| {
                (low.min(v), high.max(v))
            });
        if value - low <= threshold && high - value <= threshold {
            return None;
        }
        let median = median_in_place(&mut ring[..count]);
        let deviation = value - median;
        // A defect stands out from all its neighbours but at most one, its
        // partner in a pair; a pixel on a star's flank has brighter ones
        // toward the core. The median left the neighbours sorted.
        if deviation.abs() <= threshold
            || (deviation > 0.0 && value <= ring[count - 2])
            || (deviation < 0.0 && value >= ring[1])
        {
            return None;
        }
        // The local noise, from all 24 samples around: eight alone scatter
        // too widely to judge by.
        let outer_count = gather(
            data, x, y, channel, step, 2, width, height, channels, &mut outer,
        );
        let total = count + outer_count;
        around[..count].copy_from_slice(&ring[..count]);
        around[count..total].copy_from_slice(&outer[..outer_count]);
        let center = median_in_place(&mut around[..total]);
        for sample in &mut around[..total] {
            *sample = (*sample - center).abs();
        }
        let local = MAD_TO_SIGMA * median_in_place(&mut around[..total]);
        if deviation.abs() <= sigma * local {
            return None;
        }
        let structure = if step == 2 {
            // A raw colour frame's same-colour neighbours are two pixels
            // away, where a tight star has already faded; the other colours
            // beside it are still in its core. So judge the eight adjacent
            // photosites, each by its own colour's neighbours and noise.
            let mut adjacent = [0.0f32; 8];
            let mut adjacent_count = 0;
            for (dx, dy) in ring_offsets(1) {
                let Some((nx, ny)) = offset(x, y, dx, dy, width, height) else {
                    continue;
                };
                let neighbor = data[(ny * width + nx) * channels + channel];
                let mut own = [0.0f32; 8];
                let own_count = gather(
                    data, nx, ny, channel, step, 1, width, height, channels, &mut own,
                );
                if neighbor.is_finite() && own_count >= 5 {
                    adjacent[adjacent_count] = (neighbor - median_in_place(&mut own[..own_count]))
                        / noise[channel * planes + plane(nx, ny)];
                    adjacent_count += 1;
                }
            }
            if adjacent_count >= 5 {
                median_in_place(&mut adjacent[..adjacent_count])
            } else {
                0.0
            }
        } else if outer_count >= 8 {
            (median - median_in_place(&mut outer[..outer_count])) / frame_sigma
        } else {
            0.0
        };
        Some(Deviation {
            sample: ((y * width + x) * channels + channel) as u32,
            hot: deviation > 0.0,
            excess: deviation,
            noise: frame_sigma,
            structure,
        })
    };
    // The GPU screens every sample against its neighbours' extremes, as
    // `examine` does first, and flags the few it cannot rule out, so only
    // those are examined here: the same deviations, in the same order.
    #[cfg(feature = "gpu")]
    if let Some(flags) = gpu_candidates(image, step, sigma, &noise, gpu) {
        const WORDS: usize = 1024;
        let found = flags
            .par_chunks(WORDS)
            .enumerate()
            .flat_map_iter(|(chunk, words)| {
                let examine = &examine;
                words.iter().enumerate().flat_map(move |(index, &word)| {
                    let first = (chunk * WORDS + index) * 32;
                    (0..32)
                        .filter(move |bit| word >> bit & 1 == 1)
                        .filter_map(move |bit| {
                            let sample = first + bit;
                            let pixel = sample / channels;
                            examine(pixel % width, pixel / width, sample % channels)
                        })
                })
            })
            .collect();
        return Some(found);
    }
    #[cfg(not(feature = "gpu"))]
    let _ = gpu;
    let found = (0..height)
        .into_par_iter()
        .flat_map_iter(|y| {
            let examine = &examine;
            (0..width)
                .flat_map(move |x| (0..channels).filter_map(move |channel| examine(x, y, channel)))
        })
        .collect();
    Some(found)
}

/// The GPU's screen for [`frame_deviations`]: a bit for each sample, set
/// where its distance from its same-plane neighbours' extremes may pass
/// `sigma` times its plane's noise. `None` when the frame is small, the
/// thresholds are not all finite (the CPU then examines everything), `gpu`
/// keeps the screen on the CPU, or no GPU is in use.
#[cfg(feature = "gpu")]
fn gpu_candidates(
    image: &LinearImage,
    step: usize,
    sigma: f32,
    noise: &[f32],
    gpu: &crate::GpuOptions,
) -> Option<Vec<u32>> {
    // As `examine` forms them, so the GPU compares against the same values.
    let thresholds: Vec<f32> = noise.iter().map(|&noise| sigma * noise).collect();
    if thresholds.len() > 4 || !thresholds.iter().all(|threshold| threshold.is_finite()) {
        return None;
    }
    let mut padded = [0.0_f32; 4];
    padded[..thresholds.len()].copy_from_slice(&thresholds);
    crate::gpu::defect_candidates(
        &image.data,
        (image.width, image.height, image.channels, step),
        padded,
        gpu,
    )
}

/// Robust sigma of each channel and CFA plane's deviation from its
/// neighbours' median, measured on a sparse grid of samples.
fn plane_noise(image: &LinearImage, step: usize) -> Option<Vec<f32>> {
    let (width, height, channels) = (image.width, image.height, image.channels);
    let planes = step * step;
    let mut sigmas = Vec::with_capacity(channels * planes);
    let mut ring = [0.0f32; 8];
    for channel in 0..channels {
        for plane in 0..planes {
            let (px, py) = (plane % step, plane / step);
            let mut deviations = Vec::new();
            // Rows and columns a multiple of the CFA period apart keep the
            // plane, and a prime stride avoids aliasing with the scene.
            for y in (step + py..height.saturating_sub(step)).step_by(7 * step) {
                for x in (step + px..width.saturating_sub(step)).step_by(5 * step) {
                    let value = image.data[(y * width + x) * channels + channel];
                    if !value.is_finite() {
                        continue;
                    }
                    let count = gather(
                        &image.data,
                        x,
                        y,
                        channel,
                        step,
                        1,
                        width,
                        height,
                        channels,
                        &mut ring,
                    );
                    if count >= 5 {
                        deviations.push(value - median_in_place(&mut ring[..count]));
                    }
                }
            }
            if deviations.len() < 64 {
                return None;
            }
            let center = median_in_place(&mut deviations.clone());
            for deviation in &mut deviations {
                *deviation = (*deviation - center).abs();
            }
            let sigma = MAD_TO_SIGMA * median_in_place(&mut deviations);
            if !(sigma.is_finite() && sigma > 0.0) {
                return None;
            }
            sigmas.push(sigma);
        }
    }
    Some(sigmas)
}

/// Offsets, in plane steps, of the samples at Chebyshev distance `ring`.
fn ring_offsets(ring: isize) -> impl Iterator<Item = (isize, isize)> {
    (-ring..=ring).flat_map(move |dy| {
        (-ring..=ring)
            .filter(move |&dx| dx.abs() == ring || dy.abs() == ring)
            .map(move |dx| (dx, dy))
    })
}

fn offset(
    x: usize,
    y: usize,
    dx: isize,
    dy: isize,
    width: usize,
    height: usize,
) -> Option<(usize, usize)> {
    Some((
        x.checked_add_signed(dx).filter(|&x| x < width)?,
        y.checked_add_signed(dy).filter(|&y| y < height)?,
    ))
}

/// Copy the finite same-plane samples at Chebyshev distance `ring` steps
/// from `(x, y)` into `into`, returning how many there were.
#[allow(clippy::too_many_arguments)]
fn gather(
    data: &[f32],
    x: usize,
    y: usize,
    channel: usize,
    step: usize,
    ring: isize,
    width: usize,
    height: usize,
    channels: usize,
    into: &mut [f32],
) -> usize {
    let step = step as isize;
    let mut count = 0;
    for (dx, dy) in ring_offsets(ring) {
        let Some((nx, ny)) = offset(x, y, dx * step, dy * step, width, height) else {
            continue;
        };
        let value = data[(ny * width + nx) * channels + channel];
        if value.is_finite() {
            into[count] = value;
            count += 1;
        }
    }
    count
}

/// Median of a small scratch slice; reorders it.
fn median_in_place(values: &mut [f32]) -> f32 {
    values.sort_unstable_by(f32::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seiza_fits::HeaderValue;

    const SIZE: usize = 96;
    const SKY: f32 = 100.0;
    const NOISE: f32 = 2.0;

    /// Deterministic normal deviates, so no test depends on a crate's RNG.
    struct Normal(u64);

    impl Normal {
        fn next(&mut self) -> f32 {
            let mut uniform = || {
                self.0 = self
                    .0
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((self.0 >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
            };
            let (u, v) = (uniform(), uniform());
            ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
        }
    }

    /// A frame large enough for the GPU: sky with a gradient and noise,
    /// stars, hot, warm and dead pixels, a block of NaN and an infinite
    /// sample, `channels` interleaved.
    #[cfg(feature = "gpu")]
    fn screened_field(width: usize, height: usize, channels: usize) -> LinearImage {
        let mut normal = Normal(0x5851_f42d_4c95_7f2d ^ channels as u64);
        let stars: Vec<(f32, f32, f32)> = (0..300)
            .map(|star| {
                let at = |modulus: usize| ((star * 7919 + 13) % modulus) as f32 + 0.37;
                (
                    at(width),
                    at(height * 31) / 31.0,
                    200.0 + (star % 17) as f32 * 300.0,
                )
            })
            .collect();
        let mut data = vec![0.0_f32; width * height * channels];
        for (pixel, samples) in data.chunks_exact_mut(channels).enumerate() {
            let (x, y) = ((pixel % width) as f32, (pixel / width) as f32);
            let mut light = 0.0;
            for &(sx, sy, flux) in &stars {
                let (dx, dy) = (x - sx, y - sy);
                if dx.abs() < 6.0 && dy.abs() < 6.0 {
                    light += flux * (-(dx * dx + dy * dy) / 2.0).exp();
                }
            }
            for (channel, sample) in samples.iter_mut().enumerate() {
                let gain = 1.0 - 0.2 * channel as f32;
                *sample = gain * (SKY + 0.01 * x + light) + NOISE * normal.next();
            }
        }
        for defect in 0..2000 {
            let sample = (defect * 104_729 + 7) % data.len();
            data[sample] = match defect % 3 {
                0 => data[sample] + 800.0,
                1 => data[sample] + 25.0,
                _ => 0.0,
            };
        }
        for y in 300..306 {
            let row = y * width * channels;
            data[row + 400 * channels..row + 411 * channels].fill(f32::NAN);
        }
        data[(500 * width + 700) * channels] = f32::INFINITY;
        LinearImage::new(width, height, channels, data).unwrap()
    }

    /// The GPU flags every sample the CPU's first test keeps, and few more,
    /// on mono, raw colour and RGB frames; screened so, the frame's
    /// deviations are bit for bit those examining every sample finds.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_gpu_screen_keeps_every_deviation() {
        let Some(adapter) = crate::gpu::adapter_description() else {
            eprintln!("no GPU: the GPU defect screen not compared");
            return;
        };
        eprintln!("comparing the defect screen on {adapter}");
        let layout = BayerLayout {
            pattern: seiza_fits::BayerPattern::Rggb,
            x_offset: 0,
            y_offset: 0,
        };
        let sigma = DefectDetectionOptions::default().sigma;
        for (channels, bayer) in [(1, None), (1, Some(layout)), (3, None)] {
            let image = screened_field(1101, 1000, channels);
            let step = if bayer.is_some() { 2 } else { 1 };
            let noise = plane_noise(&image, step).unwrap();
            let flags = gpu_candidates(&image, step, sigma, &noise, &crate::GpuOptions::default())
                .expect("the GPU screens");
            let (width, height) = (image.width, image.height);
            let planes = step * step;
            let (mut kept, mut flagged) = (0, 0);
            let mut ring = [0.0f32; 8];
            for (sample, &value) in image.data.iter().enumerate() {
                let set = flags[sample / 32] >> (sample % 32) & 1 == 1;
                flagged += usize::from(set);
                let (pixel, channel) = (sample / channels, sample % channels);
                let (x, y) = (pixel % width, pixel / width);
                if !value.is_finite() {
                    continue;
                }
                let count = gather(
                    &image.data,
                    x,
                    y,
                    channel,
                    step,
                    1,
                    width,
                    height,
                    channels,
                    &mut ring,
                );
                if count < 5 {
                    continue;
                }
                let threshold = sigma * noise[channel * planes + (y % step) * step + x % step];
                let (low, high) = ring[..count]
                    .iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), &v| {
                        (low.min(v), high.max(v))
                    });
                if value - low <= threshold && high - value <= threshold {
                    continue;
                }
                kept += 1;
                assert!(
                    set,
                    "{channels} channel(s), step {step}: sample {sample} unflagged"
                );
            }
            eprintln!("{channels} channel(s), step {step}: {kept} kept, {flagged} flagged");
            assert!(kept > 1000 && flagged <= kept + kept / 100);
            let screened =
                frame_deviations(&image, bayer, sigma, &crate::GpuOptions::default()).unwrap();
            let examined =
                frame_deviations(&image, bayer, sigma, &crate::GpuOptions::cpu()).unwrap();
            assert!(!examined.is_empty());
            assert_eq!(screened.len(), examined.len());
            for (a, b) in screened.iter().zip(&examined) {
                assert_eq!(
                    (
                        a.sample,
                        a.hot,
                        a.excess.to_bits(),
                        a.noise.to_bits(),
                        a.structure.to_bits()
                    ),
                    (
                        b.sample,
                        b.hot,
                        b.excess.to_bits(),
                        b.noise.to_bits(),
                        b.structure.to_bits()
                    ),
                );
            }
        }
    }

    const STARS: [(f32, f32, f32); 9] = [
        (18.3, 20.6, 900.0),
        (47.8, 15.2, 400.0),
        (75.1, 22.9, 1500.0),
        (22.6, 49.4, 250.0),
        (50.2, 47.7, 2500.0),
        (77.9, 52.1, 600.0),
        (16.4, 78.8, 1200.0),
        (46.9, 75.3, 300.0),
        (79.5, 80.2, 800.0),
    ];

    /// The star field moved by `(dx, dy)`, its stars `exp(-r^2 / psf)`, with
    /// noise and `defects` added: `(pixel, amplitude)`, or a dead pixel for
    /// a negative amplitude. A colour frame is RGGB, each colour at its own
    /// level.
    fn light(
        (dx, dy): (f32, f32),
        psf: f32,
        cfa: bool,
        defects: &[(usize, f32)],
        noise: &mut Normal,
    ) -> LinearImage {
        let gain = |x: usize, y: usize| match (cfa, x % 2, y % 2) {
            (false, ..) => 1.0,
            (true, 0, 0) => 0.7,
            (true, 1, 1) => 0.5,
            _ => 1.0,
        };
        let mut data = (0..SIZE * SIZE)
            .map(|index| {
                let (x, y) = ((index % SIZE) as f32, (index / SIZE) as f32);
                let scene = STARS
                    .iter()
                    .map(|&(sx, sy, peak)| {
                        let r2 = (x - sx - dx).powi(2) + (y - sy - dy).powi(2);
                        peak * (-r2 / psf).exp()
                    })
                    .sum::<f32>();
                (SKY + scene) * gain(index % SIZE, index / SIZE) + NOISE * noise.next()
            })
            .collect::<Vec<_>>();
        for &(pixel, amplitude) in defects {
            if amplitude < 0.0 {
                data[pixel] = 0.0;
            } else {
                data[pixel] += amplitude;
            }
        }
        LinearImage::new(SIZE, SIZE, 1, data).unwrap()
    }

    const DITHER: [(f32, f32); 8] = [
        (0.0, 0.0),
        (4.3, -2.1),
        (-3.6, 3.8),
        (2.2, 5.1),
        (-5.4, -4.7),
        (6.1, 1.4),
        (-1.3, -6.2),
        (3.7, -5.5),
    ];

    /// Hot pixels from blazing to faint, and a dead one, away from the edge.
    fn defects() -> Vec<(usize, f32)> {
        [
            (30, 33, 20_000.0),
            (61, 9, 3_000.0),
            (12, 61, 200.0),
            (85, 37, 40.0),
            (66, 88, 25.0),
            (40, 62, -1.0),
        ]
        .into_iter()
        .map(|(x, y, amplitude)| (y * SIZE + x, amplitude))
        .collect()
    }

    /// Write each light, with a CFA header when `cfa`, and learn the map.
    fn detect(shifts: &[(f32, f32)], psf: f32, cfa: bool, defects: &[(usize, f32)]) -> DefectMap {
        let directory = tempfile::tempdir().unwrap();
        let headers = if cfa {
            vec![("BAYERPAT".to_string(), HeaderValue::String("RGGB".into()))]
        } else {
            Vec::new()
        };
        let mut noise = Normal(17);
        let paths = shifts
            .iter()
            .enumerate()
            .map(|(index, &shift)| {
                let path = directory.path().join(format!("light-{index}.fits"));
                let image = light(shift, psf, cfa, defects, &mut noise);
                crate::write_processed_image_fits_f32(&path, &image, &headers, &[]).unwrap();
                path
            })
            .collect::<Vec<_>>();
        detect_defects(
            &paths,
            &CalibrationMasters::default(),
            &DefectDetectionOptions::default(),
        )
        .unwrap()
    }

    fn pixels(defects: &[(usize, f32)]) -> Vec<u32> {
        let mut pixels = defects
            .iter()
            .map(|&(pixel, _)| pixel as u32)
            .collect::<Vec<_>>();
        pixels.sort_unstable();
        pixels
    }

    #[test]
    fn dithered_lights_reveal_hot_and_dead_pixels_and_no_stars() {
        for cfa in [false, true] {
            let map = detect(&DITHER, 3.2, cfa, &defects());
            assert_eq!(map.pixels(), pixels(&defects()), "cfa {cfa}");
            assert_eq!((map.width(), map.height(), map.frames()), (SIZE, SIZE, 8));
            // Each hot pixel's excess is learned, as a dark would record it;
            // a dead pixel has none to subtract.
            let mut sorted = defects();
            sorted.sort_by_key(|&(pixel, _)| pixel);
            for (&(_, amplitude), &excess) in sorted.iter().zip(map.excess()) {
                let expected = amplitude.max(0.0);
                assert!(
                    (excess - expected).abs() <= 3.0 + 0.02 * expected,
                    "cfa {cfa}: learned {excess} for {amplitude}"
                );
            }
            assert_eq!(map.exposure_seconds(), None);
        }
    }

    #[test]
    fn lights_without_defects_give_an_empty_map() {
        for cfa in [false, true] {
            assert!(detect(&DITHER, 3.2, cfa, &[]).is_empty(), "cfa {cfa}");
        }
    }

    #[test]
    fn tight_stars_that_never_moved_stay_out_of_the_map() {
        // A guided session that never dithered: the sky moves a fraction of
        // a pixel, and stars small enough to look like defects in their own
        // plane sit on the same pixels in every frame. Mono stars are 1.4
        // pixels across; colour stars 2.2, one pixel in each colour.
        let still = DITHER.map(|(dx, dy)| (dx * 0.05, dy * 0.05));
        for (cfa, psf) in [(false, 0.7), (true, 1.75)] {
            let map = detect(&still, psf, cfa, &defects());
            assert_eq!(map.pixels(), pixels(&defects()), "cfa {cfa}");
        }
    }

    #[test]
    fn fewer_than_three_lights_cannot_tell_a_defect_from_the_sky() {
        let map = detect(&DITHER[..2], 3.2, false, &defects());
        assert!(map.is_empty());
        assert_eq!(map.frames(), 2);
    }

    #[test]
    fn a_long_session_is_sampled_evenly() {
        assert_eq!(evenly_spaced(5, 16), vec![0, 1, 2, 3, 4]);
        assert_eq!(evenly_spaced(100, 4), vec![0, 33, 66, 99]);
    }

    #[test]
    fn repair_takes_the_median_of_healthy_same_plane_neighbours() {
        let width = 8;
        let data = (0..width * width)
            .map(|index| (index % width) as f32 * 10.0 + (index / width) as f32)
            .collect::<Vec<_>>();
        let original = LinearImage::new(width, width, 1, data).unwrap();
        // Two adjacent defects: neither may lend the other its value.
        let (left, right) = (3 * width + 3, 3 * width + 4);
        let map =
            DefectMap::new(width, width, vec![right as u32, left as u32, left as u32]).unwrap();
        assert_eq!(map.pixels(), [left as u32, right as u32]);

        let mut image = original.clone();
        image.data[left] = 9_999.0;
        image.data[right] = -9_999.0;
        assert_eq!(map.repair(&mut image, None, None).unwrap(), 2);
        // Each defect keeps seven healthy neighbours, one side short, so
        // their median sits a column off the true value.
        assert_eq!(image.data[left], 32.0);
        assert_eq!(image.data[right], 44.0);
        let untouched = |image: &LinearImage| {
            (0..image.data.len())
                .filter(|&index| index != left && index != right)
                .all(|index| image.data[index] == original.data[index])
        };
        assert!(untouched(&image));

        // A raw colour frame takes neighbours two pixels away, its own
        // colour, all healthy and symmetric: the ramp's own value.
        let layout = BayerLayout {
            pattern: seiza_fits::BayerPattern::Rggb,
            x_offset: 0,
            y_offset: 0,
        };
        let mut image = original.clone();
        image.data[left] = 9_999.0;
        image.data[right] = -9_999.0;
        map.repair(&mut image, Some(layout), None).unwrap();
        assert_eq!(image.data[left], original.data[left]);
        assert_eq!(image.data[right], original.data[right]);
        assert!(untouched(&image));

        // A frame of another size is not this sensor's.
        let mut other =
            LinearImage::new(width + 2, width, 1, vec![1.0; (width + 2) * width]).unwrap();
        assert_eq!(map.repair(&mut other, None, None).unwrap(), 0);
        assert!(other.data.iter().all(|&value| value == 1.0));
    }

    /// A map with one steady hot pixel in the middle of a 16 x 16 frame.
    fn hot_map(excess: f32, exposure_seconds: Option<f64>) -> DefectMap {
        let map: DefectMap = serde_json::from_value(serde_json::json!({
            "width": 16, "height": 16, "pixels": [8 * 16 + 8], "excess": [excess],
            "exposure_seconds": exposure_seconds,
        }))
        .unwrap();
        map.validate().unwrap();
        map
    }

    #[test]
    fn a_star_on_a_hot_pixel_keeps_its_core() {
        // The dither has carried a star's core onto the hot pixel: taking
        // its neighbours' median would cut the core to a fraction; taking
        // the pixel's learned excess away leaves the star as it was.
        let star = (0..16 * 16)
            .map(|index| {
                let (x, y) = ((index % 16) as f32 - 8.0, (index / 16) as f32 - 8.0);
                100.0 + 1000.0 * (-(x * x + y * y) / 3.2).exp()
            })
            .collect::<Vec<_>>();
        let centre = 8 * 16 + 8;
        let mut image = LinearImage::new(16, 16, 1, star.clone()).unwrap();
        image.data[centre] += 500.0;
        hot_map(500.0, None).repair(&mut image, None, None).unwrap();
        assert_eq!(image.data, star);

        // The excess scales with the light's exposure: half the exposure,
        // half the dark current.
        let mut image = LinearImage::new(16, 16, 1, star.clone()).unwrap();
        image.data[centre] += 250.0;
        hot_map(500.0, Some(120.0))
            .repair(&mut image, None, Some(60.0))
            .unwrap();
        assert_eq!(image.data, star);

        // A light in which the pixel is not hot, as one of another
        // temperature might be, would be left far below its neighbours:
        // their median stands in.
        let mut image = LinearImage::new(16, 16, 1, vec![100.0; 256]).unwrap();
        hot_map(500.0, None).repair(&mut image, None, None).unwrap();
        assert_eq!(image.data[centre], 100.0);

        // A map given as bare pixels takes the median, star or not, which
        // here leaves the core a third short of its 1 100.
        let mut image = LinearImage::new(16, 16, 1, star.clone()).unwrap();
        image.data[centre] += 500.0;
        let bare = DefectMap::new(16, 16, vec![centre as u32]).unwrap();
        bare.repair(&mut image, None, None).unwrap();
        assert!(image.data[centre] < 750.0, "{}", image.data[centre]);
    }

    #[test]
    fn maps_and_options_refuse_what_cannot_be_right() {
        assert!(DefectMap::new(4, 4, vec![16]).is_err());
        assert!(DefectMap::new(0, 4, vec![0]).is_err());
        // What lights nobody could read give: it repairs nothing.
        assert!(DefectMap::default().validate().is_ok());
        let unordered: DefectMap =
            serde_json::from_str(r#"{"width":4,"height":4,"pixels":[3,1]}"#).unwrap();
        assert!(unordered.validate().is_err());

        for broken in [
            r#"{"width":4,"height":4,"pixels":[1,5],"excess":[1.0]}"#,
            r#"{"width":4,"height":4,"pixels":[1,5],"excess":[1.0,-1.0]}"#,
            r#"{"width":4,"height":4,"pixels":[1],"excess":[1.0],"exposure_seconds":0.0}"#,
        ] {
            let map: DefectMap = serde_json::from_str(broken).unwrap();
            assert!(map.validate().is_err(), "{broken}");
        }

        let map = DefectMap::new(4, 4, vec![5, 1]).unwrap();
        let json = serde_json::to_string(&map).unwrap();
        assert_eq!(json, r#"{"width":4,"height":4,"frames":0,"pixels":[1,5]}"#);
        assert_eq!(serde_json::from_str::<DefectMap>(&json).unwrap(), map);

        for options in [
            DefectDetectionOptions {
                sigma: 0.0,
                ..Default::default()
            },
            DefectDetectionOptions {
                persistence: 1.5,
                ..Default::default()
            },
            DefectDetectionOptions {
                maximum_frames: 2,
                ..Default::default()
            },
        ] {
            assert!(detect_defects(&[], &CalibrationMasters::default(), &options).is_err());
        }
    }
}
