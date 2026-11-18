//! A sharpening model's run: an image brought from its own PSF to a narrower
//! one.
//!
//! The model is told the PSF at every pixel, and everything it is told is
//! made from the image alone ([`crate::psf`]): the image's PSF field,
//! sampled at a grid of nodes; the ratio each star's width is to be brought
//! to, from the nodes' widths and the two shares a caller asks for; the
//! image's point sources, rendered with their nearest node's stamp; and the
//! noise map a noise model reads. The stamp encoder runs once for each node,
//! and its vectors are interpolated to every pixel of each tile's coarsest
//! grid, so the output does not depend on where the tiles fall.

use crate::backend::{Backend, BackendOptions, CoreMlUnits, Precision, runtime};
use crate::bundle::{InputPlane, ModelBundle, ModelCard, ResidualMode, SharpenCard, Task};
use crate::error::ModelError;
use crate::model::{ImageView, Progress, compile_cache};
use crate::noise::{image_map, noise_input};
use crate::psf::{Field, detections, field_point_map, stamp_planes};
use crate::session::{OrtGraph, TensorShape};
use crate::sky::{network_input, scales};
use crate::stats::median_f32;
use crate::tiles::{Tiling, infer_at};
use rayon::prelude::*;

/// The fine part of a sharpening's change is what a Gaussian of this many
/// pixels takes out of it, and its local power is taken over a Gaussian of
/// [`FINE_AREA`] pixels.
const FINE_SPLIT: f64 = 2.0;
const FINE_AREA: f64 = 4.0;
/// A blur reads this many of its σ either way.
const BLUR_REACH: f64 = 8.0;
/// The scales' and the roll-off's medians read every 17th pixel.
const LEVEL_STRIDE: usize = 17;

/// What a caller asks of a sharpening, and what it may watch and stop.
#[derive(Clone, Copy, Default)]
pub struct SharpenOptions<'a> {
    /// The share of its width every star is brought to, above 0 and at most
    /// 1, which leaves stars as they are. Never below the model's floor in
    /// pixels. None takes the model's own.
    pub stars: Option<f32>,
    /// The share asked of everything that is not a star. None takes the
    /// model's own. A model with one strength reads `stars` alone.
    pub rest: Option<f32>,
    /// How hard the finest part of the change is held back under noise,
    /// after the model: of the change's detail under 2 px, each pixel keeps
    /// the share its local power is of that power and the square of this
    /// times the image's noise. 0 keeps all of it.
    pub rolloff: f32,
    /// Told after each tile.
    pub progress: Option<&'a (dyn Fn(Progress) + Sync)>,
    /// Asked before each tile: true stops the run.
    pub cancel: Option<&'a (dyn Fn() -> bool + Sync)>,
}

/// What one channel was read by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharpenedChannel {
    /// The image's background.
    pub background: f32,
    /// The image's median noise.
    pub noise: f32,
    /// The rows and columns of nodes its PSF field was sampled at.
    pub psf_nodes: (usize, usize),
    /// How many stars the PSF field was measured from.
    pub psf_stars: usize,
    /// Whether too few stars left one PSF for the whole image.
    pub psf_global: bool,
    /// The median of the nodes' widths, in pixels.
    pub psf_width_px: f32,
    /// How many point sources were found and rendered.
    pub points: usize,
}

/// What ran, for a file's provenance and a host's cache key: a host keys
/// cached results on `card_sha256`, `backend`, `precision`, the three
/// strengths and [`OUTPUT_VERSION`](crate::OUTPUT_VERSION).
#[derive(Clone, Debug, PartialEq)]
pub struct SharpenReport {
    /// The model's name.
    pub model: String,
    /// The hash of the body graph that ran.
    pub sha256: String,
    /// The hash of the model's card ([`ModelBundle::card_sha256`]).
    pub card_sha256: String,
    /// The backend asked for.
    pub backend: Backend,
    /// The body graph's precision.
    pub precision: Precision,
    /// A tile's side.
    pub tile: usize,
    /// The share of their width stars were brought to.
    pub stars: f32,
    /// The share asked of everything else.
    pub rest: f32,
    /// How hard the change's finest part was held back.
    pub rolloff: f32,
    /// What each channel was read by.
    pub channels: Vec<SharpenedChannel>,
}

/// A sharpened image.
#[derive(Clone, Debug)]
pub struct Sharpened {
    image: Vec<f32>,
    width: usize,
    height: usize,
    channels: usize,
    /// What ran.
    pub report: SharpenReport,
}

impl Sharpened {
    /// The image, laid out as the one given to [`SharpenModel::sharpen`]
    /// was. A pixel that was not finite is NaN.
    pub fn image(&self) -> &[f32] {
        &self.image
    }

    /// The image, taken out of the result.
    pub fn into_image(self) -> Vec<f32> {
        self.image
    }

    /// The image's width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The image's height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// The image's channels.
    pub fn channels(&self) -> usize {
        self.channels
    }
}

/// A sharpening model, open on a backend.
pub struct SharpenModel {
    card: ModelCard,
    sharpen: SharpenCard,
    card_sha256: String,
    sha256: String,
    backend: Backend,
    precision: Precision,
    tiling: Tiling,
    encoder: OrtGraph,
    body: OrtGraph,
}

impl SharpenModel {
    /// Open `bundle` on `backend`, once its encoder and the body graph that
    /// backend runs match its card. The encoder runs on ONNX Runtime's CPU
    /// provider whatever the backend. The first open on Core ML or TensorRT
    /// compiles the body, which can take minutes.
    pub fn open(
        bundle: &ModelBundle,
        backend: Backend,
        options: &BackendOptions,
    ) -> Result<Self, ModelError> {
        let card = bundle.card().clone();
        let Some(sharpen) = card.sharpen.clone().filter(|_| card.task == Task::Sharpen) else {
            return Err(ModelError::ModelContract(format!(
                "{} is a {} model, not a sharpening model",
                card.name, card.task
            )));
        };
        let runtime = runtime(options)?;
        if let Some(status) = Backend::available(options)
            .into_iter()
            .find(|s| s.backend == backend)
            && let Some(reason) = status.unavailable
        {
            return Err(ModelError::BackendUnavailable { backend, reason });
        }
        let precision = options.precision.unwrap_or(backend.default_precision());
        if backend == Backend::CoreMl(CoreMlUnits::NeuralEngine) && precision != Precision::F16 {
            return Err(ModelError::ModelContract(
                "the Neural Engine runs only an f16 graph: an f32 one would run on the CPU \
                 without saying so"
                    .to_string(),
            ));
        }
        let graph = bundle.graph(precision).ok_or_else(|| {
            ModelError::ModelContract(format!("{} has no {precision} graph", card.name))
        })?;
        bundle.verify_graph(graph)?;
        bundle.verify_encoder()?;

        let tiling = card.tiling.tiling(options.tile.or(backend.default_tile()));
        tiling.tiles(tiling.tile, tiling.tile)?;
        let (tile, cells) = (tiling.tile, tiling.tile / tiling.multiple);
        let cache = compile_cache(
            options,
            &graph.sha256,
            tile,
            precision,
            backend,
            &runtime.info.version,
        );
        if let Some(cache) = &cache {
            std::fs::create_dir_all(cache).map_err(ModelError::io(cache))?;
        }
        let side = sharpen.encoder.side;
        let encoder = OrtGraph::open(
            runtime,
            &bundle.root().join(&sharpen.encoder.file),
            &card.name,
            Backend::Cpu,
            &[],
            &[TensorShape {
                name: sharpen.encoder.input.clone(),
                planes: sharpen.encoder.planes.len(),
                height: side,
                width: side,
            }],
            &TensorShape {
                name: sharpen.encoder.output.clone(),
                planes: sharpen.encoder.features,
                height: 1,
                width: 1,
            },
            None,
        )?;
        let dimensions: Vec<(&str, usize)> = card
            .dimensions
            .iter()
            .map(|dimension| (dimension.as_str(), tile))
            .chain(
                sharpen
                    .vector
                    .dimensions
                    .iter()
                    .map(|dimension| (dimension.as_str(), cells)),
            )
            .collect();
        let body = OrtGraph::open(
            runtime,
            &bundle.path_of(graph),
            &card.name,
            backend,
            &dimensions,
            &[
                TensorShape {
                    name: card.inputs[0].tensor.clone(),
                    planes: card.inputs[0].planes.len(),
                    height: tile,
                    width: tile,
                },
                TensorShape {
                    name: sharpen.vector.tensor.clone(),
                    planes: sharpen.encoder.features,
                    height: cells,
                    width: cells,
                },
            ],
            &TensorShape {
                name: card.output.clone(),
                planes: 1,
                height: tile,
                width: tile,
            },
            cache.as_deref(),
        )?;
        Ok(Self {
            sha256: graph.sha256.clone(),
            card_sha256: bundle.card_sha256().to_string(),
            card,
            sharpen,
            backend,
            precision,
            tiling,
            encoder,
            body,
        })
    }

    /// The backend this model's body runs on.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// The tiling this model runs with.
    pub fn tiling(&self) -> Tiling {
        self.tiling
    }

    /// Sharpen `image`. Each channel of a colour image runs alone through
    /// the mono model, with a PSF field, point sources and scales of its
    /// own, and is brought to the asked share of its own stars' width.
    ///
    /// A channel whose PSF cannot be measured, for want of stars, stops the
    /// run with [`ModelError::PsfField`].
    pub fn sharpen(
        &mut self,
        image: ImageView<'_>,
        options: &SharpenOptions<'_>,
    ) -> Result<Sharpened, ModelError> {
        let asked = Asked::from(options, &self.sharpen)?;
        let mut out = vec![0.0_f32; image.data.len()];
        let mut channels = Vec::with_capacity(image.channels);
        for channel in 0..image.channels {
            let plane = image.plane(channel);
            let report = |tile: usize, tiles: usize| {
                if let Some(progress) = options.progress {
                    progress(Progress {
                        channel,
                        channels: image.channels,
                        tile,
                        tiles,
                    });
                }
            };
            let (encoder, body) = (&mut self.encoder, &mut self.body);
            let (found, read) = sharpen_plane(
                &mut |stamp, vector| encoder.run(&[stamp], vector),
                &mut |planes, vectors, residual| body.run(&[planes, vectors], residual),
                &plane,
                image.width,
                image.height,
                &self.card,
                &self.sharpen,
                &self.tiling,
                &asked,
                &|| options.cancel.is_some_and(|cancel| cancel()),
                &report,
            )?;
            for (sample, value) in out
                .iter_mut()
                .skip(channel)
                .step_by(image.channels)
                .zip(found)
            {
                *sample = value;
            }
            channels.push(read);
        }
        Ok(Sharpened {
            image: out,
            width: image.width,
            height: image.height,
            channels: image.channels,
            report: SharpenReport {
                model: self.card.name.clone(),
                sha256: self.sha256.clone(),
                card_sha256: self.card_sha256.clone(),
                backend: self.backend,
                precision: self.precision,
                tile: self.tiling.tile,
                stars: asked.stars,
                rest: asked.rest,
                rolloff: asked.rolloff,
                channels,
            },
        })
    }
}

/// What a run asks for, once checked.
#[derive(Clone, Copy, Debug)]
struct Asked {
    stars: f32,
    rest: f32,
    rolloff: f32,
}

impl Asked {
    fn from(options: &SharpenOptions<'_>, card: &SharpenCard) -> Result<Self, ModelError> {
        let share = |name: &str, asked: Option<f32>, own: f64| {
            let value = asked.unwrap_or(own as f32);
            if value > 0.0 && value <= 1.0 {
                Ok(value)
            } else {
                Err(ModelError::InvalidOption(format!(
                    "{name} are brought to a share of their width above 0 and at most 1, not \
                     {value}"
                )))
            }
        };
        if !(options.rolloff >= 0.0 && options.rolloff.is_finite()) {
            return Err(ModelError::InvalidOption(format!(
                "a roll-off is 0 or more, not {}",
                options.rolloff
            )));
        }
        Ok(Self {
            stars: share("stars", options.stars, card.strengths.stars)?,
            rest: share("what is not a star", options.rest, card.strengths.rest)?,
            rolloff: options.rolloff,
        })
    }
}

/// Fills a vector from a stamp's planes.
type Encode<'a> = dyn FnMut(&[f32], &mut [f32]) -> Result<(), ModelError> + 'a;
/// Fills a tile's residual from its planes and its vector map.
type Body<'a> = dyn FnMut(&[f32], &[f32], &mut [f32]) -> Result<(), ModelError> + 'a;

/// One plane sharpened, and what it was read by. `encode` fills a vector from
/// a stamp's planes, and `body` a tile's residual from its planes and its
/// vector map.
#[allow(clippy::too_many_arguments)]
fn sharpen_plane(
    encode: &mut Encode<'_>,
    body: &mut Body<'_>,
    plane: &[f32],
    width: usize,
    height: usize,
    card: &ModelCard,
    sharpen: &SharpenCard,
    tiling: &Tiling,
    asked: &Asked,
    stop: &dyn Fn() -> bool,
    report: &dyn Fn(usize, usize),
) -> Result<(Vec<f32>, SharpenedChannel), ModelError> {
    // A pixel that is not finite is one nothing covers.
    let plane: Vec<f32> = plane
        .par_iter()
        .map(|&value| if value.is_finite() { value } else { f32::NAN })
        .collect();
    let plane = plane.as_slice();
    let reading = &card.reading;
    let sigma = noise_input(image_map(plane, width, height, reading), reading);
    let (background, noise) = scales(plane, &sigma)
        .ok_or_else(|| ModelError::InvalidImage("the image holds no finite pixel".to_string()))?;
    if !(noise > 0.0 && noise.is_finite()) {
        return Err(ModelError::InvalidImage(format!(
            "the image's noise measures {noise}, so it has no scale to be read in"
        )));
    }
    let field = Field::measure(plane, width, height, &sharpen.psf, sharpen.encoder.side)?;
    let typical = field.typical_width().ok_or_else(|| {
        ModelError::InvalidImage("the image's PSF has no width at any node".to_string())
    })?;
    let points = detections(plane, width, height, &sharpen.points);

    let inputs = card.planes();
    let pixels = width * height;
    let mut planes = network_input(
        plane,
        &sigma,
        background,
        noise,
        reading.noise_plane,
        &inputs,
    );
    for (out, input) in planes.chunks_mut(pixels).zip(&inputs) {
        let ratio = |strength: f32| {
            field
                .ratio_map(width, height, f64::from(strength), sharpen.floor_px)
                .unwrap_or_else(|| vec![1.0; pixels])
        };
        match input {
            InputPlane::Image | InputPlane::Noise => {}
            InputPlane::Ratio => out.copy_from_slice(&ratio(asked.stars)),
            InputPlane::NonstellarRatio => out.copy_from_slice(&ratio(asked.rest)),
            InputPlane::Points => {
                let rendered = field_point_map(&points, &field, width, height, background);
                out.par_iter_mut()
                    .zip(rendered.par_iter())
                    .for_each(|(out, &value)| *out = (value / noise).asinh());
            }
        }
    }

    let features = sharpen.encoder.features;
    let mut vectors = vec![0.0_f32; field.stamps.len() * features];
    for (stamp, vector) in field.stamps.iter().zip(vectors.chunks_mut(features)) {
        if stop() {
            return Err(ModelError::Cancelled);
        }
        encode(&stamp_planes(stamp, &sharpen.encoder), vector)?;
    }
    let cells = tiling.tile / tiling.multiple;
    let mut map = vec![0.0_f32; features * cells * cells];
    let residual = infer_at(
        &mut |tile, corner, residual| {
            field.sampled_features(
                &vectors,
                features,
                corner,
                tiling.tile,
                tiling.multiple,
                &mut map,
            );
            body(tile, &map, residual)
        },
        &planes,
        inputs.len(),
        width,
        height,
        tiling,
        1.0,
        stop,
        &mut |tile, tiles| report(tile, tiles),
    )?;
    drop(planes);

    let mut output: Vec<f32> = plane
        .par_iter()
        .zip(residual.par_iter())
        .map(|(&value, &residual)| {
            if !value.is_finite() {
                return f32::NAN;
            }
            let reference = (value - background) / noise;
            let applied = match sharpen.residual {
                ResidualMode::Asinh => (reference.asinh() + residual).sinh(),
                ResidualMode::Linear => reference + residual,
            };
            background + noise * applied
        })
        .collect();
    if asked.rolloff > 0.0 {
        output = rolloff_step(&output, plane, width, height, asked.rolloff, &sigma);
    }
    let read = SharpenedChannel {
        background,
        noise,
        psf_nodes: (field.ys.len(), field.xs.len()),
        psf_stars: field.stars,
        psf_global: field.global,
        psf_width_px: typical as f32,
        points: points.len(),
    };
    Ok((output, read))
}

/// Where an index past either end of an axis of `length` falls, by
/// reflection about its outer pixels.
fn reflected(index: isize, length: usize) -> usize {
    if length == 1 {
        return 0;
    }
    let period = 2 * (length as isize - 1);
    let folded = index.rem_euclid(period);
    (if folded < length as isize {
        folded
    } else {
        period - folded
    }) as usize
}

/// `plane` blurred by a round Gaussian of `sigma` pixels, reflected past its
/// edges. The harness multiplies the image's Fourier transform by the
/// Gaussian's; for the widths used here that is this sum to a part in 10⁸.
fn blurred(plane: &[f32], width: usize, height: usize, sigma: f64) -> Vec<f32> {
    let reach = (BLUR_REACH * sigma).ceil() as isize;
    let weights: Vec<f64> = (-reach..=reach)
        .map(|offset| {
            (-((offset * offset) as f64) / (2.0 * sigma * sigma)).exp()
                / (sigma * (2.0 * std::f64::consts::PI).sqrt())
        })
        .collect();
    let columns: Vec<usize> = (-reach..width as isize + reach)
        .map(|x| reflected(x, width))
        .collect();
    let mut across = vec![0.0_f64; width * height];
    across
        .par_chunks_mut(width)
        .zip(plane.par_chunks(width))
        .for_each(|(out, line)| {
            for (x, value) in out.iter_mut().enumerate() {
                *value = weights
                    .iter()
                    .zip(&columns[x..])
                    .map(|(weight, &column)| weight * f64::from(line[column]))
                    .sum();
            }
        });
    let mut out = vec![0.0_f32; width * height];
    out.par_chunks_mut(width).enumerate().for_each(|(y, line)| {
        let mut sums = vec![0.0_f64; width];
        for (offset, weight) in (-reach..=reach).zip(&weights) {
            let row = reflected(y as isize + offset, height);
            for (sum, value) in sums.iter_mut().zip(&across[row * width..][..width]) {
                *sum += weight * value;
            }
        }
        for (value, sum) in line.iter_mut().zip(sums) {
            *value = sum as f32;
        }
    });
    out
}

/// A sharpened image with the fine part of its change from `image`, its
/// input, held back where it lies under `amount` times `sigma`, the input's
/// noise map: the harness's function of the same name. NaN stays NaN.
pub(crate) fn rolloff_step(
    output: &[f32],
    image: &[f32],
    width: usize,
    height: usize,
    amount: f32,
    sigma: &[f32],
) -> Vec<f32> {
    let finite = |index: usize| output[index].is_finite() && image[index].is_finite();
    let mut sample: Vec<f32> = (0..output.len())
        .filter(|&index| finite(index))
        .step_by(LEVEL_STRIDE)
        .map(|index| output[index])
        .collect();
    let Some(level) = median_f32(&mut sample) else {
        return output.to_vec();
    };
    let filled = |plane: &[f32]| -> Vec<f32> {
        (0..plane.len())
            .into_par_iter()
            .map(|index| if finite(index) { plane[index] } else { level })
            .collect()
    };
    let (sharp, soft) = (filled(output), filled(image));
    let change: Vec<f32> = sharp
        .iter()
        .zip(&soft)
        .map(|(sharp, soft)| sharp - soft)
        .collect();
    let smooth = blurred(&change, width, height, FINE_SPLIT);
    let fine: Vec<f32> = change
        .iter()
        .zip(&smooth)
        .map(|(change, smooth)| change - smooth)
        .collect();
    let squared: Vec<f32> = fine.iter().map(|fine| fine * fine).collect();
    let power = blurred(&squared, width, height, FINE_AREA);
    (0..output.len())
        .into_par_iter()
        .map(|index| {
            if !finite(index) {
                return output[index];
            }
            let power = power[index].max(0.0);
            let noise = if sigma[index].is_nan() {
                0.0
            } else {
                sigma[index]
            };
            let floor = (amount * noise) * (amount * noise);
            let kept = power / (power + floor);
            let kept = if kept.is_nan() { 1.0 } else { kept };
            sharp[index] - (1.0 - kept) * fine[index]
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn a_blur_keeps_a_level_and_spreads_a_point_as_its_gaussian() {
        let (width, height) = (48, 40);
        let flat = blurred(&vec![3.0; width * height], width, height, 2.0);
        assert!(flat.iter().all(|value| (value - 3.0).abs() < 1e-5));
        let mut point = vec![0.0_f32; width * height];
        point[20 * width + 24] = 1.0;
        let spread = blurred(&point, width, height, 2.0);
        let peak = 1.0 / (2.0 * std::f32::consts::PI * 4.0);
        assert!((spread[20 * width + 24] - peak).abs() < 1e-6);
        let two_out = peak * (-0.5_f32).exp();
        assert!((spread[20 * width + 26] - two_out).abs() < 1e-6);
        assert!((spread.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn a_roll_off_holds_back_fine_change_under_the_noise_and_keeps_it_above() {
        let (width, height) = (64, 64);
        let image = vec![100.0_f32; width * height];
        // A fine checkerboard of change, small on the left and large on the
        // right, and a pixel nothing covers.
        let mut output: Vec<f32> = (0..width * height)
            .map(|index| {
                let (x, y) = (index % width, index / width);
                let sign = if (x + y) % 2 == 0 { 1.0 } else { -1.0 };
                100.0 + sign * if x < 32 { 0.1 } else { 50.0 }
            })
            .collect();
        output[5] = f32::NAN;
        let sigma = vec![2.0_f32; width * height];
        let rolled = rolloff_step(&output, &image, width, height, 1.0, &sigma);
        assert!(rolled[5].is_nan(), "NaN stays NaN");
        let change = |index: usize| (rolled[index] - 100.0).abs();
        assert!(change(32 * width + 8) < 0.01, "under the noise, held back");
        assert!(change(32 * width + 56) > 49.0, "far above it, kept");
        let untouched = rolloff_step(
            &output,
            &image,
            width,
            height,
            1.0,
            &vec![0.0; width * height],
        );
        assert_eq!(
            untouched[32 * width + 8],
            output[32 * width + 8],
            "no noise, no roll-off"
        );
    }

    #[test]
    fn a_strength_must_be_a_share() {
        let temp = tempfile::tempdir().unwrap();
        crate::bundle::tests::write_sharpen_bundle(temp.path(), "sharpen-test");
        let bundle = ModelBundle::open(temp.path()).unwrap();
        let card = bundle.card().sharpen.as_ref().unwrap();
        let own = Asked::from(&SharpenOptions::default(), card).unwrap();
        assert_eq!((own.stars, own.rest, own.rolloff), (0.46, 0.5, 0.0));
        for (stars, rolloff) in [(0.0, 0.0), (1.5, 0.0), (f32::NAN, 0.0), (0.5, -1.0)] {
            let options = SharpenOptions {
                stars: Some(stars),
                rolloff,
                ..Default::default()
            };
            assert!(matches!(
                Asked::from(&options, card),
                Err(ModelError::InvalidOption(_))
            ));
        }
    }
}
