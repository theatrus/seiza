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
//!
//! An image whose stars are far wider than the model was trained on, and
//! whose card names a scaled pass ([`ScaledCard`]), is also sharpened binned,
//! where its stars are near the width the card names, with a PSF field, point
//! sources and scales measured from the binned image as any image's are. The
//! binned run's change, brought back up, makes the finest scales of the
//! result, and the full-size run's the coarser ones. The stars keep the
//! full-size change ([`ScaledCard::stars`]): brought up from the binned grid,
//! a star comes out square, with a dark moat.

use crate::backend::{Backend, BackendOptions, CoreMlUnits, Precision, runtime};
use crate::bundle::{
    InputPlane, ModelBundle, ModelCard, ResidualMode, ScaledCard, SharpenCard, Task,
};
use crate::error::ModelError;
use crate::model::{ImageView, Progress, compile_cache};
use crate::noise::{image_map, noise_input};
use crate::psf::{Field, Point, detections, field_point_map, stamp_planes};
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
/// The scaled pass bins an image by this many pixels a side or more, never
/// less: binned by less, its stars stay about as wide as most of training's.
const SCALE_LEAST: usize = 2;

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
    /// The share asked of everything that is not a star on the binned image,
    /// where the card's scaled pass bins it ([`ScaledCard`]): above 0 and at
    /// most 1, and never below the model's floor in the binned image's
    /// pixels. None takes the card's. A card without the scaled pass, or a
    /// model with one strength, does not read it.
    pub binned_rest: Option<f32>,
    /// Sharpen at full size alone, never binned, whatever the card says.
    pub full_size_only: bool,
    /// Told after each tile, the binned run's counted with the full size's.
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
    /// How many pixels a side the channel was also sharpened binned by, for
    /// the scaled pass; 1 where it was not, as when its stars are not wide
    /// enough, or no PSF could be measured on the binned image.
    pub binned: usize,
}

/// What ran, for a file's provenance and a host's cache key: a host keys
/// cached results on `card_sha256`, `backend`, `precision`, `stars`, `rest`,
/// `rolloff`, `binned_rest`, `full_size_only` and
/// [`OUTPUT_VERSION`](crate::OUTPUT_VERSION).
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
    /// The share asked of everything else on a binned image, for the scaled
    /// pass; None where the card has none or the run was held to full size.
    pub binned_rest: Option<f32>,
    /// Whether the run was held to full size.
    pub full_size_only: bool,
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
            options.device_id,
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
    /// run with [`ModelError::PsfField`]. One whose stars are far wider than
    /// the model was trained on, under a card with a scaled pass, is also
    /// sharpened binned ([`ScaledCard`]); where the binned image's PSF cannot
    /// be measured, the channel is sharpened at full size alone, and its
    /// report says so.
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
                binned_rest: asked.scaled.map(|scaled| scaled.rest),
                full_size_only: options.full_size_only,
                channels,
            },
        })
    }
}

/// What a run asks for, once checked.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Asked {
    stars: f32,
    rest: f32,
    rolloff: f32,
    /// The scaled pass, where the card has one and the run is not held to
    /// full size.
    scaled: Option<Scaled>,
}

/// The scaled pass a run follows: the card's, with the rest's binned ask.
#[derive(Clone, Copy, Debug)]
struct Scaled {
    stars_px: f64,
    blend: f64,
    rest: f32,
    /// The star light, in the pixel noise, from which a star keeps the
    /// full-size change ([`ScaledCard::stars`]).
    stars: f64,
}

impl Asked {
    pub(crate) fn from(
        options: &SharpenOptions<'_>,
        card: &SharpenCard,
    ) -> Result<Self, ModelError> {
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
        // A binned ask is checked even where nothing reads it.
        let binned_rest = options
            .binned_rest
            .map(|rest| share("what is not a star binned", Some(rest), 1.0))
            .transpose()?;
        let scaled = match card.scaled {
            Some(ScaledCard {
                stars_px,
                blend,
                rest,
                stars,
            }) if !options.full_size_only => Some(Scaled {
                stars_px,
                blend,
                rest: binned_rest.unwrap_or(rest as f32),
                stars,
            }),
            _ => None,
        };
        Ok(Self {
            stars: share("stars", options.stars, card.strengths.stars)?,
            rest: share("what is not a star", options.rest, card.strengths.rest)?,
            rolloff: options.rolloff,
            scaled,
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
    let read = Read::of(plane, width, height, card, sharpen)?;

    // The binned image and its reading come first, so the progress counts
    // its tiles from the start.
    let binned = match asked.scaled {
        Some(scaled) => match scale_factor(read.typical, scaled.stars_px) {
            1 => None,
            factor => Binned::of(plane, width, height, factor, &read, asked, card, sharpen)?,
        },
        None => None,
    };
    let full_tiles = tiling.tiles(width, height)?;
    let tiles = full_tiles
        + binned
            .as_ref()
            .map_or(Ok(0), |binned| tiling.tiles(binned.width, binned.height))?;

    let mut output = network_output(
        encode,
        body,
        plane,
        width,
        height,
        &read,
        (Ratio::Asked(asked.stars), Ratio::Asked(asked.rest)),
        card,
        sharpen,
        tiling,
        stop,
        &mut |tile| report(tile, tiles),
    )?;
    if let (Some(binned), Some(scaled)) = (&binned, asked.scaled) {
        let made = network_output(
            encode,
            body,
            &binned.plane,
            binned.width,
            binned.height,
            &binned.read,
            (Ratio::Given(&binned.stars), Ratio::Asked(scaled.rest)),
            card,
            sharpen,
            tiling,
            stop,
            &mut |tile| report(full_tiles + tile, tiles),
        )?;
        let change: Vec<f32> = made
            .par_iter()
            .zip(binned.plane.par_iter())
            .map(|(&made, &image)| {
                let change = made - image;
                if change.is_nan() { 0.0 } else { change }
            })
            .collect();
        let up = brought_up(
            &change,
            binned.width,
            binned.height,
            binned.factor,
            width,
            height,
        );
        // The stars keep the full-size change: the point sources rendered
        // as the network reads them, over the pixel noise.
        let stars = (scaled.stars > 0.0).then(|| {
            star_share(
                field_point_map(&read.points, &read.field, width, height, read.background),
                &read.sigma,
                scaled.stars,
            )
        });
        output = scale_blended(
            plane,
            &output,
            &up,
            width,
            height,
            scaled.blend * read.typical,
            stars.as_deref(),
        );
    }
    if asked.rolloff > 0.0 {
        output = rolloff_step(&output, plane, width, height, asked.rolloff, &read.sigma);
    }
    let channel = SharpenedChannel {
        background: read.background,
        noise: read.noise,
        psf_nodes: (read.field.ys.len(), read.field.xs.len()),
        psf_stars: read.field.stars,
        psf_global: read.field.global,
        psf_width_px: read.typical as f32,
        points: read.points.len(),
        binned: binned.as_ref().map_or(1, |binned| binned.factor),
    };
    Ok((output, channel))
}

/// What a run reads of an image, made from the image alone: its noise map,
/// background and noise, its PSF field and the median of its nodes' widths,
/// and its point sources.
pub(crate) struct Read {
    pub(crate) sigma: Vec<f32>,
    pub(crate) background: f32,
    pub(crate) noise: f32,
    pub(crate) field: Field,
    pub(crate) typical: f64,
    pub(crate) points: Vec<Point>,
}

impl Read {
    pub(crate) fn of(
        plane: &[f32],
        width: usize,
        height: usize,
        card: &ModelCard,
        sharpen: &SharpenCard,
    ) -> Result<Self, ModelError> {
        let reading = &card.reading;
        let sigma = noise_input(image_map(plane, width, height, reading), reading);
        let (background, noise) = scales(plane, &sigma).ok_or_else(|| {
            ModelError::InvalidImage("the image holds no finite pixel".to_string())
        })?;
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
        Ok(Self {
            sigma,
            background,
            noise,
            field,
            typical,
            points,
        })
    }
}

/// An image binned for the scaled pass, what it reads, and the stars' ratio
/// its run is told.
pub(crate) struct Binned {
    pub(crate) plane: Vec<f32>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) factor: usize,
    pub(crate) read: Read,
    pub(crate) stars: Vec<f32>,
}

impl Binned {
    /// `plane`, whose reading is `read`, binned `factor` × `factor`, with the
    /// stars' ratio the full size asks binned too and held to the model's
    /// floor in the binned image's own pixels, as the harness's
    /// `smaller_ratio` makes it. None where the binned image cannot be read:
    /// no PSF measured on it, or none with a width.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn of(
        plane: &[f32],
        width: usize,
        height: usize,
        factor: usize,
        read: &Read,
        asked: &Asked,
        card: &ModelCard,
        sharpen: &SharpenCard,
    ) -> Result<Option<Self>, ModelError> {
        let (binned, binned_width, binned_height) = block_mean(plane, width, height, factor);
        let binned_read = match Read::of(&binned, binned_width, binned_height, card, sharpen) {
            Ok(binned_read) => binned_read,
            Err(ModelError::PsfField(_) | ModelError::InvalidImage(_)) => return Ok(None),
            Err(other) => return Err(other),
        };
        let floor = sharpen.floor_px;
        let full = read
            .field
            .ratio_map(width, height, f64::from(asked.stars), floor)
            .unwrap_or_else(|| vec![1.0; width * height]);
        let (stars, _, _) = block_mean(&full, width, height, factor);
        drop(full);
        // The narrowest the binned run may ask: the floor in the binned
        // image's pixels, which is its ratio map asked for nothing.
        let narrowest = binned_read
            .field
            .ratio_map(binned_width, binned_height, 0.0, floor)
            .unwrap_or_else(|| vec![1.0; binned.len()]);
        let stars = stars
            .into_par_iter()
            .zip(narrowest.par_iter())
            .map(|(ratio, &narrowest)| {
                let ratio = if ratio.is_nan() { 1.0 } else { ratio };
                ratio.max(narrowest).min(1.0)
            })
            .collect();
        Ok(Some(Self {
            plane: binned,
            width: binned_width,
            height: binned_height,
            factor,
            read: binned_read,
            stars,
        }))
    }
}

/// A ratio plane a run is told: asked as a share of each node's width, or
/// given whole.
enum Ratio<'a> {
    Asked(f32),
    Given(&'a [f32]),
}

/// The network's output on `plane`, read by `read` and told `ratios`, the
/// stars' and the rest's: the residual applied to the image, before any
/// roll-off. `report` is told how many of this run's tiles are done.
#[allow(clippy::too_many_arguments)]
fn network_output(
    encode: &mut Encode<'_>,
    body: &mut Body<'_>,
    plane: &[f32],
    width: usize,
    height: usize,
    read: &Read,
    ratios: (Ratio<'_>, Ratio<'_>),
    card: &ModelCard,
    sharpen: &SharpenCard,
    tiling: &Tiling,
    stop: &dyn Fn() -> bool,
    report: &mut dyn FnMut(usize),
) -> Result<Vec<f32>, ModelError> {
    let Read {
        sigma,
        background,
        noise,
        field,
        points,
        ..
    } = read;
    let (background, noise) = (*background, *noise);
    let inputs = card.planes();
    let pixels = width * height;
    let mut planes = network_input(
        plane,
        sigma,
        background,
        noise,
        card.reading.noise_plane,
        &inputs,
    );
    let fill = |out: &mut [f32], ratio: &Ratio<'_>| match ratio {
        Ratio::Asked(strength) => out.copy_from_slice(
            &field
                .ratio_map(width, height, f64::from(*strength), sharpen.floor_px)
                .unwrap_or_else(|| vec![1.0; pixels]),
        ),
        Ratio::Given(ratio) => out.copy_from_slice(ratio),
    };
    for (out, input) in planes.chunks_mut(pixels).zip(&inputs) {
        match input {
            InputPlane::Image | InputPlane::Noise => {}
            InputPlane::Ratio => fill(out, &ratios.0),
            InputPlane::NonstellarRatio => fill(out, &ratios.1),
            InputPlane::Points => {
                let rendered = field_point_map(points, field, width, height, background);
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
        &mut |tile, _| report(tile),
    )?;
    drop(planes);

    Ok(plane
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
        .collect())
}

/// How many pixels a side an image whose stars are `width` px across is
/// binned by to bring them near `stars_px`: the nearest whole number, a half
/// to the even one, as the harness's `scale_factor` rounds it; 1, not at
/// all, where that is under [`SCALE_LEAST`].
pub(crate) fn scale_factor(width: f64, stars_px: f64) -> usize {
    if !(stars_px > 0.0 && width > 0.0) {
        return 1;
    }
    let factor = (width / stars_px).round_ties_even();
    if factor >= SCALE_LEAST as f64 && factor.is_finite() {
        factor as usize
    } else {
        1
    }
}

/// `plane` binned `factor` × `factor` by the mean of each block's finite
/// pixels, its last blocks taking what is left of it; NaN where a block has
/// none. With the binned width and height. Each block is summed row by row
/// in single precision and the rows' sums added in turn, as the harness's
/// `block_mean` sums them.
pub(crate) fn block_mean(
    plane: &[f32],
    width: usize,
    height: usize,
    factor: usize,
) -> (Vec<f32>, usize, usize) {
    let (binned_width, binned_height) = (width.div_ceil(factor), height.div_ceil(factor));
    let mut out = vec![0.0_f32; binned_width * binned_height];
    out.par_chunks_mut(binned_width)
        .enumerate()
        .for_each(|(row, line)| {
            let rows = row * factor..((row + 1) * factor).min(height);
            for (column, value) in line.iter_mut().enumerate() {
                let columns = column * factor..((column + 1) * factor).min(width);
                let (mut total, mut count) = (0.0_f32, 0_usize);
                for y in rows.clone() {
                    let mut sum = 0.0_f32;
                    for &sample in &plane[y * width + columns.start..y * width + columns.end] {
                        if sample.is_finite() {
                            sum += sample;
                            count += 1;
                        }
                    }
                    total += sum;
                }
                *value = if count > 0 {
                    (f64::from(total) / count as f64) as f32
                } else {
                    f32::NAN
                };
            }
        });
    (out, binned_width, binned_height)
}

/// Keys' cubic convolution (a = −½) at `position` along an axis of `length`
/// samples: the four samples about it, held to the axis, and their weights,
/// as the harness's `cubic_taps` makes them.
fn cubic_taps(position: f64, length: usize) -> ([usize; 4], [f32; 4]) {
    let base = position.floor();
    let (mut taps, mut weights) = ([0_usize; 4], [0.0_f32; 4]);
    for (index, offset) in (-1_i32..3).enumerate() {
        let distance = (position - base - f64::from(offset)).abs();
        let weight = if distance <= 1.0 {
            1.5 * distance.powi(3) - 2.5 * distance.powi(2) + 1.0
        } else if distance < 2.0 {
            -0.5 * distance.powi(3) + 2.5 * distance.powi(2) - 4.0 * distance + 2.0
        } else {
            0.0
        };
        taps[index] = (base as i64 + i64::from(offset)).clamp(0, length as i64 - 1) as usize;
        weights[index] = weight as f32;
    }
    (taps, weights)
}

/// A binned plane, `width` × `height` and finite, put back up `factor` times
/// by Keys' cubic, each binned pixel at the middle of the block it was made
/// from, and cut to `out_width` × `out_height` from its corner: the
/// harness's `brought_up`, down its columns first and then along its rows,
/// in single precision.
pub(crate) fn brought_up(
    plane: &[f32],
    width: usize,
    height: usize,
    factor: usize,
    out_width: usize,
    out_height: usize,
) -> Vec<f32> {
    let taps = |count: usize, length: usize| -> Vec<([usize; 4], [f32; 4])> {
        (0..count)
            .map(|index| cubic_taps((index as f64 + 0.5) / factor as f64 - 0.5, length))
            .collect()
    };
    let (down, across) = (taps(out_height, height), taps(out_width, width));
    let mut tall = vec![0.0_f32; width * out_height];
    tall.par_chunks_mut(width)
        .zip(down.par_iter())
        .for_each(|(line, (rows, weights))| {
            for (x, value) in line.iter_mut().enumerate() {
                let mut sum = plane[rows[0] * width + x] * weights[0];
                for (&row, &weight) in rows.iter().zip(weights).skip(1) {
                    sum += plane[row * width + x] * weight;
                }
                *value = sum;
            }
        });
    let mut out = vec![0.0_f32; out_width * out_height];
    out.par_chunks_mut(out_width)
        .zip(tall.par_chunks(width))
        .for_each(|(line, source)| {
            for (value, (columns, weights)) in line.iter_mut().zip(&across) {
                let mut sum = source[columns[0]] * weights[0];
                for (&column, &weight) in columns.iter().zip(weights).skip(1) {
                    sum += source[column] * weight;
                }
                *value = sum;
            }
        });
    out
}

/// Where the stars are, from 0 to 1: `rendered`, the point sources'
/// light above the background, over `least` times the pixel noise `sigma`,
/// held to 1; 0 where that is not a number. The harness's `star_share`,
/// made in `rendered`'s place.
pub(crate) fn star_share(mut rendered: Vec<f32>, sigma: &[f32], least: f64) -> Vec<f32> {
    let least = least as f32;
    rendered
        .par_iter_mut()
        .zip(sigma.par_iter())
        .for_each(|(value, &sigma)| {
            let share = *value / (least * sigma);
            *value = if share.is_nan() {
                0.0
            } else {
                share.clamp(0.0, 1.0)
            };
        });
    rendered
}

/// `output`, a sharpening of `image`, with the part of its change under a
/// Gaussian of `sigma` pixels taken from `smaller`, the binned run's change
/// brought back up, and the part above it kept: the harness's
/// `scale_blended`. With `stars`, a share from 0 to 1 at each pixel
/// ([`star_share`]), `output` is kept whole in that share. NaN where
/// `output` is NaN.
pub(crate) fn scale_blended(
    image: &[f32],
    output: &[f32],
    smaller: &[f32],
    width: usize,
    height: usize,
    sigma: f64,
    stars: Option<&[f32]>,
) -> Vec<f32> {
    let change: Vec<f32> = output
        .par_iter()
        .zip(image.par_iter())
        .map(|(&output, &image)| {
            let change = output - image;
            if change.is_nan() { 0.0 } else { change }
        })
        .collect();
    let coarse = blurred(&change, width, height, sigma);
    drop(change);
    let smooth = blurred(smaller, width, height, sigma);
    (0..output.len())
        .into_par_iter()
        .map(|index| {
            if !output[index].is_finite() {
                return output[index];
            }
            let blended = image[index] + coarse[index] + smaller[index] - smooth[index];
            match stars {
                Some(stars) => stars[index] * output[index] + (1.0 - stars[index]) * blended,
                None => blended,
            }
        })
        .collect()
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
        assert!(
            own.scaled.is_none(),
            "a card without the scaled pass never bins"
        );
        let binned = SharpenOptions {
            binned_rest: Some(0.0),
            ..Default::default()
        };
        assert!(
            matches!(
                Asked::from(&binned, card),
                Err(ModelError::InvalidOption(_))
            ),
            "a binned ask is checked even where nothing reads it"
        );

        let mut scaled = card.clone();
        scaled.scaled = Some(ScaledCard {
            stars_px: 3.5,
            blend: 0.5,
            rest: 0.35,
            stars: 2.0,
        });
        let own = Asked::from(&SharpenOptions::default(), &scaled)
            .unwrap()
            .scaled
            .unwrap();
        assert_eq!(
            (own.stars_px, own.blend, own.rest, own.stars),
            (3.5, 0.5, 0.35, 2.0)
        );
        let told = SharpenOptions {
            binned_rest: Some(0.4),
            ..Default::default()
        };
        assert_eq!(
            Asked::from(&told, &scaled).unwrap().scaled.unwrap().rest,
            0.4
        );
        let held = SharpenOptions {
            full_size_only: true,
            ..told
        };
        assert!(Asked::from(&held, &scaled).unwrap().scaled.is_none());
    }

    #[test]
    fn an_image_binned_to_almost_nothing_is_read_as_unreadable_not_a_panic() {
        let temp = tempfile::tempdir().unwrap();
        crate::bundle::tests::write_sharpen_bundle(temp.path(), "sharpen-test");
        let bundle = ModelBundle::open(temp.path()).unwrap();
        let card = bundle.card();
        let sharpen = card.sharpen.as_ref().unwrap();
        for (width, height) in [(1, 1), (3, 2), (7, 5)] {
            let plane: Vec<f32> = (0..width * height)
                .map(|index| 100.0 + index as f32)
                .collect();
            assert!(
                Read::of(&plane, width, height, card, sharpen).is_err(),
                "{width} × {height}"
            );
        }
    }

    #[test]
    fn an_image_is_binned_only_where_its_stars_are_far_wider_than_asked() {
        assert_eq!(scale_factor(11.55, 3.5), 3, "the deep Trunk");
        assert_eq!(scale_factor(7.93, 3.5), 2, "the multi-night Trunk");
        assert_eq!(scale_factor(2.33, 3.5), 1, "the California");
        // Halves go to the even factor, as the harness's `round` takes them.
        assert_eq!(scale_factor(5.25, 3.5), 2);
        assert_eq!(scale_factor(8.75, 3.5), 2);
        assert_eq!(scale_factor(12.25, 3.5), 4);
        assert_eq!(
            scale_factor(5.2, 3.5),
            1,
            "binned by less than 2, not at all"
        );
        assert_eq!(scale_factor(f64::NAN, 3.5), 1);
        assert_eq!(scale_factor(11.55, 0.0), 1);
    }

    #[test]
    fn binning_takes_each_block_s_finite_mean_and_bringing_up_puts_it_back_at_its_middle() {
        // Five by three, binned two by two: the last column and row take what
        // is left, and a block of nothing but NaN is NaN.
        let mut plane: Vec<f32> = (0..15).map(|index| index as f32).collect();
        plane[0] = f32::NAN;
        let (binned, width, height) = block_mean(&plane, 5, 3, 2);
        assert_eq!((width, height), (3, 2));
        assert_eq!(binned[0], (1.0 + 5.0 + 6.0) / 3.0);
        assert_eq!(binned[2], (4.0 + 9.0) / 2.0);
        assert_eq!(binned[3], (10.0 + 11.0) / 2.0);
        let (none, _, _) = block_mean(&[f32::NAN; 4], 2, 2, 2);
        assert!(none[0].is_nan());

        // A ramp comes back through whole away from the edges, each binned
        // pixel standing at the middle of its block.
        let factor = 3;
        let (wide, tall) = (60, 45);
        let ramp: Vec<f32> = (0..wide * tall)
            .map(|index| 0.5 * (index % wide) as f32 + 0.25 * (index / wide) as f32)
            .collect();
        let (small, small_width, small_height) = block_mean(&ramp, wide, tall, factor);
        let up = brought_up(&small, small_width, small_height, factor, wide, tall);
        for y in 2 * factor..tall - 2 * factor {
            for x in 2 * factor..wide - 2 * factor {
                let index = y * wide + x;
                assert!(
                    (up[index] - ramp[index]).abs() < 1e-3,
                    "({x}, {y}): {} against {}",
                    up[index],
                    ramp[index]
                );
            }
        }
    }

    #[test]
    fn the_scales_meet_the_binned_change_below_the_blend_and_the_full_size_change_above() {
        let (width, height) = (96, 80);
        let image = vec![100.0_f32; width * height];
        // The full-size run changes the image by a broad bump and a fine
        // checkerboard; the binned run's change is the bump alone.
        let bump = |index: usize| {
            let (x, y) = ((index % width) as f32 - 48.0, (index / width) as f32 - 40.0);
            40.0 * (-(x * x + y * y) / (2.0 * 30.0 * 30.0)).exp()
        };
        let checker = |index: usize| {
            if (index % width + index / width).is_multiple_of(2) {
                3.0
            } else {
                -3.0
            }
        };
        let mut output: Vec<f32> = (0..width * height)
            .map(|index| 100.0 + bump(index) + checker(index))
            .collect();
        output[7] = f32::NAN;
        let smaller: Vec<f32> = (0..width * height).map(bump).collect();
        let blended = scale_blended(&image, &output, &smaller, width, height, 6.0, None);
        assert!(blended[7].is_nan(), "NaN stays NaN");
        let middle = 40 * width + 48;
        // Below the blend, the binned change: no checkerboard. Above it, the
        // full size's: the bump.
        assert!(
            (blended[middle] - (100.0 + bump(middle))).abs() < 0.05 * 40.0,
            "{} against {}",
            blended[middle],
            100.0 + bump(middle)
        );
        let fine = (blended[middle] - blended[middle + 1]).abs();
        assert!(fine < 0.1, "the full size's fine change is gone: {fine}");

        // Where a star's share is 1 the full-size output stands whole, where
        // it is 0 the blend, and between them a share of each.
        let mut stars = vec![0.0_f32; width * height];
        stars[middle] = 1.0;
        stars[middle + 1] = 0.25;
        let kept = scale_blended(&image, &output, &smaller, width, height, 6.0, Some(&stars));
        assert_eq!(kept[middle], output[middle]);
        assert_eq!(kept[middle + 2], blended[middle + 2]);
        let between = 0.25 * output[middle + 1] + 0.75 * blended[middle + 1];
        assert!((kept[middle + 1] - between).abs() < 1e-4);
        assert!(kept[7].is_nan(), "NaN stays NaN");
    }

    #[test]
    fn a_star_s_share_is_its_light_over_the_noise_held_to_one() {
        let rendered = vec![0.0, 1.0, 4.0, 100.0, f32::NAN, -3.0, 1.0];
        let sigma = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        assert_eq!(
            star_share(rendered, &sigma, 2.0),
            vec![0.0, 0.5, 1.0, 1.0, 0.0, 0.0, 1.0]
        );
    }
}
