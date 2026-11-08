use crate::backend::{Backend, BackendOptions, CoreMlUnits, Precision, runtime};
use crate::bands::NoiseBands;
use crate::bundle::{HoleGuard, InputPlane, ModelBundle, ModelCard, NoiseInput, Reading, Task};
use crate::error::ModelError;
use crate::holes::{Guard, guarded, guarded_residual};
use crate::noise::{image_map, noise_input};
use crate::session::OrtRunner;
use crate::sky::{network_input, scales, seen};
use crate::tiles::{TileRunner, Tiling, infer};
use directories::ProjectDirs;
use std::path::PathBuf;

/// A borrowed image: row-major samples, channels interleaved, NaN where
/// nothing covers a pixel.
#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a> {
    data: &'a [f32],
    width: usize,
    height: usize,
    channels: usize,
}

impl<'a> ImageView<'a> {
    /// An image of `width` × `height` pixels of `channels` samples each.
    pub fn new(
        data: &'a [f32],
        width: usize,
        height: usize,
        channels: usize,
    ) -> Result<Self, ModelError> {
        if width == 0 || height == 0 || channels == 0 {
            return Err(ModelError::InvalidImage(format!(
                "an image of {width} × {height} with {channels} channels holds nothing"
            )));
        }
        if data.len() != width * height * channels {
            return Err(ModelError::InvalidImage(format!(
                "{} samples are not an image of {width} × {height} with {channels} channels",
                data.len()
            )));
        }
        Ok(Self {
            data,
            width,
            height,
            channels,
        })
    }

    fn plane(&self, channel: usize) -> Vec<f32> {
        if self.channels == 1 {
            return self.data.to_vec();
        }
        self.data
            .iter()
            .skip(channel)
            .step_by(self.channels)
            .copied()
            .collect()
    }
}

/// How far a run has gone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Progress {
    /// The channel being run, from 0.
    pub channel: usize,
    /// How many channels the image has.
    pub channels: usize,
    /// How many of this channel's tiles are done.
    pub tile: usize,
    /// How many tiles a channel takes.
    pub tiles: usize,
}

/// What a host may watch and stop during a run.
#[derive(Clone, Copy, Default)]
pub struct DenoiseOptions<'a> {
    /// Told after each tile.
    pub progress: Option<&'a (dyn Fn(Progress) + Sync)>,
    /// Asked before each tile: true stops the run.
    pub cancel: Option<&'a (dyn Fn() -> bool + Sync)>,
}

/// The scales one channel was read in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelScales {
    /// The image's background.
    pub background: f32,
    /// The image's median noise, which is what `σ̄` means in a model's output.
    pub noise: f32,
}

/// What ran, for a file's provenance and a host's cache key.
#[derive(Clone, Debug, PartialEq)]
pub struct DenoiseReport {
    /// The model's name.
    pub model: String,
    /// The hash of the graph that ran.
    pub sha256: String,
    /// The backend asked for. Core ML can place work on another unit without
    /// saying so.
    pub backend: Backend,
    /// The graph's precision.
    pub precision: Precision,
    /// A tile's side. It does not change the output, so a cache key leaves it
    /// out.
    pub tile: usize,
    /// What filled the noise plane.
    pub noise_map: NoiseInput,
    /// How the image's holes were kept from the network, when the card asks.
    pub holes: Option<HoleGuard>,
    /// The scales of each channel.
    pub channels: Vec<ChannelScales>,
}

/// A run's result. It keeps what the model found, so choosing how much of the
/// noise to keep costs a blend and no inference.
#[derive(Clone, Debug)]
pub struct Denoised {
    image: Vec<f32>,
    residual: Vec<Vec<f32>>,
    width: usize,
    height: usize,
    /// What ran.
    pub report: DenoiseReport,
}

impl Denoised {
    /// The image with these shares of what the model found taken out, laid out
    /// as the image given to [`NoiseModel::denoise`] was.
    pub fn blend(&self, bands: &NoiseBands) -> Result<Vec<f32>, ModelError> {
        bands.validate()?;
        let channels = self.residual.len();
        let view = ImageView::new(&self.image, self.width, self.height, channels)?;
        let mut out = vec![0.0_f32; self.image.len()];
        for (channel, residual) in self.residual.iter().enumerate() {
            let plane = bands.apply(&view.plane(channel), residual, self.width, self.height);
            for (sample, value) in out.iter_mut().skip(channel).step_by(channels).zip(plane) {
                *sample = value;
            }
        }
        Ok(out)
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
        self.residual.len()
    }

    /// What the model found in one channel: `f(x) − x`, in the image's units.
    pub fn residual(&self, channel: usize) -> Option<&[f32]> {
        self.residual.get(channel).map(Vec::as_slice)
    }
}

/// A noise model, open on a backend.
pub struct NoiseModel {
    card: ModelCard,
    sha256: String,
    backend: Backend,
    precision: Precision,
    tiling: Tiling,
    runner: OrtRunner,
}

impl NoiseModel {
    /// Open `bundle` on `backend`, once the graph that backend runs matches
    /// its card. The first open of a model on Core ML or TensorRT compiles it,
    /// which can take minutes; later opens read the compile cache.
    pub fn open(
        bundle: &ModelBundle,
        backend: Backend,
        options: &BackendOptions,
    ) -> Result<Self, ModelError> {
        let card = bundle.card().clone();
        if card.task != Task::Denoise {
            return Err(ModelError::ModelContract(format!(
                "{} is a {} model, not a noise model",
                card.name, card.task
            )));
        }
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

        // A graph of a fixed size takes the one tile its card gives.
        let tile = if card.dimensions.is_empty() {
            if let Some(tile) = options.tile.filter(|&tile| tile != card.tiling.tile) {
                return Err(ModelError::ModelContract(format!(
                    "{}'s graph runs only tiles of {}, not {tile}",
                    card.name, card.tiling.tile
                )));
            }
            None
        } else {
            options.tile.or(backend.default_tile())
        };
        let tiling = card.tiling.tiling(tile);
        tiling.tiles(tiling.tile, tiling.tile)?;
        let cache = compile_cache(
            options,
            &graph.sha256,
            tiling.tile,
            precision,
            backend,
            &runtime.info.version,
        );
        if let Some(cache) = &cache {
            std::fs::create_dir_all(cache).map_err(ModelError::io(cache))?;
        }
        let runner = OrtRunner::open(
            runtime,
            &bundle.path_of(graph),
            &card,
            backend,
            tiling.tile,
            cache.as_deref(),
        )?;
        let sha256 = graph.sha256.clone();
        Ok(Self {
            card,
            sha256,
            backend,
            precision,
            tiling,
            runner,
        })
    }

    /// The backend this model runs on.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// The tiling this model runs with.
    pub fn tiling(&self) -> Tiling {
        self.tiling
    }

    /// Run the model over `image`. Each channel of a colour image runs alone
    /// through a mono model, with scales of its own.
    pub fn denoise(
        &mut self,
        image: ImageView<'_>,
        options: &DenoiseOptions<'_>,
    ) -> Result<Denoised, ModelError> {
        let mut residual = Vec::with_capacity(image.channels);
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
            let (found, scales) = denoise_plane(
                &mut self.runner,
                &plane,
                image.width,
                image.height,
                &self.card.reading,
                &self.card.planes(),
                &self.tiling,
                &|| options.cancel.is_some_and(|cancel| cancel()),
                &report,
            )?;
            residual.push(found);
            channels.push(scales);
        }
        Ok(Denoised {
            image: image.data.to_vec(),
            residual,
            width: image.width,
            height: image.height,
            report: DenoiseReport {
                model: self.card.name.clone(),
                sha256: self.sha256.clone(),
                backend: self.backend,
                precision: self.precision,
                tile: self.tiling.tile,
                noise_map: self.card.reading.noise_map,
                holes: self.card.reading.holes,
                channels,
            },
        })
    }
}

/// What a model finds in one plane, and the scales it read it in. With the
/// card's hole guard, the network reads the plane with its holes filled, and
/// what it finds there comes back as a residual of the plane as given.
#[allow(clippy::too_many_arguments)]
fn denoise_plane(
    runner: &mut dyn TileRunner,
    plane: &[f32],
    width: usize,
    height: usize,
    reading: &Reading,
    inputs: &[InputPlane],
    tiling: &Tiling,
    stop: &dyn Fn() -> bool,
    report: &dyn Fn(usize, usize),
) -> Result<(Vec<f32>, ChannelScales), ModelError> {
    let (guard, mapped) = match reading.holes {
        None => (None, image_map(plane, width, height, reading)),
        Some(guard) => match guarded(plane, width, height, &guard, reading) {
            Guard::Clear(mapped) => (None, mapped),
            Guard::Holes(guarded) => {
                let mapped = image_map(&guarded.read, width, height, reading);
                (Some((guard, guarded)), mapped)
            }
        },
    };
    let read = guard.as_ref().map_or(plane, |(_, guarded)| &guarded.read);
    let sigma = noise_input(mapped, reading);
    let looked = seen(read, width, height, reading);
    let (background, noise) = scales(&looked, &sigma)
        .ok_or_else(|| ModelError::InvalidImage("the image holds no finite pixel".to_string()))?;
    if !(noise > 0.0 && noise.is_finite()) {
        return Err(ModelError::InvalidImage(format!(
            "the image's noise measures {noise}, so it has no scale to be read in"
        )));
    }
    let planes = network_input(
        &looked,
        &sigma,
        background,
        noise,
        reading.noise_plane,
        inputs,
    );
    drop((looked, sigma));
    let found = infer(
        runner,
        &planes,
        inputs.len(),
        width,
        height,
        tiling,
        noise,
        stop,
        &mut |tile, tiles| {
            report(tile, tiles);
        },
    )?;
    let found = match &guard {
        Some((guard, guarded)) => guarded_residual(plane, guarded, &found, guard),
        None => found,
    };
    Ok((found, ChannelScales { background, noise }))
}

/// Where a compiled model is kept. The tile is in the key because Core ML's
/// cache ignores the dimension override: a session at a new size would load the
/// cached graph of the old one, and fail.
///
/// Core ML always gets a folder. Without one, ONNX Runtime compiles in the
/// temp folder, and a process killed with a session open leaves two entries
/// there, the size of the model, every time.
fn compile_cache(
    options: &BackendOptions,
    sha256: &str,
    tile: usize,
    precision: Precision,
    backend: Backend,
    version: &str,
) -> Option<PathBuf> {
    if !matches!(backend, Backend::CoreMl(_)) {
        return None;
    }
    let root = options
        .cache_dir
        .clone()
        .or_else(|| {
            std::env::var_os("SEIZA_CACHE_DIR")
                .map(PathBuf::from)
                .or_else(|| {
                    ProjectDirs::from("fyi", "Seiza", "seiza")
                        .map(|dirs| dirs.cache_dir().to_path_buf())
                })
        })
        .unwrap_or_else(|| std::env::temp_dir().join("seiza"));
    let hash = sha256.get(..16).unwrap_or(sha256);
    Some(
        root.join("models")
            .join(format!("{hash}-{tile}-{precision}-{backend}-{version}")),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::{NoisePlane, SkyInput};

    const BOTH: [InputPlane; 2] = [InputPlane::Image, InputPlane::Noise];
    const TILING: Tiling = Tiling {
        tile: 64,
        overlap: 36,
        margin: 8,
        multiple: 16,
    };

    fn reading(noise_map: NoiseInput) -> Reading {
        Reading {
            noise_map,
            noise_plane: NoisePlane::Asinh,
            sky: SkyInput::Global,
            sky_block: 32,
            flat_block: 256,
            image_map_scale: 1.0,
            noise_correlation: None,
            holes: None,
        }
    }

    fn noisy(width: usize, height: usize) -> Vec<f32> {
        let mut state = 7_u64;
        (0..width * height)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                100.0 + 6.0 * ((state >> 40) as f32 / (1_u32 << 24) as f32 - 0.5)
            })
            .collect()
    }

    #[test]
    fn a_view_refuses_a_buffer_of_the_wrong_length() {
        assert!(ImageView::new(&[0.0; 5], 2, 2, 1).is_err());
        assert!(ImageView::new(&[], 0, 2, 1).is_err());
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = ImageView::new(&data, 2, 1, 3).unwrap();
        assert_eq!(
            view.plane(1),
            vec![2.0, 5.0],
            "planes come out of the interleaving"
        );
    }

    #[test]
    fn a_residual_comes_back_in_the_image_s_units() {
        let (width, height) = (96, 80);
        let image = noisy(width, height);
        let mut unit = |_: &[f32], _: usize, residual: &mut [f32]| {
            residual.fill(1.0);
            Ok(())
        };
        let (found, scales) = denoise_plane(
            &mut unit,
            &image,
            width,
            height,
            &reading(NoiseInput::Constant),
            &BOTH,
            &TILING,
            &|| false,
            &|_, _| {},
        )
        .unwrap();
        assert!(scales.noise > 1.0 && scales.noise < 3.0, "{scales:?}");
        assert!((scales.background - 100.0).abs() < 1.0, "{scales:?}");
        assert!(
            found
                .iter()
                .all(|value| (value - scales.noise).abs() < 1e-3)
        );
    }

    #[test]
    fn a_constant_map_fills_the_noise_plane_with_one_value() {
        let (width, height) = (96, 80);
        let image = noisy(width, height);
        let mut plane_values = Vec::new();
        let mut watch = |planes: &[f32], side: usize, residual: &mut [f32]| {
            plane_values.extend_from_slice(&planes[side * side..]);
            residual.fill(0.0);
            Ok(())
        };
        denoise_plane(
            &mut watch,
            &image,
            width,
            height,
            &reading(NoiseInput::Constant),
            &BOTH,
            &TILING,
            &|| false,
            &|_, _| {},
        )
        .unwrap();
        let expected = 1.0_f32.asinh();
        assert!(
            plane_values
                .iter()
                .all(|value| (value - expected).abs() < 1e-6)
        );
    }

    #[test]
    fn a_runner_gets_as_many_planes_as_the_card_lists() {
        let (width, height) = (96, 80);
        let image = noisy(width, height);
        for inputs in [
            &[InputPlane::Image][..],
            &[InputPlane::Image, InputPlane::Noise][..],
            &[InputPlane::Noise, InputPlane::Image][..],
        ] {
            let mut count = |planes: &[f32], side: usize, residual: &mut [f32]| {
                assert_eq!(planes.len(), inputs.len() * side * side);
                let noise = inputs.iter().position(|&p| p == InputPlane::Noise);
                if let Some(plane) = noise {
                    let held = &planes[plane * side * side..][..side * side];
                    let level = 1.0_f32.asinh();
                    assert!(held.iter().all(|value| (value - level).abs() < 1e-6));
                }
                residual.fill(0.0);
                Ok(())
            };
            denoise_plane(
                &mut count,
                &image,
                width,
                height,
                &reading(NoiseInput::Constant),
                inputs,
                &TILING,
                &|| false,
                &|_, _| {},
            )
            .unwrap();
        }
    }

    #[test]
    fn a_card_s_hole_guard_keeps_a_hole_from_the_network_and_puts_it_back() {
        let (width, height) = (96, 80);
        let mut image = noisy(width, height);
        let hole = 40 * width + 50;
        image[hole] -= 2000.0;
        let deepest = |holes| {
            let mut lowest = f32::INFINITY;
            let mut watch = |planes: &[f32], side: usize, residual: &mut [f32]| {
                lowest = planes[..side * side].iter().copied().fold(lowest, f32::min);
                residual.fill(0.0);
                Ok(())
            };
            let (found, _) = denoise_plane(
                &mut watch,
                &image,
                width,
                height,
                &Reading {
                    holes,
                    ..reading(NoiseInput::Constant)
                },
                &BOTH,
                &TILING,
                &|| false,
                &|_, _| {},
            )
            .unwrap();
            (lowest, found[hole])
        };
        let (seen, _) = deepest(None);
        assert!(seen < -5.0, "without the guard the network meets the hole");
        let guard = HoleGuard {
            depth: 30.0,
            margin: 3,
            fill_radius: 12,
            restore: 5.0,
        };
        let (seen, found) = deepest(Some(guard));
        assert!(seen > -3.0, "with it, the network reads only sky: {seen}");
        assert_eq!(found, 0.0, "and the hole comes back as it was");
    }

    #[test]
    fn an_image_with_no_noise_or_no_pixels_is_refused() {
        let mut zero = |_: &[f32], _: usize, residual: &mut [f32]| {
            residual.fill(0.0);
            Ok(())
        };
        for image in [vec![5.0_f32; 64 * 64], vec![f32::NAN; 64 * 64]] {
            let result = denoise_plane(
                &mut zero,
                &image,
                64,
                64,
                &reading(NoiseInput::Image),
                &BOTH,
                &TILING,
                &|| false,
                &|_, _| {},
            );
            assert!(matches!(result, Err(ModelError::InvalidImage(_))));
        }
    }

    #[test]
    fn a_blend_puts_each_channel_back_where_it_came_from() {
        let result = Denoised {
            image: vec![10.0, 20.0, 30.0, 40.0],
            residual: vec![vec![-1.0, -3.0], vec![-2.0, -4.0]],
            width: 2,
            height: 1,
            report: DenoiseReport {
                model: "test".to_string(),
                sha256: String::new(),
                backend: Backend::Cpu,
                precision: Precision::F32,
                tile: 64,
                noise_map: NoiseInput::Constant,
                holes: None,
                channels: Vec::new(),
            },
        };
        let all = result.blend(&NoiseBands::default()).unwrap();
        assert_eq!(all, vec![9.0, 18.0, 27.0, 36.0]);
        assert!(
            result
                .blend(&NoiseBands {
                    high: 2.0,
                    ..Default::default()
                })
                .is_err()
        );
    }

    #[test]
    fn the_compile_cache_is_keyed_on_the_tile_and_only_core_ml_has_one() {
        let options = BackendOptions {
            cache_dir: Some("/cache".into()),
            ..Default::default()
        };
        let engine = Backend::CoreMl(CoreMlUnits::NeuralEngine);
        let key = |tile| compile_cache(&options, "abcdef", tile, Precision::F16, engine, "1.31.1");
        assert_ne!(key(512), key(1024));
        let cpu = compile_cache(
            &options,
            "abcdef",
            512,
            Precision::F32,
            Backend::Cpu,
            "1.31.1",
        );
        assert_eq!(cpu, None);
    }
}
