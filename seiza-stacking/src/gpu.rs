//! Clamped Lanczos-3 resampling on the GPU, through wgpu: Metal, Vulkan or
//! Direct3D 12, whichever the platform has.
//!
//! [`resample_lanczos3`] does what `registration::resample_region_with_inverse`
//! does for [`crate::Interpolation::Lanczos3`], in f32: the same taps, weight
//! table, clamping rule and bilinear fallback, but the source position of
//! each pixel interpolated from a coarse grid evaluated in f64, and the
//! arithmetic in single precision. A sample differs from the CPU's by a few
//! thousandths of the frame's noise at most, except where a tap span lies
//! on the clamping threshold and the two take different branches.
//!
//! The device is opened once, the first time a frame large enough to gain
//! from it is resampled. Setting `SEIZA_GPU=off` (or `0`, `cpu`) before then
//! keeps every resample on the CPU, as a build without the `gpu` feature
//! does. When no adapter is found, or the GPU fails, resampling stays on the
//! CPU; a failure is logged once.
use crate::{LinearImage, ReferenceRegion};
use rayon::prelude::*;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// Samples below which the CPU is as quick: the GPU's fixed cost, the
/// coordinate grid and the transfers, is a few milliseconds a frame.
const MINIMUM_SAMPLES: usize = 1 << 20;

/// Reference pixels between the source positions the host evaluates.
const GRID_STEP: u32 = 16;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
    origin_x: u32,
    origin_y: u32,
    source_width: u32,
    source_height: u32,
    channels: u32,
    grid_step: u32,
    grid_columns: u32,
    clamping: f32,
    sampling: u32,
    /// The first column's grid node in the low half, the first row's in the
    /// high.
    grid_origin: u32,
    noise_spans: [f32; 4],
}

/// How [`resample`] samples the source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum GpuSampling {
    /// Clamped Lanczos-3, each channel's taps held to its span.
    Lanczos3 {
        clamp_spans: [f32; 4],
    },
    Bilinear,
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    bands_pipeline: wgpu::ComputePipeline,
    bands_layout: wgpu::BindGroupLayout,
    weights: wgpu::Buffer,
    /// An integrated GPU, which shares memory with the CPU, reads the
    /// source and writes the output in buffers the CPU maps directly;
    /// a discrete one goes through staging copies.
    mapped: bool,
    max_buffer: u64,
}

static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
static FAILED: AtomicBool = AtomicBool::new(false);

fn disabled_by_environment() -> bool {
    std::env::var("SEIZA_GPU").is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "off" | "0" | "cpu" | "false" | "no"
        )
    })
}

fn gpu() -> Option<&'static Gpu> {
    if FAILED.load(Ordering::Relaxed) {
        return None;
    }
    GPU.get_or_init(|| {
        if disabled_by_environment() {
            return None;
        }
        match Gpu::open() {
            Ok(gpu) => Some(gpu),
            Err(reason) => {
                log_once(&format!("resampling on the CPU: {reason}"));
                None
            }
        }
    })
    .as_ref()
}

fn log_once(message: &str) {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!("seiza-stacking: {message}");
    }
}

/// The GPU adapter resampling uses, as "name (backend)", or `None` when
/// resampling stays on the CPU. Opens the device if it is not open yet.
#[cfg(test)]
pub(crate) fn adapter_description() -> Option<String> {
    gpu().map(|gpu| {
        let info = gpu.device.adapter_info();
        format!("{} ({:?})", info.name, info.backend)
    })
}

impl Gpu {
    fn open() -> Result<Self, String> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|error| format!("no GPU adapter ({error})"))?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err(format!("{} is a software adapter", info.name));
        }
        let mapped = info.device_type == wgpu::DeviceType::IntegratedGpu
            && adapter
                .features()
                .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
        let supported = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: supported.max_storage_buffer_binding_size,
            max_buffer_size: supported.max_buffer_size,
            ..wgpu::Limits::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("seiza-stacking"),
            required_features: if mapped {
                wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
            } else {
                wgpu::Features::empty()
            },
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|error| format!("cannot open {} ({error})", info.name))?;
        // A device lost or a validation error must not take the process
        // down: the resample that hit it falls back to the CPU.
        device.on_uncaptured_error(std::sync::Arc::new(|error| {
            FAILED.store(true, Ordering::Relaxed);
            let message = match &error {
                wgpu::Error::Validation { description, .. }
                | wgpu::Error::Internal { description, .. } => description.clone(),
                other => other.to_string(),
            };
            log_once(&format!(
                "GPU error, resampling on the CPU from now on: {message}"
            ));
        }));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resample"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu/resample.wgsl").into()),
        });
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resample"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("resample"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("resample"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bands_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bands"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu/bands.wgsl").into()),
        });
        let bands_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bands"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, true),
                storage(5, false),
                storage(6, false),
                storage(7, false),
            ],
        });
        let bands_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bands"),
                bind_group_layouts: &[Some(&bands_layout)],
                immediate_size: 0,
            });
        let bands_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bands"),
            layout: Some(&bands_pipeline_layout),
            module: &bands_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let table = crate::registration::lanczos3_table();
        let weights = wgpu::util::DeviceExt::create_buffer_init(
            &device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("lanczos3"),
                contents: bytemuck::cast_slice(table),
                usage: wgpu::BufferUsages::STORAGE,
            },
        );
        Ok(Self {
            device,
            queue,
            pipeline,
            layout,
            bands_pipeline,
            bands_layout,
            weights,
            mapped,
            max_buffer: limits
                .max_buffer_size
                .min(limits.max_storage_buffer_binding_size),
        })
    }

    fn buffer(&self, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    fn wait(&self) -> Result<(), String> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    fn resample(
        &self,
        source: &LinearImage,
        region: ReferenceRegion,
        grid: &[[u32; 4]],
        (grid_columns, grid_origin): (u32, u32),
        sampling: GpuSampling,
    ) -> Result<LinearImage, String> {
        let channels = source.channels;
        let source_bytes = (source.data.len() * 4) as u64;
        let output_samples = region.width * region.height * channels;
        let output_bytes = (output_samples * 4) as u64;
        if source_bytes.max(output_bytes) > self.max_buffer {
            return Err("the frame is larger than a GPU buffer".into());
        }
        let params = Params {
            width: region.width as u32,
            height: region.height as u32,
            origin_x: region.x as u32,
            origin_y: region.y as u32,
            source_width: source.width as u32,
            source_height: source.height as u32,
            channels: channels as u32,
            grid_step: GRID_STEP,
            grid_columns,
            clamping: crate::registration::LANCZOS_CLAMPING,
            sampling: match sampling {
                GpuSampling::Lanczos3 { .. } => 0,
                GpuSampling::Bilinear => 1,
            },
            grid_origin,
            noise_spans: match sampling {
                GpuSampling::Lanczos3 { clamp_spans } => clamp_spans,
                GpuSampling::Bilinear => [0.0; 4],
            },
        };
        let init = |label: &str, contents: &[u8], usage: wgpu::BufferUsages| {
            wgpu::util::DeviceExt::create_buffer_init(
                &self.device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage,
                },
            )
        };
        let uniform = init(
            "params",
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
        );
        let grid = init(
            "grid",
            bytemuck::cast_slice(grid),
            wgpu::BufferUsages::STORAGE,
        );
        let source_buffer = if self.mapped {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("source"),
                size: source_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_WRITE,
                mapped_at_creation: true,
            });
            buffer
                .slice(..)
                .get_mapped_range_mut()
                .map_err(|error| error.to_string())?
                .copy_from_slice(bytemuck::cast_slice(&source.data));
            buffer.unmap();
            buffer
        } else {
            init(
                "source",
                bytemuck::cast_slice(&source.data),
                wgpu::BufferUsages::STORAGE,
            )
        };
        let output = self.buffer(
            "output",
            output_bytes,
            wgpu::BufferUsages::STORAGE
                | if self.mapped {
                    wgpu::BufferUsages::MAP_READ
                } else {
                    wgpu::BufferUsages::COPY_SRC
                },
        );
        let readback = (!self.mapped).then(|| {
            self.buffer(
                "readback",
                output_bytes,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            )
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resample"),
            layout: &self.layout,
            entries: &[
                (0, &uniform),
                (1, &source_buffer),
                (2, &grid),
                (3, &self.weights),
                (4, &output),
            ]
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }),
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                (region.width as u32).div_ceil(16),
                (region.height as u32).div_ceil(16),
                1,
            );
        }
        if let Some(readback) = &readback {
            encoder.copy_buffer_to_buffer(&output, 0, readback, 0, output_bytes);
        }
        self.queue.submit([encoder.finish()]);
        let read_from = readback.as_ref().unwrap_or(&output);
        read_from.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.wait()?;
        if FAILED.load(Ordering::Relaxed) {
            return Err("the GPU failed".into());
        }
        let view = read_from
            .slice(..)
            .get_mapped_range()
            .map_err(|error| error.to_string())?;
        // Every core copies out: one alone reads write-combined memory, as
        // an integrated GPU's can be, several times slower.
        let mut data = vec![0.0_f32; output_samples];
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut data);
        bytes
            .par_chunks_mut(1 << 20)
            .zip(view.par_chunks(1 << 20))
            .for_each(|(out, input)| out.copy_from_slice(input));
        drop(view);
        read_from.unmap();
        LinearImage::new(region.width, region.height, channels, data)
            .map_err(|error| error.to_string())
    }
}

/// Resample `region` of the reference grid from `source` through `inverse`
/// (reference position to source position) on the GPU, or `None` when the
/// CPU should: the frame is small, the channels are neither one nor three,
/// or no GPU is in use.
pub(crate) fn resample(
    source: &LinearImage,
    region: ReferenceRegion,
    inverse: &(impl Fn(f64, f64) -> (f64, f64) + Sync),
    sampling: GpuSampling,
) -> Option<LinearImage> {
    let channels = source.channels;
    // By the source, not the region: every region of a frame resamples the
    // same way, so a footprint holds what the whole grid would.
    if !matches!(channels, 1 | 3)
        || source.data.len() < MINIMUM_SAMPLES
        || source.width < 6
        || source.height < 6
        || region.width == 0
        || region.height == 0
    {
        return None;
    }
    let gpu = gpu()?;
    let step = GRID_STEP as usize;
    let (first_column, first_row) = (region.x / step, region.y / step);
    let (last_column, last_row) = (
        (region.x + region.width - 1) / step + 1,
        (region.y + region.height - 1) / step + 1,
    );
    if first_column > 0xffff || first_row > 0xffff {
        return None;
    }
    let columns = last_column - first_column + 1;
    // Each node's source position as whole pixels and the fraction past
    // them, the fraction exact to f32 whatever the position.
    let grid: Vec<[u32; 4]> = (first_row..=last_row)
        .into_par_iter()
        .flat_map_iter(|row| {
            (first_column..=last_column).map(move |column| {
                let (x, y) = ((column * step) as f64, (row * step) as f64);
                let (source_x, source_y) = inverse(x, y);
                let split = |position: f64| {
                    if position.is_finite() && position.abs() < f64::from(i32::MAX / 2) {
                        let whole = position.floor();
                        (whole as i32, (position - whole) as f32)
                    } else {
                        (0, f32::NAN)
                    }
                };
                let ((whole_x, fraction_x), (whole_y, fraction_y)) =
                    (split(source_x), split(source_y));
                [
                    whole_x as u32,
                    whole_y as u32,
                    fraction_x.to_bits(),
                    fraction_y.to_bits(),
                ]
            })
        })
        .collect();
    let origin = (first_row as u32) << 16 | first_column as u32;
    match gpu.resample(source, region, &grid, (columns as u32, origin), sampling) {
        Ok(image) => Some(image),
        Err(reason) => {
            log_once(&format!("resampling this frame on the CPU: {reason}"));
            None
        }
    }
}

/// Frames a band session takes at most: the kernel tallies each frame's
/// counts in workgroup memory.
const MAX_BAND_FRAMES: usize = 512;

/// The final integration's rejection constants, by sample count, for the
/// band kernel.
pub(crate) struct BandTables {
    /// The low and high prediction limits.
    pub(crate) thresholds: Vec<[f32; 2]>,
    /// `1 + 1/count`, then the trim factors for a member and for an
    /// extreme, then unused.
    pub(crate) factors: Vec<[f32; 4]>,
    pub(crate) minimum_sigma: f32,
    pub(crate) clipped_correction: f32,
    pub(crate) precision: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BandParams {
    samples: u32,
    frames: u32,
    channels: u32,
    weight_channels: u32,
    weighted: u32,
    minimum_sigma: f32,
    clipped_correction: f32,
    precision: f32,
}

/// One band's outputs, as `batch::Outputs` holds them.
pub(crate) struct BandOutputs<'a> {
    pub(crate) mean: &'a mut [f32],
    pub(crate) variance: &'a mut [f32],
    /// Empty unless the stack is weighted.
    pub(crate) weight: &'a mut [f32],
    pub(crate) coverage: &'a mut [u32],
    pub(crate) rejected: &'a mut [u32],
}

/// The GPU's buffers for integrating one stack band by band, made once.
pub(crate) struct BandSession {
    gpu: &'static Gpu,
    channels: usize,
    weight_channels: usize,
    weighted: bool,
    band_capacity: usize,
    frame_count: usize,
    values: wgpu::Buffer,
    /// Each frame's weight and its reciprocal per weight channel, frame
    /// after frame; the GPU holds those of the frames a band reaches.
    weight_rows: Vec<[f32; 2]>,
    weights: wgpu::Buffer,
    params: wgpu::Buffer,
    moments: wgpu::Buffer,
    tallies: wgpu::Buffer,
    counts: wgpu::Buffer,
    moments_read: Option<wgpu::Buffer>,
    tallies_read: Option<wgpu::Buffer>,
    counts_read: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    precision: BandTables,
}

impl BandSession {
    /// A session for `frame_count` frames of `channels` channels, bands of
    /// up to `band_samples` samples, weighted per `weight_channels` channels
    /// when `weights` is given (one row per frame), or `None` when the CPU
    /// should integrate: a small band, too many frames, or no GPU.
    pub(crate) fn new(
        frame_count: usize,
        channels: usize,
        band_samples: usize,
        weights: Option<(&[f32], usize)>,
        tables: BandTables,
    ) -> Option<Self> {
        if frame_count == 0
            || frame_count > MAX_BAND_FRAMES
            || band_samples * frame_count < MINIMUM_SAMPLES
            || std::env::var("SEIZA_GPU_BANDS").is_ok_and(|value| value == "off")
        {
            return None;
        }
        let gpu = gpu()?;
        let values_bytes = (band_samples * frame_count * 4) as u64;
        if values_bytes > gpu.max_buffer || (band_samples as u64) * 16 > gpu.max_buffer {
            return None;
        }
        let weighted = weights.is_some();
        let (weight_rows, weight_channels) = match weights {
            Some((weights, weight_channels)) => (
                weights
                    .iter()
                    .map(|&weight| [weight, 1.0 / weight])
                    .collect::<Vec<_>>(),
                weight_channels,
            ),
            None => (vec![[1.0_f32, 1.0]; frame_count], 1),
        };
        let init = |label: &str, contents: &[u8], usage: wgpu::BufferUsages| {
            wgpu::util::DeviceExt::create_buffer_init(
                &gpu.device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage,
                },
            )
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let values = gpu.buffer(
            "band-values",
            values_bytes,
            storage
                | if gpu.mapped {
                    wgpu::BufferUsages::MAP_WRITE
                } else {
                    wgpu::BufferUsages::COPY_DST
                },
        );
        let thresholds = init(
            "band-thresholds",
            bytemuck::cast_slice(&tables.thresholds),
            storage,
        );
        let factors = init(
            "band-factors",
            bytemuck::cast_slice(&tables.factors),
            storage,
        );
        let weights = gpu.buffer(
            "band-weights",
            (weight_rows.len() * 8) as u64,
            storage | wgpu::BufferUsages::COPY_DST,
        );
        let params = gpu.buffer(
            "band-params",
            std::mem::size_of::<BandParams>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let readable = if gpu.mapped {
            wgpu::BufferUsages::MAP_READ
        } else {
            wgpu::BufferUsages::COPY_SRC
        };
        let moments = gpu.buffer(
            "band-moments",
            (band_samples * 16) as u64,
            storage | readable,
        );
        let tallies = gpu.buffer(
            "band-tallies",
            (band_samples * 8) as u64,
            storage | readable,
        );
        let counts = gpu.buffer(
            "band-counts",
            (frame_count * 8) as u64,
            storage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let staging = |label: &str, size: u64| {
            gpu.buffer(
                label,
                size,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            )
        };
        let moments_read =
            (!gpu.mapped).then(|| staging("band-moments-read", (band_samples * 16) as u64));
        let tallies_read =
            (!gpu.mapped).then(|| staging("band-tallies-read", (band_samples * 8) as u64));
        let counts_read = staging("band-counts-read", (frame_count * 8) as u64);
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bands"),
            layout: &gpu.bands_layout,
            entries: &[
                (0, &params),
                (1, &values),
                (2, &weights),
                (3, &thresholds),
                (4, &factors),
                (5, &moments),
                (6, &tallies),
                (7, &counts),
            ]
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }),
        });
        Some(Self {
            gpu,
            channels,
            weight_channels,
            weighted,
            band_capacity: band_samples,
            frame_count,
            values,
            weight_rows,
            weights,
            params,
            moments,
            tallies,
            counts,
            moments_read,
            tallies_read,
            counts_read,
            bind_group,
            precision: tables,
        })
    }

    /// Integrate one band: `frames[i]` holds frame `i`'s samples of it, or
    /// is empty when the band does not reach the frame. Writes the band's
    /// outputs and adds each frame's finite and integrated samples to
    /// `counts`.
    pub(crate) fn integrate(
        &self,
        frames: &[&[f32]],
        outputs: BandOutputs<'_>,
        counts: &mut [(usize, usize)],
    ) -> Result<(), String> {
        let gpu = self.gpu;
        let samples = outputs.mean.len();
        if samples > self.band_capacity || frames.len() != self.frame_count {
            return Err("the band does not fit the session".into());
        }
        // Frames the band reaches, in order; the others hold only NaN.
        let reaching: Vec<usize> = (0..frames.len())
            .filter(|&index| !frames[index].is_empty())
            .collect();
        if reaching.is_empty() {
            outputs.mean.fill(0.0);
            return Ok(());
        }
        let bytes = samples * 4;
        if gpu.mapped {
            let range = self.values.slice(..(reaching.len() * bytes) as u64);
            range.map_async(wgpu::MapMode::Write, |_| {});
            gpu.wait()?;
            let mut view = range
                .get_mapped_range_mut()
                .map_err(|error| error.to_string())?;
            for (slot, &index) in reaching.iter().enumerate() {
                view.slice(slot * bytes..(slot + 1) * bytes)
                    .copy_from_slice(bytemuck::cast_slice(frames[index]));
            }
            drop(view);
            self.values.unmap();
        } else {
            for (slot, &index) in reaching.iter().enumerate() {
                gpu.queue.write_buffer(
                    &self.values,
                    (slot * bytes) as u64,
                    bytemuck::cast_slice(frames[index]),
                );
            }
        }
        // The kernel numbers frames as the band reaches them.
        let weights: Vec<[f32; 2]> = reaching
            .iter()
            .flat_map(|&index| {
                self.weight_rows[index * self.weight_channels..(index + 1) * self.weight_channels]
                    .iter()
                    .copied()
            })
            .collect();
        gpu.queue
            .write_buffer(&self.weights, 0, bytemuck::cast_slice(&weights));
        let params = BandParams {
            samples: samples as u32,
            frames: reaching.len() as u32,
            channels: self.channels as u32,
            weight_channels: self.weight_channels as u32,
            weighted: u32::from(self.weighted),
            minimum_sigma: self.precision.minimum_sigma,
            clipped_correction: self.precision.clipped_correction,
            precision: self.precision.precision,
        };
        gpu.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        gpu.queue
            .write_buffer(&self.counts, 0, &vec![0u8; reaching.len() * 8]);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&gpu.bands_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let groups = samples.div_ceil(256) as u32;
            let (x, y) = (groups.min(32768), groups.div_ceil(32768));
            pass.dispatch_workgroups(x, y, 1);
        }
        let output_bytes = (samples * 16) as u64;
        let tally_bytes = (samples * 8) as u64;
        if let (Some(moments_read), Some(tallies_read)) = (&self.moments_read, &self.tallies_read) {
            encoder.copy_buffer_to_buffer(&self.moments, 0, moments_read, 0, output_bytes);
            encoder.copy_buffer_to_buffer(&self.tallies, 0, tallies_read, 0, tally_bytes);
        }
        let count_bytes = (reaching.len() * 8) as u64;
        encoder.copy_buffer_to_buffer(&self.counts, 0, &self.counts_read, 0, count_bytes);
        gpu.queue.submit([encoder.finish()]);
        let moments_from = self.moments_read.as_ref().unwrap_or(&self.moments);
        let tallies_from = self.tallies_read.as_ref().unwrap_or(&self.tallies);
        moments_from
            .slice(..output_bytes)
            .map_async(wgpu::MapMode::Read, |_| {});
        tallies_from
            .slice(..tally_bytes)
            .map_async(wgpu::MapMode::Read, |_| {});
        self.counts_read
            .slice(..count_bytes)
            .map_async(wgpu::MapMode::Read, |_| {});
        gpu.wait()?;
        if FAILED.load(Ordering::Relaxed) {
            return Err("the GPU failed".into());
        }
        {
            let moments = moments_from
                .slice(..output_bytes)
                .get_mapped_range()
                .map_err(|error| error.to_string())?;
            let tallies = tallies_from
                .slice(..tally_bytes)
                .get_mapped_range()
                .map_err(|error| error.to_string())?;
            let moments: &[[f32; 4]] = bytemuck::cast_slice(&moments);
            let tallies: &[[u32; 2]] = bytemuck::cast_slice(&tallies);
            let BandOutputs {
                mean,
                variance,
                weight,
                coverage,
                rejected,
            } = outputs;
            const CHUNK: usize = 1 << 16;
            mean.par_chunks_mut(CHUNK)
                .zip(variance.par_chunks_mut(CHUNK))
                .zip(coverage.par_chunks_mut(CHUNK))
                .zip(rejected.par_chunks_mut(CHUNK))
                .zip(moments.par_chunks(CHUNK))
                .zip(tallies.par_chunks(CHUNK))
                .for_each(
                    |(((((mean, variance), coverage), rejected), moments), tallies)| {
                        for (((((mean, variance), coverage), rejected), moment), tally) in mean
                            .iter_mut()
                            .zip(variance.iter_mut())
                            .zip(coverage.iter_mut())
                            .zip(rejected.iter_mut())
                            .zip(moments)
                            .zip(tallies)
                        {
                            *mean = moment[0];
                            *variance = moment[1];
                            *coverage = tally[0];
                            *rejected = tally[1];
                        }
                    },
                );
            if !weight.is_empty() {
                weight
                    .par_chunks_mut(CHUNK)
                    .zip(moments.par_chunks(CHUNK))
                    .for_each(|(weight, moments)| {
                        for (weight, moment) in weight.iter_mut().zip(moments) {
                            *weight = moment[2];
                        }
                    });
            }
            let frame_counts = self
                .counts_read
                .slice(..count_bytes)
                .get_mapped_range()
                .map_err(|error| error.to_string())?;
            let frame_counts: &[[u32; 2]] = bytemuck::cast_slice(&frame_counts);
            for (&index, count) in reaching.iter().zip(frame_counts) {
                counts[index].0 += count[0] as usize;
                counts[index].1 += count[1] as usize;
            }
        }
        moments_from.unmap();
        tallies_from.unmap();
        self.counts_read.unmap();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registration::{Sampling, SimilarityTransform};

    /// Sky of 100 with noise of 5, stars of every brightness up to
    /// saturated ones, and a patch of `NaN`.
    fn star_field(width: usize, height: usize, channels: usize) -> LinearImage {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ (width * height * channels) as u64;
        let mut uniform = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let stars: Vec<(f64, f64, f64, f64)> = (0..400)
            .map(|_| {
                (
                    uniform() * width as f64,
                    uniform() * height as f64,
                    10.0_f64.powf(1.0 + uniform() * 3.8),
                    0.8 + uniform() * 1.5,
                )
            })
            .collect();
        let mut data = vec![0.0_f32; width * height * channels];
        for (pixel, samples) in data.chunks_exact_mut(channels).enumerate() {
            let (x, y) = ((pixel % width) as f64, (pixel / width) as f64);
            let mut light = 0.0;
            for &(sx, sy, peak, sigma) in &stars {
                let (dx, dy) = (x - sx, y - sy);
                if dx.abs() < 8.0 && dy.abs() < 8.0 {
                    light += peak * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
                }
            }
            for (channel, sample) in samples.iter_mut().enumerate() {
                let noise = (uniform() + uniform() + uniform() - 1.5) * 10.0;
                let value = (100.0 + light * (1.0 - 0.2 * channel as f64) + noise).min(60_000.0);
                *sample = value as f32;
            }
        }
        for y in 300..310 {
            for x in 400..420 {
                for channel in 0..channels {
                    data[(y * width + x) * channels + channel] = f32::NAN;
                }
            }
        }
        LinearImage::new(width, height, channels, data).unwrap()
    }

    /// The GPU resamples as the CPU does, to a small fraction of the noise,
    /// with the same coverage, over a whole grid and a region of it.
    #[test]
    fn gpu_resampling_matches_the_cpu() {
        let Some(adapter) = adapter_description() else {
            eprintln!("no GPU: GPU resampling not compared");
            return;
        };
        eprintln!("comparing resampling on {adapter}");
        let transform = SimilarityTransform {
            scale: 1.0004,
            rotation_radians: 0.4_f64.to_radians(),
            translation_x: 3.3,
            translation_y: -2.7,
        };
        let inverse = transform.inverse_map();
        for (channels, lanczos) in [(1, true), (3, true), (1, false)] {
            let source = star_field(1100, 1000, channels);
            let spans = crate::registration::lanczos_clamp_spans(&source);
            let mut clamp_spans = [0.0; 4];
            clamp_spans[..channels].copy_from_slice(&spans);
            let (gpu_sampling, cpu_sampling) = if lanczos {
                (GpuSampling::Lanczos3 { clamp_spans }, Sampling::Lanczos3)
            } else {
                (GpuSampling::Bilinear, Sampling::Bilinear)
            };
            for region in [
                ReferenceRegion {
                    x: 0,
                    y: 0,
                    width: 1100,
                    height: 1000,
                },
                ReferenceRegion {
                    x: 37,
                    y: 501,
                    width: 700,
                    height: 333,
                },
            ] {
                let what = format!("{channels} channel(s), lanczos {lanczos}, {region:?}");
                let gpu = resample(&source, region, &inverse, gpu_sampling).expect(&what);
                let cpu = crate::registration::resample_region_on_cpu(
                    &source,
                    region,
                    &inverse,
                    cpu_sampling,
                    &spans,
                )
                .unwrap();
                let (mut squares, mut both, mut far, mut coverage) = (0.0_f64, 0, 0, 0);
                for (&gpu, &cpu) in gpu.data.iter().zip(&cpu.data) {
                    if gpu.is_finite() != cpu.is_finite() {
                        coverage += 1;
                    } else if cpu.is_finite() {
                        let difference = f64::from(gpu - cpu) / 5.0;
                        squares += difference * difference;
                        both += 1;
                        far += usize::from(difference.abs() > 0.1);
                    }
                }
                let rms = (squares / both as f64).sqrt();
                assert!(rms < 0.005, "{what}: RMS {rms} of the noise");
                assert!(
                    far * 10_000 < both,
                    "{what}: {far} of {both} beyond 0.1 noise"
                );
                assert!(
                    coverage * 10_000 < both,
                    "{what}: coverage differs at {coverage}"
                );
            }
        }
    }

    /// The GPU's band integration decides and integrates as the CPU's does,
    /// weighted or not, mono or colour, on frames with heavy outliers.
    #[test]
    fn gpu_band_integration_matches_the_cpu() {
        let Some(adapter) = adapter_description() else {
            eprintln!("no GPU: GPU band integration not compared");
            return;
        };
        eprintln!("comparing band integration on {adapter}");
        for (width, height, channels, depth, weighted) in [
            (400, 300, 3, 12, false),
            (400, 300, 3, 12, true),
            (300, 200, 1, 25, true),
        ] {
            let what = format!("{width}x{height}x{channels}, {depth} frames, weighted {weighted}");
            let frames = crate::batch::tests::synthetic_frames(width, height, channels, depth, 7);
            let options = crate::BatchStackOptions {
                frame_weights: weighted
                    .then(|| crate::batch::tests::pattern_weights(depth, channels)),
                scratch_directory: Some(tempfile::tempdir().unwrap().keep()),
                ..crate::BatchStackOptions::default()
            };
            let cpu = crate::integrate_registered_frames(depth, &options, |_, index| {
                Ok(frames[index].clone())
            })
            .unwrap();
            let gpu = crate::integrate_registered_frames_banded(
                depth,
                &options,
                |index| Ok(frames[index].clone()),
                |_, _, _| {},
            )
            .unwrap();
            let samples = cpu.snapshot.coverage.len();
            let same = (0..samples)
                .filter(|&at| {
                    cpu.snapshot.coverage[at] == gpu.snapshot.coverage[at]
                        && cpu.snapshot.rejected_samples[at] == gpu.snapshot.rejected_samples[at]
                })
                .count();
            assert!(
                same * 1000 >= samples * 999,
                "{what}: {same} of {samples} decided alike"
            );
            let mut far = 0;
            for (&gpu, &cpu) in gpu.snapshot.image.data.iter().zip(&cpu.snapshot.image.data) {
                assert_eq!(gpu.is_finite(), cpu.is_finite(), "{what}");
                if cpu.is_finite() && (gpu - cpu).abs() > 1.0e-3 * cpu.abs().max(1.0) {
                    far += 1;
                }
            }
            assert!(far * 1000 < samples, "{what}: {far} means apart");
            for (gpu, cpu) in gpu.frames.iter().zip(&cpu.frames) {
                assert_eq!(gpu.finite_samples, cpu.finite_samples, "{what}");
                let apart = gpu.integrated_samples.abs_diff(cpu.integrated_samples);
                assert!(
                    apart * 1000 < cpu.finite_samples,
                    "{what}: integrated {apart} apart"
                );
            }
        }
    }

    /// Both shaders parse and validate, so a machine with a GPU never meets
    /// one that does not compile; this runs without a GPU.
    #[test]
    fn the_shaders_validate() {
        for (name, source) in [
            ("resample", include_str!("gpu/resample.wgsl")),
            ("bands", include_str!("gpu/bands.wgsl")),
        ] {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(source)));
        }
    }
}
