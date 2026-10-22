//! Stacking's heavy passes on the GPU, through wgpu: Metal, Vulkan or
//! Direct3D 12, whichever the platform has. [`defect_candidates`] screens
//! a light for defect learning, [`vng`] debayers a Bayer mosaic's interior,
//! [`detect_stars`] finds a frame's stars, [`resample`] registers a frame onto the reference grid
//! (clamped Lanczos-3 or bilinear), [`GpuMoments`] holds a live stack's
//! accumulator and [`BandSession`] runs the final integration's rejection
//! passes, each as its CPU twin does in f32. A resampled sample differs from
//! the CPU's by a few thousandths of the frame's noise at most, except where
//! a tap span lies on the clamping threshold and the two take different
//! branches; `docs/design/gpu-stacking.md` gives each stage's accuracy.
//!
//! The device is opened once, the first time a frame large enough to gain
//! from it arrives. Setting `SEIZA_GPU=off` (or `0`, `cpu`) before then keeps
//! everything on the CPU, as a build without the `gpu` feature does. When no
//! adapter is found, or the GPU fails, the work stays on the CPU; a failure
//! is logged once.
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
    classify_pipeline: wgpu::ComputePipeline,
    integrate_pipeline: wgpu::ComputePipeline,
    accumulate_layout: wgpu::BindGroupLayout,
    balance_pipeline: wgpu::ComputePipeline,
    vng_pipeline: wgpu::ComputePipeline,
    vng_layout: wgpu::BindGroupLayout,
    defects_pipeline: wgpu::ComputePipeline,
    defects_layout: wgpu::BindGroupLayout,
    /// Star detection's passes, in order, and their layout.
    detect_pipelines: [wgpu::ComputePipeline; 6],
    detect_layout: wgpu::BindGroupLayout,
    weights: wgpu::Buffer,
    /// An integrated GPU, which shares memory with the CPU, reads the
    /// source and writes the output in buffers the CPU maps directly;
    /// a discrete one goes through staging copies.
    mapped: bool,
    max_buffer: u64,
    /// Frame-sized buffers kept between calls: making a new one maps and
    /// clears fresh memory, which took longer than the upload itself.
    pool: std::sync::Mutex<Vec<(wgpu::BufferUsages, wgpu::Buffer)>>,
}

/// Bytes each core copies at a time out of a mapped buffer.
const COPY_CHUNK: usize = 1 << 20;

/// A mapped buffer's words, copied straight into a vector rather than over
/// zeros written first, which on a frame cost as much as the copy. On
/// `every_core`, for a read nothing else waits beside, such as a live
/// stack's moments: one core alone reads write-combined memory, as an
/// integrated GPU's can be, several times slower. A frame's output is read
/// on one core, since frames are read on several at once: copying each on
/// every core cost 30 CPU-s more on the M45 stack on an Iris Xe, for no
/// time saved.
fn copy_out<T: bytemuck::Pod + Send + Sync>(view: &[u8], every_core: bool) -> Vec<T> {
    let Ok(words) = bytemuck::try_cast_slice::<u8, T>(view) else {
        // A mapping is aligned to 8 bytes, so this is not reached.
        return bytemuck::pod_collect_to_vec(view);
    };
    if !every_core {
        return words.to_vec();
    }
    let mut out = Vec::new();
    words
        .par_iter()
        .with_min_len(COPY_CHUNK / std::mem::size_of::<T>())
        .copied()
        .collect_into_vec(&mut out);
    out
}

/// Bytes the pool keeps at most, past which a returned buffer is dropped.
const POOL_BYTES: u64 = 1 << 30;

static GPU: OnceLock<Option<Gpu>> = OnceLock::new();

/// Each stage's time in one GPU call, printed to stderr when
/// `SEIZA_GPU_TRACE` is set, to see where the time goes on a machine.
pub(crate) struct Trace {
    enabled: bool,
    last: std::time::Instant,
    parts: Vec<(&'static str, f64)>,
}

impl Trace {
    pub(crate) fn new() -> Self {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        Self {
            enabled: *ENABLED.get_or_init(|| std::env::var_os("SEIZA_GPU_TRACE").is_some()),
            last: std::time::Instant::now(),
            parts: Vec::new(),
        }
    }

    pub(crate) fn mark(&mut self, stage: &'static str) {
        if self.enabled {
            let now = std::time::Instant::now();
            self.parts
                .push((stage, (now - self.last).as_secs_f64() * 1000.0));
            self.last = now;
        }
    }

    pub(crate) fn print(&self, what: &str) {
        if self.enabled {
            let total: f64 = self.parts.iter().map(|(_, ms)| ms).sum();
            let parts = self
                .parts
                .iter()
                .map(|(stage, ms)| format!("{stage} {ms:.1}"))
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!("seiza-stacking gpu: {what}: {total:.1} ms ({parts})");
        }
    }
}
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
        let mut trace = Trace::new();
        let opened = Gpu::open();
        trace.mark("open");
        trace.print("device");
        match opened {
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
        let accumulate_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("accumulate"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu/accumulate.wgsl").into()),
        });
        let accumulate_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("accumulate"),
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
                storage(2, false),
                storage(3, false),
                storage(4, false),
                storage(5, false),
                storage(6, false),
                storage(7, false),
            ],
        });
        let accumulate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("accumulate"),
                bind_group_layouts: &[Some(&accumulate_layout)],
                immediate_size: 0,
            });
        let accumulate_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&accumulate_pipeline_layout),
                module: &accumulate_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let classify_pipeline = accumulate_pipeline("classify");
        let integrate_pipeline = accumulate_pipeline("integrate");
        let vng_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vng"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu/vng.wgsl").into()),
        });
        let vng_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vng"),
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
                storage(2, false),
                storage(3, false),
            ],
        });
        let vng_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vng"),
            bind_group_layouts: &[Some(&vng_layout)],
            immediate_size: 0,
        });
        let vng_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&vng_pipeline_layout),
                module: &vng_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let balance_pipeline = vng_pipeline("balance");
        let vng_pipeline = vng_pipeline("main");
        let defects_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("defects"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu/defects.wgsl").into()),
        });
        let defects_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("defects"),
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
                storage(2, false),
            ],
        });
        let defects_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("defects"),
                bind_group_layouts: &[Some(&defects_layout)],
                immediate_size: 0,
            });
        let defects_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("defects"),
            layout: Some(&defects_pipeline_layout),
            module: &defects_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let detect_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("detect"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu/detect.wgsl").into()),
        });
        let detect_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("detect"),
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
                storage(2, false),
                storage(3, false),
                storage(4, false),
                storage(5, false),
                storage(6, false),
            ],
        });
        let detect_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("detect"),
                bind_group_layouts: &[Some(&detect_layout)],
                immediate_size: 0,
            });
        let detect_pipelines = [
            "luminance",
            "normalize",
            "background",
            "count_rows",
            "offsets",
            "write_rows",
        ]
        .map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&detect_pipeline_layout),
                module: &detect_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
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
            classify_pipeline,
            integrate_pipeline,
            accumulate_layout,
            balance_pipeline,
            vng_pipeline,
            vng_layout,
            defects_pipeline,
            defects_layout,
            detect_pipelines,
            detect_layout,
            weights,
            mapped,
            max_buffer: limits
                .max_buffer_size
                .min(limits.max_storage_buffer_binding_size),
            pool: std::sync::Mutex::new(Vec::new()),
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

    /// A buffer of `usage` holding at least `size` bytes, from the pool when
    /// one there is no more than twice as large, or else a new one.
    fn take(&self, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let mut pool = self
            .pool
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let found = pool
            .iter()
            .enumerate()
            .filter(|(_, (kind, buffer))| {
                *kind == usage && buffer.size() >= size && buffer.size() <= 2 * size
            })
            .min_by_key(|(_, (_, buffer))| buffer.size())
            .map(|(index, _)| index);
        match found {
            Some(index) => pool.swap_remove(index).1,
            None => {
                drop(pool);
                self.buffer(label, size, usage)
            }
        }
    }

    /// Return a buffer [`Self::take`] gave, unmapped, for a later call.
    fn give(&self, usage: wgpu::BufferUsages, buffer: wgpu::Buffer) {
        let mut pool = self
            .pool
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let held: u64 = pool.iter().map(|(_, buffer)| buffer.size()).sum();
        if held + buffer.size() <= POOL_BYTES {
            pool.push((usage, buffer));
        }
    }

    /// The usage of a buffer the host fills and the GPU reads.
    fn input_usage(&self) -> wgpu::BufferUsages {
        wgpu::BufferUsages::STORAGE
            | if self.mapped {
                wgpu::BufferUsages::MAP_WRITE
            } else {
                wgpu::BufferUsages::COPY_DST
            }
    }

    /// A pooled buffer of [`Self::input_usage`] holding `data`.
    fn upload(&self, label: &str, data: &[f32]) -> Result<wgpu::Buffer, String> {
        let bytes = std::mem::size_of_val(data) as u64;
        let buffer = self.take(label, bytes, self.input_usage());
        if self.mapped {
            self.map(&[buffer.slice(..bytes)], wgpu::MapMode::Write, None)?;
            // One core writes: copying on every core, the call waited for
            // its parts behind other frames' work in the pool, and on the
            // M45 stack took 0.5 s longer and, on an Iris Xe, 40 CPU-s more.
            let copied = buffer
                .slice(..bytes)
                .get_mapped_range_mut()
                .map(|mut view| view.slice(..).copy_from_slice(bytemuck::cast_slice(data)));
            buffer.unmap();
            copied.map_err(|error| error.to_string())?;
        } else {
            self.queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        }
        Ok(buffer)
    }

    /// Run `passes`, each a pipeline of `layout` over its workgroups, in
    /// order, their bindings `inputs` and then an output of
    /// `output_samples`, and read the output back.
    fn compute<T: bytemuck::Pod + Send + Sync>(
        &self,
        layout: &wgpu::BindGroupLayout,
        passes: &[(&wgpu::ComputePipeline, (u32, u32))],
        inputs: &[&wgpu::Buffer],
        output_samples: usize,
        trace: &mut Trace,
    ) -> Result<Vec<T>, String> {
        let output_bytes = (output_samples * std::mem::size_of::<T>()) as u64;
        let output_usage = wgpu::BufferUsages::STORAGE
            | if self.mapped {
                wgpu::BufferUsages::MAP_READ
            } else {
                wgpu::BufferUsages::COPY_SRC
            };
        let output = self.take("output", output_bytes, output_usage);
        let readback_usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let readback = (!self.mapped).then(|| self.take("readback", output_bytes, readback_usage));
        let entries: Vec<wgpu::BindGroupEntry> = inputs
            .iter()
            .chain([&&output])
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &entries,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &bind_group, &[]);
            for &(pipeline, (columns, rows)) in passes {
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(columns, rows, 1);
            }
        }
        if let Some(readback) = &readback {
            encoder.copy_buffer_to_buffer(&output, 0, readback, 0, output_bytes);
        }
        trace.mark("upload");
        let submission = self.queue.submit([encoder.finish()]);
        let read_from = readback.as_ref().unwrap_or(&output);
        let result = self
            .map(
                &[read_from.slice(..output_bytes)],
                wgpu::MapMode::Read,
                Some(submission),
            )
            .and_then(|()| {
                trace.mark("kernel");
                if FAILED.load(Ordering::Relaxed) {
                    return Err("the GPU failed".to_string());
                }
                let view = read_from
                    .slice(..output_bytes)
                    .get_mapped_range()
                    .map_err(|error| error.to_string())?;
                let data = copy_out(&view, false);
                drop(view);
                read_from.unmap();
                trace.mark("copy out");
                Ok(data)
            });
        self.give(output_usage, output);
        if let Some(readback) = readback {
            self.give(readback_usage, readback);
        }
        result
    }

    /// Wait for `submission`, and those before it, to finish, but not for
    /// work other threads submitted since.
    fn wait_for(&self, submission: wgpu::SubmissionIndex) -> Result<(), String> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    /// Map `slices` for reading once `submission`, which last uses them, is
    /// done, or for writing at once when no submission in flight uses them,
    /// as a buffer from the pool or a session's between calls. Waiting for
    /// the queue to empty instead held every thread behind every other's
    /// kernels.
    fn map(
        &self,
        slices: &[wgpu::BufferSlice<'_>],
        mode: wgpu::MapMode,
        submission: Option<wgpu::SubmissionIndex>,
    ) -> Result<(), String> {
        let (sender, receiver) = std::sync::mpsc::channel();
        for slice in slices {
            let sender = sender.clone();
            slice.map_async(mode, move |result| {
                let _ = sender.send(result);
            });
        }
        drop(sender);
        match submission {
            Some(submission) => self.wait_for(submission)?,
            None => {
                self.device
                    .poll(wgpu::PollType::Poll)
                    .map_err(|error| error.to_string())?;
            }
        }
        // Whichever thread polls when a mapping is ready runs its callback,
        // a moment after any poll returns, so the callbacks are awaited
        // rather than looked for.
        let started = std::time::Instant::now();
        let mut pending = slices.len();
        let mut waited_for_all = false;
        while pending > 0 {
            match receiver.recv_timeout(std::time::Duration::from_millis(10)) {
                Ok(result) => {
                    result.map_err(|error| error.to_string())?;
                    pending -= 1;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    return Err("a buffer's mapping was dropped".into());
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if FAILED.load(Ordering::Relaxed) {
                        return Err("the GPU failed".into());
                    }
                    if !waited_for_all && started.elapsed().as_millis() > 100 {
                        // Still in use after all: wait for the queue.
                        waited_for_all = true;
                        self.wait()?;
                    } else {
                        self.device
                            .poll(wgpu::PollType::Poll)
                            .map_err(|error| error.to_string())?;
                    }
                }
            }
        }
        Ok(())
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
        trace: &mut Trace,
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
        let source_buffer = self.upload("source", &source.data)?;
        let workgroups = (
            (region.width as u32).div_ceil(16),
            (region.height as u32).div_ceil(16),
        );
        let data = self.compute(
            &self.layout,
            &[(&self.pipeline, workgroups)],
            &[&uniform, &source_buffer, &grid, &self.weights],
            output_samples,
            trace,
        );
        self.give(self.input_usage(), source_buffer);
        let data = data?;
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
    let mut trace = Trace::new();
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
    trace.mark("grid");
    let result = gpu.resample(
        source,
        region,
        &grid,
        (columns as u32, origin),
        sampling,
        &mut trace,
    );
    trace.print(&format!(
        "resample {}x{}x{channels}",
        region.width, region.height
    ));
    match result {
        Ok(image) => Some(image),
        Err(reason) => {
            log_once(&format!("resampling this frame on the CPU: {reason}"));
            None
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VngParams {
    width: u32,
    height: u32,
    padding: [u32; 2],
    balance: [f32; 4],
    channels: [u32; 4],
}

/// VNG demosaicing of a `width` by `height` Bayer `mosaic` on the GPU, as
/// `image::vng_rows` does it on the mosaic balanced by `balance`: the
/// interleaved RGB of every pixel two or more from each edge, scaled back
/// from the balance and with its own sample, and anything on the two-pixel
/// border, which the caller fills. `channels` is the colour of each Bayer
/// phase, column parity | row parity << 1. `None` when the CPU should: the
/// mosaic is small, `SEIZA_GPU_DEBAYER=off`, or no GPU is in use.
pub(crate) fn vng(
    mosaic: &[f32],
    (width, height): (usize, usize),
    channels: [u32; 4],
    balance: [f32; 3],
) -> Option<Vec<f32>> {
    if mosaic.len() < MINIMUM_SAMPLES
        || width < 5
        || height < 5
        || std::env::var("SEIZA_GPU_DEBAYER").is_ok_and(|value| value == "off")
    {
        return None;
    }
    let gpu = gpu()?;
    if (mosaic.len() as u64) * 12 > gpu.max_buffer {
        return None;
    }
    let mut trace = Trace::new();
    let params = VngParams {
        width: width as u32,
        height: height as u32,
        padding: [0; 2],
        balance: [balance[0], balance[1], balance[2], 1.0],
        channels,
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
    let uniform = init(
        "vng-params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    );
    let balanced_bytes = (mosaic.len() * 4) as u64;
    let balanced = gpu.take("balanced", balanced_bytes, wgpu::BufferUsages::STORAGE);
    let result = gpu.upload("mosaic", mosaic).and_then(|source| {
        let rgb = gpu.compute(
            &gpu.vng_layout,
            &[
                (
                    &gpu.balance_pipeline,
                    ((width as u32).div_ceil(16), (height as u32).div_ceil(16)),
                ),
                (
                    &gpu.vng_pipeline,
                    (
                        (width as u32 - 4).div_ceil(16),
                        (height as u32 - 4).div_ceil(16),
                    ),
                ),
            ],
            &[&uniform, &source, &balanced],
            mosaic.len() * 3,
            &mut trace,
        );
        gpu.give(gpu.input_usage(), source);
        rgb
    });
    gpu.give(wgpu::BufferUsages::STORAGE, balanced);
    trace.print(&format!("vng {width}x{height}"));
    match result {
        Ok(rgb) => Some(rgb),
        Err(reason) => {
            log_once(&format!("debayering this frame on the CPU: {reason}"));
            None
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DefectParams {
    width: u32,
    height: u32,
    channels: u32,
    step: u32,
    samples: u32,
    row: u32,
    padding: [u32; 2],
    thresholds: [f32; 4],
}

/// The first test `defects::frame_deviations` makes of every sample, on the
/// GPU: a bit for each sample of `data` (`width` by `height`, `channels`
/// interleaved, same-plane neighbours `step` apart), 32 to a word, set where
/// the sample may stand more than its plane's threshold from its eight
/// neighbours' extremes. The test is loosened a little, so every sample the
/// CPU's exact test keeps is flagged, and some it would not. `thresholds`
/// holds `sigma` times each plane's noise, by `channel * step^2 + plane`.
/// `None` when the CPU should: the frame is small,
/// `SEIZA_GPU_DEFECTS=off`, or no GPU is in use.
pub(crate) fn defect_candidates(
    data: &[f32],
    (width, height, channels, step): (usize, usize, usize, usize),
    thresholds: [f32; 4],
) -> Option<Vec<u32>> {
    if data.len() < MINIMUM_SAMPLES
        || data.len() != width * height * channels
        || channels * step * step > 4
        || data.len() > u32::MAX as usize
        || std::env::var("SEIZA_GPU_DEFECTS").is_ok_and(|value| value == "off")
    {
        return None;
    }
    let gpu = gpu()?;
    if (data.len() as u64) * 4 > gpu.max_buffer {
        return None;
    }
    let mut trace = Trace::new();
    let groups = data.len().div_ceil(256) as u32;
    let columns = groups.min(4096);
    let params = DefectParams {
        width: width as u32,
        height: height as u32,
        channels: channels as u32,
        step: step as u32,
        samples: data.len() as u32,
        row: columns * 256,
        padding: [0; 2],
        thresholds,
    };
    let uniform = wgpu::util::DeviceExt::create_buffer_init(
        &gpu.device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("defect-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        },
    );
    let result = gpu.upload("defects", data).and_then(|source| {
        let flags = gpu.compute::<u32>(
            &gpu.defects_layout,
            &[(&gpu.defects_pipeline, (columns, groups.div_ceil(columns)))],
            &[&uniform, &source],
            data.len().div_ceil(32),
            &mut trace,
        );
        gpu.give(gpu.input_usage(), source);
        flags
    });
    trace.print(&format!("defects {width}x{height}x{channels}"));
    match result {
        Ok(flags) => Some(flags),
        Err(reason) => {
            log_once(&format!(
                "screening this frame for defects on the CPU: {reason}"
            ));
            None
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DetectParams {
    width: u32,
    height: u32,
    channels: u32,
    tile: u32,
    tiles_x: u32,
    tiles_y: u32,
    sigma: f32,
    capacity: u32,
    row: u32,
    padding: [u32; 3],
}

/// Samples a tile holds at most for the GPU's background pass, whose
/// workgroup holds a tile.
const DETECTION_TILE_SAMPLES: u32 = 4096;

/// Star detection's per-pixel work on the GPU, as `registration::detect`
/// does it on the CPU, and the stars `seiza::detect_stars_from_excess`
/// gathers from the pixels that pass. The luminance is normalized in f32
/// where the CPU works in f64, so a pixel on the threshold can fall either
/// way; the tiles' medians and robust sigmas are exact. `None` when the CPU
/// should: the frame is small, its tiles too large to sort, more pixels
/// pass than the GPU keeps room for, `SEIZA_GPU_DETECT=off`, or no GPU is
/// in use.
pub(crate) fn detect_stars(
    image: &LinearImage,
    config: &seiza::DetectConfig,
) -> Option<Vec<seiza::DetectedStar>> {
    let (width, height, channels) = (image.width, image.height, image.channels);
    let pixels = width * height;
    let tile = config.tile_size;
    if !matches!(channels, 1 | 3)
        || pixels < MINIMUM_SAMPLES
        || tile == 0
        || tile.saturating_mul(tile) > DETECTION_TILE_SAMPLES
        || width > u32::MAX as usize / 2
        || height > u32::MAX as usize / 2
        || config.backend == seiza::DetectBackend::U8
        || std::env::var("SEIZA_GPU_DETECT").is_ok_and(|value| value == "off")
    {
        return None;
    }
    let gpu = gpu()?;
    if (image.data.len() as u64) * 4 > gpu.max_buffer {
        return None;
    }
    let mut trace = Trace::new();
    let result = gpu.detect(image, config, &mut trace);
    trace.print(&format!("detect {width}x{height}x{channels}"));
    match result {
        Ok(Some(positives)) => Some(seiza::detect_stars_from_excess(
            width as u32,
            height as u32,
            &positives,
            config,
        )),
        Ok(None) => None,
        Err(reason) => {
            log_once(&format!(
                "detecting this frame's stars on the CPU: {reason}"
            ));
            None
        }
    }
}

impl Gpu {
    /// [`detect_stars`]' passes: the pixels that pass, or `None` when more
    /// pass than there is room for.
    fn detect(
        &self,
        image: &LinearImage,
        config: &seiza::DetectConfig,
        trace: &mut Trace,
    ) -> Result<Option<Vec<(u32, f32)>>, String> {
        let (width, height) = (image.width as u32, image.height as u32);
        let pixels = width * height;
        let tile = config.tile_size;
        let (tiles_x, tiles_y) = (width.div_ceil(tile), height.div_ceil(tile));
        // Stars and noise above the threshold are a few hundredths of a
        // frame; nebulosity a tile's median leaves behind can be more.
        let capacity = (pixels / 8).max(1 << 16).min((self.max_buffer / 8) as u32);
        let groups = pixels.div_ceil(256);
        let columns = groups.min(4096);
        let params = DetectParams {
            width,
            height,
            channels: image.channels as u32,
            tile,
            tiles_x,
            tiles_y,
            sigma: config.sigma,
            capacity,
            row: columns * 256,
            padding: [0; 3],
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
            "detect-params",
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
        );
        let extremes = init(
            "detect-extremes",
            bytemuck::cast_slice(&[u32::MAX, 0]),
            wgpu::BufferUsages::STORAGE,
        );
        let scratch = wgpu::BufferUsages::STORAGE;
        let luma = self.take("detect-luma", u64::from(pixels) * 4, scratch);
        let tiles = self.take("detect-tiles", u64::from(tiles_x * tiles_y) * 8, scratch);
        let rows_bytes = (2 * u64::from(height) + 1) * 4;
        let rows = self.take(
            "detect-rows",
            rows_bytes,
            scratch | wgpu::BufferUsages::COPY_SRC,
        );
        let output_usage = wgpu::BufferUsages::STORAGE
            | if self.mapped {
                wgpu::BufferUsages::MAP_READ
            } else {
                wgpu::BufferUsages::COPY_SRC
            };
        let positives = self.take("detect-positives", u64::from(capacity) * 8, output_usage);
        let total_read = self.buffer(
            "detect-total",
            4,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let frame = self.upload("detect-frame", &image.data);
        let result = frame.and_then(|frame| {
            trace.mark("upload");
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("detect"),
                layout: &self.detect_layout,
                entries: &[
                    (0, &uniform),
                    (1, &frame),
                    (2, &luma),
                    (3, &extremes),
                    (4, &tiles),
                    (5, &rows),
                    (6, &positives),
                ]
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding,
                    resource: buffer.as_entire_binding(),
                }),
            });
            let per_pixel = (columns, groups.div_ceil(columns));
            let per_row = (height.min(65535), height.div_ceil(65535));
            let dispatches = [
                per_pixel,
                per_pixel,
                (tiles_x, tiles_y),
                per_row,
                (1, 1),
                per_row,
            ];
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_bind_group(0, &bind_group, &[]);
                for (pipeline, (x, y)) in self.detect_pipelines.iter().zip(dispatches) {
                    pass.set_pipeline(pipeline);
                    pass.dispatch_workgroups(x, y, 1);
                }
            }
            encoder.copy_buffer_to_buffer(&rows, rows_bytes - 4, &total_read, 0, 4);
            let submission = self.queue.submit([encoder.finish()]);
            let mapped = self.map(
                &[total_read.slice(..)],
                wgpu::MapMode::Read,
                Some(submission),
            );
            self.give(self.input_usage(), frame);
            mapped
        });
        let found = result.and_then(|()| {
            trace.mark("kernel");
            if FAILED.load(Ordering::Relaxed) {
                return Err("the GPU failed".to_string());
            }
            let total: u32 = bytemuck::pod_read_unaligned(
                &total_read
                    .slice(..)
                    .get_mapped_range()
                    .map_err(|error| error.to_string())?[..4],
            );
            total_read.unmap();
            if total > capacity {
                return Ok(None);
            }
            let bytes = u64::from(total) * 8;
            if bytes == 0 {
                return Ok(Some(Vec::new()));
            }
            let read_from = if self.mapped {
                self.map(&[positives.slice(..bytes)], wgpu::MapMode::Read, None)?;
                None
            } else {
                let readback = self.buffer(
                    "detect-readback",
                    bytes,
                    wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                );
                let mut encoder = self.device.create_command_encoder(&Default::default());
                encoder.copy_buffer_to_buffer(&positives, 0, &readback, 0, bytes);
                let submission = self.queue.submit([encoder.finish()]);
                self.map(&[readback.slice(..)], wgpu::MapMode::Read, Some(submission))?;
                Some(readback)
            };
            let source = read_from.as_ref().unwrap_or(&positives);
            let view = source
                .slice(..bytes)
                .get_mapped_range()
                .map_err(|error| error.to_string())?;
            let entries: Vec<[u32; 2]> = copy_out(&view, false);
            drop(view);
            source.unmap();
            trace.mark("copy out");
            Ok(Some(
                entries
                    .into_iter()
                    .map(|[index, excess]| (index, f32::from_bits(excess)))
                    .collect(),
            ))
        });
        self.give(scratch, luma);
        self.give(scratch, tiles);
        self.give(scratch | wgpu::BufferUsages::COPY_SRC, rows);
        self.give(output_usage, positives);
        found
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
        let mut trace = Trace::new();
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
            let total = reaching.len() * bytes;
            let range = self.values.slice(..total as u64);
            gpu.map(&[range], wgpu::MapMode::Write, None)?;
            let copied = range.get_mapped_range_mut().map(|mut view| {
                for (slot, &index) in reaching.iter().enumerate() {
                    view.slice(slot * bytes..(slot + 1) * bytes)
                        .copy_from_slice(bytemuck::cast_slice(frames[index]));
                }
            });
            self.values.unmap();
            copied.map_err(|error| error.to_string())?;
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
        trace.mark("upload");
        let submission = gpu.queue.submit([encoder.finish()]);
        let moments_from = self.moments_read.as_ref().unwrap_or(&self.moments);
        let tallies_from = self.tallies_read.as_ref().unwrap_or(&self.tallies);
        gpu.map(
            &[
                moments_from.slice(..output_bytes),
                tallies_from.slice(..tally_bytes),
                self.counts_read.slice(..count_bytes),
            ],
            wgpu::MapMode::Read,
            Some(submission),
        )?;
        trace.mark("kernel");
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
        trace.mark("copy out");
        trace.print(&format!(
            "band of {samples} samples, {} frames",
            reaching.len()
        ));
        Ok(())
    }
}

/// A live stack's moments, as `stack::Accumulator` keeps them.
#[derive(Debug, Default)]
pub(crate) struct MomentVectors {
    pub(crate) mean: Vec<f32>,
    pub(crate) m2: Vec<f32>,
    pub(crate) count: Vec<u32>,
    pub(crate) rejected: Vec<u32>,
    /// `None` when frames are weighted equally.
    pub(crate) weight_sum: Option<Vec<f32>>,
}

/// A live stack's moments, borrowed.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MomentSlices<'a> {
    pub(crate) mean: &'a [f32],
    pub(crate) m2: &'a [f32],
    pub(crate) count: &'a [u32],
    pub(crate) rejected: &'a [u32],
    pub(crate) weight_sum: Option<&'a [f32]>,
}

/// Where a frame's samples land on the grid: rows `first_row..` of the
/// grid, `span` samples of each from `first_sample` on, in a grid whose
/// rows hold `grid_row` samples.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FramePlacement {
    pub(crate) first_row: usize,
    pub(crate) rows: usize,
    pub(crate) first_sample: usize,
    pub(crate) span: usize,
    pub(crate) grid_row: usize,
}

/// Delta-sigma rejection's settings, or `None` for none.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuRejection {
    pub(crate) low_sigma: f32,
    pub(crate) high_sigma: f32,
    pub(crate) warmup: u32,
    pub(crate) minimum_sigma: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AccumulateParams {
    first_row: u32,
    rows: u32,
    first_sample: u32,
    span: u32,
    grid_row: u32,
    channels: u32,
    weighted: u32,
    rejecting: u32,
    low_sigma: f32,
    high_sigma: f32,
    warmup: u32,
    minimum_sigma: f32,
    admit_at: u32,
    padding: [u32; 3],
    weights: [f32; 4],
}

/// A live stack's moments held on the GPU, which classifies frames against
/// them and integrates the frames admitted.
pub(crate) struct GpuMoments {
    gpu: &'static Gpu,
    samples: usize,
    weighted: bool,
    mean: wgpu::Buffer,
    m2: wgpu::Buffer,
    count: wgpu::Buffer,
    rejected: wgpu::Buffer,
    /// One unused word when frames are weighted equally.
    weight_sum: wgpu::Buffer,
    params: wgpu::Buffer,
    counters: wgpu::Buffer,
    counters_read: wgpu::Buffer,
}

impl GpuMoments {
    /// Moments of at least a million samples moved to the GPU, or `None`
    /// when they stay on the CPU.
    pub(crate) fn upload(moments: MomentSlices<'_>) -> Option<Self> {
        if moments.mean.len() < MINIMUM_SAMPLES
            || std::env::var("SEIZA_GPU_ACCUMULATOR").is_ok_and(|value| value == "off")
        {
            return None;
        }
        let gpu = gpu()?;
        let samples = moments.mean.len();
        if (samples as u64) * 4 > gpu.max_buffer {
            return None;
        }
        let init = |label: &str, contents: &[u8]| {
            wgpu::util::DeviceExt::create_buffer_init(
                &gpu.device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                },
            )
        };
        let unused = [0.0_f32];
        let state = Self {
            gpu,
            samples,
            weighted: moments.weight_sum.is_some(),
            mean: init("mean", bytemuck::cast_slice(moments.mean)),
            m2: init("m2", bytemuck::cast_slice(moments.m2)),
            count: init("count", bytemuck::cast_slice(moments.count)),
            rejected: init("rejected", bytemuck::cast_slice(moments.rejected)),
            weight_sum: init(
                "weight-sum",
                bytemuck::cast_slice(moments.weight_sum.unwrap_or(&unused)),
            ),
            params: gpu.buffer(
                "accumulate-params",
                std::mem::size_of::<AccumulateParams>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            counters: gpu.buffer(
                "accumulate-counters",
                16,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            ),
            counters_read: gpu.buffer(
                "accumulate-counters-read",
                16,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
        };
        gpu.wait().ok()?;
        (!FAILED.load(Ordering::Relaxed)).then_some(state)
    }

    /// Classify `frame`, placed as `placement` says, against the moments,
    /// and integrate it if rejection would accept at least `admit_at` of its
    /// finite samples: how many it would accept, and, when it was
    /// integrated, how many were accepted and refused there. One submission,
    /// one wait: the GPU decides the admission itself.
    pub(crate) fn admit(
        &mut self,
        frame: &[f32],
        placement: FramePlacement,
        weights: &[f32],
        rejection: Option<GpuRejection>,
        admit_at: usize,
    ) -> Result<(usize, Option<(usize, usize)>), String> {
        let gpu = self.gpu;
        let mut trace = Trace::new();
        let source = gpu.upload("frame", frame)?;
        trace.mark("upload");
        // No frame holds `u32::MAX` samples, so a count past it, which
        // admits none, admits none here either.
        let admit_at = u32::try_from(admit_at).unwrap_or(u32::MAX);
        let counts = self.dispatch(&source, placement, weights, rejection, admit_at);
        gpu.give(gpu.input_usage(), source);
        let [accepted, _, integrated, refused] = counts?;
        trace.mark("admit");
        trace.print(&format!("admit {} samples", frame.len()));
        let admitted = accepted >= admit_at;
        Ok((
            accepted as usize,
            admitted.then_some((integrated as usize, refused as usize)),
        ))
    }

    fn dispatch(
        &self,
        frame: &wgpu::Buffer,
        placement: FramePlacement,
        weights: &[f32],
        rejection: Option<GpuRejection>,
        admit_at: u32,
    ) -> Result<[u32; 4], String> {
        let gpu = self.gpu;
        let mut frame_weights = [1.0_f32; 4];
        for (weight, &value) in frame_weights.iter_mut().zip(weights) {
            *weight = value;
        }
        let rejection = rejection.unwrap_or(GpuRejection {
            low_sigma: 0.0,
            high_sigma: 0.0,
            warmup: u32::MAX,
            minimum_sigma: 0.0,
        });
        let params = AccumulateParams {
            first_row: placement.first_row as u32,
            rows: placement.rows as u32,
            first_sample: placement.first_sample as u32,
            span: placement.span as u32,
            grid_row: placement.grid_row as u32,
            channels: weights.len().max(1) as u32,
            weighted: u32::from(self.weighted),
            rejecting: u32::from(rejection.warmup != u32::MAX),
            low_sigma: rejection.low_sigma,
            high_sigma: rejection.high_sigma,
            warmup: rejection.warmup,
            minimum_sigma: rejection.minimum_sigma,
            admit_at,
            padding: [0; 3],
            weights: frame_weights,
        };
        gpu.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        gpu.queue.write_buffer(&self.counters, 0, &[0u8; 16]);
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("accumulate"),
            layout: &gpu.accumulate_layout,
            entries: &[
                (0, &self.params),
                (1, frame),
                (2, &self.mean),
                (3, &self.m2),
                (4, &self.count),
                (5, &self.rejected),
                (6, &self.weight_sum),
                (7, &self.counters),
            ]
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }),
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            // Each dispatch sees the last one's writes: `integrate` reads the
            // count `classify` left.
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (
                (placement.span as u32).div_ceil(64),
                (placement.rows as u32).div_ceil(4),
            );
            for pipeline in [&gpu.classify_pipeline, &gpu.integrate_pipeline] {
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(groups.0, groups.1, 1);
            }
        }
        encoder.copy_buffer_to_buffer(&self.counters, 0, &self.counters_read, 0, 16);
        let submission = gpu.queue.submit([encoder.finish()]);
        gpu.map(
            &[self.counters_read.slice(..)],
            wgpu::MapMode::Read,
            Some(submission),
        )?;
        if FAILED.load(Ordering::Relaxed) {
            return Err("the GPU failed".into());
        }
        let counts: [u32; 4] = bytemuck::pod_read_unaligned(
            &self
                .counters_read
                .slice(..)
                .get_mapped_range()
                .map_err(|error| error.to_string())?[..16],
        );
        self.counters_read.unmap();
        Ok(counts)
    }

    /// Samples the moments hold.
    pub(crate) fn samples(&self) -> usize {
        self.samples
    }

    /// Whether frames are weighted.
    pub(crate) fn weighted(&self) -> bool {
        self.weighted
    }

    /// The moments, read back.
    pub(crate) fn read(&self) -> Result<MomentVectors, String> {
        let gpu = self.gpu;
        let mut trace = Trace::new();
        let bytes = (self.samples * 4) as u64;
        let usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let sources: Vec<&wgpu::Buffer> = if self.weighted {
            vec![
                &self.mean,
                &self.m2,
                &self.count,
                &self.rejected,
                &self.weight_sum,
            ]
        } else {
            vec![&self.mean, &self.m2, &self.count, &self.rejected]
        };
        let staging: Vec<wgpu::Buffer> = sources
            .iter()
            .map(|_| gpu.take("moments-read", bytes, usage))
            .collect();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for (source, staging) in sources.iter().zip(&staging) {
            encoder.copy_buffer_to_buffer(source, 0, staging, 0, bytes);
        }
        let submission = gpu.queue.submit([encoder.finish()]);
        let slices: Vec<wgpu::BufferSlice<'_>> = staging
            .iter()
            .map(|staging| staging.slice(..bytes))
            .collect();
        gpu.map(&slices, wgpu::MapMode::Read, Some(submission))?;
        if FAILED.load(Ordering::Relaxed) {
            return Err("the GPU failed".into());
        }
        let mut words: Vec<Vec<u32>> = Vec::with_capacity(staging.len());
        for staging in &staging {
            let view = staging
                .slice(..bytes)
                .get_mapped_range()
                .map_err(|error| error.to_string())?;
            let out = copy_out::<u32>(&view, true);
            drop(view);
            staging.unmap();
            words.push(out);
        }
        for staging in staging {
            gpu.give(usage, staging);
        }
        let float = |words: Vec<u32>| -> Vec<f32> { bytemuck::cast_vec(words) };
        let mut words = words.into_iter();
        let moments = MomentVectors {
            mean: float(words.next().unwrap_or_default()),
            m2: float(words.next().unwrap_or_default()),
            count: words.next().unwrap_or_default(),
            rejected: words.next().unwrap_or_default(),
            weight_sum: words.next().map(float),
        };
        trace.mark("read");
        trace.print(&format!("moments read back, {} samples", self.samples));
        Ok(moments)
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
            ("accumulate", include_str!("gpu/accumulate.wgsl")),
            ("vng", include_str!("gpu/vng.wgsl")),
            ("defects", include_str!("gpu/defects.wgsl")),
            ("detect", include_str!("gpu/detect.wgsl")),
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
