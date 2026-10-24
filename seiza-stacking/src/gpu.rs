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
//! Each entry takes the [`GpuOptions`] of the call it serves, and leaves its
//! work to the CPU when they do not allow its pass. The device is opened
//! once, the first time a call that allows the GPU has a frame large enough
//! to gain from it. When no adapter is found, or the GPU fails, the work
//! stays on the CPU; a failure is logged once.
use crate::image::ClipMask;
use crate::saturation::ClipView;
use crate::{GpuOptions, LinearImage, ReferenceRegion};
use rayon::prelude::*;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

mod timing;
use timing::GpuTimer;
pub(crate) mod gather;
mod memory;

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
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// Resampling a frame and its clip mask at once.
    masked_pipeline: wgpu::ComputePipeline,
    masked_layout: wgpu::BindGroupLayout,
    sparse_masked_pipeline: wgpu::ComputePipeline,
    sparse_masked_layout: wgpu::BindGroupLayout,
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
    detect_pipelines: [wgpu::ComputePipeline; 5],
    detect_layout: wgpu::BindGroupLayout,
    scatter_pipeline: wgpu::ComputePipeline,
    scatter_layout: wgpu::BindGroupLayout,
    normalize_pipeline: wgpu::ComputePipeline,
    normalize_layout: wgpu::BindGroupLayout,
    gather: OnceLock<gather::Pipelines>,
    weights: wgpu::Buffer,
    /// An integrated GPU, which shares memory with the CPU, reads the
    /// source and writes the output in buffers the CPU maps directly;
    /// a discrete one goes through staging copies.
    mapped: bool,
    max_buffer: u64,
    /// Bytes of memory the device has, where its backend says.
    memory: Option<u64>,
    /// Frame-sized buffers kept between calls: making a new one maps and
    /// clears fresh memory, which took longer than the upload itself.
    pool: std::sync::Mutex<Vec<(wgpu::BufferUsages, wgpu::Buffer)>>,
    /// Frame calls running at once, which [`Gpu::admitted`] holds to each
    /// call's [`GpuOptions::concurrency`].
    calls: (std::sync::Mutex<usize>, std::sync::Condvar),
}

/// A frame call's place among those [`Gpu::admitted`] lets run at once,
/// given back when dropped. The GPU runs one kernel at a time, so a frame
/// waiting for one, rather than in the GPU's queue, holds no buffer while it
/// waits.
struct FrameCall<'a>(&'a Gpu);

impl Drop for FrameCall<'_> {
    fn drop(&mut self) {
        let (count, freed) = &self.0.calls;
        *count.lock().unwrap_or_else(|poison| poison.into_inner()) -= 1;
        // Every waiter, as calls may allow different numbers at once.
        freed.notify_all();
    }
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
    // A frame's samples go into a buffer a finished frame gave back.
    let mut out = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let mut recycled = Some(crate::recycle::take(words.len()));
        (&mut recycled as &mut dyn std::any::Any)
            .downcast_mut::<Option<Vec<T>>>()
            .and_then(Option::take)
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    if !every_core {
        out.extend_from_slice(words);
        return out;
    }
    words
        .par_iter()
        .with_min_len(COPY_CHUNK / std::mem::size_of::<T>())
        .copied()
        .collect_into_vec(&mut out);
    out
}

/// Whether the first output returns to the CPU and stays on the GPU.
#[derive(Clone, Copy, PartialEq)]
enum OutputRetention {
    Discard,
    Keep,
    Resident,
}

/// One binding of a [`Gpu::compute_keeping`] pass: a buffer the pass reads,
/// or an output of so many samples it writes and is read back.
enum Bound<'a> {
    Input(&'a wgpu::Buffer),
    Output(usize),
    /// Cleared on the GPU before dispatch, for an atomic counter.
    ZeroedOutput(usize),
}

/// Which samples a resident buffer holds: where they are, how many, and a
/// fingerprint of 1024 of them, so samples changed in place since, or a new
/// vector that came to lie where an old one was freed, are not mistaken for
/// them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResidentKey {
    address: usize,
    samples: usize,
    fingerprint: u64,
}

impl ResidentKey {
    fn of(data: &[f32]) -> Self {
        let samples = data.len();
        let taken = samples.min(1024);
        let mut fingerprint = 0xcbf2_9ce4_8422_2325_u64;
        for at in 0..taken {
            let sample = data[at * samples / taken];
            fingerprint =
                (fingerprint ^ u64::from(sample.to_bits())).wrapping_mul(0x0100_0000_01b3);
        }
        Self {
            address: data.as_ptr() as usize,
            samples,
            fingerprint,
        }
    }
}

/// A frame's samples this thread put on the GPU, or had the GPU make.
struct Resident {
    buffer: wgpu::Buffer,
    usage: wgpu::BufferUsages,
    key: ResidentKey,
}

thread_local! {
    /// The frame this thread keeps on the GPU, at most one.
    static RESIDENT: std::cell::RefCell<Option<Resident>> =
        const { std::cell::RefCell::new(None) };
}

/// Keep `buffer`, holding the samples `key` names, resident for this
/// thread, in place of the frame it kept before.
fn keep_resident(buffer: wgpu::Buffer, usage: wgpu::BufferUsages, key: ResidentKey) {
    let previous = RESIDENT.with(|resident| {
        resident
            .borrow_mut()
            .replace(Resident { buffer, usage, key })
    });
    if let (Some(previous), Some(gpu)) = (previous, GPU.get().and_then(Option::as_ref)) {
        gpu.give(previous.usage, previous.buffer);
    }
}

/// Hand the frame this thread keeps on the GPU back to the pool.
pub(crate) fn release_resident() {
    let previous = RESIDENT.with(|resident| resident.borrow_mut().take());
    if let (Some(previous), Some(gpu)) = (previous, GPU.get().and_then(Option::as_ref)) {
        gpu.give(previous.usage, previous.buffer);
    }
}

/// The samples this thread keeps on the GPU for `data`, read back, or
/// `None` when it keeps none for them, or keeps them where they cannot be
/// read back (an uploaded frame, which is `data` copied).
#[cfg(test)]
pub(crate) fn resident_samples(data: &[f32]) -> Option<Vec<f32>> {
    let gpu = opened()?;
    let key = ResidentKey::of(data);
    let (buffer, usage) = RESIDENT.with(|resident| {
        resident
            .borrow()
            .as_ref()
            .filter(|resident| resident.key == key)
            .map(|resident| (resident.buffer.clone(), resident.usage))
    })?;
    let bytes = (data.len() * 4) as u64;
    let read_from = if usage.contains(wgpu::BufferUsages::MAP_READ) {
        gpu.map(&[buffer.slice(..bytes)], wgpu::MapMode::Read, None)
            .ok()?;
        buffer
    } else if usage.contains(wgpu::BufferUsages::COPY_SRC) {
        let readback = gpu.buffer(
            "resident-readback",
            bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&buffer, 0, &readback, 0, bytes);
        let submission = gpu.queue.submit([encoder.finish()]);
        gpu.map(&[readback.slice(..)], wgpu::MapMode::Read, Some(submission))
            .ok()?;
        readback
    } else {
        return None;
    };
    let samples = copy_out(&read_from.slice(..bytes).get_mapped_range().ok()?, false);
    read_from.unmap();
    Some(samples)
}

/// The frame this thread keeps on the GPU for `data`, taken out of its
/// keeping, or `None` when it keeps none for these samples. A frame's
/// resampled samples are taken so, to be normalized there and integrated.
pub(crate) fn take_resident(
    data: &[f32],
    (width, height, channels): (usize, usize, usize),
) -> Option<GpuFrame> {
    let key = ResidentKey::of(data);
    let resident = RESIDENT.with(|resident| {
        let mut resident = resident.borrow_mut();
        resident
            .as_ref()
            .is_some_and(|resident| resident.key == key)
            .then(|| resident.take())
            .flatten()
    })?;
    Some(GpuFrame {
        buffer: Some(resident.buffer),
        usage: resident.usage,
        width,
        height,
        channels,
        normalization_timer: None,
    })
}

/// Whether some live stack's moments are on the GPU, which is when a
/// prepared frame's samples there are worth keeping for it.
pub(crate) fn accumulating() -> bool {
    ACCUMULATORS.load(Ordering::Relaxed) > 0
}

/// Live stacks whose moments are on the GPU.
static ACCUMULATORS: AtomicUsize = AtomicUsize::new(0);

/// A prepared frame's samples on the GPU, `width` by `height` with
/// `channels` interleaved, which a live stack's accumulator there takes in
/// place of uploading them. Back to the pool when dropped.
pub(crate) struct GpuFrame {
    buffer: Option<wgpu::Buffer>,
    usage: wgpu::BufferUsages,
    width: usize,
    height: usize,
    channels: usize,
    normalization_timer: Option<GpuTimer>,
}

impl Drop for GpuFrame {
    fn drop(&mut self) {
        if let (Some(buffer), Some(gpu)) = (self.buffer.take(), GPU.get().and_then(Option::as_ref))
        {
            gpu.give(self.usage, buffer);
        }
    }
}

/// [`NormalizationMap::apply_region`]'s coefficients for a region, as
/// [`crate::normalization::NormalizationMap::gpu_coefficients`] gives them.
pub(crate) struct NormalizationCoefficients {
    pub(crate) global: bool,
    pub(crate) tile_columns: u32,
    /// Each column's lower and upper tile and fraction's bits; empty for a
    /// global map, and so `rows`.
    pub(crate) columns: Vec<[u32; 3]>,
    pub(crate) rows: Vec<[u32; 3]>,
    pub(crate) gains: Vec<f32>,
    pub(crate) offsets: Vec<f32>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NormalizeParams {
    width: u32,
    height: u32,
    channels: u32,
    global: u32,
    tile_columns: u32,
    row: u32,
    padding: [u32; 2],
}

impl GpuFrame {
    /// Apply a normalization where the frame lies, as the CPU applies it.
    /// Not waited for: the accumulator's work is submitted after it.
    pub(crate) fn normalize(
        &mut self,
        coefficients: &NormalizationCoefficients,
        trace: bool,
    ) -> Result<(), String> {
        let gpu = opened().ok_or("no GPU")?;
        let buffer = self.buffer.as_ref().ok_or("the frame has left the GPU")?;
        let samples = (self.width * self.height * self.channels) as u32;
        let groups = samples.div_ceil(256);
        let columns = groups.min(4096);
        let params = NormalizeParams {
            width: self.width as u32,
            height: self.height as u32,
            channels: self.channels as u32,
            global: u32::from(coefficients.global),
            tile_columns: coefficients.tile_columns,
            row: columns * 256,
            padding: [0; 2],
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
        // A binding holds a word at least.
        let words = |values: &[[u32; 3]]| -> Vec<u32> {
            let mut words: Vec<u32> = values.iter().flatten().copied().collect();
            if words.is_empty() {
                words.push(0);
            }
            words
        };
        let uniform = init(
            "normalize-params",
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
        );
        let column_weights = init(
            "normalize-columns",
            bytemuck::cast_slice(&words(&coefficients.columns)),
            storage,
        );
        let row_weights = init(
            "normalize-rows",
            bytemuck::cast_slice(&words(&coefficients.rows)),
            storage,
        );
        let gains = init(
            "normalize-gains",
            bytemuck::cast_slice(&coefficients.gains),
            storage,
        );
        let offsets = init(
            "normalize-offsets",
            bytemuck::cast_slice(&coefficients.offsets),
            storage,
        );
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("normalize"),
            layout: &gpu.normalize_layout,
            entries: &[
                (0, &uniform),
                (1, &column_weights),
                (2, &row_weights),
                (3, &gains),
                (4, &offsets),
                (5, buffer),
            ]
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }),
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let timer = GpuTimer::new(gpu, trace);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("normalize"),
                timestamp_writes: timer.as_ref().map(GpuTimer::writes),
            });
            pass.set_pipeline(&gpu.normalize_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(columns, groups.div_ceil(columns), 1);
        }
        if let Some(timer) = &timer {
            timer.resolve(&mut encoder);
        }
        gpu.queue.submit([encoder.finish()]);
        self.normalization_timer = timer;
        if FAILED.load(Ordering::Relaxed) {
            return Err("the GPU failed".into());
        }
        Ok(())
    }

    /// The frame's samples read back, for a frame that must be integrated
    /// on the CPU after all; `None` when the GPU cannot give them.
    pub(crate) fn download(&self) -> Option<Vec<f32>> {
        let gpu = opened()?;
        let buffer = self.buffer.as_ref()?;
        let bytes = (self.width * self.height * self.channels * 4) as u64;
        let read_from = if self.usage.contains(wgpu::BufferUsages::MAP_READ) {
            gpu.wait().ok()?;
            gpu.map(&[buffer.slice(..bytes)], wgpu::MapMode::Read, None)
                .ok()?;
            buffer.clone()
        } else {
            let readback = gpu.buffer(
                "frame-readback",
                bytes,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, bytes);
            let submission = gpu.queue.submit([encoder.finish()]);
            gpu.map(&[readback.slice(..)], wgpu::MapMode::Read, Some(submission))
                .ok()?;
            readback
        };
        let samples = copy_out(&read_from.slice(..bytes).get_mapped_range().ok()?, false);
        read_from.unmap();
        Some(samples)
    }
}

/// Releases this thread's resident frame when dropped, at the end of the
/// work that reads it.
pub(crate) struct ResidentScope;

impl Drop for ResidentScope {
    fn drop(&mut self) {
        release_resident();
    }
}

/// A debayered frame's samples on the GPU, all but the border the CPU
/// fills; [`Self::finish`] puts that in and keeps the frame resident.
pub(crate) struct VngOutput {
    buffer: wgpu::Buffer,
    usage: wgpu::BufferUsages,
    width: usize,
    height: usize,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScatterParams {
    entries: u32,
    row: u32,
    padding: [u32; 2],
}

impl VngOutput {
    /// Write `rgb`'s two-pixel border, which the CPU filled, into the GPU's
    /// copy, and keep it resident as `rgb`'s samples.
    pub(crate) fn finish(self, rgb: &[f32]) {
        let Some(gpu) = opened() else {
            return;
        };
        let (width, height) = (self.width, self.height);
        if rgb.len() != width * height * 3 {
            gpu.give(self.usage, self.buffer);
            return;
        }
        let mut patch: Vec<[u32; 2]> = Vec::with_capacity((4 * width + 4 * height) * 3);
        let mut push = |pixel: usize| {
            let first = pixel * 3;
            for (offset, value) in rgb[first..first + 3].iter().enumerate() {
                patch.push([(first + offset) as u32, value.to_bits()]);
            }
        };
        for y in 0..height {
            if y < 2 || y + 2 >= height {
                (0..width).for_each(|x| push(y * width + x));
            } else {
                for x in [0, 1, width - 2, width - 1] {
                    push(y * width + x);
                }
            }
        }
        let entries = patch.len() as u32;
        let groups = entries.div_ceil(256);
        let columns = groups.min(4096);
        let params = ScatterParams {
            entries,
            row: columns * 256,
            padding: [0; 2],
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
            "scatter-params",
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
        );
        let patch = init(
            "scatter-patch",
            bytemuck::cast_slice(&patch),
            wgpu::BufferUsages::STORAGE,
        );
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scatter"),
            layout: &gpu.scatter_layout,
            entries: &[(0, &uniform), (1, &patch), (2, &self.buffer)].map(|(binding, buffer)| {
                wgpu::BindGroupEntry {
                    binding,
                    resource: buffer.as_entire_binding(),
                }
            }),
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&gpu.scatter_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(columns, groups.div_ceil(columns), 1);
        }
        // Not waited for: what reads the frame next is submitted after it.
        gpu.queue.submit([encoder.finish()]);
        keep_resident(self.buffer, self.usage, ResidentKey::of(rgb));
    }
}

/// Bytes the pool keeps at most, past which a returned buffer is dropped,
/// and no more than a sixteenth of the device's memory: 512 MiB on an 8 GB
/// card.
const POOL_BYTES: u64 = 1 << 30;

/// Bytes of memory planned for a device whose backend does not say how much
/// it has: the smallest discrete GPUs these passes run well on.
const UNREPORTED_MEMORY: u64 = 4 << 30;

static GPU: OnceLock<Option<Gpu>> = OnceLock::new();

/// Each stage's time in one GPU call, printed to stderr when the call's
/// [`GpuOptions::trace`] asks.
pub(crate) struct Trace {
    enabled: bool,
    last: std::time::Instant,
    parts: Vec<(&'static str, f64)>,
    device_ms: Option<f64>,
}

impl Trace {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            last: std::time::Instant::now(),
            parts: Vec::new(),
            device_ms: None,
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
                .map(|(stage, ms)| format!("{stage} {ms:.3}"))
                .collect::<Vec<_>>()
                .join(", ");
            let held = opened()
                .and_then(|gpu| memory::in_use(&gpu.adapter, &gpu.device))
                .map(|bytes| format!(", device memory {} MiB", bytes >> 20))
                .unwrap_or_default();
            eprintln!("seiza-stacking gpu: {what}: {total:.1} ms ({parts}){held}");
            if let Some(ms) = self.device_ms {
                eprintln!("seiza-stacking gpu: {what}: device compute {ms:.3} ms");
            }
        }
    }
}
static FAILED: AtomicBool = AtomicBool::new(false);

/// The GPU, opened the first time, for a call `options` let use it.
fn gpu(options: &GpuOptions) -> Option<&'static Gpu> {
    if !options.enabled || FAILED.load(Ordering::Relaxed) {
        return None;
    }
    GPU.get_or_init(|| {
        let mut trace = Trace::new(options.trace);
        let opened = Gpu::open();
        trace.mark("open");
        trace.print("device");
        match opened {
            Ok(gpu) => {
                if options.trace {
                    let info = gpu.device.adapter_info();
                    eprintln!(
                        "seiza-stacking gpu: adapter: {} ({:?}), device timestamps {}, memory {}",
                        info.name,
                        info.backend,
                        gpu.device
                            .features()
                            .contains(wgpu::Features::TIMESTAMP_QUERY),
                        gpu.memory
                            .map_or("unknown".into(), |bytes| format!("{} MiB", bytes >> 20)),
                    );
                }
                Some(gpu)
            }
            Err(reason) => {
                log_once(&format!("resampling on the CPU: {reason}"));
                None
            }
        }
    })
    .as_ref()
}

/// The GPU, when a call opened it, for work on what is there already.
fn opened() -> Option<&'static Gpu> {
    if FAILED.load(Ordering::Relaxed) {
        return None;
    }
    GPU.get().and_then(Option::as_ref)
}

pub(crate) fn log_once(message: &str) {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!("seiza-stacking: {message}");
    }
}

/// The GPU adapter resampling uses, as "name (backend)", or `None` when
/// resampling stays on the CPU. Opens the device if it is not open yet.
#[cfg(test)]
pub(crate) fn adapter_description() -> Option<String> {
    gpu(&GpuOptions::default()).map(|gpu| {
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
        let memory = memory::capacity(&adapter);
        let supported = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: supported.max_storage_buffer_binding_size,
            max_buffer_size: supported.max_buffer_size,
            ..wgpu::Limits::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("seiza-stacking"),
            required_features: (if mapped {
                wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
            } else {
                wgpu::Features::empty()
            }) | (adapter.features() & wgpu::Features::TIMESTAMP_QUERY),
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
        let masked_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resample-masked"),
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
                storage(5, true),
                storage(6, false),
            ],
        });
        let masked_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("resample-masked"),
                bind_group_layouts: &[Some(&masked_layout)],
                immediate_size: 0,
            });
        let masked_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("resample-masked"),
            layout: Some(&masked_pipeline_layout),
            module: &module,
            entry_point: Some("with_mask"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sparse_masked_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("resample-sparse-mask"),
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
                    storage(5, true),
                    storage(6, false),
                    storage(7, false),
                ],
            });
        let sparse_masked_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("resample-sparse-mask"),
                bind_group_layouts: &[Some(&sparse_masked_layout)],
                immediate_size: 0,
            });
        let sparse_masked_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("resample-sparse-mask"),
                layout: Some(&sparse_masked_pipeline_layout),
                module: &module,
                entry_point: Some("with_sparse_mask"),
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
        let scatter_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scatter"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu/scatter.wgsl").into()),
        });
        let scatter_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scatter"),
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
        let scatter_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("scatter"),
                bind_group_layouts: &[Some(&scatter_layout)],
                immediate_size: 0,
            });
        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scatter"),
            layout: Some(&scatter_pipeline_layout),
            module: &scatter_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let normalize_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("normalize"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu/normalize.wgsl").into()),
        });
        let normalize_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("normalize"),
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
            ],
        });
        let normalize_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("normalize"),
                bind_group_layouts: &[Some(&normalize_layout)],
                immediate_size: 0,
            });
        let normalize_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("normalize"),
            layout: Some(&normalize_pipeline_layout),
            module: &normalize_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let detect_pipelines = [
            "luminance",
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
            adapter,
            device,
            queue,
            pipeline,
            layout,
            masked_pipeline,
            masked_layout,
            sparse_masked_pipeline,
            sparse_masked_layout,
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
            scatter_pipeline,
            scatter_layout,
            normalize_pipeline,
            normalize_layout,
            gather: OnceLock::new(),
            weights,
            mapped,
            max_buffer: limits
                .max_buffer_size
                .min(limits.max_storage_buffer_binding_size),
            memory,
            pool: std::sync::Mutex::new(Vec::new()),
            calls: (std::sync::Mutex::new(0), std::sync::Condvar::new()),
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

    /// Bytes of idle buffers the pool keeps at most.
    fn pool_bytes(&self) -> u64 {
        POOL_BYTES.min(self.memory.unwrap_or(UNREPORTED_MEMORY) / 16)
    }

    /// Bytes of device memory a call's `options` plan its work in: what the
    /// device has, or less where they cap it.
    fn memory_for(&self, options: &GpuOptions) -> u64 {
        match (self.memory, options.memory_bytes) {
            (Some(device), Some(cap)) => device.min(cap),
            (device, cap) => device.or(cap).unwrap_or(UNREPORTED_MEMORY),
        }
    }

    /// Return a buffer [`Self::take`] gave, unmapped, for a later call.
    fn give(&self, usage: wgpu::BufferUsages, buffer: wgpu::Buffer) {
        let mut pool = self
            .pool
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let held: u64 = pool.iter().map(|(_, buffer)| buffer.size()).sum();
        if held + buffer.size() <= self.pool_bytes() {
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
            // Reuse host staging and submit this frame's copy immediately,
            // rather than allocating an upload on every queue write and
            // leaving it to the next compute submission to flush.
            let usage = wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC;
            let upload = self.take("frame-upload", bytes, usage);
            self.map(&[upload.slice(..bytes)], wgpu::MapMode::Write, None)?;
            let copied = upload
                .slice(..bytes)
                .get_mapped_range_mut()
                .map(|mut view| view.slice(..).copy_from_slice(bytemuck::cast_slice(data)));
            upload.unmap();
            copied.map_err(|error| error.to_string())?;
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&upload, 0, &buffer, 0, bytes);
            self.queue.submit([encoder.finish()]);
            self.give(usage, upload);
        }
        Ok(buffer)
    }

    /// Plan uploads before taking a GPU-call permit: Rayon work may run
    /// another frame's task while it waits for its own parts.
    fn clip_patch(&self, mask: &LinearImage) -> Option<Vec<[u32; 2]>> {
        if self.mapped {
            return None;
        }
        let capacity = mask.data.len().div_ceil(128).clamp(1024, 1 << 18);
        let patch: Vec<[u32; 2]> = mask
            .data
            .par_iter()
            .with_min_len(64 * 1024)
            .enumerate()
            .filter_map(|(index, value)| {
                (value.to_bits() != 0).then_some([index as u32, value.to_bits()])
            })
            .take_any(capacity + 1)
            .collect();
        (patch.len() <= capacity).then_some(patch)
    }

    /// Expand sparse raw clipping on the GPU. Fractional values, negative
    /// zeros and non-finite samples keep their exact bits.
    fn upload_clip_mask(
        &self,
        mask: ClipView<'_>,
        patch: Option<&[[u32; 2]]>,
        trace: &Trace,
    ) -> Result<(wgpu::Buffer, wgpu::BufferUsages), String> {
        let (width, height, _) = mask.shape();
        let pixels = width * height;
        let patch = match (patch, mask) {
            (Some(patch), _) => patch,
            (None, ClipView::Image(mask)) => {
                if trace.enabled {
                    eprintln!(
                        "seiza-stacking gpu: clip mask upload dense: {} bytes",
                        mask.data.len() * 4
                    );
                }
                return Ok((self.upload("mask", &mask.data)?, self.input_usage()));
            }
            (None, ClipView::Listed { .. }) => return Err("a listed mask needs its list".into()),
        };
        if trace.enabled {
            eprintln!(
                "seiza-stacking gpu: clip mask upload: {} bytes for {pixels} pixels",
                patch.len() * 8,
            );
        }
        // Cleared and scattered on the GPU, never mapped, whatever the GPU.
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let buffer = self.take("mask", (pixels * 4) as u64, usage);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&buffer, 0, None);
        if !patch.is_empty() {
            let groups = (patch.len() as u32).div_ceil(256);
            let columns = groups.min(4096);
            let params = ScatterParams {
                entries: patch.len() as u32,
                row: columns * 256,
                padding: [0; 2],
            };
            let uniform = wgpu::util::DeviceExt::create_buffer_init(
                &self.device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("mask-scatter-params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                },
            );
            let patch = wgpu::util::DeviceExt::create_buffer_init(
                &self.device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("mask-scatter-patch"),
                    contents: bytemuck::cast_slice(patch),
                    usage: wgpu::BufferUsages::STORAGE,
                },
            );
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mask-scatter"),
                layout: &self.scatter_layout,
                entries: &[(0, &uniform), (1, &patch), (2, &buffer)].map(|(binding, buffer)| {
                    wgpu::BindGroupEntry {
                        binding,
                        resource: buffer.as_entire_binding(),
                    }
                }),
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.scatter_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(columns, groups.div_ceil(columns), 1);
        }
        // Resampling is submitted after this clear and scatter, with no
        // CPU wait or readback between the two GPU stages.
        self.queue.submit([encoder.finish()]);
        Ok((buffer, usage))
    }

    /// A buffer holding `data`: the one this thread keeps resident when it
    /// holds these very samples, or else `data` uploaded, which this thread
    /// then keeps resident in its place. A frame's star detection and its
    /// resampling read the same samples, so the second reads them where the
    /// first left them.
    fn source(&self, label: &str, data: &[f32]) -> Result<wgpu::Buffer, String> {
        let key = ResidentKey::of(data);
        let held = RESIDENT.with(|resident| {
            resident
                .borrow()
                .as_ref()
                .filter(|resident| resident.key == key)
                .map(|resident| resident.buffer.clone())
        });
        if let Some(buffer) = held {
            return Ok(buffer);
        }
        let buffer = self.upload(label, data)?;
        keep_resident(buffer.clone(), self.input_usage(), key);
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
        let bindings: Vec<Bound<'_>> = inputs
            .iter()
            .map(|&buffer| Bound::Input(buffer))
            .chain([Bound::Output(output_samples)])
            .collect();
        self.compute_keeping(layout, passes, &bindings, OutputRetention::Discard, trace)
            .map(|(mut outputs, _)| outputs.swap_remove(0))
    }

    /// [`Self::compute`] with `bindings` in binding order, each an input
    /// buffer or an output of so many samples. `keep` selects the first
    /// output's ownership: a CPU vector, or just its resident GPU buffer.
    /// Other outputs return as CPU vectors.
    #[allow(clippy::type_complexity)]
    fn compute_keeping<T: bytemuck::Pod + Send + Sync>(
        &self,
        layout: &wgpu::BindGroupLayout,
        passes: &[(&wgpu::ComputePipeline, (u32, u32))],
        bindings: &[Bound<'_>],
        keep: OutputRetention,
        trace: &mut Trace,
    ) -> Result<(Vec<Vec<T>>, Option<(wgpu::Buffer, wgpu::BufferUsages)>), String> {
        let outputs: Vec<usize> = bindings
            .iter()
            .filter_map(|binding| match binding {
                Bound::Output(samples) | Bound::ZeroedOutput(samples) => Some(*samples),
                Bound::Input(_) => None,
            })
            .collect();
        let bytes: Vec<u64> = outputs
            .iter()
            .map(|&samples| (samples * std::mem::size_of::<T>()) as u64)
            .collect();
        let output_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | if self.mapped {
                wgpu::BufferUsages::MAP_READ
            } else {
                wgpu::BufferUsages::COPY_SRC
            };
        let output_buffers: Vec<wgpu::Buffer> = bytes
            .iter()
            .map(|&bytes| self.take("output", bytes, output_usage))
            .collect();
        let skip = usize::from(keep == OutputRetention::Resident);
        let readback_usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let readbacks: Option<Vec<wgpu::Buffer>> = (!self.mapped).then(|| {
            bytes
                .iter()
                .skip(skip)
                .map(|&bytes| self.take("readback", bytes, readback_usage))
                .collect()
        });
        let mut next_output = output_buffers.iter();
        let entries: Vec<wgpu::BindGroupEntry> = bindings
            .iter()
            .enumerate()
            .map(|(binding, bound)| {
                let buffer = match bound {
                    Bound::Input(buffer) => *buffer,
                    Bound::Output(_) | Bound::ZeroedOutput(_) => {
                        next_output.next().expect("an output for each")
                    }
                };
                wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                }
            })
            .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &entries,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for (binding, buffer) in bindings
            .iter()
            .filter(|binding| !matches!(binding, Bound::Input(_)))
            .zip(&output_buffers)
        {
            if matches!(binding, Bound::ZeroedOutput(_)) {
                encoder.clear_buffer(buffer, 0, None);
            }
        }
        let timer = GpuTimer::new(self, trace.enabled);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: timer.as_ref().map(GpuTimer::writes),
            });
            pass.set_bind_group(0, &bind_group, &[]);
            for &(pipeline, (columns, rows)) in passes {
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(columns, rows, 1);
            }
        }
        if let Some(timer) = &timer {
            timer.resolve(&mut encoder);
        }
        if let Some(readbacks) = &readbacks {
            for ((output, readback), &bytes) in output_buffers
                .iter()
                .skip(skip)
                .zip(readbacks)
                .zip(&bytes[skip..])
            {
                encoder.copy_buffer_to_buffer(output, 0, readback, 0, bytes);
            }
        }
        trace.mark("upload");
        let submission = self.queue.submit([encoder.finish()]);
        let read_from = readbacks.as_deref().unwrap_or(&output_buffers[skip..]);
        let read_bytes = &bytes[skip..];
        let mut slices: Vec<wgpu::BufferSlice<'_>> = read_from
            .iter()
            .zip(read_bytes)
            .map(|(buffer, &bytes)| buffer.slice(..bytes))
            .collect();
        if let Some(timer) = &timer {
            slices.push(timer.readback.slice(..));
        }
        let mapped = if slices.is_empty() {
            Ok(())
        } else {
            self.map(&slices, wgpu::MapMode::Read, Some(submission))
        };
        let result = mapped.and_then(|()| {
            trace.mark("queue and readback");
            if let Some(timer) = &timer {
                timer.record(trace)?;
            }
            if FAILED.load(Ordering::Relaxed) {
                return Err("the GPU failed".to_string());
            }
            let mut data = Vec::with_capacity(outputs.len());
            if skip != 0 {
                data.push(Vec::new());
            }
            for (buffer, &bytes) in read_from.iter().zip(read_bytes) {
                let view = buffer
                    .slice(..bytes)
                    .get_mapped_range()
                    .map_err(|error| error.to_string())?;
                data.push(copy_out(&view, false));
                drop(view);
                buffer.unmap();
            }
            trace.mark("copy out");
            Ok(data)
        });
        if let Some(readbacks) = readbacks {
            for readback in readbacks {
                self.give(readback_usage, readback);
            }
        }
        let mut output_buffers = output_buffers.into_iter();
        let kept = match &result {
            Ok(_) if keep != OutputRetention::Discard => {
                output_buffers.next().map(|buffer| (buffer, output_usage))
            }
            _ => None,
        };
        for buffer in output_buffers {
            self.give(output_usage, buffer);
        }
        result.map(|data| (data, kept))
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

    /// Wait until fewer than `concurrency` frame calls run.
    fn admitted(&self, concurrency: usize) -> FrameCall<'_> {
        let (count, freed) = &self.calls;
        let mut running = count.lock().unwrap_or_else(|poison| poison.into_inner());
        while *running >= concurrency.max(1) {
            running = freed
                .wait(running)
                .unwrap_or_else(|poison| poison.into_inner());
        }
        *running += 1;
        FrameCall(self)
    }

    fn wait(&self) -> Result<(), String> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    #[allow(clippy::too_many_arguments)]
    fn resample(
        &self,
        source: &LinearImage,
        region: ReferenceRegion,
        grid: &[[u32; 4]],
        (grid_columns, grid_origin): (u32, u32),
        sampling: GpuSampling,
        mask: Option<ClipView<'_>>,
        trace: &mut Trace,
        compact_mask: bool,
        mask_patch: Option<&[[u32; 2]]>,
        retention: OutputRetention,
    ) -> Result<(LinearImage, Option<ClipMask>), String> {
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
        let grid_buffer = init(
            "grid",
            bytemuck::cast_slice(grid),
            wgpu::BufferUsages::STORAGE,
        );
        let source_buffer = self.source("source", &source.data)?;
        let workgroups = (
            (region.width as u32).div_ceil(16),
            (region.height as u32).div_ceil(16),
        );
        let mask_buffer = mask
            .map(|mask| self.upload_clip_mask(mask, mask_patch, trace))
            .transpose()?;
        let mut bindings = vec![
            Bound::Input(&uniform),
            Bound::Input(&source_buffer),
            Bound::Input(&grid_buffer),
            Bound::Input(&self.weights),
            Bound::Output(output_samples),
        ];
        // The mask goes through the same positions in the same pass.
        let capacity = (region.width * region.height)
            .div_ceil(128)
            .clamp(1024, 1 << 18)
            .min(region.width * region.height);
        let sparse = compact_mask && mask_buffer.is_some();
        let (layout, pipeline) = match &mask_buffer {
            Some((mask_buffer, _)) => {
                bindings.push(Bound::Input(mask_buffer));
                if sparse {
                    bindings.push(Bound::Output(2 * capacity));
                    bindings.push(Bound::ZeroedOutput(1));
                    (&self.sparse_masked_layout, &self.sparse_masked_pipeline)
                } else {
                    bindings.push(Bound::Output(region.width * region.height));
                    (&self.masked_layout, &self.masked_pipeline)
                }
            }
            None => (&self.layout, &self.pipeline),
        };
        let data = self.compute_keeping::<f32>(
            layout,
            &[(pipeline, workgroups)],
            &bindings,
            retention,
            trace,
        );
        drop(bindings);
        drop(source_buffer);
        if let Some((mask_buffer, usage)) = mask_buffer {
            self.give(usage, mask_buffer);
        }
        let (mut outputs, output) = data?;
        let mask = if sparse {
            let count = outputs.pop().unwrap()[0].to_bits() as usize;
            if count > capacity {
                if trace.enabled {
                    eprintln!(
                        "seiza-stacking gpu: clip mask dense fallback: {count} contributions exceed {capacity}"
                    );
                }
                // A heavily clipped image may not be sparse. Repeat with
                // a dense mask; no contribution may be silently dropped.
                if let Some((buffer, usage)) = output {
                    self.give(usage, buffer);
                }
                drop(outputs);
                return self.resample(
                    source,
                    region,
                    grid,
                    (grid_columns, grid_origin),
                    sampling,
                    mask,
                    trace,
                    false,
                    mask_patch,
                    retention,
                );
            }
            let words = outputs.pop().unwrap();
            if trace.enabled {
                eprintln!(
                    "seiza-stacking gpu: clip mask readback: {count} contributions, {} bytes for {} pixels",
                    8 * capacity + 4,
                    region.width * region.height
                );
            }
            Some(ClipMask::Sparse {
                width: region.width,
                height: region.height,
                entries: words
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .take(count)
                    .map(|entry| (entry[0].to_bits() as usize, entry[1]))
                    .collect(),
            })
        } else if outputs.len() > 1 {
            Some(ClipMask::Dense(
                LinearImage::new(region.width, region.height, 1, outputs.pop().unwrap())
                    .map_err(|error| error.to_string())?,
            ))
        } else {
            None
        };
        let data = outputs.swap_remove(0);
        // The resampled frame stays on the GPU, for its normalization and
        // the accumulator to take; see `take_resident`.
        if let Some((buffer, usage)) = output {
            keep_resident(buffer, usage, ResidentKey::of(&data));
        }
        let image = if retention != OutputRetention::Keep {
            LinearImage {
                width: region.width,
                height: region.height,
                channels,
                data,
            }
        } else {
            LinearImage::new(region.width, region.height, channels, data)
                .map_err(|error| error.to_string())?
        };
        Ok((image, mask))
    }
}

/// Resample `region` of the reference grid from `source` through `inverse`
/// (reference position to source position) on the GPU, or `None` when the
/// CPU should: the frame is small, the channels are neither one nor three,
/// or no GPU is in use.
#[cfg(test)]
pub(crate) fn resample(
    source: &LinearImage,
    region: ReferenceRegion,
    inverse: &(impl Fn(f64, f64) -> (f64, f64) + Sync),
    sampling: GpuSampling,
    options: &GpuOptions,
) -> Option<LinearImage> {
    resample_masked(source, None, region, inverse, sampling, options).map(|(image, _)| image)
}

/// [`resample`], and `mask`, the frame's clip mask, one sample a source
/// pixel, resampled bilinearly in the same pass, at the same positions, as
/// [`resample`] alone would resample it. The mask comes back only when it
/// was taken: it matches the frame's shape. `None` when the CPU should: the
/// frame is small, `options` keep resampling there, or no GPU is in use.
pub(crate) fn resample_masked(
    source: &LinearImage,
    mask: Option<ClipView<'_>>,
    region: ReferenceRegion,
    inverse: &(impl Fn(f64, f64) -> (f64, f64) + Sync),
    sampling: GpuSampling,
    options: &GpuOptions,
) -> Option<(LinearImage, Option<LinearImage>)> {
    let (image, mask) = resample_masked_inner(
        source,
        mask,
        region,
        inverse,
        sampling,
        options,
        false,
        OutputRetention::Keep,
    )?;
    let mask = match mask {
        Some(ClipMask::Dense(mask)) => Some(mask),
        None => None,
        Some(ClipMask::Sparse { .. }) => return None,
    };
    Some((image, mask))
}

/// Resample for clipping statistics: a bounded sparse list when possible,
/// with a dense fallback when the mask has too many nonzero contributions.
pub(crate) fn resample_clipped(
    source: &LinearImage,
    mask: Option<ClipView<'_>>,
    region: ReferenceRegion,
    inverse: &(impl Fn(f64, f64) -> (f64, f64) + Sync),
    sampling: GpuSampling,
    options: &GpuOptions,
) -> Option<(LinearImage, Option<ClipMask>)> {
    resample_masked_inner(
        source,
        mask,
        region,
        inverse,
        sampling,
        options,
        true,
        OutputRetention::Keep,
    )
}

/// Resample into an owned GPU frame without a full image readback.
/// The shape-only image and GPU frame stay together until admission.
pub(crate) fn resample_retained(
    source: &LinearImage,
    mask: Option<ClipView<'_>>,
    region: ReferenceRegion,
    inverse: &(impl Fn(f64, f64) -> (f64, f64) + Sync),
    sampling: GpuSampling,
    options: &GpuOptions,
) -> Option<(LinearImage, Option<ClipMask>, GpuFrame)> {
    let (image, mask) = resample_masked_inner(
        source,
        mask,
        region,
        inverse,
        sampling,
        options,
        true,
        OutputRetention::Resident,
    )?;
    let frame = take_resident(&image.data, (image.width, image.height, image.channels))?;
    Some((image, mask, frame))
}

#[allow(clippy::too_many_arguments)]
fn resample_masked_inner(
    source: &LinearImage,
    mask: Option<ClipView<'_>>,
    region: ReferenceRegion,
    inverse: &(impl Fn(f64, f64) -> (f64, f64) + Sync),
    sampling: GpuSampling,
    options: &GpuOptions,
    compact_mask: bool,
    retention: OutputRetention,
) -> Option<(LinearImage, Option<ClipMask>)> {
    let channels = source.channels;
    let mask = mask.filter(|mask| mask.shape() == (source.width, source.height, 1));
    // By the source, not the region: every region of a frame resamples the
    // same way, so a footprint holds what the whole grid would.
    if !matches!(channels, 1 | 3)
        || source.data.len() < MINIMUM_SAMPLES
        || source.width < 6
        || source.height < 6
        || region.width == 0
        || region.height == 0
        || !options.resampling
    {
        return None;
    }
    let gpu = gpu(options)?;
    let mut trace = Trace::new(options.trace);
    // A listed mask is scattered from its list on every GPU; an image is
    // listed here, for a compact readback on a discrete GPU.
    let patch = match mask {
        Some(ClipView::Listed { pixels, .. }) => Some(
            pixels
                .iter()
                .map(|&pixel| [pixel, 1.0_f32.to_bits()])
                .collect::<Vec<_>>(),
        ),
        Some(ClipView::Image(mask)) if compact_mask => gpu.clip_patch(mask),
        _ => None,
    };
    trace.mark("mask plan");
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
    let _call = gpu.admitted(options.concurrency);
    trace.mark("admission wait");
    let result = gpu.resample(
        source,
        region,
        &grid,
        (columns as u32, origin),
        sampling,
        mask,
        &mut trace,
        compact_mask,
        patch.as_deref(),
        retention,
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
/// mosaic is small, `options` keep debayering there, or no GPU is in use.
pub(crate) fn vng(
    mosaic: &[f32],
    (width, height): (usize, usize),
    channels: [u32; 4],
    balance: [f32; 3],
    options: &GpuOptions,
) -> Option<(Vec<f32>, Option<VngOutput>)> {
    if mosaic.len() < MINIMUM_SAMPLES || width < 5 || height < 5 || !options.debayer {
        return None;
    }
    let gpu = gpu(options)?;
    let _call = gpu.admitted(options.concurrency);
    if (mosaic.len() as u64) * 12 > gpu.max_buffer {
        return None;
    }
    let mut trace = Trace::new(options.trace);
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
        let rgb = gpu.compute_keeping::<f32>(
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
            &[
                Bound::Input(&uniform),
                Bound::Input(&source),
                Bound::Input(&balanced),
                Bound::Output(mosaic.len() * 3),
            ],
            OutputRetention::Keep,
            &mut trace,
        );
        gpu.give(gpu.input_usage(), source);
        rgb
    });
    gpu.give(wgpu::BufferUsages::STORAGE, balanced);
    trace.print(&format!("vng {width}x{height}"));
    match result {
        Ok((mut outputs, output)) => Some((
            outputs.swap_remove(0),
            output.map(|(buffer, usage)| VngOutput {
                buffer,
                usage,
                width,
                height,
            }),
        )),
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
/// `None` when the CPU should: the frame is small, `options` keep the
/// screen there, or no GPU is in use.
pub(crate) fn defect_candidates(
    data: &[f32],
    (width, height, channels, step): (usize, usize, usize, usize),
    thresholds: [f32; 4],
    options: &GpuOptions,
) -> Option<Vec<u32>> {
    if data.len() < MINIMUM_SAMPLES
        || data.len() != width * height * channels
        || channels * step * step > 4
        || data.len() > u32::MAX as usize
        || !options.defects
    {
        return None;
    }
    let gpu = gpu(options)?;
    let _call = gpu.admitted(options.concurrency);
    if (data.len() as u64) * 4 > gpu.max_buffer {
        return None;
    }
    let mut trace = Trace::new(options.trace);
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
/// pass than the GPU keeps room for, `options` keep detection there, or no
/// GPU is in use.
pub(crate) fn detect_stars(
    image: &LinearImage,
    config: &seiza::DetectConfig,
    options: &GpuOptions,
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
        || !options.detection
    {
        return None;
    }
    let gpu = gpu(options)?;
    let _call = gpu.admitted(options.concurrency);
    if (image.data.len() as u64) * 4 > gpu.max_buffer {
        return None;
    }
    let mut trace = Trace::new(options.trace);
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
        // A mono frame's luminance is the frame, read where it lies, and the
        // binding holds a word.
        let luma_bytes = if image.channels == 3 {
            u64::from(pixels) * 4
        } else {
            4
        };
        let luma = self.take("detect-luma", luma_bytes, scratch);
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
        let frame = self.source("detect-frame", &image.data);
        let timer = GpuTimer::new(self, trace.enabled);
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
            let dispatches = [per_pixel, (tiles_x, tiles_y), per_row, (1, 1), per_row];
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("detect"),
                    timestamp_writes: timer.as_ref().map(GpuTimer::writes),
                });
                pass.set_bind_group(0, &bind_group, &[]);
                for (pipeline, (x, y)) in self.detect_pipelines.iter().zip(dispatches) {
                    pass.set_pipeline(pipeline);
                    pass.dispatch_workgroups(x, y, 1);
                }
            }
            if let Some(timer) = &timer {
                timer.resolve(&mut encoder);
            }
            encoder.copy_buffer_to_buffer(&rows, rows_bytes - 4, &total_read, 0, 4);
            let submission = self.queue.submit([encoder.finish()]);
            drop(frame);
            let mut slices = vec![total_read.slice(..)];
            if let Some(timer) = &timer {
                slices.push(timer.readback.slice(..));
            }
            self.map(&slices, wgpu::MapMode::Read, Some(submission))
        });
        let found = result.and_then(|()| {
            trace.mark("queue and count");
            if let Some(timer) = &timer {
                timer.record(trace)?;
            }
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

/// One reusable set of GPU band buffers. Submission and collection are
/// separate so another set can be staged while this one's work runs.
pub(crate) struct BandSession {
    gpu: &'static Gpu,
    channels: usize,
    weight_channels: usize,
    weighted: bool,
    band_capacity: usize,
    frame_count: usize,
    values: wgpu::Buffer,
    /// A single reusable upload on a discrete GPU, packed frame by frame.
    upload: Option<wgpu::Buffer>,
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
    trace: bool,
    pending: Option<PendingBand>,
}

struct PendingBand {
    samples: usize,
    reaching: Vec<usize>,
    submission: Option<wgpu::SubmissionIndex>,
    trace: Trace,
    timer: Option<GpuTimer>,
}

pub(crate) enum BandStageError {
    Gpu(String),
    Source(crate::Error),
}

impl BandSession {
    /// Larger bands amortize file reads and submission overhead: 512 MiB
    /// bands took the 200-frame 61 MP Pleiades stack's final integration on
    /// an Iris Xe from 25 s to 17.5 s, against 128 MiB, and left the M4 Max's
    /// 2.3 s as it was. Stay within half the caller's band memory, both
    /// device binding limits, a 512 MiB input band, and a sixteenth of the
    /// device memory planned for: the two sessions' bands and, on a
    /// discrete GPU, their uploads take four times that.
    pub(crate) fn rows(
        (width, height, channels): (usize, usize, usize),
        frame_count: usize,
        memory: usize,
        options: &GpuOptions,
    ) -> Option<usize> {
        if frame_count == 0 || frame_count > MAX_BAND_FRAMES || !options.integration {
            return None;
        }
        let gpu = gpu(options)?;
        let max_buffer = usize::try_from(gpu.max_buffer).unwrap_or(usize::MAX);
        let max_samples = (max_buffer / (frame_count * 4)).min(max_buffer / 16);
        let row_samples = width * channels;
        if row_samples > max_samples {
            return None;
        }
        let planned = usize::try_from(gpu.memory_for(options) / 16).unwrap_or(usize::MAX);
        let samples =
            ((memory / 2).min(512 * 1024 * 1024).min(planned) / (frame_count * 4)).min(max_samples);
        Some((samples / row_samples).clamp(1, height))
    }

    /// A session for `frame_count` frames of `channels` channels, bands of
    /// up to `band_samples` samples, weighted per `weight_channels` channels
    /// when `weights` is given (one row per frame), or `None` when the CPU
    /// should integrate: a small band, too many frames, `options` keep
    /// integration there, or no GPU.
    pub(crate) fn new(
        frame_count: usize,
        channels: usize,
        band_samples: usize,
        weights: Option<(&[f32], usize)>,
        tables: BandTables,
        options: &GpuOptions,
    ) -> Option<Self> {
        if frame_count == 0
            || frame_count > MAX_BAND_FRAMES
            || band_samples * frame_count < MINIMUM_SAMPLES
            || !options.integration
        {
            return None;
        }
        let gpu = gpu(options)?;
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
        let upload = (!gpu.mapped).then(|| {
            gpu.buffer(
                "band-upload",
                values_bytes,
                wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            )
        });
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
            upload,
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
            trace: options.trace,
            pending: None,
        })
    }

    /// Read each reaching frame directly into a disjoint part of mapped
    /// upload memory. Sources normalize bounded decoded runs before writing
    /// them; no full band or serialized packing copy sits between them.
    pub(crate) fn submit_from(
        &mut self,
        source: &impl crate::batch::BandSource,
        top: usize,
        samples: usize,
        reaching: Vec<usize>,
    ) -> Result<(), BandStageError> {
        self.submit_with(samples, reaching.clone(), |range| {
            let bytes = samples * 4;
            // Pairs keep view offsets aligned to eight bytes even when a
            // frame has an odd number of f32 samples. Each view is created
            // on its writing thread: wgpu's write-only slices are not Send.
            reaching
                .par_chunks(2)
                .enumerate()
                .try_for_each(|(pair, frames)| {
                    let start = pair * 2 * bytes;
                    let mut view = range
                        .slice(start as u64..(start + frames.len() * bytes) as u64)
                        .get_mapped_range_mut()
                        .map_err(|error| BandStageError::Gpu(error.to_string()))?;
                    for (slot, &index) in frames.iter().enumerate() {
                        source
                            .read_into(index, top, samples, &mut |start, values| {
                                let at = slot * bytes + start * 4;
                                view.slice(at..at + values.len() * 4)
                                    .copy_from_slice(bytemuck::cast_slice(values));
                            })
                            .map_err(BandStageError::Source)?;
                    }
                    Ok(())
                })
        })
    }

    fn submit_with(
        &mut self,
        samples: usize,
        reaching: Vec<usize>,
        fill: impl FnOnce(wgpu::BufferSlice<'_>) -> Result<(), BandStageError>,
    ) -> Result<(), BandStageError> {
        if self.pending.is_some() {
            return Err(BandStageError::Gpu(
                "the previous band has not been collected".into(),
            ));
        }
        if samples == 0 || samples > self.band_capacity {
            return Err(BandStageError::Gpu(
                "the band does not fit the session".into(),
            ));
        }
        let gpu = self.gpu;
        let mut trace = Trace::new(self.trace);
        if reaching.is_empty() {
            self.pending = Some(PendingBand {
                samples,
                reaching,
                submission: None,
                trace,
                timer: None,
            });
            return Ok(());
        }
        let bytes = samples * 4;
        let total = (reaching.len() * bytes) as u64;
        let upload = self.upload.as_ref().unwrap_or(&self.values);
        let range = upload.slice(..total);
        gpu.map(&[range], wgpu::MapMode::Write, None)
            .map_err(BandStageError::Gpu)?;
        trace.mark("map upload");
        let copied = fill(range);
        trace.mark("read into staging");
        upload.unmap();
        trace.mark("unmap upload");
        copied?;
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
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        if self.upload.is_some() {
            encoder.copy_buffer_to_buffer(upload, 0, &self.values, 0, total);
        }
        encoder.clear_buffer(&self.counts, 0, None);
        let timer = GpuTimer::new(gpu, self.trace);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("band rejection"),
                timestamp_writes: timer.as_ref().map(GpuTimer::writes),
            });
            pass.set_pipeline(&gpu.bands_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let groups = samples.div_ceil(256) as u32;
            let (x, y) = (groups.min(32768), groups.div_ceil(32768));
            pass.dispatch_workgroups(x, y, 1);
        }
        if let Some(timer) = &timer {
            timer.resolve(&mut encoder);
        }
        let output_bytes = (samples * 16) as u64;
        let tally_bytes = (samples * 8) as u64;
        if let (Some(moments_read), Some(tallies_read)) = (&self.moments_read, &self.tallies_read) {
            encoder.copy_buffer_to_buffer(&self.moments, 0, moments_read, 0, output_bytes);
            encoder.copy_buffer_to_buffer(&self.tallies, 0, tallies_read, 0, tally_bytes);
        }
        let count_bytes = (reaching.len() * 8) as u64;
        encoder.copy_buffer_to_buffer(&self.counts, 0, &self.counts_read, 0, count_bytes);
        trace.mark("encode");
        let submission = gpu.queue.submit([encoder.finish()]);
        self.pending = Some(PendingBand {
            samples,
            reaching,
            submission: Some(submission),
            trace,
            timer,
        });
        Ok(())
    }

    /// Wait for this set's submitted band, write its outputs, and add its
    /// per-frame counts. The buffers are reusable after a successful return.
    pub(crate) fn collect(
        &mut self,
        outputs: BandOutputs<'_>,
        counts: &mut [(usize, usize)],
    ) -> Result<(), String> {
        let PendingBand {
            samples,
            reaching,
            submission,
            mut trace,
            timer,
        } = self.pending.take().ok_or("no band has been submitted")?;
        if outputs.mean.len() != samples || counts.len() != self.frame_count {
            return Err("the outputs do not fit the submitted band".into());
        }
        let Some(submission) = submission else {
            outputs.mean.fill(0.0);
            return Ok(());
        };
        let gpu = self.gpu;
        let output_bytes = (samples * 16) as u64;
        let tally_bytes = (samples * 8) as u64;
        let count_bytes = (reaching.len() * 8) as u64;
        // Host elapsed time while other bands were staged, not GPU time.
        trace.mark("in flight");
        let moments_from = self.moments_read.as_ref().unwrap_or(&self.moments);
        let tallies_from = self.tallies_read.as_ref().unwrap_or(&self.tallies);
        let mut slices = vec![
            moments_from.slice(..output_bytes),
            tallies_from.slice(..tally_bytes),
            self.counts_read.slice(..count_bytes),
        ];
        if let Some(timer) = &timer {
            slices.push(timer.readback.slice(..));
        }
        gpu.map(&slices, wgpu::MapMode::Read, Some(submission))?;
        trace.mark("wait");
        if let Some(timer) = &timer {
            timer.record(&mut trace)?;
        }
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
    trace: bool,
}

impl Drop for GpuMoments {
    fn drop(&mut self) {
        ACCUMULATORS.fetch_sub(1, Ordering::Relaxed);
    }
}

impl GpuMoments {
    /// Moments of at least a million samples moved to the GPU, or `None`
    /// when they stay on the CPU, as they do when `options` keep the
    /// accumulator there, or when it would crowd the device's memory; see
    /// [`Self::memory_needed`].
    pub(crate) fn upload(moments: MomentSlices<'_>, options: &GpuOptions) -> Option<Self> {
        if moments.mean.len() < MINIMUM_SAMPLES || !options.accumulator {
            return None;
        }
        let gpu = gpu(options)?;
        let samples = moments.mean.len();
        if (samples as u64) * 4 > gpu.max_buffer {
            return None;
        }
        let needed = Self::memory_needed(
            samples,
            moments.weight_sum.is_some(),
            options.concurrency,
            gpu.mapped,
        )
        .saturating_add(gpu.pool_bytes());
        let memory = gpu.memory_for(options);
        let fits = Self::fits(needed, memory);
        if options.trace {
            eprintln!(
                "seiza-stacking gpu: accumulator on the {}: it and its frames need {} MiB, \
                 against three quarters of {} MiB",
                if fits { "GPU" } else { "CPU" },
                needed >> 20,
                memory >> 20
            );
        }
        if !fits {
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
            trace: options.trace,
        };
        gpu.wait().ok()?;
        if FAILED.load(Ordering::Relaxed) {
            return None;
        }
        ACCUMULATORS.fetch_add(1, Ordering::Relaxed);
        Some(state)
    }

    /// Bytes a GPU accumulator of `samples` takes, with the frames that are
    /// on the GPU for it at once, besides the pool's idle buffers: the
    /// moments, five words a sample when `weighted` and four when not; each
    /// pipeline worker's queued frame and the one it builds, at the most
    /// workers a pipeline derives, and the frame integrating; and the
    /// `concurrency` frame calls' buffers, three frames each on a discrete
    /// GPU, with its staging copies, and one where the CPU maps them. A
    /// frame is counted at the grid's size, which an expanded canvas's
    /// frames fill only in part.
    ///
    /// The Pleiades stack, 61 million samples, held 5.2 GiB on an M4 Max,
    /// against the 5.8 GiB counted here with its pool. On an 8 GB RTX 2080
    /// SUPER, a mosaic of 128 million samples whose frames each covered 50
    /// million held 7.3 GB with the desktop, and every pass slowed as
    /// Windows paged the card's memory: 207 to 276 s for 129 frames, where
    /// an accumulator on the CPU took 120 (#362).
    fn memory_needed(samples: usize, weighted: bool, concurrency: usize, mapped: bool) -> u64 {
        let frame = (samples as u64).saturating_mul(4);
        let moments = frame.saturating_mul(if weighted { 5 } else { 4 });
        let frames = (2 * crate::pipeline::MAXIMUM_DERIVED_WORKERS as u64 + 1)
            .saturating_add((concurrency as u64).saturating_mul(if mapped { 1 } else { 3 }));
        moments.saturating_add(frame.saturating_mul(frames))
    }

    /// Whether `needed` bytes leave a quarter of `memory` free, for other
    /// programs, the frame calls' smaller buffers and the allocator's
    /// blocks.
    fn fits(needed: u64, memory: u64) -> bool {
        needed <= memory / 4 * 3
    }

    /// Classify `frame`, placed as `placement` says, against the moments,
    /// and integrate it if rejection would accept at least `admit_at` of its
    /// finite samples: how many it would accept, and, when it was
    /// integrated, how many were accepted and refused there. One submission,
    /// one wait: the GPU decides the admission itself. `staged` holds the
    /// frame's samples where they already lie on the GPU, which are not
    /// uploaded again.
    pub(crate) fn admit(
        &mut self,
        frame: &[f32],
        staged: Option<GpuFrame>,
        placement: FramePlacement,
        weights: &[f32],
        rejection: Option<GpuRejection>,
        admit_at: usize,
    ) -> Result<(usize, Option<(usize, usize)>), String> {
        let gpu = self.gpu;
        let mut trace = Trace::new(self.trace);
        let samples = placement.rows * placement.span;
        let mut normalization_timer = None;
        let held = staged.and_then(|mut staged| {
            (staged.width * staged.height * staged.channels == samples)
                .then(|| {
                    normalization_timer = staged.normalization_timer.take();
                    staged.buffer.take().map(|buffer| (buffer, staged.usage))
                })
                .flatten()
        });
        let (source, usage) = match held {
            Some(held) => held,
            None if frame.len() == samples => (gpu.upload("frame", frame)?, gpu.input_usage()),
            None => return Err("the frame's samples are neither here nor on the GPU".into()),
        };
        trace.mark("upload");
        // No frame holds `u32::MAX` samples, so a count past it, which
        // admits none, admits none here either.
        let admit_at = u32::try_from(admit_at).unwrap_or(u32::MAX);
        let counts = self.dispatch(
            &source,
            placement,
            weights,
            rejection,
            admit_at,
            &mut trace,
            normalization_timer.as_ref(),
        );
        gpu.give(usage, source);
        let [accepted, _, integrated, refused] = counts?;
        trace.mark("admit");
        trace.print(&format!("admit {samples} samples"));
        let admitted = accepted >= admit_at;
        Ok((
            accepted as usize,
            admitted.then_some((integrated as usize, refused as usize)),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        &self,
        frame: &wgpu::Buffer,
        placement: FramePlacement,
        weights: &[f32],
        rejection: Option<GpuRejection>,
        admit_at: u32,
        trace: &mut Trace,
        normalization_timer: Option<&GpuTimer>,
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
        let timer = GpuTimer::new(gpu, self.trace);
        {
            // Each dispatch sees the last one's writes: `integrate` reads the
            // count `classify` left.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("admission"),
                timestamp_writes: timer.as_ref().map(GpuTimer::writes),
            });
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
        if let Some(timer) = &timer {
            timer.resolve(&mut encoder);
        }
        encoder.copy_buffer_to_buffer(&self.counters, 0, &self.counters_read, 0, 16);
        let submission = gpu.queue.submit([encoder.finish()]);
        let mut slices = vec![self.counters_read.slice(..)];
        for timer in [timer.as_ref(), normalization_timer].into_iter().flatten() {
            slices.push(timer.readback.slice(..));
        }
        gpu.map(&slices, wgpu::MapMode::Read, Some(submission))?;
        if let Some(timer) = &timer {
            timer.record(trace)?;
        }
        if let Some(timer) = normalization_timer {
            let ms = timer.elapsed_ms()?;
            eprintln!("seiza-stacking gpu: normalize: device compute {ms:.3} ms");
        }
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
        let mut trace = Trace::new(self.trace);
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

    #[test]
    fn an_accumulator_that_would_crowd_the_card_stays_on_the_cpu() {
        // #362: a weighted mosaic of 127,780,092 samples, three frame calls
        // on an 8 GB discrete card, whose pool keeps 512 MiB.
        let mosaic = GpuMoments::memory_needed(127_780_092, true, 3, false) + (512 << 20);
        assert!(!GpuMoments::fits(mosaic, 8 << 30));
        assert!(GpuMoments::fits(mosaic, 32 << 30));
        // The Pleiades stack's 61 million samples held 5.2 GiB on an M4 Max,
        // whose CPU maps the buffers, with a 1 GiB pool.
        let pleiades = GpuMoments::memory_needed(61_171_488, true, 3, true) + (1 << 30);
        assert_eq!(pleiades >> 20, 5924);
        assert!(GpuMoments::fits(pleiades, 110_100 << 20));
        // On a discrete card its frame calls stage their copies: 8.1 GiB of
        // VRAM on an RTX 5090, with the desktop, so not on an 8 GB card.
        let discrete = GpuMoments::memory_needed(61_171_488, true, 3, false) + (512 << 20);
        assert!(!GpuMoments::fits(discrete, 8 << 30));
        assert!(GpuMoments::fits(discrete, 12 << 30));
        // Unweighted moments are four words a sample, not five.
        assert_eq!(
            GpuMoments::memory_needed(1 << 20, true, 3, true)
                - GpuMoments::memory_needed(1 << 20, false, 3, true),
            4 << 20
        );
        // A concurrency past any count does not overflow.
        assert_eq!(
            GpuMoments::memory_needed(1 << 20, true, usize::MAX, false),
            u64::MAX
        );
    }

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

    /// Each pass runs on the GPU under the defaults, and on the CPU, `None`
    /// here, when the call's options keep it there or keep the GPU out.
    #[test]
    fn a_call_keeps_each_pass_where_its_options_say() {
        if adapter_description().is_none() {
            eprintln!("no GPU: options not compared");
            return;
        }
        let everywhere = GpuOptions::default();
        let cpu = GpuOptions::cpu();
        let without = |pass: fn(&mut GpuOptions)| {
            let mut options = GpuOptions::default();
            pass(&mut options);
            options
        };
        let mono = star_field(1100, 1000, 1);
        let rgb = star_field(1100, 1000, 3);
        let region = ReferenceRegion {
            x: 0,
            y: 0,
            width: 1100,
            height: 1000,
        };
        let identity = |x: f64, y: f64| (x, y);
        let resample = |options: &GpuOptions| {
            resample(&rgb, region, &identity, GpuSampling::Bilinear, options).is_some()
        };
        assert!(resample(&everywhere));
        assert!(!resample(&without(|options| options.resampling = false)));
        assert!(!resample(&cpu));
        let vng = |options: &GpuOptions| {
            vng(&mono.data, (1100, 1000), [0, 1, 1, 2], [1.0; 3], options).is_some()
        };
        assert!(vng(&everywhere));
        assert!(!vng(&without(|options| options.debayer = false)));
        assert!(!vng(&cpu));
        let config = seiza::DetectConfig {
            backend: seiza::DetectBackend::F32,
            ..seiza::DetectConfig::default()
        };
        let detect = |options: &GpuOptions| detect_stars(&rgb, &config, options).is_some();
        assert!(detect(&everywhere));
        assert!(!detect(&without(|options| options.detection = false)));
        assert!(!detect(&cpu));
        let screen = |options: &GpuOptions| {
            defect_candidates(&mono.data, (1100, 1000, 1, 2), [50.0; 4], options).is_some()
        };
        assert!(screen(&everywhere));
        assert!(!screen(&without(|options| options.defects = false)));
        assert!(!screen(&cpu));
        let samples = 1100 * 1000;
        let (mean, m2) = (vec![0.0_f32; samples], vec![0.0_f32; samples]);
        let (count, rejected) = (vec![0_u32; samples], vec![0_u32; samples]);
        let upload = |options: &GpuOptions| {
            let moments = MomentSlices {
                mean: &mean,
                m2: &m2,
                count: &count,
                rejected: &rejected,
                weight_sum: None,
            };
            GpuMoments::upload(moments, options).is_some()
        };
        assert!(upload(&everywhere));
        assert!(!upload(&without(|options| options.accumulator = false)));
        assert!(!upload(&cpu));
    }

    /// A frame and its clip mask resampled in one pass come out as each does
    /// resampled alone, to the bit: the mask bilinearly, the frame clamped
    /// Lanczos-3 or bilinear, over the whole grid and a region of it.
    #[test]
    fn a_frame_and_its_mask_resample_together_as_apart() {
        if adapter_description().is_none() {
            eprintln!("no GPU: joint resampling not compared");
            return;
        }
        let transform = SimilarityTransform {
            scale: 0.9996,
            rotation_radians: 1.3_f64.to_radians(),
            translation_x: -4.6,
            translation_y: 7.1,
        };
        let inverse = transform.inverse_map();
        let source = star_field(1100, 1000, 3);
        let mut mask = star_field(1100, 1000, 1);
        for value in &mut mask.data {
            if value.is_finite() {
                *value = (*value / 60_000.0).min(1.0);
            }
        }
        let spans = crate::registration::lanczos_clamp_spans(&source);
        let mut clamp_spans = [0.0; 4];
        clamp_spans[..3].copy_from_slice(&spans);
        for sampling in [GpuSampling::Lanczos3 { clamp_spans }, GpuSampling::Bilinear] {
            for region in [
                ReferenceRegion {
                    x: 0,
                    y: 0,
                    width: 1100,
                    height: 1000,
                },
                ReferenceRegion {
                    x: 37,
                    y: 211,
                    width: 640,
                    height: 480,
                },
            ] {
                let alone =
                    resample(&source, region, &inverse, sampling, &GpuOptions::default()).unwrap();
                let mask_alone = resample(
                    &mask,
                    region,
                    &inverse,
                    GpuSampling::Bilinear,
                    &GpuOptions::default(),
                )
                .unwrap();
                let (together, mask_together) = resample_masked(
                    &source,
                    Some(ClipView::Image(&mask)),
                    region,
                    &inverse,
                    sampling,
                    &GpuOptions::default(),
                )
                .unwrap();
                let mask_together = mask_together.expect("the mask was taken");
                for (a, b) in alone.data.iter().zip(&together.data) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
                for (a, b) in mask_alone.data.iter().zip(&mask_together.data) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            }
        }
        release_resident();
    }

    #[test]
    fn a_listed_clip_mask_resamples_as_its_image_does() {
        if adapter_description().is_none() {
            return;
        }
        let (width, height) = (1100, 1000);
        let source = LinearImage::new(
            width,
            height,
            1,
            (0..width * height)
                .map(|index| 100.0 + (index % 127) as f32)
                .collect(),
        )
        .unwrap();
        let inverse = SimilarityTransform {
            scale: 1.0,
            rotation_radians: 0.002,
            translation_x: 0.3,
            translation_y: -0.25,
        }
        .inverse_map();
        let region = ReferenceRegion {
            x: 37,
            y: 211,
            width: 640,
            height: 480,
        };
        // A star's core, and enough scattered pixels that the compact
        // readback overflows and the frame is resampled again whole.
        for step in [0, 131] {
            let mut pixels: Vec<u32> = (240..248)
                .flat_map(|y| (540..548).map(move |x| (y * width + x) as u32))
                .collect();
            if step > 0 {
                pixels.extend((0..width * height).step_by(step).map(|pixel| pixel as u32));
                pixels.sort_unstable();
                pixels.dedup();
            }
            let listed = ClipView::Listed {
                width,
                height,
                pixels: &pixels,
            };
            let mut image = LinearImage::new(width, height, 1, vec![0.0; width * height]).unwrap();
            for &pixel in &pixels {
                image.data[pixel as usize] = 1.0;
            }
            let same = |a: &[f32], b: &[f32]| {
                a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
            };
            let resample = |mask| {
                resample_masked(
                    &source,
                    Some(mask),
                    region,
                    &inverse,
                    GpuSampling::Bilinear,
                    &GpuOptions::default(),
                )
                .unwrap()
            };
            let ((from_list, list_mask), (from_image, image_mask)) =
                (resample(listed), resample(ClipView::Image(&image)));
            assert!(same(&from_list.data, &from_image.data));
            assert!(same(&list_mask.unwrap().data, &image_mask.unwrap().data));
            let clipped = |mask| {
                let (_, mask) = resample_clipped(
                    &source,
                    Some(mask),
                    region,
                    &inverse,
                    GpuSampling::Bilinear,
                    &GpuOptions::default(),
                )
                .unwrap();
                match mask.unwrap() {
                    ClipMask::Dense(mask) => (true, mask.data),
                    ClipMask::Sparse { entries, .. } => {
                        let mut data = vec![0.0; region.width * region.height];
                        for (index, value) in entries {
                            data[index] = value;
                        }
                        (false, data)
                    }
                }
            };
            let ((list_dense, from_list), (image_dense, from_image)) =
                (clipped(listed), clipped(ClipView::Image(&image)));
            assert_eq!(list_dense, step > 0, "only the scattered mask overflows");
            assert_eq!(image_dense, list_dense);
            assert!(same(&from_list, &from_image));
        }
        release_resident();
    }

    #[test]
    fn gpu_clipping_lists_preserve_every_contribution_and_dense_overflow() {
        if adapter_description().is_none() {
            return;
        }
        let (width, height) = (1100, 1000);
        let source = LinearImage::new(
            width,
            height,
            3,
            (0..width * height * 3)
                .map(|index| 100.0 + (index % 127) as f32)
                .collect(),
        )
        .unwrap();
        let transform = SimilarityTransform {
            scale: 1.0,
            rotation_radians: 0.0,
            translation_x: 0.3,
            translation_y: -0.25,
        };
        let inverse = transform.inverse_map();
        for density in [1, 2, 0] {
            let mut mask = LinearImage::new(width, height, 1, vec![0.0; width * height]).unwrap();
            if density == 2 {
                mask.data.fill(1.0);
            } else if density == 1 {
                for y in 240..248 {
                    for x in 540..548 {
                        mask.data[y * width + x] = 0.25;
                    }
                }
                mask.data[300 * width + 100] = f32::NAN;
                mask.data[350 * width + 110] = -0.0;
            }
            for region in [
                ReferenceRegion {
                    x: 0,
                    y: 0,
                    width,
                    height,
                },
                ReferenceRegion {
                    x: 37,
                    y: 211,
                    width: 640,
                    height: 480,
                },
            ] {
                let (dense_image, dense) = resample_masked(
                    &source,
                    Some(ClipView::Image(&mask)),
                    region,
                    &inverse,
                    GpuSampling::Bilinear,
                    &GpuOptions::default(),
                )
                .unwrap();
                let dense = dense.unwrap();
                let (image, compact) = resample_clipped(
                    &source,
                    Some(ClipView::Image(&mask)),
                    region,
                    &inverse,
                    GpuSampling::Bilinear,
                    &GpuOptions::default(),
                )
                .unwrap();
                assert!(
                    dense_image
                        .data
                        .iter()
                        .zip(&image.data)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                match compact.unwrap() {
                    ClipMask::Dense(mask) => {
                        assert_eq!(density, 2, "only the full mask should overflow");
                        assert!(
                            dense
                                .data
                                .iter()
                                .zip(mask.data)
                                .all(|(a, b)| a.to_bits() == b.to_bits())
                        );
                    }
                    ClipMask::Sparse {
                        width,
                        height,
                        entries,
                    } => {
                        assert_ne!(density, 2);
                        assert_eq!((width, height), (region.width, region.height));
                        let expected = dense
                            .data
                            .iter()
                            .enumerate()
                            .filter(|(_, value)| value.is_finite() && **value > 0.0)
                            .map(|(index, value)| (index, value.to_bits()))
                            .collect::<Vec<_>>();
                        let mut actual = entries
                            .into_iter()
                            .map(|(index, value)| (index, value.to_bits()))
                            .collect::<Vec<_>>();
                        actual.sort_unstable_by_key(|entry| entry.0);
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
        // CPU plans run before GPU admission, including from a bounded
        // one-thread pool with several frame tasks and one GPU-call permit.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let mut mask = LinearImage::new(width, height, 1, vec![0.0; width * height]).unwrap();
        mask.data[244 * width + 541] = 1.0;
        pool.install(|| {
            (0..4).into_par_iter().for_each(|_| {
                let (_, mask) = resample_clipped(
                    &source,
                    Some(ClipView::Image(&mask)),
                    ReferenceRegion {
                        x: 37,
                        y: 211,
                        width: 640,
                        height: 480,
                    },
                    &inverse,
                    GpuSampling::Bilinear,
                    &GpuOptions {
                        concurrency: 1,
                        ..GpuOptions::default()
                    },
                )
                .unwrap();
                assert!(
                    matches!(mask, Some(ClipMask::Sparse { entries, .. }) if entries.len() == 4)
                );
            })
        });
        release_resident();
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
                let gpu = resample(
                    &source,
                    region,
                    &inverse,
                    gpu_sampling,
                    &GpuOptions::default(),
                )
                .expect(&what);
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
            (300, 500, 1, 25, true),
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
            // Enough samples to use the GPU, with both buffer sets reused
            // and a shorter tail. Band boundaries must change no bits,
            // including variances, rejection decisions and frame counts.
            let rows = MINIMUM_SAMPLES.div_ceil(width * channels * depth) + 1;
            let banded = crate::integrate_registered_frames_banded(
                depth,
                &crate::BatchStackOptions {
                    band_memory_bytes: 2 * rows * width * channels * depth * 4,
                    ..options.clone()
                },
                |index| Ok(frames[index].clone()),
                |_, _, _| {},
            )
            .unwrap();
            crate::batch::tests::assert_same_result(&banded, &[], &gpu, &[]);
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
            ("scatter", include_str!("gpu/scatter.wgsl")),
            ("normalize", include_str!("gpu/normalize.wgsl")),
            ("gather", include_str!("gpu/gather.wgsl")),
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
