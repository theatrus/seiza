//! Compact, exact readback for the CPU's existing fits. Plans are shared by
//! frames of the same reference and upload their sample indices only once.
use super::{Bound, GpuFrame, OutputRetention, Trace, opened};
use crate::normalization::NormalizationMode;
use std::sync::{Arc, Mutex, OnceLock};

/// Created only when compact live-only readback is used. Cache-backed and
/// shared-memory stacks do not pay for these extra pipelines at device open.
pub(super) struct Pipelines {
    layout: wgpu::BindGroupLayout,
    compute: [wgpu::ComputePipeline; 2],
}
impl Pipelines {
    fn new(gpu: &super::Gpu) -> Self {
        let device = &gpu.device;
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let gather_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("preparation samples"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gather.wgsl").into()),
        });
        let gather_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preparation samples"),
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
                storage(3, false),
                storage(4, false),
            ],
        });
        let gather_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("preparation samples"),
                bind_group_layouts: &[Some(&gather_layout)],
                immediate_size: 0,
            });
        let gather_pipelines = ["gather", "count_finite"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&gather_pipeline_layout),
                module: &gather_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        Self {
            layout: gather_layout,
            compute: gather_pipelines,
        }
    }
}

#[derive(PartialEq)]
struct Key {
    shape: (usize, usize, usize),
    mode: NormalizationMode,
    stars: Vec<(f64, f64)>,
}

pub(crate) struct SamplePlan {
    lookup: Vec<u32>,
    indices: wgpu::Buffer,
    count: usize,
}

/// One recent reference is cached, rather than retaining an unbounded set
/// of image-sized lookup tables. In-flight frames own their plan by Arc.
struct CachedPlan {
    key: Key,
    plan: Option<Arc<SamplePlan>>,
}
static PLAN: OnceLock<Mutex<Option<CachedPlan>>> = OnceLock::new();

pub(crate) fn plan(
    shape: (usize, usize, usize),
    mode: NormalizationMode,
    stars: &[(f64, f64)],
) -> Option<Arc<SamplePlan>> {
    let gpu = opened()?;
    // A shared-memory adapter already lets the CPU read its output mapping
    // directly. Avoid a second GPU scan and a lookup table on that path.
    if gpu.mapped {
        return None;
    }
    let samples = shape.0.checked_mul(shape.1)?.checked_mul(shape.2)?;
    if samples > u32::MAX as usize {
        return None;
    }
    let key = Key {
        shape,
        mode,
        stars: stars.to_vec(),
    };
    let mut cache = PLAN
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    if let Some(cached) = cache.as_ref().filter(|cached| cached.key == key) {
        return cached.plan.clone();
    }
    let mut indices = crate::normalization::preparation_sample_indices(shape, mode, stars, None);
    // A small tile may need almost every pixel; keep the existing dense path
    // when gathering would transfer more indices and values than it saves.
    if indices.len() >= samples / 2 {
        *cache = Some(CachedPlan { key, plan: None });
        return None;
    }
    if indices.is_empty() {
        indices.push(0);
    }
    let mut lookup = vec![u32::MAX; samples];
    for (at, &index) in indices.iter().enumerate() {
        lookup[index as usize] = at as u32;
    }
    let count = indices.len();
    let indices = wgpu::util::DeviceExt::create_buffer_init(
        &gpu.device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("preparation-sample-indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::STORAGE,
        },
    );
    let plan = Arc::new(SamplePlan {
        lookup,
        indices,
        count,
    });
    *cache = Some(CachedPlan {
        key,
        plan: Some(plan.clone()),
    });
    Some(plan)
}

pub(crate) struct FrameSamples {
    plan: Arc<SamplePlan>,
    values: Vec<f32>,
    pub(crate) finite: usize,
}

impl FrameSamples {
    /// Where each sample of the frame lies among those gathered, or
    /// `u32::MAX` where it was not gathered.
    pub(crate) fn lookup(&self) -> &[u32] {
        &self.plan.lookup
    }

    pub(crate) fn pixels(&self) -> crate::image::SampledPixels<'_> {
        crate::image::SampledPixels {
            lookup: &self.plan.lookup,
            values: &self.values,
            tiles: None,
        }
    }
}

impl GpuFrame {
    pub(crate) fn gather(
        &self,
        plan: Arc<SamplePlan>,
        options: &crate::GpuOptions,
    ) -> Result<FrameSamples, String> {
        let gpu = opened().ok_or("no GPU")?;
        let _call = gpu.admitted(options.concurrency);
        let pipelines = gpu.gather.get_or_init(|| Pipelines::new(gpu));
        let tracing = options.trace;
        let samples = self.width * self.height * self.channels;
        let columns = (samples as u32).div_ceil(256).min(4096);
        let params = [samples as u32, plan.count as u32, columns * 256, 0];
        let uniform = wgpu::util::DeviceExt::create_buffer_init(
            &gpu.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("preparation-sample-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let bindings = [
            Bound::Input(&uniform),
            Bound::Input(self.buffer.as_ref().ok_or("the frame has left the GPU")?),
            Bound::Input(&plan.indices),
            Bound::Output(plan.count),
            Bound::ZeroedOutput(1),
        ];
        let groups = |count: usize| {
            let groups = (count as u32).div_ceil(256);
            (groups.min(columns), groups.div_ceil(columns))
        };
        let mut trace = Trace::new(tracing);
        let (mut outputs, _) = gpu.compute_keeping::<f32>(
            &pipelines.layout,
            &[
                (&pipelines.compute[0], groups(plan.count)),
                (&pipelines.compute[1], groups(samples)),
            ],
            &bindings,
            OutputRetention::Discard,
            &mut trace,
        )?;
        trace.print("preparation samples");
        if tracing {
            eprintln!(
                "seiza-stacking gpu: preparation readback: {} of {} image bytes",
                plan.count * 4 + 4,
                samples * 4
            );
        }
        let finite = outputs.pop().unwrap()[0].to_bits() as usize;
        Ok(FrameSamples {
            plan,
            values: outputs.pop().unwrap(),
            finite,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_gather_preserves_sample_bits_and_counts_nonfinite_values() {
        let Some(gpu) = super::super::gpu(&crate::GpuOptions::default()) else {
            return;
        };
        if gpu.mapped {
            return;
        }
        let (width, height) = (1025, 1027);
        let data = (0..width * height)
            .map(|index| match index % 997 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => f32::NEG_INFINITY,
                3 => -0.0,
                _ => (index as f32 * 0.03125).sin() + 100.0,
            })
            .collect::<Vec<_>>();
        let frame = GpuFrame {
            buffer: Some(gpu.upload("gather test", &data).unwrap()),
            usage: gpu.input_usage(),
            width,
            height,
            channels: 1,
            normalization_timer: None,
        };
        let plan = plan((width, height, 1), NormalizationMode::Global, &[]).unwrap();
        let expected = crate::normalization::preparation_sample_indices(
            (width, height, 1),
            NormalizationMode::Global,
            &[],
            None,
        )
        .iter()
        .map(|&index| data[index as usize].to_bits())
        .collect::<Vec<_>>();
        for trace in [false, true] {
            let gathered = frame
                .gather(
                    plan.clone(),
                    &crate::GpuOptions {
                        trace,
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(
                gathered.finite,
                data.iter().filter(|value| value.is_finite()).count()
            );
            assert_eq!(
                gathered
                    .values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
            );
        }
    }
}
