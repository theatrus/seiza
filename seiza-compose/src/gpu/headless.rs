//! The GPU compositor without a window: a picture rendered from a
//! [`Source`] into a [`Sink`], tile by tile, as [`crate::render::render`]
//! renders it on the CPU.

use super::{DrawArea, GpuCompositor, GpuInput, GpuScene, GpuView, TileDraw};
use crate::picture::Picture;
use crate::render::{PaddedPlanes, Plan, RenderOptions, Sink, Source};
use crate::{Error, ReferenceRegion, Result};

/// A GPU device to render on.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GpuContext {
    /// A device on the system's GPU, or `None` when it has none wgpu can
    /// use.
    pub fn new() -> Option<Self> {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                    ..Default::default()
                })
                .await
                .ok()?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("seiza compose"),
                    required_limits: adapter.limits(),
                    ..Default::default()
                })
                .await
                .ok()?;
            Some(Self { device, queue })
        })
    }

    /// The device and queue, for this crate's own GPU checks.
    #[cfg(test)]
    pub(crate) fn parts(&self) -> (&wgpu::Device, &wgpu::Queue) {
        (&self.device, &self.queue)
    }

    /// A context on a device and queue the host already has.
    pub fn from_device(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            device: device.clone(),
            queue: queue.clone(),
        }
    }
}

/// Renders `region` of `picture` on the GPU, as [`crate::render::render`]
/// renders it on the CPU: in tiles of `options.tile`, each with the margin
/// its masks and heals read, reflected at the region's edge. Inputs go to
/// the GPU as half floats, divided by their largest sample in the region.
/// An uncovered pixel is `NaN`, as on the CPU.
pub fn render_gpu(
    picture: &Picture,
    source: &dyn Source,
    region: ReferenceRegion,
    sink: &dyn Sink,
    context: &GpuContext,
    options: &RenderOptions<'_>,
) -> Result<()> {
    let (width, height) = source.size();
    if region.width == 0
        || region.height == 0
        || region.x + region.width > width
        || region.y + region.height > height
    {
        return Err(Error::Color(format!(
            "the region {}x{} at ({}, {}) is not within the inputs' {width}x{height}",
            region.width, region.height, region.x, region.y
        )));
    }
    // Finish steps past what one draw holds are drawn as steps on the
    // stack, whose reaches add up, so a tile takes the larger margin.
    let plan = {
        let finish = Plan::new(picture);
        let stacked = Plan::new(&super::stack_finish(picture));
        if stacked.margin > finish.margin {
            stacked
        } else {
            finish
        }
    };
    // Each input's largest sample in the region: what a texel of 1 stands
    // for.
    let mut scales = vec![f32::MIN_POSITIVE; plan.inputs.len()];
    for row in (region.y..region.y + region.height).step_by(256) {
        let strip = ReferenceRegion {
            x: region.x,
            y: row,
            width: region.width,
            height: 256.min(region.y + region.height - row),
        };
        for (scale, input) in scales.iter_mut().zip(&plan.inputs) {
            for sample in source.read(*input, strip)? {
                if sample.is_finite() {
                    *scale = scale.max(sample);
                }
            }
        }
    }
    let largest_input = plan.inputs.iter().copied().max().unwrap_or(0);
    let mut placed = vec![
        GpuInput {
            layer: 0,
            scale: 1.0
        };
        largest_input + 1
    ];
    for (layer, (input, scale)) in plan.inputs.iter().zip(&scales).enumerate() {
        placed[*input] = GpuInput {
            layer,
            scale: *scale,
        };
    }
    let scene = GpuScene::new(picture, &placed);
    let mut compositor = GpuCompositor::new(
        &context.device,
        &context.queue,
        wgpu::TextureFormat::Rgba32Float,
    );
    let tile = options.tile.max(16);
    let tiles: Vec<ReferenceRegion> = (0..region.height.div_ceil(tile))
        .flat_map(|row| {
            (0..region.width.div_ceil(tile)).map(move |column| {
                let x = region.x + column * tile;
                let y = region.y + row * tile;
                ReferenceRegion {
                    x,
                    y,
                    width: tile.min(region.x + region.width - x),
                    height: tile.min(region.y + region.height - y),
                }
            })
        })
        .collect();
    for (done, tile) in tiles.iter().enumerate() {
        if options.stop.is_some_and(|stop| stop()) {
            return Err(Error::Cancelled);
        }
        let pixels = draw_tile(
            &mut compositor,
            &scene,
            &plan,
            &scales,
            source,
            region,
            *tile,
            context,
        )?;
        sink.put(*tile, &pixels)?;
        if let Some(progress) = options.progress {
            progress((done + 1) as f32 / tiles.len() as f32);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn draw_tile(
    compositor: &mut GpuCompositor,
    scene: &GpuScene,
    plan: &Plan,
    scales: &[f32],
    source: &dyn Source,
    region: ReferenceRegion,
    tile: ReferenceRegion,
    context: &GpuContext,
) -> Result<Vec<f32>> {
    let PaddedPlanes {
        planes,
        origin,
        width,
        height,
    } = plan.padded_planes(source, region, tile)?;
    let covered: Vec<bool> = (0..width * height)
        .map(|at| planes.iter().all(|plane| plane[at].is_finite()))
        .collect();
    // The inputs as the GPU reads them: one layer each, divided by its
    // scale, within 0 to 1.
    let layers = planes.len().max(1);
    let mut bits = Vec::with_capacity(width * height * layers);
    for (plane, scale) in planes.iter().zip(scales) {
        bits.extend(plane.iter().map(|sample| {
            let value = if sample.is_finite() {
                (sample / scale).clamp(0.0, 1.0)
            } else {
                0.0
            };
            half::f16::from_f32(value).to_bits()
        }));
    }
    bits.resize(width * height * layers, 0);
    let texture = context.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("compose inputs"),
        size: wgpu::Extent3d {
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: layers as u32,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    context.queue.write_texture(
        texture.as_image_copy(),
        bytemuck::cast_slice(&bits),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width as u32 * 2),
            rows_per_image: Some(height as u32),
        },
        texture.size(),
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let inputs = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compose inputs"),
            layout: compositor.tile_layout(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
    let targets = compositor.targets(width as u32, height as u32);
    let left = origin[0] as f32;
    let top = origin[1] as f32;
    let whole = [0, 0, width as u32, height as u32];
    compositor.draw(
        scene,
        Some(0),
        &targets,
        GpuView {
            center: [
                origin[0] as f64 + width as f64 / 2.0,
                origin[1] as f64 + height as f64 / 2.0,
            ],
            zoom: 1.0,
        },
        DrawArea {
            viewport: whole,
            scissor: whole,
        },
        &[TileDraw {
            inputs: &inputs,
            rect: [left, top, left + width as f32, top + height as f32],
            uv_rect: [0.0, 0.0, 1.0, 1.0],
        }],
        &[],
        0.0,
        true,
    );
    let row_bytes = (width as u32 * 16).next_multiple_of(256);
    let buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("compose readback"),
        size: u64::from(row_bytes) * height as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        targets.finished.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: None,
            },
        },
        targets.finished.size(),
    );
    context.queue.submit([encoder.finish()]);
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    context
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| Error::Color(format!("the GPU did not finish: {error}")))?;
    let mapped = buffer
        .slice(..)
        .get_mapped_range()
        .map_err(|error| Error::Color(format!("the GPU's result could not be read: {error}")))?;
    let margin = plan.margin;
    let mut out = Vec::with_capacity(tile.width * tile.height * 3);
    for row in 0..tile.height {
        let line: &[f32] = bytemuck::cast_slice(
            &mapped[(row + margin) * row_bytes as usize
                ..(row + margin) * row_bytes as usize + width * 16],
        );
        for column in 0..tile.width {
            let at = (row + margin) * width + column + margin;
            if covered[at] {
                let texel = &line[(column + margin) * 4..(column + margin) * 4 + 3];
                out.extend_from_slice(texel);
            } else {
                out.extend_from_slice(&[f32::NAN; 3]);
            }
        }
    }
    Ok(out)
}
