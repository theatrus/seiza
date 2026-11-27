//! A picture's color for a display, on the GPU: what a host's present pass
//! does to the compositor's finished picture.

use crate::space::WorkingSpace;
use bytemuck::{Pod, Zeroable};

/// The WGSL of `to_display(color, mapping)`, the GPU twin of
/// [`WorkingSpace::map_to_display`], for a host to include in its present
/// shader.
pub const DISPLAY_SHADER: &str = include_str!("display.wgsl");

/// The `DisplayMapping` uniform [`DISPLAY_SHADER`] reads, laid out as WGSL
/// lays out a uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct DisplayMapping {
    /// The matrix from working light to display light, by columns of four.
    pub matrix: [f32; 12],
    /// The display's luminance weights, and a 0.
    pub weights: [f32; 4],
}

impl DisplayMapping {
    /// The mapping from a picture in `working` to a display whose
    /// primaries are `display`'s.
    pub fn new(working: WorkingSpace, display: WorkingSpace) -> Self {
        let rows = working.matrix_to(display);
        let mut matrix = [0.0; 12];
        for column in 0..3 {
            for row in 0..3 {
                matrix[column * 4 + row] = rows[row][column];
            }
        }
        let [r, g, b] = display.luma_weights();
        Self {
            matrix,
            weights: [r, g, b, 0.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuContext;

    /// Colors through the shader, read back.
    fn on_gpu(context: &GpuContext, mapping: DisplayMapping, colors: &[[f32; 3]]) -> Vec<[f32; 3]> {
        use wgpu::util::DeviceExt;
        let (device, queue) = context.parts();
        let source = format!(
            "{DISPLAY_SHADER}
            @group(0) @binding(0) var<uniform> mapping: DisplayMapping;
            @group(0) @binding(1) var<storage, read_write> colors: array<vec4f>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3u) {{
                if id.x < arrayLength(&colors) {{
                    colors[id.x] = vec4f(to_display(colors[id.x].rgb, mapping), 1.0);
                }}
            }}"
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("display test"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("display test"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mapping"),
            contents: bytemuck::bytes_of(&mapping),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let padded: Vec<[f32; 4]> = colors.iter().map(|c| [c[0], c[1], c[2], 1.0]).collect();
        let storage = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("colors"),
            contents: bytemuck::cast_slice(&padded),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let size = storage.size();
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("display test"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: storage.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(colors.len().div_ceil(64) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, size);
        queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let view = readback.slice(..).get_mapped_range().unwrap();
        let data: Vec<[f32; 4]> = bytemuck::cast_slice(&view).to_vec();
        data.iter().map(|c| [c[0], c[1], c[2]]).collect()
    }

    #[test]
    fn the_gpu_maps_colors_for_a_display_as_the_cpu_does() {
        let Some(context) = GpuContext::new() else {
            eprintln!("no GPU: the display mapping was not checked");
            return;
        };
        // A grid of colors, HDR highlights and colors outside the display.
        let mut colors = Vec::new();
        for r in 0..9 {
            for g in 0..9 {
                for b in 0..9 {
                    colors.push([r as f32 * 0.25, g as f32 * 0.2, b as f32 * 0.15]);
                }
            }
        }
        for (working, display) in [
            (WorkingSpace::Rec2020, WorkingSpace::DisplayP3),
            (WorkingSpace::Rec2020, WorkingSpace::Srgb),
            (WorkingSpace::Srgb, WorkingSpace::Srgb),
        ] {
            let gpu = on_gpu(&context, DisplayMapping::new(working, display), &colors);
            for (color, shown) in colors.iter().zip(&gpu) {
                let cpu = working.map_to_display(*color, display);
                for channel in 0..3 {
                    assert!(
                        (cpu[channel] - shown[channel]).abs() < 1e-4 * cpu[channel].max(1.0),
                        "{working:?} to {display:?}, {color:?}: {cpu:?} on the CPU, {shown:?}"
                    );
                }
            }
        }
    }
}
