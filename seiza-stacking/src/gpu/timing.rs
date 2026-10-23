//! Optional device timestamps for traced compute passes. Resolve and map
//! them beside the pass's existing outputs, without another queue wait.

use super::{Gpu, Trace};

pub(super) struct GpuTimer {
    queries: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    pub(super) readback: wgpu::Buffer,
    period_ns: f32,
}

impl GpuTimer {
    pub(super) fn new(gpu: &Gpu, enabled: bool) -> Option<Self> {
        if !enabled
            || !gpu
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return None;
        }
        Some(Self {
            queries: gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("compute-timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }),
            resolved: gpu.buffer(
                "resolved-timestamps",
                16,
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            ),
            readback: gpu.buffer(
                "timestamp-readback",
                16,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            period_ns: gpu.queue.get_timestamp_period(),
        })
    }

    pub(super) fn writes(&self) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.queries,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        }
    }

    pub(super) fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.queries, 0..2, &self.resolved, 0);
        encoder.copy_buffer_to_buffer(&self.resolved, 0, &self.readback, 0, 16);
    }

    /// Called after the existing output wait has mapped our readback too.
    pub(super) fn record(&self, trace: &mut Trace) -> Result<(), String> {
        trace.device_ms = Some(trace.device_ms.unwrap_or(0.0) + self.elapsed_ms()?);
        Ok(())
    }

    pub(super) fn elapsed_ms(&self) -> Result<f64, String> {
        let view = self
            .readback
            .slice(..)
            .get_mapped_range()
            .map_err(|error| error.to_string())?;
        let ticks: &[u64] = bytemuck::cast_slice(&view);
        let ms = ticks[1].wrapping_sub(ticks[0]) as f64 * f64::from(self.period_ns) / 1_000_000.0;
        drop(view);
        self.readback.unmap();
        Ok(ms)
    }
}
