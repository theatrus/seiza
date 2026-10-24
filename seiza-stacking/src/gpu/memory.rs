//! How much memory the device has, which sets what stacking keeps there,
//! and how much it holds, for traces.
//!
//! wgpu offers neither, so each is asked of the backend beneath it. The
//! capacity is fixed for a device, rather than the budget the system grants
//! at the moment, so a machine plans the same way on every run.

/// Bytes of device memory the GPU's buffers can use without the system
/// paging them, or `None` when the backend does not say: Metal's
/// recommended working set, which on Apple silicon is most of the unified
/// memory; Vulkan's largest device-local heap; and on Direct3D 12 the
/// dedicated memory of a discrete GPU, or the system memory an integrated
/// one may share as well.
pub(super) fn capacity(adapter: &wgpu::Adapter) -> Option<u64> {
    let info = adapter.get_info();
    let bytes = match info.backend {
        #[cfg(target_vendor = "apple")]
        wgpu::Backend::Metal => metal_capacity(adapter),
        #[cfg(not(any(target_vendor = "apple", target_arch = "wasm32")))]
        wgpu::Backend::Vulkan => vulkan_capacity(adapter),
        #[cfg(windows)]
        wgpu::Backend::Dx12 => {
            dx12_capacity(adapter, info.device_type == wgpu::DeviceType::IntegratedGpu)
        }
        _ => None,
    };
    bytes.filter(|&bytes| bytes > 0)
}

/// Bytes the device holds for this process now, where the backend says:
/// Metal's allocated size, DXGI's local plus non-local usage on Direct3D
/// 12, or wgpu's allocator reservations elsewhere. DXGI also counts the
/// committed/shared allocations which bypass wgpu's allocator on an iGPU.
pub(super) fn in_use(_adapter: &wgpu::Adapter, device: &wgpu::Device) -> Option<u64> {
    #[cfg(windows)]
    if _adapter.get_info().backend == wgpu::Backend::Dx12 {
        return dx12_in_use(_adapter);
    }
    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::MTLDevice;
        // SAFETY: reads a property of the device and keeps nothing.
        if let Some(device) = unsafe { device.as_hal::<wgpu::hal::api::Metal>() } {
            return Some(device.raw_device().currentAllocatedSize() as u64);
        }
    }
    device
        .generate_allocator_report()
        .map(|report| report.total_reserved_bytes)
}

#[cfg(target_vendor = "apple")]
fn metal_capacity(adapter: &wgpu::Adapter) -> Option<u64> {
    use objc2_metal::MTLDevice;
    // SAFETY: reads a property of the device and keeps nothing.
    let adapter = unsafe { adapter.as_hal::<wgpu::hal::api::Metal>() }?;
    Some(adapter.raw_device().recommendedMaxWorkingSetSize())
}

#[cfg(not(any(target_vendor = "apple", target_arch = "wasm32")))]
fn vulkan_capacity(adapter: &wgpu::Adapter) -> Option<u64> {
    /// `VK_MEMORY_HEAP_DEVICE_LOCAL_BIT`.
    const DEVICE_LOCAL: u32 = 1;
    // SAFETY: reads the physical device's memory properties and keeps
    // nothing.
    let adapter = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
    let properties = unsafe {
        adapter
            .shared_instance()
            .raw_instance()
            .get_physical_device_memory_properties(adapter.raw_physical_device())
    };
    // A discrete GPU's other device-local heap, where it has one, is the
    // small window the CPU writes through.
    properties.memory_heaps[..properties.memory_heap_count as usize]
        .iter()
        .filter(|heap| heap.flags.as_raw() & DEVICE_LOCAL != 0)
        .map(|heap| heap.size)
        .max()
}

#[cfg(windows)]
fn dx12_capacity(adapter: &wgpu::Adapter, integrated: bool) -> Option<u64> {
    // SAFETY: reads the adapter's description and keeps nothing.
    let adapter = unsafe { adapter.as_hal::<wgpu::hal::api::Dx12>() }?;
    let description = unsafe { adapter.raw_adapter().GetDesc1() }.ok()?;
    let dedicated = description.DedicatedVideoMemory as u64;
    Some(if integrated {
        dedicated + description.SharedSystemMemory as u64
    } else {
        dedicated
    })
}

#[cfg(windows)]
fn dx12_in_use(adapter: &wgpu::Adapter) -> Option<u64> {
    use windows::Win32::Graphics::Dxgi::{
        DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
    };

    // SAFETY: only reads this process's usage on the adapter and keeps nothing.
    let adapter = unsafe { adapter.as_hal::<wgpu::hal::api::Dx12>() }?;
    let local = adapter
        .raw_adapter()
        .query_video_memory_info(DXGI_MEMORY_SEGMENT_GROUP_LOCAL)
        .ok()?;
    let non_local = adapter
        .raw_adapter()
        .query_video_memory_info(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)
        .ok()?;
    Some(local.CurrentUsage + non_local.CurrentUsage)
}
