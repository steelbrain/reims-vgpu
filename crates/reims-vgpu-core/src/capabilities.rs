//! Host execution capabilities visible to semantic planning.

use reims_vgpu_protocol::TexelLayout;

/// Maximum number of FIFO channels represented by the device's `u32` masks.
pub const MAX_CHANNELS: usize = 32;
const _: () = assert!(MAX_CHANNELS <= u32::BITS as usize);

/// Host limits used to reduce the device-info values advertised to the guest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceInfoLimits {
    pub max_sample_count: u32,
    pub d24_stencil8: bool,
    pub max_threads_per_threadgroup: [u32; 3],
    pub max_threadgroup_memory_bytes: u32,
    pub native_fp16: bool,
}

/// Host-GPU facts available to semantic planning.
///
/// These describe how the executor can implement an already-decoded command.
/// They do not advertise guest protocol features or select resource lifetimes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutorCapabilities {
    pub device_info: DeviceInfoLimits,
    pub max_compute_workgroup_invocations: u32,
    pub thread_execution_width: u32,
    pub max_render_target_dimension: u32,
    pub deferred_gpu_only_content: bool,
    pub storage_image_write_without_format: bool,
}

impl Default for ExecutorCapabilities {
    fn default() -> Self {
        Self {
            device_info: DeviceInfoLimits {
                max_sample_count: 1,
                d24_stencil8: false,
                max_threads_per_threadgroup: [128, 128, 64],
                max_threadgroup_memory_bytes: 16_384,
                native_fp16: false,
            },
            max_compute_workgroup_invocations: 128,
            thread_execution_width: 1,
            max_render_target_dimension: 4096,
            deferred_gpu_only_content: false,
            storage_image_write_without_format: false,
        }
    }
}

/// Read-only host capability service used by semantic planning.
///
/// This port answers how an already-decoded operation can be implemented. It
/// neither advertises guest protocol features nor owns execution, lifecycle,
/// presentation, or observation services.
pub trait CapabilityService: std::fmt::Debug + Send + Sync {
    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities::default()
    }

    fn render_target_layout_supported(&self, layout: TexelLayout) -> bool {
        matches!(layout, TexelLayout::Rgba8 | TexelLayout::Bgra8)
    }

    fn sampled_layout_linear_filter_supported(&self, _layout: TexelLayout) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::{CapabilityService, ExecutorCapabilities};
    use reims_vgpu_protocol::TexelLayout;

    #[derive(Debug)]
    struct ConservativeCapabilities;

    impl CapabilityService for ConservativeCapabilities {}

    #[test]
    fn conservative_capabilities_do_not_enable_optional_execution_paths() {
        let caps = ExecutorCapabilities::default();
        assert_eq!(caps.device_info.max_sample_count, 1);
        assert!(!caps.device_info.d24_stencil8);
        assert!(!caps.device_info.native_fp16);
        assert!(!caps.storage_image_write_without_format);
        assert!(!caps.deferred_gpu_only_content);
    }

    #[test]
    fn conservative_layout_support_is_independently_queryable() {
        let service = ConservativeCapabilities;
        assert!(service.render_target_layout_supported(TexelLayout::Rgba8));
        assert!(service.render_target_layout_supported(TexelLayout::Bgra8));
        assert!(!service.render_target_layout_supported(TexelLayout::Rgba16Float));
    }
}
