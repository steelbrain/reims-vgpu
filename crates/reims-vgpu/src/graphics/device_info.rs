//! The stage-3 device-info caps blob: the host fills [`RgfxDeviceInfo`] from the
//! platform's [`DeviceCaps`] so the guest bundle's `MTLDevice` answers
//! `supportsFamily:`, the limits and the feature queries the same as the host
//! device (Metal on macOS, the Vulkan platform on Linux). The owner's rule is
//! that the feature support the guest sees is the maximum of what the platform
//! supports; a query the platform cannot answer (a runtime gap, an `Err` in the
//! caps) is reported as unsupported here, never guessed yes.
//!
//! The decoder does not touch this: [`super::GraphicsDevice`] writes it once at
//! the device's init into the region at
//! [`RGFX_DEVICE_INFO_OFFSET`](reims_graphics::wire::RGFX_DEVICE_INFO_OFFSET),
//! and the guest bundle reads it there, not a guest command.

use reims_graphics::wire::{RGFX_DEVICE_INFO_VERSION, RGFX_DEVICE_NAME_LEN, RgfxDeviceInfo};
use reims_vgpu_core::platform::caps::DeviceCaps;
use reims_vgpu_core::raise::RaiseResult;

/// 1 when the query answered a present `true`; 0 for a present `false` or a host
/// gap (an `Err`). A gap is never reported as support.
fn bit(answer: &RaiseResult<bool>) -> u8 {
    u8::from(matches!(answer, Ok(true)))
}

/// The device-info caps blob for `caps`: every value the guest bundle reads once
/// at init, mirrored from the host device's capabilities.
pub fn device_info(caps: &DeviceCaps) -> RgfxDeviceInfo {
    let mut name = [0u8; RGFX_DEVICE_NAME_LEN];
    let name_bytes = caps.name.as_deref().unwrap_or("").as_bytes();
    let name_len = name_bytes.len().min(RGFX_DEVICE_NAME_LEN);
    name[..name_len].copy_from_slice(&name_bytes[..name_len]);

    let t = caps.max_threads_per_threadgroup;
    let l = &caps.limits;
    let f = &caps.flags;
    RgfxDeviceInfo {
        version: RGFX_DEVICE_INFO_VERSION,
        name_len: name_len as u32,
        registry_id: caps.registry_id,
        gpu_core_count: caps.gpu_core_count.unwrap_or(0),
        max_threads_per_threadgroup: [t.width, t.height, t.depth],
        max_vertex_amplification_count: caps.max_vertex_amplification_count,
        heap_buffer_page_private: match &caps.heap_buffer_page_private {
            Ok(sa) => [sa.size, sa.align],
            Err(_) => [0, 0],
        },
        min_linear_alignment_bgra8: *caps.min_linear_alignment_bgra8.as_ref().unwrap_or(&0),
        // In the host `Limits` field order; the bundle reads them in this order.
        limits: [
            l.device_linear_texture_alignment_bytes,
            l.iosurface_read_only_texture_alignment_bytes,
            l.iosurface_texture_alignment_bytes,
            l.linear_texture_alignment_bytes,
            l.max_compute_buffers,
            l.max_compute_local_memory_sizes,
            l.max_compute_samplers,
            l.max_compute_textures,
            l.max_compute_threadgroup_memory,
            l.max_compute_threadgroup_memory_alignment_bytes,
            l.max_fragment_buffers,
            l.max_fragment_samplers,
            l.max_fragment_textures,
            l.max_predicated_nesting_depth,
            l.max_rasterization_rate_layer_count,
            l.max_texture_height_2d,
            l.max_texture_layers,
            l.max_texture_width_2d,
            l.max_threadgroup_memory_length,
            l.max_tile_buffers,
            l.max_tile_samplers,
            l.max_tile_textures,
            l.max_total_compute_threads_per_threadgroup,
            l.max_vertex_buffers,
            l.max_vertex_samplers,
            l.max_vertex_textures,
            l.max_viewport_count,
        ],
        families: core::array::from_fn(|i| bit(&caps.families[i])),
        sample_counts: core::array::from_fn(|i| bit(&caps.sample_counts[i])),
        primitive_types: core::array::from_fn(|i| bit(&caps.primitive_types[i])),
        write_rounding_modes: core::array::from_fn(|i| bit(&caps.write_rounding_modes[i])),
        // In the host `Flags` field order; the bundle reads them in this order.
        flags: [
            u8::from(f.are_raster_order_groups_supported),
            u8::from(f.is_depth24_stencil8_pixel_format_supported),
            u8::from(f.is_framebuffer_read_supported),
            u8::from(f.is_large_mrt_supported),
            u8::from(f.is_rgb10a2_gamma_supported),
            u8::from(f.supports_buffer_with_address_ranges),
            u8::from(f.supports_buffer_with_iosurface),
            u8::from(f.supports_command_buffer_jump),
            u8::from(f.supports_compute_compressed_texture_write),
            u8::from(f.supports_conditional_load_store),
            u8::from(f.supports_dynamic_attribute_stride),
            u8::from(f.supports_extended_xr10_formats),
            u8::from(f.supports_float16_bcubic_filtering),
            u8::from(f.supports_heap_with_address_ranges),
            u8::from(f.supports_image_blocks),
            u8::from(f.supports_memory_order_atomics),
            u8::from(f.supports_native_hardware_fp16),
            u8::from(f.supports_non_square_tile_shaders),
            u8::from(f.supports_programmable_sample_positions),
            u8::from(f.supports_resource_detach_backing),
            u8::from(f.supports_shared_texture_handles),
            u8::from(f.supports_simd_reduction),
            u8::from(f.supports_simd_shuffle_and_fill),
            u8::from(f.supports_texture_2d_multisample_array),
            u8::from(f.supports_tile_shaders),
        ],
        name,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::raise::Raise;
    use reims_vgpu_core::testkit::FakePlatform;

    use super::*;

    /// The blob mirrors the device's caps: the name, a limit and a flag by
    /// position, and the version.
    #[test]
    fn mirrors_the_device_caps() {
        let p = Arc::new(FakePlatform::new());
        let caps = p.device().caps();
        let info = device_info(&caps);

        assert_eq!(info.version, RGFX_DEVICE_INFO_VERSION);
        assert_eq!(info.registry_id, caps.registry_id);
        assert_eq!(
            &info.name[..info.name_len as usize],
            caps.name.as_deref().unwrap().as_bytes()
        );
        assert_eq!(info.gpu_core_count, caps.gpu_core_count.unwrap());
        // The last limit by position is max_viewport_count.
        assert_eq!(info.limits[26], caps.limits.max_viewport_count);
        // The first flag by position is are_raster_order_groups_supported.
        assert_eq!(
            info.flags[0],
            u8::from(caps.flags.are_raster_order_groups_supported)
        );
    }

    /// A present no reads 0, a present yes reads 1, and a platform gap (an `Err`)
    /// reads 0 — never a spurious yes.
    #[test]
    fn a_platform_gap_is_reported_as_unsupported() {
        let p = Arc::new(FakePlatform::new());
        let mut caps = (*p.device().caps()).clone();
        // The fake answers sample count 8 (index 3) with a present no.
        assert_eq!(device_info(&caps).sample_counts[3], 0);
        // A family the platform could not answer at all (a gap) also reads 0.
        caps.families[6] = Err(Raise::foreign("device_info_test_gap"));
        assert_eq!(device_info(&caps).families[6], 0);
        // A present yes still reads 1.
        caps.families[0] = Ok(true);
        assert_eq!(device_info(&caps).families[0], 1);
    }
}
