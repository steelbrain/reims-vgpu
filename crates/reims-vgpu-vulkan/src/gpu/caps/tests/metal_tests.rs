//! The tests of `caps` that compare with Metal on this host: macOS only.

use reims_vgpu_core::platform::caps::LINEAR_ALIGNMENT_FORMAT;
use reims_vgpu_metal::MetalPlatform;

use super::*;

/// The runtime's answers are its device's, and the computed ones follow the documented facts:
/// the M3 Max under MoltenVK, Metal's argument tables, the five public primitive types;
/// `isDepth24Stencil8PixelFormatSupported` is no, as Metal answers here
#[test]
fn caps_are_the_runtimes_and_the_documented_facts() {
    let d = dev();
    let caps = d.caps();
    assert!(Arc::ptr_eq(&caps, &d.caps()), "read once, then shared");
    let props = d.raw.properties();
    assert_eq!(caps.name.as_deref(), Some(props.name.as_str()));
    assert_eq!(
        Some(caps.registry_id),
        crate::os::default_registry_id(),
        "Metal's default device's"
    );
    assert_eq!(
        caps.gpu_core_count,
        crate::os::default_gpu_core_count(),
        "Metal's default device's"
    );
    assert!(caps.gpu_core_count.is_some_and(|n| n > 0));
    assert_eq!(caps.limits.max_compute_buffers, 31);
    assert_eq!(caps.limits.max_vertex_textures, 128);
    assert_eq!(caps.limits.max_fragment_samplers, 16);
    assert_eq!(
        caps.limits.max_predicated_nesting_depth,
        u64::from(u32::MAX)
    );
    assert_eq!(caps.limits.max_rasterization_rate_layer_count, 2);
    assert_eq!(
        caps.max_vertex_amplification_count, 2,
        "the stand-in, under 8"
    );
    assert!(caps.limits.max_texture_width_2d >= 16384);
    assert_eq!(
        caps.limits.max_threadgroup_memory_length,
        props.max_threadgroup_memory_length
    );
    assert!(caps.flags.supports_tile_shaders && caps.flags.supports_image_blocks);
    assert!(
        !caps.flags.is_depth24_stencil8_pixel_format_supported,
        "an Apple GPU's no, on every device (the runtime's D663)"
    );
    if let Some(metal) = MetalPlatform::system_default_device() {
        assert!(
            !metal
                .caps()
                .flags
                .is_depth24_stencil8_pixel_format_supported
        );
    }
    assert!(
        caps.flags.supports_command_buffer_jump,
        "the adapter's predication"
    );
    let f = &caps.flags;
    assert!(
        f.supports_buffer_with_address_ranges
            && f.supports_heap_with_address_ranges
            && f.supports_resource_detach_backing,
        "MoltenVK imports host memory"
    );
    assert_eq!(caps.primitive_types[..5], [true; 5].map(Ok));
    assert_eq!(
        caps.heap_buffer_page_private,
        Ok(SizeAndAlign {
            size: 16384,
            align: 256
        })
    );
    assert_eq!(caps.min_linear_alignment_bgra8, Ok(16));
    assert_eq!(
        LINEAR_ALIGNMENT_FORMAT,
        u64::from(PixelFormat::BGRA8Unorm.as_raw())
    );
}

/// Every stand-in is the Metal platform's answer on this host, and `supportsMemoryOrderAtomics`
/// is Metal's, true, where the runtime takes the atomics on this device
/// ([`memory_order_atomics`])
#[test]
fn stand_ins_are_the_metal_platforms_here() {
    let Some(metal) = MetalPlatform::system_default_device() else {
        return;
    };
    let m = metal.caps();
    let ours = dev().caps();
    let f = &m.flags;
    assert_eq!(f.is_large_mrt_supported, ours.flags.is_large_mrt_supported);
    assert_eq!(f.is_rgb10a2_gamma_supported, stand_in::RGB10A2_GAMMA);
    assert_eq!(
        (
            f.supports_command_buffer_jump,
            m.limits.max_predicated_nesting_depth
        ),
        (
            ours.flags.supports_command_buffer_jump,
            ours.limits.max_predicated_nesting_depth
        ),
        "the adapter's predication answers Metal's"
    );
    assert_eq!(
        f.supports_compute_compressed_texture_write,
        ours.flags.supports_compute_compressed_texture_write
    );
    assert_eq!(
        f.supports_conditional_load_store,
        ours.flags.supports_conditional_load_store
    );
    assert_eq!(
        f.supports_float16_bcubic_filtering,
        stand_in::FLOAT16_BCUBIC_FILTERING
    );
    let metals = true;
    assert_eq!(
        (
            f.supports_memory_order_atomics,
            ours.flags.supports_memory_order_atomics
        ),
        (metals, metals)
    );
    assert_eq!(
        f.supports_texture_2d_multisample_array,
        stand_in::TEXTURE_2D_MULTISAMPLE_ARRAY
    );
    assert_eq!(
        m.primitive_types[5..],
        stand_in::PRIVATE_PRIMITIVE_TYPES.map(Ok)
    );
    assert_eq!(
        m.limits.max_compute_threadgroup_memory,
        ours.limits.max_compute_threadgroup_memory
    );
    assert_eq!(
        m.max_vertex_amplification_count,
        ours.max_vertex_amplification_count
    );
    assert_eq!(
        (f.supports_tile_shaders, f.supports_non_square_tile_shaders),
        (stand_in::TILE_SHADERS, stand_in::TILE_SHADERS)
    );
    assert_eq!(f.supports_image_blocks, stand_in::IMAGE_BLOCKS);
    assert_eq!(
        ours.registry_id, m.registry_id,
        "the default device's, Metal's"
    );
    assert_eq!(m.families, stand_in::FAMILIES.map(Ok));
    assert_eq!(
        f.supports_extended_xr10_formats,
        stand_in::EXTENDED_XR10_FORMATS
    );
    assert_eq!(
        ours.write_rounding_modes, m.write_rounding_modes,
        "the runtime's answer"
    );
    // The answers both platforms take from the device agree where the runtime reproduces
    // Metal's, and `gpuCoreCount` is Metal's own for this GPU.
    assert_eq!(ours.gpu_core_count, m.gpu_core_count);
    assert_eq!(
        ours.max_threads_per_threadgroup,
        m.max_threads_per_threadgroup
    );
    assert_eq!(
        ours.min_linear_alignment_bgra8,
        m.min_linear_alignment_bgra8
    );
    assert_eq!(ours.heap_buffer_page_private, m.heap_buffer_page_private);
    assert_eq!(
        ours.limits.max_compute_buffers,
        m.limits.max_compute_buffers
    );
    assert_eq!(ours.limits.max_texture_layers, m.limits.max_texture_layers);
    assert_eq!(
        (
            ours.limits.device_linear_texture_alignment_bytes,
            ours.limits.linear_texture_alignment_bytes,
            ours.limits.iosurface_texture_alignment_bytes,
            ours.limits.iosurface_read_only_texture_alignment_bytes,
        ),
        (
            m.limits.device_linear_texture_alignment_bytes,
            m.limits.linear_texture_alignment_bytes,
            m.limits.iosurface_texture_alignment_bytes,
            m.limits.iosurface_read_only_texture_alignment_bytes,
        ),
        "the runtime's alignments"
    );
    assert_eq!(
        ours.limits.max_threadgroup_memory_length,
        m.limits.max_threadgroup_memory_length
    );
}

/// Every answer of `caps`, by its selector, in the order core's `DeviceCaps` holds them.
fn answers(caps: &DeviceCaps) -> Vec<(String, String)> {
    let mut out = vec![
        ("name".to_owned(), format!("{:?}", caps.name)),
        ("registryID".to_owned(), format!("{:#x}", caps.registry_id)),
        (
            "gpuCoreCount".to_owned(),
            format!("{:?}", caps.gpu_core_count),
        ),
        (
            "maxThreadsPerThreadgroup".to_owned(),
            format!("{:?}", caps.max_threads_per_threadgroup),
        ),
        (
            "maxVertexAmplificationCount".to_owned(),
            caps.max_vertex_amplification_count.to_string(),
        ),
    ];
    // `Debug` of the structs names every field: one answer a field.
    let fields = |debug: String| {
        let body = debug
            .split_once('{')
            .and_then(|(_, rest)| rest.rsplit_once('}'))
            .map(|(body, _)| body.to_owned())
            .unwrap_or_default();
        body.split(',')
            .filter_map(|f| f.split_once(':'))
            .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
            .collect::<Vec<_>>()
    };
    out.extend(fields(format!("{:?}", caps.limits)));
    out.extend(fields(format!("{:?}", caps.flags)));
    for (count, answer) in SAMPLE_COUNTS.iter().zip(&caps.sample_counts) {
        out.push((
            format!("supportsTextureSampleCount:{count}"),
            format!("{answer:?}"),
        ));
    }
    for (t, answer) in caps.primitive_types.iter().enumerate() {
        out.push((format!("supportsPrimitiveType:{t}"), format!("{answer:?}")));
    }
    for (family, answer) in FAMILIES.iter().zip(&caps.families) {
        out.push((format!("supportsFamily:{family}"), format!("{answer:?}")));
    }
    for (mode, answer) in WRITE_ROUNDING_MODES.iter().zip(&caps.write_rounding_modes) {
        out.push((
            format!("supportsTextureWriteRoundingMode:{mode}"),
            format!("{answer:?}"),
        ));
    }
    out.push((
        "heapBufferSizeAndAlignWithLength:options:".to_owned(),
        format!("{:?}", caps.heap_buffer_page_private),
    ));
    out.push((
        "minimumLinearTextureAlignmentForPixelFormat:".to_owned(),
        format!("{:?}", caps.min_linear_alignment_bgra8),
    ));
    out
}

/// Every answer of the device's caps is the Metal platform's on this host, field for field:
/// every limit, flag, sample count, primitive type, family and rounding mode core asks, and the
/// heap-buffer and linear-alignment queries
#[test]
fn every_cap_is_the_metal_platforms_here() {
    let Some(metal) = MetalPlatform::system_default_device() else {
        return;
    };
    let (ours, theirs) = (answers(&dev().caps()), answers(&metal.caps()));
    assert_eq!(ours.len(), theirs.len());
    let differ: Vec<String> = ours
        .iter()
        .zip(&theirs)
        .filter(|(o, t)| o != t)
        .map(|((name, o), (_, t))| format!("{name}: ours {o}, Metal's {t}"))
        .collect();
    assert!(
        differ.is_empty(),
        "{} differ:\n{}",
        differ.len(),
        differ.join("\n")
    );
}
