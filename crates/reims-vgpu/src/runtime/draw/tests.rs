use super::*;
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
use crate::runtime::decode::resource::OBJECT_TYPE_TEXTURE;
use crate::runtime::gva_mem::write_task_gva_arm64e;
use crate::runtime::host::FakeHost;

fn sampled_d2_shape() -> reims_vgpu_core::SampledImageShape {
    reims_vgpu_core::sampled_image_shape(reims_vgpu_core::SampledImageKind::D2).unwrap()
}

#[test]
fn a_retained_direct_image_refusal_selects_the_copying_rail() {
    use reims_vgpu_memory::{
        GuestImageBindingDisposition as Disposition, GuestImageBindingRequirement as Requirement,
    };

    use crate::runtime::draw::sampled_source::LinearDirectAdmission as Admission;

    let admitted = sampled_source::direct_binding_requirement(
        Some(Disposition::Direct(Requirement {
            allocation_len: 0x4000,
        })),
        true,
    );
    assert_eq!(
        admitted.clone().requirement(),
        Some(Requirement {
            allocation_len: 0x4000,
        })
    );
    assert_eq!(admitted.route(), "lin_direct_admitted");

    // The three refusing outcomes all select the copying rail, and the point of
    // the enum is that they no longer say so in the same words: a backend that
    // refused a layout, a backend that answered nothing, and a bind with no
    // allocation to ask about are three different repairs.
    for (disposition, asked, expected, route) in [
        (
            Some(Disposition::Refused),
            true,
            Admission::BackendRefused,
            "lin_direct_backend_refused",
        ),
        (
            None,
            true,
            Admission::BackendSilent,
            "lin_direct_backend_silent",
        ),
        (
            None,
            false,
            Admission::NoBindingRequest,
            "lin_direct_no_binding_request",
        ),
    ] {
        let outcome = sampled_source::direct_binding_requirement(disposition, asked);
        assert_eq!(outcome, expected);
        assert_eq!(outcome.route(), route);
        assert_eq!(outcome.requirement(), None);
    }
}

/// Complete the allocation-level half of a synthetic linear texture record.
///
/// The declaration describes a view-compatible texture shape; these fields
/// independently describe how the complete mip chain is packed in its backing.
/// Keeping both halves in one fixture helper prevents a test from constructing
/// a record that the producer never emits.
fn write_linear_texture_packing(
    desc: &mut [u8],
    levels: u16,
    slices: u16,
    base_offset: u64,
    bytes_per_slice: u64,
) {
    use crate::runtime::decode::resource::{
        TEXTURE_DESC_BASE_OFFSET, TEXTURE_DESC_BYTES_PER_SLICE, TEXTURE_DESC_DECLARATION,
        TEXTURE_DESC_MIPMAP_LEVEL_COUNT, TEXTURE_DESC_MIP_LEVEL_RECORD_LEN,
        TEXTURE_DESC_SLICE_COUNT,
    };
    use reims_vgpu_core::endian::{st16, st64};

    st16(&mut desc[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], levels);
    st16(&mut desc[TEXTURE_DESC_SLICE_COUNT..], slices);
    st64(&mut desc[TEXTURE_DESC_BASE_OFFSET..], base_offset);
    st64(&mut desc[TEXTURE_DESC_BYTES_PER_SLICE..], bytes_per_slice);

    let declaration = TEXTURE_DESC_DECLARATION
        + usize::from(levels.saturating_sub(1)) * TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
    let array_length = declaration
        + core::mem::offset_of!(
            reims_vgpu_wire::ops::texture::TextureDescriptorBody,
            array_length
        );
    st16(&mut desc[array_length..], slices);
}

fn mapping_target_storage(mapping_id: u32) -> ColorTargetStorage {
    ColorTargetStorage::Mapping(mapping_id)
}

fn linear_target_storage(gva: u64, row_stride: u32, height: u32) -> ColorTargetStorage {
    ColorTargetStorage::Linear(LinearColorTarget {
        allocation_gva: gva,
        allocation_size: u64::from(row_stride) * u64::from(height),
        plane_offset: 0,
        row_stride,
    })
}

#[test]
fn m2v_draw_boundary_preserves_the_engine_vk_call_slug() {
    let req = DrawEncodeRequest {
        pipeline_ref: 73,
        ..DrawEncodeRequest::default()
    };
    let err = reims_vgpu_vulkan::engine::vk_call::exec_submit_device_lost_fixture();

    let line = linux_m2v_draw_failure(&err, &req).render();
    assert!(
        line.starts_with("linux_m2v_draw reason=vk_exec_submit vk_result="),
        "the delegated VkCall slug must be the boundary's primary reason: {line}"
    );
    assert!(line.contains(" pipe=73 task=0 geom=0x0"));
    assert!(
        !line.contains("reason=vk_engine_vk_untyped"),
        "the boundary must not flatten a typed VkCall back into DrawError prose: {line}"
    );
}

/// A presented colour attachment over `texture_ref` that clears to opaque
/// black and stores — the pass shape seven bodies need so a draw resolves, and
/// which none of them is about. The attachments that carry a real load action
/// or clear colour stay written out.
fn clear_black_attachment(texture_ref: u32) -> crate::runtime::decode::render::ColorAttachment {
    use crate::runtime::decode::render::ColorAttachment;
    ColorAttachment {
        texture_ref,
        resolve_texture_ref: 0,
        level: 0,
        slice: 0,
        depth_plane: 0,
        load_action: MTL_LOAD_ACTION_CLEAR,
        store_action: MTL_STORE_ACTION_STORE,
        clear_color: [0.0, 0.0, 0.0, 1.0],
    }
}

/// The triangle every target-resolution test in this file draws: three vertices
/// of one instance, primitive type 3, from vertex 0.
///
/// Named because it used to be spelled `3, 1, 3, 0, 0` at four call sites — five
/// positional `u32`s with two `3`s among them, where a transposition compiles and
/// draws something plausible.
fn test_triangle() -> reims_vgpu_core::draw::DrawArgs {
    reims_vgpu_core::draw::DrawArgs {
        vertex_count: 3,
        instance_count: 1,
        primitive_topology: reims_vgpu_protocol::PrimitiveTopology::Triangle,
        first_vertex: 0,
        base_instance: 0,
    }
}

/// `mrt_draw_request` for the single-RT [`test_triangle`] these tests resolve
/// targets with: attachment in slot 0, no depth. Five bodies passed those
/// trailing literals; only `pipeline_ref` and the attachment ever varied.
fn single_rt_draw_request<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    pipeline_ref: u32,
    att: crate::runtime::decode::render::ColorAttachment,
) -> Option<DrawEncodeRequest> {
    mrt_draw_request(
        state,
        host,
        1,
        pipeline_ref,
        &[(0u32, att)],
        &[],
        test_triangle(),
    )
    .ok()
    .flatten()
}

/// One reflection binding of `kind` at `metal_index` for the
/// `frag_declared_unbound` guard tests. Only kind + index are load-bearing.
fn rb(
    kind: reims_vgpu_core::ShaderResourceKind,
    metal_index: u32,
) -> reims_vgpu_core::ShaderResourceBinding {
    reims_vgpu_core::ShaderResourceBinding {
        kind,
        metal_index,
        descriptor: None,
        extent: None,
        footprint: None,
        texture_shape: None,
        access: None,
    }
}

/// The scatter band names the run count the coalescer finds, not the page count.
///
/// The reading this census exists to take is "could this window have been served
/// by N RAMBlock references instead of an allocation", and N is runs, not pages.
/// A four-page window laid out as one 16 KiB granule is one run and the answer
/// is one reference; the same four pages shuffled are four. Banding by page
/// count would report the same number for both and rank the decision wrongly.
#[test]
fn the_packed_scatter_band_counts_runs_and_not_pages() {
    use crate::runtime::bound_buffers::packed_scatter_band;

    const PAGE: u64 = 4096;
    let four_pages_one_run = [0x1000u64, 0x2000, 0x3000, 0x4000];
    let four_pages_two_runs = [0x1000u64, 0x2000, 0x9000, 0xa000];
    let four_pages_four_runs = [0x1000u64, 0x3000, 0x5000, 0x7000];

    assert_eq!(
        packed_scatter_band(&four_pages_one_run, PAGE),
        "zc_packed_scatter_runs_1"
    );
    assert_eq!(
        packed_scatter_band(&four_pages_two_runs, PAGE),
        "zc_packed_scatter_runs_2"
    );
    assert_eq!(
        packed_scatter_band(&four_pages_four_runs, PAGE),
        "zc_packed_scatter_runs_3_4"
    );

    // Every band boundary is the count the coalescer reports, so a window built
    // to have exactly N runs must land in the band that contains N.
    for runs in 1usize..=80 {
        let gpas: Vec<u64> = (0..runs).map(|i| (i as u64 + 1) * 2 * PAGE).collect();
        assert_eq!(
            reims_vgpu_paging::runs::contig_run_count(&gpas, PAGE),
            runs,
            "the fixture itself must have {runs} runs"
        );
        let band = packed_scatter_band(&gpas, PAGE);
        let expected = match runs {
            1 => "zc_packed_scatter_runs_1",
            2 => "zc_packed_scatter_runs_2",
            3..=4 => "zc_packed_scatter_runs_3_4",
            5..=8 => "zc_packed_scatter_runs_5_8",
            9..=16 => "zc_packed_scatter_runs_9_16",
            17..=64 => "zc_packed_scatter_runs_17_64",
            _ => "zc_packed_scatter_runs_65_up",
        };
        assert_eq!(band, expected, "{runs} runs");
    }
}

/// The window math the three sampled zero-copy rails now share.
///
/// Each rail used to carry its own copy, so each could have drifted alone. The
/// two refusals are the ones that matter: a stride narrower than one tight row
/// describes an image the rows do not fit in, and a stride that is not a whole
/// number of texels has no `bufferRowLength` — that field counts texels, not
/// bytes, so the copy would stride to the wrong place.
#[test]
fn strided_window_extent_measures_padded_rows_and_refuses_unrepresentable_strides() {
    // Tight rows: the extent is exactly the image, and no row length is needed.
    assert_eq!(strided_window_extent(64, 32, 4, 256), Some((256 * 32, 0)));
    // Padded rows: the extent stops after the last row's texels, because the
    // trailing padding of the final row may not be mapped.
    assert_eq!(
        strided_window_extent(64, 32, 4, 320),
        Some((320 * 31 + 256, 80))
    );
    // A single row has no padding to skip, whatever the stride says.
    assert_eq!(strided_window_extent(64, 1, 4, 320), Some((256, 80)));
    // Narrower than one tight row.
    assert_eq!(strided_window_extent(64, 32, 4, 255), None);
    // Not a whole number of texels.
    assert_eq!(strided_window_extent(64, 32, 4, 258), None);
    // Zero height has no last row to measure to.
    assert_eq!(strided_window_extent(64, 0, 4, 256), None);
    // Single-byte texels (the IOSurface plane view NV12 luma plane) take every stride.
    assert_eq!(strided_window_extent(64, 4, 1, 64), Some((256, 0)));
    assert_eq!(strided_window_extent(64, 4, 1, 96), Some((96 * 3 + 64, 96)));
}

#[test]
fn iosurface_texture_zero_copy_declines_transient_host_mappings() {
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use reims_vgpu_paging::geometry::{
        MAPPER_PAGE_ENTRY_PFN_SHIFT as PAGE_ENTRY_PFN_SHIFT,
        MAPPER_PAGE_ENTRY_VALID as PAGE_ENTRY_VALID,
    };

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let mid = 7u32;
    let width = 128u32;
    let height = 128u32;
    let page_count = 16u32;
    let base_pfn = 0x100u32;
    let page = 1u64 << PAGE_SHIFT_X86;
    for i in 0..page_count {
        host.map_range(((base_pfn + i) as u64) << PAGE_SHIFT_X86, page as usize, 0);
    }
    assert!(state.map_surface(mid));
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.pages.entries = (0..page_count)
            .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
            .collect();
    }
    assert!(state.set_mapping_geom(mid, width, height, MTL_FORMAT_BGRA8_UNORM));

    assert!(try_iosurface_texture_sample_zero_copy(
        &mut state, &mut host, mid, width, height, true
    )
    .is_none());
    assert_eq!(
        host.map_pages_calls, 0,
        "transient hosts must decline before creating an importable view"
    );
}

#[test]
fn mapping_sampled_planes_reuse_one_resource_owned_import() {
    // The guest-RAM map resolves once per process and every test in this binary
    // shares it, so a fixture that maps its own RAM has to discard whatever an
    // earlier test resolved. Without this the alias rail asks a map built from
    // some other test's host and refuses — which is the right refusal against
    // the wrong host.
    crate::runtime::guest_ram_map::reset();
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use reims_vgpu_paging::geometry::{
        MAPPER_PAGE_ENTRY_PFN_SHIFT as PAGE_ENTRY_PFN_SHIFT,
        MAPPER_PAGE_ENTRY_VALID as PAGE_ENTRY_VALID,
    };

    let page = 1u64 << PAGE_SHIFT_X86;
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.stable_map_pages = true;
    let mid = 17u32;
    let gpa0 = 0x4100_0000u64;
    let pages = 16u32;
    host.map_range(gpa0, (u64::from(pages) * page) as usize, 0);
    assert!(state.map_surface(mid));
    {
        let mapping = state.surfaces.mappings.get_mut(&mid).unwrap();
        mapping.pages.entries = (0..pages)
            .map(|i| {
                ((((gpa0 >> PAGE_SHIFT_X86) as u32) + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID
            })
            .collect();
    }
    assert!(state.set_mapping_geom(
        mid,
        128,
        128,
        reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB,
    ));
    crate::runtime::guest_ram::latch_import_limits(page, 1 << 30, 1 << 30);
    let iosurface_texture_witnesses = crate::runtime::drain::store_route_count("gw_rail_iosurface");
    let iosurface_plane_view_witnesses = crate::runtime::drain::store_route_count("gw_rail_t5");
    let iosurface_texture =
        try_iosurface_texture_sample_zero_copy(&mut state, &mut host, mid, 128, 128, true)
            .expect("the mapping's color plane is sampleable");
    let SampledSourceRequest::GuestImage(
        iosurface_texture,
        iosurface_texture_format,
        iosurface_texture_identity,
        ..,
    ) = iosurface_texture
    else {
        panic!("the mapping stays guest-backed")
    };
    assert_eq!(
        iosurface_texture_format,
        reims_vgpu_protocol::ImageFormat::srgb(reims_vgpu_protocol::TexelLayout::Bgra8).unwrap()
    );
    assert!(iosurface_texture_identity.is_some());
    assert_eq!(
        crate::runtime::drain::store_route_count("gw_rail_iosurface"),
        iosurface_texture_witnesses + 1,
        "the imported source and its copied fallback share one witness"
    );
    let iosurface_texture_import = iosurface_texture
        .direct
        .as_ref()
        .expect("mapping image has direct memory")
        .import
        .clone();
    assert!(iosurface_texture.transfer.pages.is_some());

    let iosurface_plane_view = try_iosurface_plane_view_sample_zero_copy(
        &mut state,
        &mut host,
        mid,
        objects::IOSurfacePlaneViewDescriptor {
            pixel_format: MTL_FORMAT_BGRA8_UNORM,
            width: 128,
            height: 128,
            depth: 1,
            plane_index: 0,
        },
        true,
    )
    .expect("the serialized plane view is sampleable");
    let SampledSourceRequest::GuestImage(
        iosurface_plane_view,
        iosurface_plane_view_format,
        iosurface_plane_view_identity,
        ..,
    ) = iosurface_plane_view
    else {
        panic!("the plane view stays guest-backed")
    };
    assert_eq!(
        iosurface_plane_view_format,
        reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Bgra8)
    );
    assert!(iosurface_plane_view_identity.is_some());
    assert_eq!(
        crate::runtime::drain::store_route_count("gw_rail_t5"),
        iosurface_plane_view_witnesses + 1,
        "the imported plane and its copied fallback share one witness"
    );
    let iosurface_plane_view_import = iosurface_plane_view
        .direct
        .as_ref()
        .expect("plane view has direct memory")
        .import
        .clone();
    assert!(iosurface_plane_view.transfer.pages.is_some());
    crate::runtime::guest_ram::forget_import_limits();

    assert_eq!(
        iosurface_texture_import.id(),
        iosurface_plane_view_import.id(),
        "two views of one mapping must retain the mapping's one import"
    );
    assert!(std::sync::Arc::ptr_eq(
        &iosurface_texture_import,
        &iosurface_plane_view_import
    ));
    assert_eq!(
        state.surfaces.mappings[&mid]
            .materialization
            .view()
            .and_then(|view| view.import())
            .map(|import| import.id()),
        Some(iosurface_texture_import.id())
    );
}

#[test]
fn small_mapping_sampled_plane_uses_its_imported_copy_source() {
    // The guest-RAM map resolves once per process and every test in this binary
    // shares it, so a fixture that maps its own RAM has to discard whatever an
    // earlier test resolved. Without this the alias rail asks a map built from
    // some other test's host and refuses — which is the right refusal against
    // the wrong host.
    crate::runtime::guest_ram_map::reset();
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use reims_vgpu_paging::geometry::{
        MAPPER_PAGE_ENTRY_PFN_SHIFT as PAGE_ENTRY_PFN_SHIFT,
        MAPPER_PAGE_ENTRY_VALID as PAGE_ENTRY_VALID,
    };

    let page = 1u64 << PAGE_SHIFT_X86;
    let (width, height) = (16u32, 16u32);

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.stable_map_pages = true;
    let mid = 18u32;
    let gpa = 0x4200_0000u64;
    host.map_range(gpa, page as usize, 0);
    assert!(state.map_surface(mid));
    state.surfaces.mappings.get_mut(&mid).unwrap().pages.entries =
        vec![((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT | PAGE_ENTRY_VALID];
    assert!(state.set_mapping_geom(mid, width, height, MTL_FORMAT_BGRA8_UNORM));
    crate::runtime::guest_ram::latch_import_limits(page, 1 << 30, 1 << 30);
    let sampled =
        try_iosurface_texture_sample_zero_copy(&mut state, &mut host, mid, width, height, true)
            .expect("a directly-backed sampled resource has no size crossover");
    let SampledSourceRequest::GuestImage(source, ..) = sampled else {
        panic!("the mapping stays guest-backed")
    };
    crate::runtime::guest_ram::forget_import_limits();

    assert!(
        source.transfer.pages.is_some(),
        "the transfer fallback remains GPU-addressable"
    );
    assert_eq!(
        source.direct.as_ref().expect("direct source").import.id(),
        source.transfer.pages.as_ref().unwrap()[0]
            .guest
            .import()
            .id()
    );

    let sampled =
        try_iosurface_texture_sample_zero_copy(&mut state, &mut host, mid, width, height, false)
            .expect("the same mapped bytes retain their transfer representation");
    assert!(
        matches!(sampled, SampledSourceRequest::GuestRuns(..)),
        "a non-2D descriptor must not be offered as a 2D direct image"
    );
}

#[test]
fn linear_volume_gather_carries_every_depth_plane() {
    let (width, height, depth, row_stride) = (64u32, 64u32, 5u32, 256u64);
    let layout = crate::runtime::decode::resource::TextureLevelLayout {
        offset: 0,
        size: row_stride * u64::from(height) * u64::from(depth),
        row_stride,
        width,
        height,
        depth,
    };
    let (span, row_length) = strided_level_extent(&layout, 4).unwrap();
    assert_eq!(span, row_stride * u64::from(height) * u64::from(depth));
    assert_eq!(row_length, 0);
}

#[test]
fn reflected_array_shape_uses_the_declared_array_length_and_level_pitch() {
    let level = crate::runtime::decode::resource::TextureLevelLayout {
        offset: 0,
        size: 0x1_0000,
        row_stride: 0x1_0000,
        width: 0x4000,
        height: 1,
        depth: 1,
    };
    let texture = TextureDescriptor {
        allocation_size: level.size * 3,
        declaration: Some(reims_vgpu_protocol::TextureDeclaration {
            texture_type: crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_1D_ARRAY as u8,
            framebuffer_only: false,
            is_drawable: false,
            write_swizzle_enabled: None,
            allow_gpu_optimized_contents: false,
            usage: 0,
            pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            width: level.width,
            height: level.height,
            depth: level.depth,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 3,
            resource_options: 0,
            protection_options: 0,
            swizzle: None,
        }),
        bytes_per_slice: level.size,
        slice_count: 3,
        levels: vec![level],
        ..Default::default()
    };
    let shape =
        reims_vgpu_core::sampled_image_shape(reims_vgpu_core::SampledImageKind::D1Array).unwrap();
    let image = declared_guest_image_layout(shape, &texture, &level, None).unwrap();
    assert_eq!(
        image,
        reims_vgpu_memory::GuestImageLayout::D1Array {
            width: level.width,
            layers: 3,
            array_pitch: texture.bytes_per_slice,
        }
    );
    assert_eq!(
        image.visible_span(level.row_stride, 4),
        Some(level.size * 3)
    );
}

#[test]
fn array_view_selection_moves_geometry_and_backing_offset_together() {
    use crate::runtime::decode::resource::{
        TEXTURE_VIEW_MTL_TYPE_2D, TEXTURE_VIEW_MTL_TYPE_2D_ARRAY,
    };

    let level = crate::runtime::decode::resource::TextureLevelLayout {
        offset: 0,
        size: 0x4000,
        row_stride: 0x100,
        width: 64,
        height: 64,
        depth: 1,
    };
    let texture = TextureDescriptor {
        allocation_size: 0x4000 * 4,
        declaration: Some(reims_vgpu_protocol::TextureDeclaration {
            texture_type: TEXTURE_VIEW_MTL_TYPE_2D_ARRAY as u8,
            framebuffer_only: false,
            is_drawable: false,
            write_swizzle_enabled: None,
            allow_gpu_optimized_contents: false,
            usage: 0,
            pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            width: level.width,
            height: level.height,
            depth: 1,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 4,
            resource_options: 0,
            protection_options: 0,
            swizzle: None,
        }),
        bytes_per_slice: 0x4000,
        slice_count: 4,
        levels: vec![level],
        ..Default::default()
    };
    let array_shape =
        reims_vgpu_core::sampled_image_shape(reims_vgpu_core::SampledImageKind::D2Array).unwrap();
    let range = TextureViewRange {
        level_base: 0,
        level_count: 1,
        slice_base: 1,
        slice_count: 2,
    };
    assert_eq!(
        declared_guest_image_selection(
            array_shape,
            &texture,
            &level,
            Some(TEXTURE_VIEW_MTL_TYPE_2D_ARRAY),
            Some(range),
        ),
        Some((
            reims_vgpu_memory::GuestImageLayout::D2Array {
                width: 64,
                height: 64,
                layers: 2,
                array_pitch: 0x4000,
            },
            0x4000,
        ))
    );

    let d2_shape =
        reims_vgpu_core::sampled_image_shape(reims_vgpu_core::SampledImageKind::D2).unwrap();
    assert_eq!(
        declared_guest_image_selection(
            d2_shape,
            &texture,
            &level,
            Some(TEXTURE_VIEW_MTL_TYPE_2D),
            Some(TextureViewRange {
                slice_base: 2,
                slice_count: 1,
                ..range
            }),
        ),
        Some((
            reims_vgpu_memory::GuestImageLayout::D2 {
                width: 64,
                height: 64,
            },
            0x8000,
        ))
    );

    assert!(declared_guest_image_selection(
        array_shape,
        &texture,
        &level,
        Some(TEXTURE_VIEW_MTL_TYPE_2D_ARRAY),
        Some(TextureViewRange {
            slice_base: 3,
            slice_count: 2,
            ..range
        }),
    )
    .is_none());
}

/// A cube reaches the zero-copy rail as the six-slice array it is.
///
/// Before this was expressed here the selector answered `None` for every cube,
/// the caller fell back to a single-plane span, and the backend then refused
/// the whole draw for describing one face where six were declared. The span is
/// the reading that says the layout is right: six faces at `bytes_per_slice`.
#[test]
fn a_cube_view_selects_its_six_faces_as_consecutive_array_slices() {
    use crate::runtime::decode::resource::{
        TEXTURE_VIEW_MTL_TYPE_CUBE, TEXTURE_VIEW_MTL_TYPE_CUBE_ARRAY,
    };

    let level = crate::runtime::decode::resource::TextureLevelLayout {
        offset: 0,
        size: 0x4000,
        row_stride: 0x100,
        width: 64,
        height: 64,
        depth: 1,
    };
    let cube = TextureDescriptor {
        allocation_size: 0x4000 * 6,
        declaration: Some(reims_vgpu_protocol::TextureDeclaration {
            texture_type: TEXTURE_VIEW_MTL_TYPE_CUBE as u8,
            framebuffer_only: false,
            is_drawable: false,
            write_swizzle_enabled: None,
            allow_gpu_optimized_contents: false,
            usage: 0,
            pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            width: level.width,
            height: level.height,
            depth: 1,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 1,
            resource_options: 0,
            protection_options: 0,
            swizzle: None,
        }),
        bytes_per_slice: 0x4000,
        slice_count: 1,
        cube_faces: true,
        levels: vec![level],
        ..Default::default()
    };
    // One declared slice, six physical ones. The two counts are the same fact
    // the selector has to agree with.
    assert_eq!(
        cube.physical_slice_count(),
        Some(reims_vgpu_protocol::CUBE_FACES)
    );

    let cube_shape =
        reims_vgpu_core::sampled_image_shape(reims_vgpu_core::SampledImageKind::Cube).unwrap();
    let selected = declared_guest_image_selection(
        cube_shape,
        &cube,
        &level,
        Some(TEXTURE_VIEW_MTL_TYPE_CUBE),
        None,
    );
    assert_eq!(
        selected,
        Some((
            reims_vgpu_memory::GuestImageLayout::D2Array {
                width: 64,
                height: 64,
                layers: 6,
                array_pitch: 0x4000,
            },
            0,
        ))
    );
    // The whole point of the layout: the span the caller derives from it now
    // covers all six faces rather than the first one.
    assert_eq!(
        selected.unwrap().0.visible_span(level.row_stride, 4),
        Some(level.size * 6)
    );

    // Cube-array storage under a plain-cube bind is a refusal, not the first
    // six faces of a longer array. `sampled_image_shape` refuses `CubeArray`
    // itself, so nothing downstream could tell the two apart.
    let mut cube_array = cube.clone();
    cube_array.declaration = cube_array.declaration.map(|mut d| {
        d.texture_type = TEXTURE_VIEW_MTL_TYPE_CUBE_ARRAY as u8;
        d.array_length = 2;
        d
    });
    cube_array.slice_count = 2;
    cube_array.allocation_size = 0x4000 * 12;
    assert!(declared_guest_image_selection(
        cube_shape,
        &cube_array,
        &level,
        Some(TEXTURE_VIEW_MTL_TYPE_CUBE_ARRAY),
        None,
    )
    .is_none());
}

/// The allocation selector admits a cube only when both sides say cube.
///
/// Its bail used to read `shape.cube || texture.cube_faces`, refusing a cube
/// named on either side, and admitting cubes meant dropping both terms.
/// Dropping the second is not obviously additive: it also stops refusing a
/// *non-cube* view over cube storage, a shape the six-face reasoning never
/// covered and which the `shape.cube` arm of the selection path does not reach.
/// That was recorded as the leading suspect for why admitting cubes might cost
/// more than it bought.
///
/// It is not admitted, and this test is the evidence. Both disagreeing cells
/// are still refused, by the layer-agreement checks further down rather than by
/// the bail: `storage_layers` becomes six for cube storage, so a flat bind's
/// single layer no longer matches, and a cube bind over storage declaring no
/// faces fails the same comparison from the other side.
///
/// That is worth a test even though no single line here enforces it. An
/// explicit `shape.cube != texture.cube_faces` guard was written first and then
/// removed: with it in place this test passed, and with it deleted this test
/// still passed, so it was refusing nothing and would have been a second rule
/// sitting beside the one that already works. What follows pins the behaviour
/// -- all four cells of bind-cube against storage-cube -- rather than any one
/// line's spelling, so it keeps holding whichever layer does the refusing.
#[test]
fn a_cube_allocation_is_admitted_only_when_bind_and_storage_agree() {
    use crate::runtime::decode::resource::{TEXTURE_VIEW_MTL_TYPE_2D, TEXTURE_VIEW_MTL_TYPE_CUBE};

    let level = crate::runtime::decode::resource::TextureLevelLayout {
        offset: 0,
        size: 0x4000,
        row_stride: 0x100,
        width: 64,
        height: 64,
        depth: 1,
    };
    let declaration = |texture_type: u16| reims_vgpu_protocol::TextureDeclaration {
        texture_type: texture_type as u8,
        framebuffer_only: false,
        is_drawable: false,
        write_swizzle_enabled: None,
        allow_gpu_optimized_contents: false,
        usage: 0,
        pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
        width: level.width,
        height: level.height,
        depth: 1,
        mipmap_level_count: 1,
        sample_count: 1,
        array_length: 1,
        resource_options: 0,
        protection_options: 0,
        swizzle: None,
    };
    let descriptor = |texture_type: u16, cube_faces: bool| TextureDescriptor {
        allocation_size: 0x4000 * if cube_faces { 6 } else { 1 },
        declaration: Some(declaration(texture_type)),
        bytes_per_slice: 0x4000,
        slice_count: 1,
        cube_faces,
        levels: vec![level],
        ..Default::default()
    };

    let cube_shape =
        reims_vgpu_core::sampled_image_shape(reims_vgpu_core::SampledImageKind::Cube).unwrap();
    let flat_shape =
        reims_vgpu_core::sampled_image_shape(reims_vgpu_core::SampledImageKind::D2).unwrap();

    // Agreeing, cube on both sides: the one newly-admitted cell. Six faces of
    // the ordinary slice-major packing.
    let (layout, ..) = declared_guest_image_allocation(
        cube_shape,
        &descriptor(TEXTURE_VIEW_MTL_TYPE_CUBE, true),
        Some(TEXTURE_VIEW_MTL_TYPE_CUBE),
        None,
        4,
    )
    .expect("a cube view of cube storage is the shape this rail now describes");
    assert!(
        matches!(
            layout.base().map(|base| base.layout),
            Some(reims_vgpu_memory::GuestImageLayout::D2Array { layers: 6, .. })
        ),
        "a cube's six faces are six slices, not one plane: {layout:?}",
    );

    // Agreeing, cube on neither side: admitted before the relaxation and still
    // admitted. This is what makes the change additive rather than a trade.
    assert!(
        declared_guest_image_allocation(
            flat_shape,
            &descriptor(TEXTURE_VIEW_MTL_TYPE_2D, false),
            Some(TEXTURE_VIEW_MTL_TYPE_2D),
            None,
            4,
        )
        .is_some(),
        "relaxing the cube bail must not disturb the plain 2-D case",
    );

    // Disagreeing: a non-cube view over cube storage. Refused before the
    // relaxation, and dropping `texture.cube_faces` from the bail would have
    // admitted it silently. This assertion is the reason the bail is an
    // inequality rather than two dropped terms.
    assert!(
        declared_guest_image_allocation(
            flat_shape,
            &descriptor(TEXTURE_VIEW_MTL_TYPE_CUBE, true),
            Some(TEXTURE_VIEW_MTL_TYPE_CUBE),
            None,
            4,
        )
        .is_none(),
        "a non-cube view over cube storage is not a shape the cube rule reaches",
    );

    // Disagreeing, the mirror: a cube bind whose storage declares no faces, so
    // there is nothing to divide into six.
    assert!(
        declared_guest_image_allocation(
            cube_shape,
            &descriptor(TEXTURE_VIEW_MTL_TYPE_2D, false),
            Some(TEXTURE_VIEW_MTL_TYPE_2D),
            None,
            4,
        )
        .is_none(),
        "a cube bind over storage that declares no faces has no six slices to take",
    );
}

#[test]
fn one_and_three_dimensional_views_keep_native_zero_copy_geometry() {
    let declaration = |texture_type: u8, width: u32, height: u32, depth: u32| {
        reims_vgpu_protocol::TextureDeclaration {
            texture_type,
            framebuffer_only: false,
            is_drawable: false,
            write_swizzle_enabled: None,
            allow_gpu_optimized_contents: false,
            usage: 0,
            pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            width,
            height,
            depth,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 1,
            resource_options: 0,
            protection_options: 0,
            swizzle: None,
        }
    };

    let d1_level = crate::runtime::decode::resource::TextureLevelLayout {
        size: 64,
        row_stride: 64,
        width: 16,
        height: 1,
        depth: 1,
        ..Default::default()
    };
    let d1 = TextureDescriptor {
        allocation_size: d1_level.size,
        declaration: Some(declaration(
            crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_1D as u8,
            16,
            1,
            1,
        )),
        bytes_per_slice: 0,
        slice_count: 1,
        levels: vec![d1_level],
        ..Default::default()
    };
    let d1_shape =
        reims_vgpu_core::sampled_image_shape(reims_vgpu_core::SampledImageKind::D1).unwrap();
    assert_eq!(
        declared_guest_image_layout(
            d1_shape,
            &d1,
            &d1_level,
            Some(crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_1D),
        ),
        Some(reims_vgpu_memory::GuestImageLayout::D1 { width: 16 })
    );
    let mut d1_mipped = d1.clone();
    d1_mipped.mipmap_level_count = 2;
    assert!(
        declared_guest_image_allocation(
            d1_shape,
            &d1_mipped,
            Some(crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_1D),
            None,
            4,
        )
        .is_none(),
        "an incomplete mip declaration must not become an image"
    );

    let d3_level = crate::runtime::decode::resource::TextureLevelLayout {
        size: 384,
        row_stride: 32,
        width: 8,
        height: 4,
        depth: 3,
        ..Default::default()
    };
    let d3 = TextureDescriptor {
        allocation_size: d3_level.size,
        declaration: Some(declaration(
            crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_3D as u8,
            8,
            4,
            3,
        )),
        bytes_per_slice: 0,
        slice_count: 1,
        levels: vec![d3_level],
        ..Default::default()
    };
    let d3_shape =
        reims_vgpu_core::sampled_image_shape(reims_vgpu_core::SampledImageKind::D3).unwrap();
    assert_eq!(
        declared_guest_image_layout(
            d3_shape,
            &d3,
            &d3_level,
            Some(crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_3D),
        ),
        Some(reims_vgpu_memory::GuestImageLayout::D3 {
            width: 8,
            height: 4,
            depth: 3,
            depth_pitch: 128,
        })
    );

    assert_eq!(
        declared_guest_image_layout(
            d3_shape,
            &d3,
            &d3_level,
            Some(crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_2D),
        ),
        None,
        "cross-type views need an explicit reinterpretation contract"
    );
}

#[test]
fn one_and_three_dimensional_mip_chains_preserve_every_declared_offset() {
    use crate::runtime::decode::resource::{TEXTURE_VIEW_MTL_TYPE_1D, TEXTURE_VIEW_MTL_TYPE_3D};

    let declaration = |texture_type: u8, width, height, depth, mipmap_level_count| {
        reims_vgpu_protocol::TextureDeclaration {
            texture_type,
            framebuffer_only: false,
            is_drawable: false,
            write_swizzle_enabled: None,
            allow_gpu_optimized_contents: false,
            usage: 0,
            pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            width,
            height,
            depth,
            mipmap_level_count,
            sample_count: 1,
            array_length: 1,
            resource_options: 0,
            protection_options: 0,
            swizzle: None,
        }
    };
    let d1_levels = vec![
        crate::runtime::decode::resource::TextureLevelLayout {
            offset: 0,
            size: 64,
            row_stride: 64,
            width: 16,
            height: 1,
            depth: 1,
        },
        crate::runtime::decode::resource::TextureLevelLayout {
            offset: 64,
            size: 32,
            row_stride: 32,
            width: 8,
            height: 1,
            depth: 1,
        },
    ];
    let d1 = TextureDescriptor {
        allocation_size: 0x160,
        mipmap_level_count: 2,
        base_offset: 0x100,
        bytes_per_slice: 0,
        slice_count: 1,
        declaration: Some(declaration(TEXTURE_VIEW_MTL_TYPE_1D as u8, 16, 1, 1, 2)),
        levels: d1_levels,
        ..Default::default()
    };
    let d1_shape =
        reims_vgpu_core::sampled_image_shape(reims_vgpu_core::SampledImageKind::D1).unwrap();
    let (allocation, view) =
        declared_guest_image_allocation(d1_shape, &d1, Some(TEXTURE_VIEW_MTL_TYPE_1D), None, 4)
            .expect("the complete 1D chain is representable");
    assert_eq!(view.mip_level_count, 2);
    assert_eq!(
        allocation
            .mips
            .iter()
            .map(|mip| (mip.resource_relative_offset, mip.layout.width()))
            .collect::<Vec<_>>(),
        [(0x100, 16), (0x140, 8)]
    );

    let d3_levels = vec![
        crate::runtime::decode::resource::TextureLevelLayout {
            offset: 0,
            size: 512,
            row_stride: 32,
            width: 8,
            height: 4,
            depth: 4,
        },
        crate::runtime::decode::resource::TextureLevelLayout {
            offset: 512,
            size: 64,
            row_stride: 16,
            width: 4,
            height: 2,
            depth: 2,
        },
    ];
    let d3 = TextureDescriptor {
        allocation_size: 576,
        mipmap_level_count: 2,
        bytes_per_slice: 0,
        slice_count: 1,
        declaration: Some(declaration(TEXTURE_VIEW_MTL_TYPE_3D as u8, 8, 4, 4, 2)),
        levels: d3_levels,
        ..Default::default()
    };
    let d3_shape =
        reims_vgpu_core::sampled_image_shape(reims_vgpu_core::SampledImageKind::D3).unwrap();
    let (allocation, _) =
        declared_guest_image_allocation(d3_shape, &d3, Some(TEXTURE_VIEW_MTL_TYPE_3D), None, 4)
            .expect("the complete 3D chain is representable");
    assert_eq!(
        allocation.mips[1].layout,
        reims_vgpu_memory::GuestImageLayout::D3 {
            width: 4,
            height: 2,
            depth: 2,
            depth_pitch: 32,
        }
    );
}

#[test]
fn multi_mip_requests_are_not_representable_by_the_single_level_fallback() {
    let texture = TextureDescriptor {
        mipmap_level_count: 4,
        ..Default::default()
    };
    assert_eq!(
        requested_linear_mip_range(&texture, LinearSampleSelection::default()),
        (0, 4)
    );
    assert_eq!(
        requested_linear_mip_range(
            &texture,
            LinearSampleSelection {
                level: 2,
                range: Some(TextureViewRange {
                    level_base: 2,
                    level_count: 1,
                    slice_base: 0,
                    slice_count: 1,
                }),
                ..Default::default()
            }
        ),
        (2, 1),
        "a one-level view may still use the level-local fallback"
    );
}

#[test]
fn a_three_dimensional_view_preserves_its_shape_on_the_guest_image_transfer_rail() {
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE,
        OBJECT_TYPE_TEXTURE_VIEW, TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_DECLARATION,
        TEXTURE_DESC_DEPTH, TEXTURE_DESC_HEIGHT, TEXTURE_DESC_PIXEL_FORMAT,
        TEXTURE_DESC_ROW_STRIDE, TEXTURE_DESC_USED_SIZE, TEXTURE_DESC_WIDTH,
        TEXTURE_VIEW_DESC_BASE_REF, TEXTURE_VIEW_DESC_LEN, TEXTURE_VIEW_DESC_LEVEL_BASE,
        TEXTURE_VIEW_DESC_LEVEL_COUNT, TEXTURE_VIEW_DESC_OPCODE, TEXTURE_VIEW_DESC_PIXEL_FORMAT,
        TEXTURE_VIEW_DESC_SLICE_BASE, TEXTURE_VIEW_DESC_SLICE_COUNT, TEXTURE_VIEW_DESC_TEXTURE_REF,
        TEXTURE_VIEW_DESC_TEXTURE_TYPE, TEXTURE_VIEW_MIN_RANGED, TEXTURE_VIEW_MTL_TYPE_3D,
        TEXTURE_VIEW_OPCODE_RANGED,
    };
    use reims_vgpu_core::endian::{st16, st32, st64};

    crate::runtime::guest_ram_map::reset();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    host.stable_map_pages = true;
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 16);
    assert!(state.set_object_list(1, 0, 32));
    crate::runtime::guest_ram::latch_import_limits(1 << PAGE_SHIFT_ARM64E, 1 << 30, 1 << 30);

    let (base_ref, view_ref, handle) = (5u32, 8u32, 4u32);
    let (width, height, depth, row_stride) = (8u32, 4u32, 3u32, 32u32);
    let allocation_size = 1u64 << PAGE_SHIFT_ARM64E;
    let mut base = vec![0u8; TEXTURE_DESC_BASE_LEN];
    st64(&mut base[0..], allocation_size);
    st32(&mut base[8..], handle);
    write_linear_texture_packing(
        &mut base,
        1,
        1,
        0,
        u64::from(row_stride) * u64::from(height) * u64::from(depth),
    );
    st32(
        &mut base[TEXTURE_DESC_USED_SIZE..],
        row_stride * height * depth,
    );
    st32(&mut base[TEXTURE_DESC_ROW_STRIDE..], row_stride);
    st32(&mut base[TEXTURE_DESC_WIDTH..], width);
    st32(&mut base[TEXTURE_DESC_HEIGHT..], height);
    st32(&mut base[TEXTURE_DESC_DEPTH..], depth);
    base[TEXTURE_DESC_DECLARATION] = TEXTURE_VIEW_MTL_TYPE_3D as u8;
    st16(
        &mut base[TEXTURE_DESC_PIXEL_FORMAT..],
        reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
    );
    let base_gva = 0x800;
    write_task_gva_arm64e(&mut host, &state.tasks[1], base_gva, &base);
    let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut entry[0..],
        (OBJECT_TYPE_TEXTURE as u32) | ((base.len() as u32) << 8),
    );
    entry[4..12].copy_from_slice(&base_gva.to_le_bytes());
    write_task_gva_arm64e(
        &mut host,
        &state.tasks[1],
        list_object_entry_offset(base_ref, 32).unwrap(),
        &entry,
    );

    let mut view = vec![0u8; TEXTURE_VIEW_MIN_RANGED];
    let view_len = view.len() as u32;
    st32(
        &mut view[TEXTURE_VIEW_DESC_OPCODE..],
        TEXTURE_VIEW_OPCODE_RANGED,
    );
    st32(&mut view[TEXTURE_VIEW_DESC_LEN..], view_len);
    st32(&mut view[TEXTURE_VIEW_DESC_TEXTURE_REF..], view_ref);
    st32(&mut view[TEXTURE_VIEW_DESC_BASE_REF..], base_ref);
    st16(
        &mut view[TEXTURE_VIEW_DESC_PIXEL_FORMAT..],
        reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
    );
    st16(
        &mut view[TEXTURE_VIEW_DESC_TEXTURE_TYPE..],
        TEXTURE_VIEW_MTL_TYPE_3D,
    );
    st64(&mut view[TEXTURE_VIEW_DESC_LEVEL_BASE..], 0);
    st64(&mut view[TEXTURE_VIEW_DESC_LEVEL_COUNT..], 1);
    st64(&mut view[TEXTURE_VIEW_DESC_SLICE_BASE..], 0);
    st64(&mut view[TEXTURE_VIEW_DESC_SLICE_COUNT..], 1);
    let view_gva = 0x400;
    write_task_gva_arm64e(&mut host, &state.tasks[1], view_gva, &view);
    let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut entry[0..],
        (OBJECT_TYPE_TEXTURE_VIEW as u32) | ((view.len() as u32) << 8),
    );
    entry[4..12].copy_from_slice(&view_gva.to_le_bytes());
    write_task_gva_arm64e(
        &mut host,
        &state.tasks[1],
        list_object_entry_offset(view_ref, 32).unwrap(),
        &entry,
    );

    let shape =
        reims_vgpu_core::sampled_image_shape(reims_vgpu_core::SampledImageKind::D3).unwrap();
    let (_, _, _, source) =
        resolve_sampled_source(&mut state, &mut host, 1, view_ref, None, true, shape)
            .expect("the exact D3 view resolves");
    let SampledSourceRequest::GuestImage(source, ..) = source else {
        panic!("a D3 view over an importable allocation must remain an image")
    };
    assert_eq!(
        source
            .allocation
            .base()
            .expect("one decoded base mip")
            .layout,
        reims_vgpu_memory::GuestImageLayout::D3 {
            width,
            height,
            depth,
            depth_pitch: u64::from(row_stride) * u64::from(height),
        }
    );
    assert!(
        source.direct.is_none(),
        "pre-populated guest bytes must be copied into a newly created image"
    );
    assert_eq!(source.transfer.total_len, allocation_size);

    crate::runtime::guest_ram::forget_import_limits();
    crate::runtime::guest_ram_map::reset();
}

#[test]
fn tight_linear_cpu_fallback_reads_every_depth_plane() {
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let native = [3u8, 5, 7, 255].repeat(2 * 2 * 3);
    let (bytes, format) = load_tight_linear_rgba_with(
        2,
        2,
        3,
        MTL_FORMAT_BGRA8_UNORM,
        NativeUploads::BGRA8,
        "test_volume_load",
        |dst| {
            dst.copy_from_slice(&native);
            true
        },
    )
    .unwrap();
    assert_eq!(format.layout(), TexelLayout::Bgra8);
    assert_eq!(bytes, native);
}

#[test]
fn cpu_portability_store_publishes_composite() {
    use crate::model::SurfaceWriteKind;
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use reims_vgpu_paging::geometry::{
        MAPPER_PAGE_ENTRY_PFN_SHIFT as PAGE_ENTRY_PFN_SHIFT,
        MAPPER_PAGE_ENTRY_VALID as PAGE_ENTRY_VALID,
    };

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let (mid, width, height) = (5u32, 64u32, 48u32);
    assert!(state.map_surface(mid));
    {
        let mapping = state.surfaces.mappings.get_mut(&mid).unwrap();
        mapping.lifecycle.active = true;
        mapping.publish_geometry_for_test(width, height, MTL_FORMAT_BGRA8_UNORM);
        mapping.pages.entries = vec![(1 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        mapping.content.guest_page_generation = 2;
    }
    state
        .presentation
        .present
        .establish_console(width, height, 0);
    state.presentation.present.cross_content_boundary();

    publish_surface_store(
        &mut state,
        &mut host,
        mid,
        width,
        height,
        MTL_FORMAT_BGRA8_UNORM,
    );

    assert_eq!(state.surface_write_kind(mid), SurfaceWriteKind::Composite);
    assert_eq!(state.presentation.present.early_composite_mapping(), mid);
}

#[test]
fn frag_unbound_scan_reports_only_missing_standard_kinds() {
    use crate::runtime::draw::{FragUnbound, FragUnboundClass};
    use reims_vgpu_core::ShaderResourceKind as K;
    let gap = |class, metal_index| FragUnbound { class, metal_index };
    let tex = |i| gap(FragUnboundClass::Texture, i);
    let buf = |i| gap(FragUnboundClass::Buffer, i);
    let smp = |i| gap(FragUnboundClass::Sampler, i);
    // Shader declares buffer 1+2, texture 3, sampler 0, an embedded arg-buffer
    // texture (index 9), plus other synthetic kinds (color input, threadgroup
    // buffer, storage image, constexpr sampler) that reach the shader by other
    // paths and must NOT be reported as standard-unbound.
    let bindings = [
        rb(K::Buffer, 1),
        rb(K::Buffer, 2),
        rb(K::Texture, 3),
        rb(K::Sampler, 0),
        rb(K::EmbeddedArgBufferTexture, 9),
        rb(K::ColorInput, 0),
        rb(K::ThreadgroupBuffer, 5),
        rb(K::StorageImage, 4),
        rb(K::StaticSampler, 1),
    ];
    // All standard resources bound → no gap. Unsupported synthetic resources
    // are handled by the earlier reflection-interface preflight.
    let unbound = frag_unbound_scan(
        &bindings,
        |i| [1, 2].contains(&i),
        |i| i == 3,
        |i| i == 0,
        |_| true,
    );
    assert!(unbound.is_empty());

    // Drop the texture bind → exactly tex3 reported (synthetics stay silent).
    let unbound = frag_unbound_scan(
        &bindings,
        |i| [1, 2].contains(&i),
        |_| false,
        |i| i == 0,
        |_| true,
    );
    assert_eq!(unbound, vec![tex(3)]);

    // Drop buffer 2 + sampler 0 → both reported, ordered by declaration.
    let unbound = frag_unbound_scan(&bindings, |i| i == 1, |i| i == 3, |_| false, |_| true);
    assert_eq!(unbound, vec![buf(2), smp(0)]);

    // An unprovided texture the translated module never declares is NOT a gap:
    // the reflection comes from the AIR signature, so a `[[texture(n)]]` the
    // shader never samples produces an entry for a descriptor the SPIR-V does
    // not carry. Reporting it was a false alarm on three rails.
    let unbound = frag_unbound_scan(
        &bindings,
        |i| [1, 2].contains(&i),
        |_| false,
        |i| i == 0,
        |_| false,
    );
    assert!(
        unbound.is_empty(),
        "an undeclared texture is not an unbound descriptor: {unbound:?}"
    );

    // ...but a buffer and a sampler are still reported, because the module
    // predicate is asked of textures only.
    let unbound = frag_unbound_scan(&bindings, |i| i == 1, |i| i == 3, |_| false, |_| false);
    assert_eq!(unbound, vec![buf(2), smp(0)]);

    // The class survives the scan as a type, so a consumer that needs the SPIR-V
    // binding planner does not have to parse it back out of a formatted
    // string. `Display` is the only place the prefix exists.
    assert_eq!(tex(3).to_string(), "tex3");
    assert_eq!(buf(2).to_string(), "buf2");
    assert_eq!(smp(0).to_string(), "smp0");
}

#[test]
fn reflected_static_sampler_maps_exact_state_and_rejects_unimplemented_modes() {
    use reims_vgpu_core::{
        ReflectedSamplerAddressMode as SamplerAddressMode,
        ReflectedSamplerBorderColor as SamplerBorderColor,
        ReflectedSamplerCompareFunction as SamplerCompareFunction,
        ReflectedSamplerCoordinates as SamplerCoordinates, ReflectedSamplerFilter as SamplerFilter,
        ReflectedSamplerMipFilter as SamplerMipFilter,
        ReflectedSamplerReduction as SamplerReduction,
        ReflectedStaticSamplerState as StaticSamplerState, SamplerAddressMode as EngineAddress,
        SamplerFilter as EngineFilter, SamplerMipFilter as EngineMip,
    };

    let mut state = StaticSamplerState {
        min_filter: SamplerFilter::Linear,
        mag_filter: SamplerFilter::Linear,
        mip_filter: SamplerMipFilter::None,
        address_mode_s: SamplerAddressMode::ClampToEdge,
        address_mode_t: SamplerAddressMode::ClampToEdge,
        address_mode_r: SamplerAddressMode::ClampToEdge,
        coordinates: SamplerCoordinates::Normalized,
        compare_function: SamplerCompareFunction::Never,
        max_anisotropy: 1,
        lod_min_clamp: 0.0,
        lod_max_clamp: 65504.0,
        border_color: SamplerBorderColor::TransparentBlack,
        reduction: SamplerReduction::WeightedAverage,
        lod_bias: 0.0,
        raw_words: [0x807b_ff00_0008_0a49, 0],
    };
    let mapped =
        reflected_static_sampler_resource("fragment", 65, state).expect("supported sampler");
    assert_eq!(mapped.binding, 65);
    assert_eq!(mapped.min_filter, EngineFilter::Linear);
    assert_eq!(mapped.mag_filter, EngineFilter::Linear);
    assert_eq!(mapped.mip_filter, EngineMip::NotMipmapped);
    assert_eq!(mapped.address_mode_u, EngineAddress::ClampToEdge);
    assert_eq!(mapped.address_mode_v, EngineAddress::ClampToEdge);
    assert_eq!(mapped.address_mode_w, EngineAddress::ClampToEdge);
    assert_eq!(mapped.lod_min_f32(), 0.0);
    assert_eq!(mapped.lod_max_f32(), 65504.0);
    assert!(!mapped.unnormalized_coordinates);

    state.min_filter = SamplerFilter::Nearest;
    state.mag_filter = SamplerFilter::Nearest;
    state.address_mode_s = SamplerAddressMode::Repeat;
    state.address_mode_t = SamplerAddressMode::Repeat;
    state.address_mode_r = SamplerAddressMode::Repeat;
    let repeat = reflected_static_sampler_resource("fragment", 66, state).expect("repeat sampler");
    assert_eq!(repeat.min_filter, EngineFilter::Nearest);
    assert_eq!(repeat.address_mode_u, EngineAddress::Repeat);

    state.min_filter = SamplerFilter::Bicubic;
    assert_eq!(
        reflected_static_sampler_resource("fragment", 66, state)
            .unwrap_err()
            .slug(),
        "draw_prepare_static_sampler_min_filter_unsupported"
    );
    state.min_filter = SamplerFilter::Nearest;
    state.reduction = SamplerReduction::Minimum;
    assert_eq!(
        reflected_static_sampler_resource("fragment", 66, state)
            .unwrap_err()
            .slug(),
        "draw_prepare_static_sampler_reduction_unsupported"
    );
    state.reduction = SamplerReduction::WeightedAverage;
    state.lod_bias = 1.0;
    assert_eq!(
        reflected_static_sampler_resource("fragment", 66, state)
            .unwrap_err()
            .slug(),
        "draw_prepare_static_sampler_lod_bias_unsupported"
    );
}

#[test]
fn depth_stencil_triviality_matches_no_op_state() {
    use crate::runtime::decode::resource::{DepthStencilDescriptor, DepthStencilFace};
    // compare Always (7), no write, no stencil → equivalent to no depth test.
    let trivial = DepthStencilDescriptor {
        depth_compare_function: 7,
        depth_write_enabled: false,
        front_stencil_present: false,
        back_stencil_present: false,
        ..Default::default()
    };
    assert!(depth_stencil_descriptor_is_trivial(&trivial));

    // Metal's default face objects are present but do no stencil work. Native
    // state creation reports neither reads nor writes for this pair.
    let default_face = DepthStencilFace {
        compare_function: 7,
        read_mask: u32::MAX,
        write_mask: u32::MAX,
        ..Default::default()
    };
    let present_defaults = DepthStencilDescriptor {
        front_stencil_present: true,
        back_stencil_present: true,
        front_face: default_face,
        back_face: default_face,
        ..trivial.clone()
    };
    assert!(depth_stencil_descriptor_is_trivial(&present_defaults));

    // A real compare function (Less=1) occludes → non-trivial.
    assert!(!depth_stencil_descriptor_is_trivial(
        &DepthStencilDescriptor {
            depth_compare_function: 1,
            ..trivial.clone()
        }
    ));

    // A nontrivial masked comparison reads stencil. A zero read mask makes the
    // same comparison inert.
    let mut compares = present_defaults.clone();
    compares.front_face.compare_function = 1;
    assert!(!depth_stencil_descriptor_is_trivial(&compares));
    compares.front_face.read_mask = 0;
    assert!(depth_stencil_descriptor_is_trivial(&compares));

    // Any non-Keep operation writes when the write mask is nonzero. Masking the
    // write out makes it inert, irrespective of which outcome owns the op.
    let mut writes = present_defaults.clone();
    writes.front_face.depth_stencil_pass_operation = 2;
    assert!(!depth_stencil_descriptor_is_trivial(&writes));
    writes.front_face.write_mask = 0;
    assert!(depth_stencil_descriptor_is_trivial(&writes));

    // An absent face's body is not semantic input.
    writes.front_stencil_present = false;
    writes.front_face.write_mask = u32::MAX;
    assert!(depth_stencil_descriptor_is_trivial(&writes));
    // Depth write on → non-trivial even with compare Always.
    assert!(!depth_stencil_descriptor_is_trivial(
        &DepthStencilDescriptor {
            depth_write_enabled: true,
            ..trivial.clone()
        }
    ));
}

#[test]
fn invalid_fixed_function_pipeline_state_refuses_semantic_preparation() {
    use crate::runtime::decode::resource::{
        DepthStencilDescriptor, PipelineColorAttachment, RenderPipelineDescriptor,
    };
    use reims_vgpu_protocol::PipelineStateDecodeError;

    let pipeline = RenderPipelineDescriptor {
        color_attachments: vec![PipelineColorAttachment {
            slot: 0,
            blending_enabled: true,
            src_rgb: 99,
            ..PipelineColorAttachment::default()
        }],
        ..RenderPipelineDescriptor::default()
    };
    assert!(matches!(
        semantic_blend_states(&pipeline),
        Err(DrawPreparationDecline::BlendState {
            reason: PipelineStateDecodeError::BlendFactor(99)
        })
    ));

    let request = DrawEncodeRequest::default();
    let invalid_depth = DepthStencilDescriptor {
        depth_compare_function: 99,
        depth_write_enabled: true,
        ..DepthStencilDescriptor::default()
    };
    assert!(matches!(
        semantic_depth_state(&invalid_depth, &request),
        Err(DrawPreparationDecline::DepthCompare {
            reason: PipelineStateDecodeError::CompareFunction(99)
        })
    ));

    let mut invalid_stencil = DepthStencilDescriptor {
        depth_compare_function: 7,
        front_stencil_present: true,
        ..DepthStencilDescriptor::default()
    };
    invalid_stencil.front_face.stencil_failure_operation = 99;
    assert!(matches!(
        semantic_depth_state(&invalid_stencil, &request),
        Err(DrawPreparationDecline::StencilState {
            face: "front",
            reason: PipelineStateDecodeError::StencilOperation(99)
        })
    ));
}

#[test]
fn bound_depth_stencil_that_cannot_resolve_returns_named_reason() {
    // A guest that binds a depth-stencil ref (`ds_ref != 0`) whose object-list
    // entry does not resolve must surface a *specific* reason, not `None`: the
    // draw silently disables the depth test otherwise, and every other
    // depth/stencil error on this path is already fail-visible. With an empty
    // state the lookup misses at the entry-missing rung; execution maps that
    // rung to `DrawPreparationDecline::DepthStencilStateMissing` and refuses.
    let state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let host = FakeHost::new();
    let err = load_depth_stencil_descriptor(&state, &host, /*task*/ 4, /*ds_ref*/ 9)
        .expect_err("unresolvable bound depth-stencil must report a reason");
    assert_eq!(
        err,
        crate::observe::ladder_slug!("depth_stencil", no_list_entry)
    );
}

/// A draw that cannot load its pipeline says why, and an unbound ref says
/// nothing.
///
/// `load_render_pipeline` used to return a bare `None` from all five of its
/// failure points while its compute sibling named all of its own, so every
/// caller's coarse `MissingPipeline` was the whole story on the rail that runs
/// per frame. The silent half is the one worth a test: `pipeline_ref == 0` is
/// "no pipeline bound", and ref 0 is a *valid* object-list index, so without the
/// guard an unbound ref would read entry 0 and report a rung for it.
#[test]
fn a_pipeline_that_cannot_load_names_the_rung_and_an_unbound_ref_stays_quiet() {
    let state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let host = FakeHost::new();

    let cap = crate::observe::FailCapture::start();
    assert!(load_render_pipeline(&state, &host, /*task*/ 4, /*pipe*/ 9).is_none());
    assert_eq!(
        cap.one("draw_load_pipeline"),
        "draw_load_pipeline fail reason=no_list_entry task=4 pipe_ref=9"
    );
    drop(cap);

    let cap = crate::observe::FailCapture::start();
    assert!(load_render_pipeline(&state, &host, 4, 0).is_none());
    assert!(
        cap.lines().is_empty(),
        "an unbound pipeline ref must spend no line: {:?}",
        cap.lines()
    );
}

#[test]
fn index_load_failures_report_the_specific_reason() {
    // The Vulkan indexed-draw path collapsed eleven distinct load failures into
    // one `index_buffer_miss`; each now names the failing check so a boot log
    // says *which* one fired.
    let state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let host = FakeHost::new();

    // Unsupported MTLIndexType (only 0=u16 / 1=u32 exist).
    let bad_type = IndexedDrawInfo {
        base_vertex: 0,
        index_type: reims_vgpu_protocol::decode_index_type(5),
        index_count: 3,
        index_buffer_ref: 9,
        index_buffer_offset: 0,
        index_start: 0,
    };
    assert_eq!(
        load_index_bytes_reason(&state, &host, 4, &bad_type),
        Err(IndexLoadReason::TypeUnsupported)
    );

    // Valid type + count, but the bound index buffer ref resolves to nothing on
    // an empty state → the entry-missing site, not a generic miss.
    let unresolved = IndexedDrawInfo {
        base_vertex: 0,
        index_type: reims_vgpu_protocol::decode_index_type(1),
        index_count: 6,
        index_buffer_ref: 9,
        index_buffer_offset: 0,
        index_start: 0,
    };
    assert_eq!(
        load_index_bytes_reason(&state, &host, 4, &unresolved),
        Err(IndexLoadReason::EntryMissing)
    );
}

#[test]
fn indirect_index_start_is_scaled_only_after_the_type_decodes() {
    let info = |raw, index_start| IndexedDrawInfo {
        base_vertex: 0,
        index_type: reims_vgpu_protocol::decode_index_type(raw),
        index_count: 1,
        index_buffer_ref: 1,
        index_buffer_offset: 0x100,
        index_start,
    };

    assert_eq!(info(0, 3).resolved_byte_offset(), Ok(0x106));
    assert_eq!(info(1, 3).resolved_byte_offset(), Ok(0x10c));
    assert_eq!(
        info(7, 3).resolved_byte_offset(),
        Err(IndexLoadReason::TypeUnsupported)
    );
}

/// Eleven checks, eleven names, one namespace.
///
/// What this asserts is the
/// *prefix*, because bare names (`out_of_bounds`, `read_fail`) would match
/// three other rails on a `grep reason=` and the reader could not tell an
/// index buffer from a blit row.
#[test]
fn every_index_load_failure_has_its_own_namespaced_name() {
    use crate::observe::Decline as _;
    const ALL: &[IndexLoadReason] = &[
        IndexLoadReason::TypeUnsupported,
        IndexLoadReason::CountOverflow,
        IndexLoadReason::CountZero,
        IndexLoadReason::EntryMissing,
        IndexLoadReason::ObjectType,
        IndexLoadReason::DescRead,
        IndexLoadReason::DescDecode,
        IndexLoadReason::BackingMissing,
        IndexLoadReason::OffsetOverflow,
        IndexLoadReason::OutOfBounds,
        IndexLoadReason::ReadFail,
        IndexLoadReason::BaseVertexOutOfRange,
    ];
    let mut slugs: Vec<&str> = ALL.iter().map(|r| r.slug()).collect();
    for s in &slugs {
        assert!(
            s.starts_with("draw_index_"),
            "{s} is not namespaced to the indexed-draw path"
        );
    }
    let n = slugs.len();
    slugs.sort_unstable();
    slugs.dedup();
    assert_eq!(slugs.len(), n, "two index-load checks answer with one name");
}

/// The status carries the check *and* the class, and cannot render a line for
/// a success.
///
/// The class is not derivable from the slug and the caller acts on it —
/// `BackendUnavailable` makes the exec loop honour the pass clear, `WritebackFailed` does
/// not — so a reader correlating a dropped draw with a black frame needs both
/// on the line.
#[test]
fn encode_status_renders_its_check_beside_the_class_it_collapsed_to() {
    use crate::observe::{Emit, Refusal as _};
    assert_eq!(
        Emit::refusal(
            "draw_encode_fail",
            &EncodeStatus::MissingMtlb("draw_vertex_mtlb_load")
        )
        .expect("a refusal must render a line")
        .render(),
        "draw_encode_fail reason=draw_vertex_mtlb_load class=missing_mtlb"
    );
    assert_eq!(
        Emit::refusal(
            "render_icb",
            &EncodeStatus::WritebackFailed("icb_exec_writeback_none")
        )
        .expect("a refusal must render a line")
        .render(),
        "render_icb reason=icb_exec_writeback_none class=writeback_failed"
    );
    // I2's carve-out, enforced by the type: there is no line to send for a
    // success, so no call site can log one by forgetting a guard.
    assert!(
        Emit::refusal("draw_encode_fail", &EncodeStatus::Ok).is_none(),
        "Ok is control flow and must not be representable as a line"
    );
    assert_eq!(EncodeStatus::Ok.refusal(), None);
    assert_eq!(EncodeStatus::Ok.class(), "ok");
}

/// A vertex reflection that trips the shader-pull coverage gate: writes
/// Position, reads VertexIndex, binds a Buffer at each of `bindings`.
fn shader_pull_reflection(bindings: &[u32]) -> reims_vgpu_core::ShaderInterface {
    use reims_vgpu_core::{
        ReflectedShaderStage, ShaderBufferExtent, ShaderDescriptorLocation, ShaderInterface,
        ShaderResourceBinding, ShaderResourceKind,
    };
    ShaderInterface {
        stage: ReflectedShaderStage::Vertex,
        bindings: bindings
            .iter()
            .map(|&binding| ShaderResourceBinding {
                kind: ShaderResourceKind::Buffer,
                metal_index: binding,
                descriptor: Some(ShaderDescriptorLocation {
                    set: 0,
                    binding,
                    count: 1,
                }),
                // What the translator emits for a buffer carrying neither an
                // object size nor a type name: the class that forbids narrowing.
                extent: Some(ShaderBufferExtent::Unknown),
                footprint: None,
                texture_shape: None,
                access: None,
            })
            .collect(),
        local_size: None,
        unsupported: None,
    }
}

/// Regression: WebKit's glyph vertex shader declares stride-48 stage-in on
/// buffer 1 but never reads it as an attribute — it indexes the same buffer
/// as a per-glyph `StorageBuffer` (binding 1) by gl_InstanceIndex. Skipping
/// every stage-in buffer left binding 1 unbound, so glyphs collapsed to
/// zero-area quads (blank Safari body text). A stage-in buffer whose binding
/// the vertex SPIR-V declares as a StorageBuffer must still be bound.
/// `swap_rb_channels` must be byte-identical to the `src.to_vec()` +
/// in-place `chunks_exact_mut(4)` swizzle it replaces — including the tail
/// (a non-multiple-of-4 remainder copied through unchanged) — and be its own
/// inverse (BGRA<->RGBA).
#[test]
fn swap_rb_channels_matches_two_pass_and_preserves_tail() {
    fn two_pass(src: &[u8]) -> Vec<u8> {
        let mut v = src.to_vec();
        for px in v.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
        v
    }
    for len in [0usize, 4, 8, 5, 7, 9, 260, 263] {
        let src: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
        let got = swap_rb_channels(&src);
        assert_eq!(
            got,
            two_pass(&src),
            "len={len} must match the two-pass idiom"
        );
        assert_eq!(got.len(), src.len(), "len={len} length preserved");
        // Round-trip: swapping twice restores the original bytes exactly.
        assert_eq!(
            swap_rb_channels(&got),
            src,
            "len={len} swap is its own inverse"
        );
    }
}

/// `reorder_rb_in_place` must touch nothing when the order it holds is already
/// the order asked for, and match `swap_rb_channels` when it is not.
///
/// The no-op half is the whole point of threading the order rather than
/// normalizing: an IOSurface texture composite Store's readback now arrives BGRA, so this is
/// the call that used to be a 776 us whole-frame pass and is now a compare. A
/// future edit that made it exchange unconditionally would restore that cost
/// silently — the pixels would still be right.
#[test]
fn reorder_rb_in_place_is_a_no_op_when_the_orders_already_agree() {
    for len in [0usize, 4, 8, 5, 260, 263] {
        let src: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
        for order in [false, true] {
            let mut same = src.clone();
            crate::runtime::draw::reorder_rb_in_place(&mut same, order, order);
            assert_eq!(
                same, src,
                "len={len} order={order}: agreement must not copy"
            );
        }
        // Disagreement in either direction is exactly the established swizzle,
        // tail included.
        let mut to_bgra = src.clone();
        crate::runtime::draw::reorder_rb_in_place(&mut to_bgra, false, true);
        assert_eq!(to_bgra, swap_rb_channels(&src), "len={len} rgba->bgra");
        let mut to_rgba = src.clone();
        crate::runtime::draw::reorder_rb_in_place(&mut to_rgba, true, false);
        assert_eq!(to_rgba, swap_rb_channels(&src), "len={len} bgra->rgba");
    }
}

/// Vulkan-arm only: a stage-in buffer can also be a direct shader buffer.
#[test]
fn stage_in_buffer_read_as_ssbo_is_bound_as_storage() {
    let reflection = shader_pull_reflection(&[1]);
    // Buffer 1 is stage-in AND read as SSBO -> must be exposed as storage.
    assert!(vertex_buffer_needs_storage_binding(&reflection, 1, true));
    // A plain non-stage-in buffer is always storage.
    assert!(vertex_buffer_needs_storage_binding(&reflection, 2, false));
    // A stage-in buffer the shader does NOT read as an SSBO stays stage-in only.
    assert!(!vertex_buffer_needs_storage_binding(&reflection, 3, true));
}

/// Resident GVA chain wiring: the identity is built only for GVA color0
/// (never IOSurface texture), and its extent is color0's declared geometry — the one
/// place a draw states what it renders into.
#[test]
fn gva_chain_identity_rules() {
    use crate::model::TargetIdentity;
    let executor = crate::runtime::executor::VulkanExecutor::default();
    let mut req = DrawEncodeRequest::default();
    assert_eq!(
        gva_chain_identity(&executor, &req, 0),
        None,
        "no colors → no identity"
    );
    req.colors.push(ColorRtRequest {
        slot: 0,
        texture_ref: 9,
        storage: linear_target_storage(0x1234_0000, 16 * 4, 16),
        width: 16,
        height: 16,
        ..Default::default()
    });
    assert_eq!(
        gva_chain_identity(&executor, &req, 0),
        Some(TargetIdentity::Gva {
            gva: 0x1234_0000,
            width: 16,
            height: 16,
            generation: 0,
            format: reims_vgpu_core::pixel_format::TexelLayout::Rgba8,
        }),
        "color0 declares the extent"
    );
    req.colors[0].width = 0;
    assert_eq!(
        gva_chain_identity(&executor, &req, 0),
        None,
        "a zero-extent attachment has no identity"
    );
    req.colors[0].width = 16;
    req.colors[0].storage = mapping_target_storage(5);
    assert_eq!(
        gva_chain_identity(&executor, &req, 0),
        None,
        "IOSurface texture targets never take the GVA identity"
    );
    req.colors[0].storage = ColorTargetStorage::None;
    assert_eq!(
        gva_chain_identity(&executor, &req, 0),
        None,
        "gva=0 → no identity"
    );
}

#[test]
fn render_chain_identity_covers_iosurface_texture_and_gva_targets() {
    use crate::model::TargetIdentity;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    assert!(state.map_surface(5));
    let mut req = DrawEncodeRequest::default();
    req.colors.push(ColorRtRequest {
        slot: 0,
        texture_ref: 9,
        storage: mapping_target_storage(5),
        width: 64,
        height: 32,
        ..Default::default()
    });
    assert!(matches!(
        render_chain_identity(&state, &req, 0),
        Some(TargetIdentity::Surface {
            id: 5,
            width: 64,
            height: 32,
            ..
        })
    ));

    req.colors[0].storage = linear_target_storage(0x1234_0000, 64 * 4, 32);
    assert_eq!(
        render_chain_identity(&state, &req, 0),
        Some(TargetIdentity::Gva {
            gva: 0x1234_0000,
            width: 64,
            height: 32,
            generation: 0,
            format: reims_vgpu_core::pixel_format::TexelLayout::Rgba8,
        })
    );
}

/// The last record of a resident render-pass chain is both the chain's consumer
/// and the packet's guest-visible Store, and it must name the resident it loads
/// from so it can skip its own readback.
///
/// Refusing `chain_from_resident` here cost the entire remaining composite
/// readback population — `iosurface_keep_chain_from_resident` measured equal to
/// `surface_deferred` in every window of one boot. The assertion that matters is
/// the *equality*: `retarget_render_pass_draw` builds every record of a packet
/// from one attachment template, so the record that loads from the resident is by
/// construction the record that renders into it, and a Store naming a different
/// slot than its own LOAD would pin an image its frame is not in.
#[test]
fn a_chained_composite_store_names_the_resident_it_loads_from() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    assert!(state.map_surface(7));
    let mut req = DrawEncodeRequest::default();
    req.colors.push(ColorRtRequest {
        slot: 0,
        texture_ref: 3,
        storage: mapping_target_storage(7),
        width: 128,
        height: 64,
        load_action: reims_vgpu_protocol::pass_action::LoadAction::Load,
        store_action: reims_vgpu_protocol::pass_action::StoreAction::Store,
        ..Default::default()
    });

    let unchained = iosurface_texture_store_identity(&state, &req, true);
    assert!(
        unchained.is_some(),
        "an unchained composite Store resolves its resident"
    );

    req.chain_from_resident = true;
    assert_eq!(
        iosurface_texture_store_identity(&state, &req, true),
        unchained,
        "a chained Store must name the same resident an unchained one does — it is \
         the same attachment template, so the chain cannot move the slot"
    );
    assert_eq!(
        iosurface_texture_store_identity(&state, &req, true),
        render_chain_identity(&state, &req, 0),
        "the Store identity and the LoadFromTarget identity must be one slot"
    );

    // The gates that are still refusals. `writeback_guest` is the one that
    // separates the packet's last record from its intermediates, and an
    // intermediate has no guest Store to defer.
    assert_eq!(
        iosurface_texture_store_identity(&state, &req, false),
        None,
        "an intermediate record stores nothing guest-visible"
    );
    req.colors[0].store_action = reims_vgpu_protocol::pass_action::StoreAction::DontCare;
    assert_eq!(
        iosurface_texture_store_identity(&state, &req, true),
        None,
        "a record that discards its target has no frame to defer"
    );
    req.colors[0].store_action = reims_vgpu_protocol::pass_action::StoreAction::Store;
    req.colors[0].storage = ColorTargetStorage::None;
    assert_eq!(
        iosurface_texture_store_identity(&state, &req, true),
        None,
        "a GVA target is the other rail's; this one requires a mapping"
    );
}

/// An intermediate record renders into the surface resident too, so it must be
/// able to ask whether that image is already current — even though it has no
/// guest Store of its own to defer.
///
/// Keying the LOAD's currency check on the *Store* identity broke this, and the
/// cost was not a lost elision but a loop. Record 1 of a chain has
/// `writeback_guest == false`, so the check never ran; its LOAD fell through to a
/// CPU seed; the seed found the host cache ceded to the resident rail and read the
/// mapping's guest pages; and reading them landed the window the rail had just
/// armed, which advanced the epoch and cost the *next* LOAD its elision too. One
/// boot measured `surface_flush / surface_resident` at 1369/1373 — one flush per
/// arm.
#[test]
fn an_intermediate_record_can_still_ask_about_the_resident_it_renders_into() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    assert!(state.map_surface(7));
    let mut req = DrawEncodeRequest::default();
    req.colors.push(ColorRtRequest {
        slot: 0,
        texture_ref: 3,
        storage: mapping_target_storage(7),
        width: 128,
        height: 64,
        load_action: reims_vgpu_protocol::pass_action::LoadAction::Load,
        store_action: reims_vgpu_protocol::pass_action::StoreAction::Store,
        ..Default::default()
    });

    // The query the LOAD actually asks. It takes no `writeback_guest`, so an
    // intermediate and a final record get the same answer by construction — which
    // is the property, and it is structural rather than asserted.
    let (identity, mapping_epoch) = iosurface_texture_load_currency_query(&state, &req).expect(
        "a LOAD into a mapped IOSurface texture surface is a candidate the resident could serve",
    );
    assert_eq!(
        Some(identity.clone()),
        render_chain_identity(&state, &req, 0),
        "the LOAD must ask about the slot the record actually renders into"
    );
    assert_eq!(
        Some(identity),
        iosurface_texture_store_identity(&state, &req, true),
        "the Store identity is the same slot, restricted — not a different one"
    );
    assert_eq!(
        mapping_epoch,
        Some(0),
        "a freshly mapped surface has published nothing, and 0 is that value — the \
         `is_some` guard in `iosurface_texture_resident_is_current` is what keeps it from \
         matching an unstamped slot"
    );
    assert_eq!(
        iosurface_texture_store_identity(&state, &req, false),
        None,
        "…while only the packet's last record may leave its frame on the resident"
    );

    // The refusals. A LOAD the resident cannot serve must not produce a query at
    // all, or the counters below it would divide all draws instead of candidates.
    req.colors[0].load_action = reims_vgpu_protocol::pass_action::LoadAction::Clear;
    assert!(
        iosurface_texture_load_currency_query(&state, &req).is_none(),
        "a CLEAR has no prior content to be current"
    );
    req.colors[0].load_action = reims_vgpu_protocol::pass_action::LoadAction::Load;
    req.colors[0].target_seed_rgba = Some(vec![0u8; 128 * 64 * 4]);
    assert!(
        iosurface_texture_load_currency_query(&state, &req).is_none(),
        "an explicit seed was already selected by RT provenance"
    );
    req.colors[0].target_seed_rgba = None;
    req.colors[0].store_action = reims_vgpu_protocol::pass_action::StoreAction::DontCare;
    assert!(
        iosurface_texture_load_currency_query(&state, &req).is_none(),
        "a record that discards its target renders into no resident worth naming"
    );
    req.colors[0].store_action = reims_vgpu_protocol::pass_action::StoreAction::Store;
    req.colors[0].storage = ColorTargetStorage::None;
    assert!(
        iosurface_texture_load_currency_query(&state, &req).is_none(),
        "a GVA target is the other rail's"
    );
}

/// Attachment aliasing follows the serialized texture reference and load
/// action, independent of where its initial contents happen to reside.
#[test]
fn attachment_alias_selection_uses_only_the_texture_contract() {
    use reims_vgpu_core::AttachmentInitial;

    let mut req = DrawEncodeRequest::default();
    req.colors.push(ColorRtRequest {
        slot: 0,
        texture_ref: 42,
        storage: linear_target_storage(0x9000, 8 * 4, 8),
        width: 8,
        height: 8,
        load_action: reims_vgpu_protocol::pass_action::LoadAction::Load,
        ..Default::default()
    });
    assert_eq!(
        fragment_attachment_alias_initial(&req, 0, 42),
        Some((8, 8, AttachmentInitial::Seed)),
        "LOAD names the attachment whether or not a CPU seed exists"
    );
    req.chain_from_resident = true;
    assert_eq!(
        fragment_attachment_alias_initial(&req, 0, 42),
        Some((8, 8, AttachmentInitial::Seed)),
        "chain position does not change the texture identity"
    );
    req.colors[0].target_seed_rgba = Some(vec![0u8; 8 * 8 * 4]);
    assert_eq!(
        fragment_attachment_alias_initial(&req, 0, 42),
        Some((8, 8, AttachmentInitial::Seed)),
        "CPU seed availability does not change the texture identity"
    );
    req.colors[0].storage = mapping_target_storage(9);
    assert_eq!(
        fragment_attachment_alias_initial(&req, 0, 42),
        Some((8, 8, AttachmentInitial::Seed)),
        "backing kind does not change the texture identity"
    );
}

/// Vulkan-path only: the attachment alias resolver belongs to draw preparation.
#[test]
fn attachment_alias_preserves_each_declared_load_action() {
    use reims_vgpu_core::AttachmentInitial;

    let task_id = std::process::id();
    let texture_ref = 0xe000_0000u32.wrapping_add(task_id);
    let target_gva = 0x0abc_d000;
    let seed = vec![10, 0, 0, 255, 0, 0, 0, 0];
    let mut req = DrawEncodeRequest {
        task_id,
        colors: vec![ColorRtRequest {
            slot: 0,
            texture_ref,
            storage: linear_target_storage(target_gva, 2 * 4, 1),
            width: 2,
            height: 1,
            load_action: reims_vgpu_protocol::pass_action::LoadAction::Load,
            target_seed_rgba: Some(seed.clone()),
            ..Default::default()
        }],
        ..Default::default()
    };

    let (width, height, initial) =
        fragment_attachment_alias_initial(&req, 0, texture_ref).expect("attachment alias");
    assert_eq!((width, height), (2, 1));
    assert_eq!(initial, AttachmentInitial::Seed);
    assert!(fragment_attachment_alias_initial(&req, 1, texture_ref).is_none());
    assert!(fragment_attachment_alias_initial(&req, 0, texture_ref + 1).is_none());

    req.colors[0].storage = mapping_target_storage(9);
    assert_eq!(
        fragment_attachment_alias_initial(&req, 0, texture_ref),
        Some((2, 1, AttachmentInitial::Seed))
    );
    req.colors[0].load_action = reims_vgpu_protocol::pass_action::LoadAction::DontCare;
    assert_eq!(
        fragment_attachment_alias_initial(&req, 0, texture_ref),
        Some((2, 1, AttachmentInitial::DontCare))
    );
    req.colors[0].load_action = reims_vgpu_protocol::pass_action::LoadAction::Clear;
    req.colors[0].clear_color = [0.25, 0.5, 0.75, 1.0];
    assert_eq!(
        fragment_attachment_alias_initial(&req, 0, texture_ref),
        Some((2, 1, AttachmentInitial::Clear([0.25, 0.5, 0.75, 1.0]),))
    );
}

#[test]
fn attachment_alias_source_keeps_one_texture_identity() {
    use reims_vgpu_core::AttachmentInitial;

    let identity = crate::model::TargetIdentity::Gva {
        gva: 0x4000,
        width: 2,
        height: 1,
        generation: 7,
        format: reims_vgpu_core::pixel_format::TexelLayout::Rgba8,
    };
    let format = reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Rgba8);
    let clear = attachment_alias_source(
        identity.clone(),
        format,
        AttachmentInitial::Clear([0.25, 0.5, 0.75, 1.0]),
    );
    assert!(matches!(
        clear,
        SampledSourceRequest::Attachment(
            ref held,
            AttachmentInitial::Clear([0.25, 0.5, 0.75, 1.0]),
            held_format,
        ) if held == &identity && held_format == format
    ));

    assert!(matches!(
        attachment_alias_source(
            identity.clone(),
            format,
            AttachmentInitial::Seed,
        ),
        SampledSourceRequest::Attachment(
            ref held,
            AttachmentInitial::Seed,
            held_format,
        ) if held == &identity && held_format == format
    ));
    assert!(matches!(
        attachment_alias_source(
            identity.clone(),
            format,
            AttachmentInitial::DontCare,
        ),
        SampledSourceRequest::Attachment(
            ref held,
            AttachmentInitial::DontCare,
            held_format,
        ) if held == &identity && held_format == format
    ));
    let load = attachment_alias_source(identity.clone(), format, AttachmentInitial::Seed);
    let SampledSourceRequest::Attachment(held, AttachmentInitial::Seed, held_format) = load else {
        panic!("LOAD must keep the attachment source");
    };
    assert_eq!(held, identity);
    assert_eq!(held_format, format);
}

#[test]
fn tight_linear_load_uses_one_bulk_read_and_converts_rows() {
    let mut calls = 0;
    let (rgba, fmt) = load_tight_linear_rgba_with(
        2,
        2,
        1,
        MTL_FORMAT_BGRA8_UNORM,
        NativeUploads::NONE,
        "test_tight_load",
        |native| {
            calls += 1;
            assert_eq!(native.len(), 16);
            native.copy_from_slice(&[3, 2, 1, 255, 6, 5, 4, 255, 9, 8, 7, 255, 12, 11, 10, 255]);
            true
        },
    )
    .expect("tight sample loads");

    assert_eq!(calls, 1);
    assert_eq!(fmt.layout(), TexelLayout::Rgba8);
    assert_eq!(
        rgba,
        [1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255, 10, 11, 12, 255,]
    );
}

/// A native BGRA8 upload keeps the guest bytes verbatim (no CPU channel
/// swap) and reports `Bgra8` so the engine binds a BGRA8 image — the
/// Safari-scroll fallback hot path. Same read count as the swizzled path.
#[test]
fn tight_linear_native_bgra8_keeps_bytes_and_reports_bgra8() {
    let bgra = [3, 2, 1, 255, 6, 5, 4, 255, 9, 8, 7, 255, 12, 11, 10, 255];
    let mut calls = 0;
    let (bytes, fmt) = load_tight_linear_rgba_with(
        2,
        2,
        1,
        MTL_FORMAT_BGRA8_UNORM,
        NativeUploads::BGRA8,
        "test_tight_load",
        |native| {
            calls += 1;
            native.copy_from_slice(&bgra);
            true
        },
    )
    .expect("tight native sample loads");
    assert_eq!(calls, 1);
    assert_eq!(fmt.layout(), TexelLayout::Bgra8);
    assert_eq!(bytes, bgra, "native BGRA8 upload must not swizzle");
}

/// **The half-float regression gate.** A `RGBA16Float` sampled texture reaches
/// the GPU as the guest's own bytes, in an `R16G16B16A16_SFLOAT` image.
///
/// Without the native arm this same call returns four bytes a texel through
/// `f16_to_unorm8_lut`, which is what the second half asserts: the value `2.0`
/// — ordinary for an extended-range compositor, and the whole reason the guest
/// chose a float format — comes back as `255`, indistinguishable from `1.0`.
/// Two half-floats above the unit interval that differ by 100 % arrive equal.
#[test]
fn a_half_float_sampled_texture_keeps_its_bytes_when_the_caller_takes_native_layouts() {
    // Two texels: (1.0, 0.5, 0.0, 1.0) and (2.0, 1.0, 0.0, 1.0), IEEE binary16
    // little-endian, which is byte-for-byte what `R16G16B16A16_SFLOAT` samples.
    let guest: [u8; 16] = [
        0x00, 0x3c, 0x00, 0x38, 0x00, 0x00, 0x00, 0x3c, // 1.0, 0.5, 0.0, 1.0
        0x00, 0x40, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x3c, // 2.0, 1.0, 0.0, 1.0
    ];
    let (bytes, fmt) = load_tight_linear_rgba_with(
        2,
        1,
        1,
        pixel_format::MTL_FORMAT_RGBA16_FLOAT,
        NativeUploads::ALL,
        "test_tight_load",
        |dst| {
            assert_eq!(dst.len(), 16, "eight bytes a texel, not four");
            dst.copy_from_slice(&guest);
            true
        },
    )
    .expect("half-float sample loads");
    assert_eq!(fmt.layout(), TexelLayout::Rgba16Float);
    assert_eq!(bytes, guest, "a half-float upload must not convert");

    // The same source through the lossy arm, so the gate states what it is
    // guarding against rather than only what it wants.
    let (narrowed, narrowed_fmt) = load_tight_linear_rgba_with(
        2,
        1,
        1,
        pixel_format::MTL_FORMAT_RGBA16_FLOAT,
        NativeUploads::NONE,
        "test_tight_load",
        |dst| {
            dst.copy_from_slice(&guest);
            true
        },
    )
    .expect("half-float sample loads through the convert arm too");
    assert_eq!(narrowed_fmt.layout(), TexelLayout::Rgba8);
    assert_eq!(narrowed.len(), 8, "converted to four bytes a texel");
    assert_eq!(
        narrowed[0], narrowed[4],
        "1.0 and 2.0 both clamp to 255 — this is the loss the native arm avoids"
    );
}

/// The two-channel companion takes the same rail at its own four-byte width,
/// which is the width the RGBA8 output happens to share — so this is the case
/// that would still pass if the gate were re-narrowed to `native_len ==
/// rgba_len`, and it is here to pin the *layout*, not the length.
#[test]
fn a_two_channel_half_float_sampled_texture_reports_rg16_float() {
    let guest: [u8; 8] = [0x00, 0x3c, 0x00, 0x38, 0x00, 0x40, 0x00, 0x3c];
    let (bytes, fmt) = load_tight_linear_rgba_with(
        2,
        1,
        1,
        pixel_format::MTL_FORMAT_RG16_FLOAT,
        NativeUploads::ALL,
        "test_tight_load",
        |dst| {
            dst.copy_from_slice(&guest);
            true
        },
    )
    .expect("two-channel half-float sample loads");
    assert_eq!(fmt.layout(), TexelLayout::Rg16Float);
    assert_eq!(bytes, guest);
}

/// A padded source stride takes the straight row copy at the layout's own texel
/// width. The row loop used to size its output from `RGBA8_BPP`, which for an
/// eight-byte texel is half the rows' length — so this is the case that caught
/// the allocation and not just the format.
#[test]
fn a_padded_half_float_row_copies_straight_through_at_eight_bytes_a_texel() {
    assert_eq!(
        linear_native_upload_format(pixel_format::MTL_FORMAT_RGBA16_FLOAT, NativeUploads::ALL),
        Some(TexelLayout::Rgba16Float)
    );
    assert_eq!(
        linear_native_upload_format(pixel_format::MTL_FORMAT_RGBA16_FLOAT, NativeUploads::BGRA8),
        None,
        "a caller that did not opt in must still get the RGBA8 convert"
    );
    assert_eq!(
        TexelLayout::Rgba16Float.bytes_per_texel(),
        8,
        "the row copy sizes its output from this"
    );
}

#[test]
fn tight_rgba_linear_load_preserves_native_bytes() {
    let native = [1, 2, 3, 4, 5, 6, 7, 8];
    let (rgba, fmt) = load_tight_linear_rgba_with(
        2,
        1,
        1,
        pixel_format::MTL_FORMAT_RGBA8_UNORM,
        NativeUploads::NONE,
        "test_tight_load",
        |dst| {
            dst.copy_from_slice(&native);
            true
        },
    )
    .expect("tight RGBA sample loads");
    assert_eq!(fmt.layout(), TexelLayout::Rgba8);
    assert_eq!(rgba, native);
}

/// **The sRGB regression gate for the CPU upload rails.** These paths reach a
/// linear byte *layout* from an sRGB Metal format, which is the right layout —
/// the two share one — and for a long time that was all they carried, so every
/// CPU-uploaded sRGB texture was bound through a `_UNORM` view and never
/// decoded, while the zero-copy rails beside them bound `_SRGB` and were.
///
/// Both halves are asserted here because either alone is a bug: the layout must
/// stay identical to the linear sibling's, *and* the transfer function must
/// survive to the Vulkan format the bind uses.
/// A mapping's declared format is a **colour-space** answer and must never be
/// able to fail a bind.
///
/// This is the shape the first draft of `mapping_declared_format` got wrong: it
/// ran the answer through `effective_view_sample_format` and inherited that
/// function's `Option`, so a mapping declaring a format outside the
/// bytes-per-pixel table would have dropped the sampled bind — losing guest work
/// over a question that has a correct answer for every `u16`.
#[test]
fn a_mappings_declared_format_answers_for_every_mapping() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    // A mapping this device holds no entry for has declared nothing, and the
    // store rule's own "nothing declared" answer is what comes back.
    assert_eq!(
        mapping_declared_format(&state, 7, None),
        MTL_FORMAT_BGRA8_UNORM
    );
    // One that declares a format outside every table this crate carries still
    // answers, and answers "not sRGB".
    state.surfaces.mappings.insert(
        7,
        crate::model::SurfaceMappingEntry {
            lifecycle: crate::model::SurfaceMappingLifecycle {
                active: true,
                ..Default::default()
            },
            ..Default::default()
        }
        .with_geometry_for_test(4, 4, 0xfffe),
    );
    assert_eq!(mapping_declared_format(&state, 7, None), 0xfffe);
    assert!(!pixel_format::is_srgb(mapping_declared_format(
        &state, 7, None
    )));
    // A declared sRGB surface reaches the bind as sRGB.
    state
        .surfaces
        .mappings
        .get_mut(&7)
        .expect("just inserted")
        .publish_geometry_for_test(4, 4, pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB);
    assert!(pixel_format::is_srgb(mapping_declared_format(
        &state, 7, None
    )));
    // A type-8 view's format is what the guest says it is reading, so it wins
    // over the mapping's own — including when it takes the sRGB back off.
    assert_eq!(
        mapping_declared_format(&state, 7, Some(MTL_FORMAT_BGRA8_UNORM)),
        MTL_FORMAT_BGRA8_UNORM
    );
}

#[test]
fn the_cpu_upload_rails_carry_the_srgb_transfer_function_to_the_bind() {
    use reims_vgpu_vulkan::translate::pixel::{vk_sampled_bytes, vk_texel_layout};

    // Native-upload rail: an sRGB format resolves to exactly its linear
    // sibling's layout. That fold is correct and is not what was lost.
    assert_eq!(
        linear_native_upload_format(
            pixel_format::MTL_FORMAT_RGBA8_UNORM_SRGB,
            NativeUploads::NONE
        ),
        linear_native_upload_format(pixel_format::MTL_FORMAT_RGBA8_UNORM, NativeUploads::NONE),
    );
    assert_eq!(
        linear_native_upload_format(
            pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB,
            NativeUploads::BGRA8
        ),
        Some(TexelLayout::Bgra8),
    );

    // Tight-load rail, converting arm: the BGRA swap still happens when the
    // caller did not opt into a native BGRA8 upload, and the qualifier rides
    // through the channel exchange — a swap moves no value across the transfer
    // function.
    let native = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let (bytes, fmt) = load_tight_linear_rgba_with(
        2,
        1,
        1,
        pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB,
        NativeUploads::NONE,
        "test_tight_load",
        |dst| {
            dst.copy_from_slice(&native);
            true
        },
    )
    .expect("tight sRGB BGRA sample loads");
    assert_eq!(fmt.layout(), TexelLayout::Rgba8, "layout is the sibling's");
    assert_eq!(
        bytes,
        [3, 2, 1, 4, 7, 6, 5, 8],
        "channel swap still applied"
    );
    assert_eq!(
        fmt.srgb_source(),
        Some(pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB),
        "the source's transfer function survives the reorder"
    );
    // The whole point: the bind decodes. Bound as RGBA because that is the
    // order the swap produced, and sRGB because that is what the guest stored.
    assert_eq!(
        vk_sampled_bytes(fmt),
        reims_vgpu_vulkan::format::vk_image_format(
            reims_vgpu_protocol::ImageFormat::srgb(TexelLayout::Rgba8).unwrap()
        ),
        "the CPU rung must bind the same colour space the zero-copy rail does"
    );

    // Tight-load rail, native arm: no conversion at all, qualifier still there.
    let (_, native_fmt) = load_tight_linear_rgba_with(
        2,
        1,
        1,
        pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB,
        NativeUploads::BGRA8,
        "test_tight_load",
        |dst| {
            dst.copy_from_slice(&native);
            true
        },
    )
    .expect("native sRGB BGRA sample loads");
    assert_eq!(native_fmt.layout(), TexelLayout::Bgra8);
    assert_eq!(
        vk_sampled_bytes(native_fmt),
        reims_vgpu_vulkan::format::vk_image_format(
            reims_vgpu_protocol::ImageFormat::srgb(TexelLayout::Bgra8).unwrap()
        )
    );

    // A linear source must reach the linear spelling, or every bind decodes
    // twice and the fix is worse than the bug it replaced.
    let (_, linear_fmt) = load_tight_linear_rgba_with(
        2,
        1,
        1,
        pixel_format::MTL_FORMAT_BGRA8_UNORM,
        NativeUploads::BGRA8,
        "test_tight_load",
        |dst| {
            dst.copy_from_slice(&native);
            true
        },
    )
    .expect("native linear BGRA sample loads");
    assert_eq!(linear_fmt.srgb_source(), None);
    assert_eq!(
        vk_sampled_bytes(linear_fmt),
        vk_texel_layout(TexelLayout::Bgra8)
    );
}

#[test]
fn color_target_diag_names_every_mrt_slot() {
    let colors = vec![
        ColorRtRequest {
            slot: 0,
            texture_ref: 11,
            storage: mapping_target_storage(1),
            width: 1920,
            height: 1080,
            format: MTL_FORMAT_BGRA8_UNORM,
            load_action: reims_vgpu_protocol::pass_action::LoadAction::Load,
            store_action: reims_vgpu_protocol::pass_action::StoreAction::Store,
            ..Default::default()
        },
        ColorRtRequest {
            slot: 2,
            texture_ref: 17,
            storage: linear_target_storage(0x1234_5000, 960 * 8, 540),
            width: 960,
            height: 540,
            format: pixel_format::MTL_FORMAT_RGBA16_FLOAT,
            load_action: reims_vgpu_protocol::pass_action::LoadAction::Clear,
            store_action: reims_vgpu_protocol::pass_action::StoreAction::Store,
            ..Default::default()
        },
    ];
    assert_eq!(
        color_target_diag(&colors),
        "s0:r11:mid1:gva=0x0:1920x1080:fmt=0x50:l1:s1,\
s2:r17:mid0:gva=0x12345000:960x540:fmt=0x73:l2:s1"
    );
}

#[test]
fn missing_pipeline_is_soft() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.define_task(1, 0x1000, 2);
    let mut host = FakeHost::new();
    let req = DrawEncodeRequest {
        task_id: 1,
        pipeline_ref: 99,
        vertex_count: 3,
        instance_count: 1,
        primitive_topology: reims_vgpu_protocol::PrimitiveTopology::Triangle,
        first_vertex: 0,
        colors: vec![ColorRtRequest {
            slot: 0,
            texture_ref: 1,
            storage: mapping_target_storage(3),
            width: 4,
            height: 4,
            format: MTL_FORMAT_BGRA8_UNORM,
            store_action: reims_vgpu_protocol::pass_action::StoreAction::Store,
            ..Default::default()
        }],
        ..Default::default()
    };
    // Spelled here rather than behind a wrapper, because the two arms' wrappers
    // used to pass *different* `force_full_store` values for the same call.
    // These are the arguments `exec` passes for a single-record draw that owns
    // its writeback.
    let st = encode_draw_chain(&mut state, &mut host, &req, true, false).status;
    assert!(matches!(
        st,
        EncodeStatus::MissingPipeline(_)
            | EncodeStatus::MissingMtlb(_)
            | EncodeStatus::BackendUnavailable(_)
    ));
    let _ = pixel_format::RGBA8_BPP;
}

/// The three bind bounds, each against the thing that sets it.
///
/// They were one constant, and it was Metal's *buffer* table — so the texture
/// bound was a buffer fact and the sampler bound was a buffer fact, and neither
/// could move without moving the other two. Asserting the values alone would
/// re-freeze exactly that; what has to hold is that each equals its own basis,
/// so the test reads each from where it comes from.
#[test]
fn each_bind_slot_bound_equals_its_own_basis() {
    use reims_vgpu_vulkan::spirv_bind::{
        COLOR_INPUT_BINDING_BASE, SAMPLER_BINDING_BASE, TEXTURE_BINDING_BASE,
    };

    // Buffers: an argument table, and the one class where Apple's serializer
    // and Metal's encoder name the same number.
    assert_eq!(
        MAX_BUFFER_BIND_SLOTS,
        reims_vgpu_wire::ops::bind_limit::BUFFER
    );

    // Textures and samplers: band widths in the flat descriptor binding space,
    // not tables. A texture at the bound would carry sampler 0's binding.
    assert_eq!(
        MAX_TEXTURE_BIND_SLOTS,
        SAMPLER_BINDING_BASE - TEXTURE_BINDING_BASE
    );
    assert_eq!(
        MAX_SAMPLER_BIND_SLOTS,
        COLOR_INPUT_BINDING_BASE - SAMPLER_BINDING_BASE
    );

    // The texture bound is *not* the buffer table, which is what it used to be.
    assert_ne!(MAX_TEXTURE_BIND_SLOTS, MAX_BUFFER_BIND_SLOTS);

    // No product byte-size budget: host_alloc_len only rejects >usize/isize.
    assert_eq!(host_alloc_len(64 << 20), Some(64 << 20));
    assert_eq!(host_alloc_len(0), Some(0));
}

/// Each class's bound reaches every consumer through one accessor.
///
/// The value is not the assertion — [`each_bind_slot_bound_equals_its_own_basis`]
/// owns that. What this holds is that [`BindTableClass::table`] answers with the
/// class's *own* constant, so a consumer asking the accessor cannot silently get
/// another class's table the way twenty-two hand-written comparisons could.
#[test]
fn the_bind_table_accessor_answers_with_each_class_own_bound() {
    assert_eq!(BindTableClass::Buffer.table(), MAX_BUFFER_BIND_SLOTS);
    assert_eq!(BindTableClass::Texture.table(), MAX_TEXTURE_BIND_SLOTS);
    assert_eq!(BindTableClass::Sampler.table(), MAX_SAMPLER_BIND_SLOTS);
    // The three are distinct numbers, which is the whole reason one shared
    // constant was wrong for two of them.
    assert_ne!(
        BindTableClass::Buffer.table(),
        BindTableClass::Texture.table()
    );
    assert_ne!(
        BindTableClass::Texture.table(),
        BindTableClass::Sampler.table()
    );
}

/// A live bind past its class's table is reported, naming the class and stage;
/// an in-range bind and a cleared slot are not.
///
/// The cleared-slot case is the one worth pinning: a zero ref past the table
/// loses no guest work, and reporting it would turn expected control flow into a
/// refused draw.
#[test]
fn a_live_bind_past_its_table_is_reported_and_a_cleared_one_is_not() {
    use crate::runtime::decode::render::Stage;

    let in_range = DrawEncodeRequest {
        vertex_buffers: vec![BufferBind {
            index: MAX_BUFFER_BIND_SLOTS - 1,
            buffer_ref: 7,
            offset: 0,
            attribute_stride: None,
            ..Default::default()
        }]
        .into(),
        fragment_textures: vec![TextureBind {
            index: MAX_TEXTURE_BIND_SLOTS - 1,
            texture_ref: 9,
            ..Default::default()
        }]
        .into(),
        vertex_samplers: vec![SamplerBind {
            index: MAX_SAMPLER_BIND_SLOTS - 1,
            sampler_ref: 11,
            lod_clamp: None,
        }]
        .into(),
        ..Default::default()
    };
    assert_eq!(first_bind_past_table(&in_range), None);

    // One slot further in each class, still live: each is reported, and the
    // report names the class whose table it crossed rather than a shared one.
    for (req, class, stage, index, resource_ref) in [
        (
            DrawEncodeRequest {
                fragment_buffers: vec![BufferBind {
                    index: MAX_BUFFER_BIND_SLOTS,
                    buffer_ref: 7,
                    offset: 0,
                    attribute_stride: None,
                    ..Default::default()
                }]
                .into(),
                ..Default::default()
            },
            BindTableClass::Buffer,
            Stage::Fragment,
            MAX_BUFFER_BIND_SLOTS,
            7,
        ),
        (
            DrawEncodeRequest {
                vertex_textures: vec![TextureBind {
                    index: MAX_TEXTURE_BIND_SLOTS,
                    texture_ref: 9,
                    ..Default::default()
                }]
                .into(),
                ..Default::default()
            },
            BindTableClass::Texture,
            Stage::Vertex,
            MAX_TEXTURE_BIND_SLOTS,
            9,
        ),
        (
            DrawEncodeRequest {
                fragment_samplers: vec![SamplerBind {
                    index: MAX_SAMPLER_BIND_SLOTS,
                    sampler_ref: 11,
                    lod_clamp: None,
                }]
                .into(),
                ..Default::default()
            },
            BindTableClass::Sampler,
            Stage::Fragment,
            MAX_SAMPLER_BIND_SLOTS,
            11,
        ),
    ] {
        assert_eq!(
            first_bind_past_table(&req),
            Some(PastTableBind {
                class,
                stage: match stage {
                    Stage::Vertex => reims_vgpu_core::ShaderStage::Vertex,
                    Stage::Fragment => reims_vgpu_core::ShaderStage::Fragment,
                    Stage::Unknown => reims_vgpu_core::ShaderStage::Unknown,
                },
                index,
                resource_ref,
            }),
            "{} slot {index} past a {}-entry table",
            class.name(),
            class.table()
        );
    }

    // A cleared slot, at an index no table can name, in every class.
    let cleared = DrawEncodeRequest {
        vertex_buffers: vec![BufferBind {
            index: MAX_BUFFER_BIND_SLOTS + 4,
            buffer_ref: 0,
            offset: 0,
            attribute_stride: None,
            ..Default::default()
        }]
        .into(),
        fragment_textures: vec![TextureBind {
            index: MAX_TEXTURE_BIND_SLOTS + 4,
            texture_ref: 0,
            ..Default::default()
        }]
        .into(),
        vertex_samplers: vec![SamplerBind {
            index: MAX_SAMPLER_BIND_SLOTS + 4,
            sampler_ref: 0,
            lod_clamp: None,
        }]
        .into(),
        ..Default::default()
    };
    assert_eq!(first_bind_past_table(&cleared), None);
}

#[test]
fn vulkan_sampler_missing_entry_returns_exact_decline() {
    let state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let host = FakeHost::new();
    let error = load_vulkan_sampler(&state, &host, 7, 11, 64)
        .expect_err("an empty object list cannot resolve sampler 11");
    assert_eq!(
        error.slug(),
        crate::observe::ladder_slug!("draw_prepare_sampler", no_list_entry)
    );
    assert_eq!(
        error.fields(),
        vec![("sampler_ref", "11".into()), ("binding", "64".into()),]
    );
}

#[test]
fn vertex_attribute_preparation_returns_exact_declines() {
    use crate::runtime::decode::resource::VertexAttribute;

    let mut attribute = VertexAttribute {
        location: 3,
        format: 99,
        buffer_index: 2,
        stride: 16,
        ..VertexAttribute::default()
    };
    let format = prepare_vertex_attribute_format(&attribute)
        .expect_err("unknown MTLVertexFormat must be typed before request validation");
    assert_eq!(format.slug(), "draw_prepare_vertex_attribute_format");
    assert_eq!(
        format.fields(),
        vec![
            ("location", "3".into()),
            ("buffer_index", "2".into()),
            ("raw_format", "99".into()),
            ("value", "99".into()),
        ]
    );

    attribute.format = 30;
    attribute.declared_step_function = Some(9);
    let step = prepare_vertex_step_function(&attribute)
        .expect_err("unknown MTLVertexStepFunction must be typed before request validation");
    assert_eq!(step.slug(), "draw_prepare_vertex_step_function_unsupported");
    assert_eq!(
        step.fields(),
        vec![
            ("location", "3".into()),
            ("buffer_index", "2".into()),
            ("value", "9".into()),
        ]
    );

    // A tessellation step rate must not render as an unrecognised value. Both
    // reach the same `DrawPreparationDecline` variant, so for a long time both
    // reached the same slug too: that variant returned a fixed string where its
    // `TranslateReason`-carrying siblings delegate, and the split
    // `translate::reason` introduced never got as far as the log. The `value`
    // field was the only thing telling 3 apart from 9, which is exactly what the
    // second slug exists to stop a reader having to do.
    //
    // This is the assertion that would have caught it, and it is deliberately a
    // comparison of the two rather than a check of one: a fixed string passes
    // any single-slug assertion.
    for mtl in [3u32, 4] {
        attribute.declared_step_function = Some(mtl);
        let patch = prepare_vertex_step_function(&attribute)
            .expect_err("a per-patch step rate has no VkVertexInputRate");
        assert_eq!(
            patch.slug(),
            "draw_prepare_vertex_step_function_per_patch",
            "MTLVertexStepFunction {mtl} is a declared SDK value this backend \
             recognises and cannot spell in Vulkan, not a value it failed to \
             recognise"
        );
        assert_ne!(
            patch.slug(),
            step.slug(),
            "a tessellation step rate and a corrupt ordinal must not share a \
             slug; a driven boot's log cannot tell them apart if they do"
        );
        assert_eq!(
            patch.fields(),
            vec![
                ("location", "3".into()),
                ("buffer_index", "2".into()),
                ("value", mtl.to_string()),
            ]
        );
    }
}

#[test]
fn vulkan_sampler_preserves_guest_coordinate_and_filter_state() {
    use crate::runtime::decode::resource::SamplerDescriptor;
    use reims_vgpu_core::{
        SamplerAddressMode, SamplerBorderColor, SamplerCompareFunction, SamplerFilter,
        SamplerMipFilter,
    };

    let decoded = SamplerDescriptor {
        min_filter: 0,
        mag_filter: 1,
        mip_filter: 2,
        s_address: 2,
        t_address: 3,
        r_address: 5,
        max_anisotropy: 4,
        lod_min_clamp: 1.25,
        lod_max_clamp: 7.5,
        compare_function: 3,
        border_color: 2,
        normalized_coordinates: false,
        support_argument_buffers: false,
        lod_average: false,
    };
    let sampler = vulkan_sampler_resource(5, 67, &decoded).expect("supported sampler");

    assert_eq!(sampler.binding, 67);
    assert_eq!(sampler.min_filter, SamplerFilter::Nearest);
    assert_eq!(sampler.mag_filter, SamplerFilter::Linear);
    assert_eq!(sampler.mip_filter, SamplerMipFilter::Linear);
    assert_eq!(sampler.address_mode_u, SamplerAddressMode::Repeat);
    assert_eq!(sampler.address_mode_v, SamplerAddressMode::MirrorRepeat);
    assert_eq!(
        sampler.address_mode_w,
        SamplerAddressMode::ClampToBorderColor
    );
    assert_eq!(sampler.border_color, SamplerBorderColor::OpaqueWhite);
    assert_eq!(sampler.compare_function, SamplerCompareFunction::LessEqual);
    assert_eq!(sampler.lod_min, 1.25f32.to_bits());
    assert_eq!(sampler.lod_max, 7.5f32.to_bits());
    assert_eq!(sampler.max_anisotropy, 4);
    assert!(sampler.unnormalized_coordinates);

    let mut bad = decoded;
    bad.min_filter = 9;
    let min = vulkan_sampler_resource(5, 67, &bad).expect_err("unknown min filter");
    assert_eq!(min.slug(), "draw_prepare_sampler_min_filter_translation");
    assert_eq!(
        min.fields(),
        vec![
            ("sampler_ref", "5".into()),
            ("binding", "67".into()),
            ("value", "9".into()),
        ]
    );

    bad.min_filter = 0;
    bad.mag_filter = 9;
    let mag = vulkan_sampler_resource(5, 67, &bad).expect_err("unknown mag filter");
    assert_eq!(mag.slug(), "draw_prepare_sampler_mag_filter_translation");
}

/// qemu-shim Store policy: Clear/DontCare/force_full full-write; Load+seed
/// may diff-only. Prevents Clear+partial logo-mid residual.
#[test]
fn store_seed_policy_clear_full_load_diff() {
    let seed = [1u8, 2, 3, 4];
    assert!(store_seed_policy(false, MTL_LOAD_ACTION_CLEAR, Some(&seed)).is_none());
    assert!(store_seed_policy(false, MTL_LOAD_ACTION_DONT_CARE, Some(&seed)).is_none());
    assert!(store_seed_policy(true, MTL_LOAD_ACTION_LOAD, Some(&seed)).is_none());
    assert_eq!(
        store_seed_policy(false, MTL_LOAD_ACTION_LOAD, Some(&seed)),
        Some(seed.as_slice())
    );
    assert!(store_seed_policy(false, MTL_LOAD_ACTION_LOAD, None).is_none());
}

/// Premult One/OneMinusSrcAlpha Load: transparent draw keeps seed; opaque black wins.
#[test]
fn load_composite_premult_restores_seed_under_transparent() {
    let mut draw = vec![0u8; 8];
    draw[0..4].copy_from_slice(&[255, 255, 255, 255]); // chrome
    draw[4..8].copy_from_slice(&[0, 0, 0, 0]); // uncovered (clear A=0)
    let mut seed = vec![0u8; 8];
    seed[0..4].copy_from_slice(&[203, 203, 203, 255]);
    seed[4..8].copy_from_slice(&[203, 203, 203, 255]);
    let (out, blended) = load_composite_premult_one_omsa(&draw, &seed);
    assert_eq!(blended, 1);
    assert_eq!(&out[0..4], &[255, 255, 255, 255]);
    assert_eq!(&out[4..8], &[203, 203, 203, 255], "A=0 keeps Load seed");
    draw[4..8].copy_from_slice(&[0, 0, 0, 255]);
    let (out2, _) = load_composite_premult_one_omsa(&draw, &seed);
    assert_eq!(&out2[4..8], &[0, 0, 0, 255], "opaque black stays black");
}

#[test]
fn a8_sample_preserves_alpha_coverage() {
    let native = [0, 17, 255];
    let (rgba, fmt) = load_tight_linear_rgba_with(
        3,
        1,
        1,
        pixel_format::MTL_FORMAT_A8_UNORM,
        NativeUploads::BGRA8,
        "test_tight_load",
        |dst| {
            dst.copy_from_slice(&native);
            true
        },
    )
    .expect("A8 sample loads");
    assert_eq!(
        fmt.layout(),
        TexelLayout::Rgba8,
        "A8 needs a real convert; native flag does not apply"
    );
    assert_eq!(
        rgba,
        [0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 255],
        "A8 has no RGB channels; its alpha is the sampled mask payload"
    );
}

/// Metal blend factors/ops must map into engine blend types (Linux path was silent-None).
#[test]
fn blend_state_maps_src_alpha_one_minus() {
    let b = reims_vgpu_protocol::blend_state(
        &crate::runtime::decode::resource::PipelineColorAttachment {
            src_rgb: 4,   // SrcAlpha
            dst_rgb: 5,   // OneMinusSrcAlpha
            op_rgb: 0,    // Add
            src_alpha: 1, // One
            dst_alpha: 5, // OneMinusSrcAlpha
            op_alpha: 0,  // Add
            ..Default::default()
        },
    )
    .expect("map");
    assert_eq!(b.src_color, reims_vgpu_core::BlendFactor::SrcAlpha);
    assert_eq!(b.dst_color, reims_vgpu_core::BlendFactor::OneMinusSrcAlpha);
    assert_eq!(b.color_op, reims_vgpu_core::BlendOp::Add);
    assert_eq!(b.src_alpha, reims_vgpu_core::BlendFactor::One);
    assert!(reims_vgpu_protocol::blend_factor(99).is_err());
    assert!(reims_vgpu_protocol::blend_operation(9).is_err());
}

/// qemu-shim: guest Load with unresolvable IOSurface texture pages still encodes
/// (archive NULL seed / Metal Clear invent) — does not drop the pass.
#[test]
fn mrt_draw_request_load_seed_miss_still_encodes() {
    use crate::runtime::decode::render::ColorAttachment;
    use crate::runtime::host::FakeHost;
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.define_task(1, 0x1000, 2);
    // IOSurface texture registered with geom but empty page table → seed read fails.
    assert!(state.map_surface(9));
    assert!(state.set_mapping_geom(9, 8, 8, MTL_FORMAT_BGRA8_UNORM));
    // gen must be non-zero for Load path to attempt a snapshot (archive).
    state
        .surfaces
        .mappings
        .get_mut(&9)
        .unwrap()
        .content
        .guest_page_generation = 1;
    state.fixtures.texture_to_mapping.insert((1, 42), 9);
    let att = ColorAttachment {
        texture_ref: 42,
        resolve_texture_ref: 0,
        level: 0,
        slice: 0,
        depth_plane: 0,
        load_action: MTL_LOAD_ACTION_LOAD,
        store_action: MTL_STORE_ACTION_STORE,
        clear_color: [1.0, 1.0, 1.0, 1.0], // would paint solid white if Clear invented
    };
    let slots = [(0u32, att)];
    let req = mrt_draw_request(&mut state, &mut host, 1, 1, &slots, &[], test_triangle());
    // Archive: seed miss still builds the job (NULL seed). Product must not
    // drop the pass — that freezes lagging dual-mid on stale logo.
    let req = req
        .expect("attachment planning must succeed")
        .expect("Load seed miss must still encode (archive NULL seed)");
    assert!(
        req.colors[0].target_seed_rgba.is_none(),
        "seed miss leaves seed None (Metal Clear invent, full Store)"
    );
    assert_eq!(
        req.colors[0].load_action,
        reims_vgpu_protocol::pass_action::LoadAction::Load
    );
}

#[test]
fn mrt_draw_request_preserves_each_mismatched_attachment_geometry() {
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.define_task(1, 0x1000, 2);
    assert!(state.map_surface(9));
    assert!(state.map_surface(10));
    assert!(state.set_mapping_geom(9, 8, 8, MTL_FORMAT_BGRA8_UNORM));
    assert!(state.set_mapping_geom(10, 4, 8, MTL_FORMAT_BGRA8_UNORM));
    state.fixtures.texture_to_mapping.insert((1, 42), 9);
    state.fixtures.texture_to_mapping.insert((1, 43), 10);
    let slots = [
        (0, clear_black_attachment(42)),
        (1, clear_black_attachment(43)),
    ];

    let request = mrt_draw_request(&mut state, &mut host, 1, 1, &slots, &[], test_triangle())
        .expect("different attachment geometries are one valid Metal pass")
        .expect("two bound attachments produce a draw request");
    assert_eq!(request.colors.len(), 2);
    assert_eq!((request.colors[0].width, request.colors[0].height), (8, 8));
    assert_eq!((request.colors[1].width, request.colors[1].height), (4, 8));
}

#[test]
fn mrt_draw_request_refuses_unknown_pass_actions_before_execution() {
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.define_task(1, 0x1000, 2);
    assert!(state.map_surface(9));
    assert!(state.set_mapping_geom(9, 8, 8, MTL_FORMAT_BGRA8_UNORM));
    state.fixtures.texture_to_mapping.insert((1, 42), 9);

    let mut attachment = clear_black_attachment(42);
    attachment.load_action = 3;
    assert!(matches!(
        mrt_draw_request(
            &mut state,
            &mut host,
            1,
            7,
            &[(0, attachment)],
            &[],
            test_triangle(),
        ),
        Err(reims_vgpu_core::AttachmentPlanDecline::PassAction {
            slot: 0,
            reason: reims_vgpu_protocol::PassActionDecodeError::Load(3),
        })
    ));

    attachment.load_action = MTL_LOAD_ACTION_CLEAR;
    attachment.store_action = 4;
    assert!(matches!(
        mrt_draw_request(
            &mut state,
            &mut host,
            1,
            7,
            &[(0, attachment)],
            &[],
            test_triangle(),
        ),
        Err(reims_vgpu_core::AttachmentPlanDecline::PassAction {
            slot: 0,
            reason: reims_vgpu_protocol::PassActionDecodeError::Store(4),
        })
    ));
}

#[test]
fn mrt_draw_request_keeps_multisample_source_and_resolve_destination_distinct() {
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use reims_vgpu_protocol::pass_action::MTL_STORE_ACTION_MULTISAMPLE_RESOLVE;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.define_task(1, 0x1000, 2);
    for (texture_ref, mapping_id) in [(42, 9), (43, 10)] {
        assert!(state.map_surface(mapping_id));
        assert!(state.set_mapping_geom(mapping_id, 64, 64, MTL_FORMAT_BGRA8_UNORM));
        state
            .state
            .fixtures
            .texture_to_mapping
            .insert((1, texture_ref), mapping_id);
    }
    let mut att = clear_black_attachment(42);
    att.resolve_texture_ref = 43;
    att.store_action = MTL_STORE_ACTION_MULTISAMPLE_RESOLVE;

    let req = single_rt_draw_request(&mut state, &mut host, 7, att)
        .expect("matching source and resolve geometry is representable");
    assert_eq!(req.colors.len(), 1);
    assert_eq!(req.colors[0].texture_ref, 43, "the published target");
    assert_eq!(
        req.colors[0].multisample_source_ref, 42,
        "the raster attachment remains separately named"
    );
    assert_eq!(
        req.colors[0].store_action,
        reims_vgpu_protocol::pass_action::StoreAction::MultisampleResolve
    );
}

#[test]
fn mrt_draw_request_gets_attachment_samples_from_the_bound_pipeline_before_encode() {
    use crate::runtime::decode::resource::RenderPipelineDescriptor;
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use reims_vgpu_protocol::pass_action::MTL_STORE_ACTION_MULTISAMPLE_RESOLVE;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.define_task(1, 0x1000, 2);
    for (texture_ref, mapping_id) in [(42, 9), (43, 10)] {
        assert!(state.map_surface(mapping_id));
        assert!(state.set_mapping_geom(mapping_id, 64, 64, MTL_FORMAT_BGRA8_UNORM));
        state
            .state
            .fixtures
            .texture_to_mapping
            .insert((1, texture_ref), mapping_id);
    }
    state.task_objects.render_pipelines.register(
        1,
        reims_vgpu_protocol::SerializerRef::new(7),
        crate::runtime::pipeline_resolve::retained_pipeline_with_desc_for_test(
            RenderPipelineDescriptor {
                raster_sample_count: 4,
                ..RenderPipelineDescriptor::default()
            },
        ),
    );
    let mut att = clear_black_attachment(42);
    att.resolve_texture_ref = 43;
    att.store_action = MTL_STORE_ACTION_MULTISAMPLE_RESOLVE;

    let req = single_rt_draw_request(&mut state, &mut host, 7, att)
        .expect("matching source and resolve geometry is representable");
    assert_eq!(req.colors[0].sample_count, 4);
    assert_eq!(
        req.colors[0].texture_ref, 43,
        "the published resolve target"
    );
    assert_eq!(
        req.colors[0].multisample_source_ref, 42,
        "the multisample source retains its own identity"
    );
}

#[test]
fn mrt_draw_request_refuses_a_resolve_destination_with_different_geometry() {
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use reims_vgpu_protocol::pass_action::MTL_STORE_ACTION_MULTISAMPLE_RESOLVE;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.define_task(1, 0x1000, 2);
    for (texture_ref, mapping_id, width) in [(42, 9, 64), (43, 10, 32)] {
        assert!(state.map_surface(mapping_id));
        assert!(state.set_mapping_geom(mapping_id, width, 64, MTL_FORMAT_BGRA8_UNORM));
        state
            .state
            .fixtures
            .texture_to_mapping
            .insert((1, texture_ref), mapping_id);
    }
    let mut att = clear_black_attachment(42);
    att.resolve_texture_ref = 43;
    att.store_action = MTL_STORE_ACTION_MULTISAMPLE_RESOLVE;
    assert!(matches!(
        mrt_draw_request(
            &mut state,
            &mut host,
            1,
            7,
            &[(0, att)],
            &[],
            test_triangle(),
        ),
        Err(
            reims_vgpu_core::AttachmentPlanDecline::ResolveTargetMismatch {
                slot: 0,
                source_ref: 42,
                resolve_ref: 43,
                source_width: 64,
                source_height: 64,
                resolve_width: 32,
                resolve_height: 64,
                source_format: MTL_FORMAT_BGRA8_UNORM,
                resolve_format: MTL_FORMAT_BGRA8_UNORM,
            }
        )
    ));
}

#[test]
fn mrt_draw_request_distinguishes_missing_resolve_target_from_no_attachment() {
    use reims_vgpu_core::{
        pixel_format::MTL_FORMAT_BGRA8_UNORM, AttachmentPlanDecline, AttachmentTargetRole,
    };
    use reims_vgpu_protocol::pass_action::MTL_STORE_ACTION_MULTISAMPLE_RESOLVE;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.define_task(1, 0x1000, 2);
    assert!(state.map_surface(9));
    assert!(state.set_mapping_geom(9, 64, 64, MTL_FORMAT_BGRA8_UNORM));
    state.fixtures.texture_to_mapping.insert((1, 42), 9);

    let mut attached = clear_black_attachment(42);
    attached.resolve_texture_ref = 43;
    attached.store_action = MTL_STORE_ACTION_MULTISAMPLE_RESOLVE;
    assert!(matches!(
        mrt_draw_request(
            &mut state,
            &mut host,
            1,
            7,
            &[(0, attached)],
            &[],
            test_triangle(),
        ),
        Err(AttachmentPlanDecline::TargetUnresolved {
            slot: 0,
            texture_ref: 43,
            role: AttachmentTargetRole::Resolve,
        })
    ));

    assert!(
        mrt_draw_request(&mut state, &mut host, 1, 7, &[], &[], test_triangle(),)
            .is_ok_and(|request| request.is_none()),
        "an absent attachment set is expected control flow, not a refusal"
    );
}

/// qemu-shim: type-8 view of IOSurface texture is a valid color RT (archive
/// resource_resolve_texture view chain). Without this, App Store UI pipes
/// that bind a view as color attachment drop the entire MRT pass.
#[test]
fn mrt_draw_request_type8_view_of_iosurface_texture_as_color_rt() {
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE_VIEW,
        TEXTURE_VIEW_DESC_BASE_REF, TEXTURE_VIEW_DESC_LEN, TEXTURE_VIEW_DESC_LEVEL_BASE,
        TEXTURE_VIEW_DESC_LEVEL_COUNT, TEXTURE_VIEW_DESC_OPCODE, TEXTURE_VIEW_DESC_PIXEL_FORMAT,
        TEXTURE_VIEW_DESC_SLICE_BASE, TEXTURE_VIEW_DESC_SLICE_COUNT, TEXTURE_VIEW_DESC_TEXTURE_REF,
        TEXTURE_VIEW_DESC_TEXTURE_TYPE, TEXTURE_VIEW_MIN_RANGED, TEXTURE_VIEW_MTL_TYPE_2D,
        TEXTURE_VIEW_OPCODE_RANGED,
    };
    use crate::runtime::host::FakeHost;
    use reims_vgpu_core::endian::{st16, st32, st64};
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    // One-level page table: GVA pages 0..7 → data PFNs (blit_exec pattern).

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    // Object list at GVA page 0; count covers live residual slot 211.
    assert!(state.set_object_list(1, 0, 256));

    // Base IOSurface texture mid 9 latched as texture ref 3.
    assert!(state.map_surface(9));
    assert!(state.set_mapping_geom(9, 64, 64, MTL_FORMAT_BGRA8_UNORM));
    state
        .surfaces
        .mappings
        .get_mut(&9)
        .unwrap()
        .content
        .guest_page_generation = 1;
    state.fixtures.texture_to_mapping.insert((1, 3), 9);

    // Type-8 view ref 211 → base 3 (identity, level 0) — live residual slot.
    let view_ref = 211u32;
    let base_ref = 3u32;
    let len = TEXTURE_VIEW_MIN_RANGED;
    let mut desc = vec![0u8; len];
    st32(
        &mut desc[TEXTURE_VIEW_DESC_OPCODE..],
        TEXTURE_VIEW_OPCODE_RANGED,
    );
    st32(&mut desc[TEXTURE_VIEW_DESC_LEN..], len as u32);
    st32(&mut desc[TEXTURE_VIEW_DESC_TEXTURE_REF..], view_ref);
    st32(&mut desc[TEXTURE_VIEW_DESC_BASE_REF..], base_ref);
    st16(
        &mut desc[TEXTURE_VIEW_DESC_PIXEL_FORMAT..],
        MTL_FORMAT_BGRA8_UNORM,
    );
    st16(
        &mut desc[TEXTURE_VIEW_DESC_TEXTURE_TYPE..],
        TEXTURE_VIEW_MTL_TYPE_2D,
    );
    st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_BASE..], 0);
    st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_COUNT..], 1);
    st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_BASE..], 0);
    st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_COUNT..], 1);
    let desc_gva = 0x280u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &desc);
    let off = list_object_entry_offset(view_ref, 256).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (OBJECT_TYPE_TEXTURE_VIEW as u32) | ((len as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    let att = clear_black_attachment(view_ref);
    let req = single_rt_draw_request(&mut state, &mut host, 12, att)
        .expect("type-8 view of IOSurface texture must resolve as color RT");
    assert_eq!(req.colors[0].mapping_id(), 9);
    assert_eq!(req.colors[0].width, 64);
    assert_eq!(req.colors[0].height, 64);
    assert_eq!(req.colors[0].texture_ref, view_ref);
}

/// Archive `REIMS_VGPU_RESOURCE_RESOLVE_MAX_VIEW_CHAIN`: nested type-8 → type-8 →
/// IOSurface texture must collapse to the non-view base. One-hop resolve left the mid
/// base as type-8 and dropped the MRT pass (`view_base_or_swizzle`).
#[test]
fn mrt_draw_request_nested_type8_view_chain_to_iosurface_texture() {
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE_VIEW,
        TEXTURE_VIEW_DESC_BASE_REF, TEXTURE_VIEW_DESC_LEN, TEXTURE_VIEW_DESC_LEVEL_BASE,
        TEXTURE_VIEW_DESC_LEVEL_COUNT, TEXTURE_VIEW_DESC_OPCODE, TEXTURE_VIEW_DESC_PIXEL_FORMAT,
        TEXTURE_VIEW_DESC_SLICE_BASE, TEXTURE_VIEW_DESC_SLICE_COUNT, TEXTURE_VIEW_DESC_TEXTURE_REF,
        TEXTURE_VIEW_DESC_TEXTURE_TYPE, TEXTURE_VIEW_MIN_RANGED, TEXTURE_VIEW_MTL_TYPE_2D,
        TEXTURE_VIEW_OPCODE_RANGED,
    };
    use crate::runtime::host::FakeHost;
    use reims_vgpu_core::endian::{st16, st32, st64};
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    fn write_type8_view(
        host: &mut FakeHost,
        state: &Device,
        view_ref: u32,
        base_ref: u32,
        desc_gva: u64,
    ) {
        let len = TEXTURE_VIEW_MIN_RANGED;
        let mut desc = vec![0u8; len];
        st32(
            &mut desc[TEXTURE_VIEW_DESC_OPCODE..],
            TEXTURE_VIEW_OPCODE_RANGED,
        );
        st32(&mut desc[TEXTURE_VIEW_DESC_LEN..], len as u32);
        st32(&mut desc[TEXTURE_VIEW_DESC_TEXTURE_REF..], view_ref);
        st32(&mut desc[TEXTURE_VIEW_DESC_BASE_REF..], base_ref);
        st16(
            &mut desc[TEXTURE_VIEW_DESC_PIXEL_FORMAT..],
            MTL_FORMAT_BGRA8_UNORM,
        );
        st16(
            &mut desc[TEXTURE_VIEW_DESC_TEXTURE_TYPE..],
            TEXTURE_VIEW_MTL_TYPE_2D,
        );
        st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_BASE..], 0);
        st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_COUNT..], 1);
        st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_BASE..], 0);
        st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_COUNT..], 1);
        write_task_gva_arm64e(&mut *host, &state.tasks[1], desc_gva, &desc);
        let off = list_object_entry_offset(view_ref, 256).unwrap();
        let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_TEXTURE_VIEW as u32) | ((len as u32) << 8);
        st32(&mut list_entry[0..], packed);
        list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        write_task_gva_arm64e(&mut *host, &state.tasks[1], off, &list_entry);
    }

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 256));

    // IOSurface texture mid 9 as texture ref 3.
    assert!(state.map_surface(9));
    assert!(state.set_mapping_geom(9, 64, 64, MTL_FORMAT_BGRA8_UNORM));
    state
        .surfaces
        .mappings
        .get_mut(&9)
        .unwrap()
        .content
        .guest_page_generation = 1;
    state.fixtures.texture_to_mapping.insert((1, 3), 9);

    // Inner view 8 → base 3 (IOSurface texture); outer view 211 → base 8 (type-8).
    write_type8_view(&mut host, &state, 8, 3, 0x280);
    write_type8_view(&mut host, &state, 211, 8, 0x300);

    let att = clear_black_attachment(211);
    let req = single_rt_draw_request(&mut state, &mut host, 12, att)
        .expect("nested type-8→type-8→IOSurface texture must resolve as color RT");
    assert_eq!(req.colors[0].mapping_id(), 9);
    assert_eq!(req.colors[0].width, 64);
    assert_eq!(req.colors[0].height, 64);
    assert_eq!(req.colors[0].texture_ref, 211);
}

/// Archive resolve_texture rejects non-identity swizzle for RT resolve.
#[test]
fn mrt_draw_request_type8_swizzled_view_rejected_as_color_rt() {
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE_VIEW,
        TEXTURE_VIEW_DESC_BASE_REF, TEXTURE_VIEW_DESC_LEN, TEXTURE_VIEW_DESC_LEVEL_BASE,
        TEXTURE_VIEW_DESC_LEVEL_COUNT, TEXTURE_VIEW_DESC_OPCODE, TEXTURE_VIEW_DESC_PIXEL_FORMAT,
        TEXTURE_VIEW_DESC_SLICE_BASE, TEXTURE_VIEW_DESC_SLICE_COUNT, TEXTURE_VIEW_DESC_SWIZZLE,
        TEXTURE_VIEW_DESC_TEXTURE_REF, TEXTURE_VIEW_DESC_TEXTURE_TYPE, TEXTURE_VIEW_MIN_SWIZZLE,
        TEXTURE_VIEW_MTL_TYPE_2D, TEXTURE_VIEW_OPCODE_SWIZZLE,
    };
    use crate::runtime::host::FakeHost;
    use reims_vgpu_core::endian::{st16, st32, st64};
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));
    assert!(state.map_surface(9));
    assert!(state.set_mapping_geom(9, 64, 64, MTL_FORMAT_BGRA8_UNORM));
    state.fixtures.texture_to_mapping.insert((1, 3), 9);

    let view_ref = 8u32;
    let len = TEXTURE_VIEW_MIN_SWIZZLE;
    let mut desc = vec![0u8; len];
    st32(
        &mut desc[TEXTURE_VIEW_DESC_OPCODE..],
        TEXTURE_VIEW_OPCODE_SWIZZLE,
    );
    st32(&mut desc[TEXTURE_VIEW_DESC_LEN..], len as u32);
    st32(&mut desc[TEXTURE_VIEW_DESC_TEXTURE_REF..], view_ref);
    st32(&mut desc[TEXTURE_VIEW_DESC_BASE_REF..], 3);
    st16(
        &mut desc[TEXTURE_VIEW_DESC_PIXEL_FORMAT..],
        MTL_FORMAT_BGRA8_UNORM,
    );
    st16(
        &mut desc[TEXTURE_VIEW_DESC_TEXTURE_TYPE..],
        TEXTURE_VIEW_MTL_TYPE_2D,
    );
    st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_BASE..], 0);
    st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_COUNT..], 1);
    st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_BASE..], 0);
    st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_COUNT..], 1);
    // Non-identity BGRA → RGBA channel remap.
    desc[TEXTURE_VIEW_DESC_SWIZZLE..TEXTURE_VIEW_DESC_SWIZZLE + 4].copy_from_slice(&[2u8, 1, 0, 3]);
    let desc_gva = 0x280u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &desc);
    let off = list_object_entry_offset(view_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (OBJECT_TYPE_TEXTURE_VIEW as u32) | ((len as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    let att = clear_black_attachment(view_ref);
    assert!(
        matches!(
            mrt_draw_request(
                &mut state,
                &mut host,
                1,
                12,
                &[(0u32, att)],
                &[],
                test_triangle()
            ),
        Err(reims_vgpu_core::AttachmentPlanDecline::TargetUnresolved {
            slot: 0,
            texture_ref,
            role: reims_vgpu_core::AttachmentTargetRole::Source,
            }) if texture_ref == view_ref
        ),
        "swizzled type-8 must not resolve as color RT"
    );
}

/// qemu-shim: type-2 linear RGBA16Float is a valid color RT. Stale
/// `texture_to_mapping` from a prior IOSurface texture at the same ref must not
/// fail-closed (live residual ref=199 type=2 fmt=0x73).
#[test]
fn mrt_draw_request_type2_rgba16f_as_color_rt_despite_stale_iosurface_latch() {
    use crate::runtime::decode::resource::{
        list_object_entry_offset, LINEAR_DESC_HANDLE, LINEAR_DESC_SIZE, OBJECT_LIST_ENTRY_LEN,
        OBJECT_TYPE_TEXTURE, RESOURCE_PAGE_SHIFT, TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_HEIGHT,
        TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE, TEXTURE_DESC_WIDTH,
    };
    use crate::runtime::host::FakeHost;
    use reims_vgpu_core::endian::{st16, st32, st64};
    use reims_vgpu_core::pixel_format::{MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_RGBA16_FLOAT};

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 16);
    assert!(state.set_object_list(1, 0, 256));

    // Stale IOSurface texture latch at ref 199 (guest recycled the ref to type-2).
    assert!(state.map_surface(99));
    assert!(state.set_mapping_geom(99, 64, 64, MTL_FORMAT_BGRA8_UNORM));
    state.fixtures.texture_to_mapping.insert((1, 199), 99);

    // Live type-2 RGBA16Float 480×64 bpr=3840 (live residual shape).
    let tex_ref = 199u32;
    let w = 480u32;
    let h = 64u32;
    let bpr = 3840u32;
    let handle = 8u32; // GVA page under setup_task_pages data
    let alloc = (bpr as u64) * (h as u64);
    let mut desc = vec![0u8; TEXTURE_DESC_BASE_LEN];
    st64(&mut desc[LINEAR_DESC_SIZE..], alloc);
    st32(&mut desc[LINEAR_DESC_HANDLE..], handle);
    st32(&mut desc[TEXTURE_DESC_ROW_STRIDE..], bpr);
    st32(&mut desc[TEXTURE_DESC_WIDTH..], w);
    st32(&mut desc[TEXTURE_DESC_HEIGHT..], h);
    write_linear_texture_packing(&mut desc, 1, 1, 0, alloc);
    st16(
        &mut desc[TEXTURE_DESC_PIXEL_FORMAT..],
        MTL_FORMAT_RGBA16_FLOAT,
    );
    let desc_gva = 0x280u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &desc);
    let off = list_object_entry_offset(tex_ref, 256).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (OBJECT_TYPE_TEXTURE as u32) | ((TEXTURE_DESC_BASE_LEN as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    let att = clear_black_attachment(tex_ref);
    let req = single_rt_draw_request(&mut state, &mut host, 12, att)
        .expect("type-2 RGBA16F RT must resolve despite stale IOSurface texture latch");
    assert_eq!(req.colors[0].mapping_id(), 0);
    assert_eq!(req.colors[0].width, w);
    assert_eq!(req.colors[0].height, h);
    assert_eq!(req.colors[0].format, MTL_FORMAT_RGBA16_FLOAT);
    assert_eq!(
        req.colors[0].target_gva(),
        (handle as u64) << RESOURCE_PAGE_SHIFT
    );
    // Stale latch must be dropped.
    assert!(!state
        .fixtures
        .texture_to_mapping
        .contains_key(&(1, tex_ref)));
}

/// Live IOSurface texture descriptor mapping_id wins over a stale texture_to_mapping
/// latch (dual-mid recycled-ref residual: full desktop Store must land on
/// the mid named by the live descriptor, not a prior latch).
#[test]
fn mrt_draw_request_iosurface_texture_live_mapping_overrides_stale_latch() {
    use crate::runtime::host::FakeHost;
    use reims_vgpu_core::endian::{st16, st32, st64};
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    // 1-level page table: GVA page 0 → data pfn 4.
    let dir_gpa = 2u64 << PAGE_SHIFT_ARM64E;
    let root_gpa = 3u64 << PAGE_SHIFT_ARM64E;
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x4000, 0);
    host.map_range(data_gpa, 0x200, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    st32(&mut d[..4], 4);
    let _ = host.write_gpa(root_gpa, &d[..4]);
    state.define_task(1, 0x1000, 2);
    assert!(state.set_object_list(1, 0, 8));
    // Live mapper IOSurface texture at ref=1 resolves mapper identity 4.
    let mut entry = [0u8; 12];
    st32(&mut entry[0..], 11u32 | (0x38u32 << 8));
    entry[4..12].copy_from_slice(&0x40u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 12, &entry);
    let mut desc = [0u8; 0x38];
    st64(&mut desc[0..], 4);
    st32(&mut desc[0x08..], 0x0c);
    st32(&mut desc[0x0c..], 0x30);
    st32(&mut desc[0x10..], 1);
    st16(&mut desc[0x16..], MTL_FORMAT_BGRA8_UNORM);
    st32(&mut desc[0x18..], 64);
    st32(&mut desc[0x1c..], 32);
    st32(&mut desc[0x20..], 1);
    st16(&mut desc[0x24..], 1);
    st16(&mut desc[0x26..], 1);
    st16(&mut desc[0x28..], 1);
    let _ = host.write_gpa(data_gpa + 0x40, &desc);

    // Both mids exist; stale latch points ref 1 at mid 3.
    assert!(state.map_surface(3));
    assert!(state.set_mapping_geom(3, 64, 32, MTL_FORMAT_BGRA8_UNORM));
    assert!(state.map_surface(4));
    assert!(state.set_mapping_geom(4, 64, 32, MTL_FORMAT_BGRA8_UNORM));
    state.fixtures.texture_to_mapping.insert((1, 1), 3);

    let att = clear_black_attachment(1);
    assert!(single_rt_draw_request(&mut state, &mut host, 12, att).is_none());
    assert_eq!(
        state.fixtures.texture_to_mapping.get(&(1, 1)).copied(),
        Some(3)
    );

    assert!(state.map_mapper_surface(
        reims_vgpu_protocol::MapperSurfaceRef::new(4),
        reims_vgpu_protocol::MapperResolvedSurfaceId::new(4)
    ));
    assert!(state.set_mapping_geom(4, 64, 32, MTL_FORMAT_BGRA8_UNORM));
    let req = single_rt_draw_request(&mut state, &mut host, 12, att)
        .expect("live IOSurface texture RT must resolve");
    assert_eq!(
        req.colors[0].mapping_id(),
        4,
        "live descriptor mapping_id=4 must beat stale latch mid=3"
    );
    assert_eq!(
        state.fixtures.texture_to_mapping.get(&(1, 1)).copied(),
        Some(4)
    );
}

/// Color RT materialization does not rematerialize non-zero view mips.
#[test]
fn mrt_draw_request_type8_nonzero_level_rejected_as_color_rt() {
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE_VIEW,
        TEXTURE_VIEW_DESC_BASE_REF, TEXTURE_VIEW_DESC_LEN, TEXTURE_VIEW_DESC_LEVEL_BASE,
        TEXTURE_VIEW_DESC_LEVEL_COUNT, TEXTURE_VIEW_DESC_OPCODE, TEXTURE_VIEW_DESC_PIXEL_FORMAT,
        TEXTURE_VIEW_DESC_SLICE_BASE, TEXTURE_VIEW_DESC_SLICE_COUNT, TEXTURE_VIEW_DESC_TEXTURE_REF,
        TEXTURE_VIEW_DESC_TEXTURE_TYPE, TEXTURE_VIEW_MIN_RANGED, TEXTURE_VIEW_MTL_TYPE_2D,
        TEXTURE_VIEW_OPCODE_RANGED,
    };
    use crate::runtime::host::FakeHost;
    use reims_vgpu_core::endian::{st16, st32, st64};
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));
    assert!(state.map_surface(9));
    assert!(state.set_mapping_geom(9, 64, 64, MTL_FORMAT_BGRA8_UNORM));
    state.fixtures.texture_to_mapping.insert((1, 3), 9);

    let view_ref = 8u32;
    let len = TEXTURE_VIEW_MIN_RANGED;
    let mut desc = vec![0u8; len];
    st32(
        &mut desc[TEXTURE_VIEW_DESC_OPCODE..],
        TEXTURE_VIEW_OPCODE_RANGED,
    );
    st32(&mut desc[TEXTURE_VIEW_DESC_LEN..], len as u32);
    st32(&mut desc[TEXTURE_VIEW_DESC_TEXTURE_REF..], view_ref);
    st32(&mut desc[TEXTURE_VIEW_DESC_BASE_REF..], 3);
    st16(
        &mut desc[TEXTURE_VIEW_DESC_PIXEL_FORMAT..],
        MTL_FORMAT_BGRA8_UNORM,
    );
    st16(
        &mut desc[TEXTURE_VIEW_DESC_TEXTURE_TYPE..],
        TEXTURE_VIEW_MTL_TYPE_2D,
    );
    st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_BASE..], 1); // mip 1
    st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_COUNT..], 1);
    st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_BASE..], 0);
    st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_COUNT..], 1);
    let desc_gva = 0x280u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &desc);
    let off = list_object_entry_offset(view_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (OBJECT_TYPE_TEXTURE_VIEW as u32) | ((len as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    let att = clear_black_attachment(view_ref);
    assert!(
        matches!(
            mrt_draw_request(
                &mut state,
                &mut host,
                1,
                12,
                &[(0u32, att)],
                &[],
                test_triangle()
            ),
        Err(reims_vgpu_core::AttachmentPlanDecline::TargetUnresolved {
            slot: 0,
            texture_ref,
            role: reims_vgpu_core::AttachmentTargetRole::Source,
            }) if texture_ref == view_ref
        ),
        "type-8 level_base!=0 must not resolve as color RT"
    );
}

/// Archive collapses a type-8 view's mip level into linear geometry:
/// a level-1 view of a type-2 texture is a color RT at that level's
/// plane (offset/dims/stride from the descriptor's level record) —
/// compositor blur/backdrop pyramids render into successive mips.
#[test]
fn mrt_draw_request_type8_mip_level_view_of_linear_as_color_rt() {
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE,
        OBJECT_TYPE_TEXTURE_VIEW, TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_LEVEL_RECORDS,
        TEXTURE_DESC_MIP_LEVEL_RECORD_LEN, TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE,
        TEXTURE_DESC_USED_SIZE, TEXTURE_DESC_WIDTH, TEXTURE_LEVEL_HEIGHT, TEXTURE_LEVEL_OFFSET,
        TEXTURE_LEVEL_ROW_STRIDE, TEXTURE_LEVEL_SIZE, TEXTURE_LEVEL_WIDTH,
        TEXTURE_VIEW_DESC_BASE_REF, TEXTURE_VIEW_DESC_LEN, TEXTURE_VIEW_DESC_LEVEL_BASE,
        TEXTURE_VIEW_DESC_LEVEL_COUNT, TEXTURE_VIEW_DESC_OPCODE, TEXTURE_VIEW_DESC_PIXEL_FORMAT,
        TEXTURE_VIEW_DESC_SLICE_BASE, TEXTURE_VIEW_DESC_SLICE_COUNT, TEXTURE_VIEW_DESC_TEXTURE_REF,
        TEXTURE_VIEW_DESC_TEXTURE_TYPE, TEXTURE_VIEW_MIN_RANGED, TEXTURE_VIEW_MTL_TYPE_2D,
        TEXTURE_VIEW_OPCODE_RANGED,
    };
    use crate::runtime::host::FakeHost;
    use reims_vgpu_core::endian::{st16, st32, st64};
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    // Type-2 base with 2 mips: L0 64x32 bpr 256; L1 at +0x2000, 32x16 bpr 128.
    let base_ref = 5u32;
    let body = TEXTURE_DESC_BASE_LEN + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
    let mut b = vec![0u8; body];
    st64(&mut b[0..], 0x20000); // allocation_size
    st32(&mut b[8..], 0x20); // handle
    write_linear_texture_packing(&mut b, 2, 1, 0, 0x2800);
    st32(&mut b[TEXTURE_DESC_USED_SIZE..], 64 * 32 * 4);
    st32(&mut b[TEXTURE_DESC_ROW_STRIDE..], 256);
    st32(&mut b[TEXTURE_DESC_WIDTH..], 64);
    st32(&mut b[TEXTURE_DESC_WIDTH + 4..], 32); // height
    let rec = TEXTURE_DESC_LEVEL_RECORDS;
    st64(&mut b[rec + TEXTURE_LEVEL_OFFSET..], 0x2000);
    st64(&mut b[rec + TEXTURE_LEVEL_SIZE..], 32 * 16 * 4);
    st64(&mut b[rec + TEXTURE_LEVEL_ROW_STRIDE..], 128);
    st32(&mut b[rec + TEXTURE_LEVEL_WIDTH..], 32);
    st32(&mut b[rec + TEXTURE_LEVEL_HEIGHT..], 16);
    st32(&mut b[rec + TEXTURE_LEVEL_HEIGHT + 4..], 1); // depth
    st16(
        &mut b[TEXTURE_DESC_PIXEL_FORMAT + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN..],
        MTL_FORMAT_BGRA8_UNORM,
    );
    let base_desc_gva = 0x200u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], base_desc_gva, &b);
    let off = list_object_entry_offset(base_ref, 32).unwrap();
    let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut le[0..],
        (OBJECT_TYPE_TEXTURE as u32) | ((body as u32) << 8),
    );
    le[4..12].copy_from_slice(&base_desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);

    // Type-8 view: level_base=1 over the type-2 base.
    let view_ref = 8u32;
    let len = TEXTURE_VIEW_MIN_RANGED;
    let mut desc = vec![0u8; len];
    st32(
        &mut desc[TEXTURE_VIEW_DESC_OPCODE..],
        TEXTURE_VIEW_OPCODE_RANGED,
    );
    st32(&mut desc[TEXTURE_VIEW_DESC_LEN..], len as u32);
    st32(&mut desc[TEXTURE_VIEW_DESC_TEXTURE_REF..], view_ref);
    st32(&mut desc[TEXTURE_VIEW_DESC_BASE_REF..], base_ref);
    st16(
        &mut desc[TEXTURE_VIEW_DESC_PIXEL_FORMAT..],
        MTL_FORMAT_BGRA8_UNORM,
    );
    st16(
        &mut desc[TEXTURE_VIEW_DESC_TEXTURE_TYPE..],
        TEXTURE_VIEW_MTL_TYPE_2D,
    );
    st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_BASE..], 1);
    st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_COUNT..], 1);
    st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_BASE..], 0);
    st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_COUNT..], 1);
    let desc_gva = 0x400u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &desc);
    let off = list_object_entry_offset(view_ref, 32).unwrap();
    let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut le[0..],
        (OBJECT_TYPE_TEXTURE_VIEW as u32) | ((len as u32) << 8),
    );
    le[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);

    let att = clear_black_attachment(view_ref);
    let req = single_rt_draw_request(&mut state, &mut host, 12, att)
        .expect("mip-1 view of linear texture must resolve as color RT");
    let c0 = &req.colors[0];
    assert_eq!(c0.mapping_id(), 0);
    assert_eq!(
        c0.target_gva(),
        ((0x20u64) << PAGE_SHIFT_ARM64E) + 0x2000,
        "RT gva = allocation base + level-1 offset"
    );
    assert_eq!((c0.width, c0.height), (32, 16), "level-1 dims");
    assert_eq!(c0.row_stride(), 128, "level-1 row stride");
}

#[test]
fn view_swizzle_remaps_rgba8_pixels() {
    // Every CPU remap must report itself: this is the path the Vulkan
    // pathway replaced with a component mapping, and an unreported
    // invocation is a texture that silently lost its zero-copy crossing.
    crate::runtime::census::view_swizzle_census::reset_for_tests();
    let capture = crate::observe::FailCapture::start();
    // Reims VGPU selectors: 0=zero 1=one 2=R 3=G 4=B 5=A → BGRA order + forced alpha one.
    let plan = pixel_format::swizzle_plan(&[4, 3, 2, 1]).unwrap();
    let mut rgba = vec![10u8, 20, 30, 40, 50, 60, 70, 80];
    apply_view_swizzle_rgba8(&mut rgba, Some(&plan), 1).unwrap();
    assert_eq!(&rgba[0..4], &[30, 20, 10, 255]);
    assert_eq!(&rgba[4..8], &[70, 60, 50, 255]);
    // Identity is a no-op.
    let id = pixel_format::swizzle_identity();
    let before = rgba.clone();
    apply_view_swizzle_rgba8(&mut rgba, Some(&id), 1).unwrap();
    assert_eq!(rgba, before);
    // No plan leaves buffer untouched.
    apply_view_swizzle_rgba8(&mut rgba, None, 1).unwrap();
    assert_eq!(rgba, before);
    // Odd length fails visibly.
    let mut bad = vec![1u8, 2, 3];
    assert!(apply_view_swizzle_rgba8(&mut bad, Some(&plan), 1).is_none());
    // One non-identity remap ran and said so; the identity and None calls did
    // not, and neither did the length-rejected one. Read off the always-on sink
    // rather than a counter: the line is what a boot actually has to show.
    let log = capture.lines().join("\n");
    assert_eq!(
        log.match_indices("view_swizzle_cpu_remap").count(),
        1,
        "exactly one CPU remap must be reported"
    );
    crate::runtime::census::view_swizzle_census::reset_for_tests();
}

#[test]
fn view_format_reinterprets_bgra_storage_as_rgba() {
    // Physical BGRA bytes B,G,R,A = 10,20,30,40.
    // As BGRA8 → RGBA sample: (30,20,10,40).
    // As RGBA8 view override → sample: (10,20,30,40) (byte reinterpret).
    let raw = [10u8, 20, 30, 40];
    let mut as_bgra = [0u8; 4];
    assert!(pixel_format::convert_row_to_rgba8(
        MTL_FORMAT_BGRA8_UNORM,
        &raw,
        1,
        &mut as_bgra
    ));
    assert_eq!(as_bgra, [30, 20, 10, 40]);
    let mut as_rgba = [0u8; 4];
    assert!(pixel_format::convert_row_to_rgba8(
        pixel_format::MTL_FORMAT_RGBA8_UNORM,
        &raw,
        1,
        &mut as_rgba
    ));
    assert_eq!(as_rgba, [10, 20, 30, 40]);
    // Combined path uses effective format.
    let fmt = effective_view_sample_format(
        MTL_FORMAT_BGRA8_UNORM,
        Some(pixel_format::MTL_FORMAT_RGBA8_UNORM),
    )
    .unwrap();
    let mut out = [0u8; 4];
    assert!(pixel_format::convert_row_to_rgba8(fmt, &raw, 1, &mut out));
    assert_eq!(out, [10, 20, 30, 40]);
}

/// A solid landing puts the same bytes in the guest's pages as the full-image
/// one it replaced.
///
/// The repeated-row writer converts once and copies that conversion to every
/// destination row, where the full-image writer converted each of its identical
/// rows. Those are two spellings of one result and this asserts they agree, over
/// a destination whose row stride is wider than its tight row — the case where a
/// stride mistake in the repeated path would write the right bytes to the wrong
/// offsets and a tight-stride test would not see it.
///
/// Fails without the change only in the direction that matters: it is the
/// equivalence, not the speed, that a future edit to `SourceRows` could break.
#[test]
fn a_solid_gva_landing_matches_the_full_image_landing_it_replaced() {
    use crate::runtime::host::FakeHost;
    use reims_vgpu_core::endian::st32;
    use reims_vgpu_core::pixel_format::{solid_rgba8, MTL_FORMAT_BGRA8_UNORM};
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    // Two identical guests, so the two writers land into two byte-for-byte
    // equal address spaces and the comparison is of the writes alone.
    fn guest(page_shift: u32) -> (FakeHost, Device) {
        let mut host = FakeHost::new();
        let dir_gpa = 2u64 << page_shift;
        let root_gpa = 3u64 << page_shift;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 1 << page_shift, 0);
        // Eight data pages, contiguous, so the destination span resolves whole.
        for p in 0..8u64 {
            host.map_range((5 + p) << page_shift, 1 << page_shift, 0);
        }
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        for p in 0..8u64 {
            st32(&mut d[..4], (5 + p) as u32);
            let _ = host.write_gpa(root_gpa + (1 + p) * 4, &d[..4]);
        }
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        state.page_shift = page_shift;
        state.define_task(1, 0x1000, 2);
        (host, state)
    }

    let page_shift = PAGE_SHIFT_X86;
    let gva = 1u64 << page_shift;
    let (w, h) = (7u32, 5u32);
    let bpr = 64u32; // wider than the 28-byte tight row, on purpose
    let clear = [0.2_f64, 0.4, 0.6, 1.0];

    let (mut h1, mut s1) = guest(page_shift);
    assert!(
        write_gva_solid8(
            &mut s1,
            &mut h1,
            1,
            gva,
            w,
            h,
            bpr,
            MTL_FORMAT_BGRA8_UNORM,
            &clear
        )
        .is_ok(),
        "the solid landing must succeed"
    );

    let (mut h2, mut s2) = guest(page_shift);
    let full = solid_rgba8(w, h, &clear);
    assert!(
        write_gva_rgba8(
            &mut s2,
            &mut h2,
            1,
            gva,
            w,
            h,
            bpr,
            MTL_FORMAT_BGRA8_UNORM,
            &full
        )
        .is_ok(),
        "the full-image landing must succeed"
    );

    let span = (h as usize) * (bpr as usize);
    let mut a = vec![0u8; span];
    let mut b = vec![0u8; span];
    assert!(gva_mem::read_task_gva(&h1, &s1.tasks[1], gva, &mut a, page_shift).is_ok());
    assert!(gva_mem::read_task_gva(&h2, &s2.tasks[1], gva, &mut b, page_shift).is_ok());
    assert_eq!(a, b, "the two landings must be byte-identical");
    // …and not both empty, which would make the assertion above vacuous.
    assert!(a.iter().any(|&x| x != 0), "the landing wrote something");
}

/// Regression: type-2/3 GVA Stores must walk with device page_shift (x86=12).
/// Using the arm64e-default fallback made every `linux_m2v_store gva=… ok=0`
/// on Ventura/Tahoe x86 product boots.
#[test]
fn write_gva_rgba8_uses_device_page_shift_x86() {
    use crate::runtime::host::FakeHost;
    use reims_vgpu_core::endian::st32;
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    let page_shift = PAGE_SHIFT_X86;
    let mut host = FakeHost::new();
    let dir_gpa = 2u64 << page_shift;
    let root_gpa = 3u64 << page_shift;
    // data for GVA page 1 (write_gva_rgba8 rejects gva==0 as "no target")
    let data_gpa = 5u64 << page_shift;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    host.map_range(data_gpa, 0x1000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    // PTE index 1 → pfn 5 (GVA 0x1000)
    st32(&mut d[..4], 5);
    let _ = host.write_gpa(root_gpa + 4, &d[..4]);

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.page_shift = page_shift;
    state.define_task(1, 0x1000, 2);

    let gva = 1u64 << page_shift; // 0x1000
                                  // Baseline walker under x86 tables.
    let probe = [0x11u8, 0x22, 0x33, 0x44];
    assert!(
        gva_mem::write_task_gva(&mut host, &state.tasks[1], gva, &probe, page_shift).is_ok(),
        "direct x86 GVA write must work"
    );

    // Tight RGBA8 2×2 → BGRA rows at GVA 0x1000.
    let rgba = [
        10u8, 20, 30, 255, // R G B A
        40, 50, 60, 255, //
        70, 80, 90, 255, //
        100, 110, 120, 255,
    ];
    assert!(
        write_gva_rgba8(
            &mut state,
            &mut host,
            1,
            gva,
            2,
            2,
            8, // bpr = 2*4
            MTL_FORMAT_BGRA8_UNORM,
            &rgba,
        )
        .is_ok(),
        "x86 page_shift=12 GVA store must succeed"
    );
    let mut back = [0u8; 8];
    assert!(gva_mem::read_task_gva(&host, &state.tasks[1], gva, &mut back, page_shift).is_ok());
    // BGRA row0: B,G,R,A = 30,20,10,255
    assert_eq!(&back[..4], &[30, 20, 10, 255]);
}

/// An encode Store of a type-2/3 GVA wallpaper layer lands in the texture_ref
/// and GVA caches and NOT in the surface_id map — three separate namespaces
/// that happen to be keyed by the same integer.
///
/// That a *sample* then reaches those bytes is `type3_sample_uses_type2_gva_cache`'s
/// job: it builds the object-list entry and the descriptor the resolver needs.
/// This fixture has neither, and the only rung that ever served it took its
/// geometry from whichever cache entry was keyed by the ref rather than from a
/// decoded descriptor. That rung is gone, so what this test can still assert
/// about the resolver is that it declines.
#[test]
fn gva_layer_host_cache_roundtrip_for_sample() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let tex_ref = 54u32;
    let gva = 0x2c48000u64;
    let w = 4u32;
    let h = 3u32;
    // Sky-blue solid (pipe-59 class): R G B A
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for px in rgba.chunks_exact_mut(4) {
        px[0] = 81;
        px[1] = 126;
        px[2] = 185;
        px[3] = 255;
    }
    host_cache_store_gva_layer(
        &mut state,
        &mut crate::runtime::host::FakeHost::new(),
        0,
        tex_ref,
        OBJECT_TYPE_TEXTURE,
        gva,
        w,
        h,
        &rgba,
        true,
    );
    let cached = crate::runtime::surface_cache::get_texture(&state, 0, tex_ref, w, h)
        .expect("texture_ref encode cache");
    // BGRA storage
    assert_eq!(&cached[0..4], &[185, 126, 81, 255]);
    assert!(crate::runtime::surface_cache::get(&state, tex_ref, w, h).is_none());
    assert_eq!(
        crate::runtime::surface_cache::get_gva(&state, gva, w, h).unwrap()[0],
        185
    );
    // No object list, so no decoded descriptor and therefore no geometry this
    // call may serve bytes at. Declining is the contract; inventing a geometry
    // from a cache entry is what used to happen.
    let mut host = crate::runtime::host::FakeHost::new();
    assert!(
        resolve_sampled_source(
            &mut state,
            &mut host,
            0,
            tex_ref,
            None,
            true,
            sampled_d2_shape(),
        )
        .is_none(),
        "a ref with no object-list entry must resolve to no sampled source"
    );
}
/// Guest-CPU-produced tight linear textures: unchanged native bytes must
/// reuse the memoized RGBA Arc under a stable generation identity; a guest
/// write must be observed and produce a new generation.
#[test]
fn guest_linear_memo_reuses_arc_and_observes_guest_writes() {
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, TEXTURE_DESC_BASE_LEN,
        TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE, TEXTURE_DESC_USED_SIZE,
        TEXTURE_DESC_WIDTH,
    };
    use reims_vgpu_core::endian::{st16, st32, st64};
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let dir_pfn = 2u32;
    let root_pfn = 3u32;
    let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_ARM64E;
    let root_gpa = (root_pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x4000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], root_pfn);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    assert!(host.write_gpa(dir_gpa, &d).is_ok());
    for i in 0..4u32 {
        let pfn = 4 + i;
        host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        assert!(host.write_gpa(root_gpa + (i as u64) * 4, &pte).is_ok());
    }
    state.define_task(1, 0x1000, dir_pfn);
    assert!(state.set_object_list(1, 0, 32));

    // Tight 4x2 BGRA8: bpr 16, texels at handle-page 1 (gva 0x4000).
    let tex_ref = 6u32;
    let body = TEXTURE_DESC_BASE_LEN;
    let mut b = vec![0u8; body];
    st64(&mut b[0..], 0x1000); // allocation_size
    st32(&mut b[8..], 1); // handle -> base gva 1 << page_shift
    write_linear_texture_packing(&mut b, 1, 1, 0, 16 * 2);
    st32(&mut b[TEXTURE_DESC_USED_SIZE..], 16 * 2);
    st32(&mut b[TEXTURE_DESC_ROW_STRIDE..], 16);
    st32(&mut b[TEXTURE_DESC_WIDTH..], 4);
    st32(&mut b[TEXTURE_DESC_WIDTH + 4..], 2); // height
    st16(&mut b[TEXTURE_DESC_PIXEL_FORMAT..], MTL_FORMAT_BGRA8_UNORM);
    let desc_gva = 0x200u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &b);
    let off = list_object_entry_offset(tex_ref, 32).unwrap();
    let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut le[0..],
        (OBJECT_TYPE_TEXTURE as u32) | ((body as u32) << 8),
    );
    le[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);
    let texel_gva = 1u64 << PAGE_SHIFT_ARM64E;
    let bgra = [7u8, 5, 3, 255].repeat(8);
    write_task_gva_arm64e(&mut host, &state.tasks[1], texel_gva, &bgra);

    // The caller resolves the object-list entry + decodes the descriptor
    // once and threads them in; the list is immutable for the draw.
    let le_entry = objects::lookup_list_entry(&state, &host, 1, tex_ref)
        .expect("object-list entry must resolve");
    let td = decode_texture_descriptor(
        &objects::read_descriptor(&state, &host, 1, &le_entry).expect("descriptor must read"),
    )
    .expect("descriptor must decode");

    let (w, h, rgba1, id1, fmt1) =
        load_linear_from_host_caches(&mut state, &mut host, 1, tex_ref, &td)
            .expect("guest tight linear must load");
    assert_eq!((w, h), (4, 2));
    assert_eq!(
        fmt1.layout(),
        TexelLayout::Bgra8,
        "the tight guest-memo path uploads native BGRA8 (no CPU swizzle)"
    );
    assert_eq!(&rgba1[..4], &[7, 5, 3, 255], "native BGRA8, unswizzled");
    let id1 = id1.expect("guest memo path must carry an identity");
    assert_eq!(id1.key, texel_gva);
    assert_ne!(id1.generation, 0, "0 means no host content yet");

    let (_, _, rgba2, id2, _) =
        load_linear_from_host_caches(&mut state, &mut host, 1, tex_ref, &td)
            .expect("repeat load must succeed");
    assert!(
        std::sync::Arc::ptr_eq(&rgba1, &rgba2),
        "unchanged native bytes must reuse the memoized Arc"
    );
    assert_eq!(id2.expect("identity").generation, id1.generation);

    // A direct guest write must be observed on the very next load.
    let bgra_new = [90u8, 60, 30, 255].repeat(8);
    write_task_gva_arm64e(&mut host, &state.tasks[1], texel_gva, &bgra_new);
    let (_, _, rgba3, id3, _) =
        load_linear_from_host_caches(&mut state, &mut host, 1, tex_ref, &td)
            .expect("post-write load must succeed");
    assert!(!std::sync::Arc::ptr_eq(&rgba1, &rgba3));
    assert_eq!(&rgba3[..4], &[90, 60, 30, 255], "native BGRA8, unswizzled");
    assert_ne!(id3.expect("identity").generation, id1.generation);
}

/// Padded-stride BGRA8 (the Safari-scroll former `lin_guest_fb` hot path)
/// now rides the guest-linear memo (gva recurrence measured ~99% under
/// scroll). Assert it uploads the guest's NATIVE BGRA8 bytes (`byte_format
/// == Bgra8`, no CPU channel swap), carries a memo identity so the engine
/// skips its content hash + upload, and that the row gather takes exactly
/// the tight texels — skipping the padding — into the tight output.
#[test]
fn padded_bgra8_memoized_uploads_native_without_swizzle() {
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, TEXTURE_DESC_BASE_LEN,
        TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE, TEXTURE_DESC_USED_SIZE,
        TEXTURE_DESC_WIDTH,
    };
    use reims_vgpu_core::endian::{st16, st32, st64};
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let dir_pfn = 2u32;
    let root_pfn = 3u32;
    let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_ARM64E;
    let root_gpa = (root_pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x4000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], root_pfn);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    assert!(host.write_gpa(dir_gpa, &d).is_ok());
    for i in 0..4u32 {
        let pfn = 4 + i;
        host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        assert!(host.write_gpa(root_gpa + (i as u64) * 4, &pte).is_ok());
    }
    state.define_task(1, 0x1000, dir_pfn);
    assert!(state.set_object_list(1, 0, 32));

    // 4x2 BGRA8 with a PADDED row stride: tight = 16, bpr = 24 (8 pad bytes
    // per row). Padding declines the tight-stride memo loader.
    let tex_ref = 6u32;
    let (w, h) = (4u32, 2u32);
    let tight = 16u32;
    let bpr = 24u32;
    let body = TEXTURE_DESC_BASE_LEN;
    let mut b = vec![0u8; body];
    st64(&mut b[0..], 0x1000); // allocation_size
    st32(&mut b[8..], 1); // handle -> base gva 1 << page_shift
    write_linear_texture_packing(&mut b, 1, 1, 0, u64::from(bpr * h));
    st32(&mut b[TEXTURE_DESC_USED_SIZE..], bpr * h);
    st32(&mut b[TEXTURE_DESC_ROW_STRIDE..], bpr);
    st32(&mut b[TEXTURE_DESC_WIDTH..], w);
    st32(&mut b[TEXTURE_DESC_WIDTH + 4..], h);
    st16(&mut b[TEXTURE_DESC_PIXEL_FORMAT..], MTL_FORMAT_BGRA8_UNORM);
    let desc_gva = 0x200u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &b);
    let off = list_object_entry_offset(tex_ref, 32).unwrap();
    let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut le[0..],
        (OBJECT_TYPE_TEXTURE as u32) | ((body as u32) << 8),
    );
    le[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);

    // Write two padded rows: each 16 tight BGRA bytes then 8 pad bytes.
    let texel_gva = 1u64 << PAGE_SHIFT_ARM64E;
    let row0: Vec<u8> = [1u8, 2, 3, 255].repeat(4); // 16 bytes
    let row1: Vec<u8> = [10u8, 20, 30, 255].repeat(4);
    let pad = [0xEEu8; 8];
    let mut backing = Vec::new();
    backing.extend_from_slice(&row0);
    backing.extend_from_slice(&pad);
    backing.extend_from_slice(&row1);
    backing.extend_from_slice(&pad);
    assert_eq!(backing.len(), (bpr * h) as usize);
    write_task_gva_arm64e(&mut host, &state.tasks[1], texel_gva, &backing);

    let le_entry = objects::lookup_list_entry(&state, &host, 1, tex_ref)
        .expect("object-list entry must resolve");
    let td = decode_texture_descriptor(
        &objects::read_descriptor(&state, &host, 1, &le_entry).expect("descriptor must read"),
    )
    .expect("descriptor must decode");

    let (gw, gh, rgba, identity, fmt) =
        load_linear_from_host_caches(&mut state, &mut host, 1, tex_ref, &td)
            .expect("padded BGRA8 must load via the memo");
    assert_eq!((gw, gh), (w, h));
    assert_eq!(
        fmt.layout(),
        TexelLayout::Bgra8,
        "padded BGRA8 must upload native (no CPU swizzle)"
    );
    let id = identity.expect("the padded memo path carries a producer identity");
    assert_ne!(id.generation, 0, "0 means no host content yet");
    // Tight output = the two source rows concatenated, native BGRA order,
    // padding stripped. Length is w*h*4 regardless of format.
    let mut want = Vec::new();
    want.extend_from_slice(&row0);
    want.extend_from_slice(&row1);
    assert_eq!(
        &rgba[..],
        &want[..],
        "native bytes gathered, padding skipped"
    );
    assert_eq!(rgba.len(), (tight * h) as usize);

    // A repeat bind of unchanged content reuses the memoized Arc (the whole
    // point — the engine then skips its content hash + upload).
    let (_, _, rgba2, id2, fmt2) =
        load_linear_from_host_caches(&mut state, &mut host, 1, tex_ref, &td)
            .expect("repeat padded load must succeed");
    assert!(
        std::sync::Arc::ptr_eq(&rgba, &rgba2),
        "unchanged padded bytes must reuse the memoized Arc"
    );
    assert_eq!(fmt2.layout(), TexelLayout::Bgra8);
    assert_eq!(id2.expect("identity").generation, id.generation);
}

/// Black-load-seed-discard regression: GVA identity wins over colliding
/// texture/surface namespaces, and a zero-RGB result remains valid.
///
/// The second half pins the ref door's currency test. A LOAD seed is the
/// attachment's prior content and the matching Store writes the composite back,
/// so a ref entry produced at another address must not be served as this one's
/// — that hands the pass another allocation's picture and arms the next frame
/// to load what this one stored. The same entry *is* served once the address
/// agrees, which is the case the door exists for: the GVA entry aged out of its
/// byte cap while the uncapped ref entry survived.
#[test]
fn color_load_seed_uses_provenance_and_preserves_black() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let task_id = std::process::id();
    let texture_ref = 0xe000_0000u32.wrapping_add(task_id);
    let target_gva = 0x5000_0000u64 + ((task_id as u64) << 12);
    let (w, h) = (2, 1);

    // Same numeric ref in the surface_id namespace must be irrelevant.
    crate::runtime::surface_cache::store(
        &mut state,
        texture_ref,
        w,
        h,
        vec![0, 0, 200, 255, 0, 0, 200, 255],
    );
    crate::runtime::surface_cache::store_texture(
        &mut state,
        task_id,
        texture_ref,
        w,
        h,
        vec![0, 180, 0, 255, 0, 180, 0, 255],
        target_gva,
    );
    crate::runtime::surface_cache::store_gva_owned(
        &mut state,
        target_gva,
        w,
        h,
        vec![0, 0, 0, 255, 0, 0, 0, 255],
        0,
        None,
        true,
    );

    let seed = seed_color_load(
        &mut state,
        &mut host,
        task_id,
        texture_ref,
        target_gva,
        w,
        h,
    )
    .expect("exact GVA cache seed");
    assert_eq!(seed, vec![0, 0, 0, 255, 0, 0, 0, 255]);

    // A different address with no GVA entry of its own must NOT be handed the
    // ref entry: those pixels were produced over `target_gva`, and serving them
    // here composites this pass onto another allocation's picture. The colliding
    // surface namespace (red) must stay unreachable either way.
    assert!(
        seed_color_load(
            &mut state,
            &mut host,
            task_id,
            texture_ref,
            target_gva + 0x1000,
            w,
            h,
        )
        .is_none(),
        "a ref entry produced at another address is not this attachment's prior content"
    );

    // Same address, GVA entry gone: this is the case the ref door is for, and it
    // serves green — never the colliding surface namespace's red.
    crate::runtime::surface_cache::evict_gva(&mut state, target_gva);
    let texture_seed = seed_color_load(
        &mut state,
        &mut host,
        task_id,
        texture_ref,
        target_gva,
        w,
        h,
    )
    .expect("texture-ref cache seed at the address that produced it");
    assert_eq!(texture_seed, vec![0, 180, 0, 255, 0, 180, 0, 255]);
}

/// A draw whose colour0 LOAD seed was elided must leave the encode holding
/// either a chain or a seed — never neither.
///
/// `mrt_draw_request` sets [`DrawEncodeRequest::gva_load_source`] to
/// [`GvaLoadSource::Resident`] when the
/// engine still holds what the render Store published into the target's guest
/// pages, and it pays for that by leaving `colors[0].target_seed_rgba` as `None`
/// while the attachment still says `MTL_LOAD_ACTION_LOAD`. The attachment is then
/// only as defined as the encode side makes it: `resolve_gva_load_source` either
/// chains off the resident or reads the seed back, and a pass that does neither
/// loads undefined content.
///
/// **The regression this catches is the re-seed arm being dropped or
/// short-circuited**, which is the arm that no boot has yet exercised — the
/// commit that introduced the elision measured `gvaseed_chained` equal to
/// `gvaseed_elided` exactly, 4 475 of each, and `gvaseed_reseeded` zero. A rail
/// that never fires under the workload is a rail that regresses silently, and
/// this one regresses into a *plausible* frame rather than a blank one: the pass
/// composites onto whatever the attachment happened to contain. Deleting the
/// `None` arm, returning early from it, or letting it count without re-reading
/// would all keep every boot green and every existing test passing.
///
/// The race it stands in for is real. The allocation generation is resolved
/// after the request is built, so a page set that moved in between names a different
/// target — `gva_chain_identity` then resolves to an identity the registry has no
/// resident for, exactly as it does here, and the seed is already gone.
///
/// There is no engine in a unit test, so `resident_content_ready` is false for
/// every identity and the chain arm cannot be taken. That is the point: this is
/// the seedless case by construction, and the only correct behaviour is to put
/// the seed back. The guest pages below are real, so `Some` here means
/// `seed_color_load` actually re-read the attachment rather than the arm merely
/// counting itself.
#[test]
fn a_gva_load_from_resident_draw_with_no_resident_puts_the_seed_back() {
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, TEXTURE_DESC_BASE_LEN,
        TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE, TEXTURE_DESC_USED_SIZE,
        TEXTURE_DESC_WIDTH,
    };
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_core::endian::{st16, st32, st64};
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let dir_pfn = 2u32;
    let root_pfn = 3u32;
    let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_ARM64E;
    let root_gpa = (root_pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x4000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], root_pfn);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    assert!(host.write_gpa(dir_gpa, &d).is_ok());
    for i in 0..4u32 {
        let pfn = 4 + i;
        host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        assert!(host.write_gpa(root_gpa + (i as u64) * 4, &pte).is_ok());
    }
    state.define_task(1, 0x1000, dir_pfn);
    assert!(state.set_object_list(1, 0, 32));

    // The attachment the pass loads: a tight 4x2 BGRA8 linear texture whose
    // texels live in guest pages, so the re-seed has something real to find.
    let tex_ref = 6u32;
    let body = TEXTURE_DESC_BASE_LEN;
    let mut b = vec![0u8; body];
    st64(&mut b[0..], 0x1000);
    st32(&mut b[8..], 1); // handle -> base gva 1 << page_shift
    write_linear_texture_packing(&mut b, 1, 1, 0, 16 * 2);
    st32(&mut b[TEXTURE_DESC_USED_SIZE..], 16 * 2);
    st32(&mut b[TEXTURE_DESC_ROW_STRIDE..], 16);
    st32(&mut b[TEXTURE_DESC_WIDTH..], 4);
    st32(&mut b[TEXTURE_DESC_WIDTH + 4..], 2); // height
    st16(&mut b[TEXTURE_DESC_PIXEL_FORMAT..], MTL_FORMAT_BGRA8_UNORM);
    let desc_gva = 0x200u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &b);
    let off = list_object_entry_offset(tex_ref, 32).unwrap();
    let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut le[0..],
        (crate::runtime::decode::resource::OBJECT_TYPE_TEXTURE as u32) | ((body as u32) << 8),
    );
    le[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);
    let texel_gva = 1u64 << PAGE_SHIFT_ARM64E;
    let bgra = [7u8, 5, 3, 255].repeat(8);
    write_task_gva_arm64e(&mut host, &state.tasks[1], texel_gva, &bgra);

    // The request `mrt_draw_request` produces when it elides: LOAD, no seed, and
    // the flag that says the absence was deliberate.
    let req = DrawEncodeRequest {
        task_id: 1,
        gva_load_source: GvaLoadSource::Resident,
        colors: vec![ColorRtRequest {
            slot: 0,
            texture_ref: tex_ref,
            storage: linear_target_storage(texel_gva, 16, 2),
            width: 4,
            height: 2,
            format: MTL_FORMAT_BGRA8_UNORM,
            load_action: reims_vgpu_protocol::pass_action::LoadAction::Load,
            store_action: reims_vgpu_protocol::pass_action::StoreAction::Store,
            target_seed_rgba: None,
            ..Default::default()
        }],
        ..Default::default()
    };
    assert!(
        crate::runtime::draw::gva_chain_identity(state.executor.as_ref(), &req, 0).is_some(),
        "the elided request must still name an identity — the arm under test is \
         the one where that identity has no ready resident, not the one where \
         there is nothing to look up"
    );

    let chained_before = store_route_count("gvaseed_chained");
    let reseeded_before = store_route_count("gvaseed_reseeded");
    let mut chain_load_from_target = false;
    let resolution = crate::runtime::draw::execution::resolve_gva_load_source(
        &mut state,
        &mut host,
        &req,
        0,
        None,
        &mut chain_load_from_target,
    );

    assert_eq!(
        store_route_count("gvaseed_chained"),
        chained_before,
        "there is no engine resident here, so nothing may report a chain"
    );
    assert_eq!(
        store_route_count("gvaseed_reseeded"),
        reseeded_before + 1,
        "the seedless LOAD must take the re-seed arm exactly once"
    );
    assert!(
        resolution.identity.is_none() && resolution.guest_seed.is_none() && !chain_load_from_target,
        "a re-seeded pass loads from its attachment, not from a resident"
    );

    // The property, and the reason the counter alone is not enough: the seed is
    // back, and it holds the guest's pixels rather than an empty buffer.
    let seed = resolution
        .cpu_seed
        .as_ref()
        .expect("a LOAD whose elision was not honoured must have its seed restored");
    assert_eq!(
        seed.len(),
        4 * 2 * 4,
        "the restored seed must cover the whole attachment"
    );
    // BGRA8 in guest memory, RGBA8 in the seed: the linear GVA door of
    // `seed_color_load` converts, so the guest's `[7, 5, 3, 255]` arrives
    // channel-reversed. Asserting the converted value rather than the raw one is
    // deliberate — it pins that these bytes came through the seed path and not
    // from some other buffer that happened to be the right length.
    assert_eq!(
        &seed[..4],
        &[3, 5, 7, 255],
        "the restored seed must be the attachment's own guest texels"
    );
    assert!(
        seed.chunks_exact(4).all(|px| px == [3, 5, 7, 255]),
        "every texel of the re-read attachment, not just the first row"
    );
}

/// A GVA LOAD that names authoritative guest pages stays in that representation
/// until Vulkan preparation. Supplying a stable target allocation must produce
/// a bounded guest seed without calling the CPU RGBA fallback.
#[test]
fn a_gva_guest_page_load_becomes_an_importable_seed_without_cpu_bytes() {
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let import = std::sync::Arc::new(
        crate::runtime::guest_ram::GuestRamImport::new_host_allocation(0x2000_0000, 0x4000, 0x1000)
            .expect("aligned packed allocation"),
    );
    let backing = reims_vgpu_memory::GuestTargetMemory {
        backing: reims_vgpu_memory::GuestTargetBacking {
            allocation_host_ptr: import.host_base(),
            allocation_len: import.len(),
            resource_offset: 0,
            resource_len: 0x4000,
            plane_offset: 0x200,
            row_pitch: 16,
        },
        import,
        footprint: crate::runtime::guest_ram::GuestPageFootprint::new(
            std::sync::Arc::from([0x8000, 0x9000, 0xa000, 0xb000]),
            0x1000,
        )
        .expect("footprint"),
    };
    let req = DrawEncodeRequest {
        task_id: 1,
        gva_load_source: GvaLoadSource::GuestPages,
        colors: vec![ColorRtRequest {
            slot: 0,
            texture_ref: 6,
            storage: linear_target_storage(0x1200, 16, 2),
            width: 4,
            height: 2,
            format: MTL_FORMAT_BGRA8_UNORM,
            load_action: reims_vgpu_protocol::pass_action::LoadAction::Load,
            store_action: reims_vgpu_protocol::pass_action::StoreAction::Store,
            ..Default::default()
        }],
        ..Default::default()
    };
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let guest_before = store_route_count("gvaseed_guest_pages");
    let cpu_before = store_route_count("gvaseed_guest_cpu_fallback");
    let mut chain = false;

    let resolution = crate::runtime::draw::execution::resolve_gva_load_source(
        &mut state,
        &mut host,
        &req,
        7,
        Some(&backing),
        &mut chain,
    );

    assert!(resolution.identity.is_none() && !chain);
    assert!(resolution.cpu_seed.is_none());
    let seed = resolution
        .guest_seed
        .expect("the stable allocation supplies the LOAD seed");
    assert_eq!(seed.source.total_len, 32);
    assert_eq!(seed.source.runs[0].host_ptr, 0x2000_0200);
    assert!(req.colors[0].target_seed_rgba.is_none());
    assert_eq!(store_route_count("gvaseed_guest_pages"), guest_before + 1);
    assert_eq!(
        store_route_count("gvaseed_guest_cpu_fallback"),
        cpu_before,
        "the guest-page source must not be materialized as RGBA bytes"
    );
}

/// A IOSurface plane view ref is not itself a surface id. The descriptor's surface_id
/// remains authoritative even when the numeric ref collides with another
/// live display mapping (live app-launch ref=2 -> sid=71 class).
#[test]
fn iosurface_plane_view_sample_uses_descriptor_surface_id_not_ref_collision() {
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
    use crate::runtime::gva_mem;
    use reims_vgpu_core::endian::st32;
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();

    // One-level x86 GVA table: pages 0..2 -> data PFNs 4..6.
    let dir_pfn = 2u32;
    let root_pfn = 3u32;
    let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_X86;
    let root_gpa = (root_pfn as u64) << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    let mut dir = [0u8; 8];
    st32(&mut dir[DIRECTORY_ROOT_PFN as usize..], root_pfn);
    st32(&mut dir[DIRECTORY_DEPTH as usize..], 1);
    assert!(host.write_gpa(dir_gpa, &dir).is_ok());
    for i in 0..3u32 {
        let pfn = 4 + i;
        host.map_range((pfn as u64) << PAGE_SHIFT_X86, 0x1000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        assert!(host.write_gpa(root_gpa + (i as u64) * 4, &pte).is_ok());
    }
    state.define_task(1, 0x1000, dir_pfn);
    assert!(state.set_object_list(1, 0, 32));

    let texture_ref = 2u32;
    let surface_id = 71u32;
    let desc_gva = 0x1000u64;
    let built = reims_vgpu_wire::device_desc::IOSurfacePlaneViewBuilder::new(surface_id, 0, 0, 0)
        .with_len(objects::IOSURFACE_PLANE_VIEW_MIN_LEN);
    let desc = built.bytes();
    assert!(
        gva_mem::write_task_gva(&mut host, &state.tasks[1], desc_gva, desc, PAGE_SHIFT_X86,)
            .is_ok()
    );
    let list_off = list_object_entry_offset(texture_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32)
        | ((objects::IOSURFACE_PLANE_VIEW_MIN_LEN as u32) << 8);
    st32(&mut list_entry, packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        list_off,
        &list_entry,
        PAGE_SHIFT_X86,
    )
    .is_ok());

    // The lower numeric ref intentionally collides with an unrelated map.
    assert!(state.map_surface(texture_ref));
    assert!(state.set_mapping_geom(texture_ref, 8, 8, MTL_FORMAT_BGRA8_UNORM));
    state
        .surfaces
        .mappings
        .get_mut(&texture_ref)
        .unwrap()
        .pages
        .entries = vec![1];
    crate::runtime::surface_cache::store(
        &mut state,
        texture_ref,
        8,
        8,
        [0u8, 0, 255, 255].repeat(8 * 8),
    );

    assert!(state.map_surface(surface_id));
    assert!(state.set_mapping_geom(surface_id, 4, 3, MTL_FORMAT_BGRA8_UNORM));
    state
        .surfaces
        .mappings
        .get_mut(&surface_id)
        .unwrap()
        .pages
        .entries = vec![1];
    crate::runtime::surface_cache::store(
        &mut state,
        surface_id,
        4,
        3,
        [255u8, 0, 0, 255].repeat(4 * 3),
    );

    let (width, height, sampled_mid, sampled) = resolve_sampled_source(
        &mut state,
        &mut host,
        1,
        texture_ref,
        None,
        true,
        sampled_d2_shape(),
    )
    .expect("IOSurface plane view descriptor surface must sample");
    assert_eq!((width, height, sampled_mid), (4, 3, surface_id));
    let SampledSourceRequest::Bytes(sampled, _, layout, _) = sampled else {
        panic!("cache-backed fixture unexpectedly resolved a resident target");
    };
    // The host-cache rung uploads the scanout cache's BGRA8 verbatim, so this is
    // surface 71's stored pixel unswapped. It stays the discriminant this test
    // exists for: the colliding ref 2 holds [0,0,255,255], so reading the wrong
    // surface still fails here. Asserting the layout beside the bytes is what
    // keeps the pair honest — bytes alone would also pass if the layout drifted
    // to RGBA8 and every sampled frame came out channel-swapped.
    assert_eq!(layout.layout(), TexelLayout::Bgra8);
    assert_eq!(&sampled[..4], &[255, 0, 0, 255]);

    // Threading the caller-resolved resource must produce a byte-identical
    // sample to retrieving that same retained object inside the resolver.
    let threaded_resource = objects::resolve_resource(&state, &host, 1, texture_ref).ok();
    assert!(
        threaded_resource.is_some(),
        "IOSurface plane view fixture must expose a resource to thread"
    );
    let (tw, th, tmid, tsrc) = resolve_sampled_source(
        &mut state,
        &mut host,
        1,
        texture_ref,
        threaded_resource,
        true,
        sampled_d2_shape(),
    )
    .expect("threaded-resource sample must resolve");
    assert_eq!(
        (tw, th, tmid),
        (width, height, sampled_mid),
        "threaded resource changed the resolved geometry/mid"
    );
    let SampledSourceRequest::Bytes(tsampled, _, _, _) = tsrc else {
        panic!("threaded-resource sample changed the source variant");
    };
    assert_eq!(
        tsampled, sampled,
        "threaded resource must yield byte-identical sampled content"
    );
}

/// The sampled ladder's host-cache rung must offer a content identity, and that
/// identity must move whenever the cached frame does.
///
/// Both halves are load-bearing and they fail in opposite directions. With no
/// identity at all the engine has nothing to match on and re-digests the frame
/// on every bind — measured at 116 lookups a second over 201 MB, hashed twice
/// each, which was 73 % of the draw's timed work. With an identity that does
/// *not* move when the cache is rewritten, the engine binds the texture it
/// already holds and the newer frame is never uploaded: a stale compositing
/// layer, served silently, which is the one wrong answer this value can give.
///
/// The generation is asserted to equal the cache entry's own `host_gen` rather
/// than merely "some fresh number", because that equality is the whole
/// coherence argument — every writer of `host_surfaces` re-takes it from
/// `next_sampled_content_generation` in the same breath as it changes the bytes.
#[test]
fn iosurface_texture_host_cache_rung_identity_tracks_the_cached_frame() {
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
    use crate::runtime::gva_mem;
    use reims_vgpu_core::endian::st32;
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();

    let dir_pfn = 2u32;
    let root_pfn = 3u32;
    let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_X86;
    let root_gpa = (root_pfn as u64) << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    let mut dir = [0u8; 8];
    st32(&mut dir[DIRECTORY_ROOT_PFN as usize..], root_pfn);
    st32(&mut dir[DIRECTORY_DEPTH as usize..], 1);
    assert!(host.write_gpa(dir_gpa, &dir).is_ok());
    for i in 0..3u32 {
        let pfn = 4 + i;
        host.map_range((pfn as u64) << PAGE_SHIFT_X86, 0x1000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        assert!(host.write_gpa(root_gpa + (i as u64) * 4, &pte).is_ok());
    }
    state.define_task(1, 0x1000, dir_pfn);
    assert!(state.set_object_list(1, 0, 32));

    let texture_ref = 2u32;
    let surface_id = 71u32;
    let desc_gva = 0x1000u64;
    let built = reims_vgpu_wire::device_desc::IOSurfacePlaneViewBuilder::new(surface_id, 0, 0, 0)
        .with_len(objects::IOSURFACE_PLANE_VIEW_MIN_LEN);
    let desc = built.bytes();
    assert!(
        gva_mem::write_task_gva(&mut host, &state.tasks[1], desc_gva, desc, PAGE_SHIFT_X86,)
            .is_ok()
    );
    let list_off = list_object_entry_offset(texture_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32)
        | ((objects::IOSURFACE_PLANE_VIEW_MIN_LEN as u32) << 8);
    st32(&mut list_entry, packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        list_off,
        &list_entry,
        PAGE_SHIFT_X86,
    )
    .is_ok());

    assert!(state.map_surface(surface_id));
    assert!(state.set_mapping_geom(surface_id, 4, 3, MTL_FORMAT_BGRA8_UNORM));
    state
        .surfaces
        .mappings
        .get_mut(&surface_id)
        .unwrap()
        .pages
        .entries = vec![1];

    // An unrelated surface, stored twice, so the device-global generation
    // counter is past its first value before this one is stored. Without it a
    // `generation == host_gen` assertion would also hold for any producer that
    // hardcoded 1, and 1 is exactly what a broken one would most likely emit.
    assert!(state.map_surface(72));
    for _ in 0..2 {
        crate::runtime::surface_cache::store(&mut state, 72, 2, 2, vec![7u8; 2 * 2 * 4]);
    }

    // Blue frame (BGRA), then the same surface repainted red.
    crate::runtime::surface_cache::store(
        &mut state,
        surface_id,
        4,
        3,
        [255u8, 0, 0, 255].repeat(4 * 3),
    );

    let resolve = |state: &mut Device, host: &mut FakeHost| {
        let (_, _, _, src) =
            resolve_sampled_source(state, host, 1, texture_ref, None, true, sampled_d2_shape())
                .expect("host-cache rung must serve the stored frame");
        let SampledSourceRequest::Bytes(bytes, identity, layout, _) = src else {
            panic!("cache-backed fixture unexpectedly resolved a resident target");
        };
        // The cache holds BGRA8 and the upload declares BGRA8, so the bytes go
        // up untouched. Asserting the pair together is the point: either one
        // alone permits the channel-swapped frame that the other rules out.
        assert_eq!(
            layout.layout(),
            TexelLayout::Bgra8,
            "the scanout cache's bytes are BGRA8; declaring RGBA8 samples R and B swapped"
        );
        (bytes, identity)
    };

    let (first_bytes, first_id) = resolve(&mut state, &mut host);
    let first_id = first_id.expect("the host-cache rung must offer a content identity");
    assert_eq!(
        &first_bytes[..4],
        &[255u8, 0, 0, 255],
        "the stored BGRA bytes must reach the engine verbatim, not channel-swapped"
    );
    assert!(
        std::sync::Arc::ptr_eq(
            &first_bytes,
            &crate::runtime::surface_cache::get_shared_with_gen(&state, surface_id, 4, 3)
                .expect("stored")
                .0
        ),
        "the rung must hand over the cache's own allocation; a fresh Vec here is \
         the full-frame copy this rail exists to avoid"
    );
    assert_eq!(
        first_id.key,
        (1u64 << 62) | surface_id as u64,
        "bit 62 alone is the IOSurface texture host-cache namespace"
    );
    let stored_gen = crate::runtime::surface_cache::get_shared_with_gen(&state, surface_id, 4, 3)
        .expect("the frame just stored must be readable")
        .1;
    assert_eq!(
        first_id.generation, stored_gen,
        "the identity must carry the cache entry's own host_gen, not a private counter"
    );

    // Re-resolving an untouched cache must repeat the identity exactly — that
    // repetition is what the engine reads as "these are the bytes you hold".
    let (again_bytes, again_id) = resolve(&mut state, &mut host);
    assert_eq!(
        again_id,
        Some(first_id),
        "an untouched cache entry must keep its identity, or nothing is ever elided"
    );
    assert_eq!(
        again_bytes, first_bytes,
        "identity repeated but bytes moved"
    );

    // Repaint. The bytes change, so the identity must too.
    crate::runtime::surface_cache::store(
        &mut state,
        surface_id,
        4,
        3,
        [0u8, 0, 255, 255].repeat(4 * 3),
    );
    let (second_bytes, second_id) = resolve(&mut state, &mut host);
    let second_id = second_id.expect("a rewritten cache entry is still identifiable");
    assert_ne!(
        second_bytes, first_bytes,
        "fixture is not exercising the property: the repaint did not reach the sample"
    );
    assert_eq!(
        second_id.key, first_id.key,
        "the same surface must keep one key across repaints"
    );
    assert_ne!(
        second_id.generation, first_id.generation,
        "a repainted frame under an unchanged identity binds the stale texture forever"
    );
}

/// Live Safari app-launch class: the surface backing base carries an unknown
/// 2-byte IOSurface FourCC (`LA08`) while the IOSurface plane view descriptor carries
/// the exact RG8 Metal view. Defaulting the base to BGRA asks for a
/// 632-byte row against the wire's 320-byte row and drops the draw.
#[test]
fn iosurface_plane_view_sample_uses_serialized_rg8_view_over_unknown_surface_fourcc() {
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
    use crate::runtime::gva_mem;
    use reims_vgpu_core::endian::{st16, st32, st64};
    use reims_vgpu_core::pixel_format::MTL_FORMAT_RG8_UNORM;
    use reims_vgpu_paging::geometry::{
        DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN, MAPPER_PAGE_ENTRY_PFN_SHIFT as PAGE_ENTRY_PFN_SHIFT,
        MAPPER_PAGE_ENTRY_VALID as PAGE_ENTRY_VALID,
    };
    use reims_vgpu_protocol::{
        DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_BPE, DEVICE_DESC_BPR, DEVICE_DESC_DIMS,
        DEVICE_DESC_LEN, DEVICE_DESC_PIXEL_FORMAT,
    };

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();

    // One-level x86 GVA table for the object list and IOSurface plane view descriptor.
    let dir_pfn = 2u32;
    let root_pfn = 3u32;
    let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_X86;
    let root_gpa = (root_pfn as u64) << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    let mut dir = [0u8; 8];
    st32(&mut dir[DIRECTORY_ROOT_PFN as usize..], root_pfn);
    st32(&mut dir[DIRECTORY_DEPTH as usize..], 1);
    assert!(host.write_gpa(dir_gpa, &dir).is_ok());
    for i in 0..3u32 {
        let pfn = 4 + i;
        host.map_range((pfn as u64) << PAGE_SHIFT_X86, 0x1000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        assert!(host.write_gpa(root_gpa + (i as u64) * 4, &pte).is_ok());
    }
    state.define_task(1, 0x1000, dir_pfn);
    assert!(state.set_object_list(1, 0, 256));

    let texture_ref = 248u32;
    let surface_id = 9u32;
    let width = 158u32;
    let height = 154u32;
    let surface_bpr = 320u32;
    let desc_gva = 0x1000u64;
    let built = reims_vgpu_wire::device_desc::IOSurfacePlaneViewBuilder::new(
        surface_id,
        0,
        texture_ref,
        reims_vgpu_wire::device_desc::IOSURFACE_PLANE_VIEW_RECORD_TAG_PLANE,
    )
    .geometry(MTL_FORMAT_RG8_UNORM, width, height, 1);
    let desc = built.bytes();
    let desc_len = desc.len();
    assert!(
        gva_mem::write_task_gva(&mut host, &state.tasks[1], desc_gva, desc, PAGE_SHIFT_X86,)
            .is_ok()
    );
    let list_off = list_object_entry_offset(texture_ref, 256).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((desc_len as u32) << 8);
    st32(&mut list_entry, packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        list_off,
        &list_entry,
        PAGE_SHIFT_X86,
    )
    .is_ok());

    // Exact live geometry: 13 x86 pages, 320-byte rows, two bytes/texel.
    let page = 1u64 << PAGE_SHIFT_X86;
    let page_count = 13u32;
    let gpa0 = 0x5100_0000u64;
    host.map_range(gpa0, (page * page_count as u64) as usize, 0);
    let mut native = vec![0u8; (surface_bpr * height) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let off = y * surface_bpr as usize + x * 2;
            native[off] = (x % 251) as u8 + 1;
            native[off + 1] = (y % 251) as u8 + 1;
        }
    }
    assert!(host.write_gpa(gpa0, &native).is_ok());

    assert!(state.map_surface(surface_id));
    {
        let m = state.surfaces.mappings.get_mut(&surface_id).unwrap();
        m.lifecycle.active = true;
        m.pages.entries = (0..page_count)
            .map(|i| {
                let pfn = ((gpa0 >> PAGE_SHIFT_X86) as u32) + i;
                (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID
            })
            .collect();
    }
    assert!(state.set_mapping_geom(surface_id, width, height, 0));
    let mut device_desc = vec![0u8; DEVICE_DESC_LEN];
    st32(&mut device_desc[DEVICE_DESC_PIXEL_FORMAT..], 0x4c41_3038);
    st32(
        &mut device_desc[DEVICE_DESC_ALLOC_SIZE..],
        (page * page_count as u64) as u32,
    );
    st64(
        &mut device_desc[DEVICE_DESC_DIMS..],
        ((width as u64) << 8) | ((height as u64) << 40),
    );
    st32(&mut device_desc[DEVICE_DESC_BPR..], surface_bpr);
    st16(&mut device_desc[DEVICE_DESC_BPE..], 2);
    assert!(state.set_mapping_device_desc(surface_id, &device_desc));

    let (sample_w, sample_h, sample_mid, sampled) = resolve_sampled_source(
        &mut state,
        &mut host,
        1,
        texture_ref,
        None,
        true,
        sampled_d2_shape(),
    )
    .expect("serialized RG8 view must sample the 2-byte surface");
    assert_eq!(
        (sample_w, sample_h, sample_mid),
        (width, height, surface_id)
    );
    let SampledSourceRequest::Bytes(sampled, _, byte_format, _) = sampled else {
        panic!("serialized view unexpectedly resolved a resident target");
    };
    // Native RG8 upload: two bytes per texel, tight rows (an R8G8_UNORM
    // Vulkan image samples these identically to the old CPU (r,g,0,255)
    // RGBA8 expansion).
    assert_eq!(byte_format.layout(), TexelLayout::Rg8);
    assert_eq!(sampled.len(), (width * height * 2) as usize);
    assert_eq!(&sampled[..4], &[1, 1, 2, 1]);
    let last = ((height - 1) as usize * width as usize + (width - 1) as usize) * 2;
    assert_eq!(
        &sampled[last..last + 2],
        &[158, 154],
        "row padding must not enter the RG8 view"
    );
}

/// The IOSurface plane view view memo: unchanged plane bytes reuse the converted Arc and
/// carry a stable content identity (engine upload skipped); a guest write
/// to the plane is observed on the next bind and mints a new generation.
#[test]
fn iosurface_plane_view_memo_reuses_unchanged_planes_and_invalidates_on_write() {
    use reims_vgpu_core::endian::{st16, st32, st64};
    use reims_vgpu_core::pixel_format::MTL_FORMAT_RG8_UNORM;
    use reims_vgpu_paging::geometry::{
        MAPPER_PAGE_ENTRY_PFN_SHIFT as PAGE_ENTRY_PFN_SHIFT,
        MAPPER_PAGE_ENTRY_VALID as PAGE_ENTRY_VALID,
    };
    use reims_vgpu_protocol::{
        DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_BPE, DEVICE_DESC_BPR, DEVICE_DESC_DIMS,
        DEVICE_DESC_LEN, DEVICE_DESC_PIXEL_FORMAT,
    };

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let surface_id = 9u32;
    let width = 158u32;
    let height = 154u32;
    let surface_bpr = 320u32;
    let page = 1u64 << PAGE_SHIFT_X86;
    let page_count = 13u32;
    let gpa0 = 0x5100_0000u64;
    host.map_range(gpa0, (page * page_count as u64) as usize, 0);
    let mut native = vec![0u8; (surface_bpr * height) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let off = y * surface_bpr as usize + x * 2;
            native[off] = (x % 251) as u8 + 1;
            native[off + 1] = (y % 251) as u8 + 1;
        }
    }
    assert!(host.write_gpa(gpa0, &native).is_ok());
    assert!(state.map_surface(surface_id));
    {
        let m = state.surfaces.mappings.get_mut(&surface_id).unwrap();
        m.lifecycle.active = true;
        m.pages.entries = (0..page_count)
            .map(|i| {
                let pfn = ((gpa0 >> PAGE_SHIFT_X86) as u32) + i;
                (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID
            })
            .collect();
    }
    assert!(state.set_mapping_geom(surface_id, width, height, 0));
    let mut device_desc = vec![0u8; DEVICE_DESC_LEN];
    st32(&mut device_desc[DEVICE_DESC_PIXEL_FORMAT..], 0x4c41_3038);
    st32(
        &mut device_desc[DEVICE_DESC_ALLOC_SIZE..],
        (page * page_count as u64) as u32,
    );
    st64(
        &mut device_desc[DEVICE_DESC_DIMS..],
        ((width as u64) << 8) | ((height as u64) << 40),
    );
    st32(&mut device_desc[DEVICE_DESC_BPR..], surface_bpr);
    st16(&mut device_desc[DEVICE_DESC_BPE..], 2);
    assert!(state.set_mapping_device_desc(surface_id, &device_desc));
    let view = objects::IOSurfacePlaneViewDescriptor {
        pixel_format: MTL_FORMAT_RG8_UNORM,
        width,
        height,
        depth: 1,
        plane_index: 0,
    };

    let (w1, h1, rgba1, id1, fmt1) =
        load_iosurface_plane_view_rgba(&mut state, &mut host, 1, 248, surface_id, view)
            .expect("first materialization");
    assert_eq!((w1, h1), (width, height));
    assert_eq!(
        fmt1.layout(),
        TexelLayout::Rg8,
        "an RG8 chroma plane uploads at native footprint, not CPU-expanded to RGBA8"
    );
    // Native footprint: two bytes per texel, tight rows (no RGBA8 expand).
    assert_eq!(rgba1.len(), (width * height * 2) as usize);
    assert_ne!(
        id1.generation, 0,
        "0 means no host content yet; every real store takes a fresh generation"
    );
    assert_eq!(
        id1.key,
        (1u64 << 63) | surface_id as u64,
        "identity key namespaces IOSurface plane view content above GVA keys"
    );

    let (_, _, rgba2, id2, _) =
        load_iosurface_plane_view_rgba(&mut state, &mut host, 1, 248, surface_id, view)
            .expect("memo revalidation");
    assert!(
        std::sync::Arc::ptr_eq(&rgba1, &rgba2),
        "unchanged plane bytes must reuse the native allocation"
    );
    assert_eq!(id1, id2, "unchanged content keeps its identity");

    // Guest CPU writes one texel; the next bind must observe it.
    assert!(host.write_gpa(gpa0 + 6, &[0xAA, 0xBB]).is_ok());
    let (_, _, rgba3, id3, _) =
        load_iosurface_plane_view_rgba(&mut state, &mut host, 1, 248, surface_id, view)
            .expect("re-materialization after guest write");
    assert!(
        id3.generation > id2.generation,
        "guest write mints a new generation"
    );
    // Native RG8: texel 3 sits at tight byte offset 3*2 = 6 (an R8G8_UNORM
    // Vulkan image samples this to (0xAA/255, 0xBB/255, 0, 1), identical to
    // the CPU-expanded (0xAA,0xBB,0,255) RGBA8 the old path produced).
    assert_eq!(
        &rgba3[6..8],
        &[0xAA, 0xBB],
        "the new native plane bytes must be observed"
    );
    assert!(!std::sync::Arc::ptr_eq(&rgba1, &rgba3));
}

#[test]
fn iosurface_plane_view_materializes_only_when_base_identity_differs() {
    use reims_vgpu_core::pixel_format::MTL_FORMAT_RG8_UNORM;

    let exact = objects::IOSurfacePlaneViewDescriptor {
        pixel_format: MTL_FORMAT_BGRA8_UNORM,
        width: 1920,
        height: 1080,
        depth: 1,
        plane_index: 0,
    };
    assert!(!iosurface_plane_view_requires_materialization(
        true,
        1920,
        1080,
        MTL_FORMAT_BGRA8_UNORM,
        exact
    ));
    assert!(iosurface_plane_view_requires_materialization(
        true, 1920, 1080, 0, exact
    ));

    let rg8_view = objects::IOSurfacePlaneViewDescriptor {
        pixel_format: MTL_FORMAT_RG8_UNORM,
        width: 158,
        height: 154,
        depth: 1,
        plane_index: 0,
    };
    assert!(iosurface_plane_view_requires_materialization(
        true,
        158,
        154,
        MTL_FORMAT_BGRA8_UNORM,
        rg8_view
    ));
    assert!(iosurface_plane_view_requires_materialization(
        false,
        158,
        154,
        MTL_FORMAT_RG8_UNORM,
        rg8_view
    ));
    let volume = objects::IOSurfacePlaneViewDescriptor { depth: 2, ..exact };
    assert!(iosurface_plane_view_requires_materialization(
        true,
        1920,
        1080,
        MTL_FORMAT_BGRA8_UNORM,
        volume
    ));
}

#[test]
fn texture_view_declines_are_specific_and_log_safe() {
    let cases = [
        TextureViewDecline::HopEntryMissing { texture_ref: 1 },
        TextureViewDecline::HopObjectNotView {
            texture_ref: 1,
            object_type: ObjectKind::Texture,
        },
        TextureViewDecline::HopDescriptorMissing {
            texture_ref: 1,
            descriptor_length: 4,
        },
        TextureViewDecline::HopDecode {
            texture_ref: 1,
            opcode: 9,
            declared: 4,
            descriptor_len: 4,
            bytes_hex: "01020304".into(),
            reason: DecodeStatus::ErrShort("res_texture_view_short"),
        },
        TextureViewDecline::HopZeroBase {
            texture_ref: 1,
            opcode: 9,
        },
        TextureViewDecline::ChainLevelOverflow { texture_ref: 1 },
        TextureViewDecline::ChainSliceOverflow { texture_ref: 1 },
        TextureViewDecline::ChainLevelOutOfRange {
            texture_ref: 1,
            outer_base: 2,
            outer_count: 3,
            inner_count: 4,
        },
        TextureViewDecline::ChainSliceOutOfRange {
            texture_ref: 1,
            outer_base: 2,
            outer_count: 3,
            inner_count: 4,
        },
        TextureViewDecline::HopTextureTypeUnsupported {
            texture_ref: 1,
            texture_type: u16::MAX,
        },
        TextureViewDecline::HopSwizzleInvalid {
            texture_ref: 1,
            selectors: [0, 1, 2, 9],
        },
        TextureViewDecline::ChainSelfOrZero {
            base: 1,
            next: 1,
            depth: 1,
        },
        TextureViewDecline::ChainOverflow { base: 1, depth: 8 },
    ];
    let mut slugs = std::collections::HashSet::new();
    for decline in cases {
        assert!(slugs.insert(decline.slug()), "duplicate {}", decline.slug());
        for (_, value) in decline.fields() {
            assert!(!value.contains(char::is_whitespace));
        }
    }
    assert_eq!(slugs.len(), 13);
}

#[test]
fn texture_view_decline_preserves_decode_leaf_and_chain_identity() {
    let decode = TextureViewDecline::HopDecode {
        texture_ref: 7,
        opcode: 9,
        declared: 12,
        descriptor_len: 8,
        bytes_hex: "01020304".into(),
        reason: DecodeStatus::ErrShort("res_texture_view_short"),
    };
    assert_eq!(decode.slug(), "res_texture_view_short");
    let fields = decode.fields();
    assert!(fields.contains(&("texture_ref", "7".into())));
    assert!(fields.contains(&("opcode", "0x9".into())));
    assert!(fields.contains(&("declared", "12".into())));
    assert!(fields.contains(&("descriptor_len", "8".into())));
    assert!(fields.contains(&("bytes", "01020304".into())));

    let chain = TextureViewDecline::ChainSelfOrZero {
        base: 11,
        next: 11,
        depth: 3,
    };
    assert_eq!(chain.slug(), "texture_view_chain_self_or_zero");
    let fields = chain.fields();
    assert!(fields.contains(&("base", "11".into())));
    assert!(fields.contains(&("next", "11".into())));
    assert!(fields.contains(&("depth", "3".into())));
}

/// Every IOSurface plane view view refusal names its rail (`iosurface_plane_view_`), renders
/// whitespace-free fields, and is distinct — the same discipline the
/// capture and import rails took, so `grep reason=iosurface_plane_view_…` stays
/// answerable against the blit rail's `t5_*` copy vocabulary next door.
#[test]
fn every_iosurface_plane_view_reason_is_namespaced_distinct_and_log_safe() {
    use crate::observe::Decline as _;
    const ALL: &[IOSurfacePlaneViewDecline] = &[
        IOSurfacePlaneViewDecline::UnsupportedDepth { depth: 0 },
        IOSurfacePlaneViewDecline::Unresolved,
        IOSurfacePlaneViewDecline::FormatBpp,
        IOSurfacePlaneViewDecline::NoMapping,
        IOSurfacePlaneViewDecline::SampleWindow {
            base_w: 0,
            base_h: 0,
            base_fmt: 0,
            desc: None,
        },
        IOSurfacePlaneViewDecline::Span {
            pages: 0,
            page_bytes: 0,
            span_end: 0,
            bpr: 0,
        },
        IOSurfacePlaneViewDecline::TightOverflow { bpp: 0 },
        IOSurfacePlaneViewDecline::NativeLen { tight: 0 },
        IOSurfacePlaneViewDecline::Read {
            base_w: 0,
            base_h: 0,
            base_fmt: 0,
            off: 0,
            bpr: 0,
            span_end: 0,
            pages: 0,
        },
        IOSurfacePlaneViewDecline::RgbaStride,
        IOSurfacePlaneViewDecline::RgbaLen { stride: 0 },
        IOSurfacePlaneViewDecline::Convert { row: 0, bpp: 0 },
    ];
    let mut slugs: Vec<&str> = Vec::new();
    for d in ALL {
        assert!(
            d.slug().starts_with("iosurface_plane_view_"),
            "{} is not namespaced to the IOSurface plane view view rail",
            d.slug()
        );
        for (k, v) in d.fields() {
            assert!(!k.contains(' ') && !v.contains(' '), "{k}={v}");
        }
        slugs.push(d.slug());
    }
    slugs.sort_unstable();
    let before = slugs.len();
    slugs.dedup();
    assert_eq!(
        before,
        slugs.len(),
        "duplicate IOSurfacePlaneViewDecline slug"
    );
}

/// `SampleWindow` is the only variant carrying transcribed field logic: the
/// base geometry plus the decoded device descriptor, or `desc=missing` when
/// the descriptor could not be decoded. Both branches must render exactly
/// what the old ad-hoc `detail` string did.
#[test]
fn sample_window_renders_the_descriptor_or_its_absence() {
    let present = IOSurfacePlaneViewDecline::SampleWindow {
        base_w: 320,
        base_h: 240,
        base_fmt: 0x50,
        desc: Some((64, 64, 0x4c41_3038, 256, 4096)),
    };
    assert_eq!(
            crate::observe::Emit::decline("iosurface_plane_view_draw_view", &present).render(),
            "iosurface_plane_view_draw_view reason=iosurface_plane_view_sample_window base=320x240 base_fmt=0x50 desc=64x64 desc_fmt=0x4c413038 bpr=256 alloc=4096"
        );

    let missing = IOSurfacePlaneViewDecline::SampleWindow {
        base_w: 320,
        base_h: 240,
        base_fmt: 0x50,
        desc: None,
    };
    assert_eq!(
        crate::observe::Emit::decline("iosurface_plane_view_draw_view", &missing).render(),
        "iosurface_plane_view_draw_view reason=iosurface_plane_view_sample_window base=320x240 base_fmt=0x50 desc=missing"
    );
}

/// A secondary MRT attachment binds **its own** blend, not slot 0's and not
/// "unblended because secondaries are always masks".
///
/// The regression this locks: `caches.rs` forced every secondary attachment
/// `blend_enable(false)`, justified by a comment claiming the decode side
/// carried no per-attachment blend state. It carried it all along — the
/// that asked to blend slot 1 got a raw store instead.
#[test]
fn a_secondary_mrt_slot_binds_its_own_blend() {
    use crate::runtime::decode::resource::{PipelineColorAttachment, RenderPipelineDescriptor};
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);

    // Slot 0 blends src-alpha over; slot 1 blends ONE/ONE (additive). Two
    // different blends so borrowing slot 0's would be visible.
    let pipeline = RenderPipelineDescriptor {
        color_attachments: vec![
            PipelineColorAttachment {
                slot: 0,
                blending_enabled: true,
                src_rgb: 4, // MTLBlendFactorSourceAlpha
                dst_rgb: 5, // MTLBlendFactorOneMinusSourceAlpha
                op_rgb: 0,  // MTLBlendOperationAdd
                src_alpha: 4,
                dst_alpha: 5,
                op_alpha: 0,
                ..PipelineColorAttachment::default()
            },
            PipelineColorAttachment {
                slot: 1,
                blending_enabled: true,
                src_rgb: 1, // MTLBlendFactorOne
                dst_rgb: 1, // MTLBlendFactorOne
                op_rgb: 0,
                src_alpha: 1,
                dst_alpha: 1,
                op_alpha: 0,
                ..PipelineColorAttachment::default()
            },
        ],
        ..RenderPipelineDescriptor::default()
    };

    let colors = vec![
        ColorRtRequest {
            slot: 0,
            texture_ref: 10,
            storage: linear_target_storage(0x1000, 64 * 4, 64),
            width: 64,
            height: 64,
            format: MTL_FORMAT_BGRA8_UNORM,
            ..ColorRtRequest::default()
        },
        ColorRtRequest {
            slot: 1,
            texture_ref: 11,
            storage: linear_target_storage(0x2000, 64 * 4, 64),
            width: 64,
            height: 64,
            format: reims_vgpu_core::pixel_format::MTL_FORMAT_RG16_FLOAT,
            ..ColorRtRequest::default()
        },
    ];
    let primary = crate::model::TargetIdentity::Gva {
        gva: 0x1000,
        width: 64,
        height: 64,
        generation: 0,
        format: reims_vgpu_core::pixel_format::TexelLayout::Rgba8,
    };

    let mut host = crate::runtime::host::FakeHost::new();
    let blend_states = [(
        1,
        semantic_blend_state(&pipeline.color_attachments[1])
            .expect("the fixture declares a valid slot-1 blend state"),
    )];
    let secs = build_secondary_targets(
        &mut state,
        &mut host,
        1,
        &colors,
        &pipeline,
        &primary,
        &blend_states,
    )
    .expect("a contiguous, resolvable secondary builds");
    assert_eq!(secs.len(), 1, "one secondary attachment expected");
    let blend = secs[0].blend.expect(
        "slot 1 declares blending_enabled — before this fix every secondary \
             was forced unblended",
    );
    use reims_vgpu_core::{BlendFactor, BlendOp};
    assert_eq!(blend.src_color, BlendFactor::One, "slot 1's own src factor");
    assert_eq!(blend.dst_color, BlendFactor::One, "slot 1's own dst factor");
    assert_eq!(blend.color_op, BlendOp::Add);
    // The tell that it is not slot 0's: slot 0 asked for SrcAlpha/OneMinusSrcAlpha.
    assert_ne!(blend.src_color, BlendFactor::SrcAlpha);

    // A slot the pipeline does not blend stays unblended rather than
    // inheriting slot 0's — there is no `or_else(first())` fallback here.
    let unblended = RenderPipelineDescriptor {
        color_attachments: vec![
            pipeline.color_attachments[0],
            PipelineColorAttachment {
                slot: 1,
                blending_enabled: false,
                ..PipelineColorAttachment::default()
            },
        ],
        ..RenderPipelineDescriptor::default()
    };
    let mut host = crate::runtime::host::FakeHost::new();
    let secs =
        build_secondary_targets(&mut state, &mut host, 1, &colors, &unblended, &primary, &[])
            .expect("the same secondary builds whether or not its slot blends");
    assert_eq!(secs.len(), 1);
    assert!(
        secs[0].blend.is_none(),
        "slot 1 declares no blend; it must not inherit slot 0's"
    );
    let _ = &mut state;
}

/// A secondary colour attachment this device cannot build refuses the draw.
///
/// Every arm here used to `return Vec::new()`, which the caller could not tell
/// from the `Vec::new()` that means "the guest declared one render target" — so
/// it took the single-RT path and **executed the draw**. The guest asked for two
/// attachments and got one, its fragment shader's `location` 1 output went
/// nowhere, and a later pass sampling that attachment read whatever was in those
/// pages before. Nothing the guest can observe distinguished that from a draw it
/// had only ever asked one target for.
///
/// The last case is the one that makes the rest meaningful: a genuine single-RT
/// draw must still be `Ok(empty)`, or "refuse when you cannot build it" would
/// have been bought by refusing draws that were never MRT at all.
#[test]
fn an_unbuildable_secondary_refuses_the_draw_rather_than_dropping_to_single_rt() {
    use crate::runtime::census::present_proxy::MrtDrop;
    use crate::runtime::decode::resource::{PipelineColorAttachment, RenderPipelineDescriptor};

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let pipeline = RenderPipelineDescriptor {
        color_attachments: vec![
            PipelineColorAttachment {
                slot: 0,
                ..PipelineColorAttachment::default()
            },
            PipelineColorAttachment {
                slot: 1,
                ..PipelineColorAttachment::default()
            },
        ],
        ..RenderPipelineDescriptor::default()
    };
    let primary = crate::model::TargetIdentity::Gva {
        gva: 0x1000,
        width: 64,
        height: 64,
        generation: 0,
        format: reims_vgpu_core::pixel_format::TexelLayout::Rgba8,
    };
    let slot0 = ColorRtRequest {
        slot: 0,
        texture_ref: 10,
        storage: linear_target_storage(0x1000, 64 * 4, 64),
        width: 64,
        height: 64,
        format: MTL_FORMAT_BGRA8_UNORM,
        ..ColorRtRequest::default()
    };
    // A secondary that builds, so each case below differs from it in exactly the
    // one field its reason names.
    let good_slot1 = ColorRtRequest {
        slot: 1,
        texture_ref: 11,
        storage: linear_target_storage(0x2000, 64 * 4, 64),
        width: 64,
        height: 64,
        format: reims_vgpu_core::pixel_format::MTL_FORMAT_RG16_FLOAT,
        ..ColorRtRequest::default()
    };

    let mut build = |slot1: &ColorRtRequest| {
        let mut host = crate::runtime::host::FakeHost::new();
        build_secondary_targets(
            &mut state,
            &mut host,
            1,
            &[slot0.clone(), slot1.clone()],
            &pipeline,
            &primary,
            &[],
        )
    };

    // The control: this exact list builds, so every refusal below is caused by
    // the one field it changes and not by the fixture.
    assert!(
        build(&good_slot1).is_ok(),
        "the unmodified fixture must build, or the cases below prove nothing"
    );
    let mismatched = build(&ColorRtRequest {
        width: 32,
        ..good_slot1.clone()
    })
    .expect("Metal permits attachments with different geometry");
    assert_eq!(mismatched.len(), 1);
    assert_eq!((mismatched[0].width, mismatched[0].height), (32, 64));

    for (reason, slot1) in [
        (
            // Slot 2 where the render pass maps location 1 → attachment 1.
            MrtDrop::NonContiguousSlot,
            ColorRtRequest {
                slot: 2,
                ..good_slot1.clone()
            },
        ),
        (
            // No engine mapping, so the layout would have to be guessed.
            MrtDrop::UnknownFormat,
            ColorRtRequest {
                format: 0xfff0,
                ..good_slot1.clone()
            },
        ),
        (
            // Neither a linear GVA nor a surface mapping names a resident.
            MrtDrop::NoIdentity,
            ColorRtRequest {
                storage: ColorTargetStorage::None,
                ..good_slot1.clone()
            },
        ),
        (
            // Resolves to the primary's own resident: a pass that reads and
            // writes one image through two attachments.
            MrtDrop::AliasesPrimary,
            ColorRtRequest {
                storage: linear_target_storage(0x1000, 64 * 4, 64),
                ..good_slot1.clone()
            },
        ),
    ] {
        let refusal = build(&slot1).expect_err(&format!(
            "{reason:?} must refuse the draw, not drop to single-RT"
        ));
        assert_eq!(
            refusal.reason, reason,
            "the refusal names the check that bailed"
        );
        assert_eq!(
            refusal.slot, slot1.slot,
            "the refusal names the guest's own slot number, so a reader knows \
             which attachment of the list failed"
        );
    }

    // A guest that declared one render target is not an MRT draw and must not be
    // refused: `Ok(empty)` is the classic single-RT path, byte-identical.
    let mut host = crate::runtime::host::FakeHost::new();
    let single = build_secondary_targets(
        &mut state,
        &mut host,
        1,
        std::slice::from_ref(&slot0),
        &pipeline,
        &primary,
        &[],
    )
    .expect("a single-attachment draw is not a refusal");
    assert!(
        single.is_empty(),
        "one colour attachment produces no secondaries"
    );
}

/// Two attachments over one destination are refused wherever the pair sits, not
/// only when one of them is slot 0.
///
/// A pass that writes one image through two attachments has no correct
/// rendering, which is why the primary case is a refusal rather than a silent
/// drop. The same is true of two secondaries, and the check used to name
/// `primary` alone — so slots 1 and 2 over one span were admitted and drawn.
#[test]
fn two_secondaries_over_one_destination_refuse_the_draw_like_a_primary_alias() {
    use crate::runtime::census::present_proxy::MrtDrop;
    use crate::runtime::decode::resource::{PipelineColorAttachment, RenderPipelineDescriptor};
    use reims_vgpu_core::pixel_format::{MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_RG16_FLOAT};

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let pipeline = RenderPipelineDescriptor {
        color_attachments: (0..3)
            .map(|slot| PipelineColorAttachment {
                slot,
                ..PipelineColorAttachment::default()
            })
            .collect(),
        ..RenderPipelineDescriptor::default()
    };
    let primary = crate::model::TargetIdentity::Gva {
        gva: 0x1000,
        width: 64,
        height: 64,
        generation: 0,
        format: reims_vgpu_core::pixel_format::TexelLayout::Rgba8,
    };
    let rt = |slot: u32, texture_ref: u32, target_gva: u64, format: u16| ColorRtRequest {
        slot,
        texture_ref,
        storage: linear_target_storage(target_gva, 64 * 4, 64),
        width: 64,
        height: 64,
        format,
        ..ColorRtRequest::default()
    };
    let mut build = |colors: &[ColorRtRequest]| {
        let mut host = crate::runtime::host::FakeHost::new();
        build_secondary_targets(&mut state, &mut host, 1, colors, &pipeline, &primary, &[])
    };

    // The control: three distinct destinations build, so the refusal below is
    // caused by the aliasing address and not by the third attachment existing.
    let distinct = [
        rt(0, 10, 0x1000, MTL_FORMAT_BGRA8_UNORM),
        rt(1, 11, 0x2000, MTL_FORMAT_RG16_FLOAT),
        rt(2, 12, 0x3000, MTL_FORMAT_RG16_FLOAT),
    ];
    assert_eq!(
        build(&distinct).expect("three distinct spans build").len(),
        2,
        "the control must build both secondaries, or the case below proves nothing"
    );

    // Slots 1 and 2 name one span. Neither is the primary, so the old check saw
    // nothing.
    let aliased = [
        rt(0, 10, 0x1000, MTL_FORMAT_BGRA8_UNORM),
        rt(1, 11, 0x2000, MTL_FORMAT_RG16_FLOAT),
        rt(2, 12, 0x2000, MTL_FORMAT_RG16_FLOAT),
    ];
    let refusal = build(&aliased).expect_err("two secondaries over one span is not renderable");
    assert_eq!(refusal.slot, 2, "the second of the pair is the one refused");
    assert_eq!(refusal.reason, MrtDrop::AliasesPrimary);
}

/// Recycled texture_ref must not serve a prior full-frame encode as a
/// different-sized linear sample (namespace / geom-match class).
#[test]
fn texture_ref_cache_geom_mismatch_does_not_hit_get_texture() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let tex_ref = 53u32;
    let mut full = vec![0u8; 1920 * 1152 * 4];
    full[3] = 255;
    host_cache_store_rgba8(&mut state, 0, tex_ref, 1920, 1152, &full);
    // Exact geom hit
    assert!(crate::runtime::surface_cache::get_texture(&state, 0, tex_ref, 1920, 1152).is_some());
    // Wrong geom (type-3 L0 recycle) miss
    assert!(crate::runtime::surface_cache::get_texture(&state, 0, tex_ref, 115, 16).is_none());
    // surface_id map must stay empty for texture_ref stores
    assert!(crate::runtime::surface_cache::get(&state, tex_ref, 1920, 1152).is_none());
}

/// A guest-run host pointer is only valid when the host has declared
/// `map_pages` views stable. Arm64 MMIO remap views are transient, so the
/// runtime must decline to produce runs at all rather than hand the engine a
/// pointer whose view can be released before the submission gathers from it.
///
/// The two arms differ **only** in `stable_map_pages`, so the walkable arm is
/// the control: without it this would pass on a build that declined every span
/// for some unrelated reason, which is exactly how a decline test goes hollow.
#[test]
fn guest_runs_decline_on_unstable_host_mappings() {
    use reims_vgpu_core::endian::st32;
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    let page_shift = crate::model::PAGE_SHIFT_X86;
    let page = 1u64 << page_shift;
    let gva = 8u64;

    // A task whose page table really does resolve `[gva, gva+16)` onto `data0`.
    let walkable =
        |stable: bool| -> Result<Vec<reims_vgpu_memory::GuestRun>, super::WindowRefusal> {
            let mut host = FakeHost::new();
            host.strict_linux_map = true;
            host.stable_map_pages = stable;
            let (dir_gpa, root_gpa, data0) =
                (2u64 << page_shift, 3u64 << page_shift, 4u64 << page_shift);
            for gpa in [dir_gpa, root_gpa, data0] {
                host.map_range(gpa, page as usize, 0);
            }
            let mut d = [0u8; 8];
            st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
            st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
            host.write_gpa(dir_gpa, &d).unwrap();
            let mut pte = [0u8; 4];
            st32(&mut pte, 4);
            host.write_gpa(root_gpa, &pte).unwrap();

            let mut state = Device::new(DeviceId(1), page_shift);
            state.define_task(1, page, 2);
            task_gva_guest_run_window(&state, &mut host, 1, gva, 16).map(|(_, runs)| runs)
        };

    assert!(
        walkable(true).is_ok_and(|runs| !runs.is_empty()),
        "control: a host promising a stable alias resolves this span"
    );
    // Named, not merely absent: this refusal is now counted, and it must land
    // in the route that says the host would not promise the alias rather than
    // in one of the two that report a page table the guest wrote.
    assert_eq!(
        walkable(false).err(),
        Some(super::WindowRefusal::NoAlias),
        "the same span must yield no runs when the host will not promise the \
         alias outlives the submission that gathers from it"
    );
}

/// A resource-sized stable alias remains admissible when the optional
/// whole-RAMBlock map refuses. The two imports have different contract
/// lifetimes and different sizes; coupling them made unrelated guest RAM a
/// side channel for whether this resource could be zero-copy.
#[test]
fn a_packed_alias_is_independent_of_the_whole_ram_map() {
    let page_shift = PAGE_SHIFT_X86;
    let page = 1u64 << page_shift;
    // A host that reports no guest RAM at all: the optional whole map resolves
    // to a standing refusal even though the backend supports resource imports.
    let mut host = FakeHost::new();
    host.stable_map_pages = true;
    crate::runtime::guest_ram_map::reset();
    crate::runtime::guest_ram::latch_import_limits(page, 1 << 30, 1 << 30);
    assert!(
        crate::runtime::guest_ram_map::standing_refusal(&mut host).is_some(),
        "the premise is a map that refused"
    );
    assert_eq!(
        crate::runtime::guest_ram::host_allocation_import_align(page),
        Ok(page),
        "a legal resource allocation must not inherit the whole-map refusal"
    );
    // The three answers are the complete contract for this one allocation.
    assert!(crate::runtime::guest_ram::granularity().is_some());
    assert!(crate::runtime::guest_ram::import_span_max().is_some());
    assert!(crate::runtime::guest_ram::import_budget().is_some());
    crate::runtime::guest_ram_map::reset();
}

/// Draw buffers retain one whole-resource allocation and carry each decoded
/// offset to the engine beside it. Two offsets of one guest buffer must share
/// the allocation instead of creating two exact-window resources.
#[test]
fn draw_buffers_bind_one_packed_resource_at_each_decoded_offset() {
    // The guest-RAM map resolves once per process and every test in this binary
    // shares it, so a fixture that maps its own RAM has to discard whatever an
    // earlier test resolved. Without this the alias rail asks a map built from
    // some other test's host and refuses — which is the right refusal against
    // the wrong host.
    crate::runtime::guest_ram_map::reset();
    use reims_vgpu_core::endian::st32;
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    let page_shift = PAGE_SHIFT_X86;
    let page = 1u64 << page_shift;
    let mut host = FakeHost::new();
    host.stable_map_pages = true;
    let (dir_gpa, root_gpa, data_gpa) = (2 * page, 3 * page, 4 * page);
    host.map_range(dir_gpa, page as usize, 0);
    host.map_range(root_gpa, page as usize, 0);
    host.map_range(data_gpa, (6 * page) as usize, 0);
    let mut directory = [0u8; 8];
    st32(&mut directory[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut directory[DIRECTORY_DEPTH as usize..], 1);
    host.write_gpa(dir_gpa, &directory).unwrap();
    let mut ptes = [0u8; 24];
    for (i, pte) in ptes.chunks_exact_mut(4).enumerate() {
        st32(pte, 4 + i as u32);
    }
    host.write_gpa(root_gpa, &ptes).unwrap();

    crate::runtime::guest_ram::latch_import_limits(page, 1 << 30, 1 << 30);
    let mut state = Device::new(DeviceId(1), page_shift);
    state.define_task(1, page, 2);
    let backing = super::BufferBacking {
        gva: 0x800,
        size: 0x800,
    };
    let first = super::load_buffer_content_resolved(
        &mut state,
        &mut host,
        1,
        7,
        0,
        true,
        Some(0x400),
        &backing,
    )
    .expect("the small declared resource resolves");
    let reims_vgpu_core::BufferContent::GuestRuns(first_source) = first else {
        panic!("a small resource must not be routed through a CPU snapshot")
    };
    assert_eq!(first_source.source_offset, 0);
    assert_eq!(first_source.total_len, 0x400);
    let first_pages = first_source
        .pages
        .clone()
        .expect("the retained allocation has one import view");
    state
        .bound_buffers
        .packed_available(1, 7, backing.gva, backing.size)
        .expect("the first draw bind retained its allocation");

    let content = super::load_buffer_content_resolved(
        &mut state,
        &mut host,
        1,
        7,
        0x400,
        true,
        Some(0x400),
        &backing,
    )
    .expect("the retained resource serves a later offset directly");
    let reims_vgpu_core::BufferContent::GuestRuns(source) = content else {
        panic!("the draw stays on the guest-memory rail")
    };
    crate::runtime::guest_ram::forget_import_limits();

    assert_eq!(source.source_offset, 0x400);
    assert_eq!(source.total_len, 0x400);
    let pages = source.pages.expect("the exact window is GPU-readable");
    assert!(std::sync::Arc::ptr_eq(&first_pages, &pages));
    assert_eq!(
        state.bound_buffers.len(),
        0,
        "packed resource offsets must not populate the exact-window fallback"
    );
}

/// A type-2/3 render target keeps the allocation its descriptor declared all
/// the way to the backend. The target plane is an offset inside that retained
/// allocation; neither its GVA nor its dimensions are used to manufacture a
/// narrower parent import.
#[test]
fn gva_target_backing_retains_the_declared_parent_allocation() {
    crate::runtime::guest_ram_map::reset();
    use reims_vgpu_core::endian::st32;
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    let page = 1u64 << PAGE_SHIFT_X86;
    let mut host = FakeHost::new();
    host.stable_map_pages = true;
    host.map_range(2 * page, page as usize, 0);
    host.map_range(3 * page, page as usize, 0);
    host.map_range(4 * page, (6 * page) as usize, 0);
    let mut directory = [0u8; 8];
    st32(&mut directory[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut directory[DIRECTORY_DEPTH as usize..], 1);
    host.write_gpa(2 * page, &directory).unwrap();
    let mut ptes = [0u8; 24];
    for (i, pte) in ptes.chunks_exact_mut(4).enumerate() {
        st32(pte, 4 + i as u32);
    }
    host.write_gpa(3 * page, &ptes).unwrap();

    crate::runtime::guest_ram::latch_import_limits(page, 1 << 30, 1 << 30);
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    state.define_task(1, page, 2);
    let linear = LinearColorTarget {
        allocation_gva: page,
        allocation_size: 2 * page,
        plane_offset: 0x400,
        row_stride: 256,
    };
    let req = DrawEncodeRequest {
        task_id: 1,
        colors: vec![ColorRtRequest {
            texture_ref: 7,
            storage: ColorTargetStorage::Linear(linear),
            width: 64,
            height: 16,
            format: reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            store_action: reims_vgpu_protocol::pass_action::StoreAction::Store,
            ..Default::default()
        }],
        ..Default::default()
    };

    let memory = super::gva_guest_target_backing(&mut state, &mut host, &req)
        .expect("the complete declared allocation has a stable packed alias");
    let packed = state
        .bound_buffers
        .packed_available(1, 7, linear.allocation_gva, linear.allocation_size)
        .expect("the target is owned by the resource's retained allocation");
    assert!(std::sync::Arc::ptr_eq(&memory.import, &packed.import));
    assert_eq!(memory.footprint, packed.footprint);
    assert_eq!(
        memory.backing.plane_offset,
        packed.head + linear.plane_offset,
        "the plane offset stays relative to the actual imported parent"
    );
    assert_eq!(memory.backing.row_pitch, u64::from(linear.row_stride));
    assert_eq!(memory.backing.allocation_len, packed.import.len());
    assert_eq!(
        memory.backing.resource_offset, packed.head,
        "the guest allocation remains a distinct window within a broader import"
    );
    assert_eq!(memory.backing.resource_len, linear.allocation_size);

    let mut secondary = req.colors[0].clone();
    secondary.slot = 1;
    let colors = [ColorRtRequest::default(), secondary];
    let primary = crate::model::TargetIdentity::Gva {
        gva: 0xdead_0000,
        width: 64,
        height: 16,
        generation: 0,
        format: reims_vgpu_core::pixel_format::TexelLayout::Bgra8,
    };
    let secondaries = super::build_secondary_targets(
        &mut state,
        &mut host,
        1,
        &colors,
        &crate::runtime::decode::resource::RenderPipelineDescriptor::default(),
        &primary,
        &[],
    )
    .expect("the same allocation contract applies to MRT slot 1");
    let secondary_memory = secondaries[0]
        .target_guest
        .as_ref()
        .expect("a secondary carries its canonical guest allocation to the backend");
    assert!(std::sync::Arc::ptr_eq(
        &secondary_memory.import,
        &memory.import
    ));
    assert_eq!(secondary_memory.backing, memory.backing);

    let incomplete = DrawEncodeRequest {
        colors: vec![ColorRtRequest {
            texture_ref: 8,
            storage: ColorTargetStorage::Linear(LinearColorTarget {
                allocation_gva: 5 * page,
                allocation_size: 2 * page,
                plane_offset: 0,
                row_stride: 256,
            }),
            width: 64,
            height: 16,
            format: reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            store_action: reims_vgpu_protocol::pass_action::StoreAction::Store,
            ..Default::default()
        }],
        task_id: 1,
        ..Default::default()
    };
    assert!(
        super::gva_guest_target_backing(&mut state, &mut host, &incomplete).is_none(),
        "an incomplete parent allocation must stay on the copying fallback"
    );
    assert!(matches!(
        state.bound_buffers.packed(1, 8),
        Some(crate::runtime::bound_buffers::PackedBufferResolution::Unavailable { .. })
    ));
    crate::runtime::guest_ram::forget_import_limits();
    crate::runtime::guest_ram_map::reset();
}

#[test]
fn sampled_plane_keeps_its_copy_source_and_checks_the_packed_extent() {
    use crate::runtime::bound_buffers::PackedBuffer;
    use crate::runtime::guest_ram::GuestRamImport;

    let import = std::sync::Arc::new(
        GuestRamImport::new_host_allocation(0x7f00_0000_0000, 0x8000, 0x1000)
            .expect("aligned packed allocation"),
    );
    let packed = PackedBuffer {
        gva: 0x1800,
        size: 0x6000,
        head: 0x800,
        import: std::sync::Arc::clone(&import),
        gpas: std::sync::Arc::new(vec![0; 8]),
        footprint: crate::runtime::guest_ram::GuestPageFootprint::new(
            std::sync::Arc::from([0x1000]),
            0x1000,
        )
        .unwrap(),
        runs: std::sync::Arc::new(Vec::new()),
        pages: std::sync::Arc::new(Vec::new()),
        sampled_image_requirements: std::collections::HashMap::new(),
        buffer_window_page_sets: std::collections::HashMap::new(),
        owned_alias: None,
    };
    assert!(
        super::linear_sample_from_packed(
            &packed,
            0x7000,
            0x1001,
            128,
            reims_vgpu_memory::GuestImageLayout::D2 {
                width: 128,
                height: 16,
            },
            reims_vgpu_core::pixel_format::TexelLayout::Rgba8,
            reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Rgba8,),
            super::LinearSampleIdentity {
                key: 1,
                generation: 1,
            },
            crate::runtime::gather_witness::GatherVouch::Fresh,
            reims_vgpu_core::pixel_format::SwizzlePlan::default(),
        )
        .is_none(),
        "a sampled copy may not extend the retained allocation to fit"
    );

    let import_owners = std::sync::Arc::strong_count(&packed.import);
    let page_list_owners = std::sync::Arc::strong_count(&packed.gpas);
    let run_owners = std::sync::Arc::strong_count(&packed.runs);
    let guest_ref_owners = std::sync::Arc::strong_count(&packed.pages);
    let request = super::linear_sample_from_packed(
        &packed,
        0x1000,
        0x2000,
        128,
        reims_vgpu_memory::GuestImageLayout::D2 {
            width: 128,
            height: 16,
        },
        reims_vgpu_core::pixel_format::TexelLayout::Rgba8,
        reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Rgba8),
        super::LinearSampleIdentity {
            key: 1,
            generation: 1,
        },
        crate::runtime::gather_witness::GatherVouch::Fresh,
        reims_vgpu_core::pixel_format::SwizzlePlan::default(),
    )
    .expect("the retained allocation supplies this plane's copy source");
    let super::SampledSourceRequest::GuestImage(
        source,
        _,
        Some(identity),
        crate::runtime::gather_witness::GatherVouch::Fresh,
        _,
    ) = request
    else {
        panic!("a packed source must retain its copied-content identity")
    };
    assert_eq!((identity.key, identity.generation), (1, 1));
    assert_eq!(
        std::sync::Arc::strong_count(&packed.import),
        import_owners + 1,
        "the image materialization retains its backing allocation"
    );
    assert_eq!(
        std::sync::Arc::strong_count(&packed.gpas),
        page_list_owners,
        "the physical construction list does not travel to execution"
    );
    assert_eq!(
        std::sync::Arc::strong_count(&packed.runs),
        run_owners + 1,
        "execution retains the run source it consumes"
    );
    assert_eq!(
        std::sync::Arc::strong_count(&packed.pages),
        guest_ref_owners + 1,
        "execution retains the bounded guest reference it consumes"
    );
    assert_eq!(source.transfer.source_offset, 0x1000);
    assert_eq!(source.transfer.total_len, 0x2000);
    assert_eq!(
        source
            .direct
            .as_ref()
            .expect("direct source")
            .backing
            .plane_offset,
        packed.head + 0x1000
    );
}

/// The window refusals name **which** check refused, because the census that
/// ranks them turns on telling two of them apart.
///
/// `span_unmapped` is a page the guest never mapped, which a mapped-range
/// record could answer without walking at all. `untileable` is a walk that
/// finished and still could not bind, which no record upstream of the walk
/// would have saved. Both used to be one silent `None`, and a reading that
/// merged them would argue for machinery against a population that is not
/// there — so the discrimination is the measurement, not a tidiness.
#[test]
fn a_window_refusal_names_which_check_refused() {
    use super::WindowRefusal;
    use reims_vgpu_core::endian::st32;
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    let page_shift = crate::model::PAGE_SHIFT_X86;
    let page = 1u64 << page_shift;

    // GVA page 0 maps to `leaf_pfn`; GVA page 1 is left unmapped. `back_leaf`
    // decides whether the host has RAM behind the frame the PTE names, which is
    // what separates a walk that fails from an import that does.
    let build = |leaf_pfn: u32, back_leaf: bool| {
        let mut host = FakeHost::new();
        host.strict_linux_map = true;
        host.stable_map_pages = true;
        let (dir_gpa, root_gpa) = (2u64 << page_shift, 3u64 << page_shift);
        host.map_range(dir_gpa, page as usize, 0);
        host.map_range(root_gpa, page as usize, 0);
        if back_leaf {
            host.map_range((leaf_pfn as u64) << page_shift, page as usize, 0);
        }
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        let mut pte = [0u8; 4];
        st32(&mut pte, leaf_pfn);
        host.write_gpa(root_gpa, &pte).unwrap();
        let mut state = Device::new(DeviceId(1), page_shift);
        state.define_task(1, page, 2);
        (host, state)
    };

    // Control: wholly inside the mapped page, with RAM behind it.
    let (mut host, state) = build(4, true);
    assert!(
        task_gva_guest_run_window(&state, &mut host, 1, 8, 16).is_ok(),
        "control: this span resolves and binds"
    );

    // The same start, but the span reaches into the page the guest never mapped.
    let (mut host, state) = build(4, true);
    assert_eq!(
        task_gva_guest_run_window(&state, &mut host, 1, 8, page).err(),
        Some(WindowRefusal::SpanUnmapped),
        "a page absent from the task's own table"
    );

    // Every page of the span resolves — the PTE is there — but no host range
    // backs the frame, so the walk finishes and the import is what refuses.
    let (mut host, state) = build(9, false);
    assert_eq!(
        task_gva_guest_run_window(&state, &mut host, 1, 8, 16).err(),
        Some(WindowRefusal::Untileable),
        "a walk that finished and still could not bind"
    );
}

/// The synchronous GVA Store's write must be bounded to the pages the command
/// named, and that set has to be taken before the GPU round trip.
///
/// `encode_draw_chain` encodes, submits, waits and reads back before the Store
/// resolves `target_gva`. The guest runs on its own vCPUs across that gap and
/// can hand the range to something else, so a write authorised by a walk taken
/// after the readback is authorised by the wrong page table -- the same shape
/// that was scattering deferred render targets into other owners' memory.
///
/// The two permissive arms are locked deliberately, because a bound that
/// refuses when it should not drops live Stores: a record that does not own
/// guest writeback and a target with no GVA both stay unbounded.
/// Guest page frame the [`StoreRig`] page table hands out for entry `i`, and the
/// spare frames a test re-points an entry at.
const STORE_RIG_PT_BASE: u32 = 4;

/// A task whose page table maps the first eight virtual pages one-to-one onto
/// guest frames `STORE_RIG_PT_BASE + i`, plus the frames themselves.
///
/// Every bound test needs the same thing: a walkable task, a target GVA, and a
/// way to re-point one of its PTEs at a frame the command never named. The rig
/// exists so the interesting half of each test is the write, not the walk.
struct StoreRig {
    host: FakeHost,
    state: Device,
    root_gpa: u64,
}

impl StoreRig {
    /// Task 1, `entries` mapped virtual pages, `page_shift` geometry.
    fn new(entries: u32) -> Self {
        use reims_vgpu_core::endian::st32;
        use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        let mut host = FakeHost::new();
        let state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let (dir_pfn, root_pfn) = (2u32, 3u32);
        let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_ARM64E;
        let root_gpa = (root_pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x4000, 0);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], root_pfn);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        let mut rig = Self {
            host,
            state,
            root_gpa,
        };
        for i in 0..entries {
            rig.host
                .map_range(Self::frame_gpa(STORE_RIG_PT_BASE + i), 0x4000, 0);
            rig.point(i, STORE_RIG_PT_BASE + i);
        }
        rig.state.define_task(1, 0x1000, dir_pfn);
        rig
    }

    /// Guest physical base of frame `pfn`.
    fn frame_gpa(pfn: u32) -> u64 {
        (pfn as u64) << PAGE_SHIFT_ARM64E
    }

    /// Point virtual page `entry` of the task at guest frame `pfn`.
    fn point(&mut self, entry: u32, pfn: u32) {
        use reims_vgpu_core::endian::st32;
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        let _ = self
            .host
            .write_gpa(self.root_gpa + (entry as u64) * 4, &pte);
    }

    /// Virtual address of the task's page `entry`.
    fn gva(entry: u32) -> u64 {
        (entry as u64) << PAGE_SHIFT_ARM64E
    }
}

/// A page the pre-submit walk could not resolve takes the ordered page list
/// away entirely rather than shortening it.
///
/// The two forms of [`StoreTargetPages`] fail differently on a short walk and
/// only one of them can fail closed. The membership form is a subset, so a
/// missing page can only refuse a row that wanted it. The ordered form is read
/// *positionally* — `references_for_runs` takes index `i` as page `i` of the
/// window — so a list with a hole shifts every page after it, and the GPU copy
/// built from it lands the frame at guest addresses the command never named,
/// converting nothing and checking nothing on the way.
#[test]
fn a_short_pre_submit_walk_takes_the_ordered_page_list_away() {
    const PAGE: u64 = 1 << PAGE_SHIFT_ARM64E;
    let gva = PAGE;
    let span = PAGE * 3;
    let complete = vec![0x10 * PAGE, 0x11 * PAGE, 0x12 * PAGE];

    let whole = StoreTargetPages {
        set: complete.iter().copied().collect(),
        ordered: complete.clone(),
        span,
    };
    assert_eq!(
        whole.ordered_complete(gva, PAGE),
        Some(&complete[..]),
        "a walk that resolved every page of the span is positionally readable"
    );

    // The middle page dropped, which is what `task_gva_page_gpas` leaves behind
    // for a span with an unmapped page in it.
    let short = StoreTargetPages {
        set: [complete[0], complete[2]].into_iter().collect(),
        ordered: vec![complete[0], complete[2]],
        span,
    };
    assert!(
        short.ordered_complete(gva, PAGE).is_none(),
        "a hole must take the whole list, not hand back a shifted one"
    );
    assert_eq!(
        short.membership().len(),
        2,
        "the membership form stays usable: a subset can only refuse"
    );
}

#[test]
fn a_synchronous_gva_store_is_bounded_to_the_pages_the_command_named() {
    let rig = StoreRig::new(8);
    let (mut host, mut state, root_gpa) = (rig.host, rig.state, rig.root_gpa);
    let pt_base = STORE_RIG_PT_BASE;

    let page = 1u64 << PAGE_SHIFT_ARM64E;
    // 64x64 BGRA8 with a tight stride is exactly one 16 KiB page.
    let c0 = ColorRtRequest {
        slot: 0,
        texture_ref: 7,
        resource: None,
        storage: linear_target_storage(page, 64 * 4, 64),
        width: 64,
        height: 64,
        format: reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
        sample_count: 1,
        load_action: reims_vgpu_protocol::pass_action::LoadAction::Load,
        store_action: reims_vgpu_protocol::pass_action::StoreAction::Store,
        clear_color: [0.0; 4],
        target_seed_rgba: None,
        multisample_source_ref: 0,
    };

    let armed = sync_store_allowed_pages(&state, &host, 1, Some(&c0), true)
        .expect("a resolvable GVA target must be bounded");
    let mut resolve = c0.clone();
    resolve.store_action = reims_vgpu_protocol::pass_action::StoreAction::MultisampleResolve;
    resolve.sample_count = 4;
    resolve.multisample_source_ref = 8;
    assert!(
        sync_store_allowed_pages(&state, &host, 1, Some(&resolve), true).is_some(),
        "a resolve publishes into the same guest destination and needs the same bound"
    );
    assert_eq!(
        armed.membership().len(),
        1,
        "64x64 BGRA8 tight covers one 16 KiB page"
    );
    assert!(armed
        .membership()
        .contains(&((pt_base as u64 + 1) << PAGE_SHIFT_ARM64E)));

    // The guest hands that virtual page to a different allocation while the GPU
    // is working. The write that follows must not reach the new owner.
    let mut pte = [0u8; 4];
    reims_vgpu_core::endian::st32(&mut pte, pt_base + 6);
    let _ = host.write_gpa(root_gpa + 4, &pte);
    let rgba = vec![0xffu8; 64 * 64 * 4];
    let err = write_gva_rgba8_within(
        &mut state,
        &mut host,
        1,
        page,
        64,
        64,
        64 * 4,
        c0.format,
        &rgba,
        Some(armed.membership()),
    )
    .expect_err("a page the command never named must be refused");
    assert!(
        matches!(err, crate::runtime::host::MemError::WriteOutsideWindow),
        "expected WriteOutsideWindow, got {err:?}"
    );
    // Unbounded, it lands in whatever owns that page now -- this is the write
    // the crash reports are of, and 0xff is the white pixel run the guest
    // kernel reported over its freed heap element.
    assert!(
        write_gva_rgba8_within(
            &mut state,
            &mut host,
            1,
            page,
            64,
            64,
            64 * 4,
            c0.format,
            &rgba,
            None,
        )
        .is_ok(),
        "without the bound the same write succeeds into the new owner's page"
    );

    // Permissive arms stay permissive.
    assert!(
        sync_store_allowed_pages(&state, &host, 1, Some(&c0), false).is_none(),
        "a record that does not own guest writeback is not this rail"
    );
    let no_gva = ColorRtRequest {
        storage: ColorTargetStorage::None,
        ..c0
    };
    assert!(
        sync_store_allowed_pages(&state, &host, 1, Some(&no_gva), true).is_none(),
        "no GVA target, nothing to bound"
    );
}

/// A draw the engine never attempted is counted, with the vertices it cost.
///
/// `encode_draw_chain`'s skipped-draw tail spends one `linux_clear_store
/// draws_skipped` line per `(pipeline, slug)`, so a pipeline refused every frame
/// reports one line for however many draws it lost. The two census counters do
/// not dedupe, and they are therefore the only readings that can be summed into
/// "what did this refusal cost the guest". A bare zero from them has to mean no
/// draw was skipped, never that nobody was counting.
///
/// Both are asserted because they fail differently. Dropping the draw count
/// loses the rate; wiring the vertex count to a neighbouring field of
/// `DrawEncodeRequest` — `instance_count` and `first_vertex` sit beside
/// `vertex_count` and every one of them is a `u32` — still moves a counter, and
/// only a magnitude check separates a skipped six-vertex quad from a skipped
/// fifty-four-vertex pass. The delta form is deliberate: the census map is
/// process-global and the rest of the suite shares it.
#[test]
fn a_draw_skipped_after_an_engine_refusal_is_counted_with_the_vertices_it_cost() {
    use crate::runtime::drain::store_route_count;

    const DRAWS: &str = "draws_skipped_after_engine_refusal";
    const VERTICES: &str = "draws_skipped_after_engine_refusal_vertices";
    /// Not 1, 3 or 6: a count that could be confused with an instance count, a
    /// triangle or a full-screen quad cannot show the vertex counter reading
    /// the wrong field.
    const VERTEX_COUNT: u32 = 54;

    let rig = StoreRig::new(8);
    let (mut host, mut state) = (rig.host, rig.state);

    let req = DrawEncodeRequest {
        task_id: 1,
        // No pipeline, so the engine draw is never attempted at all — the
        // cheapest way to the tail, and the arm whose refusal the emitter has
        // to name itself because there is no engine slug to borrow.
        pipeline_ref: 0,
        vertex_count: VERTEX_COUNT,
        instance_count: 1,
        primitive_topology: reims_vgpu_protocol::PrimitiveTopology::Triangle,
        first_vertex: 0,
        colors: vec![ColorRtRequest {
            slot: 0,
            texture_ref: 7,
            // 64x64 BGRA8 at a tight stride is exactly one 16 KiB page of the
            // rig's walkable task, so the CLEAR seed Store lands and the tail's
            // `any_store` precondition is met.
            storage: linear_target_storage(StoreRig::gva(1), 64 * 4, 64),
            width: 64,
            height: 64,
            format: reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            load_action: reims_vgpu_protocol::pass_action::LoadAction::Clear,
            store_action: reims_vgpu_protocol::pass_action::StoreAction::Store,
            ..Default::default()
        }],
        ..Default::default()
    };

    let draws_before = store_route_count(DRAWS);
    let vertices_before = store_route_count(VERTICES);

    let cap = crate::observe::FailCapture::start();
    let st = encode_draw_chain(&mut state, &mut host, &req, true, false).status;
    let lines = cap.lines();
    drop(cap);

    assert!(
        matches!(st, EncodeStatus::Ok),
        "the CLEAR seed Store landed, so the record stored: {st:?}"
    );
    assert!(
        lines.iter().any(|l| {
            l.contains("reason=draws_skipped_after_engine_refusal")
                && l.contains("refused_by=engine_draw_not_attempted")
        }),
        "the counted skip is the one the tail names: {lines:?}"
    );
    assert_eq!(
        store_route_count(DRAWS),
        draws_before + 1,
        "one skipped draw is one count, whatever the line dedup did with it"
    );
    assert_eq!(
        store_route_count(VERTICES),
        vertices_before + u64::from(VERTEX_COUNT),
        "the vertices banded beside the draw are the draw's own vertex count"
    );
}

/// An IOSurface texture sample must resolve the mapping *before* it reads its geometry.
///
/// A mapped surface with a live `MappingInternal` can have no latched W×H yet:
/// that geometry lives in the guest device-surface descriptor, and
/// `mapper::resolve_mapping_backing` (reached through
/// `ensure_resolved_for_scanout`) is the only thing that decodes it and calls
/// `set_mapping_geom`. Reading `has_geom` first therefore returns `None` on a
/// perfectly serviceable surface and the bind silently loses the texture.
///
/// The fixture carries geometry ONLY in the descriptor — nothing calls
/// `set_mapping_geom` — so this passes with the resolve first and fails with it
/// after the geometry read.
#[test]
fn iosurface_texture_sample_resolves_geometry_before_reading_it() {
    use crate::runtime::host::HostMemory;
    use reims_vgpu_core::endian::{st32, st64};
    use reims_vgpu_paging::geometry::{
        MAPPER_PAGE_ENTRY_PFN_SHIFT as PAGE_ENTRY_PFN_SHIFT,
        MAPPER_PAGE_ENTRY_VALID as PAGE_ENTRY_VALID,
    };
    use reims_vgpu_paging::mapper::{
        MAPPING_INTERNAL_BACKPTR, MAPPING_INTERNAL_DESC_PTR, MAPPING_INTERNAL_EXPECTED_SIZE,
        MAPPING_INTERNAL_ID, MAPPING_INTERNAL_PAGE_COUNT, MAPPING_INTERNAL_PAGE_FIELD_48,
        MAPPING_INTERNAL_PAGE_FIELD_50, MAPPING_INTERNAL_SIZE, MAPPING_PAGE_TABLE_FROM_F48,
    };
    use reims_vgpu_protocol::{
        DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_BPR, DEVICE_DESC_DIMS, DEVICE_DESC_LEN,
        DEVICE_DESC_PIXEL_FORMAT, DEVICE_DESC_PLANE_COUNT,
    };

    // Kernel VA base: the mapper walk refuses anything `guest_kernel_va` rejects.
    const KVA: u64 = 0xffff_fe00_1000_0000;
    fn put_u32(h: &mut FakeHost, kva: u64, v: u32) {
        h.map_range(kva, 4, 0);
        h.put_u32(kva, v);
    }
    fn put_u64(h: &mut FakeHost, kva: u64, v: u64) {
        h.map_range(kva, 8, 0);
        let _ = h.write_gpa(kva, &v.to_le_bytes());
    }

    let mid = 3u32;
    // 64×64 BGRA8 at a packed 256-byte row is 16 KiB — exactly one arm64e guest
    // page, so a one-entry page table covers the whole sample window.
    let (w, h) = (64u32, 64u32);
    let bpr = w * RGBA8_BPP;
    let alloc = bpr * h;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let internal = KVA;
    let mapper = KVA + 0x1000;
    let page_obj = KVA + 0x2000;
    let table = KVA + 0x3000;
    let desc_kva = KVA + 0x4000;

    put_u64(&mut host, internal + MAPPING_INTERNAL_BACKPTR, mapper);
    put_u32(&mut host, internal + MAPPING_INTERNAL_ID, mid);
    put_u32(
        &mut host,
        internal + MAPPING_INTERNAL_SIZE,
        MAPPING_INTERNAL_EXPECTED_SIZE,
    );
    put_u64(
        &mut host,
        internal + MAPPING_INTERNAL_PAGE_FIELD_48,
        page_obj,
    );
    put_u64(&mut host, internal + MAPPING_INTERNAL_PAGE_FIELD_50, 0);
    put_u64(&mut host, internal + MAPPING_INTERNAL_PAGE_COUNT, 1);
    put_u64(&mut host, internal + MAPPING_INTERNAL_DESC_PTR, desc_kva);
    put_u64(&mut host, page_obj + MAPPING_PAGE_TABLE_FROM_F48, table);

    let pfn = 0x1e88c_u32;
    let page_gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    put_u32(
        &mut host,
        table,
        (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
    );
    // Guest pages ARE the surface content.
    host.map_range(page_gpa, alloc as usize, 0);
    let bgra = [0x11u8, 0x22, 0x33, 0xff].repeat((w * h) as usize);
    host.write_gpa(page_gpa, &bgra).expect("seed guest pages");

    // The device-surface descriptor is the only carrier of this surface's dims.
    let mut desc = vec![0u8; DEVICE_DESC_LEN];
    st32(
        &mut desc[DEVICE_DESC_PIXEL_FORMAT..],
        u32::from(MTL_FORMAT_BGRA8_UNORM),
    );
    st32(&mut desc[DEVICE_DESC_ALLOC_SIZE..], alloc);
    // width u24 @ bit 8, height u24 @ bit 40.
    st64(
        &mut desc[DEVICE_DESC_DIMS..],
        ((w as u64) << 8) | ((h as u64) << 40),
    );
    st32(&mut desc[DEVICE_DESC_BPR..], bpr);
    desc[DEVICE_DESC_PLANE_COUNT] = 0;
    host.map_range(desc_kva, DEVICE_DESC_LEN, 0);
    host.write_gpa(desc_kva, &desc).expect("seed descriptor");

    state.observe_mapper_device(mapper);
    assert!(state.attach_mapping_internal(mid, internal));
    {
        let m = state.surfaces.mappings.get(&mid).expect("mapping");
        assert!(m.lifecycle.active && m.lifecycle.internal_kva != 0);
        assert!(
            !m.has_geometry(),
            "the fixture's whole point is that no geometry is latched yet"
        );
    }

    let (gw, gh, rgba) = load_iosurface_mapping_rgba(&mut state, &mut host, mid, None).expect(
        "a mapped IOSurface texture surface whose dims are still only in the device descriptor \
         must sample, not drop the bind",
    );
    assert_eq!((gw, gh), (w, h), "geometry must come from the resolve");
    assert_eq!(rgba.len(), (w * h * RGBA8_BPP) as usize);
    assert_eq!(
        &rgba[..4],
        &[0x33, 0x22, 0x11, 0xff],
        "guest BGRA page bytes converted to tight RGBA8"
    );
    assert!(
        state
            .surfaces
            .mappings
            .get(&mid)
            .expect("mapping")
            .has_geometry(),
        "the resolve must have latched the descriptor geometry"
    );
}

/// A pass dropped because one of its colour slots would not resolve is
/// counted, and counted per occurrence.
///
/// This builder used to spend a fail line here carrying a reason re-derived by
/// `sample_miss_detail` — a second walk of the object list, in the sampled
/// vocabulary, guessing at what `lookup_render_target` had just decided. Both
/// callers already report the `None` with a reason, and the resolve now reports
/// the check, so the line went and the counter stayed: it is the only thing
/// that separates "a slot would not resolve" from the other ways this builder
/// returns nothing, and the only one of the three that carries magnitude, since
/// both caller lines are per-occurrence but neither is per-slot.
#[test]
fn a_pass_dropped_for_an_unresolvable_colour_slot_is_counted_every_time() {
    use crate::runtime::drain::store_route_count;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let before = store_route_count("mrt_slot_unresolved");

    // An unbound slot is not this: it is skipped, and a pass of nothing but
    // unbound slots resolves to a request with no colours rather than a loss.
    assert!(single_rt_draw_request(&mut state, &mut host, 7, clear_black_attachment(0)).is_none());
    assert_eq!(
        store_route_count("mrt_slot_unresolved"),
        before,
        "an unbound colour slot is not an unresolvable one"
    );

    // A bound ref with nothing under it drops the whole pass, every time.
    for _ in 0..3 {
        assert!(
            single_rt_draw_request(&mut state, &mut host, 7, clear_black_attachment(0x5d1))
                .is_none()
        );
    }
    assert_eq!(store_route_count("mrt_slot_unresolved"), before + 3);
}

/// The degradation dedupe is keyed by `(pipeline_ref, slug)`, and both encode
/// arms depend on that being true.
///
/// It is what makes a per-draw degradation reportable at all: the depth and
/// stencil LOAD substitutions sit inside the draw path, so without a first-only
/// key has to separate slugs as well as pipelines, or one degradation would
/// hits, since its depth and stencil substitutions can both fire on one pass.
#[test]
fn a_degradation_reports_once_per_pipeline_and_slug() {
    // Pipeline refs local to this test so a sibling cannot consume the first
    // fire out from under it — the dedupe set is process-wide by design.
    let (a, b) = (0x5eed_0001, 0x5eed_0002);
    assert!(degrade_log_first(a, "depth_load_readback_failed"));
    assert!(
        !degrade_log_first(a, "depth_load_readback_failed"),
        "a repeat of the same degradation on the same pipeline must stay quiet"
    );
    assert!(
        degrade_log_first(a, "stencil_load_readback_failed"),
        "a different degradation on the same pipeline is a different report"
    );
    assert!(
        degrade_log_first(b, "depth_load_readback_failed"),
        "the same degradation on a different pipeline is a different report"
    );
}

/// A sampled bind accepts only the texture object kinds declared by the object
/// list, regardless of whether another object's descriptor is long enough to
/// resemble texture geometry.
#[test]
fn a_sampled_ref_naming_another_object_kind_is_refused() {
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE, TEXTURE_DESC_BASE_LEN,
    };
    use crate::runtime::gva_mem::write_task_gva_arm64e;
    use crate::runtime::objects::OBJECT_TYPE_REF_TEXTURE;
    use reims_vgpu_core::endian::{st32, st64};
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    // One task, one object list, and two refs into it — same descriptor bytes,
    // different object type in the entry.
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let (dir_pfn, root_pfn) = (2u32, 3u32);
    let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_ARM64E;
    let root_gpa = (root_pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x4000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], root_pfn);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    assert!(host.write_gpa(dir_gpa, &d).is_ok());
    for i in 0..4u32 {
        let pfn = 4 + i;
        host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        assert!(host.write_gpa(root_gpa + (i as u64) * 4, &pte).is_ok());
    }
    state.define_task(1, 0x1000, dir_pfn);
    assert!(state.set_object_list(1, 0, 32));

    let body = TEXTURE_DESC_BASE_LEN;
    let mut desc = vec![0u8; body];
    st64(&mut desc[0..], 0x1000);
    st32(&mut desc[8..], 1);
    let desc_gva = 0x200u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &desc);
    let put_entry = |host: &mut FakeHost, obj_ref: u32, object_type: u8| {
        let off = list_object_entry_offset(obj_ref, 32).expect("entry in range");
        let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
        st32(&mut le[0..], (object_type as u32) | ((body as u32) << 8));
        le[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        write_task_gva_arm64e(host, &state.tasks[1], off, &le);
    };
    let (texture_ref, view_ref) = (6u32, 7u32);
    put_entry(&mut host, texture_ref, OBJECT_TYPE_TEXTURE);
    put_entry(&mut host, view_ref, OBJECT_TYPE_REF_TEXTURE);

    assert!(
        super::sampled_texture_descriptor(&state, &host, 1, texture_ref).is_some(),
        "a texture ref resolves"
    );
    assert!(
        super::sampled_texture_descriptor(&state, &host, 1, view_ref).is_none(),
        "a non-texture object must not acquire texture meaning from its byte shape"
    );
}

/// Every way a buffer ref fails names itself, distinctly, in the counted
/// vocabulary.
///
/// The five conditions used to be five `observe::fail` lines with **no
/// `reason=` field**, which is the one field the fail log is ranked by — so
/// "how often did a draw fail to resolve a buffer" had no answer, and the first
/// rung was invisible to the grep that finds every other rail's. Distinctness
/// is the other half: two conditions sharing a slug share `fail_once`'s latch
/// wherever one is used with it, and the five are five different findings.
#[test]
fn every_buffer_span_refusal_has_its_own_reason_slug() {
    use crate::runtime::objects::{BufferSpanRefusal, LadderRung};

    let all = [
        BufferSpanRefusal::Rung(LadderRung::NoListEntry),
        BufferSpanRefusal::Rung(LadderRung::WrongType {
            got: ObjectKind::Texture,
        }),
        BufferSpanRefusal::Rung(LadderRung::DescRead { declared_len: 96 }),
        BufferSpanRefusal::Decode,
        BufferSpanRefusal::NoBacking,
    ];
    let slugs: Vec<&str> = all.iter().map(|r| super::buffer_refusal_slug(*r)).collect();
    let mut sorted = slugs.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        all.len(),
        "five conditions, five slugs: {slugs:?}"
    );
    for slug in &slugs {
        assert!(
            slug.starts_with("draw_buffer"),
            "the role says which rail failed: {slug}"
        );
    }
    // The three ladder rungs must be spelled by the macro, not by hand — this
    // is what keeps a sixth spelling of "the guest named nothing" from existing.
    assert_eq!(
        slugs[0],
        crate::observe::ladder_slug!("draw_buffer", no_list_entry)
    );
    assert_eq!(
        slugs[1],
        crate::observe::ladder_slug!("draw_buffer", wrong_type)
    );
    assert_eq!(
        slugs[2],
        crate::observe::ladder_slug!("draw_buffer", desc_read)
    );

    // The detail field carries what each refusal knows and nothing it does not.
    assert_eq!(super::buffer_refusal_detail(all[1], 12), "ty=texture");
    assert_eq!(super::buffer_refusal_detail(all[2], 12), "desc_len=96");
    assert_eq!(super::buffer_refusal_detail(all[4], 12), "shift=12");
    assert!(super::buffer_refusal_detail(all[0], 12).is_empty());
    assert!(super::buffer_refusal_detail(all[3], 12).is_empty());
}

/// The bind's stride wins over the pipeline's, and only where it exists.
///
/// Four cases, and three of them are the ones a shorter test would miss. The
/// interesting field is `Option<u64>` against a `u32` consumer, so:
///
/// * `Some(0)` must not read as "no stride". A zero stride is a legal Metal
///   request — every vertex fetched from one address — and collapsing it onto
///   the absent case would silently restore the pipeline's stride instead.
/// * a stride wider than `u32` must fall back rather than truncate. Both
///   backends carry a 32-bit stride, and `s as u32` on a guest `u64` fetches at
///   an unrelated number rather than at the one asked for.
/// * a bind at a *different* index must not answer for this one.
#[test]
fn a_bind_stride_overrides_the_pipeline_stride_only_where_it_exists() {
    let bind = |index: u32, attribute_stride: Option<u64>| BufferBind {
        index,
        buffer_ref: 1,
        offset: 0,
        attribute_stride,
        ..Default::default()
    };

    // No bind at this index: the pipeline's stride stands.
    assert_eq!(bind_attribute_stride(&[bind(3, Some(64))], 0, 12), 12);
    // A bind carrying none: likewise.
    assert_eq!(bind_attribute_stride(&[bind(0, None)], 0, 12), 12);
    // A bind carrying one: it wins.
    assert_eq!(bind_attribute_stride(&[bind(0, Some(64))], 0, 12), 64);
    // Zero is a stride, not an absence.
    assert_eq!(bind_attribute_stride(&[bind(0, Some(0))], 0, 12), 0);
    // Past `u32`: the pipeline's stands rather than a truncation of the guest's.
    assert_eq!(
        bind_attribute_stride(&[bind(0, Some(u64::from(u32::MAX) + 1))], 0, 12),
        12,
        "a stride the backend cannot carry must not be truncated into one it can"
    );
}

/// The GVA resident sample rung must fail closed, and this pins the direction
/// it fails in.
///
/// `try_gva_resident_sample` declines to read the guest's pages at all and
/// hands the draw an engine image instead, on the strength of one claim: a
/// render Store published this exact page set and nothing has written it since.
/// Where no Store ever stamped the span, there is no evidence for that claim,
/// and serving a resident anyway binds whatever image is parked under a
/// colliding identity while the guest's own pixels go unread. The symptom is a
/// frame that stops updating and stays wrong — the worst class this crate has,
/// because every layer above the rung sees a successful bind.
///
/// The refusal is therefore asserted twice, and the second half is the half
/// that catches the regression. A rung that lost its `GvaWriteReach::Quiet`
/// test would still answer `None` here, since no engine exists in a unit test
/// to hold a resident under any identity, so the return value alone cannot tell
/// a working guard from a deleted one. The census route can: `gvaw_no_entry`
/// means the witness was asked and had nothing, while `gvarung_resident_absent`
/// in its place means the rung walked past the witness and went straight to the
/// registry.
///
/// The second leg is the orphan. A witness entry at the same address and extent
/// under a stale page-set generation must not answer for the allocation that
/// replaced it: the generation is recomputed from the pages as they stand now
/// rather than read back out of whatever entry the address happens to find, so
/// a moved page list misses instead of being wrong.
#[test]
fn a_gva_span_no_store_has_stamped_refuses_the_resident_sample_rung() {
    use crate::runtime::decode::resource::{TextureDescriptor, TextureLevelLayout};
    use crate::runtime::drain::store_route_count;
    use crate::runtime::gva_store_witness::{note_store, GvaTargetKey};
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    // GVA pages 0..7 → data PFNs 4..11: the one-level table every task-GVA walk
    // in this file uses.
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);

    // A 32x32 BGRA8 linear texture whose tight level-0 span sits wholly inside
    // mapped GVA page 2. The span has to walk, or the rung declines on the short
    // walk before it ever reaches the witness and the test proves nothing.
    let page = 1u64 << PAGE_SHIFT_ARM64E;
    let (width, height, row_stride) = (32u32, 32u32, 128u64);
    let span = row_stride * u64::from(height);
    let tex = TextureDescriptor {
        allocation_size: page,
        handle: 2,
        mipmap_level_count: 1,
        base_offset: 0,
        bytes_per_slice: span,
        slice_count: 1,
        cube_faces: false,
        compressed_layout: false,
        bytes_per_element: 4,
        used_size: width * height * 4,
        row_stride: row_stride as u32,
        width,
        height,
        depth: 1,
        declaration: Some(crate::runtime::heap_query::TextureDescriptor {
            texture_type: 2,
            framebuffer_only: false,
            is_drawable: false,
            write_swizzle_enabled: None,
            allow_gpu_optimized_contents: false,
            usage: 0,
            pixel_format: MTL_FORMAT_BGRA8_UNORM,
            width,
            height,
            depth: 1,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 1,
            resource_options: 0,
            protection_options: 0,
            swizzle: None,
        }),
        levels: vec![TextureLevelLayout {
            offset: 0,
            size: span,
            row_stride,
            width,
            height,
            depth: 1,
        }],
    };
    let gva = tex
        .level_gva(0, state.page_shift)
        .expect("the fixture descriptor must name a level-0 window")
        .0;
    assert_eq!(gva, 2 * page, "handle 2 at the arm64e shift is GVA page 2");
    let gpas: Vec<u64> =
        gva_mem::task_gva_page_gpa_set(&host, &state.tasks, 1, gva, span, state.page_shift)
            .into_iter()
            .collect();
    assert_eq!(
        gpas.len(),
        1,
        "fixture: the level-0 span must resolve to its one mapped page, or the \
         rung declines on the walk instead of on the witness"
    );

    let no_entry = store_route_count("gvaw_no_entry");
    let absent = store_route_count("gvarung_resident_absent");
    let served = store_route_count("gvarung_resident");
    let resource = std::sync::Arc::new(crate::model::TaskResource::new(
        reims_vgpu_protocol::ObjectListEntry::new(ObjectKind::Buffer, 0, 0),
        std::sync::Arc::from([]),
    ));
    let resource = state.task_objects.resources.register(1, 7, resource);
    let sampled_only = store_route_count("gvarung_sampled_only");
    assert!(
        super::try_gva_resident_sample(&mut state, &mut host, 1, 7, &resource, &tex).is_none(),
        "a texture never used as an attachment cannot own a render-target resident"
    );
    assert_eq!(store_route_count("gvarung_sampled_only"), sampled_only + 1);
    assert_eq!(
        store_route_count("gvaw_no_entry"),
        no_entry,
        "sampled-only construction state answers before the mutable Store witness"
    );
    resource.note_render_target_use();

    assert!(
        super::try_gva_resident_sample(&mut state, &mut host, 1, 7, &resource, &tex).is_none(),
        "no Store has stamped this span, so nothing licenses serving a resident for it"
    );
    assert_eq!(
        store_route_count("gvaw_no_entry"),
        no_entry + 1,
        "the witness has to be the thing that refused, and has to name itself"
    );
    assert_eq!(
        store_route_count("gvarung_resident_absent"),
        absent,
        "the registry must not be consulted for a span the witness knows nothing about"
    );
    assert_eq!(store_route_count("gvarung_resident"), served);

    // Same address, same extent, a page-set generation that is not this span's:
    // an entry the guest's recycling left behind. Recomputing the generation
    // makes it unreachable rather than merely out-voted.
    let orphan = GvaTargetKey {
        task_id: 1,
        resource: resource.semantic_id().unwrap(),
        gva,
        generation: 0xdead_beef,
        width,
        height,
        // What the descriptor declares, so the orphan differs from the live key
        // in the generation alone.
        bgra: true,
    };
    let orphan_write = state
        .resource_write_stamp_for(orphan.resource)
        .expect("the orphan names a live resource");
    note_store(&mut state, orphan, &gpas, orphan_write);
    assert!(
        super::try_gva_resident_sample(&mut state, &mut host, 1, 7, &resource, &tex).is_none(),
        "a stale page set stamped at this address must not answer for the one that replaced it"
    );
    assert_eq!(
        store_route_count("gvaw_no_entry"),
        no_entry + 2,
        "the orphan is a miss, not a hit under a different generation"
    );
    assert_eq!(store_route_count("gvarung_resident_absent"), absent);
    assert_eq!(store_route_count("gvarung_resident"), served);
}

/// The depth attachment's identity is the guest texture the pass bound, so two
/// draws into one depth texture name one resident and the second allocates
/// nothing.
///
/// This is the whole of what makes the depth rail cost amortise. The engine's
/// reuse test is `ResidentTargetSlot::reusable_for` and it is keyed on this
/// value, so an identity that varied per draw — or that collided between two
/// guest textures — would put the rail straight back to one image per draw, or
/// fuse two guests' depth buffers into one. Neither is visible in a frame until
/// a workload is large enough to notice, which is why it is pinned here rather
/// than left to a boot.
///
/// Not a GPU test: nothing in this crate can drive `registry_ensure_depth`
/// without a device, so the gate sits at the key that rail is keyed on.
#[test]
fn depth_and_stencil_attachments_are_keyed_on_the_guest_texture_the_pass_bound() {
    use crate::model::TargetIdentity;
    use crate::runtime::draw::execution::depth_stencil_chain_identity;
    use reims_vgpu_protocol::{ResourceId, ResourceObject};

    let id = |r: &DrawEncodeRequest, st: bool, resource: ResourceId<ResourceObject>| {
        let attachment_ref = r
            .depth_attach
            .as_ref()
            .map(|depth| depth.texture_ref)
            .or_else(|| r.stencil_attach.as_ref().map(|stencil| stencil.texture_ref))
            .unwrap_or(0);
        depth_stencil_chain_identity(r, attachment_ref, st, resource)
    };
    let req = |depth_ref: u32, w: u32, h: u32| DrawEncodeRequest {
        colors: vec![ColorRtRequest {
            width: w,
            height: h,
            ..Default::default()
        }],
        depth_attach: Some(DepthAttachmentState {
            texture_ref: depth_ref,
            ..Default::default()
        }),
        ..DrawEncodeRequest::default()
    };

    let first_resource = ResourceId::new(17, 3);
    let first =
        id(&req(42, 1024, 768), false, first_resource).expect("a bound depth texture has one");
    assert_eq!(
        first,
        TargetIdentity::Texture {
            ref_: 17,
            width: 1024,
            height: 768,
            generation: 3,
            stencil: false,
        },
        "the canonical resource lifetime, not its task-local ref, is the key"
    );
    assert_eq!(
        id(&req(42, 1024, 768), false, first_resource).as_ref(),
        Some(&first),
        "a second draw into the same depth texture resolves the same resident"
    );
    assert_ne!(
        id(&req(42, 1024, 768), false, ResourceId::new(18, 1),),
        Some(first.clone()),
        "the same task-local ref under another canonical resource must not fuse"
    );
    assert_ne!(
        id(&req(42, 1024, 768), false, ResourceId::new(17, 4),),
        Some(first.clone()),
        "reusing one resource slot at a new generation must not inherit depth"
    );
    assert_ne!(
        id(&req(42, 800, 600), false, first_resource),
        Some(first.clone()),
        "geometry is part of the key, so a resized attachment recreates"
    );

    // The stencil aspect selects the image format, so it partitions the key. Two
    // residents, each stable, rather than one retired and recreated on every
    // alternation between a stencil draw and a depth-only one.
    assert_ne!(
        id(&req(42, 1024, 768), true, first_resource),
        Some(first.clone()),
        "a stencil-carrying depth attachment is its own resident"
    );
    let stencil_only = DrawEncodeRequest {
        colors: vec![ColorRtRequest {
            width: 1024,
            height: 768,
            ..Default::default()
        }],
        stencil_attach: Some(StencilAttachmentState {
            texture_ref: 42,
            ..Default::default()
        }),
        ..DrawEncodeRequest::default()
    };
    assert_eq!(
        id(&stencil_only, true, first_resource),
        Some(TargetIdentity::Texture {
            ref_: 17,
            width: 1024,
            height: 768,
            generation: 3,
            stencil: true,
        }),
        "a stencil-only attachment uses its own generational texture identity"
    );

    // No depth texture in the pass descriptor names no resident. A backend may
    // still need a draw-owned native image to model Metal's implicit depth value
    // for a bound test state, but that image is not a guest pass attachment.
    assert_eq!(
        id(&req(0, 1024, 768), false, first_resource),
        None,
        "an unbound depth attachment names no resident"
    );
    assert_eq!(
        depth_stencil_chain_identity(
            &DrawEncodeRequest {
                colors: vec![ColorRtRequest {
                    width: 1024,
                    height: 768,
                    ..Default::default()
                }],
                ..DrawEncodeRequest::default()
            },
            0,
            false,
            first_resource,
        ),
        None,
        "and neither does a pass with no depth attachment at all"
    );
}

#[test]
fn depth_identity_separates_equal_refs_across_tasks_and_object_reuse() {
    let state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let resources = &state.task_objects.resources;
    let make = || {
        std::sync::Arc::new(crate::model::TaskResource::new(
            reims_vgpu_protocol::ObjectListEntry::new(ObjectKind::Texture, 0, 0),
            std::sync::Arc::from([]),
        ))
    };
    let first = resources.register(1, 42, make()).semantic_id().unwrap();
    let other_task = resources.register(2, 42, make()).semantic_id().unwrap();
    assert_ne!(
        first, other_task,
        "task-local ref 42 is not a global identity"
    );

    assert!(resources.delete(1, 42));
    let replacement = resources.register(1, 42, make()).semantic_id().unwrap();
    assert_ne!(
        first, replacement,
        "object deletion ends the old generation"
    );
}

#[test]
fn pass_depth_and_stencil_attachments_are_independent_and_invalid_pairs_refuse() {
    use reims_vgpu_protocol::pass_action::{LoadAction, StoreAction};

    let state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let resources = &state.task_objects.resources;
    let resource = resources.register(
        1,
        42,
        std::sync::Arc::new(crate::model::TaskResource::new(
            reims_vgpu_protocol::ObjectListEntry::new(ObjectKind::Texture, 0, 0),
            std::sync::Arc::from([]),
        )),
    );
    let depth_owner = resource.lifetime_ref().id();
    let stencil_resource = resources.register(
        1,
        43,
        std::sync::Arc::new(crate::model::TaskResource::new(
            reims_vgpu_protocol::ObjectListEntry::new(ObjectKind::Texture, 0, 0),
            std::sync::Arc::from([]),
        )),
    );
    let stencil_owner = stencil_resource.lifetime_ref().id();

    let base = DrawEncodeRequest {
        task_id: 1,
        colors: vec![ColorRtRequest {
            width: 8,
            height: 6,
            ..Default::default()
        }],
        depth_attach: Some(DepthAttachmentState {
            texture_ref: 42,
            load_action: LoadAction::Clear,
            store_action: StoreAction::Store,
            clear_depth: 0.25,
        }),
        depth_attachment_resource: Some(resource),
        // Nil state is deliberate: pass load/store does not depend on this.
        depth_stencil_ref: 0,
        ..DrawEncodeRequest::default()
    };
    let attachment = semantic_depth_attachment(&base)
        .expect("the declared attachment is supported")
        .expect("a bound depth texture is a pass attachment");
    assert_eq!(
        attachment.depth,
        Some(reims_vgpu_core::DepthAspectAttachment {
            load_action: LoadAction::Clear,
            store_action: StoreAction::Store,
            clear_value: 0.25,
        })
    );
    assert_eq!(attachment.stencil, None);
    assert_eq!(attachment.resource_lifetime.id(), depth_owner);
    assert!(attachment.resource_lifetime.is_live());

    let stencil_only = DrawEncodeRequest {
        colors: vec![ColorRtRequest {
            width: 8,
            height: 6,
            ..Default::default()
        }],
        stencil_attach: Some(StencilAttachmentState {
            texture_ref: 43,
            load_action: LoadAction::Clear,
            store_action: StoreAction::Store,
            clear_stencil: 7,
        }),
        stencil_attachment_resource: Some(stencil_resource),
        ..DrawEncodeRequest::default()
    };
    let attachment = semantic_depth_attachment(&stencil_only)
        .expect("a stencil-only attachment is supported")
        .expect("a bound stencil texture is a pass attachment");
    assert_eq!(attachment.depth, None);
    assert_eq!(
        attachment.stencil,
        Some(reims_vgpu_core::StencilAttachment {
            load_action: LoadAction::Clear,
            store_action: StoreAction::Store,
            clear_value: 7,
        })
    );
    assert_eq!(attachment.resource_lifetime.id(), stencil_owner);

    let mismatch = DrawEncodeRequest {
        stencil_attach: Some(StencilAttachmentState {
            texture_ref: 43,
            ..Default::default()
        }),
        ..base
    };
    assert!(matches!(
        semantic_depth_attachment(&mismatch),
        Err(DrawPreparationDecline::DepthStencilAttachmentMismatch {
            depth_ref: 42,
            stencil_ref: 43,
        })
    ));
}

/// Only a texture gap the fragment module statically uses is substituted for.
///
/// The three narrowings are the whole content of the rule and each one fails in
/// a different direction. Filling a `DeclaredUnused` gap pays a descriptor for a
/// variable nothing references and destroys the census that separated it from a
/// real violation; filling an `Ambiguous` one picks between two variables on a
/// binding, which is a guess; filling a buffer or sampler gap invents a resource
/// where the caller either has its own default or has no neutral at all. Leaving
/// a `Used` texture gap alone is the one that kills the host process, because
/// Mesa's Intel driver divides `(use_count << 7)` by the array size its own
/// zero-fill gave the binding the layout never declared.
#[test]
fn only_a_statically_used_texture_gap_is_bound_as_null() {
    use reims_vgpu_core::DescriptorUse;

    let gap = |class, metal_index| FragUnbound { class, metal_index };
    let uses = vec![
        (gap(FragUnboundClass::Texture, 3), DescriptorUse::Used),
        (
            gap(FragUnboundClass::Texture, 4),
            DescriptorUse::DeclaredUnused,
        ),
        (gap(FragUnboundClass::Texture, 5), DescriptorUse::Ambiguous),
        (
            gap(FragUnboundClass::Texture, 6),
            DescriptorUse::NotDeclared,
        ),
        (gap(FragUnboundClass::Buffer, 7), DescriptorUse::Used),
        (gap(FragUnboundClass::Sampler, 8), DescriptorUse::Used),
    ];

    assert_eq!(frag_unbound_textures_to_bind_null(&uses), vec![3]);
    // Nothing flagged, nothing substituted — the hot path.
    assert!(frag_unbound_textures_to_bind_null(&[]).is_empty());
}

#[test]
fn every_fragment_gap_uses_the_executable_descriptor_verdict() {
    use reims_vgpu_core::{DescriptorUse, PreparedShaderStage, PreparedShaderVariant};
    use reims_vgpu_protocol::PreparedShaderId;
    use std::sync::Arc;

    let variant = PreparedShaderVariant {
        program: PreparedShaderStage {
            id: PreparedShaderId::new(1),
            used_descriptor_bindings: Arc::from([]),
        },
        samplers: Arc::from([]),
        declared_bindings: Arc::from([41, 52, 63]),
        descriptor_uses: Arc::from([
            (41, DescriptorUse::DeclaredUnused),
            (52, DescriptorUse::Used),
            (63, DescriptorUse::Ambiguous),
        ]),
        texture_uses: Arc::from([(2, DescriptorUse::NotDeclared)]),
        buffer_binding_base: 40,
        texture_binding_base: 50,
        sampler_binding_base: 60,
        word_count: 0,
    };
    let gap = |class, metal_index| FragUnbound { class, metal_index };

    assert_eq!(
        frag_unbound_static_use(&gap(FragUnboundClass::Buffer, 1), &variant),
        DescriptorUse::DeclaredUnused
    );
    assert_eq!(
        frag_unbound_static_use(&gap(FragUnboundClass::Texture, 2), &variant),
        DescriptorUse::NotDeclared
    );
    assert_eq!(
        frag_unbound_static_use(&gap(FragUnboundClass::Sampler, 3), &variant),
        DescriptorUse::Ambiguous
    );
}

#[test]
fn only_unrepaired_fragment_gaps_reach_the_failure_path() {
    use reims_vgpu_core::DescriptorUse;

    let gap = |class| FragUnbound {
        class,
        metal_index: 1,
    };
    assert!(!frag_unbound_requires_report(
        gap(FragUnboundClass::Buffer),
        DescriptorUse::DeclaredUnused
    ));
    assert!(!frag_unbound_requires_report(
        gap(FragUnboundClass::Sampler),
        DescriptorUse::NotDeclared
    ));
    assert!(!frag_unbound_requires_report(
        gap(FragUnboundClass::Texture),
        DescriptorUse::Used
    ));
    assert!(frag_unbound_requires_report(
        gap(FragUnboundClass::Buffer),
        DescriptorUse::Used
    ));
    assert!(frag_unbound_requires_report(
        gap(FragUnboundClass::Sampler),
        DescriptorUse::Ambiguous
    ));
}

/// A retained depth-stencil state is served without consulting guest memory,
/// and the delete opcode is what takes it away.
///
/// The point is the *absence* of the guest read, so the test removes the guest:
/// the host has no object list and no descriptor bytes at all, which is exactly
/// what `resolve_descriptor` needs and cannot get. A build that still walks
/// guest memory per draw fails here with a ladder slug; the retaining one
/// answers from the registry.
///
/// This is what makes the 0.43-0.47 us a chain go away — an object-list lookup,
/// a descriptor read, an `Arc<[u8]>` allocation and a decode, on every draw that
/// bound any depth state, for an immutable object the guest publishes once and
/// deletes explicitly.
#[test]
fn a_retained_depth_stencil_state_is_served_without_reading_guest_memory() {
    use crate::runtime::decode::resource::DepthStencilDescriptor;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let host = FakeHost::new();
    state.define_task(2, 0x2000, 9);

    // Nothing is published, so the only way to an answer is the registry.
    assert!(
        super::load_depth_stencil_descriptor(&state, &host, 2, 7).is_err(),
        "with no retained state and no guest bytes there is no answer to give"
    );

    let retained = DepthStencilDescriptor {
        depth_stencil_id: 7,
        depth_compare_function: 3,
        depth_write_enabled: true,
        ..Default::default()
    };
    state.task_objects.depth_stencil.register(
        2,
        reims_vgpu_protocol::SerializerRef::new(7),
        std::sync::Arc::new(retained.clone()),
    );

    assert_eq!(
        super::load_depth_stencil_descriptor(&state, &host, 2, 7).ok(),
        Some(retained),
        "the retained state answers with no guest read available at all"
    );

    assert!(state
        .task_objects
        .depth_stencil
        .delete(2, reims_vgpu_protocol::SerializerRef::new(7)));
    assert!(
        super::load_depth_stencil_descriptor(&state, &host, 2, 7).is_err(),
        "and the delete is a real invalidation, not a counter"
    );
}

/// Every physical slice is read from its own offset, an inter-slice advance
/// apart.
///
/// A cube's six faces and a texture array's layers are the same packing:
/// `base_offset` names slice 0 and each later slice sits one `bytes_per_slice`
/// further on. The loader used to name only a level, resolving it against
/// slice 0 unconditionally, so every layer of a multi-slice texture served the
/// first layer's bytes — six identical faces that read as a working cube. The
/// two slices below carry deliberately different content, so a loader that
/// ignores the slice returns the same bytes twice and fails here.
///
/// This is also why the arithmetic bounding the read moved onto
/// `subresource_offset`: `base_offset + level.offset` is the true end of the
/// allocation only for slice 0, and would bound a read of the last slice
/// against the extent of the first.
#[test]
fn a_later_slice_is_read_at_its_own_inter_slice_advance() {
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, TEXTURE_DESC_BASE_LEN,
        TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE, TEXTURE_DESC_USED_SIZE,
        TEXTURE_DESC_WIDTH,
    };
    use reims_vgpu_core::endian::{st16, st32, st64};
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let page = 1u64 << PAGE_SHIFT_ARM64E;
    let dir_pfn = 2u32;
    let root_pfn = 3u32;
    let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_ARM64E;
    let root_gpa = (root_pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x4000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], root_pfn);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    assert!(host.write_gpa(dir_gpa, &d).is_ok());
    for i in 0..4u32 {
        let pfn = 4 + i;
        host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        assert!(host.write_gpa(root_gpa + (i as u64) * 4, &pte).is_ok());
    }
    state.define_task(1, 0x1000, dir_pfn);
    assert!(state.set_object_list(1, 0, 32));

    // Two slices of a tight 4x2 BGRA8 level, one whole page apart. The advance
    // is deliberately far larger than the 32 bytes a slice occupies: the
    // contract says slices are separated by `bytes_per_slice`, not packed.
    let tex_ref = 6u32;
    let slices = 2u16;
    let bytes_per_slice = page;
    let mut b = vec![0u8; TEXTURE_DESC_BASE_LEN];
    st64(&mut b[0..], 4 * page); // allocation_size
    st32(&mut b[8..], 1); // handle -> slice 0 at gva 1 << page_shift
    write_linear_texture_packing(&mut b, 1, slices, 0, bytes_per_slice);
    st32(&mut b[TEXTURE_DESC_USED_SIZE..], 16 * 2);
    st32(&mut b[TEXTURE_DESC_ROW_STRIDE..], 16);
    st32(&mut b[TEXTURE_DESC_WIDTH..], 4);
    st32(&mut b[TEXTURE_DESC_WIDTH + 4..], 2); // height
    st16(&mut b[TEXTURE_DESC_PIXEL_FORMAT..], MTL_FORMAT_BGRA8_UNORM);
    let desc_gva = 0x200u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &b);
    let off = list_object_entry_offset(tex_ref, 32).unwrap();
    let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut le[0..],
        (OBJECT_TYPE_TEXTURE as u32) | ((TEXTURE_DESC_BASE_LEN as u32) << 8),
    );
    le[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);

    let slice0 = [7u8, 5, 3, 255].repeat(8);
    let slice1 = [90u8, 60, 30, 255].repeat(8);
    write_task_gva_arm64e(&mut host, &state.tasks[1], page, &slice0);
    write_task_gva_arm64e(&mut host, &state.tasks[1], page + bytes_per_slice, &slice1);

    let load = |state: &mut Device, host: &mut FakeHost, slice: u32| {
        texture_view::load_linear_texture_host(
            state,
            host,
            1,
            tex_ref,
            slice,
            0,
            None,
            NativeUploads::NONE,
            crate::runtime::render_writeback::SettleSite::LinearTextureSampled,
        )
        .expect("both slices are inside the declared allocation")
        .0
    };

    // `NativeUploads::NONE` says the executor accepts no native BGRA8, so the
    // loader converts each row on the way out. The two patterns arrive
    // channel-swapped and still distinct, which is all this test reads them for.
    let got0 = load(&mut state, &mut host, 0);
    let got1 = load(&mut state, &mut host, 1);
    assert_eq!(
        &got0[..4],
        &[3, 5, 7, 255],
        "slice 0 still reads from base_offset"
    );
    assert_eq!(
        &got1[..4],
        &[30, 60, 90, 255],
        "slice 1 must read one bytes_per_slice on, not repeat slice 0"
    );
}

/// Dispatching on the guest's texture declaration selects exactly what
/// dispatching on the shader's reflected shape selected.
///
/// `declared_guest_image_selection` used to `match` on the shape the *shader*
/// declared and then check the guest's texture descriptor agreed. That is the
/// wrong way round for the contract — a Metal texture object carries its own
/// type and a shader must match it, so the shape can only veto — and it is also
/// the reason the whole resolution had to be redone on every draw, because a
/// shape belongs to whichever pipeline is bound while a declaration belongs to
/// the texture.
///
/// The two must be the same function, and the only way to know that over the
/// whole input space is to keep both and compare. This walks every shape kind
/// against every storage type, every view type including none, a level that is
/// and is not one texel tall, and slice ranges that are in and out of bounds —
/// and asserts the two agree on **every** cell, `None` included, because a cell
/// where one refuses and the other selects is exactly the silent wrong-view bug
/// this inversion could have introduced.
#[test]
fn the_declaration_and_the_shader_shape_select_the_same_layout() {
    use crate::runtime::decode::resource::{
        TEXTURE_VIEW_MTL_TYPE_1D, TEXTURE_VIEW_MTL_TYPE_1D_ARRAY, TEXTURE_VIEW_MTL_TYPE_2D,
        TEXTURE_VIEW_MTL_TYPE_2D_ARRAY, TEXTURE_VIEW_MTL_TYPE_3D, TEXTURE_VIEW_MTL_TYPE_CUBE,
        TEXTURE_VIEW_MTL_TYPE_CUBE_ARRAY,
    };
    use crate::runtime::draw::sampled_source::selection_dispatched_on_the_shader_shape as oracle;

    const TYPES: [u16; 7] = [
        TEXTURE_VIEW_MTL_TYPE_1D,
        TEXTURE_VIEW_MTL_TYPE_1D_ARRAY,
        TEXTURE_VIEW_MTL_TYPE_2D,
        TEXTURE_VIEW_MTL_TYPE_2D_ARRAY,
        TEXTURE_VIEW_MTL_TYPE_3D,
        TEXTURE_VIEW_MTL_TYPE_CUBE,
        TEXTURE_VIEW_MTL_TYPE_CUBE_ARRAY,
    ];
    let kinds = [
        reims_vgpu_core::SampledImageKind::D1,
        reims_vgpu_core::SampledImageKind::D1Array,
        reims_vgpu_core::SampledImageKind::D2,
        reims_vgpu_core::SampledImageKind::D2Array,
        reims_vgpu_core::SampledImageKind::D3,
        reims_vgpu_core::SampledImageKind::Cube,
    ];

    let mut compared = 0usize;
    let mut reached: std::collections::BTreeMap<u16, usize> = std::collections::BTreeMap::new();
    for storage in TYPES {
        for array_length in [1u16, 2, 6] {
            for height in [1u32, 64] {
                let level = crate::runtime::decode::resource::TextureLevelLayout {
                    offset: 0,
                    size: 0x4000,
                    row_stride: 0x100,
                    width: 64,
                    height,
                    depth: 1,
                };
                let cube_faces = matches!(
                    storage,
                    TEXTURE_VIEW_MTL_TYPE_CUBE | TEXTURE_VIEW_MTL_TYPE_CUBE_ARRAY
                );
                let physical_slices = u64::from(array_length) * if cube_faces { 6 } else { 1 };
                let texture = TextureDescriptor {
                    allocation_size: 0x4000 * physical_slices,
                    declaration: Some(reims_vgpu_protocol::TextureDeclaration {
                        texture_type: storage as u8,
                        framebuffer_only: false,
                        is_drawable: false,
                        write_swizzle_enabled: None,
                        allow_gpu_optimized_contents: false,
                        usage: 0,
                        pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
                        width: level.width,
                        height: level.height,
                        depth: 1,
                        mipmap_level_count: 1,
                        sample_count: 1,
                        array_length,
                        resource_options: 0,
                        protection_options: 0,
                        swizzle: None,
                    }),
                    bytes_per_slice: 0x4000,
                    slice_count: u32::from(array_length),
                    cube_faces,
                    levels: vec![level],
                    ..Default::default()
                };
                for kind in kinds {
                    let Some(shape) = reims_vgpu_core::sampled_image_shape(kind) else {
                        continue;
                    };
                    for view_type in [None].into_iter().chain(TYPES.map(Some)) {
                        for range in [
                            None,
                            Some(TextureViewRange {
                                level_base: 0,
                                level_count: 1,
                                slice_base: 0,
                                slice_count: 1,
                            }),
                            Some(TextureViewRange {
                                level_base: 0,
                                level_count: 1,
                                slice_base: 1,
                                slice_count: 1,
                            }),
                            Some(TextureViewRange {
                                level_base: 0,
                                level_count: 1,
                                slice_base: 0,
                                slice_count: 6,
                            }),
                            // Out of bounds on every array_length above.
                            Some(TextureViewRange {
                                level_base: 0,
                                level_count: 1,
                                slice_base: 5,
                                slice_count: 4,
                            }),
                            // A zero count, which must refuse rather than
                            // producing a layerless image.
                            Some(TextureViewRange {
                                level_base: 0,
                                level_count: 1,
                                slice_base: 0,
                                slice_count: 0,
                            }),
                        ] {
                            let declared_now = view_type.unwrap_or(storage);
                            let want = oracle(shape, &texture, &level, view_type, range);
                            let got = declared_guest_image_selection(
                                shape, &texture, &level, view_type, range,
                            );
                            assert_eq!(
                                got, want,
                                "storage={storage} array_length={array_length} height={height} \
                                 kind={kind:?} view_type={view_type:?} range={range:?}"
                            );
                            compared += 1;
                            if got.is_some() {
                                // Coverage is per *dispatch arm*, keyed by the
                                // type the declaration named, because a bulk
                                // count of selections cannot see an arm that
                                // stopped being reachable — which is exactly
                                // how the cube arm sat unexercised here while
                                // the total read healthy.
                                *reached.entry(declared_now).or_insert(0usize) += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    // A run where an arm never selected compares `None` against `None` for
    // every one of its cells and proves nothing about it, which is how this
    // test fails silently when a grid field stops being representable. The
    // cube arm did exactly that on the first draft — the allocation was one
    // sixth of what a cube's six faces need, so every cube cell was refused
    // before the dispatch and a deliberate break to the six-face rule went
    // undetected. Name every arm and require each to have selected.
    assert!(compared >= 2000, "the grid shrank: compared={compared}");
    for arm in [
        TEXTURE_VIEW_MTL_TYPE_1D,
        TEXTURE_VIEW_MTL_TYPE_1D_ARRAY,
        TEXTURE_VIEW_MTL_TYPE_2D,
        TEXTURE_VIEW_MTL_TYPE_2D_ARRAY,
        TEXTURE_VIEW_MTL_TYPE_3D,
        TEXTURE_VIEW_MTL_TYPE_CUBE,
    ] {
        assert!(
            reached.get(&arm).copied().unwrap_or(0) > 0,
            "no cell selected through the {arm} arm, so nothing here tests it: {reached:?}"
        );
    }
}
