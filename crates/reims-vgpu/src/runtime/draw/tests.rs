use super::*;
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
use crate::runtime::gva_mem::write_task_gva_arm64e;
use crate::runtime::host::FakeHost;

#[cfg(feature = "backend-vulkan")]
#[test]
fn m2v_draw_boundary_preserves_the_engine_vk_call_slug() {
    let req = DrawEncodeRequest {
        pipeline_ref: 73,
        ..DrawEncodeRequest::default()
    };
    let err = crate::backend::vulkan::engine::vk_call::exec_submit_device_lost_fixture();

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
fn test_triangle() -> crate::contract::draw::DrawArgs {
    crate::contract::draw::DrawArgs {
        vertex_count: 3,
        instance_count: 1,
        primitive_type: 3,
        first_vertex: 0,
        base_instance: 0,
    }
}

/// `mrt_draw_request` for the single-RT [`test_triangle`] these tests resolve
/// targets with: attachment in slot 0, no depth. Five bodies passed those
/// trailing literals; only `pipeline_ref` and the attachment ever varied.
fn single_rt_draw_request<M: HostMemory + HostOps>(
    state: &mut DeviceState,
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
}

/// One reflection binding of `kind` at `metal_index` for the
/// `frag_declared_unbound` guard tests. Only kind + index are load-bearing.
#[cfg(feature = "backend-vulkan")]
fn rb(
    kind: metal2vulkan::reflect::ResourceKind,
    metal_index: u32,
) -> metal2vulkan::reflect::ResourceBinding {
    metal2vulkan::reflect::ResourceBinding {
        kind,
        metal_index,
        descriptor: None,
        param_index: None,
        stage_input_location: None,
        address_space: None,
        declared_size: None,
        extent: None,
        footprint: None,
        type_layout: None,
        type_name: None,
        texture_shape: None,
        embedded_source: None,
        access: None,
        static_sampler: None,
    }
}

/// A proven shader extent narrows the transfer without changing rail admission.
///
/// The full guest window clears the gather crossover; its 64-byte bounded
/// object is still served through that rail and the held-resolution registry,
/// but only those 64 bytes are walked and copied. This is the behavioral split
/// that an earlier implementation lost by comparing the cap to the floor.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_small_reflected_object_stays_on_the_gather_rail_and_moves_only_its_extent() {
    use super::vulkan::gather_span_if_eligible;

    let full = ZERO_COPY_BUFFER_MIN_BYTES * 4;
    assert_eq!(gather_span_if_eligible(full, Some(64)), Some(64));
    assert_eq!(gather_span_if_eligible(full, None), Some(full));
    assert_eq!(
        gather_span_if_eligible(ZERO_COPY_BUFFER_MIN_BYTES - 1, Some(64)),
        None,
        "a genuinely sub-floor guest window stays off the gather rail"
    );
}

/// The scatter band names the run count the coalescer finds, not the page count.
///
/// The reading this census exists to take is "could this window have been served
/// by N RAMBlock references instead of an allocation", and N is runs, not pages.
/// A four-page window laid out as one 16 KiB granule is one run and the answer
/// is one reference; the same four pages shuffled are four. Banding by page
/// count would report the same number for both and rank the decision wrongly.
#[cfg(feature = "backend-vulkan")]
#[test]
fn the_packed_scatter_band_counts_runs_and_not_pages() {
    use super::vulkan::packed_scatter_band;

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

/// The zero-copy buffer floor may veto a gather and nothing else.
///
/// `ZERO_COPY_BUFFER_MIN_BYTES` was measured against one outcome and its own doc
/// says which: removing it moved `submit_us` by 13 % because every gathered bind
/// is a recorded GPU gather, while `stage_us` — the CPU read it was believed to
/// be saving — barely moved. It was nevertheless asked at the top of the rail,
/// before the window is walked, which is before "would this gather?" can be
/// answered at all. So it also refused windows that resolve to pages, which are
/// an offset into an import this device already holds for the VM's lifetime and
/// cost no submit.
///
/// That refusal was not free, because a resolution is cached and a refusal is
/// not: a refused bind re-resolves its backing and re-reads its bytes on the CPU
/// on every later draw. A driven macos-13 Maps leg scored `zc_buffer_below_floor`
/// 2 909 326 against `zc_buffer_held` 1 820 542.
///
/// The asymmetry is the whole rule, so this is a truth table and not a
/// threshold check. The first row is the one that regressed to a repeated CPU
/// read, and a change that reinstates "sub-floor never binds" fails on it.
#[cfg(feature = "backend-vulkan")]
#[test]
fn the_zero_copy_buffer_floor_vetoes_a_gather_and_nothing_else() {
    use super::vulkan::window_binds_zero_copy;

    assert!(
        window_binds_zero_copy(false, false),
        "a sub-floor window that resolved to pages needs no gather, so the \
         floor has nothing to say about it — this is the bind that otherwise \
         takes a fresh CPU read on every draw forever"
    );
    assert!(
        !window_binds_zero_copy(true, false),
        "a sub-floor window that would be gathered is exactly what the floor \
         was measured against"
    );
    assert!(
        window_binds_zero_copy(true, true),
        "an eligible window still gathers"
    );
    assert!(
        window_binds_zero_copy(false, true),
        "an eligible window that needs no gather binds directly"
    );
}

/// The window math the three sampled zero-copy rails now share.
///
/// Each rail used to carry its own copy, so each could have drifted alone. The
/// two refusals are the ones that matter: a stride narrower than one tight row
/// describes an image the rows do not fit in, and a stride that is not a whole
/// number of texels has no `bufferRowLength` — that field counts texels, not
/// bytes, so the copy would stride to the wrong place.
#[test]
#[cfg(feature = "backend-vulkan")]
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
    // Single-byte texels (the type-5 NV12 luma plane) take every stride.
    assert_eq!(strided_window_extent(64, 4, 1, 64), Some((256, 0)));
    assert_eq!(strided_window_extent(64, 4, 1, 96), Some((96 * 3 + 64, 96)));
}

#[test]
#[cfg(feature = "backend-vulkan")]
fn type11_zero_copy_declines_transient_host_mappings() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
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
        let m = state.mappings.get_mut(&mid).unwrap();
        m.page_entries = (0..page_count)
            .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
            .collect();
    }
    assert!(state.set_mapping_geom(mid, width, height, MTL_FORMAT_BGRA8_UNORM));

    let resource = crate::model::TaskResource::new(Default::default(), std::sync::Arc::from([]));
    // The window's pages are one GPA-contiguous run, so a host that would
    // reconstruct even *this* is the only kind that has to fall back -- and it
    // has to fall back on the alias verdict rather than on the page walk, which
    // is why the map is asked before the refusal rather than skipped.
    host.transient_map_pages = true;
    assert!(try_type11_sample_zero_copy(
        &mut state,
        &mut host,
        mid,
        width,
        height,
        resource.lifetime_ref(),
    )
    .is_none());
    assert!(
        host.map_pages_calls > 0,
        "the refusal must come from the host's answer about the view it built, \
         not from declining to ask"
    );
    assert!(
        state
            .mappings
            .get(&mid)
            .is_some_and(|m| m.contig_import.is_none()),
        "and nothing importable may be retained over a view the host owes a \
         release for"
    );
}

#[test]
#[cfg(feature = "backend-vulkan")]
fn mapping_sampled_planes_reuse_one_resource_owned_import() {
    // The guest-RAM map resolves once per process and every test in this binary
    // shares it, so a fixture that maps its own RAM has to discard whatever an
    // earlier test resolved. Without this the alias rail asks a map built from
    // some other test's host and refuses — which is the right refusal against
    // the wrong host.
    crate::runtime::guest_ram_map::reset();
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let page = 1u64 << PAGE_SHIFT_X86;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let mid = 17u32;
    let gpa0 = 0x4100_0000u64;
    let pages = 16u32;
    host.map_range(gpa0, (u64::from(pages) * page) as usize, 0);
    assert!(state.map_surface(mid));
    {
        let mapping = state.mappings.get_mut(&mid).unwrap();
        mapping.page_entries = (0..pages)
            .map(|i| {
                ((((gpa0 >> PAGE_SHIFT_X86) as u32) + i) << PAGE_ENTRY_PFN_SHIFT)
                    | PAGE_ENTRY_VALID
            })
            .collect();
    }
    assert!(state.set_mapping_geom(
        mid,
        128,
        128,
        crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB,
    ));
    crate::runtime::guest_ram::latch_import_limits(page, 1 << 30, 1 << 30);
    let type11_witnesses = crate::runtime::drain::store_route_count("gw_rail_t11");
    let type5_witnesses = crate::runtime::drain::store_route_count("gw_rail_t5");
    let type11_resource =
        crate::model::TaskResource::new(Default::default(), std::sync::Arc::from([]));
    let type5_resource =
        crate::model::TaskResource::new(Default::default(), std::sync::Arc::from([]));

    let type11 = try_type11_sample_zero_copy(
        &mut state,
        &mut host,
        mid,
        128,
        128,
        type11_resource.lifetime_ref(),
    )
    .expect("the mapping's color plane is sampleable");
    let SampledSourceRequest::GuestRuns(
        type11,
        _,
        type11_format,
        _,
        type11_identity,
        ..
    ) = type11
    else {
        panic!("the mapping stays guest-backed")
    };
    assert_eq!(type11_format, ash::vk::Format::B8G8R8A8_SRGB);
    assert_eq!(type11_identity, None);
    assert_eq!(
        crate::runtime::drain::store_route_count("gw_rail_t11"),
        type11_witnesses,
        "a direct resource has no copied image whose freshness needs witnessing"
    );
    let type11_import = type11
        .direct_image
        .expect("stable allocation binds directly")
        .import;

    let type5 = try_type5_sample_zero_copy(
        &mut state,
        &mut host,
        mid,
        objects::Type5TextureView {
            pixel_format: MTL_FORMAT_BGRA8_UNORM,
            width: 128,
            height: 128,
            depth: 1,
            plane_index: 0,
        },
        type5_resource.lifetime_ref(),
    )
    .expect("the serialized plane view is sampleable");
    let SampledSourceRequest::GuestRuns(type5, _, type5_format, _, type5_identity, ..) = type5 else {
        panic!("the plane view stays guest-backed")
    };
    assert_eq!(type5_format, ash::vk::Format::B8G8R8A8_UNORM);
    assert_eq!(type5_identity, None);
    assert_eq!(
        crate::runtime::drain::store_route_count("gw_rail_t5"),
        type5_witnesses,
        "a direct plane has no copied image whose freshness needs witnessing"
    );
    let type5_import = type5
        .direct_image
        .expect("stable allocation binds directly")
        .import;
    crate::runtime::guest_ram::forget_import_limits();

    assert_eq!(
        type11_import.id(),
        type5_import.id(),
        "two views of one mapping must retain the mapping's one import"
    );
    assert!(std::sync::Arc::ptr_eq(&type11_import, &type5_import));
    assert_eq!(
        state.mappings[&mid]
            .contig_import
            .as_ref()
            .map(|import| import.id()),
        Some(type11_import.id())
    );
}

#[test]
#[cfg(feature = "backend-vulkan")]
fn small_mapping_sampled_plane_uses_its_direct_resource() {
    // The guest-RAM map resolves once per process and every test in this binary
    // shares it, so a fixture that maps its own RAM has to discard whatever an
    // earlier test resolved. Without this the alias rail asks a map built from
    // some other test's host and refuses — which is the right refusal against
    // the wrong host.
    crate::runtime::guest_ram_map::reset();
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let page = 1u64 << PAGE_SHIFT_X86;
    let (width, height) = (16u32, 16u32);
    let span = u64::from(width) * u64::from(height) * 4;
    assert!(span < SAMPLED_GATHER_MIN_BYTES);

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let mid = 18u32;
    let gpa = 0x4200_0000u64;
    host.map_range(gpa, page as usize, 0);
    assert!(state.map_surface(mid));
    state.mappings.get_mut(&mid).unwrap().page_entries = vec![
        ((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT | PAGE_ENTRY_VALID,
    ];
    assert!(state.set_mapping_geom(mid, width, height, MTL_FORMAT_BGRA8_UNORM));
    crate::runtime::guest_ram::latch_import_limits(page, 1 << 30, 1 << 30);
    let resource = crate::model::TaskResource::new(Default::default(), std::sync::Arc::from([]));

    let sampled = try_type11_sample_zero_copy(
        &mut state,
        &mut host,
        mid,
        width,
        height,
        resource.lifetime_ref(),
    )
    .expect("a directly-backed sampled resource has no size crossover");
    let SampledSourceRequest::GuestRuns(source, ..) = sampled else {
        panic!("the mapping stays guest-backed")
    };
    crate::runtime::guest_ram::forget_import_limits();

    assert!(
        source.direct_image.is_some(),
        "the gather floor may govern a copy, not replace a valid direct resource"
    );
}

#[test]
#[cfg(feature = "backend-vulkan")]
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
#[cfg(feature = "backend-vulkan")]
fn tight_linear_cpu_fallback_reads_every_depth_plane() {
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;

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
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::model::SurfaceWriteKind;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let (mid, width, height) = (5u32, 64u32, 48u32);
    assert!(state.map_surface(mid));
    {
        let mapping = state.mappings.get_mut(&mid).unwrap();
        mapping.mapped = true;
        mapping.has_geom = true;
        mapping.width = width;
        mapping.height = height;
        mapping.format = MTL_FORMAT_BGRA8_UNORM;
        mapping.page_entries = vec![(1 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        mapping.content_generation = 2;
    }
    state.present.valid = true;
    state.present.width = width;
    state.present.height = height;
    state.present.frame_flush_seen = true;

    publish_surface_store(
        &mut state,
        &mut host,
        mid,
        width,
        height,
        MTL_FORMAT_BGRA8_UNORM,
    );

    assert_eq!(state.surface_write_kind(mid), SurfaceWriteKind::Composite);
    assert_eq!(state.present.early_front_mapping, mid);
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn frag_unbound_scan_reports_only_missing_standard_kinds() {
    use crate::runtime::draw::{FragUnbound, FragUnboundClass};
    use metal2vulkan::reflect::ResourceKind as K;
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
    // binding relocation does not have to parse it back out of a formatted
    // string. `Display` is the only place the prefix exists.
    assert_eq!(tex(3).to_string(), "tex3");
    assert_eq!(buf(2).to_string(), "buf2");
    assert_eq!(smp(0).to_string(), "smp0");
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn reflected_static_sampler_maps_exact_state_and_rejects_unimplemented_modes() {
    use crate::backend::vulkan::engine::{
        SamplerAddressMode as EngineAddress, SamplerFilter as EngineFilter,
        SamplerMipFilter as EngineMip,
    };
    use metal2vulkan::reflect::{
        SamplerAddressMode, SamplerBorderColor, SamplerCompareFunction, SamplerCoordinates,
        SamplerFilter, SamplerMipFilter, SamplerReduction, StaticSamplerState,
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

#[cfg(feature = "backend-vulkan")]
#[test]
fn reflected_sampled_collision_includes_sampler_band_only() {
    use metal2vulkan::reflect::ShaderStage;

    let vertex = shader_pull_reflection(&[64]);
    let mut fragment = shader_pull_reflection(&[64]);
    fragment.stage = ShaderStage::Fragment;
    assert!(reflected_sampled_binding_collision(&vertex, &fragment));

    fragment.bindings[0].descriptor.as_mut().unwrap().binding = 65;
    assert!(!reflected_sampled_binding_collision(&vertex, &fragment));

    fragment.bindings[0].descriptor.as_mut().unwrap().binding = 96;
    assert!(!reflected_sampled_binding_collision(&vertex, &fragment));
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn depth_stencil_triviality_matches_no_op_state() {
    use crate::runtime::decode::resource::DepthStencilDescriptor;
    // compare Always (7), no write, no stencil → equivalent to no depth test.
    let trivial = DepthStencilDescriptor {
        depth_compare_function: 7,
        depth_write_enabled: false,
        front_stencil_enabled: false,
        back_stencil_enabled: false,
        ..Default::default()
    };
    assert!(depth_stencil_descriptor_is_trivial(&trivial));

    // A real compare function (Less=1) occludes → non-trivial.
    assert!(!depth_stencil_descriptor_is_trivial(
        &DepthStencilDescriptor {
            depth_compare_function: 1,
            ..trivial.clone()
        }
    ));
    // Depth write on → non-trivial even with compare Always.
    assert!(!depth_stencil_descriptor_is_trivial(
        &DepthStencilDescriptor {
            depth_write_enabled: true,
            ..trivial.clone()
        }
    ));
    // Either stencil face enabled → non-trivial.
    assert!(!depth_stencil_descriptor_is_trivial(
        &DepthStencilDescriptor {
            front_stencil_enabled: true,
            ..trivial.clone()
        }
    ));
    assert!(!depth_stencil_descriptor_is_trivial(
        &DepthStencilDescriptor {
            back_stencil_enabled: true,
            ..trivial
        }
    ));
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn bound_depth_stencil_that_cannot_resolve_returns_named_reason() {
    // A guest that binds a depth-stencil ref (`ds_ref != 0`) whose object-list
    // entry does not resolve must surface a *specific* reason, not `None`: the
    // draw silently disables the depth test otherwise, and every other
    // depth/stencil degradation on this path is already fail-visible. With an
    // empty state the lookup misses → the entry-missing reason, which the caller
    // logs as `shader_state_degraded reason=depth_stencil_no_list_entry`.
    let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
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
    let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
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
    let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let host = FakeHost::new();

    // Unsupported MTLIndexType (only 0=u16 / 1=u32 exist).
    let bad_type = IndexedDrawInfo {
        base_vertex: 0,
        index_type: 5,
        index_count: 3,
        index_buffer_ref: 9,
        index_buffer_offset: 0,
    };
    assert_eq!(
        load_index_bytes_reason(&state, &host, 4, &bad_type),
        Err(IndexLoadReason::TypeUnsupported)
    );

    // Valid type + count, but the bound index buffer ref resolves to nothing on
    // an empty state → the entry-missing site, not a generic miss.
    let unresolved = IndexedDrawInfo {
        base_vertex: 0,
        index_type: 1,
        index_count: 6,
        index_buffer_ref: 9,
        index_buffer_offset: 0,
    };
    assert_eq!(
        load_index_bytes_reason(&state, &host, 4, &unresolved),
        Err(IndexLoadReason::EntryMissing)
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
/// `NoMetal` makes the exec loop honour the pass clear, `WritebackFailed` does
/// not — so a reader correlating a dropped draw with a black frame needs both
/// on the line.
#[test]
fn encode_status_renders_its_check_beside_the_class_it_collapsed_to() {
    use crate::observe::{Emit, Refusal as _};
    assert_eq!(
        Emit::refusal(
            "draw_encode_fail",
            &EncodeStatus::MissingMtlb("draw_mtl_vertex_mtlb_load")
        )
        .expect("a refusal must render a line")
        .render(),
        "draw_encode_fail reason=draw_mtl_vertex_mtlb_load class=missing_mtlb"
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

    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    {
        let backend =
            crate::backend::metal::error::Status::execute("metal_render_command_buffer_failed");
        let carried = EncodeStatus::MetalBackend(backend);
        assert_eq!(carried.class(), "metal_execute");
        assert_eq!(
            Emit::refusal("draw_encode_fail", &carried)
                .expect("the draw carrier must retain the backend refusal")
                .render(),
            "draw_encode_fail reason=metal_render_command_buffer_failed \
                 class=execute recovery=metal_failed"
        );
    }
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
#[test]
fn explicit_metal_sampler_and_depth_binds_return_typed_missing_entry_declines() {
    use crate::observe::Emit;

    let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();

    let sampler = load_sampler(&state, &host, 4, 77, 3)
        .expect_err("a nonzero sampler ref in an empty object list must decline");
    assert_eq!(
        sampler,
        MetalStateDecline::SamplerEntryMissing {
            sampler_ref: 77,
            index: 3,
        }
    );
    assert_eq!(
        Emit::decline("metal_draw_sampler_fallback", &sampler)
            .field("task", 4)
            .field("pipe", 19)
            .field("stage", "fragment")
            .render(),
        "metal_draw_sampler_fallback reason=metal_sampler_entry_missing \
             sampler_ref=77 index=3 task=4 pipe=19 stage=fragment"
    );

    let depth = load_depth_stencil_state(&state, &host, 4, 88)
        .expect_err("a nonzero depth-stencil ref in an empty object list must decline");
    assert_eq!(
        depth,
        MetalStateDecline::DepthStencilEntryMissing {
            depth_stencil_ref: 88,
        }
    );
    assert_eq!(
        Emit::decline("metal_draw_depth_stencil_fallback", &depth)
            .field("task", 4)
            .field("pipe", 19)
            .render(),
        "metal_draw_depth_stencil_fallback reason=metal_depth_stencil_entry_missing \
             depth_stencil_ref=88 task=4 pipe=19"
    );
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
#[test]
fn metal_state_decode_declines_delegate_the_exact_resource_decoder_reason() {
    use crate::observe::{Decline as _, Emit};

    let sampler = MetalStateDecline::SamplerDecode {
        sampler_ref: 41,
        index: 6,
        reason: DecodeStatus::ErrShort("res_sampler_short"),
    };
    assert_eq!(sampler.slug(), "res_sampler_short");
    assert_eq!(
        Emit::decline("metal_draw_sampler_fallback", &sampler).render(),
        "metal_draw_sampler_fallback reason=res_sampler_short \
             class=short sampler_ref=41 index=6"
    );

    let depth = MetalStateDecline::DepthStencilDecode {
        depth_stencil_ref: 52,
        reason: DecodeStatus::ErrShort("res_depth_stencil_short"),
    };
    assert_eq!(depth.slug(), "res_depth_stencil_short");
    assert_eq!(
        Emit::decline("metal_draw_depth_stencil_fallback", &depth).render(),
        "metal_draw_depth_stencil_fallback reason=res_depth_stencil_short \
             class=short depth_stencil_ref=52"
    );
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
#[test]
fn every_metal_icb_inheritance_check_is_unique_namespaced_and_log_safe() {
    use crate::observe::Decline as _;

    let all = vec![
        MetalIcbInheritanceDecline::CullModeUnsupported { value: 3 },
        MetalIcbInheritanceDecline::FrontFacingUnsupported { value: 2 },
        MetalIcbInheritanceDecline::BindSlotPastTable {
            bind: PastTableBind {
                class: BindTableClass::Buffer,
                stage: crate::runtime::decode::render::Stage::Vertex,
                index: MAX_BUFFER_BIND_SLOTS,
                resource_ref: 1,
            },
        },
        MetalIcbInheritanceDecline::VertexBufferMissing {
            buffer_ref: 3,
            index: 4,
            offset: 5,
        },
        MetalIcbInheritanceDecline::FragmentBufferMissing {
            buffer_ref: 6,
            index: 7,
            offset: 8,
        },
        MetalIcbInheritanceDecline::VertexTextureMissing {
            texture_ref: 11,
            index: 12,
            detail: "no list entry".into(),
        },
        MetalIcbInheritanceDecline::FragmentTextureMissing {
            texture_ref: 13,
            index: 14,
            detail: "guest\nread failed".into(),
        },
        MetalIcbInheritanceDecline::PipelineRefZero,
        MetalIcbInheritanceDecline::PipelineMissing { pipeline_ref: 17 },
        MetalIcbInheritanceDecline::VertexMtlbMissing { function_ref: 18 },
        MetalIcbInheritanceDecline::FragmentMtlbMissing { function_ref: 19 },
        MetalIcbInheritanceDecline::VertexLibraryLoad {
            function_ref: 20,
            detail: "Metal error".into(),
        },
        MetalIcbInheritanceDecline::FragmentLibraryLoad {
            function_ref: 21,
            detail: "Metal error".into(),
        },
        MetalIcbInheritanceDecline::VertexFunctionCount {
            function_ref: 22,
            count: 2,
        },
        MetalIcbInheritanceDecline::FragmentFunctionCount {
            function_ref: 23,
            count: 0,
        },
        MetalIcbInheritanceDecline::VertexFunctionGet {
            function_ref: 24,
            detail: "function missing".into(),
        },
        MetalIcbInheritanceDecline::FragmentFunctionGet {
            function_ref: 25,
            detail: "function missing".into(),
        },
        MetalIcbInheritanceDecline::VertexDescriptorMissing {
            pipeline_ref: 26,
            attribute_count: 3,
        },
        // The sibling of the entry above, and the pair is the point: an empty
        // vertex block and a partial one are different refusals, and the
        // builder used to report the second as a success.
        MetalIcbInheritanceDecline::VertexAttributeUnencodable {
            pipeline_ref: 26,
            location: 3,
            value: 99,
        },
        MetalIcbInheritanceDecline::RenderPipelineCreate {
            pipeline_ref: 27,
            detail: "pipeline failed".into(),
        },
        MetalIcbInheritanceDecline::AllocationFailed {
            what: "inherited_texture",
        },
    ];

    let mut slugs = all.iter().map(Decline::slug).collect::<Vec<_>>();
    assert_eq!(slugs.len(), 26, "the fixture must cover every check");
    for decline in &all {
        assert!(decline.slug().starts_with("metal_icb_inherit_"));
        for (key, value) in decline.fields() {
            assert!(!key.is_empty());
            assert!(
                !value.chars().any(char::is_whitespace),
                "{} rendered a non-token field {key}={value:?}",
                decline.slug()
            );
        }
    }
    slugs.sort_unstable();
    slugs.dedup();
    assert_eq!(slugs.len(), all.len(), "two ICB checks share one reason");
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
#[test]
fn metal_icb_inheritance_line_keeps_pipeline_and_sanitized_driver_detail() {
    use crate::observe::Emit;

    let decline = MetalIcbInheritanceDecline::RenderPipelineCreate {
        pipeline_ref: 71,
        detail: "Error Domain=MTLLibrary Code=3".into(),
    };
    assert_eq!(
        Emit::decline("metal_icb_inheritance", &decline)
            .field("task", 2)
            .field("pipe", 71)
            .field("icb", 99)
            .render(),
        "metal_icb_inheritance reason=metal_icb_inherit_render_pipeline_create \
             pipeline_ref=71 detail=Error_Domain=MTLLibrary_Code=3 \
             task=2 pipe=71 icb=99"
    );
}

/// A vertex reflection that trips the shader-pull coverage gate: writes
/// Position, reads VertexIndex, binds a Buffer at each of `bindings`.
#[cfg(feature = "backend-vulkan")]
fn shader_pull_reflection(bindings: &[u32]) -> metal2vulkan::reflect::ShaderReflection {
    use metal2vulkan::reflect::{
        BufferExtent, DescriptorLocation, ResourceBinding, ResourceKind, ShaderReflection,
        ShaderStage, VertexBuiltins, REFLECTION_VERSION,
    };
    ShaderReflection {
        reflection_version: REFLECTION_VERSION,
        stage: ShaderStage::Vertex,
        entry_point: None,
        bindings: bindings
            .iter()
            .map(|&binding| ResourceBinding {
                kind: ResourceKind::Buffer,
                metal_index: binding,
                descriptor: Some(DescriptorLocation {
                    set: 0,
                    binding,
                    count: 1,
                }),
                param_index: None,
                stage_input_location: None,
                address_space: None,
                declared_size: None,
                // What the translator emits for a buffer carrying neither an
                // object size nor a type name: the class that forbids narrowing.
                extent: Some(BufferExtent::Unknown),
                footprint: None,
                type_layout: None,
                type_name: None,
                texture_shape: None,
                embedded_source: None,
                access: None,
                static_sampler: None,
            })
            .collect(),
        argument_buffer_fields: vec![],
        vertex_attributes: vec![],
        varyings: vec![],
        render_targets: vec![],
        depth_members: vec![],
        depth_qualifier: None,
        stencil_members: vec![],
        local_size: None,
        vertex_builtins: Some(VertexBuiltins {
            uses_vertex_index: true,
            uses_instance_index: false,
            writes_position: true,
        }),
        tessellation: None,
        imageblock_layouts: vec![],
        implicit_imageblock_attachments: vec![],
        fragment_imageblock: None,
        datalayout: None,
        function_constants: vec![],
    }
}

/// Regression: WebKit's glyph vertex shader declares stride-48 stage-in on
/// buffer 1 but never reads it as an attribute — it indexes the same buffer
/// as a per-glyph `StorageBuffer` (binding 1) by gl_InstanceIndex. Skipping
/// every stage-in buffer left binding 1 unbound, so glyphs collapsed to
/// zero-area quads (blank Safari body text). A stage-in buffer whose binding
/// the vertex SPIR-V declares as a StorageBuffer must still be bound.
#[cfg(feature = "backend-vulkan")]
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
/// normalizing: a type-11 composite Store's readback now arrives BGRA, so this is
/// the call that used to be a 776 us whole-frame pass and is now a compare. A
/// future edit that made it exchange unconditionally would restore that cost
/// silently — the pixels would still be right.
#[cfg(feature = "backend-vulkan")]
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
#[cfg(feature = "backend-vulkan")]
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
/// (never type-11), and its extent is color0's declared geometry — the one
/// place a draw states what it renders into.
#[cfg(feature = "backend-vulkan")]
#[test]
fn gva_chain_identity_rules() {
    use crate::backend::vulkan::engine::TargetIdentity;
    let mut req = DrawEncodeRequest::default();
    assert_eq!(gva_chain_identity(&req), None, "no colors → no identity");
    req.colors.push(ColorRtRequest {
        slot: 0,
        texture_ref: 9,
        mapping_id: 0,
        target_gva: 0x1234_0000,
        width: 16,
        height: 16,
        ..Default::default()
    });
    assert_eq!(
        gva_chain_identity(&req),
        Some(TargetIdentity::Gva {
            gva: 0x1234_0000,
            width: 16,
            height: 16,
            generation: 0,
            format: crate::backend::vulkan::translate::pixel::RESIDENT_RGBA_FORMAT,
        }),
        "color0 declares the extent"
    );
    req.colors[0].width = 0;
    assert_eq!(
        gva_chain_identity(&req),
        None,
        "a zero-extent attachment has no identity"
    );
    req.colors[0].width = 16;
    req.colors[0].mapping_id = 5;
    assert_eq!(
        gva_chain_identity(&req),
        None,
        "type-11 targets never take the GVA identity"
    );
    req.colors[0].mapping_id = 0;
    req.colors[0].target_gva = 0;
    assert_eq!(gva_chain_identity(&req), None, "gva=0 → no identity");
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn render_chain_identity_covers_type11_and_gva_targets() {
    use crate::backend::vulkan::engine::TargetIdentity;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    assert!(state.map_surface(5));
    let mut req = DrawEncodeRequest::default();
    req.colors.push(ColorRtRequest {
        slot: 0,
        texture_ref: 9,
        mapping_id: 5,
        width: 64,
        height: 32,
        ..Default::default()
    });
    assert!(matches!(
        render_chain_identity(&state, &req),
        Some(TargetIdentity::Surface {
            id: 5,
            width: 64,
            height: 32,
            ..
        })
    ));

    req.colors[0].mapping_id = 0;
    req.colors[0].target_gva = 0x1234_0000;
    assert_eq!(
        render_chain_identity(&state, &req),
        Some(TargetIdentity::Gva {
            gva: 0x1234_0000,
            width: 64,
            height: 32,
            generation: 0,
            format: crate::backend::vulkan::translate::pixel::RESIDENT_RGBA_FORMAT,
        })
    );
}

/// The last record of a resident render-pass chain is both the chain's consumer
/// and the packet's guest-visible Store, and it must name the resident it loads
/// from so it can skip its own readback.
///
/// Refusing `chain_from_resident` here cost the entire remaining composite
/// readback population — `t11_keep_chain_from_resident` measured equal to
/// `surface_deferred` in every window of one boot. The assertion that matters is
/// the *equality*: `retarget_render_pass_draw` builds every record of a packet
/// from one attachment template, so the record that loads from the resident is by
/// construction the record that renders into it, and a Store naming a different
/// slot than its own LOAD would pin an image its frame is not in.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_chained_composite_store_names_the_resident_it_loads_from() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    assert!(state.map_surface(7));
    let mut req = DrawEncodeRequest::default();
    req.colors.push(ColorRtRequest {
        slot: 0,
        texture_ref: 3,
        mapping_id: 7,
        width: 128,
        height: 64,
        load_action: MTL_LOAD_ACTION_LOAD,
        store_action: MTL_STORE_ACTION_STORE,
        ..Default::default()
    });

    let unchained = type11_store_identity(&state, &req, true);
    assert!(
        unchained.is_some(),
        "an unchained composite Store resolves its resident"
    );

    req.chain_from_resident = true;
    assert_eq!(
        type11_store_identity(&state, &req, true),
        unchained,
        "a chained Store must name the same resident an unchained one does — it is \
         the same attachment template, so the chain cannot move the slot"
    );
    assert_eq!(
        type11_store_identity(&state, &req, true),
        render_chain_identity(&state, &req),
        "the Store identity and the LoadFromTarget identity must be one slot"
    );

    // The gates that are still refusals. `writeback_guest` is the one that
    // separates the packet's last record from its intermediates, and an
    // intermediate has no guest Store to defer.
    assert_eq!(
        type11_store_identity(&state, &req, false),
        None,
        "an intermediate record stores nothing guest-visible"
    );
    req.colors[0].store_action = crate::contract::pass_action::MTL_STORE_ACTION_DONT_CARE;
    assert_eq!(
        type11_store_identity(&state, &req, true),
        None,
        "a record that discards its target has no frame to defer"
    );
    req.colors[0].store_action = MTL_STORE_ACTION_STORE;
    req.colors[0].mapping_id = 0;
    assert_eq!(
        type11_store_identity(&state, &req, true),
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
#[cfg(feature = "backend-vulkan")]
#[test]
fn an_intermediate_record_can_still_ask_about_the_resident_it_renders_into() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    assert!(state.map_surface(7));
    let mut req = DrawEncodeRequest::default();
    req.colors.push(ColorRtRequest {
        slot: 0,
        texture_ref: 3,
        mapping_id: 7,
        width: 128,
        height: 64,
        load_action: MTL_LOAD_ACTION_LOAD,
        store_action: MTL_STORE_ACTION_STORE,
        ..Default::default()
    });

    // The query the LOAD actually asks. It takes no `writeback_guest`, so an
    // intermediate and a final record get the same answer by construction — which
    // is the property, and it is structural rather than asserted.
    let (identity, mapping_epoch) = type11_load_currency_query(&state, &req)
        .expect("a LOAD into a mapped type-11 surface is a candidate the resident could serve");
    assert_eq!(
        Some(identity.clone()),
        render_chain_identity(&state, &req),
        "the LOAD must ask about the slot the record actually renders into"
    );
    assert_eq!(
        Some(identity),
        type11_store_identity(&state, &req, true),
        "the Store identity is the same slot, restricted — not a different one"
    );
    assert_eq!(
        mapping_epoch,
        Some(0),
        "a freshly mapped surface has published nothing, and 0 is that value — the \
         `is_some` guard in `type11_resident_is_current` is what keeps it from \
         matching an unstamped slot"
    );
    assert_eq!(
        type11_store_identity(&state, &req, false),
        None,
        "…while only the packet's last record may leave its frame on the resident"
    );

    // The refusals. A LOAD the resident cannot serve must not produce a query at
    // all, or the counters below it would divide all draws instead of candidates.
    req.colors[0].load_action = crate::contract::pass_action::MTL_LOAD_ACTION_CLEAR;
    assert!(
        type11_load_currency_query(&state, &req).is_none(),
        "a CLEAR has no prior content to be current"
    );
    req.colors[0].load_action = MTL_LOAD_ACTION_LOAD;
    req.colors[0].target_seed_rgba = Some(vec![0u8; 128 * 64 * 4]);
    assert!(
        type11_load_currency_query(&state, &req).is_none(),
        "an explicit seed was already selected by RT provenance"
    );
    req.colors[0].target_seed_rgba = None;
    req.colors[0].store_action = crate::contract::pass_action::MTL_STORE_ACTION_DONT_CARE;
    assert!(
        type11_load_currency_query(&state, &req).is_none(),
        "a record that discards its target renders into no resident worth naming"
    );
    req.colors[0].store_action = MTL_STORE_ACTION_STORE;
    req.colors[0].mapping_id = 0;
    assert!(
        type11_load_currency_query(&state, &req).is_none(),
        "a GVA target is the other rail's"
    );
}

/// The guest half of the type-11 seed currency test.
///
/// `surface_content_epoch` witnesses only writers inside this crate — every
/// caller of `mark_mapping_written` is one — and a surface's pages are plain
/// guest RAM the guest CPU stores into with no device operation at all. Without
/// this half the elision answers "current" for a resident the guest has since
/// overwritten, the Store publishes that resident back over the guest's bytes,
/// and the epoch still has not moved: the fixpoint that turns one bad frame
/// into a permanently corrupted surface.
///
/// Every answer here is "written" except the one case that is provably not.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_guest_write_since_the_store_refuses_the_resident() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::runtime::host::{FakeHost, HostOps};

    let entry_for =
        |gpa: u64| (((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let page = state.page_size();
    assert!(state.map_surface(7));
    // Two pages of guest storage, stated as page-table entries the way a
    // resolved mapping carries them.
    let gpas = [0x40 * page, 0x91 * page];
    {
        let m = state.mappings.get_mut(&7).expect("mapped above");
        m.page_entries = gpas.iter().map(|g| entry_for(*g)).collect();
    }

    assert!(
        type11_guest_wrote_since_store(&state, &host, 7),
        "no Store has stamped this surface, so nothing vouches for any resident"
    );
    assert!(
        type11_guest_wrote_since_store(&state, &host, 999),
        "a mapping this device does not know cannot be declared unwritten"
    );

    // The Store's side: register the pages and record the generation.
    let token = crate::runtime::mapper::ensure_guest_write_token(&mut state, &mut host, 7)
        .expect("FakeHost observes guest writes");
    let stamped = host.guest_write_gen(token).expect("a live token has one");
    state
        .mappings
        .get_mut(&7)
        .expect("mapped above")
        .guest_write_gen_at_store = stamped;
    assert!(
        !type11_guest_wrote_since_store(&state, &host, 7),
        "nothing has written the pages since the Store, so the resident still holds them"
    );

    // A guest CPU store into one of the surface's pages. No device operation is
    // involved, which is the whole point: `surface_content_epoch` does not move.
    let epoch_before = state.mappings[&7].surface_content_epoch;
    host.guest_wrote_page(gpas[1]);
    assert_eq!(
        state.mappings[&7].surface_content_epoch, epoch_before,
        "the device-side epoch cannot see a guest CPU write — if it could, this \
         whole rail would be unnecessary"
    );
    assert!(
        type11_guest_wrote_since_store(&state, &host, 7),
        "the hypervisor saw the write, so the resident is stale"
    );

    // Idempotent: asking again must give the same answer. A consume-on-read
    // report would tell the first draw of the frame the surface is dirty and
    // every later draw that it is clean.
    assert!(
        type11_guest_wrote_since_store(&state, &host, 7),
        "the refusal must survive being read"
    );

    // Re-stamping after re-seeding restores reuse, so the rail recovers rather
    // than latching the other way.
    let restamped = host.guest_write_gen(token).expect("still live");
    state
        .mappings
        .get_mut(&7)
        .expect("mapped above")
        .guest_write_gen_at_store = restamped;
    assert!(!type11_guest_wrote_since_store(&state, &host, 7));

    // Retiring the page list retires the token with it: a generation recorded
    // against pages the surface no longer owns vouches for nothing.
    assert!(state.unmap_surface(7));
    assert!(
        type11_guest_wrote_since_store(&state, &host, 7),
        "a surface whose page list is gone has no vouched resident"
    );
    crate::runtime::mapper::flush_retired_views(&mut state, &mut host);
    assert_eq!(
        host.tracked_guest_write_sets(),
        0,
        "the token must reach the host, or every mapping incarnation leaks a set"
    );
}

/// The verdict must keep its refusals apart, because two rails now report it.
///
/// [`type11_guest_wrote_since_store`] collapses everything that is not `Clean`
/// to `true`, which is the right answer for a gate and the wrong one for a
/// census: "this rail was never stamped" and "the guest rewrites this surface
/// every frame" produce the same refusal and mean opposite things. The sampled
/// ladder's `t11rung_host_cache_gw_*` counters exist to tell them apart, so a
/// verdict that pooled them would make that reading a lie.
#[cfg(feature = "backend-vulkan")]
#[test]
fn the_guest_write_verdict_separates_its_refusals() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::runtime::host::{FakeHost, HostOps};

    let entry_for =
        |gpa: u64| (((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let page = state.page_size();
    assert!(state.map_surface(7));
    let gpas = [0x40 * page, 0x91 * page];
    {
        let m = state.mappings.get_mut(&7).expect("mapped above");
        m.page_entries = gpas.iter().map(|g| entry_for(*g)).collect();
    }

    assert_eq!(
        mapping_guest_write_verdict(&state, &host, 999),
        GuestWriteVerdict::NoMapping
    );
    assert_eq!(
        mapping_guest_write_verdict(&state, &host, 7),
        GuestWriteVerdict::NoStamp,
        "registered pages but no Store stamp is its own finding"
    );

    let token = crate::runtime::mapper::ensure_guest_write_token(&mut state, &mut host, 7)
        .expect("FakeHost observes guest writes");
    state
        .mappings
        .get_mut(&7)
        .expect("mapped above")
        .guest_write_gen_at_store = host.guest_write_gen(token).expect("a live token has one");
    assert_eq!(
        mapping_guest_write_verdict(&state, &host, 7),
        GuestWriteVerdict::Clean
    );

    host.guest_wrote_page(gpas[0]);
    assert_eq!(
        mapping_guest_write_verdict(&state, &host, 7),
        GuestWriteVerdict::Wrote,
        "a real guest write must not be reported as a missing stamp"
    );

    // A host that cannot answer for the token is neither clean nor a write.
    let blind = FakeHost::new();
    assert_eq!(
        mapping_guest_write_verdict(&state, &blind, 7),
        GuestWriteVerdict::Unreadable,
        "an unknown token is an unreadable answer, not an observed write"
    );
}

/// A page list replaced in place must not leave the token vouching for it.
///
/// The lifecycle mutators retire the token eagerly, but they are not the only
/// writers of `page_entries`: the mapper's plan adoption and the type-4 page
/// refresh both replace the list and bump `map_generation` without going near
/// them, and both used to retire the contiguous view while leaving the token
/// behind. A token that outlives its list watches pages the surface no longer
/// owns — it would report nothing while the guest writes the new ones.
///
/// `map_generation` is the key rather than a third retirement call site,
/// because the next writer of the list would forget the call and not the bump.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_replaced_page_list_invalidates_the_token_it_was_built_for() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::runtime::host::{FakeHost, HostOps};

    let entry_for =
        |gpa: u64, shift: u32| (((gpa >> shift) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let page = state.page_size();
    assert!(state.map_surface(7));
    {
        let m = state.mappings.get_mut(&7).expect("mapped above");
        m.page_entries = vec![entry_for(0x40 * page, PAGE_SHIFT_X86)];
    }
    let token = crate::runtime::mapper::ensure_guest_write_token(&mut state, &mut host, 7)
        .expect("FakeHost observes guest writes");
    let stamped = host.guest_write_gen(token).expect("a live token has one");
    state
        .mappings
        .get_mut(&7)
        .expect("mapped above")
        .guest_write_gen_at_store = stamped;
    assert!(!type11_guest_wrote_since_store(&state, &host, 7));

    // Exactly what the mapper's adoption and the type-4 refresh do: swap the
    // list and bump the generation, without touching the token.
    {
        let m = state.mappings.get_mut(&7).expect("mapped above");
        m.page_entries = vec![entry_for(0x88 * page, PAGE_SHIFT_X86)];
        DeviceState::bump_map_generation(m);
    }
    assert_eq!(
        state.mappings[&7].guest_write_token, token,
        "the point of the test is that nothing retired it"
    );
    assert!(
        type11_guest_wrote_since_store(&state, &host, 7),
        "a token built for the old pages cannot vouch for the new ones"
    );

    // The next Store rebuilds it against the pages the surface now owns, and
    // hands the old registration back for release.
    let fresh = crate::runtime::mapper::ensure_guest_write_token(&mut state, &mut host, 7)
        .expect("the new list is trackable");
    assert_ne!(fresh, token, "a new list needs a new registration");
    crate::runtime::mapper::flush_retired_views(&mut state, &mut host);
    assert_eq!(
        host.tracked_guest_write_sets(),
        1,
        "the stale set must be released, not leaked alongside the new one"
    );

    // And it watches the new pages: a write to one of them moves the fresh
    // generation, and a write to the retired page does not.
    let restamped = host.guest_write_gen(fresh).expect("live");
    state
        .mappings
        .get_mut(&7)
        .expect("mapped above")
        .guest_write_gen_at_store = restamped;
    host.guest_wrote_page(0x40 * page);
    assert!(
        !type11_guest_wrote_since_store(&state, &host, 7),
        "the surface no longer owns that page"
    );
    host.guest_wrote_page(0x88 * page);
    assert!(type11_guest_wrote_since_store(&state, &host, 7));
}

/// A host with no dirty bitmap must lose the elision, not gain a wrong frame.
///
/// The conservative default is the whole reason `guest_write_gen` answers an
/// `Option`: a host that cannot observe guest writes has to be told apart from
/// one that observed none, and only the second may license a reuse.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_host_that_cannot_observe_guest_writes_never_vouches() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::runtime::host::FakeHost;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.guest_writes_unobservable = true;
    let page = state.page_size();
    assert!(state.map_surface(7));
    {
        let m = state.mappings.get_mut(&7).expect("mapped above");
        m.page_entries = vec![
            ((((0x40 * page) >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
        ];
    }
    assert_eq!(
        crate::runtime::mapper::ensure_guest_write_token(&mut state, &mut host, 7),
        None,
        "the refusal belongs at registration, not at every read"
    );
    assert!(
        type11_guest_wrote_since_store(&state, &host, 7),
        "with no witness the pages must read as written"
    );
}

/// Records 2+ of an armed resident chain bind the attachment alias from
/// the resident target; unarmed LOAD without a seed stays None (guest
/// reload would expose pre-pass bytes — existing contract).
#[cfg(feature = "backend-vulkan")]
#[test]
fn attachment_alias_resident_chain_selection() {
    let mut req = DrawEncodeRequest::default();
    req.colors.push(ColorRtRequest {
        slot: 0,
        texture_ref: 42,
        mapping_id: 0,
        target_gva: 0x9000,
        width: 8,
        height: 8,
        load_action: MTL_LOAD_ACTION_LOAD,
        ..Default::default()
    });
    assert_eq!(
        fragment_attachment_alias_sample(&req, 0, 42),
        None,
        "unarmed LOAD without seed must not alias"
    );
    req.chain_from_resident = true;
    assert_eq!(
        fragment_attachment_alias_sample(&req, 0, 42),
        Some((8, 8, AttachmentAliasSample::ResidentChain)),
        "armed chain aliases from the resident target"
    );
    // CPU seed still wins when present (record after a non-resident hop).
    let seed = vec![0u8; 8 * 8 * 4];
    req.colors[0].target_seed_rgba = Some(seed);
    assert!(matches!(
        fragment_attachment_alias_sample(&req, 0, 42),
        Some((8, 8, AttachmentAliasSample::Seed(_, _)))
    ));
}

/// Vulkan-arm only: `AttachmentAliasSample` and its resolver are
/// `backend-vulkan` items.
#[cfg(feature = "backend-vulkan")]
#[test]
fn gva_attachment_alias_samples_the_in_process_chain() {
    let task_id = std::process::id();
    let texture_ref = 0xe000_0000u32.wrapping_add(task_id);
    let target_gva = 0x0abc_d000;
    let seed = vec![10, 0, 0, 255, 0, 0, 0, 0];
    let mut req = DrawEncodeRequest {
        task_id,
        colors: vec![ColorRtRequest {
            slot: 0,
            texture_ref,
            mapping_id: 0,
            target_gva,
            width: 2,
            height: 1,
            load_action: MTL_LOAD_ACTION_LOAD,
            target_seed_rgba: Some(seed.clone()),
            ..Default::default()
        }],
        ..Default::default()
    };

    let (width, height, sample) =
        fragment_attachment_alias_sample(&req, 0, texture_ref).expect("GVA alias");
    assert_eq!((width, height), (2, 1));
    let AttachmentAliasSample::Seed(actual, _) = sample else {
        panic!("Load alias must use the chained seed");
    };
    assert_eq!(actual, seed);
    assert!(fragment_attachment_alias_sample(&req, 1, texture_ref).is_none());
    assert!(fragment_attachment_alias_sample(&req, 0, texture_ref + 1).is_none());

    req.colors[0].mapping_id = 9;
    assert!(fragment_attachment_alias_sample(&req, 0, texture_ref).is_none());
    req.colors[0].mapping_id = 0;
    req.colors[0].load_action = MTL_LOAD_ACTION_DONT_CARE;
    assert!(fragment_attachment_alias_sample(&req, 0, texture_ref).is_none());
    req.colors[0].load_action = MTL_LOAD_ACTION_CLEAR;
    req.colors[0].clear_color = [0.25, 0.5, 0.75, 1.0];
    assert_eq!(
        fragment_attachment_alias_sample(&req, 0, texture_ref),
        Some((2, 1, AttachmentAliasSample::Clear([0.25, 0.5, 0.75, 1.0])))
    );
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
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    // A mapping this device holds no entry for has declared nothing, and the
    // store rule's own "nothing declared" answer is what comes back.
    assert_eq!(
        mapping_declared_format(&state, 7, None),
        MTL_FORMAT_BGRA8_UNORM
    );
    // One that declares a format outside every table this crate carries still
    // answers, and answers "not sRGB".
    state.mappings.insert(
        7,
        crate::model::MappingEntry {
            mapped: true,
            has_geom: true,
            width: 4,
            height: 4,
            format: 0xfffe,
            ..Default::default()
        },
    );
    assert_eq!(mapping_declared_format(&state, 7, None), 0xfffe);
    assert!(!pixel_format::is_srgb(mapping_declared_format(&state, 7, None)));
    // A declared sRGB surface reaches the bind as sRGB.
    state.mappings.get_mut(&7).expect("just inserted").format =
        pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB;
    assert!(pixel_format::is_srgb(mapping_declared_format(&state, 7, None)));
    // A type-8 view's format is what the guest says it is reading, so it wins
    // over the mapping's own — including when it takes the sRGB back off.
    assert_eq!(
        mapping_declared_format(&state, 7, Some(MTL_FORMAT_BGRA8_UNORM)),
        MTL_FORMAT_BGRA8_UNORM
    );
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn the_cpu_upload_rails_carry_the_srgb_transfer_function_to_the_bind() {
    use crate::backend::vulkan::translate::pixel::{vk_sampled_bytes, vk_texel_layout};

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
        ash::vk::Format::R8G8B8A8_SRGB,
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
        ash::vk::Format::B8G8R8A8_SRGB
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

#[cfg(feature = "backend-vulkan")]
#[test]
fn color_target_diag_names_every_mrt_slot() {
    let colors = vec![
        ColorRtRequest {
            slot: 0,
            texture_ref: 11,
            mapping_id: 1,
            width: 1920,
            height: 1080,
            format: MTL_FORMAT_BGRA8_UNORM,
            load_action: MTL_LOAD_ACTION_LOAD,
            store_action: MTL_STORE_ACTION_STORE,
            ..Default::default()
        },
        ColorRtRequest {
            slot: 2,
            texture_ref: 17,
            target_gva: 0x1234_5000,
            width: 960,
            height: 540,
            format: pixel_format::MTL_FORMAT_RGBA16_FLOAT,
            load_action: MTL_LOAD_ACTION_CLEAR,
            store_action: MTL_STORE_ACTION_STORE,
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
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.define_task(1, 0x1000, 2);
    let mut host = FakeHost::new();
    let req = DrawEncodeRequest {
        task_id: 1,
        pipeline_ref: 99,
        vertex_count: 3,
        instance_count: 1,
        primitive_type: 3, // triangle
        first_vertex: 0,
        colors: vec![ColorRtRequest {
            slot: 0,
            texture_ref: 1,
            mapping_id: 3,
            width: 4,
            height: 4,
            format: MTL_FORMAT_BGRA8_UNORM,
            store_action: MTL_STORE_ACTION_STORE,
            ..Default::default()
        }],
        ..Default::default()
    };
    let mut req = req;
    // Spelled here rather than behind a wrapper, because the two arms' wrappers
    // used to pass *different* `force_full_store` values for the same call.
    // These are the arguments `exec` passes for a single-record draw that owns
    // its writeback.
    let st = encode_draw_chain(&mut state, &mut host, &mut req, true, false).0;
    assert!(matches!(
        st,
        EncodeStatus::MissingPipeline(_) | EncodeStatus::MissingMtlb(_) | EncodeStatus::NoMetal(_)
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
    use crate::runtime::spirv_bind::{
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
                stage,
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

/// The direct-Metal draw path refuses a past-table bind instead of encoding
/// without it.
///
/// Metal-only, because the refusal is inside `encode_draw_chain`'s Metal body.
/// The Vulkan arm asks the same question at the head of `try_metal2vulkan_draw`,
/// covered by `a_bind_past_its_table_refuses_the_draw_before_anything_resolves`
/// in that module — and that one does run on a Linux host, so the shared check
/// is exercised even though this case cannot be.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
#[test]
fn the_metal_draw_arm_refuses_a_bind_past_its_table() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let mut req = DrawEncodeRequest {
        colors: vec![ColorRtRequest {
            width: 64,
            height: 64,
            ..Default::default()
        }],
        vertex_textures: vec![TextureBind {
            index: MAX_TEXTURE_BIND_SLOTS,
            texture_ref: 9,
            ..Default::default()
        }]
        .into(),
        ..Default::default()
    };
    let st = encode_draw_chain(&mut state, &mut host, &mut req, false, false).0;
    assert!(
        matches!(st, EncodeStatus::BadArgs("draw_mtl_bind_slot_past_table")),
        "expected a past-table refusal, got {st:?}"
    );
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn vulkan_sampler_missing_entry_returns_exact_decline() {
    let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
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

#[cfg(feature = "backend-vulkan")]
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

#[cfg(feature = "backend-vulkan")]
#[test]
fn vulkan_sampler_preserves_guest_coordinate_and_filter_state() {
    use crate::backend::vulkan::engine::{
        SamplerAddressMode, SamplerBorderColor, SamplerCompareFunction, SamplerFilter,
        SamplerMipFilter,
    };
    use crate::runtime::decode::resource::SamplerDescriptor;

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
#[cfg(feature = "backend-vulkan")]
#[test]
fn blend_state_maps_src_alpha_one_minus() {
    let b = translate::blend::state(
        &crate::runtime::decode::resource::PipelineColorAttachment {
            src_rgb: 4,   // SrcAlpha
            dst_rgb: 5,   // OneMinusSrcAlpha
            op_rgb: 0,    // Add
            src_alpha: 1, // One
            dst_alpha: 5, // OneMinusSrcAlpha
            op_alpha: 0,  // Add
            ..Default::default()
        },
        [0.0; 4],
    )
    .expect("map");
    assert_eq!(
        b.src_color,
        crate::backend::vulkan::engine::BlendFactor::SrcAlpha
    );
    assert_eq!(
        b.dst_color,
        crate::backend::vulkan::engine::BlendFactor::OneMinusSrcAlpha
    );
    assert_eq!(b.color_op, crate::backend::vulkan::engine::BlendOp::Add);
    assert_eq!(
        b.src_alpha,
        crate::backend::vulkan::engine::BlendFactor::One
    );
    assert!(translate::blend::factor(99).is_err());
    assert!(translate::blend::operation(9).is_err());
}

/// qemu-shim: guest Load with unresolvable type-11 pages still encodes
/// (archive NULL seed / Metal Clear invent) — does not drop the pass.
#[test]
fn mrt_draw_request_load_seed_miss_still_encodes() {
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::render::ColorAttachment;
    use crate::runtime::host::FakeHost;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.define_task(1, 0x1000, 2);
    // Type-11 registered with geom but empty page table → seed read fails.
    assert!(state.map_surface(9));
    assert!(state.set_mapping_geom(9, 8, 8, MTL_FORMAT_BGRA8_UNORM));
    // gen must be non-zero for Load path to attempt a snapshot (archive).
    state.mappings.get_mut(&9).unwrap().content_generation = 1;
    state.texture_to_mapping.insert((1, 42), 9);
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
    let req = req.expect("Load seed miss must still encode (archive NULL seed)");
    assert!(
        req.colors[0].target_seed_rgba.is_none(),
        "seed miss leaves seed None (Metal Clear invent, full Store)"
    );
    assert_eq!(req.colors[0].load_action, MTL_LOAD_ACTION_LOAD);
}

#[test]
fn mrt_draw_request_keeps_multisample_source_and_resolve_destination_distinct() {
    use crate::contract::pass_action::MTL_STORE_ACTION_MULTISAMPLE_RESOLVE;
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.define_task(1, 0x1000, 2);
    for (texture_ref, mapping_id) in [(42, 9), (43, 10)] {
        assert!(state.map_surface(mapping_id));
        assert!(state.set_mapping_geom(mapping_id, 64, 64, MTL_FORMAT_BGRA8_UNORM));
        state
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
        MTL_STORE_ACTION_MULTISAMPLE_RESOLVE
    );
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn mrt_draw_request_gets_attachment_samples_from_the_bound_pipeline_before_encode() {
    use crate::contract::pass_action::MTL_STORE_ACTION_MULTISAMPLE_RESOLVE;
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::resource::RenderPipelineDescriptor;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.define_task(1, 0x1000, 2);
    for (texture_ref, mapping_id) in [(42, 9), (43, 10)] {
        assert!(state.map_surface(mapping_id));
        assert!(state.set_mapping_geom(mapping_id, 64, 64, MTL_FORMAT_BGRA8_UNORM));
        state
            .texture_to_mapping
            .insert((1, texture_ref), mapping_id);
    }
    state.task_render_pipeline_states.register(
        1,
        7,
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
    assert_eq!(req.colors[0].texture_ref, 43, "the published resolve target");
    assert_eq!(
        req.colors[0].multisample_source_ref, 42,
        "the multisample source retains its own identity"
    );
}

#[test]
fn mrt_draw_request_refuses_a_resolve_destination_with_different_geometry() {
    use crate::contract::pass_action::MTL_STORE_ACTION_MULTISAMPLE_RESOLVE;
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.define_task(1, 0x1000, 2);
    for (texture_ref, mapping_id, width) in [(42, 9, 64), (43, 10, 32)] {
        assert!(state.map_surface(mapping_id));
        assert!(state.set_mapping_geom(mapping_id, width, 64, MTL_FORMAT_BGRA8_UNORM));
        state
            .texture_to_mapping
            .insert((1, texture_ref), mapping_id);
    }
    let mut att = clear_black_attachment(42);
    att.resolve_texture_ref = 43;
    att.store_action = MTL_STORE_ACTION_MULTISAMPLE_RESOLVE;
    let cap = crate::observe::FailCapture::start();

    assert!(single_rt_draw_request(&mut state, &mut host, 7, att).is_none());
    assert!(cap.lines().iter().any(|line| {
        line.contains("render_resolve_target_mismatch")
            && line.contains("source_geom=64x64")
            && line.contains("resolve_geom=32x64")
    }));
}

/// qemu-shim: type-8 view of type-11 is a valid color RT (archive
/// resource_resolve_texture view chain). Without this, App Store UI pipes
/// that bind a view as color attachment drop the entire MRT pass.
#[test]
fn mrt_draw_request_type8_view_of_type11_as_color_rt() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE_VIEW,
        TEXTURE_VIEW_DESC_BASE_REF, TEXTURE_VIEW_DESC_LEN, TEXTURE_VIEW_DESC_LEVEL_BASE,
        TEXTURE_VIEW_DESC_LEVEL_COUNT, TEXTURE_VIEW_DESC_OPCODE, TEXTURE_VIEW_DESC_PIXEL_FORMAT,
        TEXTURE_VIEW_DESC_SLICE_BASE, TEXTURE_VIEW_DESC_SLICE_COUNT, TEXTURE_VIEW_DESC_TEXTURE_REF,
        TEXTURE_VIEW_DESC_TEXTURE_TYPE, TEXTURE_VIEW_MIN_RANGED, TEXTURE_VIEW_MTL_TYPE_2D,
        TEXTURE_VIEW_OPCODE_RANGED,
    };
    use crate::runtime::host::FakeHost;

    // One-level page table: GVA pages 0..7 → data PFNs (blit_exec pattern).

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    // Object list at GVA page 0; count covers live residual slot 211.
    assert!(state.set_object_list(1, 0, 256));

    // Base type-11 mid 9 latched as texture ref 3.
    assert!(state.map_surface(9));
    assert!(state.set_mapping_geom(9, 64, 64, MTL_FORMAT_BGRA8_UNORM));
    state.mappings.get_mut(&9).unwrap().content_generation = 1;
    state.texture_to_mapping.insert((1, 3), 9);

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
        .expect("type-8 view of type-11 must resolve as color RT");
    assert_eq!(req.colors[0].mapping_id, 9);
    assert_eq!(req.colors[0].width, 64);
    assert_eq!(req.colors[0].height, 64);
    assert_eq!(req.colors[0].texture_ref, view_ref);
}

/// Archive `REIMS_VGPU_RESOURCE_RESOLVE_MAX_VIEW_CHAIN`: nested type-8 → type-8 →
/// type-11 must collapse to the non-view base. One-hop resolve left the mid
/// base as type-8 and dropped the MRT pass (`view_base_or_swizzle`).
#[test]
fn mrt_draw_request_nested_type8_view_chain_to_type11() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE_VIEW,
        TEXTURE_VIEW_DESC_BASE_REF, TEXTURE_VIEW_DESC_LEN, TEXTURE_VIEW_DESC_LEVEL_BASE,
        TEXTURE_VIEW_DESC_LEVEL_COUNT, TEXTURE_VIEW_DESC_OPCODE, TEXTURE_VIEW_DESC_PIXEL_FORMAT,
        TEXTURE_VIEW_DESC_SLICE_BASE, TEXTURE_VIEW_DESC_SLICE_COUNT, TEXTURE_VIEW_DESC_TEXTURE_REF,
        TEXTURE_VIEW_DESC_TEXTURE_TYPE, TEXTURE_VIEW_MIN_RANGED, TEXTURE_VIEW_MTL_TYPE_2D,
        TEXTURE_VIEW_OPCODE_RANGED,
    };
    use crate::runtime::host::FakeHost;

    fn write_type8_view(
        host: &mut FakeHost,
        state: &DeviceState,
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

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 256));

    // type-11 mid 9 as texture ref 3.
    assert!(state.map_surface(9));
    assert!(state.set_mapping_geom(9, 64, 64, MTL_FORMAT_BGRA8_UNORM));
    state.mappings.get_mut(&9).unwrap().content_generation = 1;
    state.texture_to_mapping.insert((1, 3), 9);

    // Inner view 8 → base 3 (type-11); outer view 211 → base 8 (type-8).
    write_type8_view(&mut host, &state, 8, 3, 0x280);
    write_type8_view(&mut host, &state, 211, 8, 0x300);

    let att = clear_black_attachment(211);
    let req = single_rt_draw_request(&mut state, &mut host, 12, att)
        .expect("nested type-8→type-8→type-11 must resolve as color RT");
    assert_eq!(req.colors[0].mapping_id, 9);
    assert_eq!(req.colors[0].width, 64);
    assert_eq!(req.colors[0].height, 64);
    assert_eq!(req.colors[0].texture_ref, 211);
}

/// Archive resolve_texture rejects non-identity swizzle for RT resolve.
#[test]
fn mrt_draw_request_type8_swizzled_view_rejected_as_color_rt() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE_VIEW,
        TEXTURE_VIEW_DESC_BASE_REF, TEXTURE_VIEW_DESC_LEN, TEXTURE_VIEW_DESC_LEVEL_BASE,
        TEXTURE_VIEW_DESC_LEVEL_COUNT, TEXTURE_VIEW_DESC_OPCODE, TEXTURE_VIEW_DESC_PIXEL_FORMAT,
        TEXTURE_VIEW_DESC_SLICE_BASE, TEXTURE_VIEW_DESC_SLICE_COUNT, TEXTURE_VIEW_DESC_SWIZZLE,
        TEXTURE_VIEW_DESC_TEXTURE_REF, TEXTURE_VIEW_DESC_TEXTURE_TYPE, TEXTURE_VIEW_MIN_SWIZZLE,
        TEXTURE_VIEW_MTL_TYPE_2D, TEXTURE_VIEW_OPCODE_SWIZZLE,
    };
    use crate::runtime::host::FakeHost;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));
    assert!(state.map_surface(9));
    assert!(state.set_mapping_geom(9, 64, 64, MTL_FORMAT_BGRA8_UNORM));
    state.texture_to_mapping.insert((1, 3), 9);

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
        mrt_draw_request(
            &mut state,
            &mut host,
            1,
            12,
            &[(0u32, att)],
            &[],
            test_triangle()
        )
        .is_none(),
        "swizzled type-8 must not resolve as color RT"
    );
}

/// qemu-shim: type-2 linear RGBA16Float is a valid color RT. Stale
/// `texture_to_mapping` from a prior type-11 at the same ref must not
/// fail-closed (live residual ref=199 type=2 fmt=0x73).
#[test]
fn mrt_draw_request_type2_rgba16f_as_color_rt_despite_stale_t11_latch() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::pixel_format::{MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_RGBA16_FLOAT};
    use crate::runtime::decode::resource::{
        list_object_entry_offset, LINEAR_DESC_HANDLE, LINEAR_DESC_SIZE, OBJECT_LIST_ENTRY_LEN,
        OBJECT_TYPE_TEXTURE, RESOURCE_PAGE_SHIFT, TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_HEIGHT,
        TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE, TEXTURE_DESC_WIDTH,
    };
    use crate::runtime::host::FakeHost;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 16);
    assert!(state.set_object_list(1, 0, 256));

    // Stale type-11 latch at ref 199 (guest recycled the ref to type-2).
    assert!(state.map_surface(99));
    assert!(state.set_mapping_geom(99, 64, 64, MTL_FORMAT_BGRA8_UNORM));
    state.texture_to_mapping.insert((1, 199), 99);

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
        .expect("type-2 RGBA16F RT must resolve despite stale type-11 latch");
    assert_eq!(req.colors[0].mapping_id, 0);
    assert_eq!(req.colors[0].width, w);
    assert_eq!(req.colors[0].height, h);
    assert_eq!(req.colors[0].format, MTL_FORMAT_RGBA16_FLOAT);
    assert_eq!(
        req.colors[0].target_gva,
        (handle as u64) << RESOURCE_PAGE_SHIFT
    );
    // Stale latch must be dropped.
    assert!(!state.texture_to_mapping.contains_key(&(1, tex_ref)));
}

/// Live type-11 descriptor mapping_id wins over a stale texture_to_mapping
/// latch (dual-mid recycled-ref residual: full desktop Store must land on
/// the mid named by the live descriptor, not a prior latch).
#[test]
fn mrt_draw_request_type11_live_mapping_overrides_stale_latch() {
    use crate::contract::endian::{st16, st32};
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::host::FakeHost;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    // Live type-11 at ref=1 → mapping_id=4 (descriptor first u32).
    let mut entry = [0u8; 12];
    st32(&mut entry[0..], 11u32 | (0x20u32 << 8));
    entry[4..12].copy_from_slice(&0x40u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 12, &entry);
    let mut desc = [0u8; 0x20];
    st32(&mut desc[0..], 4); // live mapping_id = 4
    st16(&mut desc[0x16..], MTL_FORMAT_BGRA8_UNORM);
    st32(&mut desc[0x18..], 64);
    st32(&mut desc[0x1c..], 32);
    let _ = host.write_gpa(data_gpa + 0x40, &desc);

    // Both mids exist; stale latch points ref 1 at mid 3.
    assert!(state.map_surface(3));
    assert!(state.set_mapping_geom(3, 64, 32, MTL_FORMAT_BGRA8_UNORM));
    assert!(state.map_surface(4));
    assert!(state.set_mapping_geom(4, 64, 32, MTL_FORMAT_BGRA8_UNORM));
    state.texture_to_mapping.insert((1, 1), 3);

    let att = clear_black_attachment(1);
    let req = single_rt_draw_request(&mut state, &mut host, 12, att)
        .expect("live type-11 RT must resolve");
    assert_eq!(
        req.colors[0].mapping_id, 4,
        "live descriptor mapping_id=4 must beat stale latch mid=3"
    );
    assert_eq!(state.texture_to_mapping.get(&(1, 1)).copied(), Some(4));
}

/// Color RT materialization does not rematerialize non-zero view mips.
#[test]
fn mrt_draw_request_type8_nonzero_level_rejected_as_color_rt() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE_VIEW,
        TEXTURE_VIEW_DESC_BASE_REF, TEXTURE_VIEW_DESC_LEN, TEXTURE_VIEW_DESC_LEVEL_BASE,
        TEXTURE_VIEW_DESC_LEVEL_COUNT, TEXTURE_VIEW_DESC_OPCODE, TEXTURE_VIEW_DESC_PIXEL_FORMAT,
        TEXTURE_VIEW_DESC_SLICE_BASE, TEXTURE_VIEW_DESC_SLICE_COUNT, TEXTURE_VIEW_DESC_TEXTURE_REF,
        TEXTURE_VIEW_DESC_TEXTURE_TYPE, TEXTURE_VIEW_MIN_RANGED, TEXTURE_VIEW_MTL_TYPE_2D,
        TEXTURE_VIEW_OPCODE_RANGED,
    };
    use crate::runtime::host::FakeHost;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));
    assert!(state.map_surface(9));
    assert!(state.set_mapping_geom(9, 64, 64, MTL_FORMAT_BGRA8_UNORM));
    state.texture_to_mapping.insert((1, 3), 9);

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
        mrt_draw_request(
            &mut state,
            &mut host,
            1,
            12,
            &[(0u32, att)],
            &[],
            test_triangle()
        )
        .is_none(),
        "type-8 level_base!=0 must not resolve as color RT"
    );
}

/// Archive collapses a type-8 view's mip level into linear geometry:
/// a level-1 view of a type-2 texture is a color RT at that level's
/// plane (offset/dims/stride from the descriptor's level record) —
/// compositor blur/backdrop pyramids render into successive mips.
#[test]
fn mrt_draw_request_type8_mip_level_view_of_linear_as_color_rt() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE,
        OBJECT_TYPE_TEXTURE_VIEW, TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_LEVEL_RECORDS,
        TEXTURE_DESC_MIPMAP_LEVEL_COUNT, TEXTURE_DESC_MIP_LEVEL_RECORD_LEN,
        TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE, TEXTURE_DESC_USED_SIZE,
        TEXTURE_DESC_WIDTH, TEXTURE_LEVEL_HEIGHT, TEXTURE_LEVEL_OFFSET, TEXTURE_LEVEL_ROW_STRIDE,
        TEXTURE_LEVEL_SIZE, TEXTURE_LEVEL_WIDTH, TEXTURE_VIEW_DESC_BASE_REF, TEXTURE_VIEW_DESC_LEN,
        TEXTURE_VIEW_DESC_LEVEL_BASE, TEXTURE_VIEW_DESC_LEVEL_COUNT, TEXTURE_VIEW_DESC_OPCODE,
        TEXTURE_VIEW_DESC_PIXEL_FORMAT, TEXTURE_VIEW_DESC_SLICE_BASE,
        TEXTURE_VIEW_DESC_SLICE_COUNT, TEXTURE_VIEW_DESC_TEXTURE_REF,
        TEXTURE_VIEW_DESC_TEXTURE_TYPE, TEXTURE_VIEW_MIN_RANGED, TEXTURE_VIEW_MTL_TYPE_2D,
        TEXTURE_VIEW_OPCODE_RANGED,
    };
    use crate::runtime::host::FakeHost;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    // Type-2 base with 2 mips: L0 64x32 bpr 256; L1 at +0x2000, 32x16 bpr 128.
    let base_ref = 5u32;
    let body = TEXTURE_DESC_BASE_LEN + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
    let mut b = vec![0u8; body];
    st64(&mut b[0..], 0x20000); // allocation_size
    st32(&mut b[8..], 0x20); // handle
    st16(&mut b[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 2);
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
    assert_eq!(c0.mapping_id, 0);
    assert_eq!(
        c0.target_gva,
        ((0x20u64) << PAGE_SHIFT_ARM64E) + 0x2000,
        "RT gva = allocation base + level-1 offset"
    );
    assert_eq!((c0.width, c0.height), (32, 16), "level-1 dims");
    assert_eq!(c0.row_stride, 128, "level-1 row stride");
}

#[test]
fn view_swizzle_remaps_rgba8_pixels() {
    // Every CPU remap must report itself: this is the path the Vulkan
    // pathway replaced with a component mapping, and an unreported
    // invocation is a texture that silently lost its zero-copy crossing.
    crate::runtime::census::view_swizzle_census::reset_for_tests();
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
    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
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
    use crate::contract::endian::st32;
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::contract::pixel_format::{solid_rgba8, MTL_FORMAT_BGRA8_UNORM};
    use crate::runtime::host::FakeHost;

    // Two identical guests, so the two writers land into two byte-for-byte
    // equal address spaces and the comparison is of the writes alone.
    fn guest(page_shift: u32) -> (FakeHost, DeviceState) {
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
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    use crate::contract::endian::st32;
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::host::FakeHost;

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

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
#[cfg(feature = "backend-vulkan")]
#[test]
fn gva_layer_host_cache_roundtrip_for_sample() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
        resolve_sampled_source(&mut state, &mut host, 0, tex_ref, None, true).is_none(),
        "a ref with no object-list entry must resolve to no sampled source"
    );
}
/// Guest-CPU-produced tight linear textures: unchanged native bytes must
/// reuse the memoized RGBA Arc under a stable generation identity; a guest
/// write must be observed and produce a new generation.
#[cfg(feature = "backend-vulkan")]
#[test]
fn guest_linear_memo_reuses_arc_and_observes_guest_writes() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, TEXTURE_DESC_BASE_LEN,
        TEXTURE_DESC_MIPMAP_LEVEL_COUNT, TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE,
        TEXTURE_DESC_USED_SIZE, TEXTURE_DESC_WIDTH,
    };

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    st16(&mut b[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 1);
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
#[cfg(feature = "backend-vulkan")]
#[test]
fn padded_bgra8_memoized_uploads_native_without_swizzle() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, TEXTURE_DESC_BASE_LEN,
        TEXTURE_DESC_MIPMAP_LEVEL_COUNT, TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE,
        TEXTURE_DESC_USED_SIZE, TEXTURE_DESC_WIDTH,
    };

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    st16(&mut b[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 1);
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
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
/// `mrt_draw_request` sets [`DrawEncodeRequest::gva_load_from_resident`] when the
/// engine still holds what the render Store published into the target's guest
/// pages, and it pays for that by leaving `colors[0].target_seed_rgba` as `None`
/// while the attachment still says `MTL_LOAD_ACTION_LOAD`. The attachment is then
/// only as defined as the encode side makes it: `honour_gva_load_elision` either
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
/// The race it stands in for is real. `gva_alloc_gen` is recomputed after the
/// request is built, so a page set that moved in between names a different
/// target — `gva_chain_identity` then resolves to an identity the registry has no
/// resident for, exactly as it does here, and the seed is already gone.
///
/// There is no engine in a unit test, so `resident_content_ready` is false for
/// every identity and the chain arm cannot be taken. That is the point: this is
/// the seedless case by construction, and the only correct behaviour is to put
/// the seed back. The guest pages below are real, so `Some` here means
/// `seed_color_load` actually re-read the attachment rather than the arm merely
/// counting itself.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_gva_load_from_resident_draw_with_no_resident_puts_the_seed_back() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, TEXTURE_DESC_BASE_LEN,
        TEXTURE_DESC_MIPMAP_LEVEL_COUNT, TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE,
        TEXTURE_DESC_USED_SIZE, TEXTURE_DESC_WIDTH,
    };
    use crate::runtime::drain::store_route_count;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    st16(&mut b[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 1);
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
    let mut req = DrawEncodeRequest {
        task_id: 1,
        gva_load_from_resident: true,
        colors: vec![ColorRtRequest {
            slot: 0,
            texture_ref: tex_ref,
            mapping_id: 0,
            target_gva: texel_gva,
            row_stride: 16,
            width: 4,
            height: 2,
            format: MTL_FORMAT_BGRA8_UNORM,
            load_action: MTL_LOAD_ACTION_LOAD,
            store_action: MTL_STORE_ACTION_STORE,
            target_seed_rgba: None,
            ..Default::default()
        }],
        ..Default::default()
    };
    assert!(
        crate::runtime::draw::vulkan::gva_chain_identity(&req).is_some(),
        "the elided request must still name an identity — the arm under test is \
         the one where that identity has no ready resident, not the one where \
         there is nothing to look up"
    );

    let chained_before = store_route_count("gvaseed_chained");
    let reseeded_before = store_route_count("gvaseed_reseeded");
    let mut chain_load_from_target = false;
    let identity = crate::runtime::draw::vulkan::honour_gva_load_elision(
        &mut state,
        &mut host,
        &mut req,
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
        identity.is_none() && !chain_load_from_target,
        "a re-seeded pass loads from its attachment, not from a resident"
    );

    // The property, and the reason the counter alone is not enough: the seed is
    // back, and it holds the guest's pixels rather than an empty buffer.
    let seed = req.colors[0]
        .target_seed_rgba
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

/// A type-5 ref is not itself a surface id. The descriptor's surface_id
/// remains authoritative even when the numeric ref collides with another
/// live display mapping (live app-launch ref=2 -> sid=71 class).
#[cfg(feature = "backend-vulkan")]
#[test]
fn type5_sample_uses_descriptor_surface_id_not_ref_collision() {
    use crate::contract::endian::st32;
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
    use crate::runtime::gva_mem;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
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
    let built = reims_vgpu_wire::device_desc::Type5Builder::new(surface_id, 0, 0, 0)
        .with_len(objects::TYPE5_MIN_LEN);
    let desc = built.bytes();
    assert!(
        gva_mem::write_task_gva(&mut host, &state.tasks[1], desc_gva, desc, PAGE_SHIFT_X86,)
            .is_ok()
    );
    let list_off = list_object_entry_offset(texture_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((objects::TYPE5_MIN_LEN as u32) << 8);
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
    state.mappings.get_mut(&texture_ref).unwrap().page_entries = vec![1];
    crate::runtime::surface_cache::store(
        &mut state,
        texture_ref,
        8,
        8,
        [0u8, 0, 255, 255].repeat(8 * 8),
    );

    assert!(state.map_surface(surface_id));
    assert!(state.set_mapping_geom(surface_id, 4, 3, MTL_FORMAT_BGRA8_UNORM));
    state.mappings.get_mut(&surface_id).unwrap().page_entries = vec![1];
    crate::runtime::surface_cache::store(
        &mut state,
        surface_id,
        4,
        3,
        [255u8, 0, 0, 255].repeat(4 * 3),
    );

    let (width, height, sampled_mid, sampled) =
        resolve_sampled_source(&mut state, &mut host, 1, texture_ref, None, true)
            .expect("type-5 descriptor surface must sample");
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
        "type-5 fixture must expose a resource to thread"
    );
    let (tw, th, tmid, tsrc) =
        resolve_sampled_source(&mut state, &mut host, 1, texture_ref, threaded_resource, true)
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
#[cfg(feature = "backend-vulkan")]
#[test]
fn type11_host_cache_rung_identity_tracks_the_cached_frame() {
    use crate::contract::endian::st32;
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
    use crate::runtime::gva_mem;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
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
    let built = reims_vgpu_wire::device_desc::Type5Builder::new(surface_id, 0, 0, 0)
        .with_len(objects::TYPE5_MIN_LEN);
    let desc = built.bytes();
    assert!(
        gva_mem::write_task_gva(&mut host, &state.tasks[1], desc_gva, desc, PAGE_SHIFT_X86,)
            .is_ok()
    );
    let list_off = list_object_entry_offset(texture_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((objects::TYPE5_MIN_LEN as u32) << 8);
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
    state.mappings.get_mut(&surface_id).unwrap().page_entries = vec![1];

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

    let resolve = |state: &mut DeviceState, host: &mut FakeHost| {
        let (_, _, _, src) = resolve_sampled_source(state, host, 1, texture_ref, None, true)
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
        "bit 62 alone is the type-11 host-cache namespace"
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

/// Live Safari app-launch class: the type-4 base carries an unknown
/// 2-byte IOSurface FourCC (`LA08`) while the type-5 descriptor carries
/// the exact RG8 Metal view. Defaulting the base to BGRA asks for a
/// 632-byte row against the wire's 320-byte row and drops the draw.
#[cfg(feature = "backend-vulkan")]
#[test]
fn type5_sample_uses_serialized_rg8_view_over_unknown_surface_fourcc() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::contract::iosurface_pages::{
        DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_BPE, DEVICE_DESC_BPR, DEVICE_DESC_DIMS,
        DEVICE_DESC_LEN, DEVICE_DESC_PIXEL_FORMAT, PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID,
    };
    use crate::contract::pixel_format::MTL_FORMAT_RG8_UNORM;
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
    use crate::runtime::gva_mem;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    // The subject here is the serialized RG8 *byte* conversion, so the host is
    // one that hands back reconstructed views and every zero-copy rung declines.
    // Without this the surface's pages resolve as guest RAM and the resolver
    // correctly returns a resident instead, which proves nothing about RG8.
    host.transient_map_pages = true;

    // One-level x86 GVA table for the object list and type-5 descriptor.
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
    let built = reims_vgpu_wire::device_desc::Type5Builder::new(
        surface_id,
        0,
        texture_ref,
        reims_vgpu_wire::device_desc::TYPE5_RECORD_TAG_PLANE,
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
        let m = state.mappings.get_mut(&surface_id).unwrap();
        m.mapped = true;
        m.page_entries = (0..page_count)
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

    let (sample_w, sample_h, sample_mid, sampled) =
        resolve_sampled_source(&mut state, &mut host, 1, texture_ref, None, true)
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

/// The type-5 view memo: unchanged plane bytes reuse the converted Arc and
/// carry a stable content identity (engine upload skipped); a guest write
/// to the plane is observed on the next bind and mints a new generation.
#[cfg(feature = "backend-vulkan")]
#[test]
fn type5_view_memo_reuses_unchanged_planes_and_invalidates_on_write() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::iosurface_pages::{
        DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_BPE, DEVICE_DESC_BPR, DEVICE_DESC_DIMS,
        DEVICE_DESC_LEN, DEVICE_DESC_PIXEL_FORMAT, PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID,
    };
    use crate::contract::pixel_format::MTL_FORMAT_RG8_UNORM;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
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
        let m = state.mappings.get_mut(&surface_id).unwrap();
        m.mapped = true;
        m.page_entries = (0..page_count)
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
    let view = objects::Type5TextureView {
        pixel_format: MTL_FORMAT_RG8_UNORM,
        width,
        height,
        depth: 1,
        plane_index: 0,
    };

    let (w1, h1, rgba1, id1, fmt1) =
        load_type5_view_rgba(&mut state, &mut host, 1, 248, surface_id, view)
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
        "identity key namespaces type-5 content above GVA keys"
    );

    let (_, _, rgba2, id2, _) =
        load_type5_view_rgba(&mut state, &mut host, 1, 248, surface_id, view)
            .expect("memo revalidation");
    assert!(
        std::sync::Arc::ptr_eq(&rgba1, &rgba2),
        "unchanged plane bytes must reuse the native allocation"
    );
    assert_eq!(id1, id2, "unchanged content keeps its identity");

    // Guest CPU writes one texel; the next bind must observe it.
    assert!(host.write_gpa(gpa0 + 6, &[0xAA, 0xBB]).is_ok());
    let (_, _, rgba3, id3, _) =
        load_type5_view_rgba(&mut state, &mut host, 1, 248, surface_id, view)
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

#[cfg(feature = "backend-vulkan")]
#[test]
fn type5_view_materializes_only_when_base_identity_differs() {
    use crate::contract::pixel_format::MTL_FORMAT_RG8_UNORM;

    let exact = objects::Type5TextureView {
        pixel_format: MTL_FORMAT_BGRA8_UNORM,
        width: 1920,
        height: 1080,
        depth: 1,
        plane_index: 0,
    };
    assert!(!type5_view_requires_materialization(
        true,
        1920,
        1080,
        MTL_FORMAT_BGRA8_UNORM,
        exact
    ));
    assert!(type5_view_requires_materialization(
        true, 1920, 1080, 0, exact
    ));

    let rg8_view = objects::Type5TextureView {
        pixel_format: MTL_FORMAT_RG8_UNORM,
        width: 158,
        height: 154,
        depth: 1,
        plane_index: 0,
    };
    assert!(type5_view_requires_materialization(
        true,
        158,
        154,
        MTL_FORMAT_BGRA8_UNORM,
        rg8_view
    ));
    assert!(type5_view_requires_materialization(
        false,
        158,
        154,
        MTL_FORMAT_RG8_UNORM,
        rg8_view
    ));
    let volume = objects::Type5TextureView { depth: 2, ..exact };
    assert!(type5_view_requires_materialization(
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
            object_type: 2,
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
        TextureViewDecline::HopLevelOverflow {
            texture_ref: 1,
            level_base: u64::MAX,
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
    assert_eq!(slugs.len(), 9);
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

/// Every type-5 view refusal names its rail (`type5_view_`), renders
/// whitespace-free fields, and is distinct — the same discipline the
/// capture and import rails took, so `grep reason=type5_view_…` stays
/// answerable against the blit rail's `t5_*` copy vocabulary next door.
#[cfg(feature = "backend-vulkan")]
#[test]
fn every_type5_view_reason_is_namespaced_distinct_and_log_safe() {
    use crate::observe::Decline as _;
    const ALL: &[Type5ViewDecline] = &[
        Type5ViewDecline::UnsupportedDepth { depth: 0 },
        Type5ViewDecline::Unresolved,
        Type5ViewDecline::FormatBpp,
        Type5ViewDecline::NoMapping,
        Type5ViewDecline::SampleWindow {
            base_w: 0,
            base_h: 0,
            base_fmt: 0,
            desc: None,
        },
        Type5ViewDecline::Span {
            pages: 0,
            page_bytes: 0,
            span_end: 0,
            bpr: 0,
        },
        Type5ViewDecline::TightOverflow { bpp: 0 },
        Type5ViewDecline::NativeLen { tight: 0 },
        Type5ViewDecline::Read {
            base_w: 0,
            base_h: 0,
            base_fmt: 0,
            off: 0,
            bpr: 0,
            span_end: 0,
            pages: 0,
        },
        Type5ViewDecline::RgbaStride,
        Type5ViewDecline::RgbaLen { stride: 0 },
        Type5ViewDecline::Convert { row: 0, bpp: 0 },
    ];
    let mut slugs: Vec<&str> = Vec::new();
    for d in ALL {
        assert!(
            d.slug().starts_with("type5_view_"),
            "{} is not namespaced to the type-5 view rail",
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
    assert_eq!(before, slugs.len(), "duplicate Type5ViewDecline slug");
}

/// `SampleWindow` is the only variant carrying transcribed field logic: the
/// base geometry plus the decoded device descriptor, or `desc=missing` when
/// the descriptor could not be decoded. Both branches must render exactly
/// what the old ad-hoc `detail` string did.
#[cfg(feature = "backend-vulkan")]
#[test]
fn sample_window_renders_the_descriptor_or_its_absence() {
    let present = Type5ViewDecline::SampleWindow {
        base_w: 320,
        base_h: 240,
        base_fmt: 0x50,
        desc: Some((64, 64, 0x4c41_3038, 256, 4096)),
    };
    assert_eq!(
            crate::observe::Emit::decline("type5_draw_view", &present).render(),
            "type5_draw_view reason=type5_view_sample_window base=320x240 base_fmt=0x50 desc=64x64 desc_fmt=0x4c413038 bpr=256 alloc=4096"
        );

    let missing = Type5ViewDecline::SampleWindow {
        base_w: 320,
        base_h: 240,
        base_fmt: 0x50,
        desc: None,
    };
    assert_eq!(
        crate::observe::Emit::decline("type5_draw_view", &missing).render(),
        "type5_draw_view reason=type5_view_sample_window base=320x240 base_fmt=0x50 desc=missing"
    );
}

/// A secondary MRT attachment binds **its own** blend, not slot 0's and not
/// "unblended because secondaries are always masks".
///
/// The regression this locks: `caches.rs` forced every secondary attachment
/// `blend_enable(false)`, justified by a comment claiming the decode side
/// carried no per-attachment blend state. It carried it all along — the
/// Metal arm reads exactly these fields per slot — so a guest MRT pipeline
/// that asked to blend slot 1 got a raw store instead.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_secondary_mrt_slot_binds_its_own_blend() {
    use crate::runtime::decode::resource::{PipelineColorAttachment, RenderPipelineDescriptor};
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);

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
            target_gva: 0x1000,
            width: 64,
            height: 64,
            format: MTL_FORMAT_BGRA8_UNORM,
            ..ColorRtRequest::default()
        },
        ColorRtRequest {
            slot: 1,
            texture_ref: 11,
            target_gva: 0x2000,
            width: 64,
            height: 64,
            format: crate::contract::pixel_format::MTL_FORMAT_RG16_FLOAT,
            ..ColorRtRequest::default()
        },
    ];
    let primary = crate::backend::vulkan::engine::TargetIdentity::Gva {
        gva: 0x1000,
        width: 64,
        height: 64,
        generation: 0,
        format: crate::backend::vulkan::translate::pixel::RESIDENT_RGBA_FORMAT,
    };

    let mut host = crate::runtime::host::FakeHost::new();
    let secs = build_secondary_targets(
        &mut state, &mut host, 1, &colors, &pipeline, &primary, 64, 64, [0.0; 4],
    )
    .expect("a contiguous, geometry-matching, resolvable secondary builds");
    assert_eq!(secs.len(), 1, "one secondary attachment expected");
    let blend = secs[0].blend.expect(
        "slot 1 declares blending_enabled — before this fix every secondary \
             was forced unblended",
    );
    use crate::backend::vulkan::engine::{BlendFactor, BlendOp};
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
    let secs = build_secondary_targets(
        &mut state, &mut host, 1, &colors, &unblended, &primary, 64, 64, [0.0; 4],
    )
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
#[cfg(feature = "backend-vulkan")]
#[test]
fn an_unbuildable_secondary_refuses_the_draw_rather_than_dropping_to_single_rt() {
    use crate::runtime::census::present_proxy::MrtDrop;
    use crate::runtime::decode::resource::{PipelineColorAttachment, RenderPipelineDescriptor};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
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
    let primary = crate::backend::vulkan::engine::TargetIdentity::Gva {
        gva: 0x1000,
        width: 64,
        height: 64,
        generation: 0,
        format: crate::backend::vulkan::translate::pixel::RESIDENT_RGBA_FORMAT,
    };
    let slot0 = ColorRtRequest {
        slot: 0,
        texture_ref: 10,
        target_gva: 0x1000,
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
        target_gva: 0x2000,
        width: 64,
        height: 64,
        format: crate::contract::pixel_format::MTL_FORMAT_RG16_FLOAT,
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
            64,
            64,
            [0.0; 4],
        )
    };

    // The control: this exact list builds, so every refusal below is caused by
    // the one field it changes and not by the fixture.
    assert!(
        build(&good_slot1).is_ok(),
        "the unmodified fixture must build, or the cases below prove nothing"
    );

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
            MrtDrop::GeometryMismatch,
            ColorRtRequest {
                width: 32,
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
                target_gva: 0,
                mapping_id: 0,
                ..good_slot1.clone()
            },
        ),
        (
            // Resolves to the primary's own resident: a pass that reads and
            // writes one image through two attachments.
            MrtDrop::AliasesPrimary,
            ColorRtRequest {
                target_gva: 0x1000,
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
        64,
        64,
        [0.0; 4],
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
#[cfg(feature = "backend-vulkan")]
#[test]
fn two_secondaries_over_one_destination_refuse_the_draw_like_a_primary_alias() {
    use crate::contract::pixel_format::{MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_RG16_FLOAT};
    use crate::runtime::census::present_proxy::MrtDrop;
    use crate::runtime::decode::resource::{PipelineColorAttachment, RenderPipelineDescriptor};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let pipeline = RenderPipelineDescriptor {
        color_attachments: (0..3)
            .map(|slot| PipelineColorAttachment {
                slot,
                ..PipelineColorAttachment::default()
            })
            .collect(),
        ..RenderPipelineDescriptor::default()
    };
    let primary = crate::backend::vulkan::engine::TargetIdentity::Gva {
        gva: 0x1000,
        width: 64,
        height: 64,
        generation: 0,
        format: crate::backend::vulkan::translate::pixel::RESIDENT_RGBA_FORMAT,
    };
    let rt = |slot: u32, texture_ref: u32, target_gva: u64, format: u16| ColorRtRequest {
        slot,
        texture_ref,
        target_gva,
        width: 64,
        height: 64,
        format,
        ..ColorRtRequest::default()
    };
    let mut build = |colors: &[ColorRtRequest]| {
        let mut host = crate::runtime::host::FakeHost::new();
        build_secondary_targets(
            &mut state, &mut host, 1, colors, &pipeline, &primary, 64, 64, [0.0; 4],
        )
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

#[cfg(feature = "backend-vulkan")]
#[test]
fn fixed_state_gap_names_every_unrepresented_field() {
    // In-contract cull (Back) + winding (CCW) are HONORED by the Vulkan
    // raster state; the depth test + attachment and the stencil test
    // (op state + reference + clear) are honored via `resources.depth` — so
    // cull/front/depth_stencil/depth_attach/stencil_ref/stencil_attach are no
    // longer gaps. Only depth bias remains unrepresented.
    // MTLCullModeBack / MTLWindingCounterClockwise, per `translate::raster`,
    // which is now the only place those SDK values are spelled.
    let mut req = DrawEncodeRequest {
        cull_mode: Some(2),
        front_facing: Some(1),
        depth_bias: Some([1.25, 2.5, 0.0]),
        depth_stencil_ref: 77,
        stencil_ref: Some((3, 4)),
        depth_attach: Some(DepthAttachment::default()),
        stencil_attach: Some(StencilAttachment::default()),
        ..DrawEncodeRequest::default()
    };
    assert_eq!(vulkan_fixed_state_gap(&req), "bias:1.250/2.500/0.000");

    // An OUT-OF-CONTRACT cull/winding value stays a gap (fail-visible), never
    // coerced to a face that silently draws or drops geometry.
    req.depth_bias = None;
    req.depth_stencil_ref = 0;
    req.stencil_ref = None;
    req.depth_attach = None;
    req.stencil_attach = None;
    req.cull_mode = Some(9);
    req.front_facing = Some(7);
    assert_eq!(vulkan_fixed_state_gap(&req), "cull:9,front:7");

    req.cull_mode = None;
    req.front_facing = None;
    assert_eq!(vulkan_fixed_state_gap(&req), "");
}

/// Recycled texture_ref must not serve a prior full-frame encode as a
/// different-sized linear sample (namespace / geom-match class).
#[test]
fn texture_ref_cache_geom_mismatch_does_not_hit_get_texture() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let tex_ref = 53u32;
    let mut full = vec![0u8; 1920 * 1152 * 4];
    full[3] = 255;
    host_cache_store_rgba8(&mut state, 0, tex_ref, 1920, 1152, &full);
    // Exact geom hit
    assert!(
        crate::runtime::surface_cache::get_texture(&state, 0, tex_ref, 1920, 1152).is_some()
    );
    // Wrong geom (type-3 L0 recycle) miss
    assert!(crate::runtime::surface_cache::get_texture(&state, 0, tex_ref, 115, 16).is_none());
    // surface_id map must stay empty for texture_ref stores
    assert!(crate::runtime::surface_cache::get(&state, tex_ref, 1920, 1152).is_none());
}

/// A guest-run host pointer is only valid when the host said the view it just
/// handed back is guest RAM itself. A view the host owes a release for can be
/// released before the submission gathers from it, so the runtime must decline
/// to produce runs at all rather than hand the engine that pointer.
///
/// The two arms differ **only** in `transient_map_pages`, so the stable arm is
/// the control: without it this would pass on a build that declined every span
/// for some unrelated reason, which is exactly how a decline test goes hollow.
///
/// The stable arm is also the regression this test now carries. It asks nothing
/// of the device and everything of the call, so a span whose pages coalesce to
/// one GPA-contiguous stretch resolves on **any** host — which is what every
/// draw-time buffer bind on arm64 is, and what a device-wide stability flag
/// refused 19 885 times in one boot.
#[test]
#[cfg(feature = "backend-vulkan")]
fn guest_runs_decline_on_transient_host_mappings() {
    use crate::contract::endian::st32;
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    let page_shift = crate::model::PAGE_SHIFT_X86;
    let page = 1u64 << page_shift;
    let gva = 8u64;

    // A task whose page table really does resolve `[gva, gva+16)` onto `data0`.
    let walkable = |transient: bool| -> Result<
        Vec<crate::backend::vulkan::engine::GuestRun>,
        super::vulkan::WindowRefusal,
    > {
        let mut host = FakeHost::new();
        host.strict_linux_map = true;
        host.transient_map_pages = transient;
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

        let mut state = DeviceState::new(DeviceId(1), page_shift);
        state.define_task(1, page, 2);
        task_gva_guest_run_window(&state, &mut host, 1, gva, 16).map(|(_, runs)| runs)
    };

    assert!(
        walkable(false).is_ok_and(|runs| !runs.is_empty()),
        "control: a contiguous stretch is guest RAM and resolves, with nothing \
         asked of the device"
    );
    // Named, not merely absent: this refusal is counted, and it must land in the
    // route that says the view was one the host owes a release for rather than
    // in one of the two that report a page table the guest wrote.
    assert_eq!(
        walkable(true).err(),
        Some(super::vulkan::WindowRefusal::NoAlias),
        "the same span must yield no runs when the host built the view rather \
         than naming guest RAM, since the gather outlives its release"
    );
}

/// A host whose guest-RAM map is standing on a refusal may not import through
/// the packed-alias rail either.
///
/// The rail builds its own host allocation rather than consuming a RAMBlock, so
/// it used to ask only the three import *latches* — granularity, span max,
/// budget — which are published once at device create from what the host
/// supports and stay `Some` for the whole boot. The map's refusal is a later and
/// different judgement, about what this guest fits into that host, and
/// `standing_refusal`'s own doc says a caller must ask it rather than re-read a
/// latch that "is the same answer for one of the four refusals and silence for
/// the other three".
///
/// Silence is the defect. `ImportExceedsHeap`'s doc says refusing the whole map
/// rather than the oversized block is what makes it safe, because "all or
/// nothing keeps the boot on the one arm that is tested end to end" — and this
/// rail kept importing on exactly the host that had been taken off that arm.
#[test]
#[cfg(feature = "backend-vulkan")]
fn the_packed_alias_rail_refuses_on_a_host_whose_map_refused() {
    let page_shift = PAGE_SHIFT_X86;
    let page = 1u64 << page_shift;
    // A host that reports no guest RAM at all: the map resolves to a standing
    // refusal even though every latch below is set.
    let mut host = FakeHost::new();
    crate::runtime::guest_ram_map::reset();
    crate::runtime::guest_ram::latch_import_limits(page, 1 << 30, 1 << 30);
    assert!(
        crate::runtime::guest_ram_map::standing_refusal(&mut host).is_some(),
        "the premise is a map that refused"
    );
    assert_eq!(
        crate::runtime::guest_ram_map::packed_alias_import_align(&mut host, page),
        None,
        "the latches all answer, and the rail must still refuse"
    );
    // And the latches really do all answer, which is why asking them alone was
    // silent about this host.
    assert!(crate::runtime::guest_ram::granularity().is_some());
    assert!(crate::runtime::guest_ram::import_span_max().is_some());
    assert!(crate::runtime::guest_ram::import_budget().is_some());
    crate::runtime::guest_ram_map::reset();
}

/// One whole-buffer host allocation is made per task/reference and offset binds
/// become slices of it. The second lookup must not ask the host to reconstruct
/// the same virtual range again.
#[test]
#[cfg(feature = "backend-vulkan")]
fn packed_buffer_alias_is_reused_across_offsets() {
    // The guest-RAM map resolves once per process and every test in this binary
    // shares it, so a fixture that maps its own RAM has to discard whatever an
    // earlier test resolved. Without this the alias rail asks a map built from
    // some other test's host and refuses — which is the right refusal against
    // the wrong host.
    crate::runtime::guest_ram_map::reset();
    use crate::contract::endian::st32;
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    let page_shift = PAGE_SHIFT_X86;
    let page = 1u64 << page_shift;
    let mut host = FakeHost::new();
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
    let mut state = DeviceState::new(DeviceId(1), page_shift);
    state.define_task(1, page, 2);
    let backing = super::BufferBacking {
        gva: 0x800,
        size: 0x4800,
    };
    assert!(super::vulkan::ensure_packed_resource(
        &mut state,
        &mut host,
        1,
        7,
        &backing,
        super::vulkan::PackedResourceRail::Buffer,
    ));
    let calls = host.map_pages_calls;
    let owners_before = {
        let packed = state
            .bound_buffers
            .packed_available(1, 7, backing.gva, backing.size)
            .expect("the fully mapped allocation must pack");
        (
            std::sync::Arc::strong_count(&packed.import),
            std::sync::Arc::strong_count(&packed.gpas),
            std::sync::Arc::strong_count(&packed.runs),
            std::sync::Arc::strong_count(&packed.pages),
        )
    };
    assert!(super::vulkan::ensure_packed_resource(
        &mut state,
        &mut host,
        1,
        7,
        &backing,
        super::vulkan::PackedResourceRail::Buffer,
    ));
    crate::runtime::guest_ram::forget_import_limits();

    assert_eq!(
        host.map_pages_calls, calls,
        "the second offset reuses the alias"
    );
    let packed = state
        .bound_buffers
        .packed_available(1, 7, backing.gva, backing.size)
        .expect("the retained allocation remains available");
    assert_eq!(
        owners_before,
        (
            std::sync::Arc::strong_count(&packed.import),
            std::sync::Arc::strong_count(&packed.gpas),
            std::sync::Arc::strong_count(&packed.runs),
            std::sync::Arc::strong_count(&packed.pages),
        ),
        "ensuring a warm resource borrows every retained payload"
    );
    let a = super::vulkan::slice_packed_buffer(packed, 0, 0x800).unwrap();
    let b = super::vulkan::slice_packed_buffer(packed, 0x1000, 0x800).unwrap();
    assert_eq!(
        a.pages.as_ref().unwrap()[0].guest.import().id(),
        packed.import.id()
    );
    assert_eq!(
        b.pages.as_ref().unwrap()[0].guest.import().id(),
        packed.import.id()
    );
    assert_eq!(a.source_offset, 0);
    assert_eq!(b.source_offset, 0x1000);
    assert!(
        std::sync::Arc::ptr_eq(&a.runs, &b.runs),
        "offset binds share the resource's one run list"
    );
    assert!(
        std::sync::Arc::ptr_eq(a.pages.as_ref().unwrap(), b.pages.as_ref().unwrap()),
        "offset binds share the resource's one bounded import reference"
    );
    drop(a);
    drop(b);
    let import = std::sync::Arc::clone(&packed.import);
    let gpas = std::sync::Arc::clone(&packed.gpas);
    let runs = std::sync::Arc::clone(&packed.runs);
    let pages = std::sync::Arc::clone(&packed.pages);
    let import_owners = std::sync::Arc::strong_count(&import);
    let page_list_owners = std::sync::Arc::strong_count(&gpas);
    let run_owners = std::sync::Arc::strong_count(&runs);
    let guest_ref_owners = std::sync::Arc::strong_count(&pages);

    let content = super::vulkan::load_buffer_content(
        &mut state,
        &mut host,
        1,
        7,
        None,
        0x800,
        true,
        Some(0x800),
    )
    .expect("the retained resource serves a later offset directly");
    let crate::backend::vulkan::engine::BufferContent::GuestRuns(source) = content else {
        panic!("the packed resource stays on the guest-memory rail")
    };
    assert_eq!(source.source_offset, 0x800);
    assert_eq!(source.total_len, 0x800);
    assert_eq!(
        std::sync::Arc::strong_count(&import),
        import_owners,
        "a warm bind borrows the retained resource instead of cloning its allocation"
    );
    assert_eq!(
        std::sync::Arc::strong_count(&gpas),
        page_list_owners,
        "the physical construction list does not travel to execution"
    );
    assert_eq!(
        std::sync::Arc::strong_count(&runs),
        run_owners + 1,
        "execution owns exactly the run source it consumes"
    );
    assert_eq!(
        std::sync::Arc::strong_count(&pages),
        guest_ref_owners + 1,
        "execution owns exactly the bounded guest references it consumes"
    );
    assert_eq!(
        state.bound_buffers.len(),
        0,
        "a packed resource does not grow the per-offset resolution registry"
    );
}

#[test]
#[cfg(feature = "backend-vulkan")]
fn sampled_plane_keeps_the_packed_allocation_and_checks_its_extent() {
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
        runs: std::sync::Arc::new(Vec::new()),
        pages: std::sync::Arc::new(Vec::new()),
    };
    let resource = crate::model::TaskResource::new(Default::default(), std::sync::Arc::from([]));
    let sampled = super::vulkan::sampled_backing_from_packed(
        &packed,
        0x1000,
        512,
        0x2000,
        resource.lifetime_ref(),
    )
    .expect("the plane lies inside the packed allocation");
    assert_eq!(sampled.backing.allocation_host_ptr, import.host_base());
    assert_eq!(sampled.backing.allocation_len, import.len());
    assert_eq!(sampled.backing.plane_offset, 0x1800);
    assert_eq!(sampled.backing.row_pitch, 512);
    assert!(std::sync::Arc::ptr_eq(&sampled.import, &import));
    assert!(
        super::vulkan::sampled_backing_from_packed(
            &packed,
            0x7000,
            512,
            0x1001,
            resource.lifetime_ref(),
        )
        .is_none(),
        "a plane may not extend the imported allocation to fit"
    );
    drop(sampled);

    let import_owners = std::sync::Arc::strong_count(&packed.import);
    let page_list_owners = std::sync::Arc::strong_count(&packed.gpas);
    let run_owners = std::sync::Arc::strong_count(&packed.runs);
    let guest_ref_owners = std::sync::Arc::strong_count(&packed.pages);
    let request = super::vulkan::direct_linear_sample_from_packed(
        &packed,
        0x1000,
        512,
        0x2000,
        128,
        crate::contract::pixel_format::TexelLayout::Rgba8,
        ash::vk::Format::R8G8B8A8_UNORM,
        crate::contract::pixel_format::SwizzlePlan::default(),
        resource.lifetime_ref(),
    )
    .expect("the retained allocation directly supplies this plane");
    let super::vulkan::SampledSourceRequest::GuestRuns(source, _, _, 1, None, _, _) = request else {
        panic!("a direct single plane has no copied-content identity")
    };
    assert_eq!(
        std::sync::Arc::strong_count(&packed.import),
        import_owners + 1,
        "execution retains the allocation its direct image consumes"
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
    assert!(source.direct_image.is_some());
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
#[cfg(feature = "backend-vulkan")]
fn a_window_refusal_names_which_check_refused() {
    use super::vulkan::WindowRefusal;
    use crate::contract::endian::st32;
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    let page_shift = crate::model::PAGE_SHIFT_X86;
    let page = 1u64 << page_shift;

    // GVA page 0 maps to `leaf_pfn`; GVA page 1 is left unmapped. `back_leaf`
    // decides whether the host has RAM behind the frame the PTE names, which is
    // what separates a walk that fails from an import that does.
    let build = |leaf_pfn: u32, back_leaf: bool| {
        let mut host = FakeHost::new();
        host.strict_linux_map = true;
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
        let mut state = DeviceState::new(DeviceId(1), page_shift);
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
    state: DeviceState,
    root_gpa: u64,
}

impl StoreRig {
    /// Task 1, `entries` mapped virtual pages, `page_shift` geometry.
    fn new(entries: u32) -> Self {
        use crate::contract::endian::st32;
        use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        let mut host = FakeHost::new();
        let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
        use crate::contract::endian::st32;
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

#[cfg(feature = "backend-vulkan")]
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
        mapping_id: 0,
        target_gva: page,
        row_stride: 64 * 4,
        width: 64,
        height: 64,
        format: crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM,
        sample_count: 1,
        load_action: MTL_LOAD_ACTION_LOAD,
        store_action: MTL_STORE_ACTION_STORE,
        clear_color: [0.0; 4],
        target_seed_rgba: None,
        multisample_source_ref: 0,
    };

    let armed = sync_store_allowed_pages(&state, &host, 1, Some(&c0), true)
        .expect("a resolvable GVA target must be bounded");
    let mut resolve = c0.clone();
    resolve.store_action =
        crate::contract::pass_action::MTL_STORE_ACTION_MULTISAMPLE_RESOLVE;
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
    crate::contract::endian::st32(&mut pte, pt_base + 6);
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
        target_gva: 0,
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
#[cfg(feature = "backend-vulkan")]
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

    let mut req = DrawEncodeRequest {
        task_id: 1,
        // No pipeline, so the engine draw is never attempted at all — the
        // cheapest way to the tail, and the arm whose refusal the emitter has
        // to name itself because there is no engine slug to borrow.
        pipeline_ref: 0,
        vertex_count: VERTEX_COUNT,
        instance_count: 1,
        primitive_type: 3,
        first_vertex: 0,
        colors: vec![ColorRtRequest {
            slot: 0,
            texture_ref: 7,
            // 64x64 BGRA8 at a tight stride is exactly one 16 KiB page of the
            // rig's walkable task, so the CLEAR seed Store lands and the tail's
            // `any_store` precondition is met.
            target_gva: StoreRig::gva(1),
            row_stride: 64 * 4,
            width: 64,
            height: 64,
            format: crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            load_action: MTL_LOAD_ACTION_CLEAR,
            store_action: MTL_STORE_ACTION_STORE,
            ..Default::default()
        }],
        ..Default::default()
    };

    let draws_before = store_route_count(DRAWS);
    let vertices_before = store_route_count(VERTICES);

    let cap = crate::observe::FailCapture::start();
    let st = encode_draw_chain(&mut state, &mut host, &mut req, true, false).0;
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

/// A scissored Store carries the same bound as a full-image one, on both of its
/// rails.
///
/// `write_gva_rgba8_rect` is what the Metal encode path calls when a Load seed
/// means only the scissor rect may be overwritten. It sits behind the same GPU
/// round trip as the full-image Store -- `render_core_mrt` submits, waits and
/// reads back before the writeback loop resolves `target_gva` -- so an
/// unbounded rect write scatters exactly the same way, over a 64-row stripe
/// instead of a whole frame.
///
/// Both rails are exercised because the writer picks between them on span
/// contiguity, and only one of them existed before the bound: a packed span maps
/// once through `map_fresh_span_within`, a fragmented one writes each row
/// through `write_span_within`. A bound on one and not the other is no bound at
/// all, since the guest chooses which by how it laid the allocation out.
#[test]
fn a_scissored_gva_store_is_bounded_on_both_its_rails() {
    // 64x128 BGRA8, tight 256-byte rows: exactly two 16 KiB pages, so page 2
    // holds rows 64..127 and any rect tall enough to reach them must be
    // authorised for it.
    const W: u32 = 64;
    const H: u32 = 128;
    const BPR: u32 = W * 4;
    let fmt = crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    let target = ColorRtRequest {
        slot: 0,
        texture_ref: 7,
        mapping_id: 0,
        target_gva: StoreRig::gva(1),
        row_stride: BPR,
        width: W,
        height: H,
        format: fmt,
        sample_count: 1,
        load_action: MTL_LOAD_ACTION_LOAD,
        store_action: MTL_STORE_ACTION_STORE,
        clear_color: [0.0; 4],
        target_seed_rgba: None,
        multisample_source_ref: 0,
    };
    // Full height, left half only: the partial store the Load seed forces, and
    // it crosses the page boundary at row 64.
    let rect = (0u32, 0u32, W / 2, H);
    let rgba = vec![0xffu8; (W * H * 4) as usize];
    let moved_to = STORE_RIG_PT_BASE + 6;

    // `packed` picks whether the two pages of the span are adjacent guest
    // frames, which is the only thing that decides the writer's rail.
    for packed in [true, false] {
        let mut rig = StoreRig::new(8);
        if !packed {
            rig.point(2, STORE_RIG_PT_BASE + 5);
        }
        let armed = sync_store_target_pages(&rig.state, &rig.host, 1, &target)
            .expect("a resolvable GVA target must be bounded");
        assert_eq!(
            armed.membership().len(),
            2,
            "64x128 BGRA8 tight covers two 16 KiB pages"
        );

        // The guest re-points the span's second page mid-flight.
        rig.point(2, moved_to);
        let (x, y, rw, rh) = rect;
        assert!(
            !write_gva_rgba8_rect(
                &mut rig.state,
                &mut rig.host,
                1,
                target.target_gva,
                W,
                H,
                BPR,
                fmt,
                &rgba,
                mapping_write::Rect {
                    origin_x: x,
                    origin_y: y,
                    width: rw,
                    height: rh
                },
                Some(armed.membership())
            ),
            "packed={packed}: a page the command never named must be refused"
        );
        let mut victim = [0u8; 4];
        let _ = rig
            .host
            .read_gpa(StoreRig::frame_gpa(moved_to), &mut victim);
        assert_eq!(
            victim, [0u8; 4],
            "packed={packed}: the refused write must leave the new owner untouched"
        );

        // Unbounded, the same call lands in whatever owns that page now. This
        // is the write the crash reports are of: 0xff is the white pixel run
        // the guest kernel reported over its freed heap element.
        assert!(
            write_gva_rgba8_rect(
                &mut rig.state,
                &mut rig.host,
                1,
                target.target_gva,
                W,
                H,
                BPR,
                fmt,
                &rgba,
                mapping_write::Rect {
                    origin_x: x,
                    origin_y: y,
                    width: rw,
                    height: rh
                },
                None
            ),
            "packed={packed}: without the bound the same write succeeds"
        );
        let _ = rig
            .host
            .read_gpa(StoreRig::frame_gpa(moved_to), &mut victim);
        assert_eq!(
            victim, [0xffu8; 4],
            "packed={packed}: and it succeeds by painting the new owner's page"
        );
    }
}

/// A type-11 sample must resolve the mapping *before* it reads its geometry.
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
fn type11_sample_resolves_geometry_before_reading_it() {
    use crate::contract::endian::{st32, st64};
    use crate::contract::iosurface_pages::{
        DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_BPR, DEVICE_DESC_DIMS, DEVICE_DESC_LEN,
        DEVICE_DESC_PIXEL_FORMAT, DEVICE_DESC_PLANE_COUNT, MAPPING_INTERNAL_BACKPTR,
        MAPPING_INTERNAL_DESC_PTR, MAPPING_INTERNAL_EXPECTED_SIZE, MAPPING_INTERNAL_ID,
        MAPPING_INTERNAL_PAGE_COUNT, MAPPING_INTERNAL_PAGE_FIELD_48,
        MAPPING_INTERNAL_PAGE_FIELD_50, MAPPING_INTERNAL_SIZE, MAPPING_PAGE_TABLE_FROM_F48,
        PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID,
    };
    use crate::runtime::host::HostMemory;

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

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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

    state.mapper_device_kva = mapper;
    assert!(state.attach_mapping_internal(mid, internal));
    {
        let m = state.mappings.get(&mid).expect("mapping");
        assert!(m.mapped && m.mapping_internal != 0);
        assert!(
            !m.has_geom,
            "the fixture's whole point is that no geometry is latched yet"
        );
    }

    let (gw, gh, rgba) = load_type11_mapping_rgba(&mut state, &mut host, mid, None).expect(
        "a mapped type-11 surface whose dims are still only in the device descriptor \
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
        state.mappings.get(&mid).expect("mapping").has_geom,
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

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
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
/// gate the only two options are a flood or the silence the Metal arm had. The
/// key has to separate slugs as well as pipelines, or one degradation would
/// mask a different one on the same pipeline — which is the case the Metal arm
/// hits, since its depth and stencil substitutions can both fire on one pass.
#[cfg(any(
    feature = "backend-vulkan",
    all(feature = "backend-metal", target_os = "macos")
))]
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

/// A sampled bind naming an object that is not a texture is reported, and the
/// answer it gets does not change.
///
/// The two sampled-source sites in `draw::vulkan` resolved a `texture_ref`
/// without the second ladder rung, so a ref naming any object with a long
/// enough descriptor decoded as a texture and produced a plausible extent —
/// while `mipmap`, `draw::texture_view`, `compute_exec` and
/// `draw::render_target` all ask the type for the same field. Turning the rung
/// into a refusal here would lose a draw on the pathway this device is verified
/// on, so it reports; this pins both halves of that, because a probe that
/// quietly started refusing would be the bug it was added to measure.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_sampled_ref_naming_another_object_kind_is_reported_and_still_resolves() {
    use crate::contract::endian::{st32, st64};
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE, TEXTURE_DESC_BASE_LEN,
    };
    use crate::runtime::gva_mem::write_task_gva_arm64e;
    use crate::runtime::objects::OBJECT_TYPE_REF_TEXTURE;

    // One task, one object list, and two refs into it — same descriptor bytes,
    // different object type in the entry.
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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

    crate::observe::redirect_logs_for_tests();
    let before = std::fs::read_to_string(crate::observe::fail_log_path())
        .unwrap_or_default()
        .len();

    // A real texture resolves and says nothing.
    assert!(
        super::vulkan::sampled_texture_descriptor(&state, &host, 1, texture_ref).is_some(),
        "a texture ref resolves"
    );
    // Another kind resolves exactly the same way — the bytes are the bytes —
    // and the rung is named.
    let (entry, bytes) = super::vulkan::sampled_texture_descriptor(&state, &host, 1, view_ref)
        .expect("the answer does not change: the descriptor still resolves");
    assert_eq!(entry.object_type, OBJECT_TYPE_REF_TEXTURE);
    assert_eq!(bytes.len(), body);

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).unwrap_or_default();
    let tail = &log[before.min(log.len())..];
    let line = tail
        .lines()
        .find(|l| l.contains("draw_sampled_texture_wrong_type"))
        .expect("the second ladder rung must name itself");
    assert!(
        line.contains(&format!("ref={view_ref}")) && line.contains("object_type=5"),
        "the line must name the ref and what it found: {line}"
    );
    assert!(
        !tail
            .lines()
            .any(|l| l.contains("draw_sampled_texture_wrong_type") && l.contains("ref=6")),
        "a texture ref must not be reported"
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
        BufferSpanRefusal::Rung(LadderRung::WrongType { got: 2 }),
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
    assert_eq!(super::buffer_refusal_detail(all[1], 12), "ty=2");
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
        "a stride neither backend can carry must not be truncated into one they can"
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
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_gva_span_no_store_has_stamped_refuses_the_resident_sample_rung() {
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::resource::{TextureDescriptor, TextureLevelLayout};
    use crate::runtime::drain::store_route_count;
    use crate::runtime::gva_store_witness::{note_store, GvaTargetKey};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
        data_offset: 0,
        bytes_per_element: 4,
        used_size: width * height * 4,
        row_stride: row_stride as u32,
        width,
        height,
        depth: 1,
        pixel_format: MTL_FORMAT_BGRA8_UNORM,
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

    assert!(
        super::try_gva_resident_sample(&mut state, &mut host, 1, 7, &tex).is_none(),
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
        gva,
        generation: 0xdead_beef,
        width,
        height,
        // What the descriptor declares, so the orphan differs from the live key
        // in the generation alone.
        bgra: true,
    };
    note_store(&mut state, &mut host, orphan, &gpas);
    assert!(
        super::try_gva_resident_sample(&mut state, &mut host, 1, 7, &tex).is_none(),
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
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_depth_attachment_is_keyed_on_the_guest_texture_the_pass_bound() {
    use crate::backend::vulkan::engine::TargetIdentity;
    use crate::runtime::decode::render::DepthAttachment;
    use crate::runtime::draw::vulkan::depth_chain_identity;

    let id = |r: &DrawEncodeRequest, st: bool| depth_chain_identity(r, st);
    let req = |depth_ref: u32, w: u32, h: u32| DrawEncodeRequest {
        colors: vec![ColorRtRequest {
            width: w,
            height: h,
            ..Default::default()
        }],
        depth_attach: Some(DepthAttachment {
            texture_ref: depth_ref,
            ..Default::default()
        }),
        ..DrawEncodeRequest::default()
    };

    let first = id(&req(42, 1024, 768), false).expect("a bound depth texture has one");
    assert_eq!(
        first,
        TargetIdentity::Texture {
            ref_: 42,
            width: 1024,
            height: 768,
            generation: 0,
            stencil: false,
        },
        "the guest's own texture ref is the key"
    );
    assert_eq!(
        id(&req(42, 1024, 768), false).as_ref(),
        Some(&first),
        "a second draw into the same depth texture resolves the same resident"
    );
    assert_ne!(
        id(&req(43, 1024, 768), false),
        Some(first.clone()),
        "two guest depth textures must not fuse into one resident"
    );
    assert_ne!(
        id(&req(42, 800, 600), false),
        Some(first.clone()),
        "geometry is part of the key, so a resized attachment recreates"
    );

    // The stencil aspect selects the image format, so it partitions the key. Two
    // residents, each stable, rather than one retired and recreated on every
    // alternation between a stencil draw and a depth-only one.
    assert_ne!(
        id(&req(42, 1024, 768), true),
        Some(first),
        "a stencil-carrying depth attachment is its own resident"
    );

    // No depth texture in the pass descriptor: nothing to key a resident on, and
    // the engine's transient fallback is what runs.
    assert_eq!(
        id(&req(0, 1024, 768), false),
        None,
        "an unbound depth attachment names no resident"
    );
    assert_eq!(
        depth_chain_identity(
            &DrawEncodeRequest {
                colors: vec![ColorRtRequest {
                    width: 1024,
                    height: 768,
                    ..Default::default()
                }],
                ..DrawEncodeRequest::default()
            },
            false
        ),
        None,
        "and neither does a pass with no depth attachment at all"
    );
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
#[cfg(feature = "backend-vulkan")]
fn only_a_statically_used_texture_gap_is_given_a_neutral_image() {
    use crate::runtime::spirv_bind::DescriptorUse;

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
        // Both other classes answer `Used` unconditionally from
        // `frag_unbound_static_use`, so they are exactly the case that would slip
        // through a filter written on the verdict alone.
        (gap(FragUnboundClass::Buffer, 7), DescriptorUse::Used),
        (gap(FragUnboundClass::Sampler, 8), DescriptorUse::Used),
    ];

    assert_eq!(frag_unbound_textures_to_neutralize(&uses), vec![3]);
    // Nothing flagged, nothing substituted — the hot path.
    assert!(frag_unbound_textures_to_neutralize(&[]).is_empty());
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
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_retained_depth_stencil_state_is_served_without_reading_guest_memory() {
    use crate::runtime::decode::resource::DepthStencilDescriptor;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
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
    state
        .task_depth_stencil_states
        .register(2, 7, std::sync::Arc::new(retained.clone()));

    assert_eq!(
        super::load_depth_stencil_descriptor(&state, &host, 2, 7).ok(),
        Some(retained),
        "the retained state answers with no guest read available at all"
    );

    assert!(state.task_depth_stencil_states.delete(2, 7));
    assert!(
        super::load_depth_stencil_descriptor(&state, &host, 2, 7).is_err(),
        "and the delete is a real invalidation, not a counter"
    );
}

/// A CPU read of a buffer's guest bytes pays whatever the buffer reference
/// names before it reads them.
///
/// This is the third of the three terms every raw task-GVA reader owes, and the
/// one `read_buffer_bytes_resolved` did not carry: it settled submitted writes
/// and never paid an owed frame, because it held `DeviceState` shared. A
/// writeback debt is a frame this device rendered and deliberately did *not*
/// submit, so the settle finds nothing outstanding and returns at once with the
/// frame still in a host resident — and the read serves the guest the frame
/// before it.
///
/// The observable is that the ledger was **asked**: the debt is taken. Whether
/// the payment then lands is `pay`'s business and needs an engine; that the
/// reader asked at all is this site's obligation and is what regressed.
#[test]
#[cfg(feature = "backend-vulkan")]
fn a_buffer_read_pays_the_frame_its_reference_owes_before_reading_the_pages() {
    use crate::runtime::writeback_debt::test_resident_identity;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let (task_id, buffer_ref) = (3u32, 21u32);
    state.define_task(task_id, 0x1000, 9);

    // The reference is itself a mapping id, which is the dominant spelling and
    // the one `mappings_named_by` resolves without a per-task registration.
    assert_eq!(
        state.pending_writebacks.arm(
            buffer_ref,
            test_resident_identity(buffer_ref, 64, 64, 1),
            64,
            64,
            1,
        ),
        None,
    );
    assert!(
        state.pending_writebacks.get(buffer_ref).is_some(),
        "the surface behind this reference owes a frame"
    );

    let backing = BufferBacking {
        gva: 0x8000,
        size: 0x1000,
    };
    // The read itself has no guest pages behind it and answers `None`. That is
    // deliberate: the payment is owed *before* the bytes are touched, so a
    // failing read must not be able to excuse a skipped one.
    let _ = read_buffer_bytes_resolved(&mut state, &mut host, task_id, buffer_ref, &backing, 0, None);

    assert!(
        state.pending_writebacks.get(buffer_ref).is_none(),
        "the reader read the guest's pages while a rendered frame was still owed to them"
    );
}

/// The buffer-backed texture rail pays for its *texture* reference too.
///
/// A type-8 buffer-backed texture is two contract references over one
/// allocation — the texture object the guest binds and samples, and the type-1
/// buffer that owns the storage — and a debt may be armed under either. The
/// three sibling sampled rails all pay for the texture reference; this one made
/// no payment at all, which is why a rendered icon or glyph atlas could stay in
/// a host resident while the bind served the frame before it.
#[test]
#[cfg(feature = "backend-vulkan")]
fn the_buffer_backed_texture_rail_pays_for_its_texture_reference() {
    use crate::runtime::writeback_debt::test_resident_identity;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let (task_id, texture_ref, buffer_ref) = (4u32, 31u32, 32u32);
    state.define_task(task_id, 0x1000, 9);

    assert_eq!(
        state.pending_writebacks.arm(
            texture_ref,
            test_resident_identity(texture_ref, 32, 32, 1),
            32,
            32,
            1,
        ),
        None,
    );

    let bt = BufferTextureDescriptor {
        new_texture_ref: texture_ref,
        buffer_ref,
        offset: 0,
        bytes_per_row: 0,
        desc: crate::runtime::heap_query::TextureDescriptor {
            texture_type: 2,
            framebuffer_only: false,
            is_drawable: false,
            allow_gpu_optimized_contents: false,
            usage: 0,
            pixel_format: crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            width: 32,
            height: 32,
            depth: 1,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 1,
            resource_options: 0,
            protection_options: 0,
            swizzle: None,
        },
    };
    // The buffer does not resolve, so the load answers `None`. The payment is
    // owed regardless — it precedes every way this rail can fail.
    let _ = load_buffer_texture_rgba(&mut state, &mut host, task_id, texture_ref, &bt);

    assert!(
        state.pending_writebacks.get(texture_ref).is_none(),
        "the sampled bind never asked the ledger what its texture reference owed"
    );
}
