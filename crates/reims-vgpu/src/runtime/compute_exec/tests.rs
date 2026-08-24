#![allow(
    clippy::field_reassign_with_default,
    reason = "wire fixtures are assembled field by field to keep each protocol case explicit"
)]

use super::*;
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
use crate::runtime::decode::compute;
use crate::runtime::decode::resource::{
    list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_BUFFER, OBJECT_TYPE_IOSURFACE,
    OBJECT_TYPE_SERIALIZER_RESOURCE, OBJECT_TYPE_TEXTURE, RESOURCE_PAGE_SHIFT,
};
/// Compute-pipeline descriptor constants used by the backend execute test.
use crate::runtime::decode::resource::{
    OBJECT_TYPE_FUNCTION, PIPELINE_TAG_KERNEL_FUNC, SERIALIZER_RESOURCE_FIRST_TLVS,
    SERIALIZER_RESOURCE_OBJECT_COMPUTE_PIPELINE,
};
use crate::runtime::gva_mem;
use crate::runtime::gva_mem::write_task_gva_arm64e;
use crate::runtime::host::FakeHost;
use reims_vgpu_core::endian::{st32, st64};
use reims_vgpu_paging::geometry::{
    DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN, MAPPER_PAGE_ENTRY_PFN_SHIFT as PAGE_ENTRY_PFN_SHIFT,
    MAPPER_PAGE_ENTRY_VALID as PAGE_ENTRY_VALID,
};
use reims_vgpu_wire::device_desc::IOSurfacePlaneViewBuilder;

fn barrier_test_resource(
    kind: reims_vgpu_protocol::ObjectKind,
) -> std::sync::Arc<crate::model::TaskResource> {
    std::sync::Arc::new(crate::model::TaskResource::new(
        crate::runtime::decode::resource::ListObjectEntry::new(kind, 0, 0),
        std::sync::Arc::from(Vec::<u8>::new()),
    ))
}

#[test]
fn compute_barriers_preserve_scope_resource_generation_and_lifetime() {
    let state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();

    let mut scope = ComputeCommand::default();
    scope.kind = Kind::BarrierScope;
    scope.barrier_scope = 3;
    assert_eq!(
        resolved_compute_barrier(&state, &host, 1, &scope),
        Ok(Some(reims_vgpu_core::ComputeBarrier::Scope(
            reims_vgpu_core::MemoryBarrierScope::from_bits(3).unwrap(),
        )))
    );

    let first = state.task_objects.resources.register(
        1,
        9,
        barrier_test_resource(reims_vgpu_protocol::ObjectKind::Buffer),
    );
    let mut resources = ComputeCommand::default();
    resources.kind = Kind::BarrierResources;
    resources.resources = vec![reims_vgpu_protocol::ObjectTableRef::new(9)];
    let first_barrier = resolved_compute_barrier(&state, &host, 1, &resources)
        .expect("resource-list barrier resolves")
        .expect("nonempty resource list");
    let reims_vgpu_core::ComputeBarrier::Resources(first_resources) = first_barrier else {
        panic!("resource-list barrier changed kind")
    };
    assert_eq!(first_resources[0].id, first.semantic_id().unwrap());
    assert_eq!(first_resources[0].lifetime.id(), first.lifetime().id());

    assert!(state.task_objects.resources.delete(1, 9));
    let replacement = state.task_objects.resources.register(
        1,
        9,
        barrier_test_resource(reims_vgpu_protocol::ObjectKind::Buffer),
    );
    let replacement_barrier = resolved_compute_barrier(&state, &host, 1, &resources)
        .expect("replacement resource resolves")
        .expect("nonempty resource list");
    let reims_vgpu_core::ComputeBarrier::Resources(replacement_resources) = replacement_barrier
    else {
        panic!("resource-list barrier changed kind")
    };
    assert_eq!(
        replacement_resources[0].id,
        replacement.semantic_id().unwrap()
    );
    assert_ne!(first_resources[0].id, replacement_resources[0].id);
    assert_ne!(
        first_resources[0].lifetime.id(),
        replacement_resources[0].lifetime.id()
    );

    resources.resources = vec![reims_vgpu_protocol::ObjectTableRef::new(99)];
    assert_eq!(
        resolved_compute_barrier(&state, &host, 1, &resources),
        Err(ComputeBarrierRefusal::ResourceUnavailable {
            index: 0,
            object_ref: 99,
        })
    );

    resources.resources.clear();
    assert_eq!(
        resolved_compute_barrier(&state, &host, 1, &resources),
        Ok(None)
    );
}

#[test]
fn compute_barriers_fail_visible_and_survive_a_refused_dispatch() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut seg = crate::runtime::compute_session::ComputeSegment::default();

    let mut barrier = ComputeCommand::default();
    barrier.kind = Kind::BarrierScope;
    barrier.barrier_scope = 1;
    assert_eq!(
        apply_record(&mut state, &mut host, 1, &barrier, &mut seg),
        None
    );
    assert_eq!(seg.pending_barriers.len(), 1);
    barrier.barrier_scope = 2;
    assert_eq!(
        apply_record(&mut state, &mut host, 1, &barrier, &mut seg),
        None
    );
    assert_eq!(
        seg.pending_barriers,
        vec![
            reims_vgpu_core::ComputeBarrier::Scope(reims_vgpu_core::MemoryBarrierScope::BUFFERS),
            reims_vgpu_core::ComputeBarrier::Scope(reims_vgpu_core::MemoryBarrierScope::TEXTURES),
        ]
    );

    let mut dispatch = ComputeCommand::default();
    dispatch.kind = Kind::DispatchThreadgroups;
    dispatch.grid = compute::Size3 { x: 1, y: 1, z: 1 };
    dispatch.threads_per_threadgroup = compute::Size3 { x: 1, y: 1, z: 1 };
    assert!(matches!(
        apply_record(&mut state, &mut host, 1, &dispatch, &mut seg),
        Some(ComputeStatus::MissingPipeline(_))
    ));
    assert_eq!(seg.pending_barriers.len(), 2);
    assert_eq!(
        retire_pending_compute_barriers(ComputeStatus::Ok, &mut seg.pending_barriers),
        ComputeStatus::Ok
    );
    assert!(seg.pending_barriers.is_empty());

    barrier.barrier_scope = 8;
    assert_eq!(
        apply_record(&mut state, &mut host, 1, &barrier, &mut seg),
        None
    );
    assert_eq!(
        apply_record(&mut state, &mut host, 1, &dispatch, &mut seg),
        Some(ComputeStatus::Unsupported(
            "compute_barrier_scope_unsupported"
        ))
    );
}

#[test]
fn compute_fence_wait_becomes_a_dependency_and_an_unsatisfied_wait_blocks_dispatch() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut seg = crate::runtime::compute_session::ComputeSegment::default();

    let mut update = ComputeCommand::default();
    update.kind = Kind::UpdateFence;
    update.fence_ref = 17;
    assert_eq!(
        apply_record(&mut state, &mut host, 1, &update, &mut seg),
        None
    );

    let mut wait = ComputeCommand::default();
    wait.kind = Kind::WaitFence;
    wait.fence_ref = 17;
    assert_eq!(
        apply_record(&mut state, &mut host, 1, &wait, &mut seg),
        None
    );
    assert_eq!(
        seg.pending_barriers,
        vec![reims_vgpu_core::ComputeBarrier::Fence]
    );
    assert_eq!(seg.barrier_block, None);

    let mut pending = crate::runtime::compute_session::ComputeSegment::default();
    wait.fence_ref = 23;
    assert_eq!(
        apply_record(&mut state, &mut host, 1, &wait, &mut pending),
        None
    );
    assert_eq!(pending.pending_barriers, Vec::new());
    assert_eq!(
        pending.barrier_block,
        Some(ComputeBarrierRefusal::FenceWaitPending { fence_ref: 23 })
    );
}

/// A whole-workgroup dispatch of `counts` groups of `local` threads.
///
/// Fixtures here name workgroup counts, because that is what they dispatch.
/// The plan also carries the exact thread grid a translated kernel would cull
/// against, and for whole workgroups that is every thread of every group — so
/// the local size has to be named rather than assumed, and a fixture whose
/// SPIR-V declares `LocalSize 64 1 1` says so here too.
fn whole_workgroups(
    counts: [u32; 3],
    local: [u32; 3],
) -> reims_vgpu_protocol::dispatch::WorkgroupPlan {
    reims_vgpu_protocol::dispatch::workgroup_counts(counts, local, false)
        .expect("fixture dispatch dimensions are non-zero")
}

#[test]
fn a_single_level_resident_never_replaces_a_complete_sampled_mip_chain() {
    assert!(can_bind_linear_target_resident(false, false, true));
    assert!(
        !can_bind_linear_target_resident(false, true, true),
        "a complete sampled allocation owns levels absent from the render resident"
    );
    assert!(!can_bind_linear_target_resident(true, false, true));
    assert!(!can_bind_linear_target_resident(false, false, false));
}

#[test]
fn argument_buffer_reflection_decline_carries_the_owner_coordinate() {
    use crate::observe::Decline as _;
    let decline = ComputeReflectionDecline::ReflectedResourceUnsupported {
        pipeline_ref: 7,
        index: 9,
        binding: Some(41),
        kind: "embedded_texture",
    };
    assert_eq!(decline.slug(), "compute_reflection_resource_unsupported");
    assert_eq!(
        decline.fields(),
        vec![
            ("pipeline_ref", "7".into()),
            ("index", "9".into()),
            ("binding", "41".into()),
            ("kind", "embedded_texture".into()),
        ]
    );

    let interface = ComputeReflectionDecline::ReflectedInterfaceUnsupported {
        pipeline_ref: 7,
        feature: "kernel_imageblock",
        count: 2,
    };
    assert_eq!(interface.slug(), "compute_reflection_interface_unsupported");
    assert_eq!(
        interface.fields(),
        vec![
            ("pipeline_ref", "7".into()),
            ("feature", "kernel_imageblock".into()),
            ("count", "2".into()),
        ]
    );
}

/// A IOSurface plane view view names its IOSurface plane on the wire (record `+0x20`, the
/// `newTextureWithDescriptor:iosurface:plane:` argument). When two planes share
/// geometry and bytes-per-element the geometry scan cannot separate them and
/// falls back to inventing a packed window at offset 0 — which is the *first*
/// plane's bytes. The wire index is the only key, and this path already decoded
/// it, so a compute stage of the alpha plane must not read the luma plane.
///
/// Shape is the live v0a8 (biplanar video + alpha) layout scaled down: plane 0
/// and plane 2 are both R8 at identical dims, plane 1 is the RG8 chroma.
#[test]
fn stage_texture_iosurface_plane_view_plane_index_beats_the_ambiguous_geometry_scan() {
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
    use reims_vgpu_core::endian::st16;
    use reims_vgpu_core::pixel_format::{MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_R8_UNORM};
    use reims_vgpu_protocol::{
        DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_LEN, DEVICE_DESC_PLANES, DEVICE_DESC_PLANE_COUNT,
        DEVICE_PLANE_BPE, DEVICE_PLANE_BPR, DEVICE_PLANE_DESC_LEN, DEVICE_PLANE_DIMS,
        DEVICE_PLANE_OFFSET, DEVICE_PLANE_SIZE,
    };

    // elemW@0, width u24@1, elemH@4, height u24@5.
    fn plane_dims(width: u32, height: u32) -> u64 {
        ((width as u64 & 0xff_ffff) << 8) | ((height as u64 & 0xff_ffff) << 40)
    }

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    let sid = 3u32;
    let iosurface_plane_view_ref = 10u32;
    let pfn = 0x20u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0x5a);
    assert!(state.map_surface(sid));
    {
        let m = state.surfaces.mappings.get_mut(&sid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    assert!(state.set_mapping_geom(sid, 4, 4, MTL_FORMAT_BGRA8_UNORM));

    let mut device_desc = vec![0u8; DEVICE_DESC_LEN];
    st32(&mut device_desc[DEVICE_DESC_ALLOC_SIZE..], 192);
    device_desc[DEVICE_DESC_PLANE_COUNT] = 3;
    // (offset, size, w, h, bpr, bpe): Y and alpha are indistinguishable by dims.
    let planes = [
        (0u32, 64u32, 4u32, 4u32, 16u32, 1u16),
        (64, 64, 2, 2, 16, 2),
        (128, 64, 4, 4, 16, 1),
    ];
    for (i, (off, size, w, h, bpr, bpe)) in planes.iter().enumerate() {
        let base = DEVICE_DESC_PLANES + i * DEVICE_PLANE_DESC_LEN;
        st32(&mut device_desc[base + DEVICE_PLANE_OFFSET..], *off);
        st32(&mut device_desc[base + DEVICE_PLANE_SIZE..], *size);
        st64(
            &mut device_desc[base + DEVICE_PLANE_DIMS..],
            plane_dims(*w, *h),
        );
        st32(&mut device_desc[base + DEVICE_PLANE_BPR..], *bpr);
        st16(&mut device_desc[base + DEVICE_PLANE_BPE..], *bpe);
    }
    assert!(state.set_mapping_device_desc(sid, &device_desc));

    // 56-byte IOSurface plane view blob: 8-byte head, then kind/blob_len/own_ref and a 0x24
    // record whose `+0x20` carries the plane index.
    let desc_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E;
    let iosurface_plane_view_desc = IOSurfacePlaneViewBuilder::new(sid, 0, 10, 0x42)
        .unknown(0x01)
        .geometry(MTL_FORMAT_R8_UNORM, 4, 4, 1)
        .trailer([1, 0, 1, 0, 1, 0, 0x10, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        // IOSurface plane index = 2 (alpha)
        .plane_index(2);
    let iosurface_plane_view_desc = iosurface_plane_view_desc.bytes();
    write_task_gva_arm64e(
        &mut host,
        &state.tasks[1],
        desc_gva,
        iosurface_plane_view_desc,
    );
    let off = list_object_entry_offset(iosurface_plane_view_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed =
        (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((iosurface_plane_view_desc.len() as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    let staged = stage_texture_raw(
        &mut state,
        &mut host,
        1,
        iosurface_plane_view_ref,
        33,
        ComputeTextureStage::Storage2d,
    )
    .expect("a IOSurface plane view plane view over a mapped surface must stage");
    match staged.writeback {
        TextureWriteback::IOSurface {
            surface_offset,
            surface_bpr,
            ..
        } => {
            assert_eq!(
                (surface_offset, surface_bpr),
                (128, 16),
                "plane 2's own window; (0, 128) is the invented packed window \
                 over plane 0 that the ambiguous geometry scan falls back to"
            );
        }
        _ => panic!("a IOSurface plane view view over a surface mapping must write back as IOSurface texture"),
    }
}

#[test]
fn compute_bind_overflow_drops_the_bind_but_keeps_in_cap_and_unbinds() {
    let mut acc = ComputeAccum::default();
    // In-cap buffer bind (ref != 0) is kept.
    acc.bind_buffers(
        5,
        &[BufferBinding {
            ref_: 7,
            ..Default::default()
        }],
    );
    assert_eq!(acc.buffers.len(), 1);
    assert_eq!(acc.buffers[0].index, 5);

    // Over-cap buffer bind (index 40 > MAX_COMPUTE_BUFFER_SLOTS) is dropped —
    // the drop is fail-visible via `ComputeBindOverflow` (the log itself is a
    // global sink; the line's shape is asserted by
    // `every_compute_bind_table_renders_its_own_slug`). No new buffer slot appears.
    acc.bind_buffers(
        MAX_COMPUTE_BUFFER_SLOTS + 9,
        &[BufferBinding {
            ref_: 9,
            ..Default::default()
        }],
    );
    assert_eq!(acc.buffers.len(), 1, "over-cap bind must not be stored");

    // Boundary: index == MAX is OUT of range (the backend sizes its arg-table
    // array to MAX and guards `idx >= MAX`), so it must be dropped too — this
    // is the off-by-one the `>` → `>=` alignment fixed. Slot MAX-1 is the last
    // valid slot and is kept.
    acc.bind_buffers(
        MAX_COMPUTE_BUFFER_SLOTS,
        &[BufferBinding {
            ref_: 11,
            ..Default::default()
        }],
    );
    assert_eq!(
        acc.buffers.len(),
        1,
        "index == MAX is out of range, dropped"
    );
    acc.bind_buffers(
        MAX_COMPUTE_BUFFER_SLOTS - 1,
        &[BufferBinding {
            ref_: 12,
            ..Default::default()
        }],
    );
    assert_eq!(
        acc.buffers.len(),
        2,
        "index == MAX-1 is the last valid slot"
    );

    // A zero-ref entry is an unbind: expected control flow, no new slot.
    acc.bind_buffers(
        6,
        &[BufferBinding {
            ref_: 0,
            ..Default::default()
        }],
    );
    assert_eq!(acc.buffers.len(), 2, "unbind (ref==0) adds no slot");

    // Threadgroup memory has no cap here: the accumulator keeps whatever slot
    // the guest names, and the one backend with an argument table refuses at its
    // own encoder. A cap here would also have bound the Vulkan rail, which
    // consumes none of these binds.
    acc.set_threadgroup_memory(u32::MAX, 256);
    acc.set_threadgroup_memory(64, 512);
    assert_eq!(
        acc.threadgroup_memory.len(),
        2,
        "a slot past any host table is still recorded, not dropped here"
    );
}

/// Each argument table renders its own `reason=`, and the line keeps the shape
/// the log has always carried.
///
/// The slugs used to live inside a `format!` string, where a later decline
/// spelling one of them would have shared this path's `fail_once` latch and
/// silenced one of the two for the boot, with nothing failing. They are
/// `slug()` bodies now, where a reader looking for the vocabulary will find
/// them; this pins that moving them did not change what a reader greps for, and
/// that the three tables stay distinguishable.
///
/// There were four. The threadgroup-memory table left this enum when its cap
/// did: it is bounded by a Metal argument table rather than by the protocol, so
/// the refusal belongs to the encoder that owns the table and is named there.
#[test]
fn every_compute_bind_table_renders_its_own_slug() {
    use crate::observe::Emit;
    use crate::runtime::compute_exec::ComputeBindOverflow as O;

    assert_eq!(
        Emit::decline(
            "compute_bind_overflow",
            &O::Buffer {
                index: 40,
                arg: 9,
                cap: MAX_COMPUTE_BUFFER_SLOTS,
            },
        )
        .render(),
        "compute_bind_overflow reason=buffer_index_overflow index=40 arg=9 cap=31"
    );

    let slugs: Vec<&str> = [
        O::Buffer {
            index: 0,
            arg: 0,
            cap: 0,
        },
        O::Texture {
            index: 0,
            arg: 0,
            cap: 0,
        },
        O::Sampler {
            index: 0,
            arg: 0,
            cap: 0,
        },
    ]
    .iter()
    .map(|o| crate::observe::Decline::slug(o))
    .collect();
    let unique: std::collections::BTreeSet<&str> = slugs.iter().copied().collect();
    assert_eq!(
        unique.len(),
        slugs.len(),
        "two tables sharing a slug would share fail_once's latch: {slugs:?}"
    );
}

/// A nil entry over an *occupied* compute slot clears it.
///
/// `compute_bind_overflow_drops_the_bind_but_keeps_in_cap_and_unbinds` covers
/// the easy half — a nil at a slot that was already empty adds nothing — and
/// that half passes whether the arm clears or merely skips. This is the half
/// that separates them, and it is the half with a consequence: a retained bind
/// is staged again on the next dispatch, and a texture slot the guest unbound
/// still receives `writeback_texture`'s output into its guest surface.
///
/// The rule is the render rail's, over the same `[first][count][ref x count]`
/// wire form: `ExecResult::buffer_unbinds` states that explicit nil entries
/// "must remove prior slot state rather than silently retaining a stale
/// resource", and `exec::apply_binds` retains-by-slot to do it. All three
/// compute bind kinds are asserted, because the arm was wrong in all three.
#[test]
fn a_nil_entry_clears_an_occupied_compute_slot() {
    let mut acc = ComputeAccum::default();
    acc.bind_buffers(
        3,
        &[BufferBinding {
            ref_: 77,
            ..Default::default()
        }],
    );
    acc.bind_textures(3, &[RefBinding { ref_: 78 }]);
    acc.bind_samplers(
        3,
        &[SamplerBinding {
            ref_: 79,
            ..Default::default()
        }],
    );
    assert_eq!(
        (acc.buffers.len(), acc.textures.len(), acc.samplers.len()),
        (1, 1, 1)
    );

    acc.bind_buffers(
        3,
        &[BufferBinding {
            ref_: 0,
            ..Default::default()
        }],
    );
    acc.bind_textures(3, &[RefBinding { ref_: 0 }]);
    acc.bind_samplers(
        3,
        &[SamplerBinding {
            ref_: 0,
            ..Default::default()
        }],
    );
    assert_eq!(
        (acc.buffers.len(), acc.textures.len(), acc.samplers.len()),
        (0, 0, 0),
        "a nil entry must remove the slot it names, not leave the previous bind live"
    );
}

#[test]
fn accum_pipeline_buffer_texture_sampler() {
    let mut acc = ComputeAccum::default();
    acc.set_pipeline(9);
    acc.bind_buffers(
        1,
        &[BufferBinding {
            ref_: 3,
            offset: 16,
            attribute_stride: 0,
            has_attribute_stride: false,
        }],
    );
    acc.bind_textures(0, &[RefBinding { ref_: 10 }, RefBinding { ref_: 11 }]);
    acc.bind_samplers(
        0,
        &[SamplerBinding {
            ref_: 20,
            lod_min_bits: 0,
            lod_max_bits: 0,
            has_lod_clamp: false,
        }],
    );
    assert_eq!(acc.pipeline_ref, 9);
    assert_eq!(acc.buffers.len(), 1);
    assert_eq!(acc.textures.len(), 2);
    assert_eq!(acc.textures[1].texture_ref, 11);
    assert_eq!(acc.samplers[0].sampler_ref, 20);
}

#[test]
fn accum_stage_in_tg_imageblock_and_control_fail_closed() {
    let mut acc = ComputeAccum::default();
    acc.set_stage_in_region(StageInRegion {
        origin_x: 1,
        origin_y: 2,
        origin_z: 0,
        size_x: 8,
        size_y: 4,
        size_z: 1,
    });
    assert!(acc.stage_in_region.is_some());
    acc.set_stage_in_region_indirect(3, 16);
    assert!(acc.stage_in_region.is_none());
    assert_eq!(
        acc.stage_in_region_indirect.as_ref().map(|i| i.buffer_ref),
        Some(3)
    );
    acc.set_threadgroup_memory(2, 256);
    acc.set_threadgroup_memory(2, 512);
    assert_eq!(acc.threadgroup_memory.len(), 1);
    assert_eq!(acc.threadgroup_memory[0].length, 512);
    acc.set_imageblock(4, 4);
    assert_eq!(acc.imageblock.as_ref().map(|d| d.width), Some(4));

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut seg = crate::runtime::compute_session::ComputeSegment {
        acc,
        ..Default::default()
    };
    let mut cmd = ComputeCommand::default();
    // Empty start-do-while encodes without a condition buffer.
    cmd.kind = Kind::ControlStartDoWhile;
    let st = apply_record(&mut state, &mut host, 1, &cmd, &mut seg);
    assert!(
        matches!(
            st,
            Some(ComputeStatus::Ok)
                | Some(ComputeStatus::BackendFailed(_))
                | Some(ComputeStatus::Unsupported(_))
        ),
        "unexpected {st:?}"
    );
    cmd.kind = Kind::ExecuteCommandsInBuffer;
    cmd.indirect_command_buffer_ref = 99;
    let st = apply_record(&mut state, &mut host, 1, &cmd, &mut seg);
    // Missing object-list entry → MissingBuffer; still latches sequencing.
    assert!(
        matches!(
            st,
            Some(ComputeStatus::MissingBuffer(_)) | Some(ComputeStatus::Unsupported(_))
        ),
        "unexpected {st:?}"
    );
    assert!(seg.block.is_some());
    if let Some(s) = seg.session.take() {
        let _ = s.finish(&mut host, &mut state, 1);
    }
}

#[test]
fn vulkan_ignores_stage_input_regions_without_a_stage_input_descriptor() {
    let mut acc = ComputeAccum::default();
    assert!(!super::linux_stage_input_or_imageblock_unsupported(
        false, &acc
    ));
    assert!(super::linux_stage_input_or_imageblock_unsupported(
        true, &acc
    ));

    acc.set_stage_in_region(StageInRegion {
        origin_x: 0,
        origin_y: 0,
        origin_z: 0,
        size_x: 1,
        size_y: 1,
        size_z: 1,
    });
    assert!(!super::linux_stage_input_or_imageblock_unsupported(
        false, &acc
    ));

    acc.set_stage_in_region_indirect(3, 16);
    assert!(acc.stage_in_region.is_none());
    assert!(!super::linux_stage_input_or_imageblock_unsupported(
        false, &acc
    ));
    assert!(super::linux_stage_input_or_imageblock_unsupported(
        true, &acc
    ));
}

#[test]
fn resolve_indirect_threadgroups_from_buffer() {
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    let args = [2u32, 3, 1];
    let arg_bytes: Vec<u8> = args.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf_gva = 5u64 << RESOURCE_PAGE_SHIFT;
    write_task_gva_arm64e(&mut host, &state.tasks[1], buf_gva, &arg_bytes);
    let mut bdesc = vec![0u8; 16];
    st64(&mut bdesc[0..], 12);
    st32(&mut bdesc[8..], 5);
    let bdesc_gva = 0x180u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], bdesc_gva, &bdesc);
    {
        let off = list_object_entry_offset(7, 32).unwrap();
        let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_BUFFER as u32) | (16u32 << 8);
        st32(&mut le[0..], packed);
        le[4..12].copy_from_slice(&bdesc_gva.to_le_bytes());
        write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);
    }

    let mut cmd = ComputeCommand::default();
    cmd.kind = Kind::DispatchThreadgroupsIndirect;
    cmd.indirect_buffer_ref = 7;
    cmd.indirect_buffer_offset = 0;
    cmd.threads_per_threadgroup = compute::Size3 { x: 8, y: 1, z: 1 };
    let dims = resolve_dispatch_dims(&mut state, &host, 1, &cmd).unwrap();
    // The grid comes from the indirect buffer and the threadgroup from the
    // wire, which is the whole point of this arm — asserting them as one flat
    // septuple could not say which source each half came from.
    assert_eq!(dims.grid, Extent3 { x: 2, y: 3, z: 1 });
    assert_eq!(dims.threadgroup, Extent3 { x: 8, y: 1, z: 1 });
    assert!(!dims.dispatch_threads);
}

/// `DispatchThreadsIndirect` reads both extents from the buffer, at the two
/// halves of `MTLDispatchThreadsIndirectArguments`.
///
/// Six distinct values, because this arm used to be six
/// `u32_dim(ld32(&raw[N..]))` calls differing only by the literals
/// `0, 4, 8, 12, 16, 20`. A transposition among those dispatches a valid grid
/// of the wrong shape, which nothing downstream can tell from the right one —
/// so every component here differs from every other, and `threads_per_threadgroup`
/// on the wire is set to a value appearing nowhere in the buffer, because this
/// arm must not read it.
#[test]
fn dispatch_threads_indirect_reads_both_extents_from_the_buffer() {
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    // threadsPerGrid[3] then threadsPerThreadgroup[3], LE u32 each.
    let args = [11u32, 22, 33, 44, 55, 66];
    let arg_bytes: Vec<u8> = args.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf_gva = 5u64 << RESOURCE_PAGE_SHIFT;
    write_task_gva_arm64e(&mut host, &state.tasks[1], buf_gva, &arg_bytes);
    let mut bdesc = vec![0u8; 16];
    st64(&mut bdesc[0..], 24);
    st32(&mut bdesc[8..], 5);
    let bdesc_gva = 0x180u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], bdesc_gva, &bdesc);
    {
        let off = list_object_entry_offset(7, 32).unwrap();
        let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_BUFFER as u32) | (16u32 << 8);
        st32(&mut le[0..], packed);
        le[4..12].copy_from_slice(&bdesc_gva.to_le_bytes());
        write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);
    }

    let mut cmd = ComputeCommand::default();
    cmd.kind = Kind::DispatchThreadsIndirect;
    cmd.indirect_buffer_ref = 7;
    cmd.indirect_buffer_offset = 0;
    cmd.threads_per_threadgroup = compute::Size3 {
        x: 99,
        y: 99,
        z: 99,
    };

    let dims = resolve_dispatch_dims(&mut state, &host, 1, &cmd).unwrap();
    assert_eq!(
        dims.grid,
        Extent3 {
            x: 11,
            y: 22,
            z: 33
        }
    );
    assert_eq!(
        dims.threadgroup,
        Extent3 {
            x: 44,
            y: 55,
            z: 66
        }
    );
    assert!(dims.dispatch_threads);
}

/// The wire shape that used to be "recovered" is now a named refusal.
///
/// `grid = [45, u64::MAX, 1]`, `tg = [32, 0, 1]` is a threadgroup with **zero**
/// threads next to a grid axis of `u64::MAX` — both `y` components garbage while
/// `x` and `z` are sane, which is a decode defect, not a sentinel the guest
/// sends on purpose. The dispatch dimensions are taken from the wire and
/// nowhere else, so this refuses with the slug of the check that refused it
/// rather than substituting a grid derived from whichever bound texture happens
/// to be largest.
#[test]
fn the_zero_threadgroup_wire_shape_is_refused_by_name() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 8, 8);
    // A bound write target at a plausible full-screen geometry: the deleted
    // heuristic sourced its invented grid from exactly this, so its presence is
    // what makes the refusal meaningful rather than incidental.
    assert!(state.map_surface(3));
    assert!(state.set_mapping_geom(3, 1440, 1080, 0x73));

    let mut cmd = ComputeCommand::default();
    cmd.kind = Kind::DispatchThreadgroups;
    cmd.grid = compute::Size3 {
        x: 45,
        y: u64::MAX,
        z: 1,
    };
    cmd.threads_per_threadgroup = compute::Size3 { x: 32, y: 0, z: 1 };
    assert_eq!(
        resolve_dispatch_dims(&mut state, &host, 1, &cmd).unwrap_err(),
        ComputeStatus::BadGrid("compute_grid_dim_range"),
    );
    // Each garbage component refuses on its own account: `u64::MAX` overflows
    // `u32` and `0` is not a dispatchable extent.
    cmd.grid.y = 3;
    assert_eq!(
        resolve_dispatch_dims(&mut state, &host, 1, &cmd).unwrap_err(),
        ComputeStatus::BadGrid("compute_grid_dim_range"),
        "tg.y == 0 must still refuse"
    );
    cmd.threads_per_threadgroup.y = 32;
    assert!(
        resolve_dispatch_dims(&mut state, &host, 1, &cmd).is_ok(),
        "a wholly sane grid must pass"
    );
}

/// The rail's whole point after this migration: a refusal renders the
/// registered slug of the check that refused, and a success cannot render
/// anything at all. Before the payload existed, every `MissingTexture`
/// site produced the same untyped line and 25 checks were indistinguishable.
#[test]
fn a_compute_refusal_names_its_check_and_ok_names_nothing() {
    use crate::observe::{Emit, Refusal};

    assert!(ComputeStatus::Ok.refusal().is_none());
    assert!(
        Emit::refusal("compute_record", &ComputeStatus::Ok).is_none(),
        "an Ok must not be loggable — that is what keeps the sink clean"
    );

    let st = ComputeStatus::MissingTexture("compute_stage_tex_iosurface_plane_view_no_map");
    assert_eq!(
        st.refusal(),
        Some("compute_stage_tex_iosurface_plane_view_no_map")
    );
    assert_eq!(st.class(), "missing_texture");
    let line = Emit::refusal("compute_record", &st)
        .expect("a refusal renders a line")
        .field("pipe", 7)
        .render();
    assert_eq!(
        line,
        "compute_record reason=compute_stage_tex_iosurface_plane_view_no_map \
             class=missing_texture pipe=7"
    );
}

/// Two different buffer-staging checks, two different slugs — the property
/// that a shared `MissingBuffer` could not express.
///
/// The window path's assertion used to read `compute_buf_win_no_backing`, and
/// this comment used to say `ref=0` made "both paths refuse on their first
/// gate". Only one of them did. `no_backing` is the *last* of that path's four
/// refusals and was returned for all four, so the slug named the fourth gate for
/// a record that never reached the first — and this test asserted that as the
/// intended behaviour. Each refusal now answers under its own name.
#[test]
fn the_buffer_paths_refuse_under_their_own_names() {
    use crate::observe::Refusal;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    // `ref=0` is the unbound sentinel, and the window path now says that rather
    // than reporting the outcome of a resolution it never attempted.
    let window = read_buffer_window(&state, &host, 1, 0, 0, 4);
    assert_eq!(
        window.err().and_then(|e| e.refusal()),
        Some("compute_buf_win_ref_unbound")
    );

    // A bound-looking ref naming an empty list slot refuses on the first rung,
    // under this rail's own role — distinct from both of the above.
    let unresolvable = read_buffer_window(&state, &host, 1, 7, 0, 4);
    assert_eq!(
        unresolvable.err().and_then(|e| e.refusal()),
        Some(crate::observe::ladder_slug!(
            "compute_buf_win",
            no_list_entry
        ))
    );

    let bind = ComputeBufferBind {
        index: 0,
        buffer_ref: 0,
        offset: 0,
        attribute_stride: 0,
        has_attribute_stride: false,
    };
    let staged = stage_buffer_with_extent(&mut state, &mut host, 1, &bind, None);
    assert_eq!(
        staged.err().and_then(|e| e.refusal()),
        Some(crate::observe::ladder_slug!(
            "compute_stage_buf",
            no_list_entry
        ))
    );
}

#[test]
fn a_compute_extent_stages_only_the_proven_prefix_from_the_bound_offset() {
    let mut host = FakeHost::new();
    host.stable_map_pages = true;
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    let buffer_gva = 5u64 << RESOURCE_PAGE_SHIFT;
    let contents: Vec<u8> = (0..64).collect();
    write_task_gva_arm64e(&mut host, &state.tasks[1], buffer_gva, &contents);
    let descriptor_gva = 0x180u64;
    let mut descriptor = [0u8; 16];
    st64(&mut descriptor[0..], contents.len() as u64);
    st32(&mut descriptor[8..], 5);
    write_task_gva_arm64e(&mut host, &state.tasks[1], descriptor_gva, &descriptor);
    let entry_offset = list_object_entry_offset(7, 32).unwrap();
    let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(&mut entry[0..], (OBJECT_TYPE_BUFFER as u32) | (16u32 << 8));
    entry[4..12].copy_from_slice(&descriptor_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], entry_offset, &entry);

    let bind = ComputeBufferBind {
        index: 0,
        buffer_ref: 7,
        offset: 8,
        attribute_stride: 0,
        has_attribute_stride: false,
    };
    crate::runtime::guest_ram_map::reset();
    crate::runtime::guest_ram::latch_import_limits(1 << PAGE_SHIFT_ARM64E, 1 << 30, 1 << 30);
    let staged =
        stage_buffer_with_extent(&mut state, &mut host, 1, &bind, Some(12)).expect("stages");
    crate::runtime::guest_ram::forget_import_limits();
    crate::runtime::guest_ram_map::reset();
    assert_eq!(staged.gva, buffer_gva + 8);
    match staged.input {
        VulkanBufferInput::HostBytes(_) => panic!("an importable buffer must retain guest pages"),
        VulkanBufferInput::GuestPages(source) => {
            assert_eq!(source.total_len, 12);
            assert_eq!(source.source_offset, 8);
        }
    }
    assert!(staged.bytes.is_empty(), "the input has one typed owner");
    assert_eq!(staged.pages.len(), 1);

    let private_descriptor_gva = 0x1c0u64;
    st64(&mut descriptor[8..], (1u64 << 32) | 5);
    write_task_gva_arm64e(
        &mut host,
        &state.tasks[1],
        private_descriptor_gva,
        &descriptor,
    );
    let private_entry_offset = list_object_entry_offset(8, 32).unwrap();
    entry[4..12].copy_from_slice(&private_descriptor_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], private_entry_offset, &entry);
    let private_bind = ComputeBufferBind {
        buffer_ref: 8,
        ..bind
    };
    let staged = stage_buffer_with_extent(&mut state, &mut host, 1, &private_bind, Some(12))
        .expect("private buffer stages");
    match staged.input {
        VulkanBufferInput::HostBytes(bytes) => assert_eq!(bytes, contents[8..20]),
        VulkanBufferInput::GuestPages(_) => {
            panic!("a private buffer must snapshot its transfer bytes")
        }
    }
}

/// Callers must pass page_shift explicitly; 12 and 14 place handle differently.
#[test]
fn buffer_backing_gva_requires_explicit_page_shift() {
    use crate::runtime::decode::resource::BufferDescriptor;
    let d = BufferDescriptor {
        allocation_size: 0x1000,
        handle64: 0x101,
        handle: 0x101,
        is_private: false,
    };
    let (gva12, _) = d.backing_gva_size(PAGE_SHIFT_X86).expect("12");
    let (gva14, _) = d.backing_gva_size(PAGE_SHIFT_ARM64E).expect("14");
    assert_eq!(gva12, 0x101000, "x86 handle<<12");
    assert_eq!(gva14, 0x404000, "arm handle<<14");
    assert_ne!(gva12, gva14);
}

#[test]
fn dispatch_missing_pipeline() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));
    let acc = ComputeAccum::default();
    let mut cmd = ComputeCommand::default();
    cmd.kind = Kind::DispatchThreadgroups;
    cmd.grid = compute::Size3 { x: 1, y: 1, z: 1 };
    cmd.threads_per_threadgroup = compute::Size3 { x: 1, y: 1, z: 1 };
    let st = execute_dispatch(&mut state, &mut host, 1, &acc, &cmd, &[]);
    // The slug names *which* pipeline check refused, and it differs by
    // backend: both arms open with `pipeline_ref == 0`, and before the
    // status carried a reason the two were indistinguishable in the log.

    assert_eq!(
        st,
        ComputeStatus::MissingPipeline("compute_vk_pipeline_ref_zero")
    );
}

/// A dispatch reaches the Vulkan encoder while nested sessions refuse by name.
#[test]
fn dispatch_backend_unavailable_with_texture_binds() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));
    let mut acc = ComputeAccum::default();
    acc.set_pipeline(42);
    acc.bind_textures(0, &[RefBinding { ref_: 111 }]);
    acc.bind_buffers(
        0,
        &[BufferBinding {
            ref_: 7,
            offset: 0,
            attribute_stride: 0,
            has_attribute_stride: false,
        }],
    );
    let mut cmd = ComputeCommand::default();
    cmd.kind = Kind::DispatchThreadgroups;
    cmd.grid = compute::Size3 {
        x: 60,
        y: u64::MAX,
        z: 1,
    };
    cmd.threads_per_threadgroup = compute::Size3 { x: 32, y: 0, z: 1 };
    let st = execute_dispatch(&mut state, &mut host, 1, &acc, &cmd, &[]);
    assert!(
        matches!(
            st,
            ComputeStatus::MissingPipeline(_)
                | ComputeStatus::MissingMtlb(_)
                | ComputeStatus::MissingTexture(_)
                | ComputeStatus::BackendFailed(_)
                | ComputeStatus::Unsupported(_)
        ),
        "vulkan path attempts encode, got {st:?}"
    );
    // Nested sessions are decoded but not implemented.
    let mut session = crate::runtime::compute_session::ComputeSession { control_depth: 0 };
    let st2 = execute_dispatch_nested(&mut state, &mut host, 1, &acc, &cmd, &mut session);
    assert_eq!(
        st2,
        ComputeStatus::Unsupported("compute_nested_session_unimplemented")
    );
}

#[test]
fn dispatch_missing_pipeline_not_backend_unavailable() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));
    let acc = ComputeAccum::default();
    let mut cmd = ComputeCommand::default();
    cmd.kind = Kind::DispatchThreadgroups;
    cmd.grid = compute::Size3 { x: 1, y: 1, z: 1 };
    cmd.threads_per_threadgroup = compute::Size3 { x: 1, y: 1, z: 1 };
    let st = execute_dispatch(&mut state, &mut host, 1, &acc, &cmd, &[]);
    assert_eq!(
        st,
        ComputeStatus::MissingPipeline("compute_vk_pipeline_ref_zero")
    );
}

#[test]
fn compute_pipeline_state_is_immutable_until_delete_and_reuse() {
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    let blob_gva = 0x180u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], blob_gva, &[1, 2, 3, 4]);
    let mut function = [0u8; 32];
    st64(&mut function[0..], blob_gva);
    st32(&mut function[8..], 4);
    let function_gva = 0x100u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], function_gva, &function);
    let function_off = list_object_entry_offset(5, 32).unwrap();
    let mut function_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut function_entry[0..],
        (OBJECT_TYPE_FUNCTION as u32) | (32u32 << 8),
    );
    function_entry[4..12].copy_from_slice(&function_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], function_off, &function_entry);
    let second_function_off = list_object_entry_offset(9, 32).unwrap();
    write_task_gva_arm64e(
        &mut host,
        &state.tasks[1],
        second_function_off,
        &function_entry,
    );

    let mut descriptor = vec![0u8; 32];
    st32(
        &mut descriptor[0..],
        SERIALIZER_RESOURCE_OBJECT_COMPUTE_PIPELINE,
    );
    st32(&mut descriptor[4..], 32);
    descriptor[SERIALIZER_RESOURCE_FIRST_TLVS] = 1;
    descriptor[SERIALIZER_RESOURCE_FIRST_TLVS + 1] = PIPELINE_TAG_KERNEL_FUNC;
    descriptor[SERIALIZER_RESOURCE_FIRST_TLVS + 2] = 4;
    st32(&mut descriptor[SERIALIZER_RESOURCE_FIRST_TLVS + 3..], 5);
    let descriptor_gva = 0x140u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], descriptor_gva, &descriptor);
    let off = list_object_entry_offset(6, 32).unwrap();
    let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut entry[0..],
        (OBJECT_TYPE_SERIALIZER_RESOURCE as u32) | (32u32 << 8),
    );
    entry[4..12].copy_from_slice(&descriptor_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &entry);

    let first = load_compute_pipeline(&state, &host, 1, 6).expect("first pipeline");
    let first_identity = state
        .task_objects
        .compute_pipelines
        .identity(1, reims_vgpu_protocol::SerializerRef::new(6))
        .expect("first identity");
    let first_function_identity = state
        .task_objects
        .functions
        .identity(1, reims_vgpu_protocol::SerializerRef::new(5))
        .expect("first function identity");
    assert_eq!(first.kernel_func_ref, 5);
    assert_eq!(&*first.kernel_mtlb, &[1, 2, 3, 4]);

    assert!(state
        .task_objects
        .functions
        .delete(1, reims_vgpu_protocol::SerializerRef::new(5)));
    write_task_gva_arm64e(&mut host, &state.tasks[1], blob_gva, &[9, 8, 7, 6]);
    let replacement_function = crate::runtime::mtlb::load_mtlb(
        &state,
        &host,
        1,
        5,
        crate::runtime::mtlb::AirLoadRail::Compute,
    )
    .expect("replacement function");
    let replacement_function_identity = state
        .task_objects
        .functions
        .identity(1, reims_vgpu_protocol::SerializerRef::new(5))
        .expect("replacement function identity");
    assert_eq!(&*replacement_function, &[9, 8, 7, 6]);
    assert_eq!(
        first_function_identity.index(),
        replacement_function_identity.index()
    );
    assert_ne!(
        first_function_identity.generation(),
        replacement_function_identity.generation()
    );

    st32(&mut descriptor[SERIALIZER_RESOURCE_FIRST_TLVS + 3..], 9);
    write_task_gva_arm64e(&mut host, &state.tasks[1], descriptor_gva, &descriptor);
    let retained = load_compute_pipeline(&state, &host, 1, 6).expect("retained pipeline");
    assert!(std::sync::Arc::ptr_eq(&first, &retained));
    assert_eq!(retained.kernel_func_ref, 5);
    assert_eq!(
        &*retained.kernel_mtlb,
        &[1, 2, 3, 4],
        "a live pipeline retains the function payload it was constructed from"
    );

    assert!(state
        .task_objects
        .compute_pipelines
        .delete(1, reims_vgpu_protocol::SerializerRef::new(6)));
    let replacement = load_compute_pipeline(&state, &host, 1, 6).expect("replacement pipeline");
    let replacement_identity = state
        .task_objects
        .compute_pipelines
        .identity(1, reims_vgpu_protocol::SerializerRef::new(6))
        .expect("replacement identity");
    assert_eq!(replacement.kernel_func_ref, 9);
    assert_eq!(&*replacement.kernel_mtlb, &[9, 8, 7, 6]);
    assert_eq!(first_identity.index(), replacement_identity.index());
    assert_ne!(
        first_identity.generation(),
        replacement_identity.generation()
    );
}

/// One condition, one line, one refusal slug — from every rail that asks.
///
/// Three sites used to ask whether a staged texture's format has a storage
/// selector, and each answered with its own event name, its own `reason=` and
/// its own `Unsupported(..)` string, dropping a different field apiece. A grep
/// for any one of them found a third of the losses and could not say which rail
/// it came from. The assertions below are on the line's shape, so a fourth
/// spelling has to fail here before it can reach the log.
#[test]
fn dispatch_buffer_kernel_mul3add1() {
    use std::path::PathBuf;
    let mtlb_paths = [
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/compute_mul3add1.mtlb"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/compute_mul3add1.mtlb"),
    ];
    let mtlb = mtlb_paths
        .iter()
        .find_map(|p| std::fs::read(p).ok())
        .expect("compute_mul3add1.mtlb fixture");

    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    let blob_gva = 4u64 << RESOURCE_PAGE_SHIFT;
    write_task_gva_arm64e(&mut host, &state.tasks[1], blob_gva, &mtlb);
    let mut fdesc = vec![0u8; 32];
    st64(&mut fdesc[0..], blob_gva);
    st32(&mut fdesc[8..], mtlb.len() as u32);
    let fdesc_gva = 0x100u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], fdesc_gva, &fdesc);
    {
        let off = list_object_entry_offset(5, 32).unwrap();
        let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_FUNCTION as u32) | (32u32 << 8);
        st32(&mut le[0..], packed);
        le[4..12].copy_from_slice(&fdesc_gva.to_le_bytes());
        write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);
    }

    let mut pdesc = vec![0u8; 32];
    st32(&mut pdesc[0..], SERIALIZER_RESOURCE_OBJECT_COMPUTE_PIPELINE);
    st32(&mut pdesc[4..], 32); // declared descriptor length
    pdesc[SERIALIZER_RESOURCE_FIRST_TLVS] = 1;
    pdesc[SERIALIZER_RESOURCE_FIRST_TLVS + 1] = PIPELINE_TAG_KERNEL_FUNC;
    pdesc[SERIALIZER_RESOURCE_FIRST_TLVS + 2] = 4;
    st32(&mut pdesc[SERIALIZER_RESOURCE_FIRST_TLVS + 3..], 5);
    let pdesc_gva = 0x140u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], pdesc_gva, &pdesc);
    {
        let off = list_object_entry_offset(6, 32).unwrap();
        let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_SERIALIZER_RESOURCE as u32) | (32u32 << 8);
        st32(&mut le[0..], packed);
        le[4..12].copy_from_slice(&pdesc_gva.to_le_bytes());
        write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);
    }

    let data = [1u32, 2, 3, 4];
    let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf_gva = 5u64 << RESOURCE_PAGE_SHIFT;
    write_task_gva_arm64e(&mut host, &state.tasks[1], buf_gva, &data_bytes);
    let mut bdesc = vec![0u8; 16];
    st64(&mut bdesc[0..], 16);
    st32(&mut bdesc[8..], 5);
    let bdesc_gva = 0x180u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], bdesc_gva, &bdesc);
    {
        let off = list_object_entry_offset(7, 32).unwrap();
        let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_BUFFER as u32) | (16u32 << 8);
        st32(&mut le[0..], packed);
        le[4..12].copy_from_slice(&bdesc_gva.to_le_bytes());
        write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);
    }
    let resource = state.task_objects.resources.register(
        1,
        7,
        std::sync::Arc::new(crate::model::TaskResource::new(
            crate::runtime::decode::resource::ListObjectEntry::new(
                reims_vgpu_protocol::ObjectKind::Buffer,
                0,
                0,
            ),
            std::sync::Arc::from(bdesc.clone()),
        )),
    );
    let resource_id = resource.semantic_id().expect("canonical buffer identity");
    let before = state
        .task_objects
        .resources
        .resource_node(resource_id)
        .expect("canonical buffer")
        .content
        .snapshot();

    let mut acc = ComputeAccum::default();
    acc.set_pipeline(6);
    let bindings = vec![
        BufferBinding {
            ref_: 7,
            offset: 0,
            attribute_stride: 0,
            has_attribute_stride: false,
        },
        BufferBinding {
            // The kernel declares no buffer 1. Reflection must discard this
            // extra encoder bind before object resolution; the nonexistent ref
            // makes a pre-reflection staging pass fail as MissingBuffer.
            ref_: 31,
            offset: 0,
            attribute_stride: 0,
            has_attribute_stride: false,
        },
    ];
    acc.bind_buffers(0, &bindings);

    let mut cmd = ComputeCommand::default();
    cmd.kind = Kind::DispatchThreadgroups;
    cmd.grid = compute::Size3 { x: 1, y: 1, z: 1 };
    cmd.threads_per_threadgroup = compute::Size3 { x: 4, y: 1, z: 1 };
    let st = execute_dispatch(&mut state, &mut host, 1, &acc, &cmd, &[]);
    assert!(
        matches!(
            st,
            ComputeStatus::Ok
                | ComputeStatus::BackendFailed(_)
                | ComputeStatus::BadGrid(_)
                | ComputeStatus::Unsupported(_)
        ),
        "unexpected {st:?}"
    );
    if st == ComputeStatus::Ok {
        let mut back = [0u8; 16];
        assert!(gva_mem::read_task_gva(
            &host,
            &state.tasks[1],
            buf_gva,
            &mut back,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
        let out: Vec<u32> = back
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(out, vec![4, 7, 10, 13]);
        let after = state
            .task_objects
            .resources
            .resource_node(resource_id)
            .expect("canonical buffer")
            .content
            .snapshot();
        assert!(after.current > before.current);
        assert!(after.current_in_gpu());
        assert!(after.current_in_guest());
    }
}

#[test]
fn dispatch_missing_texture_fails() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));
    // Pipeline without function still fails earlier; bind texture only.
    let mut acc = ComputeAccum::default();
    acc.set_pipeline(1);
    acc.bind_textures(0, &[RefBinding { ref_: 99 }]);
    let mut cmd = ComputeCommand::default();
    cmd.kind = Kind::DispatchThreadgroups;
    cmd.grid = compute::Size3 { x: 1, y: 1, z: 1 };
    cmd.threads_per_threadgroup = compute::Size3 { x: 1, y: 1, z: 1 };
    // Missing pipeline object → MissingPipeline before texture stage.
    let st = execute_dispatch(&mut state, &mut host, 1, &acc, &cmd, &[]);
    assert!(matches!(
        st,
        ComputeStatus::MissingPipeline(_)
            | ComputeStatus::MissingTexture(_)
            | ComputeStatus::BackendFailed(_)
            | ComputeStatus::Unsupported(_)
    ));
}

/// Live CI wallpaper: IOSurface plane view RefTexture → surface backing surface_id must stage via
/// ensure_surface + mapping (same order as the `runtime::draw` sample). Without
/// ensure, stage fell through to type-2/3 with the IOSurface plane view ref → always
/// MissingTexture (`compute_stage_tex … ot=5`).
#[test]
fn stage_texture_iosurface_plane_view_ref_resolves_surface_mapping() {
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use reims_vgpu_paging::geometry::{
        MAPPER_PAGE_ENTRY_PFN_SHIFT as PAGE_ENTRY_PFN_SHIFT,
        MAPPER_PAGE_ENTRY_VALID as PAGE_ENTRY_VALID,
    };

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    let sid = 3u32;
    let iosurface_plane_view_ref = 10u32;
    // Pre-mapped surface backing surface (CI storage target) with one valid page.
    let pfn = 0x20u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0x5a);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    assert!(state.map_surface(sid));
    {
        let m = state.surfaces.mappings.get_mut(&sid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![entry];
    }
    assert!(state.set_mapping_geom(sid, 4, 4, MTL_FORMAT_BGRA8_UNORM));

    // Object-list: IOSurface plane view at ref 10 → surface_id=3 (mapping already seeded).
    let desc_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E; // data pfn base 4 + 2
    let iosurface_plane_view_desc = IOSurfacePlaneViewBuilder::new(sid, 0, 0, 0).with_len(16);
    let iosurface_plane_view_desc = iosurface_plane_view_desc.bytes();
    write_task_gva_arm64e(
        &mut host,
        &state.tasks[1],
        desc_gva,
        iosurface_plane_view_desc,
    );
    let off = list_object_entry_offset(iosurface_plane_view_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((16u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    crate::runtime::guest_ram::latch_import_limits(1 << PAGE_SHIFT_ARM64E, 1 << 30, 1 << 30);
    let staged = stage_texture_raw(
        &mut state,
        &mut host,
        1,
        iosurface_plane_view_ref,
        32,
        ComputeTextureStage::Storage2d,
    )
    .expect("IOSurface plane view→surface stage must succeed after ensure");
    crate::runtime::guest_ram::forget_import_limits();
    assert_eq!((staged.width, staged.height), (4, 4));
    assert!(match &staged.input {
        VulkanTextureInput::GuestPages(source) => source.total_len == 4 * 4 * 4,
        VulkanTextureInput::GuestImage(_) => false,
        VulkanTextureInput::HostBytes(bytes) => bytes.len() == 4 * 4 * 4,
        VulkanTextureInput::Resident(_) | VulkanTextureInput::TargetResident(_) => false,
    });
    assert!(staged.bytes.is_empty(), "the input has one typed owner");
    assert!(matches!(
        staged.writeback,
        TextureWriteback::IOSurface { mapping_id: 3, .. }
    ));
}

/// A IOSurface plane view record is the exact Metal view, even when its single-plane
/// backing already has valid base geometry. Live pipe 5 exposes each row
/// of a 1920-wide BGRA8 surface as a 480-wide RGBA32Uint view so one
/// `uint4` image write stores four packed BGRA pixels.
#[test]
fn stage_texture_iosurface_plane_view_record_reshapes_stageable_single_plane_surface() {
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
    use reims_vgpu_core::pixel_format::{
        MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_R32_UINT, MTL_FORMAT_RGBA32_UINT,
    };
    use reims_vgpu_paging::geometry::{
        MAPPER_PAGE_ENTRY_PFN_SHIFT as PAGE_ENTRY_PFN_SHIFT,
        MAPPER_PAGE_ENTRY_VALID as PAGE_ENTRY_VALID,
    };

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    let sid = 3u32;
    let iosurface_plane_view_ref = 10u32;
    let pfn = 0x20u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0x5a);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    assert!(state.map_surface(sid));
    {
        let m = state.surfaces.mappings.get_mut(&sid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![entry];
    }
    assert!(state.set_mapping_geom(sid, 4, 4, MTL_FORMAT_BGRA8_UNORM));

    // Same 16 bytes per logical row: 4 BGRA8 texels = one RGBA32Uint texel.
    let desc_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E;
    let iosurface_plane_view_desc = IOSurfacePlaneViewBuilder::new(sid, 0, 10, 0x42)
        .unknown(0x02)
        .geometry(MTL_FORMAT_RGBA32_UINT, 1, 4, 1)
        .trailer([1, 0, 1, 0, 1, 0, 0x10, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let iosurface_plane_view_desc = iosurface_plane_view_desc.bytes();
    write_task_gva_arm64e(
        &mut host,
        &state.tasks[1],
        desc_gva,
        iosurface_plane_view_desc,
    );
    let off = list_object_entry_offset(iosurface_plane_view_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed =
        (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((iosurface_plane_view_desc.len() as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    let staged = stage_texture_raw(
        &mut state,
        &mut host,
        1,
        iosurface_plane_view_ref,
        33,
        ComputeTextureStage::Storage2d,
    )
    .expect("serialized IOSurface plane view view must override base surface geometry");
    assert_eq!((staged.width, staged.height), (1, 4));
    assert_eq!(
        staged.storage_format,
        Some(reims_vgpu_protocol::StorageImageFormat::Rgba32Uint)
    );
    assert!(match &staged.input {
        VulkanTextureInput::GuestPages(source) => source.total_len == 4 * 16,
        VulkanTextureInput::GuestImage(_) => false,
        VulkanTextureInput::HostBytes(bytes) => bytes.len() == 4 * 16,
        VulkanTextureInput::Resident(_) | VulkanTextureInput::TargetResident(_) => false,
    });
    assert!(
        staged.bytes.is_empty(),
        "guest pages must not be materialized"
    );
    match staged.writeback {
        TextureWriteback::IOSurface {
            mapping_id,
            surface_bpr,
            width,
            height,
            format,
            ..
        } => {
            assert_eq!(mapping_id, sid);
            assert_eq!(surface_bpr, 128);
            assert_eq!((width, height), (1, 4));
            assert_eq!(pixel_format::bytes_per_pixel(format), Some(16));
        }
        _ => panic!("expected IOSurface writeback through the texture view"),
    }

    // A sampled R32Uint view retains its exact format/geometry. R32Uint is
    // now a storage-capable format (its selector maps to the R32ui storage
    // path), so `storage_format` is populated — but it is inert here: this
    // view is staged sampled (`is_storage=false`, binding 32), and the
    // selector is only consulted on the storage-bind path.
    let reshaped = IOSurfacePlaneViewBuilder::new(sid, 0, 10, 0x42)
        .unknown(0x02)
        .geometry(MTL_FORMAT_R32_UINT, 4, 4, 1)
        .trailer([1, 0, 1, 0, 1, 0, 0x10, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    // Construction bytes are immutable for a published resource lifetime.
    // Retire that lifetime before publishing a replacement descriptor at the
    // same object-table reference.
    assert!(state.delete_object(1, iosurface_plane_view_ref));
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, reshaped.bytes());
    let sampled = stage_texture_raw(
        &mut state,
        &mut host,
        1,
        iosurface_plane_view_ref,
        32,
        ComputeTextureStage::Sampled2d,
    )
    .expect("sample-only R32Uint view must stage from the same IOSurface bytes");
    assert_eq!((sampled.width, sampled.height), (4, 4));
    assert_eq!(sampled.pixel_format, MTL_FORMAT_R32_UINT);
    assert_eq!(
        sampled.storage_format,
        Some(reims_vgpu_protocol::StorageImageFormat::R32Uint)
    );
    assert!(match &sampled.input {
        VulkanTextureInput::GuestPages(source) => source.total_len == 4 * 4 * 4,
        VulkanTextureInput::GuestImage(_) => false,
        VulkanTextureInput::HostBytes(bytes) => bytes.len() == 4 * 4 * 4,
        VulkanTextureInput::Resident(_) | VulkanTextureInput::TargetResident(_) => false,
    });
    assert!(sampled.bytes.is_empty(), "the input has one typed owner");
    assert!(matches!(sampled.writeback, TextureWriteback::None));
}

/// Biplanar surface (device_desc plane_count=2) + IOSurface plane view args plane record:
/// stage the named plane view (R8 Y) from the plane offset — live class
/// `compute_dispatch st=Unsupported` / `iosurface_texture_fail reason=multiplane`
/// (wallpaper '420f', journal 2026-07-14 compute census).
#[test]
fn stage_texture_iosurface_plane_view_record_stages_biplanar_y_plane() {
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
    use reims_vgpu_core::endian::{st16, st64};
    use reims_vgpu_core::pixel_format::MTL_FORMAT_R8_UNORM;
    use reims_vgpu_protocol::{
        DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_LEN, DEVICE_DESC_PLANES, DEVICE_DESC_PLANE_COUNT,
        DEVICE_PLANE_BPE, DEVICE_PLANE_BPR, DEVICE_PLANE_DESC_LEN, DEVICE_PLANE_DIMS,
        DEVICE_PLANE_OFFSET, DEVICE_PLANE_SIZE,
    };

    let pack_dims = |w: u64, h: u64| ((w & 0xffffff) << 8) | ((h & 0xffffff) << 40);

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    let sid = 3u32;
    let iosurface_plane_view_ref = 10u32;
    let pfn = 0x20u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0x77);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;

    // Device surface: 2 planes — Y 16×8 R8 bpr=64 off=0, UV 8×4 RG8 off=512.
    let mut dev = vec![0u8; DEVICE_DESC_LEN];
    st32(&mut dev[DEVICE_DESC_ALLOC_SIZE..], 0x4000);
    dev[DEVICE_DESC_PLANE_COUNT] = 2;
    let p0 = DEVICE_DESC_PLANES;
    st32(&mut dev[p0 + DEVICE_PLANE_OFFSET..], 0);
    st32(&mut dev[p0 + DEVICE_PLANE_SIZE..], 512);
    st64(&mut dev[p0 + DEVICE_PLANE_DIMS..], pack_dims(16, 8));
    st32(&mut dev[p0 + DEVICE_PLANE_BPR..], 64);
    st16(&mut dev[p0 + DEVICE_PLANE_BPE..], 1);
    let p1 = DEVICE_DESC_PLANES + DEVICE_PLANE_DESC_LEN;
    st32(&mut dev[p1 + DEVICE_PLANE_OFFSET..], 512);
    st32(&mut dev[p1 + DEVICE_PLANE_SIZE..], 256);
    st64(&mut dev[p1 + DEVICE_PLANE_DIMS..], pack_dims(8, 4));
    st32(&mut dev[p1 + DEVICE_PLANE_BPR..], 64);
    st16(&mut dev[p1 + DEVICE_PLANE_BPE..], 2);

    assert!(state.map_surface(sid));
    {
        let m = state.surfaces.mappings.get_mut(&sid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![entry];
        m.publish_device_desc_for_test(&dev);
        m.publish_geometry_for_test(16, 8, 0); // surface-level FourCC has no single MTL format
    }
    assert!(objects::mapping_is_multiplanar(
        state.surfaces.mappings.get(&sid).unwrap()
    ));

    // IOSurface plane view descriptor: sid + args blob carrying the R8 16×8 plane record.
    let desc_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E;
    // tag, unk, fmt=R8
    let iosurface_plane_view_desc = IOSurfacePlaneViewBuilder::new(sid, 0, 10, 0x42)
        .unknown(0x01)
        .geometry(0x0a, 16, 8, 1);
    let iosurface_plane_view_desc = iosurface_plane_view_desc.bytes();
    write_task_gva_arm64e(
        &mut host,
        &state.tasks[1],
        desc_gva,
        iosurface_plane_view_desc,
    );
    let off = list_object_entry_offset(iosurface_plane_view_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed =
        (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((iosurface_plane_view_desc.len() as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    let staged = stage_texture_raw(
        &mut state,
        &mut host,
        1,
        iosurface_plane_view_ref,
        32,
        ComputeTextureStage::Storage2d,
    )
    .expect("plane record must stage the Y plane of a biplanar surface");
    assert_eq!((staged.width, staged.height), (16, 8));
    assert_eq!(
        staged.storage_format,
        Some(reims_vgpu_protocol::StorageImageFormat::R8Unorm)
    );
    assert!(match &staged.input {
        VulkanTextureInput::GuestPages(source) => source.total_len == 16 * 8,
        VulkanTextureInput::GuestImage(_) => false,
        VulkanTextureInput::HostBytes(bytes) => bytes.len() == 16 * 8,
        VulkanTextureInput::Resident(_) | VulkanTextureInput::TargetResident(_) => false,
    });
    assert!(
        staged.bytes.is_empty(),
        "guest pages must not be materialized"
    );
    match staged.writeback {
        TextureWriteback::IOSurface {
            mapping_id,
            surface_offset,
            surface_bpr,
            ..
        } => {
            assert_eq!(mapping_id, sid);
            assert_eq!(surface_offset, 0);
            assert_eq!(surface_bpr, 64);
        }
        _ => panic!("expected IOSurface writeback"),
    }
    let sampled = stage_texture_raw(
        &mut state,
        &mut host,
        1,
        iosurface_plane_view_ref,
        32,
        ComputeTextureStage::Sampled2d,
    )
    .expect("sampled IOSurface plane view plane must stage without writeback");
    assert!(!sampled.is_storage);
    assert!(matches!(sampled.writeback, TextureWriteback::None));
    let _ = MTL_FORMAT_R8_UNORM;
}

/// Biplanar surface **without** a plane record still fails closed
/// (no BGRA invent over multi-plane bytes).
#[test]
fn stage_texture_iosurface_plane_view_multiplanar_without_record_fails_closed() {
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
    use reims_vgpu_protocol::{DEVICE_DESC_LEN, DEVICE_DESC_PLANE_COUNT};

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    let sid = 3u32;
    let iosurface_plane_view_ref = 10u32;
    let pfn = 0x20u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0x5a);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    let mut dev = vec![0u8; DEVICE_DESC_LEN];
    dev[DEVICE_DESC_PLANE_COUNT] = 2;
    assert!(state.map_surface(sid));
    {
        let m = state.surfaces.mappings.get_mut(&sid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![entry];
        m.publish_device_desc_for_test(&dev);
        m.publish_geometry_for_test(16, 8, 0);
    }

    // IOSurface plane view descriptor with sid but NO args record.
    let desc_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E;
    let iosurface_plane_view_desc = IOSurfacePlaneViewBuilder::new(sid, 0, 0, 0).with_len(8);
    let iosurface_plane_view_desc = iosurface_plane_view_desc.bytes();
    write_task_gva_arm64e(
        &mut host,
        &state.tasks[1],
        desc_gva,
        iosurface_plane_view_desc,
    );
    let off = list_object_entry_offset(iosurface_plane_view_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed =
        (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((iosurface_plane_view_desc.len() as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    match stage_texture_raw(
        &mut state,
        &mut host,
        1,
        iosurface_plane_view_ref,
        32,
        ComputeTextureStage::Storage2d,
    ) {
        Err(ComputeStatus::Unsupported(_)) => {}
        Err(other) => panic!("expected Unsupported, got {other:?}"),
        Ok(_) => panic!("multiplanar without plane record must fail closed"),
    }
}

/// A linear texture (ot=2) whose numeric ref equals an existing surface mid
/// must NOT be reinterpreted as that mapping — it resolves through its own
/// (here invalid) descriptor and fails linear, never staging the collided
/// surface's pixels. Live class: `ref=N ot=2` MissingTexture where mid N is
/// the biplanar wallpaper.
#[test]
fn stage_texture_linear_ref_does_not_collide_with_surface_mid() {
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use reims_vgpu_paging::geometry::{
        MAPPER_PAGE_ENTRY_PFN_SHIFT as PAGE_ENTRY_PFN_SHIFT,
        MAPPER_PAGE_ENTRY_VALID as PAGE_ENTRY_VALID,
    };

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    // Surface mid 7 exists with full geometry + a mapped page.
    let colliding_mid = 7u32;
    let pfn = 0x20u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0x33);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    assert!(state.map_surface(colliding_mid));
    {
        let m = state.surfaces.mappings.get_mut(&colliding_mid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![entry];
    }
    assert!(state.set_mapping_geom(colliding_mid, 4, 4, MTL_FORMAT_BGRA8_UNORM));

    // Object-list ref 7 = a TEXTURE (ot=2) object with a non-decodable desc.
    let desc_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E;
    let bogus_desc = vec![0u8; 16];
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &bogus_desc);
    let off = list_object_entry_offset(colliding_mid, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (OBJECT_TYPE_TEXTURE as u32) | ((bogus_desc.len() as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    // Must fail linear (bogus desc), NOT succeed against surface mid 7.
    if let Ok(s) = stage_texture_raw(
        &mut state,
        &mut host,
        1,
        colliding_mid,
        32,
        ComputeTextureStage::Storage2d,
    ) {
        panic!(
            "linear ref must not stage collided surface mid ({}x{})",
            s.width, s.height
        )
    }
}

#[test]
fn equal_heap_placements_stage_one_shared_residency_identity() {
    use crate::runtime::decode::resource::{
        list_object_entry_offset, HEAP_TEXTURE_DESCRIPTOR, HEAP_TEXTURE_HEAP_REF, HEAP_TEXTURE_LEN,
        HEAP_TEXTURE_OFFSET, HEAP_TEXTURE_OPCODE, HEAP_TEXTURE_USE_OFFSET, OBJECT_LIST_ENTRY_LEN,
        OBJECT_TYPE_TEXTURE_VIEW,
    };
    use reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA32_FLOAT;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    let texture_ref = 20u32;
    let alias_ref = 21u32;
    let heap_ref = 19u32;
    let desc_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E;
    let mut desc = vec![0u8; HEAP_TEXTURE_LEN];
    st32(&mut desc[0..], HEAP_TEXTURE_OPCODE);
    st32(&mut desc[4..], HEAP_TEXTURE_LEN as u32);
    st32(&mut desc[8..], texture_ref);
    st32(&mut desc[HEAP_TEXTURE_HEAP_REF..], heap_ref);
    // PGSerializedTextureDescriptor: 2D, GPU-optimized, usage=3,
    // RGBA32Float, 180x135x1, one mip/sample/array element, private.
    st32(
        &mut desc[HEAP_TEXTURE_DESCRIPTOR..],
        2 | (1 << 6) | (3 << 8) | ((MTL_FORMAT_RGBA32_FLOAT as u32) << 16),
    );
    st32(&mut desc[HEAP_TEXTURE_DESCRIPTOR + 4..], 180);
    st32(&mut desc[HEAP_TEXTURE_DESCRIPTOR + 8..], 135);
    st32(&mut desc[HEAP_TEXTURE_DESCRIPTOR + 12..], 1);
    desc[HEAP_TEXTURE_DESCRIPTOR + 16..HEAP_TEXTURE_DESCRIPTOR + 18]
        .copy_from_slice(&1u16.to_le_bytes());
    desc[HEAP_TEXTURE_DESCRIPTOR + 18..HEAP_TEXTURE_DESCRIPTOR + 20]
        .copy_from_slice(&1u16.to_le_bytes());
    desc[HEAP_TEXTURE_DESCRIPTOR + 20..HEAP_TEXTURE_DESCRIPTOR + 22]
        .copy_from_slice(&1u16.to_le_bytes());
    desc[HEAP_TEXTURE_DESCRIPTOR + 22..HEAP_TEXTURE_DESCRIPTOR + 24]
        .copy_from_slice(&0x20u16.to_le_bytes());
    st32(&mut desc[HEAP_TEXTURE_USE_OFFSET..], 1);
    st64(&mut desc[HEAP_TEXTURE_OFFSET..], 0);
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &desc);

    let alias_desc_gva = (4u64 + 3) << PAGE_SHIFT_ARM64E;
    let mut alias_desc = desc.clone();
    st32(&mut alias_desc[8..], alias_ref);
    write_task_gva_arm64e(&mut host, &state.tasks[1], alias_desc_gva, &alias_desc);

    let entry_offset = list_object_entry_offset(texture_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (OBJECT_TYPE_TEXTURE_VIEW as u32) | ((HEAP_TEXTURE_LEN as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], entry_offset, &list_entry);
    let alias_entry_offset = list_object_entry_offset(alias_ref, 32).unwrap();
    list_entry[4..12].copy_from_slice(&alias_desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], alias_entry_offset, &list_entry);

    let staged = stage_texture_raw(
        &mut state,
        &mut host,
        1,
        texture_ref,
        33,
        ComputeTextureStage::Storage2d,
    )
    .expect("live opcode-0x15 heap texture must stage");
    assert_eq!((staged.width, staged.height), (180, 135));
    assert_eq!(staged.pixel_format, MTL_FORMAT_RGBA32_FLOAT);
    assert_eq!(
        staged.storage_format,
        Some(reims_vgpu_protocol::StorageImageFormat::Rgba32Float)
    );
    assert!(matches!(
        &staged.input,
        VulkanTextureInput::HostBytes(bytes) if bytes.len() == 180 * 135 * 16
    ));
    assert!(matches!(staged.writeback, TextureWriteback::None));
    let residency = staged.residency.expect("heap texture needs GPU residency");
    assert!(residency.key.is_heap());
    assert!(!residency.key.is_linear());
    assert_eq!(
        residency.key.origin,
        state
            .task_objects
            .resources
            .heap_storage_origin(1, texture_ref)
            .unwrap()
    );
    assert_eq!(residency.seed_generation, 0);

    let alias = stage_texture_raw(
        &mut state,
        &mut host,
        1,
        alias_ref,
        34,
        ComputeTextureStage::Storage2d,
    )
    .expect("an equal explicit placement must stage");
    let alias_residency = alias.residency.expect("heap alias needs GPU residency");
    assert_eq!(
        residency.key, alias_residency.key,
        "equal heap generation/range and image view must acquire one resident"
    );
}

/// UnmapMemory removes the guest page-table alias, not the discrete
/// type-2/3 texture body. Compute writeback must retain raw output, mirror
/// normalized color for render sampling, and complete without attempting
/// a fail-closed write into freed guest pages.
#[test]
fn linear_writeback_retains_cache_when_guest_gva_is_unmapped() {
    use crate::runtime::surface_cache;
    use reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let task_id = 6u32;
    let texture_ref = 11u32;
    let gva = 0x101000u64;
    let rgba = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let staged = StagedTexture {
        resource_ref: texture_ref,
        binding: 32,
        array_element: 0,
        descriptor_count: 1,

        pixel_format: MTL_FORMAT_RGBA8_UNORM,
        storage_format: Some(reims_vgpu_protocol::StorageImageFormat::Rgba8Unorm),
        view_swizzle: reims_vgpu_protocol::SwizzlePlan::default(),
        width: 2,
        height: 2,
        multisampled: false,
        bytes: rgba.clone(),
        is_storage: true,
        residency: None,
        input: VulkanTextureInput::HostBytes(Vec::new()),
        writeback: TextureWriteback::Linear {
            pages: crate::runtime::draw::StoreTargetPages::empty(),
            texture_ref,
            gva,
            pixel_format: MTL_FORMAT_RGBA8_UNORM,
            row_stride: 8,
            width: 2,
            height: 2,
            bpp: 4,
        },
    };

    assert_eq!(
        writeback_texture(&mut state, &mut host, task_id, &staged),
        Ok(GuestMaterialization::HostOnly)
    );
    assert_eq!(
        surface_cache::get_linear_texture(
            &state,
            &surface_cache::LinearWindow {
                task_id,
                texture_ref,
                gva,
                pixel_format: MTL_FORMAT_RGBA8_UNORM,
                width: 2,
                height: 2,
                row_stride: 8,
            },
        ),
        Some(rgba.as_slice())
    );
    assert_eq!(
        &surface_cache::get_texture(&state, task_id, texture_ref, 2, 2).unwrap()[..4],
        &[3, 2, 1, 4],
        "RGBA compute output mirrors into the BGRA render-sample cache"
    );
}

/// No product MiB budget on compute staging — guest size is authoritative.
/// Full-screen wide-gamut (live SkyLight 1928×1920 RGBA16F ≈ 28.2 MiB) must
/// be host-addressable (usize), not rejected by an arbitrary cap.
#[test]
fn compute_stage_admits_full_screen_wide_gamut_without_cap() {
    use crate::runtime::draw::host_alloc_len;
    use reims_vgpu_core::pixel_format::{bytes_per_pixel, MTL_FORMAT_RGBA16_FLOAT};
    let bpp = bytes_per_pixel(MTL_FORMAT_RGBA16_FLOAT).expect("rgba16f bpp") as u64;
    let need = 1928u64 * 1920 * bpp;
    assert!(
        host_alloc_len(need).is_some(),
        "full-screen RGBA16Float ({need} bytes) must be host-addressable"
    );
}

/// IOSurface plane view surface id must not be re-resolved through this task's object
/// list: slot `sid` can be a different texture-ref object (id collision).
/// Live class: ensure=1 then MissingTexture when resolve_iosurface_texture_ref(task,sid)
/// returned the wrong mapping.
#[test]
fn stage_texture_iosurface_plane_view_ignores_task_object_list_slot_collision() {
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use reims_vgpu_paging::geometry::{
        MAPPER_PAGE_ENTRY_PFN_SHIFT as PAGE_ENTRY_PFN_SHIFT,
        MAPPER_PAGE_ENTRY_VALID as PAGE_ENTRY_VALID,
    };

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    let sid = 3u32;
    let iosurface_plane_view_ref = 10u32;
    let pfn = 0x21u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0xa5);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    assert!(state.map_surface(sid));
    {
        let m = state.surfaces.mappings.get_mut(&sid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![entry];
    }
    assert!(state.set_mapping_geom(sid, 4, 4, MTL_FORMAT_BGRA8_UNORM));

    // Poison: object-list slot `sid` is IOSurface texture with mapping_id=99 (not mapped).
    // Pre-fix path would resolve_iosurface_texture_ref(task, sid) → 99 → MissingTexture.
    let poison_desc_gva = (4u64 + 1) << PAGE_SHIFT_ARM64E;
    let mut iosurf = vec![0u8; 64];
    st32(&mut iosurf[0..], 99); // fake mapping_id
    write_task_gva_arm64e(&mut host, &state.tasks[1], poison_desc_gva, &iosurf);
    let off_sid = list_object_entry_offset(sid, 32).unwrap();
    let mut le_sid = [0u8; OBJECT_LIST_ENTRY_LEN];
    // IOSurface texture = OBJECT_TYPE_IOSURFACE
    let packed_iosurface = (OBJECT_TYPE_IOSURFACE as u32) | ((64u32) << 8);
    st32(&mut le_sid[0..], packed_iosurface);
    le_sid[4..12].copy_from_slice(&poison_desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off_sid, &le_sid);

    // IOSurface plane view at ref 10 → surface_id 3
    let desc_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E;
    let iosurface_plane_view_desc = IOSurfacePlaneViewBuilder::new(sid, 0, 0, 0).with_len(16);
    let iosurface_plane_view_desc = iosurface_plane_view_desc.bytes();
    write_task_gva_arm64e(
        &mut host,
        &state.tasks[1],
        desc_gva,
        iosurface_plane_view_desc,
    );
    let off = list_object_entry_offset(iosurface_plane_view_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((16u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    let staged = stage_texture_raw(
        &mut state,
        &mut host,
        1,
        iosurface_plane_view_ref,
        32,
        ComputeTextureStage::Storage2d,
    )
    .expect("IOSurface plane view must stage mapping sid, not poisoned IOSurface texture slot");
    assert_eq!((staged.width, staged.height), (4, 4));
    assert!(matches!(
        staged.writeback,
        TextureWriteback::IOSurface { mapping_id: 3, .. }
    ));
}

/// IOSurface plane view whose surface_id never maps must fail MissingTexture (not pretend
/// type-2/3 success).
#[test]
fn stage_texture_iosurface_plane_view_without_surface_is_missing() {
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    let iosurface_plane_view_ref = 11u32;
    let sid = 99u32; // no mapping
    let desc_gva = (4u64 + 3) << PAGE_SHIFT_ARM64E;
    let iosurface_plane_view_desc = IOSurfacePlaneViewBuilder::new(sid, 0, 0, 0).with_len(16);
    let iosurface_plane_view_desc = iosurface_plane_view_desc.bytes();
    write_task_gva_arm64e(
        &mut host,
        &state.tasks[1],
        desc_gva,
        iosurface_plane_view_desc,
    );
    let off = list_object_entry_offset(iosurface_plane_view_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((16u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    let st = stage_texture_raw(
        &mut state,
        &mut host,
        1,
        iosurface_plane_view_ref,
        32,
        ComputeTextureStage::Sampled2d,
    );
    assert!(matches!(st, Err(ComputeStatus::MissingTexture(_))));
}

#[test]
fn incomplete_compute_engine_call_fires_stall_proxy() {
    use reims_vgpu_core::ComputeRequest;
    use std::time::Duration;

    let pipe = 0xf000_0000 | (std::process::id() & 0x0fff_ffff);
    let req = ComputeRequest {
        program: reims_vgpu_core::PreparedShaderStage {
            id: reims_vgpu_protocol::PreparedShaderId::new(1),
            ..Default::default()
        },
        entry: "main".into(),
        dispatch: whole_workgroups([1, 1, 1], [1, 1, 1]),
        ..Default::default()
    };
    let done = spawn_compute_engine_stall_watchdog(pipe, &req, Duration::from_millis(10));
    std::thread::sleep(Duration::from_millis(40));
    done.store(true, std::sync::atomic::Ordering::Release);

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(log.lines().any(|line| {
        line.contains("compute_engine_stall reason=backend_call_unreturned")
            && line.contains(&format!("pipe={pipe}"))
    }));
    let base = format!("/tmp/reims-vgpu-compute-stall-pipe-{pipe}");
    let _ = std::fs::remove_file(format!("{base}.spv"));
    let _ = std::fs::remove_file(format!("{base}.txt"));
}

#[test]
fn storage_access_proxy_names_writeonly_seed_cost() {
    let pipe = 0xd000_0000 | (std::process::id() & 0x0fff_ffff);
    log_storage_image_access(pipe, 34, "write_only", 29_614_080);

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(log.lines().any(|line| {
        line.contains(&format!("compute_linux storage_access pipe={pipe}"))
            && line.contains("bind=34")
            && line.contains("access=write_only")
            && line.contains("seed=1")
            && line.contains("bytes=29614080")
    }));
}
fn setup_linear_task_x86(host: &mut FakeHost, state: &mut Device, pfns: &[u32]) {
    let page = 1u64 << PAGE_SHIFT_X86;
    let dir_gpa = 2u64 << PAGE_SHIFT_X86;
    let root_gpa = 3u64 << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, page as usize, 0);
    host.map_range(root_gpa, page as usize, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    host.write_gpa(dir_gpa, &d).unwrap();
    for (i, pfn) in pfns.iter().enumerate() {
        host.map_range((*pfn as u64) << PAGE_SHIFT_X86, page as usize, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, *pfn);
        host.write_gpa(root_gpa + (i as u64) * 4, &pte).unwrap();
    }
    state.define_task(1, page, 2);
}

/// A complete sampled mip allocation remains stageable when Vulkan cannot
/// retain a host-pointer import for it.
///
/// Compute used to ask only for the retained packed resource and refuse the
/// whole bind when that optional rail was unavailable. The task page plan is
/// the allocation contract on the copying rail too, so the fallback must carry
/// every byte and every physical page rather than narrowing the source to the
/// selected mip level.
#[test]
fn a_complete_mip_allocation_falls_back_to_the_task_page_plan() {
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    host.stable_map_pages = true;
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    setup_linear_task_x86(&mut host, &mut state, &[4]);

    let page = 1u64 << PAGE_SHIFT_X86;
    let allocation_gva = 0x800;
    let allocation_size = page - allocation_gva;
    let source =
        complete_mip_transfer_source(&state, &mut host, 1, allocation_gva, allocation_size, None)
            .expect("the copying rail resolves the complete mip allocation");

    assert_eq!(source.source_offset, 0);
    assert_eq!(source.total_len, allocation_size);
    assert_eq!(source.row_length_texels, 0);
    assert_eq!(
        source
            .physical_pages
            .as_ref()
            .expect("physical identity accompanies the transfer")
            .pages()
            .len(),
        1
    );
    assert_eq!(
        source.runs.iter().map(|run| run.len).sum::<u64>(),
        allocation_size
    );
}

#[test]
fn bulk_linear_read_destrides_span_with_one_view() {
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    setup_linear_task_x86(&mut host, &mut state, &[4, 5]);
    let (tight, stride, h) = (8usize, 16u64, 3u32);
    let gva = 0x40u64;
    // Rows y at gva + y*stride: payload y*3+1..; padding sentinel 0xEE.
    for y in 0..h as u64 {
        let mut row = vec![0xEEu8; stride as usize];
        for (i, b) in row[..tight].iter_mut().enumerate() {
            *b = (y as u8) * 3 + 1 + i as u8;
        }
        host.write_gpa((4u64 << PAGE_SHIFT_X86) + gva + y * stride, &row)
            .unwrap();
    }
    let mut bytes = vec![0u8; tight * h as usize];
    assert!(read_linear_texture_bulk(
        &mut state, &mut host, 1, gva, stride, tight, h, &mut bytes
    ));
    for y in 0..h as usize {
        for i in 0..tight {
            assert_eq!(
                bytes[y * tight + i],
                (y as u8) * 3 + 1 + i as u8,
                "y={y} i={i}"
            );
        }
    }
}

#[test]
fn bulk_linear_write_scatters_rows_and_preserves_padding() {
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    setup_linear_task_x86(&mut host, &mut state, &[4, 5]);
    let (tight, stride, h) = (8usize, 16u64, 3u32);
    let gva = 0x40u64;
    let data0 = 4u64 << PAGE_SHIFT_X86;
    // Sentinel-fill the whole span so untouched padding is provable.
    host.write_gpa(data0 + gva, &vec![0xEEu8; (h as u64 * stride) as usize])
        .unwrap();
    let bytes: Vec<u8> = (0..tight * h as usize).map(|i| i as u8 + 1).collect();
    assert!(write_linear_texture_bulk(
        &mut state, &mut host, 1, gva, stride, tight, h, &bytes, None
    ));
    for y in 0..h as u64 {
        let mut row = vec![0u8; stride as usize];
        host.read_gpa(data0 + gva + y * stride, &mut row).unwrap();
        assert_eq!(
            &row[..tight],
            &bytes[(y as usize) * tight..(y as usize) * tight + tight],
            "row y={y}"
        );
        if y + 1 < h as u64 {
            assert!(
                row[tight..].iter().all(|&b| b == 0xEE),
                "padding must stay untouched y={y}"
            );
        }
    }
}

/// The linear compute rail carries the deferred-window bound on BOTH of its
/// writers, so which one a real flush takes cannot decide whether the bound
/// applies.
///
/// `write_linear_guest_within` tries the packed bulk view first and drops to a
/// per-row `write_task_gva_product_within` when the span is fragmented. Here the
/// task's page table resolves `data0`, and the window says it was armed on a
/// page it does not own — the shape the guest produces by releasing a GPU
/// allocation and letting the range be re-pointed. Both writers must refuse, and
/// `data0` must be byte-unchanged: an unbounded writer would have scattered a
/// compute storage image into whatever owns it now.
#[test]
fn a_linear_flush_cannot_write_pages_its_window_was_not_armed_on() {
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    setup_linear_task_x86(&mut host, &mut state, &[4, 5]);
    let (tight, stride, h) = (8usize, 16u64, 3u32);
    let gva = 0x40u64;
    let data0 = 4u64 << PAGE_SHIFT_X86;
    let span = (h as u64) * stride;
    host.write_gpa(data0 + gva, &vec![0xEEu8; span as usize])
        .unwrap();
    let bytes: Vec<u8> = (0..tight * h as usize).map(|i| i as u8 + 1).collect();

    // A page set naming somewhere this window's GVA does not resolve to.
    let foreign: std::collections::HashSet<u64> = [9u64 << PAGE_SHIFT_X86].into_iter().collect();
    assert!(
        !write_linear_texture_bulk(
            &mut state,
            &mut host,
            1,
            gva,
            stride,
            tight,
            h,
            &bytes,
            Some(&foreign)
        ),
        "the bulk view must refuse a span outside the armed pages"
    );
    assert_eq!(
        write_linear_guest_within(
            &mut state,
            &mut host,
            1,
            gva,
            stride,
            tight,
            h,
            &bytes,
            "test",
            Some(&foreign),
        ),
        LinearWrite::Failed,
        "the per-row fallback must refuse it too, not quietly land the rows"
    );
    let mut back = vec![0u8; span as usize];
    host.read_gpa(data0 + gva, &mut back).unwrap();
    assert!(
        back.iter().all(|&b| b == 0xEE),
        "not one byte may reach a page the window was not armed on"
    );

    // Control: armed on the page it actually resolves to, the same flush lands.
    let armed: std::collections::HashSet<u64> = [data0].into_iter().collect();
    assert_eq!(
        write_linear_guest_within(
            &mut state,
            &mut host,
            1,
            gva,
            stride,
            tight,
            h,
            &bytes,
            "test",
            Some(&armed),
        ),
        LinearWrite::Written,
        "a window writing its own pages must still write"
    );
    host.read_gpa(data0 + gva, &mut back).unwrap();
    assert_eq!(&back[..tight], &bytes[..tight]);
}

/// Fragmented PFNs under strict Linux map: the packed view fails, both
/// bulk helpers return false, and the per-row fallback stays load-bearing.
#[cfg(not(target_os = "macos"))]
#[test]
fn bulk_linear_helpers_fall_back_on_fragmented_span() {
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    // Non-adjacent leaf PFNs: 4 then 10.
    setup_linear_task_x86(&mut host, &mut state, &[4, 10]);
    let page = 1u64 << PAGE_SHIFT_X86;
    let (tight, stride, h) = (8usize, 16u64, 3u32);
    // Span crosses the page boundary → packed map_pages fails.
    let gva = page - stride;
    let mut bytes = vec![0u8; tight * h as usize];
    assert!(!read_linear_texture_bulk(
        &mut state, &mut host, 1, gva, stride, tight, h, &mut bytes
    ));
    assert!(!write_linear_texture_bulk(
        &mut state, &mut host, 1, gva, stride, tight, h, &bytes, None
    ));
    // The fallback primitive still lands bytes across the fragmented span.
    let payload = [0xABu8; 8];
    assert!(
        gva_mem::write_task_gva_product_within(&mut state, &mut host, 1, gva, &payload, None)
            .is_ok()
    );
}

/// A staged compute buffer records the pages it resolved to, and the writeback
/// that runs after the dispatch is bounded to them.
///
/// `stage_buffer` reads the guest bytes before the dispatch; `writeback_buffer`
/// runs at the far end of the dispatch and, in a nested session, after however
/// many more jobs accumulated. That is the longest arm-to-write gap on this
/// rail, and the guest can hand the range to something else across it.
#[test]
fn a_staged_buffer_carries_the_pages_its_writeback_is_bounded_to() {
    use reims_vgpu_core::endian::st32;
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let (dir_pfn, root_pfn, pt_base) = (2u32, 3u32, 4u32);
    let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_ARM64E;
    let root_gpa = (root_pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x4000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], root_pfn);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    for i in 0..8u32 {
        let pfn = pt_base + i;
        host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        let _ = host.write_gpa(root_gpa + (i as u64) * 4, &pte);
    }
    state.define_task(1, 0x1000, dir_pfn);

    let page = 1u64 << PAGE_SHIFT_ARM64E;
    let pages = staged_span_pages(&state, &host, 1, page, 0x100);
    assert_eq!(pages.len(), 1, "a 256-byte span sits in one page");
    assert!(pages.contains(&((pt_base as u64 + 1) << PAGE_SHIFT_ARM64E)));

    // The guest re-points that virtual page while the dispatch is in flight.
    let mut pte = [0u8; 4];
    st32(&mut pte, pt_base + 6);
    let _ = host.write_gpa(root_gpa + 4, &pte);
    let bytes = vec![0xffu8; 0x100];
    let err = crate::runtime::gva_mem::write_task_gva_product_within(
        &mut state,
        &mut host,
        1,
        page,
        &bytes,
        Some(&pages),
    )
    .expect_err("a page the dispatch never named must be refused");
    assert!(
        matches!(err, crate::runtime::host::MemError::WriteOutsideWindow),
        "expected WriteOutsideWindow, got {err:?}"
    );
    // Unbounded, the same write lands in whatever owns that page now.
    assert!(
        crate::runtime::gva_mem::write_task_gva_product_within(
            &mut state, &mut host, 1, page, &bytes, None,
        )
        .is_ok(),
        "without the bound the write reaches the new owner"
    );

    // An unresolvable span records nothing rather than an empty authorisation,
    // so the writeback stays unbounded and the writer fails closed itself.
    assert!(staged_span_pages(&state, &host, 1, 0, 0x100).is_empty());
    assert!(staged_span_pages(&state, &host, 1, page, 0).is_empty());
}

/// Every destination reaches a licence; nothing is turned away for its shape.
///
/// Both destination shapes have one, and the shapes differ only in which licence
/// answers — a guest-linear plane goes to `licence_gva_plane` and a tiled surface
/// mapping to `licence_iosurface_texture_surface`. Residency is not a shape at all: a
/// registered resident is a perfectly good source for a copy, and
/// what holds it across a submitted-not-waited copy is the engine's pin, taken
/// where the write debt is armed and released from the ring slot's cleanup.
///
/// That third case is the regression this test exists for. Residency *was* a
/// refusal here, and it reached 81 of the 89 linear windows a driven macos-13
/// boot produces — a rule written to be safe that turned out to be most of the
/// traffic the arm exists to remove. Re-adding it would read as caution and cost
/// 91 % of the saving, so it is asserted against directly.
///
/// Every refusal answers `Host`, so the return value alone cannot say *which*
/// gate fired, and a window that fell through to the licence check reads
/// identically to one an earlier gate caught. The census route is what
/// distinguishes them, so each case is asserted on its own counter.
///
/// The IOSurface texture cases assert the thing that is easiest to regress back to. That
/// class was the largest this arm did not reach — 35 of the 51 storage
/// destinations of a driven macos-13 boot — and the reason was a `return` on the
/// destination's *shape*, before anything about the surface had been asked. A
/// tiled surface mapping is not a guest-linear plane, which is true, and does
/// not make it unreachable. So `compute_dst_host_not_linear` is asserted at
/// zero: reintroducing that early answer would read as a correct statement about
/// the GVA licence and put 91 % of the class back on the readback rail.
///
/// Vulkan-only: the direct arm is a `VK_EXT_external_memory_host` import, and
#[test]
fn a_licence_and_not_the_destinations_shape_decides_the_direct_arm() {
    use crate::runtime::drain::census::store_route_count;
    use reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM;
    use reims_vgpu_core::ComputeImageDestination;
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let held = reims_vgpu_protocol::StorageImageFormat::Rgba8Unorm;

    let linear = |pages: crate::runtime::draw::StoreTargetPages| TextureWriteback::Linear {
        pages,
        texture_ref: 44,
        gva: 0x101000,
        pixel_format: MTL_FORMAT_RGBA8_UNORM,
        row_stride: 8,
        width: 2,
        height: 2,
        bpp: 4,
    };
    let staged = |writeback, residency| StagedTexture {
        resource_ref: 44,
        binding: 32,
        array_element: 0,
        descriptor_count: 1,

        pixel_format: MTL_FORMAT_RGBA8_UNORM,
        storage_format: Some(reims_vgpu_protocol::StorageImageFormat::Rgba8Unorm),
        view_swizzle: reims_vgpu_protocol::SwizzlePlan::default(),
        width: 2,
        height: 2,
        multisampled: false,
        bytes: vec![0u8; 16],
        is_storage: true,
        residency,
        input: VulkanTextureInput::HostBytes(Vec::new()),
        writeback,
    };
    let is_host = |d: &ComputeImageDestination| matches!(d, ComputeImageDestination::Host);

    // A window whose pages never resolved cannot be licensed, so even the
    // transient linear shape reads back. This is also the arm that holds on a
    // host with no guest-RAM import, where `references_for_runs` refuses.
    let before = store_route_count("compute_dst_host_unlicensed");
    assert!(
        is_host(&direct_destination(
            &mut state,
            &mut host,
            &staged(
                linear(crate::runtime::draw::StoreTargetPages::empty()),
                None
            ),
            held,
        )),
        "an unlicensed window reads back"
    );
    assert_eq!(
        store_route_count("compute_dst_host_unlicensed"),
        before + 1,
        "the licence refusal is the gate that caught it"
    );

    // An IOSurface texture destination is not a guest-linear plane at all. It is also the
    // largest class this arm does not reach, so the same call must band whether
    // a raw copy could ever have served it — the route counter says how many
    // there are and the split says how many are reachable.
    let iosurface_texture = |mapping_id, format| TextureWriteback::IOSurface {
        mapping_id,
        surface_offset: 0,
        surface_bpr: 8,
        span_end: 16,
        width: 2,
        height: 2,
        format,
    };
    let unlicensed = || store_route_count("compute_dst_host_iosurface_texture_unlicensed");
    for mapping_id in [1, 2] {
        state.surfaces.mappings.insert(
            mapping_id,
            crate::model::SurfaceMappingEntry {
                lifecycle: crate::model::SurfaceMappingLifecycle {
                    active: true,
                    ..Default::default()
                },
                ..Default::default()
            }
            .with_geometry_for_test(2, 2, MTL_FORMAT_RGBA8_UNORM),
        );
    }
    // Mapping 1 is staged at the texel the dispatch holds, so a raw copy could
    // serve it; mapping 2 is staged at a different one, and a copy converts
    // nothing, so no licence could land it however the pages resolve. The format
    // that decides is the bind's own, not the mapping's declaration — the bind
    // may be a IOSurface plane view view reinterpreting the surface, and the staged format is
    // the one both the seeding read and the landing write are arithmetic over.
    //
    // Mapping 3 is never registered, so there is nothing to write into.
    //
    // All three answer `Host` here, and for three different reasons, only one of
    // which is about the format: `FakeHost` publishes no guest-RAM import, so
    // even the agreeing mapping's licence is refused at the reference walk. What
    // this asserts is that all three *reached* the licence — the arm no longer
    // answers `Host` on the shape of the destination alone, which is what it did
    // for 35 of the 51 storage destinations of a driven macos-13 boot.
    let before = unlicensed();
    let not_linear = store_route_count("compute_dst_host_not_linear");
    for (mapping_id, format) in [
        (1, MTL_FORMAT_RGBA8_UNORM),
        (2, reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA16_FLOAT),
        (3, MTL_FORMAT_RGBA8_UNORM),
    ] {
        assert!(
            is_host(&direct_destination(
                &mut state,
                &mut host,
                &staged(iosurface_texture(mapping_id, format), None),
                held,
            )),
            "no guest-RAM import, so every IOSurface texture licence is refused here"
        );
    }
    assert_eq!(
        unlicensed(),
        before + 3,
        "the IOSurface texture licence is what refused, not the destination's shape"
    );
    // A delta and not an absolute: these counters are process-global and this
    // suite runs serially in one binary, so a zero read absolutely would be
    // asserting about every other test too.
    assert_eq!(
        store_route_count("compute_dst_host_not_linear"),
        not_linear,
        "an IOSurface texture destination is no longer turned away for not being linear"
    );

    // And the case this test exists for: a resident window is routed on its
    // destination like any other, so it reaches the licence. Asserted against
    // the same linear writeback the first case used, so residency is the only
    // term that differs — and on the *licence's* counter, which is what says it
    // got that far. `FakeHost` publishes no guest-RAM import, so the licence
    // itself refuses here and the answer is still `Host`; what this asserts is
    // that residency was not what decided it.
    let before = store_route_count("compute_dst_host_unlicensed");
    assert!(
        is_host(&direct_destination(
            &mut state,
            &mut host,
            &staged(
                linear(crate::runtime::draw::StoreTargetPages::empty()),
                Some(ComputeStorageResidencyCandidate {
                    key: crate::model::ComputeStorageResidencyKey::linear(
                        reims_vgpu_protocol::ResourceId::new(44, 1),
                        0x101000,
                        8,
                        0x101010,
                        2,
                        2,
                        MTL_FORMAT_RGBA8_UNORM,
                    ),
                    seed_generation: 0,
                }),
            ),
            held,
        )),
        "the licence is what refuses on a host with no guest-RAM import"
    );
    assert_eq!(
        store_route_count("compute_dst_host_unlicensed"),
        before + 1,
        "a resident window is routed on its destination, not on its residency"
    );
    assert_eq!(
        store_route_count("compute_dst_host_resident"),
        0,
        "and nothing refuses on residency at all any more"
    );
}

/// A staged window keeps the walk's *order*, and a scattered mapping proves it.
///
/// The compute rail carried its destination pages as a bare `HashSet`, which is
/// enough to bound a write and not enough to place one: a direct copy into guest
/// pages hands its runs to `references_for_runs`, which consumes them in
/// **guest-virtual** order. For a scattered mapping that order is not ascending
/// GPA, so recovering it by sorting the set would land the window's rows in the
/// wrong pages — silently, with every page still inside the write bound.
///
/// The mapping here descends: virtual page `i` sits at pfn `pt_base + 7 - i`, so
/// a sorted set would answer with the runs reversed. The second half pins the
/// other thing a set cannot say — whether the walk resolved every page it was
/// asked for.
#[test]
fn a_staged_window_records_its_pages_in_guest_virtual_order() {
    use reims_vgpu_core::endian::st32;
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let (dir_pfn, root_pfn, pt_base) = (2u32, 3u32, 4u32);
    let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_ARM64E;
    let root_gpa = (root_pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x4000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], root_pfn);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    for i in 0..8u32 {
        let pfn = pt_base + 7 - i;
        host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        let _ = host.write_gpa(root_gpa + (i as u64) * 4, &pte);
    }
    state.define_task(1, 0x1000, dir_pfn);

    let page = 1u64 << PAGE_SHIFT_ARM64E;
    let gpa_of = |pfn: u32| (pfn as u64) << PAGE_SHIFT_ARM64E;
    let gva = page; // virtual page 1

    let pages = staged_window_pages(&state, &host, 1, gva, page, 3);
    let want = [
        gpa_of(pt_base + 6),
        gpa_of(pt_base + 5),
        gpa_of(pt_base + 4),
    ];
    assert_eq!(
        pages.ordered_complete(gva, page),
        Some(&want[..]),
        "the record must read in GVA order, not ascending GPA"
    );
    assert_eq!(
        pages.membership().len(),
        3,
        "and it bounds the same three pages a set would have"
    );

    // A walk that could not resolve every page of its span refuses to place
    // anything, rather than answering with the pages it did find.
    let short_gva = page * 7;
    let short = staged_window_pages(&state, &host, 1, short_gva, page, 2);
    assert!(
        !short.membership().is_empty(),
        "the page that did resolve is still bound"
    );
    assert!(
        short.ordered_complete(short_gva, page).is_none(),
        "an incomplete walk cannot be placed"
    );

    // Nothing to walk records nothing, which is distinct from a complete record
    // of zero pages — no span can produce one of those.
    assert!(staged_window_pages(&state, &host, 1, 0, page, 3)
        .membership()
        .is_empty());
    assert!(staged_window_pages(&state, &host, 1, gva, 0, 3)
        .membership()
        .is_empty());
}

/// The two halves of a resident answer partition; neither rail can see both.
///
/// `StagedTexture` used to carry this enum as a `bool` and an `Option` side by
/// side, rebuilt independently by all three staging rails. Nothing then stopped
/// a rail from setting both — and the two consumers read one field each, so a
/// binding claiming to be seeded *and* sampled would have been dispatched as
/// both a storage seed skip and a resident sampled bind, seeding one image from a
/// placeholder. The state is unrepresentable now; this pins the accessors that
/// replaced the two fields so a later variant cannot answer to both consumers
/// or to neither.
#[test]
fn a_resident_answer_is_a_seed_or_a_sample_and_never_both() {
    let key = crate::model::ComputeStorageResidencyKey::surface(7, 3, 0, 16, 64, 4, 4, 0x50);
    for (serve, what) in [
        (ResidentServe::Seed(11), "seed"),
        (ResidentServe::Sample(key, 12), "sample"),
    ] {
        assert_eq!(
            serve.seed_generation().is_some(),
            serve.sample_source().is_none(),
            "exactly one accessor must answer for the {what} variant"
        );
    }
    assert_eq!(ResidentServe::Seed(11).seed_generation(), Some(11));
    assert_eq!(
        ResidentServe::Sample(key, 12).sample_source(),
        Some((key, 12))
    );

    // And "no resident" answers neither, which is what makes `serve.is_none()`
    // the one gate the rails use to decide they must read the guest window.
    let none: Option<ResidentServe> = None;
    assert!(none.and_then(ResidentServe::seed_generation).is_none());
    assert!(none.and_then(ResidentServe::sample_source).is_none());
}

/// An `MTLDispatchType` the contract does not declare is named, counted, and
/// substituted — and the two it does declare cross untouched and unremarked.
///
/// `WRITE_DESCRIPTOR` puts this ordinal on the wire unbounded: the decoder
/// stores `d.dispatch_type.get()` with no range check, so whatever the guest
/// wrote reaches the accumulator. An unknown value must be reported before the
/// compatibility substitution to `Serial`.
///
/// Both halves are the test. A substitution nobody can see is the failure this
/// commit exists to end; a line spent on the ordinary `Serial` and `Concurrent`
/// records would be a flood on a per-segment path and would bury the one line
/// that means something.
#[test]
fn an_undeclared_dispatch_type_is_named_and_counted_before_it_becomes_serial() {
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_protocol::dispatch::{MTL_DISPATCH_TYPE_CONCURRENT, MTL_DISPATCH_TYPE_SERIAL};

    let before = store_route_count("compute_dispatch_type_unknown");

    let cap = crate::observe::FailCapture::start();
    assert_eq!(
        accepted_dispatch_type(3, MTL_DISPATCH_TYPE_SERIAL),
        MTL_DISPATCH_TYPE_SERIAL
    );
    assert_eq!(
        accepted_dispatch_type(3, MTL_DISPATCH_TYPE_CONCURRENT),
        MTL_DISPATCH_TYPE_CONCURRENT
    );
    assert!(
        cap.lines().is_empty(),
        "a declared dispatch type must spend no line: {:?}",
        cap.lines()
    );
    assert_eq!(
        store_route_count("compute_dispatch_type_unknown"),
        before,
        "a declared dispatch type is not a substitution"
    );
    drop(cap);

    // An ordinal outside the pair: substituted, named once, counted every time.
    let cap = crate::observe::FailCapture::start();
    for _ in 0..3 {
        assert_eq!(
            accepted_dispatch_type(3, 0x5e01),
            MTL_DISPATCH_TYPE_SERIAL,
            "an unrecognised dispatch type encodes Serial"
        );
    }
    let line = cap.one("compute_dispatch_type");
    assert!(
        line.contains("reason=compute_dispatch_type_unknown")
            && line.contains(" task=3 declared=24065"),
        "the substitution must name the value that caused it: {line}"
    );
    assert_eq!(
        store_route_count("compute_dispatch_type_unknown"),
        before + 3,
        "the line is deduped per value; the count is not"
    );
}

/// A stage-input the decoder had to truncate must refuse its pipeline, not
/// arrive as "this kernel declares no stage-input".
///
/// The two were one `None` for as long as the caps could be crossed. The
/// consequence is not only a wrong Metal PSO: on the Vulkan arm
/// `dispatch_compute_vulkan` refuses any pipeline whose `stage_input.is_some()`,
/// so collapsing `OverCap` into `Absent` is what lets an unsupported dispatch
/// through the one guard that exists for it.
#[test]
fn a_truncated_stage_input_is_not_the_same_as_an_absent_one() {
    use crate::runtime::decode::resource::{ComputeStageInputAttribute, ComputeStageInputLayout};

    assert_eq!(classify_stage_input(None), StageInputVerdict::Absent);

    let empty = ComputeStageInputDescriptor::default();
    assert_eq!(
        classify_stage_input(Some(&empty)),
        StageInputVerdict::Absent,
        "a block naming nothing is a kernel with no stage-input"
    );

    let mut used = ComputeStageInputDescriptor::default();
    used.attributes.push(ComputeStageInputAttribute::default());
    used.layouts.push(ComputeStageInputLayout::default());
    assert_eq!(classify_stage_input(Some(&used)), StageInputVerdict::Use);

    let mut dropped_attr = used.clone();
    dropped_attr.dropped_attributes = 1;
    assert_eq!(
        classify_stage_input(Some(&dropped_attr)),
        StageInputVerdict::OverCap,
        "a dropped attribute refuses the pipeline"
    );

    let mut dropped_layout = used.clone();
    dropped_layout.dropped_layouts = 1;
    assert_eq!(
        classify_stage_input(Some(&dropped_layout)),
        StageInputVerdict::OverCap,
        "a dropped layout refuses the pipeline"
    );

    // Both together, and with the entry lists empty — the ordering matters, or
    // an over-cap block whose kept entries are all beyond the cap reads as
    // absent, which is the collapse this test exists to forbid.
    let mut all_dropped = ComputeStageInputDescriptor::default();
    all_dropped.dropped_attributes = 3;
    all_dropped.dropped_layouts = 2;
    assert_eq!(
        classify_stage_input(Some(&all_dropped)),
        StageInputVerdict::OverCap
    );
}

/// Every disjoint live window remains resident until a guest-owned lifetime or
/// content transition retires it. Heap textures have no guest fallback at all;
/// mapping windows have a fallback, but discarding their current GPU replica
/// still costs guest work and is not a cache policy.
#[test]
fn live_compute_mirrors_are_not_evicted_by_an_invented_capacity() {
    use super::{
        note_storage_residency_writeback, ComputeStorageResidencyCandidate, StagedTexture,
        TextureWriteback,
    };
    use crate::model::ComputeStorageResidencyKey;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    const LIVE_WINDOWS: u32 = 32;
    const SURFACE_RESOURCE_REF: u32 = 99;

    let staged = |key: ComputeStorageResidencyKey| StagedTexture {
        resource_ref: key
            .resource()
            .map(|resource| resource.index())
            .unwrap_or(SURFACE_RESOURCE_REF),
        binding: 33,
        array_element: 0,
        descriptor_count: 1,

        pixel_format: key.pixel_format,
        storage_format: Some(reims_vgpu_protocol::StorageImageFormat::Rgba8Uint),
        view_swizzle: reims_vgpu_protocol::SwizzlePlan::default(),
        width: key.width,
        height: key.height,
        multisampled: false,
        bytes: Vec::new(),
        // The mirror is only armed for a storage output, which is what makes a
        // heap texture's engine copy the sole content.
        is_storage: true,
        residency: Some(ComputeStorageResidencyCandidate {
            key,
            seed_generation: 1,
        }),
        input: VulkanTextureInput::HostBytes(Vec::new()),
        writeback: TextureWriteback::None,
    };

    for tex in 0..LIVE_WINDOWS {
        let key = ComputeStorageResidencyKey::heap_placement(
            reims_vgpu_protocol::ResourceId::new(tex, 1),
            0,
            0x100,
            16,
            16,
            0x50,
        );
        assert!(matches!(
            key.origin,
            crate::model::ComputeStorageOrigin::HeapPlacement { .. }
        ));
        note_storage_residency_writeback(&mut state, &staged(key));
    }
    for window in 0..LIVE_WINDOWS {
        let start = u64::from(window) * 0x100;
        let key = ComputeStorageResidencyKey::surface(7, 1, start, 16, start + 0x100, 4, 4, 0x50);
        note_storage_residency_writeback(&mut state, &staged(key));
    }

    assert_eq!(
        state.content.compute_residency.len(),
        (2 * LIVE_WINDOWS) as usize,
        "every independently live mirror remains owned"
    );
    for tex in 0..LIVE_WINDOWS {
        let key = ComputeStorageResidencyKey::heap_placement(
            reims_vgpu_protocol::ResourceId::new(tex, 1),
            0,
            0x100,
            16,
            16,
            0x50,
        );
        assert!(
            state.content.compute_residency.contains(&key),
            "heap texture {tex} lost its mirror"
        );
    }
}

/// A bind past the argument table refuses the dispatch, rather than letting it
/// run with the guest's binding absent.
///
/// The bind walk has no dispatch to refuse, so it records instead, and
/// `resolve_dispatch_dims_reported` — the one gate both executors pass through
/// — is where the refusal lands. Driven through that function rather than
/// asserting the field, because the field is bookkeeping and the refusal is the
/// behaviour.
///
/// The dims themselves are deliberately resolvable here: a dispatch that would
/// otherwise have succeeded is the only one that proves the refusal came from
/// the bind and not from the grid.
#[test]
fn a_bind_past_the_argument_table_refuses_the_dispatch() {
    let host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);

    let mut cmd = ComputeCommand::default();
    cmd.kind = Kind::DispatchThreadgroups;
    cmd.grid = compute::Size3 { x: 4, y: 1, z: 1 };
    cmd.threads_per_threadgroup = compute::Size3 { x: 8, y: 1, z: 1 };

    let mut acc = ComputeAccum::default();
    assert!(
        resolve_dispatch_dims_reported(&mut state, &host, 1, &cmd, &acc).is_ok(),
        "the dispatch resolves before any bind, so a later refusal is the bind's"
    );

    acc.bind_textures(MAX_COMPUTE_TEXTURE_SLOTS + 3, &[RefBinding { ref_: 9 }]);
    assert!(
        acc.textures.is_empty(),
        "the slot is past the table, so there was nowhere to record it"
    );
    let refused = resolve_dispatch_dims_reported(&mut state, &host, 1, &cmd, &acc)
        .expect_err("a dispatch missing a binding the guest asked for is refused");
    assert_eq!(
        refused,
        ComputeStatus::Unsupported("compute_dispatch_bind_past_table")
    );

    // Sticky: the binding stays unrepresentable, so the next dispatch is
    // refused too rather than quietly running without it.
    assert!(
        resolve_dispatch_dims_reported(&mut state, &host, 1, &cmd, &acc).is_err(),
        "one refused bind refuses every dispatch that would have used it"
    );

    // ...until the guest clears that slot, which makes what this accumulator
    // holds equal to what the guest asked for again.
    acc.bind_textures(MAX_COMPUTE_TEXTURE_SLOTS + 3, &[RefBinding { ref_: 0 }]);
    assert!(
        resolve_dispatch_dims_reported(&mut state, &host, 1, &cmd, &acc).is_ok(),
        "a nil bind at the refused slot retires the refusal with it"
    );

    // A clear at a different slot is not that slot, and must not lift it.
    acc.bind_textures(MAX_COMPUTE_TEXTURE_SLOTS + 5, &[RefBinding { ref_: 9 }]);
    acc.bind_textures(2, &[RefBinding { ref_: 0 }]);
    assert!(
        resolve_dispatch_dims_reported(&mut state, &host, 1, &cmd, &acc).is_err(),
        "clearing an unrelated slot leaves the refused one refused"
    );
}

/// A texture the kernel samples and the guest never bound stays explicitly null,
/// and one the kernel merely declares does not.
///
/// This is the repair for the hole that killed a host. The descriptor set layout
/// this device builds is assembled from what the guest bound, so a sampled image
/// the guest left empty is absent from the layout entirely — and Mesa's Intel
/// driver scores each *used* binding as `(use_count << 7) / array_size` over an
/// array it sized to `max_binding + 1` and zero-filled, so the hole divides by
/// zero and `vkCreateComputePipelines` raises `SIGFPE` rather than returning an
/// error this device could decline on.
///
/// The negative half is not decoration. Provisioning for a declared-and-unused
/// variable is legal but pays a descriptor for nothing, and it would destroy the
/// census that separated the two populations in the first place — so a pass that
/// filled both would look identical to this one on a boot and be wrong.
#[test]
fn a_sampled_image_the_kernel_uses_and_the_guest_left_empty_stays_null() {
    let spirv = reims_vgpu_vulkan::spirv_bind::test_module_with_two_sampled_images(33, 34);
    // In production this candidate population comes from the translator's
    // sampled-resource reflection.
    let reflected_sampled = [33, 34];

    // Nothing bound: 33 is sampled and needs a null descriptor; 34 is declared
    // and never referenced, so it stays out of the layout.
    assert_eq!(
        reims_vgpu_vulkan::spirv_bind::null_statically_used_bindings(
            &spirv,
            &reflected_sampled,
            &[],
        ),
        vec![33]
    );

    // The guest bound it after all — there is nothing to substitute.
    assert_eq!(
        reims_vgpu_vulkan::spirv_bind::null_statically_used_bindings(
            &spirv,
            &reflected_sampled,
            &[33],
        ),
        Vec::<u32>::new()
    );

    // A binding the guest supplied that the module does not carry is not
    // invented back into the list.
    assert_eq!(
        reims_vgpu_vulkan::spirv_bind::null_statically_used_bindings(
            &spirv,
            &reflected_sampled,
            &[99],
        ),
        vec![33]
    );
}

/// Build one task with a buffer at ref 7 and a buffer-backed texture at ref 10
/// over it, and return the state, host and the texture's first-row GVA.
///
/// `bytes_per_row` and `offset` are the two fields
/// `newTextureWithDescriptor:offset:bytesPerRow:` adds to a texture
/// descriptor, so they are what a case here varies.
fn buffer_backed_texture_task(
    width: u32,
    height: u32,
    offset: u64,
    bytes_per_row: u64,
    allocation_size: u64,
) -> (Device, FakeHost, u64) {
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use reims_vgpu_wire::ops::backed_texture::{
        BufferTextureBody, BUFFER_TEXTURE_TOTAL_LEN, OPCODE_BUFFER_TEXTURE,
    };

    const OP_HEADER: usize = reims_vgpu_wire::OP_HEADER_LEN;
    const BUFFER_REF: u32 = 7;
    const TEXTURE_REF: u32 = 10;
    const BUFFER_HANDLE: u32 = 5;

    let mut host = FakeHost::new();
    host.stable_map_pages = true;
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    // The buffer that owns the storage.
    let buffer_gva = u64::from(BUFFER_HANDLE) << RESOURCE_PAGE_SHIFT;
    let mut buffer_descriptor = [0u8; 16];
    st64(&mut buffer_descriptor[0..], allocation_size);
    st32(&mut buffer_descriptor[8..], BUFFER_HANDLE);
    let buffer_descriptor_gva = 0x180u64;
    write_task_gva_arm64e(
        &mut host,
        &state.tasks[1],
        buffer_descriptor_gva,
        &buffer_descriptor,
    );
    let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(&mut entry[0..], (OBJECT_TYPE_BUFFER as u32) | (16u32 << 8));
    entry[4..12].copy_from_slice(&buffer_descriptor_gva.to_le_bytes());
    write_task_gva_arm64e(
        &mut host,
        &state.tasks[1],
        list_object_entry_offset(BUFFER_REF, 32).unwrap(),
        &entry,
    );

    // The opcode-9 record placing a texture over it.
    let mut record = vec![0u8; BUFFER_TEXTURE_TOTAL_LEN as usize];
    st32(&mut record[0..], OPCODE_BUFFER_TEXTURE);
    st32(&mut record[4..], BUFFER_TEXTURE_TOTAL_LEN);
    st32(&mut record[8..], TEXTURE_REF);
    let buffer_ref_at = OP_HEADER + std::mem::offset_of!(BufferTextureBody, buffer_ref);
    let offset_at = OP_HEADER + std::mem::offset_of!(BufferTextureBody, offset);
    let bytes_per_row_at = OP_HEADER + std::mem::offset_of!(BufferTextureBody, bytes_per_row);
    let descriptor_at = OP_HEADER + std::mem::offset_of!(BufferTextureBody, desc);
    st32(&mut record[buffer_ref_at..], BUFFER_REF);
    st64(&mut record[offset_at..], offset);
    st64(&mut record[bytes_per_row_at..], bytes_per_row);
    // 2D, usage=3, BGRA8Unorm, one mip / sample / array element, shared.
    st32(
        &mut record[descriptor_at..],
        2 | (3 << 8) | ((MTL_FORMAT_BGRA8_UNORM as u32) << 16),
    );
    st32(&mut record[descriptor_at + 4..], width);
    st32(&mut record[descriptor_at + 8..], height);
    st32(&mut record[descriptor_at + 12..], 1);
    for (field, value) in [(16usize, 1u16), (18, 1), (20, 1), (22, 0)] {
        record[descriptor_at + field..descriptor_at + field + 2]
            .copy_from_slice(&value.to_le_bytes());
    }
    let record_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E;
    write_task_gva_arm64e(&mut host, &state.tasks[1], record_gva, &record);
    let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut entry[0..],
        (crate::runtime::decode::resource::OBJECT_TYPE_TEXTURE_VIEW as u32)
            | (BUFFER_TEXTURE_TOTAL_LEN << 8),
    );
    entry[4..12].copy_from_slice(&record_gva.to_le_bytes());
    write_task_gva_arm64e(
        &mut host,
        &state.tasks[1],
        list_object_entry_offset(TEXTURE_REF, 32).unwrap(),
        &entry,
    );

    (state, host, buffer_gva + offset)
}

/// A texture over an `MTLBuffer`'s storage is a linear window like any other.
///
/// It used to be refused outright as `compute_buffer_texture_unsupported`,
/// which cost the guest every compute bind of one — and the construction is
/// how a guest hands the same bytes to a kernel as texels and as a buffer, so
/// the loss is not a corner. The window must reach the guest pages through the
/// **buffer's** alias: a second alias over one allocation is exactly what this
/// construction exists to avoid.
#[test]
fn a_buffer_backed_texture_stages_as_a_linear_window_over_its_buffer() {
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let (mut state, mut host, first_row_gva) =
        buffer_backed_texture_task(60, 16, 0x100, 256, 0x4000);
    crate::runtime::guest_ram_map::reset();
    crate::runtime::guest_ram::latch_import_limits(1 << PAGE_SHIFT_ARM64E, 1 << 30, 1 << 30);
    let staged = stage_texture_raw(
        &mut state,
        &mut host,
        1,
        10,
        32,
        ComputeTextureStage::Sampled2d,
    )
    .expect("a buffer-backed texture must stage");
    crate::runtime::guest_ram::forget_import_limits();
    crate::runtime::guest_ram_map::reset();

    assert_eq!((staged.width, staged.height), (60, 16));
    assert_eq!(staged.pixel_format, MTL_FORMAT_BGRA8_UNORM);
    match &staged.input {
        VulkanTextureInput::GuestPages(source) => {
            // The last row is tight, not a full stride: Metal does not require
            // the final row's trailing pad to be inside the allocation.
            assert_eq!(source.total_len, 15 * 256 + 60 * 4);
            assert_eq!(source.source_offset, 0x100);
            // 256 bytes of stride over a 4-byte texel is 64 texels of row
            // length against 60 texels of content, which is what tells the
            // copy where each row starts.
            assert_eq!(source.row_length_texels, 64);
        }
        _ => panic!("an importable buffer texture must retain guest pages"),
    }
    // Keyed by the buffer, so a kernel binding ref 7 as a buffer and ref 10 as
    // a texture shares one alias rather than importing the allocation twice.
    assert!(state.bound_buffers.packed(1, 7).is_some());
    assert!(state.bound_buffers.packed(1, 10).is_none());
    assert_eq!(first_row_gva, (5u64 << RESOURCE_PAGE_SHIFT) + 0x100);
}

/// The window must fit the buffer the guest named.
///
/// A placement is arithmetic over an allocation this device does not own, so
/// an offset and pitch that reach past the end are refused by name rather than
/// read — the CPU fallback walks by GVA and would happily read whatever the
/// next allocation put there.
#[test]
fn a_buffer_backed_texture_past_the_end_of_its_buffer_is_refused_by_name() {
    // 16 rows of 256 bytes from 0x100 reaches 0x1000, one byte past a 0xfff
    // allocation.
    let (mut state, mut host, _) = buffer_backed_texture_task(64, 16, 0x100, 256, 0xfff);
    match stage_texture_raw(
        &mut state,
        &mut host,
        1,
        10,
        32,
        ComputeTextureStage::Sampled2d,
    ) {
        Err(ComputeStatus::MissingTexture("compute_buffer_tex_span_oob")) => {}
        Err(other) => panic!("expected span_oob, got {}", other.reason()),
        Ok(_) => panic!("a window past the allocation must be refused, not staged"),
    }
}

/// A `bytesPerRow` of zero is the API's own spelling of one tight row, which
/// is what Metal accepts for a 1D or texture-buffer texture. It is a declared
/// value of the field, so it is read rather than treated as a missing pitch.
#[test]
fn a_zero_bytes_per_row_buffer_texture_is_tight_rows() {
    let (mut state, mut host, _) = buffer_backed_texture_task(64, 1, 0, 0, 0x4000);
    crate::runtime::guest_ram_map::reset();
    crate::runtime::guest_ram::latch_import_limits(1 << PAGE_SHIFT_ARM64E, 1 << 30, 1 << 30);
    let staged = stage_texture_raw(
        &mut state,
        &mut host,
        1,
        10,
        32,
        ComputeTextureStage::Sampled2d,
    )
    .expect("a tight-row buffer-backed texture must stage");
    crate::runtime::guest_ram::forget_import_limits();
    crate::runtime::guest_ram_map::reset();
    match &staged.input {
        VulkanTextureInput::GuestPages(source) => {
            assert_eq!(source.total_len, 64 * 4);
            // Tight rows need no row length: the copy's own extent is the pitch.
            assert_eq!(source.row_length_texels, 0);
        }
        _ => panic!("expected guest pages"),
    }
}
