#![allow(
    clippy::field_reassign_with_default,
    reason = "wire fixtures are assembled field by field to keep each protocol case explicit"
)]

use super::*;
use crate::contract::endian::{st32, st64};
use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
use crate::runtime::decode::compute;
use crate::runtime::decode::resource::{
    list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_BUFFER, OBJECT_TYPE_IOSURFACE,
    RESOURCE_PAGE_SHIFT,
};
/// Compute-pipeline descriptor constants, used only by the Metal-arm
/// execute tests below.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
use crate::runtime::decode::resource::{
    OBJECT_TYPE_FUNCTION, PIPELINE_TAG_KERNEL_FUNC, TYPE7_FIRST_TLVS, TYPE7_OBJECT_COMPUTE_PIPELINE,
};
use crate::runtime::gva_mem;
use crate::runtime::gva_mem::write_task_gva_arm64e;
use crate::runtime::host::FakeHost;
use reims_vgpu_wire::device_desc::Type5Builder;

#[cfg(feature = "backend-vulkan")]
#[test]
fn spirv_word_parser_splits_short_header_from_misalignment() {
    assert_eq!(
        spirv_words_le(&[0; 16]).unwrap_err(),
        ComputeSpirvDecline::HeaderTooShort {
            len: 16,
            minimum: 20
        }
    );
    assert_eq!(
        spirv_words_le(&[0; 21]).unwrap_err(),
        ComputeSpirvDecline::LengthMisaligned {
            len: 21,
            alignment: 4
        }
    );
    assert_eq!(spirv_words_le(&[0; 20]).unwrap().len(), 5);
}

#[test]
fn compute_spirv_declines_are_distinct_and_log_safe() {
    use crate::observe::Decline as _;
    let declines = [
        ComputeSpirvDecline::HeaderTooShort {
            len: 16,
            minimum: 20,
        },
        ComputeSpirvDecline::LengthMisaligned {
            len: 21,
            alignment: 4,
        },
    ];
    assert_ne!(declines[0].slug(), declines[1].slug());
    for decline in declines {
        assert!(decline
            .slug()
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte == b'_'));
        for (_, value) in decline.fields() {
            assert!(!value.contains(char::is_whitespace));
        }
    }
}

/// A type-5 view names its IOSurface plane on the wire (record `+0x20`, the
/// `newTextureWithDescriptor:iosurface:plane:` argument). When two planes share
/// geometry and bytes-per-element the geometry scan cannot separate them and
/// falls back to inventing a packed window at offset 0 — which is the *first*
/// plane's bytes. The wire index is the only key, and this path already decoded
/// it, so a compute stage of the alpha plane must not read the luma plane.
///
/// Shape is the live v0a8 (biplanar video + alpha) layout scaled down: plane 0
/// and plane 2 are both R8 at identical dims, plane 1 is the RG8 chroma.
#[test]
fn stage_texture_type5_plane_index_beats_the_ambiguous_geometry_scan() {
    use crate::contract::endian::st16;
    use crate::contract::iosurface_pages::{
        DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_LEN, DEVICE_DESC_PLANES, DEVICE_DESC_PLANE_COUNT,
        DEVICE_PLANE_BPE, DEVICE_PLANE_BPR, DEVICE_PLANE_DESC_LEN, DEVICE_PLANE_DIMS,
        DEVICE_PLANE_OFFSET, DEVICE_PLANE_SIZE, PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID,
    };
    use crate::contract::pixel_format::{MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_R8_UNORM};
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};

    // elemW@0, width u24@1, elemH@4, height u24@5.
    fn plane_dims(width: u32, height: u32) -> u64 {
        ((width as u64 & 0xff_ffff) << 8) | ((height as u64 & 0xff_ffff) << 40)
    }

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    let sid = 3u32;
    let type5_ref = 10u32;
    let pfn = 0x20u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0x5a);
    assert!(state.map_surface(sid));
    {
        let m = state.mappings.get_mut(&sid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
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

    // 56-byte type-5 blob: 8-byte head, then kind/blob_len/own_ref and a 0x24
    // record whose `+0x20` carries the plane index.
    let desc_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E;
    let type5_desc = Type5Builder::new(sid, 0, 10, 0x42)
        .unknown(0x01)
        .geometry(MTL_FORMAT_R8_UNORM, 4, 4, 1)
        .trailer([1, 0, 1, 0, 1, 0, 0x10, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        // IOSurface plane index = 2 (alpha)
        .plane_index(2);
    let type5_desc = type5_desc.bytes();
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, type5_desc);
    let off = list_object_entry_offset(type5_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((type5_desc.len() as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    let staged = stage_texture_raw(&mut state, &mut host, 1, type5_ref, 33, true)
        .expect("a type-5 plane view over a mapped surface must stage");
    match staged.writeback {
        TextureWriteback::Type11 {
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
        _ => panic!("a type-5 view over a surface mapping must write back as type-11"),
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

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut seg = crate::runtime::compute_session::ComputeSegment {
        acc,
        session: None,
        block: None,
    };
    let mut cmd = ComputeCommand::default();
    // Empty start-do-while encodes without a condition buffer.
    cmd.kind = Kind::ControlStartDoWhile;
    let st = apply_record(&mut state, &mut host, 1, &cmd, &mut seg);
    assert!(
        matches!(
            st,
            Some(ComputeStatus::Ok)
                | Some(ComputeStatus::NoMetal(_))
                | Some(ComputeStatus::MetalFailed(_))
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
fn resolve_indirect_threadgroups_from_buffer() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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

    let st = ComputeStatus::MissingTexture("compute_stage_tex_type5_no_map");
    assert_eq!(st.refusal(), Some("compute_stage_tex_type5_no_map"));
    assert_eq!(st.class(), "missing_texture");
    let line = Emit::refusal("compute_record", &st)
        .expect("a refusal renders a line")
        .field("pipe", 7)
        .render();
    assert_eq!(
        line,
        "compute_record reason=compute_stage_tex_type5_no_map \
             class=missing_texture pipe=7"
    );

    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    {
        let status = crate::backend::metal::error::Status::args(
            "metal_compute_reflection_usage_output_missing",
        )
        .field("capacity", 8usize);
        let st = ComputeStatus::MetalBackend(status);
        assert_eq!(st.class(), "metal_args");
        assert_eq!(
            Emit::refusal("compute_record", &st)
                .expect("the exact Metal refusal must survive the runtime carrier")
                .render(),
            "compute_record reason=metal_compute_reflection_usage_output_missing \
                 class=args capacity=8 recovery=metal_failed"
        );
    }
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

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    let staged = stage_buffer(&state, &host, 1, &bind);
    assert_eq!(
        staged.err().and_then(|e| e.refusal()),
        Some(crate::observe::ladder_slug!(
            "compute_stage_buf",
            no_list_entry
        ))
    );
}

/// Callers must pass page_shift explicitly; 12 and 14 place handle differently.
#[test]
fn buffer_backing_gva_requires_explicit_page_shift() {
    use crate::runtime::decode::resource::BufferDescriptor;
    let d = BufferDescriptor {
        allocation_size: 0x1000,
        handle64: 0x101,
        handle: 0x101,
    };
    let (gva12, _) = d.backing_gva_size(PAGE_SHIFT_X86).expect("12");
    let (gva14, _) = d.backing_gva_size(PAGE_SHIFT_ARM64E).expect("14");
    assert_eq!(gva12, 0x101000, "x86 handle<<12");
    assert_eq!(gva14, 0x404000, "arm handle<<14");
    assert_ne!(gva12, gva14);
}

#[test]
fn dispatch_missing_pipeline() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));
    let acc = ComputeAccum::default();
    let mut cmd = ComputeCommand::default();
    cmd.kind = Kind::DispatchThreadgroups;
    cmd.grid = compute::Size3 { x: 1, y: 1, z: 1 };
    cmd.threads_per_threadgroup = compute::Size3 { x: 1, y: 1, z: 1 };
    let st = execute_dispatch(&mut state, &mut host, 1, &acc, &cmd);
    // The slug names *which* pipeline check refused, and it differs by
    // backend: both arms open with `pipeline_ref == 0`, and before the
    // status carried a reason the two were indistinguishable in the log.
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    assert_eq!(
        st,
        ComputeStatus::MissingPipeline("compute_mtl_pipeline_ref_zero")
    );
    #[cfg(feature = "backend-vulkan")]
    assert_eq!(
        st,
        ComputeStatus::MissingPipeline("compute_vk_pipeline_ref_zero")
    );
}

/// Linux without vulkan feature: dispatch is NoMetal (census). With
/// backend-vulkan, missing pipeline is MissingPipeline (real encode path).
#[test]
#[cfg(feature = "backend-vulkan")]
fn dispatch_nometal_with_texture_binds() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    let st = execute_dispatch(&mut state, &mut host, 1, &acc, &cmd);
    assert!(
        matches!(
            st,
            ComputeStatus::MissingPipeline(_)
                | ComputeStatus::MissingMtlb(_)
                | ComputeStatus::MissingTexture(_)
                | ComputeStatus::MetalFailed(_)
                | ComputeStatus::Unsupported(_)
        ),
        "vulkan path attempts encode, got {st:?}"
    );
    // Nested short-circuit remains NoMetal on Linux (SPI not wired).
    let mut session = crate::runtime::compute_session::ComputeSession { control_depth: 0 };
    let st2 = execute_dispatch_nested(&mut state, &mut host, 1, &acc, &cmd, &mut session);
    assert_eq!(st2, ComputeStatus::NoMetal("compute_nested_no_vulkan_path"));
}

#[test]
#[cfg(feature = "backend-vulkan")]
fn dispatch_missing_pipeline_not_nometal() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));
    let acc = ComputeAccum::default();
    let mut cmd = ComputeCommand::default();
    cmd.kind = Kind::DispatchThreadgroups;
    cmd.grid = compute::Size3 { x: 1, y: 1, z: 1 };
    cmd.threads_per_threadgroup = compute::Size3 { x: 1, y: 1, z: 1 };
    let st = execute_dispatch(&mut state, &mut host, 1, &acc, &cmd);
    assert_eq!(
        st,
        ComputeStatus::MissingPipeline("compute_vk_pipeline_ref_zero")
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
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn a_format_with_no_storage_selector_refuses_the_same_way_from_every_rail() {
    use crate::runtime::compute_exec::{split_staged_textures, ComputeStatus, StagedTexture};

    let mut no_selector = StagedTexture {
        binding: 33,
        texture_ref: 44,
        // A sample-only format: `contract::pixel_format::storage_selector` has
        // no entry for it by design, which is exactly the class this refuses.
        pixel_format: crate::contract::pixel_format::MTL_FORMAT_R32_FLOAT,
        storage_selector: None,
        width: 4,
        height: 4,
        bytes: vec![0; 64],
        is_storage: true,
        #[cfg(feature = "backend-vulkan")]
        residency: None,
        #[cfg(feature = "backend-vulkan")]
        seed_skipped: false,
        #[cfg(feature = "backend-vulkan")]
        sample_resident: None,
        writeback: TextureWriteback::None,
    };

    assert_eq!(
        no_selector.storage_selector_or_refuse(7, 9),
        Err(ComputeStatus::Unsupported("compute_no_backend_selector")),
        "the refusal slug is one string, not one per rail"
    );
    assert_eq!(
        split_staged_textures(std::slice::from_mut(&mut no_selector), 7, 9).err(),
        Some(ComputeStatus::Unsupported("compute_no_backend_selector")),
        "the split refuses through the same helper, not a second copy of it"
    );

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    let line = log
        .lines()
        .rev()
        .find(|l| l.starts_with("compute_texture_format "))
        .expect("a lost bind must name itself");
    for field in [
        "reason=no_backend_selector",
        "task=7",
        "pipe=9",
        "bind=33",
        "ref=44",
        "storage=1",
    ] {
        assert!(
            line.contains(field),
            "the line must carry {field}; one of the three copies dropped it: {line}"
        );
    }

    // And a format that does have a selector goes through.
    let mut ok = StagedTexture {
        storage_selector: Some(5),
        ..no_selector
    };
    let (storage, sampled) =
        split_staged_textures(std::slice::from_mut(&mut ok), 7, 9).expect("selector present");
    assert_eq!((storage.len(), sampled.len()), (1, 0));
}

#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
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
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    st32(&mut pdesc[0..], TYPE7_OBJECT_COMPUTE_PIPELINE);
    st32(&mut pdesc[4..], 32); // declared descriptor length
    pdesc[TYPE7_FIRST_TLVS] = 1;
    pdesc[TYPE7_FIRST_TLVS + 1] = PIPELINE_TAG_KERNEL_FUNC;
    pdesc[TYPE7_FIRST_TLVS + 2] = 4;
    st32(&mut pdesc[TYPE7_FIRST_TLVS + 3..], 5);
    let pdesc_gva = 0x140u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], pdesc_gva, &pdesc);
    {
        let off = list_object_entry_offset(6, 32).unwrap();
        let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_TYPE7 as u32) | (32u32 << 8);
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

    let mut acc = ComputeAccum::default();
    acc.set_pipeline(6);
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
    cmd.grid = compute::Size3 { x: 1, y: 1, z: 1 };
    cmd.threads_per_threadgroup = compute::Size3 { x: 4, y: 1, z: 1 };
    let st = execute_dispatch(&mut state, &mut host, 1, &acc, &cmd);
    assert!(
        matches!(
            st,
            ComputeStatus::Ok | ComputeStatus::MetalFailed(_) | ComputeStatus::BadGrid(_)
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
    }
}

#[test]
fn dispatch_missing_texture_fails() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    // Non-Apple metal stubs short-circuit to NoMetal (Linux product).
    let st = execute_dispatch(&mut state, &mut host, 1, &acc, &cmd);
    assert!(matches!(
        st,
        ComputeStatus::MissingPipeline(_)
            | ComputeStatus::MissingTexture(_)
            | ComputeStatus::MetalFailed(_)
            | ComputeStatus::NoMetal(_)
    ));
}

/// Live CI wallpaper: type-5 RefTexture → type-4 surface_id must stage via
/// ensure_surface + mapping (same order as the `runtime::draw` sample). Without
/// ensure, stage fell through to type-2/3 with the type-5 ref → always
/// MissingTexture (`compute_stage_tex … ot=5`).
#[test]
fn stage_texture_type5_ref_resolves_surface_mapping() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    let sid = 3u32;
    let type5_ref = 10u32;
    // Pre-mapped type-4 surface (CI storage target) with one valid page.
    let pfn = 0x20u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0x5a);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    assert!(state.map_surface(sid));
    {
        let m = state.mappings.get_mut(&sid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
    }
    assert!(state.set_mapping_geom(sid, 4, 4, MTL_FORMAT_BGRA8_UNORM));

    // Object-list: type-5 at ref 10 → surface_id=3 (mapping already seeded).
    let desc_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E; // data pfn base 4 + 2
    let type5_desc = Type5Builder::new(sid, 0, 0, 0).with_len(16);
    let type5_desc = type5_desc.bytes();
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, type5_desc);
    let off = list_object_entry_offset(type5_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((16u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    let staged = stage_texture_raw(&mut state, &mut host, 1, type5_ref, 32, true)
        .expect("type-5→surface stage must succeed after ensure");
    assert_eq!((staged.width, staged.height), (4, 4));
    assert_eq!(staged.bytes.len(), 4 * 4 * 4);
    assert!(matches!(
        staged.writeback,
        TextureWriteback::Type11 { mapping_id: 3, .. }
    ));
}

/// A type-5 record is the exact Metal view, even when its single-plane
/// backing already has valid base geometry. Live pipe 5 exposes each row
/// of a 1920-wide BGRA8 surface as a 480-wide RGBA32Uint view so one
/// `uint4` image write stores four packed BGRA pixels.
#[test]
fn stage_texture_type5_record_reshapes_stageable_single_plane_surface() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::{
        StorageImageSelector, MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_R32_UINT, MTL_FORMAT_RGBA32_UINT,
    };
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    let sid = 3u32;
    let type5_ref = 10u32;
    let pfn = 0x20u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0x5a);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    assert!(state.map_surface(sid));
    {
        let m = state.mappings.get_mut(&sid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
    }
    assert!(state.set_mapping_geom(sid, 4, 4, MTL_FORMAT_BGRA8_UNORM));

    // Same 16 bytes per logical row: 4 BGRA8 texels = one RGBA32Uint texel.
    let desc_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E;
    let type5_desc = Type5Builder::new(sid, 0, 10, 0x42)
        .unknown(0x02)
        .geometry(MTL_FORMAT_RGBA32_UINT, 1, 4, 1)
        .trailer([1, 0, 1, 0, 1, 0, 0x10, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let type5_desc = type5_desc.bytes();
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, type5_desc);
    let off = list_object_entry_offset(type5_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((type5_desc.len() as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    let staged = stage_texture_raw(&mut state, &mut host, 1, type5_ref, 33, true)
        .expect("serialized type-5 view must override base surface geometry");
    assert_eq!((staged.width, staged.height), (1, 4));
    assert_eq!(
        staged.storage_selector,
        Some(StorageImageSelector::Rgba32Uint as u32)
    );
    assert_eq!(staged.bytes.len(), 4 * 16);
    assert!(staged.bytes.iter().all(|&b| b == 0x5a));
    match staged.writeback {
        TextureWriteback::Type11 {
            mapping_id,
            surface_bpr,
            width,
            height,
            bpp,
            ..
        } => {
            assert_eq!(mapping_id, sid);
            assert_eq!(surface_bpr, 128);
            assert_eq!((width, height, bpp), (1, 4, 16));
        }
        _ => panic!("expected Type11 writeback through the texture view"),
    }

    // A sampled R32Uint view retains its exact format/geometry. R32Uint is
    // now a storage-capable format (its selector maps to the R32ui storage
    // path), so `storage_selector` is populated — but it is inert here: this
    // view is staged sampled (`is_storage=false`, binding 32), and the
    // selector is only consulted on the storage-bind path.
    let reshaped = Type5Builder::new(sid, 0, 10, 0x42)
        .unknown(0x02)
        .geometry(MTL_FORMAT_R32_UINT, 4, 4, 1)
        .trailer([1, 0, 1, 0, 1, 0, 0x10, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, reshaped.bytes());
    let sampled = stage_texture_raw(&mut state, &mut host, 1, type5_ref, 32, false)
        .expect("sample-only R32Uint view must stage from the same IOSurface bytes");
    assert_eq!((sampled.width, sampled.height), (4, 4));
    assert_eq!(sampled.pixel_format, MTL_FORMAT_R32_UINT);
    assert_eq!(
        sampled.storage_selector,
        Some(StorageImageSelector::R32Uint as u32)
    );
    assert_eq!(sampled.bytes.len(), 4 * 4 * 4);
    assert!(matches!(sampled.writeback, TextureWriteback::None));
}

/// Biplanar surface (device_desc plane_count=2) + type-5 args plane record:
/// stage the named plane view (R8 Y) from the plane offset — live class
/// `compute_dispatch st=Unsupported` / `type11_fail reason=multiplane`
/// (wallpaper '420f', journal 2026-07-14 compute census).
#[test]
fn stage_texture_type5_record_stages_biplanar_y_plane() {
    use crate::contract::endian::{st16, st64};
    use crate::contract::iosurface_pages::{
        DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_LEN, DEVICE_DESC_PLANES, DEVICE_DESC_PLANE_COUNT,
        DEVICE_PLANE_BPE, DEVICE_PLANE_BPR, DEVICE_PLANE_DESC_LEN, DEVICE_PLANE_DIMS,
        DEVICE_PLANE_OFFSET, DEVICE_PLANE_SIZE, PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID,
    };
    use crate::contract::pixel_format::MTL_FORMAT_R8_UNORM;
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};

    let pack_dims = |w: u64, h: u64| ((w & 0xffffff) << 8) | ((h & 0xffffff) << 40);

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    let sid = 3u32;
    let type5_ref = 10u32;
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
        let m = state.mappings.get_mut(&sid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
        m.device_desc = dev;
        m.has_geom = true;
        m.width = 16;
        m.height = 8;
        m.format = 0; // surface-level FourCC has no single MTL format
    }
    assert!(objects::mapping_is_multiplanar(
        state.mappings.get(&sid).unwrap()
    ));

    // Type-5 descriptor: sid + args blob carrying the R8 16×8 plane record.
    let desc_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E;
    // tag, unk, fmt=R8
    let type5_desc = Type5Builder::new(sid, 0, 10, 0x42)
        .unknown(0x01)
        .geometry(0x0a, 16, 8, 1);
    let type5_desc = type5_desc.bytes();
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, type5_desc);
    let off = list_object_entry_offset(type5_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((type5_desc.len() as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    let staged = stage_texture_raw(&mut state, &mut host, 1, type5_ref, 32, true)
        .expect("plane record must stage the Y plane of a biplanar surface");
    assert_eq!((staged.width, staged.height), (16, 8));
    assert_eq!(
        staged.storage_selector,
        Some(crate::contract::pixel_format::StorageImageSelector::R8Unorm as u32)
    );
    assert_eq!(staged.bytes.len(), 16 * 8);
    assert!(staged.bytes.iter().all(|&b| b == 0x77));
    match staged.writeback {
        TextureWriteback::Type11 {
            mapping_id,
            surface_offset,
            surface_bpr,
            ..
        } => {
            assert_eq!(mapping_id, sid);
            assert_eq!(surface_offset, 0);
            assert_eq!(surface_bpr, 64);
        }
        _ => panic!("expected Type11 writeback"),
    }
    let sampled = stage_texture_raw(&mut state, &mut host, 1, type5_ref, 32, false)
        .expect("sampled type-5 plane must stage without writeback");
    assert!(!sampled.is_storage);
    assert!(matches!(sampled.writeback, TextureWriteback::None));
    let _ = MTL_FORMAT_R8_UNORM;
}

/// Biplanar surface **without** a plane record still fails closed
/// (no BGRA invent over multi-plane bytes).
#[test]
fn stage_texture_type5_multiplanar_without_record_fails_closed() {
    use crate::contract::iosurface_pages::{
        DEVICE_DESC_LEN, DEVICE_DESC_PLANE_COUNT, PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID,
    };
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    let sid = 3u32;
    let type5_ref = 10u32;
    let pfn = 0x20u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0x5a);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    let mut dev = vec![0u8; DEVICE_DESC_LEN];
    dev[DEVICE_DESC_PLANE_COUNT] = 2;
    assert!(state.map_surface(sid));
    {
        let m = state.mappings.get_mut(&sid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
        m.device_desc = dev;
        m.has_geom = true;
        m.width = 16;
        m.height = 8;
        m.format = 0;
    }

    // Type-5 descriptor with sid but NO args record.
    let desc_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E;
    let type5_desc = Type5Builder::new(sid, 0, 0, 0).with_len(8);
    let type5_desc = type5_desc.bytes();
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, type5_desc);
    let off = list_object_entry_offset(type5_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((type5_desc.len() as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    match stage_texture_raw(&mut state, &mut host, 1, type5_ref, 32, true) {
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
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
        let m = state.mappings.get_mut(&colliding_mid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
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
    if let Ok(s) = stage_texture_raw(&mut state, &mut host, 1, colliding_mid, 32, true) {
        panic!(
            "linear ref must not stage collided surface mid ({}x{})",
            s.width, s.height
        )
    }
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn stage_heap_texture_uses_host_only_residency_identity() {
    use crate::contract::pixel_format::{StorageImageSelector, MTL_FORMAT_RGBA32_FLOAT};
    use crate::runtime::decode::resource::{
        list_object_entry_offset, HEAP_TEXTURE_DESCRIPTOR, HEAP_TEXTURE_HEAP_REF, HEAP_TEXTURE_LEN,
        HEAP_TEXTURE_OFFSET, HEAP_TEXTURE_OPCODE, HEAP_TEXTURE_USE_OFFSET, OBJECT_LIST_ENTRY_LEN,
        OBJECT_TYPE_TEXTURE_VIEW,
    };

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    let texture_ref = 20u32;
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

    let entry_offset = list_object_entry_offset(texture_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (OBJECT_TYPE_TEXTURE_VIEW as u32) | ((HEAP_TEXTURE_LEN as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], entry_offset, &list_entry);

    let staged = stage_texture_raw(&mut state, &mut host, 1, texture_ref, 33, true)
        .expect("live opcode-0x15 heap texture must stage");
    assert_eq!((staged.width, staged.height), (180, 135));
    assert_eq!(staged.pixel_format, MTL_FORMAT_RGBA32_FLOAT);
    assert_eq!(
        staged.storage_selector,
        Some(StorageImageSelector::Rgba32Float as u32)
    );
    assert_eq!(staged.bytes.len(), 180 * 135 * 16);
    assert!(matches!(staged.writeback, TextureWriteback::None));
    let residency = staged.residency.expect("heap texture needs GPU residency");
    assert!(residency.key.is_heap());
    assert!(!residency.key.is_linear());
    assert_eq!(residency.key.map_generation, 1);
    assert_eq!(residency.key.texture_ref, texture_ref);
    assert_eq!(residency.seed_generation, 0);
}

#[cfg(feature = "backend-vulkan")]
/// UnmapMemory removes the guest page-table alias, not the discrete
/// type-2/3 texture body. Compute writeback must retain raw output, mirror
/// normalized color for render sampling, and complete without attempting
/// a fail-closed write into freed guest pages.
#[test]
fn linear_writeback_retains_cache_when_guest_gva_is_unmapped() {
    use crate::contract::pixel_format::MTL_FORMAT_RGBA8_UNORM;
    use crate::runtime::surface_cache;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let task_id = 6u32;
    let texture_ref = 11u32;
    let gva = 0x101000u64;
    let rgba = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let staged = StagedTexture {
        binding: 32,
        #[cfg(all(feature = "backend-metal", target_os = "macos"))]
        texture_ref: 44,
        pixel_format: MTL_FORMAT_RGBA8_UNORM,
        storage_selector: Some(5),
        width: 2,
        height: 2,
        bytes: rgba.clone(),
        is_storage: true,
        residency: None,
        serve: None,
        writeback: TextureWriteback::Linear {
            pages: Default::default(),
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
        Ok(())
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
        &surface_cache::get_texture(&state, texture_ref, 2, 2).unwrap()[..4],
        &[3, 2, 1, 4],
        "RGBA compute output mirrors into the BGRA render-sample cache"
    );
}

/// No product MiB budget on compute staging — guest size is authoritative.
/// Full-screen wide-gamut (live SkyLight 1928×1920 RGBA16F ≈ 28.2 MiB) must
/// be host-addressable (usize), not rejected by an arbitrary cap.
#[test]
fn compute_stage_admits_full_screen_wide_gamut_without_cap() {
    use crate::contract::pixel_format::{bytes_per_pixel, MTL_FORMAT_RGBA16_FLOAT};
    use crate::runtime::draw::host_alloc_len;
    let bpp = bytes_per_pixel(MTL_FORMAT_RGBA16_FLOAT).expect("rgba16f bpp") as u64;
    let need = 1928u64 * 1920 * bpp;
    assert!(
        host_alloc_len(need).is_some(),
        "full-screen RGBA16Float ({need} bytes) must be host-addressable"
    );
}

/// Type-5 surface id must not be re-resolved through this task's object
/// list: slot `sid` can be a different texture-ref object (id collision).
/// Live class: ensure=1 then MissingTexture when resolve_type11_ref(task,sid)
/// returned the wrong mapping.
#[test]
fn stage_texture_type5_ignores_task_object_list_slot_collision() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    let sid = 3u32;
    let type5_ref = 10u32;
    let pfn = 0x21u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0xa5);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    assert!(state.map_surface(sid));
    {
        let m = state.mappings.get_mut(&sid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
    }
    assert!(state.set_mapping_geom(sid, 4, 4, MTL_FORMAT_BGRA8_UNORM));

    // Poison: object-list slot `sid` is type-11 with mapping_id=99 (not mapped).
    // Pre-fix path would resolve_type11_ref(task, sid) → 99 → MissingTexture.
    let poison_desc_gva = (4u64 + 1) << PAGE_SHIFT_ARM64E;
    let mut iosurf = vec![0u8; 64];
    st32(&mut iosurf[0..], 99); // fake mapping_id
    write_task_gva_arm64e(&mut host, &state.tasks[1], poison_desc_gva, &iosurf);
    let off_sid = list_object_entry_offset(sid, 32).unwrap();
    let mut le_sid = [0u8; OBJECT_LIST_ENTRY_LEN];
    // type-11 = OBJECT_TYPE_IOSURFACE
    let packed_t11 = (OBJECT_TYPE_IOSURFACE as u32) | ((64u32) << 8);
    st32(&mut le_sid[0..], packed_t11);
    le_sid[4..12].copy_from_slice(&poison_desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off_sid, &le_sid);

    // type-5 at ref 10 → surface_id 3
    let desc_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E;
    let type5_desc = Type5Builder::new(sid, 0, 0, 0).with_len(16);
    let type5_desc = type5_desc.bytes();
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, type5_desc);
    let off = list_object_entry_offset(type5_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((16u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    let staged = stage_texture_raw(&mut state, &mut host, 1, type5_ref, 32, true)
        .expect("type-5 must stage mapping sid, not poisoned type-11 slot");
    assert_eq!((staged.width, staged.height), (4, 4));
    assert!(matches!(
        staged.writeback,
        TextureWriteback::Type11 { mapping_id: 3, .. }
    ));
}

/// Type-5 whose surface_id never maps must fail MissingTexture (not pretend
/// type-2/3 success).
#[test]
fn stage_texture_type5_without_surface_is_missing() {
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    let type5_ref = 11u32;
    let sid = 99u32; // no mapping
    let desc_gva = (4u64 + 3) << PAGE_SHIFT_ARM64E;
    let type5_desc = Type5Builder::new(sid, 0, 0, 0).with_len(16);
    let type5_desc = type5_desc.bytes();
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, type5_desc);
    let off = list_object_entry_offset(type5_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((16u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    let st = stage_texture_raw(&mut state, &mut host, 1, type5_ref, 32, false);
    assert!(matches!(st, Err(ComputeStatus::MissingTexture(_))));
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn incomplete_compute_engine_call_fires_stall_proxy() {
    use crate::backend::vulkan::engine::ComputeRequest;
    use std::time::Duration;

    let pipe = 0xf000_0000 | (std::process::id() & 0x0fff_ffff);
    let req = ComputeRequest {
        spirv: vec![0x0723_0203],
        entry: "main".into(),
        grid: [1, 1, 1],
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

#[cfg(feature = "backend-vulkan")]
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

#[cfg(feature = "backend-vulkan")]
#[test]
fn storage_format_specialization_preserves_raw_views_and_runtime_shape() {
    use crate::backend::vulkan::engine::StorageImageFormat as V;
    use crate::runtime::spirv_bind::ImageFormat as S;

    assert_eq!(
        mtl_to_engine_sampled(pixel_format::MTL_FORMAT_R32_UINT),
        Some(V::R32Uint)
    );
    assert_eq!(
        mtl_to_engine_sampled(pixel_format::MTL_FORMAT_RGB9E5_FLOAT),
        Some(V::Rgb9e5Ufloat)
    );

    // A uint shader over a BGRA8Unorm surface is a deliberate raw byte view
    // (byte order preserved) — unaffected by the without-format flag.
    assert_eq!(
        specialized_storage_image_format(V::Bgra8Unorm, S::Rgba8Uint, true),
        Ok(S::Rgba8Uint)
    );
    assert_eq!(
        specialized_storage_image_format(V::Bgra8Unorm, S::Rgba8Uint, false),
        Ok(S::Rgba8Uint)
    );
    assert_eq!(
        specialized_storage_image_format(V::Rgba16Float, S::Rgba32Float, true),
        Ok(S::Rgba16Float)
    );
    // BGRA8Unorm normalized color store (the desktop composite): with the
    // device feature it retargets to a format-less `Unknown` storage image
    // (viewed B8G8R8A8_UNORM — no R/B swap); without it, degrades to the
    // swapped Rgba8Unorm view.
    assert_eq!(
        specialized_storage_image_format(V::Bgra8Unorm, S::Rgba32Float, true),
        Ok(S::Unknown)
    );
    assert_eq!(
        specialized_storage_image_format(V::Bgra8Unorm, S::Rgba32Float, false),
        Ok(S::Rgba8Unorm)
    );
    assert_eq!(
        specialized_storage_image_format(V::Bgra8Unorm, S::Rgba8Unorm, true),
        Ok(S::Unknown)
    );
    assert_eq!(
        specialized_storage_image_format(V::Bgra8Unorm, S::Rgba16Uint, true),
        Err("spirv_guest_numeric_class_mismatch")
    );
    assert_eq!(
        specialized_storage_image_format(V::Bgra8Unorm, S::Unsupported(0), true),
        Err("spirv_storage_format_unsupported")
    );

    // R32Uint storage (the captured VTMTS 4K coverage-buffer case): the
    // guest surface is a single 32-bit uint channel but the translator
    // declares a 4x8-bit `Rgba8ui` write image. Despite equal bytes/texel
    // (4 == 4) this must NOT take the raw-view early return (which would
    // adopt Rgba8ui and keep only the low byte of each written lane) — it
    // specializes the SPIR-V storage image to single-channel `R32ui` so the
    // view is VK_FORMAT_R32_UINT and a written `uint4`.x is the full u32.
    assert_eq!(
        specialized_storage_image_format(V::R32Uint, S::Rgba8Uint, true),
        Ok(S::R32ui)
    );
    assert_eq!(
        specialized_storage_image_format(V::R32Uint, S::Rgba16Uint, true),
        Ok(S::R32ui)
    );
    // A float/unorm-class write shader over an R32Uint surface is a genuine
    // numeric-class mismatch, not a raw view — still rejected.
    assert_eq!(
        specialized_storage_image_format(V::R32Uint, S::Rgba8Unorm, true),
        Err("spirv_guest_numeric_class_mismatch")
    );
    // The R32-single-channel sint/float and packed Rgb9e5 storage paths
    // stay guarded off (no live capture yet justifies enabling them). The
    // guard is UPSTREAM: `storage_selector` returns None for them, so they
    // are rejected at stage time (`stage_tex_fmt_storage`) and never reach
    // the specializer at all — unlike R32Uint, which is now mapped.
    assert_eq!(
        pixel_format::storage_selector(pixel_format::MTL_FORMAT_R32_SINT),
        None
    );
    assert_eq!(
        pixel_format::storage_selector(pixel_format::MTL_FORMAT_R32_FLOAT),
        None
    );

    // The selector/engine plumbing that lets R32Uint reach the specializer:
    // storage_selector now maps 0x35, and both format bridges round-trip it.
    assert_eq!(
        pixel_format::storage_selector(pixel_format::MTL_FORMAT_R32_UINT),
        Some(pixel_format::StorageImageSelector::R32Uint)
    );
    assert_eq!(
        simg_u32_to_engine_storage(pixel_format::StorageImageSelector::R32Uint as u32),
        Some(V::R32Uint)
    );
    assert_eq!(
        spirv_image_format_to_engine_storage(S::R32ui),
        Some(V::R32Uint)
    );
}

/// `metal2vulkan` lowers a generic `texture2d<float, access::write>` to SPIR-V
/// `R32f` (enum value 3). Decoding that as `Unsupported(3)` made the format
/// unspecializable, so every dispatch binding such an image died as
/// `storage_format_specialize_mismatch` — 142 dropped dispatches in one x86
/// desktop boot. It specializes exactly like the `R32ui` case: the declared
/// format is a placeholder, the bound guest surface decides the view.
#[cfg(feature = "backend-vulkan")]
#[test]
fn r32f_write_images_specialize_to_the_bound_guest_surface() {
    use crate::backend::vulkan::engine::StorageImageFormat as V;
    use crate::runtime::spirv_bind::ImageFormat as S;

    assert_eq!(
        spirv_image_format_to_engine_storage(S::R32Float),
        Some(V::R32Float)
    );

    // Wider float surfaces: the placeholder widens to the guest's own format
    // so all four written lanes are stored, not just `.x`.
    assert_eq!(
        specialized_storage_image_format(V::Rgba32Float, S::R32Float, true),
        Ok(S::Rgba32Float)
    );
    assert_eq!(
        specialized_storage_image_format(V::Rgba16Float, S::R32Float, true),
        Ok(S::Rgba16Float)
    );
    // Narrower single-channel float: still class-matched to the guest surface.
    assert_eq!(
        specialized_storage_image_format(V::R16Float, S::R32Float, true),
        Ok(S::R16Float)
    );
    // An R32Float guest surface is an exact match — the raw view is correct.
    assert_eq!(
        specialized_storage_image_format(V::R32Float, S::R32Float, true),
        Ok(S::R32Float)
    );
    // BGRA8Unorm is the desktop composite target. Bytes/texel are equal (4 == 4)
    // so the raw-view early return would view a BGRA surface as a single
    // 32-bit float; a normalized color store must retarget instead.
    assert_eq!(
        specialized_storage_image_format(V::Bgra8Unorm, S::R32Float, true),
        Ok(S::Unknown)
    );
    assert_eq!(
        specialized_storage_image_format(V::Bgra8Unorm, S::R32Float, false),
        Ok(S::Rgba8Unorm)
    );
    // A float-class shader over a uint surface stays a class mismatch.
    assert_eq!(
        specialized_storage_image_format(V::R32Uint, S::R32Float, true),
        Err("spirv_guest_numeric_class_mismatch")
    );
}

/// Equal bytes per texel inside one numeric class is a coincidence, not a raw
/// view, and adopting the shader's placeholder there silently drops channels.
///
/// The guest's decode-time HEIC downsample writes chroma as
/// `OpVectorShuffle … 1 2 1 2` — two live lanes — into an `Rg16Float` surface
/// that `metal2vulkan` declared `R32f`. Both are four float bytes, so a
/// width-only raw-view test kept `R32f`, `OpImageWrite` stored lane `.x` as one
/// f32, and the guest read those four bytes back as two halves: the second
/// chroma channel was never written and the first was destroyed. On screen that
/// is the wallpaper speckle class; measured off-screen it is
/// `.agents/repros/heic-decode-isolation.sh` going from dB 0.23 to dB 7.11
/// between a 1921-wide source and a 1984-wide one.
///
/// The same trap had already been carved out by name for `R32Uint` under
/// `Rgba8Uint`. These are one rule, so the rule is asserted here for every
/// same-class equal-width pair the format table can produce.
#[cfg(feature = "backend-vulkan")]
#[test]
fn same_class_equal_width_storage_formats_take_the_guest_surface_not_the_placeholder() {
    use crate::backend::vulkan::engine::StorageImageFormat as V;
    use crate::runtime::spirv_bind::ImageFormat as S;

    // 4 bytes both sides, float class both sides — the measured case.
    assert_eq!(
        specialized_storage_image_format(V::Rg16Float, S::R32Float, true),
        Ok(S::Rg16Float)
    );
    // Same width and class again, and a colour store: `.x` as one f32 would be
    // three channels short.
    assert_eq!(
        specialized_storage_image_format(V::Rgba8Unorm, S::R32Float, true),
        Ok(S::Rgba8Unorm)
    );
    // 2 bytes both sides.
    assert_eq!(
        specialized_storage_image_format(V::Rg8Unorm, S::R16Float, true),
        Ok(S::Rg8Unorm)
    );
    // The genuine raw view — integer shader over a normalized surface — is not
    // disturbed by narrowing the width test.
    assert_eq!(
        specialized_storage_image_format(V::Rgba8Unorm, S::Rgba8Uint, true),
        Ok(S::Rgba8Uint)
    );
}

/// x86 (12-bit) task page table with depth-1 root; ptes map gva page i →
/// `pfns[i]`. Mirrors the gva_view multi-import fixture.
fn setup_linear_task_x86(host: &mut FakeHost, state: &mut DeviceState, pfns: &[u32]) {
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

#[test]
fn bulk_linear_read_destrides_span_with_one_view() {
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
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
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
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
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
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
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
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
    use crate::contract::endian::st32;
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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

/// The two halves of a resident answer partition; neither rail can see both.
///
/// `StagedTexture` used to carry this enum as a `bool` and an `Option` side by
/// side, rebuilt independently by all three staging rails. Nothing then stopped
/// a rail from setting both — and the two consumers read one field each, so a
/// binding claiming to be seeded *and* sampled would have been dispatched as
/// both a storage seed skip and a copy-on-sample, seeding one image from a
/// placeholder. The state is unrepresentable now; this pins the accessors that
/// replaced the two fields so a later variant cannot answer to both consumers
/// or to neither.
#[test]
fn a_resident_answer_is_a_seed_or_a_sample_and_never_both() {
    let key = crate::model::ComputeStorageResidencyKey {
        mapping_id: 7,
        map_generation: 3,
        surface_offset: 0,
        surface_bpr: 16,
        span_end: 64,
        width: 4,
        height: 4,
        pixel_format: 0x50,
        texture_ref: 9,
    };
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
/// wrote reaches the accumulator. What used to happen next was
/// `if x == CONCURRENT { CONCURRENT } else { SERIAL }`, at the far end of the
/// rail inside `execute_dispatch_metal` — so a guest asking for a dispatch type
/// this device has no contract for got a *serial* encoder, silently, on the one
/// arm that read the field at all.
///
/// Both halves are the test. A substitution nobody can see is the failure this
/// commit exists to end; a line spent on the ordinary `Serial` and `Concurrent`
/// records would be a flood on a per-segment path and would bury the one line
/// that means something.
#[test]
fn an_undeclared_dispatch_type_is_named_and_counted_before_it_becomes_serial() {
    use crate::contract::dispatch::{MTL_DISPATCH_TYPE_CONCURRENT, MTL_DISPATCH_TYPE_SERIAL};
    use crate::runtime::drain::store_route_count;

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

/// A heap texture's residency mirror is never evicted by the per-mapping cap.
///
/// `compute_storage_residency` holds three keyings and only one of them has a
/// guest fallback. A mapping-backed entry dropped by the cap costs the next read
/// its resident and sends it back to that mapping's guest pages. A **heap**
/// texture has no guest pages at all — it is host-only — so an absent entry
/// stages `vec![0; need]` and the kernel reads a blank texture.
///
/// The two are kept apart by `note_storage_residency_writeback` returning before
/// the cap runs, not by the cap's own filter: `ComputeStorageResidencyKey::heap`
/// and `::linear` both set `mapping_id` to 0, so they would share one bucket if
/// the eviction ever saw them. An audit that read the filter alone concluded
/// heap textures were already being evicted into zero-filled binds. It was
/// wrong, and it was wrong by one early return.
///
/// So drive well past `STORAGE_RESIDENCY_WINDOWS_PER_MAPPING` distinct heap
/// textures and assert every one is still there. This fails the moment that
/// early return moves.
#[test]
#[cfg(feature = "backend-vulkan")]
fn a_heap_texture_mirror_outlives_the_per_mapping_cap() {
    use super::{
        note_storage_residency_writeback, ComputeStorageResidencyCandidate, StagedTexture,
        TextureWriteback, STORAGE_RESIDENCY_WINDOWS_PER_MAPPING,
    };
    use crate::model::ComputeStorageResidencyKey;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    // Four times the cap, so this stays a real margin if the cap is retuned.
    const HEAP_TEXTURES: u32 = 4 * STORAGE_RESIDENCY_WINDOWS_PER_MAPPING as u32;

    let staged = |key: ComputeStorageResidencyKey| StagedTexture {
        binding: 33,
        #[cfg(all(feature = "backend-metal", target_os = "macos"))]
        texture_ref: key.texture_ref,
        pixel_format: key.pixel_format,
        storage_selector: Some(0),
        width: key.width,
        height: key.height,
        bytes: Vec::new(),
        // The mirror is only armed for a storage output, which is what makes a
        // heap texture's engine copy the sole content.
        is_storage: true,
        residency: Some(ComputeStorageResidencyCandidate {
            key,
            seed_generation: 1,
        }),
        serve: None,
        writeback: TextureWriteback::None,
    };

    for tex in 0..HEAP_TEXTURES {
        let key = ComputeStorageResidencyKey::heap(1, tex, 16, 16, 0x50);
        assert_eq!(
            key.mapping_id, 0,
            "a heap key sits in the same bucket a linear key does; that is the \
             whole reason this test exists"
        );
        note_storage_residency_writeback(&mut state, &staged(key));
    }

    assert_eq!(
        state.compute_storage_residency.len(),
        HEAP_TEXTURES as usize,
        "every heap texture's mirror is retained; the per-mapping cap must not \
         reach a key whose loss stages a blank texture"
    );
    for tex in 0..HEAP_TEXTURES {
        let key = ComputeStorageResidencyKey::heap(1, tex, 16, 16, 0x50);
        assert!(
            state.compute_storage_residency.contains_key(&key),
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
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);

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
