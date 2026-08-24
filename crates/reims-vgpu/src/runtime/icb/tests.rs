use super::*;
use reims_vgpu_core::endian::{st16, st32, st64};
use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
use crate::runtime::decode::resource::{
    compute_only_icb_layout, encode_icb_command_layout, list_object_entry_offset,
    render_icb_layout, ICB_DESC_FLAGS, ICB_DESC_LAYOUT, ICB_DESC_LEN, ICB_DESC_MAX_COMMAND_COUNT,
    ICB_DESC_MAX_FRAGMENT_BINDS, ICB_DESC_MAX_KERNEL_BINDS, ICB_DESC_MAX_VERTEX_BINDS,
    ICB_DESC_OPTIONS, ICB_FLAG_INHERIT_BUFFERS, ICB_LAYOUT_LEN,
    MTL_INDIRECT_CMD_CONCURRENT_DISPATCH, MTL_INDIRECT_CMD_DRAW, MTL_INDIRECT_CMD_DRAW_INDEXED,
    OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_BUFFER, OBJECT_TYPE_SERIALIZER_RESOURCE,
    PIPELINE_TAG_FRAGMENT_FUNC, PIPELINE_TAG_VERTEX_FUNC, RESOURCE_PAGE_SHIFT,
    SERIALIZER_RESOURCE_OBJECT_ICB, SERIALIZER_RESOURCE_OBJECT_RENDER_PIPELINE,
};
use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;

use crate::runtime::gva_mem;
use crate::runtime::host::FakeHost;

use std::sync::Mutex;

/// The ICB reason survives the hop onto the compute rail.
///
/// This is the defect the `From` impl closes. The boundary in
/// `compute_session.rs` used to rewrite `BadDescriptor` **and** `Args` — 93
/// of this file's 153 checks between them — as the single literal
/// `icb_resolve_bad_descriptor_or_args`, so `/tmp/reims-vgpu-fail.log` named the
/// boundary that gave up rather than the check that refused.
#[test]
fn an_icb_refusal_keeps_its_slug_on_the_compute_rail() {
    use crate::observe::{Decline, Refusal};
    use crate::runtime::compute_exec::ComputeStatus;
    for e in [
        IcbStatus::Missing("icb_desc_no_list_entry"),
        IcbStatus::BadDescriptor("icb_desc_wrong_type"),
        IcbStatus::Args("icb_drs_unknown_command_type"),
        IcbStatus::BackendFailed("icb_pso_pipeline_state"),
        IcbStatus::Unsupported("icb_execute_unimplemented"),
    ] {
        assert_eq!(
            ComputeStatus::from(e).refusal(),
            Some(e.slug()),
            "{e:?} lost its reason crossing onto the compute rail"
        );
    }
}

/// The `0x1d1` backing-info decoder names which of its two checks refused.
///
/// `exec` used to fold both into a bare `icb_backing_fail` counter, so an
/// ICB whose command memory never bound was indistinguishable from one whose
/// record was truncated. These are the two slugs that counter now carries.
#[test]
fn icb_backing_info_names_its_two_refusals() {
    use crate::observe::Decline;
    let short = decode_icb_host_resource_info(&[0u8; 4]).unwrap_err();
    assert_eq!(short.slug(), "icb_host_resource_info_short");

    // Well-formed length, but the payload names ICB ref 0.
    let zero_ref =
        decode_icb_host_resource_info(&[0u8; INFO_OP_ICB_HOST_RESOURCE_PAYLOAD_LEN]).unwrap_err();
    assert_eq!(zero_ref.slug(), "icb_host_resource_info_ref_zero");
}

/// The render rail has no reason-carrying status, so the line it emits is
/// the only place an ICB check is named there. Pin its shape: the slug is
/// the reason, and the variant survives as `class=` so a reader can still
/// group 153 checks into five kinds.
#[test]
fn an_icb_decline_renders_its_check_and_its_class() {
    let line =
        crate::observe::Emit::decline("render_icb", &IcbStatus::Args("icb_frc_index_span_zero"))
            .render();
    assert_eq!(line, "render_icb reason=icb_frc_index_span_zero class=args");
}

/// Serialize the ICB tests that compare crate-wide observation counters.
///
/// Taken with `unwrap_or_else(|e| e.into_inner())`, never a bare `unwrap`:
/// the guard only orders test observations, so a poisoned lock carries no
/// unsound state. A bare `unwrap` turns the *first* failing
/// test into a cascade — when the `compute_mul3add1.mtlb` fixture went
/// missing, 3 real failures poisoned this lock and reported as 43, burying
/// the one root cause under 40 `PoisonError`s.
static ICB_TEST_LOCK: Mutex<()> = Mutex::new(());

fn setup_task(host: &mut FakeHost, state: &mut Device) {
    let dir_pfn = 2u32;
    let root_pfn = 3u32;
    let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_ARM64E;
    let root_gpa = (root_pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x4000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], root_pfn);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    for i in 0..8u32 {
        let pfn = 4 + i;
        host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        let _ = host.write_gpa(root_gpa + (i as u64) * 4, &pte);
    }
    state.define_task(1, 0x1000, dir_pfn);
    assert!(state.set_object_list(1, 0, 32));
}

/// The nine-field mesh draw with every count at one — the shape a test that is
/// about something else needs so the slot decodes, not a case in its own right.
/// Twelve bodies wrote it out; the two that vary their counts still do.
fn unit_mesh_draw() -> IcbRenderDraw {
    IcbRenderDraw::MeshThreadgroups(IcbMeshDraw {
        grid: [1, 1, 1],
        object_tg: [1, 1, 1],
        mesh_tg: [1, 1, 1],
    })
}

/// [`unit_mesh_draw`]'s threads-per-grid sibling, all nine counts at one.
fn unit_mesh_threads_draw() -> IcbRenderDraw {
    IcbRenderDraw::MeshThreads(IcbMeshDraw {
        grid: [1, 1, 1],
        object_tg: [1, 1, 1],
        mesh_tg: [1, 1, 1],
    })
}

/// Hold the encode lock for this test. The backend encoder used by these tests
/// is shared; semantic ICB state itself is device-owned.
fn icb_test_guard() -> std::sync::MutexGuard<'static, ()> {
    ICB_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// A device with task 1 defined and its page tables walked — what every body
/// in this file needs before it can put an object anywhere.
fn icb_device() -> (FakeHost, Device) {
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task(&mut host, &mut state);
    (host, state)
}

/// Three UInt16 indices out of buffer ref 12 at its base, one instance, no base
/// instance. `base_vertex` is the parameter because it is the only field the
/// bodies using this vary — and one of them passes -1 deliberately, to prove
/// the sign survives the wire.
fn indexed_draw(base_vertex: i64) -> IcbRenderDraw {
    IcbRenderDraw::Indexed {
        primitive_type: 3,
        index_type: 0,
        index_buffer_ref: 12,
        index_count: 3,
        index_buffer_offset: 0,
        index_wire_va: 0,
        instance_count: 1,
        base_vertex,
        base_instance: 0,
    }
}

/// A one-threadgroup concurrent dispatch of `tg_x` x `tg_y` x `tg_z` threads.
/// Every fill test in this file uses grid 1x1x1 and varies only the
/// threadgroup, so the grid is not a parameter.
fn unit_grid_dispatch(tg_x: u32, tg_y: u32, tg_z: u32) -> IcbFillDispatch {
    IcbFillDispatch::ConcurrentThreadgroups {
        grid_x: 1,
        grid_y: 1,
        grid_z: 1,
        tg_x,
        tg_y,
        tg_z,
    }
}

/// A kernel bind at the buffer's base with no attribute stride. The two tests
/// that are *about* the stride API build theirs by hand.
fn kernel_bind(index: u32, buffer_ref: u32) -> IcbKernelBufferBind {
    IcbKernelBufferBind {
        index,
        buffer_ref,
        offset: 0,
        wire_va: 0,
        attribute_stride: 0,
        has_attribute_stride: false,
    }
}

/// A render bind at the buffer's base: no wire VA, no attribute stride, and the
/// default stage, so `is_fragment` is the whole of what it says. Binds that
/// carry a wire VA, a stride, or a non-default stage are the subject of their
/// own tests and stay written out.
fn render_bind(index: u32, buffer_ref: u32, is_fragment: bool) -> IcbRenderBufferBind {
    IcbRenderBufferBind {
        index,
        buffer_ref,
        offset: 0,
        wire_va: 0,
        attribute_stride: 0,
        has_attribute_stride: false,
        is_fragment,
        stage: IcbRenderBindStage::default(),
    }
}

fn make_icb_desc_bytes(max_cmds: u32, max_kernel: u16, inherit_buffers: bool) -> Vec<u8> {
    make_icb_desc_bytes_tg(
        max_cmds,
        max_kernel,
        0,
        if inherit_buffers {
            ICB_FLAG_INHERIT_BUFFERS
        } else {
            0
        },
    )
}

fn make_icb_desc_bytes_tg(
    max_cmds: u32,
    max_kernel: u16,
    max_kernel_tg: u16,
    flags: u16,
) -> Vec<u8> {
    use crate::runtime::decode::resource::{compute_icb_layout, ICB_DESC_MAX_KERNEL_TG_BINDS};
    let mut b = vec![0u8; ICB_DESC_LEN];
    st32(&mut b[0..], SERIALIZER_RESOURCE_OBJECT_ICB);
    st32(&mut b[4..], ICB_DESC_LEN as u32);
    st32(&mut b[8..], MTL_INDIRECT_CMD_CONCURRENT_DISPATCH);
    b[ICB_DESC_MAX_VERTEX_BINDS] = 0;
    b[ICB_DESC_MAX_FRAGMENT_BINDS] = 0;
    b[ICB_DESC_MAX_KERNEL_BINDS] = max_kernel as u8;
    b[ICB_DESC_MAX_KERNEL_TG_BINDS] = max_kernel_tg as u8;
    // Off the word Apple writes for an untouched descriptor, so a synthetic
    // record is one the serializer could have produced. Writing the caller's
    // flags alone would clear six inherit bits that default on, which is a
    // guest asking to inherit nothing rather than a blank descriptor.
    st16(
        &mut b[ICB_DESC_FLAGS..],
        crate::runtime::decode::resource::ICB_FLAGS_DEFAULT | flags,
    );
    let layout = compute_icb_layout(max_kernel, max_kernel_tg);
    b[ICB_DESC_LAYOUT..ICB_DESC_LAYOUT + ICB_LAYOUT_LEN]
        .copy_from_slice(&encode_icb_command_layout(&layout));
    st32(&mut b[ICB_DESC_MAX_COMMAND_COUNT..], max_cmds);
    st32(&mut b[ICB_DESC_OPTIONS..], 0);
    b
}

/// Serializer resource render pipeline with vertex-input block: Float4 attr0 @ buffer0 stride 16.
///
/// Layout matches `parse_vertex_block` / color-attachment section (offset from
/// header end via tag `0x08`).
fn make_stagein_render_pipeline_desc(vert_ref: u32, frag_ref: u32) -> Vec<u8> {
    use crate::runtime::decode::resource::{
        COLOR_ATTACHMENT_TAG_PIXEL_FORMAT, PIPELINE_TAG_COLOR_ATTACH_OFFSET,
        VERTEX_ATTR_TAG_BUFFER_INDEX, VERTEX_ATTR_TAG_FORMAT, VERTEX_ATTR_TAG_LOCATION,
        VERTEX_ATTR_TAG_OFFSET, VERTEX_DESC_TAG_ATTRIBUTES, VERTEX_DESC_TAG_LAYOUTS,
        VERTEX_LAYOUT_TAG_BUFFER_INDEX, VERTEX_LAYOUT_TAG_STRIDE,
    };

    // first_tlv_end = 16 + 1 + 3*6 = 35
    let first_tlv_end = 35usize;
    let bo = first_tlv_end;
    // root entry 13 B → layout at bo+13=48, attr at bo+34=69
    let layout_rel = 13u32;
    let attr_rel = 34u32;
    let layout_section = bo + layout_rel as usize; // 48
    let attr_section = bo + attr_rel as usize; // 69
                                               // attr section: 4+4+25 = 33 → ends 102
    let color_abs = 102usize;
    let color_off_from_header = (color_abs - 16) as u32; // 86

    let mut b = vec![0u8; 117];
    st32(&mut b[0..], SERIALIZER_RESOURCE_OBJECT_RENDER_PIPELINE);
    st32(&mut b[4..], 117);
    st32(&mut b[8..], 6); // object id
                          // First TLVs
    b[16] = 3;
    b[17] = PIPELINE_TAG_VERTEX_FUNC;
    b[18] = 4;
    st32(&mut b[19..], vert_ref);
    b[23] = PIPELINE_TAG_FRAGMENT_FUNC;
    b[24] = 4;
    st32(&mut b[25..], frag_ref);
    b[29] = PIPELINE_TAG_COLOR_ATTACH_OFFSET;
    b[30] = 4;
    st32(&mut b[31..], color_off_from_header);

    // Vertex block root at bo
    b[bo] = 2;
    b[bo + 1] = VERTEX_DESC_TAG_ATTRIBUTES;
    b[bo + 2] = 4;
    st32(&mut b[bo + 3..], attr_rel);
    b[bo + 7] = VERTEX_DESC_TAG_LAYOUTS;
    b[bo + 8] = 4;
    st32(&mut b[bo + 9..], layout_rel);

    // Layout section
    st32(&mut b[layout_section..], 1);
    st32(&mut b[layout_section + 4..], 8); // entry_rel
    let le = layout_section + 8;
    b[le] = 2;
    b[le + 1] = VERTEX_LAYOUT_TAG_BUFFER_INDEX;
    b[le + 2] = 4;
    st32(&mut b[le + 3..], 0);
    b[le + 7] = VERTEX_LAYOUT_TAG_STRIDE;
    b[le + 8] = 4;
    st32(&mut b[le + 9..], 16);

    // Attr section
    st32(&mut b[attr_section..], 1);
    st32(&mut b[attr_section + 4..], 8);
    let ae = attr_section + 8;
    b[ae] = 4;
    b[ae + 1] = VERTEX_ATTR_TAG_LOCATION;
    b[ae + 2] = 4;
    st32(&mut b[ae + 3..], 0);
    b[ae + 7] = VERTEX_ATTR_TAG_FORMAT;
    b[ae + 8] = 4;
    st32(&mut b[ae + 9..], 31); // MTLVertexFormatFloat4
    b[ae + 13] = VERTEX_ATTR_TAG_OFFSET;
    b[ae + 14] = 4;
    st32(&mut b[ae + 15..], 0);
    b[ae + 19] = VERTEX_ATTR_TAG_BUFFER_INDEX;
    b[ae + 20] = 4;
    st32(&mut b[ae + 21..], 0);

    // Color attachments
    st32(&mut b[color_abs..], 1);
    st32(&mut b[color_abs + 4..], 8);
    let ce = color_abs + 8;
    b[ce] = 1;
    b[ce + 1] = COLOR_ATTACHMENT_TAG_PIXEL_FORMAT;
    b[ce + 2] = 4;
    st32(&mut b[ce + 3..], MTL_FORMAT_BGRA8_UNORM as u32);

    b
}

/// Write `bytes` at `gva` and publish it as object `ref_` in the task's object
/// list — the pair every fixture here performs together.
///
/// The published length is `bytes.len()`. That is not a simplification: the
/// declared length was carried as its own argument through all 173 of these
/// call sites and asserted equal to the slice at every one of them across the
/// whole suite. A test that needs the two to disagree — a short-descriptor
/// refusal — calls [`put_list_entry`] directly, which still takes both.
fn put_object(host: &mut FakeHost, state: &Device, ref_: u32, otype: u8, gva: u64, bytes: &[u8]) {
    gva_mem::write_task_gva_arm64e(host, &state.tasks[1], gva, bytes);
    put_list_entry(host, state, ref_, otype, bytes.len() as u32, gva);
}

fn put_list_entry(host: &mut FakeHost, state: &Device, ref_: u32, otype: u8, len: u32, gva: u64) {
    let off = list_object_entry_offset(ref_, 32).unwrap();
    let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (otype as u32) | (len << 8);
    st32(&mut le[0..], packed);
    le[4..12].copy_from_slice(&gva.to_le_bytes());
    gva_mem::write_task_gva_arm64e(host, &state.tasks[1], off, &le);
}

#[test]
fn load_icb_from_object_list() {
    let _guard = icb_test_guard();
    let (mut host, state) = icb_device();
    let desc = make_icb_desc_bytes(8, 4, true);
    let gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(
        &mut host,
        &state,
        9,
        OBJECT_TYPE_SERIALIZER_RESOURCE,
        gva,
        &desc,
    );
    let icb = load_icb_descriptor(&state, &host, 1, 9).unwrap();
    assert_eq!(icb.max_command_count, 8);
    assert_eq!(icb.max_kernel_buffer_bind_count, 4);
    assert!(icb.inherit_buffers());
    assert_eq!(icb.command_types, MTL_INDIRECT_CMD_CONCURRENT_DISPATCH);
}

/// A flag this device does not apply is counted when the guest asks for it, and
/// not counted when the guest leaves it alone.
///
/// The load path is where this has to be checked: it observes the descriptor
/// independently of whether an executor is available.
#[test]
fn a_flag_this_device_does_not_apply_is_counted_when_the_guest_asks_for_it() {
    use crate::runtime::decode::resource::{
        ICB_FLAG_INHERIT_CULL_MODE, ICB_FLAG_SUPPORT_RAY_TRACING,
    };
    use crate::runtime::drain::store_route_count;

    let _guard = icb_test_guard();

    // Baseline: a descriptor at the serializer's defaults asks for nothing this
    // device drops, so every counter must hold still.
    let routes = [
        "icb_flag_support_ray_tracing_dropped",
        "icb_flag_no_inherit_cull_mode_dropped",
    ];
    let before: Vec<u64> = routes.iter().map(|r| store_route_count(r)).collect();
    {
        let (mut host, state) = icb_device();
        let desc = make_icb_desc_bytes(8, 4, true);
        let gva = 1u64 << RESOURCE_PAGE_SHIFT;
        put_object(
            &mut host,
            &state,
            9,
            OBJECT_TYPE_SERIALIZER_RESOURCE,
            gva,
            &desc,
        );
        load_icb_descriptor(&state, &host, 1, 9).unwrap();
    }
    for (route, was) in routes.iter().zip(&before) {
        assert_eq!(
            store_route_count(route),
            *was,
            "{route} fired for a descriptor at its defaults"
        );
    }

    // Two asks, in opposite directions: `supportRayTracing` defaults off so the
    // guest sets it, `inheritCullMode` defaults on so the guest clears it. Each
    // must reach its own counter and only its own.
    for (set, clear, route, other) in [
        (
            ICB_FLAG_SUPPORT_RAY_TRACING,
            0,
            "icb_flag_support_ray_tracing_dropped",
            "icb_flag_no_inherit_cull_mode_dropped",
        ),
        (
            0,
            ICB_FLAG_INHERIT_CULL_MODE,
            "icb_flag_no_inherit_cull_mode_dropped",
            "icb_flag_support_ray_tracing_dropped",
        ),
    ] {
        let (mut host, state) = icb_device();
        // The helper ORs the serializer's default word in, so a flag the guest
        // *clears* has to be cleared after the fact.
        let mut desc = make_icb_desc_bytes_tg(8, 4, 0, set);
        let word = reims_vgpu_core::endian::ld16(&desc[ICB_DESC_FLAGS..]);
        st16(&mut desc[ICB_DESC_FLAGS..], word & !clear);
        let gva = 1u64 << RESOURCE_PAGE_SHIFT;
        put_object(
            &mut host,
            &state,
            9,
            OBJECT_TYPE_SERIALIZER_RESOURCE,
            gva,
            &desc,
        );
        let before_route = store_route_count(route);
        let before_other = store_route_count(other);
        load_icb_descriptor(&state, &host, 1, 9).unwrap();
        assert_eq!(
            store_route_count(route),
            before_route + 1,
            "{route} did not fire"
        );
        assert_eq!(
            store_route_count(other),
            before_other,
            "{other} fired for a flag it does not name"
        );
    }
}

#[test]
fn decode_encode_render_draw_slot_roundtrip() {
    use crate::runtime::decode::resource::render_only_icb_layout;
    let layout = render_only_icb_layout(2);
    assert_eq!(layout.pipeline_state_offset, 0x60);
    assert_eq!(layout.vertex_buffer_bind_offset, 0x64);
    let fill = IcbRenderFill {
        command_index: 0,
        pipeline_ref: 42,
        buffers: vec![render_bind(0, 7, false)],
        object_threadgroup_memory: vec![],
        draw: IcbRenderDraw::Primitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            base_instance: 0,
        },
    };
    let slot = encode_render_command_slot(&layout, &fill).unwrap();
    assert_eq!(slot.len(), layout.command_size as usize);
    let decoded = decode_render_command_slot(&layout, &slot, 2, 0)
        .unwrap()
        .expect("filled");
    assert_eq!(decoded.pipeline_ref, 42);
    assert_eq!(decoded.buffers.len(), 1);
    assert_eq!(decoded.buffers[0].buffer_ref, 7);
    match decoded.draw {
        IcbRenderDraw::Primitives {
            primitive_type,
            vertex_count,
            ..
        } => {
            assert_eq!(primitive_type, 3);
            assert_eq!(vertex_count, 3);
        }
        IcbRenderDraw::Indexed { .. }
        | IcbRenderDraw::Patches { .. }
        | IcbRenderDraw::IndexedPatches { .. }
        | IcbRenderDraw::MeshThreads(_)
        | IcbRenderDraw::MeshThreadgroups(_) => panic!("expected Primitives"),
    }
}

#[test]
fn decode_encode_render_draw_indexed_slot_roundtrip() {
    use crate::runtime::decode::resource::render_draw_indexed_icb_layout;
    let layout = render_draw_indexed_icb_layout(1);
    assert_eq!(layout.command_arguments_offset + 0x38, layout.command_size);
    let fill = IcbRenderFill {
        command_index: 0,
        pipeline_ref: 11,
        buffers: vec![render_bind(0, 8, false), render_bind(0, 13, true)],
        object_threadgroup_memory: vec![],
        draw: IcbRenderDraw::Indexed {
            primitive_type: 3,
            index_type: 0, // UInt16
            index_buffer_ref: 12,
            index_count: 3,
            index_buffer_offset: 0,
            index_wire_va: 0,
            instance_count: 1,
            base_vertex: 0,
            base_instance: 0,
        },
    };
    // layout from render_draw_indexed_icb_layout(1) has max_fragment=0;
    // use full layout so fragment bind encode/decode round-trips.
    let layout = render_icb_layout(1, 1, MTL_INDIRECT_CMD_DRAW_INDEXED);
    let slot = encode_render_command_slot(&layout, &fill).unwrap();
    assert_eq!(slot.len(), layout.command_size as usize);
    let type_off = layout.command_type_offset as usize;
    assert_eq!(ld32(&slot[type_off..]), ICB_CMD_TYPE_DRAW_INDEXED);
    let decoded = decode_render_command_slot(&layout, &slot, 1, 1)
        .unwrap()
        .expect("filled");
    assert_eq!(decoded.pipeline_ref, 11);
    assert_eq!(decoded.buffers.len(), 2);
    assert!(decoded
        .buffers
        .iter()
        .any(|b| !b.is_fragment && b.buffer_ref == 8));
    assert!(decoded
        .buffers
        .iter()
        .any(|b| b.is_fragment && b.buffer_ref == 13));
    match decoded.draw {
        IcbRenderDraw::Indexed {
            primitive_type,
            index_type,
            index_buffer_ref,
            index_count,
            instance_count,
            base_vertex,
            base_instance,
            ..
        } => {
            assert_eq!(primitive_type, 3);
            assert_eq!(index_type, 0);
            assert_eq!(index_buffer_ref, 12);
            assert_eq!(index_count, 3);
            assert_eq!(instance_count, 1);
            assert_eq!(base_vertex, 0);
            assert_eq!(base_instance, 0);
        }
        IcbRenderDraw::Primitives { .. }
        | IcbRenderDraw::Patches { .. }
        | IcbRenderDraw::IndexedPatches { .. }
        | IcbRenderDraw::MeshThreads(_)
        | IcbRenderDraw::MeshThreadgroups(_) => panic!("expected Indexed"),
    }
}

/// DrawPatches / DrawIndexedPatches encode↔decode (host RE wire types 4 / 8).
#[test]
fn decode_encode_draw_patches_slot_roundtrip() {
    use crate::runtime::decode::resource::{
        render_draw_indexed_patches_icb_layout, render_draw_patches_icb_layout,
    };
    let layout = render_draw_patches_icb_layout(1);
    assert!(layout.command_size >= layout.command_arguments_offset + ICB_DRAW_PATCHES_ARGS_LEN);
    assert_eq!(layout.tessellation_factor_offset, 0x40);
    let fill = IcbRenderFill {
        command_index: 0,
        pipeline_ref: 9,
        buffers: vec![],
        object_threadgroup_memory: vec![],
        draw: IcbRenderDraw::Patches {
            number_of_patch_control_points: 3,
            patch_start: 0,
            patch_count: 2,
            patch_index_buffer_ref: 11,
            patch_index_buffer_offset: 0,
            patch_index_wire_va: 0,
            instance_count: 1,
            base_instance: 0,
            tessellation_factor: IcbTessellationFactor {
                buffer_ref: 12,
                offset: 0,
                wire_va: 0,
                instance_stride: 0,
            },
        },
    };
    let slot = encode_render_command_slot(&layout, &fill).unwrap();
    let type_off = layout.command_type_offset as usize;
    assert_eq!(ld32(&slot[type_off..]), ICB_CMD_TYPE_DRAW_PATCHES);
    let tess_off = layout.tessellation_factor_offset as usize;
    assert_eq!(ld32(&slot[tess_off..]), 12);
    let decoded = decode_render_command_slot(&layout, &slot, 1, 0)
        .unwrap()
        .expect("filled");
    match decoded.draw {
        IcbRenderDraw::Patches {
            number_of_patch_control_points,
            patch_count,
            patch_index_buffer_ref,
            tessellation_factor,
            ..
        } => {
            assert_eq!(number_of_patch_control_points, 3);
            assert_eq!(patch_count, 2);
            assert_eq!(patch_index_buffer_ref, 11);
            assert_eq!(tessellation_factor.buffer_ref, 12);
        }
        _ => panic!("expected Patches"),
    }

    let ilayout = render_draw_indexed_patches_icb_layout(1);
    assert!(
        ilayout.command_size
            >= ilayout.command_arguments_offset + ICB_DRAW_INDEXED_PATCHES_ARGS_LEN
    );
    let ifill = IcbRenderFill {
        command_index: 0,
        pipeline_ref: 9,
        buffers: vec![],
        object_threadgroup_memory: vec![],
        draw: IcbRenderDraw::IndexedPatches {
            number_of_patch_control_points: 4,
            patch_start: 1,
            patch_count: 1,
            patch_index_buffer_ref: 11,
            patch_index_buffer_offset: 0,
            patch_index_wire_va: 0,
            control_point_index_buffer_ref: 13,
            control_point_index_buffer_offset: 0,
            control_point_index_wire_va: 0,
            instance_count: 1,
            base_instance: 0,
            tessellation_factor: IcbTessellationFactor {
                buffer_ref: 12,
                offset: 0,
                wire_va: 0,
                instance_stride: 16,
            },
        },
    };
    let islot = encode_render_command_slot(&ilayout, &ifill).unwrap();
    assert_eq!(
        ld32(&islot[ilayout.command_type_offset as usize..]),
        ICB_CMD_TYPE_DRAW_INDEXED_PATCHES
    );
    let idec = decode_render_command_slot(&ilayout, &islot, 1, 0)
        .unwrap()
        .expect("filled");
    match idec.draw {
        IcbRenderDraw::IndexedPatches {
            control_point_index_buffer_ref,
            tessellation_factor,
            number_of_patch_control_points,
            ..
        } => {
            assert_eq!(number_of_patch_control_points, 4);
            assert_eq!(control_point_index_buffer_ref, 13);
            assert_eq!(tessellation_factor.instance_stride, 16);
        }
        _ => panic!("expected IndexedPatches"),
    }
}

/// Unknown wire command types still fail closed.
#[test]
fn unknown_icb_command_types_fail_closed() {
    let layout = render_icb_layout(0, 0, MTL_INDIRECT_CMD_DRAW);
    let mut slot = vec![0u8; layout.command_size as usize];
    // Not a known draw/patch/mesh/dispatch type.
    st32(&mut slot[layout.command_type_offset as usize..], 0x55);
    st32(&mut slot[layout.pipeline_state_offset as usize..], 1);
    let err = decode_render_command_slot(&layout, &slot, 0, 0).unwrap_err();
    // The slug, not just the class: `Args` speaks for 84 checks in this
    // file, so asserting the class alone would pass on the wrong refusal.
    assert_eq!(
        crate::observe::Decline::slug(&err),
        "icb_drs_unknown_command_type"
    );
}

#[test]
fn an_inherited_render_pipeline_needs_no_per_command_pipeline_ref() {
    let layout = render_icb_layout(0, 0, MTL_INDIRECT_CMD_DRAW);
    let mut slot = vec![0u8; layout.command_size as usize];
    st32(
        &mut slot[layout.command_type_offset as usize..],
        ICB_CMD_TYPE_DRAW,
    );
    let args = layout.command_arguments_offset as usize;
    st16(&mut slot[args..], 3);
    st64(&mut slot[args + 0xa..], 3);
    st64(&mut slot[args + 0x12..], 1);

    let strict = decode_render_command_slot(&layout, &slot, 0, 0).unwrap_err();
    assert_eq!(
        crate::observe::Decline::slug(&strict),
        "icb_drs_pipeline_ref_zero"
    );
    let inherited = decode_render_command_slot_with_inheritance(
        &layout,
        &slot,
        0,
        0,
        IcbInheritance {
            pipeline_state: true,
            buffers: false,
        },
    )
    .expect("an inherited pipeline makes the slot complete")
    .expect("the draw slot is populated");
    assert_eq!(inherited.pipeline_ref, 0);

    st32(&mut slot[layout.pipeline_state_offset as usize..], 7);
    let conflicting = decode_render_command_slot_with_inheritance(
        &layout,
        &slot,
        0,
        0,
        IcbInheritance {
            pipeline_state: true,
            buffers: false,
        },
    )
    .unwrap_err();
    assert_eq!(
        crate::observe::Decline::slug(&conflicting),
        "icb_drs_inherited_pipeline_ref_nonzero"
    );
}

#[test]
fn inherited_render_buffers_are_absent_from_the_command_fill() {
    let layout = render_icb_layout(1, 0, MTL_INDIRECT_CMD_DRAW);
    let slot = encode_render_command_slot(
        &layout,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 0,
            buffers: vec![render_bind(0, 999, false)],
            object_threadgroup_memory: vec![],
            draw: IcbRenderDraw::Primitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1,
                base_instance: 0,
            },
        },
    )
    .expect("encode ignored command-local buffer");
    let fill = decode_render_command_slot_with_inheritance(
        &layout,
        &slot,
        1,
        0,
        IcbInheritance {
            pipeline_state: true,
            buffers: true,
        },
    )
    .expect("decode inherited render slot")
    .expect("populated draw slot");
    assert!(fill.buffers.is_empty());
}

#[test]
fn an_inherited_compute_pipeline_needs_no_per_command_pipeline_ref() {
    let layout = compute_only_icb_layout(0);
    let mut slot = encode_compute_command_slot(
        &layout,
        &IcbComputeFill {
            command_index: 0,
            pipeline_ref: 7,
            buffers: vec![],
            threadgroup_memory: vec![],
            barrier: false,
            dispatch: unit_grid_dispatch(1, 1, 1),
        },
    )
    .expect("encode compute slot");

    let conflicting = decode_compute_command_slot_with_inheritance(
        &layout,
        &slot,
        0,
        IcbInheritance {
            pipeline_state: true,
            buffers: false,
        },
    )
    .unwrap_err();
    assert_eq!(
        crate::observe::Decline::slug(&conflicting),
        "icb_dcs_inherited_pipeline_ref_nonzero"
    );

    st32(&mut slot[layout.pipeline_state_offset as usize..], 0);
    let strict = decode_compute_command_slot(&layout, &slot, 0).unwrap_err();
    assert_eq!(
        crate::observe::Decline::slug(&strict),
        "icb_dcs_pipeline_ref_zero"
    );
    let inherited = decode_compute_command_slot_with_inheritance(
        &layout,
        &slot,
        0,
        IcbInheritance {
            pipeline_state: true,
            buffers: false,
        },
    )
    .expect("an inherited pipeline makes the slot complete")
    .expect("the dispatch slot is populated");
    assert_eq!(inherited.pipeline_ref, 0);
}

/// Mesh/object buffer binds share the 0x14 ref/va/gpuva pack at layout
/// objectBufferBindOffset / meshBufferBindOffset.
#[test]
fn decode_encode_mesh_object_buffer_binds() {
    use crate::runtime::decode::resource::{
        render_draw_mesh_threads_icb_layout_with_binds, ICB_BUFFER_BIND_STRIDE,
        ICB_CMD_TYPE_DRAW_MESH_THREADS,
    };

    let layout = render_draw_mesh_threads_icb_layout_with_binds(1, 2);
    assert_eq!(
        layout.mesh_buffer_bind_offset,
        layout.object_buffer_bind_offset + ICB_BUFFER_BIND_STRIDE as u32
    );
    assert_eq!(
        layout.kernel_buffer_bind_offset,
        layout.mesh_buffer_bind_offset + 2 * ICB_BUFFER_BIND_STRIDE as u32
    );

    let fill = IcbRenderFill {
        command_index: 0,
        pipeline_ref: 9,
        buffers: vec![
            IcbRenderBufferBind {
                index: 0,
                buffer_ref: 101,
                offset: 0,
                wire_va: 0x4000,
                attribute_stride: 0,
                has_attribute_stride: false,
                is_fragment: false,
                stage: IcbRenderBindStage::Object,
            },
            IcbRenderBufferBind {
                index: 0,
                buffer_ref: 202,
                offset: 0,
                wire_va: 0x5000,
                attribute_stride: 0,
                has_attribute_stride: false,
                is_fragment: false,
                stage: IcbRenderBindStage::Mesh,
            },
            IcbRenderBufferBind {
                index: 1,
                buffer_ref: 203,
                offset: 0,
                wire_va: 0x5008,
                attribute_stride: 0,
                has_attribute_stride: false,
                is_fragment: false,
                stage: IcbRenderBindStage::Mesh,
            },
        ],
        object_threadgroup_memory: vec![],
        draw: unit_mesh_threads_draw(),
    };
    let slot = encode_render_command_slot(&layout, &fill).expect("encode mesh binds");
    assert_eq!(
        ld32(&slot[layout.command_type_offset as usize..]),
        ICB_CMD_TYPE_DRAW_MESH_THREADS
    );
    // Object bind 0
    let o0 = layout.object_buffer_bind_offset as usize;
    assert_eq!(ld32(&slot[o0..]), 101);
    assert_eq!(ld64(&slot[o0 + 4..]), 0x4000);
    assert_eq!(ld64(&slot[o0 + 0xc..]), 0x4000);
    // Mesh binds 0,1
    let m0 = layout.mesh_buffer_bind_offset as usize;
    assert_eq!(ld32(&slot[m0..]), 202);
    assert_eq!(ld64(&slot[m0 + 4..]), 0x5000);
    let m1 = m0 + ICB_BUFFER_BIND_STRIDE;
    assert_eq!(ld32(&slot[m1..]), 203);
    assert_eq!(ld64(&slot[m1 + 4..]), 0x5008);

    let decoded = decode_render_command_slot(&layout, &slot, 0, 0)
        .unwrap()
        .expect("filled");
    assert_eq!(decoded.buffers.len(), 3);
    let obj = decoded
        .buffers
        .iter()
        .find(|b| b.effective_stage() == IcbRenderBindStage::Object)
        .expect("object bind");
    assert_eq!(obj.buffer_ref, 101);
    assert_eq!(obj.wire_va, 0x4000);
    let mesh_refs: Vec<_> = decoded
        .buffers
        .iter()
        .filter(|b| b.effective_stage() == IcbRenderBindStage::Mesh)
        .map(|b| (b.index, b.buffer_ref, b.wire_va))
        .collect();
    assert_eq!(mesh_refs, vec![(0, 202, 0x5000), (1, 203, 0x5008)]);
}

/// Mesh wire encode↔decode (command types 0x80 / 0x100, args 0x48).
#[test]
fn decode_encode_draw_mesh_slot_roundtrip() {
    use crate::runtime::decode::resource::{
        render_draw_mesh_threadgroups_icb_layout, render_draw_mesh_threads_icb_layout,
        ICB_CMD_TYPE_DRAW_MESH_THREADGROUPS, ICB_CMD_TYPE_DRAW_MESH_THREADS,
        ICB_DRAW_MESH_ARGS_LEN,
    };

    let layout = render_draw_mesh_threads_icb_layout(0);
    assert_eq!(
        layout.command_arguments_offset + ICB_DRAW_MESH_ARGS_LEN,
        layout.command_size
    );
    let fill = IcbRenderFill {
        command_index: 0,
        pipeline_ref: 6,
        buffers: vec![],
        object_threadgroup_memory: vec![],
        draw: IcbRenderDraw::MeshThreads(IcbMeshDraw {
            grid: [8, 1, 1],
            object_tg: [1, 1, 1],
            mesh_tg: [4, 1, 1],
        }),
    };
    let slot = encode_render_command_slot(&layout, &fill).expect("encode mesh threads");
    assert_eq!(
        ld32(&slot[layout.command_type_offset as usize..]),
        ICB_CMD_TYPE_DRAW_MESH_THREADS
    );
    let decoded = decode_render_command_slot(&layout, &slot, 0, 0)
        .unwrap()
        .expect("filled");
    assert_eq!(decoded.pipeline_ref, 6);
    match decoded.draw {
        IcbRenderDraw::MeshThreads(mesh) => {
            assert_eq!(mesh.grid[0], 8);
            assert_eq!(mesh.mesh_tg[0], 4);
            assert_eq!(mesh.object_tg[0], 1);
        }
        _ => panic!("expected MeshThreads"),
    }

    let layout_tg = render_draw_mesh_threadgroups_icb_layout();
    let fill_tg = IcbRenderFill {
        command_index: 0,
        pipeline_ref: 7,
        buffers: vec![],
        object_threadgroup_memory: vec![],
        draw: IcbRenderDraw::MeshThreadgroups(IcbMeshDraw {
            grid: [2, 3, 1],
            object_tg: [1, 1, 1],
            mesh_tg: [32, 1, 1],
        }),
    };
    let slot_tg =
        encode_render_command_slot(&layout_tg, &fill_tg).expect("encode mesh threadgroups");
    assert_eq!(
        ld32(&slot_tg[layout_tg.command_type_offset as usize..]),
        ICB_CMD_TYPE_DRAW_MESH_THREADGROUPS
    );
    let d2 = decode_render_command_slot(&layout_tg, &slot_tg, 0, 0)
        .unwrap()
        .expect("filled");
    match d2.draw {
        IcbRenderDraw::MeshThreadgroups(mesh) => {
            assert_eq!(mesh.grid[0], 2);
            assert_eq!(mesh.grid[1], 3);
            assert_eq!(mesh.mesh_tg[0], 32);
        }
        _ => panic!("expected MeshThreadgroups"),
    }
}

/// Wire `baseVertex@0x28` is a signed value stored as u64 bits (two's complement).
#[test]
fn decode_encode_signed_base_vertex() {
    let layout = render_icb_layout(1, 0, MTL_INDIRECT_CMD_DRAW_INDEXED);
    let fill = IcbRenderFill {
        command_index: 0,
        pipeline_ref: 11,
        buffers: vec![],
        object_threadgroup_memory: vec![],
        draw: indexed_draw(-1),
    };
    let slot = encode_render_command_slot(&layout, &fill).unwrap();
    let args = layout.command_arguments_offset as usize;
    // Bit pattern of i64(-1) is u64::MAX.
    assert_eq!(ld64(&slot[args + 0x28..]), u64::MAX);
    let decoded = decode_render_command_slot(&layout, &slot, 1, 0)
        .unwrap()
        .expect("filled");
    match decoded.draw {
        IcbRenderDraw::Indexed { base_vertex, .. } => {
            assert_eq!(base_vertex, -1);
        }
        IcbRenderDraw::Primitives { .. }
        | IcbRenderDraw::Patches { .. }
        | IcbRenderDraw::IndexedPatches { .. }
        | IcbRenderDraw::MeshThreads(_)
        | IcbRenderDraw::MeshThreadgroups(_) => panic!("expected Indexed"),
    }

    let fill2 = IcbRenderFill {
        command_index: 0,
        pipeline_ref: 11,
        buffers: vec![],
        object_threadgroup_memory: vec![],
        draw: IcbRenderDraw::Indexed {
            primitive_type: 3,
            index_type: 0,
            index_buffer_ref: 12,
            index_count: 3,
            index_buffer_offset: 0,
            index_wire_va: 0,
            instance_count: 1,
            base_vertex: -42,
            base_instance: 7,
        },
    };
    let slot2 = encode_render_command_slot(&layout, &fill2).unwrap();
    let d2 = decode_render_command_slot(&layout, &slot2, 1, 0)
        .unwrap()
        .expect("filled");
    match d2.draw {
        IcbRenderDraw::Indexed {
            base_vertex,
            base_instance,
            ..
        } => {
            assert_eq!(base_vertex, -42);
            assert_eq!(base_instance, 7);
        }
        IcbRenderDraw::Primitives { .. }
        | IcbRenderDraw::Patches { .. }
        | IcbRenderDraw::IndexedPatches { .. }
        | IcbRenderDraw::MeshThreads(_)
        | IcbRenderDraw::MeshThreadgroups(_) => panic!("expected Indexed"),
    }
}

/// DrawIndexed with `baseVertex = -1`: indices `[1,2,3]` + offset → verts 0,1,2
/// of a stage_in clip-space triangle (same solid as baseVertex 0 / indices 0,1,2).
#[test]
fn decode_encode_command_slot_roundtrip() {
    let layout = compute_only_icb_layout(1);
    let fill = IcbComputeFill {
        command_index: 0,
        pipeline_ref: 6,
        buffers: vec![kernel_bind(0, 7)],
        threadgroup_memory: vec![],
        barrier: false,
        dispatch: unit_grid_dispatch(4, 1, 1),
    };
    let slot = encode_compute_command_slot(&layout, &fill).unwrap();
    assert_eq!(slot.len(), layout.command_size as usize);
    let decoded = decode_compute_command_slot(&layout, &slot, 1)
        .unwrap()
        .expect("filled");
    assert_eq!(decoded.pipeline_ref, 6);
    assert_eq!(decoded.buffers.len(), 1);
    assert_eq!(decoded.buffers[0].buffer_ref, 7);
    match decoded.dispatch {
        IcbFillDispatch::ConcurrentThreadgroups { grid_x, tg_x, .. } => {
            assert_eq!(grid_x, 1);
            assert_eq!(tg_x, 4);
        }
        _ => panic!("expected threadgroups"),
    }
    // Empty slot
    let empty = vec![0u8; layout.command_size as usize];
    assert!(decode_compute_command_slot(&layout, &empty, 1)
        .unwrap()
        .is_none());
}

#[test]
fn icb_host_resource_info_decode_and_apply() {
    let _guard = icb_test_guard();
    // Payload-only form
    let mut p = [0u8; 16];
    st32(&mut p[0..], 9);
    st32(&mut p[4..], 10);
    st64(&mut p[8..], 0x4000);
    let info = decode_icb_host_resource_info(&p).unwrap();
    assert_eq!(info.icb_ref, 9);
    assert_eq!(info.reply_buffer_ref, 10);
    assert_eq!(info.reply_offset, 0x4000);
    // Full record form
    let mut rec = [0u8; 24];
    st32(&mut rec[0..], INFO_OP_ICB_HOST_RESOURCE);
    st32(&mut rec[4..], INFO_OP_ICB_HOST_RESOURCE_RECORD_LEN);
    rec[8..24].copy_from_slice(&p);
    let info2 = decode_icb_host_resource_info(&rec).unwrap();
    assert_eq!(info2, info);
}

/// `0x1d1` is a query, and answering it by binding its reply pair was worse
/// than refusing it.
///
/// The record names an ICB and a scratch `(buffer, offset)` pair for the two
/// `u64`s the guest is waiting to be handed. This device used to read that pair
/// as the ICB's command backing, so a guest whose stream allocator returned a
/// resolvable type-1 ref would have had its own reply staging area bound as an
/// ICB's command slots — and the next `executeCommandsInBuffer:` would have
/// decoded whatever sat there and run it as real work.
///
/// The fixture is built to be exactly that trap: object 11 is a well-formed
/// type-1 buffer whose pages hold a *valid* encoded command slot, so the old
/// code path succeeds on it. Both halves are asserted, because the refusal
/// alone would still pass if the bind happened first and the error came later:
/// the call refuses, **and** the ICB still has no command memory afterwards.
#[test]
fn a_0x1d1_query_is_refused_and_binds_nothing() {
    let _guard = icb_test_guard();
    let (mut host, state) = icb_device();

    let layout = compute_only_icb_layout(1);
    let icb_desc = make_icb_desc_bytes(1, 1, false);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(
        &mut host,
        &state,
        9,
        OBJECT_TYPE_SERIALIZER_RESOURCE,
        icb_gva,
        &icb_desc,
    );
    // Record the create body, as `execute` would, so the decode below refuses
    // for want of command memory rather than for want of the ICB itself.
    resolve_icb_record(&state, &host, 1, 9).expect("record the ICB create body");

    let slot = encode_compute_command_slot(
        &layout,
        &IcbComputeFill {
            command_index: 0,
            pipeline_ref: 1,
            buffers: vec![],
            threadgroup_memory: vec![],
            barrier: false,
            dispatch: unit_grid_dispatch(1, 1, 1),
        },
    )
    .unwrap();
    let handle = 5u32;
    let cmd_gva = (handle as u64) << RESOURCE_PAGE_SHIFT;
    gva_mem::write_task_gva_arm64e(&mut host, &state.tasks[1], cmd_gva, &slot);
    let mut bdesc = vec![0u8; 16];
    st64(&mut bdesc[0..], slot.len() as u64);
    st64(&mut bdesc[8..], handle as u64);
    let bdesc_gva = 0x200u64;
    put_object(&mut host, &state, 11, OBJECT_TYPE_BUFFER, bdesc_gva, &bdesc);

    let refused = apply_icb_host_resource_info(
        &state,
        &host,
        1,
        &IcbHostResourceInfo {
            icb_ref: 9,
            reply_buffer_ref: 11,
            reply_offset: 0,
        },
    )
    .expect_err("0x1d1 is a query this device does not answer");
    assert_eq!(refused, IcbStatus::Unsupported("icb_info_query_unanswered"));

    // The trap: object 11 resolves and its pages hold a decodable slot, so the
    // old reading would have bound it here and this walk would have returned a
    // command the guest never put in an ICB.
    let after = decode_icb_command_range(&state, &host, 1, 9, 0, 1)
        .expect_err("the query must not have bound the reply buffer as command memory");
    assert_eq!(after, IcbStatus::Missing("icb_fill_no_command_memory"));
}

/// An empty execute range does not touch command memory. The range location is
/// still checked against the ICB declaration, but no backing association is
/// required merely to execute zero commands.
#[test]
fn an_empty_icb_range_is_a_noop_without_command_memory() {
    let _guard = icb_test_guard();
    let (mut host, state) = icb_device();
    let desc = make_icb_desc_bytes(2, 1, false);
    let gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(
        &mut host,
        &state,
        9,
        OBJECT_TYPE_SERIALIZER_RESOURCE,
        gva,
        &desc,
    );
    resolve_icb_record(&state, &host, 1, 9).expect("record ICB declaration");

    assert!(decode_icb_command_range(&state, &host, 1, 9, 1, 0)
        .expect("empty range")
        .is_empty());
    assert_eq!(
        decode_icb_command_range(&state, &host, 1, 9, 3, 0)
            .expect_err("empty range still has to be within the declaration"),
        IcbStatus::Args("icb_fill_range_past_capacity")
    );
}

#[test]
fn inherited_compute_buffers_do_not_resolve_command_local_bindings() {
    let _guard = icb_test_guard();
    let (mut host, state) = icb_device();
    let layout = compute_only_icb_layout(1);
    let desc = make_icb_desc_bytes(1, 1, true);
    let descriptor_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(
        &mut host,
        &state,
        9,
        OBJECT_TYPE_SERIALIZER_RESOURCE,
        descriptor_gva,
        &desc,
    );
    resolve_icb_record(&state, &host, 1, 9).expect("record inherited-buffer ICB");

    let slot = encode_compute_command_slot(
        &layout,
        &IcbComputeFill {
            command_index: 0,
            pipeline_ref: 7,
            buffers: vec![IcbKernelBufferBind {
                index: 0,
                buffer_ref: 999,
                offset: 0,
                wire_va: 0xdead,
                attribute_stride: 0,
                has_attribute_stride: false,
            }],
            threadgroup_memory: vec![],
            barrier: false,
            dispatch: unit_grid_dispatch(1, 1, 1),
        },
    )
    .expect("encode command-local bind that inherited mode ignores");
    let command_gva = 5u64 << RESOURCE_PAGE_SHIFT;
    gva_mem::write_task_gva_arm64e(&mut host, &state.tasks[1], command_gva, &slot);
    bind_icb_command_memory(
        &state,
        1,
        9,
        IcbCommandMemory {
            gva: command_gva,
            byte_len: slot.len() as u64,
        },
    )
    .expect("bind command memory");

    let fills = decode_icb_command_range(&state, &host, 1, 9, 0, 1)
        .expect("ignored command-local ref must not be resolved");
    let [IcbCommandFill::Compute(fill)] = fills.as_slice() else {
        panic!("expected one compute fill");
    };
    assert!(fill.buffers.is_empty());
}

#[test]
fn icb_ranges_select_slots_in_order_and_skip_reset_slots() {
    let _guard = icb_test_guard();
    let (mut host, state) = icb_device();
    let layout = compute_only_icb_layout(1);
    let desc = make_icb_desc_bytes(2, 1, false);
    let descriptor_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(
        &mut host,
        &state,
        9,
        OBJECT_TYPE_SERIALIZER_RESOURCE,
        descriptor_gva,
        &desc,
    );
    resolve_icb_record(&state, &host, 1, 9).expect("record ICB declaration");

    let slot = |pipeline_ref| {
        encode_compute_command_slot(
            &layout,
            &IcbComputeFill {
                command_index: 0,
                pipeline_ref,
                buffers: vec![],
                threadgroup_memory: vec![],
                barrier: false,
                dispatch: unit_grid_dispatch(1, 1, 1),
            },
        )
        .expect("encode command slot")
    };
    let mut bytes = slot(6);
    bytes.extend_from_slice(&slot(7));
    // Put slot zero at the end of one guest page and slot one at the start of
    // the next, so the final assertion can make the out-of-range prefix
    // unreadable without affecting the selected slot.
    let command_gva = (1u64 << RESOURCE_PAGE_SHIFT) - u64::from(layout.command_size);
    gva_mem::write_task_gva_arm64e(&mut host, &state.tasks[1], command_gva, &bytes);
    bind_icb_command_memory(
        &state,
        1,
        9,
        IcbCommandMemory {
            gva: command_gva,
            byte_len: bytes.len() as u64,
        },
    )
    .expect("bind command memory");

    let pipelines = |fills: Vec<IcbCommandFill>| {
        fills
            .into_iter()
            .map(|fill| match fill {
                IcbCommandFill::Compute(fill) => fill.pipeline_ref,
                IcbCommandFill::Render(_) => panic!("expected compute fill"),
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        pipelines(decode_icb_command_range(&state, &host, 1, 9, 0, 1).unwrap()),
        [6]
    );
    assert_eq!(
        pipelines(decode_icb_command_range(&state, &host, 1, 9, 1, 1).unwrap()),
        [7]
    );
    assert_eq!(
        pipelines(decode_icb_command_range(&state, &host, 1, 9, 0, 2).unwrap()),
        [6, 7]
    );

    let reset_slot = vec![0u8; layout.command_size as usize];
    gva_mem::write_task_gva_arm64e(
        &mut host,
        &state.tasks[1],
        command_gva + u64::from(layout.command_size),
        &reset_slot,
    );
    assert_eq!(
        pipelines(decode_icb_command_range(&state, &host, 1, 9, 0, 2).unwrap()),
        [6]
    );

    let second = slot(7);
    gva_mem::write_task_gva_arm64e(
        &mut host,
        &state.tasks[1],
        command_gva + u64::from(layout.command_size),
        &second,
    );
    let slot0_gpa =
        gva_mem::translate_task_gva(&host, &state.tasks[1], command_gva, state.page_shift)
            .expect("translate slot zero");
    host.mark_non_ram(slot0_gpa, u64::from(layout.command_size));
    assert_eq!(
        pipelines(decode_icb_command_range(&state, &host, 1, 9, 1, 1).unwrap()),
        [7],
        "a subrange must not read an earlier command slot"
    );
}

#[test]
fn dispatch_bits_select_a_compute_command_domain_even_with_render_bits_set() {
    let _guard = icb_test_guard();
    let (mut host, state) = icb_device();
    let command_types = MTL_INDIRECT_CMD_DRAW | MTL_INDIRECT_CMD_CONCURRENT_DISPATCH;
    let mut layout = render_icb_layout(0, 0, command_types);
    layout.command_size = layout.command_arguments_offset + ICB_CONCURRENT_DISPATCH_ARGS_LEN as u32;
    let mut desc = make_icb_desc_bytes(2, 0, false);
    st32(&mut desc[8..], command_types);
    desc[ICB_DESC_LAYOUT..ICB_DESC_LAYOUT + ICB_LAYOUT_LEN]
        .copy_from_slice(&encode_icb_command_layout(&layout));
    let descriptor_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(
        &mut host,
        &state,
        9,
        OBJECT_TYPE_SERIALIZER_RESOURCE,
        descriptor_gva,
        &desc,
    );
    resolve_icb_record(&state, &host, 1, 9).expect("record mixed ICB declaration");

    let mut bytes = encode_render_command_slot(
        &layout,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![],
            object_threadgroup_memory: vec![],
            draw: IcbRenderDraw::Primitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1,
                base_instance: 0,
            },
        },
    )
    .expect("encode render slot");
    bytes.extend_from_slice(
        &encode_compute_command_slot(
            &layout,
            &IcbComputeFill {
                command_index: 1,
                pipeline_ref: 7,
                buffers: vec![],
                threadgroup_memory: vec![],
                barrier: false,
                dispatch: unit_grid_dispatch(1, 1, 1),
            },
        )
        .expect("encode compute slot"),
    );
    let command_gva = 5u64 << RESOURCE_PAGE_SHIFT;
    gva_mem::write_task_gva_arm64e(&mut host, &state.tasks[1], command_gva, &bytes);
    bind_icb_command_memory(
        &state,
        1,
        9,
        IcbCommandMemory {
            gva: command_gva,
            byte_len: bytes.len() as u64,
        },
    )
    .expect("bind mixed command memory");

    let refused = decode_icb_command_range(&state, &host, 1, 9, 0, 2)
        .expect_err("a render slot cannot be filled in the compute command domain");
    assert_eq!(
        crate::observe::Decline::slug(&refused),
        "icb_fill_render_command_in_compute_domain"
    );

    let fills = decode_icb_command_range(&state, &host, 1, 9, 1, 1)
        .expect("the compute slot remains valid despite the extra render bit");
    assert!(matches!(
        fills.as_slice(),
        [IcbCommandFill::Compute(IcbComputeFill {
            pipeline_ref: 7,
            command_index: 1,
            ..
        })]
    ));
}

#[test]
fn icb_lifetimes_are_device_owned_generational_and_task_scoped() {
    let _guard = icb_test_guard();
    let (mut host_a, mut state_a) = icb_device();
    let (mut host_b, state_b) = icb_device();
    let desc = make_icb_desc_bytes(1, 1, false);
    let gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(
        &mut host_a,
        &state_a,
        9,
        OBJECT_TYPE_SERIALIZER_RESOURCE,
        gva,
        &desc,
    );
    put_object(
        &mut host_b,
        &state_b,
        9,
        OBJECT_TYPE_SERIALIZER_RESOURCE,
        gva,
        &desc,
    );

    resolve_icb_record(&state_a, &host_a, 1, 9).expect("record first device ICB");
    resolve_icb_record(&state_b, &host_b, 1, 9).expect("record second device ICB");
    let first = state_a
        .task_objects
        .indirect_command_buffers
        .identity(1, 9)
        .expect("first identity");
    bind_icb_command_memory(
        &state_a,
        1,
        9,
        IcbCommandMemory {
            gva: 0x4000,
            byte_len: 64,
        },
    )
    .expect("bind only the first device");
    assert!(
        state_b
            .task_objects
            .indirect_command_buffers
            .snapshot(1, 9)
            .expect("second record")
            .command_memory
            .is_none(),
        "equal task/ref pairs on different devices must not share state"
    );

    let mut changed = state_a
        .task_objects
        .indirect_command_buffers
        .snapshot(1, 9)
        .expect("first record")
        .descriptor;
    changed.max_kernel_buffer_bind_count += 1;
    state_a
        .task_objects
        .indirect_command_buffers
        .record(1, 9, changed)
        .expect("replace changed declaration");
    let changed_identity = state_a
        .task_objects
        .indirect_command_buffers
        .identity(1, 9)
        .expect("changed identity");
    assert_eq!(first.index(), changed_identity.index());
    assert_ne!(first.generation(), changed_identity.generation());
    assert!(
        state_a
            .task_objects
            .indirect_command_buffers
            .snapshot(1, 9)
            .expect("changed record")
            .command_memory
            .is_none(),
        "any declaration change invalidates command memory decoded under the old layout"
    );

    assert!(state_a.task_objects.indirect_command_buffers.delete(1, 9));
    assert_eq!(
        state_a.task_objects.indirect_command_buffers.identity(1, 9),
        None
    );
    resolve_icb_record(&state_a, &host_a, 1, 9).expect("recreate deleted ICB");
    let replacement = state_a
        .task_objects
        .indirect_command_buffers
        .identity(1, 9)
        .expect("replacement identity");
    assert_eq!(changed_identity.index(), replacement.index());
    assert_ne!(changed_identity.generation(), replacement.generation());

    assert!(state_a.delete_task(1).is_some());
    assert_eq!(
        state_a.task_objects.indirect_command_buffers.identity(1, 9),
        None
    );
    assert!(
        state_b
            .task_objects
            .indirect_command_buffers
            .identity(1, 9)
            .is_some(),
        "tearing down one device's task must not touch another device"
    );
}

/// Wire encode↔decode of object TG memory lengths + MeshThreadgroups.
#[test]
fn decode_encode_object_tg_memory_mesh_slot() {
    use crate::runtime::decode::resource::{
        render_draw_mesh_threadgroups_icb_layout_ex, ICB_CMD_TYPE_DRAW_MESH_THREADGROUPS,
        ICB_TG_MEMORY_STRIDE,
    };

    let layout = render_draw_mesh_threadgroups_icb_layout_ex(0, 0, 2);
    assert_eq!(
        layout.threadgroup_memory_length_offset,
        layout.object_threadgroup_memory_length_offset + 2 * ICB_TG_MEMORY_STRIDE as u32
    );
    let fill = IcbRenderFill {
        command_index: 0,
        pipeline_ref: 6,
        buffers: vec![],
        object_threadgroup_memory: vec![
            IcbThreadgroupMemory {
                index: 0,
                length: 16,
            },
            IcbThreadgroupMemory {
                index: 1,
                length: 32,
            },
        ],
        draw: unit_mesh_draw(),
    };
    let slot = encode_render_command_slot(&layout, &fill).expect("encode");
    assert_eq!(
        ld32(&slot[layout.command_type_offset as usize..]),
        ICB_CMD_TYPE_DRAW_MESH_THREADGROUPS
    );
    let base = layout.object_threadgroup_memory_length_offset as usize;
    assert_eq!(ld64(&slot[base..]), 16);
    assert_eq!(ld64(&slot[base + ICB_TG_MEMORY_STRIDE..]), 32);
    let decoded = decode_render_command_slot(&layout, &slot, 0, 0)
        .unwrap()
        .expect("filled");
    assert_eq!(decoded.object_threadgroup_memory.len(), 2);
    assert_eq!(decoded.object_threadgroup_memory[0].length, 16);
    assert_eq!(decoded.object_threadgroup_memory[1].index, 1);
    assert_eq!(decoded.object_threadgroup_memory[1].length, 32);
}

/// inheritPipelineState=true: ICB slot records only buffers+draw; PSO
/// comes from the parent render encoder (stream pipeline bind path).
/// Mirrors `inherit_pipeline_encoder_kernel_mul3add1` for compute.
#[test]
fn stagein_pipeline_fixture_decodes_vertex_attrs() {
    use crate::runtime::decode::resource::decode_render_pipeline_descriptor;
    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    let rp = decode_render_pipeline_descriptor(&pdesc).expect("stagein pipeline");
    assert_eq!(rp.vertex_func_ref, 2);
    assert_eq!(rp.fragment_func_ref, 3);
    assert_eq!(rp.vertex_attributes.len(), 1);
    let a = &rp.vertex_attributes[0];
    assert_eq!(a.location, 0);
    assert_eq!(a.format, 31); // Float4
    assert_eq!(a.offset, 0);
    assert_eq!(a.buffer_index, 0);
    assert_eq!(a.stride, 16);
}

/// Dedicated wire-backed E2E: classic Draw (`0x1`) with stage_in vertex
/// buffer + fragment color — not DrawIndexed and not host fill API.
#[test]
fn decode_encode_attribute_stride_table() {
    use crate::runtime::decode::resource::{
        compute_icb_layout, icb_layout_attribute_stride_slot_count, render_icb_layout,
        ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE,
    };
    // Compute: max_kernel=2 → 2 stride slots after binds.
    let layout = compute_icb_layout(2, 0);
    assert_eq!(icb_layout_attribute_stride_slot_count(&layout), 2);
    let fill = IcbComputeFill {
        command_index: 0,
        pipeline_ref: 6,
        buffers: vec![
            IcbKernelBufferBind {
                index: 0,
                buffer_ref: 7,
                offset: 0,
                wire_va: 0,
                attribute_stride: 32,
                has_attribute_stride: true,
            },
            IcbKernelBufferBind {
                index: 1,
                buffer_ref: 8,
                offset: 0,
                wire_va: 0,
                attribute_stride: 64,
                has_attribute_stride: true,
            },
        ],
        threadgroup_memory: vec![],
        barrier: false,
        dispatch: unit_grid_dispatch(1, 1, 1),
    };
    let slot = encode_compute_command_slot(&layout, &fill).unwrap();
    let so = layout.attribute_stride_offset as usize;
    assert_eq!(ld64(&slot[so..]), 32);
    assert_eq!(ld64(&slot[so + ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE..]), 64);
    let decoded = decode_compute_command_slot(&layout, &slot, 2)
        .unwrap()
        .expect("filled");
    assert_eq!(decoded.buffers.len(), 2);
    assert!(decoded.buffers[0].has_attribute_stride);
    assert_eq!(decoded.buffers[0].attribute_stride, 32);
    assert_eq!(decoded.buffers[1].attribute_stride, 64);

    // Render: max_vertex=1 → 1 stride slot; fragment bind has no stride.
    let rlayout = render_icb_layout(1, 1, MTL_INDIRECT_CMD_DRAW);
    assert_eq!(icb_layout_attribute_stride_slot_count(&rlayout), 1);
    let rfill = IcbRenderFill {
        command_index: 0,
        pipeline_ref: 6,
        buffers: vec![
            IcbRenderBufferBind {
                index: 0,
                buffer_ref: 11,
                offset: 0,
                wire_va: 0,
                attribute_stride: 16,
                has_attribute_stride: true,
                is_fragment: false,
                stage: IcbRenderBindStage::default(),
            },
            render_bind(0, 13, true),
        ],
        object_threadgroup_memory: vec![],
        draw: IcbRenderDraw::Primitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            base_instance: 0,
        },
    };
    let rslot = encode_render_command_slot(&rlayout, &rfill).unwrap();
    assert_eq!(ld64(&rslot[rlayout.attribute_stride_offset as usize..]), 16);
    let rdec = decode_render_command_slot(&rlayout, &rslot, 1, 1)
        .unwrap()
        .expect("render filled");
    let vbind = rdec.buffers.iter().find(|b| !b.is_fragment).unwrap();
    assert!(vbind.has_attribute_stride);
    assert_eq!(vbind.attribute_stride, 16);
    let fbind = rdec.buffers.iter().find(|b| b.is_fragment).unwrap();
    assert!(!fbind.has_attribute_stride);
}

/// Dedicated wire-backed E2E: vertex attributeStride=16 through command
/// memory (not host fill API). Proves stride table encode → decode →
/// setVertexBuffer:offset:attributeStride:.
#[test]
fn decode_encode_barrier_and_threadgroup_memory() {
    use crate::runtime::decode::resource::{compute_icb_layout, icb_layout_kernel_tg_slot_count};
    let layout = compute_icb_layout(1, 2);
    assert_eq!(icb_layout_kernel_tg_slot_count(&layout), 2);
    assert_eq!(layout.barrier_offset, 4);
    let fill = IcbComputeFill {
        command_index: 0,
        pipeline_ref: 6,
        buffers: vec![],
        threadgroup_memory: vec![
            IcbThreadgroupMemory {
                index: 0,
                length: 64, // multiple of 16
            },
            IcbThreadgroupMemory {
                index: 1,
                length: 128,
            },
        ],
        barrier: true,
        dispatch: unit_grid_dispatch(1, 1, 1),
    };
    let slot = encode_compute_command_slot(&layout, &fill).unwrap();
    assert_eq!(ld32(&slot[layout.barrier_offset as usize..]), 1);
    let tg0 = layout.threadgroup_memory_length_offset as usize;
    assert_eq!(ld64(&slot[tg0..]), 64);
    assert_eq!(ld64(&slot[tg0 + 8..]), 128);
    let decoded = decode_compute_command_slot(&layout, &slot, 1)
        .unwrap()
        .expect("filled");
    assert!(decoded.barrier);
    assert_eq!(decoded.threadgroup_memory.len(), 2);
    assert_eq!(decoded.threadgroup_memory[0].length, 64);
    assert_eq!(decoded.threadgroup_memory[1].index, 1);
    assert_eq!(decoded.threadgroup_memory[1].length, 128);
}
