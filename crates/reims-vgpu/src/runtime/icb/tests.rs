use super::*;
use crate::contract::endian::{st16, st32, st64};
use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
/// Page-entry bits for hand-mapping a draw target. Metal-arm only, same reason
/// as the compute-pipeline block below.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
use crate::contract::pass_action::MTL_STORE_ACTION_STORE;
use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
use crate::runtime::decode::resource::{
    compute_only_icb_layout, encode_icb_command_layout, list_object_entry_offset,
    render_icb_layout, ICB_DESC_FLAGS, ICB_DESC_LAYOUT, ICB_DESC_LEN, ICB_DESC_MAX_COMMAND_COUNT,
    ICB_DESC_MAX_FRAGMENT_BINDS, ICB_DESC_MAX_KERNEL_BINDS, ICB_DESC_MAX_VERTEX_BINDS,
    ICB_DESC_OPTIONS, ICB_FLAG_INHERIT_BUFFERS, ICB_LAYOUT_LEN,
    MTL_INDIRECT_CMD_CONCURRENT_DISPATCH, MTL_INDIRECT_CMD_DRAW, MTL_INDIRECT_CMD_DRAW_INDEXED,
    OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_BUFFER, OBJECT_TYPE_TYPE7, PIPELINE_TAG_FRAGMENT_FUNC,
    PIPELINE_TAG_VERTEX_FUNC, RESOURCE_PAGE_SHIFT, TYPE7_OBJECT_ICB, TYPE7_OBJECT_RENDER_PIPELINE,
};
/// Compute-pipeline and function descriptor constants, used only by the
/// Metal-arm execute tests below. Kept in their own gated `use` so the Vulkan arm
/// does not carry unused imports.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
use crate::runtime::decode::resource::{
    OBJECT_TYPE_FUNCTION, PIPELINE_TAG_KERNEL_FUNC, TYPE7_FIRST_TLVS, TYPE7_OBJECT_COMPUTE_PIPELINE,
};
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
use crate::runtime::draw::{
    encode_icb_execute_and_writeback, BufferBind, ColorRtRequest, DrawEncodeRequest, EncodeStatus,
};
use crate::runtime::gva_mem;
use crate::runtime::host::FakeHost;
/// Readback, the draw encoder and the fixture directory. Every Metal-arm ICB
/// test needs this same set; it was spelled inside 29 test bodies before.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
use crate::runtime::mapping_write;
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
use std::path::PathBuf;
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
        IcbStatus::MetalFailed("icb_pso_pipeline_state"),
        IcbStatus::NoMetal("icb_frc_no_metal"),
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

/// Process-global ICB cache is shared across tests — serialize metal ICB tests.
///
/// Taken with `unwrap_or_else(|e| e.into_inner())`, never a bare `unwrap`:
/// the guard only orders access to a process-global cache, so a poisoned
/// lock carries no unsound state. A bare `unwrap` turns the *first* failing
/// test into a cascade — when the `compute_mul3add1.mtlb` fixture went
/// missing, 3 real failures poisoned this lock and reported as 43, burying
/// the one root cause under 40 `PoisonError`s.
static ICB_TEST_LOCK: Mutex<()> = Mutex::new(());

fn setup_task(host: &mut FakeHost, state: &mut DeviceState) {
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

/// The shared opening of every `compute_mul3add1.mtlb` body: the encode lock,
/// a cleared ICB cache, the kernel blob, and a task-1 device with its page
/// tables walked. Six bodies opened with these eleven lines; what they vary
/// starts at the ICB descriptor, so that is where the fixture stops.
///
/// The guard is returned rather than taken inside, because it has to outlive
/// the body — `clear_icb_cache` and the ICB cache it clears are process-global.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn mul3add1_fixture() -> (
    std::sync::MutexGuard<'static, ()>,
    Vec<u8>,
    FakeHost,
    DeviceState,
) {
    let guard = icb_test_guard();
    let mtlb = read_fixture("compute_mul3add1.mtlb");
    let (host, state) = icb_device();
    (guard, mtlb, host, state)
}

/// Hold the encode lock for this test and clear the process-global ICB cache
/// under it. Thirty-six bodies opened with these two statements; taking the
/// lock without clearing, or clearing without holding it, are both bugs the
/// pairing prevents.
fn icb_test_guard() -> std::sync::MutexGuard<'static, ()> {
    let guard = ICB_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    clear_icb_cache();
    guard
}

/// A device with task 1 defined and its page tables walked — what every body
/// in this file needs before it can put an object anywhere.
fn icb_device() -> (FakeHost, DeviceState) {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task(&mut host, &mut state);
    (host, state)
}

/// A shader blob out of `tests/fixtures`. Forty-four reads spelled out the
/// join and then repeated the file name in the `expect`, which reported the
/// name and swallowed the `io::Error` saying *why*; this reports both.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn read_fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// [`fill_render_command`] against the fixture every render body in this file
/// builds: task 1, ICB object ref 9.
///
/// Both are constant across every caller — they are what `icb_device` and
/// `mul3add1_fixture` set up — so they were four lines of noise in front of the
/// `IcbRenderFill` that is the actual subject of each test.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn fill_render(
    state: &DeviceState,
    host: &FakeHost,
    fill: &IcbRenderFill,
) -> Result<(), IcbStatus> {
    fill_render_command(state, host, 1, 9, fill)
}

/// [`fill_compute_command`] against the same fixture. See [`fill_render`].
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn fill_compute(
    state: &DeviceState,
    host: &FakeHost,
    fill: &IcbComputeFill,
) -> Result<(), IcbStatus> {
    fill_compute_command(state, host, 1, 9, fill)
}

/// The guest's `ExecuteCommandsInBuffer` (0xe4) over `[location, length)` of
/// ICB object `icb_ref` — the command eight bodies build field by field on a
/// `Default` before handing it to `ComputeSession::encode_icb`.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn execute_icb_command(
    icb_ref: u32,
    location: u64,
    length: u64,
) -> crate::runtime::decode::compute::Command {
    use crate::runtime::decode::compute::{Command, Kind};
    Command {
        kind: Kind::ExecuteCommandsInBuffer,
        indirect_command_buffer_ref: icb_ref,
        indirect_command_range_location: location,
        indirect_command_range_length: length,
        ..Default::default()
    }
}

/// The four little-endian `u32`s at task 1's `gva`, which is what every
/// compute-writeback body in this file checks its kernel by. Panics rather than
/// returning: a read that fails here is the fixture broken, not the assertion.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn read_u32x4(host: &FakeHost, state: &DeviceState, gva: u64) -> Vec<u32> {
    let mut back = [0u8; 16];
    gva_mem::read_task_gva(host, &state.tasks[1], gva, &mut back, PAGE_SHIFT_ARM64E)
        .expect("readback");
    back.chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
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
    st32(&mut b[0..], TYPE7_OBJECT_ICB);
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

/// Render ICB create body (Draw and/or DrawIndexed commandTypes).
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn make_render_icb_desc_bytes(
    max_cmds: u32,
    max_vertex: u16,
    max_fragment: u16,
    command_types: u32,
) -> Vec<u8> {
    make_render_icb_desc_bytes_ex(max_cmds, max_vertex, max_fragment, 0, 0, command_types, 0)
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn make_render_icb_desc_bytes_flags(
    max_cmds: u32,
    max_vertex: u16,
    max_fragment: u16,
    command_types: u32,
    flags: u16,
) -> Vec<u8> {
    make_render_icb_desc_bytes_ex(
        max_cmds,
        max_vertex,
        max_fragment,
        0,
        0,
        command_types,
        flags,
    )
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn make_render_icb_desc_bytes_ex(
    max_cmds: u32,
    max_vertex: u16,
    max_fragment: u16,
    max_object: u16,
    max_mesh: u16,
    command_types: u32,
    flags: u16,
) -> Vec<u8> {
    use crate::runtime::decode::resource::{
        render_icb_layout_ex, ICB_DESC_MAX_MESH_BINDS, ICB_DESC_MAX_OBJECT_BINDS,
        ICB_FLAG_INHERIT_PIPELINE_STATE,
    };
    let mut b = vec![0u8; ICB_DESC_LEN];
    st32(&mut b[0..], TYPE7_OBJECT_ICB);
    st32(&mut b[4..], ICB_DESC_LEN as u32);
    st32(&mut b[8..], command_types);
    b[ICB_DESC_MAX_VERTEX_BINDS] = max_vertex as u8;
    b[ICB_DESC_MAX_FRAGMENT_BINDS] = max_fragment as u8;
    b[ICB_DESC_MAX_KERNEL_BINDS] = 0;
    b[ICB_DESC_MAX_OBJECT_BINDS] = max_object as u8;
    b[ICB_DESC_MAX_MESH_BINDS] = max_mesh as u8;
    // See `make_icb_desc_bytes_tg`: off the serializer's default word.
    st16(
        &mut b[ICB_DESC_FLAGS..],
        crate::runtime::decode::resource::ICB_FLAGS_DEFAULT | flags,
    );
    let layout = render_icb_layout_ex(
        max_vertex,
        max_fragment,
        max_object,
        max_mesh,
        0,
        command_types,
    );
    b[ICB_DESC_LAYOUT..ICB_DESC_LAYOUT + ICB_LAYOUT_LEN]
        .copy_from_slice(&encode_icb_command_layout(&layout));
    st32(&mut b[ICB_DESC_MAX_COMMAND_COUNT..], max_cmds);
    st32(&mut b[ICB_DESC_OPTIONS..], 0);
    let _ = ICB_FLAG_INHERIT_PIPELINE_STATE;
    b
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn load_oracle_mtlb() -> (Vec<u8>, Vec<u8>) {
    let vtx = read_fixture("oracle_draw_vtx.mtlb");
    let frag = read_fixture("oracle_draw_frag.mtlb");
    (vtx, frag)
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn load_stagein_mtlb() -> (Vec<u8>, Vec<u8>) {
    let vtx = read_fixture("render_stagein_vtx.mtlb");
    let frag = read_fixture("render_stagein_frag.mtlb");
    (vtx, frag)
}

/// Minimal compute pipeline type-7 descriptor: one first-TLV entry naming the
/// kernel function ref. Eight call sites built these same seven lines. Gated
/// like its constants and all eight callers, which are Metal-arm execute tests.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn make_compute_pipeline_desc(kernel_ref: u32) -> Vec<u8> {
    let mut pdesc = vec![0u8; 32];
    st32(&mut pdesc[0..], TYPE7_OBJECT_COMPUTE_PIPELINE);
    st32(&mut pdesc[4..], 32);
    pdesc[TYPE7_FIRST_TLVS] = 1;
    pdesc[TYPE7_FIRST_TLVS + 1] = PIPELINE_TAG_KERNEL_FUNC;
    pdesc[TYPE7_FIRST_TLVS + 2] = 4;
    st32(&mut pdesc[TYPE7_FIRST_TLVS + 3..], kernel_ref);
    pdesc
}

/// Minimal render pipeline type-7 descriptor: a first-TLV block carrying only
/// the vertex and fragment function refs — no vertex-input and no colour
/// attachment, unlike [`make_stagein_render_pipeline_desc`]. Six call sites
/// built these same twelve lines. Gated like its constants and all six callers,
/// which are Metal-arm execute tests.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn make_render_pipeline_desc(vert_ref: u32, frag_ref: u32) -> Vec<u8> {
    let mut pdesc = vec![0u8; 16 + 1 + 6 + 6];
    let blen = pdesc.len() as u32;
    st32(&mut pdesc[0..], TYPE7_OBJECT_RENDER_PIPELINE);
    st32(&mut pdesc[4..], blen);
    st32(&mut pdesc[8..], 6);
    pdesc[TYPE7_FIRST_TLVS] = 2;
    pdesc[TYPE7_FIRST_TLVS + 1] = PIPELINE_TAG_VERTEX_FUNC;
    pdesc[TYPE7_FIRST_TLVS + 2] = 4;
    st32(&mut pdesc[TYPE7_FIRST_TLVS + 3..], vert_ref);
    pdesc[TYPE7_FIRST_TLVS + 7] = PIPELINE_TAG_FRAGMENT_FUNC;
    pdesc[TYPE7_FIRST_TLVS + 8] = 4;
    st32(&mut pdesc[TYPE7_FIRST_TLVS + 9..], frag_ref);
    pdesc
}

/// Type-7 render pipeline with vertex-input block: Float4 attr0 @ buffer0 stride 16.
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
    st32(&mut b[0..], TYPE7_OBJECT_RENDER_PIPELINE);
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

/// Compact type-7 mesh pipeline (host SPI shape): tag `0x14` section offset
/// + optional object `0x01` + mesh `0x02` + fragment `0x03`.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn make_mesh_render_pipeline_desc(
    object_ref: Option<u32>,
    mesh_ref: u32,
    frag_ref: u32,
) -> Vec<u8> {
    use crate::runtime::decode::resource::{
        PIPELINE_TAG_MESH_FRAGMENT_FUNC, PIPELINE_TAG_MESH_FUNC, PIPELINE_TAG_MESH_SECTION_OFFSET,
        PIPELINE_TAG_OBJECT_FUNC, TYPE7_OBJECT_RENDER_PIPELINE,
    };
    let mut fields = Vec::new();
    // Section offset filled after we know field payload size (matches SPI:
    // offset from header end to trailing color/rest region).
    fields.push((PIPELINE_TAG_MESH_SECTION_OFFSET, 0u32));
    if let Some(oref) = object_ref {
        fields.push((PIPELINE_TAG_OBJECT_FUNC, oref));
    }
    fields.push((PIPELINE_TAG_MESH_FUNC, mesh_ref));
    fields.push((PIPELINE_TAG_MESH_FRAGMENT_FUNC, frag_ref));
    let n = fields.len();
    // header 16 + fieldCount 1 + n * (tag+len+u32=6)
    let first_tlv_len = 1 + n * 6;
    let mut b = vec![0u8; 16 + first_tlv_len];
    let blen = b.len() as u32;
    st32(&mut b[0..], TYPE7_OBJECT_RENDER_PIPELINE);
    st32(&mut b[4..], blen);
    st32(&mut b[8..], 6);
    b[16] = n as u8;
    // Mesh section offset = first-subrecord size (no color block in fixture).
    fields[0].1 = first_tlv_len as u32;
    let mut p = 17;
    for (tag, val) in fields {
        b[p] = tag;
        b[p + 1] = 4;
        st32(&mut b[p + 2..], val);
        p += 6;
    }
    b
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn put_type1_buffer(
    host: &mut FakeHost,
    state: &DeviceState,
    obj_ref: u32,
    handle: u32,
    bytes: &[u8],
) {
    let gva = (handle as u64) << RESOURCE_PAGE_SHIFT;
    gva_mem::write_task_gva_arm64e(host, &state.tasks[1], gva, bytes);
    let mut bdesc = vec![0u8; 16];
    st64(&mut bdesc[0..], bytes.len() as u64);
    st64(&mut bdesc[8..], handle as u64);
    // Descriptors must sit past the object-list region (32×12 = 0x180).
    let bdesc_gva = 0x200u64 + (obj_ref as u64) * 0x20;
    put_object(host, state, obj_ref, OBJECT_TYPE_BUFFER, bdesc_gva, &bdesc);
}

/// Write `bytes` at `gva` and publish it as object `ref_` in the task's object
/// list — the pair every fixture here performs together.
///
/// The published length is `bytes.len()`. That is not a simplification: the
/// declared length was carried as its own argument through all 173 of these
/// call sites and asserted equal to the slice at every one of them across the
/// whole suite. A test that needs the two to disagree — a short-descriptor
/// refusal — calls [`put_list_entry`] directly, which still takes both.
fn put_object(
    host: &mut FakeHost,
    state: &DeviceState,
    ref_: u32,
    otype: u8,
    gva: u64,
    bytes: &[u8],
) {
    gva_mem::write_task_gva_arm64e(host, &state.tasks[1], gva, bytes);
    put_list_entry(host, state, ref_, otype, bytes.len() as u32, gva);
}

/// Map a 4x4 BGRA8 render target for a draw to land in, one guest page backed
/// at `pfn`, and return its mapping id.
///
/// Every draw test in this file needs exactly this surface and differs only in
/// which page it takes, so `pfn` is the one parameter — distinct per test so
/// two fixtures never share a guest page. The mapping is marked internal and
/// mapped by hand because `map_surface` alone leaves it without page entries,
/// which is the state a real `MapMemory2` would have already left behind.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
/// Read back [`map_draw_target`]'s 4x4 surface and assert every texel is the
/// colour the ICB test shaders write: `float4(0.4, 0.267, 0.133, 1)`, which is
/// BGRA 34, 68, 102, 255.
///
/// The tolerance is +/-2 per channel, for float-to-unorm rounding across
/// different Metal devices. `what` names the command shape under test so a
/// failure says which of the eighteen callers produced the wrong pixels.
fn assert_target_is_shader_solid(
    state: &mut DeviceState,
    host: &mut FakeHost,
    mapping_id: u32,
    what: &str,
) {
    assert_target_texels(state, host, mapping_id, [34, 68, 102, 255], 2, what);
}

/// Same readback for the stage-in shaders, whose blue channel carries the
/// surface id so a test can tell which surface the stage-in attributes came
/// from. Tolerance is +/-1 rather than +/-2 because these write exact byte
/// values rather than a float4 that has to round.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn assert_stagein_solid(
    state: &mut DeviceState,
    host: &mut FakeHost,
    mapping_id: u32,
    sid: u8,
    what: &str,
) {
    assert_target_texels(
        state,
        host,
        mapping_id,
        [0x22, 0x44, 0x60 + sid, 0xff],
        1,
        what,
    );
}

/// Read back [`map_draw_target`]'s 4x4 surface and assert every texel is `want`
/// within `tol` per channel. `what` names the command shape under test so a
/// failure says which of the twenty-nine callers produced the wrong pixels.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn assert_target_texels(
    state: &mut DeviceState,
    host: &mut FakeHost,
    mapping_id: u32,
    want: [u8; 4],
    tol: i32,
    what: &str,
) {
    let mut back = vec![0u8; 4 * 4 * 4];
    assert!(mapping_write::read_rect_raw(
        state,
        host,
        mapping_id,
        mapping_write::Rect {
            origin_x: 0,
            origin_y: 0,
            width: 4,
            height: 4
        },
        &mut back,
        16
    ));
    let near = |g: u8, w: u8| (g as i32 - w as i32).abs() <= tol;
    for (p, px) in back.chunks_exact(4).enumerate() {
        assert!(
            near(px[0], want[0])
                && near(px[1], want[1])
                && near(px[2], want[2])
                && near(px[3], want[3]),
            "pixel {p} = {px:02x?}; want ~{want:02x?} ({what})"
        );
    }
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
/// The one-triangle draw every ICB pixel test issues into [`map_draw_target`]'s
/// surface: pipeline 6, three vertices, one instance, triangles, over a
/// zero-filled 4x4 BGRA8 seed.
///
/// A test that needs a different shape spells its own literal; this is only the
/// two dozen that wanted exactly this one.
fn draw_request(mapping_id: u32) -> DrawEncodeRequest {
    DrawEncodeRequest {
        task_id: 1,
        pipeline_ref: 6,
        vertex_count: 3,
        instance_count: 1,
        primitive_type: 3,
        colors: vec![ColorRtRequest {
            slot: 0,
            mapping_id,
            width: 4,
            height: 4,
            format: crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            store_action: MTL_STORE_ACTION_STORE,
            target_seed_rgba: Some(vec![0u8; 4 * 4 * 4]),
            ..Default::default()
        }],
        ..Default::default()
    }
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
/// Publish an MTLB `blob` at `blob_page`'s GVA page and describe it as function
/// object `ref_`, whose 32-byte descriptor lives at `desc_gva` and carries
/// `(blob gva, blob len)`. Fifty-six test bodies spelled these six lines.
///
/// All four values are the caller's because they are fixture identities that
/// must not collide inside one test: two functions in the same pipeline take
/// different pages, different refs and different descriptor addresses.
fn put_function_object(
    host: &mut FakeHost,
    state: &DeviceState,
    ref_: u32,
    desc_gva: u64,
    blob_page: u64,
    blob: &[u8],
) {
    let blob_gva = blob_page << RESOURCE_PAGE_SHIFT;
    gva_mem::write_task_gva_arm64e(host, &state.tasks[1], blob_gva, blob);
    let mut fdesc = vec![0u8; 32];
    st64(&mut fdesc[0..], blob_gva);
    st32(&mut fdesc[8..], blob.len() as u32);
    put_object(host, state, ref_, OBJECT_TYPE_FUNCTION, desc_gva, &fdesc);
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
/// Publish an encoded ICB command-memory `slot` at `cmd_handle`'s page, wrap it
/// in buffer object 10, and associate that buffer with ICB object 9 — the
/// sequence `execute` performs before any fill, spelled in thirteen test bodies
/// before this.
///
/// `cmd_handle` is the caller's because it is the slot's GVA page index and must
/// not collide with the other fixtures a given test has already placed; refs 9
/// and 10 and descriptor GVA 0x1a0 are this file's fixture identities and are
/// the same at every one of those sites.
fn associate_icb_command_memory(
    host: &mut FakeHost,
    state: &DeviceState,
    cmd_handle: u32,
    slot: &[u8],
) {
    let cmd_gva = u64::from(cmd_handle) << RESOURCE_PAGE_SHIFT;
    gva_mem::write_task_gva_arm64e(host, &state.tasks[1], cmd_gva, slot);
    let mut cmd_bdesc = vec![0u8; 16];
    st64(&mut cmd_bdesc[0..], slot.len() as u64);
    st64(&mut cmd_bdesc[8..], u64::from(cmd_handle));
    put_object(host, state, 10, OBJECT_TYPE_BUFFER, 0x1a0, &cmd_bdesc);
    // `&*host`, not `host`: this takes `&M` while `put_object` above needs
    // `&mut FakeHost`, and the reborrow is what lets one binding serve both.
    //
    // Straight to the association rather than through `0x1d1`. That record is a
    // query this device refuses — see `apply_icb_host_resource_info` — so
    // routing every ICB fixture through it would have made this helper the one
    // thing keeping the old misreading alive.
    associate_icb_backing_buffer_ref(state, &*host, 1, 9, 10)
        .expect("associate ICB command memory");
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn map_draw_target(host: &mut FakeHost, state: &mut DeviceState, pfn: u32) -> u32 {
    let mapping_id = 9u32;
    host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
    state.map_surface(mapping_id);
    {
        let m = state.mappings.get_mut(&mapping_id).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    assert!(state.set_mapping_geom(mapping_id, 4, 4, MTL_FORMAT_BGRA8_UNORM));
    mapping_id
}

fn put_list_entry(
    host: &mut FakeHost,
    state: &DeviceState,
    ref_: u32,
    otype: u8,
    len: u32,
    gva: u64,
) {
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
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, gva, &desc);
    let icb = load_icb_descriptor(&state, &host, 1, 9).unwrap();
    assert_eq!(icb.max_command_count, 8);
    assert_eq!(icb.max_kernel_buffer_bind_count, 4);
    assert!(icb.inherit_buffers());
    assert_eq!(icb.command_types, MTL_INDIRECT_CMD_CONCURRENT_DISPATCH);
}

/// A flag this device does not apply is counted when the guest asks for it, and
/// not counted when the guest leaves it alone.
///
/// The load path is where this has to be checked rather than at the decoder:
/// `load_icb_descriptor` is what both backends call, and it is the only place
/// that sees a decoded descriptor on the Vulkan arm at all. A counter sited in
/// `materialize_metal_icb` would read structurally zero there for a reason that
/// has nothing to do with what the guest asked for.
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
        put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, gva, &desc);
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
        let word = crate::contract::endian::ld16(&desc[ICB_DESC_FLAGS..]);
        st16(&mut desc[ICB_DESC_FLAGS..], word & !clear);
        let gva = 1u64 << RESOURCE_PAGE_SHIFT;
        put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, gva, &desc);
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
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn materialize_and_execute_empty_range() {
    use crate::backend::metal::raw_metal::execute_commands_in_buffer;
    use crate::backend::metal::runtime::{system_device, thread_queue};
    use metal::MTLDispatchType;

    let _guard = icb_test_guard();
    let (mut host, state) = icb_device();
    let desc = make_icb_desc_bytes(8, 4, true);
    let gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, gva, &desc);
    let (d, icb) = resolve_metal_icb(&state, &host, 1, 9).expect("materialize");
    assert_eq!(d.max_command_count, 8);
    assert_eq!(icb.size(), 8);

    let device = system_device().unwrap();
    let queue = thread_queue(device);
    let cb = queue.new_command_buffer().to_owned();
    let enc = cb.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial);
    execute_commands_in_buffer(enc, icb.as_ref(), 0, 0);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}

#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn fill_and_execute_mul3add1_writeback() {
    use crate::runtime::compute_session::ComputeSession;

    let (_guard, mtlb, mut host, mut state) = mul3add1_fixture();

    // ICB object ref 9: 1 command, maxKernel=1, no inherit (explicit fills).
    let icb_desc = make_icb_desc_bytes(1, 1, false);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    // Function + pipeline + data buffer (mul3add1).
    put_function_object(&mut host, &state, 5, 0x100, 2, &mtlb);

    let pdesc = make_compute_pipeline_desc(5);
    let pdesc_gva = 0x140u64;
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, pdesc_gva, &pdesc);

    let data = [1u32, 2, 3, 4];
    let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf_gva = 3u64 << RESOURCE_PAGE_SHIFT;
    gva_mem::write_task_gva_arm64e(&mut host, &state.tasks[1], buf_gva, &data_bytes);
    let mut bdesc = vec![0u8; 16];
    st64(&mut bdesc[0..], 16);
    st32(&mut bdesc[8..], 3);
    let bdesc_gva = 0x180u64;
    put_object(&mut host, &state, 7, OBJECT_TYPE_BUFFER, bdesc_gva, &bdesc);

    // Host fill of command slot 0 (no stream fill opcode yet).
    fill_compute(
        &state,
        &host,
        &IcbComputeFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![kernel_bind(0, 7)],
            threadgroup_memory: vec![],
            barrier: false,
            dispatch: unit_grid_dispatch(4, 1, 1),
        },
    )
    .expect("fill");

    // Cache hit on re-resolve (same size / command count).
    let (d_a, icb_a) = resolve_metal_icb(&state, &host, 1, 9).unwrap();
    let (d_b, icb_b) = resolve_metal_icb(&state, &host, 1, 9).unwrap();
    assert_eq!(d_a.max_command_count, d_b.max_command_count);
    assert_eq!(icb_a.size(), icb_b.size());
    assert_eq!(icb_a.size(), 1);

    // Product execute 0xe4 range [0,1] on a compute session + writeback.
    let mut session = ComputeSession::open(0).expect("session");
    let cmd = execute_icb_command(9, 0, 1);
    assert_eq!(
        session.encode_icb(
            &mut state,
            &mut host,
            1,
            &cmd,
            &crate::runtime::compute_exec::ComputeAccum::default()
        ),
        crate::runtime::compute_exec::ComputeStatus::Ok
    );
    assert_eq!(
        session.finish(&mut host, &mut state, 1),
        crate::runtime::compute_exec::ComputeStatus::Ok
    );

    let out = read_u32x4(&host, &state, buf_gva);
    assert_eq!(
        out,
        vec![4, 7, 10, 13],
        "ICB fill+execute mul3add1 writeback"
    );
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

/// Pixel-level ICB DrawPatches oracle: triangle patch + constant tess
/// factors 1.0 → solid BGRA fill (full-screen control points).
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn fill_render_draw_patches_tessellation_oracle() {
    use crate::runtime::decode::resource::MTL_INDIRECT_CMD_DRAW_PATCHES;

    let _guard = icb_test_guard();

    let vert_mtlb = read_fixture("icb_tess_vtx.metallib");
    let frag_mtlb = read_fixture("icb_tess_frag.metallib");

    let (mut host, mut state) = icb_device();

    let icb_desc = make_render_icb_desc_bytes(1, 1, 0, MTL_INDIRECT_CMD_DRAW_PATCHES);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &vert_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    // Vertex-input Float4 @ buffer0 stride 16 (step defaults to PerPatchControlPoint).
    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    // Full-screen clip-space triangle as 3 control points.
    let tri: [[f32; 4]; 3] = [
        [-1.0, -1.0, 0.0, 1.0],
        [3.0, -1.0, 0.0, 1.0],
        [-1.0, 3.0, 0.0, 1.0],
    ];
    let cp_bytes: Vec<u8> = tri
        .iter()
        .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
        .collect();
    put_type1_buffer(&mut host, &state, 11, 4, &cp_bytes);

    // MTLTriangleTessellationFactorsHalf: edge[3] + inside, half 1.0 = 0x3c00.
    let tess_bytes: [u8; 8] = [
        0x00, 0x3c, // edge0
        0x00, 0x3c, // edge1
        0x00, 0x3c, // edge2
        0x00, 0x3c, // inside
    ];
    put_type1_buffer(&mut host, &state, 12, 5, &tess_bytes);

    fill_render(
        &state,
        &host,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![render_bind(0, 11, false)],
            object_threadgroup_memory: vec![],
            draw: IcbRenderDraw::Patches {
                number_of_patch_control_points: 3,
                patch_start: 0,
                patch_count: 1,
                patch_index_buffer_ref: 0, // null — sequential control points
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
        },
    )
    .expect("fill DrawPatches tessellation");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x38);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "DrawPatches tessellation ICB execute"
    );

    assert_target_is_shader_solid(
        &mut state,
        &mut host,
        mapping_id,
        "DrawPatches tessellation",
    );
}

/// Pixel-level ICB DrawIndexedPatches: dummy control point at index 0,
/// real triangle at indices [1,2,3] → same solid BGRA as DrawPatches.
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn fill_render_draw_indexed_patches_tessellation_oracle() {
    use crate::runtime::decode::resource::MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES;

    let _guard = icb_test_guard();

    let vert_mtlb = read_fixture("icb_tess_vtx.metallib");
    let frag_mtlb = read_fixture("icb_tess_frag.metallib");

    let (mut host, mut state) = icb_device();

    let icb_desc = make_render_icb_desc_bytes(1, 1, 0, MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &vert_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    // Control points: dummy at 0, real full-screen triangle at 1,2,3.
    let cps: [[f32; 4]; 4] = [
        [0.0, 0.0, 0.0, 1.0],
        [-1.0, -1.0, 0.0, 1.0],
        [3.0, -1.0, 0.0, 1.0],
        [-1.0, 3.0, 0.0, 1.0],
    ];
    let cp_bytes: Vec<u8> = cps
        .iter()
        .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
        .collect();
    put_type1_buffer(&mut host, &state, 11, 4, &cp_bytes);

    // UInt16 control-point indices [1,2,3].
    let indices: [u16; 3] = [1, 2, 3];
    let index_bytes: Vec<u8> = indices.iter().flat_map(|v| v.to_le_bytes()).collect();
    put_type1_buffer(&mut host, &state, 13, 5, &index_bytes);

    let tess_bytes: [u8; 8] = [0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c];
    put_type1_buffer(&mut host, &state, 12, 6, &tess_bytes);

    fill_render(
        &state,
        &host,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![render_bind(0, 11, false)],
            object_threadgroup_memory: vec![],
            draw: IcbRenderDraw::IndexedPatches {
                number_of_patch_control_points: 3,
                patch_start: 0,
                patch_count: 1,
                patch_index_buffer_ref: 0,
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
                    instance_stride: 0,
                },
            },
        },
    )
    .expect("fill DrawIndexedPatches tessellation");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x39);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "DrawIndexedPatches tessellation ICB execute"
    );

    assert_target_is_shader_solid(
        &mut state,
        &mut host,
        mapping_id,
        "DrawIndexedPatches tessellation",
    );
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

/// Pixel-level ICB drawMeshThreads: mesh emits full-screen triangle,
/// fragment solid BGRA matching tess/draw oracles.
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn fill_render_draw_mesh_threads_oracle() {
    use crate::runtime::decode::resource::MTL_INDIRECT_CMD_DRAW_MESH_THREADS;

    let _guard = icb_test_guard();

    let mesh_mtlb = read_fixture("icb_mesh.metallib");
    let frag_mtlb = read_fixture("icb_mesh_frag.metallib");

    let (mut host, mut state) = icb_device();

    let icb_desc = make_render_icb_desc_bytes(1, 0, 0, MTL_INDIRECT_CMD_DRAW_MESH_THREADS);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    // Mesh function lives in the type-7 "vertex" function slot (no guest
    // mesh-pipeline descriptor yet — host-fill only path).
    put_function_object(&mut host, &state, 2, 0x200, 2, &mesh_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    // No vertex attributes needed for mesh; reuse stage-in desc helper with empty attrs.
    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    fill_render(
        &state,
        &host,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![],
            object_threadgroup_memory: vec![],
            draw: unit_mesh_threads_draw(),
        },
    )
    .expect("fill DrawMeshThreads");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x3a);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "DrawMeshThreads ICB execute"
    );

    assert_target_is_shader_solid(&mut state, &mut host, mapping_id, "DrawMeshThreads");
}

/// Pixel-level ICB drawMeshThreadgroups (same solid BGRA).
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn fill_render_draw_mesh_threadgroups_oracle() {
    use crate::runtime::decode::resource::MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS;

    let _guard = icb_test_guard();

    let mesh_mtlb = read_fixture("icb_mesh.metallib");
    let frag_mtlb = read_fixture("icb_mesh_frag.metallib");

    let (mut host, mut state) = icb_device();

    let icb_desc = make_render_icb_desc_bytes(1, 0, 0, MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &mesh_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    fill_render(
        &state,
        &host,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![],
            object_threadgroup_memory: vec![],
            draw: unit_mesh_draw(),
        },
    )
    .expect("fill DrawMeshThreadgroups");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x3b);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "DrawMeshThreadgroups ICB execute"
    );

    assert_target_is_shader_solid(&mut state, &mut host, mapping_id, "DrawMeshThreadgroups");
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
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn fill_render_negative_base_vertex_stagein_oracle() {
    let _guard = icb_test_guard();

    let (vert_mtlb, frag_mtlb) = load_stagein_mtlb();
    let (mut host, mut state) = icb_device();

    let icb_desc = make_render_icb_desc_bytes(1, 1, 1, MTL_INDIRECT_CMD_DRAW_INDEXED);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &vert_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    // Clip-space full-cover triangle at vertex indices 0,1,2.
    let tri: [[f32; 4]; 3] = [
        [-1.0, -1.0, 0.0, 1.0],
        [3.0, -1.0, 0.0, 1.0],
        [-1.0, 3.0, 0.0, 1.0],
    ];
    let pos_bytes: Vec<u8> = tri
        .iter()
        .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
        .collect();
    put_type1_buffer(&mut host, &state, 11, 4, &pos_bytes);

    // indices [1,2,3] + baseVertex(-1) → vertex_id 0,1,2.
    let indices: [u16; 3] = [1, 2, 3];
    let index_bytes: Vec<u8> = indices.iter().flat_map(|v| v.to_le_bytes()).collect();
    put_type1_buffer(&mut host, &state, 12, 5, &index_bytes);

    let sid = 7u8;
    let r = (0x60u32 + sid as u32) as f32 / 255.0;
    let color = [r, 0x44 as f32 / 255.0, 0x22 as f32 / 255.0, 1.0f32];
    let color_bytes: Vec<u8> = color.iter().flat_map(|f| f.to_le_bytes()).collect();
    put_type1_buffer(&mut host, &state, 13, 6, &color_bytes);

    fill_render(
        &state,
        &host,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![render_bind(0, 11, false), render_bind(0, 13, true)],
            object_threadgroup_memory: vec![],
            draw: indexed_draw(-1),
        },
    )
    .expect("fill DrawIndexed baseVertex=-1");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x37);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "negative baseVertex stage_in DrawIndexed"
    );

    assert_stagein_solid(&mut state, &mut host, mapping_id, sid, "baseVertex=-1");
}

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
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn buffer_backed_fill_execute_mul3add1() {
    use crate::runtime::compute_session::ComputeSession;

    let (_guard, mtlb, mut host, mut state) = mul3add1_fixture();

    let icb_desc = make_icb_desc_bytes(1, 1, false);
    let layout = compute_only_icb_layout(1);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 5, 0x100, 2, &mtlb);

    let pdesc = make_compute_pipeline_desc(5);
    let pdesc_gva = 0x140u64;
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, pdesc_gva, &pdesc);

    let data = [1u32, 2, 3, 4];
    let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf_gva = 3u64 << RESOURCE_PAGE_SHIFT;
    gva_mem::write_task_gva_arm64e(&mut host, &state.tasks[1], buf_gva, &data_bytes);
    let mut bdesc = vec![0u8; 16];
    st64(&mut bdesc[0..], 16);
    st32(&mut bdesc[8..], 3);
    let bdesc_gva = 0x180u64;
    put_object(&mut host, &state, 7, OBJECT_TYPE_BUFFER, bdesc_gva, &bdesc);

    // Guest-style fill: command slot in a type-1 backing buffer (handle 4).
    let slot = encode_compute_command_slot(
        &layout,
        &IcbComputeFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![kernel_bind(0, 7)],
            threadgroup_memory: vec![],
            barrier: false,
            dispatch: unit_grid_dispatch(4, 1, 1),
        },
    )
    .unwrap();
    let cmd_handle = 4u32;
    let cmd_mem_gva = (cmd_handle as u64) << RESOURCE_PAGE_SHIFT;
    gva_mem::write_task_gva_arm64e(&mut host, &state.tasks[1], cmd_mem_gva, &slot);
    let mut cmd_bdesc = vec![0u8; 16];
    st64(&mut cmd_bdesc[0..], slot.len() as u64);
    st64(&mut cmd_bdesc[8..], cmd_handle as u64);
    let cmd_bdesc_gva = 0x1c0u64;
    put_object(
        &mut host,
        &state,
        10,
        OBJECT_TYPE_BUFFER,
        cmd_bdesc_gva,
        &cmd_bdesc,
    );

    // Auto-bind via type-1 buffer_ref (sync path / 0x1d1 payload).
    let mem = associate_icb_backing_buffer_ref(&state, &host, 1, 9, 10).expect("associate");
    assert_eq!(mem.gva, cmd_mem_gva);
    assert_eq!(mem.byte_len, layout.command_size as u64);

    let mut session = ComputeSession::open(0).expect("session");
    let cmd = execute_icb_command(9, 0, 1);
    assert_eq!(
        session.encode_icb(
            &mut state,
            &mut host,
            1,
            &cmd,
            &crate::runtime::compute_exec::ComputeAccum::default()
        ),
        crate::runtime::compute_exec::ComputeStatus::Ok
    );
    assert_eq!(
        session.finish(&mut host, &mut state, 1),
        crate::runtime::compute_exec::ComputeStatus::Ok
    );

    let out = read_u32x4(&host, &state, buf_gva);
    assert_eq!(
        out,
        vec![4, 7, 10, 13],
        "type-1 associated ICB fill+execute mul3add1"
    );
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
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);
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

/// Product DrawIndexed ICB fill + execute: oracle fullscreen triangle via
/// indices [0,1,2], fragment constant color → solid BGRA writeback.
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn fill_render_draw_indexed_execute_oracle() {
    let _guard = icb_test_guard();

    let (vert_mtlb, frag_mtlb) = load_oracle_mtlb();
    let (mut host, mut state) = icb_device();

    // ICB: DrawIndexed, maxFragment=1 for color constant at fragment buffer 0.
    let icb_desc = make_render_icb_desc_bytes(1, 0, 1, MTL_INDIRECT_CMD_DRAW_INDEXED);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    // Vertex function (ref 2) + fragment function (ref 3).
    // Descriptor GVAs stay past object-list region (32×12 = 0x180).
    put_function_object(&mut host, &state, 2, 0x200, 2, &vert_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    // Render pipeline type-7 (ref 6): vertex=2, fragment=3.
    let pdesc = make_render_pipeline_desc(2, 3);
    let pdesc_gva = 0x240u64;
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, pdesc_gva, &pdesc);

    // Index buffer (ref 12, handle 4): UInt16 [0,1,2].
    let indices: [u16; 3] = [0, 1, 2];
    let index_bytes: Vec<u8> = indices.iter().flat_map(|v| v.to_le_bytes()).collect();
    put_type1_buffer(&mut host, &state, 12, 4, &index_bytes);

    // Fragment color buffer (ref 13, handle 5): RGBA float4 for sid=7.
    let sid = 7u8;
    let r = (0x60u32 + sid as u32) as f32 / 255.0;
    let color = [r, 0x44 as f32 / 255.0, 0x22 as f32 / 255.0, 1.0f32];
    let color_bytes: Vec<u8> = color.iter().flat_map(|f| f.to_le_bytes()).collect();
    put_type1_buffer(&mut host, &state, 13, 5, &color_bytes);

    // Mapping for color writeback (4×4 BGRA).
    let mapping_id = map_draw_target(&mut host, &mut state, 0x30);

    fill_render(
        &state,
        &host,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![render_bind(0, 13, true)],
            object_threadgroup_memory: vec![],
            draw: indexed_draw(0),
        },
    )
    .expect("fill_render DrawIndexed");

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok
    );

    // BGRA writeback: B=0x22 G=0x44 R=0x67 A=0xff (oracle sid=7).
    assert_stagein_solid(
        &mut state,
        &mut host,
        mapping_id,
        sid,
        "ICB DrawIndexed oracle",
    );
}

/// Buffer-backed DrawIndexed: encode slot → 0x1d1 associate → execute re-fill.
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn buffer_backed_render_draw_indexed_fill_execute() {
    let _guard = icb_test_guard();

    let (vert_mtlb, frag_mtlb) = load_oracle_mtlb();
    let (mut host, mut state) = icb_device();

    let max_v = 0u16;
    let max_f = 1u16;
    let layout = render_icb_layout(max_v, max_f, MTL_INDIRECT_CMD_DRAW_INDEXED);
    let icb_desc = make_render_icb_desc_bytes(1, max_v, max_f, MTL_INDIRECT_CMD_DRAW_INDEXED);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &vert_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_render_pipeline_desc(2, 3);
    let pdesc_gva = 0x240u64;
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, pdesc_gva, &pdesc);

    let indices: [u16; 3] = [0, 1, 2];
    let index_bytes: Vec<u8> = indices.iter().flat_map(|v| v.to_le_bytes()).collect();
    put_type1_buffer(&mut host, &state, 12, 4, &index_bytes);

    let sid = 7u8;
    let r = (0x60u32 + sid as u32) as f32 / 255.0;
    let color = [r, 0x44 as f32 / 255.0, 0x22 as f32 / 255.0, 1.0f32];
    let color_bytes: Vec<u8> = color.iter().flat_map(|f| f.to_le_bytes()).collect();
    put_type1_buffer(&mut host, &state, 13, 5, &color_bytes);

    let slot = encode_render_command_slot(
        &layout,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![render_bind(0, 13, true)],
            object_threadgroup_memory: vec![],
            draw: indexed_draw(0),
        },
    )
    .unwrap();

    associate_icb_command_memory(&mut host, &state, 6, &slot);

    let mapping_id = map_draw_target(&mut host, &mut state, 0x31);

    // Execute path re-fills from command memory (DrawIndexed + fragment bind).
    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok
    );

    assert_stagein_solid(
        &mut state,
        &mut host,
        mapping_id,
        sid,
        "buffer-backed DrawIndexed",
    );
}

/// Wire-backed E2E: DrawPatches tessellation.
///
/// Guest path only — no `fill_render_command` host API:
/// encode slot → type-1 command memory → `0x1d1` associate → execute
/// re-fills via `fill_icb_from_command_memory` → solid BGRA.
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn wire_backed_draw_patches_tessellation_e2e() {
    use crate::runtime::decode::resource::{
        render_draw_patches_icb_layout, MTL_INDIRECT_CMD_DRAW_PATCHES,
    };

    let _guard = icb_test_guard();

    let vert_mtlb = read_fixture("icb_tess_vtx.metallib");
    let frag_mtlb = read_fixture("icb_tess_frag.metallib");

    let (mut host, mut state) = icb_device();

    let layout = render_draw_patches_icb_layout(1);
    let icb_desc = make_render_icb_desc_bytes(1, 1, 0, MTL_INDIRECT_CMD_DRAW_PATCHES);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &vert_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    let tri: [[f32; 4]; 3] = [
        [-1.0, -1.0, 0.0, 1.0],
        [3.0, -1.0, 0.0, 1.0],
        [-1.0, 3.0, 0.0, 1.0],
    ];
    let cp_bytes: Vec<u8> = tri
        .iter()
        .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
        .collect();
    put_type1_buffer(&mut host, &state, 11, 4, &cp_bytes);

    let tess_bytes: [u8; 8] = [0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c];
    put_type1_buffer(&mut host, &state, 12, 5, &tess_bytes);

    // Absolute wire VAs for control-point bind + tess factor (base+0).
    let cp_wire = (4u64) << RESOURCE_PAGE_SHIFT;
    let tess_wire = (5u64) << RESOURCE_PAGE_SHIFT;

    let slot = encode_render_command_slot(
        &layout,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![IcbRenderBufferBind {
                index: 0,
                buffer_ref: 11,
                offset: 0,
                wire_va: cp_wire,
                attribute_stride: 0,
                has_attribute_stride: false,
                is_fragment: false,
                stage: IcbRenderBindStage::Vertex,
            }],
            object_threadgroup_memory: vec![],
            draw: IcbRenderDraw::Patches {
                number_of_patch_control_points: 3,
                patch_start: 0,
                patch_count: 1,
                patch_index_buffer_ref: 0,
                patch_index_buffer_offset: 0,
                patch_index_wire_va: 0,
                instance_count: 1,
                base_instance: 0,
                tessellation_factor: IcbTessellationFactor {
                    buffer_ref: 12,
                    offset: 0,
                    wire_va: tess_wire,
                    instance_stride: 0,
                },
            },
        },
    )
    .expect("encode DrawPatches slot");

    // Command memory only — never call fill_render_command.
    associate_icb_command_memory(&mut host, &state, 6, &slot);

    // Explicit re-fill (what execute does) — proves decode+resolve path.
    fill_icb_from_command_memory(&state, &host, 1, 9, 0, 1)
        .expect("fill_icb_from_command_memory DrawPatches");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x3c);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "wire-backed DrawPatches execute"
    );

    assert_target_is_shader_solid(&mut state, &mut host, mapping_id, "wire-backed DrawPatches");
}

/// Dedicated wire-backed E2E: DrawIndexedPatches tessellation (not via
/// host fill API or DrawPatches-only path).
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn wire_backed_draw_indexed_patches_tessellation_e2e() {
    use crate::runtime::decode::resource::{
        render_draw_indexed_patches_icb_layout, MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES,
    };

    let _guard = icb_test_guard();

    let vert_mtlb = read_fixture("icb_tess_vtx.metallib");
    let frag_mtlb = read_fixture("icb_tess_frag.metallib");

    let (mut host, mut state) = icb_device();

    let layout = render_draw_indexed_patches_icb_layout(1);
    let icb_desc = make_render_icb_desc_bytes(1, 1, 0, MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &vert_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    // Control points: dummy at 0, real triangle at 1,2,3.
    let cps: [[f32; 4]; 4] = [
        [0.0, 0.0, 0.0, 1.0],
        [-1.0, -1.0, 0.0, 1.0],
        [3.0, -1.0, 0.0, 1.0],
        [-1.0, 3.0, 0.0, 1.0],
    ];
    let cp_bytes: Vec<u8> = cps
        .iter()
        .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
        .collect();
    put_type1_buffer(&mut host, &state, 11, 4, &cp_bytes);

    let indices: [u16; 3] = [1, 2, 3];
    let index_bytes: Vec<u8> = indices.iter().flat_map(|v| v.to_le_bytes()).collect();
    put_type1_buffer(&mut host, &state, 13, 5, &index_bytes);

    let tess_bytes: [u8; 8] = [0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c];
    put_type1_buffer(&mut host, &state, 12, 6, &tess_bytes);

    let cp_wire = (4u64) << RESOURCE_PAGE_SHIFT;
    let index_wire = (5u64) << RESOURCE_PAGE_SHIFT;
    let tess_wire = (6u64) << RESOURCE_PAGE_SHIFT;

    let slot = encode_render_command_slot(
        &layout,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![IcbRenderBufferBind {
                index: 0,
                buffer_ref: 11,
                offset: 0,
                wire_va: cp_wire,
                attribute_stride: 0,
                has_attribute_stride: false,
                is_fragment: false,
                stage: IcbRenderBindStage::Vertex,
            }],
            object_threadgroup_memory: vec![],
            draw: IcbRenderDraw::IndexedPatches {
                number_of_patch_control_points: 3,
                patch_start: 0,
                patch_count: 1,
                patch_index_buffer_ref: 0,
                patch_index_buffer_offset: 0,
                patch_index_wire_va: 0,
                control_point_index_buffer_ref: 13,
                control_point_index_buffer_offset: 0,
                control_point_index_wire_va: index_wire,
                instance_count: 1,
                base_instance: 0,
                tessellation_factor: IcbTessellationFactor {
                    buffer_ref: 12,
                    offset: 0,
                    wire_va: tess_wire,
                    instance_stride: 0,
                },
            },
        },
    )
    .expect("encode DrawIndexedPatches slot");

    associate_icb_command_memory(&mut host, &state, 7, &slot);

    fill_icb_from_command_memory(&state, &host, 1, 9, 0, 1)
        .expect("fill_icb_from_command_memory DrawIndexedPatches");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x48);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "wire-backed DrawIndexedPatches execute"
    );

    assert_target_is_shader_solid(
        &mut state,
        &mut host,
        mapping_id,
        "wire-backed DrawIndexedPatches",
    );
}

/// Object+mesh host fill: dual-function metallib (object type 8 + mesh type 7)
/// → drawMeshThreadgroups → solid BGRA (object sets mesh grid via payload).
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn fill_render_object_mesh_threadgroups_oracle() {
    use crate::runtime::decode::resource::MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS;

    let _guard = icb_test_guard();

    let om_mtlb = read_fixture("icb_object_mesh.metallib");
    let frag_mtlb = read_fixture("icb_mesh_frag.metallib");

    let (mut host, mut state) = icb_device();

    // maxObjectTG=1 so create body + materialize allow object TG memory binds.
    let icb_desc =
        make_render_icb_desc_bytes_ex(1, 0, 0, 0, 0, MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS, 0);
    // Patch max_object_tg create byte + layout with 1 object TG slot.
    {
        use crate::runtime::decode::resource::{
            encode_icb_command_layout, render_draw_mesh_threadgroups_icb_layout_ex,
            ICB_DESC_LAYOUT, ICB_DESC_MAX_OBJECT_TG_BINDS, ICB_LAYOUT_LEN,
        };
        let mut b = icb_desc.clone();
        b[ICB_DESC_MAX_OBJECT_TG_BINDS] = 1;
        let layout = render_draw_mesh_threadgroups_icb_layout_ex(0, 0, 1);
        b[ICB_DESC_LAYOUT..ICB_DESC_LAYOUT + ICB_LAYOUT_LEN]
            .copy_from_slice(&encode_icb_command_layout(&layout));
        let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
        put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &b);
    }

    put_function_object(&mut host, &state, 2, 0x200, 2, &om_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    fill_render(
        &state,
        &host,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![],
            // length 0 = clear; exercise API path with empty vec is fine.
            // Non-zero object TG mem optional for this payload-only shader.
            object_threadgroup_memory: vec![],
            draw: unit_mesh_draw(),
        },
    )
    .expect("fill object+mesh threadgroups");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x3e);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "object+mesh ICB execute"
    );

    assert_target_is_shader_solid(&mut state, &mut host, mapping_id, "object+mesh");
}

/// Dedicated wire-backed E2E: dual-export object+mesh metallib in classic
/// type-7 vertex slot (no mesh SPI tag 0x14) through command memory.
/// Not claimed via mesh SPI separate-ref E2E.
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn wire_backed_dual_export_object_mesh_e2e() {
    use crate::runtime::decode::resource::{
        decode_render_pipeline_descriptor, render_draw_mesh_threadgroups_icb_layout,
        MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS,
    };

    let _guard = icb_test_guard();

    let om_mtlb = read_fixture("icb_object_mesh.metallib");
    let frag_mtlb = read_fixture("icb_mesh_frag.metallib");

    let (mut host, mut state) = icb_device();

    let layout = render_draw_mesh_threadgroups_icb_layout();
    let icb_desc = make_render_icb_desc_bytes(1, 0, 0, MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    // Dual-export object+mesh in classic "vertex" function slot only.
    put_function_object(&mut host, &state, 2, 0x200, 2, &om_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    // Classic type-7 shape: tag 0x01/0x02 only (no mesh SPI 0x14).
    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    let decoded = decode_render_pipeline_descriptor(&pdesc).expect("classic pipeline");
    assert_eq!(decoded.vertex_func_ref, 2);
    assert_eq!(decoded.fragment_func_ref, 3);
    assert_eq!(decoded.object_func_ref, 0);
    assert_eq!(decoded.mesh_func_ref, 0);
    assert!(!decoded.has_color_attachment_offset || decoded.color_attachment_offset != 0);
    // Mesh SPI shape uses tag 0x14; classic stagein has tag 0x08.
    // object/mesh refs must stay zero so fill uses dual-export scan of vertex metallib.
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    let fill = IcbRenderFill {
        command_index: 0,
        pipeline_ref: 6,
        buffers: vec![],
        object_threadgroup_memory: vec![],
        draw: unit_mesh_draw(),
    };
    let slot = encode_render_command_slot(&layout, &fill).expect("encode dual-export slot");

    associate_icb_command_memory(&mut host, &state, 5, &slot);

    fill_icb_from_command_memory(&state, &host, 1, 9, 0, 1)
        .expect("fill_icb_from_command_memory dual-export object+mesh");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x4b);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "wire-backed dual-export object+mesh execute"
    );

    assert_target_is_shader_solid(
        &mut state,
        &mut host,
        mapping_id,
        "wire-backed dual-export object+mesh",
    );
}

/// Separate object + mesh + fragment function refs via mesh SPI type-7
/// tags 0x01 / 0x02 / 0x03 under section tag 0x14 (not dual-export).
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn fill_render_separate_object_mesh_func_refs_oracle() {
    use crate::runtime::decode::resource::{
        decode_render_pipeline_descriptor, MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS,
    };

    let _guard = icb_test_guard();

    let obj_mtlb = read_fixture("icb_object_stage.metallib");
    let mesh_mtlb = read_fixture("icb_mesh_with_payload.metallib");
    let frag_mtlb = read_fixture("icb_mesh_frag.metallib");

    let (mut host, mut state) = icb_device();

    let icb_desc =
        make_render_icb_desc_bytes_ex(1, 0, 0, 0, 0, MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS, 0);
    {
        use crate::runtime::decode::resource::{
            encode_icb_command_layout, render_draw_mesh_threadgroups_icb_layout_ex,
            ICB_DESC_LAYOUT, ICB_DESC_MAX_OBJECT_TG_BINDS, ICB_LAYOUT_LEN,
        };
        let mut b = icb_desc.clone();
        b[ICB_DESC_MAX_OBJECT_TG_BINDS] = 1;
        let layout = render_draw_mesh_threadgroups_icb_layout_ex(0, 0, 1);
        b[ICB_DESC_LAYOUT..ICB_DESC_LAYOUT + ICB_LAYOUT_LEN]
            .copy_from_slice(&encode_icb_command_layout(&layout));
        let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
        put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &b);
    }

    // Object function ref 2, mesh ref 4, fragment ref 3 — three distinct objects.
    put_function_object(&mut host, &state, 2, 0x200, 2, &obj_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    put_function_object(&mut host, &state, 4, 0x260, 4, &mesh_mtlb);

    let pdesc = make_mesh_render_pipeline_desc(Some(2), 4, 3);
    let decoded = decode_render_pipeline_descriptor(&pdesc).expect("mesh pipeline decode");
    assert_eq!(decoded.object_func_ref, 2);
    assert_eq!(decoded.mesh_func_ref, 4);
    assert_eq!(decoded.fragment_func_ref, 3);
    assert_eq!(decoded.vertex_func_ref, 0);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x280, &pdesc);

    fill_render(
        &state,
        &host,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![],
            object_threadgroup_memory: vec![],
            draw: unit_mesh_draw(),
        },
    )
    .expect("fill separate object+mesh+frag refs");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x3f);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "separate object/mesh func-ref ICB execute"
    );

    assert_target_is_shader_solid(
        &mut state,
        &mut host,
        mapping_id,
        "separate object/mesh refs",
    );
}

/// Dedicated wire-backed E2E: mesh SPI pipeline shape (tag 0x14 + object/
/// mesh/frag refs 0x01/0x02/0x03) through command memory — not dual-export
/// and not host `fill_render_command` API.
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn wire_backed_mesh_spi_pipeline_e2e() {
    use crate::runtime::decode::resource::{
        decode_render_pipeline_descriptor, render_draw_mesh_threadgroups_icb_layout,
        MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS,
    };

    let _guard = icb_test_guard();

    let obj_mtlb = read_fixture("icb_object_stage.metallib");
    let mesh_mtlb = read_fixture("icb_mesh_with_payload.metallib");
    let frag_mtlb = read_fixture("icb_mesh_frag.metallib");

    let (mut host, mut state) = icb_device();

    let layout = render_draw_mesh_threadgroups_icb_layout();
    let icb_desc = make_render_icb_desc_bytes(1, 0, 0, MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    // Three distinct function objects: object=2, frag=3, mesh=4.
    put_function_object(&mut host, &state, 2, 0x200, 2, &obj_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    put_function_object(&mut host, &state, 4, 0x260, 4, &mesh_mtlb);

    let pdesc = make_mesh_render_pipeline_desc(Some(2), 4, 3);
    let decoded = decode_render_pipeline_descriptor(&pdesc).expect("mesh SPI pipeline");
    assert_eq!(decoded.object_func_ref, 2);
    assert_eq!(decoded.mesh_func_ref, 4);
    assert_eq!(decoded.fragment_func_ref, 3);
    assert_eq!(decoded.vertex_func_ref, 0);
    assert!(decoded.has_color_attachment_offset); // tag 0x14 shape
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x280, &pdesc);

    let fill = IcbRenderFill {
        command_index: 0,
        pipeline_ref: 6,
        buffers: vec![],
        object_threadgroup_memory: vec![],
        draw: unit_mesh_draw(),
    };
    let slot = encode_render_command_slot(&layout, &fill).expect("encode mesh SPI slot");

    associate_icb_command_memory(&mut host, &state, 5, &slot);

    fill_icb_from_command_memory(&state, &host, 1, 9, 0, 1)
        .expect("fill_icb_from_command_memory mesh SPI pipeline");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x47);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "wire-backed mesh SPI pipeline execute"
    );

    assert_target_is_shader_solid(
        &mut state,
        &mut host,
        mapping_id,
        "wire-backed mesh SPI pipeline",
    );
}

/// Pixel: setMeshBuffer — mesh stage reads scale from buffer(0).
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn fill_render_mesh_buffer_bind_oracle() {
    use crate::runtime::decode::resource::MTL_INDIRECT_CMD_DRAW_MESH_THREADS;

    let _guard = icb_test_guard();

    let mesh_mtlb = read_fixture("icb_mesh_buf.metallib");
    let frag_mtlb = read_fixture("icb_mesh_frag.metallib");

    let (mut host, mut state) = icb_device();

    // max_mesh=1 so create/layout allow mesh buffer bind table.
    let icb_desc =
        make_render_icb_desc_bytes_ex(1, 0, 0, 0, 1, MTL_INDIRECT_CMD_DRAW_MESH_THREADS, 0);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &mesh_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    // scale = 1.0f LE (handle must be a setup_task-mapped page pfn).
    let scale_bytes = 1.0f32.to_le_bytes();
    put_type1_buffer(&mut host, &state, 7, 4, &scale_bytes);

    fill_render(
        &state,
        &host,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![IcbRenderBufferBind {
                index: 0,
                buffer_ref: 7,
                offset: 0,
                wire_va: 0,
                attribute_stride: 0,
                has_attribute_stride: false,
                is_fragment: false,
                stage: IcbRenderBindStage::Mesh,
            }],
            object_threadgroup_memory: vec![],
            draw: unit_mesh_threads_draw(),
        },
    )
    .expect("fill mesh buffer bind");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x40);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "mesh buffer bind ICB execute"
    );

    assert_target_is_shader_solid(&mut state, &mut host, mapping_id, "mesh buffer bind");
}

/// Pixel: setObjectBuffer — object stage reads scale from buffer(0) → payload.
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn fill_render_object_buffer_bind_oracle() {
    use crate::runtime::decode::resource::MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS;

    let _guard = icb_test_guard();

    let om_mtlb = read_fixture("icb_object_buf.metallib");
    let frag_mtlb = read_fixture("icb_mesh_frag.metallib");

    let (mut host, mut state) = icb_device();

    // max_object=1 for object buffer bind table.
    let icb_desc =
        make_render_icb_desc_bytes_ex(1, 0, 0, 1, 0, MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS, 0);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &om_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    let scale_bytes = 1.0f32.to_le_bytes();
    put_type1_buffer(&mut host, &state, 7, 4, &scale_bytes);

    fill_render(
        &state,
        &host,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![IcbRenderBufferBind {
                index: 0,
                buffer_ref: 7,
                offset: 0,
                wire_va: 0,
                attribute_stride: 0,
                has_attribute_stride: false,
                is_fragment: false,
                stage: IcbRenderBindStage::Object,
            }],
            object_threadgroup_memory: vec![],
            draw: unit_mesh_draw(),
        },
    )
    .expect("fill object buffer bind");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x41);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "object buffer bind ICB execute"
    );

    assert_target_is_shader_solid(&mut state, &mut host, mapping_id, "object buffer bind");
}

/// Wire-backed E2E: encode mesh buffer bind + MeshThreads → command memory
/// → 0x1d1 → fill_icb_from_command_memory → execute (setMeshBuffer path).
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn wire_backed_mesh_buffer_bind_e2e() {
    use crate::runtime::decode::resource::{
        encode_icb_command_layout, render_draw_mesh_threads_icb_layout_with_binds, ICB_DESC_LAYOUT,
        ICB_LAYOUT_LEN, MTL_INDIRECT_CMD_DRAW_MESH_THREADS,
    };

    let _guard = icb_test_guard();

    let mesh_mtlb = read_fixture("icb_mesh_buf.metallib");
    let frag_mtlb = read_fixture("icb_mesh_frag.metallib");

    let (mut host, mut state) = icb_device();

    let mut icb_desc =
        make_render_icb_desc_bytes_ex(1, 0, 0, 0, 1, MTL_INDIRECT_CMD_DRAW_MESH_THREADS, 0);
    let layout = render_draw_mesh_threads_icb_layout_with_binds(0, 1);
    icb_desc[ICB_DESC_LAYOUT..ICB_DESC_LAYOUT + ICB_LAYOUT_LEN]
        .copy_from_slice(&encode_icb_command_layout(&layout));
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &mesh_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    let scale_bytes = 1.0f32.to_le_bytes();
    put_type1_buffer(&mut host, &state, 7, 4, &scale_bytes);
    let scale_gva = 4u64 << RESOURCE_PAGE_SHIFT;

    let fill = IcbRenderFill {
        command_index: 0,
        pipeline_ref: 6,
        buffers: vec![IcbRenderBufferBind {
            index: 0,
            buffer_ref: 7,
            offset: 0,
            wire_va: scale_gva,
            attribute_stride: 0,
            has_attribute_stride: false,
            is_fragment: false,
            stage: IcbRenderBindStage::Mesh,
        }],
        object_threadgroup_memory: vec![],
        draw: unit_mesh_threads_draw(),
    };
    let slot = encode_render_command_slot(&layout, &fill).expect("encode mesh bind slot");

    associate_icb_command_memory(&mut host, &state, 5, &slot);

    fill_icb_from_command_memory(&state, &host, 1, 9, 0, 1)
        .expect("fill_icb_from_command_memory mesh buffer bind");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x42);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "wire-backed mesh buffer bind execute"
    );

    assert_target_is_shader_solid(
        &mut state,
        &mut host,
        mapping_id,
        "wire-backed mesh buffer bind",
    );
}

/// Wire-backed E2E: encode object buffer bind + MeshThreadgroups → command
/// memory → 0x1d1 → fill_icb_from_command_memory → execute (setObjectBuffer).
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn wire_backed_object_buffer_bind_e2e() {
    use crate::runtime::decode::resource::{
        encode_icb_command_layout, render_draw_mesh_threadgroups_icb_layout_ex, ICB_DESC_LAYOUT,
        ICB_LAYOUT_LEN, MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS,
    };

    let _guard = icb_test_guard();

    let om_mtlb = read_fixture("icb_object_buf.metallib");
    let frag_mtlb = read_fixture("icb_mesh_frag.metallib");

    let (mut host, mut state) = icb_device();

    let mut icb_desc =
        make_render_icb_desc_bytes_ex(1, 0, 0, 1, 0, MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS, 0);
    let layout = render_draw_mesh_threadgroups_icb_layout_ex(1, 0, 0);
    icb_desc[ICB_DESC_LAYOUT..ICB_DESC_LAYOUT + ICB_LAYOUT_LEN]
        .copy_from_slice(&encode_icb_command_layout(&layout));
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &om_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    let scale_bytes = 1.0f32.to_le_bytes();
    put_type1_buffer(&mut host, &state, 7, 4, &scale_bytes);
    let scale_gva = 4u64 << RESOURCE_PAGE_SHIFT;

    let fill = IcbRenderFill {
        command_index: 0,
        pipeline_ref: 6,
        buffers: vec![IcbRenderBufferBind {
            index: 0,
            buffer_ref: 7,
            offset: 0,
            wire_va: scale_gva,
            attribute_stride: 0,
            has_attribute_stride: false,
            is_fragment: false,
            stage: IcbRenderBindStage::Object,
        }],
        object_threadgroup_memory: vec![],
        draw: unit_mesh_draw(),
    };
    let slot = encode_render_command_slot(&layout, &fill).expect("encode object bind slot");

    associate_icb_command_memory(&mut host, &state, 5, &slot);

    fill_icb_from_command_memory(&state, &host, 1, 9, 0, 1)
        .expect("fill_icb_from_command_memory object buffer bind");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x43);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "wire-backed object buffer bind execute"
    );

    assert_target_is_shader_solid(
        &mut state,
        &mut host,
        mapping_id,
        "wire-backed object buffer bind",
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

/// Dedicated: non-multiple-of-16 object TG length is fail-closed (Args).
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn fill_render_object_tg_memory_bad_length_rejected() {
    use crate::runtime::decode::resource::MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS;

    let _guard = icb_test_guard();

    let om_mtlb = read_fixture("icb_object_tg.metallib");
    let frag_mtlb = read_fixture("icb_mesh_frag.metallib");

    let (mut host, state) = icb_device();

    {
        use crate::runtime::decode::resource::{
            encode_icb_command_layout, render_draw_mesh_threadgroups_icb_layout_ex,
            ICB_DESC_LAYOUT, ICB_DESC_MAX_OBJECT_TG_BINDS, ICB_LAYOUT_LEN,
        };
        let mut b = make_render_icb_desc_bytes_ex(
            1,
            0,
            0,
            0,
            0,
            MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS,
            0,
        );
        b[ICB_DESC_MAX_OBJECT_TG_BINDS] = 1;
        let layout = render_draw_mesh_threadgroups_icb_layout_ex(0, 0, 1);
        b[ICB_DESC_LAYOUT..ICB_DESC_LAYOUT + ICB_LAYOUT_LEN]
            .copy_from_slice(&encode_icb_command_layout(&layout));
        let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
        put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &b);
    }

    put_function_object(&mut host, &state, 2, 0x200, 2, &om_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    let err = fill_render(
        &state,
        &host,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![],
            object_threadgroup_memory: vec![IcbThreadgroupMemory {
                index: 0,
                length: 8, // not multiple of 16
            }],
            draw: unit_mesh_draw(),
        },
    );
    assert_eq!(
        err,
        Err(IcbStatus::Args("icb_frc_object_tg_length_alignment")),
        "length 8 must fail closed, and name the alignment check"
    );
}

/// Dedicated host-fill: object TG length 16 used by object shader → solid BGRA.
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn fill_render_object_tg_memory_oracle() {
    use crate::runtime::decode::resource::MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS;

    let _guard = icb_test_guard();

    let om_mtlb = read_fixture("icb_object_tg.metallib");
    let frag_mtlb = read_fixture("icb_mesh_frag.metallib");

    let (mut host, mut state) = icb_device();

    {
        use crate::runtime::decode::resource::{
            encode_icb_command_layout, render_draw_mesh_threadgroups_icb_layout_ex,
            ICB_DESC_LAYOUT, ICB_DESC_MAX_OBJECT_TG_BINDS, ICB_LAYOUT_LEN,
        };
        let mut b = make_render_icb_desc_bytes_ex(
            1,
            0,
            0,
            0,
            0,
            MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS,
            0,
        );
        b[ICB_DESC_MAX_OBJECT_TG_BINDS] = 1;
        let layout = render_draw_mesh_threadgroups_icb_layout_ex(0, 0, 1);
        b[ICB_DESC_LAYOUT..ICB_DESC_LAYOUT + ICB_LAYOUT_LEN]
            .copy_from_slice(&encode_icb_command_layout(&layout));
        let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
        put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &b);
    }

    put_function_object(&mut host, &state, 2, 0x200, 2, &om_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    fill_render(
        &state,
        &host,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![],
            object_threadgroup_memory: vec![IcbThreadgroupMemory {
                index: 0,
                length: 16,
            }],
            draw: unit_mesh_draw(),
        },
    )
    .expect("fill object TG memory");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x45);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "object TG memory ICB execute"
    );

    assert_target_is_shader_solid(&mut state, &mut host, mapping_id, "object TG memory");
}

/// Dedicated wire-backed E2E: object TG length 16 through command memory.
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn wire_backed_object_tg_memory_e2e() {
    use crate::runtime::decode::resource::{
        encode_icb_command_layout, render_draw_mesh_threadgroups_icb_layout_ex, ICB_DESC_LAYOUT,
        ICB_DESC_MAX_OBJECT_TG_BINDS, ICB_LAYOUT_LEN, MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS,
    };

    let _guard = icb_test_guard();

    let om_mtlb = read_fixture("icb_object_tg.metallib");
    let frag_mtlb = read_fixture("icb_mesh_frag.metallib");

    let (mut host, mut state) = icb_device();

    let layout = render_draw_mesh_threadgroups_icb_layout_ex(0, 0, 1);
    let mut icb_desc =
        make_render_icb_desc_bytes_ex(1, 0, 0, 0, 0, MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS, 0);
    icb_desc[ICB_DESC_MAX_OBJECT_TG_BINDS] = 1;
    icb_desc[ICB_DESC_LAYOUT..ICB_DESC_LAYOUT + ICB_LAYOUT_LEN]
        .copy_from_slice(&encode_icb_command_layout(&layout));
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &om_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    let fill = IcbRenderFill {
        command_index: 0,
        pipeline_ref: 6,
        buffers: vec![],
        object_threadgroup_memory: vec![IcbThreadgroupMemory {
            index: 0,
            length: 16,
        }],
        draw: unit_mesh_draw(),
    };
    let slot = encode_render_command_slot(&layout, &fill).expect("encode object TG slot");

    associate_icb_command_memory(&mut host, &state, 5, &slot);

    fill_icb_from_command_memory(&state, &host, 1, 9, 0, 1)
        .expect("fill_icb_from_command_memory object TG");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x46);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "wire-backed object TG execute"
    );

    assert_target_is_shader_solid(&mut state, &mut host, mapping_id, "wire-backed object TG");
}

/// Wire-backed E2E: drawMeshThreads.
///
/// Same guest path as patches: encode slot into type-1 command memory,
/// `0x1d1` bind, execute re-fills from wire (no host `fill_render_command`).
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn wire_backed_mesh_threads_e2e() {
    use crate::runtime::decode::resource::{
        render_draw_mesh_threads_icb_layout, MTL_INDIRECT_CMD_DRAW_MESH_THREADS,
    };

    let _guard = icb_test_guard();

    let mesh_mtlb = read_fixture("icb_mesh.metallib");
    let frag_mtlb = read_fixture("icb_mesh_frag.metallib");

    let (mut host, mut state) = icb_device();

    let layout = render_draw_mesh_threads_icb_layout(0);
    let icb_desc = make_render_icb_desc_bytes(1, 0, 0, MTL_INDIRECT_CMD_DRAW_MESH_THREADS);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &mesh_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    let slot = encode_render_command_slot(
        &layout,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![],
            object_threadgroup_memory: vec![],
            draw: unit_mesh_threads_draw(),
        },
    )
    .expect("encode MeshThreads slot");

    associate_icb_command_memory(&mut host, &state, 6, &slot);

    fill_icb_from_command_memory(&state, &host, 1, 9, 0, 1)
        .expect("fill_icb_from_command_memory MeshThreads");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x3d);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "wire-backed MeshThreads execute"
    );

    assert_target_is_shader_solid(&mut state, &mut host, mapping_id, "wire-backed MeshThreads");
}

/// Dedicated wire-backed E2E: drawMeshThreadgroups (no object buffer binds).
///
/// Separate from object-buffer E2E so MeshThreadgroups command-memory path
/// cannot regress silently when that other test is simplified.
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn wire_backed_mesh_threadgroups_e2e() {
    use crate::runtime::decode::resource::{
        render_draw_mesh_threadgroups_icb_layout, MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS,
    };

    let _guard = icb_test_guard();

    let mesh_mtlb = read_fixture("icb_mesh.metallib");
    let frag_mtlb = read_fixture("icb_mesh_frag.metallib");

    let (mut host, mut state) = icb_device();

    let layout = render_draw_mesh_threadgroups_icb_layout();
    let icb_desc = make_render_icb_desc_bytes(1, 0, 0, MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &mesh_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    let slot = encode_render_command_slot(
        &layout,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![],
            object_threadgroup_memory: vec![],
            draw: unit_mesh_draw(),
        },
    )
    .expect("encode MeshThreadgroups slot");

    associate_icb_command_memory(&mut host, &state, 6, &slot);

    fill_icb_from_command_memory(&state, &host, 1, 9, 0, 1)
        .expect("fill_icb_from_command_memory MeshThreadgroups");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x44);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "wire-backed MeshThreadgroups execute"
    );

    assert_target_is_shader_solid(
        &mut state,
        &mut host,
        mapping_id,
        "wire-backed MeshThreadgroups",
    );
}

/// inheritBuffers=true: ICB slot records only draw+PSO; fragment color
/// comes from the parent encoder (stream bind path).
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn inherit_buffers_encoder_fragment_color() {
    let _guard = icb_test_guard();

    let (vert_mtlb, frag_mtlb) = load_oracle_mtlb();
    let (mut host, mut state) = icb_device();

    // inheritBuffers bit1; maxFragment still advertises encoder bind capacity.
    let icb_desc = make_render_icb_desc_bytes_flags(
        1,
        0,
        1,
        MTL_INDIRECT_CMD_DRAW_INDEXED,
        ICB_FLAG_INHERIT_BUFFERS,
    );
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &vert_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_render_pipeline_desc(2, 3);
    let pdesc_gva = 0x240u64;
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, pdesc_gva, &pdesc);

    let indices: [u16; 3] = [0, 1, 2];
    let index_bytes: Vec<u8> = indices.iter().flat_map(|v| v.to_le_bytes()).collect();
    put_type1_buffer(&mut host, &state, 12, 4, &index_bytes);

    let sid = 7u8;
    let r = (0x60u32 + sid as u32) as f32 / 255.0;
    let color = [r, 0x44 as f32 / 255.0, 0x22 as f32 / 255.0, 1.0f32];
    let color_bytes: Vec<u8> = color.iter().flat_map(|f| f.to_le_bytes()).collect();
    put_type1_buffer(&mut host, &state, 13, 5, &color_bytes);

    // Fill: pipeline + draw only — no fragment buffer in the ICB slot.
    fill_render(
        &state,
        &host,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![],
            object_threadgroup_memory: vec![],
            draw: indexed_draw(0),
        },
    )
    .expect("fill inherit draw");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x32);

    // Stream-style fragment bind on the encode request (parent encoder).
    let req = DrawEncodeRequest {
        fragment_buffers: vec![BufferBind {
            index: 0,
            buffer_ref: 13,
            offset: 0,
            attribute_stride: None,
            ..Default::default()
        }]
        .into(),
        viewports: vec![[0.0, 0.0, 4.0, 4.0, 0.0, 1.0]],
        ..draw_request(mapping_id)
    };
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "inheritBuffers encoder fragment path"
    );

    assert_stagein_solid(
        &mut state,
        &mut host,
        mapping_id,
        sid,
        "inheritBuffers encoder",
    );
}

/// inheritPipelineState=true: ICB slot records only buffers+draw; PSO
/// comes from the parent render encoder (stream pipeline bind path).
/// Mirrors `inherit_pipeline_encoder_kernel_mul3add1` for compute.
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn inherit_pipeline_encoder_fragment_color() {
    use crate::runtime::decode::resource::ICB_FLAG_INHERIT_PIPELINE_STATE;

    let _guard = icb_test_guard();

    let (vert_mtlb, frag_mtlb) = load_oracle_mtlb();
    let (mut host, mut state) = icb_device();

    // inheritPipelineState bit0; maxFragment for fragment color in ICB slot.
    let icb_desc = make_render_icb_desc_bytes_flags(
        1,
        0,
        1,
        MTL_INDIRECT_CMD_DRAW_INDEXED,
        ICB_FLAG_INHERIT_PIPELINE_STATE,
    );
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &vert_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_render_pipeline_desc(2, 3);
    let pdesc_gva = 0x240u64;
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, pdesc_gva, &pdesc);

    let indices: [u16; 3] = [0, 1, 2];
    let index_bytes: Vec<u8> = indices.iter().flat_map(|v| v.to_le_bytes()).collect();
    put_type1_buffer(&mut host, &state, 12, 4, &index_bytes);

    let sid = 7u8;
    let r = (0x60u32 + sid as u32) as f32 / 255.0;
    let color = [r, 0x44 as f32 / 255.0, 0x22 as f32 / 255.0, 1.0f32];
    let color_bytes: Vec<u8> = color.iter().flat_map(|f| f.to_le_bytes()).collect();
    put_type1_buffer(&mut host, &state, 13, 5, &color_bytes);

    // Fill: buffers + draw only — pipeline_ref 0 (inherited from parent).
    fill_render(
        &state,
        &host,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 0,
            buffers: vec![render_bind(0, 13, true)],
            object_threadgroup_memory: vec![],
            draw: indexed_draw(0),
        },
    )
    .expect("fill inheritPipeline draw");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x34);

    // Stream-style pipeline on the encode request (parent encoder).
    let req = DrawEncodeRequest {
        viewports: vec![[0.0, 0.0, 4.0, 4.0, 0.0, 1.0]],
        ..draw_request(mapping_id)
    };
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "inheritPipelineState encoder fragment path"
    );

    assert_stagein_solid(
        &mut state,
        &mut host,
        mapping_id,
        sid,
        "inheritPipelineState encoder",
    );
}

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

/// ICB Draw with `[[stage_in]]` vertex descriptor: position buffer +
/// fragment color → solid BGRA (render_stagein fixtures).
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn fill_render_stagein_draw_execute_oracle() {
    let _guard = icb_test_guard();

    let (vert_mtlb, frag_mtlb) = load_stagein_mtlb();
    let (mut host, mut state) = icb_device();

    // Draw (not indexed): max_vertex=1 for position buffer bind.
    let icb_desc = make_render_icb_desc_bytes(1, 1, 1, MTL_INDIRECT_CMD_DRAW);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &vert_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    let pdesc_gva = 0x240u64;
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, pdesc_gva, &pdesc);

    // Fullscreen triangle positions (clip space), stride 16 Float4.
    let tri: [[f32; 4]; 3] = [
        [-1.0, -1.0, 0.0, 1.0],
        [3.0, -1.0, 0.0, 1.0],
        [-1.0, 3.0, 0.0, 1.0],
    ];
    let pos_bytes: Vec<u8> = tri
        .iter()
        .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
        .collect();
    put_type1_buffer(&mut host, &state, 11, 4, &pos_bytes);

    let sid = 7u8;
    let r = (0x60u32 + sid as u32) as f32 / 255.0;
    let color = [r, 0x44 as f32 / 255.0, 0x22 as f32 / 255.0, 1.0f32];
    let color_bytes: Vec<u8> = color.iter().flat_map(|f| f.to_le_bytes()).collect();
    put_type1_buffer(&mut host, &state, 13, 5, &color_bytes);

    fill_render(
        &state,
        &host,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![render_bind(0, 11, false), render_bind(0, 13, true)],
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
    .expect("fill_render stage_in Draw");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x33);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "stage_in ICB execute"
    );

    assert_stagein_solid(&mut state, &mut host, mapping_id, sid, "ICB stage_in");
}

/// Dedicated wire-backed E2E: classic Draw (`0x1`) with stage_in vertex
/// buffer + fragment color — not DrawIndexed and not host fill API.
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn wire_backed_draw_primitives_stagein_e2e() {
    use crate::runtime::decode::resource::{render_icb_layout, MTL_INDIRECT_CMD_DRAW};

    let _guard = icb_test_guard();

    let (vert_mtlb, frag_mtlb) = load_stagein_mtlb();
    let (mut host, mut state) = icb_device();

    let layout = render_icb_layout(1, 1, MTL_INDIRECT_CMD_DRAW);
    let icb_desc = make_render_icb_desc_bytes(1, 1, 1, MTL_INDIRECT_CMD_DRAW);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &vert_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    let tri: [[f32; 4]; 3] = [
        [-1.0, -1.0, 0.0, 1.0],
        [3.0, -1.0, 0.0, 1.0],
        [-1.0, 3.0, 0.0, 1.0],
    ];
    let pos_bytes: Vec<u8> = tri
        .iter()
        .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
        .collect();
    put_type1_buffer(&mut host, &state, 11, 4, &pos_bytes);

    let sid = 7u8;
    let r = (0x60u32 + sid as u32) as f32 / 255.0;
    let color = [r, 0x44 as f32 / 255.0, 0x22 as f32 / 255.0, 1.0f32];
    let color_bytes: Vec<u8> = color.iter().flat_map(|f| f.to_le_bytes()).collect();
    put_type1_buffer(&mut host, &state, 13, 5, &color_bytes);

    let pos_wire = (4u64) << RESOURCE_PAGE_SHIFT;
    let color_wire = (5u64) << RESOURCE_PAGE_SHIFT;

    let slot = encode_render_command_slot(
        &layout,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![
                IcbRenderBufferBind {
                    index: 0,
                    buffer_ref: 11,
                    offset: 0,
                    wire_va: pos_wire,
                    attribute_stride: 0,
                    has_attribute_stride: false,
                    is_fragment: false,
                    stage: IcbRenderBindStage::Vertex,
                },
                IcbRenderBufferBind {
                    index: 0,
                    buffer_ref: 13,
                    offset: 0,
                    wire_va: color_wire,
                    attribute_stride: 0,
                    has_attribute_stride: false,
                    is_fragment: true,
                    stage: IcbRenderBindStage::Fragment,
                },
            ],
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
    .expect("encode Draw primitives slot");

    associate_icb_command_memory(&mut host, &state, 6, &slot);

    fill_icb_from_command_memory(&state, &host, 1, 9, 0, 1)
        .expect("fill_icb_from_command_memory Draw primitives");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x49);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "wire-backed Draw primitives execute"
    );

    assert_stagein_solid(
        &mut state,
        &mut host,
        mapping_id,
        sid,
        "wire-backed Draw stage_in",
    );
}

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

/// Host fill with attributeStride on vertex bind — stage_in Draw still solid.
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn fill_render_attribute_stride_stagein_execute() {
    let _guard = icb_test_guard();

    let (vert_mtlb, frag_mtlb) = load_stagein_mtlb();
    let (mut host, mut state) = icb_device();

    let icb_desc = make_render_icb_desc_bytes(1, 1, 1, MTL_INDIRECT_CMD_DRAW);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &vert_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    // Tight Float4 positions (stride 16) — attributeStride matches layout stride.
    let tri: [[f32; 4]; 3] = [
        [-1.0, -1.0, 0.0, 1.0],
        [3.0, -1.0, 0.0, 1.0],
        [-1.0, 3.0, 0.0, 1.0],
    ];
    let pos_bytes: Vec<u8> = tri
        .iter()
        .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
        .collect();
    put_type1_buffer(&mut host, &state, 11, 4, &pos_bytes);

    let sid = 7u8;
    let r = (0x60u32 + sid as u32) as f32 / 255.0;
    let color = [r, 0x44 as f32 / 255.0, 0x22 as f32 / 255.0, 1.0f32];
    let color_bytes: Vec<u8> = color.iter().flat_map(|f| f.to_le_bytes()).collect();
    put_type1_buffer(&mut host, &state, 13, 5, &color_bytes);

    fill_render(
        &state,
        &host,
        &IcbRenderFill {
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
        },
    )
    .expect("fill with attributeStride");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x36);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok
    );

    assert_stagein_solid(
        &mut state,
        &mut host,
        mapping_id,
        sid,
        "attributeStride stage_in",
    );
}

/// Dedicated wire-backed E2E: vertex attributeStride=16 through command
/// memory (not host fill API). Proves stride table encode → decode →
/// setVertexBuffer:offset:attributeStride:.
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn wire_backed_attribute_stride_stagein_e2e() {
    use crate::runtime::decode::resource::{render_icb_layout, MTL_INDIRECT_CMD_DRAW};

    let _guard = icb_test_guard();

    let (vert_mtlb, frag_mtlb) = load_stagein_mtlb();
    let (mut host, mut state) = icb_device();

    let layout = render_icb_layout(1, 1, MTL_INDIRECT_CMD_DRAW);
    assert!(icb_layout_attribute_stride_slot_count(&layout) >= 1);
    let icb_desc = make_render_icb_desc_bytes(1, 1, 1, MTL_INDIRECT_CMD_DRAW);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &vert_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_stagein_render_pipeline_desc(2, 3);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    let tri: [[f32; 4]; 3] = [
        [-1.0, -1.0, 0.0, 1.0],
        [3.0, -1.0, 0.0, 1.0],
        [-1.0, 3.0, 0.0, 1.0],
    ];
    let pos_bytes: Vec<u8> = tri
        .iter()
        .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
        .collect();
    put_type1_buffer(&mut host, &state, 11, 4, &pos_bytes);

    let sid = 7u8;
    let r = (0x60u32 + sid as u32) as f32 / 255.0;
    let color = [r, 0x44 as f32 / 255.0, 0x22 as f32 / 255.0, 1.0f32];
    let color_bytes: Vec<u8> = color.iter().flat_map(|f| f.to_le_bytes()).collect();
    put_type1_buffer(&mut host, &state, 13, 5, &color_bytes);

    let pos_wire = (4u64) << RESOURCE_PAGE_SHIFT;
    let color_wire = (5u64) << RESOURCE_PAGE_SHIFT;

    let fill = IcbRenderFill {
        command_index: 0,
        pipeline_ref: 6,
        buffers: vec![
            IcbRenderBufferBind {
                index: 0,
                buffer_ref: 11,
                offset: 0,
                wire_va: pos_wire,
                attribute_stride: 16,
                has_attribute_stride: true,
                is_fragment: false,
                stage: IcbRenderBindStage::Vertex,
            },
            IcbRenderBufferBind {
                index: 0,
                buffer_ref: 13,
                offset: 0,
                wire_va: color_wire,
                attribute_stride: 0,
                has_attribute_stride: false,
                is_fragment: true,
                stage: IcbRenderBindStage::Fragment,
            },
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
    let slot = encode_render_command_slot(&layout, &fill).expect("encode attributeStride slot");
    // Wire carries non-zero stride at attributeStrideOffset.
    assert_eq!(
        ld64(&slot[layout.attribute_stride_offset as usize..]),
        16,
        "attribute stride table on wire"
    );

    associate_icb_command_memory(&mut host, &state, 6, &slot);

    fill_icb_from_command_memory(&state, &host, 1, 9, 0, 1)
        .expect("fill_icb_from_command_memory attributeStride");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x4a);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok,
        "wire-backed attributeStride execute"
    );

    assert_stagein_solid(
        &mut state,
        &mut host,
        mapping_id,
        sid,
        "wire-backed attributeStride",
    );
}

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

/// Host fill + execute with barrier + threadgroup memory lengths (no shader
/// dependency — mul3add1 ignores TG mem; verifies Metal accepts the record).
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn fill_compute_barrier_and_tg_memory_execute() {
    use crate::runtime::compute_session::ComputeSession;

    let (_guard, mtlb, mut host, mut state) = mul3add1_fixture();

    let icb_desc = make_icb_desc_bytes_tg(1, 1, 1, 0);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 5, 0x100, 2, &mtlb);

    let pdesc = make_compute_pipeline_desc(5);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x140, &pdesc);

    let data = [1u32, 2, 3, 4];
    let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf_gva = 3u64 << RESOURCE_PAGE_SHIFT;
    gva_mem::write_task_gva_arm64e(&mut host, &state.tasks[1], buf_gva, &data_bytes);
    let mut bdesc = vec![0u8; 16];
    st64(&mut bdesc[0..], 16);
    st32(&mut bdesc[8..], 3);
    put_object(&mut host, &state, 7, OBJECT_TYPE_BUFFER, 0x180, &bdesc);

    fill_compute(
        &state,
        &host,
        &IcbComputeFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![kernel_bind(0, 7)],
            threadgroup_memory: vec![IcbThreadgroupMemory {
                index: 0,
                length: 16,
            }],
            barrier: true,
            dispatch: unit_grid_dispatch(4, 1, 1),
        },
    )
    .expect("fill with barrier+tg");

    let mut session = ComputeSession::open(0).expect("session");
    let cmd = execute_icb_command(9, 0, 1);
    assert_eq!(
        session.encode_icb(
            &mut state,
            &mut host,
            1,
            &cmd,
            &crate::runtime::compute_exec::ComputeAccum::default()
        ),
        crate::runtime::compute_exec::ComputeStatus::Ok
    );
    assert_eq!(
        session.finish(&mut host, &mut state, 1),
        crate::runtime::compute_exec::ComputeStatus::Ok
    );

    let out = read_u32x4(&host, &state, buf_gva);
    assert_eq!(out, vec![4, 7, 10, 13], "barrier+tg fill still mul3add1");
}

/// inheritBuffers=true: ICB slot records only PSO+dispatch; kernel buffer
/// comes from the parent compute encoder (stream `ComputeAccum`).
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn inherit_buffers_encoder_kernel_mul3add1() {
    use crate::runtime::compute_exec::{ComputeAccum, ComputeBufferBind};
    use crate::runtime::compute_session::ComputeSession;

    let (_guard, mtlb, mut host, mut state) = mul3add1_fixture();

    // inheritBuffers; maxKernel still advertises encoder bind capacity.
    let icb_desc = make_icb_desc_bytes_tg(1, 1, 0, ICB_FLAG_INHERIT_BUFFERS);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 5, 0x100, 2, &mtlb);

    let pdesc = make_compute_pipeline_desc(5);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x140, &pdesc);

    let data = [1u32, 2, 3, 4];
    let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf_gva = 3u64 << RESOURCE_PAGE_SHIFT;
    gva_mem::write_task_gva_arm64e(&mut host, &state.tasks[1], buf_gva, &data_bytes);
    let mut bdesc = vec![0u8; 16];
    st64(&mut bdesc[0..], 16);
    st32(&mut bdesc[8..], 3);
    put_object(&mut host, &state, 7, OBJECT_TYPE_BUFFER, 0x180, &bdesc);

    // Fill: pipeline + dispatch only — no kernel buffer in the ICB slot.
    fill_compute(
        &state,
        &host,
        &IcbComputeFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![],
            threadgroup_memory: vec![],
            barrier: false,
            dispatch: unit_grid_dispatch(4, 1, 1),
        },
    )
    .expect("fill inheritBuffers dispatch");

    // Stream-style kernel bind on the parent encoder.
    let mut acc = ComputeAccum::default();
    acc.buffers.push(ComputeBufferBind {
        index: 0,
        buffer_ref: 7,
        offset: 0,
        attribute_stride: 0,
        has_attribute_stride: false,
    });

    let mut session = ComputeSession::open(0).expect("session");
    let cmd = execute_icb_command(9, 0, 1);
    assert_eq!(
        session.encode_icb(&mut state, &mut host, 1, &cmd, &acc),
        crate::runtime::compute_exec::ComputeStatus::Ok,
        "inheritBuffers encoder kernel path"
    );
    assert_eq!(
        session.finish(&mut host, &mut state, 1),
        crate::runtime::compute_exec::ComputeStatus::Ok
    );

    let out = read_u32x4(&host, &state, buf_gva);
    assert_eq!(
        out,
        vec![4, 7, 10, 13],
        "inheritBuffers encoder mul3add1 writeback"
    );
}

/// inheritPipelineState=true: ICB slot records only buffers+dispatch; PSO
/// comes from the parent compute encoder.
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn inherit_pipeline_encoder_kernel_mul3add1() {
    use crate::runtime::compute_exec::ComputeAccum;
    use crate::runtime::compute_session::ComputeSession;
    use crate::runtime::decode::resource::ICB_FLAG_INHERIT_PIPELINE_STATE;

    let (_guard, mtlb, mut host, mut state) = mul3add1_fixture();

    let icb_desc = make_icb_desc_bytes_tg(1, 1, 0, ICB_FLAG_INHERIT_PIPELINE_STATE);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 5, 0x100, 2, &mtlb);

    let pdesc = make_compute_pipeline_desc(5);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x140, &pdesc);

    let data = [1u32, 2, 3, 4];
    let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf_gva = 3u64 << RESOURCE_PAGE_SHIFT;
    gva_mem::write_task_gva_arm64e(&mut host, &state.tasks[1], buf_gva, &data_bytes);
    let mut bdesc = vec![0u8; 16];
    st64(&mut bdesc[0..], 16);
    st32(&mut bdesc[8..], 3);
    put_object(&mut host, &state, 7, OBJECT_TYPE_BUFFER, 0x180, &bdesc);

    // Fill: buffers + dispatch only — pipeline_ref 0 (inherited).
    fill_compute(
        &state,
        &host,
        &IcbComputeFill {
            command_index: 0,
            pipeline_ref: 0,
            buffers: vec![kernel_bind(0, 7)],
            threadgroup_memory: vec![],
            barrier: false,
            dispatch: unit_grid_dispatch(4, 1, 1),
        },
    )
    .expect("fill inheritPipeline dispatch");

    let mut acc = ComputeAccum::default();
    acc.set_pipeline(6);

    let mut session = ComputeSession::open(0).expect("session");
    let cmd = execute_icb_command(9, 0, 1);
    assert_eq!(
        session.encode_icb(&mut state, &mut host, 1, &cmd, &acc),
        crate::runtime::compute_exec::ComputeStatus::Ok,
        "inheritPipelineState encoder path"
    );
    assert_eq!(
        session.finish(&mut host, &mut state, 1),
        crate::runtime::compute_exec::ComputeStatus::Ok
    );

    let out = read_u32x4(&host, &state, buf_gva);
    assert_eq!(
        out,
        vec![4, 7, 10, 13],
        "inheritPipelineState encoder mul3add1 writeback"
    );
}

/// Install a type-2 linear texture (single level) and write seed texels.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
// A test fixture builder: object ref, handle, geometry and seed texels.
#[allow(clippy::too_many_arguments)]
fn put_type2_texture(
    host: &mut FakeHost,
    state: &mut DeviceState,
    obj_ref: u32,
    handle: u32,
    width: u32,
    height: u32,
    pixel_format: u16,
    seed: &[u8],
) {
    use crate::runtime::decode::resource::{
        OBJECT_TYPE_TEXTURE, TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_DATA_OFFSET, TEXTURE_DESC_HEIGHT,
        TEXTURE_DESC_MIPMAP_LEVEL_COUNT, TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE,
        TEXTURE_DESC_USED_SIZE, TEXTURE_DESC_WIDTH,
    };
    let bpp = crate::contract::pixel_format::bytes_per_pixel(pixel_format).unwrap_or(4);
    let row_stride = width * bpp;
    let size = (row_stride as u64) * (height as u64);
    let mut desc = vec![0u8; TEXTURE_DESC_BASE_LEN];
    st64(&mut desc[0..], size.max(0x1000));
    st32(&mut desc[8..], handle);
    st16(&mut desc[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 1);
    st32(&mut desc[TEXTURE_DESC_DATA_OFFSET..], 0);
    st32(&mut desc[TEXTURE_DESC_USED_SIZE..], size as u32);
    st32(&mut desc[TEXTURE_DESC_ROW_STRIDE..], row_stride);
    st32(&mut desc[TEXTURE_DESC_WIDTH..], width);
    st32(&mut desc[TEXTURE_DESC_HEIGHT..], height);
    st16(&mut desc[TEXTURE_DESC_PIXEL_FORMAT..], pixel_format);
    // Descriptor GVAs stay below the first resource page (handle<<14).
    let desc_gva = 0x200u64 + (obj_ref as u64) * 0x80;
    put_object(host, state, obj_ref, OBJECT_TYPE_TEXTURE, desc_gva, &desc);
    let data_gva = (handle as u64) << RESOURCE_PAGE_SHIFT;
    let mut page = vec![0u8; size as usize];
    let n = seed.len().min(page.len());
    page[..n].copy_from_slice(&seed[..n]);
    gva_mem::write_task_gva_arm64e(host, &state.tasks[1], data_gva, &page);
}

/// Minimal type-7 sampler (36 B): clamp-to-edge, nearest, normalized coords.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn put_type7_sampler(host: &mut FakeHost, state: &DeviceState, obj_ref: u32, normalized: bool) {
    use crate::runtime::decode::resource::{sampler_desc as off, TYPE7_OBJECT_SAMPLER};
    let mut desc = vec![0u8; off::LEN];
    st32(&mut desc[off::TAG..], TYPE7_OBJECT_SAMPLER);
    st32(&mut desc[off::DECLARED_LEN..], off::LEN as u32);
    st32(&mut desc[off::ID..], obj_ref);
    // Address modes ClampToEdge=0 at bits 8/12/16; filters nearest=0.
    // Normalized coords bit 31 when requested.
    let mut bits = 0u32;
    if normalized {
        bits |= 0x8000_0000;
    }
    st32(&mut desc[off::STATE_BITS..], bits);
    st32(&mut desc[off::FLAGS..], 0);
    st32(&mut desc[off::LOD_MIN..], 0f32.to_bits());
    st32(&mut desc[off::LOD_MAX..], f32::MAX.to_bits());
    let desc_gva = 0x300u64 + (obj_ref as u64) * 0x40;
    put_object(host, state, obj_ref, OBJECT_TYPE_TYPE7, desc_gva, &desc);
}

/// Textures/samplers always bind on the parent compute encoder at ICB
/// execute (classic `MTLIndirectComputeCommand` has no setTexture/setSampler).
///
/// Metal on this stack rejects `supportIndirectCommandBuffers` for kernels
/// that declare direct `texture`/`sampler` arguments — those need argument
/// buffers. Product still applies stream texture/sampler state before
/// `executeCommandsInBuffer`. This oracle uses ICB-capable mul3add1 for the
/// dispatch while staging a storage texture + sampler on the encoder:
/// buffer writeback proves execute; texture writeback preserves seed
/// (kernel does not touch the texture).
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn icb_parent_encoder_texture_and_sampler_binds() {
    use crate::contract::pixel_format::MTL_FORMAT_RGBA8_UNORM;
    use crate::runtime::compute_exec::{ComputeAccum, ComputeSamplerBind, ComputeTextureBind};
    use crate::runtime::compute_session::ComputeSession;

    let (_guard, mtlb, mut host, mut state) = mul3add1_fixture();

    let icb_desc = make_icb_desc_bytes(1, 1, false);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 5, 0x100, 2, &mtlb);

    let pdesc = make_compute_pipeline_desc(5);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x140, &pdesc);

    let data = [1u32, 2, 3, 4];
    let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf_gva = 3u64 << RESOURCE_PAGE_SHIFT;
    gva_mem::write_task_gva_arm64e(&mut host, &state.tasks[1], buf_gva, &data_bytes);
    let mut bdesc = vec![0u8; 16];
    st64(&mut bdesc[0..], 16);
    st32(&mut bdesc[8..], 3);
    put_object(&mut host, &state, 7, OBJECT_TYPE_BUFFER, 0x180, &bdesc);

    const W: u32 = 2;
    const H: u32 = 2;
    let seed = vec![0xCDu8; (W * H * 4) as usize];
    put_type2_texture(
        &mut host,
        &mut state,
        11,
        4,
        W,
        H,
        MTL_FORMAT_RGBA8_UNORM,
        &seed,
    );
    put_type7_sampler(&mut host, &state, 14, true);

    fill_compute(
        &state,
        &host,
        &IcbComputeFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![kernel_bind(0, 7)],
            threadgroup_memory: vec![],
            barrier: false,
            dispatch: unit_grid_dispatch(4, 1, 1),
        },
    )
    .expect("fill mul3add1 ICB");

    // Stream binds texture + sampler on the parent encoder (always, not inherit flags).
    let mut acc = ComputeAccum::default();
    acc.set_pipeline(6);
    acc.textures.push(ComputeTextureBind {
        index: 0,
        texture_ref: 11,
    });
    acc.samplers.push(ComputeSamplerBind {
        index: 0,
        sampler_ref: 14,
        lod_min_bits: 0,
        lod_max_bits: 0,
        has_lod_clamp: false,
    });

    let mut session = ComputeSession::open(0).expect("session");
    let cmd = execute_icb_command(9, 0, 1);
    assert_eq!(
        session.encode_icb(&mut state, &mut host, 1, &cmd, &acc),
        crate::runtime::compute_exec::ComputeStatus::Ok,
        "ICB execute with parent-encoder texture+sampler"
    );
    assert_eq!(
        session.finish(&mut host, &mut state, 1),
        crate::runtime::compute_exec::ComputeStatus::Ok
    );

    let out = read_u32x4(&host, &state, buf_gva);
    assert_eq!(
        out,
        vec![4, 7, 10, 13],
        "mul3add1 still correct with encoder tex/samp"
    );

    // Storage writeback path: kernel unused the texture — seed preserved via flush.
    let tex_gva = 4u64 << RESOURCE_PAGE_SHIFT;
    let mut tex_back = vec![0u8; seed.len()];
    assert!(gva_mem::read_task_gva(
        &host,
        &state.tasks[1],
        tex_gva,
        &mut tex_back,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    assert_eq!(
        tex_back, seed,
        "encoder storage texture writeback preserves seed"
    );
}

/// Real texture-using ICB kernel via **argument buffers**: AB holds
/// `texture2d<uint, write>`; stream texture is packaged into the AB, bound
/// as kernel buffer 0 on the ICB command, and `useResource`d on the parent
/// encoder. Oracle: xyplane-v1 `[x,y,5,255]`.
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn icb_argument_buffer_storage_texture_xyplane() {
    use crate::contract::pixel_format::MTL_FORMAT_RGBA8_UINT;
    use crate::runtime::compute_exec::{ComputeAccum, ComputeTextureBind};
    use crate::runtime::compute_session::ComputeSession;

    let _guard = icb_test_guard();
    let mtlb = std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/icb_ab_storage_xyplane.metallib"),
    )
    .expect("icb_ab_storage_xyplane.metallib");

    let (mut host, mut state) = icb_device();

    // maxKernel=1 for the AB buffer slot; no inheritBuffers — AB recorded on ICB.
    let icb_desc = make_icb_desc_bytes(1, 1, false);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 5, 0x100, 2, &mtlb);

    let pdesc = make_compute_pipeline_desc(5);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x140, &pdesc);

    const W: u32 = 4;
    const H: u32 = 4;
    let seed = vec![0x00u8; (W * H * 4) as usize];
    put_type2_texture(
        &mut host,
        &mut state,
        11,
        4,
        W,
        H,
        MTL_FORMAT_RGBA8_UINT,
        &seed,
    );

    // Fill: pipeline + dispatch only (AB buffer patched at execute).
    fill_compute(
        &state,
        &host,
        &IcbComputeFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![],
            threadgroup_memory: vec![],
            barrier: false,
            dispatch: unit_grid_dispatch(W, H, 1),
        },
    )
    .expect("fill AB storage ICB");

    let mut acc = ComputeAccum::default();
    acc.set_pipeline(6);
    acc.textures.push(ComputeTextureBind {
        index: 0,
        texture_ref: 11,
    });

    let mut session = ComputeSession::open(0).expect("session");
    let cmd = execute_icb_command(9, 0, 1);
    assert_eq!(
        session.encode_icb(&mut state, &mut host, 1, &cmd, &acc),
        crate::runtime::compute_exec::ComputeStatus::Ok,
        "AB storage texture ICB execute"
    );
    assert_eq!(
        session.finish(&mut host, &mut state, 1),
        crate::runtime::compute_exec::ComputeStatus::Ok
    );

    let data_gva = 4u64 << RESOURCE_PAGE_SHIFT;
    let mut back = vec![0u8; (W * H * 4) as usize];
    assert!(gva_mem::read_task_gva(
        &host,
        &state.tasks[1],
        data_gva,
        &mut back,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    for y in 0..H {
        for x in 0..W {
            let i = ((y * W + x) * 4) as usize;
            assert_eq!(
                &back[i..i + 4],
                &[x as u8, y as u8, 5, 255],
                "AB storage texel ({x},{y})"
            );
        }
    }
}

/// Sampled + storage textures + sampler in one argument buffer under ICB.
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn icb_argument_buffer_sample_and_write() {
    use crate::contract::pixel_format::{MTL_FORMAT_RGBA8_UINT, MTL_FORMAT_RGBA8_UNORM};
    use crate::runtime::compute_exec::{ComputeAccum, ComputeSamplerBind, ComputeTextureBind};
    use crate::runtime::compute_session::ComputeSession;

    let _guard = icb_test_guard();
    let mtlb = std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/icb_ab_sample_xyplane.metallib"),
    )
    .expect("icb_ab_sample_xyplane.metallib");

    let (mut host, mut state) = icb_device();

    let icb_desc = make_icb_desc_bytes(1, 1, false);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 5, 0x100, 2, &mtlb);

    let pdesc = make_compute_pipeline_desc(5);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x140, &pdesc);

    const W: u32 = 4;
    const H: u32 = 4;
    let mut input = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let i = ((y * W + x) * 4) as usize;
            input[i] = x as u8;
            input[i + 1] = y as u8;
            input[i + 2] = 5;
            input[i + 3] = 255;
        }
    }
    put_type2_texture(
        &mut host,
        &mut state,
        11,
        4,
        W,
        H,
        MTL_FORMAT_RGBA8_UNORM,
        &input,
    );
    let out_seed = vec![0x11u8; (W * H * 4) as usize];
    put_type2_texture(
        &mut host,
        &mut state,
        12,
        5,
        W,
        H,
        MTL_FORMAT_RGBA8_UINT,
        &out_seed,
    );
    put_type7_sampler(&mut host, &state, 14, true);

    fill_compute(
        &state,
        &host,
        &IcbComputeFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![],
            threadgroup_memory: vec![],
            barrier: false,
            dispatch: unit_grid_dispatch(W, H, 1),
        },
    )
    .expect("fill AB sample ICB");

    let mut acc = ComputeAccum::default();
    acc.set_pipeline(6);
    // Order matches AB members: id(0)=in, id(1)=out, id(2)=sampler.
    acc.textures.push(ComputeTextureBind {
        index: 0,
        texture_ref: 11,
    });
    acc.textures.push(ComputeTextureBind {
        index: 1,
        texture_ref: 12,
    });
    acc.samplers.push(ComputeSamplerBind {
        index: 0,
        sampler_ref: 14,
        lod_min_bits: 0,
        lod_max_bits: 0,
        has_lod_clamp: false,
    });

    let mut session = ComputeSession::open(0).expect("session");
    let cmd = execute_icb_command(9, 0, 1);
    assert_eq!(
        session.encode_icb(&mut state, &mut host, 1, &cmd, &acc),
        crate::runtime::compute_exec::ComputeStatus::Ok,
        "AB sample+write ICB execute"
    );
    assert_eq!(
        session.finish(&mut host, &mut state, 1),
        crate::runtime::compute_exec::ComputeStatus::Ok
    );

    let out_gva = 5u64 << RESOURCE_PAGE_SHIFT;
    let mut back = vec![0u8; (W * H * 4) as usize];
    assert!(gva_mem::read_task_gva(
        &host,
        &state.tasks[1],
        out_gva,
        &mut back,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    for y in 0..H {
        for x in 0..W {
            let i = ((y * W + x) * 4) as usize;
            assert_eq!(
                &back[i..i + 4],
                &[x as u8, y as u8, 5, 255],
                "AB sample out texel ({x},{y})"
            );
        }
    }
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
#[test]
fn resolve_bind_offset_from_wire_va() {
    let (mut host, state) = icb_device();
    // 32 B buffer at handle 4: 16 B pad + 16 B color payload.
    let mut bytes = vec![0u8; 32];
    let sid = 7u8;
    let r = (0x60u32 + sid as u32) as f32 / 255.0;
    let color = [r, 0x44 as f32 / 255.0, 0x22 as f32 / 255.0, 1.0f32];
    let color_bytes: Vec<u8> = color.iter().flat_map(|f| f.to_le_bytes()).collect();
    bytes[16..32].copy_from_slice(&color_bytes);
    put_type1_buffer(&mut host, &state, 13, 4, &bytes);
    let base = (4u64) << RESOURCE_PAGE_SHIFT;
    assert_eq!(
        offset_from_wire_va(&state, &host, 1, 13, base + 16).unwrap(),
        16
    );
    assert_eq!(offset_from_wire_va(&state, &host, 1, 13, 0).unwrap(), 0);
    assert!(offset_from_wire_va(&state, &host, 1, 13, base - 1).is_err());
    assert!(offset_from_wire_va(&state, &host, 1, 13, base + 32).is_err());
}

/// Host fill with non-zero fragment bind offset into a padded type-1 buffer.
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn fill_render_nonzero_bind_offset_oracle() {
    let _guard = icb_test_guard();

    let (vert_mtlb, frag_mtlb) = load_oracle_mtlb();
    let (mut host, mut state) = icb_device();

    let icb_desc = make_render_icb_desc_bytes(1, 0, 1, MTL_INDIRECT_CMD_DRAW_INDEXED);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &vert_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_render_pipeline_desc(2, 3);
    let pdesc_gva = 0x240u64;
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, pdesc_gva, &pdesc);

    let indices: [u16; 3] = [0, 1, 2];
    // Index buffer: 4 B pad + 6 B indices (offset 4).
    let mut index_bytes = vec![0u8; 4];
    index_bytes.extend(indices.iter().flat_map(|v| v.to_le_bytes()));
    put_type1_buffer(&mut host, &state, 12, 4, &index_bytes);
    let index_base = (4u64) << RESOURCE_PAGE_SHIFT;

    let sid = 7u8;
    let r = (0x60u32 + sid as u32) as f32 / 255.0;
    let color = [r, 0x44 as f32 / 255.0, 0x22 as f32 / 255.0, 1.0f32];
    let color_bytes: Vec<u8> = color.iter().flat_map(|f| f.to_le_bytes()).collect();
    // Color buffer: 16 B pad + float4 (offset 16).
    let mut color_buf = vec![0xAAu8; 16];
    color_buf.extend_from_slice(&color_bytes);
    put_type1_buffer(&mut host, &state, 13, 5, &color_buf);
    let color_base = (5u64) << RESOURCE_PAGE_SHIFT;

    fill_render(
        &state,
        &host,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![IcbRenderBufferBind {
                index: 0,
                buffer_ref: 13,
                offset: 16,
                wire_va: 0,
                attribute_stride: 0,
                has_attribute_stride: false, // host fill uses offset directly
                is_fragment: true,
                stage: IcbRenderBindStage::default(),
            }],
            object_threadgroup_memory: vec![],
            draw: IcbRenderDraw::Indexed {
                primitive_type: 3,
                index_type: 0,
                index_buffer_ref: 12,
                index_count: 3,
                index_buffer_offset: 4,
                index_wire_va: 0,
                instance_count: 1,
                base_vertex: 0,
                base_instance: 0,
            },
        },
    )
    .expect("fill with non-zero offsets");

    let mapping_id = map_draw_target(&mut host, &mut state, 0x34);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok
    );

    assert_stagein_solid(
        &mut state,
        &mut host,
        mapping_id,
        sid,
        "nonzero bind offset",
    );
    let _ = (index_base, color_base);
}

/// Buffer-backed fill: wire VA = type-1 base+offset → resolve → execute.
#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn buffer_backed_nonzero_wire_va_offset() {
    let _guard = icb_test_guard();

    let (vert_mtlb, frag_mtlb) = load_oracle_mtlb();
    let (mut host, mut state) = icb_device();

    let max_v = 0u16;
    let max_f = 1u16;
    let layout = render_icb_layout(max_v, max_f, MTL_INDIRECT_CMD_DRAW_INDEXED);
    let icb_desc = make_render_icb_desc_bytes(1, max_v, max_f, MTL_INDIRECT_CMD_DRAW_INDEXED);
    let icb_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    put_object(&mut host, &state, 9, OBJECT_TYPE_TYPE7, icb_gva, &icb_desc);

    put_function_object(&mut host, &state, 2, 0x200, 2, &vert_mtlb);

    put_function_object(&mut host, &state, 3, 0x220, 3, &frag_mtlb);

    let pdesc = make_render_pipeline_desc(2, 3);
    put_object(&mut host, &state, 6, OBJECT_TYPE_TYPE7, 0x240, &pdesc);

    let indices: [u16; 3] = [0, 1, 2];
    let mut index_bytes = vec![0u8; 8]; // pad 8
    index_bytes.extend(indices.iter().flat_map(|v| v.to_le_bytes()));
    put_type1_buffer(&mut host, &state, 12, 4, &index_bytes);
    let index_wire = ((4u64) << RESOURCE_PAGE_SHIFT) + 8;

    let sid = 7u8;
    let r = (0x60u32 + sid as u32) as f32 / 255.0;
    let color = [r, 0x44 as f32 / 255.0, 0x22 as f32 / 255.0, 1.0f32];
    let color_bytes: Vec<u8> = color.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut color_buf = vec![0u8; 24]; // pad 24
    color_buf.extend_from_slice(&color_bytes);
    put_type1_buffer(&mut host, &state, 13, 5, &color_buf);
    let color_wire = ((5u64) << RESOURCE_PAGE_SHIFT) + 24;

    let slot = encode_render_command_slot(
        &layout,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 6,
            buffers: vec![IcbRenderBufferBind {
                index: 0,
                buffer_ref: 13,
                offset: 0,
                wire_va: color_wire,
                attribute_stride: 0,
                has_attribute_stride: false,
                is_fragment: true,
                stage: IcbRenderBindStage::default(),
            }],
            object_threadgroup_memory: vec![],
            draw: IcbRenderDraw::Indexed {
                primitive_type: 3,
                index_type: 0,
                index_buffer_ref: 12,
                index_count: 3,
                index_buffer_offset: 0,
                index_wire_va: index_wire,
                instance_count: 1,
                base_vertex: 0,
                base_instance: 0,
            },
        },
    )
    .unwrap();

    // Decode + resolve without host fill API.
    let mut decoded = decode_render_command_slot(&layout, &slot, max_v, max_f)
        .unwrap()
        .expect("slot");
    assert_eq!(decoded.buffers[0].wire_va, color_wire);
    match decoded.draw {
        IcbRenderDraw::Indexed { index_wire_va, .. } => {
            assert_eq!(index_wire_va, index_wire);
        }
        _ => panic!("indexed"),
    }
    resolve_render_fill_offsets(&state, &host, 1, &mut decoded).unwrap();
    assert_eq!(decoded.buffers[0].offset, 24);
    match decoded.draw {
        IcbRenderDraw::Indexed {
            index_buffer_offset,
            ..
        } => assert_eq!(index_buffer_offset, 8),
        _ => panic!("indexed"),
    }

    associate_icb_command_memory(&mut host, &state, 6, &slot);

    let mapping_id = map_draw_target(&mut host, &mut state, 0x35);

    let req = draw_request(mapping_id);
    assert_eq!(
        encode_icb_execute_and_writeback(&mut state, &mut host, &req, 9, 0, 1),
        EncodeStatus::Ok
    );

    assert_stagein_solid(&mut state, &mut host, mapping_id, sid, "wire_va offset");
}

/// Each render bind stage is bounded by its own field of the create descriptor.
///
/// The four maxima are decoded separately and handed to Metal separately, so
/// mapping them onto one bound would report the wrong table for three stages out
/// of four — the failure mode `8ad945e` found on the type-1 buffer span.
#[test]
fn a_render_icb_bind_is_bounded_by_the_count_its_own_stage_declared() {
    use crate::observe::Decline;

    let desc = IndirectCommandBufferDescriptor {
        max_vertex_buffer_bind_count: 4,
        max_fragment_buffer_bind_count: 2,
        max_object_buffer_bind_count: 3,
        max_mesh_buffer_bind_count: 1,
        ..Default::default()
    };

    // The last in-range index of each stage is accepted, and the first
    // out-of-range one is refused under that stage's own slug.
    for (stage, declared, slug) in [
        (
            IcbRenderBindStage::Vertex,
            4u32,
            "icb_frc_vertex_bind_index_past_max",
        ),
        (
            IcbRenderBindStage::Fragment,
            2,
            "icb_frc_fragment_bind_index_past_max",
        ),
        (
            IcbRenderBindStage::Object,
            3,
            "icb_frc_object_bind_index_past_max",
        ),
        (
            IcbRenderBindStage::Mesh,
            1,
            "icb_frc_mesh_bind_index_past_max",
        ),
    ] {
        assert!(
            refuse_render_bind_past_declared_max(stage, declared - 1, &desc).is_ok(),
            "{stage:?} refused its own last declared index"
        );
        let refusal = refuse_render_bind_past_declared_max(stage, declared, &desc)
            .expect_err("index past the declared count must refuse");
        assert_eq!(refusal.slug(), slug);
    }
}

/// A stage the guest declared no binds for admits no index at all, including 0.
///
/// `max_* == 0` is the common case for stages a pipeline does not use, and `0`
/// is the index a zeroed fill record carries, so this is the arm a malformed
/// record reaches first.
#[test]
fn a_render_icb_stage_declaring_no_binds_refuses_index_zero() {
    let desc = IndirectCommandBufferDescriptor::default();
    for stage in [
        IcbRenderBindStage::Vertex,
        IcbRenderBindStage::Fragment,
        IcbRenderBindStage::Object,
        IcbRenderBindStage::Mesh,
    ] {
        assert_eq!(stage.declared_bind_count(&desc), 0, "{stage:?}");
        assert!(refuse_render_bind_past_declared_max(stage, 0, &desc).is_err());
    }
}

/// An execute that filled no slots says so, and one that met a different
/// refusal still forwards it.
///
/// The rule this pins used to be a wildcard — `Err(IcbStatus::Missing(_)) => {}`
/// — copied into the render arm and the compute arm. It swallowed both slugs
/// `decode_icb_command_range` raises under `Missing`, and only one of them had
/// been argued for. The four cases below are the whole vocabulary that reaches
/// this function: filled, unfilled, the other `Missing`, and everything else.
#[test]
fn an_icb_execute_that_filled_no_slots_is_counted_and_not_swallowed() {
    use crate::runtime::drain::store_route_count;
    const ROUTE: &str = "icb_executed_without_command_memory";

    let quiet = store_route_count(ROUTE);
    assert_eq!(
        icb_fill_outcome(Ok(()), 1, 9),
        Ok(()),
        "a filled ICB is carried on from"
    );
    assert_eq!(
        store_route_count(ROUTE),
        quiet,
        "a filled ICB costs the guest nothing and must not count"
    );

    // The unfilled case: control flow is unchanged so the caller still does its
    // writeback, but the lost commands are now counted.
    assert_eq!(
        icb_fill_outcome(Err(IcbStatus::Missing(ICB_FILL_NO_COMMAND_MEMORY)), 1, 9),
        Ok(()),
        "an empty execute is a no-op, not a reason to skip the writeback"
    );
    assert_eq!(
        store_route_count(ROUTE),
        quiet + 1,
        "the commands the guest encoded into this ICB were lost, and it says so"
    );

    // The other `Missing` slug, and a variant from another class. Both forward,
    // so the caller declines by the name of the check that actually refused.
    for other in [
        IcbStatus::Missing("icb_fill_not_cached"),
        IcbStatus::Args("icb_fill_zero_command_size"),
    ] {
        assert_eq!(
            icb_fill_outcome(Err(other), 1, 9),
            Err(other),
            "{other:?} names a different loss and must not be swallowed"
        );
    }
    assert_eq!(
        store_route_count(ROUTE),
        quiet + 1,
        "a forwarded refusal is not an empty execute"
    );
}
