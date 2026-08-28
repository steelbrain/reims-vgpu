#![allow(
    clippy::field_reassign_with_default,
    clippy::too_many_arguments,
    reason = "wire fixtures are assembled field by field to keep each protocol case explicit"
)]

use super::*;
use crate::contract::endian::{st16, st32, st64};
use crate::contract::pixel_format::{MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_RGBA8_UNORM};
use crate::model::{DeviceId, FENCE_DOMAIN_BLIT, PAGE_SHIFT_ARM64E};
use crate::runtime::decode::blit::{self, Point, Size};
use crate::runtime::decode::resource::{
    list_object_entry_offset, LINEAR_DESC_HANDLE, LINEAR_DESC_MIN_LEN, LINEAR_DESC_SIZE,
    OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_BUFFER, RESOURCE_PAGE_SHIFT,
};
use crate::runtime::gva_mem::write_task_gva_arm64e;
use crate::runtime::host::FakeHost;
use crate::runtime::objects;

/// The channel is the whole diagnostic for this rail: 177 checks collapse
/// into eight statuses, so a refusal that reaches the dispatch line without a
/// reason says almost nothing. An uninstrumented site used to render a bare
/// `reason=` with nothing after it — not greppable, and indistinguishable from
/// a missing field rather than from a missing *reason*.
#[test]
fn an_unattributed_refusal_names_the_gap_rather_than_rendering_an_empty_reason() {
    use crate::observe::{Emit, Refusal};

    clear_blit_fail_reason();
    assert_eq!(
        BlitStatus::Bounds.refusal(),
        Some("blit_unattributed"),
        "a refusal with an empty channel must still name something"
    );
    let line = Emit::refusal("blit", &BlitStatus::Bounds)
        .expect("a refusal produces a line")
        .render();
    assert_eq!(line, "blit reason=blit_unattributed");
    assert!(
        !line.contains("reason= "),
        "the line rendered an empty reason: {line}"
    );

    // An instrumented site names its own check instead.
    let st = br(BlitStatus::Bounds, "fill_out_of_range");
    assert_eq!(st.refusal(), Some("fill_out_of_range"));
}

/// `Ok`, a zero-extent no-op and a soft fence wait are control flow. The
/// dispatch sites count them as success or as pending, and the guest re-polls
/// the wait every drain — logging any of them floods the always-on sink.
/// `Emit::refusal` returns `None`, so no caller can log one by accident.
#[test]
fn zero_extent_and_pending_fences_are_control_flow_not_refusals() {
    use crate::observe::{Emit, Refusal};

    // A stale channel value must not resurrect a success as a refusal.
    let _ = br(BlitStatus::Bounds, "fill_out_of_range");
    for ok in [
        BlitStatus::Ok,
        BlitStatus::ZeroExtent,
        BlitStatus::FencePending,
    ] {
        assert_eq!(ok.refusal(), None, "{ok:?} is not a refusal");
        assert!(
            Emit::refusal("blit", &ok).is_none(),
            "{ok:?} produced a loggable line"
        );
    }
}

/// A task-1 device with the arm64e page tables every blit fixture walks.
fn blit_device() -> (FakeHost, DeviceState) {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    (host, state)
}

/// A copy command naming its two object refs. Every rectangular blit case
/// starts here and then sets the one field it is about.
fn copy_cmd(copy_kind: CopyKind, source: u32, destination: u32) -> Command {
    let mut cmd = Command::default();
    cmd.kind = Kind::Copy;
    cmd.copy_kind = copy_kind;
    cmd.source = source;
    cmd.destination = destination;
    cmd
}

/// Back `mapping_id` with one guest data page at `pfn` and mark it mapped.
/// This is the surface state every type-11 / type-5 install needs before it
/// can attach geometry or a descriptor.
fn map_one_page_surface(host: &mut FakeHost, state: &mut DeviceState, mapping_id: u32, pfn: u32) {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
    state.map_surface(mapping_id);
    let m = state.mappings.get_mut(&mapping_id).unwrap();
    m.mapped = true;
    m.mapping_internal = 1;
    m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
}

/// Publish object-list entry `obj_ref`: type and descriptor length packed
/// into word 0, descriptor GVA in the following eight bytes.
fn write_list_entry(
    host: &mut FakeHost,
    state: &DeviceState,
    obj_ref: u32,
    object_type: u32,
    desc_len: u32,
    desc_gva: u64,
) {
    let off = list_object_entry_offset(obj_ref, 16).unwrap();
    let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(&mut entry[0..], object_type | (desc_len << 8));
    entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(host, &state.tasks[1], off, &entry);
}

/// Install type-1 buffer: object-list at GVA 0, descriptor at GVA 0x100 + ref*0x20.
fn install_buffer(
    host: &mut FakeHost,
    state: &mut DeviceState,
    obj_ref: u32,
    handle: u32,
    size: u64,
) {
    assert!(state.set_object_list(1, 0, 16));
    let mut desc = vec![0u8; LINEAR_DESC_MIN_LEN];
    st64(&mut desc[LINEAR_DESC_SIZE..], size);
    st64(&mut desc[LINEAR_DESC_HANDLE..], handle as u64);
    let desc_gva = 0x100u64 + (obj_ref as u64) * 0x20;
    write_task_gva_arm64e(host, &state.tasks[1], desc_gva, &desc);
    write_list_entry(
        host,
        state,
        obj_ref,
        OBJECT_TYPE_BUFFER as u32,
        LINEAR_DESC_MIN_LEN as u32,
        desc_gva,
    );
    let e = objects::lookup_list_entry(state, host, 1, obj_ref).expect("entry");
    assert_eq!(e.object_type, OBJECT_TYPE_BUFFER);
}

/// A blit's destination bound is the pages the command named, and it holds
/// against a guest that re-points the range while the copy is running.
///
/// A blit does not wait for the GPU, which is why these writes were argued
/// to need no bound. But a copy is a loop: it re-derives its destination
/// from the same base address on every chunk, and the guest's vCPUs run
/// throughout. `FakeHost::arm_rewire` puts the guest edit where it actually
/// happens -- between two iterations, fired from the source read the loop
/// performs anyway -- so this is the mechanism, not a model of it.
///
/// Both directions are asserted. Bounded, the copy refuses and the page the
/// guest handed to something else is still zero. Unbounded, the identical
/// loop reports success and paints that page, which is the guest heap and
/// kernel corruption this class is made of.
#[test]
fn a_blit_destination_is_bounded_against_a_guest_that_repoints_it_mid_copy() {
    use crate::runtime::host::Rewire;
    const PAGE: u64 = 1 << PAGE_SHIFT_ARM64E;
    // Six pages is the smallest copy that spans more than one 64 KiB chunk,
    // and a second chunk is what gives the guest an instant to edit in.
    const PAGES: u64 = 6;
    const LEN: u64 = PAGES * PAGE;
    // Virtual page N maps to guest frame DATA_BASE + N.
    const DATA_BASE: u32 = 4;
    const SRC_PAGE: u64 = 1;
    const DST_PAGE: u64 = 8;
    // Outside both buffers: the allocation the guest hands the range to.
    const VICTIM_PAGE: u64 = 15;
    let victim_gpa = (DATA_BASE as u64 + VICTIM_PAGE) << PAGE_SHIFT_ARM64E;
    let root_gpa = 3u64 << PAGE_SHIFT_ARM64E;
    // The second chunk starts at source page 4; reading it is the loop
    // saying "chunk one is written, chunk two is not".
    let second_chunk_src_gpa = (DATA_BASE as u64 + SRC_PAGE + 4) << PAGE_SHIFT_ARM64E;
    // Destination virtual page 12 is written by that second chunk.
    let moved_entry_gpa = root_gpa + 12 * 4;

    // `rig` is a whole scenario because the unbounded arm must run against a
    // guest in the same state, not against one the bounded arm already
    // refused half a copy into.
    let rig = |host: &mut FakeHost, state: &mut DeviceState| {
        gva_mem::define_task_pages_arm64e(host, state, DATA_BASE, 16);
        install_buffer(host, state, 7, SRC_PAGE as u32, LEN);
        install_buffer(host, state, 8, DST_PAGE as u32, LEN);
        // Distinctive source bytes: a page of zeroes proves nothing about
        // where a copy landed.
        let payload = vec![0xabu8; LEN as usize];
        write_task_gva_arm64e(
            host,
            &state.tasks[1],
            SRC_PAGE << PAGE_SHIFT_ARM64E,
            &payload,
        );
        host.arm_rewire(Rewire {
            on_read_gpa: second_chunk_src_gpa,
            on_read_len: 1,
            pte_gpa: moved_entry_gpa,
            bytes: (DATA_BASE + VICTIM_PAGE as u32).to_le_bytes().to_vec(),
        });
    };
    let victim_head = |host: &FakeHost| {
        let mut out = [0u8; 8];
        let _ = host.read_gpa(victim_gpa, &mut out);
        out
    };

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    rig(&mut host, &mut state);
    let mut cmd = copy_cmd(CopyKind::BufferToBuffer, 7, 8);
    cmd.size = LEN;
    let st = execute_blit(&mut state, &mut host, 1, &cmd);
    assert_eq!(
        host.rewires_fired(),
        1,
        "the guest edit never fired -- the copy did not read its second chunk, \
         so this test proved nothing"
    );
    assert_eq!(
        st,
        BlitStatus::GuestIo,
        "a destination page the command never named must be refused"
    );
    assert_eq!(
        victim_head(&host),
        [0u8; 8],
        "the refused copy still reached the allocation the guest moved the range to"
    );

    // The same loop with no window, on a guest in the same state.
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    rig(&mut host, &mut state);
    assert!(
        copy_bytes_within(
            &mut host,
            &mut state,
            1,
            SRC_PAGE << PAGE_SHIFT_ARM64E,
            DST_PAGE << PAGE_SHIFT_ARM64E,
            LEN,
            None,
        )
        .is_ok(),
        "without the bound the copy reports success"
    );
    assert_eq!(host.rewires_fired(), 1, "same guest, same instant");
    assert_eq!(
        victim_head(&host),
        [0xabu8; 8],
        "and it succeeds by painting a page it was never given"
    );
}

#[test]
fn range_overlap_helper() {
    assert!(ranges_overlap(0, 10, 5, 10));
    assert!(!ranges_overlap(0, 10, 10, 5));
    assert!(!ranges_overlap(0, 0, 0, 5));
}

#[test]
fn range_fits_helper() {
    assert!(range_fits(0, 10, 10));
    assert!(range_fits(5, 5, 10));
    assert!(!range_fits(5, 6, 10));
    assert!(!range_fits(11, 0, 10));
}

#[test]
fn decode_fill_and_plan() {
    let mut v = vec![0u8; 0x20];
    st32(&mut v[0..], wire_blit::OPCODE_FILL_BUFFER);
    st32(&mut v[4..], 0x20);
    st32(&mut v[8..], 3);
    st64(&mut v[0x0c..], 0x10);
    st64(&mut v[0x14..], 8);
    v[0x1c] = 0xa5;
    let cmd = blit::decode(&v).unwrap();
    assert_eq!(cmd.kind, Kind::FillBuffer);
    assert_eq!(cmd.buffer, 3);
    assert_eq!(cmd.range_location, 0x10);
    assert_eq!(cmd.range_length, 8);
    assert_eq!(cmd.fill_value, 0xa5);
}

#[test]
fn decode_b2b() {
    let mut v = vec![0u8; 0x28];
    st32(&mut v[0..], wire_blit::OPCODE_COPY_BUFFER_TO_BUFFER);
    st32(&mut v[4..], 0x28);
    st32(&mut v[8..], 1);
    st32(&mut v[12..], 2);
    st64(&mut v[0x10..], 4);
    st64(&mut v[0x18..], 8);
    st64(&mut v[0x20..], 16);
    let cmd = blit::decode(&v).unwrap();
    assert_eq!(cmd.copy_kind, CopyKind::BufferToBuffer);
    assert_eq!(cmd.size, 16);
    assert_eq!(cmd.source_offset, 4);
    assert_eq!(cmd.destination_offset, 8);
}

#[test]
fn fill_buffer_roundtrip() {
    let (mut host, mut state) = blit_device();
    install_buffer(&mut host, &mut state, 7, 1, 256);
    let mut cmd = Command::default();
    cmd.kind = Kind::FillBuffer;
    cmd.buffer = 7;
    cmd.range_location = 16;
    cmd.range_length = 8;
    cmd.fill_value = 0x5a;
    assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);
    let mut out = [0u8; 8];
    let gva = (1u64 << RESOURCE_PAGE_SHIFT) + 16;
    assert!(
        gva_mem::read_task_gva(&host, &state.tasks[1], gva, &mut out, PAGE_SHIFT_ARM64E).is_ok()
    );
    assert_eq!(out, [0x5a; 8]);
}

/// The pattern fill lands a repeating 32-bit unit, in the right order and
/// on the right phase.
///
/// Deliberately not four equal bytes: a pattern of `0x89abcdef` fails if
/// the executor wrote the `u32` big-endian, wrote only its low byte, or
/// started the repeat anywhere but the range's first byte, and each of
/// those produces bytes that look filled. The range starts at 16 rather
/// than 0 so a fill that ignored `range_location` also fails, and the
/// length is not a multiple of the pattern so the partial tail is checked.
#[test]
fn fill_buffer_pattern4_roundtrip() {
    let (mut host, mut state) = blit_device();
    install_buffer(&mut host, &mut state, 7, 1, 256);
    let mut cmd = Command::default();
    cmd.kind = Kind::FillBufferPattern4;
    cmd.buffer = 7;
    cmd.range_location = 16;
    cmd.range_length = 10;
    cmd.fill_pattern = 0x89ab_cdef;
    assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);
    let mut out = [0u8; 12];
    let gva = (1u64 << RESOURCE_PAGE_SHIFT) + 16;
    assert!(
        gva_mem::read_task_gva(&host, &state.tasks[1], gva, &mut out, PAGE_SHIFT_ARM64E).is_ok()
    );
    assert_eq!(
        &out[..10],
        &[0xef, 0xcd, 0xab, 0x89, 0xef, 0xcd, 0xab, 0x89, 0xef, 0xcd],
        "the pattern must repeat little-endian from the range's first byte"
    );
    assert_eq!(&out[10..], &[0, 0], "the fill wrote past its length");
}

/// A range whose start is not pattern-aligned is refused, by name.
///
/// The record says what repeats and not what phase it repeats on, and the
/// two readings — anchored to the range, anchored to the buffer — disagree
/// for exactly these ranges. Filling one under either reading writes
/// plausible bytes into guest memory that may be wrong, which is worse than
/// a refusal the fail log explains. A non-zero reading of
/// `fill_pattern4_unaligned_range` on a driven boot is the argument for
/// deriving the phase rule; until then it is a healthy zero.
#[test]
fn an_unaligned_pattern_fill_is_refused_rather_than_guessed() {
    let (mut host, mut state) = blit_device();
    install_buffer(&mut host, &mut state, 7, 1, 256);
    let mut cmd = Command::default();
    cmd.kind = Kind::FillBufferPattern4;
    cmd.buffer = 7;
    cmd.range_location = 17;
    cmd.range_length = 8;
    cmd.fill_pattern = 0x89ab_cdef;
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::Unsupported
    );
    assert_eq!(
        blit_fail_reason(),
        "fill_pattern4_unaligned_range",
        "the refusal must name the phase question rather than reporting a \
         generic unsupported"
    );
    // And nothing was written: a refusal that had already painted part of
    // the range would be worse than the fill it declined to do.
    let mut out = [0u8; 8];
    let gva = (1u64 << RESOURCE_PAGE_SHIFT) + 17;
    assert!(
        gva_mem::read_task_gva(&host, &state.tasks[1], gva, &mut out, PAGE_SHIFT_ARM64E).is_ok()
    );
    assert_eq!(out, [0u8; 8]);
}

/// The byte fill and the pattern fill are one write path.
///
/// `write_fill_range` is `write_fill_pattern` with a one-byte pattern, and
/// this asserts the two agree on the bytes rather than only on the code
/// being shared — the divergence this rail keeps producing is two arms of
/// one guest-memory write drifting, and a shared body is only worth
/// anything if the equivalence is checked.
#[test]
fn a_byte_fill_and_a_four_equal_byte_pattern_fill_write_the_same_bytes() {
    let read_back = |cmd: &Command| {
        let (mut host, mut state) = blit_device();
        install_buffer(&mut host, &mut state, 7, 1, 256);
        assert_eq!(execute_blit(&mut state, &mut host, 1, cmd), BlitStatus::Ok);
        let mut out = [0u8; 12];
        let gva = (1u64 << RESOURCE_PAGE_SHIFT) + 16;
        assert!(
            gva_mem::read_task_gva(&host, &state.tasks[1], gva, &mut out, PAGE_SHIFT_ARM64E)
                .is_ok()
        );
        out
    };
    let mut byte_fill = Command::default();
    byte_fill.kind = Kind::FillBuffer;
    byte_fill.buffer = 7;
    byte_fill.range_location = 16;
    byte_fill.range_length = 10;
    byte_fill.fill_value = 0x5a;
    let mut pattern_fill = byte_fill.clone();
    pattern_fill.kind = Kind::FillBufferPattern4;
    pattern_fill.fill_value = 0;
    pattern_fill.fill_pattern = u32::from_le_bytes([0x5a; 4]);
    assert_eq!(read_back(&byte_fill), read_back(&pattern_fill));
}

/// The reason channel names *which* collapsed check fired for a coarse
/// `BlitStatus`, distinguishes distinct causes, is reset per command so a stale
/// slug never leaks across blits, and stays empty after a successful blit.
#[test]
fn blit_fail_reason_names_distinct_causes_and_resets_per_command() {
    let (mut host, mut state) = blit_device();
    install_buffer(&mut host, &mut state, 7, 1, 256);

    // ref==0 → MissingResource, reason "buf_ref_zero".
    let mut cmd = Command::default();
    cmd.kind = Kind::FillBuffer;
    cmd.buffer = 0;
    cmd.range_length = 8;
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::MissingResource
    );
    assert_eq!(blit_fail_reason(), "buf_ref_zero");

    // Unbound ref → same coarse status, DIFFERENT reason "buf_no_list_entry".
    cmd.buffer = 42; // never installed
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::MissingResource
    );
    assert_eq!(blit_fail_reason(), "buf_no_list_entry");

    // In-bounds range on a valid buffer → the channel is reset at entry and the
    // successful blit leaves it empty (no stale "buf_no_list_entry").
    cmd.buffer = 7;
    cmd.range_location = 16;
    cmd.range_length = 8;
    assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);
    assert_eq!(blit_fail_reason(), "");

    // Out-of-range fill on a valid buffer → Bounds, reason "fill_range_oob".
    cmd.range_location = 250;
    cmd.range_length = 64; // 250+64 > 256
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::Bounds
    );
    assert_eq!(blit_fail_reason(), "fill_range_oob");
}

#[test]
fn copy_buffer_to_buffer_roundtrip() {
    let (mut host, mut state) = blit_device();
    install_buffer(&mut host, &mut state, 1, 1, 256);
    install_buffer(&mut host, &mut state, 2, 2, 256);
    let src_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    let pat = [1u8, 2, 3, 4, 5, 6, 7, 8];
    write_task_gva_arm64e(&mut host, &state.tasks[1], src_gva + 4, &pat);
    let mut cmd = copy_cmd(CopyKind::BufferToBuffer, 1, 2);
    cmd.source_offset = 4;
    cmd.destination_offset = 8;
    cmd.size = 8;
    assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);
    let mut out = [0u8; 8];
    let dst_gva = (2u64 << RESOURCE_PAGE_SHIFT) + 8;
    assert!(
        gva_mem::read_task_gva(&host, &state.tasks[1], dst_gva, &mut out, PAGE_SHIFT_ARM64E)
            .is_ok()
    );
    assert_eq!(out, pat);
}

#[test]
fn copy_b2b_overlap_rejected() {
    let (mut host, mut state) = blit_device();
    install_buffer(&mut host, &mut state, 1, 1, 256);
    let mut cmd = copy_cmd(CopyKind::BufferToBuffer, 1, 1);
    cmd.source_offset = 0;
    cmd.destination_offset = 4;
    cmd.size = 16;
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::Overlap
    );
}

#[test]
fn copy_buffer_to_type11_roundtrip() {
    let (mut host, mut state) = blit_device();
    // Buffer with 8 BGRA pixels (one row of 2 pixels for a 2x1 copy).
    install_buffer(&mut host, &mut state, 1, 1, 256);
    let pat = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    let src_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    write_task_gva_arm64e(&mut host, &state.tasks[1], src_gva, &pat);

    // Type-11 object ref 3 → mapping 9, 2x2 BGRA.
    let mapping_id = 9u32;
    install_type11(&mut host, &mut state, 3, mapping_id, 0x20);

    let mut cmd = copy_cmd(CopyKind::BufferToTexture, 1, 3);
    cmd.source_offset = 0;
    cmd.source_bytes_per_row = 8;
    cmd.source_size = Size {
        width: 2,
        height: 1,
        depth: 1,
    };
    cmd.destination_origin = Point { x: 0, y: 1, z: 0 };
    assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);

    // Read back the written row via mapping_write.
    let mut back = [0u8; 8];
    assert!(mapping_write::read_rect_raw(
        &mut state,
        &mut host,
        mapping_id,
        mapping_write::Rect {
            origin_x: 0,
            origin_y: 1,
            width: 2,
            height: 1
        },
        &mut back,
        8
    ));
    assert_eq!(back, pat);
    // Blit again — unified memory: pages are the only content; gen advances.
    let gen_before = state.mappings[&mapping_id].content_generation;
    assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);
    assert!(state.mappings[&mapping_id].content_generation > gen_before);
}

/// type-11→type-11 copy lands source bytes in dest pages (unified content).
#[test]
fn copy_type11_to_type11_writes_dst_pages() {
    let (mut host, mut state) = blit_device();
    install_type11(&mut host, &mut state, 3, 3, 0x20);
    install_type11(&mut host, &mut state, 4, 4, 0x21);
    // Seed source mid=3 pages with a known pattern.
    let src_pat = [9u8, 8, 7, 6, 5, 4, 3, 2, 1, 0, 11, 12, 13, 14, 15, 16];
    assert!(mapping_write::write_rect_raw(
        &mut state,
        &mut host,
        3,
        mapping_write::Rect {
            origin_x: 0,
            origin_y: 0,
            width: 2,
            height: 2
        },
        &src_pat,
        8
    ));
    // Origins default to zero: this is the whole 2×2 surface.
    let mut cmd = copy_cmd(CopyKind::TextureToTexture, 3, 4);
    cmd.source_size = Size {
        width: 2,
        height: 2,
        depth: 1,
    };
    assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);
    let mut back = [0u8; 16];
    assert!(mapping_write::read_rect_raw(
        &mut state,
        &mut host,
        4,
        mapping_write::Rect {
            origin_x: 0,
            origin_y: 0,
            width: 2,
            height: 2
        },
        &mut back,
        8
    ));
    assert_eq!(back, src_pat, "dest pages hold blit content (one copy)");
}

/// A region whose *extent* reaches past a texture is refused, in each of the
/// three copy executors, exactly as its *origin* already was.
///
/// This is the diff the two arms needed. One wire record names an origin and a
/// size; the origin check refused out of range while the extent check clamped
/// and returned `Ok`, so the same malformed region got opposite treatment
/// depending on which half of it was wrong. The origin cases live in
/// `copy_executor_reason_slugs_name_distinct_sites` above; these are their
/// counterparts, deliberately built the same way so the pair reads as one
/// property.
#[test]
fn an_extent_past_the_edge_is_refused_like_an_origin_past_the_edge() {
    let (mut host, mut state) = blit_device();
    install_type11(&mut host, &mut state, 3, 3, 0x20); // 2×2 BGRA, mid 3
    install_type11(&mut host, &mut state, 4, 4, 0x21); // 2×2 BGRA, mid 4
    install_buffer(&mut host, &mut state, 5, 5, 4096);

    // Origin in range, extent past the edge: 3 wide out of a 2-wide texture.
    // Before this, each of these copied 2 and answered Ok.
    let over = Size {
        width: 3,
        height: 1,
        depth: 1,
    };

    let mut cmd = copy_cmd(CopyKind::TextureToTexture, 3, 4);
    cmd.source_size = over;
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::Bounds
    );
    assert_eq!(blit_fail_reason(), "t2t_extent_oob");

    let mut cmd = copy_cmd(CopyKind::TextureToBuffer, 3, 5);
    cmd.source_size = over;
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::Bounds
    );
    assert_eq!(blit_fail_reason(), "t2b_extent_oob");

    let mut cmd = copy_cmd(CopyKind::BufferToTexture, 5, 3);
    cmd.source_size = over;
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::Bounds
    );
    assert_eq!(blit_fail_reason(), "b2t_extent_oob");

    // An extent that exactly fills the target is not past the edge. The bound
    // is inclusive, and a refusal here would decline every full-surface copy —
    // which is most of them.
    let mut cmd = copy_cmd(CopyKind::TextureToTexture, 3, 4);
    cmd.source_size = Size {
        width: 2,
        height: 2,
        depth: 1,
    };
    assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);
}

/// The reason channel names *which* collapsed check fired inside each of the
/// rectangular copy executors (texture↔texture, texture→buffer, buffer→texture),
/// distinguishes distinct causes, and is reset to empty by a subsequent success —
/// so a `blit_fail reason=<slug> st=Bounds` dispatch line always carries the
/// specific failing site rather than a bare coarse status.
#[test]
fn copy_executor_reason_slugs_name_distinct_sites() {
    let (mut host, mut state) = blit_device();
    install_type11(&mut host, &mut state, 3, 3, 0x20); // 2×2 BGRA, mid 3
    install_type11(&mut host, &mut state, 4, 4, 0x21); // 2×2 BGRA, mid 4
    install_buffer(&mut host, &mut state, 5, 5, 4096);

    // texture→texture: destination origin past a 2×2 target → Bounds.
    let mut cmd = copy_cmd(CopyKind::TextureToTexture, 3, 4);
    cmd.destination_origin.x = 3; // > width 2
    cmd.source_size = Size {
        width: 1,
        height: 1,
        depth: 1,
    };
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::Bounds
    );
    assert_eq!(blit_fail_reason(), "t2t_origin_oob");

    // texture→texture: a type-11 endpoint with a non-zero z origin (type-11 is
    // 2D) → Unsupported, a DIFFERENT reason under the same executor.
    let mut cmd = copy_cmd(CopyKind::TextureToTexture, 3, 4);
    cmd.source_origin.z = 1;
    cmd.source_size = Size {
        width: 1,
        height: 1,
        depth: 1,
    };
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::Unsupported
    );
    assert_eq!(blit_fail_reason(), "t2t_t11_z");

    // texture→buffer: source origin past bounds → Bounds "t2b_origin_oob".
    let mut cmd = copy_cmd(CopyKind::TextureToBuffer, 3, 5);
    cmd.source_origin.x = 3; // > width 2
    cmd.source_size = Size {
        width: 1,
        height: 1,
        depth: 1,
    };
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::Bounds
    );
    assert_eq!(blit_fail_reason(), "t2b_origin_oob");

    // buffer→texture: destination origin past bounds → Bounds "b2t_origin_oob".
    let mut cmd = copy_cmd(CopyKind::BufferToTexture, 5, 3);
    cmd.destination_origin.x = 3; // > width 2
    cmd.source_size = Size {
        width: 1,
        height: 1,
        depth: 1,
    };
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::Bounds
    );
    assert_eq!(blit_fail_reason(), "b2t_origin_oob");

    // A full-target valid type-11→type-11 copy succeeds and resets the channel,
    // so no stale slug leaks into the next command's dispatch line.
    let mut cmd = copy_cmd(CopyKind::TextureToTexture, 3, 4);
    cmd.source_size = Size {
        width: 2,
        height: 2,
        depth: 1,
    };
    assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);
    assert_eq!(blit_fail_reason(), "");
}

/// Install type-11 object-list entry + mapping pages (2×2 BGRA).
fn install_type11(
    host: &mut FakeHost,
    state: &mut DeviceState,
    obj_ref: u32,
    mapping_id: u32,
    pfn: u32,
) {
    map_one_page_surface(host, state, mapping_id, pfn);
    assert!(state.set_mapping_geom(mapping_id, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    install_type11_plane(
        host,
        state,
        obj_ref,
        mapping_id,
        MTL_FORMAT_BGRA8_UNORM,
        2,
        2,
    );
}

/// Shared biplanar mapping: plane0 Y 4×2 R8 @512 bpr=64; plane1 UV 2×1 RG8 @1024 bpr=64.
fn install_biplanar_mapping(
    host: &mut FakeHost,
    state: &mut DeviceState,
    mapping_id: u32,
    pfn: u32,
) {
    use crate::contract::iosurface_pages::{
        DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_LEN, DEVICE_DESC_PLANES, DEVICE_DESC_PLANE_COUNT,
        DEVICE_PLANE_BPE, DEVICE_PLANE_BPR, DEVICE_PLANE_DESC_LEN, DEVICE_PLANE_DIMS,
        DEVICE_PLANE_OFFSET, DEVICE_PLANE_SIZE,
    };
    map_one_page_surface(host, state, mapping_id, pfn);
    let mut device = vec![0u8; DEVICE_DESC_LEN];
    st32(&mut device[DEVICE_DESC_ALLOC_SIZE..], 0x2000);
    device[DEVICE_DESC_PLANE_COUNT] = 2;
    let pack = |w: u32, h: u32| ((w as u64 & 0xffffff) << 8) | ((h as u64 & 0xffffff) << 40);
    let p0 = DEVICE_DESC_PLANES;
    st32(&mut device[p0 + DEVICE_PLANE_OFFSET..], 512);
    st32(&mut device[p0 + DEVICE_PLANE_SIZE..], 512);
    st64(&mut device[p0 + DEVICE_PLANE_DIMS..], pack(4, 2));
    st32(&mut device[p0 + DEVICE_PLANE_BPR..], 64);
    st16(&mut device[p0 + DEVICE_PLANE_BPE..], 1);
    let p1 = DEVICE_DESC_PLANES + DEVICE_PLANE_DESC_LEN;
    st32(&mut device[p1 + DEVICE_PLANE_OFFSET..], 1024);
    st32(&mut device[p1 + DEVICE_PLANE_SIZE..], 256);
    st64(&mut device[p1 + DEVICE_PLANE_DIMS..], pack(2, 1));
    st32(&mut device[p1 + DEVICE_PLANE_BPR..], 64);
    st16(&mut device[p1 + DEVICE_PLANE_BPE..], 2);
    assert!(state.set_mapping_device_desc(mapping_id, &device));
    // Surface-level geom is not the plane; leave has_geom false until texture latch.
}

fn install_type11_plane(
    host: &mut FakeHost,
    state: &mut DeviceState,
    obj_ref: u32,
    mapping_id: u32,
    format: u16,
    width: u32,
    height: u32,
) {
    use crate::runtime::decode::resource::OBJECT_TYPE_IOSURFACE;
    // iosurface desc layout: surfaceID @0, format @0x16, width @0x18, height @0x1c.
    const DESC_LEN: usize = 0x20;
    assert!(state.set_object_list(1, 0, 16));
    let mut desc = vec![0u8; DESC_LEN];
    st32(&mut desc[0..], mapping_id);
    st16(&mut desc[0x16..], format);
    st32(&mut desc[0x18..], width);
    st32(&mut desc[0x1c..], height);
    let desc_gva = 0x180u64 + (obj_ref as u64) * 0x40;
    write_task_gva_arm64e(host, &state.tasks[1], desc_gva, &desc);
    write_list_entry(
        host,
        state,
        obj_ref,
        OBJECT_TYPE_IOSURFACE as u32,
        DESC_LEN as u32,
        desc_gva,
    );
}

/// Install a type-5 RefTexture (object_type=5) that names an IOSurface
/// mapping via `surfaceID@+0` and a serialized 0x62 color-view record
/// (fmt@+0x16, w@+0x18, h@+0x1c, depth@+0x20, plane@+0x34 — the live blit-
/// source layout from `decode_type5_texture_view_live_0x62_color_window_view`).
/// Also installs a single-page mapping at `mapping_id` so the resolve lands.
fn install_type5(
    host: &mut FakeHost,
    state: &mut DeviceState,
    obj_ref: u32,
    mapping_id: u32,
    pfn: u32,
    format: u16,
    width: u32,
    height: u32,
) {
    // Mapping (surfaceID == mapping_id): mapped, one data page, latched geom.
    map_one_page_surface(host, state, mapping_id, pfn);
    assert!(state.set_mapping_geom(mapping_id, width, height, format));
    // Type-5 descriptor: 56-byte blob, 0x62 color-view record.
    assert!(state.set_object_list(1, 0, 16));
    let built = reims_vgpu_wire::device_desc::Type5Builder::new(
        mapping_id,
        0,
        obj_ref,
        objects::TYPE5_RECORD_TAG_COLOR_VIEW,
    )
    .geometry(format, width, height, 1)
    .with_len(56);
    let desc = built.bytes();
    let desc_len = desc.len();
    let desc_gva = 0x180u64 + (obj_ref as u64) * 0x40;
    write_task_gva_arm64e(host, &state.tasks[1], desc_gva, desc);
    write_list_entry(
        host,
        state,
        obj_ref,
        objects::OBJECT_TYPE_REF_TEXTURE as u32,
        desc_len as u32,
        desc_gva,
    );
    let e = objects::lookup_list_entry(state, host, 1, obj_ref).expect("type5 entry");
    assert_eq!(e.object_type, objects::OBJECT_TYPE_REF_TEXTURE);
}

/// A blit source must read the plane the wire named, and the only shape that
/// can prove it is two planes that share geometry and bytes-per-element.
///
/// This branch resolved type-5 views through `type11_sample_window`, which
/// takes no plane index and picks a plane by matching width, height and bpe.
/// On the v0a8 shape the live apple.com hero produces — Y and alpha both R8
/// at the luma geometry — that scan matches *two* records, takes neither, and
/// returns the invented packed window: plane 0's bytes at offset 0. So a COPY
/// from the alpha plane silently read luma, the copy succeeded, and nothing
/// downstream could tell.
///
/// The fixture is that shape at a size that fits one page. Plane 2 is byte-
/// identical in geometry to plane 0 and differs only in offset, so an
/// assertion on the offset is exactly the assertion that the index was used.
#[test]
fn a_type5_blit_source_reads_the_plane_the_wire_named() {
    use crate::contract::endian::st64;
    use crate::contract::iosurface_pages::{
        DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_LEN, DEVICE_DESC_PLANES, DEVICE_DESC_PLANE_COUNT,
        DEVICE_PLANE_BPE, DEVICE_PLANE_BPR, DEVICE_PLANE_DESC_LEN, DEVICE_PLANE_DIMS,
        DEVICE_PLANE_OFFSET, DEVICE_PLANE_SIZE,
    };
    use crate::contract::pixel_format::{MTL_FORMAT_R8_UNORM, MTL_FORMAT_RG8_UNORM};
    // Device-plane dims word: width u24@1, height u24@5 (`decode_device_plane`).
    let pack_plane_dims =
        |w: u32, h: u32| ((w as u64 & 0xffffff) << 8) | ((h as u64 & 0xffffff) << 40);

    let (mut host, mut state) = blit_device();
    let (mapping_id, obj_ref) = (34u32, 12u32);
    let (w, h) = (8u32, 4u32);
    // Plane 2 is the alpha plane: same 8x4 R8 as plane 0, different offset.
    const ALPHA_OFFSET: u32 = 128;
    install_type5(
        &mut host,
        &mut state,
        obj_ref,
        mapping_id,
        0x30,
        MTL_FORMAT_R8_UNORM,
        w,
        h,
    );
    set_type5_record_plane(&mut host, &state, obj_ref, 2);

    let mut desc = vec![0u8; DEVICE_DESC_LEN];
    st32(&mut desc[DEVICE_DESC_ALLOC_SIZE..], 0x1000);
    desc[DEVICE_DESC_PLANE_COUNT] = 3;
    // (offset, size, width, height, bpr, bpe)
    let planes = [
        (0u32, 32u32, 8u32, 4u32, 8u32, 1u16),
        (64, 16, 4, 2, 8, 2),
        (ALPHA_OFFSET, 32, 8, 4, 8, 1),
    ];
    for (i, (off, size, pw, ph, bpr, bpe)) in planes.iter().enumerate() {
        let base = DEVICE_DESC_PLANES + i * DEVICE_PLANE_DESC_LEN;
        st32(&mut desc[base + DEVICE_PLANE_OFFSET..], *off);
        st32(&mut desc[base + DEVICE_PLANE_SIZE..], *size);
        st64(
            &mut desc[base + DEVICE_PLANE_DIMS..],
            pack_plane_dims(*pw, *ph),
        );
        st32(&mut desc[base + DEVICE_PLANE_BPR..], *bpr);
        st16(&mut desc[base + DEVICE_PLANE_BPE..], *bpe);
    }
    state
        .mappings
        .get_mut(&mapping_id)
        .expect("mapping")
        .device_desc = desc;

    // The scan is genuinely ambiguous on this descriptor: without the index
    // it resolves nothing at all, which is what this test exists to exclude.
    let ambiguous = {
        let m = state.mappings.get(&mapping_id).unwrap();
        mapping_write::type11_sample_window(m, w, h, MTL_FORMAT_R8_UNORM)
    };
    assert!(
        ambiguous.is_none(),
        "fixture must be ambiguous by geometry, or it proves nothing"
    );
    // Sanity: plane 1 is a different geometry, so the ambiguity is between
    // planes 0 and 2 specifically and not a descriptor that resolves nothing.
    let uv = {
        let m = state.mappings.get(&mapping_id).unwrap();
        mapping_write::type5_sample_window(m, 1, 4, 2, MTL_FORMAT_RG8_UNORM)
    };
    assert_eq!(uv.map(|(off, _, _)| off), Some(64));

    let backing = resolve_texture_backing(&mut state, &mut host, 1, obj_ref, 0, 0)
        .expect("type-5 blit source must resolve");
    match backing {
        TextureBacking::Type11(t) => assert_eq!(
            t.surface_offset, ALPHA_OFFSET as u64,
            "the wire named plane 2, and only the wire index can reach it"
        ),
        TextureBacking::Linear(_) => panic!("expected Type11 backing, got Linear"),
    }
}

/// Overwrite the plane index in an installed type-5 record.
///
/// `install_type5` leaves it 0 (the field sits past the fields it writes and
/// the blob is zeroed), which is the one value that cannot distinguish a
/// used index from a dropped one.
fn set_type5_record_plane(host: &mut FakeHost, state: &DeviceState, obj_ref: u32, plane: u32) {
    let off = objects::TYPE5_ARG_RECORD + objects::TYPE5_RECORD_PLANE;
    let desc_gva = 0x180u64 + (obj_ref as u64) * 0x40;
    let mut word = [0u8; 4];
    st32(&mut word, plane);
    write_task_gva_arm64e(host, &state.tasks[1], desc_gva + off as u64, &word);
    let entry = objects::lookup_list_entry(state, host, 1, obj_ref).expect("type5 entry");
    let desc = objects::read_descriptor(state, host, 1, &entry).expect("type5 desc");
    assert_eq!(
        objects::decode_type5_texture_view(&desc)
            .expect("view")
            .plane_index,
        plane
    );
}

/// Regression guard for the type-5 RefTexture blit-source branch
/// (`resolve_texture_backing_depth` ~588): a type-5 object whose 0x62 record
/// names a BGRA8 view must resolve to a `Type11` backing carrying the VIEW
/// geometry/format (not the base mapping's), so a blit copy from a media /
/// window backing lands. Mirrors the type-11 install fixtures.
#[test]
fn type5_ref_texture_resolves_as_type11_blit_backing() {
    use crate::contract::pixel_format::bytes_per_pixel;
    let (mut host, mut state) = blit_device();
    let mapping_id = 34u32;
    let obj_ref = 12u32;
    let (w, h, fmt) = (2u32, 2u32, MTL_FORMAT_BGRA8_UNORM);
    install_type5(&mut host, &mut state, obj_ref, mapping_id, 0x30, fmt, w, h);
    let backing = resolve_texture_backing(&mut state, &mut host, 1, obj_ref, 0, 0)
        .expect("type-5 blit source must resolve");
    match backing {
        TextureBacking::Type11(t) => {
            assert_eq!(t.mapping_id, mapping_id, "backs the named surface");
            assert_eq!((t.width, t.height), (w, h), "view geometry, not base");
            assert_eq!(t.pixel_format, fmt);
            assert_eq!(t.bpp, bytes_per_pixel(fmt).unwrap());
            assert!(u64::from(t.row_stride) >= u64::from(w) * u64::from(t.bpp));
            assert!(t.span_end >= u64::from(t.row_stride) * u64::from(h));
        }
        TextureBacking::Linear(_) => panic!("expected Type11 backing, got Linear"),
    }
}

/// A type-5 record whose tag is neither 0x42 nor 0x62 is unknown wire → the
/// blit branch must fail closed (`t5_view_decode`), never invent geometry.
#[test]
fn type5_unknown_record_tag_fails_closed() {
    let (mut host, mut state) = blit_device();
    let (mapping_id, obj_ref) = (34u32, 12u32);
    install_type5(
        &mut host,
        &mut state,
        obj_ref,
        mapping_id,
        0x30,
        MTL_FORMAT_BGRA8_UNORM,
        2,
        2,
    );
    // Corrupt the record tag to an unknown value in-place.
    let desc_gva = 0x180u64 + (obj_ref as u64) * 0x40;
    let bad = [0x99u8];
    write_task_gva_arm64e(
        &mut host,
        &state.tasks[1],
        desc_gva + objects::TYPE5_ARG_RECORD as u64,
        &bad,
    );
    match resolve_texture_backing(&mut state, &mut host, 1, obj_ref, 0, 0) {
        Err(st) => assert_eq!(st, BlitStatus::Unsupported),
        Ok(_) => panic!("unknown type-5 record tag must fail closed"),
    }
}

#[test]
fn biplanar_type11_y_and_uv_planes_distinct() {
    use crate::contract::pixel_format::{MTL_FORMAT_R8_UNORM, MTL_FORMAT_RG8_UNORM};
    let (mut host, mut state) = blit_device();
    let mapping_id = 7u32;
    install_biplanar_mapping(&mut host, &mut state, mapping_id, 0x30);
    // Y plane texture ref 10, UV plane texture ref 11 — same mapping_id.
    install_type11_plane(
        &mut host,
        &mut state,
        10,
        mapping_id,
        MTL_FORMAT_R8_UNORM,
        4,
        2,
    );
    install_type11_plane(
        &mut host,
        &mut state,
        11,
        mapping_id,
        MTL_FORMAT_RG8_UNORM,
        2,
        1,
    );

    // Buffer with pattern for Y (4×2 R8 tight = 8 B, use bpr 4).
    let y_pat = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    {
        use crate::runtime::decode::resource::{
            list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_BUFFER,
            RESOURCE_PAGE_SHIFT,
        };
        let mut bdesc = vec![0u8; 16];
        st64(&mut bdesc[0..], 64);
        st32(&mut bdesc[8..], 2); // handle 2
        let bgva = 0x300u64;
        write_task_gva_arm64e(&mut host, &state.tasks[1], bgva, &bdesc);
        let off = list_object_entry_offset(1, 16).unwrap();
        let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_BUFFER as u32) | (16u32 << 8);
        st32(&mut le[0..], packed);
        le[4..12].copy_from_slice(&bgva.to_le_bytes());
        write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);
        let buf_gva = 2u64 << RESOURCE_PAGE_SHIFT;
        write_task_gva_arm64e(&mut host, &state.tasks[1], buf_gva, &y_pat);
    }

    let mut cmd = copy_cmd(CopyKind::BufferToTexture, 1, 10); // Y
    cmd.source_offset = 0;
    cmd.source_bytes_per_row = 4;
    cmd.source_size = Size {
        width: 4,
        height: 2,
        depth: 1,
    };
    assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);

    // Y plane base 512, bpr 64: rows at 512 and 576.
    let mut row0 = [0u8; 4];
    let mut row1 = [0u8; 4];
    assert!(mapping_write::read_rect_raw_at(
        &mut state,
        &mut host,
        mapping_id,
        mapping_write::SurfaceWindow {
            base_off: 512,
            bpr: 64,
            span_end: 512 + 64 + 4,
            bpp: 1
        },
        mapping_write::Rect {
            origin_x: 0,
            origin_y: 0,
            width: 4,
            height: 1
        },
        &mut row0,
        4
    ));
    assert!(mapping_write::read_rect_raw_at(
        &mut state,
        &mut host,
        mapping_id,
        mapping_write::SurfaceWindow {
            base_off: 512,
            bpr: 64,
            span_end: 512 + 64 + 4,
            bpp: 1
        },
        mapping_write::Rect {
            origin_x: 0,
            origin_y: 1,
            width: 4,
            height: 1
        },
        &mut row1,
        4
    ));
    assert_eq!(row0, y_pat[0..4]);
    assert_eq!(row1, y_pat[4..8]);

    // UV plane: write 2×1 RG8 (4 B) from same buffer offset 0.
    let uv_pat = [0xaau8, 0xbb, 0xcc, 0xdd];
    {
        let buf_gva = 2u64 << crate::runtime::decode::resource::RESOURCE_PAGE_SHIFT;
        write_task_gva_arm64e(&mut host, &state.tasks[1], buf_gva, &uv_pat);
    }
    cmd.destination = 11;
    cmd.source_bytes_per_row = 4;
    cmd.source_size = Size {
        width: 2,
        height: 1,
        depth: 1,
    };
    assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);

    let mut uv = [0u8; 4];
    assert!(mapping_write::read_rect_raw_at(
        &mut state,
        &mut host,
        mapping_id,
        mapping_write::SurfaceWindow {
            base_off: 1024,
            bpr: 64,
            span_end: 1024 + 4,
            bpp: 2
        },
        mapping_write::Rect {
            origin_x: 0,
            origin_y: 0,
            width: 2,
            height: 1
        },
        &mut uv,
        4
    ));
    assert_eq!(uv, uv_pat);
    // Y plane must be untouched by UV write.
    assert!(mapping_write::read_rect_raw_at(
        &mut state,
        &mut host,
        mapping_id,
        mapping_write::SurfaceWindow {
            base_off: 512,
            bpr: 64,
            span_end: 512 + 64 + 4,
            bpp: 1
        },
        mapping_write::Rect {
            origin_x: 0,
            origin_y: 0,
            width: 4,
            height: 1
        },
        &mut row0,
        4
    ));
    assert_eq!(row0, y_pat[0..4]);
}

fn install_type8_view(
    host: &mut FakeHost,
    state: &mut DeviceState,
    view_ref: u32,
    base_ref: u32,
    pixel_format: u16,
    level_base: u64,
    swizzle: Option<[u8; 4]>,
) {
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE_VIEW,
        TEXTURE_VIEW_DESC_BASE_REF, TEXTURE_VIEW_DESC_LEN, TEXTURE_VIEW_DESC_LEVEL_BASE,
        TEXTURE_VIEW_DESC_LEVEL_COUNT, TEXTURE_VIEW_DESC_OPCODE, TEXTURE_VIEW_DESC_PIXEL_FORMAT,
        TEXTURE_VIEW_DESC_SLICE_BASE, TEXTURE_VIEW_DESC_SLICE_COUNT, TEXTURE_VIEW_DESC_SWIZZLE,
        TEXTURE_VIEW_DESC_TEXTURE_REF, TEXTURE_VIEW_DESC_TEXTURE_TYPE, TEXTURE_VIEW_MIN_RANGED,
        TEXTURE_VIEW_MIN_SWIZZLE, TEXTURE_VIEW_MTL_TYPE_2D, TEXTURE_VIEW_OPCODE_RANGED,
        TEXTURE_VIEW_OPCODE_SWIZZLE,
    };
    let (opcode, len) = if swizzle.is_some() {
        (TEXTURE_VIEW_OPCODE_SWIZZLE, TEXTURE_VIEW_MIN_SWIZZLE)
    } else {
        (TEXTURE_VIEW_OPCODE_RANGED, TEXTURE_VIEW_MIN_RANGED)
    };
    let mut desc = vec![0u8; len];
    st32(&mut desc[TEXTURE_VIEW_DESC_OPCODE..], opcode);
    st32(&mut desc[TEXTURE_VIEW_DESC_LEN..], len as u32);
    st32(&mut desc[TEXTURE_VIEW_DESC_TEXTURE_REF..], view_ref);
    st32(&mut desc[TEXTURE_VIEW_DESC_BASE_REF..], base_ref);
    st16(&mut desc[TEXTURE_VIEW_DESC_PIXEL_FORMAT..], pixel_format);
    st16(
        &mut desc[TEXTURE_VIEW_DESC_TEXTURE_TYPE..],
        TEXTURE_VIEW_MTL_TYPE_2D,
    );
    st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_BASE..], level_base);
    st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_COUNT..], 1);
    st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_BASE..], 0);
    st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_COUNT..], 1);
    if let Some(sw) = swizzle {
        desc[TEXTURE_VIEW_DESC_SWIZZLE..TEXTURE_VIEW_DESC_SWIZZLE + 4].copy_from_slice(&sw);
    }
    let desc_gva = 0x280u64 + (view_ref as u64) * 0x40;
    write_task_gva_arm64e(host, &state.tasks[1], desc_gva, &desc);
    let off = list_object_entry_offset(view_ref, 16).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (OBJECT_TYPE_TEXTURE_VIEW as u32) | ((len as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(host, &state.tasks[1], off, &list_entry);
}

#[test]
fn copy_buffer_to_type8_view_of_type11() {
    let (mut host, mut state) = blit_device();
    install_buffer(&mut host, &mut state, 1, 1, 256);
    let mapping_id = 9u32;
    install_type11(&mut host, &mut state, 3, mapping_id, 0x20);
    // View ref 8 → base 3, level 0, BGRA identity.
    install_type8_view(&mut host, &mut state, 8, 3, MTL_FORMAT_BGRA8_UNORM, 0, None);
    let pat = [0xaau8, 0xbb, 0xcc, 0xdd, 0x11, 0x22, 0x33, 0x44];
    let src_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    write_task_gva_arm64e(&mut host, &state.tasks[1], src_gva, &pat);
    let mut cmd = copy_cmd(CopyKind::BufferToTexture, 1, 8); // type-8 view
    cmd.source_offset = 0;
    cmd.source_bytes_per_row = 8;
    cmd.source_size = Size {
        width: 2,
        height: 1,
        depth: 1,
    };
    cmd.destination_origin.y = 0;
    assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);
    let mut back = [0u8; 8];
    assert!(mapping_write::read_rect_raw(
        &mut state,
        &mut host,
        mapping_id,
        mapping_write::Rect {
            origin_x: 0,
            origin_y: 0,
            width: 2,
            height: 1
        },
        &mut back,
        8
    ));
    assert_eq!(back, pat);
}

#[test]
fn type8_swizzled_view_rejected_for_blit() {
    let (mut host, mut state) = blit_device();
    install_buffer(&mut host, &mut state, 1, 1, 256);
    install_type11(&mut host, &mut state, 3, 9, 0x20);
    // Non-identity swizzle BGRA order selectors.
    install_type8_view(
        &mut host,
        &mut state,
        8,
        3,
        MTL_FORMAT_BGRA8_UNORM,
        0,
        Some([4, 3, 2, 5]),
    );
    let mut cmd = copy_cmd(CopyKind::BufferToTexture, 1, 8);
    cmd.source_size = Size {
        width: 1,
        height: 1,
        depth: 1,
    };
    cmd.source_bytes_per_row = 4;
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::Unsupported
    );
}

#[test]
fn type8_level_base_on_type11_rejected() {
    // Metal forbids mipmapped IOSurfaces; view level_base=1 fail-closes.
    let (mut host, mut state) = blit_device();
    install_buffer(&mut host, &mut state, 1, 1, 256);
    install_type11(&mut host, &mut state, 3, 9, 0x20);
    install_type8_view(
        &mut host,
        &mut state,
        8,
        3,
        MTL_FORMAT_BGRA8_UNORM,
        1, // level_base
        None,
    );
    let mut cmd = copy_cmd(CopyKind::BufferToTexture, 1, 8);
    cmd.source_size = Size {
        width: 1,
        height: 1,
        depth: 1,
    };
    cmd.source_bytes_per_row = 4;
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::Unsupported
    );
}

/// A compressed level's byte arithmetic is in blocks, in both axes.
///
/// The regression this pins is 448 refused copies on one driven Asphalt 8 leg:
/// `blit_fail reason=tex_bad_bpp kind=Copy` between two BC3 textures, because
/// the whole blit path asked `bytes_per_pixel` and a BC format has none. Now the
/// backing carries its storage grid, and `exec_copy_texture_to_texture` converts
/// the command's coordinates into block space once so every per-unit helper
/// below it is correct unchanged.
///
/// Both halves are asserted because each fails differently. A wrong
/// `bytes_per_image` strides an array slice or a `z` plane past its own image; a
/// wrong `texel_offset` reads the right number of bytes from the wrong place.
#[test]
fn a_compressed_level_addresses_blocks_in_both_axes() {
    use crate::contract::pixel_format::{self as pf, BlockGeometry};
    // A 64x64 BC3 level as a guest sends it: 16 block columns of 16 bytes, and
    // 16 block rows, so one image is 4096 bytes and not 16384.
    let bc3 = LinearTextureLevel {
        base_gva: 0,
        alloc_size: 4096,
        level_offset: 0,
        row_stride: 256,
        slice_stride: 4096,
        slice_index: 0,
        width: 64,
        height: 64,
        depth: 1,
        bpp: pf::BC_BLOCK_BYTES_16,
        block: pf::block_geometry(pf::MTL_FORMAT_BC3_RGBA).expect("bc3 has a grid"),
        pixel_format: pf::MTL_FORMAT_BC3_RGBA,
    };
    assert_eq!(bc3.block, BlockGeometry { width: 4, height: 4, bytes: 16 });
    assert_eq!(
        bc3.bytes_per_image(),
        Some(4096),
        "16 block rows of 256 bytes — the texel form would say 16384 and stride \
         a slice four times past its own image"
    );
    // Block coordinates, which is what the copy hands it: block (3, 2) starts at
    // row 2 of blocks and column 3, i.e. 2*256 + 3*16.
    assert_eq!(bc3.texel_offset(3, 2, 0), Some(2 * 256 + 3 * 16));
    // The last block of the level is the last 16 bytes of the allocation, so the
    // grid and the allocation agree exactly — a level that did not would be
    // refused against `alloc_size` downstream.
    assert_eq!(bc3.texel_offset(15, 15, 0), Some(4096 - 16));

    // An uncompressed level is untouched by any of it: a 1x1 block whose bytes
    // are the bytes-per-texel gives back the products this always computed.
    let rgba = LinearTextureLevel {
        base_gva: 0,
        alloc_size: 0x1000,
        level_offset: 0,
        row_stride: 256,
        slice_stride: 0,
        slice_index: 0,
        width: 64,
        height: 4,
        depth: 1,
        bpp: 4,
        block: pf::block_geometry(MTL_FORMAT_RGBA8_UNORM).expect("rgba8 has a grid"),
        pixel_format: MTL_FORMAT_RGBA8_UNORM,
    };
    assert_eq!(rgba.block, BlockGeometry { width: 1, height: 1, bytes: 4 });
    assert_eq!(rgba.bytes_per_image(), Some(256 * 4));
    assert_eq!(rgba.texel_offset(3, 2, 0), Some(2 * 256 + 3 * 4));
}

#[test]
fn texel_offset_math() {
    let t = LinearTextureLevel {
        base_gva: 0,
        alloc_size: 0x1000,
        level_offset: 0x100,
        row_stride: 16,
        slice_stride: 64,
        slice_index: 0,
        width: 4,
        height: 4,
        depth: 1,
        bpp: 4,
        block: crate::contract::pixel_format::BlockGeometry {
            width: 1,
            height: 1,
            bytes: 4,
        },
        pixel_format: MTL_FORMAT_RGBA8_UNORM,
    };
    // (x=1,y=2) → 0x100 + 2*16 + 1*4 = 0x124
    assert_eq!(t.texel_offset(1, 2, 0), Some(0x124));
    // The plane term, which nothing asserted before: one image is
    // `row_stride * height` = 16 * 4 = 64 bytes, and `z` steps by that.
    //
    // Worth its own line now that the three coordinates arrive as one
    // `Point`: `x`, `y` and `z` are three `u64`s that a destructure could
    // cross. The strides here are 4, 16 and 64 — all different — so this
    // one call tells every pair of them apart.
    assert_eq!(t.texel_offset(1, 2, 1), Some(0x124 + 64));
    let mut t1 = t;
    t1.slice_index = 2;
    // + 2 * 64 slice stride
    assert_eq!(t1.texel_offset(1, 2, 0), Some(0x124 + 128));
}

/// Geometry this device cannot measure must refuse the copy, never hand the
/// row loop the `None` that means "authorised by the command".
///
/// `write_texture_row` consults `allowed` only when it is `Some`, so a
/// `None` produced by arithmetic that overflowed widens the write from the
/// region the guest named to every page the task can reach. Before this was
/// a `Result` the whole ladder answered `None`, so the wider write was also
/// the silent one: it reached no counter, because every early return
/// happened before `dest_window` — the only thing that reports here — was
/// ever called.
#[test]
fn an_unmeasurable_copy_region_refuses_rather_than_writing_unbounded() {
    let (host, state) = blit_device();
    let level = |row_stride: u64| LinearTextureLevel {
        base_gva: 0x1000,
        alloc_size: 0x1000,
        level_offset: 0,
        row_stride,
        slice_stride: 0,
        slice_index: 0,
        width: 4,
        height: 1,
        depth: 1,
        bpp: 4,
        block: crate::contract::pixel_format::BlockGeometry {
            width: 1,
            height: 1,
            bytes: 4,
        },
        pixel_format: MTL_FORMAT_RGBA8_UNORM,
    };

    // Row 0 resolves and row 7's `y * row_stride` does not, which is the
    // case that matters: the loop would have written its way up to the bad
    // row before anything noticed.
    let tex = TextureBacking::Linear(level(1 << 62));
    clear_blit_fail_reason();
    assert_eq!(
        texture_region_window(
            &state,
            &host,
            1,
            &tex,
            Point { x: 0, y: 0, z: 0 },
            4,
            8,
            1,
            4
        ),
        Err(BlitStatus::Bounds),
        "an overflowing region must refuse the copy"
    );
    assert_eq!(
        blit_fail_reason(),
        "tex_window_last_texel_oob",
        "the refusal must name itself on the always-on failure channel"
    );

    // A zero extent is not a failure: the row loops are `0..copy_d` /
    // `0..copy_h`, so they write nothing and no page is authorised.
    let tex = TextureBacking::Linear(level(16));
    assert_eq!(
        texture_region_window(
            &state,
            &host,
            1,
            &tex,
            Point { x: 0, y: 0, z: 0 },
            4,
            0,
            1,
            4
        ),
        Ok(Some(std::collections::HashSet::new())),
        "an empty copy authorises no page"
    );
}

/// The array-slice stride this rail charges, now read off the level layout that
/// owns it rather than from three loose arguments.
///
/// Kept here as well as beside the method because this is the rail that reads
/// it: `one_slice` is what a selected slice's offset is multiplied by, so a
/// stride that gained or lost a plane would move every slice bound on this path.
#[test]
fn derived_slice_stride_2d() {
    use crate::runtime::decode::resource::TextureLevelLayout;
    let level = |row_stride: u64, height: u32, depth: u32| TextureLevelLayout {
        offset: 0,
        size: 0,
        row_stride,
        width: 1,
        height,
        depth,
    };
    assert_eq!(level(256, 32, 1).slice_stride(), Some(256 * 32));
    assert_eq!(level(16, 4, 2).slice_stride(), Some(16 * 4 * 2));
    assert_eq!(
        level(16, 4, 0).slice_stride(),
        Some(16 * 4),
        "depth 0 is one plane, the same encoding `slice_read_span` reads"
    );
}

/// Multi-mip linear texture + multi-level type-8 view selecting L1.
#[test]
fn copy_buffer_to_multilevel_view_l1() {
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE, RESOURCE_PAGE_SHIFT,
        TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_DATA_OFFSET, TEXTURE_DESC_HEIGHT,
        TEXTURE_DESC_LEVEL_RECORDS, TEXTURE_DESC_MIPMAP_LEVEL_COUNT,
        TEXTURE_DESC_MIP_LEVEL_RECORD_LEN, TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE,
        TEXTURE_DESC_USED_SIZE, TEXTURE_DESC_WIDTH, TEXTURE_LEVEL_DEPTH, TEXTURE_LEVEL_HEIGHT,
        TEXTURE_LEVEL_OFFSET, TEXTURE_LEVEL_ROW_STRIDE, TEXTURE_LEVEL_SIZE, TEXTURE_LEVEL_WIDTH,
    };
    let (mut host, mut state) = blit_device();
    install_buffer(&mut host, &mut state, 1, 1, 512);

    // Type-2 texture handle=2, 2 mips: L0 4x2, L1 2x1, RGBA8 (bpp=4).
    let handle = 2u32;
    let levels = 2u32;
    let body = TEXTURE_DESC_BASE_LEN + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
    let mut desc = vec![0u8; body];
    st64(&mut desc[0..], 0x4000); // allocation
    st32(&mut desc[8..], handle);
    st16(&mut desc[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], levels as u16);
    st32(&mut desc[TEXTURE_DESC_DATA_OFFSET..], 0);
    st32(&mut desc[TEXTURE_DESC_USED_SIZE..], 4 * 2 * 4); // L0
    st32(&mut desc[TEXTURE_DESC_ROW_STRIDE..], 16);
    st32(&mut desc[TEXTURE_DESC_WIDTH..], 4);
    st32(&mut desc[TEXTURE_DESC_HEIGHT..], 2);
    let rec = TEXTURE_DESC_LEVEL_RECORDS;
    st64(&mut desc[rec + TEXTURE_LEVEL_OFFSET..], 32); // L1 after L0 32 bytes
    st64(&mut desc[rec + TEXTURE_LEVEL_SIZE..], 8);
    st64(&mut desc[rec + TEXTURE_LEVEL_ROW_STRIDE..], 8);
    st32(&mut desc[rec + TEXTURE_LEVEL_WIDTH..], 2);
    st32(&mut desc[rec + TEXTURE_LEVEL_HEIGHT..], 1);
    st32(&mut desc[rec + TEXTURE_LEVEL_DEPTH..], 1);
    let pf_off = TEXTURE_DESC_PIXEL_FORMAT + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
    st16(&mut desc[pf_off..], MTL_FORMAT_RGBA8_UNORM);
    let desc_gva = 0x300u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &desc);
    let off = list_object_entry_offset(4, 16).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (OBJECT_TYPE_TEXTURE as u32) | ((body as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    // View: level_base=0, level_count=2 over texture ref 4.
    install_type8_view(&mut host, &mut state, 8, 4, MTL_FORMAT_RGBA8_UNORM, 0, None);
    // Patch level_count to 2 on the installed view.
    {
        use crate::runtime::decode::resource::{
            TEXTURE_VIEW_DESC_LEVEL_COUNT, TEXTURE_VIEW_MIN_RANGED,
        };
        let view_gva = 0x280u64 + 8 * 0x40;
        let mut v = vec![0u8; TEXTURE_VIEW_MIN_RANGED];
        assert!(gva_mem::read_task_gva(
            &host,
            &state.tasks[1],
            view_gva,
            &mut v,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
        st64(&mut v[TEXTURE_VIEW_DESC_LEVEL_COUNT..], 2);
        write_task_gva_arm64e(&mut host, &state.tasks[1], view_gva, &v);
    }

    // Seed buffer with 2 RGBA pixels for L1 (2x1).
    let pat = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let src_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    write_task_gva_arm64e(&mut host, &state.tasks[1], src_gva, &pat);

    let mut cmd = copy_cmd(CopyKind::BufferToTexture, 1, 8);
    cmd.source_level = 1; // relative → absolute L1
    cmd.destination_level = 1;
    cmd.source_offset = 0;
    cmd.source_bytes_per_row = 8;
    cmd.source_size = Size {
        width: 2,
        height: 1,
        depth: 1,
    };
    assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);

    // Read L1 from texture handle 2 GVA + 32.
    let l1_gva = ((handle as u64) << RESOURCE_PAGE_SHIFT) + 32;
    let mut back = [0u8; 8];
    assert!(
        gva_mem::read_task_gva(&host, &state.tasks[1], l1_gva, &mut back, PAGE_SHIFT_ARM64E)
            .is_ok()
    );
    assert_eq!(back, pat);
}

#[test]
fn multilevel_view_relative_level_oob() {
    let (mut host, mut state) = blit_device();
    install_buffer(&mut host, &mut state, 1, 1, 64);
    install_type11(&mut host, &mut state, 3, 9, 0x20);
    // View over type-11 with level_count=1, level_base=0; command level 1 is OOB.
    install_type8_view(&mut host, &mut state, 8, 3, MTL_FORMAT_BGRA8_UNORM, 0, None);
    let mut cmd = copy_cmd(CopyKind::BufferToTexture, 1, 8);
    cmd.destination_level = 1; // relative 1 >= count 1
    cmd.source_size = Size {
        width: 1,
        height: 1,
        depth: 1,
    };
    cmd.source_bytes_per_row = 4;
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::Bounds
    );
}

#[test]
fn texture_view_type_helpers() {
    use crate::runtime::decode::resource::{
        texture_view_type_is_3d, texture_view_type_supported, texture_view_type_uses_slices,
        TEXTURE_VIEW_MTL_TYPE_2D, TEXTURE_VIEW_MTL_TYPE_2D_ARRAY,
        TEXTURE_VIEW_MTL_TYPE_2D_MULTISAMPLE, TEXTURE_VIEW_MTL_TYPE_3D, TEXTURE_VIEW_MTL_TYPE_CUBE,
    };
    assert!(texture_view_type_supported(TEXTURE_VIEW_MTL_TYPE_2D));
    assert!(texture_view_type_supported(TEXTURE_VIEW_MTL_TYPE_2D_ARRAY));
    assert!(texture_view_type_supported(TEXTURE_VIEW_MTL_TYPE_CUBE));
    assert!(texture_view_type_supported(TEXTURE_VIEW_MTL_TYPE_3D));
    assert!(!texture_view_type_supported(
        TEXTURE_VIEW_MTL_TYPE_2D_MULTISAMPLE
    ));
    assert!(texture_view_type_uses_slices(
        TEXTURE_VIEW_MTL_TYPE_2D_ARRAY
    ));
    assert!(texture_view_type_uses_slices(TEXTURE_VIEW_MTL_TYPE_CUBE));
    assert!(!texture_view_type_uses_slices(TEXTURE_VIEW_MTL_TYPE_2D));
    assert!(texture_view_type_is_3d(TEXTURE_VIEW_MTL_TYPE_3D));
}

#[test]
fn decode_copy_slice_level_0x13e() {
    use crate::runtime::decode::blit::{self};
    let mut v = vec![0u8; 0x1c];
    st32(&mut v[0..], wire_blit::OPCODE_COPY_TEXTURE_SLICES);
    st32(&mut v[4..], 0x1c);
    st32(&mut v[8..], 2);
    st32(&mut v[12..], 3);
    st16(&mut v[0x10..], 1);
    st16(&mut v[0x12..], 0);
    st16(&mut v[0x14..], 0);
    st16(&mut v[0x16..], 1);
    st16(&mut v[0x18..], 2);
    st16(&mut v[0x1a..], 3);
    let c = blit::decode(&v).unwrap();
    assert_eq!(c.copy_kind, CopyKind::TextureToTextureSliceLevel);
    assert_eq!(c.source, 2);
    assert_eq!(c.destination, 3);
    assert_eq!(c.source_slice, 1);
    assert_eq!(c.destination_level, 1);
    assert_eq!(c.slice_count, 2);
    assert_eq!(c.level_count, 3);
}

#[test]
fn slice_level_zero_counts_are_noop() {
    let (mut host, mut state) = blit_device();
    let mut cmd = copy_cmd(CopyKind::TextureToTextureSliceLevel, 1, 2);
    cmd.slice_count = 0;
    cmd.level_count = 1;
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::ZeroExtent
    );
    cmd.slice_count = 1;
    cmd.level_count = 0;
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::ZeroExtent
    );
}

/// Install a simple type-2 RGBA8 texture (single level, handle → GVA).
fn install_linear_rgba(
    host: &mut FakeHost,
    state: &mut DeviceState,
    obj_ref: u32,
    handle: u32,
    width: u32,
    height: u32,
    row_stride: u32,
) {
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE, RESOURCE_PAGE_SHIFT,
        TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_DATA_OFFSET, TEXTURE_DESC_HEIGHT,
        TEXTURE_DESC_MIPMAP_LEVEL_COUNT, TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE,
        TEXTURE_DESC_USED_SIZE, TEXTURE_DESC_WIDTH,
    };
    let _ = RESOURCE_PAGE_SHIFT;
    let mut desc = vec![0u8; TEXTURE_DESC_BASE_LEN];
    let size = (row_stride as u64) * (height as u64);
    st64(&mut desc[0..], size.max(0x1000));
    st32(&mut desc[8..], handle);
    st16(&mut desc[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 1);
    st32(&mut desc[TEXTURE_DESC_DATA_OFFSET..], 0);
    st32(&mut desc[TEXTURE_DESC_USED_SIZE..], size as u32);
    st32(&mut desc[TEXTURE_DESC_ROW_STRIDE..], row_stride);
    st32(&mut desc[TEXTURE_DESC_WIDTH..], width);
    st32(&mut desc[TEXTURE_DESC_HEIGHT..], height);
    st16(
        &mut desc[TEXTURE_DESC_PIXEL_FORMAT..],
        MTL_FORMAT_RGBA8_UNORM,
    );
    let desc_gva = 0x200u64 + (obj_ref as u64) * 0x80;
    write_task_gva_arm64e(host, &state.tasks[1], desc_gva, &desc);
    assert!(state.set_object_list(1, 0, 16));
    let off = list_object_entry_offset(obj_ref, 16).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (OBJECT_TYPE_TEXTURE as u32) | ((TEXTURE_DESC_BASE_LEN as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(host, &state.tasks[1], off, &list_entry);
}

/// A blit endpoint is a guest-byte reader, so resolving one must land whatever
/// this device still owes that resource's pages.
///
/// A render pass into an ordinary private `MTLTexture` resolves on
/// `render_target`'s linear rung, and `writeback_debt::arm_gva` leaves the result
/// in the engine's resident with a debt armed against `(task_id, texture_ref)`
/// rather than copying it into guest memory. `draw::texture_view` and
/// `compute_exec` named that resource before touching its bytes; the blit rail
/// did not, and every `copyFromTexture:` out of such a target copied the bytes
/// the pages held before the pass — zeros for a freshly allocated one.
///
/// The ledger is what this can assert on both backend arms: `pay_gva` itself is
/// `backend-vulkan`-only, but `take_gva` is not, so a resolve that named the debt
/// leaves the ledger empty and a resolve that did not leaves it holding one.
#[test]
fn a_blit_endpoint_lands_the_writeback_its_texture_still_owes() {
    use crate::runtime::writeback_debt::{GvaResourceKey, GvaWritebackDebt};

    const STRIDE: u32 = 64;
    let (mut host, mut state) = blit_device();
    install_linear_rgba(&mut host, &mut state, 2, 2, 16, 8, STRIDE);

    let key = GvaResourceKey {
        task_id: 1,
        texture_ref: 2,
    };
    let guest_write = state.buffer_write_gen.stamp(key.task_id, key.texture_ref);
    assert_eq!(
        state.pending_writebacks.arm_gva(
            key,
            GvaWritebackDebt {
                gva: 0x4000,
                row_stride: STRIDE,
                width: 16,
                height: 8,
                format: MTL_FORMAT_RGBA8_UNORM,
                generation: 3,
                guest_write,
                seq: 0,
            },
        ),
        None,
        "the resource owes exactly one frame going in"
    );
    assert!(state.pending_writebacks.has_gva(key));

    let backing = resolve_texture_backing(&mut state, &mut host, 1, 2, 0, 0)
        .expect("a linear RGBA8 texture resolves as a blit endpoint");
    assert!(
        matches!(backing, TextureBacking::Linear(_)),
        "this is the private-texture rung, not a surface"
    );
    assert!(
        !state.pending_writebacks.has_gva(key),
        "resolving a blit endpoint must land what the texture owed, not read around it"
    );
}

/// Overwrite an installed texture descriptor's `allocation_size` in place.
/// The `install_linear_*` helpers floor it at 0x1000, and a bounds test
/// needs an allocation sized to the image and nothing more.
fn set_installed_allocation_size(
    host: &mut FakeHost,
    state: &DeviceState,
    obj_ref: u32,
    alloc: u64,
) {
    let desc_gva = 0x200u64 + (obj_ref as u64) * 0x80;
    write_task_gva_arm64e(host, &state.tasks[1], desc_gva, &alloc.to_le_bytes());
}

/// The array-slice form of the corner-mask bounds defect: the selected
/// slice was charged a whole `row_stride * height` stride, so an allocation
/// the guest sized for exactly two padded slices was refused for trailing
/// padding that `texel_offset` never reaches.
#[test]
fn a_last_array_slice_is_not_charged_for_its_trailing_row_padding() {
    // 4x2 RGBA8: tight rows are 16 B, the guest pads to 24, so one slice
    // spans 48 B and the bytes read of a slice end at 24 + 16 = 40.
    const STRIDE: u32 = 24;
    const TIGHT: u64 = 4 * 4;
    const ONE_SLICE: u64 = STRIDE as u64 * 2;
    const READ: u64 = STRIDE as u64 + TIGHT;
    const EXACT: u64 = ONE_SLICE + READ;

    let (mut host, mut state) = blit_device();
    install_linear_rgba(&mut host, &mut state, 2, 2, 4, 2, STRIDE);
    set_installed_allocation_size(&mut host, &state, 2, EXACT);

    let backing = resolve_texture_backing(&mut state, &mut host, 1, 2, 0, 1)
        .expect("slice 1 fits an allocation sized for exactly two slices");
    let stride = match backing {
        TextureBacking::Linear(t) => {
            assert_eq!(t.slice_stride, ONE_SLICE, "slices stay a full stride apart");
            assert_eq!(t.slice_index, 1);
            // The last byte the reader can touch, and it is inside.
            assert_eq!(
                t.texel_offset(3, 1, 0),
                Some(ONE_SLICE + STRIDE as u64 + 12)
            );
            assert!(t.texel_offset(3, 1, 0).unwrap() + 4 <= EXACT);
            t.slice_stride
        }
        TextureBacking::Type11(_) => panic!("linear texture resolved as type-11"),
    };

    // The bound this replaced charged a second whole stride and refused this
    // allocation, so the regression is visible here rather than only on a
    // live guest.
    assert!(
        ONE_SLICE + stride > EXACT,
        "the stride form must overcount, or this case proves nothing"
    );
    set_installed_allocation_size(&mut host, &state, 2, EXACT - 1);
    match resolve_texture_backing(&mut state, &mut host, 1, 2, 0, 1) {
        Err(st) => assert_eq!(st, BlitStatus::Bounds),
        Ok(_) => panic!("one byte short of the read extent must still be refused"),
    }
}

#[test]
fn whole_surface_0x13e_single_level_copy() {
    let (mut host, mut state) = blit_device();
    // src handle=2 (4×2 RGBA, stride 16), dst handle=3
    install_linear_rgba(&mut host, &mut state, 2, 2, 4, 2, 16);
    install_linear_rgba(&mut host, &mut state, 3, 3, 4, 2, 16);
    let src_gva = 2u64 << RESOURCE_PAGE_SHIFT;
    let dst_gva = 3u64 << RESOURCE_PAGE_SHIFT;
    // Fill source with pattern (2 rows × 16 B).
    let mut pat = vec![0u8; 32];
    for (i, b) in pat.iter_mut().enumerate() {
        *b = i as u8;
    }
    write_task_gva_arm64e(&mut host, &state.tasks[1], src_gva, &pat);

    let mut cmd = copy_cmd(CopyKind::TextureToTextureSliceLevel, 2, 3);
    cmd.source_slice = 0;
    cmd.source_level = 0;
    cmd.destination_slice = 0;
    cmd.destination_level = 0;
    cmd.slice_count = 1;
    cmd.level_count = 1;
    assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);

    let mut back = vec![0u8; 32];
    assert!(gva_mem::read_task_gva(
        &host,
        &state.tasks[1],
        dst_gva,
        &mut back,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    // Only tight 4×4=16 B per row are defined; padding in stride may be zero.
    assert_eq!(&back[0..16], &pat[0..16]);
    assert_eq!(&back[16..32], &pat[16..32]);
}

/// A type-11 whole-surface `0x13e` moves every row, in order.
///
/// This arm stages the slice whole rather than a row at a time, because a
/// per-row call into the mapping rail re-pays that rail's per-*rect* costs —
/// settle, vouch, window revalidation, and a fresh import per guest page run —
/// once for every row. A driven Maps leg charged the old row loop 30.15 s of a
/// 30.28 s blit rail to move 14.6 MB.
///
/// The failure the whole-slice form can have that the row loop could not is a
/// stride one: the staging buffer is packed `row_bytes` apart while the surface
/// is `row_stride` apart, and the mapping rail is told both. Get that pair
/// backwards and rows land shifted or on top of each other. So the source rows
/// here are made distinguishable and the assertion is per row, not on the
/// buffer entire — a whole-buffer compare of a uniform fill would pass on a
/// copy that wrote row 0 twice.
#[test]
fn a_type11_whole_surface_copy_lands_every_row_in_order() {
    let (mut host, mut state) = blit_device();
    install_type11(&mut host, &mut state, 2, 20, 0x40);
    install_type11(&mut host, &mut state, 3, 21, 0x50);

    // 2×2 BGRA: two rows of 8 bytes, each row its own byte value.
    let src_pixels: [u8; 16] = [
        0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, // row 0
        0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, // row 1
    ];
    assert!(mapping_write::write_rect_raw(
        &mut state,
        &mut host,
        20,
        mapping_write::Rect {
            origin_x: 0,
            origin_y: 0,
            width: 2,
            height: 2
        },
        &src_pixels,
        8
    ));

    let mut cmd = copy_cmd(CopyKind::TextureToTextureSliceLevel, 2, 3);
    cmd.slice_count = 1;
    cmd.level_count = 1;
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::Ok,
        "a type-11 to type-11 whole-surface copy must execute"
    );

    let mut back = [0u8; 16];
    assert!(mapping_write::read_rect_raw(
        &mut state,
        &mut host,
        21,
        mapping_write::Rect {
            origin_x: 0,
            origin_y: 0,
            width: 2,
            height: 2
        },
        &mut back,
        8
    ));
    assert_eq!(&back[0..8], &src_pixels[0..8], "row 0 did not land");
    assert_eq!(&back[8..16], &src_pixels[8..16], "row 1 did not land");
}

/// Regression guard for the identity self-copy no-op: the guest issues
/// copyFromTexture:X toTexture:X with matching origin (observed live on
/// Ventura x86 media apps: src_ref==dst_ref, src_off==dst_off). This copies
/// bytes onto themselves — a no-op — and must return Ok with content
/// unchanged, NOT reject it as Overlap (which returned a spurious error to
/// the guest and dropped a copy it treats as complete).
#[test]
fn t2t_identity_self_copy_is_noop_ok() {
    let (mut host, mut state) = blit_device();
    // One 4×2 RGBA texture (stride 16), ref==handle==2.
    install_linear_rgba(&mut host, &mut state, 2, 2, 4, 2, 16);
    let gva = 2u64 << RESOURCE_PAGE_SHIFT;
    let mut pat = vec![0u8; 32];
    for (i, b) in pat.iter_mut().enumerate() {
        *b = (0xA0 + i) as u8;
    }
    write_task_gva_arm64e(&mut host, &state.tasks[1], gva, &pat);

    let mut cmd = copy_cmd(CopyKind::TextureToTexture, 2, 2); // same texture, same origin => identity
    cmd.source_size = Size {
        width: 4,
        height: 2,
        depth: 1,
    };
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::Ok,
        "identity self-copy must succeed as a no-op, not Overlap"
    );
    // Content byte-identical: the no-op touched nothing.
    let mut back = vec![0u8; 32];
    assert!(
        gva_mem::read_task_gva(&host, &state.tasks[1], gva, &mut back, PAGE_SHIFT_ARM64E).is_ok()
    );
    assert_eq!(back, pat, "identity self-copy left the bytes unchanged");
    // No overlap enrichment line was emitted (it's not a genuine overlap).
    assert!(
        note_t2t_overlap(1, 2, 2, 0, 0, 16, 16, 2, 1),
        "identity path must not have consumed the overlap dedup slot"
    );
}

/// Regression guard for the strided-column false positive: a self-copy of a
/// 1-wide column shifted N texels right (src rect x[0,1), dst rect x[2,3))
/// has strided per-row byte footprints that never collide, so it must
/// SUCCEED and actually move the bytes — the old byte-span overlap test
/// collapsed row_stride and dropped it as a phantom Overlap. Uses texel-
/// rectangle overlap (disjoint on x => no overlap).
#[test]
fn t2t_shifted_column_self_copy_moves_bytes() {
    let (mut host, mut state) = blit_device();
    // 4×4 RGBA, stride 16, ref==handle==2.
    install_linear_rgba(&mut host, &mut state, 2, 2, 4, 4, 16);
    let gva = 2u64 << RESOURCE_PAGE_SHIFT;
    // Distinct per-texel marker so a moved column is verifiable.
    let mut pat = vec![0u8; 64];
    for (i, b) in pat.iter_mut().enumerate() {
        *b = (0x10 + i) as u8;
    }
    write_task_gva_arm64e(&mut host, &state.tasks[1], gva, &pat);
    // Copy column x=0 (4 rows) to column x=2 within the same texture.
    let mut cmd = copy_cmd(CopyKind::TextureToTexture, 2, 2);
    cmd.destination_origin.x = 2;
    cmd.source_size = Size {
        width: 1,
        height: 4,
        depth: 1,
    };
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::Ok,
        "disjoint shifted column copy must succeed, not phantom-Overlap"
    );
    let mut back = vec![0u8; 64];
    assert!(
        gva_mem::read_task_gva(&host, &state.tasks[1], gva, &mut back, PAGE_SHIFT_ARM64E).is_ok()
    );
    // For each row r: dst column x=2 (bytes [r*16+8, +4)) now equals the
    // src column x=0 (bytes [r*16, +4)) as it was in the original pattern.
    for r in 0..4usize {
        let src_texel = &pat[r * 16..r * 16 + 4];
        let dst_texel = &back[r * 16 + 8..r * 16 + 12];
        assert_eq!(dst_texel, src_texel, "row {r} column x=2 holds moved src");
    }
    // No overlap enrichment (this is not a genuine overlap).
    assert!(
        note_t2t_overlap(9, 2, 2, 0, 8, 4, 16, 4, 1),
        "shifted-column path must not have consumed the overlap dedup slot"
    );
}

/// Regression guard: a GENUINELY overlapping self-copy (src rect x[0,2),
/// dst rect x[1,3) — overlap on x) is undefined in Metal and must still be
/// rejected as Overlap, with the enrichment line emitted for diagnosis.
#[test]
fn t2t_overlapping_self_copy_still_rejected() {
    let (mut host, mut state) = blit_device();
    // Unique ref (4) so the (task,src,dst) enrichment key is globally
    // distinct from the identity/shifted tests' probes.
    install_linear_rgba(&mut host, &mut state, 4, 4, 4, 4, 16);
    let mut cmd = copy_cmd(CopyKind::TextureToTexture, 4, 4);
    cmd.destination_origin.x = 1; // src x[0,2), dst x[1,3) overlap at x=1
    cmd.source_size = Size {
        width: 2,
        height: 4,
        depth: 1,
    };
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::Overlap,
        "genuinely overlapping self-copy must be rejected"
    );
    // The enrichment slot WAS consumed (a real overlap was logged).
    assert!(
        !note_t2t_overlap(1, 4, 4, 0, 4, 8, 16, 4, 1),
        "the reject path must have logged the overlap enrichment once"
    );
}

#[test]
fn whole_surface_0x13e_two_levels() {
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE, RESOURCE_PAGE_SHIFT,
        TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_DATA_OFFSET, TEXTURE_DESC_HEIGHT,
        TEXTURE_DESC_LEVEL_RECORDS, TEXTURE_DESC_MIPMAP_LEVEL_COUNT,
        TEXTURE_DESC_MIP_LEVEL_RECORD_LEN, TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE,
        TEXTURE_DESC_USED_SIZE, TEXTURE_DESC_WIDTH, TEXTURE_LEVEL_DEPTH, TEXTURE_LEVEL_HEIGHT,
        TEXTURE_LEVEL_OFFSET, TEXTURE_LEVEL_ROW_STRIDE, TEXTURE_LEVEL_SIZE, TEXTURE_LEVEL_WIDTH,
    };
    let (mut host, mut state) = blit_device();

    // Two textures, 2 mips: L0 4×2 stride16, L1 2×1 stride8.
    for (obj_ref, handle) in [(2u32, 2u32), (3u32, 3u32)] {
        let body = TEXTURE_DESC_BASE_LEN + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
        let mut desc = vec![0u8; body];
        st64(&mut desc[0..], 0x4000);
        st32(&mut desc[8..], handle);
        st16(&mut desc[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 2);
        st32(&mut desc[TEXTURE_DESC_DATA_OFFSET..], 0);
        st32(&mut desc[TEXTURE_DESC_USED_SIZE..], 32);
        st32(&mut desc[TEXTURE_DESC_ROW_STRIDE..], 16);
        st32(&mut desc[TEXTURE_DESC_WIDTH..], 4);
        st32(&mut desc[TEXTURE_DESC_HEIGHT..], 2);
        let rec = TEXTURE_DESC_LEVEL_RECORDS;
        st64(&mut desc[rec + TEXTURE_LEVEL_OFFSET..], 32);
        st64(&mut desc[rec + TEXTURE_LEVEL_SIZE..], 8);
        st64(&mut desc[rec + TEXTURE_LEVEL_ROW_STRIDE..], 8);
        st32(&mut desc[rec + TEXTURE_LEVEL_WIDTH..], 2);
        st32(&mut desc[rec + TEXTURE_LEVEL_HEIGHT..], 1);
        st32(&mut desc[rec + TEXTURE_LEVEL_DEPTH..], 1);
        let pf_off = TEXTURE_DESC_PIXEL_FORMAT + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
        st16(&mut desc[pf_off..], MTL_FORMAT_RGBA8_UNORM);
        let desc_gva = 0x200u64 + (obj_ref as u64) * 0x100;
        write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &desc);
        assert!(state.set_object_list(1, 0, 16));
        let off = list_object_entry_offset(obj_ref, 16).unwrap();
        let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_TEXTURE as u32) | ((body as u32) << 8);
        st32(&mut list_entry[0..], packed);
        list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);
    }

    // Seed L0 and L1 on source handle 2.
    let base = 2u64 << RESOURCE_PAGE_SHIFT;
    let l0 = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let l0_row1 = [
        17u8, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    ];
    let l1 = [0xaau8, 0xbb, 0xcc, 0xdd, 0x11, 0x22, 0x33, 0x44];
    write_task_gva_arm64e(&mut host, &state.tasks[1], base, &l0);
    write_task_gva_arm64e(&mut host, &state.tasks[1], base + 16, &l0_row1);
    write_task_gva_arm64e(&mut host, &state.tasks[1], base + 32, &l1);

    let mut cmd = copy_cmd(CopyKind::TextureToTextureSliceLevel, 2, 3);
    cmd.slice_count = 1;
    cmd.level_count = 2;
    assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);

    let dst = 3u64 << RESOURCE_PAGE_SHIFT;
    let mut back_l0 = [0u8; 16];
    let mut back_l1 = [0u8; 8];
    assert!(
        gva_mem::read_task_gva(&host, &state.tasks[1], dst, &mut back_l0, PAGE_SHIFT_ARM64E)
            .is_ok()
    );
    assert!(gva_mem::read_task_gva(
        &host,
        &state.tasks[1],
        dst + 32,
        &mut back_l1,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    assert_eq!(back_l0, l0);
    assert_eq!(back_l1, l1);
    let _ = RESOURCE_PAGE_SHIFT;
}

/// Install type-2 RGBA8 volume (single level, depth>1) at `handle<<14`.
fn install_linear_rgba_volume(
    host: &mut FakeHost,
    state: &mut DeviceState,
    obj_ref: u32,
    handle: u32,
    width: u32,
    height: u32,
    depth: u32,
    row_stride: u32,
) {
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE, RESOURCE_PAGE_SHIFT,
        TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_DATA_OFFSET, TEXTURE_DESC_DEPTH, TEXTURE_DESC_HEIGHT,
        TEXTURE_DESC_MIPMAP_LEVEL_COUNT, TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE,
        TEXTURE_DESC_USED_SIZE, TEXTURE_DESC_WIDTH,
    };
    let _ = RESOURCE_PAGE_SHIFT;
    let mut desc = vec![0u8; TEXTURE_DESC_BASE_LEN];
    let plane = (row_stride as u64) * (height as u64);
    let size = plane * (depth as u64);
    st64(&mut desc[0..], size.max(0x1000));
    st32(&mut desc[8..], handle);
    st16(&mut desc[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 1);
    st32(&mut desc[TEXTURE_DESC_DATA_OFFSET..], 0);
    st32(&mut desc[TEXTURE_DESC_USED_SIZE..], size as u32);
    st32(&mut desc[TEXTURE_DESC_ROW_STRIDE..], row_stride);
    st32(&mut desc[TEXTURE_DESC_WIDTH..], width);
    st32(&mut desc[TEXTURE_DESC_HEIGHT..], height);
    st32(&mut desc[TEXTURE_DESC_DEPTH..], depth);
    st16(
        &mut desc[TEXTURE_DESC_PIXEL_FORMAT..],
        MTL_FORMAT_RGBA8_UNORM,
    );
    let desc_gva = 0x200u64 + (obj_ref as u64) * 0x80;
    write_task_gva_arm64e(host, &state.tasks[1], desc_gva, &desc);
    assert!(state.set_object_list(1, 0, 16));
    let off = list_object_entry_offset(obj_ref, 16).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (OBJECT_TYPE_TEXTURE as u32) | ((TEXTURE_DESC_BASE_LEN as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(host, &state.tasks[1], off, &list_entry);
}

#[test]
fn whole_surface_0x13e_volume_depth_planes() {
    let (mut host, mut state) = blit_device();
    // 2×2×3 RGBA8, row_stride 8 → plane 16 B, volume 48 B.
    install_linear_rgba_volume(&mut host, &mut state, 2, 2, 2, 2, 3, 8);
    install_linear_rgba_volume(&mut host, &mut state, 3, 3, 2, 2, 3, 8);
    let src_gva = 2u64 << RESOURCE_PAGE_SHIFT;
    let dst_gva = 3u64 << RESOURCE_PAGE_SHIFT;
    let mut vol = vec![0u8; 48];
    for (i, b) in vol.iter_mut().enumerate() {
        *b = (i as u8).wrapping_add(1);
    }
    write_task_gva_arm64e(&mut host, &state.tasks[1], src_gva, &vol);

    let mut cmd = copy_cmd(CopyKind::TextureToTextureSliceLevel, 2, 3);
    cmd.source_slice = 0;
    cmd.destination_slice = 0;
    cmd.slice_count = 1;
    cmd.level_count = 1;
    assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);

    let mut back = vec![0u8; 48];
    assert!(gva_mem::read_task_gva(
        &host,
        &state.tasks[1],
        dst_gva,
        &mut back,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    assert_eq!(back, vol);
}

#[test]
fn whole_surface_0x13e_volume_rejects_multi_slice() {
    let (mut host, mut state) = blit_device();
    install_linear_rgba_volume(&mut host, &mut state, 2, 2, 2, 2, 2, 8);
    install_linear_rgba_volume(&mut host, &mut state, 3, 3, 2, 2, 2, 8);
    let mut cmd = copy_cmd(CopyKind::TextureToTextureSliceLevel, 2, 3);
    cmd.slice_count = 2; // Metal: 3D requires sliceCount==1
    cmd.level_count = 1;
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::Unsupported
    );
}

#[test]
fn whole_surface_0x13e_volume_rejects_nonzero_slice() {
    let (mut host, mut state) = blit_device();
    install_linear_rgba_volume(&mut host, &mut state, 2, 2, 2, 2, 2, 8);
    install_linear_rgba_volume(&mut host, &mut state, 3, 3, 2, 2, 2, 8);
    let mut cmd = copy_cmd(CopyKind::TextureToTextureSliceLevel, 2, 3); // Non-zero slice on 3D whole-surface is fail-closed (Metal forbids).
                                                                        // Status may be Bounds (slice packing) or Unsupported (3D rule).
    cmd.source_slice = 1;
    cmd.slice_count = 1;
    cmd.level_count = 1;
    let st = execute_blit(&mut state, &mut host, 1, &cmd);
    assert!(
        matches!(st, BlitStatus::Bounds | BlitStatus::Unsupported),
        "expected Bounds or Unsupported, got {st:?}"
    );
}

#[test]
fn blit_fence_update_then_wait() {
    use crate::model::FENCE_DOMAIN_BLIT;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut upd = Command::default();
    upd.kind = Kind::Fence;
    upd.opcode = wire_blit::OPCODE_UPDATE_FENCE;
    upd.fence = 7;
    assert_eq!(execute_blit_fence(&mut state, 1, &upd), BlitStatus::Ok);
    assert_eq!(state.fence_generation(1, FENCE_DOMAIN_BLIT, 7), Some(1));
    // Second update advances generation.
    assert_eq!(execute_blit_fence(&mut state, 1, &upd), BlitStatus::Ok);
    assert_eq!(state.fence_generation(1, FENCE_DOMAIN_BLIT, 7), Some(2));
    let mut wait = Command::default();
    wait.kind = Kind::Fence;
    wait.opcode = wire_blit::OPCODE_WAIT_FOR_FENCE;
    wait.fence = 7;
    assert_eq!(execute_blit_fence(&mut state, 1, &wait), BlitStatus::Ok);
}

#[test]
fn blit_fence_wait_pending_without_update() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut wait = Command::default();
    wait.kind = Kind::Fence;
    wait.opcode = wire_blit::OPCODE_WAIT_FOR_FENCE;
    wait.fence = 3;
    assert_eq!(
        execute_blit_fence(&mut state, 1, &wait),
        BlitStatus::FencePending
    );
    assert!(state.fence_generation(1, FENCE_DOMAIN_BLIT, 3).is_none());
}

#[test]
fn blit_fence_zero_ref_fails() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut upd = Command::default();
    upd.kind = Kind::Fence;
    upd.opcode = wire_blit::OPCODE_UPDATE_FENCE;
    upd.fence = 0;
    assert_eq!(
        execute_blit_fence(&mut state, 1, &upd),
        BlitStatus::MissingResource
    );
}

/// Regression guard for the pure blit-geometry helpers `clamp_extent`,
/// `texture_storage_bpp`, and the aspect routing of `copy_aspect_for_options`.
/// These feed copy bounds and row strides, so a silent break corrupts the
/// copied region: a bad clamp reads/writes out of bounds, a wrong bpp
/// miscomputes the row stride, and a wrong aspect flag copies the wrong
/// depth/stencil plane. `copy_bpp_for_options` tests already cover the bpp
/// result; this locks the extent/stride/aspect-flag contract directly.
#[test]
fn blit_geometry_helpers_clamp_bpp_and_aspect() {
    use crate::contract::pixel_format::{
        MTL_FORMAT_A8_UNORM, MTL_FORMAT_DEPTH32_FLOAT_STENCIL8, MTL_FORMAT_RGBA16_FLOAT,
    };
    use crate::runtime::decode::blit::{
        MTL_BLIT_OPTION_DEPTH_FROM_DEPTH_STENCIL, MTL_BLIT_OPTION_NONE,
        MTL_BLIT_OPTION_STENCIL_FROM_DEPTH_STENCIL,
    };

    // copy_extent: zero stays a no-op extent; in-range and exactly-max pass
    // through unchanged; over-max is `None`, which every caller turns into
    // `BlitStatus::Bounds` rather than into a smaller copy.
    assert_eq!(
        copy_extent("t", "w", 0, 100),
        Some(0),
        "zero is a Metal no-op extent, not a refusal"
    );
    assert_eq!(
        copy_extent("t", "w", 50, 100),
        Some(50),
        "in-range passes through"
    );
    assert_eq!(
        copy_extent("t", "w", 100, 100),
        Some(100),
        "exactly max passes through — the bound is inclusive"
    );
    assert_eq!(
        copy_extent("t", "w", 101, 100),
        None,
        "one past the edge refuses; it used to return 100 and copy less"
    );
    assert_eq!(copy_extent("t", "w", 150, 100), None, "and so does far past");

    // texture_storage_bpp: full-texel storage size per format; unknown fails.
    assert_eq!(texture_storage_bpp(MTL_FORMAT_BGRA8_UNORM), Ok(4));
    assert_eq!(texture_storage_bpp(MTL_FORMAT_A8_UNORM), Ok(1));
    assert_eq!(texture_storage_bpp(MTL_FORMAT_RGBA16_FLOAT), Ok(8));
    assert_eq!(
        texture_storage_bpp(0xFFFF),
        Err(BlitStatus::Unsupported),
        "unknown format must fail visibly, not invent a stride",
    );

    // copy_aspect_for_options: option bit -> (aspect, bpp).
    let with_opts = |opts: u32| {
        let mut cmd = Command::default();
        cmd.has_options = true;
        cmd.options = opts;
        cmd
    };
    // No option on a color format -> full aspect, no plane routing.
    assert_eq!(
        copy_aspect_for_options(MTL_FORMAT_BGRA8_UNORM, &with_opts(MTL_BLIT_OPTION_NONE)),
        Ok((BlitAspect::Full, 4)),
    );
    // Depth option on a depth-stencil format -> depth plane (4 B), no stencil.
    assert_eq!(
        copy_aspect_for_options(
            MTL_FORMAT_DEPTH32_FLOAT_STENCIL8,
            &with_opts(MTL_BLIT_OPTION_DEPTH_FROM_DEPTH_STENCIL),
        ),
        Ok((BlitAspect::Depth, 4)),
    );
    // Stencil option -> stencil plane (1 B), no depth.
    assert_eq!(
        copy_aspect_for_options(
            MTL_FORMAT_DEPTH32_FLOAT_STENCIL8,
            &with_opts(MTL_BLIT_OPTION_STENCIL_FROM_DEPTH_STENCIL),
        ),
        Ok((BlitAspect::Stencil, 1)),
    );
    // Unknown option bit -> visible failure (no invented aspect).
    assert_eq!(
        copy_aspect_for_options(MTL_FORMAT_DEPTH32_FLOAT_STENCIL8, &with_opts(1 << 8)),
        Err(BlitStatus::Unsupported),
    );
}

/// Regression guard: the `tex_wrong_type` enrichment is deduped per
/// `(task, ref, object_type)` — a per-draw non-texture bind must not flood
/// the always-on sink — while distinct refs/types each report once so a
/// buffer-bound-as-texture (decode bug) stays diagnosable.
#[test]
fn tex_wrong_type_enrichment_dedups_per_ref_and_type() {
    reset_tex_wrong_type_dedup_for_test();
    // First sighting of a (task, ref, type) emits; repeats are deduped.
    assert!(note_tex_wrong_type(7, 0x40, OBJECT_TYPE_BUFFER, 0, 0));
    for _ in 0..20 {
        assert!(!note_tex_wrong_type(7, 0x40, OBJECT_TYPE_BUFFER, 0, 0));
    }
    // A different ref is a distinct failure -> reports once.
    assert!(note_tex_wrong_type(7, 0x41, OBJECT_TYPE_BUFFER, 0, 0));
    // Same ref but a different actual object_type also reports (the type is
    // the diagnostic field, so a type change must not be masked).
    assert!(note_tex_wrong_type(
        7,
        0x40,
        crate::runtime::decode::resource::OBJECT_TYPE_FUNCTION,
        0,
        0
    ));
    assert!(!note_tex_wrong_type(
        7,
        0x40,
        crate::runtime::decode::resource::OBJECT_TYPE_FUNCTION,
        0,
        0
    ));
}

/// Regression guard: the `t2t_overlap` enrichment dedups per
/// `(task, src_ref, dst_ref)` — a self-overlapping copy re-issued every
/// frame must not flood — while a distinct src/dst pair reports once so a
/// genuine drop stays diagnosable.
#[test]
fn t2t_overlap_enrichment_dedups_per_pair() {
    // Unique task namespace (3) so the process-global dedup set never
    // collides with other tests; the set starts empty so first-insert is
    // deterministic without a reset.
    assert!(note_t2t_overlap(3, 0x10, 0x10, 0, 4096, 256, 1024, 8, 1));
    for _ in 0..20 {
        assert!(!note_t2t_overlap(3, 0x10, 0x10, 0, 4096, 256, 1024, 8, 1));
    }
    // A distinct destination ref is a distinct failure -> reports once.
    assert!(note_t2t_overlap(3, 0x10, 0x11, 0, 4096, 256, 1024, 8, 1));
}

/// The precondition the `repack_storage_assumed` enrichment exists for.
///
/// The texture-to-texture format check deliberately admits a zero on either
/// side, so a format-less texture can pair with a combined depth/stencil one
/// and reach the aspect repack. `bytes_per_pixel` cannot answer for a zero,
/// which is what makes the repack fall back to the aspect's own width. If a
/// zero ever became derivable this assertion fails, and the enrichment
/// beside it would be measuring nothing.
#[test]
fn a_zero_pixel_format_has_no_derivable_storage_width() {
    assert!(texture_storage_bpp(0).is_err());
    // A real format does answer, so the refusal above is about the zero and
    // not about the helper being broken for everything.
    assert_eq!(texture_storage_bpp(MTL_FORMAT_BGRA8_UNORM), Ok(4));
}

/// Regression guard: the `repack_storage_assumed` enrichment dedups per
/// `(task, side, format)` — a repack re-issued every frame must not flood —
/// while each side and each distinct format reports once, so a source and a
/// destination assuming their width are two findings rather than one.
#[test]
fn repack_storage_assumed_enrichment_dedups_per_side_and_format() {
    // Unique task namespace (7) so the process-global dedup set never
    // collides with other tests; the set starts empty so first-insert is
    // deterministic without a reset.
    assert!(note_repack_storage_assumed(7, "src", 0, 4));
    for _ in 0..20 {
        assert!(!note_repack_storage_assumed(7, "src", 0, 4));
    }
    // The other side of the same copy is a distinct finding.
    assert!(note_repack_storage_assumed(7, "dst", 0, 4));
    // So is a different format that could not be derived.
    assert!(note_repack_storage_assumed(7, "src", 0x99, 4));
}

/// Regression guard: the `copy_region_io` enrichment dedups per
/// `(task, gva_page, is_write)` — a strided multi-row failure into one page
/// must not flood — while read vs write and distinct pages each report once.
#[test]
fn copy_region_io_enrichment_dedups_per_page_and_direction() {
    // Unique task namespace (2) + page base so the process-global dedup set
    // never collides with other tests; empty-set start makes first-insert
    // deterministic without a reset.
    let shift = PAGE_SHIFT_ARM64E;
    let page = 0x5000u64 << shift;
    // Rows 0..N inside the same destination page collapse to one line.
    assert!(note_copy_region_io(2, true, page, 0, 0, 256, shift));
    for y in 1..10u64 {
        assert!(!note_copy_region_io(
            2,
            true,
            page + y * 256,
            y,
            0,
            256,
            shift
        ));
    }
    // A read at the same page is a distinct direction -> reports once.
    assert!(note_copy_region_io(2, false, page, 0, 0, 256, shift));
    // A different page reports once.
    assert!(note_copy_region_io(
        2,
        true,
        page + (1u64 << shift),
        0,
        0,
        256,
        shift
    ));
}

/// The copy path and the draw/sample path follow a type-8 view chain exactly
/// as deep as each other.
///
/// Two arms consume this one wire form: `resolve_texture_backing_depth` here,
/// and `runtime::draw::resolve_texture_view_reasoned` for sampling. They used
/// to disagree — this one stopped after five hops on a number its own comment
/// called "not a contract limit", the other after the contract's eight — so a
/// guest chain of six sampled correctly and had its copy dropped as
/// `tex_view_depth_cap`. Nothing about a copy justifies seeing a shallower
/// chain than a sample does.
///
/// So the assertion is a comparison, not a threshold: both arms are driven at
/// the deepest chain the contract admits and both must take it. A test that
/// only pinned this arm at eight would pass again the moment the other one
/// moved, which is the failure that produced the divergence in the first place.
///
/// The chain runs `MAX_TEXTURE_VIEW_CHAIN` views down to a type-11 base, each
/// view a plain identity hop, so the only thing that can refuse it is depth.
#[test]
fn a_copy_follows_a_view_chain_as_deep_as_a_sample_does() {
    use crate::runtime::draw::{resolve_texture_view_reasoned, MAX_TEXTURE_VIEW_CHAIN};

    let (mut host, mut state) = blit_device();
    install_buffer(&mut host, &mut state, 1, 1, 256);
    let mapping_id = 9u32;
    install_type11(&mut host, &mut state, 3, mapping_id, 0x20);

    // Views live at refs `outermost` down to `base_view`, each viewing the next
    // lower ref; the lowest views the type-11 at ref 3. The object list holds
    // 16 entries, so the deepest legal chain has to fit under that.
    let base_view = 4u32;
    let outermost = base_view + MAX_TEXTURE_VIEW_CHAIN as u32 - 1;
    assert!(outermost < 16, "the chain must fit the test object list");
    for view_ref in base_view..=outermost {
        let target = if view_ref == base_view {
            3
        } else {
            view_ref - 1
        };
        install_type8_view(
            &mut host,
            &mut state,
            view_ref,
            target,
            MTL_FORMAT_BGRA8_UNORM,
            0,
            None,
        );
    }

    // The sample arm resolves the whole chain to the non-view base.
    let resolved = resolve_texture_view_reasoned(&state, &host, 1, outermost)
        .expect("the sample arm follows the contract's deepest chain");
    assert_eq!(
        resolved.base_texture_ref, 3,
        "the sample arm must land on the type-11 base, not stop inside the chain"
    );

    // The copy arm must reach the same base, and land real pixels through it.
    let pat = [0xaau8, 0xbb, 0xcc, 0xdd, 0x11, 0x22, 0x33, 0x44];
    let src_gva = 1u64 << RESOURCE_PAGE_SHIFT;
    write_task_gva_arm64e(&mut host, &state.tasks[1], src_gva, &pat);
    let mut cmd = copy_cmd(CopyKind::BufferToTexture, 1, outermost);
    cmd.source_offset = 0;
    cmd.source_bytes_per_row = 8;
    cmd.source_size = Size {
        width: 2,
        height: 1,
        depth: 1,
    };
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::Ok,
        "the copy arm refused a chain the sample arm just followed"
    );
    let mut back = [0u8; 8];
    assert!(mapping_write::read_rect_raw(
        &mut state,
        &mut host,
        mapping_id,
        mapping_write::Rect {
            origin_x: 0,
            origin_y: 0,
            width: 2,
            height: 1
        },
        &mut back,
        8
    ));
    assert_eq!(
        back, pat,
        "the copy resolved to a base, but not the one holding the pixels"
    );
}

/// ...and both refuse the same chain one hop past the contract.
///
/// The companion to the test above, and it is not redundant with it. Agreement
/// has two edges, and pinning only the accepting one invites the divergence
/// back from the other side: dropping this arm's bound entirely would make the
/// deep-chain test pass while a chain the sample arm refuses became a copy this
/// one silently followed. A guest chain is guest-built and may be cyclic, so
/// "follows further than the contract" is not a generosity, it is a walk with
/// no stop condition the other arm shares.
///
/// One hop past `MAX_TEXTURE_VIEW_CHAIN`, both arms must decline.
#[test]
fn a_copy_refuses_a_view_chain_the_sample_arm_also_refuses() {
    use crate::runtime::draw::{resolve_texture_view_reasoned, MAX_TEXTURE_VIEW_CHAIN};

    let (mut host, mut state) = blit_device();
    install_buffer(&mut host, &mut state, 1, 1, 256);
    install_type11(&mut host, &mut state, 3, 9, 0x20);

    let base_view = 4u32;
    let outermost = base_view + MAX_TEXTURE_VIEW_CHAIN as u32; // one view too many
    assert!(outermost < 16, "the chain must fit the test object list");
    for view_ref in base_view..=outermost {
        let target = if view_ref == base_view {
            3
        } else {
            view_ref - 1
        };
        install_type8_view(
            &mut host,
            &mut state,
            view_ref,
            target,
            MTL_FORMAT_BGRA8_UNORM,
            0,
            None,
        );
    }

    assert!(
        resolve_texture_view_reasoned(&state, &host, 1, outermost).is_err(),
        "the sample arm must refuse a chain past the contract depth"
    );

    let mut cmd = copy_cmd(CopyKind::BufferToTexture, 1, outermost);
    cmd.source_offset = 0;
    cmd.source_bytes_per_row = 8;
    cmd.source_size = Size {
        width: 2,
        height: 1,
        depth: 1,
    };
    assert_eq!(
        execute_blit(&mut state, &mut host, 1, &cmd),
        BlitStatus::Unsupported,
        "the copy arm followed a chain the sample arm refused"
    );
}

/// The GPU whole-plane arm's cheap half, which decides whether resolving the
/// destination — and paying its debt — is worth doing at all.
///
/// The interesting refusal is `SelfCopy`. Resolving the destination is what pays
/// its writeback debt, so a command whose two endpoints are one reference would
/// pay away the very resident this arm was about to copy *from* and then find
/// nothing there. It is not a Metal restriction; it is a consequence of the order
/// this arm has to do things in.
#[test]
fn the_gpu_whole_plane_arm_refuses_before_it_resolves_anything() {
    use GpuPlaneRefusal::*;

    assert_eq!(gpu_whole_plane_admissible(1, 1, 4, 5, true), Ok(()));
    assert_eq!(
        gpu_whole_plane_admissible(2, 1, 4, 5, true),
        Err(MultiLevel),
        "the arm copies one plane, not a mip chain"
    );
    assert_eq!(
        gpu_whole_plane_admissible(1, 3, 4, 5, true),
        Err(MultiLevel),
        "nor an array slice run"
    );
    assert_eq!(
        gpu_whole_plane_admissible(1, 1, 4, 4, true),
        Err(SelfCopy),
        "resolving the destination would pay away the source's own resident"
    );
    assert_eq!(
        gpu_whole_plane_admissible(1, 1, 4, 5, false),
        Err(SrcNotResident),
        "with nothing owed the source's guest pages already hold its content"
    );
    assert_eq!(
        gpu_whole_plane_admissible(1, 1, 4, 4, false),
        Err(SelfCopy),
        "the cheapest refusal that applies is the one reported, so the counters \
         partition rather than overlap"
    );
}

/// The GPU whole-plane arm's destination half, and specifically the plane check.
///
/// `write_bgra8_from_resident_gpu` resolves the plane itself, from the mapping's
/// declaration, and takes no plane index. A type-5 view carries one on the wire
/// and can therefore name a plane at a `surface_offset` that scan does not reach.
/// Landing a frame there is silent at every layer — the pixels appear in the next
/// plane of the same IOSurface — so the disagreement has to refuse before the
/// copy, not be detected after it.
#[test]
fn the_gpu_whole_plane_arm_refuses_a_plane_the_rail_would_not_write() {
    use GpuPlaneRefusal::*;

    let dst = GpuPlane {
        width: 64,
        height: 32,
        surface_offset: 0,
        row_stride: 256,
        pixel_format: MTL_FORMAT_BGRA8_UNORM,
    };
    let window = GpuMappingWindow {
        surface_offset: 0,
        row_stride: 256,
        pixel_format: MTL_FORMAT_BGRA8_UNORM,
    };
    let src = GpuResidentSource {
        width: 64,
        height: 32,
        pixel_format: MTL_FORMAT_BGRA8_UNORM,
    };

    assert_eq!(
        gpu_whole_plane_destination(Some(dst), Some(window), src),
        Ok(())
    );
    assert_eq!(
        gpu_whole_plane_destination(None, Some(window), src),
        Err(DstNotType11),
        "a linear allocation has no mapping for the rail to name"
    );
    assert_eq!(
        gpu_whole_plane_destination(Some(dst), None, src),
        Err(DstWindowUnresolved),
        "a mapping that declines the extent has no window to write"
    );

    // The type-5 plane hazard: the guest's descriptor names the second plane of a
    // biplanar surface, the mapping's own geometry scan resolves the first.
    let second_plane = GpuPlane {
        surface_offset: 0x8000,
        ..dst
    };
    assert_eq!(
        gpu_whole_plane_destination(Some(second_plane), Some(window), src),
        Err(PlaneOffset),
        "a plane the rail would not write must never be written"
    );
    assert_eq!(
        gpu_whole_plane_destination(
            Some(dst),
            Some(GpuMappingWindow {
                row_stride: 512,
                ..window
            }),
            src
        ),
        Err(PlaneOffset),
        "and neither must a plane it would write at a different pitch"
    );

    assert_eq!(
        gpu_whole_plane_destination(
            Some(dst),
            Some(window),
            GpuResidentSource { height: 16, ..src }
        ),
        Err(GeometryDiffers),
        "a full-plane copy is not a resize"
    );
    assert_eq!(
        gpu_whole_plane_destination(
            Some(dst),
            Some(window),
            GpuResidentSource {
                pixel_format: MTL_FORMAT_RGBA8_UNORM,
                ..src
            }
        ),
        Err(FormatDiffers),
        "a copy converts nothing, so the resident must already be the destination's texel"
    );
    assert_eq!(
        gpu_whole_plane_destination(
            Some(dst),
            Some(GpuMappingWindow {
                pixel_format: MTL_FORMAT_RGBA8_UNORM,
                ..window
            }),
            src
        ),
        Err(FormatDiffers),
        "including the format the guest will read the landed bytes back as"
    );
}

/// The transfer function is not part of a stored texel, so it cannot decide a
/// copy that converts nothing.
///
/// This is the shape a driven macos-13 Maps leg actually presents, and it was
/// **1 609 of 1 609** records — the whole arm. The triple reads
/// `src=81 dst=81 mapping=80`: the guest declares its render target
/// `BGRA8Unorm_sRGB`, both endpoints of the copy agree with each other, and the
/// IOSurface mapping declares plain `BGRA8Unorm` for the very same four stored
/// bytes. Format equality calls that a disagreement forever, so the arm refused
/// every record it was written for while every real precondition passed.
///
/// `store_texel_order` is the fold, and its own doc states the rule: the sRGB
/// qualifier says how a sampler interprets bytes, not how they are stored.
/// What must still separate — channel order and texel width — survives it, which
/// is what the assertions below pin.
#[test]
fn the_gpu_whole_plane_arm_compares_stored_texels_and_not_transfer_functions() {
    use crate::contract::pixel_format::{
        MTL_FORMAT_BGRA8_UNORM_SRGB, MTL_FORMAT_R32_FLOAT, MTL_FORMAT_RGBA8_UNORM_SRGB,
    };
    use GpuPlaneRefusal::*;

    let dst = GpuPlane {
        width: 1024,
        height: 768,
        surface_offset: 0,
        row_stride: 4096,
        pixel_format: MTL_FORMAT_BGRA8_UNORM_SRGB,
    };
    let window = GpuMappingWindow {
        surface_offset: 0,
        row_stride: 4096,
        pixel_format: MTL_FORMAT_BGRA8_UNORM,
    };
    let src = GpuResidentSource {
        width: 1024,
        height: 768,
        pixel_format: MTL_FORMAT_BGRA8_UNORM_SRGB,
    };

    assert_eq!(
        gpu_whole_plane_destination(Some(dst), Some(window), src),
        Ok(()),
        "81/81/80 is one stored texel three times, which is what a copy moves"
    );

    // The fold is onto the sibling, not onto "any four-byte colour": channel
    // order still decides, from either side.
    assert_eq!(
        gpu_whole_plane_destination(
            Some(dst),
            Some(GpuMappingWindow {
                pixel_format: MTL_FORMAT_RGBA8_UNORM_SRGB,
                ..window
            }),
            src
        ),
        Err(FormatDiffers),
        "BGRA and RGBA are the same width and the same transfer function, and \
         still not the same bytes"
    );
    assert_eq!(
        gpu_whole_plane_destination(
            Some(GpuPlane {
                pixel_format: MTL_FORMAT_RGBA8_UNORM,
                ..dst
            }),
            Some(GpuMappingWindow {
                pixel_format: MTL_FORMAT_RGBA8_UNORM,
                ..window
            }),
            src
        ),
        Err(FormatDiffers),
        "and the resident's own order is compared against the folded destination"
    );

    // A format with no byte-copy layout is one the copy would have to convert,
    // which this arm does not do. It must refuse rather than fold to `None ==
    // None` and read as agreement.
    let unlayed = GpuPlane {
        pixel_format: MTL_FORMAT_R32_FLOAT,
        ..dst
    };
    assert_eq!(
        gpu_whole_plane_destination(
            Some(unlayed),
            Some(GpuMappingWindow {
                pixel_format: MTL_FORMAT_R32_FLOAT,
                ..window
            }),
            GpuResidentSource {
                pixel_format: MTL_FORMAT_R32_FLOAT,
                ..src
            }
        ),
        Err(FormatDiffers),
        "three agreeing formats with no stored-texel layout are still not a byte copy"
    );
}
