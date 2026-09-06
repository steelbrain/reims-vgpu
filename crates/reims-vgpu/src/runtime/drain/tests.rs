use super::*;

use crate::model::{PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86, PAGE_SIZE_ARM64E};

/// `process_child_packet` as a caller holding the packet at its ring position
/// calls it: with no submission read ahead of time, so the exec arm reads its
/// own.
///
/// Shadows the glob-imported one on purpose. Every test in this file is about
/// what an arm does with a packet's bytes, and none is about where the command
/// buffers came from — spelling `None` seventy times would say the opposite.
fn process_child_packet<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    channel_id: u32,
    packet: &Packet,
) -> ChildPacketDisposition {
    super::process_child_packet(state, host, channel_id, packet, None)
}
use crate::protocol::fifo::{
    DeviceInfoForm, CHILD_SHARED_STATE_INDEX, CHILD_SHARED_STATE_LEN, CHILD_SHARED_STATE_PFN,
    DEFINE_TASK_LEN, SET_OBJECT_LIST_LEN,
};

/// The two device-info request forms these tests build, named once so a test
/// cannot pin an offset the decoder does not read.
const TAHOE: DeviceInfoForm = DeviceInfoForm::WithKeyLimit;
const MONTEREY: DeviceInfoForm = DeviceInfoForm::WithoutKeyLimit;

/// I2's carve-out, asserted rather than trusted: a partial packet is the
/// normal state of a ring whose producer is mid-write, so it must not reach
/// the always-on log. A bad size or a desync must.
///
/// Without this, the flood is silent — a healthy boot would write one line
/// per drain iteration and the sink's own detector would be the only thing
/// that noticed.
#[test]
fn a_partial_packet_is_control_flow_and_never_a_logged_fault() {
    assert_eq!(PacketError::ShortHeader.fault(), None);
    assert_eq!(PacketError::Incomplete.fault(), None);
    assert_eq!(PacketError::BadSize.fault(), Some(PacketFault::BadSize));
}

#[test]
fn present_scanout_action_follows_window_active() {
    use crate::runtime::host::{FakeHost, HostActionKind};

    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.present.frame_mapping = 7;
    state.present.frame_generation = 42;

    // No host window (arm64 MMIO / REIMS_VGPU_WINDOW=0): the QEMU console is the
    // display — the present MUST enqueue the CPU ScanoutUpdate and request
    // the action boundary, or the console freezes at its last paint.
    state.present.window_active = false;
    enqueue_present_scanout(&mut state, &mut host, 1440, 1080);
    let scan: Vec<_> = host
        .actions
        .iter()
        .filter(|a| a.kind == HostActionKind::ScanoutUpdate)
        .collect();
    assert_eq!(scan.len(), 1, "windowless present must paint the console");
    assert_eq!(scan[0].a0, 7);
    assert_eq!(scan[0].a1, 1440);
    assert_eq!(scan[0].a2, 1080);
    assert_eq!(scan[0].a3, 42);
    assert_eq!(state.present.unpainted_presents, 1);
    assert!(state.pending.host_action_yield);

    // Live host window: the drain publishes + self-acks; no CPU paint
    // action (QEMU runs -display none, the surface is painted for nobody).
    host.actions.clear();
    state.present.window_active = true;
    enqueue_present_scanout(&mut state, &mut host, 1440, 1080);
    assert!(
        !host
            .actions
            .iter()
            .any(|a| a.kind == HostActionKind::ScanoutUpdate),
        "window path must not produce a QEMU paint action"
    );
}

/// The mapping the host window will show for the present just accepted.
///
/// These tests used to observe the present through the `ScanoutUpdate`
/// action's `a0`. No CPU paint action is produced per present any more, so
/// the observable moves to the retain in `state.present` — which is what the
/// window reads. The gate mirrors the `paint_mid` selection that used to pick
/// the action's mapping, so "the mapping that would have been painted" and
/// "the mapping the window shows" stay the same assertion.
fn presented_mapping(state: &DeviceState) -> Option<u32> {
    let p = &state.present;
    (p.frame_valid && p.frame_mapping != 0 && !p.frame_bgra.is_empty()).then_some(p.frame_mapping)
}

/// These selection tests run the windowless (QEMU-console) configuration,
/// where every accepted present enqueues the CPU `ScanoutUpdate` —
/// coalesced latest-wins, so however many presents a test drives, at most
/// ONE may be pending. Presence/absence per presentation path is locked by
/// `present_scanout_action_follows_window_active`; this tripwire catches a
/// re-introduced per-present backlog (the dual-mid half-frame thrash
/// class).
fn assert_coalesced_paint_action(host: &crate::runtime::host::FakeHost, ctx: &str) {
    assert!(
        host.action_count(HostActionKind::ScanoutUpdate) <= 1,
        "{ctx}: pending CPU ScanoutUpdate paints must coalesce to at most one"
    );
}

#[test]
fn exec_summary_names_the_packet_counters_and_lock_hold() {
    let result = crate::runtime::exec::ExecResult {
        task_id: 3,
        streams_loaded: 1,
        buffer_unbinds: 2,
        texture_unbinds: 3,
        sampler_unbinds: 4,
        render_attachment_resolves: 1,
        render_guest_stores: 1,
        total_us: 98,
        ..Default::default()
    };
    let line = exec_summary(1, &result, 52);
    for field in [
        "rt_resolves=1",
        "guest_stores=1",
        "render_unbinds=2/3/4",
        "total_us=98",
    ] {
        assert!(line.contains(field), "missing {field}: {line}");
    }
}

#[test]
fn sync_exec_stall_proxy_fires_at_watchdog_scale_only() {
    assert!(!sync_exec_stalled(SYNC_EXEC_STALL_US - 1));
    assert!(sync_exec_stalled(SYNC_EXEC_STALL_US));
    assert!(sync_exec_stalled(3_406_929));
}
use crate::runtime::host::{FakeHost, HostActionKind};

/// A display-present packet naming `mapping`.
///
/// The surface id goes at the offset the emitting command's trailer puts it,
/// read from `reims_vgpu_protocol::present` — the same table the decoder uses,
/// so a test cannot pin an offset the product code does not read. The payload is the
/// command's own trailer length and nothing else, which is what the guest sends:
/// `kb/pvg-display-contract.md` §8.1 measured every op6 payload as trailer-only.
///
/// Every present test built this same eight-field `Packet` by hand; only the
/// opcode and the named mapping ever differed. A test that varies the payload
/// length on purpose builds its own rather than calling this.
/// The overlong alarm, driven the way the present arm drives it: decode the
/// trailer first, then hand the decode to the alarm. The alarm no longer works
/// out the trailer width itself — `reims_vgpu_protocol::present` owns that — so
/// a test that skipped the decode would be testing a call production cannot
/// make.
fn note(state: &mut DeviceState, channel_id: u32, packet: &Packet) {
    let form = PresentForm::of(WireChannel::Child, packet.opcode).expect("a present opcode");
    let decoded = present_trailer(form, &packet.payload).expect("long enough for its trailer");
    note_display_txn_payload(state, channel_id, packet, &decoded);
}

fn present_packet(opcode: u16, mapping: u32) -> Packet {
    let form = PresentForm::of(WireChannel::Child, opcode).expect("a present opcode");
    let len = form.trailer_len();
    let mut payload = vec![0u8; len];
    let off = form.target_offset();
    payload[off..off + 4].copy_from_slice(&mapping.to_le_bytes());
    Packet {
        opcode,
        stamp_waits: Vec::new(),
        total_size: PACKET_HEADER_LEN + len as u32,
        completion_stamp: 0,
        payload,
        next_head: 0,
    }
}

fn packet_bytes(opcode: u16, stamp_value: u32, payload: &[u8]) -> Vec<u8> {
    let total = PACKET_HEADER_LEN as usize + payload.len();
    let mut v = vec![0u8; total];
    v[0..2].copy_from_slice(&opcode.to_le_bytes());
    v[2..4].copy_from_slice(&0u16.to_le_bytes());
    v[4..8].copy_from_slice(&(total as u32).to_le_bytes());
    v[8..12].copy_from_slice(&stamp_value.to_le_bytes());
    v[12..].copy_from_slice(payload);
    v
}

#[test]
fn decode_basic_packet() {
    let p = packet_bytes(ROOT_OP_DEFINE_FIFO, 7, &1u32.to_le_bytes());
    let dec = decode_packet(&p, 0, p.len() as u32, RING).unwrap();
    assert_eq!(dec.opcode, ROOT_OP_DEFINE_FIFO);
    assert_eq!(dec.completion_stamp, 7);
    assert_eq!(dec.next_head, p.len() as u32);
}

/// A ring capacity comfortably larger than any packet these tests build, so a
/// `BadSize` in one of them is about the packet and never about the ring.
const RING: u32 = 4096;

/// A packet the producer has not finished publishing is control flow, and a
/// size the ring could never hold is a fault. The two must not answer alike.
///
/// `packet_snapshot_len` deliberately snaps only the header when the packet is
/// still being written, so measuring the declared `total_size` against the
/// *snapshot* said "bad size" for a perfectly well-formed packet whose producer
/// was mid-write — a `packet_bad_size` line on the always-on channel for a
/// healthy guest, and `Incomplete` unreachable behind it. The ring's capacity is
/// what bounds a sane size, so that is what it is measured against.
#[test]
fn a_packet_still_being_written_is_incomplete_and_an_impossible_one_is_a_fault() {
    let full = packet_bytes(ROOT_OP_DEFINE_FIFO, 7, &1u32.to_le_bytes());

    // What the drain loop actually holds for a mid-write packet: the header
    // alone, because that is all `packet_snapshot_len` lets it read.
    let header_only = &full[..PACKET_HEADER_LEN as usize];
    let published = PACKET_HEADER_LEN; // the producer stopped after the header
    assert_eq!(
        packet_snapshot_len(header_only, published, RING),
        PACKET_HEADER_LEN,
        "an unpublished packet may only be snapped as far as its header"
    );
    let err = decode_packet(header_only, 0, published, RING).unwrap_err();
    assert_eq!(err, PacketError::Incomplete);
    assert_eq!(
        err.fault(),
        None,
        "a producer mid-write must not reach the always-on failure channel"
    );

    // A declared size the ring itself could never hold is the guest's error,
    // and still reads as one.
    let mut impossible = full.clone();
    impossible[PACKET_TOTAL_SIZE..PACKET_TOTAL_SIZE + 4].copy_from_slice(&(RING + 1).to_le_bytes());
    assert_eq!(
        packet_snapshot_len(&impossible, RING, RING),
        PACKET_HEADER_LEN,
        "a size past the ring is never snapped at face value"
    );
    let err = decode_packet(&impossible, 0, RING, RING).unwrap_err();
    assert_eq!(err, PacketError::BadSize);
    assert_eq!(err.fault(), Some(PacketFault::BadSize));

    // And the whole packet, published, still decodes.
    assert_eq!(
        packet_snapshot_len(&full, full.len() as u32, RING),
        full.len() as u32
    );
    assert!(decode_packet(&full, 0, full.len() as u32, RING).is_ok());
}

/// Both rings decide how much to snapshot with the same function.
///
/// They used to decide it inline, and the two copies had already parted — the
/// root ring's carried an extra arm its own caller made unreachable. A
/// divergence here is not visible in a boot: both spellings return the same
/// length for every packet a healthy guest writes, and differ only on the
/// malformed ones nothing produces.
#[test]
fn the_snapshot_rule_reads_the_same_for_both_rings() {
    let full = packet_bytes(ROOT_OP_DEFINE_FIFO, 3, &[0u8; 16]);
    let total = full.len() as u32;
    for available in [0, PACKET_HEADER_LEN, total - 1, total, total + 8] {
        for capacity in [PACKET_HEADER_LEN, total - 1, total, RING] {
            let want = if total <= capacity && available >= total {
                total
            } else {
                PACKET_HEADER_LEN
            };
            assert_eq!(
                packet_snapshot_len(&full, available, capacity),
                want,
                "available={available} capacity={capacity}"
            );
        }
    }
}

#[test]
fn display_descriptor_advertises_four_modes_incl_4k() {
    let mut host = FakeHost::new();
    let gpa = 0x7a000000u64;
    host.map_range(gpa, PAGE_SIZE_ARM64E as usize, 0);
    fill_display_descriptor(&mut host, gpa, 0, PAGE_SIZE_ARM64E);
    let mut count = [0u8; 2];
    host.read_gpa(gpa + DISPLAY_DESC_TIMING_COUNT, &mut count)
        .unwrap();
    assert_eq!(u16::from_le_bytes(count), 4);
    let read16 = |host: &mut FakeHost, off: u64| {
        let mut b = [0u8; 2];
        host.read_gpa(gpa + off, &mut b).unwrap();
        u16::from_le_bytes(b)
    };
    // Element 0 (native/preferred) stays 1920×1080; 4K is appended last so
    // boot resolution is unchanged. Stride 0x10 from base 0x210.
    assert_eq!(read16(&mut host, 0x210), 1920);
    assert_eq!(read16(&mut host, 0x212), 1080);
    assert_eq!(read16(&mut host, 0x220), 1440);
    assert_eq!(read16(&mut host, 0x230), 1280);
    assert_eq!(read16(&mut host, 0x240), 3840);
    assert_eq!(read16(&mut host, 0x242), 2160);
    // Every element carries the same 120 Hz refresh (16.16 fixed-point).
    let mut refresh = [0u8; 4];
    host.read_gpa(gpa + 0x244, &mut refresh).unwrap();
    assert_eq!(u32::from_le_bytes(refresh), DISPLAY_REFRESH_HZ << 16);
}

#[test]
fn present_page_identity_reports_alias_and_disjoint_peers() {
    use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let entry = |pfn: u32| (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    // Present-named surface (surface-id namespace): pages A.
    assert!(state.map_surface(4));
    {
        let m = state.mappings.get_mut(&4).unwrap();
        m.mapped = true;
        m.has_geom = true;
        m.width = 4;
        m.height = 4;
        m.page_entries = vec![entry(0x100), entry(0x101)];
    }
    // Composite peer aliasing the SAME pages (mapping namespace).
    assert!(state.map_surface(1));
    {
        let m = state.mappings.get_mut(&1).unwrap();
        m.mapped = true;
        m.has_geom = true;
        m.width = 4;
        m.height = 4;
        m.page_entries = vec![entry(0x100), entry(0x101)];
    }
    state
        .surface_write_kind
        .insert(1, crate::model::SurfaceWriteKind::Composite);
    // Same-geometry peer with disjoint pages.
    assert!(state.map_surface(2));
    {
        let m = state.mappings.get_mut(&2).unwrap();
        m.mapped = true;
        m.has_geom = true;
        m.width = 4;
        m.height = 4;
        m.page_entries = vec![entry(0x200), entry(0x201)];
    }
    // Different geometry: excluded entirely.
    assert!(state.map_surface(9));
    {
        let m = state.mappings.get_mut(&9).unwrap();
        m.mapped = true;
        m.has_geom = true;
        m.width = 8;
        m.height = 8;
        m.page_entries = vec![entry(0x100)];
    }
    let line = present_page_identity_line(&state, 4, 4, 4).expect("line");
    assert!(line.contains("present_page_identity mid=4 4x4 pages=2 valid=2"));
    assert!(
        line.contains("mid1:pages=2:overlap=2:ident=1:kind=Composite"),
        "alias peer must report identical pages: {line}"
    );
    assert!(
        line.contains("mid2:pages=2:overlap=0:ident=0"),
        "disjoint peer must report zero overlap: {line}"
    );
    assert!(
        !line.contains("mid9"),
        "geometry-mismatched mapping excluded: {line}"
    );
}

#[test]
fn display_swap_paints_mapping_geom_not_console_fallback() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    // Established boot console 1920×1080.
    state.present.valid = true;
    state.present.width = 1920;
    state.present.height = 1080;
    assert!(state.map_surface(3));
    {
        let m = state.mappings.get_mut(&3).unwrap();
        m.mapped = true;
        m.has_geom = true;
        m.width = 1440;
        m.height = 1080;
        m.content_generation = 5;
        m.page_entries = vec![1];
    }
    let pkt = present_packet(CHILD_OP_DISPLAY_SWAP, 3);
    process_child_packet(&mut state, &mut host, 4, &pkt);
    assert!(state.present.frame_flush_seen);
    assert_eq!(state.present.width, 1440);
    assert_eq!(state.present.height, 1080);
    // Geometry is asserted above off `state.present`. The mapping identity
    // moves from the (now absent) action's a0 to `present_mapping` — the
    // accepted present. NOT the retain: the capture fails here (no guest
    // pages), which the old action tolerated via its `paint_mid` fallback.
    assert_eq!(state.present.present_mapping, 3);
    assert_coalesced_paint_action(&host, "mapping geom, not console fallback");
}

/// A present whose named mid's last write was a CLEAR captures the surface the
/// transaction names, even when a same-geometry Composite peer holds different
/// pixels.
///
/// This exact state — a ClearOnly named mid alongside a Composite `early_front`
/// peer — is what a six-way peer resolver used to answer with the peer, on the
/// theory that a mid cleared rather than drawn held nothing worth showing. The
/// transaction payload carries exactly one field, plane 0's surface id, so the
/// named surface is the only correct capture source; substituting a peer shows a
/// buffer the guest never asked for.
#[test]
fn clear_only_present_captures_the_surface_the_transaction_names() {
    use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let w = 1920u32;
    let h = 1080u32;
    let stride = w * 4;
    let need = (stride as usize) * (h as usize);
    let page_shift = PAGE_SHIFT_X86;
    let page_size = 1u64 << page_shift;
    let pages = (need as u64).div_ceil(page_size) as usize;
    for mid in [1u32, 2u32] {
        let base_pfn = 0x100u32 + mid * 0x1000;
        let mut entries = Vec::with_capacity(pages);
        for i in 0..pages {
            let pfn = base_pfn + i as u32;
            let gpa = (pfn as u64) << page_shift;
            host.map_range(gpa, page_size as usize, 0);
            entries
                .push((((pfn as u64) << PAGE_ENTRY_PFN_SHIFT) | (PAGE_ENTRY_VALID as u64)) as u32);
        }
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = mid as u64;
            m.page_entries = entries;
        }
        assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));
    }
    // Mid 1: the Composite peer the resolver used to hand the display.
    let gray = vec![0xAAu8; need];
    assert!(write_bgra8(&mut state, &mut host, 1, &gray, stride, w, h));
    state.note_surface_composite(1);
    state.present.early_front_mapping = 1;
    state.present.valid = true;
    state.present.width = w;
    state.present.height = h;
    assert!(!state.present.frame_flush_seen);
    assert!(!state.present.frame_valid);

    // Mid 2: the surface the guest names, cleared to opaque black.
    let mut clear = vec![0u8; need];
    for px in clear.chunks_exact_mut(4) {
        px[3] = 255;
    }
    assert!(write_bgra8(&mut state, &mut host, 2, &clear, stride, w, h));
    state.note_surface_clear(2);
    assert!(
        matches!(
            state.surface_write_kind(2),
            crate::model::SurfaceWriteKind::ClearOnly
        ),
        "the named mid must be the ClearOnly case this test is about"
    );

    process_child_packet(
        &mut state,
        &mut host,
        5,
        &present_packet(CHILD_OP_DISPLAY_TRANSACTION2, 2),
    );

    assert_eq!(state.present.present_mapping, 2, "guest names mid 2");
    assert!(
        state.present.frame_flush_seen,
        "a non-init present leaves BAR1"
    );
    assert!(state.present.frame_valid);
    assert_eq!(
        state.present.frame_mapping, 2,
        "+0x188 holds the named mid, not the Composite peer"
    );
    assert_eq!(
        state.present.frame_bgra[0], 0x00,
        "captured the named surface's cleared pages, not the peer's 0xAA"
    );
    assert_eq!(
        presented_mapping(&state),
        Some(2),
        "window shows named mid 2, not peer mid 1"
    );
    assert_coalesced_paint_action(&host, "named surface, not composite peer");
}

/// CmdDeleteTask (root 0x20) must clear the task — not flood UnknownRootOpcode.
#[test]
fn delete_task_root_clears_active_task() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.define_task(3, 0x1000, 2);
    assert!(state.tasks[3].active);
    process_root_packet(
        &mut state,
        &mut host,
        &Packet {
            opcode: ROOT_OP_DELETE_TASK,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + 4,
            completion_stamp: 0,
            payload: 3u32.to_le_bytes().to_vec(),
            next_head: 0,
        },
    );
    assert!(
        !state.tasks.is_active(3),
        "DeleteTask must leave no live task 3"
    );
    assert!(
        !state
            .fails
            .iter()
            .any(|e| matches!(e, FailEvent::UnknownRootOpcode { opcode: 0x20, .. })),
        "0x20 must not be UnknownRootOpcode"
    );
}

/// CmdReplacePhysical (0x3c) is the guest saying a cached page list is stale.
///
/// It must drop that list rather than stamp and forget it. The surface id, the
/// geometry and the GPU-VA are all unchanged across the re-commit, so a device
/// that ignores this packet has no other way to learn the pages moved — which
/// is what `mapping_page_drift`'s "no packet said so" was reporting.
#[test]
fn replace_physical_drops_the_cached_page_list() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.map_surface(7);
    {
        let m = state.mappings.get_mut(&7).unwrap();
        m.mapped = true;
        m.page_entries = vec![0x11, 0x22, 0x33];
        m.backing_walk = Some(crate::model::BackingWalk {
            task_id: 0,
            backing_pfn: 0x20,
            map_generation: m.map_generation,
        });
    }
    let generation_before = state.mappings.get(&7).unwrap().map_generation;

    let mut payload = vec![0u8; 8];
    payload[4..8].copy_from_slice(&7u32.to_le_bytes()); // {task 0, object 7}
    let pkt = Packet {
        opcode: CHILD_OP_REPLACE_PHYSICAL,
        stamp_waits: Vec::new(),
        total_size: PACKET_HEADER_LEN + 8,
        completion_stamp: 0,
        payload,
        next_head: 0,
    };
    process_child_packet(&mut state, &mut host, 2, &pkt);

    assert!(
        !state
            .fails
            .iter()
            .any(|e| matches!(e, FailEvent::UnknownChildOpcode { opcode: 0x3c, .. })),
        "0x3c must not flood UnknownChildOpcode"
    );
    let m = state.mappings.get(&7).unwrap();
    assert!(
        m.page_entries.is_empty(),
        "the announced re-point must drop the stale page list"
    );
    assert_ne!(
        m.map_generation, generation_before,
        "dropping the list must bump the incarnation, which is what retires the \
         backing walk latch and any state keyed on it"
    );
}

/// A `CmdReplacePhysical` too short to hold its eight bytes reports, and does
/// not act.
///
/// The arm checked the floor itself and then called a decoder that checked it
/// again and returned a bare `Option`. The second check could not fire, and if
/// it ever had, its `None` arm dropped the packet without a word — on the
/// command that re-points a resource at different host frames, where a dropped
/// one leaves this device resolving a stale address.
#[test]
fn a_short_replace_physical_is_reported_and_not_acted_on() {
    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::default();
    // Two mappings: the one a well-formed packet would name, and the one a
    // decoder that zero-filled the words it could not read would name instead.
    // Nothing may touch either.
    for object in [0u32, 7] {
        state.mappings.insert(object, Default::default());
        if let Some(m) = state.mappings.get_mut(&object) {
            m.page_entries = vec![0x11, 0x22, 0x33];
        }
    }
    let short = |plen: usize| Packet {
        opcode: CHILD_OP_REPLACE_PHYSICAL,
        stamp_waits: Vec::new(),
        total_size: PACKET_HEADER_LEN + plen as u32,
        completion_stamp: 0,
        payload: vec![0u8; plen],
        next_head: 0,
    };
    // Seven bytes first, and alone in its own capture. It is the length a
    // decoder that read what it could and zero-filled the rest would accept —
    // the line is latched per opcode, so a later length reporting instead of
    // this one would look identical from outside.
    let cap = crate::observe::FailCapture::start();
    process_child_packet(&mut state, &mut host, 2, &short(7));
    let lines = cap.lines();
    let line = lines
        .iter()
        .find(|l| l.contains("reason=replace_physical_short"))
        .unwrap_or_else(|| panic!("a short replace-physical said nothing: {lines:?}"));
    assert!(
        line.contains("need=8") && line.contains("plen=7"),
        "the line must say what it needed and what it got: {line}"
    );
    for plen in 0..7usize {
        process_child_packet(&mut state, &mut host, 2, &short(plen));
    }
    for object in [0u32, 7] {
        assert_eq!(
            state.mappings.get(&object).map(|m| m.page_entries.len()),
            Some(3),
            "a packet with nothing to read must not re-point anything, least of \
             all whatever holds slot zero"
        );
    }
}

/// A re-point naming a mapper-ref-texture resource reaches the mapping associated with its
/// task-local ref. This is the packet family the arm used to drop entirely —
/// 57 % of the re-points on a driven boot found no mapping under `object_id` —
/// and dropping it leaves the texture's page list trusted while it names pages
/// that back something else.
#[test]
fn replace_physical_routes_through_the_texture_ref_when_no_mapping_owns_the_id() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    // Mapping 41 backs a mapper-ref-texture the guest registered at object-list
    // ref 12 of task 3. Nothing is mapped under id 12.
    state.map_surface(41);
    {
        let m = state.mappings.get_mut(&41).unwrap();
        m.mapped = true;
        m.page_entries = vec![0x51, 0x52];
    }
    state.texture_to_mapping.insert((3, 12), 41);
    let generation_before = state.mappings.get(&41).unwrap().map_generation;

    let mut payload = vec![0u8; 8];
    payload[0..4].copy_from_slice(&3u32.to_le_bytes()); // task 3
    payload[4..8].copy_from_slice(&12u32.to_le_bytes()); // object 12
    let pkt = Packet {
        opcode: CHILD_OP_REPLACE_PHYSICAL,
        stamp_waits: Vec::new(),
        total_size: PACKET_HEADER_LEN + 8,
        completion_stamp: 0,
        payload,
        next_head: 0,
    };
    process_child_packet(&mut state, &mut host, 2, &pkt);

    let m = state.mappings.get(&41).unwrap();
    assert!(
        m.page_entries.is_empty(),
        "a re-point that names a texture ref must reach the mapping behind it"
    );
    assert_ne!(
        m.map_generation, generation_before,
        "the incarnation must move, or a resident gathered from the old pages \
         stays eligible"
    );
}

/// ReplacePhysical is one resource's lifecycle transition, not a task reset.
/// Retiring every GVA resident on the task loses unrelated host-authoritative
/// frames when the compositor re-points several short-lived resources quickly.
#[test]
fn replace_physical_retires_only_the_named_resource() {
    use crate::runtime::writeback_debt::{GvaResourceKey, GvaWritebackDebt};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let key = |texture_ref| GvaResourceKey {
        task_id: 3,
        texture_ref,
    };
    let debt = |gva, generation| GvaWritebackDebt {
        gva,
        row_stride: 256,
        width: 64,
        height: 64,
        format: crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM,
        generation,
        guest_write: Default::default(),
        seq: 0,
    };
    assert_eq!(
        state.pending_writebacks.arm_gva(key(12), debt(0x4000, 1)),
        None
    );
    assert_eq!(
        state.pending_writebacks.arm_gva(key(13), debt(0x8000, 2)),
        None
    );

    let mut payload = vec![0u8; 8];
    payload[0..4].copy_from_slice(&3u32.to_le_bytes());
    payload[4..8].copy_from_slice(&12u32.to_le_bytes());
    process_child_packet(
        &mut state,
        &mut host,
        2,
        &Packet {
            opcode: CHILD_OP_REPLACE_PHYSICAL,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + 8,
            completion_stamp: 0,
            payload,
            next_head: 0,
        },
    );

    assert!(state.pending_writebacks.get_gva(key(12)).is_none());
    assert_eq!(
        state
            .pending_writebacks
            .get_gva(key(13))
            .map(|debt| debt.generation),
        Some(2),
        "an unrelated resource on the task keeps its authoritative frame"
    );
}

/// A same-number mapping owned by another task must not capture a task-local
/// resource re-point. The association names mapping 99; mapping 7 merely shares
/// its integer and has independent backing provenance.
#[test]
fn replace_physical_does_not_confuse_another_tasks_mapping_for_the_resource() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.map_surface(7);
    {
        let m = state.mappings.get_mut(&7).unwrap();
        m.mapped = true;
        m.page_entries = vec![0x77];
        m.backing_walk = Some(crate::model::BackingWalk {
            task_id: 1,
            backing_pfn: 0x20,
            map_generation: m.map_generation,
        });
    }
    state.map_surface(99);
    {
        let m = state.mappings.get_mut(&99).unwrap();
        m.mapped = true;
        m.page_entries = vec![0xaa];
    }
    state.texture_to_mapping.insert((0, 7), 99);

    let mut payload = vec![0u8; 8];
    payload[4..8].copy_from_slice(&7u32.to_le_bytes()); // {task 0, object 7}
    let pkt = Packet {
        opcode: CHILD_OP_REPLACE_PHYSICAL,
        stamp_waits: Vec::new(),
        total_size: PACKET_HEADER_LEN + 8,
        completion_stamp: 0,
        payload,
        next_head: 0,
    };
    process_child_packet(&mut state, &mut host, 2, &pkt);

    assert_eq!(
        state.mappings.get(&7).unwrap().page_entries,
        vec![0x77],
        "task 1's surface 7 is unrelated to task 0's resource 7"
    );
    assert!(
        state.mappings.get(&99).unwrap().page_entries.is_empty(),
        "the task-local association names mapping 99"
    );
}

/// A short 0x3c is a lost invalidation, not a no-op: the device would keep
/// writing through pages the guest has re-pointed. It must be named, and it
/// must not silently drop a list it could not identify.
#[test]
fn a_short_replace_physical_is_reported_and_drops_nothing() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.map_surface(7);
    {
        let m = state.mappings.get_mut(&7).unwrap();
        m.mapped = true;
        m.page_entries = vec![0x11];
    }
    let pkt = Packet {
        opcode: CHILD_OP_REPLACE_PHYSICAL,
        stamp_waits: Vec::new(),
        total_size: PACKET_HEADER_LEN + 4,
        completion_stamp: 0,
        payload: vec![0u8; 4],
        next_head: 0,
    };
    process_child_packet(&mut state, &mut host, 2, &pkt);
    assert!(
        !state.mappings.get(&7).unwrap().page_entries.is_empty(),
        "a packet too short to name an object must not invalidate one"
    );
}

#[test]
fn delete_iosurface_backing_condemns_then_second_delete_tears_down() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    assert!(state.map_surface(3));
    {
        let m = state.mappings.get_mut(&3).unwrap();
        m.page_entries = vec![0x101];
        m.mapping_internal = 0x1234;
    }
    let delete = |state: &mut DeviceState, host: &mut FakeHost| {
        let mut payload = Vec::new();
        payload.extend_from_slice(&3u32.to_le_bytes()); // objectID
        payload.extend_from_slice(&1u32.to_le_bytes()); // taskID
        process_child_packet(
            state,
            host,
            2,
            &Packet {
                opcode: CHILD_OP_DELETE_IOSURFACE_BACKING2,
                stamp_waits: Vec::new(),
                total_size: PACKET_HEADER_LEN + payload.len() as u32,
                completion_stamp: 0,
                payload,
                next_head: 0,
            },
        );
    };
    // First delete: the id may already carry a live re-used incarnation
    // (the delete trails the guest release) — condemn: retire bindings,
    // keep content state for the resolve-time fingerprint decision.
    delete(&mut state, &mut host);
    let m = state.mappings.get(&3).unwrap();
    assert!(m.mapped, "condemn keeps the slot live");
    assert!(m.page_entries.is_empty(), "bindings must be retired");
    assert_eq!(m.condemned_entries.as_deref(), Some(&[0x101u32][..]));
    // Second delete with no resolve between: genuinely dead — full
    // teardown.
    delete(&mut state, &mut host);
    let m = state.mappings.get(&3).unwrap();
    assert!(!m.mapped);
    assert!(m.page_entries.is_empty());
    assert!(m.condemned_entries.is_none());
    assert_eq!(m.mapping_internal, 0);
}

/// Direct Composite-named present (no ClearOnly pairing): the transaction
/// payload carries exactly one thing — plane 0's surface id — so the only
/// correct capture source is the surface the guest named. No comparison
/// between our own full-frame sequences may override it, however far the
/// named member's sequence lags a same-geometry peer's. Substituting the
/// "denser" peer is what shows a buffer one rotation step behind the one the
/// guest asked for: residue when a window closed in between, a stale region
/// when one moved, and visible thrash as the choice oscillates.
#[test]
fn composite_named_present_captures_the_named_member_however_far_it_lags() {
    use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let w = 1920u32;
    let h = 1080u32;
    let stride = w * 4;
    let need = (stride as usize) * (h as usize);
    let page_shift = PAGE_SHIFT_X86;
    let page_size = 1u64 << page_shift;
    let pages = (need as u64).div_ceil(page_size) as usize;
    for mid in [1u32, 5u32] {
        let base_pfn = 0x100u32 + mid * 0x1000;
        let mut entries = Vec::with_capacity(pages);
        for i in 0..pages {
            let pfn = base_pfn + i as u32;
            host.map_range((pfn as u64) << page_shift, page_size as usize, 0);
            entries
                .push((((pfn as u64) << PAGE_ENTRY_PFN_SHIFT) | (PAGE_ENTRY_VALID as u64)) as u32);
        }
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = mid as u64;
            m.page_entries = entries;
        }
        assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));
    }
    let fresh = vec![0x11u8; need];
    assert!(write_bgra8(&mut state, &mut host, 1, &fresh, stride, w, h));
    state.note_surface_composite(1);
    let stale = vec![0x55u8; need];
    assert!(write_bgra8(&mut state, &mut host, 5, &stale, stride, w, h));
    state.note_surface_composite(5);
    // Both members are genuine swapchain buffers that alternate as the presented
    // front.
    state.present.valid = true;
    state.present.width = w;
    state.present.height = h;

    let present_named = |state: &mut DeviceState, host: &mut FakeHost, mid: u32| {
        process_child_packet(
            state,
            host,
            5,
            &present_packet(CHILD_OP_DISPLAY_TRANSACTION2, mid),
        );
    };

    // Healthy alternation: both members publish, the named member is captured.
    state.note_dense_frame_published(5, w, h);
    state.note_dense_frame_published(1, w, h);
    present_named(&mut state, &mut host, 5);
    assert_eq!(
        state.present.frame_mapping, 5,
        "alternation captures the named member"
    );
    assert_eq!(state.present.frame_bgra[0], 0x55);

    // Drive the named member's full-frame sequence arbitrarily far behind its
    // peer's: mid 1 publishes a long run while mid 5 receives none. The guest
    // still names mid 5, so mid 5 is still what goes on screen.
    let lag_runs = 34u64;
    for _ in 0..lag_runs {
        state.note_dense_frame_published(1, w, h);
    }
    // Read the lag straight out of the per-mapping counters — the point of this
    // test is that the lag exists and changes nothing about what is captured.
    let named_seq = state.present.dense_frame_seq[&5];
    let peer_seq = state.present.dense_frame_seq[&1];
    assert!(
        peer_seq - named_seq >= lag_runs,
        "the lag this test needs is present: {peer_seq} - {named_seq}"
    );
    present_named(&mut state, &mut host, 5);
    assert_eq!(
        state.present.frame_mapping, 5,
        "the guest named mid 5; no sequence comparison may substitute a peer"
    );
    assert_eq!(
        state.present.frame_bgra[0], 0x55,
        "captured the named member's content, not the peer's"
    );
}

/// A display transaction cannot overtake an EXEC packet held on another
/// child channel while immutable AIR translation is loading. Repeated
/// polls hold the same packet without side effects or proxy-log flooding;
/// once ready, the packet completes normally.
#[test]
fn present_holds_for_translation_deferred_on_other_channel() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.translation_deferred_mask = 1 << 1;

    assert_eq!(
        present_named_mapping(&mut state, &mut host, 5, 2),
        ChildPacketDisposition::Deferred
    );
    assert_eq!(
        present_named_mapping(&mut state, &mut host, 5, 2),
        ChildPacketDisposition::Deferred
    );

    assert_eq!(state.present_translation_holds, 1);
    assert_eq!(state.present_translation_hold_mask, 1 << 5);
    assert_eq!(state.present.present_mapping, 0);
    assert!(!state.present.frame_flush_seen);

    state.translation_deferred_mask = 0;
    assert_eq!(
        present_named_mapping(&mut state, &mut host, 5, 2),
        ChildPacketDisposition::Complete
    );
    assert_eq!(state.present_translation_hold_mask, 0);
    assert_eq!(state.present.present_mapping, 2);
    assert!(state.present.frame_flush_seen);
}

/// The currently executing display channel cannot be an overtaken sibling
/// and is excluded from the proxy mask.
#[test]
fn present_does_not_hold_for_current_channel_translation_bit() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.translation_deferred_mask = 1 << 5;

    assert_eq!(
        present_named_mapping(&mut state, &mut host, 5, 2),
        ChildPacketDisposition::Complete
    );

    assert_eq!(state.present_translation_holds, 0);
}

/// A cold-translation EXEC owns the scheduler timeline even though its AIR
/// worker is asynchronous. A sibling Unmap must remain at FIFO head with
/// its stamp and task-map state untouched until that boundary is ready.
#[test]
fn translation_deferred_holds_sibling_unmap_head_and_stamp() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let page_size = state.page_size() as usize;
    let channel = 2u32;
    let producer_bit = 1u32 << 1;
    let sibling_bit = 1u32 << channel;
    let root_pfn = 0x10u32;
    let list_pfn = 0x20u32;
    let ring_pfn = 0x30u32;
    let stamp_pfn = 0x40u32;
    let root_gpa = state.pfn_gpa(root_pfn);
    let list_gpa = state.pfn_gpa(list_pfn);
    let ring_gpa = state.pfn_gpa(ring_pfn);
    let stamp_gpa = state.pfn_gpa(stamp_pfn);
    for gpa in [root_gpa, list_gpa, ring_gpa, stamp_gpa] {
        host.map_range(gpa, page_size, 0);
    }

    let task_id = 6u32;
    let gva = 0x101000u64;
    let length = 0x4000u64;
    let mut payload = vec![0u8; 20];
    payload[0..4].copy_from_slice(&task_id.to_le_bytes());
    payload[4..12].copy_from_slice(&gva.to_le_bytes());
    payload[12..20].copy_from_slice(&length.to_le_bytes());
    let packet = packet_bytes(CHILD_OP_UNMAP_MEMORY, 0x55, &payload);
    host.write_gpa(ring_gpa, &packet).unwrap();
    host.put_u32(list_gpa, ring_pfn);

    let regs_gpa = root_gpa + child_reg_block_offset(channel).unwrap();
    host.put_u32(regs_gpa + CHILD_REG_TAIL, packet.len() as u32);
    host.put_u32(regs_gpa + CHILD_REG_HEAD, 0);
    host.put_u32(regs_gpa + CHILD_REG_STAMP_INDEX, 1);
    host.put_u32(regs_gpa + CHILD_REG_BASE_PFN, list_pfn);
    state.gfx.root_page = root_pfn;
    state.gfx.fifo_base_page = stamp_pfn;
    state.open_child_domains_for_test(producer_bit | sibling_bit);
    state.pending.child_mask = sibling_bit;
    state.translation_deferred_mask = producer_bit;

    drain_pending(&mut state, &mut host);
    drain_pending(&mut state, &mut host);
    assert_eq!(host.get_u32(regs_gpa + CHILD_REG_HEAD), 0);
    assert_eq!(host.get_u32(stamp_gpa + 4), 0);
    assert_eq!(state.translation_order_hold_mask, sibling_bit);
    assert_eq!(state.translation_order_holds, 1, "poll retries coalesce");

    note_translation_order_hold(&mut state, ROOT_FIFO_BIT);
    assert_eq!(
        state.translation_order_holds, 1,
        "new timeline bits in one ownership interval remain one episode"
    );

    // Simulate the immutable AIR worker becoming ready. The real producer
    // retry clears this bit in process_child_packet before siblings resume.
    state.translation_deferred_mask = 0;
    drain_pending(&mut state, &mut host);
    assert_eq!(host.get_u32(regs_gpa + CHILD_REG_HEAD), packet.len() as u32);
    assert_eq!(host.get_u32(stamp_gpa + 4), 0x55);
    assert_eq!(state.translation_order_hold_mask, 0);
}

/// A cursor pitch that does not fit a word must be refused, not wrapped.
///
/// `CmdSetCursorGlyph` carries `stride` in eight bytes and the drain wanted
/// four. Taking the low four of `0x1_0000_0100` gives `0x100`, which clears the
/// geometry bound for a 64-pixel-wide sprite and makes `need` small enough that
/// a modest `mapped_length` covers it — so the glyph was accepted and then read
/// at a pitch of 256 bytes when the guest laid it out at four gigabytes and a
/// bit. A garbled pointer, and nothing on the failure channel to say why.
///
/// Compared in its own width the same record does not survive: `need` is the
/// height times the real pitch, `mapped_length` does not cover it, and
/// `cursor_glyph_mapped_len` says so.
#[test]
fn a_cursor_pitch_wider_than_a_word_is_refused_rather_than_truncated() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut payload = vec![0u8; crate::protocol::fifo::CURSOR_GLYPH_LEN];
    let put32 =
        |p: &mut Vec<u8>, at: usize, v: u32| p[at..at + 4].copy_from_slice(&v.to_le_bytes());
    let put64 =
        |p: &mut Vec<u8>, at: usize, v: u64| p[at..at + 8].copy_from_slice(&v.to_le_bytes());
    put32(&mut payload, crate::protocol::fifo::CURSOR_GLYPH_TASK_ID, 1);
    put64(
        &mut payload,
        crate::protocol::fifo::CURSOR_GLYPH_MAPPED_LENGTH,
        0x4000,
    );
    // Low word `0x100`, which is exactly the pitch a 64-wide sprite needs.
    put64(
        &mut payload,
        crate::protocol::fifo::CURSOR_GLYPH_STRIDE,
        0x1_0000_0100,
    );
    payload[crate::protocol::fifo::CURSOR_GLYPH_WIDTH..][..2].copy_from_slice(&64u16.to_le_bytes());
    payload[crate::protocol::fifo::CURSOR_GLYPH_HEIGHT..][..2]
        .copy_from_slice(&32u16.to_le_bytes());

    process_child_packet(
        &mut state,
        &mut host,
        4,
        &Packet {
            opcode: CHILD_OP_CURSOR_GLYPH,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + payload.len() as u32,
            completion_stamp: 0,
            payload,
            next_head: 0,
        },
    );

    assert!(
        !state.cursor.glyph_ready,
        "a sprite read at a pitch the guest did not use is not a glyph"
    );
    assert!(state.cursor.pixels.is_empty());
    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(
        log.contains("reason=cursor_glyph_mapped_len"),
        "the refusal names the bound the real pitch failed, not silence: {log}"
    );
}

/// The open end of the same lifetime owes the same forgetting, and its
/// sibling below is the only thing that was ever asserted about it.
///
/// A channel number is reused. A hold or a cached ring left over from the
/// previous occupant is the new channel inheriting a decision nothing made
/// about it — a deferred translation it never asked for, or a ring whose head
/// belongs to a producer that is gone.
///
/// The one thing an open does *not* clear is `pending.child_mask`. Opening a
/// channel is not a claim that nothing is pending on it, and clearing it here
/// would drop a drain the guest is owed.
#[test]
fn define_fifo_forgets_the_previous_occupants_translation_state() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let bit = 1 << 1;
    state.pending.child_mask = bit;
    state.translation_deferred_mask = bit;
    state.translation_order_hold_mask = bit;
    state.present_translation_hold_mask = bit;

    process_root_packet(
        &mut state,
        &mut host,
        &Packet {
            opcode: ROOT_OP_DEFINE_FIFO,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + 4,
            completion_stamp: 0,
            payload: 1u32.to_le_bytes().to_vec(),
            next_head: 0,
        },
    );

    assert!(state.child_domain_open(1), "the channel is open");
    assert_eq!(state.translation_deferred_mask & bit, 0);
    assert_eq!(state.translation_order_hold_mask & bit, 0);
    assert_eq!(state.present_translation_hold_mask & bit, 0);
    assert_eq!(
        state.pending.child_mask & bit,
        bit,
        "an open is not a claim that nothing is pending on the channel"
    );
}

/// A channel-lifetime command naming a domain this device has no FIFO for says
/// so.
///
/// Dropped in silence before: the arm gated on [`is_child_channel`] with an
/// `Ok(_) => {}` beside it, so a guest opening channel 0 or channel 32 set no
/// mask bit, got no ring drained, and was never told. Every packet it then
/// queued there carries a completion word nothing will publish, which is a hang
/// the log had nothing in it about — the same defect
/// `channel_id_or_refusal` was written to close at the doorbell handlers, on the
/// one path that still had it.
#[test]
fn a_channel_lifetime_outside_the_channel_range_is_reported() {
    // The root FIFO's own number and the first above the bound: `0` is not a
    // child and `MAX_CHANNELS` is one past the last, so both are ids the guest
    // can write and this device cannot drain.
    for channel in [0u32, crate::model::MAX_CHANNELS as u32] {
        for opcode in [ROOT_OP_DEFINE_FIFO, ROOT_OP_FREE_FIFO] {
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
            let mut host = FakeHost::new();
            let before = state.open_child_mask();
            let cap = crate::observe::FailCapture::start();
            process_root_packet(
                &mut state,
                &mut host,
                &Packet {
                    opcode,
                    stamp_waits: Vec::new(),
                    total_size: PACKET_HEADER_LEN + 4,
                    completion_stamp: 0,
                    payload: channel.to_le_bytes().to_vec(),
                    next_head: 0,
                },
            );
            let lines = cap.lines();
            assert!(
                lines
                    .iter()
                    .any(|l| l.contains("channel_lifetime_out_of_range")
                        && l.contains(&format!("channel={channel}"))),
                "channel {channel} on {opcode:#x} said nothing: {lines:?}"
            );
            assert_eq!(
                state.open_child_mask(),
                before,
                "an out-of-range id must move no mask bit"
            );
        }
    }
}

/// A channel definition opens the domain in the semantic model, and nothing
/// else on this device does.
///
/// The cutover's own assertion. Openness used to be a `DeviceState` field that
/// three events wrote — a definition, a locked doorbell register write, and a
/// lock-free ring — and the two doorbell events wrote it for a different
/// question. Each of the three is driven here and only the definition may move
/// the answer; the doorbells must move the pending-work mask instead, which is
/// the question they actually answer.
#[test]
fn a_definition_opens_the_domain_in_the_model_and_a_doorbell_does_not() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let channel = 4u32;
    let bit = 1u32 << channel;

    // A lock-free ring.
    state
        .gfx
        .child_doorbell_rung
        .store(bit, std::sync::atomic::Ordering::Release);
    crate::runtime::drain::fold_rung_child_doorbells(&mut state);
    assert_eq!(
        state.pending.child_mask & bit,
        bit,
        "a ring is work waiting on the domain"
    );
    assert!(
        !state.child_domain_open(channel),
        "and it is not a claim that the guest defined one"
    );
    assert_eq!(
        state.drainable_child_mask() & bit,
        bit,
        "the union the drain walks still holds it, which is what a ring is for"
    );

    // The definition, which is the model's one event for openness.
    let transition = |state: &mut DeviceState, host: &mut FakeHost, opcode| {
        process_root_packet(
            state,
            host,
            &Packet {
                opcode,
                stamp_waits: Vec::new(),
                total_size: PACKET_HEADER_LEN + 4,
                completion_stamp: 0,
                payload: channel.to_le_bytes().to_vec(),
                next_head: 0,
            },
        );
    };
    transition(&mut state, &mut host, ROOT_OP_DEFINE_FIFO);
    assert!(state.child_domain_open(channel));
    assert_eq!(state.open_child_mask() & bit, bit);

    // The free takes it back, and takes the pending bit with it — an open is
    // not a claim that nothing is pending, but a free is.
    transition(&mut state, &mut host, ROOT_OP_FREE_FIFO);
    assert!(!state.child_domain_open(channel));
    assert_eq!(state.pending.child_mask & bit, 0);
    assert_eq!(state.drainable_child_mask() & bit, 0);
}

/// The two channel transitions the replacement model refuses and this device
/// performs are counted apart from the ones it agrees with.
///
/// Neither refusal withholds a completion word, so neither is a hang. Each is an
/// effect that would not happen — a redefinition that does not reset the
/// channel, a free that leaves the domain open — and whether this guest sends
/// either is the term the first group's cutover turns on.
#[test]
fn the_channel_transitions_the_model_would_refuse_are_counted_apart() {
    use crate::runtime::drain::store_route_count;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let channel = 2u32;
    let transition = |state: &mut DeviceState, host: &mut FakeHost, opcode| {
        process_root_packet(
            state,
            host,
            &Packet {
                opcode,
                stamp_waits: Vec::new(),
                total_size: PACKET_HEADER_LEN + 4,
                completion_stamp: 0,
                payload: channel.to_le_bytes().to_vec(),
                next_head: 0,
            },
        );
    };

    let agrees = store_route_count("channel_transition_model_agrees");
    let reopen = store_route_count("channel_open_of_open_domain");
    let orphan_free = store_route_count("channel_free_of_undefined_domain");

    // A free before any definition: the model has nothing to remove.
    transition(&mut state, &mut host, ROOT_OP_FREE_FIFO);
    assert_eq!(
        (
            store_route_count("channel_transition_model_agrees"),
            store_route_count("channel_free_of_undefined_domain"),
        ),
        (agrees, orphan_free + 1),
    );

    // The first definition is the one both agree on.
    transition(&mut state, &mut host, ROOT_OP_DEFINE_FIFO);
    assert_eq!(
        store_route_count("channel_transition_model_agrees"),
        agrees + 1,
    );

    // The second, over a domain still open, is the one the model refuses —
    // while this device resets the channel through `forget_child_channel`.
    transition(&mut state, &mut host, ROOT_OP_DEFINE_FIFO);
    assert_eq!(
        (
            store_route_count("channel_transition_model_agrees"),
            store_route_count("channel_open_of_open_domain"),
        ),
        (agrees + 1, reopen + 1),
    );

    // And the free that follows a definition is agreed on again.
    transition(&mut state, &mut host, ROOT_OP_FREE_FIFO);
    assert_eq!(
        (
            store_route_count("channel_transition_model_agrees"),
            store_route_count("channel_open_of_open_domain"),
            store_route_count("channel_free_of_undefined_domain"),
        ),
        (agrees + 2, reopen + 1, orphan_free + 1),
    );
}

/// Every packet this device dispatches is counted under the class the model
/// would read it as, and an unclassifiable one is counted apart.
///
/// The last group's denominator. `SessionModel::admit` refuses `UnknownCommand`
/// and `UnestablishedContract`, and both are `classify` answering `None` — so a
/// boot's reading on `packet_class_unclassified` is how many whole packets the
/// ordering group would refuse where this device acts.
#[test]
fn every_dispatched_packet_is_counted_under_the_class_the_model_reads_it_as() {
    use crate::model::{CHILD_OP_NOP, ROOT_OP_DEFINE_FIFO};
    use crate::runtime::drain::store_route_count;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();

    let control = store_route_count("packet_class_control");
    let unclassified = store_route_count("packet_class_unclassified");

    process_root_packet(
        &mut state,
        &mut host,
        &Packet {
            opcode: ROOT_OP_DEFINE_FIFO,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + 4,
            completion_stamp: 0,
            payload: 1u32.to_le_bytes().to_vec(),
            next_head: 0,
        },
    );
    assert_eq!(
        (
            store_route_count("packet_class_control"),
            store_route_count("packet_class_unclassified"),
        ),
        (control + 1, unclassified),
        "a channel definition is control, and the model reads it as such"
    );

    // An opcode the ledger has no row for. The model would refuse the whole
    // packet; this device reaches its own unknown-opcode arm.
    let _ = process_child_packet(
        &mut state,
        &mut host,
        1,
        &Packet {
            opcode: 0x0fff,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN,
            completion_stamp: 0,
            payload: Vec::new(),
            next_head: 0,
        },
    );
    assert_eq!(
        (
            store_route_count("packet_class_control"),
            store_route_count("packet_class_unclassified"),
        ),
        (control + 1, unclassified + 1),
        "and an opcode with no row is counted apart rather than into a class"
    );

    // A no-op is classified, which is what makes the counter above a finding
    // rather than the default for anything this test did not set up.
    let nop = store_route_count("packet_class_control");
    let _ = process_child_packet(
        &mut state,
        &mut host,
        1,
        &Packet {
            opcode: CHILD_OP_NOP,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN,
            completion_stamp: 0,
            payload: Vec::new(),
            next_head: 0,
        },
    );
    assert_eq!(
        store_route_count("packet_class_control"),
        nop + 1,
        "an acknowledged no-op is a control command and not an unknown one"
    );
}

/// A lifetime command names a task, and the census tells a defined one from a
/// task this device made up on the spot.
///
/// The group's remaining gate. `Lifecycle::apply` refuses `NoSuchTask` for
/// eleven of the twelve commands, and the refusal loses the whole packet — so
/// what a driven boot has to say is that the guest never sends one. The unmap
/// below is the case that has no ref to resolve, which is why it is the one that
/// can reach the refusal at all: the eight commands that resolve something do it
/// through the same namespace the apply reads.
#[test]
fn a_lifetime_command_on_an_undefined_task_is_told_apart_from_one_on_a_defined_task() {
    use crate::model::{CHILD_OP_UNMAP_MEMORY, ROOT_OP_DEFINE_TASK2};
    use crate::protocol::fifo;
    use crate::runtime::drain::store_route_count;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    const TASK: u32 = 3;

    let mut notice = vec![0u8; fifo::MAP_MEMORY_LEN];
    notice[fifo::MAP_MEMORY_TASK_ID..fifo::MAP_MEMORY_TASK_ID + 4]
        .copy_from_slice(&TASK.to_le_bytes());
    notice[fifo::MAP_MEMORY_GVA..fifo::MAP_MEMORY_GVA + 8]
        .copy_from_slice(&0x0000_7f12_3456_1000u64.to_le_bytes());
    notice[fifo::MAP_MEMORY_LENGTH..fifo::MAP_MEMORY_LENGTH + 8]
        .copy_from_slice(&0x1000u64.to_le_bytes());
    let unmap = Packet {
        opcode: CHILD_OP_UNMAP_MEMORY,
        stamp_waits: Vec::new(),
        total_size: PACKET_HEADER_LEN + u32::try_from(notice.len()).expect("small"),
        completion_stamp: 0,
        payload: notice,
        next_head: 0,
    };

    let defined = store_route_count("lifetime_command_in_a_defined_task");
    let undefined = store_route_count("lifetime_command_in_an_undefined_task");
    let _ = process_child_packet(&mut state, &mut host, 1, &unmap);
    assert_eq!(
        (
            store_route_count("lifetime_command_in_a_defined_task"),
            store_route_count("lifetime_command_in_an_undefined_task"),
        ),
        (defined, undefined + 1),
        "no definition opened this task, so the lifecycle owner would refuse the packet"
    );

    let mut define = vec![0u8; fifo::DEFINE_TASK_LEN];
    define[..4].copy_from_slice(
        &fifo::DefineTaskId {
            task_id: TASK,
            kernel: false,
        }
        .to_raw()
        .to_le_bytes(),
    );
    define[fifo::DEFINE_TASK_DIRECTORY_PFN..fifo::DEFINE_TASK_DIRECTORY_PFN + 4]
        .copy_from_slice(&0x1000u32.to_le_bytes());
    process_root_packet(
        &mut state,
        &mut host,
        &Packet {
            opcode: ROOT_OP_DEFINE_TASK2,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + u32::try_from(define.len()).expect("small"),
            completion_stamp: 0,
            payload: define,
            next_head: 0,
        },
    );
    assert!(state.tasks.is_active(TASK), "the definition opened it");
    // The definition itself is not counted: it is the event that makes the task
    // hold, so asking whether it already held would count every first
    // definition as a refusal of a packet the model accepts.
    assert_eq!(
        (
            store_route_count("lifetime_command_in_a_defined_task"),
            store_route_count("lifetime_command_in_an_undefined_task"),
        ),
        (defined, undefined + 1),
        "a task definition is not a command against a task"
    );

    let _ = process_child_packet(&mut state, &mut host, 1, &unmap);
    assert_eq!(
        (
            store_route_count("lifetime_command_in_a_defined_task"),
            store_route_count("lifetime_command_in_an_undefined_task"),
        ),
        (defined + 1, undefined + 1),
        "and now the same packet names a task the model holds"
    );
}

/// A doorbell makes a domain live and a definition makes it *defined*, and the
/// census tells the two apart on a device where one mask holds both.
///
/// The distinction is the whole reason the field exists. `SessionModel::admit`
/// gates on the definition alone, so a boot that put work on doorbell-only
/// domains would be a boot this cutover hangs. The old `active_child_mask`
/// is the union of three events and cannot answer that after the fact.
#[test]
fn a_packet_on_a_doorbell_only_domain_is_told_apart_from_one_on_a_defined_domain() {
    use crate::runtime::drain::store_route_count;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let channel = 1u32;
    let bit = 1u32 << channel;

    // The lock-free ring: the domain becomes live with no definition anywhere.
    state
        .gfx
        .child_doorbell_rung
        .store(bit, std::sync::atomic::Ordering::Release);
    crate::runtime::drain::fold_rung_child_doorbells(&mut state);
    assert_eq!(
        state.pending.child_mask & bit,
        bit,
        "a ring makes the domain drainable"
    );
    assert_eq!(
        u32::from(state.child_domain_open(channel)) * bit,
        0,
        "and says nothing about a definition"
    );

    let defined = store_route_count("child_packet_domain_defined");
    let undefined = store_route_count("child_packet_domain_undefined");
    let nop = Packet {
        opcode: CHILD_OP_NOP,
        stamp_waits: Vec::new(),
        total_size: PACKET_HEADER_LEN,
        completion_stamp: 0,
        payload: Vec::new(),
        next_head: 0,
    };
    let _ = process_child_packet(&mut state, &mut host, channel, &nop);
    assert_eq!(
        (
            store_route_count("child_packet_domain_defined"),
            store_route_count("child_packet_domain_undefined"),
        ),
        (defined, undefined + 1),
        "the model would have refused this packet, and the census says so"
    );

    // Now the definition packet for the same domain.
    process_root_packet(
        &mut state,
        &mut host,
        &Packet {
            opcode: ROOT_OP_DEFINE_FIFO,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + 4,
            completion_stamp: 0,
            payload: channel.to_le_bytes().to_vec(),
            next_head: 0,
        },
    );
    assert_eq!(
        u32::from(state.child_domain_open(channel)) * bit,
        bit,
        "a definition is the one event the model's open-channel set has"
    );
    let _ = process_child_packet(&mut state, &mut host, channel, &nop);
    assert_eq!(
        (
            store_route_count("child_packet_domain_defined"),
            store_route_count("child_packet_domain_undefined"),
        ),
        (defined + 1, undefined + 1),
        "and now the same packet on the same domain is one the model admits"
    );

    // A free takes the definition back, because the model's set has that event
    // too: a domain whose positions have drained stops being nameable.
    process_root_packet(
        &mut state,
        &mut host,
        &Packet {
            opcode: ROOT_OP_FREE_FIFO,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + 4,
            completion_stamp: 0,
            payload: channel.to_le_bytes().to_vec(),
            next_head: 0,
        },
    );
    assert_eq!(
        u32::from(state.child_domain_open(channel)) * bit,
        0,
        "a free is not a definition that stands"
    );
}

/// FIFO redefine/free retires scheduler ownership so a removed producer
/// cannot strand later display transactions behind a stale bit.
#[test]
fn free_fifo_clears_translation_scheduler_state() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let bit = 1 << 1;
    state.open_child_domains_for_test(bit);
    state.pending.child_mask = bit;
    state.translation_deferred_mask = bit;
    state.translation_order_hold_mask = bit;
    state.present_translation_hold_mask = bit;

    process_root_packet(
        &mut state,
        &mut host,
        &Packet {
            opcode: ROOT_OP_FREE_FIFO,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + 4,
            completion_stamp: 0,
            payload: 1u32.to_le_bytes().to_vec(),
            next_head: 0,
        },
    );

    assert!(!state.child_domain_open(1));
    assert_eq!(state.pending.child_mask & bit, 0);
    assert_eq!(state.translation_deferred_mask & bit, 0);
    assert_eq!(state.translation_order_hold_mask & bit, 0);
    assert_eq!(state.present_translation_hold_mask & bit, 0);
}

/// First Composite present takes the leave-BAR1 boundary.
#[test]
fn composite_present_sets_frame_flush_boundary() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    assert!(state.map_surface(4));
    {
        let m = state.mappings.get_mut(&4).unwrap();
        m.mapped = true;
        m.has_geom = true;
        m.width = 1920;
        m.height = 1080;
        m.content_generation = 2;
        m.page_entries = vec![1];
    }
    state.note_surface_composite(4);

    let pkt = present_packet(CHILD_OP_DISPLAY_TRANSACTION2, 4);
    process_child_packet(&mut state, &mut host, 5, &pkt);
    assert!(state.present.frame_flush_seen);
    assert_coalesced_paint_action(&host, "composite sets flush boundary");
}

#[test]
fn display_swap_without_geom_holds_last_frame() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.present.valid = true;
    state.present.width = 1920;
    state.present.height = 1080;
    assert!(state.map_surface(9));
    // Mapped but no has_geom — do not resize/paint.
    let pkt = present_packet(CHILD_OP_DISPLAY_SWAP, 9);
    process_child_packet(&mut state, &mut host, 4, &pkt);
    assert!(state.present.frame_flush_seen);
    assert_eq!(state.present.present_mapping, 9);
    // Console size unchanged; no scanout HostAction.
    assert_eq!(state.present.width, 1920);
    assert_eq!(state.present.height, 1080);
    assert!(host.actions.is_empty());
}

#[test]
fn map_surface_clears_stale_geom() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    assert!(state.map_surface(1));
    assert!(state.set_mapping_geom(1, 1920, 1080, 0x73));
    assert!(state.mappings[&1].has_geom);
    assert!(state.map_surface(1));
    let m = &state.mappings[&1];
    assert!(!m.has_geom);
    assert_eq!(m.width, 0);
    assert_eq!(m.height, 0);
}

/// x86 Ventura/Tahoe display pipe: present opcode 6 paints like DisplaySwap.
#[test]
fn present_x86_op6_paints_surface_id_mapping() {
    use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x71u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    assert!(state.map_surface(5));
    {
        let m = state.mappings.get_mut(&5).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    assert!(state.set_mapping_geom(5, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    let px = [0x22u8; 16];
    assert!(write_bgra8(&mut state, &mut host, 5, &px, 8, 2, 2));

    process_child_packet(
        &mut state,
        &mut host,
        5,
        &present_packet(CHILD_OP_DISPLAY_TRANSACTION2, 5),
    );
    assert_eq!(state.present.present_mapping, 5);
    assert!(state.present.frame_flush_seen);
    assert!(state.present.frame_valid || state.present.frame_encode_pending);
    assert!(
        state.present.frame_valid || state.present.frame_encode_pending,
        "op6 present hands the window a frame (or defers to encode)"
    );
    assert_coalesced_paint_action(&host, "x86 op6 present");
}

/// qemu-shim: each accepted DisplaySwap with geom increments unpainted
/// presents; host paint clears the counter (entry-side backpressure).
#[test]
fn display_swap_unpainted_presents_counts_until_paint() {
    use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x70u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    assert!(state.map_surface(3));
    {
        let m = state.mappings.get_mut(&3).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    assert!(state.set_mapping_geom(3, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    let px = [0x11u8; 16];
    assert!(write_bgra8(&mut state, &mut host, 3, &px, 8, 2, 2));

    let swap = |state: &mut DeviceState, host: &mut FakeHost| {
        process_child_packet(state, host, 4, &present_packet(CHILD_OP_DISPLAY_SWAP, 3));
    };

    assert_eq!(state.present.unpainted_presents, 0);
    swap(&mut state, &mut host);
    assert_eq!(
        state.present.unpainted_presents, 1,
        "first accepted DisplaySwap counts as unpainted"
    );
    swap(&mut state, &mut host);
    assert_eq!(
        state.present.unpainted_presents, 2,
        "second accepted DisplaySwap reaches apple-gfx pending_frames cap"
    );
    // process_child_packet itself does not gate — drain_child_fifo does.
    // Counter keeps climbing if tests call process directly (stamp still fires).
    swap(&mut state, &mut host);
    assert_eq!(state.present.unpainted_presents, 3);
    note_present_paint_consumed(&mut state);
    assert_eq!(
        state.present.unpainted_presents, 0,
        "host paint clears entry-side present backpressure"
    );
    // Gate predicate used by drain_child_fifo.
    assert!(
        state.present.unpainted_presents < MAX_UNPAINTED_PRESENTS,
        "after paint, DisplaySwap entry is open"
    );
}

/// PVG present completion: every accepted DisplaySwap sets pending bit 1
/// on the display shared page and pokes the display IRQ when the guest
/// enable mask asks for present notifications (completion block after
/// +0x188 retain). ONLINE pending (bit 2) must be preserved (guest
/// read-clears the word).
#[test]
fn display_swap_signals_present_complete_on_shared_page() {
    use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x70u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    assert!(state.map_surface(3));
    {
        let m = state.mappings.get_mut(&3).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    assert!(state.set_mapping_geom(3, 2, 2, MTL_FORMAT_BGRA8_UNORM));

    // Display shared page with enable mask asking for present events;
    // a stale ONLINE pending bit must survive the present OR.
    let shared = 0x9000_0000u64;
    host.map_range(shared, 0x1000, 0);
    state.display.shared_gpa = shared;
    state.display.display_index = 0;
    host.put_u32(
        shared + DISPLAY_SHARED_ENABLE_MASK,
        DISPLAY_PRESENT_EVENT_MASK | DISPLAY_ONLINE_EVENT_MASK,
    );
    host.put_u32(shared + DISPLAY_SHARED_PENDING, DISPLAY_ONLINE_EVENT_MASK);

    process_child_packet(
        &mut state,
        &mut host,
        4,
        &present_packet(CHILD_OP_DISPLAY_SWAP, 3),
    );

    let mut le = [0u8; 4];
    assert!(host
        .read_gpa(shared + DISPLAY_SHARED_PENDING, &mut le)
        .is_ok());
    let pending = u32::from_le_bytes(le);
    assert_ne!(
        pending & DISPLAY_PRESENT_EVENT_MASK,
        0,
        "present completion must set pending bit 1"
    );
    assert_ne!(
        pending & DISPLAY_ONLINE_EVENT_MASK,
        0,
        "present completion must not clobber other pending events"
    );
    assert_ne!(
        state
            .gfx
            .interrupt_status_disp
            .load(std::sync::atomic::Ordering::Acquire)
            & 1,
        0,
        "display IRQ status must name display 0"
    );

    // Enable mask without the present bit: NEITHER the pending bit nor the IRQ.
    //
    // The pending half of this used to assert the opposite — that the bit is
    // written whether or not the guest asked for the class. The guest's own
    // interrupt handler is what makes that wrong: it read-clears
    // `pending & enable_mask` and leaves every other bit exactly where it found
    // it, so a bit set for a disabled class is never cleared by anyone. It is
    // then carried forward by every later read-modify-write of the word, which
    // is what a live macOS 13 guest showed — `+0x100` reading `0x3` against an
    // enable mask of `0xc`, both bits set by this device and neither wanted.
    state
        .gfx
        .interrupt_status_disp
        .store(0, std::sync::atomic::Ordering::Release);
    host.put_u32(shared + DISPLAY_SHARED_ENABLE_MASK, 0);
    host.put_u32(shared + DISPLAY_SHARED_PENDING, 0);
    signal_display_present_complete(&mut state, &mut host);
    assert!(host
        .read_gpa(shared + DISPLAY_SHARED_PENDING, &mut le)
        .is_ok());
    assert_eq!(
        u32::from_le_bytes(le) & DISPLAY_PRESENT_EVENT_MASK,
        0,
        "a present bit the guest disabled is a bit nothing will ever clear"
    );
    assert_eq!(
        state
            .gfx
            .interrupt_status_disp
            .load(std::sync::atomic::Ordering::Acquire),
        0,
        "no display IRQ when the guest did not ask for present events"
    );

    // An unreadable enable mask must not be read as permission. The guest
    // published this page's address itself, so a read of it the host cannot
    // perform is not a reason to start signalling classes nobody asked for.
    state.display.shared_gpa = 0xdead_0000_0000;
    signal_display_present_complete(&mut state, &mut host);
    assert_eq!(
        state
            .gfx
            .interrupt_status_disp
            .load(std::sync::atomic::Ordering::Acquire),
        0,
        "an unreadable enable mask must not authorise a present event"
    );
}

/// qemu-shim: entry gate holds when unpainted_presents >= MAX (apple-gfx
/// pending_frames >= 2). Stamp of accepted presents remains at retain.
#[test]
fn display_swap_entry_gated_when_unpainted_at_cap() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.present.unpainted_presents = MAX_UNPAINTED_PRESENTS;
    assert!(
        state.present.unpainted_presents >= MAX_UNPAINTED_PRESENTS,
        "drain_child_fifo must hold DisplaySwap when unpainted at cap"
    );
    note_present_paint_consumed(&mut state);
    assert!(
        state.present.unpainted_presents < MAX_UNPAINTED_PRESENTS,
        "paint re-opens DisplaySwap entry"
    );
}

/// One starvation episode per held present, and the present is named by its
/// ordering position rather than by a ring head — a declined present has no
/// ring position, because the head moved past it when it arrived.
#[test]
fn present_action_starvation_proxy_is_once_per_held_position() {
    use reims_vgpu_core::identity::IngressOrdinal;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    state.present.unpainted_presents = MAX_UNPAINTED_PRESENTS;

    note_present_backpressure_hold(&mut state, 5, IngressOrdinal(464));
    note_present_backpressure_hold(&mut state, 5, IngressOrdinal(464));
    assert_eq!(state.present.backpressure_hold_count, 1);

    note_present_paint_consumed(&mut state);
    state.present.unpainted_presents = MAX_UNPAINTED_PRESENTS;
    note_present_backpressure_hold(&mut state, 5, IngressOrdinal(464));
    assert_eq!(
        state.present.backpressure_hold_count, 2,
        "a later hold after paint is a distinct starvation episode"
    );
    // And a different position is a different episode without any paint
    // between, which a head-keyed record could not say for two presents that
    // arrived at the same ring offset on different laps.
    note_present_backpressure_hold(&mut state, 5, IngressOrdinal(465));
    assert_eq!(state.present.backpressure_hold_count, 3);
}

#[test]
fn child_drain_yields_after_present_for_display_consumer() {
    use crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let page_size = state.page_size() as usize;
    let channel = 5u32;
    let root_pfn = 0x10u32;
    let list_pfn = 0x20u32;
    let ring_pfn = 0x30u32;
    let stamp_pfn = 0x40u32;
    let root_gpa = state.pfn_gpa(root_pfn);
    let list_gpa = state.pfn_gpa(list_pfn);
    let ring_gpa = state.pfn_gpa(ring_pfn);
    let stamp_gpa = state.pfn_gpa(stamp_pfn);
    for gpa in [root_gpa, list_gpa, ring_gpa, stamp_gpa] {
        host.map_range(gpa, page_size, 0);
    }

    assert!(state.map_surface(4));
    assert!(state.set_mapping_geom(4, 2, 2, MTL_FORMAT_BGRA8_UNORM));

    let form = PresentForm::Transaction2;
    let mut payload = vec![0u8; form.trailer_len()];
    payload[form.target_offset()..form.target_offset() + 4].copy_from_slice(&4u32.to_le_bytes());
    let first = packet_bytes(CHILD_OP_DISPLAY_TRANSACTION2, 21, &payload);
    let second = packet_bytes(CHILD_OP_DISPLAY_TRANSACTION2, 22, &payload);
    let mut ring = first.clone();
    ring.extend_from_slice(&second);
    host.write_gpa(ring_gpa, &ring).unwrap();
    host.put_u32(list_gpa, ring_pfn);

    let regs_gpa = root_gpa + child_reg_block_offset(channel).unwrap();
    host.put_u32(regs_gpa + CHILD_REG_TAIL, ring.len() as u32);
    host.put_u32(regs_gpa + CHILD_REG_HEAD, 0);
    host.put_u32(regs_gpa + CHILD_REG_STAMP_INDEX, 1);
    host.put_u32(regs_gpa + CHILD_REG_BASE_PFN, list_pfn);
    state.gfx.root_page = root_pfn;
    state.gfx.fifo_base_page = stamp_pfn;
    state.open_child_domains_for_test(1u32 << channel);
    state.pending.child_mask = 1u32 << channel;

    drain_pending(&mut state, &mut host);
    assert_eq!(
        host.get_u32(regs_gpa + CHILD_REG_HEAD),
        first.len() as u32,
        "the first drain slice must stop after accepting one present"
    );
    assert_ne!(state.pending.child_mask & (1u32 << channel), 0);
    assert_eq!(state.present.unpainted_presents, 1);
    assert!(
        state.pending.host_action_yield,
        "an accepted present must end the drain slice"
    );
    assert_coalesced_paint_action(&host, "first present");

    // The ack. `device_drain` calls this itself after publishing the frame to
    // the host window; it used to arrive from QEMU's DisplaySurface paint
    // after the lock was released. Either way it reopens the queued channel
    // for its next ordered packet — that contract is what this test locks.
    note_present_paint_consumed(&mut state);
    host.actions.clear();
    drain_pending(&mut state, &mut host);
    assert_eq!(host.get_u32(regs_gpa + CHILD_REG_HEAD), ring.len() as u32);
    assert_eq!(state.present.unpainted_presents, 1);
    assert_coalesced_paint_action(&host, "second present after ack");
    assert_eq!(host.get_u32(stamp_gpa + 4), 22);
}

/// Mode switch (1920→1440) is a new surface identity: reset
/// content_generation (Load/scanout semantics restart).
#[test]
fn set_mapping_geom_size_change_resets_content_generation() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    assert!(state.map_surface(4));
    assert!(state.set_mapping_geom(4, 1920, 1080, 0x73));
    {
        let m = state.mappings.get_mut(&4).unwrap();
        m.content_generation = 42;
    }
    assert!(state.set_mapping_geom(4, 1440, 1080, 0x73));
    let m = &state.mappings[&4];
    assert_eq!(m.width, 1440);
    assert_eq!(m.height, 1080);
    assert_eq!(
        m.content_generation, 0,
        "new size must not keep prior gen (new surface identity)"
    );
    // Same declaration again: preserve gen (no identity change).
    //
    // The format is held at `0x73` here on purpose. This line used to re-declare
    // at `0x50` while asserting only that the *size* was unchanged, so it was
    // reading a format change as "nothing changed" — and the reset it was
    // pinning tested two thirds of the declaration. A format change withdraws
    // the claim too, and that axis is covered by
    // `model::state::mapping_declaration_tests`.
    {
        let m = state.mappings.get_mut(&4).unwrap();
        m.content_generation = 3;
    }
    assert!(state.set_mapping_geom(4, 1440, 1080, 0x73));
    assert_eq!(
        state.mappings[&4].content_generation, 3,
        "an unchanged declaration preserves generation"
    );
}

/// Archive render_wait_surface helper: no rings → no-op, no panic.
#[test]
fn drain_other_child_fifos_is_a_safe_noop_without_rings() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.open_child_domains_for_test((1 << 1) | (1 << 4));
    state.pending.child_mask = 1 << 1;
    state.gfx.control_fifo = 1;
    // No root_page / rings: the drain returns immediately.
    drain_other_child_fifos(&mut state, &mut host, 4);
    assert_eq!(
        state.pending.child_mask, 0,
        "the sibling drain consumes the pending mask"
    );
}

#[test]
fn poll_rescue_only_publishes_work_for_async_drain() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.gfx.control_fifo = 0x1000;
    state
        .gfx
        .fifo_read
        .store(3, std::sync::atomic::Ordering::Release);
    state.gfx.fifo_written = 4;
    state.open_child_domains_for_test((1 << 2) | (1 << 5));

    assert!(publish_stranded_fifos(&mut state, &mut host));
    assert!(state.pending.main_drain);
    assert_eq!(state.pending.child_mask, (1 << 2) | (1 << 5));
    assert!(host.bh_scheduled);
    assert_eq!(
        state
            .gfx
            .fifo_read
            .load(std::sync::atomic::Ordering::Acquire),
        3,
        "poll context must not drain"
    );
}

/// Archive render_wait_surface: no inflight async job for mapping ⇒ no-op,
/// returns current content_generation. Does not drain other FIFOs.
#[test]
fn wait_surface_noop_when_no_async_job() {
    use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x22u32;
    host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
    assert!(state.map_surface(7));
    {
        let m = state.mappings.get_mut(&7).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    assert!(state.set_mapping_geom(7, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    assert!(write_bgra8(
        &mut state,
        &mut host,
        7,
        &[0x55u8; 16],
        8,
        2,
        2
    ));
}

/// qemu-shim dual-mid: incomplete last_store on one mid (logo/partial)
/// must fire thrash `nz_swing` when DisplaySwap alternates full vs sparse.
/// Regression gate for P1 dual-mid flicker (measure before fix).
#[test]
fn display_swap_encodes_at_present_after_wait_surface() {
    use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;
    use crate::runtime::scanout::{copy_to_bgra8, ScanoutCopyResult};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x21u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    assert!(state.map_surface(3));
    {
        let m = state.mappings.get_mut(&3).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
    }
    assert!(state.set_mapping_geom(3, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    let px = [
        0x11u8, 0x22, 0x33, 0xFF, 0x11, 0x22, 0x33, 0xFF, 0x11, 0x22, 0x33, 0xFF, 0x11, 0x22, 0x33,
        0xFF,
    ];
    assert!(write_bgra8(&mut state, &mut host, 3, &px, 8, 2, 2));
    let gen = state.mappings.get(&3).unwrap().content_generation;
    let pkt = present_packet(CHILD_OP_DISPLAY_SWAP, 3);
    process_child_packet(&mut state, &mut host, 4, &pkt);
    assert!(state.present.frame_flush_seen);
    assert!(
        state.present.frame_valid,
        "DisplaySwap freezes surface at present after wait_surface"
    );
    // Capture forces one host blit of +0x188 (encode_pending) so early
    // painted mid/gen cannot Unchanged-skip logo/pill onto frozen EFI.
    assert!(
        state.present.frame_encode_pending,
        "successful capture must force first paint of retain"
    );
    assert_eq!(state.present.frame_mapping, 3);
    assert_eq!(presented_mapping(&state), Some(3));
    assert_coalesced_paint_action(&host, "encode at present");
    // Host paint re-shows frozen snapshot.
    let mut dst = vec![0u8; 16];
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, 3, &mut dst, 8, 2, 2, gen),
        ScanoutCopyResult::Painted
    );
    assert_eq!(&dst[..], &px[..]);
    assert!(!state.present.frame_encode_pending);

    // Guest mutates mapping after stamp (recycle) — re-show still frozen.
    let mut_px = [0xAAu8; 16];
    assert!(write_bgra8(&mut state, &mut host, 3, &mut_px, 8, 2, 2));
    state.present.painted_generation = 0;
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, 3, &mut dst, 8, 2, 2, gen),
        ScanoutCopyResult::Painted
    );
    assert_eq!(
        &dst[..],
        &px[..],
        "post-stamp guest writes must not change retained present frame"
    );
}

/// qemu-shim: DisplaySwap capture fail must not drop PGDisplay +0x188 retain.
/// hostPresentCount re-shows the last successful present until capture works.
#[test]
fn display_swap_capture_fail_keeps_prior_retain() {
    use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;
    use crate::runtime::scanout::{copy_to_bgra8, ScanoutCopyResult};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x40u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    assert!(state.map_surface(5));
    {
        let m = state.mappings.get_mut(&5).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    assert!(state.set_mapping_geom(5, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    let full = [
        0x11u8, 0x22, 0x33, 0xFF, 0x44, 0x55, 0x66, 0xFF, 0xAA, 0, 0, 0xFF, 0xAA, 0, 0, 0xFF,
    ];
    assert!(write_bgra8(&mut state, &mut host, 5, &full, 8, 2, 2));

    let swap = |state: &mut DeviceState, host: &mut FakeHost, mid: u32| {
        host.actions.clear();
        process_child_packet(state, host, 4, &present_packet(CHILD_OP_DISPLAY_SWAP, mid));
    };

    // First swap: full dock composite retained.
    swap(&mut state, &mut host, 5);
    assert!(state.present.frame_valid);
    assert_eq!(state.present.frame_mapping, 5);
    let gen_ok = state.present.frame_generation;
    let mut dst = vec![0u8; 16];
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, 5, &mut dst, 8, 2, 2, gen_ok),
        ScanoutCopyResult::Painted
    );
    assert_eq!(&dst[..], &full[..]);

    // Second swap: pages unreadable + host-cache gone → capture fails.
    {
        let m = state.mappings.get_mut(&5).unwrap();
        m.page_entries.clear();
        // Bump gen so HostAction is distinct; guest would still name mid 5.
        m.content_generation = gen_ok + 1;
    }
    crate::runtime::surface_cache::forget(&mut state, 5);
    swap(&mut state, &mut host, 5);
    assert!(
        state.present.frame_encode_pending,
        "capture fail must set pending retry"
    );
    assert!(
        state.present.frame_valid,
        "PGDisplay +0x188 prior retain must survive capture fail"
    );
    assert_eq!(
        state.present.frame_mapping, 5,
        "prior retain mapping unchanged"
    );
    assert_eq!(
        presented_mapping(&state),
        Some(5),
        "window still shows the prior retain after a capture fail"
    );
    assert_coalesced_paint_action(&host, "capture fail keeps prior retain");
    // hostPresentCount / HostAction still shows the last good full composite.
    state.present.painted_generation = 0;
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, 5, &mut dst, 8, 2, 2, gen_ok + 1),
        ScanoutCopyResult::Painted
    );
    assert_eq!(
        &dst[..],
        &full[..],
        "capture-fail DisplaySwap must re-show prior full retain, not black/empty"
    );
}

/// qemu-shim dual-mid: each CmdDisplaySwap freezes that mid's **full** guest
/// composite (dock strip pattern); hostPresentCount re-shows the latest
/// retain only — never mixes mid A partial with mid B full.
#[test]
fn display_swap_dual_mid_full_composites_both_retain() {
    use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;
    use crate::runtime::scanout::{copy_to_bgra8, ScanoutCopyResult};

    // 4×2 BGRA: row0 = "dock" strip (distinct L/R icons), row1 = wallpaper.
    fn frame(left: [u8; 4], right: [u8; 4], wall: [u8; 4]) -> Vec<u8> {
        let mut v = Vec::with_capacity(32);
        v.extend_from_slice(&left);
        v.extend_from_slice(&right);
        v.extend_from_slice(&wall);
        v.extend_from_slice(&wall);
        v
    }
    let full_a = frame(
        [0x11, 0x22, 0x33, 0xFF],
        [0x44, 0x55, 0x66, 0xFF],
        [0xAA, 0x00, 0x00, 0xFF],
    );
    let full_b = frame(
        [0x77, 0x88, 0x99, 0xFF],
        [0xBB, 0xCC, 0xDD, 0xFF],
        [0x00, 0xAA, 0x00, 0xFF],
    );
    // Partial dock: left icons only, right = wallpaper (residual as-t4 shape).
    let partial_b = frame(
        [0x77, 0x88, 0x99, 0xFF],
        [0x00, 0xAA, 0x00, 0xFF],
        [0x00, 0xAA, 0x00, 0xFF],
    );

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    for (mid, pfn) in [(3u32, 0x30u32), (4u32, 0x31u32)] {
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    }
    assert!(write_bgra8(&mut state, &mut host, 3, &full_a, 8, 2, 2));
    assert!(write_bgra8(&mut state, &mut host, 4, &partial_b, 8, 2, 2));

    let swap = |state: &mut DeviceState, host: &mut FakeHost, mid: u32| {
        host.actions.clear();
        process_child_packet(state, host, 4, &present_packet(CHILD_OP_DISPLAY_SWAP, mid));
    };

    // Present mid3 full dock → +0x188.
    swap(&mut state, &mut host, 3);
    assert!(state.present.frame_valid);
    assert_eq!(state.present.frame_mapping, 3);
    let mut dst = vec![0u8; 16];
    let gen3 = state.present.frame_generation;
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, 3, &mut dst, 8, 2, 2, gen3),
        ScanoutCopyResult::Painted
    );
    assert_eq!(
        &dst[..],
        &full_a[..],
        "mid3 present freezes full dock into +0x188"
    );

    // Present mid4 while guest still has partial dock on mid4 (overwrites +0x188).
    swap(&mut state, &mut host, 4);
    assert_eq!(
        state.present.frame_mapping, 4,
        "DisplaySwap must re-retain mid4"
    );
    let gen4p = state.present.frame_generation;
    state.present.painted_generation = 0; // force paint (same gen as mid3 possible)
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, 4, &mut dst, 8, 2, 2, gen4p),
        ScanoutCopyResult::Painted
    );
    assert_eq!(
        &dst[..],
        &partial_b[..],
        "mid4 present shows guest partial until full composite lands"
    );
    // Late HostAction for mid3: encodeCurrentFrame shows current +0x188
    // (mid4 partial), not a mid3 backlog and not live mid3 if recycled.
    state.present.painted_generation = 0;
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, 3, &mut dst, 8, 2, 2, gen3),
        ScanoutCopyResult::Painted
    );
    assert_eq!(
        &dst[..],
        &partial_b[..],
        "late mid3 HostAction paints current +0x188 (mid4)"
    );

    // Guest finishes full dock on mid4; DisplaySwap freezes full composite.
    assert!(write_bgra8(&mut state, &mut host, 4, &full_b, 8, 2, 2));
    swap(&mut state, &mut host, 4);
    let gen4 = state.present.frame_generation;
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, 4, &mut dst, 8, 2, 2, gen4),
        ScanoutCopyResult::Painted
    );
    assert_eq!(
        &dst[..],
        &full_b[..],
        "mid4 after full composite: both dock L/R present"
    );
    // Live mid3 rewrite must not affect +0x188 hostPresentCount re-show.
    assert!(write_bgra8(&mut state, &mut host, 3, &full_a, 8, 2, 2));
    state.present.painted_generation = 0;
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, 3, &mut dst, 8, 2, 2, gen3),
        ScanoutCopyResult::Painted
    );
    assert_eq!(
        &dst[..],
        &full_b[..],
        "+0x188 still mid4 full after live mid3 page writes"
    );
}

/// Double-buffer present: alternating DisplaySwap mapping ids each paint
/// the named surface (guest mid3/mid4). Both composites land independently.
#[test]
fn display_swap_alternating_mappings_both_paint() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    for mid in [3u32, 4u32] {
        assert!(state.map_surface(mid));
        let m = state.mappings.get_mut(&mid).unwrap();
        m.mapped = true;
        m.has_geom = true;
        m.width = 1440;
        m.height = 1080;
        m.content_generation = mid * 10;
        m.page_entries = vec![1];
    }
    for mid in [3u32, 4u32, 3u32] {
        host.actions.clear();
        let pkt = present_packet(CHILD_OP_DISPLAY_SWAP, mid);
        process_child_packet(&mut state, &mut host, 4, &pkt);
        assert!(state.present.frame_flush_seen);
        assert_eq!(state.present.present_mapping, mid);
        assert_eq!(state.present.width, 1440);
        assert_eq!(state.present.height, 1080);
        assert_coalesced_paint_action(&host, "alternating mappings");
    }
}

/// qemu-shim present contract: only CmdDisplaySwap (ch4 op8) paints after
/// the first frame boundary — writebacks and ch2 present-into-mid must not.
#[test]
fn only_display_swap_paints_after_frame_flush_seen() {
    use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::protocol::pixel_format::MTL_FORMAT_RGBA16_FLOAT;
    use crate::runtime::scanout::note_front_buffer_writeback;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.present.frame_flush_seen = true;
    state.present.valid = true;
    state.present.width = 1440;
    state.present.height = 1080;
    state.present.present_mapping = 3;
    state.present.host_mapping = 3;
    // Back buffer mid=4 writeback (compositor composite into non-front).
    assert!(state.map_surface(4));
    {
        let m = state.mappings.get_mut(&4).unwrap();
        m.mapped = true;
        m.has_geom = true;
        m.width = 1440;
        m.height = 1080;
        m.format = MTL_FORMAT_RGBA16_FLOAT;
        m.content_generation = 9;
        m.page_entries = vec![(1u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    note_front_buffer_writeback(
        &mut state,
        &mut host,
        4,
        1440,
        1080,
        MTL_FORMAT_RGBA16_FLOAT,
    );
    assert!(
        host.actions.is_empty(),
        "post-boundary writeback must not paint"
    );
    assert_eq!(
        state.present.present_mapping, 3,
        "writeback must not rename presented mid after DisplaySwap"
    );

    // DisplaySwap with geom → paint named mapping.
    assert!(state.map_surface(5));
    {
        let m = state.mappings.get_mut(&5).unwrap();
        m.mapped = true;
        m.has_geom = true;
        m.width = 1440;
        m.height = 1080;
        m.content_generation = 10;
        m.page_entries = vec![(2u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    let pkt = present_packet(CHILD_OP_DISPLAY_SWAP, 5);
    process_child_packet(&mut state, &mut host, 4, &pkt);
    assert_coalesced_paint_action(&host, "post-flush display swap");
    assert_eq!(state.present.present_mapping, 5);
}

#[test]
fn display_online_waits_for_enable_mask_then_signals() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let gpa = 0x7b000000u64;
    host.map_range(gpa, PAGE_SIZE_ARM64E as usize, 0);
    state.display.shared_gpa = gpa;
    state.display.display_index = 0;
    // No enable mask yet — even after divisor ticks, no IRQ.
    state.display.poll_ctr = DISPLAY_ONLINE_POLL_DIVISOR - 1;
    try_display_online(&mut state, &mut host);
    assert!(host.actions.is_empty());
    assert_eq!(state.display.online_tries, 0);
    // Guest enable() published bit 2.
    let mut m = [0u8; 4];
    st32(&mut m, DISPLAY_ONLINE_EVENT_MASK);
    host.write_gpa(gpa + DISPLAY_SHARED_ENABLE_MASK, &m)
        .unwrap();
    state.display.poll_ctr = DISPLAY_ONLINE_POLL_DIVISOR - 1;
    try_display_online(&mut state, &mut host);
    assert_eq!(host.actions.len(), 1);
    assert_eq!(host.actions[0].kind, HostActionKind::IrqGfxPulse);
    assert_eq!(state.display.online_tries, 1);
    let mut pending = [0u8; 4];
    host.read_gpa(gpa + DISPLAY_SHARED_PENDING, &mut pending)
        .unwrap();
    assert_eq!(ld32(&pending), DISPLAY_ONLINE_EVENT_MASK);
    // After ack, no more asserts.
    state.display.online_acked = true;
    host.actions.clear();
    state.display.poll_ctr = DISPLAY_ONLINE_POLL_DIVISOR - 1;
    try_display_online(&mut state, &mut host);
    assert!(host.actions.is_empty());
}

/// The first ONLINE pulse goes on the poll that first sees the enable bit, and
/// does not wait out the re-assert cadence.
///
/// The guest is racing us for it. macOS's WindowServer asks
/// `AppleParavirtFramebuffer` for a mappable VRAM aperture ~13 s into boot and
/// aborts if it does not get one, taking the desktop with it for the life of the
/// boot. This device used to gate the *enable-mask read* on the same divisor
/// that paces the re-assert, so `apply_setup_shared_state` zeroing `poll_ctr`
/// bought the handshake a full divisor of latency — measured at 2139-2365 ms
/// across three boots, which is most of the margin WindowServer leaves.
///
/// So this is the ordering assertion and not a timing one: with the enable bit
/// already set, one poll is enough. It fails on the previous code, which needed
/// fifty.
#[test]
fn the_first_online_pulse_does_not_wait_out_the_reassert_cadence() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let gpa = 0x7b000000u64;
    host.map_range(gpa, PAGE_SIZE_ARM64E as usize, 0);
    state.display.shared_gpa = gpa;
    state.display.display_index = 0;
    let mut m = [0u8; 4];
    st32(&mut m, DISPLAY_ONLINE_EVENT_MASK);
    host.write_gpa(gpa + DISPLAY_SHARED_ENABLE_MASK, &m)
        .unwrap();

    // Exactly the state `apply_setup_shared_state` leaves behind, and then one
    // single poll.
    state.display.poll_ctr = 0;
    state.display.online_tries = 0;
    try_display_online(&mut state, &mut host);

    assert_eq!(
        state.display.online_tries, 1,
        "the enable bit was set, so the first poll should have pulsed ONLINE \
         rather than waiting for poll {DISPLAY_ONLINE_POLL_DIVISOR}"
    );
    assert_eq!(host.actions.len(), 1);
    assert_eq!(host.actions[0].kind, HostActionKind::IrqGfxPulse);

    // And the re-assert is still paced: the very next poll is not a multiple of
    // the divisor, so it must not pulse again.
    host.actions.clear();
    try_display_online(&mut state, &mut host);
    assert!(
        host.actions.is_empty(),
        "the divisor still paces every pulse after the first"
    );
    assert_eq!(state.display.online_tries, 1);
}

/// Display-lifecycle instrumentation: SETUP_SHARED_STATE, ONLINE ack, and the
/// first ONLINE signal each leave an always-on line so a bad boot has a
/// display-lifecycle timeline to correlate with post_converge_regress. A
/// SETUP_SHARED_STATE while already ONLINE logs reinit=1 — the post-converge
/// display rebuild that is the standing overlay lead.
#[test]
fn display_lifecycle_events_are_always_logged() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let index = 0u32;
    let pfn = 0x7bu32;
    let gpa = state.pfn_gpa(pfn);
    host.map_range(gpa, PAGE_SIZE_ARM64E as usize, 0);

    let mut payload = vec![0u8; CHILD_SHARED_STATE_LEN];
    payload[CHILD_SHARED_STATE_INDEX..CHILD_SHARED_STATE_INDEX + 4]
        .copy_from_slice(&index.to_le_bytes());
    payload[CHILD_SHARED_STATE_PFN..CHILD_SHARED_STATE_PFN + 4].copy_from_slice(&pfn.to_le_bytes());
    let setup = Packet {
        opcode: CHILD_OP_SETUP_SHARED_STATE,
        stamp_waits: Vec::new(),
        total_size: PACKET_HEADER_LEN + CHILD_SHARED_STATE_LEN as u32,
        completion_stamp: 0,
        payload,
        next_head: 0,
    };

    // First setup: reinit=0 (initial display registration).
    process_child_packet(&mut state, &mut host, 4, &setup);
    // Guest ack.
    let ack = Packet {
        opcode: CHILD_OP_ONLINE_ACK,
        stamp_waits: Vec::new(),
        total_size: PACKET_HEADER_LEN,
        completion_stamp: 0,
        payload: vec![],
        next_head: 0,
    };
    process_child_packet(&mut state, &mut host, 4, &ack);
    assert!(state.display.online_acked);
    // First ONLINE signal (guest published enable bit 2).
    let mut m = [0u8; 4];
    st32(&mut m, DISPLAY_ONLINE_EVENT_MASK);
    host.write_gpa(gpa + DISPLAY_SHARED_ENABLE_MASK, &m)
        .unwrap();
    state.display.online_acked = false;
    state.display.poll_ctr = DISPLAY_ONLINE_POLL_DIVISOR - 1;
    try_display_online(&mut state, &mut host);

    // Second setup while previously ONLINE: reinit=1 (the post-converge rebuild).
    state.display.online_acked = true;
    process_child_packet(&mut state, &mut host, 4, &setup);

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(
        log.contains(&format!(
            "display_shared_state_setup index={index} gpa={gpa:#x} reinit=0"
        )),
        "initial setup must log reinit=0"
    );
    assert!(
        log.contains(&format!(
            "display_shared_state_setup index={index} gpa={gpa:#x} reinit=1"
        )),
        "re-setup while ONLINE must log reinit=1"
    );
    assert!(
        log.contains(&format!("display_online_ack index={index}")),
        "ONLINE ack must be logged"
    );
    assert!(
        log.contains(&format!("display_online_signal index={index}")),
        "first ONLINE signal must be logged"
    );
}

/// Both ways the ONLINE handshake can end without a display say so.
///
/// The rail has one success line (`display_online_signal`, on the first pulse)
/// and two silent exits, and the silent ones are the states a user actually
/// notices: a desktop that never appears. Neither is expected control flow —
/// the guest published the shared page in both cases — so both are on the fail
/// channel, and both are latched because each recurs on every poll for the rest
/// of the boot.
///
/// The third exit, `mask & ONLINE == 0`, has its own test below. It used to be
/// described here as "deliberately absent from this test and from the log",
/// and the second half of that stopped being true when it gained a bounded
/// report of its own.
#[test]
fn the_two_ways_online_gives_up_are_both_fail_visible() {
    let index = 3u32;

    // Exhaustion. This case sets `online_tries` to the cap directly, and what
    // that models is a guest that **enabled** the display and acked none of the
    // 150 ONLINE pulses — not, as this comment used to say, one that never
    // enabled. The increment sits past the enable-mask check, so a guest that
    // never enables cannot reach this branch at all, which is what
    // `a_guest_that_never_enables_cannot_reach_the_online_cap` pins.
    {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let gpa = 0x7c000000u64;
        host.map_range(gpa, PAGE_SIZE_ARM64E as usize, 0);
        state.display.shared_gpa = gpa;
        state.display.display_index = index;
        state.display.online_tries = DISPLAY_ONLINE_MAX_TRIES;
        try_display_online(&mut state, &mut host);
        assert!(
            host.actions.is_empty(),
            "past the cap nothing is asserted; only the reason is new"
        );
    }

    // The enable mask itself unreadable: every try is spent against nothing,
    // and at the cap that is indistinguishable from the case above.
    //
    // The address is chosen so the 4-byte read walks off the end of the address
    // space and `read_gpa` answers `MemError::Overflow`. `FakeHost` returns
    // zeroes for an address that is merely unmapped, which is a *readable* mask
    // with no bits set — the third exit, not this one — so an unmapped GPA
    // would test the wrong thing.
    let unreadable = u64::MAX - DISPLAY_SHARED_ENABLE_MASK - 1;
    {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        state.display.shared_gpa = unreadable;
        state.display.display_index = index;
        state.display.poll_ctr = DISPLAY_ONLINE_POLL_DIVISOR - 1;
        try_display_online(&mut state, &mut host);
        assert!(host.actions.is_empty());
        assert_eq!(
            state.display.online_tries, 0,
            "an unreadable mask is not an assert"
        );
    }

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(
        log.contains(&format!(
            "display_online_abandoned index={index} tries={DISPLAY_ONLINE_MAX_TRIES}"
        )),
        "giving up on ONLINE for the rest of the boot must name itself"
    );
    assert!(
        log.contains(&format!(
            "display_online_mask_unreadable gpa={:#x}",
            unreadable + DISPLAY_SHARED_ENABLE_MASK
        )),
        "an unreadable enable mask must not look like a guest that declined"
    );
}

/// A guest that publishes the display shared page and never enables it cannot
/// reach the ONLINE cap, and now says so on its own.
///
/// This is the case `display_online_abandoned` was worded as covering and
/// structurally cannot: `online_tries` is incremented at the tail of
/// `try_display_online`, past the enable-mask check, so a guest sitting at that
/// check leaves the counter at zero however long it polls. Before the report
/// below it emitted nothing at all — no `signal`, no `abandoned` — so the state
/// a user sees as a black screen was the one state with a clean log.
///
/// Both halves are asserted because each fails differently. The counter staying
/// at zero is what makes the *old* wording impossible; the line appearing is
/// what makes the state visible. A fix that only did the second would leave the
/// abandon line still able to mean two things.
#[test]
fn a_guest_that_never_enables_cannot_reach_the_online_cap() {
    // Its own display index: the report is latched per index by `first_sight`,
    // so sharing one with another test would consume the latch and leave this
    // asserting on a line some earlier test emitted.
    let index = 21u32;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let gpa = 0x7d000000u64;
    // Mapped and readable, and left as zeroes: a readable mask with the enable
    // bit clear is exactly the state under test, and an *unmapped* page would
    // take the unreadable arm instead.
    host.map_range(gpa, PAGE_SIZE_ARM64E as usize, 0);
    state.display.shared_gpa = gpa;
    state.display.display_index = index;

    // Past the reporting bound, so the next acted poll is the one that reports.
    // Poll the full divisor so the cadence gate is crossed rather than assumed.
    state.display.poll_ctr = DISPLAY_ONLINE_MAX_TRIES * DISPLAY_ONLINE_POLL_DIVISOR + 1;
    for _ in 0..DISPLAY_ONLINE_POLL_DIVISOR {
        try_display_online(&mut state, &mut host);
    }

    assert_eq!(
        state.display.online_tries, 0,
        "no ONLINE pulse was sent, so the cap can never be reached this way — \
         which is why the abandon line cannot mean what it used to say"
    );
    assert!(
        host.actions.is_empty(),
        "and nothing was asserted at the guest"
    );

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(
        log.contains(&format!("display_online_never_enabled index={index}")),
        "a display that never comes online must not leave a clean log"
    );
    assert!(
        !log.contains(&format!("display_online_abandoned index={index}")),
        "and it must not borrow the other exit's name"
    );
}

/// The VBL census reports the delivered rate, and separates the two ways it can
/// deliver nothing.
///
/// VBL paces the guest compositor, so the rate we deliver caps guest frame rate
/// however fast the present path is — and nothing measured it: a driven boot
/// emitted zero lines matching `vbl` anywhere in the always-on channel. The
/// three properties that make the new line readable are asserted here, because
/// each one is a way the reading could have been wrong:
///
/// - only deliveries report, so the line's cadence is the thing it measures;
/// - the rate is over the window since the last report, not the process
///   lifetime, so an early stall does not depress it forever;
/// - `not_online` and `not_claimed` stay separate, because "the display never
///   came up" and "the 8 ms limiter is working correctly at 125 Hz" are opposite
///   conclusions from the same low delivered count.
///
/// The head of each arm reports at the finer `VBL_REPORT_EARLY` cadence, so the
/// first assertion below counts sixteen lines rather than one. Both cadences
/// measure the window they cover, which is what keeps the rate comparable across
/// the boundary between them.
#[test]
fn the_vbl_census_reports_window_rate_and_separates_the_silent_arms() {
    use crate::runtime::drain::{VblCensus, VBL_DELIVERED, VBL_NOT_CLAIMED, VBL_NOT_ONLINE};
    let c = VblCensus::default();

    // The silent arms never report, however many times they are taken.
    for i in 0..5000u64 {
        assert!(c.note(VBL_NOT_ONLINE, i).is_none());
        assert!(c.note(VBL_NOT_CLAIMED, i).is_none());
    }

    // 1024 deliveries at the 8 ms grid. The head of an arm reports every 64 so
    // the display-link latch window is visible at all, so that is 16 lines, and
    // each covers 64 deliveries over 512 ms — still the grid rate, because the
    // window and the step are the same quantity.
    let mut lines = Vec::new();
    for i in 1..=1024u64 {
        if let Some(l) = c.note(VBL_DELIVERED, i * 8) {
            lines.push(l);
        }
    }
    assert_eq!(
        lines.len(),
        16,
        "one report per 64 deliveries over the head"
    );
    assert!(
        lines.iter().all(|l| l.contains("window_hz=125.0")),
        "every early window is 64 deliveries over 512 ms: {lines:?}"
    );
    let line = lines.last().expect("the head reports");
    assert!(line.contains("delivered=1024"), "{line}");
    assert!(
        line.contains("not_online=5000") && line.contains("not_claimed=5000"),
        "the silent arms must stay separable and counted: {line}"
    );

    // A second window at half the rate must read half, not an average dragged
    // toward the first window — this is the property that makes a live reading
    // of a *current* stall possible at all.
    let base = 1024 * 8;
    let mut second = None;
    for i in 1..=1024u64 {
        if let Some(l) = c.note(VBL_DELIVERED, base + i * 16) {
            second = Some(l);
        }
    }
    let second = second.expect("second window must report");
    assert!(second.contains("delivered=2048"), "{second}");
    assert!(
        second.contains("window_hz=62.5"),
        "the window rate must not be a lifetime average: {second}"
    );
}

/// The drain-duty census answers "is the worker saturated, and by which phase",
/// which requires three properties the return value can be asserted on:
///
/// - duty is busy time over *elapsed* time, so a worker holding the lock for
///   most of the window reads near 1 and an idle one reads near 0 — the two
///   readings that point at opposite halves of the ~2 Hz question;
/// - the two phases stay separate, because "guest work is slow" and "our export
///   is slow" are different fixes drawn from the same high duty;
/// - each report resets the window, so a live reading tracks the current stall
///   instead of a lifetime average.
#[test]
fn the_drain_duty_census_reads_a_rate_over_its_window_and_splits_the_two_phases() {
    use crate::runtime::drain::{DrainDutyCensus, DrainPhase, FlushRail};
    let c = DrainDutyCensus::default();

    // The first call only arms the window: reporting here would divide the whole
    // pre-drain idle stretch into one tranche and read an absurd duty. Its own
    // work is still counted — it is real time the worker spent — so it lands in
    // the window it opens.
    assert!(c.note(0, 0, 5_000).is_none(), "first call arms only");

    // A saturated second: ten 90 ms tranches, 60 ms of it our export.
    let mut line = None;
    for i in 1..=10u64 {
        if let Some(l) = c.note(30_000, 60_000, 5_000 + i * 100) {
            line = Some(l);
        }
    }
    let line = line.expect("a full second must report");
    assert!(
        line.contains("tranches=11"),
        "the arming call counts: {line}"
    );
    assert!(
        line.contains("duty=0.900"),
        "900 ms busy in a 1000 ms window is duty 0.9: {line}"
    );
    assert!(
        line.contains("drain_us=300000") && line.contains("publish_us=600000"),
        "the phases must stay separable — this is which half to attack: {line}"
    );
    assert!(line.contains("max_tranche_us=90000"), "{line}");

    // Phases are attributions inside `drain_us`, not a partition of it, so they
    // are reported with their own counts and are allowed to overlap each other.
    // What must hold is that each lands in its own bucket — a fused figure would
    // make "the draws are slow" and "the flushes are slow" the same reading.
    for _ in 0..3 {
        c.note_phase(DrainPhase::Draw, 20_000);
    }
    c.note_phase(DrainPhase::Compute, 7_000);
    c.note_phase(DrainPhase::Flush(FlushRail::Render), 11_000);

    // An idle window must read near zero rather than inheriting the busy one,
    // and `skipped` must survive as its own arm: a worker that keeps bailing
    // before the lock looks identical to an idle one in the duty alone.
    c.note_skipped();
    c.note_skipped();
    let mut idle = None;
    for i in 1..=10u64 {
        if let Some(l) = c.note(500, 0, 6_000 + i * 100) {
            idle = Some(l);
        }
    }
    let idle = idle.expect("second window must report");
    assert!(
        idle.contains("duty=0.005"),
        "the window must not average in the previous busy one: {idle}"
    );
    assert!(idle.contains("skipped=2"), "{idle}");
    assert!(
        idle.contains("draw_us=60000")
            && idle.contains("draws=3")
            && idle.contains("compute_us=7000")
            && idle.contains("computes=1")
            && idle.contains("flush_us=11000")
            && idle.contains("flushes=1"),
        "each phase must land in its own bucket with its own count: {idle}"
    );
}

/// A mean flush cost cannot name the defect; the tail can.
///
/// `flush_us/flushes` reads the same for "every flush costs 7.7 ms" and "most
/// are free and one blocked 30 ms", and those are different defects with
/// different fixes — eliminate the per-frame work, versus find the one stall.
/// Likewise `max_tranche_us` is a max with no count, so one 38 ms tranche and
/// three 20 ms ones are indistinguishable. Both gaps are what let a continuous
/// per-frame cost read as an occasional hitch.
#[test]
fn the_drain_duty_census_separates_a_flush_tail_from_a_flush_mean() {
    use crate::runtime::drain::{DrainDutyCensus, DrainPhase, FlushRail};
    let c = DrainDutyCensus::default();
    assert!(c.note(0, 0, 5_000).is_none(), "first call arms only");

    // Nine cheap flushes and one long one: the mean is unremarkable, the tail
    // is the whole story.
    for _ in 0..9 {
        c.note_phase(DrainPhase::Flush(FlushRail::Linear), 1_000);
    }
    c.note_phase(DrainPhase::Flush(FlushRail::Render), 30_000);
    // Two tranches over a frame budget and one comfortably under it.
    c.note(30_000, 0, 5_500);
    c.note(9_000, 0, 5_600);
    c.note(1_000, 0, 5_700);
    let line = c
        .note(0, 0, 6_100)
        .expect("a full window must report")
        .to_string();

    assert!(line.contains("max_flush_us=30000"), "{line}");
    assert!(line.contains("flush_us=39000 flushes=10"), "{line}");
    // Mean is 3.9 ms and would look healthy against an 8 ms budget; the tail is
    // nearly four times the whole budget.
    // Five tranches, not four: the call that closes the window is itself a
    // tranche and is counted in the window it reports.
    assert!(
        line.contains("slow_tranches=2/5"),
        "two of five tranches held the lock at least a frame: {line}"
    );
    // The threshold is derived from the delivered VBL cadence, not written
    // down, so it tracks the refresh rate rather than aging beside it.
    assert!(line.contains("slow_us=8333"), "{line}");

    // The same window, split by rail. Nine cheap flushes on one rail and one
    // expensive flush on another is exactly the shape the aggregate cannot
    // express: `flushes=10` reads as one busy mechanism, while the count says
    // the linear rail owns nine tenths of it and the cost says the render rail
    // owns three quarters. Those two readings point at different code.
    let rails = c
        .take_flush_rails()
        .expect("a window that flushed must split");
    assert!(rails.contains("win_ms=1100"), "{rails}");
    assert!(rails.contains("render_us=30000 render=1"), "{rails}");
    assert!(rails.contains("linear_us=9000 linear=9"), "{rails}");
    assert!(rails.contains("gva_us=0 gva=0"), "{rails}");
    assert!(rails.contains("storage_us=0 storage=0"), "{rails}");
    assert!(rails.contains("linear_max_us=1000"), "{rails}");
    assert!(rails.contains("render_max_us=30000"), "{rails}");
    // The rails are a partition of the aggregate, not a second measurement of
    // it: 30000 + 9000 is the `flush_us=39000` asserted above and 1 + 9 its
    // `flushes=10`. A split that did not reconcile would be worse than none.

    // The window resets with the line, so an idle one says nothing at all
    // rather than repeating the busy window's attribution.
    assert!(c.take_flush_rails().is_none(), "{rails}");
}

/// The render rail's 6.9 ms has to divide before it can be fixed.
///
/// `flush_rails` names the rail; it does not say whether the cost is the GPU
/// round trip or the bytes. Those have opposite fixes — a dirty rect shrinks
/// the copy and does nothing at all to a fence wait — so a split that fuses
/// them licenses the wrong change.
#[test]
fn the_readback_split_divides_a_round_trip_from_the_bytes_it_carried() {
    use crate::runtime::drain::{DrainDutyCensus, ReadbackPhase};
    let c = DrainDutyCensus::default();
    assert!(c.note(0, 0, 5_000).is_none(), "first call arms only");
    assert!(
        c.take_readback_split().is_none(),
        "a window with no readback must stay silent"
    );

    // One flush's worth: a cheap submit, a long fence, a moderate copy out and
    // a moderate copy into guest pages.
    c.note_readback(ReadbackPhase::Submit, 120);
    c.note_readback(ReadbackPhase::Fence, 5_400);
    c.note_readback(ReadbackPhase::Map, 800);
    c.note_readback(ReadbackPhase::Write, 600);
    // A second flush that waited far longer on the GPU.
    c.note_readback(ReadbackPhase::Submit, 100);
    c.note_readback(ReadbackPhase::Fence, 9_000);
    c.note_readback(ReadbackPhase::Map, 750);
    c.note_readback(ReadbackPhase::Write, 640);
    // The two host-side halves the GPU rail leaves behind when it stops moving
    // bytes. Reported here so a phase added to the enum without a slot in the
    // census arrays fails this test rather than panicking the emitter on a live
    // boot — which is where the `[_; 4]` that outlived a six-variant enum showed
    // up, on the reporting path, after every compile check had passed.
    c.note_readback(ReadbackPhase::Vouch, 300);
    c.note_readback(ReadbackPhase::Resolve, 90);
    assert!(c.note(0, 0, 6_100).is_some(), "a full window must report");

    let split = c
        .take_readback_split()
        .expect("a window that read back must split");
    assert!(split.contains("win_ms=1100"), "{split}");
    assert!(split.contains("submit_us=220 submit=2"), "{split}");
    assert!(split.contains("fence_us=14400 fence=2"), "{split}");
    assert!(split.contains("map_us=1550 map=2"), "{split}");
    assert!(split.contains("write_us=1240 write=2"), "{split}");
    assert!(split.contains("vouch_us=300 vouch=1"), "{split}");
    assert!(split.contains("resolve_us=90 resolve=1"), "{split}");
    // The tail matters for the same reason it does on the rail above: a mean
    // fence of 7.2 ms and a worst of 9 ms is a steady tax, not a hitch.
    assert!(split.contains("fence_max_us=9000"), "{split}");

    assert!(c.take_readback_split().is_none(), "the window must reset");
}

/// The offer side of the window cadence needs its own census, because the
/// present side cannot see a frame that never arrived.
///
/// `host_window_cadence` reads `presents == offered` with `busy_acquire=0`, so
/// nothing downstream drops a frame; the deficit against the host's 120 Hz is
/// entirely in the offer rate. `publish_window_frame` has three separate ways to
/// return without offering one, and they point at unrelated fixes — `same_key`
/// dominating means the guest's own present cadence is the ceiling, while
/// `fresh` near the tranche rate would mean the loss is downstream of here.
#[test]
fn the_window_publish_census_keeps_its_three_refusals_apart() {
    use crate::runtime::drain::{WindowPublish, WindowPublishCensus};
    let c = WindowPublishCensus::default();
    assert!(
        c.take(1_000).is_none(),
        "a window with no publish attempt must stay silent"
    );

    c.note(WindowPublish::Fresh);
    c.note(WindowPublish::SameKey);
    c.note(WindowPublish::SameKey);
    c.note(WindowPublish::SameKey);
    c.note(WindowPublish::NoFrame);
    let line = c.take(1_000).expect("attempts must report");
    assert!(line.contains("fresh=1"), "{line}");
    assert!(line.contains("same_key=3"), "{line}");
    assert!(line.contains("no_frame=1"), "{line}");
    // Reported even at zero: a refusal class that vanishes from the line when
    // it stops firing is one a reader cannot tell from a class that was never
    // compiled in, and "the window is gone" reading zero is the answer that
    // rules out a whole branch.
    assert!(line.contains("no_window=0"), "{line}");

    assert!(c.take(1_000).is_none(), "the window must reset");
}

/// A writeback must not copy a frame in order to hand back the frame it copied.
///
/// The fragmented landing path staged every row into a whole-frame buffer before
/// handing runs to the mapper. When the mapping's row pitch is the packed row
/// length and no conversion is owed, that buffer is byte-for-byte the source it
/// was built from — an 8.29 MB allocation and copy, 95 times a second on the
/// composite surface, to produce a slice already in hand.
///
/// Both halves are asserted, because eliding a copy is only correct if the bytes
/// are the same: the census must show a fragmented landing with no staging pass,
/// **and** the guest's pages must hold exactly what was written.
#[test]
fn a_fragmented_writeback_stages_nothing_when_the_staged_frame_is_the_source() {
    use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    // This test prices the fallback after a host declines a scattered alias.
    // The default fixture can build a packed bounce view and would correctly
    // take the contiguous arm instead.
    host.strict_linux_map = true;
    // Large enough to span several guest pages: a frame that fits in one page is
    // contiguous by construction and would never reach the path under test.
    let (w, h) = (256u32, 64u32);
    let stride = w * 4;
    let need = (stride as usize) * (h as usize);
    let page_size = 1u64 << PAGE_SHIFT_ARM64E;
    let pages = (need as u64).div_ceil(page_size) as usize;
    // Deliberately non-consecutive guest frames, so the mapping cannot resolve
    // to one packed host run and the fragmented path is the one under test.
    let mut entries = Vec::with_capacity(pages);
    for i in 0..pages {
        let pfn = 0x300u32 + (i as u32) * 4;
        host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, page_size as usize, 0);
        entries.push((pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID);
    }
    assert!(state.map_surface(9));
    {
        let m = state.mappings.get_mut(&9).unwrap();
        m.mapped = true;
        m.mapping_internal = 9;
        m.page_entries = entries;
    }
    assert!(state.set_mapping_geom(9, w, h, MTL_FORMAT_BGRA8_UNORM));

    // A gradient rather than a constant: a staging bug that lands the wrong row
    // is invisible against a frame of identical bytes.
    let frame: Vec<u8> = (0..need).map(|i| (i % 251) as u8).collect();
    // Drain whatever earlier tests in this binary left in the shared census.
    let _ = super::census::SURFACE_WRITE.take(0);
    assert!(write_bgra8(&mut state, &mut host, 9, &frame, stride, w, h));

    let line = super::census::SURFACE_WRITE
        .take(1_000)
        .expect("a writeback must report");
    assert!(
        line.contains("contig=0 frag=1"),
        "the fragmented path is the one under test: {line}"
    );
    assert!(
        line.contains("stage_us=0 stage=0"),
        "a staged frame identical to its source must not be built: {line}"
    );
    assert!(
        line.contains("land=1"),
        "the bytes must still reach the guest: {line}"
    );

    let landed = crate::runtime::surface_cache::get(&state, 9, w, h)
        .expect("the writeback publishes its own frame");
    assert_eq!(landed, &frame[..], "the elided copy must change no byte");
}

/// A writeback whose caller owns the frame must publish it, not duplicate it.
///
/// Landing the frame in guest pages and publishing it to the host cache are two
/// different obligations over the same bytes, and only the first one is a copy.
/// The cache stores its frames behind an `Arc` so an entry and a deferred window
/// can name one allocation; a caller arriving with one and still paying a
/// whole-frame memcpy — 1.21 ms per flush on the composite, more than landing
/// the bytes costs — is the copy this asserts is gone.
///
/// Pointer identity is the assertion because it is the only thing that
/// distinguishes publishing from copying: the bytes are equal either way.
#[test]
fn an_owned_writeback_publishes_its_frame_to_the_cache_without_copying_it() {
    use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8_owned;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let (w, h) = (64u32, 16u32);
    let stride = w * 4;
    let need = (stride as usize) * (h as usize);
    let pfn = 0x480u32;
    host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x8000, 0);
    assert!(state.map_surface(7));
    {
        let m = state.mappings.get_mut(&7).unwrap();
        m.mapped = true;
        m.mapping_internal = 7;
        m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    assert!(state.set_mapping_geom(7, w, h, MTL_FORMAT_BGRA8_UNORM));

    let frame = std::sync::Arc::new((0..need).map(|i| (i % 251) as u8).collect::<Vec<u8>>());
    assert!(write_bgra8_owned(
        &mut state, &mut host, 7, &frame, stride, w, h
    ));

    let published = crate::runtime::surface_cache::get(&state, 7, w, h)
        .expect("a non-skipping writeback publishes its frame");
    assert_eq!(
        published.as_ptr(),
        frame.as_ptr(),
        "the caller's allocation must be published, not duplicated"
    );
    assert_eq!(published, &frame[..], "and it must be the frame written");

    // A pitch the cache's contract does not describe must not be shared: the
    // entry promises a tight frame at its geometry, and handing a padded
    // allocation to later readers as though it were one is the failure the
    // guard exists for.
    let padded_stride = stride + 4;
    let padded = std::sync::Arc::new(vec![0x5Au8; (padded_stride as usize) * (h as usize)]);
    assert!(write_bgra8_owned(
        &mut state,
        &mut host,
        7,
        &padded,
        padded_stride,
        w,
        h
    ));
    let repacked = crate::runtime::surface_cache::get(&state, 7, w, h).unwrap();
    assert_ne!(
        repacked.as_ptr(),
        padded.as_ptr(),
        "a padded frame must be repacked rather than shared"
    );
    assert!(repacked.iter().all(|&b| b == 0x5A), "and must be correct");
}

/// A writeback whose frame is borrowed must drop the cache entry, not keep it.
///
/// `write_bgra8_uncached` exists for the deferred render flush, whose frame is
/// the engine's readback staging buffer under a lease: it goes back to the pool
/// a moment later, so nothing host-side may still be naming it. The frame is
/// landed in the guest's pages either way, and the only question this settles is
/// what the cache holds afterwards.
///
/// Leaving the previous entry behind is the failure. A reader that hits one is
/// served a whole frame that is one or more paints old, with nothing in the log
/// saying so — a stale compositing layer, which is the corruption class this
/// device is chasing rather than a performance detail. Dropping it sends every
/// reader to the guest pages this write has just filled, which is correct by
/// construction.
///
/// Both halves are asserted from a *populated* starting state, because an
/// assertion that the cache is empty is vacuous against a cache that was never
/// filled.
#[test]
fn a_borrowed_writeback_drops_the_cache_entry_rather_than_leaving_it_stale() {
    use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::{write_bgra8_owned, write_bgra8_uncached};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let (w, h) = (64u32, 16u32);
    let stride = w * 4;
    let need = (stride as usize) * (h as usize);
    let pfn = 0x4A0u32;
    host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x8000, 0);
    assert!(state.map_surface(9));
    {
        let m = state.mappings.get_mut(&9).unwrap();
        m.mapped = true;
        m.mapping_internal = 9;
        m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    assert!(state.set_mapping_geom(9, w, h, MTL_FORMAT_BGRA8_UNORM));

    // An owning writeback first, so there is an entry to be left behind.
    let first = std::sync::Arc::new(vec![0x11u8; need]);
    assert!(write_bgra8_owned(
        &mut state, &mut host, 9, &first, stride, w, h
    ));
    assert!(
        crate::runtime::surface_cache::get(&state, 9, w, h).is_some(),
        "the owning writeback must publish, or this test proves nothing"
    );

    let second: Vec<u8> = (0..need).map(|i| (i % 241) as u8).collect();
    assert!(write_bgra8_uncached(
        &mut state, &mut host, 9, &second, stride, w, h
    ));

    assert!(
        crate::runtime::surface_cache::get(&state, 9, w, h).is_none(),
        "a borrowed frame must not be left named by a cache entry, and the \
         previous frame must not be left standing in its place"
    );
    // And the obligation the writeback actually owes: the bytes are in the
    // guest's pages, which is where every reader that now misses will look.
    let mut landed = vec![0u8; need];
    host.read_gpa((pfn as u64) << PAGE_SHIFT_ARM64E, &mut landed)
        .expect("the mapping's pages must be readable");
    assert_eq!(landed, second, "the borrowed frame must have landed");
}

/// A cache entry that is replaced every frame must not be reallocated every
/// frame.
///
/// The host-side duplicate is a fresh multi-megabyte `Vec` per writeback, and
/// the allocation is the expensive half rather than the copy: the pages come
/// back untouched and the fill faults every one of them in, then the buffer is
/// dropped and the next flush repeats it. Measured at 1.21 ms per flush against
/// 0.72 ms for landing the whole frame in the guest's pages.
///
/// The second assertion is the one that makes reuse safe rather than merely
/// cheap. The `Arc` exists so a deferred window can hold the exact frame it
/// armed on; writing through it would rewrite that window's pixels underneath
/// it, which is a frame landing in the wrong layer.
#[test]
fn the_surface_cache_reuses_its_buffer_but_never_one_someone_else_is_holding() {
    use crate::runtime::surface_cache;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let (w, h) = (16u32, 8u32);
    let need = (w as usize) * (h as usize) * 4;

    surface_cache::store_rows(&mut state, 4, w, h, &vec![0x11u8; need], w * 4);
    let first = surface_cache::get(&state, 4, w, h).unwrap().as_ptr();

    // Nothing else holds the frame: the entry's own allocation is rewritten.
    surface_cache::store_rows(&mut state, 4, w, h, &vec![0x22u8; need], w * 4);
    let second = surface_cache::get(&state, 4, w, h).unwrap();
    assert_eq!(second.as_ptr(), first, "an unheld buffer must be reused");
    assert_eq!(second[0], 0x22, "and must hold the new frame");

    // A window armed on this allocation is exactly what the `Arc` is for.
    let held = state.host_surfaces.get(&4).unwrap().bgra.clone();
    surface_cache::store_rows(&mut state, 4, w, h, &vec![0x33u8; need], w * 4);
    assert_eq!(
        held[0], 0x22,
        "a frame someone else is holding must not be rewritten"
    );
    let third = surface_cache::get(&state, 4, w, h).unwrap();
    assert_eq!(
        third[0], 0x33,
        "and the entry must still take the new frame"
    );
    assert_ne!(
        third.as_ptr(),
        held.as_ptr(),
        "which means it had to allocate"
    );
}

/// The largest phase of the largest rail is three full-frame passes under one
/// name, and a total cannot say which of them it is.
///
/// `write_us` covers the staged buffer, the bytes reaching guest pages and the
/// host-side cache duplicate. Those have entirely different fixes — two are
/// per-flush multi-megabyte allocations that could be reused and one is the work
/// the guest actually asked for — so a reader holding only the sum has no way to
/// tell an unavoidable cost from two avoidable ones. The path counts are part of
/// the answer, not decoration: `stage_us=0` means the contiguous path, and
/// without `contig`/`frag` that is indistinguishable from free staging.
#[test]
fn the_write_split_separates_the_guest_s_bytes_from_the_buffers_built_around_them() {
    use crate::runtime::drain::{SurfaceWriteCensus, SurfaceWritePhase};
    let c = SurfaceWriteCensus::default();
    assert!(
        c.take(1_000).is_none(),
        "a window with no surface write must stay silent"
    );

    // One fragmented write: staged, landed, then cached.
    c.note_path(false, 8_290_000);
    c.note(SurfaceWritePhase::Stage, 1_200);
    c.note(SurfaceWritePhase::Land, 1_500);
    c.note(SurfaceWritePhase::Cache, 1_100);
    // One contiguous write: no staging pass at all.
    c.note_path(true, 8_290_000);
    c.note(SurfaceWritePhase::Land, 900);
    c.note(SurfaceWritePhase::Cache, 1_300);

    let line = c.take(1_000).expect("traffic must report");
    assert!(line.contains("contig=1 frag=1"), "{line}");
    assert!(line.contains("bytes=16580000"), "{line}");
    assert!(line.contains("stage_us=1200 stage=1"), "{line}");
    assert!(line.contains("land_us=2400 land=2"), "{line}");
    assert!(line.contains("cache_us=2400 cache=2"), "{line}");
    // The tail, for the same reason every other split here carries one: a mean
    // cannot tell a steady tax from an occasional stall.
    assert!(line.contains("land_max_us=1500"), "{line}");

    assert!(c.take(1_000).is_none(), "the window must reset");
}

/// Whether the readback's fence wait has anywhere to hide is a measurement, and
/// it is this one.
///
/// The proposal it exists to judge — submit the copy at the arm rather than at
/// the flush — is worth its complexity only if wall clock separates the two. A
/// census that attributed *every* flush to the newest arm would report a
/// plausible age even when several windows were outstanding and the pairing was
/// meaningless, so the ambiguous case is counted apart rather than averaged in.
#[test]
fn the_resident_arm_age_refuses_to_pair_a_flush_with_an_arm_it_cannot_name() {
    use crate::runtime::drain::ResidentArmCensus;
    let c = ResidentArmCensus::default();
    assert!(
        c.take(1_000).is_none(),
        "a window with no resident traffic must stay silent"
    );

    // One arm, one flush 4 ms later: the pairing is unambiguous.
    c.note_arm(100_000);
    c.note_flush(104_000);
    // A second, longer interval.
    c.note_arm(200_000);
    c.note_flush(211_000);
    // Two arms before a flush: the age of "the arm" is not a number, so this
    // one is counted as ambiguous instead of credited to the newer arm.
    c.note_arm(300_000);
    c.note_arm(300_500);
    c.note_flush(309_000);

    let line = c.take(1_000).expect("traffic must report");
    assert!(line.contains("arms=4 flushes=3"), "{line}");
    assert!(line.contains("aged=2"), "{line}");
    assert!(line.contains("age_us=15000"), "{line}");
    assert!(line.contains("max_age_us=11000"), "{line}");
    assert!(line.contains("multi=1"), "{line}");

    // A window refused before it reaches the flush site leaves its arm
    // uncounted; the next arm makes the following flush ambiguous once, and
    // then the pairing recovers on its own rather than sticking wrong forever.
    c.note_arm(400_000);
    // (no flush — the window drifted out through one of the refusals)
    c.note_arm(500_000);
    c.note_flush(505_000);
    c.note_arm(600_000);
    c.note_flush(607_000);
    let line = c.take(1_000).expect("traffic must report");
    assert!(
        line.contains("multi=1"),
        "the stale arm is ambiguous once: {line}"
    );
    assert!(
        line.contains("aged=1") && line.contains("age_us=7000"),
        "and the next pairing is exact again: {line}"
    );
}

/// The vCPU's wait must be measured where the guest pays it.
///
/// Every other figure about a long tranche is taken from the side that *holds*
/// the device lock, which leaves "the drain held it 38 ms" and "the guest
/// missed a frame" joined by an inference. This census measures the blocked
/// side: how long the guest's MMIO access was actually stopped.
///
/// `uncontended` is counted separately and deliberately. A window with a large
/// `max_wait_us` and a huge `uncontended` count is a rare collision; the same
/// max with few uncontended acquisitions is a worker that owns the lock. Those
/// are opposite diagnoses and a wait-only counter cannot tell them apart.
#[test]
fn the_vcpu_lock_census_reports_the_blocked_side_and_separates_free_acquisitions() {
    use crate::runtime::drain::VcpuLockCensus;
    let c = VcpuLockCensus::default();
    assert!(c.note_wait(1, 5_000).is_none(), "first call arms only");

    for _ in 0..500 {
        assert!(
            c.note_uncontended(|| 5_050).is_none(),
            "free acquisitions inside the window must not report"
        );
    }
    // Two waits shorter than a frame, one longer.
    c.note_wait(200, 5_100);
    c.note_wait(900, 5_200);
    c.note_wait(30_000, 5_300);
    let line = c.note_wait(50, 6_100).expect("a full window must report");

    assert!(line.contains("max_wait_us=30000"), "{line}");
    assert!(line.contains("uncontended=500"), "{line}");
    // Only the 30 ms wait cost the guest a whole frame; the sub-millisecond
    // ones are collisions the guest would never notice.
    assert!(line.contains("frame_waits=1"), "{line}");
    // Five, not three: the call that arms the window and the call that closes
    // it are both real waits and are counted in the window they bound.
    assert!(line.contains("waits=5"), "{line}");
    assert!(line.contains("wait_us=31151"), "{line}");
}

/// The stall the PCI pathway actually has: a doorbell the guest thinks landed.
///
/// `reims-vgpu-pci` exposes no IOSFC region, so `lock_device_for_vcpu` — and
/// with it the whole `vcpu_lock_wait` census — is unreachable on x86. Its
/// silence there is structural, not a result, and reading it as "the drain
/// never stalled the guest" is the error the census next door was rebuilt to
/// stop making. x86's vCPU does not block; it queues, and the queued write does
/// not run until the drain worker's tranche ends. That delay is what this
/// measures.
#[test]
fn the_doorbell_census_separates_a_deferred_apply_from_a_direct_one() {
    use crate::runtime::drain::census::UNCONTENDED_POLL;
    use crate::runtime::drain::DoorbellCensus;
    let c = DoorbellCensus::default();
    assert!(
        c.note_queued(0x100c, 1, 5_000).is_none(),
        "first call arms only"
    );

    for _ in 0..300 {
        assert!(
            c.note_direct(|| 5_050).is_none(),
            "direct applies inside the window must not report"
        );
    }
    // Two delays the guest would never notice, one that costs it a whole frame.
    c.note_queued(0x100c, 120, 5_100);
    c.note_queued(0x1020, 700, 5_200);
    c.note_queued(0x100c, 43_000, 5_300);
    let line = c
        .note_queued(0x100c, 80, 6_100)
        .expect("a full window must report");

    assert!(line.contains("queued=5"), "{line}");
    assert!(line.contains("direct=300"), "{line}");
    assert!(line.contains("age_us=43901"), "{line}");
    assert!(line.contains("max_age_us=43000"), "{line}");
    assert!(line.contains("frame_late=1"), "{line}");

    // Which registers deferred, and how badly — the whole point of the
    // breakdown is that "half the doorbells queue" does not say whether one
    // register is responsible or every one of them is.
    assert!(line.contains("offsets=2 shown=2"), "{line}");
    assert!(
        line.contains("off_0x100c=4/43000"),
        "the busiest register must lead, with its own worst age: {line}"
    );
    assert!(line.contains("off_0x1020=1/700"), "{line}");

    // And a window that only ever applied directly still reports, so "nothing
    // was ever deferred" and "no MMIO reached this device" stay distinguishable.
    let c = DoorbellCensus::default();
    let mut line = None;
    for i in 0..=UNCONTENDED_POLL {
        let now = if i == 0 { 5_000 } else { 6_200 };
        if let Some(l) = c.note_direct(|| now) {
            assert!(line.is_none(), "one report per window");
            line = Some(l);
        }
    }
    let line = line.expect("a window of direct applies must report");
    assert!(line.contains("queued=0"), "{line}");
    assert!(line.contains("frame_late=0"), "{line}");
    assert!(
        line.contains(&format!("direct={}", UNCONTENDED_POLL + 1)),
        "{line}"
    );
}

/// A window in which the vCPU never blocked must still report.
///
/// As first shipped the census was driven only from the wait path, so zero
/// waits emitted zero lines — and a silent log then means both "the drain never
/// stalled the guest" and "no IOSFC traffic reached this device at all". A live
/// driven boot produced exactly that silence, which reads as the reassuring one
/// of the two and is worthless as evidence either way. The free path now drives
/// the same report, so the strong negative (`waits=0` beside a large
/// `uncontended`) is something the log can actually say.
#[test]
fn the_vcpu_lock_census_reports_a_window_that_never_blocked() {
    use crate::runtime::drain::census::UNCONTENDED_POLL;
    use crate::runtime::drain::VcpuLockCensus;
    let c = VcpuLockCensus::default();
    let mut line = None;
    // The clock is only read at a poll, so the window spans exactly one poll
    // interval: the acquisition that arms it and the one that closes it.
    for i in 0..=UNCONTENDED_POLL {
        let now = if i == 0 { 5_000 } else { 6_400 };
        if let Some(l) = c.note_uncontended(|| now) {
            assert!(line.is_none(), "one report per window");
            line = Some(l);
        }
    }
    let line = line.expect("a window of free acquisitions must report");
    assert!(line.contains("waits=0"), "{line}");
    assert!(line.contains("frame_waits=0"), "{line}");
    assert!(line.contains("max_wait_us=0"), "{line}");
    assert!(line.contains("win_ms=1400"), "{line}");
    // Every acquisition up to and including the one that closed the window.
    assert!(
        line.contains(&format!("uncontended={}", UNCONTENDED_POLL + 1)),
        "{line}"
    );
}

/// A guest display reinit (SETUP_SHARED_STATE while already ONLINE) that
/// arrives *after* boot-convergence self-labels with one correlated
/// `post_converge_display_reinit` line — the smoking gun for the intermittent
/// post-converge boot-progress overlay. Before
/// convergence the same reinit must NOT emit the correlated line (a display
/// re-register during normal boot bring-up is expected, not the overlay).
#[test]
fn signal_display_vbl_after_online_uses_shared_time_limiter() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let last_ms = std::sync::atomic::AtomicU64::new(0);
    let gpa = 0x7c000000u64;
    host.map_range(gpa, PAGE_SIZE_ARM64E as usize, 0);
    state.display.shared_gpa = gpa;
    state.display.display_index = 0;
    state.display.online_acked = true;
    // This test is about the limiter, so it models a guest that asked for VBL.
    // Without the bit the path declines before the limiter is reached and every
    // count below reads zero — see
    // `signal_display_vbl_declines_a_class_the_guest_did_not_enable`.
    host.put_u32(gpa + DISPLAY_SHARED_ENABLE_MASK, DISPLAY_VBL_EVENT_MASK);

    // Microseconds: the base must exceed one grid interval from the zero the
    // limiter starts at, or the very first claim is refused as too early.
    let base = 5_000_000;
    signal_display_vbl_at(&mut state, &mut host, &last_ms, base);
    assert_eq!(host.actions.len(), 1);
    assert!(
        !claim_display_vbl(&last_ms, base),
        "the contended path cannot claim the locked path's interval"
    );
    signal_display_vbl_at(
        &mut state,
        &mut host,
        &last_ms,
        base + DISPLAY_VBL_MIN_INTERVAL_US - 1,
    );
    assert_eq!(
        host.actions.len(),
        1,
        "polls inside the interval must not over-signal"
    );
    signal_display_vbl_at(
        &mut state,
        &mut host,
        &last_ms,
        base + DISPLAY_VBL_MIN_INTERVAL_US,
    );
    assert_eq!(
        host.actions.len(),
        2,
        "the exact interval boundary must signal"
    );
    assert_eq!(host.actions[0].kind, HostActionKind::IrqGfxPulse);
    let mut pending = [0u8; 4];
    host.read_gpa(gpa + DISPLAY_SHARED_PENDING, &mut pending)
        .unwrap();
    assert_ne!(ld32(&pending) & DISPLAY_VBL_EVENT_MASK, 0);
    assert_ne!(
        state
            .gfx
            .interrupt_status_disp
            .load(std::sync::atomic::Ordering::Acquire)
            & 1,
        0
    );
}

/// A guest that has not enabled VBL in the shared page's mask gets no VBL:
/// no pending bit, no interrupt.
///
/// The guest's interrupt handler read-clears `pending & enable_mask` and leaves
/// every other bit exactly where it found it, so a bit this device sets for a
/// disabled class is one that nothing will ever clear. It is not an ignored
/// notification; it is a permanent residue in a word this device keeps
/// read-modify-writing.
///
/// Both x86 rails measured here disable VBL: a macOS 11 guest published mask
/// `0xe` and a macOS 13 guest `0xc`, and each left `+0x100` holding precisely
/// the bits this device had set and the guest had not asked for — `0x1` and
/// `0x3`. Meanwhile the census reported `delivered=13312 window_hz=120.0`,
/// describing a 120 Hz display link to a guest that had asked for none of it.
///
/// The mask is re-read per tick rather than latched: `enableVBLInterrupt` and
/// `disableVBLInterrupt` are a `lock or` and a `lock and` on that same word, so
/// the guest may turn the class on or off at any moment and expects the next
/// tick to honour it.
#[test]
fn signal_display_vbl_declines_a_class_the_guest_did_not_enable() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let last_ms = std::sync::atomic::AtomicU64::new(0);
    let gpa = 0x7c000000u64;
    host.map_range(gpa, PAGE_SIZE_ARM64E as usize, 0);
    state.display.shared_gpa = gpa;
    state.display.display_index = 0;
    state.display.online_acked = true;

    // The two masks a real x86 guest was measured publishing. Neither carries
    // the VBL bit, so neither is owed a VBL pending bit — but `0x0e` carries the
    // *transaction* class (macOS 11 arms that instead of VBL), and the tick owes
    // that guest an interrupt for it. `0x0c` carries only online and offline,
    // neither of which a refresh tick signals, so nothing at all is owed there.
    //
    // `owes_irq` is what separates "this device declined a class" from "this
    // device went quiet": both masks decline VBL and only one of them means the
    // guest hears nothing.
    for (mask, owes_irq) in [(0x0eu32, true), (0x0cu32, false)] {
        host.put_u32(gpa + DISPLAY_SHARED_ENABLE_MASK, mask);
        host.put_u32(gpa + DISPLAY_SHARED_PENDING, 0);
        host.actions.clear();
        state
            .gfx
            .interrupt_status_disp
            .store(0, std::sync::atomic::Ordering::Release);
        // Well past one grid interval, so the limiter is not what refuses.
        last_ms.store(0, std::sync::atomic::Ordering::Release);
        signal_display_vbl_at(&mut state, &mut host, &last_ms, 5_000_000);

        let mut pending = [0u8; 4];
        host.read_gpa(gpa + DISPLAY_SHARED_PENDING, &mut pending)
            .unwrap();
        assert_eq!(
            ld32(&pending) & DISPLAY_VBL_EVENT_MASK,
            0,
            "mask {mask:#x} does not carry VBL, so the pending bit must stay clear"
        );
        assert_eq!(
            host.actions.len(),
            usize::from(owes_irq),
            "mask {mask:#x}: interrupt owed = {owes_irq}"
        );
        assert_eq!(
            state
                .gfx
                .interrupt_status_disp
                .load(std::sync::atomic::Ordering::Acquire)
                != 0,
            owes_irq,
            "mask {mask:#x}: display IRQ status owed = {owes_irq}"
        );
    }

    // And the same tick delivers once the guest turns the class on, which is
    // what says the decline above is the mask and not some other refusal.
    host.put_u32(
        gpa + DISPLAY_SHARED_ENABLE_MASK,
        0x0e | DISPLAY_VBL_EVENT_MASK,
    );
    host.actions.clear();
    last_ms.store(0, std::sync::atomic::Ordering::Release);
    signal_display_vbl_at(&mut state, &mut host, &last_ms, 5_000_000);
    let mut pending = [0u8; 4];
    host.read_gpa(gpa + DISPLAY_SHARED_PENDING, &mut pending)
        .unwrap();
    assert_ne!(
        ld32(&pending) & DISPLAY_VBL_EVENT_MASK,
        0,
        "the guest enabled VBL, so the pending bit is owed"
    );
    assert_eq!(
        host.actions.len(),
        1,
        "the guest enabled VBL, so an IRQ is owed"
    );
}

/// The VBL limiter is phase-locked to a fixed interval grid so poll jitter
/// cannot alias the delivered rate down to ~60 Hz (the boot-to-boot fps split).
/// The VBL we deliver must be the refresh rate we advertise.
///
/// These are two different constants reaching the guest by two different
/// routes: `DISPLAY_REFRESH_HZ` goes into the mode timing table it reads, and
/// the limiter interval paces the interrupt it is actually woken by. The guest
/// honours the interrupt — a driven Safari measured its own
/// `requestAnimationFrame` at exactly the delivered rate — so when the two
/// disagree, the timing table is a lie the guest never notices and we cannot
/// see either.
///
/// They did disagree: the limiter was a hardcoded 8 ms, which is 125 Hz, while
/// the table advertised 120. Asserting the identity rather than the value is
/// deliberate — it stays true if the advertised rate ever changes, and it is
/// the only form of this test that cannot itself go stale.
#[test]
fn delivered_vbl_cadence_equals_the_advertised_refresh_rate() {
    let delivered_hz = 1_000_000.0 / DISPLAY_VBL_MIN_INTERVAL_US as f64;
    assert!(
        (delivered_hz - DISPLAY_REFRESH_HZ as f64).abs() < 0.5,
        "advertising {DISPLAY_REFRESH_HZ} Hz but delivering {delivered_hz:.1} Hz \
         (interval {DISPLAY_VBL_MIN_INTERVAL_US} us)"
    );

    // The millisecond grid this replaced could not express the answer at all:
    // 120 Hz is 8333 us, and every whole-millisecond interval near it is wrong
    // by at least 4%. That is why the units changed rather than the number.
    assert_ne!(
        DISPLAY_VBL_MIN_INTERVAL_US % 1000,
        0,
        "a whole-millisecond interval cannot express {DISPLAY_REFRESH_HZ} Hz"
    );
}

/// Polls spaced just under the interval — the worst aliasing case — must still
/// converge to roughly the grid rate, NOT halve.
#[test]
fn claim_display_vbl_phase_locks_grid_under_jittery_polls() {
    use std::sync::atomic::AtomicU64;
    let interval = DISPLAY_VBL_MIN_INTERVAL_US;
    // Legacy "reset to now" behaviour would need two of these ~(interval-1)ms
    // polls per claim -> half rate. Phase-locking must claim on (nearly) every
    // poll once warmed up, because a late poll advances the grid by exactly one
    // interval and the next poll is already past the new deadline.
    let last = AtomicU64::new(0);
    let step = interval - 1; // poll spacing in the aliasing danger zone
    let polls = 64u64;
    let mut claims = 0u64;
    for i in 1..=polls {
        if claim_display_vbl(&last, i * step) {
            claims += 1;
        }
    }
    // Wall time covered is polls*step; a phase-locked grid delivers about one
    // VBL per interval, i.e. ~polls*step/interval claims — far above the
    // half-rate (~polls/2) the "reset to now" limiter produced.
    let grid_expected = polls * step / interval;
    assert!(
        claims >= grid_expected - 1,
        "phase-locked claims {claims} should track the grid rate ~{grid_expected}, not halve"
    );
    assert!(
        claims > polls * 2 / 3,
        "claims {claims} aliased below the grid — the 60-Hz-latch regression"
    );
}

/// A stall longer than two intervals (drain worker held the lock) resyncs the
/// phase to `now` rather than firing a back-dated burst of catch-up VBLs.
#[test]
fn claim_display_vbl_long_stall_resyncs_without_burst() {
    use std::sync::atomic::{AtomicU64, Ordering};
    let interval = DISPLAY_VBL_MIN_INTERVAL_US;
    let last = AtomicU64::new(1_000);
    // A single poll after a 10*interval stall claims exactly once and lands the
    // grid at `now` (no accumulated catch-up credit).
    let now = 1_000 + 10 * interval;
    assert!(claim_display_vbl(&last, now));
    assert_eq!(
        last.load(Ordering::Acquire),
        now,
        "long stall resyncs to now"
    );
    // The immediately following poll one interval later claims once more — a
    // steady single-VBL cadence, not a burst.
    assert!(claim_display_vbl(&last, now + interval));
    assert!(!claim_display_vbl(&last, now + interval)); // same instant: no double
}

/// Stand up a display whose shared page is mapped and whose ONLINE is acked, so
/// `signal_display_vbl_at` reaches the enable-mask read.
#[cfg(test)]
fn one_shot_display() -> (DeviceState, FakeHost, u64) {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let gpa = 0x7c000000u64;
    host.map_range(gpa, PAGE_SIZE_ARM64E as usize, 0);
    state.display.shared_gpa = gpa;
    state.display.display_index = 0;
    state.display.online_acked = true;
    (state, host, gpa)
}

#[cfg(test)]
fn set_enable_mask(host: &mut FakeHost, gpa: u64, mask: u32) {
    let mut m = [0u8; 4];
    st32(&mut m, mask);
    host.write_gpa(gpa + DISPLAY_SHARED_ENABLE_MASK, &m)
        .unwrap();
}

/// Did this tick hand the guest a VBL? Consumes the pending bit the way the
/// guest's ISR does, so the next tick starts from a clean word.
#[cfg(test)]
fn took_vbl(host: &mut FakeHost, gpa: u64) -> bool {
    let mut pending = [0u8; 4];
    host.read_gpa(gpa + DISPLAY_SHARED_PENDING, &mut pending)
        .unwrap();
    let got = ld32(&pending) & DISPLAY_VBL_EVENT_MASK != 0;
    let mut zero = [0u8; 4];
    st32(&mut zero, 0);
    host.write_gpa(gpa + DISPLAY_SHARED_PENDING, &zero).unwrap();
    got
}

/// A tick that samples the guest mid-turnaround must not spend the grid slot the
/// guest is about to ask for.
///
/// macOS 13 arms VBL one shot at a time — it clears bit 0 inside its handler and
/// sets it again when it next wants a frame — so a poll is as likely to find the
/// mask disarmed as armed. While the claim happened *before* the mask read, that
/// disarmed sample advanced the grid timestamp, and the guest's re-arm a
/// millisecond later then waited a further full interval. Its delivery rate
/// aliased to every other grid point, which is the 60-vs-120 boot-to-boot split.
#[test]
fn a_disarmed_tick_does_not_spend_the_grid_slot() {
    use std::sync::atomic::AtomicU64;
    let interval = DISPLAY_VBL_MIN_INTERVAL_US;
    let (mut state, mut host, gpa) = one_shot_display();
    let last = AtomicU64::new(0);

    // The guest arms, and one interval in it is served.
    set_enable_mask(
        &mut host,
        gpa,
        DISPLAY_VBL_EVENT_MASK | DISPLAY_ONLINE_EVENT_MASK,
    );
    signal_display_vbl_at(&mut state, &mut host, &last, interval);
    assert!(took_vbl(&mut host, gpa), "an armed guest is served");

    // It disarms inside its handler. The next grid point finds nothing armed.
    set_enable_mask(&mut host, gpa, DISPLAY_ONLINE_EVENT_MASK);
    signal_display_vbl_at(&mut state, &mut host, &last, 2 * interval);
    assert!(
        !took_vbl(&mut host, gpa),
        "a disarmed guest is owed nothing"
    );

    // It re-arms a millisecond later. A full interval has now passed since the
    // last *delivery*, so it must be served on the very next poll rather than
    // waiting out another interval it already waited.
    set_enable_mask(
        &mut host,
        gpa,
        DISPLAY_VBL_EVENT_MASK | DISPLAY_ONLINE_EVENT_MASK,
    );
    signal_display_vbl_at(&mut state, &mut host, &last, 2 * interval + 1_000);
    assert!(
        took_vbl(&mut host, gpa),
        "the disarmed tick spent this guest's slot — the 60-Hz latch"
    );
}

/// Reading the mask before claiming must not let the guest outrun the refresh
/// rate the timing table advertises.
///
/// This is the direction the fix could have widened: the limiter is the only
/// thing standing between a 240 Hz poll and a guest that holds VBL armed
/// continuously, and it now runs after a read that succeeds far more often.
#[test]
fn a_continuously_armed_guest_is_still_capped_at_the_advertised_rate() {
    use std::sync::atomic::AtomicU64;
    let interval = DISPLAY_VBL_MIN_INTERVAL_US;
    let (mut state, mut host, gpa) = one_shot_display();
    let last = AtomicU64::new(0);
    set_enable_mask(
        &mut host,
        gpa,
        DISPLAY_VBL_EVENT_MASK | DISPLAY_ONLINE_EVENT_MASK,
    );

    // Poll far faster than the grid, the way the 4 ms PCI heartbeat oversamples
    // an 8333 us interval, and never disarm.
    let step = interval / 8;
    let polls = 400u64;
    let mut delivered = 0u64;
    for i in 1..=polls {
        signal_display_vbl_at(&mut state, &mut host, &last, i * step);
        if took_vbl(&mut host, gpa) {
            delivered += 1;
        }
    }
    let grid = polls * step / interval;
    assert!(
        delivered <= grid,
        "delivered {delivered} exceeds the {grid} the advertised rate allows"
    );
    assert!(
        delivered >= grid - 1,
        "delivered {delivered} falls short of the grid {grid}"
    );
}

/// Both reporting arms measure their own window, and neither resets the other's.
///
/// A guest that arms VBL one shot at a time keeps `delivered` and `not_enabled`
/// both live, so both report — and they shared one `last_report` pair, which made
/// every window wrong precisely on the boots worth reading. The symptom in a real
/// log is `window_hz=0.0`: an arm reaching its own 1024 subtracted the other
/// arm's 1024 and got a zero-length count.
#[test]
fn the_two_reporting_vbl_arms_do_not_share_one_window() {
    use crate::runtime::drain::{VblCensus, VBL_DELIVERED, VBL_NOT_ENABLED};
    let c = VblCensus::default();

    // A one-shot guest takes both arms on the same timeline — a tick either
    // found it armed or did not — so both cross 1024 over one 8 ms grid and both
    // report at the same instant. Each is 1024 events over its own 8192 ms, so
    // both lines must read 125 Hz; with one shared pair whichever reports second
    // subtracts the first's count and prints the `window_hz=0.0` that a real
    // driven boot is full of.
    let mut served = None;
    let mut declined = None;
    for i in 1..=1024u64 {
        if let Some(l) = c.note(VBL_DELIVERED, i * 8) {
            served = Some(l);
        }
        if let Some(l) = c.note(VBL_NOT_ENABLED, i * 8) {
            declined = Some(l);
        }
    }
    for (line, arm) in [
        (
            served.expect("delivered reports at its own 1024"),
            "delivered",
        ),
        (
            declined.expect("not_enabled reports at its own 1024"),
            "not_enabled",
        ),
    ] {
        assert!(line.contains(&format!("arm={arm}")), "{line}");
        assert!(
            !line.contains("window_hz=0.0"),
            "{arm} had the other arm's count subtracted from it: {line}"
        );
        assert!(
            line.contains("window_hz=125.0"),
            "{arm} saw 1024 events over 8192 ms, which is 125 Hz: {line}"
        );
    }
}

/// After online is acked, a stale ONLINE bit (bit2) left in pending is
/// suppressed by the present/VBL signalers instead of re-delivered — else the
/// guest re-runs process_online → connectionChange → boot-progress overlay.
/// The signaler still records `stale_online_pending`
/// (measure + fix together). Pre-ack the bit is preserved (see the present
/// completion test) — the suppression is gated strictly on `online_acked`.
#[test]
fn acked_stale_online_bit_is_suppressed_not_redelivered() {
    let _proxy = crate::runtime::census::present_proxy::test_exclusive();
    crate::runtime::census::present_proxy::reset_for_test();
    // Per-process fail log under `cfg(test)`, so a delta is exact.
    let logged = || {
        std::fs::read_to_string(crate::observe::fail_log_path())
            .unwrap_or_default()
            .matches("stale_online_pending src=")
            .count()
    };
    let before = logged();

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let gpa = 0x7d00_0000u64;
    host.map_range(gpa, PAGE_SIZE_ARM64E as usize, 0);
    state.display.shared_gpa = gpa;
    state.display.display_index = 0;
    state.display.online_acked = true;
    host.put_u32(
        gpa + DISPLAY_SHARED_ENABLE_MASK,
        DISPLAY_PRESENT_EVENT_MASK | DISPLAY_ONLINE_EVENT_MASK,
    );
    // Stale ONLINE bit left in pending (the try_display_online/ack race).
    host.put_u32(gpa + DISPLAY_SHARED_PENDING, DISPLAY_ONLINE_EVENT_MASK);

    signal_display_present_complete(&mut state, &mut host);

    let mut pending = [0u8; 4];
    host.read_gpa(gpa + DISPLAY_SHARED_PENDING, &mut pending)
        .unwrap();
    let p = ld32(&pending);
    assert_ne!(
        p & DISPLAY_PRESENT_EVENT_MASK,
        0,
        "present completion still sets the present bit"
    );
    assert_eq!(
        p & DISPLAY_ONLINE_EVENT_MASK,
        0,
        "a stale acked ONLINE bit must be suppressed, not re-delivered"
    );
    assert_eq!(
        logged(),
        before + 1,
        "the suppressed stale online must still be named on the always-on log"
    );
}

/// RE: Unmap is PT-only. Discrete encode must stay in host_cache so sample
/// hits GVA key after remount even when guest pages are new zeros.
/// MapMemory2 stays notify-only (no invent write).
/// HostOps GVA views covering the range **must** be retired (Apple unmapMemory).
#[test]
fn unmap_memory_retains_gva_host_cache_for_sample() {
    use crate::model::GvaHostView;
    use crate::model::CHILD_OP_UNMAP_MEMORY;
    use crate::protocol::endian::{st32, st64};
    use crate::runtime::surface_cache;

    let page_shift = PAGE_SHIFT_X86;
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), page_shift);
    let gva = 0x2c22000u64;
    let w = 32u32;
    let h = 24u32;
    let need = (w * h * 4) as usize;
    let mut bgra = vec![0u8; need];
    for px in bgra.chunks_exact_mut(4) {
        px[0] = 185;
        px[1] = 126;
        px[2] = 81;
        px[3] = 255;
    }
    surface_cache::store_gva_owned(&mut state, gva, w, h, bgra, 0, None, true);
    // Simulated HostOps view of the same GVA (zero-copy import substrate).
    state.gva_host_views.push(GvaHostView {
        task_id: 1,
        gva,
        length: 0x10000,
        ptr: 0xfeed_0000,
        ptr_len: 0x10000,
        ..Default::default()
    });
    // Unrelated range must survive.
    state.gva_host_views.push(GvaHostView {
        task_id: 1,
        gva: 0x4000_0000,
        length: 0x1000,
        ptr: 0xcafe_0000,
        ptr_len: 0x1000,
        ..Default::default()
    });

    let mut unmap_pl = vec![0u8; 20];
    st32(&mut unmap_pl[0..4], 1);
    st64(&mut unmap_pl[4..12], gva);
    st64(&mut unmap_pl[12..20], 0x10000);
    let unmap = Packet {
        opcode: CHILD_OP_UNMAP_MEMORY,
        stamp_waits: Vec::new(),
        total_size: PACKET_HEADER_LEN + 20,
        completion_stamp: 0,
        payload: unmap_pl,
        next_head: 0,
    };
    process_child_packet(&mut state, &mut host, 2, &unmap);

    // Still sampleable from host_cache (no size gate; no Map rehydrate write).
    let got = surface_cache::get_gva(&state, gva, w, h).expect("retain after Unmap");
    assert_eq!(&got[0..4], &[185, 126, 81, 255]);
    // HostOps view of the unmapped range is gone; other GVA view kept.
    assert_eq!(state.gva_host_views.len(), 1);
    assert_eq!(state.gva_host_views[0].ptr, 0xcafe_0000);
    assert_eq!(state.retired_views, vec![(0xfeed_0000, 0x10000)]);
}

/// The guest hands a span's pages back on the strength of a completion it read
/// before a writeback this device queued into them has landed, and the kernel
/// can reuse the page before the copy arrives. The unmap is the last point that
/// can hold that write, so it settles whenever a writeback is outstanding and
/// the span cannot be proven disjoint — here the task has no page table, so the
/// walk names nothing and the answer is `Unnamed`, the fail-safe direction.
///
/// Vulkan rail by name: the write debt is the Vulkan engine's, and with no
/// device context its settle has nothing to wait on and clears the flag, which
/// is the observable that proves the release asked.
#[cfg(feature = "backend-vulkan")]
#[test]
fn unmap_memory_settles_an_outstanding_writeback_before_the_pages_go_back() {
    use crate::backend::vulkan::engine;
    use crate::model::CHILD_OP_UNMAP_MEMORY;
    use crate::protocol::endian::{st32, st64};

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    engine::arm_guest_write_debt_for_tests(&[0x1234_5000]);
    assert!(engine::guest_writes_outstanding());

    let mut unmap_pl = vec![0u8; 20];
    st32(&mut unmap_pl[0..4], 1);
    st64(&mut unmap_pl[4..12], 0x2c22000);
    st64(&mut unmap_pl[12..20], 0x10000);
    let unmap = Packet {
        opcode: CHILD_OP_UNMAP_MEMORY,
        stamp_waits: Vec::new(),
        total_size: PACKET_HEADER_LEN + 20,
        completion_stamp: 0,
        payload: unmap_pl,
        next_head: 0,
    };
    process_child_packet(&mut state, &mut host, 2, &unmap);

    assert!(
        !engine::guest_writes_outstanding(),
        "the release must settle a writeback it cannot prove disjoint"
    );
}

/// RE pageBacking Invalidate: clr hostValid → bump content_generation.
#[test]
fn invalidate_resources_bumps_mapping_content_generation() {
    use crate::model::CHILD_OP_INVALIDATE_RESOURCES;
    use crate::protocol::endian::st32;
    use crate::protocol::fifo::CHILD_INVALIDATE_PAGEON_FLAGS;

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    assert!(state.map_surface(0x2a));
    {
        let m = state.mappings.get_mut(&0x2a).unwrap();
        m.content_generation = 7;
    }
    let mut pl = vec![0u8; 16];
    st32(&mut pl[0..], 0);
    st32(&mut pl[4..], 1);
    st32(&mut pl[8..], 0x2a);
    st32(&mut pl[12..], CHILD_INVALIDATE_PAGEON_FLAGS);
    process_child_packet(
        &mut state,
        &mut host,
        4,
        &Packet {
            opcode: CHILD_OP_INVALIDATE_RESOURCES,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + 16,
            completion_stamp: 0,
            payload: pl,
            next_head: 0,
        },
    );
    assert_eq!(state.mappings[&0x2a].content_generation, 8);
}

/// MapMemory2 product path must **not** write guest GVA (flush disabled after
/// freelist PTE panic correlation). Helper still unit-tested in surface_cache.
#[test]
fn map_memory2_does_not_flush_gva_host_cache_on_wire() {
    use crate::model::CHILD_OP_MAP_MEMORY2;
    use crate::protocol::endian::{st32, st64};
    use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::runtime::surface_cache;

    let page_shift = PAGE_SHIFT_X86;
    let mut host = FakeHost::new();
    let dir_gpa = 2u64 << page_shift;
    let root_gpa = 3u64 << page_shift;
    let data_gpa = 5u64 << page_shift;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    host.map_range(data_gpa, 0x1000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    st32(&mut d[..4], 5);
    let _ = host.write_gpa(root_gpa + 4, &d[..4]);

    let mut state = DeviceState::new(DeviceId(1), page_shift);
    state.define_task(1, 0x1000, 2);
    let gva = 1u64 << page_shift;
    let mut bgra = vec![0u8; 16];
    bgra[0] = 185;
    bgra[1] = 126;
    bgra[2] = 81;
    bgra[3] = 255;
    surface_cache::store_gva_owned(&mut state, gva, 2, 2, bgra, 0, None, true);

    let mut pl = vec![0u8; 20];
    st32(&mut pl[0..], 1);
    st64(&mut pl[4..], gva);
    st64(&mut pl[12..], 0x1000);
    process_child_packet(
        &mut state,
        &mut host,
        2,
        &Packet {
            opcode: CHILD_OP_MAP_MEMORY2,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + 20,
            completion_stamp: 0,
            payload: pl,
            next_head: 0,
        },
    );
    let mut probe = [0u8; 4];
    host.read_gpa(data_gpa, &mut probe).unwrap();
    assert_eq!(
        probe,
        [0, 0, 0, 0],
        "product MapMemory2 must stay notify-only for GVA (no auto flush)"
    );
}

/// Synchronize 0x35 is stamp + wait only — no host_cache→guest write (RE audit).
#[test]
fn synchronize_resources_does_not_write_guest_pages() {
    use crate::model::CHILD_OP_SYNCHRONIZE_RESOURCES;
    use crate::protocol::endian::st32;
    use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::surface_cache;

    let page_shift = PAGE_SHIFT_X86;
    let page_size = 1u64 << page_shift;
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), page_shift);
    let mid = 0x2au32;
    let w = 2u32;
    let h = 2u32;
    let pfn = 0x7000u32;
    let gpa = (pfn as u64) << page_shift;
    host.map_range(gpa, page_size as usize, 0);
    assert!(state.map_surface(mid));
    {
        let m = state.mappings.get_mut(&mid).unwrap();
        m.mapped = true;
        m.page_entries =
            vec![(((pfn as u64) << PAGE_ENTRY_PFN_SHIFT) | (PAGE_ENTRY_VALID as u64)) as u32];
    }
    assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));
    let mut bgra = vec![0u8; (w * h * 4) as usize];
    bgra[0] = 0x10;
    bgra[1] = 0x20;
    bgra[2] = 0x30;
    bgra[3] = 0xff;
    surface_cache::store(&mut state, mid, w, h, bgra);

    let mut pl = vec![0u8; 12];
    st32(&mut pl[0..], 1);
    st32(&mut pl[4..], 1);
    st32(&mut pl[8..], mid);
    process_child_packet(
        &mut state,
        &mut host,
        4,
        &Packet {
            opcode: CHILD_OP_SYNCHRONIZE_RESOURCES,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + 12,
            completion_stamp: 0,
            payload: pl,
            next_head: 0,
        },
    );
    let mut probe = [0u8; 4];
    host.read_gpa(gpa, &mut probe).unwrap();
    assert_eq!(
        probe,
        [0, 0, 0, 0],
        "Synchronize must not write host_cache into guest pages"
    );
}

/// set guestValid alone must not bump host content generation.
#[test]
fn invalidate_without_clr_host_does_not_bump_generation() {
    use crate::model::CHILD_OP_INVALIDATE_RESOURCES;
    use crate::protocol::endian::st32;

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    assert!(state.map_surface(0x2a));
    {
        let m = state.mappings.get_mut(&0x2a).unwrap();
        m.content_generation = 7;
    }
    // LE bytes 00 00 00 01 = only set_guest_valid
    let mut pl = vec![0u8; 16];
    st32(&mut pl[0..], 0);
    st32(&mut pl[4..], 1);
    st32(&mut pl[8..], 0x2a);
    st32(&mut pl[12..], 0x0100_0000);
    process_child_packet(
        &mut state,
        &mut host,
        4,
        &Packet {
            opcode: CHILD_OP_INVALIDATE_RESOURCES,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + 16,
            completion_stamp: 0,
            payload: pl,
            next_head: 0,
        },
    );
    assert_eq!(state.mappings[&0x2a].content_generation, 7);
}

/// `present_unbacked` gate: a member presented twice with no full-frame Store
/// **naming it** in between is being shown content the guest never sent for it.
/// `note_dense_frame_published` is the only site that advances
/// `dense_frame_seq`, so an unchanged seq across a member's own two presents is
/// the exact structural witness.
///
/// The gate used to be described as covering "a full-frame Store or an
/// inter-buffer seed". `62587b1` deleted the peer front seed, so only the first
/// half survives.
///
/// Healthy alternation must stay quiet: each buffer advances on its own turn.
#[test]
fn present_backing_gate_fires_only_when_a_member_gained_nothing() {
    let w = 1920u32;
    let h = 1080u32;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    for mid in [1u32, 5u32] {
        state.map_surface(mid);
        state.note_dense_frame_published(mid, w, h);
    }

    // First present of each member has no prior witness — never a report.
    assert_eq!(state.note_present_backing(1), None);
    assert_eq!(state.note_present_backing(5), None);

    // Healthy a/b alternation: each member gets its own full frame before its
    // next present, so the seq advances and the gate stays silent.
    for _ in 0..4 {
        state.note_dense_frame_published(1, w, h);
        assert_eq!(state.note_present_backing(1), None);
        state.note_dense_frame_published(5, w, h);
        assert_eq!(state.note_present_backing(5), None);
    }

    // Mid 5 now goes dark: every full frame lands on mid 1, but the guest keeps
    // naming mid 5 at present. Each of those presents shows content mid 5 never
    // received, and each is reported (once per present, not once per lifetime).
    for _ in 0..3 {
        state.note_dense_frame_published(1, w, h);
        assert_eq!(state.note_present_backing(1), None);
        assert!(
            state.note_present_backing(5).is_some(),
            "no full-frame store named mid 5"
        );
    }

    // Backing is the seq itself, whatever advanced it: a member that reaches the
    // source's seq is quiet again on its next present.
    state.present.dense_frame_seq.insert(
        5,
        state.present.dense_frame_seq.get(&1).copied().unwrap_or(0),
    );
    assert_eq!(state.note_present_backing(5), None);

    // A recycled mapping id must not compare against its predecessor's witness.
    state.unmap_surface(5);
    state.map_surface(5);
    state.note_dense_frame_published(5, w, h);
    assert_eq!(state.note_present_backing(5), None);
}

/// The other half of the gate: a surface presented for the first time since it
/// was created, with no full-frame Store ever naming it, is **uninitialized** —
/// so the screen goes black, not stale.
///
/// The seq comparison above cannot see this. It checks for a *repeat* — this
/// present's seq against the previous present's — while "never written" is a
/// *state*, and `forget_compositor_mapping` prunes both witnesses on teardown so
/// a re-created surface arrives with neither. Measured on a live boot: the guest
/// re-created its scanout surfaces and presented mid 6 at `gen=0` with
/// `px0=[0,0,0,0]`, and `present_unbacked` fired **zero** times for the whole
/// boot. The guest was awake throughout — see `note_present_backing`.
#[test]
fn present_backing_gate_reports_a_surface_nothing_ever_stored() {
    let w = 1920u32;
    let h = 1080u32;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    state.map_surface(6);

    // Never Stored, first present: the black-screen case.
    assert_eq!(
        state.note_present_backing(6),
        Some(crate::model::PresentBacking::NeverStored),
        "an uninitialized surface must not be presented silently"
    );

    // Reported once per lifetime, not once per present: the witness is recorded
    // on every call, so the next present of the same unbacked surface is the
    // `Restaled` case and carries that reason instead.
    assert_eq!(
        state.note_present_backing(6),
        Some(crate::model::PresentBacking::Restaled { seq: 0 }),
        "the second present of the same surface is a restale, and says so"
    );

    // A surface the guest did Store into is quiet on its first present — this is
    // what keeps the new arm from firing on every healthy mapping.
    state.map_surface(7);
    state.note_dense_frame_published(7, w, h);
    assert_eq!(state.note_present_backing(7), None);

    // And re-creation re-arms it: the teardown prunes the witness, so the next
    // incarnation is judged on its own Stores, not its predecessor's.
    state.unmap_surface(7);
    state.map_surface(7);
    assert_eq!(
        state.note_present_backing(7),
        Some(crate::model::PresentBacking::NeverStored),
        "a re-created surface is uninitialized again until something Stores it"
    );
}

/// An unbacked present is only a *loss* when nothing carries it, and a build
/// that cannot answer must keep the loud reading.
///
/// The structural gate above reads `dense_frame_seq`, which only
/// `publish_surface_store` advances — i.e. only when a Store's pixels reached the
/// mapping's guest pages. The resident rail renders into the registry and skips
/// that write, so "unbacked" stopped implying "shows black": one 524 s boot
/// emitted four `reason=…never_stored` lines each claiming the surface was
/// uninitialized, against exactly one `host_window_slate*` line in the whole run
/// (a `covered=1` boot run) with `presents == offered` and `direct_frac=1.00` in
/// every cadence window bracketing them. A resident carried all four.
///
/// So the channel turns on the carrier, and the `None` arm is the whole content
/// of the rule: `carried != Some(true)` and `carried == Some(false)` differ only
/// where the build cannot tell, which is precisely where demoting a possible
/// black frame to a census would go unnoticed.
#[test]
fn an_unbacked_present_fails_unless_a_resident_positively_carries_it() {
    use super::{carrier_word, unbacked_present_is_a_loss};

    assert!(
        !unbacked_present_is_a_loss(Some(true)),
        "a resident carried the frame, so no guest work was lost — census"
    );
    assert!(
        unbacked_present_is_a_loss(Some(false)),
        "nothing can carry this present, so it shows black — failure channel"
    );
    assert!(
        unbacked_present_is_a_loss(None),
        "a build that cannot answer must not downgrade a possible black frame"
    );

    // The field has to distinguish all three, or the log cannot tell "nothing
    // carried it" from "we did not look" — the difference between a defect and
    // an unmeasured build.
    let words = [
        carrier_word(Some(true)),
        carrier_word(Some(false)),
        carrier_word(None),
    ];
    assert_eq!(words, ["resident", "nothing", "unknown"]);
    assert_eq!(
        words
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        3,
        "each carrier state needs its own word"
    );
}

/// The two arms of the gate must name themselves, and `Restaled` must carry the
/// seq that did not move — two presents quoting the same number are the same
/// guest frame shown twice, which is the whole diagnostic.
#[test]
fn present_backing_names_its_own_reason_and_restale_carries_its_seq() {
    use crate::model::PresentBacking;
    use crate::observe::Decline;

    let restaled = PresentBacking::Restaled { seq: 41 };
    let never = PresentBacking::NeverStored;
    assert_ne!(
        restaled.slug(),
        never.slug(),
        "two distinct findings must not share a slug"
    );
    assert_eq!(
        restaled.fields(),
        vec![("since_seq", "41".to_string())],
        "a restale without its seq is half a diagnostic"
    );
    assert!(
        never.fields().is_empty(),
        "never-stored has no seq to report — there was never one"
    );

    // Rendered through the same builder the emission site uses, so the test pins
    // the line a reader will grep rather than the accessor.
    let line = crate::observe::Emit::decline("present_unbacked", &restaled)
        .field("mid", 4u32)
        .field("carried", super::carrier_word(Some(false)))
        .render();
    assert!(line.contains("reason=present_backing_restaled"), "{line}");
    assert!(line.contains("since_seq=41"), "{line}");
    assert!(line.contains("carried=nothing"), "{line}");
}

/// An AIR-load hold is control flow; a hold that outlives the device is the
/// failure. The two must not share a channel.
///
/// `observe::off` prefixes `OFF `, `observe::fail` does not, and the failure
/// channel is the one place a bad boot explains itself. `translation_order_hold`
/// and `exec_translation_deferred` park a FIFO until an AIR module finishes
/// loading — the packet is retried, not consumed — and both of their resolution
/// lines (`translation_order_release`, `exec_translation_ready`) were already
/// census. Logging only the wait half as a failure put one control-flow pair
/// across both channels, and cost 126 of boot 87's 300 failure lines, 42 %.
///
/// The real loss needs no age, depth or timeout to detect: at reset, a mask still
/// standing means guest packets are parked behind a load that never finished.
#[test]
fn a_translation_hold_is_census_and_only_an_unreleased_one_fails() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);

    // The wait: census only, nothing on the failure channel.
    {
        let cap = crate::observe::FailCapture::start();
        super::note_translation_order_hold(&mut state, 0b101);
        let lines = cap.lines();
        assert!(
            lines
                .iter()
                .any(|l| l.starts_with("OFF translation_order_hold")),
            "the hold must still be logged, on the census channel: {lines:?}"
        );
        assert!(
            !lines
                .iter()
                .any(|l| l.starts_with("translation_order_hold")),
            "a resolver saying `not ready yet` is not a failure: {lines:?}"
        );
    }

    // Released while the device is still alive: nothing failed.
    {
        let cap = crate::observe::FailCapture::start();
        super::release_translation_order_holds(&mut state);
        assert_eq!(state.translation_order_hold_mask, 0);
        state.reset();
        assert!(
            !cap.lines()
                .iter()
                .any(|l| l.starts_with("translation_hold_unreleased")),
            "a hold that released before teardown is not a loss: {:?}",
            cap.lines()
        );
    }

    // A hold still standing at reset IS a loss, and it says so on the failure
    // channel carrying the masks it read.
    {
        let mut stuck = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        super::note_translation_order_hold(&mut stuck, 0b110);
        stuck.translation_deferred_mask = 0b10;
        let cap = crate::observe::FailCapture::start();
        stuck.reset();
        let line = cap.one("translation_hold_unreleased");
        assert!(
            line.contains("held_mask=0x6") && line.contains("producer_mask=0x2"),
            "the failure must carry what it read: {line}"
        );
        assert_eq!(stuck.translation_order_hold_mask, 0, "reset still resets");
    }
}

/// The display channel's flush fence is a real command, not an unknown opcode.
///
/// The guest emits it from the failure and teardown legs of a present, on the
/// display channel, carrying stamps and no payload. It was landing in the
/// unknown-opcode arm, which reports a real Apple command as a device defect —
/// and does so on the display channel exactly when the present path is already
/// in trouble and someone is reading the log.
///
/// Retiring the stamps is the whole contract, so the assertion is that the
/// packet completes and reports nothing, while a payload — which the command
/// cannot have, since it allocates no command bytes — still says so.
#[test]
fn the_display_flush_fence_is_a_named_command_and_not_a_defect() {
    use crate::runtime::drain::store_route_count;
    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let fence = |plen: usize| Packet {
        opcode: CHILD_OP_NOP,
        stamp_waits: Vec::new(),
        total_size: PACKET_HEADER_LEN + plen as u32,
        completion_stamp: 0,
        payload: vec![0u8; plen],
        next_head: 0,
    };

    let n = store_route_count("child_nop");
    let disposition = process_child_packet(&mut state, &mut host, 4, &fence(0));
    assert_eq!(
        disposition,
        ChildPacketDisposition::Complete,
        "the stamps must retire, or the guest waits on this fence forever"
    );
    assert_eq!(
        store_route_count("child_nop"),
        n + 1,
        "the command is counted like every other decoded one"
    );
    assert!(
        !state.fails.iter().any(|e| matches!(
            e,
            FailEvent::UnknownChildOpcode {
                opcode: CHILD_OP_NOP,
                ..
            }
        )),
        "a real Apple command must not be reported as an unknown opcode"
    );

    // A payload is the one thing that would falsify the stamps-only reading.
    process_child_packet(&mut state, &mut host, 4, &fence(4));
    assert_eq!(store_route_count("child_nop"), n + 2);
}

/// A display-transaction command longer than its declared trailer must alarm,
/// and a conformant one must stay silent.
///
/// The guest's display pipe serializes only plane 0's surface id into a
/// fixed-size command, so decoding a single surface is the whole contract. The
/// one thing that would falsify that is the payload growing past its declared
/// size, which would mean a plane list had appeared and this decode had silently
/// become a truncation. That is the alarm; the conformant sizes are not events.
///
/// Latched per `(opcode, payload_len)`, because a guest that grew the command
/// grew it for every frame.
#[test]
fn an_overlong_display_transaction_alarms_once_per_shape() {
    use crate::runtime::drain::store_route_count;
    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
    let packet = |opcode: u16, plen: usize| Packet {
        opcode,
        stamp_waits: Vec::new(),
        total_size: PACKET_HEADER_LEN + plen as u32,
        completion_stamp: 0,
        payload: vec![0u8; plen],
        next_head: 0,
    };

    // Every command at exactly its declared trailer is conformant and silent.
    let quiet = store_route_count("display_txn_payload_overlong");
    note(&mut state, 5, &packet(CHILD_OP_DISPLAY_TRANSACTION2, 0x0c));
    note(&mut state, 5, &packet(CHILD_OP_DISPLAY_TRANSACTION3, 0x24));
    note(&mut state, 4, &packet(CHILD_OP_DISPLAY_SWAP, 0x0c));
    assert_eq!(
        store_route_count("display_txn_payload_overlong"),
        quiet,
        "a command at its declared size is the contract, not an event"
    );
    assert!(state.display.txn_payload_samples.is_empty());

    // A payload past the trailer is the one thing that falsifies the decode.
    for _ in 0..8 {
        note(&mut state, 5, &packet(CHILD_OP_DISPLAY_TRANSACTION2, 64));
    }
    assert_eq!(
        store_route_count("display_txn_payload_overlong"),
        quiet + 8,
        "the counter carries the magnitude on every occurrence"
    );
    assert_eq!(
        state.display.txn_payload_samples.len(),
        1,
        "but the line is latched per shape"
    );

    // The gamma variant's trailer is larger, so 0x24 is conformant there while
    // the same length would be overlong for op6 - the sizes are per command.
    note(&mut state, 5, &packet(CHILD_OP_DISPLAY_TRANSACTION2, 0x24));
    assert_eq!(
        state.display.txn_payload_samples.len(),
        2,
        "0x24 is this command's overlong even though it is op7's exact size"
    );
}

/// The alarm carries the bytes it is alarming about, and it does not explain
/// op8 with op6's structure.
///
/// Both halves come from the same reading of a driven arm64 boot, where this
/// alarm fires on **every** present — 1 668 times in 212 s, always `op=0x8
/// plen=40 trailer=12` — and printed a plane-list explanation that cannot apply
/// to op8, which serializes no transaction. So the message was wrong about a
/// first-class pathway's normal traffic, and the reader was told nothing about
/// the 28 undecoded bytes that would let anyone act on it.
///
/// The line is latched per `(opcode, length)`, so the dump costs one line per
/// shape per boot. Without it, learning what those bytes hold needs a rebuild
/// and a reboot.
#[test]
fn the_overlong_alarm_dumps_the_tail_and_explains_the_right_command() {
    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_ARM64E);
    let packet = |opcode: u16, payload: Vec<u8>| Packet {
        opcode,
        stamp_waits: Vec::new(),
        total_size: PACKET_HEADER_LEN + payload.len() as u32,
        completion_stamp: 0,
        payload,
        next_head: 0,
    };

    // op8 at the length the arm64 guest actually sends: 12 known, 28 unnamed.
    let mut body = vec![0u8; 12];
    body.extend((0..28u8).map(|i| 0xa0 + i));
    let cap = crate::observe::FailCapture::start();
    note(&mut state, 4, &packet(CHILD_OP_DISPLAY_SWAP, body));
    let lines = cap.lines();
    let line = lines
        .iter()
        .find(|l| l.contains("display_txn_payload_overlong"))
        .unwrap_or_else(|| panic!("the alarm did not fire: {lines:?}"));
    assert!(
        line.contains("undecoded=28"),
        "the line must say how much it could not read: {line}"
    );
    assert!(
        line.contains("tail=0xa0a1a2a3"),
        "the line must carry the undecoded bytes, starting at the trailer: {line}"
    );
    // Asserted on the *claim*, not on the words. The op8 message does mention a
    // plane list — to say these bytes are not one, which is the conclusion a
    // reader would otherwise draw from the alarm's name and its op6 sibling.
    // What it must not do is assert one may have appeared.
    assert!(
        !line.contains("may have appeared"),
        "op8 serializes no transaction, so a plane list is not a thing it can \
         have grown; claiming one sends the next reader looking for a structure \
         that cannot exist: {line}"
    );
    assert!(
        line.contains("serializes no transaction"),
        "the line must say why the plane-list reading does not apply here, or \
         the next reader re-derives it: {line}"
    );

    // op6 does serialize a transaction, so the plane-list reading is its own
    // and must survive. Same alarm, different explanation.
    let cap = crate::observe::FailCapture::start();
    note(
        &mut state,
        5,
        &packet(CHILD_OP_DISPLAY_TRANSACTION2, vec![7u8; 64]),
    );
    let lines = cap.lines();
    let line = lines
        .iter()
        .find(|l| l.contains("display_txn_payload_overlong"))
        .unwrap_or_else(|| panic!("the alarm did not fire for op6: {lines:?}"));
    assert!(
        line.contains("may have appeared"),
        "op6 is the transaction command, so the plane-list reading is its own \
         and must survive the op8 correction: {line}"
    );

    // The dump is bounded: the length is guest-controlled, so a pathological
    // payload must not turn one latched line into an unbounded one.
    let cap = crate::observe::FailCapture::start();
    note(
        &mut state,
        5,
        &packet(CHILD_OP_DISPLAY_TRANSACTION2, vec![0u8; 4096]),
    );
    let lines = cap.lines();
    let line = lines
        .iter()
        .find(|l| l.contains("display_txn_payload_overlong"))
        .unwrap_or_else(|| panic!("the alarm did not fire for the long payload: {lines:?}"));
    assert!(
        line.contains("undecoded=4084") && line.contains("..."),
        "the full length is reported but the dump is truncated and says so: {line}"
    );
    assert!(
        line.len() < 512,
        "a guest-sized payload must not produce a guest-sized log line: {}",
        line.len()
    );
}

/// A present a resident carried is not a black present.
///
/// When the window presents the engine's own resident, the capture skips the
/// full-frame GPU→CPU readback on purpose, so `frame_bgra` is empty by design and
/// any `max_rgb == 0` test reports black on every present — a live boot logged
/// 1338 `present_black_retain` records against 1312 presents. An always-on
/// failure sink that fires on every healthy frame cannot surface the unhealthy
/// one, so "no pixels" must be its own verdict rather than folded into "black".
#[test]
fn a_resident_carried_present_is_unsampled_not_black() {
    assert_eq!(
        present_content_verdict(&[], 0),
        PresentContentVerdict::Unsampled,
        "no CPU pixels means no evidence, not evidence of black"
    );
    // A genuinely black sampled frame must still be caught — that is the record's
    // whole purpose, and the fix must not trade one blind spot for another.
    assert_eq!(
        present_content_verdict(&[0, 0, 0, 255], 0),
        PresentContentVerdict::Black,
        "an opaque all-zero-RGB frame is still black"
    );
    assert_eq!(
        present_content_verdict(&[0, 0, 0x40, 255], 0x40),
        PresentContentVerdict::Content
    );
}

/// Send a device-info request and read back the pairs the guest would parse.
///
/// `max_key` is exclusive and `count` is a pair capacity — the two words the
/// reply is bound by. Returns the reply page as (key, value) pairs, stopping at
/// the zero terminator the way the guest's own walker does.
#[cfg(test)]
fn device_info_reply(max_key: u32, count: u32) -> Vec<(u32, u32)> {
    const REPLY_PFN: u32 = 0x40;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();

    let mut payload = vec![0u8; TAHOE.reply_pfn_offset() + 4];
    st32(
        &mut payload[TAHOE
            .key_table_len_offset()
            .expect("the newer form carries one")..],
        max_key,
    );
    st32(&mut payload[TAHOE.pair_capacity_offset()..], count);
    st32(&mut payload[TAHOE.reply_pfn_offset()..], REPLY_PFN);
    process_root_packet(
        &mut state,
        &mut host,
        &Packet {
            opcode: ROOT_OP_DEVICE_INFO_TAHOE,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + payload.len() as u32,
            completion_stamp: 0,
            payload,
            next_head: 0,
        },
    );

    let page_size = 1usize << PAGE_SHIFT_ARM64E;
    let gpa = pfn_to_gpa(REPLY_PFN, PAGE_SHIFT_ARM64E);
    let mut out = Vec::new();
    for i in 0..count.min((page_size / DEVICE_INFO_REPLY_PAIR_LEN) as u32) {
        let mut pair = [0u8; DEVICE_INFO_REPLY_PAIR_LEN];
        host.read_gpa(
            gpa + u64::from(i) * DEVICE_INFO_REPLY_PAIR_LEN as u64,
            &mut pair,
        )
        .expect("reply page is readable");
        let key = ld32(&pair[0..4]);
        if key == 0 {
            break;
        }
        out.push((key, ld32(&pair[4..8])));
    }
    out
}

/// The guest's first request word is its key-table *length*, not a highest key.
///
/// It writes `highest_key_it_parses + 1` — 18, against a walker whose jump table
/// runs `case 0` through `case 17` — so a key at or above it is discarded on
/// arrival. The word used to be read as an opcode and never consulted, so the
/// reply named every key in the table whatever the guest said it could take,
/// spending a pair slot per key on a reply the guest asks for exactly once.
///
/// Read as a maximum rather than a length it invents a key that does not exist,
/// which is why the boundary is the assertion: drive a length of 4 and the reply
/// must stop at key 3, not key 4.
#[test]
fn the_device_info_reply_stops_below_the_guests_key_table_length() {
    const PAIRS_PER_PAGE: u32 = (PAGE_SIZE_ARM64E as usize / DEVICE_INFO_REPLY_PAIR_LEN) as u32;
    let keys: Vec<u32> = device_info_reply(4, PAIRS_PER_PAGE)
        .into_iter()
        .map(|p| p.0)
        .collect();
    assert_eq!(
        keys,
        vec![1, 2, 3],
        "a table of four arms is cases 0..=3, so key 3 is the last one worth \
         sending; a key 4 here would be the length read as a maximum"
    );

    let full: Vec<u32> =
        device_info_reply(DEVICE_INFO_KEY_BUFFER_WITH_IOSURFACE + 1, PAIRS_PER_PAGE)
            .into_iter()
            .map(|p| p.0)
            .collect();
    assert_eq!(
        full.last().copied(),
        Some(DEVICE_INFO_KEY_BUFFER_WITH_IOSURFACE),
        "the length this guest sends admits every key its walker has an arm for"
    );
    assert!(
        !full.contains(&(DEVICE_INFO_KEY_BUFFER_WITH_IOSURFACE + 1)),
        "and nothing at or above it: the guest discards those on arrival"
    );
}

/// A `CmdGetComputeInfo` too short to hold its request reports, and does not
/// answer.
///
/// The arm carried its own literal floor and `reply_compute_info` carried the
/// same one, so neither could be wrong without the other being wrong too — and
/// no test built a request at all, so neither was ever exercised. The command
/// is one the guest blocks on: `createComputePipeline` stalls until the reply
/// lands, so a request this device cannot read has to say so.
#[test]
fn a_short_compute_info_request_is_reported_and_not_answered() {
    use crate::protocol::fifo::COMPUTE_INFO_REQUEST_LEN;
    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let short = |plen: usize| Packet {
        opcode: CHILD_OP_GET_COMPUTE_INFO,
        stamp_waits: Vec::new(),
        total_size: PACKET_HEADER_LEN + plen as u32,
        completion_stamp: 0,
        // Non-zero, so a decoder that read past the end of a shorter payload
        // would find a reply address rather than zero and try to use it.
        payload: vec![0xffu8; plen],
        next_head: 0,
    };
    // One byte under, alone in its own capture: the largest length a decoder
    // that read what it could would still accept.
    let cap = crate::observe::FailCapture::start();
    process_child_packet(
        &mut state,
        &mut host,
        4,
        &short(COMPUTE_INFO_REQUEST_LEN - 1),
    );
    let lines = cap.lines();
    let line = lines
        .iter()
        .find(|l| l.contains("reason=compute_info_short"))
        .unwrap_or_else(|| panic!("a short compute-info said nothing: {lines:?}"));
    assert!(
        line.contains("need=24") && line.contains("plen=23"),
        "the line must say what it needed and what it got: {line}"
    );
    for plen in 0..COMPUTE_INFO_REQUEST_LEN - 1 {
        process_child_packet(&mut state, &mut host, 4, &short(plen));
    }
    assert!(
        host.actions.is_empty(),
        "a request with nothing to read must not answer into guest memory"
    );
}

/// `CmdGetComputeInfo` carries the same word and it means the same thing.
///
/// Its guest sends 5 against a walker of `case 0` through `case 4`, so the reply
/// may name keys 1..=4 and no more — there is no key 5, and reading that 5 as a
/// maximum is how "the guest asked about five keys, we answer three" becomes a
/// two-key gap when it is a one-key one.
///
/// That one key is 2, `SupportsIndirectCommandBuffers`, left out on purpose:
/// nothing in either Metal plugin reads the word the guest stores it in. Pinned
/// here so the absence stays a decision instead of looking like an oversight.
#[test]
fn the_compute_info_reply_answers_three_of_the_guests_four_keys() {
    let answered: Vec<u32> = compute_info_caps().iter().map(|&(k, _)| k).collect();
    assert_eq!(
        answered,
        vec![
            COMPUTE_INFO_KEY_MAX_TOTAL_THREADS,
            COMPUTE_INFO_KEY_THREAD_EXECUTION_WIDTH,
            COMPUTE_INFO_KEY_STATIC_THREADGROUP_MEMORY,
        ],
        "key 2 is deliberately absent; adding it means answering it 0, never 1"
    );

    // Under the guest's own table length every answered key is sendable, and one
    // arm shorter drops the last of them. An inclusive read would keep it.
    let sendable = |table_len: u32| -> Vec<u32> {
        answered
            .iter()
            .copied()
            .filter(|&key| key < table_len)
            .collect()
    };
    assert_eq!(
        sendable(COMPUTE_INFO_KEY_STATIC_THREADGROUP_MEMORY + 1),
        answered
    );
    assert_eq!(
        sendable(COMPUTE_INFO_KEY_STATIC_THREADGROUP_MEMORY),
        vec![
            COMPUTE_INFO_KEY_MAX_TOTAL_THREADS,
            COMPUTE_INFO_KEY_THREAD_EXECUTION_WIDTH
        ]
    );
}

/// A device-info reply the guest's own `count` cut short says which keys it lost.
///
/// `count` is how many pairs the guest's buffer holds and it bounds the reply.
/// Ask for fewer than the table offers and the tail is simply not written — and
/// the guest issues this command once, frees the buffer, and answers every later
/// reader from what it parsed, so there is no second, larger ask. Every key still
/// offered at that point is the guest's own key-table length says it parses, so a key dropped
/// here is a capability it spends the rest of the boot without.
///
/// The loss used to be silent, which read exactly like a table with nothing more
/// to say.
#[test]
fn a_device_info_reply_cut_short_by_the_guests_count_names_the_keys_it_lost() {
    const CEILING: u32 = 18;
    const ASKED: u32 = 5;
    let carried = device_info_reply(CEILING, ASKED);
    assert_eq!(
        carried.len(),
        ASKED as usize,
        "a five-pair buffer carries five pairs"
    );

    // What the reply could not carry, derived rather than spelled out so this
    // stays true when a key is added to either end of the table.
    let dropped: Vec<u32> = DEVICE_INFO_CAPS
        .iter()
        .map(|&(key, _)| key)
        .filter(|&key| key < CEILING)
        .skip(ASKED as usize)
        .collect();
    assert!(
        !dropped.is_empty(),
        "a five-pair ask must drop keys the guest parses, or this proves nothing"
    );

    // The fail log is process-global and appended to by every test in this
    // binary, so match this reply's own bound as well as the reason — a bare
    // `find` on the reason returns whichever test emitted first.
    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    let line = log
        .lines()
        .rfind(|l| {
            l.contains("reason=reply_pairs_exhausted") && l.contains(&format!("count={ASKED}"))
        })
        .expect("a truncated device-info reply names itself, and the count that bound it");
    assert!(
        line.contains(&format!("wrote={ASKED}")),
        "and how many pairs it managed: {line}"
    );
    assert!(
        line.contains(&format!("dropped={dropped:?}")),
        "and every key it could not carry: {line}"
    );
}

/// Send a compute-info request under a live task and read the reply page back.
///
/// Returns `(pairs_the_guest_would_parse, the_byte_after_the_reply)`. The
/// second is the point: the guest's walker stops at the zero pair, so a reply
/// that ran one pair long would be invisible to a reader that only checks the
/// pairs.
#[cfg(test)]
fn compute_info_reply(key_table_len: u32, count: u32) -> (Vec<(u32, u32)>, u8) {
    use crate::runtime::host::FakeHost;
    /// Inside the first data page the task fixture maps, and non-zero because
    /// `reply_gva == 0` is the request's own "nothing to answer into".
    const REPLY_GVA: u64 = 0x100;
    const POISON: u8 = 0xa5;

    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    crate::runtime::gva_mem::define_task_pages_arm64e(&mut host, &mut state, 0x40, 1);
    let data_gpa = 0x40u64 << PAGE_SHIFT_ARM64E;
    let poison = vec![POISON; PAGE_SIZE_ARM64E as usize];
    host.write_gpa(data_gpa, &poison).expect("data page");

    let mut payload = vec![0u8; crate::protocol::fifo::COMPUTE_INFO_REQUEST_LEN];
    st32(&mut payload[0..], 1); // the task the fixture defined
    st32(&mut payload[4..], 7); // a pipeline ref; the reply does not read it
    st32(&mut payload[8..], key_table_len);
    st32(&mut payload[12..], count);
    st32(&mut payload[16..], REPLY_GVA as u32);
    assert_eq!(
        process_child_packet(
            &mut state,
            &mut host,
            4,
            &Packet {
                opcode: CHILD_OP_GET_COMPUTE_INFO,
                stamp_waits: Vec::new(),
                total_size: PACKET_HEADER_LEN + payload.len() as u32,
                completion_stamp: 0,
                payload,
                next_head: 0,
            },
        ),
        ChildPacketDisposition::Complete,
        "the guest blocks on this stamp"
    );

    let at = data_gpa + REPLY_GVA;
    let mut out = Vec::new();
    let mut pairs = 0u64;
    while pairs < u64::from(count) {
        let mut pair = [0u8; DEVICE_INFO_REPLY_PAIR_LEN];
        host.read_gpa(at + pairs * DEVICE_INFO_REPLY_PAIR_LEN as u64, &mut pair)
            .expect("reply page is readable");
        let key = ld32(&pair[0..4]);
        if key == 0 {
            break;
        }
        out.push((key, ld32(&pair[4..8])));
        pairs += 1;
    }
    // One byte past everything the reply could legitimately have written: the
    // answers, plus the stop word when there was room and reason for one.
    let terminated = out.len() < count as usize;
    let span = (out.len() + usize::from(terminated)) * DEVICE_INFO_REPLY_PAIR_LEN;
    let mut after = [0u8; 1];
    host.read_gpa(at + span as u64, &mut after)
        .expect("reply page is readable");
    (out, after[0])
}

/// The compute-info reply, driven end to end: what the guest reads back, the
/// stop word it walks to, and the byte past it that must still be untouched.
///
/// Every other test of this command asserts on `compute_info_caps()` — the
/// table, not the reply. The reply was built by a hand-rolled loop that applied
/// the guest's parse ceiling and its pair count as its own two conditions and
/// wrote one pair per guest-memory call, and nothing drove it. It now goes
/// through the same encoder as its device-info twin, which is exactly the kind
/// of change a table-only test cannot see.
#[test]
fn the_compute_info_reply_stops_at_the_terminator_and_writes_nothing_past_it() {
    const POISON: u8 = 0xa5;
    let answers = compute_info_caps();
    let ceiling = COMPUTE_INFO_KEY_STATIC_THREADGROUP_MEMORY + 1;

    // A page of pair slots, as a live guest sends: every answer fits, and the
    // guest asked for more than it got, so it gets the stop word.
    let (pairs, after) = compute_info_reply(ceiling, 512);
    assert_eq!(
        pairs,
        answers.to_vec(),
        "the reply is the table, in table order"
    );
    assert_eq!(
        after, POISON,
        "the stop word is the last thing written; the slot after it is the \
         guest's"
    );

    // The ceiling is exclusive, and one arm shorter drops the last answer
    // rather than keeping it.
    let (short_ceiling, _) = compute_info_reply(COMPUTE_INFO_KEY_STATIC_THREADGROUP_MEMORY, 512);
    assert_eq!(short_ceiling, answers[..2].to_vec());

    // A guest that says it will consume exactly as many pairs as there are
    // answers needs no stop word, and must not be written one.
    let (exact, after_exact) = compute_info_reply(ceiling, answers.len() as u32);
    assert_eq!(exact, answers.to_vec());
    assert_eq!(
        after_exact, POISON,
        "a full reply carries no terminator, so the pair after the last answer \
         is untouched"
    );
}

/// A compute-info reply its guest's own count cut short names the keys it lost.
///
/// The sibling device-info reply has said so since the loss was found to be
/// silent there; this one broke out of its loop and said nothing. Each key it
/// drops is a compute limit the guest then sizes dispatches without, and the
/// command is issued once per pipeline creation with no larger re-ask.
#[test]
fn a_compute_info_reply_cut_short_by_the_guests_count_names_the_keys_it_lost() {
    const ASKED: u32 = 2;
    let answers = compute_info_caps();
    let (pairs, _) = compute_info_reply(COMPUTE_INFO_KEY_STATIC_THREADGROUP_MEMORY + 1, ASKED);
    assert_eq!(pairs.len(), ASKED as usize, "a two-pair ask carries two");

    let dropped: Vec<u32> = answers
        .iter()
        .map(|&(key, _)| key)
        .skip(ASKED as usize)
        .collect();
    assert!(
        !dropped.is_empty(),
        "a two-pair ask must drop an answer, or this proves nothing"
    );
    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    let line = log
        .lines()
        .rfind(|l| {
            l.contains("get_compute_info truncated") && l.contains(&format!("count={ASKED}"))
        })
        .expect("a truncated compute-info reply names itself and the count that bound it");
    assert!(
        line.contains(&format!("wrote={ASKED}")),
        "and how many pairs it managed: {line}"
    );
    assert!(
        line.contains(&format!("dropped={dropped:?}")),
        "and every key it could not carry: {line}"
    );
}

/// `CmdGetComputeInfo` answers the keys the guest asked about, and its
/// threadgroup limits are the host's rather than a fixed pair.
///
/// The reply used to be the constant triple `(1, 1024), (3, 32), (4, 0)`. The
/// guest sizes its dispatches from key 1, so promising 1024 on a device whose
/// `maxComputeWorkGroupInvocations` is the Vulkan floor of 128 hands it a
/// threadgroup the host will reject. Key 3 is vendor-dependent and 32 is only
/// right for some parts.
///
/// Key 4, `staticThreadgroupMemoryLength`, is a property of the pipeline and
/// not of the device, so no device limit answers it and it stays 0 — asserted
/// so that stops being silent.
///
/// Read the name of this test narrowly. A device limit is the *right* answer for
/// key 3 and the *available* one for key 1; `compute_info_caps`'s own doc
/// records that key 1 is per-pipeline in Metal too, so the device number
/// over-promises on that arm. What is asserted below is the pair of invariants
/// that hold whichever number lands there.
#[test]
fn the_compute_info_reply_answers_device_limits_not_a_fixed_triple() {
    let caps = compute_info_caps();
    let keys: Vec<u32> = caps.iter().map(|&(k, _)| k).collect();
    assert_eq!(
        keys,
        vec![
            COMPUTE_INFO_KEY_MAX_TOTAL_THREADS,
            COMPUTE_INFO_KEY_THREAD_EXECUTION_WIDTH,
            COMPUTE_INFO_KEY_STATIC_THREADGROUP_MEMORY,
        ]
    );
    let max_total = caps[0].1;
    let width = caps[1].1;
    // No device may be resolved in a unit test, so the floor is the answer;
    // what must hold either way is that the guest is never handed a
    // threadgroup budget of zero, nor a wave width it would divide by.
    assert!(max_total >= 1, "a zero budget refuses every dispatch");
    assert!(width >= 1, "a zero wave width is not a divisor");
    assert!(
        max_total >= width,
        "a threadgroup that cannot hold one wave is not answerable"
    );
    assert_eq!(
        caps[2].1, 0,
        "static threadgroup memory is per-pipeline; no device limit answers it"
    );
}

/// Every control arm that guards on payload length names a short payload.
///
/// The dispatch is acknowledged either way — `drain_main_fifo` writes the root
/// completion stamp after the match and `drain_child_fifo` calls `write_stamp`
/// the same way — so an arm that merely skips tells the guest its command
/// completed while nothing happened, and leaves no record. The symptom then
/// surfaces arbitrarily far downstream: a channel that never drains, an object
/// list that never binds, a display that never onlines.
///
/// One packet per arm, each one byte too short, asserting the arm both refuses
/// (state unchanged) and says so. `define_task2` and `setup_shared_state`
/// already named theirs and are included so the vocabulary stays one word.
#[test]
fn every_short_control_packet_names_itself() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let short = |opcode: u16, len: usize| Packet {
        opcode,
        stamp_waits: Vec::new(),
        total_size: PACKET_HEADER_LEN + len as u32,
        completion_stamp: 0,
        payload: vec![0u8; len],
        next_head: 0,
    };

    let before_mask = state.open_child_mask();
    for (opcode, need) in [
        (ROOT_OP_DEVICE_INFO_TAHOE, TAHOE.reply_pfn_offset() + 4),
        (
            ROOT_OP_DEVICE_INFO_MONTEREY,
            MONTEREY.reply_pfn_offset() + 4,
        ),
        (ROOT_OP_DEFINE_FIFO, 4),
        (ROOT_OP_FREE_FIFO, 4),
        (ROOT_OP_SET_OBJECT_LIST, SET_OBJECT_LIST_LEN),
        (ROOT_OP_DEFINE_TASK2, DEFINE_TASK_LEN),
    ] {
        process_root_packet(&mut state, &mut host, &short(opcode, need - 1));
    }
    assert_eq!(
        state.open_child_mask(),
        before_mask,
        "a short DEFINE_FIFO must not open a channel"
    );

    for (opcode, need) in [
        (CHILD_OP_SET_OBJECT_LIST, SET_OBJECT_LIST_LEN),
        (CHILD_OP_DELETE_RESOURCE, 8),
        (CHILD_OP_CURSOR_SHOW, 8),
        (CHILD_OP_SETUP_SHARED_STATE, CHILD_SHARED_STATE_LEN),
        // Both FIFOs carry DEFINE_TASK2 and one function handles both, so the
        // root case above does not cover the child site.
        (CHILD_OP_DEFINE_TASK2, DEFINE_TASK_LEN),
        // The map pair reaches the same handler and used to fall through a
        // `plen >= 20` guard into the family's generic census line, which
        // prints the first two words of a record it could not read.
        (CHILD_OP_MAP_MEMORY2, crate::protocol::fifo::MAP_MEMORY_LEN),
        (CHILD_OP_UNMAP_MEMORY, crate::protocol::fifo::MAP_MEMORY_LEN),
        // A delete with no id used to default to task `0`, which is the kernel
        // task, and retire its resolutions without a line.
        (CHILD_OP_DELETE_TASK, crate::protocol::fifo::DELETE_TASK_LEN),
        // A retirement too short to name a surface is one whose backing stays
        // live while the guest recycles the id.
        (
            CHILD_OP_DELETE_IOSURFACE_BACKING2,
            crate::protocol::fifo::DELETE_BACKING_LEN,
        ),
        // The two framing bounds of a destroy command; the record-length one is
        // driven below, where a record can claim more than the payload holds.
        (
            CHILD_OP_DELETE_OBJECT,
            crate::protocol::fifo::DELETE_OBJECT_LEN,
        ),
    ] {
        process_child_packet(&mut state, &mut host, 4, &short(opcode, need - 1));
    }
    assert_eq!(
        state.display.shared_gpa, 0,
        "a short SETUP_SHARED_STATE must not latch a display page"
    );

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    for reason in [
        // The question's name, not the opcode's: `reims_vgpu_core::query`
        // names the two device-info questions, and the drain no longer has a
        // slug of its own to disagree with it.
        "reason=device_info_short site=root",
        "reason=device_info_legacy_short site=root",
        // The kind's name, as with the two device-info questions above: the
        // slug is `reims_vgpu_core::control::ControlKind`'s and the drain has
        // none of its own to disagree with it.
        "reason=define_channel_short site=root",
        "reason=free_channel_short site=root",
        "reason=set_object_list_short site=root",
        "reason=define_task2_short site=root",
        "reason=set_object_list_short site=ch4",
        // `0x25` is `CmdDeleteResource`. The slug must not say `delete_object`:
        // that is `0x28`, a different command with a different payload, and a
        // reader triaging this line would otherwise chase the wrong opcode.
        "reason=delete_resource_short site=ch4",
        "reason=cursor_show_short site=ch4",
        "reason=setup_shared_state_short site=ch4",
        "reason=define_task2_short site=ch4",
        "reason=map_memory2_short site=ch4",
        "reason=unmap_memory_short site=ch4",
        "reason=delete_task_short site=ch4",
        "reason=delete_iosurface_backing2_short site=ch4",
        "reason=delete_object_short site=ch4",
    ] {
        assert!(
            log.contains(reason),
            "a short packet was dropped without naming itself: {reason}"
        );
    }
}

/// Opcode 1 on the root channel is `CmdDisplaySetSharedStatePage`, the same
/// command it is on a child channel, and its payload is read as that command's.
///
/// These two tests replace a pair that pinned the opposite: that the arm read
/// the first payload word as a nested opcode and dispatched on its low half.
/// There is no wrapper command in this protocol — see
/// `ROOT_OP_SETUP_SHARED_STATE` — so what the old arm dispatched on was a
/// display pipe index.
#[test]
fn root_opcode_one_sets_up_the_display_shared_state() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();

    // `{u32 pipe index, u32 shared-state page PFN}`.
    let mut payload = 2u32.to_le_bytes().to_vec();
    payload.extend_from_slice(&0x40u32.to_le_bytes());
    process_root_packet(
        &mut state,
        &mut host,
        &Packet {
            opcode: ROOT_OP_SETUP_SHARED_STATE,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + CHILD_SHARED_STATE_LEN as u32,
            completion_stamp: 0,
            payload,
            next_head: 0,
        },
    );

    assert_eq!(state.display.display_index, 2, "the pipe index latched");
    assert_eq!(
        state.display.shared_gpa,
        state.pfn_gpa(0x40),
        "and the shared-state page, from the same payload the child arm reads"
    );
    assert!(
        !state
            .fails
            .iter()
            .any(|e| matches!(e, FailEvent::UnknownRootOpcode { .. })),
        "opcode 1 has a handler on the root channel"
    );
}

/// The first payload word is this command's own data, and the old arm's reading
/// of it as an opcode is what this pins shut.
///
/// A pipe index of `ROOT_OP_DELETE_TASK` is an ordinary index — Apple's own
/// numbering has no reason to avoid it — and under the old arm it deleted a
/// task instead of registering a display. The task is defined here so the
/// wrong behaviour would be *visible* rather than a no-op: if this ever
/// regresses, the task is gone and the display never latched.
#[test]
fn a_pipe_index_that_looks_like_an_opcode_is_still_a_pipe_index() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.define_task(3, 0x1000, 2);

    let mut payload = u32::from(ROOT_OP_DELETE_TASK).to_le_bytes().to_vec();
    payload.extend_from_slice(&3u32.to_le_bytes());
    process_root_packet(
        &mut state,
        &mut host,
        &Packet {
            opcode: ROOT_OP_SETUP_SHARED_STATE,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + CHILD_SHARED_STATE_LEN as u32,
            completion_stamp: 0,
            payload,
            next_head: 0,
        },
    );

    assert_eq!(
        state.display.display_index,
        u32::from(ROOT_OP_DELETE_TASK),
        "the word is a pipe index and latched as one"
    );
    assert_eq!(
        state.display.shared_gpa,
        state.pfn_gpa(3),
        "and the second word is the page, not a task id"
    );
}

/// A packet whose stamp wait is unmet is held, not run: the head does not move,
/// no completion stamp is written, and the same packet runs once the awaited
/// slot reaches its value.
///
/// Both halves are the test. Only asserting the hold would pass for a device
/// that dropped the packet, and only asserting the release would pass for one
/// that never held. The measured workload is 44 % held, so a release that never
/// fires is a hang rather than a slow path.
#[test]
fn a_packet_whose_stamp_wait_is_unmet_is_parked_until_the_slot_reaches_it() {
    use crate::model::DeviceId;
    use crate::runtime::host::FakeHost;

    const AWAITED_SLOT: u32 = 5;
    const AWAITED_VALUE: u32 = 7;
    const ROOT_STAMP: u32 = 0xABC;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let page_size = 1usize << PAGE_SHIFT_X86;

    let fifo_pfn = 0x40u32;
    let fifo_gpa = (fifo_pfn as u64) << PAGE_SHIFT_X86;
    host.map_range(fifo_gpa, 3 * page_size, 0);
    state.gfx.fifo_base_page = fifo_pfn;
    state.gfx.fifo_start = page_size as u32;
    state.gfx.fifo_length = 2 * page_size as u32;

    // One root packet carrying a single wait record. The opcode has to be one
    // the model has a class for — an unknown one is refused at admission and
    // never parks — so this is a channel definition, whose payload is a zeroed
    // block long enough for its own layout.
    const BODY: u32 = 64;
    let total = PACKET_HEADER_LEN + PACKET_STAMP_LEN + BODY;
    let mut packet = vec![0u8; total as usize];
    st16(&mut packet[PACKET_OPCODE..], ROOT_OP_DEFINE_FIFO);
    st16(&mut packet[PACKET_STAMP_COUNT..], 1);
    st32(&mut packet[PACKET_TOTAL_SIZE..], total);
    st32(&mut packet[PACKET_COMPLETION_STAMP..], ROOT_STAMP);
    st32(&mut packet[PACKET_HEADER_LEN as usize..], AWAITED_SLOT);
    st32(&mut packet[PACKET_HEADER_LEN as usize + 4..], AWAITED_VALUE);
    gpa_map::write_bytes(&mut host, fifo_gpa + page_size as u64, &packet, page_size)
        .expect("seed the root ring");
    state
        .gfx
        .fifo_read
        .store(0, std::sync::atomic::Ordering::Release);
    state.gfx.fifo_written = total;

    let slot_gpa = |slot: u32| fifo_gpa + stamp_slot_offset(slot, page_size as u64).unwrap();
    let read_slot = |host: &FakeHost, slot: u32| {
        let mut v = [0u8; 4];
        crate::runtime::host::HostMemory::read_gpa(host, slot_gpa(slot), &mut v).expect("slot");
        ld32(&v)
    };

    // Slot 5 stands at 0, so the wait is seven short of satisfied.
    drain_main_fifo(&mut state, &mut host);

    // **The head moves and the work does not.** That is the switch: the ring
    // position is free the moment the bytes are out of it, and what holds the
    // packet is an ordering position in the model. The old shape left the head
    // where it was, which stopped every packet behind this one on a wait that
    // had nothing to do with them.
    assert_eq!(
        state
            .gfx
            .fifo_read
            .load(std::sync::atomic::Ordering::Acquire),
        total,
        "the ring position is released at arrival, whatever the model then decides"
    );
    assert_eq!(
        read_slot(&host, 0),
        0,
        "and no completion stamp may be written, or the guest is told a packet \
         finished that never started"
    );
    assert_eq!(
        state.parked.len(),
        1,
        "the packet is held as a position, with its bytes"
    );
    assert!(
        state.pending.main_drain,
        "a parked position is unfinished work: clearing the flag would leave the \
         retry to whichever later doorbell happened to set it again"
    );

    // A second drain with nothing changed finds an empty ring and must not run
    // the parked packet, invent a completion, or lose it.
    drain_main_fifo(&mut state, &mut host);
    assert_eq!(read_slot(&host, 0), 0, "still nothing ran");
    assert_eq!(state.parked.len(), 1, "and the position is still held");

    // Another timeline publishes the awaited stamp.
    gpa_map::write_u32(&mut host, slot_gpa(AWAITED_SLOT), AWAITED_VALUE, page_size)
        .expect("publish the awaited stamp");
    drain_main_fifo(&mut state, &mut host);

    assert_eq!(
        read_slot(&host, 0),
        ROOT_STAMP,
        "once the slot reaches the awaited value the parked packet runs, and its \
         completion stamp lands exactly once"
    );
    assert!(
        state.parked.is_empty(),
        "the position is gone, because it completed"
    );
    assert!(
        !state.pending.main_drain,
        "and a device holding no positions is not pending work"
    );
}

/// A wait naming a slot past the stamp page runs the packet rather than holding
/// it, and says why.
///
/// This is the one case where running unordered is the better answer, and the
/// asymmetry is the reason: an ordering slip loses one packet's ordering, while
/// a timeline parked on a wait nothing can ever satisfy loses the guest. No
/// drain writes a slot outside the page — `write_stamp` returns early on the
/// same `stamp_slot_offset` that refuses here — so the hold would be permanent.
#[test]
fn a_stamp_wait_naming_a_slot_that_cannot_exist_runs_rather_than_parking() {
    use crate::model::DeviceId;
    use crate::runtime::host::FakeHost;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let page_size = 1usize << PAGE_SHIFT_X86;

    let fifo_pfn = 0x40u32;
    let fifo_gpa = (fifo_pfn as u64) << PAGE_SHIFT_X86;
    host.map_range(fifo_gpa, 3 * page_size, 0);
    state.gfx.fifo_base_page = fifo_pfn;
    state.gfx.fifo_start = page_size as u32;
    state.gfx.fifo_length = 2 * page_size as u32;

    // One past the last slot the guest page can hold, which `stamp_slot_offset`
    // refuses and `stamp_slot_index`'s mask does not fold back into range.
    let bad_slot = stamp_slot_count(page_size as u64);
    assert!(
        stamp_slot_offset(bad_slot, page_size as u64).is_none(),
        "the test's premise: this slot has no offset in the stamp page"
    );

    const ROOT_STAMP: u32 = 0xFEED;
    const BODY: u32 = 64;
    let total = PACKET_HEADER_LEN + PACKET_STAMP_LEN + BODY;
    let mut packet = vec![0u8; total as usize];
    st16(&mut packet[PACKET_OPCODE..], ROOT_OP_DEFINE_FIFO);
    st16(&mut packet[PACKET_STAMP_COUNT..], 1);
    st32(&mut packet[PACKET_TOTAL_SIZE..], total);
    st32(&mut packet[PACKET_COMPLETION_STAMP..], ROOT_STAMP);
    st32(&mut packet[PACKET_HEADER_LEN as usize..], bad_slot);
    st32(&mut packet[PACKET_HEADER_LEN as usize + 4..], 0xFFFF);
    gpa_map::write_bytes(&mut host, fifo_gpa + page_size as u64, &packet, page_size)
        .expect("seed the root ring");
    state
        .gfx
        .fifo_read
        .store(0, std::sync::atomic::Ordering::Release);
    state.gfx.fifo_written = total;

    drain_main_fifo(&mut state, &mut host);

    assert_eq!(
        state
            .gfx
            .fifo_read
            .load(std::sync::atomic::Ordering::Acquire),
        total,
        "an undecidable wait must not stop the timeline, because nothing will \
         ever decide it"
    );
    assert!(
        state.parked.is_empty(),
        "and nothing is parked on it, because a position waiting on a slot no \
         drain writes would hold its channel's publication head forever"
    );
}

/// The ledger separates a word this device is *holding* from one it has already
/// handed to the publication rail.
///
/// That split is the whole reason the type exists: only the first is publishable
/// early, and a repair that treats the second the same way releases packets
/// ahead of the settle `write_stamp` carries — which is the unsoundness that got
/// an earlier attempt reverted.
#[test]
fn the_stamp_ledger_separates_a_held_word_from_one_already_published() {
    let page = 1u64 << PAGE_SHIFT_X86;
    let mut ledger = StampLedger::default();
    let wait = |slot: u32, value: u32| StampWait { index: slot, value };

    assert_eq!(
        ledger.classify(wait(2, 1)),
        UnmetSource::Absent,
        "a slot nothing has stamped is work we have not been given"
    );

    // Coalescing holds a word: publishable early, and safe to publish.
    ledger.owe(2, 0x398f, page);
    assert_eq!(
        ledger.classify(wait(2, 0x398f)),
        UnmetSource::Coalesced,
        "a value in the latch is one this device could pay now"
    );
    assert_eq!(
        ledger.classify(wait(2, 0x3990)),
        UnmetSource::Absent,
        "but only up to the value actually latched"
    );

    // Publication moves it: nothing left to pay, the GPU has to retire it.
    ledger.wrote(2, 0x398f, page);
    assert_eq!(
        ledger.classify(wait(2, 0x398f)),
        UnmetSource::Queued,
        "once write_stamp has it, holding is correct and there is nothing to flush"
    );

    // A later latch on the same slot is owed again, above what was published.
    ledger.owe(2, 0x3995, page);
    assert_eq!(
        ledger.classify(wait(2, 0x3995)),
        UnmetSource::Coalesced,
        "the next coalesced run is owed even though an older value was published"
    );

    // Publishing past an owed value clears it rather than leaving a stale debt.
    ledger.wrote(2, 0x3999, page);
    assert_eq!(
        ledger.classify(wait(2, 0x3995)),
        UnmetSource::Queued,
        "a write past the latch drops the debt, or the ledger would offer to pay \
         a word that has already gone"
    );

    // Wrapping order, matching `StampWait::satisfied_by`.
    ledger.owe(6, 0xffff_fff0, page);
    ledger.owe(6, 4, page);
    assert_eq!(
        ledger.classify(wait(6, 4)),
        UnmetSource::Coalesced,
        "across the wrap 4 is later than 0xffff_fff0, by signed difference"
    );

    // The bound: a slot the stamp page cannot hold is never recorded, so neither
    // map can outgrow the page and there is nothing to scan for.
    let bad = stamp_slot_count(page);
    assert!(stamp_slot_offset(bad, page).is_none(), "test premise");
    ledger.owe(bad, 1, page);
    ledger.wrote(bad, 1, page);
    assert_eq!(
        ledger.classify(wait(bad, 1)),
        UnmetSource::Absent,
        "a slot outside the stamp page is refused by both recorders"
    );
}

/// A stamp wait is decided by a signed wrapping difference, so a slot that has
/// wrapped past `u32::MAX` still satisfies the waits behind it.
///
/// Every assertion here that mentions the wrap is one a plain `current >= value`
/// gets backwards, and getting it backwards is not a slow path: it reports every
/// wait on the far side of the wrap as unmet forever, because the slot only ever
/// climbs further away. A slot ticking once per submission reaches the wrap on a
/// long-lived channel, so this is a property of running for a while rather than
/// of any unusual guest.
#[test]
fn a_stamp_wait_is_decided_by_a_signed_wrapping_difference() {
    let wait = |value| StampWait { index: 1, value };

    assert!(wait(5).satisfied_by(5), "reaching the value satisfies it");
    assert!(wait(5).satisfied_by(6), "and so does passing it");
    assert!(!wait(5).satisfied_by(4), "one short does not");

    // The slot has wrapped; the awaited value has not. `4 >= 0xFFFF_FFFE` is
    // false, and the wait is nonetheless long satisfied.
    assert!(
        wait(0xFFFF_FFFE).satisfied_by(4),
        "a slot six ticks past the wrap satisfies a wait two before it"
    );
    assert!(
        !wait(4).satisfied_by(0xFFFF_FFFE),
        "and the reverse pair is still unsatisfied, which a plain unsigned \
         compare also gets backwards"
    );

    // The window the signed difference is correct over, at both ends.
    assert!(
        wait(0).satisfied_by(0x7FFF_FFFF),
        "just inside the window ahead"
    );
    assert!(
        !wait(0).satisfied_by(0x8000_0000),
        "and 2^31 ahead reads as behind, which is the documented limit rather \
         than a bug: the protocol has no way to tell that apart from 2^31 behind"
    );
}

/// The stamp records between a packet's header and its payload are decoded in
/// order, and the payload the decoder hands on begins after them rather than at
/// `+0x0C`.
///
/// Both halves matter and they fail differently. A record read at the wrong
/// stride yields a wait naming a slot nobody writes, which
/// `note_packet_stamp_waits` reports as permanently unmet forever after; a
/// payload that began at `+0x0C` would hand every handler the record bytes as
/// its first words.
#[test]
fn a_packets_stamp_records_are_decoded_and_the_payload_starts_after_them() {
    const STAMPS: u16 = 3;
    let payload = [0xAAu32, 0xBB];
    let stamps_len = STAMPS as usize * PACKET_STAMP_LEN as usize;
    let total = PACKET_HEADER_LEN as usize + stamps_len + payload.len() * 4;

    let mut v = vec![0u8; total];
    v[0..2].copy_from_slice(&ROOT_OP_DEFINE_FIFO.to_le_bytes());
    v[2..4].copy_from_slice(&STAMPS.to_le_bytes());
    v[4..8].copy_from_slice(&(total as u32).to_le_bytes());
    v[8..12].copy_from_slice(&9u32.to_le_bytes());
    // Distinguishable record bytes: if the payload slice began at +0x0C these
    // would show up as the payload's first words.
    for i in 0..stamps_len / 4 {
        let at = PACKET_HEADER_LEN as usize + i * 4;
        v[at..at + 4].copy_from_slice(&(0xF0u32 + i as u32).to_le_bytes());
    }
    for (i, w) in payload.iter().enumerate() {
        let at = PACKET_HEADER_LEN as usize + stamps_len + i * 4;
        v[at..at + 4].copy_from_slice(&w.to_le_bytes());
    }

    let dec = decode_packet(&v, 0, total as u32, RING).unwrap();
    assert_eq!(
        dec.stamp_count(),
        STAMPS,
        "the count is carried, not consumed"
    );
    assert_eq!(
        dec.stamp_waits,
        vec![
            StampWait {
                index: 0xF0,
                value: 0xF1
            },
            StampWait {
                index: 0xF2,
                value: 0xF3
            },
            StampWait {
                index: 0xF4,
                value: 0xF5
            },
        ],
        "each record is one {{index, value}} pair read at an 8-byte stride"
    );
    assert_eq!(
        dec.payload.len(),
        payload.len() * 4,
        "the payload excludes the records"
    );
    assert_eq!(
        ld32(&dec.payload[0..]),
        0xAA,
        "and starts at the first byte after them, not at +0x0C"
    );

    // A packet declaring more records than it has room for is the guest's
    // error, not a short read of ours.
    let mut liar = v.clone();
    liar[2..4].copy_from_slice(&u16::MAX.to_le_bytes());
    assert_eq!(
        decode_packet(&liar, 0, total as u32, RING).unwrap_err(),
        PacketError::BadSize,
        "a stamp count the packet cannot hold is refused before any record is \
         reached, which is what bounds the skip without a constant"
    );
}

/// Every command the reference host dispatches names itself when this device
/// declines it, instead of arriving as an undecodable opcode.
///
/// The host's FIFO drain bounds the header opcode at [`CHILD_OP_MAX`] and
/// indexes one flat table, so each of these numbers reaches a real handler with
/// a real contract. Landing them in the unknown-opcode arm said the opposite —
/// that this device could not tell what the guest had asked for — and for
/// `CmdDeleteObject` it said nothing at all, because that arm was a silent
/// no-op named for a present it never performed.
///
/// The assertion is the pair: the typed record names the command, *and* no
/// unknown-opcode record is raised for the same packet. Only checking the first
/// would pass on a device that raised both.
#[test]
fn a_dispatched_command_this_device_declines_names_itself() {
    let mut host = FakeHost::new();
    // Every packet here is conformant for its own opcode, so what is being
    // tested is the decline and not the shape. The floor differs by command:
    // eight bytes for most, twelve for `CmdDeleteObject`, whose payload is a
    // task id plus a self-describing record that has to reach the decline rather
    // than being refused for its shape on the way there.
    let plain = || {
        let mut payload = vec![0u8; 8];
        st32(&mut payload[0..], 0x11);
        st32(&mut payload[4..], 0x22);
        payload
    };
    let conformant_delete_object = || {
        let mut payload = vec![0u8; 4 + reims_vgpu_wire::ops::destroy::DELETE_TOTAL_LEN as usize];
        st32(&mut payload[0..], 0x11);
        st32(
            &mut payload[4..],
            reims_vgpu_wire::ops::destroy::OPCODE_DELETE_TEXTURE,
        );
        st32(
            &mut payload[8..],
            reims_vgpu_wire::ops::destroy::DELETE_TOTAL_LEN,
        );
        st32(&mut payload[12..], 0x20);
        payload
    };
    for (opcode, expected, payload) in [
        (CHILD_OP_DEBUG, UnimplementedCommand::Debug, plain()),
        (
            CHILD_OP_DELETE_OBJECT,
            UnimplementedCommand::DeleteObject,
            conformant_delete_object(),
        ),
        (
            CHILD_OP_DISPLAY_SLEEP_STATE,
            UnimplementedCommand::DisplaySleepState,
            plain(),
        ),
        (
            CHILD_OP_DISPLAY_SET_PROPERTIES,
            UnimplementedCommand::DisplaySetProperties,
            plain(),
        ),
        (CHILD_OP_DELAY, UnimplementedCommand::Delay, plain()),
    ] {
        let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
        let plen = payload.len() as u32;
        let pkt = Packet {
            opcode,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + plen,
            completion_stamp: 0,
            payload,
            next_head: 0,
        };
        assert_eq!(
            process_child_packet(&mut state, &mut host, 2, &pkt),
            ChildPacketDisposition::Complete,
            "{opcode:#x}: the stamps still retire, or the guest waits forever"
        );
        assert!(
            state.fails.iter().any(|e| matches!(
                e,
                FailEvent::UnimplementedChildCommand { command, opcode: op, .. }
                    if *command == expected && *op == opcode
            )),
            "{opcode:#x} must be reported as {} and not swallowed; got {:?}",
            expected.command(),
            state.fails
        );
        assert!(
            !state
                .fails
                .iter()
                .any(|e| matches!(e, FailEvent::UnknownChildOpcode { .. })),
            "{opcode:#x} is a command with a handler in the host's table, so it \
             must not also read as undecodable"
        );
    }
}

/// A `CmdDeleteObject` packet is bounded before its record is read at all.
///
/// The command carries `{u32 task}` then a record that states its own byte
/// length at offset 8, so a conformant packet is at least twelve bytes and the
/// record must fit in what follows the id. A packet failing either bound is
/// corrupt: nothing may be retired on the strength of it, and the record must
/// not even be parsed — the bound is what stands between a corrupt length and a
/// read past the payload.
///
/// "Was the record reached" is measured by whether any decline from the *record*
/// path fired. Every exit from that path emits one of the `delete_object_…`
/// reasons, and the bound sits upstream of all of them, so a packet the bound
/// should have refused must produce none. Measuring it this way rather than
/// through a teardown counter is deliberate: this arm retires nothing, so a
/// counter of retirements reads zero whether the bound held or not, and the gate
/// would pass on an implementation that had no bound at all.
#[test]
fn a_delete_object_record_must_fit_the_payload_that_carries_it() {
    let mut host = FakeHost::new();
    let packet = |payload: Vec<u8>| Packet {
        opcode: CHILD_OP_DELETE_OBJECT,
        stamp_waits: Vec::new(),
        total_size: PACKET_HEADER_LEN + payload.len() as u32,
        completion_stamp: 0,
        payload,
        next_head: 0,
    };
    // Every exit from the record path, and only those. The packet-shape
    // refusals are `delete_object_short` and `delete_object_record_short`, which
    // share a prefix with these — hence the exact names rather than a prefix
    // test, which would match the very refusals the bound is supposed to raise.
    const RECORD_PATH_REASONS: [&str; 3] = [
        "delete_object_record_malformed",
        "delete_object_not_a_destroy_record",
        "delete_object_ref_unreadable",
    ];

    // One byte under the floor: there is no length word to read at offset 8.
    let under_floor = vec![0u8; 11];
    // At the floor, with a record claiming one byte more than the payload can
    // hold. `9 + 4 = 13 > 12`, so the record overruns by exactly one byte —
    // the off-by-one a `>=` in place of a `>` would let through.
    let mut overrun = vec![0u8; 12];
    st32(&mut overrun[0..], 0x11);
    st32(&mut overrun[8..], 9);
    // A `u32` length whose `+ 4` would wrap. The bound must still refuse it
    // rather than wrapping to a small number and reading the record as valid.
    let mut wrapping = vec![0u8; 12];
    st32(&mut wrapping[0..], 0x11);
    st32(&mut wrapping[8..], u32::MAX);

    for (what, payload) in [
        ("a payload under the floor", under_floor),
        ("a record overrunning by one byte", overrun),
        ("a record length whose bound arithmetic overflows", wrapping),
    ] {
        let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
        let cap = crate::observe::FailCapture::start();
        let disposition = process_child_packet(&mut state, &mut host, 3, &packet(payload));
        let lines = cap.lines();
        drop(cap);
        assert_eq!(
            disposition,
            ChildPacketDisposition::Complete,
            "{what}: a malformed packet must still retire its stamps, or the guest waits forever"
        );
        assert!(
            !lines.iter().any(|l| RECORD_PATH_REASONS
                .iter()
                .any(|r| l.contains(&format!("reason={r}")))),
            "{what}: the bound must refuse it before the record is read; got {lines:?}"
        );
    }

    // Exactly filling the payload is conformant: `8 + 4 = 12`. This is the
    // boundary on the accepting side, so a bound written one too tight would
    // refuse it — and the record path must be reached. The record is an
    // eight-byte one whose opcode is not a destroy, so the exit is the counter
    // for a record this arm will not retire on.
    let mut exact = vec![0u8; 12];
    st32(&mut exact[0..], 0x11);
    st32(&mut exact[8..], 8);
    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
    let cap = crate::observe::FailCapture::start();
    process_child_packet(&mut state, &mut host, 3, &packet(exact));
    let lines = cap.lines();
    drop(cap);
    assert!(
        lines
            .iter()
            .any(|l| l.contains("reason=delete_object_not_a_destroy_record")),
        "a record that exactly fills the payload is well formed and must be read, \
         then refused for its own opcode rather than for the packet's shape; got {lines:?}"
    );

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    for reason in [
        "reason=delete_object_short site=ch3",
        "reason=delete_object_record_short site=ch3",
    ] {
        assert!(
            log.contains(reason),
            "a malformed CmdDeleteObject was dropped without naming itself: {reason}"
        );
    }
}

/// A destroy whose ref names an object of another kind retires no pipeline.
///
/// **A pipeline table entry is a tombstone, so one wrong retirement is
/// permanent.** `PipelineTable::declare` refuses an id it already holds and
/// `peek` answers `AbsentBecause::Retired` for ever after, so a transaction that
/// binds the real pipeline is refused at admission from then on — the whole
/// packet, not the draw.
///
/// It was measured on a guest. A driven macos-26 boot answered `TypeDiffers` for
/// **34 of the 36** destroys that reached this arm, and refused **914 exec
/// packets** with `pipeline_absent_retired`; the macos-15 boot beside it deletes
/// four pipelines and refuses none. The census that says which ref space the
/// destroy's number belongs to already ran two blocks earlier; this is that
/// answer being acted on instead of only counted.
#[test]
fn a_pipeline_destroy_whose_ref_names_another_kind_retires_no_pipeline() {
    use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::runtime::decode::resource::{OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE};
    use reims_vgpu_core::pipeline::PipelineState;
    use reims_vgpu_wire::ops::destroy::{DELETE_TOTAL_LEN, OPCODE_DELETE_RENDER_PIPELINE_STATE};

    const TASK: u32 = 2;
    const REF: u32 = 3;

    // One flat page table, GVA page 0 → data pfn 4, exactly as the object-list
    // fixtures build it: the list entry is read *through the task's own page
    // table*, so a raw GPA write would leave the census reading nothing and the
    // gate untested.
    let mut host = FakeHost::new();
    let dir_gpa = 2u64 << PAGE_SHIFT_X86;
    let root_gpa = 3u64 << PAGE_SHIFT_X86;
    let data_gpa = 4u64 << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x4000, 0);
    host.map_range(data_gpa, 0x200, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    st32(&mut d[..4], 4);
    let _ = host.write_gpa(root_gpa, &d[..4]);

    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
    state.define_task(TASK, 0x1000, 2);
    assert!(state.set_object_list(TASK, 0, 8));

    // The slot holds a **texture**, which is not what a render-pipeline destroy
    // names. Its descriptor is never read: the gate is the entry's type.
    let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(&mut entry, u32::from(OBJECT_TYPE_TEXTURE) | (4u32 << 8));
    entry[4..].copy_from_slice(&0x40u64.to_le_bytes());
    let _ = host.write_gpa(
        data_gpa + u64::from(REF) * OBJECT_LIST_ENTRY_LEN as u64,
        &entry,
    );

    // The name the ref already has, declared directly so the test's subject is
    // the destroy and not the descriptor read. `name_resource` answers from
    // `DeviceState::object_name` before the guest's list, so this is exactly the
    // name the destroy would resolve to and retire.
    assert_eq!(
        crate::runtime::objects::probe_list_entry(&state, &host, TASK, REF).map(|e| e.object_type),
        Some(OBJECT_TYPE_TEXTURE),
        "the fixture's list entry is what the census reads"
    );
    let name = state
        .declare_object(TASK, REF, reims_vgpu_core::lifecycle::Storage::NoBytes)
        .expect("the task is open")
        .id;
    assert!(state.declare_pipeline(name), "declared for the first time");
    state.advance_pipeline(name, PipelineState::Translating);
    state.advance_pipeline(name, PipelineState::Compiling);
    assert!(state.ready_pipeline(name));

    let mut payload = vec![0u8; 4 + DELETE_TOTAL_LEN as usize];
    st32(&mut payload[0..], TASK);
    st32(&mut payload[4..], OPCODE_DELETE_RENDER_PIPELINE_STATE);
    st32(&mut payload[8..], DELETE_TOTAL_LEN);
    st32(&mut payload[12..], REF);
    process_child_packet(
        &mut state,
        &mut host,
        4,
        &Packet {
            opcode: CHILD_OP_DELETE_OBJECT,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + payload.len() as u32,
            completion_stamp: 0,
            payload,
            next_head: 0,
        },
    );

    assert!(
        state.pipeline_is_ready(name),
        "the destroy's ref is a number in another space; retiring the object \
         that merely shares it tombstones a pipeline the guest still binds"
    );
}

/// `CmdDeleteObject` must not retire an object-table entry, however exactly its
/// record's ref matches one.
///
/// This is the arm's load-bearing safety property, and it is a property about
/// **namespaces** rather than about arithmetic. The record's ref belongs to the
/// serializer's per-kind ref space; the object table is keyed by the kernel
/// object-list ref. Both are small integers allocated from zero, so they collide
/// constantly, and a collision is the *only* way the object table can be reached
/// from here — a hit would necessarily be destroying an unrelated object.
///
/// The test therefore rigs the collision deliberately: every ref the packets
/// name is also a live object-table entry under the same task. An implementation
/// that keys the table with the record's ref passes nothing here; it deletes all
/// every one and fails on the first assertion.
///
/// The kinds are exercised across the family. Sampler and render-pipeline
/// records may retire their separate typed registries, but even then must leave
/// the resource table alone. Conversely, resource deletion must not cross into
/// the pipeline registry.
///
/// The two retained registries are Vulkan-arm structures
/// ([`crate::backend::vulkan::pipeline_resolve::retained`] and
/// [`crate::backend::vulkan::pipeline_resolve`] are both gated on them), so the
/// statements that populate them are gated the same way — that part is a fact
/// about the build.
///
/// Whether a delete *empties* one is a different question with a different
/// owner: the running rail's. The Vulkan rail keeps the tables and the guest's
/// delete is their invalidation; the Metal rail resolves both kinds out of the
/// guest's own object list on every use and keeps none, so there is nothing for
/// it to retire. A `--backend both` binary compiles the tables for its Metal
/// boot too, which is why those assertions ask `selected().rail()` rather than
/// the feature. Everything else here — above all that no destroy kind crosses
/// into the resource-list ref space — runs on every arm and every rail.
#[test]
fn a_delete_object_never_retires_an_object_table_entry_its_ref_collides_with() {
    use reims_vgpu_wire::ops::destroy::{
        DELETE_TOTAL_LEN, OPCODE_DELETE_DEPTH_STENCIL_STATE, OPCODE_DELETE_RENDER_PIPELINE_STATE,
        OPCODE_DELETE_SAMPLER_STATE, OPCODE_DELETE_TEXTURE,
    };
    let mut host = FakeHost::new();
    let destroy_packet = |task: u32, record_opcode: u32, object_ref: u32| {
        let mut payload = vec![0u8; 4 + DELETE_TOTAL_LEN as usize];
        st32(&mut payload[0..], task);
        st32(&mut payload[4..], record_opcode);
        st32(&mut payload[8..], DELETE_TOTAL_LEN);
        st32(&mut payload[12..], object_ref);
        Packet {
            opcode: CHILD_OP_DELETE_OBJECT,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + payload.len() as u32,
            completion_stamp: 0,
            payload,
            next_head: 0,
        }
    };

    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
    state.define_task(2, 0x2000, 9);
    assert!(state.set_object_list(2, 3, 64));
    for ref_ in [10, 11, 12, 13, 14, 15] {
        assert!(state.insert_object(2, ref_));
    }
    state.task_sampler_states.register(
        2,
        11,
        std::sync::Arc::new(crate::model::TaskSamplerState {
            descriptor: Default::default(),
        }),
    );
    #[cfg(feature = "backend-vulkan")]
    {
        state
            .task_depth_stencil_states
            .register(2, 12, std::sync::Arc::new(Default::default()));
        crate::backend::vulkan::pipeline_resolve::retained(&state)
            .expect("this rail owns the device's rail-state slot")
            .register(
                2,
                13,
                crate::backend::vulkan::pipeline_resolve::retained_pipeline_for_test(),
            );
        crate::backend::vulkan::pipeline_resolve::retained(&state)
            .expect("this rail owns the device's rail-state slot")
            .register(
                2,
                15,
                crate::backend::vulkan::pipeline_resolve::retained_pipeline_for_test(),
            );
    }

    // Same task, same number, well-formed record: the collision that an arm
    // keying the object table would act on.
    assert_eq!(
        process_child_packet(
            &mut state,
            &mut host,
            4,
            &destroy_packet(2, OPCODE_DELETE_TEXTURE, 10)
        ),
        ChildPacketDisposition::Complete,
        "the stamps must retire, or the guest waits forever"
    );
    assert!(
        state.objects.contains(&(2, 10)),
        "the record's ref is a serializer ref; keying the object table with it \
         destroys an unrelated object that merely shares the integer"
    );

    process_child_packet(
        &mut state,
        &mut host,
        4,
        &destroy_packet(2, OPCODE_DELETE_SAMPLER_STATE, 11),
    );
    assert!(
        state.objects.contains(&(2, 11)),
        "sampler deletion must not cross into the resource-list ref space"
    );
    assert!(
        state.task_sampler_states.get(2, 11).is_none(),
        "the sampler opcode retires the same integer only in its own ref space"
    );

    process_child_packet(
        &mut state,
        &mut host,
        4,
        &destroy_packet(2, OPCODE_DELETE_DEPTH_STENCIL_STATE, 12),
    );
    #[cfg(feature = "backend-vulkan")]
    assert_eq!(
        state.task_depth_stencil_states.get(2, 12).is_none(),
        crate::backend::selected().rail() == crate::backend::Rail::Vulkan,
        "the depth-stencil destroy opcode retires the retained state of the rail \
         that has one — which is the whole invalidation behind retaining it — \
         and leaves the other rail's compiled-but-unused table alone"
    );
    assert!(
        state.objects.contains(&(2, 12)),
        "depth-stencil deletion must not cross into the resource-list ref space"
    );

    process_child_packet(
        &mut state,
        &mut host,
        4,
        &destroy_packet(2, OPCODE_DELETE_RENDER_PIPELINE_STATE, 13),
    );
    #[cfg(feature = "backend-vulkan")]
    assert_eq!(
        crate::backend::vulkan::pipeline_resolve::retained(&state)
            .expect("this rail owns the device's rail-state slot")
            .get(2, 13)
            .is_none(),
        crate::backend::selected().rail() == crate::backend::Rail::Vulkan,
        "the render-pipeline destroy opcode retires the retained state of the \
         rail that has one, and no other rail's"
    );
    assert!(
        state.objects.contains(&(2, 13)),
        "pipeline deletion must not cross into the resource-list ref space"
    );

    assert!(
        state.delete_object(2, 15),
        "the colliding resource-list object exists"
    );
    #[cfg(feature = "backend-vulkan")]
    assert!(
        crate::backend::vulkan::pipeline_resolve::retained(&state)
            .expect("this rail owns the device's rail-state slot")
            .contains(2, 15),
        "resource-list deletion must not cross into the pipeline ref space"
    );

    // The unclaimed number inside the destroy span is refused before the ref is
    // even read, so it must also leave the table alone.
    process_child_packet(&mut state, &mut host, 4, &destroy_packet(2, 0x3ec, 14));
    assert!(
        state.objects.contains(&(2, 14)),
        "0x3ec is unclaimed inside the destroy span and names no destroy at all"
    );
}

/// The kind is decoded off the record and counted per kind.
///
/// The kind lives in the record's own opcode and nowhere else, so a counter that
/// did not read it there would be counting the packet rather than the object.
/// The distribution across kinds was the open question this arm left behind, and
/// the split is what answered it: a driven boot reads 2148 sampler states, 4
/// depth-stencil states, 4 render pipeline states and 2 fences, against 5
/// functions and 3 compute pipeline states this device retains nothing by ref
/// for. One merged counter could not have said that, which is why the split is
/// the measurement and not decoration.
#[test]
fn a_delete_object_counts_the_kind_its_record_names() {
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_wire::ops::destroy::{
        DELETE_TOTAL_LEN, OPCODE_DELETE_FENCE, OPCODE_DELETE_SAMPLER_STATE,
    };
    let mut host = FakeHost::new();
    let destroy_packet = |record_opcode: u32| {
        let mut payload = vec![0u8; 4 + DELETE_TOTAL_LEN as usize];
        st32(&mut payload[0..], 2);
        st32(&mut payload[4..], record_opcode);
        st32(&mut payload[8..], DELETE_TOTAL_LEN);
        st32(&mut payload[12..], 40);
        Packet {
            opcode: CHILD_OP_DELETE_OBJECT,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + payload.len() as u32,
            completion_stamp: 0,
            payload,
            next_head: 0,
        }
    };

    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
    state.define_task(2, 0x2000, 9);

    let sampler_before = store_route_count("child_delete_object_sampler_state");
    let fence_before = store_route_count("child_delete_object_fence");

    process_child_packet(
        &mut state,
        &mut host,
        4,
        &destroy_packet(OPCODE_DELETE_SAMPLER_STATE),
    );
    assert_eq!(
        store_route_count("child_delete_object_sampler_state"),
        sampler_before + 1,
        "the sampler-state kind must be counted under its own name"
    );
    assert_eq!(
        store_route_count("child_delete_object_fence"),
        fence_before,
        "a sampler-state record must not move another kind's counter"
    );

    // The one kind this device holds anything by ref for, and now the one whose
    // delete retires it — see
    // `a_fence_delete_forgets_every_generation_its_ref_held` for the hazard and
    // the boot that established the ref space.
    process_child_packet(
        &mut state,
        &mut host,
        4,
        &destroy_packet(OPCODE_DELETE_FENCE),
    );
    assert_eq!(
        store_route_count("child_delete_object_fence"),
        fence_before + 1,
        "the fence kind must be counted apart, since it is the one that could leak"
    );
}

/// A fence delete forgets every generation its ref held, so a reused ref does
/// not start life already signalled.
///
/// The hazard the retirement removes: a wait is satisfied when the stored
/// generation is at or past its target, so a generation that outlives its fence
/// makes the *next* fence to get that ref pass its first wait with nothing
/// behind it. The census is kept as the denominator — a driven boot answered
/// `delete_fence_ref_live=2`, `delete_fence_ref_absent=0`, which is what
/// established that the serializer's fence ref space and a fence record's are
/// one — and `_absent` climbing is what would put this arm back in doubt.
#[test]
fn a_fence_delete_forgets_every_generation_its_ref_held() {
    use crate::model::{FENCE_DOMAIN_BLIT, FENCE_DOMAIN_RENDER};
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_wire::ops::destroy::{DELETE_TOTAL_LEN, OPCODE_DELETE_FENCE};

    let mut host = FakeHost::new();
    let destroy_packet = |object_ref: u32| {
        let mut payload = vec![0u8; 4 + DELETE_TOTAL_LEN as usize];
        st32(&mut payload[0..], 2);
        st32(&mut payload[4..], OPCODE_DELETE_FENCE);
        st32(&mut payload[8..], DELETE_TOTAL_LEN);
        st32(&mut payload[12..], object_ref);
        Packet {
            opcode: CHILD_OP_DELETE_OBJECT,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + payload.len() as u32,
            completion_stamp: 0,
            payload,
            next_head: 0,
        }
    };

    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
    state.define_task(2, 0x2000, 9);

    let live = store_route_count("delete_fence_ref_live");
    let absent = store_route_count("delete_fence_ref_absent");

    process_child_packet(&mut state, &mut host, 4, &destroy_packet(40));
    assert_eq!(
        (
            store_route_count("delete_fence_ref_live"),
            store_route_count("delete_fence_ref_absent"),
        ),
        (live, absent + 1),
        "no generation is held for this ref, so the delete names nothing here"
    );

    // The same ref, now holding a generation in two of the four encoder domains
    // this device splits one guest fence into. Both are that object's and both
    // go.
    state.set_fence_generation(2, FENCE_DOMAIN_BLIT, 40, 3);
    state.set_fence_generation(2, FENCE_DOMAIN_RENDER, 40, 9);
    // A neighbour under a different ref, to say the retirement is keyed and not
    // a sweep of the task.
    state.set_fence_generation(2, FENCE_DOMAIN_RENDER, 41, 5);
    let domains = store_route_count("delete_fence_ref_live_domains");
    process_child_packet(&mut state, &mut host, 4, &destroy_packet(40));
    assert_eq!(
        (
            store_route_count("delete_fence_ref_live"),
            store_route_count("delete_fence_ref_absent"),
        ),
        (live + 1, absent + 1),
        "and now it names one, which is what the boot's reading established"
    );
    assert_eq!(
        store_route_count("delete_fence_ref_live_domains"),
        domains + 2,
        "both domains held it, and the count is how many rather than whether"
    );
    assert_eq!(
        state.fence_generation(2, FENCE_DOMAIN_BLIT, 40),
        None,
        "the deleted fence keeps no generation in any domain"
    );
    assert_eq!(state.fence_generation(2, FENCE_DOMAIN_RENDER, 40), None);
    assert_eq!(
        state.fence_generation(2, FENCE_DOMAIN_RENDER, 41),
        Some(5),
        "and a fence the guest did not delete is untouched"
    );

    // The reuse this exists for: the ref comes back and its first wait must not
    // be satisfied by the dead fence's progress.
    assert_eq!(
        state.fence_generation(2, FENCE_DOMAIN_RENDER, 40),
        None,
        "a reused ref starts unsignalled"
    );
}

/// A destroy of a kind this device retains nothing for says what its ref names
/// in the guest's object list, which is what decides whether a retirement could
/// ever be keyed on it.
///
/// The integer resolving proves little: the object table's own boot found 22
/// sampler-delete refs live under a *different* task, which is a collision. The
/// type agreeing is the evidence, so the census reports the pair and not the
/// hit.
#[test]
fn a_destroy_this_device_cannot_retire_says_what_its_ref_names_in_the_object_list() {
    use crate::runtime::decode::resource::{OBJECT_TYPE_FUNCTION, OBJECT_TYPE_SERIALIZER_OBJECT};
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_wire::ops::destroy::{DELETE_TOTAL_LEN, OPCODE_DELETE_FUNCTION};

    use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_ARM64E);
    // One task, a one-level page table mapping GVA page 0 to data pfn 4, and an
    // object list of eight slots at GVA 0. Stated here rather than shared,
    // because the slots this test fills are its own.
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

    let destroy_packet = |object_ref: u32| {
        let mut payload = vec![0u8; 4 + DELETE_TOTAL_LEN as usize];
        st32(&mut payload[0..], 1);
        st32(&mut payload[4..], OPCODE_DELETE_FUNCTION);
        st32(&mut payload[8..], DELETE_TOTAL_LEN);
        st32(&mut payload[12..], object_ref);
        Packet {
            opcode: CHILD_OP_DELETE_OBJECT,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + payload.len() as u32,
            completion_stamp: 0,
            payload,
            next_head: 0,
        }
    };
    let put_entry = |host: &mut FakeHost, slot: u32, object_type: u8| {
        let mut entry = [0u8; 12];
        st32(&mut entry[0..], u32::from(object_type) | (16u32 << 8));
        entry[4..12].copy_from_slice(&0x100u64.to_le_bytes());
        let _ = host.write_gpa(data_gpa + u64::from(slot) * 12, &entry);
    };

    let absent = store_route_count("delete_object_ref_no_list_entry");
    let agrees = store_route_count("delete_object_ref_type_agrees");
    let differs = store_route_count("delete_object_ref_type_differs");

    // Slot 4 holds nothing.
    process_child_packet(&mut state, &mut host, 4, &destroy_packet(4));
    assert_eq!(
        (
            store_route_count("delete_object_ref_no_list_entry"),
            store_route_count("delete_object_ref_type_agrees"),
            store_route_count("delete_object_ref_type_differs"),
        ),
        (absent + 1, agrees, differs),
        "an empty slot is neither an agreement nor a disagreement"
    );

    // Slot 5 holds a serializer object, which a function destroy does not name.
    put_entry(&mut host, 5, OBJECT_TYPE_SERIALIZER_OBJECT);
    process_child_packet(&mut state, &mut host, 4, &destroy_packet(5));
    assert_eq!(
        (
            store_route_count("delete_object_ref_type_agrees"),
            store_route_count("delete_object_ref_type_differs"),
        ),
        (agrees, differs + 1),
        "a live entry of the wrong type is a collision and says the spaces differ"
    );

    // Slot 6 holds a function, which is the pairing that would say one space.
    put_entry(&mut host, 6, OBJECT_TYPE_FUNCTION);
    process_child_packet(&mut state, &mut host, 4, &destroy_packet(6));
    assert_eq!(
        (
            store_route_count("delete_object_ref_type_agrees"),
            store_route_count("delete_object_ref_type_differs"),
        ),
        (agrees + 1, differs + 1),
        "and the type agreeing is the evidence a retirement could be keyed here"
    );
}

/// The host's retired slots are reported as retired, not as undecodable.
///
/// Fifteen opcodes share one deprecated handler on the reference host: it
/// accepts the packet, ignores the payload and retires the stamps. A guest still
/// emitting one is an old guest, which is a different thing from a guest sending
/// something this device cannot decode — and the two used to produce the same
/// record.
///
/// Driven off the closure ledger — the rows `reims_vgpu_core::control` calls
/// [`ControlKind::RetiredSlot`] — rather than a list beside the arm, so a slot
/// the ledger judges retired cannot be left without one. This device used to
/// keep its own transcription of the fifteen and drive this test off that, so
/// the test and the arm shared a source and neither could catch the other being
/// wrong about which numbers they are.
#[test]
fn a_retired_slot_is_reported_as_retired_and_not_as_undecodable() {
    let mut host = FakeHost::new();
    let retired: Vec<u16> = reims_vgpu_protocol::packets::LEDGER
        .iter()
        .filter(|p| p.channel == crate::protocol::packets::Channel::Child)
        .filter(|p| {
            reims_vgpu_core::control::ControlKind::of(p.channel, p.opcode)
                == Some(reims_vgpu_core::control::ControlKind::RetiredSlot)
        })
        .map(|p| p.opcode)
        .collect();
    assert_eq!(retired.len(), 15, "the reference host's retired slots");
    for opcode in retired {
        let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
        let pkt = Packet {
            opcode,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN,
            completion_stamp: 0,
            payload: Vec::new(),
            next_head: 0,
        };
        assert_eq!(
            process_child_packet(&mut state, &mut host, 2, &pkt),
            ChildPacketDisposition::Complete
        );
        assert!(
            state.fails.iter().any(|e| matches!(
                e,
                FailEvent::UnimplementedChildCommand {
                    command: UnimplementedCommand::Deprecated,
                    opcode: op,
                    ..
                } if *op == opcode
            )),
            "{opcode:#x} is one of the host's retired slots; got {:?}",
            state.fails
        );
        assert!(
            !state
                .fails
                .iter()
                .any(|e| matches!(e, FailEvent::UnknownChildOpcode { .. })),
            "{opcode:#x} has a handler on the host, so it is not undecodable"
        );
    }
}

/// `CmdSynchronizeAndDiscardResources` and `CmdDiscardResources` carry the same
/// record layout as `CmdSynchronizeResources`, and this device reads all three
/// with one decoder.
///
/// The reference host validates the three with byte-for-byte the same check —
/// `{u32 task, u32 count}` then `count` four-byte object ids — so a payload that
/// is well-formed for one is well-formed for all of them. The way to prove these
/// two reach that decoder rather than being waved through is to hand them a
/// payload that *fails* it: a count the packet has no room for. A swallowed
/// command would report nothing.
///
/// A valid packet is fully handled and a malformed one proves the payload
/// reached the shared decoder.
#[test]
fn the_discarding_commands_share_the_synchronize_record_layout() {
    use crate::protocol::fifo::ResourceListDecodeError;
    let mut host = FakeHost::new();
    // One record: header plus a single four-byte object id.
    let mut good = vec![0u8; 12];
    st32(&mut good[0..], 7); // task
    st32(&mut good[4..], 1); // count
    st32(&mut good[8..], 0x2a); // object id
                                // The same header claiming four records in a packet that holds one.
    let mut liar = good.clone();
    st32(&mut liar[4..], 4);

    for opcode in [
        CHILD_OP_SYNCHRONIZE_AND_DISCARD_RESOURCES,
        CHILD_OP_DISCARD_RESOURCES,
    ] {
        let packet = |payload: &Vec<u8>| Packet {
            opcode,
            stamp_waits: Vec::new(),
            total_size: PACKET_HEADER_LEN + payload.len() as u32,
            completion_stamp: 0,
            payload: payload.clone(),
            next_head: 0,
        };

        let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
        assert_eq!(
            process_child_packet(&mut state, &mut host, 2, &packet(&good)),
            ChildPacketDisposition::Complete
        );
        assert!(
            !state
                .fails
                .iter()
                .any(|e| matches!(e, FailEvent::UnimplementedChildCommand { .. })),
            "{opcode:#x}: a valid discard is fully handled"
        );
        assert!(
            !state
                .fails
                .iter()
                .any(|e| matches!(e, FailEvent::UnknownChildOpcode { .. })),
            "{opcode:#x} has a handler on the host and a decoder here"
        );

        // The malformed one proves the payload reached the decoder.
        let cap = crate::observe::FailCapture::start();
        let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
        process_child_packet(&mut state, &mut host, 2, &packet(&liar));
        let line = cap.one("map_family");
        assert!(
            line.contains(&format!(
                "reason={}",
                crate::observe::Decline::slug(&ResourceListDecodeError::Truncated {
                    count: 4,
                    plen: 12,
                    need: 24,
                })
            )),
            "{opcode:#x}: a record layout the guest and this device disagree on \
             must name the check that refused; got {line}"
        );
    }
}

/// Each member of the map/lifecycle family is dispatched under its own identity.
///
/// Six opcodes share one body, and the member is bound at the dispatch arm
/// rather than re-read from the packet inside that body. That makes a dead
/// branch a compile error, but it moves the risk to the binding: an arm naming
/// the wrong member compiles, and the packet then silently runs another
/// command's branch. Nothing in the toolchain sees that, so it is asserted here
/// by the effect each branch has and no other has.
///
/// `CmdMapMemory2` and `CmdUnmapMemory` are deliberately absent. They share one
/// branch and this device does not yet do anything different for the two, so
/// there is no effect that separates them — asserting one would be asserting the
/// log string, not the behaviour. The pair's shared branch is covered by
/// `map_memory2_does_not_flush_gva_host_cache_on_wire` and the view-retire tests.
#[test]
fn each_map_family_command_takes_its_own_branch() {
    use crate::protocol::endian::st32;
    use crate::protocol::fifo::CHILD_INVALIDATE_PAGEON_FLAGS;

    const MAPPING: u32 = 0x2a;

    // `{u32 task, u32 count}` then one 8-byte validity record: the invalidate
    // layout. The synchronise family reads the same header and 4-byte ids, so
    // this payload is well-formed for either and the opcode is the only thing
    // that decides which decoder sees it.
    let mut invalidate = vec![0u8; 16];
    st32(&mut invalidate[4..], 1);
    st32(&mut invalidate[8..], MAPPING);
    st32(&mut invalidate[12..], CHILD_INVALIDATE_PAGEON_FLAGS);
    // `{u32 task, u32 count}` then one 4-byte object id.
    let mut synchronize = vec![0u8; 12];
    st32(&mut synchronize[4..], 1);
    st32(&mut synchronize[8..], MAPPING);
    // `{u32 object_id, u32 task_id}`.
    let mut delete_backing = vec![0u8; 8];
    st32(&mut delete_backing[0..], MAPPING);

    // Each row is a payload and the two effects that tell this branch from its
    // five neighbours: whether the named mapping's content generation moved, and
    // whether the command reported itself unimplemented.
    for (opcode, payload, bumps_generation, declined) in [
        (CHILD_OP_INVALIDATE_RESOURCES, &invalidate, true, None),
        (CHILD_OP_SYNCHRONIZE_RESOURCES, &synchronize, false, None),
        (
            CHILD_OP_SYNCHRONIZE_AND_DISCARD_RESOURCES,
            &synchronize,
            false,
            None,
        ),
        (
            CHILD_OP_DELETE_IOSURFACE_BACKING2,
            &delete_backing,
            false,
            None,
        ),
    ] {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
        assert!(state.map_surface(MAPPING));
        state.mappings.get_mut(&MAPPING).unwrap().content_generation = 7;

        assert_eq!(
            process_child_packet(
                &mut state,
                &mut host,
                4,
                &Packet {
                    opcode,
                    stamp_waits: Vec::new(),
                    total_size: PACKET_HEADER_LEN + payload.len() as u32,
                    completion_stamp: 0,
                    payload: payload.clone(),
                    next_head: 0,
                },
            ),
            ChildPacketDisposition::Complete,
            "{opcode:#x}: the guest is waiting on the stamp whatever the branch did"
        );

        // `CmdInvalidateResources` is the only member that reads validity
        // records, so the generation bump is its signature. A mis-bound arm
        // either loses the bump or produces one for a command that never
        // invalidates anything.
        let generation = state.mappings.get(&MAPPING).map(|m| m.content_generation);
        if bumps_generation {
            assert_eq!(
                generation,
                Some(8),
                "{opcode:#x}: this command clears host validity, so the mapping's \
                 content generation must move"
            );
        } else {
            assert_ne!(
                generation,
                Some(8),
                "{opcode:#x}: this command does not touch validity, so a bump means \
                 it ran the invalidate branch"
            );
        }

        // `CmdDeleteIOSurfaceBacking2` is the only member that ends a backing's
        // lifetime, and with no resolved pages there is nothing a stale delete
        // could hurt, so it unmaps outright. The slot survives — the id can be
        // reused — so what separates it is the mapped bit, not the entry.
        assert_eq!(
            state.mappings[&MAPPING].mapped,
            opcode != CHILD_OP_DELETE_IOSURFACE_BACKING2,
            "{opcode:#x}: only the backing delete unmaps the surface"
        );

        let reported = state.fails.iter().find_map(|e| match e {
            FailEvent::UnimplementedChildCommand { command, .. } => Some(*command),
            _ => None,
        });
        assert_eq!(
            reported, declined,
            "{opcode:#x}: the half of this command that is not implemented must be \
             named, and a fully-handled one must report nothing"
        );
    }
}

/// An opcode above the host's dispatch ceiling is reported apart from one that
/// is merely unassigned.
///
/// Both leave the guest's work undone, and both raise the unknown-opcode record,
/// so on their own they read identically. They are not the same event: an
/// unassigned slot in range is a guest asking for a command this host generation
/// does not have, while a value the reference host refuses before it indexes its
/// table means the header itself is wrong — a desynced ring or a corrupt packet,
/// which is a transport bug and not a missing feature.
#[test]
fn an_opcode_past_the_dispatch_ceiling_is_reported_apart_from_an_unassigned_slot() {
    let mut host = FakeHost::new();
    let packet = |opcode: u16| Packet {
        opcode,
        stamp_waits: Vec::new(),
        total_size: PACKET_HEADER_LEN,
        completion_stamp: 0,
        payload: Vec::new(),
        next_head: 0,
    };

    // 0x0b is inside the ceiling and has no handler on the reference host.
    let cap = crate::observe::FailCapture::start();
    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
    process_child_packet(&mut state, &mut host, 2, &packet(0x0b));
    assert!(
        state
            .fails
            .iter()
            .any(|e| matches!(e, FailEvent::UnknownChildOpcode { opcode: 0x0b, .. })),
        "an unassigned in-range slot is still an unknown opcode"
    );
    assert!(
        !cap.lines()
            .iter()
            .any(|l| l.starts_with("child_opcode_out_of_range")),
        "an in-range opcode is not a malformed header; got {:?}",
        cap.lines()
    );
    drop(cap);

    let cap = crate::observe::FailCapture::start();
    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
    process_child_packet(&mut state, &mut host, 2, &packet(CHILD_OP_MAX + 1));
    let line = cap.one("child_opcode_out_of_range");
    assert!(
        line.contains(&format!("opcode={:#x}", CHILD_OP_MAX + 1))
            && line.contains(&format!("max={CHILD_OP_MAX:#x}")),
        "the line has to carry both the value and the ceiling it broke; got {line}"
    );
    assert!(
        state
            .fails
            .iter()
            .any(|e| matches!(e, FailEvent::UnknownChildOpcode { .. })),
        "the guest's work is still lost, so the record it shares with an \
         unassigned slot is still raised"
    );
}

/// A declined command says its piece once and is counted every time.
///
/// A command the guest re-issues every frame would put a line per packet into
/// the always-on log — and a flood is how the refusals around it stop being
/// read. The latch is what makes the record survivable; the route counter is
/// what keeps the rate knowable, because emitters dedupe and counters do not,
/// and quoting one as the other is the mistake this pair exists to prevent.
///
/// Driven off `CmdDelay` rather than off `CmdDeleteObject`, whose per-frame rate
/// is what made the latch necessary in the first place — a driven boot sends
/// about 1 990 of those in 25 s. `CmdDelay` reaches the same latch through a
/// packet whose shape is a single fixed floor, so the test is about the latch
/// and not about a record layout.
#[test]
fn a_declined_command_is_latched_in_the_log_and_counted_in_the_census() {
    use crate::runtime::drain::store_route_count;
    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let mut payload = vec![0u8; 8];
    st32(&mut payload[0..], 0x11);
    let pkt = Packet {
        opcode: CHILD_OP_DELAY,
        stamp_waits: Vec::new(),
        total_size: PACKET_HEADER_LEN + payload.len() as u32,
        completion_stamp: 0,
        payload,
        next_head: 0,
    };

    let route = UnimplementedCommand::Delay.slug();
    let before = store_route_count(route);
    let cap = crate::observe::FailCapture::start();
    for _ in 0..3 {
        process_child_packet(&mut state, &mut host, 2, &pkt);
    }
    let lines: Vec<String> = cap
        .lines()
        .into_iter()
        .filter(|l| l.contains(&format!("reason={route}")))
        .collect();
    assert_eq!(
        lines.len(),
        1,
        "three identical declines are one line, or the guest's frame rate sets \
         the log's line rate; got {lines:?}"
    );
    drop(cap);
    assert_eq!(
        store_route_count(route) - before,
        3,
        "the census counts every packet, which is the rate the latched line \
         cannot carry"
    );
}

/// No display signal path may set a pending bit for a class the guest has not
/// enabled — swept over every possible enable word.
///
/// This is the invariant behind the whole two-word mailbox, and it is the one a
/// second reader of the enable mask broke: the guest's ISR clears
/// `pending &= ~enable`, so a bit set outside the mask is not an ignored
/// notification but a word it will never clear, carried forward by every later
/// read-modify-write for the life of the boot.
///
/// The sweep is over all 16 masks rather than the two the x86 rails happen to
/// publish, because "which classes a guest arms" is a per-generation decision
/// and a rail this device has not booted yet is exactly where a missing check
/// would first cost guest work. Both signal paths run under each mask against a
/// zeroed pending word, so anything left set outside the mask came from a path
/// that did not consult it.
#[test]
fn no_display_signal_path_sets_a_bit_the_guest_did_not_enable() {
    use crate::model::DISPLAY_EVENT_MASK_ALL;
    for mask in 0..=DISPLAY_EVENT_MASK_ALL {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let gpa = 0x7c000000u64;
        host.map_range(gpa, PAGE_SIZE_ARM64E as usize, 0);
        state.display.shared_gpa = gpa;
        state.display.display_index = 0;
        state.display.online_acked = true;
        host.put_u32(gpa + DISPLAY_SHARED_ENABLE_MASK, mask);
        host.put_u32(gpa + DISPLAY_SHARED_PENDING, 0);

        let last_us = std::sync::atomic::AtomicU64::new(0);
        signal_display_vbl_at(&mut state, &mut host, &last_us, 5_000_000);
        signal_display_present_complete(&mut state, &mut host);

        let mut pending = [0u8; 4];
        host.read_gpa(gpa + DISPLAY_SHARED_PENDING, &mut pending)
            .unwrap();
        let set = ld32(&pending);
        assert_eq!(
            set & !mask,
            0,
            "enable=0x{mask:x} left pending=0x{set:x}: bit(s) 0x{:x} are outside \
             the guest's mask and its ISR will never clear them",
            set & !mask
        );
    }
}

/// The display-offline class is never signalled, whatever the guest enables.
///
/// Bit 3 dispatches to the guest's *offline* event source: signalling it tells a
/// healthy guest its display went away. This device creates the scanout once and
/// keeps it for the life of the VM, so the honest state of that bit is always
/// clear — and a guest arming it is ordinary, not a request. The distinction
/// matters because an enable word of `0xe` reads as "the guest wants something
/// we do not send", which is true and harmless, rather than as a missing signal.
#[test]
fn display_offline_is_never_signalled_even_when_the_guest_arms_it() {
    use crate::model::{DISPLAY_EVENT_MASK_ALL, DISPLAY_OFFLINE_EVENT_MASK};
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let gpa = 0x7c000000u64;
    host.map_range(gpa, PAGE_SIZE_ARM64E as usize, 0);
    state.display.shared_gpa = gpa;
    state.display.display_index = 0;
    state.display.online_acked = true;
    // Everything armed, including offline: the most permissive guest there is.
    host.put_u32(gpa + DISPLAY_SHARED_ENABLE_MASK, DISPLAY_EVENT_MASK_ALL);
    host.put_u32(gpa + DISPLAY_SHARED_PENDING, 0);

    let last_us = std::sync::atomic::AtomicU64::new(0);
    signal_display_vbl_at(&mut state, &mut host, &last_us, 5_000_000);
    signal_display_present_complete(&mut state, &mut host);

    let mut pending = [0u8; 4];
    host.read_gpa(gpa + DISPLAY_SHARED_PENDING, &mut pending)
        .unwrap();
    assert_eq!(
        ld32(&pending) & DISPLAY_OFFLINE_EVENT_MASK,
        0,
        "a signalled offline bit tears down a live display pipe"
    );
}

/// A guest that arms only the transaction class still gets a refresh tick.
///
/// The two x86 rails measured here disagree about which event class carries
/// "the frame you were showing is done with": a macOS 13 guest arms VBL and
/// never arms the transaction class, a macOS 11 guest does the exact opposite.
/// The tick has to honour whichever one the guest published, because the guest
/// side of the transaction class retires the *live* transaction and then
/// consumes the ring — a per-refresh edge, not a per-present one.
///
/// Getting this wrong is a hang and not a slowdown. The wait it feeds (the
/// window server's transaction-queue drain) sleeps with no deadline, so a device
/// that raises the class only when a present happens rings it once, the frame
/// goes live, and nothing ever retires it.
#[test]
fn the_refresh_tick_signals_the_transaction_class_when_that_is_what_the_guest_armed() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let gpa = 0x7c000000u64;
    host.map_range(gpa, PAGE_SIZE_ARM64E as usize, 0);
    state.display.shared_gpa = gpa;
    state.display.display_index = 0;
    state.display.online_acked = true;
    // Exactly what a macOS 11 guest publishes: transaction + online + offline,
    // and no VBL for the life of the boot.
    host.put_u32(
        gpa + DISPLAY_SHARED_ENABLE_MASK,
        DISPLAY_PRESENT_EVENT_MASK | DISPLAY_ONLINE_EVENT_MASK | 0x8,
    );
    host.put_u32(gpa + DISPLAY_SHARED_PENDING, 0);

    let last_us = std::sync::atomic::AtomicU64::new(0);
    signal_display_vbl_at(&mut state, &mut host, &last_us, 5_000_000);

    let mut pending = [0u8; 4];
    host.read_gpa(gpa + DISPLAY_SHARED_PENDING, &mut pending)
        .unwrap();
    assert_ne!(
        ld32(&pending) & DISPLAY_PRESENT_EVENT_MASK,
        0,
        "the tick must raise the class the guest armed, or its queue drain never wakes"
    );
    assert_eq!(
        ld32(&pending) & DISPLAY_VBL_EVENT_MASK,
        0,
        "VBL was not armed and must not be written"
    );
    assert_eq!(
        host.actions.len(),
        1,
        "one tick raises at most one interrupt, whatever it signalled"
    );
    assert_eq!(host.actions[0].kind, HostActionKind::IrqGfxPulse);
}

/// Arming both classes still costs one interrupt and one pending write.
///
/// Nothing observed arms both, but the tick reads a guest-owned word and must
/// not turn a mask it did not expect into a doubled doorbell rate.
#[test]
fn a_refresh_tick_that_signals_both_classes_raises_one_interrupt() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let gpa = 0x7c000000u64;
    host.map_range(gpa, PAGE_SIZE_ARM64E as usize, 0);
    state.display.shared_gpa = gpa;
    state.display.display_index = 0;
    state.display.online_acked = true;
    host.put_u32(
        gpa + DISPLAY_SHARED_ENABLE_MASK,
        DISPLAY_VBL_EVENT_MASK | DISPLAY_PRESENT_EVENT_MASK,
    );
    host.put_u32(gpa + DISPLAY_SHARED_PENDING, 0);

    let last_us = std::sync::atomic::AtomicU64::new(0);
    signal_display_vbl_at(&mut state, &mut host, &last_us, 5_000_000);

    let mut pending = [0u8; 4];
    host.read_gpa(gpa + DISPLAY_SHARED_PENDING, &mut pending)
        .unwrap();
    assert_eq!(
        ld32(&pending),
        DISPLAY_VBL_EVENT_MASK | DISPLAY_PRESENT_EVENT_MASK
    );
    assert_eq!(host.actions.len(), 1);
}

/// The interval audit is counted on every verdict, not only on a finding.
///
/// This is the difference between "the audit ran and found nothing" and "the
/// audit never ran", and a dozen panicking macos-26 boots were read as the
/// first when the log could only support the second: the fail line fires for a
/// finding and is deduped on top of that, so silence was the expected output of
/// both. Only the census can say a clean pairing was actually observed.
#[test]
fn the_map_interval_audit_counts_a_clean_pairing_and_not_only_a_finding() {
    use crate::runtime::drain::store_route_count;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();

    // task@0 u32, gva@4 u64, length@12 u64 — the layout `apply_map_family`
    // decodes and the one the guest's own allocator receives.
    let payload = |task: u32, gva: u64, len: u64| {
        let mut p = vec![0u8; 20];
        p[0..4].copy_from_slice(&task.to_le_bytes());
        p[4..12].copy_from_slice(&gva.to_le_bytes());
        p[12..20].copy_from_slice(&len.to_le_bytes());
        p
    };
    let packet = |opcode: u16, payload: Vec<u8>| Packet {
        opcode,
        stamp_waits: Vec::new(),
        total_size: PACKET_HEADER_LEN + payload.len() as u32,
        completion_stamp: 0,
        payload,
        next_head: 0,
    };

    let clean = store_route_count("map_audit_consistent");
    let bad = store_route_count("map_audit_unmap_of_unmapped");

    let gva = 0x4000_0000u64;
    let len = 4u64 << PAGE_SHIFT_X86;
    process_child_packet(
        &mut state,
        &mut host,
        7,
        &packet(CHILD_OP_MAP_MEMORY2, payload(3, gva, len)),
    );
    process_child_packet(
        &mut state,
        &mut host,
        7,
        &packet(CHILD_OP_UNMAP_MEMORY, payload(3, gva, len)),
    );
    assert_eq!(
        store_route_count("map_audit_consistent"),
        clean + 2,
        "a matched map and unmap must leave a positive reading; without one, an \
         audit that never ran reads exactly like an audit that found nothing"
    );

    // The second release of the same range is the shape the guest asserts on,
    // and it must be counted under its own name rather than absorbed.
    process_child_packet(
        &mut state,
        &mut host,
        7,
        &packet(CHILD_OP_UNMAP_MEMORY, payload(3, gva, len)),
    );
    assert_eq!(store_route_count("map_audit_unmap_of_unmapped"), bad + 1);
    assert_eq!(
        store_route_count("map_audit_consistent"),
        clean + 2,
        "a finding must not also be counted as a clean pairing"
    );
}

/// Handing a completion word to a rail leaves the device's ledger, and does
/// **not** move the ordering plane's timeline; putting it where the guest can
/// read it does.
///
/// The regression this pins is one that shipped. Publishing at `write_stamp`
/// looked right on every counter it emitted, and a driven boot priced it
/// exactly: `stamp_wait_model_ahead = 12 824` against `stamp_unmet_queued =
/// 12 824`, the same number twice. `write_stamp` hands the word to the
/// GPU-ordered rail; the word lands when that rail's submission retires. Every
/// wait in between is one the plane called satisfied while the guest could
/// still read the old value — so a cutover publishing there would release,
/// twelve thousand times a boot, work ordered behind GPU work that had not
/// finished. Submission is not completion, and this is the assertion that says
/// so in the two directions that matter.
#[test]
fn owing_a_word_is_not_publishing_it_and_only_the_landing_moves_the_plane() {
    use reims_vgpu_core::identity::StampSlot;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);

    super::note_stamp_no_longer_owed(&mut state, 0, 7);
    assert_eq!(
        state
            .stamp_ledger
            .classify(super::StampWait { index: 0, value: 7 }),
        super::UnmetSource::Queued,
        "the device's ledger has it: `Queued` is its name for a word that has \
         been through the write door and is with a rail"
    );
    assert_eq!(
        state.published_completion_stamp(StampSlot(0)),
        None,
        "and the ordering plane must not: the guest cannot read it yet"
    );

    super::note_stamp_visible(&state, 0, 7, "stamp_visible_inline");
    assert_eq!(
        state.published_completion_stamp(StampSlot(0)).map(|v| v.0),
        Some(7),
        "the landing is the plane's event, and the root slot is the one \
         `write_stamp` never reaches"
    );

    super::note_stamp_visible(&state, 3, 4, "stamp_visible_observed");
    assert_eq!(
        state.published_completion_stamp(StampSlot(3)).map(|v| v.0),
        Some(4)
    );
    assert_eq!(
        state.published_completion_stamp(StampSlot(0)).map(|v| v.0),
        Some(7),
        "slots are independent timelines"
    );
}

/// [`arrival`] — the step both drain loops used to spell for themselves.
mod arrivals {
    use super::*;
    use crate::runtime::host::FakeHost;

    /// Two guest pages: enough for a child ring's page list to be a list.
    const PAGE: u64 = 4096;
    const ROOT_BASE: u64 = 0x10_0000;
    const CHILD_PAGE_0: u64 = 0x20_0000;
    const CHILD_PAGE_1: u64 = 0x21_0000;
    const CAPACITY: u32 = 2 * PAGE as u32;

    /// A host with a root ring and a child ring, both holding `bytes` at
    /// offset zero.
    fn host_with(bytes: &[u8]) -> FakeHost {
        let mut host = FakeHost::new();
        for base in [ROOT_BASE, CHILD_PAGE_0, CHILD_PAGE_1] {
            host.map_range(base, PAGE as usize, 0);
        }
        host.map_range(ROOT_BASE + PAGE, PAGE as usize, 0);
        host.write_gpa(ROOT_BASE, bytes).expect("mapped");
        host.write_gpa(CHILD_PAGE_0, bytes).expect("mapped");
        host
    }

    fn root() -> Ring<'static> {
        Ring::Root {
            base_gpa: ROOT_BASE,
            capacity: CAPACITY,
        }
    }

    const CHILD_PAGES: [u64; 2] = [CHILD_PAGE_0, CHILD_PAGE_1];

    fn child() -> Ring<'static> {
        Ring::Child {
            page_gpas: &CHILD_PAGES,
            capacity: CAPACITY,
            page_shift: PAGE_SHIFT_X86,
        }
    }

    /// The two rings take the same bytes to the same packet.
    ///
    /// **This is what the extraction is for.** One span and one page list are
    /// two ways of reaching bytes, not two definitions of what a packet is —
    /// and the two copies of header-snapshot-decode had already drifted once,
    /// which `packet_snapshot_len`'s own doc records.
    #[test]
    fn both_rings_take_the_same_bytes_to_the_same_packet() {
        let bytes = packet_bytes(ROOT_OP_DEFINE_FIFO, 7, &1u32.to_le_bytes());
        let host = host_with(&bytes);
        let tail = bytes.len() as u32;

        let (Arrival::Packet(from_root), Arrival::Packet(from_child)) = (
            arrival(&root(), &host, 0, tail),
            arrival(&child(), &host, 0, tail),
        ) else {
            panic!("a whole published packet arrives from either ring");
        };
        assert_eq!(from_root, from_child);
        assert_eq!(from_root.opcode, ROOT_OP_DEFINE_FIFO);
        assert_eq!(from_root.completion_stamp, 7);
        assert_eq!(from_root.next_head, tail);
    }

    /// A head that has caught its tail is nothing, and is not a read.
    #[test]
    fn a_head_that_has_caught_the_tail_is_nothing() {
        let host = FakeHost::new();
        assert!(matches!(arrival(&root(), &host, 12, 12), Arrival::Nothing));
    }

    /// A producer mid-write is nothing, on both rings and at both stages: too
    /// few bytes for a header, and a header whose packet is not all there.
    ///
    /// Neither reaches the always-on channel, which is the carve-out
    /// [`PacketError::fault`] makes mechanical and the reason `Nothing` is one
    /// answer rather than three.
    #[test]
    fn a_producer_mid_write_is_nothing_on_either_ring() {
        let bytes = packet_bytes(ROOT_OP_DEFINE_FIFO, 7, &1u32.to_le_bytes());
        let host = host_with(&bytes);

        for ring in [root(), child()] {
            assert!(
                matches!(
                    arrival(&ring, &host, 0, PACKET_HEADER_LEN - 1),
                    Arrival::Nothing
                ),
                "fewer bytes published than a header"
            );
            assert!(
                matches!(
                    arrival(&ring, &host, 0, PACKET_HEADER_LEN),
                    Arrival::Nothing
                ),
                "a header published and its payload not yet"
            );
        }
    }

    /// A tail behind its head is a fault, on either ring.
    #[test]
    fn a_desynced_head_and_tail_is_a_fault() {
        let host = FakeHost::new();
        for ring in [root(), child()] {
            assert!(matches!(
                arrival(&ring, &host, 8, CAPACITY + 64),
                Arrival::Fault(PacketFault::DesyncedHeadTail)
            ));
        }
    }

    /// A declared size the ring itself could never hold is the guest's error,
    /// and comes back as the fault its own ring reports.
    #[test]
    fn a_size_past_the_ring_is_a_fault_from_either_ring() {
        let mut bytes = packet_bytes(ROOT_OP_DEFINE_FIFO, 7, &1u32.to_le_bytes());
        bytes[PACKET_TOTAL_SIZE..PACKET_TOTAL_SIZE + 4]
            .copy_from_slice(&(CAPACITY + 1).to_le_bytes());
        let host = host_with(&bytes);

        for ring in [root(), child()] {
            assert!(matches!(
                arrival(&ring, &host, 0, CAPACITY),
                Arrival::Fault(PacketFault::BadSize)
            ));
        }
    }

    /// Each ring reports a failed read under its own name, which is what the
    /// two drains' `FailEvent` variants are keyed on.
    #[test]
    fn a_read_that_fails_is_named_by_the_ring_it_failed_on() {
        // No RAM at all, so the header read cannot land.
        let mut host = FakeHost::new();
        host.mark_non_ram(0, u64::MAX);

        assert!(matches!(
            arrival(&root(), &host, 0, CAPACITY),
            Arrival::Fault(PacketFault::RootHeaderRead)
        ));
        assert!(matches!(
            arrival(&child(), &host, 0, CAPACITY),
            Arrival::Fault(PacketFault::ChildHeaderRead)
        ));
    }
}
