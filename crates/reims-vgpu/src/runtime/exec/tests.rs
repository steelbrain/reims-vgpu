// A wire constant, and so the same on every rail. It was gated on
// `backend-vulkan` while the only test naming it was the Vulkan-only compute
// preflight; the exec-walk suites now name it 103 times outside any such
// region, and the gate made the Metal arm's test build fail with 103
// `cannot find value` errors that no `--features backend-vulkan` build could
// see. A `cfg` on a wire constant has no correct arm.
use reims_vgpu_wire::OP_HEADER_LEN;

use super::*;
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
use crate::protocol::endian::{st16, st32, st64};
use crate::runtime::host::FakeHost;
use crate::runtime::render_pass::{
    PASS_ATTACH_CLEAR_COLOR, PASS_ATTACH_LOAD_ACTION, PASS_ATTACH_STORE_ACTION, PASS_ATTACH_TEXREF,
    PASS_COLOR_ATTACH_OFF, PASS_COLOR_ATTACH_STRIDE,
};
use reims_vgpu_protocol::pass_action::{MTL_LOAD_ACTION_CLEAR, MTL_STORE_ACTION_STORE};

#[test]
fn render_pass_chain_edges_follow_the_decoded_encoder() {
    assert_eq!(render_pass_chain_position(0, 1), (false, false));
    assert_eq!(render_pass_chain_position(0, 3), (false, true));
    assert_eq!(render_pass_chain_position(1, 3), (true, true));
    assert_eq!(render_pass_chain_position(2, 3), (true, false));
}

/// The abandon line must say how much guest work it dropped.
///
/// This break was silent, and the counter that would have caught it
/// (`metal_draws_fail`) stays 0 on this path because the draw encoded
/// `Ok` — so `packet_failed` is false and the packet-level line is
/// suppressed too. The whole value of the line is the amount lost:
/// breaking at 0 of 8 drops a whole composite, breaking at 7 of 8 drops
/// one draw, and `di` alone does not distinguish them at a glance.
#[test]
fn chain_abandon_reports_how_many_draws_were_lost() {
    let render = |index, total| {
        crate::observe::Emit::decline(
            "draw_chain_abandon",
            &ChainAbandonDecline {
                index,
                total,
                pipeline_ref: 0x41,
            },
        )
        .render()
    };

    let first_of_eight = render(0, 8);
    assert!(
        first_of_eight.contains("reason=draw_chain_abandoned_without_color0"),
        "{first_of_eight}"
    );
    assert!(first_of_eight.contains("di=0/8"), "{first_of_eight}");
    assert!(first_of_eight.contains("lost=7"), "{first_of_eight}");
    assert!(first_of_eight.contains("pipe=65"), "{first_of_eight}");

    // The last record of a list abandons nothing after it. Reporting a
    // loss here would send a reader hunting for draws that never existed.
    assert!(render(7, 8).contains("lost=0"), "{}", render(7, 8));
}

#[test]
fn short_payload_noop() {
    let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let (submission, r) = read_exec_submission(&state, &host, &[0u8; 4]);
    assert!(
        submission.is_none(),
        "four bytes are not a submission header"
    );
    assert_eq!(r.streams_loaded, 0);
}

/// An exec packet naming a slot that is not live must be refused under the
/// word the guest sent, not silently re-aimed at slot `word >> 1`.
///
/// Slot 3 is live and slot 6 is not, so word `6` names a dead slot whose
/// halved form is live — the exact ambiguity the two boots that justified
/// this deletion measured on every single exec decode. The old fallback
/// answered `3` here, and `3` is a different task: everything the packet
/// goes on to do, including its guest writes, would run against page tables
/// the guest never named for this work.
///
/// `task_id` is the separator because it is what the crate acts as and what
/// `exec_summary` reports. Asserting only "no streams loaded" would pass
/// either way — with no page tables mapped nothing loads regardless, which
/// is a probe that cannot distinguish the cases.
#[test]
fn an_exec_packet_naming_a_dead_slot_is_refused_not_aimed_at_its_neighbour() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let host = FakeHost::new();
    state.define_task(3, 0x1_0000, 2);
    assert!(state.tasks[3].active);
    assert!(
        !state.tasks.is_active(6),
        "slot 6 must be dead for this to bite"
    );

    let mut payload = vec![0u8; CHILD_EXEC_INDIRECT_HEADER_LEN as usize];
    st32(&mut payload[CHILD_EXEC_INDIRECT_TASK_ID as usize..], 6);
    st32(&mut payload[CHILD_EXEC_INDIRECT_CMDBUF_COUNT as usize..], 1);

    let (_submission, r) = read_exec_submission(&state, &host, &payload);
    assert_eq!(
        r.task_id, 6,
        "the refusal must name the word the guest sent, not the slot we \
         would have substituted"
    );
    assert_eq!(r.streams_loaded, 0);
    assert!(!r.saw_draw);
}

/// Bytes `+0x08..0x18` of a resource-table record are zero on every build
/// this project has measured, and their meaning is unrecovered. A guest that
/// starts setting them is telling this device something it cannot act on, so
/// the record must raise a line rather than pass unread.
///
/// The record with the populated tail is second, and the first is clean: a
/// check that fired on the *table* rather than the record would pass this
/// too, so the assertion names the object id.
#[test]
fn a_resource_record_that_populates_its_unrecovered_tail_says_so() {
    use crate::protocol::fifo::{
        CHILD_EXEC_RESOURCE_OBJECT_ID, CHILD_EXEC_RESOURCE_TAIL, CHILD_EXEC_RESOURCE_VALIDITY_OPS,
    };
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let host = FakeHost::new();
    state.define_task(3, 0x1_0000, 2);

    const N_RES: u32 = 2;
    let table_len = N_RES as usize * CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as usize;
    let mut payload = vec![
        0u8;
        CHILD_EXEC_INDIRECT_HEADER_LEN as usize
            + table_len
            + CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN as usize
    ];
    st32(&mut payload[CHILD_EXEC_INDIRECT_TASK_ID as usize..], 3);
    st32(
        &mut payload[CHILD_EXEC_INDIRECT_RESOURCE_COUNT as usize..],
        N_RES,
    );
    st32(&mut payload[CHILD_EXEC_INDIRECT_CMDBUF_COUNT as usize..], 1);
    for (i, id) in [0x40u32, 0x41].into_iter().enumerate() {
        let off = CHILD_EXEC_INDIRECT_HEADER_LEN as usize
            + i * CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as usize;
        st32(
            &mut payload[off + CHILD_EXEC_RESOURCE_OBJECT_ID as usize..],
            id,
        );
        st32(
            &mut payload[off + CHILD_EXEC_RESOURCE_VALIDITY_OPS as usize..],
            0x0000_0001,
        );
    }
    // One byte, in the last record, at the far end of the tail: the widest
    // gap between "the decoder read the tail" and "the decoder read a dword
    // it already had".
    let last =
        CHILD_EXEC_INDIRECT_HEADER_LEN as usize + CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as usize;
    payload[last + CHILD_EXEC_RESOURCE_TAIL as usize + 15] = 0xa5;

    let cb = CHILD_EXEC_INDIRECT_HEADER_LEN as usize + table_len;
    st64(
        &mut payload[cb + CHILD_EXEC_INDIRECT_CMDBUF_GVA as usize..],
        0xdead_0000,
    );
    st64(
        &mut payload[cb + CHILD_EXEC_INDIRECT_CMDBUF_LENGTH as usize..],
        64,
    );

    let cap = crate::observe::sink::FailCapture::start();
    // The resource table is consumed by the *execution* half, so this goes
    // through the one execution door: read at arrival, run at release. The
    // resolved work is empty because the command buffers are unreadable, which
    // is what this fixture is about — there are no records for the model to
    // have resolved.
    let mut host = host;
    let (submission, r) = read_exec_submission(&state, &host, &payload);
    let submission = submission.expect("the header parses");
    let resolved = reims_vgpu_core::exec::ExecWork::default();
    let r = execute_planned(
        &mut state,
        &mut host,
        RetainedInputs {
            submission: &submission,
            resolved: &resolved,
        },
        r,
    );
    assert_eq!(r.task_id, 3);
    assert_eq!(r.streams_loaded, 0, "no page table backs the cmdbuf gva");
    let line = cap.one("exec_res_table");
    assert!(line.contains("reason=exec_res_tail_populated"), "{line}");
    assert!(line.contains(" object=65 "), "{line}");
    assert!(line.contains(" tail_nz=1"), "{line}");
}

/// One segment header whose declared length runs `overshoot` bytes past the
/// buffer, followed by `tail` bytes of would-be records.
fn truncated_segment(type_: u8, overshoot: usize, tail: usize) -> Vec<u8> {
    use reims_vgpu_protocol::segment::SEGMENT_HEADER_LEN;
    let mut stream = vec![0u8; SEGMENT_HEADER_LEN + tail];
    st32(
        &mut stream[0..4],
        (SEGMENT_HEADER_LEN + tail + overshoot) as u32,
    );
    stream[4] = type_;
    stream
}

fn sink_body() -> String {
    std::fs::read_to_string(crate::observe::fail_log_path()).unwrap_or_default()
}

#[test]
fn a_stream_that_will_not_frame_says_so_instead_of_executing_nothing() {
    use reims_vgpu_protocol::segment::SegmentKind;
    // The defect this pins: `walk_stream` opened with `Err(_) => return`, so a
    // stream the framing decoder rejected executed zero records and produced
    // zero log lines — byte-for-byte indistinguishable at the sink from an
    // idle guest that submitted nothing.
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    let before = sink_body().len();
    // Task id doubles as the flood-latch discriminant, so it must be one no
    // other test in this process has already burned.
    let task_id = 0x5731_0001;
    walk_stream(
        &mut state,
        &mut host,
        task_id,
        &truncated_segment(SegmentKind::Render.wire_type(), 64, 0),
        &mut out,
        &mut acc,
        None,
    );
    let added = sink_body()[before..].to_string();
    assert!(
        added.contains("stream_frame_fail"),
        "a stream that will not frame must reach the always-on sink, got:\n{added}"
    );
    assert!(
        added.contains("reason=framing_segment_length_past_stream_end"),
        "the line must name which framing check refused, not just that one \
         did — 17 checks shared `ErrBadLength` in the framer this replaced. \
         got:\n{added}"
    );
    assert!(
        added.contains(&format!("task={task_id}")),
        "the line must carry the task whose work was dropped, got:\n{added}"
    );
}

/// A well-formed segment carrying a record that overruns it names the check
/// rather than looking like end-of-records.
///
/// `Err(_) => break` in the walker this replaced treated a self-inconsistent
/// segment exactly like `Done`: the remaining records went unexecuted with
/// nothing logged.
///
/// The *segment*-level half of that defect is no longer expressible. The framer
/// this device carried handed the record walker an already-parsed `Segment`
/// whose `command_offset`/`command_length` could name bytes the buffer did not
/// hold, and `validate_segment` re-checked them on every record — sixteen
/// re-validation refusals for a state a caller should not have been able to
/// construct. `FramedSegment` carries the command window as a slice, so a
/// segment cannot claim bytes that are not there and the whole re-validation
/// family is gone with the state it guarded.
///
/// What is left is the record's own length, which is still the guest's, and
/// that is what this drives.
#[test]
fn a_record_overrunning_its_segment_names_the_check_rather_than_ending_quietly() {
    use reims_vgpu_protocol::segment::{SegmentKind, SEGMENT_HEADER_LEN};
    // Two records in an info segment: one well formed, then one whose declared
    // length runs past the window it sits in.
    let mut commands = Vec::new();
    for (opcode, length) in [(0x1d1u32, 8u32), (0x1d1, 64)] {
        let mut hdr = [0u8; 8];
        st32(&mut hdr[0..4], opcode);
        st32(&mut hdr[4..8], length);
        commands.extend_from_slice(&hdr);
    }
    let before = sink_body().len();
    let mut handled = 0usize;
    walk_segment_records(
        SegmentKind::Info,
        SEGMENT_HEADER_LEN as u32,
        &commands,
        |_, _| handled += 1,
    );
    let added = sink_body()[before..].to_string();
    assert_eq!(handled, 1, "the well-formed record ahead of it still ran");
    assert!(
        added.contains("stream_record_fail"),
        "dropping a segment's records must reach the sink, got:\n{added}"
    );
    assert!(
        added.contains("reason=stream_rec_bad_length") && added.contains("length=64"),
        "the line must name the check and the length the guest declared, \
         got:\n{added}"
    );
    assert!(
        added.contains("seg=info"),
        "the line must say which segment family lost its records, got:\n{added}"
    );
}

#[test]
fn walking_a_well_formed_segment_to_its_end_logs_nothing() {
    use reims_vgpu_protocol::segment::{SegmentKind, SEGMENT_HEADER_LEN};
    // The other half of the obligation: every segment ends, so if the end
    // produced a line the sink would carry one per segment per frame.
    let mut records = [0u8; 8];
    st32(&mut records[0..4], 0x190);
    st32(&mut records[4..8], 8);
    let before = sink_body().len();
    let mut handled = 0usize;
    walk_segment_records(
        SegmentKind::Event,
        SEGMENT_HEADER_LEN as u32,
        &records,
        |_, _| handled += 1,
    );
    let added = sink_body()[before..].to_string();
    assert_eq!(handled, 1, "the one record is handed over");
    assert!(
        !added.contains("stream_record_fail"),
        "end-of-segment is control flow and must stay out of the log, got:\n{added}"
    );
}

/// An unknown segment family stops the walk and says so; the type-5 envelope is
/// skipped in silence.
///
/// The walker this replaced ended in `_ => {}`, which gave one silence to two
/// very different things: a protection envelope is a contract-correct skip, and
/// an unrecognised type is wire format this host has never seen.
///
/// **The unknown case now ends the stream rather than being stepped over, and
/// that is a behaviour change.** `FramingRefusal::UnknownType` states the
/// reason: a family whose record framing is unknown can only be skipped on its
/// declared length, and skipping on that hands the next segment an encoder
/// state derived from bytes nothing here understands. The reference host
/// rejects a non-continuation type it has no decoder for rather than stepping
/// over it. So the segments before the unknown one execute, the ones after it
/// do not, and the line says how many ran.
#[test]
fn an_unknown_segment_family_ends_the_walk_and_the_envelope_does_not() {
    use crate::model::FENCE_DOMAIN_EVENT;
    use reims_vgpu_protocol::segment::{
        segment_role, SegmentKind, SegmentRole, SEGMENT_HEADER_LEN, SEGMENT_TYPE_PROTECTION_OPTIONS,
    };

    assert_eq!(
        segment_role(SEGMENT_TYPE_PROTECTION_OPTIONS),
        Some(SegmentRole::ProtectionEnvelope)
    );
    assert_eq!(segment_role(6), None, "6 is not a family this host knows");
    assert_eq!(segment_role(0xff), None);

    // A protection envelope, then an event segment that signals, then an
    // unknown family, then a second event segment that must not run.
    let push = |stream: &mut Vec<u8>, type_: u8, body: &[u8]| {
        let mut hdr = [0u8; 8];
        st32(&mut hdr[0..4], (SEGMENT_HEADER_LEN + body.len()) as u32);
        hdr[4] = type_;
        stream.extend_from_slice(&hdr);
        stream.extend_from_slice(body);
    };
    let signal = |event_ref: u32, value: u64| {
        let mut rec = vec![0u8; 0x14];
        st32(&mut rec[0..4], 0x191);
        st32(&mut rec[4..8], 0x14);
        st32(&mut rec[8..12], event_ref);
        st64(&mut rec[12..20], value);
        rec
    };
    let mut stream = Vec::new();
    push(&mut stream, SEGMENT_TYPE_PROTECTION_OPTIONS, &[0u8; 8]);
    push(&mut stream, SegmentKind::Event.wire_type(), &signal(21, 4));
    push(&mut stream, 6, &[0u8; 8]);
    push(&mut stream, SegmentKind::Event.wire_type(), &signal(22, 9));

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    let task_id = 0x5731_0002;
    let cap = crate::observe::FailCapture::start();
    walk_stream(
        &mut state, &mut host, task_id, &stream, &mut out, &mut acc, None,
    );
    let lines = cap.lines();

    assert_eq!(
        state.fence_generation(task_id, FENCE_DOMAIN_EVENT, 21),
        Some(4),
        "the segments ahead of the unknown family still execute — and the \
         envelope ahead of them is skipped, not walked as records"
    );
    assert_eq!(
        state.fence_generation(task_id, FENCE_DOMAIN_EVENT, 22),
        None,
        "the walk stops at the unknown family"
    );
    assert!(
        lines.iter().any(|l| l.contains("stream_frame_fail")
            && l.contains("reason=framing_segment_type_unknown")
            && l.contains("wire_type=0x6")
            && l.contains("segments_before=2")),
        "the line must name the type and how many segments ran: {lines:?}"
    );
    assert!(
        !lines
            .iter()
            .any(|l| l.contains("framing_envelope_window_not_payload")),
        "an eight-byte envelope window is the payload and must not refuse: {lines:?}"
    );
}

#[test]
fn event_segment_signal_wait_in_stream() {
    use crate::model::FENCE_DOMAIN_EVENT;
    use reims_vgpu_protocol::sync::{OPCODE_SIGNAL_EVENT, OPCODE_WAIT_EVENT};
    // The record's own body, from the crate that owns its layout.
    let signal_wait_payload_len = core::mem::size_of::<reims_vgpu_wire::ops::event::SignalWait>();
    use reims_vgpu_protocol::segment::{SegmentKind, SEGMENT_HEADER_LEN};

    fn push_segment(buf: &mut Vec<u8>, type_: u8, payload: &[u8]) {
        let len = (SEGMENT_HEADER_LEN + payload.len()) as u32;
        let mut hdr = [0u8; 8];
        st32(&mut hdr[0..4], len);
        hdr[4] = type_;
        buf.extend_from_slice(&hdr);
        buf.extend_from_slice(payload);
    }
    let push_event_record = |buf: &mut Vec<u8>, opcode: u32, event_ref: u32, value: u64| {
        let mut payload = vec![0u8; signal_wait_payload_len];
        st32(&mut payload[0..4], event_ref);
        st64(&mut payload[4..12], value);
        let len = (OP_HEADER_LEN + signal_wait_payload_len) as u32;
        let mut hdr = [0u8; 8];
        st32(&mut hdr[0..4], opcode);
        st32(&mut hdr[4..8], len);
        buf.extend_from_slice(&hdr);
        buf.extend_from_slice(&payload);
    };

    let mut records = Vec::new();
    push_event_record(&mut records, OPCODE_SIGNAL_EVENT, 11, 7);
    push_event_record(&mut records, OPCODE_WAIT_EVENT, 11, 7);
    push_event_record(&mut records, OPCODE_WAIT_EVENT, 11, 8); // pending
    let mut stream = Vec::new();
    push_segment(&mut stream, SegmentKind::Event.wire_type(), &records);

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    walk_stream(&mut state, &mut host, 1, &stream, &mut out, &mut acc, None);

    // The signal landed, and the pending wait for 8 left it alone. The
    // three per-op counters this used to assert had no product reader; the
    // generation store is what the next wait actually reads.
    assert_eq!(state.fence_generation(1, FENCE_DOMAIN_EVENT, 11), Some(7));
}

/// The bounded event wait is refused, says so, and moves no generation.
///
/// `waitForEvent:value:timeoutMS:` is a settled row: this device runs no clock
/// against the guest's, so executing it as the unbounded wait it resembles
/// turns a guest's timeout into a hang. The refusal used to live in the
/// planner, one arm past a decoder that had lifted the record — so a reader
/// found it as a gap in the planner rather than as a decision about the wire,
/// and the settled row was refused in two places.
///
/// It is refused at the lift now, and this is the check that the refusal still
/// *reaches* the guest-visible failure channel from the segment walk. The row's
/// own name is what the line carries, so a reader lands on the ledger.
#[test]
fn a_bounded_event_wait_is_refused_by_contract_and_leaves_the_generation_alone() {
    use crate::model::FENCE_DOMAIN_EVENT;
    use reims_vgpu_protocol::segment::{SegmentKind, SEGMENT_HEADER_LEN};
    use reims_vgpu_protocol::sync::{OPCODE_SIGNAL_EVENT, OPCODE_WAIT_EVENT_TIMEOUT};

    // The bounded wait's own body: the signal/wait pair plus its timeout word.
    let body = core::mem::size_of::<reims_vgpu_wire::ops::event::SignalWait>() + 4;
    let mut records = Vec::new();
    for (opcode, value) in [(OPCODE_SIGNAL_EVENT, 5u64), (OPCODE_WAIT_EVENT_TIMEOUT, 9)] {
        let mut payload = vec![0u8; body];
        st32(&mut payload[0..4], 3);
        st64(&mut payload[4..12], value);
        let mut hdr = [0u8; 8];
        st32(&mut hdr[0..4], opcode);
        st32(&mut hdr[4..8], (OP_HEADER_LEN + body) as u32);
        records.extend_from_slice(&hdr);
        records.extend_from_slice(&payload);
    }
    let mut stream = Vec::new();
    let len = (SEGMENT_HEADER_LEN + records.len()) as u32;
    let mut hdr = [0u8; 8];
    st32(&mut hdr[0..4], len);
    hdr[4] = SegmentKind::Event.wire_type();
    stream.extend_from_slice(&hdr);
    stream.extend_from_slice(&records);

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    let cap = crate::observe::FailCapture::start();
    walk_stream(&mut state, &mut host, 1, &stream, &mut out, &mut acc, None);
    let lines = cap.lines();

    assert!(
        lines.iter().any(|l| l.starts_with("event_record ")
            && l.contains("reason=decode_opcode_refused_by_contract")
            && l.contains("opcode=0x192")),
        "a bounded wait must be refused by name, not dropped: {lines:?}"
    );
    // The signal ahead of it still landed — the refusal is the one record's and
    // not the segment's — and the wait moved nothing.
    assert_eq!(state.fence_generation(1, FENCE_DOMAIN_EVENT, 3), Some(5));
}

#[test]
fn multi_attachment_decode_in_pass() {
    // The record's own length: a descriptor short of `RenderPassBody` is
    // refused now rather than read at the offsets that fit.
    let mut payload = vec![0u8; wire_pass::RENDER_PASS_TOTAL_LEN as usize - OP_HEADER_LEN];
    for (i, tex) in [(0u32, 41u32), (1u32, 42u32)] {
        let slot = PASS_COLOR_ATTACH_OFF + i as usize * PASS_COLOR_ATTACH_STRIDE;
        st32(&mut payload[slot + PASS_ATTACH_TEXREF..], tex);
        st16(
            &mut payload[slot + PASS_ATTACH_LOAD_ACTION..],
            MTL_LOAD_ACTION_CLEAR,
        );
        st16(
            &mut payload[slot + PASS_ATTACH_STORE_ACTION..],
            MTL_STORE_ACTION_STORE,
        );
        st64(
            &mut payload[slot + PASS_ATTACH_CLEAR_COLOR..],
            1.0f64.to_bits(),
        );
        st64(
            &mut payload[slot + PASS_ATTACH_CLEAR_COLOR + 8..],
            0.0f64.to_bits(),
        );
        st64(
            &mut payload[slot + PASS_ATTACH_CLEAR_COLOR + 16..],
            0.0f64.to_bits(),
        );
        st64(
            &mut payload[slot + PASS_ATTACH_CLEAR_COLOR + 24..],
            1.0f64.to_bits(),
        );
    }
    let a0 = render_pass::decode_color_attachment(&payload, 0);
    let a1 = render_pass::decode_color_attachment(&payload, 1);
    assert_eq!(a0.texture_ref, 41);
    assert_eq!(a1.texture_ref, 42);
    let mut cmd = vec![0u8; OP_HEADER_LEN + payload.len()];
    st32(&mut cmd[0..], wire_pass::OPCODE_RENDER_PASS);
    st32(&mut cmd[4..], (OP_HEADER_LEN + payload.len()) as u32);
    cmd[OP_HEADER_LEN..].copy_from_slice(&payload);
    // The record's own lift, which is what production reads. Slot 0's ref
    // arrives through `RenderPassBody::color[0]` rather than through a
    // separately decoded `color0` field — the two used to be lifted apart and
    // could disagree.
    let framed = reims_vgpu_protocol::decode::op(&cmd, 0).expect("the record frames");
    let reims_vgpu_protocol::decode::render::RenderRecord::WriteDescriptor(d) =
        reims_vgpu_protocol::decode::render::decode(&framed).expect("the descriptor lifts")
    else {
        panic!("0x1a lifted as something other than a pass descriptor");
    };
    assert_eq!(d.descriptor.color[0].prefix.texture_ref.get(), 41);
    assert_eq!(d.descriptor.color[1].prefix.texture_ref.get(), 42);
}

/// An indexed draw whose record named no index buffer says so.
///
/// `count` takes `index_count` and `indexed` stays `None`, so the record
/// executes as a non-indexed draw of `index_count` vertices — a draw call
/// the guest never made, built from one it did. Metal has no such form:
/// `drawIndexedPrimitives` takes its index buffer as an argument, so a zero
/// ref has nothing to mean.
///
/// Asserts the line and, separately, that a well-formed indexed draw does
/// not produce it — the counter is only useful if it is quiet on the path
/// that works.
#[test]
fn an_indexed_draw_with_no_index_buffer_is_named() {
    // ARM compact indexed payload: prim@0, indexBufferRef@4, count@8:u16,
    // offset@0xa:u16 — total record 0x14.
    let record = |index_buffer_ref: u32| {
        let mut cmd = vec![0u8; 0x14];
        st32(&mut cmd[0..], wire_render::OPCODE_DRAW_INDEXED);
        st32(&mut cmd[4..], 0x14);
        st32(&mut cmd[OP_HEADER_LEN..], 3); // primitiveType
        st32(&mut cmd[OP_HEADER_LEN + 4..], index_buffer_ref);
        cmd[OP_HEADER_LEN + 8..OP_HEADER_LEN + 10].copy_from_slice(&6u16.to_le_bytes());
        cmd
    };
    let run = |cmd: &[u8]| {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum {
            pipeline_ref: 5,
            ..Default::default()
        };
        handle_render_record(
            &mut state,
            &host,
            1,
            wire_render::OPCODE_DRAW_INDEXED,
            cmd,
            &mut out,
            &mut acc,
        );
        acc
    };

    let good = run(&record(42));
    assert!(
        good.indexed.is_some(),
        "a record naming an index buffer is an indexed draw"
    );

    let bad = run(&record(0));
    assert!(
        bad.indexed.is_none(),
        "behaviour is unchanged: still no index buffer to draw with"
    );

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(
        log.contains("reason=indexed_without_index_buffer"),
        "an indexed draw reinterpreted as non-indexed must say so"
    );
    assert!(
        log.contains(&format!("op={:#x}", wire_render::OPCODE_DRAW_INDEXED)),
        "the line must name which indexed form fired, since each reads the \
         ref at a different offset"
    );
}

/// A depth attachment this device cannot honour is dropped, and says so.
///
/// A non-zero `level` binds a mip of the depth texture and a non-zero
/// `resolve_texture_ref` is a multisample depth resolve; both are real Metal
/// and neither is implemented here. The gate that drops them was a bare `if`
/// with no else, so the pass ran on with no depth attachment at all — depth
/// testing gone for every draw in it, which reads as wrong occlusion rather
/// than as a missing frame, and left nothing in the log to connect the two.
///
/// Both halves are asserted: the attachment is still refused (unchanged
/// behaviour) and the refusal is now named.
#[test]
fn an_unsupported_depth_attachment_is_named_not_just_dropped() {
    use crate::runtime::render_pass::{
        PASS_ATTACH_DEPTH_PLANE, PASS_ATTACH_LEVEL, PASS_ATTACH_RESOLVEREF, PASS_ATTACH_SLICE,
        PASS_ATTACH_TEXREF, PASS_DEPTH_ATTACH_OFF, PASS_STENCIL_ATTACH_OFF,
    };
    let pass = |level: u16, resolve: u32| {
        // The record's own length. A pass descriptor shorter than
        // `RenderPassBody` is refused now rather than read at whichever offsets
        // fit, so a fixture that was 0x200 bytes was testing a record Apple's
        // serializer does not emit.
        let mut payload = vec![0u8; wire_pass::RENDER_PASS_TOTAL_LEN as usize - OP_HEADER_LEN];
        st32(
            &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_ATTACH_TEXREF..],
            77,
        );
        payload[PASS_DEPTH_ATTACH_OFF + PASS_ATTACH_LEVEL
            ..PASS_DEPTH_ATTACH_OFF + PASS_ATTACH_LEVEL + 2]
            .copy_from_slice(&level.to_le_bytes());
        st32(
            &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_ATTACH_RESOLVEREF..],
            resolve,
        );
        // A stencil slot this device *can* honour, so the two aspects stay
        // separable and the depth arm is the only one under test.
        st32(
            &mut payload[PASS_STENCIL_ATTACH_OFF + PASS_ATTACH_TEXREF..],
            88,
        );
        let mut cmd = vec![0u8; OP_HEADER_LEN + payload.len()];
        st32(&mut cmd[0..], wire_pass::OPCODE_RENDER_PASS);
        st32(&mut cmd[4..], (OP_HEADER_LEN + payload.len()) as u32);
        cmd[OP_HEADER_LEN..].copy_from_slice(&payload);
        cmd
    };
    let run = |cmd: &[u8]| {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        handle_render_record(
            &mut state,
            &host,
            1,
            wire_pass::OPCODE_RENDER_PASS,
            cmd,
            &mut out,
            &mut acc,
        );
        acc
    };

    let ok = run(&pass(0, 0));
    assert!(
        ok.depth_attach.is_some() && ok.stencil_attach.is_some(),
        "a level-0 depth attachment with no resolve is honoured"
    );
    assert!(
        ok.bind_snapshot().is_ok(),
        "a pass this device can bind whole refuses nothing"
    );

    for (level, resolve) in [(1u16, 0u32), (0, 99)] {
        let acc = run(&pass(level, resolve));
        assert!(
            acc.depth_attach.is_none(),
            "level={level} resolve={resolve} must still be refused"
        );
        assert!(
            acc.stencil_attach.is_some(),
            "refusing depth must not take the stencil attachment with it"
        );
        // Leaving the attachment out is not enough on its own. A pass that
        // then *runs* has depth testing off for every draw in it, so the near
        // geometry stops occluding the far and the colour target — which was
        // correct before the pass — is overwritten with a picture assembled in
        // the wrong order. A pass with no depth attachment is also exactly what
        // a guest that wanted none produces, so nothing downstream can tell.
        assert!(
            matches!(
                acc.bind_snapshot(),
                Err(StreamRefusal::Pass(
                    StreamDrawDrop::DepthStencilUnsupported { .. }
                ))
            ),
            "level={level} resolve={resolve}: dropping the attachment must \
             also refuse the draws that would run without it"
        );
    }

    // `slice` and `depth_plane` are the two sixteen-bit fields above
    // `level` in the shared attachment prefix, and this arm read neither
    // until they were decodable. A depth buffer bound at slice 5 was read
    // as slice 0 and silently accepted, which is a depth test against the
    // wrong layer rather than a missing one.
    //
    // Driven from both slots, because "the two arms consume one wire form"
    // is exactly the shape that drifts: the stencil arm is a second call to
    // the same rule and nothing but this proves it is still the same one.
    for (field, at) in [
        ("slice", PASS_ATTACH_SLICE),
        ("plane", PASS_ATTACH_DEPTH_PLANE),
    ] {
        let mut cmd = pass(0, 0);
        let slot = OP_HEADER_LEN + PASS_DEPTH_ATTACH_OFF + at;
        cmd[slot..slot + 2].copy_from_slice(&5u16.to_le_bytes());
        let acc = run(&cmd);
        assert!(
            acc.depth_attach.is_none(),
            "a depth attachment naming {field} 5 must be refused, not read as 0"
        );
        assert!(
            acc.stencil_attach.is_some(),
            "refusing depth for {field} must not take the stencil attachment with it"
        );

        let mut cmd = pass(0, 0);
        let slot = OP_HEADER_LEN + PASS_STENCIL_ATTACH_OFF + at;
        cmd[slot..slot + 2].copy_from_slice(&5u16.to_le_bytes());
        let acc = run(&cmd);
        assert!(
            acc.stencil_attach.is_none(),
            "a stencil attachment naming {field} 5 must be refused, not read as 0"
        );
        assert!(
            acc.depth_attach.is_some(),
            "refusing stencil for {field} must not take the depth attachment with it"
        );
    }

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(
        log.contains("stream_depth_stencil_unsupported"),
        "an unsupported depth attachment was dropped without naming itself"
    );
    assert!(
        log.contains("aspect=depth") && log.contains("aspect=stencil"),
        "the line must say which aspect was lost, and both arms must reach it"
    );
    assert!(
        log.contains("slice=5"),
        "the line must carry the slice; it was undecodable before the shared \
         prefix was derived"
    );
}

/// A pass declaring more render-target array layers than this device draws
/// refuses the stream's draws.
///
/// Layered rendering picks the layer per draw, from the vertex stage's
/// `[[render_target_array_index]]`, and this device binds the attachment whole
/// and draws into layer 0. So geometry the guest aimed at layer 3 lands on top
/// of layer 0's content and layers 1..n keep whatever they held through a
/// `Clear` the guest asked to apply to all of them — the same shape of loss the
/// colour subresource arm below refuses, with the coordinate chosen per draw
/// instead of per pass.
///
/// This counted and rendered anyway until the arms beside it stopped doing so.
#[test]
fn a_pass_declaring_more_array_layers_than_this_device_draws_refuses_the_draws() {
    use crate::protocol::endian::st32;
    use crate::runtime::render_pass::{PASS_ATTACH_TEXREF, PASS_COLOR_ATTACH_OFF};

    // A full-length record, not the `PASS_MIN_PAYLOAD` one the arms below use:
    // the array length is read only from the whole `RenderPassBody`, and a
    // short record falls into the per-attachment views that do not reach it.
    // The offset comes from `offset_of!` for the reason the device's own
    // constants do, so a wire rename fails the build here too.
    const ARRAY_LENGTH_AT: usize = OP_HEADER_LEN
        + core::mem::offset_of!(wire_pass::RenderPassBody, render_target_array_length);

    let pass = |layers: u32| {
        let total = OP_HEADER_LEN + wire_pass::RENDER_PASS_TOTAL_LEN as usize;
        let mut cmd = vec![0u8; total];
        st32(&mut cmd[0..], wire_pass::OPCODE_RENDER_PASS);
        st32(&mut cmd[4..], total as u32);
        let slot = OP_HEADER_LEN + PASS_COLOR_ATTACH_OFF;
        st32(&mut cmd[slot + PASS_ATTACH_TEXREF..], 77);
        st32(&mut cmd[ARRAY_LENGTH_AT..], layers);
        cmd
    };
    let run = |cmd: &[u8]| {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        handle_render_record(
            &mut state,
            &host,
            1,
            wire_pass::OPCODE_RENDER_PASS,
            cmd,
            &mut out,
            &mut acc,
        );
        acc
    };

    // One layer is the API default and is what this device draws, so nothing is
    // refused. Zero is the same statement written the other way — a pass that
    // did not set the property at all.
    for layers in [0u32, 1] {
        assert!(
            run(&pass(layers)).bind_snapshot().is_ok(),
            "layers={layers} is what this device draws; nothing is refused"
        );
    }

    for layers in [2u32, 6] {
        assert!(
            matches!(
                run(&pass(layers)).bind_snapshot(),
                Err(StreamRefusal::Pass(
                    StreamDrawDrop::PassArrayLengthUnsupported { .. }
                ))
            ),
            "layers={layers}: a pass this device would draw only layer 0 of must \
             refuse its draws rather than land geometry meant for another layer \
             on top of layer 0's content"
        );
    }

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(
        log.contains("stream_pass_array_length_unsupported"),
        "a pass declaring layers this device does not draw said nothing"
    );
    assert!(
        log.contains("length=6"),
        "the line must carry the declared layer count: 2 layers and 6 are \
         different readings, and it is the whole of what this arm reports"
    );
}

/// An attachment's store-action options asking for programmable sample
/// positions refuse the stream's draws.
///
/// `MTLStoreActionOptions` declares exactly one flag,
/// `CustomSamplePositions`, and it changes where the samples a resolve reads
/// were taken. This device sets no programmable sample positions, so a resolve
/// it produces reads the default ones — and that is byte-for-byte what a guest
/// asking for the default also gets, which is why this used to be a count
/// nothing downstream could act on.
///
/// `MTLStoreActionOptionNone` is honoured. It is the API default, it asks for
/// nothing, and the resolve this device already performs is the one it names.
///
/// The three forms are driven separately because they do not share a record:
/// the colour options are a `u64` followed by an index in a 20-byte record, and
/// the depth and stencil forms are 16 bytes with no index at all. An arm
/// reading the wrong one would take the index for the options.
#[test]
fn store_action_options_asking_for_custom_sample_positions_refuse_the_draws() {
    use crate::protocol::endian::{st32, st64};

    let record = |opcode: u32, options: u64| {
        let total = if opcode == wire_render::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS {
            wire_render::SET_COLOR_STORE_ACTION_OPTIONS_TOTAL_LEN
        } else {
            wire_render::SET_STORE_ACTION_OPTIONS_TOTAL_LEN
        } as usize;
        let mut cmd = vec![0u8; total];
        st32(&mut cmd[0..], opcode);
        st32(&mut cmd[4..], total as u32);
        st64(&mut cmd[OP_HEADER_LEN..], options);
        if opcode == wire_render::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS {
            st32(&mut cmd[OP_HEADER_LEN + 8..], 3);
        }
        cmd
    };
    let run = |opcode: u32, options: u64| {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        handle_render_record(
            &mut state,
            &host,
            1,
            opcode,
            &record(opcode, options),
            &mut out,
            &mut acc,
        );
        acc
    };

    for opcode in [
        wire_render::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS,
        wire_render::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS,
        wire_render::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS,
    ] {
        assert!(
            run(opcode, 0).bind_snapshot().is_ok(),
            "op {opcode:#x}: the none option asks for nothing and is what this \
             device already does; nothing is refused"
        );
        // Which attachment asked, and — for the colour form alone — which slot.
        // Read off the refusal rather than off the shared fail log: three other
        // arms report an `aspect` too, so a log line saying `aspect=stencil`
        // does not say that *this* record named the stencil attachment.
        let (expected_aspect, expected_slot) = match opcode {
            wire_render::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS => ("depth", 0),
            wire_render::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS => ("stencil", 0),
            _ => ("color", 3),
        };
        // The declared flag, and three values the SDK does not declare — the
        // ones the capture's own fixtures carry. A parse that masked to the low
        // bit would read `0x1111` as the flag and `0x2222` as none, from two
        // values that are equally undeclared.
        for requested in [1u64, 0x1111, 0x2222, 0x3333] {
            let Err(StreamRefusal::Pass(StreamDrawDrop::StoreActionOptionsUnsupported {
                aspect,
                slot,
                options,
            })) = run(opcode, requested).bind_snapshot()
            else {
                panic!(
                    "op {opcode:#x} options={requested:#x}: a resolve taken at \
                     the default sample positions is what a guest asking for \
                     the default also gets, so this must refuse rather than \
                     count"
                );
            };
            assert_eq!(
                (aspect, slot, options),
                (expected_aspect, expected_slot, requested),
                "op {opcode:#x}: the refusal named a different record than the \
                 one that produced it"
            );
        }
    }

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(
        log.contains("stream_store_action_options_unsupported"),
        "an attachment asking for programmable sample positions said nothing"
    );
}

/// A pass declaring a default raster sample count this device cannot rasterize
/// at refuses the stream's draws.
///
/// `defaultRasterSampleCount` says how many fragments the rasterizer produces
/// per pixel. Every render rail here produces one, so a pass asking for four
/// used to render at one and raise a counter — and a pass rendered at one
/// sample is exactly what a guest asking for one sample also produces, so
/// nothing downstream could tell. Coverage decides which fragments run, so what
/// the guest got back was a different picture and, for an occlusion query, a
/// different number.
///
/// The device advertises `DEVICE_INFO_KEY_MAX_SAMPLE_COUNT` above 1, so a guest
/// is entitled to ask; this is the refusal that says what that costs.
#[test]
fn a_pass_declaring_a_raster_sample_count_this_device_cannot_rasterize_refuses_the_draws() {
    use crate::protocol::endian::st32;

    let record = |count: u32| {
        let total = wire_pass::DEFAULT_RASTER_SAMPLE_COUNT_TOTAL_LEN as usize;
        let mut cmd = vec![0u8; total];
        st32(&mut cmd[0..], wire_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT);
        st32(&mut cmd[4..], total as u32);
        st32(&mut cmd[OP_HEADER_LEN..], count);
        cmd
    };
    let run = |cmd: &[u8]| {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        handle_render_record(
            &mut state,
            &host,
            1,
            wire_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT,
            cmd,
            &mut out,
            &mut acc,
        );
        acc
    };

    // One sample is the API default and is what this device rasterizes at, so
    // the record is honoured and nothing is refused.
    assert!(
        run(&record(1)).bind_snapshot().is_ok(),
        "one sample per pixel is what this device rasterizes at; nothing is \
         refused"
    );

    // Zero is not a Metal sample count. It refuses with the rest rather than
    // being read as "the guest asked for nothing", because a record this device
    // cannot honour is not made honourable by naming an impossible value.
    for count in [0u32, 2, 4, 8] {
        assert!(
            matches!(
                run(&record(count)).bind_snapshot(),
                Err(StreamRefusal::Pass(
                    StreamDrawDrop::PassRasterSampleCountUnsupported { .. }
                ))
            ),
            "count={count}: a pass this device would rasterize at one sample \
             must refuse its draws rather than hand the guest a picture \
             assembled from different coverage"
        );
    }

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(
        log.contains("stream_pass_raster_sample_count_unsupported"),
        "a pass declaring a sample count this device does not rasterize at \
         said nothing"
    );
    assert!(
        log.contains("count=8"),
        "the line must carry the requested count: 2 samples and 8 are different \
         readings, and it is the whole of what this arm reports"
    );
}

/// A colour attachment naming a slice or a depth plane refuses the stream's
/// draws, while a resolve target becomes the direct single-sample target.
///
/// Every consumer binds the texture whole, so a pass this device ran would go
/// into level 0 slice 0 plane 0 regardless — a guest drawing a cube face
/// overwrites face 0, and a guest drawing a mip overwrites the image every
/// other level is sampled from. Nothing downstream can tell that happened,
/// because a pass into the base level is exactly what a guest that asked for
/// the base level also produces.
///
/// This used to assert the opposite, on the argument that "the pass still runs
/// -- reporting must not cost the guest its draw". That argument does not
/// survive asking *whose* pixels: the guest does not lose a draw and get a
/// blurry one, it loses a **different subresource** that was correct before the
/// pass and that the pass never named as its target.
///
/// The `slice` and `depth_plane` arms are the ones that could not have been
/// written before: those fields did not exist, because the decoder read
/// `level` thirty-two bits wide and swallowed the slice into it.
///
#[test]
fn a_colour_attachment_naming_a_subresource_this_device_cannot_bind_refuses_the_draws() {
    use crate::protocol::endian::st32;
    use crate::runtime::render_pass::{
        PASS_ATTACH_DEPTH_PLANE, PASS_ATTACH_LEVEL, PASS_ATTACH_RESOLVEREF, PASS_ATTACH_SLICE,
        PASS_ATTACH_TEXREF, PASS_COLOR_ATTACH_OFF,
    };

    let pass_resolving = |level: u16, slice: u16, plane: u16, resolve: u32| {
        // Full length: see `an_unsupported_depth_attachment_is_named_not_just_dropped`.
        let total = wire_pass::RENDER_PASS_TOTAL_LEN as usize;
        let mut cmd = vec![0u8; total];
        st32(&mut cmd[0..], wire_pass::OPCODE_RENDER_PASS);
        st32(&mut cmd[4..], total as u32);
        let slot = OP_HEADER_LEN + PASS_COLOR_ATTACH_OFF;
        st32(&mut cmd[slot + PASS_ATTACH_TEXREF..], 77);
        st32(&mut cmd[slot + PASS_ATTACH_RESOLVEREF..], resolve);
        cmd[slot + PASS_ATTACH_LEVEL..slot + PASS_ATTACH_LEVEL + 2]
            .copy_from_slice(&level.to_le_bytes());
        cmd[slot + PASS_ATTACH_SLICE..slot + PASS_ATTACH_SLICE + 2]
            .copy_from_slice(&slice.to_le_bytes());
        cmd[slot + PASS_ATTACH_DEPTH_PLANE..slot + PASS_ATTACH_DEPTH_PLANE + 2]
            .copy_from_slice(&plane.to_le_bytes());
        cmd
    };
    let pass = |level: u16, slice: u16, plane: u16| pass_resolving(level, slice, plane, 0);
    let run = |cmd: &[u8]| {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        handle_render_record(
            &mut state,
            &host,
            1,
            wire_pass::OPCODE_RENDER_PASS,
            cmd,
            &mut out,
            &mut acc,
        );
        acc
    };

    // Subresource 0/0/0 is what this device binds, so it reports nothing and
    // the stream stays representable.
    let acc = run(&pass(0, 0, 0));
    assert_eq!(
        acc.color_slots.len(),
        1,
        "the plain attachment still reaches the slot list"
    );
    assert!(
        acc.bind_snapshot().is_ok(),
        "the base subresource is what this device binds; nothing is refused"
    );

    // A mip level is the one coordinate this device renders into rather than
    // past: `render_target`'s linear rung resolves the named level's own plane
    // out of the guest allocation. Refusing it dropped every pass of macOS 26's
    // blur pyramid.
    let acc = run(&pass(3, 0, 0));
    assert_eq!(acc.color_slots.len(), 1);
    assert!(
        acc.bind_snapshot().is_ok(),
        "a colour attachment naming a mip level must not refuse the stream: \
         the level resolves to its own plane"
    );

    for (level, slice, plane) in [(0u16, 5u16, 0u16), (0, 0, 2)] {
        let acc = run(&pass(level, slice, plane));
        // The attachment still reaches the slot list, because the refusal is
        // the stream's and not the attachment's: what is refused is encoding
        // draws against a target this device would bind at the wrong place.
        assert_eq!(acc.color_slots.len(), 1);
        assert!(
            matches!(
                acc.bind_snapshot(),
                Err(StreamRefusal::Pass(
                    StreamDrawDrop::ColorSubresourceUnsupported { .. }
                ))
            ),
            "level={level} slice={slice} plane={plane}: a pass this device \
             would render into the base of must refuse its draws rather than \
             overwrite a subresource the guest did not name"
        );
    }

    // The source and resolve destination stay distinct through stream decode.
    // Collapsing them here turns a resolve operation into single-sample drawing
    // and loses coverage before either backend sees the request.
    let acc = run(&pass_resolving(0, 0, 0, 0x99));
    assert_eq!(acc.color_slots.len(), 1);
    assert_eq!(acc.color_slots[0].1.texture_ref, 77);
    assert_eq!(acc.color_slots[0].1.resolve_texture_ref, 0x99);
    assert!(
        acc.bind_snapshot().is_ok(),
        "a base-subresource resolve is representable by the stream"
    );

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(
        log.contains("stream_color_subresource_unsupported"),
        "a colour attachment bound at the wrong subresource said nothing"
    );
    assert!(
        log.contains("slice=5"),
        "the line must carry the slice; before the decode fix it was folded \
         into the level and could not be reported"
    );
    assert!(
        log.contains("plane=2"),
        "the line must carry the depth plane"
    );
}

/// The pass-extent census bands agree with the scissor-union census's.
///
/// The two answer the same question from two different sources — the pass
/// descriptor and the draw stream — and the whole reason to have both is to
/// read them side by side. Bands that drifted apart would make that
/// comparison silently wrong rather than obviously so.
///
/// Declared twice because `coverage_band` is behind `backend-vulkan` and
/// this census runs on every backend. This is the comparison that keeps the
/// duplication honest.
#[test]
fn the_two_coverage_censuses_use_the_same_bands() {
    // Every boundary of every band, plus one over the top.
    for pct in [0u64, 1, 5, 6, 10, 11, 25, 26, 50, 51, 99, 100, 101] {
        let band = pass_extent_band(pct);
        assert!(
            band < PASS_EXTENT_SLUGS.len(),
            "pct {pct} banded out of range"
        );
        #[cfg(feature = "backend-vulkan")]
        assert_eq!(
            band,
            crate::runtime::draw::coverage_band_for_test(pct),
            "pct {pct}: the two censuses band it differently"
        );
    }
    // The bands are ordered, so a larger fraction never scores lower.
    let mut last = 0usize;
    for pct in 0..=100u64 {
        let b = pass_extent_band(pct);
        assert!(b >= last, "pct {pct} banded below its predecessor");
        last = b;
    }
    assert_eq!(pass_extent_band(100), PASS_EXTENT_SLUGS.len() - 1);
}

/// A stated extent is scored against the attachment, and only when both are
/// real.
#[test]
fn the_pass_extent_census_scores_a_fraction_and_clamps_it() {
    use crate::runtime::drain::store_route_count;

    // A pass covering a quarter of its attachment.
    let before = store_route_count("pass_extent_le25");
    note_pass_extent_coverage(960, 540, 1920, 1080);
    assert_eq!(
        store_route_count("pass_extent_le25"),
        before + 1,
        "960x540 of 1920x1080 is 25%"
    );

    // A pass stating more than the attachment holds. Metal permits it and
    // the rasteriser clips, so this reads full rather than over 100%.
    let before = store_route_count("pass_extent_full");
    note_pass_extent_coverage(4096, 4096, 1920, 1080);
    assert_eq!(store_route_count("pass_extent_full"), before + 1);

    // Neither a missing extent nor a geometry-less attachment is scored:
    // there is no fraction to take, and counting it as zero would put every
    // unstated pass in the bottom band and make the census read as damage.
    let before: u64 = PASS_EXTENT_SLUGS.iter().map(|s| store_route_count(s)).sum();
    note_pass_extent_coverage(0, 0, 1920, 1080);
    note_pass_extent_coverage(100, 100, 0, 0);
    assert_eq!(
        PASS_EXTENT_SLUGS
            .iter()
            .map(|s| store_route_count(s))
            .sum::<u64>(),
        before
    );
}

/// The extent census scores whichever resolve arm supplied the mapping id,
/// and only for slot 0.
///
/// This is the arm-parity test. The census used to hang off the mapper-ref-texture
/// resolve alone, so on the x86/Vulkan pathway — where the workload takes
/// the backing arm — every band read zero, which is indistinguishable from a
/// guest that never states an extent. A backing attachment *is* its own
/// mapping id, so the only difference between the two call sites is which
/// id they pass, and this pins that the scoring does not care which.
#[test]
fn the_pass_extent_census_scores_either_resolve_arm() {
    use crate::runtime::drain::store_route_count;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    // The pass's stated extent, which is the only thing the census reads off
    // the record — the two numbers rather than a whole decoded command.
    let extent = (960u64, 540u64);
    // One mapping, reached by the id either arm would hand over.
    assert!(state.map_surface(7));
    let _ = state.set_mapping_geom(7, 1920, 1080, 0);

    let before = store_route_count("pass_extent_le25");
    note_pass_extent_for_slot(&state, 1, 0, 7, extent);
    assert_eq!(
        store_route_count("pass_extent_le25"),
        before + 1,
        "slot 0 was not scored"
    );

    // A slot the device does not treat as the pass's attachment, and a
    // mapping id with no geometry yet, are both silent — the first because
    // the census is defined on slot 0, the second because there is no
    // fraction to take.
    let before: u64 = PASS_EXTENT_SLUGS.iter().map(|s| store_route_count(s)).sum();
    note_pass_extent_for_slot(&state, 1, 1, 7, extent);
    note_pass_extent_for_slot(&state, 1, 0, 4242, extent);
    assert_eq!(
        PASS_EXTENT_SLUGS
            .iter()
            .map(|s| store_route_count(s))
            .sum::<u64>(),
        before
    );
}

#[test]
fn stream_accum_upserts_buffer_and_viewport() {
    // wire opcodes via wire_render import

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();

    // setVertexBuffer multi-entry: first=2 count=1 ref=9 offset=16
    // payload = first:u32 + count:u32 + {ref:u32, offset:u64}
    let mut vb = vec![0u8; OP_HEADER_LEN + 8 + 12];
    let vb_len = vb.len() as u32;
    st32(&mut vb[0..], wire_render::OPCODE_SET_VERTEX_BUFFER);
    st32(&mut vb[4..], vb_len);
    st32(&mut vb[8..], 2); // first
    st32(&mut vb[12..], 1); // count
    st32(&mut vb[16..], 9); // ref
    st64(&mut vb[20..], 16); // offset
    handle_render_record(
        &mut state,
        &host,
        0,
        wire_render::OPCODE_SET_VERTEX_BUFFER,
        &vb,
        &mut out,
        &mut acc,
    );
    assert_eq!(acc.vertex_buffers.len(), 1);
    assert_eq!(acc.vertex_buffers[0].index, 2);
    assert_eq!(acc.vertex_buffers[0].buffer_ref, 9);
    assert_eq!(acc.vertex_buffers[0].offset, 16);

    // overwrite same slot
    st32(&mut vb[16..], 10);
    handle_render_record(
        &mut state,
        &host,
        0,
        wire_render::OPCODE_SET_VERTEX_BUFFER,
        &vb,
        &mut out,
        &mut acc,
    );
    assert_eq!(acc.vertex_buffers.len(), 1);
    assert_eq!(acc.vertex_buffers[0].buffer_ref, 10);

    // fragment buffer multi-entry: first=0 count=1 ref=7 offset=0
    let mut fb = vec![0u8; OP_HEADER_LEN + 8 + 12];
    let fb_len = fb.len() as u32;
    st32(&mut fb[0..], wire_render::OPCODE_SET_FRAGMENT_BUFFER);
    st32(&mut fb[4..], fb_len);
    st32(&mut fb[8..], 0); // first
    st32(&mut fb[12..], 1); // count
    st32(&mut fb[16..], 7); // ref
    st64(&mut fb[20..], 0); // offset
    handle_render_record(
        &mut state,
        &host,
        0,
        wire_render::OPCODE_SET_FRAGMENT_BUFFER,
        &fb,
        &mut out,
        &mut acc,
    );
    assert_eq!(acc.fragment_buffers.len(), 1);

    // viewport
    let mut vp = vec![0u8; OP_HEADER_LEN + 48];
    st32(&mut vp[0..], wire_render::OPCODE_SET_VIEWPORT);
    st32(&mut vp[4..], (OP_HEADER_LEN + 48) as u32);
    for i in 0..6 {
        let bits = (i as f64 + 1.0).to_bits();
        st64(&mut vp[OP_HEADER_LEN + i * 8..], bits);
    }
    handle_render_record(
        &mut state,
        &host,
        0,
        wire_render::OPCODE_SET_VIEWPORT,
        &vp,
        &mut out,
        &mut acc,
    );
    assert_eq!(acc.viewports.len(), 1);
    let v = acc.viewports[0];
    assert!((v[0] - 1.0).abs() < 1e-9);
    assert!((v[5] - 6.0).abs() < 1e-9);
}

#[test]
fn wide_indexed_draw_reaches_pending_draw() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum {
        pipeline_ref: 61,
        ..Default::default()
    };
    let mut command = vec![0u8; 0x20];
    let op = wire_render::OPCODE_DRAW_INDEXED_WIDE;
    st32(&mut command[0..], op);
    st32(&mut command[4..], 0x20);
    st16(&mut command[8..], 3);
    st16(&mut command[10..], 0);
    st32(&mut command[12..], 0x3e);
    st32(&mut command[16..], 6);
    st32(&mut command[24..], 0x10100);
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);

    assert!(acc.saw_draw);
    assert!(out.saw_draw);
    assert_eq!(acc.draws.len(), 1);
    let indexed = acc.draws[0].indexed.as_ref().expect("indexed draw");
    assert_eq!(indexed.index_type, 0);
    assert_eq!(indexed.index_buffer_ref, 0x3e);
    assert_eq!(indexed.index_count, 6);
    assert_eq!(indexed.index_buffer_offset, 0x10100);
    assert_eq!(
        acc.draws[0].draw,
        DrawArgs {
            vertex_count: 6,
            instance_count: 1,
            primitive_type: 3,
            first_vertex: 0,
            base_instance: 0
        }
    );
}

/// A base vertex and a base instance survive the whole accumulator hop.
///
/// Both had a home in every backend already — Metal's `render_core_mrt`
/// takes a base instance and `ReimsVgpuIndexedDraw` a base vertex, Vulkan's
/// `DrawRequest` and `IndexedDrawResource` the same two — and both were fed
/// a hardcoded zero from here, because nothing upstream decoded a draw form
/// that carries them. This is the seam that was missing, so it is the seam
/// worth pinning: a regression to a literal `0` anywhere between decode and
/// `DrawEncodeRequest` fails here.
#[test]
fn a_base_vertex_and_base_instance_reach_the_pending_draw() {
    use crate::protocol::endian::st16;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum {
        pipeline_ref: 61,
        ..Default::default()
    };
    let op = wire_render::OPCODE_DRAW_INDEXED_INSTANCED_BASE;
    let total = reims_vgpu_wire::ops::render::DRAW_INDEXED_INSTANCED_BASE_TOTAL_LEN;
    let mut command = vec![0u8; total as usize];
    st32(&mut command[0..], op);
    st32(&mut command[4..], total);
    st16(&mut command[8..], 3); // primitiveType
    st16(&mut command[10..], 1); // indexType UInt32
    st32(&mut command[12..], 0x3e); // index buffer ref
    st16(&mut command[16..], 0x40); // index buffer offset (first, on this form)
    st16(&mut command[18..], 6); // index count
    st16(&mut command[20..], 9); // instanceCount
    st16(&mut command[22..], 0xfffb); // baseVertex = -5
    st16(&mut command[24..], 7); // baseInstance
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);

    assert_eq!(acc.draws.len(), 1, "the draw must not be dropped");
    assert_eq!(acc.draws[0].draw.instance_count, 9);
    assert_eq!(acc.draws[0].draw.base_instance, 7);
    let indexed = acc.draws[0].indexed.as_ref().expect("indexed draw");
    assert_eq!(indexed.index_count, 6);
    assert_eq!(indexed.index_buffer_offset, 0x40);
    assert_eq!(indexed.base_vertex, -5, "a negative base vertex survives");

    // And onward into the request the backends receive. `retarget_render_
    // pass_draw` is the path records 2+ of a chained pass take, and it
    // rebuilds every draw argument from the template.
    let template = draw::DrawEncodeRequest::default();
    let req = retarget_render_pass_draw(&template, &acc.draws[0]);
    assert_eq!(req.base_instance, 7);
    assert_eq!(req.instance_count, 9);
}

/// Every decoded draw in a stream reaches the draw list.
///
/// `MAX_DRAWS_PER_STREAM = 64` truncated `acc.draws` inside a bare `if` with
/// no `else`, so a compositor stream with more records than that lost every
/// draw past the 64th with nothing on any channel — no counter, no line, and
/// an `ExecResult` describing the truncated list as a fully executed pass.
/// 71 is chosen to straddle that old ceiling: this test fails on the capped
/// code at exactly 64.
#[test]
fn every_decoded_draw_in_a_stream_reaches_the_draw_list() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum {
        pipeline_ref: 61,
        ..Default::default()
    };
    let mut command = vec![0u8; 0x20];
    let op = wire_render::OPCODE_DRAW_INDEXED_WIDE;
    st32(&mut command[0..], op);
    st32(&mut command[4..], 0x20);
    st16(&mut command[8..], 3);
    st32(&mut command[12..], 0x3e);
    st32(&mut command[16..], 6);
    let records = 71;
    for _ in 0..records {
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
    }

    assert_eq!(acc.draws.len(), records, "no draw may be truncated away");
    assert_eq!(
        acc.dropped_no_pipeline, 0,
        "all of these had a pipeline bound"
    );

    // With no pipeline latched the same record is the other arm: still not
    // a `PendingDraw`, but counted rather than vanishing.
    let mut unbound = StreamAccum::default();
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut unbound);
    assert_eq!(unbound.dropped_no_pipeline, 1);
    assert!(unbound.draws.is_empty());
}

/// Setting the render pipeline to ref 0 unbinds it rather than being ignored.
///
/// `SetPipeline` was guarded `if cmd.pipeline_ref != 0`, and the arm that
/// caught the failed guard is the match's bare `_ => {}`. So a zero ref left
/// whatever pipeline was latched before it in place, and every following draw
/// encoded against a pipeline the guest had stopped asking for — a wrong
/// frame, silently, with `dropped_no_pipeline` reading zero because the draws
/// were not dropped at all.
///
/// This asserts the outcome rather than the field: after a zero ref, a draw
/// that would otherwise have been kept lands in `dropped_no_pipeline` and no
/// `PendingDraw` carries the stale ref. On the guarded code the draw is
/// pushed with pipeline 61 and both assertions fail.
#[test]
fn setting_the_render_pipeline_to_ref_zero_unbinds_it() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum {
        pipeline_ref: 61,
        ..Default::default()
    };

    let mut set_pipeline = vec![0u8; wire_render::SET_STATE_TOTAL_LEN as usize];
    st32(
        &mut set_pipeline[0..],
        wire_render::OPCODE_SET_RENDER_PIPELINE_STATE,
    );
    st32(&mut set_pipeline[4..], wire_render::SET_STATE_TOTAL_LEN);
    st32(&mut set_pipeline[8..], 0);
    handle_render_record(
        &mut state,
        &host,
        1,
        wire_render::OPCODE_SET_RENDER_PIPELINE_STATE,
        &set_pipeline,
        &mut out,
        &mut acc,
    );
    assert_eq!(
        acc.pipeline_ref, 0,
        "the decoded ref is what the accumulator latches"
    );

    let mut draw = vec![0u8; 0x20];
    let op = wire_render::OPCODE_DRAW_INDEXED_WIDE;
    st32(&mut draw[0..], op);
    st32(&mut draw[4..], 0x20);
    st16(&mut draw[8..], 3);
    st32(&mut draw[12..], 0x3e);
    st32(&mut draw[16..], 6);
    handle_render_record(&mut state, &host, 1, op, &draw, &mut out, &mut acc);

    assert_eq!(
        acc.dropped_no_pipeline, 1,
        "a draw after an unbind is declined by name, not encoded against the old pipeline"
    );
    assert!(
        acc.draws.is_empty(),
        "no draw may carry the pipeline the guest unbound"
    );
}

/// A stream that binds once and draws many times must not copy its bind
/// tables per draw.
///
/// This is the property that makes an unbounded draw list affordable, and
/// therefore the property the cap's removal rests on. It is asserted by
/// pointer identity because that is the only thing that distinguishes a
/// shared table from an equal copy — `assert_eq!` on the contents passes
/// either way, which is exactly how a regression here would hide.
#[test]
fn draws_sharing_a_bind_table_share_its_allocation() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum {
        pipeline_ref: 61,
        vertex_buffers: Arc::new(vec![BufferBind {
            index: 0,
            buffer_ref: 9,
            offset: 0,
            attribute_stride: None,
            ..Default::default()
        }]),
        ..Default::default()
    };
    let mut command = vec![0u8; 0x20];
    let op = wire_render::OPCODE_DRAW_INDEXED_WIDE;
    st32(&mut command[0..], op);
    st32(&mut command[4..], 0x20);
    st16(&mut command[8..], 3);
    st32(&mut command[12..], 0x3e);
    st32(&mut command[16..], 6);
    for _ in 0..100 {
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
    }

    assert_eq!(acc.draws.len(), 100);
    for (i, pd) in acc.draws.iter().enumerate() {
        assert!(
            Arc::ptr_eq(&pd.vertex_buffers, &acc.vertex_buffers),
            "draw {i} copied a bind table nothing had changed"
        );
    }
}

/// Backend preparation consumes the same retained tables the recorded draw
/// owns. Copying at this boundary would preserve pixels while putting one heap
/// allocation and element copy back in front of every draw, so content equality
/// is not a sufficient regression check.
#[test]
fn draw_preparation_keeps_every_recorded_bind_table_allocation() {
    let buffer = || {
        Arc::new(vec![BufferBind {
            index: 0,
            buffer_ref: 9,
            offset: 16,
            attribute_stride: None,
            ..Default::default()
        }])
    };
    let texture = || {
        Arc::new(vec![TextureBind {
            index: 1,
            texture_ref: 10,
            ..Default::default()
        }])
    };
    let sampler = || {
        Arc::new(vec![SamplerBind {
            index: 2,
            sampler_ref: 11,
            lod_clamp: None,
        }])
    };
    let pd = PendingDraw {
        vertex_buffers: buffer(),
        fragment_buffers: buffer(),
        vertex_textures: texture(),
        fragment_textures: texture(),
        vertex_samplers: sampler(),
        fragment_samplers: sampler(),
        ..Default::default()
    };
    let mut req = crate::runtime::draw::DrawEncodeRequest::default();
    fill_draw_binds_from_pending(&mut req, &pd);

    assert!(Arc::ptr_eq(&req.vertex_buffers, &pd.vertex_buffers));
    assert!(Arc::ptr_eq(&req.fragment_buffers, &pd.fragment_buffers));
    assert!(Arc::ptr_eq(&req.vertex_textures, &pd.vertex_textures));
    assert!(Arc::ptr_eq(&req.fragment_textures, &pd.fragment_textures));
    assert!(Arc::ptr_eq(&req.vertex_samplers, &pd.vertex_samplers));
    assert!(Arc::ptr_eq(&req.fragment_samplers, &pd.fragment_samplers));
}

/// A bind that changes after a draw must not reach back into that draw.
///
/// The other half of the copy-on-write contract: sharing is only safe if a
/// later mutation forks. `Arc::make_mut` is what does that, and a mutation
/// site that reached the `Vec` some other way would silently rewrite a
/// snapshot the guest already committed to.
#[test]
fn a_bind_after_a_draw_does_not_rewrite_that_draws_snapshot() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum {
        pipeline_ref: 61,
        vertex_buffers: Arc::new(vec![BufferBind {
            index: 0,
            buffer_ref: 9,
            offset: 0,
            attribute_stride: None,
            ..Default::default()
        }]),
        ..Default::default()
    };
    let mut command = vec![0u8; 0x20];
    let op = wire_render::OPCODE_DRAW_INDEXED_WIDE;
    st32(&mut command[0..], op);
    st32(&mut command[4..], 0x20);
    st16(&mut command[8..], 3);
    st32(&mut command[12..], 0x3e);
    st32(&mut command[16..], 6);
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);

    apply_binds(
        // A bare entry, declared here: `apply_binds` is generic over the entry
        // type precisely so it takes whichever wire shape a record carries, and
        // this test is about the walk rather than about any one of them.
        &[(77u32, 0u64)],
        0,
        BindTarget {
            stage: ShaderStage::Vertex,
            class: BindClass::Buffer,
        },
        BindTables {
            vertex: &mut acc.vertex_buffers,
            fragment: &mut acc.fragment_buffers,
            refused: &mut acc.unrepresentable,
        },
        |b| b.index,
        |index, (buffer_ref, offset): &(u32, u64)| {
            Some(BufferBind {
                index,
                buffer_ref: *buffer_ref,
                offset: *offset,
                ..Default::default()
            })
        },
    );

    assert_eq!(
        acc.draws[0].vertex_buffers[0].buffer_ref, 9,
        "the committed draw kept the buffer it was encoded with"
    );
    assert_eq!(acc.vertex_buffers[0].buffer_ref, 77);
}

#[test]
fn a_recorded_buffer_bind_retains_its_object_across_offset_change_and_ref_reuse() {
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_BUFFER,
    };
    use crate::runtime::gva_mem::{define_task_pages_arm64e, write_task_gva_arm64e};
    use crate::runtime::objects;

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));
    let put_buffer = |host: &mut FakeHost, state: &DeviceState, handle: u32, size: u64| {
        let descriptor_gva = 0x180;
        let mut descriptor = [0u8; 16];
        st64(&mut descriptor, size);
        st32(&mut descriptor[8..], handle);
        write_task_gva_arm64e(host, &state.tasks[1], descriptor_gva, &descriptor);
        let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        st32(&mut entry, u32::from(OBJECT_TYPE_BUFFER) | (16 << 8));
        st64(&mut entry[4..], descriptor_gva);
        write_task_gva_arm64e(
            host,
            &state.tasks[1],
            list_object_entry_offset(7, 32).unwrap(),
            &entry,
        );
    };
    put_buffer(&mut host, &state, 5, 0x1000);

    let total = OP_HEADER_LEN + render_pass::BIND_ENTRIES + render_pass::BUFFER_BIND_ENTRY_SIZE;
    let mut bind = vec![0u8; total];
    st32(&mut bind, wire_render::OPCODE_SET_VERTEX_BUFFER);
    st32(&mut bind[4..], total as u32);
    st32(&mut bind[OP_HEADER_LEN + render_pass::BIND_COUNT..], 1);
    st32(&mut bind[OP_HEADER_LEN + render_pass::BIND_ENTRIES..], 7);
    let mut out = ExecResult::default();
    let mut acc = StreamAccum {
        pipeline_ref: 61,
        ..Default::default()
    };
    handle_render_record(
        &mut state,
        &host,
        1,
        wire_render::OPCODE_SET_VERTEX_BUFFER,
        &bind,
        &mut out,
        &mut acc,
    );
    let first = acc.vertex_buffers[0]
        .resource
        .clone()
        .expect("setter retain");

    let offset_total = OP_HEADER_LEN + render_pass::BUFFER_OFFSET_PAYLOAD_LEN;
    let mut offset = vec![0u8; offset_total];
    st32(&mut offset, wire_render::OPCODE_SET_VERTEX_BUFFER_OFFSET);
    st32(&mut offset[4..], offset_total as u32);
    st64(
        &mut offset[OP_HEADER_LEN + render_pass::BUFFER_OFFSET_VALUE..],
        0x80,
    );
    handle_render_record(
        &mut state,
        &host,
        1,
        wire_render::OPCODE_SET_VERTEX_BUFFER_OFFSET,
        &offset,
        &mut out,
        &mut acc,
    );
    assert_eq!(acc.vertex_buffers[0].offset, 0x80);
    assert!(Arc::ptr_eq(
        &first,
        acc.vertex_buffers[0].resource.as_ref().unwrap()
    ));

    let mut draw = vec![0u8; 0x20];
    let draw_op = wire_render::OPCODE_DRAW_INDEXED_WIDE;
    st32(&mut draw, draw_op);
    st32(&mut draw[4..], 0x20);
    st16(&mut draw[8..], 3);
    st32(&mut draw[12..], 0x3e);
    st32(&mut draw[16..], 6);
    handle_render_record(&mut state, &host, 1, draw_op, &draw, &mut out, &mut acc);
    assert!(state.delete_object(1, 7));
    put_buffer(&mut host, &state, 6, 0x2000);
    let replacement = objects::resolve_resource(&state, &host, 1, 7).unwrap();
    assert!(!Arc::ptr_eq(&first, &replacement));
    let recorded = acc.draws[0].vertex_buffers[0].resource.as_ref().unwrap();
    assert!(Arc::ptr_eq(recorded, &first));
    assert_eq!(
        objects::resolve_buffer_span_from_resource(&state, recorded),
        Ok((5u64 << PAGE_SHIFT_ARM64E, 0x1000))
    );
}

#[test]
fn a_texture_slot_replaces_object_identity_only_on_a_later_setter() {
    use crate::model::TaskResource;
    use crate::runtime::decode::resource::ListObjectEntry;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    // Published through the namespace, because that is the only issuer of the
    // name the memo is keyed by.
    state.define_task(1, 0x1000, 1);
    let name = state
        .declare_object(1, 9, reims_vgpu_core::lifecycle::Storage::NoBytes)
        .expect("task 1 is defined")
        .id;
    let first = state.task_resources.register(
        1,
        name,
        Arc::new(TaskResource::new(ListObjectEntry::default(), Arc::from([]))),
    );
    let total = OP_HEADER_LEN + render_pass::BIND_ENTRIES + 4;
    let mut command = vec![0u8; total];
    st32(&mut command, wire_render::OPCODE_SET_FRAGMENT_TEXTURE);
    st32(&mut command[4..], total as u32);
    st32(&mut command[OP_HEADER_LEN + render_pass::BIND_COUNT..], 1);
    st32(&mut command[OP_HEADER_LEN + render_pass::BIND_ENTRIES..], 9);
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    handle_render_record(
        &mut state,
        &host,
        1,
        wire_render::OPCODE_SET_FRAGMENT_TEXTURE,
        &command,
        &mut out,
        &mut acc,
    );
    assert!(Arc::ptr_eq(
        acc.fragment_textures[0].resource.as_ref().unwrap(),
        &first
    ));

    assert!(state.task_resources.delete(1, name));
    // A second declaration into the same slot, which is a new name: the
    // generation is what tells the replacement from what it replaced.
    let replacement_name = state
        .declare_object(1, 9, reims_vgpu_core::lifecycle::Storage::NoBytes)
        .expect("task 1 is defined")
        .id;
    assert_ne!(replacement_name, name);
    let replacement = state.task_resources.register(
        1,
        replacement_name,
        Arc::new(TaskResource::new(ListObjectEntry::default(), Arc::from([]))),
    );
    assert!(Arc::ptr_eq(
        acc.fragment_textures[0].resource.as_ref().unwrap(),
        &first
    ));
    handle_render_record(
        &mut state,
        &host,
        1,
        wire_render::OPCODE_SET_FRAGMENT_TEXTURE,
        &command,
        &mut out,
        &mut acc,
    );
    assert!(Arc::ptr_eq(
        acc.fragment_textures[0].resource.as_ref().unwrap(),
        &replacement
    ));
}

#[test]
fn accepted_render_without_executor_is_fail_visible() {
    // The emit is deduped per opcode process-wide; hold the shared latch
    // lock and clear it so this test always observes its first-sighting line.
    let _guard = UNIMPL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset_unimplemented_opcode_dedup_for_test();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum {
        pipeline_ref: 0xface,
        ..Default::default()
    };
    let task_id = 0xfeed;
    let mut command = vec![0u8; OP_HEADER_LEN];
    // An opcode no decoder on this rail claims, found rather than named. It has
    // been three different literals over this test's life — each stopped working
    // the moment that number turned out to be a record — so it is searched for
    // against the two predicates that decide the routing: no `RenderKind` names
    // it and `render_spi` does not own it.
    //
    // What is under test is the `ErrUnknownOpcode` arm, which is the successor
    // to the decoder's `OtherAccepted` catch-all and keeps its line and its wire
    // capture. The old `unclaimed_accepted_opcode` searched inside an "accepted
    // window" that no longer exists.
    let op = (0x00..0x400u32)
        .find(|op| {
            reims_vgpu_protocol::render::RenderKind::of_opcode(*op).is_none()
                && !crate::runtime::decode::render_spi::is_unsettled(*op)
                && reims_vgpu_protocol::closure::find(
                    reims_vgpu_protocol::closure::Rail::Render,
                    *op,
                )
                .is_none()
        })
        .expect("some opcode in the render space is claimed by no decoder");
    st32(&mut command[0..], op);
    st32(&mut command[4..], OP_HEADER_LEN as u32);
    handle_render_record(&mut state, &host, task_id, op, &command, &mut out, &mut acc);

    let body = std::fs::read_to_string(crate::observe::fail_log_path())
        .expect("reims-vgpu-fail.log readable");
    let want = format!(
        "render_unimplemented reason=accepted_without_executor task=65261 opcode={op:#x} len=8"
    );
    assert!(
        body.lines()
            .any(|line| line.contains(&want) && line.contains("pipeline=64206")),
        "no line matching {want:?}"
    );
}

/// Regression guard: the accepted-without-executor line is deduped to ONE
/// emission per distinct opcode (a per-draw undecoded op must not flood the
/// always-on sink), while distinct opcodes still each report once and the
/// raw wire is captured. This locks the anti-flood behavior that replaced
/// the ~2620-line-per-workload per-draw emit.
#[test]
fn unimplemented_render_opcode_dedups_per_opcode_with_wire() {
    let _guard = UNIMPL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset_unimplemented_opcode_dedup_for_test();
    let task = 0x5151u32;
    let acc = StreamAccum {
        pipeline_ref: 0x1234,
        ..Default::default()
    };
    let wire: Vec<u8> = vec![0xde, 0xad, 0xbe, 0xef, 0x10, 0x00, 0x00, 0x00];

    // First sighting of an opcode emits; every repeat is deduped (no flood).
    assert!(
        note_unimplemented_render_opcode(0x7c, &wire, task, &acc),
        "first sighting must emit",
    );
    for _ in 0..24 {
        assert!(
            !note_unimplemented_render_opcode(0x7c, &wire, task, &acc),
            "a repeated opcode must be deduped",
        );
    }
    // A distinct opcode reports once independently of the first.
    assert!(note_unimplemented_render_opcode(0x9a, &wire, task, &acc));
    assert!(!note_unimplemented_render_opcode(0x9a, &wire, task, &acc));
    // Out-of-range opcodes (decode desync) are also deduped, not flooded.
    assert!(note_unimplemented_render_opcode(
        0x1_0001, &wire, task, &acc
    ));
    assert!(!note_unimplemented_render_opcode(
        0x1_0001, &wire, task, &acc
    ));

    // The first-sighting line captured the raw wire for offline decode.
    let body = std::fs::read_to_string(crate::observe::fail_log_path())
        .expect("reims-vgpu-fail.log readable");
    assert!(
        body.lines().any(|l| l.contains(&format!("task={task}"))
            && l.contains("opcode=0x7c")
            && l.contains("hex=deadbeef10000000")),
        "the raw wire must be captured on first sighting",
    );
}

/// The render rail's boundary counter must name the *check* that dropped the
/// draw, not the class it was flattened into.
///
/// Before `EncodeStatus` carried its reason this line read
/// `draw_encode_fail reason=bad_args`, and `bad_args` alone spoke for eight
/// distinct refusals in `encode_draw_chain_inner` — a zero-size target, a
/// vertexless draw, an MRT slot with no backing. A window that never painted
/// gave you the class and never the cause.
#[test]
fn a_dropped_draw_names_which_check_refused_not_just_its_class() {
    let task = 81u32;
    // Distinct from every other pipeline in the suite: `fail_once` latches per
    // (reason, pipeline) for the whole process.
    let pipe = 249_001u32;
    note_draw_encode_fail(
        task,
        pipe,
        EncodeStatus::BadArgs("draw_mtl_zero_geom"),
        1,
        3,
    );
    let body = sink_body();
    assert!(
        body.lines().any(|l| l
            .contains("draw_encode_fail reason=draw_mtl_zero_geom class=bad_args")
            && l.contains(&format!("pipe={pipe}"))
            && l.contains(&format!("task={task}"))
            && l.contains("di=1/3")),
        "the boundary line must carry the specific check and the class:\n{body}"
    );

    // Latched per (reason, pipeline): the guest re-submits the same failing
    // draw every frame, so a repeat adds nothing the first line did not…
    note_draw_encode_fail(
        task,
        pipe,
        EncodeStatus::BadArgs("draw_mtl_zero_geom"),
        2,
        3,
    );
    // …but a *different* check on the same pipeline is a different event and
    // must still be visible. Latching on the class would have hidden it, which
    // is exactly the failure this migration removes.
    note_draw_encode_fail(
        task,
        pipe,
        EncodeStatus::MetalFailed("draw_mtl_core_failed"),
        2,
        3,
    );
    let body = sink_body();
    assert_eq!(
        body.matches("reason=draw_mtl_zero_geom").count(),
        1,
        "a re-attempted refusal must log once:\n{body}"
    );
    assert!(
        body.contains("reason=draw_mtl_core_failed"),
        "a second check on the same pipeline must not be latched away:\n{body}"
    );

    // Success never reaches the sink — `Emit::refusal` has no line to send for
    // `Ok`, so the carve-out is enforced by the type rather than by a `return`
    // a future arm could forget.
    let before = sink_body().matches("draw_encode_fail").count();
    note_draw_encode_fail(task, pipe, EncodeStatus::Ok, 0, 1);
    assert_eq!(
        sink_body().matches("draw_encode_fail").count(),
        before,
        "an Ok encode logged a failure line"
    );
}

#[test]
fn zero_ref_render_bind_unbinds_existing_slots() {
    // wire opcodes via wire_render import

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    let mut buffer = vec![0u8; OP_HEADER_LEN + 8 + 12];
    st32(&mut buffer[0..], wire_render::OPCODE_SET_VERTEX_BUFFER);
    st32(&mut buffer[4..], (OP_HEADER_LEN + 8 + 12) as u32);
    st32(&mut buffer[8..], 0);
    st32(&mut buffer[12..], 1);
    st32(&mut buffer[16..], 41);
    handle_render_record(
        &mut state,
        &host,
        0,
        wire_render::OPCODE_SET_VERTEX_BUFFER,
        &buffer,
        &mut out,
        &mut acc,
    );
    st32(&mut buffer[16..], 0);
    handle_render_record(
        &mut state,
        &host,
        0,
        wire_render::OPCODE_SET_VERTEX_BUFFER,
        &buffer,
        &mut out,
        &mut acc,
    );
    assert!(acc.vertex_buffers.is_empty());

    for (opcode, bound) in [
        (wire_render::OPCODE_SET_FRAGMENT_TEXTURE, 42u32),
        (wire_render::OPCODE_SET_FRAGMENT_SAMPLER, 43u32),
    ] {
        let mut command = vec![0u8; OP_HEADER_LEN + 8 + 4];
        st32(&mut command[0..], opcode);
        st32(&mut command[4..], (OP_HEADER_LEN + 8 + 4) as u32);
        st32(&mut command[8..], 3);
        st32(&mut command[12..], 1);
        st32(&mut command[16..], bound);
        handle_render_record(&mut state, &host, 0, opcode, &command, &mut out, &mut acc);
        st32(&mut command[16..], 0);
        handle_render_record(&mut state, &host, 0, opcode, &command, &mut out, &mut acc);
    }
    assert!(acc.fragment_textures.is_empty());
    assert!(acc.fragment_samplers.is_empty());
    assert_eq!(out.buffer_unbinds, 1);
    assert_eq!(out.texture_unbinds, 1);
    assert_eq!(out.sampler_unbinds, 1);
}

/// x86 backing display mid: clear-only stream must Store solid BGRA into pages.
#[test]
fn clear_only_backing_surface_writes_guest_pages() {
    use crate::protocol::endian::{st32, st64};
    use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::runtime::objects::{self, OBJECT_TYPE_BACKING};
    use crate::runtime::render_pass::ColorAttachment;

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.page_shift = PAGE_SHIFT_X86;
    // Surface pages at pfn 0x40 (one 4K page is enough for 16×16).
    let page = 0x40u64 << PAGE_SHIFT_X86;
    host.map_range(page, 0x2000, 0);
    // Task directory so object-list GVA reads work.
    let dir_gpa = 2u64 << PAGE_SHIFT_X86;
    let root_gpa = 3u64 << PAGE_SHIFT_X86;
    let data_gpa = 4u64 << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    host.map_range(data_gpa, 0x200, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    st32(&mut d[..4], 4);
    let _ = host.write_gpa(root_gpa, &d[..4]);
    // Map the backing GVA page onto the surface pages. The device refuses a
    // backing it cannot translate rather than reusing the GVA as a GPA, so
    // the task's page table has to carry this the way a guest's does.
    st32(&mut d[..4], 0x40);
    let _ = host.write_gpa(root_gpa + 0x40 * 4, &d[..4]);
    state.define_task(1, 0x1000, 2);
    assert!(state.set_object_list(1, 0, 8));
    // Backing at surface_id=5.
    let mut entry = [0u8; 12];
    st32(
        &mut entry[0..],
        (OBJECT_TYPE_BACKING as u32) | (0x30u32 << 8),
    );
    entry[4..12].copy_from_slice(&0x80u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 5 * 12, &entry);
    let mut desc = vec![0u8; 0x30];
    st64(&mut desc[0..], 0x1000);
    st32(&mut desc[8..], 0x40); // identity pfn
    st32(&mut desc[0xc..], 0x4247_5241); // 'BGRA'
    desc[0x10] = 1;
    st32(&mut desc[0x18..], 16);
    st32(&mut desc[0x1c..], 16);
    st32(&mut desc[0x20..], 64);
    let _ = host.write_gpa(data_gpa + 0x80, &desc);

    assert!(objects::resolve_backing(&mut state, &host, 5));
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    acc.clears.push(ColorAttachment {
        texture_ref: 5,
        resolve_texture_ref: 0,
        level: 0,
        slice: 0,
        depth_plane: 0,
        load_action: MTL_LOAD_ACTION_CLEAR,
        store_action: MTL_STORE_ACTION_STORE,
        clear_color: [1.0, 0.0, 0.0, 1.0], // red → BGRA (0,0,255,255)
    });
    finish_stream(&mut state, &mut host, 1, &mut out, &acc);
    assert!(
        out.clears_applied >= 1,
        "backing clear must apply, got {}",
        out.clears_applied
    );
    // Read first pixel from guest page (BGRA).
    let mut px = [0u8; 4];
    assert!(host.read_gpa(page, &mut px).is_ok());
    assert_eq!(px, [0, 0, 255, 255], "expected opaque red BGRA, got {px:?}");
    let m = state.mappings.get(&5).expect("mapping");
    assert!(m.content_generation > 0 || m.mapped);
    let _ = PAGE_ENTRY_VALID;
    let _ = PAGE_ENTRY_PFN_SHIFT;
}

/// Archive DrawJob: clear-only packets store immediately; multi-draw packets
/// keep CLEAR as private Metal seed (no pre-draw guest clear).
#[test]
fn finish_stream_clear_only_branch_without_draws() {
    use crate::runtime::render_pass::ColorAttachment;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    acc.clears.push(ColorAttachment {
        texture_ref: 99,
        resolve_texture_ref: 0,
        level: 0,
        slice: 0,
        depth_plane: 0,
        load_action: MTL_LOAD_ACTION_CLEAR,
        store_action: MTL_STORE_ACTION_STORE,
        clear_color: [0.0, 0.0, 0.0, 1.0],
    });
    // No draws → clear-only branch (attempts apply_clear; unresolvable ref).
    finish_stream(&mut state, &mut host, 1, &mut out, &acc);
    assert_eq!(out.metal_draws_ok, 0);
    assert_eq!(out.metal_draws_fail, 0);
}

/// A clear-only integer attachment publishes the integer components themselves.
///
/// The GPU draw path already carries the clear as a typed union member. This
/// exercises the other publication path: no draw exists, so `finish_stream`
/// must materialize the result directly in the mapper-ref-texture mapping's native texels.
#[test]
fn clear_only_rg16uint_publishes_native_guest_texels() {
    use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::protocol::pixel_format::{MTL_FORMAT_RG16_UINT, RG16_BPP};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let (task_id, texture_ref, mapping_id) = (1u32, 0x91u32, 17u32);
    let pfn = 0x71u32;
    let page = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(page, 1usize << PAGE_SHIFT_ARM64E, 0xcc);

    assert!(state.map_surface(mapping_id));
    {
        let mapping = state.mappings.get_mut(&mapping_id).expect("mapping");
        mapping.mapped = true;
        mapping.mapping_internal = 1;
        mapping.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    assert!(state.set_mapping_geom(mapping_id, 2, 2, MTL_FORMAT_RG16_UINT));
    state
        .texture_to_mapping
        .insert((task_id, texture_ref), mapping_id);

    let (_, row_stride, _) = mapping_write::mapper_ref_texture_sample_window(
        state.mappings.get(&mapping_id).expect("mapping"),
        2,
        2,
        MTL_FORMAT_RG16_UINT,
    )
    .expect("the mapper-ref-texture has a sample window");
    assert!(row_stride >= 2 * RG16_BPP);

    let mut acc = StreamAccum::default();
    acc.clears.push(ColorAttachment {
        texture_ref,
        load_action: MTL_LOAD_ACTION_CLEAR,
        store_action: MTL_STORE_ACTION_STORE,
        clear_color: [1.0, 258.0, 65_535.0, 0.0],
        ..Default::default()
    });
    let mut out = ExecResult::default();
    finish_stream(&mut state, &mut host, task_id, &mut out, &acc);

    assert_eq!(out.clears_applied, 1, "the clear-only Store must publish");
    let expected = [1u16.to_le_bytes(), 258u16.to_le_bytes()]
        .concat()
        .repeat(2);
    for y in 0..2u64 {
        let mut row = vec![0u8; expected.len()];
        host.read_gpa(page + y * u64::from(row_stride), &mut row)
            .expect("read clear result");
        assert_eq!(row, expected, "row {y} keeps the integer clear values");
    }
    if row_stride > 2 * RG16_BPP {
        let mut padding = [0u8; 1];
        host.read_gpa(page + u64::from(2 * RG16_BPP), &mut padding)
            .expect("read row padding");
        assert_eq!(padding, [0xcc], "the writer must not clear row padding");
    }
}

/// The same clear representation reaches a linear normal-texture target, including
/// its guest-declared row pitch rather than a tightly packed substitute.
#[test]
fn clear_only_rg16uint_publishes_native_linear_gva_rows() {
    use crate::protocol::pixel_format::MTL_FORMAT_RG16_UINT;
    use crate::runtime::decode::resource::{
        list_object_entry_offset, LINEAR_DESC_HANDLE, LINEAR_DESC_SIZE, OBJECT_LIST_ENTRY_LEN,
        OBJECT_TYPE_TEXTURE, TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_HEIGHT, TEXTURE_DESC_PIXEL_FORMAT,
        TEXTURE_DESC_ROW_STRIDE, TEXTURE_DESC_WIDTH,
    };
    use crate::runtime::gva_mem::{
        define_task_pages_arm64e, read_task_gva_by_id, write_task_gva_arm64e,
    };
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    define_task_pages_arm64e(&mut host, &mut state, 4, 16);
    assert!(state.set_object_list(1, 0, 256));

    let (texture_ref, handle, width, height, row_stride) = (200u32, 8u32, 2u32, 2u32, 16u32);
    let target_gva = (handle as u64) << PAGE_SHIFT_ARM64E;
    let mut prior = vec![0xcc; (row_stride * height) as usize];
    write_task_gva_arm64e(&mut host, &state.tasks[1], target_gva, &prior);
    crate::runtime::surface_cache::store_texture(
        &mut state,
        1,
        texture_ref,
        width,
        height,
        vec![0x11; (width * height * 4) as usize],
        target_gva,
    );
    crate::runtime::surface_cache::store_gva_owned(
        &mut state,
        target_gva,
        width,
        height,
        vec![0x22; (width * height * 4) as usize],
        OBJECT_TYPE_TEXTURE,
        None,
        true,
    );

    let mut desc = vec![0u8; TEXTURE_DESC_BASE_LEN];
    st64(
        &mut desc[LINEAR_DESC_SIZE..],
        u64::from(row_stride) * u64::from(height),
    );
    st32(&mut desc[LINEAR_DESC_HANDLE..], handle);
    st32(&mut desc[TEXTURE_DESC_ROW_STRIDE..], row_stride);
    st32(&mut desc[TEXTURE_DESC_WIDTH..], width);
    st32(&mut desc[TEXTURE_DESC_HEIGHT..], height);
    st16(&mut desc[TEXTURE_DESC_PIXEL_FORMAT..], MTL_FORMAT_RG16_UINT);
    let desc_gva = 0x280u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &desc);
    let entry_off = list_object_entry_offset(texture_ref, 256).expect("object-list ref");
    let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut entry,
        (OBJECT_TYPE_TEXTURE as u32) | ((desc.len() as u32) << 8),
    );
    entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], entry_off, &entry);

    let mut acc = StreamAccum::default();
    acc.clears.push(ColorAttachment {
        texture_ref,
        load_action: MTL_LOAD_ACTION_CLEAR,
        store_action: MTL_STORE_ACTION_STORE,
        clear_color: [1.0, 258.0, 0.0, 0.0],
        ..Default::default()
    });
    let mut out = ExecResult::default();
    finish_stream(&mut state, &mut host, 1, &mut out, &acc);
    assert_eq!(
        out.clears_applied, 1,
        "the linear clear-only Store must publish"
    );
    assert!(
        crate::runtime::surface_cache::get_texture(&state, 1, texture_ref, width, height).is_none(),
        "the object-keyed copy must not outlive pixels published to guest pages"
    );
    assert!(
        !crate::runtime::surface_cache::has_gva(&state, target_gva, width, height),
        "the GVA-keyed copy must not outlive pixels published to guest pages"
    );

    read_task_gva_by_id(
        &host,
        &state.tasks,
        1,
        target_gva,
        &mut prior,
        PAGE_SHIFT_ARM64E,
    )
    .expect("read linear clear result");
    let expected = [1u16.to_le_bytes(), 258u16.to_le_bytes()]
        .concat()
        .repeat(width as usize);
    assert_eq!(&prior[..expected.len()], expected);
    assert_eq!(
        &prior[row_stride as usize..row_stride as usize + expected.len()],
        expected
    );
    assert!(
        prior[expected.len()..row_stride as usize]
            .iter()
            .all(|&byte| byte == 0xcc),
        "the clear must leave guest row padding untouched"
    );
}

#[test]
fn finish_stream_with_draws_skips_guest_clear_prelude() {
    use crate::runtime::draw::BufferBind;
    use crate::runtime::render_pass::ColorAttachment;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    let att = ColorAttachment {
        texture_ref: 99,
        resolve_texture_ref: 0,
        level: 0,
        slice: 0,
        depth_plane: 0,
        load_action: MTL_LOAD_ACTION_CLEAR,
        store_action: MTL_STORE_ACTION_STORE,
        clear_color: [1.0, 0.0, 0.0, 1.0],
    };
    acc.clears.push(att);
    acc.saw_draw = true;
    acc.color_slots.push((0, att));
    acc.draws.push(PendingDraw {
        visibility: None,
        pipeline_ref: 1,
        draw: DrawArgs {
            vertex_count: 3,
            instance_count: 1,
            primitive_type: 3,
            first_vertex: 0,
            base_instance: 0,
        },
        indexed: None,
        vertex_buffers: Arc::new(vec![BufferBind {
            index: 0,
            buffer_ref: 1,
            offset: 0,
            attribute_stride: None,
            ..Default::default()
        }]),
        fragment_buffers: Arc::default(),
        vertex_textures: Arc::default(),
        fragment_textures: Arc::default(),
        vertex_samplers: Arc::default(),
        fragment_samplers: Arc::default(),
        viewports: Vec::new(),
        scissors: Vec::new(),
        blend_color: None,
        cull_mode: None,
        front_facing: None,
        fill_mode: None,
        depth_clip_mode: None,
        line_width: None,
        depth_bias: None,
        depth_stencil_ref: 0,
        stencil_ref: None,
        depth_attach: None,
        stencil_attach: None,
    });
    finish_stream(&mut state, &mut host, 1, &mut out, &acc);
    // Unresolvable RT → mrt_request fail before encode (not NoMetal); no clear.
    assert_eq!(
        out.clears_applied, 0,
        "unresolvable multi-draw must not guest-clear"
    );
}

/// Linux NoMetal: draws fail but CLEAR seed still Stores into backing pages.
#[test]
fn nometal_draw_falls_back_to_backing_clear() {
    use crate::protocol::endian::{st32, st64};
    use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::runtime::draw::BufferBind;
    use crate::runtime::objects::{self, OBJECT_TYPE_BACKING};
    use crate::runtime::render_pass::ColorAttachment;

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.page_shift = PAGE_SHIFT_X86;
    let page = 0x50u64 << PAGE_SHIFT_X86;
    host.map_range(page, 0x2000, 0);
    let dir_gpa = 2u64 << PAGE_SHIFT_X86;
    let root_gpa = 3u64 << PAGE_SHIFT_X86;
    let data_gpa = 4u64 << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    host.map_range(data_gpa, 0x200, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    st32(&mut d[..4], 4);
    let _ = host.write_gpa(root_gpa, &d[..4]);
    // As above: the backing GVA has to translate, not be assumed identity.
    st32(&mut d[..4], 0x50);
    let _ = host.write_gpa(root_gpa + 0x50 * 4, &d[..4]);
    state.define_task(1, 0x1000, 2);
    assert!(state.set_object_list(1, 0, 8));
    let mut entry = [0u8; 12];
    st32(
        &mut entry[0..],
        (OBJECT_TYPE_BACKING as u32) | (0x30u32 << 8),
    );
    entry[4..12].copy_from_slice(&0x80u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 5 * 12, &entry);
    let mut desc = vec![0u8; 0x30];
    st64(&mut desc[0..], 0x1000);
    st32(&mut desc[8..], 0x50);
    st32(&mut desc[0xc..], 0x4247_5241);
    desc[0x10] = 1;
    st32(&mut desc[0x18..], 16);
    st32(&mut desc[0x1c..], 16);
    st32(&mut desc[0x20..], 64);
    let _ = host.write_gpa(data_gpa + 0x80, &desc);
    assert!(objects::resolve_backing(&mut state, &host, 5));

    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    let att = ColorAttachment {
        texture_ref: 5,
        resolve_texture_ref: 0,
        level: 0,
        slice: 0,
        depth_plane: 0,
        load_action: MTL_LOAD_ACTION_CLEAR,
        store_action: MTL_STORE_ACTION_STORE,
        clear_color: [0.0, 1.0, 0.0, 1.0], // green
    };
    acc.clears.push(att);
    acc.saw_draw = true;
    acc.color_slots.push((0, att));
    acc.draws.push(PendingDraw {
        visibility: None,
        pipeline_ref: 7,
        draw: DrawArgs {
            vertex_count: 3,
            instance_count: 1,
            primitive_type: 3,
            first_vertex: 0,
            base_instance: 0,
        },
        indexed: None,
        vertex_buffers: Arc::new(vec![BufferBind {
            index: 0,
            buffer_ref: 1,
            offset: 0,
            attribute_stride: None,
            ..Default::default()
        }]),
        fragment_buffers: Arc::default(),
        vertex_textures: Arc::default(),
        fragment_textures: Arc::default(),
        vertex_samplers: Arc::default(),
        fragment_samplers: Arc::default(),
        viewports: Vec::new(),
        scissors: Vec::new(),
        blend_color: None,
        cull_mode: None,
        front_facing: None,
        fill_mode: None,
        depth_clip_mode: None,
        line_width: None,
        depth_bias: None,
        depth_stencil_ref: 0,
        stencil_ref: None,
        depth_attach: None,
        stencil_attach: None,
    });
    let mut second = acc.draws[0].clone();
    second.pipeline_ref = 8;
    acc.draws.push(second);
    finish_stream(&mut state, &mut host, 1, &mut out, &acc);
    assert_eq!(
        out.render_attachment_resolves, 1,
        "one render stream resolves its fixed attachment set once"
    );
    // Non-Apple: Linux encode Stores CLEAR load into backing (Ok) or
    // NoMetal clear fallback — either path must land green BGRA.
    #[cfg(feature = "backend-vulkan")]
    {
        assert!(
            out.metal_draws_ok >= 1 || out.clears_applied >= 1 || out.metal_draws_fail >= 1,
            "expected clear store path: ok={} clear={} fail={}",
            out.metal_draws_ok,
            out.clears_applied,
            out.metal_draws_fail
        );
        let mut px = [0u8; 4];
        assert!(host.read_gpa(page, &mut px).is_ok());
        // BGRA green = [0, 255, 0, 255]
        assert_eq!(px, [0, 255, 0, 255], "got {px:?}");
    }
}

/// Multi-draw packets force full-frame store on the final record even when
/// that draw carries a partial scissor (dock damage over chained wallpaper).
#[test]
fn multi_draw_force_full_store_flag_for_chained_packet() {
    assert_eq!(multi_draw_store_plan(0, 0), (false, false));
    assert_eq!(multi_draw_store_plan(1, 0), (true, false));
    assert_eq!(multi_draw_store_plan(3, 0), (false, false));
    assert_eq!(multi_draw_store_plan(3, 1), (false, false));
    assert_eq!(multi_draw_store_plan(3, 2), (true, true));
}

/// qemu-shim style: multi-draw plan is one guest writeback on the last record
/// only, with force_full so a partial scissor cannot leave wallpaper only in
/// host chain memory (archive DrawJob single completion writeback).
#[test]
fn multi_draw_store_plan_matches_archive_drawjob_writeback() {
    // Every packet size and every record within it. The whole contract is two
    // predicates over (draw_count, di), so stating it over a range costs
    // nothing and covers the boundary at draw_count == 1, where force_full
    // flips — which one packet of five does not reach.
    for n in 1..8usize {
        for di in 0..n {
            let (wb, full) = multi_draw_store_plan(n, di);
            let last = di + 1 == n;
            assert_eq!(
                wb, last,
                "writeback is the last record only (n={n} di={di})"
            );
            assert_eq!(
                full,
                last && n > 1,
                "force_full on the last record of a multi-draw packet only \
                 (n={n} di={di}); a single-draw packet may keep a local scissor"
            );
        }
    }
    assert_eq!(
        multi_draw_store_plan(0, 0),
        (false, false),
        "an empty packet writes nothing back"
    );
}

#[test]
fn multi_draw_chain_source_preserves_portable_unified_output() {
    assert_eq!(
        multi_draw_chain_source(true, false),
        MultiDrawChainSource::Resident
    );
    assert_eq!(
        multi_draw_chain_source(false, true),
        MultiDrawChainSource::Cpu
    );
    assert_eq!(
        multi_draw_chain_source(false, false),
        MultiDrawChainSource::Missing
    );
}

#[test]
fn render_pass_template_reuses_attachment_without_load_seed() {
    let first = draw::DrawEncodeRequest {
        task_id: 1,
        pipeline_ref: 7,
        vertex_count: 3,
        instance_count: 1,
        primitive_type: 3,
        colors: vec![draw::ColorRtRequest {
            slot: 0,
            texture_ref: 11,
            mapping_id: 3,
            target_gva: 0,
            row_stride: 0,
            width: 1920,
            height: 1080,
            format: 0x50,
            sample_count: 1,
            load_action: MTL_LOAD_ACTION_CLEAR,
            store_action: MTL_STORE_ACTION_STORE,
            clear_color: [0.1, 0.2, 0.3, 1.0],
            target_seed: Some(crate::runtime::draw::LoadSeed::rgba8(vec![0xbb; 16])),
            multisample_source_ref: 0,
        }],
        ..Default::default()
    };
    let template = render_pass_attachment_template(&first);
    assert!(template.colors[0].target_seed.is_none());
    assert_eq!(template.colors[0].load_action, MTL_LOAD_ACTION_LOAD);
    assert_eq!(template.colors[0].mapping_id, 3);
    assert_eq!(
        (template.colors[0].width, template.colors[0].height),
        (1920, 1080)
    );

    let draw = PendingDraw {
        pipeline_ref: 42,
        draw: DrawArgs {
            vertex_count: 6,
            instance_count: 2,
            primitive_type: 4,
            first_vertex: 9,
            base_instance: 0,
        },
        ..Default::default()
    };
    let req = retarget_render_pass_draw(&template, &draw);
    assert_eq!(req.pipeline_ref, 42);
    assert_eq!(
        (
            req.vertex_count,
            req.instance_count,
            req.primitive_type,
            req.first_vertex
        ),
        (6, 2, 4, 9)
    );
    assert_eq!(req.colors.len(), 1);
    assert_eq!(req.colors[0].mapping_id, 3);
    assert_eq!(
        first.colors[0].target_seed.as_ref().map(|s| s.bytes.len()),
        Some(16)
    );
}

#[test]
fn dropped_clear_logs_once_per_reason_target() {
    // Unique keys per case so no shared-static reset is needed (the dedup set
    // is process-global). First sighting of a (reason, tex_ref) emits (true);
    // an immediate repeat is suppressed (false); a distinct target logs again.
    assert!(note_clear_dropped(
        "nonstore_store_action",
        0x9001,
        "store_action=0 load_action=clear"
    ));
    assert!(!note_clear_dropped(
        "nonstore_store_action",
        0x9001,
        "store_action=0 load_action=clear"
    ));
    assert!(note_clear_dropped(
        "nonstore_store_action",
        0x9002,
        "store_action=0 load_action=clear"
    ));
    // A different reason on the same target is a distinct blind spot and logs.
    assert!(note_clear_dropped(
        "target_unresolved",
        0x9001,
        "color_target_request=none"
    ));
    assert!(!note_clear_dropped(
        "target_unresolved",
        0x9001,
        "color_target_request=none"
    ));
}

/// A store-action override reaches the attachment it names.
///
/// `setColorStoreAction:atIndex:` replaces what the render-pass descriptor
/// declared for one attachment, and this device honours that declared action in
/// `encode_draw_chain`'s writeback loop. So dropping the override lost work in
/// both directions, and the expensive direction is the one asserted here: a pass
/// declared `DontCare` and overridden to `Store` is content the guest asked to
/// keep and never got back.
///
/// The record is applied **by pass slot, not by position**. The fixture declares
/// slots 0 and 3 with the *unwanted* one first, so a rail indexing into the
/// vector would write the override onto slot 0 and this would fail rather than
/// pass by coincidence.
#[test]
fn a_store_action_override_reaches_the_slot_it_names() {
    use reims_vgpu_protocol::pass_action::MTL_STORE_ACTION_DONT_CARE;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    let att = |texture_ref: u32| ColorAttachment {
        texture_ref,
        resolve_texture_ref: 0,
        level: 0,
        slice: 0,
        depth_plane: 0,
        load_action: MTL_LOAD_ACTION_CLEAR,
        store_action: MTL_STORE_ACTION_DONT_CARE,
        clear_color: [0.0; 4],
    };
    acc.color_slots.push((0, att(90)));
    acc.color_slots.push((3, att(93)));

    let record = |action: u32, index: u32| {
        let total = reims_vgpu_wire::OP_HEADER_LEN + 8;
        let mut c = vec![0u8; total];
        st32(&mut c[0..], wire_render::OPCODE_SET_COLOR_STORE_ACTION);
        st32(&mut c[4..], total as u32);
        st32(&mut c[reims_vgpu_wire::OP_HEADER_LEN..], action);
        st32(&mut c[reims_vgpu_wire::OP_HEADER_LEN + 4..], index);
        c
    };

    let command = record(MTL_STORE_ACTION_STORE as u32, 3);
    handle_render_record(
        &mut state,
        &host,
        1,
        wire_render::OPCODE_SET_COLOR_STORE_ACTION,
        &command,
        &mut out,
        &mut acc,
    );
    assert_eq!(
        acc.color_slots[1].1.store_action, MTL_STORE_ACTION_STORE,
        "the override must reach the attachment at pass slot 3"
    );
    assert_eq!(
        acc.color_slots[0].1.store_action, MTL_STORE_ACTION_DONT_CARE,
        "slot 0 is at position 0 and was not named; a rail indexing by \
         position would have written it instead"
    );

    // A slot the pass never declared has nothing to override, and says so
    // rather than inventing an attachment the guest did not ask for.
    let before = crate::runtime::drain::store_route_count("render_store_action_slot_undeclared");
    let command = record(MTL_STORE_ACTION_STORE as u32, 5);
    handle_render_record(
        &mut state,
        &host,
        1,
        wire_render::OPCODE_SET_COLOR_STORE_ACTION,
        &command,
        &mut out,
        &mut acc,
    );
    assert_eq!(
        crate::runtime::drain::store_route_count("render_store_action_slot_undeclared") - before,
        1,
        "an override for an undeclared slot must name itself"
    );
    assert_eq!(
        acc.color_slots.len(),
        2,
        "and must not add an attachment the pass never declared"
    );
}

/// The depth and stencil overrides reach their own attachments, and name
/// themselves when there is none.
///
/// The colour test above shares this arm's `u16` narrowing and its opcode
/// match, but not the two branches here: these records carry **no index** —
/// there is one depth and one stencil attachment — so they are a different
/// lookup, and one of them landing on the other's attachment is the failure
/// this fixture is shaped to catch. Both are set in one record pair and both
/// are asserted, with the two starting from different actions so a rail writing
/// the wrong one cannot pass.
#[test]
fn a_depth_or_stencil_store_action_override_reaches_its_own_attachment() {
    use crate::runtime::drain::store_route_count;
    use crate::runtime::render_pass::StencilAttachment;
    use reims_vgpu_protocol::pass_action::MTL_STORE_ACTION_DONT_CARE;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();

    let record = |opcode: u32, action: u64| {
        let total = reims_vgpu_wire::OP_HEADER_LEN + 8;
        let mut c = vec![0u8; total];
        st32(&mut c[0..], opcode);
        st32(&mut c[4..], total as u32);
        crate::protocol::endian::st64(&mut c[reims_vgpu_wire::OP_HEADER_LEN..], action);
        c
    };
    let mut send = |acc: &mut StreamAccum, opcode: u32, action: u64| {
        let c = record(opcode, action);
        handle_render_record(&mut state, &host, 1, opcode, &c, &mut out, acc);
    };

    // No attachment declared: both must name themselves rather than return
    // quietly, which is the branch nothing else in the suite reaches.
    for (opcode, route) in [
        (
            wire_render::OPCODE_SET_DEPTH_STORE_ACTION,
            "render_store_action_no_depth_attachment",
        ),
        (
            wire_render::OPCODE_SET_STENCIL_STORE_ACTION,
            "render_store_action_no_stencil_attachment",
        ),
    ] {
        let before = store_route_count(route);
        send(&mut acc, opcode, u64::from(MTL_STORE_ACTION_STORE));
        assert_eq!(
            store_route_count(route) - before,
            1,
            "an override with no attachment to override must name itself"
        );
    }

    // Declared, and starting from different actions so neither branch can pass
    // by writing the other's attachment.
    acc.depth_attach = Some(DepthAttachment {
        store_action: MTL_STORE_ACTION_DONT_CARE,
        ..Default::default()
    });
    acc.stencil_attach = Some(StencilAttachment {
        store_action: MTL_STORE_ACTION_STORE,
        ..Default::default()
    });
    send(
        &mut acc,
        wire_render::OPCODE_SET_DEPTH_STORE_ACTION,
        u64::from(MTL_STORE_ACTION_STORE),
    );
    send(
        &mut acc,
        wire_render::OPCODE_SET_STENCIL_STORE_ACTION,
        u64::from(MTL_STORE_ACTION_DONT_CARE),
    );
    assert_eq!(
        acc.depth_attach.unwrap().store_action,
        MTL_STORE_ACTION_STORE,
        "the depth override did not reach the depth attachment"
    );
    assert_eq!(
        acc.stencil_attach.unwrap().store_action,
        MTL_STORE_ACTION_DONT_CARE,
        "the stencil override did not reach the stencil attachment"
    );

    // A mode past `u16` is left alone rather than narrowed into a different
    // action — the one case where applying the record is worse than not.
    let before = store_route_count("render_store_action_out_of_range");
    send(
        &mut acc,
        wire_render::OPCODE_SET_DEPTH_STORE_ACTION,
        u64::from(u32::MAX),
    );
    assert_eq!(
        store_route_count("render_store_action_out_of_range") - before,
        1
    );
    assert_eq!(
        acc.depth_attach.unwrap().store_action,
        MTL_STORE_ACTION_STORE,
        "an out-of-range mode must not have been narrowed onto the attachment"
    );
}

/// A plural scissor record reaches the accumulator whole.
///
/// Before `0x83`/`0x76` were decoded the record reached no arm at all, so a
/// guest setting its scissor through `setScissorRects:count:` got none. Then it
/// got the first and a counter for the rest. Now it gets all of them, and this
/// asserts the tail specifically: the fixture gives all three rects distinct,
/// non-empty values, so a rail that kept only entry 0 — or that copied entry 0
/// three times — fails here rather than passing on a degenerate fixture.
#[test]
fn a_plural_scissor_record_reaches_the_accumulator_whole() {
    use crate::protocol::endian::st64;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();

    let rects = [
        ScissorRect {
            x: 11,
            y: 22,
            width: 33,
            height: 44,
        },
        ScissorRect {
            x: 55,
            y: 66,
            width: 77,
            height: 88,
        },
        ScissorRect {
            x: 99,
            y: 100,
            width: 101,
            height: 102,
        },
    ];
    let op = wire_render::OPCODE_SET_SCISSOR_RECTS;
    let total = reims_vgpu_wire::OP_HEADER_LEN
        + render_pass::SCISSOR_RECTS_COUNT_LEN
        + rects.len() * render_pass::SCISSOR_PAYLOAD_LEN;
    let mut command = vec![0u8; total];
    st32(&mut command[0..], op);
    st32(&mut command[4..], total as u32);
    st64(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN..],
        rects.len() as u64,
    );
    let e0 = reims_vgpu_wire::OP_HEADER_LEN + render_pass::SCISSOR_RECTS_COUNT_LEN;
    for (n, r) in rects.iter().enumerate() {
        let at = e0 + n * render_pass::SCISSOR_PAYLOAD_LEN;
        for (i, val) in [r.x, r.y, r.width, r.height].into_iter().enumerate() {
            st64(&mut command[at + i * 8..], u64::from(val));
        }
    }

    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
    assert_eq!(
        acc.scissors,
        rects.to_vec(),
        "every rect the guest set, in the guest's order"
    );

    // The singular opcode is the same record at length one, and replaces the
    // array rather than appending to it.
    let total = reims_vgpu_wire::OP_HEADER_LEN + render_pass::SCISSOR_PAYLOAD_LEN;
    let mut command = vec![0u8; total];
    let op = wire_render::OPCODE_SET_SCISSOR;
    st32(&mut command[0..], op);
    st32(&mut command[4..], total as u32);
    for (i, val) in [1u64, 2, 3, 4].into_iter().enumerate() {
        st64(&mut command[reims_vgpu_wire::OP_HEADER_LEN + i * 8..], val);
    }
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
    assert_eq!(
        acc.scissors,
        vec![ScissorRect {
            x: 1,
            y: 2,
            width: 3,
            height: 4
        }],
        "a record of one leaves one, not one prepended to the previous three"
    );
}

/// An empty rect anywhere in a plural record refuses the whole record.
///
/// The singular arm has always refused an empty rect and kept the previous one,
/// because this rail cannot express "clip everything" and adopting a zero rect
/// would leave the next draw's clip to whatever the backend makes of it. At
/// array width the same reasoning forbids adopting the record with the empty
/// slots left out: slot order is what a shader's `[[viewport_array_index]]`
/// selects, so dropping slot 1 silently renumbers slot 2.
#[test]
fn an_empty_rect_in_a_plural_scissor_record_keeps_the_previous_state() {
    use crate::protocol::endian::st64;
    use crate::runtime::drain::store_route_count;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();

    let good = ScissorRect {
        x: 3,
        y: 4,
        width: 5,
        height: 6,
    };
    acc.scissors = vec![good];

    // Two rects, the second of them zero-width.
    let op = wire_render::OPCODE_SET_SCISSOR_RECTS;
    let total = reims_vgpu_wire::OP_HEADER_LEN
        + render_pass::SCISSOR_RECTS_COUNT_LEN
        + 2 * render_pass::SCISSOR_PAYLOAD_LEN;
    let mut command = vec![0u8; total];
    st32(&mut command[0..], op);
    st32(&mut command[4..], total as u32);
    st64(&mut command[reims_vgpu_wire::OP_HEADER_LEN..], 2);
    let e0 = reims_vgpu_wire::OP_HEADER_LEN + render_pass::SCISSOR_RECTS_COUNT_LEN;
    for (i, val) in [11u64, 22, 33, 44].into_iter().enumerate() {
        st64(&mut command[e0 + i * 8..], val);
    }
    let e1 = e0 + render_pass::SCISSOR_PAYLOAD_LEN;
    for (i, val) in [55u64, 66, 0, 88].into_iter().enumerate() {
        st64(&mut command[e1 + i * 8..], val);
    }

    let before = store_route_count("render_scissor_empty_kept_previous");
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
    assert_eq!(
        store_route_count("render_scissor_empty_kept_previous") - before,
        1,
        "an empty rect must name itself even when it is not the only one"
    );
    assert_eq!(
        acc.scissors,
        vec![good],
        "the record is refused whole, including the non-empty rect beside the empty one"
    );
}

/// A bind past the table's last slot says how many slots it dropped, and
/// which of the three tables lost them.
///
/// The counter no longer fires on anything Apple's serializer can write:
/// `MAX_TEXTURE_BIND_SLOTS` is now 128, Apple's own table exactly. So the
/// record here reaches past *that*, which only a guest writing its own stream
/// can do — and the walk still has to say what it dropped rather than end with
/// a bare `break`.
/// The count is the argument for widening the tables, so it has to be the
/// number of slots lost rather than one event — and it has to name the
/// table, because the three do not lose the same thing. The sibling slugs
/// must stay still while the texture one moves; a shared counter that
/// incremented for all three is what this replaced.
#[test]
fn a_bind_past_the_last_table_slot_reports_what_it_dropped() {
    use crate::runtime::drain::store_route_count;

    // Past Apple's own table, which is the only way to reach this bound now.
    const COUNT: u32 = MAX_TEXTURE_BIND_SLOTS + 9;
    let entry = render_pass::REF_BIND_ENTRY_SIZE;
    let total =
        reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_ENTRIES + (COUNT as usize) * entry;
    let mut command = vec![0u8; total];
    let op = wire_render::OPCODE_SET_VERTEX_TEXTURE;
    st32(&mut command[0..], op);
    st32(&mut command[4..], total as u32);
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_FIRST..],
        0,
    );
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_COUNT..],
        COUNT,
    );
    for i in 0..COUNT as usize {
        let at = reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_ENTRIES + i * entry;
        st32(&mut command[at..], 0x4000 + i as u32);
    }

    // The record itself must survive the lift; a cap that refused it whole is
    // what this counter exists to distinguish from.
    let framed = reims_vgpu_protocol::decode::op(&command, 0).expect("the record frames");
    let reims_vgpu_protocol::decode::render::RenderRecord::BindTextures(bind) =
        reims_vgpu_protocol::decode::render::decode(&framed)
            .expect("an over-table texture run must lift")
    else {
        panic!("a texture bind lifted as another record");
    };
    assert_eq!(bind.entries.len(), COUNT as usize);

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    let before = store_route_count(BindClass::Texture.past_table_route());
    let before_buf = store_route_count(BindClass::Buffer.past_table_route());
    let before_smp = store_route_count(BindClass::Sampler.past_table_route());
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
    assert_eq!(
        store_route_count(BindClass::Texture.past_table_route()) - before,
        (COUNT - MAX_TEXTURE_BIND_SLOTS) as u64,
        "the counter must name every slot dropped, not the one event"
    );
    assert_eq!(
        store_route_count(BindClass::Buffer.past_table_route()),
        before_buf,
        "a texture bind must not move the buffer table's counter"
    );
    assert_eq!(
        store_route_count(BindClass::Sampler.past_table_route()),
        before_smp,
        "a texture bind must not move the sampler table's counter"
    );
    assert_eq!(
        acc.vertex_textures.len(),
        MAX_TEXTURE_BIND_SLOTS as usize,
        "every slot the table does hold must still be bound"
    );
}

/// The dropped slots reach the **fail channel**, not only the census.
///
/// The counter above is per-window and cumulative and lives in an `OFF` line
/// among a hundred routes, where a zero route is simply absent. That is not the
/// always-on failure path a dropped guest record is owed, and the compute rail
/// gives the identical loss one — so the shape of the line is asserted here
/// rather than left to whoever next reads the log.
///
/// `apple_table` is the field that makes a reading actionable, and for textures
/// it now reads equal to `table` — 128 against 128 — which is the line saying
/// there is nothing left for an Apple guest to lose here. And
/// `table=` is the *class's* own bound, so the two lines below carry different
/// numbers for the same field — which is the point of splitting the constant.
#[test]
fn a_bind_past_the_table_renders_a_fail_line_naming_the_table() {
    use crate::observe::Emit;

    let line = Emit::decline(
        "render_bind_overflow",
        &BindSlotPastTable {
            class: BindClass::Texture,
            stage: ShaderStage::Vertex,
            index: MAX_TEXTURE_BIND_SLOTS,
            slots: 9,
        },
    )
    .render();
    assert_eq!(
        line,
        "render_bind_overflow reason=render_texture_bind_slot_past_table \
         stage=vertex index=128 slots=9 table=128 apple_table=128"
    );

    // The slug is the class's, so a buffer drop cannot be mistaken for a
    // texture one — that split is the whole reason there are three routes.
    let buffers = Emit::decline(
        "render_bind_overflow",
        &BindSlotPastTable {
            class: BindClass::Buffer,
            stage: ShaderStage::Fragment,
            index: MAX_BUFFER_BIND_SLOTS,
            slots: 1,
        },
    )
    .render();
    assert!(
        buffers.contains("reason=render_buffer_bind_slot_past_table")
            && buffers.contains("stage=fragment")
            && buffers.contains("table=31")
            && buffers.contains("apple_table=31"),
        "{buffers}"
    );
}

/// A bind at the last slot Apple's *sampler* table can name still binds.
///
/// The three classes carry three counters and three bounds, and the risk that
/// creates is the opposite of the one it fixes: a per-class bound invites
/// bounding each table by what Apple's serializer emits, which is the mistake
/// [`reims_vgpu_wire::ops::bind_limit`]'s own doc names — it would refuse a
/// guest that writes its own stream. So each bound is a *host* fact, and this
/// pins the case where the two differ most: a sampler at index 20, above
/// Apple's 16-entry sampler table and below [`MAX_SAMPLER_BIND_SLOTS`], binds
/// rather than being counted away.
#[test]
fn a_sampler_above_apples_table_but_inside_ours_still_binds() {
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_wire::ops::bind_limit;

    const FIRST: u32 = 20;
    const { assert!(FIRST >= bind_limit::SAMPLER && FIRST < MAX_SAMPLER_BIND_SLOTS) };

    let entry = render_pass::REF_BIND_ENTRY_SIZE;
    let total = reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_ENTRIES + entry;
    let mut command = vec![0u8; total];
    let op = wire_render::OPCODE_SET_VERTEX_SAMPLER;
    st32(&mut command[0..], op);
    st32(&mut command[4..], total as u32);
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_FIRST..],
        FIRST,
    );
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_COUNT..],
        1,
    );
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_ENTRIES..],
        0x3333,
    );

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    let before = store_route_count(BindClass::Sampler.past_table_route());
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);

    assert_eq!(
        store_route_count(BindClass::Sampler.past_table_route()),
        before,
        "the bound is the host table, not Apple's — this slot is inside it"
    );
    assert_eq!(
        acc.vertex_samplers
            .iter()
            .map(|s| s.index)
            .collect::<Vec<_>>(),
        vec![FIRST]
    );
}

/// A texture bind at an index past the old 32-wide band survives, and its
/// descriptor binding is its own.
///
/// This is the slot class the device used to drop: Apple's serializer emits
/// `setVertexTextures:withRange:` over ranges reaching 128, and everything from
/// 32 up was refused because `metal2vulkan` numbers its bands 32 apart, so
/// texture 68 and sampler 36 would have been one descriptor binding. Nothing
/// about the *information* forced that — the SPIR-V type says which class a
/// variable is — so `spirv_bind::widen_sampled_bands` moves the sampler band out
/// of the way and the whole 128-entry table becomes reachable.
///
/// Asserted three ways, because each alone could pass while the slot is still
/// lost: the accumulator keeps the bind, nothing is counted against the table,
/// and the binding it will carry is below the sampler band rather than inside it.
#[test]
fn a_texture_bind_past_the_old_band_binds_and_keeps_its_own_descriptor() {
    use crate::runtime::drain::store_route_count;
    use crate::runtime::spirv_bind::{SAMPLER_BINDING_BASE, TEXTURE_BINDING_BASE};
    use reims_vgpu_wire::ops::bind_limit;

    // Past the old 32-wide band, inside Apple's table, and far enough in that
    // the old numbering would have put it under a sampler.
    const FIRST: u32 = 68;
    const COUNT: u32 = 4;
    const { assert!(FIRST >= 32 && FIRST + COUNT <= bind_limit::TEXTURE) };

    let entry = render_pass::REF_BIND_ENTRY_SIZE;
    let total =
        reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_ENTRIES + (COUNT as usize) * entry;
    let mut command = vec![0u8; total];
    let op = wire_render::OPCODE_SET_VERTEX_TEXTURE;
    st32(&mut command[0..], op);
    st32(&mut command[4..], total as u32);
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_FIRST..],
        FIRST,
    );
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_COUNT..],
        COUNT,
    );
    for i in 0..COUNT as usize {
        let at = reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_ENTRIES + i * entry;
        st32(&mut command[at..], 0x9000 + i as u32);
    }

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    let before = store_route_count(BindClass::Texture.past_table_route());
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);

    assert_eq!(
        acc.vertex_textures
            .iter()
            .map(|t| t.index)
            .collect::<Vec<_>>(),
        (FIRST..FIRST + COUNT).collect::<Vec<_>>(),
        "every slot of a run past the old band must bind"
    );
    assert_eq!(
        store_route_count(BindClass::Texture.past_table_route()),
        before,
        "and none of them may be counted as lost"
    );
    // Each one's descriptor binding stays inside the texture band, so no
    // sampler can be reached by the same number.
    for index in FIRST..FIRST + COUNT {
        assert!(
            TEXTURE_BINDING_BASE + index < SAMPLER_BINDING_BASE,
            "texture {index} must not carry a sampler's binding"
        );
    }
}

/// The last slot of the texture band binds, and the same index in the buffer
/// table does not.
///
/// One constant used to bound all three classes at Metal's buffer table, so
/// texture index 31 — a slot the descriptor binding band has room for and every
/// backend can hold — was dropped because a *buffer* runs out there. Splitting
/// the bound recovers it, and the way to see that the split is real rather than
/// a rename is that the same index now gets two different answers.
#[test]
fn the_last_texture_slot_binds_where_the_same_buffer_slot_does_not() {
    use crate::runtime::drain::store_route_count;

    const LAST_TEXTURE: u32 = MAX_TEXTURE_BIND_SLOTS - 1;
    // The premise of the test: the two bounds disagree at exactly this index.
    const { assert!(LAST_TEXTURE >= MAX_BUFFER_BIND_SLOTS) };

    let one_bind = |op: u32, first: u32, obj: u32| {
        let total = reims_vgpu_wire::OP_HEADER_LEN
            + render_pass::BIND_ENTRIES
            + render_pass::REF_BIND_ENTRY_SIZE;
        let mut command = vec![0u8; total];
        st32(&mut command[0..], op);
        st32(&mut command[4..], total as u32);
        st32(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_FIRST..],
            first,
        );
        st32(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_COUNT..],
            1,
        );
        st32(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_ENTRIES..],
            obj,
        );
        command
    };

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();

    let tex_before = store_route_count(BindClass::Texture.past_table_route());
    let op = wire_render::OPCODE_SET_VERTEX_TEXTURE;
    let command = one_bind(op, LAST_TEXTURE, 0x7001);
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
    assert_eq!(
        acc.vertex_textures
            .iter()
            .map(|t| t.index)
            .collect::<Vec<_>>(),
        vec![LAST_TEXTURE],
        "the last slot of the texture band is inside the band and must bind"
    );
    assert_eq!(
        store_route_count(BindClass::Texture.past_table_route()),
        tex_before,
        "and nothing may be counted as lost for it"
    );

    // The buffer table really does end one slot earlier, so the same index
    // there is still a refusal — the split is a split, not a widening of all
    // three. A buffer entry is `{ref:u32, offset:u64}`, not the bare ref the
    // texture and sampler records carry, so it is built here rather than shared.
    let buf_before = store_route_count(BindClass::Buffer.past_table_route());
    let op = wire_render::OPCODE_SET_VERTEX_BUFFER;
    let mut command = vec![0u8; OP_HEADER_LEN + 8 + 12];
    let total = command.len() as u32;
    st32(&mut command[0..], op);
    st32(&mut command[4..], total);
    st32(&mut command[8..], LAST_TEXTURE); // first
    st32(&mut command[12..], 1); // count
    st32(&mut command[16..], 0x7002); // ref
    st64(&mut command[20..], 0); // offset
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
    assert!(
        acc.vertex_buffers.is_empty(),
        "the buffer argument table ends at {MAX_BUFFER_BIND_SLOTS}"
    );
    assert_eq!(
        store_route_count(BindClass::Buffer.past_table_route()) - buf_before,
        1,
        "and the slot it refused must be counted against the buffer table"
    );
}

/// Every bind record lands in exactly one reach band, and the top band
/// fires on the same records the drop counter counts slots for.
///
/// The bands are what make a zero from `*_bind_slot_past_table` readable: a
/// workload whose every record stops at slot 4 and one whose every record
/// stops at slot 30 both drop nothing, and only the second says the bound
/// is nearly spent. So the band has to be chosen from the reach the guest
/// *asked for*, before the walk truncates it — which is what the `le_table`
/// case below would catch if the census moved inside the loop.
#[test]
fn every_bind_record_lands_in_one_reach_band_and_the_top_one_reconciles() {
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_wire::ops::bind_limit;

    let texture_record = |first: u32, count: u32| {
        let entry = render_pass::REF_BIND_ENTRY_SIZE;
        let total =
            reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_ENTRIES + (count as usize) * entry;
        let mut command = vec![0u8; total];
        let op = wire_render::OPCODE_SET_VERTEX_TEXTURE;
        st32(&mut command[0..], op);
        st32(&mut command[4..], total as u32);
        st32(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_FIRST..],
            first,
        );
        st32(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_COUNT..],
            count,
        );
        for i in 0..count as usize {
            let at = reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_ENTRIES + i * entry;
            st32(&mut command[at..], 0x4000 + i as u32);
        }
        (op, command)
    };

    let bands = [
        "render_bind_reach_texture_le16",
        "render_bind_reach_texture_le_table",
        "render_bind_reach_texture_over_table",
    ];
    let read = || bands.map(store_route_count);

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();

    // Reach exactly Apple's sampler-table size: the lowest band, inclusive.
    let before = read();
    let (op, command) = texture_record(0, bind_limit::SAMPLER);
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
    assert_eq!(
        read()
            .iter()
            .zip(before)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>(),
        vec![1, 0, 0],
        "a reach of exactly {} is inside every one of Apple's tables",
        bind_limit::SAMPLER
    );

    // One past it, still inside this device's table.
    let before = read();
    let (op, command) = texture_record(0, bind_limit::SAMPLER + 1);
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
    assert_eq!(
        read()
            .iter()
            .zip(before)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>(),
        vec![0, 1, 0],
        "one slot past Apple's sampler table is headroom being spent, not a loss"
    );

    // Past this device's table: the band and the slot counter must agree
    // that the same record crossed, in their own units.
    let before = read();
    let before_slots = store_route_count(BindClass::Texture.past_table_route());
    let (op, command) = texture_record(MAX_TEXTURE_BIND_SLOTS - 1, 4);
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
    assert_eq!(
        read()
            .iter()
            .zip(before)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>(),
        vec![0, 0, 1],
        "a record reaching past the bound is one record in the top band"
    );
    assert_eq!(
        store_route_count(BindClass::Texture.past_table_route()) - before_slots,
        3,
        "and three slots in the drop counter — records here, slots there"
    );
}

/// Two guarded arms used to drop a decoded record into the `_ => {}`
/// catch-all, and both now name what they did instead.
///
/// The five `has_*` guards beside them are the decoder saying a field was
/// absent, so falling through those costs nothing. These two tested decoded
/// *values*, which is a different thing: an empty scissor leaves the
/// previous rect clipping later draws, and an ICB execute naming no buffer
/// loses the whole batch that buffer holds.
#[test]
fn a_decoded_record_that_no_arm_applies_names_what_happened_instead() {
    use crate::runtime::drain::store_route_count;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();

    // `MTLScissorRect` is four NSUInteger. Set a real rect, then replace it
    // with an empty one and check the real one is what survives.
    let scissor = |w: u64, h: u64| {
        let total = wire_render::SET_SCISSOR_TOTAL_LEN as usize;
        let mut command = vec![0u8; total];
        let op = wire_render::OPCODE_SET_SCISSOR;
        st32(&mut command[0..], op);
        st32(&mut command[4..], total as u32);
        let p = reims_vgpu_wire::OP_HEADER_LEN;
        st64(&mut command[p..], 7);
        st64(&mut command[p + 8..], 9);
        st64(&mut command[p + 16..], w);
        st64(&mut command[p + 24..], h);
        (op, command)
    };
    let (op, command) = scissor(64, 32);
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
    assert_eq!(
        acc.scissors,
        vec![ScissorRect {
            x: 7,
            y: 9,
            width: 64,
            height: 32
        }]
    );

    let before = store_route_count("render_scissor_empty_kept_previous");
    let (op, command) = scissor(0, 32);
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
    assert_eq!(
        store_route_count("render_scissor_empty_kept_previous") - before,
        1,
        "an empty scissor must name itself"
    );
    assert_eq!(
        acc.scissors,
        vec![ScissorRect {
            x: 7,
            y: 9,
            width: 64,
            height: 32
        }],
        "and behaviour is unchanged: the previous rect is still what is kept"
    );

    // `executeCommandsInBuffer:` naming no buffer.
    let before = store_route_count("render_icb_execute_unnamed");
    let total = wire_render::EXECUTE_COMMANDS_INDIRECT_TOTAL_LEN as usize;
    let mut command = vec![0u8; total];
    let op = wire_render::OPCODE_EXECUTE_COMMANDS_INDIRECT;
    st32(&mut command[0..], op);
    st32(&mut command[4..], total as u32);
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
    assert_eq!(
        store_route_count("render_icb_execute_unnamed") - before,
        1,
        "an ICB execute naming no buffer loses the whole batch and must say so"
    );
    assert!(
        acc.execute_icb.is_empty(),
        "and still does not queue an execute against ref 0"
    );

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(log.contains("reason=render_scissor_empty_kept_previous"));
    assert!(log.contains("reason=render_icb_execute_unnamed"));
}

/// A sampler bind's own LOD clamps reach the slot they were sent for.
///
/// `setVertexSamplerStates:lodMinClamps:lodMaxClamps:withRange:` and its
/// fragment sibling carry a clamp pair **per entry**, so one sampler object
/// bound at two slots can be clamped differently at each — that is the whole
/// reason the pair rides on the bind rather than on the object. The record used
/// to be read for its refs and counted for its clamps, so both slots sampled
/// the object's own range.
///
/// The two-slot form is what this drives, because the one-slot form cannot
/// tell a per-entry pair from a per-record one — the reading the wire module
/// warns about at `SamplerLodBind`.
#[test]
fn a_sampler_bind_carries_its_own_lod_clamps_per_slot() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    let head = OP_HEADER_LEN;

    // Head (first, count) then two 12-byte entries: ref, lodMin, lodMax.
    let entry = render_pass::SAMPLER_LOD_BIND_ENTRY_SIZE;
    let total = head + render_pass::BIND_ENTRIES + 2 * entry;
    let op = wire_render::OPCODE_SET_FRAGMENT_SAMPLER_LOD;
    let mut command = vec![0u8; total];
    st32(&mut command[0..], op);
    st32(&mut command[4..], total as u32);
    st32(&mut command[head + render_pass::BIND_FIRST..], 2);
    st32(&mut command[head + render_pass::BIND_COUNT..], 2);
    let e0 = head + render_pass::BIND_ENTRIES;
    st32(&mut command[e0..], 0x51);
    st32(&mut command[e0 + 4..], 0.25f32.to_bits());
    st32(&mut command[e0 + 8..], 0.75f32.to_bits());
    let e1 = e0 + entry;
    st32(&mut command[e1..], 0x51); // the *same* sampler object
    st32(&mut command[e1 + 4..], 0.5f32.to_bits());
    st32(&mut command[e1 + 8..], 0.875f32.to_bits());
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);

    let binds: Vec<_> = acc.fragment_samplers.as_ref().clone();
    assert_eq!(binds.len(), 2, "both slots bound");
    assert_eq!(
        (binds[0].index, binds[1].index),
        (2, 3),
        "slots first..first+count"
    );
    assert_eq!(
        binds[0].lod_clamp,
        Some((0.25f32.to_bits(), 0.75f32.to_bits()))
    );
    assert_eq!(
        binds[1].lod_clamp,
        Some((0.5f32.to_bits(), 0.875f32.to_bits())),
        "one sampler object, two slots, two clamps — a per-record pair would \
         put slot 2's range here"
    );
    assert!(
        acc.vertex_samplers.as_ref().is_empty(),
        "the fragment opcode must not fill the vertex table"
    );

    // The plain bind carries no clamps, and `None` there is not `(0.0, 0.0)`:
    // it means the sampler object's own range stands.
    let mut acc = StreamAccum::default();
    let total = head + render_pass::BIND_ENTRIES + render_pass::REF_BIND_ENTRY_SIZE;
    let op = wire_render::OPCODE_SET_FRAGMENT_SAMPLER;
    let mut command = vec![0u8; total];
    st32(&mut command[0..], op);
    st32(&mut command[4..], total as u32);
    st32(&mut command[head + render_pass::BIND_FIRST..], 0);
    st32(&mut command[head + render_pass::BIND_COUNT..], 1);
    st32(&mut command[head + render_pass::BIND_ENTRIES..], 0x51);
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
    let binds: Vec<_> = acc.fragment_samplers.as_ref().clone();
    assert_eq!(binds.len(), 1);
    assert_eq!(binds[0].lod_clamp, None);
}

/// An indirect draw reaches the draw list with the counts its buffer holds.
///
/// Both forms used to raise a counter and reach
/// `note_unimplemented_render_opcode`, so the geometry never rendered. The
/// counts are now read from the guest buffer the same way the compute rail's
/// `DispatchThreadgroupsIndirect` already reads its grid — which is what makes
/// the render rail's old "it cannot be executed from the record" a divergence
/// between two arms rather than a property of the protocol.
///
/// Every value in each argument block differs from every other, because they
/// are all 32-bit and a transposition draws a valid primitive of the wrong
/// shape. The indexed block additionally must not put `indexStart` where
/// `first_vertex` goes: that would shift the vertex fetch instead of the index
/// fetch, and both still render.
#[test]
fn an_indirect_draw_takes_its_counts_from_the_guest_buffer() {
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_BUFFER, RESOURCE_PAGE_SHIFT,
    };
    use crate::runtime::gva_mem::{self, write_task_gva_arm64e};

    // A buffer at ref 7 holding `words`, resolvable through the task's
    // own page table — the shape `resolve_indirect_threadgroups_from_buffer`
    // uses on the compute rail.
    let build = |words: &[u32]| {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
        assert!(state.set_object_list(1, 0, 32));
        let bytes: Vec<u8> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buf_gva = 5u64 << RESOURCE_PAGE_SHIFT;
        write_task_gva_arm64e(&mut host, &state.tasks[1], buf_gva, &bytes);
        let mut bdesc = vec![0u8; 16];
        st64(&mut bdesc[0..], bytes.len() as u64);
        st32(&mut bdesc[8..], 5);
        let bdesc_gva = 0x180u64;
        write_task_gva_arm64e(&mut host, &state.tasks[1], bdesc_gva, &bdesc);
        let off = list_object_entry_offset(7, 32).unwrap();
        let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
        st32(&mut le[0..], (OBJECT_TYPE_BUFFER as u32) | (16u32 << 8));
        le[4..12].copy_from_slice(&bdesc_gva.to_le_bytes());
        write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);
        (host, state)
    };

    // --- 0x10, unindexed. Offset first on the wire, then buffer, then type.
    {
        let (host, mut state) = build(&[11, 22, 33, 44]);
        let mut acc = StreamAccum {
            pipeline_ref: 0x41,
            ..Default::default()
        };
        let mut out = ExecResult::default();
        let total = wire_render::DRAW_INDIRECT_TOTAL_LEN as usize;
        let op = wire_render::OPCODE_DRAW_INDIRECT;
        let mut command = vec![0u8; total];
        st32(&mut command[0..], op);
        st32(&mut command[4..], total as u32);
        let p = OP_HEADER_LEN;
        st64(&mut command[p..], 0); // indirect_buffer_offset
        st32(&mut command[p + 8..], 7); // indirect_buffer_ref
        st16(&mut command[p + 12..], 4); // MTLPrimitiveTypeTriangleStrip
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);

        assert_eq!(
            acc.draws.len(),
            1,
            "the draw the guest asked for is recorded"
        );
        assert_eq!(
            acc.draws[0].draw,
            crate::protocol::draw::DrawArgs {
                vertex_count: 11,
                instance_count: 22,
                primitive_type: 4,
                first_vertex: 33,
                base_instance: 44,
            }
        );
        assert!(acc.saw_draw);
        assert!(
            acc.draws[0].indexed.is_none(),
            "an unindexed indirect draw must not carry an index buffer"
        );
    }

    // --- 0x11, indexed. `indexStart` counts indices, so it scales the byte
    // offset by the width `index_type` declares rather than landing raw.
    {
        let (host, mut state) = build(&[11, 22, 33, 44, 55]);
        let mut acc = StreamAccum {
            pipeline_ref: 0x41,
            ..Default::default()
        };
        let mut out = ExecResult::default();
        let total = wire_render::DRAW_INDEXED_INDIRECT_TOTAL_LEN as usize;
        let op = wire_render::OPCODE_DRAW_INDEXED_INDIRECT;
        let mut command = vec![0u8; total];
        st32(&mut command[0..], op);
        st32(&mut command[4..], total as u32);
        let p = OP_HEADER_LEN;
        st16(&mut command[p..], 3); // primitive_type
        st16(&mut command[p + 2..], 1); // MTLIndexTypeUInt32
        st32(&mut command[p + 4..], 0x3e); // index_buffer_ref
        st32(&mut command[p + 8..], 7); // indirect_buffer_ref
        st64(&mut command[p + 12..], 0x100); // index_buffer_offset
        st64(&mut command[p + 20..], 0); // indirect_buffer_offset
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);

        assert_eq!(acc.draws.len(), 1);
        let pd = &acc.draws[0];
        assert_eq!(pd.draw.vertex_count, 11, "indexCount");
        assert_eq!(pd.draw.instance_count, 22);
        assert_eq!(pd.draw.primitive_type, 3, "from the record, not the block");
        assert_eq!(
            pd.draw.first_vertex, 0,
            "indexStart offsets the index buffer, never the vertex fetch"
        );
        assert_eq!(pd.draw.base_instance, 55);
        let idx = pd.indexed.as_ref().expect("the indexed form carries one");
        assert_eq!(idx.index_buffer_ref, 0x3e);
        assert_eq!(idx.index_count, 11);
        assert_eq!(idx.base_vertex, 44, "baseVertex, the block's signed field");
        assert_eq!(
            idx.index_buffer_offset,
            0x100 + 33 * 4,
            "indexStart 33 at four bytes per UInt32 index, past the record's own offset"
        );
    }

    // --- A buffer this device cannot read is a refused draw, not a zero one.
    // There is no count in the record to fall back to.
    {
        let (host, mut state) = build(&[11, 22, 33, 44]);
        let mut acc = StreamAccum {
            pipeline_ref: 0x41,
            ..Default::default()
        };
        let mut out = ExecResult::default();
        let total = wire_render::DRAW_INDIRECT_TOTAL_LEN as usize;
        let op = wire_render::OPCODE_DRAW_INDIRECT;
        let mut command = vec![0u8; total];
        st32(&mut command[0..], op);
        st32(&mut command[4..], total as u32);
        st32(&mut command[OP_HEADER_LEN + 8..], 0x5151); // a ref nothing holds
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
        assert!(
            acc.draws.is_empty(),
            "a draw whose counts could not be read must not be recorded with invented ones"
        );
        assert!(
            !acc.saw_draw,
            "and must not claim the stream has draws in it"
        );
    }
}

/// Every `executeCommandsInBuffer:` in a stream is kept, in stream order.
///
/// This is work, not state. The field behind it was an `Option` assigned with
/// `=`, so the stream's capacity for these was one and a second record
/// overwrote the first — the first ICB's commands never ran, and nothing
/// counted it or logged it. A bound of one with no constant anywhere, which is
/// why none of the five bound scans could see it.
///
/// Both wire forms are driven, because they reach the same field through
/// different payloads: `0x14` names an args buffer whose contents carry the
/// range, `0x15` carries the range literally.
#[test]
fn every_icb_execute_in_a_stream_is_kept_in_order() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();

    // `0x14`: icb_ref @0, indirect_buffer_ref @4, indirect_buffer_offset @8.
    let mut indirect = |icb: u32, args: u32, off: u64| {
        let total = wire_render::EXECUTE_COMMANDS_INDIRECT_TOTAL_LEN as usize;
        let op = wire_render::OPCODE_EXECUTE_COMMANDS_INDIRECT;
        let mut command = vec![0u8; total];
        st32(&mut command[0..], op);
        st32(&mut command[4..], total as u32);
        let p = reims_vgpu_wire::OP_HEADER_LEN;
        st32(&mut command[p..], icb);
        st32(&mut command[p + 4..], args);
        st64(&mut command[p + 8..], off);
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
    };
    indirect(7171, 5151, 0x1111);
    indirect(7172, 5152, 0x2222);

    // `0x15`: icb_ref @0, range_location @4, range_length @12 — unaligned
    // after the ref, which is why it is its own struct rather than the above
    // with a wider tail.
    {
        let total = wire_render::EXECUTE_COMMANDS_RANGE_TOTAL_LEN as usize;
        let op = wire_render::OPCODE_EXECUTE_COMMANDS_RANGE;
        let mut command = vec![0u8; total];
        st32(&mut command[0..], op);
        st32(&mut command[4..], total as u32);
        let p = reims_vgpu_wire::OP_HEADER_LEN;
        st32(&mut command[p..], 7173);
        st64(&mut command[p + 4..], 0x1100);
        st64(&mut command[p + 12..], 0x2200);
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
    }

    let refs: Vec<u32> = acc.execute_icb.iter().map(|e| e.icb_ref).collect();
    assert_eq!(
        refs,
        vec![7171, 7172, 7173],
        "three executes went in and the stream must hold three, in the order \
         the guest wrote them"
    );
    assert!(!acc.execute_icb[0].is_range, "0x14 is the indirect form");
    assert_eq!(acc.execute_icb[0].args_buffer_ref, 5151);
    assert_eq!(acc.execute_icb[0].args_buffer_offset, 0x1111);
    assert_eq!(acc.execute_icb[1].args_buffer_ref, 5152);
    assert!(
        acc.execute_icb[2].is_range,
        "0x15 is the literal-range form"
    );
    assert_eq!(acc.execute_icb[2].range_location, 0x1100);
    assert_eq!(acc.execute_icb[2].range_length, 0x2200);
}

/// The pass opens once, so only the first execute in it may clear.
///
/// Each ICB execute opens its own host pass over the same attachments. Leaving
/// the stream's `CLEAR` on the later ones re-runs the clear inside what Metal
/// treats as one pass, wiping whatever the execute before it drew — the same
/// failure the multi-draw chain describes at `di > 0` and forces `LOAD` for.
#[test]
fn a_later_icb_execute_opens_its_pass_with_load() {
    let slots = vec![
        (
            0u32,
            ColorAttachment {
                texture_ref: 11,
                load_action: MTL_LOAD_ACTION_CLEAR,
                store_action: MTL_STORE_ACTION_STORE,
                clear_color: [0.25, 0.5, 0.75, 1.0],
                ..Default::default()
            },
        ),
        (
            1u32,
            ColorAttachment {
                texture_ref: 12,
                load_action: reims_vgpu_protocol::pass_action::MTL_LOAD_ACTION_DONT_CARE,
                store_action: MTL_STORE_ACTION_STORE,
                clear_color: [0.0; 4],
                // A second attachment whose action is *not* CLEAR, so the
                // helper is shown rewriting every slot rather than only the
                // one that would have re-cleared.
                slice: 3,
                ..Default::default()
            },
        ),
    ];
    let loading = color_slots_loading(&slots);
    assert_eq!(loading.len(), slots.len(), "no attachment is dropped");
    for ((slot, att), (orig_slot, orig)) in loading.iter().zip(slots.iter()) {
        assert_eq!(
            slot, orig_slot,
            "the slot index is the pass's, not an index"
        );
        assert_eq!(att.load_action, MTL_LOAD_ACTION_LOAD);
        // Everything else is the stream's own. The clear colour in particular
        // is carried rather than blanked: it is unread on this path, and a
        // zero here would be an invented value in a decoded record.
        assert_eq!(att.texture_ref, orig.texture_ref);
        assert_eq!(att.store_action, orig.store_action);
        assert_eq!(att.clear_color, orig.clear_color);
        assert_eq!(att.slice, orig.slice);
        assert_eq!(att.level, orig.level);
        assert_eq!(att.depth_plane, orig.depth_plane);
        assert_eq!(att.resolve_texture_ref, orig.resolve_texture_ref);
    }
}

/// A buffer-offset record that lands on nothing says so, both ways.
///
/// `setVertexBufferOffset:atIndex:` is the second record the guest spends on
/// a slot, and both of its miss paths were silent: an index past the table,
/// and an index inside it whose slot this device never bound. The second is
/// the sharper one — Metal requires a live bind at that index and encoder
/// state does not outlive the encoder, so a firing means this device's table
/// and the guest's disagree.
#[test]
fn a_buffer_offset_that_lands_on_nothing_reports_which_way_it_missed() {
    use crate::runtime::drain::store_route_count;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();

    // index:u32 @0, offset:u64 @4 — a different payload shape from the plural
    // binds, which is why it takes its own offsets rather than `BIND_*`.
    let offset_record = |index: u32| {
        let total = reims_vgpu_wire::OP_HEADER_LEN + render_pass::BUFFER_OFFSET_PAYLOAD_LEN;
        let mut command = vec![0u8; total];
        let op = wire_render::OPCODE_SET_VERTEX_BUFFER_OFFSET;
        st32(&mut command[0..], op);
        st32(&mut command[4..], total as u32);
        st32(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN + render_pass::BUFFER_OFFSET_INDEX..],
            index,
        );
        st64(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN + render_pass::BUFFER_OFFSET_VALUE..],
            0x5555,
        );
        (op, command)
    };

    // Inside the table, but nothing is bound there.
    let before_unbound = store_route_count("render_buffer_offset_slot_unbound");
    let (op, command) = offset_record(3);
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
    assert_eq!(
        store_route_count("render_buffer_offset_slot_unbound") - before_unbound,
        1,
        "an offset for a slot this device never bound must be named"
    );

    // Past the table entirely.
    let before_past = store_route_count("render_buffer_offset_slot_past_table");
    let (op, command) = offset_record(MAX_BUFFER_BIND_SLOTS + 4);
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
    assert_eq!(
        store_route_count("render_buffer_offset_slot_past_table") - before_past,
        1,
        "an offset past the table bound must be named separately"
    );
    assert_eq!(
        store_route_count("render_buffer_offset_slot_unbound") - before_unbound,
        1,
        "a slot past the table is not also an unbound slot inside it"
    );
}

/// The three ICB blit records are told apart rather than refused as one.
///
/// They used to be declined before decode under a single shared reason,
/// which said three different things with one word. Only two of them are
/// losses: skipping Metal's optimize hint is semantically correct, while a
/// dropped reset leaves commands live that the guest retired and a dropped
/// copy leaves the destination holding what it held before. A counter that
/// cannot tell those apart cannot answer the question they exist to answer.
#[test]
fn each_icb_blit_record_reaches_a_counter_that_names_which_one_it_is() {
    use crate::protocol::endian::st64;
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_wire::ops::blit as wire;

    let range = |op: u32| {
        let total = wire::ICB_RANGE_TOTAL_LEN as usize;
        let mut v = vec![0u8; total];
        st32(&mut v[0..], op);
        st32(&mut v[4..], total as u32);
        st32(&mut v[reims_vgpu_wire::OP_HEADER_LEN..], 6161);
        st64(&mut v[reims_vgpu_wire::OP_HEADER_LEN + 4..], 0x3300);
        st64(&mut v[reims_vgpu_wire::OP_HEADER_LEN + 12..], 0x4400);
        v
    };
    let copy = || {
        let total = wire::COPY_ICB_TOTAL_LEN as usize;
        let mut v = vec![0u8; total];
        st32(&mut v[0..], wire_blit::OPCODE_COPY_ICB);
        st32(&mut v[4..], total as u32);
        st32(&mut v[reims_vgpu_wire::OP_HEADER_LEN..], 7171);
        st32(&mut v[reims_vgpu_wire::OP_HEADER_LEN + 4..], 7272);
        st64(&mut v[reims_vgpu_wire::OP_HEADER_LEN + 8..], 0x1100);
        st64(&mut v[reims_vgpu_wire::OP_HEADER_LEN + 16..], 0x2200);
        st64(&mut v[reims_vgpu_wire::OP_HEADER_LEN + 24..], 0x3300);
        v
    };

    for (op, command, route) in [
        (
            wire_blit::OPCODE_OPTIMIZE_ICB,
            range(wire_blit::OPCODE_OPTIMIZE_ICB),
            "blit_noop_icb_optimize",
        ),
        (
            wire_blit::OPCODE_RESET_ICB,
            range(wire_blit::OPCODE_RESET_ICB),
            "blit_icb_reset_dropped",
        ),
        (wire_blit::OPCODE_COPY_ICB, copy(), "blit_icb_copy_dropped"),
    ] {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let before = store_route_count(route);
        handle_blit_record(&mut state, &mut host, 1, op, &command);
        assert_eq!(
            store_route_count(route),
            before + 1,
            "op {op:#x} did not reach {route}"
        );
    }

    // The optimize hint is the one that is *not* a loss, so it must not
    // reach either of the dropped-work counters. Sharing one would put a
    // correct no-op in the same bucket as stale commands executing.
    for route in ["blit_icb_reset_dropped", "blit_icb_copy_dropped"] {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let before = store_route_count(route);
        let command = range(wire_blit::OPCODE_OPTIMIZE_ICB);
        handle_blit_record(
            &mut state,
            &mut host,
            1,
            wire_blit::OPCODE_OPTIMIZE_ICB,
            &command,
        );
        assert_eq!(
            store_route_count(route),
            before,
            "the optimize hint was counted as {route}"
        );
    }
}

/// The five `BlitEncoderSPI` records each reach a route that names them.
///
/// All five answered `blit_decode_unknown_opcode` until the wire capture
/// drove this class with the capability forced on, and three of them are
/// writes to guest-visible memory. The routes are not interchangeable and
/// the test says so in both directions: the two texture fills are lost work
/// and must not land on the invalidate's no-op counter, while the
/// compressed-texture invalidate is a correct skip and must not land on
/// either dropped-fill counter. Sharing one bucket would make a driven
/// boot's reading unusable for deciding which executor to build.
#[test]
fn each_blit_spi_record_reaches_a_counter_that_names_which_one_it_is() {
    use crate::protocol::endian::st64;
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_wire::ops::blit as wire;

    // A texture fill of either form: identical through the region, then the
    // tail that tells the two apart. Zero-filled past that, which is what
    // the guest's staged-bytes fill of length 0 would look like — the
    // routing under test is by opcode, not by any value in the tail.
    let texture_fill = |op: u32, total: u32| {
        let mut v = vec![0u8; total as usize];
        st32(&mut v[0..], op);
        st32(&mut v[4..], total);
        let p = reims_vgpu_wire::OP_HEADER_LEN;
        st32(&mut v[p..], 4242); // texture
        st16(&mut v[p + 4..], 3); // level
        st16(&mut v[p + 6..], 5); // slice
        st64(&mut v[p + 8..], 0x44); // size w/h/d
        st64(&mut v[p + 16..], 0x55);
        st64(&mut v[p + 24..], 1);
        st64(&mut v[p + 32..], 0x11); // origin x/y/z
        st64(&mut v[p + 40..], 0x22);
        st64(&mut v[p + 48..], 0x33);
        v
    };
    let invalidate = |op: u32, total: u32| {
        let mut v = vec![0u8; total as usize];
        st32(&mut v[0..], op);
        st32(&mut v[4..], total);
        st32(&mut v[reims_vgpu_wire::OP_HEADER_LEN..], 4242);
        v
    };

    const COLOR: &str = "blit_fill_texture_color_dropped";
    const BYTES: &str = "blit_fill_texture_bytes_dropped";
    const INVALID: &str = "blit_noop_invalidate_compressed";

    for (op, command, route) in [
        (
            wire_blit::OPCODE_FILL_TEXTURE_COLOR,
            texture_fill(
                wire_blit::OPCODE_FILL_TEXTURE_COLOR,
                wire::FILL_TEXTURE_COLOR_TOTAL_LEN,
            ),
            COLOR,
        ),
        (
            wire_blit::OPCODE_FILL_TEXTURE_BYTES,
            texture_fill(
                wire_blit::OPCODE_FILL_TEXTURE_BYTES,
                wire::FILL_TEXTURE_BYTES_TOTAL_LEN,
            ),
            BYTES,
        ),
        (
            wire_blit::OPCODE_INVALIDATE_COMPRESSED_TEXTURE,
            invalidate(
                wire_blit::OPCODE_INVALIDATE_COMPRESSED_TEXTURE,
                wire::REF_TOTAL_LEN,
            ),
            INVALID,
        ),
        (
            wire_blit::OPCODE_INVALIDATE_COMPRESSED_TEXTURE_SLICE_LEVEL,
            invalidate(
                wire_blit::OPCODE_INVALIDATE_COMPRESSED_TEXTURE_SLICE_LEVEL,
                wire::REF_SLICE_LEVEL_TOTAL_LEN,
            ),
            INVALID,
        ),
    ] {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let others: Vec<(&str, u64)> = [COLOR, BYTES, INVALID]
            .into_iter()
            .filter(|r| *r != route)
            .map(|r| (r, store_route_count(r)))
            .collect();
        let before = store_route_count(route);
        handle_blit_record(&mut state, &mut host, 1, op, &command);
        assert_eq!(
            store_route_count(route),
            before + 1,
            "op {op:#x} did not reach {route}"
        );
        for (other, was) in others {
            assert_eq!(
                store_route_count(other),
                was,
                "op {op:#x} also reached {other}; the two losses are not the \
                 same loss and one counter cannot answer for both"
            );
        }
    }

    // The pattern fill is the one of the five that is *executed*, so it
    // must not appear on any of the three counters above. It fails on a
    // missing buffer here, which is the executor running rather than the
    // record being dropped.
    let mut v = vec![0u8; wire::FILL_BUFFER_PATTERN4_TOTAL_LEN as usize];
    st32(&mut v[0..], wire_blit::OPCODE_FILL_BUFFER_PATTERN4);
    st32(&mut v[4..], wire::FILL_BUFFER_PATTERN4_TOTAL_LEN);
    st32(&mut v[reims_vgpu_wire::OP_HEADER_LEN..], 7);
    st64(&mut v[reims_vgpu_wire::OP_HEADER_LEN + 4..], 0);
    st64(&mut v[reims_vgpu_wire::OP_HEADER_LEN + 12..], 8);
    st32(&mut v[reims_vgpu_wire::OP_HEADER_LEN + 20..], 0x89ab_cdef);
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let before: Vec<u64> = [COLOR, BYTES, INVALID]
        .into_iter()
        .map(store_route_count)
        .collect();
    handle_blit_record(
        &mut state,
        &mut host,
        1,
        wire_blit::OPCODE_FILL_BUFFER_PATTERN4,
        &v,
    );
    for (route, was) in [COLOR, BYTES, INVALID].into_iter().zip(before) {
        assert_eq!(
            store_route_count(route),
            was,
            "the executed pattern fill was counted as {route}"
        );
    }
}

/// A strided vertex bind reaches the bind table carrying its stride.
///
/// Three claims, and they failed in three different eras. The record used to
/// be refused before decode, so the buffer never bound at all. Then the bind
/// landed and the stride was stepped over and counted. Now the stride travels
/// on the bind, so the assertion is that the *third* field of the twenty-byte
/// entry is the guest's number and not padding.
#[test]
fn a_strided_vertex_bind_lands_in_the_table_carrying_its_stride() {
    use crate::protocol::endian::st64;

    let total = reims_vgpu_wire::OP_HEADER_LEN
        + render_pass::BIND_ENTRIES
        + render_pass::BUFFER_STRIDE_BIND_ENTRY_SIZE;
    let mut command = vec![0u8; total];
    st32(
        &mut command[0..],
        wire_render::OPCODE_SET_VERTEX_BUFFER_STRIDE,
    );
    st32(&mut command[4..], total as u32);
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_FIRST..],
        4,
    );
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_COUNT..],
        1,
    );
    let e = reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_ENTRIES;
    st32(&mut command[e..], 5151);
    st64(&mut command[e + 4..], 0x2345);
    st64(&mut command[e + 12..], 0x3456);

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    handle_render_record(
        &mut state,
        &host,
        1,
        wire_render::OPCODE_SET_VERTEX_BUFFER_STRIDE,
        &command,
        &mut out,
        &mut acc,
    );
    assert_eq!(
        acc.vertex_buffers.len(),
        1,
        "the buffer did not bind; this record used to be refused whole"
    );
    let b = &acc.vertex_buffers[0];
    assert_eq!((b.index, b.buffer_ref, b.offset), (4, 5151, 0x2345));
    assert_eq!(
        b.attribute_stride,
        Some(0x3456),
        "the stride is the entry's third field, not padding stepped over"
    );
    assert!(
        acc.fragment_buffers.is_empty(),
        "a vertex bind reached the fragment table"
    );

    // The plain bind carries no stride table, and `None` is not `Some(0)`: a
    // zero stride is a legal Metal request that fetches every vertex from one
    // address, so the two cannot share a spelling.
    let plain_total = reims_vgpu_wire::OP_HEADER_LEN
        + render_pass::BIND_ENTRIES
        + render_pass::BUFFER_BIND_ENTRY_SIZE;
    let mut plain = vec![0u8; plain_total];
    st32(&mut plain[0..], wire_render::OPCODE_SET_VERTEX_BUFFER);
    st32(&mut plain[4..], plain_total as u32);
    st32(
        &mut plain[reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_COUNT..],
        1,
    );
    st32(
        &mut plain[reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_ENTRIES..],
        5151,
    );
    handle_render_record(
        &mut state,
        &host,
        1,
        wire_render::OPCODE_SET_VERTEX_BUFFER,
        &plain,
        &mut out,
        &mut acc,
    );
    // By index, not by position: the strided bind above took slot 4 and is
    // still in the table, so `[0]` is whichever landed first rather than the
    // one this record wrote.
    let plain_bind = acc
        .vertex_buffers
        .iter()
        .find(|b| b.index == 0)
        .expect("the plain bind landed at slot 0");
    assert_eq!(
        plain_bind.attribute_stride, None,
        "a plain vertex bind reported a stride it does not carry"
    );
}

/// Every state this rail decodes and does not apply reaches its own counter,
/// and the ones with an API default stay quiet when the guest asks for it.
///
/// The counters are the whole reason those opcodes are decoded: each is the
/// measured argument for whether implementing that state is worth building,
/// and a counter nobody reads back cannot be shown to be wired up. Nine of
/// them had no such test until now.
///
/// The tessellated patch draws are deliberately not in the default half. They
/// have no default to be at -- a patch draw is geometry the guest asked for,
/// so every one is a loss and is counted unconditionally.
///
/// The two *indirect* draws used to be here on the same reading and are not
/// any more: both now read their counts out of the guest buffer and reach the
/// draw list, which is what
/// [`an_indirect_draw_takes_its_counts_from_the_guest_buffer`] holds.
///
/// `setTriangleFillMode:` and `setDepthClipMode:` used to be the first two
/// rows here and are not any more: both now reach a backend, and
/// [`a_fill_mode_and_a_depth_clip_mode_reach_the_stream_state`] is what
/// replaced their rows.
#[test]
fn every_decoded_but_unapplied_render_state_reaches_its_own_counter() {
    use crate::protocol::endian::st64;
    use crate::runtime::drain::store_route_count;

    // (opcode, total length, payload writer, route, whether a default-valued
    // record of the same opcode must NOT count).
    type Writer = fn(&mut [u8]);
    // No `at_default` writer for the mode-shaped records any more: the only
    // two whose default half this exercised were the fill mode and the depth
    // clip mode, and both now reach a backend. The float pair below still has
    // one.
    let float_non_default: Writer = |p| st32(p, 2.5f32.to_bits());
    let float_at_default: Writer = |p| st32(p, 1.0f32.to_bits());

    // `OPCODE_SET_LINE_WIDTH` is no longer here: it shares the float record's
    // decode arm with the tessellation scale below, but it is now latched into
    // the stream's state and carried to the running rail, so it drops nothing
    // and counts nothing. See
    // `a_line_width_reaches_the_stream_state_and_a_draw`.
    let cases: &[(u32, usize, Writer, Option<Writer>, &str)] = &[
        (
            wire_render::OPCODE_SET_TESSELLATION_FACTOR_SCALE,
            12,
            float_non_default,
            Some(float_at_default),
            "render_tessellation_scale_dropped",
        ),
        (
            wire_render::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT,
            reims_vgpu_wire::OP_HEADER_LEN
                + render_pass::AMPLIFICATION_COUNT_LEN
                + 2 * render_pass::AMPLIFICATION_MAPPING_SIZE,
            // Two views. One is Metal's default and means no amplification,
            // so the default arm below asks for one and must not count.
            |p| st32(p, 2),
            Some(|p| st32(p, 1)),
            "render_vertex_amplification_dropped",
        ),
        (
            wire_render::OPCODE_SET_VERTEX_AMPLIFICATION_MODE,
            reims_vgpu_wire::OP_HEADER_LEN + 8,
            |p| {
                st32(p, 0x5555);
                st32(&mut p[4..], 0x6666);
            },
            Some(|p| {
                st32(p, 0);
                st32(&mut p[4..], 0);
            }),
            "render_vertex_amplification_dropped",
        ),
        // The tile family. The four bind opcodes each get a one-slot record
        // at their own entry stride, so a route that fired from the wrong
        // arm would have to have accepted the wrong length first.
        (
            wire_tile::OPCODE_SET_TILE_BUFFER,
            reims_vgpu_wire::OP_HEADER_LEN
                + render_pass::BIND_ENTRIES
                + render_pass::BUFFER_BIND_ENTRY_SIZE,
            |p| {
                st32(&mut p[render_pass::BIND_FIRST..], 3);
                st32(&mut p[render_pass::BIND_COUNT..], 1);
            },
            None,
            "render_tile_buffer_bind_dropped",
        ),
        (
            wire_tile::OPCODE_SET_TILE_BUFFER_OFFSET,
            20,
            |p| {
                st32(p, 4);
                st64(&mut p[4..], 0x2345);
            },
            None,
            "render_tile_buffer_bind_dropped",
        ),
        (
            wire_tile::OPCODE_SET_TILE_TEXTURE,
            reims_vgpu_wire::OP_HEADER_LEN
                + render_pass::BIND_ENTRIES
                + render_pass::REF_BIND_ENTRY_SIZE,
            |p| {
                st32(&mut p[render_pass::BIND_FIRST..], 2);
                st32(&mut p[render_pass::BIND_COUNT..], 1);
            },
            None,
            "render_tile_texture_bind_dropped",
        ),
        (
            wire_tile::OPCODE_SET_TILE_SAMPLER,
            reims_vgpu_wire::OP_HEADER_LEN
                + render_pass::BIND_ENTRIES
                + render_pass::REF_BIND_ENTRY_SIZE,
            |p| {
                st32(&mut p[render_pass::BIND_FIRST..], 4);
                st32(&mut p[render_pass::BIND_COUNT..], 1);
            },
            None,
            "render_tile_sampler_bind_dropped",
        ),
        (
            wire_tile::OPCODE_SET_TILE_SAMPLER_LOD,
            reims_vgpu_wire::OP_HEADER_LEN
                + render_pass::BIND_ENTRIES
                + render_pass::SAMPLER_LOD_BIND_ENTRY_SIZE,
            |p| {
                st32(&mut p[render_pass::BIND_FIRST..], 5);
                st32(&mut p[render_pass::BIND_COUNT..], 1);
            },
            None,
            "render_tile_sampler_bind_dropped",
        ),
        // The three dispatches. Their default arm is a grid with a zero
        // dimension, which Metal dispatches nothing for -- dropping one
        // loses no work, so counting it would inflate the very number this
        // counter exists to be.
        (
            wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE,
            32,
            |p| {
                st64(p, 0x11);
                st64(&mut p[8..], 0x22);
                st64(&mut p[16..], 0x33);
            },
            Some(|p| {
                st64(p, 0x11);
                st64(&mut p[8..], 0x22);
                st64(&mut p[16..], 0);
            }),
            "render_tile_dispatch_dropped",
        ),
        (
            wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION,
            84,
            |p| {
                st64(p, 0x11);
                st64(&mut p[8..], 0x22);
                st64(&mut p[16..], 0x33);
            },
            Some(|p| st64(&mut p[16..], 0)),
            "render_tile_dispatch_dropped",
        ),
        (
            wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX,
            84,
            |p| {
                st64(p, 0x11);
                st64(&mut p[8..], 0x22);
                st64(&mut p[16..], 0x33);
            },
            Some(|p| st64(&mut p[16..], 0)),
            "render_tile_dispatch_dropped",
        ),
        // Not a dropped command but an unanswered question, so it has no
        // default arm: every one of these leaves the guest reading its own
        // stale ring as a tile geometry.
        (
            wire_tile::OPCODE_GET_TILE_DIMENSIONS,
            20,
            |p| {
                st32(p, 5151);
                st64(&mut p[4..], 0x9999);
            },
            None,
            "render_tile_dimensions_unanswered",
        ),
        (
            wire_tile::OPCODE_SET_TILE_THREADGROUP_MEMORY,
            28,
            |p| {
                st64(p, 0x1234);
                st64(&mut p[8..], 0x2345);
                st32(&mut p[16..], 5);
            },
            None,
            "render_tile_threadgroup_memory_dropped",
        ),
        // The store-action options. Their record is four bytes longer on
        // the colour form than on the other two, so a route reached from
        // the wrong arm would have had to accept the wrong length first. The
        // default arm is `MTLStoreActionOptionNone`, which asks for nothing and
        // must not count.
        (
            wire_render::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS,
            20,
            |p| {
                st64(p, 0x1111);
                st32(&mut p[8..], 3);
            },
            Some(|p| {
                st64(p, 0);
                st32(&mut p[8..], 3);
            }),
            "render_store_action_options_dropped",
        ),
        (
            wire_render::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS,
            16,
            |p| st64(p, 0x2222),
            Some(|p| st64(p, 0)),
            "render_store_action_options_dropped",
        ),
        (
            wire_render::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS,
            16,
            |p| st64(p, 0x3333),
            Some(|p| st64(p, 0)),
            "render_store_action_options_dropped",
        ),
        (
            wire_render::OPCODE_SET_TESSELLATION_FACTOR_BUFFER,
            28,
            |p| {
                st32(p, 5151);
                st64(&mut p[4..], 0x3456);
                st64(&mut p[12..], 0x4567);
            },
            None,
            "render_tessellation_factor_buffer_dropped",
        ),
        // The patch draws. The two `0x0c` rows are the point: one opcode,
        // two lengths, and both must reach the counter -- a length-based
        // dispatch that refused one of them would read as a healthy zero.
        (
            wire_render::OPCODE_DRAW_PATCHES,
            24,
            |_p| {},
            None,
            "render_draw_patches_dropped",
        ),
        (
            wire_render::OPCODE_DRAW_PATCHES_WIDE,
            56,
            |_p| {},
            None,
            "render_draw_patches_dropped",
        ),
        (
            wire_render::OPCODE_DRAW_PATCHES_WIDE,
            68,
            |_p| {},
            None,
            "render_draw_patches_dropped",
        ),
        (
            wire_render::OPCODE_DRAW_INDEXED_PATCHES,
            32,
            |_p| {},
            None,
            "render_draw_patches_dropped",
        ),
        (
            wire_render::OPCODE_DRAW_PATCHES_INDIRECT,
            36,
            |_p| {},
            None,
            "render_draw_patches_indirect_dropped",
        ),
        (
            wire_render::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT,
            48,
            |_p| {},
            None,
            "render_draw_patches_indirect_dropped",
        ),
    ];

    let run = |op: u32, total: usize, write: Writer| {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        let mut command = vec![0u8; total];
        st32(&mut command[0..], op);
        st32(&mut command[4..], total as u32);
        write(&mut command[reims_vgpu_wire::OP_HEADER_LEN..]);
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
    };

    for (op, total, write, default_write, route) in cases {
        let before = store_route_count(route);
        run(*op, *total, *write);
        assert_eq!(
            store_route_count(route),
            before + 1,
            "op {op:#x} did not reach {route}"
        );
        if let Some(default_write) = default_write {
            let before = store_route_count(route);
            run(*op, *total, *default_write);
            assert_eq!(
                store_route_count(route),
                before,
                "op {op:#x} counted a guest asking for the API default, which \
                 is what this rail already does -- that turns the healthy \
                 zero back into a flood"
            );
        }
    }
}

/// `setTriangleFillMode:` and `setDepthClipMode:` land in the stream's state
/// and travel to a draw, ordinal for ordinal.
///
/// Both records share one 16-byte wire form and one decode arm, so the opcode
/// is the only thing that says which state a record sets — swapping the two
/// arms compiles, renders, and wireframes a pass that asked to be clamped.
/// Each is therefore driven on its own and the *other* slot asserted still
/// unset.
///
/// The default value is latched too. A stream that sets Lines and then sets
/// Fill again is asking for Fill, so an arm that skipped `mode == 0` — which
/// is what the counter these replaced did, and correctly, since a counter is
/// only interested in the non-default — would leave the rest of the pass
/// wireframed.
#[test]
fn a_fill_mode_and_a_depth_clip_mode_reach_the_stream_state() {
    use crate::protocol::endian::st64;

    let drive = |op: u32, mode: u64| {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        let mut command = vec![0u8; wire_render::SET_MODE_TOTAL_LEN as usize];
        st32(&mut command[0..], op);
        st32(&mut command[4..], wire_render::SET_MODE_TOTAL_LEN);
        st64(&mut command[reims_vgpu_wire::OP_HEADER_LEN..], mode);
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
        acc
    };

    for mode in [0u64, 1] {
        let acc = drive(wire_render::OPCODE_SET_TRIANGLE_FILL_MODE, mode);
        assert_eq!(acc.fill_mode, Some(mode as u32), "fill mode {mode}");
        assert_eq!(
            acc.depth_clip_mode, None,
            "fill mode {mode} set the sibling"
        );
        let acc = drive(wire_render::OPCODE_SET_DEPTH_CLIP_MODE, mode);
        assert_eq!(acc.depth_clip_mode, Some(mode as u32), "depth clip {mode}");
        assert_eq!(acc.fill_mode, None, "depth clip {mode} set the sibling");
    }

    // The record's field is 64 bits wide and the backend takes 32. A word
    // whose low half is zero must not arrive as the Metal default, which
    // would render silently; it arrives as a value no `MTLTriangleFillMode`
    // has, and the backend names it.
    let acc = drive(wire_render::OPCODE_SET_TRIANGLE_FILL_MODE, 1u64 << 32);
    assert_eq!(acc.fill_mode, Some(u32::MAX));

    // And a draw carries them. `bind_snapshot` builds its `PendingDraw` with
    // `..Default::default()`, so a field added to the accumulator and not to
    // the snapshot reaches no draw at all and nothing else would say so.
    let mut acc = drive(wire_render::OPCODE_SET_TRIANGLE_FILL_MODE, 1);
    acc.depth_clip_mode = Some(1);
    let pd = acc.bind_snapshot().expect("state is representable");
    assert_eq!(pd.fill_mode, Some(1));
    assert_eq!(pd.depth_clip_mode, Some(1));
    let mut req = crate::runtime::draw::DrawEncodeRequest::default();
    fill_draw_binds_from_pending(&mut req, &pd);
    assert_eq!(req.fill_mode, Some(1));
    assert_eq!(req.depth_clip_mode, Some(1));
}

/// `setLineWidth:` lands in the stream's state and travels to a draw, bit for
/// bit.
///
/// The fifth encoder raster state, and it used to be dropped with a count
/// whenever it was not 1.0. It shares its decode arm and its wire form with
/// `setTessellationFactorScale:`, which still has no carrier — so the opcode
/// is the only thing separating a state that is now honoured from one that is
/// still lost, and an arm that latched the wrong one would compile and would
/// widen every line by the tessellation scale.
///
/// The default is latched too, for the reason the fill mode's is: a stream
/// that widens a line and then narrows it again is asking for the narrow one,
/// and an arm that skipped 1.0 would leave the rest of the pass thick.
#[test]
fn a_line_width_reaches_the_stream_state_and_a_draw() {
    let drive = |op: u32, width: f32| {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        let mut command = vec![0u8; 12];
        st32(&mut command[0..], op);
        st32(&mut command[4..], 12);
        st32(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN..],
            width.to_bits(),
        );
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
        acc
    };

    for width in [1.0f32, 2.5, 0.5] {
        let acc = drive(wire_render::OPCODE_SET_LINE_WIDTH, width);
        assert_eq!(
            acc.line_width.map(f32::to_bits),
            Some(width.to_bits()),
            "line width {width}"
        );
    }

    // The sibling opcode must not reach the same slot. Both are one `f32` in
    // one twelve-byte record; only the opcode tells them apart.
    let acc = drive(wire_render::OPCODE_SET_TESSELLATION_FACTOR_SCALE, 2.5);
    assert_eq!(acc.line_width, None);

    // And a draw carries it — `bind_snapshot` builds with
    // `..Default::default()`, so a field added to the accumulator and not to
    // the snapshot reaches no draw at all.
    let acc = drive(wire_render::OPCODE_SET_LINE_WIDTH, 2.5);
    let pd = acc.bind_snapshot().expect("state is representable");
    assert_eq!(pd.line_width, Some(2.5));
    let mut req = crate::runtime::draw::DrawEncodeRequest::default();
    fill_draw_binds_from_pending(&mut req, &pd);
    assert_eq!(req.line_width, Some(2.5));
}

/// Every command buffer the submission declares is visited, however many
/// there are.
///
/// A fixed ceiling used to truncate the table with `.min()` before the loop
/// started, so a guest submitting more than it dropped the remainder whole —
/// no encode, no refusal, no line. That is the worst shape a loss can take:
/// the report comes back well-formed and the missing draws are simply not in
/// it, so nothing downstream can tell a short submission from a truncated one.
///
/// Nothing derived the ceiling. The payload-length check above the loop
/// already bounds the count — the guest cannot declare a table longer than the
/// descriptors it supplied — so the ceiling only ever cut submissions that
/// were entirely well-formed.
///
/// The probe is the per-descriptor `len=0` skip line, because it fires from
/// inside the loop body: one line per descriptor actually reached. Counting
/// them measures how far the loop went, which is exactly what the ceiling
/// changed. The count is deliberately above any round number a re-introduced
/// ceiling would pick.
#[test]
fn every_declared_command_buffer_is_visited_not_just_the_first_sixteen() {
    const N_CB: u32 = 33;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let host = FakeHost::new();
    state.define_task(3, 0x1_0000, 2);

    let mut payload = vec![
        0u8;
        CHILD_EXEC_INDIRECT_HEADER_LEN as usize
            + N_CB as usize * CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN as usize
    ];
    st32(&mut payload[CHILD_EXEC_INDIRECT_TASK_ID as usize..], 3);
    st32(
        &mut payload[CHILD_EXEC_INDIRECT_CMDBUF_COUNT as usize..],
        N_CB,
    );
    // Distinct per-descriptor GVAs so the lines cannot be confused for one
    // descriptor reported repeatedly. Length stays zero: this test is about
    // which descriptors are reached, not what loading one does.
    for i in 0..N_CB as usize {
        let off = CHILD_EXEC_INDIRECT_HEADER_LEN as usize
            + i * CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN as usize;
        st64(
            &mut payload[off + CHILD_EXEC_INDIRECT_CMDBUF_GVA as usize..],
            0x1_0000 + i as u64,
        );
    }

    let cap = crate::observe::sink::FailCapture::start();
    let (_submission, r) = read_exec_submission(&state, &host, &payload);
    assert_eq!(r.task_id, 3);
    let visited: Vec<String> = cap
        .lines()
        .into_iter()
        .filter(|l| l.split_whitespace().next() == Some("exec_cmdbuf"))
        .collect();
    assert_eq!(
        visited.len(),
        N_CB as usize,
        "the loop stopped short of the declared table: {visited:?}"
    );
    // Name the last one explicitly, so a future truncation that happens to
    // keep the count (by reporting something else per descriptor) still fails.
    assert!(
        visited
            .iter()
            .any(|l| l.contains(&format!("i={}", N_CB - 1))),
        "the final declared command buffer was never reached: {visited:?}"
    );
}

/// The command-stream reader returns the buffers it could read and drops the
/// ones it could not, so its length is the loaded count and not the declared
/// one.
///
/// The two are different numbers and the difference is guest-visible draws:
/// a descriptor the reader skips is a command buffer whose records never run,
/// and a caller that took `cmdbuf_count` for the answer would report a full
/// submission. Every other test of this loop drives it through
/// `process_exec_indirect2` with nothing readable behind the GVAs, so the
/// **successful** read is what this adds.
#[test]
fn the_command_stream_reader_returns_what_it_read_and_not_what_was_declared() {
    use crate::model::PAGE_SHIFT_ARM64E;
    use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();

    // A one-level page table for task 3: GVA page 0 resolves to data pfn 4.
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
    state.define_task(3, 0x1_0000, 2);
    let _ = host.write_gpa(data_gpa, &[0xa5u8; 16]);

    // Three descriptors: one readable, one declaring zero bytes, one whose GVA
    // does not walk. The first is the case no other test reaches.
    const N_CB: u32 = 3;
    let mut payload = vec![
        0u8;
        CHILD_EXEC_INDIRECT_HEADER_LEN as usize
            + N_CB as usize * CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN as usize
    ];
    st32(&mut payload[CHILD_EXEC_INDIRECT_TASK_ID as usize..], 3);
    st32(
        &mut payload[CHILD_EXEC_INDIRECT_CMDBUF_COUNT as usize..],
        N_CB,
    );
    let desc = |i: usize| {
        CHILD_EXEC_INDIRECT_HEADER_LEN as usize + i * CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN as usize
    };
    st64(
        &mut payload[desc(0) + CHILD_EXEC_INDIRECT_CMDBUF_GVA as usize..],
        0,
    );
    st64(
        &mut payload[desc(0) + CHILD_EXEC_INDIRECT_CMDBUF_LENGTH as usize..],
        16,
    );
    // Descriptor 1 keeps length 0.
    st64(
        &mut payload[desc(2) + CHILD_EXEC_INDIRECT_CMDBUF_GVA as usize..],
        0xdead_0000,
    );
    st64(
        &mut payload[desc(2) + CHILD_EXEC_INDIRECT_CMDBUF_LENGTH as usize..],
        32,
    );

    let cap = crate::observe::sink::FailCapture::start();
    let streams = super::load_command_streams(
        &state,
        &host,
        3,
        &payload,
        u64::from(CHILD_EXEC_INDIRECT_HEADER_LEN),
        N_CB,
    );
    assert_eq!(
        streams.len(),
        1,
        "three declared, one readable — the result is the loaded count"
    );
    assert_eq!(
        streams[0],
        vec![0xa5u8; 16],
        "and it is the guest's bytes, at the length the descriptor declared"
    );
    // Each loss says which it was, rather than one merged line: a zero-length
    // descriptor and a GVA that does not walk are different guest problems.
    let lines: Vec<String> = cap
        .lines()
        .into_iter()
        .filter(|l| l.split_whitespace().next() == Some("exec_cmdbuf"))
        .collect();
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert!(lines.iter().any(|l| l.contains("len=0")), "{lines:?}");
    assert!(lines.iter().any(|l| l.contains("gva_fail")), "{lines:?}");
}

/// A bind the stream's tables could not hold refuses the draws that read it,
/// and leaves the ones recorded before it alone.
///
/// The walk has always stopped at the first slot past its class's argument
/// table and said so. What followed was the bug: the six tables kept the state
/// they could represent, and every later draw in the stream was encoded against
/// it — a frame computed from state the guest never asked for, with nothing
/// downstream able to notice, because a shader that does not sample the missing
/// slot is indistinguishable from one whose bind landed.
///
/// [`crate::runtime::draw::first_bind_past_table`] cannot cover this and says
/// so: it reads the six tables of a built request, and the refused bind is
/// exactly the one that never entered them.
///
/// Driven through `handle_render_record` for all three records rather than by
/// setting the field, because the field is bookkeeping and the refusal is the
/// behaviour. The three assertions that matter are separate on purpose — a
/// refusal that also dropped the earlier draw would pass a test that only
/// counted the later one.
#[test]
fn a_bind_past_the_table_refuses_the_draws_that_would_read_it() {
    use crate::runtime::drain::store_route_count;

    /// Past Apple's buffer table and past this device's, which are the same
    /// number — see the `const` assertion beside [`BindClass`].
    const FIRST: u32 = MAX_BUFFER_BIND_SLOTS + 4;

    let draw = |acc: &mut StreamAccum| {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut command = vec![0u8; 0x20];
        let op = wire_render::OPCODE_DRAW_INDEXED_WIDE;
        st32(&mut command[0..], op);
        st32(&mut command[4..], 0x20);
        st16(&mut command[8..], 3);
        st32(&mut command[12..], 0x3e);
        st32(&mut command[16..], 6);
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, acc);
    };

    let mut acc = StreamAccum {
        pipeline_ref: 61,
        ..Default::default()
    };
    draw(&mut acc);
    assert_eq!(
        acc.draws.len(),
        1,
        "a draw with complete bind state is recorded"
    );
    assert!(acc.bind_snapshot().is_ok());

    // One buffer bind whose whole run sits past the table.
    let entry = render_pass::BUFFER_BIND_ENTRY_SIZE;
    let total = reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_ENTRIES + entry;
    let mut command = vec![0u8; total];
    let op = wire_render::OPCODE_SET_VERTEX_BUFFER;
    st32(&mut command[0..], op);
    st32(&mut command[4..], total as u32);
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_FIRST..],
        FIRST,
    );
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_COUNT..],
        1,
    );
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render_pass::BIND_ENTRIES..],
        0x4444,
    );
    {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
    }

    let StreamRefusal::Bind(over) = acc
        .unrepresentable
        .expect("the walk recorded the bind it could not hold")
    else {
        panic!("a bind past the table is not a pass refusal");
    };
    assert_eq!(over.index, FIRST);
    assert!(matches!(over.class, BindClass::Buffer));
    assert!(acc.bind_snapshot().is_err());

    let before = store_route_count("render_draw_refused_unrepresentable");
    draw(&mut acc);
    assert_eq!(
        acc.draws.len(),
        1,
        "the draw after the refused bind is not recorded"
    );
    assert_eq!(
        store_route_count("render_draw_refused_unrepresentable"),
        before + 1,
        "and the refusal is counted"
    );

    assert_eq!(
        acc.draws[0].pipeline_ref, 61,
        "the draw recorded before the refused bind still stands"
    );
}

/// A `SetBufferOffset` naming a slot past the buffer table refuses the stream's
/// draws, and says which slot on the fail channel.
///
/// The second record a guest spends on an unreachable slot. It has always been
/// counted and it has never been on the always-on failure path — the same gap
/// [`BindSlotPastTable`]'s doc argues about for the bind itself: a census route
/// reading zero is absent from its `OFF` line, so the first time a guest lost
/// one, nothing said so.
///
/// Driven without a prior bind at that slot on purpose. In a conforming stream
/// the bind came first and already refused, because Metal requires a buffer
/// bound at the index before `setVertexBufferOffset:atIndex:` — and a stream
/// where it did *not* come first is exactly the one where relying on that would
/// be wrong. So this drives the case the reasoning does not cover.
#[test]
fn a_buffer_offset_past_the_table_refuses_the_stream() {
    use crate::runtime::drain::store_route_count;
    use crate::runtime::render_pass::{BUFFER_OFFSET_INDEX, BUFFER_OFFSET_PAYLOAD_LEN};

    const FIRST: u32 = MAX_BUFFER_BIND_SLOTS + 1;

    let total = OP_HEADER_LEN + BUFFER_OFFSET_PAYLOAD_LEN;
    let mut command = vec![0u8; total];
    let op = wire_render::OPCODE_SET_VERTEX_BUFFER_OFFSET;
    st32(&mut command[0..], op);
    st32(&mut command[4..], total as u32);
    st32(&mut command[OP_HEADER_LEN + BUFFER_OFFSET_INDEX..], FIRST);

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum {
        pipeline_ref: 61,
        ..Default::default()
    };
    assert!(
        acc.bind_snapshot().is_ok(),
        "nothing is refused before the record arrives"
    );

    let before = store_route_count("render_buffer_offset_slot_past_table");
    handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);

    assert_eq!(
        store_route_count("render_buffer_offset_slot_past_table"),
        before + 1,
        "the counter still says how much"
    );
    assert!(
        matches!(
            acc.bind_snapshot(),
            Err(StreamRefusal::BufferOffset(over)) if over.index == FIRST
        ),
        "and the record now refuses the draws that would run without it"
    );

    // The two records that name an unreachable slot keep separate slugs, so a
    // reader can tell the bind from the offset — and separate `fail_once`
    // latches, so neither hides the other's first sighting.
    let line = crate::observe::Emit::decline(
        "render_buffer_offset",
        &BufferOffsetSlotPastTable {
            stage: ShaderStage::Vertex,
            index: FIRST,
        },
    )
    .render();
    assert_eq!(
        line,
        format!(
            "render_buffer_offset reason=render_buffer_offset_slot_past_table \
             stage=vertex index={FIRST} table=31 apple_table=31"
        )
    );
}

/// `setVisibilityResultMode:offset:` reaches the accumulator, and the pass's
/// buffer ref reaches it beside the mode.
///
/// These two used to be counted and dropped, one counter each. They are decoded
/// from separate records and mean nothing apart — the mode says what to count,
/// the pass says which guest buffer the count lands in — so a fixture that
/// drove only one of them would have passed against a device that still lost
/// the other.
///
/// `MTLVisibilityResultModeDisabled` is 0, and it is the guest *disarming* the
/// query rather than an unknown ordinal, so it must clear the arming rather
/// than record a mode of zero. That is the case the `Option` exists for and the
/// one a naive `mode: u32` field would get wrong.
#[test]
fn an_armed_visibility_query_and_its_buffer_both_reach_the_accumulator() {
    let mut state = DeviceState::new(DeviceId(0), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();

    let mut arm = |acc: &mut StreamAccum, offset: u64, mode: u64| {
        let total = wire_render::SET_VISIBILITY_RESULT_MODE_TOTAL_LEN as usize;
        let mut command = vec![0u8; total];
        st32(
            &mut command[0..],
            wire_render::OPCODE_SET_VISIBILITY_RESULT_MODE,
        );
        st32(&mut command[4..], total as u32);
        // Offset first, mode second — the wire's own order, which is the
        // reverse of the selector's.
        st64(&mut command[reims_vgpu_wire::OP_HEADER_LEN..], offset);
        st64(&mut command[reims_vgpu_wire::OP_HEADER_LEN + 8..], mode);
        handle_render_record(
            &mut state,
            &host,
            1,
            wire_render::OPCODE_SET_VISIBILITY_RESULT_MODE,
            &command,
            &mut out,
            acc,
        );
    };

    // Counting at 0x1234.
    arm(&mut acc, 0x1234, 2);
    assert_eq!(
        acc.visibility,
        Some(crate::runtime::draw::VisibilityArming {
            mode: 2,
            offset: 0x1234
        }),
        "a counting query keeps both its mode and the offset it writes to"
    );

    // A second record replaces the first: this is encoder state, and Metal's
    // second `setVisibilityResultMode:` genuinely supersedes the first.
    arm(&mut acc, 0x20, 1);
    assert_eq!(
        acc.visibility,
        Some(crate::runtime::draw::VisibilityArming {
            mode: 1,
            offset: 0x20
        }),
        "a second arming replaces the first rather than accumulating"
    );

    // Disabled clears it, rather than arming a query with mode 0.
    arm(&mut acc, 0x20, 0);
    assert_eq!(
        acc.visibility, None,
        "MTLVisibilityResultModeDisabled disarms; it is not a third mode"
    );
}

/// A visibility count lands in the guest's buffer, at the offset the guest
/// named, in the width and byte order it will read.
///
/// The two ends of this path each had a test and the middle had none: the
/// engine's counts are pinned against real hardware in `vk_engine_parity`, and
/// `an_armed_visibility_query_and_its_buffer_both_reach_the_accumulator` pins
/// the decode. Between them sits the part a guest actually depends on — that
/// the number reaches `base + offset` of the buffer the *pass* named, as a
/// little-endian `u64`. Every one of those four is a way to write a plausible
/// wrong answer into memory the guest will cull on, and none of them shows up
/// as a wrong picture.
///
/// Two offsets, because several are legal in one pass and are independent
/// questions: a writeback that kept one result per pass would pass a
/// single-offset fixture.
#[test]
fn a_visibility_count_lands_at_the_guest_offset_the_pass_named() {
    use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_BUFFER, RESOURCE_PAGE_SHIFT,
    };
    use crate::runtime::gva_mem;
    use crate::runtime::host::HostMemory;

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);

    // A task with a one-level page directory, the same shape the ICB fixtures
    // build: without it no GVA resolves and the writeback would refuse for a
    // reason that has nothing to do with what is under test.
    let (dir_pfn, root_pfn) = (2u32, 3u32);
    let dir_gpa = u64::from(dir_pfn) << PAGE_SHIFT_ARM64E;
    let root_gpa = u64::from(root_pfn) << PAGE_SHIFT_ARM64E;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x4000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], root_pfn);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    for i in 0..8u32 {
        let pfn = 4 + i;
        host.map_range(u64::from(pfn) << PAGE_SHIFT_ARM64E, 0x4000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        let _ = host.write_gpa(root_gpa + u64::from(i) * 4, &pte);
    }
    state.define_task(1, 0x1000, dir_pfn);
    assert!(state.set_object_list(1, 0, 32));

    // A buffer of one page at handle 5, named by object ref 7 — the
    // `handle << page_shift` shape `resolve_buffer_span` decodes.
    const BUF_REF: u32 = 7;
    const BUF_HANDLE: u32 = 5;
    const BUF_SIZE: u64 = 64;
    let buf_gva = u64::from(BUF_HANDLE) << RESOURCE_PAGE_SHIFT;
    let desc_gva = 0x1a0u64;
    let mut bdesc = vec![0u8; 16];
    st64(&mut bdesc[0..], BUF_SIZE);
    st64(&mut bdesc[8..], u64::from(BUF_HANDLE));
    gva_mem::write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &bdesc);
    let entry_off = list_object_entry_offset(BUF_REF, 32).unwrap();
    let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut le[0..],
        u32::from(OBJECT_TYPE_BUFFER) | ((bdesc.len() as u32) << 8),
    );
    le[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    gva_mem::write_task_gva_arm64e(&mut host, &state.tasks[1], entry_off, &le);

    let mut acc = StreamAccum {
        visibility_buffer_ref: BUF_REF,
        ..Default::default()
    };
    let mut counts = std::collections::BTreeMap::new();
    counts.insert(0u64, 64u64);
    counts.insert(16u64, 4095u64);
    write_visibility_results(&mut state, &mut host, 1, &acc, &counts);

    let mut got = [0u8; 8];
    gva_mem::read_task_gva(&host, &state.tasks[1], buf_gva, &mut got, PAGE_SHIFT_ARM64E)
        .expect("read the guest's visibility buffer back");
    assert_eq!(
        u64::from_le_bytes(got),
        64,
        "the count for offset 0 lands at the buffer's base, little-endian"
    );
    gva_mem::read_task_gva(
        &host,
        &state.tasks[1],
        buf_gva + 16,
        &mut got,
        PAGE_SHIFT_ARM64E,
    )
    .expect("read the second offset back");
    assert_eq!(
        u64::from_le_bytes(got),
        4095,
        "a second offset in the same pass is its own independent answer"
    );

    // A word that would run past the guest's allocation is refused rather than
    // written. The offset and the buffer arrive in different records, so this
    // pairing is checked nowhere else.
    let before = crate::runtime::drain::store_route_count("visibility_result_offset_past_buffer");
    let mut past = std::collections::BTreeMap::new();
    past.insert(BUF_SIZE - 4, 1u64);
    write_visibility_results(&mut state, &mut host, 1, &acc, &past);
    assert_eq!(
        crate::runtime::drain::store_route_count("visibility_result_offset_past_buffer"),
        before + 1,
        "a word straddling the end of the buffer is refused, and says so"
    );

    // Armed with no buffer named: nowhere to write, and it must not be silent.
    let quiet = crate::runtime::drain::store_route_count("visibility_result_no_buffer");
    acc.visibility_buffer_ref = 0;
    write_visibility_results(&mut state, &mut host, 1, &acc, &counts);
    assert_eq!(
        crate::runtime::drain::store_route_count("visibility_result_no_buffer"),
        quiet + 1
    );
}

/// A render encoder's `updateFence:` and `waitForFence:` reach the render-fence
/// domain.
///
/// The regression: this arm matched a *render* opcode against the blit
/// encoder's fence constants. Each encoder numbers its selectors in its own
/// space, so the comparison could never succeed, and every render fence the
/// guest encoded went to the unknown-opcode arm and was dropped. The two pairs
/// are far enough apart that no value collides, which is why this failed
/// wholesale rather than intermittently.
///
/// Asserting on the generation store rather than on the absence of a log line:
/// the store is what a later wait actually reads, so it is the thing whose loss
/// costs the guest its ordering.
#[test]
fn a_render_encoder_fence_reaches_the_render_fence_domain() {
    use crate::model::{FENCE_DOMAIN_BLIT, FENCE_DOMAIN_RENDER};
    use reims_vgpu_protocol::segment::{SegmentKind, SEGMENT_HEADER_LEN};
    use reims_vgpu_wire::ops::render as wire_render;

    const FENCE_REF: u32 = 6464;
    const STAGES_FRAGMENT: u32 = 2;

    fn push_fence(buf: &mut Vec<u8>, opcode: u32, fence_ref: u32) {
        let mut hdr = [0u8; 8];
        st32(&mut hdr[0..4], opcode);
        st32(&mut hdr[4..8], wire_render::FENCE_TOTAL_LEN);
        buf.extend_from_slice(&hdr);
        let mut payload = [0u8; 8];
        st32(&mut payload[0..4], fence_ref);
        st32(&mut payload[4..8], STAGES_FRAGMENT);
        buf.extend_from_slice(&payload);
    }

    let mut records = Vec::new();
    push_fence(&mut records, wire_render::OPCODE_UPDATE_FENCE, FENCE_REF);
    push_fence(&mut records, wire_render::OPCODE_UPDATE_FENCE, FENCE_REF);
    push_fence(&mut records, wire_render::OPCODE_WAIT_FOR_FENCE, FENCE_REF);

    let mut stream = vec![0u8; SEGMENT_HEADER_LEN];
    let stream_len = stream.len() + records.len();
    st32(&mut stream[0..4], stream_len as u32);
    stream[4] = SegmentKind::Render.wire_type();
    stream.extend_from_slice(&records);

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    walk_stream(&mut state, &mut host, 1, &stream, &mut out, &mut acc, None);

    // Two updates: the first seeds the generation, the second advances it. A
    // dropped fence leaves `None` here, which is what this used to read.
    assert_eq!(
        state.fence_generation(1, FENCE_DOMAIN_RENDER, FENCE_REF),
        Some(2),
        "both updates landed on the render-fence domain"
    );
    // And they landed on the *render* domain specifically — the constants this
    // arm used to compare against belong to the blit encoder.
    assert_eq!(
        state.fence_generation(1, FENCE_DOMAIN_BLIT, FENCE_REF),
        None,
        "a render fence does not touch the blit encoder's domain"
    );
}

/// `MTLLoadActionClear` seeds the pass whatever the store action says, and only
/// `MTLStoreActionStore` lets that colour reach the guest's pages.
///
/// The two used to be one test of one flag. A `Clear` + `DontCare` attachment
/// was dropped from `StreamAccum::clears` outright, which took the pass's CLEAR
/// **seed** with it — so a drawn pass began on the attachment's stale contents.
/// The store action never said anything about that; it says only that the
/// result is not preserved afterwards.
///
/// macOS 26 sends the pair 23 times in a 25 s Safari drag and macOS 14 twice,
/// against zero on 11/12/13. The branch that dropped it was written as a
/// healthy-zero alarm and those are firings of it.
///
/// Both halves are asserted here because fixing one without the other is a
/// plausible half-repair in either direction: seeding but also publishing
/// invents content the guest said it did not want, and filtering the publish
/// without restoring the seed leaves the original bug.
#[test]
fn a_clear_seeds_the_pass_for_any_store_action_and_publishes_only_for_store() {
    use crate::runtime::render_pass::{
        PASS_ATTACH_LOAD_ACTION, PASS_ATTACH_STORE_ACTION, PASS_ATTACH_TEXREF,
        PASS_COLOR_ATTACH_OFF,
    };
    use reims_vgpu_protocol::pass_action::MTL_STORE_ACTION_DONT_CARE;

    let seeded = |store_action: u16| {
        let mut payload = vec![0u8; 0x400];
        st32(
            &mut payload[PASS_COLOR_ATTACH_OFF + PASS_ATTACH_TEXREF..],
            7,
        );
        payload[PASS_COLOR_ATTACH_OFF + PASS_ATTACH_LOAD_ACTION
            ..PASS_COLOR_ATTACH_OFF + PASS_ATTACH_LOAD_ACTION + 2]
            .copy_from_slice(&MTL_LOAD_ACTION_CLEAR.to_le_bytes());
        payload[PASS_COLOR_ATTACH_OFF + PASS_ATTACH_STORE_ACTION
            ..PASS_COLOR_ATTACH_OFF + PASS_ATTACH_STORE_ACTION + 2]
            .copy_from_slice(&store_action.to_le_bytes());
        let mut cmd = vec![0u8; OP_HEADER_LEN + payload.len()];
        st32(&mut cmd[0..], wire_pass::OPCODE_RENDER_PASS);
        st32(&mut cmd[4..], (OP_HEADER_LEN + payload.len()) as u32);
        cmd[OP_HEADER_LEN..].copy_from_slice(&payload);

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        handle_render_record(
            &mut state,
            &host,
            1,
            wire_pass::OPCODE_RENDER_PASS,
            &cmd,
            &mut out,
            &mut acc,
        );
        acc
    };

    for store_action in [MTL_STORE_ACTION_STORE, MTL_STORE_ACTION_DONT_CARE] {
        let acc = seeded(store_action);
        assert!(
            acc.clears.iter().any(|a| a.texture_ref == 7),
            "load=Clear store={store_action} must still seed the pass; \
             a drawn pass would otherwise start on the attachment's stale contents"
        );
    }

    // ...and the store action decides only whether that colour is published.
    assert!(
        seeded(MTL_STORE_ACTION_STORE)
            .clears_reaching_guest_pages()
            .any(|a| a.texture_ref == 7),
        "a Store result is the guest's to read"
    );
    assert_eq!(
        seeded(MTL_STORE_ACTION_DONT_CARE)
            .clears_reaching_guest_pages()
            .count(),
        0,
        "DontCare says the result is dropped, so writing the clear colour into \
         guest pages would be inventing content the guest declined"
    );
}

/// The two resolve-carrying actions have different publication contracts.
#[test]
fn a_clear_distinguishes_resolve_only_from_store_and_resolve() {
    use reims_vgpu_protocol::pass_action::{
        MTL_STORE_ACTION_DONT_CARE, MTL_STORE_ACTION_MULTISAMPLE_RESOLVE,
        MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE,
    };

    let att = |store: u16, texture_ref: u32, resolve_texture_ref: u32| ColorAttachment {
        texture_ref,
        resolve_texture_ref,
        store_action: store,
        load_action: MTL_LOAD_ACTION_CLEAR,
        ..Default::default()
    };

    assert_eq!(
        clear_publish_target(&att(MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE, 4, 3)),
        ClearPublish::StoredAndResolved {
            source: 4,
            resolve: 3
        },
        "store-and-resolve retains the multisample source and publishes the resolve"
    );
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    assert!(
        !apply_clear(
            &mut state,
            &mut host,
            1,
            &att(MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE, 4, 3),
        ),
        "the single-sample clear rail must not claim both destinations were published"
    );
    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(
        log.lines().any(|line| {
            line.contains("reason=clear_store_and_multisample_resolve_unsupported")
                && line.contains("source=4")
                && line.contains("resolve=3")
        }),
        "the unsupported two-destination clear must be fail-visible"
    );
    assert_eq!(
        clear_publish_target(&att(MTL_STORE_ACTION_MULTISAMPLE_RESOLVE, 4, 3)),
        ClearPublish::Resolved(3),
        "resolve-only publishes only the single-sample destination"
    );

    // A plain single-sample store keeps the attachment the guest declared,
    // level and all — it is not retargeted.
    assert_eq!(
        clear_publish_target(&att(MTL_STORE_ACTION_STORE, 4, 0)),
        ClearPublish::Direct
    );

    // **And it keeps it even when a resolve texture is named**, which is a
    // deliberate narrowing and the one case a reader is most likely to assume
    // went the other way. The retarget used to key on `resolve_texture_ref != 0`
    // alone, so a descriptor that set `resolveTexture` alongside
    // `MTLStoreActionStore` landed its clear in the resolve texture at level
    // zero — a different surface from the one the guest declared. The store
    // action is what says whether a resolve happens; a resolve texture the
    // action does not name is not this pass's destination.
    assert_eq!(
        clear_publish_target(&att(MTL_STORE_ACTION_STORE, 4, 3)),
        ClearPublish::Direct,
        "a non-resolving store publishes into the texture it declared, whatever \
         resolve texture the descriptor also carries"
    );

    // A resolve with nowhere to resolve into is still a named refusal, for both
    // actions. This is the half that must NOT be lost by admitting the pair
    // above.
    for store in [
        MTL_STORE_ACTION_MULTISAMPLE_RESOLVE,
        MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE,
    ] {
        assert_eq!(
            clear_publish_target(&att(store, 4, 0)),
            ClearPublish::ResolveTargetMissing,
            "a resolve naming no resolve texture has nowhere to publish"
        );
    }

    // Nothing to publish: an action that keeps no single sample, and no
    // attachment at all. Neither is a loss.
    assert_eq!(
        clear_publish_target(&att(MTL_STORE_ACTION_DONT_CARE, 4, 3)),
        ClearPublish::NotPublished
    );
    assert_eq!(
        clear_publish_target(&att(MTL_STORE_ACTION_STORE, 0, 0)),
        ClearPublish::NotPublished
    );
}

/// A successful draw in an earlier render stream does not suppress the clear
/// fallback for a later stream whose own draw failed.
#[test]
fn clear_fallback_draw_accounting_is_scoped_to_one_render_stream() {
    let at_entry = (7, 3);
    let mut out = ExecResult {
        metal_draws_ok: 7,
        metal_draws_fail: 4,
        ..Default::default()
    };
    assert_eq!(
        stream_draw_delta(&out, at_entry),
        StreamDrawDelta { ok: 0, fail: 1 },
        "the earlier streams' seven successful draws are not this stream's success"
    );

    out.metal_draws_ok += 1;
    assert_eq!(
        stream_draw_delta(&out, at_entry),
        StreamDrawDelta { ok: 1, fail: 1 },
        "a draw that lands in this stream suppresses its destructive fallback"
    );
}

/// A multisample colour attachment has no single-sample linear publication.
///
/// # The contract
///
/// On rail macos-15 the guest renders 300x300 tiles into normal-texture linear
/// textures whose descriptors declare `sampleCount = 4`, and it sizes those
/// allocations to match: the boot's `gva_view_fragmented ... pages=352` at two
/// separate targets fixes `height * bpr` inside `(351, 352] * 4096`, which for
/// `height = 300` admits one plausible stride — `4800`, exactly four times the
/// single-sample tight row of a 300-wide BGRA8 texture. Ordinary single-sample
/// 300x300 surfaces in the same boot report `bpr=1216`.
///
/// So the guest strided that span for four samples per pixel, and this device
/// has never established what the samples' layout inside it is. A single-sample
/// image written there is not a partial answer, it is the wrong content: it
/// fills 1200 of every 4800 bytes and leaves the rest.
///
/// `apply_clear`'s sibling arm already states the rule for
/// `StoreAndMultisampleResolve` — "treating the source as a linear image would
/// write only one sample" — and could not apply it to a plain `Store`, because
/// `clear_publish_target` decides from the store action alone and never sees a
/// sample count.
///
/// # What this guards
///
/// The clear fallback is reached exactly when the draw's Store was refused, and
/// on this rail that refusal is `read_target_multisample_image`, which is
/// itself correct. What followed it was a solid beige or near-black 300x300
/// clear written over the guest's four-sample pages, twice a boot.
#[test]
fn a_multisample_linear_target_keeps_its_guest_bytes_instead_of_a_one_sample_clear() {
    use crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::resource::{
        list_object_entry_offset, LINEAR_DESC_HANDLE, LINEAR_DESC_SIZE, OBJECT_LIST_ENTRY_LEN,
        OBJECT_TYPE_TEXTURE, TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_HEIGHT, TEXTURE_DESC_PIXEL_FORMAT,
        TEXTURE_DESC_ROW_STRIDE, TEXTURE_DESC_SAMPLE_COUNT, TEXTURE_DESC_TRAILER_HEIGHT,
        TEXTURE_DESC_TRAILER_WIDTH, TEXTURE_DESC_WIDTH,
    };
    use crate::runtime::gva_mem::{
        define_task_pages_arm64e, read_task_gva_by_id, write_task_gva_arm64e,
    };

    // Both arms of one comparison: the same texture, the same clear, differing
    // only in the sample count its descriptor declares. Single-sample must keep
    // publishing — this narrows the clear rail, it does not close it.
    let published = |sample_count: u16| -> (bool, Vec<u8>) {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        define_task_pages_arm64e(&mut host, &mut state, 4, 16);
        assert!(state.set_object_list(1, 0, 256));

        // Four samples per pixel in the stride, as the rail's own span
        // arithmetic gives it: `bpr = samples * width * 4`.
        let (texture_ref, handle, width, height) = (200u32, 8u32, 2u32, 2u32);
        let row_stride = u32::from(sample_count) * width * 4;
        let target_gva = (handle as u64) << PAGE_SHIFT_ARM64E;
        let mut pages = vec![0xcc; (row_stride * height) as usize];
        write_task_gva_arm64e(&mut host, &state.tasks[1], target_gva, &pages);

        let mut desc = vec![0u8; TEXTURE_DESC_BASE_LEN];
        st64(
            &mut desc[LINEAR_DESC_SIZE..],
            u64::from(row_stride) * u64::from(height),
        );
        st32(&mut desc[LINEAR_DESC_HANDLE..], handle);
        st32(&mut desc[TEXTURE_DESC_ROW_STRIDE..], row_stride);
        st32(&mut desc[TEXTURE_DESC_WIDTH..], width);
        st32(&mut desc[TEXTURE_DESC_HEIGHT..], height);
        st16(
            &mut desc[TEXTURE_DESC_PIXEL_FORMAT..],
            MTL_FORMAT_BGRA8_UNORM,
        );
        // The trailer states the extent a second time, and the decode returns
        // the sample count only when the two statements agree.
        st32(&mut desc[TEXTURE_DESC_TRAILER_WIDTH..], width);
        st32(&mut desc[TEXTURE_DESC_TRAILER_HEIGHT..], height);
        st16(&mut desc[TEXTURE_DESC_SAMPLE_COUNT..], sample_count);
        let desc_gva = 0x280u64;
        write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &desc);
        let entry_off = list_object_entry_offset(texture_ref, 256).expect("object-list ref");
        let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        st32(
            &mut entry,
            (OBJECT_TYPE_TEXTURE as u32) | ((desc.len() as u32) << 8),
        );
        entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        write_task_gva_arm64e(&mut host, &state.tasks[1], entry_off, &entry);

        let mut acc = StreamAccum::default();
        acc.clears.push(ColorAttachment {
            texture_ref,
            load_action: MTL_LOAD_ACTION_CLEAR,
            store_action: MTL_STORE_ACTION_STORE,
            clear_color: [1.0, 1.0, 1.0, 1.0],
            ..Default::default()
        });
        let mut out = ExecResult::default();
        finish_stream(&mut state, &mut host, 1, &mut out, &acc);
        read_task_gva_by_id(
            &host,
            &state.tasks,
            1,
            target_gva,
            &mut pages,
            PAGE_SHIFT_ARM64E,
        )
        .expect("read the target's guest pages");
        (out.clears_applied == 1, pages)
    };

    let (one_published, one_pages) = published(1);
    assert!(
        one_published,
        "a single-sample linear Store still publishes its clear"
    );
    assert!(
        one_pages.contains(&0xff),
        "the single-sample arm must actually have written the clear, or the \
         multisample arm below proves nothing"
    );

    let (four_published, four_pages) = published(4);
    assert!(
        !four_published,
        "a four-sample attachment has no single-sample linear publication"
    );
    assert!(
        four_pages.iter().all(|&byte| byte == 0xcc),
        "the guest's four-sample span must be left exactly as it was found, not \
         overwritten with one sample's worth of clear colour"
    );

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(
        log.lines().any(|line| {
            line.contains("reason=clear_multisample_source_not_linear")
                && line.contains("samples=4")
        }),
        "the refused publication must be fail-visible: a clear this device drops \
         silently is the one outcome the ground rules forbid"
    );
}

/// The info rail had exactly one arm and seventeen silences. A boot in which
/// the guest asked eighteen questions and got no answers read identically to a
/// boot in which it asked none, and the answer it read back was whatever its
/// reply buffer last held.
#[test]
fn an_unanswered_info_query_names_the_question_it_did_not_answer() {
    use reims_vgpu_wire::ops::info as w;
    let cap = crate::observe::sink::FailCapture::start();
    super::note_info_record_unanswered(9, w::OPCODE_HEAP_HOST_RESOURCE_INFO, 0x18);
    let line = cap.one("info_record");
    assert!(
        line.contains("reason=info_query_unanswered"),
        "the ledger has this query as unresolved, so the line must say so: {line}"
    );
    assert!(
        line.contains("op=0x1cf") && line.contains("task=9"),
        "the line must name the query and whose task asked it: {line}"
    );
    assert!(
        line.contains("selector=heapHostResourceInfo"),
        "ten of these selectors write the same record and differ only in what \
         they ask about, so the selector is the whole content of the line: {line}"
    );
}

/// An opcode on a rail whose whole space is enumerated is not a dropped
/// command — it is a stream this project cannot frame.
#[test]
fn an_info_opcode_outside_the_declared_space_says_it_is_unjudged() {
    let cap = crate::observe::sink::FailCapture::start();
    super::note_info_record_unanswered(4, 0x1ff, 0x18);
    let line = cap.one("info_record");
    assert!(
        line.contains("reason=info_opcode_unjudged"),
        "an opcode the manifest and the ledger both lack must not read as a \
         merely-unanswered query: {line}"
    );
}

/// The ledger and the rail disagreeing is its own finding. Nothing in the
/// shipped ledger produces it today — every info row is unresolved — so the
/// arm is proved against a row rather than against the rail, which is also the
/// only way it stays proved once a row is closed.
#[test]
fn a_ledger_row_the_rail_cannot_honour_reports_the_disagreement() {
    use crate::observe::Decline as _;
    use reims_vgpu_protocol::closure::{Closure, Op, Rail};
    const CLAIMED: Op = Op {
        rail: Rail::Info,
        opcode: Some(0x1c2),
        selector: "computePipelineStateInfo:info:",
        closure: Closure::Implemented {
            evidence: "a row that claims an answer this rail does not produce",
        },
    };
    let decline = super::report::InfoRecordUnanswered {
        opcode: 0x1c2,
        len: 0x18,
        judged: Some(&CLAIMED),
    };
    assert_eq!(decline.slug(), "info_query_ledger_disagrees");
    let fields = decline.fields();
    assert!(
        fields.contains(&("closure", "implemented".to_string())),
        "the line must carry what the ledger claimed, or the reader cannot tell \
         which of the two is wrong: {fields:?}"
    );
}

/// A vertex-amplification count of one whose view mapping is not the identity
/// is a loss, and it is not the same loss as a count above one.
///
/// The mappings offset the viewport and render-target *array indices* the views
/// rasterise into. A count of one with a zero mapping is the API default and
/// asks for what this rail already does; a count of one with a non-zero offset
/// asks for a draw aimed at a different array slice, and this rail aims it at
/// slice zero. Until the mappings were lifted, `count > 1` reported those two
/// as the same record and the second cost the guest a wrongly-targeted draw
/// with nothing said.
///
/// The two routes are pinned apart because the losses are different: one
/// renders one view where several were asked for, and the other renders the
/// right number of views into the wrong slice.
#[test]
fn a_view_mapping_that_offsets_a_view_is_named_apart_from_a_count_above_one() {
    use crate::runtime::drain::store_route_count;

    let record = |count: u32, viewport: u32, render_target: u32| {
        let total = reims_vgpu_wire::OP_HEADER_LEN + 4 + count as usize * 8;
        let mut v = vec![0u8; total];
        st32(
            &mut v[0..],
            wire_render::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT,
        );
        st32(&mut v[4..], total as u32);
        st32(&mut v[reims_vgpu_wire::OP_HEADER_LEN..], count);
        for i in 0..count as usize {
            let e = reims_vgpu_wire::OP_HEADER_LEN + 4 + i * 8;
            st32(&mut v[e..], viewport);
            st32(&mut v[e + 4..], render_target);
        }
        v
    };

    let run = |command: Vec<u8>| {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        crate::runtime::decode::render_spi::decode(&command)
            .expect("a well-formed amplification count decodes");
        handle_render_record(
            &mut state,
            &host,
            1,
            wire_render::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT,
            &command,
            &mut out,
            &mut acc,
        );
    };

    let dropped = "render_vertex_amplification_dropped";
    let mapping = "render_vertex_amplification_view_mapping_dropped";

    // One view, identity mapping: the API default, and neither route moves.
    let (a, b) = (store_route_count(dropped), store_route_count(mapping));
    run(record(1, 0, 0));
    assert_eq!(store_route_count(dropped), a, "one view is not a loss");
    assert_eq!(
        store_route_count(mapping),
        b,
        "and the identity mapping asks for nothing"
    );

    // One view, offset mapping: the mapping route alone.
    run(record(1, 2, 3));
    assert_eq!(
        store_route_count(dropped),
        a,
        "the count is still the default, so the count route must not move"
    );
    assert_eq!(
        store_route_count(mapping) - b,
        1,
        "a draw aimed at a different array slice is a loss and must be named"
    );

    // Two views, identity mappings: the count route alone.
    run(record(2, 0, 0));
    assert_eq!(store_route_count(dropped) - a, 1);
    assert_eq!(
        store_route_count(mapping) - b,
        1,
        "an identity mapping is not a second loss"
    );

    // Two views, offset mappings: both.
    run(record(2, 4, 5));
    assert_eq!(store_route_count(dropped) - a, 2);
    assert_eq!(store_route_count(mapping) - b, 2);
}

/// All four residency forms reach the residency routes, including the two the
/// render encoder inherits rather than declares.
///
/// `useHeaps:count:` and `useResources:count:usage:` are declared on the shared
/// encoder base class, so they are absent from the serializer manifest while
/// being callable on a render encoder — the worked example the closure ledger's
/// `OFF_MANIFEST` names. The render rail once knew only the two
/// `stages:`-qualified overrides, and an unqualified record reached no arm and
/// was reported as `accepted_without_executor`: a residency declaration whose
/// class nothing classified, sitting in a counter that says only "some opcode".
///
/// The decoder's own tests pin that all four decode. This is the other half:
/// that the executor routes all four by *class* rather than dropping half the
/// family into the unimplemented counter. Driven per opcode, because that is the
/// axis the regression ran along.
#[test]
fn every_residency_form_reaches_a_residency_route_and_not_the_unimplemented_one() {
    use crate::runtime::drain::store_route_count;
    use crate::runtime::render_pass;

    // Each form's refs start at its own offset — the heads are three different
    // sizes — so the record is built from the decoder's own constants rather
    // than from one literal that would be wrong for three of the four.
    let record = |op: u32, refs_at: usize| {
        let total = reims_vgpu_wire::OP_HEADER_LEN + refs_at + 4;
        let mut v = vec![0u8; total];
        st32(&mut v[0..], op);
        st32(&mut v[4..], total as u32);
        st32(
            &mut v[reims_vgpu_wire::OP_HEADER_LEN + render_pass::RESIDENCY_COUNT..],
            1,
        );
        v
    };

    let forms = [
        (
            wire_render::OPCODE_USE_RESOURCE,
            render_pass::USE_RESOURCE_REFS,
            "render_residency_empty",
        ),
        (
            wire_render::OPCODE_USE_HEAP,
            render_pass::USE_HEAP_REFS,
            "render_residency_heap",
        ),
        (
            wire_render::OPCODE_USE_RESOURCES_NO_STAGES,
            render_pass::USE_RESOURCES_NO_STAGES_REFS,
            "render_residency_empty",
        ),
        (
            wire_render::OPCODE_USE_HEAPS_NO_STAGES,
            render_pass::USE_HEAPS_NO_STAGES_REFS,
            "render_residency_heap",
        ),
    ];

    for (op, refs_at, route) in forms {
        let command = record(op, refs_at);
        let framed = reims_vgpu_protocol::decode::op(&command, 0)
            .unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
        // `lift`, not `decode`: every residency row is unsettled, so `decode`
        // refuses it on principle and the layout question is this one.
        let decoded = reims_vgpu_protocol::decode::residency::lift(
            reims_vgpu_protocol::closure::Rail::Render,
            &framed,
        )
        .unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
        // The `useResource` forms carry a usage argument and this fixture
        // writes zero into it; the `useHeap` forms carry none at all. Those are
        // different records and `Option` is what keeps them apart — which is
        // the whole point of the field, so the assertion is written to allow
        // both rather than to pick one.
        assert!(
            decoded
                .usage
                .is_none_or(|u| u == reims_vgpu_protocol::residency::ResourceUsage(0)),
            "op {op:#x}: the fixture declares no usage and the lift read {:?}",
            decoded.usage
        );

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        let before = store_route_count(route);
        let cap = crate::observe::sink::FailCapture::start();
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
        assert_eq!(
            store_route_count(route) - before,
            1,
            "op {op:#x} did not reach {route}"
        );
        assert!(
            !cap.lines()
                .iter()
                .any(|l| l.contains("accepted_without_executor")),
            "op {op:#x} was reported as an opcode with no arm rather than as a \
             residency declaration: {:?}",
            cap.lines()
        );
    }
}

/// A residency declaration that names a GPU **write** is not the record the
/// no-op argument covers.
///
/// The whole family used to reach one counter, so a guest declaring that the
/// GPU would write a resource through a path this rail never bound produced the
/// same number as a guest declaring it would read one it already has. The first
/// leaves the guest reading back content it believes was just produced; the
/// second costs nothing. This pins that they are told apart, and that the write
/// case additionally reaches the always-on channel with the declaration on it.
#[test]
fn a_residency_write_declaration_is_named_rather_than_counted_with_the_reads() {
    use crate::runtime::drain::store_route_count;
    use crate::runtime::render_pass;
    use reims_vgpu_protocol::residency::{RenderStages, ResourceUsage};

    // `useResource:usage:stages:`: count at +0, usage and stages as two u16
    // sharing the word at +4, refs from +8.
    let record = |usage: u16, stages: u16| {
        let total = reims_vgpu_wire::OP_HEADER_LEN + render_pass::USE_RESOURCE_REFS + 4;
        let mut v = vec![0u8; total];
        st32(&mut v[0..], wire_render::OPCODE_USE_RESOURCE);
        st32(&mut v[4..], total as u32);
        st32(&mut v[reims_vgpu_wire::OP_HEADER_LEN..], 1);
        v[reims_vgpu_wire::OP_HEADER_LEN + 4..reims_vgpu_wire::OP_HEADER_LEN + 6]
            .copy_from_slice(&usage.to_le_bytes());
        v[reims_vgpu_wire::OP_HEADER_LEN + 6..reims_vgpu_wire::OP_HEADER_LEN + 8]
            .copy_from_slice(&stages.to_le_bytes());
        v
    };

    let bytes = record(ResourceUsage::WRITE as u16, RenderStages::FRAGMENT as u16);
    let framed = reims_vgpu_protocol::decode::op(&bytes, 0).expect("the record frames");
    let decoded = reims_vgpu_protocol::decode::residency::lift(
        reims_vgpu_protocol::closure::Rail::Render,
        &framed,
    )
    .expect("a well-formed useResource lifts");
    assert_eq!(
        decoded.usage,
        Some(ResourceUsage(ResourceUsage::WRITE)),
        "the usage half must survive the lift — it is what decides whether \
         answering by doing nothing is sound"
    );
    assert_eq!(decoded.stages, Some(RenderStages(RenderStages::FRAGMENT)));

    let run = |usage: u16, stages: u16| {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        let command = record(usage, stages);
        handle_render_record(
            &mut state,
            &host,
            1,
            wire_render::OPCODE_USE_RESOURCE,
            &command,
            &mut out,
            &mut acc,
        );
    };

    let before_read = store_route_count("render_residency_read");
    run(ResourceUsage::READ as u16, RenderStages::VERTEX as u16);
    assert_eq!(
        store_route_count("render_residency_read") - before_read,
        1,
        "a read declaration is the case the no-op argument covers"
    );

    let cap = crate::observe::sink::FailCapture::start();
    let before_write = store_route_count("render_residency_write");
    run(ResourceUsage::WRITE as u16, RenderStages::FRAGMENT as u16);
    assert_eq!(
        store_route_count("render_residency_write") - before_write,
        1,
        "a write declaration must not be counted with the reads"
    );
    let line = cap.one("render_residency");
    assert!(
        line.contains("reason=render_residency_write_dropped"),
        "a GPU write through a path this rail did not bind is lost guest \
         content, not a hint: {line}"
    );
    assert!(
        line.contains("usage=0x2") && line.contains("stages=0x2"),
        "the line must carry the declaration, or it cannot be acted on: {line}"
    );

    // A usage bit the API does not declare is not narrowed into the bits it
    // shares: `READ|0x8` must not read as the one case this rail is sure it
    // owes nothing on.
    let before_undeclared = store_route_count("render_residency_undeclared");
    run((ResourceUsage::READ | 0x8) as u16, 0);
    assert_eq!(
        store_route_count("render_residency_undeclared") - before_undeclared,
        1,
        "an undeclared usage bit must not classify as a read"
    );
    assert_eq!(
        store_route_count("render_residency_read") - before_read,
        1,
        "and it must not also count as one"
    );
}

/// The three compute-rail classes the closure ledger now routes each reach a
/// counter that names them, and none of them reaches another's.
///
/// Before the ledger answered the class, all three were arms of this device's
/// own `compute::Kind`, decoded in the same pass that read the fields — so a
/// record's class and its layout were one verdict and a misclassification read
/// fields at the wrong offsets. The routing is what this asserts, in both
/// directions: a barrier must not be counted as a residency declaration and a
/// residency declaration must not be counted as a barrier, because the first is
/// ordering this device argues it already provides and the second is a hint it
/// argues it does not need. One bucket for both would make a driven boot's
/// reading unusable for checking either argument.
#[test]
fn each_ledger_routed_compute_class_reaches_a_counter_that_names_which_one_it_is() {
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_wire::ops::compute as wire_c;
    use reims_vgpu_wire::ops::render as wire_r;

    // `count` refs and nothing else — the shape of both the resource barrier
    // and the unqualified heap declaration.
    let counted = |op: u32, refs: &[u32]| {
        let total = (OP_HEADER_LEN + 4 + refs.len() * 4) as u32;
        let mut v = vec![0u8; total as usize];
        st32(&mut v[0..], op);
        st32(&mut v[4..], total);
        st32(&mut v[OP_HEADER_LEN..], refs.len() as u32);
        for (i, r) in refs.iter().enumerate() {
            st32(&mut v[OP_HEADER_LEN + 4 + i * 4..], *r);
        }
        v
    };
    // `useResources:count:usage:` puts its usage word between the count and the
    // refs; the heap form has no such word at all.
    let use_resources = |usage: u32, refs: &[u32]| {
        let total = (OP_HEADER_LEN + 8 + refs.len() * 4) as u32;
        let mut v = vec![0u8; total as usize];
        st32(&mut v[0..], wire_r::OPCODE_USE_RESOURCES_NO_STAGES);
        st32(&mut v[4..], total);
        st32(&mut v[OP_HEADER_LEN..], refs.len() as u32);
        st32(&mut v[OP_HEADER_LEN + 4..], usage);
        for (i, r) in refs.iter().enumerate() {
            st32(&mut v[OP_HEADER_LEN + 8 + i * 4..], *r);
        }
        v
    };
    let bare = |op: u32, payload_len: usize| {
        let total = (OP_HEADER_LEN + payload_len) as u32;
        let mut v = vec![0u8; total as usize];
        st32(&mut v[0..], op);
        st32(&mut v[4..], total);
        v
    };

    let mut scope = bare(
        wire_c::OPCODE_MEMORY_BARRIER_SCOPE,
        wire_c::MEMORY_BARRIER_SCOPE_TOTAL_LEN as usize - OP_HEADER_LEN,
    );
    st16(&mut scope[OP_HEADER_LEN..], 3);

    const BARRIER: &str = "compute_noop_barrier";
    const FLUSH: &str = "compute_noop_flush_compressed_reinterpretation";
    const HEAP: &str = "compute_residency_heap";
    const READ: &str = "compute_residency_read";
    const WRITE: &str = "compute_residency_write";
    const ALL: [&str; 5] = [BARRIER, FLUSH, HEAP, READ, WRITE];

    for (op, command, route) in [
        (
            wire_c::OPCODE_MEMORY_BARRIER_RESOURCES,
            counted(wire_c::OPCODE_MEMORY_BARRIER_RESOURCES, &[5151, 4343]),
            BARRIER,
        ),
        (wire_c::OPCODE_MEMORY_BARRIER_SCOPE, scope, BARRIER),
        (
            wire_c::OPCODE_INSERT_COMPRESSED_TEXTURE_FLUSH,
            bare(wire_c::OPCODE_INSERT_COMPRESSED_TEXTURE_FLUSH, 0),
            FLUSH,
        ),
        (
            wire_r::OPCODE_USE_HEAPS_NO_STAGES,
            counted(wire_r::OPCODE_USE_HEAPS_NO_STAGES, &[77]),
            HEAP,
        ),
        (
            wire_r::OPCODE_USE_RESOURCES_NO_STAGES,
            use_resources(
                reims_vgpu_protocol::residency::ResourceUsage::READ,
                &[88, 99],
            ),
            READ,
        ),
        (
            wire_r::OPCODE_USE_RESOURCES_NO_STAGES,
            use_resources(reims_vgpu_protocol::residency::ResourceUsage::WRITE, &[88]),
            WRITE,
        ),
    ] {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut seg = crate::runtime::compute_session::ComputeSegment::default();
        let before: Vec<u64> = ALL.iter().map(|r| store_route_count(r)).collect();
        handle_compute_record(&mut state, &mut host, 1, op, &command, &mut out, &mut seg);
        for (name, was) in ALL.iter().zip(before) {
            let now = store_route_count(name);
            let want = was + u64::from(*name == route);
            assert_eq!(
                now, want,
                "{op:#x} should have been counted once as {route} and {name} moved from {was} to \
                 {now}"
            );
        }
        // None of the three owns anything: the accumulator, the session and the
        // sequencing block are what separates them from the rest of the rail,
        // and a record that touched one of them would not be separable.
        assert!(
            seg.session.is_none()
                && seg.block.is_none()
                && seg.acc.pipeline_ref == 0
                && seg.acc.buffers.is_empty()
                && seg.acc.textures.is_empty()
                && seg.acc.samplers.is_empty()
                && seg.acc.threadgroup_memory.is_empty()
                && seg.acc.stage_in_region.is_none()
                && seg.acc.stage_in_region_indirect.is_none()
                && seg.acc.imageblock.is_none()
                && seg.acc.dispatch_type == reims_vgpu_protocol::compute::DispatchType::Serial,
            "{op:#x} moved segment state it does not own"
        );
    }
}

/// A ledger-routed compute record whose body does not frame is declined under
/// its own name rather than silently doing nothing.
///
/// The three no-op classes are still decoded, and this is why: the day one of
/// their no-op arguments stops holding, the shape is what has to be read, and a
/// route that counted on the opcode alone would have been recording a decode
/// that never happened.
#[test]
fn a_ledger_routed_compute_record_short_of_its_body_is_declined_and_not_counted() {
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_wire::ops::render as wire_r;

    // `count` says two refs and the record carries none of them.
    let total = (OP_HEADER_LEN + 4) as u32;
    let mut command = vec![0u8; total as usize];
    st32(&mut command[0..], wire_r::OPCODE_USE_HEAPS_NO_STAGES);
    st32(&mut command[4..], total);
    st32(&mut command[OP_HEADER_LEN..], 2);

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut seg = crate::runtime::compute_session::ComputeSegment::default();
    let before = store_route_count("compute_residency_heap");
    handle_compute_record(
        &mut state,
        &mut host,
        1,
        wire_r::OPCODE_USE_HEAPS_NO_STAGES,
        &command,
        &mut out,
        &mut seg,
    );
    assert_eq!(
        store_route_count("compute_residency_heap"),
        before,
        "a declaration whose refs are not there was priced as one that was"
    );
}

/// An `MTLDispatchType` the contract does not declare refuses its record, is
/// counted, and leaves the pass's type where it was.
///
/// This is the one place the compute cutover changed what a guest gets, and it
/// is the ordinal rule moving to the crate that owns closed ordinals. The
/// device used to store `writeDescriptor`'s word unbounded and narrow it at the
/// far end of the rail — `if x == CONCURRENT { CONCURRENT } else { SERIAL }`,
/// inside `execute_dispatch_metal`, on the one arm that read the field at all —
/// so an unrecognised ordinal silently became `Serial` and, on the Vulkan arm,
/// was stored and read by nobody. `reims_vgpu_protocol` refuses it at the lift
/// instead, which is what makes `ComputeAccum::dispatch_type` a total type.
///
/// The census the old substitution raised is kept, because what it was for —
/// deciding whether an unrecognised ordinal is a guest asking for something new
/// or this device reading the wrong offset — is still exactly what a firing
/// would mean.
///
/// The three claims are separate on purpose. That the record is refused is not
/// the same as that the counter moved, and neither says the pass kept the type
/// a *previous* record set — which is the half the substitution used to get
/// wrong in the other direction, by overwriting `Concurrent` with `Serial`.
#[test]
fn an_undeclared_dispatch_type_refuses_its_record_and_leaves_the_pass_type_alone() {
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_protocol::compute::DispatchType;
    use reims_vgpu_wire::ops::compute as wire_c;

    let descriptor = |word: u32| {
        let total = (OP_HEADER_LEN + 4) as u32;
        let mut v = vec![0u8; total as usize];
        st32(&mut v[0..], wire_c::OPCODE_WRITE_DESCRIPTOR);
        st32(&mut v[4..], total);
        st32(&mut v[OP_HEADER_LEN..], word);
        v
    };
    let run = |seg: &mut crate::runtime::compute_session::ComputeSegment, word: u32| {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let mut out = ExecResult::default();
        handle_compute_record(
            &mut state,
            &mut host,
            7,
            wire_c::OPCODE_WRITE_DESCRIPTOR,
            &descriptor(word),
            &mut out,
            seg,
        );
    };

    let mut seg = crate::runtime::compute_session::ComputeSegment::default();
    let before = store_route_count("compute_dispatch_type_unknown");

    // The two the contract declares cross untouched and spend no line.
    let cap = crate::observe::FailCapture::start();
    run(&mut seg, 1);
    assert_eq!(seg.acc.dispatch_type, DispatchType::Concurrent);
    run(&mut seg, 0);
    assert_eq!(seg.acc.dispatch_type, DispatchType::Serial);
    assert!(
        cap.lines().is_empty(),
        "a declared dispatch type must spend no line: {:?}",
        cap.lines()
    );
    assert_eq!(
        store_route_count("compute_dispatch_type_unknown"),
        before,
        "a declared dispatch type is not an unknown one"
    );
    drop(cap);

    // An ordinal outside the pair: refused by name, counted every time, and the
    // pass keeps the type the guest last actually stated.
    run(&mut seg, 1);
    assert_eq!(seg.acc.dispatch_type, DispatchType::Concurrent);
    let cap = crate::observe::FailCapture::start();
    for _ in 0..3 {
        run(&mut seg, 0x5e01);
    }
    assert_eq!(
        seg.acc.dispatch_type,
        DispatchType::Concurrent,
        "a refused descriptor must not quietly reset the pass to Serial"
    );
    let line = cap.one("compute_record");
    assert!(
        line.contains("undefined_ordinal") || line.contains("dispatch_type"),
        "the refusal must name the field the ordinal was in: {line}"
    );
    assert_eq!(
        store_route_count("compute_dispatch_type_unknown"),
        before + 3,
        "the line is deduped; the count is not"
    );
}

/// The three render-rail classes the closure ledger now routes each reach a
/// counter that names them, and none of them reaches another's.
///
/// Before the ledger answered the class, all three were arms of this device's
/// own `render::Kind`, decoded in the same pass that read the fields. The
/// routing is what this asserts, in both directions — a barrier must not be
/// counted as a residency declaration and a residency declaration must not be
/// counted as a barrier, because the first is ordering this device argues it
/// already provides and the second is a hint it argues it does not need.
///
/// It also pins the claim that makes this group a group: none of the three
/// touches `StreamAccum`. The indirect-command executions look equally
/// self-contained and are not — they push onto `acc.execute_icb` — which is why
/// they stayed behind.
#[test]
fn each_ledger_routed_render_class_reaches_a_counter_that_names_which_one_it_is() {
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_wire::ops::render as wire_r;

    let head_then_refs = |op: u32, head: &[u8], refs: &[u32]| {
        let total = (OP_HEADER_LEN + head.len() + refs.len() * 4) as u32;
        let mut v = vec![0u8; total as usize];
        st32(&mut v[0..], op);
        st32(&mut v[4..], total);
        v[OP_HEADER_LEN..OP_HEADER_LEN + head.len()].copy_from_slice(head);
        for (i, r) in refs.iter().enumerate() {
            st32(&mut v[OP_HEADER_LEN + head.len() + i * 4..], *r);
        }
        v
    };
    // `useResource:usage:stages:` — count, then a 16-bit usage and a 16-bit
    // stages sharing one word.
    let use_resource = |usage: u16, stages: u16, refs: &[u32]| {
        let mut head = vec![0u8; 8];
        st32(&mut head[0..], refs.len() as u32);
        st16(&mut head[4..], usage);
        st16(&mut head[6..], stages);
        head_then_refs(wire_r::OPCODE_USE_RESOURCE, &head, refs)
    };
    // `useHeap:stages:` — no usage at all, so `stages` sits alone at `+4` and
    // the refs begin at `+6`.
    let use_heap = |stages: u16, refs: &[u32]| {
        let mut head = vec![0u8; 6];
        st32(&mut head[0..], refs.len() as u32);
        st16(&mut head[4..], stages);
        head_then_refs(wire_r::OPCODE_USE_HEAP, &head, refs)
    };
    let bare = |op: u32, total: u32| {
        let mut v = vec![0u8; total as usize];
        st32(&mut v[0..], op);
        st32(&mut v[4..], total);
        v
    };

    const BARRIER: &str = "render_noop_barrier";
    const HEAP: &str = "render_residency_heap";
    const EMPTY: &str = "render_residency_empty";
    const READ: &str = "render_residency_read";
    const WRITE: &str = "render_residency_write";
    const ALL: [&str; 5] = [BARRIER, HEAP, EMPTY, READ, WRITE];

    let mut resources_head = vec![0u8; 8];
    st32(&mut resources_head[0..], 2);
    let barrier_resources = head_then_refs(
        wire_r::OPCODE_MEMORY_BARRIER_RESOURCES,
        &resources_head,
        &[5151, 4343],
    );

    for (op, command, route) in [
        (
            wire_r::OPCODE_MEMORY_BARRIER_RESOURCES,
            barrier_resources,
            BARRIER,
        ),
        (
            wire_r::OPCODE_MEMORY_BARRIER_SCOPE,
            bare(
                wire_r::OPCODE_MEMORY_BARRIER_SCOPE,
                wire_r::MEMORY_BARRIER_SCOPE_TOTAL_LEN,
            ),
            BARRIER,
        ),
        (
            wire_r::OPCODE_TEXTURE_BARRIER,
            bare(
                wire_r::OPCODE_TEXTURE_BARRIER,
                wire_r::TEXTURE_BARRIER_TOTAL_LEN,
            ),
            BARRIER,
        ),
        (wire_r::OPCODE_USE_HEAP, use_heap(1, &[77]), HEAP),
        // A `useResource` declaring no access at all is its own class: it is
        // not the read case, and reporting it as one would let the reading
        // that confirms the no-op absorb a shape the argument does not
        // describe.
        (
            wire_r::OPCODE_USE_RESOURCE,
            use_resource(0, 1, &[88]),
            EMPTY,
        ),
        (
            wire_r::OPCODE_USE_RESOURCE,
            use_resource(
                reims_vgpu_protocol::residency::ResourceUsage::READ as u16,
                1,
                &[88, 99],
            ),
            READ,
        ),
        (
            wire_r::OPCODE_USE_RESOURCE,
            use_resource(
                reims_vgpu_protocol::residency::ResourceUsage::WRITE as u16,
                1,
                &[88],
            ),
            WRITE,
        ),
    ] {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        let before: Vec<u64> = ALL.iter().map(|r| store_route_count(r)).collect();
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
        for (name, was) in ALL.iter().zip(before) {
            let now = store_route_count(name);
            let want = was + u64::from(*name == route);
            assert_eq!(
                now, want,
                "{op:#x} should have been counted once as {route} and {name} moved from {was} to \
                 {now}"
            );
        }
        assert!(
            acc.pipeline_ref == 0 && acc.execute_icb.is_empty() && acc.viewports.is_empty(),
            "{op:#x} moved stream state it does not own"
        );
    }
}

/// A render fence's direction comes from the same answer that says it is a
/// fence, so a fence that is neither an update nor a wait is unrepresentable.
///
/// The arm this replaces read the opcode a second time to pick the direction
/// and carried a third case for neither — a state
/// `reims_vgpu_protocol::sync::fence_kind` cannot produce, because the same
/// function decides both that this is a fence and which side of one it is.
#[test]
fn a_render_fence_orders_in_the_direction_its_own_opcode_names() {
    use crate::runtime::plan::event_sync::Domain;
    use reims_vgpu_wire::ops::render as wire_r;

    let fence = |op: u32, fence_ref: u32| {
        let mut v = vec![0u8; wire_r::FENCE_TOTAL_LEN as usize];
        st32(&mut v[0..], op);
        st32(&mut v[4..], wire_r::FENCE_TOTAL_LEN);
        st32(&mut v[OP_HEADER_LEN..], fence_ref);
        v
    };
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();

    // An update lands on the render domain's generation; the wait that follows
    // it is satisfied by that update rather than left pending, which is the
    // only way to tell the two directions apart from outside.
    handle_render_record(
        &mut state,
        &host,
        1,
        wire_r::OPCODE_UPDATE_FENCE,
        &fence(wire_r::OPCODE_UPDATE_FENCE, 4242),
        &mut out,
        &mut acc,
    );
    let after_update = fence_exec::execute_fence(
        &mut state,
        1,
        Domain::RenderFence,
        4242,
        crate::runtime::plan::event_sync::FenceAction::Wait,
    );
    assert!(
        !matches!(
            after_update,
            crate::runtime::fence_exec::FenceStatus::Missing
        ),
        "the update must have reached the render domain: {after_update:?}"
    );
    assert!(
        acc.pipeline_ref == 0 && acc.execute_icb.is_empty(),
        "a fence moves no stream state"
    );
}

/// Each ledger-settled render-encoder state record writes the one accumulator
/// field it names, and nothing else in the accumulator moves.
///
/// **This is the group's disjointness claim, made structurally rather than
/// argued.** W8 and W10 could say "this record touches no stream state at all";
/// the encoder's own records all land in `StreamAccum`, so that test separates
/// nothing here. What separates these fifteen is that each is the *sole writer*
/// of its field — so the fingerprint below is destructured out of the whole
/// accumulator (a field added to `StreamAccum` fails to compile until it is
/// named here) and exactly one entry is allowed to differ per record.
///
/// It replaces the per-arm assertions that moved with the legacy decoder's
/// state arms, and is stronger than they were: those checked that the field
/// they cared about got the right value, and none of them could see whether the
/// record had also moved a field belonging to a different class.
#[test]
fn each_render_state_record_writes_the_one_field_it_names_and_no_other() {
    use reims_vgpu_wire::ops::render as wire_r;

    /// Every field of `StreamAccum`, as text, in declaration order.
    ///
    /// Destructured rather than field-accessed so this cannot fall behind the
    /// struct: a new field is a compile error here, which is the point.
    fn fingerprint(acc: &StreamAccum) -> Vec<(&'static str, String)> {
        let StreamAccum {
            pipeline_ref,
            clears,
            color_slots,
            color_targets,
            draws,
            saw_draw,
            execute_icb,
            vertex_buffers,
            fragment_buffers,
            vertex_textures,
            fragment_textures,
            vertex_samplers,
            fragment_samplers,
            viewports,
            scissors,
            indexed,
            blend_color,
            cull_mode,
            front_facing,
            fill_mode,
            depth_clip_mode,
            line_width,
            depth_bias,
            depth_stencil_ref,
            stencil_ref,
            depth_attach,
            stencil_attach,
            visibility_buffer_ref,
            visibility,
            dropped_no_pipeline,
            dropped_zero_count,
            unrepresentable,
        } = acc;
        vec![
            ("pipeline_ref", format!("{pipeline_ref:?}")),
            ("clears", format!("{clears:?}")),
            ("color_slots", format!("{color_slots:?}")),
            ("color_targets", format!("{color_targets:?}")),
            ("draws", format!("{draws:?}")),
            ("saw_draw", format!("{saw_draw:?}")),
            ("execute_icb", format!("{execute_icb:?}")),
            ("vertex_buffers", format!("{vertex_buffers:?}")),
            ("fragment_buffers", format!("{fragment_buffers:?}")),
            ("vertex_textures", format!("{vertex_textures:?}")),
            ("fragment_textures", format!("{fragment_textures:?}")),
            ("vertex_samplers", format!("{vertex_samplers:?}")),
            ("fragment_samplers", format!("{fragment_samplers:?}")),
            ("viewports", format!("{viewports:?}")),
            ("scissors", format!("{scissors:?}")),
            ("indexed", format!("{indexed:?}")),
            ("blend_color", format!("{blend_color:?}")),
            ("cull_mode", format!("{cull_mode:?}")),
            ("front_facing", format!("{front_facing:?}")),
            ("fill_mode", format!("{fill_mode:?}")),
            ("depth_clip_mode", format!("{depth_clip_mode:?}")),
            ("line_width", format!("{line_width:?}")),
            ("depth_bias", format!("{depth_bias:?}")),
            ("depth_stencil_ref", format!("{depth_stencil_ref:?}")),
            ("stencil_ref", format!("{stencil_ref:?}")),
            ("depth_attach", format!("{depth_attach:?}")),
            ("stencil_attach", format!("{stencil_attach:?}")),
            (
                "visibility_buffer_ref",
                format!("{visibility_buffer_ref:?}"),
            ),
            ("visibility", format!("{visibility:?}")),
            ("dropped_no_pipeline", format!("{dropped_no_pipeline:?}")),
            ("dropped_zero_count", format!("{dropped_zero_count:?}")),
            ("unrepresentable", format!("{unrepresentable:?}")),
        ]
    }

    let record = |op: u32, total: u32, body: &[u8]| {
        let mut v = vec![0u8; total as usize];
        st32(&mut v[0..], op);
        st32(&mut v[4..], total);
        v[OP_HEADER_LEN..OP_HEADER_LEN + body.len()].copy_from_slice(body);
        v
    };
    let word = |value: u64| value.to_le_bytes().to_vec();
    let f32s = |vals: &[f32]| -> Vec<u8> {
        vals.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    };
    // The singular viewport is six `f64`; the singular scissor is four `u64`.
    let viewport_bytes =
        |v: [f64; 6]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>() };
    let scissor_bytes =
        |r: [u64; 4]| -> Vec<u8> { r.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>() };

    let mut vis = Vec::new();
    vis.extend_from_slice(&0x1234u64.to_le_bytes());
    vis.extend_from_slice(&2u64.to_le_bytes());

    // Two viewports and two scissors behind their plural heads, to prove the
    // plural forms reach the same field as the singular ones rather than a
    // second slot.
    let mut viewports_body = 2u32.to_le_bytes().to_vec();
    viewports_body.extend(viewport_bytes([1.0, 2.0, 3.0, 4.0, 0.0, 1.0]));
    viewports_body.extend(viewport_bytes([5.0, 6.0, 7.0, 8.0, 0.0, 1.0]));
    let mut scissors_body = 2u64.to_le_bytes().to_vec();
    scissors_body.extend(scissor_bytes([1, 2, 3, 4]));
    scissors_body.extend(scissor_bytes([5, 6, 7, 8]));

    let cases: Vec<(&str, Vec<u8>, &str)> = vec![
        (
            "setRenderPipelineState:",
            record(
                wire_r::OPCODE_SET_RENDER_PIPELINE_STATE,
                wire_r::SET_STATE_TOTAL_LEN,
                &7171u32.to_le_bytes(),
            ),
            "pipeline_ref",
        ),
        (
            "setDepthStencilState:",
            record(
                wire_r::OPCODE_SET_DEPTH_STENCIL_STATE,
                wire_r::SET_STATE_TOTAL_LEN,
                &6161u32.to_le_bytes(),
            ),
            "depth_stencil_ref",
        ),
        (
            "setStencilFrontReferenceValue:backReferenceValue:",
            record(
                wire_r::OPCODE_SET_STENCIL_REFERENCE,
                wire_r::SET_STENCIL_REFERENCE_TOTAL_LEN,
                &[11u32.to_le_bytes(), 22u32.to_le_bytes()].concat(),
            ),
            "stencil_ref",
        ),
        (
            "setBlendColorRed:green:blue:alpha:",
            record(
                wire_r::OPCODE_SET_BLEND_COLOR,
                wire_r::SET_BLEND_COLOR_TOTAL_LEN,
                &f32s(&[0.25, 0.5, 0.75, 1.0]),
            ),
            "blend_color",
        ),
        (
            "setCullMode:",
            record(
                wire_r::OPCODE_SET_CULL_MODE,
                wire_r::SET_MODE_TOTAL_LEN,
                &word(2),
            ),
            "cull_mode",
        ),
        (
            "setFrontFacingWinding:",
            record(
                wire_r::OPCODE_SET_FRONT_FACING,
                wire_r::SET_MODE_TOTAL_LEN,
                &word(1),
            ),
            "front_facing",
        ),
        (
            "setDepthClipMode:",
            record(
                wire_r::OPCODE_SET_DEPTH_CLIP_MODE,
                wire_r::SET_MODE_TOTAL_LEN,
                &word(1),
            ),
            "depth_clip_mode",
        ),
        (
            "setTriangleFillMode:",
            record(
                wire_r::OPCODE_SET_TRIANGLE_FILL_MODE,
                wire_r::SET_MODE_TOTAL_LEN,
                &word(1),
            ),
            "fill_mode",
        ),
        (
            "setDepthBias:slopeScale:clamp:",
            record(
                wire_r::OPCODE_SET_DEPTH_BIAS,
                wire_r::SET_DEPTH_BIAS_TOTAL_LEN,
                &f32s(&[0.25, 1.5, 2.25]),
            ),
            "depth_bias",
        ),
        (
            "setLineWidth:",
            record(
                wire_r::OPCODE_SET_LINE_WIDTH,
                wire_r::SET_FLOAT_TOTAL_LEN,
                &f32s(&[3.5]),
            ),
            "line_width",
        ),
        (
            "setViewport:",
            record(
                wire_r::OPCODE_SET_VIEWPORT,
                wire_r::SET_VIEWPORT_TOTAL_LEN,
                &viewport_bytes([1.0, 2.0, 3.0, 4.0, 0.0, 1.0]),
            ),
            "viewports",
        ),
        (
            "setViewports:count:",
            record(
                wire_r::OPCODE_SET_VIEWPORTS,
                (OP_HEADER_LEN + viewports_body.len()) as u32,
                &viewports_body,
            ),
            "viewports",
        ),
        (
            "setScissorRect:",
            record(
                wire_r::OPCODE_SET_SCISSOR,
                wire_r::SET_SCISSOR_TOTAL_LEN,
                &scissor_bytes([1, 2, 3, 4]),
            ),
            "scissors",
        ),
        (
            "setScissorRects:count:",
            record(
                wire_r::OPCODE_SET_SCISSOR_RECTS,
                (OP_HEADER_LEN + scissors_body.len()) as u32,
                &scissors_body,
            ),
            "scissors",
        ),
        (
            "setVisibilityResultMode:offset:",
            record(
                wire_r::OPCODE_SET_VISIBILITY_RESULT_MODE,
                wire_r::SET_VISIBILITY_RESULT_MODE_TOTAL_LEN,
                &vis,
            ),
            "visibility",
        ),
    ];

    for (selector, command, field) in cases {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        let before = fingerprint(&acc);
        let opcode = ld32(&command[0..]);
        handle_render_record(&mut state, &host, 1, opcode, &command, &mut out, &mut acc);
        let after = fingerprint(&acc);
        for ((name, was), (_, now)) in before.iter().zip(after.iter()) {
            if *name == field {
                assert_ne!(
                    was, now,
                    "{selector} ({opcode:#x}) left {name} at {was}, so the record reached no field"
                );
            } else {
                assert_eq!(
                    was, now,
                    "{selector} ({opcode:#x}) moved {name} from {was} to {now}, which belongs to \
                     another record class"
                );
            }
        }
    }
}

/// A plural viewport or scissor record of count zero keeps the previous state.
///
/// The singular form is the plural at length one, so an empty array can only be
/// the plural — and it is the one shape where "replace the state" and "the guest
/// bound none" are the same assignment for opposite reasons. The legacy decoder
/// refused it as `ErrBadLength`, a reading this keeps and names.
#[test]
fn a_plural_viewport_or_scissor_of_count_zero_keeps_what_the_stream_already_bound() {
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_wire::ops::render as wire_r;

    let record = |op: u32, body: &[u8]| {
        let total = (OP_HEADER_LEN + body.len()) as u32;
        let mut v = vec![0u8; total as usize];
        st32(&mut v[0..], op);
        st32(&mut v[4..], total);
        v[OP_HEADER_LEN..].copy_from_slice(body);
        v
    };
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    acc.viewports.push([1.0, 2.0, 3.0, 4.0, 0.0, 1.0]);
    acc.scissors.push(ScissorRect {
        x: 1,
        y: 2,
        width: 3,
        height: 4,
    });

    for (op, body, route) in [
        (
            wire_r::OPCODE_SET_VIEWPORTS,
            0u32.to_le_bytes().to_vec(),
            "render_viewport_count_zero",
        ),
        (
            wire_r::OPCODE_SET_SCISSOR_RECTS,
            0u64.to_le_bytes().to_vec(),
            "render_scissor_count_zero",
        ),
    ] {
        let before = store_route_count(route);
        let command = record(op, &body);
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
        assert_eq!(
            store_route_count(route),
            before + 1,
            "{op:#x} should have been counted as {route}"
        );
    }
    assert_eq!(acc.viewports.len(), 1, "the bound viewport must stand");
    assert_eq!(acc.scissors.len(), 1, "the bound scissor must stand");
}

/// Each of the twelve bind rows writes the one argument table its opcode names,
/// and none of the other five.
///
/// **The stage is the opcode and nothing else.** No wire field carries it, so a
/// record routed by layout rather than by kind would find no stage and bind
/// into whichever table it reached — which is the defect this asserts against
/// rather than describes. Six tables and twelve rows, so a single misrouted
/// arm shows up as a texture in the sampler table or a vertex bind in the
/// fragment one, both of which bind nothing the shader asked for and refuse
/// nothing either.
///
/// The two offset rows write a slot an earlier bind established, so the
/// accumulator is seeded before them: an offset naming an unbound slot lands on
/// nothing by design, and would make this test pass for the wrong reason.
#[test]
fn each_render_bind_row_writes_the_one_argument_table_its_opcode_names() {
    use reims_vgpu_wire::ops::render as wire_r;
    use reims_vgpu_wire::OP_HEADER_LEN as HEAD;

    // `first`, `count`, then `count` entries of `entry_len` bytes.
    let bind = |op: u32, first: u32, entry_len: usize, entry: &[u8]| {
        let total = HEAD + render_pass::BIND_ENTRIES + entry_len;
        let mut v = vec![0u8; total];
        st32(&mut v[0..], op);
        st32(&mut v[4..], total as u32);
        st32(&mut v[HEAD + render_pass::BIND_FIRST..], first);
        st32(&mut v[HEAD + render_pass::BIND_COUNT..], 1);
        v[HEAD + render_pass::BIND_ENTRIES..HEAD + render_pass::BIND_ENTRIES + entry.len()]
            .copy_from_slice(entry);
        v
    };
    let flat = |op: u32, body: &[u8]| {
        let total = HEAD + body.len();
        let mut v = vec![0u8; total];
        st32(&mut v[0..], op);
        st32(&mut v[4..], total as u32);
        v[HEAD..].copy_from_slice(body);
        v
    };

    let buffer_entry = {
        let mut e = vec![0u8; render_pass::BUFFER_BIND_ENTRY_SIZE];
        st32(&mut e[0..], 5151);
        st64(&mut e[4..], 0x100);
        e
    };
    let buffer_stride_entry = {
        let mut e = vec![0u8; render_pass::BUFFER_STRIDE_BIND_ENTRY_SIZE];
        st32(&mut e[0..], 5151);
        st64(&mut e[4..], 0x100);
        st64(&mut e[12..], 0x20);
        e
    };
    let ref_entry = {
        let mut e = vec![0u8; render_pass::REF_BIND_ENTRY_SIZE];
        st32(&mut e[0..], 0x51);
        e
    };
    let sampler_lod_entry = {
        let mut e = vec![0u8; render_pass::SAMPLER_LOD_BIND_ENTRY_SIZE];
        st32(&mut e[0..], 0x51);
        st32(&mut e[4..], 0.5f32.to_bits());
        st32(&mut e[8..], 4.0f32.to_bits());
        e
    };
    // `index`, then a 64-bit offset — and the strided form's stride after it.
    let offset_body = {
        let mut b = vec![0u8; 12];
        st32(&mut b[0..], 0);
        st64(&mut b[4..], 0x900);
        b
    };
    let offset_stride_body = {
        let mut b = vec![0u8; 20];
        st32(&mut b[0..], 0);
        st64(&mut b[4..], 0x900);
        st64(&mut b[12..], 0x30);
        b
    };

    const VB: &str = "vertex_buffers";
    const FB: &str = "fragment_buffers";
    const VT: &str = "vertex_textures";
    const FT: &str = "fragment_textures";
    const VS: &str = "vertex_samplers";
    const FS: &str = "fragment_samplers";
    const ALL: [&str; 6] = [VB, FB, VT, FT, VS, FS];

    let tables = |acc: &StreamAccum| -> Vec<(&'static str, String)> {
        vec![
            (VB, format!("{:?}", acc.vertex_buffers)),
            (FB, format!("{:?}", acc.fragment_buffers)),
            (VT, format!("{:?}", acc.vertex_textures)),
            (FT, format!("{:?}", acc.fragment_textures)),
            (VS, format!("{:?}", acc.vertex_samplers)),
            (FS, format!("{:?}", acc.fragment_samplers)),
        ]
    };

    for (selector, command, table, seed) in [
        (
            "setVertexBuffers:offsets:withRange:",
            bind(
                wire_r::OPCODE_SET_VERTEX_BUFFER,
                0,
                render_pass::BUFFER_BIND_ENTRY_SIZE,
                &buffer_entry,
            ),
            VB,
            false,
        ),
        (
            "setVertexBuffers:offsets:attributeStrides:withRange:",
            bind(
                wire_r::OPCODE_SET_VERTEX_BUFFER_STRIDE,
                0,
                render_pass::BUFFER_STRIDE_BIND_ENTRY_SIZE,
                &buffer_stride_entry,
            ),
            VB,
            false,
        ),
        (
            "setFragmentBuffers:offsets:withRange:",
            bind(
                wire_r::OPCODE_SET_FRAGMENT_BUFFER,
                0,
                render_pass::BUFFER_BIND_ENTRY_SIZE,
                &buffer_entry,
            ),
            FB,
            false,
        ),
        (
            "setVertexTextures:withRange:",
            bind(
                wire_r::OPCODE_SET_VERTEX_TEXTURE,
                0,
                render_pass::REF_BIND_ENTRY_SIZE,
                &ref_entry,
            ),
            VT,
            false,
        ),
        (
            "setFragmentTextures:withRange:",
            bind(
                wire_r::OPCODE_SET_FRAGMENT_TEXTURE,
                0,
                render_pass::REF_BIND_ENTRY_SIZE,
                &ref_entry,
            ),
            FT,
            false,
        ),
        (
            "setVertexSamplerStates:withRange:",
            bind(
                wire_r::OPCODE_SET_VERTEX_SAMPLER,
                0,
                render_pass::REF_BIND_ENTRY_SIZE,
                &ref_entry,
            ),
            VS,
            false,
        ),
        (
            "setFragmentSamplerStates:withRange:",
            bind(
                wire_r::OPCODE_SET_FRAGMENT_SAMPLER,
                0,
                render_pass::REF_BIND_ENTRY_SIZE,
                &ref_entry,
            ),
            FS,
            false,
        ),
        (
            "setVertexSamplerStates:lodMinClamps:lodMaxClamps:withRange:",
            bind(
                wire_r::OPCODE_SET_VERTEX_SAMPLER_LOD,
                0,
                render_pass::SAMPLER_LOD_BIND_ENTRY_SIZE,
                &sampler_lod_entry,
            ),
            VS,
            false,
        ),
        (
            "setFragmentSamplerStates:lodMinClamps:lodMaxClamps:withRange:",
            bind(
                wire_r::OPCODE_SET_FRAGMENT_SAMPLER_LOD,
                0,
                render_pass::SAMPLER_LOD_BIND_ENTRY_SIZE,
                &sampler_lod_entry,
            ),
            FS,
            false,
        ),
        (
            "setVertexBufferOffset:atIndex:",
            flat(wire_r::OPCODE_SET_VERTEX_BUFFER_OFFSET, &offset_body),
            VB,
            true,
        ),
        (
            "setVertexBufferOffset:attributeStride:atIndex:",
            flat(
                wire_r::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE,
                &offset_stride_body,
            ),
            VB,
            true,
        ),
        (
            "setFragmentBufferOffset:atIndex:",
            flat(wire_r::OPCODE_SET_FRAGMENT_BUFFER_OFFSET, &offset_body),
            FB,
            true,
        ),
    ] {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        if seed {
            // The slot the offset record moves. Both stages are seeded, so a
            // fragment offset reaching the vertex table would still be caught.
            let occupant = BufferBind {
                index: 0,
                buffer_ref: 5151,
                offset: 0x100,
                ..Default::default()
            };
            Arc::make_mut(&mut acc.vertex_buffers).push(occupant.clone());
            Arc::make_mut(&mut acc.fragment_buffers).push(occupant);
        }
        let before = tables(&acc);
        let opcode = ld32(&command[0..]);
        handle_render_record(&mut state, &host, 1, opcode, &command, &mut out, &mut acc);
        let after = tables(&acc);
        for ((name, was), (_, now)) in before.iter().zip(after.iter()) {
            if *name == table {
                assert_ne!(
                    was, now,
                    "{selector} ({opcode:#x}) left {name} at {was}, so the record reached no table"
                );
            } else {
                assert_eq!(
                    was, now,
                    "{selector} ({opcode:#x}) moved {name} from {was} to {now}, which its opcode \
                     does not name"
                );
            }
        }
        assert!(
            ALL.contains(&table),
            "the expected table must be one of the six"
        );
        assert!(
            acc.unrepresentable.is_none(),
            "{selector} ({opcode:#x}) refused the stream's draws"
        );
    }
}

/// A pass descriptor shorter than the record refuses instead of reading the
/// attachments that happen to fit.
///
/// **This is the reading W11c changed, and it is asserted in both directions.**
/// The legacy decoder accepted anything from depth + stencil + one colour slot
/// and answered the other seven slots as unattached — which is exactly what a
/// guest attaching one slot also produces, so a truncated descriptor and a
/// one-attachment pass were the same state. Apple's serializer writes all 592
/// bytes. The short form is now named under the row's own opcode and moves no
/// pass state; the full-length record with the same first attachment is
/// honoured, which is what says the refusal is about the length and not about
/// the bytes.
#[test]
fn a_pass_descriptor_short_of_its_record_is_refused_rather_than_read_at_the_offsets_that_fit() {
    use crate::runtime::render_pass::{PASS_ATTACH_TEXREF, PASS_COLOR_ATTACH_OFF};

    let pass = |payload_len: usize| {
        let mut payload = vec![0u8; payload_len];
        st32(
            &mut payload[PASS_COLOR_ATTACH_OFF + PASS_ATTACH_TEXREF..],
            4242,
        );
        let mut cmd = vec![0u8; OP_HEADER_LEN + payload.len()];
        st32(&mut cmd[0..], wire_pass::OPCODE_RENDER_PASS);
        st32(&mut cmd[4..], (OP_HEADER_LEN + payload.len()) as u32);
        cmd[OP_HEADER_LEN..].copy_from_slice(&payload);
        cmd
    };
    let run = |cmd: &[u8]| {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        handle_render_record(
            &mut state,
            &host,
            1,
            wire_pass::OPCODE_RENDER_PASS,
            cmd,
            &mut out,
            &mut acc,
        );
        acc
    };

    let full = wire_pass::RENDER_PASS_TOTAL_LEN as usize - OP_HEADER_LEN;
    let honoured = run(&pass(full));
    assert_eq!(
        honoured.color_slots.len(),
        1,
        "the full-length record's colour slot 0 must still reach the slot list"
    );

    // One byte short of the record. Every offset this attachment lives at is
    // still inside the buffer, which is precisely why the old reading could not
    // tell the two apart.
    let short = run(&pass(full - 1));
    assert!(
        short.color_slots.is_empty()
            && short.clears.is_empty()
            && short.color_targets.is_empty()
            && short.depth_attach.is_none()
            && short.stencil_attach.is_none(),
        "a short pass descriptor moved pass state: {short:?}"
    );
}

/// Each store-action override lands on the attachment its own target names.
///
/// The record's target is a `StoreActionTarget` and not an opcode read a second
/// time, so the fourth case the legacy arm carried — a store action for none of
/// the three — is unrepresentable. What is still worth asserting is that the
/// three that do exist do not reach each other: a colour override landing on the
/// depth attachment, or on the wrong colour slot, changes what the pass
/// publishes to guest pages with nothing on any channel to say so.
#[test]
fn each_store_action_override_reaches_the_attachment_its_target_names() {
    use crate::runtime::render_pass::{PASS_ATTACH_STORE_ACTION, PASS_ATTACH_TEXREF};
    use crate::runtime::render_pass::{PASS_COLOR_ATTACH_OFF, PASS_COLOR_ATTACH_STRIDE};
    use crate::runtime::render_pass::{PASS_DEPTH_ATTACH_OFF, PASS_STENCIL_ATTACH_OFF};
    use reims_vgpu_wire::ops::render as wire_r;

    // A pass declaring colour slots 0 and 3, a depth and a stencil attachment,
    // every one of them `DontCare` so an override to `Store` is visible.
    let mut payload = vec![0u8; wire_pass::RENDER_PASS_TOTAL_LEN as usize - OP_HEADER_LEN];
    for (base, texture_ref) in [
        (PASS_DEPTH_ATTACH_OFF, 44u32),
        (PASS_STENCIL_ATTACH_OFF, 55),
        (PASS_COLOR_ATTACH_OFF, 41),
        (PASS_COLOR_ATTACH_OFF + 3 * PASS_COLOR_ATTACH_STRIDE, 43),
    ] {
        st32(&mut payload[base + PASS_ATTACH_TEXREF..], texture_ref);
        st16(&mut payload[base + PASS_ATTACH_STORE_ACTION..], 0);
    }
    let mut descriptor = vec![0u8; OP_HEADER_LEN + payload.len()];
    st32(&mut descriptor[0..], wire_pass::OPCODE_RENDER_PASS);
    st32(&mut descriptor[4..], (OP_HEADER_LEN + payload.len()) as u32);
    descriptor[OP_HEADER_LEN..].copy_from_slice(&payload);

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    handle_render_record(
        &mut state,
        &host,
        1,
        wire_pass::OPCODE_RENDER_PASS,
        &descriptor,
        &mut out,
        &mut acc,
    );
    assert_eq!(acc.color_slots.len(), 2, "the pass declared slots 0 and 3");
    assert!(acc.depth_attach.is_some() && acc.stencil_attach.is_some());

    // `setColorStoreAction:atIndex:` — a 32-bit action then the slot index.
    let colour = |action: u32, index: u32| {
        let mut v = vec![0u8; wire_r::SET_COLOR_STORE_ACTION_TOTAL_LEN as usize];
        st32(&mut v[0..], wire_r::OPCODE_SET_COLOR_STORE_ACTION);
        st32(&mut v[4..], wire_r::SET_COLOR_STORE_ACTION_TOTAL_LEN);
        st32(&mut v[OP_HEADER_LEN..], action);
        st32(&mut v[OP_HEADER_LEN + 4..], index);
        v
    };
    // The depth and stencil forms carry no index: one 64-bit action.
    let bare = |op: u32, action: u64| {
        let mut v = vec![0u8; wire_r::SET_MODE_TOTAL_LEN as usize];
        st32(&mut v[0..], op);
        st32(&mut v[4..], wire_r::SET_MODE_TOTAL_LEN);
        st64(&mut v[OP_HEADER_LEN..], action);
        v
    };

    // `setColorStoreAction:atIndex:` naming slot 3, then the depth and stencil
    // forms — all three against one pass that declares all four attachments, so
    // an override reaching the wrong one is visible rather than merely absent.
    for command in [
        colour(MTL_STORE_ACTION_STORE.into(), 3),
        bare(
            wire_r::OPCODE_SET_DEPTH_STORE_ACTION,
            MTL_STORE_ACTION_STORE.into(),
        ),
        bare(
            wire_r::OPCODE_SET_STENCIL_STORE_ACTION,
            MTL_STORE_ACTION_STORE.into(),
        ),
    ] {
        let opcode = ld32(&command[0..]);
        handle_render_record(&mut state, &host, 1, opcode, &command, &mut out, &mut acc);
    }

    let slot0 = acc.color_slots.iter().find(|(s, _)| *s == 0).unwrap().1;
    let slot3 = acc.color_slots.iter().find(|(s, _)| *s == 3).unwrap().1;
    assert_eq!(
        slot0.store_action, 0,
        "the colour override named slot 3 and reached slot 0"
    );
    assert_eq!(
        slot3.store_action, MTL_STORE_ACTION_STORE,
        "the colour override did not reach the slot its index names"
    );
    assert_eq!(
        acc.depth_attach.unwrap().store_action,
        MTL_STORE_ACTION_STORE,
        "the depth override did not reach the depth attachment"
    );
    assert_eq!(
        acc.stencil_attach.unwrap().store_action,
        MTL_STORE_ACTION_STORE,
        "the stencil override did not reach the stencil attachment"
    );
}

/// A draw that carries no instance-count argument and one that carries zero are
/// different draws.
///
/// **This is the reading W11d changed.** `drawPrimitives:vertexStart:vertexCount:`
/// has no `instanceCount:` argument at all, and Metal's default for it is one
/// instance — so one is what the guest asked for. `instanceCount:0` is a guest
/// that wrote the argument and wrote zero, which draws nothing; that is also
/// what it asked for, and it reaches the backend.
///
/// The `.max(1)` this replaces sat in the decoder and applied to both, so the
/// two were one number by the time anything downstream saw them. Three arms of
/// this device disagreed about the zero — `backend::metal::render` refuses it by
/// name, a refusal the clamp made unreachable, and `runtime::icb` passed the
/// same argument to Metal unclamped — so the clamp could not have been
/// describing the contract. The census survives at the lift and is asserted
/// here, because a zero reaching a backend is exactly the event worth a line.
#[test]
fn a_draw_with_no_instance_count_and_one_asking_for_zero_are_not_the_same_draw() {
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_wire::ops::render as wire_r;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    acc.pipeline_ref = 7171;

    // `drawPrimitives:vertexStart:vertexCount:` — primitive first at 32 bits,
    // then two 16-bit counts. No instance count on the wire at all.
    let mut plain = vec![0u8; wire_r::DRAW_TOTAL_LEN as usize];
    st32(&mut plain[0..], wire_r::OPCODE_DRAW);
    st32(&mut plain[4..], wire_r::DRAW_TOTAL_LEN);
    st32(&mut plain[OP_HEADER_LEN..], 3);
    st16(&mut plain[OP_HEADER_LEN + 4..], 0);
    st16(&mut plain[OP_HEADER_LEN + 6..], 6);
    handle_render_record(
        &mut state,
        &host,
        1,
        wire_r::OPCODE_DRAW,
        &plain,
        &mut out,
        &mut acc,
    );
    assert_eq!(acc.draws.len(), 1, "the plain draw was not recorded");
    assert_eq!(
        acc.draws[0].draw.instance_count, 1,
        "a selector with no instanceCount: argument draws one instance"
    );

    // `drawPrimitives:vertexStart:vertexCount:instanceCount:` with the argument
    // written as zero.
    let mut zero = vec![0u8; wire_r::DRAW_INSTANCED_TOTAL_LEN as usize];
    st32(&mut zero[0..], wire_r::OPCODE_DRAW_INSTANCED);
    st32(&mut zero[4..], wire_r::DRAW_INSTANCED_TOTAL_LEN);
    st16(&mut zero[OP_HEADER_LEN..], 0);
    st16(&mut zero[OP_HEADER_LEN + 2..], 6);
    st16(&mut zero[OP_HEADER_LEN + 4..], 0);
    st16(&mut zero[OP_HEADER_LEN + 6..], 3);
    let before = store_route_count("draw_instance_count_zero");
    handle_render_record(
        &mut state,
        &host,
        1,
        wire_r::OPCODE_DRAW_INSTANCED,
        &zero,
        &mut out,
        &mut acc,
    );
    assert_eq!(
        store_route_count("draw_instance_count_zero"),
        before + 1,
        "a written zero instance count was not counted"
    );
    assert_eq!(
        acc.draws.len(),
        2,
        "the zero-instance draw was not recorded"
    );
    assert_eq!(
        acc.draws[1].draw.instance_count, 0,
        "a guest that wrote instanceCount:0 had its argument overruled"
    );
}

/// The compact and wide encodings of one draw shape reach the same record.
///
/// Six selectors arrive in two encodings — 16-bit counts and 64-bit ones — with
/// different field orders, and that is an encoding rather than a meaning. Both
/// must produce the same `PendingDraw`, or a guest whose count crossed 65535
/// would draw different geometry than the same guest below it. The one field
/// where they genuinely disagree, `base_vertex`, is not exercised here: the
/// compact form's is truncated to sixteen bits by Apple's serializer upstream of
/// this device, so equal inputs are not the right comparison for it.
#[test]
fn a_compact_draw_and_its_wide_encoding_record_the_same_draw() {
    use reims_vgpu_wire::ops::render as wire_r;

    let run = |command: &[u8]| {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        acc.pipeline_ref = 7171;
        let opcode = ld32(&command[0..]);
        handle_render_record(&mut state, &host, 1, opcode, command, &mut out, &mut acc);
        acc.draws.first().map(|d| d.draw)
    };

    // Compact: primitive at 32 bits, then `vertexStart` and `vertexCount` at 16.
    let mut compact = vec![0u8; wire_r::DRAW_TOTAL_LEN as usize];
    st32(&mut compact[0..], wire_r::OPCODE_DRAW);
    st32(&mut compact[4..], wire_r::DRAW_TOTAL_LEN);
    st32(&mut compact[OP_HEADER_LEN..], 3);
    st16(&mut compact[OP_HEADER_LEN + 4..], 12);
    st16(&mut compact[OP_HEADER_LEN + 6..], 900);

    // Wide: the same three fields, the counts at 64 bits.
    let mut wide = vec![0u8; wire_r::DRAW_WIDE_TOTAL_LEN as usize];
    st32(&mut wide[0..], wire_r::OPCODE_DRAW_WIDE);
    st32(&mut wide[4..], wire_r::DRAW_WIDE_TOTAL_LEN);
    st32(&mut wide[OP_HEADER_LEN..], 3);
    st64(&mut wide[OP_HEADER_LEN + 4..], 12);
    st64(&mut wide[OP_HEADER_LEN + 12..], 900);

    let a = run(&compact).expect("the compact draw was not recorded");
    let b = run(&wide).expect("the wide draw was not recorded");
    assert_eq!(
        a, b,
        "the two encodings of one draw shape produced different draws"
    );
    assert_eq!(
        (a.primitive_type, a.first_vertex, a.vertex_count),
        (3, 12, 900)
    );
    assert_eq!(
        a.instance_count, 1,
        "neither encoding carries an instance count, so both draw one"
    );
}

/// A sink whose segment pairing is broken names the loss instead of absorbing
/// it.
///
/// The two cases here are unreachable from `walk_stream`, which pairs its own
/// calls — and that is exactly why they are worth a test. The sink exists so
/// that a *different* walker can drive it, and the first thing a different
/// walker can get wrong is the pairing. Both losses are whole compute
/// dispatches, so neither may be silent.
#[test]
fn a_compute_record_with_no_segment_open_is_a_named_loss() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    let task_id = 0x5731_0101;
    let before = sink_body().len();
    let mut sink = StreamSink {
        state: &mut state,
        host: &mut host,
        task_id,
        out: &mut out,
        acc: &mut acc,
        compute: None,
    };
    // No `begin_segment`. The record's bytes are never read, because the sink
    // refuses before reaching the handler.
    sink.record(SegmentKind::Compute, 0x1234, &[0u8; OP_HEADER_LEN]);
    let added = sink_body().split_off(before);
    assert!(
        added.contains("exec_compute_record_unopened")
            && added.contains(&format!("task={task_id}")),
        "a compute record with no open segment must name the lost dispatch: {added}"
    );
}

#[test]
fn a_segment_opened_over_an_open_compute_segment_commits_it_and_says_so() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    let task_id = 0x5731_0102;
    let before = sink_body().len();
    let mut sink = StreamSink {
        state: &mut state,
        host: &mut host,
        task_id,
        out: &mut out,
        acc: &mut acc,
        compute: None,
    };
    sink.begin_segment(SegmentKind::Compute);
    assert!(sink.compute.is_some(), "a compute segment opens an encoder");
    sink.begin_segment(SegmentKind::Render);
    assert!(
        sink.compute.is_none(),
        "the unended compute segment is committed rather than carried into the render one"
    );
    let added = sink_body().split_off(before);
    assert!(
        added.contains("exec_segment_unended") && added.contains(&format!("task={task_id}")),
        "opening a segment over an open compute segment must be named: {added}"
    );
}

/// A submission is executed from the buffers it was read with.
///
/// **The property the replacement architecture's parking rests on.** An
/// admitted packet may be held — behind a pipeline still compiling, or behind a
/// completion word another packet owes — and run later. If execution re-read
/// the command buffers out of guest memory at that point, it would run whatever
/// the guest had written there in the meantime, which is neither the stream the
/// device judged nor a stream the guest asked to run twice.
///
/// The test states it the only way that distinguishes the two: the guest's page
/// is overwritten with zeroes between the read and the execution. A device that
/// re-read would signal nothing.
#[test]
fn a_submission_executes_the_streams_it_was_read_with_and_not_guest_memory_again() {
    use crate::model::FENCE_DOMAIN_EVENT;
    use crate::protocol::fifo::{
        CHILD_EXEC_INDIRECT_CMDBUF_GVA, CHILD_EXEC_INDIRECT_CMDBUF_LENGTH,
    };
    use crate::runtime::gva_mem::{define_task_pages_arm64e, write_task_gva_arm64e};
    use reims_vgpu_protocol::segment::{SegmentKind, SEGMENT_HEADER_LEN};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    define_task_pages_arm64e(&mut host, &mut state, 4, 16);

    // One event segment carrying one signal: the smallest record whose effect
    // is a value this test can read back out of the model.
    let (event_ref, value) = (21u32, 4u64);
    let mut record = vec![0u8; 0x14];
    st32(&mut record[0..4], 0x191);
    st32(&mut record[4..8], 0x14);
    st32(&mut record[8..12], event_ref);
    st64(&mut record[12..20], value);
    let mut stream = vec![0u8; SEGMENT_HEADER_LEN];
    st32(
        &mut stream[0..4],
        (SEGMENT_HEADER_LEN + record.len()) as u32,
    );
    stream[4] = SegmentKind::Event.wire_type();
    stream.extend_from_slice(&record);

    let stream_gva = 8u64 << PAGE_SHIFT_ARM64E;
    write_task_gva_arm64e(&mut host, &state.tasks[1], stream_gva, &stream);

    let mut payload = vec![
        0u8;
        CHILD_EXEC_INDIRECT_HEADER_LEN as usize
            + CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN as usize
    ];
    st32(&mut payload[CHILD_EXEC_INDIRECT_TASK_ID as usize..], 1);
    st32(&mut payload[CHILD_EXEC_INDIRECT_CMDBUF_COUNT as usize..], 1);
    let cb = CHILD_EXEC_INDIRECT_HEADER_LEN as usize;
    st64(
        &mut payload[cb + CHILD_EXEC_INDIRECT_CMDBUF_GVA as usize..],
        stream_gva,
    );
    st64(
        &mut payload[cb + CHILD_EXEC_INDIRECT_CMDBUF_LENGTH as usize..],
        stream.len() as u64,
    );

    let mut out = ExecResult::default();
    let mut measured_ns = 0u64;
    let submission = read_submission(&state, &host, &payload, &mut out, &mut measured_ns)
        .expect("the packet names a live task and one command buffer");
    assert_eq!(
        submission.streams.len(),
        1,
        "the header declares one command buffer and it is read here, once"
    );
    assert_eq!(out.streams_loaded, 1);
    assert_eq!(
        state.fence_generation(1, FENCE_DOMAIN_EVENT, event_ref),
        None,
        "reading a submission runs none of it"
    );

    // The guest reuses the page under a packet this device is still holding,
    // and the read-back is what keeps this test from passing vacuously: a
    // silently ineffective overwrite would leave the original stream in guest
    // memory and both readings would signal.
    write_task_gva_arm64e(
        &mut host,
        &state.tasks[1],
        stream_gva,
        &vec![0u8; stream.len()],
    );
    let mut readback = vec![0xffu8; stream.len()];
    crate::runtime::gva_mem::read_task_gva_by_id(
        &host,
        &state.tasks,
        1,
        stream_gva,
        &mut readback,
        PAGE_SHIFT_ARM64E,
    )
    .expect("the page is mapped");
    assert!(
        readback.iter().all(|b| *b == 0),
        "the overwrite must have landed, or this test proves nothing"
    );

    execute_submission(
        &mut state,
        &mut host,
        &submission,
        None,
        &mut out,
        &mut measured_ns,
    );
    assert_eq!(
        state.fence_generation(1, FENCE_DOMAIN_EVENT, event_ref),
        Some(value),
        "the signal the read submission carries is the one that runs, whatever \
         the guest has since written where it came from"
    );
}

/// The plan step answers `false` for a pipeline whose inputs this device cannot
/// load, because a missing plan input is deterministic and not asynchronous
/// work.
///
/// **What "plan" means, asserted at the seam.** `preflight_submission` is the
/// middle of read / plan / execute and it exists as its own step because it is
/// the only one a caller may run at a moment of its own choosing — it takes a
/// shared `&DeviceState`, mutates nothing the guest can see, and is a function
/// of bytes the submission already holds. `true` therefore has exactly one
/// meaning: *a translation is running and will finish*. A pipeline whose AIR
/// cannot be loaded at all is not that — normal execution reports it precisely
/// — and answering `true` for one would defer the packet forever.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_pipeline_whose_inputs_cannot_load_is_not_a_pending_translation() {
    use reims_vgpu_protocol::segment::{SegmentKind, SEGMENT_HEADER_LEN};
    use wire_render::OPCODE_SET_RENDER_PIPELINE_STATE;

    let mut records = [0u8; 12];
    st32(&mut records[0..4], OPCODE_SET_RENDER_PIPELINE_STATE);
    st32(&mut records[4..8], 12);
    st32(&mut records[8..12], 41);
    let mut stream = vec![0u8; SEGMENT_HEADER_LEN];
    let stream_len = stream.len() + records.len();
    st32(&mut stream[0..4], stream_len as u32);
    stream[4] = SegmentKind::Render.wire_type();
    stream.extend_from_slice(&records);

    let state = DeviceState::new(crate::model::DeviceId(1), 12);
    let host = crate::runtime::host::FakeHost::new();
    let submission = super::ExecSubmission::stated(1, vec![stream]);
    let mut measured_ns = 0u64;
    // The transaction the walk would have built for that record, stated here
    // because this test drives the rail directly rather than through admission.
    let mut builder = reims_vgpu_core::exec::ExecBuilder::new();
    builder
        .begin_encoder(
            reims_vgpu_protocol::segment::SegmentKind::Render,
            reims_vgpu_protocol::segment::SegmentLifetime::SELF_CONTAINED,
        )
        .expect("a fresh encoder");
    builder
        .record(
            reims_vgpu_core::exec::ResolvedOperation::Render(
                reims_vgpu_core::render::RenderOp::SetPipeline {
                    pipeline: reims_vgpu_core::identity::ResourceId {
                        slot: reims_vgpu_core::identity::ObjectListRef(41),
                        generation: reims_vgpu_core::identity::SlotGeneration(1),
                    },
                },
            ),
            &mut |_: &reims_vgpu_core::access::Participation| {
                unreachable!("setting a pipeline names no memory")
            },
        )
        .expect("the record places");
    builder.end_segment().expect("the encoder ends");
    let resolved = builder.finish().expect("the transaction finishes");
    assert_eq!(resolved.render_pipeline_leases().len(), 1);

    assert!(
        super::preflight_submission(&state, &host, &submission, &resolved, &mut measured_ns)
            .is_empty(),
        "no object list, so pipeline 41 has no AIR to await"
    );
    // And it is a function of its inputs: asked again, the same answer.
    assert!(
        super::preflight_submission(&state, &host, &submission, &resolved, &mut measured_ns)
            .is_empty()
    );
}

/// The two walks over one stream reach the same records in the same order.
///
/// This is the whole of [`ResolvedCursor`]'s claim, and it is worth a test
/// rather than a comment: the model resolves a packet's records at ingress and
/// this executor walks the same bytes again when the model releases it, so
/// "the *n*th record here is the *n*th record there" is the only thing that
/// lets the second walk stop decoding. A framer that counted a protection
/// envelope, a segment continuation or a refused record differently on one side
/// than the other would put the cursor off by one, and every later record would
/// be read as the wrong operation rather than as a refusal.
///
/// A mixed stream on purpose — three encoder families, a continuation and an
/// envelope — because a stream of one family could not see a divergence that
/// only a boundary produces.
#[test]
fn the_executing_walk_and_the_resolving_walk_reach_the_same_records() {
    use reims_vgpu_core::access::{AccessRefusal, BackingId, Participation, ResourceKey};
    use reims_vgpu_core::identity::{ChannelId, ObjectListRef, ResourceId, SlotGeneration};
    use reims_vgpu_core::resolve::RefResolver;
    use reims_vgpu_protocol::segment::{SegmentKind, SEGMENT_HEADER_LEN};
    use reims_vgpu_protocol::sync::{OPCODE_SIGNAL_EVENT, OPCODE_WAIT_EVENT};

    struct Everything;
    impl RefResolver for Everything {
        fn resource(&self, object_ref: u32) -> Option<ResourceId> {
            Some(ResourceId {
                slot: ObjectListRef(object_ref),
                generation: SlotGeneration(1),
            })
        }
    }

    fn record(buf: &mut Vec<u8>, opcode: u32, payload: &[u8]) {
        let len = (OP_HEADER_LEN + payload.len()) as u32;
        let mut hdr = [0u8; 8];
        st32(&mut hdr[0..4], opcode);
        st32(&mut hdr[4..8], len);
        buf.extend_from_slice(&hdr);
        buf.extend_from_slice(payload);
    }

    /// The two encoder-lifetime bytes are written from both ends of the edge,
    /// exactly as the serializer writes them: the segment that continues sets
    /// `continues_previous`, and the one it continues sets `continues_into_next`
    /// — a continuation whose predecessor did not say so is refused, which is
    /// how this fixture first found out it was writing only one of the two.
    fn segment(buf: &mut Vec<u8>, type_: u8, lifetime: (bool, bool), payload: &[u8]) {
        let len = (SEGMENT_HEADER_LEN + payload.len()) as u32;
        let mut hdr = [0u8; 8];
        st32(&mut hdr[0..4], len);
        hdr[4] = type_;
        hdr[5] = u8::from(lifetime.0);
        hdr[6] = u8::from(lifetime.1);
        buf.extend_from_slice(&hdr);
        buf.extend_from_slice(payload);
    }

    let mut blit = Vec::new();
    record(
        &mut blit,
        reims_vgpu_wire::ops::blit::OPCODE_GENERATE_MIPMAPS,
        &7u32.to_le_bytes(),
    );
    record(
        &mut blit,
        reims_vgpu_wire::ops::blit::OPCODE_GENERATE_MIPMAPS,
        &9u32.to_le_bytes(),
    );

    let event_body = core::mem::size_of::<reims_vgpu_wire::ops::event::SignalWait>();
    let mut events = Vec::new();
    for (opcode, value) in [(OPCODE_SIGNAL_EVENT, 7u64), (OPCODE_WAIT_EVENT, 7)] {
        let mut payload = vec![0u8; event_body];
        st32(&mut payload[0..4], 11);
        st64(&mut payload[4..12], value);
        record(&mut events, opcode, &payload);
    }

    let mut stream = Vec::new();
    segment(
        &mut stream,
        SegmentKind::Blit.wire_type(),
        (false, true),
        &blit,
    );
    // A continuation: one encoder across two segments, which the model records
    // into a single `ResolvedStream` and this executor visits as two segments.
    // The flat order is what has to agree, and this is the case that proves it
    // is the flat order and not a per-segment index.
    segment(
        &mut stream,
        SegmentKind::Blit.wire_type(),
        (true, false),
        &blit,
    );
    segment(
        &mut stream,
        SegmentKind::Event.wire_type(),
        (false, false),
        &events,
    );

    let mut source = |part: &Participation| -> Result<_, AccessRefusal> {
        Ok(part.resolve(
            ChannelId(2),
            ResourceKey {
                backing: BackingId(u64::from(part.resource.slot.0)),
                heap: None,
            },
            None,
            None,
        ))
    };
    let work = reims_vgpu_core::walk::exec(
        &stream,
        &Everything,
        &mut source,
        reims_vgpu_core::exec::ExecBuilder::new(),
    )
    .expect("every record in the fixture resolves");
    assert_eq!(work.record_count(), 6, "the fixture's own record count");

    // The executor's walk, driven for its opcodes only: what is under test is
    // the correspondence, and running the handlers would need a whole host.
    let mut cursor = ResolvedCursor::new(&work);
    let mut seen = Vec::new();
    for framed in
        reims_vgpu_protocol::segment::SegmentStream::new(&stream).expect("the fixture frames")
    {
        let framed = framed.expect("the fixture frames");
        let reims_vgpu_protocol::segment::SegmentBody::Encoder { kind, commands } = framed.body
        else {
            continue;
        };
        for op in reims_vgpu_wire::op::OpStream::new(commands) {
            let op = op.expect("the fixture frames");
            seen.push(
                cursor
                    .step(kind, op.opcode())
                    .unwrap_or_else(|| {
                        panic!(
                            "record {} of the executing walk found no resolved twin",
                            seen.len()
                        )
                    })
                    .class(),
            );
        }
        cursor.end_segment();
    }

    use reims_vgpu_core::operation::OperationClass;
    assert_eq!(
        seen,
        vec![
            OperationClass::Blit,
            OperationClass::Blit,
            OperationClass::Blit,
            OperationClass::Blit,
            OperationClass::Event,
            OperationClass::Event,
        ],
        "the order agrees across the continuation and the family boundary"
    );

    // A transaction missing one record reads as **one** absent twin and not as
    // a shift. This is the whole reason the cursor is keyed by position: a
    // count-based cursor would hand every record after the gap its
    // predecessor's answer, and the last one would fall off the end — six
    // wrong readings from one omission, each of them plausible.
    let mut gapped = work.clone();
    let removed = gapped.streams[0].records.remove(1);
    assert_eq!(
        removed.at,
        reims_vgpu_core::stream::StreamPosition {
            segment: 0,
            record: 1,
        },
    );
    let mut cursor = ResolvedCursor::new(&gapped);
    let mut found = Vec::new();
    for framed in
        reims_vgpu_protocol::segment::SegmentStream::new(&stream).expect("the fixture frames")
    {
        let framed = framed.expect("the fixture frames");
        let reims_vgpu_protocol::segment::SegmentBody::Encoder { kind, commands } = framed.body
        else {
            continue;
        };
        for op in reims_vgpu_wire::op::OpStream::new(commands) {
            let op = op.expect("the fixture frames");
            found.push(cursor.step(kind, op.opcode()).is_some());
        }
        cursor.end_segment();
    }
    assert_eq!(
        found,
        vec![true, false, true, true, true, true],
        "exactly the removed record has no twin; everything after it still finds its own"
    );
}

/// A submission's bytes are framed once per stream, and the plan step frames
/// none of them.
///
/// Seam 6's "zero re-scans of already resolved EXEC bytes" was unmeasured for
/// as long as nothing counted the framings. The violation it names had a
/// specific shape on this rail: two pre-scans walked every stream through the
/// same framer looking for `SetPipeline` records before execution walked it
/// again, so a boot's segment census read three times the truth and the
/// leases the rail was asked about were a different set from the ones
/// admission readied. `preflight_submission` answers from
/// `reims_vgpu_core::exec::ExecWork` now — the walk's own answer — so the
/// measurable statement is that it adds no framing at all.
///
/// Two streams rather than one, because a per-*submission* framer would pass a
/// one-stream fixture and be a re-scan on every real packet.
#[test]
fn a_submission_frames_each_stream_once_and_preflight_frames_none() {
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_protocol::segment::{SegmentKind, SEGMENT_HEADER_LEN};
    use reims_vgpu_protocol::sync::OPCODE_SIGNAL_EVENT;

    // One event segment carrying one signal: enough to be walked, and no rail
    // work at all, so the count is about the framer and not about a backend.
    let signal_len = core::mem::size_of::<reims_vgpu_wire::ops::event::SignalWait>();
    let event_stream = |event_ref: u32, value: u64| {
        let mut records = vec![0u8; OP_HEADER_LEN + signal_len];
        st32(&mut records[0..4], OPCODE_SIGNAL_EVENT);
        st32(&mut records[4..8], (OP_HEADER_LEN + signal_len) as u32);
        st32(&mut records[OP_HEADER_LEN..], event_ref);
        st64(&mut records[OP_HEADER_LEN + 4..], value);
        let mut stream = vec![0u8; SEGMENT_HEADER_LEN];
        st32(
            &mut stream[0..4],
            (SEGMENT_HEADER_LEN + records.len()) as u32,
        );
        stream[4] = SegmentKind::Event.wire_type();
        stream.extend_from_slice(&records);
        stream
    };

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let submission = ExecSubmission::stated(9, vec![event_stream(11, 7), event_stream(12, 3)]);
    let resolved = reims_vgpu_core::exec::ExecWork::default();

    let before = store_route_count("exec_stream_framed");
    let mut measured_ns = 0u64;
    let pending = preflight_submission(&state, &host, &submission, &resolved, &mut measured_ns);
    assert!(pending.is_empty(), "the fixture resolves no leases");
    assert_eq!(
        store_route_count("exec_stream_framed"),
        before,
        "the plan step answers from the resolved work, never from the packet's bytes"
    );

    // The task id rides `ExecResult` from the *read* half, which this fixture
    // states rather than performs, so what says the run happened is the two
    // generations below and not a field nothing here set.
    let _ = execute_planned(
        &mut state,
        &mut host,
        RetainedInputs {
            submission: &submission,
            resolved: &resolved,
        },
        ExecResult::default(),
    );
    assert_eq!(
        store_route_count("exec_stream_framed") - before,
        2,
        "two streams, framed once each: one more is a re-scan"
    );
    // Both signals landed, so the two framings were two executions and not one
    // execution counted twice.
    assert_eq!(
        state.fence_generation(9, crate::model::FENCE_DOMAIN_EVENT, 11),
        Some(7)
    );
    assert_eq!(
        state.fence_generation(9, crate::model::FENCE_DOMAIN_EVENT, 12),
        Some(3)
    );
}
