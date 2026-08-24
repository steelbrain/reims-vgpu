// Only the compute-preflight test names these opcodes, and that test is
// Vulkan-only — this device compiles no compute preflight without it.
use reims_vgpu_wire::ops::compute as wire_compute;

use reims_vgpu_wire::OP_HEADER_LEN;

use super::*;
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
use crate::runtime::decode::render::{
    PASS_ATTACH_CLEAR_COLOR, PASS_ATTACH_LOAD_ACTION, PASS_ATTACH_STORE_ACTION, PASS_ATTACH_TEXREF,
    PASS_COLOR_ATTACH_OFF, PASS_COLOR_ATTACH_STRIDE,
};
use crate::runtime::host::FakeHost;
use reims_vgpu_core::endian::{st16, st32, st64};
use reims_vgpu_protocol::pass_action::{MTL_LOAD_ACTION_CLEAR, MTL_STORE_ACTION_STORE};

#[test]
fn a_batched_resident_failure_preserves_the_exact_successful_prefix() {
    let first_identity = crate::model::TargetIdentity::Gva {
        gva: 0x4000,
        width: 8,
        height: 8,
        generation: 3,
        format: reims_vgpu_core::pixel_format::TexelLayout::Rgba8,
    };
    let records = vec![
        PreparedResidentRecord {
            req: draw::DrawEncodeRequest::default(),
            pipeline_ref: 7,
            icb: false,
            draw_index: 2,
        },
        PreparedResidentRecord {
            req: draw::DrawEncodeRequest::default(),
            pipeline_ref: 8,
            icb: false,
            draw_index: 3,
        },
    ];
    let progress = draw::PreparedM2vProgress {
        completed: vec![
            draw::M2vDrawSpan::ResidentChain {
                submission: reims_vgpu_protocol::SubmissionId::new(1),
                identity: first_identity.clone(),
                visibility_samples: None,
            },
            draw::M2vDrawSpan::None,
        ],
        failure: None,
    };
    let mut out = ExecResult::default();
    let mut visibility = std::collections::BTreeMap::new();

    let failure = apply_prepared_resident_prefix(&mut out, &records, progress, &mut visibility, 5)
        .expect_err("the second completion is a refusal");

    assert_eq!(failure, (Some(first_identity), 1));
    assert_eq!(out.draws_ok, 1);
    assert_eq!(out.draws_fail, 1);
}

#[test]
fn a_malformed_compute_barrier_blocks_the_later_dispatch() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut seg = crate::runtime::compute_session::ComputeSegment::default();

    handle_compute_record(
        &mut state,
        &mut host,
        1,
        wire_compute::OPCODE_MEMORY_BARRIER_SCOPE,
        &[0; 4],
        &mut out,
        &mut seg,
    );

    let mut dispatch = crate::runtime::decode::compute::Command::default();
    dispatch.kind = ComputeKind::DispatchThreadgroups;
    assert_eq!(
        compute_exec::apply_record(&mut state, &mut host, 1, &dispatch, &mut seg),
        Some(ComputeStatus::Unsupported(
            "compute_barrier_decode_malformed"
        ))
    );
}

#[test]
fn compute_fences_reach_the_dependency_accumulator_from_the_stream_walker() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut seg = crate::runtime::compute_session::ComputeSegment::default();

    let command = |opcode| {
        let mut bytes = [0u8; wire_compute::FENCE_TOTAL_LEN as usize];
        st32(&mut bytes[0..4], opcode);
        st32(&mut bytes[4..8], wire_compute::FENCE_TOTAL_LEN);
        st32(&mut bytes[8..12], 41);
        bytes
    };
    for opcode in [
        wire_compute::OPCODE_UPDATE_FENCE,
        wire_compute::OPCODE_WAIT_FOR_FENCE,
    ] {
        let bytes = command(opcode);
        handle_compute_record(&mut state, &mut host, 1, opcode, &bytes, &mut out, &mut seg);
    }

    assert_eq!(state.fence_generation(1, 41), Some(1));
    assert_eq!(
        seg.pending_barriers,
        vec![reims_vgpu_core::ComputeBarrier::Fence]
    );
}

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
/// (`draws_fail`) stays 0 on this path because the draw encoded
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
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let r = process_exec_indirect2(&mut state, &mut host, &[0u8; 4]);
    assert_eq!(r.streams_loaded, 0);
}

/// The packet's semantic identity must reach the backend at the exact point
/// its command streams end. Without this event a backend has no contract-owned
/// lifetime for an independently recorded encoder and can only infer one from
/// timing or from the next packet arriving.
#[test]
fn an_exec_packet_closes_its_exact_backend_submission() {
    use crate::runtime::executor::*;
    use reims_vgpu_core::{
        CapabilityService, ComputeResidencyService, ExecutionPort, GuestWriteService,
        PresentationService, ReadbackService, ResidentService,
    };
    use std::sync::Mutex;

    #[derive(Debug, Default)]
    struct CloseProbe {
        closed: Mutex<Vec<reims_vgpu_protocol::SubmissionIdentity>>,
    }

    impl ExecutionPort for CloseProbe {
        type Submission = ResolvedSubmission;
        type Completion = ExecutionCompletion;
        type Error = DrawError;

        fn execute(&self, _submission: Self::Submission) -> Result<Self::Completion, Self::Error> {
            unreachable!("the test packet contains no executable stream")
        }
    }
    impl ResidentService for CloseProbe {}
    impl GuestWriteService for CloseProbe {}
    impl ComputeResidencyService for CloseProbe {}
    impl CapabilityService for CloseProbe {}
    impl PresentationService for CloseProbe {}
    impl ReadbackService for CloseProbe {
        type Error = DrawError;

        fn read_target(
            &self,
            _identity: &crate::model::TargetIdentity,
        ) -> Result<reims_vgpu_core::TargetReadback, Self::Error> {
            unreachable!("the test packet performs no readback")
        }
    }
    impl GuestPageTransferService for CloseProbe {}
    impl ResidentCopyService for CloseProbe {}
    impl CompletionService for CloseProbe {}
    impl SubmissionBatchService for CloseProbe {
        fn close_submission(
            &self,
            identity: reims_vgpu_protocol::SubmissionIdentity,
        ) -> Result<(), DrawError> {
            self.closed.lock().unwrap().push(identity);
            Ok(())
        }
    }
    impl GuestImportService for CloseProbe {}
    impl GuestImagePlanningService for CloseProbe {}
    impl MaintenanceService for CloseProbe {}
    impl SessionService for CloseProbe {}
    impl ObservationService for CloseProbe {}
    impl ShaderTranslationService for CloseProbe {}
    impl RenderBufferPlanningService for CloseProbe {}
    impl WindowPresentationService for CloseProbe {}
    impl Executor for CloseProbe {}

    let probe = std::sync::Arc::new(CloseProbe::default());
    let mut state = Device::new_with_executor(DeviceId(1), PAGE_SHIFT_X86, probe.clone());
    let mut host = FakeHost::new();
    state.define_task(3, 0x1_0000, 2);

    // A declared zero-length command buffer is visited and refused, leaving an
    // otherwise empty but well-formed submission. That isolates the boundary
    // event from draw execution.
    let mut payload = vec![
        0u8;
        CHILD_EXEC_INDIRECT_HEADER_LEN as usize
            + CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN as usize
    ];
    st32(&mut payload[CHILD_EXEC_INDIRECT_TASK_ID as usize..], 3);
    st32(&mut payload[CHILD_EXEC_INDIRECT_CMDBUF_COUNT as usize..], 1);

    let result = process_exec_indirect2(&mut state, &mut host, &payload);
    assert_eq!(result.task_id, 3);
    let closed = probe.closed.lock().unwrap();
    assert_eq!(closed.len(), 1, "one packet has one close boundary");
    assert_eq!(closed[0].task.get(), 3);
    assert_ne!(
        closed[0].id.get(),
        0,
        "a packet identity is never standalone"
    );
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
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    state.define_task(3, 0x1_0000, 2);
    assert!(state.tasks[3].active);
    assert!(
        !state.tasks.is_active(6),
        "slot 6 must be dead for this to bite"
    );

    let mut payload = vec![0u8; CHILD_EXEC_INDIRECT_HEADER_LEN as usize];
    st32(&mut payload[CHILD_EXEC_INDIRECT_TASK_ID as usize..], 6);
    st32(&mut payload[CHILD_EXEC_INDIRECT_CMDBUF_COUNT as usize..], 1);

    let r = process_exec_indirect2(&mut state, &mut host, &payload);
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
    use crate::runtime::decode::fifo::{
        CHILD_EXEC_RESOURCE_OBJECT_ID, CHILD_EXEC_RESOURCE_TAIL, CHILD_EXEC_RESOURCE_VALIDITY_OPS,
    };
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
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
    let r = process_exec_indirect2(&mut state, &mut host, &payload);
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
    use crate::runtime::decode::stream::SEGMENT_HEADER_LEN;
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
    use crate::runtime::decode::stream::SEGMENT_TYPE_RENDER;
    // The defect this pins: `walk_stream` opened with `Err(_) => return`, so a
    // stream the framing decoder rejected executed zero records and produced
    // zero log lines — byte-for-byte indistinguishable at the sink from an
    // idle guest that submitted nothing.
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
        &truncated_segment(SEGMENT_TYPE_RENDER, 64, 0),
        &mut out,
        &mut acc,
    );
    let added = sink_body()[before..].to_string();
    assert!(
        added.contains("stream_frame_fail"),
        "a stream that will not frame must reach the always-on sink, got:\n{added}"
    );
    assert!(
        added.contains("reason=stream_seg_len_past_buffer_end"),
        "the line must name which framing check refused, not just that one \
         did — 17 checks shared `ErrBadLength`. got:\n{added}"
    );
    assert!(
        added.contains(&format!("task={task_id}")),
        "the line must carry the task whose work was dropped, got:\n{added}"
    );
}

/// Pipeline and bind state belong to the serialized encoder, not to the child
/// buffer that happens to carry one segment of it. The second segment omits its
/// pipeline record deliberately: resetting at the child-buffer boundary turns
/// its valid draw into `stream_draw_dropped_unbound`.
#[test]
fn a_render_encoder_continuation_keeps_pipeline_state_across_child_buffers() {
    use crate::runtime::decode::stream::{SEGMENT_HEADER_LEN, SEGMENT_TYPE_RENDER};

    fn render_segment(records: &[u8], continues_previous: bool, continues_next: bool) -> Vec<u8> {
        let mut bytes = vec![0u8; SEGMENT_HEADER_LEN];
        st32(
            &mut bytes[0..4],
            (SEGMENT_HEADER_LEN + records.len()) as u32,
        );
        bytes[4] = SEGMENT_TYPE_RENDER;
        bytes[5] = u8::from(continues_previous);
        bytes[6] = u8::from(continues_next);
        bytes.extend_from_slice(records);
        bytes
    }

    let mut set_pipeline = vec![0u8; wire_render::SET_STATE_TOTAL_LEN as usize];
    st32(
        &mut set_pipeline[0..4],
        wire_render::OPCODE_SET_RENDER_PIPELINE_STATE,
    );
    st32(&mut set_pipeline[4..8], wire_render::SET_STATE_TOTAL_LEN);
    st32(&mut set_pipeline[8..12], 0x41);

    let mut draw = vec![0u8; wire_render::DRAW_TOTAL_LEN as usize];
    st32(&mut draw[0..4], wire_render::OPCODE_DRAW);
    st32(&mut draw[4..8], wire_render::DRAW_TOTAL_LEN);
    st32(&mut draw[8..12], 3);
    st16(&mut draw[12..14], 0);
    st16(&mut draw[14..16], 3);

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut open = None;

    walk_submitted_stream(
        &mut state,
        &mut host,
        1,
        0,
        &render_segment(&set_pipeline, false, true),
        &mut out,
        &mut open,
    );
    walk_submitted_stream(
        &mut state,
        &mut host,
        1,
        1,
        &render_segment(&draw, true, true),
        &mut out,
        &mut open,
    );

    let Some(OpenEncoder::Render(acc)) = open.as_ref() else {
        panic!("the continued render encoder must remain open")
    };
    assert_eq!(acc.pipeline_ref, 0x41);
    assert_eq!(acc.draws.len(), 1, "the continuation draw must be retained");
    assert_eq!(acc.dropped_no_pipeline, 0);

    walk_submitted_stream(
        &mut state,
        &mut host,
        1,
        2,
        &render_segment(&[], true, false),
        &mut out,
        &mut open,
    );
    assert!(
        open.is_none(),
        "the closing segment owns encoder retirement"
    );
}

#[test]
fn submission_context_preserves_every_child_buffer_segment_in_order() {
    use crate::runtime::decode::stream::{
        SEGMENT_HEADER_LEN, SEGMENT_TYPE_COMPUTE, SEGMENT_TYPE_RENDER,
    };

    fn segment(type_: u8, continues_previous: bool, continues_next: bool) -> Vec<u8> {
        let mut bytes = vec![0u8; SEGMENT_HEADER_LEN];
        st32(&mut bytes[0..4], SEGMENT_HEADER_LEN as u32);
        bytes[4] = type_;
        bytes[5] = u8::from(continues_previous);
        bytes[6] = u8::from(continues_next);
        bytes
    }

    let streams = vec![
        segment(SEGMENT_TYPE_RENDER, false, true),
        segment(SEGMENT_TYPE_COMPUTE, true, false),
    ];
    let boundaries = semantic_submission_segments(&streams);
    assert_eq!(
        boundaries.as_ref(),
        [
            SegmentBoundary {
                stream_index: 0,
                index: 0,
                kind: SegmentKind::Render,
                continues_previous: false,
                continues_next: true,
            },
            SegmentBoundary {
                stream_index: 1,
                index: 0,
                kind: SegmentKind::Compute,
                continues_previous: true,
                continues_next: false,
            },
        ]
    );
}

#[test]
fn a_truncated_segment_names_the_check_rather_than_looking_like_end_of_records() {
    use crate::runtime::decode::stream::{
        segment_type_name, Segment, SEGMENT_HEADER_LEN, SEGMENT_TYPE_INFO,
    };
    // `Err(_) => break` treated a self-inconsistent segment exactly like
    // `Done`: the remaining records went unexecuted with nothing logged.
    let stream = vec![0u8; SEGMENT_HEADER_LEN + 4];
    // A segment claiming a longer body than the buffer holds, handed straight
    // to the record walker — the shape `iter_segments` would have rejected but
    // that an already-parsed `Segment` can still carry.
    let seg = Segment {
        offset: 0,
        length: (SEGMENT_HEADER_LEN + 64) as u32,
        type_: SEGMENT_TYPE_INFO,
        command_offset: SEGMENT_HEADER_LEN as u32,
        command_length: 64,
        ..Segment::default()
    };
    let before = sink_body().len();
    let mut handled = 0usize;
    walk_segment_records(&stream, &seg, |_, _| handled += 1);
    let added = sink_body()[before..].to_string();
    assert_eq!(handled, 0, "the malformed segment yields no records");
    assert!(
        added.contains("stream_record_fail"),
        "dropping a segment's records must reach the sink, got:\n{added}"
    );
    assert!(
        added.contains("reason=stream_reval_span_oob"),
        "the line must name the failing re-validation check, got:\n{added}"
    );
    assert!(
        added.contains(&format!(
            "seg={}",
            segment_type_name(u32::from(SEGMENT_TYPE_INFO))
        )),
        "the line must say which segment family lost its records, got:\n{added}"
    );
}

#[test]
fn walking_a_well_formed_segment_to_its_end_logs_nothing() {
    use crate::runtime::decode::stream::{iter_segments, SEGMENT_HEADER_LEN, SEGMENT_TYPE_EVENT};
    // The other half of the obligation: `Done` is how every segment ends, so
    // if it produced a line the sink would carry one per segment per frame.
    let mut records = [0u8; 8];
    st32(&mut records[0..4], 0x190);
    st32(&mut records[4..8], 8);
    let mut stream = vec![0u8; SEGMENT_HEADER_LEN];
    st32(
        &mut stream[0..4],
        (SEGMENT_HEADER_LEN + records.len()) as u32,
    );
    stream[4] = SEGMENT_TYPE_EVENT;
    stream.extend_from_slice(&records);

    let segs = iter_segments(&stream).expect("a well-formed stream frames");
    let before = sink_body().len();
    let mut handled = 0usize;
    walk_segment_records(&stream, &segs[0], |_, _| handled += 1);
    let added = sink_body()[before..].to_string();
    assert_eq!(handled, 1, "the one record is handed over");
    assert!(
        !added.contains("stream_record_fail"),
        "end-of-segment is control flow and must stay out of the log, got:\n{added}"
    );
}

#[test]
fn an_unknown_segment_family_is_refused_and_the_type_5_envelope_is_not() {
    use crate::observe::Refusal;
    use crate::runtime::decode::stream::{
        segment_disposition, SegmentDisposition, SEGMENT_TYPE_BLIT, SEGMENT_TYPE_PROTECTION_OPTIONS,
    };
    // `walk_stream` ended in `_ => {}`, which gave one silence to two very
    // different things. Type 5 is a contract-correct skip; type 6 is wire
    // format the host has never seen.
    assert_eq!(
        segment_disposition(SEGMENT_TYPE_PROTECTION_OPTIONS),
        SegmentDisposition::Envelope
    );
    assert_eq!(
        segment_disposition(SEGMENT_TYPE_PROTECTION_OPTIONS).refusal(),
        None,
        "the envelope arrives on healthy frames; a line here is a flood"
    );
    assert_eq!(
        segment_disposition(SEGMENT_TYPE_BLIT),
        SegmentDisposition::Walk
    );
    assert_eq!(
        segment_disposition(6).refusal(),
        Some("stream_segment_type_unknown")
    );
    assert_eq!(
        segment_disposition(0xff).refusal(),
        Some("stream_segment_type_unknown")
    );
}

#[test]
fn render_preflight_collects_content_pipelines_without_duplicates() {
    use crate::runtime::decode::stream::{SEGMENT_HEADER_LEN, SEGMENT_TYPE_RENDER};
    use wire_render::OPCODE_SET_RENDER_PIPELINE_STATE;

    let mut records = Vec::new();
    for pipeline in [41u32, 77, 41] {
        let mut cmd = [0u8; 12];
        st32(&mut cmd[0..4], OPCODE_SET_RENDER_PIPELINE_STATE);
        st32(&mut cmd[4..8], 12);
        st32(&mut cmd[8..12], pipeline);
        records.extend_from_slice(&cmd);
    }
    let mut stream = vec![0u8; SEGMENT_HEADER_LEN];
    let stream_len = stream.len() + records.len();
    st32(&mut stream[0..4], stream_len as u32);
    stream[4] = SEGMENT_TYPE_RENDER;
    stream.extend_from_slice(&records);

    assert_eq!(render_pipeline_refs(&stream), vec![41, 77]);
}

#[test]
fn compute_preflight_collects_pipeline_and_local_size_without_duplicates() {
    use crate::runtime::decode::stream::{SEGMENT_HEADER_LEN, SEGMENT_TYPE_COMPUTE};

    let mut records = Vec::new();
    let mut pipeline = [0u8; 12];
    st32(&mut pipeline[0..4], wire_compute::OPCODE_SET_PIPELINE_STATE);
    st32(&mut pipeline[4..8], 12);
    st32(&mut pipeline[8..12], 20);
    records.extend_from_slice(&pipeline);
    for opcode in [
        wire_compute::OPCODE_DISPATCH_THREADGROUPS,
        wire_compute::OPCODE_DISPATCH_THREADGROUPS,
        wire_compute::OPCODE_DISPATCH_THREADS,
    ] {
        let mut dispatch = [0u8; 56];
        st32(&mut dispatch[0..4], opcode);
        st32(&mut dispatch[4..8], 56);
        st64(&mut dispatch[8..16], 6);
        st64(&mut dispatch[16..24], 11);
        st64(&mut dispatch[24..32], 1);
        st64(&mut dispatch[32..40], 16);
        st64(&mut dispatch[40..48], 16);
        st64(&mut dispatch[48..56], 1);
        records.extend_from_slice(&dispatch);
    }
    let mut stream = vec![0u8; SEGMENT_HEADER_LEN];
    let stream_len = stream.len() + records.len();
    st32(&mut stream[0..4], stream_len as u32);
    stream[4] = SEGMENT_TYPE_COMPUTE;
    stream.extend_from_slice(&records);

    assert_eq!(compute_translation_inputs(&stream), vec![(20, [16, 16, 1])]);
}

#[test]
fn event_segment_signal_wait_in_stream() {
    use crate::runtime::decode::event::SIGNAL_WAIT_PAYLOAD_LEN;
    use crate::runtime::decode::stream::{SEGMENT_HEADER_LEN, SEGMENT_TYPE_EVENT};

    fn push_segment(buf: &mut Vec<u8>, type_: u8, payload: &[u8]) {
        let len = (SEGMENT_HEADER_LEN + payload.len()) as u32;
        let mut hdr = [0u8; 8];
        st32(&mut hdr[0..4], len);
        hdr[4] = type_;
        buf.extend_from_slice(&hdr);
        buf.extend_from_slice(payload);
    }
    fn push_event_record(buf: &mut Vec<u8>, opcode: u32, event_ref: u32, value: u64) {
        let mut payload = [0u8; SIGNAL_WAIT_PAYLOAD_LEN];
        st32(&mut payload[0..4], event_ref);
        st64(&mut payload[4..12], value);
        let len = (OP_HEADER_LEN + SIGNAL_WAIT_PAYLOAD_LEN) as u32;
        let mut hdr = [0u8; 8];
        st32(&mut hdr[0..4], opcode);
        st32(&mut hdr[4..8], len);
        buf.extend_from_slice(&hdr);
        buf.extend_from_slice(&payload);
    }

    let mut records = Vec::new();
    push_event_record(&mut records, event_decode::OP_SIGNAL_EVENT, 11, 7);
    push_event_record(&mut records, event_decode::OP_WAIT_EVENT, 11, 7);
    push_event_record(&mut records, event_decode::OP_WAIT_EVENT, 11, 8); // pending
    let mut stream = Vec::new();
    push_segment(&mut stream, SEGMENT_TYPE_EVENT, &records);

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    walk_stream(&mut state, &mut host, 1, &stream, &mut out, &mut acc);

    // The signal landed, and the pending wait for 8 left it alone. The
    // three per-op counters this used to assert had no product reader; the
    // generation store is what the next wait actually reads.
    assert_eq!(state.event_generation(1, 11), Some(7));
}

#[test]
fn multi_attachment_decode_in_pass() {
    let mut payload = vec![0u8; PASS_COLOR_ATTACH_OFF + PASS_COLOR_ATTACH_STRIDE * 2];
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
    let a0 = decode_color_attachment(&payload, 0);
    let a1 = decode_color_attachment(&payload, 1);
    assert_eq!(a0.texture_ref, 41);
    assert_eq!(a1.texture_ref, 42);
    let mut cmd = vec![0u8; OP_HEADER_LEN + payload.len()];
    st32(&mut cmd[0..], wire_pass::OPCODE_RENDER_PASS);
    st32(&mut cmd[4..], (OP_HEADER_LEN + payload.len()) as u32);
    cmd[OP_HEADER_LEN..].copy_from_slice(&payload);
    let c = render::decode(&cmd).unwrap();
    assert_eq!(c.kind, RenderKind::RenderPass);
    assert_eq!(c.color0.texture_ref, 41);
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
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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

/// Primitive topology is parsed at the stream-normalization boundary. An
/// ordinal outside the advertised semantic enum must refuse this draw rather
/// than reaching the executor as a triangle through a defaulting conversion.
#[test]
fn an_unknown_primitive_topology_refuses_the_draw_without_a_fallback() {
    let mut command = vec![0u8; wire_render::DRAW_TOTAL_LEN as usize];
    st32(&mut command[0..4], wire_render::OPCODE_DRAW);
    st32(&mut command[4..8], wire_render::DRAW_TOTAL_LEN);
    st32(&mut command[8..12], 5); // outside the public MTLPrimitiveType enum
    st16(&mut command[12..14], 0);
    st16(&mut command[14..16], 3);

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum {
        pipeline_ref: 0x41,
        ..Default::default()
    };
    handle_render_record(
        &mut state,
        &host,
        1,
        wire_render::OPCODE_DRAW,
        &command,
        &mut out,
        &mut acc,
    );

    assert!(acc.saw_draw, "the decoded guest operation was observed");
    assert!(
        acc.draws.is_empty(),
        "an unknown topology must not become an executable draw"
    );
    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(
        log.contains("reason=unknown_primitive_type") && log.contains("raw=5"),
        "the refusal must preserve the contract field and exact ordinal; got:\n{log}"
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
    use crate::runtime::decode::render::{
        PASS_ATTACH_DEPTH_PLANE, PASS_ATTACH_LEVEL, PASS_ATTACH_RESOLVEREF, PASS_ATTACH_SLICE,
        PASS_ATTACH_TEXREF, PASS_DEPTH_ATTACH_OFF, PASS_STENCIL_ATTACH_OFF,
    };
    let pass = |level: u16, resolve: u32| {
        let mut payload = vec![0u8; 0x200];
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
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    use crate::runtime::decode::render::{PASS_ATTACH_TEXREF, PASS_COLOR_ATTACH_OFF};
    use reims_vgpu_core::endian::st32;

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
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    use reims_vgpu_core::endian::st32;

    let record = |count: u32| {
        let total = wire_pass::DEFAULT_RASTER_SAMPLE_COUNT_TOTAL_LEN as usize;
        let mut cmd = vec![0u8; total];
        st32(&mut cmd[0..], wire_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT);
        st32(&mut cmd[4..], total as u32);
        st32(&mut cmd[OP_HEADER_LEN..], count);
        cmd
    };
    let run = |cmd: &[u8]| {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    use crate::runtime::decode::render::{
        PASS_ATTACH_DEPTH_PLANE, PASS_ATTACH_LEVEL, PASS_ATTACH_RESOLVEREF, PASS_ATTACH_SLICE,
        PASS_ATTACH_TEXREF, PASS_COLOR_ATTACH_OFF, PASS_MIN_PAYLOAD,
    };
    use reims_vgpu_core::endian::st32;

    let pass_resolving = |level: u16, slice: u16, plane: u16, resolve: u32| {
        let total = OP_HEADER_LEN + PASS_MIN_PAYLOAD;
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
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    // and loses coverage before the backend sees the request.
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
/// Declared twice because `coverage_band` belongs to the Vulkan execution path and
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

    // The instrument clamps a malformed over-attachment reading to its top
    // band. Product execution refuses this shape before it reaches a backend.
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

#[test]
fn pass_extent_zero_is_per_axis_default_and_large_values_refuse() {
    let extent = render_target_extent(0, 4).expect("representable extent");
    assert_eq!(extent.width, None);
    assert_eq!(extent.height.map(std::num::NonZeroU32::get), Some(4));

    let error = render_target_extent(u64::from(u32::MAX) + 1, 0)
        .expect_err("the semantic image geometry is u32");
    assert_eq!(error.axis, "width");
    assert_eq!(error.raw, u64::from(u32::MAX) + 1);
    assert_eq!(
        crate::observe::Decline::slug(&error),
        "render_target_extent_unrepresentable"
    );
}

/// The extent census scores whichever resolve arm supplied the mapping id,
/// and only for slot 0.
///
/// This is the arm-parity test. The census used to hang off the IOSurface texture
/// resolve alone, so on the x86/Vulkan pathway — where the workload takes
/// the surface backing arm — every band read zero, which is indistinguishable from a
/// guest that never states an extent. A surface backing attachment *is* its own
/// mapping id, so the only difference between the two call sites is which
/// id they pass, and this pins that the scoring does not care which.
#[test]
fn the_pass_extent_census_scores_either_resolve_arm() {
    use crate::runtime::drain::store_route_count;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let cmd = crate::runtime::decode::render::Command {
        pass_render_target_width: 960,
        pass_render_target_height: 540,
        ..Default::default()
    };
    // One mapping, reached by the id either arm would hand over.
    assert!(state.map_surface(7));
    let _ = state.set_mapping_geom(7, 1920, 1080, 0);

    let before = store_route_count("pass_extent_le25");
    note_pass_extent_for_slot(&state, 1, 0, 7, &cmd);
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
    note_pass_extent_for_slot(&state, 1, 1, 7, &cmd);
    note_pass_extent_for_slot(&state, 1, 0, 4242, &cmd);
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

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    assert_eq!(indexed.index_type, Ok(reims_vgpu_protocol::IndexType::U16));
    assert_eq!(indexed.index_buffer_ref, 0x3e);
    assert_eq!(indexed.index_count, 6);
    assert_eq!(indexed.index_buffer_offset, 0x10100);
    assert_eq!(
        acc.draws[0].draw,
        DrawArgs {
            vertex_count: 6,
            instance_count: 1,
            primitive_topology: reims_vgpu_protocol::PrimitiveTopology::Triangle,
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
    use reims_vgpu_core::endian::st16;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
        &[crate::runtime::decode::render::DecodedBufferBind {
            buffer_ref: 77,
            offset: 0,
            attribute_stride: None,
        }],
        0,
        BindTarget {
            stage: Stage::Vertex,
            class: BindClass::Buffer,
        },
        BindTables {
            vertex: &mut acc.vertex_buffers,
            fragment: &mut acc.fragment_buffers,
            refused: &mut acc.unrepresentable,
        },
        |b| b.index,
        |index, b: crate::runtime::decode::render::DecodedBufferBind| {
            Some(BufferBind {
                index,
                buffer_ref: b.buffer_ref,
                offset: b.offset,
                attribute_stride: b.attribute_stride,
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
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));
    let put_buffer = |host: &mut FakeHost, state: &Device, handle: u32, size: u64| {
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

    let total = OP_HEADER_LEN + render::BIND_ENTRIES + render::BUFFER_BIND_ENTRY_SIZE;
    let mut bind = vec![0u8; total];
    st32(&mut bind, wire_render::OPCODE_SET_VERTEX_BUFFER);
    st32(&mut bind[4..], total as u32);
    st32(&mut bind[OP_HEADER_LEN + render::BIND_COUNT..], 1);
    st32(&mut bind[OP_HEADER_LEN + render::BIND_ENTRIES..], 7);
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

    let offset_total = OP_HEADER_LEN + render::BUFFER_OFFSET_PAYLOAD_LEN;
    let mut offset = vec![0u8; offset_total];
    st32(&mut offset, wire_render::OPCODE_SET_VERTEX_BUFFER_OFFSET);
    st32(&mut offset[4..], offset_total as u32);
    st64(
        &mut offset[OP_HEADER_LEN + render::BUFFER_OFFSET_VALUE..],
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

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let first = state.task_objects.resources.register(
        1,
        9,
        Arc::new(TaskResource::new(
            ListObjectEntry::new(reims_vgpu_protocol::ObjectKind::Buffer, 0, 0),
            Arc::from([]),
        )),
    );
    let total = OP_HEADER_LEN + render::BIND_ENTRIES + 4;
    let mut command = vec![0u8; total];
    st32(&mut command, wire_render::OPCODE_SET_FRAGMENT_TEXTURE);
    st32(&mut command[4..], total as u32);
    st32(&mut command[OP_HEADER_LEN + render::BIND_COUNT..], 1);
    st32(&mut command[OP_HEADER_LEN + render::BIND_ENTRIES..], 9);
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

    assert!(state.task_objects.resources.delete(1, 9));
    let replacement = state.task_objects.resources.register(
        1,
        9,
        Arc::new(TaskResource::new(
            ListObjectEntry::new(reims_vgpu_protocol::ObjectKind::Buffer, 0, 0),
            Arc::from([]),
        )),
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
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum {
        pipeline_ref: 0xface,
        ..Default::default()
    };
    let task_id = 0xfeed;
    let mut command = vec![0u8; OP_HEADER_LEN];
    // An opcode inside the encoder's range that no arm claims, found rather
    // than named. It used to be `wire_render::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE`, which stopped working
    // the moment that bound was corrected to `0xa6` -- because `0xa6` is a
    // record this rail now decodes. `0x99` was the replacement and lasted
    // one commit, until `setVertexAmplificationMode:value:` turned out to be
    // that number. The catch-all is what is under test, not any literal.
    let op = render::unclaimed_accepted_opcode();
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
    note_draw_encode_fail(task, pipe, EncodeStatus::BadArgs("draw_zero_geom"), 1, 3);
    let body = sink_body();
    assert!(
        body.lines().any(
            |l| l.contains("draw_encode_fail reason=draw_zero_geom class=bad_args")
                && l.contains(&format!("pipe={pipe}"))
                && l.contains(&format!("task={task}"))
                && l.contains("di=1/3")
        ),
        "the boundary line must carry the specific check and the class:\n{body}"
    );

    // Latched per (reason, pipeline): the guest re-submits the same failing
    // draw every frame, so a repeat adds nothing the first line did not…
    note_draw_encode_fail(task, pipe, EncodeStatus::BadArgs("draw_zero_geom"), 2, 3);
    // …but a *different* check on the same pipeline is a different event and
    // must still be visible. Latching on the class would have hidden it, which
    // is exactly the failure this migration removes.
    note_draw_encode_fail(
        task,
        pipe,
        EncodeStatus::BackendFailed("draw_core_failed"),
        2,
        3,
    );
    let body = sink_body();
    assert_eq!(
        body.matches("reason=draw_zero_geom").count(),
        1,
        "a re-attempted refusal must log once:\n{body}"
    );
    assert!(
        body.contains("reason=draw_core_failed"),
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

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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

/// x86 surface backing display mid: clear-only stream must Store solid BGRA into pages.
#[test]
fn clear_only_surface_backing_writes_guest_pages() {
    use crate::runtime::decode::render::ColorAttachment;
    use crate::runtime::objects::{self, OBJECT_TYPE_SURFACE};
    use reims_vgpu_core::endian::{st32, st64};
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use reims_vgpu_paging::geometry::{
        MAPPER_PAGE_ENTRY_PFN_SHIFT as PAGE_ENTRY_PFN_SHIFT,
        MAPPER_PAGE_ENTRY_VALID as PAGE_ENTRY_VALID,
    };

    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    // Surface backing at surface_id=5.
    let mut entry = [0u8; 12];
    st32(
        &mut entry[0..],
        (OBJECT_TYPE_SURFACE as u32) | (0x30u32 << 8),
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

    assert!(objects::resolve_surface_backing(&mut state, &host, 5));
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
        "surface backing clear must apply, got {}",
        out.clears_applied
    );
    // Read first pixel from guest page (BGRA).
    let mut px = [0u8; 4];
    assert!(host.read_gpa(page, &mut px).is_ok());
    assert_eq!(px, [0, 0, 255, 255], "expected opaque red BGRA, got {px:?}");
    let m = state.surfaces.mappings.get(&5).expect("mapping");
    assert!(m.content.guest_page_generation > 0 || m.lifecycle.active);
    let _ = PAGE_ENTRY_VALID;
    let _ = PAGE_ENTRY_PFN_SHIFT;
}

/// Archive DrawJob: clear-only packets store immediately; multi-draw packets
/// keep CLEAR as private Metal seed (no pre-draw guest clear).
#[test]
fn finish_stream_clear_only_branch_without_draws() {
    use crate::runtime::decode::render::ColorAttachment;
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    assert_eq!(out.draws_ok, 0);
    assert_eq!(out.draws_fail, 0);
}

#[test]
fn finish_stream_with_draws_skips_guest_clear_prelude() {
    use crate::runtime::decode::render::ColorAttachment;
    use crate::runtime::draw::BufferBind;
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    acc.push_draw(PendingDraw {
        icb_ref: None,
        visibility: None,
        pipeline_ref: 1,
        draw: DrawArgs {
            vertex_count: 3,
            instance_count: 1,
            primitive_topology: reims_vgpu_protocol::PrimitiveTopology::Triangle,
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
        render_target_extent: Default::default(),
        cull_mode: reims_vgpu_protocol::CullMode::None,
        front_face_ccw: false,
        fill_mode: reims_vgpu_protocol::FillMode::Fill,
        line_width: reims_vgpu_core::LineWidth::ONE,
        depth_clip_mode: reims_vgpu_protocol::DepthClipMode::Clip,
        depth_bias: None,
        depth_stencil_ref: 0,
        stencil_ref: None,
        depth_attach: None,
        stencil_attach: None,
    });
    finish_stream(&mut state, &mut host, 1, &mut out, &acc);
    // Unresolvable RT → mrt_request fail before encode (not BackendUnavailable); no clear.
    assert_eq!(
        out.clears_applied, 0,
        "unresolvable multi-draw must not guest-clear"
    );
}

/// Linux BackendUnavailable: draws fail but CLEAR seed still Stores into surface backing pages.
#[test]
fn backend_unavailable_draw_falls_back_to_surface_backing_clear() {
    use crate::runtime::decode::render::ColorAttachment;
    use crate::runtime::draw::BufferBind;
    use crate::runtime::objects::{self, OBJECT_TYPE_SURFACE};
    use reims_vgpu_core::endian::{st32, st64};
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
        (OBJECT_TYPE_SURFACE as u32) | (0x30u32 << 8),
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
    assert!(objects::resolve_surface_backing(&mut state, &host, 5));

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
    acc.push_draw(PendingDraw {
        icb_ref: None,
        visibility: None,
        pipeline_ref: 7,
        draw: DrawArgs {
            vertex_count: 3,
            instance_count: 1,
            primitive_topology: reims_vgpu_protocol::PrimitiveTopology::Triangle,
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
        render_target_extent: Default::default(),
        cull_mode: reims_vgpu_protocol::CullMode::None,
        front_face_ccw: false,
        fill_mode: reims_vgpu_protocol::FillMode::Fill,
        line_width: reims_vgpu_core::LineWidth::ONE,
        depth_clip_mode: reims_vgpu_protocol::DepthClipMode::Clip,
        depth_bias: None,
        depth_stencil_ref: 0,
        stencil_ref: None,
        depth_attach: None,
        stencil_attach: None,
    });
    let mut second = acc.draws[0].clone();
    second.pipeline_ref = 8;
    acc.push_draw(second);
    finish_stream(&mut state, &mut host, 1, &mut out, &acc);
    assert_eq!(
        out.render_attachment_resolves, 1,
        "one render stream resolves its fixed attachment set once"
    );
    // Non-Apple: Linux encode Stores CLEAR load into surface backing (Ok) or
    // BackendUnavailable clear fallback — either path must land green BGRA.
    {
        assert!(
            out.draws_ok >= 1 || out.clears_applied >= 1 || out.draws_fail >= 1,
            "expected clear store path: ok={} clear={} fail={}",
            out.draws_ok,
            out.clears_applied,
            out.draws_fail
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
fn multi_draw_chain_source_preserves_cpu_materialized_output() {
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
        primitive_topology: reims_vgpu_protocol::PrimitiveTopology::Triangle,
        colors: vec![draw::ColorRtRequest {
            slot: 0,
            texture_ref: 11,
            resource: None,
            storage: draw::ColorTargetStorage::Mapping(3),
            width: 1920,
            height: 1080,
            format: 0x50,
            sample_count: 1,
            load_action: reims_vgpu_protocol::pass_action::LoadAction::Clear,
            store_action: reims_vgpu_protocol::pass_action::StoreAction::Store,
            clear_color: [0.1, 0.2, 0.3, 1.0],
            target_seed_rgba: Some(vec![0xbb; 16]),
            multisample_source_ref: 0,
        }],
        ..Default::default()
    };
    let template = render_pass_attachment_template(&first);
    assert!(template.colors[0].target_seed_rgba.is_none());
    assert_eq!(
        template.colors[0].load_action,
        reims_vgpu_protocol::pass_action::LoadAction::Load
    );
    assert_eq!(template.colors[0].mapping_id(), 3);
    assert_eq!(
        (template.colors[0].width, template.colors[0].height),
        (1920, 1080)
    );

    let draw = PendingDraw {
        pipeline_ref: 42,
        draw: DrawArgs {
            vertex_count: 6,
            instance_count: 2,
            primitive_topology: reims_vgpu_protocol::PrimitiveTopology::TriangleStrip,
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
            req.primitive_topology,
            req.first_vertex
        ),
        (
            6,
            2,
            reims_vgpu_protocol::PrimitiveTopology::TriangleStrip,
            9,
        )
    );
    assert_eq!(req.colors.len(), 1);
    assert_eq!(req.colors[0].mapping_id(), 3);
    assert_eq!(
        first.colors[0].target_seed_rgba.as_ref().map(Vec::len),
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
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    use crate::runtime::decode::render::StencilAttachment;
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_protocol::pass_action::MTL_STORE_ACTION_DONT_CARE;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();

    let record = |opcode: u32, action: u64| {
        let total = reims_vgpu_wire::OP_HEADER_LEN + 8;
        let mut c = vec![0u8; total];
        st32(&mut c[0..], opcode);
        st32(&mut c[4..], total as u32);
        reims_vgpu_core::endian::st64(&mut c[reims_vgpu_wire::OP_HEADER_LEN..], action);
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
    use reims_vgpu_core::endian::st64;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
        + render::SCISSOR_RECTS_COUNT_LEN
        + rects.len() * render::SCISSOR_PAYLOAD_LEN;
    let mut command = vec![0u8; total];
    st32(&mut command[0..], op);
    st32(&mut command[4..], total as u32);
    st64(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN..],
        rects.len() as u64,
    );
    let e0 = reims_vgpu_wire::OP_HEADER_LEN + render::SCISSOR_RECTS_COUNT_LEN;
    for (n, r) in rects.iter().enumerate() {
        let at = e0 + n * render::SCISSOR_PAYLOAD_LEN;
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
    let total = reims_vgpu_wire::OP_HEADER_LEN + render::SCISSOR_PAYLOAD_LEN;
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
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_core::endian::st64;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
        + render::SCISSOR_RECTS_COUNT_LEN
        + 2 * render::SCISSOR_PAYLOAD_LEN;
    let mut command = vec![0u8; total];
    st32(&mut command[0..], op);
    st32(&mut command[4..], total as u32);
    st64(&mut command[reims_vgpu_wire::OP_HEADER_LEN..], 2);
    let e0 = reims_vgpu_wire::OP_HEADER_LEN + render::SCISSOR_RECTS_COUNT_LEN;
    for (i, val) in [11u64, 22, 33, 44].into_iter().enumerate() {
        st64(&mut command[e0 + i * 8..], val);
    }
    let e1 = e0 + render::SCISSOR_PAYLOAD_LEN;
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
    let entry = render::REF_BIND_ENTRY_SIZE;
    let total = reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES + (COUNT as usize) * entry;
    let mut command = vec![0u8; total];
    let op = wire_render::OPCODE_SET_VERTEX_TEXTURE;
    st32(&mut command[0..], op);
    st32(&mut command[4..], total as u32);
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_FIRST..],
        0,
    );
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_COUNT..],
        COUNT,
    );
    for i in 0..COUNT as usize {
        let at = reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES + i * entry;
        st32(&mut command[at..], 0x4000 + i as u32);
    }

    // The record itself must survive decode; a cap that refused it whole is
    // what this counter exists to distinguish from.
    let c = render::decode(&command).expect("an over-table texture run must decode");
    assert_eq!(c.ref_binds.len(), COUNT as usize);

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
            stage: render::Stage::Vertex,
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
            stage: render::Stage::Fragment,
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

    let entry = render::REF_BIND_ENTRY_SIZE;
    let total = reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES + entry;
    let mut command = vec![0u8; total];
    let op = wire_render::OPCODE_SET_VERTEX_SAMPLER;
    st32(&mut command[0..], op);
    st32(&mut command[4..], total as u32);
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_FIRST..],
        FIRST,
    );
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_COUNT..],
        1,
    );
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES..],
        0x3333,
    );

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
/// variable is — so metal2vulkan assigns non-overlapping texture and sampler
/// ranges and the whole 128-entry table becomes reachable.
///
/// Asserted three ways, because each alone could pass while the slot is still
/// lost: the accumulator keeps the bind, nothing is counted against the table,
/// and the binding it will carry is below the sampler band rather than inside it.
#[test]
fn a_texture_bind_past_the_old_band_binds_and_keeps_its_own_descriptor() {
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_vulkan::spirv_bind::{SAMPLER_BINDING_BASE, TEXTURE_BINDING_BASE};
    use reims_vgpu_wire::ops::bind_limit;

    // Past the old 32-wide band, inside Apple's table, and far enough in that
    // the old numbering would have put it under a sampler.
    const FIRST: u32 = 68;
    const COUNT: u32 = 4;
    const { assert!(FIRST >= 32 && FIRST + COUNT <= bind_limit::TEXTURE) };

    let entry = render::REF_BIND_ENTRY_SIZE;
    let total = reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES + (COUNT as usize) * entry;
    let mut command = vec![0u8; total];
    let op = wire_render::OPCODE_SET_VERTEX_TEXTURE;
    st32(&mut command[0..], op);
    st32(&mut command[4..], total as u32);
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_FIRST..],
        FIRST,
    );
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_COUNT..],
        COUNT,
    );
    for i in 0..COUNT as usize {
        let at = reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES + i * entry;
        st32(&mut command[at..], 0x9000 + i as u32);
    }

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
        let total =
            reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES + render::REF_BIND_ENTRY_SIZE;
        let mut command = vec![0u8; total];
        st32(&mut command[0..], op);
        st32(&mut command[4..], total as u32);
        st32(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_FIRST..],
            first,
        );
        st32(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_COUNT..],
            1,
        );
        st32(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES..],
            obj,
        );
        command
    };

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
        let entry = render::REF_BIND_ENTRY_SIZE;
        let total =
            reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES + (count as usize) * entry;
        let mut command = vec![0u8; total];
        let op = wire_render::OPCODE_SET_VERTEX_TEXTURE;
        st32(&mut command[0..], op);
        st32(&mut command[4..], total as u32);
        st32(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_FIRST..],
            first,
        );
        st32(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_COUNT..],
            count,
        );
        for i in 0..count as usize {
            let at = reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES + i * entry;
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

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    let head = OP_HEADER_LEN;

    // Head (first, count) then two 12-byte entries: ref, lodMin, lodMax.
    let entry = render::SAMPLER_LOD_BIND_ENTRY_SIZE;
    let total = head + render::BIND_ENTRIES + 2 * entry;
    let op = wire_render::OPCODE_SET_FRAGMENT_SAMPLER_LOD;
    let mut command = vec![0u8; total];
    st32(&mut command[0..], op);
    st32(&mut command[4..], total as u32);
    st32(&mut command[head + render::BIND_FIRST..], 2);
    st32(&mut command[head + render::BIND_COUNT..], 2);
    let e0 = head + render::BIND_ENTRIES;
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
    let total = head + render::BIND_ENTRIES + render::REF_BIND_ENTRY_SIZE;
    let op = wire_render::OPCODE_SET_FRAGMENT_SAMPLER;
    let mut command = vec![0u8; total];
    st32(&mut command[0..], op);
    st32(&mut command[4..], total as u32);
    st32(&mut command[head + render::BIND_FIRST..], 0);
    st32(&mut command[head + render::BIND_COUNT..], 1);
    st32(&mut command[head + render::BIND_ENTRIES..], 0x51);
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

    // A type-1 buffer at ref 7 holding `words`, resolvable through the task's
    // own page table — the shape `resolve_indirect_threadgroups_from_buffer`
    // uses on the compute rail.
    let build = |words: &[u32]| {
        let mut host = FakeHost::new();
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
            reims_vgpu_core::draw::DrawArgs {
                vertex_count: 11,
                instance_count: 22,
                primitive_topology: reims_vgpu_protocol::PrimitiveTopology::TriangleStrip,
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
        assert_eq!(
            pd.draw.primitive_topology,
            reims_vgpu_protocol::PrimitiveTopology::Triangle,
            "from the record, not the block"
        );
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
            idx.index_buffer_offset, 0x100,
            "the record's byte offset remains independent of index element width"
        );
        assert_eq!(
            idx.index_start, 33,
            "the indirect element offset stays typed"
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
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    assert_eq!(
        acc.render_work,
        [
            RenderWork::ExecuteIcb(0),
            RenderWork::ExecuteIcb(1),
            RenderWork::ExecuteIcb(2),
        ],
        "the typed payload store must not become a second ordering authority"
    );
}

#[test]
fn icb_replay_uses_the_declared_inheritance_and_keeps_draw_arguments() {
    use crate::runtime::icb::{
        IcbRenderBindStage, IcbRenderBufferBind, IcbRenderDraw, IcbRenderFill,
    };

    let state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let inherited = PendingDraw {
        pipeline_ref: 0x51,
        fragment_textures: Arc::new(vec![TextureBind {
            index: 2,
            texture_ref: 0x77,
            ..Default::default()
        }]),
        vertex_buffers: Arc::new(vec![BufferBind {
            index: 0,
            buffer_ref: 0x88,
            ..Default::default()
        }]),
        ..Default::default()
    };
    let execute = RenderIcbExecute {
        icb_ref: 0x91,
        is_range: true,
        range_location: 0,
        range_length: 1,
        args_buffer_ref: 0,
        args_buffer_offset: 0,
        inherited,
    };
    let descriptor = reims_vgpu_protocol::IndirectCommandBufferDescriptor {
        flags: 0b11,
        ..Default::default()
    };
    let fill = IcbRenderFill {
        command_index: 0,
        pipeline_ref: 0,
        buffers: vec![IcbRenderBufferBind {
            index: 0,
            buffer_ref: 0x99,
            stage: IcbRenderBindStage::Vertex,
            ..Default::default()
        }],
        object_threadgroup_memory: Vec::new(),
        draw: IcbRenderDraw::Primitives {
            primitive_type: 3,
            vertex_start: 4,
            vertex_count: 6,
            instance_count: 2,
            base_instance: 9,
        },
    };
    let draw = pending_draw_from_icb(&state, &host, 1, &execute, &descriptor, fill)
        .expect("the inherited command is representable")
        .expect("the command is not an empty draw");
    assert_eq!(draw.pipeline_ref, 0x51);
    assert_eq!(draw.icb_ref, Some(0x91));
    assert_eq!(draw.draw.vertex_count, 6);
    assert_eq!(draw.draw.instance_count, 2);
    assert_eq!(draw.draw.first_vertex, 4);
    assert_eq!(draw.draw.base_instance, 9);
    assert_eq!(draw.fragment_textures[0].texture_ref, 0x77);
    assert_eq!(
        draw.vertex_buffers[0].buffer_ref, 0x88,
        "inherited encoder buffers remain authoritative over per-command slots"
    );
}

#[test]
fn icb_replay_refuses_a_per_command_pipeline_when_pipeline_state_is_inherited() {
    use crate::runtime::icb::{IcbRenderDraw, IcbRenderFill};

    let state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let execute = RenderIcbExecute {
        icb_ref: 0x91,
        is_range: true,
        range_location: 0,
        range_length: 1,
        args_buffer_ref: 0,
        args_buffer_offset: 0,
        inherited: PendingDraw {
            pipeline_ref: 0x51,
            ..Default::default()
        },
    };
    let descriptor = reims_vgpu_protocol::IndirectCommandBufferDescriptor {
        flags: 1,
        ..Default::default()
    };
    let fill = IcbRenderFill {
        command_index: 0,
        pipeline_ref: 0x61,
        buffers: Vec::new(),
        object_threadgroup_memory: Vec::new(),
        draw: IcbRenderDraw::Primitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            base_instance: 0,
        },
    };

    let refusal = pending_draw_from_icb(&state, &host, 1, &execute, &descriptor, fill).unwrap_err();
    assert_eq!(
        crate::observe::Decline::slug(&refusal),
        "render_icb_inherited_pipeline_ref_nonzero"
    );
}

#[test]
fn direct_icb_barrier_and_direct_expand_in_encoder_order() {
    use crate::runtime::decode::resource::{render_icb_layout, MTL_INDIRECT_CMD_DRAW};
    use crate::runtime::gva_mem;
    use crate::runtime::icb::{encode_render_command_slot, IcbRenderDraw, IcbRenderFill};

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    let layout = render_icb_layout(0, 0, MTL_INDIRECT_CMD_DRAW);
    let descriptor = reims_vgpu_protocol::IndirectCommandBufferDescriptor {
        command_types: MTL_INDIRECT_CMD_DRAW,
        max_command_count: 1,
        flags: 0b11,
        layout,
        ..Default::default()
    };
    state
        .task_objects
        .indirect_command_buffers
        .record(1, 0x91, descriptor)
        .unwrap();
    let slot = encode_render_command_slot(
        &layout,
        &IcbRenderFill {
            command_index: 0,
            pipeline_ref: 0,
            buffers: Vec::new(),
            object_threadgroup_memory: Vec::new(),
            draw: IcbRenderDraw::Primitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1,
                base_instance: 5,
            },
        },
    )
    .unwrap();
    let command_gva = 5u64 << crate::runtime::decode::resource::RESOURCE_PAGE_SHIFT;
    gva_mem::write_task_gva_arm64e(&mut host, &state.tasks[1], command_gva, &slot);
    assert!(state.task_objects.indirect_command_buffers.bind(
        1,
        0x91,
        reims_vgpu_protocol::IcbCommandMemory {
            gva: command_gva,
            byte_len: slot.len() as u64,
        },
    ));

    let direct = |pipeline_ref| PendingDraw {
        pipeline_ref,
        draw: DrawArgs {
            vertex_count: 3,
            instance_count: 1,
            primitive_topology: reims_vgpu_protocol::PrimitiveTopology::Triangle,
            first_vertex: 0,
            base_instance: 0,
        },
        ..Default::default()
    };
    let mut acc = StreamAccum::default();
    acc.push_draw(direct(0x41));
    acc.push_icb(RenderIcbExecute {
        icb_ref: 0x91,
        is_range: true,
        range_location: 0,
        range_length: 1,
        args_buffer_ref: 0,
        args_buffer_offset: 0,
        inherited: direct(0x51),
    });
    acc.push_barrier(reims_vgpu_core::RenderBarrier::Texture);
    acc.push_draw(direct(0x61));

    let mut out = ExecResult::default();
    let expanded = expand_render_work(&state, &host, 1, &mut out, &acc);
    assert_eq!(
        expanded
            .draws
            .iter()
            .map(|draw| draw.pipeline_ref)
            .collect::<Vec<_>>(),
        [0x41, 0x51, 0x61]
    );
    assert_eq!(expanded.draws[1].icb_ref, Some(0x91));
    assert_eq!(expanded.draws[1].draw.base_instance, 5);
    assert_eq!(
        expanded.barriers_before_draw,
        [(2, reims_vgpu_core::RenderBarrier::Texture)]
    );
    assert_eq!(out.render_icb_fail, 0);
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

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();

    // index:u32 @0, offset:u64 @4 — a different payload shape from the plural
    // binds, which is why it takes its own offsets rather than `BIND_*`.
    let offset_record = |index: u32| {
        let total = reims_vgpu_wire::OP_HEADER_LEN + render::BUFFER_OFFSET_PAYLOAD_LEN;
        let mut command = vec![0u8; total];
        let op = wire_render::OPCODE_SET_VERTEX_BUFFER_OFFSET;
        st32(&mut command[0..], op);
        st32(&mut command[4..], total as u32);
        st32(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BUFFER_OFFSET_INDEX..],
            index,
        );
        st64(
            &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BUFFER_OFFSET_VALUE..],
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

/// Residency hints remain measurable no-ops, while barriers retain their exact
/// position and decoded dependency domain in the stream.
///
/// The barrier used to share the residency answer on the claim that a submit
/// boundary was a memory dependency. It is not: the resolved command must
/// reach the backend at its exact position while residency hints remain no-ops.
#[test]
fn render_barriers_preserve_position_and_scope_while_residency_stays_a_noop() {
    use crate::runtime::drain::store_route_count;
    use crate::runtime::executor::*;
    use reims_vgpu_core::{
        CapabilityService, ComputeResidencyService, ExecutionPort, GuestWriteService,
        PresentationService, ReadbackService, ResidentService,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug, Default)]
    struct BarrierProbe {
        flushes: AtomicUsize,
    }

    impl ExecutionPort for BarrierProbe {
        type Submission = ResolvedSubmission;
        type Completion = ExecutionCompletion;
        type Error = DrawError;

        fn execute(&self, _submission: Self::Submission) -> Result<Self::Completion, Self::Error> {
            unreachable!("this test submits no draw")
        }
    }
    impl ResidentService for BarrierProbe {}
    impl GuestWriteService for BarrierProbe {}
    impl ComputeResidencyService for BarrierProbe {}
    impl CapabilityService for BarrierProbe {}
    impl PresentationService for BarrierProbe {}
    impl ReadbackService for BarrierProbe {
        type Error = DrawError;

        fn read_target(
            &self,
            _identity: &crate::model::TargetIdentity,
        ) -> Result<reims_vgpu_core::TargetReadback, Self::Error> {
            unreachable!("this test reads no target")
        }
    }
    impl GuestPageTransferService for BarrierProbe {}
    impl ResidentCopyService for BarrierProbe {}
    impl CompletionService for BarrierProbe {}
    impl SubmissionBatchService for BarrierProbe {
        fn flush_submission_tail(&self) {
            self.flushes.fetch_add(1, Ordering::Relaxed);
        }
    }
    impl GuestImportService for BarrierProbe {}
    impl GuestImagePlanningService for BarrierProbe {}
    impl MaintenanceService for BarrierProbe {}
    impl SessionService for BarrierProbe {}
    impl ObservationService for BarrierProbe {}
    impl ShaderTranslationService for BarrierProbe {}
    impl RenderBufferPlanningService for BarrierProbe {}
    impl WindowPresentationService for BarrierProbe {}
    impl Executor for BarrierProbe {}

    for (op, route, payload_len, expected_flushes) in [
        (
            wire_render::OPCODE_USE_RESOURCE,
            "render_noop_residency_hint",
            render::USE_RESOURCE_REFS + 4,
            0,
        ),
        (
            wire_render::OPCODE_USE_HEAP,
            "render_noop_residency_hint",
            render::USE_HEAP_REFS + 4,
            0,
        ),
        (
            wire_render::OPCODE_MEMORY_BARRIER_RESOURCES,
            "render_barrier_resources",
            core::mem::size_of::<wire_render::MemoryBarrierResources>()
                + core::mem::size_of::<wire_render::RefBind>(),
            0,
        ),
        (
            wire_render::OPCODE_MEMORY_BARRIER_SCOPE,
            "render_barrier_scope",
            core::mem::size_of::<wire_render::MemoryBarrierScope>(),
            0,
        ),
        (
            wire_render::OPCODE_TEXTURE_BARRIER,
            "render_barrier_texture",
            0,
            0,
        ),
    ] {
        let probe = Arc::new(BarrierProbe::default());
        let mut state = Device::new_with_executor(DeviceId(1), PAGE_SHIFT_ARM64E, probe.clone());
        let mut host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();

        let total = reims_vgpu_wire::OP_HEADER_LEN + payload_len;
        let mut command = vec![0u8; total];
        st32(&mut command[0..], op);
        st32(&mut command[4..], total as u32);
        if payload_len > 0 {
            // One resource named, so the count-led extent is satisfied.
            st32(&mut command[reims_vgpu_wire::OP_HEADER_LEN..], 1);
        }

        let before = store_route_count(route);
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, &mut acc);
        assert_eq!(
            store_route_count(route),
            before + 1,
            "op {op:#x} did not reach {route}"
        );
        assert_eq!(
            probe.flushes.load(Ordering::Relaxed),
            0,
            "record walking cannot submit draws that finish_stream has not executed"
        );
        if op == wire_render::OPCODE_TEXTURE_BARRIER {
            assert_eq!(
                acc.render_work,
                [RenderWork::Barrier(reims_vgpu_core::RenderBarrier::Texture)],
                "the barrier must retain its exact position before execution"
            );
        }
        finish_stream(&mut state, &mut host, 1, &mut out, &acc);
        assert_eq!(
            probe.flushes.load(Ordering::Relaxed),
            expected_flushes,
            "op {op:#x} applied the wrong submission boundary"
        );
    }

    // The exact render-target/fragment-to-fragment form retains all four
    // decoded API fields. Vulkan may project this common shape more narrowly;
    // other admitted forms remain typed and use its conservative dependency.
    let probe = Arc::new(BarrierProbe::default());
    let mut state = Device::new_with_executor(DeviceId(1), PAGE_SHIFT_ARM64E, probe.clone());
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    let mut command = vec![0u8; wire_render::MEMORY_BARRIER_SCOPE_TOTAL_LEN as usize];
    st32(&mut command[0..], wire_render::OPCODE_MEMORY_BARRIER_SCOPE);
    st32(
        &mut command[4..],
        wire_render::MEMORY_BARRIER_SCOPE_TOTAL_LEN,
    );
    command[reims_vgpu_wire::OP_HEADER_LEN..][..4].copy_from_slice(&[4, 0, 2, 2]);
    handle_render_record(
        &mut state,
        &host,
        1,
        wire_render::OPCODE_MEMORY_BARRIER_SCOPE,
        &command,
        &mut out,
        &mut acc,
    );
    assert_eq!(
        acc.render_work,
        [RenderWork::Barrier(reims_vgpu_core::RenderBarrier::Scope {
            scope: reims_vgpu_core::MemoryBarrierScope::RENDER_TARGETS,
            after: reims_vgpu_core::RenderBarrierStages::FRAGMENT,
            before: reims_vgpu_core::RenderBarrierStages::FRAGMENT,
        })]
    );
    assert_eq!(probe.flushes.load(Ordering::Relaxed), 0);

    let barrier = reims_vgpu_core::RenderBarrier::Texture;
    let mut cursor = 0;
    let positions = [(1, barrier.clone()), (1, barrier.clone()), (3, barrier)];
    let mut pending = Vec::new();
    append_render_barriers_at(&positions, &mut cursor, 0, &mut pending);
    assert!(pending.is_empty());
    append_render_barriers_at(&positions, &mut cursor, 1, &mut pending);
    assert_eq!(pending.len(), 2);
    append_render_barriers_at(&positions, &mut cursor, 2, &mut pending);
    assert_eq!(pending.len(), 2);
    append_render_barriers_at(&positions, &mut cursor, 3, &mut pending);
    assert_eq!(pending.len(), 3);
    assert_eq!(cursor, positions.len());
}

#[test]
fn render_barrier_resources_are_resolved_generationally_and_invalid_terms_refuse() {
    use crate::model::TaskResource;
    use crate::runtime::decode::resource::ListObjectEntry;
    use reims_vgpu_protocol::{ObjectKind, ObjectTableRef};

    fn resource() -> Arc<TaskResource> {
        Arc::new(TaskResource::new(
            ListObjectEntry::new(ObjectKind::Buffer, 0, 0),
            Arc::from([]),
        ))
    }

    let state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let first = state.task_objects.resources.register(1, 77, resource());
    let command = render::Command {
        kind: RenderKind::BarrierResources,
        barrier_after_stages: 1,
        barrier_before_stages: 2,
        barrier_resources: vec![ObjectTableRef::new(77)],
        ..Default::default()
    };
    let first_barrier = resolved_render_barrier(&state, &host, 1, &command)
        .expect("valid barrier")
        .expect("nonempty barrier");
    let reims_vgpu_core::RenderBarrier::Resources { resources, .. } = &first_barrier else {
        panic!("resource-list barrier changed domain");
    };
    assert_eq!(resources[0].id, first.semantic_id().unwrap());
    assert_eq!(resources[0].lifetime.id(), first.lifetime_ref().id());

    assert!(state.task_objects.resources.delete(1, 77));
    let replacement = state.task_objects.resources.register(1, 77, resource());
    let replacement_barrier = resolved_render_barrier(&state, &host, 1, &command)
        .expect("valid replacement barrier")
        .expect("nonempty replacement barrier");
    let reims_vgpu_core::RenderBarrier::Resources { resources, .. } = replacement_barrier else {
        panic!("replacement changed domain");
    };
    assert_eq!(resources[0].id, replacement.semantic_id().unwrap());
    assert_ne!(
        first_barrier,
        reims_vgpu_core::RenderBarrier::Resources {
            resources,
            after: reims_vgpu_core::RenderBarrierStages::VERTEX,
            before: reims_vgpu_core::RenderBarrierStages::FRAGMENT,
        }
    );

    let unsupported = render::Command {
        kind: RenderKind::BarrierScope,
        barrier_scope: 1,
        barrier_after_stages: 4,
        barrier_before_stages: 2,
        ..Default::default()
    };
    assert_eq!(
        resolved_render_barrier(&state, &host, 1, &unsupported),
        Err(RenderBarrierRefusal::UnsupportedStages {
            side: "after",
            raw: 4,
        })
    );
    let unidentified = render::Command {
        kind: RenderKind::BarrierScope,
        barrier_scope: 1,
        barrier_unidentified_u8: 1,
        barrier_after_stages: 1,
        barrier_before_stages: 2,
        ..Default::default()
    };
    assert_eq!(
        resolved_render_barrier(&state, &host, 1, &unidentified),
        Err(RenderBarrierRefusal::UnidentifiedField { raw: 1 })
    );
}

#[test]
fn a_render_barrier_survives_a_refused_draw_until_one_is_recorded() {
    let barrier = reims_vgpu_core::RenderBarrier::Texture;
    let mut pending = vec![barrier.clone()];
    retire_render_barriers_after(EncodeStatus::MissingPipeline("test_refusal"), &mut pending);
    assert_eq!(pending, [barrier], "a pre-backend refusal consumes nothing");
    retire_render_barriers_after(EncodeStatus::Ok, &mut pending);
    assert!(
        pending.is_empty(),
        "the recorded draw consumes the dependency"
    );
}

#[test]
fn a_malformed_render_barrier_refuses_only_later_state_snapshots() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    assert!(
        acc.bind_snapshot().is_ok(),
        "state before the barrier is valid"
    );

    let mut command = vec![0u8; wire_render::MEMORY_BARRIER_SCOPE_TOTAL_LEN as usize];
    st32(&mut command, wire_render::OPCODE_MEMORY_BARRIER_SCOPE);
    st32(
        &mut command[4..],
        wire_render::MEMORY_BARRIER_SCOPE_TOTAL_LEN,
    );
    command[reims_vgpu_wire::OP_HEADER_LEN..][..4].copy_from_slice(&[1, 1, 1, 2]);
    handle_render_record(
        &mut state,
        &host,
        1,
        wire_render::OPCODE_MEMORY_BARRIER_SCOPE,
        &command,
        &mut out,
        &mut acc,
    );
    assert!(matches!(
        acc.bind_snapshot(),
        Err(StreamRefusal::Barrier(
            RenderBarrierRefusal::UnidentifiedField { raw: 1 }
        ))
    ));
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
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_core::endian::st64;
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
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_core::endian::st64;
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
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    use reims_vgpu_core::endian::st64;

    let total = reims_vgpu_wire::OP_HEADER_LEN
        + render::BIND_ENTRIES
        + render::BUFFER_STRIDE_BIND_ENTRY_SIZE;
    let mut command = vec![0u8; total];
    st32(
        &mut command[0..],
        wire_render::OPCODE_SET_VERTEX_BUFFER_STRIDE,
    );
    st32(&mut command[4..], total as u32);
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_FIRST..],
        4,
    );
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_COUNT..],
        1,
    );
    let e = reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES;
    st32(&mut command[e..], 5151);
    st64(&mut command[e + 4..], 0x2345);
    st64(&mut command[e + 12..], 0x3456);

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    let plain_total =
        reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES + render::BUFFER_BIND_ENTRY_SIZE;
    let mut plain = vec![0u8; plain_total];
    st32(&mut plain[0..], wire_render::OPCODE_SET_VERTEX_BUFFER);
    st32(&mut plain[4..], plain_total as u32);
    st32(
        &mut plain[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_COUNT..],
        1,
    );
    st32(
        &mut plain[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES..],
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
/// `setTriangleFillMode:`, `setLineWidth:`, and `setDepthClipMode:` used to be
/// rows here and are not any more: all now reach a backend. Their state tests
/// replace the old dropped counters.
#[test]
fn every_decoded_but_unapplied_render_state_reaches_its_own_counter() {
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_core::endian::st64;

    // (opcode, total length, payload writer, route, whether a default-valued
    // record of the same opcode must NOT count).
    type Writer = fn(&mut [u8]);
    // No `at_default` writer for the mode-shaped records any more: the only
    // two whose default half this exercised were the fill mode and the depth
    // clip mode, and both now reach a backend. The float pair below still has
    // one.
    let float_non_default: Writer = |p| st32(p, 2.5f32.to_bits());
    let float_at_default: Writer = |p| st32(p, 1.0f32.to_bits());

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
                + render::AMPLIFICATION_COUNT_LEN
                + 2 * render::AMPLIFICATION_MAPPING_SIZE,
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
            reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES + render::BUFFER_BIND_ENTRY_SIZE,
            |p| {
                st32(&mut p[render::BIND_FIRST..], 3);
                st32(&mut p[render::BIND_COUNT..], 1);
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
            reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES + render::REF_BIND_ENTRY_SIZE,
            |p| {
                st32(&mut p[render::BIND_FIRST..], 2);
                st32(&mut p[render::BIND_COUNT..], 1);
            },
            None,
            "render_tile_texture_bind_dropped",
        ),
        (
            wire_tile::OPCODE_SET_TILE_SAMPLER,
            reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES + render::REF_BIND_ENTRY_SIZE,
            |p| {
                st32(&mut p[render::BIND_FIRST..], 4);
                st32(&mut p[render::BIND_COUNT..], 1);
            },
            None,
            "render_tile_sampler_bind_dropped",
        ),
        (
            wire_tile::OPCODE_SET_TILE_SAMPLER_LOD,
            reims_vgpu_wire::OP_HEADER_LEN
                + render::BIND_ENTRIES
                + render::SAMPLER_LOD_BIND_ENTRY_SIZE,
            |p| {
                st32(&mut p[render::BIND_FIRST..], 5);
                st32(&mut p[render::BIND_COUNT..], 1);
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
        // the wrong arm would have had to accept the wrong length first.
        (
            wire_render::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS,
            20,
            |p| {
                st64(p, 0x1111);
                st32(&mut p[8..], 3);
            },
            None,
            "render_store_action_options_dropped",
        ),
        (
            wire_render::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS,
            16,
            |p| st64(p, 0x2222),
            None,
            "render_store_action_options_dropped",
        ),
        (
            wire_render::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS,
            16,
            |p| st64(p, 0x3333),
            None,
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
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
/// and travel to a draw as semantic state.
///
/// Both records share one 16-byte wire form and one decode arm, so the opcode
/// is the only thing that says which state a record sets — swapping the two
/// arms compiles, renders, and wireframes a pass that asked to be clamped.
/// Each is therefore driven on its own and the sibling asserted at its Metal
/// default.
///
/// The default value is latched too. A stream that sets Lines and then sets
/// Fill again is asking for Fill, so an arm that skipped `mode == 0` — which
/// is what the counter these replaced did, and correctly, since a counter is
/// only interested in the non-default — would leave the rest of the pass
/// wireframed.
#[test]
fn a_fill_mode_and_a_depth_clip_mode_reach_the_stream_state() {
    use reims_vgpu_core::endian::st64;

    let drive = |op: u32, mode: u64| {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
        assert_eq!(
            acc.raster.fill_mode,
            Ok(if mode == 0 {
                reims_vgpu_protocol::FillMode::Fill
            } else {
                reims_vgpu_protocol::FillMode::Lines
            }),
            "fill mode {mode}"
        );
        assert_eq!(
            acc.raster.depth_clip_mode,
            Ok(reims_vgpu_protocol::DepthClipMode::Clip),
            "fill mode {mode} set the sibling"
        );
        let acc = drive(wire_render::OPCODE_SET_DEPTH_CLIP_MODE, mode);
        assert_eq!(
            acc.raster.depth_clip_mode,
            Ok(if mode == 0 {
                reims_vgpu_protocol::DepthClipMode::Clip
            } else {
                reims_vgpu_protocol::DepthClipMode::Clamp
            }),
            "depth clip {mode}"
        );
        assert_eq!(
            acc.raster.fill_mode,
            Ok(reims_vgpu_protocol::FillMode::Fill),
            "depth clip {mode} set the sibling"
        );
    }

    // The record's field is 64 bits wide and the backend takes 32. A word
    // whose low half is zero must not arrive as the Metal default, which
    // would render silently; it arrives as a value no `MTLTriangleFillMode`
    // has, and makes the stream unrepresentable by its own name.
    let acc = drive(wire_render::OPCODE_SET_TRIANGLE_FILL_MODE, 1u64 << 32);
    assert_eq!(
        acc.raster.fill_mode,
        Err(RasterStateRefusal {
            field: RasterStateField::FillMode,
            raw: 1u64 << 32,
        })
    );
    assert!(matches!(
        acc.bind_snapshot(),
        Err(StreamRefusal::Raster(RasterStateRefusal {
            field: RasterStateField::FillMode,
            raw,
        })) if raw == 1u64 << 32
    ));

    // And a draw carries them. `bind_snapshot` builds its `PendingDraw` with
    // `..Default::default()`, so a field added to the accumulator and not to
    // the snapshot reaches no draw at all and nothing else would say so.
    let mut acc = drive(wire_render::OPCODE_SET_TRIANGLE_FILL_MODE, 1);
    acc.raster.depth_clip_mode = Ok(reims_vgpu_protocol::DepthClipMode::Clamp);
    let pd = acc.bind_snapshot().expect("state is representable");
    assert_eq!(pd.fill_mode, reims_vgpu_protocol::FillMode::Lines);
    assert_eq!(
        pd.depth_clip_mode,
        reims_vgpu_protocol::DepthClipMode::Clamp
    );
    let mut req = crate::runtime::draw::DrawEncodeRequest::default();
    fill_draw_binds_from_pending(&mut req, &pd);
    assert_eq!(req.fill_mode, reims_vgpu_protocol::FillMode::Lines);
    assert_eq!(
        req.depth_clip_mode,
        reims_vgpu_protocol::DepthClipMode::Clamp
    );
}

#[test]
fn line_width_latches_every_bit_pattern_and_reaches_the_draw() {
    use reims_vgpu_core::endian::st32;

    let drive = |bits: u32| {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        let mut command = vec![0u8; 12];
        st32(&mut command[0..], wire_render::OPCODE_SET_LINE_WIDTH);
        st32(&mut command[4..], 12);
        st32(&mut command[reims_vgpu_wire::OP_HEADER_LEN..], bits);
        handle_render_record(
            &mut state,
            &host,
            1,
            wire_render::OPCODE_SET_LINE_WIDTH,
            &command,
            &mut out,
            &mut acc,
        );
        acc
    };

    assert_eq!(
        StreamAccum::default().raster.line_width,
        reims_vgpu_core::LineWidth::ONE
    );
    for bits in [0.0f32.to_bits(), 4.0f32.to_bits(), f32::NAN.to_bits()] {
        let acc = drive(bits);
        assert_eq!(acc.raster.line_width.bits(), bits);
        let pd = acc.bind_snapshot().expect("line width is semantic state");
        assert_eq!(pd.line_width.bits(), bits);
        let mut req = crate::runtime::draw::DrawEncodeRequest::default();
        fill_draw_binds_from_pending(&mut req, &pd);
        assert_eq!(req.line_width.bits(), bits);
    }
}

#[test]
fn invalid_sticky_raster_state_refuses_until_that_field_is_replaced() {
    use reims_vgpu_core::endian::st64;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    let mut drive = |op: u32, raw: u64, acc: &mut StreamAccum| {
        let mut command = vec![0u8; wire_render::SET_MODE_TOTAL_LEN as usize];
        st32(&mut command[0..], op);
        st32(&mut command[4..], wire_render::SET_MODE_TOTAL_LEN);
        st64(&mut command[reims_vgpu_wire::OP_HEADER_LEN..], raw);
        handle_render_record(&mut state, &host, 1, op, &command, &mut out, acc);
    };

    drive(wire_render::OPCODE_SET_CULL_MODE, 1u64 << 32, &mut acc);
    assert!(matches!(
        acc.bind_snapshot(),
        Err(StreamRefusal::Raster(RasterStateRefusal {
            field: RasterStateField::CullMode,
            raw,
        })) if raw == 1u64 << 32
    ));

    drive(wire_render::OPCODE_SET_CULL_MODE, 2, &mut acc);
    assert_eq!(
        acc.bind_snapshot().unwrap().cull_mode,
        reims_vgpu_protocol::CullMode::Back
    );

    drive(wire_render::OPCODE_SET_FRONT_FACING, 7, &mut acc);
    assert!(matches!(
        acc.bind_snapshot(),
        Err(StreamRefusal::Raster(RasterStateRefusal {
            field: RasterStateField::FrontFacing,
            raw: 7,
        }))
    ));
    drive(wire_render::OPCODE_SET_FRONT_FACING, 1, &mut acc);
    let snapshot = acc.bind_snapshot().unwrap();
    assert_eq!(snapshot.cull_mode, reims_vgpu_protocol::CullMode::Back);
    assert!(snapshot.front_face_ccw);
}

#[test]
fn attachment_actions_refuse_snapshots_until_the_exact_attachment_is_replaced() {
    let mut acc = StreamAccum {
        depth_attach: Some(DepthAttachment {
            texture_ref: 11,
            load_action: 3,
            store_action: MTL_STORE_ACTION_STORE,
            clear_depth: 0.25,
            ..Default::default()
        }),
        ..Default::default()
    };
    assert!(matches!(
        acc.bind_snapshot(),
        Err(StreamRefusal::PassAction(PassActionRefusal {
            aspect: "depth",
            action: PassActionKind::Load,
            raw: 3,
            ..
        }))
    ));

    acc.depth_attach.as_mut().unwrap().load_action =
        reims_vgpu_protocol::pass_action::MTL_LOAD_ACTION_LOAD;
    acc.stencil_attach = Some(StencilAttachment {
        texture_ref: 12,
        load_action: MTL_LOAD_ACTION_CLEAR,
        store_action: 4,
        clear_stencil: 9,
        ..Default::default()
    });
    assert!(matches!(
        acc.bind_snapshot(),
        Err(StreamRefusal::PassAction(PassActionRefusal {
            aspect: "stencil",
            action: PassActionKind::Store,
            raw: 4,
            ..
        }))
    ));

    acc.stencil_attach.as_mut().unwrap().store_action = MTL_STORE_ACTION_STORE;
    acc.color_slots.push((
        2,
        ColorAttachment {
            texture_ref: 13,
            load_action: MTL_LOAD_ACTION_CLEAR,
            store_action: 5,
            ..Default::default()
        },
    ));
    assert!(matches!(
        acc.bind_snapshot(),
        Err(StreamRefusal::PassAction(PassActionRefusal {
            aspect: "color",
            slot: 2,
            action: PassActionKind::Store,
            raw: 5,
        }))
    ));

    acc.color_slots[0].1.store_action = MTL_STORE_ACTION_STORE;
    let snapshot = acc
        .bind_snapshot()
        .expect("all attachment actions are semantic");
    assert_eq!(
        snapshot.depth_attach.unwrap().load_action,
        reims_vgpu_protocol::pass_action::LoadAction::Load
    );
    assert_eq!(
        snapshot.stencil_attach.unwrap().store_action,
        reims_vgpu_protocol::pass_action::StoreAction::Store
    );
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
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
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
    let r = process_exec_indirect2(&mut state, &mut host, &payload);
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
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    let entry = render::BUFFER_BIND_ENTRY_SIZE;
    let total = reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES + entry;
    let mut command = vec![0u8; total];
    let op = wire_render::OPCODE_SET_VERTEX_BUFFER;
    st32(&mut command[0..], op);
    st32(&mut command[4..], total as u32);
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_FIRST..],
        FIRST,
    );
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_COUNT..],
        1,
    );
    st32(
        &mut command[reims_vgpu_wire::OP_HEADER_LEN + render::BIND_ENTRIES..],
        0x4444,
    );
    {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
    use crate::runtime::decode::render::{BUFFER_OFFSET_INDEX, BUFFER_OFFSET_PAYLOAD_LEN};
    use crate::runtime::drain::store_route_count;

    const FIRST: u32 = MAX_BUFFER_BIND_SLOTS + 1;

    let total = OP_HEADER_LEN + BUFFER_OFFSET_PAYLOAD_LEN;
    let mut command = vec![0u8; total];
    let op = wire_render::OPCODE_SET_VERTEX_BUFFER_OFFSET;
    st32(&mut command[0..], op);
    st32(&mut command[4..], total as u32);
    st32(&mut command[OP_HEADER_LEN + BUFFER_OFFSET_INDEX..], FIRST);

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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
            stage: render::Stage::Vertex,
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
    let mut state = Device::new(DeviceId(0), PAGE_SHIFT_ARM64E);
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
        acc.visibility.resolved(),
        Ok(Some(crate::runtime::draw::VisibilityArming {
            mode: reims_vgpu_protocol::VisibilityResultMode::Counting,
            offset: 0x1234
        })),
        "a counting query keeps both its mode and the offset it writes to"
    );

    // A second record replaces the first: this is encoder state, and Metal's
    // second `setVisibilityResultMode:` genuinely supersedes the first.
    arm(&mut acc, 0x20, 1);
    assert_eq!(
        acc.visibility.resolved(),
        Ok(Some(crate::runtime::draw::VisibilityArming {
            mode: reims_vgpu_protocol::VisibilityResultMode::Boolean,
            offset: 0x20
        })),
        "a second arming replaces the first rather than accumulating"
    );

    // Disabled clears it, rather than arming a query with mode 0.
    arm(&mut acc, 0x20, 0);
    assert_eq!(
        acc.visibility.resolved(),
        Ok(None),
        "MTLVisibilityResultModeDisabled disarms; it is not a third mode"
    );

    // Preserve all 64 wire bits. Narrowing this value before validation used
    // to turn it into Disabled and silently disarm a query the guest did not
    // ask to disarm.
    let invalid = (1u64 << 32) | 1;
    arm(&mut acc, 0x40, invalid);
    assert_eq!(
        acc.visibility.resolved(),
        Err(VisibilityStateRefusal { raw: invalid })
    );
    assert!(matches!(
        acc.bind_snapshot(),
        Err(StreamRefusal::Visibility(VisibilityStateRefusal { raw })) if raw == invalid
    ));

    // A later valid setter replaces only this sticky field and makes snapshots
    // executable again.
    arm(&mut acc, 0x48, 2);
    assert_eq!(
        acc.bind_snapshot().unwrap().visibility,
        Some(crate::runtime::draw::VisibilityArming {
            mode: reims_vgpu_protocol::VisibilityResultMode::Counting,
            offset: 0x48,
        })
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
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_BUFFER, RESOURCE_PAGE_SHIFT,
    };
    use crate::runtime::gva_mem;
    use crate::runtime::host::HostMemory;
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);

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

    // A type-1 buffer of one page at handle 5, named by object ref 7 — the
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
fn a_render_encoder_fence_reaches_the_shared_fence_object() {
    use crate::runtime::decode::stream::{SEGMENT_HEADER_LEN, SEGMENT_TYPE_RENDER};
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
    stream[4] = SEGMENT_TYPE_RENDER;
    stream.extend_from_slice(&records);

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    walk_stream(&mut state, &mut host, 1, &stream, &mut out, &mut acc);

    // Two updates: the first seeds the generation, the second advances it. A
    // dropped fence leaves `None` here, which is what this used to read.
    assert_eq!(
        state.fence_generation(1, FENCE_REF),
        Some(2),
        "both render updates landed on the fence object"
    );
    assert_eq!(
        acc.render_work,
        vec![RenderWork::Barrier(reims_vgpu_core::RenderBarrier::Fence {
            after: reims_vgpu_core::RenderBarrierStages::FRAGMENT,
            before: reims_vgpu_core::RenderBarrierStages::FRAGMENT,
        })],
        "the wait keeps both stage masks at its encoder position"
    );
}

/// A nil fence is an unbound operation, so its companion stage word carries no
/// dependency to interpret. Serializer storage outside the live reference may
/// contain any bits; those bits must not poison the encoder and suppress later
/// work. The same unknown mask on a live fence remains a typed refusal.
#[test]
fn a_nil_render_fence_does_not_validate_stale_stage_bits() {
    use crate::runtime::decode::stream::{SEGMENT_HEADER_LEN, SEGMENT_TYPE_RENDER};
    use reims_vgpu_wire::ops::render as wire_render;

    const LIVE_FENCE_REF: u32 = 6464;
    const STAGES_FRAGMENT: u32 = 2;
    const STAGES_UNKNOWN: u32 = 1 << 5;

    fn stream_with_fences(records: &[(u32, u32, u32)]) -> Vec<u8> {
        let mut payload = Vec::new();
        for &(opcode, fence_ref, stages) in records {
            let mut hdr = [0u8; 8];
            st32(&mut hdr[0..4], opcode);
            st32(&mut hdr[4..8], wire_render::FENCE_TOTAL_LEN);
            payload.extend_from_slice(&hdr);
            let mut fence = [0u8; 8];
            st32(&mut fence[0..4], fence_ref);
            st32(&mut fence[4..8], stages);
            payload.extend_from_slice(&fence);
        }

        let mut stream = vec![0u8; SEGMENT_HEADER_LEN];
        let stream_len = stream.len() + payload.len();
        st32(&mut stream[0..4], stream_len as u32);
        stream[4] = SEGMENT_TYPE_RENDER;
        stream.extend_from_slice(&payload);
        stream
    }

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    let stream = stream_with_fences(&[
        (wire_render::OPCODE_UPDATE_FENCE, 0, STAGES_UNKNOWN),
        (
            wire_render::OPCODE_UPDATE_FENCE,
            LIVE_FENCE_REF,
            STAGES_FRAGMENT,
        ),
        (
            wire_render::OPCODE_WAIT_FOR_FENCE,
            LIVE_FENCE_REF,
            STAGES_FRAGMENT,
        ),
    ]);
    walk_stream(&mut state, &mut host, 1, &stream, &mut out, &mut acc);

    assert!(
        acc.unrepresentable.is_none(),
        "stage bits beside an unbound fence do not invalidate the stream"
    );
    assert_eq!(state.fence_generation(1, 0), None);
    assert_eq!(state.fence_generation(1, LIVE_FENCE_REF), Some(1));
    assert_eq!(
        acc.render_work,
        vec![RenderWork::Barrier(reims_vgpu_core::RenderBarrier::Fence {
            after: reims_vgpu_core::RenderBarrierStages::FRAGMENT,
            before: reims_vgpu_core::RenderBarrierStages::FRAGMENT,
        })]
    );

    let mut live_state = Device::new(DeviceId(2), PAGE_SHIFT_ARM64E);
    let mut live_host = FakeHost::new();
    let mut live_out = ExecResult::default();
    let mut live_acc = StreamAccum::default();
    let live_unknown = stream_with_fences(&[(
        wire_render::OPCODE_UPDATE_FENCE,
        LIVE_FENCE_REF,
        STAGES_UNKNOWN,
    )]);
    walk_stream(
        &mut live_state,
        &mut live_host,
        1,
        &live_unknown,
        &mut live_out,
        &mut live_acc,
    );
    assert_eq!(
        live_acc.unrepresentable,
        Some(StreamRefusal::Barrier(
            RenderBarrierRefusal::FenceStagesUnsupported {
                raw: STAGES_UNKNOWN,
            }
        )),
        "the same unknown stage mask on a live fence remains fail-closed"
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
    use crate::runtime::decode::render::{
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

        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
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

/// The dirty-flag predicate is pointer identity, so an equal-but-separate table
/// is a *changed* state and not an unchanged one.
///
/// This is the property the whole reading rests on. `render_encoder_delta`
/// is allowed to be conservative — calling a genuinely unchanged pair "changed"
/// only forgoes a saving — but it must never call a changed pair unchanged, and
/// the way that would happen is by comparing contents instead of allocations.
/// Two tables holding equal bytes are two `Set` records; the guest re-stated
/// its binds and the encoder state was rebuilt, whatever the bytes say.
#[test]
fn encoder_state_is_unchanged_only_when_the_tables_are_the_same_allocation() {
    let base = PendingDraw {
        pipeline_ref: 7,
        ..Default::default()
    };

    let shared = PendingDraw {
        vertex_buffers: base.vertex_buffers.clone(),
        fragment_buffers: base.fragment_buffers.clone(),
        vertex_textures: base.vertex_textures.clone(),
        fragment_textures: base.fragment_textures.clone(),
        vertex_samplers: base.vertex_samplers.clone(),
        fragment_samplers: base.fragment_samplers.clone(),
        ..base.clone()
    };
    assert!(
        render_encoder_delta(&base, &shared).all_unchanged(),
        "cloned handles are the same allocation, so no Set record separated them"
    );
    assert!(render_encoder_delta(&base, &shared).all_unchanged());

    // A different pipeline with every table shared is still a changed state:
    // the reflected interface the binds project onto has moved.
    let repipelined = PendingDraw {
        pipeline_ref: 8,
        ..shared.clone()
    };
    assert!(
        !render_encoder_delta(&base, &repipelined).all_unchanged(),
        "a SetPipeline between the two draws changes which binding each bind lands on"
    );
    assert_eq!(
        render_encoder_delta(&base, &repipelined),
        reims_vgpu_core::RenderEncoderDelta {
            pipeline: true,
            ..reims_vgpu_core::RenderEncoderDelta::NONE_CHANGED
        }
    );

    // Equal contents in a fresh allocation: the guest re-stated the class, so
    // this must read as changed even though a byte comparison would not.
    let restated = PendingDraw {
        vertex_textures: std::sync::Arc::new((*base.vertex_textures).clone()),
        ..shared.clone()
    };
    assert!(
        !render_encoder_delta(&base, &restated).all_unchanged(),
        "an equal-but-separate table is a Set record and must never read as unchanged"
    );
    let restated_delta = render_encoder_delta(&base, &restated);
    assert!(restated_delta.vertex_textures);
    assert!(!restated_delta.pipeline);
    assert!(!restated_delta.vertex_buffers);
    assert!(!restated_delta.fragment_buffers);
    assert!(!restated_delta.fragment_textures);
    assert!(!restated_delta.vertex_samplers);
    assert!(!restated_delta.fragment_samplers);
}
