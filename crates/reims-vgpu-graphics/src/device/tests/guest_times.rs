//! The guest's times of its command buffers (`RGFX_FEATURE_GUEST_TIMES`, `device::guest_times`):
//! a buffer whose stream starts with them gives its spans to the host's trace as a queue's thread
//! takes it, by the clock offset ring 0's header holds.

use super::*;
use crate::ops::{
    RGFX_COMMAND_BUFFER_OPS_OFFSET, RGFX_OP_GUEST_TIMES, RgfxCommandBuffer, RgfxOpGuestTimes,
    RgfxOpHeader,
};

const QUEUE: u64 = (9 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
/// The buffer the host holds [`HOLD`] in its call, so the next one waits behind it.
const HELD: u64 = 100;
const HOLD: Duration = Duration::from_millis(20);

/// What a [`TimesHost`] saw: the trace's events, and the command buffers it carried out.
#[derive(Default)]
struct TimesSeen {
    events: Mutex<Vec<(&'static str, Vec<u64>)>>,
    buffers: Mutex<Vec<u64>>,
}

/// A host that offers `GUEST_TIMES`, records its trace's events, and completes each command
/// buffer, holding [`HELD`] for [`HOLD`] first.
struct TimesHost(Arc<TimesSeen>);

impl Host for TimesHost {
    fn features(&self) -> u64 {
        RGFX_FEATURE_GUEST_TIMES
    }
    fn raise_interrupt(&self, _vector: u32) {}
    fn trace(&self, kind: &'static str, values: &[u64]) {
        self.0.events.lock().push((kind, values.to_vec()));
    }
    fn stage3_message(&self, completion: &crate::device::CompletionPoster, message: &[u8]) {
        let (header, _) = RgfxMsgHeader::read_from_prefix(message).unwrap();
        if header.r#type != RGFX_CMD_COMMAND_BUFFER {
            return;
        }
        let (cb, _) = RgfxCommandBuffer::read_from_prefix(message).unwrap();
        if cb.cb_obj_id == HELD {
            std::thread::sleep(HOLD);
        }
        self.0.buffers.lock().push(cb.cb_obj_id);
        let done = RgfxCbComplete {
            header: RgfxMsgHeader {
                r#type: RGFX_EVT_CB_COMPLETE,
                size: size_of::<RgfxCbComplete>() as u32,
                seqno: header.seqno,
            },
            cb_obj_id: cb.cb_obj_id,
            gpu_start_ns: 0,
            gpu_end_ns: 0,
            status: RGFX_CB_OK,
            error_code: 0,
        };
        completion.post(done.as_bytes());
    }
}

/// A started rig over a [`TimesHost`] whose guest took `GUEST_TIMES` where `took`, with
/// [`QUEUE`] made and its ring idle.
fn rig(seen: &Arc<TimesSeen>, took: bool) -> Rig {
    let mut rig = Rig::with_host(Box::new(TimesHost(Arc::clone(seen))));
    if took {
        rig.device.mmio_write(
            RGFX_REG_DRIVER_FEATURES_HI.into(),
            (RGFX_FEATURE_GUEST_TIMES >> 32) as u32,
        );
    }
    rig.start();
    let seqno = send_queue_create(&mut rig, QUEUE);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    wait_queue_idle(&rig);
    rig
}

/// Ring 0's clock offset and its bound, as the kext writes them: the offset, then the bound.
fn set_clock(rig: &Rig, offset_ns: i64, bound_ns: u32) {
    rig.region()
        .atomic_u64(RGFX_CONTROL_SUBMIT_OFFSET + 0x18)
        .store(offset_ns.cast_unsigned(), Ordering::SeqCst);
    rig.region()
        .atomic_u32(RGFX_CONTROL_SUBMIT_OFFSET + 0x20)
        .store(bound_ns, Ordering::Release);
}

/// A `COMMAND_BUFFER` `cb_obj_id` of [`QUEUE`] at `seqno`, its stream an empty blit pass, the
/// guest's `times` first where given.
fn buffer(cb_obj_id: u64, seqno: u64, times: Option<RgfxOpGuestTimes>) -> Vec<u8> {
    let mut ops = Vec::new();
    if let Some(times) = times {
        ops.extend_from_slice(
            RgfxOpHeader {
                tag: RGFX_OP_GUEST_TIMES,
                flags: 0,
                len: 32,
            }
            .as_bytes(),
        );
        ops.extend_from_slice(times.as_bytes());
    }
    for tag in [
        crate::ops::RGFX_OP_ENCODER_BEGIN_BLIT,
        crate::ops::RGFX_OP_ENCODER_END,
    ] {
        ops.extend_from_slice(
            RgfxOpHeader {
                tag,
                flags: 0,
                len: 8,
            }
            .as_bytes(),
        );
    }
    let mut m = RgfxCommandBuffer {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_COMMAND_BUFFER,
            size: RGFX_COMMAND_BUFFER_OPS_OFFSET + ops.len() as u32,
            seqno,
        },
        queue_obj_id: QUEUE,
        cb_obj_id,
        op_bytes: ops.len() as u64,
    }
    .as_bytes()
    .to_vec();
    m.extend_from_slice(&ops);
    m
}

/// The guest's times of a buffer posted now, on the guest's clock `offset_ns` ahead of the
/// host's: made 3 ms and committed 1 ms before its post.
fn posted_now(offset_ns: i64) -> RgfxOpGuestTimes {
    let posted = now_ns().checked_add_signed(offset_ns).unwrap();
    RgfxOpGuestTimes {
        began_ns: posted - 3_000_000,
        committed_ns: posted - 1_000_000,
        posted_ns: posted,
    }
}

/// The `cbguest` events the host's trace holds.
fn events(seen: &TimesSeen) -> Vec<Vec<u64>> {
    (seen.events.lock().iter())
        .filter(|(kind, _)| *kind == "cbguest")
        .map(|(_, values)| values.clone())
        .collect()
}

/// A stamped buffer's spans reach the trace as its queue's thread takes it: its encode and its
/// wait to post from its stamps; its transport from its post, mapped by ring 0's offset, to its
/// take, within the span the test saw; and a buffer posted while the thread carried out the one
/// before it waited behind that one (`queued`), its transport from the thread's return
#[test]
fn a_stamped_buffer_s_spans_reach_the_trace() {
    const OFFSET: i64 = 5_000_000_000;
    let seen = Arc::new(TimesSeen::default());
    let rig = rig(&seen, true);
    set_clock(&rig, OFFSET, 2_000);

    let before = now_ns();
    let at = publish_queue_message(&rig, 0, 1, buffer(101, 1, Some(posted_now(OFFSET))));
    assert!(eventually(|| seen.buffers.lock().len() == 1));
    let took = now_ns() - before;
    let first = events(&seen);
    assert_eq!(first.len(), 1, "{first:?}");
    let e = &first[0];
    assert_eq!(e.len(), 8, "{e:?}");
    assert_eq!(e[..4], [QUEUE, 1, 3_000_000 - 1_000_000, 1_000_000]);
    assert_eq!(
        (e[4], e[6]),
        (0, 2_000),
        "nothing before it on the ring: {e:?}"
    );
    assert!(
        e[5] + e[7] <= took,
        "transport {} and held {} past the {took} ns the test saw",
        e[5],
        e[7]
    );
    wait_queue_idle(&rig);

    // Two at once: the second waits behind the first, which the host holds.
    let mut first = buffer(HELD, 2, Some(posted_now(OFFSET)));
    let second = buffer(102, 3, Some(posted_now(OFFSET)));
    first.extend_from_slice(&second);
    publish_queue_message(&rig, at, 2, first);
    assert!(eventually(|| seen.buffers.lock().len() == 3));
    let all = events(&seen);
    assert_eq!(all.len(), 3, "{all:?}");
    assert_eq!(all[1][..2], [QUEUE, 2]);
    let e = &all[2];
    assert_eq!(e[..2], [QUEUE, 3]);
    let hold = HOLD.as_nanos() as u64;
    assert!(
        e[4] >= hold,
        "queued {} behind a buffer held {hold} ns",
        e[4]
    );
    assert_eq!(e[6], 2_000);
}

/// A buffer whose stream does not start with the guest's times gives no event, nor does a
/// stamped one where the guest did not take `GUEST_TIMES`: the device reads nothing of them
#[test]
fn a_buffer_without_its_times_gives_no_event() {
    for took in [true, false] {
        let seen = Arc::new(TimesSeen::default());
        let rig = rig(&seen, took);
        set_clock(&rig, 0, 1_000);
        let mut two = buffer(101, 1, None);
        if !took {
            two.extend_from_slice(&buffer(102, 2, Some(posted_now(0))));
        }
        let n = if took { 1 } else { 2 };
        publish_queue_message(&rig, 0, 1, two);
        assert!(eventually(|| seen.buffers.lock().len() == n), "took {took}");
        wait_queue_idle(&rig);
        assert_eq!(events(&seen), Vec::<Vec<u64>>::new(), "took {took}");
    }
}

/// While ring 0's bound is 0 (the guest holds no reading of the host's clock) a stamped buffer's
/// event carries its guest's spans alone: no queued, no transport
#[test]
fn with_no_clock_offset_no_transport_is_given() {
    let seen = Arc::new(TimesSeen::default());
    let rig = rig(&seen, true);
    set_clock(&rig, 1_000_000, 0);
    publish_queue_message(&rig, 0, 1, buffer(101, 1, Some(posted_now(1_000_000))));
    assert!(eventually(|| seen.buffers.lock().len() == 1));
    let all = events(&seen);
    assert_eq!(all.len(), 1, "{all:?}");
    assert_eq!(all[0].len(), 5, "{all:?}");
    assert_eq!(all[0][..4], [QUEUE, 1, 2_000_000, 1_000_000]);
}
