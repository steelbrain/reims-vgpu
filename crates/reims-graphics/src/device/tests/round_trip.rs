//! What a guest's synchronous ring-0 create costs from its post to its answer, as the kext waits
//! for it: a measurement, run by hand (`--run-ignored only --no-capture`).

use std::time::Instant;

use super::*;

/// A host that answers a render pipeline's create at once, with its reflection, as a create the
/// host's caches hold is answered, and carries out every other message doing nothing.
struct AnswerHost;

impl Host for AnswerHost {
    fn raise_interrupt(&self, _vector: u32) {}
    fn scanout_written(&self, _offset: u64, _length: u64) -> bool {
        false
    }
    fn stage3_message(&self, completion: &CompletionPoster, message: &[u8]) {
        let (header, _) = RgfxMsgHeader::read_from_prefix(message).unwrap();
        if header.r#type == RGFX_CMD_RENDER_PIPELINE_CREATE {
            let mut reflection = RgfxPipelineReflection::new_zeroed();
            reflection.header = RgfxMsgHeader {
                r#type: RGFX_EVT_PIPELINE_REFLECTION,
                size: size_of::<RgfxPipelineReflection>() as u32,
                seqno: header.seqno,
            };
            completion.post(reflection.as_bytes());
        }
    }
}

/// The next event of the completion ring, its seqno, read as soon as it is posted (spinning).
#[allow(clippy::disallowed_methods)] // a measurement: the wall clock's time
fn answer_now(rig: &mut Rig) -> u64 {
    let data = RGFX_CONTROL_RING_DATA_SIZE as u64;
    let at = |count: u64| RGFX_CONTROL_COMPLETE_OFFSET + 256 + (count % data);
    let start = Instant::now();
    loop {
        let produced = (rig.region())
            .atomic_u64(RGFX_CONTROL_COMPLETE_OFFSET)
            .load(Ordering::Acquire);
        if produced != rig.consumed {
            let head = rig.region().read(at(rig.consumed), 16).unwrap();
            let size = u32::from_le_bytes(head[4..8].try_into().unwrap()) as u64;
            let r#type = u32::from_le_bytes(head[0..4].try_into().unwrap());
            let seqno = u64::from_le_bytes(head[8..16].try_into().unwrap());
            rig.consumed += size;
            (rig.region())
                .atomic_u64(RGFX_CONTROL_COMPLETE_OFFSET + 0x40)
                .store(rig.consumed, Ordering::Release);
            if r#type != RGFX_MSG_PAD {
                return seqno;
            }
            continue;
        }
        assert!(start.elapsed() < Duration::from_secs(2), "no answer");
        std::hint::spin_loop();
    }
}

/// Microseconds' mean, median and 99th percentile.
fn stats(mut us: Vec<f64>) -> (f64, f64, f64) {
    us.sort_by(f64::total_cmp);
    let mean = us.iter().sum::<f64>() / us.len() as f64;
    (mean, us[us.len() / 2], us[us.len() * 99 / 100])
}

/// A context's synchronous creates, each posted once the one before it is answered, as the kext
/// posts a create it waits for: back to back (`warm`), each after `ASYNC` of the context's creates
/// it does not wait for (`behind`), and each after a pause longer than a lane's thread waits
/// (`cold`). Prints the mean, median and 99th percentile of each, from the post's doorbell to
/// the answer read.
#[test]
#[ignore = "a measurement, run by hand"]
#[allow(clippy::disallowed_methods)] // a measurement: the wall clock's time
fn a_sync_create_s_round_trip() {
    const CONTEXT: u64 = 5;
    const ROUNDS: usize = 5000;
    const ASYNC: usize = 4;
    const COLD: usize = 5;
    let mut rig = Rig::with_host(Box::new(AnswerHost));
    rig.start();
    let mut next = 1u64;
    let mut obj = || {
        next += 1;
        (CONTEXT << RGFX_CONTEXT_OBJ_ID_SHIFT) | next
    };
    let round = |rig: &mut Rig, obj_id: u64| {
        let t0 = Instant::now();
        let seqno = rig.send(object_message(RGFX_CMD_RENDER_PIPELINE_CREATE, obj_id));
        assert_eq!(answer_now(rig), seqno);
        t0.elapsed().as_secs_f64() * 1e6
    };
    // The first, which starts what carries the context's messages out.
    let first = round(&mut rig, obj());
    let warm: Vec<f64> = (0..ROUNDS).map(|_| round(&mut rig, obj())).collect();
    let behind: Vec<f64> = (0..ROUNDS)
        .map(|_| {
            for _ in 0..ASYNC {
                rig.send(object_message(RGFX_CMD_EVENT_CREATE, obj()));
            }
            round(&mut rig, obj())
        })
        .collect();
    let cold: Vec<f64> = (0..COLD)
        .map(|_| {
            std::thread::sleep(Duration::from_millis(1200));
            round(&mut rig, obj())
        })
        .collect();
    println!("a_sync_create_s_round_trip: first {first:.1} us");
    for (name, us) in [("warm", warm), ("behind", behind), ("cold", cold)] {
        let n = us.len();
        let (mean, p50, p99) = stats(us);
        println!(
            "a_sync_create_s_round_trip: {name}: mean {mean:.1} us, p50 {p50:.1}, p99 {p99:.1} \
             ({n} rounds)"
        );
    }
}

/// A context's bursts of `BURST` creates it does not wait for, each followed by a synchronous
/// create, beside `IDLE` other contexts whose lanes have been made and have nothing to do. Prints
/// the time from the burst's first post to the create's answer, with no other context and with
/// them.
#[test]
#[ignore = "a measurement, run by hand"]
#[allow(clippy::disallowed_methods)] // a measurement: the wall clock's time
fn a_burst_beside_idle_lanes() {
    const CONTEXT: u64 = 5;
    const BURST: usize = 200;
    const ROUNDS: usize = 300;
    for idle in [0u64, 75] {
        let mut rig = Rig::with_host(Box::new(AnswerHost));
        rig.start();
        let mut next = 1u64;
        let mut obj = |context: u64| {
            next += 1;
            (context << RGFX_CONTEXT_OBJ_ID_SHIFT) | next
        };
        for other in 0..idle {
            let seqno = rig.send(object_message(
                RGFX_CMD_RENDER_PIPELINE_CREATE,
                obj(CONTEXT + 1 + other),
            ));
            assert_eq!(answer_now(&mut rig), seqno);
        }
        let us: Vec<f64> = (0..ROUNDS)
            .map(|_| {
                let t0 = Instant::now();
                for _ in 0..BURST {
                    rig.send(object_message(RGFX_CMD_EVENT_CREATE, obj(CONTEXT)));
                }
                let seqno = rig.send(object_message(
                    RGFX_CMD_RENDER_PIPELINE_CREATE,
                    obj(CONTEXT),
                ));
                assert_eq!(answer_now(&mut rig), seqno);
                t0.elapsed().as_secs_f64() * 1e6
            })
            .collect();
        let (mean, p50, p99) = stats(us);
        println!(
            "a_burst_beside_idle_lanes: {idle} idle lanes: mean {mean:.1} us, p50 {p50:.1}, p99 \
             {p99:.1} ({BURST} creates and a synchronous one, {ROUNDS} rounds)"
        );
    }
}

/// A host that answers a render pipeline's create at once, as [`AnswerHost`] does, holds each
/// `BUFFER_CREATE` while `held` is set, and completes each command buffer at once.
struct HoldingHost {
    held: Arc<AtomicBool>,
}

impl Host for HoldingHost {
    fn raise_interrupt(&self, _vector: u32) {}
    fn scanout_written(&self, _offset: u64, _length: u64) -> bool {
        false
    }
    fn stage3_message(&self, completion: &CompletionPoster, message: &[u8]) {
        let (header, _) = RgfxMsgHeader::read_from_prefix(message).unwrap();
        match header.r#type {
            RGFX_CMD_BUFFER_CREATE => {
                while self.held.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            RGFX_CMD_COMMAND_BUFFER => {
                let (cb, _) = crate::ops::RgfxCommandBuffer::read_from_prefix(message).unwrap();
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
            _ => AnswerHost.stage3_message(completion, message),
        }
    }
}

/// Publishes an empty `COMMAND_BUFFER` `cb_obj_id` of queue `queue` at the start of the
/// submission ring at `submit`, with seqno `seqno`, and rings its doorbell.
fn publish_command_buffer_at(rig: &Rig, submit: u64, queue: u64, cb_obj_id: u64, seqno: u64) {
    use crate::ops::{RGFX_COMMAND_BUFFER_OPS_OFFSET, RgfxCommandBuffer};
    let cb = RgfxCommandBuffer {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_COMMAND_BUFFER,
            size: RGFX_COMMAND_BUFFER_OPS_OFFSET,
            seqno,
        },
        queue_obj_id: queue,
        cb_obj_id,
        op_bytes: 0,
    };
    let bytes = cb.as_bytes();
    rig.region().write(submit + 256, bytes);
    rig.region()
        .atomic_u64(submit)
        .store(bytes.len() as u64, Ordering::SeqCst);
    rig.device
        .mmio_write(RGFX_REG_DOORBELL.into(), submit as u32);
}

/// Where the `n`th waiting queue's submission ring is, and its completion ring after it.
fn waiting_rings(n: u64) -> (u64, u64) {
    let submit = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000 + n * 0x4_0000;
    (submit, submit + 0x2_0000)
}

/// A context's bursts of `BURST` creates it does not wait for, each followed by a synchronous
/// create, beside `WAITING` other contexts, each with a queue whose command buffer waits for a
/// create of its context's the host holds. Prints the time from the burst's first post to the
/// create's answer, with no queue waiting and with them.
#[test]
#[ignore = "a measurement, run by hand"]
#[allow(clippy::disallowed_methods)] // a measurement: the wall clock's time
fn a_burst_beside_waiting_queues() {
    const CONTEXT: u64 = 5;
    const BURST: usize = 200;
    const ROUNDS: usize = 300;
    let span = u64::from(RGFX_RING_HEADER_SIZE + QUEUE_RING_DATA);
    for waiting in [0u64, 75] {
        let held = Arc::new(AtomicBool::new(false));
        let mut rig = Rig::with_host(Box::new(HoldingHost {
            held: Arc::clone(&held),
        }));
        rig.start();
        let mut next = 1u64;
        let mut obj = |context: u64| {
            next += 1;
            (context << RGFX_CONTEXT_OBJ_ID_SHIFT) | next
        };
        // Each other context's queue, made.
        let mut queues = Vec::new();
        for n in 0..waiting {
            let context = CONTEXT + 1 + n;
            let (submit, complete) = waiting_rings(n);
            for ring in [submit, complete] {
                rig.send(block_message(
                    RGFX_CMD_BLOCK_GRANT,
                    context as u32,
                    ring,
                    span,
                ));
            }
            let queue = obj(context);
            let mut create = RgfxQueueCreate::read_from_bytes(&queue_create(queue)).unwrap();
            create.submit_offset = submit;
            create.complete_offset = complete;
            let seqno = rig.send(create.as_bytes().to_vec());
            assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
            queues.push(queue);
        }
        // A create of each held, and a command buffer of each that waits for it.
        held.store(true, Ordering::SeqCst);
        for n in 0..waiting {
            let context = CONTEXT + 1 + n;
            rig.send(object_message(RGFX_CMD_BUFFER_CREATE, obj(context)));
        }
        for (n, &queue) in queues.iter().enumerate() {
            let context = CONTEXT + 1 + n as u64;
            let (submit, _) = waiting_rings(n as u64);
            publish_command_buffer_at(&rig, submit, queue, obj(context), 1);
        }
        std::thread::sleep(Duration::from_millis(200));
        let us: Vec<f64> = (0..ROUNDS)
            .map(|_| {
                let t0 = Instant::now();
                for _ in 0..BURST {
                    rig.send(object_message(RGFX_CMD_EVENT_CREATE, obj(CONTEXT)));
                }
                let seqno = rig.send(object_message(
                    RGFX_CMD_RENDER_PIPELINE_CREATE,
                    obj(CONTEXT),
                ));
                assert_eq!(answer_now(&mut rig), seqno);
                t0.elapsed().as_secs_f64() * 1e6
            })
            .collect();
        held.store(false, Ordering::SeqCst);
        let (mean, p50, p99) = stats(us);
        println!(
            "a_burst_beside_waiting_queues: {waiting} waiting queues: mean {mean:.1} us, p50 \
             {p50:.1}, p99 {p99:.1} ({BURST} creates and a synchronous one, {ROUNDS} rounds)"
        );
    }
}
