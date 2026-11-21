//! A command buffer's thread carries out the jobs of its context's lane it waits for (`lanes`, "A
//! command buffer's thread"): a `BUFFER_AT_OFFSET` or an `OBJECT_DESTROY` of no queue that no
//! thread of the lane's carries out yet. The lane keeps its order, each job is carried out once,
//! and the buffer is handed over after every message of its context's published before its take.
//!
//! It also takes the ring-0 messages it waits for where no thread takes ring 0
//! (`Shared::take_for`), as the control thread would, while the next is one of a context's lane:
//! ring 0 is taken, carried out and answered in ring order whichever thread takes each message.

use super::*;
use crate::ops::RgfxCommandBuffer;

/// A host that records each message it is handed ([`Seen`]) and each of ring 0's takes, takes
/// `slow`'s while over some of ring 0's messages, answers each where `answers` is set, and
/// completes each command buffer at once.
#[derive(Clone, Default)]
struct HelpHost {
    seen: Arc<Mutex<Vec<Seen>>>,
    /// The microseconds the host takes over a ring-0 message, by seqno.
    slow: Arc<Mutex<HashMap<u64, u64>>>,
    /// Ring 0's takes (`r0take`, [`Host::trace`]), each as a [`Seen`] of the taking thread.
    takes: Arc<Mutex<Vec<Seen>>>,
    /// Whether the host answers each ring-0 message it is handed with a `COMPLETE`.
    answers: Arc<AtomicBool>,
}

/// A message a [`HelpHost`] was handed: its type, its seqno (ring 0's, or the queue ring's for a
/// command buffer) and the thread it came on ([`thread_name`]).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Seen {
    r#type: u32,
    seqno: u64,
    thread: Option<String>,
}

impl HelpHost {
    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().clone()
    }

    /// The ring-0 messages of seqnos `sent` it was handed, in the order it was.
    fn ring0(&self, sent: &[u64]) -> Vec<Seen> {
        (self.seen().into_iter())
            .filter(|s| s.r#type != RGFX_CMD_COMMAND_BUFFER && sent.contains(&s.seqno))
            .collect()
    }

    /// Ring 0's takes of the messages of seqnos `sent`, in the order they were taken.
    fn takes(&self, sent: &[u64]) -> Vec<Seen> {
        (self.takes.lock().iter())
            .filter(|s| sent.contains(&s.seqno))
            .cloned()
            .collect()
    }
}

/// The seqnos of `seen`, in its order.
fn seqnos(seen: &[Seen]) -> Vec<u64> {
    seen.iter().map(|s| s.seqno).collect()
}

impl Host for HelpHost {
    fn raise_interrupt(&self, _vector: u32) {}
    fn stage3_message(&self, completion: &CompletionPoster, message: &[u8]) {
        let (header, _) = RgfxMsgHeader::read_from_prefix(message).unwrap();
        if header.r#type != RGFX_CMD_COMMAND_BUFFER {
            let slow = self.slow.lock().get(&header.seqno).copied();
            if let Some(us) = slow {
                std::thread::sleep(Duration::from_micros(us));
            }
        }
        self.seen.lock().push(Seen {
            r#type: header.r#type,
            seqno: header.seqno,
            thread: thread_name(),
        });
        if header.r#type != RGFX_CMD_COMMAND_BUFFER && self.answers.load(Ordering::SeqCst) {
            let done = RgfxComplete {
                header: RgfxMsgHeader {
                    r#type: RGFX_EVT_COMPLETE,
                    size: size_of::<RgfxComplete>() as u32,
                    seqno: header.seqno,
                },
                status: RGFX_STATUS_OK,
                reserved_14: 0,
            };
            completion.post(done.as_bytes());
        }
        if header.r#type == RGFX_CMD_COMMAND_BUFFER {
            let (cb, _) = RgfxCommandBuffer::read_from_prefix(message).unwrap();
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
    fn trace(&self, kind: &'static str, values: &[u64]) {
        if kind == "r0take" {
            self.takes.lock().push(Seen {
                r#type: values[0] as u32,
                seqno: values[1],
                thread: thread_name(),
            });
        }
    }
}

/// The tests' queue, of context 5.
const QUEUE: u64 = (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;

/// Object `n` of the queue's context.
fn object(n: u64) -> u64 {
    (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | n
}

/// A started rig over `host` with [`QUEUE`] made and its ring asleep.
fn queue_rig(host: &HelpHost) -> Rig {
    let mut rig = Rig::with_host(Box::new(host.clone()));
    rig.start();
    let seqno = send_queue_create(&mut rig, QUEUE);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    wait_queue_idle(&rig);
    rig
}

/// Whether `seen` came on the tests' queue's service thread.
fn on_queue(seen: &Seen) -> bool {
    (seen.thread.as_deref()).is_some_and(|name| name.starts_with("reims-vgpu.graphics-queue-"))
}

/// Publishes `command` on ring 0 as [`Rig::send`] does, but rings no doorbell: as a message the
/// guest published while the control thread is awake, or one whose doorbell has not yet woken
/// it. The command's `seqno`.
fn publish_unrung(rig: &mut Rig, mut command: Vec<u8>) -> u64 {
    rig.seqno += 1;
    command[8..16].copy_from_slice(&rig.seqno.to_le_bytes());
    set_size_of(&mut command);
    rig.publish(&command);
    rig.seqno
}

/// Waits, [`PATIENCE`] polls at most, for the control thread to sleep: ring 0's `consumer_idle`
/// set and the thread past its last look at the ring.
fn wait_control_asleep(rig: &Rig) {
    assert!(
        eventually(|| rig.device.shared.submit.lock().asleep),
        "the control thread never slept"
    );
    assert_eq!(control_idle(rig), 1);
}

/// Ring 0's `consumer_idle`.
fn control_idle(rig: &Rig) -> u32 {
    rig.region()
        .atomic_u32(RGFX_CONTROL_SUBMIT_OFFSET + 0x80)
        .load(Ordering::SeqCst)
}

/// With the control thread asleep, as one the guest's doorbell has not yet woken, a command
/// buffer's thread takes the ring-0 messages of its context's lane it waits for itself, in ring
/// order, each once, and hands them to the lane, which carries them out and answers them in that
/// order; the buffer is handed over after them, and ring 0's `consumer_idle` stays set, the control
/// thread's to clear.
#[test]
fn a_buffer_s_thread_takes_the_ring_0_messages_it_waits_for_in_ring_order() {
    let host = HelpHost::default();
    let mut rig = queue_rig(&host);
    host.answers.store(true, Ordering::SeqCst);
    wait_control_asleep(&rig);
    let sent: Vec<u64> = [
        (RGFX_CMD_BUFFER_AT_OFFSET, object(2)),
        (RGFX_CMD_EVENT_CREATE, object(3)),
        (RGFX_CMD_OBJECT_DESTROY, object(2)),
        (RGFX_CMD_BUFFER_CREATE, object(4)),
    ]
    .into_iter()
    .map(|(r#type, obj)| publish_unrung(&mut rig, object_message(r#type, obj)))
    .collect();
    publish_command_buffer(&rig, QUEUE, 100, 1, true);
    let done = first_queue_completion(&rig).expect("the buffer's thread took ring 0's messages");
    assert_eq!(done.cb_obj_id, 100);

    let takes = host.takes(&sent);
    assert_eq!(seqnos(&takes), sent, "{takes:?}");
    assert!(takes.iter().all(on_queue), "{takes:?}");
    let ring0 = host.ring0(&sent);
    assert_eq!(seqnos(&ring0), sent, "{ring0:?}");
    let seen = host.seen();
    let buffer = (seen.iter())
        .position(|s| s.r#type == RGFX_CMD_COMMAND_BUFFER)
        .expect("the buffer was handed over");
    let last = (seen.iter())
        .rposition(|s| s.r#type != RGFX_CMD_COMMAND_BUFFER && sent.contains(&s.seqno))
        .unwrap();
    assert!(last < buffer, "{seen:?}");
    let answers: Vec<u64> = (sent.iter())
        .map(|_| {
            let (seqno, status) = rig.complete();
            assert_eq!(status, RGFX_STATUS_OK);
            seqno
        })
        .collect();
    assert_eq!(answers, sent);
    assert_eq!(
        control_idle(&rig),
        1,
        "the buffer's thread cleared consumer_idle"
    );
}

/// A command buffer's thread takes only ring-0 messages of a context's lane: at any other, here a
/// `GET_DISPLAYS`, which the control thread answers itself, it stops and leaves it on the ring,
/// and the buffer waits; the control thread, woken by the guest's doorbell, takes it and what
/// follows, so ring 0 is taken, carried out and answered in ring order, whichever thread took each
/// message.
#[test]
fn a_buffer_s_thread_leaves_ring_0_at_a_message_of_no_context_s_lane() {
    let host = HelpHost::default();
    let mut rig = queue_rig(&host);
    host.answers.store(true, Ordering::SeqCst);
    wait_control_asleep(&rig);
    let first = publish_unrung(
        &mut rig,
        object_message(RGFX_CMD_BUFFER_AT_OFFSET, object(2)),
    );
    let displays = publish_unrung(&mut rig, get_displays());
    let last = publish_unrung(
        &mut rig,
        object_message(RGFX_CMD_BUFFER_AT_OFFSET, object(3)),
    );
    let sent = [first, displays, last];
    publish_command_buffer(&rig, QUEUE, 100, 1, true);
    assert!(
        eventually(|| host.takes(&sent).len() == 1),
        "the buffer's thread did not take the create before GET_DISPLAYS"
    );
    assert_eq!(rig.complete(), (first, RGFX_STATUS_OK));
    assert!(
        rig.quiet_for(50),
        "GET_DISPLAYS was answered with the control thread asleep"
    );
    let takes = host.takes(&sent);
    assert_eq!(seqnos(&takes), [first], "{takes:?}");
    assert!(on_queue(&takes[0]), "{takes:?}");
    assert!(
        !host
            .seen()
            .iter()
            .any(|s| s.r#type == RGFX_CMD_COMMAND_BUFFER),
        "the buffer was handed over before GET_DISPLAYS was taken"
    );
    assert_eq!(control_idle(&rig), 1);

    rig.device
        .mmio_write(RGFX_REG_DOORBELL.into(), RGFX_CONTROL_RING_ID);
    let done = first_queue_completion(&rig).expect("the buffer completed");
    assert_eq!(done.cb_obj_id, 100);
    let takes = host.takes(&sent);
    assert_eq!(seqnos(&takes), sent, "{takes:?}");
    assert_eq!(
        takes[1].thread.as_deref(),
        Some("reims-vgpu.graphics-control"),
        "{takes:?}"
    );
    let answer = rig.event().expect("GET_DISPLAYS' answer");
    let header = RgfxMsgHeader::read_from_prefix(&answer).unwrap().0;
    assert_eq!((header.r#type, header.seqno), (RGFX_EVT_DISPLAYS, displays));
    assert_eq!(rig.complete(), (last, RGFX_STATUS_OK));
    assert_eq!(seqnos(&host.ring0(&sent)), [first, last]);
}

/// With ring 0's threads holding off, as threads not yet woken, a command buffer's thread carries
/// out the buffer-at-offset creates and the destroy published before the buffer itself, in ring
/// order, each once, then hands the buffer over; the lane's thread, let go, finds nothing left.
#[test]
fn a_buffer_s_thread_carries_out_the_creates_and_destroys_it_waits_for() {
    let host = HelpHost::default();
    let mut rig = queue_rig(&host);
    let lanes = Arc::clone(&rig.device.shared.lanes);
    lanes.hold_threads(true);
    let sent: Vec<u64> = [
        (RGFX_CMD_BUFFER_AT_OFFSET, object(2)),
        (RGFX_CMD_BUFFER_AT_OFFSET, object(3)),
        (RGFX_CMD_OBJECT_DESTROY, object(2)),
        (RGFX_CMD_BUFFER_AT_OFFSET, object(4)),
    ]
    .into_iter()
    .map(|(r#type, obj)| rig.send(object_message(r#type, obj)))
    .collect();
    publish_command_buffer(&rig, QUEUE, 100, 1, true);
    let done = first_queue_completion(&rig).expect("the buffer waited for ring 0's threads");
    assert_eq!(done.cb_obj_id, 100);

    let ring0 = host.ring0(&sent);
    assert_eq!(
        ring0.iter().map(|s| s.seqno).collect::<Vec<_>>(),
        sent,
        "{ring0:?}"
    );
    assert!(ring0.iter().all(on_queue), "{ring0:?}");
    let seen = host.seen();
    let buffer = (seen.iter())
        .position(|s| s.r#type == RGFX_CMD_COMMAND_BUFFER)
        .expect("the buffer was handed over");
    let last = (seen.iter())
        .rposition(|s| s.r#type != RGFX_CMD_COMMAND_BUFFER && sent.contains(&s.seqno))
        .unwrap();
    assert!(last < buffer, "{seen:?}");

    lanes.hold_threads(false);
    assert!(eventually(|| lanes.idle()), "the lanes stayed busy");
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(host.ring0(&sent), ring0, "a job was carried out twice");
}

/// A command buffer's thread carries out only a `BUFFER_AT_OFFSET` or an `OBJECT_DESTROY` of no
/// queue: at any other message it waits for the lane's thread, which carries that out, and the
/// jobs after it are then the lane's thread's or the buffer's thread's, still in ring order.
#[test]
fn a_buffer_s_thread_leaves_other_kinds_to_the_lane_s_thread() {
    let host = HelpHost::default();
    let mut rig = queue_rig(&host);
    let lanes = Arc::clone(&rig.device.shared.lanes);
    lanes.hold_threads(true);
    let sent: Vec<u64> = [
        (RGFX_CMD_BUFFER_AT_OFFSET, object(2)),
        (RGFX_CMD_EVENT_CREATE, object(3)),
        (RGFX_CMD_BUFFER_AT_OFFSET, object(4)),
    ]
    .into_iter()
    .map(|(r#type, obj)| rig.send(object_message(r#type, obj)))
    .collect();
    publish_command_buffer(&rig, QUEUE, 100, 1, true);
    assert!(
        eventually(|| host.ring0(&sent).len() == 1),
        "the buffer's thread did not carry out the create before the event's"
    );
    std::thread::sleep(Duration::from_millis(50));
    let ring0 = host.ring0(&sent);
    assert_eq!(ring0.len(), 1, "{ring0:?}");
    assert!(on_queue(&ring0[0]), "{ring0:?}");
    assert!(
        !host
            .seen()
            .iter()
            .any(|s| s.r#type == RGFX_CMD_COMMAND_BUFFER),
        "the buffer was handed over before the event's create"
    );

    lanes.hold_threads(false);
    let done = first_queue_completion(&rig).expect("the buffer completed");
    assert_eq!(done.cb_obj_id, 100);
    let ring0 = host.ring0(&sent);
    assert_eq!(
        ring0.iter().map(|s| s.seqno).collect::<Vec<_>>(),
        sent,
        "{ring0:?}"
    );
    assert_eq!(ring0[1].thread.as_deref(), Some(RING0), "{ring0:?}");
}

/// What a command buffer's thread may carry out ([`Shared::helps`]): a `BUFFER_AT_OFFSET`, and an
/// `OBJECT_DESTROY` of an object that is no live queue; not a queue's destroy, a destroy too short
/// to name its object, nor any other kind.
#[test]
fn only_a_buffer_at_offset_and_a_destroy_of_no_queue_are_a_buffer_s_thread_s() {
    let host = HelpHost::default();
    let rig = queue_rig(&host);
    let message = |bytes: Vec<u8>| {
        let (header, _) = RgfxMsgHeader::read_from_prefix(&bytes).unwrap();
        Message { header, bytes }
    };
    let helps = |bytes: Vec<u8>| rig.device.shared.helps(&message(bytes));
    assert!(helps(object_message(RGFX_CMD_BUFFER_AT_OFFSET, object(2))));
    assert!(helps(object_message(RGFX_CMD_OBJECT_DESTROY, object(2))));
    assert!(!helps(object_message(RGFX_CMD_OBJECT_DESTROY, QUEUE)));
    assert!(!helps(
        object_message(RGFX_CMD_OBJECT_DESTROY, object(2))[..16].to_vec()
    ));
    for r#type in [
        RGFX_CMD_QUEUE_CREATE,
        RGFX_CMD_BUFFER_CREATE,
        RGFX_CMD_EVENT_CREATE,
        RGFX_CMD_BUFFER_OVER_PAGES,
        RGFX_CMD_RENDER_PIPELINE_CREATE,
        RGFX_CMD_ICB_CREATE,
    ] {
        assert!(!helps(object_message(r#type, object(2))), "{type:#x}");
    }
    assert!(!helps(context_close(5)));
    assert!(!helps(block_message(
        RGFX_CMD_BLOCK_RELEASE,
        5,
        QUEUE_SUBMIT,
        0x1000
    )));
}

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Random runs of a context's ring-0 messages, each followed by a command buffer of its queue,
/// some of the messages slow, some published with no doorbell (left to the buffer's thread or an
/// awake control thread to take), some runs with ring 0's threads holding off until the buffer is
/// handed over: every message is taken once and carried out once, in ring order, and each buffer
/// after every message published before it. A run of creates and destroys alone is carried out on
/// the buffer's thread while ring 0's threads hold off. `RGFX_HELP_FUZZ_SEED` sets the seed.
#[test]
fn buffers_wait_for_their_context_s_messages_over_random_interleavings() {
    let seed = std::env::var("RGFX_HELP_FUZZ_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x6865_6c64);
    let mut rng = Rng(seed);
    // Which messages ring no doorbell, drawn apart, so `rng`'s runs stay the seed's.
    let mut unrung = Rng(seed ^ 0x0075_7275_6e67);
    let host = HelpHost::default();
    let mut rig = queue_rig(&host);
    let lanes = Arc::clone(&rig.device.shared.lanes);
    const RUNS: u64 = 48;
    let mut at = 0;
    let mut runs: Vec<(Vec<u64>, bool)> = Vec::new();
    let mut next_obj = 2;
    for run in 0..RUNS {
        let count = rng.below(6);
        let kinds: Vec<u32> = (0..count)
            .map(|_| match rng.below(20) {
                0..10 => RGFX_CMD_BUFFER_AT_OFFSET,
                10..14 => RGFX_CMD_OBJECT_DESTROY,
                14..17 => RGFX_CMD_EVENT_CREATE,
                _ => RGFX_CMD_BUFFER_CREATE,
            })
            .collect();
        let only_helped =
            (kinds.iter()).all(|&k| k == RGFX_CMD_BUFFER_AT_OFFSET || k == RGFX_CMD_OBJECT_DESTROY);
        let held = only_helped && count > 0 && rng.below(3) == 0;
        if held {
            // Every buffer before it handed over, so the hold keeps only this run's jobs off.
            assert!(eventually(
                || queue_events(&rig, run as usize).len() >= run as usize
            ));
            lanes.hold_threads(true);
        }
        let mut sent = Vec::new();
        for r#type in kinds {
            next_obj += 1;
            let message = object_message(r#type, object(next_obj));
            let seqno = if unrung.below(3) == 0 {
                publish_unrung(&mut rig, message)
            } else {
                rig.send(message)
            };
            if rng.below(4) == 0 {
                host.slow.lock().insert(seqno, rng.below(1500));
            }
            sent.push(seqno);
        }
        at = publish_queue_message(&rig, at, run + 1, command_buffer(QUEUE, 1000 + run, 0));
        if held {
            assert!(
                eventually(|| queue_events(&rig, run as usize + 1).len() > run as usize),
                "run {run} (seed {seed:#x}): the buffer waited for ring 0's threads"
            );
            lanes.hold_threads(false);
        }
        runs.push((sent, held));
        if rng.below(3) == 0 {
            std::thread::sleep(Duration::from_micros(rng.below(2000)));
        }
    }
    let events = queue_events(&rig, RUNS as usize);
    assert_eq!(events.len(), RUNS as usize, "seed {seed:#x}: {events:?}");
    assert!(eventually(|| lanes.idle()));

    let seen = host.seen();
    let all: Vec<u64> = runs.iter().flat_map(|(sent, _)| sent.clone()).collect();
    let ring0 = host.ring0(&all);
    assert_eq!(
        ring0.iter().map(|s| s.seqno).collect::<Vec<_>>(),
        all,
        "seed {seed:#x}: ring 0's messages out of order, missed or carried out twice"
    );
    assert_eq!(
        seqnos(&host.takes(&all)),
        all,
        "seed {seed:#x}: ring 0's messages taken out of order, missed or taken twice"
    );
    let index = |r#type: u32, seqno: u64| {
        (seen.iter())
            .position(|s| s.r#type == r#type && s.seqno == seqno)
            .unwrap_or_else(|| panic!("seed {seed:#x}: {type:#x} {seqno} never handed"))
    };
    let mut before: Vec<u64> = Vec::new();
    for (run, (sent, held)) in runs.iter().enumerate() {
        before.extend(sent);
        let buffer = index(RGFX_CMD_COMMAND_BUFFER, run as u64 + 1);
        for &seqno in &before {
            let at = (seen.iter())
                .position(|s| s.r#type != RGFX_CMD_COMMAND_BUFFER && s.seqno == seqno)
                .unwrap();
            assert!(
                at < buffer,
                "seed {seed:#x}: buffer {} handed over before ring-0 message {seqno}",
                run + 1
            );
        }
        if *held {
            for &seqno in sent {
                let s = &seen[(seen.iter())
                    .position(|s| s.r#type != RGFX_CMD_COMMAND_BUFFER && s.seqno == seqno)
                    .unwrap()];
                assert!(
                    on_queue(s),
                    "seed {seed:#x}: run {run}'s {s:?} not on the buffer's thread"
                );
            }
        }
    }
}
