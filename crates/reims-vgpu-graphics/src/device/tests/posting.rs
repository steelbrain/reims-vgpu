//! Posts onto a command queue's completion ring from the host's threads, as a decoder's
//! handlers make them: what a full ring tells the host, and in what order its records reach the
//! guest.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use parking_lot::Mutex;
use zerocopy::{FromBytes, IntoBytes};

use super::{
    QUEUE_COMPLETE, QUEUE_RING_DATA, Rig, eventually, object_message, present, send_queue_create,
    vblank,
};
use crate::device::{CompletionPoster, Host, PresentAnswer, RingDegradation, Then};
use crate::wire::*;

/// The queue the tests make, in context 5.
const QUEUE: u64 = (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;

/// A host that keeps each queue's completion poster as the device tells it of the queue, and
/// records what the device told it of its completion rings, the threads it raised on, and the
/// posts it timed for the trace, which it times on the tests' platform handler thread
/// ([`returns_at_once`]) alone.
#[derive(Clone, Default)]
struct PostingHost {
    posters: Arc<Mutex<Vec<CompletionPoster>>>,
    degraded: Arc<Mutex<Vec<RingDegradation>>>,
    raised_on: Arc<Mutex<Vec<Option<String>>>>,
    timed: Arc<Mutex<Vec<(&'static str, Vec<u64>)>>>,
}

impl Host for PostingHost {
    fn raise_interrupt(&self, _vector: u32) {
        let thread = std::thread::current();
        self.raised_on.lock().push(thread.name().map(str::to_owned));
    }
    fn queue_made(&self, completion: &CompletionPoster) {
        self.posters.lock().push(completion.clone());
    }
    fn ring_degraded(&self, kind: RingDegradation) {
        self.degraded.lock().push(kind);
    }
    fn trace(&self, kind: &'static str, values: &[u64]) {
        if matches!(kind, "qraise" | "qhanded") {
            self.timed.lock().push((kind, values.to_vec()));
        }
    }
    fn times_posts_here(&self) -> bool {
        std::thread::current().name() == Some("platform-handler")
    }
}

impl PostingHost {
    /// The poster of the last queue made.
    fn poster(&self) -> CompletionPoster {
        self.posters.lock().last().cloned().expect("a queue made")
    }

    fn degraded(&self) -> Vec<RingDegradation> {
        self.degraded.lock().clone()
    }

    /// The timed posts' kinds and values ([`Host::trace`]).
    fn timed(&self) -> Vec<(&'static str, Vec<u64>)> {
        self.timed.lock().clone()
    }
}

/// A started device with [`QUEUE`] made over the tests' rings, its host `host`.
fn queue_rig(host: &PostingHost) -> Rig {
    let mut rig = Rig::with_host(Box::new(host.clone()));
    rig.start();
    let seqno = send_queue_create(&mut rig, QUEUE);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    rig
}

/// The `CB_COMPLETE` of command buffer `cb_obj_id`, 48 bytes: the tests' 64 KiB ring holds 1365.
fn record(cb_obj_id: u64) -> Vec<u8> {
    RgfxCbComplete {
        header: RgfxMsgHeader {
            r#type: RGFX_EVT_CB_COMPLETE,
            size: size_of::<RgfxCbComplete>() as u32,
            seqno: cb_obj_id,
        },
        cb_obj_id,
        gpu_start_ns: 0,
        gpu_end_ns: 0,
        status: RGFX_CB_OK,
        error_code: 0,
    }
    .as_bytes()
    .to_vec()
}

/// How many records of [`record`]'s size the tests' ring holds from empty.
const HOLDS: u64 = QUEUE_RING_DATA as u64 / size_of::<RgfxCbComplete>() as u64;

/// The tests' queue completion ring's producer count.
fn produced(rig: &Rig) -> u64 {
    rig.region()
        .atomic_u64(QUEUE_COMPLETE)
        .load(Ordering::SeqCst)
}

/// The guest takes every record on the tests' queue completion ring and rings nothing: the
/// cb ids of those it took, in ring order, past `consumed`, its count before; its count after.
fn take_all(rig: &Rig, consumed: u64) -> (Vec<u64>, u64) {
    let data = u64::from(QUEUE_RING_DATA);
    let end = produced(rig);
    let mut at = consumed;
    let mut taken = Vec::new();
    while at < end {
        let off = QUEUE_COMPLETE + 256 + at % data;
        let head = rig.region().read(off, 16).unwrap();
        let ty = u32::from_le_bytes(head[0..4].try_into().unwrap());
        let size = u64::from(u32::from_le_bytes(head[4..8].try_into().unwrap()));
        if ty == RGFX_EVT_CB_COMPLETE {
            let body = rig.region().read(off + 16, 8).unwrap();
            taken.push(u64::from_le_bytes(body[0..8].try_into().unwrap()));
        }
        at += size;
    }
    rig.region()
        .atomic_u64(QUEUE_COMPLETE + 0x40)
        .store(end, Ordering::SeqCst);
    (taken, end)
}

/// A command queue's completion ring's full spell is told to the host once, at its first post
/// that waits for room and found none, and the post that waited once it goes on; posts that find
/// room tell nothing, and the next spell is told again
#[test]
fn a_full_queue_ring_tells_its_spell_and_its_waited_posts() {
    use RingDegradation::{QueuePostWaited, QueueRingFull};
    let host = PostingHost::default();
    let rig = queue_rig(&host);
    let poster = host.poster();
    for cb in 0..HOLDS {
        poster.post(&record(cb));
    }
    assert_eq!(produced(&rig), HOLDS * 48);
    assert_eq!(host.degraded(), []);

    // One more waits for room, on a thread of its own: the spell is told, no waited post yet.
    let waiting = {
        let poster = poster.clone();
        std::thread::Builder::new()
            .name("waiting-poster".into())
            .spawn(move || poster.post(&record(HOLDS)))
            .unwrap()
    };
    assert!(
        eventually(|| !host.degraded().is_empty()),
        "the full ring was never told"
    );
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(host.degraded(), [QueueRingFull]);
    assert!(!waiting.is_finished(), "posted onto a full ring");

    // The guest takes every record: the post goes on, and is told.
    let (taken, consumed) = take_all(&rig, 0);
    assert_eq!(taken, (0..HOLDS).collect::<Vec<_>>());
    waiting.join().unwrap();
    assert_eq!(host.degraded(), [QueueRingFull, QueuePostWaited]);

    // Room: nothing is told. Full again: a spell of its own.
    poster.post(&record(HOLDS + 1));
    assert_eq!(host.degraded(), [QueueRingFull, QueuePostWaited]);
    let mut cb = HOLDS + 2;
    while poster.try_post(&record(cb)) {
        cb += 1;
    }
    assert_eq!(
        host.degraded(),
        [QueueRingFull, QueuePostWaited],
        "a post that waits for nothing told a spell"
    );
    let waiting = {
        let poster = poster.clone();
        std::thread::Builder::new()
            .name("waiting-poster".into())
            .spawn(move || poster.post(&record(cb)))
            .unwrap()
    };
    assert!(
        eventually(|| host.degraded().len() == 3),
        "the second spell was never told"
    );
    assert_eq!(host.degraded()[2], QueueRingFull);
    take_all(&rig, consumed);
    waiting.join().unwrap();
    assert_eq!(
        host.degraded(),
        [
            QueueRingFull,
            QueuePostWaited,
            QueueRingFull,
            QueuePostWaited
        ]
    );
}

/// What ran after a record ([`Then`]): its tag, the thread it ran on and the ring's producer
/// count then.
type Ran = Arc<Mutex<Vec<(u64, Option<String>, u64)>>>;

/// What follows a record: notes `tag` in `ran`, with its thread and the ring's producer count.
fn noting(rig: &Rig, ran: &Ran, tag: u64) -> Option<Then> {
    let (ran, region) = (Arc::clone(ran), Arc::clone(&rig.device.shared.region));
    Some(Box::new(move || {
        let produced = region.atomic_u64(QUEUE_COMPLETE).load(Ordering::SeqCst);
        let thread = std::thread::current();
        ran.lock()
            .push((tag, thread.name().map(str::to_owned), produced));
    }))
}

/// Runs `post` on a thread of its own, as a platform's handler thread: whether it returned
/// within two seconds.
fn returns_at_once(post: impl FnOnce() + Send + 'static) -> bool {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("platform-handler".into())
        .spawn(move || {
            post();
            let _ = tx.send(());
        })
        .unwrap();
    rx.recv_timeout(Duration::from_secs(2)).is_ok()
}

/// Whether `name` is the tests' queue's service thread's.
fn on_service_thread(name: &Option<String>) -> bool {
    name.as_deref() == Some(&*format!("reims-vgpu.graphics-queue-{QUEUE:#x}"))
}

/// A post that waits for nothing onto a full queue ring returns at once, the record handed to
/// the queue's service thread with what follows it; a post that waits goes behind them, and waits
/// for nothing either. Once the guest takes the ring, the service thread posts them in their
/// order, each followed by what follows it, then the waiting post's; it tells the ring's full
/// spell, and each of the three as a post that waited
#[test]
fn a_post_unwaiting_onto_a_full_queue_ring_returns_at_once_and_its_records_follow_in_order() {
    use RingDegradation::{QueuePostWaited, QueueRingFull};
    let host = PostingHost::default();
    let rig = queue_rig(&host);
    let poster = host.poster();
    for cb in 0..HOLDS {
        poster.post(&record(cb));
    }
    let ran = Ran::default();
    let posted = {
        let (poster, a, b) = (poster.clone(), noting(&rig, &ran, 1), noting(&rig, &ran, 2));
        returns_at_once(move || {
            poster.post_unwaiting(&record(HOLDS), a);
            poster.post_unwaiting(&record(HOLDS + 1), b);
            poster.post(&record(HOLDS + 2));
        })
    };
    assert!(posted, "a post behind records handed over waited for room");
    assert_eq!(rig.device.handed_over(), 2);
    assert_eq!(produced(&rig), HOLDS * 48, "posted onto a full ring");
    assert!(ran.lock().is_empty(), "ran before its record was posted");
    assert!(
        eventually(|| host.degraded() == [QueueRingFull]),
        "the spell was not told: {:?}",
        host.degraded()
    );

    let (taken, consumed) = take_all(&rig, 0);
    assert_eq!(taken, (0..HOLDS).collect::<Vec<_>>());
    assert!(
        eventually(|| produced(&rig) >= consumed + 16 + 3 * 48),
        "the handed records were never posted"
    );
    let (taken, _) = take_all(&rig, consumed);
    assert_eq!(taken, [HOLDS, HOLDS + 1, HOLDS + 2]);
    // Each ran on the service thread right after its own record: the first with its record on
    // the ring and the second's not yet, past the 16 bytes of padding at the ring's end.
    let ran = ran.lock().clone();
    assert_eq!(ran.iter().map(|r| r.0).collect::<Vec<_>>(), [1, 2]);
    assert!(
        ran.iter().all(|(_, thread, _)| on_service_thread(thread)),
        "{ran:?}"
    );
    assert_eq!(ran[0].2, consumed + 16 + 48);
    assert_eq!(ran[1].2, consumed + 16 + 2 * 48);
    let told = [
        QueueRingFull,
        QueuePostWaited,
        QueuePostWaited,
        QueuePostWaited,
    ];
    assert!(
        eventually(|| host.degraded() == told),
        "{:?}",
        host.degraded()
    );
}

/// A post that waits for nothing, finding records handed over and not yet posted, goes behind
/// them though the ring has room by then, posting them first, and a tried post goes nowhere:
/// here the guest takes the ring while the service thread, waiting for room, has not looked again
#[test]
fn a_post_unwaiting_goes_behind_records_handed_over_though_the_ring_has_room() {
    let host = PostingHost::default();
    let rig = queue_rig(&host);
    rig.device
        .shared
        .room_poll_ns
        .store(60_000_000_000, Ordering::Relaxed);
    let poster = host.poster();
    for cb in 0..HOLDS {
        poster.post(&record(cb));
    }
    poster.post_unwaiting(&record(HOLDS), None);
    // The service thread found no room for the record and looks again only after its poll; the
    // guest makes room, ringing nothing.
    std::thread::sleep(Duration::from_millis(50));
    let (taken, consumed) = take_all(&rig, 0);
    assert_eq!(taken.len() as u64, HOLDS);
    assert!(!poster.try_post(&record(99)), "a tried post went ahead");
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(
        produced(&rig),
        consumed,
        "a post went ahead of a record handed over"
    );

    poster.post_unwaiting(&record(HOLDS + 1), None);
    assert_eq!(produced(&rig), consumed + 16 + 2 * 48);
    let (taken, _) = take_all(&rig, consumed);
    assert_eq!(taken, [HOLDS, HOLDS + 1]);
    assert_eq!(rig.device.handed_over(), 2);
}

/// On a thread where the host times posts, a raise of a queue's vector is traced as `qraise`, with
/// the queue, and a post of the records handed over on its ring as `qhanded`, with the queue and
/// how many it posted, after the raises among them; on another thread neither is
#[test]
fn a_host_that_times_posts_on_a_thread_is_told_of_the_raises_and_handed_posts_there() {
    let host = PostingHost::default();
    let rig = queue_rig(&host);
    rig.device
        .shared
        .room_poll_ns
        .store(60_000_000_000, Ordering::Relaxed);
    let arm = || {
        rig.region()
            .atomic_u32(QUEUE_COMPLETE + 0xc0)
            .store(1, Ordering::SeqCst);
    };
    let poster = host.poster();
    let raised = host.raised_on.lock().len();
    arm();
    let posted = {
        let poster = poster.clone();
        returns_at_once(move || poster.post_unwaiting(&record(0), None))
    };
    assert!(posted);
    arm();
    poster.post(&record(1));
    assert_eq!(host.raised_on.lock().len(), raised + 2);
    let timed = host.timed();
    assert_eq!(timed.len(), 1, "{timed:?}");
    assert_eq!((timed[0].0, timed[0].1[0]), ("qraise", QUEUE));

    // A record handed over behind a full ring: the service thread finds no room and looks again
    // only after its poll, and the guest makes room, ringing nothing.
    for cb in 2..HOLDS {
        poster.post(&record(cb));
    }
    poster.post_unwaiting(&record(HOLDS), None);
    std::thread::sleep(Duration::from_millis(50));
    let (taken, consumed) = take_all(&rig, 0);
    assert_eq!(taken.len() as u64, HOLDS);
    arm();
    let queue = rig.device.shared.queues.lock().by_obj[&QUEUE].clone();
    let shared = Arc::clone(&rig.device.shared);
    assert!(returns_at_once(move || assert!(shared.post_handed(&queue))));
    assert_eq!(produced(&rig), consumed + 16 + 48);
    let timed = host.timed();
    let kinds: Vec<_> = timed
        .iter()
        .map(|(kind, values)| (*kind, values[0]))
        .collect();
    assert_eq!(
        kinds,
        [("qraise", QUEUE), ("qraise", QUEUE), ("qhanded", QUEUE)]
    );
    assert_eq!(timed[2].1[1], 1, "the records it posted");
    assert!(timed[2].1[2] >= timed[1].1[1], "its time holds its raise's");
}

/// The records handed over to a queue's service thread when the device is reset are dropped,
/// none posted, and what follows each is run there; a post through the poster after the reset
/// posts nothing and runs what follows it at once
#[test]
fn a_reset_drops_the_records_handed_over_and_runs_what_follows_them() {
    let host = PostingHost::default();
    let rig = queue_rig(&host);
    let poster = host.poster();
    for cb in 0..HOLDS {
        poster.post(&record(cb));
    }
    let ran = Ran::default();
    poster.post_unwaiting(&record(HOLDS), noting(&rig, &ran, 1));
    poster.post_unwaiting(&record(HOLDS + 1), noting(&rig, &ran, 2));
    std::thread::sleep(Duration::from_millis(20));
    assert!(ran.lock().is_empty());

    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    assert!(eventually(|| ran.lock().len() == 2), "{:?}", ran.lock());
    poster.post_unwaiting(&record(HOLDS + 2), noting(&rig, &ran, 3));
    let ran = ran.lock().clone();
    assert_eq!(ran.iter().map(|r| r.0).collect::<Vec<_>>(), [1, 2, 3]);
    assert!(ran.iter().all(|&(_, _, produced)| produced == HOLDS * 48));
    assert_eq!(produced(&rig), HOLDS * 48, "posted after the reset");
}

/// The records handed over to a queue's service thread when the queue is destroyed are dropped
/// and what follows each is run, in order; a post after the destroy runs what follows it at once
#[test]
fn a_queue_s_destroy_drops_its_handed_records_and_runs_what_follows_them() {
    let host = PostingHost::default();
    let mut rig = queue_rig(&host);
    let poster = host.poster();
    for cb in 0..HOLDS {
        poster.post(&record(cb));
    }
    let ran = Ran::default();
    poster.post_unwaiting(&record(HOLDS), noting(&rig, &ran, 1));
    poster.post_unwaiting(&record(HOLDS + 1), noting(&rig, &ran, 2));

    rig.send(object_message(RGFX_CMD_OBJECT_DESTROY, QUEUE));
    assert!(eventually(|| ran.lock().len() == 2), "{:?}", ran.lock());
    assert!(eventually(|| rig.device.queue_count() == 0));
    poster.post_unwaiting(&record(HOLDS + 2), noting(&rig, &ran, 3));
    let ran = ran.lock().clone();
    assert_eq!(ran.iter().map(|r| r.0).collect::<Vec<_>>(), [1, 2, 3]);
    assert_eq!(produced(&rig), HOLDS * 48, "posted after the destroy");
}

/// A host that keeps each present's answer for the test to give, and records what the device
/// told it of its completion rings.
#[derive(Clone, Default)]
struct AnsweringHost {
    answers: Arc<Mutex<Vec<PresentAnswer>>>,
    degraded: Arc<Mutex<Vec<RingDegradation>>>,
}

impl Host for AnsweringHost {
    fn raise_interrupt(&self, _vector: u32) {}
    fn present_surface(
        &self,
        _display: u32,
        _iosurface_id: u32,
        _generation: Option<u64>,
        answer: PresentAnswer,
    ) {
        self.answers.lock().push(answer);
    }
    fn ring_degraded(&self, kind: RingDegradation) {
        self.degraded.lock().push(kind);
    }
}

/// An answer given off the control thread onto a full control ring returns at once: handed to the
/// control thread, it goes behind the answer the control thread waits to post and ahead of that
/// thread's next, and tells nothing itself. A `VBLANK` that finds it handed over is dropped though
/// the guest made room meanwhile, ringing nothing, and a tried post goes nowhere; once the guest
/// rings, every answer comes in that order
#[test]
fn an_answer_given_onto_a_full_control_ring_waits_for_nothing_and_keeps_its_place() {
    use RingDegradation::{ControlRingFull, VblankDropped};
    let host = AnsweringHost::default();
    let mut rig = Rig::with_host(Box::new(host.clone()));
    rig.start();
    let shown = rig.send(present(0));
    assert!(eventually(|| host.answers.lock().len() == 1), "no present");
    rig.device
        .shared
        .room_poll_ns
        .store(60_000_000_000, Ordering::Relaxed);
    // 24-byte answers: the 32 KiB ring holds 1365 of them, and the control thread waits for room
    // for the next.
    let count = 1400;
    for _ in 0..count {
        rig.send(vblank(1));
    }
    assert!(
        eventually(|| *host.degraded.lock() == [ControlRingFull]),
        "the ring never filled"
    );
    let answer = host.answers.lock().pop().unwrap();
    assert!(
        returns_at_once(move || answer.answer(RGFX_STATUS_OK)),
        "an answer waited for room"
    );

    // The guest takes the ring, ringing nothing: the control thread looks again only after its
    // long poll.
    let mut seqnos = Vec::new();
    while let Some(bytes) = rig.event_now() {
        seqnos.push(complete_seqno(&bytes));
    }
    assert!(
        !seqnos.contains(&shown),
        "the answer went ahead of the post waiting"
    );
    rig.refresh();
    assert_eq!(*host.degraded.lock(), [ControlRingFull, VblankDropped]);
    assert!(
        !(rig.device.shared)
            .control_poster(rig.device.shared.epoch.load(Ordering::Acquire))
            .try_post(&crate::control::complete(1, 0)),
        "a tried post went ahead of the answer handed over"
    );
    assert!(
        rig.event_now().is_none(),
        "posted ahead of the answer handed over"
    );

    // The guest rings: the answer the control thread waited to post, then the one handed over,
    // then the rest of its answers.
    rig.device
        .mmio_write(RGFX_REG_DOORBELL.into(), RGFX_CONTROL_RING_ID);
    while seqnos.len() < count as usize + 1 {
        seqnos.push(complete_seqno(&rig.event().expect("an answer")));
    }
    let at = seqnos.iter().position(|&s| s == shown).unwrap();
    let waited = seqnos[at - 1];
    let mut expected: Vec<u64> = (shown + 1..=waited).collect();
    expected.push(shown);
    expected.extend(waited + 1..=shown + count);
    assert_eq!(seqnos, expected);
}

/// The seqno of the `COMPLETE` in `bytes`.
fn complete_seqno(bytes: &[u8]) -> u64 {
    let (header, _) = RgfxMsgHeader::read_from_prefix(bytes).unwrap();
    assert_eq!(header.r#type, RGFX_EVT_COMPLETE);
    header.seqno
}
