//! Posts onto a command queue's completion ring from the host's threads, as a decoder's
//! handlers make them: what a full ring tells the host, and in what order its records reach the
//! guest.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use parking_lot::Mutex;
use zerocopy::IntoBytes;

use super::{QUEUE_COMPLETE, QUEUE_RING_DATA, Rig, eventually, send_queue_create};
use crate::device::{CompletionPoster, Host, RingDegradation};
use crate::wire::*;

/// The queue the tests make, in context 5.
const QUEUE: u64 = (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;

/// A host that keeps each queue's completion poster as the device tells it of the queue, and
/// records what the device told it of its completion rings and the threads it raised on.
#[derive(Clone, Default)]
struct PostingHost {
    posters: Arc<Mutex<Vec<CompletionPoster>>>,
    degraded: Arc<Mutex<Vec<RingDegradation>>>,
    raised_on: Arc<Mutex<Vec<Option<String>>>>,
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
}

impl PostingHost {
    /// The poster of the last queue made.
    fn poster(&self) -> CompletionPoster {
        self.posters.lock().last().cloned().expect("a queue made")
    }

    fn degraded(&self) -> Vec<RingDegradation> {
        self.degraded.lock().clone()
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
        std::thread::spawn(move || poster.post(&record(HOLDS)))
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
        std::thread::spawn(move || poster.post(&record(cb)))
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
