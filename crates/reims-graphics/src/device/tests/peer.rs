//! The wire round trip with a fake peer: [`FakeGuest`] speaks the wire to the real [`Device`] as
//! the guest's kext and bundle do, against a host that answers as a decoder would, with no GPU.
//! It records what crosses the wire as lines of a trace:
//! - every register read and write, doorbells among them;
//! - every record the guest publishes, and every record it takes from a completion ring;
//! - each arm of a completion ring and its re-check;
//! - each interrupt the guest takes;
//! - each ring header where the traffic has settled, every nonzero word of it.
//!
//! A script's trace is kept in a file beside this one (`peer_wire2.trace` for the wire-2
//! guest's). The test fails where the device's traffic differs from it. Where a change means
//! to alter that traffic, the file is recorded again from its own build
//! (`RVG_PEER_RECORD=1 cargo nextest run -p reims-graphics -E 'test(/peer/)'`), and its diff is
//! reviewed with the change.
//!
//! The script fixes every choice the wire leaves to timing:
//! - the guest arms a completion ring before it publishes what that ring answers, and waits for
//!   the interrupt rather than polling, so each answer raises exactly once;
//! - it publishes on a submission ring only once the host has taken everything on it and sleeps,
//!   so each post rings the doorbell;
//! - it takes a header snapshot only where every submission ring is idle and every answer it
//!   waits for is taken.

use std::collections::VecDeque;

use parking_lot::Condvar;

use super::*;
use crate::ops::{RGFX_COMMAND_BUFFER_OPS_OFFSET, RgfxCommandBuffer};

/// The feature bits the peer host offers: the real host's.
const PEER_FEATURES: u64 =
    (RGFX_FEATURE_SURFACE_PAGES | RGFX_FEATURE_SURFACE_CACHE | RGFX_FEATURE_CB_SUBMITTED) as u64;

/// How many 1 ms looks the guest takes for what the host owes it: about 2 s.
const PATIENCE_MS: u32 = 2000;

/// The interrupts the host raised and the guest has not yet taken, in order.
#[derive(Default)]
struct Raised {
    vectors: Mutex<VecDeque<u32>>,
    moved: Condvar,
}

/// A host that answers each `COMMAND_BUFFER` with its `CB_COMPLETE` (`RGFX_CB_OK`, GPU times 0)
/// on the queue's completion ring, as [`QueueTestHost`] does, and keeps every interrupt for the
/// guest to take.
struct PeerHost {
    raised: Arc<Raised>,
}

impl Host for PeerHost {
    fn raise_interrupt(&self, vector: u32) {
        self.raised.vectors.lock().push_back(vector);
        self.raised.moved.notify_all();
    }
    fn scanout_written(&self, _offset: u64, _length: u64) -> bool {
        false
    }
    fn features(&self) -> u64 {
        PEER_FEATURES
    }
    fn stage3_message(&self, completion: &CompletionPoster, message: &[u8]) {
        let (header, _) = RgfxMsgHeader::read_from_prefix(message).unwrap();
        if header.r#type != RGFX_CMD_COMMAND_BUFFER {
            return;
        }
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

/// A register's name in the trace.
fn register_name(offset: u32) -> String {
    match offset {
        RGFX_REG_MAGIC => "MAGIC".into(),
        RGFX_REG_VERSION => "VERSION".into(),
        RGFX_REG_FEATURES_LO => "FEATURES_LO".into(),
        RGFX_REG_FEATURES_HI => "FEATURES_HI".into(),
        RGFX_REG_STATUS => "STATUS".into(),
        RGFX_REG_REGION_SIZE_LO => "REGION_SIZE_LO".into(),
        RGFX_REG_REGION_SIZE_HI => "REGION_SIZE_HI".into(),
        RGFX_REG_DOORBELL => "DOORBELL".into(),
        other => format!("{other:#05x}"),
    }
}

/// A message type's name in the trace, for the types the script meets.
fn type_name(r#type: u32) -> String {
    match r#type {
        RGFX_MSG_PAD => "PAD".into(),
        RGFX_CMD_GET_DISPLAYS => "GET_DISPLAYS".into(),
        RGFX_CMD_QUEUE_CREATE => "QUEUE_CREATE".into(),
        RGFX_CMD_OBJECT_DESTROY => "OBJECT_DESTROY".into(),
        RGFX_CMD_CONTEXT_CLOSE => "CONTEXT_CLOSE".into(),
        RGFX_CMD_COMMAND_BUFFER => "COMMAND_BUFFER".into(),
        RGFX_EVT_COMPLETE => "COMPLETE".into(),
        RGFX_EVT_DISPLAYS => "DISPLAYS".into(),
        RGFX_EVT_CB_COMPLETE => "CB_COMPLETE".into(),
        other => format!("{other:#x}"),
    }
}

/// The 8-byte words of `bytes` past the 16-byte header, little-endian, in hex.
fn body_words(bytes: &[u8]) -> String {
    let words: Vec<String> = bytes[16.min(bytes.len())..]
        .chunks(8)
        .map(|w| {
            let mut word = [0u8; 8];
            word[..w.len()].copy_from_slice(w);
            format!("{:#x}", u64::from_le_bytes(word))
        })
        .collect();
    words.join(" ")
}

/// One ring pair as the guest keeps it: a submission ring it produces into and a completion ring
/// it consumes from, with its counts.
struct GuestRing {
    /// The pair's name in the trace.
    name: &'static str,
    submit: u64,
    complete: u64,
    submit_data: u64,
    complete_data: u64,
    /// The id its doorbell writes.
    ring_id: u32,
    /// The MSI-X vector its completion ring raises.
    vector: u32,
    /// The guest's count of the submission ring.
    produced: u64,
    /// The guest's count of the completion ring.
    consumed: u64,
    /// The last seqno it stamped.
    seqno: u64,
}

impl GuestRing {
    /// Ring 0, the control ring pair.
    fn control() -> GuestRing {
        GuestRing {
            name: "ring0",
            submit: RGFX_CONTROL_SUBMIT_OFFSET,
            complete: RGFX_CONTROL_COMPLETE_OFFSET,
            submit_data: RGFX_CONTROL_RING_DATA_SIZE.into(),
            complete_data: RGFX_CONTROL_RING_DATA_SIZE.into(),
            ring_id: RGFX_CONTROL_RING_ID,
            vector: 0,
            produced: 0,
            consumed: 0,
            seqno: 0,
        }
    }

    /// The tests' queue ring pair ([`QUEUE_SUBMIT`], [`QUEUE_COMPLETE`]), raising vector 3.
    fn queue(name: &'static str) -> GuestRing {
        GuestRing {
            name,
            submit: QUEUE_SUBMIT,
            complete: QUEUE_COMPLETE,
            submit_data: QUEUE_RING_DATA.into(),
            complete_data: QUEUE_RING_DATA.into(),
            ring_id: QUEUE_SUBMIT as u32,
            vector: 3,
            produced: 0,
            consumed: 0,
            seqno: 0,
        }
    }
}

/// The guest: the device it drives, what it has seen, and its rings.
struct FakeGuest {
    rig: Rig,
    raised: Arc<Raised>,
    trace: Vec<String>,
    control: GuestRing,
}

impl FakeGuest {
    fn new() -> FakeGuest {
        let raised = Arc::new(Raised::default());
        let rig = Rig::with_host(Box::new(PeerHost {
            raised: Arc::clone(&raised),
        }));
        FakeGuest {
            rig,
            raised,
            trace: Vec::new(),
            control: GuestRing::control(),
        }
    }

    fn region(&self) -> &Region {
        self.rig.region()
    }

    fn note(&mut self, line: String) {
        self.trace.push(line.trim_end().to_owned());
    }

    /// A register read, as the kext makes it.
    fn read(&mut self, offset: u32) -> u32 {
        let value = self.rig.device.mmio_read(offset.into());
        self.note(format!("read {} = {value:#x}", register_name(offset)));
        value
    }

    /// A register write, as the vCPU makes it (QEMU's write path, which offers the host a
    /// doorbell's command buffer on the vCPU's thread).
    fn write(&mut self, offset: u32, value: u32) {
        self.note(format!("write {} = {value:#x}", register_name(offset)));
        self.rig.device.mmio_write_taking(offset.into(), value);
    }

    /// Waits about [`PATIENCE_MS`] ms for `done`, panicking with `what` where it never holds.
    fn wait_for(&self, what: &str, done: impl Fn(&FakeGuest) -> bool) {
        for _ in 0..PATIENCE_MS {
            if done(self) {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(done(self), "{what}");
    }

    /// Whether the host has taken everything on `ring`'s submission ring and its consumer
    /// sleeps.
    fn submit_idle(&self, ring: &GuestRing) -> bool {
        let region = self.region();
        region.atomic_u64(ring.submit + 0x40).load(Ordering::SeqCst) == ring.produced
            && region.atomic_u32(ring.submit + 0x80).load(Ordering::SeqCst) == 1
    }

    /// The kext's start: the device checked, reset, the rings' counts taken as they stand, then
    /// the driver ready.
    fn start(&mut self) {
        assert_eq!(self.read(RGFX_REG_MAGIC), RGFX_MAGIC);
        assert_eq!(self.read(RGFX_REG_VERSION), RGFX_VERSION);
        let size = (u64::from(self.read(RGFX_REG_REGION_SIZE_HI)) << 32)
            | u64::from(self.read(RGFX_REG_REGION_SIZE_LO));
        assert_eq!(size, REGION_SIZE);
        self.read(RGFX_REG_FEATURES_LO);
        self.read(RGFX_REG_FEATURES_HI);
        self.write(RGFX_REG_STATUS, RGFX_STATUS_RESET);
        let region = self.rig.region();
        self.control.produced = region
            .atomic_u64(RGFX_CONTROL_SUBMIT_OFFSET)
            .load(Ordering::SeqCst);
        self.control.consumed = region
            .atomic_u64(RGFX_CONTROL_COMPLETE_OFFSET + 0x40)
            .load(Ordering::SeqCst);
        self.write(RGFX_REG_STATUS, RGFX_STATUS_DRIVER_READY);
        self.read(RGFX_REG_STATUS);
    }

    /// Publishes `messages` on `ring`'s submission ring with one producer store, each stamped
    /// with the ring's next seqno and its size, then rings the doorbell where the host's consumer
    /// sleeps (`consumer_idle` 1). Waits first for the host to take what came before and sleep,
    /// so the doorbell is rung. The messages' seqnos.
    fn post(&mut self, ring: &mut GuestRing, messages: Vec<Vec<u8>>) -> Vec<u64> {
        self.wait_for("the host never took the ring", |g| g.submit_idle(ring));
        let data = ring.submit_data;
        let mut seqnos = Vec::new();
        for mut message in messages {
            ring.seqno += 1;
            message[8..16].copy_from_slice(&ring.seqno.to_le_bytes());
            set_size_of(&mut message);
            let len = message.len() as u64;
            let left = data - ring.produced % data;
            assert!(
                len + if left < len { left } else { 0 } <= data,
                "the script fills the ring"
            );
            if left < len {
                let mut pad = [0u8; 8];
                pad[4..8].copy_from_slice(&(left as u32).to_le_bytes());
                self.region()
                    .write(ring.submit + 256 + ring.produced % data, &pad);
                ring.produced += left;
                self.note(format!("post {} PAD size={left}", ring.name));
            }
            self.region()
                .write(ring.submit + 256 + ring.produced % data, &message);
            ring.produced += len;
            self.note(format!(
                "post {} {} seqno={} size={len} {}",
                ring.name,
                type_name(u32::from_le_bytes(message[0..4].try_into().unwrap())),
                ring.seqno,
                body_words(&message)
            ));
            seqnos.push(ring.seqno);
        }
        self.region()
            .atomic_u64(ring.submit)
            .store(ring.produced, Ordering::SeqCst);
        std::sync::atomic::fence(Ordering::SeqCst);
        let idle = self
            .region()
            .atomic_u32(ring.submit + 0x80)
            .load(Ordering::SeqCst);
        if idle == 1 {
            self.write(RGFX_REG_DOORBELL, ring.ring_id);
        }
        seqnos
    }

    /// Sets `ring`'s completion ring's `armed`, fences, then reads its `producer`: whether a
    /// record came that the guest has not taken (the bundle's `armAndCheck`).
    fn arm_and_check(&mut self, ring: &GuestRing) -> bool {
        self.region()
            .atomic_u32(ring.complete + 0xc0)
            .store(1, Ordering::SeqCst);
        std::sync::atomic::fence(Ordering::SeqCst);
        let produced = self
            .region()
            .atomic_u64(ring.complete)
            .load(Ordering::SeqCst);
        let more = produced != ring.consumed;
        self.note(format!(
            "arm {} recheck {}",
            ring.name,
            if more { "more" } else { "empty" }
        ));
        more
    }

    /// Waits for the host to raise `ring`'s vector, and takes that interrupt: the oldest one
    /// raised, which must be `ring`'s.
    fn take_interrupt(&mut self, ring: &GuestRing) {
        let vector = {
            let mut vectors = self.raised.vectors.lock();
            for _ in 0..PATIENCE_MS {
                if !vectors.is_empty() {
                    break;
                }
                self.raised
                    .moved
                    .wait_for(&mut vectors, Duration::from_millis(1));
            }
            let vector = vectors.pop_front();
            assert!(vector.is_some(), "no interrupt for {}", ring.name);
            vector.unwrap_or_default()
        };
        self.note(format!("interrupt {vector}"));
        assert_eq!(vector, ring.vector, "the interrupt of another ring");
    }

    /// Takes `count` records from `ring`'s completion ring, waiting for each, storing the
    /// consumer count after each, passing over pads.
    fn take(&mut self, ring: &mut GuestRing, count: usize) -> Vec<Vec<u8>> {
        let data = ring.complete_data;
        let mut taken = Vec::new();
        while taken.len() < count {
            self.wait_for("the host never posted the answer", |g| {
                g.region().atomic_u64(ring.complete).load(Ordering::SeqCst) != ring.consumed
            });
            let at = ring.complete + 256 + ring.consumed % data;
            let head = self.region().read(at, 8).unwrap();
            let r#type = u32::from_le_bytes(head[0..4].try_into().unwrap());
            let size = u32::from_le_bytes(head[4..8].try_into().unwrap()) as usize;
            let bytes = self.region().read(at, size).unwrap();
            ring.consumed += size as u64;
            self.region()
                .atomic_u64(ring.complete + 0x40)
                .store(ring.consumed, Ordering::SeqCst);
            if r#type == RGFX_MSG_PAD {
                self.note(format!("take {} PAD size={size}", ring.name));
                continue;
            }
            let seqno = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
            self.note(format!(
                "take {} {} seqno={seqno} size={size} {}",
                ring.name,
                type_name(r#type),
                body_words(&bytes)
            ));
            taken.push(bytes);
        }
        taken
    }

    /// What the bundle's completion thread and the kext's service do once woken: take the
    /// interrupt, take `count` records, then arm and look again until nothing more came.
    fn answers(&mut self, ring: &mut GuestRing, count: usize) -> Vec<Vec<u8>> {
        self.take_interrupt(ring);
        let mut taken = self.take(ring, count);
        while self.arm_and_check(ring) {
            taken.extend(self.take(ring, 1));
        }
        taken
    }

    /// A ring-0 command and its one answer.
    fn command(&mut self, message: Vec<u8>) -> Vec<u8> {
        let mut control = std::mem::replace(&mut self.control, GuestRing::control());
        self.post(&mut control, vec![message]);
        let mut answer = self.answers(&mut control, 1);
        self.control = control;
        answer.pop().unwrap()
    }

    /// A ring-0 command the host does not answer.
    fn command_unanswered(&mut self, message: Vec<u8>) {
        let mut control = std::mem::replace(&mut self.control, GuestRing::control());
        self.post(&mut control, vec![message]);
        self.control = control;
    }

    /// The headers of the four rings once the traffic settled: ring 0's submission ring and
    /// each of `live`'s taken and idle, and no interrupt the guest has not taken. Each header's
    /// nonzero 8-byte words, by offset.
    fn settle(&mut self, live: &[&GuestRing]) {
        self.wait_for("ring 0 never went idle", |g| g.submit_idle(&g.control));
        for ring in live {
            self.wait_for("a submission ring never went idle", |g| g.submit_idle(ring));
        }
        let unclaimed: Vec<u32> = self.raised.vectors.lock().drain(..).collect();
        if !unclaimed.is_empty() {
            self.note(format!("unclaimed interrupts {unclaimed:?}"));
        }
        for (name, at) in [
            ("ring0.submit", RGFX_CONTROL_SUBMIT_OFFSET),
            ("ring0.complete", RGFX_CONTROL_COMPLETE_OFFSET),
            ("queue.submit", QUEUE_SUBMIT),
            ("queue.complete", QUEUE_COMPLETE),
        ] {
            let header = self.region().read(at, 256).unwrap();
            let words: Vec<String> = header
                .chunks(8)
                .enumerate()
                .filter_map(|(i, w)| {
                    let word = u64::from_le_bytes(w.try_into().unwrap());
                    (word != 0).then(|| format!("+{:#04x}={word:#x}", i * 8))
                })
                .collect();
            self.note(format!("header {name} {}", words.join(" ")));
        }
    }

    /// Makes the tests' queue rings new (zeroed headers, as the bundle hands them over) and
    /// creates queue `obj_id` over them on ring 0; arms its completion ring, as the bundle's
    /// completion thread does when it takes the ring.
    fn create_queue(&mut self, obj_id: u64, name: &'static str) -> GuestRing {
        for at in [QUEUE_SUBMIT, QUEUE_COMPLETE] {
            self.region().write(at, &[0u8; 256]);
        }
        let answer = self.command(queue_create(obj_id));
        let complete = RgfxComplete::read_from_bytes(&answer).unwrap();
        assert_eq!(complete.status, RGFX_STATUS_OK);
        let queue = GuestRing::queue(name);
        assert!(!self.arm_and_check(&queue));
        queue
    }
}

/// A `COMMAND_BUFFER` of `cb_obj_id` on queue `queue`, with no ops.
fn empty_command_buffer(queue: u64, cb_obj_id: u64) -> Vec<u8> {
    RgfxCommandBuffer {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_COMMAND_BUFFER,
            size: RGFX_COMMAND_BUFFER_OPS_OFFSET,
            seqno: 0,
        },
        queue_obj_id: queue,
        cb_obj_id,
        op_bytes: 0,
    }
    .as_bytes()
    .to_vec()
}

/// The wire-2 guest's script: what today's kext and bundle do, with every feature register
/// left unwritten.
/// - start, then `GET_DISPLAYS` and a command of no known type;
/// - a queue, a buffer on it alone, then two in one post and one doorbell, which raise once;
/// - the queue's `OBJECT_DESTROY` (no answer), seen through by the next ring-0 answer;
/// - a second queue over the same rings, a buffer on it, then its context's `CONTEXT_CLOSE`;
/// - a reset.
fn wire2_script(guest: &mut FakeGuest) {
    const CONTEXT: u32 = 7;
    const QUEUE_A: u64 = ((CONTEXT as u64) << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    const QUEUE_B: u64 = ((CONTEXT as u64) << RGFX_CONTEXT_OBJ_ID_SHIFT) | 2;

    guest.start();
    guest.settle(&[]);
    // The control ring is armed before its first command, as the kext's service leaves it.
    let control = std::mem::replace(&mut guest.control, GuestRing::control());
    assert!(!guest.arm_and_check(&control));
    guest.control = control;

    guest.command(get_displays());
    let unknown = guest.command(object_message(0x7777, 0));
    let unknown = RgfxComplete::read_from_bytes(&unknown).unwrap();
    assert_ne!(unknown.status, RGFX_STATUS_OK);
    guest.settle(&[]);

    let mut a = guest.create_queue(QUEUE_A, "queueA");
    guest.settle(&[&a]);
    guest.post(&mut a, vec![empty_command_buffer(QUEUE_A, 0x100)]);
    assert_eq!(guest.answers(&mut a, 1).len(), 1);
    guest.post(
        &mut a,
        vec![
            empty_command_buffer(QUEUE_A, 0x101),
            empty_command_buffer(QUEUE_A, 0x102),
        ],
    );
    assert_eq!(guest.answers(&mut a, 2).len(), 2);
    guest.settle(&[&a]);

    guest.command_unanswered(object_message(RGFX_CMD_OBJECT_DESTROY, QUEUE_A));
    guest.command(get_displays());
    assert_eq!(guest.rig.device.queue_count(), 0);
    guest.settle(&[]);

    let mut b = guest.create_queue(QUEUE_B, "queueB");
    guest.settle(&[&b]);
    guest.post(&mut b, vec![empty_command_buffer(QUEUE_B, 0x200)]);
    assert_eq!(guest.answers(&mut b, 1).len(), 1);
    guest.settle(&[&b]);
    let closed = guest.command(context_close(CONTEXT));
    assert_eq!(
        RgfxComplete::read_from_bytes(&closed).unwrap().status,
        RGFX_STATUS_OK
    );
    assert_eq!(guest.rig.device.queue_count(), 0);
    guest.settle(&[]);

    guest.write(RGFX_REG_STATUS, RGFX_STATUS_RESET);
    guest.read(RGFX_REG_STATUS);
}

/// The trace of `script` against a new device, as text: one line a step.
fn trace_of(script: fn(&mut FakeGuest)) -> String {
    let mut guest = FakeGuest::new();
    script(&mut guest);
    let mut text = guest.trace.join("\n");
    text.push('\n');
    text
}

/// `trace` against the recorded file `name` beside this one; with `RVG_PEER_RECORD=1`, writes
/// it there instead.
fn check_recorded(name: &str, recorded: &str, trace: &str) {
    if std::env::var_os("RVG_PEER_RECORD").is_some_and(|v| v == "1") {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/device/tests")
            .join(name);
        std::fs::write(&path, trace).unwrap();
        return;
    }
    if recorded == trace {
        return;
    }
    let first = recorded
        .lines()
        .zip(trace.lines())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| recorded.lines().count().min(trace.lines().count()));
    panic!(
        "the wire traffic differs from {name} at line {}:\n  recorded: {:?}\n  now:      {:?}\n\
         (RVG_PEER_RECORD=1 records it again from this build)",
        first + 1,
        recorded.lines().nth(first),
        trace.lines().nth(first),
    );
}

/// A wire-2 guest's traffic with the device is the recorded one: the registers it reads, what
/// the host answers, the interrupts, and the rings' headers where the traffic settled.
#[test]
fn peer_wire2_traffic_is_the_recorded_one() {
    check_recorded(
        "peer_wire2.trace",
        include_str!("peer_wire2.trace"),
        &trace_of(wire2_script),
    );
}

/// The script's traffic is the same from run to run: the trace holds nothing that depends on
/// timing.
#[test]
fn peer_wire2_traffic_is_the_same_each_run() {
    let first = trace_of(wire2_script);
    for _ in 0..4 {
        assert_eq!(trace_of(wire2_script), first);
    }
}
