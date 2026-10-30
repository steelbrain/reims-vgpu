//! [`Device`]: the device's host side, as the host's process runs it. It owns the region (BAR 2),
//! answers the registers (BAR 0), takes the guest's control commands on a thread of its own, and
//! sends the display's events on another.
//!
//! **Threads.** `reims-vgpu.graphics-control` takes the control submission ring: woken by a
//! doorbell (the host's `ioeventfd`, or a `DOORBELL` write through [`Device::mmio_write`]) or by
//! the guest's `STATUS`, it takes every message the guest has published, answers each on the
//! completion ring, then sleeps with the ring's `consumer_idle` handshake (`ring`). While the
//! completion ring has no room, it waits for the guest to take more, polling, and takes no
//! command meanwhile. The framebuffer kext is its sole producer (the display commands).
//! `reims-vgpu.graphics-device` takes the device submission ring the same way, its sole producer
//! the Metal bundle: it hands every message (the object lifecycle — queue, heap, buffer, texture,
//! sampler, pipeline and sync-object creates and destroys) to the host's stage-3 decoder, and any
//! reply the host owes it posts on the device completion ring. The two rings each keep a single
//! producer and a single consumer, so their `consumer_idle` handshakes never race.
//! `reims-vgpu.graphics-refresh` runs at the display's refresh rate
//! ([`Screen::mode`]): at each refresh it asks the host whether the guest wrote the scanout
//! ([`Host::scanout_written`]) and has the screen show it where it did, sends `VBLANK` while the
//! guest has it on (dropping one the completion ring has no room for), and sends
//! `DISPLAY_CHANGED` once the display's mode is not the one the guest was last told.
//!
//! **Per-queue rings.** A `QUEUE_CREATE` on the control ring, where its ring parameters are valid,
//! makes a command queue's own submission+completion ring pair at the guest-given region offsets
//! and starts a service thread of its own (`reims-vgpu.graphics-queue-*`). The thread takes that
//! queue's submission ring with the same `consumer_idle` handshake the control thread uses, hands
//! each `COMMAND_BUFFER` message to the host ([`Host::queue_message`]) with a
//! [`QueueCompletionPoster`] onto the queue's completion ring, and the host posts the buffer's
//! completion there, raising the queue's own MSI-X vector. Its doorbell is the submission ring's
//! region offset (the value the guest writes to `DOORBELL`, as the control ring's is 0). A
//! `QUEUE_CREATE` whose ring parameters are refused makes no per-queue ring; a `*_DESTROY` of a
//! queue, and a reset or the device's end, end its service.
//!
//! **The host's calls.** No call of the host's into the device waits for either thread, and no
//! callback of the host's runs inside one: each runs on one of the two threads, with no lock of
//! the device's held, so a callback may wait for something the host holds while it calls the
//! device (QEMU's BQL). [`Device::mmio_write`]'s reset waits only for the message the control
//! thread is taking, which never waits for the host.
//!
//! **The end.** The last [`Device`] handle's drop ends both threads without waiting for them: each
//! sees the end at its next wake (the control thread at once, the refresh thread within a
//! refresh), and the last of them to let go drops its reference to the region and the host's
//! callbacks.
//!
//! **Reset.** Writing `STATUS` 0, and the host's reset, put the device back as it was made: no
//! command is taken until the guest writes `STATUS` 1, both rings are as nothing had passed
//! through them (the submission ring's `consumer_idle` 1, so the guest rings for its first
//! message), the display has no scanout, no cursor image (and shows its cursor once it has one)
//! and no `VBLANK`, and its refresh count is 0.
//! What the guest wrote in its allocations stays.
//!
//! **A broken ring.** A count or a message size the protocol refuses (`ring::Broken`) is logged,
//! and the device takes nothing more from the ring until it is reset; the guest's driver sees its
//! commands go unanswered. A completion ring whose guest count is out of reach gets no event
//! until the reset.

use std::collections::HashMap;
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

use parking_lot::Mutex;

use crate::control;
use crate::display::{Mode, Scanout, Screen};
use crate::region::Region;
use crate::ring::{Consumer, Posted, Producer, RingAt, sized};
use crate::wake::{self, Wake};
use crate::wire::*;

/// The region's size: room for two scanout buffers of an 8K display after the rings, and a
/// power of two, as a BAR's size is.
pub const REGION_SIZE: u64 = 256 << 20;

/// How long the control thread waits before it looks for room in the completion ring again.
const ROOM_POLL: Duration = Duration::from_millis(1);

/// The smallest a command queue's data ring may be: a power of two large enough to hold a
/// completion record with room to spare. A `QUEUE_CREATE` with a smaller ring gets no per-queue
/// ring.
const MIN_QUEUE_RING_DATA: u64 = 4096;

/// The control submission ring.
const SUBMIT: RingAt = RingAt {
    header: RGFX_CONTROL_SUBMIT_OFFSET,
    data_size: RGFX_CONTROL_RING_DATA_SIZE as u64,
};
/// The control completion ring.
const COMPLETE: RingAt = RingAt {
    header: RGFX_CONTROL_COMPLETE_OFFSET,
    data_size: RGFX_CONTROL_RING_DATA_SIZE as u64,
};
/// The device submission ring: the Metal bundle's own, parallel to the control ring, so each ring
/// keeps a single producer and consumer (`wire`'s layout). It carries the object lifecycle.
const DEVICE_SUBMIT: RingAt = RingAt {
    header: RGFX_DEVICE_SUBMIT_OFFSET,
    data_size: RGFX_CONTROL_RING_DATA_SIZE as u64,
};
/// The device completion ring: the bundle's own, for the stage-3 replies the host owes it.
const DEVICE_COMPLETE: RingAt = RingAt {
    header: RGFX_DEVICE_COMPLETE_OFFSET,
    data_size: RGFX_CONTROL_RING_DATA_SIZE as u64,
};

/// What the device asks of the host that runs it (QEMU, through the C API's callbacks). Each is
/// called on one of the device's threads, never inside a call of the host's into the device, and
/// with no lock of the device's held; the two threads may call at once.
pub trait Host: Send + Sync + 'static {
    /// Raise MSI-X vector `vector` (0, the control completion ring's).
    fn raise_interrupt(&self, vector: u32);

    /// Whether the guest wrote any of the `length` bytes at `offset` in the region since the last
    /// time this asked, which starts the next span: the host's dirty log of the range, taken and
    /// cleared. On the refresh thread.
    fn scanout_written(&self, offset: u64, length: u64) -> bool;

    /// Hands a stage-3 message — the object-lifecycle creates and destroy, the pipeline creates,
    /// and any `COMMAND_BUFFER` not on a per-queue ring — to the host's decoder. The device thread
    /// calls this with the message's whole bytes, in the order the guest published them on the
    /// device ring. Any reply the host owes (a pipeline reflection, a command buffer's completion)
    /// it posts onto the device completion ring itself, through [`Device::completion_poster`]; this
    /// returns nothing. The default does nothing, for a host with no stage-3 decoder.
    fn stage3_message(&self, message: &[u8]) {
        let _ = message;
    }

    /// Hands a message taken from a command queue's own submission ring — a `COMMAND_BUFFER` — to
    /// the host's decoder, to run on that queue. A queue's service thread calls this with the
    /// message's whole bytes, in the order the guest published them, and `completion` to post the
    /// buffer's `CB_COMPLETE` (and so raise the queue's MSI-X vector) onto that queue's completion
    /// ring; this returns nothing. The default does nothing, for a host with no stage-3 decoder.
    fn queue_message(&self, completion: &QueueCompletionPoster, message: &[u8]) {
        let _ = (completion, message);
    }
}

/// The device.
pub struct Device {
    /// What its threads share.
    shared: Arc<Shared>,
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("region", &self.shared.region)
            .field("status", &self.shared.status.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

/// What the device's handle and its threads share.
pub(crate) struct Shared {
    /// The host's callbacks.
    pub(crate) host: Box<dyn Host>,
    /// What shows the display.
    pub(crate) screen: Box<dyn Screen>,
    /// BAR 2, which what shows the display may hold too.
    pub(crate) region: Arc<Region>,
    /// `STATUS`.
    status: AtomicU32,
    /// Moved on at each reset: work begun before one drops what it would send.
    epoch: AtomicU64,
    /// Set at the device's end.
    ending: AtomicBool,
    /// The control thread's own wake.
    wake: Wake,
    /// The host's doorbell file for ring 0, once given.
    doorbell: OnceLock<OwnedFd>,
    /// The control submission ring, held while the control thread takes a message.
    submit: Mutex<Submit>,
    /// The display, as the guest has set it.
    pub(crate) display: Mutex<DisplayState>,
    /// The control completion ring.
    completion: Mutex<Producer>,
    /// The device service thread's own wake.
    device_wake: Wake,
    /// The device submission ring (the Metal bundle's own), held while the device thread takes a
    /// message. Its sole producer is the bundle and its sole consumer the device thread.
    device_submit: Mutex<Submit>,
    /// The device completion ring (the bundle's own): the stage-3 replies the host posts it.
    device_completion: Mutex<Producer>,
    /// The device submission ring's count through the last message the host has carried out: a
    /// command buffer waits for it to pass the creates published before it (`take_queue_messages`).
    device_done: AtomicU64,
    /// The command queues' own rings and service threads, by queue id and by ring id.
    queues: Mutex<Queues>,
    /// A weak handle to this shared state, set at the device's make: a queue's service thread and
    /// its completion poster hold it, so neither keeps the device alive.
    weak_self: OnceLock<Weak<Shared>>,
}

/// The command queues' per-queue rings, keyed both by the queue's guest-picked id (for a destroy)
/// and by its ring id — the submission ring's region offset, the `DOORBELL` value (for a wake).
#[derive(Default)]
struct Queues {
    /// By the queue's guest-picked `obj_id`.
    by_obj: HashMap<u64, Arc<QueueShared>>,
    /// By the queue's ring id (its submission ring's region offset).
    by_ring: HashMap<u64, Arc<QueueShared>>,
}

/// One command queue's own rings, shared by its service thread and its completion poster.
pub(crate) struct QueueShared {
    /// The queue's guest-picked id.
    obj_id: u64,
    /// The ring id: the submission ring's region offset, the `DOORBELL` value that wakes it.
    ring_id: u64,
    /// The completion ring's MSI-X vector.
    vector: u32,
    /// The submission ring, held while the service thread takes a message.
    submit: Mutex<Submit>,
    /// The completion ring.
    complete: Mutex<Producer>,
    /// The service thread's own wake.
    wake: Wake,
    /// Set when the queue is destroyed, or at a reset or the device's end.
    ending: AtomicBool,
}

impl QueueShared {
    /// The queue's service is ending (a destroy, a reset, or the device's end).
    fn ended(&self) -> bool {
        self.ending.load(Ordering::Acquire)
    }

    /// Ends the queue's service: its thread stops at its next wake.
    fn end(&self) {
        self.ending.store(true, Ordering::Release);
        self.wake.wake();
    }
}

/// The host's end of the control submission ring.
struct Submit {
    /// The consumer.
    consumer: Consumer,
    /// The ring is broken: nothing is taken until a reset.
    broken: bool,
}

/// The display, as the guest has set it, and what the device has told it.
#[derive(Debug, Default)]
pub(crate) struct DisplayState {
    /// The pixels it shows.
    pub(crate) scanout: Option<Scanout>,
    /// `VBLANK` is on.
    pub(crate) vblank: bool,
    /// Refreshes since the last reset.
    refreshes: u64,
    /// The mode the guest was last told, in a `DISPLAYS` or a `DISPLAY_CHANGED`'s wake.
    pub(crate) told: Option<Mode>,
    /// A `DISPLAY_CHANGED` is owed.
    changed: bool,
}

impl Device {
    /// A device over `region` (BAR 2: [`REGION_SIZE`] bytes, of which what shows the display
    /// may hold a reference to read the scanout from), `host` and `screen`, with its two threads
    /// started; `None` where the region is smaller than the rings and a page of allocations, or
    /// the host refuses a pipe or a thread.
    pub fn new(
        region: Arc<Region>,
        host: Box<dyn Host>,
        screen: Box<dyn Screen>,
    ) -> Option<Device> {
        if region.size() < RGFX_ALLOCATIONS_OFFSET + 4096 {
            return None;
        }
        let mut consumer = Consumer::new(SUBMIT);
        consumer.reset(&region);
        let mut producer = Producer::new(COMPLETE);
        producer.reset(&region);
        let mut device_consumer = Consumer::new(DEVICE_SUBMIT);
        device_consumer.reset(&region);
        let mut device_producer = Producer::new(DEVICE_COMPLETE);
        device_producer.reset(&region);
        let shared = Arc::new(Shared {
            host,
            screen,
            region,
            status: AtomicU32::new(RGFX_STATUS_RESET),
            epoch: AtomicU64::new(0),
            ending: AtomicBool::new(false),
            wake: Wake::new()?,
            doorbell: OnceLock::new(),
            submit: Mutex::new(Submit {
                consumer,
                broken: false,
            }),
            display: Mutex::new(DisplayState::default()),
            completion: Mutex::new(producer),
            device_wake: Wake::new()?,
            device_submit: Mutex::new(Submit {
                consumer: device_consumer,
                broken: false,
            }),
            device_completion: Mutex::new(device_producer),
            device_done: AtomicU64::new(0),
            queues: Mutex::new(Queues::default()),
            weak_self: OnceLock::new(),
        });
        // A queue's service thread and completion poster reach the shared state weakly.
        let _ = shared.weak_self.set(Arc::downgrade(&shared));
        let control = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("reims-vgpu.graphics-control".to_owned())
            .spawn(move || control.control_thread())
            .ok()?;
        let device = Arc::clone(&shared);
        let started = std::thread::Builder::new()
            .name("reims-vgpu.graphics-device".to_owned())
            .spawn(move || device.device_service());
        if started.is_err() {
            shared.end();
            return None;
        }
        let refresh = Arc::clone(&shared);
        let started = std::thread::Builder::new()
            .name("reims-vgpu.graphics-refresh".to_owned())
            .spawn(move || refresh.refresh_thread());
        if started.is_err() {
            shared.end();
            return None;
        }
        Some(Device { shared })
    }

    /// The region's first byte, which the host maps into the guest as BAR 2.
    pub fn region_base(&self) -> *mut u8 {
        self.shared.region.base()
    }

    /// The region's size: BAR 2's.
    pub fn region_size(&self) -> u64 {
        self.shared.region.size()
    }

    /// How many command queues have a per-queue ring and service thread: the tests check a
    /// `QUEUE_CREATE` with valid ring parameters made one.
    pub fn queue_count(&self) -> usize {
        self.shared.queues.lock().by_obj.len()
    }

    /// A 32-bit read of BAR 0 at `offset`. Offsets with no register read 0.
    pub fn mmio_read(&self, offset: u64) -> u32 {
        let size = self.shared.region.size();
        match u32::try_from(offset).unwrap_or(u32::MAX) {
            RGFX_REG_MAGIC => RGFX_MAGIC,
            RGFX_REG_VERSION => RGFX_VERSION,
            RGFX_REG_STATUS => self.shared.status.load(Ordering::Acquire),
            RGFX_REG_REGION_SIZE_LO => (size & 0xffff_ffff) as u32,
            RGFX_REG_REGION_SIZE_HI => (size >> 32) as u32,
            _ => 0,
        }
    }

    /// A 32-bit write of BAR 0 at `offset`: `STATUS` 0 resets the device and 1 starts it taking
    /// commands; `DOORBELL` with ring 0 wakes the control thread. Other values and offsets are
    /// ignored. A reset waits for the message the control thread is taking.
    pub fn mmio_write(&self, offset: u64, value: u32) {
        match u32::try_from(offset).unwrap_or(u32::MAX) {
            RGFX_REG_STATUS if value == RGFX_STATUS_RESET => self.reset(),
            RGFX_REG_STATUS if value == RGFX_STATUS_DRIVER_READY => {
                self.shared
                    .status
                    .store(RGFX_STATUS_DRIVER_READY, Ordering::Release);
                self.shared.wake.wake();
                self.shared.device_wake.wake();
            }
            RGFX_REG_DOORBELL => self.shared.ring_doorbell(u64::from(value)),
            _ => {}
        }
    }

    /// Takes a copy of `fd`, the host's doorbell file for ring 0 (its `ioeventfd`'s, readable
    /// once the guest writes 0 to `DOORBELL`), which the control thread then sleeps on too.
    /// False where `fd` is not open, or the device has one already.
    pub fn set_doorbell(&self, fd: RawFd) -> bool {
        let Some(copy) = wake::duplicate(fd) else {
            return false;
        };
        if self.shared.doorbell.set(copy).is_err() {
            return false;
        }
        self.shared.wake.wake();
        true
    }

    /// The host's reset of the device, as `STATUS` 0 (the module doc).
    pub fn reset(&self) {
        self.shared.reset();
    }

    /// A handle onto the device completion ring that posts host-to-guest stage-3 replies from any
    /// thread: the stage-3 decoder's command-buffer completions (for buffers not on a per-queue
    /// ring), which the GPU raises on a thread of its own, and its pipeline-reflection replies. It
    /// posts onto the bundle's own completion ring, not the control one, so the display events and
    /// the stage-3 replies never share a ring. The handle holds no strong reference to the device,
    /// so it never keeps it alive; a post after the device has ended is dropped.
    pub fn completion_poster(&self) -> ControlCompletionPoster {
        ControlCompletionPoster {
            shared: Arc::downgrade(&self.shared),
        }
    }
}

/// A handle that posts stage-3 replies onto the device completion ring from any thread ([`Device`]'s
/// module doc and [`Device::completion_poster`]): it writes the message and raises the device
/// vector where the guest armed the ring, through that ring's [`Producer`] and interrupt path. It
/// holds a weak reference to the device, so it does not keep it alive; a post after the device has
/// ended, or dropped while it is reset, is dropped. (Its name is kept for the C API; it drives the
/// device ring, not the control one.)
#[derive(Clone)]
pub struct ControlCompletionPoster {
    /// The device's shared state, weakly, so the handle outlives nothing.
    shared: Weak<Shared>,
}

impl ControlCompletionPoster {
    /// Posts `message` (a whole event, its `size` set and a multiple of 8) onto the device
    /// completion ring, waiting for room while the device lives and is not reset, and raising the
    /// device vector where the guest armed the ring. Does nothing where the device has ended.
    pub fn post(&self, message: &[u8]) {
        let Some(shared) = self.shared.upgrade() else {
            return;
        };
        let epoch = shared.epoch.load(Ordering::Acquire);
        shared.device_post_waiting(message, epoch);
    }
}

/// A handle that posts events onto one command queue's completion ring from any thread: the
/// stage-3 decoder's command-buffer completions for buffers submitted on that queue, which the GPU
/// raises on a thread of its own. It writes the message and raises the queue's own MSI-X vector
/// where the guest armed the ring, through that queue's [`Producer`]. It holds weak references to
/// the device and the queue, so it keeps neither alive; a post after the device has ended, after
/// the queue is destroyed, or while the device is reset, is dropped.
#[derive(Clone)]
pub struct QueueCompletionPoster {
    /// The device's shared state, weakly.
    shared: Weak<Shared>,
    /// The queue's own rings, weakly.
    queue: Weak<QueueShared>,
}

impl QueueCompletionPoster {
    /// Posts `message` (a whole event, its `size` set and a multiple of 8) onto the queue's
    /// completion ring, waiting for room while the device and the queue live and the device is not
    /// reset, and raising the queue's MSI-X vector where the guest armed the ring. Does nothing
    /// where the device has ended or the queue is gone.
    pub fn post(&self, message: &[u8]) {
        let (Some(shared), Some(queue)) = (self.shared.upgrade(), self.queue.upgrade()) else {
            return;
        };
        let epoch = shared.epoch.load(Ordering::Acquire);
        shared.post_queue_waiting(&queue, message, epoch);
    }
}

impl Drop for Device {
    /// The device's end: both threads end without being waited for (the module doc).
    fn drop(&mut self) {
        self.shared.end();
    }
}

/// What a post into the completion ring did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Post {
    /// Published, and the interrupt raised where the guest had armed the ring.
    Done,
    /// No room until the guest takes more.
    NoRoom,
    /// Dropped: a reset came after the work began, the device is not started, or the ring is
    /// broken.
    Dropped,
}

impl Shared {
    /// The device's end: every thread stops at its next wake.
    fn end(&self) {
        self.ending.store(true, Ordering::Release);
        self.wake.wake();
        self.device_wake.wake();
        for queue in self.queues.lock().by_obj.values() {
            queue.end();
        }
    }

    fn ending(&self) -> bool {
        self.ending.load(Ordering::Acquire)
    }

    fn started(&self) -> bool {
        self.status.load(Ordering::Acquire) == RGFX_STATUS_DRIVER_READY
    }

    /// The reset (the module doc).
    fn reset(&self) {
        self.status.store(RGFX_STATUS_RESET, Ordering::Release);
        self.epoch.fetch_add(1, Ordering::AcqRel);
        let mut submit = self.submit.lock();
        let mut display = self.display.lock();
        let mut completion = self.completion.lock();
        let mut device_submit = self.device_submit.lock();
        let mut device_completion = self.device_completion.lock();
        submit.consumer.reset(&self.region);
        submit.broken = false;
        completion.reset(&self.region);
        device_submit.consumer.reset(&self.region);
        device_submit.broken = false;
        self.device_done.store(0, Ordering::Release);
        device_completion.reset(&self.region);
        *display = DisplayState::default();
        drop(device_completion);
        drop(device_submit);
        drop(completion);
        drop(display);
        drop(submit);
        // The per-queue rings are the guest's to make again after a reset: end their services.
        let mut queues = self.queues.lock();
        for queue in queues.by_obj.values() {
            queue.end();
        }
        queues.by_obj.clear();
        queues.by_ring.clear();
        drop(queues);
        self.screen.scanout(None);
        self.screen.cursor_image(None);
        self.screen.cursor_moved(0, 0, true);
    }

    /// The control thread (the module doc).
    fn control_thread(&self) {
        while !self.ending() {
            self.take_messages();
            // Asleep with `consumer_idle` set only while the device takes the ring.
            let idle = {
                let submit = self.submit.lock();
                if !self.started() || submit.broken {
                    false
                } else if submit.consumer.sleep_or_pending(&self.region) {
                    continue;
                } else {
                    true
                }
            };
            if self.ending() {
                break;
            }
            self.wake.sleep(self.doorbell.get().map(AsRawFd::as_raw_fd));
            if idle {
                self.submit.lock().consumer.awake(&self.region);
            }
        }
    }

    /// Takes and answers every message the guest has published, while the device is started and
    /// the ring is not broken. Each message is taken and done with the ring held, so a reset
    /// comes before or after it whole; its answer is posted with nothing held, so the interrupt
    /// is raised with no lock of the device's held, and dropped where a reset came between.
    fn take_messages(&self) {
        loop {
            let (answer, epoch) = {
                let mut submit = self.submit.lock();
                if !self.started() || submit.broken || self.ending() {
                    return;
                }
                let epoch = self.epoch.load(Ordering::Acquire);
                match submit.consumer.next(&self.region) {
                    Ok(Some(message)) => (control::answer(self, &message), epoch),
                    Ok(None) => return,
                    Err(broken) => {
                        log::error!(
                            target: "reims_graphics",
                            "the control submission ring is broken ({broken}): it takes nothing \
                             more until the device is reset"
                        );
                        submit.broken = true;
                        return;
                    }
                }
            };
            if let Some(answer) = answer {
                self.post_waiting(&answer, epoch);
            }
        }
    }

    /// Posts `message`, waiting for room while the device is not reset or ended.
    fn post_waiting(&self, message: &[u8], epoch: u64) {
        while self.post(message, epoch) == Post::NoRoom {
            if self.ending() {
                return;
            }
            std::thread::sleep(ROOM_POLL);
        }
    }

    /// Posts `message` into the completion ring, for work begun at reset count `epoch`, and
    /// raises the ring's interrupt where the guest armed it, with no lock held.
    fn post(&self, message: &[u8], epoch: u64) -> Post {
        let raise = {
            let mut completion = self.completion.lock();
            if self.epoch.load(Ordering::Acquire) != epoch || !self.started() {
                return Post::Dropped;
            }
            match completion.post(&self.region, message) {
                Posted::Yes { raise } => raise,
                Posted::NoRoom => return Post::NoRoom,
                Posted::Broken(broken) => {
                    log::warn!(
                        target: "reims_graphics",
                        "the control completion ring is broken ({broken}): an event is dropped"
                    );
                    return Post::Dropped;
                }
            }
        };
        if raise {
            self.host.raise_interrupt(RGFX_CONTROL_VECTOR);
        }
        Post::Done
    }

    /// Posts `message` onto the device completion ring, waiting for room while the device is not
    /// reset or ended.
    fn device_post_waiting(&self, message: &[u8], epoch: u64) {
        while self.device_post(message, epoch) == Post::NoRoom {
            if self.ending() {
                return;
            }
            std::thread::sleep(ROOM_POLL);
        }
    }

    /// Posts `message` into the device completion ring, for work begun at reset count `epoch`, and
    /// raises the device ring's interrupt where the guest armed it, with no lock held.
    fn device_post(&self, message: &[u8], epoch: u64) -> Post {
        let raise = {
            let mut completion = self.device_completion.lock();
            if self.epoch.load(Ordering::Acquire) != epoch || !self.started() {
                return Post::Dropped;
            }
            match completion.post(&self.region, message) {
                Posted::Yes { raise } => raise,
                Posted::NoRoom => return Post::NoRoom,
                Posted::Broken(broken) => {
                    log::warn!(
                        target: "reims_graphics",
                        "the device completion ring is broken ({broken}): an event is dropped"
                    );
                    return Post::Dropped;
                }
            }
        };
        if raise {
            self.host.raise_interrupt(RGFX_DEVICE_VECTOR);
        }
        Post::Done
    }

    /// The device service thread (the module doc): it takes the device submission ring with the
    /// control thread's `consumer_idle` handshake, hands every message to the host's stage-3
    /// decoder, and sleeps on the device wake, woken by the device doorbell, the guest's start or
    /// the device's end.
    fn device_service(&self) {
        while !self.ending() {
            self.take_device_messages();
            let idle = {
                let submit = self.device_submit.lock();
                if !self.started() || submit.broken {
                    false
                } else if submit.consumer.sleep_or_pending(&self.region) {
                    continue;
                } else {
                    true
                }
            };
            if self.ending() {
                break;
            }
            self.device_wake.sleep(None);
            if idle {
                self.device_submit.lock().consumer.awake(&self.region);
            }
        }
    }

    /// Takes and hands to the host every message the bundle has published on the device submission
    /// ring, while the device is started and the ring is not broken. Each message is a stage-3
    /// message: it is taken with the ring held and handed to [`control::device_message`] (so a
    /// queue create or destroy drives the per-queue rings too) with nothing held; any reply the
    /// host owes it posts on the device completion ring itself, so nothing is posted here.
    fn take_device_messages(&self) {
        loop {
            let (message, through) = {
                let mut submit = self.device_submit.lock();
                if !self.started() || submit.broken || self.ending() {
                    return;
                }
                match submit.consumer.next(&self.region) {
                    Ok(Some(message)) => (message, submit.consumer.consumed()),
                    Ok(None) => return,
                    Err(broken) => {
                        log::error!(
                            target: "reims_graphics",
                            "the device submission ring is broken ({broken}): it takes nothing \
                             more until the device is reset"
                        );
                        submit.broken = true;
                        return;
                    }
                }
            };
            control::device_message(self, &message);
            self.device_done.store(through, Ordering::Release);
        }
    }

    /// A doorbell write of ring id `ring_id`: wakes the control thread (ring id 0) or the service
    /// thread of the queue whose submission ring is at that region offset; an unknown id is
    /// ignored.
    fn ring_doorbell(&self, ring_id: u64) {
        if ring_id == u64::from(RGFX_CONTROL_RING_ID) {
            self.wake.wake();
            return;
        }
        if ring_id == u64::from(RGFX_DEVICE_RING_ID) {
            self.device_wake.wake();
            return;
        }
        if let Some(queue) = self.queues.lock().by_ring.get(&ring_id) {
            queue.wake.wake();
        }
    }

    /// A `QUEUE_CREATE`'s ring pair, where its parameters are valid: each data ring a power of two
    /// no smaller than [`MIN_QUEUE_RING_DATA`], each ring 8-aligned in the guest's allocations and
    /// within the region, and the two not overlapping. `None` where any is refused, so the queue
    /// gets no per-queue ring.
    fn validate_rings(&self, q: &RgfxQueueCreate) -> Option<(RingAt, RingAt)> {
        let submit_data = u64::from(q.submit_data_size);
        let complete_data = u64::from(q.complete_data_size);
        if !submit_data.is_power_of_two() || !complete_data.is_power_of_two() {
            return None;
        }
        if submit_data < MIN_QUEUE_RING_DATA || complete_data < MIN_QUEUE_RING_DATA {
            return None;
        }
        if !q.submit_offset.is_multiple_of(8) || !q.complete_offset.is_multiple_of(8) {
            return None;
        }
        if q.submit_offset < RGFX_ALLOCATIONS_OFFSET || q.complete_offset < RGFX_ALLOCATIONS_OFFSET
        {
            return None;
        }
        let header = u64::from(RGFX_RING_HEADER_SIZE);
        let submit_span = header + submit_data;
        let complete_span = header + complete_data;
        if !self.region.contains(q.submit_offset, submit_span)
            || !self.region.contains(q.complete_offset, complete_span)
        {
            return None;
        }
        // The two rings must not overlap each other.
        let (s0, s1) = (q.submit_offset, q.submit_offset + submit_span);
        let (c0, c1) = (q.complete_offset, q.complete_offset + complete_span);
        if s0 < c1 && c0 < s1 {
            return None;
        }
        Some((
            RingAt {
                header: q.submit_offset,
                data_size: submit_data,
            },
            RingAt {
                header: q.complete_offset,
                data_size: complete_data,
            },
        ))
    }

    /// Makes a command queue's own submission+completion ring pair and starts its service thread,
    /// where the create's ring parameters are valid (otherwise nothing, and the queue runs its
    /// command buffers on the control ring). The host has already made the decoder's queue handle,
    /// so a command buffer the service thread takes can name it.
    pub(crate) fn create_queue(&self, q: &RgfxQueueCreate) {
        let Some((submit_at, complete_at)) = self.validate_rings(q) else {
            log::warn!(
                target: "reims_graphics",
                "queue {:#x}: its ring parameters are refused (submit {:#x}/{}, complete \
                 {:#x}/{}); it gets no per-queue ring",
                q.obj_id, q.submit_offset, q.submit_data_size, q.complete_offset,
                q.complete_data_size
            );
            return;
        };
        let Some(me) = self.weak_self.get().and_then(Weak::upgrade) else {
            return;
        };
        let Some(wake) = Wake::new() else {
            log::error!(
                target: "reims_graphics",
                "queue {:#x}: no wake pipe; it gets no per-queue ring", q.obj_id
            );
            return;
        };
        // Take the ring headers as the guest left them: it zeroes both before the create, and may
        // publish its first command buffer on the submission ring before the host gets here (the
        // create rides the device ring, the buffer the queue's own). A reset here would clear
        // that producer count and lose the buffer, and the guest, seeing `consumer_idle` 0, rang
        // no doorbell for it. The service thread takes what is published before it first sleeps,
        // and sets `consumer_idle` itself when it does.
        let consumer = Consumer::new(submit_at);
        let producer = Producer::new(complete_at);
        let queue = Arc::new(QueueShared {
            obj_id: q.obj_id,
            ring_id: q.submit_offset,
            vector: q.vector,
            submit: Mutex::new(Submit {
                consumer,
                broken: false,
            }),
            complete: Mutex::new(producer),
            wake,
            ending: AtomicBool::new(false),
        });
        {
            let mut queues = self.queues.lock();
            // A re-create under the same queue id or ring offset ends the old one first.
            if let Some(old) = queues.by_obj.remove(&q.obj_id) {
                queues.by_ring.remove(&old.ring_id);
                old.end();
            }
            if let Some(old) = queues.by_ring.remove(&q.submit_offset) {
                queues.by_obj.remove(&old.obj_id);
                old.end();
            }
            queues.by_obj.insert(q.obj_id, Arc::clone(&queue));
            queues.by_ring.insert(q.submit_offset, Arc::clone(&queue));
        }
        let service = Arc::clone(&queue);
        let started = std::thread::Builder::new()
            .name(format!("reims-vgpu.graphics-queue-{:#x}", q.obj_id))
            .spawn(move || me.queue_service(&service));
        if started.is_err() {
            let mut queues = self.queues.lock();
            queues.by_obj.remove(&q.obj_id);
            queues.by_ring.remove(&q.submit_offset);
            queue.end();
            log::error!(
                target: "reims_graphics",
                "queue {:#x}: no service thread; it gets no per-queue ring", q.obj_id
            );
        }
    }

    /// Ends and forgets the queue `obj_id`'s per-queue service, where it has one (a `*_DESTROY` of
    /// a command queue). An id that is not a queue's is ignored.
    pub(crate) fn destroy_queue(&self, obj_id: u64) {
        let mut queues = self.queues.lock();
        if let Some(queue) = queues.by_obj.remove(&obj_id) {
            queues.by_ring.remove(&queue.ring_id);
            queue.end();
        }
    }

    /// A completion poster onto `queue`'s completion ring, held by its service thread (the module
    /// doc and [`QueueCompletionPoster`]).
    fn queue_poster(&self, queue: &Arc<QueueShared>) -> QueueCompletionPoster {
        QueueCompletionPoster {
            shared: self.weak_self.get().cloned().unwrap_or_else(Weak::new),
            queue: Arc::downgrade(queue),
        }
    }

    /// A queue's service thread (the module doc): it takes the queue's submission ring with the
    /// control thread's `consumer_idle` handshake, hands each message to the host, and sleeps on
    /// the queue's own wake, woken by its doorbell or its end.
    fn queue_service(&self, queue: &Arc<QueueShared>) {
        let poster = self.queue_poster(queue);
        while !self.ending() && !queue.ended() {
            self.take_queue_messages(queue, &poster);
            let idle = {
                let submit = queue.submit.lock();
                if !self.started() || submit.broken || queue.ended() {
                    false
                } else if submit.consumer.sleep_or_pending(&self.region) {
                    continue;
                } else {
                    true
                }
            };
            if self.ending() || queue.ended() {
                break;
            }
            queue.wake.sleep(None);
            if idle {
                queue.submit.lock().consumer.awake(&self.region);
            }
        }
    }

    /// Takes and hands to the host every message the guest has published on `queue`'s submission
    /// ring, while the device is started and the ring is not broken. Each message is taken with the
    /// ring held and handed over with nothing held; the completion the host owes is posted onto the
    /// queue's completion ring through `poster`, from the GPU's thread.
    fn take_queue_messages(&self, queue: &Arc<QueueShared>, poster: &QueueCompletionPoster) {
        loop {
            let message = {
                let mut submit = queue.submit.lock();
                if !self.started() || submit.broken || self.ending() || queue.ended() {
                    return;
                }
                match submit.consumer.next(&self.region) {
                    Ok(Some(message)) => message,
                    Ok(None) => return,
                    Err(broken) => {
                        log::error!(
                            target: "reims_graphics",
                            "queue {:#x}'s submission ring is broken ({broken}): it takes nothing \
                             more until the device is reset", queue.obj_id
                        );
                        submit.broken = true;
                        return;
                    }
                }
            };
            // Order this command buffer after the resource creates the guest posted before it.
            // Creates ride the device ring, whose sole consumer is the device thread; command
            // buffers ride per-queue rings (this thread). A create a buffer uses is posted before
            // it in guest order, so it is within the device ring's producer at the moment this
            // thread takes the buffer. Wait for the device thread to have carried out every message
            // up to there, so the resource is registered before the buffer's ops name it: the
            // consumer's own count moves when a message is taken, before it is decoded, so it
            // is not enough. This thread only reads the device ring's counts; it never consumes
            // it, so the device ring keeps one consumer and its `consumer_idle` handshake is
            // untouched (draining it here would break both).
            let created_through = {
                self.device_submit
                    .lock()
                    .consumer
                    .published_count(&self.region)
            };
            while self.device_done.load(Ordering::Acquire) < created_through {
                if self.ending() || queue.ended() {
                    return;
                }
                std::thread::sleep(ROOM_POLL);
            }
            self.host.queue_message(poster, &message.bytes);
        }
    }

    /// Posts `message` onto `queue`'s completion ring, waiting for room while the device is not
    /// reset or ended and the queue lives.
    fn post_queue_waiting(&self, queue: &QueueShared, message: &[u8], epoch: u64) {
        while self.post_queue(queue, message, epoch) == Post::NoRoom {
            if self.ending() || queue.ended() {
                return;
            }
            std::thread::sleep(ROOM_POLL);
        }
    }

    /// Posts `message` into `queue`'s completion ring, for work begun at reset count `epoch`, and
    /// raises the queue's MSI-X vector where the guest armed it, with no lock held.
    fn post_queue(&self, queue: &QueueShared, message: &[u8], epoch: u64) -> Post {
        let raise = {
            let mut completion = queue.complete.lock();
            if self.epoch.load(Ordering::Acquire) != epoch || !self.started() || queue.ended() {
                return Post::Dropped;
            }
            match completion.post(&self.region, message) {
                Posted::Yes { raise } => raise,
                Posted::NoRoom => return Post::NoRoom,
                Posted::Broken(broken) => {
                    log::warn!(
                        target: "reims_graphics",
                        "queue {:#x}'s completion ring is broken ({broken}): an event is dropped",
                        queue.obj_id
                    );
                    return Post::Dropped;
                }
            }
        };
        if raise {
            self.host.raise_interrupt(queue.vector);
        }
        Post::Done
    }

    /// The refresh thread (the module doc): a refresh at each period of the display's mode, on
    /// the monotonic clock, a late one taken at once and the next a period after it.
    fn refresh_thread(&self) {
        let mut next = now_ns();
        while !self.ending() {
            let period = self.screen.mode().period_ns();
            next = next.saturating_add(period);
            let now = now_ns();
            if next > now {
                std::thread::sleep(Duration::from_nanos(next - now));
            } else {
                next = now;
            }
            if self.ending() {
                break;
            }
            self.refresh(now_ns());
        }
    }

    /// One refresh, at `now` on the monotonic clock.
    pub(crate) fn refresh(&self, now: u64) {
        let epoch = self.epoch.load(Ordering::Acquire);
        let mode = self.screen.mode();
        let (scanout, vblank, changed) = {
            let mut display = self.display.lock();
            display.refreshes = display.refreshes.wrapping_add(1);
            if display.told.is_some_and(|told| told != mode) {
                display.told = Some(mode);
                display.changed = true;
            }
            (
                display.scanout,
                display.vblank.then_some(display.refreshes),
                display.changed,
            )
        };
        if let Some(scanout) = scanout
            && self.host.scanout_written(scanout.offset, scanout.span())
        {
            self.screen.present();
        }
        if let Some(count) = vblank {
            let event = RgfxVblank {
                header: event_header(RGFX_EVT_VBLANK),
                display: 0,
                reserved_14: 0,
                count,
                host_time_ns: now,
            };
            // One the ring has no room for is dropped; the next carries the count.
            let _ = self.post(&sized(&event), epoch);
        }
        if changed {
            let event = RgfxDisplayChanged {
                header: event_header(RGFX_EVT_DISPLAY_CHANGED),
                display: 0,
                reserved_14: 0,
            };
            if self.post(&sized(&event), epoch) == Post::Done {
                self.display.lock().changed = false;
            }
        }
    }
}

/// The header of an event that answers no command: `seqno` 0, the size set when it is sized.
fn event_header(r#type: u32) -> RgfxMsgHeader {
    RgfxMsgHeader {
        r#type,
        size: 0,
        seqno: 0,
    }
}

/// The host's monotonic clock, in nanoseconds.
fn now_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `clock_gettime` writes the timespec.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    let secs = u64::try_from(ts.tv_sec).unwrap_or(0);
    let nanos = u64::try_from(ts.tv_nsec).unwrap_or(0);
    secs.saturating_mul(1_000_000_000).saturating_add(nanos)
}

#[cfg(test)]
pub(crate) mod tests;
