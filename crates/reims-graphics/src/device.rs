//! [`Device`]: the device's host side, as the host's process runs it. It owns the region (BAR 2),
//! answers the registers (BAR 0), takes the guest's control commands on a thread of its own, and
//! sends the display's events on another.
//!
//! **Threads.** `reims-vgpu.graphics-control` takes the control submission ring: woken by a
//! doorbell (the host's `ioeventfd`, or a `DOORBELL` write through [`Device::mmio_write`]) or by
//! the guest's `STATUS`, it takes every message the guest has published, answers each on the
//! completion ring, then sleeps with the ring's `consumer_idle` handshake (`ring`). While the
//! completion ring has no room, it waits for the guest to take more, polling, and takes no
//! command meanwhile. `reims-vgpu.graphics-refresh` runs at the display's refresh rate
//! ([`Screen::mode`]): at each refresh it asks the host whether the guest wrote the scanout
//! ([`Host::scanout_written`]) and has the screen show it where it did, sends `VBLANK` while the
//! guest has it on (dropping one the completion ring has no room for), and sends
//! `DISPLAY_CHANGED` once the display's mode is not the one the guest was last told.
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

use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
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
    host: Box<dyn Host>,
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
        });
        let control = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("reims-vgpu.graphics-control".to_owned())
            .spawn(move || control.control_thread())
            .ok()?;
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
            }
            RGFX_REG_DOORBELL if value == RGFX_CONTROL_RING_ID => self.shared.wake.wake(),
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
    /// The device's end: both threads stop at their next wake.
    fn end(&self) {
        self.ending.store(true, Ordering::Release);
        self.wake.wake();
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
        submit.consumer.reset(&self.region);
        submit.broken = false;
        completion.reset(&self.region);
        *display = DisplayState::default();
        drop(completion);
        drop(display);
        drop(submit);
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
