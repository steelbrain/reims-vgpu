//! [`Device`]: the device's host side, as the host's process runs it. It owns the region (BAR 2),
//! answers the registers (BAR 0), takes the guest's control commands on a thread of its own,
//! carries each process's out on a thread of the process's, and sends the display's events on
//! another.
//!
//! **Threads.** `reims-vgpu.graphics-control` takes the control submission ring: woken by a
//! doorbell (the host's `ioeventfd`, or a `DOORBELL` write through [`Device::mmio_write`]) or by
//! the guest's `STATUS`, it takes every message the guest has published, answers each on the
//! completion ring, then sleeps with the ring's `consumer_idle` handshake (`ring`). While the
//! completion ring has no room, it waits for the guest to take more (`Room`), and takes no
//! command meanwhile. The guest's kexts are its sole producer, under one lock: the framebuffer's
//! display commands, and the object lifecycle the accelerator posts for the guest's processes
//! (creates and destroys, `QUEUE_CREATE`, `CONTEXT_CLOSE`), which the control thread hands, with
//! no ring held, to the lane of the process's context (`lanes`): the context's lane carries its
//! messages out in ring order, handing its creates and destroys to the host's decoder
//! ([`Host::stage3_message`]), so one context's slow message holds up only its own later ones.
//! The presents' lane carries the `PRESENT_SURFACE`s out, in ring order, and a block's lane the
//! block's `SURFACE_PAGES` and `SURFACE_DOWNLOAD`s, whichever process posted them, but for a
//! download the host carries out on the control thread, where the block's lane has nothing left
//! and the host need wait for nothing ([`Host::try_surface_download`]). Ring 0's threads,
//! `reims-vgpu.graphics-ring0-*`, serve the lanes, each lane with a job on one of them.
//! The replies (a pipeline's reflection, a `COMPLETE`) come back on the control completion ring.
//! Every other ring is a command queue's own (below), so each ring keeps a single producer and a
//! single consumer, and their `consumer_idle` handshakes never race.
//! `reims-vgpu.graphics-refresh` runs at the display's refresh rate
//! ([`Screen::mode`]): at each refresh it asks the host whether the guest wrote the scanout
//! ([`Host::scanout_written`]) and has the screen show it where it did, sends `VBLANK` while the
//! guest has it on, stamped with the refresh's time on its grid (`RefreshGrid`) as a panel's
//! hardware stamps its vblank (dropping one the completion ring has no room for), sends
//! `DISPLAY_CHANGED` once the display's mode is not the one the guest was last told, and then
//! tells the host of the refresh ([`Host::refreshed`]), which answers there the presents the
//! display first shows at it.
//!
//! **Per-queue rings.** A `QUEUE_CREATE` on ring 0, where its ring parameters are valid (each
//! ring whole in one block its process holds, among them), makes the
//! queue's own submission+completion ring pair at the guest-given region offsets and starts a
//! service thread of its own (`reims-vgpu.graphics-queue-*`) before it is answered, so the process
//! may write the ring once the create returns. The host is told of the queue then too, before the
//! answer ([`Host::queue_made`]): a buffer of another queue may name the queue's buffers from the
//! answer on, before the queue's thread has had the host make its handle. The thread first has
//! the host make the queue's handle (the `QUEUE_CREATE` handed to [`Host::stage3_message`]), then
//! takes the queue's
//! submission ring with the same `consumer_idle` handshake the control thread uses, hands each
//! `COMMAND_BUFFER` to the host with a [`CompletionPoster`] onto the queue's completion ring (a
//! `COMMAND_BUFFER_IN_REGION`'s message is taken from the process's block it names, with the
//! reference's seqno; a message of any other type is refused there and goes no further:
//! lifecycle messages ride ring 0, through the kext, which checks that their sender owns what
//! they name), and the host
//! posts what it owes there (a buffer's completion), raising the queue's own MSI-X vector. Its
//! doorbell is the submission ring's region offset (the value the guest writes to `DOORBELL`, as
//! the control ring's is 0). A command buffer is handed to the host only once every ring-0 message
//! of the buffer's context's published before the thread took the buffer is carried out by the
//! context's lane, and the control thread has taken every message before those (`backlog`:
//! another context's backlog holds it up only where it lies before one of its own, and only for
//! as long as the control thread takes to hand it to its lane), so the creates it names are made,
//! or, for a pipeline the host compiles on threads of its own, known to the host, which holds a
//! buffer that names it until it is made. The thread sleeps for those at the gate, which wakes
//! it only once they are done, whatever other processes' messages (`gate`). An
//! `OBJECT_DESTROY` of the queue, a
//! `CONTEXT_CLOSE` of its process (the host then
//! drops the process's objects, [`Host::context_closed`]), a reset and the device's end, end a
//! queue's service.
//!
//! **The host's calls.** No call of the host's into the device waits for its threads, and no
//! callback of the host's runs inside one but [`Device::mmio_write_taking`]'s: each runs on a
//! thread of the device's, or, for [`Host::raise_interrupt`], on the thread that posts its event,
//! with no lock held that a call of the device's waits for, directly or through a thread of the
//! device's ([`Host`]), so a callback may call the device at once, and may wait for something the
//! host holds while it calls the device (QEMU's BQL). [`Device::mmio_write`]'s reset waits only
//! for the message the control thread is taking, which never waits for the host.
//!
//! **An answer on the doorbell's thread.** A `DOORBELL` write of a command queue's ring through
//! [`Device::mmio_write_taking`] offers the queue's next `COMMAND_BUFFER` to the host on the
//! writing thread (the guest's vCPU), ahead of the queue's service thread
//! ([`Host::answer_command_buffer`]), which the host may answer there: it posts the buffer's
//! `CB_SUBMITTED`, raising the queue's vector inside the call. The buffer stays on the ring, and
//! the service thread takes it next, as it takes any, and hands it to the host
//! ([`Host::stage3_message`]) without waiting for ring 0 again: the doorbell's thread found ring 0
//! carried out as far as the buffer waits for. The offer is made only where nothing need be
//! waited for and nothing allocated: no lock of the device's is held elsewhere (each is tried; the
//! queues' table, held elsewhere, is waited for as [`Device::mmio_write`] waits for it, and no
//! offer is made), the service thread holds no message of the queue's, none answered is left for
//! it to take, the buffer fits the thread's copy of it (`DOORBELL_MESSAGE_MAX`), at most
//! `DOORBELL_READS` of ring 0's messages are left to read, and every ring-0 message the buffer
//! waits for is carried out. Otherwise the service thread takes the buffer as it takes any, so
//! the queue's events keep its order either way. The service thread's hold on the queue's
//! messages lasts through the host's begin of the buffer it hands over, which places that buffer
//! among the queue's ([`QueueClaim`]): a host that then walks and commits the buffer on that
//! thread lets the hold go first, and offers the queue's next buffer there, as a doorbell's
//! thread does, so a buffer rung for meanwhile is answered before that walk, not after it.
//!
//! **An answer on a lane's thread.** A buffer the service thread takes and must wait for ring 0
//! for is filed at the queue gate with its message ([`gate`]'s offer). The lane whose move ends
//! the wait offers it to the host on the lane's thread ([`Host::answer_command_buffer`]) before
//! it wakes the service thread, where ring 0 is carried out as far as the buffer waits for, so
//! the host may post its `CB_SUBMITTED` there, as on a doorbell's thread, and the queue's
//! thread's wake no longer comes before it. The service thread hands the buffer over once no
//! offer of it is under way, as it hands any.
//!
//! **The end.** The last [`Device`] handle's drop ends its threads without waiting for them: each
//! sees the end at its next wake (the control thread and the lanes' at once, or once the message
//! a lane carries out returns, the refresh thread within a refresh), and the last of them to let
//! go drops its reference to the region and the host's callbacks.
//!
//! **Reset.** Writing `STATUS` 0, and the host's reset, put the device back as it was made, as a
//! device's function reset does: no command is taken until the guest writes `STATUS` 1, the
//! guest's driver has taken no feature (`DRIVER_FEATURES` reads 0), both rings are as nothing
//! had passed through them (the submission ring's `consumer_idle` 1, so the guest rings for its
//! first message), the display keeps its scanout (below) but has no cursor image (and shows its
//! cursor once it has one) and no `VBLANK`, and its refresh count is 0. Every command queue ends,
//! and once a take of its ring or a post onto it under way has finished nothing touches its rings
//! again; no block is any process's (the kext grants them again), and the host drops everything
//! it made for the guest's processes ([`Host::device_reset`], on the control thread before it
//! takes anything the guest publishes after the reset, and once every message a lane began before
//! it has returned; a lane's message not yet begun is dropped). Nothing begun before the reset
//! reaches
//! the guest after it: its answers and events are dropped, and a queue made before it hands the
//! host nothing more, not even its own destroy, which the host's reset stands for.
//! What the guest wrote in its allocations stays.
//!
//! **The scanout across a reset.** A reset leaves the display's scanout as the last `SET_SCANOUT`
//! set it, as a GPU's display engine goes on scanning out across a reset of its driver: the
//! screen is told nothing, and each refresh shows the guest's writes there as before, until the
//! guest sets another scanout or blanks it (width 0). So a guest driver that resets the device
//! and then fails leaves the host's window showing what the guest draws next: the kext's start
//! resets the device to take it, and where it fails after that (the device lists no usable
//! display, `IOFramebuffer`'s start fails) it resets it again, and the OS falls back to the
//! firmware's framebuffer, which the firmware's scanout, kept through both resets, shows.
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

use parking_lot::{Condvar, Mutex, MutexGuard};
use zerocopy::{FromBytes, IntoBytes};

use crate::control;
use crate::device::backlog::Backlog;
use crate::device::gate::{Gate, Look, Offer};
use crate::device::lanes::{Hand, Key, Lanes};
use crate::display::{Mode, Scanout, Screen};
use crate::ops::RgfxCommandBuffer;
use crate::region::Region;
use crate::ring::{Consumer, Message, Posted, Producer, RingAt, sized};
use crate::wake::{self, Wake};
use crate::wire::*;

mod backlog;
mod gate;
mod lanes;

/// How many of ring 0's messages a doorbell's take reads ([`Backlog::read`]) before it leaves the
/// buffer to its queue's thread.
const DOORBELL_READS: usize = 32;

/// The region's size: the control rings, the framebuffer's scanout span and cursor, and room for
/// the guest processes' rings, buffers, heaps and surfaces (a 3840x2400 BGRA8 surface is 37 MB);
/// a power of two, as a BAR's size is.
///
/// 1 GiB is the most an x86-64 macOS guest takes (measured on macOS 26, q35, OVMF): macOS's PCI
/// configurator lays the bus out again at boot and places every 64-bit prefetchable BAR below
/// 4 GiB, never in the host bridge's 64-bit window, and it misreads a 4 GiB BAR. Below 4 GiB the
/// window ends at 0xE000_0000 (the ECAM area follows), so a 1 GiB BAR needs the window to start
/// at or below 0x4000_0000: the host caps the guest's RAM below 4 GiB at 1 GiB (QEMU's
/// `-machine max-ram-below-4g=1G`). With q35's default of 2 GiB there the BAR gets no address.
pub const REGION_SIZE: u64 = 1 << 30;

/// How long a post waits for room in a full completion ring before it looks again, where nothing
/// wakes it sooner ([`Room`]): the guest makes room by moving the ring's consumer count, which
/// rings nothing.
const ROOM_POLL: Duration = Duration::from_millis(1);

/// Where the posts onto one full completion ring wait for room, each on the thread that posts
/// it. The guest makes room by moving the ring's consumer count, a write to the region that
/// wakes nothing, so a waiting post looks again after the device's poll ([`ROOM_POLL`]); what
/// the host does see wakes it at once: the device's reset, the queue's end and the device's, and
/// the ring's doorbell, which the guest rings for what it publishes while the ring's consumer
/// sleeps (`consumer_idle`; the bundle also once as it finds a queue's submission ring full). So a
/// queue's doorbell wakes a post of a GPU thread's when the process submits again, and ring 0's,
/// rung through [`Device::mmio_write`] or the host's doorbell file, a post of a thread other than
/// the control thread (a compile's reflection, an answer given on a GPU thread) when the kext posts
/// while the control thread sleeps. The guest's take of a full ring rings nothing, so the control
/// thread's own post, and any post while its ring's consumer is awake, looks again only at the
/// poll. The wake for that take is the wire's version 3 `CONSUMED` doorbell, which the guest
/// writes after it takes from a ring the host waits on.
#[derive(Default)]
struct Room {
    /// Moved on at each wake, so a wake between a post's look and its wait ends the wait.
    wakes: AtomicU64,
    /// Held around a wait's look at `wakes` and by a wake's notify.
    lock: Mutex<()>,
    /// Notified at each wake.
    woken: Condvar,
}

impl Room {
    /// The wakes so far, read before a post looks for room.
    fn wakes(&self) -> u64 {
        self.wakes.load(Ordering::Acquire)
    }

    /// Wakes the posts waiting for room: they look again at once.
    fn wake(&self) {
        self.wakes.fetch_add(1, Ordering::AcqRel);
        drop(self.lock.lock());
        self.woken.notify_all();
    }

    /// Waits for a wake after `seen` ([`Self::wakes`] before the post that found no room), or
    /// for `poll`.
    fn wait(&self, seen: u64, poll: Duration) {
        let mut held = self.lock.lock();
        if self.wakes.load(Ordering::Acquire) == seen {
            self.woken.wait_for(&mut held, poll);
        }
    }
}

/// The smallest a command queue's data ring may be: a power of two large enough to hold a
/// completion record with room to spare. A `QUEUE_CREATE` with a smaller ring gets no per-queue
/// ring.
const MIN_QUEUE_RING_DATA: u64 = 4096;

/// The feature bits whose meaning the host knows, which a `DRIVER_FEATURES` write may take:
/// `RGFX_FEATURE_*`, bits 0 to 8 and 11 to 25.
const KNOWN_FEATURES: u64 = (RGFX_FEATURE_SURFACE_PAGES
    | RGFX_FEATURE_SURFACE_CACHE
    | RGFX_FEATURE_CB_SUBMITTED
    | RGFX_FEATURE_QUIET_SCHEDULED
    | RGFX_FEATURE_CONSUMED_DOORBELL
    | RGFX_FEATURE_QUEUE_BARRIER
    | RGFX_FEATURE_CB_PARTS
    | RGFX_FEATURE_PAGES_BUFFERS
    | RGFX_FEATURE_QUEUE_VECTORS
    | RGFX_FEATURE_SURFACE_ORDER
    | RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS
    | RGFX_FEATURE_UNSHARED_QUEUES
    | RGFX_FEATURE_MANAGED_AT_OFFSET
    | RGFX_FEATURE_HEAP_OVER_PAGES
    | RGFX_FEATURE_SUBMIT_ROOM
    | RGFX_FEATURE_ICB_SLOTS
    | RGFX_FEATURE_COMMIT_ORDER
    | RGFX_FEATURE_SURFACE_DEFERRED
    | RGFX_FEATURE_HOST_TIME
    | RGFX_FEATURE_MANAGED_TEXTURE_SLABS
    | RGFX_FEATURE_MANAGED_BUFFER_SLABS
    | RGFX_FEATURE_SMALL_ID_CHUNKS
    | RGFX_FEATURE_DISPLAY_COLOR
    | RGFX_FEATURE_MESH_PIPELINES) as u64;

/// The `RGFX_QUEUE_*` flags the host carries out, each with the feature the guest's driver must
/// take for a `QUEUE_CREATE` to ask for it: [`RGFX_QUEUE_UNSHARED`] alone, so a create that asks
/// for any other flag is refused. The host hands a queue's flags to [`Host::stage3_message`] in
/// its `QUEUE_CREATE`'s tail.
const QUEUE_FLAGS: &[(u32, u64)] = &[(RGFX_QUEUE_UNSHARED, RGFX_FEATURE_UNSHARED_QUEUES as u64)];

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
/// called on one of the device's threads, or, for [`Host::raise_interrupt`], on the thread that
/// posts the event it raises for; never inside a call of the host's into the device but
/// [`Device::mmio_write_taking`], which calls [`Host::answer_command_buffer`] and may raise there.
/// Each is called with no lock held that a call of the device's waits for, directly or through a
/// thread of the device's, so each may call the device at once. Of its own the device holds only
/// a command queue's claim on its messages, around [`Host::stage3_message`] and
/// [`Host::answer_command_buffer`] of the queue's messages and the raises inside them; a call of
/// the device's only tries it.
pub trait Host: Send + Sync + 'static {
    /// Raise MSI-X vector `vector`, the vector of the completion ring an event was published on
    /// (0 for the control ring's; a queue's, as its `QUEUE_CREATE` gives it), after the event. On
    /// the thread that posts it: the control, refresh or a queue's service thread, one of ring 0's
    /// threads, which serve the lanes (`reims-vgpu.graphics-ring0-*`), the host's inside
    /// [`Device::mmio_write_taking`], or one of the host's that holds a
    /// [`CompletionPoster`] or a [`CompleteAnswer`]. Locks may be held around it, none that a call
    /// of the device's waits for: a queue's claim on its messages, on its service thread and
    /// inside [`Device::mmio_write_taking`], and those the host holds while it posts or answers
    /// (`reims-vgpu`'s decoder, at a command buffer's `CB_SUBMITTED` posted as the buffer is
    /// answered or taken).
    fn raise_interrupt(&self, vector: u32);

    /// Whether the guest wrote any of the `length` bytes at `offset` in the region since the last
    /// time this asked, which starts the next span: the host's dirty log of the range, taken and
    /// cleared. On the refresh thread.
    fn scanout_written(&self, offset: u64, length: u64) -> bool;

    /// Context `context` was closed (its process closed the device or died): the host drops every
    /// object whose id lies in the context's range (`context << RGFX_CONTEXT_OBJ_ID_SHIFT` and up,
    /// below the next context's), and gives `answer` ([`RGFX_STATUS_OK`]) once every command
    /// buffer of the context it had committed has finished, from any thread: the kext then gives
    /// the context's blocks out again. Called on the context's lane, in ring order, after the
    /// context's queues were ended. The default drops nothing and answers at once, for a host with
    /// no stage-3 decoder.
    fn context_closed(&self, context: u32, answer: CompleteAnswer) {
        let _ = context;
        answer.answer(RGFX_STATUS_OK);
    }

    /// `BLOCK_RELEASE`: context `release.context` freed its region block (`release.offset`,
    /// `release.size`), in which no live queue's ring lies. The host gives `answer` from any
    /// thread: [`RGFX_STATUS_REFUSED`] at once where an object it holds lies in the block, else
    /// [`RGFX_STATUS_OK`] once every command buffer of the context it had committed has finished,
    /// and, of a shared block, every one each other holder had committed before its
    /// `BLOCK_UNSHARE`, its close, or this release where neither came first, each of which the
    /// device hands the host as an unshare first ([`Host::block_unshared`]), and once the host's
    /// display no longer reads a surface's texture over the block. The kext gives the block out
    /// again only on OK, and the memory of a surface's block in place back to the guest's
    /// system. Called on the
    /// context's lane, in ring order, after the lanes of the other contexts that held the block,
    /// and the block's own lane, have carried out what they were handed before it.
    /// The default answers OK at once, for a host with no stage-3 decoder.
    fn block_released(&self, release: &RgfxBlockRelease, answer: CompleteAnswer) {
        let _ = release;
        answer.answer(RGFX_STATUS_OK);
    }

    /// `BLOCK_UNSHARE`: context `unshare.context` let go of the shared block of `unshare.size`
    /// bytes at `unshare.offset`, which it no longer names, and the kext unwired the block's pages
    /// in its process. The command buffers of the context's the host had committed before this
    /// may still use the block: the host's answer to the block's `BLOCK_RELEASE`, by whichever
    /// holder, waits for them too ([`Host::block_released`]). Called once the region has ended
    /// the context's hold, and before the block's release: on the context's lane, in ring order,
    /// for its unshare and its close, and on the releasing context's lane for a holder whose
    /// unshare comes after the release, once that holder's lane has carried out what it was
    /// handed before the release. Never twice for one hold: an unshare that comes after the
    /// release or the close is not handed on.
    /// The default does nothing, for a host with no stage-3 decoder.
    fn block_unshared(&self, unshare: &RgfxBlockGrant) {
        let _ = unshare;
    }

    /// The device was reset ([`Device::reset`], or the guest's `STATUS` 0): the host drops every
    /// object of every context, as each context's close does ([`Host::context_closed`]), with no
    /// answer owed, and everything else it keeps for the guest's processes, so that the device is
    /// as one just made (the module doc). Called on the control thread once the reset's queues
    /// have ended and their rings are no longer touched and every ring-0 message a lane began
    /// before the reset has returned, and before any ring-0 message the guest publishes after the
    /// reset is carried out; resets in a row may be told once. A
    /// queue's service thread that has not yet seen its end may still hand the host a message of
    /// the queue's after this: its poster is then [`CompletionPoster::stale`], which the host
    /// checks under the lock this takes, and carries nothing of it out. The default does
    /// nothing, for a host with no stage-3 decoder.
    fn device_reset(&self) {}

    /// The device's feature bits, which `FEATURES_LO` and `FEATURES_HI` read: the
    /// `RGFX_FEATURE_*` the host carries out. The default is none, for a host with no stage-3
    /// decoder.
    fn features(&self) -> u64 {
        0
    }

    /// The bytes of the host GPU's memory the device caches of deferred blocks may hold
    /// ([`RGFX_FEATURE_SURFACE_DEFERRED`]), which `DEFERRED_BUDGET_LO` and `_HI` read: the same
    /// for the device's life, so the kext's one read at its start holds, and 0 where the host
    /// does not offer the feature. The default is 0.
    fn deferred_budget(&self) -> u64 {
        0
    }

    /// `SURFACE_PAGES`: the guest RAM runs of the IOSurface whose shared block `pages` names,
    /// read out of the block, before its first `BLOCK_SHARE`. The host maps them as one view for
    /// the block's `BUFFER_OVER_SURFACE_PAGES` buffers, until the block's `BLOCK_RELEASE`, where
    /// it carries out [`RGFX_FEATURE_SURFACE_PAGES`], and answers [`RGFX_STATUS_OK`]; else
    /// [`RGFX_STATUS_REFUSED`], with no view. `flags` are the message's `RGFX_SURFACE_PAGES_*`
    /// ([`RgfxSurfacePagesCache::flags`], 0 for a message without them):
    /// [`RGFX_SURFACE_PAGES_CACHED`], where a host that carries out
    /// [`RGFX_FEATURE_SURFACE_CACHE`] gives the block a device cache, which its buffers are, and
    /// with it [`RGFX_SURFACE_PAGES_DEFERRED`], no runs and a cache of zeros alone, or
    /// [`RGFX_SURFACE_PAGES_ATTACH`], the runs of such a block's pages
    /// ([`RGFX_FEATURE_SURFACE_DEFERRED`]); and [`RGFX_SURFACE_PAGES_PRIVATE`], where the block is a process's pages block, whose view a
    /// host that carries out [`RGFX_FEATURE_PAGES_BUFFERS`] keeps for its `BUFFER_OVER_PAGES`
    /// buffers alone. The host gives `answer` from any thread once it has carried the message
    /// out, and carries out every later message that names the block as if after it. Called on
    /// the block's lane (on one of ring 0's threads), in ring order, after the messages of
    /// its context's published before it that name the block. The default maps nothing, and
    /// refuses at once.
    fn surface_pages(
        &self,
        pages: &RgfxSurfacePages,
        runs: Vec<RgfxPageRun>,
        flags: u32,
        answer: CompleteAnswer,
    ) {
        let _ = (pages, runs, flags);
        answer.answer(RGFX_STATUS_REFUSED);
    }

    /// `SURFACE_DOWNLOAD`: the device cache of the IOSurface whose block lies at `block_offset`
    /// written into its pages ([`RGFX_FEATURE_SURFACE_CACHE`]); the host gives `answer`
    /// ([`RGFX_STATUS_OK`]) from any thread once the pages hold it. Called on the block's lane, in
    /// ring order, after the block's `SURFACE_PAGES` published before it. The default answers at
    /// once, for a host that keeps no caches.
    fn surface_download(&self, block_offset: u64, answer: CompleteAnswer) {
        let _ = block_offset;
        answer.answer(RGFX_STATUS_OK);
    }

    /// [`Host::surface_download`], offered on the control thread where the block's lane has
    /// nothing left to carry out, so the download keeps its place after the block's
    /// `SURFACE_PAGES`: the host carries it out here, giving `answer` as `surface_download`
    /// does, where it can without waiting (every lock it needs is tried, and it commits no copy
    /// now: nothing to copy, or a copy already on its way), and gives `answer` back, having done
    /// nothing, otherwise. The device then hands the download to the block's lane, so ring 0 goes
    /// on meanwhile. A lock's download of a surface the host already copied is so answered with
    /// no lane thread woken between. The default carries out none.
    ///
    /// # Errors
    /// `answer`, untouched, where the host leaves the download to the block's lane.
    fn try_surface_download(
        &self,
        block_offset: u64,
        answer: CompleteAnswer,
    ) -> Result<(), CompleteAnswer> {
        let _ = block_offset;
        Err(answer)
    }

    /// Hands a stage-3 message to the host's decoder: from ring 0, on its context's lane, an
    /// object-lifecycle create or destroy; for a command queue, on its service thread, its
    /// `QUEUE_CREATE` first and its `OBJECT_DESTROY` last, its `COMMAND_BUFFER`s between them
    /// coming through [`Host::command_buffer`]. Each ring's messages come whole, in the
    /// order the guest published them (ring 0's, each context's in that order, on one thread at a
    /// time), with `completion` to post what the host owes onto that
    /// ring's completion ring (a pipeline's reflection on the control one, a buffer's
    /// `CB_COMPLETE` on the queue's), raising its MSI-X vector; this returns nothing. A ring-0
    /// message may be carried out after this returns (a pipeline compile on the host's own
    /// threads, so ring 0 goes on meanwhile): the host then posts its reply when it is done, from
    /// that thread, and holds a later command buffer that names its object until it is. The
    /// default does nothing, for a host with no stage-3 decoder.
    fn stage3_message(&self, completion: &CompletionPoster, message: &[u8]) {
        let _ = (completion, message);
    }

    /// A `QUEUE_CREATE` made the command queue `completion` posts for: its rings and its service
    /// thread are the device's. Called on the context's lane with no lock held, before the create
    /// is answered, so before the guest's processes know the queue, whose buffers a buffer of
    /// another queue may name from the answer on. The queue's service thread may hand the host the
    /// queue's create, and the buffers published before it, before or after this
    /// ([`Host::stage3_message`]). [`CompletionPoster::published_seqno`] reads what the queue's
    /// submission ring has published, until the queue ends. Not called for a create the device
    /// refuses. The default does nothing.
    fn queue_made(&self, completion: &CompletionPoster) {
        let _ = completion;
    }

    /// Hands a `COMMAND_BUFFER` from a command queue's own submission ring, or from the block a
    /// `COMMAND_BUFFER_IN_REGION` there names, to the host's decoder, on the queue's service
    /// thread, as [`Host::stage3_message`] hands a message: `message` is the copy the device took
    /// of it out of the region (the region's module doc), which the host keeps for as long as it
    /// carries the buffer out (on a thread of its own, after this returns), so it copies the
    /// message no more. `claim` is the queue's claim on its ring's messages, held into the call:
    /// a host that carries out more of the buffer here than its begin, which places it among the
    /// queue's buffers, lets it go first ([`QueueClaim::let_go`]), so the queue's next buffer is
    /// answered meanwhile as it would be with that work on a thread of the host's. The default
    /// hands the message to [`Host::stage3_message`], the claim held.
    fn command_buffer(
        &self,
        completion: &CompletionPoster,
        message: Vec<u8>,
        claim: &mut QueueClaim<'_>,
    ) {
        let _ = claim;
        self.stage3_message(completion, &message);
    }

    /// Offers the host the `COMMAND_BUFFER` `message` at the head of a command queue's
    /// submission ring, on the thread of a `DOORBELL` write through [`Device::mmio_write_taking`]
    /// (the module doc), or on the lane's thread that ends the queue's service thread's wait for
    /// ring 0 with the message taken, with `completion` onto the queue's completion ring. The
    /// queue's service thread hands the message to [`Host::command_buffer`] next, as it hands
    /// any. The host answers it here where it may post the buffer's `CB_SUBMITTED` before then,
    /// and posts it through `completion` waiting for nothing ([`CompletionPoster::try_post`]),
    /// keeping what it took for the buffer for the hand-over; true where it answered it, and
    /// false, with nothing done, otherwise. It waits for nothing (no lock held elsewhere, no room
    /// on a ring) and allocates nothing, on the guest's vCPU. A message it answered is not
    /// offered again; one it did not may be, at a later doorbell. The default answers none.
    fn answer_command_buffer(&self, completion: &CompletionPoster, message: &[u8]) -> bool {
        let _ = (completion, message);
        false
    }

    /// The objects the host names IOSurface `iosurface_id` by: the live textures over its plane 0,
    /// one of which a `PRESENT_SURFACE` of it shows ([`Host::present_surface`]). Asked on the
    /// control thread as it hands a present to the presents' lane, with the lanes' lock held, so
    /// the present waits for the lanes whose messages before it change them (`device::lanes`):
    /// it waits for nothing and calls nothing of the device's. The default names none, for a host
    /// with no stage-3 decoder.
    fn surface_textures(&self, iosurface_id: u32) -> Vec<u64> {
        let _ = iosurface_id;
        Vec::new()
    }

    /// A refresh of display 0, on the refresh thread, after its `VBLANK` (while the guest has
    /// them on): the vblank at which the presents the window has shown since the last one are
    /// first on the display, which the host answers here ([`Host::present_surface`]). The default
    /// does nothing.
    fn refreshed(&self) {}

    /// Shows the IOSurface of global ID `iosurface_id` on display 0 in place of the scanout
    /// (`PRESENT_SURFACE`; 0 shows the scanout again), on the presents' lane, in ring order, once
    /// the lanes have carried out the messages before it that change the textures over the
    /// surface ([`Host::surface_textures`]), and gives `answer`
    /// once the display shows it: [`RGFX_STATUS_OK`], or [`RGFX_STATUS_INVALID_ARGUMENT`], with
    /// nothing changed, where the host holds no texture over that surface. The guest's display
    /// pipe completes its transaction on that answer, so it is given when the surface is shown,
    /// not before. `generation`, where the present carries one, is the CPU's generation of the
    /// surface ([`RgfxPresentSurfaceGen`]): a surface with a device cache shows its pages as of
    /// it. The default shows nothing and answers at once, refusing every surface but 0, for a
    /// host with no stage-3 decoder.
    fn present_surface(&self, iosurface_id: u32, generation: Option<u64>, answer: PresentAnswer) {
        let _ = generation;
        answer.answer(if iosurface_id == 0 {
            RGFX_STATUS_OK
        } else {
            RGFX_STATUS_INVALID_ARGUMENT
        });
    }

    /// A present that keeps what display 0 shows (`PRESENT_SURFACE` with [`RGFX_PRESENT_KEEP`]),
    /// on the presents' lane, in ring order: the colour staged since the last present
    /// ([`Host::display_color`]) applies as the display shows it, and `answer` is given then, as
    /// [`Host::present_surface`] gives its. The default answers [`RGFX_STATUS_OK`] at once.
    fn present_keep(&self, answer: PresentAnswer) {
        answer.answer(RGFX_STATUS_OK);
    }

    /// Part of one colour stage of display 0 (`DISPLAY_COLOR`), its fields and the entries it
    /// carries, on the presents' lane, in ring order with the presents: the host stages it, and
    /// the next present ([`Host::present_surface`], [`Host::present_keep`]) sets the stages staged
    /// whole, as the display shows that present. Not answered. The default drops it, for a host
    /// with no window.
    fn display_color(&self, color: &RgfxDisplayColor, entries: &[RgfxColorEntry]) {
        let _ = (color, entries);
    }
}

/// The largest `COMMAND_BUFFER` message a doorbell's thread offers the host
/// ([`Device::mmio_write_taking`]): the size of the thread's copy of the message, on its stack,
/// and a bound on the host's scans of its op stream there.
const DOORBELL_MESSAGE_MAX: usize = 16 << 10;

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
    /// `DRIVER_FEATURES`: the feature bits the guest's driver took, each one the host offers and
    /// knows ([`KNOWN_FEATURES`]); 0 from a reset to the driver's write.
    driver_features: AtomicU64,
    /// Moved on at each reset: work begun before one drops what it would send.
    epoch: AtomicU64,
    /// A reset the host has not been told of yet ([`Host::device_reset`]): set with the control
    /// submission ring held, so the control thread sees it before any message published after
    /// the reset.
    reset_owed: AtomicBool,
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
    /// Where posts onto the control completion ring wait for room.
    room: Room,
    /// How long a post waits for room before it looks again, in nanoseconds: [`ROOM_POLL`]; the
    /// tests make it long to see what wakes a post.
    room_poll_ns: AtomicU64,
    /// The control submission ring's count through the last message the control thread has
    /// carried out, handed to the host or handed to a lane: a command buffer waits for it to pass
    /// the last message published before it that is its context's or every buffer's
    /// ([`backlog`]), and for its context's lane to carry out its context's ([`lanes`]).
    control_done: AtomicU64,
    /// Where a queue's thread waits for `control_done` and its context's lane, woken only by the
    /// move it waits for, a reset or the end ([`gate`]).
    gate: Gate,
    /// Ring 0's lanes, which carry out each context's object-lifecycle messages, and the
    /// presents, off the control thread ([`lanes`]).
    lanes: Arc<Lanes>,
    /// The command queues' own rings and service threads, by queue id and by ring id.
    queues: Mutex<Queues>,
    /// A weak handle to this shared state, set at the device's make: a queue's service thread and
    /// its completion poster hold it, so neither keeps the device alive.
    weak_self: OnceLock<Weak<Shared>>,
    /// Where the tests stop the control thread once it has found no reset came since it took a
    /// message it carried out or handed on with no ring held, before it moves `control_done` past
    /// it: called with the count it moves to.
    #[cfg(test)]
    before_count: Mutex<Option<Stop>>,
}

/// What [`Shared::before_count`] holds.
#[cfg(test)]
type Stop = Arc<dyn Fn(u64) + Send + Sync>;

/// The command queues' per-queue rings, keyed both by the queue's guest-picked id (for a destroy)
/// and by its ring id — the submission ring's region offset, the `DOORBELL` value (for a wake).
#[derive(Default)]
struct Queues {
    /// By the queue's guest-picked `obj_id`.
    by_obj: HashMap<u64, Arc<QueueShared>>,
    /// By the queue's ring id (its submission ring's region offset).
    by_ring: HashMap<u64, Arc<QueueShared>>,
}

impl Queues {
    /// Forgets `queue` and ends its service.
    fn remove(&mut self, queue: &Arc<QueueShared>) {
        if self
            .by_obj
            .get(&queue.obj_id)
            .is_some_and(|q| Arc::ptr_eq(q, queue))
        {
            self.by_obj.remove(&queue.obj_id);
        }
        if self
            .by_ring
            .get(&queue.ring_id)
            .is_some_and(|q| Arc::ptr_eq(q, queue))
        {
            self.by_ring.remove(&queue.ring_id);
        }
        queue.end();
    }
}

/// One command queue's own rings, shared by its service thread and its completion poster.
pub(crate) struct QueueShared {
    /// The queue's guest-picked id.
    obj_id: u64,
    /// The context (the process) it belongs to: its id's bits from [`RGFX_CONTEXT_OBJ_ID_SHIFT`].
    context: u32,
    /// The `QUEUE_CREATE` that made it, which the host is handed first.
    create: RgfxQueueCreate,
    /// The `RGFX_QUEUE_*` flags its create asked for, all of them ones the host carries out
    /// ([`QUEUE_FLAGS`]): the host is handed them in the create's tail.
    flags: u32,
    /// The reset count its `QUEUE_CREATE` was taken at: a reset since ends it, and its posts are
    /// dropped.
    epoch: u64,
    /// The ring id: the submission ring's region offset, the `DOORBELL` value that wakes it.
    ring_id: u64,
    /// The completion ring's MSI-X vector.
    vector: u32,
    /// The submission ring, held while the service thread takes a message.
    submit: Mutex<Submit>,
    /// The highest `seqno` of a message taken off the submission ring, 0 before one is.
    taken_seqno: AtomicU64,
    /// Held by whoever takes the submission ring's messages or offers them to the host: the
    /// service thread, from its take of a message to its hand-over, or to the host's begin of a
    /// command buffer where the host lets it go there ([`QueueClaim`]), or a doorbell's thread,
    /// which tries it and offers the host a message only where it holds it. It holds the `seqno`
    /// of the message at the ring's head that the host answered on a doorbell's thread
    /// ([`Host::answer_command_buffer`]), until the service thread takes it.
    consuming: Mutex<Option<u64>>,
    /// The completion ring.
    complete: Mutex<Producer>,
    /// Where posts onto the completion ring wait for room.
    room: Room,
    /// The service thread's own wake.
    wake: Wake,
    /// Set when the queue is destroyed, or at a reset or the device's end.
    ending: AtomicBool,
}

impl QueueShared {
    /// The `QUEUE_CREATE` the host is handed: the create as the guest sent it, with the tail of
    /// its flags where it has any, as the kext sends it.
    fn create_message(&self) -> Vec<u8> {
        let mut message = self.create.as_bytes().to_vec();
        if self.flags != 0 {
            let tail = RgfxQueueCreateFlags {
                flags: self.flags,
                reserved_04: 0,
            };
            message.extend_from_slice(tail.as_bytes());
        }
        message
    }

    /// The queue's service is ending (a destroy, a reset, or the device's end).
    fn ended(&self) -> bool {
        self.ending.load(Ordering::Acquire)
    }

    /// Ends the queue's service: its thread stops at its next wake, and a post waiting for room
    /// on its completion ring stops waiting.
    fn end(&self) {
        self.ending.store(true, Ordering::Release);
        self.wake.wake();
        self.room.wake();
    }

    /// A message of `seqno` was taken off its submission ring.
    fn took(&self, seqno: u64) {
        self.taken_seqno.fetch_max(seqno, Ordering::AcqRel);
    }

    /// The highest `seqno` the guest has published on its submission ring: of a message the host
    /// took, or of one it has yet to take; `None` where the queue has ended. Waits for a take of
    /// the ring under way, which waits for nothing.
    fn published_seqno(&self, region: &Region) -> Option<u64> {
        let submit = self.submit.lock();
        if self.ended() {
            return None;
        }
        let waiting = submit.consumer.highest_published_seqno(region);
        Some(
            self.taken_seqno
                .load(Ordering::Acquire)
                .max(waiting.unwrap_or(0)),
        )
    }

    /// Its doorbell was rung: its service thread wakes, and so does a post waiting for room on
    /// its completion ring, as the guest may have taken completions before it rang.
    fn rung(&self) {
        self.wake.wake();
        self.room.wake();
    }

    /// Waits out a take of the submission ring or a post onto the completion ring that was under
    /// way when the queue ended: each checks [`Self::ended`] with its ring held, so none starts
    /// after this, and the queue's rings are not touched again.
    fn fence(&self) {
        debug_assert!(self.ended());
        drop(self.submit.lock());
        drop(self.complete.lock());
    }
}

/// The host's end of the control submission ring.
struct Submit {
    /// The consumer.
    consumer: Consumer,
    /// The ring is broken: nothing is taken until a reset.
    broken: bool,
    /// Ring 0's messages, by whose each is, for the command buffers' waits ([`Backlog`]); empty
    /// on a queue's ring.
    backlog: Backlog,
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
        Device::with_lanes(region, host, screen, Lanes::default())
    }

    /// [`Self::new`], over `lanes`, whose threads wait as long as they keep (the tests; else
    /// [`crate::keep::KEEP`]).
    fn with_lanes(
        region: Arc<Region>,
        host: Box<dyn Host>,
        screen: Box<dyn Screen>,
        lanes: Lanes,
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
            driver_features: AtomicU64::new(0),
            epoch: AtomicU64::new(0),
            reset_owed: AtomicBool::new(false),
            ending: AtomicBool::new(false),
            wake: Wake::new()?,
            doorbell: OnceLock::new(),
            submit: Mutex::new(Submit {
                consumer,
                broken: false,
                backlog: Backlog::with_room(),
            }),
            display: Mutex::new(DisplayState::default()),
            completion: Mutex::new(producer),
            room: Room::default(),
            room_poll_ns: AtomicU64::new(u64::try_from(ROOM_POLL.as_nanos()).unwrap_or(u64::MAX)),
            control_done: AtomicU64::new(0),
            gate: Gate::default(),
            lanes: Arc::new(lanes),
            queues: Mutex::new(Queues::default()),
            weak_self: OnceLock::new(),
            #[cfg(test)]
            before_count: Mutex::new(None),
        });
        // A queue's service thread and completion poster reach the shared state weakly.
        let _ = shared.weak_self.set(Arc::downgrade(&shared));
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

    /// How many command queues have a per-queue ring and service thread: the tests check a
    /// `QUEUE_CREATE` with valid ring parameters made one.
    pub fn queue_count(&self) -> usize {
        self.shared.queues.lock().by_obj.len()
    }

    /// Whether the control thread and every queue's service thread sleep with no wake pending
    /// ([`Wake::parked`]), and no lane has a message to carry out: what they write of the rings
    /// stays as it is until the guest acts.
    #[cfg(test)]
    pub(crate) fn parked(&self) -> bool {
        let doorbell = self.shared.doorbell.get().map(AsRawFd::as_raw_fd);
        self.shared.wake.parked(doorbell)
            && self.shared.lanes.idle()
            && self
                .shared
                .queues
                .lock()
                .by_obj
                .values()
                .all(|queue| queue.wake.parked(None))
    }

    /// A 32-bit read of BAR 0 at `offset`. Offsets with no register read 0, and so do
    /// `HOST_TIME_LO` and `_HI` where the driver did not take `HOST_TIME`.
    pub fn mmio_read(&self, offset: u64) -> u32 {
        let size = self.shared.region.size();
        let host_time = || {
            self.shared.driver_features.load(Ordering::Acquire) & u64::from(RGFX_FEATURE_HOST_TIME)
                != 0
        };
        match u32::try_from(offset).unwrap_or(u32::MAX) {
            RGFX_REG_MAGIC => RGFX_MAGIC,
            RGFX_REG_VERSION => RGFX_VERSION,
            RGFX_REG_FEATURES_LO => (self.shared.host.features() & 0xffff_ffff) as u32,
            RGFX_REG_FEATURES_HI => (self.shared.host.features() >> 32) as u32,
            RGFX_REG_STATUS => self.shared.status.load(Ordering::Acquire),
            RGFX_REG_DRIVER_FEATURES_LO => {
                (self.shared.driver_features.load(Ordering::Acquire) & 0xffff_ffff) as u32
            }
            RGFX_REG_DRIVER_FEATURES_HI => {
                (self.shared.driver_features.load(Ordering::Acquire) >> 32) as u32
            }
            RGFX_REG_REGION_SIZE_LO => (size & 0xffff_ffff) as u32,
            RGFX_REG_REGION_SIZE_HI => (size >> 32) as u32,
            RGFX_REG_DEFERRED_BUDGET_LO => {
                (self.shared.host.deferred_budget() & 0xffff_ffff) as u32
            }
            RGFX_REG_DEFERRED_BUDGET_HI => (self.shared.host.deferred_budget() >> 32) as u32,
            // The clock the refresh thread's grid stamps each VBLANK on.
            RGFX_REG_HOST_TIME_LO if host_time() => (now_ns() & 0xffff_ffff) as u32,
            RGFX_REG_HOST_TIME_HI if host_time() => (now_ns() >> 32) as u32,
            _ => 0,
        }
    }

    /// A 32-bit write of BAR 0 at `offset`: `STATUS` 0 resets the device and 1 starts it taking
    /// commands; `DRIVER_FEATURES_LO` and `_HI` take, before `STATUS` 1, the bits written that the
    /// host offers and knows; `DOORBELL` with a ring's id wakes its consumer. Other values and
    /// offsets (`CONSUMED` among them) are ignored. A reset waits for the message the control
    /// thread is taking from the ring, and no write waits for the host's decoder.
    pub fn mmio_write(&self, offset: u64, value: u32) {
        match u32::try_from(offset).unwrap_or(u32::MAX) {
            RGFX_REG_STATUS if value == RGFX_STATUS_RESET => self.reset(),
            RGFX_REG_DRIVER_FEATURES_LO => self.shared.take_driver_features(0, value),
            RGFX_REG_DRIVER_FEATURES_HI => self.shared.take_driver_features(32, value),
            RGFX_REG_STATUS if value == RGFX_STATUS_DRIVER_READY => {
                self.shared
                    .status
                    .store(RGFX_STATUS_DRIVER_READY, Ordering::Release);
                self.shared.wake.wake();
                for queue in self.shared.queues.lock().by_obj.values() {
                    queue.wake.wake();
                }
            }
            RGFX_REG_DOORBELL => self.shared.ring_doorbell(u64::from(value)),
            _ => {}
        }
    }

    /// [`Self::mmio_write`], but a `DOORBELL` write of a command queue's ring first offers the
    /// queue's next `COMMAND_BUFFER` to the host on the calling thread
    /// ([`Host::answer_command_buffer`]; the module doc): the host may post its `CB_SUBMITTED`
    /// there, calling [`Host::raise_interrupt`] inside this call. It waits for the queues' table
    /// where it is held, as `mmio_write` does, and for nothing else, and the offer allocates
    /// nothing.
    pub fn mmio_write_taking(&self, offset: u64, value: u32) {
        match u32::try_from(offset).unwrap_or(u32::MAX) {
            RGFX_REG_DOORBELL => self.shared.ring_taking(u64::from(value)),
            _ => self.mmio_write(offset, value),
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

/// The `COMPLETE` a ring-0 command owes, which the host gives from any thread once it has carried
/// the command out: a `PRESENT_SURFACE` once the display shows the surface
/// ([`Host::present_surface`]), a `CONTEXT_CLOSE` and a `BLOCK_RELEASE` once the command buffers
/// they wait for have finished ([`Host::context_closed`], [`Host::block_released`]), a
/// `SURFACE_PAGES` once its view is made ([`Host::surface_pages`]). It holds a
/// weak reference to the device; an answer after the device has ended, or for a command taken
/// before a reset, is dropped, and so is one never given.
pub struct CompleteAnswer {
    /// The device's shared state, weakly.
    shared: Weak<Shared>,
    /// The command's `seqno`.
    seqno: u64,
    /// The reset count the command was taken at.
    epoch: u64,
}

/// A `PRESENT_SURFACE`'s answer.
pub type PresentAnswer = CompleteAnswer;

impl std::fmt::Debug for CompleteAnswer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompleteAnswer")
            .field("seqno", &self.seqno)
            .finish_non_exhaustive()
    }
}

/// A command queue's claim on its submission ring's messages, which its service thread holds as
/// it hands a `COMMAND_BUFFER` to the host ([`Host::command_buffer`]). While it is held no
/// doorbell's thread offers the host the queue's next message
/// ([`Device::mmio_write_taking`]), so the next buffer is not answered ahead of the one handed
/// over, which takes its place among the queue's buffers at its begin. A host that carries out
/// more of the buffer on the service thread than its begin (its walk and its commit) lets the
/// claim go once it is begun ([`Self::let_go`]): the queue's next buffer may then be answered
/// as it would be with that work on another thread, at its doorbell or here.
pub struct QueueClaim<'a> {
    /// The device's shared state.
    shared: &'a Shared,
    /// The queue whose messages it claims.
    queue: &'a Arc<QueueShared>,
    /// The queue's `consuming`, until the claim is let go.
    held: Option<MutexGuard<'a, Option<u64>>>,
}

impl<'a> QueueClaim<'a> {
    /// Lets go of the claim, then offers the queue's next message to the host on this thread as a
    /// doorbell's thread does ([`Host::answer_command_buffer`]): a buffer the guest published and
    /// rang for before the claim was let go, and now waits for its `CB_SUBMITTED`, is answered
    /// here where it need wait for nothing, not after the work this thread carries out next. A
    /// doorbell rung afterwards offers its buffer itself. Once let go, it stays so for the call.
    pub fn let_go(&mut self) {
        if self.held.take().is_some() {
            self.shared.answer_on_doorbell(self.queue);
        }
    }

    /// The queue's `consuming` again, taken where the claim was let go.
    fn retake(self) -> MutexGuard<'a, Option<u64>> {
        let QueueClaim { queue, held, .. } = self;
        held.unwrap_or_else(|| queue.consuming.lock())
    }
}

impl CompleteAnswer {
    /// Posts the `COMPLETE` with `status` onto the control completion ring, waiting for room.
    pub fn answer(self, status: u32) {
        if let Some(shared) = self.shared.upgrade() {
            shared.post_waiting(&crate::control::complete(self.seqno, status), self.epoch);
        }
    }
}

/// A handle that posts events from any thread onto the completion ring of the ring a stage-3
/// message came from: a command queue's (the decoder's command-buffer completions for buffers
/// submitted on that queue, which the GPU raises on a thread of its own), or the control one (a
/// pipeline create's reflection). It writes the message and raises the ring's MSI-X vector where
/// the guest armed it, through that ring's `Producer`. It holds weak references to the device
/// and the queue, so it keeps neither alive; a post after the device has ended, after the queue is
/// destroyed, or after a reset that came since the message it posts for was taken (since the
/// queue was made, for a queue's), is dropped.
#[derive(Clone)]
pub struct CompletionPoster {
    /// The device's shared state, weakly.
    shared: Weak<Shared>,
    /// The queue's own rings, weakly, and its id; `None` for the control completion ring.
    queue: Option<(Weak<QueueShared>, u64)>,
    /// The reset count the work it posts for began at.
    epoch: u64,
}

impl CompletionPoster {
    /// The id of the queue whose ring the poster posts on; `None` for the control ring. A message
    /// the host takes off a queue's submission ring is handed with that queue's poster, so this
    /// names the queue (and the context) the message came from, whatever the message says.
    pub fn queue_obj_id(&self) -> Option<u64> {
        self.queue.as_ref().map(|(_, obj_id)| *obj_id)
    }

    /// The highest `seqno` the guest has published on the submission ring of the queue this posts
    /// for, of a message the host has taken or of one it has yet to take: no command buffer of a
    /// greater seqno is on the ring. `None` for the control ring's poster, or where the device or
    /// the queue has ended. Waits for a take of the ring under way, which waits for nothing.
    pub fn published_seqno(&self) -> Option<u64> {
        let shared = self.shared.upgrade()?;
        let (queue, _) = self.queue.as_ref()?;
        queue.upgrade()?.published_seqno(&shared.region)
    }

    /// Whether the device was reset since the work this posts for began (or has ended): what it
    /// would post is dropped, and the host carries out nothing more of that work
    /// ([`Host::device_reset`]).
    pub fn stale(&self) -> bool {
        self.shared
            .upgrade()
            .is_none_or(|shared| shared.epoch.load(Ordering::Acquire) != self.epoch)
    }

    /// Posts `message` (a whole event, its `size` set and a multiple of 8) onto the ring's
    /// completion ring, waiting for room while the device (and the queue) live and the device is
    /// not reset, and raising the ring's MSI-X vector where the guest armed it. Does nothing where
    /// the device has ended or the queue is gone.
    pub fn post(&self, message: &[u8]) {
        let Some(shared) = self.shared.upgrade() else {
            return;
        };
        let epoch = self.epoch;
        match &self.queue {
            None => shared.post_waiting(message, epoch),
            Some((queue, _)) => {
                if let Some(queue) = queue.upgrade() {
                    shared.post_queue_waiting(&queue, message, epoch);
                }
            }
        }
    }

    /// [`Self::post`], waiting for nothing: false, with nothing posted, where another thread is
    /// posting onto the ring or it has no room; true where `message` is posted, or dropped as
    /// `post` drops it.
    pub fn try_post(&self, message: &[u8]) -> bool {
        let Some(shared) = self.shared.upgrade() else {
            return true;
        };
        let epoch = self.epoch;
        let posted = match &self.queue {
            None => shared.post_on(message, epoch, false),
            Some((queue, _)) => match queue.upgrade() {
                Some(queue) => shared.post_queue_on(&queue, message, epoch, false),
                None => Post::Dropped,
            },
        };
        !matches!(posted, Post::NoRoom | Post::Busy)
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
    /// Not posted: another thread held the ring, and the post waits for nothing.
    Busy,
}

impl Shared {
    /// The device's end: every thread stops at its next wake.
    fn end(&self) {
        self.ending.store(true, Ordering::Release);
        self.wake.wake();
        for queue in self.queues.lock().by_obj.values() {
            queue.end();
        }
        self.lanes.end();
        self.gate.every();
        self.room.wake();
    }

    fn ending(&self) -> bool {
        self.ending.load(Ordering::Acquire)
    }

    fn started(&self) -> bool {
        self.status.load(Ordering::Acquire) == RGFX_STATUS_DRIVER_READY
    }

    /// The reset (the module doc). It waits for the message the control thread is taking, and
    /// for a take of a queue's ring or a post onto one under way, none of which waits for the
    /// host; the host is told on the control thread ([`Self::tell_reset`]).
    fn reset(&self) {
        self.status.store(RGFX_STATUS_RESET, Ordering::Release);
        self.driver_features.store(0, Ordering::Release);
        let epoch = self.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        // Nothing the lanes were handed before it is begun now.
        self.lanes.reset(epoch);
        let mut submit = self.submit.lock();
        let mut display = self.display.lock();
        let mut completion = self.completion.lock();
        submit.consumer.reset(&self.region);
        submit.broken = false;
        submit.backlog.clear();
        self.control_done.store(0, Ordering::Release);
        completion.reset(&self.region);
        // The scanout stays (the module doc): the screen is told nothing of it.
        *display = DisplayState {
            scanout: display.scanout,
            ..DisplayState::default()
        };
        // With the ring held: the control thread tells the host before any message after this.
        self.reset_owed.store(true, Ordering::Release);
        drop(completion);
        drop(display);
        drop(submit);
        // The per-queue rings are the guest's to make again after a reset: end their services.
        let ended: Vec<Arc<QueueShared>> = {
            let mut queues = self.queues.lock();
            queues.by_ring.clear();
            queues.by_obj.drain().map(|(_, queue)| queue).collect()
        };
        for queue in &ended {
            queue.end();
        }
        // No ended queue's thread reads or writes its rings past this, and no block is any
        // process's: the guest's kext gives its blocks out again.
        for queue in &ended {
            queue.fence();
        }
        self.region.revoke_all();
        self.gate.every();
        self.room.wake();
        self.wake.wake();
        self.screen.cursor_image(None);
        self.screen.cursor_moved(0, 0, true);
    }

    /// A write of `DRIVER_FEATURES`' half at bit `shift` (0 for `_LO`, 32 for `_HI`): that half
    /// becomes the bits of `value` the host offers ([`Host::features`]) and knows
    /// ([`KNOWN_FEATURES`]), the other half kept. Ignored once the driver is ready: the features
    /// are fixed from `STATUS` 1 to the next reset.
    fn take_driver_features(&self, shift: u32, value: u32) {
        if self.status.load(Ordering::Acquire) == RGFX_STATUS_DRIVER_READY {
            return;
        }
        let half = 0xffff_ffff_u64 << shift;
        let taken = (u64::from(value) << shift) & self.host.features() & KNOWN_FEATURES;
        let _ = self
            .driver_features
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                Some((old & !half) | taken)
            });
    }

    /// The `RGFX_QUEUE_*` flags a `QUEUE_CREATE` may ask for: those the host carries out
    /// ([`QUEUE_FLAGS`]) whose feature the guest's driver took.
    fn queue_flags_allowed(&self) -> u32 {
        let taken = self.driver_features.load(Ordering::Acquire);
        QUEUE_FLAGS
            .iter()
            .filter(|&&(_, feature)| taken & feature != 0)
            .fold(0, |allowed, &(flag, _)| allowed | flag)
    }

    /// Tells the host of a reset it has not been told of ([`Host::device_reset`]), on the control
    /// thread, before the thread takes anything the guest published after it.
    fn tell_reset(&self) {
        if self.reset_owed.swap(false, Ordering::AcqRel) {
            // After every message a lane began before the reset, as the control thread's own
            // message in flight comes before it.
            self.lanes.wait_begun(self.epoch.load(Ordering::Acquire));
            self.host.device_reset();
            log::info!(target: "reims_graphics", "the device was reset");
        }
    }

    /// The control thread (the module doc).
    fn control_thread(&self) {
        while !self.ending() {
            self.tell_reset();
            self.take_messages();
            // Asleep with `consumer_idle` set only while the device takes the ring.
            let (idle, epoch) = {
                let submit = self.submit.lock();
                let epoch = self.epoch.load(Ordering::Acquire);
                if !self.started() || submit.broken {
                    (false, epoch)
                } else if submit.consumer.sleep_or_pending(&self.region) {
                    continue;
                } else {
                    (true, epoch)
                }
            };
            if self.ending() {
                break;
            }
            self.wake.sleep(self.doorbell.get().map(AsRawFd::as_raw_fd));
            // A doorbell may follow the guest's take of completions.
            self.room.wake();
            // A reset while it slept set the ring up anew, its `consumer_idle` 1, so the guest
            // rings for its first message: that stands. The reset moves the count before it takes
            // the ring, so one that has not yet set the ring up sets it after this.
            if idle {
                let submit = self.submit.lock();
                if self.epoch.load(Ordering::Acquire) == epoch {
                    submit.consumer.awake(&self.region);
                }
            }
        }
    }

    /// Takes and answers every message the guest has published, while the device is started and
    /// the ring is not broken. A display command but `PRESENT_SURFACE` is taken and done with the
    /// ring held, so a reset comes before or after it whole. An object-lifecycle message and a
    /// `PRESENT_SURFACE` are taken with it held and carried out with nothing held, so a reset never
    /// waits for the host, which may wait for its decoder there, or answer a present at once. An
    /// answer is posted with nothing of the device's held, so the interrupt is raised with no lock
    /// of the device's held, and dropped where a reset came after its message was taken. A reset
    /// is told to the host before any message published after it is taken ([`Self::tell_reset`]).
    fn take_messages(&self) {
        loop {
            self.tell_reset();
            let (answer, epoch, room) = {
                let mut submit = self.submit.lock();
                if !self.started() || submit.broken || self.ending() {
                    return;
                }
                if self.reset_owed.load(Ordering::Acquire) {
                    continue;
                }
                let epoch = self.epoch.load(Ordering::Acquire);
                let next = submit.consumer.next(&self.region);
                // A guest post waiting for room, told with nothing held: after this take, or
                // after the last, where the ring is empty.
                let room = next.is_ok() && self.room_wanted(&submit.consumer);
                if let Ok(Some(message)) = &next {
                    let through = submit.consumer.consumed();
                    submit.backlog.took(message, through);
                }
                match next {
                    Ok(Some(message)) if control::is_unheld(&message) => {
                        let through = submit.consumer.consumed();
                        drop(submit);
                        if room {
                            self.post_room(epoch);
                        }
                        // A context's object-lifecycle message and a present go to their lane,
                        // which answers them; the rest is carried out here.
                        let answer = match Key::of(&message) {
                            Some(key) => {
                                if let Some(message) = self.download_here(key, message, epoch) {
                                    self.hand(key, message, through, epoch);
                                }
                                None
                            }
                            None => control::unheld(self, &message, epoch),
                        };
                        // A reset meanwhile restarted the count; this message's is moot. The look
                        // and the move are made with the ring held, as a reset's move to 0 is, so
                        // no reset comes between them.
                        let submit = self.submit.lock();
                        if self.epoch.load(Ordering::Acquire) == epoch {
                            #[cfg(test)]
                            {
                                let stop = self.before_count.lock().clone();
                                if let Some(stop) = stop {
                                    stop(through);
                                }
                            }
                            self.control_done.store(through, Ordering::Release);
                            self.gate.counted(through);
                        }
                        drop(submit);
                        (answer, epoch, false)
                    }
                    Ok(Some(message)) => {
                        let answer = control::answer(self, &message);
                        let done = submit.consumer.consumed();
                        self.control_done.store(done, Ordering::Release);
                        self.gate.counted(done);
                        (answer, epoch, room)
                    }
                    Ok(None) => {
                        drop(submit);
                        if room {
                            self.post_room(epoch);
                        }
                        return;
                    }
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
            if room {
                self.post_room(epoch);
            }
            if let Some(answer) = answer {
                self.post_waiting(&answer, epoch);
            }
        }
    }

    /// Whether every ring-0 message of context `context`'s through count `through` is carried out:
    /// taken by the control thread, and carried out by it or by the context's lane.
    fn carried_out(&self, context: u32, through: u64) -> bool {
        self.control_done.load(Ordering::Acquire) >= through && self.lanes.through(context, through)
    }

    /// A `SURFACE_DOWNLOAD` for the lane `key` of a block with nothing left to carry out on it,
    /// taken at reset count `epoch`, offered to the host here, on the control thread
    /// ([`Host::try_surface_download`]): `None` where the host carried it out, else `message`
    /// back, for the lane. The control thread alone hands the block's lane its jobs, so it stays
    /// with none until this returns, and every message of the block's before the download is
    /// carried out, as on the lane; nothing waits on the lanes for a download.
    fn download_here(&self, key: Key, message: Message, epoch: u64) -> Option<Message> {
        if message.header.r#type != RGFX_CMD_SURFACE_DOWNLOAD || !self.lanes.lane_idle(key) {
            return Some(message);
        }
        let Ok((download, _)) = RgfxSurfaceDownload::read_from_prefix(&message.bytes) else {
            return Some(message);
        };
        let answer = self.complete_answer(message.header.seqno, epoch);
        match self
            .host
            .try_surface_download(download.block_offset, answer)
        {
            Ok(()) => None,
            Err(_) => Some(message),
        }
    }

    /// Hands `message`, taken at reset count `epoch`, ring 0's count through it `through`, to the
    /// lane `key` ([`lanes`]), starting one of ring 0's threads for the lane where it has none and
    /// none waits for a lane. Where no thread can be started, the lane's jobs are carried out
    /// here.
    fn hand(&self, key: Key, message: Message, through: u64, epoch: u64) {
        let hand = self.lanes.hand(key, message, through, epoch, |surface| {
            self.host.surface_textures(surface)
        });
        let Hand::Start(index) = hand else {
            return;
        };
        let lanes = Arc::clone(&self.lanes);
        let shared = self.weak_self.get().cloned().unwrap_or_default();
        let started = std::thread::Builder::new()
            .name(lanes::thread_name(index))
            .spawn(move || serve_ring0(&lanes, &shared, index, key));
        if let Err(e) = started {
            log::warn!(
                target: "reims_graphics",
                "{} could not be started ({e}): the messages of {key:?}'s lane are carried out on \
                 the control thread",
                lanes::thread_name(index)
            );
            self.lanes.unstarted(index, key);
            let shared = self.weak_self.get().cloned().unwrap_or_default();
            serve_lane(&self.lanes, &shared, key, None);
        }
    }

    /// After a take of a submission ring: whether the guest took `SUBMIT_ROOM` and a producer
    /// waits for room on it ([`Consumer::room_wanted`], which clears the flag).
    fn room_wanted(&self, consumer: &Consumer) -> bool {
        self.driver_features.load(Ordering::Acquire) & u64::from(RGFX_FEATURE_SUBMIT_ROOM) != 0
            && consumer.room_wanted(&self.region)
    }

    /// Posts ring 0's `SUBMIT_ROOM` record on the control completion ring, for work begun at reset
    /// count `epoch`: the guest's posts waiting for room on ring 0 look again.
    fn post_room(&self, epoch: u64) {
        log::debug!(target: "reims_graphics::room", "SUBMIT_ROOM posted for the control ring");
        self.post_waiting(&submit_room(), epoch);
    }

    /// Posts `message`, waiting for room ([`Room`]) while the device is not reset or ended.
    fn post_waiting(&self, message: &[u8], epoch: u64) {
        loop {
            let seen = self.room.wakes();
            if self.post(message, epoch) != Post::NoRoom || self.ending() {
                return;
            }
            self.room.wait(seen, self.room_poll());
        }
    }

    /// How long a post waits for room before it looks again ([`ROOM_POLL`]).
    fn room_poll(&self) -> Duration {
        Duration::from_nanos(self.room_poll_ns.load(Ordering::Relaxed))
    }

    /// Posts `message` into the completion ring, for work begun at reset count `epoch`, and
    /// raises the ring's interrupt where the guest armed it, with no lock held.
    fn post(&self, message: &[u8], epoch: u64) -> Post {
        self.post_on(message, epoch, true)
    }

    /// [`Self::post`], which waits for the ring held elsewhere where `wait`, and else is
    /// [`Post::Busy`].
    fn post_on(&self, message: &[u8], epoch: u64, wait: bool) -> Post {
        let raise = {
            let Some(mut completion) = lock_ring(&self.completion, wait) else {
                return Post::Busy;
            };
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

    /// `CONTEXT_CLOSE` of `context` (the module doc), on its lane: its queues end, then
    /// the host drops its objects and gives `answer` once its command buffers have finished.
    pub(crate) fn close_context(&self, context: u32, answer: CompleteAnswer) {
        let ended: Vec<Arc<QueueShared>> = {
            let mut queues = self.queues.lock();
            let ended: Vec<Arc<QueueShared>> = queues
                .by_obj
                .values()
                .filter(|q| q.context == context)
                .cloned()
                .collect();
            for queue in &ended {
                queues.remove(queue);
            }
            ended
        };
        // No ended queue's thread reads or writes its rings past this, so the kext may give
        // their blocks out again once the close is answered.
        for queue in &ended {
            queue.fence();
        }
        // Their threads waiting for ring 0 look again, and go.
        if !ended.is_empty() {
            self.gate.every();
        }
        // Every shared block it held is let go of as by its BLOCK_UNSHARE, so the block's
        // release, by whichever holder is last, waits for the context's command buffers as well:
        // the kext may let go of the context's wire of the block's pages before this close is
        // answered (a close it took for refused), and the other holders' wires go at that
        // release's answer. Those it alone held too: another holder's BLOCK_SHARE runs on that
        // holder's lane, which the close does not wait for, so it may come after this.
        for (offset, size) in self.region.revoke_context(context) {
            self.host.block_unshared(&RgfxBlockGrant {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_BLOCK_UNSHARE,
                    size: size_of::<RgfxBlockGrant>() as u32,
                    seqno: 0,
                },
                context,
                offset,
                size,
                ..RgfxBlockGrant::default()
            });
        }
        self.host.context_closed(context, answer);
        log::info!(target: "reims_graphics", "context {context} closed");
    }

    /// `BLOCK_GRANT`: the region keeps the block as its context's, where it lies in the
    /// processes' part of the region (past the device info and the scanout, before the cursor),
    /// or in the scanout's span, which the kext grants to the context that draws the display into
    /// it.
    pub(crate) fn grant_block(&self, grant: &RgfxBlockGrant) {
        let end = self.region.size().saturating_sub(RGFX_CURSOR_SPAN);
        let grant_end = grant.offset.checked_add(grant.size);
        let within =
            |from: u64, to: u64| grant.offset >= from && grant_end.is_some_and(|e| e <= to);
        if !within(RGFX_CONTEXT_ALLOCATIONS_OFFSET, end)
            && !within(RGFX_ALLOCATIONS_OFFSET, RGFX_CONTEXT_ALLOCATIONS_OFFSET)
        {
            log::warn!(
                target: "reims_graphics",
                "a BLOCK_GRANT of {:#x} bytes at {:#x} lies outside the processes' blocks and the \
                 scanout: ignored",
                grant.size, grant.offset
            );
            return;
        }
        self.region.grant(grant.context, grant.offset, grant.size);
    }

    /// `BLOCK_SHARE` and `BLOCK_UNSHARE`: the region keeps, or ends, the context's hold of the
    /// shared block, where it lies in the processes' part of the region; the host is told of an
    /// unshare ([`Host::block_unshared`]).
    pub(crate) fn share_block(&self, share: &RgfxBlockGrant, holds: bool) {
        let end = self.region.size().saturating_sub(RGFX_CURSOR_SPAN);
        if share.offset < RGFX_CONTEXT_ALLOCATIONS_OFFSET
            || share.offset.checked_add(share.size).is_none_or(|e| e > end)
        {
            log::warn!(
                target: "reims_graphics",
                "a {} of {:#x} bytes at {:#x} lies outside the processes' blocks: ignored",
                if holds { "BLOCK_SHARE" } else { "BLOCK_UNSHARE" },
                share.size, share.offset
            );
            return;
        }
        if holds {
            self.region.share(share.context, share.offset, share.size);
        } else if self.region.unshare(share.context, share.offset, share.size) {
            // Only where it ended a hold: one the block's release or the context's close ended
            // first let the host wait for the context's buffers there already, and the block's
            // offset may be another block's by now.
            self.host.block_unshared(share);
        }
    }

    /// `SURFACE_PAGES` (`seqno`, taken at reset count `epoch`): its runs read out of its block
    /// for the host ([`Host::surface_pages`]), which answers it, where the block lies in the
    /// processes' blocks and its runs, whole, in the block, and `None`; else logged and
    /// [`RGFX_STATUS_INVALID_ARGUMENT`], with no view. `flags`: the message's
    /// `RGFX_SURFACE_PAGES_*`, 0 for one without them.
    pub(crate) fn surface_pages(
        &self,
        pages: &RgfxSurfacePages,
        flags: u32,
        seqno: u64,
        epoch: u64,
    ) -> Option<u32> {
        let end = self.region.size().saturating_sub(RGFX_CURSOR_SPAN);
        let block_end = pages.block_offset.checked_add(pages.block_size);
        let runs_bytes = u64::from(pages.run_count) * size_of::<RgfxPageRun>() as u64;
        let runs_end = pages.runs_offset.checked_add(runs_bytes);
        let fits = pages.block_offset >= RGFX_CONTEXT_ALLOCATIONS_OFFSET
            && block_end.is_some_and(|e| e <= end)
            && pages.runs_offset >= pages.block_offset
            && pages.runs_offset.is_multiple_of(8)
            && runs_end.zip(block_end).is_some_and(|(r, b)| r <= b);
        let runs = if fits {
            usize::try_from(runs_bytes)
                .ok()
                .and_then(|len| self.region.read(pages.runs_offset, len))
        } else {
            None
        };
        let Some(bytes) = runs else {
            log::warn!(
                target: "reims_graphics",
                "a SURFACE_PAGES of {} runs at {:#x} for the block of {:#x} bytes at {:#x} lies \
                 outside it: the block gets no view of its surface's pages",
                pages.run_count, pages.runs_offset, pages.block_size, pages.block_offset
            );
            return Some(RGFX_STATUS_INVALID_ARGUMENT);
        };
        let runs = bytes
            .as_chunks::<{ size_of::<RgfxPageRun>() }>()
            .0
            .iter()
            .filter_map(|run| RgfxPageRun::read_from_bytes(run).ok())
            .collect();
        (self.host).surface_pages(pages, runs, flags, self.complete_answer(seqno, epoch));
        None
    }

    /// Whether a live queue's submission or completion ring lies in the `size` bytes at `offset`
    /// (a `BLOCK_RELEASE` of a block that holds one is refused).
    pub(crate) fn rings_in(&self, offset: u64, size: u64) -> bool {
        let end = offset.saturating_add(size);
        let meets = |at: u64, data: u32| {
            let ring_end = at.saturating_add(u64::from(RGFX_RING_HEADER_SIZE) + u64::from(data));
            at < end && offset < ring_end
        };
        self.queues.lock().by_obj.values().any(|q| {
            meets(q.create.submit_offset, q.create.submit_data_size)
                || meets(q.create.complete_offset, q.create.complete_data_size)
        })
    }

    /// A completion poster onto the control completion ring, for the replies of a ring-0 message
    /// taken at reset count `epoch`.
    pub(crate) fn control_poster(&self, epoch: u64) -> CompletionPoster {
        CompletionPoster {
            shared: self.weak_self.get().cloned().unwrap_or_else(Weak::new),
            queue: None,
            epoch,
        }
    }

    /// A doorbell write of ring id `ring_id`: wakes the control thread (ring id 0) or the service
    /// thread of the queue whose submission ring is at that region offset; an unknown id is
    /// ignored.
    fn ring_doorbell(&self, ring_id: u64) {
        if ring_id == u64::from(RGFX_CONTROL_RING_ID) {
            self.wake.wake();
            self.room.wake();
            return;
        }
        if let Some(queue) = self.queues.lock().by_ring.get(&ring_id) {
            queue.rung();
        }
    }

    /// [`Self::ring_doorbell`], where a command queue's ring is rung, after its next message is
    /// offered to the host on this thread ([`Self::answer_on_doorbell`]). The queues' table, held
    /// elsewhere, is waited for as `ring_doorbell` waits for it, and nothing is offered.
    fn ring_taking(&self, ring_id: u64) {
        if ring_id == u64::from(RGFX_CONTROL_RING_ID) {
            self.wake.wake();
            self.room.wake();
            return;
        }
        let queue = match self.queues.try_lock() {
            Some(queues) => queues.by_ring.get(&ring_id).cloned(),
            None => return self.ring_doorbell(ring_id),
        };
        if let Some(queue) = queue {
            self.answer_on_doorbell(&queue);
            queue.rung();
        }
    }

    /// Offers `queue`'s next message to the host on this thread, a doorbell's (the module doc),
    /// where it is a `COMMAND_BUFFER` of at most [`DOORBELL_MESSAGE_MAX`] bytes, none answered is
    /// left for the service thread to take, the device is started and the queue lives, and nothing
    /// need be waited for: `queue`'s `consuming`, its submission ring and the control submission
    /// ring are each free, at most
    /// [`DOORBELL_READS`] of ring 0's messages are left to read, which ring 0's index has room for,
    /// and the control thread has carried out every ring-0 message the buffer waits for, as the
    /// service thread waits for before it hands a buffer over. The message stays on the ring (but
    /// for pads before it, which carry nothing), copied onto this thread's stack for the host;
    /// where the host answers it, the service thread, which takes it next, hands it over waiting
    /// for ring 0 no more. Allocates nothing.
    fn answer_on_doorbell(&self, queue: &Arc<QueueShared>) {
        let Some(mut consuming) = queue.consuming.try_lock() else {
            return;
        };
        if consuming.is_some() {
            return;
        }
        let Some(mut submit) = queue.submit.try_lock() else {
            return;
        };
        if !self.started() || submit.broken || self.ending() || queue.ended() {
            return;
        }
        let mut words = [0u64; DOORBELL_MESSAGE_MAX / 8];
        let bytes = words.as_mut_bytes();
        let Ok(Some(header)) = submit.consumer.peek_into(&self.region, bytes) else {
            return;
        };
        let size = header.size as usize;
        if header.r#type != RGFX_CMD_COMMAND_BUFFER || size > bytes.len() {
            return;
        }
        // The order the service thread keeps (`take_queue_messages`), read after the peek, of as
        // much of ring 0 as a doorbell reads.
        {
            let Some(mut control) = self.submit.try_lock() else {
                return;
            };
            let control = &mut *control;
            if !control.backlog.room_for(DOORBELL_READS) {
                return;
            }
            let done = self.control_done.load(Ordering::Acquire);
            let read = (control.backlog).read(
                &control.consumer,
                &self.region,
                |context, through| {
                    done >= through && self.lanes.try_through(context, through) == Some(true)
                },
                DOORBELL_READS,
            );
            if !read
                || done < control.backlog.waits_for(queue.context)
                || (self.lanes).try_through(queue.context, control.backlog.own(queue.context))
                    != Some(true)
            {
                return;
            }
        }
        // The host may raise inside its answer, which waits for no lock a call of the device's
        // waits for (a reset fences the ring): the queue's claim alone is held there, as around
        // the service thread's hand-overs.
        drop(submit);
        let poster = self.queue_poster(queue);
        if self.host.answer_command_buffer(&poster, &bytes[..size]) {
            *consuming = Some(header.seqno);
        }
    }

    /// A `QUEUE_CREATE`'s ring pair, where its parameters are valid: each data ring a power of two
    /// no smaller than [`MIN_QUEUE_RING_DATA`], each ring 8-aligned in the guest's allocations and
    /// within the region, the two not overlapping, and each whole in one block the queue's context
    /// holds ([`Region::held_by`]), as the kext checked: the host posts its completions there and
    /// reads its counts, which must be the process's own memory, not another's or the scanout's.
    /// `None` where any is refused, so the queue is refused.
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
        let context = (q.obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT) as u32;
        if !self.region.held_by(context, q.submit_offset, submit_span)
            || !self
                .region
                .held_by(context, q.complete_offset, complete_span)
        {
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

    /// `QUEUE_CREATE` (the module doc), taken at reset count `epoch` with its tail's `flags` (0
    /// where it has none): makes a command queue's own submission+completion ring pair and starts
    /// its service thread, where the flags are ones the host carries out for the features the
    /// driver took and the ring parameters are valid; the status its `COMPLETE` carries. A queue
    /// made is told to the host before this returns ([`Host::queue_made`]), which is all of the
    /// host's it waits for; it waits for no thread of the device's.
    pub(crate) fn create_queue(&self, q: &RgfxQueueCreate, flags: u32, epoch: u64) -> u32 {
        let refused_flags = flags & !self.queue_flags_allowed();
        if refused_flags != 0 {
            log::warn!(
                target: "reims_graphics",
                "queue {:#x}: it asks for flags {refused_flags:#x}, which the host does not \
                 carry out or whose feature the driver did not take, so it is refused", q.obj_id
            );
            return RGFX_STATUS_INVALID_ARGUMENT;
        }
        let Some((submit_at, complete_at)) = self.validate_rings(q) else {
            log::warn!(
                target: "reims_graphics",
                "queue {:#x}: its ring parameters are refused (submit {:#x}/{}, complete \
                 {:#x}/{}; a ring must lie whole in one block of its process's)",
                q.obj_id, q.submit_offset, q.submit_data_size, q.complete_offset,
                q.complete_data_size
            );
            return RGFX_STATUS_INVALID_ARGUMENT;
        };
        let Some(me) = self.weak_self.get().and_then(Weak::upgrade) else {
            return RGFX_STATUS_INVALID_ARGUMENT;
        };
        let Some(wake) = Wake::new() else {
            log::error!(
                target: "reims_graphics",
                "queue {:#x}: no wake pipe; it is refused", q.obj_id
            );
            return RGFX_STATUS_INVALID_ARGUMENT;
        };
        // Take the ring headers as the guest left them: it zeroes both before the create, and may
        // publish its first command buffer on the submission ring before the host gets here (the
        // create rides ring 0, the buffer the queue's own). A reset here would clear that producer
        // count and lose the buffer, and the guest, seeing `consumer_idle` 0, rang no doorbell for
        // it. The service thread takes what is published before it first sleeps, and sets
        // `consumer_idle` itself when it does.
        let consumer = Consumer::new(submit_at);
        let producer = Producer::new(complete_at);
        let queue = Arc::new(QueueShared {
            obj_id: q.obj_id,
            context: (q.obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT) as u32,
            create: *q,
            flags,
            epoch,
            ring_id: q.submit_offset,
            vector: q.vector,
            submit: Mutex::new(Submit {
                consumer,
                broken: false,
                backlog: Backlog::default(),
            }),
            taken_seqno: AtomicU64::new(0),
            consuming: Mutex::new(None),
            complete: Mutex::new(producer),
            room: Room::default(),
            wake,
            ending: AtomicBool::new(false),
        });
        let ended: Vec<Arc<QueueShared>> = {
            let mut queues = self.queues.lock();
            // A reset since the create was taken ended every queue: this one is not made.
            if self.epoch.load(Ordering::Acquire) != epoch {
                return RGFX_STATUS_REFUSED;
            }
            // A re-open under the same queue id or ring offset ends the old one first.
            let ended: Vec<Arc<QueueShared>> = [
                queues.by_obj.get(&q.obj_id).cloned(),
                queues.by_ring.get(&q.submit_offset).cloned(),
            ]
            .into_iter()
            .flatten()
            .collect();
            for old in &ended {
                queues.remove(old);
            }
            queues.by_obj.insert(q.obj_id, Arc::clone(&queue));
            queues.by_ring.insert(q.submit_offset, Arc::clone(&queue));
            ended
        };
        for old in &ended {
            old.fence();
        }
        if !ended.is_empty() {
            self.gate.every();
        }
        let service = Arc::clone(&queue);
        let started = std::thread::Builder::new()
            .name(format!("reims-vgpu.graphics-queue-{:#x}", q.obj_id))
            .spawn(move || me.queue_service(&service));
        if started.is_err() {
            self.queues.lock().remove(&queue);
            log::error!(
                target: "reims_graphics",
                "queue {:#x}: no service thread; it is refused", q.obj_id
            );
            return RGFX_STATUS_INVALID_ARGUMENT;
        }
        // Before the answer: a buffer of another queue may name this one's from then on.
        self.host.queue_made(&self.queue_poster(&queue));
        RGFX_STATUS_OK
    }

    /// Ends and forgets the queue `obj_id`'s per-queue service, where it has one (an
    /// `OBJECT_DESTROY` of it); false where `obj_id` is no queue's.
    pub(crate) fn destroy_queue(&self, obj_id: u64) -> bool {
        let queue = {
            let mut queues = self.queues.lock();
            let Some(queue) = queues.by_obj.get(&obj_id).cloned() else {
                return false;
            };
            queues.remove(&queue);
            queue
        };
        // Its rings are no longer a live queue's (`rings_in`): nothing touches them past this.
        queue.fence();
        // Its thread, where it waits for ring 0, looks again and goes.
        self.gate.every();
        true
    }

    /// The answer ring-0 command `seqno`, taken at reset count `epoch`, owes ([`CompleteAnswer`]).
    pub(crate) fn complete_answer(&self, seqno: u64, epoch: u64) -> CompleteAnswer {
        CompleteAnswer {
            shared: self.weak_self.get().cloned().unwrap_or_else(Weak::new),
            seqno,
            epoch,
        }
    }

    /// A completion poster onto `queue`'s completion ring, held by its service thread (the module
    /// doc and [`CompletionPoster`]).
    fn queue_poster(&self, queue: &Arc<QueueShared>) -> CompletionPoster {
        CompletionPoster {
            shared: self.weak_self.get().cloned().unwrap_or_else(Weak::new),
            queue: Some((Arc::downgrade(queue), queue.obj_id)),
            epoch: queue.epoch,
        }
    }

    /// A queue's service thread (the module doc): it takes the queue's submission ring with the
    /// control thread's `consumer_idle` handshake, hands each message to the host, and sleeps on
    /// the queue's own wake, woken by its doorbell or its end.
    fn queue_service(&self, queue: &Arc<QueueShared>) {
        let poster = self.queue_poster(queue);
        // The host makes the queue's handle first, so a command buffer on it can name it.
        self.host.stage3_message(&poster, &queue.create_message());
        self.queue_loop(queue, &poster);
        // A closed queue's handle goes with it (a process's other objects go at its context's
        // close). Nothing is posted for it: the queue has ended. A reset's queue's goes with the
        // host's reset ([`Host::device_reset`]).
        if !self.ending() && self.epoch.load(Ordering::Acquire) == queue.epoch {
            let destroy = RgfxObjectDestroy {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_OBJECT_DESTROY,
                    size: size_of::<RgfxObjectDestroy>() as u32,
                    seqno: 0,
                },
                obj_id: queue.obj_id,
            };
            self.host.stage3_message(&poster, destroy.as_bytes());
        }
    }

    /// The loop of a queue's service thread: take, then sleep, until the queue or the device ends.
    fn queue_loop(&self, queue: &Arc<QueueShared>, poster: &CompletionPoster) {
        while !self.ending() && !queue.ended() {
            self.take_queue_messages(queue, poster);
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
                // An ended queue's rings may be another process's block by now (`fence`).
                let submit = queue.submit.lock();
                if !queue.ended() {
                    submit.consumer.awake(&self.region);
                }
            }
        }
    }

    /// Takes and hands to the host every message the guest has published on `queue`'s submission
    /// ring, while the device is started and the ring is not broken. Each message is taken with
    /// the ring held and handed over with only the queue's `consuming` held, which the host may
    /// let go once a `COMMAND_BUFFER` is begun ([`QueueClaim`]); the completion the host owes is
    /// posted onto the queue's completion ring through `poster`, from the GPU's thread.
    fn take_queue_messages(&self, queue: &Arc<QueueShared>, poster: &CompletionPoster) {
        let mut consuming = queue.consuming.lock();
        loop {
            let (next, room) = {
                let mut submit = queue.submit.lock();
                if !self.started() || submit.broken || self.ending() || queue.ended() {
                    return;
                }
                let next = submit.consumer.next(&self.region);
                // A commit waiting for room: after this take, a doorbell's thread's before it, or
                // the last, where the ring is empty.
                let room = next.is_ok() && self.room_wanted(&submit.consumer);
                if let Ok(Some(message)) = &next {
                    queue.took(message.header.seqno);
                }
                if let Err(broken) = &next {
                    log::error!(
                        target: "reims_graphics",
                        "queue {:#x}'s submission ring is broken ({broken}): it takes nothing \
                         more until the device is reset", queue.obj_id
                    );
                    submit.broken = true;
                }
                (next, room)
            };
            // Told with nothing held, on the queue's completion ring.
            if room {
                log::debug!(
                    target: "reims_graphics::room",
                    "SUBMIT_ROOM posted for queue {:#x}'s submission ring", queue.obj_id
                );
                self.post_queue_waiting(queue, &submit_room(), queue.epoch);
            }
            let message = match next {
                Ok(Some(message))
                    if message.header.r#type == RGFX_CMD_COMMAND_BUFFER
                        || message.header.r#type == RGFX_CMD_COMMAND_BUFFER_IN_REGION =>
                {
                    message
                }
                Ok(Some(message)) => {
                    // A queue's ring carries only its process's command buffers; every other
                    // message rides ring 0, through the kext, which checks that its sender owns
                    // the object it names. Refused here, and the ring goes on.
                    log::warn!(
                        target: "reims_graphics",
                        "queue {:#x}'s submission ring carried a message of type {:#x} \
                         (seqno {}), which it may not carry: refused",
                        queue.obj_id, message.header.r#type, message.header.seqno
                    );
                    continue;
                }
                Ok(None) | Err(_) => return,
            };
            // Order this command buffer after the creates its process posted before it. Creates
            // ride ring 0, whose sole consumer is the control thread; command buffers ride the
            // queues' own rings (this thread). A create a buffer uses is posted before it in guest
            // order, so it is within ring 0's producer at the moment this thread takes the buffer.
            // Wait for the control thread to have carried out or handed to a lane every message
            // up to the last of them, of the buffer's context's or every buffer's ([`Backlog`]),
            // and for the context's lane to have carried out the context's ([`lanes`]), so the
            // resource is registered before the buffer's ops name it. This thread only reads ring
            // 0; it never consumes it, so it keeps one consumer and its `consumer_idle` handshake
            // is untouched. It sleeps at the gate, woken once `control_done` reaches what it
            // waits for, then once its context's lane has carried out its own, or at a reset, the
            // queue's end or the device's ([`gate`]). A buffer the host answered on a doorbell's
            // thread waits no more: that thread found ring 0 carried out through what the buffer
            // waits for.
            let answered = consuming.take() == Some(message.header.seqno);
            let (waits_for, own) = if answered {
                (0, 0)
            } else {
                let mut control = self.submit.lock();
                let control = &mut *control;
                (control.backlog).read(
                    &control.consumer,
                    &self.region,
                    |context, through| self.carried_out(context, through),
                    usize::MAX,
                );
                let backlog = &control.backlog;
                (backlog.waits_for(queue.context), backlog.own(queue.context))
            };
            // A buffer that waits files itself at the gate with its message, which the lane whose
            // move ends the wait offers to the host before it wakes this thread
            // ([`Self::answer_offered`]), as a doorbell's thread offers one that need wait for
            // nothing; then this thread hands it over as any, and the host takes the answer
            // there in place of its own.
            let mut message = message;
            let waits = self.control_done.load(Ordering::Acquire) < waits_for
                || !self.lanes.through(queue.context, own);
            let offer = (waits && message.header.r#type == RGFX_CMD_COMMAND_BUFFER).then(|| {
                let bytes = std::mem::take(&mut message.bytes);
                Arc::new(Offer::new(Arc::clone(queue), bytes, waits_for, own))
            });
            self.gate.wait_offering(
                || {
                    if self.ending()
                        || queue.ended()
                        || self.epoch.load(Ordering::Acquire) != queue.epoch
                    {
                        Look::Done
                    } else if self.control_done.load(Ordering::Acquire) < waits_for {
                        Look::Count(waits_for)
                    } else if !self.lanes.through(queue.context, own) {
                        Look::Lane(queue.context, own)
                    } else {
                        Look::Done
                    }
                },
                offer.as_ref(),
            );
            if let Some(offer) = &offer {
                message.bytes = offer.withdraw();
            }
            // A reset since the queue was made ended it: nothing more of it goes to the host.
            if self.epoch.load(Ordering::Acquire) != queue.epoch {
                return;
            }
            // A buffer too large for the ring: its message, from the block, with nothing held,
            // once the block's grant (posted before the buffer) has been carried out.
            let message = if message.header.r#type == RGFX_CMD_COMMAND_BUFFER_IN_REGION {
                match self.command_buffer_in_region(queue, &message) {
                    Some(message) => message,
                    None => {
                        log::warn!(
                            target: "reims_graphics",
                            "queue {:#x}'s submission ring carried a COMMAND_BUFFER_IN_REGION \
                             (seqno {}) that names no whole COMMAND_BUFFER in a block of its \
                             process's: refused",
                            queue.obj_id, message.header.seqno
                        );
                        continue;
                    }
                }
            } else {
                message
            };
            // The host may let the claim go once the buffer is begun ([`QueueClaim`]); a buffer
            // a doorbell's thread answered meanwhile is the ring's next, which this takes next.
            let mut claim = QueueClaim {
                shared: self,
                queue,
                held: Some(consuming),
            };
            self.host.command_buffer(poster, message.bytes, &mut claim);
            consuming = claim.retake();
        }
    }

    /// Offers the host, on the lane's thread whose move ended its queue's thread's wait at the
    /// gate, the `COMMAND_BUFFER` the waiting thread holds ([`Offer`]), before that thread wakes:
    /// where ring 0 is carried out as far as the buffer waits for and the device and its queue go
    /// on, the host may post its `CB_SUBMITTED` here ([`Host::answer_command_buffer`]), a
    /// thread's wake sooner than its queue's thread would at its take. Whether it was offered:
    /// not where the thread still waits for more than this lane's move.
    fn answer_offered(&self, offer: &Offer) -> bool {
        let queue = &offer.queue;
        if self.control_done.load(Ordering::Acquire) < offer.waits_for
            || !self.lanes.through(queue.context, offer.own)
        {
            return false;
        }
        if self.started()
            && !self.ending()
            && !queue.ended()
            && self.epoch.load(Ordering::Acquire) == queue.epoch
        {
            let poster = self.queue_poster(queue);
            self.host.answer_command_buffer(&poster, &offer.message());
        }
        true
    }

    /// The `COMMAND_BUFFER` message a `COMMAND_BUFFER_IN_REGION` `reference` on `queue`'s ring
    /// names, copied out of the block once, as a ring's message is, with the reference's seqno;
    /// `None` where the span lies in no block of the queue's process ([`Region::held_by`]) or
    /// holds no whole `COMMAND_BUFFER` of its length.
    fn command_buffer_in_region(
        &self,
        queue: &QueueShared,
        reference: &Message,
    ) -> Option<Message> {
        let (r, _) = RgfxCommandBufferInRegion::read_from_prefix(&reference.bytes).ok()?;
        if !r.region_offset.is_multiple_of(8)
            || !r.byte_length.is_multiple_of(8)
            || r.byte_length < size_of::<RgfxCommandBuffer>() as u64
            || r.byte_length > u64::from(u32::MAX)
        {
            return None;
        }
        let mut bytes = self.region.read_held(
            queue.context,
            r.region_offset,
            usize::try_from(r.byte_length).ok()?,
        )?;
        let (mut header, _) = RgfxMsgHeader::read_from_prefix(&bytes).ok()?;
        if header.r#type != RGFX_CMD_COMMAND_BUFFER || u64::from(header.size) != r.byte_length {
            return None;
        }
        header.seqno = reference.header.seqno;
        bytes[..size_of::<RgfxMsgHeader>()].copy_from_slice(header.as_bytes());
        Some(Message { header, bytes })
    }

    /// Posts `message` onto `queue`'s completion ring, waiting for room ([`Room`]) while the
    /// device is not reset or ended and the queue lives.
    fn post_queue_waiting(&self, queue: &QueueShared, message: &[u8], epoch: u64) {
        loop {
            let seen = queue.room.wakes();
            if self.post_queue(queue, message, epoch) != Post::NoRoom
                || self.ending()
                || queue.ended()
            {
                return;
            }
            queue.room.wait(seen, self.room_poll());
        }
    }

    /// Posts `message` into `queue`'s completion ring, for work begun at reset count `epoch`, and
    /// raises the queue's MSI-X vector where the guest armed it, with no lock held.
    fn post_queue(&self, queue: &QueueShared, message: &[u8], epoch: u64) -> Post {
        self.post_queue_on(queue, message, epoch, true)
    }

    /// [`Self::post_queue`], which waits for the ring held elsewhere where `wait`, and else is
    /// [`Post::Busy`].
    fn post_queue_on(&self, queue: &QueueShared, message: &[u8], epoch: u64, wait: bool) -> Post {
        let raise = {
            let Some(mut completion) = lock_ring(&queue.complete, wait) else {
                return Post::Busy;
            };
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

    /// The refresh thread (the module doc): a refresh at each tick of its grid
    /// ([`RefreshGrid`]), stamped with the tick's time however late the thread woke for it.
    fn refresh_thread(&self) {
        let mut grid = RefreshGrid::new(now_ns());
        while !self.ending() {
            let now = now_ns();
            let tick = grid.next(self.screen.mode().period_ns(), now);
            if tick > now {
                std::thread::sleep(Duration::from_nanos(tick - now));
            }
            if self.ending() {
                break;
            }
            self.refresh(tick);
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
        self.host.refreshed();
    }
}

/// The times of a display's refreshes on the monotonic clock: each a period of the display's mode
/// after the one before, as a panel's vblanks come, however late the refresh thread wakes for one.
/// Where the next is past already (the thread was held over a period), the grid skips to its last
/// tick not after now, so it keeps its phase and takes no refreshes in a burst.
pub(crate) struct RefreshGrid {
    /// The last tick's time.
    last: u64,
}

impl RefreshGrid {
    /// A grid whose ticks follow `start`.
    pub(crate) fn new(start: u64) -> Self {
        Self { last: start }
    }

    /// The next tick's time, `period` after the last, or the grid's last tick not after `now`
    /// where that one is past.
    pub(crate) fn next(&mut self, period: u64, now: u64) -> u64 {
        let mut tick = self.last.saturating_add(period);
        if tick <= now && period > 0 {
            tick += (now - tick) / period * period;
        }
        self.last = tick;
        tick
    }
}

/// A completion ring locked: waiting for it where `wait`, else only where no other thread holds
/// it.
/// Ring 0's thread `index` ([`lanes`], "Threads"): serves the lane `first`, then each lane it is
/// given, until it has waited [`crate::keep::KEEP`] for one, or the device ends.
fn serve_ring0(lanes: &Lanes, shared: &Weak<Shared>, index: usize, first: Key) {
    let mut key = first;
    loop {
        serve_lane(lanes, shared, key, Some(index));
        match lanes.park(index) {
            Some(next) => key = next,
            None => return,
        }
    }
}

/// Serves the lane `key` ([`lanes`]): carries out each of its jobs in turn, with no lock of the
/// device's held, as the control thread carries out an object-lifecycle message, and posts its
/// answer, until the lane has no job left or the device ends; on ring 0's thread `spare`, which
/// then waits for another lane, or, where `None`, on the control thread. A job taken before a
/// reset is not carried out.
fn serve_lane(lanes: &Lanes, shared: &Weak<Shared>, key: Key, spare: Option<usize>) {
    while let Some(job) = lanes.next(key, spare) {
        let Some(shared) = shared.upgrade() else {
            lanes.finished(key);
            return;
        };
        let answer = if shared.epoch.load(Ordering::Acquire) == job.epoch {
            control::unheld(&shared, &job.message, job.epoch)
        } else {
            None
        };
        if let Some((context, done)) = lanes.finished(key) {
            (shared.gate).carried_out_offering(context, done, |offer| shared.answer_offered(offer));
        }
        if let Some(answer) = answer {
            shared.post_waiting(&answer, job.epoch);
        }
    }
}

fn lock_ring(ring: &Mutex<Producer>, wait: bool) -> Option<MutexGuard<'_, Producer>> {
    if wait {
        Some(ring.lock())
    } else {
        ring.try_lock()
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

/// A `SUBMIT_ROOM` record: its header alone, `seqno` 0.
fn submit_room() -> Vec<u8> {
    sized(&event_header(RGFX_EVT_SUBMIT_ROOM))
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
