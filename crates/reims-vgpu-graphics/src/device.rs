//! [`Device`]: the device's host side, as the host's process runs it. It owns the region (BAR 2),
//! answers the registers (BAR 0), takes the guest's control commands on a thread of its own,
//! carries each process's out on a thread of the process's, and sends each display's events on
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
//! `reims-vgpu.graphics-refresh` runs at the display's refresh rate ([`Screen::mode`]): at each
//! refresh it has the screen show the scanout again while the guest has one
//! ([`Screen::present`]), sends `VBLANK` while the guest has it on, stamped with the refresh's
//! time on its grid (`RefreshGrid`) as a panel's hardware stamps its vblank (dropping one the
//! completion ring has no room for), sends
//! `DISPLAY_CHANGED` once the display's mode is not the one the guest was last told, and then
//! tells the host of the refresh ([`Host::refreshed`]), which answers there the presents the
//! display first shows at it. A device of several displays has a refresh thread for each, on a
//! grid of its own at its own screen's rate (`reims-vgpu.graphics-refresh-<i>` for display `i`
//! from 1), whose events name the display, and go only to a guest that took
//! `RGFX_FEATURE_MULTI_DISPLAY` (`crate::display`'s module doc, "Several displays").
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
//! it only once they are done, whatever other processes' messages (`gate`), but where its
//! context's lane holds a `BUFFER_AT_OFFSET` or an `OBJECT_DESTROY` of no queue it waits for that
//! no thread of the lane's carries out yet, the creates and destroys the bundle posts ahead of
//! each commit: it carries that out itself, as the lane's job, in place of sleeping while a
//! ring-0 thread is woken for it (`lanes`, "A command buffer's thread"). An
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
//! for the message the control thread is taking, which never waits for the host. A completion
//! record a thread that may not raise hands over ([`CompletionPoster::post_elsewhere`]), and one a
//! thread that may not wait finds no room for, or finds records handed over before
//! ([`CompletionPoster::post_unwaiting`]), is posted, and raised for, by its queue's service thread,
//! or by the next thread of the device's that posts onto that ring, which posts it first. A
//! command's answer the host gives off the control thread ([`CompleteAnswer::answer`]) waits for no
//! room either: one the control completion ring has no room for, or that finds answers handed over
//! before it, is posted by the control thread, ahead of its own posts.
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
//! thread does, so a buffer published meanwhile is answered before that walk, not after it.
//! Until it takes the hold back the thread marks the ring walking (`RGFX_CONSUMER_WALKING`): the
//! guest rings for a buffer it publishes during the walk that asks for its `CB_SUBMITTED`, and the
//! doorbell's write answers it there; it rings for nothing else, which the thread takes after the
//! walk as it would with no doorbell.
//!
//! **Routed doorbells.** Where the guest's driver took `RGFX_FEATURE_DOORBELL_ASKS`, it rings a
//! queue's doorbell for a buffer that asks for its `CB_SUBMITTED` with the ring's id and
//! `RGFX_DOORBELL_ASKS`, and for anything else with the id alone, and only the asking write is
//! offered to the host. Each queue then has a bell, an eventfd its service thread sleeps on
//! beside its wake, and the host is asked to route the ring's plain doorbell into it
//! ([`Host::route_doorbell`]; QEMU: a KVM `ioeventfd` matching the ring's id), so a plain
//! doorbell wakes the queue's thread with no exit of the vCPU to the host's process and nothing
//! of the device's run on it. The asking write matches no route and comes through
//! [`Device::mmio_write_taking`] as every doorbell did. The writes a wake coalesces cost nothing:
//! a woken thread takes every message the guest published. At most [`MAX_DOORBELL_ROUTES`]
//! queues are routed at once; a queue past them, a route the host refuses, and every write
//! before the host takes the route, keep the device's write, as before.
//!
//! **Doorbell pages.** Where the host maps BAR 4 and the guest's driver took
//! `RGFX_FEATURE_DOORBELL_PAGES`, the kext binds a doorbell page to each context it opens
//! (`DOORBELL_PAGE_BIND`, a register write before the process learns of the page) and maps the
//! page into the context's process, which writes its rings' ids there as it would hand them to
//! the kext's doorbell call. A page's write ([`Device::doorbell_write`]) is that `DOORBELL` write
//! through [`Device::mmio_write_taking`] for a ring of the bound context's queues, and is dropped
//! and counted for any other (ring 0's, another context's, an unbound page's): the device trusts
//! the page's address for the context, never the value written. A queue whose context has a
//! page is routed there ([`DoorbellAt::Page`]), where the host routes pages; otherwise on
//! `DOORBELL`, as before. A page bound to another context reaches that context's rings from the
//! bind on, and the routes on it of its old context's queues go, their doorbells left to
//! `DOORBELL`: the host wakes no ring at a page but its context's. A reset unbinds every page.
//!
//! The host's route functions may wait for the host's own lock (QEMU's BQL), so the device calls
//! them holding no lock of its own, and no function a host calls under that lock waits for a
//! thread inside one. The calls, and what their thread holds:
//! - a `QUEUE_CREATE` routes the new queue, and drops the route of a queue it remakes or of one
//!   whose service thread could not start; an `OBJECT_DESTROY` of a queue and a `CONTEXT_CLOSE`
//!   drop their queues' routes. Each runs on the context's lane, on one of ring 0's threads or
//!   the control thread ([`lanes`]), which holds no lock of the device's as it carries out a
//!   lane's job, and the call comes after the queues' table and the ended queues' rings are let
//!   go;
//! - a reset (`STATUS` 0 through [`Device::mmio_write`] or [`Device::mmio_write_taking`], or
//!   [`Device::reset`]) drops every ended queue's route on the host's thread that resets, after
//!   the reset let go of ring 0, the display, the completion ring, the queues' table and each
//!   ended queue's rings. QEMU's registers run with no BQL (a lockless region), and its reset
//!   holds it already, which its route function sees (`bql_locked`);
//! - a `DOORBELL_PAGE_BIND` write that binds a page to another context (through
//!   [`Device::mmio_write`] or [`Device::mmio_write_taking`]) drops the routes on the page of its
//!   old context's queues on the host's thread that writes it, after the queues' table is let go.
//!
//! The device's functions a host calls under its lock wait at most for ring 0's message the
//! control thread holds (a display message, which waits for no thread of the device's), for a
//! take of a queue's ring or a post onto one under way, and, in a reset, for the queues' table:
//! none of these is held across a route call, and none waits for a lane. The release of the
//! process's last device waits for the device's threads, a route call among them, at most 5 s
//! (docs/api.md, "The process's exit"). Tested by
//! `the_host_s_lock_held_around_its_calls_while_a_route_waits_for_it_holds_up_nothing`.
//!
//! A `DOORBELL` write of ring 0 through [`Device::mmio_write_taking`] answers ring 0's next
//! message on the writing thread where it is a `SURFACE_DOWNLOAD` the host has carried out
//! already ([`Host::surface_downloaded`]: a CPU lock of a surface whose pages hold its device
//! cache, a decoder's output whose early copy has landed, or a cache nothing wrote), and the
//! control thread sleeps with every answer of what it took posted: the doorbell's thread takes
//! the download, posts its `COMPLETE` and raises the control vector, so the lock's answer no
//! longer waits for the control thread's wake. In every other case, and where any lock it tries
//! is held, the control thread takes the message as it takes any. A host whose ring-0 doorbell
//! is a file the control thread polls (QEMU's ioeventfd, [`Device::set_doorbell`]) never calls
//! the write for it, and so never answers there.
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
//! a lane carries out returns, the refresh thread within a refresh, or within
//! [`REFRESH_SLEEP_PART`] where a refresh is longer), and the last of them to let go drops its
//! reference to the region and the host's callbacks.
//!
//! **Reset.** Writing `STATUS` 0, and the host's reset, put the device back as it was made, as a
//! device's function reset does: no command is taken until the guest writes `STATUS` 1, the
//! guest's driver has taken no feature (`DRIVER_FEATURES` reads 0), both rings are as nothing
//! had passed through them (the submission ring's `consumer_idle` 1, so the guest rings for its
//! first message), each display keeps its scanout (below) but has no cursor image (and shows its
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
//! firmware's framebuffer, which the firmware's scanout, kept through both resets, shows. The
//! display's panel is on after a reset, whatever `SET_DISPLAY_POWER` last set: the screen is told
//! where it was off.
//!
//! **A broken ring.** A count or a message size the protocol refuses (`ring::Broken`) is logged,
//! and the device takes nothing more from the ring until it is reset; the guest's driver sees its
//! commands go unanswered. A completion ring whose guest count is out of reach gets no event
//! until the reset.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

use parking_lot::{Condvar, Mutex, MutexGuard};
use zerocopy::{FromBytes, IntoBytes};

use crate::clock::now_ns;
use crate::control;
use crate::device::backlog::Backlog;
use crate::device::desktop::{Bounds, Place};
use crate::device::gate::{Gate, Look, Offer, Waited};
use crate::device::lanes::{Hand, Key, Lanes};
use crate::display::{Mode, Scanout, Screen, offered_in_room, scanout_room, scanout_room_at};
use crate::ops::RgfxCommandBuffer;
use crate::region::Region;
use crate::ring::{Consumer, Message, Posted, Producer, RingAt, sized};
use crate::wake::{self, Doorbell, RawDoorbell, Wake};
use crate::wire::*;

mod backlog;
mod desktop;
mod gate;
mod guest_times;
mod lanes;

/// How many of ring 0's messages a doorbell's take reads ([`Backlog::read`]) before it leaves the
/// buffer to its queue's thread.
const DOORBELL_READS: usize = 32;

/// The most command queues whose plain doorbells the device routes to the host at once
/// ([`Host::route_doorbell`]); a queue made past them keeps its doorbell on the device's write
/// (on the guest's vCPU), and is counted. KVM keeps at most 1000 devices on a VM's MMIO bus, every
/// `ioeventfd` of every device among them (each queue's route is one), and QEMU ends the VM where
/// KVM refuses one, so the device takes a share of them that leaves the rest of the VM's devices
/// theirs.
pub const MAX_DOORBELL_ROUTES: u32 = 128;

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

/// The longest the refresh thread sleeps at once: a longer wait for its next tick is slept in
/// parts, so the thread, which holds the region and the host's callbacks, sees the device's end
/// within this of it whatever the mode's refresh (the tests' is once in 1000 s). A refresh at
/// 20 Hz or more is one sleep, as before. Windows charges the region to the host's commit limit
/// whole for as long as it lives (`region.rs`).
const REFRESH_SLEEP_PART: Duration = Duration::from_millis(50);

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
/// `RGFX_FEATURE_*`, bits 0 to 8 and 11 to 50.
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
    | RGFX_FEATURE_MESH_PIPELINES
    | RGFX_FEATURE_SHARED_EVENT_NAMES
    | RGFX_FEATURE_OPENCL_MODE
    | RGFX_FEATURE_PRIVATE_RENDER_STATE
    | RGFX_FEATURE_SURFACE_CACHE_REFUSED
    | RGFX_FEATURE_SURFACE_DEFERRED_ASK) as u64
    | RGFX_FEATURE_SURFACE_DEFERRED_CARRY
    | RGFX_FEATURE_SCHEDULED_ASK
    | RGFX_FEATURE_PAGES_16K
    | RGFX_FEATURE_DISPLAY_SCALE
    | RGFX_FEATURE_TEXTURE_DID_MODIFY_DATA
    | RGFX_FEATURE_DISPLAY_POWER
    | RGFX_FEATURE_RATE_MAPS
    | RGFX_FEATURE_FUNCTION_POINTERS
    | RGFX_FEATURE_DYNAMIC_LIBRARIES
    | RGFX_FEATURE_PIPELINE_KEYS
    | RGFX_FEATURE_ICB_SLOTS_IN_PAGES
    | RGFX_FEATURE_DOORBELL_ASKS
    | RGFX_FEATURE_GUEST_TIMES
    | RGFX_FEATURE_READ_ONLY_PAGES
    | RGFX_FEATURE_DOORBELL_PAGES
    | RGFX_FEATURE_PIPELINE_ANSWERS
    | RGFX_FEATURE_MULTI_DISPLAY
    | RGFX_FEATURE_POINTER_OVER
    | RGFX_FEATURE_DISPLAY_BOUNDS;

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
/// [`Device::mmio_write_taking`], which calls [`Host::answer_command_buffer`] and
/// [`Host::surface_downloaded`] and may raise there. Each is called with no lock held that a call
/// of the device's waits for, directly or through a thread of the device's, so each may call the
/// device at once, but [`Host::surface_downloaded`], which waits for nothing and calls nothing
/// of the device's: it is asked with the control rings' locks held (a reset waits for the
/// submission ring's). Of its own the device holds otherwise only a command queue's claim on its
/// messages, around [`Host::stage3_message`] and [`Host::answer_command_buffer`] of the queue's
/// messages and the raises inside them; a call of the device's only tries it.
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

    /// The feature bits the host offers only to a guest whose driver took
    /// [`RGFX_FEATURE_PAGES_16K`], which [`Host::features`] then offers: from the
    /// `DRIVER_FEATURES` write that takes it to the next reset, `FEATURES_LO` and `_HI` read them
    /// beside [`Host::features`], and a write may take them. A host whose page is 16 KiB offers
    /// its features over the guest's RAM so, as the guest's runs are whole pages of its own only
    /// where they are whole 16 KiB pages. The default is none.
    fn features_with_16k_pages(&self) -> u64 {
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
    /// [`RGFX_FEATURE_SURFACE_CACHE`] gives the block a device cache, which its buffers are,
    /// placed by the surface's `layout` where the message carries one, and
    /// with it [`RGFX_SURFACE_PAGES_DEFERRED`], no runs and a cache of zeros alone, or
    /// [`RGFX_SURFACE_PAGES_ATTACH`], the runs of such a block's pages
    /// ([`RGFX_FEATURE_SURFACE_DEFERRED`]); [`RGFX_SURFACE_PAGES_PRIVATE`], where the block is a process's pages block, whose view a
    /// host that carries out [`RGFX_FEATURE_PAGES_BUFFERS`] keeps for its `BUFFER_OVER_PAGES`
    /// buffers alone; and [`RGFX_SURFACE_PAGES_ASK_CACHE`], with the surface's `layout`, where a
    /// host that carries out [`RGFX_FEATURE_SURFACE_CACHE_REFUSED`] answers
    /// [`RGFX_STATUS_CACHE`] for a surface whose pages its textures cannot use in place. The host
    /// gives `answer` from any thread once it has carried the message
    /// out, and carries out every later message that names the block as if after it. Called on
    /// the block's lane (on one of ring 0's threads), in ring order, after the messages of
    /// its context's published before it that name the block. The default maps nothing, and
    /// refuses at once.
    fn surface_pages(
        &self,
        pages: &RgfxSurfacePages,
        runs: Vec<RgfxPageRun>,
        flags: u32,
        layout: Option<RgfxSurfacePagesLayout>,
        answer: CompleteAnswer,
    ) {
        let _ = (pages, runs, flags, layout);
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

    /// A `SURFACE_DOWNLOAD` of the block at `block_offset`, at ring 0's head, offered on the
    /// thread of a `DOORBELL` write of ring 0 through [`Device::mmio_write_taking`] (the guest's
    /// vCPU), where the device would hand it to [`Host::try_surface_download`] next: the control
    /// thread sleeps with every message before it carried out, and the block's lane has nothing
    /// left. Whether the surface's pages hold its device cache already, so the download is
    /// carried out with nothing to copy and nothing to wait for; where so, the host does here
    /// what that download does, and the device answers it ([`RGFX_STATUS_OK`]) on this thread.
    /// Where not, nothing is done, and the control thread takes the download as it takes any. It
    /// waits for nothing (each lock tried) and allocates and frees nothing. The default carries
    /// out none.
    fn surface_downloaded(&self, block_offset: u64) -> bool {
        let _ = block_offset;
        false
    }

    /// Hands a stage-3 message to the host's decoder: from ring 0, on its context's lane, an
    /// object-lifecycle create or destroy (on one of ring 0's threads, or, for a
    /// `BUFFER_AT_OFFSET` or an `OBJECT_DESTROY` of no queue, on the service thread of a queue of
    /// the context whose command buffer waits for it, `lanes`); for a command queue, on its
    /// service thread, its
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

    /// Routes a command queue's plain doorbell into `bell`, a file its service thread sleeps on
    /// (an eventfd): from then on the host takes a write of `ring_id` (the queue's submission
    /// ring's region offset) at `at` where no thread of its runs the write, and makes `bell`
    /// readable, which wakes the thread as the write through [`Device::mmio_write_taking`] or
    /// [`Device::doorbell_write`] would (QEMU: a KVM `ioeventfd` that matches the value at the
    /// address). `at` is `DOORBELL` ([`DoorbellAt::Register`]), or, for a queue whose context has a
    /// doorbell page ([`RGFX_FEATURE_DOORBELL_PAGES`]), that page's first byte in BAR 4
    /// ([`DoorbellAt::Page`]), which only the host's [`Host::routes_doorbell_pages`] is asked for.
    /// A write of `ring_id` with [`RGFX_DOORBELL_ASKS`], for a buffer the host may answer at its
    /// doorbell, matches no route and reaches the device's write as every write did. `route`
    /// names the route, a number the device gives no other: [`Host::unroute_doorbell`] drops it,
    /// and a route of a `ring_id` at `at` another route of the host's holds replaces that one.
    /// The host takes a copy of `bell` and may take the route after the call: until then, and
    /// where it refuses it, the writes reach the device's write. Whether it took the route, so
    /// the device counts it among its [`MAX_DOORBELL_ROUTES`]; false where it refuses it. Asked
    /// only where
    /// the guest's driver took [`RGFX_FEATURE_DOORBELL_ASKS`], on the context's lane (one of
    /// ring 0's threads, or the control thread) as the queue is made, after its service thread
    /// started and before the create is answered, with no lock of the device's held: it may wait
    /// for a lock the host holds while it calls the device (the module doc, "Routed doorbells").
    /// The default takes none.
    fn route_doorbell(&self, route: u64, at: DoorbellAt, ring_id: u32, bell: RawDoorbell) -> bool {
        let _ = (route, at, ring_id, bell);
        false
    }

    /// Whether the host routes doorbells on the doorbell pages ([`Host::route_doorbell`] with
    /// [`DoorbellAt::Page`]): where it does not, a queue whose context has a page gets no route,
    /// and its plain doorbells reach [`Device::doorbell_write`]. The default is false.
    fn routes_doorbell_pages(&self) -> bool {
        false
    }

    /// Drops the route [`Host::route_doorbell`] took as `route`, of ring `ring_id` at `at`, once
    /// its queue has ended, or, at a doorbell page, once the page is bound to another context:
    /// once this returns no write reaches the file, as the ring may be a later queue's, and the
    /// host closes its copy. On the thread that ends the queue, with no lock of the device's held:
    /// the context's lane (an `OBJECT_DESTROY` of the queue, its context's `CONTEXT_CLOSE`, or a
    /// `QUEUE_CREATE` that remakes it), where it may wait for a lock the host holds while it calls
    /// the device, or the host's own inside [`Device::mmio_write`], [`Device::mmio_write_taking`]
    /// (a `STATUS` 0 write, or a `DOORBELL_PAGE_BIND` write that binds a page to another context)
    /// or [`Device::reset`], where it must not wait for a lock the host holds around that call.
    /// The routes the device holds at its end
    /// are the host's to drop: no call drops them. The default does nothing.
    fn unroute_doorbell(&self, route: u64, at: DoorbellAt, ring_id: u32) {
        let _ = (route, at, ring_id);
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

    /// Hands a `SCHEDULED_ASK` from a command queue's own submission ring to the host, on the
    /// queue's service thread, where the guest took `SCHEDULED_ASK`: a thread of the queue's
    /// process waits for the queue's command buffer the ask names, which the service thread
    /// handed over before it ([`Host::command_buffer`]), to be scheduled. The host posts the
    /// buffer's `CB_SCHEDULED` through `completion` where the buffer may be scheduled now, and
    /// otherwise changes nothing; the device posts nothing for the ask. It waits for nothing of
    /// ring 0's: the buffer it names was begun after what it waited for. The default does
    /// nothing.
    fn scheduled_ask(&self, completion: &CompletionPoster, message: &[u8]) {
        let _ = (completion, message);
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

    /// A refresh of display `display`, on its refresh thread, after its `VBLANK` (while the guest
    /// has them on): the vblank at which the presents the window has shown on it since the last
    /// one are first on the display, which the host answers here ([`Host::present_surface`]). The
    /// default does nothing.
    fn refreshed(&self, display: u32) {
        let _ = display;
    }

    /// Shows the IOSurface of global ID `iosurface_id` on display `display` (one the guest sees,
    /// below the device's count) in place of its scanout
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
    fn present_surface(
        &self,
        display: u32,
        iosurface_id: u32,
        generation: Option<u64>,
        answer: PresentAnswer,
    ) {
        let _ = (display, generation);
        answer.answer(if iosurface_id == 0 {
            RGFX_STATUS_OK
        } else {
            RGFX_STATUS_INVALID_ARGUMENT
        });
    }

    /// A present that keeps what display `display` shows (`PRESENT_SURFACE` with
    /// [`RGFX_PRESENT_KEEP`]), on the presents' lane, in ring order: the colour staged for it since
    /// its last present ([`Host::display_color`]) applies as the display shows it, and `answer` is
    /// given then, as [`Host::present_surface`] gives its. The default answers [`RGFX_STATUS_OK`]
    /// at once.
    fn present_keep(&self, display: u32, answer: PresentAnswer) {
        let _ = display;
        answer.answer(RGFX_STATUS_OK);
    }

    /// Part of one colour stage of display `color.display` (`DISPLAY_COLOR`, of a display the
    /// guest sees), its fields and the entries it carries, on the presents' lane, in ring order
    /// with the presents: the host stages it, and that display's next present
    /// ([`Host::present_surface`], [`Host::present_keep`]) sets the stages staged whole, as the
    /// display shows that present. Not answered. The default drops it, for a host with no window.
    fn display_color(&self, color: &RgfxDisplayColor, entries: &[RgfxColorEntry]) {
        let _ = (color, entries);
    }

    /// One event of the device's timeline for the host's profiling trace, `kind` and its `values`,
    /// timed by the host as it is called, on the thread it happened on: a diagnostic, which
    /// changes nothing the device does. Ring 0's events and their values:
    /// - `r0wake`: the control thread woke from its sleep (a doorbell, a `STATUS` write or the
    ///   device's end);
    /// - `r0take type seqno obj_id`: the control thread took a message (`obj_id` the 8 bytes after
    ///   its header, 0 for a message shorter);
    /// - `r0job type seqno`, `r0jobdone seqno`: a lane's job began and was carried out;
    /// - `r0post type seqno raise`: an answer or event went on the control completion ring,
    ///   `raise` 1 where its interrupt is raised next.
    ///
    /// And one of each command buffer a queue's thread takes, where the guest stamps them
    /// ([`RGFX_FEATURE_GUEST_TIMES`]), on that thread, as it hands the buffer to the host:
    /// - `cbguest queue seqno encode post_wait queued transport bound held`: the buffer's spans
    ///   before its take, in nanoseconds ([`guest_times`]), `queued`, `transport` and the clocks'
    ///   `bound` only where the guest holds the offset of its clock from the host's, and `held`,
    ///   from its take to the event.
    ///
    /// The default writes nothing.
    fn trace(&self, kind: &'static str, values: &[u64]) {
        let _ = (kind, values);
    }

    /// The device met `kind` on a completion ring ([`RingDegradation`]): the host counts it with
    /// the process's degradations. Called on the thread that posted, with no lock of the device's
    /// held, after the device logged the kind's line. The default does nothing.
    fn ring_degraded(&self, kind: RingDegradation) {
        let _ = kind;
    }
}

/// Where the guest writes a command queue's doorbell, which the host routes there
/// ([`Host::route_doorbell`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoorbellAt {
    /// `DOORBELL` in BAR 0 ([`RGFX_REG_DOORBELL`]).
    Register,
    /// The first byte of doorbell page `n` in BAR 4 ([`RGFX_PCI_BAR_DOORBELLS`]), at
    /// `n * RGFX_DOORBELL_PAGE_SIZE`: the page bound to the queue's context.
    Page(u32),
}

/// What the device tells its host of its completion rings ([`Host::ring_degraded`]). Every
/// process's ring-0 answers, the `PRESENT_SURFACE` answers and the display's events share the
/// control completion ring, which the guest's kext drains on its work loop; a command queue's
/// ring takes its command buffers' records, which the guest's process drains on its completion
/// thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RingDegradation {
    /// A post found the ring with no room where the post before it on the ring found room: a
    /// spell begins in which the control thread takes no ring-0 command until its answer is
    /// posted (or queued behind the answers another thread is posting), the other posts that
    /// wait for room wait, the answers given elsewhere wait for the control thread to post them
    /// ([`CompleteAnswer::answer`]), and a `VBLANK` is dropped.
    ControlRingFull,
    /// A post that waits for room found none at its first look, and went on once the guest took
    /// more, the device was reset or it ended.
    ControlPostWaited,
    /// A `VBLANK` found no room and was dropped: the guest gets no event for that refresh, and the
    /// next `VBLANK` carries the count.
    VblankDropped,
    /// A post that waits for room found a command queue's completion ring with none where the
    /// post that waited on it before had gone on: a spell begins in which the queue's posts that
    /// wait for room wait for the guest's process to take more.
    QueueRingFull,
    /// A post onto a command queue's completion ring that waits for room found none at its first
    /// look, and went on once the guest took more, the queue ended, the device was reset or it
    /// ended.
    QueuePostWaited,
}

impl RingDegradation {
    /// The kind's code, which the process's degradation counts name.
    pub const fn code(self) -> &'static str {
        match self {
            Self::ControlRingFull => "control-ring-full",
            Self::ControlPostWaited => "control-ring-post-waited",
            Self::VblankDropped => "vblank-dropped",
            Self::QueueRingFull => "queue-ring-full",
            Self::QueuePostWaited => "queue-ring-post-waited",
        }
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
    /// The displays, display `i` at index `i`: 1 to [`RGFX_DISPLAYS_MAX`], fixed at the make.
    pub(crate) displays: Box<[Display]>,
    /// BAR 2, which what shows the displays may hold too.
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
    doorbell: OnceLock<Doorbell>,
    /// The control submission ring, held while the control thread takes a message.
    submit: Mutex<Submit>,
    /// The control completion ring.
    completion: Mutex<Producer>,
    /// Where posts onto the control completion ring wait for room.
    room: Room,
    /// How long a post waits for room before it looks again, in nanoseconds: [`ROOM_POLL`]; the
    /// tests make it long to see what wakes a post.
    room_poll_ns: AtomicU64,
    /// Whether the last post onto the control completion ring found no room: set and cleared with
    /// the ring held, so a full spell is told once, at its first post that found no room
    /// ([`RingDegradation::ControlRingFull`]).
    control_full: AtomicBool,
    /// The answers a thread that may not wait for room on the control completion ring handed
    /// over ([`CompleteAnswer::answer`]), for the control thread, or the next thread that posts
    /// onto the ring waiting for room, to post in their order.
    control_handed: Mutex<Handed>,
    /// Whether `control_handed` holds an answer or a thread is posting them: every post onto the
    /// control completion ring then goes behind them.
    control_handing: AtomicBool,
    /// Where the host's pointer and the guest's cursor are, for `POINTER_OVER` ([`Pointer`]).
    pointer: Mutex<Pointer>,
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
    /// The queues' doorbells the host routes now ([`Host::route_doorbell`]), at most
    /// [`MAX_DOORBELL_ROUTES`].
    routes: AtomicU32,
    /// The last route's number: each route takes the next.
    last_route: AtomicU64,
    /// The context each doorbell page is bound to ([`RGFX_REG_DOORBELL_PAGE_BIND`]), 0 for none:
    /// [`RGFX_DOORBELL_PAGES`] of them, all 0 after a reset.
    pages: Box<[AtomicU32]>,
    /// How many doorbell page writes were dropped ([`Shared::page_doorbell`]): of an unbound page,
    /// at another offset than a page's first byte, or of a ring not the page's context's.
    pub(crate) pages_dropped: AtomicU64,
    /// Whether the guest bound a doorbell page yet: its first bind is logged, a diagnostic that
    /// says the guest's processes ring at their pages.
    pages_bound: AtomicBool,
    /// A weak handle to this shared state, set at the device's make: a queue's service thread and
    /// its completion poster hold it, so neither keeps the device alive.
    weak_self: OnceLock<Weak<Shared>>,
    /// How many completion records were handed over to the queues' threads to post
    /// ([`CompletionPoster::post_elsewhere`], [`CompletionPoster::post_unwaiting`]).
    handed_over: AtomicU64,
    /// When the last line of `handed_over` was written, on the host's monotonic clock
    /// ([`Shared::tell_queue_ring`]).
    handed_logged_ns: AtomicU64,
    /// The longest a record handed over waited for a thread to take it to post, in nanoseconds,
    /// since that line ([`Shared::tell_queue_ring`]).
    handed_wait_ns: AtomicU64,
    /// The command buffers' guest times since the last line of them ([`guest_times`]), kept
    /// while that line's target logs.
    guest_times: Mutex<guest_times::Window>,
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

/// How long a crossing holds the pointer's positions back from the guest's pointing device
/// ([`Shared::pointer_at`]), in nanoseconds: until the guest's cursor shows on the pointer's
/// display, or this long after the `POINTER_OVER` that moves it there, should the guest not show
/// it (a hidden cursor) or be slow to.
const CROSSING_HOLD_NS: u64 = 100_000_000;

/// What `POINTER_OVER` follows ([`Shared::pointer_at`]): the display the guest's cursor last
/// showed on, what the guest was told, the displays' places in the guest's desktop, and where a
/// captured pointer that went past its window's edge is ([`Away`]), each of one reset count. A
/// reset makes it new: the cursor on display 0, the guest told nothing, no display placed.
#[derive(Default)]
struct Pointer {
    /// The reset count it is of.
    epoch: u64,
    /// The display of the guest's last `MOVE_CURSOR` that showed its cursor.
    cursor: u32,
    /// The (pointer's display, cursor's display) the guest was last told the pointer is over the
    /// first while its cursor is on the second, until the cursor shows on the pointer's display.
    told: Option<(u32, u32)>,
    /// When it was told, on the device's clock: the pointer's positions over that display are
    /// held back until the cursor shows there, or [`CROSSING_HOLD_NS`] after.
    told_at: u64,
    /// That `POINTER_OVER`, until it is posted: one that found the ring held by another thread,
    /// full, or behind answers handed over goes at the next refresh of any display.
    owed: Option<RgfxPointerOver>,
    /// The buttons of the last position handed on to the guest's pointing device: a position
    /// whose buttons differ is never held back.
    buttons: u32,
    /// Each display's rectangle in the guest's desktop, as its last `DISPLAY_BOUNDS` gave it.
    bounds: [Option<Bounds>; RGFX_DISPLAYS_MAX as usize],
    /// The captured pointer gone past its window's edge, while it is.
    away: Option<Away>,
}

/// A window's captured pointer gone past the image's edge onto the guest's desktop
/// ([`PointerLink::leave`]): where it is, and how its moves are scaled.
#[derive(Clone, Copy, Debug)]
struct Away {
    /// The desktop's points a pixel of the window's mode spans, across and down: the window's
    /// moves, in those pixels, move the pointer that far, wherever it is.
    per_pixel: (f64, f64),
    /// The pointer's point in the desktop.
    at: (f64, f64),
    /// The display it is on.
    on: u32,
}

/// Where a window's captured pointer is on the guest's desktop ([`PointerLink::away_moved`]): a
/// display, and the pixel of its scanout, which is `width` by `height`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spot {
    /// The display's id.
    pub display: u32,
    /// The pixel's column.
    pub x: u16,
    /// The pixel's row.
    pub y: u16,
    /// The scanout's width.
    pub width: u16,
    /// The scanout's height.
    pub height: u16,
}

impl Pointer {
    /// The state of reset count `epoch`: made new where it is of an earlier one.
    fn at(&mut self, epoch: u64) -> &mut Pointer {
        if self.epoch != epoch {
            *self = Pointer {
                epoch,
                ..Pointer::default()
            };
        }
        self
    }
}

/// A handle that tells the guest where the host's pointer is, from any thread ([`Self::pointer_at`]):
/// a display's window holds one. It holds the device weakly, so it keeps it alive no longer.
#[derive(Clone)]
pub struct PointerLink {
    /// The device's shared state, weakly.
    shared: Weak<Shared>,
}

impl PointerLink {
    /// The host's pointer is at (`x`, `y`) in display `display`'s pixels of its mode, over the
    /// display (its window), with the `RGFX_POINTER_BUTTON_*` `buttons` held: where the guest took
    /// [`RGFX_FEATURE_POINTER_OVER`] and [`RGFX_FEATURE_MULTI_DISPLAY`] and its cursor shows on
    /// another display, the guest is sent `POINTER_OVER`, once for that display and the display
    /// the cursor is on ([`RgfxPointerOver`]). Call it before the position goes to the guest's
    /// pointing device, so the guest moves its cursor onto the display first. Waits for no other
    /// thread: a `POINTER_OVER` that finds the ring held or full goes at the next refresh. Does
    /// nothing on a device of one display, for a guest that sees one or did not take
    /// `POINTER_OVER`, for a display it does not see, or once the device has ended.
    ///
    /// Whether the position goes on to the guest's pointing device now: not while the guest has
    /// yet to move its cursor onto the display, after the `POINTER_OVER` that has it do so, for up
    /// to 100 ms, so that the guest never takes the position as one of the display its cursor is
    /// still on (where its tablet's place would land at the same spot of the other display), but
    /// for a position whose buttons differ from the last that went on. True wherever nothing is
    /// told.
    pub fn pointer_at(&self, display: u32, x: i32, y: i32, buttons: u32) -> bool {
        self.shared
            .upgrade()
            .is_none_or(|shared| shared.pointer_at(display, x, y, buttons))
    }

    /// Whether the guest's desktop puts another display past display `display`'s edge toward
    /// `outward` (each of across and down -1, 0 or 1; both for the corner), at the row or column
    /// of its pixel `at` of the window's `mode`: where the window's captured pointer, moving
    /// toward its image's edge there, goes on onto that display ([`Self::leave`]). False where the
    /// guest did not take [`RGFX_FEATURE_POINTER_OVER`] and [`RGFX_FEATURE_DISPLAY_BOUNDS`] or
    /// gave no rectangle of the displays, and on a device of one display.
    pub fn adjoins(
        &self,
        display: u32,
        at: (u16, u16),
        mode: (u16, u16),
        outward: (i8, i8),
    ) -> bool {
        self.shared
            .upgrade()
            .is_some_and(|shared| shared.adjoins(display, at, mode, outward))
    }

    /// The captured pointer of display `display`'s window, at its pixel `at` of the window's
    /// `mode`, leaves the window for the guest's desktop, from that pixel: its moves go on there
    /// ([`Self::away_moved`]) until [`Self::come_back`]. False, and nothing done, where the
    /// desktop has no place for the display ([`Self::adjoins`]).
    pub fn leave(&self, display: u32, at: (u16, u16), mode: (u16, u16)) -> bool {
        self.shared
            .upgrade()
            .is_some_and(|shared| shared.leave(display, at, mode))
    }

    /// The pointer gone onto the desktop moved by (`dx`, `dy`), in pixels of the mode of the
    /// window it left: it goes onto the display whose rectangle holds where it lands, or is held
    /// at the edge of the one it is on where no display does (`desktop::step`). Where it is now;
    /// `None` where it is not away, or the guest reset the device or gave no place for the
    /// display it is on, when the window takes it back.
    pub fn away_moved(&self, dx: f64, dy: f64) -> Option<Spot> {
        self.shared.upgrade()?.away_moved(dx, dy)
    }

    /// The pointer is its window's again: back over the window's display, or capture let go.
    pub fn come_back(&self) {
        if let Some(shared) = self.shared.upgrade() {
            shared.come_back();
        }
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
    /// Whether a post that waits for room found none on the completion ring, and no post onto
    /// it has gone in since: a full spell is told once, at its first such post
    /// ([`RingDegradation::QueueRingFull`]).
    full: AtomicBool,
    /// What the posts that hand records over noted of the ring's full spells and waits, for the
    /// service thread to tell.
    untold: Mutex<Untold>,
    /// The records a thread that may not post onto the completion ring, or may not wait for room
    /// on it, handed over ([`CompletionPoster::post_elsewhere`],
    /// [`CompletionPoster::post_unwaiting`]), for the service thread, or the next thread of the
    /// device's that posts onto the ring, to post in their order.
    handed: Mutex<Handed>,
    /// Whether `handed` holds a record or a thread is posting them: every post onto the
    /// completion ring then goes behind them ([`Shared::post_behind_handed`],
    /// [`Shared::post_queue_unwaiting`]).
    handing: AtomicBool,
    /// The service thread's own wake.
    wake: Wake,
    /// Its bell, which the service thread sleeps on beside `wake`, where the guest's driver took
    /// [`RGFX_FEATURE_DOORBELL_ASKS`] and the host has eventfds ([`wake::bell`]): the file the
    /// host makes readable for the ring's plain doorbell where it routes it.
    bell: Option<Doorbell>,
    /// The route of its plain doorbell the host took ([`Host::route_doorbell`]), 0 where none:
    /// the queue's end takes it and drops it ([`Shared::unroute`]).
    route: AtomicU64,
    /// Where that route is: `u32::MAX` for `DOORBELL`, else the doorbell page's index. Stored
    /// before `route`, whose SeqCst store and swap order it for the queue's ender.
    route_page: AtomicU32,
    /// Set when the queue is destroyed, or at a reset or the device's end.
    ending: AtomicBool,
}

/// What follows a record posted through [`CompletionPoster::post_unwaiting`]: run once, right
/// after the record is posted, or dropped (a reset since its work began, the queue's end or the
/// device's), on the thread that posted it.
pub type Then = Box<dyn FnOnce() + Send>;

/// A queue's records handed over to be posted onto its completion ring
/// ([`CompletionPoster::post_elsewhere`], [`CompletionPoster::post_unwaiting`]).
#[derive(Default)]
struct Handed {
    /// The records, oldest first.
    records: VecDeque<HandedRecord>,
    /// Whether a thread is posting them now: a post onto the ring then joins them, and that
    /// thread posts it in its turn.
    posting: bool,
    /// Set as the queue's service thread ends, with every record left let go of: a record handed
    /// over after it would never be posted, so it is dropped where it is handed over, and what
    /// follows it run there ([`Shared::close_handed`]).
    closed: bool,
}

/// One record handed over ([`Handed`]).
struct HandedRecord {
    message: Vec<u8>,
    /// The reset count it was posted at.
    epoch: u64,
    /// When it was handed over, on the host's monotonic clock; 0 for a post that joined them.
    at: u64,
    /// What follows its post ([`Then`]).
    then: Option<Then>,
    /// When it began to wait for room, on the host's monotonic clock: a post of it found the ring
    /// with none, or it was handed over behind others in a full spell of the ring; 0 where
    /// neither.
    no_room_since: u64,
}

/// What a queue's posts noted of its full completion ring for its service thread to tell
/// ([`Shared::tell_queue_ring`]).
#[derive(Default)]
struct Untold {
    /// Full spells begun.
    spells: u32,
    /// Records handed over that found no room, posted since.
    waits: u32,
    /// The longest of their waits, in nanoseconds.
    longest_ns: u64,
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
        // SeqCst, as the route's store and swap are ([`Shared::route`]): either the queue's maker
        // sees the end after it stored its route, or the ender sees the route.
        self.ending.store(true, Ordering::SeqCst);
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
    /// Ring 0's: the control thread sleeps with nothing in flight, every answer of what it took
    /// posted, from its look that found the ring empty to its wake (`control_thread`): a
    /// doorbell's thread may then take the ring's next message ([`Shared::download_on_doorbell`]).
    /// False on a queue's ring.
    asleep: bool,
}

/// One of the device's displays: what shows it, and what the guest set of it.
pub(crate) struct Display {
    /// What shows it.
    pub(crate) screen: Box<dyn Screen>,
    /// What the guest set of it, and what the device told it.
    pub(crate) state: Mutex<DisplayState>,
}

/// A display, as the guest has set it, and what the device has told it.
#[derive(Debug, Default)]
pub(crate) struct DisplayState {
    /// The pixels it shows.
    pub(crate) scanout: Option<Scanout>,
    /// `VBLANK` is on.
    pub(crate) vblank: bool,
    /// The panel is off (`SET_DISPLAY_POWER`): the screen shows black.
    pub(crate) asleep: bool,
    /// Refreshes since the last reset.
    refreshes: u64,
    /// The mode the guest was last told, in a `DISPLAYS` or a `DISPLAY_CHANGED`'s wake.
    pub(crate) told: Option<Mode>,
    /// A `DISPLAY_CHANGED` is owed.
    changed: bool,
}

impl Device {
    /// A device of one display over `region` (BAR 2: [`REGION_SIZE`] bytes, of which what shows
    /// the display may hold a reference to read the scanout from), `host` and `screen`, with its
    /// threads started; `None` where the region is smaller than the rings and a page of
    /// allocations, or the host refuses a pipe or a thread.
    pub fn new(
        region: Arc<Region>,
        host: Box<dyn Host>,
        screen: Box<dyn Screen>,
    ) -> Option<Device> {
        Device::with_screens(region, host, vec![screen])
    }

    /// [`Self::new`], for a device of `screens.len()` displays, display `i` shown by `screens[i]`
    /// (docs/graphics-kext/design.md, "Multiple displays"): `None` also where there are none or
    /// more than [`RGFX_DISPLAYS_MAX`]. A device of two or more offers
    /// [`RGFX_FEATURE_MULTI_DISPLAY`]; one of one is what a device was before displays had
    /// rooms of their own.
    pub fn with_screens(
        region: Arc<Region>,
        host: Box<dyn Host>,
        screens: Vec<Box<dyn Screen>>,
    ) -> Option<Device> {
        Device::with_lanes(region, host, screens, Lanes::default())
    }

    /// [`Self::with_screens`], over `lanes`, whose threads wait as long as they keep (the tests;
    /// else [`crate::keep::KEEP`]).
    fn with_lanes(
        region: Arc<Region>,
        host: Box<dyn Host>,
        screens: Vec<Box<dyn Screen>>,
        lanes: Lanes,
    ) -> Option<Device> {
        if region.size() < RGFX_ALLOCATIONS_OFFSET + 4096
            || screens.is_empty()
            || screens.len() > RGFX_DISPLAYS_MAX as usize
        {
            return None;
        }
        let displays = screens
            .into_iter()
            .map(|screen| Display {
                screen,
                state: Mutex::new(DisplayState::default()),
            })
            .collect();
        let mut consumer = Consumer::new(SUBMIT);
        consumer.reset(&region);
        let mut producer = Producer::new(COMPLETE);
        producer.reset(&region);
        let shared = Arc::new(Shared {
            host,
            displays,
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
                asleep: false,
            }),
            completion: Mutex::new(producer),
            room: Room::default(),
            room_poll_ns: AtomicU64::new(u64::try_from(ROOM_POLL.as_nanos()).unwrap_or(u64::MAX)),
            control_full: AtomicBool::new(false),
            control_handed: Mutex::new(Handed::default()),
            control_handing: AtomicBool::new(false),
            pointer: Mutex::new(Pointer::default()),
            control_done: AtomicU64::new(0),
            gate: Gate::default(),
            lanes: Arc::new(lanes),
            queues: Mutex::new(Queues::default()),
            routes: AtomicU32::new(0),
            last_route: AtomicU64::new(0),
            pages: (0..RGFX_DOORBELL_PAGES)
                .map(|_| AtomicU32::new(0))
                .collect(),
            pages_dropped: AtomicU64::new(0),
            pages_bound: AtomicBool::new(false),
            weak_self: OnceLock::new(),
            handed_over: AtomicU64::new(0),
            handed_logged_ns: AtomicU64::new(0),
            handed_wait_ns: AtomicU64::new(0),
            guest_times: Mutex::new(guest_times::Window::default()),
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
        for display in 0..shared.display_count() {
            let refresh = Arc::clone(&shared);
            let name = match display {
                0 => "reims-vgpu.graphics-refresh".to_owned(),
                _ => format!("reims-vgpu.graphics-refresh-{display}"),
            };
            let started = std::thread::Builder::new()
                .name(name)
                .spawn(move || refresh.refresh_thread(display));
            if started.is_err() {
                shared.end();
                return None;
            }
        }
        Some(Device { shared })
    }

    /// The region's first byte, which the host maps into the guest as BAR 2.
    pub fn region_base(&self) -> *mut u8 {
        self.shared.region.base()
    }

    /// How many completion records were handed over to the queues' threads to post, from threads
    /// that may not post them ([`CompletionPoster::post_elsewhere`]) or may not wait for room
    /// ([`CompletionPoster::post_unwaiting`]).
    pub fn handed_over(&self) -> u64 {
        self.shared.handed_over.load(Ordering::Relaxed)
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

    /// Whether the control thread sleeps with no wake pending ([`Wake::parked`]): it takes no lock
    /// until it is woken.
    #[cfg(test)]
    pub(crate) fn control_parked(&self) -> bool {
        let doorbell = self.shared.doorbell.get().map(wake::raw);
        self.shared.wake.parked(doorbell)
    }

    /// Whether the control thread and every queue's service thread sleep with no wake pending
    /// ([`Wake::parked`]), no lane has a message to carry out, and each of ring 0's threads waits
    /// for a lane with the lanes' lock let go ([`Lanes::parked`]): what they write of the rings
    /// stays as it is until the guest acts, and until then none takes a lock that a doorbell's
    /// thread tries ([`Shared::download_on_doorbell`], [`Shared::answer_on_doorbell`]) but a ring-0
    /// thread that has waited [`crate::keep::KEEP`] for a lane, and goes.
    #[cfg(test)]
    pub(crate) fn parked(&self) -> bool {
        self.control_parked()
            && self.shared.lanes.parked()
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
            RGFX_REG_FEATURES_LO => (self.shared.offered() & 0xffff_ffff) as u32,
            RGFX_REG_FEATURES_HI => (self.shared.offered() >> 32) as u32,
            RGFX_REG_STATUS => self.shared.status.load(Ordering::Acquire),
            RGFX_REG_DRIVER_FEATURES_LO => {
                (self.shared.driver_features.load(Ordering::Acquire) & 0xffff_ffff) as u32
            }
            RGFX_REG_DRIVER_FEATURES_HI => {
                (self.shared.driver_features.load(Ordering::Acquire) >> 32) as u32
            }
            RGFX_REG_REGION_SIZE_LO => (size & 0xffff_ffff) as u32,
            RGFX_REG_REGION_SIZE_HI => (size >> 32) as u32,
            RGFX_REG_DEFERRED_BUDGET_LO => (self.shared.deferred_budget() & 0xffff_ffff) as u32,
            RGFX_REG_DEFERRED_BUDGET_HI => (self.shared.deferred_budget() >> 32) as u32,
            // The clock the refresh thread's grid stamps each VBLANK on.
            RGFX_REG_HOST_TIME_LO if host_time() => (now_ns() & 0xffff_ffff) as u32,
            RGFX_REG_HOST_TIME_HI if host_time() => (now_ns() >> 32) as u32,
            _ => 0,
        }
    }

    /// A 32-bit write of BAR 0 at `offset`: `STATUS` 0 resets the device and 1 starts it taking
    /// commands; `DRIVER_FEATURES_LO` and `_HI` take, before `STATUS` 1, the bits written that the
    /// host offers and knows; `DOORBELL` with a ring's id wakes its consumer;
    /// `DOORBELL_PAGE_BIND` binds a doorbell page to a context where the driver took
    /// [`RGFX_FEATURE_DOORBELL_PAGES`]. Other values and offsets (`CONSUMED` among them) are
    /// ignored. A reset waits for the message the control
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
            RGFX_REG_DOORBELL_PAGE_BIND => self.shared.bind_page(value),
            _ => {}
        }
    }

    /// [`Self::mmio_write`], but a `DOORBELL` write of a command queue's ring (with
    /// [`RGFX_DOORBELL_ASKS`], where the guest's driver took [`RGFX_FEATURE_DOORBELL_ASKS`]) first
    /// offers the queue's next `COMMAND_BUFFER` to the host on the calling thread
    /// ([`Host::answer_command_buffer`]; the module doc): the host may post its `CB_SUBMITTED`
    /// there, calling [`Host::raise_interrupt`] inside this call. A write of ring 0's
    /// (`RGFX_CONTROL_RING_ID`) first answers the ring's next message there where it is a
    /// `SURFACE_DOWNLOAD` the host has carried out already ([`Host::surface_downloaded`]),
    /// posting its `COMPLETE` and raising the control vector inside this call. It waits for the
    /// queues' table where it is held, as `mmio_write` does, and for nothing else, and the offer
    /// allocates nothing.
    pub fn mmio_write_taking(&self, offset: u64, value: u32) {
        match u32::try_from(offset).unwrap_or(u32::MAX) {
            RGFX_REG_DOORBELL => self.shared.ring_taking(u64::from(value)),
            _ => self.mmio_write(offset, value),
        }
    }

    /// A 32-bit write of BAR 4, the doorbell pages ([`RGFX_PCI_BAR_DOORBELLS`]), at `offset`. A
    /// write at a page's first byte of a ring of the command queues of the context the page is
    /// bound to does what that `DOORBELL` write through [`Self::mmio_write_taking`] does: it wakes
    /// the queue's thread, and where it asks, offers the queue's next `COMMAND_BUFFER` to the host
    /// on the calling thread, which may raise inside this call. Any other write is dropped and
    /// counted, logged once a kind: one of an unbound page or at another offset (`unbound`), and
    /// one of ring 0's id or of another context's ring (`foreign`), so a process reaches no ring
    /// but its own context's through its page. A ring of no queue is ignored, as at `DOORBELL`.
    /// It waits for the queues' table where it is held, and for nothing else.
    pub fn doorbell_write(&self, offset: u64, value: u32) {
        self.shared.page_doorbell(offset, value);
    }

    /// Takes a copy of `doorbell`, the host's doorbell for ring 0 (its `ioeventfd`'s, readable
    /// or set once the guest writes 0 to `DOORBELL`: a file on Linux and macOS, an event on
    /// Windows), which the control thread then sleeps on too. False where `doorbell` is not open,
    /// or the device has one already.
    pub fn set_doorbell(&self, doorbell: RawDoorbell) -> bool {
        let Some(copy) = wake::duplicate(doorbell) else {
            return false;
        };
        if self.shared.doorbell.set(copy).is_err() {
            return false;
        }
        self.shared.wake.wake();
        true
    }

    /// The handle a display's window tells the guest of the host's pointer through
    /// ([`PointerLink::pointer_at`]).
    pub fn pointer_link(&self) -> PointerLink {
        PointerLink {
            shared: Arc::downgrade(&self.shared),
        }
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
    /// Whether the let-go marked the queue's submission ring walking, which the retake clears.
    walking: bool,
}

impl<'a> QueueClaim<'a> {
    /// Lets go of the claim, marks the queue's submission ring walking
    /// ([`RGFX_CONSUMER_WALKING`]) until the claim is taken back, then offers the queue's next
    /// message to the host on this thread as a doorbell's thread does
    /// ([`Host::answer_command_buffer`]): a buffer the guest published before the claim was let
    /// go, and now waits for its `CB_SUBMITTED`, is answered here where it need wait for nothing,
    /// not after the work this thread carries out next. The guest rings a walking ring's doorbell
    /// for a buffer that asks for its `CB_SUBMITTED` alone, which the doorbell's thread offers
    /// the host in the write; the rest it publishes during that work this thread takes after it,
    /// with no doorbell. Once let go, it stays so for the call.
    pub fn let_go(&mut self) {
        if self.held.take().is_some() {
            self.walking = self.shared.mark_walking(self.queue);
            self.shared.answer_on_doorbell(self.queue);
        }
    }

    /// The queue's `consuming` again, taken where the claim was let go, with the queue's
    /// submission ring no longer marked walking, as while this thread takes.
    fn retake(self) -> MutexGuard<'a, Option<u64>> {
        let QueueClaim {
            shared,
            queue,
            held,
            walking,
        } = self;
        let consuming = held.unwrap_or_else(|| queue.consuming.lock());
        if walking {
            shared.mark_awake(queue);
        }
        consuming
    }
}

impl CompleteAnswer {
    /// Posts the `COMPLETE` with `status` onto the control completion ring, waiting for nothing,
    /// on whichever thread gives it: the platform's that run a command buffer's handlers, where a
    /// drain's or a display hold's answer is given as the buffer completes, hold up every queue's
    /// work behind them while they wait. Where the ring has room and no answer is handed over
    /// before it, it is posted here, and the control vector raised here; otherwise it is handed to
    /// the control thread, which posts the answers in their order before its own posts, waiting
    /// for room, and every post onto the ring that waits goes behind them.
    pub fn answer(self, status: u32) {
        if let Some(shared) = self.shared.upgrade() {
            shared.post_unwaiting(&crate::control::complete(self.seqno, status), self.epoch);
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

    /// Whether `other` posts as this does: onto the same ring of the same device, for work begun
    /// at the same reset count. What either posts, and whether it is dropped, is then the same.
    pub fn posts_as(&self, other: &CompletionPoster) -> bool {
        let same_queue = match (&self.queue, &other.queue) {
            (None, None) => true,
            (Some((a, a_id)), Some((b, b_id))) => a_id == b_id && Weak::ptr_eq(a, b),
            _ => false,
        };
        same_queue && self.epoch == other.epoch && Weak::ptr_eq(&self.shared, &other.shared)
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

    /// [`Self::post`], waiting for no room and for no other thread, from a thread that may not
    /// wait (a platform's thread that runs a command buffer's submission or completion handlers,
    /// or its shared events' notifications, whose waits hold up every queue's work behind them),
    /// then `then` ([`Then`]). Where no record is handed over before it on the ring
    /// ([`Self::post_elsewhere`], or by an earlier call of this) and the ring has room, `message`
    /// is posted here, the ring's vector raised here as [`Self::post`] raises it, and `then` run
    /// here. Otherwise it is handed to the queue's service thread, behind those records, which
    /// posts it, raises the vector and runs `then` there, in their order, waiting for room as it
    /// waits for its own posts; every post onto the ring from then on goes after it, as after a
    /// record [`Self::post_elsewhere`] hands over. The only locks it takes are the ring's, held
    /// elsewhere only across a write into it, and the records', held only across a hand-over or
    /// a take of one. Where the device has ended, the queue is gone, or a reset came since the
    /// work began, `message` is dropped and `then` run here. A poster of the control ring posts
    /// as [`CompleteAnswer::answer`] posts, then runs `then` here.
    pub fn post_unwaiting(&self, message: &[u8], then: Option<Then>) {
        let Some(shared) = self.shared.upgrade() else {
            return run(then);
        };
        match &self.queue {
            None => {
                shared.post_unwaiting(message, self.epoch);
                run(then);
            }
            Some((queue, _)) => match queue.upgrade() {
                Some(queue) => shared.post_queue_unwaiting(&queue, message, self.epoch, then),
                None => run(then),
            },
        }
    }

    /// [`Self::post`], from a thread that may neither post onto the ring nor raise its vector: a
    /// thread of the host's, or one of the library's the host's raise callback is not named for
    /// (the window's render thread, a commit's inside a call of the host's into the library),
    /// which runs a command buffer's submission handler inside its commit. The message is handed
    /// to the queue's service thread, which posts it and raises the vector, after the records
    /// handed over before it; every post onto the ring from then on goes after it. This waits for
    /// nothing and raises nothing: a full ring holds back the service thread, not this one. A
    /// poster of the control ring, onto which no submission handler posts, posts at once.
    pub fn post_elsewhere(&self, message: &[u8]) {
        let Some(shared) = self.shared.upgrade() else {
            return;
        };
        match &self.queue {
            None => shared.post_waiting(message, self.epoch),
            Some((queue, _)) => {
                if let Some(queue) = queue.upgrade() {
                    shared.hand_over(&queue, message, self.epoch);
                }
            }
        }
    }
}

/// When a record handed over behind others starts to wait for room: now where `queue`'s ring is in
/// a full spell, so it counts among the posts that waited ([`RingDegradation::QueuePostWaited`]),
/// else 0.
fn behind_since(queue: &QueueShared) -> u64 {
    if queue.full.load(Ordering::Relaxed) {
        now_ns()
    } else {
        0
    }
}

/// Runs `then`, where there is one.
fn run(then: Option<Then>) {
    if let Some(then) = then {
        then();
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
        // No page is any context's: the kext binds them again as its contexts open.
        for page in &self.pages {
            page.store(0, Ordering::Release);
        }
        let epoch = self.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        // Nothing the lanes were handed before it is begun now.
        self.lanes.reset(epoch);
        let mut submit = self.submit.lock();
        let mut states: Vec<MutexGuard<'_, DisplayState>> =
            self.displays.iter().map(|d| d.state.lock()).collect();
        let mut completion = self.completion.lock();
        submit.consumer.reset(&self.region);
        submit.broken = false;
        submit.backlog.clear();
        self.control_done.store(0, Ordering::Release);
        completion.reset(&self.region);
        // Each display's scanout stays (the module doc): its screen is told nothing of it. Its
        // panel is on.
        let woken: Vec<bool> = (states.iter_mut())
            .map(|state| {
                let woken = state.asleep;
                **state = DisplayState {
                    scanout: state.scanout,
                    ..DisplayState::default()
                };
                woken
            })
            .collect();
        // With the ring held: the control thread tells the host before any message after this.
        self.reset_owed.store(true, Ordering::Release);
        drop(completion);
        drop(states);
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
            self.unroute(queue);
        }
        self.region.revoke_all();
        self.gate.every();
        self.room.wake();
        self.wake.wake();
        for (display, woken) in self.displays.iter().zip(woken) {
            display.screen.cursor_image(None);
            display.screen.cursor_moved(0, 0, true);
            if woken {
                display.screen.power(true);
            }
        }
    }

    /// The feature bits the device offers a guest whose driver took `taken`: the host's
    /// ([`Host::features`]), [`RGFX_FEATURE_MULTI_DISPLAY`], [`RGFX_FEATURE_POINTER_OVER`] and
    /// [`RGFX_FEATURE_DISPLAY_BOUNDS`] on a device of two displays or more (the device's to offer,
    /// whatever the host's answer), and where `taken` holds [`RGFX_FEATURE_PAGES_16K`] and the
    /// host offers it, those it offers only then ([`Host::features_with_16k_pages`]).
    fn offered_to(&self, taken: u64) -> u64 {
        let displays =
            RGFX_FEATURE_MULTI_DISPLAY | RGFX_FEATURE_POINTER_OVER | RGFX_FEATURE_DISPLAY_BOUNDS;
        let features = (self.host.features() & !displays)
            | if self.display_count() > 1 {
                displays
            } else {
                0
            };
        if taken & features & RGFX_FEATURE_PAGES_16K != 0 {
            features | self.host.features_with_16k_pages()
        } else {
            features
        }
    }

    /// The feature bits `FEATURES_LO` and `_HI` read: those offered to the guest's driver as it
    /// took them so far ([`Self::offered_to`]).
    fn offered(&self) -> u64 {
        self.offered_to(self.driver_features.load(Ordering::Acquire))
    }

    /// Whether the guest's driver took `feature`.
    pub(crate) fn driver_took(&self, feature: u64) -> bool {
        self.driver_features.load(Ordering::Acquire) & feature != 0
    }

    /// How many displays the device has: 1 to [`RGFX_DISPLAYS_MAX`], fixed at its make.
    pub(crate) fn display_count(&self) -> u32 {
        u32::try_from(self.displays.len()).unwrap_or(RGFX_DISPLAYS_MAX)
    }

    /// How many displays the guest sees: every one where its driver took
    /// [`RGFX_FEATURE_MULTI_DISPLAY`], else display 0 alone, as on a device of one display.
    pub(crate) fn displays_seen(&self) -> u32 {
        if self.driver_took(RGFX_FEATURE_MULTI_DISPLAY) {
            self.display_count()
        } else {
            1
        }
    }

    /// Display `id`, where the guest sees it ([`Self::displays_seen`]): a command naming any other
    /// is refused `NO_DISPLAY`, as by a device without it.
    pub(crate) fn seen_display(&self, id: u32) -> Option<&Display> {
        (id < self.displays_seen())
            .then(|| self.displays.get(id as usize))
            .flatten()
    }

    /// The guest's `MOVE_CURSOR` showed its cursor on display `id`, one it sees: the display
    /// `POINTER_OVER` follows ([`Self::pointer_at`]).
    pub(crate) fn cursor_shown_on(&self, id: u32) {
        let epoch = self.epoch.load(Ordering::Acquire);
        let mut pointer = self.pointer.lock();
        let pointer = pointer.at(epoch);
        pointer.cursor = id;
        if pointer.told.is_some_and(|(over, _)| over == id) {
            // The guest moved its cursor where it was told: a later move elsewhere is told anew.
            log::debug!(
                target: "reims_vgpu_graphics",
                "the guest's cursor showed on display {id} {} us after the POINTER_OVER",
                now_ns().saturating_sub(pointer.told_at) / 1000
            );
            pointer.told = None;
            pointer.owed = None;
        }
    }

    /// `DISPLAY_BOUNDS`: display `command.display`'s rectangle in the guest's desktop, kept until
    /// the next of it or a reset; one of no area forgets it. Dropped for a display the guest does
    /// not see.
    pub(crate) fn display_bounds(&self, command: &RgfxDisplayBounds) {
        if self.seen_display(command.display).is_none() {
            return;
        }
        let epoch = self.epoch.load(Ordering::Acquire);
        let mut pointer = self.pointer.lock();
        if let Some(slot) = pointer.at(epoch).bounds.get_mut(command.display as usize) {
            *slot = Bounds::of(command);
        }
    }

    /// Whether the window's captured pointer may go past its image's edge onto the guest's
    /// desktop: the guest sees several displays, took `POINTER_OVER`, which moves its cursor
    /// there, and `DISPLAY_BOUNDS`, which places them, and runs the device.
    fn carries_beyond(&self) -> bool {
        let wanted = RGFX_FEATURE_POINTER_OVER | RGFX_FEATURE_DISPLAY_BOUNDS;
        self.displays_seen() > 1
            && self.driver_features.load(Ordering::Acquire) & wanted == wanted
            && self.started()
    }

    /// Each display's scanout size, of those the guest sees, read before the pointer's lock.
    fn scanout_sizes(&self) -> Vec<Option<(u32, u32)>> {
        (0..self.displays_seen())
            .map(|id| {
                let display = self.displays.get(id as usize)?;
                let scanout = display.state.lock().scanout?;
                (scanout.width > 0 && scanout.height > 0).then_some((scanout.width, scanout.height))
            })
            .collect()
    }

    /// The displays' places in the desktop: a display's where the guest gave its rectangle and
    /// it has a scanout.
    fn places(sizes: &[Option<(u32, u32)>], bounds: &[Option<Bounds>]) -> Vec<Option<Place>> {
        sizes
            .iter()
            .zip(bounds)
            .map(|(&pixels, &bounds)| {
                Some(Place {
                    bounds: bounds?,
                    pixels: pixels?,
                })
            })
            .collect()
    }

    /// [`PointerLink::adjoins`].
    pub(crate) fn adjoins(
        &self,
        id: u32,
        at: (u16, u16),
        mode: (u16, u16),
        outward: (i8, i8),
    ) -> bool {
        if !self.carries_beyond() || mode.0 == 0 || mode.1 == 0 {
            return false;
        }
        let sizes = self.scanout_sizes();
        let epoch = self.epoch.load(Ordering::Acquire);
        let mut pointer = self.pointer.lock();
        let places = Self::places(&sizes, &pointer.at(epoch).bounds);
        let Some(place) = places.get(id as usize).copied().flatten() else {
            return false;
        };
        let centre = place.point_of(pixel_centre(at, mode, place.pixels));
        let half = place.points_per_pixel();
        let past = |toward: i8, at: f64, min: f64, max: f64, half: f64| match toward.signum() {
            1 => max + half / 2.0,
            -1 => min - half / 2.0,
            _ => at,
        };
        let bounds = place.bounds;
        let point = (
            past(outward.0, centre.0, bounds.min.0, bounds.max.0, half.0),
            past(outward.1, centre.1, bounds.min.1, bounds.max.1, half.1),
        );
        desktop::display_at(&places, id as usize, point).is_some_and(|d| d != id as usize)
    }

    /// [`PointerLink::leave`].
    pub(crate) fn leave(&self, id: u32, at: (u16, u16), mode: (u16, u16)) -> bool {
        if !self.carries_beyond() || mode.0 == 0 || mode.1 == 0 {
            return false;
        }
        let sizes = self.scanout_sizes();
        let epoch = self.epoch.load(Ordering::Acquire);
        let mut pointer = self.pointer.lock();
        let pointer = pointer.at(epoch);
        let places = Self::places(&sizes, &pointer.bounds);
        let Some(place) = places.get(id as usize).copied().flatten() else {
            return false;
        };
        let per_scanout_pixel = place.points_per_pixel();
        pointer.away = Some(Away {
            per_pixel: (
                per_scanout_pixel.0 * f64::from(place.pixels.0) / f64::from(mode.0),
                per_scanout_pixel.1 * f64::from(place.pixels.1) / f64::from(mode.1),
            ),
            at: place.point_of(pixel_centre(at, mode, place.pixels)),
            on: id,
        });
        true
    }

    /// [`PointerLink::away_moved`].
    pub(crate) fn away_moved(&self, dx: f64, dy: f64) -> Option<Spot> {
        let carries = self.carries_beyond();
        let sizes = self.scanout_sizes();
        let epoch = self.epoch.load(Ordering::Acquire);
        let mut pointer = self.pointer.lock();
        let pointer = pointer.at(epoch);
        let away = pointer.away.take()?;
        if !carries {
            return None;
        }
        let places = Self::places(&sizes, &pointer.bounds);
        let to = (
            away.at.0 + dx * away.per_pixel.0,
            away.at.1 + dy * away.per_pixel.1,
        );
        let (on, at) = desktop::step(&places, away.on as usize, away.at, to)?;
        let place = places.get(on).copied().flatten()?;
        let on = u32::try_from(on).ok()?;
        pointer.away = Some(Away { at, on, ..away });
        let pixel = place.pixel_of(at);
        let narrow = |v: u32| u16::try_from(v).unwrap_or(u16::MAX);
        Some(Spot {
            display: on,
            x: narrow(pixel.0),
            y: narrow(pixel.1),
            width: narrow(place.pixels.0),
            height: narrow(place.pixels.1),
        })
    }

    /// [`PointerLink::come_back`].
    pub(crate) fn come_back(&self) {
        let epoch = self.epoch.load(Ordering::Acquire);
        self.pointer.lock().at(epoch).away = None;
    }

    /// [`PointerLink::pointer_at`]: the host's pointer is over display `id` at (`x`, `y`), with
    /// `buttons` held. Whether the position goes on to the guest's pointing device now.
    pub(crate) fn pointer_at(&self, id: u32, x: i32, y: i32, buttons: u32) -> bool {
        let seen = self.displays_seen();
        if seen < 2 || id >= seen || !self.driver_took(RGFX_FEATURE_POINTER_OVER) || !self.started()
        {
            return true;
        }
        let epoch = self.epoch.load(Ordering::Acquire);
        let now = now_ns();
        let goes_on = {
            let mut pointer = self.pointer.lock();
            let pointer = pointer.at(epoch);
            let buttons_changed = std::mem::replace(&mut pointer.buttons, buttons) != buttons;
            if pointer.cursor == id {
                return true;
            }
            if pointer.told == Some((id, pointer.cursor)) {
                // Told already: held until the cursor shows there, for a while at most.
                return buttons_changed || now.saturating_sub(pointer.told_at) >= CROSSING_HOLD_NS;
            }
            pointer.told = Some((id, pointer.cursor));
            pointer.told_at = now;
            pointer.owed = Some(RgfxPointerOver {
                header: event_header(RGFX_EVT_POINTER_OVER),
                display: id,
                x,
                y,
                buttons,
            });
            buttons_changed
        };
        self.post_pointer(epoch, false);
        goes_on
    }

    /// Posts the `POINTER_OVER` owed at reset count `epoch`, where one is ([`Pointer::owed`]),
    /// waiting for the ring held elsewhere and for room where `wait`. One that finds answers
    /// handed over before it, or that is not posted, stays owed.
    fn post_pointer(&self, epoch: u64, wait: bool) {
        let Some(event) = self.pointer.lock().at(epoch).owed else {
            return;
        };
        if self.control_handing.load(Ordering::Acquire) {
            return;
        }
        if matches!(
            self.post_on(&sized(&event), epoch, wait),
            Post::Done | Post::Dropped
        ) {
            let mut pointer = self.pointer.lock();
            let pointer = pointer.at(epoch);
            if pointer.owed == Some(event) {
                pointer.owed = None;
            }
        }
    }

    /// The mode `display` offers its guest now: its screen's ([`Screen::mode`]), sized to its
    /// scanout room on a device of more than one display ([`offered_in_room`]).
    pub(crate) fn offered_mode(&self, display: &Display) -> Mode {
        offered_in_room(display.screen.mode(), self.display_count())
    }

    /// Whether `len` bytes at `offset` lie in display `id`'s scanout room
    /// ([`scanout_room_at`]: the whole of [`RGFX_SCANOUT_SPAN`] at [`RGFX_ALLOCATIONS_OFFSET`] on
    /// a device of one display), before the next display's room or the accelerator kext's blocks
    /// ([`RGFX_CONTEXT_ALLOCATIONS_OFFSET`]), and in the region, `offset` a multiple of 4. A
    /// scanout past its room would show another display's, or the processes' blocks.
    pub(crate) fn in_scanout_room(&self, id: u32, offset: u64, len: u64) -> bool {
        let count = self.display_count();
        let (start, room) = (scanout_room_at(id, count), scanout_room(count));
        offset.is_multiple_of(4)
            && offset >= start
            && len <= room
            && offset - start <= room - len
            && self.region.contains(offset, len)
    }

    /// What `DEFERRED_BUDGET_LO` and `_HI` read: the host's budget where the device offers a
    /// deferral ([`RGFX_FEATURE_SURFACE_DEFERRED`] or [`RGFX_FEATURE_SURFACE_DEFERRED_ASK`]),
    /// else 0, as the host answers where it offers neither.
    fn deferred_budget(&self) -> u64 {
        let deferrals =
            u64::from(RGFX_FEATURE_SURFACE_DEFERRED | RGFX_FEATURE_SURFACE_DEFERRED_ASK);
        if self.offered() & deferrals != 0 {
            self.host.deferred_budget()
        } else {
            0
        }
    }

    /// A write of `DRIVER_FEATURES`' half at bit `shift` (0 for `_LO`, 32 for `_HI`): that half
    /// becomes the bits of `value` the device offers and the host knows ([`KNOWN_FEATURES`]), the
    /// other half kept, where the offer is the one to the bits taken after the write
    /// ([`Self::offered_to`]): a write that takes [`RGFX_FEATURE_PAGES_16K`] may take what it
    /// offers in the same write, and one that lets it go lets those go too. Ignored once the
    /// driver is ready: the features are fixed from `STATUS` 1 to the next reset.
    fn take_driver_features(&self, shift: u32, value: u32) {
        if self.status.load(Ordering::Acquire) == RGFX_STATUS_DRIVER_READY {
            return;
        }
        let half = 0xffff_ffff_u64 << shift;
        let written = u64::from(value) << shift;
        let _ = self
            .driver_features
            .try_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                let asked = ((old & !half) | written) & KNOWN_FEATURES;
                Some(asked & self.offered_to(asked))
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
            // At debug: reims-vgpu's own line of the reset, at info, carries its counts.
            log::debug!(target: "reims_vgpu_graphics", "the device was reset");
        }
    }

    /// The control thread (the module doc).
    fn control_thread(&self) {
        while !self.ending() {
            // The answers handed over to it, posted first, as a post onto the ring would.
            if self.control_handing.load(Ordering::Acquire) {
                self.post_behind_control_handed(None);
            }
            self.tell_reset();
            self.take_messages();
            // Asleep with `consumer_idle` set only while the device takes the ring.
            let (idle, epoch) = {
                let mut submit = self.submit.lock();
                let epoch = self.epoch.load(Ordering::Acquire);
                if !self.started() || submit.broken {
                    (false, epoch)
                } else if submit.consumer.sleep_or_pending(&self.region) {
                    continue;
                } else {
                    submit.asleep = true;
                    (true, epoch)
                }
            };
            if self.ending() {
                break;
            }
            self.wake.sleep(self.doorbell.get().map(wake::raw));
            self.host.trace("r0wake", &[]);
            // A doorbell may follow the guest's take of completions.
            self.room.wake();
            // A reset while it slept set the ring up anew, its `consumer_idle` 1, so the guest
            // rings for its first message: that stands. The reset moves the count before it takes
            // the ring, so one that has not yet set the ring up sets it after this.
            if idle {
                let mut submit = self.submit.lock();
                submit.asleep = false;
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
                    let obj_id = message
                        .bytes
                        .get(16..24)
                        .and_then(|b| <[u8; 8]>::try_from(b).ok())
                        .map_or(0, u64::from_le_bytes);
                    self.host.trace(
                        "r0take",
                        &[
                            u64::from(message.header.r#type),
                            message.header.seqno,
                            obj_id,
                        ],
                    );
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
                            target: "reims_vgpu_graphics", kind:% = broken;
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
        let hand = self
            .lanes
            .hand(key, message, through, epoch, &self.region, |surface| {
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
                target: "reims_vgpu_graphics",
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
        log::debug!(target: "reims_vgpu_graphics::room", "SUBMIT_ROOM posted for the control ring");
        self.post_waiting(&submit_room(), epoch);
    }

    /// Posts `message`, waiting for room ([`Room`]) while the device is not reset or ended,
    /// behind the answers handed over before it ([`Self::post_unwaiting`]). A post that waited is
    /// told to the host ([`RingDegradation::ControlPostWaited`]).
    fn post_waiting(&self, message: &[u8], epoch: u64) {
        if self.control_handing.load(Ordering::Acquire)
            && self.post_behind_control_handed(Some((message, epoch)))
        {
            return;
        }
        self.post_waiting_now(message, epoch);
    }

    /// [`Self::post_waiting`] of `message` with no answer handed over before it.
    fn post_waiting_now(&self, message: &[u8], epoch: u64) {
        let mut waited_since = None;
        loop {
            let seen = self.room.wakes();
            if self.post(message, epoch) != Post::NoRoom || self.ending() {
                if let Some(since) = waited_since {
                    self.post_waited(since);
                }
                return;
            }
            waited_since.get_or_insert_with(now_ns);
            self.room.wait(seen, self.room_poll());
        }
    }

    /// A post onto the control completion ring that waited for room since `since_ns`
    /// ([`now_ns`]) went on.
    fn post_waited(&self, since_ns: u64) {
        log::warn!(
            target: "reims_vgpu_graphics",
            "a post onto the full control completion ring waited {:.1} ms for room",
            now_ns().saturating_sub(since_ns) as f64 / 1e6
        );
        self.host.ring_degraded(RingDegradation::ControlPostWaited);
    }

    /// The first post of a full spell of the control completion ring found no room for
    /// `message`.
    fn control_ring_filled(&self, message: &[u8]) {
        let ty = RgfxMsgHeader::read_from_prefix(message).map_or(0, |(header, _)| header.r#type);
        log::warn!(
            target: "reims_vgpu_graphics",
            "the control completion ring is full: a message of type {ty:#x} found no room, and \
             ring 0's answers and events wait for the guest to take more"
        );
        self.host.ring_degraded(RingDegradation::ControlRingFull);
    }

    /// How long a post waits for room before it looks again ([`ROOM_POLL`]).
    fn room_poll(&self) -> Duration {
        Duration::from_nanos(self.room_poll_ns.load(Ordering::Relaxed))
    }

    /// Posts `message` into the completion ring, for work begun at reset count `epoch`, and
    /// raises the ring's interrupt where the guest armed it, with no lock held. A full spell of
    /// the ring is told at its first post that finds no room
    /// ([`RingDegradation::ControlRingFull`]).
    fn post(&self, message: &[u8], epoch: u64) -> Post {
        self.post_on(message, epoch, true)
    }

    /// [`Self::post`], which waits for the ring held elsewhere where `wait`, and else is
    /// [`Post::Busy`], as it is where an answer is handed over before it.
    fn post_on(&self, message: &[u8], epoch: u64, wait: bool) -> Post {
        if !wait && self.control_handing.load(Ordering::Acquire) {
            return Post::Busy;
        }
        self.post_telling(message, epoch, wait, true)
    }

    /// [`Self::post_on`], which tells a full spell that begins at this post only where `tells`:
    /// an answer that waits for nothing leaves it to the control thread's post of it, so the
    /// spell's line is never written on the platform's threads.
    fn post_telling(&self, message: &[u8], epoch: u64, wait: bool, tells: bool) -> Post {
        let raise = {
            let Some(mut completion) = lock_ring(&self.completion, wait) else {
                return Post::Busy;
            };
            if self.epoch.load(Ordering::Acquire) != epoch || !self.started() {
                return Post::Dropped;
            }
            match completion.post(&self.region, message) {
                Posted::Yes { raise } => {
                    self.control_full.store(false, Ordering::Relaxed);
                    raise
                }
                Posted::NoRoom => {
                    let began = tells && !self.control_full.swap(true, Ordering::Relaxed);
                    drop(completion);
                    if began {
                        self.control_ring_filled(message);
                    }
                    return Post::NoRoom;
                }
                Posted::Broken(broken) => {
                    log::warn!(
                        target: "reims_vgpu_graphics", kind:% = broken;
                        "the control completion ring is broken ({broken}): an event is dropped"
                    );
                    return Post::Dropped;
                }
            }
        };
        if let Ok((header, _)) = RgfxMsgHeader::read_from_prefix(message) {
            let raised = u64::from(raise);
            (self.host).trace("r0post", &[u64::from(header.r#type), header.seqno, raised]);
        }
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
            self.unroute(queue);
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
        // At debug: reims-vgpu's own line of the close, at info, says what it let go of.
        log::debug!(target: "reims_vgpu_graphics", "context {context} closed");
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
                target: "reims_vgpu_graphics",
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
                target: "reims_vgpu_graphics",
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
    /// `RGFX_SURFACE_PAGES_*`, 0 for one without them; `layout`: the surface's, where the message
    /// asks whether its block needs a device cache.
    pub(crate) fn surface_pages(
        &self,
        pages: &RgfxSurfacePages,
        flags: u32,
        layout: Option<RgfxSurfacePagesLayout>,
        seqno: u64,
        epoch: u64,
    ) -> Option<u32> {
        let Some(runs) = page_runs(&self.region, pages) else {
            log::warn!(
                target: "reims_vgpu_graphics",
                "a SURFACE_PAGES of {} runs at {:#x} for the block of {:#x} bytes at {:#x} lies \
                 outside it: the block gets no view of its surface's pages",
                pages.run_count, pages.runs_offset, pages.block_size, pages.block_offset
            );
            return Some(RGFX_STATUS_INVALID_ARGUMENT);
        };
        let answer = self.complete_answer(seqno, epoch);
        (self.host).surface_pages(pages, runs, flags, layout, answer);
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

    /// A doorbell write of `value`, a ring's id ([`Self::doorbell_value`]): wakes the control
    /// thread (ring id 0) or the service thread of the queue whose submission ring is at that
    /// region offset; an unknown id is ignored.
    fn ring_doorbell(&self, value: u64) {
        let (ring_id, _) = self.doorbell_value(value);
        if ring_id == u64::from(RGFX_CONTROL_RING_ID) {
            self.wake.wake();
            self.room.wake();
            return;
        }
        if let Some(queue) = self.queues.lock().by_ring.get(&ring_id) {
            queue.rung();
        }
    }

    /// The ring a `DOORBELL` write of `value` rings, and whether the guest rang it for a buffer
    /// the device may answer at the doorbell: where the guest's driver took
    /// [`RGFX_FEATURE_DOORBELL_ASKS`], the write that holds [`RGFX_DOORBELL_ASKS`] alone, the rest
    /// of it the ring's id; otherwise any write, as before the feature.
    fn doorbell_value(&self, value: u64) -> (u64, bool) {
        if self.driver_took(RGFX_FEATURE_DOORBELL_ASKS) {
            let asks = u64::from(RGFX_DOORBELL_ASKS);
            (value & !asks, value & asks != 0)
        } else {
            (value, true)
        }
    }

    /// [`Self::ring_doorbell`], after the rung ring's next message is offered to the host on this
    /// thread: a command queue's, where the guest rang for a buffer that asks
    /// ([`Self::doorbell_value`], [`Self::answer_on_doorbell`]), or ring 0's, where it is a
    /// `SURFACE_DOWNLOAD` the host has carried out already ([`Self::download_on_doorbell`]).
    /// The queues' table, held elsewhere, is waited for as `ring_doorbell` waits for it, and
    /// nothing is offered.
    fn ring_taking(&self, value: u64) {
        let (ring_id, asks) = self.doorbell_value(value);
        if ring_id == u64::from(RGFX_CONTROL_RING_ID) {
            self.download_on_doorbell();
            self.wake.wake();
            self.room.wake();
            return;
        }
        let queue = match self.queues.try_lock() {
            Some(queues) => queues.by_ring.get(&ring_id).cloned(),
            None => return self.ring_doorbell(value),
        };
        if let Some(queue) = queue {
            if asks {
                self.answer_on_doorbell(&queue);
            }
            queue.rung();
        }
    }

    /// A `DOORBELL_PAGE_BIND` write of `value` ([`RGFX_REG_DOORBELL_PAGE_BIND`]): binds its page to
    /// its context, or unbinds it for context 0, where the guest's driver took
    /// [`RGFX_FEATURE_DOORBELL_PAGES`]; ignored otherwise.
    fn bind_page(&self, value: u32) {
        if !self.driver_took(RGFX_FEATURE_DOORBELL_PAGES) {
            return;
        }
        let page = value & 0xff;
        let context = value >> RGFX_DOORBELL_PAGE_BIND_CONTEXT_SHIFT;
        let Some(bound) = self.pages.get(page as usize) else {
            return;
        };
        // SeqCst, as a route's look at its page is ([`Self::route`]): a queue routed on the page
        // meanwhile is seen below, or sees the page's new context and drops its route itself.
        if bound.swap(context, Ordering::SeqCst) == context {
            return;
        }
        if context != 0 && !self.pages_bound.swap(true, Ordering::Relaxed) {
            log::info!(
                target: "reims_vgpu_graphics",
                "the guest binds its first doorbell page, {page}, to context {context}: its \
                 processes ring their queues at their pages"
            );
        }
        // The routes on the page of a context it is no longer bound to go, so the host wakes no
        // ring on the page but its new context's: the kext unbinds a page only once its context's
        // queues are gone, and a page bound again under live queues of its old context leaves
        // their doorbells to `DOORBELL`.
        let moved: Vec<Arc<QueueShared>> = self
            .queues
            .lock()
            .by_obj
            .values()
            .filter(|queue| {
                queue.route.load(Ordering::SeqCst) != 0
                    && queue.route_page.load(Ordering::Relaxed) == page
                    && queue.context != context
            })
            .cloned()
            .collect();
        for queue in moved {
            self.unroute(&queue);
        }
    }

    /// The doorbell page bound to `context`, where one is: a queue of the context's is routed
    /// there ([`Self::route`]).
    fn page_of(&self, context: u32) -> Option<u32> {
        if context == 0 || !self.driver_took(RGFX_FEATURE_DOORBELL_PAGES) {
            return None;
        }
        self.pages
            .iter()
            .position(|bound| bound.load(Ordering::Acquire) == context)
            .and_then(|page| u32::try_from(page).ok())
    }

    /// A write of `value` at `offset` in BAR 4 ([`Device::doorbell_write`]).
    fn page_doorbell(&self, offset: u64, value: u32) {
        let page_size = u64::from(RGFX_DOORBELL_PAGE_SIZE);
        let page = offset / page_size;
        let context = self
            .pages
            .get(usize::try_from(page).unwrap_or(usize::MAX))
            .map_or(0, |bound| bound.load(Ordering::Acquire));
        if context == 0 || !offset.is_multiple_of(page_size) {
            self.pages_dropped.fetch_add(1, Ordering::Relaxed);
            log::warn!(
                target: "reims_vgpu_graphics", kind = "unbound";
                "a doorbell page write at {offset:#x} names no bound page's doorbell: dropped"
            );
            return;
        }
        let (ring_id, asks) = self.doorbell_value(u64::from(value));
        // The table tried first, as `ring_taking` tries it: held elsewhere, it is waited for, and
        // nothing is offered.
        let (queue, offer) = match self.queues.try_lock() {
            Some(queues) => (queues.by_ring.get(&ring_id).cloned(), true),
            None => (self.queues.lock().by_ring.get(&ring_id).cloned(), false),
        };
        let foreign = ring_id == u64::from(RGFX_CONTROL_RING_ID)
            || queue.as_ref().is_some_and(|queue| queue.context != context);
        if foreign {
            self.pages_dropped.fetch_add(1, Ordering::Relaxed);
            log::warn!(
                target: "reims_vgpu_graphics", kind = "foreign";
                "doorbell page {page}, context {context}'s: a write of ring {ring_id:#x}, not one \
                 of the context's, is dropped"
            );
            return;
        }
        let Some(queue) = queue else {
            return;
        };
        if asks && offer {
            self.answer_on_doorbell(&queue);
        }
        queue.rung();
    }

    /// Marks `queue`'s submission ring walking while its service thread works on something other
    /// than its take ([`QueueClaim::let_go`]), where the device is started, the ring is not broken
    /// and the queue lives: whether it was marked.
    fn mark_walking(&self, queue: &QueueShared) -> bool {
        let submit = queue.submit.lock();
        if !self.started() || submit.broken || self.ending() || queue.ended() {
            return false;
        }
        submit.consumer.mark_walking(&self.region);
        true
    }

    /// Clears the walking mark of [`Self::mark_walking`] where the queue still lives: an ended queue's
    /// rings may be another process's block by now ([`QueueShared::fence`]).
    fn mark_awake(&self, queue: &QueueShared) {
        let submit = queue.submit.lock();
        if !queue.ended() {
            submit.consumer.awake(&self.region);
        }
    }

    /// Takes ring 0's next message on this thread, a doorbell's (the module doc), and answers it
    /// here, where it is a `SURFACE_DOWNLOAD` the host has carried out already
    /// ([`Host::surface_downloaded`]): its `COMPLETE` is posted, and the control vector raised
    /// with no lock held, before the doorbell write returns. Only where nothing need be waited
    /// for, and in ring 0's order, as the control thread would take it ([`Self::download_here`]):
    /// the device is started, not reset since the host was last told, and the ring not broken;
    /// the control thread sleeps with every answer of what it took posted (`asleep`), so the
    /// download is the next message after all it carried out, and no answer of its comes after
    /// this one; no answer given elsewhere waits to be posted ([`CompleteAnswer::answer`]); the
    /// block's lane has nothing left; the control completion ring has room; and
    /// each lock is tried, the control submission ring's, the completion ring's and the lanes'.
    /// Otherwise the message stays, and the control thread, which the write wakes, takes it as
    /// it takes any. No buffer waits for a download's count (`backlog`: a download is no
    /// buffer's), so the queue gate is not told of it. Allocates and frees nothing.
    fn download_on_doorbell(&self) {
        let raise = {
            let Some(mut submit) = self.submit.try_lock() else {
                return;
            };
            if !submit.asleep
                || !self.started()
                || submit.broken
                || self.ending()
                || self.reset_owed.load(Ordering::Acquire)
                // Its answer goes behind the answers handed over, which the control thread posts.
                || self.control_handing.load(Ordering::Acquire)
            {
                return;
            }
            let epoch = self.epoch.load(Ordering::Acquire);
            let mut words = [0u64; size_of::<RgfxSurfaceDownload>().div_ceil(8)];
            let bytes = words.as_mut_bytes();
            let Ok(Some(header)) = submit.consumer.peek_into(&self.region, bytes) else {
                return;
            };
            if header.r#type != RGFX_CMD_SURFACE_DOWNLOAD || header.size as usize > bytes.len() {
                return;
            }
            let Ok((download, _)) = RgfxSurfaceDownload::read_from_prefix(bytes) else {
                return;
            };
            if self.lanes.try_lane_idle(Key::Block(download.block_offset)) != Some(true) {
                return;
            }
            let Some(mut completion) = self.completion.try_lock() else {
                return;
            };
            // As a post checks it (`post_on`): a reset under way drops the answer.
            if self.epoch.load(Ordering::Acquire) != epoch || !self.started() {
                return;
            }
            let answer = RgfxComplete {
                header: RgfxMsgHeader {
                    r#type: RGFX_EVT_COMPLETE,
                    size: size_of::<RgfxComplete>() as u32,
                    seqno: header.seqno,
                },
                status: RGFX_STATUS_OK,
                reserved_14: 0,
            };
            if !completion.has_room(&self.region, answer.as_bytes().len())
                || !self.host.surface_downloaded(download.block_offset)
            {
                return;
            }
            submit.consumer.take_header(&self.region, &header);
            let through = submit.consumer.consumed();
            submit.backlog.took_unowned(through);
            self.control_done.store(through, Ordering::Release);
            match completion.post(&self.region, answer.as_bytes()) {
                Posted::Yes { raise } => raise,
                Posted::NoRoom | Posted::Broken(_) => false,
            }
        };
        if raise {
            self.host.raise_interrupt(RGFX_CONTROL_VECTOR);
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
                target: "reims_vgpu_graphics",
                "queue {:#x}: it asks for flags {refused_flags:#x}, which the host does not \
                 carry out or whose feature the driver did not take, so it is refused", q.obj_id
            );
            return RGFX_STATUS_INVALID_ARGUMENT;
        }
        let Some((submit_at, complete_at)) = self.validate_rings(q) else {
            log::warn!(
                target: "reims_vgpu_graphics",
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
                target: "reims_vgpu_graphics",
                "queue {:#x}: no wake pipe; it is refused", q.obj_id
            );
            return RGFX_STATUS_INVALID_ARGUMENT;
        };
        // A guest that rings an asking buffer's doorbell apart from the rest lets the host take
        // the rest where it runs nothing of the device's ([`Host::route_doorbell`]).
        let bell = if self.driver_took(RGFX_FEATURE_DOORBELL_ASKS) {
            wake::bell()
        } else {
            None
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
                asleep: false,
            }),
            taken_seqno: AtomicU64::new(0),
            consuming: Mutex::new(None),
            complete: Mutex::new(producer),
            room: Room::default(),
            full: AtomicBool::new(false),
            untold: Mutex::new(Untold::default()),
            handed: Mutex::new(Handed::default()),
            handing: AtomicBool::new(false),
            wake,
            bell,
            route: AtomicU64::new(0),
            route_page: AtomicU32::new(u32::MAX),
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
            self.unroute(old);
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
            self.unroute(&queue);
            log::error!(
                target: "reims_vgpu_graphics",
                "queue {:#x}: no service thread; it is refused", q.obj_id
            );
            return RGFX_STATUS_INVALID_ARGUMENT;
        }
        self.route(&queue);
        // Before the answer: a buffer of another queue may name this one's from then on.
        self.host.queue_made(&self.queue_poster(&queue));
        RGFX_STATUS_OK
    }

    /// Has the host route `queue`'s plain doorbell into its bell ([`Host::route_doorbell`]), where
    /// it has one, while the device routes fewer than [`MAX_DOORBELL_ROUTES`]. A queue past them,
    /// or whose route the host refuses, keeps its doorbell on the device's write, as every queue
    /// had it, and the first of each is logged, the rest counted. A queue ended meanwhile drops
    /// the route it took at once.
    fn route(&self, queue: &QueueShared) {
        let Some(bell) = &queue.bell else {
            return;
        };
        // The guest writes a ring's id as 32 bits: a ring past them is never rung.
        let Ok(ring_id) = u32::try_from(queue.ring_id) else {
            return;
        };
        // On its context's doorbell page where it has one, which its process writes: a host that
        // routes no page leaves the queue's writes to the page's write.
        let at = match self.page_of(queue.context) {
            Some(page) if self.host.routes_doorbell_pages() => DoorbellAt::Page(page),
            Some(_) => return,
            None => DoorbellAt::Register,
        };
        if self
            .routes
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < MAX_DOORBELL_ROUTES).then_some(n + 1)
            })
            .is_err()
        {
            log::warn!(
                target: "reims_vgpu_graphics", kind = "full";
                "queue {:#x}: the device routes {MAX_DOORBELL_ROUTES} queues' doorbells to the \
                 host already, so this one's runs on the guest's vCPU", queue.obj_id
            );
            return;
        }
        let route = self.last_route.fetch_add(1, Ordering::AcqRel) + 1;
        if !self
            .host
            .route_doorbell(route, at, ring_id, wake::raw(bell))
        {
            self.routes.fetch_sub(1, Ordering::AcqRel);
            log::warn!(
                target: "reims_vgpu_graphics", kind = "refused";
                "queue {:#x}: the host refused its doorbell's route, so its doorbell runs on the \
                 guest's vCPU", queue.obj_id
            );
            return;
        }
        queue.route_page.store(
            match at {
                DoorbellAt::Register => u32::MAX,
                DoorbellAt::Page(page) => page,
            },
            Ordering::Relaxed,
        );
        // SeqCst, as the end's store is ([`QueueShared::end`]) and a bind's ([`Self::bind_page`]):
        // a queue that ended meanwhile, or a page bound to another context, is seen here, or its
        // ender or binder sees the route.
        queue.route.store(route, Ordering::SeqCst);
        let moved = match at {
            DoorbellAt::Register => false,
            DoorbellAt::Page(page) => {
                self.pages[page as usize].load(Ordering::SeqCst) != queue.context
            }
        };
        if moved || queue.ending.load(Ordering::SeqCst) {
            self.unroute(queue);
        }
    }

    /// Drops the route of `queue`'s doorbell the host took ([`Host::unroute_doorbell`]), once,
    /// where it took one; called once the queue has ended.
    fn unroute(&self, queue: &QueueShared) {
        let route = queue.route.swap(0, Ordering::SeqCst);
        if route != 0 {
            let at = match queue.route_page.load(Ordering::Relaxed) {
                u32::MAX => DoorbellAt::Register,
                page => DoorbellAt::Page(page),
            };
            // A routed ring's id is 32 bits ([`Self::route`]).
            self.host.unroute_doorbell(route, at, queue.ring_id as u32);
            self.routes.fetch_sub(1, Ordering::AcqRel);
        }
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
        self.unroute(&queue);
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
        // Nothing is posted onto its ring any more: what follows the records handed over is run.
        self.close_handed(queue);
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
            // The records handed over to it, posted first, as far as its ring has room: those
            // left wait for its next look, while it takes the guest's buffers meanwhile.
            if queue.handing.load(Ordering::Acquire) {
                self.post_handed(queue);
            }
            self.tell_queue_ring(queue);
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
            // On its bell too, which the host makes readable for a doorbell it routes: the
            // writes it holds wake the thread once, and the take finds every buffer they rang for.
            // With records handed over left, it looks again after the device's poll, as a post
            // waiting for room does: the guest's take of its completions rings nothing.
            let poll = queue
                .handing
                .load(Ordering::Acquire)
                .then(|| self.room_poll());
            queue
                .wake
                .sleep_for(queue.bell.as_ref().map(wake::raw), poll);
            // A doorbell may follow the guest's take of completions, as the control thread's
            // may: a routed one rings nothing of the device's but the bell.
            queue.room.wake();
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
        // Where the guest stamps its buffers ([`guest_times`]): when this thread came back to the
        // ring after the buffer it handed over before, on the host's clock; 0 before its first.
        let guest_times = self.driver_took(RGFX_FEATURE_GUEST_TIMES);
        let mut ring_free_ns = 0;
        loop {
            let (next, room, taken_ns) = {
                let mut submit = queue.submit.lock();
                if !self.started() || submit.broken || self.ending() || queue.ended() {
                    return;
                }
                let next = submit.consumer.next(&self.region);
                // The buffer's take, before it waits at the gate.
                let taken_ns = if guest_times { now_ns() } else { 0 };
                // A commit waiting for room: after this take, a doorbell's thread's before it, or
                // the last, where the ring is empty.
                let room = next.is_ok() && self.room_wanted(&submit.consumer);
                if let Ok(Some(message)) = &next {
                    queue.took(message.header.seqno);
                }
                if let Err(broken) = &next {
                    log::error!(
                        target: "reims_vgpu_graphics", kind:% = broken;
                        "queue {:#x}'s submission ring is broken ({broken}): it takes nothing \
                         more until the device is reset", queue.obj_id
                    );
                    submit.broken = true;
                }
                (next, room, taken_ns)
            };
            // Told with nothing held, on the queue's completion ring.
            if room {
                log::debug!(
                    target: "reims_vgpu_graphics::room",
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
                // A thread waits for a buffer of the queue handed over before it to be
                // scheduled: handed to the host at once, with the queue's claim held, as the
                // buffer it names is begun.
                Ok(Some(message))
                    if message.header.r#type == RGFX_CMD_SCHEDULED_ASK
                        && self.driver_features.load(Ordering::Acquire)
                            & RGFX_FEATURE_SCHEDULED_ASK
                            != 0 =>
                {
                    self.host.scheduled_ask(poster, &message.bytes);
                    continue;
                }
                Ok(Some(message)) => {
                    // A queue's ring carries only its process's command buffers, and the asks
                    // of its threads that wait for them; every other message rides ring 0,
                    // through the kext, which checks that its sender owns the object it names.
                    // Refused here, and the ring goes on.
                    log::warn!(
                        target: "reims_vgpu_graphics",
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
            // queue's end or the device's ([`gate`]), but carries out itself a job of the lane's
            // it waits for that no thread of the lane's carries out yet, where it may
            // ([`Self::helps`]). A buffer the host answered on a doorbell's
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
            // A job of its context's lane it waits for, which no thread of the lane's carries out
            // yet, it carries out itself, in place of sleeping while a ring-0 thread is woken for
            // it ([`Self::help_lane`]).
            while self.gate.wait_offering(
                || {
                    if self.ending()
                        || queue.ended()
                        || self.epoch.load(Ordering::Acquire) != queue.epoch
                    {
                        Look::Done
                    } else if (self.lanes).has_help(queue.context, own, |m| self.helps(m)) {
                        Look::Help
                    } else if self.control_done.load(Ordering::Acquire) < waits_for {
                        Look::Count(waits_for)
                    } else if !self.lanes.through(queue.context, own) {
                        Look::Lane(queue.context, own)
                    } else {
                        Look::Done
                    }
                },
                offer.as_ref(),
            ) == Waited::Help
            {
                self.help_lane(queue.context, own);
            }
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
                            target: "reims_vgpu_graphics",
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
            if guest_times {
                self.note_guest_times(queue, &message, taken_ns, ring_free_ns);
            }
            // The host may let the claim go once the buffer is begun ([`QueueClaim`]); a buffer
            // a doorbell's thread answered meanwhile is the ring's next, which this takes next.
            let mut claim = QueueClaim {
                shared: self,
                queue,
                held: Some(consuming),
                walking: false,
            };
            self.host.command_buffer(poster, message.bytes, &mut claim);
            consuming = claim.retake();
            if guest_times {
                ring_free_ns = now_ns();
            }
        }
    }

    /// The guest's times of the `COMMAND_BUFFER` `message` ([`guest_times`]), taken off `queue`'s
    /// ring at `taken_ns` by a thread that came back to the ring at `ring_free_ns`: its `cbguest`
    /// event in the host's trace, and its spans in the once-a-second line where its target logs.
    /// Nothing for a buffer whose stream does not start with its times.
    fn note_guest_times(
        &self,
        queue: &QueueShared,
        message: &Message,
        taken_ns: u64,
        ring_free_ns: u64,
    ) {
        let Some(times) = guest_times::of(&message.bytes) else {
            return;
        };
        let clock = guest_times::clock(&self.region);
        let noted_ns = now_ns();
        let spans = guest_times::spans(&times, clock, taken_ns, ring_free_ns, noted_ns);
        self.host
            .trace("cbguest", &spans.event(queue.obj_id, message.header.seqno));
        if log::log_enabled!(target: guest_times::TARGET, log::Level::Debug)
            && let Some(line) = self.guest_times.lock().add(&spans, noted_ns)
        {
            log::debug!(target: guest_times::TARGET, "{line}");
        }
    }

    /// On the thread of a queue of context `context` whose command buffer waits for the context's
    /// lane through ring 0's count `own`: carries out the lane's front job, where the lane's thread
    /// carries none out and it is one the buffer waits for and [`Self::helps`] takes
    /// ([`Lanes::help`]), as the lane's thread would ([`carry_out_job`]).
    fn help_lane(&self, context: u32, own: u64) {
        if let Some(job) = self.lanes.help(context, own, |m| self.helps(m)) {
            carry_out_job(&self.lanes, self, Key::Context(context), job);
        }
    }

    /// Whether a command buffer's thread may carry out ring-0 `message` of its context's lane
    /// ([`Lanes::help`]): a `BUFFER_AT_OFFSET`, or an `OBJECT_DESTROY` of no queue, which the host
    /// carries out with no thread of the device's to wait for. A queue's destroy routes its
    /// doorbell (the host's lock) and ends a queue's thread, and a close, a block's release or a
    /// create the host makes elsewhere may wait long or for a queue's buffers: each is left to
    /// the lane's thread. The bundle holds these two kinds and posts them in one call ahead of
    /// each command buffer it commits (docs/graphics-kext/design.md, "Destroys and creates no
    /// one waits for go together"), so they are what a buffer finds left on its lane.
    fn helps(&self, message: &Message) -> bool {
        match message.header.r#type {
            RGFX_CMD_BUFFER_AT_OFFSET => true,
            RGFX_CMD_OBJECT_DESTROY => RgfxObjectDestroy::read_from_prefix(&message.bytes)
                .is_ok_and(|(m, _)| !self.queues.lock().by_obj.contains_key(&m.obj_id)),
            _ => false,
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
    /// Behind records handed over before it, it is handed over too, posted in its turn
    /// ([`Self::post_behind_handed`]), and waits for nothing.
    fn post_queue_waiting(&self, queue: &QueueShared, message: &[u8], epoch: u64) {
        if queue.handing.load(Ordering::Acquire) && self.post_behind_handed(queue, message, epoch) {
            return;
        }
        self.post_queue_now(queue, message, epoch);
    }

    /// [`Self::post_queue_waiting`] of `message` with no record handed over before it. A full
    /// spell of the ring is told at its first post that found no room, and each post that
    /// waited once it goes on ([`RingDegradation::QueueRingFull`],
    /// [`RingDegradation::QueuePostWaited`]).
    fn post_queue_now(&self, queue: &QueueShared, message: &[u8], epoch: u64) {
        let mut waited_since = None;
        loop {
            let seen = queue.room.wakes();
            if self.post_queue(queue, message, epoch) != Post::NoRoom
                || self.ending()
                || queue.ended()
            {
                if let Some(since) = waited_since {
                    self.queue_post_waited(queue, since);
                }
                return;
            }
            if waited_since.is_none() {
                waited_since = Some(now_ns());
                if !queue.full.swap(true, Ordering::Relaxed) {
                    self.queue_ring_filled(queue, message);
                }
            }
            queue.room.wait(seen, self.room_poll());
        }
    }

    /// The first post that waits for room in a full spell of `queue`'s completion ring found
    /// none for `message`. The line names the thread that waits, as a wait on a thread that runs
    /// the platform's handlers holds up the work behind them there.
    fn queue_ring_filled(&self, queue: &QueueShared, message: &[u8]) {
        let ty = RgfxMsgHeader::read_from_prefix(message).map_or(0, |(header, _)| header.r#type);
        let thread = std::thread::current();
        log::warn!(
            target: "reims_vgpu_graphics",
            "queue {:#x}'s completion ring is full: a message of type {ty:#x} found no room, and \
             its posts that wait for room wait for the guest to take more, on {}",
            queue.obj_id,
            thread.name().unwrap_or("an unnamed thread")
        );
        self.host.ring_degraded(RingDegradation::QueueRingFull);
    }

    /// A post onto `queue`'s completion ring that waited for room since `since_ns` ([`now_ns`])
    /// went on.
    fn queue_post_waited(&self, queue: &QueueShared, since_ns: u64) {
        let thread = std::thread::current();
        log::warn!(
            target: "reims_vgpu_graphics",
            "a post onto queue {:#x}'s full completion ring waited {:.1} ms for room, on {}",
            queue.obj_id,
            now_ns().saturating_sub(since_ns) as f64 / 1e6,
            thread.name().unwrap_or("an unnamed thread")
        );
        self.host.ring_degraded(RingDegradation::QueuePostWaited);
    }

    /// Posts `message` into `queue`'s completion ring, for work begun at reset count `epoch`, and
    /// raises the queue's MSI-X vector where the guest armed it, with no lock held.
    fn post_queue(&self, queue: &QueueShared, message: &[u8], epoch: u64) -> Post {
        self.post_queue_on(queue, message, epoch, true)
    }

    /// [`Self::post_queue`], which waits for the ring held elsewhere where `wait`, and else is
    /// [`Post::Busy`].
    fn post_queue_on(&self, queue: &QueueShared, message: &[u8], epoch: u64, wait: bool) -> Post {
        // A post that waits for nothing goes behind no record handed over: busy until they are
        // posted.
        if !wait && queue.handing.load(Ordering::Acquire) {
            return Post::Busy;
        }
        let raise = {
            let Some(mut completion) = lock_ring(&queue.complete, wait) else {
                return Post::Busy;
            };
            if self.epoch.load(Ordering::Acquire) != epoch || !self.started() || queue.ended() {
                return Post::Dropped;
            }
            match completion.post(&self.region, message) {
                Posted::Yes { raise } => {
                    queue.full.store(false, Ordering::Relaxed);
                    raise
                }
                Posted::NoRoom => return Post::NoRoom,
                Posted::Broken(broken) => {
                    log::warn!(
                        target: "reims_vgpu_graphics", kind:% = broken;
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

    /// Display `display`'s refresh thread (the module doc): a refresh at each tick of its grid
    /// ([`RefreshGrid`]), stamped with the tick's time however late the thread woke for it.
    fn refresh_thread(&self, display: u32) {
        let Some(screen) = self.displays.get(display as usize).map(|d| &d.screen) else {
            return;
        };
        let mut grid = RefreshGrid::new(now_ns());
        let part = u64::try_from(REFRESH_SLEEP_PART.as_nanos()).unwrap_or(u64::MAX);
        while !self.ending() {
            let now = now_ns();
            let tick = grid.next(screen.mode().period_ns(), now);
            let mut left = tick.saturating_sub(now);
            while left > part {
                std::thread::sleep(REFRESH_SLEEP_PART);
                if self.ending() {
                    return;
                }
                left = tick.saturating_sub(now_ns());
            }
            if left > 0 {
                std::thread::sleep(Duration::from_nanos(left));
            }
            if self.ending() {
                break;
            }
            self.refresh(display, tick);
        }
    }

    /// One refresh of display `id`, at `now` on the monotonic clock. Its events go to a guest
    /// that sees it ([`Self::displays_seen`]) alone: a guest that did not take
    /// [`RGFX_FEATURE_MULTI_DISPLAY`] gets none of a display but 0's.
    pub(crate) fn refresh(&self, id: u32, now: u64) {
        let Some(display) = self.displays.get(id as usize) else {
            return;
        };
        let epoch = self.epoch.load(Ordering::Acquire);
        let mode = self.offered_mode(display);
        let seen = id < self.displays_seen();
        let (scanout, vblank, changed) = {
            let mut display = display.state.lock();
            display.refreshes = display.refreshes.wrapping_add(1);
            if display.told.is_some_and(|told| told != mode) {
                display.told = Some(mode);
                display.changed = true;
            }
            (
                display.scanout,
                (display.vblank && seen).then_some(display.refreshes),
                display.changed && seen,
            )
        };
        if scanout.is_some() {
            display.screen.present();
        }
        if let Some(count) = vblank {
            let event = RgfxVblank {
                header: event_header(RGFX_EVT_VBLANK),
                display: id,
                reserved_14: 0,
                count,
                host_time_ns: now,
            };
            // One the ring has no room for is dropped, as is one that finds answers handed over
            // before it, which wait for room; the next carries the count.
            let posted = if self.control_handing.load(Ordering::Acquire) {
                Post::NoRoom
            } else {
                self.post(&sized(&event), epoch)
            };
            if posted == Post::NoRoom {
                log::warn!(
                    target: "reims_vgpu_graphics",
                    "the VBLANK of refresh {count} found the control completion ring full and \
                     was dropped: the next carries the count"
                );
                self.host.ring_degraded(RingDegradation::VblankDropped);
            }
        }
        if changed {
            let event = RgfxDisplayChanged {
                header: event_header(RGFX_EVT_DISPLAY_CHANGED),
                display: id,
                reserved_14: 0,
            };
            // Told again at the next refresh where answers handed over wait before it.
            if !self.control_handing.load(Ordering::Acquire)
                && self.post(&sized(&event), epoch) == Post::Done
            {
                display.state.lock().changed = false;
            }
        }
        // A POINTER_OVER the window's thread could not post.
        self.post_pointer(epoch, true);
        self.host.refreshed(id);
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

/// The runs a `SURFACE_PAGES` lists, read out of its block in `region`, where the block lies in
/// the processes' blocks and its runs, whole, in the block; else `None`.
fn page_runs(region: &Region, pages: &RgfxSurfacePages) -> Option<Vec<RgfxPageRun>> {
    let end = region.size().saturating_sub(RGFX_CURSOR_SPAN);
    let block_end = pages.block_offset.checked_add(pages.block_size);
    let runs_bytes = u64::from(pages.run_count) * size_of::<RgfxPageRun>() as u64;
    let runs_end = pages.runs_offset.checked_add(runs_bytes);
    let fits = pages.block_offset >= RGFX_CONTEXT_ALLOCATIONS_OFFSET
        && block_end.is_some_and(|e| e <= end)
        && pages.runs_offset >= pages.block_offset
        && pages.runs_offset.is_multiple_of(8)
        && runs_end.zip(block_end).is_some_and(|(r, b)| r <= b);
    if !fits {
        return None;
    }
    let bytes = region.read(pages.runs_offset, usize::try_from(runs_bytes).ok()?)?;
    let runs = bytes
        .as_chunks::<{ size_of::<RgfxPageRun>() }>()
        .0
        .iter()
        .filter_map(|run| RgfxPageRun::read_from_bytes(run).ok())
        .collect();
    Some(runs)
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
        carry_out_job(lanes, &shared, key, job);
    }
}

/// Carries out `job`, which the lane `key` gave this thread ([`Lanes::next`], or [`Lanes::help`]
/// on a command buffer's thread), with no lock of the device's held: tells the lanes it is done,
/// offers the host each command buffer its carrying out ends the wait of at the gate, then posts
/// its answer. A job taken before a reset is not carried out.
fn carry_out_job(lanes: &Lanes, shared: &Shared, key: Key, job: lanes::Job) {
    let seqno = job.message.header.seqno;
    (shared.host).trace("r0job", &[u64::from(job.message.header.r#type), seqno]);
    let answer = if shared.epoch.load(Ordering::Acquire) == job.epoch {
        control::unheld(shared, &job.message, job.epoch)
    } else {
        None
    };
    (shared.host).trace("r0jobdone", &[seqno]);
    if let Some((context, done)) = lanes.finished(key) {
        (shared.gate).carried_out_offering(context, done, |offer| shared.answer_offered(offer));
    }
    if let Some(answer) = answer {
        shared.post_waiting(&answer, job.epoch);
    }
}

impl Shared {
    /// Posts `message`, for work begun at reset count `epoch`, onto the control completion ring,
    /// waiting for nothing ([`CompleteAnswer::answer`]): here where no answer is handed over before
    /// it and the ring has room, else handed to the control thread behind those, which it wakes
    /// unless a thread is posting them already.
    fn post_unwaiting(&self, message: &[u8], epoch: u64) {
        if !self.control_handing.load(Ordering::Acquire)
            && self.post_telling(message, epoch, true, false) != Post::NoRoom
        {
            return;
        }
        let wake = {
            let mut handed = self.control_handed.lock();
            handed.records.push_back(HandedRecord {
                message: message.to_vec(),
                epoch,
                at: now_ns(),
                then: None,
                no_room_since: 0,
            });
            self.control_handing.store(true, Ordering::Release);
            !handed.posting
        };
        if wake {
            self.wake.wake();
        }
    }

    /// [`Self::post_behind_handed`] on the control completion ring: `message` (where given)
    /// joins the answers handed over where a thread is posting them; else this thread posts them,
    /// oldest first, then it, waiting for room. False, with nothing done, where none is held.
    fn post_behind_control_handed(&self, message: Option<(&[u8], u64)>) -> bool {
        {
            let mut handed = self.control_handed.lock();
            if handed.records.is_empty() && !handed.posting {
                self.control_handing.store(false, Ordering::Release);
                return false;
            }
            if let Some((message, epoch)) = message {
                handed.records.push_back(HandedRecord {
                    message: message.to_vec(),
                    epoch,
                    at: 0,
                    then: None,
                    no_room_since: 0,
                });
            }
            if handed.posting {
                return true;
            }
            handed.posting = true;
        }
        loop {
            let record = {
                let mut handed = self.control_handed.lock();
                match handed.records.pop_front() {
                    Some(next) => next,
                    None => {
                        handed.posting = false;
                        self.control_handing.store(false, Ordering::Release);
                        return true;
                    }
                }
            };
            self.post_waiting_now(&record.message, record.epoch);
        }
    }

    /// Hands `message`, posted at reset count `epoch`, to `queue`'s service thread to post
    /// ([`CompletionPoster::post_elsewhere`]), after the records handed over before it, and wakes
    /// that thread unless a thread is posting them already. Waits for nothing.
    fn hand_over(&self, queue: &QueueShared, message: &[u8], epoch: u64) {
        self.hand_over_then(queue, message, epoch, None, 0);
    }

    /// [`Self::hand_over`], `then` run once `message` is posted or dropped ([`Then`]), and, where
    /// the post that handed it found the ring with no room, since when (`no_room_since`, 0 where
    /// it did not look); dropped at once, and `then` run here, where the queue's service thread
    /// has ended.
    fn hand_over_then(
        &self,
        queue: &QueueShared,
        message: &[u8],
        epoch: u64,
        then: Option<Then>,
        no_room_since: u64,
    ) {
        let wake = {
            let mut handed = queue.handed.lock();
            if handed.closed {
                drop(handed);
                return run(then);
            }
            handed.records.push_back(HandedRecord {
                message: message.to_vec(),
                epoch,
                at: now_ns(),
                then,
                no_room_since,
            });
            queue.handing.store(true, Ordering::Release);
            !handed.posting
        };
        self.handed_over.fetch_add(1, Ordering::Relaxed);
        if wake {
            queue.wake.wake();
        }
    }

    /// [`CompletionPoster::post_unwaiting`] onto `queue`'s ring: posted here where no record is
    /// handed over before it and the ring has room; else handed over behind them, and as many of
    /// them as the ring takes now posted here, in order ([`Self::post_handed`]).
    fn post_queue_unwaiting(
        &self,
        queue: &QueueShared,
        message: &[u8],
        epoch: u64,
        then: Option<Then>,
    ) {
        if queue.handing.load(Ordering::Acquire) {
            self.hand_over_then(queue, message, epoch, then, behind_since(queue));
            if !self.post_handed(queue) {
                queue.wake.wake();
            }
            return;
        }
        if self.post_queue(queue, message, epoch) != Post::NoRoom {
            return run(then);
        }
        self.note_no_room(queue);
        self.hand_over_then(queue, message, epoch, then, now_ns());
    }

    /// `queue`'s service thread has ended: no record handed over from now on is posted
    /// ([`Handed::closed`]), and those left are dropped, what follows each run here, in their
    /// order. A thread posting them still runs what follows those it took.
    fn close_handed(&self, queue: &QueueShared) {
        let left = {
            let mut handed = queue.handed.lock();
            handed.closed = true;
            if !handed.posting {
                queue.handing.store(false, Ordering::Release);
            }
            std::mem::take(&mut handed.records)
        };
        for record in left {
            run(record.then);
        }
    }

    /// Hands `message` over behind the records handed over on `queue`'s ring, for a post that
    /// may wait, and posts as many of them as the ring takes now, in order, waking the service
    /// thread for the rest. False, with nothing done, where none is held any more: the caller
    /// posts it at once.
    fn post_behind_handed(&self, queue: &QueueShared, message: &[u8], epoch: u64) -> bool {
        {
            let mut handed = queue.handed.lock();
            if handed.records.is_empty() && !handed.posting {
                queue.handing.store(false, Ordering::Release);
                return false;
            }
            handed.records.push_back(HandedRecord {
                message: message.to_vec(),
                epoch,
                at: 0,
                then: None,
                no_room_since: behind_since(queue),
            });
        }
        if !self.post_handed(queue) {
            queue.wake.wake();
        }
        true
    }

    /// Posts the records handed over on `queue`'s ring, oldest first, as far as the ring has room
    /// now, waiting for no room, with no lock held across a post, and runs what follows each
    /// right after it ([`Then`]); one posted after a reset since it was handed over is dropped as a
    /// post drops it. True where none is left; false where one found no room (it is kept, first,
    /// for a later look) or another thread is posting them (which posts any handed over meanwhile
    /// in its turn). A record that found no room, and its spell of the ring, are noted for the
    /// service thread to tell ([`Self::tell_queue_ring`]).
    fn post_handed(&self, queue: &QueueShared) -> bool {
        {
            let mut handed = queue.handed.lock();
            if handed.posting {
                return false;
            }
            if handed.records.is_empty() {
                queue.handing.store(false, Ordering::Release);
                return true;
            }
            handed.posting = true;
        }
        loop {
            let mut record = {
                let mut handed = queue.handed.lock();
                match handed.records.pop_front() {
                    Some(next) => next,
                    None => {
                        handed.posting = false;
                        queue.handing.store(false, Ordering::Release);
                        return true;
                    }
                }
            };
            if self.post_queue(queue, &record.message, record.epoch) == Post::NoRoom {
                if record.no_room_since == 0 {
                    record.no_room_since = now_ns();
                    self.note_no_room(queue);
                }
                let mut handed = queue.handed.lock();
                handed.records.push_front(record);
                handed.posting = false;
                return false;
            }
            if record.at != 0 {
                let waited = now_ns().saturating_sub(record.at);
                self.handed_wait_ns.fetch_max(waited, Ordering::Relaxed);
            }
            if record.no_room_since != 0 {
                let mut untold = queue.untold.lock();
                untold.waits += 1;
                untold.longest_ns = untold
                    .longest_ns
                    .max(now_ns().saturating_sub(record.no_room_since));
            }
            run(record.then);
        }
    }

    /// A post found `queue`'s completion ring with no room and handed its record over: where the
    /// ring had room at its post before, a full spell begins, noted for the service thread to
    /// tell ([`Self::tell_queue_ring`]).
    fn note_no_room(&self, queue: &QueueShared) {
        if !queue.full.swap(true, Ordering::Relaxed) {
            queue.untold.lock().spells += 1;
        }
    }

    /// On `queue`'s service thread: tells the host of the full spells and the posts that waited
    /// for room noted since it last told ([`RingDegradation::QueueRingFull`],
    /// [`RingDegradation::QueuePostWaited`]), with a line each, and at most once a second how many
    /// records were handed over in all, and the longest one waited for a thread to post it since
    /// the last such line: the process's first at info, a path a guest's load takes, and the later
    /// ones at debug. Told here, not where they were noted: the platform's threads that hand
    /// records over write no line.
    fn tell_queue_ring(&self, queue: &QueueShared) {
        let untold = std::mem::take(&mut *queue.untold.lock());
        if untold.spells > 0 {
            log::warn!(
                target: "reims_vgpu_graphics",
                "queue {:#x}'s completion ring is full: posts that wait for nothing hand their \
                 records to its service thread, which posts them as the guest takes more",
                queue.obj_id
            );
            for _ in 0..untold.spells {
                self.host.ring_degraded(RingDegradation::QueueRingFull);
            }
        }
        if untold.waits > 0 {
            log::warn!(
                target: "reims_vgpu_graphics",
                "{} records handed over onto queue {:#x}'s full completion ring waited for room, \
                 the longest {:.1} ms",
                untold.waits,
                queue.obj_id,
                untold.longest_ns as f64 / 1e6
            );
            for _ in 0..untold.waits {
                self.host.ring_degraded(RingDegradation::QueuePostWaited);
            }
        }
        let now = now_ns();
        let last = self.handed_logged_ns.load(Ordering::Relaxed);
        let over = self.handed_over.load(Ordering::Relaxed);
        if over > 0
            && now.saturating_sub(last) >= 1_000_000_000
            && self
                .handed_logged_ns
                .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            let waited = self.handed_wait_ns.swap(0, Ordering::Relaxed);
            let level = if last == 0 {
                log::Level::Info
            } else {
                log::Level::Debug
            };
            log::log!(
                target: "reims_vgpu_graphics::handed",
                level,
                "{over} completion records handed over to the queues' threads in all, from \
                 threads that may not post them or wait for room; the longest wait for a thread \
                 to post one since the last line: {} us",
                waited / 1_000
            );
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

/// The centre of pixel `at` of a window's `mode`, as a place over a scanout of `pixels`: the
/// window's mode is the scanout's but while the two change.
fn pixel_centre(at: (u16, u16), mode: (u16, u16), pixels: (u32, u32)) -> (f64, f64) {
    (
        (f64::from(at.0) + 0.5) * f64::from(pixels.0) / f64::from(mode.0),
        (f64::from(at.1) + 0.5) * f64::from(pixels.1) / f64::from(mode.1),
    )
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

#[cfg(test)]
pub(crate) mod tests;
