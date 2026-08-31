//! The process-wide registry of bound devices, and the entry surface
//! `crate::qemu::abi` wraps.
//!
//! One `BoundDevice` per QEMU device, keyed by the opaque handle the C shim
//! holds, plus the twelve functions that shim calls: create/reset/destroy, the
//! two MMIO windows, drain, poll, the action queue, and the backend name. The
//! locking policy lives here too, because it is a property of this map rather
//! than of any one device — `lock_for_drain` and `lock_device_for_vcpu` are two
//! answers to "who may block whom" and are chosen per caller. (Named in prose:
//! a bare name in a `//!` doc resolves to nothing whatever it names.)
//!
//! Split out of the crate root, which was 800 lines of which 690 were this. The
//! root is the module table and the three-arm guards now, and the two sibling
//! entry-surface modules — `display_surface` and `window_publish` — already had
//! this shape: a private `mod`, with the names `crate::qemu::abi` reaches re-exported
//! at the root.

/// The device side of that window: its link, its four QEMU entry points, and
/// the two publish paths the drain and the poll call. Unconditional, because
/// the entry points keep a stub arm without the feature and the QEMU ABI
/// surface must be the same shape either way.
mod window_publish;
pub(crate) use window_publish::{
    device_window_run_main, device_window_set_early_fb, device_window_start, device_window_stop,
};

/// The display half of the QEMU ABI surface: console-feed ownership, scanout
/// and EFI-console copies, and the cursor glyph.
mod display_surface;
#[cfg(any(test, feature = "host-window"))]
pub(crate) use display_surface::host_console_uses_bar1;
pub(crate) use display_surface::{
    device_console_feed, device_cursor_glyph_copy, device_cursor_glyph_info,
    device_efi_console_copy, device_scanout_copy, device_scanout_may_paint, ConsoleFeed,
    CursorGlyphInfo,
};

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use crate::qemu::host_ops::{NullHost, QemuHost, ReimsVgpuHostOps};
// The four names the two chapter modules below reach through `use super::*`,
// and this module uses itself. They were the crate root's "convenience
// re-exports used by qemu ABI and tests" and came with the registry.
use crate::model::{Device, DeviceId};
use crate::runtime::{HostAction, HostOps};

#[cfg(feature = "backend-metal")]
type SelectedBackend = crate::backend::metal::MetalBackend;

#[cfg(feature = "backend-vulkan")]
type SelectedBackend = crate::backend::vulkan::VulkanBackend;

/// Mutable protocol/backend state. The drain worker may hold this lock across
/// shader translation and a GPU wait, so MMIO producers must never wait for it.
struct DeviceInner {
    device: Device<SelectedBackend>,
    /// Actions for the QEMU BH to apply after drain.
    actions: VecDeque<HostAction>,
}

#[derive(Clone, Copy, Debug)]
struct QueuedGfxWrite {
    offset: u64,
    data: u64,
    size: u32,
    /// When the vCPU published this write, or `None` when it was applied
    /// straight through without ever entering the queue.
    ///
    /// The guest's store retires the moment this is pushed, so the guest cannot
    /// see the delay; the age measured against this stamp is the only place the
    /// deferral becomes visible. See [`crate::runtime::drain::DoorbellCensus`].
    queued_at: Option<std::time::Instant>,
}

/// One live device. Registry lookup and MMIO ingress remain short even while
/// `inner` is owned by the ordered render worker.
struct BoundDevice {
    inner: Mutex<DeviceInner>,
    gfx_ingress: Mutex<VecDeque<QueuedGfxWrite>>,
    gfx_read_cache: Mutex<HashMap<(u64, u32), u64>>,
    gfx_read_busy_logged: AtomicBool,
    /// Prompt HostActions (IRQ pulses, cursor moves): poppable without `inner`
    /// so the BH delivers them while the drain worker still owns the device
    /// lock. Scanout/glyph actions stay in `DeviceInner::actions`.
    prompt_actions: Mutex<VecDeque<HostAction>>,
    /// Lock-free clones of the read-to-clear interrupt-status registers
    /// (`state.gfx.interrupt_status_disp` / `_gpu`): the guest ISR read at
    /// 0x1014/0x1018 must observe live bits mid-drain, never a stale cache.
    intr_disp: Arc<AtomicU32>,
    intr_gpu: Arc<AtomicU32>,
    /// Child channels the guest has rung, OR'd from the vCPU thread with no
    /// device lock; see [`crate::model::GfxRegs::child_doorbell_rung`].
    child_doorbell_rung: Arc<AtomicU32>,
    /// Lock-free clone of the fault status (0x102c) — the ISR's third read.
    intr_fault: Arc<AtomicU32>,
    /// Lock-free clone of the main-FIFO consumer counter (0x100c): the guest
    /// `writeFifo` producer spin must observe drain progress live, not a
    /// cached snapshot from before the tranche.
    fifo_read_live: Arc<AtomicU32>,
    /// An accepted present is waiting for QEMU to consume its scanout action.
    /// Kept outside `inner` so new worker wakeups can yield without racing the
    /// main-loop copy for the same device lock.
    present_action_pending: AtomicBool,
    /// Monotonic per-boot publication of `frame_flush_seen`. QEMU refresh must
    /// never reinterpret a contended device lock as a return to BAR1/EFI.
    present_boundary_seen: AtomicBool,
    /// Monotonic reset sequence for cross-boot lifecycle diagnosis.
    reset_count: AtomicU64,
    /// Lock-free snapshot of the display VBL state, republished on every
    /// lock-acquired `device_poll`. Lets a *contended* poll (the drain worker
    /// owns `inner`) still pulse VBL so the guest keeps its display time base
    /// under load — without it, `device_poll` early-returns on the `try_lock`
    /// miss and drops the VBL entirely (kb present-thrash-proxies: VBL collapses
    /// to ~7 Hz under interaction). `vbl_shared_gpa == 0` ⇒ not online yet.
    vbl_shared_gpa: AtomicU64,
    vbl_display_index: AtomicU32,
    vbl_online: AtomicBool,
    /// Guest page size, published with the rest of the lock-free VBL snapshot.
    /// The refresh tick's pending write is page-bounded like every other write
    /// into the shared page, and this arm has no `DeviceState` to ask.
    vbl_page_size: AtomicU64,
    /// Wall-clock ms of the last VBL claimed by either the locked or contended
    /// poll path. One shared limiter keeps guest pacing independent of which
    /// path happens to win the device lock.
    vbl_last_us: AtomicU64,
    /// QEMU HostOps (GPA / clock / schedule worker). None in pure unit tests.
    ops: Option<ReimsVgpuHostOps>,
    /// Host-owned presentation window ([[host-window]]), once
    /// `device_window_start` has spawned it. `None` on a normal QEMU-display
    /// boot (the window is opt-in behind `REIMS_VGPU_WINDOW`).
    #[cfg(feature = "host-window")]
    window: Mutex<Option<window_publish::WindowLink>>,
    /// Early-boot framebuffer (BAR1 GOP) registered by the C shim, shown in the
    /// window until the product present path latches.
    #[cfg(feature = "host-window")]
    early_fb: Mutex<Option<window_publish::EarlyFb>>,
    /// Monotonic ns of the last early frame pushed to the window (poll-path
    /// throttle so the pre-boundary pump does not memcpy the FB every 4 ms).
    #[cfg(feature = "host-window")]
    early_last_ns: AtomicU64,
}

type DeviceMap = HashMap<u64, Arc<BoundDevice>>;

static DEVICES: Lazy<Mutex<DeviceMap>> = Lazy::new(|| Mutex::new(HashMap::new()));
static NEXT_ID: Lazy<Mutex<u64>> = Lazy::new(|| Mutex::new(1));

fn device_slot(id: u64) -> Option<Arc<BoundDevice>> {
    DEVICES.lock().get(&id).cloned()
}

fn schedule_device(slot: &BoundDevice) {
    let Some(ops) = slot.ops else {
        return;
    };
    if let Some(schedule) = ops.schedule_bh {
        // SAFETY: QEMU owns ctx for the device lifetime; schedule_bh is the
        // thread-safe wake callback supplied by the shim.
        unsafe { schedule(ops.ctx) }
    }
}

#[inline]
fn publish_present_boundary(slot: &BoundDevice, frame_flush_seen: bool) {
    if frame_flush_seen {
        slot.present_boundary_seen.store(true, Ordering::Release);
    }
}

fn apply_gfx_write(inner: &mut DeviceInner, slot: &BoundDevice, write: QueuedGfxWrite) {
    match write.queued_at {
        Some(at) => crate::runtime::drain::note_doorbell_queued(
            write.offset,
            at.elapsed().as_micros() as u64,
        ),
        None => crate::runtime::drain::note_doorbell_direct(),
    }
    if let Some(ops) = slot.ops {
        let mut host = QemuHost::new(&ops, &mut inner.actions, &slot.prompt_actions);
        inner
            .device
            .gfx_write(&mut host, write.offset, write.data, write.size);
    } else {
        let mut host = NullHost;
        inner
            .device
            .gfx_write(&mut host, write.offset, write.data, write.size);
    }
}

/// Apply queued MMIO writes in publication order. Lock order is ingress then
/// inner everywhere; producers use `try_lock` for inner and therefore never
/// wait behind shader translation/GPU work.
fn lock_for_drain(slot: &BoundDevice) -> parking_lot::MutexGuard<'_, DeviceInner> {
    let mut ingress = slot.gfx_ingress.lock();
    let mut inner = slot.inner.lock();
    while let Some(write) = ingress.pop_front() {
        apply_gfx_write(&mut inner, slot, write);
    }
    drop(ingress);
    // Here rather than only inside `drain_pending`, because this is the one
    // point every entry to the drain passes through. `device_drain` returns
    // before `drain_pending` when the device has no host ops, and
    // `publish_stranded_fifos` re-publishes from `active_child_mask` — a ring
    // left unfolded would be invisible to both.
    crate::runtime::drain::fold_rung_child_doorbells(&mut inner.device.state);
    inner
}

fn make_backend() -> SelectedBackend {
    #[cfg(feature = "backend-metal")]
    {
        crate::backend::metal::MetalBackend::new()
    }
    #[cfg(feature = "backend-vulkan")]
    {
        crate::backend::vulkan::VulkanBackend::new()
    }
}

/// Create a device. `ops` is the QEMU host-service table (nullable for tests).
///
/// `page_shift` must be [`crate::model::PAGE_SHIFT_X86`] (12) or [`crate::model::PAGE_SHIFT_ARM64E`] (14).
/// There is no default (including no `0` → arm); unsupported values return `None`.
pub fn device_create(ops: Option<ReimsVgpuHostOps>, page_shift: u32) -> Option<u64> {
    use crate::model::{PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
    if page_shift != PAGE_SHIFT_ARM64E && page_shift != PAGE_SHIFT_X86 {
        return None;
    }
    let mut id_guard = NEXT_ID.lock();
    let id = *id_guard;
    *id_guard = id.saturating_add(1);
    drop(id_guard);
    let backend = make_backend();
    let dev = Device::new(DeviceId(id), backend, page_shift);
    let intr_disp = Arc::clone(&dev.state.gfx.interrupt_status_disp);
    let intr_gpu = Arc::clone(&dev.state.gfx.interrupt_status_gpu);
    let child_doorbell_rung = Arc::clone(&dev.state.gfx.child_doorbell_rung);
    let intr_fault = Arc::clone(&dev.state.gfx.interrupt_fault);
    let fifo_read_live = Arc::clone(&dev.state.gfx.fifo_read);
    DEVICES.lock().insert(
        id,
        Arc::new(BoundDevice {
            inner: Mutex::new(DeviceInner {
                device: dev,
                actions: VecDeque::new(),
            }),
            gfx_ingress: Mutex::new(VecDeque::new()),
            gfx_read_cache: Mutex::new(HashMap::new()),
            gfx_read_busy_logged: AtomicBool::new(false),
            prompt_actions: Mutex::new(VecDeque::new()),
            intr_disp,
            intr_gpu,
            child_doorbell_rung,
            intr_fault,
            fifo_read_live,
            present_action_pending: AtomicBool::new(false),
            present_boundary_seen: AtomicBool::new(false),
            reset_count: AtomicU64::new(0),
            vbl_shared_gpa: AtomicU64::new(0),
            vbl_display_index: AtomicU32::new(0),
            vbl_online: AtomicBool::new(false),
            vbl_page_size: AtomicU64::new(0),
            vbl_last_us: AtomicU64::new(0),
            ops,
            #[cfg(feature = "host-window")]
            window: Mutex::new(None),
            #[cfg(feature = "host-window")]
            early_fb: Mutex::new(None),
            #[cfg(feature = "host-window")]
            early_last_ns: AtomicU64::new(0),
        }),
    );
    // The completion thread's way back to the guest. Installed here rather than
    // built into the engine because the engine must not know what a
    // `BoundDevice` is, and looked up by id rather than captured by `Arc` so a
    // stale hook cannot keep a torn-down device alive.
    #[cfg(feature = "backend-vulkan")]
    crate::backend::vulkan::engine::stamp_completion::install_announce(std::sync::Arc::new(
        move |index: u32| announce_stamp_interrupt(id, index),
    ));
    Some(id)
}

/// Raise the gfx interrupt for a stamp whose word the GPU has written.
///
/// Runs on the engine's completion thread, so it may touch nothing that needs
/// the device `inner` lock — the drain worker holds it for most of a busy
/// second and blocking here would put the guest's wakeup behind exactly the work
/// it is waiting to be told about.
///
/// It does not have to. This is the same three-step the display VBL already
/// takes from a contended poll ([`vbl_contended_pulse`]): OR the bit into the
/// lock-free `Arc<AtomicU32>` clone of the interrupt-status register that the
/// guest ISR reads at 0x1018, push the pulse onto `prompt_actions` — which
/// `device_pop_action` drains without the device lock — and let `enqueue`'s
/// prompt rail call the thread-safe `notify_actions`. The stamp *word* is
/// already in guest memory by construction: the submission that signalled this
/// thread's timeline value wrote it.
#[cfg(feature = "backend-vulkan")]
fn announce_stamp_interrupt(id: u64, index: u32) {
    let Some(slot) = device_slot(id) else {
        // The device is gone, so there is no interrupt-status register to set
        // and nobody to interrupt. Quiet: this is teardown, not a loss.
        return;
    };
    slot.intr_gpu
        .fetch_or(1u32 << (index & 0x1f), Ordering::AcqRel);
    let Some(ops) = slot.ops else {
        return;
    };
    let mut scratch = VecDeque::new();
    QemuHost::new(&ops, &mut scratch, &slot.prompt_actions).enqueue(HostAction::irq_gfx());
}

pub fn device_reset(id: u64) -> bool {
    if let Some(slot) = device_slot(id) {
        let mut d = lock_for_drain(&slot);
        let seq = slot.reset_count.fetch_add(1, Ordering::Relaxed) + 1;
        let state = &d.device.state;
        let mappings = state.mappings.len();
        let tasks = state.tasks.live_count();
        let host_surfaces = state.host_surfaces.len();
        let host_textures = state.host_texture_surfaces.len();
        let host_gvas = state.host_gva_surfaces.len();
        let host_linear = state.host_linear_textures.len();
        let frame_valid = state.present.frame_valid;
        let frame_mapping = state.present.frame_mapping;
        let boundary = state.present.frame_flush_seen;
        let views = if let Some(ops) = slot.ops {
            let DeviceInner { device, actions } = &mut *d;
            let mut host = QemuHost::new(&ops, actions, &slot.prompt_actions);
            device.reset_with_host(&mut host)
        } else {
            d.device.reset();
            0
        };
        crate::observe::off(format!(
            "device_reset id={id} seq={seq} mappings={mappings} tasks={tasks} host_surface={host_surfaces} host_texture={host_textures} host_gva={host_gvas} host_linear={host_linear} frame_valid={} frame_mapping={frame_mapping} boundary={} unmapped_views={views}",
            u8::from(frame_valid),
            u8::from(boundary)
        ));
        d.actions.clear();
        slot.prompt_actions.lock().clear();
        slot.gfx_read_cache.lock().clear();
        slot.present_action_pending.store(false, Ordering::Release);
        slot.present_boundary_seen.store(false, Ordering::Release);
        crate::runtime::census::present_proxy::reset_for_device();
        true
    } else {
        false
    }
}

pub fn device_destroy(id: u64) -> bool {
    DEVICES.lock().remove(&id).is_some()
}

pub fn device_gfx_read(id: u64, offset: u64, size: u32) -> Option<u64> {
    use crate::model::{
        GFX_REG_FIFO_READ, GFX_REG_INTR_FAULT, GFX_REG_INTR_STATUS_DISP, GFX_REG_INTR_STATUS_GPU,
    };
    let slot = device_slot(id)?;
    // Guest spin/ISR registers: served lock-free from the shared atomics so a
    // drain-tranche-held device lock never turns a fresh stamp signal into a
    // stale cached mask (0x1014/0x1018 r2c) nor hides drain progress from the
    // writeFifo producer spin (0x100c) or the ISR fault read (0x102c).
    if size == 4 {
        if offset == GFX_REG_INTR_STATUS_DISP {
            return Some(slot.intr_disp.swap(0, Ordering::AcqRel) as u64);
        }
        if offset == GFX_REG_INTR_STATUS_GPU {
            return Some(slot.intr_gpu.swap(0, Ordering::AcqRel) as u64);
        }
        if offset == GFX_REG_FIFO_READ {
            return Some(slot.fifo_read_live.load(Ordering::Acquire) as u64);
        }
        if offset == GFX_REG_INTR_FAULT {
            return Some(slot.intr_fault.load(Ordering::Acquire) as u64);
        }
    }
    if let Some(mut d) = slot.inner.try_lock() {
        let value = d.device.gfx_read(offset, size);
        slot.gfx_read_cache.lock().insert((offset, size), value);
        slot.gfx_read_busy_logged.store(false, Ordering::Relaxed);
        return Some(value);
    }
    let value = slot
        .gfx_read_cache
        .lock()
        .get(&(offset, size))
        .copied()
        .unwrap_or(0);
    if !slot.gfx_read_busy_logged.swap(true, Ordering::Relaxed) {
        crate::observe::fail(format!(
            "device_lock_busy reason=gfx_read_deferred offset={offset:#x} size={size} cached={value:#x}"
        ));
    }
    Some(value)
}

pub fn device_gfx_write(id: u64, offset: u64, data: u64, size: u32) -> bool {
    use crate::model::{GFX_REG_INTR_STATUS_DISP, GFX_REG_INTR_STATUS_GPU};
    let Some(slot) = device_slot(id) else {
        return false;
    };
    // Interrupt-status mask clears are order-independent of FIFO doorbells;
    // apply them lock-free instead of queueing behind a busy drain tranche.
    if size == 4 {
        if offset == GFX_REG_INTR_STATUS_DISP {
            slot.intr_disp.fetch_and(!(data as u32), Ordering::AcqRel);
            return true;
        }
        if offset == GFX_REG_INTR_STATUS_GPU {
            slot.intr_gpu.fetch_and(!(data as u32), Ordering::AcqRel);
            return true;
        }
        // The child doorbell, which measurement says is the *entire* queueing
        // stall on this pathway: `gfx_doorbell_delay` reads `offsets=1` on
        // every window that queued anything, ~100 rings a second applied up to
        // 45 ms late, and that delay is the drain tranche the write could not
        // take the lock through.
        //
        // It is the one register that can be served this way, because it
        // carries no state the decode depends on — its effect is to say a
        // channel has work. `fold_rung_child_doorbells` turns the bit into
        // `active_child_mask` / `pending.child_mask`, which is exactly what the
        // locked handler in `crate::runtime::mmio` does for the same register.
        //
        // The channel-number check mirrors that handler rather than trusting
        // the guest: a value outside the channel range names no channel, and
        // shifting by it would be undefined. An out-of-range ring still
        // schedules nothing here, as it does there — but it is reported rather
        // than dropped in silence, through the one spelling of the rule that
        // both handlers share.
        if offset == crate::model::GFX_REG_CHILD_DOORBELL
            || offset == crate::model::GFX_REG_CHILD_REPLAY_DOORBELL
        {
            let channel = data as u32;
            if crate::model::accept_child_channel(channel, "lock_free_child_doorbell") {
                slot.child_doorbell_rung
                    .fetch_or(1u32 << channel, Ordering::AcqRel);
                crate::runtime::drain::note_doorbell_lock_free();
                schedule_device(&slot);
            }
            return true;
        }
    }
    let mut write = QueuedGfxWrite {
        offset,
        data,
        size,
        queued_at: None,
    };
    let mut ingress = slot.gfx_ingress.lock();
    if ingress.is_empty() {
        if let Some(mut inner) = slot.inner.try_lock() {
            apply_gfx_write(&mut inner, &slot, write);
            return true;
        }
    }
    // Stamped only on the path that actually defers, so the direct path pays no
    // clock read at all.
    write.queued_at = Some(std::time::Instant::now());
    ingress.push_back(write);
    drop(ingress);
    schedule_device(&slot);
    true
}

/// Take the device lock from the vCPU thread, measuring the wait.
///
/// The guest's MMIO access is stopped for exactly as long as this blocks, and
/// the drain worker holds this same lock across a full-surface readback. Every
/// other figure about that stall is taken from the holder's side, which makes
/// the step to "the guest missed a frame" an inference; this measures it where
/// it is actually paid.
///
/// The uncontended path takes `try_lock` and never reads the clock, so a fast
/// access pays nothing for the instrument.
fn lock_device_for_vcpu(slot: &BoundDevice) -> impl std::ops::DerefMut<Target = DeviceInner> + '_ {
    if let Some(guard) = slot.inner.try_lock() {
        crate::runtime::drain::note_vcpu_lock_free();
        return guard;
    }
    let waited = std::time::Instant::now();
    let guard = slot.inner.lock();
    crate::runtime::drain::note_vcpu_lock_wait(waited.elapsed().as_micros() as u64);
    guard
}

pub fn device_iosfc_read(id: u64, offset: u64, size: u32) -> Option<u64> {
    let slot = device_slot(id)?;
    let d = lock_device_for_vcpu(&slot);
    Some(d.device.iosfc_read(offset, size))
}

pub fn device_iosfc_write(id: u64, offset: u64, data: u64, size: u32) -> bool {
    let Some(slot) = device_slot(id) else {
        return false;
    };
    let mut d = lock_device_for_vcpu(&slot);
    if let Some(ops) = slot.ops {
        let DeviceInner { device, actions } = &mut *d;
        let mut host = QemuHost::new(&ops, actions, &slot.prompt_actions);
        device.iosfc_write(&mut host, offset, data, size);
    } else {
        let mut host = NullHost;
        d.device.iosfc_write(&mut host, offset, data, size);
    }
    true
}

/// Worker body: drain pending FIFOs using QEMU GPA callbacks; enqueue HostActions.
pub fn device_drain(id: u64) -> bool {
    let Some(slot) = device_slot(id) else {
        return false;
    };
    // Opens this worker's wall-clock accounting: everything since the previous
    // exit was the condvar wait the C shim parks in, and everything from here to
    // the lock is contention with the vCPU thread. `duty` says how much of a
    // second the worker was busy; these say what the rest of it was.
    let entry_us = crate::runtime::drain::note_drain_entry();
    // The action BH needs the same device state to copy +0x188. A doorbell may
    // wake this worker before that BH runs; do not reacquire the lock and hide
    // the queued scanout behind another synchronous render/compute tranche.
    if slot.present_action_pending.load(Ordering::Acquire) {
        crate::runtime::drain::note_drain_skipped();
        crate::runtime::drain::note_drain_exit(entry_us, true);
        return true;
    }
    let mut d = lock_for_drain(&slot);
    crate::runtime::drain::note_drain_lock_wait(
        crate::observe::elapsed_us().saturating_sub(entry_us),
    );
    let Some(ops) = slot.ops else {
        // No host services — nothing to resolve from guest RAM. The lock wait is
        // already banked, so this closes with no post-tranche span rather than
        // leaving the entry open for the next one to absorb.
        crate::runtime::drain::note_drain_exit(crate::observe::elapsed_us(), false);
        return true;
    };
    let DeviceInner { device, actions } = &mut *d;
    let mut host = QemuHost::new(&ops, actions, &slot.prompt_actions);
    // Presentation-path selector for this tranche: with a live host window the
    // drain publishes frames + self-acks; without one every present must
    // enqueue the CPU `ScanoutUpdate` and the ack belongs to the console paint
    // (see `enqueue_present_scanout` / the drain tail below).
    #[cfg(feature = "host-window")]
    {
        device.state.present.window_active = slot.window.lock().is_some();
    }
    #[cfg(not(feature = "host-window"))]
    {
        device.state.present.window_active = false;
    }
    // Split the tranche's two phases: guest work, then our host-window export.
    // Both hold the device lock, and which one owns the worker's wall clock is
    // the question `drain_duty` exists to answer.
    let tranche_started = std::time::Instant::now();
    // The same instant on the crate's own clock, so a lookup inside the drain
    // can say how late in this tranche it happened without threading a start
    // time through every call. See `census::tranche_elapsed_us`.
    crate::runtime::drain::note_tranche_started(crate::observe::elapsed_us());
    device.drain(&mut host);
    // The tail is timed apart from `Device::drain` because it is inside
    // `drain_us` and inside no `DrainPhase`, and that residue is a third of the
    // drain worker's wall clock on every workload measured — 933 ms a second of
    // `drain_us` against 604 of `draw_us` on a driven `blur=40` boot. A gap that
    // size on the one thread every guest packet serializes through cannot be
    // left to inference.
    let tail_started = std::time::Instant::now();
    // Submit any deferred draw batch before the worker sleeps: consumers
    // inside the tranche flush on their own (engine begin_entry), this bounds
    // only the idle-tail latency of the last same-target run.
    #[cfg(feature = "backend-vulkan")]
    crate::backend::vulkan::engine::flush_batched_draws();
    let tail_us = tail_started.elapsed().as_micros() as u64;
    let boundary_started = std::time::Instant::now();
    publish_present_boundary(&slot, device.state.present.frame_flush_seen);
    crate::runtime::drain::note_drain_tail(tail_us, boundary_started.elapsed().as_micros() as u64);
    let drain_us = tranche_started.elapsed().as_micros() as u64;
    let publish_started = std::time::Instant::now();
    // Push the finished present frame to the host-owned window (if running).
    // Off the QEMU main loop; a small dedicated mutex, never the render lock.
    // Refresh the guest cursor position before publishing. The protocol only
    // updates it on CURSOR_SHOW / CURSOR_GLYPH and on the display-IRQ doorbell,
    // so a pointer that moves without changing shape can leave x/y stale and the
    // window overlay then appears to track only shape changes. One 4-byte guest
    // read per tranche, and only when a window is actually consuming frames.
    #[cfg(feature = "host-window")]
    if device.state.present.window_active {
        crate::runtime::drain::sample_cursor_position(&mut device.state, &host);
    }
    #[cfg(feature = "host-window")]
    window_publish::publish_window_frame(&slot, &mut device.state);
    crate::runtime::drain::note_drain_tranche(
        &host,
        drain_us,
        publish_started.elapsed().as_micros() as u64,
    );
    // Everything from here to the return is `gap_post_us`: the per-tranche
    // sweeps below run on the worker's own wall clock and are outside both
    // `drain_us` and `publish_us`, so `duty` cannot see them.
    let busy_end_us = crate::observe::elapsed_us();
    use crate::runtime::drain::{post_sweep, PostSweep};
    // Same one-second cadence, so the cache trend lines up row-for-row with
    // `store_routes` and `drain_duty`. Measure-only; see `note_cache_levels`.
    post_sweep(PostSweep::CacheLevels, || {
        crate::runtime::surface_cache::note_cache_levels(&device.state, &host)
    });
    // Per tranche rather than per census window, unlike the levels above: this
    // measures how long a slot the guest named takes to appear, so the sampling
    // interval is the resolution of the answer. Returns immediately when nothing
    // is watched, which is every tranche on every rail but macos-26.
    post_sweep(PostSweep::SlotRecheck, || {
        crate::runtime::objects::slot_recheck::sweep(&device.state, &host)
    });
    // Beside it and on the same cadence: a page the guest released is judged
    // against the write census, which only moves when this device writes. Also
    // returns immediately when nothing is watched.
    post_sweep(PostSweep::ReleasedPages, || {
        crate::runtime::released_pages::sweep(&mut device.state);
        crate::runtime::released_pages::note_levels(&device.state);
    });
    // The bind registry's own levels, on that same cadence and read against the
    // `bb_retire_*` routes: what the retirements dropped, and what the survivors
    // look like.
    #[cfg(feature = "backend-vulkan")]
    post_sweep(PostSweep::BindLevels, || {
        crate::runtime::bound_buffers::note_registry_levels(&device.state)
    });
    // The present-completion ack, re-homed off the QEMU paint — ONLY while the
    // host window is the display. With the window live no per-present
    // `ScanoutUpdate` is enqueued, so `display_surface::device_scanout_copy` —
    // the only other caller of `note_present_paint_consumed` — will not run for
    // this present.
    // Acking here clears `unpainted_presents` (releasing the DisplaySwap
    // backpressure gate at `MAX_UNPAINTED_PRESENTS`) and `host_action_yield`, so
    // the check below leaves `present_action_pending` clear and the worker keeps
    // draining. Without this the display channel wedges on the second present.
    //
    // On the window path the ack is deliberately NOT keyed on "a frame was
    // published": the publish legitimately early-returns on a duplicate
    // (mapping, generation), on a frame not yet valid, and on a short buffer. An
    // ack that fired only when the window took a fresh frame would wedge on the
    // first repeated present.
    //
    // Without a window the QEMU console owns the paint: `enqueue_present_scanout`
    // enqueued the `ScanoutUpdate`, `host_action_yield` stays set, the flag below
    // arms `present_action_pending`, and `device_scanout_copy` both paints and
    // acks. Pre-acking here would let `device_scanout_copy`'s nonblocking
    // `try_lock` path swallow the paint as `Unchanged` under worker contention —
    // the frozen-console class this split fixes.
    if device.state.present.window_active {
        crate::runtime::drain::note_present_paint_consumed(&mut device.state);
    }
    if device.state.pending.host_action_yield {
        slot.present_action_pending.store(true, Ordering::Release);
    }
    crate::runtime::drain::note_drain_exit(busy_end_us, false);
    true
}

/// Periodic tick (gfx_update / poll): archive `poll_tick` subset.
///
/// - Dekker rescue: publish main/child/iosfc work to the asynchronous drain
///   owner when producer state may have advanced without a doorbell.
/// - Re-drive display ONLINE after guest enable() publishes the mask.
///
/// Enqueues HostActions (gfx IRQ / scanout); QEMU must deliver actions after
/// this call.
pub fn device_poll(id: u64) -> bool {
    // Before the lock, and before the `device_slot` miss can return: this is the
    // only periodic callback that runs on a thread other than the drain worker,
    // so it is the only place that can report a driver call the drain worker is
    // still inside. See `observe::driver_watch` for what that failure looks like
    // from the log (it looks like nothing at all).
    crate::observe::driver_watch::note_tick();
    let Some(slot) = device_slot(id) else {
        return false;
    };
    let Some(mut d) = slot.inner.try_lock() else {
        // Contended: the drain worker owns `inner` doing present/GPU-encode.
        // The full poll below would early-return and drop the VBL — under load
        // that starves the guest's only display time base (present-complete is
        // inert; kb present-thrash-proxies). Pulse VBL lock-free from the state
        // the last successful poll published, so pacing survives the contention.
        vbl_contended_pulse(&slot);
        return true;
    };
    let Some(ops) = slot.ops else {
        return true;
    };
    let DeviceInner { device, actions } = &mut *d;
    let mut host = QemuHost::new(&ops, actions, &slot.prompt_actions);
    // Before the rescue reads `active_child_mask`, which is the mask a
    // lock-free ring lands in only once folded. Without this the Dekker rescue
    // could not see the very channels the doorbell rail is responsible for.
    crate::runtime::drain::fold_rung_child_doorbells(&mut device.state);
    crate::runtime::drain::publish_stranded_fifos(&mut device.state, &mut host);
    crate::runtime::drain::try_display_online(&mut device.state, &mut host);
    // After ONLINE, pulse VBL so the guest compositor has a display time base
    //. Missing VBL → clear-only dual-mid present thrash.
    crate::runtime::drain::signal_display_vbl(&mut device.state, &mut host, &slot.vbl_last_us);
    // Republish the lock-free VBL snapshot for the contended fast path above.
    // These change only at online-ack/reinit, but publishing every poll keeps
    // the snapshot fresh with no extra synchronization on the rare-change path.
    slot.vbl_shared_gpa
        .store(device.state.display.shared_gpa, Ordering::Release);
    slot.vbl_display_index
        .store(device.state.display.display_index, Ordering::Release);
    slot.vbl_online
        .store(device.state.display.online_acked, Ordering::Release);
    slot.vbl_page_size
        .store(device.state.page_size(), Ordering::Release);
    // Census both source polls and the independently time-gated VBL rate.
    // Drive bounded maintenance from the poll heartbeat, which ticks even when
    // the guest stops publishing. The wall clock returns already-dead resources
    // and free-pool memory; it has no authority over live residency, which is
    // governed by resource lifetime and allocation pressure.
    #[cfg(feature = "backend-vulkan")]
    {
        crate::backend::vulkan::engine::maintain_resources(crate::observe::elapsed_ms() as u64);
        crate::runtime::mapper::drain_deferred_unmaps(&mut host);
    }
    // Pre-boundary early-console → host window (headless-safe: the heartbeat
    // drives poll even under -display none). No-op post-boundary or with no
    // window attached.
    #[cfg(feature = "host-window")]
    {
        let now_ns = host.mono_ns();
        window_publish::publish_window_early_frame(&slot, &device.state, &host, now_ns);
    }

    // Track the cursor from the 4 ms poll heartbeat, not only from the drain.
    //
    // The guest updates its hardware-cursor position in the shared page as the
    // pointer moves but does not doorbell every move, so the position is only
    // noticed when something reads that page. `device_drain` reads it (and
    // republishes the overlay) once per tranche — but on an idle macOS desktop
    // the guest produces almost no FIFO traffic, so drains run at ~15/s and the
    // cursor visibly steps at that rate. Dragging a window or crossing the dock
    // makes the guest draw continuously, the drain runs at full rate, and the
    // same cursor is smooth: the tell that this is drain cadence, not the
    // overlay.
    //
    // The poll runs at 4 ms regardless of guest activity, so sampling and
    // publishing here lets the overlay track at up to 250 Hz on an idle screen.
    // On a static frame `publish_window_frame` takes its unchanged-key path: a
    // 4-byte guest read, a key compare, and — only when the cursor fingerprint
    // moved — one Arc-clone republish with no pixel copy. The poll holds `inner`
    // here, so this cannot race the drain worker's own publish.
    #[cfg(feature = "host-window")]
    {
        device.state.present.window_active = slot.window.lock().is_some();
        if device.state.present.window_active {
            crate::runtime::drain::sample_cursor_position(&mut device.state, &host);
            window_publish::publish_window_frame(&slot, &mut device.state);
        }
    }
    true
}

/// Lock-free VBL pulse for a `device_poll` that could not take `inner`.
///
/// Raises the display VBL — OR the VBL bit into the shared-page pending word,
/// set the read-to-clear display interrupt bit, enqueue the gfx IRQ pulse — all
/// through paths that never touch the device `inner` lock (guest-memory RMW via
/// HostOps, the `Arc<AtomicU32>` interrupt clone, the lock-free `prompt_actions`
/// queue). Uses the VBL state the last lock-acquired poll published. No-op until
/// ONLINE is acked. It shares the same time limiter as the locked path, so a
/// change in lock ownership cannot change the guest's pacing rate.
///
/// The pending-word RMW can race the worker's own present-complete write; the
/// loser drops one bit for one heartbeat (re-raised ~16 ms later). Both writers
/// clear the acked ONLINE bit, so a torn write cannot resurrect it — far better
/// than dropping ~90% of VBLs, which is the pre-fix behaviour under load.
fn vbl_contended_pulse(slot: &BoundDevice) {
    let gpa = slot.vbl_shared_gpa.load(Ordering::Acquire);
    let now = crate::observe::elapsed_ms() as u64;
    if gpa == 0 || !slot.vbl_online.load(Ordering::Acquire) {
        crate::runtime::drain::note_vbl(crate::runtime::drain::VBL_NOT_ONLINE, now);
        return;
    }
    let Some(ops) = slot.ops else {
        return;
    };
    let page_size = slot.vbl_page_size.load(Ordering::Acquire);
    if page_size == 0 {
        // The locked poll publishes this with the rest of the snapshot, so a
        // zero means no locked poll has run since bind. Nothing is owed yet.
        crate::runtime::drain::note_vbl(crate::runtime::drain::VBL_NOT_ONLINE, now);
        return;
    }
    let mut scratch = VecDeque::new();
    let mut host = QemuHost::new(&ops, &mut scratch, &slot.prompt_actions);
    // One body decides what a refresh tick writes, and both poll arms call it.
    // This arm used to carry its own copy, and the copy had already lost a term:
    // it never read the enable word at all, so it set a pending bit the guest's
    // ISR would never clear and counted the write as `delivered`.
    //
    // The shared limiter lives in there too, so both arms report into one census
    // and neither can spend a grid slot on a tick that found the guest disarmed.
    crate::runtime::drain::signal_display_refresh_classes(
        &mut host,
        gpa,
        slot.vbl_display_index.load(Ordering::Acquire),
        &slot.intr_disp,
        page_size as usize,
        &slot.vbl_last_us,
        crate::observe::elapsed_us(),
    );
}

/// Pop one HostAction for the QEMU BH. Returns false if the queue is empty.
///
/// Prompt actions (IRQ pulses, cursor moves) pop without the device lock so
/// they deliver mid-drain; lock-owning actions (scanout, cursor glyph) keep
/// their after-drain semantics behind `try_lock`.
pub fn device_pop_action(id: u64) -> Option<HostAction> {
    let slot = device_slot(id)?;
    {
        let mut q = slot.prompt_actions.lock();
        if let Some(a) = q.pop_front() {
            // The hop this closes is enqueue-to-BH, so it is banked when the
            // queue empties rather than per action: an IRQ pulse behind a cursor
            // move waited for the same BH and would double-count. See
            // `irq_wait_us`.
            if q.is_empty() {
                crate::runtime::drain::note_irq_delivered();
            }
            return Some(a);
        }
    }
    let mut d = slot.inner.try_lock()?;
    d.actions.pop_front()
}

pub fn backend_name() -> &'static str {
    #[cfg(feature = "backend-metal")]
    {
        "metal"
    }
    #[cfg(feature = "backend-vulkan")]
    {
        "vulkan"
    }
}

/// Run one C ABI entry body, turning a panic into `on_panic` rather than
/// letting it unwind into QEMU's C frames.
///
/// `entry` is the C symbol this body belongs to, and it is what the record
/// names — a panic here is the largest thing this device can drop (the whole
/// call, not one refused record), so it goes through the always-on failure path
/// like any other loss of guest work. See [`crate::observe::panic`] for why the
/// location needs a hook and why arming it lives on this path.
///
/// Every caller must pass its own symbol name. A copied call site that still
/// names its neighbour reports the wrong entry point for a panic, and nothing
/// checks for it — a source scan used to.
pub fn unwind_safe<T, F>(entry: &'static str, f: F, on_panic: T) -> T
where
    F: FnOnce() -> T + std::panic::UnwindSafe,
{
    crate::observe::panic::arm();
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(payload) => {
            crate::observe::panic::report(entry, payload.as_ref());
            on_panic
        }
    }
}

#[cfg(test)]
mod tests;
