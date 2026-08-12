//! The host-owned presentation window: its device-side link, its four QEMU
//! entry points, and the two publish paths the drain and the poll call.
//!
//! One of the four jobs the crate root used to hold, and the only one that is
//! entirely optional — every item here is behind `host-window` except the four
//! `device_window_*` entry points, which keep a stub arm so the QEMU ABI
//! surface is the same shape on a build without the feature.
//!
//! `use super::*` rather than a named import list, for the reason
//! `crate::runtime::draw::vulkan` gives: this is a chapter of the crate root that was
//! lifted out whole, and it reaches back for the registry (`DEVICES`,
//! `device_slot`, `BoundDevice`) that is the root's own job to own. Naming
//! forty root items here would make the move look like a redesign and would go
//! stale on the next one. Gated, because without the feature all that survives
//! here is four stub functions that take a `u64` and return `false`, and they
//! reach back for nothing.

#[cfg(feature = "host-window")]
use super::*;

/// Link to a running host-owned presentation window ([[host-window]]).
///
/// Held on the device so the drain can publish finished frames into `frames`
/// (latest-wins) and skip re-publishing an unchanged present via `last`. The
/// input back-channel is the window thread's `InputSink`, not stored here — it
/// pushes onto `prompt_actions` directly.
#[cfg(feature = "host-window")]
type WindowFrameKey = (u32, u32, u32, u64);

/// `pub(super)` for `device::tests::window_publish_key_advances_for_in_place_present`,
/// which is `cfg`-ed to macOS + `host-window` — a combination no Linux arm
/// compiles, so a private spelling here builds green on every arm anyone runs
/// and fails only on the arm64 macOS pathway. It did.
#[cfg(feature = "host-window")]
pub(super) fn window_frame_key(present: &crate::model::PresentState) -> WindowFrameKey {
    #[cfg(target_os = "macos")]
    let present_epoch = present.present_epoch;
    #[cfg(not(target_os = "macos"))]
    let present_epoch = 0;
    (
        present.frame_mapping,
        present.frame_generation,
        // The pixel stamp beside the page stamp. A lazy type-11 Store publishes
        // a new frame without writing a guest page, so `frame_generation` holds
        // still across frames that genuinely differ and this is the only term
        // that moves — see `PresentState::frame_content_epoch`.
        present.frame_content_epoch,
        present_epoch,
    )
}

#[cfg(feature = "host-window")]
pub(crate) struct WindowLink {
    /// Shared latest-frame slot the window thread reads each redraw.
    frames: crate::host_window::present::FrameSlot,
    /// Wakes the window's event loop once a frame has landed in `frames`.
    ///
    /// The window used to find frames by polling the slot every 2 ms, which on a
    /// driven boot asked for 494 redraws a second to serve 8.7 — see
    /// `host_window::present::ENGINE_WINDOW_REDRAW_BACKSTOP`. The publisher is
    /// the only thing that knows a frame exists, so it is the only thing that
    /// can end the polling.
    wake: crate::host_window::present::WindowWakeHandle,
    /// `(mapping_id, generation, content_epoch, present_epoch)` of the last
    /// frame published.
    ///
    /// The resource generation alone is insufficient: the guest can update a
    /// resident in place and present it again without changing that identity.
    /// On macOS, `present_epoch` advances once per accepted capture, so those
    /// frames publish while drain passes with no new DisplaySwap remain
    /// deduplicated. It remains zero on Linux, preserving the verified
    /// `(mapping_id, generation)` publication contract there.
    last: WindowFrameKey,
    /// Monotonic frame sequence stamped onto each published [`Frame`] so the
    /// window uploads only new frames (skips the per-vblank re-upload of
    /// unchanged content). Bumped on every write.
    seq: u64,
    /// Dedup latch for the `frame_bgra_short` drop log: the `(w,h)` last logged
    /// as short, so a persistent mismatch logs once per geometry instead of
    /// every present. Cleared when a well-formed frame publishes.
    ///
    /// Both platforms: the CPU-fallback publish arm is shared since the two
    /// publish paths were unified, and a present with no resident behind it
    /// (firmware framebuffer, a cleared-but-never-rendered mapping, the frames
    /// after a device reset) is normal on macOS too.
    bgra_short_geom: Option<(u32, u32)>,
    /// Set to ask the window thread to exit (VM teardown); the thread polls it.
    stop: crate::host_window::present::StopFlag,
    /// Window thread handle. `device_window_stop` sets `stop` and joins it, so
    /// the window's Vulkan objects tear down before QEMU teardown proceeds
    /// (avoids the driver-unload-during-exit crash class).
    thread: Option<std::thread::JoinHandle<Result<(), crate::host_window::present::WindowError>>>,
    /// Published after the process-main AppKit loop has destroyed the native
    /// window and its Vulkan objects.
    #[cfg(target_os = "macos")]
    exited: crate::host_window::present::ExitedFlag,
}

/// Registered early-boot framebuffer (BAR1 GOP host RAM) the C shim hands the
/// device so the window can show UEFI/OpenCore/boot.efi output before the
/// product present path latches. `ptr` is a stable RAMBlock host pointer valid
/// for the device lifetime; the guest writes it live (a torn read only flickers
/// one early frame).
#[cfg(feature = "host-window")]
#[derive(Clone, Copy)]
pub(crate) struct EarlyFb {
    ptr: usize,
    stride: u32,
    width: u32,
    height: u32,
}

/// Start the host-owned presentation window for `id` ([[host-window]]).
///
/// Wires the window's input `InputSink` to the device prompt-action rail (push
/// and `notify_actions`, both thread-safe) via a `Weak` device ref so the window
/// never keeps a destroyed device alive. Frames reach the window through
/// [`publish_window_frame`], called by the drain. Idempotent; `true` on success.
#[cfg(feature = "host-window")]
pub fn device_window_start(id: u64, width: u32, height: u32) -> bool {
    use crate::host_window::present::{FrameSlot, InputSink, WindowConfig, WindowWaker};
    let Some(slot) = device_slot(id) else {
        return false;
    };
    let mut link = slot.window.lock();
    if link.is_some() {
        return true; // already running (idempotent)
    }
    // FrameSlot is a std::sync::Mutex (owned by the window module); lib.rs's
    // bare `Mutex` is parking_lot, so qualify it here.
    let frames: FrameSlot = Arc::new(std::sync::Mutex::new(None));
    // Created here rather than on the window thread so the link holds it before
    // the loop exists: an unarmed waker is a no-op and the window's backstop
    // covers the gap, so a publish that beats the loop's creation costs latency
    // rather than a frame.
    let wake = WindowWaker::new();
    // Weak so a live window does not pin a destroyed device; post-destroy input
    // upgrades to None and is dropped (the guest is gone anyway).
    let weak = Arc::downgrade(&slot);
    let on_input: InputSink = Arc::new(move |action: HostAction| {
        let Some(dev) = weak.upgrade() else {
            return;
        };
        dev.prompt_actions.lock().push_back(action);
        // Wake the HostAction-delivery BH so the guest sees the input without
        // waiting for the next drain tranche (same rail as IRQ/cursor).
        if let Some(ops) = dev.ops {
            if let Some(notify) = ops.notify_actions {
                // SAFETY: QEMU owns ctx for the device lifetime; notify_actions
                // is the thread-safe BH-schedule callback.
                unsafe { notify(ops.ctx) }
            }
        }
    });
    let cfg = WindowConfig {
        title: "Reims vGPU".to_string(),
        width: if width == 0 {
            crate::model::EFI_BOOT_WIDTH
        } else {
            width
        },
        height: if height == 0 {
            crate::model::EFI_BOOT_HEIGHT
        } else {
            height
        },
    };
    let stop: crate::host_window::present::StopFlag =
        Arc::new(std::sync::atomic::AtomicBool::new(false));
    #[cfg(target_os = "macos")]
    let (thread, exited) = {
        let exited: crate::host_window::present::ExitedFlag =
            Arc::new(std::sync::atomic::AtomicBool::new(false));
        if let Err(error) = crate::host_window::present::start_main_thread(
            id,
            cfg,
            on_input,
            Arc::clone(&frames),
            Arc::clone(&stop),
            Arc::clone(&exited),
            Arc::clone(&wake),
        ) {
            crate::observe::Emit::decline("host_window_start", &error)
                .field("id", id)
                .fail();
            return false;
        }
        (None, exited)
    };
    #[cfg(not(target_os = "macos"))]
    let thread = Some(crate::host_window::present::spawn(
        cfg,
        on_input,
        Arc::clone(&frames),
        Arc::clone(&stop),
        Arc::clone(&wake),
    ));
    *link = Some(WindowLink {
        frames,
        wake,
        last: (u32::MAX, u32::MAX, u32::MAX, u64::MAX),
        seq: 0,
        bgra_short_geom: None,
        stop,
        thread,
        #[cfg(target_os = "macos")]
        exited,
    });
    crate::observe::off(format!(
        "host_window_start id={id} {}x{}",
        if width == 0 {
            crate::model::EFI_BOOT_WIDTH
        } else {
            width
        },
        if height == 0 {
            crate::model::EFI_BOOT_HEIGHT
        } else {
            height
        }
    ));
    true
}

/// No-op stub when the `host-window` feature is off: the FFI symbol still links
/// (so the C shim binds regardless) but there is no window to start.
#[cfg(not(feature = "host-window"))]
pub fn device_window_start(_id: u64, _width: u32, _height: u32) -> bool {
    false
}

/// Run the main-thread-owned macOS window. QEMU calls this from its process-main
/// UI entry after device realize; it blocks until the window exits.
#[cfg(all(feature = "host-window", target_os = "macos"))]
pub fn device_window_run_main(id: u64) -> bool {
    match crate::host_window::present::run_main_thread(id) {
        Ok(()) => true,
        Err(error) => {
            crate::observe::Emit::decline("host_window_main", &error)
                .field("id", id)
                .fail();
            false
        }
    }
}

#[cfg(not(all(feature = "host-window", target_os = "macos")))]
pub fn device_window_run_main(_id: u64) -> bool {
    false
}

/// Publish the current finished present frame into the window's frame slot, if a
/// window is running and this present has not been published yet. Runs on the
/// drain worker under no device lock of its own (its own small mutex), so it
/// never contends the render tranche. Latest-wins.
#[cfg(feature = "host-window")]
pub(crate) fn publish_window_frame(slot: &BoundDevice, state: &mut crate::model::DeviceState) {
    use crate::runtime::drain::{note_window_publish, WindowPublish};
    let mut guard = slot.window.lock();
    let Some(link) = guard.as_mut() else {
        // No window consumes the capture: revert the next capture to the full
        // readback path (a torn-down window must not leave `frame_bgra` stale
        // behind an unreset `display_from_resident`).
        state.present.display_from_resident = false;
        note_window_publish(WindowPublish::NoWindow);
        return;
    };
    let p = &state.present;
    if !p.frame_valid || p.frame_width == 0 || p.frame_height == 0 {
        note_window_publish(WindowPublish::NoFrame);
        return;
    }
    let key = window_frame_key(p);
    if key == link.last {
        note_window_publish(WindowPublish::SameKey);
        return;
    }
    note_window_publish(WindowPublish::Fresh);
    // Copied out rather than held behind `p`: the branches below assign
    // `state.present.display_from_resident`, and the frame bytes are the only
    // thing that still has to be read through the borrow.
    let (mapping, width, height, generation) = (
        p.frame_mapping,
        p.frame_width,
        p.frame_height,
        p.frame_generation,
    );
    let need = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(4);
    if need == 0 {
        return;
    }
    let present_identity =
        crate::runtime::present_identity::surface_identity(state, mapping, width, height);
    // Keep the resident this present names alive across the idle sweep below,
    // then reclaim targets idle past the wall-clock age threshold so VRAM returns
    // to the working-set baseline after a compositing burst instead of being held
    // for the guest lifetime.
    let now_ms = crate::observe::elapsed_ms() as u64;
    crate::backend::vulkan::engine::touch_resident_target(Some(&present_identity), now_ms);
    crate::backend::vulkan::engine::maintain_idle_residents(Some(&present_identity), now_ms);
    // The window presenting from the engine's own device can take the resident
    // as it stands, so the frame never crosses host memory. `display_from_resident`
    // is what tells the NEXT capture not to read it back, and it is only set
    // when a resident actually carried this one.
    if crate::backend::vulkan::engine::window_present_attached()
        && crate::backend::vulkan::engine::resident_presentable(&present_identity, width, height)
    {
        let resident_source = crate::backend::vulkan::engine::WindowPresentSource {
            width,
            height,
            identity: present_identity,
        };
        let published = window_write_frame(link, width, height, Vec::new(), Some(resident_source));
        crate::runtime::census::present_proxy::window_publish::note(published);
        if published {
            link.last = key;
            state.present.display_from_resident = true;
        }
        return;
    }
    // Say why the direct present was not taken, because the fallback below
    // copies the whole framebuffer through host memory on every frame and the
    // difference between the two is the window's frame rate. Silence here is
    // what let `direct_frac` sit at 0.00 for a whole boot with no cause named.
    if !crate::backend::vulkan::engine::window_present_attached() {
        crate::runtime::drain::note_store_route("winpub_window_not_attached");
    } else if let Some(route) = crate::backend::vulkan::engine::resident_present_decline_route(
        &present_identity,
        width,
        height,
    ) {
        crate::runtime::drain::note_store_route(route);
    }
    // No resident carries this present (firmware framebuffer, a mapping the
    // compositor cleared but never rendered into, the frames after a device
    // reset), or the window is driving its own device because the engine's
    // cannot present to this surface. Either way the window needs CPU pixels,
    // and the next capture must read them back.
    state.present.display_from_resident = false;
    if state.present.frame_bgra.len() < need {
        // No usable CPU frame: nothing to publish. Reachable via keep-prior
        // when a capture FAILS at a new/larger geometry (dims advanced, the
        // buffer kept the smaller prior), and on the present right after a
        // resident-carried one, whose capture deliberately left the buffer
        // empty. Skipping is correct (never publish a short/torn frame; the
        // window holds its last good frame), but silence would hide "the window
        // froze because captures keep failing at this geometry". Fail-visible +
        // deduped per geometry so a persistent mismatch logs once, not every
        // present (no flood).
        if link.bgra_short_geom != Some((width, height)) {
            link.bgra_short_geom = Some((width, height));
            crate::observe::off(format!(
                "publish_window_frame DROP reason=frame_bgra_short mid={} {}x{} \
                 have={} need={need} gen={}",
                mapping,
                width,
                height,
                state.present.frame_bgra.len(),
                generation
            ));
        }
        crate::runtime::census::present_proxy::window_publish::note(false);
        return;
    }
    // A well-formed frame cleared the short-buffer condition; re-arm the latch
    // so a later mismatch at the same geometry logs again.
    link.bgra_short_geom = None;
    let bgra = state.present.frame_bgra[..need].to_vec();
    let published = window_write_frame(link, width, height, bgra, None);
    crate::runtime::census::present_proxy::window_publish::note(published);
    if published {
        link.last = key;
    }
}

/// Write a frame into the window's slot, stamping the next monotonic `seq` so
/// the window prepares only new content, and wake the window's event loop to
/// come and read it. Returns false if the slot lock is
/// poisoned (a panicked window thread — the window is gone, drop the publish).
/// Bound so the inner guard drops before the caller's outer `window` guard.
///
/// The wake is after the store and outside the slot lock: the loop wakes to a
/// frame that is already there, and it never blocks on a lock this thread is
/// still holding. A wake that does not land is not a lost frame — the window's
/// backstop still runs — which is why nothing here checks that it did.
#[cfg(feature = "host-window")]
fn window_write_frame(
    link: &mut WindowLink,
    width: u32,
    height: u32,
    bgra: Vec<u8>,
    resident: Option<crate::backend::vulkan::engine::WindowPresentSource>,
) -> bool {
    link.seq = link.seq.wrapping_add(1);
    let frame = std::sync::Arc::new(crate::host_window::present::Frame {
        seq: link.seq,
        width,
        height,
        bgra,
        resident,
    });
    let stored = match link.frames.lock() {
        Ok(mut slot_frame) => {
            *slot_frame = Some(frame);
            true
        }
        Err(_) => false,
    };
    if stored {
        link.wake.wake();
    }
    stored
}

/// Stop the host-owned window during VM teardown. Sets the stop flag so the
/// event loop exits, then waits for its Vulkan objects to tear down before QEMU
/// proceeds to process/driver teardown. Linux joins the dedicated window thread;
/// macOS waits for the process-main loop's exit publication. Idempotent; no-op
/// without a window.
#[cfg(feature = "host-window")]
pub fn device_window_stop(id: u64) -> bool {
    let Some(slot) = device_slot(id) else {
        return false;
    };
    let link = slot.window.lock().take();
    let Some(mut link) = link else {
        return true; // no window (or already stopped)
    };
    link.stop.store(true, std::sync::atomic::Ordering::Relaxed);
    #[cfg(target_os = "macos")]
    {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !link.exited.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        if !link.exited.load(Ordering::Acquire) {
            crate::observe::fail(format!(
                "host_window_stop FAIL reason=main_thread_teardown_timeout id={id}"
            ));
            return false;
        }
    }
    if let Some(thread) = link.thread.take() {
        // The window thread's `WindowError` return was discarded here, so a
        // `build_event_loop`/`run_app` failure on the Linux spawn path vanished
        // with no line. Emit the typed decline instead. (macOS never takes this
        // branch — its window runs on the process main thread, so `thread` is
        // None; the join runs only on the Linux `spawn` path.)
        match thread.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                crate::observe::Emit::decline("host_window_run", &error)
                    .field("id", id)
                    .fail();
            }
            // A panic in the window thread; the default panic hook already wrote
            // its message to stderr, and there is no guest command to decline.
            Err(_) => {}
        }
    }
    true
}

#[cfg(not(feature = "host-window"))]
pub fn device_window_stop(_id: u64) -> bool {
    false
}

/// Register the early-boot framebuffer (BAR1 GOP host RAM) so the window can
/// show UEFI/OpenCore/boot.efi output before the product present path latches.
/// `ptr` is a stable RAMBlock host pointer valid for the device lifetime.
///
/// SAFETY: the caller guarantees `ptr` addresses at least `stride * height`
/// readable bytes for the device lifetime (the QEMU BAR1 RAMBlock).
#[cfg(feature = "host-window")]
pub fn device_window_set_early_fb(
    id: u64,
    ptr: usize,
    stride: u32,
    width: u32,
    height: u32,
) -> bool {
    let Some(slot) = device_slot(id) else {
        return false;
    };
    if ptr == 0 || stride == 0 || width == 0 || height == 0 {
        return false;
    }
    *slot.early_fb.lock() = Some(EarlyFb {
        ptr,
        stride,
        width,
        height,
    });
    true
}

#[cfg(not(feature = "host-window"))]
pub fn device_window_set_early_fb(
    _id: u64,
    _ptr: usize,
    _stride: u32,
    _width: u32,
    _height: u32,
) -> bool {
    false
}

/// Pre-boundary early-console pump: while the guest is still on the BAR1/EFI
/// console (no product present latched), push that framebuffer to the window so
/// early boot is visible. Runs on the poll (heartbeat) path so it works headless
/// (`-display none` never ticks `gfx_update`). Gated by `host_console_uses_bar1`
/// — the same protocol-state ownership rule the C `gfx_update` uses, so the
/// window never fights the product present for the frame — and throttled to
/// ~30 fps so it does not memcpy the FB every 4 ms.
#[cfg(feature = "host-window")]
pub(crate) fn publish_window_early_frame<
    M: crate::runtime::host::HostMemory + crate::runtime::host::HostOps,
>(
    slot: &BoundDevice,
    state: &crate::model::DeviceState,
    host: &M,
    now_ns: u64,
) {
    let mut guard = slot.window.lock();
    let Some(link) = guard.as_mut() else {
        return;
    };
    // Console-ownership gate (mirror of host_console_uses_bar1): only feed the
    // window while it is on the early console, never after the product present
    // owns it or a same-geom early front is latched (the drain publishes those).
    let early_latched = crate::runtime::scanout::early_scanout_target(state).is_some();
    if !host_console_uses_bar1(state.present.frame_flush_seen, early_latched) {
        return;
    }
    // ~30 fps throttle (33 ms) on the 4 ms poll.
    let last = slot.early_last_ns.load(Ordering::Relaxed);
    if now_ns.saturating_sub(last) < 33_000_000 {
        return;
    }
    let w = crate::model::EFI_BOOT_WIDTH;
    let h = crate::model::EFI_BOOT_HEIGHT;
    let stride = w.saturating_mul(4);
    let mut buf = vec![0u8; (stride as usize).saturating_mul(h as usize)];
    // Prefer the guest-programmed EFI FB (kernel-relocated console), else the
    // BAR1 GOP framebuffer the option ROM drives — the same order as C's
    // reims_vgpu_pci_copy_early_console.
    let painted = if state.gfx.efi_fb_start != 0 {
        crate::runtime::scanout::paint_efi_console(state, host, &mut buf, stride, w, h)
    } else {
        false
    };
    let painted = painted || copy_early_bar1(slot, &mut buf, stride, w, h);
    if !painted {
        return;
    }
    slot.early_last_ns.store(now_ns, Ordering::Relaxed);
    // Early boot frames come from the BAR1 GOP framebuffer, not a resident
    // target, so there is no resident source to hand over.
    window_write_frame(link, w, h, buf, None);
}

/// Copy the registered BAR1 early framebuffer into `dst` (tight BGRA8). Returns
/// false when no early FB is registered or its geometry cannot cover the request.
#[cfg(feature = "host-window")]
fn copy_early_bar1(slot: &BoundDevice, dst: &mut [u8], dst_stride: u32, w: u32, h: u32) -> bool {
    let efb = *slot.early_fb.lock();
    let Some(efb) = efb else {
        return false;
    };
    if efb.ptr == 0 || efb.width < w || efb.height < h {
        return false;
    }
    let src_len = (efb.stride as usize).saturating_mul(efb.height as usize);
    // SAFETY: efb.ptr is the BAR1 RAMBlock host pointer registered by the C shim
    // at realize, valid for the device lifetime and at least stride*height bytes
    // (device_window_set_early_fb contract). The guest may write concurrently; a
    // torn read only flickers one early-boot frame.
    let src = unsafe { std::slice::from_raw_parts(efb.ptr as *const u8, src_len) };
    let row = (w as usize).saturating_mul(4);
    for y in 0..h as usize {
        let so = y.saturating_mul(efb.stride as usize);
        let doff = y.saturating_mul(dst_stride as usize);
        if so + row > src.len() || doff + row > dst.len() {
            return false;
        }
        dst[doff..doff + row].copy_from_slice(&src[so..so + row]);
    }
    true
}
