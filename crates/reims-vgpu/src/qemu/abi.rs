//! Versioned C ABI for the QEMU thin shim.
//!
//! - opaque handles
//! - repr(C) fixed-width structures
//! - ABI version + struct-size fields
//! - explicit status codes
//! - catch_unwind on every entry
//!
//! # Which thread arrives at each entry point
//!
//! This is the boundary QEMU's threads cross, so it is the only place the map
//! is true for both shims at once. Nothing here enforces it — it is what the two
//! shims do, and it is why [`crate::backend::vulkan::engine`]'s lock census
//! separates `worker` from `device`.
//!
//! | Work | x86 / PCI | arm64 / MMIO |
//! |---|---|---|
//! | Guest command execution (`device_drain`) | dedicated `reims-vgpu-pci-drain` thread | **the vCPU thread, inside its MMIO store** |
//! | `HostAction` delivery (`device_pop_action`) | main-loop BH | main-loop BH |
//! | Poll / re-drive (`device_poll`) | 4 ms heartbeat thread | 4 ms main-loop timer |
//! | Window event loop and `vkQueuePresentKHR` | dedicated `reims-vgpu-window` thread | **process main thread** (AppKit) |
//!
//! **No pathway executes guest GPU work on QEMU's main loop**, and neither of
//! the two exceptions above is an oversight:
//!
//! * The arm64 drain runs on the vCPU because the mapper rail resolves guest
//!   *virtual* addresses, and `reims_vgpu_shim_read_kva` needs `current_cpu`
//!   set. Moving it to a worker would have to fall back to `first_cpu` or
//!   `do_run_on_cpu`, and the shim header records why that is an AB-BA hang
//!   rather than a slower answer. The cost is real and is the guest's own: a
//!   tranche stalls the vCPU that handed it over, and `engine_lock`'s `device`
//!   counters are what price it.
//! * The macOS window loop runs on the process main thread because AppKit
//!   requires it. QEMU's Darwin wrapper has already moved emulation off that
//!   thread by then, so the two do not share.
//!
//! # Safety
//! Every exported unsafe entry point is called through the matching C header.
//! Opaque device pointers must come from `reims_vgpu_device_create`; input buffers must
//! remain readable for their declared lengths; output buffers and callback
//! tables must remain writable for the duration of the call.
#![allow(
    clippy::missing_safety_doc,
    reason = "the shared QEMU C ABI safety contract is documented at module scope"
)]

use crate::qemu::host_ops::ReimsVgpuHostOps;
use crate::runtime::host::HostAction;
use crate::{
    backend_name, device_console_feed, device_create, device_cursor_glyph_copy,
    device_cursor_glyph_info, device_destroy, device_drain, device_efi_console_copy,
    device_gfx_read, device_gfx_write, device_iosfc_read, device_iosfc_write, device_poll,
    device_pop_action, device_reset, device_scanout_copy, device_scanout_may_paint,
    device_window_run_main, device_window_set_early_fb, device_window_start, device_window_stop,
    unwind_safe, ConsoleFeed, CursorGlyphInfo,
};
use std::os::raw::{c_char, c_int};
use std::slice;

/// Bump when breaking the C shim contract.
///
/// v15 adds `reims_vgpu_qemu_scanout_may_paint`, the console-ownership verdict
/// for one presented mapping. v14 moved the three-way *kind* into Rust but went
/// on exporting it as an input, and the shims did with it what shims do: the x86
/// one rebuilt "may this paint" from the kind and the mapping id, the arm64 one
/// built nothing and painted every present it was handed. Exporting the inputs
/// to a rule is exporting the rule.
/// v14 replaces `present_boundary_seen` + `early_scanout_target` with the single
/// `console_feed`. The shims took the old pair together and branched on it, so
/// the console-ownership rule existed in C twice and in Rust once more; the
/// branch is product policy and a thin shim does not hold one. Removing both
/// symbols rather than leaving them is deliberate — a shim that can still
/// assemble its own answer will eventually do so again.
/// v13 adds `guest_written_pages` on [`ReimsVgpuHostOps`]: the per-page form of
/// v12's generation. A whole-set generation is enough to decide whether to reuse
/// a host-side copy, and not enough to decide what to write back — a deferred
/// writeback that discards its frame because one page moved loses the Store, and
/// one that writes the whole frame anyway loses the guest's own store.
/// v12 adds the guest-write tracking triple on [`ReimsVgpuHostOps`]:
/// `track_guest_writes`, `untrack_guest_writes`, `guest_write_gen`. A surface's
/// pages are plain guest RAM and the guest CPU stores into them with no device
/// operation, so no counter this crate keeps can witness such a store and every
/// host-side copy of those pages is stale from that instant with nothing to say
/// so. The hypervisor's dirty bitmap is the only witness; these are the door.
/// v11 adds `reims_vgpu_qemu_window_run_main`, which lets the Darwin MMIO shim make the
/// AppKit-owned winit loop QEMU's process-main UI entry.
/// v9 adds the host-window lifecycle + early framebuffer: `reims_vgpu_qemu_window_stop`
/// (close + join on VM teardown), `reims_vgpu_qemu_window_set_early_fb` (register BAR1
/// GOP so the window shows early boot), and the `WindowClosed` HostAction kind
/// (11) the window emits on a UI close so the shim requests a VM shutdown.
/// v8 adds `reims_vgpu_qemu_window_start` (host-owned presentation window; see
/// [[host-window]]). The symbol is always present; when the staticlib was built
/// without the `host-window` feature it returns `REIMS_VGPU_QEMU_ERR_STATE` so the C
/// shim falls back to QEMU's own display.
pub const REIMS_VGPU_QEMU_ABI_VERSION: u32 = 19;

#[repr(C)]
pub struct ReimsVgpuQemuCreateInfo {
    pub abi_version: u32,
    pub struct_size: u32,
    /// QEMU host-service table (GPA / clock / schedule_bh). Null for tests.
    pub host_ops: *const ReimsVgpuHostOps,
    /// Guest page shift: 12 (x86 Tahoe) or 14 (arm64e). 0 is invalid (no default).
    pub guest_page_shift: u32,
}

#[repr(C)]
pub struct ReimsVgpuQemuDevice {
    pub abi_version: u32,
    pub struct_size: u32,
    pub handle: u64,
}

pub const REIMS_VGPU_QEMU_OK: c_int = 0;
pub const REIMS_VGPU_QEMU_ERR_ARGS: c_int = 1;
pub const REIMS_VGPU_QEMU_ERR_STATE: c_int = 2;
pub const REIMS_VGPU_QEMU_ERR_PANIC: c_int = 3;
/// pop_action: queue empty (not a hard failure).
pub const REIMS_VGPU_QEMU_EMPTY: c_int = 4;

/// Why `guest_ram_regions` refused, when it did. Negative so one return carries
/// both a span count and a named refusal.
///
/// Same two-copies problem as everything else crossing this boundary, and
/// `the_abi_header_agrees_on_the_guest_ram_codes` is the only comparison. A
/// drift here reads as "this machine has no RAM" for what was an argument bug,
/// on exactly one pathway.
pub const REIMS_VGPU_GUEST_RAM_ERR_ARGS: c_int = -1;
pub const REIMS_VGPU_GUEST_RAM_ERR_NO_RAM: c_int = -2;

/// What a successful `map_pages` says about the pointer it just returned; see
/// [`crate::runtime::host::PageAlias`].
///
/// Non-negative so one return carries both the verdict and the failure codes,
/// the same shape `guest_ram_regions` uses. `TRANSIENT` is 0 because that is
/// what every previous ABI's success return meant, so a shim built against an
/// older header and loaded anyway degrades to the conservative answer rather
/// than to the one that licenses retaining a view past its release.
pub const MAP_PAGES_TRANSIENT: c_int = 0;
pub const MAP_PAGES_STABLE: c_int = 1;

fn copy_host_ops(ops: *const ReimsVgpuHostOps) -> Option<ReimsVgpuHostOps> {
    if ops.is_null() {
        return None;
    }
    // SAFETY: QEMU passes a live ReimsVgpuHostOps for the device lifetime.
    let ops = unsafe { &*ops };
    if ops.abi_version != REIMS_VGPU_QEMU_ABI_VERSION {
        return None;
    }
    if (ops.struct_size as usize) < std::mem::size_of::<ReimsVgpuHostOps>() {
        return None;
    }
    Some(*ops)
}

/// SAFETY: `out` must be valid for write when non-null; `info` may be null (defaults).
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_device_create(
    info: *const ReimsVgpuQemuCreateInfo,
    out: *mut ReimsVgpuQemuDevice,
) -> c_int {
    unwind_safe(
        "reims_vgpu_qemu_device_create",
        || {
            if out.is_null() {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            let mut ops = None;
            let mut page_shift = 0u32;
            if !info.is_null() {
                // SAFETY: caller-provided create info.
                let info = unsafe { &*info };
                if info.abi_version != REIMS_VGPU_QEMU_ABI_VERSION {
                    return REIMS_VGPU_QEMU_ERR_ARGS;
                }
                if (info.struct_size as usize) < std::mem::size_of::<ReimsVgpuQemuCreateInfo>() {
                    return REIMS_VGPU_QEMU_ERR_ARGS;
                }
                page_shift = info.guest_page_shift;
                if !info.host_ops.is_null() {
                    match copy_host_ops(info.host_ops) {
                        Some(o) => ops = Some(o),
                        None => return REIMS_VGPU_QEMU_ERR_ARGS,
                    }
                }
            }
            let handle = match device_create(ops, page_shift) {
                Some(h) => h,
                None => return REIMS_VGPU_QEMU_ERR_ARGS,
            };
            if handle == 0 {
                return REIMS_VGPU_QEMU_ERR_STATE;
            }
            // SAFETY: out is non-null.
            unsafe {
                *out = ReimsVgpuQemuDevice {
                    abi_version: REIMS_VGPU_QEMU_ABI_VERSION,
                    struct_size: std::mem::size_of::<ReimsVgpuQemuDevice>() as u32,
                    handle,
                };
            }
            REIMS_VGPU_QEMU_OK
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// SAFETY: handle from create; no-op if unknown.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_device_reset(handle: u64) -> c_int {
    unwind_safe(
        "reims_vgpu_qemu_device_reset",
        || {
            if handle == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            if device_reset(handle) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_ERR_STATE
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// SAFETY: handle from create; destroy is idempotent for unknown ids (ERR_STATE).
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_device_destroy(handle: u64) -> c_int {
    unwind_safe(
        "reims_vgpu_qemu_device_destroy",
        || {
            if handle == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            if device_destroy(handle) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_ERR_STATE
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Start the host-owned presentation window (winit + VkSurfaceKHR) for this
/// device ([[host-window]]). Spawns the window on a dedicated thread; the drain
/// publishes each finished present frame to it, and window input (keys, pointer,
/// wheel) is injected via the neutral `Input*` prompt-action rail.
///
/// `width`/`height` seed the initial window size (0 → the boot EFI geometry).
/// Idempotent: a second call while the window is up is a no-op success.
///
/// Returns `REIMS_VGPU_QEMU_ERR_STATE` when the staticlib was built without the
/// `host-window` feature (C then leaves QEMU's own display in charge) or when
/// the handle is unknown.
///
/// SAFETY: `handle` from create.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_window_start(
    handle: u64,
    width: u32,
    height: u32,
) -> c_int {
    unwind_safe(
        "reims_vgpu_qemu_window_start",
        || {
            if handle == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            if device_window_start(handle, width, height) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_ERR_STATE
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Run a main-thread-owned host window until it exits.
///
/// Returns after UI close/backend stop. `REIMS_VGPU_QEMU_ERR_STATE` means this build or
/// platform has no process-main host window for `handle`.
///
/// SAFETY: `handle` from create; call on the same main thread as window start.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_window_run_main(handle: u64) -> c_int {
    unwind_safe(
        "reims_vgpu_qemu_window_run_main",
        || {
            if handle == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            if device_window_run_main(handle) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_ERR_STATE
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Stop the host-owned window and join its thread (VM teardown). Sets the stop
/// flag, the event loop exits, and the window's Vulkan objects tear down before
/// this returns — so call it before `reims_vgpu_qemu_device_destroy` and before the
/// process/driver teardown. Idempotent; `REIMS_VGPU_QEMU_OK` even with no window.
/// SAFETY: `handle` from create.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_window_stop(handle: u64) -> c_int {
    unwind_safe(
        "reims_vgpu_qemu_window_stop",
        || {
            if handle == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            if device_window_stop(handle) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_ERR_STATE
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Register the early-boot framebuffer (BAR1 GOP host RAM) so the window shows
/// UEFI/OpenCore/boot.efi output before the product present path latches. `ptr`
/// must stay valid (and hold at least `stride * height` bytes) for the device
/// lifetime — pass the BAR1 RAMBlock host pointer. Tight BGRA8 assumed.
///
/// SAFETY: `handle` from create; `ptr` valid for `stride * height` bytes for the
/// device lifetime.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_window_set_early_fb(
    handle: u64,
    ptr: *const u8,
    stride: u32,
    width: u32,
    height: u32,
) -> c_int {
    unwind_safe(
        "reims_vgpu_qemu_window_set_early_fb",
        || {
            if handle == 0 || ptr.is_null() {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            if device_window_set_early_fb(handle, ptr as usize, stride, width, height) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_ERR_STATE
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Write backend name into caller buffer (NUL-terminated).
/// SAFETY: buf must have buf_len bytes when non-null.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_backend_name(buf: *mut c_char, buf_len: usize) -> c_int {
    unwind_safe(
        "reims_vgpu_qemu_backend_name",
        || {
            if buf.is_null() || buf_len == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            // SAFETY: the header requires `buf` valid for `buf_len` bytes.
            unsafe { crate::qemu::cstr::write_c_str(buf, buf_len, backend_name()) };
            REIMS_VGPU_QEMU_OK
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// ABI version getter (no allocation).
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_abi_version() -> u32 {
    unwind_safe(
        "reims_vgpu_qemu_abi_version",
        || REIMS_VGPU_QEMU_ABI_VERSION,
        0,
    )
}

/// Gfx MMIO read. SAFETY: out_val non-null.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_gfx_read(
    handle: u64,
    offset: u64,
    size: u32,
    out_val: *mut u64,
) -> c_int {
    unwind_safe(
        "reims_vgpu_qemu_gfx_read",
        || {
            if handle == 0 || out_val.is_null() {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            match device_gfx_read(handle, offset, size) {
                Some(v) => {
                    // SAFETY: out_val non-null.
                    unsafe {
                        *out_val = v;
                    }
                    REIMS_VGPU_QEMU_OK
                }
                None => REIMS_VGPU_QEMU_ERR_STATE,
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Gfx MMIO write (may schedule QEMU BH via HostOps).
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_gfx_write(
    handle: u64,
    offset: u64,
    data: u64,
    size: u32,
) -> c_int {
    unwind_safe(
        "reims_vgpu_qemu_gfx_write",
        || {
            if handle == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            if device_gfx_write(handle, offset, data, size) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_ERR_STATE
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Iosfc MMIO read. SAFETY: out_val non-null.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_iosfc_read(
    handle: u64,
    offset: u64,
    size: u32,
    out_val: *mut u64,
) -> c_int {
    unwind_safe(
        "reims_vgpu_qemu_iosfc_read",
        || {
            if handle == 0 || out_val.is_null() {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            match device_iosfc_read(handle, offset, size) {
                Some(v) => {
                    unsafe {
                        *out_val = v;
                    }
                    REIMS_VGPU_QEMU_OK
                }
                None => REIMS_VGPU_QEMU_ERR_STATE,
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Iosfc MMIO write (may schedule QEMU BH via HostOps).
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_iosfc_write(
    handle: u64,
    offset: u64,
    data: u64,
    size: u32,
) -> c_int {
    unwind_safe(
        "reims_vgpu_qemu_iosfc_write",
        || {
            if handle == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            if device_iosfc_write(handle, offset, data, size) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_ERR_STATE
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// BH body: drain pending FIFOs (GPA via HostOps). Then pop actions with
/// [`reims_vgpu_qemu_device_pop_action`].
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_device_drain(handle: u64) -> c_int {
    unwind_safe(
        "reims_vgpu_qemu_device_drain",
        || {
            if handle == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            // Before the drain, so the first tranche's engine-lock acquires are
            // already attributed to the worker. Entering this function is the
            // only property that distinguishes the drain thread from a vCPU
            // inside an MMIO store, and telling those apart is what makes a
            // stalled guest attributable — see `EngineLockSite`.
            #[cfg(feature = "backend-vulkan")]
            crate::backend::vulkan::engine::mark_drain_thread();
            if device_drain(handle) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_ERR_STATE
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Periodic poll (gfx_update): display ONLINE re-drive after guest enable().
/// Deliver HostActions with [`reims_vgpu_qemu_device_pop_action`] after this call.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_device_poll(handle: u64) -> c_int {
    unwind_safe(
        "reims_vgpu_qemu_device_poll",
        || {
            if handle == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            if device_poll(handle) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_ERR_STATE
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Pop one HostAction for the QEMU BH. Returns REIMS_VGPU_QEMU_OK with *out filled,
/// REIMS_VGPU_QEMU_EMPTY when the queue is empty.
/// SAFETY: out non-null when a value is expected.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_device_pop_action(
    handle: u64,
    out: *mut HostAction,
) -> c_int {
    unwind_safe(
        "reims_vgpu_qemu_device_pop_action",
        || {
            if handle == 0 || out.is_null() {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            match device_pop_action(handle) {
                Some(a) => {
                    // SAFETY: out non-null.
                    unsafe {
                        *out = a;
                    }
                    REIMS_VGPU_QEMU_OK
                }
                None => REIMS_VGPU_QEMU_EMPTY,
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Which source owns the host console right now, as one answer.
///
/// REIMS_VGPU_QEMU_OK fills `*out_kind` with a `REIMS_VGPU_CONSOLE_FEED_*`;
/// ERR_STATE if no device. The four geometry outs are written only for `Early`,
/// and may be null when the caller wants the kind alone.
///
/// This replaced `present_boundary_seen` + `early_scanout_target`, which the
/// shims took together and recombined into a three-way branch of their own. The
/// branch is the console-ownership rule, it is product policy, and C is a thin
/// shim — so it is answered here, by [`crate::device_console_feed`], which is
/// the same predicate the host-window path already uses.
///
/// SAFETY: `out_kind` non-null; the geometry outs either null or writable.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_console_feed(
    handle: u64,
    out_kind: *mut u32,
    out_mapping_id: *mut u32,
    out_width: *mut u32,
    out_height: *mut u32,
    out_generation: *mut u32,
) -> c_int {
    unwind_safe(
        "reims_vgpu_qemu_console_feed",
        || {
            if handle == 0 || out_kind.is_null() {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            let Some(feed) = device_console_feed(handle) else {
                return REIMS_VGPU_QEMU_ERR_STATE;
            };
            unsafe {
                *out_kind = feed.kind();
            }
            if let ConsoleFeed::Early {
                mapping_id,
                width,
                height,
                generation,
            } = feed
            {
                // Written one at a time rather than behind a single all-or-nothing
                // null check: a caller that wants only the kind passes null for
                // every one of these, and refusing that would put the shim back in
                // the business of holding a scratch tuple it does not use.
                unsafe {
                    if !out_mapping_id.is_null() {
                        *out_mapping_id = mapping_id;
                    }
                    if !out_width.is_null() {
                        *out_width = width;
                    }
                    if !out_height.is_null() {
                        *out_height = height;
                    }
                    if !out_generation.is_null() {
                        *out_generation = generation;
                    }
                }
            }
            REIMS_VGPU_QEMU_OK
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// May a present naming `mapping_id` paint the host console right now?
///
/// REIMS_VGPU_QEMU_OK fills `*out_may` with 0 or 1; ERR_STATE if no device.
///
/// This is the answer, not the inputs. Both shims call it before painting a
/// presented mapping. The x86 shim used to assemble it from
/// [`reims_vgpu_qemu_console_feed`]'s `out_kind` and `out_mapping_id`, and the
/// arm64 shim did not gate at all — so the same present that the x86 console
/// refused as a pre-boundary steal was painted on arm64. Exporting the kind
/// without the verdict is what let those two drift; see
/// [`crate::device_scanout_may_paint`].
///
/// SAFETY: `out_may` non-null and writable.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_scanout_may_paint(
    handle: u64,
    mapping_id: u32,
    out_may: *mut u32,
) -> c_int {
    unwind_safe(
        "reims_vgpu_qemu_scanout_may_paint",
        || {
            if handle == 0 || out_may.is_null() {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            let Some(may) = device_scanout_may_paint(handle, mapping_id) else {
                return REIMS_VGPU_QEMU_ERR_STATE;
            };
            unsafe {
                *out_may = u32::from(may);
            }
            REIMS_VGPU_QEMU_OK
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Pre-boundary early console: copy guest EFI FB (MMIO 0x1210) into `dst`.
///
/// REIMS_VGPU_QEMU_OK when efi_fb_start is programmed and GPA read succeeds.
/// REIMS_VGPU_QEMU_EMPTY when efi_fb_start == 0 (C falls back to BAR1 GOP RAM).
///
/// SAFETY: `dst` valid for dst_stride*height.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_efi_console_copy(
    handle: u64,
    dst: *mut u8,
    dst_stride: u32,
    width: u32,
    height: u32,
) -> c_int {
    unwind_safe(
        "reims_vgpu_qemu_efi_console_copy",
        || {
            if handle == 0 || dst.is_null() || width == 0 || height == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            let len = (dst_stride as usize).saturating_mul(height as usize);
            if len == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            let buf = unsafe { slice::from_raw_parts_mut(dst, len) };
            if device_efi_console_copy(handle, buf, dst_stride, width, height) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_EMPTY
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Fill a host BGRA8 framebuffer (QEMU DisplaySurface) from a guest mapping.
///
/// `generation` is HostAction.a3 (0 = always paint). Returns REIMS_VGPU_QEMU_EMPTY when
/// content is unchanged (C should skip console update).
///
/// SAFETY: `dst` must be valid for `dst_stride * height` bytes.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_scanout_copy(
    handle: u64,
    mapping_id: u32,
    dst: *mut u8,
    dst_stride: u32,
    width: u32,
    height: u32,
    generation: u32,
) -> c_int {
    use crate::runtime::scanout::ScanoutCopyResult;
    unwind_safe(
        "reims_vgpu_qemu_scanout_copy",
        || {
            if handle == 0 || dst.is_null() || width == 0 || height == 0 || dst_stride == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            let nbytes = (height as usize).saturating_mul(dst_stride as usize);
            if nbytes == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            // SAFETY: caller owns DisplaySurface buffer for nbytes.
            let buf = unsafe { slice::from_raw_parts_mut(dst, nbytes) };
            match device_scanout_copy(
                handle, mapping_id, buf, dst_stride, width, height, generation,
            ) {
                ScanoutCopyResult::Painted => REIMS_VGPU_QEMU_OK,
                ScanoutCopyResult::Unchanged => REIMS_VGPU_QEMU_EMPTY,
                ScanoutCopyResult::Failed => REIMS_VGPU_QEMU_ERR_STATE,
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Cursor glyph geometry. Returns REIMS_VGPU_QEMU_EMPTY when no glyph is ready.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_cursor_glyph_info(
    handle: u64,
    out: *mut CursorGlyphInfo,
) -> c_int {
    unwind_safe(
        "reims_vgpu_qemu_cursor_glyph_info",
        || {
            if handle == 0 || out.is_null() {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            match device_cursor_glyph_info(handle) {
                Some(info) => {
                    unsafe {
                        *out = info;
                    }
                    REIMS_VGPU_QEMU_OK
                }
                None => REIMS_VGPU_QEMU_EMPTY,
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Copy glyph pixels as QEMUCursor ARGB (`0xAARRGGBB`). `count` is capacity in
/// u32 pixels; on success writes min(count, pixel_count) and returns OK.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_cursor_glyph_copy(
    handle: u64,
    out_argb: *mut u32,
    count: usize,
) -> c_int {
    unwind_safe(
        "reims_vgpu_qemu_cursor_glyph_copy",
        || {
            if handle == 0 || out_argb.is_null() || count == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            // SAFETY: caller buffer for count u32s.
            let buf = unsafe { slice::from_raw_parts_mut(out_argb, count) };
            match device_cursor_glyph_copy(handle, buf) {
                Some(n) if n > 0 => REIMS_VGPU_QEMU_OK,
                _ => REIMS_VGPU_QEMU_EMPTY,
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Read `#define NAME <decimal|0xhex>[u]` out of the shared ABI header.
///
/// Test-only, and the only thing in the toolchain that reads the header at all:
/// Rust does not include it and the shims do not read Rust, so every constant
/// crossing the boundary exists as two copies with nothing comparing them. Each
/// caller is the sole check that one of them has not drifted. Takes the first
/// token after the name, because several of these carry a trailing `/* ... */`.
///
/// Hex is accepted because a register-window size is spelled `0x4000` on both
/// sides and restating it in decimal on one of them would be a second
/// transcription for a reader to get wrong — which is the defect this whole
/// function exists to catch.
#[cfg(test)]
pub(crate) fn header_define(name: &str) -> u32 {
    const HEADER: &str = include_str!("../../include/reims_vgpu_qemu_abi.h");
    let tok = HEADER
        .lines()
        .find_map(|l| l.strip_prefix(&format!("#define {name} ")))
        .unwrap_or_else(|| panic!("the shared ABI header must define {name}"))
        .split_whitespace()
        .next()
        .unwrap_or_else(|| panic!("{name} must have a value"))
        .trim_end_matches('u');
    match tok.strip_prefix("0x") {
        Some(hex) => u32::from_str_radix(hex, 16),
        None => tok.parse(),
    }
    .unwrap_or_else(|e| panic!("{name} must be a plain decimal or 0x-hex literal: {e}"))
}

/// [`header_define`] for a `#define NAME <signed decimal>`.
///
/// The refusal codes are negative so one return value can carry both an owned
/// fd and a named refusal, and `u32::from_str` cannot read them. Same job and
/// same reason: these constants exist twice with nothing comparing them.
#[cfg(test)]
pub(crate) fn header_define_i32(name: &str) -> i32 {
    const HEADER: &str = include_str!("../../include/reims_vgpu_qemu_abi.h");
    HEADER
        .lines()
        .find_map(|l| l.strip_prefix(&format!("#define {name} ")))
        .unwrap_or_else(|| panic!("the shared ABI header must define {name}"))
        .split_whitespace()
        .next()
        .unwrap_or_else(|| panic!("{name} must have a value"))
        .parse()
        .unwrap_or_else(|e| panic!("{name} must be a plain signed decimal literal: {e}"))
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::model::{PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};

    /// `reims_vgpu_qemu_device_pop_action` writes a Rust [`HostAction`] straight
    /// through the caller's `ReimsVgpuHostAction *`, so the two declarations are
    /// one struct and the header is the only other place it is written down.
    ///
    /// There used to be a third: a `#[repr(C)] ReimsVgpuHostAction` beside a
    /// `#[repr(u32)] ReimsVgpuHostActionKind`, both mirroring the runtime pair,
    /// with a test that compared the two *Rust* spellings to each other. That
    /// test could not see the header, which is the copy that can actually drift
    /// away from the compiled shim. The mirrors are gone and this checks what
    /// they were standing in for: the field names, their order, and the C types
    /// the shim's compiler lays out from.
    ///
    /// The `u32` kind followed by four `u64`s is why `size_of` is 40 and not 36
    /// — both compilers pad `kind` out to the 8-byte alignment `a0` demands.
    #[test]
    fn the_abi_header_agrees_on_the_host_action_layout() {
        const HEADER: &str = include_str!("../../include/reims_vgpu_qemu_abi.h");
        let body = HEADER
            .split_once("typedef struct ReimsVgpuHostAction {")
            .expect("the header must declare ReimsVgpuHostAction")
            .1
            .split_once('}')
            .expect("the declaration must be closed")
            .0;
        let fields: Vec<&str> = body
            .split(';')
            .map(str::trim)
            .filter(|f| !f.is_empty())
            .collect();
        assert_eq!(
            fields,
            vec![
                "uint32_t kind",
                "uint64_t a0",
                "uint64_t a1",
                "uint64_t a2",
                "uint64_t a3",
            ],
            "the C declaration must stay the one Rust's #[repr(C)] HostAction lays out"
        );

        assert_eq!(std::mem::size_of::<HostAction>(), 40);
        assert_eq!(std::mem::align_of::<HostAction>(), 8);
        assert_eq!(std::mem::offset_of!(HostAction, kind), 0);
        assert_eq!(std::mem::offset_of!(HostAction, a0), 8);
        assert_eq!(std::mem::offset_of!(HostAction, a1), 16);
        assert_eq!(std::mem::offset_of!(HostAction, a2), 24);
        assert_eq!(std::mem::offset_of!(HostAction, a3), 32);
        assert_eq!(
            std::mem::size_of::<crate::runtime::host::HostActionKind>(),
            4,
            "the kind word is the u32 the shim switches on"
        );
    }

    /// The version is the handshake itself: `copy_host_ops` refuses an ops table
    /// whose `abi_version` is not this exact number, so a header and a staticlib
    /// that disagree do not degrade — every device_create fails and the guest
    /// gets no GPU at all. That is a loud failure at boot; this makes it a loud
    /// failure at build, which is where a version bump that touched only one of
    /// the two files is cheap to fix.
    #[test]
    fn the_abi_header_agrees_on_the_version() {
        assert_eq!(
            header_define("REIMS_VGPU_QEMU_ABI_VERSION"),
            REIMS_VGPU_QEMU_ABI_VERSION,
            "the shim header and the staticlib disagree on the ABI version"
        );
    }

    /// The five entry-point return codes agree with the shim header.
    ///
    /// `REIMS_VGPU_QEMU_OK` is the one that matters most and reads the most
    /// harmless. Both shims' `deliver_actions` drain the action queue with
    /// `while (rc = pop_action(..)) == REIMS_VGPU_QEMU_OK`, so a drift there does
    /// not misreport anything — the loop simply never runs, every HostAction the
    /// device queues is silently left in it, and IRQ pulses, scanout updates and
    /// window input all stop reaching the guest with no failure on any channel.
    /// `_EMPTY` is its partner: the value that legitimately ends that loop.
    ///
    /// All five rather than the two the shims name today, on the same reasoning
    /// as the guest-page family — the error codes are what a new entry point
    /// reaches for next, and pinning only what is read now leaves the rest to
    /// drift until something depends on them.
    #[test]
    fn the_abi_header_agrees_on_the_entry_point_return_codes() {
        for (name, ours) in [
            ("REIMS_VGPU_QEMU_OK", REIMS_VGPU_QEMU_OK),
            ("REIMS_VGPU_QEMU_ERR_ARGS", REIMS_VGPU_QEMU_ERR_ARGS),
            ("REIMS_VGPU_QEMU_ERR_STATE", REIMS_VGPU_QEMU_ERR_STATE),
            ("REIMS_VGPU_QEMU_ERR_PANIC", REIMS_VGPU_QEMU_ERR_PANIC),
            ("REIMS_VGPU_QEMU_EMPTY", REIMS_VGPU_QEMU_EMPTY),
        ] {
            assert_eq!(
                header_define_i32(name),
                ours,
                "{name} has drifted from the staticlib's value"
            );
        }
    }

    /// Both guest-RAM refusal codes exist twice and nothing in the build
    /// compares them. A drift makes the shim say "bad arguments" and the
    /// staticlib hear "this machine has no RAM" — which is the difference
    /// between a caller bug on this build and a board that was never given
    /// memory, and it would send a reader to the wrong half of the tree.
    ///
    /// The consequence is larger than for most codes on this boundary: this
    /// call is the door to every guest-memory import, so a refusal it
    /// misattributes is the device running its copying rails for a whole boot
    /// with the wrong explanation in the log.
    #[test]
    fn the_abi_header_agrees_on_the_guest_ram_codes() {
        for (name, ours) in [
            (
                "REIMS_VGPU_GUEST_RAM_ERR_ARGS",
                REIMS_VGPU_GUEST_RAM_ERR_ARGS,
            ),
            (
                "REIMS_VGPU_GUEST_RAM_ERR_NO_RAM",
                REIMS_VGPU_GUEST_RAM_ERR_NO_RAM,
            ),
        ] {
            assert_eq!(
                header_define_i32(name),
                ours,
                "{name} has drifted from the staticlib's value"
            );
        }
    }

    /// `map_pages` returns one of these on success and the two spellings have
    /// no other comparison.
    ///
    /// A drift is the worst-behaved kind on this boundary because both values
    /// are *valid* returns: the shim saying "I built you a view" and the
    /// staticlib hearing "this is guest RAM, keep it" produces a retained host
    /// pointer into VA the shim deallocates at `unmap_pages`. Nothing reports
    /// it — the symptom is wrong pixels, or a fault a long way from the call.
    #[test]
    fn the_abi_header_agrees_on_the_map_pages_alias_codes() {
        for (name, ours) in [
            ("REIMS_VGPU_MAP_PAGES_TRANSIENT", MAP_PAGES_TRANSIENT),
            ("REIMS_VGPU_MAP_PAGES_STABLE", MAP_PAGES_STABLE),
        ] {
            assert_eq!(
                header_define_i32(name),
                ours,
                "{name} has drifted from the staticlib's value"
            );
        }
    }

    /// The shim writes `ReimsVgpuGuestRamRegion`s straight through the caller's
    /// array, so the C declaration and Rust's `#[repr(C)] GuestRamRegion` are
    /// one struct written down twice.
    ///
    /// A field reordered on one side is not a decode error here — it is a host
    /// address read as a length and imported as a span, which is the one
    /// failure the bound in `runtime::guest_ram` cannot catch, because the
    /// numbers it checks would all be self-consistent. Hence the field names
    /// and their order, not just the size.
    #[test]
    fn the_abi_header_agrees_on_the_guest_ram_region_layout() {
        use crate::runtime::guest_ram::GuestRamRegion;

        const HEADER: &str = include_str!("../../include/reims_vgpu_qemu_abi.h");
        let body = HEADER
            .split_once("typedef struct ReimsVgpuGuestRamRegion {")
            .expect("the header must declare ReimsVgpuGuestRamRegion")
            .1
            .split_once('}')
            .expect("the declaration must be closed")
            .0;
        let fields: Vec<&str> = body
            .split(';')
            .map(str::trim)
            .filter(|f| !f.is_empty())
            .collect();
        assert_eq!(
            fields,
            vec!["uint64_t gpa_base", "uint64_t host_va", "uint64_t len"],
            "the C declaration must stay the one Rust's #[repr(C)] GuestRamRegion lays out"
        );

        assert_eq!(std::mem::size_of::<GuestRamRegion>(), 24);
        assert_eq!(std::mem::align_of::<GuestRamRegion>(), 8);
        assert_eq!(std::mem::offset_of!(GuestRamRegion, gpa_base), 0);
        assert_eq!(std::mem::offset_of!(GuestRamRegion, host_va), 8);
        assert_eq!(std::mem::offset_of!(GuestRamRegion, len), 16);
    }


    /// Every code the header defines maps to its own variant. A code that fell
    /// through to `UnknownCode` would still log a number, but the reader would
    /// be told the shim is newer than the staticlib when in fact the mapping
    /// simply forgot an arm.
    #[test]
    fn every_guest_ram_code_maps_to_its_own_named_check() {
        use crate::observe::Decline as _;
        use crate::runtime::host::GuestRamRegionsError;
        let mut slugs = Vec::new();
        for code in [
            REIMS_VGPU_GUEST_RAM_ERR_ARGS,
            REIMS_VGPU_GUEST_RAM_ERR_NO_RAM,
        ] {
            let mapped = GuestRamRegionsError::from_code(code);
            assert!(
                !matches!(mapped, GuestRamRegionsError::UnknownCode(_)),
                "{code} has no named arm"
            );
            slugs.push(mapped.slug());
        }
        let count = slugs.len();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), count, "two guest-RAM codes share a slug");
        // A code past the end is the one case that *should* be unknown, and it
        // must carry the number rather than swallow it.
        let future = GuestRamRegionsError::from_code(-99);
        assert_eq!(future, GuestRamRegionsError::UnknownCode(-99));
        assert!(future.to_string().contains("code=-99"));
    }

    #[test]
    fn create_reset_destroy() {
        let mut dev = ReimsVgpuQemuDevice {
            abi_version: 0,
            struct_size: 0,
            handle: 0,
        };
        let info = ReimsVgpuQemuCreateInfo {
            abi_version: REIMS_VGPU_QEMU_ABI_VERSION,
            struct_size: std::mem::size_of::<ReimsVgpuQemuCreateInfo>() as u32,
            host_ops: std::ptr::null(),
            guest_page_shift: PAGE_SHIFT_ARM64E, // arm64e — must choose 12 or 14 explicitly
        };
        let rc = unsafe { reims_vgpu_qemu_device_create(&info, &mut dev) };
        assert_eq!(rc, REIMS_VGPU_QEMU_OK);
        assert_ne!(dev.handle, 0);
        assert_eq!(
            unsafe { reims_vgpu_qemu_device_reset(dev.handle) },
            REIMS_VGPU_QEMU_OK
        );
        assert_eq!(
            unsafe { reims_vgpu_qemu_device_destroy(dev.handle) },
            REIMS_VGPU_QEMU_OK
        );
    }

    #[test]
    fn create_rejects_zero_page_shift() {
        let mut dev = ReimsVgpuQemuDevice {
            abi_version: 0,
            struct_size: 0,
            handle: 0,
        };
        let info = ReimsVgpuQemuCreateInfo {
            abi_version: REIMS_VGPU_QEMU_ABI_VERSION,
            struct_size: std::mem::size_of::<ReimsVgpuQemuCreateInfo>() as u32,
            host_ops: std::ptr::null(),
            guest_page_shift: 0,
        };
        let rc = unsafe { reims_vgpu_qemu_device_create(&info, &mut dev) };
        assert_eq!(rc, REIMS_VGPU_QEMU_ERR_ARGS);
    }

    #[test]
    fn backend_name_metal_default() {
        let mut buf = [0i8; 32];
        assert_eq!(
            unsafe { reims_vgpu_qemu_backend_name(buf.as_mut_ptr(), buf.len()) },
            REIMS_VGPU_QEMU_OK
        );
        let s = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_str()
            .unwrap();
        assert!(s == "metal" || s == "vulkan", "got {s}");
    }

    #[test]
    fn mmio_version_roundtrip() {
        let mut dev = ReimsVgpuQemuDevice {
            abi_version: 0,
            struct_size: 0,
            handle: 0,
        };
        let info = ReimsVgpuQemuCreateInfo {
            abi_version: REIMS_VGPU_QEMU_ABI_VERSION,
            struct_size: std::mem::size_of::<ReimsVgpuQemuCreateInfo>() as u32,
            host_ops: std::ptr::null(),
            guest_page_shift: PAGE_SHIFT_X86,
        };
        assert_eq!(
            unsafe { reims_vgpu_qemu_device_create(&info, &mut dev) },
            REIMS_VGPU_QEMU_OK
        );
        // 0x1034 version handshake
        assert_eq!(
            unsafe { reims_vgpu_qemu_gfx_write(dev.handle, 0x1034, 0x3e, 4) },
            REIMS_VGPU_QEMU_OK
        );
        let mut val = 0u64;
        assert_eq!(
            unsafe { reims_vgpu_qemu_gfx_read(dev.handle, 0x1034, 4, &mut val) },
            REIMS_VGPU_QEMU_OK
        );
        assert_eq!(val, 0x3e);
        assert_eq!(
            unsafe { reims_vgpu_qemu_device_destroy(dev.handle) },
            REIMS_VGPU_QEMU_OK
        );
    }

    #[test]
    fn drain_pop_empty() {
        let mut dev = ReimsVgpuQemuDevice {
            abi_version: 0,
            struct_size: 0,
            handle: 0,
        };
        let info = ReimsVgpuQemuCreateInfo {
            abi_version: REIMS_VGPU_QEMU_ABI_VERSION,
            struct_size: std::mem::size_of::<ReimsVgpuQemuCreateInfo>() as u32,
            host_ops: std::ptr::null(),
            guest_page_shift: PAGE_SHIFT_X86, // x86 Tahoe
        };
        assert_eq!(
            unsafe { reims_vgpu_qemu_device_create(&info, &mut dev) },
            REIMS_VGPU_QEMU_OK
        );
        assert_eq!(
            unsafe { reims_vgpu_qemu_device_drain(dev.handle) },
            REIMS_VGPU_QEMU_OK
        );
        let mut action = HostAction::default();
        assert_eq!(
            unsafe { reims_vgpu_qemu_device_pop_action(dev.handle, &mut action) },
            REIMS_VGPU_QEMU_EMPTY
        );
        assert_eq!(
            unsafe { reims_vgpu_qemu_device_destroy(dev.handle) },
            REIMS_VGPU_QEMU_OK
        );
    }
}
