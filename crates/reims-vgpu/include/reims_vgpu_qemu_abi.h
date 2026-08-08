/* reims_vgpu_qemu_abi.h — versioned C ABI for reims-vgpu staticlib.
 *
 * QEMU thin shims include this header:
 *   - hw/display/reims-vgpu-mmio.c (sysbus, arm/vmapple)
 *   - hw/display/reims-vgpu-pci.c  (PCI, x86 Tahoe)
 *
 * C owns only QOM/realize, MemoryRegionOps, IRQ/console/BH, HostOps memory
 * callbacks. All protocol state, FIFO drain, decode, mapper, and GPU work live
 * in the staticlib.
 *
 * Opaque handles only; no Rust types. ABI version must match
 * REIMS_VGPU_QEMU_ABI_VERSION in the staticlib.
 */
#ifndef REIMS_VGPU_QEMU_ABI_H
#define REIMS_VGPU_QEMU_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* v18: ReimsVgpuHostOps.dmabuf_for_pages and every REIMS_VGPU_DMABUF_* removed.
 *      v17's spans replaced the mechanism outright: guest pages reach the host
 *      GPU by importing the RAMBlock mapping QEMU already holds, on Linux,
 *      Windows and macOS alike, rather than through a Linux-only udmabuf fd.
 *      Removed rather than left in place — a half-retired rail with two
 *      spellings is the divergence class AGENTS.md warns about, and a caller
 *      that can still ask for an fd eventually will.
 * v17: ReimsVgpuHostOps.guest_ram_regions — where guest RAM lives in this
 *      process, as stable (gpa_base, host_va, len) spans that hold for the VM's
 *      lifetime. Guest pages reach the host GPU by importing the mapping QEMU
 *      already holds over a RAMBlock, once, after which a resource is an
 *      (offset, len) inside that import rather than a page list somebody has to
 *      export. So the device needs the block, not a view of the pages it
 *      happens to want this frame.
 *
 *      It is deliberately NOT map_pages with a different return type.
 *      map_pages answers "give me a view of these specific pages" and one shim
 *      builds a transient mapping the caller must release; this answers "where
 *      does guest RAM live" and never allocates, never releases and never
 *      changes its answer.
 *
 *      The shim exports the spans and nothing else — not whether this host can
 *      import them, not which primitive the backend would use, not how the
 *      ranges coalesced. A caller holding those three can rebuild the rule, and
 *      per AGENTS.md it eventually will.
 * v16: ReimsVgpuHostOps.dmabuf_for_pages — a run of guest pages as one Linux
 *      dma-buf fd, so the host GPU can read and write those pages directly
 *      instead of through a CPU copy in each direction. The shim answers with
 *      the fd or with a named REIMS_VGPU_DMABUF_ERR_* code; it never exports
 *      whether this host has udmabuf, whether guest RAM is fd-backed, or how
 *      the run list coalesced, because a caller holding those three can rebuild
 *      the rule and eventually will.
 * v15: reims_vgpu_qemu_scanout_may_paint — the console-ownership *verdict* for a
 *      presented mapping, which v14 left in C. v14 moved the three-way kind into
 *      Rust but kept exporting it as an input, and the x86 shim promptly rebuilt
 *      "may this paint" out of the kind and the mapping id while the arm64 shim
 *      built nothing and painted unconditionally. Exporting inputs instead of the
 *      answer is what lets two shims disagree; this exports the answer.
 * v14: reims_vgpu_qemu_console_feed replaces reims_vgpu_qemu_present_boundary_seen
 *      and reims_vgpu_qemu_early_scanout_target. The shims took the old pair
 *      together and branched on it, so the console-ownership rule lived in C
 *      twice over. It is product policy; a thin shim does not hold one. Both old
 *      symbols are removed rather than kept — a shim that can still assemble its
 *      own answer will eventually do so again.
 * v13: ReimsVgpuHostOps.guest_written_pages — the per-page form of v12's
 *      generation. The generation says a surface's pages moved; this says
 *      which, which is what a deferred writeback needs in order to land its
 *      frame without replacing the guest's own stores.
 * v12: ReimsVgpuHostOps.track_guest_writes / untrack_guest_writes /
 *      guest_write_gen — the hypervisor dirty bitmap, the only witness for a
 *      write to a surface's guest pages that no device operation made. Every
 *      host-side copy of those pages is stale the instant the guest CPU stores
 *      into them, and nothing this device counts can see that store.
 * v11: reims_vgpu_qemu_window_run_main — run the host window as QEMU's process-main
 *      UI loop (required by AppKit on Darwin).
 * v10: ReimsVgpuHostOps.map_pages_stable — whether a map_pages view is a stable
 *      guest-RAM alias (x86 PCI: direct RAMBlock pointer, unmap is a no-op) or
 *      a transient mapping (arm sysbus: mach_vm_remap). Gates GPU-direct
 *      writeback's cached host-pointer imports.
 * v9: host-window lifecycle + early FB — reims_vgpu_qemu_window_stop (close + join on
 *     teardown), reims_vgpu_qemu_window_set_early_fb (register BAR1 GOP so the window
 *     shows early boot), and the WindowClosed HostAction (11) the window emits
 *     on a UI close so the shim requests a VM shutdown.
 * v8: reims_vgpu_qemu_window_start (host-owned presentation window; winit +
 *     VkSurfaceKHR). Always linkable; returns REIMS_VGPU_QEMU_ERR_STATE when the
 *     staticlib lacks the host-window feature — C then keeps QEMU's display.
 * v7: ReimsVgpuHostOps.notify_actions (schedule the HostAction-delivery BH from any
 *     thread so IRQ pulses reach the guest mid-drain — ack fast).
 * v6: ReimsVgpuHostOps.is_ram_gpa (reject non-RAM PFNs on mapper / map_pages paths).
 * v5: ReimsVgpuQemuCreateInfo.guest_page_shift (12 = x86 Tahoe, 14 = arm64e). */
#define REIMS_VGPU_QEMU_ABI_VERSION 18u

#define REIMS_VGPU_QEMU_OK 0
#define REIMS_VGPU_QEMU_ERR_ARGS 1
#define REIMS_VGPU_QEMU_ERR_STATE 2
#define REIMS_VGPU_QEMU_ERR_PANIC 3
#define REIMS_VGPU_QEMU_EMPTY 4

/*
 * Why guest_ram_regions refused, when it did. Negative so one return value
 * carries both a span count (>= 0) and a named refusal, and distinct per check
 * because the two say different things about where to look: bad arguments are a
 * caller bug on this build, and an address space with no writable RAM span in
 * it is a machine that was never given any — a `-M` without memory, or a call
 * made before the board finished wiring its RAM up.
 */
#define REIMS_VGPU_GUEST_RAM_ERR_ARGS -1
#define REIMS_VGPU_GUEST_RAM_ERR_NO_RAM -2

/*
 * One span of guest RAM: where it starts in guest physical address space, where
 * this process mapped it, and how long it is. `host_va` is a host pointer
 * carried as a uint64_t so the two declarations have one layout on every target
 * the shims build for.
 *
 * Rust `runtime::guest_ram::GuestRamRegion` is the other declaration, and
 * `qemu::abi::the_abi_header_agrees_on_the_guest_ram_region_layout` is the only
 * thing that compares them.
 */
typedef struct ReimsVgpuGuestRamRegion {
    uint64_t gpa_base;
    uint64_t host_va;
    uint64_t len;
} ReimsVgpuGuestRamRegion;

/*
 * Largest scanout / surface edge the device accepts, in pixels.
 *
 * The basis is the allocation it bounds, not a device capability. Every host
 * pixel buffer here is tightly packed BGRA8, so this edge squared times 4 is
 * the largest single surface the device can be asked to hold: 8192 gives
 * 256 MiB, which is the figure `surface_cache`'s GVA cache cap is reasoned
 * against at its own eviction site. The wire fields are 16-bit and would admit
 * 65535 — a 16 GiB surface out of one corrupt guest word — so a ceiling is
 * required, and this is the product's.
 *
 * Rust `model::MAX_SCANOUT_DIM` owns it: the bound is product policy and every
 * geometry accept/refuse in the device tests against the Rust constant. This
 * define exists only so the two QEMU shims stop each carrying a private copy of
 * the number. A duplicated bound is a bound that can drift, and a drift here is
 * a geometry one pathway accepts and the other silently drops.
 * `model::regs::the_abi_header_agrees_on_the_scanout_bound` fails if they part.
 */
#define REIMS_VGPU_MAX_SCANOUT_DIM 8192u

/*
 * The single EFI pre-boot mode this device advertises, in pixels.
 *
 * Rust `model::regs::EFI_BOOT_{WIDTH,HEIGHT}` owns it. The device reports the
 * pair to the firmware through `GFX_REG_EFI_MODE_SIZE` with a mode count of 1,
 * so the pre-boot console geometry is this device's own declaration and the
 * guest has no other mode to select — and `scanout::paint_efi_console` refuses
 * a console paint whose width is not this one.
 *
 * A shim carrying its own copy is therefore not a convenience: it sizes the
 * `DisplaySurface` the console is painted into, and a drift makes Rust refuse
 * every paint into a surface the shim just created, silently and on whichever
 * pathway drifted. Both shims had a private copy and nothing compared any of
 * the three. `model::regs::the_abi_header_agrees_on_the_efi_boot_mode` does.
 */
#define REIMS_VGPU_EFI_BOOT_WIDTH 1920u
#define REIMS_VGPU_EFI_BOOT_HEIGHT 1080u

/*
 * Gfx MMIO window size, in bytes: the sysbus region on the MMIO shim and BAR0
 * on the PCI one.
 *
 * Rust `model::regs::GFX_MMIO_SIZE` owns it, because it is the bound on the
 * sparse register store every `gfx_read`/`gfx_write` indexes. The two must
 * agree in one direction that matters: a window larger than the Rust bound is
 * guest-addressable space whose accesses Rust drops on the floor.
 *
 * The iosfc window is deliberately *not* here. Rust keeps no per-offset state
 * for that rail — it decodes five named registers and ignores the rest — so a
 * mirrored size would be a second source of truth nothing checks against the
 * one that actually sizes the `MemoryRegion`.
 * `model::regs::the_abi_header_agrees_on_the_gfx_window_size` fails on a drift.
 */
#define REIMS_VGPU_GFX_MMIO_SIZE 0x4000u

/* HostAction kinds — match Rust HostActionKind (runtime::host). */
#define REIMS_VGPU_HOST_ACTION_NONE 0u
#define REIMS_VGPU_HOST_ACTION_IRQ_GFX 1u
#define REIMS_VGPU_HOST_ACTION_IRQ_IOSFC 2u
#define REIMS_VGPU_HOST_ACTION_SCANOUT 3u
#define REIMS_VGPU_HOST_ACTION_CURSOR 4u
#define REIMS_VGPU_HOST_ACTION_TRACE 5u
#define REIMS_VGPU_HOST_ACTION_CURSOR_GLYPH 6u
/*
 * 7 is a retired wire value: it named a pre-host-window QEMU GL/dmabuf scanout
 * action that no longer exists on either side. The numbering below stays where
 * it is so the values remain the ones already compiled into the shim; do not
 * reuse 7 for a new action.
 */
/*
 * Host-owned-window input (see Rust runtime::input / kb host-window). Rust maps
 * the window's platform events into these neutral wire forms; the shim replays
 * them through qemu_input_*, which owns the QEMU-side keycode/button ABI.
 *
 * INPUT_KEY:            a0 = Linux evdev keycode (KEY_*), a1 = 1 down / 0 up.
 *                       -> qemu_input_event_send_key_linux (QEMU owns evdev->qcode).
 * INPUT_POINTER_MOVE:   a0 = x px, a1 = y px, a2 = surface width, a3 = height.
 *                       Absolute (usb-tablet); shim scales via qemu_input_queue_abs.
 * INPUT_POINTER_BUTTON: a0 = neutral Reims VGPU button code (ReimsVgpuButton), a1 = 1/0.
 *                       Wheel notches arrive as a down+up pair; shim maps the
 *                       code to QEMU InputButton.
 */
#define REIMS_VGPU_HOST_ACTION_INPUT_KEY 8u
#define REIMS_VGPU_HOST_ACTION_INPUT_POINTER_MOVE 9u
#define REIMS_VGPU_HOST_ACTION_INPUT_POINTER_BUTTON 10u
/*
 * The host-owned window was closed through its UI. No payload; the shim turns
 * it into qemu_system_shutdown_request — the window is the VM's display, so
 * closing it closes the machine.
 */
#define REIMS_VGPU_HOST_ACTION_WINDOW_CLOSED 11u

/* Neutral pointer/wheel button codes (ReimsVgpuButton) carried in INPUT_POINTER_BUTTON
 * a0. Stable wire contract owned by Rust; the shim maps to QEMU InputButton. */
#define REIMS_VGPU_BUTTON_LEFT 0u
#define REIMS_VGPU_BUTTON_MIDDLE 1u
#define REIMS_VGPU_BUTTON_RIGHT 2u
#define REIMS_VGPU_BUTTON_WHEEL_UP 3u
#define REIMS_VGPU_BUTTON_WHEEL_DOWN 4u
#define REIMS_VGPU_BUTTON_SIDE 5u
#define REIMS_VGPU_BUTTON_EXTRA 6u
#define REIMS_VGPU_BUTTON_WHEEL_LEFT 7u
#define REIMS_VGPU_BUTTON_WHEEL_RIGHT 8u

/*
 * Host services QEMU provides to Rust (apple-gfx raiseInterrupt / readMemory
 * equivalents). QEMU owns the function pointers and ctx for the device life.
 */
typedef struct ReimsVgpuHostOps {
    uint32_t abi_version;
    uint32_t struct_size;
    void *ctx;
    /* 0 = success. */
    int (*read_gpa)(void *ctx, uint64_t gpa, uint8_t *buf, size_t len);
    int (*write_gpa)(void *ctx, uint64_t gpa, const uint8_t *buf, size_t len);
    uint64_t (*mono_ns)(void *ctx);
    /* Safe from any thread; schedules a oneshot main-loop BH. */
    void (*schedule_bh)(void *ctx);
    /* Guest kernel VA read (cpu_memory_rw_debug). 0 = success. */
    int (*read_kva)(void *ctx, uint64_t kva, uint8_t *buf, size_t len);
    /* Guest CPU X-register read (iosfc mapper directed handoff). 0 = success. */
    int (*read_xreg)(void *ctx, uint32_t index, uint64_t *out);
    /*
     * Build one contiguous host-VA view of `count` guest pages (each
     * REIMS_VGPU_GUEST_PAGE_SIZE_ARM64E, page-aligned GPAs into guest RAM) via
     * mach_vm_remap — the ParavirtualizedGraphics mapMemory model: the view
     * aliases guest RAM, so CPU/GPU writes through it *are* guest memory.
     * 0 = success, fills *out_ptr (view length = count * page size).
     */
    int (*map_pages)(void *ctx, const uint64_t *gpas, size_t count,
                     void **out_ptr);
    /*
     * Release a transient view from map_pages (len = count * page size).
     * No-op when map_pages_stable is 1.
     */
    void (*unmap_pages)(void *ctx, void *ptr, size_t len);
    /*
     * 1 if `gpa` translates to guest RAM (MemoryRegion is_ram), 0 otherwise.
     * Mapper page-entry accept and multi-import fail closed on non-RAM PFNs.
     */
    int (*is_ram_gpa)(void *ctx, uint64_t gpa);
    /*
     * Where guest RAM lives in this process. Writes at most `max` spans into
     * `out` and returns the total number this host has, which may be greater
     * than `max` — that is how a caller learns its array was short rather than
     * silently importing part of the guest's memory. Or a negative
     * REIMS_VGPU_GUEST_RAM_ERR_* naming the check that refused.
     *
     * The spans are stable for the VM's lifetime: they are the mappings QEMU
     * made when it created the RAMBlocks, not a view built for this call.
     * Nothing is allocated and there is nothing to release.
     *
     * Adjacent flat-view ranges that are contiguous in *both* address spaces
     * are coalesced, so an ordinary machine answers with one or two spans no
     * matter how many MemoryRegion slices the board built. Only writable RAM is
     * reported: ROM and ROMD ranges are RAM-backed but the guest cannot store
     * into them, and a device that wrote them through the GPU would be writing
     * bytes the guest believes are fixed.
     *
     * Absent (NULL) on any shim that cannot answer, which callers must read as
     * a refusal rather than as a reason to reach for map_pages instead.
     */
    int (*guest_ram_regions)(void *ctx, ReimsVgpuGuestRamRegion *out,
                             size_t max);
    /*
     * Safe from any thread; schedules the HostAction-delivery BH (the queue
     * drained via reims_vgpu_qemu_device_pop_action). Distinct from schedule_bh,
     * which wakes the ordered drain worker: prompt actions (IRQ pulses,
     * cursor moves) must reach the guest while a drain tranche is still
     * running, not after it.
     */
    void (*notify_actions)(void *ctx);
    /*
     * 1 if map_pages returns a *stable* alias of guest RAM: the pointer stays
     * valid for the device lifetime, unmap_pages is a no-op, and the address
     * is never recycled for other memory. 0 if the view is a transient mapping
     * that unmap_pages tears down.
     *
     * This is a claim about a CPU-side *view* and nothing else. It says nothing
     * about the GPU rail: guest RAM reaches the GPU by importing the spans
     * guest_ram_regions names, which are QEMU's own RAMBlock mappings and never
     * a view this call built. Default (absent field / older shim) must be
     * treated as 0.
     */
    int map_pages_stable;
    /*
     * Guest-write tracking. A surface's pages are plain guest RAM: the guest
     * CPU stores into them with no device operation, so no counter the Rust
     * device keeps can witness such a store, and any host-side copy of those
     * pages is silently stale from that instant. The hypervisor's dirty
     * bitmap is the only witness, and these three calls are the door to it.
     *
     * track_guest_writes registers `count` page-aligned GPAs (each of
     * `page_size` bytes) as one tracked set and returns a non-zero opaque
     * token, or 0 when this host cannot observe such writes at all. Callers
     * must read a 0 token as "assume written on every check".
     *
     * untrack_guest_writes releases a token. guest_write_gen returns a
     * monotonic count of host observations that some page of the set was
     * written, or 0 for an unknown token; it is safe from any thread, while
     * track/untrack are not (they mutate QEMU MemoryRegion logging state and
     * must run with the BQL held).
     */
    uint64_t (*track_guest_writes)(void *ctx, const uint64_t *gpas, size_t count,
                                   size_t page_size);
    void (*untrack_guest_writes)(void *ctx, uint64_t token);
    uint64_t (*guest_write_gen)(void *ctx, uint64_t token);
    /*
     * Which pages of the set were written, not just whether any were.
     *
     * Fills `out` with the page-aligned GPAs of `token`'s set whose most recent
     * observed write is newer than `since_gen` — a value the caller previously
     * read from guest_write_gen and recorded next to a host-side copy — and
     * returns how many. Safe from any thread.
     *
     * Returns -1 for every case where the answer is not knowable and the caller
     * must assume the whole set was written: an unknown token, a token whose
     * generation is still unreadable, a `since_gen` of 0, or more written pages
     * than `max` can hold. A truncated list would say "these pages and no
     * others", which turns a conservative caller into a wrong one.
     *
     * A whole-set generation is enough to decide whether to *reuse* a copy. It
     * is not enough to decide what to *write back*: a writeback that discards
     * its whole frame because one page moved loses the Store, and one that
     * writes the whole frame anyway loses the guest's own store. This is the
     * call that lets a writeback do neither.
     */
    int64_t (*guest_written_pages)(void *ctx, uint64_t token, uint64_t since_gen,
                                   uint64_t *out, size_t max);
} ReimsVgpuHostOps;

/*
 * Guest page geometry, arch-qualified. There is deliberately no bare
 * `REIMS_VGPU_GUEST_PAGE_SIZE`: that name reads as "the guest's page size", is
 * 16 KiB whatever the guest is, and the Rust side records the same spelling as
 * the cause of x86 wild writes into stamp slots (`model::regs`, which carries
 * the same prohibition for the same reason). A shim that needs one of these
 * names the arch it is; a shim that cannot must take the page shift from the
 * device.
 *
 * `..._pinned_to_the_rust_page_geometry` fails if any of the four drifts.
 */
#define REIMS_VGPU_GUEST_PAGE_SIZE_ARM64E 16384u
#define REIMS_VGPU_GUEST_PAGE_SHIFT_ARM64E 14u
#define REIMS_VGPU_GUEST_PAGE_SHIFT_X86_64 12u
#define REIMS_VGPU_GUEST_PAGE_SIZE_X86_64 4096u

/* Action for the QEMU BH after drain (IRQ pulse, scanout, cursor). */
typedef struct ReimsVgpuHostAction {
    uint32_t kind;
    uint64_t a0;
    uint64_t a1;
    uint64_t a2;
    uint64_t a3;
} ReimsVgpuHostAction;

typedef struct ReimsVgpuQemuCreateInfo {
    uint32_t abi_version;
    uint32_t struct_size;
    /* Nullable only for pure unit tests; product always passes HostOps. */
    const ReimsVgpuHostOps *host_ops;
    /*
     * Guest page shift for PFN↔GPA (and related wire math).
     * Must be set explicitly: 14 = arm64e (16 KiB), 12 = x86_64 Tahoe (4 KiB).
     * 0 is invalid (no default).
     */
    uint32_t guest_page_shift;
} ReimsVgpuQemuCreateInfo;

typedef struct ReimsVgpuQemuDevice {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t handle;
} ReimsVgpuQemuDevice;

int reims_vgpu_qemu_device_create(const ReimsVgpuQemuCreateInfo *info, ReimsVgpuQemuDevice *out);
int reims_vgpu_qemu_device_reset(uint64_t handle);
int reims_vgpu_qemu_device_destroy(uint64_t handle);

/*
 * Start the host-owned presentation window (winit + VkSurfaceKHR) for this
 * device — replaces QEMU's own display. The drain publishes each finished
 * present frame to it; window input (keys/pointer/wheel) is injected through
 * the neutral Input* prompt-action rail (qemu_input_*). width/height seed the
 * initial size (0 → boot EFI geometry). Idempotent.
 *
 * REIMS_VGPU_QEMU_OK on success; REIMS_VGPU_QEMU_ERR_STATE when the staticlib was built
 * without the host-window feature (caller keeps QEMU's display) or the handle
 * is unknown. Call once at realize, gated on REIMS_VGPU_WINDOW; pair with -display none.
 */
int reims_vgpu_qemu_window_start(uint64_t handle, uint32_t width, uint32_t height);

/*
 * Run the main-thread-owned host window until UI close or backend stop. Call on
 * the same process main thread as reims_vgpu_qemu_window_start.
 */
int reims_vgpu_qemu_window_run_main(uint64_t handle);

/*
 * Stop the host-owned window during VM teardown. Sets the stop flag, waits for
 * the event loop to exit, and ensures the window's Vulkan objects tear down
 * before this returns. Call it before reims_vgpu_qemu_device_destroy and process/driver
 * teardown. Idempotent; REIMS_VGPU_QEMU_OK even with no window.
 */
int reims_vgpu_qemu_window_stop(uint64_t handle);

/*
 * Register the early-boot framebuffer (BAR1 GOP host RAM) so the window shows
 * UEFI/OpenCore/boot.efi output before the product present path latches. ptr
 * must stay valid (>= stride*height bytes) for the device lifetime — pass the
 * BAR1 RAMBlock host pointer. Tight BGRA8 assumed. Call once at realize after
 * reims_vgpu_qemu_window_start.
 */
int reims_vgpu_qemu_window_set_early_fb(uint64_t handle, const uint8_t *ptr,
                                 uint32_t stride, uint32_t width,
                                 uint32_t height);
int reims_vgpu_qemu_backend_name(char *buf, size_t buf_len);
uint32_t reims_vgpu_qemu_abi_version(void);

int reims_vgpu_qemu_gfx_read(uint64_t handle, uint64_t offset, uint32_t size,
                      uint64_t *out_val);
int reims_vgpu_qemu_gfx_write(uint64_t handle, uint64_t offset, uint64_t data,
                       uint32_t size);
int reims_vgpu_qemu_iosfc_read(uint64_t handle, uint64_t offset, uint32_t size,
                        uint64_t *out_val);
int reims_vgpu_qemu_iosfc_write(uint64_t handle, uint64_t offset, uint64_t data,
                         uint32_t size);

/* BH body: drain pending FIFOs (uses HostOps GPA). Then pop actions. */
int reims_vgpu_qemu_device_drain(uint64_t handle);
/* gfx_update tick: re-drive display ONLINE after guest enable() (pending+IRQ). */
int reims_vgpu_qemu_device_poll(uint64_t handle);
/* Returns REIMS_VGPU_QEMU_OK + fills *out, or REIMS_VGPU_QEMU_EMPTY. */
int reims_vgpu_qemu_device_pop_action(uint64_t handle, ReimsVgpuHostAction *out);

/*
 * Which source owns the host console right now.
 *
 * This is the whole console-ownership decision, answered in one call, from
 * protocol state only — never content, sparsity, boot stage or any screenshot
 * heuristic. Rust decides; the shim paints what it is told and holds no rule of
 * its own. Both shims previously rebuilt this three-way from two separate
 * queries and their own branching, which made the rule exist twice in C and a
 * third time in Rust (`host_console_uses_bar1`), free to drift apart.
 */
#define REIMS_VGPU_CONSOLE_FEED_FIRMWARE 0u /* BAR1 UEFI GOP / guest efi_fb */
#define REIMS_VGPU_CONSOLE_FEED_EARLY 1u    /* latched early front; out_* valid */
#define REIMS_VGPU_CONSOLE_FEED_PRODUCT 2u  /* compositor present owns the console */

/*
 * REIMS_VGPU_QEMU_OK fills *out_kind with one of the three above. The four
 * geometry outs are filled only for _EARLY (the logo + progress pill front
 * mapping to re-pull); they are left untouched otherwise, so a caller that only
 * wants the kind may pass NULL for them.
 *
 * _FIRMWARE holds until the guest crosses the first product present boundary
 * (DisplaySwap / frame_flush_seen) — the cutover is NOT the first early logo
 * writeback. Once _PRODUCT is reported it never returns to either of the other
 * two: the boundary is latched monotonically, because a flush-less (ClearOnly)
 * present clears `frame_flush_seen` and re-arming the early paint on that
 * flickers stale pre-boundary content against live presents.
 */
int reims_vgpu_qemu_console_feed(uint64_t handle, uint32_t *out_kind,
                                 uint32_t *out_mapping_id, uint32_t *out_width,
                                 uint32_t *out_height, uint32_t *out_generation);

/*
 * May a present naming mapping_id paint the host console right now?
 * REIMS_VGPU_QEMU_OK fills *out_may with 0 or 1.
 *
 * This is the verdict, not the inputs it is derived from. Call it before
 * painting a presented mapping; do NOT rebuild it from console_feed's out_kind
 * and out_mapping_id. That reconstruction is what the two shims had drifted on:
 * the x86 shim refused a pre-boundary clear-only present naming an unlatched
 * mapping, and the arm64 shim painted the same present without asking.
 */
int reims_vgpu_qemu_scanout_may_paint(uint64_t handle, uint32_t mapping_id,
                                      uint32_t *out_may);

/*
 * Fill a QEMU DisplaySurface (BGRA8, dst_stride bytes/row) from the guest
 * mapping named by mapping_id — or the EFI FB / black clear fallback.
 * generation is HostAction.a3 (0 = always paint). REIMS_VGPU_QEMU_EMPTY = unchanged.
 * C owns the surface; Rust owns the guest-side resolve + format convert.
 */
int reims_vgpu_qemu_scanout_copy(uint64_t handle, uint32_t mapping_id, uint8_t *dst,
                          uint32_t dst_stride, uint32_t width, uint32_t height,
                          uint32_t generation);

/*
 * Pre-boundary early console: guest-programmed EFI FB (MMIO 0x1210 start +
 * 0x1228 stride), contract path for boot.efi / kernel console after it leaves
 * BAR1 linear GOP (serial: "console relocated to 0x…").
 *
 * REIMS_VGPU_QEMU_OK: copy into dst succeeds.
 * REIMS_VGPU_QEMU_EMPTY: efi_fb_start == 0 — C should fall back to BAR1 GOP RAM.
 */
int reims_vgpu_qemu_efi_console_copy(uint64_t handle, uint8_t *dst, uint32_t dst_stride,
                              uint32_t width, uint32_t height);

typedef struct ReimsVgpuCursorGlyphInfo {
    uint32_t width;
    uint32_t height;
    uint32_t hot_x;
    uint32_t hot_y;
    uint32_t pixel_count;
} ReimsVgpuCursorGlyphInfo;

/* Glyph ready in Rust; C builds QEMUCursor. EMPTY when no glyph. */
int reims_vgpu_qemu_cursor_glyph_info(uint64_t handle, ReimsVgpuCursorGlyphInfo *out);
/* out_argb: QEMUCursor 0xAARRGGBB, capacity count pixels. */
int reims_vgpu_qemu_cursor_glyph_copy(uint64_t handle, uint32_t *out_argb,
                               size_t count);

#ifdef __cplusplus
}
#endif

#endif /* REIMS_VGPU_QEMU_ABI_H */
