//! The graphics device: the host side of the reims-graphics device (PCI 1234:5247, our own
//! protocol, `docs/graphics-kext/design.md`), [`RvgGraphicsDeviceConfig`], and the
//! `rvg_graphics_device_*` functions.
//!
//! The host makes the device, maps its region into the guest as BAR 2 (64-bit, prefetchable),
//! sends BAR 0's accesses to it, and BAR 4's (the doorbell pages) where it maps them, hands it the
//! `ioeventfd` it registers on `DOORBELL`, and raises MSI-X vector 0 when it asks. The guest's protocol is the wire header's
//! (`reims_graphics_wire.h`); this API needs none of it.

use std::ffi::c_void;
use std::sync::Arc;

use reims_vgpu::{
    GraphicsCallbacks, GraphicsConfig, GraphicsDevice, GraphicsRaiseFn, GuestArch, PoolAddFn,
    PoolConfig, PoolRemoveFn, RawDoorbell,
};
#[cfg(not(windows))]
use reims_vgpu::{GraphicsRouteDoorbellFn, GraphicsRouteDoorbellPageFn};

use crate::cursor::{
    RvgCursorGlyphAtFn, RvgCursorGlyphFn, RvgCursorMoveAtFn, RvgCursorMoveFn, RvgCursorShowAtFn,
    RvgCursorShowFn, RvgDisplayCoord, cursor_glyph, cursor_glyph_at, cursor_move, cursor_move_at,
    cursor_show, cursor_show_at,
};
use crate::ffi;
use crate::handle;
use crate::memory::{RvgMemoryMap, RvgMemoryRangeFile, memory_map};
use crate::sized::{self, HostStruct};
use crate::status::{RVG_ERR_INVALID_ARGUMENT, RVG_OK, RvgStatus, status};
use crate::table::{RvgDestroyFn, Table, WithTable};
use crate::window::{RvgWindowConfig, window_callbacks};
use crate::{Plat, null_handle};

/// The PCI vendor ID of the graphics device: a placeholder until its own is picked.
pub const RVG_GRAPHICS_PCI_VENDOR_ID: u16 = 0x1234;

/// The PCI device ID of the graphics device: a placeholder, as the vendor ID is.
pub const RVG_GRAPHICS_PCI_DEVICE_ID: u16 = 0x5247;

/// The PCI class of the graphics device: a display controller of class "other".
pub const RVG_GRAPHICS_PCI_CLASS: u16 = 0x0380;

/// The BAR of the graphics device's registers: 32-bit MMIO accesses, which the host sends to
/// rvg_graphics_device_mmio_read and rvg_graphics_device_mmio_write_taking, below the MSI-X
/// table.
pub const RVG_GRAPHICS_PCI_BAR_REGISTERS: u32 = 0;

/// The size of BAR 0.
pub const RVG_GRAPHICS_REGISTERS_SIZE: u32 = 0x1000;

/// Where in BAR 0 the MSI-X table starts, the PBA after it: the host's, not the library's.
pub const RVG_GRAPHICS_MSIX_OFFSET: u32 = 0x800;

/// The graphics device's MSI-X vectors.
pub const RVG_GRAPHICS_MSIX_VECTORS: u32 = 1;

/// The BAR of the graphics device's region (rvg_graphics_device_region): 64-bit and
/// prefetchable, over the region's memory, which the guest maps write-back.
pub const RVG_GRAPHICS_PCI_BAR_REGION: u32 = 2;

/// The register the guest writes a ring's id to as its doorbell: the host's `ioeventfd` for it
/// matches the value RVG_GRAPHICS_DOORBELL_CONTROL, 4 bytes wide.
pub const RVG_GRAPHICS_REG_DOORBELL: u32 = 0x020;

/// The control ring's id, which the guest writes to RVG_GRAPHICS_REG_DOORBELL.
pub const RVG_GRAPHICS_DOORBELL_CONTROL: u32 = 0;

/// The most command queues' doorbells the device routes to the host at once (route_doorbell,
/// RvgGraphicsCallbacks).
pub const RVG_GRAPHICS_DOORBELL_ROUTES: u32 = 128;

/// Where the guest's allocations start in the region, the scanout among them: below it are the
/// control rings.
pub const RVG_GRAPHICS_ALLOCATIONS_OFFSET: u64 = 0x40_0000;

/// The BAR of the graphics device's doorbell pages, where the host maps it
/// (RvgGraphicsDeviceConfig's `doorbell_pages`): 32-bit MMIO, not prefetchable,
/// RVG_GRAPHICS_DOORBELLS_SIZE bytes, whose 32-bit writes the host sends to
/// rvg_graphics_device_doorbell_write and whose reads it answers 0 itself.
pub const RVG_GRAPHICS_PCI_BAR_DOORBELLS: u32 = 4;

/// The size of BAR 4: RVG_GRAPHICS_DOORBELL_PAGES pages of RVG_GRAPHICS_DOORBELL_PAGE_SIZE bytes.
pub const RVG_GRAPHICS_DOORBELLS_SIZE: u32 = 0x40_0000;

/// The bytes of one doorbell page; page `n`'s doorbell is its first byte, at
/// `n * RVG_GRAPHICS_DOORBELL_PAGE_SIZE` in BAR 4.
pub const RVG_GRAPHICS_DOORBELL_PAGE_SIZE: u32 = 0x4000;

/// How many doorbell pages BAR 4 holds.
pub const RVG_GRAPHICS_DOORBELL_PAGES: u32 = 256;

/// The most displays a graphics device has (RvgGraphicsDeviceConfig's `display_count`).
pub const RVG_GRAPHICS_DISPLAYS_MAX: u32 = 4;

const _: () = {
    use reims_vgpu_graphics::wire;
    assert!(RVG_GRAPHICS_PCI_VENDOR_ID == wire::RGFX_PCI_VENDOR_ID);
    assert!(RVG_GRAPHICS_PCI_DEVICE_ID == wire::RGFX_PCI_DEVICE_ID);
    assert!(RVG_GRAPHICS_PCI_CLASS == wire::RGFX_PCI_CLASS);
    assert!(RVG_GRAPHICS_PCI_BAR_REGISTERS == wire::RGFX_PCI_BAR_REGISTERS);
    assert!(RVG_GRAPHICS_REGISTERS_SIZE == wire::RGFX_REGISTERS_SIZE);
    assert!(RVG_GRAPHICS_MSIX_OFFSET == wire::RGFX_MSIX_OFFSET);
    assert!(RVG_GRAPHICS_MSIX_VECTORS == wire::RGFX_MSIX_VECTORS);
    assert!(RVG_GRAPHICS_PCI_BAR_REGION == wire::RGFX_PCI_BAR_REGION);
    assert!(RVG_GRAPHICS_REG_DOORBELL == wire::RGFX_REG_DOORBELL);
    assert!(RVG_GRAPHICS_DOORBELL_CONTROL == wire::RGFX_CONTROL_RING_ID);
    assert!(RVG_GRAPHICS_DOORBELL_ROUTES == reims_vgpu_graphics::device::MAX_DOORBELL_ROUTES);
    assert!(RVG_GRAPHICS_ALLOCATIONS_OFFSET == wire::RGFX_ALLOCATIONS_OFFSET);
    assert!(RVG_GRAPHICS_PCI_BAR_DOORBELLS == wire::RGFX_PCI_BAR_DOORBELLS);
    assert!(RVG_GRAPHICS_DOORBELLS_SIZE == wire::RGFX_DOORBELLS_SIZE);
    assert!(RVG_GRAPHICS_DOORBELL_PAGE_SIZE == wire::RGFX_DOORBELL_PAGE_SIZE);
    assert!(RVG_GRAPHICS_DOORBELL_PAGES == wire::RGFX_DOORBELL_PAGES);
    assert!(RVG_GRAPHICS_DISPLAYS_MAX == wire::RGFX_DISPLAYS_MAX);
};

/// `raise_interrupt(ctx, vector)`: raise the device's MSI-X vector `vector` (0).
///
/// Thread: the one that posts the event it raises for: the device's control thread
/// (`reims-vgpu.graphics-control`), one of ring 0's threads (`reims-vgpu.graphics-ring0-<i>`),
/// which carry out a guest process's control messages, its presents and a block's pages and
/// downloads, each on its lane, a display's refresh thread
/// (`reims-vgpu.graphics-refresh`, and `reims-vgpu.graphics-refresh-<i>` for display i from 1),
/// a command queue's service thread
/// (`reims-vgpu.graphics-queue-<id>`) or one of its lane's threads, which walk and commit its
/// command buffers (`reims-vgpu.graphics-lane-<id>`), a pipeline compile's thread
/// (`reims-vgpu.graphics-compile-<n>`), a thread that makes IOSurfaces' and pages blocks' views
/// (`reims-vgpu.graphics-surfaces-<i>`), a thread that makes indirect command buffers
/// (`reims-vgpu.icb-create-<i>`), a thread that lets go of a closed process's objects and answers
/// its close (`reims-vgpu.graphics-closes-<i>`), a thread that moves a staged copy of a surface's
/// device cache into its pages and answers what waited for it
/// (`reims-vgpu.graphics-into-pages-<i>`), a video session's picture landing thread, which posts
/// a decode's completion as the decode is done (`reims-vgpu.graphics-video-land`), or the
/// platform's thread that runs a command buffer's scheduled or completed handler; or the host's
/// own, inside rvg_graphics_device_mmio_write_taking.
/// Never inside another call of the host's into the library. The library may hold locks of its
/// own around it (docs/api.md section 5 names them), but none that a function it may call waits
/// for, directly or through a thread of the library's, so it may call them at once. It may wait
/// for a lock the host holds while it calls the device (QEMU's BQL), but not for one the host
/// holds while it calls rvg_graphics_device_mmio_write_taking. May call: rvg_status_message and
/// the rvg_graphics_device_* functions, except the release of its own device. Without it the
/// guest gets no interrupt.
pub type RvgGraphicsRaiseFn = Option<unsafe extern "C-unwind" fn(ctx: *mut c_void, vector: u32)>;

/// `route_doorbell(ctx, route, ring_id, bell)`: with `bell` 0 or more, route a command queue's
/// plain doorbell into the file `bell`: from then on the host takes a write of exactly `ring_id`
/// to RVG_GRAPHICS_REG_DOORBELL without calling rvg_graphics_device_mmio_write_taking, and makes
/// `bell` readable (QEMU: memory_region_add_eventfd on the register, matching `ring_id`, which
/// KVM's ioeventfd takes in the kernel). `bell` is an eventfd the queue's thread sleeps on, which
/// the host duplicates during the call; `route` names the route, a number the device gives no
/// other. A route of a `ring_id` another route holds replaces that one. Return true where the
/// host takes the route, during the call or after it: until then a write comes through
/// rvg_graphics_device_mmio_write_taking as before. Return false to refuse it: the queue's
/// writes all come through there. With `bell` -1, drop route `route` (of `ring_id`), whose queue
/// ended, before returning: no write reaches the host's copy once the call returns, as the ring
/// may be a later queue's; the answer is not read. The device offers the reims-graphics wire's
/// DOORBELL_ASKS only where the host gives this function, on Linux (a host on macOS or Windows
/// has no kernel ioeventfd, and its guest's doorbells stay as they were; a device on Windows
/// calls neither route function), and routes a queue only where
/// the guest's driver took it, whose write for a command buffer
/// the device may answer at its doorbell holds bit 0 beside the ring's id and so matches no
/// route; at most RVG_GRAPHICS_DOORBELL_ROUTES at once; and it drops none at its end: the host
/// drops those it holds as it releases the device.
///
/// Thread: one of ring 0's threads (`reims-vgpu.graphics-ring0-<i>`) or the control thread
/// (`reims-vgpu.graphics-control`) as a queue is made, destroyed or its process closed, where it
/// may wait for a lock the host holds while it calls the device (QEMU's BQL): no function of this
/// header the host calls under that lock waits for a thread inside it, but the release of the
/// process's last device, at most 5 s; with `bell` -1 also the host's own inside rvg_graphics_device_mmio_write_taking
/// (a write of 0 to STATUS) or rvg_graphics_device_reset, where it must not wait for a lock the
/// host holds around that call. The library holds no lock around it that a function of this
/// header waits for, and it may call nothing of the library's. Without it every doorbell write
/// comes through rvg_graphics_device_mmio_write_taking.
pub type RvgGraphicsRouteDoorbellFn = Option<
    unsafe extern "C-unwind" fn(ctx: *mut c_void, route: u64, ring_id: u32, bell: i32) -> bool,
>;

/// `route_doorbell_page(ctx, route, page, ring_id, bell)`: as route_doorbell, for a write of
/// exactly `ring_id` at doorbell page `page`'s doorbell, `page * RVG_GRAPHICS_DOORBELL_PAGE_SIZE`
/// in BAR 4, in place of RVG_GRAPHICS_REG_DOORBELL (QEMU: memory_region_add_eventfd on BAR 4 at
/// that offset, matching `ring_id`): the route of a command queue whose guest process has that
/// page, which its process writes. Until the host takes the route, and where it refuses it, the
/// writes come through rvg_graphics_device_doorbell_write. A route of a `ring_id` at `page`
/// another route holds replaces that one. With `bell` -1, drop route `route` (of `ring_id` at
/// `page`), as route_doorbell does. The device asks it only of a host that maps BAR 4
/// (RvgGraphicsDeviceConfig's `doorbell_pages`), and routes a queue on its page only where the
/// host gives this function; it counts these routes among RVG_GRAPHICS_DOORBELL_ROUTES.
///
/// Thread: as route_doorbell's, with `bell` -1 also inside rvg_graphics_device_mmio_write_taking
/// (a write of 0 to STATUS) or rvg_graphics_device_reset; never inside
/// rvg_graphics_device_doorbell_write. The library holds no lock around it that a function of
/// this header waits for, and it may call nothing of the library's. Without it the writes of a
/// queue whose process has a page all come through rvg_graphics_device_doorbell_write.
pub type RvgGraphicsRouteDoorbellPageFn = Option<
    unsafe extern "C-unwind" fn(
        ctx: *mut c_void,
        route: u64,
        page: u32,
        ring_id: u32,
        bell: i32,
    ) -> bool,
>;

/// `pool_add(ctx, physical_address, host_address, length)`: map `length` bytes of host memory
/// from `host_address` into the guest at `physical_address`, a piece of the guest's Shared pool
/// (RvgGraphicsDeviceConfig's `pool_physical_address` and `pool_physical_length`, which hold the
/// piece), and call rvg_graphics_device_pool_added with `physical_address` once the guest's
/// address space has it (QEMU: a RAM region over the memory, added on its main loop, unmergeable
/// with its neighbours). The memory stays mapped, and the device asks no other piece over its
/// addresses, until the host has said the piece is removed (rvg_graphics_device_pool_removed).
/// `physical_address`, `host_address` and `length` are multiples of 2 MiB, 4 KiB and 2 MiB.
///
/// Thread: the device's pool thread (`reims-vgpu.graphics-pool`), never inside a call of the
/// host's, with no lock of the library's held. It must not wait for a lock the host holds while
/// it calls the device (QEMU's BQL): QEMU queues the add for its main loop. One piece at a time,
/// in the order the device places them, each before its pool_remove. May call: nothing of the
/// library's. Needed with a pool.
pub type RvgGraphicsPoolAddFn = Option<
    unsafe extern "C-unwind" fn(
        ctx: *mut c_void,
        physical_address: u64,
        host_address: *mut c_void,
        length: u64,
    ),
>;

/// `pool_remove(ctx, physical_address, length)`: unmap the piece of the pool at
/// `physical_address` that pool_add mapped, and call rvg_graphics_device_pool_removed with
/// `physical_address` once it is out of the guest's address space and no thread of the host's
/// touches its memory (QEMU: once its region's last reference goes, past its RCU grace period).
/// The device frees the memory only after that. A remove of a piece whose add the host has not
/// carried out yet follows that add: the host carries out the add, then the remove.
///
/// Thread: the device's pool thread (`reims-vgpu.graphics-pool`), before it gives the piece's
/// block back, never inside a call of the host's, with no lock of the library's held. It must not
/// wait for a lock the host holds while it calls the device (QEMU queues the remove for its main
/// loop). Once a piece, after its pool_add. Not called once rvg_graphics_device_release has
/// begun: the host removes every piece it added before it releases the device. May call: nothing
/// of the library's. Needed with a pool.
pub type RvgGraphicsPoolRemoveFn =
    Option<unsafe extern "C-unwind" fn(ctx: *mut c_void, physical_address: u64, length: u64)>;

/// The graphics device's callbacks.
///
/// The cursor's three tell the host the guest's cursor on the device's display, for a host that
/// draws it (QEMU's console, for its SPICE, VNC, GTK and D-Bus displays); the device's window
/// shows it either way. Thread, and what they may call: as RvgCursorGlyphFn's, on the display
/// thread, one at a time with every display callback of the process, in the order the device
/// took the guest's cursor and display power commands and resets. The guest sees nothing of
/// whether the host gives them. They are display 0's; on a device of several displays
/// (RvgGraphicsDeviceConfig's `display_count`) the `_at` three are those of displays 1 and up,
/// each called as display 0's is, naming the display, in order with every display callback.
#[repr(C)]
#[derive(Clone, Copy)]
#[expect(
    missing_docs,
    reason = "the fields' names say what they hold, as the header shows them"
)]
pub struct RvgGraphicsCallbacks {
    /// sizeof(RvgGraphicsCallbacks) as the host was compiled.
    pub struct_size: u32,
    pub ctx: *mut c_void,
    /// Runs once when the device's threads have ended, after their last callback: possibly on
    /// one of them, after rvg_graphics_device_release has returned.
    pub destroy: RvgDestroyFn,
    pub raise_interrupt: RvgGraphicsRaiseFn,
    /// The guest set its cursor's image: `rgba` holds its pixels as straight-alpha RGBA (the
    /// guest gives them premultiplied, BGRA), each side 1 to 256 pixels, the hot spot inside.
    /// Followed by `cursor_show` where whether the cursor shows changed.
    pub cursor_glyph: RvgCursorGlyphFn,
    /// Whether the cursor shows, where that changed: 1 once the guest has an image while it
    /// shows its cursor and its display is awake; 0 once it hides it, removes its image, which a
    /// reset removes, or its display sleeps (the panel off). It is hidden until the first call.
    pub cursor_show: RvgCursorShowFn,
    /// The guest moved its cursor, or a reset put it at (0, 0):
    /// rvg_graphics_device_cursor_position reads where its hot spot is.
    pub cursor_move: RvgCursorMoveFn,
    /// Routes a command queue's plain doorbell into its bell, or drops the route. A host built
    /// before it, whose `struct_size` ends before it, gives none.
    pub route_doorbell: RvgGraphicsRouteDoorbellFn,
    /// Routes a command queue's plain doorbell on its process's doorbell page into its bell, or
    /// drops the route. A host built before it, whose `struct_size` ends before it, gives none.
    pub route_doorbell_page: RvgGraphicsRouteDoorbellPageFn,
    /// cursor_glyph of display 1 and up. A host built before it, whose `struct_size` ends before
    /// it, gives none.
    pub cursor_glyph_at: RvgCursorGlyphAtFn,
    /// cursor_show of display 1 and up, hidden until its first call. A host built before it gives
    /// none.
    pub cursor_show_at: RvgCursorShowAtFn,
    /// cursor_move of display 1 and up: rvg_graphics_device_cursor_position_at reads where. A
    /// host built before it gives none.
    pub cursor_move_at: RvgCursorMoveAtFn,
    /// Maps a piece of the guest's Shared pool. A host built before it, whose `struct_size` ends
    /// before it, gives none, and has no pool.
    pub pool_add: RvgGraphicsPoolAddFn,
    /// Unmaps a piece of the guest's Shared pool. A host built before it gives none.
    pub pool_remove: RvgGraphicsPoolRemoveFn,
}

// SAFETY: repr(C), a u32 first; a pointer and optional function pointers.
unsafe impl HostStruct for RvgGraphicsCallbacks {
    /// The first version ended before `route_doorbell`.
    const V1_SIZE: usize = std::mem::offset_of!(RvgGraphicsCallbacks, route_doorbell);
}

/// The architecture of the macOS guest a graphics device serves: one of RVG_GUEST_ARCH_*.
pub type RvgGuestArch = u32;

// The constants below are RvgGuestArch values, typed `u32` so that cbindgen writes them with the
// header's other constants, ahead of its types.

/// An x86-64 guest: the default, and what a host built before `guest_arch` serves.
pub const RVG_GUEST_ARCH_X86_64: u32 = 0;

/// An arm64 guest.
pub const RVG_GUEST_ARCH_ARM64: u32 = 1;

/// The graphics device's config.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RvgGraphicsDeviceConfig {
    /// sizeof(RvgGraphicsDeviceConfig) as the host was compiled.
    pub struct_size: u32,
    /// NULL is a table with every callback absent.
    pub callbacks: *const RvgGraphicsCallbacks,
    /// NULL is no window: the display offers 1920 x 1080 at 60 Hz and shows nothing. Otherwise
    /// the library opens a window for the display, with its input as the window's callbacks,
    /// sized for its mode (1920 x 1080 where it has none). The window keeps its size: the
    /// display offers the guest one mode, the window's size in pixels at the refresh rate of the
    /// host screen it opens on, and tells the guest when the person resizes it. The guest's
    /// scanout is blitted out of the region into the window's drawables on the GPU at each of the
    /// display's refreshes, at the screen's rate, while the display shows no presented surface.
    pub window: *const RvgWindowConfig,
    /// The guest's RAM, or NULL. With it, where the host's page is the guest's 4 KiB, or 16 KiB
    /// for a guest that lays its pages out in 16 KiB (`RGFX_FEATURE_PAGES_16K`), the GPU uses an
    /// IOSurface's own pages in place, through views of them; without it each surface's GPU view
    /// is a copy of its pages in the region. On a host GPU that shares the CPU's memory the device
    /// answers `recommendedMaxWorkingSetSize` as three quarters of the RAM the map holds, rounded
    /// down to 16 KiB, or the host GPU's answer where that is less; with no map, and on a GPU of
    /// its own memory, the host GPU's. The map's pages stay the guest's RAM while the device lives:
    /// a host that discards guest RAM (a balloon, virtio-mem, hot-unplug) passes NULL. Its ranges
    /// may not overlap.
    pub memory_map: *const RvgMemoryMap,
    /// With `memory_map`: the files behind its ranges, NULL for none, one entry per range, each
    /// duplicated during the create. Refused (RVG_ERR_INVALID_ARGUMENT), in this order: files
    /// without a map; an entry whose `fd` is below -1 or whose `flags` are not 0; for an `fd` of
    /// 0 or more, a file `fstat` cannot read, one holding fewer than its `offset` and the range's
    /// length in bytes, or an `offset` that is not a multiple of the host's page size; a file that
    /// cannot be duplicated. A refusal logs a line saying which. On Windows each entry names a
    /// section by its `handle`, NULL for none, and its `fd` checks are a `handle`'s: a handle of
    /// which no view the library may write maps its `offset` and the range's length in bytes (one
    /// that names no section, a section it may not map so, or one holding fewer bytes) is
    /// refused where a file `fstat` cannot read or one holding fewer bytes is.
    pub memory_files: *const RvgMemoryRangeFile,
    /// The guest's architecture, RVG_GUEST_ARCH_*, which the device's answers to the guest's Metal
    /// follow where an Apple silicon Mac's GPU and an Intel or AMD Mac's answer apart: an arm64
    /// guest's device answers `isFramebufferReadSupported` as the host's GPU can, an x86-64
    /// guest's no; and the GPU's results that Metal leaves to the Mac's GPU (what a new Private
    /// texture reads before anything writes it, the encoding a blit through a compressed texture
    /// writes, a half NaN clear's sign, how far a Clear load reaches past the render area) are an
    /// Apple silicon Mac's for an arm64 guest and an Intel or AMD Mac's for an x86-64 guest,
    /// whatever the host's GPU. Those results are the process's: the guest of its first graphics
    /// device decides them for every device it makes. Refused (RVG_ERR_INVALID_ARGUMENT): any
    /// other value. A host built before this field, whose `struct_size` ends before it, serves an
    /// x86-64 guest.
    pub guest_arch: RvgGuestArch,
    /// 1 where the host maps BAR 4, the doorbell pages (RVG_GRAPHICS_PCI_BAR_DOORBELLS), and
    /// sends each 32-bit write of it to rvg_graphics_device_doorbell_write, answering its reads
    /// 0: the device then offers the wire's DOORBELL_PAGES, on which the guest's kext maps each
    /// process's own page into it, and the process rings its queues there with no call into the
    /// kext. 0 for none, as a host built before this field, whose `struct_size` ends before it,
    /// has. Refused (RVG_ERR_INVALID_ARGUMENT): any other value.
    pub doorbell_pages: u32,
    /// How many displays the device has: 0 and 1 are one display, as a host built before this
    /// field, whose `struct_size` ends before it, has; 2 to RVG_GRAPHICS_DISPLAYS_MAX make a device
    /// of that many, each with its own scanout, mode, cursor, presents and colour, which the
    /// guest's driver sees where it takes the reims-graphics wire's MULTI_DISPLAY (one that does
    /// not sees display 0 alone). Each display's scanout keeps to an even share of the scanout's
    /// room, and is offered the largest mode of its window's ratio that fits there. Display 0 is
    /// the one every other function and callback names; rvg_graphics_device_read_display_at,
    /// rvg_graphics_device_cursor_position_at and the `_at` cursor callbacks name each display.
    /// Refused (RVG_ERR_INVALID_ARGUMENT): any other value.
    pub display_count: u32,
    /// The window configs of displays 1 to `display_count - 1`, an array of that many pointers,
    /// each to a window config read as `window` is, its window opened as `window`'s, on the window
    /// crate's port of its display (titled "Reims vGPU (port N)" and placed on host screen N, as
    /// REIMS_VGPU_WINDOW_SCREENS picks), given exactly where `window` is: with `window` and more
    /// than one display, `display_count - 1` non-NULL pointers; otherwise NULL. Refused
    /// (RVG_ERR_INVALID_ARGUMENT): one given where `window` is NULL or the device has one
    /// display, missing where `window` is given and it has several, or a NULL entry. A host built
    /// before this field gives none.
    pub display_windows: *const *const RvgWindowConfig,
    /// The first guest physical address of the range the host can give the guest's Shared pool,
    /// with no RAM or device under it, which the device fills with blocks of host memory as the
    /// pool grows (pool_add). The device takes from its start as many bytes as its GPU's memory
    /// heaps hold, every heap summed, rounded up to 2 MiB, at most `pool_physical_length`, and none
    /// where the GPU answers no heap; rvg_graphics_device_pool_range answers what it took, which
    /// the host keeps free while the device lives. 0, with `pool_physical_length` 0, for none, as a
    /// host built before this field, whose `struct_size` ends before it, has. Refused
    /// (RVG_ERR_INVALID_ARGUMENT): a range not in whole multiples of 2 MiB, one past the address
    /// space, one a range of `memory_map` overlaps, and one given without both pool_add and
    /// pool_remove.
    pub pool_physical_address: u64,
    /// The range's length in bytes, the most the pool may take: 0 for none.
    pub pool_physical_length: u64,
}

// SAFETY: repr(C), a u32 first; pointers and u32s.
unsafe impl HostStruct for RvgGraphicsDeviceConfig {
    /// The first version ended before `guest_arch`.
    const V1_SIZE: usize = std::mem::offset_of!(RvgGraphicsDeviceConfig, guest_arch);
}

/// The graphics device.
pub struct RvgGraphicsDevice {
    /// The windows' tables, display 0's first, for a device with windows: held here so that a
    /// table with no function still has its destroy at the release. They go before the device,
    /// whose windows hold the tables' other references until they close.
    window_tables: Vec<Arc<Table>>,
    device: WithTable<GraphicsDevice<Plat>>,
}

impl Drop for RvgGraphicsDevice {
    /// The windows' tables first, display 0's first, then the device (`crate::display`'s order).
    fn drop(&mut self) {
        drop(std::mem::take(&mut self.window_tables));
    }
}

/// The graphics device behind a handle, or the process's end for NULL.
///
/// # Safety
/// `device` is NULL or a live graphics device handle.
unsafe fn live<'a>(device: *const RvgGraphicsDevice, call: &'static str) -> &'a RvgGraphicsDevice {
    // SAFETY: the caller's contract.
    unsafe { handle::get(device) }.unwrap_or_else(|| null_handle(call))
}

/// A new graphics device, its region mapped and zeroed, its threads started, with a window for
/// each display where the config gives them. Its config and window configs are read, with their
/// tables, before any table is the library's: a struct_size refusal runs no destroy, and a later
/// refusal runs each, the windows' first, display 0's first. Refusals: RVG_ERR_INVALID_ARGUMENT,
/// RVG_ERR_UNSUPPORTED, RVG_ERR_HOST_RESOURCES (the host refused the region's memory, a pipe or
/// a thread, or has no GPU device the library can use: none, or one its driver refused). Calls
/// back: never, but a refusal's destroys.
///
/// # Safety
/// `config` is NULL or a live config; `device` is NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_graphics_device_create(
    config: *const RvgGraphicsDeviceConfig,
    device: *mut *mut RvgGraphicsDevice,
) -> RvgStatus {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { create(config, device) }
    })
}

/// rvg_graphics_device_create's body.
///
/// # Safety
/// As rvg_graphics_device_create's.
unsafe fn create(
    config: *const RvgGraphicsDeviceConfig,
    device: *mut *mut RvgGraphicsDevice,
) -> RvgStatus {
    // SAFETY: the caller's contract.
    let c = match unsafe { sized::read(config) } {
        Ok(Some(c)) => c,
        Ok(None) => return RVG_ERR_INVALID_ARGUMENT,
        Err(s) => return s,
    };
    // SAFETY: the config's contract: NULL or a live table.
    let callbacks = match unsafe { sized::read_or_zero(c.callbacks) } {
        Ok(t) => t,
        Err(s) => return s,
    };
    // SAFETY: the config's contract: NULL or a live window config.
    let window = match unsafe { sized::read(c.window) } {
        Ok(w) => w,
        Err(s) => return s,
    };
    let window = match window {
        None => None,
        // SAFETY: the window config's contract: NULL or a live table.
        Some(w) => match unsafe { sized::read_or_zero(w.callbacks) } {
            Ok(t) => Some((w, t)),
            Err(s) => return s,
        },
    };
    // The windows of displays 1 and up, read as display 0's where it has one and the count is
    // one the device takes; a NULL entry and a count it does not take are refused below.
    let count = c.display_count.max(1);
    let more = match window {
        Some(_) if count <= RVG_GRAPHICS_DISPLAYS_MAX && !c.display_windows.is_null() => count - 1,
        _ => 0,
    };
    let mut display_windows = Vec::new();
    for i in 0..more as usize {
        // SAFETY: the config's contract: `display_count - 1` readable pointers.
        let entry = unsafe { c.display_windows.add(i).read() };
        // SAFETY: the entry's contract: NULL or a live window config.
        let w = match unsafe { sized::read(entry) } {
            Ok(w) => w,
            Err(s) => return s,
        };
        let Some(w) = w else {
            display_windows.push(None);
            continue;
        };
        // SAFETY: the window config's contract: NULL or a live table.
        match unsafe { sized::read_or_zero(w.callbacks) } {
            Ok(t) => display_windows.push(Some((w, t))),
            Err(s) => return s,
        }
    }
    // From here the tables are the library's: a refusal ends them before it returns, the
    // windows' first, display 0's first, as a release does.
    let table = Table::new(callbacks.ctx, callbacks.destroy);
    let window_tables: Vec<Arc<Table>> = (window.iter())
        .chain(display_windows.iter().flatten())
        .map(|(_, t)| Table::new(t.ctx, t.destroy))
        .collect();
    let refuse = |s: RvgStatus| {
        for t in &window_tables {
            t.end();
        }
        table.end();
        s
    };
    if device.is_null() {
        return refuse(RVG_ERR_INVALID_ARGUMENT);
    }
    // Windows given exactly where display 0's is: one for each display from 1, none without it.
    let windows_given = !c.display_windows.is_null();
    let windows_wanted = window.is_some() && count > 1;
    if count > RVG_GRAPHICS_DISPLAYS_MAX
        || windows_given != windows_wanted
        || display_windows.iter().any(Option::is_none)
    {
        return refuse(RVG_ERR_INVALID_ARGUMENT);
    }
    let guest_arch = match c.guest_arch {
        RVG_GUEST_ARCH_X86_64 => GuestArch::X86_64,
        RVG_GUEST_ARCH_ARM64 => GuestArch::Arm64,
        _ => return refuse(RVG_ERR_INVALID_ARGUMENT),
    };
    let doorbell_pages = match c.doorbell_pages {
        0 => false,
        1 => true,
        _ => return refuse(RVG_ERR_INVALID_ARGUMENT),
    };
    // SAFETY: the config's contract: NULL or a live map, and NULL or its files.
    let memory_map = match unsafe { memory_map(c.memory_map, c.memory_files) } {
        Ok(m) => m,
        Err(s) => return refuse(s),
    };
    let pool = match (c.pool_physical_address, c.pool_physical_length) {
        (0, 0) => None,
        (physical_address, physical_length) => Some(PoolConfig {
            physical_address,
            physical_length,
        }),
    };
    let raise_interrupt = callbacks.raise_interrupt.map(|f| {
        let t = table.clone();
        let raise: GraphicsRaiseFn = Arc::new(move |vector| {
            // SAFETY: the host's function over its ctx.
            ffi::host_call("raise_interrupt", || unsafe { f(t.ctx(), vector) });
        });
        raise
    });
    // A Windows device makes no bell to route a queue's doorbell into (the host has no kernel
    // ioeventfd there; `reims-vgpu-graphics`'s `wake::bell`), so it calls neither route function,
    // and none is handed to it.
    #[cfg(windows)]
    let (route_doorbell, route_doorbell_page) = (None, None);
    #[cfg(not(windows))]
    let route_doorbell = callbacks.route_doorbell.map(|f| {
        let t = table.clone();
        let route: GraphicsRouteDoorbellFn = Arc::new(move |route, ring_id, bell| {
            // SAFETY: the host's function over its ctx.
            ffi::host_call("route_doorbell", || unsafe {
                f(t.ctx(), route, ring_id, bell.unwrap_or(-1))
            })
        });
        route
    });
    #[cfg(not(windows))]
    let route_doorbell_page = callbacks.route_doorbell_page.map(|f| {
        let t = table.clone();
        let route: GraphicsRouteDoorbellPageFn = Arc::new(move |route, page, ring_id, bell| {
            // SAFETY: the host's function over its ctx.
            ffi::host_call("route_doorbell_page", || unsafe {
                f(t.ctx(), route, page, ring_id, bell.unwrap_or(-1))
            })
        });
        route
    });
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks {
            raise_interrupt,
            cursor_glyph: cursor_glyph(&table, callbacks.cursor_glyph),
            cursor_show: cursor_show(&table, callbacks.cursor_show),
            cursor_move: cursor_move(&table, callbacks.cursor_move),
            cursor_glyph_at: cursor_glyph_at(&table, callbacks.cursor_glyph_at),
            cursor_show_at: cursor_show_at(&table, callbacks.cursor_show_at),
            cursor_move_at: cursor_move_at(&table, callbacks.cursor_move_at),
            route_doorbell,
            route_doorbell_page,
            pool_add: pool_add(&table, callbacks.pool_add),
            pool_remove: pool_remove(&table, callbacks.pool_remove),
        },
        window: (window.iter().zip(&window_tables))
            .map(|((w, wc), t)| w.to_config(window_callbacks(t, wc)))
            .next(),
        display_count: c.display_count,
        display_windows: (display_windows.iter().flatten())
            .zip(window_tables.iter().skip(1))
            .map(|((w, wc), t)| w.to_config(window_callbacks(t, wc)))
            .collect(),
        memory_map,
        guest_arch,
        doorbell_pages,
        pool,
    };
    match GraphicsDevice::with_platform(<Plat as reims_vgpu::ApiPlatform>::process(), config) {
        Ok(d) => {
            let h = handle::new(RvgGraphicsDevice {
                window_tables,
                device: WithTable::new(d, table),
            });
            // SAFETY: checked non-NULL; the caller's contract.
            unsafe { device.write(h) };
            RVG_OK
        }
        Err(e) => refuse(status(e)),
    }
}

/// The host's pool_add, called with the table's ctx.
fn pool_add(table: &Arc<Table>, f: RvgGraphicsPoolAddFn) -> Option<PoolAddFn> {
    let f = f?;
    let t = table.clone();
    let add: PoolAddFn = Arc::new(move |physical_address, host, length| {
        // SAFETY: the host's function over its ctx.
        ffi::host_call("pool_add", || unsafe {
            f(t.ctx(), physical_address, host.cast(), length)
        });
    });
    Some(add)
}

/// The host's pool_remove, called with the table's ctx.
fn pool_remove(table: &Arc<Table>, f: RvgGraphicsPoolRemoveFn) -> Option<PoolRemoveFn> {
    let f = f?;
    let t = table.clone();
    let remove: PoolRemoveFn = Arc::new(move |physical_address, length| {
        // SAFETY: the host's function over its ctx.
        ffi::host_call("pool_remove", || unsafe {
            f(t.ctx(), physical_address, length)
        });
    });
    Some(remove)
}

/// The host mapped the pool's piece at `physical_address` that pool_add asked for: the guest's
/// address space has it from now on. RVG_ERR_INVALID_ARGUMENT for a device without a pool, an
/// address the device asked no add at, or one the host said it added already. On any thread, the
/// host's lock (QEMU's BQL) held or not: waits for nothing. Calls back: never.
///
/// # Safety
/// `device` is a live graphics device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_graphics_device_pool_added(
    device: *mut RvgGraphicsDevice,
    physical_address: u64,
) -> RvgStatus {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe {
            pool_call(device, "rvg_graphics_device_pool_added", |d| {
                d.pool_added(physical_address)
            })
        }
    })
}

/// The host took the pool's piece at `physical_address` out of the guest's address space, and no
/// thread of its own touches its memory: asked to by pool_remove, or of its own, as it does with
/// every piece it added before it releases the device. The device may free the piece's memory
/// from now on. RVG_ERR_INVALID_ARGUMENT for a device without a pool, or an address the device
/// asked no add at. On any thread, the host's lock (QEMU's BQL) held or not: waits for nothing.
/// Calls back: never.
///
/// # Safety
/// `device` is a live graphics device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_graphics_device_pool_removed(
    device: *mut RvgGraphicsDevice,
    physical_address: u64,
) -> RvgStatus {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe {
            pool_call(device, "rvg_graphics_device_pool_removed", |d| {
                d.pool_removed(physical_address)
            })
        }
    })
}

/// Asks for at least `length` more bytes of the guest's Shared pool now, ahead of the guest's own
/// ask, as that ask does: the device's pool thread takes a block of at least `length` bytes (64 MiB
/// for the guest's ask) and asks the host to map it (pool_add). QEMU calls it for its device's
/// `x-pool-grow` property. RVG_ERR_INVALID_ARGUMENT for a device without a pool or a `length` of
/// 0. On any thread: waits for nothing. Calls back: never.
///
/// # Safety
/// `device` is a live graphics device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_graphics_device_pool_grow(
    device: *mut RvgGraphicsDevice,
    length: u64,
) -> RvgStatus {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe {
            pool_call(device, "rvg_graphics_device_pool_grow", |d| {
                d.pool_grow(length)
            })
        }
    })
}

/// The range of guest physical addresses the device's Shared pool took, of those its config gave
/// (`pool_physical_address`, `pool_physical_length`): its first address, and, through `length`
/// where it is not NULL, its length, a multiple of 2 MiB; 0 and 0 for a device without a pool. The
/// device takes it at its creation: from the config's address, as many bytes as the GPU's memory
/// heaps hold, every heap summed, whatever memory lies under each (on the Vulkan platform, their
/// sizes in VkPhysicalDeviceMemoryProperties), rounded up to 2 MiB, at most the config's length,
/// and logs a warning where the config's length holds less; none, with a warning, where the GPU
/// answers no heap. QEMU reserves it at realize, after the create, and gives the firmware its end.
/// Calls back: never.
///
/// # Safety
/// `device` is a live graphics device handle; `length` is NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_graphics_device_pool_range(
    device: *const RvgGraphicsDevice,
    length: *mut u64,
) -> u64 {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        let d = unsafe { live(device, "rvg_graphics_device_pool_range") };
        let range = d.device.pool_range().unwrap_or_default();
        // SAFETY: the caller's contract.
        unsafe { handle::write_out(length, range.physical_length) };
        range.physical_address
    })
}

/// The body of the pool's functions inside their `ffi::export`: `call` on the device, its answer
/// as a status.
///
/// # Safety
/// `device` is a live graphics device handle.
unsafe fn pool_call(
    device: *mut RvgGraphicsDevice,
    name: &'static str,
    call: impl FnOnce(&GraphicsDevice<Plat>) -> Result<(), reims_vgpu::Error>,
) -> RvgStatus {
    // SAFETY: the caller's contract.
    let d = unsafe { live(device, name) };
    match call(&d.device) {
        Ok(()) => RVG_OK,
        Err(e) => status(e),
    }
}

/// Drops the host's reference; the last one closes the window, waiting for nothing, and ends the
/// device's threads without waiting for them: the callbacks' destroy runs once both have ended,
/// possibly after this returns, and the region stays mapped until then and until the window's
/// GPU work has let it go. The release of the process's last graphics device then waits, at most
/// 5 s, for its threads, its objects and their GPU work to let go of the GPU device, and lets go
/// of every GPU device the library keeps for the process, so that none is live as the process
/// exits: on NVIDIA's driver an exit with one live can end in SIGSEGV inside the driver's exit
/// handler. A device callback still running holds the wait up, one that waits for a lock the host
/// holds around this call for the whole 5 s; past them the library keeps the GPU device for the
/// process, and logs a warning saying what held it. A device created after it runs on a new GPU
/// device. The host unmaps BAR 2 from the guest first. Calls back: never.
///
/// # Safety
/// `device` is NULL or a live graphics device handle, not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_graphics_device_release(device: *mut RvgGraphicsDevice) {
    ffi::export(|| {
        if device.is_null() {
            return;
        }
        // SAFETY: the caller's contract.
        unsafe { handle::release(device) };
        // Where that ended the process's last device, the GPU devices go: the Rust API decides
        // and logs (`GraphicsDevice::let_go_of_gpu`).
        let _ = GraphicsDevice::<Plat>::let_go_of_gpu(reims_vgpu::LET_GO_BOUND);
    })
}

/// Quiesces the library ahead of the process's exit: the host's last call before it exits, made on
/// the thread that then exits, whether its graphics devices were released or kept to the exit. Not
/// from an exit handler, where the runtime supports no quiesce: a host whose exit reaches one with
/// no call made and a graphics device unreleased ends with `_exit` there instead. It takes no
/// handle: the GPU devices the library keeps are the process's. Its answer is whether the host may
/// then call `exit`. The library's threads park for good at their next call into it, and its calls
/// in progress on other threads are waited for; on the platform over metal2vulkan-rt, then, where a
/// GPU device is still live, the runtime's threads end and the GPU completes what was submitted,
/// and GPU work asked for on this thread after it is refused. Nothing is freed, and a call on
/// another thread after it never returns. Every wait ends within `timeout_ms`. True where the calls
/// ended in time and, on the Vulkan platform, the library never opened a GPU device in the process,
/// or, on the platform over metal2vulkan-rt, it keeps none (the release of the last graphics device
/// let go of them). False otherwise, the device stopped or not, and where a call outlasted the
/// wait: the library then writes its degradation counts, and the host runs its own exit work and
/// ends with `_exit` rather than `exit`, since neither platform's runtime supports an `exit` with a
/// device live (on NVIDIA's driver one can end in SIGSEGV inside the driver's exit handler), and on
/// Linux the Vulkan platform's runtime can end the process with `_exit` inside its own exit
/// handler, before the host's ran, while its work outlasts the let-go of its device (a pipeline's
/// making in the driver). A second call on the same thread answers at once, with the first answer.
/// Calls back: never.
#[unsafe(no_mangle)]
pub extern "C" fn rvg_quiesce_for_exit(timeout_ms: u64) -> bool {
    ffi::export(|| {
        GraphicsDevice::<Plat>::quiesce_for_exit(std::time::Duration::from_millis(timeout_ms))
    })
}

/// The device's region: its first byte, page-aligned, and, through `size` where it is not NULL,
/// its size, a power of two. The host maps it into the guest as BAR 2, once; it is memory of the
/// host's process (on Linux a shared mapping of a memfd of the library's, on macOS private
/// anonymous memory), zeroed, mapped while the device lives (rvg_graphics_device_release). Calls
/// back: never.
///
/// # Safety
/// `device` is a live graphics device handle; `size` is NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_graphics_device_region(
    device: *const RvgGraphicsDevice,
    size: *mut u64,
) -> *mut c_void {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        let d = unsafe { live(device, "rvg_graphics_device_region") };
        // SAFETY: the caller's contract.
        unsafe { handle::write_out(size, d.device.region_size()) };
        d.device.region_base().cast()
    })
}

/// A 32-bit read of BAR 0 at `offset`, below RVG_GRAPHICS_MSIX_OFFSET; offsets with no register
/// read 0. On any thread, the vCPU's included. Calls back: never.
///
/// # Safety
/// `device` is a live graphics device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_graphics_device_mmio_read(
    device: *mut RvgGraphicsDevice,
    offset: u64,
) -> u32 {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { live(device, "rvg_graphics_device_mmio_read") }
            .device
            .mmio_read(offset)
    })
}

/// A 32-bit write of BAR 0 at `offset`, below RVG_GRAPHICS_MSIX_OFFSET: STATUS, and DOORBELL
/// where the host's ioeventfd did not take the write; other offsets are ignored. A write of 0 to
/// STATUS resets the device, and waits for the command the device is taking, which waits for
/// nothing of the host's.
///
/// A write to RVG_GRAPHICS_REG_DOORBELL of a command queue's ring (any value but
/// RVG_GRAPHICS_DOORBELL_CONTROL) wakes the queue's thread, and may answer the queue's next
/// command buffer on the calling thread, ahead of the queue's own thread (where the guest's
/// driver took the wire's DOORBELL_ASKS, only a write that holds bit 0 beside the ring's id, which
/// no route_doorbell route matches), posting the event that says it is submitted there: raise_interrupt is then called on the calling thread, inside this
/// call, after that event is published. The host's raise_interrupt must therefore not wait for a
/// lock the host holds while it calls this. A buffer is answered only where nothing need be
/// waited for (each lock the answer needs is tried, and the queue's thread holds none of the
/// queue's messages), and the answer allocates nothing; the queue's thread then takes the
/// buffer, as it takes any. A write of RVG_GRAPHICS_DOORBELL_CONTROL may likewise take the
/// control ring's next message on the calling thread and answer it there, raising the interrupt
/// inside the call, where it is a surface download whose pages hold the surface's GPU copy
/// already and the control thread sleeps with every earlier message carried out (each lock
/// tried, nothing allocated); else the control thread takes it. What the guest sees is the same
/// either way but for when the event comes. On any thread, the vCPU's included. Calls back:
/// raise_interrupt, as above.
///
/// # Safety
/// `device` is a live graphics device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_graphics_device_mmio_write_taking(
    device: *mut RvgGraphicsDevice,
    offset: u64,
    value: u32,
) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { live(device, "rvg_graphics_device_mmio_write_taking") }
            .device
            .mmio_write_taking(offset, value);
    })
}

/// A 32-bit write of BAR 4, the doorbell pages, at `offset`, where the host maps it
/// (RvgGraphicsDeviceConfig's `doorbell_pages`) and no route_doorbell_page route took the write.
/// A write of a command queue's ring's id at a page's doorbell (`page *
/// RVG_GRAPHICS_DOORBELL_PAGE_SIZE`) does what that write to RVG_GRAPHICS_REG_DOORBELL through
/// rvg_graphics_device_mmio_write_taking does, where the ring is one of the queues of the guest
/// process the page is bound to: it wakes the queue's thread, and may answer its next command
/// buffer on the calling thread, calling raise_interrupt there inside this call. Every other
/// write is dropped and counted, logged once a kind: an unbound page's, one at another offset,
/// and one of the control ring's id or of another process's ring. On any thread, the vCPU's
/// included. Calls back: raise_interrupt, as rvg_graphics_device_mmio_write_taking does.
///
/// # Safety
/// `device` is a live graphics device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_graphics_device_doorbell_write(
    device: *mut RvgGraphicsDevice,
    offset: u64,
    value: u32,
) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { live(device, "rvg_graphics_device_doorbell_write") }
            .device
            .doorbell_write(offset, value);
    })
}

/// Hands the device the host's doorbell for the control ring: `fd`, readable once the guest
/// writes RVG_GRAPHICS_DOORBELL_CONTROL to RVG_GRAPHICS_REG_DOORBELL (QEMU: the read side of
/// the EventNotifier of memory_region_add_eventfd, an eventfd, or a pipe where the host has
/// none). The device takes a copy of it, and reads what it holds when it wakes; the host may
/// close its own. RVG_ERR_INVALID_ARGUMENT where `fd` is not open or the device has one already.
/// Calls back: never.
///
/// # Safety
/// `device` is a live graphics device handle.
#[cfg(not(windows))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_graphics_device_set_doorbell(
    device: *mut RvgGraphicsDevice,
    fd: i32,
) -> RvgStatus {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { set_doorbell(device, fd) }
    })
}

/// Hands the device the host's doorbell for the control ring, on Windows: `event`, a handle to an
/// event that is set once the guest writes RVG_GRAPHICS_DOORBELL_CONTROL to
/// RVG_GRAPHICS_REG_DOORBELL (QEMU: the event of the EventNotifier of memory_region_add_eventfd,
/// a manual-reset event, which QEMU sets for a write it matches). The device takes a copy of it
/// (DuplicateHandle), waits for it, and resets it when it wakes; the host may close its own.
/// RVG_ERR_INVALID_ARGUMENT where `event` is not a handle of this process or the device has one
/// already. Calls back: never.
///
/// # Safety
/// `device` is a live graphics device handle.
#[cfg(windows)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_graphics_device_set_doorbell(
    device: *mut RvgGraphicsDevice,
    event: *mut c_void,
) -> RvgStatus {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { set_doorbell(device, event) }
    })
}

/// The body of `rvg_graphics_device_set_doorbell` on each OS, inside its `ffi::export`:
/// `doorbell` is its `fd`, or on Windows its `event`.
///
/// # Safety
/// `device` is a live graphics device handle.
unsafe fn set_doorbell(device: *mut RvgGraphicsDevice, doorbell: RawDoorbell) -> RvgStatus {
    // SAFETY: the caller's contract.
    let d = unsafe { live(device, "rvg_graphics_device_set_doorbell") };
    if d.device.set_doorbell(doorbell) {
        RVG_OK
    } else {
        RVG_ERR_INVALID_ARGUMENT
    }
}

/// The guest's bus reset: as a write of 0 to STATUS, the device is as one just made. It takes no
/// command until the guest writes 1, its rings are as new, the display keeps its scanout but has
/// no cursor image or VBLANK, its panel is on, and the device drops every object the guest's
/// processes made, on its control thread before it takes the guest's next command. Waits for the
/// command the device is taking, and for a take of a queue's ring or a post onto one under way.
/// Calls back: never.
///
/// # Safety
/// `device` is a live graphics device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_graphics_device_reset(device: *mut RvgGraphicsDevice) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { live(device, "rvg_graphics_device_reset") }
            .device
            .reset();
    })
}

/// What the device's display shows now, read back into `dst`: the guest's scanout, or the
/// IOSurface its display pipe presents, through the display's colour, drawn as the device's
/// window draws it into a drawable of the display's mode, the size the guest is told (the
/// window's size in pixels, or 1920 x 1080 with no window); black where the guest shows nothing,
/// and while the guest's display sleeps (the panel off).
/// `*height` rows of `*width * 4` bytes each, BGRA8: B, G, R and A in memory, pixman's
/// x8r8g8b8 on a little-endian host. `width` and `height` (each NULL or writable) get the mode's
/// size whatever the result, so a call with a NULL `dst` asks the size. RVG_ERR_INVALID_ARGUMENT,
/// with nothing written to `dst`, where it is NULL or `dst_length` is below the mode's
/// `width * height * 4` bytes; RVG_ERR_METAL where the platform refuses the draw or its
/// texture; RVG_ERR_GPU where the draw ended with an error. Waits on the calling thread for the draw on the GPU, and for no lock
/// the device's callbacks may wait on, so the host may hold its own (QEMU's BQL) but need not:
/// QEMU calls it on a thread of its pool. The guest sees nothing of it: no present is answered,
/// and the mode is only asked. One read runs at a time. On any thread. Calls back: never.
///
/// # Safety
/// `device` is a live graphics device handle; `dst` is NULL or holds `dst_length` writable bytes;
/// `width` and `height` are each NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_graphics_device_read_display(
    device: *mut RvgGraphicsDevice,
    dst: *mut u8,
    dst_length: usize,
    width: *mut u32,
    height: *mut u32,
) -> RvgStatus {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe {
            read_display(
                "rvg_graphics_device_read_display",
                device,
                0,
                (dst, dst_length),
                (width, height),
            )
        }
    })
}

/// Where the guest last put its cursor's hot spot, in the display's pixels, each axis kept to 0
/// to 65535: (0, 0) before it moved it and after a reset. The host reads it at its cursor_move
/// (RvgGraphicsCallbacks). On any thread. Calls back: never.
///
/// # Safety
/// `device` is a live graphics device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_graphics_device_cursor_position(
    device: *const RvgGraphicsDevice,
) -> RvgDisplayCoord {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        let (x, y) = unsafe { live(device, "rvg_graphics_device_cursor_position") }
            .device
            .cursor_position();
        RvgDisplayCoord { x, y }
    })
}

/// What display `display` of the device shows now, read back into `dst`, as
/// rvg_graphics_device_read_display reads display 0's, which is this with `display` 0: at the
/// display's own mode, the size the guest is told for it. RVG_ERR_INVALID_ARGUMENT, with nothing
/// written to `dst` and `*width` and `*height` 0, for a display at or past the device's count
/// (RvgGraphicsDeviceConfig's `display_count`); otherwise as rvg_graphics_device_read_display.
/// Reads of one display run one at a time. On any thread. Calls back: never.
///
/// # Safety
/// `device` is a live graphics device handle; `dst` is NULL or holds `dst_length` writable bytes;
/// `width` and `height` are each NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_graphics_device_read_display_at(
    device: *mut RvgGraphicsDevice,
    display: u32,
    dst: *mut u8,
    dst_length: usize,
    width: *mut u32,
    height: *mut u32,
) -> RvgStatus {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe {
            read_display(
                "rvg_graphics_device_read_display_at",
                device,
                display,
                (dst, dst_length),
                (width, height),
            )
        }
    })
}

/// The body of `rvg_graphics_device_read_display` and `rvg_graphics_device_read_display_at`
/// (`call`), inside their `ffi::export`: display `display` into `dst` and its length, its size
/// out through `size`'s width and height.
///
/// # Safety
/// As rvg_graphics_device_read_display_at's.
unsafe fn read_display(
    call: &'static str,
    device: *mut RvgGraphicsDevice,
    display: u32,
    (dst, dst_length): (*mut u8, usize),
    (width, height): (*mut u32, *mut u32),
) -> RvgStatus {
    // SAFETY: the caller's contract.
    let d = unsafe { live(device, call) };
    let mut none: [u8; 0] = [];
    let dst: &mut [u8] = if dst.is_null() {
        &mut none
    } else {
        // SAFETY: the caller's contract.
        unsafe { std::slice::from_raw_parts_mut(dst, dst_length) }
    };
    let ((w, h), read) = d.device.read_display_at(display, dst);
    // SAFETY: the caller's contract.
    unsafe {
        handle::write_out(width, w);
        handle::write_out(height, h);
    }
    match read {
        Ok(()) => RVG_OK,
        Err(e) => status(e),
    }
}

/// Where the guest last put display `display`'s cursor's hot spot, in that display's pixels, as
/// rvg_graphics_device_cursor_position reads display 0's, which is this with `display` 0: (0, 0)
/// before the guest moved it, after a reset, and for a display at or past the device's count. The
/// host reads it at its cursor_move_at. On any thread. Calls back: never.
///
/// # Safety
/// `device` is a live graphics device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_graphics_device_cursor_position_at(
    device: *const RvgGraphicsDevice,
    display: u32,
) -> RvgDisplayCoord {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        let (x, y) = unsafe { live(device, "rvg_graphics_device_cursor_position_at") }
            .device
            .cursor_position_at(display);
        RvgDisplayCoord { x, y }
    })
}

#[cfg(test)]
mod tests;
