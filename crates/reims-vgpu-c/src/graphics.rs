//! The graphics device: the host side of the reims-graphics device (PCI 1234:5247, our own
//! protocol, `docs/graphics-kext/design.md`), [`RvgGraphicsDeviceConfig`], and the
//! `rvg_graphics_device_*` functions.
//!
//! The host makes the device, maps its region into the guest as BAR 2 (64-bit, prefetchable),
//! sends BAR 0's accesses to it, hands it the `ioeventfd` it registers on `DOORBELL`, and raises
//! MSI-X vector 0 when it asks. The guest's protocol is the wire header's
//! (`reims_graphics_wire.h`); this API needs none of it.

use std::ffi::c_void;
use std::sync::Arc;

use reims_vgpu::{GraphicsCallbacks, GraphicsConfig, GraphicsDevice, GraphicsRaiseFn};

use crate::device::{RvgMemoryMap, RvgMemoryRangeFile, memory_map};
use crate::ffi;
use crate::handle;
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
/// rvg_graphics_device_mmio_read and rvg_graphics_device_mmio_write, below the MSI-X table.
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

/// Where the guest's allocations start in the region, the scanout among them: below it are the
/// control rings, which the host's dirty log need not cover.
pub const RVG_GRAPHICS_ALLOCATIONS_OFFSET: u64 = 0x40_0000;

const _: () = {
    use reims_graphics::wire;
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
    assert!(RVG_GRAPHICS_ALLOCATIONS_OFFSET == wire::RGFX_ALLOCATIONS_OFFSET);
};

/// `raise_interrupt(ctx, vector)`: raise the device's MSI-X vector `vector` (0).
///
/// Thread: the one that posts the event it raises for: the device's control thread
/// (`reims-vgpu.graphics-control`), a lane that carries out a guest process's control messages
/// (`reims-vgpu.graphics-context-<n>`), its presents (`reims-vgpu.graphics-presents`) or a
/// block's pages and downloads (`reims-vgpu.graphics-block-<offset>`), its refresh thread
/// (`reims-vgpu.graphics-refresh`), a command queue's service thread
/// (`reims-vgpu.graphics-queue-<id>`) or one of its lane's threads, which walk and commit its
/// command buffers (`reims-vgpu.graphics-lane-<id>`), a pipeline compile's thread
/// (`reims-vgpu.graphics-compile-<n>`), a thread that makes IOSurfaces' and pages blocks' views
/// (`reims-vgpu.graphics-surfaces-<i>`), a thread that makes indirect command buffers
/// (`reims-vgpu.icb-create-<i>`), a thread that lets go of a closed process's objects and answers
/// its close (`reims-vgpu.graphics-closes-<i>`), or the platform's thread that runs a command
/// buffer's scheduled or completed handler; or the host's own, inside
/// rvg_graphics_device_mmio_write_taking.
/// Never inside another call of the host's into the library. The library may hold locks of its
/// own around it (docs/api.md section 5 names them), but none that a function it may call waits
/// for, directly or through a thread of the library's, so it may call them at once. It may wait
/// for a lock the host holds while it calls the device (QEMU's BQL), but not for one the host
/// holds while it calls rvg_graphics_device_mmio_write_taking. May call: rvg_status_message and
/// the rvg_graphics_device_* functions, except the release of its own device. Without it the
/// guest gets no interrupt.
pub type RvgGraphicsRaiseFn = Option<unsafe extern "C-unwind" fn(ctx: *mut c_void, vector: u32)>;

/// `scanout_written(ctx, offset, length)`: whether the guest wrote any of the `length` bytes at
/// `offset` in the region since the last call, which starts the next span: the host's dirty log
/// of the guest's writes to that range, taken and cleared (QEMU: memory_region_set_log on the
/// region, and memory_region_snapshot_and_clear_dirty). The range lies at or past
/// RVG_GRAPHICS_ALLOCATIONS_OFFSET. The display shows the scanout again only where it answers
/// true.
///
/// Thread: the device's refresh thread (`reims-vgpu.graphics-refresh`), at each refresh of the
/// display, never inside a call of the host's into the library, and with no lock of the
/// library's held: it may wait for a lock the host holds while it calls the device (QEMU's BQL).
/// May call: as raise_interrupt. Without it, every refresh shows the scanout again.
pub type RvgGraphicsScanoutWrittenFn =
    Option<unsafe extern "C-unwind" fn(ctx: *mut c_void, offset: u64, length: u64) -> bool>;

/// The graphics device's callbacks.
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
    pub scanout_written: RvgGraphicsScanoutWrittenFn,
}

// SAFETY: repr(C), a u32 first; a pointer and optional function pointers.
unsafe impl HostStruct for RvgGraphicsCallbacks {}

/// The graphics device's config.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RvgGraphicsDeviceConfig {
    /// sizeof(RvgGraphicsDeviceConfig) as the host was compiled.
    pub struct_size: u32,
    /// NULL is a table with every callback absent.
    pub callbacks: *const RvgGraphicsCallbacks,
    /// NULL is no window: the display offers 1920 x 1080 at 60 Hz and shows nothing. Otherwise
    /// the library opens a window for the display, as for a display's (RvgDisplayConfig.window),
    /// with its input as the window's callbacks; its flags are ignored, and it opens sized for
    /// its mode (1920 x 1080 where it has none). The window keeps its size: the display offers
    /// the guest one mode, the window's size in pixels at the refresh rate of the host screen it
    /// opens on, and tells the guest when the person resizes it. The guest's scanout is blitted
    /// out of the region into the window's drawables on the GPU, at the screen's rate, where
    /// scanout_written says the guest wrote it.
    pub window: *const RvgWindowConfig,
    /// The guest's RAM, or NULL. With it, where the host's page is the guest's 4 KiB, the GPU uses
    /// an IOSurface's own pages in place, through views of them made as a device's memory map
    /// makes its views (RvgDeviceConfig.memory_map); without it each surface's GPU view is a copy
    /// of its pages in the region. The map's pages stay the guest's RAM while the device lives:
    /// a host that discards guest RAM (a balloon, virtio-mem, hot-unplug) passes NULL.
    pub memory_map: *const RvgMemoryMap,
    /// With `memory_map`: the files behind its ranges, checked and duplicated as
    /// RvgDeviceConfig.memory_files are, with the same refusals; NULL for none. Files without a
    /// map are refused (RVG_ERR_INVALID_ARGUMENT).
    pub memory_files: *const RvgMemoryRangeFile,
}

// SAFETY: repr(C), a u32 first; pointers.
unsafe impl HostStruct for RvgGraphicsDeviceConfig {
    /// The first version ends at `window`: a host built against it passes 24, and has no
    /// `memory_map`.
    const V1_SIZE: usize = 24;
}

const _: () = assert!(
    std::mem::offset_of!(RvgGraphicsDeviceConfig, memory_map) == RvgGraphicsDeviceConfig::V1_SIZE
);

/// The graphics device.
pub struct RvgGraphicsDevice {
    /// The window's table, for a device with a window: held here so that a table with no
    /// function still has its destroy at the release. It goes before the device, whose window
    /// holds the table's other references until it closes.
    window_table: Option<Arc<Table>>,
    device: WithTable<GraphicsDevice<Plat>>,
}

impl Drop for RvgGraphicsDevice {
    /// The window's table first, then the device (`crate::display`'s order).
    fn drop(&mut self) {
        drop(self.window_table.take());
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
/// its display where the config gives one. Its config and window config are read, with their
/// tables, before either table is the library's: a struct_size refusal runs no destroy, and a
/// later refusal runs both, the window's first. Refusals: RVG_ERR_INVALID_ARGUMENT,
/// RVG_ERR_UNSUPPORTED, RVG_ERR_HOST_RESOURCES (the host refused the region's memory, a pipe or
/// a thread). Calls back: never, but a refusal's destroys.
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
    // From here the tables are the library's: a refusal ends them before it returns, the
    // window's first, as a release does.
    let table = Table::new(callbacks.ctx, callbacks.destroy);
    let window_table = window.as_ref().map(|(_, t)| Table::new(t.ctx, t.destroy));
    let refuse = |s: RvgStatus| {
        if let Some(t) = &window_table {
            t.end();
        }
        table.end();
        s
    };
    if device.is_null() {
        return refuse(RVG_ERR_INVALID_ARGUMENT);
    }
    // SAFETY: the config's contract: NULL or a live map, and NULL or its files.
    let memory_map = match unsafe { memory_map(c.memory_map, c.memory_files) } {
        Ok(m) => m,
        Err(s) => return refuse(s),
    };
    let raise_interrupt = callbacks.raise_interrupt.map(|f| {
        let t = table.clone();
        let raise: GraphicsRaiseFn = Arc::new(move |vector| {
            // SAFETY: the host's function over its ctx.
            ffi::host_call("raise_interrupt", || unsafe { f(t.ctx(), vector) });
        });
        raise
    });
    let scanout_written = callbacks.scanout_written.map(|f| {
        let t = table.clone();
        let written: reims_vgpu::ScanoutWrittenFn = Arc::new(move |offset, length| {
            // SAFETY: the host's function over its ctx.
            ffi::host_call("scanout_written", || unsafe { f(t.ctx(), offset, length) })
        });
        written
    });
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks {
            raise_interrupt,
            scanout_written,
        },
        window: match (&window, &window_table) {
            (Some((w, wc)), Some(t)) => Some(w.to_config(window_callbacks(t, wc))),
            _ => None,
        },
        memory_map,
    };
    match GraphicsDevice::with_platform(<Plat as reims_vgpu::ApiPlatform>::process(), config) {
        Ok(d) => {
            let h = handle::new(RvgGraphicsDevice {
                window_table,
                device: WithTable::new(d, table),
            });
            // SAFETY: checked non-NULL; the caller's contract.
            unsafe { device.write(h) };
            RVG_OK
        }
        Err(e) => refuse(status(e)),
    }
}

/// Drops the host's reference; the last one closes the window, waiting for nothing, and ends the
/// device's threads without waiting for them: the callbacks' destroy runs once both have ended,
/// possibly after this returns, and the region stays mapped until then and until the window's
/// GPU work has let it go. The host unmaps BAR 2 from the guest first. Calls back: never.
///
/// # Safety
/// `device` is NULL or a live graphics device handle, not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_graphics_device_release(device: *mut RvgGraphicsDevice) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { handle::release(device) }
    })
}

/// The device's region: its first byte, page-aligned, and, through `size` where it is not NULL,
/// its size, a power of two. The host maps it into the guest as BAR 2, once; it is private
/// anonymous memory of the host's process, zeroed, mapped while the device lives
/// (rvg_graphics_device_release). Calls back: never.
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
/// nothing of the host's. On any thread, the vCPU's included. Calls back: never.
///
/// # Safety
/// `device` is a live graphics device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_graphics_device_mmio_write(
    device: *mut RvgGraphicsDevice,
    offset: u64,
    value: u32,
) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { live(device, "rvg_graphics_device_mmio_write") }
            .device
            .mmio_write(offset, value);
    })
}

/// rvg_graphics_device_mmio_write, but a write to RVG_GRAPHICS_REG_DOORBELL of a command queue's
/// ring (any value but RVG_GRAPHICS_DOORBELL_CONTROL) may answer the queue's next command buffer
/// on the calling thread, ahead of the queue's own thread, posting the event that says it is
/// submitted there: raise_interrupt is then called on the calling thread, inside this call, after
/// that event is published. The host's raise_interrupt must therefore not wait for a lock the host
/// holds while it calls this. A buffer is answered only where nothing need be waited for (each
/// lock the answer needs is tried, and the queue's thread holds none of the queue's messages), and
/// the answer allocates nothing; the queue's thread then takes the buffer, as it takes any, and
/// the write wakes it, as rvg_graphics_device_mmio_write does. What the guest sees is the same
/// either way but for when the event comes. It waits for nothing rvg_graphics_device_mmio_write
/// does not. On any thread, the vCPU's included. Calls back: raise_interrupt, as above.
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

/// Hands the device the host's doorbell for the control ring: `fd`, readable once the guest
/// writes RVG_GRAPHICS_DOORBELL_CONTROL to RVG_GRAPHICS_REG_DOORBELL (QEMU: the read side of
/// the EventNotifier of memory_region_add_eventfd, an eventfd, or a pipe where the host has
/// none). The device takes a copy of it, and reads what it holds when it wakes; the host may
/// close its own. RVG_ERR_INVALID_ARGUMENT where `fd` is not open or the device has one already.
/// Calls back: never.
///
/// # Safety
/// `device` is a live graphics device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_graphics_device_set_doorbell(
    device: *mut RvgGraphicsDevice,
    fd: i32,
) -> RvgStatus {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        let d = unsafe { live(device, "rvg_graphics_device_set_doorbell") };
        if d.device.set_doorbell(fd) {
            RVG_OK
        } else {
            RVG_ERR_INVALID_ARGUMENT
        }
    })
}

/// The guest's bus reset: as a write of 0 to STATUS, the device is as one just made. It takes no
/// command until the guest writes 1, its rings are as new, the display keeps its scanout but has
/// no cursor image or VBLANK, and the device drops every object the guest's processes made, on
/// its control thread before it takes the guest's next command. Waits for the command the device
/// is taking, and for a take of a queue's ring or a post onto one under way. Calls back: never.
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

#[cfg(test)]
mod tests;
