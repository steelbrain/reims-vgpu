//! EFI_GRAPHICS_OUTPUT_PROTOCOL on product BAR1 (BGRA8 linear).
//!
//! Protocol handlers are thin UEFI adapters over [`reims_vgpu_efi::paint`] —
//! unit tests drive those paint paths on the host without boot services.

use core::ptr;
use core::slice;
use reims_vgpu_efi::paint::{
    self, BltPixel, BltRect, GopStatus, BLT_BUFFER_TO_VIDEO, BLT_VIDEO_FILL, BLT_VIDEO_TO_BUFFER,
    FB_BYTES, FB_H, FB_W, PIXEL_BGR,
};
use uefi::prelude::*;
use uefi::{boot, guid};

pub const GOP_GUID: uefi::Guid = guid!("9042a9de-23dc-4a38-96fb-7aded080516a");

#[repr(C)]
#[derive(Clone, Copy)]
struct ModeInfo {
    version: u32,
    horizontal_resolution: u32,
    vertical_resolution: u32,
    pixel_format: u32,
    pixel_information: [u32; 4],
    pixels_per_scan_line: u32,
}

#[repr(C)]
struct Mode {
    max_mode: u32,
    mode: u32,
    info: *mut ModeInfo,
    size_of_info: usize,
    frame_buffer_base: u64,
    frame_buffer_size: usize,
}

#[repr(C)]
pub struct Gop {
    query_mode: unsafe extern "efiapi" fn(
        this: *mut Gop,
        mode_number: u32,
        size_of_info: *mut usize,
        info: *mut *mut ModeInfo,
    ) -> Status,
    set_mode: unsafe extern "efiapi" fn(this: *mut Gop, mode_number: u32) -> Status,
    blt: unsafe extern "efiapi" fn(
        this: *mut Gop,
        blt_buffer: *mut BltPixel,
        blt_operation: u32,
        source_x: usize,
        source_y: usize,
        destination_x: usize,
        destination_y: usize,
        width: usize,
        height: usize,
        delta: usize,
    ) -> Status,
    mode: *mut Mode,
}

/// Private context: Gop + Mode + ModeInfo + fb pointer (pool-allocated, permanent).
#[repr(C)]
pub struct GopCtx {
    gop: Gop,
    mode: Mode,
    info: ModeInfo,
    fb: *mut u8,
}

unsafe fn ctx_from_this(this: *mut Gop) -> *mut GopCtx {
    // Gop is the first field of GopCtx.
    this as *mut GopCtx
}

fn status_from(g: GopStatus) -> Status {
    match g {
        GopStatus::Success => Status::SUCCESS,
        GopStatus::InvalidParameter => Status::INVALID_PARAMETER,
        GopStatus::Unsupported => Status::UNSUPPORTED,
        GopStatus::DeviceError => Status::DEVICE_ERROR,
    }
}

/// UEFI Spec: QueryMode returns a **callee-allocated** pool buffer the caller frees.
unsafe extern "efiapi" fn gop_query(
    this: *mut Gop,
    mode_number: u32,
    size_of_info: *mut usize,
    info: *mut *mut ModeInfo,
) -> Status {
    if this.is_null() || size_of_info.is_null() || info.is_null() {
        return Status::INVALID_PARAMETER;
    }
    if paint::query_mode_ok(mode_number) != GopStatus::Success {
        return Status::INVALID_PARAMETER;
    }
    let ctx = &*ctx_from_this(this);
    let layout = core::alloc::Layout::new::<ModeInfo>();
    let raw = match boot::allocate_pool(boot::MemoryType::BOOT_SERVICES_DATA, layout.size()) {
        Ok(p) => p,
        Err(e) => return e.status(),
    };
    let out = raw.as_ptr() as *mut ModeInfo;
    ptr::write(out, ctx.info);
    *size_of_info = size_of::<ModeInfo>();
    *info = out;
    Status::SUCCESS
}

unsafe extern "efiapi" fn gop_set(this: *mut Gop, mode_number: u32) -> Status {
    if this.is_null() {
        return Status::INVALID_PARAMETER;
    }
    if paint::set_mode_ok(mode_number) != GopStatus::Success {
        return Status::UNSUPPORTED;
    }
    let ctx = &*ctx_from_this(this);
    if ctx.fb.is_null() {
        return Status::DEVICE_ERROR;
    }
    let fb = slice::from_raw_parts_mut(ctx.fb, FB_BYTES);
    paint::clear_black(fb);
    Status::SUCCESS
}

unsafe extern "efiapi" fn gop_blt(
    this: *mut Gop,
    blt_buffer: *mut BltPixel,
    blt_operation: u32,
    source_x: usize,
    source_y: usize,
    destination_x: usize,
    destination_y: usize,
    width: usize,
    height: usize,
    delta: usize,
) -> Status {
    if this.is_null() {
        return Status::INVALID_PARAMETER;
    }
    let ctx = &*ctx_from_this(this);
    if ctx.fb.is_null() {
        return Status::DEVICE_ERROR;
    }
    let fb = slice::from_raw_parts_mut(ctx.fb, FB_BYTES);

    // An empty rectangle is refused before anything is derived from it. `paint`
    // states the same rule and would refuse too, but the pitch and the buffer
    // span are both computed from `width`/`height` on the way there, and a zero
    // pitch is what the `.max(1)` this replaces was hiding.
    if width == 0 || height == 0 {
        return Status::INVALID_PARAMETER;
    }

    let rect = BltRect {
        source_x,
        source_y,
        destination_x,
        destination_y,
        width,
        height,
    };

    // Bound BltBuffer for ops that need it.
    let needs_buf = matches!(
        blt_operation,
        BLT_VIDEO_FILL | BLT_BUFFER_TO_VIDEO | BLT_VIDEO_TO_BUFFER
    );
    let mut buf_storage: Option<&mut [BltPixel]> = None;
    if needs_buf {
        if blt_buffer.is_null() {
            return Status::INVALID_PARAMETER;
        }
        let Some(row_px) = paint::row_pitch_pixels(delta, width) else {
            return Status::INVALID_PARAMETER;
        };
        // The length is *exactly* what the operation reaches, because the UEFI
        // Blt signature carries no buffer size: the caller's rectangle and
        // `Delta` are the only statement of how large the BltBuffer is, and the
        // spec requires it to be at least this large. `paint::blt` re-checks
        // the same span against `buf.len()`, which on this path can only ever
        // compare the number to itself — that check exists for the crate's own
        // host tests, which pass a real slice. Both spans come from
        // `BltRect::buffer_pixels_needed` so the two cannot disagree.
        let n = match blt_operation {
            BLT_VIDEO_FILL => 1usize,
            BLT_BUFFER_TO_VIDEO => rect.buffer_pixels_needed(source_x, source_y, row_px),
            BLT_VIDEO_TO_BUFFER => rect.buffer_pixels_needed(destination_x, destination_y, row_px),
            _ => 0,
        };
        if n == 0 {
            return Status::INVALID_PARAMETER;
        }
        buf_storage = Some(slice::from_raw_parts_mut(blt_buffer, n));
    }

    status_from(paint::blt(
        fb,
        buf_storage,
        blt_operation,
        source_x,
        source_y,
        destination_x,
        destination_y,
        width,
        height,
        delta,
        None,
    ))
}

/// Allocate permanent `GopCtx`, fill slate, return interface pointer for install.
pub fn build_ctx(fb_base: u64) -> Result<*mut Gop, Status> {
    let layout = core::alloc::Layout::new::<GopCtx>();
    let raw = match boot::allocate_pool(boot::MemoryType::BOOT_SERVICES_DATA, layout.size()) {
        Ok(p) => p,
        Err(e) => return Err(e.status()),
    };

    let ctx = raw.as_ptr() as *mut GopCtx;
    unsafe {
        ptr::write_bytes(ctx as *mut u8, 0, size_of::<GopCtx>());
        (*ctx).fb = fb_base as *mut u8;
        (*ctx).info = ModeInfo {
            version: 0,
            horizontal_resolution: FB_W as u32,
            vertical_resolution: FB_H as u32,
            pixel_format: PIXEL_BGR,
            pixel_information: [0; 4],
            pixels_per_scan_line: FB_W as u32,
        };
        (*ctx).mode = Mode {
            max_mode: 1,
            mode: 0,
            info: core::ptr::addr_of_mut!((*ctx).info),
            size_of_info: size_of::<ModeInfo>(),
            frame_buffer_base: fb_base,
            // Full BAR1 aperture (16 MiB), not only the visible 1920×1080 window.
            // VMware A/B keeps console on the display BAR after "console relocated";
            // advertising the full aperture matches a permanent VRAM region rather
            // than a tight boot-services-sized buffer (live freeze: relocate to
            // system RAM at 0xf1000000 while BAR1 goes idle).
            frame_buffer_size: 16 * 1024 * 1024,
        };
        (*ctx).gop = Gop {
            query_mode: gop_query,
            set_mode: gop_set,
            blt: gop_blt,
            mode: core::ptr::addr_of_mut!((*ctx).mode),
        };
        let fb = slice::from_raw_parts_mut((*ctx).fb, FB_BYTES);
        paint::fill_slate(fb);
        Ok(core::ptr::addr_of_mut!((*ctx).gop))
    }
}
