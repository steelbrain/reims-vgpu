//! Zero copy on VideoToolbox: a decoded frame written by a pixel transfer
//! straight into the guest's buffer, with no copy by the CPU.
//!
//! On macOS the host lends a held guest buffer as its own IOSurface, the
//! one reims-vgpu's Metal platform made over the guest's pages
//! (`DeviceMemory::io_surface`). The writer wraps it in a pixel buffer at
//! the guest's planes (`CVPixelBufferCreateWithIOSurface`) and has a
//! [`Transfer`] write the frame into it, then releases the pixel buffer
//! before it returns: a reference of the IOSurface is a hold the guest
//! sees, since the host's mapper answers the guest's unmap Busy while
//! anything but it retains the surface, so nothing made of it is kept, in a
//! `DeviceImports` or anywhere, past the write.
//!
//! **A read-only destination panics the Mac.** Apple's scaler wrote into an
//! IOSurface over pages read-only for its clients, and the kernel panicked,
//! twice. A pixel transfer may run on that scaler, and the guest decides
//! whether its buffers are read-only. So nothing reaches CoreVideo or
//! VideoToolbox until three checks of the writer's own have passed, besides
//! the host's refusal to lend such a buffer at all (reims-vgpu's
//! `device_lend.rs`):
//! 1. the host's word, read before the surface is: `LentIoSurface::writable`,
//!    which reims-vgpu asks again of its record of the surface;
//! 2. the surface's base address: a protected surface has none;
//! 3. the process's own mapping of the surface's bytes, from its base address
//!    to its allocation's end: every page writable (`mach_vm_region`), as a
//!    surface over pages read-only for its clients is not, its mapping
//!    read-only (reims-vgpu-metal's `surface_bytes.rs`: a write through it is
//!    a bus error, measured there on the M3 Max).
//!
//! Then the frame and the buffer must be one format and one size, and an
//! even one, so that the transfer copies and neither converts nor scales
//! nor repeats an edge, and the surface's planes must be those core checked
//! against its memory (`GuestTarget`), with the pixel buffer's the same, so
//! the transfer writes nowhere else.
//! What fails a check is declined with why, and core copies that frame on
//! the CPU.

use std::ffi::c_void;

use reims_vgpu_video_core::backend::{BackendError, DeviceWrite, PixelFormat};
use reims_vgpu_video_core::device_memory::{GuestTarget, LentIoSurface};
use reims_vgpu_video_core::sync::Mutex;

use super::ffi::core_foundation::{OSStatus, OSType};
use super::ffi::core_video::{
    CVPixelBufferCreateWithIOSurface, CVPixelBufferRef, IOSurfaceRef, kCVReturnSuccess,
};
use super::ffi::io_surface::{
    IOSurfaceGetAllocSize, IOSurfaceGetBaseAddress, IOSurfaceGetBaseAddressOfPlane,
    IOSurfaceGetBytesPerRowOfPlane, IOSurfaceGetHeight, IOSurfaceGetHeightOfPlane,
    IOSurfaceGetPixelFormat, IOSurfaceGetPlaneCount, IOSurfaceGetWidth, IOSurfaceGetWidthOfPlane,
};
use super::ffi::mach::{
    KERN_SUCCESS, VM_PROT_WRITE, VM_REGION_BASIC_INFO_64, VM_REGION_BASIC_INFO_COUNT_64,
    mach_port_t, mach_task_self_, mach_vm_region, vm_region_basic_info_data_64_t,
};
use super::ffi::video_toolbox::VTPixelTransferSessionTransferImage;
use super::pixel_buffer::{PixelBuffer, fourcc_text};
use super::session::{self, Session};

/// What a transfer writes: a decoded frame's pixel buffer, its format, and
/// the picture's size at its top left.
pub(crate) struct Source<'a> {
    pub(crate) image: &'a PixelBuffer,
    pub(crate) format: PixelFormat,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// One plane of a destination surface, as the surface answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SurfacePlane {
    /// Its first byte's offset from the surface's base address.
    pub(crate) offset: usize,
    /// The bytes from one row to the next.
    pub(crate) bytes_per_row: usize,
    /// Its width in elements.
    pub(crate) width: usize,
    /// Its rows.
    pub(crate) height: usize,
}

/// A destination surface as the writer reads it: its own answers, and a
/// pixel buffer over it. The guest's IOSurface ([`IoSurface`]), or a test's
/// stand-in.
pub(crate) trait Destination {
    /// Its base address: NULL for a protected surface.
    fn base_address(&self) -> *mut c_void;
    /// The bytes of its allocation.
    fn alloc_size(&self) -> usize;
    /// Its pixel format.
    fn pixel_format(&self) -> OSType;
    /// Its width and height in pixels.
    fn size(&self) -> (usize, usize);
    /// Its planes, in order.
    fn planes(&self) -> Vec<SurfacePlane>;
    /// A new pixel buffer over it, which a transfer writes.
    fn pixel_buffer(&self) -> Result<PixelBuffer, String>;
}

/// What writes a frame into a pixel buffer over the guest's buffer:
/// VideoToolbox's pixel transfer, or a test's stand-in.
pub(crate) trait Transfer: Send + Sync {
    /// Writes `source` into `destination`, the same format and size, and
    /// returns once it is there: the borrow of `destination` is exclusive
    /// for as long as the transfer writes it.
    fn transfer(&self, source: &PixelBuffer, destination: &mut PixelBuffer)
    -> Result<(), OSStatus>;
}

/// VideoToolbox's pixel transfer, as a decoder's frames write through it:
/// one session, made at the first write, used by one write at a time and
/// invalidated with the decoder's last frame. The session has no property
/// set: a transfer of one format and size into another buffer copies
/// (module docs; measured against the CPU's copy).
#[derive(Default)]
pub(crate) struct PixelTransfer(Mutex<Option<Session>>);

impl Transfer for PixelTransfer {
    fn transfer(
        &self,
        source: &PixelBuffer,
        destination: &mut PixelBuffer,
    ) -> Result<(), OSStatus> {
        let mut session = self.0.lock();
        if session.is_none() {
            *session = Some(session::pixel_transfer()?);
        }
        let Some(session) = session.as_ref() else {
            return Err(-1);
        };
        // SAFETY: a live session and two live pixel buffers; the transfer
        // is synchronous, done when it returns.
        let status = unsafe {
            VTPixelTransferSessionTransferImage(
                session.as_ptr(),
                source.as_ptr(),
                destination.as_ptr(),
            )
        };
        if status == 0 { Ok(()) } else { Err(status) }
    }
}

/// A declined write, with why.
fn declined(why: impl Into<String>) -> Result<DeviceWrite, BackendError> {
    Ok(DeviceWrite::Declined(why.into()))
}

/// The guest buffer's IOSurface, as the host lent it, for as long as the
/// lend.
pub(crate) struct IoSurface<'a>(LentIoSurface<'a>);

impl<'a> IoSurface<'a> {
    /// The IOSurface `lent`.
    pub(crate) fn of(lent: LentIoSurface<'a>) -> Self {
        Self(lent)
    }

    fn as_ptr(&self) -> IOSurfaceRef {
        self.0.as_ptr().as_ptr()
    }
}

impl Destination for IoSurface<'_> {
    fn base_address(&self) -> *mut c_void {
        // SAFETY: a live IOSurface, which the host's lend keeps.
        unsafe { IOSurfaceGetBaseAddress(self.as_ptr()) }
    }

    fn alloc_size(&self) -> usize {
        // SAFETY: as above.
        unsafe { IOSurfaceGetAllocSize(self.as_ptr()) }
    }

    fn pixel_format(&self) -> OSType {
        // SAFETY: as above.
        unsafe { IOSurfaceGetPixelFormat(self.as_ptr()) }
    }

    fn size(&self) -> (usize, usize) {
        // SAFETY: as above.
        unsafe {
            (
                IOSurfaceGetWidth(self.as_ptr()),
                IOSurfaceGetHeight(self.as_ptr()),
            )
        }
    }

    fn planes(&self) -> Vec<SurfacePlane> {
        let surface = self.as_ptr();
        let base = self.base_address().addr();
        // SAFETY: as above, each plane asked under its count.
        unsafe {
            (0..IOSurfaceGetPlaneCount(surface))
                .map_while(|plane| {
                    let at = IOSurfaceGetBaseAddressOfPlane(surface, plane).addr();
                    Some(SurfacePlane {
                        offset: at.checked_sub(base)?,
                        bytes_per_row: IOSurfaceGetBytesPerRowOfPlane(surface, plane),
                        width: IOSurfaceGetWidthOfPlane(surface, plane),
                        height: IOSurfaceGetHeightOfPlane(surface, plane),
                    })
                })
                .collect()
        }
    }

    fn pixel_buffer(&self) -> Result<PixelBuffer, String> {
        let mut buffer: CVPixelBufferRef = std::ptr::null_mut();
        // SAFETY: a live IOSurface, no attributes, and a live local for the
        // out pointer; the pixel buffer retains the surface until it is
        // released.
        let status = unsafe {
            CVPixelBufferCreateWithIOSurface(
                std::ptr::null(),
                self.as_ptr(),
                std::ptr::null(),
                &raw mut buffer,
            )
        };
        // SAFETY: a Create call's result, or NULL: taken before the status
        // is read, so that no reference of the guest's surface outlives
        // the call whatever it answers.
        let buffer = unsafe { PixelBuffer::take(buffer) };
        if status != kCVReturnSuccess {
            return Err(format!(
                "CVPixelBufferCreateWithIOSurface answered {status}"
            ));
        }
        buffer.ok_or_else(|| "CVPixelBufferCreateWithIOSurface gave none".to_owned())
    }
}

/// Whether every page of the `length` bytes from `start` is mapped
/// writable in this process (module docs, the third check); `Err` says
/// where not.
pub(crate) fn writable_mapping(start: usize, length: usize) -> Result<(), String> {
    let end = start
        .checked_add(length)
        .ok_or("the buffer's bytes pass the address space")?;
    let (mut at, end) = (start as u64, end as u64);
    while at < end {
        let (mut address, mut size) = (at, 0u64);
        let mut info = vm_region_basic_info_data_64_t::default();
        let mut count = VM_REGION_BASIC_INFO_COUNT_64;
        let mut object: mach_port_t = 0;
        // SAFETY: the process's own task port; live locals for every out
        // pointer, `info` with room for the count of ints given.
        let status = unsafe {
            mach_vm_region(
                mach_task_self_,
                &raw mut address,
                &raw mut size,
                VM_REGION_BASIC_INFO_64,
                (&raw mut info).cast(),
                &raw mut count,
                &raw mut object,
            )
        };
        if status != KERN_SUCCESS {
            return Err(format!(
                "the buffer's mapping at {at:#x} is not to be read: mach_vm_region answered {status}"
            ));
        }
        if address > at || size == 0 {
            return Err(format!("the buffer's bytes at {at:#x} are not mapped"));
        }
        let protection = info.protection;
        if protection & VM_PROT_WRITE == 0 {
            return Err(format!(
                "the buffer's mapping at {address:#x} is not writable (protection {protection:#x})"
            ));
        }
        at = address.saturating_add(size);
    }
    Ok(())
}

/// Writes `source` into `target`'s IOSurface by `transfer` (module docs):
/// [`DeviceWrite::Declined`], with why, where the host lends none or a
/// check fails, and nothing then reaches CoreVideo or VideoToolbox.
pub(crate) fn write(
    source: &Source<'_>,
    target: &GuestTarget<'_>,
    transfer: &dyn Transfer,
) -> Result<DeviceWrite, BackendError> {
    let lent = match target.memory().io_surface() {
        Ok(lent) => lent,
        Err(error) => {
            return declined(format!(
                "the host lends no IOSurface of the buffer: {error}"
            ));
        }
    };
    write_into(source, target, lent, &IoSurface::of(lent), transfer)
}

/// [`write()`] into `destination`, `lent`'s surface as the writer reads it.
pub(crate) fn write_into(
    source: &Source<'_>,
    target: &GuestTarget<'_>,
    lent: LentIoSurface<'_>,
    destination: &dyn Destination,
    transfer: &dyn Transfer,
) -> Result<DeviceWrite, BackendError> {
    // The first check, before the surface is read at all.
    if !lent.writable() {
        return declined(
            "the host would not write the buffer itself: one the guest made read-only for its \
             clients, or a protected one",
        );
    }
    let format = target.pixel_format();
    if format != source.format {
        return declined(format!(
            "a {} buffer for {} frames, which a transfer would convert",
            format.fourcc(),
            source.format.fourcc()
        ));
    }
    let size = (target.width(), target.height());
    let written = target.written_size(source.width, source.height);
    let image = source.image.size();
    let image = (
        u32::try_from(image.0).unwrap_or(0),
        u32::try_from(image.1).unwrap_or(0),
    );
    if written != size || image != size {
        return declined(format!(
            "a frame of {}x{} of a picture of {}x{} into a buffer of {}x{}: the transfer writes \
             whole buffers alone, and the CPU copies what fits",
            source.width, source.height, image.0, image.1, size.0, size.1
        ));
    }
    // A transfer of an odd size writes the even part and repeats it into
    // the last column or row, where the CPU copies the frame's own samples
    // (measured on the M3 Max, 2026-10-25: 641x361 `420v`, the luma's
    // column 640 and row 360 as column 639 and row 359, and the chroma's
    // last pair and row the same).
    if size.0 % 2 == 1 || size.1 % 2 == 1 {
        return declined(format!(
            "a frame of {}x{}, an odd size, whose last column or row a transfer writes as the one before it",
            size.0, size.1
        ));
    }
    // The second check: a protected surface has no base address.
    let base = destination.base_address();
    if base.is_null() {
        return declined("the buffer's IOSurface has no base address: a protected surface");
    }
    let length = target.memory().length();
    if destination.alloc_size() < length {
        return declined(format!(
            "the buffer's IOSurface holds {} bytes, fewer than its memory's {length}",
            destination.alloc_size()
        ));
    }
    // The third: every page from the base address to the allocation's end
    // writable, as the transfer may write any of the planes' rows.
    if let Err(why) = writable_mapping(base.addr(), length) {
        return declined(why);
    }
    if destination.pixel_format() != format.fourcc().0 {
        return declined(format!(
            "the buffer's IOSurface is {}, its layout {}",
            fourcc_text(destination.pixel_format()),
            format.fourcc()
        ));
    }
    let as_usize = |side: u32| usize::try_from(side).unwrap_or(usize::MAX);
    if destination.size() != (as_usize(size.0), as_usize(size.1)) {
        let (width, height) = destination.size();
        return declined(format!(
            "the buffer's IOSurface is {width}x{height}, its layout {}x{}",
            size.0, size.1
        ));
    }
    let planes = destination.planes();
    let expected = target.planes();
    let same = planes.len() == expected.len()
        && planes.iter().zip(&expected).all(|(plane, expected)| {
            plane.offset == expected.offset
                && plane.bytes_per_row == expected.bytes_per_row
                && plane.height >= as_usize(expected.rows)
                && (plane.bytes_per_row.checked_mul(plane.height))
                    .and_then(|bytes| bytes.checked_add(plane.offset))
                    .is_some_and(|end| end <= length)
        });
    if !same {
        return declined(format!(
            "the buffer's IOSurface's planes {planes:?} are not its layout's {expected:?}"
        ));
    }
    let mut buffer = match destination.pixel_buffer() {
        Ok(buffer) => buffer,
        Err(why) => return declined(format!("no pixel buffer over the buffer: {why}")),
    };
    let geometry = buffer.planes();
    let same = buffer.format() == format.fourcc().0
        && buffer.size() == destination.size()
        && geometry.len() == planes.len()
        && geometry.iter().zip(&planes).all(|(geometry, plane)| {
            geometry.bytes_per_row == plane.bytes_per_row && geometry.height == plane.height
        });
    if !same {
        return declined(format!(
            "the pixel buffer over the buffer is laid out as {geometry:?}, its IOSurface as \
             {planes:?}"
        ));
    }
    let transferred = transfer.transfer(source.image, &mut buffer);
    // The pixel buffer's reference of the surface goes before the write
    // returns (module docs).
    drop(buffer);
    match transferred {
        Ok(()) => Ok(DeviceWrite::Written),
        Err(status) => declined(format!(
            "VTPixelTransferSessionTransferImage into the buffer answered {status}"
        )),
    }
}

#[cfg(test)]
mod tests;
