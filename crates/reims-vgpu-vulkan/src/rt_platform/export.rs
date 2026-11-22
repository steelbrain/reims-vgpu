//! The rt platform's buffers whose memory another API imports: IOSurface device caches that the
//! host's video writes its pictures into on its own GPU work ([`exportable`]), through CUDA's
//! import of an opaque fd (`GpuDevice::new_exportable_buffer_of_length`, NVDEC's) or VA-API's of
//! a dma-buf (`GpuDevice::new_dma_buf_buffer_of_length_padding_unobserved`).
//!
//! **One block a buffer.** Each is a buffer of its own exported block (rt's
//! `Device::new_exported_block`): one dedicated allocation of Private memory, exported as the
//! kind of handle its make names, which the other API imports with the allocation's size, as
//! dedicated, and the buffer's place in it ([`ExportedMemory`]). Its bytes are zeros, as
//! `newBufferWithLength:options:`'s are (rt's GPU fill ahead of the next command buffer). Its
//! memory is linear: VA-API imports it as `DRM_FORMAT_MOD_LINEAR`. Where the caller names a byte
//! of it to lie on a multiple of 256 bytes of its memory (`aligned_byte`, which the IOSurface
//! device caches name for their first plane), the buffer starts that far into the block that the
//! byte does: VA-API imports a cache's planes with each plane's first byte and rows on such
//! multiples, and CUDA maps its import from the allocation's first byte and finds the buffer at
//! its place in the mapping (reims-vgpu-video's `ImportedBuffer::import`).
//!
//! **Each export a new fd.** rt gives one fd of the memory as it makes the block, which the
//! buffer keeps; each `Buffer::export_memory` (an opaque fd's) or `Buffer::export_dma_buf` (a
//! dma-buf's) answers a duplicate of it (`dup`), the caller's own, as an import takes it.
//!
//! **The acquire.** `Buffer::external_write_done` is rt's `Block::external_writes_done`: the
//! block's acquire from the other API's queue family, recorded in rt's upload ahead of every
//! command buffer committed after it, which then reads what that API wrote. No release goes back
//! to the other API: it touches nothing of the block between the acquire and its next write of
//! a whole picture (rt's contract). NVDEC's landing meets it, writing each picture's rows whole;
//! so does VA-API's, which decodes into a cache only a picture no later picture predicts from,
//! and writes every other one there by its video processing (reims-vgpu's `graphics::vaapi`), so
//! that no decode reads a reference from the block.
//!
//! **The release.** An opaque fd's buffer that the device writes for CUDA to read, NVENC's input
//! (reims-vgpu's `graphics::nvidia::encode::fill`), is handed over by rt's
//! `BlitCommandEncoder::release_to_external` (`BlitEncoder::release_to_external`), recorded after
//! the blit's copies into it, before CUDA reads it once the command buffer completed. No acquire
//! comes back: nothing reads the buffer through the device before its next copy, which writes it
//! anew (rt's contract).
//!
//! **Deferred exports** ([`deferred`], `GpuDevice::new_deferred_exportable_buffer_of_length`). A
//! deferred buffer (`super::buffer`, "Deferred memory") may name the export its memory takes:
//! its first use then makes the memory as [`exportable`] makes a buffer's, a buffer of its own
//! exported block, on the thread of that use, and an export of the memory
//! (`Buffer::export_memory`, `Buffer::export_dma_buf`) is such a use. Until then it holds no
//! block, exports nothing and takes no acquire. Its zeros are rt's GPU fill ahead of the next
//! command buffer, as any new buffer's: another API's write that comes before the platform's
//! next work is the caller's to order after zeros of its own. Where rt refuses the export as the
//! memory is made, counted as below, the memory is made as any deferred buffer's, and the buffer
//! exports nothing.
//!
//! **Refusals.** Where rt makes no exported block (a device that exports no opaque fd, a dma-buf
//! without the foreign queue family, a limit) or no buffer in it, the call answers `None` and the
//! caller makes the buffer as any other: counted ([`EXPORT_REFUSED_CODE`]), the process's first
//! with a line naming rt's error. rt copies no frame for it. Options of a storage mode other than
//! Private are declined uncounted (rt exports Private memory alone), the caller's make counting
//! what Metal refuses of them.

use std::os::fd::OwnedFd;
use std::sync::atomic::{AtomicBool, Ordering};

use metal2vulkan_runtime as rt;

use crate::core::platform::resource::ExportedMemory;

use super::RtObj;
use super::buffer::{self, RtBuffer};
use super::device::RtDevice;

/// The log target of the exports' lines.
const TARGET: &str = "reims_vgpu_vulkan::rt_platform";

/// The code of an exportable buffer rt makes none of (the module's docs, "Refusals"), counted per
/// refusal.
pub const EXPORT_REFUSED_CODE: &str = "rt-refused-export";

/// The bytes a byte the caller names lies on a multiple of in the exported memory.
const ALIGNED_TO: u64 = 256;

/// What an exportable buffer keeps of its export (the module's docs).
pub(super) struct Exported {
    /// The exported block, whose acquire the buffer records.
    block: rt::Block,
    /// rt's fd of the block's memory, which each export duplicates.
    fd: OwnedFd,
    /// The kind of `fd`.
    handle: rt::ExportHandle,
    /// The allocation's bytes, which an import names.
    size: u64,
    /// The buffer's first byte in the allocation.
    offset: u64,
}

/// Whether the process refused an exportable buffer yet.
static REFUSED_SAID: AtomicBool = AtomicBool::new(false);

/// Counts rt's refusal `why` of an exportable buffer for `sel` ([`EXPORT_REFUSED_CODE`]); the
/// process's first writes its line.
fn refused(sel: &'static str, why: &dyn std::fmt::Display) {
    crate::answer::count_kind(EXPORT_REFUSED_CODE);
    if !REFUSED_SAID.swap(true, Ordering::Relaxed) {
        log::warn!(
            target: TARGET,
            kind:% = EXPORT_REFUSED_CODE;
            "{sel}: metal2vulkan-runtime makes no exportable buffer ({why}); the buffer is made as \
             any other, and the host's video lands its pictures another way \
             ({EXPORT_REFUSED_CODE}), counted from here"
        );
    }
}

/// The offset in a block at which a buffer starts whose byte `aligned_byte` lies on a multiple
/// of [`ALIGNED_TO`] bytes of the block; 0 where none is named.
fn start_for(aligned_byte: Option<u64>) -> u64 {
    aligned_byte.map_or(0, |byte| (ALIGNED_TO - byte % ALIGNED_TO) % ALIGNED_TO)
}

/// A buffer of zeros of `length` bytes and options `raw`, for `sel`, whose memory another API
/// imports through `handle`'s kind of handle (the module's docs), `aligned_byte` of it, where
/// given, on a multiple of 256 bytes of its memory. `None` where rt makes none, counted, or the
/// options are not Private ones, and the caller makes the buffer as any other.
pub(super) fn exportable(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    handle: rt::ExportHandle,
    length: u64,
    raw: u64,
    aligned_byte: Option<u64>,
) -> Option<RtObj<RtBuffer>> {
    let answered = buffer::private_options(raw)?;
    let (made, exported) = memory(device, sel, handle, length, aligned_byte)?;
    Some(buffer::with_export(device, made, answered, exported))
}

/// [`exportable`]'s buffer whose memory rt makes at its first use, as a deferred buffer's (the
/// module's docs, "Deferred exports"); `None` where the options are not Private ones, and the
/// caller makes the buffer as any other.
pub(super) fn deferred(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    handle: rt::ExportHandle,
    length: u64,
    raw: u64,
    aligned_byte: Option<u64>,
) -> Option<RtObj<RtBuffer>> {
    buffer::private_options(raw)?;
    let export = DeferredExport {
        sel,
        handle,
        aligned_byte,
    };
    buffer::deferred_exportable(device, sel, length, raw, export)
}

/// How a deferred buffer's memory is exported as it is made ([`deferred`]): what [`memory`] takes
/// beside its length.
pub(super) struct DeferredExport {
    /// The make's selector, which a refusal names.
    pub(super) sel: &'static str,
    /// The kind of handle the memory is exported as.
    pub(super) handle: rt::ExportHandle,
    /// The byte of the buffer that lies on a multiple of [`ALIGNED_TO`] bytes of its memory.
    pub(super) aligned_byte: Option<u64>,
}

impl DeferredExport {
    /// The memory, as [`memory`] makes it, of a buffer of `length` bytes.
    pub(super) fn memory(
        &self,
        device: &RtObj<RtDevice>,
        length: u64,
    ) -> Option<(rt::Buffer, Exported)> {
        memory(device, self.sel, self.handle, length, self.aligned_byte)
    }

    /// The buffer of `length` bytes and options `raw` it names, made at once as [`exportable`]
    /// makes it.
    pub(super) fn exportable(
        &self,
        device: &RtObj<RtDevice>,
        length: u64,
        raw: u64,
    ) -> Option<RtObj<RtBuffer>> {
        exportable(
            device,
            self.sel,
            self.handle,
            length,
            raw,
            self.aligned_byte,
        )
    }
}

/// rt's buffer of `length` bytes of zeros in its own exported block of `handle`'s kind, for
/// `sel`, `aligned_byte` of it, where given, on a multiple of 256 bytes of its memory, with what
/// it keeps of the export (the module's docs); `None` where rt makes none, counted.
fn memory(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    handle: rt::ExportHandle,
    length: u64,
    aligned_byte: Option<u64>,
) -> Option<(rt::Buffer, Exported)> {
    let offset = start_for(aligned_byte);
    let Some(span) = offset.checked_add(length) else {
        refused(sel, &format!("a buffer of {length} bytes at {offset}"));
        return None;
    };
    let start = rt::host_time();
    let made = device.rt().new_exported_block(handle, span);
    super::diag::exported_block_made(rt::host_time().saturating_sub(start), span);
    let (block, export) = match made {
        Ok(made) => made,
        Err(error) => {
            refused(sel, &error);
            return None;
        }
    };
    let made = match device.rt().new_buffer(block.id, offset, length, Some(0)) {
        Ok(made) => made,
        Err(error) => {
            refused(sel, &error);
            return None;
        }
    };
    let exported = Exported {
        block,
        fd: export.handle,
        handle,
        size: export.size,
        offset,
    };
    Some((made, exported))
}

impl Exported {
    /// The memory as a new fd of `handle`'s kind, with what an import names, for a buffer of
    /// `length` bytes; `Err` says why not: an export of another kind, or a failed `dup`.
    pub(super) fn export(
        &self,
        handle: rt::ExportHandle,
        length: u64,
    ) -> Result<ExportedMemory, String> {
        if self.handle != handle {
            return Err(format!(
                "the buffer's memory is exported as {:?}, not {handle:?}",
                self.handle
            ));
        }
        let fd =
            (self.fd.try_clone()).map_err(|error| format!("a new fd of the export: {error}"))?;
        Ok(ExportedMemory {
            fd,
            allocation_size: self.size,
            offset: self.offset,
            length,
            dedicated: true,
        })
    }

    /// rt's acquire of what the other API wrote into the block (the module's docs, "The
    /// acquire"); `Err` says why rt refused it.
    pub(super) fn acquire(&self) -> Result<(), String> {
        (self.block.external_writes_done()).map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::platform::Handle as _;
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::resource::{Buffer as _, GpuResource as _};
    use crate::core::raise::OnFault;
    use crate::rt_platform::RtPlatform;

    /// A byte the caller names lies on a multiple of 256 bytes: the buffer starts as far before
    /// the next as the byte lies past the last.
    #[test]
    fn an_aligned_byte_starts_the_buffer_before_a_multiple_of_256() {
        assert_eq!(start_for(None), 0);
        assert_eq!(start_for(Some(0)), 0);
        assert_eq!(start_for(Some(32)), 224);
        assert_eq!(start_for(Some(256)), 0);
        assert_eq!(start_for(Some(0x2d0)), 0x30);
    }

    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// An exported buffer of `length` bytes is kept by a lost device's walk by its block's number
    /// and its offset in it, rt's, as rt's fault line names the block, with no range or block of
    /// its own (`super::super::fault::buffer_parts`).
    fn kept_by_its_block(buffer: &RtObj<RtBuffer>, length: u64) {
        let made = buffer.made_rt().expect("made by rt");
        let block = (made.block().get(), made.offset());
        assert_eq!(buffer.exported_block(), Some(block));
        let parts = super::super::fault::buffer_parts(3, 7, false, buffer);
        assert!(parts.object.is_none() && parts.block.is_none());
        let held = parts.exported.expect("kept by its block");
        assert_eq!(((held.block, held.offset), held.len), (block, length));
        assert_eq!((held.context, held.id, held.storage), (3, 7, 2));
    }

    /// Exportable buffers on the GPU the test holds, where its device exports an opaque fd: a
    /// Private buffer of the asked length at a GPU address, its byte 32 on a multiple of 256
    /// bytes of its memory where asked; each export a new fd of the dedicated allocation, which
    /// holds the buffer, none as a dma-buf; the acquire taken. No address of its block goes out
    /// at the make or to the fault history, which takes none of it (rt's count of the exported
    /// blocks addressed stays), until its `gpuAddress` is read, which counts its block. A plain
    /// buffer exports nothing and takes no acquire, and its fault address is its `gpuAddress`;
    /// options of other storage are declined uncounted. Where the device exports no opaque fd, the
    /// refusal is counted and the make answers `None`.
    #[test]
    fn exportable_buffers_on_vulkan() {
        use std::os::fd::AsRawFd as _;
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let length = 3840 * 2160 * 3 / 2 + 32;
        let refusals = counted(EXPORT_REFUSED_CODE);
        let addressed = || device.rt().counters().metrics().exported_blocks_addressed;
        let before = addressed();
        let made = device.new_exportable_buffer_of_length(on, length, 0x20, Some(32));
        let Some(buffer) = made.expect("never a raise") else {
            assert_eq!(counted(EXPORT_REFUSED_CODE), refusals + 1);
            println!("the device exports no opaque fd: refused, counted");
            return;
        };
        assert_eq!(counted(EXPORT_REFUSED_CODE), refusals);
        assert_eq!((buffer.length(), buffer.storage_mode()), (length, 2));
        assert_eq!(buffer.resource_options(), 0x220);
        assert_eq!(buffer.fault_address(), None);
        kept_by_its_block(&buffer, length);
        assert_eq!(addressed() - before, 0, "no address of the block out");
        assert_ne!(buffer.gpu_address(), 0);
        assert_eq!(addressed() - before, 1, "the buffer's address read");
        assert_eq!(buffer.fault_address(), None);
        let first = buffer.export_memory().expect("an export");
        let second = buffer.export_memory().expect("another");
        assert_ne!(first.fd.as_raw_fd(), second.fd.as_raw_fd());
        assert_eq!((first.offset + 32) % 256, 0);
        assert_eq!(first.offset, 224);
        assert!(first.dedicated && first.length == length);
        assert!(first.allocation_size >= first.offset + length);
        assert_eq!(
            (second.offset, second.allocation_size),
            (first.offset, first.allocation_size)
        );
        assert!(buffer.export_dma_buf().is_err());
        assert_eq!(buffer.external_write_done(), Ok(()));
        let unaligned = device.new_exportable_buffer_of_length(on, 4096, 0x20, None);
        let unaligned = unaligned
            .expect("never a raise")
            .expect("an exportable buffer");
        assert_eq!(unaligned.export_memory().expect("an export").offset, 0);
        assert_eq!(unaligned.fault_address(), None);
        assert_eq!(addressed() - before, 1, "no address of its block out");

        let plain = (device.new_buffer_of_length(on, 4096, 0x20))
            .expect("never a raise")
            .expect("a Private buffer");
        assert!(plain.export_memory().is_err());
        assert!(plain.external_write_done().is_err());
        assert_eq!(plain.fault_address(), Some(plain.gpu_address()));
        assert_eq!(plain.exported_block(), None);
        let shared = device.new_exportable_buffer_of_length(on, 4096, 0, None);
        assert!(shared.expect("never a raise").is_none());
        assert_eq!(counted(EXPORT_REFUSED_CODE), refusals);
        assert_ne!(buffer.id(), unaligned.id());
    }

    /// Dma-buf buffers on the GPU the test holds, where its device exports a dma-buf: a Private
    /// buffer of the asked length at a GPU address, as far into its block as puts its byte 32 on
    /// a multiple of 256 bytes of the memory; each export a new dma-buf of the dedicated
    /// allocation, none as an opaque fd; the acquire taken; no address of its block out at the
    /// make or to the fault history, until its `gpuAddress` is read, as an opaque fd's. With no
    /// byte named, the buffer lies at its block's first byte. Options of other storage are
    /// declined uncounted. Where the device exports no dma-buf, the refusal is counted and the
    /// make answers `None`.
    #[test]
    fn dma_buf_buffers_on_vulkan() {
        use std::os::fd::AsRawFd as _;
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let length = 3840 * 2160 * 3 / 2 + 32;
        let refusals = counted(EXPORT_REFUSED_CODE);
        let addressed = || device.rt().counters().metrics().exported_blocks_addressed;
        let before = addressed();
        let made =
            device.new_dma_buf_buffer_of_length_padding_unobserved(on, length, 0x20, Some(32));
        let Some(buffer) = made.expect("never a raise") else {
            assert_eq!(counted(EXPORT_REFUSED_CODE), refusals + 1);
            println!("the device exports no dma-buf: refused, counted");
            return;
        };
        assert_eq!(counted(EXPORT_REFUSED_CODE), refusals);
        assert_eq!((buffer.length(), buffer.storage_mode()), (length, 2));
        assert_eq!(buffer.fault_address(), None);
        kept_by_its_block(&buffer, length);
        assert_eq!(addressed() - before, 0, "no address of the block out");
        assert_ne!(buffer.gpu_address(), 0);
        assert_eq!(addressed() - before, 1, "the buffer's address read");
        assert_eq!(buffer.fault_address(), None);
        let first = buffer.export_dma_buf().expect("an export");
        let second = buffer.export_dma_buf().expect("another");
        assert_ne!(first.fd.as_raw_fd(), second.fd.as_raw_fd());
        assert_eq!(first.offset, 224);
        assert_eq!((first.offset + 32) % 256, 0);
        assert!(first.dedicated && first.length == length);
        assert!(first.allocation_size >= first.offset + length);
        assert_eq!(
            (second.offset, second.allocation_size),
            (first.offset, first.allocation_size)
        );
        assert!(buffer.export_memory().is_err());
        assert_eq!(buffer.external_write_done(), Ok(()));
        let unaligned =
            device.new_dma_buf_buffer_of_length_padding_unobserved(on, 4096, 0x20, None);
        let unaligned = unaligned.expect("never a raise").expect("a dma-buf buffer");
        assert_eq!(unaligned.export_dma_buf().expect("an export").offset, 0);
        assert_eq!(unaligned.fault_address(), None);
        assert_eq!(addressed() - before, 1, "no address of its block out");
        let shared = device.new_dma_buf_buffer_of_length_padding_unobserved(on, 4096, 0, None);
        assert!(shared.expect("never a raise").is_none());
        assert_eq!(counted(EXPORT_REFUSED_CODE), refusals);
    }

    /// Deferred exportable buffers on the GPU the test holds, for each handle (the module's
    /// docs, "Deferred exports"): a Private buffer of the asked length that holds no memory, at
    /// GPU address 0, until its first use; an export is such a use, and makes the memory exported
    /// as the eager make's is, its byte 32 on a multiple of 256 bytes, each export a new handle
    /// of the dedicated allocation, the other handle's export refused, the acquire taken, no
    /// address of its block out at the make or to the fault history; a use that is no export
    /// makes it so too. Where the device exports no such handle, the memory is
    /// made as any deferred buffer's, the refusal counted, and the export is refused. Options of
    /// other storage make the eager make's buffer, or none, as it does.
    #[test]
    fn deferred_exportable_buffers_on_vulkan() {
        use std::os::fd::AsRawFd as _;

        use crate::core::platform::resource::{MemoryFor, SurfaceHandle};
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let length = 3840 * 2160 * 3 / 2 + 32;
        for handle in [SurfaceHandle::OpaqueFd, SurfaceHandle::DmaBuf] {
            let export = |buffer: &RtObj<RtBuffer>, as_handle: SurfaceHandle| match as_handle {
                SurfaceHandle::OpaqueFd => buffer.export_memory(),
                SurfaceHandle::DmaBuf => buffer.export_dma_buf(),
            };
            let other = match handle {
                SurfaceHandle::OpaqueFd => SurfaceHandle::DmaBuf,
                SurfaceHandle::DmaBuf => SurfaceHandle::OpaqueFd,
            };
            let refusals = counted(EXPORT_REFUSED_CODE);
            let made =
                device.new_deferred_exportable_buffer_of_length(on, length, 0x20, Some(32), handle);
            let buffer = (made.expect("never a raise")).expect("a deferred buffer");
            assert_eq!((buffer.length(), buffer.storage_mode()), (length, 2));
            assert_eq!(buffer.resource_options(), 0x220);
            assert!(
                !buffer.memory_made(),
                "{handle:?}: no memory before its first use"
            );
            assert_eq!(buffer.gpu_address(), 0);
            assert_eq!(
                buffer.fault_address(),
                None,
                "{handle:?}: no address unmade"
            );
            assert_eq!(
                counted(EXPORT_REFUSED_CODE),
                refusals,
                "{handle:?}: none asked yet"
            );
            let Ok(first) = export(&buffer, handle) else {
                assert_eq!(counted(EXPORT_REFUSED_CODE), refusals + 1);
                assert!(
                    buffer.memory_made(),
                    "{handle:?}: made as any deferred buffer's"
                );
                assert_ne!(buffer.gpu_address(), 0);
                assert!(buffer.external_write_done().is_err());
                println!("the device exports no {handle:?}: made unexported, refused, counted");
                continue;
            };
            assert!(buffer.memory_made(), "{handle:?}: the export made it");
            assert_eq!(buffer.fault_address(), None, "{handle:?}: exported");
            assert_ne!(buffer.gpu_address(), 0);
            assert_eq!(counted(EXPORT_REFUSED_CODE), refusals);
            let second = export(&buffer, handle).expect("another");
            assert_ne!(first.fd.as_raw_fd(), second.fd.as_raw_fd());
            assert_eq!(first.offset, 224);
            assert!(first.dedicated && first.length == length);
            assert!(first.allocation_size >= first.offset + length);
            assert_eq!(
                (second.offset, second.allocation_size),
                (first.offset, first.allocation_size)
            );
            assert!(export(&buffer, other).is_err());
            assert_eq!(buffer.external_write_done(), Ok(()));

            let used =
                device.new_deferred_exportable_buffer_of_length(on, 4096, 0x20, None, handle);
            let used = (used.expect("never a raise")).expect("a deferred buffer");
            assert!(used.make_memory(MemoryFor::Command));
            assert!(used.memory_made());
            let exported = export(&used, handle).expect("an export of memory made by a use");
            assert!(exported.dedicated && exported.offset == 0);

            let shared = device.new_deferred_exportable_buffer_of_length(on, 4096, 0, None, handle);
            let eager = match handle {
                SurfaceHandle::OpaqueFd => {
                    device.new_exportable_buffer_of_length(on, 4096, 0, None)
                }
                SurfaceHandle::DmaBuf => {
                    device.new_dma_buf_buffer_of_length_padding_unobserved(on, 4096, 0, None)
                }
            };
            assert_eq!(
                (shared.expect("never a raise").is_none()),
                (eager.expect("never a raise").is_none()),
                "{handle:?}: other storage as the eager make"
            );
            assert_eq!(counted(EXPORT_REFUSED_CODE), refusals);
        }
    }
}
