//! `Managed` storage (docs/graphics-kext/design.md, Decided 4): a buffer's or a texture's CPU
//! copy is memory of its process's the guest maps, and the range messages tell the host what the
//! CPU wrote. A buffer's is a region block, a range of a heap over a block or over a pages block's
//! guest RAM, or the contents of a `Shared` buffer the guest names (`MANAGED_BUFFER_CREATE`); a
//! texture's is the contents of a `Shared` buffer the guest names. Those lie wherever the buffer's
//! contents lie in the guest's memory ([`GuestBytes`]), so a guest that took
//! `RGFX_FEATURE_MANAGED_TEXTURE_SLABS` and `RGFX_FEATURE_MANAGED_BUFFER_SLABS` lays them in its
//! slabs, its own pages among them.
//!
//! - **A buffer**, on any GPU: the platform's no-copy `Managed` buffer over its CPU copy's bytes,
//!   at the guest's address. The platform keeps the copies Metal keeps: one where the GPU shares
//!   the CPU's memory; on any other GPU a GPU copy in the GPU's memory, with the guest's bytes its
//!   CPU copy, copied in whole at the make (but for the bytes a `MANAGED_BUFFER_CREATE` says read
//!   zero, which start as zeros the GPU writes), a range at each `didModifyRange:` by the next
//!   submission, and copied out whole at each `synchronizeResource:` where the blit encoder runs
//!   it (the Vulkan platform's runtime, whose `managed_copy` module says so; where the GPU's
//!   memory has no room for the copy, the buffer lies in its bytes, as a `Shared` one does, which
//!   the runtime announces). `BUFFER_DID_MODIFY_RANGE` is the platform's `didModifyRange:` of the
//!   range, and a blit's synchronize of the buffer the platform's own.
//! - **A texture on a GPU that shares the CPU's memory** (the platform's answer,
//!   [`GraphicsDecoder::set_unified_memory`]): one copy, over the Shared buffer of its CPU copy,
//!   where a texture over a buffer can have its shape ([`in_place_shape`]). Its range messages are
//!   checked and order nothing more: the platform sees the bytes at each command buffer's commit.
//! - **A texture on any other GPU**: two copies, as Metal keeps a discrete GPU's `Managed`
//!   resources. The object the command buffers use is the platform's own `Managed` texture, and
//!   its CPU copy is the guest's bytes, through the `Shared` buffer over them. A
//!   `TEXTURE_REPLACE_REGION` is the platform's `replaceRegion:` from the bytes, which is done
//!   before the call returns, and `synchronizeTexture:slice:level:` and `synchronizeResource:`
//!   copy the GPU's copy to the bytes where the blit encoder runs them
//!   ([`GraphicsDecoder::synchronize_managed`]), so the CPU sees it once the command buffer
//!   completes.
//!
//! A range message, of a buffer or of a texture, is checked on ring 0 where it comes, and carried
//! out once the process's command buffers answered `CB_SUBMITTED` before it are committed, with no
//! wait on ring 0 ([`GraphicsDecoder::after_submitted`]); every command buffer of the process
//! begun after it, on any queue, is committed after it, so it is ordered before every command
//! buffer committed after it, as Metal promises. The bytes a command buffer wrote in its own
//! `Managed` chunks of `set*Bytes:` data ride its stream instead (`RGFX_OP_UPLOADS_WRITTEN`, where
//! the guest took `RGFX_FEATURE_UPLOADS_WRITTEN`), each the platform's `didModifyRange:` from the
//! buffer's start as the buffer begins, on its thread, before its commit
//! ([`GraphicsDecoder::uploads_written`]): no command buffer committed before it uses a chunk it
//! names, as the bundle gives a chunk back to its pool only once the buffer that used it has
//! completed, so the buffer waits for no message on ring 0 for them.
//!
//! A shape the platform cannot make in place takes the two copies on any GPU: more than one level,
//! a cube, a 3D texture, a block-compressed format. No creator of a texture over a buffer makes
//! one, Metal's included, so the GPU's copy cannot be the CPU copy's linear bytes. The CPU copy
//! stays there: without it an upload (`replaceRegion:`) would be a staging blit the call waits for,
//! as a `Shared` texture's is, 340 to 420 us a call on the lane where a `Managed` one's is 2 to 4
//! (docs/graphics-kext/dgpu-oracle.md Q17). On a GPU that shares the CPU's memory the guest's CPU
//! reads of such a texture (`getBytes:`) take its GPU copy, copied on the device: there Metal's
//! CPU reads of a `Managed` texture see the GPU's writes with no synchronize (the Intel UHD 630
//! and the M3, dgpu-oracle.md Q2). The bundle knows the shape alone ([`in_place_shape`], its
//! `rgfxManagedLiesOverCopy`), so a texture of that shape whose texture over its buffer the
//! platform refuses, two copies too, reads its CPU copy, as the last synchronize left it; that is
//! logged.
//!
//! **Where the platform refuses a buffer, or a texture's GPU copy**, the create is refused, as
//! Metal's answers nil. Metal answers no `Managed` resource nil for want of the GPU's memory alone:
//! on the AMD Radeon Pro 5300M (4 GiB, recommendedMaxWorkingSetSize 4080 MiB) 12 GiB of `Managed`
//! buffers, of 1 MiB and of 64 MiB, were all made, used in one command buffer, which completed,
//! and read back as the GPU wrote them (measured, shmem `-use -check`). The platform places a
//! texture past the GPU's memory as Metal does (the Vulkan platform's runtime in system memory,
//! past a full device-local heap), and lays a buffer whose GPU copy has no room in its bytes, so
//! its refusal comes only where that memory is gone too.
//!
//! **At the copy's address.** Where the guest's driver took `RGFX_FEATURE_MANAGED_AT_COPY`
//! ([`GraphicsDecoder::set_managed_at_copy`]), which a host offers where the GPU shares the CPU's
//! memory and the platform keeps pins only in its known blocks, a `MANAGED_BUFFER_CREATE`'s
//! address is its CPU copy's first byte's: a `Managed` buffer there is its bytes, a range of the
//! block its copy lies in. A buffer the platform makes at any other address, its copy's bytes in no
//! block it knows, is kept where the platform made it, as the same create makes it where the guest
//! did not take the bit: Metal makes the buffer, so the create is not refused, and the guest's
//! pointers to it hold the address the guest posted. Each is counted ([`MANAGED_AT_COPY_UNKEPT`]),
//! the first logged.
//!
//! **After a recovery of the device** (the platform made its lost GPU again,
//! [`GpuDevice::recoveries`]), a texture of two copies keeps its CPU copy, the guest's bytes, but
//! the platform carries its GPU copy with its contents lost. Metal documents no loss of a
//! resource's contents at its GPU's recovery (a hang ends the command buffer it was in, the SDK's
//! `MTLCommandBuffer.h`), and the CPU copy holds what every upload brought, so the first command
//! buffer of the process begun past the recovery fills each such texture's GPU copy again from its
//! CPU copy, whole, before it begins ([`GraphicsDecoder::refill_managed`]). A buffer's two copies
//! are the platform's, and its runtime carries them with the host copy's bytes.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use reims_vgpu_graphics::managed::LinearLayout;
use reims_vgpu_graphics::ops::RgfxUploadWritten;
use reims_vgpu_graphics::region::Region;
use reims_vgpu_graphics::wire::{
    RGFX_MANAGED_BUFFER_ZERO_OFFSET, RGFX_MANAGED_TEXTURE_LINEAR, RgfxBufferDidModifyRange,
    RgfxBufferManagedAtOffset, RgfxBufferOverHostMemory, RgfxManagedBufferCreate,
    RgfxManagedBufferZero, RgfxManagedTextureCreate, RgfxTextureReplaceRegion,
};
use reims_vgpu_vulkan::core::host::HostPtr;
use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder};
use reims_vgpu_vulkan::core::platform::device::GpuDevice;
use reims_vgpu_vulkan::core::platform::resource::{Buffer, Texture};
use reims_vgpu_vulkan::core::platform::types::{MtlOrigin, MtlRegion, MtlSize, NsRange};
use reims_vgpu_vulkan::core::platform::values::{BufferDesc, OptionSet};
use reims_vgpu_vulkan::core::platform::{Handle, HostRange, Platform};
use reims_vgpu_vulkan::core::raise::{OnFault, RaiseResult};

use super::surface_pages::TEARDOWN;
use super::tables::{Entry, GuestBytes};
use super::worker::Worker;
use super::{
    DecodeError, Drain, GraphicsDecoder, ObjHandle, op_fault, read, read_at, refusing, texture_desc,
};

/// The degradation code of a `MANAGED_BUFFER_CREATE` at its CPU copy's address, where the guest
/// took `MANAGED_AT_COPY`, that the platform made at another (the module doc, "At the copy's
/// address"): kept there, as the create makes it without the bit, and counted per create.
pub(crate) const MANAGED_AT_COPY_UNKEPT: &str = "managed-at-copy-unkept";

/// Whether a create counted as [`MANAGED_AT_COPY_UNKEPT`] was logged: its line is logged the
/// first time alone, the rest counted (docs/conventions.md).
static AT_COPY_LOGGED: AtomicBool = AtomicBool::new(false);

/// `MTLStorageModeManaged`.
pub(super) const STORAGE_MODE_MANAGED: u64 = 1;
/// Where `MTLResourceOptions` hold the storage mode.
const STORAGE_MODE_SHIFT: u32 = 4;
/// `MTLResourceStorageModeMask`.
const STORAGE_MODE_MASK: u64 = 0xf << STORAGE_MODE_SHIFT;
/// `MTLTextureType2D`.
const TEXTURE_TYPE_2D: u32 = 2;
/// `MTLTextureType2DArray`.
const TEXTURE_TYPE_2D_ARRAY: u32 = 3;
// The linear SPI lays a 2D array's slices at their rows' bytes rounded up to 128, which a CPU
// copy's slices are where every row is a multiple of 128 ([`in_place_shape`]).
const _: () = assert!(reims_vgpu_graphics::wire::RGFX_MANAGED_ROW_ALIGN.is_multiple_of(128));
/// `MTLTextureType3D`.
const TEXTURE_TYPE_3D: u64 = 7;

/// The storage mode `options` (`MTLResourceOptions`) name.
pub(super) fn storage_mode(options: u64) -> u64 {
    (options & STORAGE_MODE_MASK) >> STORAGE_MODE_SHIFT
}

/// `options` with `Shared` storage: those of a CPU copy's buffer, and of a texture over one.
fn shared(options: u64) -> u64 {
    options & !STORAGE_MODE_MASK
}

/// A `Managed` buffer's or texture's CPU copy: the `length` bytes at `at` in the region, or in a
/// heap over a pages block's guest RAM.
pub(crate) struct ManagedCopy<P: Platform> {
    /// Where the CPU copy starts: from `ram` where it has one, else in the region.
    pub(super) at: u64,
    /// The CPU copy's bytes: the buffer's length, or the texture layout's size.
    pub(super) length: u64,
    /// A texture's `Shared` buffer over its CPU copy, where the texture is a copy of its own
    /// (two copies); `None` for a buffer, and for a texture over its CPU copy.
    pub(super) cpu: Option<P::Buffer>,
    /// A texture's layout of the block; `None` for a buffer.
    pub(super) layout: Option<LinearLayout>,
    /// The first byte of the heap over a pages block's guest RAM the CPU copy lies in
    /// (`HEAP_OVER_PAGES`); `None` where it lies in the region.
    pub(super) ram: Option<HostPtr>,
    /// A texture of two copies: the device's recoveries ([`GpuDevice::recoveries`]) as of which
    /// its GPU copy holds what the uploads from its CPU copy brought, its make's, then each
    /// refill's ([`GraphicsDecoder::refill_managed`]). Unread for any other.
    pub(super) gpu_as_of: AtomicU64,
}

/// What a no-copy `Managed` buffer over the guest's bytes holds until the platform runs its
/// deallocator, so the bytes stay mapped until then ([`GraphicsDecoder::buffer_over_bytes`]).
enum BytesHold<P: Platform> {
    /// The `Shared` buffer whose contents the bytes are (`MANAGED_BUFFER_CREATE`), which holds
    /// what keeps them mapped: its block of the region, its heap, or a pages block's view.
    Buffer(CopyBufferHold<P>),
    /// What keeps the bytes of the heap they lie in mapped, where it lies over any but the
    /// region's ([`Entry::ram_kept`], `BUFFER_MANAGED_AT_OFFSET`).
    Kept {
        _kept: Option<super::tables::BytesKeep>,
    },
}

/// A `MANAGED_BUFFER_CREATE`'s hold of its CPU copy's `Shared` buffer ([`BytesHold::Buffer`]),
/// which the deallocator of the no-copy buffer over its bytes keeps: the buffer is let go of on
/// the surface worker ([`TEARDOWN`]), never where the hold goes.
///
/// The Vulkan platform runs the deallocator from rt's release of its import of the bytes, on rt's
/// block-maker thread, or at the device's teardown on the thread that ends it, and that release
/// makes no call into rt and drops no handle of rt's (metal2vulkan-runtime's `import.rs`). The
/// CPU copy's buffer is such a handle, perhaps its device's last. So the hold hands it over: the
/// release ends once the job is queued, a lock of the worker's alone taken and no wait, and a
/// thread of the worker's, outside every callback of rt's, lets go of it after the release, where
/// rt's teardown may run if it was the device's last handle. The hold keeps the worker, so the
/// hand-over works after the decoder went too. The bytes stay mapped the hop longer, never
/// shorter: rt reads none of them once it ran the release.
///
/// A release at the device's teardown never holds one: the buffer keeps the device alive, the
/// platform's handle of it and rt's own, and rt tears a device down only once every handle of it
/// and of every object made from it is gone, so a hold still in a release keeps its device from
/// teardown until the release has run and the worker has let go of the buffer.
///
/// Where the platform refuses the no-copy buffer, it drops the deallocator unrun before the make
/// returns, on the decoder's thread (core's `NoCopyDeallocator`): the hold goes then as any hold
/// goes, the buffer on the surface worker, so a refused create keeps no handle of the
/// platform's. The guest sees nothing of it: its CPU copy's object holds the buffer.
struct CopyBufferHold<P: Platform> {
    /// The CPU copy's guest object, which names it where its release is refused.
    obj_id: u64,
    /// Taken as it goes.
    buffer: Option<P::Buffer>,
    worker: Arc<Worker>,
}

impl<P: Platform> Drop for CopyBufferHold<P> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.worker.drop_later(TEARDOWN, self.obj_id, buffer);
        }
    }
}

/// A two-copy `Managed` texture whose GPU copy a recovery lost, to fill again from its CPU copy
/// ([`GraphicsDecoder::refill_managed`]).
struct Refill<P: Platform> {
    obj_id: u64,
    texture: P::Texture,
    /// The CPU copy's buffer, which keeps a pages block's view mapped where its bytes lie in one.
    _cpu: P::Buffer,
    layout: LinearLayout,
    /// Where the CPU copy starts ([`ManagedCopy::at`], [`ManagedCopy::ram`]).
    at: u64,
    ram: Option<HostPtr>,
}

impl<P: Platform> Refill<P> {
    /// Every level and slice of the GPU copy replaced from the CPU copy's bytes, in the guest's
    /// RAM or in `region`, as a `TEXTURE_REPLACE_REGION` of each whole would.
    fn upload(&self, region: &Region) -> RaiseResult<()> {
        let three_d = self.texture.texture_type() == TEXTURE_TYPE_3D;
        for (level, l) in (0u64..).zip(self.layout.levels()) {
            for slice in 0..self.layout.slices() {
                let Some((offset, span)) = self.layout.region(level, slice, [0; 3], l.size) else {
                    continue;
                };
                let from = self.at + offset;
                let first = match self.ram {
                    Some(first) => first.wrapping_add(from),
                    None => HostPtr::from_ptr(region.base().wrapping_add(from as usize)),
                };
                // SAFETY: the bytes lie inside the layout, which lies inside the CPU copy's bytes
                // (checked at the create): in the guest's RAM mapping, which stays mapped as long
                // as the device, in a view the CPU copy's buffer keeps mapped, or in the region,
                // which the decoder keeps mapped.
                let bytes = unsafe { HostRange::new(first, span) };
                let whole = MtlRegion {
                    origin: MtlOrigin { x: 0, y: 0, z: 0 },
                    size: MtlSize {
                        width: l.size[0],
                        height: l.size[1],
                        depth: l.size[2],
                    },
                };
                let bytes_per_image = if three_d { l.bytes_per_image } else { 0 };
                (self.texture).replace_region(
                    whole,
                    level,
                    slice,
                    bytes,
                    l.bytes_per_row,
                    bytes_per_image,
                )?;
            }
        }
        Ok(())
    }
}

/// Whether a texture over a buffer can have the shape `create` names: a one-level, one-sample 2D
/// texture or 2D array of a pixel format the guest calls linear, the shapes the platform's two
/// creators over a buffer make (`newTextureWithDescriptor:offset:bytesPerRow:` a 2D texture, the
/// linear SPI a 2D array too). Neither makes a texture of more than one level, a cube, a 3D
/// texture or one of a block-compressed format, as Metal's do not.
fn in_place_shape(create: &RgfxManagedTextureCreate) -> bool {
    let d = &create.desc;
    let shaped = match d.texture_type {
        TEXTURE_TYPE_2D => d.array_length <= 1,
        TEXTURE_TYPE_2D_ARRAY => true,
        _ => false,
    };
    create.flags & RGFX_MANAGED_TEXTURE_LINEAR != 0
        && shaped
        && d.mipmap_level_count == 1
        && d.sample_count <= 1
}

/// Logs, the first time, an `UPLOADS_WRITTEN` record the host drops: `what` it does wrong.
fn log_upload_dropped(w: &RgfxUploadWritten, what: &str) {
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        log::warn!(
            target: "reims_vgpu::graphics",
            "an UPLOADS_WRITTEN record of object {:#x}, {} bytes, {what}: dropped, and its \
             command buffer runs (logged once)",
            w.buffer_obj_id,
            w.length
        );
    }
}

/// Logs, the first time, a `Managed` texture of a shape that lies over its CPU copy on a GPU that
/// shares the CPU's memory, whose texture over its CPU copy's buffer the platform refused: it is
/// two copies, and the guest, which knows the shape alone, reads its CPU copy, which shows the
/// GPU's writes only after a `synchronizeTexture:` (the module doc).
fn log_two_copies_on_unified_memory(create: &RgfxManagedTextureCreate) {
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        let d = &create.desc;
        log::warn!(
            target: "reims_vgpu::graphics",
            "Managed texture {:#x} ({}x{}, {} slices, format {}): the platform made no texture \
             over its CPU copy on unified memory, so it is two copies, and the guest's getBytes: \
             of it reads the GPU's writes only after a synchronizeTexture: (logged once)",
            create.obj_id,
            d.width,
            d.height,
            d.array_length,
            d.pixel_format
        );
    }
}

impl<P: Platform> GraphicsDecoder<P> {
    /// Whether a `Managed` texture is one copy, over its CPU copy, where it can be: the
    /// platform's answer of whether the device's GPU shares the CPU's memory
    /// (`hasUnifiedMemory`). The decoder starts with no, the two copies, which hold Metal's
    /// contract on any GPU. A `Managed` buffer's copies are the platform's on either.
    pub(crate) fn set_unified_memory(&mut self, unified: bool) {
        (self.shared.unified_memory).store(unified, std::sync::atomic::Ordering::Release);
        // A GPU of its own memory caches every surface the kext may (`surface_cache`): its
        // transfer queue is made with the first command queue.
        self.surface_caches.set_early(!unified);
    }

    /// Whether the device's GPU shares the CPU's memory ([`Self::set_unified_memory`]).
    fn unified(&self) -> bool {
        (self.shared.unified_memory).load(std::sync::atomic::Ordering::Acquire)
    }

    /// The CPU copy of the `Managed` buffer or texture `obj_id` names, where the current context
    /// may name it.
    pub(super) fn managed_copy(&self, obj_id: u64) -> Option<&ManagedCopy<P>> {
        self.entry(obj_id).and_then(|e| e.managed.as_ref())
    }

    /// A no-copy buffer of `options` over the `length` bytes at region offset `at`, which the
    /// caller checked lie in a block of the sender's, at the guest's `pin` where it has one (a
    /// BUFFER_CREATE's alike): a `Shared` buffer, or a `Managed` one over its CPU copy. It keeps
    /// the region mapped until the platform frees it.
    pub(super) fn buffer_over_block(
        &self,
        device: &P::Device,
        at: u64,
        length: u64,
        options: u64,
        pin: Option<u64>,
    ) -> RaiseResult<Option<P::Buffer>> {
        let keep = std::sync::Arc::clone(&self.region);
        // SAFETY: the span lies inside the region (the caller checked), which stays mapped until
        // its last reference goes; the deallocator holds one until the platform frees the buffer.
        let bytes = unsafe {
            HostRange::new(
                HostPtr::from_ptr(self.region.base().add(at as usize)),
                length,
            )
        };
        device.new_buffer(BufferDesc {
            length,
            options: vec![OptionSet::ResourceOptions(options)],
            pinned_gpu_address: pin,
            contents: Some(bytes),
            no_copy: true,
            deallocator: Some(Box::new(move |_, _| drop(keep))),
            zero_from: None,
        })
    }

    /// `BUFFER_OVER_HOST_MEMORY` of a `Managed` buffer, whose block the caller checked is the
    /// sender's: a no-copy `Managed` buffer over the block at the guest's address, on any GPU,
    /// whose copies are the platform's (the module doc). Refused where the platform refuses it.
    pub(super) fn managed_buffer(
        &self,
        device: &P::Device,
        m: &RgfxBufferOverHostMemory,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let options = u64::from(m.options);
        let pin = (m.gpu_address != 0).then_some(m.gpu_address);
        let kind = super::unkept::Kind::managed(self.unified());
        let Some(buffer) = kind
            .making(|| self.buffer_over_block(device, m.region_offset, m.length, options, pin))?
        else {
            return Ok(Err(DecodeError::Refused));
        };
        let live = self.made_as(kind, pin, buffer.gpu_address());
        self.store_managed_buffer(m.obj_id, buffer, (m.region_offset, m.length), live);
        Ok(Ok(()))
    }

    /// `BUFFER_MANAGED_AT_OFFSET`: a no-copy `Managed` buffer at the guest's address over its CPU
    /// copy, the bytes at `offset` in the guest's heap over host memory, or over a pages block's
    /// guest RAM (`HEAP_OVER_PAGES`), whose copies are the platform's (the module doc). Refused
    /// where the platform refuses it. Offered where `Managed` is two copies
    /// (`RGFX_FEATURE_MANAGED_AT_OFFSET`), and refused on a GPU that shares the CPU's memory. A
    /// heap the sender does not hold, or one over no block of the region, is a bad reference; a
    /// CPU copy past the heap's memory is malformed.
    pub(super) fn managed_buffer_at_offset(
        &self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Result<(), DecodeError>> {
        let Some(m) = read::<RgfxBufferManagedAtOffset>(message) else {
            return Ok(Err(DecodeError::Malformed));
        };
        let options = u64::from(m.options);
        if storage_mode(options) != STORAGE_MODE_MANAGED || self.unified() {
            return Ok(Err(DecodeError::Refused));
        }
        let Some((heap_at, heap_length)) = self.span_of(m.heap_obj_id) else {
            return Ok(Err(DecodeError::BadReference));
        };
        // A heap over pages lies over its block, and its bytes are the guest's RAM.
        let ram = self.ram_of(m.heap_obj_id);
        let memory = ram.map_or(heap_length, |(_, length)| length);
        if m.length == 0
            || m.offset
                .checked_add(m.length)
                .is_none_or(|end| end > memory)
        {
            return Ok(Err(DecodeError::Malformed));
        }
        if let Some((first, _)) = ram {
            log::trace!(
                target: "reims_vgpu::graphics::surface_pages",
                "BUFFER_MANAGED_AT_OFFSET in the heap {:#x} over pages at {:#x}: object {:#x}, \
                 {:#x} bytes at {:#x}",
                m.heap_obj_id,
                first.addr(),
                m.obj_id,
                m.length,
                m.offset
            );
        }
        let (ram, at, span) = match ram {
            Some((first, _)) => (Some(first), m.offset, (heap_at, heap_length)),
            None => (None, heap_at + m.offset, (heap_at + m.offset, m.length)),
        };
        if !matches!(self.named(m.heap_obj_id), Some(ObjHandle::Heap(_))) {
            return Ok(Err(DecodeError::BadReference));
        }
        let pin = (m.gpu_address != 0).then_some(m.gpu_address);
        let bytes = GuestBytes {
            ram,
            at,
            length: m.length,
        };
        // The buffer holds what keeps the heap's bytes mapped, as the heap does: the heap may go
        // first.
        let keep = BytesHold::Kept {
            _kept: self.ram_kept_of(m.heap_obj_id),
        };
        let kind = super::unkept::Kind::managed(self.unified());
        let Some(buffer) = kind
            .making(|| self.buffer_over_bytes(device, bytes, m.length, options, pin, keep, None))?
        else {
            return Ok(Err(DecodeError::Refused));
        };
        let live = self.made_as(kind, pin, buffer.gpu_address());
        let copy = ManagedCopy {
            at,
            length: m.length,
            cpu: None,
            layout: None,
            ram,
            gpu_as_of: AtomicU64::new(0),
        };
        self.store_managed(m.obj_id, buffer, span, copy, live);
        Ok(Ok(()))
    }

    /// `MANAGED_BUFFER_CREATE`: a `Managed` buffer at the guest's address whose CPU copy is the
    /// first `length` bytes of a `Shared` buffer over the guest's memory the sender holds,
    /// wherever they lie ([`GuestBytes`]): over a block, placed in a heap over one or over a pages
    /// block's guest RAM, or over a pages block's view. On any GPU a no-copy `Managed` buffer over
    /// those bytes, whose copies are the platform's (the module doc), which keeps the CPU copy's
    /// buffer, and so a view its bytes lie in, until the platform frees it; refused where the
    /// platform refuses it. The buffer lies over what its CPU copy lies over. A CPU copy that is
    /// no such buffer (a `Managed` one, one of the GPU's own memory, a refused one), or is shorter
    /// than the buffer, is a bad reference; a buffer of no bytes is malformed, and one of options
    /// other than `Managed`, or whose CPU copy lies over pages its process maps read-only, which
    /// the copies would write ([`Self::copy_admitted`]), is refused.
    ///
    /// Where the message carries an [`RgfxManagedBufferZero`] from a byte before the buffer's
    /// length, the guest says the CPU copy reads zero from there, and the platform is told so
    /// (`BufferDesc::zero_from`): a GPU copy of the platform's starts with zeros its GPU writes
    /// there, rather than with those bytes copied in, as Metal's GPU copy of a buffer made with no
    /// bytes holds. A message without it, or from the length or past it, tells the platform
    /// nothing, and the GPU copy starts as the CPU copy's bytes.
    pub(super) fn managed_buffer_create(
        &self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Result<(), DecodeError>> {
        let Some(m) = read::<RgfxManagedBufferCreate>(message) else {
            return Ok(Err(DecodeError::Malformed));
        };
        let options = u64::from(m.options);
        if storage_mode(options) != STORAGE_MODE_MANAGED {
            return Ok(Err(DecodeError::Refused));
        }
        if m.length == 0 {
            return Ok(Err(DecodeError::Malformed));
        }
        let zero_from = read_at::<RgfxManagedBufferZero>(message, RGFX_MANAGED_BUFFER_ZERO_OFFSET)
            .map(|zero| zero.zero_from)
            .filter(|&from| from < m.length);
        let (Some(ObjHandle::Buffer(cpu)), Some(bytes)) = (
            self.named(m.cpu_buffer_obj_id),
            self.bytes_of(m.cpu_buffer_obj_id),
        ) else {
            return Ok(Err(DecodeError::BadReference));
        };
        if m.length > bytes.length {
            return Ok(Err(DecodeError::BadReference));
        }
        let span = self.span_of(m.cpu_buffer_obj_id);
        if !self.copy_admitted("MANAGED_BUFFER_CREATE", span) {
            return Ok(Err(DecodeError::Refused));
        }
        let cpu = cpu.clone_ref();
        let pin = (m.gpu_address != 0).then_some(m.gpu_address);
        let kind = super::unkept::Kind::managed(self.unified());
        let keep = BytesHold::Buffer(CopyBufferHold {
            obj_id: m.cpu_buffer_obj_id,
            buffer: Some(cpu),
            worker: Arc::clone(&self.shared.surface_worker),
        });
        let Some(buffer) = kind.making(|| {
            self.buffer_over_bytes(device, bytes, m.length, options, pin, keep, zero_from)
        })?
        else {
            return Ok(Err(DecodeError::Refused));
        };
        let made = buffer.gpu_address();
        // The at-copy create the platform made elsewhere: kept there, as the create without the
        // bit makes it, since Metal makes the buffer (the module doc, "At the copy's address").
        if (self.shared.managed_at_copy).load(Ordering::Acquire) && pin != Some(made) {
            reims_vgpu_vulkan::answer::count_kind(MANAGED_AT_COPY_UNKEPT);
            if !AT_COPY_LOGGED.swap(true, Ordering::Relaxed) {
                log::warn!(
                    target: "reims_vgpu::graphics", kind:% = MANAGED_AT_COPY_UNKEPT;
                    "the Managed buffer {:#x} at its CPU copy {:#x}'s address {:#x}: the \
                     platform made it at {made:#x}, where it is kept, as without \
                     MANAGED_AT_COPY ({MANAGED_AT_COPY_UNKEPT}; the rest are counted)",
                    m.obj_id, m.cpu_buffer_obj_id, m.gpu_address
                );
            }
        }
        let live = self.made_as(kind, pin, made);
        let copy = ManagedCopy {
            at: bytes.at,
            length: m.length,
            cpu: None,
            layout: None,
            ram: bytes.ram,
            gpu_as_of: AtomicU64::new(0),
        };
        let mut entry = Entry::of(ObjHandle::Buffer(buffer))
            .over(span)
            .counted(live);
        entry.managed = Some(copy);
        self.store(m.obj_id, entry);
        Ok(Ok(()))
    }

    /// A no-copy buffer of `options` over the first `length` bytes of the guest's `bytes`, at the
    /// guest's `pin`: a `Managed` buffer over its CPU copy, whose bytes from `zero_from`, where
    /// the guest said so, read zero. It keeps `keep` ([`BytesHold`]) until the platform runs its
    /// deallocator, once it let go of its import of the bytes, so the guest's RAM or a pages
    /// block's view they lie in stays mapped until then.
    #[expect(
        clippy::too_many_arguments,
        reason = "the buffer's descriptor, its guest bytes and what it keeps"
    )]
    fn buffer_over_bytes(
        &self,
        device: &P::Device,
        bytes: GuestBytes,
        length: u64,
        options: u64,
        pin: Option<u64>,
        keep: BytesHold<P>,
        zero_from: Option<u64>,
    ) -> RaiseResult<Option<P::Buffer>> {
        let region = Arc::clone(&self.region);
        let first = match bytes.ram {
            Some(first) => first.wrapping_add(bytes.at),
            None => HostPtr::from_ptr(region.base().wrapping_add(bytes.at as usize)),
        };
        // SAFETY: the bytes lie in the guest's memory (the caller checked `length` against
        // them): in the guest's RAM or a pages block's view, which `keep` keeps mapped, or in the
        // region, which `region` keeps mapped; the deallocator holds both until the platform runs
        // it.
        let range = unsafe { HostRange::new(first, length) };
        device.new_buffer(BufferDesc {
            length,
            options: vec![OptionSet::ResourceOptions(options)],
            pinned_gpu_address: pin,
            contents: Some(range),
            no_copy: true,
            deallocator: Some(Box::new(move |_, _| drop((keep, region)))),
            zero_from,
        })
    }

    /// Stores the `Managed` buffer `obj_id`, whose CPU copy is the `length` bytes at `at` in the
    /// region, counted as made while it lives where `live` is given.
    fn store_managed_buffer(
        &self,
        obj_id: u64,
        buffer: P::Buffer,
        (at, length): (u64, u64),
        live: Option<super::unkept::Live>,
    ) {
        let copy = ManagedCopy {
            at,
            length,
            cpu: None,
            layout: None,
            ram: None,
            gpu_as_of: AtomicU64::new(0),
        };
        self.store_managed(obj_id, buffer, (at, length), copy, live);
    }

    /// Stores the `Managed` buffer `obj_id`, lying over the region's `span`, and its CPU copy,
    /// counted as made while it lives where `live` is given.
    fn store_managed(
        &self,
        obj_id: u64,
        buffer: P::Buffer,
        span: (u64, u64),
        copy: ManagedCopy<P>,
        live: Option<super::unkept::Live>,
    ) {
        let mut entry = Entry::of(ObjHandle::Buffer(buffer))
            .over(Some(span))
            .counted(live);
        entry.managed = Some(copy);
        self.store(obj_id, entry);
    }

    /// `BUFFER_DID_MODIFY_RANGE`: the platform's `didModifyRange:` of the range, which brings it
    /// from the CPU copy to the GPU's where the buffer is two copies, after the context's buffers
    /// answered before it are committed ([`Self::after_submitted`], which `drain` is held
    /// through), counted among the app's ranges once carried out (`summary::managed`). A bad
    /// reference for an object that is no `Managed` buffer; malformed for a range past it: both
    /// found here.
    pub(super) fn did_modify_range(
        &self,
        message: &[u8],
        drain: Option<Arc<Drain>>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let Some(m) = read::<RgfxBufferDidModifyRange>(message) else {
            return Ok(Err(DecodeError::Malformed));
        };
        // One whose create the host refused holds nothing to bring over ([`super::refused`]).
        if let Some(ObjHandle::Refused(r)) = self.named(m.obj_id) {
            r.used(m.obj_id);
            return Ok(Ok(()));
        }
        let (Some(ObjHandle::Buffer(buffer)), Some(copy)) =
            (self.named(m.obj_id), self.managed_copy(m.obj_id))
        else {
            return Ok(Err(DecodeError::BadReference));
        };
        if copy.layout.is_some()
            || m.location
                .checked_add(m.length)
                .is_none_or(|end| end > copy.length)
        {
            return Ok(Err(DecodeError::Malformed));
        }
        let buffer = buffer.clone_ref();
        self.after_submitted("BUFFER_DID_MODIFY_RANGE", m.obj_id, drain, move || {
            buffer.did_modify_range(OnFault::Return, NsRange::new(m.location, m.length))?;
            super::super::summary::managed::range_carried_out(m.length);
            Ok(())
        });
        Ok(Ok(()))
    }

    /// A command buffer's `UPLOADS_WRITTEN` records, `written`, as it begins: the platform's
    /// `didModifyRange:` of each buffer's bytes from its start, which brings them to its GPU copy
    /// by the buffer's commit, at once, as the buffer's own chunks are used by no command buffer
    /// committed before it (the module doc). A buffer whose create the host refused holds
    /// nothing to bring over ([`super::refused`]). A record of an object that is no `Managed`
    /// buffer, or of bytes past one, is dropped (logged once), and so is one the platform raises
    /// at (logged), as a range message the host refuses or cannot carry out is, and the buffer
    /// runs, as it would behind such a message, the same whether or not it was answered at its
    /// doorbell before its begin. The records carried out are counted as the buffer's upload
    /// chunks (`summary::managed`).
    pub(super) fn uploads_written(&self, written: &[RgfxUploadWritten]) {
        let (mut chunks, mut bytes) = (0, 0);
        for w in written {
            if let Some(ObjHandle::Refused(r)) = self.named(w.buffer_obj_id) {
                r.used(w.buffer_obj_id);
                continue;
            }
            let (Some(ObjHandle::Buffer(buffer)), Some(copy)) = (
                self.named(w.buffer_obj_id),
                self.managed_copy(w.buffer_obj_id),
            ) else {
                log_upload_dropped(w, "names no Managed buffer");
                continue;
            };
            if copy.layout.is_some() || w.length > copy.length {
                log_upload_dropped(w, "reaches past its Managed buffer");
                continue;
            }
            if w.length == 0 {
                continue;
            }
            let (obj_id, range) = (w.buffer_obj_id, NsRange::new(0, w.length));
            match refusing(|| buffer.did_modify_range(OnFault::Return, range)) {
                Ok(()) => {
                    chunks += 1;
                    bytes += w.length;
                }
                Err(raise) => log::warn!(
                    target: "reims_vgpu::graphics", kind:% = raise;
                    "UPLOADS_WRITTEN of object {obj_id:#x} was not carried out: the platform \
                     raised ({raise:?})"
                ),
            }
        }
        super::super::summary::managed::uploads_carried_out(chunks, bytes);
    }

    /// Whether a `Managed` object `name` makes may take the CPU copy that lies over `span` (its
    /// buffer's block): not where that is a block of pages its process maps read-only, whose
    /// copies (a synchronize, a texture's writes over its CPU copy) would write them
    /// (`GraphicsDecoder::read_only_pages_admit`).
    fn copy_admitted(&self, name: &str, span: Option<(u64, u64)>) -> bool {
        span.is_none_or(|(block, _)| self.read_only_pages_admit(name, block, None))
    }

    /// `MANAGED_TEXTURE_CREATE`: over the CPU copy's buffer in place where the GPU shares the
    /// CPU's memory and the platform makes a texture over a buffer of the shape, else the
    /// platform's own `Managed` texture beside it, refused where the platform refuses that one.
    /// The CPU copy is the buffer's contents wherever they lie in the guest's memory
    /// ([`GuestBytes`]): over a block of the sender's, placed in a heap over one or over a pages
    /// block's guest RAM, or over a pages block's view. A bad reference for a CPU copy that is no
    /// such buffer (a `Managed` one, or one of the GPU's own memory), or is shorter than the
    /// layout; refused for one over pages its process maps read-only ([`Self::copy_admitted`]).
    pub(super) fn managed_texture(
        &self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Result<(), DecodeError>> {
        let Some(m) = read::<RgfxManagedTextureCreate>(message) else {
            return Ok(Err(DecodeError::Malformed));
        };
        // Read before the make, so a texture made as the device recovers counts as made before
        // it, and is filled again ([`Self::refill_managed`]).
        let made_at = device.recoveries();
        if !self.holds_resource_id(m.obj_id, m.resource_id) {
            return Ok(Err(DecodeError::BadResourceId));
        }
        if !super::described::<P>(device, m.desc.pixel_format.into()) {
            return Ok(Err(DecodeError::Refused));
        }
        let Some(layout) = LinearLayout::of(&m) else {
            return Ok(Err(DecodeError::Malformed));
        };
        let (Some(ObjHandle::Buffer(cpu)), Some(bytes)) = (
            self.named(m.cpu_buffer_obj_id),
            self.bytes_of(m.cpu_buffer_obj_id),
        ) else {
            return Ok(Err(DecodeError::BadReference));
        };
        if layout.size() > bytes.length {
            return Ok(Err(DecodeError::BadReference));
        }
        // The texture lies over what the buffer lies over, so that block is not given out again
        // while the texture lives.
        let span = self.span_of(m.cpu_buffer_obj_id);
        if !self.copy_admitted("MANAGED_TEXTURE_CREATE", span) {
            return Ok(Err(DecodeError::Refused));
        }
        let cpu = cpu.clone_ref();
        let desc = texture_desc(&m.desc, m.resource_id);
        let in_place = |this: &Self| -> RaiseResult<Option<P::Texture>> {
            let mut over = desc;
            over.resource_options = Some(shared(u64::from(m.desc.options)));
            let descriptor = this.platform.texture_descriptor(&over);
            let Some(level) = layout.levels().first() else {
                return Ok(None);
            };
            // A 2D array by the linear SPI, which lays each slice at its rows' bytes rounded up
            // to 128 (`RgfxTexture.mm`'s rgfxBufferSlicePitch, measured on the M3 Max; the
            // Vulkan platform's runtime alike): the layout's slices, whose rows are multiples of
            // 256 bytes.
            match m.desc.texture_type {
                TEXTURE_TYPE_2D_ARRAY => cpu.new_linear_texture(
                    &descriptor,
                    0,
                    level.bytes_per_row,
                    level.bytes_per_slice,
                ),
                _ => cpu.new_texture(OnFault::Return, &descriptor, 0, level.bytes_per_row),
            }
        };
        if self.unified() && in_place_shape(&m) {
            if let Some(texture) = in_place(self)? {
                self.count_texture_pick(m.gpu_address);
                let copy = (bytes, None, layout, made_at);
                return self.store_managed_texture(m.obj_id, texture, span, copy);
            }
            log_two_copies_on_unified_memory(&m);
        }
        let descriptor = self.platform.texture_descriptor(&desc);
        match device.new_texture(OnFault::Return, &descriptor)? {
            Some(texture) => {
                self.count_texture_pick(m.gpu_address);
                let copy = (bytes, Some(cpu), layout, made_at);
                self.store_managed_texture(m.obj_id, texture, span, copy)
            }
            None => Ok(Err(DecodeError::Refused)),
        }
    }

    /// Stores a `Managed` texture lying over `span`, and its CPU copy: the guest's bytes, the
    /// `Shared` buffer over them where the texture is a copy of its own, its layout, and the
    /// device's recoveries as it was made.
    fn store_managed_texture(
        &self,
        obj_id: u64,
        texture: P::Texture,
        span: Option<(u64, u64)>,
        (bytes, cpu, layout, made_at): (GuestBytes, Option<P::Buffer>, LinearLayout, u64),
    ) -> RaiseResult<Result<(), DecodeError>> {
        let mut entry = Entry::of(ObjHandle::Texture(texture)).over(span);
        entry.managed = Some(ManagedCopy {
            at: bytes.at,
            length: layout.size(),
            cpu,
            layout: Some(layout),
            ram: bytes.ram,
            gpu_as_of: AtomicU64::new(made_at),
        });
        self.store(obj_id, entry);
        Ok(Ok(()))
    }

    /// `TEXTURE_REPLACE_REGION`: the platform's `replaceRegion:` from the CPU copy's bytes, in the
    /// guest's RAM or the region, where the texture is a copy of its own, after the context's
    /// buffers answered before it are committed ([`Self::after_submitted`], which `drain` is held
    /// through). A bad reference for an object that is no `Managed` texture; malformed for a
    /// region the layout does not hold: both found here.
    pub(super) fn replace_region(
        &self,
        message: &[u8],
        drain: Option<Arc<Drain>>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let Some(m) = read::<RgfxTextureReplaceRegion>(message) else {
            return Ok(Err(DecodeError::Malformed));
        };
        // One whose create the host refused holds nothing to bring over ([`super::refused`]).
        if let Some(ObjHandle::Refused(r)) = self.named(m.obj_id) {
            r.used(m.obj_id);
            return Ok(Ok(()));
        }
        let (Some(ObjHandle::Texture(texture)), Some(copy)) =
            (self.named(m.obj_id), self.managed_copy(m.obj_id))
        else {
            return Ok(Err(DecodeError::BadReference));
        };
        let Some(layout) = &copy.layout else {
            return Ok(Err(DecodeError::BadReference));
        };
        let (level, slice) = (u64::from(m.level), u64::from(m.slice));
        let origin = m.origin.map(u64::from);
        let size = m.size.map(u64::from);
        let (Some((offset, span)), Some(l)) = (
            layout.region(level, slice, origin, size),
            layout.level(level),
        ) else {
            return Ok(Err(DecodeError::Malformed));
        };
        let Some(cpu) = &copy.cpu else {
            return Ok(Ok(()));
        };
        let three_d = texture.texture_type() == TEXTURE_TYPE_3D;
        let (texture, region) = (texture.clone_ref(), Arc::clone(&self.region));
        // The CPU copy's buffer keeps a pages block's view mapped, where its bytes lie in one.
        let keep = cpu.clone_ref();
        let (ram, from, bytes_per_row) = (copy.ram, copy.at + offset, l.bytes_per_row);
        let bytes_per_image = if three_d { l.bytes_per_image } else { 0 };
        self.after_submitted("TEXTURE_REPLACE_REGION", m.obj_id, drain, move || {
            let first = match ram {
                Some(first) => first.wrapping_add(from),
                None => HostPtr::from_ptr(region.base().wrapping_add(from as usize)),
            };
            // SAFETY: the bytes lie inside the layout, which lies inside the CPU copy's bytes
            // (checked at the create): in the guest's RAM mapping, which stays mapped as long as
            // the device, in a view `keep`'s buffer keeps mapped, or in the region, which
            // `region` keeps mapped.
            let bytes = unsafe { HostRange::new(first, span) };
            let _keep = &keep;
            texture.replace_region(
                MtlRegion {
                    origin: MtlOrigin {
                        x: origin[0],
                        y: origin[1],
                        z: origin[2],
                    },
                    size: MtlSize {
                        width: size[0],
                        height: size[1],
                        depth: size[2],
                    },
                },
                level,
                slice,
                bytes,
                bytes_per_row,
                bytes_per_image,
            )
        });
        Ok(Ok(()))
    }

    /// After a recovery of the device ([`GpuDevice::recoveries`]), as the first command buffer of
    /// `context` begun past it begins: each of the context's two-copy `Managed` textures made
    /// before the recovery has its GPU copy filled again from its CPU copy, every level and slice
    /// whole, before this call returns, as a `TEXTURE_REPLACE_REGION` of each would. The
    /// recovery carries such a texture with its contents lost (the Vulkan platform's runtime
    /// makes it again in new memory), where Metal documents no such loss. The CPU copy holds what
    /// every upload brought and what each synchronize copied back; a GPU write no synchronize
    /// copied back before the recovery is lost with it, as Metal promises the CPU a GPU write
    /// only after a synchronize. A texture over its CPU copy, on a GPU that shares the CPU's
    /// memory, lies in the guest's memory, which keeps its bytes.
    ///
    /// The context's lock of its count is held throughout, so no other command buffer of the
    /// context begins before the refill is done; those of the lost device failed with it. A
    /// command buffer the new device runs before the runtime announces the recovery (once the new
    /// device serves) may meet a texture before its refill. A texture the platform raises at is
    /// logged and left as the recovery left it. Nothing where the device has not recovered since
    /// the context last looked: a lock of the context's own, which no other context takes.
    pub(super) fn refill_managed(&self, device: &P::Device, context: u32) {
        let now = device.recoveries();
        if now == 0 {
            return;
        }
        let Some(state) = self.context_state(context) else {
            return;
        };
        let mut seen = state.managed_recoveries.lock();
        if *seen == now {
            return;
        }
        *seen = now;
        let mut refills = Vec::new();
        state.for_each_live(|obj_id, entry| {
            let (ObjHandle::Texture(texture), Some(copy)) = (&entry.handle, &entry.managed) else {
                return;
            };
            let (Some(cpu), Some(layout)) = (&copy.cpu, &copy.layout) else {
                return;
            };
            if copy.gpu_as_of.swap(now, Ordering::AcqRel) == now {
                return;
            }
            refills.push(Refill::<P> {
                obj_id,
                texture: texture.clone_ref(),
                _cpu: cpu.clone_ref(),
                layout: layout.clone(),
                at: copy.at,
                ram: copy.ram,
            });
        });
        if refills.is_empty() {
            return;
        }
        let bytes: u64 = refills.iter().map(|r| r.layout.size()).sum();
        let mut refused = 0usize;
        for refill in &refills {
            if let Err(raise) = refusing(|| refill.upload(&self.region)) {
                refused += 1;
                log::warn!(
                    target: "reims_vgpu::graphics", kind:% = raise;
                    "the device recovered: the GPU copy of Managed texture {:#x} was not filled \
                     again from its CPU copy, the platform raised ({raise:?})",
                    refill.obj_id
                );
            }
        }
        log::info!(
            target: "reims_vgpu::graphics",
            "the device recovered: context {context}'s {} Managed textures of two copies had \
             their GPU copies filled again from their CPU copies ({bytes} bytes), {refused} of \
             them refused",
            refills.len()
        );
        drop(seen);
    }

    /// `synchronizeResource:` (`which` `None`) or `synchronizeTexture:slice:level:` of the
    /// `Managed` texture `obj_id`, where it is a copy of its own: the GPU's copy copied to its CPU
    /// copy in `enc`, there. False, encoding nothing, for any other object, a `Managed` buffer
    /// among them, which the platform's own synchronize takes. A slice or level past the texture
    /// faults the exec.
    pub(super) fn synchronize_managed(
        &self,
        obj_id: u64,
        which: Option<(u64, u64)>,
        enc: &P::BlitEncoder,
        on: OnFault,
    ) -> RaiseResult<bool> {
        let Some(copy) = self.managed_copy(obj_id) else {
            return Ok(false);
        };
        let Some(cpu) = &copy.cpu else {
            return Ok(false);
        };
        let (Some(ObjHandle::Texture(texture)), Some(layout)) = (self.named(obj_id), &copy.layout)
        else {
            return Ok(false);
        };
        let three_d = texture.texture_type() == TEXTURE_TYPE_3D;
        let all = (0..layout.levels().len() as u64)
            .flat_map(|level| (0..layout.slices()).map(move |slice| (slice, level)));
        let picked: Vec<(u64, u64)> = match which {
            Some(one) => vec![one],
            None => all.collect(),
        };
        for (slice, level) in picked {
            let Some(l) = layout.level(level).filter(|_| slice < layout.slices()) else {
                return Err(op_fault(format_args!(
                    "synchronizeTexture: slice {slice} level {level} past the texture"
                )));
            };
            enc.encode(
                on,
                BlitCmd::CopyTextureToBuffer {
                    source_texture: texture,
                    source_slice: slice,
                    source_level: level,
                    source_origin: MtlOrigin { x: 0, y: 0, z: 0 },
                    source_size: MtlSize {
                        width: l.size[0],
                        height: l.size[1],
                        depth: l.size[2],
                    },
                    destination_buffer: cpu,
                    destination_offset: l.offset + slice * l.bytes_per_slice,
                    destination_bytes_per_row: l.bytes_per_row,
                    destination_bytes_per_image: if three_d { l.bytes_per_image } else { 0 },
                },
            )?;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_vgpu_graphics::ops::blit::{
        RGFX_BLIT_RESOURCE_BUFFER, RGFX_BLIT_RESOURCE_TEXTURE, RGFX_OP_BLIT_SYNCHRONIZE_RESOURCE,
        RGFX_OP_BLIT_SYNCHRONIZE_TEXTURE, RgfxBlitOpSynchronizeResource,
        RgfxBlitOpSynchronizeTexture,
    };
    use reims_vgpu_graphics::region::Region;
    use reims_vgpu_graphics::wire::{
        RGFX_CMD_BUFFER_AT_OFFSET, RGFX_CMD_BUFFER_DID_MODIFY_RANGE,
        RGFX_CMD_BUFFER_MANAGED_AT_OFFSET, RGFX_CMD_BUFFER_OVER_HOST_MEMORY,
        RGFX_CMD_BUFFER_OVER_PAGES, RGFX_CMD_HEAP_OVER_HOST_MEMORY, RGFX_CMD_MANAGED_BUFFER_CREATE,
        RGFX_CMD_MANAGED_TEXTURE_CREATE, RGFX_CMD_OBJECT_DESTROY, RGFX_CMD_TEXTURE_REPLACE_REGION,
        RGFX_CONTEXT_ALLOCATIONS_OFFSET, RGFX_CURSOR_SPAN, RGFX_STATUS_OK,
        RGFX_SURFACE_PAGES_PRIVATE, RgfxBufferAtOffset, RgfxBufferOverPages,
        RgfxHeapOverHostMemory, RgfxMsgHeader, RgfxObjectDestroy, RgfxTextureDescriptor,
    };
    use reims_vgpu_vulkan::core::platform::command::{CommandBuffer, CommandQueue};
    use reims_vgpu_vulkan::core::testkit::{Arg, FakePlatform, GuestRam, Step};
    use zerocopy::IntoBytes;

    use super::super::blit::decode_op;
    use super::super::surface_pages::tests as pages;
    use super::*;

    /// `Managed` storage, in `MTLResourceOptions`.
    const MANAGED: u32 = 1 << 4;
    /// `MTLPixelFormatRGBA8Unorm`.
    const RGBA8: u32 = 70;
    /// The processes' blocks start here.
    const BLOCKS: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;

    struct Rig {
        p: Arc<FakePlatform>,
        device: <FakePlatform as Platform>::Device,
        region: Arc<Region>,
        dec: GraphicsDecoder<FakePlatform>,
        enc: <FakePlatform as Platform>::BlitEncoder,
    }

    /// A decoder over a region with a MiB of blocks, `unified` as the platform answers, and an
    /// open blit encoder.
    fn rig(unified: bool) -> Rig {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let size = BLOCKS + 0x10_0000 + RGFX_CURSOR_SPAN;
        let region = Arc::new(Region::new(size as usize).unwrap());
        let mut dec = GraphicsDecoder::<FakePlatform>::new(Arc::clone(&p), Arc::clone(&region));
        dec.set_unified_memory(unified);
        dec.hold_resource_ids(0, 0);
        let queue = device.new_command_queue().unwrap();
        let cb = queue.command_buffer(OnFault::Return).unwrap().unwrap();
        let enc = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        Rig {
            p,
            device,
            region,
            dec,
            enc,
        }
    }

    /// The surface pages tests' decoder, two copies, over 8 host pages of guest RAM, with a pages
    /// block at [`pages::BLOCK`] of one run, pages 2 to 5, and heap 20 over it of the run's
    /// size, and an open blit encoder; the guest's RAM and the host's page.
    fn pages_rig() -> (Rig, GuestRam, u64) {
        let mut s = pages::rig(8);
        let page = s.page;
        let private = RGFX_SURFACE_PAGES_PRIVATE;
        assert_eq!(
            s.pages_flagged(pages::BLOCK, 0, &[(2, 4)], 4 * page, private),
            RGFX_STATUS_OK
        );
        assert_eq!(s.pages_heap(20, pages::BLOCK, 4 * page), Ok(()));
        let pages::Rig {
            p,
            device,
            ram,
            mut dec,
            ..
        } = s;
        dec.set_unified_memory(false);
        dec.hold_resource_ids(0, 0);
        let region = Arc::clone(&dec.region);
        let queue = device.new_command_queue().unwrap();
        let cb = queue.command_buffer(OnFault::Return).unwrap().unwrap();
        let enc = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        let r = Rig {
            p,
            device,
            region,
            dec,
            enc,
        };
        (r, ram, page)
    }

    impl Rig {
        fn send<T: IntoBytes + zerocopy::Immutable>(&mut self, m: &T) -> Result<(), DecodeError> {
            self.dec.decode(&self.device, m.as_bytes()).unwrap()
        }

        /// A `BUFFER_OVER_HOST_MEMORY` of `length` bytes at `at`, of `options`.
        fn buffer(
            &mut self,
            obj_id: u64,
            at: u64,
            length: u64,
            options: u32,
        ) -> Result<(), DecodeError> {
            self.send(&RgfxBufferOverHostMemory {
                header: header(RGFX_CMD_BUFFER_OVER_HOST_MEMORY),
                obj_id,
                gpu_address: 0x1_0000_0000 + obj_id * 0x10_0000,
                region_offset: at,
                length,
                options,
                ..RgfxBufferOverHostMemory::default()
            })
        }

        fn modified(&mut self, obj_id: u64, location: u64, length: u64) -> Result<(), DecodeError> {
            self.send(&RgfxBufferDidModifyRange {
                header: header(RGFX_CMD_BUFFER_DID_MODIFY_RANGE),
                obj_id,
                location,
                length,
            })
        }

        /// A `Managed` RGBA8 texture of `desc`'s shape over CPU-copy buffer `cpu`.
        fn texture(
            &mut self,
            obj_id: u64,
            cpu: u64,
            desc: RgfxTextureDescriptor,
        ) -> Result<(), DecodeError> {
            self.send(&RgfxManagedTextureCreate {
                header: header(RGFX_CMD_MANAGED_TEXTURE_CREATE),
                obj_id,
                gpu_address: 0,
                resource_id: obj_id,
                cpu_buffer_obj_id: cpu,
                bytes_per_block: 4,
                block_width: 1,
                block_height: 1,
                flags: RGFX_MANAGED_TEXTURE_LINEAR,
                desc,
            })
        }

        fn replaced(
            &mut self,
            obj_id: u64,
            level: u32,
            slice: u32,
            origin: [u32; 3],
            size: [u32; 3],
        ) -> Result<(), DecodeError> {
            self.send(&RgfxTextureReplaceRegion {
                header: header(RGFX_CMD_TEXTURE_REPLACE_REGION),
                obj_id,
                level,
                slice,
                origin,
                size,
            })
        }

        fn synchronize(&self, obj_id: u64, kind: u32) -> RaiseResult<bool> {
            let m = RgfxBlitOpSynchronizeResource {
                resource_obj_id: obj_id,
                resource_kind: kind,
                reserved_0c: 0,
            };
            decode_op(
                RGFX_OP_BLIT_SYNCHRONIZE_RESOURCE,
                m.as_bytes(),
                &self.dec,
                &self.enc,
                OnFault::Return,
            )
        }

        fn synchronize_texture(&self, obj_id: u64, slice: u64, level: u64) -> RaiseResult<bool> {
            let m = RgfxBlitOpSynchronizeTexture {
                texture_obj_id: obj_id,
                slice,
                level,
            };
            decode_op(
                RGFX_OP_BLIT_SYNCHRONIZE_TEXTURE,
                m.as_bytes(),
                &self.dec,
                &self.enc,
                OnFault::Return,
            )
        }

        fn obj(&self, obj_id: u64) -> Arg {
            match self.dec.named(obj_id) {
                Some(ObjHandle::Buffer(b)) => Arg::obj(b),
                Some(ObjHandle::Texture(t)) => Arg::obj(t),
                _ => panic!("no buffer or texture {obj_id:#x}"),
            }
        }

        fn cpu_of(&self, obj_id: u64) -> Arg {
            Arg::obj(self.dec.managed_copy(obj_id).unwrap().cpu.as_ref().unwrap())
        }
    }

    /// The creates among `sels`: a descriptor's setters left out.
    fn made(sels: Vec<&'static str>) -> Vec<&'static str> {
        sels.into_iter().filter(|s| s.starts_with("new")).collect()
    }

    fn header(r#type: u32) -> RgfxMsgHeader {
        RgfxMsgHeader {
            r#type,
            size: 0,
            seqno: 0,
        }
    }

    fn rgba(width: u32, height: u32, levels: u32) -> RgfxTextureDescriptor {
        RgfxTextureDescriptor {
            pixel_format: RGBA8,
            texture_type: 2,
            width,
            height,
            depth: 1,
            mipmap_level_count: levels,
            array_length: 1,
            sample_count: 1,
            usage: 1,
            options: MANAGED,
            ..RgfxTextureDescriptor::default()
        }
    }

    /// The descriptor of a no-copy buffer over the `length` bytes at region offset `at` of
    /// `options`, pinned at `pin` or not, as the platform's call log holds it.
    fn over_block(r: &Rig, at: u64, length: u64, options: u64, pin: Option<u64>) -> Arg {
        // SAFETY: the region is the rig's, mapped while it lives; the range is only formatted.
        let bytes =
            unsafe { HostRange::new(HostPtr::from_ptr(r.region.base().add(at as usize)), length) };
        Arg::val(&BufferDesc {
            length,
            options: vec![OptionSet::ResourceOptions(options)],
            pinned_gpu_address: pin,
            contents: Some(bytes),
            no_copy: true,
            deallocator: Some(Box::new(|_, _| {})),
            zero_from: None,
        })
    }

    /// A `Managed` buffer from `BUFFER_OVER_HOST_MEMORY` is one no-copy buffer of its options over
    /// its block at the guest's address, whether or not the GPU shares the CPU's memory: the
    /// platform keeps its copies. A range message is the platform's `didModifyRange:` of the
    /// range, copying nothing itself; one past the buffer, or of a buffer not `Managed`, is
    /// refused; a synchronize is the platform's. A destroy forgets its CPU copy, and the block's
    /// span goes with it.
    #[test]
    fn a_managed_buffer_lies_over_its_block_on_any_gpu() {
        for unified in [false, true] {
            let mut r = rig(unified);
            let at = BLOCKS + 0x1000;
            let first: Vec<u8> = (0..0x100u32).map(|i| i as u8).collect();
            assert!(r.region.write(at, &first));
            let mark = r.p.call_log().mark();
            assert_eq!(r.buffer(10, at, 0x100, MANAGED), Ok(()));
            let calls = r.p.call_log().since(mark);
            assert_eq!(calls.len(), 1, "unified {unified}");
            assert_eq!(calls[0].sel, "newBufferWithDescriptor:");
            let pin = 0x1_0000_0000 + 10 * 0x10_0000;
            assert_eq!(
                calls[0].args[0],
                over_block(&r, at, 0x100, MANAGED.into(), Some(pin)),
                "unified {unified}"
            );
            let buffer = r.dec.require_buffer(10).unwrap().clone_ref();
            assert_eq!(buffer.gpu_address(), pin);
            assert_eq!(
                buffer.contents().addr(),
                r.region.base() as usize + at as usize
            );
            assert!(r.dec.managed_copy(10).unwrap().cpu.is_none());

            let mark = r.p.call_log().mark();
            assert_eq!(r.modified(10, 32, 16), Ok(()));
            let calls = r.p.call_log().since(mark);
            assert_eq!(calls.len(), 1, "unified {unified}");
            assert_eq!(calls[0].sel, "didModifyRange:");
            assert_eq!(calls[0].args[0], Arg::val(&NsRange::new(32, 16)));
            assert_eq!(
                r.region.read(at, 0x100).unwrap(),
                first,
                "the block is left alone"
            );

            // A range past the buffer, and a Shared buffer's range, are refused.
            assert_eq!(r.modified(10, 0xf8, 16), Err(DecodeError::Malformed));
            assert_eq!(r.buffer(11, at + 0x1000, 0x100, 0), Ok(()));
            assert_eq!(r.modified(11, 0, 16), Err(DecodeError::BadReference));
            assert_eq!(r.modified(12, 0, 16), Err(DecodeError::BadReference));

            // A synchronize is the platform's, the Managed buffer's as a Shared one's.
            for obj_id in [10, 11] {
                let mark = r.p.call_log().mark();
                assert!(r.synchronize(obj_id, RGFX_BLIT_RESOURCE_BUFFER).unwrap());
                assert_eq!(r.p.call_log().sels_since(mark), ["synchronizeResource:"]);
            }

            assert_eq!(
                r.send(&RgfxObjectDestroy {
                    header: header(RGFX_CMD_OBJECT_DESTROY),
                    obj_id: 10,
                }),
                Ok(())
            );
            assert!(r.dec.managed_copy(10).is_none() && r.dec.object_count() == 1);
            assert!(!r.dec.span_in_use(at, 0x100));
        }
    }

    /// The summary's `Managed` part counts what the platform's `didModifyRange:` carried out by
    /// who named it: each app's range message with its bytes, apart from the chunks of a command
    /// buffer's `UPLOADS_WRITTEN`, their bytes and the buffers that named at least one. A range or
    /// record the host refuses, drops or skips, or whose call the platform raised at, counts in
    /// neither, and a take counts from none again.
    #[test]
    fn the_summary_counts_the_app_s_ranges_apart_from_the_upload_chunks() {
        use super::super::super::summary::managed::{self, Period};

        managed::count();
        let mut r = rig(false);
        assert_eq!(r.buffer(10, BLOCKS, 0x1000, MANAGED), Ok(()));
        assert_eq!(r.buffer(11, BLOCKS + 0x1000, 0x1000, MANAGED), Ok(()));
        assert_eq!(r.buffer(12, BLOCKS + 0x2000, 0x1000, 0), Ok(()));
        let raise = || Step::raise("NSInternalInconsistencyException", "scripted");

        // The app's: two carried out; one past its buffer, a Shared buffer's and no object's
        // refused; one the platform raises at.
        assert_eq!(r.modified(10, 0x40, 0x10), Ok(()));
        assert_eq!(r.modified(11, 0, 0x800), Ok(()));
        assert_eq!(r.modified(10, 0xff8, 0x10), Err(DecodeError::Malformed));
        assert_eq!(r.modified(12, 0, 0x10), Err(DecodeError::BadReference));
        assert_eq!(r.modified(13, 0, 0x10), Err(DecodeError::BadReference));
        r.p.script().push("didModifyRange:", raise());
        assert_eq!(r.modified(10, 0, 0x20), Ok(()));

        // A buffer's records: two carried out beside a Shared buffer's, one past its buffer and
        // one of no bytes, all dropped or skipped.
        let w = |buffer_obj_id, length| RgfxUploadWritten {
            buffer_obj_id,
            length,
        };
        r.dec.uploads_written(&[
            w(10, 0x110),
            w(12, 0x10),
            w(11, 0x1000),
            w(10, 0x1001),
            w(11, 0),
        ]);
        // Buffers that carried out no chunk: none named, one whose record is no object's, and one
        // whose one call the platform raises at.
        r.dec.uploads_written(&[]);
        r.dec.uploads_written(&[w(13, 0x10)]);
        r.p.script().push("didModifyRange:", raise());
        r.dec.uploads_written(&[w(10, 0x100)]);

        let sels = r.p.call_log().sels();
        let made = sels.iter().filter(|sel| **sel == "didModifyRange:").count();
        assert_eq!(
            made,
            2 + 1 + 2 + 1,
            "the platform's calls, those it raised at among them"
        );
        assert_eq!(
            managed::take(),
            Period {
                app_ranges: 2,
                app_range_bytes: 0x10 + 0x800,
                upload_buffers: 1,
                chunks: 2,
                chunk_bytes: 0x110 + 0x1000,
            }
        );
        assert_eq!(
            managed::take(),
            Period::default(),
            "counted from none again"
        );
    }

    /// On a platform that keeps a pin only inside a block it knows, each buffer and heap a guest's
    /// create makes is counted by its kind, made and live at once, and each pin the platform
    /// dropped once under the object's kind, so the kinds' counts sum to the pins dropped; a
    /// `Managed` buffer is its own kind where the GPU shares the CPU's memory. A texture at the
    /// guest's pick is counted apart. Where the platform keeps pins anywhere, nothing is counted.
    #[test]
    fn each_pin_not_kept_is_counted_under_its_kind() {
        use super::super::unkept::{KINDS, Kind, TEXTURE_PICKS};
        use reims_vgpu_graphics::wire::{
            RGFX_CMD_BUFFER_CREATE, RGFX_CMD_HEAP_CREATE, RgfxBufferCreate, RgfxHeapCreate,
        };
        use reims_vgpu_vulkan::answer::{counted_kinds, live_now};
        let count = |code: &str| {
            (counted_kinds().into_iter())
                .find(|&(c, _)| c == code)
                .map_or(0, |(_, n)| n)
        };
        let unpinned = || KINDS.map(|k| count(k.codes().unpinned));
        let made = || KINDS.map(|k| count(k.codes().made.made));
        let rise = |after: [u64; KINDS.len()], before: [u64; KINDS.len()], kind: Kind| {
            let at = KINDS.iter().position(|&k| k == kind).unwrap();
            after[at] - before[at]
        };

        // Pins dropped where the decoder was not told: nothing counted.
        let mut r = rig(false);
        r.p.world().drop_pins(true);
        let (u, m) = (unpinned(), made());
        assert_eq!(r.buffer(10, BLOCKS, 0x100, 0), Ok(()));
        assert_eq!(r.p.world().dropped_pins(), 1);
        assert_eq!((unpinned(), made()), (u, m));

        let mut r = rig(false);
        r.dec.set_pins_in_blocks();
        r.p.world().drop_pins(true);
        let (u, m, picks) = (unpinned(), made(), count(TEXTURE_PICKS));
        let buffers = live_now(Kind::Buffer.codes().made);
        // Over the region: a Shared buffer (the texture's CPU copy below) and a heap; a `Managed`
        // buffer over its block.
        assert_eq!(r.buffer(10, BLOCKS, 0x1000, 0), Ok(()));
        assert_eq!(r.buffer(11, BLOCKS + 0x1000, 0x100, MANAGED), Ok(()));
        assert_eq!(
            r.send(&RgfxHeapOverHostMemory {
                header: header(RGFX_CMD_HEAP_OVER_HOST_MEMORY),
                obj_id: 17,
                gpu_address: 0x4_0000_0000,
                region_offset: BLOCKS + 0x1_0000,
                size_bytes: 0x1_0000,
                options: 0,
                heap_type: 1,
            }),
            Ok(())
        );
        // Placed by the device: Shared, Private and unpinned buffers, Shared and Private heaps.
        let create = |obj_id, gpu_address, options| RgfxBufferCreate {
            header: header(RGFX_CMD_BUFFER_CREATE),
            obj_id,
            gpu_address,
            length: 0x100,
            options,
            reserved_2c: 0,
        };
        assert_eq!(r.send(&create(12, 0x2_0000_0000, 0)), Ok(()));
        assert_eq!(r.send(&create(13, 0x2_0010_0000, 2 << 4)), Ok(()));
        assert_eq!(r.send(&create(14, 0, 0)), Ok(()));
        let heap = |obj_id, gpu_address, options| RgfxHeapCreate {
            header: header(RGFX_CMD_HEAP_CREATE),
            obj_id,
            gpu_address,
            size_bytes: 0x1_0000,
            options,
            heap_type: 1,
        };
        assert_eq!(r.send(&heap(15, 0x3_0000_0000, 0)), Ok(()));
        assert_eq!(r.send(&heap(16, 0x3_0010_0000, 2 << 4)), Ok(()));
        // A `Managed` texture over buffer 10 at the guest's pick.
        assert_eq!(
            r.send(&RgfxManagedTextureCreate {
                header: header(RGFX_CMD_MANAGED_TEXTURE_CREATE),
                obj_id: 18,
                gpu_address: 0x5_0000_0000,
                resource_id: 18,
                cpu_buffer_obj_id: 10,
                bytes_per_block: 4,
                block_width: 1,
                block_height: 1,
                flags: RGFX_MANAGED_TEXTURE_LINEAR,
                desc: rgba(4, 4, 1),
            }),
            Ok(())
        );
        let (u2, m2) = (unpinned(), made());
        let unpinned_by = |kind| rise(u2, u, kind);
        let made_by = |kind| rise(m2, m, kind);
        assert_eq!(
            [
                Kind::Region,
                Kind::Managed,
                Kind::Buffer,
                Kind::Private,
                Kind::Heap
            ]
            .map(unpinned_by),
            [2, 1, 1, 2, 1]
        );
        assert_eq!(
            [
                Kind::Region,
                Kind::Managed,
                Kind::Buffer,
                Kind::Private,
                Kind::Heap
            ]
            .map(made_by),
            [2, 1, 2, 2, 1]
        );
        let total: u64 = KINDS.iter().map(|&k| unpinned_by(k)).sum();
        assert_eq!(total, r.p.world().dropped_pins());
        assert_eq!(total, 7);
        assert_eq!(count(TEXTURE_PICKS) - picks, 1);
        // Live while the guest holds them.
        assert_eq!(live_now(Kind::Buffer.codes().made), buffers + 2);
        assert_eq!(
            r.send(&RgfxObjectDestroy {
                header: header(RGFX_CMD_OBJECT_DESTROY),
                obj_id: 12,
            }),
            Ok(())
        );
        assert_eq!(live_now(Kind::Buffer.codes().made), buffers + 1);

        // On a GPU that shares the CPU's memory, a `Managed` buffer is a kind of its own.
        let mut r = rig(true);
        r.dec.set_pins_in_blocks();
        r.p.world().drop_pins(true);
        let u = unpinned();
        assert_eq!(r.buffer(10, BLOCKS, 0x100, MANAGED), Ok(()));
        let u2 = unpinned();
        assert_eq!(rise(u2, u, Kind::ManagedUnified), 1);
        assert_eq!(KINDS.iter().map(|&k| rise(u2, u, k)).sum::<u64>(), 1);
    }

    /// `BUFFER_MANAGED_AT_OFFSET` on a GPU of its own memory: a no-copy `Managed` buffer at the
    /// guest's address over the heap's bytes at the offset, its CPU copy, and no other buffer.
    /// Its range messages and synchronize are the platform's; the heap's block stays in use while
    /// it lives. A heap the sender lacks or one of device memory, a range past the heap's block,
    /// a buffer not `Managed`, and the message on unified memory are refused.
    #[test]
    fn a_managed_buffer_at_an_offset_lies_over_the_heap_s_bytes() {
        let mut r = rig(false);
        let at = BLOCKS + 0x4_0000;
        assert_eq!(
            r.send(&RgfxHeapOverHostMemory {
                header: header(RGFX_CMD_HEAP_OVER_HOST_MEMORY),
                obj_id: 30,
                gpu_address: 0x2_0000_0000,
                region_offset: at,
                size_bytes: 0x2_0000,
                options: 0,
                heap_type: 1,
            }),
            Ok(())
        );
        let first: Vec<u8> = (0..0x100u32).map(|i| (i * 3) as u8).collect();
        assert!(r.region.write(at + 0x3000, &first));
        let managed = |obj_id: u64, heap: u64, offset: u64, length: u64, options: u32| {
            RgfxBufferManagedAtOffset {
                header: header(RGFX_CMD_BUFFER_MANAGED_AT_OFFSET),
                obj_id,
                gpu_address: 0x1_0000_0000 + obj_id * 0x10_0000,
                heap_obj_id: heap,
                offset,
                length,
                options,
                reserved_3c: 0,
            }
        };
        let mark = r.p.call_log().mark();
        assert_eq!(r.send(&managed(10, 30, 0x3000, 0x100, MANAGED)), Ok(()));
        let calls = r.p.call_log().since(mark);
        assert_eq!(
            made(calls.iter().map(|c| c.sel).collect()),
            ["newBufferWithDescriptor:"]
        );
        let pin = 0x1_0000_0000 + 10 * 0x10_0000;
        assert_eq!(
            calls[0].args[0],
            over_block(&r, at + 0x3000, 0x100, MANAGED.into(), Some(pin))
        );
        let buffer = r.dec.require_buffer(10).unwrap().clone_ref();
        assert_eq!(buffer.gpu_address(), pin);
        assert_eq!(
            buffer.contents().addr(),
            r.region.base() as usize + (at + 0x3000) as usize
        );
        let copy = r.dec.managed_copy(10).unwrap();
        assert_eq!((copy.at, copy.length), (at + 0x3000, 0x100));
        assert!(copy.cpu.is_none());
        assert!(r.dec.span_in_use(at + 0x3000, 0x100));

        let mark = r.p.call_log().mark();
        assert_eq!(r.modified(10, 16, 8), Ok(()));
        assert!(r.synchronize(10, RGFX_BLIT_RESOURCE_BUFFER).unwrap());
        assert_eq!(
            r.p.call_log().sels_since(mark),
            ["didModifyRange:", "synchronizeResource:"]
        );

        // Refusals: no such heap, a buffer for a heap, past the heap's block, not Managed.
        assert_eq!(
            r.send(&managed(11, 31, 0, 0x100, MANAGED)),
            Err(DecodeError::BadReference)
        );
        assert_eq!(
            r.send(&managed(12, 10, 0, 0x100, MANAGED)),
            Err(DecodeError::BadReference)
        );
        assert_eq!(
            r.send(&managed(13, 30, 0x1_ff00, 0x200, MANAGED)),
            Err(DecodeError::Malformed)
        );
        assert_eq!(
            r.send(&managed(14, 30, 0x5000, 0x100, 0)),
            Err(DecodeError::Refused)
        );
        let mut unified = rig(true);
        assert_eq!(
            unified.send(&managed(15, 30, 0, 0x100, MANAGED)),
            Err(DecodeError::Refused)
        );

        // A destroy forgets the CPU copy, and the heap's block is free once the heap goes too.
        assert_eq!(
            r.send(&RgfxObjectDestroy {
                header: header(RGFX_CMD_OBJECT_DESTROY),
                obj_id: 10,
            }),
            Ok(())
        );
        assert!(r.dec.managed_copy(10).is_none());
        assert_eq!(
            r.send(&RgfxObjectDestroy {
                header: header(RGFX_CMD_OBJECT_DESTROY),
                obj_id: 30,
            }),
            Ok(())
        );
        assert!(!r.dec.span_in_use(at, 0x2_0000) && r.dec.object_count() == 0);
    }

    /// `BUFFER_MANAGED_AT_OFFSET` in a heap over a pages block's guest RAM (`HEAP_OVER_PAGES`): a
    /// no-copy buffer over the guest's RAM at the offset, not over the region, whose range
    /// messages and synchronize are the platform's. The buffer lies over the pages block; a CPU
    /// copy past the heap's memory is malformed.
    #[test]
    fn a_managed_buffer_in_a_heap_over_pages_lies_over_the_guest_s_ram() {
        let (mut r, ram, page) = pages_rig();
        let managed = |obj_id: u64, offset: u64, length: u64| RgfxBufferManagedAtOffset {
            header: header(RGFX_CMD_BUFFER_MANAGED_AT_OFFSET),
            obj_id,
            gpu_address: 0x1_0000_0000 + obj_id * 0x10_0000,
            heap_obj_id: 20,
            offset,
            length,
            options: MANAGED,
            reserved_3c: 0,
        };
        let first: Vec<u8> = (0..0x100u32).map(|i| (i * 5) as u8).collect();
        assert!(ram.write(pages::RAM + 2 * page + 0x3000, &first));
        let mark = r.p.call_log().mark();
        assert_eq!(r.send(&managed(10, 0x3000, 0x100)), Ok(()));
        assert_eq!(
            made(r.p.call_log().sels_since(mark)),
            ["newBufferWithDescriptor:"]
        );
        let buffer = r.dec.require_buffer(10).unwrap().clone_ref();
        let heap_first = ram.ranges()[0].host().wrapping_add(2 * page);
        assert_eq!(buffer.contents(), heap_first.wrapping_add(0x3000));
        // SAFETY: the buffer lies over the guest's RAM, mapped while the rig lives.
        let seen = unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr::<u8>(), 0x100) };
        assert_eq!(seen, first);
        let copy = r.dec.managed_copy(10).unwrap();
        assert_eq!(
            (copy.at, copy.length, copy.ram),
            (0x3000, 0x100, Some(heap_first))
        );
        assert!(copy.cpu.is_none());
        assert!(r.dec.span_in_use(pages::BLOCK, 0x2000));

        let mark = r.p.call_log().mark();
        assert_eq!(r.modified(10, 16, 8), Ok(()));
        assert!(r.synchronize(10, RGFX_BLIT_RESOURCE_BUFFER).unwrap());
        assert_eq!(
            r.p.call_log().sels_since(mark),
            ["didModifyRange:", "synchronizeResource:"]
        );
        assert_eq!(
            r.send(&managed(11, 4 * page - 0x80, 0x100)),
            Err(DecodeError::Malformed)
        );

        for obj_id in [10, 20] {
            assert_eq!(
                r.send(&RgfxObjectDestroy {
                    header: header(RGFX_CMD_OBJECT_DESTROY),
                    obj_id,
                }),
                Ok(())
            );
        }
        assert!(!r.dec.span_in_use(pages::BLOCK, 0x2000));
    }

    /// On a GPU that does not share the CPU's memory a `Managed` texture is the platform's own
    /// beside its CPU copy: a replaced region is the platform's `replaceRegion:` from the
    /// region's bytes in the block, at the layout's rows; a synchronize copies the named level
    /// and slice, or every one, to the block at the layout's offsets. A CPU copy shorter than the
    /// layout, or one that is a `Managed` buffer, is refused.
    #[test]
    fn a_managed_texture_on_a_discrete_gpu_is_two_copies() {
        let mut r = rig(false);
        let at = BLOCKS + 0x1_0000;
        // 100x60 RGBA8, three levels: 42240 bytes (reims_vgpu_graphics::managed's case).
        assert_eq!(r.buffer(20, at, 42240, 0), Ok(()));
        assert_eq!(r.buffer(22, at + 0x2_0000, 42239, 0), Ok(()));
        assert_eq!(r.buffer(23, at + 0x3_0000, 42240, MANAGED), Ok(()));
        assert_eq!(
            r.texture(24, 22, rgba(100, 60, 3)),
            Err(DecodeError::BadReference)
        );
        assert_eq!(
            r.texture(24, 23, rgba(100, 60, 3)),
            Err(DecodeError::BadReference)
        );
        let mark = r.p.call_log().mark();
        assert_eq!(r.texture(21, 20, rgba(100, 60, 3)), Ok(()));
        assert_eq!(
            made(r.p.call_log().sels_since(mark)),
            ["newTextureWithDescriptor:"]
        );
        assert!(r.dec.span_in_use(at, 42240), "the texture keeps its block");

        let mark = r.p.call_log().mark();
        assert_eq!(r.replaced(21, 1, 0, [2, 1, 0], [4, 3, 1]), Ok(()));
        let calls = r.p.call_log().since(mark);
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].sel,
            "replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:"
        );
        let region = MtlRegion {
            origin: MtlOrigin { x: 2, y: 1, z: 0 },
            size: MtlSize {
                width: 4,
                height: 3,
                depth: 1,
            },
        };
        let first = r.region.base() as usize + (at + 30720 + 256 + 8) as usize;
        assert_eq!(
            calls[0].args,
            [
                Arg::val(&region),
                Arg::U64(1),
                Arg::U64(0),
                Arg::Ptr(first),
                Arg::U64(256),
                Arg::U64(0)
            ]
        );
        assert_eq!(
            r.replaced(21, 1, 0, [48, 0, 0], [4, 1, 1]),
            Err(DecodeError::Malformed)
        );
        assert_eq!(
            r.replaced(21, 0, 1, [0, 0, 0], [1, 1, 1]),
            Err(DecodeError::Malformed)
        );
        assert_eq!(
            r.replaced(20, 0, 0, [0, 0, 0], [1, 1, 1]),
            Err(DecodeError::BadReference)
        );

        let copy = "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:\
                    destinationOffset:destinationBytesPerRow:destinationBytesPerImage:";
        let mark = r.p.call_log().mark();
        assert!(r.synchronize_texture(21, 0, 2).unwrap());
        let calls = r.p.call_log().since(mark);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].sel, copy);
        assert_eq!(
            calls[0].args,
            [
                r.obj(21),
                Arg::U64(0),
                Arg::U64(2),
                Arg::val(&MtlOrigin { x: 0, y: 0, z: 0 }),
                Arg::val(&MtlSize {
                    width: 25,
                    height: 15,
                    depth: 1
                }),
                r.cpu_of(21),
                Arg::U64(38400),
                Arg::U64(256),
                Arg::U64(0)
            ]
        );
        let mark = r.p.call_log().mark();
        assert!(r.synchronize(21, RGFX_BLIT_RESOURCE_TEXTURE).unwrap());
        let calls = r.p.call_log().since(mark);
        let offsets: Vec<Arg> = calls.iter().map(|c| c.args[6].clone()).collect();
        assert!(calls.iter().all(|c| c.sel == copy));
        assert_eq!(offsets, [Arg::U64(0), Arg::U64(30720), Arg::U64(38400)]);
        assert!(
            r.synchronize_texture(21, 0, 3).is_err(),
            "a level past the texture"
        );
    }

    /// After a recovery of the device, the first command buffer of the process fills each
    /// two-copy texture made before it again from its CPU copy, every level whole, before the
    /// buffer is begun on the platform; a texture made after the recovery, and every later buffer,
    /// takes none, and a second recovery refills both textures again.
    #[test]
    fn a_recovery_fills_two_copy_textures_again_from_their_cpu_copies() {
        use reims_vgpu_graphics::ops::{RGFX_COMMAND_BUFFER_OPS_OFFSET, RgfxCommandBuffer};
        use reims_vgpu_graphics::wire::RgfxQueueCreate;
        use reims_vgpu_graphics::wire::{RGFX_CMD_COMMAND_BUFFER, RGFX_CMD_QUEUE_CREATE};
        use reims_vgpu_vulkan::core::testkit::objects::CompleteOn;

        let replace = "replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:";
        let mut r = rig(false);
        let at = BLOCKS + 0x1_0000;
        let later = at + 0x2_0000;
        // 100x60 RGBA8, three levels: 42240 bytes, its levels at 0, 30720 and 38400.
        assert_eq!(r.buffer(20, at, 42240, 0), Ok(()));
        assert_eq!(r.buffer(22, later, 42240, 0), Ok(()));
        assert_eq!(r.texture(21, 20, rgba(100, 60, 3)), Ok(()));
        let queue = RgfxQueueCreate {
            header: header(RGFX_CMD_QUEUE_CREATE),
            obj_id: 1,
            ..RgfxQueueCreate::default()
        };
        assert_eq!(r.send(&queue), Ok(()));
        let mut seqno = 0;
        let mut begin = |r: &mut Rig| -> Vec<(&'static str, Vec<Arg>)> {
            seqno += 1;
            let m = RgfxCommandBuffer {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_COMMAND_BUFFER,
                    size: RGFX_COMMAND_BUFFER_OPS_OFFSET,
                    seqno,
                },
                queue_obj_id: 1,
                cb_obj_id: 100 + seqno,
                op_bytes: 0,
            };
            let mark = r.p.call_log().mark();
            assert_eq!(r.send(&m), Ok(()));
            r.p.world().complete_all(CompleteOn::Caller);
            let calls = r.p.call_log().since(mark).into_iter();
            calls.map(|c| (c.sel, c.args)).collect()
        };
        let refills = |calls: &[(&'static str, Vec<Arg>)]| -> Vec<Vec<Arg>> {
            (calls.iter())
                .filter(|(sel, _)| *sel == replace)
                .map(|(_, args)| args.clone())
                .collect()
        };
        let whole = |r: &Rig, first: u64| -> Vec<Vec<Arg>> {
            let levels = [
                (0u64, 100u64, 60u64, 512u64),
                (30720, 50, 30, 256),
                (38400, 25, 15, 256),
            ];
            (0u64..)
                .zip(levels)
                .map(|(level, (offset, width, height, row))| {
                    let region = MtlRegion {
                        origin: MtlOrigin { x: 0, y: 0, z: 0 },
                        size: MtlSize {
                            width,
                            height,
                            depth: 1,
                        },
                    };
                    let ptr = r.region.base() as usize + (first + offset) as usize;
                    vec![
                        Arg::val(&region),
                        Arg::U64(level),
                        Arg::U64(0),
                        Arg::Ptr(ptr),
                        Arg::U64(row),
                        Arg::U64(0),
                    ]
                })
                .collect()
        };

        assert_eq!(
            refills(&begin(&mut r)),
            Vec::<Vec<Arg>>::new(),
            "no recovery, no refill"
        );
        r.p.world().recover();
        assert_eq!(r.texture(23, 22, rgba(100, 60, 3)), Ok(()));
        let calls = begin(&mut r);
        assert_eq!(
            refills(&calls),
            whole(&r, at),
            "texture 21's levels, from its CPU copy"
        );
        let first_refill = calls.iter().position(|(sel, _)| *sel == replace).unwrap();
        let opened = calls
            .iter()
            .position(|(sel, _)| *sel == "commandBuffer")
            .unwrap();
        assert!(
            first_refill < opened,
            "the refill comes before the buffer is begun: {calls:?}"
        );
        assert_eq!(
            refills(&begin(&mut r)),
            Vec::<Vec<Arg>>::new(),
            "one refill a recovery"
        );

        r.p.world().recover();
        let mut both = whole(&r, at);
        both.extend(whole(&r, later));
        let mut got = refills(&begin(&mut r));
        let key = |args: &Vec<Arg>| format!("{args:?}");
        got.sort_by_key(key);
        both.sort_by_key(key);
        assert_eq!(got, both, "a second recovery refills both textures");
    }

    /// A `Managed` texture's CPU copy is its buffer's contents wherever they lie in the guest's
    /// memory (`RGFX_FEATURE_MANAGED_TEXTURE_SLABS`): a buffer placed in a heap over a pages
    /// block's guest RAM, one placed in a heap over a block of the region, and one over a pages
    /// block's view each have a replaced region read from their own bytes, where the block their
    /// span names would hold the pages block's runs or the heap's first bytes.
    #[test]
    fn a_managed_texture_s_cpu_copy_is_its_buffer_s_bytes_wherever_they_lie() {
        let (mut r, ram, page) = pages_rig();
        // 16x16 RGBA8, one level: 16 rows of 256 bytes.
        let size = 16 * 256;
        let at_offset = |obj_id: u64, heap_obj_id: u64, offset: u64| RgfxBufferAtOffset {
            header: header(RGFX_CMD_BUFFER_AT_OFFSET),
            obj_id,
            heap_obj_id,
            offset,
            length: size,
            options: 0,
            reserved_34: 0,
        };
        // In heap 20, over the pages block's guest RAM, a page in.
        assert_eq!(r.send(&at_offset(30, 20, page)), Ok(()));
        assert_eq!(r.texture(31, 30, rgba(16, 16, 1)), Ok(()));
        // In a heap over a block of the region, 0x2000 in.
        let heap_at = pages::BLOCK + 0x1_0000;
        assert_eq!(
            r.send(&RgfxHeapOverHostMemory {
                header: header(RGFX_CMD_HEAP_OVER_HOST_MEMORY),
                obj_id: 40,
                gpu_address: 0x3_0000_0000,
                region_offset: heap_at,
                size_bytes: 0x1_0000,
                options: 0,
                heap_type: 1,
            }),
            Ok(())
        );
        assert_eq!(r.send(&at_offset(41, 40, 0x2000)), Ok(()));
        assert_eq!(r.texture(42, 41, rgba(16, 16, 1)), Ok(()));
        // Over the pages block's view, 0x100 in.
        assert_eq!(
            r.send(&RgfxBufferOverPages {
                header: header(RGFX_CMD_BUFFER_OVER_PAGES),
                obj_id: 50,
                gpu_address: 0x4_0000_0000,
                block_offset: pages::BLOCK,
                block_size: 0x2000,
                offset: 0x100,
                length: size,
                options: 0,
                reserved_44: 0,
            }),
            Ok(())
        );
        assert_eq!(r.texture(51, 50, rgba(16, 16, 1)), Ok(()));

        let contents = |r: &Rig, obj_id: u64| match r.dec.named(obj_id) {
            Some(ObjHandle::Buffer(b)) => b.contents().addr(),
            _ => panic!("no buffer {obj_id:#x}"),
        };
        // The heap over the pages block starts at the run's first page, page 2; the buffer is a
        // page in.
        let in_ram = ram.ranges()[0].host().wrapping_add(3 * page).addr();
        let in_region = r.region.base() as usize + (heap_at + 0x2000) as usize;
        for (texture, cpu, first) in [
            (31, 30, in_ram),
            (42, 41, in_region),
            (51, 50, contents(&r, 50)),
        ] {
            let mark = r.p.call_log().mark();
            assert_eq!(r.replaced(texture, 0, 0, [2, 1, 0], [4, 3, 1]), Ok(()));
            let calls = r.p.call_log().since(mark);
            assert_eq!(calls.len(), 1, "texture {texture}");
            // Row 1, texel 2.
            assert_eq!(
                calls[0].args[3],
                Arg::Ptr(first + 256 + 8),
                "texture {texture}"
            );
            assert_eq!(r.cpu_of(texture), r.obj(cpu));
        }
        // A buffer whose contents are no guest bytes is no CPU copy: a `Managed` one.
        assert_eq!(r.buffer(60, heap_at + 0x2_0000, size, MANAGED), Ok(()));
        assert_eq!(
            r.texture(61, 60, rgba(16, 16, 1)),
            Err(DecodeError::BadReference)
        );
    }

    /// A `MANAGED_BUFFER_CREATE` of `length` bytes and `options` at the guest's address over CPU
    /// copy `cpu`.
    fn managed_over(obj_id: u64, cpu: u64, length: u64, options: u32) -> RgfxManagedBufferCreate {
        RgfxManagedBufferCreate {
            header: header(RGFX_CMD_MANAGED_BUFFER_CREATE),
            obj_id,
            gpu_address: 0x1_0000_0000 + obj_id * 0x10_0000,
            cpu_buffer_obj_id: cpu,
            length,
            options,
            reserved_34: 0,
        }
    }

    /// The CPU copies of the Managed buffers below: Shared buffers of 0x400 bytes placed in heap
    /// 20, over the pages block's guest RAM (30, a page in), in heap 40, over a block of the region
    /// (41, 0x2000 in), and over the pages block's view (50, 0x100 in). Answers each buffer's id
    /// with the host address its contents start at: for the two in heaps, worked out from the
    /// guest's RAM and the region, as the fake's heaps over host memory hold no bytes of their
    /// own.
    fn shared_copies(r: &mut Rig, ram: &GuestRam, page: u64) -> [(u64, usize); 3] {
        let at_offset = |obj_id: u64, heap_obj_id: u64, offset: u64| RgfxBufferAtOffset {
            header: header(RGFX_CMD_BUFFER_AT_OFFSET),
            obj_id,
            heap_obj_id,
            offset,
            length: 0x400,
            options: 0,
            reserved_34: 0,
        };
        assert_eq!(r.send(&at_offset(30, 20, page)), Ok(()));
        assert_eq!(
            r.send(&RgfxHeapOverHostMemory {
                header: header(RGFX_CMD_HEAP_OVER_HOST_MEMORY),
                obj_id: 40,
                gpu_address: 0x3_0000_0000,
                region_offset: pages::BLOCK + 0x1_0000,
                size_bytes: 0x1_0000,
                options: 0,
                heap_type: 1,
            }),
            Ok(())
        );
        assert_eq!(r.send(&at_offset(41, 40, 0x2000)), Ok(()));
        assert_eq!(
            r.send(&RgfxBufferOverPages {
                header: header(RGFX_CMD_BUFFER_OVER_PAGES),
                obj_id: 50,
                gpu_address: 0x4_0000_0000,
                block_offset: pages::BLOCK,
                block_size: 0x2000,
                offset: 0x100,
                length: 0x400,
                options: 0,
                reserved_44: 0,
            }),
            Ok(())
        );
        let contents = |r: &Rig, obj_id: u64| match r.dec.named(obj_id) {
            Some(ObjHandle::Buffer(b)) => b.contents().addr(),
            _ => panic!("no buffer {obj_id:#x}"),
        };
        [
            // Heap 20 lies over the block's run from guest page 2.
            (30, ram.ranges()[0].host().wrapping_add(3 * page).addr()),
            (
                41,
                (r.region.base())
                    .wrapping_add((pages::BLOCK + 0x1_2000) as usize)
                    .addr(),
            ),
            (50, contents(r, 50)),
        ]
    }

    /// `MANAGED_BUFFER_CREATE`, whether or not the GPU shares the CPU's memory: a no-copy buffer
    /// of the `Managed` options at the guest's address over its CPU copy's buffer's bytes,
    /// wherever they lie (in a heap over a pages block's guest RAM, in a heap over a block of the
    /// region, in a pages block's view), which keeps that buffer: the platform keeps its copies.
    /// Its range messages and synchronize are the platform's, and it lies over what its CPU copy
    /// lies over, after the CPU copy's destroy too.
    #[test]
    fn a_managed_buffer_over_a_shared_buffer_lies_over_its_bytes_on_any_gpu() {
        for unified in [false, true] {
            let (mut r, ram, page) = pages_rig();
            r.dec.set_unified_memory(unified);
            let copies = shared_copies(&mut r, &ram, page);
            for (i, (cpu, first)) in copies.into_iter().enumerate() {
                let obj_id = 60 + i as u64;
                let mark = r.p.call_log().mark();
                assert_eq!(r.send(&managed_over(obj_id, cpu, 0x300, MANAGED)), Ok(()));
                let calls = r.p.call_log().since(mark);
                let news: Vec<_> = calls.iter().filter(|c| c.sel.starts_with("new")).collect();
                assert_eq!(news.len(), 1, "copy {cpu}, unified {unified}");
                let pin = 0x1_0000_0000 + obj_id * 0x10_0000;
                // SAFETY: the range is only formatted.
                let bytes = unsafe { HostRange::new(HostPtr::from_ptr(first as *mut u8), 0x300) };
                assert_eq!(
                    news[0].args[0],
                    Arg::val(&BufferDesc {
                        length: 0x300,
                        options: vec![OptionSet::ResourceOptions(MANAGED.into())],
                        pinned_gpu_address: Some(pin),
                        contents: Some(bytes),
                        no_copy: true,
                        deallocator: Some(Box::new(|_, _| {})),
                        zero_from: None,
                    }),
                    "copy {cpu}, unified {unified}"
                );
                let buffer = r.dec.require_buffer(obj_id).unwrap();
                assert_eq!(buffer.gpu_address(), pin);
                assert_eq!(buffer.contents().addr(), first);
                assert!(r.dec.managed_copy(obj_id).unwrap().cpu.is_none());
                let mark = r.p.call_log().mark();
                assert_eq!(r.modified(obj_id, 16, 8), Ok(()));
                assert_eq!(r.modified(obj_id, 0x2f8, 16), Err(DecodeError::Malformed));
                assert!(r.synchronize(obj_id, RGFX_BLIT_RESOURCE_BUFFER).unwrap());
                assert_eq!(
                    r.p.call_log().sels_since(mark),
                    ["didModifyRange:", "synchronizeResource:"]
                );
            }
            // The buffer lies over its CPU copy's block, and keeps its CPU copy's buffer when
            // that goes first.
            assert_eq!(
                r.send(&RgfxObjectDestroy {
                    header: header(RGFX_CMD_OBJECT_DESTROY),
                    obj_id: 41,
                }),
                Ok(())
            );
            assert!(r.dec.span_in_use(pages::BLOCK + 0x1_0000, 0x1_0000));
            assert_eq!(r.modified(61, 32, 4), Ok(()));
            assert!(r.synchronize(61, RGFX_BLIT_RESOURCE_BUFFER).unwrap());
        }
    }

    /// A `MANAGED_BUFFER_CREATE`'s deallocator, which the Vulkan platform runs from rt's release
    /// of its import of the bytes, lets go of no handle of the platform's: the CPU copy's buffer
    /// it holds, its last hold once the guest destroyed the copy, goes on the surface worker after
    /// the deallocator ran, wherever the copy lies, whether the platform runs the deallocator at
    /// the buffer's death or later. Nor do the deallocators of what the copies lie over: the
    /// buffer over the pages block's view, and the heaps over the block's RAM and over the region.
    #[test]
    fn a_managed_buffer_s_deallocator_lets_go_of_its_cpu_copy_s_buffer_on_the_surface_worker() {
        let destroy = |r: &mut Rig, obj_id: u64| {
            let m = RgfxObjectDestroy {
                header: header(RGFX_CMD_OBJECT_DESTROY),
                obj_id,
            };
            assert_eq!(r.send(&m), Ok(()), "destroy {obj_id}");
        };
        for late in [false, true] {
            let (mut r, ram, page) = pages_rig();
            let copies = shared_copies(&mut r, &ram, page);
            if late {
                r.p.world().run_deallocators_late();
            }
            for (i, (cpu, _)) in copies.into_iter().enumerate() {
                let obj_id = 60 + i as u64;
                assert_eq!(r.send(&managed_over(obj_id, cpu, 0x300, MANAGED)), Ok(()));
                let gone = match r.dec.named(cpu) {
                    Some(ObjHandle::Buffer(b)) => Arc::downgrade(b.obj()),
                    _ => panic!("no buffer {cpu:#x}"),
                };
                destroy(&mut r, cpu);
                destroy(&mut r, obj_id);
                r.dec.surface_worker().flush();
                if late {
                    assert!(
                        gone.upgrade().is_some(),
                        "copy {cpu}: the deallocator, not yet run, holds it"
                    );
                    r.dec.surface_worker().hold();
                    assert_eq!(r.p.world().run_late_deallocators(), 1, "copy {cpu}");
                    assert!(
                        gone.upgrade().is_some(),
                        "copy {cpu}: handed to the worker, which has not run"
                    );
                    r.dec.surface_worker().release();
                    r.dec.surface_worker().flush();
                }
                assert!(gone.upgrade().is_none(), "copy {cpu}, late {late}");
            }
            destroy(&mut r, 20);
            destroy(&mut r, 40);
            r.dec.surface_worker().flush();
            if late {
                r.p.world().run_late_deallocators();
                r.dec.surface_worker().flush();
            }
            let dropped = r.p.world().handles_dropped_in_deallocators();
            assert!(dropped.is_empty(), "late {late}: {dropped:?}");
        }
    }

    /// Where the guest's driver took `MANAGED_AT_COPY`, a `MANAGED_BUFFER_CREATE` at its CPU
    /// copy's address is the buffer there where the platform keeps the pin. Where the platform
    /// makes the buffer at another address (a copy in no block it knows), the buffer is kept
    /// there, counted ([`MANAGED_AT_COPY_UNKEPT`]), with every answer of the same create where the
    /// driver did not take the bit: the create's result, the platform's calls, the binding of the
    /// id, the buffer's GPU address (the platform's, not the one posted), the pins counted unkept,
    /// and a `didModifyRange:` and a synchronize of it. Where the driver did not take the bit,
    /// nothing is counted as it.
    #[test]
    fn a_managed_buffer_at_its_copy_made_elsewhere_is_kept_as_without_the_bit() {
        use super::super::unkept::Kind;
        use reims_vgpu_vulkan::answer::counted_kinds;
        let counted = |code: &str| {
            (counted_kinds().into_iter())
                .find(|&(c, _)| c == code)
                .map_or(0, |(_, n)| n)
        };
        let count = || counted(MANAGED_AT_COPY_UNKEPT);
        let unpinned = || counted(Kind::ManagedUnified.codes().unpinned);
        let (mut r, ram, page) = pages_rig();
        r.dec.set_unified_memory(true);
        r.dec.set_pins_in_blocks();
        let _ = shared_copies(&mut r, &ram, page);
        let address = |r: &Rig, obj_id: u64| match r.dec.named(obj_id) {
            Some(ObjHandle::Buffer(b)) => b.gpu_address(),
            _ => panic!("no buffer {obj_id:#x}"),
        };
        let at_copy = |obj_id, cpu, gpu_address| RgfxManagedBufferCreate {
            gpu_address,
            ..managed_over(obj_id, cpu, 0x300, MANAGED)
        };
        let decode = |r: &mut Rig, m: RgfxManagedBufferCreate| {
            r.dec.decode_refusing(&r.device, m.as_bytes()).unwrap()
        };

        // Taken, the pin kept: the buffer at its copy's address, over the pages block's view.
        r.dec.set_managed_at_copy(true);
        let before = count();
        let at = address(&r, 50);
        assert_eq!(decode(&mut r, at_copy(60, 50, at)), Ok(()));
        assert_eq!(address(&r, 60), at);
        assert_eq!(count(), before);

        // The pin dropped, taken (61) and not (62): the same create, with the same answers.
        r.p.world().drop_pins(true);
        let at = address(&r, 41);
        let mut answers = Vec::new();
        for (obj_id, taken) in [(61, true), (62, false)] {
            r.dec.set_managed_at_copy(taken);
            let (counted_before, unpinned_before) = (count(), unpinned());
            let mark = r.p.call_log().mark();
            let made = decode(&mut r, at_copy(obj_id, 41, at));
            let calls: Vec<_> = (r.p.call_log().since(mark).into_iter())
                .map(|c| (c.class, c.sel, c.args))
                .collect();
            assert_eq!(
                count() - counted_before,
                u64::from(taken),
                "counted where taken"
            );
            let placed = address(&r, obj_id);
            assert_ne!(placed, at, "where the platform placed it");
            let bound = r.dec.require_buffer(obj_id).ok().map(|b| b.gpu_address());
            let unkept = unpinned() - unpinned_before;
            let modified = r.modified(obj_id, 0, 4);
            let synchronized = r.synchronize(obj_id, RGFX_BLIT_RESOURCE_BUFFER).unwrap();
            answers.push((
                made,
                calls,
                bound == Some(placed),
                unkept,
                modified,
                synchronized,
            ));
        }
        assert_eq!(answers[0].0, Ok(()));
        assert_eq!(answers[0].3, 1, "its pin counted unkept");
        assert_eq!(
            answers[0], answers[1],
            "the bit's answers are the create's without it"
        );
        assert_eq!(count(), before + 1);
    }

    /// On metal2vulkan-runtime's platform, a `Managed` buffer over a slab's CPU copy (a Shared
    /// buffer in a heap over a block of the guest's Shared pool, at the block's GPU address,
    /// `POOL_ADDRESSES`), which the GPU reads and writes by a blit. On a GPU that shares the CPU's
    /// memory the device offers `MANAGED_AT_COPY`, the guest posts its copy's address, and the
    /// buffer is the copy's bytes at that address: the GPU reads what the CPU wrote, before and
    /// after a later CPU write with no `didModifyRange:`, and the CPU reads the GPU's write. There
    /// a create over a copy the platform placed in no block it knows (a buffer over the region, at
    /// an address of rt's own and not the guest's) is kept where rt made it, counted, with the
    /// answers of the same create without the bit: made, bound, and at rt's address rather than
    /// the one posted. On a GPU of its own memory no such bit is offered, and the buffer keeps two
    /// copies: the GPU reads its own, which holds the CPU's bytes as of the first commit after the
    /// make, until a `didModifyRange:` brings a later write over; and the CPU's bytes keep out the
    /// GPU's write.
    #[test]
    fn on_vulkan_rt_s_managed_buffer_at_its_copy_is_the_copy_s_bytes_where_memory_is_shared() {
        use std::sync::mpsc;
        use std::time::Duration;

        use reims_vgpu_graphics::wire::{
            RGFX_CMD_HEAP_OVER_POOL, RGFX_CONTEXT_OBJ_ID_SHIFT, RGFX_FEATURE_MANAGED_AT_COPY,
        };
        use reims_vgpu_vulkan::RtPlatform;
        use reims_vgpu_vulkan::answer::counted_kinds;
        use reims_vgpu_vulkan::core::platform::os::Clock as _;

        use crate::platform::ApiPlatform;
        use crate::pool::{Pool as SharedPool, PoolConfig, PoolHost};

        let _pool = reims_vgpu_vulkan::core::rt::Pool::here();
        // A platform of the test's own, whose device goes before the process exits.
        let platform = crate::rt_test_end::platform();
        let device = platform.metal_device(None).unwrap();
        let unified = (device.identity_caps().flags.has_unified_memory).unwrap_or(false);
        let pins = <RtPlatform as ApiPlatform>::PINS_IN_BLOCKS;
        let offered = crate::graphics::managed_at_copy(unified, pins, true);
        assert_eq!(
            offered == RGFX_FEATURE_MANAGED_AT_COPY,
            unified,
            "offered on unified alone"
        );
        let unkept = || {
            (counted_kinds().into_iter())
                .find(|&(c, _)| c == MANAGED_AT_COPY_UNKEPT)
                .map_or(0, |(_, n)| n)
        };

        // The guest's Shared pool, its blocks rt's, and one block of context 1's.
        let source = platform.block_source(&device).expect("rt's blocks");
        let (added, adds) = mpsc::channel();
        let host = PoolHost {
            add: Arc::new(move |at, _, _| {
                let _ = added.send(at);
            }),
            remove: Arc::new(|_, _| {}),
        };
        let clock = Arc::clone(&platform);
        let now = Box::new(move || clock.clock().now_mono());
        let range = PoolConfig {
            physical_address: 52 << 30,
            physical_length: 4 << 30,
        };
        let shared_pool = SharedPool::new(range, host, now, source).expect("a pool");
        let ledger = shared_pool.ledger();
        assert!(ledger.keep_headroom());
        let at = adds.recv_timeout(Duration::from_secs(60)).expect("a piece");
        shared_pool.added(at).unwrap();
        let (taken, takes) = mpsc::channel();
        let answer = move |block| {
            let _ = taken.send(block);
        };
        ledger.take(1, 0x1_0000, Box::new(answer)).unwrap();
        let block = (takes.recv_timeout(Duration::from_secs(60)).unwrap()).expect("a block");

        let size = BLOCKS + 0x10_0000 + RGFX_CURSOR_SPAN;
        let region = Arc::new(crate::platform::test_region(&*platform, size as usize));
        let mut dec = GraphicsDecoder::<RtPlatform>::new(Arc::clone(&platform), region);
        dec.set_unified_memory(unified);
        dec.set_pins_in_blocks();
        dec.set_pool(ledger.clone());
        dec.set_managed_at_copy(offered != 0);
        let buffer_of = |dec: &GraphicsDecoder<RtPlatform>, obj_id: u64| match dec.named(obj_id) {
            Some(ObjHandle::Buffer(b)) => b.clone_ref(),
            _ => panic!("no buffer {obj_id:#x}"),
        };

        // A copy over the region, which rt places at an address of its own, not the guest's: on
        // unified memory the guest posts the address it holds, and rt makes the buffer at its own,
        // where it is kept, as the same create without the bit is.
        if unified {
            const GUEST_ADDRESS: u64 = 0x3_0000_0000;
            let over = RgfxBufferOverHostMemory {
                header: header(RGFX_CMD_BUFFER_OVER_HOST_MEMORY),
                obj_id: 30,
                gpu_address: GUEST_ADDRESS,
                region_offset: BLOCKS + 0x8_0000,
                length: 0x400,
                options: 0,
                ..RgfxBufferOverHostMemory::default()
            };
            assert_eq!(dec.decode(&device, over.as_bytes()).unwrap(), Ok(()));
            let placed = buffer_of(&dec, 30).gpu_address();
            assert_ne!(placed, GUEST_ADDRESS, "rt keeps no pin outside its blocks");
            let mut answers = Vec::new();
            for (obj_id, taken) in [(31, true), (32, false)] {
                dec.set_managed_at_copy(taken);
                let before = unkept();
                let managed = RgfxManagedBufferCreate {
                    gpu_address: GUEST_ADDRESS,
                    ..managed_over(obj_id, 30, 0x100, MANAGED)
                };
                let made = dec.decode_refusing(&device, managed.as_bytes()).unwrap();
                assert_eq!(unkept() - before, u64::from(taken), "counted where taken");
                let bound = dec.require_buffer(obj_id).ok().map(|b| b.gpu_address());
                answers.push((made, bound.is_some_and(|at| at != GUEST_ADDRESS)));
            }
            dec.set_managed_at_copy(true);
            assert_eq!(
                answers[0],
                (Ok(()), true),
                "made, bound, at rt's own address"
            );
            assert_eq!(
                answers[0], answers[1],
                "the bit's answers are the create's without it"
            );
        }

        // The slab's copy, at the pool block's address plus its offset.
        dec.set_context(Some(1));
        let of_1 = |n: u64| (1 << RGFX_CONTEXT_OBJ_ID_SHIFT) | n;
        let heap = RgfxHeapOverHostMemory {
            header: header(RGFX_CMD_HEAP_OVER_POOL),
            obj_id: of_1(20),
            gpu_address: block.gpu_address,
            region_offset: block.offset,
            size_bytes: 0x1_0000,
            options: 0,
            // MTLHeapTypePlacement.
            heap_type: 1,
        };
        assert_eq!(dec.decode(&device, heap.as_bytes()).unwrap(), Ok(()));
        let copy = RgfxBufferAtOffset {
            header: header(RGFX_CMD_BUFFER_AT_OFFSET),
            obj_id: of_1(21),
            heap_obj_id: of_1(20),
            offset: 0x1000,
            length: 0x400,
            options: 0,
            reserved_34: 0,
        };
        assert_eq!(dec.decode(&device, copy.as_bytes()).unwrap(), Ok(()));
        let copy = buffer_of(&dec, of_1(21));
        let copy_at = block.gpu_address + 0x1000;
        assert_eq!(copy.gpu_address(), copy_at);
        let words = copy.contents().as_ptr::<u32>();
        let write = |at: usize, value: u32| {
            // SAFETY: a word of the copy's 0x400 bytes, written while no command runs.
            unsafe { words.add(at).write_volatile(value) }
        };
        // SAFETY: a word of the copy's 0x400 bytes, read while no command runs.
        let read = |at: usize| unsafe { words.add(at).read_volatile() };
        const OLD: u32 = 0x1111_1111;
        const NEW: u32 = 0x2222_2222;
        const AFTER: u32 = 0x5a5a_5a5a;
        const GPU: u32 = 0x3333_3333;
        write(0, OLD);
        write(1, AFTER);

        // The Managed buffer, at the copy's address where the guest posts it (unified memory),
        // else where rt places its GPU copy.
        let posted = if unified { copy_at } else { 0 };
        let managed = RgfxManagedBufferCreate {
            gpu_address: posted,
            ..managed_over(of_1(22), of_1(21), 0x100, MANAGED)
        };
        assert_eq!(dec.decode(&device, managed.as_bytes()).unwrap(), Ok(()));
        let buffer = buffer_of(&dec, of_1(22));
        if unified {
            assert_eq!(
                buffer.gpu_address(),
                posted,
                "at the address the guest posted"
            );
        }

        let shared = |length: u64| {
            device
                .new_buffer_of_length(OnFault::Die, length, 0)
                .unwrap()
                .unwrap()
        };
        // The word the GPU reads from the buffer, and the one it writes into the buffer's second.
        let (out, from) = (shared(4), shared(4));
        // SAFETY: a Shared buffer of 4 bytes, which no command uses yet.
        unsafe { from.contents().as_ptr::<u32>().write(GPU) };
        let queue = device.new_command_queue().unwrap();
        // The GPU's read of the buffer's first word, and its write of the second.
        let reach = || {
            let cb = queue.command_buffer(OnFault::Die).unwrap().unwrap();
            let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
            for cmd in [
                BlitCmd::CopyBufferToBuffer {
                    source_buffer: Some(&buffer),
                    source_offset: 0,
                    destination_buffer: Some(&out),
                    destination_offset: 0,
                    size: 4,
                },
                BlitCmd::CopyBufferToBuffer {
                    source_buffer: Some(&from),
                    source_offset: 0,
                    destination_buffer: Some(&buffer),
                    destination_offset: 4,
                    size: 4,
                },
                BlitCmd::EndEncoding,
            ] {
                e.encode(OnFault::Die, cmd).unwrap();
            }
            cb.commit(OnFault::Die).unwrap();
            cb.wait_until_completed(OnFault::Die).unwrap();
            assert!(cb.error().is_none(), "{:?}", cb.error());
            // SAFETY: a Shared buffer's word, read after the command buffer completed.
            unsafe { out.contents().as_ptr::<u32>().read() }
        };
        if unified {
            assert_eq!(reach(), OLD, "the CPU's bytes");
            assert_eq!(read(1), GPU, "the GPU's write, in the CPU's bytes");
            write(0, NEW);
            assert_eq!(
                reach(),
                NEW,
                "the CPU's later write, with no didModifyRange"
            );
        } else {
            assert_eq!(reach(), OLD, "the GPU's copy, as of the first commit");
            assert_eq!(read(1), AFTER, "the GPU's write, in its own copy alone");
            write(0, NEW);
            assert_eq!(reach(), OLD, "the GPU's copy, before a didModifyRange");
            (buffer.did_modify_range(OnFault::Die, NsRange::new(0, 4))).unwrap();
            assert_eq!(reach(), NEW, "the GPU's copy, after a didModifyRange");
        }
        drop((buffer, copy, from, out));
        drop(dec);
        drop(ledger);
        drop(shared_pool);
    }

    /// A `MANAGED_BUFFER_CREATE` that says its CPU copy reads zero from a byte before its length
    /// ([`RgfxManagedBufferZero`]) tells the platform so (`BufferDesc::zero_from`), on any GPU
    /// and wherever the copy lies; one from its length or past it, and one without it (the
    /// message of a guest that sends none), tell it nothing. The buffer is made as without it.
    #[test]
    fn a_managed_buffer_s_zero_bytes_reach_the_platform() {
        let (mut r, ram, page) = pages_rig();
        let copies = shared_copies(&mut r, &ram, page);
        let mut obj_id = 60;
        for (cpu, first) in copies {
            for (zero, want) in [
                (Some(0), Some(0)),
                (Some(0x101), Some(0x101)),
                (Some(0x2ff), Some(0x2ff)),
                (Some(0x300), None),
                (Some(u64::MAX), None),
                (None, None),
            ] {
                let mut message = managed_over(obj_id, cpu, 0x300, MANAGED)
                    .as_bytes()
                    .to_vec();
                if let Some(zero_from) = zero {
                    message.extend_from_slice(RgfxManagedBufferZero { zero_from }.as_bytes());
                }
                let size = u32::try_from(message.len()).unwrap();
                message[4..8].copy_from_slice(&size.to_le_bytes());
                let mark = r.p.call_log().mark();
                assert_eq!(r.dec.decode(&r.device, &message).unwrap(), Ok(()));
                let calls = r.p.call_log().since(mark);
                let news: Vec<_> = calls.iter().filter(|c| c.sel.starts_with("new")).collect();
                assert_eq!(news.len(), 1, "copy {cpu}, zero {zero:?}");
                // SAFETY: the range is only formatted.
                let bytes = unsafe { HostRange::new(HostPtr::from_ptr(first as *mut u8), 0x300) };
                assert_eq!(
                    news[0].args[0],
                    Arg::val(&BufferDesc {
                        length: 0x300,
                        options: vec![OptionSet::ResourceOptions(MANAGED.into())],
                        pinned_gpu_address: Some(0x1_0000_0000 + obj_id * 0x10_0000),
                        contents: Some(bytes),
                        no_copy: true,
                        deallocator: Some(Box::new(|_, _| {})),
                        zero_from: want,
                    }),
                    "copy {cpu}, zero {zero:?}"
                );
                assert!(r.dec.managed_copy(obj_id).is_some());
                obj_id += 1;
            }
        }
    }

    /// A `MANAGED_BUFFER_CREATE` whose buffer the platform refuses is refused, as Metal's answers
    /// nil, with no other buffer made over its CPU copy in its place.
    #[test]
    fn a_managed_buffer_over_a_shared_buffer_the_platform_refuses_is_refused() {
        let (mut r, ram, page) = pages_rig();
        let copies = shared_copies(&mut r, &ram, page);
        for (i, (cpu, _)) in copies.into_iter().enumerate() {
            let obj_id = 60 + i as u64;
            r.p.script().push("newBufferWithDescriptor:", Step::Nil);
            let mark = r.p.call_log().mark();
            assert_eq!(
                r.send(&managed_over(obj_id, cpu, 0x300, MANAGED)),
                Err(DecodeError::Refused),
                "copy {cpu}"
            );
            assert_eq!(
                made(r.p.call_log().sels_since(mark)),
                ["newBufferWithDescriptor:"],
                "copy {cpu}"
            );
            assert!(r.dec.named(obj_id).is_none());
        }
    }

    /// Where the platform refuses a `MANAGED_BUFFER_CREATE`'s buffer, it drops the deallocator
    /// unrun at the refusal (core's `NoCopyDeallocator`), and the create keeps no hold of its own
    /// past it: the CPU copy's buffer the deallocator held goes on the surface worker, wherever the
    /// copy lies, so no handle of the platform's is kept, and the region it held goes with it. The
    /// CPU copy's object is kept, and no handle is dropped inside a deallocator.
    #[test]
    fn a_refused_managed_buffer_lets_go_of_its_cpu_copy_s_buffer_and_its_region() {
        let (mut r, ram, page) = pages_rig();
        let copies = shared_copies(&mut r, &ram, page);
        for (i, (cpu, _)) in copies.into_iter().enumerate() {
            let obj_id = 60 + i as u64;
            let buffer = match r.dec.named(cpu) {
                Some(ObjHandle::Buffer(b)) => Arc::clone(b.obj()),
                _ => panic!("no buffer {cpu:#x}"),
            };
            r.dec.surface_worker().flush();
            let (handles, regions) = (buffer.handle_count(), Arc::strong_count(&r.region));
            r.p.script().push("newBufferWithDescriptor:", Step::Nil);
            assert_eq!(
                r.send(&managed_over(obj_id, cpu, 0x300, MANAGED)),
                Err(DecodeError::Refused),
                "copy {cpu}"
            );
            r.dec.surface_worker().flush();
            assert_eq!(buffer.handle_count(), handles, "copy {cpu}: let go of");
            assert_eq!(
                Arc::strong_count(&r.region),
                regions,
                "copy {cpu}: the region let go of"
            );
            assert!(r.dec.named(cpu).is_some(), "copy {cpu}: its object kept");
        }
        let dropped = r.p.world().handles_dropped_in_deallocators();
        assert!(dropped.is_empty(), "{dropped:?}");
    }

    /// Where the platform refuses a `BUFFER_MANAGED_AT_OFFSET` in a heap over a block of the
    /// guest's Shared pool (`HEAP_OVER_POOL`), the deallocator it drops at the refusal held a keep
    /// of the pool's piece, and the create keeps none past it: the piece is kept by the heap alone,
    /// so once the decoder let go of the heap the pool's release, at its last ledger's drop, waits
    /// on no keep and gives every block back to the source.
    #[test]
    fn a_refused_managed_buffer_in_a_pool_heap_lets_go_of_its_piece() {
        use std::sync::mpsc;
        use std::time::Duration;

        use reims_vgpu_graphics::wire::{RGFX_CMD_HEAP_OVER_POOL, RGFX_CONTEXT_OBJ_ID_SHIFT};
        use reims_vgpu_vulkan::core::platform::os::Clock as _;

        use crate::pool::{BlockSource, HostBlocks, Pool as SharedPool, PoolConfig, PoolHost};

        let mut r = rig(false);
        // The guest's Shared pool over host blocks, and one block of context 1's.
        let source = Arc::new(HostBlocks::new());
        let (added, adds) = mpsc::channel();
        let host = PoolHost {
            add: Arc::new(move |at, _, _| {
                let _ = added.send(at);
            }),
            remove: Arc::new(|_, _| {}),
        };
        let clock = Arc::clone(&r.p);
        let now = Box::new(move || clock.clock().now_mono());
        let range = PoolConfig {
            physical_address: 52 << 30,
            physical_length: 4 << 30,
        };
        let blocks = Arc::clone(&source) as Arc<dyn BlockSource>;
        let shared_pool = SharedPool::new(range, host, now, blocks).expect("a pool");
        let ledger = shared_pool.ledger();
        assert!(ledger.keep_headroom());
        let at = adds.recv_timeout(Duration::from_secs(60)).expect("a piece");
        shared_pool.added(at).unwrap();
        let (taken, takes) = mpsc::channel();
        let answer = move |block| {
            let _ = taken.send(block);
        };
        ledger.take(1, 0x1_0000, Box::new(answer)).unwrap();
        let block = (takes.recv_timeout(Duration::from_secs(60)).unwrap()).expect("a block");
        r.dec.set_pool(ledger.clone());
        r.dec.set_context(Some(1));
        let of_1 = |n: u64| (1 << RGFX_CONTEXT_OBJ_ID_SHIFT) | n;
        let heap = RgfxHeapOverHostMemory {
            header: header(RGFX_CMD_HEAP_OVER_POOL),
            obj_id: of_1(20),
            gpu_address: block.gpu_address,
            region_offset: block.offset,
            size_bytes: 0x1_0000,
            options: 0,
            // MTLHeapTypePlacement.
            heap_type: 1,
        };
        assert_eq!(r.send(&heap), Ok(()));
        let kept = r
            .dec
            .ram_kept_of(of_1(20))
            .expect("the heap's keep of its piece");
        let keeps = Arc::strong_count(&kept);

        r.p.script().push("newBufferWithDescriptor:", Step::Nil);
        let managed = RgfxBufferManagedAtOffset {
            header: header(RGFX_CMD_BUFFER_MANAGED_AT_OFFSET),
            obj_id: of_1(21),
            gpu_address: 0,
            heap_obj_id: of_1(20),
            offset: 0x1000,
            length: 0x100,
            options: MANAGED,
            reserved_3c: 0,
        };
        let mark = r.p.call_log().mark();
        assert_eq!(r.send(&managed), Err(DecodeError::Refused));
        assert_eq!(
            made(r.p.call_log().sels_since(mark)),
            ["newBufferWithDescriptor:"]
        );
        assert_eq!(Arc::strong_count(&kept), keeps, "no keep of the piece kept");
        assert!(r.dec.named(of_1(21)).is_none());

        // The decoder lets go of the heap, and the pool's release waits on no keep.
        drop(kept);
        drop(r);
        drop(shared_pool);
        let (ended, end) = mpsc::channel();
        let release = std::thread::Builder::new().name("x-release".into());
        let release = release.spawn(move || {
            drop(ledger);
            let _ = ended.send(());
        });
        let release = release.expect("a thread");
        assert_eq!(
            end.recv_timeout(Duration::from_secs(10)),
            Ok(()),
            "the pool's release waits on no keep"
        );
        release.join().unwrap();
        assert_eq!(source.counts().0, 0, "every block back");
    }

    /// A `MANAGED_BUFFER_CREATE` names a `Shared` buffer over the guest's memory the sender
    /// holds, of at least its length: none, a heap, a `Managed` buffer, or a shorter one is a bad
    /// reference; a buffer of no bytes is malformed; one not `Managed` is refused.
    #[test]
    fn a_managed_buffer_s_cpu_copy_is_a_shared_buffer_of_its_length() {
        let (mut r, ram, page) = pages_rig();
        let copies = shared_copies(&mut r, &ram, page);
        let cpu = copies[1].0;
        assert_eq!(
            r.buffer(70, pages::BLOCK + 0x2_0000, 0x400, MANAGED),
            Ok(())
        );
        for (copy, length, options, want) in [
            (99, 0x100, MANAGED, Err(DecodeError::BadReference)),
            (20, 0x100, MANAGED, Err(DecodeError::BadReference)),
            (70, 0x100, MANAGED, Err(DecodeError::BadReference)),
            (cpu, 0x401, MANAGED, Err(DecodeError::BadReference)),
            (cpu, 0, MANAGED, Err(DecodeError::Malformed)),
            (cpu, 0x100, 0, Err(DecodeError::Refused)),
            (cpu, 0x400, MANAGED, Ok(())),
        ] {
            assert_eq!(
                r.send(&managed_over(80, copy, length, options)),
                want,
                "copy {copy}, length {length:#x}, options {options:#x}"
            );
        }
    }

    /// On a GPU that shares the CPU's memory a one-level 2D `Managed` texture lies over its CPU
    /// copy's buffer, and its replaced regions and synchronizes copy nothing; a mipmapped one is
    /// still two copies.
    #[test]
    fn a_managed_texture_on_unified_memory_lies_over_its_block_where_it_can() {
        let mut r = rig(true);
        let at = BLOCKS + 0x1_0000;
        assert_eq!(r.buffer(20, at, 0x1_0000, 0), Ok(()));
        let mark = r.p.call_log().mark();
        assert_eq!(r.texture(21, 20, rgba(64, 64, 1)), Ok(()));
        assert_eq!(
            made(r.p.call_log().sels_since(mark)),
            ["newTextureWithDescriptor:offset:bytesPerRow:"]
        );
        let mark = r.p.call_log().mark();
        assert_eq!(r.replaced(21, 0, 0, [0, 0, 0], [64, 64, 1]), Ok(()));
        assert_eq!(
            r.replaced(21, 0, 0, [0, 0, 0], [65, 64, 1]),
            Err(DecodeError::Malformed)
        );
        assert!(r.synchronize_texture(21, 0, 0).unwrap());
        assert_eq!(
            r.p.call_log().sels_since(mark),
            ["synchronizeTexture:slice:level:"]
        );

        assert_eq!(r.buffer(30, at + 0x2_0000, 0x1_0000, 0), Ok(()));
        let mark = r.p.call_log().mark();
        assert_eq!(r.texture(31, 30, rgba(64, 64, 2)), Ok(()));
        assert_eq!(
            made(r.p.call_log().sels_since(mark)),
            ["newTextureWithDescriptor:"]
        );
        let mark = r.p.call_log().mark();
        assert_eq!(r.replaced(31, 1, 0, [0, 0, 0], [32, 32, 1]), Ok(()));
        assert_eq!(
            r.p.call_log().sels_since(mark),
            ["replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:"]
        );
    }

    /// On unified memory a one-level `Managed` 2D array lies over its CPU copy too, by the linear
    /// SPI, its slices at the layout's: rows of 256 bytes, each slice its rows' bytes after the
    /// one before. Its regions and synchronize copy nothing; a mipmapped array is two copies.
    #[test]
    fn a_managed_2d_array_on_unified_memory_lies_over_its_block() {
        let mut r = rig(true);
        let at = BLOCKS + 0x1_0000;
        let array = |levels| RgfxTextureDescriptor {
            texture_type: 3,
            array_length: 3,
            ..rgba(48, 16, levels)
        };
        assert_eq!(r.buffer(20, at, 0x1_0000, 0), Ok(()));
        let mark = r.p.call_log().mark();
        assert_eq!(r.texture(21, 20, array(1)), Ok(()));
        let creates: Vec<_> = (r.p.call_log().since(mark).into_iter())
            .filter(|c| c.sel.starts_with("new"))
            .collect();
        assert_eq!(
            creates.iter().map(|c| c.sel).collect::<Vec<_>>(),
            ["newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:"]
        );
        assert_eq!(
            creates[0].args[1..],
            [Arg::U64(0), Arg::U64(256), Arg::U64(256 * 16)]
        );
        assert!(r.dec.managed_copy(21).unwrap().cpu.is_none());
        let mark = r.p.call_log().mark();
        assert_eq!(r.replaced(21, 0, 2, [0, 0, 0], [48, 16, 1]), Ok(()));
        assert_eq!(
            r.replaced(21, 0, 3, [0, 0, 0], [48, 16, 1]),
            Err(DecodeError::Malformed)
        );
        assert!(r.synchronize_texture(21, 2, 0).unwrap());
        assert_eq!(
            r.p.call_log().sels_since(mark),
            ["synchronizeTexture:slice:level:"]
        );

        assert_eq!(r.buffer(30, at + 0x2_0000, 0x1_0000, 0), Ok(()));
        let mark = r.p.call_log().mark();
        assert_eq!(r.texture(31, 30, array(2)), Ok(()));
        assert_eq!(
            made(r.p.call_log().sels_since(mark)),
            ["newTextureWithDescriptor:"]
        );
        assert!(r.dec.managed_copy(31).unwrap().cpu.is_some());
    }

    /// Where the platform refuses a `Managed` buffer, or a texture's GPU copy, on a GPU of its own
    /// memory, the create is refused, as Metal's answers nil, with nothing else made over the
    /// guest's memory in its place: a `BUFFER_OVER_HOST_MEMORY`, a `BUFFER_MANAGED_AT_OFFSET` in
    /// a heap over host memory, and a `MANAGED_TEXTURE_CREATE` of a shape the platform makes over
    /// a buffer.
    #[test]
    fn a_managed_object_the_platform_refuses_is_refused() {
        let mut r = rig(false);
        let at = BLOCKS + 0x2000;
        r.p.script().push("newBufferWithDescriptor:", Step::Nil);
        let mark = r.p.call_log().mark();
        assert_eq!(r.buffer(10, at, 0x100, MANAGED), Err(DecodeError::Refused));
        assert_eq!(
            made(r.p.call_log().sels_since(mark)),
            ["newBufferWithDescriptor:"]
        );
        assert!(r.dec.named(10).is_none());
        assert!(!r.dec.span_in_use(at, 0x100));

        let heap_at = BLOCKS + 0x4_0000;
        assert_eq!(
            r.send(&RgfxHeapOverHostMemory {
                header: header(RGFX_CMD_HEAP_OVER_HOST_MEMORY),
                obj_id: 30,
                gpu_address: 0x2_0000_0000,
                region_offset: heap_at,
                size_bytes: 0x2_0000,
                options: 0,
                heap_type: 1,
            }),
            Ok(())
        );
        r.p.script().push("newBufferWithDescriptor:", Step::Nil);
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.send(&RgfxBufferManagedAtOffset {
                header: header(RGFX_CMD_BUFFER_MANAGED_AT_OFFSET),
                obj_id: 12,
                gpu_address: 0x1_0000_0000 + 12 * 0x10_0000,
                heap_obj_id: 30,
                offset: 0x3000,
                length: 0x100,
                options: MANAGED,
                reserved_3c: 0,
            }),
            Err(DecodeError::Refused)
        );
        assert_eq!(
            made(r.p.call_log().sels_since(mark)),
            ["newBufferWithDescriptor:"]
        );
        assert!(r.dec.named(12).is_none());

        let cpu_at = BLOCKS + 0x1_0000;
        assert_eq!(r.buffer(20, cpu_at, 0x1_0000, 0), Ok(()));
        r.p.script().push("newTextureWithDescriptor:", Step::Nil);
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.texture(21, 20, rgba(64, 64, 1)),
            Err(DecodeError::Refused)
        );
        assert_eq!(
            made(r.p.call_log().sels_since(mark)),
            ["newTextureWithDescriptor:"]
        );
        assert!(r.dec.named(21).is_none());
    }

    /// What a one-level `Managed` 2D array of 3 slices of 48 x 16 RGBA8 texels shows on `P` on
    /// unified memory, made by the decoder over its CPU copy in a region block: each texel's bytes
    /// in the copy name its slice, row and column, which the guest's `replaceRegion:` of each
    /// slice sends. In order: slice 2 read back by a blit (the GPU
    /// reads the copy where its layout puts the slice: rows 256 bytes apart, slices 16 rows), and
    /// the copy after a blit wrote 0x5a into slice 1 and synchronized it, as a guest does (the GPU
    /// writes it there, slices 0 and 2 kept); first, whether the texture lies over the copy.
    fn managed_array_in_place_on<P: crate::platform::ApiPlatform>() -> (bool, (Vec<u8>, Vec<u8>)) {
        use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder as _};
        use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
        use reims_vgpu_vulkan::core::platform::resource::Buffer as _;

        const W: u64 = 48;
        const H: u64 = 16;
        const ROW: u64 = 256;
        const SLICE: u64 = ROW * H;
        let platform = P::process();
        let device = platform.metal_device(None).unwrap();
        crate::graphics::reserve_guest_texture_ids(platform.as_ref(), &device);
        let size = BLOCKS + 0x10_0000 + RGFX_CURSOR_SPAN;
        let region = Arc::new(crate::platform::test_region(&*platform, size as usize));
        let mut dec = GraphicsDecoder::<P>::new(Arc::clone(&platform), Arc::clone(&region));
        dec.set_unified_memory(true);
        dec.hold_resource_ids(0, 0);
        let at = BLOCKS + 0x1_0000;
        let copy = |offset: u64| region.base().wrapping_add((at + offset) as usize);
        for (s, y, x) in
            (0..3).flat_map(|s| (0..H).flat_map(move |y| (0..W).map(move |x| (s, y, x))))
        {
            let texel = [s as u8, y as u8, x as u8, 0xa5];
            // SAFETY: inside the region's block the buffer below lies over, which no command
            // uses yet.
            unsafe {
                std::ptr::copy_nonoverlapping(texel.as_ptr(), copy(s * SLICE + y * ROW + x * 4), 4)
            };
        }
        let send = |dec: &mut GraphicsDecoder<P>, bytes: &[u8]| {
            assert_eq!(dec.decode(&device, bytes).unwrap(), Ok(()));
        };
        let buffer = RgfxBufferOverHostMemory {
            header: header(RGFX_CMD_BUFFER_OVER_HOST_MEMORY),
            obj_id: 20,
            gpu_address: 0,
            region_offset: at,
            length: 3 * SLICE,
            options: 0,
            ..RgfxBufferOverHostMemory::default()
        };
        send(&mut dec, buffer.as_bytes());
        let texture = RgfxManagedTextureCreate {
            header: header(RGFX_CMD_MANAGED_TEXTURE_CREATE),
            obj_id: 21,
            gpu_address: 0,
            resource_id: 21,
            cpu_buffer_obj_id: 20,
            bytes_per_block: 4,
            block_width: 1,
            block_height: 1,
            flags: RGFX_MANAGED_TEXTURE_LINEAR,
            desc: RgfxTextureDescriptor {
                texture_type: 3,
                array_length: 3,
                ..rgba(W as u32, H as u32, 1)
            },
        };
        send(&mut dec, texture.as_bytes());
        // The guest's replaceRegion: of each slice, which brings the CPU copy's bytes to the
        // texture where it is a copy of its own, and has nothing to bring where it lies over them.
        for slice in 0..3 {
            let replace = RgfxTextureReplaceRegion {
                header: header(RGFX_CMD_TEXTURE_REPLACE_REGION),
                obj_id: 21,
                level: 0,
                slice,
                origin: [0; 3],
                size: [W as u32, H as u32, 1],
            };
            send(&mut dec, replace.as_bytes());
        }
        let in_place = dec.managed_copy(21).unwrap().cpu.is_none();
        let Some(ObjHandle::Texture(t)) = dec.named(21) else {
            panic!("no texture 21");
        };
        let queue = device.new_command_queue().unwrap();
        // A blit pass of `cmds`, then, where given, the guest's synchronizeTexture: of a slice
        // through the decoder's blit op: the platform's own where the texture lies over its CPU
        // copy, else the decoder's copy of the slice into it (`synchronize_managed`).
        let run = |cmds: Vec<BlitCmd<'_, P>>, synchronized: Option<u64>| {
            let cb = queue.command_buffer(OnFault::Return).unwrap().unwrap();
            let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
            for cmd in cmds {
                e.encode(OnFault::Return, cmd).unwrap();
            }
            if let Some(slice) = synchronized {
                let m = RgfxBlitOpSynchronizeTexture {
                    texture_obj_id: 21,
                    slice,
                    level: 0,
                };
                let op = RGFX_OP_BLIT_SYNCHRONIZE_TEXTURE;
                assert!(decode_op(op, m.as_bytes(), &dec, &e, OnFault::Return).unwrap());
            }
            e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
            cb.commit(OnFault::Return).unwrap();
            cb.wait_until_completed(OnFault::Return).unwrap();
            assert!(cb.error().is_none(), "{:?}", cb.error());
        };
        let shared = || {
            device
                .new_buffer_of_length(OnFault::Return, W * H * 4, 0)
                .unwrap()
                .unwrap()
        };
        let out = shared();
        let whole = MtlSize::new(W, H, 1);
        run(
            vec![BlitCmd::CopyTextureToBuffer {
                source_texture: t,
                source_slice: 2,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: whole,
                destination_buffer: &out,
                destination_offset: 0,
                destination_bytes_per_row: W * 4,
                destination_bytes_per_image: W * H * 4,
            }],
            None,
        );
        // SAFETY: a Shared buffer's bytes, read after the blit completed.
        let read = unsafe {
            std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), (W * H * 4) as usize)
        }
        .to_vec();
        let fill = shared();
        // SAFETY: a Shared buffer of `W * H * 4` bytes, which no command uses yet.
        unsafe {
            std::ptr::write_bytes(fill.contents().as_ptr::<u8>(), 0x5a, (W * H * 4) as usize)
        };
        // The guest's synchronizeTexture: of the slice after the write.
        run(
            vec![BlitCmd::CopyBufferToTexture {
                source_buffer: &fill,
                source_offset: 0,
                source_bytes_per_row: W * 4,
                source_bytes_per_image: W * H * 4,
                source_size: whole,
                destination_texture: t,
                destination_slice: 1,
                destination_level: 0,
                destination_origin: MtlOrigin::new(0, 0, 0),
            }],
            Some(1),
        );
        // SAFETY: the region's block, which the blit's writes finished.
        let after =
            unsafe { std::slice::from_raw_parts(copy(0).cast_const(), (3 * SLICE) as usize) }
                .to_vec();
        (in_place, (read, after))
    }

    /// Checks [`managed_array_in_place_on`]'s answers against the layout, the texture over its
    /// CPU copy where `in_place`.
    fn check_managed_array_in_place(in_place: bool, (read, after): (Vec<u8>, Vec<u8>)) {
        let (w, h, row, slice) = (48usize, 16usize, 256usize, 256 * 16usize);
        for y in 0..h {
            for x in 0..w {
                let at = (y * w + x) * 4;
                assert_eq!(
                    read[at..at + 4],
                    [2, y as u8, x as u8, 0xa5],
                    "slice 2 ({x}, {y}), in place: {in_place}"
                );
            }
        }
        for s in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let at = s * slice + y * row + x * 4;
                    let want = match s {
                        1 => [0x5a; 4],
                        _ => [s as u8, y as u8, x as u8, 0xa5],
                    };
                    assert_eq!(
                        after[at..at + 4],
                        want,
                        "the copy's slice {s} ({x}, {y}), in place: {in_place}"
                    );
                }
            }
        }
    }

    /// A one-level `Managed` 2D array on unified memory shows the guest its CPU copy's bytes where
    /// the layout puts its slices: the GPU reads them once the guest's `replaceRegion:` sent them,
    /// and its writes land there once the guest synchronizes. It lies over the copy where rt holds
    /// such a texture over a buffer in place; where rt cannot, it refuses the texture over the
    /// buffer, counted under rt's code (`rt-refused-gap`; measured on a Radeon 780M's RADV), and
    /// the texture is a copy of its own, which the `replaceRegion:` fills.
    #[test]
    fn on_vulkan_a_managed_2d_array_shows_its_cpu_copy() {
        let refused = crate::platform::counted("rt-refused-gap");
        let (in_place, answers) = managed_array_in_place_on::<reims_vgpu_vulkan::RtPlatform>();
        let counted = crate::platform::counted("rt-refused-gap") - refused;
        assert_eq!(counted, u64::from(!in_place), "in place: {in_place}");
        check_managed_array_in_place(in_place, answers);
    }
}
