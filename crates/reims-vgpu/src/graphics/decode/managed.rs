//! `Managed` storage (docs/graphics-kext/design.md, Decided 4): a buffer's or a texture's CPU
//! copy is memory of its process's the guest maps, and the range messages tell the host what the
//! CPU wrote. A buffer's is a region block, a range of a heap over a block or over a pages block's
//! guest RAM, or the contents of a `Shared` buffer the guest names (`MANAGED_BUFFER_CREATE`); a
//! texture's is the contents of a `Shared` buffer the guest names. Those lie wherever the buffer's
//! contents lie in the guest's memory ([`GuestBytes`]), so a guest that took
//! `RGFX_FEATURE_MANAGED_TEXTURE_SLABS` and `RGFX_FEATURE_MANAGED_BUFFER_SLABS` lays them in its
//! slabs, its own pages among them.
//!
//! - **A GPU that shares the CPU's memory** (the platform's answer, [`GraphicsDecoder::
//!   set_unified_memory`]): one copy. A buffer is made over its block in place, as a `Shared`
//!   one is; a texture too, over the Shared buffer of its block, where a texture over a buffer
//!   can have its shape ([`in_place_shape`]). The range messages are checked and order nothing
//!   more: the platform sees the block's bytes at each command buffer's commit.
//! - **Any other GPU**: two copies, as Metal keeps a discrete GPU's `Managed` resources. The
//!   object the command buffers use is the platform's own `Managed` buffer or texture, and its
//!   CPU copy is the block, through a `Shared` buffer over it.
//!   - `BUFFER_DID_MODIFY_RANGE` copies the range from the block to the platform buffer's host
//!     copy (`contents`), and `didModifyRange:` has the platform copy it to the GPU's by its next
//!     submission, before any of its commands. A texture's `TEXTURE_REPLACE_REGION` is the
//!     platform's `replaceRegion:` from the block, which is done before the call returns.
//!   - Either is checked on ring 0 where it comes, and carried out once the process's command
//!     buffers answered `CB_SUBMITTED` before it are committed, with no wait on ring 0
//!     ([`GraphicsDecoder::after_submitted`]); every command buffer of the process begun after it,
//!     on any queue, is committed after it, so it is ordered before every command buffer
//!     committed after it, as Metal promises.
//!   - `synchronizeResource:` and `synchronizeTexture:slice:level:` copy the GPU's copy to the
//!     block where the blit encoder runs them ([`GraphicsDecoder::synchronize_managed`]), so the
//!     CPU sees it once the command buffer completes.
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
//! **Where the platform refuses a GPU's copy**, the create is refused, as Metal's answers nil.
//! Metal answers no `Managed` resource nil for want of the GPU's memory alone: on the AMD Radeon
//! Pro 5300M (4 GiB, recommendedMaxWorkingSetSize 4080 MiB) 12 GiB of `Managed` buffers, of 1 MiB
//! and of 64 MiB, were all made, used in one command buffer, which completed, and read back as the
//! GPU wrote them (measured, shmem `-use -check`). The platform places them past the GPU's memory
//! as Metal does (the Vulkan platform's runtime in system memory, past a full device-local heap),
//! so its refusal comes only where that memory is gone too.

use std::ptr;
use std::sync::Arc;

use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder};
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::resource::{Buffer, Heap, Texture};
use reims_vgpu_core::platform::types::{MtlOrigin, MtlRegion, MtlSize, NsRange};
use reims_vgpu_core::platform::values::{BufferDesc, OptionSet};
use reims_vgpu_core::platform::{Handle, HostRange, Platform};
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use reims_vgpu_graphics::managed::LinearLayout;
use reims_vgpu_graphics::wire::{
    RGFX_MANAGED_TEXTURE_LINEAR, RgfxBufferDidModifyRange, RgfxBufferManagedAtOffset,
    RgfxBufferOverHostMemory, RgfxManagedBufferCreate, RgfxManagedTextureCreate,
    RgfxTextureReplaceRegion,
};

use super::tables::{Entry, GuestBytes};
use super::{DecodeError, Drain, GraphicsDecoder, ObjHandle, op_fault, read, texture_desc};

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
    /// A `Shared` buffer over the block, where the object is a copy of its own (the two-copy
    /// form); `None` where the object lies over the block.
    pub(super) cpu: Option<P::Buffer>,
    /// A texture's layout of the block; `None` for a buffer.
    pub(super) layout: Option<LinearLayout>,
    /// The first byte of the heap over a pages block's guest RAM the CPU copy lies in
    /// (`HEAP_OVER_PAGES`); `None` where it lies in the region.
    pub(super) ram: Option<HostPtr>,
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
    /// Whether `Managed` resources are one copy, over their blocks: the platform's answer of
    /// whether the device's GPU shares the CPU's memory (`hasUnifiedMemory`). The decoder starts
    /// with no, the two copies, which hold Metal's contract on any GPU.
    pub(crate) fn set_unified_memory(&mut self, unified: bool) {
        (self.shared.unified_memory).store(unified, std::sync::atomic::Ordering::Release);
        // A GPU of its own memory caches every surface the kext may (`surface_cache`): its
        // transfer queue is made with the first command queue.
        self.surface_caches.set_early(!unified);
    }

    /// Whether `Managed` resources are one copy ([`Self::set_unified_memory`]).
    fn unified(&self) -> bool {
        (self.shared.unified_memory).load(std::sync::atomic::Ordering::Acquire)
    }

    /// The CPU copy of the `Managed` buffer or texture `obj_id` names, where the current context
    /// may name it.
    pub(super) fn managed_copy(&self, obj_id: u64) -> Option<&ManagedCopy<P>> {
        self.entry(obj_id).and_then(|e| e.managed.as_ref())
    }

    /// A `Shared` buffer over the `length` bytes at region offset `at`, which the caller checked
    /// lie in a block of the sender's, of `options`, at the guest's `pin` where it has one (a
    /// BUFFER_CREATE's alike); it keeps the region mapped until the platform frees it.
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
        })
    }

    /// `BUFFER_OVER_HOST_MEMORY` of a `Managed` buffer, whose block the caller checked is the
    /// sender's: over the block in place where the GPU shares the CPU's memory, else the
    /// platform's own `Managed` buffer at the guest's address, its host copy filled from the block
    /// and copied to the GPU's by the next submission. Refused where the platform refuses either.
    pub(super) fn managed_buffer(
        &self,
        device: &P::Device,
        m: &RgfxBufferOverHostMemory,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let options = u64::from(m.options);
        let pin = (m.gpu_address != 0).then_some(m.gpu_address);
        let (buffer, cpu) = match self.unified() {
            false => {
                let Some(buffer) = self.gpu_copy(device, m.length, options, pin)? else {
                    return Ok(Err(DecodeError::Refused));
                };
                let Some(cpu) = self.buffer_over_block(
                    device,
                    m.region_offset,
                    m.length,
                    shared(options),
                    None,
                )?
                else {
                    return Ok(Err(DecodeError::Refused));
                };
                if !self.fill_gpu_copy(&buffer, None, m.region_offset, m.length)? {
                    return Ok(Err(DecodeError::Refused));
                }
                (buffer, Some(cpu))
            }
            true => {
                match self.buffer_over_block(device, m.region_offset, m.length, options, pin)? {
                    Some(buffer) => (buffer, None),
                    None => return Ok(Err(DecodeError::Refused)),
                }
            }
        };
        self.store_managed_buffer(m.obj_id, buffer, m.region_offset, m.length, cpu);
        Ok(Ok(()))
    }

    /// `BUFFER_MANAGED_AT_OFFSET`: the platform's own `Managed` buffer at the guest's address, its
    /// CPU copy a `Shared` buffer placed at `offset` in the guest's heap over host memory, or over
    /// a pages block's guest RAM (`HEAP_OVER_PAGES`), so the buffer imports no host memory of its
    /// own. Refused where the platform refuses the GPU's copy. Offered where `Managed` is two
    /// copies (`RGFX_FEATURE_MANAGED_AT_OFFSET`), and refused on a GPU that shares the CPU's
    /// memory. A heap the sender does not hold, or one over no block of the region, is a bad
    /// reference; a CPU copy past the heap's memory is malformed.
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
        let (ram, at, span) = match ram {
            Some((first, _)) => (Some(first), m.offset, (heap_at, heap_length)),
            None => (None, heap_at + m.offset, (heap_at + m.offset, m.length)),
        };
        let Some(ObjHandle::Heap(heap)) = self.named(m.heap_obj_id) else {
            return Ok(Err(DecodeError::BadReference));
        };
        let pin = (m.gpu_address != 0).then_some(m.gpu_address);
        let Some(buffer) = self.gpu_copy(device, m.length, options, pin)? else {
            return Ok(Err(DecodeError::Refused));
        };
        let Some(cpu) = heap.new_buffer_of_length(m.length, shared(options), m.offset)? else {
            return Ok(Err(DecodeError::Refused));
        };
        if !self.fill_gpu_copy(&buffer, ram, at, m.length)? {
            return Ok(Err(DecodeError::Refused));
        }
        let copy = ManagedCopy {
            at,
            length: m.length,
            cpu: Some(cpu),
            layout: None,
            ram,
        };
        self.store_managed(m.obj_id, buffer, span, copy);
        Ok(Ok(()))
    }

    /// `MANAGED_BUFFER_CREATE`: a `Managed` buffer at the guest's address whose CPU copy is the
    /// first `length` bytes of a `Shared` buffer over the guest's memory the sender holds,
    /// wherever they lie ([`GuestBytes`]): over a block, placed in a heap over one or over a pages
    /// block's guest RAM, or over a pages block's view. Where the GPU shares the CPU's memory, one
    /// copy: a buffer of its own over those bytes, which keeps the CPU copy's buffer, and so a
    /// view its bytes lie in, until the platform frees it. Else two: the platform's own `Managed`
    /// buffer, filled from the bytes, with the CPU copy's buffer kept as its copy, refused where
    /// the platform refuses that buffer. The buffer lies over what its CPU copy lies over. A CPU
    /// copy that is no such buffer (a `Managed` one, one of the GPU's own memory, a refused one),
    /// or is shorter than the buffer, is a bad reference; a buffer of no bytes is malformed, and
    /// one of options other than `Managed` is refused.
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
        let cpu = cpu.clone_ref();
        let pin = (m.gpu_address != 0).then_some(m.gpu_address);
        let (buffer, cpu) = match self.unified() {
            false => {
                let Some(buffer) = self.gpu_copy(device, m.length, options, pin)? else {
                    return Ok(Err(DecodeError::Refused));
                };
                if !self.fill_gpu_copy(&buffer, bytes.ram, bytes.at, m.length)? {
                    return Ok(Err(DecodeError::Refused));
                }
                (buffer, Some(cpu))
            }
            true => {
                let Some(buffer) =
                    self.buffer_over_bytes(device, bytes, m.length, options, pin, cpu)?
                else {
                    return Ok(Err(DecodeError::Refused));
                };
                (buffer, None)
            }
        };
        let copy = ManagedCopy {
            at: bytes.at,
            length: m.length,
            cpu,
            layout: None,
            ram: bytes.ram,
        };
        let mut entry = Entry::of(ObjHandle::Buffer(buffer)).over(span);
        entry.managed = Some(copy);
        self.store(m.obj_id, entry);
        Ok(Ok(()))
    }

    /// A no-copy buffer of `options` over the first `length` bytes of a `Shared` buffer's
    /// contents, `bytes`, at the guest's `pin`, which keeps `keep`, that buffer, until the
    /// platform frees it, so a pages block's view its bytes lie in stays mapped: a `Managed`
    /// buffer of one copy over its CPU copy.
    fn buffer_over_bytes(
        &self,
        device: &P::Device,
        bytes: GuestBytes,
        length: u64,
        options: u64,
        pin: Option<u64>,
        keep: P::Buffer,
    ) -> RaiseResult<Option<P::Buffer>> {
        let region = Arc::clone(&self.region);
        let first = match bytes.ram {
            Some(first) => first.wrapping_add(bytes.at),
            None => HostPtr::from_ptr(region.base().wrapping_add(bytes.at as usize)),
        };
        // SAFETY: the bytes lie in the Shared buffer's contents (the caller checked `length`
        // against them): in the guest's RAM mapping, which stays mapped as long as the device, in
        // a view `keep` keeps mapped, or in the region, which `region` keeps mapped; the
        // deallocator holds both until the platform frees the buffer.
        let range = unsafe { HostRange::new(first, length) };
        device.new_buffer(BufferDesc {
            length,
            options: vec![OptionSet::ResourceOptions(options)],
            pinned_gpu_address: pin,
            contents: Some(range),
            no_copy: true,
            deallocator: Some(Box::new(move |_, _| drop((keep, region)))),
        })
    }

    /// The platform's own `Managed` buffer of `length` bytes and `options`, at `pin` where the
    /// guest has an address for it: a two-copy buffer's GPU copy. `None` where the platform
    /// refuses it.
    fn gpu_copy(
        &self,
        device: &P::Device,
        length: u64,
        options: u64,
        pin: Option<u64>,
    ) -> RaiseResult<Option<P::Buffer>> {
        device.new_buffer(BufferDesc {
            length,
            options: vec![OptionSet::ResourceOptions(options)],
            pinned_gpu_address: pin,
            contents: None,
            no_copy: false,
            deallocator: None,
        })
    }

    /// A CPU copy's byte at `at`: from `ram`, a heap over pages' first byte, where given, else in
    /// the region.
    fn cpu_bytes(&self, ram: Option<HostPtr>, at: u64) -> *const u8 {
        match ram {
            Some(first) => first.wrapping_add(at).as_ptr::<u8>(),
            None => self.region.base().wrapping_add(at as usize),
        }
    }

    /// Fills a GPU copy's host copy from the `length` bytes at `at` from `ram`, a heap over
    /// pages' first byte, or in the region where `None`, which the caller checked lie in the
    /// heap's memory or in a block of the sender's, and has the platform copy them to the GPU's
    /// by the next submission. False where the buffer has no host copy.
    fn fill_gpu_copy(
        &self,
        buffer: &P::Buffer,
        ram: Option<HostPtr>,
        at: u64,
        length: u64,
    ) -> RaiseResult<bool> {
        let contents = buffer.contents();
        if contents.is_null() {
            return Ok(false);
        }
        // SAFETY: the bytes lie inside the guest's RAM mapping, which stays mapped as long as the
        // device, or inside the region (checked by the caller), and the platform buffer's host
        // copy holds its `length` bytes; the guest's concurrent writes change only what is copied.
        unsafe {
            ptr::copy_nonoverlapping(
                self.cpu_bytes(ram, at),
                contents.as_ptr::<u8>(),
                length as usize,
            );
        }
        buffer.did_modify_range(OnFault::Return, NsRange::new(0, length))?;
        Ok(true)
    }

    /// Stores the `Managed` buffer `obj_id`, whose CPU copy is the `length` bytes at `at` in the
    /// region: `cpu`, a `Shared` buffer over them, where `buffer` is a copy of its own.
    fn store_managed_buffer(
        &self,
        obj_id: u64,
        buffer: P::Buffer,
        at: u64,
        length: u64,
        cpu: Option<P::Buffer>,
    ) {
        let copy = ManagedCopy {
            at,
            length,
            cpu,
            layout: None,
            ram: None,
        };
        self.store_managed(obj_id, buffer, (at, length), copy);
    }

    /// Stores the `Managed` buffer `obj_id`, lying over the region's `span`, and its CPU copy.
    fn store_managed(
        &self,
        obj_id: u64,
        buffer: P::Buffer,
        span: (u64, u64),
        copy: ManagedCopy<P>,
    ) {
        let mut entry = Entry::of(ObjHandle::Buffer(buffer)).over(Some(span));
        entry.managed = Some(copy);
        self.store(obj_id, entry);
    }

    /// `BUFFER_DID_MODIFY_RANGE`: the range copied from the block to the platform buffer's host
    /// copy, where the buffer has one, and `didModifyRange:`, after the context's buffers answered
    /// before it are committed ([`Self::after_submitted`], which `drain` is held through). A bad
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
        let (buffer, region) = (buffer.clone_ref(), Arc::clone(&self.region));
        let (ram, from) = (copy.ram, copy.at + m.location);
        // The CPU copy's buffer keeps a pages block's view mapped, where its bytes lie in one.
        let keep = copy.cpu.as_ref().map(|cpu| cpu.clone_ref());
        self.after_submitted("BUFFER_DID_MODIFY_RANGE", m.obj_id, drain, move || {
            if keep.is_some() {
                let contents = buffer.contents();
                let source = match ram {
                    Some(first) => first.wrapping_add(from).as_ptr::<u8>(),
                    None => region.base().wrapping_add(from as usize),
                };
                // SAFETY: the range lies inside the CPU copy (checked against its length): inside
                // the guest's RAM mapping, which stays mapped as long as the device, inside a view
                // `keep` keeps mapped, or inside the region, which `region` keeps mapped; and
                // inside the platform buffer's host copy of the same length.
                unsafe {
                    ptr::copy_nonoverlapping(
                        source,
                        contents.as_ptr::<u8>().add(m.location as usize),
                        m.length as usize,
                    );
                }
            }
            buffer.did_modify_range(OnFault::Return, NsRange::new(m.location, m.length))
        });
        Ok(Ok(()))
    }

    /// `MANAGED_TEXTURE_CREATE`: over the CPU copy's buffer in place where the GPU shares the
    /// CPU's memory and the platform makes a texture over a buffer of the shape, else the
    /// platform's own `Managed` texture beside it, refused where the platform refuses that one.
    /// The CPU copy is the buffer's contents wherever they lie in the guest's memory
    /// ([`GuestBytes`]): over a block of the sender's, placed in a heap over one or over a pages
    /// block's guest RAM, or over a pages block's view. A bad reference for a CPU copy that is no
    /// such buffer (a `Managed` one, or one of the GPU's own memory), or is shorter than the
    /// layout.
    pub(super) fn managed_texture(
        &self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Result<(), DecodeError>> {
        let Some(m) = read::<RgfxManagedTextureCreate>(message) else {
            return Ok(Err(DecodeError::Malformed));
        };
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
                return self.store_managed_texture(m.obj_id, texture, span, bytes, None, layout);
            }
            log_two_copies_on_unified_memory(&m);
        }
        let descriptor = self.platform.texture_descriptor(&desc);
        match device.new_texture(OnFault::Return, &descriptor)? {
            Some(texture) => {
                self.store_managed_texture(m.obj_id, texture, span, bytes, Some(cpu), layout)
            }
            None => Ok(Err(DecodeError::Refused)),
        }
    }

    /// Stores a `Managed` texture lying over `span`, and its CPU copy, the guest's `bytes`.
    fn store_managed_texture(
        &self,
        obj_id: u64,
        texture: P::Texture,
        span: Option<(u64, u64)>,
        bytes: GuestBytes,
        cpu: Option<P::Buffer>,
        layout: LinearLayout,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let mut entry = Entry::of(ObjHandle::Texture(texture)).over(span);
        entry.managed = Some(ManagedCopy {
            at: bytes.at,
            length: layout.size(),
            cpu,
            layout: Some(layout),
            ram: bytes.ram,
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

    /// `synchronizeResource:` (`which` `None`) or `synchronizeTexture:slice:level:` of the
    /// `Managed` object `obj_id`, where it is a copy of its own: the GPU's copy copied to the
    /// block in `enc`, there. False, encoding nothing, for any other object, which the platform's
    /// own synchronize takes. A slice or level past the texture faults the exec.
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
        match (self.named(obj_id), &copy.layout) {
            (Some(ObjHandle::Buffer(buffer)), None) if which.is_none() => {
                enc.encode(
                    on,
                    BlitCmd::CopyBufferToBuffer {
                        source_buffer: Some(buffer),
                        source_offset: 0,
                        destination_buffer: Some(cpu),
                        destination_offset: 0,
                        size: copy.length,
                    },
                )?;
            }
            (Some(ObjHandle::Texture(texture)), Some(layout)) => {
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
                            destination_bytes_per_image: if three_d {
                                l.bytes_per_image
                            } else {
                                0
                            },
                        },
                    )?;
                }
            }
            _ => return Ok(false),
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
    use reims_vgpu_core::testkit::{Arg, FakePlatform, GuestRam, Step};
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
        })
    }

    /// On a GPU that does not share the CPU's memory a `Managed` buffer is two copies: the
    /// platform's own `Managed` buffer at the guest's address, filled from the block at its
    /// create, and the block. A range message copies exactly its range from the block and tells
    /// the platform; one past the buffer, or of a buffer not `Managed`, is refused; a
    /// synchronize copies the buffer to the block in the encoder.
    #[test]
    fn a_managed_buffer_on_a_discrete_gpu_is_two_copies() {
        let mut r = rig(false);
        let at = BLOCKS + 0x1000;
        let first: Vec<u8> = (0..0x100u32).map(|i| i as u8).collect();
        assert!(r.region.write(at, &first));
        let mark = r.p.call_log().mark();
        assert_eq!(r.buffer(10, at, 0x100, MANAGED), Ok(()));
        let calls = r.p.call_log().since(mark);
        let sels: Vec<&str> = calls.iter().map(|c| c.sel).collect();
        assert_eq!(
            sels,
            [
                "newBufferWithDescriptor:",
                "newBufferWithDescriptor:",
                "didModifyRange:"
            ]
        );
        // The CPU copy's buffer is Shared, over the block, at an address of the platform's.
        assert_eq!(calls[1].args[0], over_block(&r, at, 0x100, 0, None));
        assert_eq!(calls[2].args[0], Arg::val(&NsRange::new(0, 0x100)));
        let buffer = r.dec.require_buffer(10).unwrap().clone_ref();
        assert_eq!(buffer.gpu_address(), 0x1_0000_0000 + 10 * 0x10_0000);
        let host = |b: &<FakePlatform as Platform>::Buffer, off: usize, len: usize| {
            // SAFETY: the fake's buffer holds its length of bytes.
            unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>().add(off), len) }
                .to_vec()
        };
        assert_eq!(host(&buffer, 0, 0x100), first);

        // The guest writes 32..48 and 64, and names 32..48 alone.
        assert!(r.region.write(at + 32, &[0xee; 16]));
        assert!(r.region.write(at + 64, &[0xdd]));
        let mark = r.p.call_log().mark();
        assert_eq!(r.modified(10, 32, 16), Ok(()));
        assert_eq!(host(&buffer, 32, 16), [0xee; 16]);
        assert_eq!(host(&buffer, 64, 1), [64], "only the named range is copied");
        let calls = r.p.call_log().since(mark);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].sel, "didModifyRange:");
        assert_eq!(calls[0].args[0], Arg::val(&NsRange::new(32, 16)));

        // A range past the buffer, and a Shared buffer's range, are refused.
        assert_eq!(r.modified(10, 0xf8, 16), Err(DecodeError::Malformed));
        assert_eq!(r.buffer(11, at + 0x1000, 0x100, 0), Ok(()));
        assert_eq!(r.modified(11, 0, 16), Err(DecodeError::BadReference));
        assert_eq!(r.modified(12, 0, 16), Err(DecodeError::BadReference));

        // A synchronize copies the GPU's copy to the block, and takes no platform synchronize.
        let mark = r.p.call_log().mark();
        assert!(r.synchronize(10, RGFX_BLIT_RESOURCE_BUFFER).unwrap());
        let calls = r.p.call_log().since(mark);
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].sel,
            "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"
        );
        assert_eq!(
            calls[0].args,
            [
                r.obj(10),
                Arg::U64(0),
                r.cpu_of(10),
                Arg::U64(0),
                Arg::U64(0x100)
            ]
        );
        // A Shared buffer's synchronize is the platform's.
        let mark = r.p.call_log().mark();
        assert!(r.synchronize(11, RGFX_BLIT_RESOURCE_BUFFER).unwrap());
        assert_eq!(r.p.call_log().sels_since(mark), ["synchronizeResource:"]);

        // A destroy forgets its CPU copy; the block's span goes with it.
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

    /// On a GPU that shares the CPU's memory a `Managed` buffer is one copy, over its block, and
    /// its range messages and synchronize copy nothing.
    #[test]
    fn a_managed_buffer_on_unified_memory_lies_over_its_block() {
        let mut r = rig(true);
        let at = BLOCKS + 0x2000;
        let mark = r.p.call_log().mark();
        assert_eq!(r.buffer(10, at, 0x100, MANAGED), Ok(()));
        let calls = r.p.call_log().since(mark);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].sel, "newBufferWithDescriptor:");
        let pin = 0x1_0000_0000 + 10 * 0x10_0000;
        assert_eq!(
            calls[0].args[0],
            over_block(&r, at, 0x100, MANAGED.into(), Some(pin))
        );
        let buffer = r.dec.require_buffer(10).unwrap();
        assert_eq!(buffer.gpu_address(), pin);
        assert_eq!(
            buffer.contents().addr(),
            r.region.base() as usize + at as usize
        );
        let mark = r.p.call_log().mark();
        assert_eq!(r.modified(10, 0, 0x100), Ok(()));
        assert_eq!(r.modified(10, 0, 0x101), Err(DecodeError::Malformed));
        assert!(r.synchronize(10, RGFX_BLIT_RESOURCE_BUFFER).unwrap());
        assert_eq!(
            r.p.call_log().sels_since(mark),
            ["didModifyRange:", "synchronizeResource:"]
        );
    }

    /// `BUFFER_MANAGED_AT_OFFSET` on a GPU of its own memory: the platform's own `Managed`
    /// buffer at the guest's address, filled from the heap's bytes at the offset, beside a CPU
    /// copy placed in the heap there, with no buffer of host memory of its own. Its range
    /// messages and synchronize work as a two-copy buffer's; the heap's block stays in use
    /// while it lives. A heap the sender lacks or one of device memory, a range past the heap's
    /// block, a buffer not `Managed`, and the message on unified memory are refused.
    #[test]
    fn a_managed_buffer_at_an_offset_places_its_cpu_copy_in_the_heap() {
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
        assert_eq!(
            made(r.p.call_log().sels_since(mark)),
            [
                "newBufferWithDescriptor:",
                "newBufferWithLength:options:offset:"
            ],
            "only the GPU copy is a buffer of its own; the CPU copy is placed in the heap"
        );
        let buffer = r.dec.require_buffer(10).unwrap().clone_ref();
        assert_eq!(buffer.gpu_address(), 0x1_0000_0000 + 10 * 0x10_0000);
        let host = |b: &<FakePlatform as Platform>::Buffer, off: usize, len: usize| {
            // SAFETY: the fake's buffer holds its length of bytes.
            unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>().add(off), len) }
                .to_vec()
        };
        assert_eq!(host(&buffer, 0, 0x100), first);
        let copy = r.dec.managed_copy(10).unwrap();
        assert_eq!((copy.at, copy.length), (at + 0x3000, 0x100));
        assert!(r.dec.span_in_use(at + 0x3000, 0x100));

        // A range copies from the heap's bytes at the offset; a synchronize goes to the CPU copy.
        assert!(r.region.write(at + 0x3000 + 16, &[0xab; 8]));
        assert_eq!(r.modified(10, 16, 8), Ok(()));
        assert_eq!(host(&buffer, 16, 8), [0xab; 8]);
        let mark = r.p.call_log().mark();
        assert!(r.synchronize(10, RGFX_BLIT_RESOURCE_BUFFER).unwrap());
        let calls = r.p.call_log().since(mark);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].args[2], r.cpu_of(10));

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

    /// `BUFFER_MANAGED_AT_OFFSET` in a heap over a pages block's guest RAM (`HEAP_OVER_PAGES`):
    /// the GPU copy is filled from the guest's page at the offset, and range messages copy from
    /// it, not from the region, while a synchronize goes to the CPU copy placed in the heap. The
    /// buffer lies over the pages block; a CPU copy past the heap's memory is malformed.
    #[test]
    fn a_managed_buffer_in_a_heap_over_pages_reads_the_guest_s_ram() {
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
            [
                "newBufferWithDescriptor:",
                "newBufferWithLength:options:offset:"
            ]
        );
        let buffer = r.dec.require_buffer(10).unwrap().clone_ref();
        let host = |b: &<FakePlatform as Platform>::Buffer, off: usize, len: usize| {
            // SAFETY: the fake's buffer holds its length of bytes.
            unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>().add(off), len) }
                .to_vec()
        };
        assert_eq!(host(&buffer, 0, 0x100), first);
        let copy = r.dec.managed_copy(10).unwrap();
        let heap_first = ram.ranges()[0].host().wrapping_add(2 * page);
        assert_eq!(
            (copy.at, copy.length, copy.ram),
            (0x3000, 0x100, Some(heap_first))
        );
        assert!(r.dec.span_in_use(pages::BLOCK, 0x2000));

        assert!(ram.write(pages::RAM + 2 * page + 0x3000 + 16, &[0xab; 8]));
        assert_eq!(r.modified(10, 16, 8), Ok(()));
        assert_eq!(host(&buffer, 16, 8), [0xab; 8]);
        let mark = r.p.call_log().mark();
        assert!(r.synchronize(10, RGFX_BLIT_RESOURCE_BUFFER).unwrap());
        let calls = r.p.call_log().since(mark);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].args[2], r.cpu_of(10));
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

    /// Writes `bytes` at host address `at`, as the guest's CPU writes its copy.
    fn write_at(at: usize, bytes: &[u8]) {
        // SAFETY: the callers' addresses are a live buffer's contents, of at least the bytes'
        // length, which nothing else reads meanwhile.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), at as *mut u8, bytes.len()) };
    }

    /// `MANAGED_BUFFER_CREATE` on a GPU of its own memory: the platform's own `Managed` buffer
    /// at the guest's address, filled from its CPU copy's buffer's bytes wherever they lie (in a
    /// heap over a pages block's guest RAM, in a heap over a block of the region, in a pages
    /// block's view), beside that buffer as its CPU copy, with no buffer of its own over the
    /// guest's memory. Range messages copy from the CPU copy's bytes, a synchronize goes to the
    /// CPU copy's buffer, and the buffer lies over what its CPU copy lies over, after the CPU
    /// copy's destroy too.
    #[test]
    fn a_managed_buffer_over_a_shared_buffer_is_two_copies_on_a_discrete_gpu() {
        let (mut r, ram, page) = pages_rig();
        let copies = shared_copies(&mut r, &ram, page);
        let host = |b: &<FakePlatform as Platform>::Buffer, off: usize, len: usize| {
            // SAFETY: the fake's buffer holds its length of bytes.
            unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>().add(off), len) }
                .to_vec()
        };
        for (i, (cpu, first)) in copies.into_iter().enumerate() {
            let obj_id = 60 + i as u64;
            let seed: Vec<u8> = (0..0x300u32).map(|b| (b as u8) ^ (obj_id as u8)).collect();
            write_at(first, &seed);
            let mark = r.p.call_log().mark();
            assert_eq!(r.send(&managed_over(obj_id, cpu, 0x300, MANAGED)), Ok(()));
            assert_eq!(
                made(r.p.call_log().sels_since(mark)),
                ["newBufferWithDescriptor:"],
                "copy {cpu}: only the GPU copy is a buffer of its own"
            );
            let buffer = r.dec.require_buffer(obj_id).unwrap().clone_ref();
            assert_eq!(buffer.gpu_address(), 0x1_0000_0000 + obj_id * 0x10_0000);
            assert_eq!(host(&buffer, 0, 0x300), seed, "copy {cpu}");
            assert_eq!(r.cpu_of(obj_id), r.obj(cpu));

            // A range copies from the CPU copy's bytes, that range alone.
            write_at(first + 16, &[0xab; 8]);
            write_at(first + 64, &[0xcd]);
            assert_eq!(r.modified(obj_id, 16, 8), Ok(()));
            assert_eq!(host(&buffer, 16, 8), [0xab; 8], "copy {cpu}");
            assert_eq!(host(&buffer, 64, 1), [seed[64]], "copy {cpu}");
            assert_eq!(r.modified(obj_id, 0x2f8, 16), Err(DecodeError::Malformed));
            let mark = r.p.call_log().mark();
            assert!(r.synchronize(obj_id, RGFX_BLIT_RESOURCE_BUFFER).unwrap());
            let calls = r.p.call_log().since(mark);
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].args[2], r.obj(cpu), "copy {cpu}");
        }
        // The buffer lies over its CPU copy's block, and keeps its CPU copy's buffer when that
        // goes first.
        assert_eq!(
            r.send(&RgfxObjectDestroy {
                header: header(RGFX_CMD_OBJECT_DESTROY),
                obj_id: 41,
            }),
            Ok(())
        );
        assert!(r.dec.span_in_use(pages::BLOCK + 0x1_0000, 0x1_0000));
        let first = copies[1].1;
        write_at(first + 32, &[0x5a; 4]);
        assert_eq!(r.modified(61, 32, 4), Ok(()));
        let buffer = r.dec.require_buffer(61).unwrap().clone_ref();
        assert_eq!(host(&buffer, 32, 4), [0x5a; 4]);
        assert!(r.synchronize(61, RGFX_BLIT_RESOURCE_BUFFER).unwrap());
    }

    /// `MANAGED_BUFFER_CREATE` on a GPU that shares the CPU's memory: one copy, a no-copy buffer
    /// of the `Managed` options over its CPU copy's buffer's bytes at the guest's address, which
    /// keeps that buffer. Its range messages and synchronize copy nothing.
    #[test]
    fn a_managed_buffer_over_a_shared_buffer_on_unified_memory_is_one_copy() {
        let (mut r, ram, page) = pages_rig();
        r.dec.set_unified_memory(true);
        let copies = shared_copies(&mut r, &ram, page);
        for (i, (cpu, first)) in copies.into_iter().enumerate() {
            let obj_id = 60 + i as u64;
            let mark = r.p.call_log().mark();
            assert_eq!(r.send(&managed_over(obj_id, cpu, 0x300, MANAGED)), Ok(()));
            let calls = r.p.call_log().since(mark);
            let news: Vec<_> = calls.iter().filter(|c| c.sel.starts_with("new")).collect();
            let pin = 0x1_0000_0000 + obj_id * 0x10_0000;
            let over = news.last().unwrap();
            // SAFETY: the range is only formatted.
            let bytes = unsafe { HostRange::new(HostPtr::from_ptr(first as *mut u8), 0x300) };
            assert_eq!(
                over.args[0],
                Arg::val(&BufferDesc {
                    length: 0x300,
                    options: vec![OptionSet::ResourceOptions(MANAGED.into())],
                    pinned_gpu_address: Some(pin),
                    contents: Some(bytes),
                    no_copy: true,
                    deallocator: Some(Box::new(|_, _| {})),
                }),
                "copy {cpu}"
            );
            let buffer = r.dec.require_buffer(obj_id).unwrap();
            assert_eq!(buffer.gpu_address(), pin);
            assert_eq!(buffer.contents().addr(), first);
            assert!(r.dec.managed_copy(obj_id).unwrap().cpu.is_none());
            let mark = r.p.call_log().mark();
            assert_eq!(r.modified(obj_id, 0, 0x300), Ok(()));
            assert!(r.synchronize(obj_id, RGFX_BLIT_RESOURCE_BUFFER).unwrap());
            assert_eq!(
                r.p.call_log().sels_since(mark),
                ["didModifyRange:", "synchronizeResource:"]
            );
        }
    }

    /// A `MANAGED_BUFFER_CREATE` whose GPU copy the platform refuses on a GPU of its own memory
    /// is refused, as Metal's answers nil, with no buffer made over its CPU copy in its place.
    #[test]
    fn a_managed_buffer_over_a_shared_buffer_the_gpu_has_no_memory_for_is_refused() {
        let (mut r, ram, page) = pages_rig();
        r.dec.set_unified_memory(false);
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

    /// Where the platform refuses a `Managed` object's GPU copy on a GPU of its own memory, the
    /// create is refused, as Metal's answers nil, with nothing made over the guest's memory in
    /// its place: a `BUFFER_OVER_HOST_MEMORY`, a `BUFFER_MANAGED_AT_OFFSET` in a heap over host
    /// memory, and a `MANAGED_TEXTURE_CREATE` of a shape the platform makes over a buffer.
    #[test]
    fn a_managed_object_the_gpu_has_no_memory_for_is_refused() {
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
    /// in the copy name its slice, row and column. In order: slice 2 read back by a blit (the GPU
    /// reads the copy where its layout puts the slice: rows 256 bytes apart, slices 16 rows), and
    /// the copy after a blit wrote 0x5a into slice 1 and synchronized it, as a guest does (the GPU
    /// writes it there, slices 0 and 2 kept).
    fn managed_array_in_place_on<P: crate::platform::ApiPlatform>() -> (Vec<u8>, Vec<u8>) {
        use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
        use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use reims_vgpu_core::platform::device::GpuDevice as _;
        use reims_vgpu_core::platform::resource::Buffer as _;

        const W: u64 = 48;
        const H: u64 = 16;
        const ROW: u64 = 256;
        const SLICE: u64 = ROW * H;
        let platform = P::process();
        let device = platform.metal_device(None).unwrap();
        crate::graphics::reserve_guest_texture_ids(platform.as_ref(), &device);
        let size = BLOCKS + 0x10_0000 + RGFX_CURSOR_SPAN;
        let region = Arc::new(Region::new(size as usize).unwrap());
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
        assert!(
            dec.managed_copy(21).unwrap().cpu.is_none(),
            "over its CPU copy"
        );
        let Some(ObjHandle::Texture(t)) = dec.named(21) else {
            panic!("no texture 21");
        };
        let queue = device.new_command_queue().unwrap();
        let run = |cmds: Vec<BlitCmd<'_, P>>| {
            let cb = queue.command_buffer(OnFault::Return).unwrap().unwrap();
            let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
            for cmd in cmds {
                e.encode(OnFault::Return, cmd).unwrap();
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
        run(vec![BlitCmd::CopyTextureToBuffer {
            source_texture: t,
            source_slice: 2,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: whole,
            destination_buffer: &out,
            destination_offset: 0,
            destination_bytes_per_row: W * 4,
            destination_bytes_per_image: W * H * 4,
        }]);
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
        // The guest's synchronizeTexture: of the slice after the write, which the platform's own
        // takes for a texture over its CPU copy (`synchronize_managed`).
        run(vec![
            BlitCmd::CopyBufferToTexture {
                source_buffer: &fill,
                source_offset: 0,
                source_bytes_per_row: W * 4,
                source_bytes_per_image: W * H * 4,
                source_size: whole,
                destination_texture: t,
                destination_slice: 1,
                destination_level: 0,
                destination_origin: MtlOrigin::new(0, 0, 0),
            },
            BlitCmd::SynchronizeTexture {
                texture: t,
                slice: 1,
                level: 0,
            },
        ]);
        // SAFETY: the region's block, which the blit's writes finished.
        let after =
            unsafe { std::slice::from_raw_parts(copy(0).cast_const(), (3 * SLICE) as usize) }
                .to_vec();
        (read, after)
    }

    /// Checks [`managed_array_in_place_on`]'s answers against the layout.
    fn check_managed_array_in_place((read, after): (Vec<u8>, Vec<u8>)) {
        let (w, h, row, slice) = (48usize, 16usize, 256usize, 256 * 16usize);
        for y in 0..h {
            for x in 0..w {
                let at = (y * w + x) * 4;
                assert_eq!(
                    read[at..at + 4],
                    [2, y as u8, x as u8, 0xa5],
                    "slice 2 ({x}, {y})"
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
                    assert_eq!(after[at..at + 4], want, "the copy's slice {s} ({x}, {y})");
                }
            }
        }
    }

    /// A one-level `Managed` 2D array on unified memory lies over its CPU copy where the layout
    /// puts its slices: the GPU reads and writes the copy's bytes.
    #[test]
    fn on_vulkan_a_managed_2d_array_lies_over_its_cpu_copy() {
        check_managed_array_in_place(
            managed_array_in_place_on::<reims_vgpu_vulkan::VulkanPlatform>(),
        );
    }
}
