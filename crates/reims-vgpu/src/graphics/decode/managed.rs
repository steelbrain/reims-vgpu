//! `Managed` storage (docs/graphics-kext/design.md, Decided 4): a buffer's or a texture's CPU
//! copy is a region block of its process's, which the guest maps, and the range messages tell the
//! host what the CPU wrote.
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
//! A shape the platform cannot make in place takes the two copies on any GPU: they hold Metal's
//! contract everywhere.

use std::ptr;
use std::sync::Arc;

use reims_graphics::managed::LinearLayout;
use reims_graphics::wire::{
    RGFX_MANAGED_TEXTURE_LINEAR, RgfxBufferDidModifyRange, RgfxBufferManagedAtOffset,
    RgfxBufferOverHostMemory, RgfxManagedTextureCreate, RgfxTextureReplaceRegion,
};
use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder};
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::resource::{Buffer, Heap, Texture};
use reims_vgpu_core::platform::types::{MtlOrigin, MtlRegion, MtlSize, NsRange};
use reims_vgpu_core::platform::values::{BufferDesc, OptionSet};
use reims_vgpu_core::platform::{Handle, HostRange, Platform};
use reims_vgpu_core::raise::{OnFault, RaiseResult};

use super::tables::Entry;
use super::{DecodeError, Drain, GraphicsDecoder, ObjHandle, op_fault, read, texture_desc};

/// `MTLStorageModeManaged`.
pub(super) const STORAGE_MODE_MANAGED: u64 = 1;
/// Where `MTLResourceOptions` hold the storage mode.
const STORAGE_MODE_SHIFT: u32 = 4;
/// `MTLResourceStorageModeMask`.
const STORAGE_MODE_MASK: u64 = 0xf << STORAGE_MODE_SHIFT;
/// `MTLTextureType2D`.
const TEXTURE_TYPE_2D: u32 = 2;
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

/// A `Managed` buffer's or texture's CPU copy: the `length` bytes of the block at `at` in the
/// region.
pub(crate) struct ManagedCopy<P: Platform> {
    /// Where the block starts in the region.
    pub(super) at: u64,
    /// The CPU copy's bytes: the buffer's length, or the texture layout's size.
    pub(super) length: u64,
    /// A `Shared` buffer over the block, where the object is a copy of its own (the two-copy
    /// form); `None` where the object lies over the block.
    pub(super) cpu: Option<P::Buffer>,
    /// A texture's layout of the block; `None` for a buffer.
    pub(super) layout: Option<LinearLayout>,
}

/// Whether a texture over a buffer can have the shape `create` names: a one-level, one-sample 2D
/// texture of a pixel format the guest calls linear.
fn in_place_shape(create: &RgfxManagedTextureCreate) -> bool {
    let d = &create.desc;
    create.flags & RGFX_MANAGED_TEXTURE_LINEAR != 0
        && d.texture_type == TEXTURE_TYPE_2D
        && d.mipmap_level_count == 1
        && d.sample_count <= 1
        && d.array_length <= 1
}

impl<P: Platform> GraphicsDecoder<P> {
    /// Whether `Managed` resources are one copy, over their blocks: the platform's answer of
    /// whether the device's GPU shares the CPU's memory (`hasUnifiedMemory`). The decoder starts
    /// with no, the two copies, which hold Metal's contract on any GPU.
    pub(crate) fn set_unified_memory(&mut self, unified: bool) {
        (self.shared.unified_memory).store(unified, std::sync::atomic::Ordering::Release);
        // A GPU of its own memory keeps IOSurfaces' device caches (`surface_cache`).
        self.surface_caches.set_on(!unified);
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
    /// platform's own `Managed` buffer at the guest's address, its host copy filled from the
    /// block and copied to the GPU's by the next submission.
    pub(super) fn managed_buffer(
        &self,
        device: &P::Device,
        m: &RgfxBufferOverHostMemory,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let options = u64::from(m.options);
        let pin = (m.gpu_address != 0).then_some(m.gpu_address);
        let (buffer, cpu) = if self.unified() {
            match self.buffer_over_block(device, m.region_offset, m.length, options, pin)? {
                Some(buffer) => (buffer, None),
                None => return Ok(Err(DecodeError::Refused)),
            }
        } else {
            let Some(buffer) = self.gpu_copy(device, m.length, options, pin)? else {
                return Ok(Err(DecodeError::Refused));
            };
            let Some(cpu) =
                self.buffer_over_block(device, m.region_offset, m.length, shared(options), None)?
            else {
                return Ok(Err(DecodeError::Refused));
            };
            if !self.fill_gpu_copy(&buffer, m.region_offset, m.length)? {
                return Ok(Err(DecodeError::Refused));
            }
            (buffer, Some(cpu))
        };
        self.store_managed_buffer(m.obj_id, buffer, m.region_offset, m.length, cpu);
        Ok(Ok(()))
    }

    /// `BUFFER_MANAGED_AT_OFFSET`: the platform's own `Managed` buffer at the guest's address, its
    /// CPU copy a `Shared` buffer placed at `offset` in the guest's heap over host memory, so the
    /// buffer imports no host memory of its own. Offered where `Managed` is two copies
    /// (`RGFX_FEATURE_MANAGED_AT_OFFSET`), and refused on a GPU that shares the CPU's memory. A
    /// heap the sender does not hold, or one over no block of the region, is a bad reference; a
    /// CPU copy past the heap's block is malformed.
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
        if m.length == 0
            || m.offset
                .checked_add(m.length)
                .is_none_or(|end| end > heap_length)
        {
            return Ok(Err(DecodeError::Malformed));
        }
        let at = heap_at + m.offset;
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
        if !self.fill_gpu_copy(&buffer, at, m.length)? {
            return Ok(Err(DecodeError::Refused));
        }
        self.store_managed_buffer(m.obj_id, buffer, at, m.length, Some(cpu));
        Ok(Ok(()))
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

    /// Fills a GPU copy's host copy from the `length` bytes at `at` in the region, which the
    /// caller checked lie in a block of the sender's, and has the platform copy them to the GPU's
    /// by the next submission. False where the buffer has no host copy.
    fn fill_gpu_copy(&self, buffer: &P::Buffer, at: u64, length: u64) -> RaiseResult<bool> {
        let contents = buffer.contents();
        if contents.is_null() {
            return Ok(false);
        }
        // SAFETY: the bytes lie inside the region (checked by the caller), and the platform
        // buffer's host copy holds its `length` bytes; the guest's concurrent writes change only
        // what is copied.
        unsafe {
            ptr::copy_nonoverlapping(
                self.region.base().add(at as usize),
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
        let mut entry = Entry::of(ObjHandle::Buffer(buffer)).over(Some((at, length)));
        entry.managed = Some(ManagedCopy {
            at,
            length,
            cpu,
            layout: None,
        });
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
        let (from, copies) = (copy.at + m.location, copy.cpu.is_some());
        self.after_submitted("BUFFER_DID_MODIFY_RANGE", m.obj_id, drain, move || {
            if copies {
                let contents = buffer.contents();
                // SAFETY: the range lies inside the block (checked against its length), inside
                // the region, which `region` keeps mapped, and inside the platform buffer's host
                // copy of the same length.
                unsafe {
                    ptr::copy_nonoverlapping(
                        region.base().add(from as usize),
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
    /// platform's own `Managed` texture beside it. A bad reference for a CPU copy that is no
    /// `Shared` buffer over a block of the sender's, or is shorter than the layout.
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
        let Some(layout) = LinearLayout::of(&m) else {
            return Ok(Err(DecodeError::Malformed));
        };
        let (Some(ObjHandle::Buffer(cpu)), Some((at, length))) = (
            self.named(m.cpu_buffer_obj_id),
            self.span_of(m.cpu_buffer_obj_id),
        ) else {
            return Ok(Err(DecodeError::BadReference));
        };
        if self.managed_copy(m.cpu_buffer_obj_id).is_some() || layout.size() > length {
            return Ok(Err(DecodeError::BadReference));
        }
        let cpu = cpu.clone_ref();
        let desc = texture_desc(&m.desc, m.resource_id);
        if self.unified() && in_place_shape(&m) {
            let mut over = desc;
            over.resource_options = Some(shared(u64::from(m.desc.options)));
            let descriptor = self.platform.texture_descriptor(&over);
            let row = layout.levels().first().map_or(0, |l| l.bytes_per_row);
            if let Some(texture) = cpu.new_texture(OnFault::Return, &descriptor, 0, row)? {
                return self.store_managed_texture(m.obj_id, texture, at, length, None, layout);
            }
        }
        let descriptor = self.platform.texture_descriptor(&desc);
        match device.new_texture(OnFault::Return, &descriptor)? {
            Some(texture) => {
                self.store_managed_texture(m.obj_id, texture, at, length, Some(cpu), layout)
            }
            None => Ok(Err(DecodeError::Refused)),
        }
    }

    /// Stores a `Managed` texture over the block at `at` of `length` bytes, and its CPU copy.
    fn store_managed_texture(
        &self,
        obj_id: u64,
        texture: P::Texture,
        at: u64,
        length: u64,
        cpu: Option<P::Buffer>,
        layout: LinearLayout,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let mut entry = Entry::of(ObjHandle::Texture(texture)).over(Some((at, length)));
        entry.managed = Some(ManagedCopy {
            at,
            length: layout.size(),
            cpu,
            layout: Some(layout),
        });
        self.store(obj_id, entry);
        Ok(Ok(()))
    }

    /// `TEXTURE_REPLACE_REGION`: the platform's `replaceRegion:` from the region's bytes in the
    /// block, where the texture is a copy of its own, after the context's buffers answered before
    /// it are committed ([`Self::after_submitted`], which `drain` is held through). A bad reference
    /// for an object that is no `Managed` texture; malformed for a region the layout does not
    /// hold: both found here.
    pub(super) fn replace_region(
        &self,
        message: &[u8],
        drain: Option<Arc<Drain>>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let Some(m) = read::<RgfxTextureReplaceRegion>(message) else {
            return Ok(Err(DecodeError::Malformed));
        };
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
        if copy.cpu.is_none() {
            return Ok(Ok(()));
        }
        let three_d = texture.texture_type() == TEXTURE_TYPE_3D;
        let (texture, region) = (texture.clone_ref(), Arc::clone(&self.region));
        let (from, bytes_per_row) = (copy.at + offset, l.bytes_per_row);
        let bytes_per_image = if three_d { l.bytes_per_image } else { 0 };
        self.after_submitted("TEXTURE_REPLACE_REGION", m.obj_id, drain, move || {
            // SAFETY: the region's bytes lie inside the layout, which lies inside the block
            // (checked at the create), inside the region, which `region` keeps mapped.
            let bytes = unsafe {
                HostRange::new(HostPtr::from_ptr(region.base().add(from as usize)), span)
            };
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

    use reims_graphics::ops::blit::{
        RGFX_BLIT_RESOURCE_BUFFER, RGFX_BLIT_RESOURCE_TEXTURE, RGFX_OP_BLIT_SYNCHRONIZE_RESOURCE,
        RGFX_OP_BLIT_SYNCHRONIZE_TEXTURE, RgfxBlitOpSynchronizeResource,
        RgfxBlitOpSynchronizeTexture,
    };
    use reims_graphics::region::Region;
    use reims_graphics::wire::{
        RGFX_CMD_BUFFER_DID_MODIFY_RANGE, RGFX_CMD_BUFFER_MANAGED_AT_OFFSET,
        RGFX_CMD_BUFFER_OVER_HOST_MEMORY, RGFX_CMD_HEAP_OVER_HOST_MEMORY,
        RGFX_CMD_MANAGED_TEXTURE_CREATE, RGFX_CMD_OBJECT_DESTROY, RGFX_CMD_TEXTURE_REPLACE_REGION,
        RGFX_CONTEXT_ALLOCATIONS_OFFSET, RGFX_CURSOR_SPAN, RgfxHeapOverHostMemory, RgfxMsgHeader,
        RgfxObjectDestroy, RgfxTextureDescriptor,
    };
    use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
    use reims_vgpu_core::testkit::{Arg, FakePlatform};
    use zerocopy::IntoBytes;

    use super::super::blit::decode_op;
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

    /// On a GPU that does not share the CPU's memory a `Managed` texture is the platform's own
    /// beside its CPU copy: a replaced region is the platform's `replaceRegion:` from the
    /// region's bytes in the block, at the layout's rows; a synchronize copies the named level
    /// and slice, or every one, to the block at the layout's offsets. A CPU copy shorter than the
    /// layout, or one that is a `Managed` buffer, is refused.
    #[test]
    fn a_managed_texture_on_a_discrete_gpu_is_two_copies() {
        let mut r = rig(false);
        let at = BLOCKS + 0x1_0000;
        // 100x60 RGBA8, three levels: 42240 bytes (reims_graphics::managed's case).
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
}
