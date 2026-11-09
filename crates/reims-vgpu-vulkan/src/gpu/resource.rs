//! Metal's resources and state objects over the runtime's: buffers, textures, heaps, samplers,
//! depth-stencil states, pipelines, libraries and functions, fences, and shared texture handles.
//!
//! **An object over an IOSurface** (`newBufferWithIOSurface:`, a texture made on a surface)
//! holds one reference of the surface for as long as the runtime keeps the object, as Metal's
//! does: the surface's retain count is guest-visible, and the runtime holds none of its own. The
//! reference is shared by the handle (for `iosurface`) and the runtime's deallocator of the
//! no-copy buffer over the surface's memory, which lets it go when the runtime releases the
//! buffer, after the last command buffer that used it. A command buffer that may reach a released
//! texture on the surface through its resource ID alone keeps that image's memory and not the
//! buffer, so it holds no reference of the surface.

use std::collections::BTreeMap;
use std::ptr::NonNull;
use std::sync::{Arc, Weak};

use crate::core::host::HostPtr;
use crate::core::num::ToUsize;
use crate::core::platform::resource::{
    Buffer, ComputePipeline, DepthStencilState, ExportedMemory, GpuResource, Heap, Library,
    RenderPipeline, SamplerState, Texture,
};
use crate::core::platform::types::{MtlRegion, MtlSize, NsRange, Swizzle};
use crate::core::platform::values::{SurfaceMatrix, SurfacePlane};
use crate::core::platform::{Handle, HostRange};
use crate::core::raise::{OnFault, Raise, RaiseResult};
use crate::core::sync::Mutex;
use metal2vulkan::metal::{PixelFormat, PrimitiveTopologyClass, TextureType, TextureUsage};
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, MeshPipelineDescriptor, RenderPipelineDescriptor,
};
use metal2vulkan_runtime as rt;

use crate::answer::{
    Nil, Policy, create, get, hearing, metal_aborts, plain, plain_at, refused, relay_gap, send,
    unrepresentable,
};
use crate::gpu;
use crate::gpu::backing::{Backing, REPLACE};
use crate::gpu::desc::{Maker, VulkanTextureDescriptor, created_format_refused};
use crate::gpu::device::VulkanDevice;
use crate::gpu::metal_space::Hold;
use crate::gpu::writes::{Reflected, Reflection};
use crate::os::OsSurface;
use crate::platform::{Obj, VulkanAddressRangeArray, VulkanPlatform};

/// A runtime buffer, with the device that made it and the IOSurface it retains.
#[derive(Debug)]
pub struct BufferObject {
    /// The runtime's buffer.
    pub raw: rt::Buffer,
    /// The device that made it.
    pub device: VulkanDevice,
    /// The IOSurface whose memory it is (`newBufferWithIOSurface:`): one reference, which the
    /// runtime's deallocator of the buffer shares.
    pub surface: Option<Arc<OsSurface>>,
    /// Whether it is detached, and the read-only bit it was made with ([`crate::gpu::backing`]).
    pub backing: Backing,
    /// The heap it is placed in, which its detach detaches, held while it lives
    /// ([`crate::gpu::backing`]).
    pub heap: Option<VulkanHeap>,
    /// Whether the GPU's writes to its bytes never reach their memory: it was made on a surface
    /// of [`crate::os::gpu_read_only`] (`newBufferWithIOSurface:`). Read once, as it is made, as
    /// commands ask it at every dispatch and pass: the buffer holds the surface, and a live
    /// surface's record, which decides it, stays as it was made. It is made before the surface is
    /// handed out, a lookup of the surface reads the same record, and only a surface made with the
    /// ID of one gone, or the dropping of the records of surfaces gone, changes or drops one
    /// (`records_of_live_surfaces_outlast_the_pruning`,
    /// `a_surface_taking_the_id_of_one_gone_takes_none_of_its_record`,
    /// `a_lookup_of_a_surface_answers_its_record`).
    pub gpu_read_only: bool,
}

/// `id<MTLBuffer>`.
pub type VulkanBuffer = Obj<BufferObject>;

/// A runtime texture, with the device that made it and what it is over.
#[derive(Debug)]
pub struct TextureObject {
    /// The runtime's texture.
    pub raw: rt::Texture,
    /// The device that made it.
    pub device: VulkanDevice,
    /// The IOSurface whose memory it is (`newTextureWithDescriptor:iosurface:plane:`): one
    /// reference, which the runtime's deallocator of the buffer under the texture shares.
    pub surface: Option<Arc<OsSurface>>,
    /// Where its texels lie in the bytes of `surface`, for one made on it
    /// (`newTextureWithDescriptor:iosurface:plane:`), its texture's for a view; `None` for any
    /// other.
    pub plane: Option<rt::IoSurfacePlane>,
    /// The buffer it is over or the heap it is placed in, its texture's for a view, held while
    /// it lives ([`Under`]).
    pub under: Option<Under>,
    /// The usage bits it answers beyond the runtime texture's: for one whose GPU writes never
    /// reach its memory ([`TextureObject::gpu_read_only()`]), the whole usage it answers
    /// ([`read_only_usage`]); its texture's for a view and for one opened from a shared handle;
    /// 0 for any other. Apple's private bits Metal answers (0x4000, 0x8000 and 0x10000) are the
    /// runtime's, which keeps them for a texture, its views and a texture opened from its
    /// shared handle, as Metal does (measured: [`crate::gpu::desc`]).
    pub private_usage: u64,
    /// What it holds of Metal's address space, which its views share: a range of its own, its
    /// surface's, a slot in one of Metal's pools, or none
    /// ([`crate::gpu::metal_space::texture_space`]), and its descriptor in Metal's mempool, which
    /// a view takes one of its own of ([`Hold::sharing`]). The range token of the runtime's
    /// texture, or of the buffer it is over, holds it too, until the runtime frees that
    /// ([`crate::gpu::metal_space::Hold`]).
    pub space: Arc<Hold>,
    /// Whether the GPU's writes to its texels never reach their memory
    /// ([`TextureObject::gpu_read_only()`]), read once, as it is made ([`writes_never_reach`]),
    /// as a buffer's is: the texture holds its surface, or the buffer it is over, for its life.
    pub gpu_read_only: bool,
}

/// Whether the GPU's writes to the texels of a texture on `surface`, or over `under`, never
/// reach their memory: it lies on a surface of [`crate::os::gpu_read_only`], made on it, or over
/// a buffer made on it ([`BufferObject::gpu_read_only()`]). A view's is its texture's.
pub(crate) fn writes_never_reach(surface: Option<&OsSurface>, under: Option<&Under>) -> bool {
    match (surface, under) {
        (Some(s), _) => crate::os::gpu_read_only(s),
        (None, Some(Under::Buffer(b, _))) => b.gpu_read_only,
        _ => false,
    }
}

/// `id<MTLTexture>`.
pub type VulkanTexture = Obj<TextureObject>;

/// The buffer a texture is over, or the heap it is placed in, which the texture holds as
/// Metal's does: a detached one is released with the last texture in it, so a command buffer
/// that uses such a texture fails as Metal's does ([`crate::gpu::backing`]). Each holds the
/// bytes of the buffer or heap the texture's texels may lie in, a first byte and an end, which
/// a write of the texture may change ([`crate::gpu::writes::texture_memory`]).
#[derive(Debug)]
pub enum Under {
    /// The buffer whose bytes it is.
    Buffer(VulkanBuffer, std::ops::Range<u64>),
    /// The heap it is placed in.
    Heap(VulkanHeap, std::ops::Range<u64>),
}

impl Under {
    /// Another reference to the same buffer or heap.
    fn clone_ref(&self) -> Under {
        match self {
            Under::Buffer(buffer, bytes) => Under::Buffer(buffer.clone_ref(), bytes.clone()),
            Under::Heap(heap, bytes) => Under::Heap(heap.clone_ref(), bytes.clone()),
        }
    }
}

/// The bytes of a buffer of `length` bytes a texture of `desc` over it from `offset`, its rows
/// `bytes_per_row` apart, may lie in, where Metal lays its texels out (the runtime's
/// `linear_plan.rs`, measured): a row per row of texels from the offset, a 2D array's slices
/// `bytes_per_row` times its height, rounded up to 128, apart, whatever `bytesPerImage` says, and
/// a texture buffer's texels from its offset rounded down to 16, where its functions read and
/// write them; up to the buffer's end.
fn texel_bytes(
    desc: &rt::TextureDescriptor,
    (offset, bytes_per_row): (u64, u64),
    length: u64,
) -> std::ops::Range<u64> {
    let image = bytes_per_row.saturating_mul(desc.height.max(1));
    let (slices, pitch) = match desc.texture_type {
        TextureType::Type2DArray => (
            desc.array_length.max(1),
            image.checked_next_multiple_of(128).unwrap_or(u64::MAX),
        ),
        _ => (1, image),
    };
    let start = match desc.texture_type {
        TextureType::TextureBuffer => offset - offset % 16,
        _ => offset,
    };
    let span = (pitch.saturating_mul(slices - 1)).saturating_add(image);
    start.min(length)..offset.saturating_add(span).min(length)
}

/// The runtime's matrix of a surface's `kIOSurfaceYCbCrMatrix` value, as
/// `rt::SurfaceYCbCrMatrix::from_attachment` reads the value's string.
fn ycbcr_matrix(matrix: SurfaceMatrix) -> rt::SurfaceYCbCrMatrix {
    match matrix {
        SurfaceMatrix::Itu601 => rt::SurfaceYCbCrMatrix::Itu601,
        SurfaceMatrix::Itu709 => rt::SurfaceYCbCrMatrix::Itu709,
        SurfaceMatrix::Itu2020 => rt::SurfaceYCbCrMatrix::Itu2020,
        SurfaceMatrix::Smpte240M1995 => rt::SurfaceYCbCrMatrix::Smpte240M,
        SurfaceMatrix::DciP3 => rt::SurfaceYCbCrMatrix::DciP3,
        SurfaceMatrix::P3D65 => rt::SurfaceYCbCrMatrix::P3D65,
        SurfaceMatrix::Other => rt::SurfaceYCbCrMatrix::Other,
    }
}

impl BufferObject {
    /// Whether the GPU's writes to its bytes never reach their memory: it was made on a surface
    /// of [`crate::os::gpu_read_only`] (`newBufferWithIOSurface:`), as it was made (the field of
    /// the same name).
    pub(crate) fn gpu_read_only(&self) -> bool {
        self.gpu_read_only
    }
}

/// `MTLTextureUsageRenderTarget`.
const RENDER_TARGET: TextureUsage = TextureUsage::RENDER_TARGET;

/// The usage a view of a swizzle other than the default drops of its texture's: all but
/// `ShaderRead`, `PixelFormatView` and Apple's private 0x4000 and 0x10000, as Metal's view and
/// the runtime's keep them (measured on the M3 Max: a texture of 0x1c01d answers a swizzled view
/// of 0x14011, one of 0x8005 a view of 0x1). A view of a texture whose usage this platform
/// answers itself drops it so ([`read_only_usage`]).
const SWIZZLE_DROPPED: TextureUsage = TextureUsage::ALL_KNOWN.difference(
    TextureUsage::SHADER_READ
        .union(TextureUsage::PIXEL_FORMAT_VIEW)
        .union(TextureUsage::NO_CONCURRENT_ACCESS)
        .union(TextureUsage::BLOCK_WRITES_ONLY),
);

/// The usage Metal gives a texture of `Unknown` usage: `ShaderRead | ShaderWrite | RenderTarget
/// | PixelFormatView` (0x17), as the runtime's texture answers it (its `effective_usage`,
/// measured).
const UNKNOWN_USAGE: TextureUsage = TextureUsage::SHADER_READ
    .union(TextureUsage::SHADER_WRITE)
    .union(RENDER_TARGET)
    .union(TextureUsage::PIXEL_FORMAT_VIEW);

/// `desc`, of a texture whose GPU writes never reach its memory
/// ([`TextureObject::gpu_read_only()`]), made without `RenderTarget` usage, and the whole usage
/// the texture answers ([`TextureObject::private_usage`]): its descriptor's, as Metal answers
/// it, Apple's private bits among them, which the runtime's usage holds. The runtime keeps a
/// texture of `RenderTarget` usage over a buffer's bytes in an image of its own, copying it back
/// into the bytes after each encoder that writes it, a copy the GPU faults on into such memory,
/// where Metal writes nothing of its own; no pass takes such a texture as an attachment
/// ([`crate::os::dropped_write`]). The usage answered is the descriptor's, [`UNKNOWN_USAGE`] for
/// `Unknown`, and of a swizzled texture only its `ShaderRead` and `PixelFormatView`, as the
/// runtime's texture answers (its `effective_usage` and `texture_usage`, measured). The texture
/// is made of that usage without `RenderTarget`, or of `ShaderRead` where that leaves none, as
/// the runtime would take no usage as `Unknown`, of which it would make the texture a render
/// target. But a swizzle other than the default with `ShaderWrite` or `RenderTarget` usage keeps
/// the descriptor's usage, which the runtime refuses as it refuses it over any memory: Metal's
/// validation stops it (`SwizzleWithWrites`), and with `writeSwizzleEnabled` Metal writes
/// through the swizzle, which the runtime does not build (`RuntimeGap::WriteSwizzle`). Bit 0x8
/// is answered by no texture (`TextureUsage::DROPPED`): one of it alone answers 0, as Metal's
/// does, and is made of `ShaderRead`, as Metal reads it as one of that usage (the module docs of
/// [`crate::gpu::desc`]).
pub(crate) fn read_only_usage(desc: &mut rt::TextureDescriptor) -> u64 {
    let mut usage = match desc.usage.is_empty() {
        true => UNKNOWN_USAGE,
        false => desc.usage.without_dropped(),
    };
    if !desc.swizzle.is_default() {
        let kept = TextureUsage::SHADER_READ.union(TextureUsage::PIXEL_FORMAT_VIEW);
        let writes = TextureUsage::SHADER_WRITE.union(RENDER_TARGET);
        if desc.usage.intersects(writes) {
            return u64::from(usage.intersection(kept).as_raw());
        }
        usage = usage.intersection(kept);
    }
    let made = usage.difference(RENDER_TARGET);
    desc.usage = match made.is_empty() {
        true => TextureUsage::SHADER_READ,
        false => made,
    };
    u64::from(usage.as_raw())
}

impl TextureObject {
    /// The texture's life as [`Placements`] reads it, which it does not revive: its [`space`](TextureObject::space), which every view of it shares, views of views
    /// included ([`Hold::sharing`]), so its bytes stay taken while it or a view of it lives, as a
    /// Metal view's `parentTexture` keeps the texture and its bytes. A texture on a surface's
    /// lasts until the runtime frees the buffer under it too, whose range token holds it, once
    /// the last command buffer that used it completed, as Metal's command buffer keeps the
    /// texture until then.
    pub(crate) fn life(&self) -> Weak<dyn std::any::Any + Send + Sync> {
        let space: Weak<Hold> = Arc::downgrade(&self.space);
        space
    }

    /// Whether the GPU's writes to its texels never reach their memory: it lies on a surface of
    /// [`crate::os::gpu_read_only`], made on it, a view of one made on it, or over a buffer made
    /// on it ([`writes_never_reach`]), as it was made (the field of the same name).
    pub(crate) fn gpu_read_only(&self) -> bool {
        self.gpu_read_only
    }

    /// Whether the runtime refuses the GPU's writes of its texels: one over a surface of a 4:4:4
    /// two-plane YCbCr format whose writes it does not store as Metal's GPU does
    /// ([`crate::gpu::ycbcr::writes_refused`]).
    pub(crate) fn writes_refused(&self) -> bool {
        self.surface.is_some()
            && crate::gpu::ycbcr::writes_refused(plain(|| self.raw.pixel_format()))
    }

    /// The runtime's texture, for `sel`, which reads or writes its texels. A rotated one is the
    /// runtime's too, which stores and loads a pass's texels turned in the surface's bytes as
    /// Metal does, and refuses what it does not build (a view of one, CPU access or a blit whose
    /// rows run past its buffer), each refusal dying at its send ([`crate::gpu::desc`]).
    pub fn texels(&self, _sel: &'static str) -> &rt::Texture {
        &self.raw
    }

    /// The runtime's texture, for `sel`, a blit or a CPU access that reads or writes its texels,
    /// as [`TextureObject::texels`].
    pub fn copied(&self, sel: &'static str) -> &rt::Texture {
        self.texels(sel)
    }

    /// `replaceRegion:` (`sel`) of `region` of `level` and `slice` from `bytes`, rows and images
    /// `rows` apart, of one whose GPU writes never reach its memory ([`Self::gpu_read_only()`]),
    /// whose runtime texture lies over a read-only alias the CPU cannot write: refused as the
    /// runtime refuses it, by its read of the same region (`getBytes:`, which takes the same
    /// checks), then written row by row into the guest's page of its surface, where its base
    /// address lies, as Metal's write through the base address reaches it. Where the base
    /// address is the surface's own mapping of several pages, which is read-only, Metal's write
    /// ends the process with a bus error, and this one dies ([`crate::os::base_in_pages`];
    /// measured: `tests/client_direction.rs`). A texel of a two-plane XR10 texture (550, 551) is
    /// its colour word in plane 0, 4 bytes, where Metal's description of the format says 2 a
    /// block, and the write moves those words alone, as Metal's does (measured:
    /// `cpu_writes_and_buffer_blits_of_two_plane_xr10_textures_over_surfaces_are_metals`). One of
    /// a compressed format or of a two-plane YCbCr format, which nothing measured, dies, and so
    /// does one over a buffer made on such a surface, whose bytes Metal's buffer does not map (its
    /// `contents` is NULL).
    fn replaced_on_surface(
        &self,
        sel: &'static str,
        raw: &rt::Texture,
        (region, level, slice): (MtlRegion, u64, u64),
        bytes: &[u8],
        (bytes_per_row, bytes_per_image): (u64, u64),
    ) -> RaiseResult<()> {
        let (Some(surface), Some(plane)) = (&self.surface, self.plane) else {
            unrepresentable(
                sel,
                "a CPU write of a texture over a buffer made on a surface made read-only for \
                 the GPU (kIOSurfaceClientDirection 2), whose bytes Metal's buffer does not map",
            );
        };
        let mut checked = vec![0u8; bytes.len()];
        send(sel, Policy::Dies, || {
            raw.get_bytes(
                &mut checked,
                bytes_per_row,
                bytes_per_image,
                gpu::region(region),
                level,
                slice,
            )
        })?;
        if !crate::os::base_in_pages(surface) {
            metal_aborts(
                sel,
                "a CPU write through the base address of a surface made read-only for the GPU \
                 over several of the guest's pages, its own read-only mapping (a bus error)",
            );
        }
        let pf = plain(|| raw.pixel_format());
        let block = crate::gpu::format_info::block(sel, u64::from(pf.as_raw()));
        if block.compressed() {
            unrepresentable(
                sel,
                "a CPU write of a texture of a compressed format on a surface made read-only \
                 for the GPU, which nothing measured",
            );
        }
        if crate::gpu::ycbcr::runtime_makes(pf) {
            unrepresentable(
                sel,
                "a CPU write of a texture of a two-plane YCbCr format on a surface made \
                 read-only for the GPU, which nothing measured",
            );
        }
        let unit = match crate::gpu::device::two_plane_xr10(pf) {
            true => 4,
            false => block.bytes(),
        };
        let (origin, size) = (region.origin, region.size);
        let length = size.width * unit;
        for row in 0..size.height {
            let at = plane.offset + (origin.y + row) * plane.bytes_per_row + origin.x * unit;
            let from = (row * bytes_per_row).to_usize();
            let written = bytes
                .get(from..from + length.to_usize())
                .is_some_and(|row| crate::os::write_surface_bytes(surface, at, row));
            if !written {
                unrepresentable(
                    sel,
                    &format!(
                        "a CPU write of row {row} of a texture on a surface made read-only for \
                         the GPU, past the bytes given or the guest's pages"
                    ),
                );
            }
        }
        Ok(())
    }

    /// The runtime's texture, for `_sel`, a blit that optimizes or synchronizes its contents and
    /// uses none of its texels.
    pub fn kept(&self, _sel: &'static str) -> &rt::Texture {
        &self.raw
    }
}

/// A runtime heap, with the device that made it.
#[derive(Debug)]
pub struct HeapObject {
    /// The runtime's heap.
    pub raw: rt::Heap,
    /// The device that made it.
    pub device: VulkanDevice,
    /// Its descriptor's size, which a replace's ranges total.
    pub descriptor_size: u64,
    /// Whether it is detached, and the read-only bit it was made with ([`crate::gpu::backing`]).
    pub backing: Backing,
    /// Its buffers and textures, while they live, for the ranges of its textures nothing may
    /// share ([`Placements::place`]).
    pub placed: Mutex<Placements>,
    /// What its object holds of Metal's address space beyond its range token's, which holds a
    /// heap's ranges until the runtime frees it ([`crate::gpu::metal_space::Hold`]): none.
    pub space: Hold,
}

/// A heap's buffers and textures, while they live, and their ranges of the heap.
///
/// A two-plane XR10 texture (550, 551) of a heap is one of three kinds. A texture of one level
/// within the device's limits, but for a 1D array, keeps its first level's slices in the heap's
/// bytes, where Metal lays them out, the runtime's shadow of them (`shadowed-texture`), so a
/// buffer or texture over its range sees and changes its texels as Metal's does: a 3D one its
/// first depth slice, over which Metal lays every other. A mipmapped one and a 1D array keep the first level's slices there too, as Metal's
/// CPU copies and GPU writes lay them, where Metal's paths address the later levels, and a 1D
/// array's GPU reads, at bytes that differ by path (`heap-two-plane-layout`, relayed, and in strict
/// mode dying at its creation, [`crate::answer::DIES`]). Any other lies in memory of its own, apart
/// from the heap's bytes, which the runtime announces at its creation, on the caller's thread
/// (`heap-two-plane-texture`, [`crate::answer::HEARD`]): only those are [`Alone::TwoPlane`] here,
/// the kind the creation heard, not one the format and shape would predict. So is a depth or
/// stencil texture without `allowGPUOptimizedContents` whose creation heard
/// `optimized-depth-stencil-texture` [`Alone::Optimized`], and a texture of a heap over the guest's
/// pages whose creation heard `heap-texture-untiled-layout` [`Alone::Untiled`].
///
/// Core makes a buffer over all of the bytes the heap's descriptor asked for with each heap, at
/// its creation, before anything else is asked of it: a detached heap's cover buffer, which a
/// command that lists the heap where a buffer is expected uses, and a legacy heap's private
/// buffer, through which core pages the heap's bytes to and from the guest's pages. It is told
/// by that alone: the first buffer or texture asked of the heap, made or not
/// ([`Placements::attempt`]), at offset 0 over all of those bytes ([`Placements::place_buffer`],
/// the heap's cover). A guest's buffer over all of the heap, asked later, is kept among the
/// others, as any buffer is, the runtime's nil to core's included. On Metal the cover shares every
/// byte of every texture of the heap, and it lives as long as the heap. A texture of a kind above
/// dies at its placement while the cover lives, so in every heap core makes
/// ([`Alone::dies_beside_the_cover`]): the cover, and the guest's pages under a heap over them,
/// which the guest reads with no call to die at, would show the bytes the runtime leaves in the
/// heap for it, which nothing shows are Metal's: none of a two-plane texture's texels, which lie
/// apart; an untiled one's in the runtime's layout, where Metal's are its tiles; an optimized
/// one's laid out and sized by the runtime, larger than Metal's.
///
/// In a legacy heap, core's copies through the heap's private buffer carry the bytes of its
/// textures to and from the guest's pages: the runtime lays a texture out there in Metal's tiles
/// where it holds them (on MoltenVK, a texture of one sample that Metal keeps uncompressed, of a
/// colour format, `Depth16Unorm`, `Depth32Float`, `Stencil8` or `Depth32Float_Stencil8`), and
/// else announces its layout there (`heap-texture-untiled-layout`, `heap-texture-lossless-layout`),
/// which its creation relays, the first time alone ([`crate::answer::HEARD`]).
#[derive(Debug, Default)]
pub struct Placements {
    /// Its textures nothing may share a byte with, and why.
    alone: Vec<(
        std::ops::Range<u64>,
        Weak<dyn std::any::Any + Send + Sync>,
        Alone,
    )>,
    /// Its other buffers and textures.
    others: Vec<(std::ops::Range<u64>, Weak<dyn std::any::Any + Send + Sync>)>,
    /// How many of `others` lived when the rest were last let go.
    kept: usize,
    /// Whether a buffer or texture was asked of the heap, made or not ([`Placements::attempt`]).
    made: bool,
    /// The heap's cover, core's buffer over all of it, and its range
    /// ([`Placements::place_buffer`]).
    cover: Option<(std::ops::Range<u64>, Weak<dyn std::any::Any + Send + Sync>)>,
}

/// Why a heap's texture shares no byte with another buffer or texture of the heap while it
/// lives ([`Placements::place`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alone {
    /// A two-plane XR10 texture whose texels the runtime keeps in memory of their own
    /// (`heap-two-plane-texture`), where Metal's lie in the heap's bytes, so neither resource
    /// would see or change the other's.
    TwoPlane,
    /// A depth or stencil texture without `allowGPUOptimizedContents`, over the runtime's size
    /// for it, Metal's for the texture with the property on, larger than Metal's with it off
    /// (`optimized-depth-stencil-texture`): the other's bytes may lie past Metal's texture, which
    /// shares none of them, and the placement does not tell them from Metal's.
    Optimized,
    /// A texture of a heap over the guest's pages that the runtime does not lay out in Metal's
    /// tiles (`heap-texture-untiled-layout`): its bytes in the heap are the runtime's layout,
    /// where Metal's are its tiles, so the other would meet other bytes than on Metal.
    Untiled,
}

impl Alone {
    /// The texture, and why it shares no byte.
    fn reason(self) -> &'static str {
        match self {
            Alone::TwoPlane => {
                "a two-plane XR10 texture (550, 551) of its heap past the device's limits: the \
                 runtime keeps the texture's texels in memory of their own \
                 (heap-two-plane-texture), where Metal's lie in the heap's bytes"
            }
            Alone::Optimized => {
                "a depth or stencil texture of its heap without allowGPUOptimizedContents: the \
                 runtime sizes the texture as Metal's with the property on, larger than Metal's \
                 with it off (optimized-depth-stencil-texture), so the bytes they share may lie \
                 past Metal's texture"
            }
            Alone::Untiled => {
                "a texture of its heap over the guest's pages that the runtime does not lay out \
                 in Metal's tiles: its bytes in the heap are the runtime's layout, where Metal's \
                 are its tiles (heap-texture-untiled-layout)"
            }
        }
    }

    /// The death at `sel` of a buffer or texture over a byte of such a texture that lives, or of
    /// such a texture over another's.
    fn dies(self, sel: &'static str) -> ! {
        unrepresentable(
            sel,
            &format!(
                "a buffer or texture over the bytes of a texture that lives, or such a texture \
                 over another's, the texture {}",
                self.reason()
            ),
        )
    }

    /// The death at `sel` of such a texture placed while its heap's cover lives, the buffer over
    /// all of the heap core makes with it, which shows the texture's bytes, as the guest's pages
    /// under a heap over them do ([`Placements`]).
    fn dies_beside_the_cover(self, sel: &'static str) -> ! {
        unrepresentable(
            sel,
            &format!(
                "a texture placed in a heap while the buffer over all of the heap that core makes \
                 with it lives, which shows the bytes the runtime leaves in the heap for the \
                 texture, as the guest's pages under a heap over them do, where Metal's are its \
                 own, the texture {}",
                self.reason()
            ),
        )
    }
}

/// The relay at `sel`, or its death in strict mode ([`crate::answer::inexact`]), of `what` a
/// depth or stencil texture without `allowGPUOptimizedContents`, whose creation or size query
/// the runtime announced (`optimized-depth-stencil-texture`, [`crate::answer::HEARD`]): MoltenVK
/// keeps its layout GPU-optimized, so the runtime sizes it as Metal's texture with the property
/// on, larger than Metal's with it off, where the runtime's size decides a placement, or Metal's
/// is not modelled (`crate::gpu::device::VulkanDevice::metal_answer_at`); the call goes on with
/// the runtime's size.
pub(crate) fn optimized_size(sel: &'static str, what: &str) {
    crate::answer::inexact(sel, crate::answer::Inexact::HeapLayout, || {
        optimized_line(what)
    });
}

/// The line of [`optimized_size`] for `what`.
fn optimized_line(what: &str) -> String {
    format!(
        "{what} a depth or stencil texture without allowGPUOptimizedContents, which the runtime \
         sizes as Metal's texture with the property on, larger than Metal's with it off \
         (optimized-depth-stencil-texture)"
    )
}

/// Whether the ranges share a byte.
fn overlap(a: &std::ops::Range<u64>, b: &std::ops::Range<u64>) -> bool {
    a.start < b.end && b.start < a.end
}

impl Placements {
    /// Keeps `object`, placed over `range` of the heap for `sel`, a texture nothing may share a
    /// byte with where `alone` says why. One over a byte of such a texture that lives dies, and
    /// so does such a texture over a byte of a buffer or texture that lives, the heap's cover
    /// among them, naming why ([`Alone`]). A command buffer's hold of a resource keeps it living,
    /// and a texture lives while a view of it does ([`TextureObject::life`]).
    fn place(
        &mut self,
        sel: &'static str,
        range: std::ops::Range<u64>,
        object: Weak<dyn std::any::Any + Send + Sync>,
        alone: Option<Alone>,
    ) {
        let lives =
            |(_, object): &(_, Weak<dyn std::any::Any + Send + Sync>)| object.strong_count() > 0;
        self.made = true;
        if range.is_empty() {
            return;
        }
        self.alone
            .retain(|(_, object, _)| object.strong_count() > 0);
        if alone.is_some() {
            self.others.retain(lives);
            self.kept = self.others.len();
        }
        let under = self
            .alone
            .iter()
            .find(|(placed, ..)| overlap(placed, &range));
        if let Some(&(_, _, why)) = under {
            why.dies(sel);
        }
        if let Some(why) = alone
            && self
                .others
                .iter()
                .any(|(placed, _)| overlap(placed, &range))
        {
            why.dies(sel);
        }
        let covered = |(placed, cover): &(std::ops::Range<u64>, Weak<_>)| {
            cover.strong_count() > 0 && overlap(placed, &range)
        };
        if let Some(why) = alone
            && self.cover.as_ref().is_some_and(covered)
        {
            why.dies_beside_the_cover(sel);
        }
        match alone {
            None => {
                if self.others.len() >= 2 * self.kept.max(16) {
                    self.others.retain(lives);
                    self.kept = self.others.len();
                }
                self.others.push((range, object));
            }
            Some(why) => self.alone.push((range, object, why)),
        }
    }

    /// A buffer or texture asked of the heap, whether or not it is made: whether it is the first
    /// asked. Only the first can be the heap's cover, so a guest's buffer over all of the heap is
    /// never taken for it where the runtime answered nil to core's ([`Placements::place_buffer`]).
    fn attempt(&mut self) -> bool {
        !std::mem::replace(&mut self.made, true)
    }

    /// Keeps `object`, a buffer placed over `range` of the heap for `sel`: the heap's cover where
    /// `cover`, the first buffer or texture asked of the heap ([`Placements::attempt`]), at offset
    /// 0 over all of the bytes the heap's descriptor asked for, as core makes one with each heap
    /// ([`Placements`]); any other as [`Placements::place`] keeps it.
    fn place_buffer(
        &mut self,
        sel: &'static str,
        range: std::ops::Range<u64>,
        object: Weak<dyn std::any::Any + Send + Sync>,
        cover: bool,
    ) {
        if cover {
            self.made = true;
            self.cover = Some((range, object));
            return;
        }
        self.place(sel, range, object, None);
    }
}

impl HeapObject {
    /// The runtime's nil `sel` at `offset` inside the heap of a depth or stencil texture of `desc`
    /// without `allowGPUOptimizedContents`, whose size it answers larger than Metal's
    /// (`optimized-depth-stencil-texture`): nil as well where Metal's own size and alignment, the
    /// runtime's model of them (`crate::gpu::device::VulkanDevice::metal_answer_at`), do not fit
    /// there either, as Metal answers nil for an offset off its alignment and for bytes past the
    /// heap's end (measured); where Metal's fit, or the model has none for the shape, it dies in
    /// either mode, as Metal may make the texture the runtime makes none of.
    fn optimized_nil(&self, sel: &'static str, desc: &rt::TextureDescriptor, offset: u64) {
        let heap_size = plain(|| self.raw.size());
        let metal = self.device.metal_answer_at(sel, desc);
        let metal_nil = metal.is_some_and(|metal| {
            !offset.is_multiple_of(metal.align)
                || offset
                    .checked_add(metal.size)
                    .is_none_or(|end| end > heap_size)
        });
        if !metal_nil {
            unrepresentable(
                sel,
                &optimized_line(
                    "a placement inside its heap, which the runtime answers nil where Metal's \
                     size fits or is not modelled, of",
                ),
            );
        }
    }

    /// The runtime's texture of `desc`, a depth or stencil texture without
    /// `allowGPUOptimizedContents` (`optimized-depth-stencil-texture`), made by `sel` at `offset`
    /// in a heap, of `size` bytes as the runtime answers them: it relays, or dies in strict mode
    /// ([`optimized_size`]), where Metal answers nil (its alignment missing the offset, or its
    /// bytes past the heap's end), where the runtime answers fewer bytes than Metal's, or none
    /// (`HeapObject::new_texture`), and where the runtime's model of Metal's size and alignment
    /// (`crate::gpu::device::VulkanDevice::metal_answer_at`) has none for it, as Metal may answer
    /// nil there, and as the runtime's nil dies there ([`HeapObject::optimized_nil`]).
    fn optimized_made(
        &self,
        sel: &'static str,
        desc: &rt::TextureDescriptor,
        offset: u64,
        size: Option<u64>,
    ) {
        let Some(metal) = self.device.metal_answer_at(sel, desc) else {
            optimized_size(
                sel,
                "a placement whose size and alignment the runtime does not model Metal's for, so \
                 that Metal may answer nil where the runtime makes it, of",
            );
            return;
        };
        let heap_size = plain(|| self.raw.size());
        let metal_nil = !offset.is_multiple_of(metal.align)
            || offset
                .checked_add(metal.size)
                .is_none_or(|end| end > heap_size);
        if metal_nil || size.is_none_or(|size| size < metal.size) {
            optimized_size(
                sel,
                "a placement where Metal answers nil, or that the runtime answers fewer bytes for \
                 than Metal's, of",
            );
        }
    }

    /// `detachBacking`: detached from now on by the runtime, in no memory until a replace, for a
    /// heap made over address ranges ([`Backing::detach`]).
    pub(crate) fn detach(&self) {
        self.backing.detach(|| self.raw.detach_backing());
    }
}

/// `id<MTLHeap>`.
pub type VulkanHeap = Obj<HeapObject>;

/// `id<MTLSamplerState>`.
pub type VulkanSamplerState = Obj<rt::SamplerState>;

/// The sampler states' handles, by their runtime state's resource ID: the one object each runtime
/// state is. A live runtime state's ID is its own on its device, so an ID names one live state of
/// a device; its list holds the handles of other devices' states, and of states gone.
struct SamplerHandles {
    by_id: BTreeMap<u64, Vec<Weak<rt::SamplerState>>>,
    /// The handles listed, gone ones among them.
    listed: usize,
    /// Where the next sweep of the gone ones is due: twice the handles live at the last.
    sweep_at: usize,
}

/// The handles of [`SamplerHandles`].
static SAMPLERS: Mutex<SamplerHandles> = Mutex::new(SamplerHandles {
    by_id: BTreeMap::new(),
    listed: 0,
    sweep_at: SAMPLER_SWEEP_FLOOR,
});

/// The fewest handles listed before a sweep.
const SAMPLER_SWEEP_FLOOR: usize = 256;

/// The handle of `raw`: the object a live handle to the same runtime state already is, or a new
/// one. The runtime answers one state for equal requests while it lives, as Metal answers one
/// object (one identity) on this host, so the handles of equal requests are one object too. The
/// ID it takes is in the runtime's sampler ID pool, which Metal's sampler mempool follows
/// ([`crate::gpu::metal_space`]'s `Samplers`). The handle is found by that ID, so a create costs
/// the same however many samplers live (it read every live handle before: 11 us a create with
/// 1000 alive, measured on the msi RTX 5080).
pub(crate) fn sampler_state(raw: rt::SamplerState) -> VulkanSamplerState {
    let id = plain(|| rt::SamplerState::gpu_resource_id(&raw)).0;
    let mut table = SAMPLERS.lock();
    let handles = table.by_id.entry(id).or_default();
    if let Some(held) = handles
        .iter()
        .filter_map(Obj::upgrade)
        .find(|held| **held == raw)
    {
        return held;
    }
    let before = handles.len();
    handles.retain(|weak| weak.strong_count() > 0);
    let made = Obj::new(raw);
    handles.push(made.downgrade());
    let after = handles.len();
    table.listed = table.listed + after - before;
    if table.listed >= table.sweep_at {
        table.by_id.retain(|_, listed| {
            listed.retain(|weak| weak.strong_count() > 0);
            !listed.is_empty()
        });
        table.listed = table.by_id.values().map(Vec::len).sum();
        table.sweep_at = (2 * table.listed).max(SAMPLER_SWEEP_FLOOR);
    }
    made
}

/// `id<MTLDepthStencilState>`.
pub type VulkanDepthStencilState = Obj<rt::DepthStencilState>;

/// `id<MTLFence>`.
pub type VulkanFence = Obj<rt::Fence>;

/// A runtime library, with the device that made it.
#[derive(Debug)]
pub struct LibraryObject {
    /// The runtime's library.
    pub raw: rt::Library,
    /// The device that made it.
    pub device: VulkanDevice,
}

/// `id<MTLLibrary>`.
pub type VulkanLibrary = Obj<LibraryObject>;

/// `id<MTLFunction>`.
pub type VulkanFunction = Obj<rt::Function>;

/// A runtime compute pipeline state, with the runtime's `uniqueIdentifier` of it, read when it
/// was made (`identified`). It derefs to the runtime's state.
#[derive(Debug)]
pub struct ComputePipelineObject {
    /// The runtime's state.
    pub raw: rt::ComputePipelineState,
    /// Its `uniqueIdentifier`.
    pub unique_identifier: u64,
    /// The reflection of the descriptor it was made of, where a creator of the guest's made it,
    /// whose slot writes tell which texture and buffer slots its kernel writes
    /// ([`Reflection::slots`]).
    pub reflection: Arc<Reflection>,
}

impl ComputePipelineObject {
    /// `raw`, just made, with the runtime's identifier of it: one per creation, as Metal gives a
    /// pipeline made from an identical record a new one, and as the runtime makes a new state of
    /// each request (`identified`).
    pub fn new(raw: rt::ComputePipelineState) -> ComputePipelineObject {
        let unique_identifier = identified(|| raw.unique_identifier());
        ComputePipelineObject {
            raw,
            unique_identifier,
            reflection: Reflection::new(None),
        }
    }

    /// `raw`, just made of `descriptor`, as [`ComputePipelineObject::new`] makes it.
    pub fn of(raw: rt::ComputePipelineState, descriptor: ComputePipelineDescriptor) -> Self {
        let mut object = ComputePipelineObject::new(raw);
        object.reflection = Reflection::new(Some(Reflected::Compute(Box::new(descriptor))));
        object
    }
}

impl std::ops::Deref for ComputePipelineObject {
    type Target = rt::ComputePipelineState;
    fn deref(&self) -> &rt::ComputePipelineState {
        &self.raw
    }
}

/// `id<MTLComputePipelineState>`.
pub type VulkanComputePipeline = Obj<ComputePipelineObject>;

/// A runtime render pipeline state, as core holds it, with its `uniqueIdentifier`.
#[derive(Debug)]
pub struct RenderPipelineObject {
    /// The runtime's state.
    pub raw: rt::RenderPipelineState,
    /// The runtime's identifier of it, read when it was made (`identified`).
    unique_identifier: u64,
    /// Its descriptor's `inputPrimitiveTopology`, which decides whether Metal sizes the points
    /// of a vertex function that writes no `[[point_size]]` ([`crate::gpu::encode_render`]);
    /// `Unspecified`, Metal's default, for a state of no guest descriptor's.
    pub input_primitive_topology: PrimitiveTopologyClass,
    /// The reflection of the descriptor it was made of, whose slot writes tell which slots each
    /// stage writes ([`Reflection::slots`]); of none for a state made otherwise.
    pub reflection: Arc<Reflection>,
    /// What it tells a render encoder that sets it, read at its first set
    /// ([`crate::gpu::encode_render::StateFacts`]).
    pub(crate) drawn: std::sync::OnceLock<crate::gpu::encode_render::StateFacts>,
}

/// The runtime's `uniqueIdentifier` of a state it just made
/// (`ComputePipelineState::unique_identifier`, `RenderPipelineState::unique_identifier`),
/// `identifier`'s answer, read once as the state is made, which relays the runtime's notice of it
/// once a device (`plain-state-unique-identifier`, `unmeasured-unique-identifier`,
/// [`crate::answer`]): one nonzero value for the state's life, no other live state's, and a second
/// state made from an identical record a new one. A state made without
/// `supportIndirectCommandBuffers` answers one of a process-wide count, where Metal's is an address
/// that differs from run to run (`plain-state-unique-identifier`); one made with it answers 2^36
/// plus an offset in the runtime's allocation of such states, which reuses a released state's
/// identifier, the last released first, by kind, as Metal's does, at offsets that are not Metal's
/// (`unmeasured-unique-identifier`: Metal's follow its allocator and the process's history, as how
/// the process was started). The guest's read of such a state's is relayed ([`guest_identifier`]).
fn identified(identifier: impl FnOnce() -> u64) -> u64 {
    plain(identifier)
}

/// What the guest reads (info command 460) of a pipeline state's runtime `uniqueIdentifier`,
/// `identifier`: the same, but of a state made with indirect command buffer support (`indirect`)
/// relayed as [`Inexact::UniqueIdentifier`](crate::answer::Inexact::UniqueIdentifier), not
/// Metal's address, which dies in strict mode. A plain state's is one of a count, where Metal's
/// is an address that differs from run to run (`plain-state-unique-identifier`, relayed).
fn guest_identifier(identifier: u64, indirect: bool) -> u64 {
    if indirect {
        crate::answer::inexact(
            "uniqueIdentifier",
            crate::answer::Inexact::UniqueIdentifier,
            || {
                format!(
                    "Metal's identifier of a pipeline state made with indirect command buffer \
                     support: it answers the runtime's, {identifier:#x}, an offset of its own \
                     allocation of such states"
                )
            },
        );
    }
    identifier
}

/// The descriptor a render pipeline state was made of.
#[derive(Debug, Clone)]
pub enum RenderDescriptor {
    /// A drawing pipeline's.
    Render(Box<RenderPipelineDescriptor>),
    /// A mesh render pipeline's, with what the state answers of the guest's values the runtime's
    /// descriptor does not hold.
    Mesh(Box<MeshDescriptor>),
}

/// A mesh render pipeline state's descriptor: the runtime's, and the state's own answers.
#[derive(Debug, Clone)]
pub struct MeshDescriptor {
    /// The runtime's descriptor the state was made of.
    pub descriptor: MeshPipelineDescriptor,
    /// What the state answers, and holds its draws to, of the guest's descriptor
    /// ([`MeshAnswers`]).
    pub answers: MeshAnswers,
}

/// What a mesh render pipeline state answers of its stages' limits, and what its draws are held
/// to, from the guest's descriptor values the runtime's `MeshPipelineDescriptor` does not hold
/// (`crate::gpu::states::mesh_answers` decides each).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshAnswers {
    /// Whether the state has an object function: its draws' threads per object threadgroup are
    /// held to the object stage's limits only then, as Metal ignores them without one.
    pub object: bool,
    /// `maxTotalThreadsPerObjectThreadgroup`: 0 without an object function.
    pub object_threads: u64,
    /// `maxTotalThreadsPerMeshThreadgroup`.
    pub mesh_threads: u64,
    /// `objectThreadExecutionWidth`: 0 without an object function.
    pub object_width: u64,
    /// `meshThreadExecutionWidth`.
    pub mesh_width: u64,
    /// `maxTotalThreadgroupsPerMeshGrid`.
    pub grid: u64,
    /// The object and mesh stages' maxima the guest's descriptor set (0 for one it left), and
    /// their required threadgroup sizes (`None` for none): a draw of a threadgroup past a
    /// maximum set, or of another size than a required one, dies
    /// (`crate::gpu::encode_render`'s `threads_held`).
    pub held: [MeshStageHeld; 2],
}

/// What one stage of a mesh render pipeline state holds its draws' threadgroups to
/// ([`MeshAnswers::held`]): the object stage's first, then the mesh stage's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshStageHeld {
    /// The descriptor's maximum threads per threadgroup; 0 where it set none.
    pub max: u64,
    /// The descriptor's required threadgroup size; `None` where it set none.
    pub required: Option<MtlSize>,
}

/// `id<MTLRenderPipelineState>`.
pub type VulkanRenderPipeline = Obj<RenderPipelineObject>;

/// The handle of a state a creator made, with the runtime's identifier of it. Its input primitive
/// topology is Metal's default.
#[cfg(test)]
pub(crate) fn render_pipeline(raw: rt::RenderPipelineState) -> VulkanRenderPipeline {
    render_pipeline_of(raw, PrimitiveTopologyClass::Unspecified, None)
}

/// The handle of a state a creator made, with the runtime's identifier of it, and `descriptor`,
/// where one made it, whose `inputPrimitiveTopology` is `input_primitive_topology`.
pub(crate) fn render_pipeline_of(
    raw: rt::RenderPipelineState,
    input_primitive_topology: PrimitiveTopologyClass,
    descriptor: Option<RenderDescriptor>,
) -> VulkanRenderPipeline {
    let unique_identifier = identified(|| raw.unique_identifier());
    Obj::new(RenderPipelineObject {
        raw,
        unique_identifier,
        input_primitive_topology,
        reflection: Reflection::new(descriptor.map(Reflected::Render)),
        drawn: std::sync::OnceLock::new(),
    })
}

impl RenderPipelineObject {
    /// What a mesh render pipeline state answers ([`MeshAnswers`]); `None` for any other state.
    pub(crate) fn mesh_answers(&self) -> Option<&MeshAnswers> {
        match self.reflection.descriptor() {
            Some(Reflected::Render(RenderDescriptor::Mesh(m))) => Some(&m.answers),
            _ => None,
        }
    }
}

/// A runtime shared texture handle.
#[derive(Debug)]
pub struct SharedTextureHandleObject {
    /// The runtime's handle.
    pub raw: rt::SharedTextureHandle,
    /// Its texture's [`TextureObject::private_usage`], which a texture opened from it answers.
    pub private_usage: u64,
    /// Its texture's hold on Metal's space, which a texture opened from it shares. The shared
    /// texture's range token holds it while the texture, the textures opened from its handles
    /// and the command buffers that hold any of them live, as its allocation does
    /// ([`crate::gpu::metal_space::Hold`]).
    pub space: Arc<Hold>,
}

/// `MTLSharedTextureHandle *`: in this process only, as the runtime's handle is.
pub type VulkanSharedTextureHandle = Obj<SharedTextureHandleObject>;

impl GpuResource<VulkanPlatform> for VulkanBuffer {
    fn device(&self) -> Option<VulkanDevice> {
        Some(self.device.clone_ref())
    }

    fn resource_options(&self) -> u64 {
        plain(|| self.raw.resource_options()).raw()
    }

    /// Labels are diagnostics: the runtime keeps it.
    fn set_label(&self, label: &str) -> RaiseResult<()> {
        plain(|| self.raw.set_label(label));
        Ok(())
    }

    /// The state before the change. A raw state no `MTLPurgeableState` is is refused, as Metal's
    /// validation refuses it.
    fn set_purgeable_state(&self, state: u64) -> RaiseResult<u64> {
        send("setPurgeableState:", Policy::Dies, || {
            self.raw
                .set_purgeable_state(rt::PurgeableState::from_raw(state)?)
        })
        .map(rt::PurgeableState::raw)
    }

    fn storage_mode(&self) -> u64 {
        plain(|| self.raw.storage_mode()).raw()
    }
}

impl Buffer<VulkanPlatform> for VulkanBuffer {
    /// NULL for a `Private` buffer, as Metal's, and for one made on a surface
    /// (`newBufferWithIOSurface:`), as Metal's on every surface, of `Shared` storage all the same
    /// (measured on the M3 Max: a surface of its own memory, and ones over the guest's pages
    /// without `kIOSurfaceClientDirection` and with 2 and 3). A detached buffer, or one placed in
    /// a detached heap, which lies in no memory, answers where it did before
    /// ([`Backing::kept_contents`]).
    fn contents(&self) -> HostPtr {
        if self.surface.is_some() {
            return HostPtr::NULL;
        }
        let at = |p: NonNull<u8>| HostPtr::from_ptr(p.as_ptr());
        plain(|| {
            let kept = match &self.heap {
                Some(heap) => heap
                    .backing
                    .kept_contents()
                    .map(|first| at(first).wrapping_add(self.raw.heap_offset())),
                None => self.backing.kept_contents().map(at),
            };
            kept.or_else(|| self.raw.contents().map(crate::os::shown_at).map(at))
        })
        .unwrap_or(HostPtr::NULL)
    }

    /// The buffer is detached from now on: its `contents` and GPU address stay, and the GPU
    /// reads zeros where it reached the guest's pages, as the runtime's detach leaves it in no
    /// memory. One placed in a heap detaches the heap instead ([`crate::gpu::backing`]).
    fn detach_backing(&self) -> RaiseResult<()> {
        plain(|| match &self.heap {
            Some(heap) => heap.detach(),
            None => self.backing.detach(|| self.raw.detach_backing()),
        });
        Ok(())
    }

    /// Nothing for a buffer that is not `Managed`, as Metal; for a `Managed` one, refused for a
    /// range past the buffer, as Metal's validation refuses it. A range whose end passes
    /// `u64::MAX` dies.
    fn did_modify_range(&self, _on: OnFault, range: NsRange) -> RaiseResult<()> {
        let sel = "didModifyRange:";
        let range = crate::gpu::range(range)
            .unwrap_or_else(|| unrepresentable(sel, &format!("the range {range:?}")));
        send(sel, Policy::Dies, || self.raw.did_modify_range(range))
    }

    fn gpu_address(&self) -> u64 {
        plain(|| self.raw.gpu_address()).0
    }

    /// The surface the buffer was made on, +1; `None` for any other buffer.
    fn iosurface(&self) -> Option<OsSurface> {
        plain(|| self.surface.as_deref().map(Handle::clone_ref))
    }

    fn length(&self) -> u64 {
        plain(|| self.raw.length())
    }

    /// A texture of the buffer's bytes; a 2D array's slices laid out as Metal lays them. Nil for
    /// a format Metal makes no texture of from a descriptor alone, which the runtime answers as
    /// Metal does ([`crate::gpu::no_texture::descriptor`]).
    fn new_linear_texture(
        &self,
        descriptor: &VulkanTextureDescriptor,
        offset: u64,
        bytes_per_row: u64,
        bytes_per_image: u64,
    ) -> RaiseResult<Option<VulkanTexture>> {
        let sel = "newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:";
        let fields = descriptor.fields();
        let mut desc = fields.descriptor(sel, Maker::Other);
        // A format Metal makes no texture of from a descriptor alone makes none: its usage is
        // its validation's.
        let answered = match self.gpu_read_only() && fields.formatless().is_none() {
            true => read_only_usage(&mut desc),
            false => 0,
        };
        let format = fields.raw_pixel_format();
        let Some(raw) = create(sel, Policy::Dies, Nil::Never, || {
            (self
                .raw
                .new_linear_texture(&desc, offset, bytes_per_row, bytes_per_image))
            .map_err(|e| created_format_refused(sel, "a texture of", format, e))
        })?
        else {
            return Ok(None);
        };
        let bytes = texel_bytes(&desc, (offset, bytes_per_row), plain(|| self.raw.length()));
        let under = Under::Buffer(self.clone_ref(), bytes);
        Ok(Some(self.device.texture_in(raw, under, answered)))
    }

    /// A texture of the buffer's bytes from `offset`, rows `bytes_per_row` apart. Refused as
    /// Metal's validation refuses an offset or rows off the format's alignment; nil for a format
    /// Metal makes no texture of from a descriptor alone, which the runtime answers as Metal does
    /// ([`crate::gpu::no_texture::descriptor`]). A 2D array dies, as Metal aborts at it
    /// ([`buffer_texture_type`]).
    fn new_texture(
        &self,
        _on: OnFault,
        descriptor: &VulkanTextureDescriptor,
        offset: u64,
        bytes_per_row: u64,
    ) -> RaiseResult<Option<VulkanTexture>> {
        let sel = "newTextureWithDescriptor:offset:bytesPerRow:";
        let fields = descriptor.fields();
        let mut desc = fields.descriptor(sel, Maker::Other);
        // As the linear texture's.
        let answered = match self.gpu_read_only() && fields.formatless().is_none() {
            true => read_only_usage(&mut desc),
            false => 0,
        };
        let format = fields.raw_pixel_format();
        let Some(raw) = create(sel, Policy::Dies, Nil::Never, || {
            buffer_texture_type(sel, self.raw.new_texture(&desc, offset, bytes_per_row))
                .map_err(|e| created_format_refused(sel, "a texture of", format, e))
        })?
        else {
            return Ok(None);
        };
        let bytes = texel_bytes(&desc, (offset, bytes_per_row), plain(|| self.raw.length()));
        let under = Under::Buffer(self.clone_ref(), bytes);
        Ok(Some(self.device.texture_in(raw, under, answered)))
    }

    /// The runtime's tiled texture (`Buffer::new_tiled_texture`): one level is the texture
    /// [`Buffer::new_texture`] makes; a 2D texture of more levels is one image of every level,
    /// whose first level the runtime keeps in step with the buffer's bytes from `offset`, rows
    /// `bytes_per_row` apart, and whose other levels are the image's own, as an x86 Mac's GPU
    /// keeps them in its own memory (docs/graphics-kext/dgpu-oracle.md, Q11). Refused as
    /// `new_texture`'s arguments are, and for more levels than the extent's full chain; nil for a
    /// format Metal makes no texture of from a descriptor alone, as `new_texture`'s.
    fn new_tiled_texture(
        &self,
        descriptor: &VulkanTextureDescriptor,
        offset: u64,
        bytes_per_row: u64,
    ) -> RaiseResult<Option<VulkanTexture>> {
        let sel = "newTiledTextureWithDescriptor:offset:bytesPerRow:";
        let fields = descriptor.fields();
        let mut desc = fields.descriptor(sel, Maker::Other);
        // As the linear texture's.
        let answered = match self.gpu_read_only() && fields.formatless().is_none() {
            true => read_only_usage(&mut desc),
            false => 0,
        };
        let format = fields.raw_pixel_format();
        let Some(raw) = create(sel, Policy::Dies, Nil::Never, || {
            (self.raw.new_tiled_texture(&desc, offset, bytes_per_row))
                .map_err(|e| created_format_refused(sel, "a texture of", format, e))
        })?
        else {
            return Ok(None);
        };
        let bytes = texel_bytes(&desc, (offset, bytes_per_row), plain(|| self.raw.length()));
        let under = Under::Buffer(self.clone_ref(), bytes);
        Ok(Some(self.device.texture_in(raw, under, answered)))
    }

    /// A texture of the buffer's bytes as Metal's IOSurface creator makes one over the plane of
    /// a surface the buffer stands for (`Buffer::new_texture_with_iosurface_plane`), at the
    /// plane's offset with rows its bytes per row apart: of the descriptor's options, a sampler
    /// ID of the device taken, as Metal's device takes one, and as the device's
    /// `new_texture_on_surface` makes one over an `IOSurfaceRef`, minus its reads of the surface,
    /// which the facts give ([`crate::gpu::device::surface_plane_descriptor`],
    /// [`crate::gpu::device::texture_over_plane`]). A two-plane YCbCr format the runtime reads
    /// and a two-plane XR10 one read the facts of a surface of two planes or more
    /// ([`crate::gpu::device::plane_facts_taken`]): its OSType, which says its range, its size,
    /// its plane 1 at the facts' offset in the buffer with their rows, and its YCbCr matrix,
    /// which the XR10 formats do not read; what the runtime refuses of them dies here, naming the
    /// refusal. A plane past 0 of a surface without planes dies, where Metal asserts, and a plane
    /// past a planar surface's last is nil, as Metal's. Without facts the buffer stands for no
    /// surface: the texture is the buffer's own creator's ([`Buffer::new_texture`]), which is
    /// nil for a two-plane format. The texture may change the bytes of its plane and, of a
    /// two-plane format over plane 0, of plane 1.
    fn new_texture_over_surface_plane(
        &self,
        on: OnFault,
        descriptor: &VulkanTextureDescriptor,
        plane: SurfacePlane,
    ) -> RaiseResult<Option<VulkanTexture>> {
        use crate::gpu::device::{PlaneFacts, plane_facts_taken};
        let Some(facts) = plane.facts else {
            return self.new_texture(on, descriptor, plane.offset, plane.bytes_per_row);
        };
        let sel = "newTextureWithDescriptor:iosurface:plane:";
        let fields = descriptor.fields();
        let device = &self.device.raw;
        let Some(mut desc) = crate::gpu::device::surface_plane_descriptor(device, sel, &fields)?
        else {
            return Ok(None);
        };
        let usage = match self.gpu_read_only() {
            true => read_only_usage(&mut desc),
            false => 0,
        };
        // As over an `IOSurfaceRef` (`textures_over_planes_a_surface_lacks_are_metals`).
        if plane.plane != 0 && facts.plane_count == 0 {
            let what = format!("plane {} of a surface without planes", plane.plane);
            metal_aborts(sel, &what);
        }
        if facts.plane_count != 0 && plane.plane >= facts.plane_count {
            return Ok(None);
        }
        let taken = plane_facts_taken(desc.pixel_format).filter(|_| facts.plane_count >= 2);
        let ycbcr = taken.map(|kind| {
            let size = [facts.width, facts.height];
            let (chroma, rows) = (facts.chroma_offset, facts.chroma_bytes_per_row);
            let read = rt::IoSurfaceYCbCr::new(facts.pixel_format, plane.plane, size, chroma, rows);
            match (kind, facts.ycbcr_matrix) {
                (PlaneFacts::YCbCr, Some(matrix)) => read.with_matrix(ycbcr_matrix(matrix)),
                _ => read,
            }
        });
        let at = rt::IoSurfacePlane::new(plane.offset, plane.bytes_per_row, plane.planar);
        let at = match ycbcr {
            Some(ycbcr) => at.with_ycbcr(ycbcr),
            None => at,
        };
        let raw = crate::gpu::device::texture_over_plane(sel, &fields, &self.raw, &desc, at)?;
        let Some(raw) = raw else {
            return Ok(None);
        };
        let length = plain(|| self.raw.length());
        let mut bytes = texel_bytes(&desc, (plane.offset, plane.bytes_per_row), length);
        if ycbcr.is_some() && plane.plane == 0 {
            let rows = facts.chroma_bytes_per_row.saturating_mul(facts.height);
            let end = facts.chroma_offset.saturating_add(rows).min(length);
            bytes = bytes.start.min(facts.chroma_offset.min(length))..bytes.end.max(end);
        }
        let under = Under::Buffer(self.clone_ref(), bytes);
        Ok(Some(self.device.texture_in(raw, under, usage)))
    }

    /// A detached no-copy buffer over an alias of the ranges, which total its length, from now
    /// on; NO as Metal answers ([`crate::gpu::backing`]). The runtime's refusal dies: core
    /// replaces only its own no-copy buffers, and no command buffer it has not committed holds
    /// a use of one then.
    fn replace_backing(
        &self,
        ranges: &VulkanAddressRangeArray,
        read_only: bool,
    ) -> RaiseResult<bool> {
        let length = plain(|| self.raw.length());
        self.backing
            .replace(&ranges.ranges, read_only, length, length, |first, keep| {
                send(REPLACE, Policy::Dies, || {
                    // SAFETY: the alias is mapped and readable; a read-only one only the GPU
                    // writes, whose writes this host's GPU drops (`backing`'s docs), and any
                    // other is writable. It stays mapped, with the same pages, until its
                    // keep-alive is dropped, as nothing else frees or remaps it.
                    unsafe {
                        (self.raw).rebind_bytes_no_copy(
                            crate::os::host_memory(first),
                            None,
                            Some(keep),
                        )
                    }
                })
            })
    }

    /// The runtime's export of the buffer's memory (`export_memory_fd`): a buffer the runtime
    /// made exportable ([`VulkanDevice`]'s `new_exportable_buffer_of_length`), else refused.
    fn export_memory(&self) -> Result<ExportedMemory, String> {
        let exported = plain(|| self.raw.export_memory_fd()).map_err(|error| error.to_string())?;
        Ok(exported_memory(exported))
    }

    /// The runtime's export of the buffer's memory as a dma-buf (`export_memory_dma_buf`): a
    /// buffer the runtime made so ([`VulkanDevice`]'s
    /// `new_dma_buf_buffer_of_length_padding_unobserved`), else refused.
    fn export_dma_buf(&self) -> Result<ExportedMemory, String> {
        let exported =
            (plain(|| self.raw.export_memory_dma_buf())).map_err(|error| error.to_string())?;
        Ok(exported_memory(exported))
    }

    /// The runtime's acquire of what another API wrote into the buffer's exported memory
    /// (`external_write_done`), which the device's next submission records before its commands,
    /// and after which a shadow of a texture over the buffer's bytes copies them in again at its
    /// next use in a command buffer committed after the call.
    fn external_write_done(&self) -> Result<(), String> {
        plain(|| self.raw.external_write_done()).map_err(|error| error.to_string())
    }
}

/// What an import of a buffer's exported memory names, from the runtime's export of it.
fn exported_memory(exported: rt::ExportedMemory) -> ExportedMemory {
    ExportedMemory {
        fd: exported.fd,
        allocation_size: exported.allocation_size,
        offset: exported.offset,
        length: exported.length,
        dedicated: exported.dedicated,
    }
}

impl GpuResource<VulkanPlatform> for VulkanTexture {
    fn device(&self) -> Option<VulkanDevice> {
        Some(self.device.clone_ref())
    }

    /// The runtime's.
    fn resource_options(&self) -> u64 {
        plain(|| self.raw.resource_options()).raw()
    }

    /// Labels are diagnostics: the runtime keeps it.
    fn set_label(&self, label: &str) -> RaiseResult<()> {
        plain(|| self.raw.set_label(label));
        Ok(())
    }

    /// The state before the change, where the runtime keeps the texture's state as Metal keeps
    /// it; `NonVolatile` where Metal answers that to every request. A raw state no
    /// `MTLPurgeableState` is is refused, as Metal's validation refuses it.
    fn set_purgeable_state(&self, state: u64) -> RaiseResult<u64> {
        send("setPurgeableState:", Policy::Dies, || {
            self.raw
                .set_purgeable_state(rt::PurgeableState::from_raw(state)?)
        })
        .map(rt::PurgeableState::raw)
    }

    /// The runtime's.
    fn storage_mode(&self) -> u64 {
        plain(|| self.raw.storage_mode()).raw()
    }
}

impl VulkanTexture {
    /// Whether a view of this texture in `pixel_format` (raw) is nil ([`no_view`]), but in the
    /// texture's own format and between a two-plane YCbCr format and its sRGB twin, which the
    /// runtime answers ([`crate::gpu::ycbcr::twin_view`]): Metal makes a view of a texture of the
    /// family in its own format, of any type, levels, slices and swizzle the view selectors take
    /// (measured: `own_format_views_are_metals`).
    fn no_view_of(&self, sel: &'static str, pixel_format: u64) -> bool {
        let source = u64::from(plain(|| self.raw.pixel_format()).as_raw());
        source != pixel_format
            && !crate::gpu::ycbcr::twin_view(source, pixel_format)
            && no_view(sel, pixel_format)
    }

    /// The view `call` makes of this texture, for `sel`: of the same device, over the same surface
    /// (sharing this texture's reference of it, as a view shares its texels). Metal raises for no
    /// view (measured, without API validation): nil where the runtime answers it, as Metal
    /// answers ([`no_view`] answers the formats Metal makes no view in first); a view Metal
    /// asserts at ("Texture Creation": a format of another
    /// class, a type of another family, levels or slices outside the texture, a slice count the
    /// type does not take, more than one slice of a texture over a buffer) dies there; any other
    /// refusal dies at the send, a refusal of a format named by the format refused: the texture's,
    /// of `R10Unorm_X6`, `RG10Unorm_X12` or their sRGB twins, which the runtime views in no other
    /// format, but in a two-plane YCbCr one it refuses first, else the view's. The view answers
    /// this texture's usage, and where `swizzled`, of a swizzle other than the default, only its
    /// `ShaderRead`, `PixelFormatView` and Apple bits 0x4000 and 0x10000, as the runtime's view
    /// answers ([`SWIZZLE_DROPPED`]); a view of a texture whose GPU writes never reach its memory
    /// holds all it answers in its [`TextureObject::private_usage`] ([`read_only_usage`]). A
    /// swizzled view of a texture that answers no usage (one of usage 0x8 alone) dies, as Metal
    /// asserts there (`_usage != MTLTextureUsageUnknown`, measured on the M3 Max under macOS
    /// 27.0.1), where a view in a format alone answers 0 as the texture does.
    fn view(
        &self,
        sel: &'static str,
        format: u64,
        swizzled: bool,
        call: impl FnOnce(&rt::Texture) -> Result<rt::Texture, rt::Error>,
    ) -> RaiseResult<Option<VulkanTexture>> {
        use rt::ViewProblem as P;
        if swizzled && Texture::usage(self) == 0 {
            metal_aborts(sel, "a swizzled view of a texture of no usage");
        }
        let raw = match plain_at(sel, || call(&self.raw)) {
            Ok(raw) => raw,
            Err(rt::Error::Invalid(rt::Invalid {
                problem: rt::Problem::TextureView(problem),
                ..
            })) if matches!(
                problem,
                P::FormatIncompatible
                    | P::TypeIncompatible
                    | P::LevelRange
                    | P::SliceRange
                    | P::SliceCount
                    | P::BufferSlices
                    | P::SwizzledViewOfNoUsage
            ) =>
            {
                metal_aborts(sel, &format!("a view it does not make ({problem:?})"))
            }
            Err(rt::Error::Nil(_)) => return Ok(None),
            Err(e) => {
                // The runtime refuses a view of `R10Unorm_X6`, `RG10Unorm_X12` or their sRGB
                // twins in another format by the texture's format, though its refusal names the
                // view's, but a view in a two-plane YCbCr format, which it refuses by that format
                // first (such a view is nil before it reaches the runtime, `no_view_of`).
                use metal2vulkan::metal::format::{padded_words, ycbcr};
                let source = plain(|| self.raw.pixel_format());
                let view = crate::gpu::held_format(sel, format).ok();
                let padded = padded_words(source) && !view.is_some_and(ycbcr);
                let e = match padded {
                    true => {
                        let what = format!("a view in pixel format {format} of a texture of");
                        created_format_refused(sel, &what, u64::from(source.as_raw()), e)
                    }
                    false => created_format_refused(sel, "a view of", format, e),
                };
                if relay_gap(sel, &e, None) {
                    return Ok(None);
                }
                refused(sel, &e)
            }
        };
        Ok(Some(Obj::new(TextureObject {
            raw,
            device: self.device.clone_ref(),
            surface: self.surface.clone(),
            plane: self.plane,
            under: self.under.as_ref().map(Under::clone_ref),
            private_usage: match swizzled && self.gpu_read_only() {
                true => self.private_usage & !u64::from(SWIZZLE_DROPPED.as_raw()),
                false => self.private_usage,
            },
            space: Arc::new(Hold::sharing(Arc::clone(&self.space))),
            gpu_read_only: self.gpu_read_only,
        })))
    }
}

/// The levels or slices `r` names, for `sel`; a range whose end passes `u64::MAX` dies.
fn indices(sel: &'static str, r: NsRange) -> std::ops::Range<u64> {
    crate::gpu::range(r).unwrap_or_else(|| unrepresentable(sel, &format!("the range {r:?}")))
}

/// Whether a view in `pixel_format` (raw) is nil, as Metal answers: in a format it makes no view
/// in from a texture of a format the runtime has ([`crate::gpu::no_texture::no_view`]), and in a
/// value Metal describes no format for ([`crate::gpu::format_info::described`]) but 255 and 262,
/// `MTLPixelFormatInvalid`'s 0 among them (measured: `views_metal_answers_nil_are_nil`). Before
/// anything else of the view is checked, as Metal's: its swizzle, type, levels and slices.
fn no_view(sel: &'static str, pixel_format: u64) -> bool {
    if plain(|| crate::gpu::no_texture::no_view(pixel_format)) {
        return true;
    }
    if matches!(pixel_format, 255 | 262) {
        return false;
    }
    !crate::gpu::format_info::described(sel, pixel_format)
}

/// The runtime's format of a view in `raw`, which `sel` makes, a value [`no_view`] is not nil in:
/// one this platform keeps raw ([`crate::gpu::held_format`]), 255 and 262 among them, is one
/// Metal describes no format for, as the runtime has every format Metal describes, and dies as
/// Metal's assertion ([`crate::gpu::desc::lacked_as`]: "texture view pixelFormat
/// (MTLPixelFormatInvalid) not castable", measured: `views_metal_answers_nil_are_nil`).
fn view_format(sel: &'static str, raw: u64) -> PixelFormat {
    crate::gpu::held_format(sel, raw)
        .unwrap_or_else(|raw| crate::gpu::desc::lacked_as(sel, "a view of", raw, None))
}

/// The resource ID placement of a view `sel` makes at `index`.
fn at(index: u64) -> rt::IdPlacement {
    rt::IdPlacement::At(rt::ResourceId(index))
}

impl Texture<VulkanPlatform> for VulkanTexture {
    fn array_length(&self) -> u64 {
        plain(|| self.raw.array_length())
    }

    fn depth(&self) -> u64 {
        plain(|| self.raw.depth())
    }

    fn gpu_resource_id(&self) -> u64 {
        plain(|| self.raw.gpu_resource_id()).0
    }

    fn height(&self) -> u64 {
        plain(|| self.raw.height())
    }

    /// The surface the texture (or the texture it views) was made on, +1; `None` for any other.
    fn iosurface(&self) -> Option<OsSurface> {
        plain(|| self.surface.as_deref().map(Handle::clone_ref))
    }

    fn is_shareable(&self) -> bool {
        plain(|| self.raw.is_shareable())
    }

    fn mipmap_level_count(&self) -> u64 {
        plain(|| self.raw.mipmap_level_count())
    }

    /// An in-process handle; nil for a texture that is not shareable, as Metal answers.
    fn new_shared_texture_handle(&self) -> RaiseResult<Option<VulkanSharedTextureHandle>> {
        let sel = "newSharedTextureHandle";
        let made = plain_at(sel, || self.raw.new_shared_texture_handle());
        Ok(made.ok().map(|raw| {
            Obj::new(SharedTextureHandleObject {
                raw,
                private_usage: self.private_usage,
                space: Arc::clone(&self.space),
            })
        }))
    }

    fn new_texture_view(&self, pixel_format: u64) -> RaiseResult<Option<VulkanTexture>> {
        let sel = "newTextureViewWithPixelFormat:";
        if self.no_view_of(sel, pixel_format) {
            return Ok(None);
        }
        let format = view_format(sel, pixel_format);
        self.view(sel, pixel_format, false, |t| t.new_texture_view(format))
    }

    fn new_texture_view_at_index(
        &self,
        pixel_format: u64,
        resource_index: u64,
    ) -> RaiseResult<Option<VulkanTexture>> {
        let sel = "newTextureViewWithPixelFormat:resourceIndex:";
        if self.no_view_of(sel, pixel_format) {
            return Ok(None);
        }
        let format = view_format(sel, pixel_format);
        self.view(sel, pixel_format, false, |t| {
            t.new_texture_view_with_placement(format, at(resource_index))
        })
    }

    fn new_texture_subview(
        &self,
        pixel_format: u64,
        texture_type: u64,
        level_range: NsRange,
        slice_range: NsRange,
    ) -> RaiseResult<Option<VulkanTexture>> {
        let sel = "newTextureViewWithPixelFormat:textureType:levels:slices:";
        if self.no_view_of(sel, pixel_format) {
            return Ok(None);
        }
        let kind = crate::gpu::texture_type(sel, texture_type);
        let (levels, slices) = (indices(sel, level_range), indices(sel, slice_range));
        let format = view_format(sel, pixel_format);
        self.view(sel, pixel_format, false, |t| {
            t.new_texture_view_with_levels_slices(format, kind, levels, slices)
        })
    }

    fn new_swizzled_texture_subview(
        &self,
        pixel_format: u64,
        texture_type: u64,
        level_range: NsRange,
        slice_range: NsRange,
        swizzle: Swizzle,
    ) -> RaiseResult<Option<VulkanTexture>> {
        let sel = "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:";
        if self.no_view_of(sel, pixel_format) {
            return Ok(None);
        }
        let kind = crate::gpu::texture_type(sel, texture_type);
        let (levels, slices) = (indices(sel, level_range), indices(sel, slice_range));
        let swizzle = crate::gpu::swizzle(sel, swizzle);
        let format = view_format(sel, pixel_format);
        self.view(sel, pixel_format, !swizzle.is_default(), |t| {
            t.new_texture_view_with_swizzle(format, kind, levels, slices, swizzle)
        })
    }

    fn new_swizzled_texture_subview_at_index(
        &self,
        pixel_format: u64,
        texture_type: u64,
        levels: NsRange,
        slices: NsRange,
        swizzle: Swizzle,
        resource_index: u64,
    ) -> RaiseResult<Option<VulkanTexture>> {
        let sel = "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:resourceIndex:";
        if self.no_view_of(sel, pixel_format) {
            return Ok(None);
        }
        let kind = crate::gpu::texture_type(sel, texture_type);
        let (levels, slices) = (indices(sel, levels), indices(sel, slices));
        let swizzle = crate::gpu::swizzle(sel, swizzle);
        let format = view_format(sel, pixel_format);
        self.view(sel, pixel_format, !swizzle.is_default(), |t| {
            t.new_texture_subview_with_placement(
                format,
                kind,
                levels,
                slices,
                Some(swizzle),
                at(resource_index),
            )
        })
    }

    fn new_texture_subview_at_index(
        &self,
        pixel_format: u64,
        texture_type: u64,
        levels: NsRange,
        slices: NsRange,
        resource_index: u64,
    ) -> RaiseResult<Option<VulkanTexture>> {
        let sel = "newTextureViewWithPixelFormat:textureType:levels:slices:resourceIndex:";
        if self.no_view_of(sel, pixel_format) {
            return Ok(None);
        }
        let kind = crate::gpu::texture_type(sel, texture_type);
        let (levels, slices) = (indices(sel, levels), indices(sel, slices));
        let format = view_format(sel, pixel_format);
        self.view(sel, pixel_format, false, |t| {
            t.new_texture_subview_with_placement(
                format,
                kind,
                levels,
                slices,
                None,
                at(resource_index),
            )
        })
    }

    /// The runtime's.
    fn pixel_format(&self) -> u64 {
        u64::from(plain(|| self.raw.pixel_format()).as_raw())
    }

    fn protection_options(&self) -> u64 {
        plain(|| self.raw.protection_options())
    }

    /// The runtime reads `pixel_bytes` in place, through their pointer ([`guest_bytes`]), each
    /// byte once. Refused as Metal's validation refuses a region outside the level, rows that do
    /// not cover it, or a Private texture. One of a texture over memory the runtime imports
    /// through a writable twin of read-only pages, but a surface's
    /// ([`TextureObject::gpu_read_only()`]), dies: the runtime writes through the twin, which
    /// reaches the pages, where Metal's CPU writes through their read-only mapping
    /// ([`crate::gpu::writes::kept_write`]). A surface's reads a copy of them first
    /// ([`host_bytes`]), which it writes row by row.
    fn replace_region(
        &self,
        region: MtlRegion,
        level: u64,
        slice: u64,
        pixel_bytes: HostRange<'_>,
        bytes_per_row: u64,
        bytes_per_image: u64,
    ) -> RaiseResult<()> {
        let sel = "replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:";
        let raw = self.copied(sel);
        if self.gpu_read_only() {
            let bytes = host_bytes(sel, pixel_bytes);
            let rows = (bytes_per_row, bytes_per_image);
            return self.replaced_on_surface(sel, raw, (region, level, slice), &bytes, rows);
        }
        let bytes = guest_bytes(sel, pixel_bytes);
        if crate::os::TWINS_READ_ONLY && crate::gpu::writes::texture_kept(self) {
            crate::gpu::writes::kept_write(sel, "a CPU write of a texture into");
        }
        send(sel, Policy::Dies, || {
            raw.replace_region_with_host_bytes(
                crate::gpu::region(region),
                level,
                slice,
                bytes,
                bytes_per_row,
                bytes_per_image,
            )
        })
    }

    /// The texture's resource ID: on Metal a texture's `resourceIndex` is its `gpuResourceID`
    /// (measured).
    fn resource_index(&self) -> u64 {
        plain(|| self.raw.gpu_resource_id()).0
    }

    fn sample_count(&self) -> u64 {
        plain(|| self.raw.sample_count())
    }

    fn texture_type(&self) -> u64 {
        u64::from(plain(|| self.raw.texture_type()).as_raw())
    }

    /// Its resource ID, the runtime's `gpuResourceID`: Metal's `uniqueIdentifier` of a texture is
    /// its `gpuResourceID`, and the runtime's IDs follow Metal's (measured on this host, where
    /// both run the same sequence: `texture_identifiers_are_metals` and
    /// `texture_identifiers_are_reused_as_metals`). Each creator's texture takes the lowest free
    /// ID (the device's, a view, a heap's, one over a buffer or a surface, a shareable one), none
    /// of a buffer, a heap, a sampler state, a command queue or a nil texture does, another
    /// thread's takes the next, and a placed or forced one takes the index it is given. A
    /// texture's release frees its ID, a view's included, and one released while a command buffer
    /// holds it frees it once that completed.
    fn unique_identifier(&self) -> u64 {
        plain(|| self.raw.gpu_resource_id()).0
    }

    /// The runtime's, with the bits beyond the runtime texture's it answers
    /// ([`TextureObject::private_usage`]); for one whose GPU writes never reach its memory, whose
    /// runtime texture is made of another usage, its whole usage alone ([`read_only_usage`]).
    fn usage(&self) -> u64 {
        if self.gpu_read_only() {
            return self.private_usage;
        }
        u64::from(plain(|| self.raw.usage()).as_raw()) | self.private_usage
    }

    fn width(&self) -> u64 {
        plain(|| self.raw.width())
    }
}

/// A copy of the bytes of `range`, for a write of this platform's own that takes a slice. A
/// non-empty range at a NULL or near-NULL pointer, in the zero region (`ZERO_REGION_END`), or one
/// that wraps past the top of the address space, dies at `sel`, as Metal's read of it faults:
/// core's fill of a texture with bytes reads a buffer's `contents` plus an offset, NULL plus the
/// offset for a buffer made on a surface, and Metal's `fillTexture:level:slice:region:bytes:
/// length:` ends the process with a segmentation fault there (measured on the M3 Max at offsets
/// 0, 0x10 and 0x4000, on surfaces of their own memory and over the guest's pages).
///
/// The bytes are guest memory, which another vCPU or GPU work in flight may write while they are
/// read: `HostRange` promises them mapped and readable for the call, never unchanged, so no
/// shared slice is made over them. They are copied once, through the pointer, into memory this
/// platform owns. A runtime call reads them in place instead ([`guest_bytes`]).
pub(crate) fn host_bytes(sel: &'static str, range: HostRange<'_>) -> Vec<u8> {
    first_bytes(sel, range, usize::MAX)
}

/// The bytes of `range`, for a runtime call that reads them through their pointer, each byte
/// once, into memory of its own (`rt::HostBytes`: `replaceRegion:` and `newBufferWithBytes:`),
/// with no copy of this platform's first. Where the range lies dies as [`host_bytes`] dies.
pub(crate) fn guest_bytes<'a>(sel: &'static str, range: HostRange<'a>) -> rt::HostBytes<'a> {
    let length =
        usize::try_from(range.len()).unwrap_or_else(|_| unrepresentable(sel, "that length"));
    if length == 0 {
        return rt::HostBytes::from_slice(&[]);
    }
    let first = range.ptr().addr();
    let start = NonNull::new(range.ptr().as_ptr::<u8>())
        .filter(|_| first >= crate::core::host::ZERO_REGION_END)
        .filter(|_| first.checked_add(length).is_some());
    let Some(start) = start else {
        crate::answer::metal_reads_near_null(sel, &format!("{length} bytes at {first:#x}"));
    };
    // SAFETY: `HostRange::new`'s contract keeps `start..start + length` mapped and readable for
    // the call the bytes are given to, which reads them only while it runs and keeps nothing of
    // the pointer. They are guest memory, which the guest's vCPUs and GPU work write outside the
    // program's Rust while they are read, as `HostBytes::new` allows: the call copies each byte
    // through the pointer, and no reference covers them.
    unsafe { rt::HostBytes::new(start, length) }
}

/// The first `most` bytes of `range` (all of them where it holds fewer), copied as [`host_bytes`]
/// copies them, whose checks of where the range lies take all of it.
fn first_bytes(sel: &'static str, range: HostRange<'_>, most: usize) -> Vec<u8> {
    let length =
        usize::try_from(range.len()).unwrap_or_else(|_| unrepresentable(sel, "that length"));
    if length == 0 {
        return Vec::new();
    }
    let start = range.ptr().as_ptr::<u8>();
    let first = range.ptr().addr();
    if first < crate::core::host::ZERO_REGION_END || first.checked_add(length).is_none() {
        crate::answer::metal_reads_near_null(sel, &format!("{length} bytes at {first:#x}"));
    }
    let read = length.min(most);
    let mut copy = Vec::with_capacity(read);
    // SAFETY: `HostRange::new`'s contract keeps `start..start + length` mapped and readable for
    // the call, and `read` is at most `length`; `copy` owns `read` bytes of its own, which no
    // other pointer reaches, so the two do not overlap. The copy is made through raw pointers
    // alone: no reference ever covers the guest's bytes, which may change under it. `set_len`
    // follows the copy, which wrote every one of the `read` bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(start, copy.as_mut_ptr(), read);
        copy.set_len(read);
    }
    copy
}

/// The bytes of a fill of `region` with `range` (`fillTexture:level:slice:region:bytes:length:`),
/// copied as [`host_bytes`] copies them: none for a region of no width and no height, of which
/// Metal fills nothing and reads no byte; for any other, Metal reads them whatever the length, so
/// a range starting in the zero region dies there, empty or not. Measured on the M3 Max (macOS
/// 26.7) with a `newBufferWithIOSurface:` buffer's NULL contents plus 0 and 0x10: regions of 0 by
/// 0 texels, of depth 0 or 1 and of 16 or 0 bytes, complete, filling nothing; regions of 0 by 4,
/// 4 by 0, 1 by 1 and 4 by 4 texels, of depth 0 or 1, and of 4 by 4 texels and 0 bytes, end the
/// process with a segmentation fault.
///
/// Only the range's first `most` bytes are copied, the most the fill reads
/// ([`crate::gpu::fill::MOST_PATTERN_BYTES`], or a texture buffer's
/// [`crate::gpu::fill::MOST_STREAMED_PATTERN_BYTES`]), so a range the guest sizes to the whole of
/// a large buffer is no copy of it; the checks of where it lies take all of it.
pub(crate) fn fill_bytes(
    sel: &'static str,
    region: MtlRegion,
    range: HostRange<'_>,
    most: usize,
) -> Vec<u8> {
    if region.size.width == 0 && region.size.height == 0 {
        return Vec::new();
    }
    let first = range.ptr().addr();
    if range.is_empty() && first < crate::core::host::ZERO_REGION_END {
        crate::answer::metal_reads_near_null(sel, &format!("0 bytes at {first:#x}"));
    }
    first_bytes(sel, range, most)
}

/// What the runtime answers making a texture over a buffer by `sel`,
/// `newTextureWithDescriptor:offset:bytesPerRow:`: a 2D array, which the runtime refuses as
/// Metal's validation stops it (`BufferTextureProblem::TextureType`), dies as Metal's
/// assertion, which stops it whatever handler is live (measured on the M3 Max, macOS 26.7, with
/// `MTL_DEBUG_LAYER` unset: "_mtlValidateStrideTextureParameters:1927: failed assertion `Texture
/// Descriptor Validation Linear texture: must be of type MTLTextureType2D, textureType
/// (MTLTextureType2DArray) disallowed'"). `newLinearTextureWithDescriptor:offset:bytesPerRow:
/// bytesPerImage:` makes one, laid out as Metal's (`array_over_host_memory_is_metals`). Any other
/// type the runtime refuses dies naming its refusal.
fn buffer_texture_type<T>(sel: &'static str, made: Result<T, rt::Error>) -> Result<T, rt::Error> {
    match made {
        Err(rt::Error::Invalid(rt::Invalid {
            problem:
                rt::Problem::BufferTexture(rt::BufferTextureProblem::TextureType {
                    texture_type: TextureType::Type2DArray,
                }),
            ..
        })) => metal_aborts(sel, ARRAY_OVER_A_BUFFER),
        made => made,
    }
}

/// What Metal stops at [`buffer_texture_type`]'s death: its assertion's text.
const ARRAY_OVER_A_BUFFER: &str = "a 2D array over a buffer (\"Texture Descriptor Validation \
     Linear texture: must be of type MTLTextureType2D, textureType (MTLTextureType2DArray) \
     disallowed\")";

/// The options of a resource placed in a heap, with its hazard tracking mode the default: Metal
/// gives a heap's resources the heap's tracking mode whatever they name (measured: a tracked
/// buffer or texture in an untracked heap is untracked), where the runtime refuses another.
fn with_heap_tracking(sel: &'static str, options: u64) -> rt::ResourceOptions {
    crate::gpu::resource_options(sel, options & !HAZARD_TRACKING_BITS)
}

/// `MTLResourceOptions`' hazard tracking mode, bits 8 and 9.
const HAZARD_TRACKING_BITS: u64 = 0x300;

/// What the runtime answers placing a resource in a heap, for `sel`: a storage or CPU cache mode
/// other than the heap's dies, as Metal asserts at it ("The requested storage mode … is not
/// compatible with the heap's mode", measured).
fn placed<T>(sel: &'static str, made: Result<T, rt::Error>) -> Result<T, rt::Error> {
    match made {
        Err(rt::Error::Invalid(rt::Invalid {
            problem:
                rt::Problem::Heap(
                    rt::HeapProblem::StorageMode { .. } | rt::HeapProblem::CpuCacheMode { .. },
                ),
            ..
        })) => metal_aborts(sel, "a resource of a mode other than its heap's"),
        made => made,
    }
}

impl Heap<VulkanPlatform> for VulkanHeap {
    /// The heap is detached from now on: its buffers' `contents` and its GPU address stay, and
    /// the GPU reads zeros where it reached the guest's pages ([`crate::gpu::backing`]).
    fn detach_backing(&self) -> RaiseResult<()> {
        plain(|| self.detach());
        Ok(())
    }

    fn gpu_address(&self) -> u64 {
        plain(|| self.raw.gpu_address()).0
    }

    /// A buffer `offset` bytes into a placement heap, of the heap's options
    /// ([`with_heap_tracking`]). Nil for an offset off 256 bytes and for bytes past the heap's
    /// end, as Metal answers (measured); a storage or CPU cache mode other than the heap's dies
    /// ([`placed`]). A length of 0 makes a buffer of no bytes at the offset, the heap's end
    /// included, as Metal does (measured on the M3 Max, a Shared heap of 64 KiB: length and
    /// `allocatedSize` 0, its GPU address and contents the offset into the heap's, at 0, 256,
    /// 65280 and 65536; nil at 100 and 65792). One over a byte of a texture of the heap that
    /// shares no byte while it lives dies ([`Placements::place`]); the heap's cover, core's
    /// buffer over all of it, the first asked of it, is kept as the cover, beside which such a
    /// texture dies ([`Placements`]).
    fn new_buffer_of_length(
        &self,
        length: u64,
        options: u64,
        offset: u64,
    ) -> RaiseResult<Option<VulkanBuffer>> {
        let sel = "newBufferWithLength:options:offset:";
        let options = with_heap_tracking(sel, options);
        let first = plain(|| self.placed.lock().attempt());
        create(sel, Policy::Dies, Nil::Placed, || {
            placed(sel, self.raw.new_buffer_at_offset(length, options, offset))
        })
        .map(|b| {
            b.map(|raw| {
                let buffer = self.device.placed_buffer(raw, self);
                let range = offset..offset.saturating_add(length);
                let cover = first && offset == 0 && length >= self.descriptor_size;
                let object: Weak<dyn std::any::Any + Send + Sync> = Obj::downgrade(&buffer);
                plain(|| self.placed.lock().place_buffer(sel, range, object, cover));
                buffer
            })
        })
    }

    /// A texture `offset` bytes into a placement heap, of the heap's options
    /// ([`with_heap_tracking`]). Nil for an offset off the alignment the runtime answers for the
    /// descriptor and for bytes past the heap's end, as Metal answers (measured, at its own
    /// alignment); a storage or CPU cache mode other than the heap's dies ([`placed`]). Nil for a
    /// format Metal makes no texture of from a descriptor alone, which the runtime answers as
    /// Metal does, before the heap's own rules ([`crate::gpu::no_texture::descriptor`]). One over
    /// a byte of a two-plane XR10 texture of the heap that lives apart from its bytes dies, as
    /// does such a texture over a byte of any buffer or texture of the heap that lives, the
    /// heap's cover, core's buffer over all of it, included: one whose creation the runtime
    /// announced `heap-two-plane-texture` ([`Placements::place`]). So does one over a byte of a
    /// depth or stencil texture without `allowGPUOptimizedContents`, and such a texture over
    /// another's, and a nil inside the heap of such a texture, whose size the runtime answers
    /// larger than Metal's (`optimized-depth-stencil-texture`), where Metal's own size fits there
    /// or is not modelled; where it does not fit either, the nil is Metal's
    /// ([`HeapObject::optimized_nil`]); and one over a
    /// byte of a texture over the guest's pages that the runtime does not lay out in Metal's
    /// tiles, and such a texture over another's (`heap-texture-untiled-layout`,
    /// [`Alone::Untiled`]). Each of the three kinds dies, too, where the heap's cover lives, so in
    /// every heap core makes ([`Placements`]). A 2- or 4-sample texture that Metal keeps losslessly
    /// compressed, whose image the runtime keeps apart from the heap's bytes over the guest's
    /// pages (`heap-texture-lossless-layout`), dies as it is made ([`crate::answer::HEARD`]). So
    /// does, as it is placed or answered nil, a texture whose layout in Metal's heaps the runtime
    /// does not model, placed by
    /// its device's image's size and alignment, larger or smaller than Metal's
    /// (`unmeasured-heap-texture-size`, [`crate::gpu::metal_space::unmeasured_size`]): no suite
    /// case or guest run places one. Its nil at an offset at or past the heap's end is Metal's,
    /// whatever the size, and is answered.
    ///
    /// An optimized texture made checks the runtime's size against Metal's where the runtime
    /// models Metal's (`crate::gpu::device::VulkanDevice::metal_answer_at`): one made where
    /// Metal's alignment misses the offset or Metal's bytes pass the heap's end, where Metal
    /// answers nil, dies, and so does one the runtime answers fewer bytes for than Metal's, whose
    /// range would end before Metal's texture does.
    ///
    /// In a heap over no address ranges, core's legacy heap, whose bytes core copies to and from
    /// the guest's pages, what the runtime announces of a texture's layout there
    /// (`heap-texture-untiled-layout`, `heap-texture-lossless-layout`), its bytes in the heap's
    /// memory not Metal's, is relayed alone ([`Placements`]).
    fn new_texture(
        &self,
        descriptor: &VulkanTextureDescriptor,
        offset: u64,
    ) -> RaiseResult<Option<VulkanTexture>> {
        let sel = "newTextureWithDescriptor:offset:";
        let fields = descriptor.fields();
        let mut desc = fields.descriptor(sel, Maker::Other);
        desc.resource_options = with_heap_tracking(sel, desc.resource_options.raw());
        plain(|| self.placed.lock().attempt());
        let format = fields.raw_pixel_format();
        let (made, heard) = hearing(|| {
            create(sel, Policy::Dies, Nil::Placed, || {
                let made = self.raw.new_texture_at_offset(&desc, offset);
                placed(
                    sel,
                    made.map_err(|e| created_format_refused(sel, "a texture of", format, e)),
                )
            })
        });
        // A nil at or past the heap's end is Metal's whatever the texture's size.
        let past_the_end = || matches!(made, Ok(None)) && offset >= plain(|| self.raw.size());
        if heard.contains(&rt::Degradation::UnmeasuredHeapTextureSize) && !past_the_end() {
            crate::gpu::metal_space::unmeasured_size(
                sel,
                "a placement in a heap, by its size, of",
                &desc,
            );
        }
        let optimized = heard.contains(&rt::Degradation::OptimizedDepthStencilTexture);
        if optimized && matches!(made, Ok(None)) && offset < plain(|| self.raw.size()) {
            self.optimized_nil(sel, &desc, offset);
        }
        made.map(|t| {
            t.map(|raw| {
                // Over the guest's pages alone, as an untiled one is (below).
                let lossless = heard.contains(&rt::Degradation::HeapTextureLosslessLayout)
                    && self.backing.detaches();
                if lossless && desc.sample_count > 1 {
                    crate::answer::inexact(sel, crate::answer::Inexact::HeapLayout, || {
                        format!(
                            "a {}-sample texture in a heap that Metal keeps losslessly \
                             compressed, whose image the runtime keeps in memory of its own, \
                             where Metal's lies in the heap's bytes, so a buffer or texture over \
                             them would neither see nor change it (heap-texture-lossless-layout)",
                            desc.sample_count
                        )
                    });
                }
                // Its bytes in the heap, the size the runtime placed it by, which its
                // `allocatedSize` answers with no query; the rest of the heap where it has none.
                let size = plain(|| raw.allocated_size());
                let end = size.map_or(u64::MAX, |size| offset.saturating_add(size));
                let heap_size = plain(|| self.raw.size());
                if optimized {
                    self.optimized_made(sel, &desc, offset, size);
                }
                let bytes = offset.min(heap_size)..end.min(heap_size);
                let under = Under::Heap(self.clone_ref(), bytes.clone());
                let texture = self.device.texture_in(raw, under, 0);
                // Over the guest's pages alone: in a legacy heap, over the device's memory, what
                // the runtime announces of the texture's layout there (untiled, or lossless where
                // Metal compresses it) is relayed alone.
                let untiled = heard.contains(&rt::Degradation::HeapTextureUntiledLayout)
                    && self.backing.detaches();
                let alone = if heard.contains(&rt::Degradation::HeapTwoPlaneTexture) {
                    Some(Alone::TwoPlane)
                } else if optimized {
                    Some(Alone::Optimized)
                } else {
                    untiled.then_some(Alone::Untiled)
                };
                let object = texture.life();
                plain(|| self.placed.lock().place(sel, bytes, object, alone));
                texture
            })
        })
    }

    /// A detached heap over an alias of the ranges, which total its descriptor's size, from now
    /// on; NO as Metal answers ([`crate::gpu::backing`]). Every buffer and texture in it stays,
    /// at its offset. The runtime's refusal dies: core replaces only heaps it made over address
    /// ranges, and no command buffer it has not committed holds a use of one then.
    fn replace_backing(
        &self,
        ranges: &VulkanAddressRangeArray,
        read_only: bool,
    ) -> RaiseResult<bool> {
        let size = plain(|| self.raw.size());
        self.backing.replace(
            &ranges.ranges,
            read_only,
            self.descriptor_size,
            size,
            |first, keep| {
                send(REPLACE, Policy::Dies, || {
                    // SAFETY: as a buffer's, for the heap's size; the alias starts on a page,
                    // which is the runtime's import unit on this host.
                    unsafe {
                        (self.raw).rebind_bytes_no_copy(
                            crate::os::host_memory(first),
                            None,
                            Some(keep),
                        )
                    }
                })
            },
        )
    }

    /// The heap's modes, an untracked mode where its descriptor named the default, as Metal's.
    fn resource_options(&self) -> u64 {
        plain(|| self.raw.resource_options()).raw()
    }

    /// The state before the change, as a buffer's ([`GpuResource::set_purgeable_state`]).
    fn set_purgeable_state(&self, state: u64) -> RaiseResult<u64> {
        send("setPurgeableState:", Policy::Dies, || {
            self.raw
                .set_purgeable_state(rt::PurgeableState::from_raw(state)?)
        })
        .map(rt::PurgeableState::raw)
    }

    fn storage_mode(&self) -> u64 {
        plain(|| self.raw.storage_mode()).raw()
    }
}

/// The state's ID is the runtime's. Its `resourceIndex` is its `gpuResourceID`, as Metal's is on
/// this host: the suite's expected files show both written back equal (`resourceIndex` 4 and
/// `gpuResourceID` 4 in `mutgap_tex_placement_reservations`, 7 and 7 in
/// `render_objects_placement_sampler`).
impl SamplerState for VulkanSamplerState {
    fn gpu_resource_id(&self) -> u64 {
        plain(|| rt::SamplerState::gpu_resource_id(self)).0
    }

    fn resource_index(&self) -> u64 {
        plain(|| rt::SamplerState::gpu_resource_id(self)).0
    }
}

/// The getters below Apple10 are the host's stubs' (measured on the M3 Max): `gpuResourceID` is the runtime's,
/// what plain Metal answers, 0; `resourceIndex` raises, and the runtime has no index of its own.
/// So the guest's info request, which reads both, raises, as on Metal.
impl DepthStencilState for VulkanDepthStencilState {
    fn gpu_resource_id(&self) -> RaiseResult<u64> {
        send("gpuResourceID", Policy::Caught, || {
            rt::DepthStencilState::gpu_resource_id(self)
        })
        .map(|id| id.0)
    }

    fn resource_index(&self) -> RaiseResult<u64> {
        plain_at("resourceIndex", || ());
        Err(Raise::platform("resourceIndex"))
    }
}

/// The getters are the runtime's, `resourceIndex` among them: 0 for a state without indirect
/// command buffer support, as Metal's is (`compute_pipeline_placement`), and the state's
/// `gpuResourceID` for one with it, as Metal's is (measured: compute and render states with
/// support answer the ID for both, 1 and 2 in a fresh process).
impl ComputePipeline for VulkanComputePipeline {
    fn gpu_resource_id(&self) -> u64 {
        plain(|| rt::ComputePipelineState::gpu_resource_id(self)).0
    }

    fn imageblock_memory_length(&self, imageblock_dimensions: MtlSize) -> RaiseResult<u64> {
        let dimensions = gpu::size(imageblock_dimensions);
        Ok(plain(|| {
            self.imageblock_memory_length_for_dimensions(dimensions)
        }))
    }

    fn max_total_threads_per_threadgroup(&self) -> u64 {
        u64::from(plain(|| {
            rt::ComputePipelineState::max_total_threads_per_threadgroup(self)
        }))
    }

    fn resource_index(&self) -> u64 {
        plain(|| rt::ComputePipelineState::resource_index(self))
    }

    fn static_threadgroup_memory_length(&self) -> u64 {
        u64::from(plain(|| {
            rt::ComputePipelineState::static_threadgroup_memory_length(self)
        }))
    }

    fn support_indirect_command_buffers(&self) -> bool {
        plain(|| rt::ComputePipelineState::support_indirect_command_buffers(self))
    }

    fn thread_execution_width(&self) -> u64 {
        u64::from(plain(|| {
            rt::ComputePipelineState::thread_execution_width(self)
        }))
    }

    /// The runtime's identifier of the state, read when it was made.
    fn unique_identifier(&self) -> u64 {
        plain(|| self.unique_identifier)
    }

    fn guest_unique_identifier(&self) -> u64 {
        let indirect = ComputePipeline::support_indirect_command_buffers(self);
        guest_identifier(self.unique_identifier, indirect)
    }
}

/// The getters are the runtime's, and the unique identifier the one its creation took.
/// `resourceIndex` answers as the compute state's does: 0 for a state without indirect command
/// buffer support, as Metal's is (`info_replies`' 466 reads index and ID 0), and its
/// `gpuResourceID` for one with it, as Metal's is (`mutgap3_res_render_pipeline_forced_index`
/// reads {3, 3}).
impl RenderPipeline for VulkanRenderPipeline {
    fn gpu_resource_id(&self) -> u64 {
        plain(|| self.raw.gpu_resource_id()).0
    }

    /// The runtime's answer, which is Metal's: `info_replies`' 458 reads 256 for 8 by 8 of a
    /// pipeline of one RGBA8Unorm attachment, with no fault. Metal raises at none, so any refusal
    /// of the runtime dies at the selector.
    fn imageblock_memory_length(&self, imageblock_dimensions: MtlSize) -> RaiseResult<u64> {
        let dimensions = gpu::size(imageblock_dimensions);
        send("imageblockMemoryLengthForDimensions:", Policy::Dies, || {
            self.raw.imageblock_memory_length_for_dimensions(dimensions)
        })
    }

    fn imageblock_sample_length(&self) -> u64 {
        get("imageblockSampleLength", || {
            self.raw.imageblock_sample_length()
        })
    }

    fn max_total_threads_per_threadgroup(&self) -> u64 {
        u64::from(plain(|| self.raw.max_total_threads_per_threadgroup()))
    }

    /// A mesh render pipeline state's answer, which its creation decided
    /// (`crate::gpu::states::mesh_answers`); 0 for any other state, as Metal's answers.
    fn max_total_threads_per_object_threadgroup(&self) -> u64 {
        self.mesh_answers().map_or(0, |m| m.object_threads)
    }

    /// As [`Self::max_total_threads_per_object_threadgroup`].
    fn max_total_threads_per_mesh_threadgroup(&self) -> u64 {
        self.mesh_answers().map_or(0, |m| m.mesh_threads)
    }

    /// As [`Self::max_total_threads_per_object_threadgroup`].
    fn object_thread_execution_width(&self) -> u64 {
        self.mesh_answers().map_or(0, |m| m.object_width)
    }

    /// As [`Self::max_total_threads_per_object_threadgroup`].
    fn mesh_thread_execution_width(&self) -> u64 {
        self.mesh_answers().map_or(0, |m| m.mesh_width)
    }

    /// As [`Self::max_total_threads_per_object_threadgroup`].
    fn max_total_threadgroups_per_mesh_grid(&self) -> u64 {
        self.mesh_answers().map_or(0, |m| m.grid)
    }

    fn resource_index(&self) -> u64 {
        plain(|| self.raw.resource_index())
    }

    fn support_indirect_command_buffers(&self) -> bool {
        plain(|| self.raw.support_indirect_command_buffers())
    }

    fn threadgroup_size_matches_tile_size(&self) -> bool {
        plain(|| self.raw.threadgroup_size_matches_tile_size())
    }

    fn unique_identifier(&self) -> u64 {
        self.unique_identifier
    }

    fn guest_unique_identifier(&self) -> u64 {
        guest_identifier(
            self.unique_identifier,
            self.support_indirect_command_buffers(),
        )
    }
}

impl Library<VulkanPlatform> for VulkanLibrary {
    fn function_names(&self) -> Vec<String> {
        plain(|| self.raw.function_names())
    }

    /// Nil for a name the library does not hold, as Metal answers.
    fn new_function(
        &self,
        _on: OnFault,
        function_name: &str,
    ) -> RaiseResult<Option<VulkanFunction>> {
        let sel = "newFunctionWithName:";
        crate::gpu::timing::creation(
            sel,
            "",
            || {
                create(sel, Policy::Dies, Nil::MissingFunction, || {
                    self.raw.new_function_with_name(function_name)
                })
            },
            || format!("{function_name:?}"),
            |made| match made {
                Ok(Some(_)) => "found".to_owned(),
                Ok(None) => "nil".to_owned(),
                Err(_) => "raised".to_owned(),
            },
        )
        .map(|made| made.map(Obj::new))
    }

    /// The runtime's `newFunctionWithName:constantValues:error:`: each constant by its index.
    /// A constant whose type the runtime does not know, or no function constant takes, answers
    /// nil, as a refused specialisation.
    fn new_function_with_constants(
        &self,
        _on: OnFault,
        function_name: &str,
        constants: &[crate::core::platform::resource::FunctionConstant],
    ) -> RaiseResult<Option<VulkanFunction>> {
        let sel = "newFunctionWithName:constantValues:error:";
        let Some(values) = constant_values(function_name, constants) else {
            return Ok(None);
        };
        crate::gpu::timing::creation(
            sel,
            "",
            || {
                create(sel, Policy::Dies, Nil::MissingFunction, || {
                    self.raw
                        .new_function_with_name_constant_values(function_name, &values)
                })
            },
            || format!("{function_name:?} with {} constants", constants.len()),
            |made| match made {
                Ok(Some(_)) => "found".to_owned(),
                Ok(None) => "nil".to_owned(),
                Err(_) => "raised".to_owned(),
            },
        )
        .map(|made| made.map(Obj::new))
    }

    /// The runtime's `newFunctionWithDescriptor:error:`: the descriptor's constant values nil
    /// where there are none, and each constant by its index where there are; its specialized
    /// name where it has one, which the function answers as its name and links under, as
    /// Metal's does. A constant the runtime cannot carry answers nil, as a refused
    /// specialisation; options the runtime does not know die, as its other creators' unknown
    /// raw values do.
    fn new_function_with_options(
        &self,
        _on: OnFault,
        function_name: &str,
        constants: &[crate::core::platform::resource::FunctionConstant],
        options: u64,
        specialized_name: Option<&str>,
    ) -> RaiseResult<Option<VulkanFunction>> {
        let sel = "newFunctionWithDescriptor:error:";
        let mut descriptor = rt::FunctionDescriptor::new(function_name);
        if let Some(specialized) = specialized_name {
            descriptor = descriptor.with_specialized_name(specialized);
        }
        if !constants.is_empty() {
            let Some(values) = constant_values(function_name, constants) else {
                return Ok(None);
            };
            descriptor.constant_values = Some(values);
        }
        descriptor.options = gpu::metal_value(sel, "function options", options);
        crate::gpu::timing::creation(
            sel,
            "",
            || {
                create(sel, Policy::Dies, Nil::MissingFunction, || {
                    self.raw.new_function_with_descriptor(&descriptor)
                })
            },
            || {
                format!(
                    "{function_name:?} with {} constants, options {options:#x}, specialized \
                     name {specialized_name:?}",
                    constants.len()
                )
            },
            |made| match made {
                Ok(Some(_)) => "found".to_owned(),
                Ok(None) => "nil".to_owned(),
                Err(_) => "raised".to_owned(),
            },
        )
        .map(|made| made.map(Obj::new))
    }
}

/// The runtime's constant values of `constants`, each by its index: `None`, logged, where one is
/// of a type the runtime does not know or no function constant takes.
fn constant_values(
    function_name: &str,
    constants: &[crate::core::platform::resource::FunctionConstant],
) -> Option<metal2vulkan::library::FunctionConstantValues> {
    let mut values = metal2vulkan::library::FunctionConstantValues::defaults();
    for c in constants {
        let (Ok(ty), Some(bytes)) = (
            metal2vulkan::metal::DataType::try_from(c.data_type),
            c.bytes(),
        ) else {
            log::warn!(
                "function {function_name:?}: constant {} of MTLDataType {} cannot be carried",
                c.index,
                c.data_type
            );
            return None;
        };
        values = values.with_index(
            c.index,
            metal2vulkan::library::FunctionConstantValue::new(ty, bytes.to_vec()),
        );
    }
    Some(values)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::alloc::{Layout, alloc_zeroed, dealloc};
    use std::sync::atomic::{AtomicU32, Ordering};

    use crate::core::platform::Platform;
    use crate::core::platform::descriptor::TextureDescriptor as _;
    use crate::core::platform::device::{GpuDevice, NoCopyDeallocator};
    use crate::core::platform::os::surface::{Surface, SurfacePlaneSpec, SurfaceSpec};
    use crate::core::platform::types::MtlOrigin;
    use crate::core::platform::values::{BufferDesc, HeapDesc, HeapMade, OptionSet, TextureDesc};
    use crate::core::raise::death::DeathKind;
    use crate::core::sync::Mutex;
    use crate::core::testkit::capture::catch_death;

    use super::*;
    use crate::gpu::device::tests::dev;
    use crate::os::tests::bgra_surface;
    use crate::platform::tests::platform;

    /// A no-copy buffer core makes over `pages` pages of zeros mapped read-only
    /// (`newBufferWithBytesNoCopy`), which the runtime makes over a copy of them whose GPU writes
    /// it drops on Linux (announcing `copied-host-pages`, which dies in strict mode, so it is made
    /// under the default), and the pages' first byte, which the caller frees once the buffer is
    /// gone.
    #[cfg(target_os = "linux")]
    fn over_read_only_pages(pages: u64) -> (VulkanBuffer, HostPtr) {
        use crate::core::platform::os::vm::{HostVm as _, VmProt};

        use crate::os::{OsVm, PAGE};

        let at = crate::os::allocate_aliasable(pages * PAGE).unwrap();
        assert!(
            OsVm::new()
                .protect_max(at, pages * PAGE, VmProt::READ)
                .is_ok()
        );
        // SAFETY: the pages outlive the buffer, which the caller drops before freeing them.
        let bytes = unsafe { HostRange::new(at, pages * PAGE) };
        let b = crate::answer::relaxed(|| {
            dev().new_buffer_over_host_memory(OnFault::Return, bytes, 0, None)
        })
        .unwrap()
        .expect("a buffer over read-only pages, as Metal makes");
        (b, at)
    }

    /// A buffer made on a surface whose GPU writes never reach its memory
    /// (`kIOSurfaceClientDirection` 2) over a page of zeros of the guest's, which the runtime
    /// imports through a writable twin of the surface's read-only alias on Linux
    /// ([`crate::gpu::backing::ImportSite::twins`]), and the page's first byte, which the caller
    /// frees once the buffer is gone.
    #[cfg(target_os = "linux")]
    pub(crate) fn on_read_only_surface() -> (VulkanBuffer, HostPtr) {
        use crate::core::host::PhysicalMemoryRange;
        use crate::core::num::ToU64 as _;
        use crate::core::platform::os::surface::{IoSurfaces as _, SurfaceSpec};

        use crate::os::{OsSurfaces, PAGE, READ_ONLY_DIRECTION};

        let guest = crate::os::allocate_aliasable(PAGE).unwrap();
        let spec = SurfaceSpec {
            pixel_format: u32::from_be_bytes(*b"BGRA"),
            alloc_size: PAGE,
            width: 16,
            height: 16,
            bytes_per_element: 4,
            bytes_per_row: 64,
            element_width: 1,
            element_height: 1,
            client_direction: Some(READ_ONLY_DIRECTION),
            client_address_ranges: Some(vec![PhysicalMemoryRange {
                physical_address: guest.addr().to_u64(),
                physical_length: PAGE,
            }]),
            ..SurfaceSpec::default()
        };
        let surface = OsSurfaces::new().create(&spec).unwrap();
        assert!(crate::os::gpu_read_only(&surface));
        let b = dev()
            .new_buffer_on_surface(Some(&surface))
            .unwrap()
            .expect("a buffer on the surface, as Metal makes");
        assert!(crate::gpu::writes::buffer_kept(&b));
        (b, guest)
    }

    /// `MTLResourceOptions` bit 24, which core sets on a read-only detached buffer or heap.
    pub(crate) const BIT_24: u64 = 0x100_0000;

    /// A no-copy buffer a descriptor names over `pages` pages of zeros of the platform's own,
    /// made with bit 24 ([`BIT_24`]), as the guest makes a read-only one: the runtime makes it
    /// over a copy of the pages whose GPU writes it drops on Linux (announcing
    /// `copied-host-pages`, which dies in strict mode, so it is made under the default), and
    /// imports a read-only view of them on macOS; and the pages' first byte, which the caller
    /// frees once the buffer is gone.
    pub(crate) fn read_only_buffer(pages: u64) -> (VulkanBuffer, HostPtr) {
        use crate::os::PAGE;

        let at = crate::os::allocate_aliasable(pages * PAGE).unwrap();
        // SAFETY: the pages outlive the buffer, which the caller drops before freeing them.
        let bytes = unsafe { HostRange::new(at, pages * PAGE) };
        let b = crate::answer::relaxed(|| {
            dev().new_buffer(BufferDesc {
                length: pages * PAGE,
                options: vec![
                    OptionSet::StorageMode(0),
                    OptionSet::ResourceOptionsOr(BIT_24),
                ],
                pinned_gpu_address: None,
                contents: Some(bytes),
                no_copy: true,
                deallocator: None,
            })
        })
        .unwrap()
        .expect("a read-only buffer, as Metal makes");
        (b, at)
    }

    /// A 16 × 16 `BGRA8Unorm` texture of `usage` over `b` from its start, its rows 64 bytes apart.
    pub(crate) fn bgra_over(b: &VulkanBuffer, usage: u64) -> VulkanTexture {
        let desc = platform().texture_descriptor(&TextureDesc {
            pixel_format: 80,
            width: 16,
            height: 16,
            usage,
            resource_options: Some(0),
            ..TextureDesc::default()
        });
        b.new_texture(OnFault::Die, &desc, 0, 64).unwrap().unwrap()
    }

    /// The `n` bytes from `at`, read after no GPU work uses them.
    pub(crate) fn host_bytes_at(at: HostPtr, n: usize) -> Vec<u8> {
        // SAFETY: pages the caller mapped readable, which no GPU work uses now.
        (0..n)
            .map(|i| unsafe { at.wrapping_add(i as u64).as_ptr::<u8>().read_volatile() })
            .collect()
    }

    /// A no-copy buffer core makes over pages mapped read-only, made where Metal makes it,
    /// answers its own address as its contents and takes a GPU fill into it, the command buffer
    /// completing with no error, and the pages stay zeros: on Linux the runtime makes it over a
    /// copy of the pages whose GPU writes it drops, as Metal drops a fill (`os/linux/import.rs`,
    /// `crate::gpu::writes`'s docs), so nothing of it is kept ([`crate::gpu::writes::Kept`])
    #[cfg(target_os = "linux")]
    #[test]
    fn a_fill_into_read_only_pages_is_dropped_on_linux() {
        use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
        use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use crate::core::platform::os::vm::HostVm as _;
        use crate::core::platform::resource::Buffer as _;
        use crate::core::platform::types::NsRange;

        use crate::os::PAGE;

        let (b, pages) = over_read_only_pages(2);
        assert_eq!(b.contents(), pages, "its own address");
        assert!(b.backing.read_only_now() && !crate::gpu::writes::buffer_kept(&b));
        let q = dev().new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        let fill = BlitCmd::FillBuffer {
            buffer: &b,
            range: NsRange::new(0x10, 0x20),
            value: 0xa5,
        };
        e.encode(OnFault::Die, fill).unwrap();
        e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
        // SAFETY: the pages, mapped readable, which no GPU work uses now.
        let read = |at: u64| unsafe { pages.wrapping_add(at).as_ptr::<u8>().read_volatile() };
        assert_eq!((read(0x10), read(0x2f)), (0, 0));
        drop((cb, e, b));
        assert!(
            crate::remap::VulkanVm::new()
                .deallocate(pages, 2 * PAGE)
                .is_success()
        );
    }

    /// A guest's read-only buffer (a no-copy buffer made with bit 24, [`read_only_buffer`])
    /// takes a blit's fill into it, and on Linux a copy into it too, the command buffer
    /// completing with no error, and its pages stay as they were: the runtime drops the writes,
    /// through its copy of the pages on Linux and Metal's read-only view on macOS, as Metal
    /// drops a fill (measured, `crate::gpu::writes`'s docs). What Metal does with a copy into
    /// such memory is unmeasured, so a Mac, whose GPU every process shares, runs only the fill.
    /// On Linux a blit into a texture over it dies where it is encoded, as what Metal does
    /// there is unmeasured. Its contents answer the pages' address on both
    #[test]
    fn a_blit_into_a_read_only_buffer_is_dropped() {
        use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
        use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use crate::core::platform::os::vm::HostVm as _;
        use crate::core::platform::resource::Buffer as _;
        use crate::core::platform::types::{MtlSize, NsRange};

        use crate::os::PAGE;

        let (b, pages) = read_only_buffer(1);
        assert_eq!(b.contents(), pages);
        assert_eq!(b.backing.read_only_now(), crate::os::COPIES_READ_ONLY);
        assert!(!crate::gpu::writes::buffer_kept(&b));
        let d = dev();
        let source = d
            .new_buffer_of_length(OnFault::Die, 0x40, 0)
            .unwrap()
            .unwrap();
        // SAFETY: a Shared buffer of 0x40 bytes, which no command uses yet.
        unsafe { std::ptr::write_bytes(source.contents().as_ptr::<u8>(), 0x3c, 0x40) };
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        let fill = BlitCmd::FillBuffer {
            buffer: &b,
            range: NsRange::new(0x10, 0x20),
            value: 0xa5,
        };
        e.encode(OnFault::Die, fill).unwrap();
        if crate::os::COPIES_READ_ONLY {
            let copy = BlitCmd::CopyBufferToBuffer {
                source_buffer: Some(&source),
                source_offset: 0,
                destination_buffer: Some(&b),
                destination_offset: 0x100,
                size: 0x40,
            };
            e.encode(OnFault::Die, copy).unwrap();
        }
        let texture = bgra_over(&b, 0x1);
        if crate::os::COPIES_READ_ONLY {
            let into_texture = BlitCmd::CopyBufferToTexture {
                source_buffer: &source,
                source_offset: 0,
                source_bytes_per_row: 16,
                source_bytes_per_image: 0x40,
                source_size: MtlSize::new(4, 4, 1),
                destination_texture: &texture,
                destination_slice: 0,
                destination_level: 0,
                destination_origin: MtlOrigin::new(0, 0, 0),
            };
            let sel = into_texture.selector();
            let site = catch_death(|| e.encode(OnFault::Die, into_texture)).unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
        }
        e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert_eq!(cb.status().0, 4);
        assert!(cb.error().is_none(), "{:?}", cb.error());
        assert!(host_bytes_at(pages, 0x200).iter().all(|&byte| byte == 0));
        drop((cb, e, texture, b));
        assert!(
            crate::remap::VulkanVm::new()
                .deallocate(pages, PAGE)
                .is_success()
        );
    }

    /// A CPU write of a texture over a guest's read-only buffer ([`read_only_buffer`]) dies at
    /// `replaceRegion:` on Linux, where Metal's CPU write through the read-only mapping ends the
    /// process, and the runtime's would land in its copy
    #[cfg(target_os = "linux")]
    #[test]
    fn a_cpu_write_of_a_texture_over_a_read_only_buffer_dies_on_linux() {
        use crate::core::platform::os::vm::HostVm as _;
        use crate::core::platform::resource::Texture as _;
        use crate::core::platform::types::{MtlRegion, MtlSize};

        use crate::os::PAGE;

        let (b, pages) = read_only_buffer(1);
        let texture = bgra_over(&b, 0x1);
        let texels = [0x77u8; 64];
        let region = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(16, 1, 1));
        let write = || texture.replace_region(region, 0, 0, HostRange::of_slice(&texels), 64, 0);
        let site = catch_death(write).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (
                DeathKind::Abort,
                "replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:"
            )
        );
        assert!(host_bytes_at(pages, 64).iter().all(|&byte| byte == 0));
        drop((texture, b));
        assert!(
            crate::remap::VulkanVm::new()
                .deallocate(pages, PAGE)
                .is_success()
        );
    }

    /// A host range is copied whole into memory of the platform's own, which later writes to the
    /// range do not reach; an empty range is no bytes, even at NULL; a non-empty range in the
    /// zero region, NULL or near it, or wrapping past the top of the address space, dies at the
    /// selector as a NULL access
    #[test]
    fn host_bytes_are_a_copy() {
        let mut memory = [1u8, 2, 3, 4, 5];
        let copy = host_bytes("s", HostRange::of_slice(&memory));
        memory[0] = 9;
        assert_eq!((copy, memory[0]), (vec![1, 2, 3, 4, 5], 9));
        // SAFETY: an empty range is never read.
        let empty = unsafe { HostRange::new(HostPtr::NULL, 0) };
        assert!(host_bytes("s", empty).is_empty());
        let last = u64::try_from(crate::core::host::ZERO_REGION_END - 1).unwrap();
        for at in [0, 0x10, 0x4000, last] {
            // SAFETY: a range of 4 bytes in the zero region, which the copy dies before reading.
            let near_null = unsafe { HostRange::new(HostPtr::NULL.wrapping_add(at), 4) };
            let site = catch_death(|| host_bytes("s", near_null)).unwrap_err();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::NullDeref, "s"),
                "{at:#x}"
            );
        }
        // SAFETY: a range that wraps past the top of the address space, which the copy dies
        // before reading.
        let wraps = unsafe { HostRange::new(HostPtr::from_addr(usize::MAX - 1), 4) };
        let site = catch_death(|| host_bytes("s", wraps)).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::NullDeref, "s"));
    }

    /// A device cache made as the host's video makes those it lands its pictures in (an
    /// exportable `Private` buffer, NVDEC's) of a 4K NV12 frame as CoreVideo lays it, its luma
    /// plane 32 bytes in and its chroma plane after its rows, with that first plane's byte placed
    /// for textures in place (`aligned_byte`), takes the textures over both planes in its memory:
    /// once the device's checks of its exported memory are known, the runtime's answer for each
    /// (`Buffer::texture_placement_with_iosurface_plane`, the making's own decision) is a linear
    /// image of the cache's memory, not a shadow, which copies the plane in before each command
    /// buffer's use of it. Made without the byte, as every exportable cache was, the planes lie 32
    /// bytes past the runtime's boundary, off every linear image's alignment measured (128 on an
    /// RTX 5080 Laptop GPU, 256 on RADV, 64 on lavapipe), and their textures were shadows: that
    /// answer is printed, not checked. Linux's: MoltenVK exports no buffer's memory.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_device_cache_lays_its_planes_in_its_memory_from_its_first_planes_byte() {
        const WIDTH: u64 = 3840;
        const HEIGHT: u64 = 2160;
        const LUMA_AT: u64 = 0x20;
        const CHROMA_AT: u64 = LUMA_AT + WIDTH * HEIGHT;
        const LENGTH: u64 = CHROMA_AT + WIDTH * HEIGHT / 2;
        /// `MTLResourceStorageModePrivate`.
        const PRIVATE: u64 = 2 << 4;
        let d = dev();
        let sel = "newTextureWithDescriptor:iosurface:plane:";
        // R8Unorm over plane 0 and RG8Unorm over plane 1, sampled, as a frame's decode names them.
        let planes = [
            (10, WIDTH, HEIGHT, LUMA_AT),
            (30, WIDTH / 2, HEIGHT / 2, CHROMA_AT),
        ];
        let placements = |aligned: Option<u64>| {
            let made = d.new_exportable_buffer_of_length(OnFault::Return, LENGTH, PRIVATE, aligned);
            let cache = made.expect("no raise").expect("an exportable buffer");
            let placement = |&(pixel_format, width, height, at): &(u64, u64, u64, u64)| {
                let desc = platform().texture_descriptor(&TextureDesc {
                    pixel_format,
                    width,
                    height,
                    usage: 1,
                    resource_options: Some(PRIVATE),
                    ..TextureDesc::default()
                });
                let made =
                    crate::gpu::device::surface_plane_descriptor(&d.raw, sel, &desc.fields());
                let desc = made.expect("no raise").expect("a plane's descriptor");
                let plane = rt::IoSurfacePlane::new(at, WIDTH, true);
                let answer = cache
                    .raw
                    .texture_placement_with_iosurface_plane(&desc, plane);
                answer.expect("an answer")
            };
            // The first answers start the device's checks of its exported memory.
            let _ = planes.map(|p| placement(&p));
            let checks = std::time::Duration::from_secs(60);
            assert!(
                d.raw.settle_in_place_checks(checks),
                "the device's checks answered"
            );
            planes.map(|p| placement(&p))
        };
        let unplaced = placements(None);
        eprintln!("the planes' textures over a cache made with no aligned byte: {unplaced:?}");
        assert_eq!(
            placements(Some(LUMA_AT)),
            [rt::TexturePlacement::NotHostPages; 2],
            "both planes' textures in the cache's memory"
        );
    }

    /// The bytes a runtime call reads in place are the range itself, at its pointer and of its
    /// length; an empty range is no bytes, even at NULL; a non-empty range in the zero region, or
    /// wrapping past the top of the address space, dies at the selector as a NULL access, as a
    /// copy of it does
    #[test]
    fn guest_bytes_are_the_range_in_place() {
        let memory = [1u8, 2, 3, 4, 5];
        let bytes = guest_bytes("s", HostRange::of_slice(&memory));
        let at = (bytes.pointer().as_ptr().cast_const(), bytes.len());
        assert_eq!(at, (memory.as_ptr(), 5));
        // SAFETY: an empty range is never read.
        let empty = unsafe { HostRange::new(HostPtr::NULL, 0) };
        assert!(guest_bytes("s", empty).is_empty());
        let last = u64::try_from(crate::core::host::ZERO_REGION_END - 1).unwrap();
        for at in [0, 0x10, last] {
            // SAFETY: a range of 4 bytes in the zero region, which dies before any read.
            let near_null = unsafe { HostRange::new(HostPtr::NULL.wrapping_add(at), 4) };
            let site = catch_death(|| guest_bytes("s", near_null)).unwrap_err();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::NullDeref, "s"),
                "{at:#x}"
            );
        }
        // SAFETY: a range that wraps past the top of the address space, which dies before any
        // read.
        let wraps = unsafe { HostRange::new(HostPtr::from_addr(usize::MAX - 1), 4) };
        let site = catch_death(|| guest_bytes("s", wraps)).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::NullDeref, "s"));
    }

    /// A fill's bytes are the first of its range a fill may read, however long the range: one
    /// the guest sizes to a large buffer is no copy of it; a shorter range is copied whole, and
    /// one that wraps past the top of the address space dies at the selector as a NULL access,
    /// though the bytes read lie below the top
    #[test]
    fn a_fill_copies_only_the_bytes_it_may_read() {
        let region = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(4, 4, 1));
        let long: Vec<u8> = (0..=255u8).cycle().take(1 << 20).collect();
        let most = crate::gpu::fill::MOST_PATTERN_BYTES;
        let bytes = fill_bytes("s", region, HostRange::of_slice(&long), most);
        assert_eq!(bytes, long[..most]);
        let streamed = crate::gpu::fill::MOST_STREAMED_PATTERN_BYTES;
        let bytes = fill_bytes("s", region, HostRange::of_slice(&long), streamed);
        assert_eq!(bytes, long[..streamed]);
        let short = [7u8, 8, 9];
        assert_eq!(
            fill_bytes("s", region, HostRange::of_slice(&short), most),
            short
        );
        // SAFETY: a range that wraps past the top of the address space, from a real start, which
        // the fill dies at before reading.
        let wraps =
            unsafe { HostRange::new(HostPtr::from_ptr(long.as_ptr().cast_mut()), u64::MAX) };
        let site = catch_death(|| fill_bytes("s", region, wraps, most)).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::NullDeref, "s"));
    }

    /// A Shared buffer of `length` bytes.
    pub(crate) fn shared(length: u64) -> VulkanBuffer {
        dev()
            .new_buffer_of_length(OnFault::Return, length, 0)
            .unwrap()
            .unwrap()
    }

    /// Two pages of zeroed host memory at a page boundary, and its layout.
    fn pages() -> (*mut u8, Layout) {
        let layout = Layout::from_size_align(32768, 16384).unwrap();
        // SAFETY: a layout of non-zero size.
        (unsafe { alloc_zeroed(layout) }, layout)
    }

    /// newBufferWithLength:options: makes a buffer of the length and options, of the device;
    /// nil for 0 and past the largest; a Private buffer has no contents
    #[test]
    fn buffers_of_length() {
        let d = dev();
        let b = shared(256);
        // Shared storage, the default cache mode, tracked hazards: Metal's 0x200.
        assert_eq!(
            (b.length(), b.storage_mode(), b.resource_options()),
            (256, 0, 0x200)
        );
        assert_ne!(b.contents(), HostPtr::NULL);
        assert_ne!(b.gpu_address(), 0);
        assert_eq!(GpuResource::device(&b).unwrap().id(), d.id());
        assert!(b.iosurface().is_none());
        let private = d
            .new_buffer_of_length(OnFault::Return, 64, 0x20)
            .unwrap()
            .unwrap();
        assert_eq!(
            (private.storage_mode(), private.contents()),
            (2, HostPtr::NULL)
        );
        for length in [0, u64::MAX] {
            let made = d.new_buffer_of_length(OnFault::Die, length, 0).unwrap();
            assert!(made.is_none(), "{length}");
        }
        b.set_label("reims-vgpu.test").unwrap();
    }

    /// A no-copy buffer's contents are the memory, at any offset into a page; its deallocator
    /// runs once, at the buffer's release; a refused one (length 0) is nil and never runs
    #[test]
    fn buffers_over_host_memory() {
        let d = dev();
        let (mem, layout) = pages();
        let calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&calls);
        let deallocator: NoCopyDeallocator = Box::new(move || {
            c.fetch_add(1, Ordering::SeqCst);
        });
        // SAFETY: the memory outlives the buffer (freed at the end).
        let bytes = unsafe { HostRange::new(HostPtr::from_ptr(mem.wrapping_add(100)), 5000) };
        let b = d
            .new_buffer_over_host_memory(OnFault::Return, bytes, 0, Some(deallocator))
            .unwrap()
            .unwrap();
        assert_eq!(b.contents(), bytes.ptr());
        assert_eq!(b.length(), 5000);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        drop(b);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let c = Arc::clone(&calls);
        let deallocator: NoCopyDeallocator = Box::new(move || {
            c.fetch_add(1, Ordering::SeqCst);
        });
        // SAFETY: as above; nothing is made of it.
        let empty = unsafe { HostRange::new(HostPtr::from_ptr(mem), 0) };
        let made = d
            .new_buffer_over_host_memory(OnFault::Return, empty, 0, Some(deallocator))
            .unwrap();
        assert!(made.is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        // SAFETY: allocated above with `layout`; no buffer is left over it.
        unsafe { dealloc(mem, layout) };
    }

    /// newBufferWithDescriptor: makes a no-copy buffer at a pinned address, whose deallocator
    /// gets the pointer and the length; zero and copied contents too
    #[test]
    fn buffers_of_descriptors() {
        let d = dev();
        let (mem, layout) = pages();
        let seen = Arc::new(Mutex::new(None));
        // SAFETY: the memory outlives the buffer.
        let bytes = unsafe { HostRange::new(HostPtr::from_ptr(mem), 32768) };
        // At a range freed just before, long enough for the pin's span too. Another test's
        // buffer may take it first, which answers nil (the refused deallocator never runs), so a
        // few ranges are tried.
        let (b, pin) = (0..16)
            .find_map(|_| {
                let pin = shared(65536).gpu_address();
                let s = Arc::clone(&seen);
                let made = d
                    .new_buffer(BufferDesc {
                        length: 32768,
                        options: vec![
                            OptionSet::StorageMode(0),
                            OptionSet::ResourceOptionsOr(0x100),
                        ],
                        pinned_gpu_address: Some(pin),
                        contents: Some(bytes),
                        no_copy: true,
                        deallocator: Some(Box::new(move |p, len| *s.lock() = Some((p, len)))),
                    })
                    .unwrap();
                made.map(|b| (b, pin))
            })
            .expect("a buffer pinned at a freed range");
        assert_eq!((b.gpu_address(), b.contents()), (pin, bytes.ptr()));
        assert_eq!(b.resource_options(), 0x100);
        drop(b);
        assert_eq!(*seen.lock(), Some((bytes.ptr(), 32768)));
        let zero = d
            .new_buffer(BufferDesc {
                length: 64,
                options: vec![OptionSet::ResourceOptions(0x10)],
                pinned_gpu_address: None,
                contents: None,
                no_copy: false,
                deallocator: None,
            })
            .unwrap()
            .unwrap();
        assert_eq!(zero.storage_mode(), 1);
        let words = [7u8; 16];
        let copied = d
            .new_buffer(BufferDesc {
                length: 16,
                options: vec![],
                pinned_gpu_address: None,
                contents: Some(HostRange::of_slice(&words)),
                no_copy: false,
                deallocator: None,
            })
            .unwrap()
            .unwrap();
        // SAFETY: a Shared buffer's 16 bytes, which nothing else touches.
        let read = unsafe { std::slice::from_raw_parts(copied.contents().as_ptr::<u8>(), 16) };
        assert_eq!(read, &words);
        // SAFETY: allocated above with `layout`; the buffer over it is gone.
        unsafe { dealloc(mem, layout) };
    }

    /// A pin Metal's space refuses is nil only once the runtime has checked the rest of the
    /// descriptor ([`crate::gpu::metal_space`]): a heap pinned in a live buffer's range is nil,
    /// and one of memoryless storage pinned there aborts at its selector, as anywhere
    #[test]
    fn a_pin_the_space_refuses_is_checked_after_the_descriptor() {
        let d = dev();
        let held = shared(0x4000);
        let heap = |storage| {
            d.new_heap(&HeapDesc {
                size: 0x4000,
                heap_type: 0,
                options: vec![OptionSet::StorageMode(storage)],
                address_ranges: None,
                pinned_gpu_address: Some(held.gpu_address()),
            })
        };
        assert!(heap(0).unwrap().heap.is_none());
        let site = catch_death(|| heap(3).map(|made| made.heap.is_some())).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "newHeapWithDescriptor:")
        );
    }

    /// A buffer released while a committed command buffer uses it keeps its range of Metal's
    /// space, so a buffer made meanwhile lies elsewhere, until the command buffer completes: a
    /// buffer made once the wait returns lies at its address again (`vaprobe inflight`, plain
    /// Metal), as the runtime frees the range then ([`crate::gpu::metal_space::Hold`])
    #[test]
    fn a_released_buffers_range_comes_back_once_its_command_buffer_completes() {
        use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
        use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use crate::core::platform::event::{EventRef, SharedEvent as _};
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let gate = d.new_shared_event().unwrap();
        let held = shared(0x4000);
        let at = held.gpu_address();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        cb.encode_wait(Some(EventRef::Shared(&gate)), 1).unwrap();
        let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        let fill = BlitCmd::FillBuffer {
            buffer: &held,
            range: NsRange::new(0, 16),
            value: 0x5c,
        };
        e.encode(OnFault::Return, fill).unwrap();
        e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Return).unwrap();
        drop(held);
        let meanwhile = shared(0x4000);
        assert_ne!(meanwhile.gpu_address(), at);
        gate.set_signaled_value(1).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!(shared(0x4000).gpu_address(), at);
    }

    /// A heap's range comes back once it goes, and a heap pinned there then is made
    #[test]
    fn a_released_heaps_range_comes_back() {
        let d = dev();
        let heap = |pinned_gpu_address| {
            d.new_heap(&HeapDesc {
                size: 0x8000,
                heap_type: 0,
                options: vec![OptionSet::StorageMode(0)],
                address_ranges: None,
                pinned_gpu_address,
            })
            .unwrap()
            .heap
        };
        let first = heap(None).unwrap();
        let at = first.gpu_address();
        assert!(heap(Some(at)).is_none());
        drop(first);
        assert_eq!(heap(Some(at)).unwrap().gpu_address(), at);
    }

    /// Core's fill of a texture with the bytes of a buffer made on a surface reads the buffer's
    /// NULL contents plus the offset, and dies at the fill as a NULL access, where Metal's read of
    /// the bytes ends the process with a segmentation fault, of any length; one of a region of no
    /// width and no height, whose bytes Metal never reads, fills nothing and goes on
    #[test]
    fn a_fill_with_a_surface_buffers_bytes_dies() {
        use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
        use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use crate::core::platform::types::{MtlOrigin, MtlRegion, MtlSize};
        let d = dev();
        let s = bgra_surface(64, 32);
        let b = d.new_buffer_on_surface(Some(&s)).unwrap().unwrap();
        let t = d
            .new_texture(OnFault::Return, &bgra(8, 8, 0))
            .unwrap()
            .unwrap();
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        // SAFETY: 16 bytes at the buffer's NULL contents plus 16, which the fill dies before
        // reading.
        let bytes = unsafe { HostRange::new(b.contents().wrapping_add(16), 16) };
        let fill = BlitCmd::FillTexture {
            texture: &t,
            level: 0,
            slice: 0,
            region: MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(4, 1, 1)),
            bytes,
        };
        let site = catch_death(|| e.encode(OnFault::Return, fill)).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (
                DeathKind::NullDeref,
                "fillTexture:level:slice:region:bytes:length:"
            )
        );
        // Of no bytes, as Metal reads them whatever the length, it dies all the same.
        // SAFETY: no bytes.
        let none = unsafe { HostRange::new(b.contents().wrapping_add(16), 0) };
        let site = catch_death(|| {
            let fill = BlitCmd::FillTexture {
                texture: &t,
                level: 0,
                slice: 0,
                region: MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(4, 4, 1)),
                bytes: none,
            };
            e.encode(OnFault::Return, fill)
        })
        .unwrap_err();
        assert_eq!(site.kind, DeathKind::NullDeref);
        // Of a region of no width and no height, whose bytes Metal never reads, it fills
        // nothing, and its command buffer completes.
        for depth in [0, 1] {
            let fill = BlitCmd::FillTexture {
                texture: &t,
                level: 0,
                slice: 0,
                region: MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(0, 0, depth)),
                bytes,
            };
            e.encode(OnFault::Return, fill).unwrap();
        }
        e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!(cb.status().0, 4);
    }

    /// The bytes a texture over a buffer may lie in are where Metal lays its texels out: a 2D
    /// array's slices `bytesPerRow` times its height, rounded up to 128, apart, whatever
    /// `bytesPerImage` says; a texture buffer from its offset rounded down to 16, where its
    /// functions read and write; a 2D texture its rows from its offset; up to the buffer's end
    #[test]
    fn a_texture_over_a_buffer_lies_where_metal_lays_it_out() {
        use metal2vulkan::metal::{PixelFormat, TextureType};
        let desc = |texture_type, (width, height, slices)| {
            let mut d =
                rt::TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, width, height, false);
            d.texture_type = texture_type;
            d.array_length = slices;
            d
        };
        // Two slices of 4 rows of 256 bytes, 1024 apart, whatever bytesPerImage the guest gave.
        let array = desc(TextureType::Type2DArray, (64, 4, 2));
        assert_eq!(texel_bytes(&array, (0, 256), 1 << 20), 0..2048);
        // Three slices of 3 rows of 240 bytes, 720 rounded up to 768 apart.
        let odd = desc(TextureType::Type2DArray, (60, 3, 3));
        assert_eq!(
            texel_bytes(&odd, (128, 240), 1 << 20),
            128..128 + 2 * 768 + 720
        );
        let buffer = desc(TextureType::TextureBuffer, (16, 1, 1));
        assert_eq!(texel_bytes(&buffer, (20, 64), 1 << 20), 16..84);
        let plain = desc(TextureType::Type2D, (64, 4, 1));
        assert_eq!(texel_bytes(&plain, (32, 256), 1 << 20), 32..32 + 1024);
        assert_eq!(texel_bytes(&array, (0, 256), 1500), 0..1500);
    }

    /// What a 2D array texture over a no-copy buffer of 64 KiB of host memory shows on `P`, made
    /// by `newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:` where `linear`,
    /// else by `newTextureWithDescriptor:offset:bytesPerRow:`: three slices of 60 by 3
    /// `BGRA8Unorm` texels, rows of 240 bytes from offset 128 (bytesPerImage 4096). In order:
    /// its slices read back by a blit as made; read back again after the CPU wrote other bytes
    /// over all the memory (whether it aliases the memory or copied it); and the memory after a
    /// blit copied a third pattern into every slice (where its slices lie).
    fn array_over_host_memory_on<P: Platform>(
        d: &P::Device,
        desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
        linear: bool,
    ) -> [Vec<u8>; 3]
    where
        P::Device: GpuDevice<P>,
    {
        use crate::core::host::HostPtr;
        use crate::core::platform::HostRange;
        use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
        use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use crate::core::platform::resource::Buffer as _;

        const LENGTH: usize = 0x10000;
        const SLICE: u64 = 60 * 3 * 4;
        let layout = Layout::from_size_align(LENGTH, 0x4000).unwrap();
        // SAFETY: a nonzero size.
        let memory = unsafe { alloc_zeroed(layout) };
        let write = |seed: u8| {
            for i in 0..LENGTH {
                // SAFETY: inside the memory just allocated, which no GPU work uses now.
                unsafe { *memory.add(i) = (i as u8).wrapping_mul(31) ^ (i >> 8) as u8 ^ seed };
            }
        };
        write(0);
        // SAFETY: the memory outlives the buffer, which is dropped before it is freed.
        let bytes = unsafe { HostRange::new(HostPtr::from_ptr(memory), LENGTH as u64) };
        let b = d
            .new_buffer_over_host_memory(OnFault::Return, bytes, 0, None)
            .unwrap()
            .unwrap();
        let array = desc(&TextureDesc {
            texture_type: 3,
            pixel_format: 80,
            width: 60,
            height: 3,
            array_length: 3,
            usage: 0x3,
            ..TextureDesc::default()
        });
        let t = match linear {
            true => b.new_linear_texture(&array, 128, 240, 4096),
            false => b.new_texture(OnFault::Return, &array, 128, 240),
        };
        let t = t.unwrap().unwrap();
        let shared = |length: u64| {
            d.new_buffer_of_length(OnFault::Die, length, 0)
                .unwrap()
                .unwrap()
        };
        let q = d.new_command_queue().unwrap();
        let run = |cmds: Vec<BlitCmd<'_, P>>| {
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
            for cmd in cmds {
                e.encode(OnFault::Die, cmd).unwrap();
            }
            e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
            cb.commit(OnFault::Die).unwrap();
            cb.wait_until_completed(OnFault::Die).unwrap();
            assert!(cb.error().is_none(), "{:?}", cb.error());
        };
        // Each slice's copy up into `to`, or down from it, tightly packed.
        fn slices<'a, P: Platform>(
            t: &'a P::Texture,
            to: &'a P::Buffer,
            up: bool,
        ) -> Vec<BlitCmd<'a, P>> {
            (0..3)
                .map(|slice| match up {
                    true => BlitCmd::CopyTextureToBuffer {
                        source_texture: t,
                        source_slice: slice,
                        source_level: 0,
                        source_origin: MtlOrigin::new(0, 0, 0),
                        source_size: MtlSize::new(60, 3, 1),
                        destination_buffer: to,
                        destination_offset: slice * SLICE,
                        destination_bytes_per_row: 240,
                        destination_bytes_per_image: SLICE,
                    },
                    false => BlitCmd::CopyBufferToTexture {
                        source_buffer: to,
                        source_offset: slice * SLICE,
                        source_bytes_per_row: 240,
                        source_bytes_per_image: SLICE,
                        source_size: MtlSize::new(60, 3, 1),
                        destination_texture: t,
                        destination_slice: slice,
                        destination_level: 0,
                        destination_origin: MtlOrigin::new(0, 0, 0),
                    },
                })
                .collect()
        }
        let n = usize::try_from(3 * SLICE).unwrap();
        let read = || {
            let out = shared(3 * SLICE);
            run(slices::<P>(&t, &out, true));
            // SAFETY: a Shared buffer's bytes, read after the blit completed.
            unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
        };
        let made = read();
        write(0x5a);
        let rewritten = read();
        let third = shared(3 * SLICE);
        // SAFETY: a Shared buffer of `n` bytes, which no command uses yet.
        unsafe { std::ptr::write_bytes(third.contents().as_ptr::<u8>(), 0xc3, n) };
        run(slices::<P>(&t, &third, false));
        // SAFETY: the host memory, which the blit's writes finished.
        let after = unsafe { std::slice::from_raw_parts(memory, LENGTH) }.to_vec();
        drop((t, b));
        // SAFETY: allocated above with `layout`; the buffer over it is gone.
        unsafe { dealloc(memory, layout) };
        [made, rewritten, after]
    }

    /// The runtime's refusal of a 2D array over a buffer by
    /// `newTextureWithDescriptor:offset:bytesPerRow:` dies at the creator as Metal's assertion,
    /// its line carrying the assertion's text ([`buffer_texture_type`]); its refusal of another
    /// type dies there naming the runtime's error. The creator's decision, without a device
    #[test]
    fn a_refused_array_over_a_buffer_dies_as_metals_assertion() {
        let sel = "newTextureWithDescriptor:offset:bytesPerRow:";
        for (texture_type, metals) in [
            (TextureType::Type2DArray, true),
            (TextureType::Type3D, false),
        ] {
            let refusal = rt::Error::Invalid(rt::Invalid::new(
                "new_texture",
                rt::Problem::BufferTexture(rt::BufferTextureProblem::TextureType { texture_type }),
            ));
            *crate::answer::LAST_LINE.lock() = None;
            let site = catch_death(|| {
                create(sel, Policy::Dies, Nil::Never, || {
                    buffer_texture_type::<()>(sel, Err(refusal))
                })
            })
            .unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, sel),
                "{texture_type:?}"
            );
            assert_eq!(
                line.starts_with("Metal aborts at a 2D array over a buffer")
                    && line.contains("Linear texture: must be of type MTLTextureType2D"),
                metals,
                "{line}"
            );
            assert_eq!(
                line.starts_with("the Vulkan runtime refused it"),
                !metals,
                "{line}"
            );
        }
    }

    /// The same at the device: `newTextureWithDescriptor:offset:bytesPerRow:` of a 2D array over
    /// a no-copy buffer of host memory dies at the creator as Metal's assertion, where Metal
    /// aborts ("_mtlValidateStrideTextureParameters:1927: failed assertion `Texture Descriptor
    /// Validation Linear texture: must be of type MTLTextureType2D, textureType
    /// (MTLTextureType2DArray) disallowed'", measured with `array_over_host_memory_on`, one case a
    /// process)
    #[test]
    fn an_array_over_a_buffer_dies_where_metal_aborts() {
        let sel = "newTextureWithDescriptor:offset:bytesPerRow:";
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(|| {
            array_over_host_memory_on::<VulkanPlatform>(
                &dev(),
                &|t| platform().texture_descriptor(t),
                false,
            )
        })
        .unwrap_err();
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
        assert!(
            line.contains("Linear texture: must be of type MTLTextureType2D"),
            "{line}"
        );
    }

    /// didModifyRange: is taken on a Managed and on a Shared buffer, and dies past a Managed
    /// one; setPurgeableState: dies for a raw state none is, of a buffer and of a texture
    #[test]
    fn modified_ranges_and_purgeable_states() {
        let d = dev();
        let managed = d
            .new_buffer_of_length(OnFault::Return, 256, 0x10)
            .unwrap()
            .unwrap();
        managed
            .did_modify_range(OnFault::Return, NsRange::new(0, 256))
            .unwrap();
        let past =
            catch_death(|| managed.did_modify_range(OnFault::Return, NsRange::new(200, 100)));
        assert_eq!(past.unwrap_err().file, "didModifyRange:");
        let shared = shared(256);
        shared
            .did_modify_range(OnFault::Return, NsRange::new(0, 16))
            .unwrap();
        let refused = catch_death(|| shared.set_purgeable_state(9));
        assert_eq!(refused.unwrap_err().file, "setPurgeableState:");
        let t = d
            .new_texture(OnFault::Return, &bgra(8, 8, 0))
            .unwrap()
            .unwrap();
        let refused = catch_death(|| t.set_purgeable_state(9));
        assert_eq!(refused.unwrap_err().file, "setPurgeableState:");
    }

    /// A descriptor of a `width` × `height` BGRA8Unorm 2D texture, read, written and viewed by
    /// other formats, of `options`.
    pub(crate) fn bgra(width: u64, height: u64, options: u64) -> VulkanTextureDescriptor {
        platform().texture_descriptor(&TextureDesc {
            pixel_format: 80,
            width,
            height,
            usage: 0x13,
            resource_options: Some(options),
            ..TextureDesc::default()
        })
    }

    /// The whole of `t`'s level 0, slice 0, read back: rows of `width × 4` bytes.
    pub(crate) fn read_back(t: &VulkanTexture) -> Vec<u8> {
        let (w, h) = (t.width(), t.height());
        let mut bytes = vec![0u8; usize::try_from(w * h * 4).unwrap()];
        let region = rt::Region::new_3d(0, 0, 0, w, h, 1);
        t.raw.get_bytes(&mut bytes, w * 4, 0, region, 0, 0).unwrap();
        bytes
    }

    /// `width × height × 4` bytes that differ from texel to texel.
    pub(crate) fn pattern(width: u64, height: u64) -> Vec<u8> {
        (0..width * height * 4)
            .map(|i| (i * 7 % 251) as u8)
            .collect()
    }

    /// The whole of `t`'s level 0, slice 0, replaced with `bytes`.
    pub(crate) fn fill(t: &VulkanTexture, bytes: &[u8]) {
        let (w, h) = (t.width(), t.height());
        let region = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(w, h, 1));
        t.replace_region(region, 0, 0, HostRange::of_slice(bytes), w * 4, 0)
            .unwrap();
    }

    /// A texture of a descriptor value answers its properties, and its bytes round-trip; the
    /// texture's resourceIndex is its resource ID; a 2D descriptor's chain is full
    #[test]
    fn textures_of_descriptors() {
        let d = dev();
        let desc = bgra(16, 8, 0);
        assert_eq!(
            (
                desc.pixel_format(),
                desc.texture_type(),
                desc.storage_mode()
            ),
            (80, 2, 0)
        );
        let t = d.new_texture(OnFault::Return, &desc).unwrap().unwrap();
        assert_eq!((t.width(), t.height(), t.depth()), (16, 8, 1));
        assert_eq!(
            (t.pixel_format(), t.texture_type(), t.usage()),
            (80, 2, 0x13)
        );
        assert_eq!(
            (t.mipmap_level_count(), t.sample_count(), t.array_length()),
            (1, 1, 1)
        );
        assert_eq!((t.storage_mode(), t.protection_options()), (0, 0));
        assert_ne!(t.gpu_resource_id(), 0);
        assert_eq!(t.resource_index(), t.gpu_resource_id());
        assert!(!t.is_shareable() && t.iosurface().is_none());
        assert_eq!(GpuResource::device(&t).unwrap().id(), d.id());
        t.set_label("reims-vgpu.texture").unwrap();
        let bytes = pattern(16, 8);
        fill(&t, &bytes);
        assert_eq!(read_back(&t), bytes);
        let mipmapped = platform()
            .texture_2d_descriptor(OnFault::Return, 80, 64, 32, true)
            .unwrap();
        mipmapped.set_usage(0x13);
        let m = d.new_texture(OnFault::Return, &mipmapped).unwrap().unwrap();
        assert_eq!(m.mipmap_level_count(), 7);
        let size = d.heap_texture_size_and_align(&mipmapped).unwrap();
        assert!(size.size >= 64 * 32 * 4 && size.align > 0, "{size:?}");
    }

    /// The heap sizes `vaprobe texsizes` measured of plain Metal (M3 Max, macOS 26.7: 2D,
    /// private), which decide whether Metal gives a texture of memory of its own a range of the
    /// space and its span ([`crate::gpu::metal_space::OWN_HEAP_SIZE`]), are the sizes the space
    /// takes of the runtime's answers ([`crate::gpu::metal_space::own_heap_size`], its model of
    /// Metal's) on every device: 4x4 Depth32Float, Depth16Unorm and Stencil8 render targets, a
    /// 4x4 Depth32Float_Stencil8 one read, and RGBA8 of 32x64, 63x63, 64x64 read or a render
    /// target, 64x64 of seven levels, 128x128 and 256x256 read. The guest's size query answers
    /// each on Metal, and at least each on a native device, whose image may take more
    /// (`heap-texture-larger-than-metal`, relayed there: a 4x4 `Depth32Float` texture takes 1024
    /// bytes on NVIDIA)
    #[test]
    fn heap_sizes_the_space_probes_measured_are_the_runtimes() {
        let sel = "newTextureWithDescriptor:";
        let d = dev();
        let native = crate::answer::native::placement() == crate::answer::native::Placement::Native;
        for (pixel_format, width, height, mipmap_level_count, usage, size) in [
            (252, 4, 4, 1, 0x4, 0x80),
            (250, 4, 4, 1, 0x4, 0x80),
            (253, 4, 4, 1, 0x4, 0x80),
            (260, 4, 4, 1, 0x1, 0x4080),
            (70, 32, 64, 1, 0x3, 0x2000),
            (70, 63, 63, 1, 0x3, 0x4000),
            (70, 64, 64, 1, 0x1, 0x4080),
            (70, 64, 64, 1, 0x4, 0x4080),
            (70, 64, 64, 7, 0x3, 0x5680),
            (70, 128, 128, 1, 0x1, 0x1_0200),
            (70, 256, 256, 1, 0x1, 0x4_0800),
        ] {
            let t = TextureDesc {
                pixel_format,
                width,
                height,
                mipmap_level_count,
                usage,
                resource_options: Some(0x20),
                ..TextureDesc::default()
            };
            let desc = platform().texture_descriptor(&t);
            let raw = desc.fields().descriptor(sel, Maker::Other);
            let answer = |desc: &rt::TextureDescriptor| d.size_and_align_at(sel, desc);
            let metal = d.metal_answer_at(sel, &raw);
            let ours = crate::gpu::metal_space::own_heap_size(&raw, metal, answer);
            assert_eq!(ours, Some(size), "{t:?}");
            let queried = d.heap_texture_size_and_align(&desc).unwrap();
            match native {
                true => assert!(queried.size >= size, "the query: {queried:x?} {t:?}"),
                false => assert_eq!(queried.size, size, "the query: {t:?}"),
            }
        }
    }

    /// The shapes of [`lossless_mipmapped_heap_texture_sizes_are_modelled`]: a name, pixel
    /// format, width, height, levels and usage, each `Private`, of allowGPUOptimizedContents
    /// unset (a bare one without `BlockWritesOnly`), each a mipmapped texture Metal keeps
    /// losslessly compressed.
    const LOSSLESS_MIPMAPPED_SHAPES: [(&str, u64, u64, u64, u64, u64); 5] = [
        ("an R32Uint render target", 53, 64, 64, 7, 0x4),
        ("r28's E26 and E26i", 115, 224, 928, 3, 0x1_0005),
        ("r28's E26 and E26i, bare", 115, 224, 928, 3, 0x5),
        ("r28's E26w", 115, 216, 928, 3, 0x1_0005),
        ("r28's E26w, bare", 115, 216, 928, 3, 0x5),
    ];

    /// [`LOSSLESS_MIPMAPPED_SHAPES`], each a name and its texture.
    pub(crate) fn lossless_mipmapped_shapes() -> impl Iterator<Item = (&'static str, TextureDesc)> {
        LOSSLESS_MIPMAPPED_SHAPES.into_iter().map(
            |(name, pixel_format, width, height, mipmap_level_count, usage)| {
                let t = TextureDesc {
                    pixel_format,
                    width,
                    height,
                    mipmap_level_count,
                    usage,
                    resource_options: Some(0x20),
                    ..TextureDesc::default()
                };
                (name, t)
            },
        )
    }

    /// The size query of a mipmapped texture Metal keeps losslessly compressed, whose layout in
    /// Metal's heaps the runtime models, answers Metal's as the runtime models it
    /// (`Device::metal_heap_texture_size_and_align`) on Metal, and at least that on a native
    /// device, whose image may take more (`heap-texture-larger-than-metal`, relayed there: the
    /// `R32Uint` one takes 0x5c00 bytes on NVIDIA, where Metal answers 0x5800), with no
    /// `unmeasured-heap-texture-size` notice: a 64x64 `R32Uint` render target of seven levels,
    /// and the three-level `RGBA16Float` render targets the macOS 26 boots asked the size of,
    /// with `BlockWritesOnly` and without ([`LOSSLESS_MIPMAPPED_SHAPES`]). A texture
    /// of its own memory of each is made, and so is its placement in a heap of the query's size,
    /// over no address ranges, whose layout there the runtime announces, Metal keeping it
    /// compressed (`heap-texture-lossless-layout`). On every OS: what is checked
    /// is the runtime's answer and what is made of it, on whatever device; how it compares with
    /// Metal's is `metal_tests`' (`lossless_mipmapped_heap_texture_sizes_are_metals`)
    #[test]
    fn lossless_mipmapped_heap_texture_sizes_are_modelled() {
        let sel = "heapTextureSizeAndAlignWithDescriptor:";
        let (p, d) = (platform(), dev());
        let native = crate::answer::native::placement() == crate::answer::native::Placement::Native;
        crate::answer::RELAYED.lock().clear();
        for (name, t) in lossless_mipmapped_shapes() {
            let mipmap_level_count = t.mipmap_level_count;
            let desc = p.texture_descriptor(&t);
            let ours = d.heap_texture_size_and_align(&desc).unwrap();
            let raw = desc.fields().descriptor(sel, Maker::SizeQuery);
            let runtimes =
                crate::answer::plain(|| d.raw.heap_texture_size_and_align(&raw)).unwrap();
            assert_eq!(
                (ours.size, ours.align),
                (runtimes.size, runtimes.align),
                "{name}"
            );
            let metal = crate::answer::plain(|| d.raw.metal_heap_texture_size_and_align(&raw));
            let metal = metal.unwrap().unwrap();
            let (size, metal) = ((ours.size, ours.align), (metal.size, metal.align));
            match native {
                true => assert!(
                    size.0 >= metal.0 && size.1 >= metal.1,
                    "{name}: {size:x?} {metal:x?}"
                ),
                false => assert_eq!(size, metal, "{name}"),
            }
            let own = d.new_texture(OnFault::Die, &desc).unwrap().unwrap();
            assert_eq!(own.mipmap_level_count(), mipmap_level_count, "{name}");
            let heap = d.new_heap(&HeapDesc {
                size: ours.size,
                heap_type: 1,
                options: vec![OptionSet::StorageMode(2)],
                address_ranges: None,
                pinned_gpu_address: None,
            });
            let heap = heap.unwrap().heap.unwrap();
            let before = crate::answer::RELAYED.lock().len();
            let placed = heap.new_texture(&desc, 0).unwrap().unwrap();
            assert_eq!(placed.mipmap_level_count(), mipmap_level_count, "{name}");
            let lossless = rt::Degradation::HeapTextureLosslessLayout;
            let relayed = crate::answer::RELAYED.lock()[before..].to_vec();
            assert!(relayed.contains(&lossless), "{name}: {relayed:?}");
        }
        let unmeasured = rt::Degradation::UnmeasuredHeapTextureSize;
        assert!(!crate::answer::RELAYED.lock().contains(&unmeasured));
    }

    /// A row of the wider measurement: `texture_type`, `width` by `height`, of `levels` levels (0
    /// for the full chain), `slices` slices and `samples` samples, of `usage`, with
    /// allowGPUOptimizedContents off.
    fn wide_row(
        texture_type: u64,
        (width, height): (u64, u64),
        levels: u64,
        slices: u64,
        samples: u64,
        usage: u64,
    ) -> TextureDesc {
        TextureDesc {
            texture_type,
            width,
            height,
            mipmap_level_count: match levels {
                0 => full_chain(width, height),
                n => n,
            },
            array_length: slices,
            sample_count: samples,
            usage,
            allow_gpu_optimized_contents: Some(false),
            ..TextureDesc::default()
        }
    }

    /// The levels of a full mip chain of `width` by `height`.
    fn full_chain(width: u64, height: u64) -> u64 {
        u64::from(width.max(height).ilog2()) + 1
    }

    /// The usages of a depth or stencil texture without allowGPUOptimizedContents at which Metal
    /// asserts, of each of `Depth16Unorm`, `Depth32Float`, `Stencil8` and
    /// `Depth32Float_Stencil8`, at `Private` and `Shared` storage: `ShaderRead | ShaderAtomic`,
    /// of a 64x64 2D texture, which the runtime refuses `Invalid` (`UsageForFormat`). Each name,
    /// its usage and its sample count. The usages holding neither `ShaderRead` nor
    /// `RenderTarget` but another bit take Metal's layout ([`DEPTH_MORE_USAGES`]).
    const UNMEASURED_DEPTH_USAGES: [(&str, u64, u64); 1] = [("ShaderRead | ShaderAtomic", 0x21, 1)];

    /// `heapTextureSizeAndAlignWithDescriptor:` of a depth or stencil texture with
    /// allowGPUOptimizedContents off of each of [`UNMEASURED_DEPTH_USAGES`] dies at its selector,
    /// naming the runtime's refusal (`UsageForFormat`), and so does `newTextureWithDescriptor:`
    /// of such a texture at `Private` storage: Metal asserts there. Of a 3D texture, a 1D one and
    /// one of 8 samples, at which Metal asserts, the query dies at its selector too, at the
    /// runtime's refusal
    #[test]
    fn unmeasured_depth_sizes_die_at_the_size_query() {
        let (p, d) = (platform(), dev());
        let dies = |sel: &'static str, name: &str, f: &dyn Fn(), problem: Option<&str>| {
            *crate::answer::LAST_LINE.lock() = None;
            let site = catch_death(f).unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, sel),
                "{name}: {line}"
            );
            assert!(
                line.contains("the Vulkan runtime refused it")
                    && problem.is_none_or(|problem| line.contains(problem)),
                "{name}: {line}"
            );
        };
        let shaped = |pixel_format, usage, samples, options| {
            let (texture_type, extent) = match samples {
                1 => (2, (64, 64)),
                _ => (4, (100, 37)),
            };
            let mut t = wide_row(texture_type, extent, 1, 1, samples, usage);
            t.pixel_format = pixel_format;
            t.resource_options = Some(options);
            t
        };
        for pixel_format in [250, 252, 253, 260] {
            for (storage, options) in [("Private", 0x20), ("Shared", 0)] {
                for (usage_name, usage, samples) in UNMEASURED_DEPTH_USAGES {
                    let t = shaped(pixel_format, usage, samples, options);
                    let desc = p.texture_descriptor(&t);
                    let name = format!("{pixel_format} {storage} {usage_name}");
                    let sel = "heapTextureSizeAndAlignWithDescriptor:";
                    let problem = Some("UsageForFormat");
                    dies(
                        sel,
                        &name,
                        &|| drop(d.heap_texture_size_and_align(&desc)),
                        problem,
                    );
                    if storage == "Private" {
                        let sel = "newTextureWithDescriptor:";
                        let made = || drop(d.new_texture(OnFault::Return, &desc));
                        dies(sel, &name, &made, problem);
                    }
                }
            }
            let sel = "heapTextureSizeAndAlignWithDescriptor:";
            let mut volume = shaped(pixel_format, 0x5, 1, 0x20);
            (volume.texture_type, volume.depth) = (7, 4);
            let mut line = shaped(pixel_format, 0x5, 1, 0x20);
            (line.texture_type, line.height) = (0, 1);
            let eight = shaped(pixel_format, 0x5, 8, 0x20);
            for (name, t) in [("3D", volume), ("1D", line), ("8 samples", eight)] {
                let desc = p.texture_descriptor(&t);
                let name = format!("{pixel_format} {name}");
                dies(
                    sel,
                    &name,
                    &|| drop(d.heap_texture_size_and_align(&desc)),
                    None,
                );
            }
        }
    }

    /// Views read the texture's texels as another format, a subrange of levels, a swizzle, or
    /// at a resource index
    #[test]
    fn texture_views() {
        let d = dev();
        let t = d
            .new_texture(OnFault::Return, &bgra(4, 4, 0))
            .unwrap()
            .unwrap();
        let bytes = pattern(4, 4);
        fill(&t, &bytes);
        let rgba = t.new_texture_view(70).unwrap().unwrap();
        assert_eq!(rgba.pixel_format(), 70);
        assert_eq!(read_back(&rgba), bytes, "the same bytes, read as RGBA");
        let placed = t.new_texture_view_at_index(70, 900).unwrap().unwrap();
        assert_eq!(placed.gpu_resource_id(), 900);
        let sub = t
            .new_texture_subview(80, 2, NsRange::new(0, 1), NsRange::new(0, 1))
            .unwrap()
            .unwrap();
        assert_eq!(read_back(&sub), bytes);
        let swizzled = t
            .new_swizzled_texture_subview_at_index(
                80,
                2,
                NsRange::new(0, 1),
                NsRange::new(0, 1),
                Swizzle {
                    red: 2,
                    green: 3,
                    blue: 4,
                    alpha: 5,
                },
                901,
            )
            .unwrap()
            .unwrap();
        assert_eq!(swizzled.gpu_resource_id(), 901);
        let plain_placed = t
            .new_texture_subview_at_index(80, 2, NsRange::new(0, 1), NsRange::new(0, 1), 902)
            .unwrap()
            .unwrap();
        assert_eq!(plain_placed.gpu_resource_id(), 902);
    }

    /// An RGBA8Unorm texture of `usage`, 4 by 4.
    fn d_texture(usage: u64) -> VulkanTexture {
        let desc = platform().texture_descriptor(&TextureDesc {
            pixel_format: 70,
            width: 4,
            height: 4,
            usage,
            ..TextureDesc::default()
        });
        dev().new_texture(OnFault::Return, &desc).unwrap().unwrap()
    }

    /// Metal raises at no view: one of `MTLPixelFormatInvalid` or `Unspecialized` is nil, and
    /// one Metal asserts at (another size, depth, a compressed format, another type family, levels
    /// or slices outside the texture, a slice count the type does not take) dies at its selector
    /// (measured)
    #[test]
    fn views_metal_stops_die_or_are_nil() {
        let t = d_texture(0x11);
        for format in [0, 263] {
            assert!(t.new_texture_view(format).unwrap().is_none(), "{format}");
        }
        let sel = "newTextureViewWithPixelFormat:";
        for format in [25, 252, 130] {
            let site = catch_death(|| t.new_texture_view(format).map(|v| v.is_some()));
            assert_eq!(site.map(|_| ()).unwrap_err().file, sel, "format {format}");
        }
        let sel = "newTextureViewWithPixelFormat:textureType:levels:slices:";
        let one = NsRange::new(0, 1);
        let refused = [
            (7, NsRange::new(0, 1), one),
            (2, NsRange::new(0, 2), one),
            (2, NsRange::new(1, 1), one),
            (2, NsRange::new(0, 0), one),
            (2, one, NsRange::new(0, 2)),
        ];
        for (ty, levels, slices) in refused {
            let site = catch_death(|| {
                t.new_texture_subview(70, ty, levels, slices)
                    .map(|v| v.is_some())
            });
            let site = site.map(|_| ()).unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
        }
    }

    /// A view of a texture of `R10Unorm_X6` (576) or `RG10Unorm_X12` (578) over a buffer in
    /// another format of its texel's size, which the runtime refuses, dies at its selector named
    /// by the texture's format, the one the runtime views in no other format, not by the view's
    #[test]
    fn views_of_padded_textures_die_named_by_the_texture() {
        let sel = "newTextureViewWithPixelFormat:";
        for (format, view) in [(576, 20), (578, 60)] {
            let b = shared(256 * 16);
            let desc = platform().texture_descriptor(&TextureDesc {
                pixel_format: format,
                width: 16,
                height: 16,
                usage: 0x11,
                resource_options: Some(0),
                ..TextureDesc::default()
            });
            let t = b
                .new_texture(OnFault::Return, &desc, 0, 256)
                .unwrap()
                .unwrap();
            *crate::answer::LAST_LINE.lock() = None;
            let site = catch_death(|| t.new_texture_view(view).map(|v| v.is_some()));
            let site = site.map(|_| ()).unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
            let named =
                format!("a view in pixel format {view} of a texture of pixel format {format} (");
            assert!(line.contains(&named), "{line}");
        }
    }

    /// A texture over a buffer reads the buffer's bytes, rows the given pitch apart
    #[test]
    fn textures_over_buffers() {
        let b = shared(64 * 16 * 4);
        let bytes = pattern(64, 16);
        // SAFETY: the buffer's Shared bytes, which nothing else touches yet.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.contents().as_ptr(), bytes.len())
        };
        let t = b
            .new_texture(OnFault::Return, &bgra(64, 16, 0), 0, 256)
            .unwrap()
            .unwrap();
        assert_eq!(read_back(&t), bytes);
        let linear = b
            .new_linear_texture(&bgra(64, 16, 0), 0, 256, 4096)
            .unwrap()
            .unwrap();
        assert_eq!(read_back(&linear), bytes);
        let refused = catch_death(|| b.new_texture(OnFault::Return, &bgra(64, 16, 0), 0, 200));
        assert_eq!(
            refused.unwrap_err().file,
            "newTextureWithDescriptor:offset:bytesPerRow:",
            "rows off the alignment"
        );
    }

    /// A BC1 or BC7 texture over a surface's rows, through the buffer that stands for the
    /// surface, is made, as the x86 Macs make it (docs/graphics-kext/dgpu-oracle.md, Q20), and
    /// reads back row j of its blocks from the plane's row j, as theirs read: over a surface
    /// without planes, and over plane 1 of a surface with planes, at an offset and rows that are
    /// multiples of 16 bytes. metal2vulkan makes these since its stack 34; the guest's bundle
    /// sends them
    #[test]
    fn bc_textures_over_a_surface_s_rows_are_made() {
        use crate::core::platform::values::SurfaceFacts;
        let p = platform();
        let (rows, pitch) = (64u64, 256u64);
        let b = shared(2 * rows * pitch);
        let bytes: Vec<u8> = (0..2 * rows * pitch).map(|i| (i * 7 % 251) as u8).collect();
        // SAFETY: the buffer's Shared bytes, which nothing else touches yet.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.contents().as_ptr(), bytes.len())
        };
        for (format, block) in [(130, 8), (152, 16)] {
            let desc = p.texture_descriptor(&TextureDesc {
                pixel_format: format,
                width: 64,
                height: 32,
                usage: 1,
                ..TextureDesc::default()
            });
            for planar in [false, true] {
                let offset = if planar { rows * pitch } else { 0 };
                let facts = SurfaceFacts {
                    pixel_format: u32::from_be_bytes(if planar { *b"420v" } else { *b"BGRA" }),
                    plane_count: if planar { 2 } else { 0 },
                    width: 64,
                    height: rows,
                    chroma_offset: offset,
                    chroma_bytes_per_row: if planar { pitch } else { 0 },
                    ycbcr_matrix: None,
                };
                let plane = SurfacePlane {
                    plane: u64::from(planar),
                    offset,
                    bytes_per_row: pitch,
                    planar,
                    facts: Some(facts),
                };
                let what = format!("format {format}, planar {planar}");
                let t = b
                    .new_texture_over_surface_plane(OnFault::Return, &desc, plane)
                    .unwrap()
                    .unwrap_or_else(|| panic!("{what}: nil"));
                assert_eq!((t.pixel_format(), t.width(), t.height()), (format, 64, 32));
                // 16 blocks a row, 8 rows of them.
                let row = 16 * block;
                let mut read = vec![0u8; (row * 8).to_usize()];
                let region = rt::Region::new_3d(0, 0, 0, 64, 32, 1);
                t.raw.get_bytes(&mut read, row, 0, region, 0, 0).unwrap();
                let expected: Vec<u8> = (0..8)
                    .flat_map(|j| {
                        let at = (offset + j * pitch).to_usize();
                        bytes[at..at + row.to_usize()].to_vec()
                    })
                    .collect();
                assert_eq!(read, expected, "{what}");
            }
        }
    }

    /// A texture on a surface writes the surface's memory, and holds one reference of the
    /// surface while it lives; its views share that reference. One over plane 1 of the surface,
    /// which has no planes, dies where it is made
    #[test]
    fn textures_on_surfaces() {
        let d = dev();
        let s = bgra_surface(64, 32);
        let t = d
            .new_texture_on_surface(&bgra(64, 32, 0), Some(&s), 0)
            .unwrap()
            .unwrap();
        assert_eq!(s.retain_count(), 2);
        let view = t.new_texture_view(70).unwrap().unwrap();
        assert_eq!(s.retain_count(), 2);
        assert_eq!(view.iosurface().unwrap().id(), s.id());
        let bytes = pattern(64, 32);
        fill(&t, &bytes);
        // SAFETY: the surface's memory, rows of 256 bytes, which the texture's write finished.
        let memory =
            unsafe { std::slice::from_raw_parts(s.base_address().as_ptr::<u8>(), bytes.len()) };
        assert_eq!(memory, &bytes[..]);
        assert_eq!(read_back(&t), bytes);
        drop((t, view));
        assert_eq!(s.retain_count(), 1);
        // A surface without planes has plane 0 only: a texture over its plane 1 dies where it
        // is made, as Metal asserts there (`textures_over_planes_a_surface_lacks_are_metals`).
        let site =
            catch_death(|| d.new_texture_on_surface(&bgra(64, 32, 0), Some(&s), 1)).unwrap_err();
        let sel = "newTextureWithDescriptor:iosurface:plane:";
        assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
    }

    /// A texture over a surface released while a command buffer that copies it waits keeps the
    /// surface's range of Metal's space until that command buffer completed, as Metal frees a
    /// released resource's range (`metal_space::Hold`): a buffer of the range's size made
    /// meanwhile lies elsewhere, and one made after the completion at the range
    #[test]
    fn a_surface_texture_keeps_its_range_until_its_command_buffer_completed() {
        use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
        use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use crate::core::platform::event::{EventRef, SharedEvent as _};
        use crate::core::platform::resource::Buffer as _;
        use crate::core::platform::types::MtlSize;
        use crate::core::rt::Pool;
        let (p, d) = (platform(), dev());
        // 64 by 64, 4 bytes a texel: a page, whose range is the space's 0x8000.
        let side = 64;
        let s = bgra_surface(side, side);
        let desc = p.texture_descriptor(&TextureDesc {
            pixel_format: 80,
            width: side,
            height: side,
            usage: 1,
            resource_options: Some(0),
            ..TextureDesc::default()
        });
        let q = d.new_command_queue().unwrap();
        let event = d.new_shared_event().unwrap();
        let out = d
            .new_buffer_of_length(OnFault::Die, side * side * 4, 0)
            .unwrap()
            .unwrap();
        let page = |d: &VulkanDevice| {
            d.new_buffer_of_length(OnFault::Die, 0x4000, 0)
                .unwrap()
                .unwrap()
        };
        let (cb, at) = {
            let _pool = Pool::here();
            let t = d
                .new_texture_on_surface(&desc, Some(&s), 0)
                .unwrap()
                .unwrap();
            let at = t.space.address().unwrap();
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
            let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
            e.encode(
                OnFault::Die,
                BlitCmd::CopyTextureToBuffer {
                    source_texture: &t,
                    source_slice: 0,
                    source_level: 0,
                    source_origin: MtlOrigin::new(0, 0, 0),
                    source_size: MtlSize::new(side, side, 1),
                    destination_buffer: &out,
                    destination_offset: 0,
                    destination_bytes_per_row: side * 4,
                    destination_bytes_per_image: side * side * 4,
                },
            )
            .unwrap();
            e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
            cb.commit(OnFault::Die).unwrap();
            (cb, at)
        };
        let meanwhile = page(&d);
        assert_ne!(meanwhile.gpu_address(), at, "made while the copy waits");
        {
            let _pool = Pool::here();
            event.set_signaled_value(1).unwrap();
            cb.wait_until_completed(OnFault::Die).unwrap();
            assert!(cb.error().is_none(), "{:?}", cb.error());
        }
        assert_eq!(page(&d).gpu_address(), at, "made once the copy completed");
    }

    /// A texture of its own memory and a shareable texture, released while a command buffer
    /// that copies it waits, keep their range of Metal's space until that command buffer
    /// completed, as Metal frees a released resource's range (`metal_space::Hold`, the
    /// runtime's range token of the texture): a buffer of a page made meanwhile lies elsewhere,
    /// and one made after the completion at the range
    #[test]
    fn a_texture_keeps_its_range_until_its_command_buffer_completed() {
        use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
        use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use crate::core::platform::event::{EventRef, SharedEvent as _};
        use crate::core::platform::resource::Buffer as _;
        use crate::core::platform::types::MtlSize;
        use crate::core::rt::Pool;
        let (p, d) = (platform(), dev());
        // BGRA8 at Private storage: 64 by 64 of its own memory, a heap size of a page, and a
        // shareable 8 by 8 of a page with its extra bytes; each a range of the space's 0x8000.
        let desc = |side| {
            p.texture_descriptor(&TextureDesc {
                pixel_format: 80,
                width: side,
                height: side,
                usage: 1,
                resource_options: Some(0x20),
                ..TextureDesc::default()
            })
        };
        let (own, shareable) = (desc(64), desc(8));
        let q = d.new_command_queue().unwrap();
        let out = d
            .new_buffer_of_length(OnFault::Die, 64 * 64 * 4, 0)
            .unwrap()
            .unwrap();
        let page = || {
            d.new_buffer_of_length(OnFault::Die, 0x4000, 0)
                .unwrap()
                .unwrap()
        };
        let made: [(&str, u64, &dyn Fn() -> VulkanTexture); 2] = [
            ("own memory", 64, &|| {
                (d.new_texture(OnFault::Die, &own)).unwrap().unwrap()
            }),
            ("shareable", 8, &|| {
                d.new_shared_texture(&shareable).unwrap().unwrap()
            }),
        ];
        for (what, side, make) in made {
            let event = d.new_shared_event().unwrap();
            let (cb, at) = {
                let _pool = Pool::here();
                let t = make();
                let at = t.space.address().unwrap();
                let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
                cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
                let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
                e.encode(
                    OnFault::Die,
                    BlitCmd::CopyTextureToBuffer {
                        source_texture: &t,
                        source_slice: 0,
                        source_level: 0,
                        source_origin: MtlOrigin::new(0, 0, 0),
                        source_size: MtlSize::new(side, side, 1),
                        destination_buffer: &out,
                        destination_offset: 0,
                        destination_bytes_per_row: side * 4,
                        destination_bytes_per_image: side * side * 4,
                    },
                )
                .unwrap();
                e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
                cb.commit(OnFault::Die).unwrap();
                (cb, at)
            };
            let meanwhile = page();
            assert_ne!(
                meanwhile.gpu_address(),
                at,
                "{what}: made while the copy waits"
            );
            {
                let _pool = Pool::here();
                event.set_signaled_value(1).unwrap();
                cb.wait_until_completed(OnFault::Die).unwrap();
                assert!(cb.error().is_none(), "{what}: {:?}", cb.error());
            }
            assert_eq!(
                page().gpu_address(),
                at,
                "{what}: made once the copy completed"
            );
        }
    }

    /// A sparse heap's ranges are its range token's: a heap pinned at its range while it lives is
    /// nil, as Metal answers for a pin it cannot honour, and made there once it is released
    #[test]
    fn a_sparse_heaps_range_comes_back_once_it_is_released() {
        let d = dev();
        let heap = |heap_type, storage, pinned_gpu_address| {
            d.new_heap(&HeapDesc {
                size: 0x40000,
                heap_type,
                options: vec![OptionSet::StorageMode(storage)],
                address_ranges: None,
                pinned_gpu_address,
            })
            .unwrap()
            .heap
        };
        let far = crate::gpu::metal_space::END / 2;
        let sparse = heap(2, 2, Some(far)).unwrap();
        assert!(
            heap(0, 0, Some(far)).is_none(),
            "pinned at the live sparse heap's range"
        );
        drop(sparse);
        let plain = heap(0, 0, Some(far)).unwrap();
        assert_eq!(plain.gpu_address(), far, "pinned there once it is released");
    }

    /// Every texture takes a descriptor in Metal's mempool while it lives
    /// (`metal_space::Elements`): one of memory of its own, a view, one over a buffer and one in a
    /// heap, over a surface, a shareable one and one opened from its handle each count one, which
    /// goes with it
    #[test]
    fn every_texture_takes_a_descriptor_while_it_lives() {
        use crate::core::platform::resource::{Heap as _, Texture as _};
        let (p, d) = (platform(), dev());
        let live = || crate::gpu::metal_space::process().textures_live();
        let of = |format, options| {
            p.texture_descriptor(&TextureDesc {
                pixel_format: format,
                width: 64,
                height: 64,
                usage: 1,
                resource_options: Some(options),
                ..TextureDesc::default()
            })
        };
        let (private, shared) = (of(80, 0x20), of(80, 0));
        let start = live();
        let own = d.new_texture(OnFault::Die, &private).unwrap().unwrap();
        let view = own.new_texture_view(81).unwrap().unwrap();
        let buffer = d.new_buffer_of_length(OnFault::Die, 0x1_0000, 0);
        let buffer = buffer.unwrap().unwrap();
        let over = buffer.new_texture(OnFault::Die, &shared, 0, 256);
        let heap = d.new_heap(&HeapDesc {
            size: 1 << 20,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(2)],
            address_ranges: None,
            pinned_gpu_address: None,
        });
        let heap = heap.unwrap().heap.unwrap();
        let placed = heap.new_texture(&private, 0).unwrap().unwrap();
        let s = bgra_surface(64, 64);
        let on_surface = d.new_texture_on_surface(&shared, Some(&s), 0);
        let shareable = d.new_shared_texture(&private).unwrap().unwrap();
        let handle = shareable.new_shared_texture_handle().unwrap().unwrap();
        let opened = d.open_shared_texture(Some(&handle), 0).unwrap().unwrap();
        assert_eq!(live(), start + 7);
        drop((own, view, over.unwrap().unwrap(), placed));
        drop((on_surface.unwrap().unwrap(), shareable, handle, opened));
        assert_eq!(live(), start);
    }

    /// A texture on a plane of a planar surface is over that plane's bytes, at its rows; a plane
    /// past the surface's planes is nil, as on Metal
    #[test]
    fn textures_on_surface_planes() {
        let plane = |base: u64| SurfacePlaneSpec {
            base,
            offset: base,
            width: 16,
            height: 4,
            size: 16 * 4 * 8,
            bytes_per_row: 16 * 8,
            bytes_per_element: 4,
            element_width: 1,
            element_height: 1,
            ..SurfacePlaneSpec::default()
        };
        let s = platform()
            .surfaces()
            .create(&SurfaceSpec {
                planes: Some(vec![plane(0), plane(16384)]),
                pixel_format: u32::from_be_bytes(*b"BGRA"),
                alloc_size: 32768,
                width: 16,
                height: 4,
                bytes_per_element: 4,
                bytes_per_row: 16 * 8,
                element_width: 1,
                element_height: 1,
                ..SurfaceSpec::default()
            })
            .unwrap();
        assert_eq!(s.plane_count(), 2);
        let d = dev();
        let t = d
            .new_texture_on_surface(&bgra(16, 4, 0), Some(&s), 1)
            .unwrap()
            .unwrap();
        let bytes = pattern(16, 4);
        fill(&t, &bytes);
        // SAFETY: the surface's memory, which the texture's write finished: plane 1 at 16384,
        // rows of 128 bytes of which the texture's 64 are the first.
        let memory = unsafe {
            std::slice::from_raw_parts(s.base_address().as_ptr::<u8>().add(16384), 4 * 128)
        };
        for row in 0..4 {
            assert_eq!(
                memory[row * 128..row * 128 + 64],
                bytes[row * 64..row * 64 + 64]
            );
        }
        assert_eq!(read_back(&t), bytes);
        let none = d.new_texture_on_surface(&bgra(16, 4, 0), Some(&s), 2);
        assert!(none.unwrap().is_none());
    }

    /// A shared texture's handle opens a texture over the same texels; a Shared one is nil
    #[test]
    fn shared_textures() {
        let d = dev();
        let t = d.new_shared_texture(&bgra(8, 8, 0x20)).unwrap().unwrap();
        assert!(t.is_shareable());
        let handle = t.new_shared_texture_handle().unwrap().unwrap();
        let opened = d.open_shared_texture(Some(&handle), 0).unwrap().unwrap();
        assert_eq!((opened.width(), opened.pixel_format()), (8, 80));
        assert!(d.new_shared_texture(&bgra(8, 8, 0)).unwrap().is_none());
        let plain = d
            .new_texture(OnFault::Return, &bgra(8, 8, 0))
            .unwrap()
            .unwrap();
        assert!(plain.new_shared_texture_handle().unwrap().is_none());
    }

    /// A texture opened from a shared texture's handle takes no range of Metal's space and keeps
    /// the shared texture's, as the handle does: with the shared texture and its handle released
    /// and the opened texture live, a buffer of the range's size lies elsewhere; once the opened
    /// texture is released too, at the range, as on Metal (`vaprobe sharedrelease`,
    /// `metal_space::Hold`)
    #[test]
    fn an_opened_shared_texture_keeps_its_source_range() {
        use crate::core::platform::resource::Buffer as _;
        let d = dev();
        let page = || {
            d.new_buffer_of_length(OnFault::Die, 0x4000, 0)
                .unwrap()
                .unwrap()
        };
        let t = d.new_shared_texture(&bgra(8, 8, 0x20)).unwrap().unwrap();
        let at = t.space.address().unwrap();
        let handle = t.new_shared_texture_handle().unwrap().unwrap();
        let opened = d.open_shared_texture(Some(&handle), 0).unwrap().unwrap();
        assert_eq!(opened.space.address(), Some(at));
        drop((t, handle));
        let meanwhile = page();
        assert_ne!(
            meanwhile.gpu_address(),
            at,
            "made while the opened texture lives"
        );
        drop(opened);
        assert_eq!(
            page().gpu_address(),
            at,
            "made once the opened texture is released"
        );
    }

    /// What the runtime has no form for dies at its selector: a texture of a pixel format the
    /// runtime refuses
    #[test]
    fn texture_gaps_die_at_their_selectors() {
        let desc = bgra(8, 8, 0);
        // RGB10A8_2P, a format the runtime refuses: taken and answered, and dying where a
        // texture of it is made.
        desc.set_pixel_format(624);
        assert_eq!(desc.pixel_format(), 624);
        let site = catch_death(|| dev().new_texture(OnFault::Return, &desc)).unwrap_err();
        assert_eq!(site.file, "newTextureWithDescriptor:");
    }

    /// By default the same texture is nil, relayed as a gap at its creator and counted, where
    /// Metal makes it: the guest's creation fails, and the process goes on
    #[test]
    fn a_texture_gap_is_nil_by_default() {
        use crate::answer::{Inexact, inexact_counts, relaxed};
        let counted = || {
            (inexact_counts().into_iter())
                .find(|&(k, _)| k == Inexact::RefusedCreate)
                .map_or(0, |(_, n)| n)
        };
        let desc = bgra(8, 8, 0);
        desc.set_pixel_format(624);
        let before = counted();
        let made = relaxed(|| dev().new_texture(OnFault::Return, &desc));
        assert!(
            matches!(made, Ok(None)),
            "{:?}",
            made.as_ref().map(Option::is_some)
        );
        assert_eq!(counted(), before + 1);
    }

    /// A 64 by 16 `BGRA8Unorm` render target's descriptor, Shared, of `rotation`, `framebufferOnly`
    /// as `framebuffer_only` says, as the suite's rotated mapper textures set it.
    pub(crate) fn rotated(rotation: u64, framebuffer_only: bool) -> VulkanTextureDescriptor {
        platform().texture_descriptor(&TextureDesc {
            pixel_format: 80,
            width: 64,
            height: 16,
            usage: 4,
            resource_options: Some(0),
            framebuffer_only: Some(framebuffer_only),
            rotation: Some(rotation),
            ..TextureDesc::default()
        })
    }

    /// A descriptor keeps every rotation set and answers it, as Metal's does
    #[test]
    fn a_rotation_is_kept_and_answered() {
        let desc = bgra(8, 8, 0);
        assert_eq!(desc.rotation(), 0);
        for rotation in [1, 2, 4, 5, 0x1_0000, u64::MAX, 0] {
            desc.set_rotation(rotation);
            assert_eq!(desc.rotation(), rotation);
        }
        assert_eq!(rotated(3, false).rotation(), 3);
    }

    /// A texture of a rotation other than 0 dies where Metal aborts: of a descriptor that is not
    /// framebufferOnly, of a rotation past 4, and made other than over an IOSurface. Over one, a
    /// framebufferOnly descriptor's rotations 1 to 4 are made at its size, holding the rotation,
    /// over a surface that holds the frame it stores (a quarter turn's its height wide and its
    /// width tall); a size query answers them as it answers 0. A quarter turn whose stored frame
    /// runs past the surface dies where it is made: Metal makes it, and the runtime refuses it.
    /// A view of the whole of a rotated texture in `BGRA8Unorm_sRGB` is made, of its frame and
    /// rotation, as Metal makes it (its API validation stops it); a swizzled one dies at its
    /// creation, where Metal makes it and the runtime refuses it
    #[test]
    fn rotated_textures_are_made_only_where_metal_makes_them() {
        let d = dev();
        let s = bgra_surface(64, 16);
        let on_surface = "newTextureWithDescriptor:iosurface:plane:";
        for (rotation, framebuffer_only) in [(2, false), (1, false), (5, true), (u64::MAX, true)] {
            let desc = rotated(rotation, framebuffer_only);
            let site = catch_death(|| d.new_texture_on_surface(&desc, Some(&s), 0)).unwrap_err();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, on_surface),
                "rotation {rotation}"
            );
        }
        let desc = rotated(2, true);
        let b = shared(64 * 16 * 4);
        let heap = private_heap();
        let private = rotated(2, true);
        private.set_resource_options(0x20);
        let deaths = [
            (
                "newTextureWithDescriptor:",
                catch_death(|| d.new_texture(OnFault::Return, &desc)).map(drop),
            ),
            (
                "newSharedTextureWithDescriptor:",
                catch_death(|| d.new_shared_texture(&private)).map(drop),
            ),
            (
                "newTextureWithDescriptor:offset:bytesPerRow:",
                catch_death(|| b.new_texture(OnFault::Return, &desc, 0, 256)).map(drop),
            ),
            (
                "newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:",
                catch_death(|| b.new_linear_texture(&desc, 0, 256, 4096)).map(drop),
            ),
            (
                "newTextureWithDescriptor:offset:",
                catch_death(|| heap.new_texture(&private, 0)).map(drop),
            ),
        ];
        for (sel, death) in deaths {
            assert_eq!(death.unwrap_err().file, sel);
        }
        let query = |rotation| {
            let answer = d
                .heap_texture_size_and_align(&rotated(rotation, true))
                .unwrap();
            (answer.size, answer.align)
        };
        for rotation in 1..=4 {
            assert_eq!(query(rotation), query(0), "rotation {rotation}");
        }
        let site = catch_death(|| d.heap_texture_size_and_align(&rotated(1, false))).unwrap_err();
        assert_eq!(site.file, "heapTextureSizeAndAlignWithDescriptor:");
        let tall = bgra_surface(64, 64);
        for rotation in 1..=4 {
            let desc = rotated(rotation, true);
            let t = d
                .new_texture_on_surface(&desc, Some(&tall), 0)
                .unwrap()
                .unwrap();
            let turned = plain(|| t.raw.rotation());
            assert_eq!((t.width(), t.height(), turned), (64, 16, rotation));
            let view = t.new_texture_view(81).unwrap().unwrap();
            let view_rotation = plain(|| view.raw.rotation());
            assert_eq!(
                (
                    view.width(),
                    view.height(),
                    view_rotation,
                    view.pixel_format()
                ),
                (64, 16, rotation, 81)
            );
            let all = NsRange::new(0, 1);
            let swizzle = Swizzle {
                red: 4,
                green: 3,
                blue: 2,
                alpha: 5,
            };
            let swizzled = catch_death(|| {
                t.new_swizzled_texture_subview(81, 2, all, all, swizzle)
                    .map(drop)
            });
            let sel = "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:";
            assert_eq!(swizzled.unwrap_err().file, sel);
            let made = catch_death(|| d.new_texture_on_surface(&desc, Some(&s), 0).map(drop));
            match rotation {
                2 => assert!(made.is_ok()),
                _ => assert_eq!(made.unwrap_err().file, on_surface, "rotation {rotation}"),
            }
        }
    }

    /// A framebufferOnly descriptor dies where Metal aborts at it: making a device's, a buffer's or
    /// a heap's texture. Over an IOSurface and as a shared texture it is made, and a size query
    /// answers as without it
    #[test]
    fn framebuffer_only_textures_are_made_only_where_metal_makes_them() {
        let d = dev();
        let desc = rotated(0, true);
        let b = shared(64 * 16 * 4);
        let heap = private_heap();
        let private = rotated(0, true);
        private.set_resource_options(0x20);
        let deaths = [
            (
                "newTextureWithDescriptor:",
                catch_death(|| d.new_texture(OnFault::Return, &private)).map(drop),
            ),
            (
                "newTextureWithDescriptor:offset:bytesPerRow:",
                catch_death(|| b.new_texture(OnFault::Return, &desc, 0, 256)).map(drop),
            ),
            (
                "newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:",
                catch_death(|| b.new_linear_texture(&desc, 0, 256, 4096)).map(drop),
            ),
            (
                "newTextureWithDescriptor:offset:",
                catch_death(|| heap.new_texture(&private, 0)).map(drop),
            ),
        ];
        for (sel, death) in deaths {
            assert_eq!(death.unwrap_err().file, sel);
        }
        let s = bgra_surface(64, 16);
        let on_surface = d.new_texture_on_surface(&desc, Some(&s), 0).unwrap();
        assert_eq!(on_surface.map(|t| plain(|| t.raw.rotation())), Some(0));
        assert!(d.new_shared_texture(&private).unwrap().is_some());
        let size = |desc: &VulkanTextureDescriptor| {
            let answer = d.heap_texture_size_and_align(desc).unwrap();
            (answer.size, answer.align)
        };
        let plain = rotated(0, false);
        plain.set_resource_options(0x20);
        assert_eq!(size(&private), size(&plain));
    }

    /// A placement heap of `size` bytes and the storage properties `options`, as core's value
    /// sends them.
    fn heap_of<P: Platform>(device: &P::Device, size: u64, options: &[OptionSet]) -> HeapMade<P>
    where
        P::Device: GpuDevice<P>,
    {
        device
            .new_heap(&HeapDesc {
                size,
                heap_type: 1,
                options: options.to_vec(),
                address_ranges: None,
                pinned_gpu_address: None,
            })
            .unwrap()
    }

    /// A Private, untracked placement heap of 32 KiB, as core's legacy heap.
    fn private_heap() -> VulkanHeap {
        let options = [OptionSet::StorageMode(2), OptionSet::HazardTrackingMode(0)];
        heap_of::<VulkanPlatform>(&dev(), 0x8000, &options)
            .heap
            .unwrap()
    }

    /// A placement heap's buffers and textures lie at their offsets with the heap's options, a
    /// tracked one's too; nil off the alignment and past the end; a storage mode other than the
    /// heap's dies (Metal's answers, measured)
    #[test]
    fn heap_buffers_and_textures() {
        let h = private_heap();
        let b = h.new_buffer_of_length(16, 0x20, 0x100).unwrap().unwrap();
        assert_eq!(
            (
                b.gpu_address() - h.gpu_address(),
                b.length(),
                b.resource_options()
            ),
            (0x100, 16, 0x120)
        );
        assert_eq!(GpuResource::device(&b).unwrap().id(), dev().id());
        let tracked = h.new_buffer_of_length(16, 0x220, 0).unwrap().unwrap();
        assert_eq!(tracked.resource_options(), 0x120);
        for (length, offset) in [(16, 0x10), (0x100, 0x8000), (0x9000, 0)] {
            let b = h.new_buffer_of_length(length, 0x20, offset).unwrap();
            assert!(b.is_none(), "{length:#x} at {offset:#x}");
        }
        let sel = "newBufferWithLength:options:offset:";
        let site = catch_death(|| h.new_buffer_of_length(16, 0, 0)).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
        let t = h.new_texture(&bgra(16, 16, 0x220), 0).unwrap().unwrap();
        assert_eq!((t.resource_options(), t.storage_mode()), (0x120, 2));
        assert_eq!((t.width(), t.height()), (16, 16));
        for offset in [1, 0x7f00, 0x8000] {
            let t = h.new_texture(&bgra(16, 16, 0x120), offset).unwrap();
            assert!(t.is_none(), "at {offset:#x}");
        }
        let sel = "newTextureWithDescriptor:offset:";
        let site = catch_death(|| h.new_texture(&bgra(16, 16, 0), 0)).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
    }

    /// A texture of a plain format of `ShaderWrite` usage placed in a heap over the guest's pages
    /// is made, and so are one of `BC1_RGBA` and one of a PVRTC format (`PVRTC_RGBA_4BPP`), of
    /// `ShaderRead` usage, whose blocks lie in Metal's tiles at every level
    #[test]
    fn heap_textures_over_guest_pages_in_metals_block_layout_are_made() {
        use crate::core::platform::types::ResourceAddressRange;

        const SIZE: u64 = 0x10000;
        let (p, d) = (platform(), dev());
        // Left mapped: the heap's alias shares them until the process ends.
        let pages = crate::os::allocate_aliasable(SIZE).unwrap();
        let ranges = p
            .new_address_range_array(&[ResourceAddressRange {
                address: u64::try_from(pages.addr()).unwrap(),
                length: SIZE,
            }])
            .unwrap();
        let heap = d
            .new_heap(&HeapDesc {
                size: SIZE,
                heap_type: 1,
                options: vec![OptionSet::StorageMode(0), OptionSet::HazardTrackingMode(0)],
                address_ranges: Some(&ranges),
                pinned_gpu_address: None,
            })
            .unwrap()
            .heap
            .unwrap();
        let desc = |pixel_format, usage| {
            p.texture_descriptor(&TextureDesc {
                pixel_format,
                width: 16,
                height: 16,
                usage,
                resource_options: Some(0),
                ..TextureDesc::default()
            })
        };
        assert!(heap.new_texture(&desc(70, 0x3), 0).unwrap().is_some());
        assert!(heap.new_texture(&desc(130, 0x1), 0x8000).unwrap().is_some());
        assert!(heap.new_texture(&desc(166, 0x1), 0xc000).unwrap().is_some());
    }

    /// A writable heap of 4 KiB over a page of the guest's, as core's detach heap makes one of
    /// an x86_64 guest's pages, is made on every host. The runtime's heap covers 16 KiB
    /// ([`crate::gpu::backing::HEAP_GRANULE`]), so on a host of 4 KiB pages its alias has zeros
    /// past the page: on Linux a file of their own, which the runtime imports with the guest's
    /// page as dma-bufs (`crate::gpu::backing::gather`); anonymous zeros beside them would import
    /// neither as host memory nor as a dma-buf
    #[test]
    fn a_heap_over_less_than_its_granule_is_made() {
        use crate::core::platform::types::ResourceAddressRange;

        const SIZE: u64 = 0x1000;
        let (p, d) = (platform(), dev());
        // Left mapped: the heap's alias shares them until the process ends.
        let pages = crate::os::allocate_aliasable(SIZE).unwrap();
        let ranges = p
            .new_address_range_array(&[ResourceAddressRange {
                address: u64::try_from(pages.addr()).unwrap(),
                length: SIZE,
            }])
            .unwrap();
        let desc = HeapDesc {
            size: SIZE,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(0), OptionSet::HazardTrackingMode(0)],
            address_ranges: Some(&ranges),
            pinned_gpu_address: None,
        };
        *crate::answer::LAST_LINE.lock() = None;
        let made = catch_death(|| d.new_heap(&desc).unwrap().heap.is_some());
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert!(matches!(made, Ok(true)), "{line}");
    }

    /// The least size of [`heap_over_pages`]' heaps ([`over_pages`]).
    pub(crate) const OVER_PAGES: u64 = 0x10000;

    /// The size of [`heap_over_pages`]' heaps: [`OVER_PAGES`] where the test device answers two
    /// of their `Depth32Float_Stencil8` textures in it, one beside the other, as MoltenVK does
    /// (the macOS comparisons run Metal's heaps at the same size); else the least multiple of it
    /// where they fit. A native device's image of such a texture can take more than Metal's
    /// (`heap-texture-larger-than-metal`, relayed there: RADV 26.0.8 answers 0x24008 bytes at an
    /// alignment of 0x10000), and a heap places the texture by the device's answer.
    pub(crate) fn over_pages() -> u64 {
        static SIZE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        *SIZE.get_or_init(|| {
            /// A `Shared` heap's options, untracked where its descriptor names the default.
            const OPTIONS: u64 = 0x100;
            let (_, beside, size) = depth_stencil_with(&platform(), &dev(), OPTIONS);
            (beside + size).next_multiple_of(OVER_PAGES)
        })
    }

    /// The address of [`over_pages`] bytes of pages of their own, left mapped: an alias of a heap
    /// or buffer over them shares them until the process ends.
    pub(crate) fn guest_pages() -> u64 {
        let pages = crate::os::allocate_aliasable(over_pages()).unwrap();
        u64::try_from(pages.addr()).unwrap()
    }

    /// The address range array of the [`over_pages`] bytes at `address`.
    fn ranges_at<P: Platform>(p: &P, address: u64) -> P::AddressRangeArray {
        use crate::core::platform::types::ResourceAddressRange;
        p.new_address_range_array(&[ResourceAddressRange {
            address,
            length: over_pages(),
        }])
        .unwrap()
    }

    /// A `Shared` heap of [`over_pages`] bytes as core's detach heap is made: over pages of its
    /// own (core's temporary allocation), detached at once. Core gives it the guest's pages at
    /// its first prepare ([`replace_heap`]), and makes a cover buffer over all of it, which the
    /// tests that need one make.
    fn detached_heap<P: Platform>(p: &P, d: &P::Device) -> P::Heap {
        let ranges = ranges_at(p, guest_pages());
        let heap = d
            .new_heap(&HeapDesc {
                size: over_pages(),
                heap_type: 1,
                options: vec![OptionSet::StorageMode(0), OptionSet::HazardTrackingMode(0)],
                address_ranges: Some(&ranges),
                pinned_gpu_address: None,
            })
            .unwrap()
            .heap
            .unwrap();
        heap.detach_backing().unwrap();
        heap
    }

    /// `heap`'s backing replaced with the pages at `address`, as core's first prepare gives a
    /// detached heap the guest's pages.
    fn replace_heap<P: Platform>(p: &P, heap: &P::Heap, address: u64) {
        let replaced = heap.replace_backing(&ranges_at(p, address), false);
        assert!(replaced.unwrap(), "the heap's backing replaced");
    }

    /// A heap over the pages at `address`, made as core makes its detach heap: a
    /// [`detached_heap`], given the pages by [`replace_heap`].
    pub(crate) fn heap_at<P: Platform>(p: &P, d: &P::Device, address: u64) -> P::Heap {
        let heap = detached_heap(p, d);
        replace_heap(p, &heap, address);
        heap
    }

    /// A `Shared` heap of [`over_pages`] bytes over pages of its own, a `Depth32Float_Stencil8`
    /// render target of its options, which the runtime does not lay out in Metal's tiles there
    /// (`heap-texture-untiled-layout`), and the offset past the texture's bytes at its alignment,
    /// where a second one fits. Only the Metal comparisons (macOS) call it.
    #[cfg(target_os = "macos")]
    pub(crate) fn heap_over_pages<P: Platform>(
        p: &P,
        d: &P::Device,
    ) -> (P::Heap, P::TextureDescriptor, u64) {
        heap_over_pages_at(p, d, guest_pages())
    }

    /// [`heap_over_pages`]' heap, texture and offset, the heap over the pages at `address`.
    #[cfg(target_os = "macos")]
    pub(crate) fn heap_over_pages_at<P: Platform>(
        p: &P,
        d: &P::Device,
        address: u64,
    ) -> (P::Heap, P::TextureDescriptor, u64) {
        let heap = heap_at(p, d, address);
        let (depth_stencil, beside) = depth_stencil_of(p, d, &heap);
        (heap, depth_stencil, beside)
    }

    /// [`heap_over_pages`]' texture of `heap`'s options, and the offset past its bytes.
    #[cfg(target_os = "macos")]
    fn depth_stencil_of<P: Platform>(
        p: &P,
        d: &P::Device,
        heap: &P::Heap,
    ) -> (P::TextureDescriptor, u64) {
        let (depth_stencil, beside, size) = depth_stencil_with(p, d, heap.resource_options());
        assert!(
            beside + size <= over_pages(),
            "{size:#x} beside {beside:#x}"
        );
        (depth_stencil, beside)
    }

    /// [`heap_over_pages`]' texture of the resource options `options`, the offset past its bytes
    /// at its alignment, and its size, as `d` answers them.
    fn depth_stencil_with<P: Platform>(
        p: &P,
        d: &P::Device,
        options: u64,
    ) -> (P::TextureDescriptor, u64, u64) {
        let depth_stencil = p.texture_descriptor(&TextureDesc {
            pixel_format: 260,
            width: 16,
            height: 16,
            usage: 0x4,
            resource_options: Some(options),
            ..TextureDesc::default()
        });
        let answer = d.heap_texture_size_and_align(&depth_stencil).unwrap();
        let beside = answer.size.next_multiple_of(answer.align.max(0x100));
        (depth_stencil, beside, answer.size)
    }

    /// A two-plane XR10 texture (550, 551) of `pixel_format`, 16x16, of `shape`'s type, levels,
    /// slices and depth.
    fn xr10_shaped(pixel_format: u64, shape: fn(&mut TextureDesc)) -> VulkanTextureDescriptor {
        let mut desc = TextureDesc {
            pixel_format,
            width: 16,
            height: 16,
            usage: 0x1,
            resource_options: Some(0x120),
            ..TextureDesc::default()
        };
        shape(&mut desc);
        platform().texture_descriptor(&desc)
    }

    /// A 3D two-plane XR10 texture of two depth slices, whose texels the runtime keeps over its
    /// first slice's planes in the heap's bytes, where Metal's writes lay every slice.
    const THREE_D: fn(&mut TextureDesc) = |d| {
        d.texture_type = 7;
        d.depth = 2;
    };

    /// A mipmapped two-plane XR10 texture (550, 551), whose first level's planes the heap's bytes
    /// keep and whose later levels lie apart from them (`heap-two-plane-layout`,
    /// [`crate::answer::DIES`]), dies at its placement beside the heap's cover, core's buffer over
    /// all of it, made first, in strict mode, naming the kind. A one-level 2D one, a 2D array of
    /// one level and a 3D one of two slices, whose planes the heap's bytes keep (a 3D one's first
    /// slice's), are made beside the cover. One the runtime keeps apart from the heap's bytes
    /// (`heap-two-plane-texture`), past the device's limits, dies beside it, as
    /// `placements_tell_the_heaps_cover_apart` shows of [`Alone::TwoPlane`]
    #[test]
    fn two_plane_textures_beside_the_heaps_cover() {
        let texture_sel = "newTextureWithDescriptor:offset:";
        let mipmapped: fn(&mut TextureDesc) = |d| d.mipmap_level_count = 2;
        let array: fn(&mut TextureDesc) = |d| {
            d.texture_type = 3;
            d.array_length = 2;
        };
        for pixel_format in [550, 551] {
            let h = private_heap();
            let cover = h.new_buffer_of_length(0x8000, 0x120, 0).unwrap().unwrap();
            *crate::answer::LAST_LINE.lock() = None;
            let site = catch_death(|| h.new_texture(&xr10_shaped(pixel_format, mipmapped), 0))
                .unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, texture_sel),
                "{pixel_format}: {line}"
            );
            assert!(line.contains("heap-two-plane-layout"), "{line}");
            for shape in [|_: &mut TextureDesc| {}, array, THREE_D] {
                let made = h.new_texture(&xr10_shaped(pixel_format, shape), 0).unwrap();
                assert!(made.is_some(), "{pixel_format}");
            }
            drop(cover);
        }
    }

    /// A heap's placements, of stand-in objects, by kind: the first buffer over all of the heap is
    /// its cover, beside which a texture of each kind kept apart dies while it lives, naming the
    /// cover and the kind, and is kept apart once it is let go, so a buffer over it dies; a buffer
    /// over all of the heap made after anything else is kept as any buffer is, so it dies over
    /// such a texture, and so is one asked after a first buffer the runtime answered nil to
    #[test]
    fn placements_tell_the_heaps_cover_apart() {
        const HEAP: std::ops::Range<u64> = 0..0x8000;
        let sel = "newTextureWithDescriptor:offset:";
        let weak = |o: &Arc<u8>| -> Weak<dyn std::any::Any + Send + Sync> {
            let weak: Weak<u8> = Arc::downgrade(o);
            weak
        };
        let dies = |place: &mut dyn FnMut(), named: &[&str], case: &str| {
            *crate::answer::LAST_LINE.lock() = None;
            let site = catch_death(std::panic::AssertUnwindSafe(place)).unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{case}");
            for named in named {
                assert!(line.contains(named), "{case}: {line}");
            }
        };
        let beside = "while the buffer over all of the heap that core makes with it lives";
        let kinds = [
            (Alone::TwoPlane, "heap-two-plane-texture"),
            (Alone::Optimized, "optimized-depth-stencil-texture"),
            (Alone::Untiled, "heap-texture-untiled-layout"),
        ];
        for (why, named) in kinds {
            let mut p = Placements::default();
            let cover = Arc::new(0);
            assert!(p.attempt(), "core's first buffer");
            p.place_buffer(sel, HEAP, weak(&cover), true);
            let t = Arc::new(1);
            dies(
                &mut || p.place(sel, 0x1000..0x1100, weak(&t), Some(why)),
                &[beside, named],
                &format!("{why:?} beside the cover"),
            );
            drop(cover);
            p.place(sel, 0x1000..0x1100, weak(&t), Some(why));
            let b = Arc::new(2);
            dies(
                &mut || p.place(sel, 0x1080..0x1180, weak(&b), None),
                &[named],
                &format!("a buffer over {why:?} once the cover is let go"),
            );
        }

        let guests = Arc::new(3);
        let over = Arc::new(4);
        let mut p = Placements::default();
        let two_plane = Arc::new(8);
        p.place(sel, 0..0x100, weak(&two_plane), Some(Alone::TwoPlane));
        dies(
            &mut || {
                let first = p.attempt();
                p.place_buffer(sel, HEAP, weak(&guests), first);
            },
            &["heap-two-plane-texture"],
            "a buffer over all of the heap after a two-plane texture",
        );
        let mut p = Placements::default();
        assert!(
            p.attempt(),
            "core's first buffer, which the runtime answers nil to"
        );
        assert!(!p.attempt(), "the guest's");
        p.place_buffer(sel, HEAP, weak(&guests), false);
        dies(
            &mut || p.place(sel, 0..0x100, weak(&over), Some(Alone::Optimized)),
            &["optimized-depth-stencil-texture"],
            "such a texture over a guest's buffer over all of the heap",
        );
        drop((guests, over, two_plane));
    }

    /// A buffer and a texture are made over a two-plane XR10 texture (550, 551) of its heap whose
    /// planes the heap's bytes keep, while it lives, and such a texture over them: a one-level 2D
    /// one, a 2D array of one level and a 3D one of two slices, whose texels lie over its first
    /// slice's planes, where Metal lays them. One the runtime keeps apart from the heap's bytes
    /// (`heap-two-plane-texture`), past the device's limits, is kept alone, as
    /// `placements_tell_the_heaps_cover_apart` shows of [`Alone::TwoPlane`]
    #[test]
    fn placements_over_two_plane_textures_in_the_heaps_bytes_are_made() {
        let xr10 = xr10_shaped;
        let array: fn(&mut TextureDesc) = |d| {
            d.texture_type = 3;
            d.array_length = 2;
        };
        let one_level: fn(&mut TextureDesc) = |_| {};
        for pixel_format in [550, 551] {
            for shape in [one_level, array, THREE_D] {
                let h = private_heap();
                let t = h
                    .new_texture(&xr10(pixel_format, shape), 0)
                    .unwrap()
                    .unwrap();
                let over = h.new_buffer_of_length(0x100, 0x20, 0).unwrap().unwrap();
                let u = h.new_texture(&bgra(16, 16, 0x120), 0).unwrap().unwrap();
                let again = h.new_texture(&xr10(pixel_format, shape), 0).unwrap();
                assert!(again.is_some(), "{pixel_format}");
                drop((t, over, u, again));
            }
        }
    }

    /// A heap's purgeable state answers the state before each change, Metal's sequence
    #[test]
    fn heap_purgeable_states() {
        let h = private_heap();
        let before: Vec<_> = [1, 3, 2, 4, 1, 2]
            .into_iter()
            .map(|s| h.set_purgeable_state(s).unwrap())
            .collect();
        assert_eq!(before, [2, 2, 3, 2, 4, 4]);
    }

    #[cfg(target_os = "macos")]
    pub(crate) mod metal_tests;
}
