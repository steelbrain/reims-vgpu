//! The per-shape form of a device cache's textures (the rt platform's;
//! [`super::Caches::set_shapes`]).
//!
//! In the default form every texture over a cached surface is the platform's texture over the
//! cache's buffer. In the per-shape form, which the rt platform takes
//! (`ApiPlatform::SURFACE_SHAPES`), no texture lies over the buffer. The buffer stays the device's
//! copy of the surface's pages, and every fill, download, early copy and release moves bytes
//! between it and the pages as in the default form. Each shape of texture the guest makes over the
//! surface has an image of its own (a [`Shape`]). The guest's texture is a view of that image at
//! the guest's resource index, in the guest's own pixel format, so two textures of one shape are
//! two views of one image.
//!
//! **The shape** ([`ShapeKey`]): the plane, where it lies in the buffer (its byte and its rows'
//! pitch), its size and its texel's bytes ([`texel_bytes`]), of a 2D texture of one level and one
//! slice, as Metal makes a texture over an IOSurface, and whether it takes storage. A texture
//! takes storage where it declares Metal's Unknown usage (0), `ShaderWrite` or `ShaderAtomic`.
//! A storage shape's image is made with the Unknown usage, which takes every usage and a view in
//! any format of its texel's size, so an `R32Uint` texture over a `BGRA8Unorm` storage shape is
//! a view of the same image, not a second shape. A plain shape's image, of a texture that takes no
//! storage, is made with `ShaderRead | RenderTarget` alone: a texture rt reads while a pass
//! renders into it is rt's to build only on such an image, and the long list of view formats
//! that `PixelFormatView` brings costs a compositing pass time on some GPUs. A plain image is
//! viewed in its own format and its sRGB sibling alone, as Metal views a texture with no
//! `PixelFormatView` usage: a texture over a plain shape in any other format is a view rt
//! refuses, counted ([`PLAIN_VIEW_REFUSED`]), and lies over the buffer as one whose view the
//! platform does not make.
//!
//! **A storage shape serves every texture of its extent** (its key but for the storage bit),
//! whatever the texture declares: [`ShapeTable::find`] prefers it. A texture that takes storage
//! over a plane with only a plain shape of its extent makes a storage shape beside it, and the two
//! are kept current by rule 1 below, as any two shapes of a plane are: counted while a plane holds
//! both kinds ([`ShapeCounts::order_case`], [`ShapeCounts::order_copies`]). The bound drops a
//! plain shape that a storage shape of its extent supersedes as soon as no texture references it,
//! and a dirty one once a copy out of the cache has written it back (rule 3,
//! [`ShapeTable::sweep`]).
//!
//! **Every image an entry holds is current over its whole extent.** Three rules keep it so:
//! - a new shape takes the buffer's bytes before anything reads it, once the dirty shapes are
//!   copied into the buffer (rule 2);
//! - a shape is dirty once written since it was last copied into the buffer, and a copy of the
//!   buffer into the pages first copies every dirty shape's whole extent into the buffer; a dirty
//!   shape is not dropped until such a copy (rule 3; dirty shapes agree where they overlap, so
//!   their order is free);
//! - a write through one shape is copied, at the end of the encoder that may have written it,
//!   into each other shape of its plane that a live guest texture references or that holds a
//!   write of its own, and the plane's other shapes are dropped (rule 1, [`copy_overlaps`]): the
//!   next encoder reads the write through any texture over the plane, as the Macs' next encoders
//!   read a write made through another texture over the same memory
//!   (docs/graphics-kext/dgpu-oracle.md, Q37). Every two shapes of a plane overlap from its first
//!   texel ([`ShapeKey::overlap`]).
//!
//! **Two shapes of one plane written in one encoder** cannot be ordered: Metal writes the plane's
//! one memory through both in the order the GPU runs the writes, and the per-shape form holds
//! them apart until the encoder ends. Such an encoder is refused ([`UNORDERED_WRITES`], counted
//! and logged once), and the shape the walk noted last is taken as the plane's write.
//!
//! **Dirty.** A command buffer's walk notes each shape a use may write: a blit's destination, a
//! render pass's attachment or resolve texture, a texture bound at a slot that a pipeline bound
//! with it may write (its access in the function's interface write or read-write, not read or
//! sample), and, outside a blit encoder, a texture whose usage lets a shader write it where no
//! interface is at hand (a pipeline that answers none, a use through an argument buffer),
//! counted ([`ShapeCounts::usage_writes`]). It holds each, so the bound keeps it, until the
//! buffer's finish marks it dirty; where the buffer was counted behind a deferred copy of the
//! cache, once that copy is made, as the buffer's write follows it
//! (`SurfaceCaches::shapes_written`). Every copy of the buffer into the pages (a download, a
//! release, a block named again, an early copy) copies the dirty shapes into the buffer first, in
//! its own command buffer, after the same command buffers; they are clean once it is committed
//! ([`write_back`], [`ShapeTable::written_back`]). A write the walk does not see through an
//! argument buffer with no use named is no shape's: counted ([`ShapeCounts::unseen`]), and no
//! shape is marked for it.
//!
//! **Video.** A command buffer whose work is video alone reads and writes the buffer, where its
//! decodes land their pictures and its encodes read their sources, not the shapes' images: the
//! dirty shapes are copied into the buffer before it runs, by a fill of the shapes
//! (`SurfaceCaches::before_one`), and where it may write the cache every shape owes a fill from
//! the buffer after it, its own write lost, as the decode writes its planes whole
//! ([`ShapeTable::owe_all`], counted [`ShapeCounts::video_owed`]). A command buffer of video and
//! other work writes the buffer unseen.
//!
//! **Fills.** A new shape owes its fill from the buffer. The first use of the surface, or present
//! of it, that finds a shape owing one makes a fill of the shapes alone on the transfer queue,
//! after the command buffers that used the cache, which copies the dirty shapes into the buffer
//! and then the buffer into the shapes that owe it (`SurfaceCaches::fill_shapes`). Behind command
//! buffers not finished it is asked of them as a fill of the cache is, and made in the last one's
//! finish, once their walks' shapes are marked; the uses counted after it wait for it on the GPU
//! either way. A fill of the buffer from the pages (a newer CPU generation, a recovery, a block
//! named again) refills every shape from the buffer, and the CPU's bytes win over the GPU's
//! unwritten-back ones, as they win over the buffer's.
//!
//! **Placed images.** A new shape's image is made by the platform, which clears it. Where the
//! platform places images (`ApiPlatform::PLACED_SHAPE_IMAGES`) and no command buffer begun and
//! not finished uses the cache and no copy of it waits, its image is placed instead
//! (`ApiPlatform::placed_shape_image`), in memory whose bytes are as they were, and filled at
//! its make, before the shape is answered (`super::Caches::placed_shape`). A shape that fill left
//! owing goes, its image with it, and the shape is made again with the platform's own image:
//! counted by cause ([`UNPLACED_BUSY`], [`UNPLACED_UNFILLED`]). Nothing reads a placed image's
//! bytes before the fill wrote them: every use of the cache waits for its last fill.
//!
//! **Zeros.** A cache's buffer whose memory is not made yet (`super::SurfaceCaches`'s "Deferred
//! memory") reads zero, as a new IOSurface's bytes do: every copy into it makes its memory first.
//! While it holds none and no shape holds a write the buffer lacks, a fill from it would copy
//! zeros, and a shape on the platform's own image, which the platform cleared at its make
//! ([`Shape::cleared`]), holds them already: its fill is taken as made, with no copy, and the
//! buffer's memory is left unmade (`SurfaceCaches::fill_shapes`), counted ([`ZERO_TAKEN`]). The
//! guest reads the same zeros either way. A placed image holds its memory's bytes as they were,
//! so it is never taken so: its fill makes the buffer's memory and copies its zeros, as before.
//! Where the cache would hold zeros so, a new shape is not placed, but in a block taken ahead
//! (below): the platform's own image takes its zeros from the platform's clear, with no block
//! taken for the buffer nor a copy (`super::Caches::shape`), counted ([`UNPLACED_ZEROS`]).
//!
//! Placing trades the platform's clears for takes of blocks. Measured on rt on a Radeon 780M
//! (RADV), a desktop and 3D-scene boot (11-19), against the same boot with the platform's own
//! images: 397 new shapes were placed (1164 MiB), none fell back; rt cleared 1408 textures
//! (652 MiB) where it had cleared 1993 (1721 MiB), 1069 MiB less; and rt's takes of blocks rose
//! from 45 calls (3.64 GiB) to 76 (4.39 GiB), 31 more and 0.75 GiB more taken, the longest call
//! 45.7 ms where it had been 27.5 ms (rt's allocation, `rt-diag allocation sites`). The trade
//! is taken: a clear is the GPU's work on every new shape's image, before the guest's own, and
//! placing does 1069 MiB less of it; the longer takes are rt's allocation of its blocks, which
//! keeping blocks ahead of their takes is rt's to shorten.
//!
//! **Blocks taken ahead.** rt makes a block over its unit (16 MiB, `super::PIECE`) as an
//! allocation of its own on the thread that takes it, so a new shape over the unit would take one
//! on the decoding thread, the guest's ring: a 34.5 MiB image's took 24.6 ms there, and the
//! longest of rt's takes in a desktop and 3D-scene boot 45.7 ms (**measured** on rt on a Radeon
//! 780M, 11-19). So
//! where the platform places images, a cache of one plane over the unit, of the surface's own and
//! not the host's video's, has the block of its whole-extent shape's image taken ahead, once its
//! `SURFACE_PAGES` is answered and settled (`super::Caches::take_shape_block_ahead`): on a worker
//! of its own (`super::SHAPE_BLOCKS_THREADS`), which nothing waits for, under the reset gate held
//! shared, of the length a `BGRA8Unorm` image of the plane's rows and texel bytes takes, plain or
//! storage (`ApiPlatform::shape_block`). The block is kept in the cache, and the cache's first new
//! shape takes it: where the shape is over the unit and placed now, its image is placed in the
//! block (`ApiPlatform::placed_shape_image_in`) and filled as above, though its fill would copy
//! zeros: the fill then makes the buffer's memory, which the cache's first use would make, and
//! copies its zeros, where the platform's own image would take a make on the ring. One that does
//! not fit gives the block back to rt and is placed as before, or is the platform's own where
//! its fill would copy zeros. A first new shape that finds the block asked and not yet taken
//! makes its image as before, never waiting for the take, and the block goes as it comes. Every
//! block that goes with no image in it goes back to rt. A cache asks once, and keeps no block
//! past its first new shape, so a block is held only until the surface's first texture or its
//! release. Counted: the takes, with their time and KiB (`rt-shape-blocks-ahead`), the
//! shapes over the unit that came before the block (`-late`), the blocks the image did not fit
//! (`-unfit`), the asks the platform gave none for (`-refused`), and the blocks that went unused
//! (`-unused`), beside the decoding thread's makes of shapes' images, all, those over the unit,
//! and those placed in a block taken ahead (`rt-shape-image-makes`, `-over-unit`, `-ahead`).
//!
//! **Plane shapes.** A cache of a two-plane surface the host's video writes keeps its planes in
//! a surface the platform exports (`super::exported`): each plane's whole extent is a plane
//! shape, its image the surface's plane, made with `PixelFormatView`, so a guest texture over the
//! plane in another format of the texel's size (an `R8Uint` over an `R8Unorm` plane) is a view of
//! it. The host's video writes its pictures into those images through the surface's memory: a
//! video command buffer whose decodes land there leaves the plane shapes holding the picture and
//! every other shape owing a fill ([`ShapeTable::owe_but_planes`]), and, while the cache's buffer
//! is a copy of its own, the plane shapes dirty, so the copies out and the fills of the others
//! take the picture from them (rule 3). The bound never drops a plane shape, referenced or not,
//! and rule 1 copies a write through another shape of the plane into it. The surface's memory
//! holds undefined bytes at its make, so the planes take their fill from the buffer then, at the
//! texture's make, as a placed image does, where no command buffer begun and not finished uses
//! the cache and no copy of it waits (`super::exported`, counted
//! [`ShapeCounts::planes_filled_at_make`]); a decode's begin then owes none, and its wait for the
//! cache's last fill finds that fill made ahead of it. A busy cache's planes owe theirs to its
//! next use, as any new shape's ([`ShapeCounts::planes_owed_busy`]). A picture that lands over
//! part of the planes alone (a picture smaller than them, or one not written) leaves the rest
//! holding that fill. A CPU lock of the surface never reads the surface's memory beside the
//! planes' texels (each row's padding, and the rows of room below): a fill and a write-back move
//! the shapes' texels alone, and a copy out of a cache of a surface the kext laid out moves its
//! planes' texels alone (`super::SurfaceCache`'s `texels`), so those bytes of the pages stay as
//! the guest's CPU wrote them, a new IOSurface's zeros. On Windows the planes lie over
//! Direct3D 11 textures that the pictures are drawn into, and take no fill at their make: a
//! decode that takes a new cache's first fill as met takes its plane shapes' fills as met too
//! (`ShapeTable::take_plane_fills`), as the buffer holds its make's zeros and the textures are
//! cleared to zeros wherever the picture's draw does not write them (`super::exported`).
//!
//! **References.** Each guest texture over a shape, and each guest view of one, holds a
//! [`ShapeRef`] in its decoder entry; the count of those is the shape's reference count, so every
//! way an entry goes (its destroy, its id made again, its context's close, the device's reset)
//! drops a reference with no lock and no message. The guest's release of a texture moves no byte.
//!
//! **The bound.** An entry keeps every referenced shape and, of its unreferenced ones, the
//! [`UNREFERENCED_KEPT`] most recently used ([`ShapeTable::sweep`]); the shapes' bytes count in
//! the caches' budget ([`super::SurfaceCaches::set_cache_budget`]), which trims unreferenced
//! shapes, least recently used first, and never a referenced one. The same shape made again finds
//! its image and costs nothing; one made again after its image was dropped is counted
//! ([`ShapeTable::insert`]), which is what a change of [`UNREFERENCED_KEPT`] would move.
//!
//! **Formats.** A shape is made for a texture of an ordinary colour format ([`texel_bytes`]). A
//! compressed or 4:2:2 texture reads blocks of texels, a two-plane YCbCr one both of its surface's
//! planes, which Metal makes no view of, and a depth and stencil one takes only one of its aspects
//! from a buffer; such a texture lies over the buffer as in the default form, counted.

use std::collections::VecDeque;
use std::sync::Arc;

use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder as _};
use reims_vgpu_vulkan::core::platform::command::CommandBuffer as _;
use reims_vgpu_vulkan::core::platform::types::{MtlOrigin, MtlSize, PixelFormatInfo};
use reims_vgpu_vulkan::core::raise::{OnFault, RaiseResult};
use reims_vgpu_vulkan::gpu::no_texture::no_view;

/// How many unreferenced shapes an entry keeps, the most recently used (the planner's ruling of
/// 2026-11-19: referenced shapes and 4 more).
pub(crate) const UNREFERENCED_KEPT: usize = 4;

/// The degradation code of a texture over a plain shape in a format that is neither the shape's
/// nor its sRGB sibling: a view rt refuses, as Metal's validation refuses a view in another format
/// of a texture without `PixelFormatView` usage (the module doc, "The shape"). The texture lies
/// over the buffer as one whose view the platform does not make.
pub(crate) const PLAIN_VIEW_REFUSED: &str = "rt-refused-surface-plain-view-format";

/// The counts' codes of shapes made over a plane that held a shape of the other kind, plain or
/// storage, and of rule 1's copies between a plain and a storage shape and their bytes
/// ([`ShapeCounts::order_case`]): in the process's counts line, which the boot reads at its end.
pub(crate) const ORDER_CASE: &str = "rt-shape-order-case";
pub(crate) const ORDER_COPIES: &str = "rt-shape-order-copies";
pub(crate) const ORDER_COPY_BYTES: &str = "rt-shape-order-copy-bytes";

/// The counts' codes of shapes made with the platform's own image where it places them (the
/// module doc, "Placed images"): their cache was in use, or the fill at their make left them
/// owing it ([`ShapeCounts::unplaced_busy`], [`ShapeCounts::unplaced_unfilled`]).
pub(crate) const UNPLACED_BUSY: &str = "rt-shape-unplaced-busy";
pub(crate) const UNPLACED_UNFILLED: &str = "rt-shape-unplaced-unfilled";

/// The counts' codes of the zeros (the module doc, "Zeros"): a shape whose fill was taken as made,
/// its cache's buffer holding no memory yet and its image the platform's cleared one
/// ([`ShapeCounts::zero_taken`]); and a shape made with the platform's own image where it places
/// them, as its cache would hold zeros ([`ShapeCounts::unplaced_zeros`]).
pub(crate) const ZERO_TAKEN: &str = "rt-shape-zero-taken";
pub(crate) const UNPLACED_ZEROS: &str = "rt-shape-unplaced-zeros";

/// The degradation code of an encoder that may have written two shapes of one plane: Metal
/// would write the plane's one texture memory through both in the order the GPU ran them, which
/// the per-shape form cannot know; the shape the walk noted last is taken as the plane's write
/// (rule 1, `SurfaceCaches::reconcile_shapes`).
pub(crate) const UNORDERED_WRITES: &str = "rt-refused-surface-unordered-writes";

/// How many keys of dropped shapes an entry remembers, for the count of shapes made again after
/// their image was dropped ([`ShapeTable::insert`]).
const DROPPED_REMEMBERED: usize = 16;

/// `MTLPixelFormatInfo`'s flags bit of a format of blocks (the compressed formats, `GBGR422` and
/// `BGRG422`; `super::super::video`'s own).
const BLOCKS: u64 = 0x400;

/// `MTLPixelFormat`s of depth and stencil, which a copy from a buffer fills only in part where
/// they hold both: `Depth16Unorm`, `Depth32Float`, `Stencil8`, `Depth24Unorm_Stencil8`,
/// `Depth32Float_Stencil8`, `X32_Stencil8`, `X24_Stencil8`.
const DEPTH_STENCIL_FORMATS: [u64; 7] = [250, 252, 253, 255, 260, 261, 262];

/// The bytes of a texel of `pixel_format`, described by `info`, where a texture of it can be a
/// view of a shape's image: a colour format of texels, not of blocks ([`BLOCKS`]), that Metal
/// makes views in ([`no_view`]), of some bytes. `None` for a compressed, 4:2:2, two-plane YCbCr,
/// depth or stencil format, or one the platform describes with no bytes.
pub(crate) fn texel_bytes(pixel_format: u64, info: &PixelFormatInfo) -> Option<u64> {
    let shaped = info.flags & BLOCKS == 0
        && !no_view(pixel_format)
        && !DEPTH_STENCIL_FORMATS.contains(&pixel_format);
    (shaped && info.block_bytes != 0).then_some(info.block_bytes)
}

/// What decides a texture's image in the per-shape form (the module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ShapeKey {
    /// The surface's plane the texture is over.
    pub(crate) plane: u64,
    /// Where the plane's first texel lies in the cache's buffer, and its rows' pitch there.
    pub(crate) offset: u64,
    pub(crate) bytes_per_row: u64,
    pub(crate) width: u64,
    pub(crate) height: u64,
    pub(crate) texel_bytes: u64,
    /// Its image takes storage: the Unknown usage, for a texture that declares it, `ShaderWrite`
    /// or `ShaderAtomic`; else `ShaderRead | RenderTarget` (the module doc, "The shape").
    pub(crate) storage: bool,
}

impl ShapeKey {
    /// The storage shape of the same extent: the shape [`ShapeTable::find`] prefers.
    pub(crate) fn as_storage(&self) -> ShapeKey {
        ShapeKey {
            storage: true,
            ..*self
        }
    }

    /// Whether `other` is a shape of the same plane, at the same place and pitch in the buffer.
    /// Every two shapes of a plane are, as a texture's place and pitch are its plane's, so the
    /// two overlap from the plane's first texel ([`Self::overlap`]).
    pub(crate) fn same_plane(&self, other: &ShapeKey) -> bool {
        self.plane == other.plane
            && self.offset == other.offset
            && self.bytes_per_row == other.bytes_per_row
    }

    /// The bytes of each row, and the rows, of the buffer that the shape and `other`, of the same
    /// plane ([`Self::same_plane`]), both cover: the rectangle from the plane's first texel as
    /// wide as the narrower one's rows and as high as the lower one.
    pub(crate) fn overlap(&self, other: &ShapeKey) -> (u64, u64) {
        let row = |k: &ShapeKey| k.width.saturating_mul(k.texel_bytes);
        (row(self).min(row(other)), self.height.min(other.height))
    }

    /// The bytes an image of the shape holds: its texels.
    pub(crate) fn bytes(&self) -> u64 {
        (self.width.max(1))
            .saturating_mul(self.height.max(1))
            .saturating_mul(self.texel_bytes)
    }

    /// The bytes of the buffer the shape covers: from its plane's byte, `height` rows of
    /// `bytes_per_row`, the last one only as long as the texels' row.
    pub(crate) fn span(&self) -> std::ops::Range<u64> {
        let row = self.width.saturating_mul(self.texel_bytes);
        let rows = self.height.max(1);
        let end = self
            .offset
            .saturating_add((rows - 1).saturating_mul(self.bytes_per_row))
            .saturating_add(row);
        self.offset..end
    }
}

/// A guest texture's reference to the shape it is a view of (the module doc, "References"):
/// taken again with [`ShapeRef::clone_ref`], and let go of by its drop, which takes no lock.
pub(crate) struct ShapeRef(Arc<()>);

impl ShapeRef {
    /// Another reference to the same shape.
    pub(crate) fn clone_ref(&self) -> ShapeRef {
        ShapeRef(Arc::clone(&self.0))
    }

    /// The shape's identity, as long as it lives: equal for every reference to it.
    pub(crate) fn id(&self) -> u64 {
        Arc::as_ptr(&self.0).addr() as u64
    }
}

impl std::fmt::Debug for ShapeRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ShapeRef({:#x})", self.id())
    }
}

/// A reference to a shape and its image, as the caches hand them out.
pub(crate) type Held<T> = (ShapeRef, T);

/// A guest texture's shape, in its decoder entry (`tables::Entry`'s `shape`): the block whose
/// cache holds it, the shape's identity, whether the texture may be written where it is bound
/// (its usage takes storage: Unknown, `ShaderWrite` or `ShaderAtomic`; a render pass names its
/// attachments), and the reference it holds.
#[derive(Debug)]
pub(crate) struct ShapeUse {
    pub(crate) block: u64,
    pub(crate) writes: bool,
    pub(crate) reference: ShapeRef,
    /// The rt platform's boot diagnostics' note of the texture, which a write noted through it
    /// counts against the other formats live over its plane
    /// (`reims_vgpu_vulkan::rt_platform::diag::shape_written`).
    #[cfg(feature = "rt")]
    pub(crate) diag: Option<reims_vgpu_vulkan::rt_platform::diag::Token>,
}

impl ShapeUse {
    /// Another use of the same shape, for a guest view of the texture.
    pub(crate) fn clone_ref(&self) -> ShapeUse {
        ShapeUse {
            block: self.block,
            writes: self.writes,
            reference: self.reference.clone_ref(),
            #[cfg(feature = "rt")]
            diag: self.diag.clone(),
        }
    }
}

/// What the shapes did, for the log (`CacheCounts`'s `shapes`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ShapeCounts {
    /// Shapes made, and their images' bytes.
    pub(crate) made: u64,
    pub(crate) made_bytes: u64,
    /// Textures made of a shape the entry held.
    pub(crate) found: u64,
    /// Shapes made again after their image was dropped ([`ShapeTable::insert`]).
    pub(crate) again: u64,
    /// Textures over a cache made over the buffer as in the default form, their format no
    /// shape's ([`texel_bytes`]) or their shape's image or view refused.
    pub(crate) unshaped: u64,
    /// Shapes whose image the platform did not make, and views of one it did not make.
    pub(crate) images_refused: u64,
    pub(crate) views_refused: u64,
    /// Shapes filled from the buffer, and the bytes: owed ones in a fill of the shapes alone, and
    /// all of an entry's at a fill of its buffer.
    pub(crate) fills: u64,
    pub(crate) filled_bytes: u64,
    /// Fills of the shapes alone, made on the transfer queue
    /// (`SurfaceCaches::fill_shapes`), and of those, ones asked behind command buffers not
    /// finished.
    pub(crate) fills_made: u64,
    pub(crate) fills_deferred: u64,
    /// Unreferenced shapes dropped, past the bound or the caches' budget.
    pub(crate) dropped: u64,
    pub(crate) dropped_bytes: u64,
    /// Dirty unreferenced shapes the bound kept past it, at each sweep, until a copy out of the
    /// cache writes them back ([`ShapeTable::sweep`]).
    pub(crate) kept_dirty: u64,
    /// Dirty shapes copied into their cache's buffer, and their bytes ([`write_back`]).
    pub(crate) written_back: u64,
    pub(crate) written_back_bytes: u64,
    /// Command buffers' uses that may write a cache with shapes and wrote none of them that the
    /// walk saw (through an argument buffer with no use named, or a video op beside other work):
    /// the write is not known to be any shape's, and no shape is marked dirty for it.
    pub(crate) unseen: u64,
    /// Caches made past the caches' budget, which the per-shape form keeps all the same.
    pub(crate) past_budget: u64,
    /// Uncached blocks refused, the per-shape form taking none (`rt-surface-uncached-copied`).
    pub(crate) uncached_refused: u64,
    /// Blocks named cached whose pages the host made no view of, or no buffer over, given their
    /// caches with the pages copied by the host's CPU (`rt-surface-pages-copied`).
    pub(crate) pages_copied: u64,
    /// Shapes that took another shape's write at the end of the encoder that wrote it, and the
    /// bytes they took (rule 1, [`copy_overlaps`]).
    pub(crate) overlap_copies: u64,
    pub(crate) overlap_bytes: u64,
    /// Unreferenced shapes dropped at the end of an encoder that wrote another shape of their
    /// plane (rule 1), counted in `dropped` too.
    pub(crate) overlap_dropped: u64,
    /// Texels of a shape that another shape's write covered only in part, which kept their bytes
    /// ([`copy_overlaps`]).
    pub(crate) straddled: u64,
    /// Encoders that may have written two shapes of one plane ([`UNORDERED_WRITES`]).
    pub(crate) unordered: u64,
    /// Uses of a shape's view taken as writes by the texture's usage alone, with no pipeline's
    /// interface at hand to say whether its function writes the slot (`GraphicsDecoder`'s
    /// `note_shape`).
    pub(crate) usage_writes: u64,
    /// Views of shapes' images that their usage lets a shader write, bound at a slot that no
    /// pipeline bound with them may write, so taken as read (`GraphicsDecoder`'s
    /// `note_slot_shape`).
    pub(crate) slot_reads: u64,
    /// Shapes owed a fill after a video command buffer wrote their cache's buffer, or a plane of
    /// its surface ([`ShapeTable::owe_all`], [`ShapeTable::owe_but_planes`]).
    pub(crate) video_owed: u64,
    /// Of `fills`, those of a shape a video write owed, and their bytes: the copies a picture
    /// costs past the decode's own.
    pub(crate) video_fills: u64,
    pub(crate) video_filled_bytes: u64,
    /// Plane shapes made over a surface's planes (the module doc, "Plane shapes"), and video
    /// command buffers' writes that landed in them.
    pub(crate) planes_made: u64,
    pub(crate) planes_landed: u64,
    /// Plane shapes' fills taken as met by a decode drawn over a new cache's zeros
    /// ([`ShapeTable::take_plane_fills`], on Windows).
    pub(crate) drawn_fills_taken: u64,
    /// Surfaces made for video caches' planes, those the platform made none of, and the most
    /// caches that held one at once (`super::exported`).
    pub(crate) surfaces: u64,
    pub(crate) surfaces_refused: u64,
    pub(crate) surfaces_live_most: u64,
    /// Shapes made over a plane that held a shape of the other kind, plain or storage (the module
    /// doc, "A storage shape serves every texture of its extent"); and of rule 1's copies, those
    /// between a plain and a storage shape, and their bytes.
    pub(crate) order_case: u64,
    pub(crate) order_copies: u64,
    pub(crate) order_bytes: u64,
    /// Textures over a plain shape in a format neither the shape's nor its sRGB sibling, whose
    /// view rt refuses ([`PLAIN_VIEW_REFUSED`]).
    pub(crate) plain_views_refused: u64,
    /// Textures over a plain shape that declare `PixelFormatView` usage.
    pub(crate) plain_format_views: u64,
    /// Shapes whose image the platform placed and their first fill filled at their make, and
    /// their bytes (the module doc, "Placed images").
    pub(crate) placed: u64,
    pub(crate) placed_bytes: u64,
    /// Shapes made with the platform's own image instead, where the platform places them: a
    /// command buffer begun and not finished used the cache or a copy of it waited
    /// ([`UNPLACED_BUSY`]), or the fill at their make left them owing it ([`UNPLACED_UNFILLED`]).
    pub(crate) unplaced_busy: u64,
    pub(crate) unplaced_unfilled: u64,
    /// Shapes whose fill was taken as made, their cache's buffer holding no memory yet and their
    /// image the platform's cleared one, and their bytes; and shapes made with the platform's own
    /// image where it places them, as their cache would hold zeros (the module doc, "Zeros").
    pub(crate) zero_taken: u64,
    pub(crate) zero_taken_bytes: u64,
    pub(crate) unplaced_zeros: u64,
    /// Fills of plane shapes from their cache's buffer, and of them those made at the planes'
    /// make (the module doc, "Plane shapes"); surfaces whose planes owed their fill past their
    /// make, as a command buffer begun and not finished used the cache or a copy of it waited.
    pub(crate) plane_fills: u64,
    pub(crate) planes_filled_at_make: u64,
    pub(crate) planes_owed_busy: u64,
    /// A video op's waits on the CPU for its surface's cache's last fill that found it not done
    /// (`super::Caches::wait_filled`), and the longest, in microseconds.
    pub(crate) video_fill_waits: u64,
    pub(crate) video_fill_wait_max_us: u64,
}

/// One shape's image and its state.
pub(crate) struct Shape<T> {
    pub(crate) key: ShapeKey,
    pub(crate) image: T,
    /// The bytes it counts in the caches' budget ([`ShapeKey::bytes`]).
    pub(crate) bytes: u64,
    /// The table's own reference: the others are the guest textures'.
    tag: ShapeRef,
    /// Written since it was last copied into the buffer (rule 3).
    pub(crate) dirty: bool,
    /// Owes a fill from the buffer before anything reads it (a new shape, rule 2; or the buffer
    /// took bytes no shape holds, a video op's picture).
    pub(crate) owed: bool,
    /// The fill it owes is a video write's ([`ShapeCounts::video_fills`]).
    video: bool,
    /// A plane shape: its image is a plane of the cache's surface, which the bound never drops
    /// (the module doc, "Plane shapes").
    pub(crate) plane: bool,
    /// Its image is the platform's own, which the platform cleared at its make, not placed, and
    /// no device recovery lost what it held since ([`ShapeTable::insert_cleared`]): with the
    /// cache's buffer holding no memory, it holds the zeros a fill would copy (the module doc,
    /// "Zeros").
    pub(crate) cleared: bool,
    /// The caches' clock at its last use: its make, a texture made of it, or a use of its surface
    /// while a texture referenced it ([`ShapeTable::touch`]).
    used: u64,
}

impl<T> Shape<T> {
    /// How many guest textures reference it.
    pub(crate) fn references(&self) -> usize {
        Arc::strong_count(&self.tag.0).saturating_sub(1)
    }

    /// Whether a live guest texture references it.
    pub(crate) fn referenced(&self) -> bool {
        self.references() > 0
    }

    /// Its identity, as its references name it ([`ShapeRef::id`]).
    pub(crate) fn id(&self) -> u64 {
        self.tag.id()
    }
}

/// One entry's shapes (the module doc).
pub(crate) struct ShapeTable<T> {
    shapes: Vec<Shape<T>>,
    /// The keys of the shapes dropped last, most recent at the back, at most
    /// [`DROPPED_REMEMBERED`].
    dropped: VecDeque<ShapeKey>,
}

impl<T> Default for ShapeTable<T> {
    fn default() -> Self {
        ShapeTable {
            shapes: Vec::new(),
            dropped: VecDeque::new(),
        }
    }
}

impl<T> ShapeTable<T> {
    /// A reference to the shape `key` takes, used now (`now`, the caches' clock), where the table
    /// holds one: the storage shape of its extent where the table holds one, which serves every
    /// texture; else, for a texture that takes no storage, its plain shape (the module doc).
    pub(crate) fn find(&mut self, key: &ShapeKey, now: u64) -> Option<(ShapeRef, &Shape<T>)> {
        let storage = key.as_storage();
        let at = (self.shapes.iter().position(|s| s.key == storage))
            .or_else(|| self.shapes.iter().position(|s| s.key == *key))?;
        let shape = &mut self.shapes[at];
        shape.used = now;
        Some((shape.tag.clone_ref(), shape))
    }

    /// Whether the table holds a shape of `key`'s plane of the other kind, plain or storage.
    pub(crate) fn holds_other_kind(&self, key: &ShapeKey) -> bool {
        (self.shapes.iter()).any(|s| s.key.same_plane(key) && s.key.storage != key.storage)
    }

    /// Whether a storage shape of `key`'s extent supersedes `key`, a plain shape's.
    fn superseded(&self, key: &ShapeKey) -> bool {
        !key.storage && self.shapes.iter().any(|s| s.key == key.as_storage())
    }

    /// Adds the shape of `key`, its image `image`, owing a fill from the buffer where `owed`,
    /// used now (`now`): a reference to it, and whether a shape of the same key was dropped
    /// before (one of the last [`DROPPED_REMEMBERED`] dropped), which the caches count.
    pub(crate) fn insert(
        &mut self,
        key: ShapeKey,
        image: T,
        owed: bool,
        now: u64,
    ) -> (ShapeRef, bool) {
        let again = match self.dropped.iter().position(|k| *k == key) {
            Some(at) => {
                self.dropped.remove(at);
                true
            }
            None => false,
        };
        let tag = ShapeRef(Arc::new(()));
        let reference = tag.clone_ref();
        self.shapes.push(Shape {
            key,
            image,
            bytes: key.bytes(),
            tag,
            dirty: false,
            owed,
            video: false,
            plane: false,
            cleared: false,
            used: now,
        });
        (reference, again)
    }

    /// Adds the shape of `key` as [`Self::insert`] does, its image `image` the platform's own,
    /// which the platform cleared at its make ([`Shape::cleared`]).
    pub(crate) fn insert_cleared(
        &mut self,
        key: ShapeKey,
        image: T,
        owed: bool,
        now: u64,
    ) -> (ShapeRef, bool) {
        let made = self.insert(key, image, owed, now);
        if let Some(shape) = self.shapes.last_mut() {
            shape.cleared = true;
        }
        made
    }

    /// Adds the plane shape of `key`, its image `image` a plane of the cache's surface, owing a
    /// fill from the buffer where `owed`, used now (`now`), as [`Self::insert`] adds a shape: the
    /// bound never drops it (the module doc, "Plane shapes").
    pub(crate) fn insert_plane(
        &mut self,
        key: ShapeKey,
        image: T,
        owed: bool,
        now: u64,
    ) -> (ShapeRef, bool) {
        let made = self.insert(key, image, owed, now);
        if let Some(shape) = self.shapes.last_mut() {
            shape.plane = true;
        }
        made
    }

    /// The shapes' bytes ([`ShapeKey::bytes`]).
    pub(crate) fn bytes(&self) -> u64 {
        self.shapes.iter().map(|s| s.bytes).sum()
    }

    /// Whether a live guest texture references one of the shapes.
    pub(crate) fn any_referenced(&self) -> bool {
        self.shapes.iter().any(Shape::referenced)
    }

    /// The entry's surface was used now (`now`, the caches' clock): named by a command buffer,
    /// presented or filled. Each referenced shape counts as used then, so a shape whose last
    /// texture goes keeps the last use of the surface while it was referenced as its own
    /// ([`Self::sweep`]).
    pub(crate) fn touch(&mut self, now: u64) {
        for shape in self.shapes.iter_mut().filter(|s| s.referenced()) {
            shape.used = now;
        }
    }

    /// Whether a shape owes a fill from the buffer ([`Shape::owed`]).
    pub(crate) fn owes(&self) -> bool {
        self.shapes.iter().any(|s| s.owed)
    }

    /// The shape `id` was written: dirty until it is copied into the buffer (rule 3). Whether the
    /// table holds it.
    pub(crate) fn mark_dirty(&mut self, id: u64) -> bool {
        let shape = self.shapes.iter_mut().find(|s| s.id() == id);
        shape.map(|s| s.dirty = true).is_some()
    }

    /// A video command buffer wrote the buffer, its decodes' pictures, not the images: every
    /// shape owes a fill from it, its own write before the buffer's lost (the decode writes its
    /// planes whole, and a write it does not cover was copied into the buffer before it ran);
    /// counted in `counts`.
    pub(crate) fn owe_all(&mut self, counts: &mut ShapeCounts) {
        for shape in &mut self.shapes {
            shape.owed = true;
            shape.video = true;
            shape.dirty = false;
            counts.video_owed += 1;
        }
    }

    /// A video command buffer wrote its decodes' pictures into the planes of the cache's surface,
    /// which the plane shapes' images are (the module doc, "Plane shapes"): each plane shape holds
    /// them, and is dirty where `buffer_apart`, the buffer a copy of its own that lacks them;
    /// every other shape owes a fill from the buffer, its own write before the picture's lost, as
    /// [`Self::owe_all`]'s does; counted in `counts`.
    pub(crate) fn owe_but_planes(&mut self, buffer_apart: bool, counts: &mut ShapeCounts) {
        counts.planes_landed += 1;
        for shape in &mut self.shapes {
            if shape.plane {
                shape.owed = false;
                shape.video = false;
                shape.dirty = buffer_apart;
                continue;
            }
            shape.owed = true;
            shape.video = true;
            shape.dirty = false;
            counts.video_owed += 1;
        }
    }

    /// A decode that writes the planes whole took the cache's first fill as met, its buffer
    /// holding its make's zeros, where the plane shapes' images lie over Direct3D 11 textures
    /// that its picture is drawn into, cleared to zeros wherever it does not write them (the
    /// module doc, "Plane shapes"): each plane shape that owes a fill owes none. How many owed
    /// one.
    #[cfg(any(windows, test))]
    pub(crate) fn take_plane_fills(&mut self) -> u64 {
        let mut taken = 0;
        for shape in self.shapes.iter_mut().filter(|s| s.plane && s.owed) {
            shape.owed = false;
            shape.video = false;
            taken += 1;
        }
        taken
    }

    /// The shapes `ids` were copied into the buffer ([`write_back`]), by a command buffer now
    /// committed: they hold no write the buffer lacks; counted in `counts`.
    pub(crate) fn written_back(&mut self, ids: &[u64], counts: &mut ShapeCounts) {
        for shape in (self.shapes.iter_mut()).filter(|s| s.dirty && ids.contains(&s.id())) {
            shape.dirty = false;
            counts.written_back += 1;
            counts.written_back_bytes += shape.bytes;
        }
    }

    /// The shapes.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &Shape<T>> {
        self.shapes.iter()
    }

    /// The shapes, to change their state.
    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut Shape<T>> {
        self.shapes.iter_mut()
    }

    /// The unreferenced shapes past the bound, taken out: each plain one a storage shape of its
    /// extent supersedes, and of the others all but the [`UNREFERENCED_KEPT`] most recently used;
    /// of those a dirty one is kept until a copy out of the cache writes it back (rule 3), counted
    /// in `counts`.
    pub(crate) fn sweep(&mut self, counts: &mut ShapeCounts) -> Vec<Shape<T>> {
        let mut past: Vec<(u64, u64, bool)> = Vec::new();
        let mut unreferenced: Vec<(u64, u64, bool)> = Vec::new();
        for shape in self.shapes.iter().filter(|s| !s.referenced() && !s.plane) {
            let one = (shape.used, shape.id(), shape.dirty);
            if self.superseded(&shape.key) {
                past.push(one);
            } else {
                unreferenced.push(one);
            }
        }
        if unreferenced.len() > UNREFERENCED_KEPT {
            unreferenced.sort_unstable();
            past.extend_from_slice(&unreferenced[..unreferenced.len() - UNREFERENCED_KEPT]);
        }
        if past.is_empty() {
            return Vec::new();
        }
        counts.kept_dirty += past.iter().filter(|&&(_, _, dirty)| dirty).count() as u64;
        let going: Vec<u64> = (past.iter())
            .filter(|&&(_, _, dirty)| !dirty)
            .map(|&(_, id, _)| id)
            .collect();
        self.take(|s| going.contains(&s.id()))
    }

    /// The least recently used unreferenced shape that is not dirty, with its clock, where one is
    /// held: the next the budget trims ([`Self::take_id`]).
    pub(crate) fn oldest_unreferenced(&self) -> Option<(u64, u64)> {
        (self.shapes.iter())
            .filter(|s| !s.referenced() && !s.dirty && !s.plane)
            .map(|s| (s.used, s.id()))
            .min()
    }

    /// The shape `id`, taken out.
    pub(crate) fn take_id(&mut self, id: u64) -> Option<Shape<T>> {
        self.take(|s| s.id() == id).pop()
    }

    /// The shapes `going` picks, taken out, their keys remembered as dropped: never a plane shape
    /// (the module doc, "Plane shapes").
    fn take(&mut self, going: impl Fn(&Shape<T>) -> bool) -> Vec<Shape<T>> {
        let (gone, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.shapes)
            .into_iter()
            .partition(|s| !s.plane && going(s));
        self.shapes = kept;
        for shape in &gone {
            if self.dropped.len() == DROPPED_REMEMBERED {
                self.dropped.pop_front();
            }
            self.dropped.push_back(shape.key);
        }
        gone
    }
}

/// Encodes in `cb`, in a blit encoder of its own, a copy of `buffer`, a cache's, into each shape of
/// `table` that `pick` picks, each the buffer's bytes at its place ([`ShapeKey::span`]), which
/// then owes no fill and holds no write the buffer lacks; counted in `counts`. Nothing where it
/// picks none, or the platform made no encoder.
pub(crate) fn fill<P: Platform>(
    cb: &P::CommandBuffer,
    buffer: &P::Buffer,
    table: &mut ShapeTable<P::Texture>,
    pick: impl Fn(&Shape<P::Texture>) -> bool,
    counts: &mut ShapeCounts,
) -> RaiseResult<()> {
    const ON: OnFault = OnFault::Return;
    if !table.iter().any(&pick) {
        return Ok(());
    }
    let Some(blit) = cb.blit_encoder(ON)? else {
        return Ok(());
    };
    for shape in table.iter_mut().filter(|s| pick(s)) {
        let key = shape.key;
        blit.encode(
            ON,
            BlitCmd::CopyBufferToTexture {
                source_buffer: buffer,
                source_offset: key.offset,
                source_bytes_per_row: key.bytes_per_row,
                source_bytes_per_image: key.bytes_per_row.saturating_mul(key.height),
                source_size: MtlSize::new(key.width, key.height, 1),
                destination_texture: &shape.image,
                destination_slice: 0,
                destination_level: 0,
                destination_origin: MtlOrigin::default(),
            },
        )?;
        shape.owed = false;
        shape.dirty = false;
        counts.fills += 1;
        counts.filled_bytes += shape.bytes;
        counts.plane_fills += u64::from(shape.plane);
        if std::mem::take(&mut shape.video) {
            counts.video_fills += 1;
            counts.video_filled_bytes += shape.bytes;
        }
    }
    blit.encode(ON, BlitCmd::EndEncoding)?;
    Ok(())
}

/// Encodes in `cb`, in a blit encoder of its own, a copy of each shape of `table` that `pick`
/// picks into `buffer`, a cache's, its whole extent at its place ([`ShapeKey::span`]): the ids of
/// the shapes copied, for [`ShapeTable::written_back`] once `cb` is committed. Shapes agree where
/// they overlap (rule 1), so their order is free. None where it picks none, or the platform made
/// no encoder.
pub(crate) fn write_back<P: Platform>(
    cb: &P::CommandBuffer,
    buffer: &P::Buffer,
    table: &ShapeTable<P::Texture>,
    pick: impl Fn(&Shape<P::Texture>) -> bool,
) -> RaiseResult<Vec<u64>> {
    const ON: OnFault = OnFault::Return;
    if !table.iter().any(&pick) {
        return Ok(Vec::new());
    }
    let Some(blit) = cb.blit_encoder(ON)? else {
        return Ok(Vec::new());
    };
    let mut ids = Vec::new();
    for shape in table.iter().filter(|s| pick(s)) {
        let key = shape.key;
        blit.encode(
            ON,
            BlitCmd::CopyTextureToBuffer {
                source_texture: &shape.image,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::default(),
                source_size: MtlSize::new(key.width, key.height, 1),
                destination_buffer: buffer,
                destination_offset: key.offset,
                destination_bytes_per_row: key.bytes_per_row,
                destination_bytes_per_image: key.bytes_per_row.saturating_mul(key.height),
            },
        )?;
        ids.push(shape.id());
    }
    blit.encode(ON, BlitCmd::EndEncoding)?;
    Ok(ids)
}

/// Encodes in `cb` rule 1's copy of the shape `source` of `table`, which the encoder just ended
/// may have written, into each of its plane's shapes `into` ([`ShapeKey::overlap`]): in a blit
/// encoder of its own, which first waits for `fence`, the writers'
/// (`SurfaceCaches::writes_fence`), the source's texels that the others overlap go into
/// `buffer`, the cache's, at their place; in a second, each of the others takes its overlap from
/// there. Through the buffer, the copy moves bytes whatever the shapes' formats, as a fill does,
/// and the buffer only takes a write early that the source, dirty, copies into it again. A
/// texel of another shape that the source's rows cover only in part (one wider than the
/// source's, at the end of its rows) keeps its bytes, counted in `counts`
/// ([`ShapeCounts::straddled`]). Nothing where `into` names no shape, the table holds no
/// `source`, or the platform made no encoder.
pub(crate) fn copy_overlaps<P: Platform>(
    cb: &P::CommandBuffer,
    buffer: &P::Buffer,
    table: &ShapeTable<P::Texture>,
    (source, into): (u64, &[u64]),
    fence: Option<&P::Fence>,
    counts: &mut ShapeCounts,
) -> RaiseResult<()> {
    const ON: OnFault = OnFault::Return;
    let Some(from) = table.iter().find(|s| s.id() == source) else {
        return Ok(());
    };
    let targets: Vec<&Shape<P::Texture>> = (table.iter())
        .filter(|s| s.id() != source && into.contains(&s.id()))
        .collect();
    let Some((bytes, rows)) = (targets.iter().map(|t| from.key.overlap(&t.key)))
        .reduce(|(b, r), (bytes, rows)| (b.max(bytes), r.max(rows)))
    else {
        return Ok(());
    };
    let key = from.key;
    let Some(blit) = cb.blit_encoder(ON)? else {
        return Ok(());
    };
    if let Some(fence) = fence {
        blit.encode(ON, BlitCmd::WaitForFence { fence })?;
    }
    blit.encode(
        ON,
        BlitCmd::CopyTextureToBuffer {
            source_texture: &from.image,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::default(),
            source_size: MtlSize::new(bytes.div_ceil(key.texel_bytes), rows, 1),
            destination_buffer: buffer,
            destination_offset: key.offset,
            destination_bytes_per_row: key.bytes_per_row,
            destination_bytes_per_image: key.bytes_per_row.saturating_mul(rows),
        },
    )?;
    blit.encode(ON, BlitCmd::EndEncoding)?;
    let Some(blit) = cb.blit_encoder(ON)? else {
        return Ok(());
    };
    for target in targets {
        let (bytes, rows) = key.overlap(&target.key);
        let texel = target.key.texel_bytes;
        let width = bytes / texel;
        if width * texel < bytes {
            counts.straddled += 1;
        }
        if width == 0 || rows == 0 {
            continue;
        }
        blit.encode(
            ON,
            BlitCmd::CopyBufferToTexture {
                source_buffer: buffer,
                source_offset: key.offset,
                source_bytes_per_row: key.bytes_per_row,
                source_bytes_per_image: key.bytes_per_row.saturating_mul(rows),
                source_size: MtlSize::new(width, rows, 1),
                destination_texture: &target.image,
                destination_slice: 0,
                destination_level: 0,
                destination_origin: MtlOrigin::default(),
            },
        )?;
        counts.overlap_copies += 1;
        counts.overlap_bytes += width * rows * texel;
        if target.key.storage != key.storage {
            counts.order_copies += 1;
            counts.order_bytes += width * rows * texel;
            reims_vgpu_vulkan::answer::count_kind(ORDER_COPIES);
            reims_vgpu_vulkan::answer::count_kind_by(ORDER_COPY_BYTES, width * rows * texel);
        }
    }
    blit.encode(ON, BlitCmd::EndEncoding)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A format's info as Metal answers it: a format of texels' words from +0x18 are its bytes, a
    /// packed word and two more; a format of blocks' its block's.
    fn info(block_bytes: u64, blocks: bool) -> PixelFormatInfo {
        PixelFormatInfo {
            flags: if blocks { BLOCKS } else { 0 },
            block_bytes,
            block_width: if blocks { 4 } else { block_bytes * 257 },
            ..PixelFormatInfo::default()
        }
    }

    fn key(width: u64, texel: u64) -> ShapeKey {
        ShapeKey {
            plane: 0,
            offset: 0,
            bytes_per_row: 4096,
            width,
            height: 64,
            texel_bytes: texel,
            storage: true,
        }
    }

    /// An `R32Uint` texture over a `BGRA8Unorm` surface is the same shape, a view of its image;
    /// compressed, 4:2:2, YCbCr, depth and stencil formats take no shape.
    #[test]
    fn same_sized_colour_formats_share_a_shape() {
        assert_eq!(texel_bytes(80, &info(4, false)), Some(4));
        assert_eq!(texel_bytes(53, &info(4, false)), Some(4));
        assert_eq!(texel_bytes(10, &info(1, false)), Some(1));
        assert_eq!(texel_bytes(252, &info(4, false)), None, "depth");
        assert_eq!(texel_bytes(240, &info(4, true)), None, "4:2:2");
        assert_eq!(texel_bytes(130, &info(8, true)), None, "compressed");
        assert_eq!(texel_bytes(500, &info(1, false)), None, "two-plane YCbCr");
        assert_eq!(texel_bytes(80, &info(0, false)), None, "no bytes");
    }

    /// An image's bytes are its texels'; its span in the buffer ends at its last row's texels.
    #[test]
    fn a_shape_counts_its_texels_and_span() {
        let mut k = key(100, 4);
        assert_eq!(k.bytes(), 100 * 64 * 4);
        assert_eq!(k.span(), 0..63 * 4096 + 400);
        k.offset = 8192;
        assert_eq!(k.span(), 8192..8192 + 63 * 4096 + 400);
    }

    /// Every two shapes of a plane overlap from its first texel: as wide as the narrower one's
    /// rows, by their bytes whatever each texel's, and as high as the lower one; a shape of
    /// another plane, place or pitch is not of the plane
    #[test]
    fn two_shapes_of_a_plane_overlap_from_its_first_texel() {
        let wide = key(32, 4);
        let narrow = ShapeKey {
            height: 100,
            ..key(12, 8)
        };
        assert!(wide.same_plane(&narrow));
        assert_eq!(wide.overlap(&narrow), (12 * 8, 64));
        assert_eq!(narrow.overlap(&wide), (12 * 8, 64));
        assert_eq!(wide.overlap(&key(64, 1)), (64, 64));
        for other in [
            ShapeKey { plane: 1, ..wide },
            ShapeKey { offset: 64, ..wide },
            ShapeKey {
                bytes_per_row: 8192,
                ..wide
            },
        ] {
            assert!(!wide.same_plane(&other));
        }
    }

    /// The same shape made again is found, with another reference; references count as long as
    /// they live, and their drop needs nothing of the table.
    #[test]
    fn references_count_while_their_textures_live() {
        let mut table = ShapeTable::<u32>::default();
        let (first, again) = table.insert(key(100, 4), 7, true, 1);
        assert!(!again);
        let shape = table.iter().next().unwrap();
        assert_eq!(
            (shape.references(), shape.owed, shape.dirty),
            (1, true, false)
        );
        let (second, found) = table.find(&key(100, 4), 2).unwrap();
        assert_eq!(found.image, 7);
        assert_eq!(second.id(), first.id());
        let view = second.clone_ref();
        assert_eq!(table.iter().next().unwrap().references(), 3);
        drop((first, second));
        assert_eq!(table.iter().next().unwrap().references(), 1);
        drop(view);
        assert!(!table.iter().next().unwrap().referenced());
        assert!(
            table.find(&key(101, 4), 3).is_none(),
            "another width, another shape"
        );
    }

    /// Every referenced shape is kept, and of the unreferenced the four most recently used; a
    /// shape made again after its drop is told so.
    #[test]
    fn the_bound_keeps_referenced_and_four_recent() {
        let mut table = ShapeTable::<u64>::default();
        let held: Vec<ShapeRef> = (0..3)
            .map(|w| table.insert(key(10 + w, 4), w, false, w).0)
            .collect();
        for w in 3..9 {
            drop(table.insert(key(10 + w, 4), w, false, w));
        }
        assert_eq!(table.iter().count(), 9);
        let gone = table.sweep(&mut ShapeCounts::default());
        let mut gone: Vec<u64> = gone.iter().map(|s| s.image).collect();
        gone.sort_unstable();
        assert_eq!(gone, vec![3, 4], "the two least recently used unreferenced");
        assert_eq!(table.iter().count(), 7);
        assert!(
            table.sweep(&mut ShapeCounts::default()).is_empty(),
            "within the bound"
        );
        let (_, again) = table.insert(key(13, 4), 3, true, 20);
        assert!(again, "made again after its drop");
        let (_, again) = table.insert(key(50, 4), 50, true, 21);
        assert!(!again);
        drop(held);
    }

    /// A storage shape serves every texture of its extent, and is found first; a plain shape
    /// serves only a texture that takes no storage; a texture that takes storage over a plane
    /// with only a plain shape finds none, and makes a second kind there.
    #[test]
    fn a_storage_shape_serves_every_texture_of_its_extent() {
        let plain = ShapeKey {
            storage: false,
            ..key(100, 4)
        };
        let storage = key(100, 4);
        let mut table = ShapeTable::<u32>::default();
        let (held_plain, _) = table.insert(plain, 1, true, 1);
        assert_eq!(table.find(&plain, 2).map(|(_, s)| s.image), Some(1));
        assert!(
            table.find(&storage, 3).is_none(),
            "a plain shape takes no storage"
        );
        assert!(table.holds_other_kind(&storage));
        assert!(!table.holds_other_kind(&plain));
        let (held_storage, _) = table.insert(storage, 2, true, 4);
        assert_eq!(table.find(&plain, 5).map(|(_, s)| s.image), Some(2));
        assert_eq!(table.find(&storage, 6).map(|(_, s)| s.image), Some(2));
        let other = ShapeKey {
            storage: false,
            ..key(50, 4)
        };
        assert!(table.find(&other, 7).is_none(), "another extent");
        drop((held_plain, held_storage));
    }

    /// The bound drops a plain shape that a storage shape of its extent supersedes once nothing
    /// references it, however few shapes the table holds, and a dirty one only once it is
    /// written back; a plain shape of another extent stays within the bound.
    #[test]
    fn the_bound_drops_a_superseded_plain_shape() {
        let plain = |width| ShapeKey {
            storage: false,
            ..key(width, 4)
        };
        let mut table = ShapeTable::<u64>::default();
        let (held, _) = table.insert(plain(100), 1, false, 1);
        drop(table.insert(plain(50), 2, false, 2));
        let (storage, _) = table.insert(key(100, 4), 3, false, 3);
        let mut counts = ShapeCounts::default();
        assert!(
            table.sweep(&mut counts).is_empty(),
            "the plain shape is referenced"
        );
        assert!(table.mark_dirty(held.id()));
        let id = held.id();
        drop(held);
        assert!(
            table.sweep(&mut counts).is_empty(),
            "dirty, kept until written back"
        );
        assert_eq!(counts.kept_dirty, 1);
        table.written_back(&[id], &mut counts);
        let gone: Vec<u64> = table.sweep(&mut counts).iter().map(|s| s.image).collect();
        assert_eq!(gone, vec![1]);
        assert_eq!(
            table.iter().count(),
            2,
            "the storage shape and the other extent's"
        );
        drop(storage);
    }

    /// A use of the surface counts as a use of each shape a texture references then, and of
    /// no other: a shape whose texture goes ranks by the surface's last use while it was held.
    #[test]
    fn a_use_of_the_surface_touches_its_referenced_shapes() {
        let mut table = ShapeTable::<u64>::default();
        let (held, _) = table.insert(key(10, 4), 10, false, 0);
        for w in 11..14 {
            drop(table.insert(key(w, 4), w, false, w));
        }
        assert!(table.any_referenced());
        table.touch(30);
        drop(held);
        assert!(!table.any_referenced());
        drop(table.insert(key(15, 4), 15, false, 31));
        let gone: Vec<u64> = (table.sweep(&mut ShapeCounts::default()).iter())
            .map(|s| s.image)
            .collect();
        assert_eq!(
            gone,
            vec![11],
            "the oldest untouched, not the one held until 30"
        );
    }

    /// The budget trims the least recently used unreferenced shape first, never a referenced one;
    /// the bytes go with it.
    #[test]
    fn the_budget_trims_unreferenced_oldest_first() {
        let mut table = ShapeTable::<u64>::default();
        let kept = table.insert(key(1, 4), 1, false, 5).0;
        drop(table.insert(key(2, 4), 2, false, 7));
        drop(table.insert(key(3, 4), 3, false, 6));
        let bytes = table.bytes();
        let (used, id) = table.oldest_unreferenced().unwrap();
        assert_eq!(used, 6);
        let shape = table.take_id(id).unwrap();
        assert_eq!(shape.image, 3);
        assert_eq!(table.bytes(), bytes - shape.bytes);
        let (_, id) = table.oldest_unreferenced().unwrap();
        assert_eq!(table.take_id(id).unwrap().image, 2);
        assert!(table.oldest_unreferenced().is_none());
        assert_eq!(table.iter().count(), 1);
        drop(kept);
    }

    /// A dirty shape is kept past the bound and the budget, counted, until it is written back
    /// (rule 3); then the next sweep lets it go. Only the shapes written back are made clean.
    #[test]
    fn a_dirty_shape_stays_until_it_is_written_back() {
        let mut table = ShapeTable::<u64>::default();
        let ids: Vec<u64> = (0..6)
            .map(|w| {
                let (reference, _) = table.insert(key(10 + w, 4), w, false, w);
                reference.id()
            })
            .collect();
        assert!(table.mark_dirty(ids[0]));
        assert!(table.mark_dirty(ids[2]));
        assert!(!table.mark_dirty(u64::MAX), "not the table's");
        let mut counts = ShapeCounts::default();
        let gone: Vec<u64> = table.sweep(&mut counts).iter().map(|s| s.image).collect();
        assert_eq!(gone, vec![1], "the oldest clean one");
        assert_eq!(counts.kept_dirty, 1);
        assert_eq!(table.oldest_unreferenced().map(|(used, _)| used), Some(3));
        table.written_back(&[ids[0], ids[3]], &mut counts);
        assert_eq!(
            (counts.written_back, counts.written_back_bytes),
            (1, 10 * 64 * 4)
        );
        assert!(table.iter().any(|s| s.dirty), "the one not written back");
        let gone: Vec<u64> = table.sweep(&mut counts).iter().map(|s| s.image).collect();
        assert_eq!(gone, vec![0]);
        assert_eq!(counts.kept_dirty, 1);
    }

    /// A decode drawn over a new cache's zeros takes as met the fills its plane shapes owe, and
    /// no other shape's (the module doc, "Plane shapes"): the shapes still owe a fill while
    /// another does.
    #[test]
    fn a_decode_drawn_over_zeros_takes_only_the_plane_shapes_fills() {
        let mut table = ShapeTable::<u64>::default();
        let chroma = ShapeKey {
            plane: 1,
            offset: 64 * 4096,
            ..key(32, 2)
        };
        drop(table.insert_plane(key(64, 1), 0, true, 0));
        drop(table.insert_plane(chroma, 1, true, 0));
        drop(table.insert(key(16, 4), 2, true, 0));
        table.owe_all(&mut ShapeCounts::default());
        assert_eq!(table.take_plane_fills(), 2);
        let owing: Vec<u64> = (table.iter().filter(|s| s.owed)).map(|s| s.image).collect();
        assert_eq!(owing, vec![2], "the other shape's fill is owed still");
        assert!(
            table.iter().all(|s| s.owed || !s.video),
            "a fill taken is no video write's"
        );
        assert!(table.owes());
        assert_eq!(table.take_plane_fills(), 0, "none owed");
    }

    /// On the Vulkan platform, shapes over one cache's buffer, of three formats, sizes and places,
    /// each an image of Metal's Unknown usage made as the per-shape form makes it, take the
    /// buffer's bytes at their place and the buffer's pitch ([`fill`]), and a view of each at a
    /// guest's resource index reads them; dirty, the two inside the whole go back into a buffer of
    /// zeros at their place and the buffer's pitch ([`write_back`]), and nothing else does; a fill
    /// of the shapes that owe one then encodes nothing
    #[test]
    fn on_vulkan_each_shape_takes_the_buffers_bytes_at_its_place() {
        use reims_vgpu_vulkan::core::num::ToUsize as _;
        use reims_vgpu_vulkan::core::platform::blit::BlitEncoder as _;
        use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
        use reims_vgpu_vulkan::core::platform::resource::{Buffer as _, Texture as _};
        use reims_vgpu_vulkan::core::platform::values::TextureDesc;

        use crate::platform::ApiPlatform as _;
        type V = reims_vgpu_vulkan::VulkanPlatform;
        const ON: OnFault = OnFault::Die;
        const ROW: u64 = 256;
        let platform = V::process();
        let device = platform.metal_device(None).unwrap();
        crate::graphics::reserve_guest_texture_ids(&*platform, &device);
        let queue = device.new_command_queue().unwrap();
        // The cache: 16 rows of 256 bytes, no two rows alike.
        let length = ROW * 16;
        let bytes: Vec<u8> = (0..length).map(|i| (i ^ ((i >> 8) * 37)) as u8).collect();
        let buffer = device.new_buffer_of_length(ON, length, 0).unwrap().unwrap();
        // SAFETY: a Shared buffer of `length` bytes, which no command uses yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buffer.contents().as_ptr::<u8>(),
                bytes.len(),
            );
        }
        // (pixel format, texel bytes, width, height, offset): BGRA8Unorm over the whole, RG8Unorm
        // and R32Float inside it.
        let cases: [(u64, u64, u64, u64, u64); 3] = [
            (80, 4, 64, 16, 0),
            (30, 2, 16, 8, 64),
            (55, 4, 8, 4, 2 * ROW + 32),
        ];
        let mut table = ShapeTable::default();
        let mut held = Vec::new();
        for (n, &(pixel_format, texel_bytes, width, height, offset)) in cases.iter().enumerate() {
            let key = ShapeKey {
                plane: 0,
                offset,
                bytes_per_row: ROW,
                width,
                height,
                texel_bytes,
                storage: true,
            };
            let image = platform
                .own_texture(
                    &device,
                    &TextureDesc {
                        texture_type: 2,
                        pixel_format,
                        width,
                        height,
                        usage: 0,
                        resource_options: Some(super::super::STORAGE_PRIVATE),
                        ..TextureDesc::default()
                    },
                )
                .unwrap();
            held.push(table.insert(key, image, true, n as u64).0);
        }
        let run = |encode: &dyn Fn(&<V as Platform>::CommandBuffer)| {
            let cb = queue.command_buffer(ON).unwrap().unwrap();
            encode(&cb);
            cb.commit(ON).unwrap();
            cb.wait_until_completed(ON).unwrap();
        };
        let mut counts = ShapeCounts::default();
        let cb = queue.command_buffer(ON).unwrap().unwrap();
        fill::<V>(&cb, &buffer, &mut table, |s| s.owed, &mut counts).unwrap();
        cb.commit(ON).unwrap();
        cb.wait_until_completed(ON).unwrap();
        assert_eq!(counts.fills, 3);
        assert_eq!(counts.filled_bytes, 64 * 16 * 4 + 16 * 8 * 2 + 8 * 4 * 4);
        assert!(table.iter().all(|s| !s.owed && !s.dirty));
        let mut ids = crate::graphics::guest_texture_ids();
        for (shape, &(pixel_format, texel_bytes, width, height, offset)) in table.iter().zip(&cases)
        {
            let view = (shape.image)
                .new_texture_view_at_index(pixel_format, ids.next().unwrap())
                .unwrap()
                .unwrap();
            let row = width * texel_bytes;
            let out = device
                .new_buffer_of_length(ON, row * height, 0)
                .unwrap()
                .unwrap();
            run(&|cb| {
                let blit = cb.blit_encoder(ON).unwrap().unwrap();
                blit.encode(
                    ON,
                    BlitCmd::CopyTextureToBuffer {
                        source_texture: &view,
                        source_slice: 0,
                        source_level: 0,
                        source_origin: MtlOrigin::default(),
                        source_size: MtlSize::new(width, height, 1),
                        destination_buffer: &out,
                        destination_offset: 0,
                        destination_bytes_per_row: row,
                        destination_bytes_per_image: row * height,
                    },
                )
                .unwrap();
                blit.encode(ON, BlitCmd::EndEncoding).unwrap();
            });
            // SAFETY: the Shared buffer's bytes, which no command writes any more.
            let read = unsafe {
                std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), (row * height).to_usize())
            };
            for y in 0..height {
                let at = (offset + y * ROW).to_usize();
                let want = &bytes[at..at + row.to_usize()];
                let got = &read[(y * row).to_usize()..((y + 1) * row).to_usize()];
                assert_eq!(got, want, "format {pixel_format}, row {y}");
            }
        }
        // The two inside the whole, dirty, written back into a buffer of zeros.
        let back = device.new_buffer_of_length(ON, length, 0).unwrap().unwrap();
        // SAFETY: a Shared buffer of `length` bytes, which no command uses yet.
        unsafe { std::ptr::write_bytes(back.contents().as_ptr::<u8>(), 0, length.to_usize()) };
        let inner: Vec<u64> = (table.iter())
            .filter(|s| s.key.width < 64)
            .map(|s| s.id())
            .collect();
        for &id in &inner {
            assert!(table.mark_dirty(id));
        }
        let cb = queue.command_buffer(ON).unwrap().unwrap();
        let written = write_back::<V>(&cb, &back, &table, |s| s.dirty).unwrap();
        cb.commit(ON).unwrap();
        cb.wait_until_completed(ON).unwrap();
        assert_eq!(written, inner);
        table.written_back(&written, &mut counts);
        assert_eq!(counts.written_back_bytes, 16 * 8 * 2 + 8 * 4 * 4);
        assert!(table.iter().all(|s| !s.dirty));
        let mut want = vec![0u8; length.to_usize()];
        for &(_, texel_bytes, width, height, offset) in &cases[1..] {
            let row = (width * texel_bytes).to_usize();
            for y in 0..height {
                let at = (offset + y * ROW).to_usize();
                want[at..at + row].copy_from_slice(&bytes[at..at + row]);
            }
        }
        // SAFETY: the Shared buffer's bytes, which no command writes any more.
        let read = unsafe {
            std::slice::from_raw_parts(back.contents().as_ptr::<u8>(), length.to_usize())
        };
        assert!(
            read == want.as_slice(),
            "the inner shapes alone, at their places"
        );
        // Nothing owes a fill now: nothing is encoded, nothing counted.
        let cb = queue.command_buffer(ON).unwrap().unwrap();
        fill::<V>(&cb, &buffer, &mut table, |s| s.owed, &mut counts).unwrap();
        cb.commit(ON).unwrap();
        cb.wait_until_completed(ON).unwrap();
        assert_eq!(counts.fills, 3);
        drop(held);
    }

    /// On the Vulkan platform, a shape's write reaches the other shapes of its plane through the
    /// cache's buffer whatever their formats ([`copy_overlaps`]): a BGRA8Unorm shape over the
    /// whole gives two RG8Unorm shapes at its first texel, which hold other bytes, the rows and
    /// bytes they overlap; then the narrower RG8Unorm one gives the whole its two rows of 18
    /// bytes, of which the whole's 4-byte texels take 16, the texel it covers in part keeping
    /// its bytes, counted. The buffer holds the source's overlap at its place, the rest untouched
    #[test]
    fn on_vulkan_each_shape_takes_its_planes_write_through_the_buffer() {
        use reims_vgpu_vulkan::core::num::ToUsize as _;
        use reims_vgpu_vulkan::core::platform::blit::BlitEncoder as _;
        use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
        use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
        use reims_vgpu_vulkan::core::platform::values::TextureDesc;

        use crate::platform::ApiPlatform as _;
        type V = reims_vgpu_vulkan::VulkanPlatform;
        const ON: OnFault = OnFault::Die;
        const ROW: u64 = 256;
        const LENGTH: u64 = ROW * 16;
        let platform = V::process();
        let device = platform.metal_device(None).unwrap();
        let queue = device.new_command_queue().unwrap();
        let buffer = device.new_buffer_of_length(ON, LENGTH, 0).unwrap().unwrap();
        let pattern = |seed: u64| -> Vec<u8> {
            (0..LENGTH)
                .map(|i| ((i ^ ((i >> 8) * 37)) + seed) as u8)
                .collect()
        };
        let (p, q) = (pattern(0), pattern(101));
        // The buffer's bytes become `bytes`, while no command uses it.
        let store = |bytes: &[u8]| {
            // SAFETY: a Shared buffer of LENGTH bytes, which no command uses now.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    buffer.contents().as_ptr::<u8>(),
                    bytes.len(),
                );
            }
        };
        let run = |encode: &dyn Fn(&<V as Platform>::CommandBuffer)| {
            let cb = queue.command_buffer(ON).unwrap().unwrap();
            encode(&cb);
            cb.commit(ON).unwrap();
            cb.wait_until_completed(ON).unwrap();
        };
        // (pixel format, texel bytes, width, height): the whole, and two inside it, all at the
        // plane's first texel.
        let cases: [(u64, u64, u64, u64); 3] = [(80, 4, 64, 16), (30, 2, 16, 8), (30, 2, 9, 2)];
        let mut table = ShapeTable::default();
        let mut held = Vec::new();
        for (n, &(pixel_format, texel_bytes, width, height)) in cases.iter().enumerate() {
            let key = ShapeKey {
                plane: 0,
                offset: 0,
                bytes_per_row: ROW,
                width,
                height,
                texel_bytes,
                storage: true,
            };
            let image = platform
                .own_texture(
                    &device,
                    &TextureDesc {
                        texture_type: 2,
                        pixel_format,
                        width,
                        height,
                        usage: 0,
                        resource_options: Some(super::super::STORAGE_PRIVATE),
                        ..TextureDesc::default()
                    },
                )
                .unwrap();
            held.push(table.insert(key, image, true, n as u64).0);
        }
        let ids: Vec<u64> = held.iter().map(|h| h.id()).collect();
        let (whole, wide, narrow) = (ids[0], ids[1], ids[2]);
        // Each shape's rows, as its image holds them.
        let rows_of = |table: &ShapeTable<<V as Platform>::Texture>, id: u64| -> Vec<Vec<u8>> {
            let shape = table.iter().find(|s| s.id() == id).unwrap();
            let key = shape.key;
            let row = key.width * key.texel_bytes;
            let out = device
                .new_buffer_of_length(ON, row * key.height, 0)
                .unwrap()
                .unwrap();
            run(&|cb| {
                let blit = cb.blit_encoder(ON).unwrap().unwrap();
                blit.encode(
                    ON,
                    BlitCmd::CopyTextureToBuffer {
                        source_texture: &shape.image,
                        source_slice: 0,
                        source_level: 0,
                        source_origin: MtlOrigin::default(),
                        source_size: MtlSize::new(key.width, key.height, 1),
                        destination_buffer: &out,
                        destination_offset: 0,
                        destination_bytes_per_row: row,
                        destination_bytes_per_image: row * key.height,
                    },
                )
                .unwrap();
                blit.encode(ON, BlitCmd::EndEncoding).unwrap();
            });
            // SAFETY: the Shared buffer's bytes, which no command writes any more.
            let read = unsafe {
                std::slice::from_raw_parts(
                    out.contents().as_ptr::<u8>(),
                    (row * key.height).to_usize(),
                )
            };
            read.chunks(row.to_usize()).map(<[u8]>::to_vec).collect()
        };
        // `bytes`' first `width` bytes of each of its first `height` rows.
        let corner = |bytes: &[u8], width: u64, height: u64| -> Vec<Vec<u8>> {
            (0..height)
                .map(|y| bytes[(y * ROW).to_usize()..(y * ROW + width).to_usize()].to_vec())
                .collect()
        };
        let commit = |cb: <V as Platform>::CommandBuffer| {
            cb.commit(ON).unwrap();
            cb.wait_until_completed(ON).unwrap();
        };
        // The whole takes `p`; the two inside it take zeros.
        let mut counts = ShapeCounts::default();
        store(&p);
        let cb = queue.command_buffer(ON).unwrap().unwrap();
        fill::<V>(&cb, &buffer, &mut table, |s| s.id() == whole, &mut counts).unwrap();
        commit(cb);
        store(&vec![0; LENGTH.to_usize()]);
        let cb = queue.command_buffer(ON).unwrap().unwrap();
        fill::<V>(&cb, &buffer, &mut table, |s| s.id() != whole, &mut counts).unwrap();
        commit(cb);
        assert_eq!(
            rows_of(&table, wide),
            corner(&vec![0; LENGTH.to_usize()], 32, 8)
        );

        let cb = queue.command_buffer(ON).unwrap().unwrap();
        copy_overlaps::<V>(
            &cb,
            &buffer,
            &table,
            (whole, &[wide, narrow]),
            None,
            &mut counts,
        )
        .unwrap();
        commit(cb);
        assert_eq!(rows_of(&table, wide), corner(&p, 32, 8));
        assert_eq!(rows_of(&table, narrow), corner(&p, 18, 2));
        assert_eq!(
            (
                counts.overlap_copies,
                counts.overlap_bytes,
                counts.straddled
            ),
            (2, 32 * 8 + 18 * 2, 0)
        );
        let mut want = vec![0u8; LENGTH.to_usize()];
        for y in 0..8 {
            let at = (y * ROW).to_usize();
            want[at..at + 32].copy_from_slice(&p[at..at + 32]);
        }
        // SAFETY: the Shared buffer's bytes, which no command writes any more.
        let read = unsafe {
            std::slice::from_raw_parts(buffer.contents().as_ptr::<u8>(), LENGTH.to_usize())
        };
        assert!(
            read == want.as_slice(),
            "the whole's overlap alone, at its place"
        );

        // The narrow one takes `q`, then gives the whole its two rows.
        store(&q);
        let cb = queue.command_buffer(ON).unwrap().unwrap();
        fill::<V>(&cb, &buffer, &mut table, |s| s.id() == narrow, &mut counts).unwrap();
        commit(cb);
        let cb = queue.command_buffer(ON).unwrap().unwrap();
        copy_overlaps::<V>(&cb, &buffer, &table, (narrow, &[whole]), None, &mut counts).unwrap();
        commit(cb);
        let mut want = corner(&p, ROW, 16);
        for (y, row) in want.iter_mut().enumerate().take(2) {
            let at = y * ROW.to_usize();
            row[..16].copy_from_slice(&q[at..at + 16]);
        }
        assert_eq!(
            rows_of(&table, whole),
            want,
            "16 bytes of 18; the straddled texel kept"
        );
        assert_eq!(counts.straddled, 1);
        assert_eq!(counts.overlap_copies, 3);
        drop(held);
    }
}
