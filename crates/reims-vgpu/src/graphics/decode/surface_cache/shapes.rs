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
//! slice, as Metal makes a texture over an IOSurface. The texture's usage is not part of it: the
//! image is made with Metal's Unknown usage, which takes every usage and a view in any format of
//! its texel's size, so an `R32Uint` texture over a `BGRA8Unorm` surface is a view of the same
//! image, not a second shape.
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
//! counted ([`ShapeCounts::usage_writes`]). It holds each, so the bound keeps it, until the buffer's finish marks it dirty; where
//! the buffer was counted behind a deferred copy of the cache, once that copy is made, as the
//! buffer's write follows it (`SurfaceCaches::shapes_written`). Every copy of the buffer into the
//! pages (a download, a release, a block named again, an early copy) copies the dirty shapes into
//! the buffer first, in its own command buffer, after the same command buffers; they are clean
//! once it is committed ([`write_back`], [`ShapeTable::written_back`]). A write the walk does not
//! see (through an argument buffer with no use named, or a video op's) is no shape's: counted
//! ([`ShapeCounts::unseen`]), and no shape is marked for it.
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
}

impl ShapeKey {
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
/// (its usage, `ShaderWrite` or none given; a render pass names its attachments), and the
/// reference it holds.
#[derive(Debug)]
pub(crate) struct ShapeUse {
    pub(crate) block: u64,
    pub(crate) writes: bool,
    pub(crate) reference: ShapeRef,
}

impl ShapeUse {
    /// Another use of the same shape, for a guest view of the texture.
    pub(crate) fn clone_ref(&self) -> ShapeUse {
        ShapeUse {
            block: self.block,
            writes: self.writes,
            reference: self.reference.clone_ref(),
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
    /// walk saw (through an argument buffer with no use named, or a video op): the write is not
    /// known to be any shape's, and no shape is marked dirty for it.
    pub(crate) unseen: u64,
    /// Caches made past the caches' budget, which the per-shape form keeps all the same.
    pub(crate) past_budget: u64,
    /// Uncached blocks refused, the per-shape form taking none (`rt-surface-uncached-copied`).
    pub(crate) uncached_refused: u64,
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
    /// A reference to the shape of `key`, used now (`now`, the caches' clock), where the table
    /// holds it.
    pub(crate) fn find(&mut self, key: &ShapeKey, now: u64) -> Option<(ShapeRef, &Shape<T>)> {
        let shape = self.shapes.iter_mut().find(|s| s.key == *key)?;
        shape.used = now;
        Some((shape.tag.clone_ref(), shape))
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
            used: now,
        });
        (reference, again)
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

    /// The unreferenced shapes past the bound, taken out: all but the [`UNREFERENCED_KEPT`] most
    /// recently used, and of those a dirty one is kept until a copy out of the cache writes it
    /// back (rule 3), counted in `counts`.
    pub(crate) fn sweep(&mut self, counts: &mut ShapeCounts) -> Vec<Shape<T>> {
        let mut unreferenced: Vec<(u64, u64, bool)> = (self.shapes.iter())
            .filter(|s| !s.referenced())
            .map(|s| (s.used, s.id(), s.dirty))
            .collect();
        if unreferenced.len() <= UNREFERENCED_KEPT {
            return Vec::new();
        }
        unreferenced.sort_unstable();
        let past = &unreferenced[..unreferenced.len() - UNREFERENCED_KEPT];
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
            .filter(|s| !s.referenced() && !s.dirty)
            .map(|s| (s.used, s.id()))
            .min()
    }

    /// The shape `id`, taken out.
    pub(crate) fn take_id(&mut self, id: u64) -> Option<Shape<T>> {
        self.take(|s| s.id() == id).pop()
    }

    /// The shapes `going` picks, taken out, their keys remembered as dropped.
    fn take(&mut self, going: impl Fn(&Shape<T>) -> bool) -> Vec<Shape<T>> {
        let (gone, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.shapes)
            .into_iter()
            .partition(|s| going(s));
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
        let cases: [(u64, u64, u64, u64); 3] =
            [(80, 4, 64, 16), (30, 2, 16, 8), (30, 2, 9, 2)];
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
        assert_eq!(rows_of(&table, wide), corner(&vec![0; LENGTH.to_usize()], 32, 8));

        let cb = queue.command_buffer(ON).unwrap().unwrap();
        copy_overlaps::<V>(&cb, &buffer, &table, (whole, &[wide, narrow]), None, &mut counts)
            .unwrap();
        commit(cb);
        assert_eq!(rows_of(&table, wide), corner(&p, 32, 8));
        assert_eq!(rows_of(&table, narrow), corner(&p, 18, 2));
        assert_eq!(
            (counts.overlap_copies, counts.overlap_bytes, counts.straddled),
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
        assert!(read == want.as_slice(), "the whole's overlap alone, at its place");

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
        assert_eq!(rows_of(&table, whole), want, "16 bytes of 18; the straddled texel kept");
        assert_eq!(counts.straddled, 1);
        assert_eq!(counts.overlap_copies, 3);
        drop(held);
    }
}
