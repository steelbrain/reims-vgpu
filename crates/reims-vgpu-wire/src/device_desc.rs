//! Device-side descriptors reached by GVA: the wire tag 4 surface backing record,
//! wire tag 5 reference-texture handle, and wire tag 11 mapper IOSurface texture
//! view.
//!
//! # Provenance
//!
//! **No oracle backs this module, and none can.** Like [`crate::page_table`],
//! these structures are never carried by a serializer record, so there is no
//! fixture to pin them and nothing here is replayable in CI. They come from the
//! device contract, read out of the guest's own descriptor pages.
//!
//! What that means for a reader: every field below is as trustworthy as the
//! reverse engineering behind it and no more. Where a field's *meaning* was
//! settled by a live census rather than by a symbol, the doc says which census.
//! Do not treat a value that fits as a value that was derived.
//!
//! # Why the layouts are here rather than beside their decoder
//!
//! They were eighteen literal offsets and a run of raw little-endian loads in
//! `reims_vgpu::runtime::objects`, each `ld32(&desc[CONST..])` with its own
//! bounds test — the shape every other family in this crate stopped having.
//! Nothing checked that a plane record's stride matched the offsets read inside
//! it, and one caller re-derived plane 0's width and height as
//! `SURFACE_BACKING_PLANES + 4` and `+ 8` rather than through the plane decoder, so the
//! stride was stated in one place and its interior in two.
//!
//! Align-1 views with fallible constructors are worth as much on an unverified
//! format as on a verified one: the bytes are guest-controlled either way.
//!
//! # Wire tag 4 — surface backing (`allocateBackingHandle`)
//!
//! ```text
//! +0x00  u64  length          byte length of the backing
//! +0x08  u32  backing_pfn     first guest page frame of the backing
//! +0x0c  u32  pixel_format    IOSurface OSType FourCC, or an MTL ordinal
//! +0x10  u8   plane_count     capped at PLANE_CAP by IOSurface itself
//! +0x11  u8   [3]             never observed non-zero; see below
//! +0x14       plane[0]        PLANE_STRIDE bytes each, plane_count of them
//! ```
//!
//! The three bytes at `+0x11` are the record's only undecoded interior. Across
//! one 1766 s x86/Vulkan session with a real GUI login, ≥5983 decodes over 453
//! distinct surface ids and 154 distinct geometries produced exactly two record
//! shapes and **zero** non-zero bytes there. They are padding on the evidence
//! available; they are not *known* to be padding, which is why the device still
//! reports the undecoded span rather than shrinking the record around it.
//!
//! # Wire tag 5 — reference texture handle (`allocateRefTextureHandle`)
//!
//! ```text
//! +0x00  u32  surface_id      IOSurface::getSurfaceID() = the wire tag 4 object id
//! +0x04  u32  owner_task      task whose object list holds that surface
//! +0x08       args            serialized texture args, length desc_len - 8
//! ```
//!
//! and inside `args`:
//!
//! ```text
//! +0x00  u32  kind            0x2f observed
//! +0x04  u32  blob_len        == desc_len - 8
//! +0x08  u32  own_ref         the wire tag 5 object's own ref
//! +0x0c       record          the serialized plane/view texture record
//! ```
//!
//! and inside that record:
//!
//! ```text
//! +0x00  u8   tag             0x42 plane view, 0x62 full-colour view
//! +0x01  u8   _unknown
//! +0x02  u16  pixel_format    MTLPixelFormat ordinal
//! +0x04  u32  width
//! +0x08  u32  height
//! +0x0c  u32  depth
//! +0x10       trailer
//! +0x20  u32  plane_index     newTextureWithDescriptor:iosurface:plane:
//! ```
//!
//! The record is `RECORD_MIN_LEN` bytes to the end of `depth`; `plane_index`
//! lies past that and is only present on longer blobs, which is why it has its
//! own accessor rather than a field on the minimum view.

use crate::le::{U16le, U32le, U64le};
use crate::op::op;
use crate::ops::backed_texture;
use crate::view::{view, view_at, Wire, WireError};

/// Wire object type for surface / IOSurface backing.
pub const OBJECT_TYPE_SURFACE: u8 = 4;
/// Wire object type for a reference-texture handle.
pub const OBJECT_TYPE_REF_TEXTURE: u8 = 5;

/// Header of a wire tag 4 surface backing descriptor, up to the plane array.
///
/// `plane_count` is `u8` on the wire and the three bytes after it are the
/// undecoded interior the module doc describes — they are part of this struct
/// so that `size_of` is the offset of plane 0 and the two cannot drift.
#[repr(C)]
#[derive(Debug)]
pub struct SurfaceBackingHeader {
    /// Byte length of the backing. Zero is not a surface this device accepts.
    pub length: U64le,
    /// First guest page frame of the backing. Zero is not a surface either;
    /// physical page zero is never mapped.
    pub backing_pfn: U32le,
    /// IOSurface `getPixelFormat()`. Carries **two** encodings: an OSType
    /// four-character code for media surfaces, or an `MTLPixelFormat` ordinal.
    /// They are disjoint by width — an ordinal fits in 16 bits by construction
    /// and an OSType's four non-zero character bytes cannot — so the consumer
    /// decides by that and not by plausibility.
    pub pixel_format: U32le,
    /// Planes the guest declares. IOSurface's own `getPlaneCount` caps this at
    /// [`SURFACE_BACKING_PLANE_CAP`]; a larger value is a corrupt descriptor rather than
    /// a surface with more planes.
    pub plane_count: u8,
    /// The record's only undecoded interior. Never observed non-zero. See the
    /// module doc for the census; do not repurpose these without one.
    pub reserved: [u8; 3],
}

// SAFETY: align-1 `le` scalars plus `u8`, which is align-1 and all-bytes-valid.
unsafe impl Wire for SurfaceBackingHeader {}

/// One wire tag 4 plane record.
#[repr(C)]
#[derive(Debug)]
pub struct SurfaceBackingPlaneRecord {
    /// Byte offset of the plane within the backing.
    pub offset: U32le,
    pub width: U32le,
    pub height: U32le,
    /// `getPlaneBytesPerRow` in the low 24 bits, `getPlaneBytesPerElement` in
    /// the high 8. Packed on the wire, so the two are read through
    /// [`SurfaceBackingPlaneRecord::bytes_per_row`] and
    /// [`SurfaceBackingPlaneRecord::bytes_per_element`] rather than by masking at each
    /// call site.
    pub packed_bpr: U32le,
}

// SAFETY: four align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for SurfaceBackingPlaneRecord {}

impl SurfaceBackingPlaneRecord {
    /// Low 24 bits of `packed_bpr`.
    pub fn bytes_per_row(&self) -> u32 {
        self.packed_bpr.get() & 0x00ff_ffff
    }

    /// High 8 bits of `packed_bpr`. Zero when the wire left it zero, which is
    /// not the same as a one-byte element.
    pub fn bytes_per_element(&self) -> u8 {
        (self.packed_bpr.get() >> 24) as u8
    }
}

/// Plane records this device will read, matching IOSurface's `getPlaneCount`
/// cap. A descriptor declaring more is corrupt, not wider.
pub const SURFACE_BACKING_PLANE_CAP: usize = 8;

/// Byte stride between wire tag 4 plane records.
pub const SURFACE_BACKING_PLANE_STRIDE: usize = core::mem::size_of::<SurfaceBackingPlaneRecord>();

/// Offset of plane 0, i.e. the length of the header before it.
pub const SURFACE_BACKING_PLANES: usize = core::mem::size_of::<SurfaceBackingHeader>();

/// Shortest wire tag 4 descriptor that carries a header and one plane.
pub const SURFACE_BACKING_MIN_LEN: usize = SURFACE_BACKING_PLANES + SURFACE_BACKING_PLANE_STRIDE;

/// Byte length of a wire tag 4 descriptor declaring `plane_count` planes.
///
/// The device's own census reads this as exact for every descriptor it has
/// seen: `len` is the header plus the declared planes, with nothing after.
pub const fn surface_backing_len_for(plane_count: usize) -> usize {
    SURFACE_BACKING_PLANES + plane_count * SURFACE_BACKING_PLANE_STRIDE
}

/// View the header of a wire tag 4 surface descriptor.
pub fn surface_backing_header(desc: &[u8]) -> Result<&SurfaceBackingHeader, WireError> {
    view::<SurfaceBackingHeader>(desc)
}

/// View plane `index` of a wire tag 4 surface descriptor.
///
/// Fails rather than truncating when the blob does not reach the record: a
/// declared plane whose bytes are absent is a descriptor this device could not
/// decode, not a plane of zero size.
pub fn surface_backing_plane(
    desc: &[u8],
    index: usize,
) -> Result<&SurfaceBackingPlaneRecord, WireError> {
    view_at::<SurfaceBackingPlaneRecord>(
        desc,
        SURFACE_BACKING_PLANES + index * SURFACE_BACKING_PLANE_STRIDE,
    )
}

/// Header of a wire tag 5 reference-texture descriptor.
#[repr(C)]
#[derive(Debug)]
pub struct IOSurfacePlaneViewHeader {
    /// `IOSurface::getSurfaceID()` — the wire tag 4 heap object id this handle
    /// references.
    pub surface_id: U32le,
    /// The task whose object list holds that surface.
    ///
    /// `allocateRefTextureHandle` writes it from the *accelerator's* task
    /// rather than from its own, so this is the owner of the surface and not
    /// the creator of the view; the two differ by construction. The kernel
    /// task's id is an immediate 0 and index 0 is reserved out of the 256-entry
    /// allocator before any client task exists, so every value seen here is
    /// expected to be 0.
    pub owner_task: U32le,
}

// SAFETY: two align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for IOSurfacePlaneViewHeader {}

/// Offset of `surface_id` within a wire tag 5 descriptor.
///
/// Derived from the view rather than stated, so a fixture carrying live wire
/// bytes can name the field without restating the layout. Fixtures keep their
/// literal bytes on purpose — they are what the guest emitted, and bytes
/// assembled to satisfy the reader would agree with it whether or not it is
/// right — but the *offsets they index by* must still come from one place.
pub const IOSURFACE_PLANE_VIEW_SURFACE_ID: usize =
    core::mem::offset_of!(IOSurfacePlaneViewHeader, surface_id);
/// Offset of `owner_task` within a wire tag 5 descriptor. See
/// [`IOSURFACE_PLANE_VIEW_SURFACE_ID`].
pub const IOSURFACE_PLANE_VIEW_OWNER_TASK: usize =
    core::mem::offset_of!(IOSurfacePlaneViewHeader, owner_task);

/// Offset of the serialized args blob within a wire tag 5 descriptor.
pub const IOSURFACE_PLANE_VIEW_ARGS: usize = core::mem::size_of::<IOSurfacePlaneViewHeader>();

/// Header of the wire tag 5 args blob.
#[repr(C)]
#[derive(Debug)]
pub struct IOSurfacePlaneViewArgsHeader {
    /// Kind tag. `0x2f` observed.
    pub kind: U32le,
    /// Blob length, observed equal to `desc_len - IOSURFACE_PLANE_VIEW_ARGS`.
    pub blob_len: U32le,
    /// The wire tag 5 object's own ref, the same convention the IOSurface texture
    /// descriptor uses for its object-ref field.
    pub own_ref: U32le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for IOSurfacePlaneViewArgsHeader {}

/// Offset of the serialized texture record within a wire tag 5 descriptor.
pub const IOSURFACE_PLANE_VIEW_ARG_RECORD: usize =
    IOSURFACE_PLANE_VIEW_ARGS + core::mem::size_of::<IOSurfacePlaneViewArgsHeader>();

/// Record tag for an IOSurface plane view.
pub const IOSURFACE_PLANE_VIEW_RECORD_TAG_PLANE: u8 = 0x42;
/// Record tag for a full-colour texture view.
///
/// The layout is byte-identical to the plane form — the tag distinguishes a
/// variant, not a different geometry encoding — so both decode through this
/// same view.
pub const IOSURFACE_PLANE_VIEW_RECORD_TAG_COLOR_VIEW: u8 = 0x62;

/// The serialized texture record inside a wire tag 5 args blob, to the end of
/// `depth`.
///
/// `plane_index` is deliberately not a field: it sits at
/// [`IOSURFACE_PLANE_VIEW_RECORD_PLANE`], past this record, and is absent from shorter blobs.
/// Making it a field would mean refusing every descriptor that does not carry
/// it.
#[repr(C)]
#[derive(Debug)]
pub struct IOSurfacePlaneViewTextureRecord {
    /// [`IOSURFACE_PLANE_VIEW_RECORD_TAG_PLANE`] or [`IOSURFACE_PLANE_VIEW_RECORD_TAG_COLOR_VIEW`]. Any other
    /// value is an unknown record: fail closed rather than invent geometry.
    pub tag: u8,
    pub _unknown: u8,
    /// `MTLPixelFormat` ordinal of the view.
    pub pixel_format: U16le,
    pub width: U32le,
    pub height: U32le,
    /// Observed 1 on every record. A view this device stages is 2D.
    pub depth: U32le,
}

// SAFETY: two `u8` and three align-1 `le` scalars, all bytes valid.
unsafe impl Wire for IOSurfacePlaneViewTextureRecord {}

impl IOSurfacePlaneViewTextureRecord {
    /// Whether the tag is one of the two forms this device decodes.
    pub fn tag_is_known(&self) -> bool {
        self.tag == IOSURFACE_PLANE_VIEW_RECORD_TAG_PLANE
            || self.tag == IOSURFACE_PLANE_VIEW_RECORD_TAG_COLOR_VIEW
    }
}

/// Bytes of a wire tag 5 texture record up to and including `depth`.
pub const IOSURFACE_PLANE_VIEW_RECORD_MIN_LEN: usize =
    core::mem::size_of::<IOSurfacePlaneViewTextureRecord>();

/// Offset of the plane index within a wire tag 5 texture record.
///
/// The `newTextureWithDescriptor:iosurface:plane:` plane argument. A live
/// three-plane census settles it: Y blobs carry 0, the RG8 chroma blob carries
/// 1, and a second R8 view of identical geometry carries 2. Geometry cannot
/// tell Y from alpha, so this field is the only wire key for it.
pub const IOSURFACE_PLANE_VIEW_RECORD_PLANE: usize = 0x20;

/// View the header of a wire tag 5 descriptor.
pub fn iosurface_plane_view_header(desc: &[u8]) -> Result<&IOSurfacePlaneViewHeader, WireError> {
    view::<IOSurfacePlaneViewHeader>(desc)
}

/// View the nested serializer-operation header of a wire tag 5 descriptor.
pub fn iosurface_plane_view_args_header(
    desc: &[u8],
) -> Result<&IOSurfacePlaneViewArgsHeader, WireError> {
    view_at::<IOSurfacePlaneViewArgsHeader>(desc, IOSURFACE_PLANE_VIEW_ARGS)
}

/// View the serialized texture record of a wire tag 5 descriptor.
pub fn iosurface_plane_view_texture_record(
    desc: &[u8],
) -> Result<&IOSurfacePlaneViewTextureRecord, WireError> {
    view_at::<IOSurfacePlaneViewTextureRecord>(desc, IOSURFACE_PLANE_VIEW_ARG_RECORD)
}

/// The plane index a wire tag 5 descriptor's record names, when it carries one.
///
/// `None` for a blob that stops before the field — pre-plane descriptors and
/// test fixtures — which the caller reads as plane 0.
pub fn iosurface_plane_view_record_plane_index(desc: &[u8]) -> Option<u32> {
    view_at::<U32le>(
        desc,
        IOSURFACE_PLANE_VIEW_ARG_RECORD + IOSURFACE_PLANE_VIEW_RECORD_PLANE,
    )
    .ok()
    .map(|w| w.get())
}

/// Bytes before the complete nested serializer operation in a wire tag 11
/// mapper IOSurface texture view.
pub const MAPPER_IOSURFACE_TEXTURE_OPERATION: usize = core::mem::size_of::<U64le>();

/// Which accepted serializer operation a mapper IOSurface texture embeds.
#[derive(Debug)]
pub enum MapperIOSurfaceTextureOperation<'a> {
    Legacy(&'a backed_texture::IOSurfaceTextureBody),
    Rotated(&'a backed_texture::IOSurfaceTextureRotatedBody),
    Wide(&'a backed_texture::IOSurfaceTextureWideBody),
}

/// Zero-copy view of a wire tag 11 mapper IOSurface texture object.
#[derive(Debug)]
pub struct MapperIOSurfaceTextureView<'a> {
    /// Mapper-service lookup identity. This is not an object-table ref or a GPU
    /// page-table mapping identity.
    pub mapper_ref: &'a U64le,
    pub operation: MapperIOSurfaceTextureOperation<'a>,
}

/// Why a mapper IOSurface texture object could not be viewed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapperIOSurfaceTextureError {
    Wire(WireError),
    /// The nested operation did not consume the complete outer descriptor.
    OuterLength {
        nested: usize,
        outer: usize,
    },
    /// A complete nested operation whose tag/length pair is not an accepted
    /// IOSurface texture form.
    UnknownVariant {
        opcode: u32,
        length: u32,
    },
}

impl From<WireError> for MapperIOSurfaceTextureError {
    fn from(error: WireError) -> Self {
        Self::Wire(error)
    }
}

/// View the mapper reference and complete nested IOSurface texture operation
/// carried by a wire tag 11 object.
///
/// The outer descriptor has no independent tail: after the eight-byte mapper
/// reference, exactly one nested serializer operation must consume every
/// remaining byte. Unknown opcode/length pairs are refused rather than treated
/// as a longer instance of a known descriptor.
pub fn mapper_iosurface_texture(
    desc: &[u8],
) -> Result<MapperIOSurfaceTextureView<'_>, MapperIOSurfaceTextureError> {
    let mapper_ref = view::<U64le>(desc)?;
    let nested_bytes =
        desc.get(MAPPER_IOSURFACE_TEXTURE_OPERATION..)
            .ok_or(WireError::OutOfRange {
                offset: MAPPER_IOSURFACE_TEXTURE_OPERATION,
                len: desc.len(),
            })?;
    let nested = op(nested_bytes, MAPPER_IOSURFACE_TEXTURE_OPERATION)?;
    let consumed = MAPPER_IOSURFACE_TEXTURE_OPERATION + nested.length() as usize;
    if consumed != desc.len() {
        return Err(MapperIOSurfaceTextureError::OuterLength {
            nested: consumed,
            outer: desc.len(),
        });
    }

    let operation = match (nested.opcode(), nested.length()) {
        (backed_texture::OPCODE_IOSURFACE_TEXTURE, backed_texture::IOSURFACE_TEXTURE_TOTAL_LEN) => {
            MapperIOSurfaceTextureOperation::Legacy(backed_texture::iosurface_texture(&nested)?)
        }
        (
            backed_texture::OPCODE_IOSURFACE_TEXTURE_ROTATED,
            backed_texture::IOSURFACE_TEXTURE_ROTATED_TOTAL_LEN,
        ) => MapperIOSurfaceTextureOperation::Rotated(backed_texture::iosurface_texture_rotated(
            &nested,
        )?),
        (
            backed_texture::OPCODE_IOSURFACE_TEXTURE_WIDE,
            backed_texture::IOSURFACE_TEXTURE_WIDE_TOTAL_LEN,
        ) => {
            MapperIOSurfaceTextureOperation::Wide(backed_texture::iosurface_texture_wide(&nested)?)
        }
        (opcode, length) => {
            return Err(MapperIOSurfaceTextureError::UnknownVariant { opcode, length });
        }
    };

    Ok(MapperIOSurfaceTextureView {
        mapper_ref,
        operation,
    })
}

/// Assemble a wire tag 4 descriptor by the format's own rules, for tests.
///
/// The counterpart of [`crate::page_table::Builder`], and here for the same
/// reason: a test that writes `st64(&mut desc[SURFACE_BACKING_LEN..], ..)` is writing
/// bytes chosen to satisfy the reader, so the two agree whether or not the
/// reader is right. Going through the builder means a field the layout moved
/// moves for the writer too.
///
/// It is deliberately not `cfg(test)`: `reims-vgpu`'s own tests are a different
/// crate's, and they are the ones that were spelling the offsets.
#[derive(Debug)]
pub struct SurfaceBackingBuilder {
    bytes: [u8; SURFACE_BACKING_BUILDER_CAP],
    len: usize,
}

/// Longest descriptor [`SurfaceBackingBuilder`] can assemble: header plus every plane
/// IOSurface can declare.
pub const SURFACE_BACKING_BUILDER_CAP: usize =
    SURFACE_BACKING_PLANES + SURFACE_BACKING_PLANE_CAP * SURFACE_BACKING_PLANE_STRIDE;

impl SurfaceBackingBuilder {
    /// A descriptor with a header and room for `planes` plane records.
    ///
    /// `plane_count` is written as given, including a value over
    /// [`SURFACE_BACKING_PLANE_CAP`] — a corrupt descriptor is a thing the device has to
    /// be tested against, so the builder does not clamp what the guest can
    /// write. It caps only the bytes it reserves.
    pub fn new(length: u64, backing_pfn: u32, pixel_format: u32, plane_count: u8) -> Self {
        let reserve = (plane_count as usize).min(SURFACE_BACKING_PLANE_CAP);
        let mut bytes = [0u8; SURFACE_BACKING_BUILDER_CAP];
        bytes[0x00..0x08].copy_from_slice(&length.to_le_bytes());
        bytes[0x08..0x0c].copy_from_slice(&backing_pfn.to_le_bytes());
        bytes[0x0c..0x10].copy_from_slice(&pixel_format.to_le_bytes());
        bytes[0x10] = plane_count;
        Self {
            bytes,
            len: surface_backing_len_for(reserve),
        }
    }

    /// Write plane `index`. Ignores an index past [`SURFACE_BACKING_PLANE_CAP`], which
    /// has no record to write.
    pub fn plane(
        mut self,
        index: usize,
        offset: u32,
        width: u32,
        height: u32,
        bytes_per_row: u32,
        bytes_per_element: u8,
    ) -> Self {
        if index >= SURFACE_BACKING_PLANE_CAP {
            return self;
        }
        let base = SURFACE_BACKING_PLANES + index * SURFACE_BACKING_PLANE_STRIDE;
        let packed = (bytes_per_row & 0x00ff_ffff) | (u32::from(bytes_per_element) << 24);
        self.bytes[base..base + 4].copy_from_slice(&offset.to_le_bytes());
        self.bytes[base + 4..base + 8].copy_from_slice(&width.to_le_bytes());
        self.bytes[base + 8..base + 12].copy_from_slice(&height.to_le_bytes());
        self.bytes[base + 12..base + 16].copy_from_slice(&packed.to_le_bytes());
        self.len = self.len.max(base + SURFACE_BACKING_PLANE_STRIDE);
        self
    }

    /// Truncate or extend to an exact byte length, for the short-descriptor
    /// cases. Bytes past what was written are zero.
    pub fn with_len(mut self, len: usize) -> Self {
        self.len = len.min(SURFACE_BACKING_BUILDER_CAP);
        self
    }

    /// The assembled bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

/// Assemble a wire tag 5 descriptor by the format's own rules, for tests.
///
/// Same purpose as [`SurfaceBackingBuilder`]: the descriptor's three nested headers were
/// being written by hand at their offsets, which is the reader's own arithmetic
/// spelled a second time.
#[derive(Debug)]
pub struct IOSurfacePlaneViewBuilder {
    bytes: [u8; IOSURFACE_PLANE_VIEW_BUILDER_CAP],
    len: usize,
}

/// Longest descriptor [`IOSurfacePlaneViewBuilder`] assembles: through the plane index.
pub const IOSURFACE_PLANE_VIEW_BUILDER_CAP: usize =
    IOSURFACE_PLANE_VIEW_ARG_RECORD + IOSURFACE_PLANE_VIEW_RECORD_PLANE + 4;

impl IOSurfacePlaneViewBuilder {
    /// A descriptor naming `surface_id`, owned by `owner_task`, whose args
    /// blob carries a texture record of `tag`.
    ///
    /// The blob length is written as the args bytes actually present, which is
    /// what the live wire carries; use [`IOSurfacePlaneViewBuilder::with_len`] to produce a
    /// descriptor that disagrees with itself.
    pub fn new(surface_id: u32, owner_task: u32, own_ref: u32, tag: u8) -> Self {
        let mut bytes = [0u8; IOSURFACE_PLANE_VIEW_BUILDER_CAP];
        bytes[0x00..0x04].copy_from_slice(&surface_id.to_le_bytes());
        bytes[0x04..0x08].copy_from_slice(&owner_task.to_le_bytes());
        bytes[IOSURFACE_PLANE_VIEW_ARGS..IOSURFACE_PLANE_VIEW_ARGS + 4]
            .copy_from_slice(&0x2fu32.to_le_bytes());
        bytes[IOSURFACE_PLANE_VIEW_ARGS + 8..IOSURFACE_PLANE_VIEW_ARGS + 12]
            .copy_from_slice(&own_ref.to_le_bytes());
        bytes[IOSURFACE_PLANE_VIEW_ARG_RECORD] = tag;
        let len = IOSURFACE_PLANE_VIEW_ARG_RECORD + IOSURFACE_PLANE_VIEW_RECORD_MIN_LEN;
        let blob_len = (len - IOSURFACE_PLANE_VIEW_ARGS) as u32;
        bytes[IOSURFACE_PLANE_VIEW_ARGS + 4..IOSURFACE_PLANE_VIEW_ARGS + 8]
            .copy_from_slice(&blob_len.to_le_bytes());
        Self { bytes, len }
    }

    /// Write the record's geometry.
    pub fn geometry(mut self, pixel_format: u16, width: u32, height: u32, depth: u32) -> Self {
        let r = IOSURFACE_PLANE_VIEW_ARG_RECORD;
        self.bytes[r + 0x02..r + 0x04].copy_from_slice(&pixel_format.to_le_bytes());
        self.bytes[r + 0x04..r + 0x08].copy_from_slice(&width.to_le_bytes());
        self.bytes[r + 0x08..r + 0x0c].copy_from_slice(&height.to_le_bytes());
        self.bytes[r + 0x0c..r + 0x10].copy_from_slice(&depth.to_le_bytes());
        self
    }

    /// Write the record's undecoded byte at `+0x01`, for the same reason as
    /// [`IOSurfacePlaneViewBuilder::trailer`].
    pub fn unknown(mut self, unknown: u8) -> Self {
        self.bytes[IOSURFACE_PLANE_VIEW_ARG_RECORD + 0x01] = unknown;
        self
    }

    /// Write the record's trailer, `+0x10..+0x20`.
    ///
    /// Nothing decodes these. They are here so a test carrying bytes a live
    /// capture produced keeps carrying them — a fixture zeroed for convenience
    /// stops being able to catch a decoder that starts reading them.
    pub fn trailer(
        mut self,
        trailer: [u8; IOSURFACE_PLANE_VIEW_RECORD_PLANE - IOSURFACE_PLANE_VIEW_RECORD_MIN_LEN],
    ) -> Self {
        let at = IOSURFACE_PLANE_VIEW_ARG_RECORD + IOSURFACE_PLANE_VIEW_RECORD_MIN_LEN;
        self.bytes[at..at + trailer.len()].copy_from_slice(&trailer);
        self.len = self.len.max(at + trailer.len());
        self
    }

    /// Write the `newTextureWithDescriptor:iosurface:plane:` plane index,
    /// extending the descriptor to reach it.
    pub fn plane_index(mut self, plane: u32) -> Self {
        let at = IOSURFACE_PLANE_VIEW_ARG_RECORD + IOSURFACE_PLANE_VIEW_RECORD_PLANE;
        self.bytes[at..at + 4].copy_from_slice(&plane.to_le_bytes());
        self.len = self.len.max(at + 4);
        let blob_len = (self.len - IOSURFACE_PLANE_VIEW_ARGS) as u32;
        self.bytes[IOSURFACE_PLANE_VIEW_ARGS + 4..IOSURFACE_PLANE_VIEW_ARGS + 8]
            .copy_from_slice(&blob_len.to_le_bytes());
        self
    }

    /// Truncate or extend to an exact byte length.
    pub fn with_len(mut self, len: usize) -> Self {
        self.len = len.min(IOSURFACE_PLANE_VIEW_BUILDER_CAP);
        self
    }

    /// The assembled bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mapper_texture<const N: usize>(mapper_ref: u64, opcode: u32, nested_len: u32) -> [u8; N] {
        let mut bytes = [0u8; N];
        bytes[..8].copy_from_slice(&mapper_ref.to_le_bytes());
        bytes[8..12].copy_from_slice(&opcode.to_le_bytes());
        bytes[12..16].copy_from_slice(&nested_len.to_le_bytes());
        bytes
    }

    #[test]
    fn mapper_texture_keeps_the_complete_64_bit_identity_and_nested_variant() {
        let mapper_ref = 0x1122_3344_5566_7788;

        let mut legacy = mapper_texture::<56>(
            mapper_ref,
            backed_texture::OPCODE_IOSURFACE_TEXTURE,
            backed_texture::IOSURFACE_TEXTURE_TOTAL_LEN,
        );
        legacy[52..54].copy_from_slice(&3u16.to_le_bytes());
        legacy[54] = 0x55;
        legacy[55] = 0xaa;
        let view = mapper_iosurface_texture(&legacy).expect("legacy mapper texture");
        assert_eq!(view.mapper_ref.get(), mapper_ref);
        let MapperIOSurfaceTextureOperation::Legacy(body) = view.operation else {
            panic!("legacy tag selected another variant");
        };
        assert_eq!(body.plane.get(), 3);

        let mut rotated = mapper_texture::<56>(
            mapper_ref,
            backed_texture::OPCODE_IOSURFACE_TEXTURE_ROTATED,
            backed_texture::IOSURFACE_TEXTURE_ROTATED_TOTAL_LEN,
        );
        rotated[52..54].copy_from_slice(&5u16.to_le_bytes());
        rotated[54] = 7;
        rotated[55] = 0xff;
        let view = mapper_iosurface_texture(&rotated).expect("rotated mapper texture");
        let MapperIOSurfaceTextureOperation::Rotated(body) = view.operation else {
            panic!("rotated tag selected another variant");
        };
        assert_eq!(body.plane.get(), 5);
        assert_eq!(body.rotation, 7);

        let mut wide = mapper_texture::<64>(
            mapper_ref,
            backed_texture::OPCODE_IOSURFACE_TEXTURE_WIDE,
            backed_texture::IOSURFACE_TEXTURE_WIDE_TOTAL_LEN,
        );
        wide[60..62].copy_from_slice(&9u16.to_le_bytes());
        wide[62] = 11;
        wide[63] = 0xff;
        let view = mapper_iosurface_texture(&wide).expect("wide mapper texture");
        let MapperIOSurfaceTextureOperation::Wide(body) = view.operation else {
            panic!("wide tag selected another variant");
        };
        assert_eq!(body.plane.get(), 9);
        assert_eq!(body.rotation, 11);
    }

    #[test]
    fn mapper_texture_refuses_unknown_or_non_exhaustive_nested_records() {
        let unknown = mapper_texture::<56>(0x1_0000_0001, 0x58, 48);
        assert!(matches!(
            mapper_iosurface_texture(&unknown),
            Err(MapperIOSurfaceTextureError::UnknownVariant {
                opcode: 0x58,
                length: 48,
            })
        ));

        let trailing = mapper_texture::<57>(
            7,
            backed_texture::OPCODE_IOSURFACE_TEXTURE,
            backed_texture::IOSURFACE_TEXTURE_TOTAL_LEN,
        );
        assert!(matches!(
            mapper_iosurface_texture(&trailing),
            Err(MapperIOSurfaceTextureError::OuterLength {
                nested: 56,
                outer: 57,
            })
        ));
    }

    #[test]
    fn what_the_iosurface_plane_view_builder_writes_is_what_the_views_read() {
        let d =
            IOSurfacePlaneViewBuilder::new(77, 0, 4242, IOSURFACE_PLANE_VIEW_RECORD_TAG_COLOR_VIEW)
                .geometry(80, 1024, 768, 1)
                .plane_index(2);
        let desc = d.bytes();

        let h = iosurface_plane_view_header(desc).expect("header");
        assert_eq!((h.surface_id.get(), h.owner_task.get()), (77, 0));

        let args = view_at::<IOSurfacePlaneViewArgsHeader>(desc, IOSURFACE_PLANE_VIEW_ARGS)
            .expect("args header");
        assert_eq!(args.kind.get(), 0x2f);
        assert_eq!(args.own_ref.get(), 4242);
        assert_eq!(
            args.blob_len.get() as usize,
            desc.len() - IOSURFACE_PLANE_VIEW_ARGS,
            "the blob length is the args bytes present, which is what the wire carries"
        );

        let rec = iosurface_plane_view_texture_record(desc).expect("record");
        assert!(rec.tag_is_known());
        assert_eq!(
            (
                rec.pixel_format.get(),
                rec.width.get(),
                rec.height.get(),
                rec.depth.get()
            ),
            (80, 1024, 768, 1)
        );
        assert_eq!(iosurface_plane_view_record_plane_index(desc), Some(2));
    }

    /// The builder writes what the views read — the only thing that makes
    /// either of them worth anything.
    #[test]
    fn what_the_builder_writes_is_what_the_views_read() {
        let d = SurfaceBackingBuilder::new(0x8000, 0x1234, 0x4247_5241, 2)
            .plane(0, 0, 1920, 1080, 1920 * 4, 4)
            .plane(1, 0x1000, 960, 540, 960 * 2, 2);
        let desc = d.bytes();
        assert_eq!(desc.len(), surface_backing_len_for(2));

        let h = surface_backing_header(desc).expect("header");
        assert_eq!(h.length.get(), 0x8000);
        assert_eq!(h.backing_pfn.get(), 0x1234);
        assert_eq!(h.pixel_format.get(), 0x4247_5241);
        assert_eq!(h.plane_count, 2);
        assert_eq!(h.reserved, [0, 0, 0]);

        let p0 = surface_backing_plane(desc, 0).expect("plane 0");
        assert_eq!(
            (
                p0.offset.get(),
                p0.width.get(),
                p0.height.get(),
                p0.bytes_per_row(),
                p0.bytes_per_element()
            ),
            (0, 1920, 1080, 1920 * 4, 4)
        );
        let p1 = surface_backing_plane(desc, 1).expect("plane 1");
        assert_eq!(
            (
                p1.offset.get(),
                p1.width.get(),
                p1.height.get(),
                p1.bytes_per_row(),
                p1.bytes_per_element()
            ),
            (0x1000, 960, 540, 960 * 2, 2)
        );
        // Plane 2 was never reserved, so its record is off the end.
        assert!(surface_backing_plane(desc, 2).is_err());
    }

    /// A plane count the guest could not honestly mean is written as given.
    #[test]
    fn the_builder_writes_an_over_cap_plane_count_without_clamping_it() {
        let d = SurfaceBackingBuilder::new(0x1000, 0x100, 0x4247_5241, 12);
        assert_eq!(
            surface_backing_header(d.bytes())
                .expect("header")
                .plane_count,
            12
        );
        assert_eq!(
            d.bytes().len(),
            surface_backing_len_for(SURFACE_BACKING_PLANE_CAP),
            "the bytes stop at the cap even when the count does not"
        );
    }

    /// The offsets the device used to spell as literals are what the structs
    /// lay out.
    ///
    /// These are the eighteen numbers that lived in
    /// `reims_vgpu::runtime::objects` as `pub const`s. They are asserted here
    /// against `offset_of!` so that a field added or reordered above fails the
    /// build rather than silently moving a read — which is the whole reason
    /// these records were worth a view.
    #[test]
    fn the_layout_matches_the_offsets_the_device_contract_states() {
        use core::mem::offset_of;

        assert_eq!(offset_of!(SurfaceBackingHeader, length), 0x00);
        assert_eq!(offset_of!(SurfaceBackingHeader, backing_pfn), 0x08);
        assert_eq!(offset_of!(SurfaceBackingHeader, pixel_format), 0x0c);
        assert_eq!(offset_of!(SurfaceBackingHeader, plane_count), 0x10);
        assert_eq!(offset_of!(SurfaceBackingHeader, reserved), 0x11);
        assert_eq!(SURFACE_BACKING_PLANES, 0x14);
        assert_eq!(SURFACE_BACKING_PLANE_STRIDE, 0x10);
        assert_eq!(SURFACE_BACKING_MIN_LEN, 0x24);
        assert_eq!(offset_of!(SurfaceBackingPlaneRecord, offset), 0x00);
        assert_eq!(offset_of!(SurfaceBackingPlaneRecord, width), 0x04);
        assert_eq!(offset_of!(SurfaceBackingPlaneRecord, height), 0x08);
        assert_eq!(offset_of!(SurfaceBackingPlaneRecord, packed_bpr), 0x0c);

        assert_eq!(offset_of!(IOSurfacePlaneViewHeader, surface_id), 0x00);
        assert_eq!(offset_of!(IOSurfacePlaneViewHeader, owner_task), 0x04);
        assert_eq!(IOSURFACE_PLANE_VIEW_ARGS, 0x08);
        assert_eq!(offset_of!(IOSurfacePlaneViewArgsHeader, kind), 0x00);
        assert_eq!(offset_of!(IOSurfacePlaneViewArgsHeader, blob_len), 0x04);
        assert_eq!(offset_of!(IOSurfacePlaneViewArgsHeader, own_ref), 0x08);
        assert_eq!(IOSURFACE_PLANE_VIEW_ARG_RECORD, 0x14);
        assert_eq!(offset_of!(IOSurfacePlaneViewTextureRecord, tag), 0x00);
        assert_eq!(
            offset_of!(IOSurfacePlaneViewTextureRecord, pixel_format),
            0x02
        );
        assert_eq!(offset_of!(IOSurfacePlaneViewTextureRecord, width), 0x04);
        assert_eq!(offset_of!(IOSurfacePlaneViewTextureRecord, height), 0x08);
        assert_eq!(offset_of!(IOSurfacePlaneViewTextureRecord, depth), 0x0c);
        assert_eq!(IOSURFACE_PLANE_VIEW_RECORD_MIN_LEN, 0x10);
    }

    /// A descriptor one byte short of a record is refused, not read past.
    ///
    /// The device's own decoders each carried a hand-written length test per
    /// read. These are the same tests, once, in the constructor — and the
    /// interesting direction is the short one, because that is the direction a
    /// corrupt guest descriptor arrives from.
    #[test]
    fn a_short_descriptor_is_refused_rather_than_read_past() {
        let desc = [0u8; 0x40];
        assert!(surface_backing_header(&desc[..SURFACE_BACKING_PLANES]).is_ok());
        assert!(surface_backing_header(&desc[..SURFACE_BACKING_PLANES - 1]).is_err());
        assert!(surface_backing_plane(&desc[..SURFACE_BACKING_MIN_LEN], 0).is_ok());
        assert!(surface_backing_plane(&desc[..SURFACE_BACKING_MIN_LEN - 1], 0).is_err());
        assert!(surface_backing_plane(&desc, 1).is_ok());
        assert!(surface_backing_plane(&desc[..SURFACE_BACKING_MIN_LEN], 1).is_err());

        assert!(iosurface_plane_view_header(&desc[..IOSURFACE_PLANE_VIEW_ARGS]).is_ok());
        assert!(iosurface_plane_view_header(&desc[..IOSURFACE_PLANE_VIEW_ARGS - 1]).is_err());
        let record_end = IOSURFACE_PLANE_VIEW_ARG_RECORD + IOSURFACE_PLANE_VIEW_RECORD_MIN_LEN;
        assert!(iosurface_plane_view_texture_record(&desc[..record_end]).is_ok());
        assert!(iosurface_plane_view_texture_record(&desc[..record_end - 1]).is_err());

        // The plane index sits past the record, so a blob that stops at the
        // record has no plane to report and must say `None` rather than 0.
        assert_eq!(
            iosurface_plane_view_record_plane_index(&desc[..record_end]),
            None
        );
    }

    /// `bytes_per_row` and `bytes_per_element` share one wire word, and the
    /// split is 24/8 rather than any other partition.
    ///
    /// A row stride of 0x00ffffff is 16 MiB — larger than any surface this
    /// device has seen — so the top byte cannot be reclaimed for a wider
    /// stride, and an element size is one byte by every IOSurface reading.
    #[test]
    fn the_packed_plane_word_splits_at_twenty_four_bits() {
        let mut desc = [0u8; SURFACE_BACKING_MIN_LEN];
        desc[SURFACE_BACKING_PLANES + 0x0c..SURFACE_BACKING_PLANES + 0x10]
            .copy_from_slice(&0xab_123456u32.to_le_bytes());
        let plane = surface_backing_plane(&desc, 0).expect("one plane fits");
        assert_eq!(plane.bytes_per_row(), 0x123456);
        assert_eq!(plane.bytes_per_element(), 0xab);
    }

    #[test]
    fn only_the_two_observed_record_tags_are_known() {
        let mut desc = [0u8; IOSURFACE_PLANE_VIEW_ARG_RECORD + IOSURFACE_PLANE_VIEW_RECORD_MIN_LEN];
        for (tag, known) in [
            (IOSURFACE_PLANE_VIEW_RECORD_TAG_PLANE, true),
            (IOSURFACE_PLANE_VIEW_RECORD_TAG_COLOR_VIEW, true),
            (0x00, false),
            (0x43, false),
            (0xff, false),
        ] {
            desc[IOSURFACE_PLANE_VIEW_ARG_RECORD] = tag;
            assert_eq!(
                iosurface_plane_view_texture_record(&desc)
                    .expect("full record")
                    .tag_is_known(),
                known,
                "tag {tag:#x}"
            );
        }
    }

    /// The declared length of a wire tag 4 descriptor is its header plus its
    /// planes, and nothing checked that against the stride before.
    #[test]
    fn a_descriptors_length_is_its_header_plus_its_planes() {
        assert_eq!(surface_backing_len_for(0), SURFACE_BACKING_PLANES);
        assert_eq!(surface_backing_len_for(1), SURFACE_BACKING_MIN_LEN);
        // The two shapes the live census produced: a single-plane 'BGRA'
        // surface at 36 bytes and a biplanar '420f' one at 52.
        assert_eq!(surface_backing_len_for(1), 36);
        assert_eq!(surface_backing_len_for(2), 52);
        assert_eq!(
            surface_backing_len_for(SURFACE_BACKING_PLANE_CAP),
            SURFACE_BACKING_PLANES + 8 * 0x10
        );
    }
}
