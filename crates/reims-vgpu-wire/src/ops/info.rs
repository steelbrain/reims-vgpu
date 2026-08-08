//! Info encoder records.
//!
//! The records a `PGSerializerInfoCommandEncoder` writes. Derived by calling
//! the Metal method with distinctive arguments and reading the bytes back; the
//! fixture that pins each is named in its doc. See `oracle/oracle.m`'s
//! `infoCases`.
//!
//! # These are questions, not commands
//!
//! Every other encoder class tells the device to do something. This one asks:
//! each selector takes an object and a **pointer the answer is written into**,
//! and the record has to say where that is on the far side of the wire.
//!
//! It says so the same way `setVertexBytes:` does. The encoder asks its command
//! stream for scratch through `-getBufferBytes:alignment:buffer:offset:`, and
//! records the buffer and offset the stream handed back — so every record here
//! ends with a `(reply_buffer_ref, reply_offset)` pair naming where the reply
//! goes. That is the whole shape of the class.
//!
//! **What the oracle cannot settle about that pair.** In a capture the stream
//! *is* the oracle, so `reply_offset` is whatever `CaptureCommandStream`
//! returned. A guest's real stream may hand back an offset within a buffer or
//! something else entirely; nothing here can tell.
//! `reims_vgpu::runtime::icb` calls the same field `gpu_address`, and this
//! module deliberately does not take a position on which name is right — it
//! names what the field *is on the wire*, the second half of the pair the
//! stream returned.
//!
//! # Opcodes
//!
//! `0x1c2`–`0x1d4`, a fourth space, above the blit encoder's. `0x1d1`
//! `icbHostResourceInfo:info:` is the one opcode
//! `reims_vgpu::runtime::icb::INFO_OP_ICB_HOST_RESOURCE` decodes, and its
//! record length `0x18` and its three field offsets all agree with what Apple
//! writes.

use crate::le::{F32le, U32le, U64le};
use crate::op::Op;
use crate::view::{view, Wire, WireError};

// --- The query family ------------------------------------------------------

pub const OPCODE_COMPUTE_PIPELINE_STATE_INFO: u32 = 0x1c2;
pub const OPCODE_RENDER_PIPELINE_STATE_INFO: u32 = 0x1c9;
pub const OPCODE_BUFFER_HOST_RESOURCE_INFO: u32 = 0x1cd;
pub const OPCODE_TEXTURE_HOST_RESOURCE_INFO: u32 = 0x1ce;
pub const OPCODE_HEAP_HOST_RESOURCE_INFO: u32 = 0x1cf;
pub const OPCODE_SAMPLER_HOST_RESOURCE_INFO: u32 = 0x1d0;
pub const OPCODE_ICB_HOST_RESOURCE_INFO: u32 = 0x1d1;
pub const OPCODE_RENDER_PIPELINE_HOST_RESOURCE_INFO: u32 = 0x1d2;
pub const OPCODE_COMPUTE_PIPELINE_HOST_RESOURCE_INFO: u32 = 0x1d3;
pub const OPCODE_DEPTH_STENCIL_HOST_RESOURCE_INFO: u32 = 0x1d4;
pub const QUERY_TOTAL_LEN: u32 = 24;

/// One object, and where its answer goes.
///
/// Ten selectors write this identical record and differ only in opcode. Which
/// kind of object `object_ref` names comes from the opcode and from nowhere
/// else, which is why each fixture uses a *different* stub: `icb` 7171,
/// `buffer` 5151, `texture` 4242, `heap` 6565, `sampler` 6363, `depth-stencil`
/// 6262 and `pipeline` 6161 all land in the same field at the same offset under
/// different opcodes.
///
/// `reims_vgpu::runtime::icb` decodes the `0x1d1` case at exactly these three
/// offsets and this length — **and reads the last two as something else.** It
/// calls them `buffer_ref` ("the type-1 object-list ref of the ICB command
/// backing buffer") and `gpu_address` ("guest GPU/VA of the backing"), then
/// binds them as the ICB's command memory. Agreeing on where a field is and
/// disagreeing on what it means is the drift this crate exists to catch, and it
/// went unnoticed because the sentence above only ever claimed the offsets.
///
/// The fixtures settle it against that reading, three ways:
///
/// * `reply_buffer_ref` reads **8181** in *every one* of the ten query fixtures,
///   including `info_buffer_host_resource` — where the queried object **is** a
///   buffer, at ref 5151, sitting in `object_ref`. A field that stays 8181 while
///   the object's own ref is 5151 is not that object's backing buffer.
/// * 8181 and `reply_offset`'s `0x9999` are [`STUB_STAGING_REF`-shaped]: they
///   are returned only by the capture stream's
///   `-getBufferBytes:alignment:buffer:offset:` scratch allocator, from an
///   out-parameter Apple's own selector spells `offset:`.
/// * `0x1c5` (`copyRasterizationRateParameterBuffer`) writes the identical
///   24-byte shape with a *different* pair — the caller's own buffer and
///   offset — so these two slots are a (buffer, offset) pair that varies with
///   the caller, not a resource identity.
///
/// The type encoding says the same thing before any byte is read:
/// `icbHostResourceInfo:info:` is `v32@0:8@16^{?=QQ}24`, so `info:` is a
/// **pointer to two `u64` out-params** — the guest is asking the host to write
/// two words, and the record names where. This is the `getTileDimensions:`
/// shape, not an announcement.
///
/// `reply_offset` is attributed rather than perturbed: no case moves it, because
/// the stub allocator returns one constant. That is short of this crate's
/// perturbation bar and the honest upgrade is a second staging offset in the
/// oracle. It is well short of licensing the device's reading, which has no
/// derivation at all — `git log -S gpu_address` on that file reaches the initial
/// import, `AppleParavirtBuffer._gpuAddress` appears nowhere else in the
/// repository, and the ICB stub ships no `gpuAddress` accessor for the
/// serializer to have called.
///
/// Corrected in `reims-vgpu`: `apply_icb_host_resource_info` now refuses by
/// name rather than binding the reply pair as an ICB's command memory, and
/// `IcbHostResourceInfo` carries these three field names pinned to these three
/// offsets. `PGSerializerInfoCommandEncoder` is still in the divergence
/// instrument's `UNCOVERED_CLASSES`, which is why the one class where the two
/// crates provably disagreed is the one it did not check — the disagreement was
/// found by reading, not by the instrument.
///
/// [`STUB_STAGING_REF`-shaped]: the oracle's `encoder.h` defines both constants.
#[repr(C)]
#[derive(Debug)]
pub struct Query {
    pub object_ref: U32le,
    pub reply_buffer_ref: U32le,
    pub reply_offset: U64le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for Query {}

#[inline]
pub fn is_query(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_COMPUTE_PIPELINE_STATE_INFO
            | OPCODE_RENDER_PIPELINE_STATE_INFO
            | OPCODE_BUFFER_HOST_RESOURCE_INFO
            | OPCODE_TEXTURE_HOST_RESOURCE_INFO
            | OPCODE_HEAP_HOST_RESOURCE_INFO
            | OPCODE_SAMPLER_HOST_RESOURCE_INFO
            | OPCODE_ICB_HOST_RESOURCE_INFO
            | OPCODE_RENDER_PIPELINE_HOST_RESOURCE_INFO
            | OPCODE_COMPUTE_PIPELINE_HOST_RESOURCE_INFO
            | OPCODE_DEPTH_STENCIL_HOST_RESOURCE_INFO
    )
}

pub fn query<'a>(op: &Op<'a>) -> Result<&'a Query, WireError> {
    debug_assert!(is_query(op.opcode()));
    view::<Query>(op.payload)
}

// --- 0x1c3 heapTextureDescriptorSizeAndAlign:sizeAndAlign: -----------------

pub const OPCODE_HEAP_TEXTURE_DESCRIPTOR_SIZE_AND_ALIGN: u32 = 0x1c3;
pub const HEAP_TEXTURE_DESCRIPTOR_SIZE_AND_ALIGN_TOTAL_LEN: u32 = 52;

/// A texture descriptor, and where its size and alignment go.
///
/// The one selector on this class whose first argument is not a resource. Its
/// name says "heap" and the class it lives on is a *query* class, which is what
/// made the oracle hand it a heap stub for a while — that faulted, landed on
/// `crashed`, and so measured nothing at all. The `@` in
/// `v32@0:8@16^{?=QQ}24` is an `MTLTextureDescriptor`.
///
/// The payload is [`crate::ops::texture::TextureDescriptorBody`] verbatim, the
/// same 32 bytes the creation record carries after its ref and the same
/// [`crate::ops::heap_texture`] embeds — a third reader of one struct, and the
/// reason the struct is declared apart from any of them. `packed` bit 7 is
/// unwritten here exactly as it is there.
///
/// The out-parameter is `^{?=QQ}`, a size and an alignment, which is why the
/// answer needs a reply pair and the record is not simply the creation record
/// minus its ref — that is `0x16` on `PGSerializer`, which returns through a
/// different channel. Fixture `info_heap_texture_descriptor_size_and_align`.
#[repr(C)]
#[derive(Debug)]
pub struct HeapTextureDescriptorSizeAndAlign {
    pub descriptor: crate::ops::texture::TextureDescriptorBody,
    pub reply_buffer_ref: U32le,
    pub reply_offset: U64le,
}

// SAFETY: an align-1 `Wire` struct and two align-1 all-bytes-valid `le`
// scalars.
unsafe impl Wire for HeapTextureDescriptorSizeAndAlign {}

pub fn heap_texture_descriptor_size_and_align<'a>(
    op: &Op<'a>,
) -> Result<&'a HeapTextureDescriptorSizeAndAlign, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_HEAP_TEXTURE_DESCRIPTOR_SIZE_AND_ALIGN);
    view::<HeapTextureDescriptorSizeAndAlign>(op.payload)
}

/// The same query under `-setSupportsTextureDescriptor2:`.
pub const OPCODE_HEAP_TEXTURE_DESCRIPTOR_SIZE_AND_ALIGN_WIDE: u32 = 0x1d5;
pub const HEAP_TEXTURE_DESCRIPTOR_SIZE_AND_ALIGN_WIDE_TOTAL_LEN: u32 = 60;

/// The wide descriptor and where its size and alignment go.
///
/// The fifth and last record the wide descriptor reaches, and the only one of
/// the five that is a query rather than a creation — which is worth stating
/// because the reply pair sits *after* the descriptor here, so the wide body's
/// unwritten fortieth byte is in the middle of the record rather than at its
/// end. A reader that trimmed the record to its written extent would cut the
/// reply. Fixture `info_heap_texture_descriptor_size_and_align_wide`.
#[repr(C)]
#[derive(Debug)]
pub struct HeapTextureDescriptorSizeAndAlignWide {
    pub descriptor: crate::ops::texture::WideTextureDescriptorBody,
    pub reply_buffer_ref: U32le,
    pub reply_offset: U64le,
}

// SAFETY: an align-1 `Wire` struct and two align-1 all-bytes-valid `le`
// scalars.
unsafe impl Wire for HeapTextureDescriptorSizeAndAlignWide {}

pub fn heap_texture_descriptor_size_and_align_wide<'a>(
    op: &Op<'a>,
) -> Result<&'a HeapTextureDescriptorSizeAndAlignWide, WireError> {
    debug_assert_eq!(
        op.opcode(),
        OPCODE_HEAP_TEXTURE_DESCRIPTOR_SIZE_AND_ALIGN_WIDE
    );
    view::<HeapTextureDescriptorSizeAndAlignWide>(op.payload)
}

// --- 0x1ca / 0x1cb the imageblock queries ----------------------------------

pub const OPCODE_RENDER_PIPELINE_IMAGEBLOCK: u32 = 0x1ca;
pub const OPCODE_COMPUTE_PIPELINE_IMAGEBLOCK: u32 = 0x1cb;
pub const IMAGEBLOCK_TOTAL_LEN: u32 = 48;

/// A pipeline, an imageblock size, and where the answer goes.
///
/// The reply pair moves to the tail here because the size sits between the
/// object and it, in the selector's own argument order. Fixtures
/// `info_render_pipeline_imageblock` and `info_compute_pipeline_imageblock`
/// (`0x11`/`0x22`/`0x33`, three distinct values).
#[repr(C)]
#[derive(Debug)]
pub struct ImageblockQuery {
    pub pipeline_ref: U32le,
    pub width: U64le,
    pub height: U64le,
    pub depth: U64le,
    pub reply_buffer_ref: U32le,
    pub reply_offset: U64le,
}

// SAFETY: six align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for ImageblockQuery {}

#[inline]
pub fn is_imageblock_query(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_RENDER_PIPELINE_IMAGEBLOCK | OPCODE_COMPUTE_PIPELINE_IMAGEBLOCK
    )
}

pub fn imageblock_query<'a>(op: &Op<'a>) -> Result<&'a ImageblockQuery, WireError> {
    debug_assert!(is_imageblock_query(op.opcode()));
    view::<ImageblockQuery>(op.payload)
}

// --- 0x1c4 getRasterizationRateMapInfo:layerCount:info: --------------------

pub const OPCODE_RATE_MAP_INFO: u32 = 0x1c4;
pub const RATE_MAP_INFO_TOTAL_LEN: u32 = 32;

/// The rate-map query, whose reply is variable length.
///
/// **`layerCount` does not reach the wire.** What does is the reply's byte
/// length, derived from it: `layerCount = 2` wrote 28 and `layerCount = 5`
/// wrote 40 (`info_get_rasterization_rate_map` and its `_alt`), which is
/// `20 + 4 * layers` — and that is exactly the size of the `info:` struct the
/// type encoding declares, `{?={?=QQ}{?=SS}[0{?=SS}]}`: sixteen bytes, then
/// four, then four per layer. The perturbation and the type encoding derive the
/// same arithmetic independently.
///
/// So a decoder must recover the layer count from this field rather than
/// looking for it, and a reply buffer shorter than `reply_len` is a guest bug
/// this record can detect.
#[repr(C)]
#[derive(Debug)]
pub struct RateMapInfoQuery {
    pub rate_map_ref: U32le,
    pub reply_buffer_ref: U32le,
    pub reply_offset: U64le,
    /// Bytes the reply needs: `20 + 4 * layerCount`.
    pub reply_len: U32le,
    /// Written, `0` under both layer counts captured, and not identified.
    ///
    /// **Tried:** two layer counts, which moved `reply_len` and left this at 0.
    ///
    /// **What would settle it:** a rate map with more than one *physical* size
    /// — the reply struct's second member is a pair of `unsigned short`, and
    /// nothing in these two cases varies it.
    pub unidentified_u32: U32le,
}

// SAFETY: five align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for RateMapInfoQuery {}

pub fn rate_map_info<'a>(op: &Op<'a>) -> Result<&'a RateMapInfoQuery, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_RATE_MAP_INFO);
    view::<RateMapInfoQuery>(op.payload)
}

/// Layers a reply of `reply_len` bytes holds, or `None` if the length is not
/// one this record can produce.
///
/// The inverse of the arithmetic above, kept beside it so the two cannot drift.
/// Fallible because `reply_len` is guest-controlled: a length below the fixed
/// twenty bytes, or one that is not a whole number of layers past it, did not
/// come from this serializer.
#[inline]
pub fn rate_map_layer_count(reply_len: u32) -> Option<u32> {
    const FIXED: u32 = 20;
    const PER_LAYER: u32 = 4;
    let tail = reply_len.checked_sub(FIXED)?;
    if !tail.is_multiple_of(PER_LAYER) {
        return None;
    }
    Some(tail / PER_LAYER)
}

// --- 0x1c5 copyRasterizationRateParameterBuffer:buffer:bufferOffset: -------

pub const OPCODE_COPY_RATE_PARAMETER_BUFFER: u32 = 0x1c5;
pub const COPY_RATE_PARAMETER_BUFFER_TOTAL_LEN: u32 = 24;

/// The one selector on this class that is a command rather than a question.
///
/// Its record is byte-identical to [`Query`] and means something different: the
/// second and third fields are a real destination buffer and a real offset the
/// caller chose, not scratch the stream handed back. Fixture
/// `info_copy_rasterization_rate_parameter_buffer` (rate map 6767, buffer 5151
/// at `0x1111` — an offset the *case* chose, where every query in this module
/// carries `0x9999`, the stream's).
///
/// That difference is only visible in the opcode, which is why this has its own
/// view rather than sharing [`Query`]'s.
#[repr(C)]
#[derive(Debug)]
pub struct CopyRateParameterBuffer {
    pub rate_map_ref: U32le,
    pub buffer_ref: U32le,
    pub buffer_offset: U64le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for CopyRateParameterBuffer {}

pub fn copy_rate_parameter_buffer<'a>(
    op: &Op<'a>,
) -> Result<&'a CopyRateParameterBuffer, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_COPY_RATE_PARAMETER_BUFFER);
    view::<CopyRateParameterBuffer>(op.payload)
}

// --- 0x1c6 / 0x1c7 the coordinate mappers ----------------------------------

pub const OPCODE_MAP_SCREEN_TO_PHYSICAL: u32 = 0x1c6;
pub const OPCODE_MAP_PHYSICAL_TO_SCREEN: u32 = 0x1c7;
pub const MAP_COORDINATE_TOTAL_LEN: u32 = 36;

/// Map one coordinate through a rasterization rate map.
///
/// Fixtures `info_map_screen_to_physical` (layer 3, `0.25`/`0.75`) and
/// `info_map_physical_to_screen` (layer 4, `0.125`/`0.875`) — different layers
/// and different coordinates, so the two records cannot be confused and neither
/// field pair can be swapped unseen.
///
/// **`mapCoordinateInternal:…command:` writes this same record at an opcode of
/// the caller's choosing.** Passing `0x77` produced opcode `0x77` and passing
/// `0x55` produced `0x55` (`info_map_coordinate_internal` and its `_alt`), so
/// the two fixed-opcode selectors above are wrappers over it — the third place
/// in this protocol where a `command:` argument is the opcode, after the blit
/// encoder's `optimize:withCommand:` family. Its manifest row carries no
/// opcode; see [`crate::manifest::Coverage::CoveredNoFixedOpcode`].
///
/// The coordinates are 32-bit floats, matching the selector's `{?=ff}`.
#[repr(C)]
#[derive(Debug)]
pub struct MapCoordinate {
    pub rate_map_ref: U32le,
    pub reply_buffer_ref: U32le,
    pub reply_offset: U64le,
    pub layer: U32le,
    pub x: F32le,
    pub y: F32le,
}

// SAFETY: six align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for MapCoordinate {}

#[inline]
pub fn is_map_coordinate(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_MAP_SCREEN_TO_PHYSICAL | OPCODE_MAP_PHYSICAL_TO_SCREEN
    )
}

/// Read a coordinate-mapping record.
///
/// Takes no opcode assertion, because `mapCoordinateInternal:…command:` writes
/// this layout under an opcode the guest chose and a caller who knows that is
/// entitled to read it.
pub fn map_coordinate<'a>(op: &Op<'a>) -> Result<&'a MapCoordinate, WireError> {
    view::<MapCoordinate>(op.payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::OP_HEADER_LEN;
    use core::mem::size_of;

    #[test]
    fn each_record_is_its_body_plus_the_header() {
        for (name, body, total) in [
            ("Query", size_of::<Query>(), QUERY_TOTAL_LEN),
            (
                "ImageblockQuery",
                size_of::<ImageblockQuery>(),
                IMAGEBLOCK_TOTAL_LEN,
            ),
            (
                "RateMapInfoQuery",
                size_of::<RateMapInfoQuery>(),
                RATE_MAP_INFO_TOTAL_LEN,
            ),
            (
                "CopyRateParameterBuffer",
                size_of::<CopyRateParameterBuffer>(),
                COPY_RATE_PARAMETER_BUFFER_TOTAL_LEN,
            ),
            (
                "MapCoordinate",
                size_of::<MapCoordinate>(),
                MAP_COORDINATE_TOTAL_LEN,
            ),
        ] {
            assert_eq!(
                body + OP_HEADER_LEN,
                total as usize,
                "{name}: body {body} + header does not make {total}"
            );
        }
    }

    /// The rate-map reply length and the layer count are inverses, and the
    /// inverse refuses everything the forward direction cannot produce.
    #[test]
    fn the_rate_map_reply_length_inverts_to_a_layer_count() {
        for layers in [0u32, 1, 2, 5, 64] {
            let len = 20 + 4 * layers;
            assert_eq!(rate_map_layer_count(len), Some(layers), "{layers} layers");
        }
        // Below the fixed part, and not a whole number of layers past it.
        assert_eq!(rate_map_layer_count(0), None);
        assert_eq!(rate_map_layer_count(19), None);
        assert_eq!(rate_map_layer_count(21), None);
        assert_eq!(rate_map_layer_count(u32::MAX), None);
    }

    /// The two records that share a byte layout must not share a predicate: one
    /// is a question whose reply pair the *stream* chose and the other is a
    /// command whose buffer the *guest* chose, and only the opcode says which.
    #[test]
    fn the_query_and_the_copy_command_are_the_same_size_and_not_the_same_record() {
        assert_eq!(
            size_of::<Query>(),
            size_of::<CopyRateParameterBuffer>(),
            "these two are byte-identical on the wire, which is the point"
        );
        assert!(!is_query(OPCODE_COPY_RATE_PARAMETER_BUFFER));
    }

    #[test]
    fn no_info_opcode_answers_two_shape_predicates() {
        for opcode in 0x1c0u32..=0x1e0 {
            let hits = [
                is_query(opcode),
                is_imageblock_query(opcode),
                is_map_coordinate(opcode),
                opcode == OPCODE_RATE_MAP_INFO,
                opcode == OPCODE_COPY_RATE_PARAMETER_BUFFER,
            ]
            .into_iter()
            .filter(|hit| *hit)
            .count();
            assert!(
                hits <= 1,
                "opcode {opcode:#x} answers {hits} shape predicates"
            );
        }
    }

    #[test]
    fn a_short_payload_is_refused_rather_than_read() {
        let buf = [0u8; size_of::<Query>() - 1];
        assert!(matches!(view::<Query>(&buf), Err(WireError::Short { .. })));
        let buf = [0u8; size_of::<MapCoordinate>() - 1];
        assert!(matches!(
            view::<MapCoordinate>(&buf),
            Err(WireError::Short { .. })
        ));
    }
}
