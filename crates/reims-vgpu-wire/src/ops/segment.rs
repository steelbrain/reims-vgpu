//! The segment header every encoder writes around its records.
//!
//! A command stream is a sequence of *segments*, and each segment is this
//! 8-byte header followed by the records one encoder produced. It is not an
//! operation — it has no opcode and does not go through
//! `-getCommandBufferBytes:` as a command — so it lives here rather than in a
//! per-family module, and no manifest row claims an opcode for it.
//!
//! # It is written twice
//!
//! `-beginSegment:protectionOptions:` allocates the eight bytes and writes the
//! type, whether this segment continues the preceding one, and an initially
//! clear forward-continuation byte. `-endEncoding` comes back afterwards and
//! fills in [`SegmentHeader::length`], which is why a capture taken between the
//! two reads `length == 0` — the fixtures here are exactly that capture. A
//! reader who takes those bytes as the finished header will conclude the length
//! field is always zero.
//!
//! # `segment_type` is derived, not assigned
//!
//! The byte at `+4` is a *type* because the same call on four different
//! encoder classes puts four different values there: `render_begin_segment`
//! writes `0`, `compute_begin_segment` writes `1`, `blit_begin_segment` writes
//! `2` and `info_begin_segment` writes `4`, from identical arguments. That is
//! the whole derivation, and it is why all four fixtures exist rather than one.
//!
//! Those values are also exactly `reims_vgpu::runtime::decode::stream`'s
//! `SEGMENT_TYPE_RENDER = 0`, `SEGMENT_TYPE_COMPUTE = 1`,
//! `SEGMENT_TYPE_BLIT = 2` and `SEGMENT_TYPE_INFO = 4`, and that module's
//! `SEGMENT_TYPE_OFFSET` is 4 and `SEGMENT_HEADER_LEN` is 8 — two independent
//! derivations agreeing on the whole header. The info value is the load-bearing
//! one: it is not the next number in sequence, so a device that had guessed
//! rather than derived would have written `3` there.
//!
//! Four of its six types have been observed here. The event and
//! protection-options encoders have not been driven, so this module names the
//! four it measured and nothing else — in particular it does not name `3`,
//! which the device calls `SEGMENT_TYPE_EVENT` and which nothing here has seen.

use crate::le::U32le;
use crate::view::{view, Wire, WireError};

/// Bytes the serializer allocates for a segment header.
pub const SEGMENT_HEADER_LEN: usize = 8;

/// The segment type a `PGSerializerRenderCommandEncoder` writes.
///
/// Observed, not assigned: fixture `render_begin_segment`.
pub const SEGMENT_TYPE_RENDER: u8 = 0;

/// The segment type a `PGSerializerComputeCommandEncoder` writes.
///
/// Observed, not assigned: fixture `compute_begin_segment`.
pub const SEGMENT_TYPE_COMPUTE: u8 = 1;

/// The segment type a `PGSerializerBlitCommandEncoder` writes.
///
/// Observed, not assigned: fixtures `blit_begin_segment` and
/// `blit_begin_segment_alt`.
pub const SEGMENT_TYPE_BLIT: u8 = 2;

/// The segment type a `PGSerializerInfoCommandEncoder` writes.
///
/// Observed, not assigned: fixture `info_begin_segment`. It skips `3`, which
/// belongs to an encoder class this crate has not driven.
pub const SEGMENT_TYPE_INFO: u8 = 4;

/// The segment type that introduces a protection-options envelope.
///
/// Observed, not assigned: fixtures `blit_begin_segment_protected` and
/// `..._alt`. `reims_vgpu::runtime::decode::stream` has carried this number and
/// an `Envelope` disposition for it all along, with nothing to confirm it; these
/// fixtures are that confirmation.
pub const SEGMENT_TYPE_PROTECTION_OPTIONS: u8 = 5;

/// Bytes of the record that follows a [`SEGMENT_TYPE_PROTECTION_OPTIONS`]
/// header.
pub const PROTECTION_OPTIONS_ENVELOPE_LEN: usize = 8;

/// The envelope's payload: the `protectionOptions:` argument, verbatim.
///
/// Eight bytes, all written — unlike the segment header beside it, which leaves
/// its eighth alone. `blit_begin_segment_protected` passes `0x44` and
/// `..._alt` passes `0x33`, so this is the guest's value and not a constant.
///
/// This is not an operation and has no opcode; it is the middle of a
/// three-record burst and is identified by the header that precedes it.
#[repr(C)]
#[derive(Debug)]
pub struct ProtectionOptionsEnvelope {
    pub protection_options: crate::le::U64le,
}

// SAFETY: one align-1 all-bytes-valid `le` scalar.
unsafe impl Wire for ProtectionOptionsEnvelope {}

/// View a protection-options envelope payload at the start of `buf`.
pub fn protection_options_envelope(buf: &[u8]) -> Result<&ProtectionOptionsEnvelope, WireError> {
    view::<ProtectionOptionsEnvelope>(buf)
}

/// The seven meaningful bytes of the eight-byte segment-header allocation.
///
/// The eighth is never written, so on a real wire it holds
/// whatever the ring last contained. It is deliberately not a field.
#[repr(C)]
#[derive(Debug)]
pub struct SegmentHeader {
    /// Total segment length, header included. **Zero until `-endEncoding`**,
    /// which is the state every fixture here captures.
    pub length: U32le,
    /// Which encoder wrote this segment. See [`SEGMENT_TYPE_RENDER`],
    /// [`SEGMENT_TYPE_COMPUTE`], [`SEGMENT_TYPE_BLIT`] and
    /// [`SEGMENT_TYPE_INFO`]; the raw byte is kept rather than an enum, because
    /// a guest may put any value here.
    pub segment_type: u8,
    /// Non-zero when this segment continues the encoder left open by the
    /// preceding segment. This is the `BOOL` argument to
    /// `-beginSegment:protectionOptions:` verbatim.
    pub continues_previous: u8,
    /// Non-zero when the following segment continues this segment's encoder.
    ///
    /// Header construction initializes this byte to zero. Beginning a
    /// continuation marks it in the preceding header, producing the paired
    /// relation `(continues_next, continues_previous)` across the two headers.
    pub continues_next: u8,
}

// SAFETY: an align-1 `le` scalar and three `u8`s; every byte pattern is valid
// and no field needs alignment.
unsafe impl Wire for SegmentHeader {}

/// View a segment header at the start of `buf`.
///
/// Takes bytes rather than an [`crate::op::Op`], because a segment header is
/// not an operation and has no opcode to dispatch on.
pub fn segment_header(buf: &[u8]) -> Result<&SegmentHeader, WireError> {
    view::<SegmentHeader>(buf)
}

impl SegmentHeader {
    /// Whether this segment continues an encoder opened by its predecessor.
    pub fn continues_previous(&self) -> bool {
        self.continues_previous != 0
    }

    /// Whether this segment leaves its encoder open for its successor.
    pub fn continues_next(&self) -> bool {
        self.continues_next != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    /// The body is one byte short of the allocation, and that byte is not a
    /// field. Widening it to fill the header would make the view read whatever
    /// the guest's ring last held.
    #[test]
    fn the_last_byte_of_the_header_belongs_to_no_field() {
        assert_eq!(size_of::<SegmentHeader>() + 1, SEGMENT_HEADER_LEN);
    }

    /// The measured types are distinct. If a future capture made two of them
    /// equal, the `+4` byte would no longer be shown to be a type at all — the
    /// difference between classes is the entire derivation.
    #[test]
    fn the_measured_segment_types_all_differ() {
        let types = [
            SEGMENT_TYPE_RENDER,
            SEGMENT_TYPE_COMPUTE,
            SEGMENT_TYPE_BLIT,
            SEGMENT_TYPE_INFO,
        ];
        for (i, a) in types.iter().enumerate() {
            for b in &types[i + 1..] {
                assert_ne!(a, b, "two encoder classes claim the same segment type");
            }
        }
    }

    #[test]
    fn a_header_one_byte_short_is_refused_rather_than_read() {
        let buf = [0u8; SEGMENT_HEADER_LEN - 2];
        assert!(matches!(segment_header(&buf), Err(WireError::Short { .. })));
    }

    /// The four fields occupy four distinct byte offsets in the order declared.
    ///
    /// Synthesized from the crate's own constants, so it proves the struct is
    /// self-consistent and *cannot* prove the offsets are Apple's. That is what
    /// `every_segment_header_fixture_reads_back_what_the_encoder_wrote` is for;
    /// no Apple bytes belong in this file.
    #[test]
    fn each_field_reads_its_own_byte() {
        let mut bytes = [0u8; SEGMENT_HEADER_LEN];
        bytes[..4].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        bytes[4] = 0x11;
        bytes[5] = 0x22;
        bytes[6] = 0x33;
        bytes[7] = 0xFF; // past the body; no field may see it
        let h = segment_header(&bytes).expect("fits");
        assert_eq!(h.length.get(), 0x1234_5678);
        assert_eq!(h.segment_type, 0x11);
        assert_eq!(h.continues_previous, 0x22);
        assert_eq!(h.continues_next, 0x33);
        assert!(h.continues_previous());
        assert!(h.continues_next());
    }
}
