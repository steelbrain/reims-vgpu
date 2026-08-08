//! Object destruction records.
//!
//! `PGSerializer` ships three families of eleven, one entry per object kind,
//! and only one of the three reaches the wire:
//!
//! | Selector | On the wire |
//! |---|---|
//! | `-newXRef` | nothing; it allocates a ref host-side |
//! | `-releaseXRef:` | nothing; host-side accounting |
//! | `-deleteXRef:allocator:` | this record |
//!
//! The allocator argument is the tell — a selector that takes one is a selector
//! that is about to ask for operation bytes — but the two silent families are
//! findings in their own right and are recorded as such. `oracle.m`'s
//! `lifecycleCases` drives all thirty-three every capture, so a future build
//! that starts emitting from `-releaseXRef:` fails the suite rather than
//! quietly losing the guest's release.
//!
//! # A fifth opcode space
//!
//! `0x3e8`–`0x3f7`, which is 1000–1015 in decimal and sits far above the four
//! encoder spaces (render `0x00`–`0x89`, compute `0xc8`–`0xe5`, blit
//! `0x12c`–`0x13e`, info `0x1c2`–`0x1d4`). Creation lives at the *bottom* of the
//! numbering — [`crate::ops::texture`] is opcode 1 — so creation and
//! destruction are at opposite ends of the range rather than adjacent.
//!
//! Five of the sixteen numbers in the span are not claimed by any of the eleven
//! selectors: `0x3ec`, `0x3f0`, `0x3f2`, `0x3f3` and `0x3f5`. Nothing here names
//! them. They may belong to object kinds with no `-deleteXRef:` selector, or to
//! operations that are not destruction at all; this module does not guess, and a
//! decoder that meets one is meeting something unmeasured.
//!
//! # The device does not decode any of them
//!
//! `reims_vgpu::runtime::decode` has no arm anywhere in this range. The device's
//! object teardown runs at the FIFO layer instead (`CHILD_OP_DELETE_RESOURCE`),
//! which is the kernel's protocol rather than the serializer's — so these are
//! records the guest emits into the command stream that nothing on the host
//! reads. Whether that matters is a question for a driven boot, not for this
//! crate; what this module does is make the record legible so the question can
//! be asked.

use crate::le::U32le;
use crate::op::Op;
use crate::view::{view, Wire, WireError};

pub const OPCODE_DELETE_BUFFER: u32 = 0x3e8;
pub const OPCODE_DELETE_TEXTURE: u32 = 0x3e9;
pub const OPCODE_DELETE_DEPTH_STENCIL_STATE: u32 = 0x3ea;
pub const OPCODE_DELETE_SAMPLER_STATE: u32 = 0x3eb;
pub const OPCODE_DELETE_FUNCTION: u32 = 0x3ed;
pub const OPCODE_DELETE_COMPUTE_PIPELINE_STATE: u32 = 0x3ee;
pub const OPCODE_DELETE_RENDER_PIPELINE_STATE: u32 = 0x3ef;
pub const OPCODE_DELETE_FENCE: u32 = 0x3f1;
pub const OPCODE_DELETE_HEAP: u32 = 0x3f4;
pub const OPCODE_DELETE_RASTERIZATION_RATE_MAP: u32 = 0x3f6;
pub const OPCODE_DELETE_INDIRECT_COMMAND_BUFFER: u32 = 0x3f7;

/// Every destroy record is the header and one ref.
pub const DELETE_TOTAL_LEN: u32 = 12;

/// The object being destroyed.
///
/// Eleven selectors write this identical record and differ only in opcode, so
/// **the kind comes from the opcode and from nowhere else** — the same shape as
/// [`crate::ops::info::Query`], and the same hazard: a view that read the wrong
/// four bytes would pass on all eleven if they named one object. Each fixture
/// therefore deletes a ref the serializer's own allocator handed out, and the
/// eleven refs are distinct by construction (`lifecycle_delete_*`).
///
/// There is no second field. Twelve bytes total, of which four are payload, and
/// the record carries no generation, no allocator identity and no reason.
#[repr(C)]
#[derive(Debug)]
pub struct Delete {
    pub object_ref: U32le,
}

// SAFETY: one align-1 all-bytes-valid `le` scalar.
unsafe impl Wire for Delete {}

/// Whether `opcode` is one of the eleven destroy records.
///
/// A range check would be wrong: five numbers inside the span belong to no
/// selector this capture found, and treating an unmeasured opcode as a delete
/// would destroy an object on the strength of a guess.
#[inline]
pub fn is_delete(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_DELETE_BUFFER
            | OPCODE_DELETE_TEXTURE
            | OPCODE_DELETE_DEPTH_STENCIL_STATE
            | OPCODE_DELETE_SAMPLER_STATE
            | OPCODE_DELETE_FUNCTION
            | OPCODE_DELETE_COMPUTE_PIPELINE_STATE
            | OPCODE_DELETE_RENDER_PIPELINE_STATE
            | OPCODE_DELETE_FENCE
            | OPCODE_DELETE_HEAP
            | OPCODE_DELETE_RASTERIZATION_RATE_MAP
            | OPCODE_DELETE_INDIRECT_COMMAND_BUFFER
    )
}

pub fn delete<'a>(op: &Op<'a>) -> Result<&'a Delete, WireError> {
    debug_assert!(is_delete(op.opcode()));
    view::<Delete>(op.payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::OP_HEADER_LEN;
    use core::mem::size_of;

    #[test]
    fn the_record_is_its_body_plus_the_header() {
        assert_eq!(
            size_of::<Delete>() + OP_HEADER_LEN,
            DELETE_TOTAL_LEN as usize
        );
    }

    /// The eleven opcodes are eleven, and none of the five unclaimed numbers in
    /// the span answers to [`is_delete`].
    ///
    /// The second half is the load-bearing one. A range check over
    /// `0x3e8..=0x3f7` would be shorter and would accept five opcodes no
    /// capture has ever seen — and the consequence of accepting one is
    /// destroying an object the record may not have named.
    #[test]
    fn the_span_has_five_numbers_this_module_does_not_claim() {
        let claimed: usize = (0x3e8..=0x3f7).filter(|op| is_delete(*op)).count();
        assert_eq!(claimed, 11, "the eleven selectors claim eleven opcodes");
        for unclaimed in [0x3ecu32, 0x3f0, 0x3f2, 0x3f3, 0x3f5] {
            assert!(
                !is_delete(unclaimed),
                "{unclaimed:#x} is inside the span and belongs to no selector"
            );
        }
        // Nothing outside the span either, in particular not the encoder spaces.
        for op in [0u32, 1, 0x89, 0xe5, 0x13e, 0x1d4, 0x3e7, 0x3f8] {
            assert!(!is_delete(op), "{op:#x}");
        }
    }

    #[test]
    fn a_short_payload_is_refused_rather_than_read() {
        let buf = [0u8; size_of::<Delete>() - 1];
        assert!(matches!(view::<Delete>(&buf), Err(WireError::Short { .. })));
    }
}
