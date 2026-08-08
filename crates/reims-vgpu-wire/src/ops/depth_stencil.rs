//! Opcode 4 — create depth/stencil state.
//!
//! `-[PGSerializer newDepthStencilStateWithDescriptor:allocator:]`.
//!
//! # Layout
//!
//! Total 40 bytes: the 8-byte [`crate::op::OpHeader`] then a 32-byte payload.
//!
//! ```text
//! payload +000  u32  object_ref
//! payload +004  u8   depth_state  bits [5:0]; [7:6] and +005..+007 unwritten
//! payload +008  u32  front.ops    compare[2:0] fail[5:3] depth_fail[8:6] pass[11:9]
//! payload +012  u32  front.read_mask
//! payload +016  u32  front.write_mask
//! payload +020  u32  back.ops
//! payload +024  u32  back.read_mask
//! payload +028  u32  back.write_mask
//! ```
//!
//! The two faces are the same 12-byte [`StencilFace`] twice, front first.
//!
//! # `depth_state` is a byte, not a word
//!
//! It sits in a four-byte slot and only its low six bits are written; the two
//! bits above them and the three bytes after are the guest's stale ring on a
//! real wire. Reading the slot as a `u32` and masking would give the same
//! answer for the fields named here and a wrong one for anything else, which is
//! how a field gets invented later. The type says what was measured.
//!
//! # How the layout was derived
//!
//! Perturbation, one property per case, 18 cases: the depth compare function
//! and write flag, then each of the four stencil operations and two masks on
//! each face separately. Every case gives its face a value the other face does
//! not hold, so a view that swapped them would report a value no case produced.
//!
//! `reims-vgpu`'s `decode/resource` reads this record with the same offsets
//! and the same bit assignments. It also names bits 4 and 5 of `depth_state`
//! `front_stencil_enabled` and `back_stencil_enabled`; see
//! [`DepthStencilBody::unidentified_state_bits`] for why this crate does not.

use crate::le::U32le;
use crate::op::Op;
use crate::view::{view, Wire, WireError};

/// Opcode for depth/stencil-state creation, observed on
/// `-[PGSerializer newDepthStencilStateWithDescriptor:allocator:]`.
pub const OPCODE_NEW_DEPTH_STENCIL: u32 = 4;

/// Total wire length of a depth/stencil-creation operation, header included.
pub const NEW_DEPTH_STENCIL_TOTAL_LEN: u32 = 40;

/// One stencil face: four operations packed into a word, then two masks.
#[repr(C)]
#[derive(Debug)]
pub struct StencilFace {
    /// Compare function and the three stencil operations. Prefer the accessors.
    pub ops: U32le,
    /// `readMask`. Observed: `0x11223344` on the front face, `0x99aabbcc` on
    /// the back, each in a case that moved only that one.
    pub read_mask: U32le,
    /// `writeMask`. Observed: `0x55667788` front, `0xddeeff00` back.
    pub write_mask: U32le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for StencilFace {}

impl StencilFace {
    /// `MTLCompareFunction`, `ops[2:0]`.
    ///
    /// Observed: the default Always (7) → `0x00000007`; Equal (2) → `2` on the
    /// front face and NotEqual (5) → `5` on the back.
    #[inline]
    pub fn compare_function(&self) -> u8 {
        (self.ops.get() & 0x7) as u8
    }

    /// `stencilFailureOperation`, `ops[5:3]`.
    ///
    /// Observed: IncrementClamp (3) → `0x0000001f` front (`7 | 3 << 3`),
    /// IncrementWrap (6) → `0x00000037` back.
    #[inline]
    pub fn stencil_failure_operation(&self) -> u8 {
        ((self.ops.get() >> 3) & 0x7) as u8
    }

    /// `depthFailureOperation`, `ops[8:6]`.
    ///
    /// Observed: DecrementClamp (4) → `0x00000107` front, DecrementWrap (7) →
    /// `0x000001c7` back.
    #[inline]
    pub fn depth_failure_operation(&self) -> u8 {
        ((self.ops.get() >> 6) & 0x7) as u8
    }

    /// `depthStencilPassOperation`, `ops[11:9]`.
    ///
    /// Observed: Invert (5) → `0x00000a07` front, Replace (2) → `0x00000407`
    /// back.
    #[inline]
    pub fn depth_stencil_pass_operation(&self) -> u8 {
        ((self.ops.get() >> 9) & 0x7) as u8
    }

    /// `ops[31:12]` — twenty bits the serializer writes and no case has moved.
    ///
    /// Read `0` in all 18 cases. Inside the written extent, so a real field
    /// rather than ring noise.
    #[inline]
    pub fn unidentified_ops_bits(&self) -> u32 {
        self.ops.get() >> 12
    }
}

/// Payload of a depth/stencil-creation record.
#[repr(C)]
#[derive(Debug)]
pub struct DepthStencilBody {
    /// Ref the guest's object-ref allocator assigned to the new state.
    pub object_ref: U32le,
    /// Depth compare function and write flag. Prefer the accessors.
    pub depth_state: u8,
    /// Never written by the serializer; on a real wire the guest's stale ring.
    /// Named so nothing reads them by reaching past `depth_state`.
    pub unwritten_after_depth_state: [u8; 3],
    /// Front-facing stencil.
    pub front: StencilFace,
    /// Back-facing stencil.
    pub back: StencilFace,
}

// SAFETY: `le` scalars, a `u8`, a `[u8; 3]` and two `Wire` structs, all align-1
// and valid for every byte pattern.
unsafe impl Wire for DepthStencilBody {}

impl DepthStencilBody {
    /// `MTLCompareFunction` for the depth test, `depth_state[2:0]`.
    ///
    /// Observed: the default Always (7) → written bits `0b110111`; Greater (4)
    /// → `0b110100`.
    #[inline]
    pub fn depth_compare_function(&self) -> u8 {
        self.depth_state & 0x7
    }

    /// `depthWriteEnabled`, `depth_state[3]`.
    ///
    /// Observed: `NO` → written bits `0b110111`, `YES` → `0b111111`.
    #[inline]
    pub fn depth_write_enabled(&self) -> bool {
        self.depth_state & (1 << 3) != 0
    }

    /// `depth_state[5:4]` — two bits the serializer writes that read `0b11` in
    /// every case, and that no perturbation has moved.
    ///
    /// `reims-vgpu`'s ported decoder calls them `front_stencil_enabled` and
    /// `back_stencil_enabled`. That is a plausible reading and the experiment
    /// that would confirm it has been run and does **not**: setting
    /// `frontFaceStencil`, `backFaceStencil` or both to `nil` produces a record
    /// byte-identical to one with default faces, ref aside. Metal substitutes a
    /// default face before the serializer sees the descriptor, so from this API
    /// there is no such thing as a state with a face absent.
    ///
    /// To settle them: a source other than `MTLDepthStencilDescriptor` — the
    /// guest driver's own builder, or a serializer driven from a decoded state
    /// rather than a descriptor.
    #[inline]
    pub fn unidentified_state_bits(&self) -> u8 {
        (self.depth_state >> 4) & 0x3
    }
}

/// View the payload of a depth/stencil-creation record.
///
/// Refuses a record whose opcode is not [`OPCODE_NEW_DEPTH_STENCIL`]; the
/// caller is expected to have dispatched on opcode already.
pub fn new_depth_stencil<'a>(op: &Op<'a>) -> Result<&'a DepthStencilBody, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_NEW_DEPTH_STENCIL);
    view::<DepthStencilBody>(op.payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{op, OP_HEADER_LEN};
    use core::mem::size_of;

    #[allow(clippy::too_many_arguments)]
    fn synth(
        depth_state: u8,
        front_ops: u32,
        front_read: u32,
        front_write: u32,
        back_ops: u32,
        back_read: u32,
        back_write: u32,
    ) -> [u8; 40] {
        let mut b = [0xAAu8; NEW_DEPTH_STENCIL_TOTAL_LEN as usize];
        b[0..4].copy_from_slice(&OPCODE_NEW_DEPTH_STENCIL.to_le_bytes());
        b[4..8].copy_from_slice(&NEW_DEPTH_STENCIL_TOTAL_LEN.to_le_bytes());
        b[8..12].copy_from_slice(&11u32.to_le_bytes());
        b[12] = depth_state;
        b[16..20].copy_from_slice(&front_ops.to_le_bytes());
        b[20..24].copy_from_slice(&front_read.to_le_bytes());
        b[24..28].copy_from_slice(&front_write.to_le_bytes());
        b[28..32].copy_from_slice(&back_ops.to_le_bytes());
        b[32..36].copy_from_slice(&back_read.to_le_bytes());
        b[36..40].copy_from_slice(&back_write.to_le_bytes());
        b
    }

    fn baseline() -> [u8; 40] {
        synth(
            0x37,
            7,
            0xffff_ffff,
            0xffff_ffff,
            7,
            0xffff_ffff,
            0xffff_ffff,
        )
    }

    #[test]
    fn the_payload_is_exactly_the_record_minus_its_header() {
        assert_eq!(
            size_of::<DepthStencilBody>() + OP_HEADER_LEN,
            NEW_DEPTH_STENCIL_TOTAL_LEN as usize
        );
        assert_eq!(core::mem::align_of::<DepthStencilBody>(), 1);
        assert_eq!(size_of::<StencilFace>(), 12);
    }

    #[test]
    fn the_baseline_reads_back_metals_defaults() {
        let buf = baseline();
        let o = op(&buf, 0).expect("well formed");
        let d = new_depth_stencil(&o).expect("fits");

        assert_eq!(d.object_ref.get(), 11);
        assert_eq!(d.depth_compare_function(), 7);
        assert!(!d.depth_write_enabled());
        assert_eq!(d.unidentified_state_bits(), 0b11);
        for face in [&d.front, &d.back] {
            assert_eq!(face.compare_function(), 7);
            assert_eq!(face.stencil_failure_operation(), 0);
            assert_eq!(face.depth_failure_operation(), 0);
            assert_eq!(face.depth_stencil_pass_operation(), 0);
            assert_eq!(face.read_mask.get(), 0xffff_ffff);
            assert_eq!(face.write_mask.get(), 0xffff_ffff);
            assert_eq!(face.unidentified_ops_bits(), 0);
        }
    }

    #[test]
    fn each_ops_subfield_moves_only_its_own_bits() {
        // The four packed values the oracle produced on each face, as whole
        // words. A shift error shows up as a second accessor moving.
        for (ops, cmp, fail, dfail, pass) in [
            (0x0000_0007u32, 7, 0, 0, 0),
            (0x0000_0002, 2, 0, 0, 0),
            (0x0000_001f, 7, 3, 0, 0),
            (0x0000_0107, 7, 0, 4, 0),
            (0x0000_0a07, 7, 0, 0, 5),
            (0x0000_0005, 5, 0, 0, 0),
            (0x0000_0037, 7, 6, 0, 0),
            (0x0000_01c7, 7, 0, 7, 0),
            (0x0000_0407, 7, 0, 0, 2),
        ] {
            let buf = synth(0x37, ops, 0, 0, ops, 0, 0);
            let o = op(&buf, 0).expect("well formed");
            let d = new_depth_stencil(&o).expect("fits");
            for face in [&d.front, &d.back] {
                assert_eq!(face.compare_function(), cmp, "{ops:#010x}: compare");
                assert_eq!(
                    face.stencil_failure_operation(),
                    fail,
                    "{ops:#010x}: stencil fail"
                );
                assert_eq!(
                    face.depth_failure_operation(),
                    dfail,
                    "{ops:#010x}: depth fail"
                );
                assert_eq!(
                    face.depth_stencil_pass_operation(),
                    pass,
                    "{ops:#010x}: pass"
                );
                assert_eq!(face.unidentified_ops_bits(), 0, "{ops:#010x}: high bits");
            }
        }
    }

    #[test]
    fn the_two_faces_are_read_from_different_bytes() {
        // The failure this catches is a view that reads one face twice, which
        // every same-on-both-faces test above would pass.
        let buf = synth(
            0x37,
            0x0000_0002,
            0x1122_3344,
            0x5566_7788,
            0x0000_0005,
            0x99aa_bbcc,
            0xddee_ff00,
        );
        let o = op(&buf, 0).expect("well formed");
        let d = new_depth_stencil(&o).expect("fits");
        assert_eq!(d.front.compare_function(), 2);
        assert_eq!(d.back.compare_function(), 5);
        assert_eq!(d.front.read_mask.get(), 0x1122_3344);
        assert_eq!(d.front.write_mask.get(), 0x5566_7788);
        assert_eq!(d.back.read_mask.get(), 0x99aa_bbcc);
        assert_eq!(d.back.write_mask.get(), 0xddee_ff00);
    }

    #[test]
    fn the_unwritten_bytes_change_no_accessors_answer() {
        let a = baseline();
        let mut b = a;
        b[12] |= 0xc0;
        b[13] = 0x5a;
        b[14] = 0xa5;
        b[15] = 0xff;

        let oa = op(&a, 0).expect("well formed");
        let ob = op(&b, 0).expect("well formed");
        let (da, db) = (
            new_depth_stencil(&oa).expect("fits"),
            new_depth_stencil(&ob).expect("fits"),
        );
        assert_eq!(da.depth_compare_function(), db.depth_compare_function());
        assert_eq!(da.depth_write_enabled(), db.depth_write_enabled());
        assert_eq!(da.unidentified_state_bits(), db.unidentified_state_bits());
        assert_eq!(da.front.ops.get(), db.front.ops.get());
    }

    #[test]
    fn a_truncated_depth_stencil_operation_is_refused_rather_than_read_short() {
        let buf = baseline();
        let o = op(&buf, 0).expect("well formed");
        let short = Op {
            header: o.header,
            payload: &o.payload[..16],
            offset: 0,
        };
        assert!(matches!(
            new_depth_stencil(&short),
            Err(WireError::Short { need: 32, have: 16 })
        ));
    }
}
