//! Opcode 3 — create sampler state.
//!
//! `-[PGSerializer newSamplerStateWithDescriptor:allocator:]`. Every Metal app
//! that samples a texture creates one of these, so it is on the guest's hot
//! path even though the record itself is rare per frame.
//!
//! # Layout
//!
//! Total 36 bytes: the 8-byte [`crate::op::OpHeader`] then a 28-byte payload.
//!
//! ```text
//! payload +000  u32  object_ref
//! payload +004  u32  state       every enum the descriptor carries, packed
//! payload +008  u8   flags       bits [3:0]; bits [7:4] and +009..+011 unwritten
//! payload +012  f32  lod_min_clamp
//! payload +016  f32  lod_max_clamp
//! payload +020  8 bytes NEVER WRITTEN
//! ```
//!
//! # The record does not fill its own allocation
//!
//! The serializer asks its allocator for 36 bytes, writes the header's 8 and
//! the payload's first 20, and leaves the last 8 alone. Those are the guest's
//! stale ring on a real wire. The body below therefore stops at 20 bytes rather
//! than covering the record, which is why this module has no
//! `size_of + OP_HEADER_LEN == TOTAL_LEN` assertion and has
//! [`the_body_stops_where_the_serializer_stopped_writing`] instead.
//!
//! The same measurement is why [`SamplerBody::flags`] is a `u8` and not the
//! `u32` its slot is: bits `[3:0]` of that byte are written and everything
//! above them is not.
//!
//! # How the layout was derived
//!
//! Perturbation, one property per case, 16 cases. Every field named below moved
//! in at least one of them; each accessor cites its own observations. The
//! written extent is measured rather than eyeballed — see the crate's
//! `AGENTS.md` on the two-fill capture.
//!
//! `reims-vgpu`'s `decode/resource` reads this record with the same offsets
//! and the same bit assignments, arrived at independently from a ported C
//! header with no derivation recorded. The two agree everywhere, which is worth
//! stating because it was not known before these fixtures existed.

use crate::le::{F32le, U32le};
use crate::op::Op;
use crate::view::{view, Wire, WireError};

/// Opcode for sampler-state creation, observed on
/// `-[PGSerializer newSamplerStateWithDescriptor:allocator:]`.
pub const OPCODE_NEW_SAMPLER: u32 = 3;

/// Bytes the serializer *allocates* for the record, header included.
pub const NEW_SAMPLER_TOTAL_LEN: u32 = 36;

/// Bytes of that allocation the serializer actually writes, header included.
///
/// The eight past this are never touched. See the module doc.
pub const NEW_SAMPLER_WRITTEN_LEN: u32 = 28;

/// The written part of a sampler-creation payload.
#[repr(C)]
#[derive(Debug)]
pub struct SamplerBody {
    /// Ref the guest's object-ref allocator assigned to the new sampler.
    pub object_ref: U32le,
    /// Filters, compare function, address modes, border colour, mip filter,
    /// anisotropy and coordinate normalization, packed. Prefer the accessors.
    pub state: U32le,
    /// Bits `[3:0]` are written; `[7:4]` are not. See
    /// [`SamplerBody::support_argument_buffers`] and
    /// [`SamplerBody::unidentified_flag_bits`].
    pub flags: u8,
    /// Never written by the serializer, and *not* padding in the sense of
    /// "always zero" — on a real wire these hold whatever the ring last had.
    /// Named so nothing reads them by reaching past `flags`.
    pub unwritten_after_flags: [u8; 3],
    /// `lodMinClamp`. Observed: `0.25` → `0x3e800000`.
    pub lod_min_clamp: F32le,
    /// `lodMaxClamp`. Observed: the `FLT_MAX` default → `0x7f7fffff`, and
    /// `6.5` → `0x40d00000`.
    pub lod_max_clamp: F32le,
}

// SAFETY: `le` scalars plus a `u8` and a `[u8; 3]`, all align-1 and valid for
// every byte pattern, so the struct is align-1 and all 20-byte patterns are
// valid.
unsafe impl Wire for SamplerBody {}

impl SamplerBody {
    /// `MTLSamplerMinMagFilter` for minification, `state[0]`.
    ///
    /// Observed: Nearest→`0x84000000`, Linear→`0x84000001`. The Objective-C
    /// side declares a 64-bit enum with two values, so only the low bit has
    /// been made to move; `state[1]` has not, and this crate does not claim it.
    #[inline]
    pub fn min_filter(&self) -> u8 {
        (self.state.get() & 0x1) as u8
    }

    /// `MTLSamplerMinMagFilter` for magnification, `state[2]`.
    ///
    /// Observed: Nearest→`0x84000000`, Linear→`0x84000004`. As with
    /// [`SamplerBody::min_filter`], `state[3]` has not moved.
    #[inline]
    pub fn mag_filter(&self) -> u8 {
        ((self.state.get() >> 2) & 0x1) as u8
    }

    /// Whether the guest asked for `lodAverage`, `state[4]`.
    ///
    /// Observed: `YES` → `0x84000010`, against `0x84000000` for the baseline.
    #[inline]
    pub fn lod_average(&self) -> bool {
        self.state.get() & (1 << 4) != 0
    }

    /// `MTLCompareFunction`, `state[7:5]`.
    ///
    /// Observed: Never (0) → `0x84000000`, Greater (4) → `0x84000080`, i.e. the
    /// ordinal shifted left by 5. Three bits hold the enum's whole range.
    #[inline]
    pub fn compare_function(&self) -> u8 {
        ((self.state.get() >> 5) & 0x7) as u8
    }

    /// `sAddressMode`, `state[10:8]`.
    ///
    /// Observed: MirrorRepeat (3) → `0x84000300`, ClampToBorderColor (5) →
    /// `0x84255500`'s low nibble. Each axis was perturbed to a *different*
    /// mode, so a view reading the wrong one of the three reports a value no
    /// case produced rather than a plausible one.
    #[inline]
    pub fn s_address_mode(&self) -> u8 {
        ((self.state.get() >> 8) & 0x7) as u8
    }

    /// `tAddressMode`, `state[14:12]`.
    ///
    /// Observed: ClampToZero (4) → `0x84004000`.
    #[inline]
    pub fn t_address_mode(&self) -> u8 {
        ((self.state.get() >> 12) & 0x7) as u8
    }

    /// `rAddressMode`, `state[18:16]`.
    ///
    /// Observed: Repeat (2) → `0x84020000`.
    #[inline]
    pub fn r_address_mode(&self) -> u8 {
        ((self.state.get() >> 16) & 0x7) as u8
    }

    /// `MTLSamplerBorderColor`, `state[21:20]`.
    ///
    /// Observed: OpaqueWhite (2) → `0x84255500`, whose byte at `[23:16]` is
    /// `0x25`: `5` is `rAddressMode` (ClampToBorderColor, set in the same case
    /// because the colour is meaningless without it) and `2` is this field.
    #[inline]
    pub fn border_color(&self) -> u8 {
        ((self.state.get() >> 20) & 0x3) as u8
    }

    /// `MTLSamplerMipFilter`, `state[25:24]`.
    ///
    /// Observed: NotMipmapped (0) → `0x84000000`, Nearest (1) → `0x85000000`,
    /// Linear (2) → `0x86000000`.
    #[inline]
    pub fn mip_filter(&self) -> u8 {
        ((self.state.get() >> 24) & 0x3) as u8
    }

    /// `maxAnisotropy`, `state[30:26]`.
    ///
    /// Observed: 1 → `0x84000000` and 13 → `0xb4000000`. Shifting both right by
    /// 26 and masking five bits gives exactly 1 and 13, so the value is carried
    /// verbatim rather than as a log or an index — which is worth stating,
    /// because a two-bit reading of the same two observations also fits and is
    /// wrong.
    ///
    /// Five bits hold `1..=16`, Metal's whole documented range.
    #[inline]
    pub fn max_anisotropy(&self) -> u8 {
        ((self.state.get() >> 26) & 0x1f) as u8
    }

    /// `normalizedCoordinates`, `state[31]`.
    ///
    /// Observed: `YES` → `0x84000000`, `NO` → `0x04000000`.
    #[inline]
    pub fn normalized_coordinates(&self) -> bool {
        self.state.get() & 0x8000_0000 != 0
    }

    /// `supportArgumentBuffers`, `flags[0]`.
    ///
    /// Observed: `NO` → `flags & 0xf == 0`, `YES` → `1`.
    #[inline]
    pub fn support_argument_buffers(&self) -> bool {
        self.flags & 0x1 != 0
    }

    /// `flags[3:1]` — written by the serializer, and never made to move.
    ///
    /// Read `0` in all 16 sampler cases. They are inside the written nibble, so
    /// unlike `flags[7:4]` they are a real field rather than ring noise.
    ///
    /// Tried without moving these bits: every public descriptor property,
    /// `forceSeamsOnCubemapFiltering`, `forceResourceIndex`, resource index,
    /// pixel format, reduction mode, LOD bias, and the private border-colour
    /// selector. `supportArgumentBuffers` already owns bit 0. The remaining
    /// experiment is the guest builder or host consumer; another descriptor
    /// perturbation sweep has no untested property left to offer.
    #[inline]
    pub fn unidentified_flag_bits(&self) -> u8 {
        (self.flags >> 1) & 0x7
    }
}

/// View the written part of a sampler-creation record.
///
/// Refuses a record whose opcode is not [`OPCODE_NEW_SAMPLER`]; the caller is
/// expected to have dispatched on opcode already.
pub fn new_sampler<'a>(op: &Op<'a>) -> Result<&'a SamplerBody, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_NEW_SAMPLER);
    view::<SamplerBody>(op.payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{op, OP_HEADER_LEN};
    use core::mem::size_of;

    fn synth(state: u32, flags: u8, lod_min: f32, lod_max: f32) -> [u8; 36] {
        let mut b = [0xAAu8; NEW_SAMPLER_TOTAL_LEN as usize];
        b[0..4].copy_from_slice(&OPCODE_NEW_SAMPLER.to_le_bytes());
        b[4..8].copy_from_slice(&NEW_SAMPLER_TOTAL_LEN.to_le_bytes());
        b[8..12].copy_from_slice(&7u32.to_le_bytes());
        b[12..16].copy_from_slice(&state.to_le_bytes());
        b[16] = flags;
        b[20..24].copy_from_slice(&lod_min.to_le_bytes());
        b[24..28].copy_from_slice(&lod_max.to_le_bytes());
        b
    }

    #[test]
    fn the_body_stops_where_the_serializer_stopped_writing() {
        assert_eq!(
            size_of::<SamplerBody>() + OP_HEADER_LEN,
            NEW_SAMPLER_WRITTEN_LEN as usize
        );
        assert_eq!(core::mem::align_of::<SamplerBody>(), 1);
        // The gap is the point: a body sized to the allocation would put two
        // fields' worth of the guest's stale ring inside the view.
        assert_eq!(NEW_SAMPLER_TOTAL_LEN - NEW_SAMPLER_WRITTEN_LEN, 8);
    }

    #[test]
    fn the_state_word_splits_into_the_fields_the_oracle_moved() {
        // 0x84000000: nearest/nearest, compare Never, all address modes
        // ClampToEdge, border TransparentBlack, mip NotMipmapped, anisotropy 1,
        // normalized — the baseline the oracle produced.
        let buf = synth(0x8400_0000, 0, 0.0, f32::MAX);
        let o = op(&buf, 0).expect("well formed");
        let s = new_sampler(&o).expect("fits");

        assert_eq!(s.object_ref.get(), 7);
        assert_eq!(s.min_filter(), 0);
        assert_eq!(s.mag_filter(), 0);
        assert!(!s.lod_average());
        assert_eq!(s.compare_function(), 0);
        assert_eq!(s.s_address_mode(), 0);
        assert_eq!(s.t_address_mode(), 0);
        assert_eq!(s.r_address_mode(), 0);
        assert_eq!(s.border_color(), 0);
        assert_eq!(s.mip_filter(), 0);
        assert_eq!(s.max_anisotropy(), 1);
        assert!(s.normalized_coordinates());
        assert!(!s.support_argument_buffers());
        assert_eq!(s.unidentified_flag_bits(), 0);
        assert_eq!(s.lod_min_clamp.get(), 0.0);
        assert_eq!(s.lod_max_clamp.get(), f32::MAX);
    }

    #[test]
    fn each_state_subfield_moves_only_its_own_bits() {
        // Every perturbation the oracle captured, as the whole word it produced
        // and the one accessor it should move. A shift or mask error shows up
        // as a second accessor moving.
        let base = synth(0x8400_0000, 0, 0.0, f32::MAX);
        let o = op(&base, 0).expect("well formed");
        let b = new_sampler(&o).expect("fits");
        let before = [
            b.min_filter() as u32,
            b.mag_filter() as u32,
            b.lod_average() as u32,
            b.compare_function() as u32,
            b.s_address_mode() as u32,
            b.t_address_mode() as u32,
            b.r_address_mode() as u32,
            b.border_color() as u32,
            b.mip_filter() as u32,
            b.max_anisotropy() as u32,
            b.normalized_coordinates() as u32,
        ];

        for (state, moved_index, label) in [
            (0x8400_0001u32, 0, "min_filter"),
            (0x8400_0004, 1, "mag_filter"),
            (0x8400_0010, 2, "lod_average"),
            (0x8400_0080, 3, "compare_function"),
            (0x8400_0300, 4, "s_address_mode"),
            (0x8400_4000, 5, "t_address_mode"),
            (0x8402_0000, 6, "r_address_mode"),
            (0x8500_0000, 8, "mip_filter"),
            (0x8600_0000, 8, "mip_filter_linear"),
            (0xb400_0000, 9, "max_anisotropy"),
            (0x0400_0000, 10, "normalized_coordinates"),
        ] {
            let buf = synth(state, 0, 0.0, f32::MAX);
            let o = op(&buf, 0).expect("well formed");
            let s = new_sampler(&o).expect("fits");
            let after = [
                s.min_filter() as u32,
                s.mag_filter() as u32,
                s.lod_average() as u32,
                s.compare_function() as u32,
                s.s_address_mode() as u32,
                s.t_address_mode() as u32,
                s.r_address_mode() as u32,
                s.border_color() as u32,
                s.mip_filter() as u32,
                s.max_anisotropy() as u32,
                s.normalized_coordinates() as u32,
            ];
            for (i, (a, b)) in after.iter().zip(before.iter()).enumerate() {
                if i == moved_index {
                    assert_ne!(a, b, "{label} did not move its own field");
                } else {
                    assert_eq!(a, b, "{label} moved field {i} as well as its own");
                }
            }
        }
    }

    #[test]
    fn the_unwritten_bytes_change_no_accessors_answer() {
        // Bits [7:4] of `flags` and the three bytes after it are the guest's
        // ring. Reading a field must not depend on them, which is the whole
        // reason `flags` is a `u8` rather than the `u32` its slot is.
        let a = synth(0x8400_0000, 0x00, 0.0, f32::MAX);
        let mut b = a;
        b[16] |= 0xf0;
        b[17] = 0x5a;
        b[18] = 0xa5;
        b[19] = 0xff;

        let oa = op(&a, 0).expect("well formed");
        let ob = op(&b, 0).expect("well formed");
        let (sa, sb) = (
            new_sampler(&oa).expect("fits"),
            new_sampler(&ob).expect("fits"),
        );
        assert_eq!(sa.support_argument_buffers(), sb.support_argument_buffers());
        assert_eq!(sa.unidentified_flag_bits(), sb.unidentified_flag_bits());
        assert_eq!(sa.lod_min_clamp.get(), sb.lod_min_clamp.get());
        assert_eq!(sa.lod_max_clamp.get(), sb.lod_max_clamp.get());
    }

    #[test]
    fn a_truncated_sampler_operation_is_refused_rather_than_read_short() {
        let buf = synth(0, 0, 0.0, 0.0);
        let o = op(&buf, 0).expect("well formed");
        let short = Op {
            header: o.header,
            payload: &o.payload[..8],
            offset: 0,
        };
        assert!(matches!(
            new_sampler(&short),
            Err(WireError::Short { need: 20, have: 8 })
        ));
    }
}
