//! Scaling lists: their syntax (7.3.2.1.1.1), the default lists (Tables
//! 7-3 and 7-4) and the fall-back rules A and B (Table 7-2).
//!
//! Lists are kept in the order the bitstream codes them, the zig-zag scan
//! of 8.5.6. [`ScalingLists::raster_4x4`] and [`ScalingLists::raster_8x8`]
//! give raster order, as `VAIQMatrixBufferH264` wants (va.h: "in raster
//! scan order").

use super::Error;
use super::bits::BitReader;

/// One list as the bitstream gave it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ListSyntax<const N: usize> {
    /// `*_scaling_list_present_flag[i]` was 0: the fall-back rule applies.
    NotPresent,
    /// `useDefaultScalingMatrixFlag` was set.
    UseDefault,
    /// The list, in zig-zag order.
    Explicit([u8; N]),
}

/// The scaling lists a parameter set codes: six 4x4 lists, and up to six
/// 8x8 lists (two unless `chroma_format_idc` is 3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ScalingSyntax {
    pub(crate) lists_4x4: [ListSyntax<16>; 6],
    pub(crate) lists_8x8: [ListSyntax<64>; 6],
}

impl<const N: usize> ListSyntax<N> {
    /// The list in force: as coded, the `default` list, or `fall_back`
    /// where the list is not present (Table 7-2).
    fn or(self, default: [u8; N], fall_back: [u8; N]) -> [u8; N] {
        match self {
            Self::Explicit(list) => list,
            Self::UseDefault => default,
            Self::NotPresent => fall_back,
        }
    }
}

impl ScalingSyntax {
    /// Reads the six 4x4 lists, then `lists_8x8` 8x8 lists.
    pub(crate) fn parse(r: &mut BitReader<'_>, lists_8x8: usize) -> Result<Self, Error> {
        let mut syntax = Self {
            lists_4x4: [ListSyntax::NotPresent; 6],
            lists_8x8: [ListSyntax::NotPresent; 6],
        };
        for list in &mut syntax.lists_4x4 {
            if r.bit()? {
                *list = parse_list(r)?;
            }
        }
        for list in syntax.lists_8x8.iter_mut().take(lists_8x8) {
            if r.bit()? {
                *list = parse_list(r)?;
            }
        }
        Ok(syntax)
    }

    /// Resolves the lists with fall-back rule A (an SPS), when `sequence`
    /// is `None`, or B (a PPS), from the sequence-level lists: the first
    /// list of each kind falls back to the default list (A) or to the
    /// sequence's (B), and each list after it to the one before it of its
    /// kind (Table 7-2).
    pub(crate) fn resolve(&self, sequence: Option<&ScalingLists>) -> ScalingLists {
        let [intra_y, intra_cb, intra_cr, inter_y, inter_cb, inter_cr] = self.lists_4x4;
        let [first_intra, _, _, first_inter, _, _] = sequence.map_or(
            [
                DEFAULT_4X4_INTRA,
                DEFAULT_4X4_INTRA,
                DEFAULT_4X4_INTRA,
                DEFAULT_4X4_INTER,
                DEFAULT_4X4_INTER,
                DEFAULT_4X4_INTER,
            ],
            |sequence| sequence.lists_4x4,
        );
        let intra_y = intra_y.or(DEFAULT_4X4_INTRA, first_intra);
        let intra_cb = intra_cb.or(DEFAULT_4X4_INTRA, intra_y);
        let intra_cr = intra_cr.or(DEFAULT_4X4_INTRA, intra_cb);
        let inter_y = inter_y.or(DEFAULT_4X4_INTER, first_inter);
        let inter_cb = inter_cb.or(DEFAULT_4X4_INTER, inter_y);
        let inter_cr = inter_cr.or(DEFAULT_4X4_INTER, inter_cb);

        let [
            intra_y8,
            inter_y8,
            intra_cb8,
            inter_cb8,
            intra_cr8,
            inter_cr8,
        ] = self.lists_8x8;
        let [first_intra8, first_inter8, ..] = sequence.map_or(
            [
                DEFAULT_8X8_INTRA,
                DEFAULT_8X8_INTER,
                [16; 64],
                [16; 64],
                [16; 64],
                [16; 64],
            ],
            |sequence| sequence.lists_8x8,
        );
        let intra_y8 = intra_y8.or(DEFAULT_8X8_INTRA, first_intra8);
        let inter_y8 = inter_y8.or(DEFAULT_8X8_INTER, first_inter8);
        let intra_cb8 = intra_cb8.or(DEFAULT_8X8_INTRA, intra_y8);
        let inter_cb8 = inter_cb8.or(DEFAULT_8X8_INTER, inter_y8);
        let intra_cr8 = intra_cr8.or(DEFAULT_8X8_INTRA, intra_cb8);
        let inter_cr8 = inter_cr8.or(DEFAULT_8X8_INTER, inter_cb8);
        ScalingLists {
            lists_4x4: [intra_y, intra_cb, intra_cr, inter_y, inter_cb, inter_cr],
            lists_8x8: [
                intra_y8, inter_y8, intra_cb8, inter_cb8, intra_cr8, inter_cr8,
            ],
        }
    }
}

fn parse_list<const N: usize>(r: &mut BitReader<'_>) -> Result<ListSyntax<N>, Error> {
    let mut list = [0u8; N];
    let mut last = 8u8;
    let mut next = 8u8;
    for (j, slot) in list.iter_mut().enumerate() {
        if next != 0 {
            let delta = r.se_i8(-128, 127, "delta_scale")?;
            // 7.3.2.1.1.1: nextScale is (lastScale + delta_scale + 256)
            // % 256, the sum modulo 256, which a byte's wrap is.
            next = last.wrapping_add_signed(delta);
            if j == 0 && next == 0 {
                return Ok(ListSyntax::UseDefault);
            }
        }
        let value = if next == 0 { last } else { next };
        *slot = value;
        last = value;
    }
    Ok(ListSyntax::Explicit(list))
}

/// The scaling lists in force for a picture, in zig-zag order: lists 0 to
/// 5 are 4x4 (Intra Y, Cb, Cr, then Inter), and the 8x8 lists are Intra
/// Y, Inter Y, then the 4:4:4 chroma lists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalingLists {
    /// The 4x4 lists, each in zig-zag order, as coded.
    pub lists_4x4: [[u8; 16]; 6],
    /// The 8x8 lists, each in zig-zag order, as coded.
    pub lists_8x8: [[u8; 64]; 6],
}

impl ScalingLists {
    /// `Flat_4x4_16` and `Flat_8x8_16`: what no scaling matrix means.
    pub(crate) fn flat() -> Self {
        Self {
            lists_4x4: [[16; 16]; 6],
            lists_8x8: [[16; 64]; 6],
        }
    }

    /// 4x4 list `i`, 0 to 5, in raster order; past 5, the flat list.
    pub fn raster_4x4(&self, i: usize) -> [u8; 16] {
        self.lists_4x4
            .get(i)
            .map_or([16; 16], |list| raster(list, &ZIGZAG_4X4))
    }

    /// 8x8 list `i`, 0 to 5, in raster order; past 5, the flat list.
    pub fn raster_8x8(&self, i: usize) -> [u8; 64] {
        self.lists_8x8
            .get(i)
            .map_or([16; 64], |list| raster(list, &ZIGZAG_8X8))
    }
}

/// `list`, in zig-zag order, in raster order: `zigzag` is the raster index
/// of each zig-zag index.
fn raster<const N: usize>(list: &[u8; N], zigzag: &[u8; N]) -> [u8; N] {
    let mut out = [0u8; N];
    for (&value, &at) in list.iter().zip(zigzag) {
        if let Some(slot) = out.get_mut(usize::from(at)) {
            *slot = value;
        }
    }
    out
}

/// Table 7-3, `Default_4x4_Intra`, by zig-zag index.
const DEFAULT_4X4_INTRA: [u8; 16] = [
    6, 13, 13, 20, 20, 20, 28, 28, 28, 28, 32, 32, 32, 37, 37, 42,
];
/// Table 7-3, `Default_4x4_Inter`.
const DEFAULT_4X4_INTER: [u8; 16] = [
    10, 14, 14, 20, 20, 20, 24, 24, 24, 24, 27, 27, 27, 30, 30, 34,
];
/// Table 7-4, `Default_8x8_Intra`.
const DEFAULT_8X8_INTRA: [u8; 64] = [
    6, 10, 10, 13, 11, 13, 16, 16, 16, 16, 18, 18, 18, 18, 18, 23, 23, 23, 23, 23, 23, 25, 25, 25,
    25, 25, 25, 25, 27, 27, 27, 27, 27, 27, 27, 27, 29, 29, 29, 29, 29, 29, 29, 31, 31, 31, 31, 31,
    31, 33, 33, 33, 33, 33, 36, 36, 36, 36, 38, 38, 38, 40, 40, 42,
];
/// Table 7-4, `Default_8x8_Inter`.
const DEFAULT_8X8_INTER: [u8; 64] = [
    9, 13, 13, 15, 13, 15, 17, 17, 17, 17, 19, 19, 19, 19, 19, 21, 21, 21, 21, 21, 21, 22, 22, 22,
    22, 22, 22, 22, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 27, 27, 27, 27, 27,
    27, 28, 28, 28, 28, 28, 30, 30, 30, 30, 32, 32, 32, 33, 33, 35,
];

/// The frame zig-zag scan of a 4x4 block (8.5.6, Table 8-13): the raster
/// index of each zig-zag index.
const ZIGZAG_4X4: [u8; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
/// The frame zig-zag scan of an 8x8 block (8.5.7, Table 8-14).
const ZIGZAG_8X8: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_scans_are_permutations() {
        let mut seen = [false; 64];
        for &k in &ZIGZAG_8X8 {
            seen[usize::from(k)] = true;
        }
        assert!(seen.iter().all(|&s| s));
        let mut seen = [false; 16];
        for &k in &ZIGZAG_4X4 {
            seen[usize::from(k)] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn fall_back_rule_a_copies_the_previous_list_and_defaults_the_first() {
        let mut syntax = ScalingSyntax {
            lists_4x4: [ListSyntax::NotPresent; 6],
            lists_8x8: [ListSyntax::NotPresent; 6],
        };
        syntax.lists_4x4[1] = ListSyntax::Explicit([7; 16]);
        let lists = syntax.resolve(None);
        assert_eq!(lists.lists_4x4[0], DEFAULT_4X4_INTRA);
        assert_eq!(lists.lists_4x4[1], [7; 16]);
        assert_eq!(lists.lists_4x4[2], [7; 16]);
        assert_eq!(lists.lists_4x4[3], DEFAULT_4X4_INTER);
        assert_eq!(lists.lists_4x4[5], DEFAULT_4X4_INTER);
        assert_eq!(lists.lists_8x8[0], DEFAULT_8X8_INTRA);
        assert_eq!(lists.lists_8x8[1], DEFAULT_8X8_INTER);
        // 4:4:4's 8x8 chroma lists fall back to the one two before.
        syntax.lists_8x8[0] = ListSyntax::Explicit([7; 64]);
        syntax.lists_8x8[3] = ListSyntax::UseDefault;
        let lists = syntax.resolve(None);
        assert_eq!(lists.lists_8x8[2], [7; 64]);
        assert_eq!(lists.lists_8x8[4], [7; 64]);
        assert_eq!(lists.lists_8x8[3], DEFAULT_8X8_INTER);
        assert_eq!(lists.lists_8x8[5], DEFAULT_8X8_INTER);
    }

    #[test]
    fn fall_back_rule_b_takes_the_sequence_lists() {
        let mut sequence = ScalingLists::flat();
        sequence.lists_4x4[0] = [3; 16];
        sequence.lists_4x4[3] = [4; 16];
        sequence.lists_8x8[1] = [5; 64];
        let syntax = ScalingSyntax {
            lists_4x4: [ListSyntax::NotPresent; 6],
            lists_8x8: [ListSyntax::NotPresent; 6],
        };
        let lists = syntax.resolve(Some(&sequence));
        assert_eq!(lists.lists_4x4[2], [3; 16]);
        assert_eq!(lists.lists_4x4[4], [4; 16]);
        assert_eq!(lists.lists_8x8[1], [5; 64]);
        assert_eq!(lists.lists_8x8[0], [16; 64]);
    }

    #[test]
    fn raster_order_undoes_the_zigzag_scan() {
        let mut lists = ScalingLists::flat();
        lists.lists_4x4[0] = core::array::from_fn(|k| u8::try_from(k).unwrap_or(0));
        let raster = lists.raster_4x4(0);
        // Zig-zag index 2 is raster index 4 (row 1, column 0).
        assert_eq!(raster[4], 2);
        assert_eq!(raster[1], 1);
        assert_eq!(raster[15], 15);
    }
}
