//! The ranges HEVC's parameter sets and slice segment headers are held to
//! before a picture reaches the accelerator: one past each bound is
//! refused, and the bound itself taken.

use super::*;

/// What the parser says of `spec`'s SPS: nothing, or the bound it is past.
fn sps_refusal(spec: &SpsSpec) -> Option<String> {
    sps::Sps::parse(&sps(spec)).err().map(|e| e.to_string())
}

/// What decoding an IDR picture of `sps_spec` and `pps_spec` says, and how
/// many pictures reached the accelerator.
fn idr_refusal(sps_spec: &SpsSpec, pps_spec: &PpsSpec) -> (Option<String>, usize) {
    let idr = slice(
        sps_spec,
        pps_spec,
        &SliceSpec {
            unit_type: nal_type::IDR_W_RADL,
            ..SliceSpec::default()
        },
    );
    let (error, decoded) = refused(&[sample(&[sps(sps_spec), pps(pps_spec), idr])]);
    (error.map(|e| e.to_string()), decoded)
}

fn out_of_range(name: &str) -> Option<String> {
    Some(format!("{name} is out of range"))
}

#[test]
fn sps_block_sizes_depths_and_pcm_past_their_ranges_are_refused() {
    let with = |blocks: [u32; 6], pcm: Option<[u32; 4]>| SpsSpec {
        blocks,
        pcm,
        ..SpsSpec::default()
    };
    // The default: coding blocks of 8 to 16, transform blocks of 4 to 16,
    // depths of 1.
    for (spec, refusal) in [
        // MinTbLog2SizeY under MinCbLog2SizeY: 8 against 8, then 8
        // against 16.
        (
            with([0, 1, 1, 1, 1, 1], None),
            out_of_range("MinTbLog2SizeY"),
        ),
        (with([1, 1, 1, 2, 1, 1], None), None),
        // MaxTbLog2SizeY no more than the CTB: 32 in a CTB of 16; no more
        // than 32: 64 in a CTB of 64.
        (
            with([0, 1, 0, 3, 1, 1], None),
            out_of_range("MaxTbLog2SizeY"),
        ),
        (
            with([1, 2, 1, 3, 1, 1], None),
            out_of_range("MaxTbLog2SizeY"),
        ),
        (with([1, 2, 1, 2, 3, 3], None), None),
        // Depths of CtbLog2SizeY - MinTbLog2SizeY, 2 here, at most.
        (
            with([0, 1, 0, 2, 3, 1], None),
            out_of_range("max_transform_hierarchy_depth_inter"),
        ),
        (
            with([0, 1, 0, 2, 1, 3], None),
            out_of_range("max_transform_hierarchy_depth_intra"),
        ),
        (with([0, 1, 0, 2, 2, 2], None), None),
        // PCM samples of 9 bits in an 8-bit picture.
        (
            with([0, 1, 0, 2, 1, 1], Some([8, 7, 0, 1])),
            out_of_range("PcmBitDepthY"),
        ),
        (
            with([0, 1, 0, 2, 1, 1], Some([7, 8, 0, 1])),
            out_of_range("PcmBitDepthC"),
        ),
        (with([0, 1, 0, 2, 1, 1], Some([7, 7, 0, 1])), None),
        // PCM blocks from the smallest coding block to the CTB: 8 under
        // coding blocks of 16, 32 in a CTB of 16, then 32 to 32 there.
        (
            with([1, 1, 1, 2, 1, 1], Some([7, 7, 0, 0])),
            out_of_range("Log2MinIpcmCbSizeY"),
        ),
        (
            with([0, 1, 0, 2, 1, 1], Some([7, 7, 2, 0])),
            out_of_range("Log2MinIpcmCbSizeY"),
        ),
        (
            with([0, 1, 0, 2, 1, 1], Some([7, 7, 0, 2])),
            out_of_range("Log2MaxIpcmCbSizeY"),
        ),
        (with([1, 1, 1, 2, 1, 1], Some([7, 7, 1, 1])), None),
    ] {
        assert_eq!(
            sps_refusal(&spec),
            refusal,
            "{:?} {:?}",
            spec.blocks,
            spec.pcm
        );
    }
}

#[test]
fn ctbs_of_8_are_refused_and_16_taken() {
    let pps_spec = PpsSpec::default();
    // CTBs of 8, with transform blocks of 4 and 8: the syntax allows it,
    // the Main profiles do not (A.3.2).
    let eight = SpsSpec {
        blocks: [0, 0, 0, 1, 1, 1],
        ..SpsSpec::default()
    };
    assert_eq!(sps_refusal(&eight), None);
    assert_eq!(
        idr_refusal(&eight, &pps_spec),
        (out_of_range("CtbLog2SizeY"), 0)
    );
    assert_eq!(idr_refusal(&SpsSpec::default(), &pps_spec), (None, 1));
}

#[test]
fn pps_values_past_what_their_sps_allows_are_refused() {
    let eight = SpsSpec::default();
    let ten = SpsSpec {
        profile_idc: 2,
        bit_depth_minus8: 2,
        ..SpsSpec::default()
    };
    let pps_of = |init_qp_minus26, cu_qp_delta_depth, merge_minus2| PpsSpec {
        init_qp_minus26,
        cu_qp_delta_depth,
        log2_parallel_merge_level_minus2: merge_minus2,
        ..PpsSpec::default()
    };
    // init_qp_minus26 from -(26 + QpBdOffsetY): -26 at 8 bits, -38 at 10.
    // The slice's delta keeps SliceQpY in its range, so that only the PPS
    // is past one.
    for (sps_spec, init_qp, refusal) in [
        (&eight, -27, out_of_range("init_qp_minus26")),
        (&eight, -26, None),
        (&ten, -39, out_of_range("init_qp_minus26")),
        (&ten, -38, None),
    ] {
        let pps_spec = pps_of(init_qp, None, 0);
        let idr = slice(
            sps_spec,
            &pps_spec,
            &SliceSpec {
                unit_type: nal_type::IDR_W_RADL,
                slice_qp_delta: 1,
                ..SliceSpec::default()
            },
        );
        let (error, _) = refused(&[sample(&[sps(sps_spec), pps(&pps_spec), idr])]);
        assert_eq!(error.map(|e| e.to_string()), refusal, "{init_qp}");
    }
    // Coding blocks of 8 to 16: QP deltas one level down, and a merge level
    // of the CTB's 16.
    for (pps_spec, refusal) in [
        (
            pps_of(0, Some(2), 0),
            out_of_range("diff_cu_qp_delta_depth"),
        ),
        (pps_of(0, Some(1), 0), None),
        (
            pps_of(0, None, 3),
            out_of_range("log2_parallel_merge_level_minus2"),
        ),
        (pps_of(0, None, 2), None),
    ] {
        let decoded = usize::from(refusal.is_none());
        assert_eq!(idr_refusal(&eight, &pps_spec), (refusal, decoded));
    }
}

#[test]
fn tiles_under_256_luma_samples_across_or_64_down_are_refused() {
    let sized = |width, height| SpsSpec {
        size: (width, height),
        ..SpsSpec::default()
    };
    let tiles = |columns, rows| PpsSpec {
        tiles: Some((columns, rows)),
        ..PpsSpec::default()
    };
    // CTBs of 16: 496 across is 31 CTBs, so two columns of 15 and 16, 240
    // and 256 samples; 112 down is 7, so two rows of 3 and 4, 48 and 64.
    for (sps_spec, pps_spec, refusal) in [
        (
            sized(496, 128),
            tiles(2, 1),
            out_of_range("ColumnWidthInLumaSamples"),
        ),
        (sized(512, 128), tiles(2, 1), None),
        (
            sized(512, 112),
            tiles(1, 2),
            out_of_range("RowHeightInLumaSamples"),
        ),
        (sized(512, 128), tiles(1, 2), None),
        (sized(512, 128), tiles(2, 2), None),
    ] {
        let decoded = usize::from(refusal.is_none());
        assert_eq!(
            idr_refusal(&sps_spec, &pps_spec),
            (refusal, decoded),
            "{:?} {:?}",
            sps_spec.size,
            pps_spec.tiles
        );
    }
}

#[test]
fn entry_points_past_the_tiles_and_rows_are_refused() {
    let entry_points = |sps_spec: &SpsSpec, pps_spec: &PpsSpec, entry_points| {
        let idr = slice(
            sps_spec,
            pps_spec,
            &SliceSpec {
                unit_type: nal_type::IDR_W_RADL,
                entry_points,
                ..SliceSpec::default()
            },
        );
        let (error, _) = refused(&[sample(&[sps(sps_spec), pps(pps_spec), idr])]);
        error.map(|e| e.to_string())
    };
    let small = SpsSpec::default();
    let large = SpsSpec {
        size: (512, 128),
        ..SpsSpec::default()
    };
    let wavefronts = PpsSpec {
        wavefronts: true,
        ..PpsSpec::default()
    };
    let tiles = PpsSpec {
        tiles: Some((2, 2)),
        ..PpsSpec::default()
    };
    let both = PpsSpec {
        tiles: Some((2, 1)),
        wavefronts: true,
        ..PpsSpec::default()
    };
    let past = out_of_range("num_entry_point_offsets");
    // Three CTB rows; four tiles; two tile columns of eight rows.
    for (sps_spec, pps_spec, count, refusal) in [
        (&small, &wavefronts, 3, past.clone()),
        (&small, &wavefronts, 2, None),
        (&large, &tiles, 4, past.clone()),
        (&large, &tiles, 3, None),
        (&large, &both, 16, past.clone()),
        (&large, &both, 15, None),
    ] {
        assert_eq!(entry_points(sps_spec, pps_spec, count), refusal, "{count}");
    }
}

/// What decoding an IDR picture, then `next` (its order count 1), says,
/// and how many pictures reached the accelerator.
fn second_picture_refusal(
    sps_spec: &SpsSpec,
    pps_spec: &PpsSpec,
    next: &SliceSpec,
) -> (Option<String>, usize) {
    let idr = slice(
        sps_spec,
        pps_spec,
        &SliceSpec {
            unit_type: nal_type::IDR_W_RADL,
            ..SliceSpec::default()
        },
    );
    let next = slice(
        sps_spec,
        pps_spec,
        &SliceSpec {
            poc_lsb: 1,
            ..next.clone()
        },
    );
    let (error, decoded) = refused(&[
        sample(&[sps(sps_spec), pps(pps_spec), idr]),
        sample(&[next]),
    ]);
    (error.map(|e| e.to_string()), decoded)
}

#[test]
fn a_cra_or_bla_picture_predicting_from_its_set_is_refused() {
    // Set 0 has the picture before used by the current one, set 1 kept
    // for later ones.
    let sps_spec = SpsSpec {
        sets: vec![
            Set::Explicit(vec![(0, true)], vec![]),
            Set::Explicit(vec![(0, false)], vec![]),
        ],
        ..SpsSpec::default()
    };
    let pps_spec = PpsSpec::default();
    for unit_type in [nal_type::CRA_NUT, nal_type::BLA_W_LP] {
        let irap = |set| SliceSpec {
            unit_type,
            set: Ok(set),
            ..SliceSpec::default()
        };
        assert_eq!(
            second_picture_refusal(&sps_spec, &pps_spec, &irap(0)),
            (out_of_range("NumPicTotalCurr of an IRAP picture"), 1)
        );
        assert_eq!(
            second_picture_refusal(&sps_spec, &pps_spec, &irap(1)),
            (None, 2)
        );
    }
}

#[test]
fn chroma_qp_offsets_past_12_together_are_refused() {
    let sps_spec = SpsSpec::default();
    let pps_of = |chroma_qp_offsets| PpsSpec {
        chroma_qp_offsets,
        slice_chroma_qp_offsets: true,
        ..PpsSpec::default()
    };
    for (pps_offsets, slice_offsets, refusal) in [
        (
            (12, 0),
            (1, 0),
            out_of_range("pps_cb_qp_offset + slice_cb_qp_offset"),
        ),
        ((12, 0), (0, 0), None),
        ((6, 0), (6, 0), None),
        (
            (0, -12),
            (0, -1),
            out_of_range("pps_cr_qp_offset + slice_cr_qp_offset"),
        ),
        ((-6, -6), (-6, -6), None),
    ] {
        let pps_spec = pps_of(pps_offsets);
        let idr = slice(
            &sps_spec,
            &pps_spec,
            &SliceSpec {
                unit_type: nal_type::IDR_W_RADL,
                chroma_qp_offsets: slice_offsets,
                ..SliceSpec::default()
            },
        );
        let (error, _) = refused(&[sample(&[sps(&sps_spec), pps(&pps_spec), idr])]);
        assert_eq!(
            error.map(|e| e.to_string()),
            refusal,
            "{pps_offsets:?} {slice_offsets:?}"
        );
    }
}

#[test]
fn weight_flags_past_24_are_refused() {
    let sps_spec = SpsSpec {
        sets: vec![Set::Explicit(vec![(0, true)], vec![])],
        ..SpsSpec::default()
    };
    let pps_spec = PpsSpec {
        weighted: true,
        ..PpsSpec::default()
    };
    // Each luma flag counts 1 and each chroma flag 2.
    let both = |entries| vec![(true, true); entries];
    let mixed = (0..12).map(|i| (true, i % 2 == 0)).collect::<Vec<_>>();
    for (weights, refusal) in [
        (both(9), out_of_range("sumWeightFlags")),
        (both(8), None),
        (mixed, None),
    ] {
        let entries = weights.len() as u32;
        let p = SliceSpec {
            slice_type: 1,
            active: Some((entries - 1, 0)),
            weights: Some(weights),
            ..SliceSpec::default()
        };
        let decoded = 1 + usize::from(refusal.is_none());
        assert_eq!(
            second_picture_refusal(&sps_spec, &pps_spec, &p),
            (refusal, decoded),
            "{entries}"
        );
    }
}

/// `scaling_list_data()` with every list the default but `coded`'s:
/// `(sizeId, its DC's scaling_list_dc_coef_minus8, its deltas)` for matrix
/// 0.
fn scaling_list_data(coded: (usize, i32, &[i32])) -> Vec<u8> {
    let mut w = BitWriter::new();
    let (size_id, dc_minus8, deltas) = coded;
    for size in 0..4 {
        let step = if size == 3 { 3 } else { 1 };
        for matrix in (0..6).step_by(step) {
            if size != size_id || matrix != 0 {
                w.bit(false).ue(0);
                continue;
            }
            w.bit(true);
            if size > 1 {
                w.se(dc_minus8);
            }
            let count = if size == 0 { 16 } else { 64 };
            for i in 0..count {
                w.se(deltas.get(i).copied().unwrap_or(0));
            }
        }
    }
    w.finish()
}

#[test]
fn a_scaling_list_entry_of_0_is_refused() {
    let parse =
        |coded| scaling::ScalingLists::parse(&mut BitReader::new(&scaling_list_data(coded)));
    // A 4x4 list starts from 8: -8 makes 0, -7 makes 1. A 16x16 one
    // starts from its DC, 1 here: -1 makes 0, and +255 wraps there too.
    for (coded, refused) in [
        ((0, 0, &[-8][..]), true),
        ((0, 0, &[-7][..]), false),
        ((0, 0, &[-7, 127, 127, 1][..]), true),
        ((2, -7, &[-1][..]), true),
        ((2, -7, &[0][..]), false),
        ((3, 0, &[120, 127][..]), false),
        ((3, 0, &[120, 127, 1][..]), true),
    ] {
        let result = parse(coded);
        assert_eq!(
            result.err(),
            refused.then_some(Error::OutOfRange("ScalingList")),
            "{coded:?}"
        );
    }
}

#[test]
fn short_term_sets_of_more_pictures_than_the_dpb_holds_are_refused() {
    // A DPB of 5: sets of 4 pictures besides the current one.
    let coded = |negatives: usize, positives: usize| {
        Set::Explicit(vec![(0, true); negatives], vec![(0, false); positives])
    };
    let with_sets = |sets| SpsSpec {
        sets,
        ..SpsSpec::default()
    };
    for (spec, refusal) in [
        (
            with_sets(vec![coded(5, 0)]),
            out_of_range("num_negative_pics"),
        ),
        (
            with_sets(vec![coded(3, 2)]),
            out_of_range("num_positive_pics"),
        ),
        (with_sets(vec![coded(4, 0)]), None),
        (with_sets(vec![coded(2, 2)]), None),
        // Predicted from four, each kept, with the reference picture
        // itself: five.
        (
            with_sets(vec![
                coded(4, 0),
                Set::Predicted(0, -1, vec![(true, true); 5]),
            ]),
            out_of_range("NumDeltaPocs"),
        ),
        // The same, dropping one.
        (
            with_sets(vec![
                coded(4, 0),
                Set::Predicted(
                    0,
                    -1,
                    [vec![(true, true); 4], vec![(false, false)]].concat(),
                ),
            ]),
            None,
        ),
    ] {
        assert_eq!(sps_refusal(&spec), refusal);
    }
    // A slice header's own set, against the same DPB.
    let sps_spec = SpsSpec::default();
    let pps_spec = PpsSpec::default();
    for (own, refusal) in [
        (coded(5, 0), out_of_range("num_negative_pics")),
        (coded(1, 0), None),
    ] {
        let p = SliceSpec {
            slice_type: 1,
            set: Err(own),
            ..SliceSpec::default()
        };
        let decoded = 1 + usize::from(refusal.is_none());
        assert_eq!(
            second_picture_refusal(&sps_spec, &pps_spec, &p),
            (refusal, decoded)
        );
    }
}

#[test]
fn an_sps_that_differs_in_more_than_its_vui_at_a_non_irap_picture_is_refused() {
    // Set 0 predicts from the picture before; set 1, for the CRA, keeps it.
    let first = SpsSpec {
        sets: vec![
            Set::Explicit(vec![(0, true)], vec![]),
            Set::Explicit(vec![(0, false)], vec![]),
        ],
        ..SpsSpec::default()
    };
    let pps_spec = PpsSpec::default();
    let idr = slice(
        &first,
        &pps_spec,
        &SliceSpec {
            unit_type: nal_type::IDR_W_RADL,
            ..SliceSpec::default()
        },
    );
    // The second picture comes with a new SPS of pictures alike: another
    // order count LSB length, or another VUI alone.
    let other_lsb = SpsSpec {
        lsb_minus4: 5,
        ..first.clone()
    };
    let other_vui = SpsSpec {
        vui_full_range: Some(true),
        ..first.clone()
    };
    for (unit_type, slice_type, next, refusal) in [
        (
            nal_type::TRAIL_R,
            1,
            &other_lsb,
            Some(Error::SpsChangedAtNonIrap.to_string()),
        ),
        (nal_type::TRAIL_R, 1, &other_vui, None),
        // An IRAP picture takes one, as before.
        (nal_type::CRA_NUT, 2, &other_lsb, None),
    ] {
        let picture = slice(
            next,
            &pps_spec,
            &SliceSpec {
                unit_type,
                slice_type,
                poc_lsb: 1,
                set: Ok(u32::from(unit_type == nal_type::CRA_NUT)),
                ..SliceSpec::default()
            },
        );
        let (error, decoded) = refused(&[
            sample(&[sps(&first), pps(&pps_spec), idr.clone()]),
            sample(&[sps(next), picture]),
        ]);
        let decoded_wanted = 1 + usize::from(refusal.is_none());
        assert_eq!(
            (error.map(|e| e.to_string()), decoded),
            (refusal, decoded_wanted),
            "{unit_type}"
        );
    }
}
