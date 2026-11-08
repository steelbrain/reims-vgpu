//! The bounds every H.264 picture is held to before an accelerator gets
//! it: what the standard allows of what reaches the hardware, checked in
//! one place, whatever the parsers above let through. A picture past them
//! is refused, with the error naming what it breaks, and nothing of it is
//! submitted, so a malformed stream from the guest never reaches a driver
//! as values a conforming stream cannot have. The driver's own limits, its profiles
//! and sizes, are the accelerator's to check when it is configured.

use super::Error;
use super::bits::BitReader;
use super::decoder::{PictureParams, ReferenceFrame, SliceParams};
use super::slice::SliceType;

/// The largest `SliceQPY` (7.4.3): 51, at any bit depth.
const MAX_SLICE_QP: i32 = 51;

/// `DiffPicOrderCnt()`'s range in the decoding process (8.2.1): −2^15 to
/// 2^15 − 1.
const DIFF_POC: std::ops::RangeInclusive<i64> = -(1 << 15)..=(1 << 15) - 1;

/// Checks a picture and its slices as [`super::Decoder`] hands them to its
/// accelerator.
pub(crate) fn check_picture<P>(
    params: &PictureParams<'_, P>,
    slices: &[SliceParams<'_, P>],
) -> Result<(), Error> {
    let sps = params.sps;
    let pps = params.pps;
    // QpBdOffsetY (7-4): 0 at the 8 bits this decoder takes.
    let qp_bd_offset = i32::from(sps.bit_depth_luma_minus8)
        .checked_mul(6)
        .ok_or(Error::Overflow("QpBdOffsetY"))?;
    let lowest_qp = qp_bd_offset
        .checked_neg()
        .ok_or(Error::Overflow("QpBdOffsetY"))?;
    // 7.4.2.2: pic_init_qp_minus26 is −(26 + QpBdOffsetY) to +25.
    let lowest_init = lowest_qp
        .checked_sub(26)
        .ok_or(Error::Overflow("QpBdOffsetY"))?;
    if !(lowest_init..=25).contains(&i32::from(pps.pic_init_qp_minus26)) {
        return Err(Error::OutOfRange("pic_init_qp_minus26"));
    }
    // 7.4.3.3: no more than Max(max_num_ref_frames, 1) frames are marked
    // as references, which is 16 at most, VA-API's ReferenceFrames too.
    let most = usize::from(sps.max_num_ref_frames.max(1));
    if params.references.len() > most {
        return Err(Error::TooManyReferences);
    }
    let picture_mbs = sps.frame_size_in_mbs();
    for slice in slices {
        let header = slice.header;
        // 7.4.1.2.1: the active parameter sets do not change within a
        // picture, though a sample may carry a new set with the same id
        // between two of its slices.
        if !(std::ptr::eq(&*header.sps, sps) || *header.sps == *sps)
            || !(std::ptr::eq(&*header.pps, pps) || *header.pps == *pps)
        {
            return Err(Error::ParameterSetsChangedInPicture);
        }
        // 7.4.3: first_mb_in_slice is below PicSizeInMbs of the SPS the
        // accelerator was configured for.
        if header.first_mb_in_slice >= picture_mbs {
            return Err(Error::OutOfRange("first_mb_in_slice"));
        }
        // 7.4.3: SliceQPY = 26 + pic_init_qp_minus26 + slice_qp_delta is
        // −QpBdOffsetY to +51.
        let slice_qp = i32::from(pps.pic_init_qp_minus26)
            .checked_add(26)
            .and_then(|qp| qp.checked_add(i32::from(header.slice_qp_delta)))
            .ok_or(Error::Overflow("SliceQPY"))?;
        if !(lowest_qp..=MAX_SLICE_QP).contains(&slice_qp) {
            return Err(Error::OutOfRange("SliceQPY"));
        }
        // 7.3.4: slice_data() holds at least one macroblock, so something
        // follows the header before rbsp_trailing_bits().
        if !data_follows(slice.nal, header.header_bits) {
            return Err(Error::EmptySlice);
        }
        let l0 = usize::from(header.num_ref_idx_l0_active_minus1) + 1;
        let l1 = usize::from(header.num_ref_idx_l1_active_minus1) + 1;
        let lengths = match header.slice_type {
            SliceType::P | SliceType::Sp => [l0, 0],
            SliceType::B => [l0, l1],
            SliceType::I | SliceType::Si => [0, 0],
        };
        for (list, expected) in slice.ref_lists.iter().zip(lengths) {
            // 7.4.3: 32 entries at most, VA-API's RefPicList0 and 1 too.
            if list.len() != expected || list.len() > 32 {
                return Err(Error::ReferenceListLength);
            }
            // A list names only the DPB's references, which are the ones
            // the accelerator is given.
            let in_dpb = |entry: &ReferenceFrame<'_, P>| {
                params
                    .references
                    .iter()
                    .any(|r| std::ptr::eq(r.picture, entry.picture))
            };
            if !list.iter().flatten().all(in_dpb) {
                return Err(Error::ReferenceOutsideDpb);
            }
        }
        // 8.2.1: DiffPicOrderCnt() stays within 16 bits where the decoding
        // process uses it, which for a B frame is temporal direct
        // prediction (8.4.1.2.3) and implicit weights (8.4.2.3.1), both
        // between the current frame and a short-term reference in list 0,
        // and between that and one in list 1.
        if header.slice_type == SliceType::B
            && (!header.direct_spatial_mv_pred_flag || pps.weighted_bipred_idc == 2)
        {
            let short_terms = |list: &[Option<ReferenceFrame<'_, P>>]| -> Vec<i64> {
                list.iter()
                    .flatten()
                    .filter(|r| !r.long_term)
                    .map(|r| i64::from(r.top_poc.min(r.bottom_poc)))
                    .collect()
            };
            let [l0, l1] = slice.ref_lists.each_ref().map(|list| short_terms(list));
            let current = i64::from(params.top_poc.min(params.bottom_poc));
            let fits = |a: i64, b: i64| a.checked_sub(b).is_some_and(|d| DIFF_POC.contains(&d));
            if !l0.iter().chain(&l1).all(|&r| fits(current, r))
                || !l1.iter().all(|&r1| l0.iter().all(|&r0| fits(r1, r0)))
            {
                return Err(Error::OutOfRange("DiffPicOrderCnt"));
            }
        }
    }
    // 7.4.3: the slices of a picture do not overlap, so no two start at
    // one macroblock, whatever their order.
    let mut starts: Vec<u32> = slices.iter().map(|s| s.header.first_mb_in_slice).collect();
    starts.sort_unstable();
    if starts
        .windows(2)
        .any(|pair| matches!(pair, [a, b] if a == b))
    {
        return Err(Error::OverlappingSlices);
    }
    Ok(())
}

/// Whether anything but `rbsp_trailing_bits()` follows the first
/// `header_bits` bits of the RBSP of `nal`.
fn data_follows(nal: &[u8], header_bits: usize) -> bool {
    let mut reader = BitReader::new(nal);
    u32::try_from(header_bits).is_ok_and(|bits| reader.skip(bits).is_ok())
        && reader.more_rbsp_data()
}
