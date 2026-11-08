//! The bounds every HEVC picture is held to before an accelerator gets
//! it, as H.264's are (`h264/bounds.rs`): what the standard allows of
//! what reaches the hardware, checked in one place, whatever the parsers
//! above let through. A picture past them is refused, with the error
//! naming what it breaks, and nothing of it is submitted.

use super::Error;
use super::bits::BitReader;
use super::decoder::{PictureParams, SliceParams};
use super::dpb::RpsMembership;

/// `DiffPicOrderCnt()`'s range (8.3.1): −2^15 to 2^15 − 1.
const DIFF_POC: std::ops::RangeInclusive<i64> = -(1 << 15)..=(1 << 15) - 1;

/// Checks a picture and its slice segments as [`super::Decoder`] hands
/// them to its accelerator.
pub(crate) fn check_picture<P>(
    params: &PictureParams<'_, P>,
    slices: &[SliceParams<'_>],
) -> Result<(), Error> {
    let sps = params.sps;
    // C.5.2.2: the DPB holds sps_max_dec_pic_buffering_minus1 + 1
    // pictures, the current one among them.
    if params.references.len() >= sps.dpb_size() {
        return Err(Error::TooManyReferences);
    }
    // 8.3.1: no two pictures of a coded video sequence share an order
    // count, and those the current picture predicts from are within 16
    // bits of it.
    let current = i64::from(params.poc);
    for reference in &params.references {
        if reference.poc == params.poc {
            return Err(Error::ReferenceAtCurrentPoc);
        }
        // Motion vector scaling uses it for short-term references alone
        // (8.5.3.2.8).
        let used = matches!(
            reference.membership,
            RpsMembership::StCurrBefore | RpsMembership::StCurrAfter
        );
        // In 64 bits two order counts' difference does not overflow;
        // were it to, it would be past 16 bits.
        let within = current
            .checked_sub(i64::from(reference.poc))
            .is_some_and(|difference| DIFF_POC.contains(&difference));
        if used && !within {
            return Err(Error::OutOfRange("DiffPicOrderCnt"));
        }
    }
    let first = params.header;
    for slice in slices {
        let header = slice.header;
        // 7.4.2.4.2: the active SPS does not change within a picture.
        if !(std::ptr::eq(&*header.sps, sps) || *header.sps == *sps) {
            return Err(Error::SpsChangedInPicture);
        }
        // 7.4.2.2 and 7.4.7.1: these are the same in every slice segment
        // of a picture.
        let agrees = header.nal_unit_type == first.nal_unit_type
            && header.slice_pic_order_cnt_lsb == first.slice_pic_order_cnt_lsb
            && header.short_term_rps == first.short_term_rps
            && header.long_term == first.long_term
            && header.slice_temporal_mvp_enabled_flag == first.slice_temporal_mvp_enabled_flag
            && header.pic_output_flag == first.pic_output_flag;
        if !agrees {
            return Err(Error::SliceSegmentsDisagree);
        }
        // 7.4.7.1: within the picture of the SPS the accelerator has.
        if header.slice_segment_address >= sps.pic_size_in_ctbs() {
            return Err(Error::OutOfRange("slice_segment_address"));
        }
        // 7.3.8.1: slice_segment_data() holds a CTU at least.
        if !data_follows(slice.nal, header.header_bytes) {
            return Err(Error::EmptySliceSegment);
        }
        // 7.4.7.1: lists of 15 entries at most, each a reference given.
        let in_range = |list: &Vec<usize>| {
            list.len() <= 15 && list.iter().all(|&i| i < params.references.len())
        };
        if !slice.ref_lists.iter().all(in_range) {
            return Err(Error::BadReferenceList);
        }
    }
    // 7.4.7.1: no two slice segments of a picture share an address.
    let mut addresses: Vec<u32> = slices
        .iter()
        .map(|s| s.header.slice_segment_address)
        .collect();
    addresses.sort_unstable();
    if addresses
        .windows(2)
        .any(|pair| matches!(pair, [a, b] if a == b))
    {
        return Err(Error::OverlappingSliceSegments);
    }
    Ok(())
}

/// Whether anything but `rbsp_slice_segment_trailing_bits()` follows the
/// first `header_bytes` bytes of the RBSP of `nal`, NAL unit header
/// included.
fn data_follows(nal: &[u8], header_bytes: u32) -> bool {
    let mut reader = BitReader::new(nal);
    header_bytes
        .checked_mul(8)
        .is_some_and(|bits| reader.skip(bits).is_ok())
        && reader.more_rbsp_data()
}
