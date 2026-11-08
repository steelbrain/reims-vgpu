//! Picture order counts (H.264 8.2.1), for frames: types 0, 1 and 2.

use super::Error;
use super::slice::SliceHeader;
use super::sps::Sps;
use crate::codec::num::ToUsize;

/// What the order count of one picture depends on from the pictures
/// before it.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PocState {
    /// `prevPicOrderCntMsb` and `prevPicOrderCntLsb` (type 0): from the
    /// previous reference picture.
    prev_msb: i32,
    prev_lsb: i32,
    /// `prevFrameNumOffset` and `prevFrameNum` (types 1 and 2): from the
    /// previous picture.
    prev_frame_num_offset: i32,
    prev_frame_num: u32,
}

/// A picture's order counts, and what the next picture's need of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PicOrder {
    pub(crate) top: i32,
    pub(crate) bottom: i32,
    /// `PicOrderCntMsb` (type 0).
    msb: i32,
    /// `FrameNumOffset` (types 1 and 2).
    frame_num_offset: i32,
}

impl PicOrder {
    /// `PicOrderCnt()` of a frame (8-1).
    pub(crate) fn frame(&self) -> i32 {
        self.top.min(self.bottom)
    }
}

impl PocState {
    /// The order counts of the picture `header` starts.
    pub(crate) fn compute(&self, sps: &Sps, header: &SliceHeader) -> Result<PicOrder, Error> {
        match sps.pic_order_cnt_type {
            0 => self.type0(sps, header),
            1 => self.type1(sps, header),
            _ => self.type2(sps, header),
        }
    }

    /// 8.2.1.1.
    fn type0(&self, sps: &Sps, header: &SliceHeader) -> Result<PicOrder, Error> {
        let (prev_msb, prev_lsb) = if header.idr {
            (0, 0)
        } else {
            (self.prev_msb, self.prev_lsb)
        };
        let max = i32::try_from(sps.max_pic_order_cnt_lsb()).unwrap_or(i32::MAX);
        let lsb = i32::try_from(header.pic_order_cnt_lsb).unwrap_or(0);
        // (8-3), in 64 bits: after operation 5, `prevPicOrderCntLsb` is
        // the last picture's TopFieldOrderCnt, which may be anywhere in 32
        // bits, so the differences of two take 33.
        let difference = |a: i32, b: i32| {
            i64::from(a)
                .checked_sub(i64::from(b))
                .ok_or(Error::Overflow("PicOrderCntMsb"))
        };
        let half = i64::from(max / 2);
        let msb = if lsb < prev_lsb && difference(prev_lsb, lsb)? >= half {
            prev_msb.wrapping_add(max)
        } else if lsb > prev_lsb && difference(lsb, prev_lsb)? > half {
            prev_msb.wrapping_sub(max)
        } else {
            prev_msb
        };
        let top = msb.wrapping_add(lsb);
        Ok(PicOrder {
            top,
            bottom: top.wrapping_add(header.delta_pic_order_cnt_bottom),
            msb,
            frame_num_offset: 0,
        })
    }

    /// `FrameNumOffset` (8-6, 8-11).
    fn frame_num_offset(&self, sps: &Sps, header: &SliceHeader) -> i32 {
        if header.idr {
            0
        } else if self.prev_frame_num > header.frame_num {
            self.prev_frame_num_offset
                .wrapping_add(i32::try_from(sps.max_frame_num()).unwrap_or(0))
        } else {
            self.prev_frame_num_offset
        }
    }

    /// 8.2.1.2. In 64 bits nothing overflows: absFrameNum is below
    /// 2^31 + 2^16 (FrameNumOffset is 32 bits, frame_num 16), and each of
    /// its cycles adds `cycle` offsets of below 2^31, so the sum stays
    /// below 2^62 + 2^47. The checks below refuse the stream should that
    /// ever not hold.
    fn type1(&self, sps: &Sps, header: &SliceHeader) -> Result<PicOrder, Error> {
        let frame_num_offset = self.frame_num_offset(sps, header);
        let overflow = || Error::Overflow("expectedPicOrderCnt");
        let sum = |offsets: &[i32]| {
            offsets
                .iter()
                .try_fold(0i64, |sum, &o| sum.checked_add(i64::from(o)))
                .ok_or_else(overflow)
        };
        let cycle = i64::try_from(sps.offset_for_ref_frame.len()).map_err(|_| overflow())?;
        // (8-7), and (8-8) for a non-reference picture.
        let mut abs_frame_num = if cycle != 0 {
            i64::from(frame_num_offset)
                .checked_add(i64::from(header.frame_num))
                .ok_or(Error::Overflow("absFrameNum"))?
        } else {
            0
        };
        if header.nal_ref_idc == 0 && abs_frame_num > 0 {
            abs_frame_num = abs_frame_num
                .checked_sub(1)
                .ok_or(Error::Overflow("absFrameNum"))?;
        }
        // (8-9) to (8-11).
        let mut expected = 0i64;
        if let Some(before) = abs_frame_num.checked_sub(1).filter(|_| abs_frame_num > 0) {
            let delta_per_cycle = sum(&sps.offset_for_ref_frame)?;
            let cycle_count = before.checked_div(cycle).ok_or_else(overflow)?;
            let in_cycle = before
                .checked_rem(cycle)
                .ok_or_else(overflow)?
                .unsigned_abs()
                .to_usize();
            let in_cycle = sum(sps
                .offset_for_ref_frame
                .get(..=in_cycle)
                .ok_or_else(overflow)?)?;
            expected = cycle_count
                .checked_mul(delta_per_cycle)
                .and_then(|cycles| cycles.checked_add(in_cycle))
                .ok_or_else(overflow)?;
        }
        if header.nal_ref_idc == 0 {
            expected = expected
                .checked_add(i64::from(sps.offset_for_non_ref_pic))
                .ok_or_else(overflow)?;
        }
        // (8-12) and (8-13).
        let [delta_top, delta_bottom] = header.delta_pic_order_cnt;
        let top = expected
            .checked_add(i64::from(delta_top))
            .ok_or(Error::Overflow("TopFieldOrderCnt"))?;
        let bottom = top
            .checked_add(i64::from(sps.offset_for_top_to_bottom_field))
            .and_then(|bottom| bottom.checked_add(i64::from(delta_bottom)))
            .ok_or(Error::Overflow("BottomFieldOrderCnt"))?;
        Ok(PicOrder {
            top: saturate_i32(top),
            bottom: saturate_i32(bottom),
            msb: 0,
            frame_num_offset,
        })
    }

    /// 8.2.1.3, in 64 bits, where nothing of 32-bit FrameNumOffset and a
    /// 16-bit frame_num overflows.
    fn type2(&self, sps: &Sps, header: &SliceHeader) -> Result<PicOrder, Error> {
        let frame_num_offset = self.frame_num_offset(sps, header);
        let count = if header.idr {
            0
        } else {
            // (8-14).
            let doubled = i64::from(frame_num_offset)
                .checked_add(i64::from(header.frame_num))
                .and_then(|frames| frames.checked_mul(2));
            let count = if header.nal_ref_idc == 0 {
                doubled.and_then(|doubled| doubled.checked_sub(1))
            } else {
                doubled
            };
            count.ok_or(Error::Overflow("tempPicOrderCnt"))?
        };
        Ok(PicOrder {
            top: saturate_i32(count),
            bottom: saturate_i32(count),
            msb: 0,
            frame_num_offset,
        })
    }

    /// Records a decoded picture for the next one's counts. `mmco5` says
    /// whether its marking held operation 5, and `top_after_mmco5` is its
    /// `TopFieldOrderCnt` once 8.2.1 has subtracted `tempPicOrderCnt`.
    pub(crate) fn finish(
        &mut self,
        header: &SliceHeader,
        order: &PicOrder,
        mmco5: bool,
        top_after_mmco5: i32,
    ) {
        if header.nal_ref_idc != 0 {
            if mmco5 {
                self.prev_msb = 0;
                self.prev_lsb = top_after_mmco5;
            } else {
                self.prev_msb = order.msb;
                self.prev_lsb = i32::try_from(header.pic_order_cnt_lsb).unwrap_or(0);
            }
        }
        if mmco5 {
            self.prev_frame_num_offset = 0;
            self.prev_frame_num = 0;
        } else {
            self.prev_frame_num_offset = order.frame_num_offset;
            self.prev_frame_num = header.frame_num;
        }
    }

    /// Records a frame the gap process of 8.2.5.2 inferred.
    pub(crate) fn skip_frame(&mut self, frame_num: u32, max_frame_num: u32) {
        if self.prev_frame_num > frame_num {
            self.prev_frame_num_offset = self
                .prev_frame_num_offset
                .wrapping_add(i32::try_from(max_frame_num).unwrap_or(0));
        }
        self.prev_frame_num = frame_num;
    }
}

fn saturate_i32(value: i64) -> i32 {
    // Conforming streams stay within 32 bits (8.2.1). Others saturate
    // rather than fail, since only output order depends on the counts.
    i32::try_from(value).unwrap_or(if value < 0 { i32::MIN } else { i32::MAX })
}
