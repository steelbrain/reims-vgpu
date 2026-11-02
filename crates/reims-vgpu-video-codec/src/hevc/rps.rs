//! Short-term reference picture sets (H.265 7.3.7 and 7.4.8):
//! `st_ref_pic_set()`, coded outright or predicted from an earlier set
//! (inter RPS prediction), in an SPS's list or in a slice header.

use super::Error;
use super::bits::BitReader;
use crate::num::ToUsize;

/// One short-term reference picture set, its pictures as POC deltas from
/// the current picture's.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShortTermRps {
    /// `DeltaPocS0[]` with `UsedByCurrPicS0[]`: the pictures before the
    /// current one, nearest first, so each delta below the last.
    pub(crate) negative: Vec<(i32, bool)>,
    /// `DeltaPocS1[]` with `UsedByCurrPicS1[]`: the pictures after it,
    /// nearest first.
    pub(crate) positive: Vec<(i32, bool)>,
}

/// A predicted picture's `used_by_curr_pic_flag` and whether its
/// `use_delta_flag` keeps it (inferred 1 where the first is set).
#[derive(Clone, Copy, Debug)]
struct Flags {
    used: bool,
    kept: bool,
}

impl ShortTermRps {
    /// `DeltaPocS0[]` with `UsedByCurrPicS0[]` (7.4.8): the pictures
    /// before the current one, nearest first, as the set's syntax derives
    /// them, predicted or not.
    pub fn negative(&self) -> &[(i32, bool)] {
        &self.negative
    }

    /// `DeltaPocS1[]` with `UsedByCurrPicS1[]`: the pictures after it,
    /// nearest first.
    pub fn positive(&self) -> &[(i32, bool)] {
        &self.positive
    }

    /// `NumDeltaPocs`.
    pub(crate) fn num_delta_pocs(&self) -> usize {
        self.negative.iter().chain(&self.positive).count()
    }

    /// The pictures the current picture predicts from: those flagged
    /// `UsedByCurrPicS0` and `UsedByCurrPicS1`.
    pub(crate) fn used_by_current(&self) -> usize {
        self.negative
            .iter()
            .chain(&self.positive)
            .filter(|&&(_, used)| used)
            .count()
    }

    /// `st_ref_pic_set(stRpsIdx)` (7.3.7). `earlier` holds the sets
    /// before it, so that `stRpsIdx` is their count: the SPS's sets so far,
    /// or all of them for the slice header's own set, which
    /// `in_slice_header` says this is. `max_pictures` is the SPS's
    /// `sps_max_dec_pic_buffering_minus1` of its highest sub-layer: the
    /// pictures a set names besides the current one, which 7.4.8 holds a
    /// coded set to, and which the DPB holds a predicted one to, so that
    /// every set is bounded where it is parsed.
    pub(crate) fn parse(
        r: &mut BitReader<'_>,
        earlier: &[ShortTermRps],
        in_slice_header: bool,
        max_pictures: u32,
    ) -> Result<Self, Error> {
        Self::parse_with_reference(r, earlier, in_slice_header, max_pictures).map(|(set, _)| set)
    }

    /// [`Self::parse`], with `RefRpsIdx` (7-59), the index in `earlier`
    /// of the set this one is predicted from, when it is.
    pub(crate) fn parse_with_reference(
        r: &mut BitReader<'_>,
        earlier: &[ShortTermRps],
        in_slice_header: bool,
        max_pictures: u32,
    ) -> Result<(Self, Option<usize>), Error> {
        let index = earlier.len();
        let predicted = index != 0 && r.bit()?;
        if !predicted {
            return Ok((Self::parse_explicit(r, max_pictures)?, None));
        }
        // 7-59: RefRpsIdx = stRpsIdx - (delta_idx_minus1 + 1), the delta
        // coded only in a slice header's set, and 0 elsewhere.
        let last = index
            .checked_sub(1)
            .ok_or(Error::OutOfRange("delta_idx_minus1"))?;
        let delta_idx_minus1 = if in_slice_header {
            r.ue_max(u32::try_from(last).unwrap_or(0), "delta_idx_minus1")?
                .to_usize()
        } else {
            0
        };
        let reference_index = last
            .checked_sub(delta_idx_minus1)
            .ok_or(Error::OutOfRange("delta_idx_minus1"))?;
        let reference = earlier
            .get(reference_index)
            .ok_or(Error::OutOfRange("delta_idx_minus1"))?;
        // 7-60: deltaRps = (1 - 2 * delta_rps_sign) * (abs_delta_rps_minus1 + 1).
        let sign = r.bit()?;
        let magnitude = i32::from(r.ue_u16(0x7fff, "abs_delta_rps_minus1")?)
            .checked_add(1)
            .ok_or(Error::Overflow("deltaRps"))?;
        let delta_rps = if sign {
            magnitude.checked_neg().ok_or(Error::Overflow("deltaRps"))?
        } else {
            magnitude
        };
        // used_by_curr_pic_flag[j] and use_delta_flag[j], j from 0 to
        // NumDeltaPocs[RefRpsIdx]: the reference's pictures, negative
        // first, then the reference picture itself.
        let mut flags = || -> Result<Flags, Error> {
            let used = r.bit()?;
            Ok(Flags {
                used,
                kept: used || r.bit()?,
            })
        };
        let mut with_flags = |pictures: &[(i32, bool)]| {
            pictures
                .iter()
                .map(|&(delta, _)| Ok((delta, flags()?)))
                .collect::<Result<Vec<_>, Error>>()
        };
        let s0 = with_flags(&reference.negative)?;
        let s1 = with_flags(&reference.positive)?;
        let own = flags()?;
        let set = Self::predict(delta_rps, &s0, &s1, own)?;
        if set.num_delta_pocs() > max_pictures.to_usize() {
            return Err(Error::OutOfRange("NumDeltaPocs"));
        }
        Ok((set, Some(reference_index)))
    }

    /// 7.4.8's derivation, equations 7-61 and 7-62: the reference set's
    /// pictures, `s0` before and `s1` after, shifted by `delta_rps`, and
    /// the reference picture itself (`own`), each kept where its
    /// `use_delta_flag` says, sorted into the two lists.
    fn predict(
        delta_rps: i32,
        s0: &[(i32, Flags)],
        s1: &[(i32, Flags)],
        own: Flags,
    ) -> Result<Self, Error> {
        // dPoc = DeltaPocS0/S1[RefRpsIdx][i] + deltaRps.
        let shifted = |pictures: &[(i32, Flags)]| -> Result<Vec<(i32, Flags)>, Error> {
            pictures
                .iter()
                .filter(|(_, flags)| flags.kept)
                .map(|&(delta, flags)| {
                    let d_poc = delta
                        .checked_add(delta_rps)
                        .ok_or(Error::Overflow("dPoc"))?;
                    Ok((d_poc, flags))
                })
                .collect()
        };
        let (s0, s1) = (shifted(s0)?, shifted(s1)?);
        let own = own.kept.then_some((delta_rps, own));
        let negative = s1
            .iter()
            .rev()
            .chain(&own)
            .chain(&s0)
            .filter(|&&(poc, _)| poc < 0);
        let positive = s0
            .iter()
            .rev()
            .chain(&own)
            .chain(&s1)
            .filter(|&&(poc, _)| poc > 0);
        let used = |&(poc, flags): &(i32, Flags)| (poc, flags.used);
        Ok(Self {
            negative: negative.map(used).collect(),
            positive: positive.map(used).collect(),
        })
    }

    /// A set coded outright: `num_negative_pics`, `num_positive_pics`, then
    /// each picture's distance from the one before it (7-63 to 7-66).
    fn parse_explicit(r: &mut BitReader<'_>, max_pictures: u32) -> Result<Self, Error> {
        let negatives = r.ue_max(max_pictures, "num_negative_pics")?;
        let left = max_pictures
            .checked_sub(negatives)
            .ok_or(Error::OutOfRange("num_negative_pics"))?;
        let positives = r.ue_max(left, "num_positive_pics")?;
        // Each picture's distance from the one before, plus one: at most
        // 2^15 a picture, of 16 pictures at most, so that DeltaPocS0 and
        // DeltaPocS1 stay within 20 bits.
        let mut negative = Vec::new();
        let mut poc = 0i32;
        for _ in 0..negatives {
            let minus1 = i32::from(r.ue_u16(0x7fff, "delta_poc_s0_minus1")?);
            poc = poc
                .checked_sub(minus1)
                .and_then(|poc| poc.checked_sub(1))
                .ok_or(Error::Overflow("DeltaPocS0"))?;
            negative.push((poc, r.bit()?));
        }
        let mut positive = Vec::new();
        poc = 0;
        for _ in 0..positives {
            let minus1 = i32::from(r.ue_u16(0x7fff, "delta_poc_s1_minus1")?);
            poc = poc
                .checked_add(minus1)
                .and_then(|poc| poc.checked_add(1))
                .ok_or(Error::Overflow("DeltaPocS1"))?;
            positive.push((poc, r.bit()?));
        }
        Ok(Self { negative, positive })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hevc::tests::BitWriter;

    fn explicit(w: &mut BitWriter, negative: &[(u32, bool)], positive: &[(u32, bool)]) {
        w.ue(negative.len() as u32).ue(positive.len() as u32);
        for &(minus1, used) in negative.iter().chain(positive) {
            w.ue(minus1).bit(used);
        }
    }

    #[test]
    fn explicit_sets_accumulate_their_deltas() -> Result<(), Error> {
        let mut w = BitWriter::new();
        explicit(&mut w, &[(0, true), (1, false)], &[(3, true)]);
        let bytes = w.finish();
        let set = ShortTermRps::parse(&mut BitReader::new(&bytes), &[], false, 15)?;
        assert_eq!(set.negative, [(-1, true), (-3, false)]);
        assert_eq!(set.positive, [(4, true)]);
        assert_eq!(set.used_by_current(), 2);
        Ok(())
    }

    #[test]
    fn a_predicted_set_shifts_its_reference() -> Result<(), Error> {
        // The reference: -1, -3 and +2. Predicted with deltaRps -1, every
        // picture kept: -2, -4, +1, and the reference picture itself, -1.
        let reference = ShortTermRps {
            negative: vec![(-1, true), (-3, true)],
            positive: vec![(2, true)],
        };
        let mut w = BitWriter::new();
        w.bit(true); // inter_ref_pic_set_prediction_flag
        w.bit(true).ue(0); // delta_rps_sign, abs_delta_rps_minus1: -1
        // j = 0 (-1), 1 (-3), 2 (+2), 3 (the reference picture): used,
        // used, used, and the last not used but kept.
        w.bit(true).bit(true).bit(true).bit(false).bit(true);
        let bytes = w.finish();
        let set = ShortTermRps::parse(&mut BitReader::new(&bytes), &[reference], false, 15)?;
        assert_eq!(set.negative, [(-1, false), (-2, true), (-4, true)]);
        assert_eq!(set.positive, [(1, true)]);
        Ok(())
    }

    #[test]
    fn a_slice_headers_set_names_its_reference() -> Result<(), Error> {
        // Two SPS sets; the slice header's (index 2) predicts from set 0
        // (delta_idx_minus1 1), shifted +2, dropping the shifted -1.
        let sets = [
            ShortTermRps {
                negative: vec![(-1, true), (-2, true)],
                positive: vec![],
            },
            ShortTermRps::default(),
        ];
        let mut w = BitWriter::new();
        w.bit(true).ue(1); // predicted, delta_idx_minus1 1
        w.bit(false).ue(1); // deltaRps +2
        // j = 0 (-1 -> +1): not used, dropped; j = 1 (-2 -> 0): dropped;
        // j = 2 (the reference picture, +2): used.
        w.bit(false).bit(false).bit(false).bit(false).bit(true);
        let bytes = w.finish();
        let (set, reference) =
            ShortTermRps::parse_with_reference(&mut BitReader::new(&bytes), &sets, true, 15)?;
        assert_eq!(set.negative, []);
        assert_eq!(set.positive, [(2, true)]);
        assert_eq!(reference, Some(0));
        Ok(())
    }
}
