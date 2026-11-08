//! The decoded picture buffer: the reference picture set's search and
//! marking (H.265 8.3.2), and the pictures' output latency (C.5.2.3). The
//! bumping process of output order (C.5.2.4) is the one H.264's DPB
//! shares ([`crate::codec::dpb`]). Pictures are known by a key of their own, so
//! that removals between the marking and the decode cannot mix them up.

use std::num::NonZeroU32;
use std::sync::Arc;

use super::rps::ShortTermRps;
use super::slice::LongTermPicture;
use crate::codec::dpb::Stored;

/// `PicOrderCntVal & (MaxPicOrderCntLsb - 1)`: the bits of an order count
/// below `MaxPicOrderCntLsb`, a power of two that divides 2^32, so that
/// they are the low bits of its two's complement.
pub(crate) fn poc_lsb(poc: i32, max_lsb: NonZeroU32) -> u32 {
    poc.cast_unsigned() % max_lsb
}

/// How a picture in the DPB is marked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Marking {
    Unused,
    ShortTerm,
    LongTerm,
}

/// Which of the current picture's reference picture sets a picture is
/// in (8.3.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RpsMembership {
    /// `RefPicSetStCurrBefore`: short-term, before the current picture,
    /// predicted from.
    StCurrBefore,
    /// `RefPicSetStCurrAfter`: short-term, after it, predicted from.
    StCurrAfter,
    /// `RefPicSetLtCurr`: long-term, predicted from.
    LtCurr,
    /// `RefPicSetStFoll` or `RefPicSetLtFoll`: kept for later pictures.
    Foll,
}

/// One picture in the DPB.
pub(crate) struct Entry<P, O> {
    /// Unique among the pictures the decoder has decoded.
    pub(crate) key: u64,
    pub(crate) picture: Arc<P>,
    /// `PicOrderCntVal`.
    pub(crate) poc: i32,
    pub(crate) marking: Marking,
    /// The set the current picture's RPS put it in, when it is in one.
    pub(crate) membership: Option<RpsMembership>,
    /// The frame still to be output, in display order.
    pub(crate) pending: Option<O>,
    /// `PicLatencyCount`.
    pub(crate) latency: u32,
}

impl<P, O> Stored for Entry<P, O> {
    type Output = O;

    fn order_count(&self) -> i32 {
        self.poc
    }

    fn is_reference(&self) -> bool {
        self.marking != Marking::Unused
    }

    fn pending(&self) -> Option<&O> {
        self.pending.as_ref()
    }

    fn take_pending(&mut self) -> Option<O> {
        self.pending.take()
    }
}

/// The pictures one picture's reference picture set names (8.3.2), by
/// key: `None` is "no reference picture".
#[derive(Debug, Default)]
pub(crate) struct RpsPictures {
    pub(crate) st_curr_before: Vec<Option<u64>>,
    pub(crate) st_curr_after: Vec<Option<u64>>,
    pub(crate) lt_curr: Vec<Option<u64>>,
    /// The `StFoll` and `LtFoll` pictures that are there.
    pub(crate) foll: Vec<u64>,
    /// The long-term pictures among them, current and following.
    long_term: Vec<u64>,
}

impl RpsPictures {
    /// Whether every picture the current one predicts from is there.
    pub(crate) fn current_complete(&self) -> bool {
        self.st_curr_before
            .iter()
            .chain(&self.st_curr_after)
            .chain(&self.lt_curr)
            .all(Option::is_some)
    }
}

/// The DPB: reference pictures, and in display order the pictures
/// waiting for output. `O` is what the output of a picture emits.
pub(crate) type Dpb<P, O> = crate::codec::dpb::Dpb<Entry<P, O>>;

impl<P, O> Dpb<P, O> {
    /// 8.3.2's search for the current picture's sets, at order count
    /// `poc`, without marking anything yet. Long-term pictures are found
    /// first, among all reference pictures, then short-term ones among
    /// the rest.
    pub(crate) fn find(
        &self,
        poc: i32,
        short_term: &ShortTermRps,
        long_term: &[LongTermPicture],
        max_poc_lsb: NonZeroU32,
    ) -> RpsPictures {
        let mut found = RpsPictures::default();
        let referenced = |e: &&Entry<P, O>| e.marking != Marking::Unused;
        for lt in long_term {
            // 8-5: PicOrderCntVal - DeltaPocMsbCycleLt * MaxPicOrderCntLsb
            // - (PicOrderCntVal & (MaxPicOrderCntLsb - 1)) + PocLsbLt. In
            // 64 bits nothing overflows, the cycle being in 32 bits and
            // MaxPicOrderCntLsb in 17; were it to, the count would be no
            // picture's.
            let target = lt.delta_poc_msb_cycle.map(|cycle| {
                i64::from(cycle)
                    .checked_mul(i64::from(max_poc_lsb.get()))
                    .and_then(|msb| i64::from(poc).checked_sub(msb))
                    .and_then(|count| count.checked_sub(i64::from(poc_lsb(poc, max_poc_lsb))))
                    .and_then(|count| count.checked_add(i64::from(lt.poc_lsb)))
            });
            let key = self
                .entries
                .iter()
                .filter(referenced)
                .find(|e| match target {
                    Some(full) => full == Some(i64::from(e.poc)),
                    None => poc_lsb(e.poc, max_poc_lsb) == lt.poc_lsb,
                })
                .map(|e| e.key);
            if let Some(key) = key {
                found.long_term.push(key);
            }
            if lt.used_by_current {
                found.lt_curr.push(key);
            } else if let Some(key) = key {
                found.foll.push(key);
            }
        }
        // PicOrderCntVal plus the picture's delta (8-5). Conforming streams
        // keep it within 32 bits; others' counts saturate (`order_count`),
        // and so does the sum.
        let short_term_key = |delta: i32| {
            self.entries
                .iter()
                .filter(referenced)
                .filter(|e| !found.long_term.contains(&e.key))
                .find(|e| e.poc == poc.saturating_add(delta))
                .map(|e| e.key)
        };
        let mut st_foll = Vec::new();
        for (sets, current) in [
            (&short_term.negative, &mut found.st_curr_before),
            (&short_term.positive, &mut found.st_curr_after),
        ] {
            for &(delta, used) in sets {
                let key = short_term_key(delta);
                if used {
                    current.push(key);
                } else if let Some(key) = key {
                    st_foll.push(key);
                }
            }
        }
        found.foll.extend(st_foll);
        found
    }

    /// 8.3.2's marking: the long-term pictures found are marked long-term,
    /// the short-term ones stay short-term, and every other picture is
    /// unused for reference. Each picture found records its set.
    pub(crate) fn mark(&mut self, found: &RpsPictures) {
        let sets = [
            (&found.st_curr_before, RpsMembership::StCurrBefore),
            (&found.st_curr_after, RpsMembership::StCurrAfter),
            (&found.lt_curr, RpsMembership::LtCurr),
        ];
        for entry in &mut self.entries {
            let key = entry.key;
            entry.membership = sets
                .iter()
                .find(|(set, _)| set.contains(&Some(key)))
                .map(|&(_, membership)| membership)
                .or_else(|| found.foll.contains(&key).then_some(RpsMembership::Foll));
            entry.marking = match entry.membership {
                None => Marking::Unused,
                Some(_) if found.long_term.contains(&key) => Marking::LongTerm,
                Some(_) => Marking::ShortTerm,
            };
        }
    }

    /// Marks every picture unused for reference, as an IRAP picture with
    /// `NoRaslOutputFlag` does.
    pub(crate) fn unmark_all(&mut self) {
        for entry in &mut self.entries {
            entry.marking = Marking::Unused;
            entry.membership = None;
        }
    }

    /// Whether a waiting picture has reached `SpsMaxLatencyPictures`.
    pub(crate) fn latency_reached(&self, max: Option<u32>) -> bool {
        max.is_some_and(|max| {
            self.entries
                .iter()
                .any(|e| e.pending.is_some() && e.latency >= max)
        })
    }
}
