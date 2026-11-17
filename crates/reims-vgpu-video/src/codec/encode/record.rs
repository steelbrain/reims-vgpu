//! A stream's configuration record (`avcC`, `hvcC`) as the parameter sets
//! its hardware writes make it, and each access unit's NAL units as a
//! sample carries them (ISO/IEC 14496-15): each behind a four-byte length,
//! without parameter sets and access unit delimiters ([`Record`]).
//!
//! The record starts from the stream's first parameter sets: the ones the
//! VA-API backend's encoder writes itself ([`super::encoder`]), or the ones
//! an encoder that writes its own gives before its first frame, as NVENC
//! does ([`Record::of_annex_b`]). An IDR picture's access unit whose
//! parameter sets differ from the record's makes a new record of them,
//! which goes out with that picture's sample. The record goes out with the
//! first sample too.

use std::sync::Arc;

use super::{Codec, Unsupported, h264, hevc, nal};

/// What a NAL unit is to a sample.
enum Role {
    /// A parameter set, by its index in the record's order (VPS, SPS, PPS
    /// for HEVC; SPS, PPS for H.264).
    ParameterSet(usize),
    Delimiter,
    Slice,
    Other,
}

fn unit_role(codec: Codec, unit: &[u8]) -> Role {
    let first = unit.first().copied().unwrap_or(0);
    match codec {
        Codec::H264 => match first & 0x1f {
            h264::nal_type::SPS => Role::ParameterSet(0),
            h264::nal_type::PPS => Role::ParameterSet(1),
            h264::nal_type::AUD => Role::Delimiter,
            1..=5 => Role::Slice,
            _ => Role::Other,
        },
        Codec::Hevc => match (first >> 1) & 0x3f {
            hevc::nal_type::VPS => Role::ParameterSet(0),
            hevc::nal_type::SPS => Role::ParameterSet(1),
            hevc::nal_type::PPS => Role::ParameterSet(2),
            hevc::nal_type::AUD => Role::Delimiter,
            0..=31 => Role::Slice,
            _ => Role::Other,
        },
    }
}

/// The configuration record of `codec`'s parameter set units in the
/// record's order; `None` where they are not the codec's sets or do not
/// read.
fn record_of(codec: Codec, units: &[Vec<u8>]) -> Option<Vec<u8>> {
    match (codec, units) {
        (Codec::H264, [sps, pps]) => Some(h264::avcc(sps, pps)),
        (Codec::Hevc, [vps, sps, pps]) => hevc::hvcc(vps, sps, pps),
        _ => None,
    }
}

/// A stream's configuration record, and the parameter set units in it
/// (module doc).
#[derive(Clone, Debug)]
pub struct Record {
    codec: Codec,
    record: Arc<[u8]>,
    units: Vec<Vec<u8>>,
    /// Whether the record goes out with the next sample.
    due: bool,
}

impl Record {
    /// The record of `units`, `codec`'s parameter sets in the record's
    /// order, escaped NAL units with their headers, due with the first
    /// sample; `None` where they make none.
    pub fn new(codec: Codec, units: Vec<Vec<u8>>) -> Option<Self> {
        let record = record_of(codec, &units)?;
        Some(Self {
            codec,
            record: record.into(),
            units,
            due: true,
        })
    }

    /// The record of the parameter sets in `stream`, Annex B, as an encoder
    /// that writes its own gives them before its first frame (NVENC's
    /// `sequence_parameters`); `None` where they make none.
    pub fn of_annex_b(codec: Codec, stream: &[u8]) -> Option<Self> {
        Self::new(codec, parameter_sets(codec, stream).0)
    }

    /// The record of the parameter sets in use.
    pub fn current(&self) -> Arc<[u8]> {
        Arc::clone(&self.record)
    }

    /// The record where it goes out with this sample: the first, and the
    /// first after it changed.
    pub fn take_due(&mut self) -> Option<Arc<[u8]>> {
        let due = std::mem::take(&mut self.due);
        due.then(|| self.current())
    }

    /// The NAL units of `access_unit`, Annex B, as a sample carries them:
    /// each behind a four-byte length, without parameter sets and
    /// delimiters. Where it is an IDR picture's (`idr`) and carries a whole
    /// set of parameter sets that differs from the record's, they make the
    /// new record, which is due. `Err` where it holds no slice, or its
    /// parameter sets make no record.
    pub fn sample_data(&mut self, access_unit: &[u8], idr: bool) -> Result<Vec<u8>, Unsupported> {
        let (seen, data, sliced) = parameter_sets(self.codec, access_unit);
        if !sliced {
            return Err(Unsupported("the hardware wrote no slice".into()));
        }
        if idr && !seen.is_empty() && seen.iter().all(|u| !u.is_empty()) && seen != self.units {
            let record = record_of(self.codec, &seen)
                .ok_or_else(|| Unsupported("the hardware's parameter sets do not read".into()))?;
            self.record = record.into();
            self.units = seen;
            self.due = true;
        }
        Ok(data)
    }
}

/// The parameter sets of `stream`, Annex B, in the record's order (an
/// empty unit for a set it lacks before one it has, the last of each kind
/// kept); its other units but delimiters, each behind a four-byte length;
/// and whether it holds a slice.
fn parameter_sets(codec: Codec, stream: &[u8]) -> (Vec<Vec<u8>>, Vec<u8>, bool) {
    let mut data = Vec::with_capacity(stream.len());
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let mut sliced = false;
    for unit in nal::split_annex_b(stream) {
        match unit_role(codec, unit) {
            Role::ParameterSet(index) => {
                while seen.len() <= index {
                    seen.push(Vec::new());
                }
                if let Some(set) = seen.get_mut(index) {
                    *set = unit.to_vec();
                }
            }
            Role::Delimiter => {}
            Role::Slice => {
                sliced = true;
                nal::push_length_prefixed(&mut data, unit);
            }
            Role::Other => nal::push_length_prefixed(&mut data, unit),
        }
    }
    (seen, data, sliced)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::encode::nal::START_CODE;

    /// `units` as an Annex B stream.
    fn annex_b(units: &[&[u8]]) -> Vec<u8> {
        let mut stream = Vec::new();
        for unit in units {
            stream.extend_from_slice(&START_CODE);
            stream.extend_from_slice(unit);
        }
        stream
    }

    /// `units`, each behind its four-byte length.
    fn sample(units: &[&[u8]]) -> Vec<u8> {
        let mut data = Vec::new();
        for unit in units {
            nal::push_length_prefixed(&mut data, unit);
        }
        data
    }

    /// An H.264 SPS's first bytes, of Main at `level_idc`: all the record
    /// reads of it.
    fn sps(level_idc: u8) -> Vec<u8> {
        vec![0x67, 77, 0x40, level_idc, 0xac, 0x2b]
    }

    const PPS: &[u8] = &[0x68, 0xce, 0x3c, 0x80];
    const AUD: &[u8] = &[0x09, 0xf0];
    const SEI: &[u8] = &[0x06, 0x05, 0x01, 0x00, 0x80];
    const IDR: &[u8] = &[0x65, 0x88, 0x84];
    const P: &[u8] = &[0x41, 0x9a, 0x02];

    /// The record of NVENC's parameter sets goes out with the first sample
    /// alone; each sample holds its units but parameter sets and
    /// delimiters, length-prefixed, in order.
    #[test]
    fn the_first_record_goes_with_the_first_sample_and_samples_lose_their_sets() {
        let sps = sps(31);
        let mut record = Record::of_annex_b(Codec::H264, &annex_b(&[&sps, PPS])).expect("a record");
        assert_eq!(&record.current()[..], &h264::avcc(&sps, PPS)[..]);
        let first = annex_b(&[AUD, &sps, PPS, SEI, IDR]);
        assert_eq!(record.sample_data(&first, true), Ok(sample(&[SEI, IDR])));
        assert_eq!(record.take_due(), Some(record.current()));
        assert_eq!(
            record.sample_data(&annex_b(&[AUD, P]), false),
            Ok(sample(&[P]))
        );
        assert_eq!(record.take_due(), None);
    }

    /// An IDR picture whose parameter sets differ makes a new record, due
    /// with its sample; the same sets, or new ones before a picture that is
    /// no IDR picture's, make none. An access unit with no slice is
    /// refused.
    #[test]
    fn an_idr_pictures_new_parameter_sets_make_a_new_record() {
        let (old, new) = (sps(31), sps(40));
        let mut record =
            Record::new(Codec::H264, vec![old.clone(), PPS.to_vec()]).expect("a record");
        assert!(record.take_due().is_some());
        assert_eq!(
            record.sample_data(&annex_b(&[&old, PPS, IDR]), true),
            Ok(sample(&[IDR]))
        );
        assert_eq!(record.take_due(), None);
        assert_eq!(
            record.sample_data(&annex_b(&[&new, PPS, P]), false),
            Ok(sample(&[P]))
        );
        assert_eq!(record.take_due(), None);
        assert_eq!(
            record.sample_data(&annex_b(&[&new, PPS, IDR]), true),
            Ok(sample(&[IDR]))
        );
        let due = record.take_due().expect("a new record");
        assert_eq!(&due[..], &h264::avcc(&new, PPS)[..]);
        assert_eq!(&record.current()[..], &due[..]);
        assert!(
            record
                .sample_data(&annex_b(&[&new, PPS, SEI]), true)
                .is_err()
        );
    }

    /// Units that are not a codec's whole set of parameter sets make no
    /// record.
    #[test]
    fn units_that_are_not_the_codecs_sets_make_no_record() {
        assert!(Record::new(Codec::H264, vec![sps(31)]).is_none());
        assert!(Record::new(Codec::Hevc, vec![sps(31), PPS.to_vec()]).is_none());
        assert!(Record::of_annex_b(Codec::H264, &annex_b(&[IDR])).is_none());
    }
}
