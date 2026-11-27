//! OBUs (AV1 5.3 and 6.2): their headers, and the OBUs of a sample in the low overhead
//! bitstream format (5.2), which MP4 samples and IVF frames carry. An OBU may leave out its
//! size only where it is its sample's last, whose size is then the sample's rest (AV1-ISOBMFF
//! 2.4).

use super::Error;
use super::bits::leb128;
use crate::codec::num::ToUsize;

/// `OBU_SEQUENCE_HEADER`.
pub const SEQUENCE_HEADER: u8 = 1;
/// `OBU_TEMPORAL_DELIMITER`.
pub const TEMPORAL_DELIMITER: u8 = 2;
/// `OBU_FRAME_HEADER`.
pub const FRAME_HEADER: u8 = 3;
/// `OBU_TILE_GROUP`.
pub const TILE_GROUP: u8 = 4;
/// `OBU_METADATA`.
pub const METADATA: u8 = 5;
/// `OBU_FRAME`.
pub const FRAME: u8 = 6;
/// `OBU_REDUNDANT_FRAME_HEADER`.
pub const REDUNDANT_FRAME_HEADER: u8 = 7;
/// `OBU_TILE_LIST`.
pub const TILE_LIST: u8 = 8;
/// `OBU_PADDING`.
pub const PADDING: u8 = 15;

/// `obu_extension_header()` (5.3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extension {
    /// `temporal_id`.
    pub temporal_id: u8,
    /// `spatial_id`.
    pub spatial_id: u8,
}

/// One OBU of a sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Obu<'a> {
    /// `obu_type`.
    pub obu_type: u8,
    /// Its extension header, where `obu_extension_flag` is set.
    pub extension: Option<Extension>,
    /// Its payload, `obu_size` bytes.
    pub payload: &'a [u8],
}

impl Obu<'_> {
    /// Whether the operating point whose `operating_point_idc` is `idc` drops the OBU
    /// (`drop_obu()`, 6.2.1): an OBU of a layer the point does not hold. Sequence headers and
    /// temporal delimiters, and OBUs without an extension, are every point's.
    pub fn dropped_by(&self, idc: u16) -> bool {
        if idc == 0 || matches!(self.obu_type, SEQUENCE_HEADER | TEMPORAL_DELIMITER) {
            return false;
        }
        let Some(extension) = self.extension else {
            return false;
        };
        let in_temporal_layer = (idc >> extension.temporal_id) & 1 == 1;
        // spatial_id is at most 3, so the shift is at most 11.
        let in_spatial_layer = (idc >> extension.spatial_id.saturating_add(8)) & 1 == 1;
        !in_temporal_layer || !in_spatial_layer
    }
}

/// The OBUs of `sample`, in order; the first error ends them.
pub fn obus(sample: &[u8]) -> Obus<'_> {
    Obus {
        rest: sample,
        failed: false,
    }
}

/// The iterator [`obus`] returns.
pub struct Obus<'a> {
    rest: &'a [u8],
    failed: bool,
}

impl<'a> Iterator for Obus<'a> {
    type Item = Result<Obu<'a>, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.rest.is_empty() {
            return None;
        }
        let obu = self.read();
        self.failed = obu.is_err();
        Some(obu)
    }
}

impl<'a> Obus<'a> {
    fn read(&mut self) -> Result<Obu<'a>, Error> {
        let (&first, rest) = self.rest.split_first().ok_or(Error::Truncated)?;
        if first & 0x80 != 0 {
            return Err(Error::OutOfRange("obu_forbidden_bit"));
        }
        let obu_type = (first >> 3) & 0xf;
        let has_extension = first & 0x04 != 0;
        let has_size = first & 0x02 != 0;
        let (extension, rest) = if has_extension {
            let (&byte, rest) = rest.split_first().ok_or(Error::Truncated)?;
            let extension = Extension {
                temporal_id: byte >> 5,
                spatial_id: (byte >> 3) & 3,
            };
            (Some(extension), rest)
        } else {
            (None, rest)
        };
        let (payload, rest) = if has_size {
            let (size, taken) = leb128(rest)?;
            let rest = rest.get(taken..).ok_or(Error::Truncated)?;
            if size.to_usize() > rest.len() {
                return Err(Error::ObuPastEnd);
            }
            rest.split_at(size.to_usize())
        } else {
            (rest, &[][..])
        };
        self.rest = rest;
        Ok(Obu {
            obu_type,
            extension,
            payload,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sample_splits_into_its_obus_and_the_last_may_leave_out_its_size() {
        // A temporal delimiter with a size of 0; a frame header OBU with an extension
        // (temporal 2, spatial 1) and a size of 2; a padding OBU without a size field.
        let sample = [
            0x12,
            0x00, //
            0x1e,
            0b0100_1000,
            0x02,
            0xaa,
            0xbb, //
            0x78,
            0x01,
            0x02,
            0x03,
        ];
        let got: Vec<_> = obus(&sample).collect();
        assert_eq!(
            got,
            vec![
                Ok(Obu {
                    obu_type: TEMPORAL_DELIMITER,
                    extension: None,
                    payload: &[],
                }),
                Ok(Obu {
                    obu_type: FRAME_HEADER,
                    extension: Some(Extension {
                        temporal_id: 2,
                        spatial_id: 1
                    }),
                    payload: &[0xaa, 0xbb],
                }),
                Ok(Obu {
                    obu_type: PADDING,
                    extension: None,
                    payload: &[0x01, 0x02, 0x03],
                }),
            ]
        );
    }

    #[test]
    fn an_obu_past_its_sample_or_with_the_forbidden_bit_ends_the_obus() {
        let past = [0x12, 0x05, 0x00];
        assert_eq!(
            obus(&past).collect::<Vec<_>>(),
            vec![Err(Error::ObuPastEnd)]
        );
        let forbidden = [0x92, 0x00, 0x12, 0x00];
        assert_eq!(
            obus(&forbidden).collect::<Vec<_>>(),
            vec![Err(Error::OutOfRange("obu_forbidden_bit"))]
        );
        assert_eq!(
            obus(&[0x16]).collect::<Vec<_>>(),
            vec![Err(Error::Truncated)],
            "an extension flag with no extension byte"
        );
    }

    #[test]
    fn an_operating_point_drops_the_layers_it_does_not_hold() {
        let obu = |temporal_id, spatial_id| Obu {
            obu_type: FRAME,
            extension: Some(Extension {
                temporal_id,
                spatial_id,
            }),
            payload: &[],
        };
        // Temporal layers 0 and 1 of spatial layer 0.
        let idc = 0x0103;
        assert!(!obu(0, 0).dropped_by(idc));
        assert!(!obu(1, 0).dropped_by(idc));
        assert!(obu(2, 0).dropped_by(idc));
        assert!(obu(0, 1).dropped_by(idc));
        assert!(!obu(7, 3).dropped_by(0), "idc 0 holds every layer");
    }
}
