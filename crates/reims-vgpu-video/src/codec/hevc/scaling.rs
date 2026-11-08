//! Scaling lists (H.265 7.3.4 and 7.4.5): `scaling_list_data()` in an SPS
//! or PPS, the default lists of Tables 7-5 and 7-6, and prediction from
//! another list.
//!
//! Lists are kept as the standard indexes them, `ScalingList[sizeId]
//! [matrixId][i]` with `i` in coding order, the up-right diagonal scan
//! (6.5.3). An accelerator lays them out as its API wants: VA-API's in
//! raster order (the VA-API crate's `src/va/hevc.rs`, `iq_matrix`).

use super::Error;
use super::bits::BitReader;

/// Table 7-6's default 8x8 list for intra prediction (`matrixId` 0 to 2),
/// in coding order; the 16x16 and 32x32 defaults are the same 64 values.
const DEFAULT_INTRA: [u8; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 16, 17, 16, 17, 18, //
    17, 18, 18, 17, 18, 21, 19, 20, 21, 20, 19, 21, 24, 22, 22, 24, //
    24, 22, 22, 24, 25, 25, 27, 30, 27, 25, 25, 29, 31, 35, 35, 31, //
    29, 36, 41, 44, 41, 36, 47, 54, 54, 47, 65, 70, 65, 88, 88, 115,
];

/// Table 7-6's default list for inter prediction (`matrixId` 3 to 5).
const DEFAULT_INTER: [u8; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, //
    18, 18, 18, 18, 18, 20, 20, 20, 20, 20, 20, 20, 24, 24, 24, 24, //
    24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 28, 28, 28, 28, 28, //
    28, 33, 33, 33, 33, 33, 41, 41, 41, 41, 54, 54, 54, 71, 71, 91,
];

/// The DC value of a 16x16 or 32x32 default list (7.4.5).
const DEFAULT_DC: u8 = 16;

/// Every scaling list of a picture, with the DC values of the two largest
/// sizes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalingLists {
    /// `sizeId` 0, 4x4: six lists of 16.
    pub list_4x4: [[u8; 16]; 6],
    /// `sizeId` 1, 8x8: six lists of 64.
    pub list_8x8: [[u8; 64]; 6],
    /// `sizeId` 2, 16x16: six lists of 64, upsampled by the decoder.
    pub list_16x16: [[u8; 64]; 6],
    /// `sizeId` 3, 32x32: coded for `matrixId` 0 and 3 alone in 4:2:0;
    /// the other four entries are the defaults and unused.
    pub list_32x32: [[u8; 64]; 6],
    /// `scaling_list_dc_coef_minus8[0][matrixId] + 8`.
    pub dc_16x16: [u8; 6],
    /// `scaling_list_dc_coef_minus8[1][matrixId] + 8`, used for
    /// `matrixId` 0 and 3.
    pub dc_32x32: [u8; 6],
}

impl Default for ScalingLists {
    /// The default lists (Tables 7-5 and 7-6), which apply when scaling
    /// lists are enabled and neither parameter set codes them.
    fn default() -> Self {
        let large = |matrix: usize| {
            if matrix < 3 {
                DEFAULT_INTRA
            } else {
                DEFAULT_INTER
            }
        };
        Self {
            list_4x4: [[16; 16]; 6],
            list_8x8: core::array::from_fn(large),
            list_16x16: core::array::from_fn(large),
            list_32x32: core::array::from_fn(large),
            dc_16x16: [DEFAULT_DC; 6],
            dc_32x32: [DEFAULT_DC; 6],
        }
    }
}

impl ScalingLists {
    /// `scaling_list_data()` (7.3.4), with each list its syntax leaves out
    /// taken from the defaults or from the list it predicts from.
    pub(crate) fn parse(r: &mut BitReader<'_>) -> Result<Self, Error> {
        let mut lists = Self::default();
        parse_size(r, &mut lists.list_4x4, None, 1)?;
        parse_size(r, &mut lists.list_8x8, None, 1)?;
        parse_size(r, &mut lists.list_16x16, Some(&mut lists.dc_16x16), 1)?;
        parse_size(r, &mut lists.list_32x32, Some(&mut lists.dc_32x32), 3)?;
        Ok(lists)
    }
}

/// One `sizeId`'s lists of `scaling_list_data()`, the defaults until then,
/// with their DC values for 16x16 and 32x32 (`dcs`). `matrixId` goes up in
/// steps of `step`: 3 for 32x32, whose chroma lists 4:2:0 does not code.
fn parse_size<const N: usize>(
    r: &mut BitReader<'_>,
    lists: &mut [[u8; N]; 6],
    dcs: Option<&mut [u8; 6]>,
    step: usize,
) -> Result<(), Error> {
    let has_dc = dcs.is_some();
    let mut no_dcs = [DEFAULT_DC; 6];
    let dcs = dcs.unwrap_or(&mut no_dcs);
    // The lists read so far, with their DC values, for prediction.
    let mut earlier: Vec<([u8; N], u8)> = Vec::with_capacity(6);
    for (position, (list, dc)) in (0u8..).zip(lists.iter_mut().zip(dcs.iter_mut()).step_by(step)) {
        if r.bit()? {
            let mut next: u8 = 8;
            if has_dc {
                let value = r
                    .se_range(-7, 247, "scaling_list_dc_coef_minus8")?
                    .checked_add(8)
                    .and_then(|value| u8::try_from(value).ok())
                    .ok_or(Error::OutOfRange("scaling_list_dc_coef_minus8"))?;
                *dc = value;
                next = value;
            }
            for coefficient in list.iter_mut() {
                let delta = r.se_i8(-128, 127, "scaling_list_delta_coef")?;
                // 7.3.4: nextScale = (nextScale + scaling_list_delta_coef
                // + 256) % 256, which is u8's wrapping addition.
                next = next.wrapping_add_signed(delta);
                // 7.4.5: every entry is over 0, a divisor of no scaling.
                *coefficient = Some(next)
                    .filter(|&value| value != 0)
                    .ok_or(Error::OutOfRange("ScalingList"))?;
            }
        } else {
            // scaling_list_pred_matrix_id_delta: 0 is the default list,
            // which `list` holds, else an earlier list of the same size,
            // whose DC goes with it (7.4.5).
            let delta = r.ue_u8(position, "scaling_list_pred_matrix_id_delta")?;
            if delta != 0 {
                (*list, *dc) = *position
                    .checked_sub(delta)
                    .and_then(|reference| earlier.get(usize::from(reference)))
                    .ok_or(Error::OutOfRange("scaling_list_pred_matrix_id_delta"))?;
            }
        }
        earlier.push((*list, *dc));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::hevc::tests::BitWriter;

    #[test]
    fn predicted_lists_copy_their_reference_and_its_dc() -> Result<(), Error> {
        let mut w = BitWriter::new();
        for size_id in 0..4 {
            let step = if size_id == 3 { 3 } else { 1 };
            for matrix_id in (0..6).step_by(step) {
                match (size_id, matrix_id) {
                    // 4x4 list 0 coded: deltas +1 from 8, so 9, 10, ...
                    (0, 0) | (2, 1) => {
                        w.bit(true);
                        if size_id == 2 {
                            w.se(20); // DC 28
                        }
                        for _ in 0..if size_id == 0 { 16 } else { 64 } {
                            w.se(1);
                        }
                    }
                    // 4x4 list 2 from list 0; 16x16 list 3 from list 1.
                    (0, 2) | (2, 3) => {
                        w.bit(false);
                        w.ue(2);
                    }
                    _ => {
                        w.bit(false);
                        w.ue(0);
                    }
                }
            }
        }
        let bytes = w.finish();
        let lists = ScalingLists::parse(&mut BitReader::new(&bytes))?;
        let ramp: [u8; 16] = core::array::from_fn(|i| 9 + i as u8);
        assert_eq!(lists.list_4x4[0], ramp);
        assert_eq!(lists.list_4x4[1], [16; 16]);
        assert_eq!(lists.list_4x4[2], ramp);
        assert_eq!(lists.list_8x8[4], DEFAULT_INTER);
        assert_eq!(lists.list_16x16[1][0], 29);
        assert_eq!(lists.list_16x16[1][63], 92);
        assert_eq!(lists.list_16x16[3], lists.list_16x16[1]);
        assert_eq!(lists.dc_16x16, [16, 28, 16, 28, 16, 16]);
        assert_eq!(lists.list_32x32[3], DEFAULT_INTER);
        Ok(())
    }

    #[test]
    fn a_prediction_from_a_later_list_is_refused() {
        let mut w = BitWriter::new();
        w.bit(false);
        w.ue(1); // matrixId 0 cannot predict from matrixId -1
        let bytes = w.finish();
        assert_eq!(
            ScalingLists::parse(&mut BitReader::new(&bytes)),
            Err(Error::OutOfRange("scaling_list_pred_matrix_id_delta"))
        );
    }
}
