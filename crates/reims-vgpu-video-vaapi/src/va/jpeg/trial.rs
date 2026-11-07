//! The trial decodes the engine makes at start (`super::VaapiJpegDecoder::new`):
//! a flat baseline JPEG of each sampling the driver lists, every sample of
//! a component one value, and the check of what the decode wrote.
//!
//! A flat block's only coefficient is its DC, which a decoder's IDCT turns
//! back into the value exactly (to within its rounding), so the output of
//! a decode that is right is the values and nothing else. The values are
//! none of 0, 128 and 255, which a surface the decode did not write, or
//! wrote without the stream's samples, most likely holds.

use reims_vgpu_video_codec::jpeg::{
    DEFAULT_AC_CHROMINANCE, DEFAULT_AC_LUMINANCE, DEFAULT_DC_CHROMINANCE, DEFAULT_DC_LUMINANCE,
    HuffmanTable,
};
use reims_vgpu_video_core::jpeg::{JpegOutputFormat, JpegSubsampling};

/// Each component's value: Y, Cb and Cr.
pub(super) const VALUES: [u8; 3] = [90, 110, 150];

/// The trial image's width and height: two 4:2:0 MCUs each way.
pub(super) const SIZE: u16 = 32;

/// The layout a trial of `subsampling` is decoded into at scale 1: its
/// own, which the decode writes as it is (`super::Route::Copy`), and
/// `420f` for grey, whose chroma core writes as 0x80.
pub(super) fn layout(subsampling: JpegSubsampling) -> Option<JpegOutputFormat> {
    match subsampling {
        JpegSubsampling::Yuv420 | JpegSubsampling::Grey => Some(JpegOutputFormat::Biplanar420),
        JpegSubsampling::Yuv422 => Some(JpegOutputFormat::Biplanar422),
        JpegSubsampling::Yuv444 => Some(JpegOutputFormat::Biplanar444),
        JpegSubsampling::Yuv411 => None,
    }
}

/// A baseline JPEG of [`SIZE`] by [`SIZE`] in `subsampling`, every sample
/// of each component its value in [`VALUES`]: one quantization table of 8s,
/// Annex K's Huffman tables in the stream, and each block a DC difference
/// and an end of block. `None` for a sampling with no trial.
pub(super) fn flat(subsampling: JpegSubsampling) -> Option<Vec<u8>> {
    // Each component: its id, its sampling factors, and its tables.
    let components: &[(u8, u8, u8)] = match subsampling {
        JpegSubsampling::Yuv420 => &[(1, 0x22, 0), (2, 0x11, 1), (3, 0x11, 1)],
        JpegSubsampling::Yuv422 => &[(1, 0x21, 0), (2, 0x11, 1), (3, 0x11, 1)],
        JpegSubsampling::Yuv444 => &[(1, 0x11, 0), (2, 0x11, 1), (3, 0x11, 1)],
        JpegSubsampling::Grey => &[(1, 0x11, 0)],
        JpegSubsampling::Yuv411 => return None,
    };
    let mut out = vec![0xff, 0xd8];
    let mut dqt = vec![0u8];
    dqt.extend([8u8; 64]);
    segment(&mut out, 0xdb, &dqt);
    let mut sof = vec![8];
    sof.extend(SIZE.to_be_bytes());
    sof.extend(SIZE.to_be_bytes());
    sof.push(u8::try_from(components.len()).unwrap_or(0));
    for &(id, factors, _) in components {
        sof.extend([id, factors, 0]);
    }
    segment(&mut out, 0xc0, &sof);
    for (class, id, table) in [
        (0, 0, &DEFAULT_DC_LUMINANCE),
        (1, 0, &DEFAULT_AC_LUMINANCE),
        (0, 1, &DEFAULT_DC_CHROMINANCE),
        (1, 1, &DEFAULT_AC_CHROMINANCE),
    ] {
        let mut dht = vec![(class << 4) | id];
        dht.extend(table.counts);
        dht.extend(table.values);
        segment(&mut out, 0xc4, &dht);
    }
    let mut sos = vec![u8::try_from(components.len()).unwrap_or(0)];
    for &(id, _, tables) in components {
        sos.extend([id, (tables << 4) | tables]);
    }
    sos.extend([0, 63, 0]);
    segment(&mut out, 0xda, &sos);
    // A scan of one component is of its blocks alone, 8x8 each; of
    // several, of MCUs of each component's factors' blocks.
    let (mcu_width, mcu_height) = if components.len() == 1 {
        (8, 8)
    } else {
        let factors = components[0].1;
        (8 * u16::from(factors >> 4), 8 * u16::from(factors & 0xf))
    };
    let mcus = usize::from(SIZE.div_ceil(mcu_width)) * usize::from(SIZE.div_ceil(mcu_height));
    let codes = |table: &HuffmanTable<'_>| canonical(table);
    let dc = [codes(&DEFAULT_DC_LUMINANCE), codes(&DEFAULT_DC_CHROMINANCE)];
    let ac = [codes(&DEFAULT_AC_LUMINANCE), codes(&DEFAULT_AC_CHROMINANCE)];
    let mut bits = Bits::default();
    for mcu in 0..mcus {
        for (n, &(_, factors, tables)) in components.iter().enumerate() {
            let blocks = if components.len() == 1 {
                1
            } else {
                u32::from(factors >> 4) * u32::from(factors & 0xf)
            };
            let tables = usize::from(tables);
            for block in 0..blocks {
                // The DC is the value less 128, times 8, over the table's 8;
                // each block after a component's first differs by nothing.
                let difference = if mcu == 0 && block == 0 {
                    i32::from(VALUES[n]) - 128
                } else {
                    0
                };
                let size = 32 - difference.unsigned_abs().leading_zeros();
                let (code, length) = dc[tables][size as usize];
                bits.put(code, length);
                let magnitude = if difference < 0 {
                    difference + (1 << size) - 1
                } else {
                    difference
                };
                bits.put(u32::try_from(magnitude).unwrap_or(0), size);
                // End of block.
                let (code, length) = ac[tables][0];
                bits.put(code, length);
            }
        }
    }
    out.extend(bits.finish());
    out.extend([0xff, 0xd9]);
    Some(out)
}

/// The number of samples of `planes`, decoded from [`flat`] into
/// [`layout`]`(subsampling)`, `SIZE` by `SIZE`, that are not the values
/// (luma, then the interleaved chroma), and how many it checked.
pub(super) fn wrong(subsampling: JpegSubsampling, luma: &[u8], chroma: &[u8]) -> (usize, usize) {
    let side = usize::from(SIZE);
    let (across, down) = match subsampling {
        JpegSubsampling::Yuv422 => (2, 1),
        JpegSubsampling::Yuv444 => (1, 1),
        _ => (2, 2),
    };
    let [y, cb, cr] = if subsampling == JpegSubsampling::Grey {
        [VALUES[0], 0x80, 0x80]
    } else {
        VALUES
    };
    let off = |sample: u8, value: u8| sample.abs_diff(value) > 1;
    let mut wrong = luma[..side * side].iter().filter(|&&s| off(s, y)).count();
    let pairs = (side / across) * (side / down);
    for pair in chroma[..pairs * 2].chunks_exact(2) {
        wrong += usize::from(off(pair[0], cb)) + usize::from(off(pair[1], cr));
    }
    (wrong, side * side + pairs * 2)
}

/// Each code of `table`, by its value: (code, length), as T.81's Annex C
/// assigns them.
fn canonical(table: &HuffmanTable<'_>) -> Vec<(u32, u32)> {
    let mut codes = vec![(0, 0); 256];
    let mut code = 0u32;
    let mut values = table.values.iter();
    for (length, &count) in (1u32..).zip(&table.counts) {
        for _ in 0..count {
            if let Some(&value) = values.next() {
                codes[usize::from(value)] = (code, length);
            }
            code += 1;
        }
        code <<= 1;
    }
    codes
}

fn segment(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    out.extend([0xff, marker]);
    out.extend(u16::try_from(payload.len() + 2).unwrap_or(0).to_be_bytes());
    out.extend(payload);
}

/// Entropy-coded bits, most significant first, a 0x00 stuffed after each
/// 0xff byte, the last byte padded with 1s.
#[derive(Default)]
struct Bits {
    bytes: Vec<u8>,
    current: u32,
    filled: u32,
}

impl Bits {
    fn put(&mut self, value: u32, length: u32) {
        for bit in (0..length).rev() {
            self.current = (self.current << 1) | ((value >> bit) & 1);
            self.filled += 1;
            if self.filled == 8 {
                self.push();
            }
        }
    }

    fn push(&mut self) {
        let byte = u8::try_from(self.current & 0xff).unwrap_or(0);
        self.bytes.push(byte);
        if byte == 0xff {
            self.bytes.push(0);
        }
        self.current = 0;
        self.filled = 0;
    }

    fn finish(mut self) -> Vec<u8> {
        if self.filled > 0 {
            let pad = 8 - self.filled;
            self.put((1 << pad) - 1, pad);
        }
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_video_codec::jpeg::{parse, walk};

    /// Each trial parses as the sampling's baseline frame, of Annex K's
    /// tables, and its scan walks to its last MCU.
    #[test]
    fn each_trial_is_a_whole_baseline_jpeg_of_its_sampling() {
        for subsampling in [
            JpegSubsampling::Yuv420,
            JpegSubsampling::Yuv422,
            JpegSubsampling::Yuv444,
            JpegSubsampling::Grey,
        ] {
            let data = flat(subsampling).expect("a trial");
            let image = parse(&data).expect("parses");
            assert_eq!(
                (image.frame.width, image.frame.height),
                (SIZE, SIZE),
                "{subsampling:?}"
            );
            assert_eq!(
                image.frame.subsampling().and_then(super::super::subsampling),
                Some(subsampling)
            );
            walk(&image).expect("the scan walks");
            assert!(layout(subsampling).is_some());
        }
        assert!(flat(JpegSubsampling::Yuv411).is_none());
    }

    /// The codes Annex K gives the values the trials use: luma's DC sizes 5
    /// and 6, chroma's, and each end of block.
    #[test]
    fn the_codes_are_annex_ks() {
        let dc = canonical(&DEFAULT_DC_LUMINANCE);
        assert_eq!(dc[0], (0b00, 2));
        assert_eq!(dc[6], (0b1110, 4));
        let dc = canonical(&DEFAULT_DC_CHROMINANCE);
        assert_eq!(dc[5], (0b11110, 5));
        assert_eq!(canonical(&DEFAULT_AC_LUMINANCE)[0], (0b1010, 4));
        assert_eq!(canonical(&DEFAULT_AC_CHROMINANCE)[0], (0b00, 2));
    }

    /// A decode of the values passes; one sample off by more than 1 is
    /// counted, in either plane, and grey's chroma is 0x80.
    #[test]
    fn a_sample_off_the_values_is_counted() {
        let side = usize::from(SIZE);
        let luma = vec![VALUES[0]; side * side];
        let chroma: Vec<u8> = [VALUES[1], VALUES[2]].repeat(side * side / 4);
        assert_eq!(wrong(JpegSubsampling::Yuv420, &luma, &chroma), (0, 1536));
        let mut off = luma.clone();
        off[100] = VALUES[0] + 2;
        assert_eq!(wrong(JpegSubsampling::Yuv420, &off, &chroma).0, 1);
        let neutral = vec![0x80; side * side / 2];
        assert_eq!(wrong(JpegSubsampling::Grey, &luma, &neutral).0, 0);
        assert_eq!(wrong(JpegSubsampling::Grey, &luma, &chroma).0, side * side / 2);
        let zeros = vec![0; side * side];
        assert_eq!(wrong(JpegSubsampling::Grey, &zeros, &neutral).0, side * side);
    }
}
